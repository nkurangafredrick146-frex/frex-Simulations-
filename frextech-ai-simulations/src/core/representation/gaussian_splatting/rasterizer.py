"""
Rasterizer for Gaussian Splatting.
Implements differentiable rasterization of 3D Gaussians to 2D images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import math
from scipy.spatial.transform import Rotation as R
import time


@dataclass
class Camera:
    """Camera parameters."""
    position: torch.Tensor  # [3] world position
    rotation: torch.Tensor  # [4] quaternion (w, x, y, z)
    focal_x: float  # focal length in x
    focal_y: float  # focal length in y
    principal_x: float  # principal point x
    principal_y: float  # principal point y
    width: int  # image width
    height: int  # image height
    near: float = 0.01  # near plane
    far: float = 100.0  # far plane
    
    def to_device(self, device: str):
        """Move camera parameters to device."""
        self.position = self.position.to(device)
        self.rotation = self.rotation.to(device)
        return self
    
    def get_view_matrix(self) -> torch.Tensor:
        """Get view matrix from camera pose."""
        # Convert quaternion to rotation matrix
        q = self.rotation
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        # Rotation matrix
        R_mat = torch.tensor([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ], device=self.position.device)
        
        # Translation
        t = -R_mat @ self.position
        
        # View matrix [4, 4]
        view = torch.eye(4, device=self.position.device)
        view[:3, :3] = R_mat
        view[:3, 3] = t
        
        return view
    
    def get_projection_matrix(self) -> torch.Tensor:
        """Get projection matrix."""
        fx, fy = self.focal_x, self.focal_y
        cx, cy = self.principal_x, self.principal_y
        w, h = self.width, self.height
        near, far = self.near, self.far
        
        # Perspective projection matrix
        proj = torch.zeros((4, 4), device=self.position.device)
        
        proj[0, 0] = 2 * fx / w
        proj[1, 1] = 2 * fy / h
        proj[0, 2] = 2 * cx / w - 1
        proj[1, 2] = 2 * cy / h - 1
        proj[2, 2] = (far + near) / (far - near)
        proj[2, 3] = -2 * far * near / (far - near)
        proj[3, 2] = 1.0
        
        return proj
    
    def get_view_direction(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """
        Get view direction for each pixel.
        
        Args:
            pixel_coords: Pixel coordinates [H, W, 2] or [N, 2]
            
        Returns:
            View directions in world coordinates
        """
        if pixel_coords.dim() == 3:
            H, W, _ = pixel_coords.shape
            pixel_coords = pixel_coords.view(-1, 2)
            reshape = True
        else:
            reshape = False
        
        # Convert to normalized device coordinates
        x = (pixel_coords[:, 0] - self.principal_x) / self.focal_x
        y = (pixel_coords[:, 1] - self.principal_y) / self.focal_y
        z = torch.ones_like(x)
        
        # Directions in camera space
        dirs_cam = torch.stack([x, y, z], dim=1)
        dirs_cam = F.normalize(dirs_cam, dim=1)
        
        # Convert to world space
        q = self.rotation
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        # Inverse rotation (conjugate)
        R_inv = torch.tensor([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy + 2*qz*qw, 2*qx*qz - 2*qy*qw],
            [2*qx*qy - 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz + 2*qx*qw],
            [2*qx*qz + 2*qy*qw, 2*qy*qz - 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ], device=self.position.device)
        
        dirs_world = dirs_cam @ R_inv.T
        
        if reshape:
            dirs_world = dirs_world.view(H, W, 3)
        
        return dirs_world


class GaussianRasterizer(nn.Module):
    """
    Differentiable rasterizer for 3D Gaussian splatting.
    
    Implements:
    - Projection of 3D Gaussians to 2D
    - Alpha-blended rendering
    - Differentiable depth and normal computation
    - Tile-based rasterization for efficiency
    """
    
    def __init__(
        self,
        tile_size: int = 16,
        max_gaussians_per_tile: int = 1024,
        background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        debug: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.tile_size = tile_size
        self.max_gaussians_per_tile = max_gaussians_per_tile
        self.background_color = torch.tensor(background_color, device=device)
        self.debug = debug
        self.device = device
        
        # Precompute tile grid
        self._init_tile_grid()
        
        # Statistics
        self.stats = {
            'render_time': [],
            'gaussians_per_frame': [],
            'tiles_per_frame': []
        }
    
    def _init_tile_grid(self):
        """Initialize tile grid for given image size."""
        # This will be initialized when image size is known
        self.tile_grid = None
        self.num_tiles_x = 0
        self.num_tiles_y = 0
    
    def _setup_tile_grid(self, height: int, width: int):
        """Setup tile grid for given image dimensions."""
        self.num_tiles_x = (width + self.tile_size - 1) // self.tile_size
        self.num_tiles_y = (height + self.tile_size - 1) // self.tile_size
        
        # Create tile coordinates
        tile_coords = []
        for ty in range(self.num_tiles_y):
            for tx in range(self.num_tiles_x):
                x_min = tx * self.tile_size
                x_max = min((tx + 1) * self.tile_size, width)
                y_min = ty * self.tile_size
                y_max = min((ty + 1) * self.tile_size, height)
                tile_coords.append((tx, ty, x_min, x_max, y_min, y_max))
        
        self.tile_grid = tile_coords
    
    def project_gaussians(
        self,
        positions: torch.Tensor,
        covariances: torch.Tensor,
        camera: Camera
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project 3D Gaussians to 2D image plane.
        
        Args:
            positions: Gaussian positions [N, 3]
            covariances: Gaussian covariances [N, 3, 3]
            camera: Camera parameters
            
        Returns:
            Tuple of (means_2d, covariances_2d, depths, valid_mask)
        """
        N = positions.shape[0]
        
        # Transform positions to camera space
        view_matrix = camera.get_view_matrix()
        proj_matrix = camera.get_projection_matrix()
        
        # Homogeneous coordinates
        positions_h = torch.cat([positions, torch.ones(N, 1, device=self.device)], dim=1)
        positions_cam = (view_matrix @ positions_h.T).T[:, :3]  # [N, 3]
        
        # Compute depths (negative z in camera space for OpenGL convention)
        depths = -positions_cam[:, 2]
        
        # Check visibility (frustum culling)
        valid_mask = (
            (depths > camera.near) & 
            (depths < camera.far) &
            (torch.abs(positions_cam[:, 0]) < depths * camera.width / (2 * camera.focal_x)) &
            (torch.abs(positions_cam[:, 1]) < depths * camera.height / (2 * camera.focal_y))
        )
        
        if not torch.any(valid_mask):
            return None, None, None, valid_mask
        
        # Project to 2D
        positions_cam_valid = positions_cam[valid_mask]
        
        # Perspective projection
        x_proj = camera.focal_x * positions_cam_valid[:, 0] / positions_cam_valid[:, 2] + camera.principal_x
        y_proj = camera.focal_y * positions_cam_valid[:, 1] / positions_cam_valid[:, 2] + camera.principal_y
        
        means_2d = torch.stack([x_proj, y_proj], dim=1)  # [M, 2]
        
        # Project covariance to 2D
        covariances_valid = covariances[valid_mask]
        
        # Get Jacobian of projection
        z = positions_cam_valid[:, 2]
        J = torch.zeros((len(z), 2, 3), device=self.device)
        J[:, 0, 0] = camera.focal_x / z
        J[:, 1, 1] = camera.focal_y / z
        J[:, 0, 2] = -camera.focal_x * positions_cam_valid[:, 0] / (z * z)
        J[:, 1, 2] = -camera.focal_y * positions_cam_valid[:, 1] / (z * z)
        
        # Project covariance: Σ' = J Σ J^T
        covariances_2d = J @ covariances_valid @ J.transpose(1, 2)
        
        # Add small epsilon for numerical stability
        eps = torch.eye(2, device=self.device).unsqueeze(0) * 1e-6
        covariances_2d = covariances_2d + eps
        
        # Expand to full size with invalid Gaussians having zero covariance
        means_2d_full = torch.zeros(N, 2, device=self.device)
        covariances_2d_full = torch.zeros(N, 2, 2, device=self.device)
        depths_full = torch.zeros(N, device=self.device)
        
        means_2d_full[valid_mask] = means_2d
        covariances_2d_full[valid_mask] = covariances_2d
        depths_full[valid_mask] = depths[valid_mask]
        
        return means_2d_full, covariances_2d_full, depths_full, valid_mask
    
    def sort_gaussians_by_depth(
        self,
        means_2d: torch.Tensor,
        depths: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Sort Gaussians by depth (back to front for blending).
        
        Args:
            means_2d: 2D means [N, 2]
            depths: Depths [N]
            valid_mask: Valid mask [N]
            
        Returns:
            Sorted indices
        """
        # Only sort valid Gaussians
        valid_indices = torch.where(valid_mask)[0]
        valid_depths = depths[valid_mask]
        
        # Sort by depth (descending for back-to-front)
        sorted_indices_valid = torch.argsort(valid_depths, descending=True)
        sorted_indices = valid_indices[sorted_indices_valid]
        
        return sorted_indices
    
    def tile_binning(
        self,
        means_2d: torch.Tensor,
        valid_mask: torch.Tensor,
        height: int,
        width: int
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Bin Gaussians into tiles.
        
        Args:
            means_2d: 2D means [N, 2]
            valid_mask: Valid mask [N]
            height: Image height
            width: Image width
            
        Returns:
            Dictionary mapping tile coordinates to Gaussian indices
        """
        # Initialize tile grid if needed
        if self.tile_grid is None:
            self._setup_tile_grid(height, width)
        
        # Get valid Gaussians
        valid_indices = torch.where(valid_mask)[0]
        means_2d_valid = means_2d[valid_mask]
        
        # Determine tile for each Gaussian
        tile_x = torch.clamp((means_2d_valid[:, 0] / self.tile_size).long(), 0, self.num_tiles_x - 1)
        tile_y = torch.clamp((means_2d_valid[:, 1] / self.tile_size).long(), 0, self.num_tiles_y - 1)
        
        # Bin Gaussians into tiles
        tile_bins = {}
        for i, idx in enumerate(valid_indices):
            tx, ty = tile_x[i].item(), tile_y[i].item()
            key = (tx, ty)
            if key not in tile_bins:
                tile_bins[key] = []
            tile_bins[key].append(idx)
        
        # Convert lists to tensors and limit size
        for key in list(tile_bins.keys()):
            indices = torch.tensor(tile_bins[key], device=self.device, dtype=torch.long)
            if len(indices) > self.max_gaussians_per_tile:
                # Keep only the closest Gaussians (based on screen space extent)
                # This is a simplification - real implementation would use more sophisticated criteria
                indices = indices[:self.max_gaussians_per_tile]
            tile_bins[key] = indices
        
        return tile_bins
    
    def compute_gaussian_weights(
        self,
        pixel_coords: torch.Tensor,
        means_2d: torch.Tensor,
        covariances_2d: torch.Tensor,
        opacities: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Gaussian weights for each pixel.
        
        Args:
            pixel_coords: Pixel coordinates [H, W, 2]
            means_2d: Gaussian means [N, 2]
            covariances_2d: Gaussian covariances [N, 2, 2]
            opacities: Gaussian opacities [N, 1]
            
        Returns:
            Weights [H, W, N]
        """
        H, W, _ = pixel_coords.shape
        N = means_2d.shape[0]
        
        # Reshape for broadcasting
        pixels = pixel_coords.view(-1, 2)  # [H*W, 2]
        pixels = pixels.unsqueeze(1)  # [H*W, 1, 2]
        means = means_2d.unsqueeze(0)  # [1, N, 2]
        
        # Compute Mahalanobis distance
        diff = pixels - means  # [H*W, N, 2]
        
        # Invert covariance matrices
        # Using Cholesky decomposition for stability
        try:
            L = torch.linalg.cholesky(covariances_2d)  # [N, 2, 2]
            Linv = torch.linalg.inv(L)  # [N, 2, 2]
            
            # Solve linear system: L^T y = diff^T
            diff_T = diff.transpose(1, 2)  # [H*W, 2, N]
            y = torch.linalg.solve_triangular(
                L.unsqueeze(0).transpose(2, 3),  # [1, N, 2, 2]
                diff_T.unsqueeze(1),  # [H*W, 1, 2, N]
                upper=True
            )
            
            # Mahalanobis distance squared
            mahalanobis_sq = torch.sum(y.squeeze() ** 2, dim=1)  # [H*W, N]
            
        except RuntimeError:
            # Fallback to simpler computation if Cholesky fails
            cov_inv = torch.linalg.inv(covariances_2d)  # [N, 2, 2]
            diff_T = diff.transpose(1, 2)  # [H*W, 2, N]
            mahalanobis_sq = torch.sum(
                diff_T @ cov_inv.unsqueeze(0) * diff_T,
                dim=1
            )  # [H*W, N]
        
        # Compute Gaussian weights
        # w = α * exp(-0.5 * (x-μ)^T Σ^{-1} (x-μ))
        weights = opacities.view(1, N) * torch.exp(-0.5 * mahalanobis_sq)  # [H*W, N]
        
        # Reshape back to image
        weights = weights.view(H, W, N)
        
        return weights
    
    def alpha_blend(
        self,
        colors: torch.Tensor,
        weights: torch.Tensor,
        depths: torch.Tensor,
        sorted_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Alpha blend Gaussians using back-to-front composition.
        
        Args:
            colors: Gaussian colors [N, 3]
            weights: Gaussian weights [H, W, N]
            depths: Gaussian depths [N]
            sorted_indices: Gaussians sorted back-to-front
            
        Returns:
            Tuple of (rendered_image, accumulated_alpha, depth_map)
        """
        H, W, N = weights.shape
        
        # Sort weights and colors according to depth
        weights_sorted = weights[:, :, sorted_indices]  # [H, W, N]
        colors_sorted = colors[sorted_indices]  # [N, 3]
        depths_sorted = depths[sorted_indices]  # [N]
        
        # Compute transmittance
        alpha = torch.clamp(weights_sorted, 0, 1)
        transmittance = torch.cumprod(1 - alpha + 1e-8, dim=2)
        
        # Shift transmittance for correct blending
        transmittance_shifted = torch.roll(transmittance, shifts=1, dims=2)
        transmittance_shifted[:, :, 0] = 1.0
        
        # Compute weights for blending
        blend_weights = alpha * transmittance_shifted
        
        # Blend colors
        color_contributions = blend_weights.unsqueeze(-1) * colors_sorted.view(1, 1, N, 3)
        rendered = torch.sum(color_contributions, dim=2)  # [H, W, 3]
        
        # Add background
        accumulated_alpha = 1 - torch.prod(1 - alpha + 1e-8, dim=2)
        background_weight = 1 - accumulated_alpha
        
        rendered = rendered + background_weight.unsqueeze(-1) * self.background_color
        
        # Compute depth map (weighted average)
        depth_contributions = blend_weights * depths_sorted.view(1, 1, N)
        depth_map = torch.sum(depth_contributions, dim=2)  # [H, W]
        
        # Normalize by accumulated alpha (avoid division by zero)
        depth_map = depth_map / (accumulated_alpha + 1e-8)
        
        return rendered, accumulated_alpha, depth_map
    
    def compute_normals(
        self,
        positions: torch.Tensor,
        covariances: torch.Tensor,
        rendered_image: torch.Tensor,
        depth_map: torch.Tensor,
        camera: Camera
    ) -> torch.Tensor:
        """
        Compute surface normals from Gaussian representation.
        
        Args:
            positions: Gaussian positions [N, 3]
            covariances: Gaussian covariances [N, 3, 3]
            rendered_image: Rendered image [H, W, 3]
            depth_map: Depth map [H, W]
            camera: Camera parameters
            
        Returns:
            Normal map [H, W, 3]
        """
        H, W, _ = rendered_image.shape
        
        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float()
        
        # Compute gradients of depth map
        depth_padded = F.pad(depth_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze()
        
        # Sobel filters for gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device).float()
        
        grad_x = F.conv2d(
            depth_padded.unsqueeze(0).unsqueeze(0),
            sobel_x.view(1, 1, 3, 3),
            padding=0
        ).squeeze()
        
        grad_y = F.conv2d(
            depth_padded.unsqueeze(0).unsqueeze(0),
            sobel_y.view(1, 1, 3, 3),
            padding=0
        ).squeeze()
        
        # Compute normal from gradient (in camera space)
        # For perspective projection: n = normalize(-∇z, 1)
        fx, fy = camera.focal_x, camera.focal_y
        
        # Convert gradients to 3D normals
        normals = torch.zeros(H, W, 3, device=self.device)
        normals[:, :, 0] = -grad_x / fx
        normals[:, :, 1] = -grad_y / fy
        normals[:, :, 2] = 1.0
        
        # Normalize
        normals = F.normalize(normals, dim=2)
        
        # Transform to world space
        q = camera.rotation
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        R_mat = torch.tensor([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ], device=self.device)
        
        normals_world = normals @ R_mat.T
        
        return normals_world
    
    def render(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        camera: Camera,
        render_mode: str = 'color'
    ) -> Dict[str, torch.Tensor]:
        """
        Main rendering function.
        
        Args:
            gaussian_params: Dictionary of Gaussian parameters
            camera: Camera parameters
            render_mode: 'color', 'depth', 'normal', or 'all'
            
        Returns:
            Dictionary with rendered outputs
        """
        start_time = time.time()
        
        # Extract parameters
        positions = gaussian_params['positions']
        rotations = gaussian_params['rotations']
        scales = gaussian_params['scales']
        opacities = gaussian_params['opacities']
        sh_coeffs = gaussian_params['sh_coeffs']
        
        # Move camera to device
        camera = camera.to_device(self.device)
        
        # Compute view direction for all Gaussians
        view_dirs = camera.position - positions
        view_dirs = F.normalize(view_dirs, dim=1)
        
        # Compute colors from spherical harmonics
        colors = self._compute_sh_color(sh_coeffs, view_dirs)
        
        # Compute 3D covariance matrices
        covariances_3d = self._compute_covariance_3d(rotations, scales)
        
        # Project to 2D
        means_2d, covariances_2d, depths, valid_mask = self.project_gaussians(
            positions, covariances_3d, camera
        )
        
        if not torch.any(valid_mask):
            # No visible Gaussians, return background
            H, W = camera.height, camera.width
            result = {
                'color': torch.zeros(H, W, 3, device=self.device) + self.background_color,
                'depth': torch.zeros(H, W, device=self.device),
                'alpha': torch.zeros(H, W, device=self.device),
                'valid': False
            }
            return result
        
        # Sort Gaussians by depth
        sorted_indices = self.sort_gaussians_by_depth(means_2d, depths, valid_mask)
        
        # Create pixel grid
        H, W = camera.height, camera.width
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float()
        
        # Compute Gaussian weights for each pixel
        weights = self.compute_gaussian_weights(
            pixel_coords,
            means_2d[valid_mask],
            covariances_2d[valid_mask],
            opacities[valid_mask]
        )
        
        # Alpha blend
        rendered_color, accumulated_alpha, depth_map = self.alpha_blend(
            colors[valid_mask],
            weights,
            depths[valid_mask],
            sorted_indices
        )
        
        # Prepare result
        result = {
            'color': rendered_color,
            'depth': depth_map,
            'alpha': accumulated_alpha,
            'valid': True,
            'num_visible_gaussians': valid_mask.sum().item()
        }
        
        # Compute normals if requested
        if render_mode in ['normal', 'all']:
            normals = self.compute_normals(
                positions[valid_mask],
                covariances_3d[valid_mask],
                rendered_color,
                depth_map,
                camera
            )
            result['normal'] = normals
        
        # Record statistics
        render_time = time.time() - start_time
        self.stats['render_time'].append(render_time)
        self.stats['gaussians_per_frame'].append(valid_mask.sum().item())
        self.stats['tiles_per_frame'].append(self.num_tiles_x * self.num_tiles_y)
        
        if self.debug and len(self.stats['render_time']) % 100 == 0:
            avg_time = np.mean(self.stats['render_time'][-100:])
            avg_gaussians = np.mean(self.stats['gaussians_per_frame'][-100:])
            print(f"Render stats: time={avg_time:.3f}s, gaussians={avg_gaussians:.0f}")
        
        return result
    
    def _compute_sh_color(
        self,
        sh_coeffs: torch.Tensor,
        view_dirs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute RGB color from spherical harmonics.
        
        Args:
            sh_coeffs: Spherical harmonics coefficients [N, sh_dim, 3]
            view_dirs: View directions [N, 3]
            
        Returns:
            Colors [N, 3]
        """
        N = sh_coeffs.shape[0]
        sh_degree = int(np.sqrt(sh_coeffs.shape[1])) - 1
        
        # Compute spherical harmonics basis
        sh_basis = self._compute_sh_basis(view_dirs, sh_degree)
        
        # Multiply coefficients by basis
        colors = torch.zeros(N, 3, device=self.device)
        for i in range(3):  # RGB channels
            colors[:, i] = torch.sum(sh_coeffs[:, :, i] * sh_basis, dim=1)
        
        # Apply sigmoid for valid color range
        colors = torch.sigmoid(colors)
        
        return colors
    
    def _compute_sh_basis(
        self,
        dirs: torch.Tensor,
        degree: int
    ) -> torch.Tensor:
        """
        Compute spherical harmonics basis functions.
        
        Args:
            dirs: Normalized directions [N, 3]
            degree: Maximum SH degree
            
        Returns:
            SH basis values [N, (degree+1)^2]
        """
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        
        sh_basis = []
        
        # Degree 0
        sh_basis.append(0.28209479177387814 * torch.ones_like(x))  # Y_0^0
        
        if degree >= 1:
            # Degree 1
            sh_basis.append(-0.4886025119029199 * y)  # Y_1^{-1}
            sh_basis.append(0.4886025119029199 * z)   # Y_1^0
            sh_basis.append(-0.4886025119029199 * x)  # Y_1^1
        
        if degree >= 2:
            # Degree 2
            xy = x * y
            yz = y * z
            xz = x * z
            x2 = x * x
            y2 = y * y
            z2 = z * z
            
            sh_basis.append(1.0925484305920792 * xy)                    # Y_2^{-2}
            sh_basis.append(-1.0925484305920792 * yz)                   # Y_2^{-1}
            sh_basis.append(0.31539156525252005 * (3*z2 - 1))           # Y_2^0
            sh_basis.append(-1.0925484305920792 * xz)                   # Y_2^1
            sh_basis.append(0.5462742152960396 * (x2 - y2))             # Y_2^2
        
        if degree >= 3:
            # Degree 3
            xyz = x * y * z
            x2y = x * x * y
            xy2 = x * y * y
            x2z = x * x * z
            xz2 = x * z * z
            y2z = y * y * z
            yz2 = y * z * z
            x3 = x * x * x
            y3 = y * y * y
            z3 = z * z * z
            
            sh_basis.append(-0.5900435899266435 * y * (3*x2 - y2))      # Y_3^{-3}
            sh_basis.append(2.890611442640554 * xyz)                    # Y_3^{-2}
            sh_basis.append(-0.4570457994644658 * y * (5*z2 - 1))       # Y_3^{-1}
            sh_basis.append(0.3731763325901154 * z * (5*z2 - 3))        # Y_3^0
            sh_basis.append(-0.4570457994644658 * x * (5*z2 - 1))       # Y_3^1
            sh_basis.append(1.445305721320277 * (x2z - y2z))            # Y_3^2
            sh_basis.append(-0.5900435899266435 * x * (x2 - 3*y2))      # Y_3^3
        
        sh_basis = torch.stack(sh_basis, dim=1)
        
        return sh_basis
    
    def _compute_covariance_3d(
        self,
        rotations: torch.Tensor,
        scales: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 3D covariance matrix from rotation and scale.
        
        Args:
            rotations: Quaternions [N, 4]
            scales: Scales [N, 3]
            
        Returns:
            Covariance matrices [N, 3, 3]
        """
        N = rotations.shape[0]
        
        # Normalize quaternions
        rotations = F.normalize(rotations, dim=1)
        
        # Convert quaternions to rotation matrices
        qw, qx, qy, qz = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]
        
        # Rotation matrix
        R = torch.stack([
            1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy
        ], dim=1).view(-1, 3, 3)
        
        # Scale matrix
        S = torch.diag_embed(scales)  # [N, 3, 3]
        
        # Covariance = R @ S @ S.T @ R.T
        M = R @ S
        covariance = M @ M.transpose(1, 2)
        
        # Add small epsilon for numerical stability
        eps = torch.eye(3, device=self.device).unsqueeze(0) * 1e-6
        covariance = covariance + eps
        
        return covariance
    
    def batch_render(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        cameras: List[Camera],
        render_mode: str = 'color'
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Render multiple views.
        
        Args:
            gaussian_params: Gaussian parameters
            cameras: List of camera parameters
            render_mode: Rendering mode
            
        Returns:
            List of rendering results
        """
        results = []
        for camera in cameras:
            result = self.render(gaussian_params, camera, render_mode)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        if not self.stats['render_time']:
            return {}
        
        stats = {
            'total_frames_rendered': len(self.stats['render_time']),
            'avg_render_time': np.mean(self.stats['render_time']),
            'std_render_time': np.std(self.stats['render_time']),
            'min_render_time': np.min(self.stats['render_time']),
            'max_render_time': np.max(self.stats['render_time']),
            'avg_gaussians_per_frame': np.mean(self.stats['gaussians_per_frame']),
            'avg_tiles_per_frame': np.mean(self.stats['tiles_per_frame'])
        }
        
        # Add percentiles
        if len(self.stats['render_time']) > 10:
            for p in [50, 90, 95, 99]:
                stats[f'p{p}_render_time'] = np.percentile(self.stats['render_time'], p)
        
        return stats
    
    def clear_statistics(self):
        """Clear rendering statistics."""
        for key in self.stats:
            self.stats[key] = []