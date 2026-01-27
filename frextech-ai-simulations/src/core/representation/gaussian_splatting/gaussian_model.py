"""
Gaussian Splatting 3D representation model.
Based on 3D Gaussian Splatting for Real-Time Radiance Field Rendering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
import gzip
import pickle

@dataclass
class GaussianParams:
    """Parameters for a single Gaussian."""
    position: torch.Tensor  # [3]
    rotation: torch.Tensor  # [4] quaternion (w, x, y, z)
    scale: torch.Tensor     # [3]
    opacity: torch.Tensor   # [1]
    sh_coeffs: torch.Tensor  # [N, 3] spherical harmonics coefficients
    feature_vector: Optional[torch.Tensor] = None  # [D] optional feature for neural rendering


class GaussianModel(nn.Module):
    """
    Main Gaussian Splatting model that manages a set of 3D Gaussians.
    
    Attributes:
        max_gaussians: Maximum number of Gaussians
        sh_degree: Degree of spherical harmonics (0 for diffuse, 3 for view-dependent)
        learnable_opacity: Whether opacity is learnable
        device: Device to store tensors on
    """
    
    def __init__(
        self,
        max_gaussians: int = 1000000,
        sh_degree: int = 3,
        learnable_opacity: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        init_method: str = "point_cloud",
        init_path: Optional[str] = None
    ):
        super().__init__()
        
        self.max_gaussians = max_gaussians
        self.sh_degree = sh_degree
        self.learnable_opacity = learnable_opacity
        self.device = device
        
        # Number of spherical harmonics coefficients
        self.sh_dim = (sh_degree + 1) ** 2
        
        # Initialize parameters
        self._init_parameters()
        
        # Initialize from point cloud if specified
        if init_path and init_method == "point_cloud":
            self.initialize_from_point_cloud(init_path)
        elif init_method == "random":
            self.initialize_random(1000)  # Start with 1000 random Gaussians
    
    def _init_parameters(self):
        """Initialize learnable parameters."""
        # Positions [N, 3]
        self.positions = nn.Parameter(
            torch.zeros((self.max_gaussians, 3), device=self.device),
            requires_grad=True
        )
        
        # Rotations as quaternions [N, 4]
        self.rotations = nn.Parameter(
            torch.zeros((self.max_gaussians, 4), device=self.device),
            requires_grad=True
        )
        # Initialize with identity quaternions
        with torch.no_grad():
            self.rotations[:, 0] = 1.0  # w = 1, x=y=z=0
        
        # Scales [N, 3]
        self.scales = nn.Parameter(
            torch.zeros((self.max_gaussians, 3), device=self.device),
            requires_grad=True
        )
        
        # Opacities [N, 1]
        self.opacities = nn.Parameter(
            torch.zeros((self.max_gaussians, 1), device=self.device),
            requires_grad=self.learnable_opacity
        )
        # Initialize opacities with sigmoid(0) = 0.5
        with torch.no_grad():
            self.opacities.fill_(0.0)
        
        # Spherical harmonics coefficients [N, sh_dim, 3]
        self.sh_coeffs = nn.Parameter(
            torch.zeros((self.max_gaussians, self.sh_dim, 3), device=self.device),
            requires_grad=True
        )
        # Initialize with diffuse color (only degree 0)
        with torch.no_grad():
            # Random RGB colors
            self.sh_coeffs[:, 0, :] = torch.rand((self.max_gaussians, 3), device=self.device) * 0.5 + 0.5
        
        # Feature vectors for neural rendering [N, feature_dim]
        self.feature_dim = 32  # Can be configurable
        self.features = nn.Parameter(
            torch.zeros((self.max_gaussians, self.feature_dim), device=self.device),
            requires_grad=True
        )
        
        # Active mask [N] - which Gaussians are actually used
        self.active_mask = torch.zeros(self.max_gaussians, dtype=torch.bool, device=self.device)
        
        # Gradient accumulators for adaptive density control
        self.grad_accum = torch.zeros(self.max_gaussians, device=self.device)
        self.optimizer_state = {}
        
        # Bounding box for spatial queries
        self.bounds_min = torch.zeros(3, device=self.device)
        self.bounds_max = torch.zeros(3, device=self.device)
        
        # Octree for acceleration
        self.octree = None
        self.octree_depth = 8
        
    def initialize_from_point_cloud(self, pointcloud_path: str, color_path: Optional[str] = None):
        """
        Initialize Gaussians from a point cloud.
        
        Args:
            pointcloud_path: Path to point cloud file (.ply, .pcd, .npz)
            color_path: Optional path to color file
        """
        print(f"Initializing Gaussian model from point cloud: {pointcloud_path}")
        
        # Load point cloud
        if pointcloud_path.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(pointcloud_path)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        elif pointcloud_path.endswith('.npz'):
            data = np.load(pointcloud_path)
            points = data['points']
            colors = data['colors'] if 'colors' in data else None
        else:
            raise ValueError(f"Unsupported point cloud format: {pointcloud_path}")
        
        # Convert to torch
        points_tensor = torch.from_numpy(points).float().to(self.device)
        num_points = min(points_tensor.shape[0], self.max_gaussians)
        
        # Update active mask
        self.active_mask[:num_points] = True
        self.active_mask[num_points:] = False
        
        # Set positions
        with torch.no_grad():
            self.positions[:num_points] = points_tensor[:num_points]
            
            # Initialize scales based on nearest neighbor distances
            from sklearn.neighbors import NearestNeighbors
            points_np = points_tensor[:num_points].cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(points_np)
            distances, _ = nbrs.kneighbors(points_np)
            avg_distances = torch.from_numpy(distances.mean(axis=1)).float().to(self.device)
            
            # Log scale initialization
            init_scales = torch.log(torch.clamp(avg_distances.unsqueeze(-1) * 0.3, min=1e-6))
            self.scales[:num_points] = init_scales.unsqueeze(-1).repeat(1, 3)
            
            # Initialize colors if available
            if colors is not None:
                colors_tensor = torch.from_numpy(colors[:num_points]).float().to(self.device)
                self.sh_coeffs[:num_points, 0, :] = colors_tensor
            
            # Initialize opacities
            self.opacities[:num_points].fill_(0.0)  # sigmoid(0) = 0.5
            
            # Update bounds
            self.bounds_min = points_tensor.min(dim=0)[0]
            self.bounds_max = points_tensor.max(dim=0)[0]
        
        print(f"Initialized {num_points} Gaussians from point cloud")
        
    def initialize_random(self, num_gaussians: int, bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Initialize random Gaussians within bounds.
        
        Args:
            num_gaussians: Number of Gaussians to initialize
            bounds: Optional (min, max) bounds, otherwise unit cube
        """
        num_gaussians = min(num_gaussians, self.max_gaussians)
        
        with torch.no_grad():
            if bounds is None:
                bounds_min = torch.tensor([-1, -1, -1], device=self.device)
                bounds_max = torch.tensor([1, 1, 1], device=self.device)
            else:
                bounds_min, bounds_max = bounds
            
            # Random positions
            self.positions[:num_gaussians] = torch.rand(num_gaussians, 3, device=self.device) * (bounds_max - bounds_min) + bounds_min
            
            # Random scales (log scale)
            self.scales[:num_gaussians] = torch.log(torch.rand(num_gaussians, 3, device=self.device) * 0.1 + 1e-6)
            
            # Random colors
            self.sh_coeffs[:num_gaussians, 0, :] = torch.rand(num_gaussians, 3, device=self.device)
            
            # Random opacities
            self.opacities[:num_gaussians].uniform_(-2, 2)  # sigmoid range ~0.12 to 0.88
            
            # Random rotations (normalized quaternions)
            random_quats = torch.randn(num_gaussians, 4, device=self.device)
            self.rotations[:num_gaussians] = F.normalize(random_quats, dim=1)
            
            # Update active mask and bounds
            self.active_mask[:num_gaussians] = True
            self.active_mask[num_gaussians:] = False
            self.bounds_min = bounds_min
            self.bounds_max = bounds_max
    
    def get_active_gaussians(self) -> Dict[str, torch.Tensor]:
        """Get parameters for active Gaussians only."""
        mask = self.active_mask
        n_active = mask.sum().item()
        
        return {
            'positions': self.positions[mask],
            'rotations': F.normalize(self.rotations[mask], dim=1),  # Ensure normalized
            'scales': torch.exp(self.scales[mask]),  # Convert from log scale
            'opacities': torch.sigmoid(self.opacities[mask]),
            'sh_coeffs': self.sh_coeffs[mask],
            'features': self.features[mask] if self.features is not None else None
        }
    
    def get_covariance_matrix(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute 3x3 covariance matrix for Gaussians.
        
        Args:
            idx: Optional indices of Gaussians, otherwise all active
            
        Returns:
            Covariance matrices [N, 3, 3]
        """
        if idx is None:
            mask = self.active_mask
            rotations = self.rotations[mask]
            scales = self.scales[mask]
        else:
            rotations = self.rotations[idx]
            scales = self.scales[idx]
        
        # Convert quaternions to rotation matrices
        q = rotations  # [N, 4]
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        
        # Rotation matrix from quaternion
        R = torch.stack([
            1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
            2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw,
            2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy
        ], dim=1).view(-1, 3, 3)
        
        # Scale matrix
        S = torch.diag_embed(torch.exp(scales))  # [N, 3, 3]
        
        # Covariance = R @ S @ S.T @ R.T
        M = R @ S
        covariance = M @ M.transpose(1, 2)
        
        return covariance
    
    def compute_sh_color(self, sh_coeffs: torch.Tensor, view_dir: torch.Tensor) -> torch.Tensor:
        """
        Compute RGB color from spherical harmonics coefficients and view direction.
        
        Args:
            sh_coeffs: Spherical harmonics coefficients [N, sh_dim, 3]
            view_dir: Normalized view direction [N, 3]
            
        Returns:
            Colors [N, 3]
        """
        batch_size = sh_coeffs.shape[0]
        
        # Compute spherical harmonics basis functions
        sh_basis = self._compute_sh_basis(view_dir)  # [N, sh_dim]
        
        # Multiply coefficients by basis
        color = torch.zeros(batch_size, 3, device=sh_coeffs.device)
        for i in range(3):  # RGB channels
            color[:, i] = torch.sum(sh_coeffs[:, :, i] * sh_basis, dim=1)
        
        # Apply sigmoid to get valid colors
        color = torch.sigmoid(color)
        
        return color
    
    def _compute_sh_basis(self, dirs: torch.Tensor) -> torch.Tensor:
        """
        Compute spherical harmonics basis functions up to degree sh_degree.
        
        Args:
            dirs: Normalized directions [N, 3]
            
        Returns:
            SH basis values [N, sh_dim]
        """
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        
        sh_basis = []
        
        # Degree 0
        sh_basis.append(0.28209479177387814 * torch.ones_like(x))  # Y_0^0
        
        if self.sh_degree >= 1:
            # Degree 1
            sh_basis.append(-0.4886025119029199 * y)  # Y_1^{-1}
            sh_basis.append(0.4886025119029199 * z)   # Y_1^0
            sh_basis.append(-0.4886025119029199 * x)  # Y_1^1
        
        if self.sh_degree >= 2:
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
        
        if self.sh_degree >= 3:
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
    
    def density_control(
        self,
        grad_threshold: float = 0.0002,
        density_threshold: float = 0.01,
        max_grad: float = 0.5,
        scene_extent: float = 1.0
    ):
        """
        Adaptive density control: clone, split, or prune Gaussians based on gradients.
        
        Args:
            grad_threshold: Threshold for gradient magnitude to trigger cloning/splitting
            density_threshold: Opacity threshold for pruning
            max_grad: Maximum gradient magnitude for gradient clipping
            scene_extent: Extent of the scene for scale clamping
        """
        if not hasattr(self, 'grad_accum'):
            return
        
        mask = self.active_mask
        n_active = mask.sum().item()
        
        if n_active == 0:
            return
        
        # Get gradients for positions (approximate)
        if self.positions.grad is not None:
            position_grads = self.positions.grad[mask].norm(dim=1)
            self.grad_accum[mask] += position_grads
        
        # Normalize gradients
        grad_norm = self.grad_accum[mask]
        if grad_norm.numel() > 0:
            grad_norm = grad_norm / (grad_norm.mean() + 1e-8)
        
        # Get current parameters
        positions = self.positions[mask]
        rotations = self.rotations[mask]
        scales = torch.exp(self.scales[mask])
        opacities = torch.sigmoid(self.opacities[mask])
        
        # Clone/split conditions
        clone_mask = (grad_norm > grad_threshold) & (scales.max(dim=1)[0] < scene_extent * 0.1)
        split_mask = (grad_norm > grad_threshold) & (scales.max(dim=1)[0] >= scene_extent * 0.1)
        
        # Prune conditions
        prune_mask = (opacities.squeeze() < density_threshold)
        
        # Find indices for new Gaussians
        free_indices = torch.where(~self.active_mask)[0]
        num_free = free_indices.shape[0]
        
        # Clone operations
        clone_indices = torch.where(clone_mask)[0]
        num_clone = min(clone_indices.shape[0], num_free)
        
        if num_clone > 0:
            clone_idx = clone_indices[:num_clone]
            free_idx = free_indices[:num_clone]
            
            with torch.no_grad():
                # Copy parameters
                self.positions[free_idx] = self.positions[mask][clone_idx]
                self.rotations[free_idx] = self.rotations[mask][clone_idx]
                self.scales[free_idx] = self.scales[mask][clone_idx]
                self.opacities[free_idx] = self.opacities[mask][clone_idx] * 0.8  # Reduce opacity
                self.sh_coeffs[free_idx] = self.sh_coeffs[mask][clone_idx]
                self.features[free_idx] = self.features[mask][clone_idx] if self.features is not None else None
                
                # Add small noise to positions
                self.positions[free_idx] += torch.randn_like(self.positions[free_idx]) * 0.01
                
                # Mark as active
                self.active_mask[free_idx] = True
                
                # Reset gradients for cloned Gaussians
                self.grad_accum[free_idx] = 0.0
            
            free_indices = free_indices[num_clone:]
        
        # Split operations
        split_indices = torch.where(split_mask)[0]
        num_split = min(split_indices.shape[0], free_indices.shape[0])
        
        if num_split > 0:
            split_idx = split_indices[:num_split]
            free_idx = free_indices[:num_split]
            
            with torch.no_grad():
                # Copy parameters
                self.positions[free_idx] = self.positions[mask][split_idx]
                self.rotations[free_idx] = self.rotations[mask][split_idx]
                
                # Halve the scale
                self.scales[free_idx] = self.scales[mask][split_idx] - np.log(2.0)
                
                # Reduce opacity
                self.opacities[free_idx] = self.opacities[mask][split_idx] * 0.5
                self.sh_coeffs[free_idx] = self.sh_coeffs[mask][split_idx]
                self.features[free_idx] = self.features[mask][split_idx] if self.features is not None else None
                
                # Add noise to positions
                self.positions[free_idx] += torch.randn_like(self.positions[free_idx]) * torch.exp(self.scales[free_idx]) * 0.2
                
                # Mark as active
                self.active_mask[free_idx] = True
                
                # Reset gradients
                self.grad_accum[free_idx] = 0.0
                
                # Also halve scale of original
                self.scales[mask][split_idx] = self.scales[mask][split_idx] - np.log(2.0)
        
        # Prune operations
        prune_indices = torch.where(prune_mask)[0]
        if prune_indices.shape[0] > 0:
            # Get global indices of pruned Gaussians
            active_indices = torch.where(mask)[0]
            prune_global = active_indices[prune_indices]
            
            # Mark as inactive
            self.active_mask[prune_global] = False
            
            # Reset gradients
            self.grad_accum[prune_global] = 0.0
        
        # Reset gradient accumulation
        self.grad_accum.zero_()
    
    def build_octree(self, max_depth: int = 8, min_points_per_node: int = 8):
        """Build octree for spatial acceleration."""
        from .octree import OctreeNode
        
        mask = self.active_mask
        positions = self.positions[mask].cpu().numpy()
        
        self.octree = OctreeNode(
            self.bounds_min.cpu().numpy(),
            self.bounds_max.cpu().numpy(),
            max_depth=max_depth,
            min_points=min_points_per_node
        )
        
        # Add points to octree
        indices = np.arange(len(positions))
        self.octree.insert_points(positions, indices)
        
        # Move octree to device
        self.octree.to_device(self.device)
    
    def query_octree(self, points: torch.Tensor, max_dist: float = 0.1) -> List[torch.Tensor]:
        """
        Query octree for Gaussians near given points.
        
        Args:
            points: Query points [N, 3]
            max_dist: Maximum distance for neighbors
            
        Returns:
            List of indices for each query point
        """
        if self.octree is None:
            self.build_octree()
        
        return self.octree.query_points(points.cpu().numpy(), max_dist)
    
    def save(self, path: str, compress: bool = True):
        """
        Save Gaussian model to file.
        
        Args:
            path: Path to save file
            compress: Whether to compress with gzip
        """
        # Get active parameters
        active_params = self.get_active_gaussians()
        
        # Convert to CPU and numpy for saving
        save_data = {}
        for key, value in active_params.items():
            if value is not None:
                save_data[key] = value.detach().cpu().numpy()
        
        # Add metadata
        save_data['metadata'] = {
            'sh_degree': self.sh_degree,
            'feature_dim': self.feature_dim,
            'bounds_min': self.bounds_min.cpu().numpy(),
            'bounds_max': self.bounds_max.cpu().numpy(),
            'device': str(self.device)
        }
        
        # Save
        if compress:
            with gzip.open(path, 'wb') as f:
                pickle.dump(save_data, f)
        else:
            with open(path, 'wb') as f:
                pickle.dump(save_data, f)
        
        print(f"Saved Gaussian model to {path} with {len(save_data['positions'])} Gaussians")
    
    def load(self, path: str, compress: bool = True):
        """
        Load Gaussian model from file.
        
        Args:
            path: Path to load file from
            compress: Whether file is compressed with gzip
        """
        # Load data
        if compress:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        
        # Extract parameters
        positions = torch.from_numpy(data['positions']).float().to(self.device)
        rotations = torch.from_numpy(data['rotations']).float().to(self.device)
        scales = torch.from_numpy(data['scales']).float().to(self.device)
        opacities = torch.from_numpy(data['opacities']).float().to(self.device)
        sh_coeffs = torch.from_numpy(data['sh_coeffs']).float().to(self.device)
        
        n_gaussians = positions.shape[0]
        
        # Reset and load
        with torch.no_grad():
            self.positions.zero_()
            self.rotations.zero_()
            self.rotations[:, 0] = 1.0
            self.scales.zero_()
            self.opacities.zero_()
            self.sh_coeffs.zero_()
            
            self.positions[:n_gaussians] = positions
            self.rotations[:n_gaussians] = rotations
            self.scales[:n_gaussians] = torch.log(scales)  # Convert to log scale
            self.opacities[:n_gaussians] = torch.logit(opacities)  # Convert to logit
            
            # Handle SH coefficients dimension mismatch
            loaded_sh_dim = sh_coeffs.shape[1]
            if loaded_sh_dim <= self.sh_dim:
                self.sh_coeffs[:n_gaussians, :loaded_sh_dim, :] = sh_coeffs
            else:
                self.sh_coeffs[:n_gaussians] = sh_coeffs[:, :self.sh_dim, :]
            
            # Update active mask
            self.active_mask.zero_()
            self.active_mask[:n_gaussians] = True
            
            # Update bounds
            if 'bounds_min' in data.get('metadata', {}):
                self.bounds_min = torch.from_numpy(data['metadata']['bounds_min']).to(self.device)
                self.bounds_max = torch.from_numpy(data['metadata']['bounds_max']).to(self.device)
            else:
                self.bounds_min = positions.min(dim=0)[0]
                self.bounds_max = positions.max(dim=0)[0]
        
        print(f"Loaded Gaussian model from {path} with {n_gaussians} Gaussians")
    
    def to_point_cloud(self) -> o3d.geometry.PointCloud:
        """Convert Gaussians to point cloud for visualization."""
        active_params = self.get_active_gaussians()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(active_params['positions'].cpu().numpy())
        
        # Use SH degree 0 color
        colors = torch.sigmoid(active_params['sh_coeffs'][:, 0, :]).cpu().numpy()
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def forward(
        self,
        camera_pos: torch.Tensor,
        camera_rot: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Tuple[int, int],
        render_mode: str = 'color'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for rendering.
        
        Args:
            camera_pos: Camera position [3]
            camera_rot: Camera rotation (quaternion) [4]
            intrinsics: Camera intrinsics [3, 3]
            image_size: (height, width) of output image
            render_mode: 'color', 'depth', 'normal', or 'feature'
            
        Returns:
            Dictionary with rendered image and optional buffers
        """
        # This is a placeholder - actual rendering is done in rasterizer
        # Here we just return parameters needed for rendering
        return {
            'gaussian_params': self.get_active_gaussians(),
            'camera_pos': camera_pos,
            'camera_rot': camera_rot,
            'intrinsics': intrinsics,
            'image_size': image_size
        }


class OctreeNode:
    """Octree node for spatial acceleration."""
    
    def __init__(
        self,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        max_depth: int = 8,
        min_points: int = 8,
        depth: int = 0
    ):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.center = (bounds_min + bounds_max) / 2
        self.extent = bounds_max - bounds_min
        self.max_depth = max_depth
        self.min_points = min_points
        self.depth = depth
        
        self.points = None
        self.indices = None
        self.children = None
        self.is_leaf = True
        
    def insert_points(self, points: np.ndarray, indices: np.ndarray):
        """Insert points into octree."""
        if len(points) == 0:
            return
        
        # Check if should split
        if (self.depth < self.max_depth and 
            len(points) > self.min_points and
            np.any(self.extent > 1e-6)):
            
            self.split()
            
            # Assign points to children
            for child in self.children:
                # Check which points are in child's bounds
                in_child = np.all(
                    (points >= child.bounds_min) & (points <= child.bounds_max),
                    axis=1
                )
                if np.any(in_child):
                    child.insert_points(points[in_child], indices[in_child])
        else:
            # Store points in leaf
            self.points = points
            self.indices = indices
    
    def split(self):
        """Split node into 8 children."""
        self.children = []
        self.is_leaf = False
        
        # Create 8 children
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_min = self.bounds_min.copy()
                    child_max = self.center.copy()
                    
                    if i == 1:
                        child_min[0] = self.center[0]
                        child_max[0] = self.bounds_max[0]
                    if j == 1:
                        child_min[1] = self.center[1]
                        child_max[1] = self.bounds_max[1]
                    if k == 1:
                        child_min[2] = self.center[2]
                        child_max[2] = self.bounds_max[2]
                    
                    child = OctreeNode(
                        child_min, child_max,
                        self.max_depth, self.min_points,
                        self.depth + 1
                    )
                    self.children.append(child)
    
    def query_points(self, query_points: np.ndarray, max_dist: float) -> List[np.ndarray]:
        """Query points within max_dist."""
        results = [[] for _ in range(len(query_points))]
        
        def _query_node(node, query_idx, query_point):
            if node.is_leaf and node.points is not None:
                # Compute distances
                dists = np.linalg.norm(node.points - query_point, axis=1)
                nearby = dists < max_dist
                
                if np.any(nearby):
                    results[query_idx].extend(node.indices[nearby].tolist())
            elif not node.is_leaf:
                # Check which child contains the query point
                for child in node.children:
                    if np.all(
                        (query_point >= child.bounds_min - max_dist) &
                        (query_point <= child.bounds_max + max_dist)
                    ):
                        _query_node(child, query_idx, query_point)
        
        for i, point in enumerate(query_points):
            _query_node(self, i, point)
        
        return [np.array(indices, dtype=np.int64) for indices in results]
    
    def to_device(self, device: str):
        """Move octree data to device (currently CPU only)."""
        # Octree operations are on CPU for now
        pass