"""
Post-processing for 3D World Generation
Refinement, enhancement, and optimization of generated 3D scenes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from scipy import ndimage
import cv2
from skimage import filters, morphology, segmentation
import trimesh
from PIL import Image
import mcubes

logger = logging.getLogger(__name__)

@dataclass
class PostProcessConfig:
    """Configuration for post-processing"""
    denoise_strength: float = 0.5
    sharpen_amount: float = 0.3
    smooth_iterations: int = 2
    fill_holes: bool = True
    remove_isolated: bool = True
    min_component_size: int = 100
    edge_preserve: bool = True
    color_correction: bool = True
    tone_mapping: bool = True
    export_quality: str = "high"
    compression_level: int = 6

class WorldPostProcessor:
    """Post-processor for 3D generated worlds"""
    
    def __init__(self, config: Optional[PostProcessConfig] = None):
        self.config = config or PostProcessConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize processing modules
        self._init_modules()
        
    def _init_modules(self):
        """Initialize processing modules"""
        # Denoising filters
        self.gaussian_filter_3d = self._create_gaussian_filter_3d()
        self.bilateral_filter_3d = self._create_bilateral_filter_3d()
        
        # Edge detection
        self.sobel_filter_3d = self._create_sobel_filter_3d()
        
        # Color correction LUTs
        self.color_luts = self._create_color_luts()
        
    def _create_gaussian_filter_3d(self, sigma: float = 1.0) -> torch.Tensor:
        """Create 3D Gaussian filter kernel"""
        kernel_size = 5
        coords = torch.arange(kernel_size) - kernel_size // 2
        x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _create_bilateral_filter_3d(self) -> callable:
        """Create bilateral filter function"""
        def bilateral_filter(
            volume: torch.Tensor,
            spatial_sigma: float = 1.0,
            intensity_sigma: float = 0.1
        ) -> torch.Tensor:
            """Apply bilateral filtering to 3D volume"""
            # Simplified implementation - in production would use optimized version
            kernel_size = 5
            padding = kernel_size // 2
            
            result = torch.zeros_like(volume)
            
            for d in range(padding, volume.shape[2] - padding):
                for h in range(padding, volume.shape[3] - padding):
                    for w in range(padding, volume.shape[4] - padding):
                        # Extract local patch
                        patch = volume[:, :, 
                                      d-padding:d+padding+1,
                                      h-padding:h+padding+1,
                                      w-padding:w+padding+1]
                        
                        # Spatial weights
                        coords = torch.arange(kernel_size) - padding
                        x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
                        spatial_weights = torch.exp(-(x**2 + y**2 + z**2) / (2 * spatial_sigma**2))
                        
                        # Intensity weights
                        center_val = volume[:, :, d, h, w]
                        intensity_diff = patch - center_val
                        intensity_weights = torch.exp(-intensity_diff**2 / (2 * intensity_sigma**2))
                        
                        # Combined weights
                        weights = spatial_weights * intensity_weights
                        weights = weights / weights.sum()
                        
                        # Apply weighted average
                        result[:, :, d, h, w] = (patch * weights).sum(dim=(2, 3, 4))
            
            return result
        
        return bilateral_filter
    
    def _create_sobel_filter_3d(self) -> Tuple[torch.Tensor, ...]:
        """Create 3D Sobel filters for edge detection"""
        # Sobel kernels for 3D
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32) / 32.0
        
        sobel_y = sobel_x.transpose(1, 2)
        sobel_z = sobel_x.transpose(0, 2)
        
        return (
            sobel_x.unsqueeze(0).unsqueeze(0),
            sobel_y.unsqueeze(0).unsqueeze(0),
            sobel_z.unsqueeze(0).unsqueeze(0)
        )
    
    def _create_color_luts(self) -> Dict[str, np.ndarray]:
        """Create color correction lookup tables"""
        luts = {}
        
        # Film emulation LUTs
        luts["film_contrast"] = self._create_film_contrast_lut()
        luts["warm_tone"] = self._create_warm_tone_lut()
        luts["cool_tone"] = self._create_cool_tone_lut()
        luts["cinematic"] = self._create_cinematic_lut()
        
        return luts
    
    def _create_film_contrast_lut(self) -> np.ndarray:
        """Create film contrast LUT"""
        lut = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            # S-curve for film-like contrast
            x = i / 255.0
            y = 1 / (1 + np.exp(-10 * (x - 0.5)))  # Sigmoid
            lut[i] = y
        
        return lut
    
    def _create_warm_tone_lut(self) -> np.ndarray:
        """Create warm tone LUT"""
        lut = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            # Warm tone: boost red/orange, reduce blue
            x = i / 255.0
            lut[i, 0] = x  # Red
            lut[i, 1] = x * 0.9  # Green
            lut[i, 2] = x * 0.8  # Blue
        
        return lut
    
    def _create_cool_tone_lut(self) -> np.ndarray:
        """Create cool tone LUT"""
        lut = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            # Cool tone: boost blue/cyan
            x = i / 255.0
            lut[i, 0] = x * 0.8  # Red
            lut[i, 1] = x * 0.9  # Green
            lut[i, 2] = x  # Blue
        
        return lut
    
    def _create_cinematic_lut(self) -> np.ndarray:
        """Create cinematic LUT (teal and orange)"""
        lut = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            # Teal and orange look
            x = i / 255.0
            # Boost orange in highlights, teal in shadows
            if x > 0.5:
                lut[i, 0] = x * 1.2  # Orange
                lut[i, 1] = x * 0.9
                lut[i, 2] = x * 0.8
            else:
                lut[i, 0] = x * 0.8  # Teal
                lut[i, 1] = x
                lut[i, 2] = x * 1.1
        
        return lut
    
    def process(
        self,
        world_data: Dict[str, Any],
        operations: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply post-processing operations to world data
        
        Args:
            world_data: Raw world data from generator
            operations: List of operations to apply
            **kwargs: Additional parameters for operations
        
        Returns:
            Processed world data
        """
        if operations is None:
            operations = [
                "denoise",
                "smooth",
                "enhance_edges",
                "color_correct",
                "fill_holes"
            ]
        
        processed_data = world_data.copy()
        
        for operation in operations:
            logger.info(f"Applying post-processing: {operation}")
            
            if operation == "denoise":
                processed_data = self.denoise(processed_data, **kwargs)
            elif operation == "smooth":
                processed_data = self.smooth_surfaces(processed_data, **kwargs)
            elif operation == "enhance_edges":
                processed_data = self.enhance_edges(processed_data, **kwargs)
            elif operation == "color_correct":
                processed_data = self.color_correct(processed_data, **kwargs)
            elif operation == "fill_holes":
                processed_data = self.fill_holes(processed_data, **kwargs)
            elif operation == "remove_islands":
                processed_data = self.remove_isolated_components(processed_data, **kwargs)
            elif operation == "sharp_details":
                processed_data = self.sharp_details(processed_data, **kwargs)
            elif operation == "tone_map":
                processed_data = self.tone_mapping(processed_data, **kwargs)
            elif operation == "optimize_mesh":
                processed_data = self.optimize_mesh(processed_data, **kwargs)
            else:
                logger.warning(f"Unknown operation: {operation}")
        
        return processed_data
    
    def denoise(
        self,
        world_data: Dict[str, Any],
        method: str = "bilateral",
        strength: Optional[float] = None
    ) -> Dict[str, Any]:
        """Apply denoising to 3D data"""
        strength = strength or self.config.denoise_strength
        
        if "density_grid" in world_data:
            # NeRF representation
            density = torch.from_numpy(world_data["density_grid"]).float().to(self.device)
            color = torch.from_numpy(world_data["color_grid"]).float().to(self.device)
            
            if method == "gaussian":
                density = self._apply_gaussian_filter_3d(density, sigma=strength * 2)
                color = self._apply_gaussian_filter_3d(color, sigma=strength)
            elif method == "bilateral":
                density = self.bilateral_filter_3d(
                    density.unsqueeze(0).unsqueeze(0),
                    spatial_sigma=strength,
                    intensity_sigma=strength * 0.1
                ).squeeze()
                color = self.bilateral_filter_3d(
                    color.unsqueeze(0).unsqueeze(0),
                    spatial_sigma=strength * 0.5,
                    intensity_sigma=strength * 0.05
                ).squeeze()
            
            world_data["density_grid"] = density.cpu().numpy()
            world_data["color_grid"] = color.cpu().numpy()
            
        elif "positions" in world_data:
            # Gaussian splatting representation
            # Denoise by filtering gaussian parameters
            scales = world_data["scales"]
            opacities = world_data["opacities"]
            
            # Apply mild smoothing to scales and opacities
            scales = ndimage.gaussian_filter(scales, sigma=strength, mode='nearest')
            opacities = ndimage.gaussian_filter(opacities, sigma=strength * 0.5, mode='nearest')
            
            world_data["scales"] = scales
            world_data["opacities"] = opacities
            
        elif "vertices" in world_data:
            # Mesh representation - apply mesh smoothing
            vertices = world_data["vertices"]
            triangles = world_data["triangles"]
            
            # Convert to trimesh for smoothing
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            
            # Apply Laplacian smoothing
            smoothed = trimesh.smoothing.filter_laplacian(mesh, iterations=2, volume_constraint=True)
            
            world_data["vertices"] = smoothed.vertices
            world_data["vertex_normals"] = smoothed.vertex_normals
        
        return world_data
    
    def _apply_gaussian_filter_3d(
        self,
        volume: torch.Tensor,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """Apply 3D Gaussian filter"""
        kernel = self._create_gaussian_filter_3d(sigma).to(volume.device)
        
        # Apply convolution to each channel
        if volume.ndim == 3:  # Single channel
            volume = volume.unsqueeze(0).unsqueeze(0)
            filtered = F.conv3d(volume, kernel, padding=kernel.shape[-1]//2)
            return filtered.squeeze()
        elif volume.ndim == 4:  # Multi-channel (e.g., RGB)
            volume = volume.unsqueeze(0)
            filtered = F.conv3d(volume, kernel.repeat(volume.shape[1], 1, 1, 1, 1), 
                              padding=kernel.shape[-1]//2, groups=volume.shape[1])
            return filtered.squeeze()
        else:
            raise ValueError(f"Unsupported volume shape: {volume.shape}")
    
    def smooth_surfaces(
        self,
        world_data: Dict[str, Any],
        iterations: Optional[int] = None,
        preserve_edges: bool = True
    ) -> Dict[str, Any]:
        """Smooth surfaces while preserving edges"""
        iterations = iterations or self.config.smooth_iterations
        
        if "density_grid" in world_data:
            density = world_data["density_grid"]
            
            if preserve_edges:
                # Edge-preserving smoothing
                for _ in range(iterations):
                    # Detect edges
                    edges = self._detect_edges_3d(density)
                    
                    # Apply selective smoothing
                    smoothed = ndimage.gaussian_filter(density, sigma=0.5)
                    
                    # Blend based on edge strength
                    alpha = 1 - edges
                    density = density * alpha + smoothed * (1 - alpha)
            else:
                # Regular smoothing
                density = ndimage.gaussian_filter(density, sigma=1.0, iterations=iterations)
            
            world_data["density_grid"] = density
            
        elif "vertices" in world_data:
            # Mesh smoothing
            vertices = world_data["vertices"]
            triangles = world_data["triangles"]
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            
            if preserve_edges:
                # Detect sharp edges
                edge_mask = self._detect_sharp_edges(mesh)
                
                # Apply constrained smoothing
                smoothed = trimesh.smoothing.filter_laplacian(
                    mesh, 
                    iterations=iterations,
                    volume_constraint=True,
                    laplacian_operator='cotangent'
                )
                
                # Preserve sharp edges
                smoothed_vertices = smoothed.vertices
                for i, is_sharp in enumerate(edge_mask):
                    if is_sharp:
                        smoothed_vertices[i] = vertices[i]
                
                world_data["vertices"] = smoothed_vertices
            else:
                # Regular mesh smoothing
                smoothed = trimesh.smoothing.filter_laplacian(
                    mesh, 
                    iterations=iterations,
                    volume_constraint=True
                )
                world_data["vertices"] = smoothed.vertices
        
        return world_data
    
    def _detect_edges_3d(self, volume: np.ndarray) -> np.ndarray:
        """Detect edges in 3D volume"""
        # Compute gradient magnitude
        grad_x = ndimage.sobel(volume, axis=0)
        grad_y = ndimage.sobel(volume, axis=1)
        grad_z = ndimage.sobel(volume, axis=2)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Normalize and threshold
        gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        edges = gradient_magnitude > 0.2
        
        return edges
    
    def _detect_sharp_edges(self, mesh: trimesh.Trimesh, threshold: float = 30.0) -> np.ndarray:
        """Detect sharp edges in mesh (degrees)"""
        # Compute dihedral angles
        faces = mesh.faces
        vertices = mesh.vertices
        
        # Get face normals
        face_normals = mesh.face_normals
        
        # Find edges and adjacent faces
        edges = {}
        for i, face in enumerate(faces):
            for j in range(3):
                edge = tuple(sorted([face[j], face[(j+1)%3]]))
                if edge not in edges:
                    edges[edge] = []
                edges[edge].append(i)
        
        # Mark sharp edges
        vertex_sharp = np.zeros(len(vertices), dtype=bool)
        sharp_threshold = np.radians(threshold)
        
        for edge, face_indices in edges.items():
            if len(face_indices) == 2:
                # Compute angle between face normals
                n1 = face_normals[face_indices[0]]
                n2 = face_normals[face_indices[1]]
                angle = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))
                
                if angle > sharp_threshold:
                    vertex_sharp[list(edge)] = True
        
        return vertex_sharp
    
    def enhance_edges(
        self,
        world_data: Dict[str, Any],
        strength: float = 1.0
    ) -> Dict[str, Any]:
        """Enhance edges and details"""
        
        if "density_grid" in world_data:
            density = world_data["density_grid"]
            color = world_data.get("color_grid", None)
            
            # Edge enhancement using unsharp masking
            blurred = ndimage.gaussian_filter(density, sigma=1.0)
            edges = density - blurred
            
            # Enhance edges
            enhanced = density + strength * edges
            enhanced = np.clip(enhanced, 0, 1)
            
            world_data["density_grid"] = enhanced
            
            # Enhance color details if available
            if color is not None:
                for c in range(color.shape[0]):
                    channel = color[c]
                    blurred_c = ndimage.gaussian_filter(channel, sigma=0.5)
                    edges_c = channel - blurred_c
                    enhanced_c = channel + strength * 0.5 * edges_c
                    color[c] = np.clip(enhanced_c, 0, 1)
                world_data["color_grid"] = color
        
        elif "vertices" in world_data:
            # Mesh edge enhancement by sharpening normals
            vertices = world_data["vertices"]
            triangles = world_data["triangles"]
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            
            # Compute vertex normals
            vertex_normals = mesh.vertex_normals
            
            # Sharpen normals (exaggerate differences)
            laplacian = self._mesh_laplacian(vertices, triangles)
            normal_laplacian = laplacian @ vertex_normals
            
            # Enhance normals
            enhanced_normals = vertex_normals + strength * 0.1 * normal_laplacian
            enhanced_normals = enhanced_normals / np.linalg.norm(enhanced_normals, axis=1, keepdims=True)
            
            world_data["vertex_normals"] = enhanced_normals
        
        return world_data
    
    def _mesh_laplacian(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute mesh Laplacian matrix"""
        n_vertices = len(vertices)
        laplacian = np.zeros((n_vertices, n_vertices))
        
        for face in faces:
            for i in range(3):
                v1 = face[i]
                v2 = face[(i+1)%3]
                v3 = face[(i+2)%3]
                
                # Compute cotangent weights
                vec1 = vertices[v2] - vertices[v1]
                vec2 = vertices[v3] - vertices[v1]
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                cot_weight = 1.0 / np.tan(angle)
                
                laplacian[v2, v3] += cot_weight
                laplacian[v3, v2] += cot_weight
        
        # Normalize
        row_sums = laplacian.sum(axis=1)
        laplacian = np.diag(1.0 / (row_sums + 1e-8)) @ laplacian
        
        return laplacian
    
    def color_correct(
        self,
        world_data: Dict[str, Any],
        preset: str = "cinematic",
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0
    ) -> Dict[str, Any]:
        """Apply color correction"""
        
        if not self.config.color_correction:
            return world_data
        
        if "color_grid" in world_data:
            color = world_data["color_grid"]
            
            # Convert to 0-255 for LUT application
            color_8bit = (color * 255).astype(np.uint8)
            
            # Apply LUT if preset specified
            if preset in self.color_luts:
                lut = self.color_luts[preset]
                
                # Apply LUT to each channel
                corrected = np.zeros_like(color_8bit)
                for c in range(3):
                    corrected[c] = lut[color_8bit[c], c] * 255
            
            else:
                corrected = color_8bit
            
            # Apply basic adjustments
            corrected = corrected.astype(np.float32) / 255.0
            
            # Brightness
            corrected = np.clip(corrected + brightness, 0, 1)
            
            # Contrast
            corrected = 0.5 + contrast * (corrected - 0.5)
            corrected = np.clip(corrected, 0, 1)
            
            # Saturation (convert to HSV, adjust, convert back)
            corrected_hsv = cv2.cvtColor(corrected.transpose(1, 2, 3, 0), cv2.COLOR_RGB2HSV)
            corrected_hsv[..., 1] = np.clip(corrected_hsv[..., 1] * saturation, 0, 1)
            corrected = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2RGB).transpose(3, 0, 1, 2)
            
            world_data["color_grid"] = corrected
        
        elif "vertex_colors" in world_data:
            colors = world_data["vertex_colors"]
            
            # Similar adjustments for vertex colors
            colors = np.clip(colors + brightness, 0, 1)
            colors = 0.5 + contrast * (colors - 0.5)
            colors = np.clip(colors, 0, 1)
            
            world_data["vertex_colors"] = colors
        
        return world_data
    
    def fill_holes(
        self,
        world_data: Dict[str, Any],
        max_hole_size: int = 100
    ) -> Dict[str, Any]:
        """Fill holes in 3D structures"""
        
        if not self.config.fill_holes:
            return world_data
        
        if "density_grid" in world_data:
            density = world_data["density_grid"]
            
            # Binarize
            binary = density > 0.5
            
            # Fill holes in 3D
            filled = ndimage.binary_fill_holes(binary)
            
            # Only fill small holes
            labeled, num_features = ndimage.label(~filled)
            component_sizes = ndimage.sum(~filled, labeled, range(1, num_features + 1))
            
            for i, size in enumerate(component_sizes):
                if size <= max_hole_size:
                    filled[labeled == (i + 1)] = True
            
            # Convert back to density
            hole_mask = filled & ~binary
            density[hole_mask] = 0.3  # Fill with low density
            
            world_data["density_grid"] = density
        
        elif "vertices" in world_data:
            # Mesh hole filling
            vertices = world_data["vertices"]
            triangles = world_data["triangles"]
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            
            # Fill holes
            mesh.fill_holes()
            
            world_data["vertices"] = mesh.vertices
            world_data["triangles"] = mesh.faces
        
        return world_data
    
    def remove_isolated_components(
        self,
        world_data: Dict[str, Any],
        min_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Remove isolated small components"""
        
        if not self.config.remove_isolated:
            return world_data
        
        min_size = min_size or self.config.min_component_size
        
        if "density_grid" in world_data:
            density = world_data["density_grid"]
            
            # Binarize
            binary = density > 0.5
            
            # Label connected components
            labeled, num_components = ndimage.label(binary)
            component_sizes = ndimage.sum(binary, labeled, range(1, num_components + 1))
            
            # Create mask for large enough components
            mask = np.zeros_like(binary, dtype=bool)
            for i, size in enumerate(component_sizes):
                if size >= min_size:
                    mask[labeled == (i + 1)] = True
            
            # Apply mask
            density[~mask] = 0.0
            
            world_data["density_grid"] = density
        
        return world_data
    
    def sharp_details(
        self,
        world_data: Dict[str, Any],
        amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """Sharpen fine details"""
        amount = amount or self.config.sharpen_amount
        
        if "density_grid" in world_data:
            density = world_data["density_grid"]
            
            # Unsharp masking
            blurred = ndimage.gaussian_filter(density, sigma=1.0)
            details = density - blurred
            
            # Enhance details
            sharpened = density + amount * details
            sharpened = np.clip(sharpened, 0, 1)
            
            world_data["density_grid"] = sharpened
        
        return world_data
    
    def tone_mapping(
        self,
        world_data: Dict[str, Any],
        method: str = "reinhard"
    ) -> Dict[str, Any]:
        """Apply tone mapping for better display"""
        
        if not self.config.tone_mapping:
            return world_data
        
        if "color_grid" in world_data:
            color = world_data["color_grid"]
            
            if method == "reinhard":
                # Reinhard tone mapping
                luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                max_lum = luminance.max()
                
                if max_lum > 0:
                    scaled_lum = luminance / (1 + luminance)
                    color = color * (scaled_lum / (luminance + 1e-8))[None, ...]
            
            elif method == "aces":
                # ACES filmic tone mapping
                color = 0.6 * color
                a = 2.51
                b = 0.03
                c = 2.43
                d = 0.59
                e = 0.14
                
                color = (color * (a * color + b)) / (color * (c * color + d) + e)
            
            color = np.clip(color, 0, 1)
            world_data["color_grid"] = color
        
        return world_data
    
    def optimize_mesh(
        self,
        world_data: Dict[str, Any],
        target_faces: Optional[int] = None,
        preserve_boundaries: bool = True
    ) -> Dict[str, Any]:
        """Optimize mesh by reducing polygon count"""
        
        if "vertices" not in world_data:
            return world_data
        
        vertices = world_data["vertices"]
        triangles = world_data["triangles"]
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        # Determine target face count
        if target_faces is None:
            current_faces = len(triangles)
            if current_faces > 100000:
                target_faces = current_faces // 4
            elif current_faces > 50000:
                target_faces = current_faces // 2
            else:
                # Don't simplify if already low poly
                return world_data
        
        # Simplify mesh
        simplified = mesh.simplify_quadric_decimation(target_faces)
        
        # Ensure normals are consistent
        simplified.fix_normals()
        
        world_data.update({
            "vertices": simplified.vertices,
            "triangles": simplified.faces,
            "vertex_normals": simplified.vertex_normals,
            "original_face_count": len(triangles),
            "optimized_face_count": len(simplified.faces),
            "reduction_ratio": len(simplified.faces) / len(triangles)
        })
        
        return world_data
    
    def export(
        self,
        world_data: Dict[str, Any],
        format: str,
        output_path: Path,
        **export_kwargs
    ) -> bool:
        """Export processed world to file"""
        
        try:
            if format == "glb" or format == "gltf":
                return self._export_gltf(world_data, output_path, **export_kwargs)
            elif format == "obj":
                return self._export_obj(world_data, output_path, **export_kwargs)
            elif format == "ply":
                return self._export_ply(world_data, output_path, **export_kwargs)
            elif format == "npz":
                return self._export_npz(world_data, output_path, **export_kwargs)
            elif format == "vdb":
                return self._export_vdb(world_data, output_path, **export_kwargs)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
        
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return False
    
    def _export_gltf(
        self,
        world_data: Dict[str, Any],
        output_path: Path,
        **kwargs
    ) -> bool:
        """Export as GLTF/GLB"""
        try:
            if "vertices" in world_data:
                vertices = world_data["vertices"]
                triangles = world_data["triangles"]
                vertex_colors = world_data.get("vertex_colors", None)
                
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=triangles,
                    vertex_colors=vertex_colors
                )
                
                mesh.export(output_path, file_type='glb')
                logger.info(f"Exported GLB to {output_path}")
                return True
            else:
                logger.warning("Cannot export to GLTF: No mesh data")
                return False
        
        except Exception as e:
            logger.error(f"GLTF export failed: {e}")
            return False
    
    def _export_obj(
        self,
        world_data: Dict[str, Any],
        output_path: Path,
        **kwargs
    ) -> bool:
        """Export as OBJ"""
        try:
            if "vertices" in world_data:
                vertices = world_data["vertices"]
                triangles = world_data["triangles"]
                vertex_normals = world_data.get("vertex_normals", None)
                vertex_colors = world_data.get("vertex_colors", None)
                
                # Write OBJ file
                with open(output_path, 'w') as f:
                    # Write vertices
                    for v in vertices:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    
                    # Write vertex normals if available
                    if vertex_normals is not None:
                        for vn in vertex_normals:
                            f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
                    
                    # Write faces
                    for face in triangles:
                        if vertex_normals is not None:
                            f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
                        else:
                            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                
                logger.info(f"Exported OBJ to {output_path}")
                return True
            else:
                logger.warning("Cannot export to OBJ: No mesh data")
                return False
        
        except Exception as e:
            logger.error(f"OBJ export failed: {e}")
            return False
    
    def _export_ply(
        self,
        world_data: Dict[str, Any],
        output_path: Path,
        **kwargs
    ) -> bool:
        """Export as PLY"""
        try:
            if "vertices" in world_data:
                vertices = world_data["vertices"]
                triangles = world_data["triangles"]
                vertex_colors = world_data.get("vertex_colors", None)
                
                mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=triangles,
                    vertex_colors=vertex_colors
                )
                
                mesh.export(output_path, file_type='ply')
                logger.info(f"Exported PLY to {output_path}")
                return True
            else:
                logger.warning("Cannot export to PLY: No mesh data")
                return False
        
        except Exception as e:
            logger.error(f"PLY export failed: {e}")
            return False
    
    def _export_npz(
        self,
        world_data: Dict[str, Any],
        output_path: Path,
        **kwargs
    ) -> bool:
        """Export as NPZ (NumPy compressed)"""
        try:
            # Remove large arrays if needed
            export_data = {}
            for key, value in world_data.items():
                if isinstance(value, np.ndarray) and value.nbytes > 1e9:  # > 1GB
                    logger.warning(f"Skipping large array: {key} ({value.nbytes/1e9:.2f} GB)")
                    continue
                export_data[key] = value
            
            np.savez_compressed(output_path, **export_data)
            logger.info(f"Exported NPZ to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"NPZ export failed: {e}")
            return False
    
    def _export_vdb(
        self,
        world_data: Dict[str, Any],
        output_path: Path,
        **kwargs
    ) -> bool:
        """Export as VDB (OpenVDB format)"""
        try:
            # This would require OpenVDB Python bindings
            # For now, just log that it's not implemented
            logger.warning("VDB export not implemented (requires OpenVDB)")
            return False
        
        except Exception as e:
            logger.error(f"VDB export failed: {e}")
            return False
    
    def create_preview(
        self,
        world_data: Dict[str, Any],
        camera_positions: List[Tuple[float, float, float]] = None,
        image_size: Tuple[int, int] = (512, 512),
        output_dir: Path = None
    ) -> List[Image.Image]:
        """Create 2D preview images of the 3D world"""
        
        previews = []
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default camera positions if none provided
        if camera_positions is None:
            # Front, back, left, right, top, bottom
            bounds = world_data.get("bounds", {
                "x_min": -5, "x_max": 5,
                "y_min": -5, "y_max": 5,
                "z_min": -5, "z_max": 5
            })
            
            center = np.array([
                (bounds["x_min"] + bounds["x_max"]) / 2,
                (bounds["y_min"] + bounds["y_max"]) / 2,
                (bounds["z_min"] + bounds["z_max"]) / 2
            ])
            
            radius = max(
                bounds["x_max"] - bounds["x_min"],
                bounds["y_max"] - bounds["y_min"],
                bounds["z_max"] - bounds["z_min"]
            ) / 2
            
            camera_positions = [
                center + np.array([radius * 2, 0, 0]),  # Front
                center + np.array([-radius * 2, 0, 0]), # Back
                center + np.array([0, radius * 2, 0]),  # Left
                center + np.array([0, -radius * 2, 0]), # Right
                center + np.array([0, 0, radius * 2]),  # Top
                center + np.array([0, 0, -radius * 2]), # Bottom
            ]
        
        # TODO: Implement actual rendering for different formats
        # For now, create placeholder previews
        
        for i, cam_pos in enumerate(camera_positions):
            # Create a simple preview image
            preview = self._create_placeholder_preview(world_data, cam_pos, image_size)
            previews.append(preview)
            
            if output_dir:
                preview_path = output_dir / f"preview_{i:03d}.png"
                preview.save(preview_path)
                logger.info(f"Saved preview to {preview_path}")
        
        return previews
    
    def _create_placeholder_preview(
        self,
        world_data: Dict[str, Any],
        camera_position: Tuple[float, float, float],
        image_size: Tuple[int, int]
    ) -> Image.Image:
        """Create a placeholder preview image"""
        width, height = image_size
        
        # Create a gradient background
        img = Image.new('RGB', image_size, (30, 30, 40))
        
        # Add some informative text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        info = [
            f"Format: {world_data.get('format', 'unknown')}",
            f"Resolution: {world_data.get('resolution', 'unknown')}",
            f"Camera: {camera_position}",
            "Preview rendering not implemented"
        ]
        
        y = 20
        for line in info:
            draw.text((20, y), line, fill=(255, 255, 255), font=font)
            y += 30
        
        return img
    
    def validate_world(
        self,
        world_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate world data for consistency and quality"""
        
        issues = []
        warnings = []
        
        # Check for required fields based on format
        format = world_data.get("format", "unknown")
        
        if format == "nerf":
            required = ["density_grid", "color_grid", "resolution"]
            for field in required:
                if field not in world_data:
                    issues.append(f"Missing required field: {field}")
        
        elif format == "mesh":
            required = ["vertices", "triangles"]
            for field in required:
                if field not in world_data:
                    issues.append(f"Missing required field: {field}")
            
            # Check mesh validity
            if "vertices" in world_data and "triangles" in world_data:
                vertices = world_data["vertices"]
                triangles = world_data["triangles"]
                
                if len(vertices) == 0:
                    issues.append("Mesh has no vertices")
                
                if len(triangles) == 0:
                    issues.append("Mesh has no faces")
                
                # Check for degenerate triangles
                if len(triangles) > 0:
                    import trimesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                    if not mesh.is_watertight:
                        warnings.append("Mesh is not watertight")
                    
                    if mesh.is_empty:
                        issues.append("Mesh is empty")
        
        # Check data ranges
        if "color_grid" in world_data:
            color = world_data["color_grid"]
            if color.min() < 0 or color.max() > 1:
                warnings.append(f"Color values out of range [0,1]: {color.min():.3f} - {color.max():.3f}")
        
        if "density_grid" in world_data:
            density = world_data["density_grid"]
            if density.min() < 0 or density.max() > 1:
                warnings.append(f"Density values out of range [0,1]: {density.min():.3f} - {density.max():.3f}")
        
        # Check for NaN or Inf values
        for key, value in world_data.items():
            if isinstance(value, np.ndarray):
                if np.any(np.isnan(value)):
                    issues.append(f"Array '{key}' contains NaN values")
                if np.any(np.isinf(value)):
                    issues.append(f"Array '{key}' contains Inf values")
        
        # Calculate quality metrics
        metrics = {}
        
        if "vertices" in world_data:
            vertices = world_data["vertices"]
            triangles = world_data["triangles"]
            
            metrics["vertex_count"] = len(vertices)
            metrics["face_count"] = len(triangles)
            
            # Calculate bounding box
            if len(vertices) > 0:
                bbox_min = vertices.min(axis=0)
                bbox_max = vertices.max(axis=0)
                bbox_size = bbox_max - bbox_min
                
                metrics["bbox_size"] = bbox_size.tolist()
                metrics["bbox_volume"] = np.prod(bbox_size)
        
        validation_result = {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "metrics": metrics,
            "format": format
        }
        
        return validation_result