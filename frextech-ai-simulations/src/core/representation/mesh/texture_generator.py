"""
Texture generation for meshes.
Includes UV unwrapping, texture synthesis, PBR material generation, and texture painting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union
import trimesh
import xatlas
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFilter
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors
import cv2
from dataclasses import dataclass
import json
import time
import warnings


@dataclass
class Texture:
    """Texture data structure."""
    image: np.ndarray  # [H, W, channels]
    uv_coords: np.ndarray  # [V, 2] texture coordinates
    uv_indices: np.ndarray  # [F, 3] indices into uv_coords
    material: Optional[Dict[str, Any]] = None  # PBR material parameters
    
    @property
    def height(self) -> int:
        return self.image.shape[0]
    
    @property
    def width(self) -> int:
        return self.image.shape[1]
    
    @property
    def channels(self) -> int:
        return self.image.shape[2]
    
    def to_pil(self) -> PIL.Image.Image:
        """Convert to PIL Image."""
        if self.image.dtype == np.float32:
            image_uint8 = (np.clip(self.image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = self.image
        
        return PIL.Image.fromarray(image_uint8)
    
    def save(self, path: str, format: str = 'png'):
        """Save texture to file."""
        pil_image = self.to_pil()
        pil_image.save(path, format=format.upper())
        
        # Save UV coordinates and material if needed
        if path.endswith('.json'):
            # Save metadata
            metadata = {
                'uv_coords': self.uv_coords.tolist(),
                'uv_indices': self.uv_indices.tolist(),
                'material': self.material
            }
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, image_path: str, metadata_path: Optional[str] = None) -> 'Texture':
        """Load texture from file."""
        # Load image
        pil_image = PIL.Image.open(image_path)
        image = np.array(pil_image)
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Load metadata if provided
        uv_coords = None
        uv_indices = None
        material = None
        
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            uv_coords = np.array(metadata.get('uv_coords', []))
            uv_indices = np.array(metadata.get('uv_indices', []))
            material = metadata.get('material')
        
        return cls(
            image=image,
            uv_coords=uv_coords,
            uv_indices=uv_indices,
            material=material
        )


@dataclass
class PBRMaterial:
    """PBR material parameters."""
    albedo: np.ndarray  # [3] or texture
    roughness: float  # 0.0 (smooth) to 1.0 (rough)
    metallic: float  # 0.0 (dielectric) to 1.0 (metal)
    normal: Optional[np.ndarray] = None  # [3] or normal map
    emission: Optional[np.ndarray] = None  # [3] emission color
    ao: Optional[float] = None  # ambient occlusion
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.0
    sheen: float = 0.0
    sheen_tint: float = 0.0
    subsurface: float = 0.0
    specular: float = 0.5
    specular_tint: float = 0.0
    anisotropic: float = 0.0
    ior: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'roughness': self.roughness,
            'metallic': self.metallic,
            'clearcoat': self.clearcoat,
            'clearcoat_roughness': self.clearcoat_roughness,
            'sheen': self.sheen,
            'sheen_tint': self.sheen_tint,
            'subsurface': self.subsurface,
            'specular': self.specular,
            'specular_tint': self.specular_tint,
            'anisotropic': self.anisotropic,
            'ior': self.ior
        }
        
        if isinstance(self.albedo, np.ndarray):
            result['albedo'] = self.albedo.tolist()
        else:
            result['albedo'] = self.albedo
        
        if self.normal is not None:
            if isinstance(self.normal, np.ndarray):
                result['normal'] = self.normal.tolist()
            else:
                result['normal'] = self.normal
        
        if self.emission is not None:
            result['emission'] = self.emission.tolist()
        
        if self.ao is not None:
            result['ao'] = self.ao
        
        return result


class UVUnwrapper:
    """UV unwrapping and parameterization."""
    
    def __init__(
        self,
        method: str = 'xatlas',  # 'xatlas', 'lscm', 'abf', 'simple'
        padding: int = 2,
        resolution: int = 1024,
        max_chart_area: float = 0.99,
        max_boundary_length: float = 0.25
    ):
        self.method = method
        self.padding = padding
        self.resolution = resolution
        self.max_chart_area = max_chart_area
        self.max_boundary_length = max_boundary_length
    
    def unwrap(
        self,
        mesh: Mesh,
        existing_uvs: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute UV coordinates for mesh.
        
        Args:
            mesh: Input mesh
            existing_uvs: Optional existing UVs (uv_coords, uv_indices)
            
        Returns:
            Tuple of (uv_coords, uv_indices)
        """
        if existing_uvs is not None:
            return existing_uvs
        
        if self.method == 'xatlas':
            return self._unwrap_xatlas(mesh)
        elif self.method == 'lscm':
            return self._unwrap_lscm(mesh)
        elif self.method == 'abf':
            return self._unwrap_abf(mesh)
        elif self.method == 'simple':
            return self._unwrap_simple(mesh)
        else:
            raise ValueError(f"Unknown unwrapping method: {self.method}")
    
    def _unwrap_xatlas(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
        """Unwrap using xatlas (high quality)."""
        import xatlas
        
        # Create xatlas atlas
        atlas = xatlas.Atlas()
        
        # Add mesh
        mesh_id = atlas.add_mesh(
            mesh.vertices.astype(np.float32),
            mesh.faces.astype(np.int32)
        )
        
        # Generate charts
        chart_options = xatlas.ChartOptions()
        chart_options.max_chart_area = self.max_chart_area
        chart_options.max_boundary_length = self.max_boundary_length
        
        atlas.generate(chart_options=chart_options)
        
        # Parameterize
        param_options = xatlas.PackOptions()
        param_options.padding = self.padding
        param_options.resolution = self.resolution
        
        atlas.pack(param_options=param_options)
        
        # Get results
        vmapping, indices, uvs = atlas[mesh_id]
        
        # Convert to appropriate format
        uv_coords = uvs.astype(np.float32)
        uv_indices = indices.astype(np.int32)
        
        return uv_coords, uv_indices
    
    def _unwrap_lscm(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
        """Least squares conformal maps."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        num_vertices = len(vertices)
        num_faces = len(faces)
        
        # Build cotangent Laplacian
        L = self._build_cotangent_laplacian(vertices, faces)
        
        # Select boundary vertices for pinning
        boundary = self._find_mesh_boundary(faces)
        
        if len(boundary) < 3:
            # Fallback to simple unwrapping
            return self._unwrap_simple(mesh)
        
        # Pin boundary vertices to circle
        boundary_uvs = self._map_boundary_to_circle(boundary, vertices)
        
        # Solve for interior vertices
        # Set up linear system: L * u = 0 with boundary constraints
        interior_mask = np.ones(num_vertices, dtype=bool)
        interior_mask[boundary] = False
        
        # Split Laplacian into interior and boundary parts
        L_ii = L[interior_mask, :][:, interior_mask]
        L_ib = L[interior_mask, :][:, boundary]
        
        # Right-hand side from boundary conditions
        b_u = -L_ib @ boundary_uvs[:, 0]
        b_v = -L_ib @ boundary_uvs[:, 1]
        
        # Solve for interior UVs
        u_interior = spsolve(L_ii, b_u)
        v_interior = spsolve(L_ii, b_v)
        
        # Combine interior and boundary UVs
        uv_coords = np.zeros((num_vertices, 2))
        uv_coords[interior_mask, 0] = u_interior
        uv_coords[interior_mask, 1] = v_interior
        uv_coords[boundary] = boundary_uvs
        
        # UV indices are same as face indices
        uv_indices = faces.copy()
        
        return uv_coords, uv_indices
    
    def _unwrap_abf(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
        """Angle-based flattening (simplified)."""
        # Simplified ABF implementation
        # Full ABF is complex, so we use a simplified version
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Get mesh boundary
        boundary = self._find_mesh_boundary(faces)
        
        if len(boundary) < 3:
            return self._unwrap_simple(mesh)
        
        # Map boundary to circle
        boundary_uvs = self._map_boundary_to_circle(boundary, vertices)
        
        # For interior vertices, use barycentric mapping from one triangle
        # This is very simplified - real ABF optimizes angles
        
        num_vertices = len(vertices)
        uv_coords = np.zeros((num_vertices, 2))
        uv_coords[boundary] = boundary_uvs
        
        # Use harmonic mapping for interior
        interior_mask = np.ones(num_vertices, dtype=bool)
        interior_mask[boundary] = False
        
        if np.any(interior_mask):
            # Build graph Laplacian (simpler than cotangent)
            L = self._build_graph_laplacian(faces, num_vertices)
            
            L_ii = L[interior_mask, :][:, interior_mask]
            L_ib = L[interior_mask, :][:, boundary]
            
            b_u = -L_ib @ boundary_uvs[:, 0]
            b_v = -L_ib @ boundary_uvs[:, 1]
            
            u_interior = spsolve(L_ii, b_u)
            v_interior = spsolve(L_ii, b_v)
            
            uv_coords[interior_mask, 0] = u_interior
            uv_coords[interior_mask, 1] = v_interior
        
        uv_indices = faces.copy()
        
        return uv_coords, uv_indices
    
    def _unwrap_simple(self, mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
        """Simple cylindrical or spherical projection."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Compute centroid and normalize
        centroid = vertices.mean(axis=0)
        vertices_centered = vertices - centroid
        
        # Spherical coordinates
        r = np.linalg.norm(vertices_centered, axis=1)
        theta = np.arctan2(vertices_centered[:, 1], vertices_centered[:, 0])  # azimuth
        phi = np.arccos(vertices_centered[:, 2] / (r + 1e-8))  # inclination
        
        # Map to UV: theta -> u, phi -> v
        u = (theta + np.pi) / (2 * np.pi)  # [0, 1]
        v = phi / np.pi  # [0, 1]
        
        uv_coords = np.stack([u, v], axis=1)
        uv_indices = faces.copy()
        
        return uv_coords, uv_indices
    
    def _build_cotangent_laplacian(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> sparse.csr_matrix:
        """Build cotangent Laplacian matrix."""
        num_vertices = len(vertices)
        
        I, J, V = [], [], []
        
        for face in faces:
            v0, v1, v2 = face
            p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
            
            # Compute cotangents
            cot0 = self._cotangent(p0, p1, p2)
            cot1 = self._cotangent(p1, p2, p0)
            cot2 = self._cotangent(p2, p0, p1)
            
            # Add contributions
            for (i, j, cot) in [(v1, v2, cot0), (v2, v0, cot1), (v0, v1, cot2)]:
                I.append(i)
                J.append(j)
                V.append(cot)
            
            # Diagonal placeholder
            for v in [v0, v1, v2]:
                I.append(v)
                J.append(v)
                V.append(0.0)
        
        L = sparse.csr_matrix((V, (I, J)), shape=(num_vertices, num_vertices))
        diag = -np.array(L.sum(axis=1)).flatten()
        L.setdiag(diag)
        
        return L
    
    def _build_graph_laplacian(
        self,
        faces: np.ndarray,
        num_vertices: int
    ) -> sparse.csr_matrix:
        """Build graph Laplacian (binary adjacency)."""
        I, J = [], []
        
        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        I.append(face[i])
                        J.append(face[j])
        
        data = np.ones(len(I))
        A = sparse.csr_matrix((data, (I, J)), shape=(num_vertices, num_vertices))
        
        D = sparse.diags(np.array(A.sum(axis=1)).flatten())
        L = D - A
        
        return L
    
    def _cotangent(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
    ) -> float:
        """Compute cotangent of angle at vertex a."""
        v0 = b - a
        v1 = c - a
        
        dot = np.dot(v0, v1)
        cross = np.linalg.norm(np.cross(v0, v1))
        
        return dot / (cross + 1e-8)
    
    def _find_mesh_boundary(self, faces: np.ndarray) -> np.ndarray:
        """Find boundary vertices of mesh."""
        # Count edge occurrences
        from collections import defaultdict
        
        edge_count = defaultdict(int)
        
        for face in faces:
            for i in range(3):
                v0 = face[i]
                v1 = face[(i + 1) % 3]
                edge = (min(v0, v1), max(v0, v1))
                edge_count[edge] += 1
        
        # Find boundary edges (count == 1)
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        if not boundary_edges:
            return np.array([], dtype=int)
        
        # Extract boundary vertices in order
        boundary_vertices = []
        current_edge = boundary_edges[0]
        boundary_edges_set = set(boundary_edges)
        
        while boundary_edges_set:
            boundary_vertices.append(current_edge[0])
            boundary_vertices.append(current_edge[1])
            boundary_edges_set.remove(current_edge)
            
            # Find next edge
            next_edge = None
            for edge in boundary_edges_set:
                if edge[0] == current_edge[1] or edge[1] == current_edge[1]:
                    next_edge = edge
                    break
            
            if next_edge is None:
                break
            
            current_edge = next_edge
        
        return np.unique(boundary_vertices)
    
    def _map_boundary_to_circle(
        self,
        boundary: np.ndarray,
        vertices: np.ndarray
    ) -> np.ndarray:
        """Map boundary vertices to unit circle."""
        # Sort boundary vertices along boundary
        # This is simplified - assumes boundary is already sorted
        
        num_boundary = len(boundary)
        angles = np.linspace(0, 2 * np.pi, num_boundary, endpoint=False)
        
        u = 0.5 + 0.5 * np.cos(angles)  # Map to [0, 1]
        v = 0.5 + 0.5 * np.sin(angles)
        
        return np.stack([u, v], axis=1)


class TextureSynthesizer:
    """Texture synthesis and generation."""
    
    def __init__(
        self,
        method: str = 'procedural',  # 'procedural', 'neural', 'exemplar', 'projection'
        resolution: int = 1024,
        num_channels: int = 3
    ):
        self.method = method
        self.resolution = resolution
        self.num_channels = num_channels
        
        # Neural network for texture synthesis
        if method == 'neural':
            self.generator = self._build_texture_generator()
    
    def _build_texture_generator(self) -> nn.Module:
        """Build neural texture generator."""
        class TextureGenerator(nn.Module):
            def __init__(self, latent_dim=256, num_channels=3):
                super().__init__()
                self.latent_dim = latent_dim
                self.num_channels = num_channels
                
                # Generator network
                self.net = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 4096),
                    nn.ReLU(),
                    nn.Linear(4096, 1024 * 1024 * num_channels // 256),  # Adjust for resolution
                    nn.Tanh()
                )
            
            def forward(self, z):
                batch_size = z.shape[0]
                out = self.net(z)
                # Reshape to image
                size = int(np.sqrt(out.shape[1] // self.num_channels))
                out = out.view(batch_size, self.num_channels, size, size)
                return out
        
        return TextureGenerator(num_channels=self.num_channels)
    
    def generate_procedural(
        self,
        pattern: str = 'noise',
        colors: Optional[List[Tuple[float, float, float]]] = None,
        scale: float = 1.0,
        **kwargs
    ) -> np.ndarray:
        """
        Generate procedural texture.
        
        Args:
            pattern: Pattern type ('noise', 'checker', 'gradient', 'wood', 'marble', 'clouds')
            colors: List of colors for pattern
            scale: Scale of pattern
            **kwargs: Pattern-specific parameters
            
        Returns:
            Texture image [H, W, C]
        """
        H = W = self.resolution
        
        if colors is None:
            if pattern == 'checker':
                colors = [(0.9, 0.9, 0.9), (0.1, 0.1, 0.1)]
            else:
                colors = [(0.5, 0.5, 0.8), (0.8, 0.5, 0.5)]
        
        # Create coordinate grid
        y, x = np.meshgrid(
            np.linspace(0, 1, H),
            np.linspace(0, 1, W),
            indexing='ij'
        )
        
        if pattern == 'noise':
            # Perlin-like noise
            texture = self._generate_perlin_noise(H, W, scale)
            # Convert to RGB
            texture = np.stack([texture] * 3, axis=-1)
            
        elif pattern == 'checker':
            # Checkerboard
            freq = kwargs.get('frequency', 8.0)
            u = (x * freq * scale).astype(int)
            v = (y * freq * scale).astype(int)
            mask = (u + v) % 2
            
            texture = np.zeros((H, W, 3))
            for i, color in enumerate(colors):
                texture[mask == i] = color
            
        elif pattern == 'gradient':
            # Gradient
            if kwargs.get('radial', False):
                # Radial gradient
                center_x = kwargs.get('center_x', 0.5)
                center_y = kwargs.get('center_y', 0.5)
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                t = np.clip(dist * scale * 2, 0, 1)
            else:
                # Linear gradient
                angle = kwargs.get('angle', 45.0) * np.pi / 180.0
                t = (x * np.cos(angle) + y * np.sin(angle)) * scale
                t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            
            # Interpolate between colors
            if len(colors) == 2:
                texture = np.zeros((H, W, 3))
                for c in range(3):
                    texture[:, :, c] = colors[0][c] * (1 - t) + colors[1][c] * t
            else:
                # Multi-color gradient
                texture = self._multi_color_gradient(t, colors)
        
        elif pattern == 'wood':
            # Wood grain
            angle = kwargs.get('angle', 45.0) * np.pi / 180.0
            rings = kwargs.get('rings', 20.0)
            
            # Rotated coordinates
            x_rot = x * np.cos(angle) + y * np.sin(angle)
            y_rot = -x * np.sin(angle) + y * np.cos(angle)
            
            # Wood pattern
            r = np.sqrt(x_rot**2 + y_rot**2)
            grain = np.sin(r * rings * scale * 2 * np.pi)
            
            # Add some noise
            noise = np.random.randn(H, W) * 0.1
            grain = grain + noise
            
            # Color
            dark_color = np.array(colors[0])
            light_color = np.array(colors[1]) if len(colors) > 1 else np.array([0.8, 0.6, 0.4])
            
            t = (grain - grain.min()) / (grain.max() - grain.min() + 1e-8)
            texture = dark_color * (1 - t) + light_color * t
        
        elif pattern == 'marble':
            # Marble
            turbulence = kwargs.get('turbulence', 3.0)
            
            # Base noise
            noise = self._generate_perlin_noise(H, W, scale * 2)
            
            # Marble pattern
            x_scaled = x * scale
            pattern = x_scaled + turbulence * noise
            pattern = np.sin(pattern * 2 * np.pi)
            
            # Color
            t = (pattern - pattern.min()) / (pattern.max() - pattern.min() + 1e-8)
            
            if len(colors) >= 2:
                texture = np.zeros((H, W, 3))
                for c in range(3):
                    texture[:, :, c] = colors[0][c] * (1 - t) + colors[1][c] * t
            else:
                # Default marble colors
                base = np.array([0.8, 0.8, 0.8])
                vein = np.array([0.4, 0.4, 0.5])
                texture = base * (1 - t) + vein * t
        
        elif pattern == 'clouds':
            # Cloud-like texture
            octaves = kwargs.get('octaves', 4)
            persistence = kwargs.get('persistence', 0.5)
            
            texture = np.zeros((H, W))
            amplitude = 1.0
            frequency = scale
            
            for _ in range(octaves):
                noise = self._generate_perlin_noise(H, W, frequency)
                texture += amplitude * noise
                amplitude *= persistence
                frequency *= 2
            
            # Normalize
            texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
            
            # Cloud colors
            texture = np.stack([texture] * 3, axis=-1)
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        
        return np.clip(texture, 0, 1)
    
    def _generate_perlin_noise(self, H: int, W: int, scale: float) -> np.ndarray:
        """Generate Perlin-like noise."""
        # Simplified Perlin noise
        # For production, use a proper Perlin/Simplex noise implementation
        
        # Create grid of random gradients
        grid_size = max(2, int(min(H, W) / (scale * 10)))
        grad_y, grad_x = np.meshgrid(
            np.linspace(0, H, grid_size),
            np.linspace(0, W, grid_size),
            indexing='ij'
        )
        
        # Random gradients
        angles = np.random.rand(grid_size, grid_size) * 2 * np.pi
        grad = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        
        # Interpolate to full resolution
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        x_coords = np.linspace(0, H, grid_size)
        y_coords = np.linspace(0, W, grid_size)
        
        # Interpolate gradient components
        interpolators = []
        for c in range(2):
            interp = RegularGridInterpolator(
                (x_coords, y_coords),
                grad[:, :, c],
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
            interpolators.append(interp)
        
        # Sample at all positions
        y_idx, x_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        coords = np.stack([y_idx.flatten(), x_idx.flatten()], axis=-1)
        
        noise = np.zeros(H * W)
        for interp in interpolators:
            noise += interp(coords)
        
        noise = noise.reshape(H, W) / np.sqrt(2)  # Normalize
        
        return noise
    
    def _multi_color_gradient(
        self,
        t: np.ndarray,
        colors: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """Create multi-color gradient."""
        H, W = t.shape
        texture = np.zeros((H, W, 3))
        
        # Normalize t to [0, n_colors-1]
        t_norm = t * (len(colors) - 1)
        
        for c in range(3):
            # Piecewise linear interpolation
            for i in range(len(colors) - 1):
                mask = (t_norm >= i) & (t_norm < i + 1)
                if np.any(mask):
                    local_t = t_norm[mask] - i
                    texture[mask, c] = (
                        colors[i][c] * (1 - local_t) + 
                        colors[i + 1][c] * local_t
                    )
        
        return texture
    
    def generate_from_exemplar(
        self,
        exemplar: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
        patch_size: int = 5,
        overlap: int = 1
    ) -> np.ndarray:
        """
        Generate texture from exemplar using texture synthesis.
        
        Args:
            exemplar: Exemplar texture image
            output_size: Output texture size (H, W)
            patch_size: Size of patches to copy
            overlap: Overlap between patches
            
        Returns:
            Synthesized texture
        """
        if output_size is None:
            output_size = (self.resolution, self.resolution)
        
        H, W = output_size
        exemplar_h, exemplar_w = exemplar.shape[:2]
        
        # Simple texture synthesis: tile the exemplar
        # For production, use more advanced methods like patch-based synthesis
        
        # Tile exemplar
        tiles_y = int(np.ceil(H / exemplar_h))
        tiles_x = int(np.ceil(W / exemplar_w))
        
        texture = np.zeros((H, W, exemplar.shape[2]))
        
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                y_start = ty * exemplar_h
                y_end = min((ty + 1) * exemplar_h, H)
                x_start = tx * exemplar_w
                x_end = min((tx + 1) * exemplar_w, W)
                
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                
                texture[y_start:y_end, x_start:x_end] = exemplar[:tile_h, :tile_w]
        
        return texture
    
    def generate_neural(
        self,
        latent: Optional[np.ndarray] = None,
        style: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate texture using neural network.
        
        Args:
            latent: Latent vector (optional)
            style: Style description (optional)
            
        Returns:
            Generated texture
        """
        if self.generator is None:
            raise ValueError("Neural generator not initialized")
        
        if latent is None:
            # Generate random latent
            latent = np.random.randn(1, self.generator.latent_dim).astype(np.float32)
        
        # Convert to tensor
        latent_tensor = torch.from_numpy(latent).float()
        
        # Generate texture
        with torch.no_grad():
            texture_tensor = self.generator(latent_tensor)
            texture = texture_tensor[0].permute(1, 2, 0).cpu().numpy()
        
        # Resize to target resolution
        if texture.shape[0] != self.resolution or texture.shape[1] != self.resolution:
            texture = cv2.resize(
                texture,
                (self.resolution, self.resolution),
                interpolation=cv2.INTER_LINEAR
            )
        
        return np.clip(texture, -1, 1) * 0.5 + 0.5  # Map from [-1, 1] to [0, 1]


class PBRGenerator:
    """PBR material generation."""
    
    def __init__(self):
        self.material_library = self._load_material_library()
    
    def _load_material_library(self) -> Dict[str, PBRMaterial]:
        """Load predefined material library."""
        library = {
            'plastic': PBRMaterial(
                albedo=np.array([0.8, 0.8, 0.8]),
                roughness=0.4,
                metallic=0.0,
                specular=0.5
            ),
            'metal': PBRMaterial(
                albedo=np.array([0.8, 0.8, 0.8]),
                roughness=0.1,
                metallic=1.0,
                specular=0.5
            ),
            'wood': PBRMaterial(
                albedo=np.array([0.4, 0.3, 0.2]),
                roughness=0.7,
                metallic=0.0,
                specular=0.2
            ),
            'ceramic': PBRMaterial(
                albedo=np.array([0.9, 0.9, 0.9]),
                roughness=0.1,
                metallic=0.0,
                specular=0.9
            ),
            'rubber': PBRMaterial(
                albedo=np.array([0.1, 0.1, 0.1]),
                roughness=0.9,
                metallic=0.0,
                specular=0.1
            ),
            'gold': PBRMaterial(
                albedo=np.array([1.0, 0.8, 0.2]),
                roughness=0.2,
                metallic=1.0,
                specular=0.5
            ),
            'copper': PBRMaterial(
                albedo=np.array([0.8, 0.4, 0.2]),
                roughness=0.3,
                metallic=1.0,
                specular=0.5
            ),
            'glass': PBRMaterial(
                albedo=np.array([0.9, 0.9, 0.9]),
                roughness=0.0,
                metallic=0.0,
                specular=1.0,
                ior=1.5
            ),
            'water': PBRMaterial(
                albedo=np.array([0.2, 0.4, 0.8]),
                roughness=0.0,
                metallic=0.0,
                specular=1.0,
                ior=1.33
            )
        }
        
        return library
    
    def generate_material(
        self,
        material_type: str = 'plastic',
        albedo: Optional[np.ndarray] = None,
        roughness: Optional[float] = None,
        metallic: Optional[float] = None,
        **kwargs
    ) -> PBRMaterial:
        """
        Generate PBR material.
        
        Args:
            material_type: Type of material
            albedo: Base color (optional)
            roughness: Roughness (optional)
            metallic: Metallic (optional)
            **kwargs: Additional material parameters
            
        Returns:
            PBR material
        """
        # Get base material from library
        if material_type in self.material_library:
            material = self.material_library[material_type]
        else:
            # Default material
            material = PBRMaterial(
                albedo=np.array([0.8, 0.8, 0.8]),
                roughness=0.5,
                metallic=0.0
            )
        
        # Override parameters if provided
        if albedo is not None:
            material.albedo = np.array(albedo)
        
        if roughness is not None:
            material.roughness = roughness
        
        if metallic is not None:
            material.metallic = metallic
        
        # Set additional parameters
        for key, value in kwargs.items():
            if hasattr(material, key):
                setattr(material, key, value)
        
        return material
    
    def generate_texture_maps(
        self,
        material: PBRMaterial,
        resolution: int = 1024,
        generate_normal: bool = True,
        generate_ao: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate texture maps from PBR material.
        
        Args:
            material: PBR material
            resolution: Texture resolution
            generate_normal: Whether to generate normal map
            generate_ao: Whether to generate AO map
            
        Returns:
            Dictionary of texture maps
        """
        maps = {}
        
        # Albedo map
        if isinstance(material.albedo, np.ndarray) and material.albedo.size == 3:
            # Solid color
            albedo_map = np.ones((resolution, resolution, 3)) * material.albedo
            maps['albedo'] = albedo_map
        else:
            # Assume it's already a texture
            maps['albedo'] = material.albedo
        
        # Roughness map
        roughness_map = np.ones((resolution, resolution, 1)) * material.roughness
        maps['roughness'] = roughness_map
        
        # Metallic map
        metallic_map = np.ones((resolution, resolution, 1)) * material.metallic
        maps['metallic'] = metallic_map
        
        # Normal map
        if generate_normal and material.normal is None:
            # Generate simple normal map (flat)
            normal_map = np.zeros((resolution, resolution, 3))
            normal_map[:, :, 2] = 1.0  # Default normal (0, 0, 1)
            maps['normal'] = normal_map
        elif material.normal is not None:
            maps['normal'] = material.normal
        
        # AO map
        if generate_ao:
            ao_map = np.ones((resolution, resolution, 1))
            maps['ao'] = ao_map
        
        # Emission map
        if material.emission is not None:
            emission_map = np.ones((resolution, resolution, 3)) * material.emission
            maps['emission'] = emission_map
        
        return maps


class TexturePainter:
    """Interactive texture painting."""
    
    def __init__(self, canvas_size: Tuple[int, int] = (1024, 1024)):
        self.canvas_size = canvas_size
        self.reset_canvas()
    
    def reset_canvas(self):
        """Reset painting canvas."""
        self.canvas = np.ones((*self.canvas_size, 3), dtype=np.float32)
        self.brush_size = 20
        self.brush_color = np.array([0.0, 0.0, 0.0])
        self.brush_hardness = 0.5
        self.stroke_history = []
    
    def paint_stroke(
        self,
        points: List[Tuple[int, int]],
        color: Optional[np.ndarray] = None,
        size: Optional[int] = None,
        hardness: Optional[float] = None
    ):
        """
        Paint a stroke on the canvas.
        
        Args:
            points: List of (x, y) points along stroke
            color: Stroke color
            size: Brush size
            hardness: Brush hardness (0.0 to 1.0)
        """
        if color is not None:
            self.brush_color = np.array(color)
        if size is not None:
            self.brush_size = size
        if hardness is not None:
            self.brush_hardness = hardness
        
        if len(points) < 2:
            # Single point
            self._paint_point(points[0])
        else:
            # Stroke with interpolation
            for i in range(len(points) - 1):
                self._paint_line(points[i], points[i + 1])
        
        # Record stroke
        self.stroke_history.append({
            'points': points,
            'color': self.brush_color.copy(),
            'size': self.brush_size,
            'hardness': self.brush_hardness
        })
    
    def _paint_point(self, point: Tuple[int, int]):
        """Paint a single point."""
        x, y = point
        size = self.brush_size
        
        # Create brush kernel
        kernel_size = size * 2 + 1
        y_idx, x_idx = np.ogrid[-size:size+1, -size:size+1]
        dist = np.sqrt(x_idx**2 + y_idx**2)
        
        # Hardness-based falloff
        hardness = self.brush_hardness
        inner_radius = size * hardness
        outer_radius = size
        
        mask = dist <= inner_radius
        falloff = np.where(
            dist <= inner_radius,
            1.0,
            np.clip(1 - (dist - inner_radius) / (outer_radius - inner_radius), 0, 1)
        )
        
        # Apply brush
        y_start = max(0, y - size)
        y_end = min(self.canvas_size[0], y + size + 1)
        x_start = max(0, x - size)
        x_end = min(self.canvas_size[1], x + size + 1)
        
        kernel_h = y_end - y_start
        kernel_w = x_end - x_start
        
        kernel_y_start = size - (y - y_start)
        kernel_x_start = size - (x - x_start)
        
        kernel_slice = (
            slice(kernel_y_start, kernel_y_start + kernel_h),
            slice(kernel_x_start, kernel_x_start + kernel_w)
        )
        
        brush = falloff[kernel_slice][:, :, np.newaxis]
        
        # Blend
        self.canvas[y_start:y_end, x_start:x_end] = (
            self.canvas[y_start:y_end, x_start:x_end] * (1 - brush) + 
            self.brush_color * brush
        )
    
    def _paint_line(self, p1: Tuple[int, int], p2: Tuple[int, int]):
        """Paint a line between two points."""
        x1, y1 = p1
        x2, y2 = p2
        
        # Number of interpolation steps
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        steps = max(2, int(length))
        
        for i in range(steps):
            t = i / (steps - 1)
            x = int(x1 * (1 - t) + x2 * t)
            y = int(y1 * (1 - t) + y2 * t)
            self._paint_point((x, y))
    
    def apply_filter(
        self,
        filter_type: str = 'blur',
        radius: float = 5.0,
        intensity: float = 1.0
    ):
        """
        Apply filter to canvas.
        
        Args:
            filter_type: Type of filter ('blur', 'sharpen', 'noise', 'emboss')
            radius: Filter radius
            intensity: Filter intensity
        """
        if filter_type == 'blur':
            # Gaussian blur
            from scipy.ndimage import gaussian_filter
            self.canvas = gaussian_filter(self.canvas, sigma=radius, mode='reflect')
        
        elif filter_type == 'sharpen':
            # Sharpen
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(self.canvas, sigma=radius, mode='reflect')
            self.canvas = self.canvas + intensity * (self.canvas - blurred)
            self.canvas = np.clip(self.canvas, 0, 1)
        
        elif filter_type == 'noise':
            # Add noise
            noise = np.random.randn(*self.canvas.shape) * intensity * 0.1
            self.canvas = np.clip(self.canvas + noise, 0, 1)
        
        elif filter_type == 'emboss':
            # Emboss filter
            kernel = np.array([
                [-2, -1, 0],
                [-1, 1, 1],
                [0, 1, 2]
            ]) * intensity
            
            from scipy.ndimage import convolve
            for c in range(3):
                self.canvas[:, :, c] = convolve(self.canvas[:, :, c], kernel)
            
            self.canvas = np.clip(self.canvas * 0.5 + 0.5, 0, 1)
    
    def undo(self) -> bool:
        """Undo last stroke."""
        if not self.stroke_history:
            return False
        
        # Remove last stroke
        self.stroke_history.pop()
        
        # Reset canvas and replay history
        self.canvas = np.ones((*self.canvas_size, 3), dtype=np.float32)
        for stroke in self.stroke_history:
            self.brush_color = stroke['color']
            self.brush_size = stroke['size']
            self.brush_hardness = stroke['hardness']
            self.paint_stroke(stroke['points'])
        
        return True
    
    def get_texture(self) -> np.ndarray:
        """Get current texture."""
        return self.canvas.copy()


class TextureGenerator:
    """
    Unified texture generation interface.
    
    Provides:
    - UV unwrapping
    - Texture synthesis
    - PBR material generation
    - Texture painting
    - Texture projection
    """
    
    def __init__(
        self,
        texture_resolution: int = 1024,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.texture_resolution = texture_resolution
        self.device = device
        
        # Initialize components
        self.unwrapper = UVUnwrapper(resolution=texture_resolution)
        self.synthesizer = TextureSynthesizer(resolution=texture_resolution)
        self.pbr_generator = PBRGenerator()
        self.painter = TexturePainter(canvas_size=(texture_resolution, texture_resolution))
        
        # Cache for generated textures
        self.texture_cache = {}
    
    def generate_texture_for_mesh(
        self,
        mesh: Mesh,
        texture_type: str = 'procedural',
        pattern: str = 'noise',
        colors: Optional[List[Tuple[float, float, float]]] = None,
        material_type: str = 'plastic',
        existing_uvs: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Texture:
        """
        Generate complete texture for mesh.
        
        Args:
            mesh: Input mesh
            texture_type: Type of texture ('procedural', 'neural', 'exemplar', 'painted')
            pattern: Pattern for procedural texture
            colors: Colors for texture
            material_type: PBR material type
            existing_uvs: Optional existing UV coordinates
            
        Returns:
            Complete texture with UVs
        """
        cache_key = (
            texture_type,
            pattern,
            str(colors),
            material_type,
            hash(mesh.vertices.tobytes()) if existing_uvs is None else hash(existing_uvs[0].tobytes())
        )
        
        if cache_key in self.texture_cache:
            return self.texture_cache[cache_key]
        
        # Generate UV coordinates
        uv_coords, uv_indices = self.unwrapper.unwrap(mesh, existing_uvs)
        
        # Generate texture image
        if texture_type == 'procedural':
            image = self.synthesizer.generate_procedural(pattern, colors)
        elif texture_type == 'neural':
            image = self.synthesizer.generate_neural()
        elif texture_type == 'exemplar':
            # Would need exemplar image
            image = self.synthesizer.generate_procedural('noise')
        elif texture_type == 'painted':
            image = self.painter.get_texture()
        else:
            raise ValueError(f"Unknown texture type: {texture_type}")
        
        # Generate PBR material
        material = self.pbr_generator.generate_material(material_type)
        
        # Create texture
        texture = Texture(
            image=image,
            uv_coords=uv_coords,
            uv_indices=uv_indices,
            material=material.to_dict()
        )
        
        self.texture_cache[cache_key] = texture
        return texture
    
    def generate_pbr_texture_set(
        self,
        mesh: Mesh,
        material: Union[str, PBRMaterial] = 'plastic',
        generate_maps: List[str] = None
    ) -> Dict[str, Texture]:
        """
        Generate full PBR texture set for mesh.
        
        Args:
            mesh: Input mesh
            material: Material type or PBRMaterial object
            generate_maps: Which maps to generate ['albedo', 'roughness', 'metallic', 'normal', 'ao']
            
        Returns:
            Dictionary of textures for each map
        """
        if generate_maps is None:
            generate_maps = ['albedo', 'roughness', 'metallic', 'normal', 'ao']
        
        # Get or create material
        if isinstance(material, str):
            material_obj = self.pbr_generator.generate_material(material)
        else:
            material_obj = material
        
        # Generate UVs
        uv_coords, uv_indices = self.unwrapper.unwrap(mesh)
        
        # Generate texture maps
        maps = self.pbr_generator.generate_texture_maps(
            material_obj,
            resolution=self.texture_resolution,
            generate_normal='normal' in generate_maps,
            generate_ao='ao' in generate_maps
        )
        
        # Create textures
        textures = {}
        for map_name, map_image in maps.items():
            if map_name in generate_maps:
                textures[map_name] = Texture(
                    image=map_image,
                    uv_coords=uv_coords,
                    uv_indices=uv_indices,
                    material=None  # Material stored in main texture
                )
        
        return textures
    
    def project_image_to_mesh(
        self,
        mesh: Mesh,
        image: np.ndarray,
        camera_position: np.ndarray,
        camera_direction: np.ndarray,
        fov: float = 60.0
    ) -> Texture:
        """
        Project image onto mesh from camera viewpoint.
        
        Args:
            mesh: Target mesh
            image: Source image
            camera_position: Camera position in world space
            camera_direction: Camera look direction
            fov: Field of view in degrees
            
        Returns:
            Projected texture
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Generate UVs if not present
        uv_coords, uv_indices = self.unwrapper.unwrap(mesh)
        
        # Project vertices to image plane
        # Simplified projection - assumes perspective camera
        
        # Normalize camera direction
        camera_direction = camera_direction / np.linalg.norm(camera_direction)
        
        # Create camera coordinate system
        up = np.array([0, 1, 0])  # World up
        right = np.cross(camera_direction, up)
        up = np.cross(right, camera_direction)
        
        right = right / np.linalg.norm(right)
        up = up / np.linalg.norm(up)
        
        # Project vertices
        image_h, image_w = image.shape[:2]
        fov_rad = np.radians(fov)
        
        projected_uvs = np.zeros((len(vertices), 2))
        
        for i, vertex in enumerate(vertices):
            # Vector from camera to vertex
            v = vertex - camera_position
            
            # Project onto camera plane
            x = np.dot(v, right)
            y = np.dot(v, up)
            z = np.dot(v, camera_direction)
            
            if z <= 0:
                # Behind camera
                projected_uvs[i] = [-1, -1]
                continue
            
            # Perspective projection
            scale = 1.0 / (z * np.tan(fov_rad / 2))
            screen_x = x * scale + 0.5
            screen_y = y * scale + 0.5
            
            # Convert to image coordinates
            img_x = int(screen_x * image_w)
            img_y = int(screen_y * image_h)
            
            # Clamp to image bounds
            img_x = max(0, min(image_w - 1, img_x))
            img_y = max(0, min(image_h - 1, img_y))
            
            # Store as UV
            projected_uvs[i, 0] = img_x / image_w
            projected_uvs[i, 1] = img_y / image_h
        
        # Create new texture by sampling from image
        texture_image = np.ones((self.texture_resolution, self.texture_resolution, 3))
        
        # For each texel, find corresponding vertex and sample from image
        # This is simplified - proper implementation would use rasterization
        
        # For now, just use the original image
        texture_image = cv2.resize(
            image,
            (self.texture_resolution, self.texture_resolution),
            interpolation=cv2.INTER_LINEAR
        )
        
        if texture_image.dtype == np.uint8:
            texture_image = texture_image.astype(np.float32) / 255.0
        
        texture = Texture(
            image=texture_image,
            uv_coords=projected_uvs,
            uv_indices=faces,
            material=None
        )
        
        return texture
    
    def bake_textures(
        self,
        highpoly_mesh: Mesh,
        lowpoly_mesh: Mesh,
        texture_size: int = 1024,
        bake_types: List[str] = None
    ) -> Dict[str, Texture]:
        """
        Bake textures from high-poly to low-poly mesh.
        
        Args:
            highpoly_mesh: High-polygon mesh (detail)
            lowpoly_mesh: Low-polygon mesh (base)
            texture_size: Size of baked textures
            bake_types: Types of textures to bake ['normal', 'ao', 'curvature', 'position']
            
        Returns:
            Dictionary of baked textures
        """
        if bake_types is None:
            bake_types = ['normal', 'ao']
        
        # This is a simplified implementation
        # Full baking requires ray casting and proper sampling
        
        # Generate UVs for low-poly mesh
        uv_coords, uv_indices = self.unwrapper.unwrap(lowpoly_mesh)
        
        baked_textures = {}
        
        for bake_type in bake_types:
            if bake_type == 'normal':
                # Bake normal map
                normal_map = self._bake_normal_map(highpoly_mesh, lowpoly_mesh, texture_size)
                baked_textures['normal'] = Texture(
                    image=normal_map,
                    uv_coords=uv_coords,
                    uv_indices=uv_indices
                )
            
            elif bake_type == 'ao':
                # Bake ambient occlusion
                ao_map = self._bake_ao_map(highpoly_mesh, lowpoly_mesh, texture_size)
                baked_textures['ao'] = Texture(
                    image=ao_map,
                    uv_coords=uv_coords,
                    uv_indices=uv_indices
                )
            
            elif bake_type == 'curvature':
                # Bake curvature map
                curvature_map = self._bake_curvature_map(highpoly_mesh, lowpoly_mesh, texture_size)
                baked_textures['curvature'] = Texture(
                    image=curvature_map,
                    uv_coords=uv_coords,
                    uv_indices=uv_indices
                )
        
        return baked_textures
    
    def _bake_normal_map(
        self,
        highpoly_mesh: Mesh,
        lowpoly_mesh: Mesh,
        texture_size: int
    ) -> np.ndarray:
        """Bake normal map from high-poly to low-poly mesh."""
        # Simplified: just use low-poly normals
        lowpoly_mesh.compute_normals()
        normals = lowpoly_mesh.vertex_normals
        
        # Create normal map (world space normals)
        normal_map = np.ones((texture_size, texture_size, 3)) * 0.5
        normal_map[:, :, 2] = 1.0  # Default to (0, 0, 1) in tangent space
        
        # Map normals to RGB
        # This is very simplified - proper baking requires ray casting
        
        return normal_map
    
    def _bake_ao_map(
        self,
        highpoly_mesh: Mesh,
        lowpoly_mesh: Mesh,
        texture_size: int
    ) -> np.ndarray:
        """Bake ambient occlusion map."""
        # Simplified: uniform AO
        ao_map = np.ones((texture_size, texture_size, 1))
        
        return ao_map
    
    def _bake_curvature_map(
        self,
        highpoly_mesh: Mesh,
        lowpoly_mesh: Mesh,
        texture_size: int
    ) -> np.ndarray:
        """Bake curvature map."""
        # Simplified: uniform curvature
        curvature_map = np.ones((texture_size, texture_size, 1)) * 0.5
        
        return curvature_map
    
    def clear_cache(self):
        """Clear texture cache."""
        self.texture_cache.clear()
        self.painter.reset_canvas()