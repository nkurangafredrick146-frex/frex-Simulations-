"""
Mesh generation from various representations.
Includes marching cubes, Poisson reconstruction, and neural mesh generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union
import trimesh
import skimage.measure
from scipy import ndimage
from scipy.spatial import KDTree
import open3d as o3d
import igl
import potpourri3d as pp3d
import networkx as nx
from dataclasses import dataclass
import time
import json


@dataclass
class Mesh:
    """Mesh data structure."""
    vertices: np.ndarray  # [V, 3]
    faces: np.ndarray     # [F, 3]
    vertex_normals: Optional[np.ndarray] = None  # [V, 3]
    face_normals: Optional[np.ndarray] = None    # [F, 3]
    vertex_colors: Optional[np.ndarray] = None   # [V, 3]
    texture_coords: Optional[np.ndarray] = None  # [V, 2]
    texture: Optional[np.ndarray] = None         # [H, W, 3]
    
    def to_trimesh(self) -> trimesh.Trimesh:
        """Convert to trimesh object."""
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.vertex_normals,
            vertex_colors=self.vertex_colors
        )
        return mesh
    
    def to_o3d(self) -> o3d.geometry.TriangleMesh:
        """Convert to Open3D triangle mesh."""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        
        if self.vertex_normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        if self.vertex_colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        
        return mesh
    
    def compute_normals(self):
        """Compute vertex and face normals."""
        if self.vertices is None or self.faces is None:
            return
        
        # Compute face normals
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_areas = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / (face_areas + 1e-8)
        
        # Compute vertex normals as weighted average of adjacent face normals
        vertex_normals = np.zeros_like(self.vertices)
        vertex_weights = np.zeros(len(self.vertices))
        
        for i, face in enumerate(self.faces):
            area = face_areas[i, 0]
            for j in range(3):
                vertex_normals[face[j]] += face_normals[i] * area
                vertex_weights[face[j]] += area
        
        # Normalize vertex normals
        vertex_normals = vertex_normals / (vertex_weights[:, np.newaxis] + 1e-8)
        vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8)
        
        self.face_normals = face_normals
        self.vertex_normals = vertex_normals
    
    def save(self, path: str, format: str = 'ply'):
        """Save mesh to file."""
        mesh = self.to_trimesh()
        
        if format == 'ply':
            mesh.export(path, file_type='ply')
        elif format == 'obj':
            mesh.export(path, file_type='obj')
        elif format == 'glb':
            mesh.export(path, file_type='glb')
        elif format == 'stl':
            mesh.export(path, file_type='stl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved mesh to {path} with {len(self.vertices)} vertices and {len(self.faces)} faces")
    
    @classmethod
    def load(cls, path: str) -> 'Mesh':
        """Load mesh from file."""
        mesh = trimesh.load(path)
        
        return cls(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            vertex_normals=np.asarray(mesh.vertex_normals) if mesh.vertex_normals is not None else None,
            vertex_colors=np.asarray(mesh.visual.vertex_colors[:, :3]) if hasattr(mesh.visual, 'vertex_colors') else None
        )


class MarchingCubesGenerator:
    """Generate mesh from signed distance function using marching cubes."""
    
    def __init__(
        self,
        resolution: int = 128,
        level: float = 0.0,
        gradient_direction: str = 'descent',
        step_size: float = 1.0,
        use_skimage: bool = True
    ):
        self.resolution = resolution
        self.level = level
        self.gradient_direction = gradient_direction
        self.step_size = step_size
        self.use_skimage = use_skimage
    
    def generate(
        self,
        sdf: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        smooth: bool = True,
        smoothing_sigma: float = 1.0
    ) -> Mesh:
        """
        Generate mesh from signed distance function.
        
        Args:
            sdf: Signed distance function values [res, res, res]
            bounds: (min, max) bounds of the volume
            smooth: Whether to smooth the SDF before meshing
            smoothing_sigma: Sigma for Gaussian smoothing
            
        Returns:
            Generated mesh
        """
        if smooth:
            sdf = ndimage.gaussian_filter(sdf, sigma=smoothing_sigma)
        
        bounds_min, bounds_max = bounds
        voxel_size = (bounds_max - bounds_min) / (self.resolution - 1)
        
        if self.use_skimage:
            # Use scikit-image marching cubes
            vertices, faces, normals, _ = skimage.measure.marching_cubes(
                sdf,
                level=self.level,
                spacing=voxel_size,
                gradient_direction=self.gradient_direction,
                step_size=self.step_size
            )
            
            # Transform vertices to world coordinates
            vertices = vertices + bounds_min
            
            mesh = Mesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals
            )
        else:
            # Alternative implementation using custom marching cubes
            mesh = self._custom_marching_cubes(sdf, bounds_min, voxel_size)
        
        mesh.compute_normals()
        return mesh
    
    def _custom_marching_cubes(
        self,
        sdf: np.ndarray,
        bounds_min: np.ndarray,
        voxel_size: np.ndarray
    ) -> Mesh:
        """Custom marching cubes implementation (simplified)."""
        # This is a simplified version - full implementation would be complex
        # For production, use scikit-image or other established libraries
        
        vertices_list = []
        faces_list = []
        
        # Get dimensions
        nx, ny, nz = sdf.shape
        
        # Precompute cube vertices offsets
        cube_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        # Edge table and triangle table for marching cubes
        # (Simplified - full tables are 256 entries each)
        
        for i in range(nx - 1):
            for j in range(ny - 1):
                for k in range(nz - 1):
                    # Get cube corner values
                    cube_values = [
                        sdf[i, j, k],
                        sdf[i+1, j, k],
                        sdf[i+1, j+1, k],
                        sdf[i, j+1, k],
                        sdf[i, j, k+1],
                        sdf[i+1, j, k+1],
                        sdf[i+1, j+1, k+1],
                        sdf[i, j+1, k+1]
                    ]
                    
                    # Determine cube index
                    cube_index = 0
                    for v in range(8):
                        if cube_values[v] > self.level:
                            cube_index |= (1 << v)
                    
                    # Skip empty cubes
                    if cube_index == 0 or cube_index == 255:
                        continue
                    
                    # Interpolate vertices on edges
                    # (Simplified - actual implementation would use edge table)
                    # This is just a placeholder
                    pass
        
        # Convert to arrays
        vertices = np.array(vertices_list) if vertices_list else np.zeros((0, 3))
        faces = np.array(faces_list) if faces_list else np.zeros((0, 3), dtype=int)
        
        # Scale and translate vertices
        vertices = vertices * voxel_size + bounds_min
        
        return Mesh(vertices=vertices, faces=faces)


class PoissonReconstructor:
    """Generate mesh from point cloud using Poisson surface reconstruction."""
    
    def __init__(
        self,
        depth: int = 8,
        width: float = 0.0,
        scale: float = 1.1,
        linear_fit: bool = True,
        n_threads: int = -1
    ):
        self.depth = depth
        self.width = width
        self.scale = scale
        self.linear_fit = linear_fit
        self.n_threads = n_threads
    
    def reconstruct(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        estimate_normals: bool = True,
        normal_estimation_radius: float = 0.1,
        normal_estimation_max_nn: int = 30
    ) -> Mesh:
        """
        Reconstruct mesh from point cloud using Poisson reconstruction.
        
        Args:
            points: Point cloud [N, 3]
            normals: Point normals [N, 3] (optional)
            colors: Point colors [N, 3] (optional)
            estimate_normals: Whether to estimate normals if not provided
            normal_estimation_radius: Radius for normal estimation
            normal_estimation_max_nn: Maximum neighbors for normal estimation
            
        Returns:
            Reconstructed mesh
        """
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals if needed
        if normals is None and estimate_normals:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_estimation_radius,
                    max_nn=normal_estimation_max_nn
                )
            )
            normals = np.asarray(pcd.normals)
        elif normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Perform Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.depth,
            width=self.width,
            scale=self.scale,
            linear_fit=self.linear_fit,
            n_threads=self.n_threads
        )
        
        # Convert to our mesh format
        result = Mesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None,
            vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        )
        
        return result
    
    def reconstruct_with_confidence(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        confidences: np.ndarray,
        min_confidence: float = 0.5
    ) -> Mesh:
        """
        Reconstruct mesh with confidence-weighted Poisson reconstruction.
        
        Args:
            points: Point cloud [N, 3]
            normals: Point normals [N, 3]
            confidences: Confidence values [N]
            min_confidence: Minimum confidence threshold
            
        Returns:
            Reconstructed mesh
        """
        # Filter points by confidence
        mask = confidences >= min_confidence
        filtered_points = points[mask]
        filtered_normals = normals[mask]
        
        if len(filtered_points) < 100:
            print(f"Warning: Only {len(filtered_points)} points after confidence filtering")
            # Return empty mesh
            return Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        
        # Reconstruct with filtered points
        return self.reconstruct(filtered_points, filtered_normals, estimate_normals=False)


class NeuralMeshGenerator(nn.Module):
    """
    Neural mesh generator using deep learning.
    
    Generates meshes from various inputs (images, point clouds, latent codes)
    using neural networks.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 8,
        max_vertices: int = 10000,
        max_faces: int = 20000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_vertices = max_vertices
        self.max_faces = max_faces
        self.device = device
        
        # Encoder networks
        self.point_encoder = self._build_point_encoder()
        self.image_encoder = self._build_image_encoder()
        
        # Decoder networks
        self.vertex_decoder = self._build_vertex_decoder()
        self.face_decoder = self._build_face_decoder()
        
        # Attention modules
        self.attention = nn.MultiheadAttention(latent_dim, num_heads=8, batch_first=True)
        
        # Initialize weights
        self._init_weights()
    
    def _build_point_encoder(self) -> nn.Module:
        """Build point cloud encoder."""
        encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim),
            nn.ReLU()
        )
        return encoder
    
    def _build_image_encoder(self) -> nn.Module:
        """Build image encoder using CNN."""
        encoder = nn.Sequential(
            # Input: [B, 3, 256, 256]
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.latent_dim),
            nn.ReLU()
        )
        return encoder
    
    def _build_vertex_decoder(self) -> nn.Module:
        """Build vertex decoder."""
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            ) for _ in range(self.num_layers - 2)],
            nn.Linear(self.hidden_dim, self.max_vertices * 3),
            nn.Tanh()  # Output in [-1, 1] range
        )
        return decoder
    
    def _build_face_decoder(self) -> nn.Module:
        """Build face decoder."""
        decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            ) for _ in range(self.num_layers - 2)],
            nn.Linear(self.hidden_dim, self.max_faces * 3 * 3),  # Each face has 3 vertices with 3 coordinates
            nn.Sigmoid()  # Output probabilities for face existence
        )
        return decoder
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_point_cloud(
        self,
        points: torch.Tensor,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode point cloud to latent vector.
        
        Args:
            points: Point cloud [N, 3]
            batch_size: Batch size for processing
            
        Returns:
            Latent vector [latent_dim]
        """
        if len(points) == 0:
            return torch.zeros(self.latent_dim, device=self.device)
        
        # Process in batches if needed
        if len(points) > batch_size:
            embeddings = []
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                batch_emb = self.point_encoder(batch)
                embeddings.append(batch_emb.mean(dim=0, keepdim=True))
            
            embeddings = torch.cat(embeddings, dim=0)
            latent = embeddings.mean(dim=0)
        else:
            embeddings = self.point_encoder(points)
            latent = embeddings.mean(dim=0)
        
        return latent
    
    def encode_image(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode image to latent vector.
        
        Args:
            image: Image [B, 3, H, W] or [3, H, W]
            
        Returns:
            Latent vector [B, latent_dim] or [latent_dim]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        latent = self.image_encoder(image)
        
        if latent.shape[0] == 1:
            latent = latent.squeeze(0)
        
        return latent
    
    def decode_vertices(
        self,
        latent: torch.Tensor,
        num_vertices: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decode vertices from latent vector.
        
        Args:
            latent: Latent vector [latent_dim] or [B, latent_dim]
            num_vertices: Number of vertices to generate (optional)
            
        Returns:
            Vertices [num_vertices, 3] or [B, num_vertices, 3]
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            batch_mode = False
        else:
            batch_mode = True
        
        # Decode vertices
        vertices_flat = self.vertex_decoder(latent)  # [B, max_vertices * 3]
        
        if num_vertices is None:
            num_vertices = self.max_vertices
        
        # Reshape to [B, num_vertices, 3]
        vertices = vertices_flat.view(-1, self.max_vertices, 3)
        vertices = vertices[:, :num_vertices, :]
        
        if not batch_mode:
            vertices = vertices.squeeze(0)
        
        return vertices
    
    def decode_faces(
        self,
        latent: torch.Tensor,
        vertices: torch.Tensor,
        num_faces: Optional[int] = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Decode faces from latent vector and vertices.
        
        Args:
            latent: Latent vector [latent_dim] or [B, latent_dim]
            vertices: Vertices [num_vertices, 3] or [B, num_vertices, 3]
            num_faces: Number of faces to generate (optional)
            threshold: Probability threshold for face existence
            
        Returns:
            Faces [num_faces, 3] or [B, num_faces, 3] (indices)
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            vertices = vertices.unsqueeze(0)
            batch_mode = False
        else:
            batch_mode = True
        
        batch_size = latent.shape[0]
        num_vertices = vertices.shape[1]
        
        if num_faces is None:
            num_faces = self.max_faces
        
        # Decode face probabilities
        faces_flat = self.face_decoder(latent)  # [B, max_faces * 9]
        faces_probs = faces_flat.view(batch_size, self.max_faces, 3, 3)
        
        # Get top-k faces by probability
        face_scores = faces_probs.mean(dim=(2, 3))  # [B, max_faces]
        topk_scores, topk_indices = torch.topk(face_scores, num_faces, dim=1)
        
        # Get vertex indices for top faces
        # This is simplified - actual implementation would use learned indices
        # Here we use a triangulation of the vertex set
        
        if batch_mode:
            faces = torch.zeros(batch_size, num_faces, 3, dtype=torch.long, device=self.device)
            for b in range(batch_size):
                # Simple triangulation (Delaunay would be better)
                # This is just a placeholder
                if num_vertices >= 3:
                    # Create a simple triangle fan
                    for f in range(min(num_faces, num_vertices - 2)):
                        faces[b, f] = torch.tensor([0, f+1, f+2], device=self.device)
        else:
            faces = torch.zeros(num_faces, 3, dtype=torch.long, device=self.device)
            if num_vertices >= 3:
                for f in range(min(num_faces, num_vertices - 2)):
                    faces[f] = torch.tensor([0, f+1, f+2], device=self.device)
        
        if not batch_mode:
            faces = faces.squeeze(0)
        
        return faces
    
    def generate_from_point_cloud(
        self,
        points: np.ndarray,
        num_vertices: int = 1000,
        num_faces: int = 2000
    ) -> Mesh:
        """
        Generate mesh from point cloud.
        
        Args:
            points: Point cloud [N, 3]
            num_vertices: Target number of vertices
            num_faces: Target number of faces
            
        Returns:
            Generated mesh
        """
        # Convert to tensor
        points_tensor = torch.from_numpy(points).float().to(self.device)
        
        # Encode
        with torch.no_grad():
            latent = self.encode_point_cloud(points_tensor)
            
            # Decode vertices
            vertices = self.decode_vertices(latent, num_vertices)
            
            # Decode faces
            faces = self.decode_faces(latent, vertices, num_faces)
            
            # Convert to numpy
            vertices_np = vertices.cpu().numpy()
            faces_np = faces.cpu().numpy()
        
        # Create mesh
        mesh = Mesh(vertices=vertices_np, faces=faces_np)
        mesh.compute_normals()
        
        return mesh
    
    def generate_from_image(
        self,
        image: np.ndarray,
        num_vertices: int = 1000,
        num_faces: int = 2000
    ) -> Mesh:
        """
        Generate mesh from image.
        
        Args:
            image: Input image [H, W, 3]
            num_vertices: Target number of vertices
            num_faces: Target number of faces
            
        Returns:
            Generated mesh
        """
        # Preprocess image
        image_tensor = torch.from_numpy(image).float().to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # [3, H, W]
        
        # Resize if needed
        if image_tensor.shape[1] != 256 or image_tensor.shape[2] != 256:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Encode and decode
        with torch.no_grad():
            latent = self.encode_image(image_tensor)
            vertices = self.decode_vertices(latent, num_vertices)
            faces = self.decode_faces(latent, vertices, num_faces)
            
            vertices_np = vertices.cpu().numpy()
            faces_np = faces.cpu().numpy()
        
        mesh = Mesh(vertices=vertices_np, faces=faces_np)
        mesh.compute_normals()
        
        return mesh
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        loss_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Training step.
        
        Args:
            batch: Batch containing 'points', 'vertices', 'faces', etc.
            optimizer: Optimizer
            loss_weights: Weights for different loss terms
            
        Returns:
            Dictionary of loss values
        """
        if loss_weights is None:
            loss_weights = {
                'vertex': 1.0,
                'face': 0.5,
                'latent': 0.1,
                'regularization': 0.01
            }
        
        # Forward pass
        points = batch.get('points')
        gt_vertices = batch.get('vertices')
        gt_faces = batch.get('faces')
        
        if points is not None:
            latent = self.encode_point_cloud(points)
        else:
            # Use random latent
            latent = torch.randn(1, self.latent_dim, device=self.device)
        
        pred_vertices = self.decode_vertices(latent)
        pred_faces = self.decode_faces(latent, pred_vertices)
        
        # Compute losses
        losses = {}
        
        if gt_vertices is not None:
            # Chamfer distance for vertices
            chamfer_loss = self.compute_chamfer_distance(pred_vertices, gt_vertices)
            losses['vertex'] = chamfer_loss
        
        if gt_faces is not None:
            # Face loss (simplified)
            face_loss = F.mse_loss(pred_faces.float(), gt_faces.float())
            losses['face'] = face_loss
        
        # Latent regularization
        latent_loss = torch.mean(latent ** 2)
        losses['latent'] = latent_loss
        
        # Total loss
        total_loss = sum(loss * loss_weights.get(name, 0.0) 
                        for name, loss in losses.items())
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Return loss values
        loss_dict = {name: loss.item() for name, loss in losses.items()}
        loss_dict['total'] = total_loss.item()
        
        return loss_dict
    
    def compute_chamfer_distance(
        self,
        pred_points: torch.Tensor,
        gt_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Chamfer distance between point sets.
        
        Args:
            pred_points: Predicted points [N, 3]
            gt_points: Ground truth points [M, 3]
            
        Returns:
            Chamfer distance
        """
        # For each predicted point, find nearest ground truth point
        dist_matrix = torch.cdist(pred_points, gt_points)  # [N, M]
        min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=1)  # [N]
        
        # For each ground truth point, find nearest predicted point
        min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=0)  # [M]
        
        # Chamfer distance
        chamfer_dist = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)
        
        return chamfer_dist


class MeshGenerator:
    """
    Unified mesh generation interface.
    
    Provides multiple methods for mesh generation:
    - Marching cubes from SDF
    - Poisson reconstruction from point cloud
    - Neural generation from images/point clouds
    - Primitive-based generation
    """
    
    def __init__(
        self,
        method: str = 'poisson',  # 'marching_cubes', 'poisson', 'neural', 'primitive'
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.method = method
        self.device = device
        
        # Initialize generators
        self.marching_cubes = MarchingCubesGenerator()
        self.poisson = PoissonReconstructor()
        self.neural = NeuralMeshGenerator(device=device) if method == 'neural' else None
        
        # Cache for generated meshes
        self.cache = {}
    
    def generate_from_sdf(
        self,
        sdf: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        smooth: bool = True,
        smoothing_sigma: float = 1.0
    ) -> Mesh:
        """
        Generate mesh from signed distance function using marching cubes.
        
        Args:
            sdf: Signed distance function [res, res, res]
            bounds: (min, max) bounds
            smooth: Whether to smooth SDF
            smoothing_sigma: Sigma for Gaussian smoothing
            
        Returns:
            Generated mesh
        """
        cache_key = ('sdf', hash(sdf.tobytes()), smooth, smoothing_sigma)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        mesh = self.marching_cubes.generate(sdf, bounds, smooth, smoothing_sigma)
        self.cache[cache_key] = mesh
        
        return mesh
    
    def generate_from_point_cloud(
        self,
        points: np.ndarray,
        normals: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        method: Optional[str] = None
    ) -> Mesh:
        """
        Generate mesh from point cloud.
        
        Args:
            points: Point cloud [N, 3]
            normals: Point normals [N, 3] (optional)
            colors: Point colors [N, 3] (optional)
            method: Override default method
            
        Returns:
            Generated mesh
        """
        method = method or self.method
        
        if method == 'poisson':
            cache_key = ('poisson', hash(points.tobytes()))
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            mesh = self.poisson.reconstruct(points, normals, colors)
            self.cache[cache_key] = mesh
            
        elif method == 'neural' and self.neural is not None:
            cache_key = ('neural', hash(points.tobytes()))
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            mesh = self.neural.generate_from_point_cloud(points)
            self.cache[cache_key] = mesh
            
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return mesh
    
    def generate_from_image(
        self,
        image: np.ndarray,
        num_vertices: int = 1000,
        num_faces: int = 2000
    ) -> Mesh:
        """
        Generate mesh from image using neural network.
        
        Args:
            image: Input image [H, W, 3]
            num_vertices: Target number of vertices
            num_faces: Target number of faces
            
        Returns:
            Generated mesh
        """
        if self.neural is None:
            raise ValueError("Neural generator not initialized")
        
        cache_key = ('image', hash(image.tobytes()), num_vertices, num_faces)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        mesh = self.neural.generate_from_image(image, num_vertices, num_faces)
        self.cache[cache_key] = mesh
        
        return mesh
    
    def generate_primitive(
        self,
        primitive_type: str = 'sphere',
        size: float = 1.0,
        subdivisions: int = 2
    ) -> Mesh:
        """
        Generate primitive mesh.
        
        Args:
            primitive_type: 'sphere', 'cube', 'cylinder', 'cone', 'torus'
            size: Size of primitive
            subdivisions: Number of subdivisions
            
        Returns:
            Primitive mesh
        """
        cache_key = ('primitive', primitive_type, size, subdivisions)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if primitive_type == 'sphere':
            mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=size)
        elif primitive_type == 'cube':
            mesh = trimesh.creation.box(extents=[size, size, size])
        elif primitive_type == 'cylinder':
            mesh = trimesh.creation.cylinder(radius=size/2, height=size, sections=32)
        elif primitive_type == 'cone':
            mesh = trimesh.creation.cone(radius=size/2, height=size, sections=32)
        elif primitive_type == 'torus':
            mesh = trimesh.creation.torus(major_radius=size, minor_radius=size/3)
        else:
            raise ValueError(f"Unknown primitive type: {primitive_type}")
        
        result = Mesh(
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
            vertex_normals=np.asarray(mesh.vertex_normals) if mesh.vertex_normals is not None else None
        )
        
        self.cache[cache_key] = result
        return result
    
    def clear_cache(self):
        """Clear mesh cache."""
        self.cache.clear()
    
    def export_all_formats(
        self,
        mesh: Mesh,
        base_path: str,
        formats: List[str] = ['ply', 'obj', 'glb', 'stl']
    ):
        """
        Export mesh in multiple formats.
        
        Args:
            mesh: Mesh to export
            base_path: Base path without extension
            formats: List of formats to export
        """
        for fmt in formats:
            path = f"{base_path}.{fmt}"
            mesh.save(path, format=fmt)
    
    def validate_mesh(
        self,
        mesh: Mesh,
        check_watertight: bool = True,
        check_self_intersection: bool = False,
        check_degenerate: bool = True
    ) -> Dict[str, Any]:
        """
        Validate mesh quality.
        
        Args:
            mesh: Mesh to validate
            check_watertight: Check if mesh is watertight
            check_self_intersection: Check for self-intersections
            check_degenerate: Check for degenerate faces
            
        Returns:
            Dictionary with validation results
        """
        trimesh_mesh = mesh.to_trimesh()
        
        results = {
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
            'is_watertight': False,
            'has_self_intersection': False,
            'num_degenerate_faces': 0,
            'volume': 0.0,
            'surface_area': 0.0,
            'bounds': None
        }
        
        # Check watertight
        if check_watertight:
            results['is_watertight'] = trimesh_mesh.is_watertight
            if results['is_watertight']:
                results['volume'] = trimesh_mesh.volume
        
        # Check self-intersection
        if check_self_intersection:
            results['has_self_intersection'] = trimesh_mesh.is_self_intersecting
        
        # Check degenerate faces
        if check_degenerate:
            # Faces with zero area
            areas = trimesh_mesh.area_faces
            results['num_degenerate_faces'] = np.sum(areas < 1e-8)
        
        # Compute surface area
        results['surface_area'] = trimesh_mesh.area
        
        # Compute bounds
        results['bounds'] = {
            'min': mesh.vertices.min(axis=0).tolist(),
            'max': mesh.vertices.max(axis=0).tolist()
        }
        
        return results