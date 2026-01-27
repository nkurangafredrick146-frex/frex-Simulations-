"""
Mesh refinement operations.
Includes smoothing, subdivision, simplification, hole filling, and mesh optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union
import trimesh
import open3d as o3d
import igl
import potpourri3d as pp3d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from dataclasses import dataclass
import time
import json


@dataclass
class RefinementConfig:
    """Configuration for mesh refinement."""
    
    # Smoothing
    smoothing_iterations: int = 10
    smoothing_lambda: float = 0.5
    smoothing_mu: float = -0.53
    
    # Subdivision
    subdivision_method: str = 'loop'  # 'loop', 'butterfly', 'sqrt3'
    subdivision_levels: int = 2
    
    # Simplification
    target_face_count: int = 1000
    target_reduction: float = 0.5
    simplification_aggressiveness: float = 7.0
    
    # Hole filling
    max_hole_size: int = 100
    fairing_iterations: int = 100
    
    # Remeshing
    target_edge_length: float = 0.05
    remeshing_iterations: int = 10
    
    # Mesh optimization
    laplacian_weight: float = 1.0
    position_weight: float = 0.1
    normal_weight: float = 0.5
    edge_weight: float = 0.01
    
    # Quality thresholds
    min_angle: float = 10.0  # degrees
    max_angle: float = 120.0  # degrees
    max_aspect_ratio: float = 10.0


class MeshSmoother:
    """Mesh smoothing operations."""
    
    def __init__(
        self,
        method: str = 'laplacian',  # 'laplacian', 'taubin', 'bilaplacian', 'humphrey'
        iterations: int = 10,
        lambda_param: float = 0.5,
        mu_param: float = -0.53
    ):
        self.method = method
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.mu_param = mu_param
    
    def smooth(
        self,
        mesh: Mesh,
        fixed_vertices: Optional[np.ndarray] = None,
        vertex_weights: Optional[np.ndarray] = None
    ) -> Mesh:
        """
        Smooth mesh using specified method.
        
        Args:
            mesh: Input mesh
            fixed_vertices: Boolean array of fixed vertices
            vertex_weights: Weight for each vertex (higher = less smoothing)
            
        Returns:
            Smoothed mesh
        """
        if self.method == 'laplacian':
            return self._laplacian_smooth(mesh, fixed_vertices, vertex_weights)
        elif self.method == 'taubin':
            return self._taubin_smooth(mesh, fixed_vertices, vertex_weights)
        elif self.method == 'bilaplacian':
            return self._bilaplacian_smooth(mesh, fixed_vertices, vertex_weights)
        elif self.method == 'humphrey':
            return self._humphrey_smooth(mesh, fixed_vertices, vertex_weights)
        else:
            raise ValueError(f"Unknown smoothing method: {self.method}")
    
    def _laplacian_smooth(
        self,
        mesh: Mesh,
        fixed_vertices: Optional[np.ndarray],
        vertex_weights: Optional[np.ndarray]
    ) -> Mesh:
        """Laplacian smoothing."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Build adjacency matrix
        num_vertices = len(vertices)
        adj_matrix = self._build_adjacency_matrix(faces, num_vertices)
        
        # Laplacian matrix: L = D - A
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        laplacian = sparse.diags(degree) - adj_matrix
        
        # Smoothing iterations
        for _ in range(self.iterations):
            # Compute Laplacian coordinates
            lap_coords = laplacian.dot(vertices)
            
            # Update vertices
            vertices = vertices + self.lambda_param * lap_coords
            
            # Keep fixed vertices in place
            if fixed_vertices is not None:
                vertices[fixed_vertices] = mesh.vertices[fixed_vertices]
        
        # Create smoothed mesh
        smoothed_mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        smoothed_mesh.compute_normals()
        
        return smoothed_mesh
    
    def _taubin_smooth(
        self,
        mesh: Mesh,
        fixed_vertices: Optional[np.ndarray],
        vertex_weights: Optional[np.ndarray]
    ) -> Mesh:
        """Taubin smoothing (λ-μ smoothing)."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Build adjacency matrix
        num_vertices = len(vertices)
        adj_matrix = self._build_adjacency_matrix(faces, num_vertices)
        
        # Laplacian matrix
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        laplacian = sparse.diags(degree) - adj_matrix
        
        # Taubin smoothing iterations
        for i in range(self.iterations):
            # Compute Laplacian coordinates
            lap_coords = laplacian.dot(vertices)
            
            # Alternate between positive and negative smoothing
            if i % 2 == 0:
                lambda_current = self.lambda_param
            else:
                lambda_current = self.mu_param
            
            # Update vertices
            vertices = vertices + lambda_current * lap_coords
            
            # Keep fixed vertices in place
            if fixed_vertices is not None:
                vertices[fixed_vertices] = mesh.vertices[fixed_vertices]
        
        smoothed_mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        smoothed_mesh.compute_normals()
        
        return smoothed_mesh
    
    def _bilaplacian_smooth(
        self,
        mesh: Mesh,
        fixed_vertices: Optional[np.ndarray],
        vertex_weights: Optional[np.ndarray]
    ) -> Mesh:
        """Bilaplacian smoothing (more global)."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Build adjacency and Laplacian matrices
        num_vertices = len(vertices)
        adj_matrix = self._build_adjacency_matrix(faces, num_vertices)
        degree = np.array(adj_matrix.sum(axis=1)).flatten()
        laplacian = sparse.diags(degree) - adj_matrix
        
        # Bilaplacian: L^2
        bilaplacian = laplacian.dot(laplacian)
        
        # Solve linear system: (I + λL^2) x' = x
        identity = sparse.eye(num_vertices)
        system_matrix = identity + self.lambda_param * bilaplacian
        
        # Solve for each coordinate
        new_vertices = np.zeros_like(vertices)
        for i in range(3):
            new_vertices[:, i] = spsolve(system_matrix, vertices[:, i])
        
        # Keep fixed vertices in place
        if fixed_vertices is not None:
            new_vertices[fixed_vertices] = mesh.vertices[fixed_vertices]
        
        smoothed_mesh = Mesh(
            vertices=new_vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        smoothed_mesh.compute_normals()
        
        return smoothed_mesh
    
    def _humphrey_smooth(
        self,
        mesh: Mesh,
        fixed_vertices: Optional[np.ndarray],
        vertex_weights: Optional[np.ndarray]
    ) -> Mesh:
        """Humphrey smoothing (weighted by triangle areas)."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        num_vertices = len(vertices)
        
        # Compute triangle areas
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        
        # Build weighted adjacency
        row_indices = []
        col_indices = []
        data = []
        
        for f_idx, face in enumerate(faces):
            area = areas[f_idx]
            for i in range(3):
                for j in range(3):
                    if i != j:
                        row_indices.append(face[i])
                        col_indices.append(face[j])
                        data.append(area)
        
        # Create sparse matrix
        adj_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_vertices, num_vertices)
        )
        
        # Normalize by vertex areas
        vertex_areas = np.array(adj_matrix.sum(axis=1)).flatten()
        inv_vertex_areas = 1.0 / (vertex_areas + 1e-8)
        
        # Create normalized Laplacian
        for _ in range(self.iterations):
            # Compute weighted average
            weighted_sum = adj_matrix.dot(vertices)
            new_vertices = vertices.copy()
            
            for i in range(num_vertices):
                if fixed_vertices is None or not fixed_vertices[i]:
                    if vertex_areas[i] > 0:
                        new_vertices[i] = (1 - self.lambda_param) * vertices[i] + \
                                         self.lambda_param * weighted_sum[i] * inv_vertex_areas[i]
            
            vertices = new_vertices
        
        smoothed_mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        smoothed_mesh.compute_normals()
        
        return smoothed_mesh
    
    def _build_adjacency_matrix(
        self,
        faces: np.ndarray,
        num_vertices: int
    ) -> sparse.csr_matrix:
        """Build adjacency matrix from faces."""
        row_indices = []
        col_indices = []
        
        for face in faces:
            # Add all pairs in triangle
            for i in range(3):
                for j in range(3):
                    if i != j:
                        row_indices.append(face[i])
                        col_indices.append(face[j])
        
        # Create sparse matrix (binary adjacency)
        data = np.ones(len(row_indices))
        adj_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_vertices, num_vertices)
        )
        
        return adj_matrix


class MeshSubdivider:
    """Mesh subdivision operations."""
    
    def __init__(
        self,
        method: str = 'loop',  # 'loop', 'butterfly', 'sqrt3', 'catmull-clark'
        levels: int = 2
    ):
        self.method = method
        self.levels = levels
    
    def subdivide(
        self,
        mesh: Mesh,
        interpolate_attributes: bool = True
    ) -> Mesh:
        """
        Subdivide mesh using specified method.
        
        Args:
            mesh: Input mesh
            interpolate_attributes: Whether to interpolate vertex colors/normals
            
        Returns:
            Subdivided mesh
        """
        if self.method == 'loop':
            return self._loop_subdivision(mesh, interpolate_attributes)
        elif self.method == 'butterfly':
            return self._butterfly_subdivision(mesh, interpolate_attributes)
        elif self.method == 'sqrt3':
            return self._sqrt3_subdivision(mesh, interpolate_attributes)
        elif self.method == 'catmull-clark':
            return self._catmull_clark_subdivision(mesh, interpolate_attributes)
        else:
            raise ValueError(f"Unknown subdivision method: {self.method}")
    
    def _loop_subdivision(
        self,
        mesh: Mesh,
        interpolate_attributes: bool
    ) -> Mesh:
        """Loop subdivision (for triangle meshes)."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        # Interpolate attributes if needed
        if interpolate_attributes and mesh.vertex_colors is not None:
            vertex_colors = mesh.vertex_colors.copy()
        else:
            vertex_colors = None
        
        for level in range(self.levels):
            # Build edge to new vertex mapping
            edge_to_vertex = {}
            new_vertices = []
            new_colors = []
            
            # Add original vertices to new vertices
            for i, v in enumerate(vertices):
                new_vertices.append(v)
                if vertex_colors is not None:
                    new_colors.append(vertex_colors[i])
            
            # Create new vertices for each edge
            for face in faces:
                for i in range(3):
                    v0 = face[i]
                    v1 = face[(i + 1) % 3]
                    
                    # Ensure consistent ordering
                    edge = (min(v0, v1), max(v0, v1))
                    
                    if edge not in edge_to_vertex:
                        # Compute edge midpoint
                        mid_point = (vertices[v0] + vertices[v1]) / 2
                        
                        # Interpolate color if available
                        if vertex_colors is not None:
                            mid_color = (vertex_colors[v0] + vertex_colors[v1]) / 2
                            new_colors.append(mid_color)
                        
                        edge_to_vertex[edge] = len(new_vertices)
                        new_vertices.append(mid_point)
            
            # Create new faces (4 triangles per original triangle)
            new_faces = []
            for face in faces:
                v0, v1, v2 = face
                
                # Get edge vertices
                e0 = edge_to_vertex[(min(v0, v1), max(v0, v1))]
                e1 = edge_to_vertex[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_vertex[(min(v2, v0), max(v2, v0))]
                
                # Create 4 new triangles
                new_faces.append([v0, e0, e2])
                new_faces.append([v1, e1, e0])
                new_faces.append([v2, e2, e1])
                new_faces.append([e0, e1, e2])
            
            # Update vertices and faces
            vertices = np.array(new_vertices)
            faces = np.array(new_faces)
            
            if vertex_colors is not None:
                vertex_colors = np.array(new_colors)
        
        # Create subdivided mesh
        subdivided_mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=vertex_colors
        )
        subdivided_mesh.compute_normals()
        
        return subdivided_mesh
    
    def _butterfly_subdivision(
        self,
        mesh: Mesh,
        interpolate_attributes: bool
    ) -> Mesh:
        """Butterfly subdivision (interpolating)."""
        # Simplified butterfly scheme
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        for level in range(self.levels):
            # Similar to loop but with different weights
            edge_to_vertex = {}
            new_vertices = []
            
            # Add original vertices
            new_vertices.extend(vertices)
            
            # Create new vertices with butterfly weights
            # (Simplified - full butterfly has complex stencil)
            for face in faces:
                for i in range(3):
                    v0 = face[i]
                    v1 = face[(i + 1) % 3]
                    edge = (min(v0, v1), max(v0, v1))
                    
                    if edge not in edge_to_vertex:
                        # Simple interpolation (could use butterfly weights)
                        mid_point = (vertices[v0] + vertices[v1]) / 2
                        edge_to_vertex[edge] = len(new_vertices)
                        new_vertices.append(mid_point)
            
            # Create new faces (same as loop)
            new_faces = []
            for face in faces:
                v0, v1, v2 = face
                e0 = edge_to_vertex[(min(v0, v1), max(v0, v1))]
                e1 = edge_to_vertex[(min(v1, v2), max(v1, v2))]
                e2 = edge_to_vertex[(min(v2, v0), max(v2, v0))]
                
                new_faces.append([v0, e0, e2])
                new_faces.append([v1, e1, e0])
                new_faces.append([v2, e2, e1])
                new_faces.append([e0, e1, e2])
            
            vertices = np.array(new_vertices)
            faces = np.array(new_faces)
        
        subdivided_mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        subdivided_mesh.compute_normals()
        
        return subdivided_mesh
    
    def _sqrt3_subdivision(
        self,
        mesh: Mesh,
        interpolate_attributes: bool
    ) -> Mesh:
        """√3 subdivision."""
        # This is a simplified implementation
        # Full √3 subdivision is more complex
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        for level in range(self.levels):
            # Compute face centers
            face_centers = []
            for face in faces:
                center = vertices[face].mean(axis=0)
                face_centers.append(center)
            
            face_centers = np.array(face_centers)
            
            # Update original vertices
            # (Simplified - actual √3 has specific update rules)
            new_vertices = vertices.copy()
            
            # Add face centers as new vertices
            num_original = len(vertices)
            new_vertices = np.vstack([new_vertices, face_centers])
            
            # Create new faces (3 per original face)
            new_faces = []
            for f_idx, face in enumerate(faces):
                v0, v1, v2 = face
                fc = num_original + f_idx  # Face center index
                
                new_faces.append([v0, v1, fc])
                new_faces.append([v1, v2, fc])
                new_faces.append([v2, v0, fc])
            
            vertices = new_vertices
            faces = np.array(new_faces)
        
        subdivided_mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        subdivided_mesh.compute_normals()
        
        return subdivided_mesh
    
    def _catmull_clark_subdivision(
        self,
        mesh: Mesh,
        interpolate_attributes: bool
    ) -> Mesh:
        """Catmull-Clark subdivision (for quad meshes)."""
        # Convert to quads if needed
        # This is a placeholder - full implementation is complex
        raise NotImplementedError("Catmull-Clark subdivision requires quad meshes")


class MeshSimplifier:
    """Mesh simplification operations."""
    
    def __init__(
        self,
        method: str = 'quadric',  # 'quadric', 'edge_collapse', 'vertex_clustering'
        target_face_count: int = 1000,
        target_reduction: float = 0.5,
        aggressiveness: float = 7.0
    ):
        self.method = method
        self.target_face_count = target_face_count
        self.target_reduction = target_reduction
        self.aggressiveness = aggressiveness
    
    def simplify(
        self,
        mesh: Mesh,
        preserve_border: bool = True,
        preserve_normals: bool = True,
        preserve_colors: bool = True
    ) -> Mesh:
        """
        Simplify mesh while preserving shape.
        
        Args:
            mesh: Input mesh
            preserve_border: Preserve mesh borders
            preserve_normals: Preserve vertex normals
            preserve_colors: Preserve vertex colors
            
        Returns:
            Simplified mesh
        """
        if self.method == 'quadric':
            return self._quadric_simplification(mesh, preserve_border, preserve_normals, preserve_colors)
        elif self.method == 'edge_collapse':
            return self._edge_collapse_simplification(mesh, preserve_border, preserve_normals, preserve_colors)
        elif self.method == 'vertex_clustering':
            return self._vertex_clustering_simplification(mesh)
        else:
            raise ValueError(f"Unknown simplification method: {self.method}")
    
    def _quadric_simplification(
        self,
        mesh: Mesh,
        preserve_border: bool,
        preserve_normals: bool,
        preserve_colors: bool
    ) -> Mesh:
        """Quadric error metric simplification."""
        # Use Open3D for quadric simplification
        o3d_mesh = mesh.to_o3d()
        
        # Compute target number of triangles
        current_faces = len(mesh.faces)
        target_faces = min(
            self.target_face_count,
            int(current_faces * self.target_reduction)
        )
        
        if target_faces >= current_faces:
            # No simplification needed
            return mesh
        
        # Simplify mesh
        simplified = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        
        # Convert back to our format
        simplified_mesh = Mesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles),
            vertex_normals=np.asarray(simplified.vertex_normals) if simplified.has_vertex_normals() else None,
            vertex_colors=np.asarray(simplified.vertex_colors) if simplified.has_vertex_colors() else None
        )
        
        return simplified_mesh
    
    def _edge_collapse_simplification(
        self,
        mesh: Mesh,
        preserve_border: bool,
        preserve_normals: bool,
        preserve_colors: bool
    ) -> Mesh:
        """Edge collapse simplification."""
        # Use trimesh for edge collapse
        trimesh_mesh = mesh.to_trimesh()
        
        # Compute target number of faces
        current_faces = len(mesh.faces)
        target_faces = min(
            self.target_face_count,
            int(current_faces * self.target_reduction)
        )
        
        if target_faces >= current_faces:
            return mesh
        
        # Simplify using edge collapse
        # Note: trimesh doesn't have direct edge collapse, so we use quadric
        simplified = trimesh_mesh.simplify_quadric_decimation(target_faces)
        
        simplified_mesh = Mesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.faces),
            vertex_normals=np.asarray(simplified.vertex_normals) if simplified.vertex_normals is not None else None,
            vertex_colors=np.asarray(simplified.visual.vertex_colors[:, :3]) 
            if hasattr(simplified.visual, 'vertex_colors') else None
        )
        
        return simplified_mesh
    
    def _vertex_clustering_simplification(self, mesh: Mesh) -> Mesh:
        """Vertex clustering simplification."""
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Compute bounding box and grid size
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        # Determine grid resolution based on target reduction
        current_vertices = len(vertices)
        target_vertices = int(current_vertices * self.target_reduction)
        
        # Estimate grid size (crude approximation)
        volume = bbox_size.prod()
        cell_volume = volume / target_vertices
        cell_size = cell_volume ** (1/3)
        
        # Create grid
        grid_indices = np.floor((vertices - bbox_min) / cell_size).astype(int)
        
        # Map vertices to grid cells
        cell_to_vertex = {}
        vertex_mapping = np.zeros(len(vertices), dtype=int)
        
        for i, cell in enumerate(grid_indices):
            cell_key = tuple(cell)
            if cell_key not in cell_to_vertex:
                cell_to_vertex[cell_key] = len(cell_to_vertex)
            vertex_mapping[i] = cell_to_vertex[cell_key]
        
        # Compute cell centroids
        cell_vertices = np.zeros((len(cell_to_vertex), 3))
        cell_counts = np.zeros(len(cell_to_vertex))
        
        for i, cell in enumerate(grid_indices):
            cell_key = tuple(cell)
            cell_idx = cell_to_vertex[cell_key]
            cell_vertices[cell_idx] += vertices[i]
            cell_counts[cell_idx] += 1
        
        cell_vertices /= cell_counts[:, np.newaxis]
        
        # Create new faces (remove degenerate)
        new_faces = []
        for face in faces:
            new_face = vertex_mapping[face]
            # Keep face if all vertices are different
            if len(set(new_face)) == 3:
                new_faces.append(new_face)
        
        simplified_mesh = Mesh(
            vertices=cell_vertices,
            faces=np.array(new_faces),
            vertex_normals=None,
            vertex_colors=None
        )
        simplified_mesh.compute_normals()
        
        return simplified_mesh


class MeshHoleFiller:
    """Mesh hole filling operations."""
    
    def __init__(
        self,
        max_hole_size: int = 100,
        fairing_iterations: int = 100,
        method: str = 'minimal'  # 'minimal', 'smooth', 'advanced'
    ):
        self.max_hole_size = max_hole_size
        self.fairing_iterations = fairing_iterations
        self.method = method
    
    def fill_holes(
        self,
        mesh: Mesh,
        smooth_filled: bool = True
    ) -> Mesh:
        """
        Fill holes in mesh.
        
        Args:
            mesh: Input mesh with holes
            smooth_filled: Smooth the filled regions
            
        Returns:
            Mesh with filled holes
        """
        # Convert to trimesh for hole filling
        trimesh_mesh = mesh.to_trimesh()
        
        # Fill holes
        trimesh_mesh.fill_holes(max_hole_size=self.max_hole_size)
        
        if smooth_filled:
            # Smooth the filled regions
            trimesh_mesh = self._smooth_filled_regions(trimesh_mesh)
        
        # Convert back
        filled_mesh = Mesh(
            vertices=np.asarray(trimesh_mesh.vertices),
            faces=np.asarray(trimesh_mesh.faces),
            vertex_normals=np.asarray(trimesh_mesh.vertex_normals) if trimesh_mesh.vertex_normals is not None else None,
            vertex_colors=np.asarray(trimesh_mesh.visual.vertex_colors[:, :3]) 
            if hasattr(trimesh_mesh.visual, 'vertex_colors') else None
        )
        
        return filled_mesh
    
    def _smooth_filled_regions(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Smooth the regions that were filled."""
        # This is a simplified implementation
        # In practice, you'd identify newly added faces and smooth only those
        
        # For now, just do global smoothing
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Simple Laplacian smoothing on boundary vertices
        for _ in range(self.fairing_iterations // 10):
            # Find boundary vertices
            from trimesh.grouping import group_rows
            edges = mesh.edges
            unique_edges = group_rows(edges, require_count=1)
            boundary_vertices = np.unique(edges[unique_edges].flatten())
            
            # Smooth boundary vertices
            for v_idx in boundary_vertices:
                # Find neighboring vertices
                neighbor_faces = np.where(np.any(faces == v_idx, axis=1))[0]
                neighbor_vertices = np.unique(faces[neighbor_faces].flatten())
                neighbor_vertices = neighbor_vertices[neighbor_vertices != v_idx]
                
                if len(neighbor_vertices) > 0:
                    vertices[v_idx] = vertices[neighbor_vertices].mean(axis=0)
        
        mesh.vertices = vertices
        return mesh


class MeshRefiner:
    """
    Unified mesh refinement interface.
    
    Provides:
    - Smoothing
    - Subdivision
    - Simplification
    - Hole filling
    - Quality improvement
    - Remeshing
    """
    
    def __init__(
        self,
        config: Optional[RefinementConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.config = config or RefinementConfig()
        self.device = device
        
        # Initialize components
        self.smoother = MeshSmoother(
            method='taubin',
            iterations=self.config.smoothing_iterations,
            lambda_param=self.config.smoothing_lambda,
            mu_param=self.config.smoothing_mu
        )
        
        self.subdivider = MeshSubdivider(
            method=self.config.subdivision_method,
            levels=self.config.subdivision_levels
        )
        
        self.simplifier = MeshSimplifier(
            method='quadric',
            target_face_count=self.config.target_face_count,
            target_reduction=self.config.target_reduction,
            aggressiveness=self.config.simplification_aggressiveness
        )
        
        self.hole_filler = MeshHoleFiller(
            max_hole_size=self.config.max_hole_size,
            fairing_iterations=self.config.fairing_iterations,
            method='smooth'
        )
    
    def refine_mesh(
        self,
        mesh: Mesh,
        operations: List[str] = None,
        **kwargs
    ) -> Mesh:
        """
        Apply refinement operations to mesh.
        
        Args:
            mesh: Input mesh
            operations: List of operations to apply
                       ['smooth', 'subdivide', 'simplify', 'fill_holes', 'improve_quality']
            **kwargs: Additional parameters for specific operations
            
        Returns:
            Refined mesh
        """
        if operations is None:
            operations = ['improve_quality']  # Default
        
        refined_mesh = mesh
        
        for op in operations:
            if op == 'smooth':
                refined_mesh = self.smooth(refined_mesh, **kwargs.get('smooth', {}))
            elif op == 'subdivide':
                refined_mesh = self.subdivide(refined_mesh, **kwargs.get('subdivide', {}))
            elif op == 'simplify':
                refined_mesh = self.simplify(refined_mesh, **kwargs.get('simplify', {}))
            elif op == 'fill_holes':
                refined_mesh = self.fill_holes(refined_mesh, **kwargs.get('fill_holes', {}))
            elif op == 'improve_quality':
                refined_mesh = self.improve_quality(refined_mesh, **kwargs.get('improve_quality', {}))
            elif op == 'remesh':
                refined_mesh = self.remesh(refined_mesh, **kwargs.get('remesh', {}))
            else:
                print(f"Warning: Unknown operation '{op}', skipping")
        
        return refined_mesh
    
    def smooth(
        self,
        mesh: Mesh,
        method: Optional[str] = None,
        iterations: Optional[int] = None,
        fixed_vertices: Optional[np.ndarray] = None,
        vertex_weights: Optional[np.ndarray] = None
    ) -> Mesh:
        """Smooth mesh."""
        if method is not None:
            self.smoother.method = method
        if iterations is not None:
            self.smoother.iterations = iterations
        
        return self.smoother.smooth(mesh, fixed_vertices, vertex_weights)
    
    def subdivide(
        self,
        mesh: Mesh,
        method: Optional[str] = None,
        levels: Optional[int] = None,
        interpolate_attributes: bool = True
    ) -> Mesh:
        """Subdivide mesh."""
        if method is not None:
            self.subdivider.method = method
        if levels is not None:
            self.subdivider.levels = levels
        
        return self.subdivider.subdivide(mesh, interpolate_attributes)
    
    def simplify(
        self,
        mesh: Mesh,
        method: Optional[str] = None,
        target_face_count: Optional[int] = None,
        preserve_border: bool = True
    ) -> Mesh:
        """Simplify mesh."""
        if method is not None:
            self.simplifier.method = method
        if target_face_count is not None:
            self.simplifier.target_face_count = target_face_count
        
        return self.simplifier.simplify(mesh, preserve_border)
    
    def fill_holes(
        self,
        mesh: Mesh,
        max_hole_size: Optional[int] = None,
        smooth_filled: bool = True
    ) -> Mesh:
        """Fill holes in mesh."""
        if max_hole_size is not None:
            self.hole_filler.max_hole_size = max_hole_size
        
        return self.hole_filler.fill_holes(mesh, smooth_filled)
    
    def improve_quality(
        self,
        mesh: Mesh,
        target_min_angle: Optional[float] = None,
        target_max_angle: Optional[float] = None,
        iterations: int = 10
    ) -> Mesh:
        """
        Improve mesh quality by adjusting bad triangles.
        
        Args:
            mesh: Input mesh
            target_min_angle: Target minimum angle in degrees
            target_max_angle: Target maximum angle in degrees
            iterations: Number of improvement iterations
            
        Returns:
            Improved mesh
        """
        min_angle = target_min_angle or self.config.min_angle
        max_angle = target_max_angle or self.config.max_angle
        
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        for _ in range(iterations):
            improved = False
            
            # Check each face
            for i, face in enumerate(faces):
                v0, v1, v2 = vertices[face]
                
                # Compute angles
                a = v1 - v0
                b = v2 - v1
                c = v0 - v2
                
                angles = np.array([
                    self._angle_between(-a, c),
                    self._angle_between(-b, a),
                    self._angle_between(-c, b)
                ])
                
                angles_deg = np.degrees(angles)
                
                # Check if any angle is too small or too large
                if np.any(angles_deg < min_angle) or np.any(angles_deg > max_angle):
                    # Try to improve by edge flipping
                    improved = self._try_edge_flip(vertices, faces, i, min_angle, max_angle)
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        improved_mesh = Mesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        improved_mesh.compute_normals()
        
        return improved_mesh
    
    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors."""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.arccos(np.clip(dot / norm, -1.0, 1.0))
    
    def _try_edge_flip(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_idx: int,
        min_angle: float,
        max_angle: float
    ) -> bool:
        """Try to improve triangle by flipping an edge."""
        # This is a simplified implementation
        # In practice, you'd check Delaunay condition and quality metrics
        
        face = faces[face_idx]
        
        # Find adjacent face sharing an edge
        # (Simplified - actual implementation would find all adjacent faces)
        
        return False  # No flip performed in this simplified version
    
    def remesh(
        self,
        mesh: Mesh,
        target_edge_length: Optional[float] = None,
        iterations: Optional[int] = None
    ) -> Mesh:
        """
        Remesh to achieve uniform edge lengths.
        
        Args:
            mesh: Input mesh
            target_edge_length: Target edge length
            iterations: Number of remeshing iterations
            
        Returns:
            Remeshed mesh
        """
        target_edge_length = target_edge_length or self.config.target_edge_length
        iterations = iterations or self.config.remeshing_iterations
        
        # Use Open3D for remeshing
        o3d_mesh = mesh.to_o3d()
        
        # Estimate average edge length
        edges = o3d_mesh.get_non_manifold_edges()
        if len(edges) > 0:
            # Has non-manifold edges, fix first
            o3d_mesh.remove_non_manifold_edges()
        
        # Remesh
        remeshed = o3d_mesh.remesh(
            number_of_iterations=iterations,
            target_edge_length=target_edge_length
        )
        
        remeshed_mesh = Mesh(
            vertices=np.asarray(remeshed.vertices),
            faces=np.asarray(remeshed.triangles),
            vertex_normals=np.asarray(remeshed.vertex_normals) if remeshed.has_vertex_normals() else None,
            vertex_colors=np.asarray(remeshed.vertex_colors) if remeshed.has_vertex_colors() else None
        )
        
        return remeshed_mesh
    
    def optimize_mesh(
        self,
        mesh: Mesh,
        reference_mesh: Optional[Mesh] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Mesh:
        """
        Optimize mesh using energy minimization.
        
        Args:
            mesh: Mesh to optimize
            reference_mesh: Reference mesh for position constraint
            weights: Weights for different energy terms
            
        Returns:
            Optimized mesh
        """
        if weights is None:
            weights = {
                'laplacian': self.config.laplacian_weight,
                'position': self.config.position_weight,
                'normal': self.config.normal_weight,
                'edge': self.config.edge_weight
            }
        
        vertices = mesh.vertices.copy()
        faces = mesh.faces
        
        # Build Laplacian matrix
        num_vertices = len(vertices)
        L = self._build_cotangent_laplacian(vertices, faces)
        
        # Build energy system
        A = L.T @ L * weights['laplacian']
        b = np.zeros((num_vertices, 3))
        
        # Add position constraints if reference mesh provided
        if reference_mesh is not None and weights['position'] > 0:
            # Find corresponding vertices (simplified - assumes same topology)
            if len(reference_mesh.vertices) == num_vertices:
                A += sparse.eye(num_vertices) * weights['position']
                b += reference_mesh.vertices * weights['position']
        
        # Add normal constraints
        if weights['normal'] > 0 and mesh.vertex_normals is not None:
            # This is simplified - actual normal preservation is more complex
            pass
        
        # Solve for each coordinate
        new_vertices = np.zeros_like(vertices)
        for i in range(3):
            new_vertices[:, i] = spsolve(A, b[:, i])
        
        optimized_mesh = Mesh(
            vertices=new_vertices,
            faces=faces,
            vertex_normals=None,
            vertex_colors=mesh.vertex_colors
        )
        optimized_mesh.compute_normals()
        
        return optimized_mesh
    
    def _build_cotangent_laplacian(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> sparse.csr_matrix:
        """Build cotangent Laplacian matrix."""
        num_vertices = len(vertices)
        
        # Build adjacency
        I = []  # Row indices
        J = []  # Column indices
        V = []  # Values
        
        for face in faces:
            v0, v1, v2 = face
            
            # Compute cotangents for each angle
            p0, p1, p2 = vertices[v0], vertices[v1], vertices[v2]
            
            # Edge vectors
            e0 = p2 - p1
            e1 = p0 - p2
            e2 = p1 - p0
            
            # Compute cotangents
            cot0 = self._cotangent(p0, p1, p2)
            cot1 = self._cotangent(p1, p2, p0)
            cot2 = self._cotangent(p2, p0, p1)
            
            # Add contributions to Laplacian
            # Off-diagonal entries
            for (i, j, cot) in [(v1, v2, cot0), (v2, v0, cot1), (v0, v1, cot2)]:
                I.append(i)
                J.append(j)
                V.append(cot)
            
            # Diagonal entries (will be summed later)
            for v in [v0, v1, v2]:
                I.append(v)
                J.append(v)
                V.append(0.0)  # Placeholder
        
        # Create sparse matrix
        L = sparse.csr_matrix((V, (I, J)), shape=(num_vertices, num_vertices))
        
        # Compute diagonal as negative sum of row
        diag = -np.array(L.sum(axis=1)).flatten()
        L.setdiag(diag)
        
        return L
    
    def _cotangent(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray
    ) -> float:
        """Compute cotangent of angle at vertex a in triangle abc."""
        # Vectors
        v0 = b - a
        v1 = c - a
        
        # Dot product and cross product magnitude
        dot = np.dot(v0, v1)
        cross = np.linalg.norm(np.cross(v0, v1))
        
        return dot / (cross + 1e-8)
    
    def batch_refine(
        self,
        meshes: List[Mesh],
        operations: List[str],
        parallel: bool = True,
        num_workers: int = 4
    ) -> List[Mesh]:
        """
        Refine multiple meshes.
        
        Args:
            meshes: List of input meshes
            operations: List of operations to apply
            parallel: Whether to process in parallel
            num_workers: Number of parallel workers
            
        Returns:
            List of refined meshes
        """
        if not parallel or len(meshes) < 2:
            return [self.refine_mesh(mesh, operations) for mesh in meshes]
        
        # Parallel processing
        import multiprocessing as mp
        
        def refine_single(mesh_data):
            mesh_dict, ops = mesh_data
            mesh = Mesh(**mesh_dict)
            refined = self.refine_mesh(mesh, ops)
            return {
                'vertices': refined.vertices,
                'faces': refined.faces,
                'vertex_normals': refined.vertex_normals,
                'vertex_colors': refined.vertex_colors
            }
        
        # Prepare data
        mesh_data = []
        for mesh in meshes:
            mesh_dict = {
                'vertices': mesh.vertices,
                'faces': mesh.faces,
                'vertex_normals': mesh.vertex_normals,
                'vertex_colors': mesh.vertex_colors
            }
            mesh_data.append((mesh_dict, operations))
        
        # Process in parallel
        with mp.Pool(num_workers) as pool:
            results = pool.map(refine_single, mesh_data)
        
        # Convert back to Mesh objects
        refined_meshes = []
        for result in results:
            refined_meshes.append(Mesh(**result))
        
        return refined_meshes