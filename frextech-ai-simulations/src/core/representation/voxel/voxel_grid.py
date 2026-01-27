"""
Voxel grid representation for 3D data.
Includes sparse and dense voxel grids, octrees, and operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union, Set
import trimesh
import open3d as o3d
from scipy import ndimage
from scipy.spatial import KDTree
import numba
from numba import jit, prange
from dataclasses import dataclass, field
import json
import struct
import zlib
import time
import warnings


@dataclass
class VoxelGrid:
    """Voxel grid data structure."""
    resolution: Tuple[int, int, int]  # (depth, height, width)
    voxel_size: float  # Size of each voxel in world units
    origin: np.ndarray  # [3] world coordinates of grid origin (minimum corner)
    values: np.ndarray  # [D, H, W, C] voxel values (C channels)
    occupancy: Optional[np.ndarray] = None  # [D, H, W] occupancy mask
    
    @property
    def depth(self) -> int:
        return self.resolution[0]
    
    @property
    def height(self) -> int:
        return self.resolution[1]
    
    @property
    def width(self) -> int:
        return self.resolution[2]
    
    @property
    def channels(self) -> int:
        return self.values.shape[3] if len(self.values.shape) > 3 else 1
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get world bounds of grid."""
        min_bound = self.origin
        max_bound = self.origin + np.array(self.resolution) * self.voxel_size
        return min_bound, max_bound
    
    def world_to_grid(self, points: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            points: World coordinates [N, 3]
            
        Returns:
            Grid coordinates (float) [N, 3]
        """
        return (points - self.origin) / self.voxel_size
    
    def grid_to_world(self, grid_coords: np.ndarray) -> np.ndarray:
        """
        Convert grid coordinates to world coordinates.
        
        Args:
            grid_coords: Grid coordinates [N, 3]
            
        Returns:
            World coordinates [N, 3]
        """
        return self.origin + grid_coords * self.voxel_size
    
    def sample_at_points(self, points: np.ndarray, mode: str = 'trilinear') -> np.ndarray:
        """
        Sample voxel values at world points.
        
        Args:
            points: World coordinates [N, 3]
            mode: Interpolation mode ('nearest', 'trilinear')
            
        Returns:
            Sampled values [N, C]
        """
        # Convert to grid coordinates
        grid_coords = self.world_to_grid(points)
        
        if mode == 'nearest':
            # Nearest neighbor sampling
            indices = np.floor(grid_coords + 0.5).astype(int)
            
            # Clamp to grid bounds
            indices[:, 0] = np.clip(indices[:, 0], 0, self.depth - 1)
            indices[:, 1] = np.clip(indices[:, 1], 0, self.height - 1)
            indices[:, 2] = np.clip(indices[:, 2], 0, self.width - 1)
            
            # Sample values
            samples = self.values[indices[:, 0], indices[:, 1], indices[:, 2]]
            
        elif mode == 'trilinear':
            # Trilinear interpolation
            samples = self._trilinear_interpolate(grid_coords)
            
        else:
            raise ValueError(f"Unknown interpolation mode: {mode}")
        
        return samples
    
    def _trilinear_interpolate(self, grid_coords: np.ndarray) -> np.ndarray:
        """Trilinear interpolation."""
        N = grid_coords.shape[0]
        C = self.channels
        
        # Get fractional coordinates
        x = grid_coords[:, 0]
        y = grid_coords[:, 1]
        z = grid_coords[:, 2]
        
        # Get integer coordinates
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        z0 = np.floor(z).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1
        
        # Clamp to grid bounds
        x0 = np.clip(x0, 0, self.depth - 1)
        y0 = np.clip(y0, 0, self.height - 1)
        z0 = np.clip(z0, 0, self.width - 1)
        x1 = np.clip(x1, 0, self.depth - 1)
        y1 = np.clip(y1, 0, self.height - 1)
        z1 = np.clip(z1, 0, self.width - 1)
        
        # Get interpolation weights
        xd = x - x0
        yd = y - y0
        zd = z - z0
        
        # Get values at corners
        values = self.values
        c000 = values[x0, y0, z0]
        c001 = values[x0, y0, z1]
        c010 = values[x0, y1, z0]
        c011 = values[x0, y1, z1]
        c100 = values[x1, y0, z0]
        c101 = values[x1, y0, z1]
        c110 = values[x1, y1, z0]
        c111 = values[x1, y1, z1]
        
        # Interpolate along x
        c00 = c000 * (1 - xd[:, None]) + c100 * xd[:, None]
        c01 = c001 * (1 - xd[:, None]) + c101 * xd[:, None]
        c10 = c010 * (1 - xd[:, None]) + c110 * xd[:, None]
        c11 = c011 * (1 - xd[:, None]) + c111 * xd[:, None]
        
        # Interpolate along y
        c0 = c00 * (1 - yd[:, None]) + c10 * yd[:, None]
        c1 = c01 * (1 - yd[:, None]) + c11 * yd[:, None]
        
        # Interpolate along z
        samples = c0 * (1 - zd[:, None]) + c1 * zd[:, None]
        
        return samples
    
    def set_voxel(self, grid_coord: Tuple[int, int, int], value: np.ndarray):
        """
        Set value of a specific voxel.
        
        Args:
            grid_coord: (d, h, w) grid coordinate
            value: Value to set [C]
        """
        d, h, w = grid_coord
        
        if 0 <= d < self.depth and 0 <= h < self.height and 0 <= w < self.width:
            self.values[d, h, w] = value
            
            if self.occupancy is not None:
                self.occupancy[d, h, w] = 1.0
    
    def get_voxel(self, grid_coord: Tuple[int, int, int]) -> np.ndarray:
        """
        Get value of a specific voxel.
        
        Args:
            grid_coord: (d, h, w) grid coordinate
            
        Returns:
            Voxel value [C]
        """
        d, h, w = grid_coord
        
        if 0 <= d < self.depth and 0 <= h < self.height and 0 <= w < self.width:
            return self.values[d, h, w].copy()
        else:
            return np.zeros(self.channels)
    
    def get_occupied_voxels(self, threshold: float = 0.5) -> np.ndarray:
        """
        Get coordinates of occupied voxels.
        
        Args:
            threshold: Occupancy threshold
            
        Returns:
            Grid coordinates of occupied voxels [N, 3]
        """
        if self.occupancy is None:
            # Use values as occupancy
            if self.channels == 1:
                occupied_mask = self.values.squeeze() > threshold
            else:
                # Use alpha channel or magnitude
                occupied_mask = np.linalg.norm(self.values, axis=-1) > threshold
        else:
            occupied_mask = self.occupancy > threshold
        
        occupied_indices = np.where(occupied_mask)
        occupied_coords = np.stack(occupied_indices, axis=1)
        
        return occupied_coords
    
    def get_occupied_voxel_values(self) -> np.ndarray:
        """
        Get values of occupied voxels.
        
        Returns:
            Values of occupied voxels [N, C]
        """
        occupied_coords = self.get_occupied_voxels()
        values = self.values[
            occupied_coords[:, 0],
            occupied_coords[:, 1],
            occupied_coords[:, 2]
        ]
        
        return values
    
    def to_point_cloud(self, threshold: float = 0.5) -> np.ndarray:
        """
        Convert occupied voxels to point cloud.
        
        Args:
            threshold: Occupancy threshold
            
        Returns:
            Point cloud in world coordinates [N, 3]
        """
        occupied_coords = self.get_occupied_voxels(threshold)
        
        if len(occupied_coords) == 0:
            return np.zeros((0, 3))
        
        # Convert grid coordinates to world coordinates
        world_coords = self.grid_to_world(occupied_coords.astype(float) + 0.5)
        
        return world_coords
    
    def to_mesh(
        self,
        threshold: float = 0.5,
        smoothing_iterations: int = 2
    ) -> trimesh.Trimesh:
        """
        Convert voxel grid to mesh using marching cubes.
        
        Args:
            threshold: Iso-surface threshold
            smoothing_iterations: Number of smoothing iterations
            
        Returns:
            Triangle mesh
        """
        # Extract occupancy or values for meshing
        if self.occupancy is not None:
            volume = self.occupancy
        elif self.channels == 1:
            volume = self.values.squeeze()
        else:
            # Use magnitude
            volume = np.linalg.norm(self.values, axis=-1)
        
        # Apply smoothing if requested
        if smoothing_iterations > 0:
            for _ in range(smoothing_iterations):
                volume = ndimage.gaussian_filter(volume, sigma=1.0)
        
        # Marching cubes
        try:
            vertices, faces, normals, _ = skimage.measure.marching_cubes(
                volume,
                level=threshold,
                spacing=(self.voxel_size, self.voxel_size, self.voxel_size)
            )
            
            # Translate to world coordinates
            vertices = vertices + self.origin
            
            # Create mesh
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals
            )
            
            return mesh
            
        except Exception as e:
            print(f"Marching cubes failed: {e}")
            # Return empty mesh
            return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)))
    
    def save(self, path: str, compress: bool = True):
        """
        Save voxel grid to file.
        
        Args:
            path: File path
            compress: Whether to compress with gzip
        """
        data = {
            'resolution': self.resolution,
            'voxel_size': float(self.voxel_size),
            'origin': self.origin.tolist(),
            'values': self.values.tolist(),
            'occupancy': self.occupancy.tolist() if self.occupancy is not None else None,
            'dtype': str(self.values.dtype)
        }
        
        if compress:
            import gzip
            with gzip.open(path, 'wb') as f:
                f.write(json.dumps(data).encode('utf-8'))
        else:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    @classmethod
    def load(cls, path: str, compress: bool = True) -> 'VoxelGrid':
        """
        Load voxel grid from file.
        
        Args:
            path: File path
            compress: Whether file is compressed
            
        Returns:
            Loaded voxel grid
        """
        if compress:
            import gzip
            with gzip.open(path, 'rb') as f:
                data = json.loads(f.read().decode('utf-8'))
        else:
            with open(path, 'r') as f:
                data = json.load(f)
        
        # Convert data
        resolution = tuple(data['resolution'])
        voxel_size = data['voxel_size']
        origin = np.array(data['origin'])
        values = np.array(data['values'])
        
        if data['occupancy'] is not None:
            occupancy = np.array(data['occupancy'])
        else:
            occupancy = None
        
        return cls(
            resolution=resolution,
            voxel_size=voxel_size,
            origin=origin,
            values=values,
            occupancy=occupancy
        )


@dataclass
class SparseVoxelGrid:
    """Sparse voxel grid using hash table."""
    voxel_size: float
    origin: np.ndarray
    capacity: int = 1000000
    values: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    occupancy: Dict[Tuple[int, int, int], float] = field(default_factory=dict)
    
    def world_to_grid(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid coordinates."""
        return (points - self.origin) / self.voxel_size
    
    def grid_to_world(self, grid_coords: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        return self.origin + grid_coords * self.voxel_size
    
    def get_voxel_key(self, grid_coord: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert continuous grid coordinates to discrete voxel key."""
        return (
            int(np.floor(grid_coord[0])),
            int(np.floor(grid_coord[1])),
            int(np.floor(grid_coord[2]))
        )
    
    def insert_points(
        self,
        points: np.ndarray,
        values: np.ndarray,
        update_strategy: str = 'average'  # 'average', 'overwrite', 'max'
    ):
        """
        Insert points into sparse voxel grid.
        
        Args:
            points: World points [N, 3]
            values: Point values [N, C]
            update_strategy: How to update existing voxels
        """
        grid_coords = self.world_to_grid(points)
        
        for i, grid_coord in enumerate(grid_coords):
            key = self.get_voxel_key(grid_coord)
            
            if key in self.values:
                # Existing voxel
                if update_strategy == 'average':
                    # Average with existing value
                    count = self.occupancy.get(key, 0) + 1
                    self.values[key] = (self.values[key] * (count - 1) + values[i]) / count
                    self.occupancy[key] = count
                elif update_strategy == 'overwrite':
                    # Overwrite
                    self.values[key] = values[i]
                    self.occupancy[key] = 1.0
                elif update_strategy == 'max':
                    # Take maximum
                    self.values[key] = np.maximum(self.values[key], values[i])
                    self.occupancy[key] = max(self.occupancy.get(key, 0), 1.0)
            else:
                # New voxel
                self.values[key] = values[i]
                self.occupancy[key] = 1.0
    
    def sample_at_points(
        self,
        points: np.ndarray,
        radius: float = 1.5,
        max_neighbors: int = 8
    ) -> np.ndarray:
        """
        Sample values at points using nearest neighbors.
        
        Args:
            points: World points [N, 3]
            radius: Search radius in voxel units
            max_neighbors: Maximum neighbors to consider
            
        Returns:
            Sampled values [N, C]
        """
        grid_coords = self.world_to_grid(points)
        N = len(points)
        C = next(iter(self.values.values())).shape[0] if self.values else 1
        
        samples = np.zeros((N, C))
        
        # Build KDTree for occupied voxels
        if not self.values:
            return samples
        
        occupied_keys = list(self.values.keys())
        occupied_coords = np.array(occupied_keys)
        
        if len(occupied_coords) == 0:
            return samples
        
        kdtree = KDTree(occupied_coords)
        
        # Query for each point
        for i, grid_coord in enumerate(grid_coords):
            # Find nearest occupied voxels
            distances, indices = kdtree.query(
                grid_coord,
                k=min(max_neighbors, len(occupied_coords)),
                distance_upper_bound=radius
            )
            
            # Filter out infinite distances
            valid_mask = distances < np.inf
            if not np.any(valid_mask):
                continue
            
            valid_indices = indices[valid_mask]
            valid_distances = distances[valid_mask]
            
            # Inverse distance weighting
            weights = 1.0 / (valid_distances + 1e-8)
            weights = weights / weights.sum()
            
            # Weighted average of neighbor values
            weighted_sum = np.zeros(C)
            for j, idx in enumerate(valid_indices):
                key = tuple(occupied_coords[idx])
                weighted_sum += self.values[key] * weights[j]
            
            samples[i] = weighted_sum
        
        return samples
    
    def to_dense(
        self,
        resolution: Optional[Tuple[int, int, int]] = None,
        padding: int = 1
    ) -> VoxelGrid:
        """
        Convert sparse voxel grid to dense representation.
        
        Args:
            resolution: Target resolution (optional)
            padding: Padding around occupied voxels
            
        Returns:
            Dense voxel grid
        """
        if not self.values:
            # Empty grid
            if resolution is None:
                resolution = (1, 1, 1)
            return VoxelGrid(
                resolution=resolution,
                voxel_size=self.voxel_size,
                origin=self.origin,
                values=np.zeros((*resolution, 1))
            )
        
        # Get bounds of occupied voxels
        occupied_keys = list(self.values.keys())
        occupied_coords = np.array(occupied_keys)
        
        min_coord = occupied_coords.min(axis=0) - padding
        max_coord = occupied_coords.max(axis=0) + padding + 1
        
        # Determine resolution
        if resolution is None:
            resolution = tuple((max_coord - min_coord).astype(int))
        else:
            # Ensure resolution covers occupied area
            required_res = max_coord - min_coord
            resolution = tuple(max(r, req) for r, req in zip(resolution, required_res))
        
        # Create dense grid
        values = np.zeros((*resolution, next(iter(self.values.values())).shape[0]))
        occupancy = np.zeros(resolution)
        
        # Fill with sparse values
        offset = min_coord
        
        for key, value in self.values.items():
            idx = np.array(key) - offset
            if all(0 <= idx[i] < resolution[i] for i in range(3)):
                values[tuple(idx)] = value
                occupancy[tuple(idx)] = self.occupancy.get(key, 1.0)
        
        # Update origin to match dense grid
        new_origin = self.origin + offset * self.voxel_size
        
        return VoxelGrid(
            resolution=resolution,
            voxel_size=self.voxel_size,
            origin=new_origin,
            values=values,
            occupancy=occupancy
        )


class OctreeNode:
    """Octree node for sparse hierarchical representation."""
    
    def __init__(
        self,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        depth: int = 0,
        max_depth: int = 8,
        min_points: int = 8,
        value: Optional[np.ndarray] = None
    ):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.center = (bounds_min + bounds_max) / 2
        self.extent = bounds_max - bounds_min
        self.depth = depth
        self.max_depth = max_depth
        self.min_points = min_points
        
        self.value = value
        self.points = None
        self.values = None
        self.children = None
        self.is_leaf = True
    
    def insert_points(
        self,
        points: np.ndarray,
        point_values: np.ndarray,
        update_strategy: str = 'average'
    ):
        """
        Insert points into octree.
        
        Args:
            points: Points [N, 3]
            point_values: Point values [N, C]
            update_strategy: Update strategy for node values
        """
        if len(points) == 0:
            return
        
        if self.is_leaf:
            # First insertion or leaf node
            if self.points is None:
                self.points = points
                self.values = point_values
            else:
                # Add to existing points
                self.points = np.vstack([self.points, points])
                self.values = np.vstack([self.values, point_values])
            
            # Check if should split
            if (self.depth < self.max_depth and 
                len(self.points) > self.min_points and
                np.any(self.extent > 1e-6)):
                
                self.split()
                
                # Redistribute points to children
                for child in self.children:
                    # Find points in child's bounds
                    in_child = np.all(
                        (self.points >= child.bounds_min) & 
                        (self.points <= child.bounds_max),
                        axis=1
                    )
                    
                    if np.any(in_child):
                        child.insert_points(
                            self.points[in_child],
                            self.values[in_child],
                            update_strategy
                        )
                
                # Clear leaf data
                self.points = None
                self.values = None
        
        elif not self.is_leaf:
            # Internal node, insert into children
            for child in self.children:
                # Find points in child's bounds
                in_child = np.all(
                    (points >= child.bounds_min) & 
                    (points <= child.bounds_max),
                    axis=1
                )
                
                if np.any(in_child):
                    child.insert_points(
                        points[in_child],
                        point_values[in_child],
                        update_strategy
                    )
    
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
                        self.depth + 1,
                        self.max_depth,
                        self.min_points
                    )
                    self.children.append(child)
    
    def compute_node_values(self, strategy: str = 'average'):
        """
        Compute node values from children or points.
        
        Args:
            strategy: 'average', 'max', 'min'
        """
        if self.is_leaf and self.points is not None:
            # Leaf with points
            if strategy == 'average':
                self.value = np.mean(self.values, axis=0)
            elif strategy == 'max':
                self.value = np.max(self.values, axis=0)
            elif strategy == 'min':
                self.value = np.min(self.values, axis=0)
        elif not self.is_leaf:
            # Internal node, compute from children
            child_values = []
            for child in self.children:
                child.compute_node_values(strategy)
                if child.value is not None:
                    child_values.append(child.value)
            
            if child_values:
                child_values = np.stack(child_values)
                if strategy == 'average':
                    self.value = np.mean(child_values, axis=0)
                elif strategy == 'max':
                    self.value = np.max(child_values, axis=0)
                elif strategy == 'min':
                    self.value = np.min(child_values, axis=0)
    
    def query_point(
        self,
        point: np.ndarray,
        max_depth: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Query value at point.
        
        Args:
            point: Query point
            max_depth: Maximum depth to traverse
            
        Returns:
            Node value or None
        """
        # Check if point is in bounds
        if not np.all((point >= self.bounds_min) & (point <= self.bounds_max)):
            return None
        
        if self.is_leaf:
            return self.value
        
        if max_depth is not None and self.depth >= max_depth:
            return self.value
        
        # Find child containing point
        for child in self.children:
            if np.all((point >= child.bounds_min) & (point <= child.bounds_max)):
                return child.query_point(point, max_depth)
        
        return None
    
    def query_points(
        self,
        points: np.ndarray,
        max_depth: Optional[int] = None
    ) -> np.ndarray:
        """
        Query values at multiple points.
        
        Args:
            points: Query points [N, 3]
            max_depth: Maximum depth to traverse
            
        Returns:
            Values [N, C]
        """
        N = len(points)
        C = self.value.shape[0] if self.value is not None else 1
        
        values = np.zeros((N, C))
        
        for i, point in enumerate(points):
            val = self.query_point(point, max_depth)
            if val is not None:
                values[i] = val
        
        return values


class VoxelGridOps:
    """Operations on voxel grids."""
    
    @staticmethod
    def create_from_point_cloud(
        points: np.ndarray,
        point_values: np.ndarray,
        voxel_size: float,
        resolution: Optional[Tuple[int, int, int]] = None,
        origin: Optional[np.ndarray] = None,
        occupancy_threshold: float = 0.5
    ) -> VoxelGrid:
        """
        Create voxel grid from point cloud.
        
        Args:
            points: Point cloud [N, 3]
            point_values: Point values [N, C]
            voxel_size: Size of each voxel
            resolution: Grid resolution (optional)
            origin: Grid origin (optional)
            occupancy_threshold: Occupancy threshold
            
        Returns:
            Voxel grid
        """
        if len(points) == 0:
            raise ValueError("Point cloud is empty")
        
        # Determine bounds
        points_min = points.min(axis=0)
        points_max = points.max(axis=0)
        
        # Determine origin
        if origin is None:
            origin = points_min - voxel_size  # Add padding
        
        # Determine resolution
        if resolution is None:
            bounds_size = points_max - points_min + 2 * voxel_size  # Add padding
            resolution = tuple(np.ceil(bounds_size / voxel_size).astype(int))
        
        # Create empty grid
        C = point_values.shape[1] if len(point_values.shape) > 1 else 1
        values = np.zeros((*resolution, C))
        occupancy = np.zeros(resolution)
        counts = np.zeros(resolution)
        
        # Convert points to grid coordinates
        grid_coords = (points - origin) / voxel_size
        indices = np.floor(grid_coords).astype(int)
        
        # Clamp indices to grid bounds
        for i in range(3):
            indices[:, i] = np.clip(indices[:, i], 0, resolution[i] - 1)
        
        # Accumulate values and counts
        for idx, val in zip(indices, point_values):
            key = tuple(idx)
            if C == 1:
                values[key] += val
            else:
                values[key] += val
            counts[key] += 1
        
        # Average values
        mask = counts > 0
        if C == 1:
            values[mask] /= counts[mask][:, np.newaxis]
        else:
            for c in range(C):
                values[..., c][mask] /= counts[mask]
        
        # Compute occupancy
        occupancy[mask] = 1.0
        
        return VoxelGrid(
            resolution=resolution,
            voxel_size=voxel_size,
            origin=origin,
            values=values,
            occupancy=occupancy
        )
    
    @staticmethod
    def downsample(
        grid: VoxelGrid,
        factor: int = 2,
        mode: str = 'average'
    ) -> VoxelGrid:
        """
        Downsample voxel grid.
        
        Args:
            grid: Input voxel grid
            factor: Downsampling factor
            mode: Downsampling mode ('average', 'max', 'min')
            
        Returns:
            Downsampled voxel grid
        """
        D, H, W = grid.depth, grid.height, grid.width
        C = grid.channels
        
        # Compute new resolution
        new_D = (D + factor - 1) // factor
        new_H = (H + factor - 1) // factor
        new_W = (W + factor - 1) // factor
        
        # Create new values array
        new_values = np.zeros((new_D, new_H, new_W, C))
        
        if mode == 'average':
            # Average pooling
            for d in range(new_D):
                for h in range(new_H):
                    for w in range(new_W):
                        d_start = d * factor
                        d_end = min((d + 1) * factor, D)
                        h_start = h * factor
                        h_end = min((h + 1) * factor, H)
                        w_start = w * factor
                        w_end = min((w + 1) * factor, W)
                        
                        patch = grid.values[d_start:d_end, h_start:h_end, w_start:w_end]
                        new_values[d, h, w] = np.mean(patch, axis=(0, 1, 2))
        
        elif mode == 'max':
            # Max pooling
            for d in range(new_D):
                for h in range(new_H):
                    for w in range(new_W):
                        d_start = d * factor
                        d_end = min((d + 1) * factor, D)
                        h_start = h * factor
                        h_end = min((h + 1) * factor, H)
                        w_start = w * factor
                        w_end = min((w + 1) * factor, W)
                        
                        patch = grid.values[d_start:d_end, h_start:h_end, w_start:w_end]
                        new_values[d, h, w] = np.max(patch, axis=(0, 1, 2))
        
        elif mode == 'min':
            # Min pooling
            for d in range(new_D):
                for h in range(new_H):
                    for w in range(new_W):
                        d_start = d * factor
                        d_end = min((d + 1) * factor, D)
                        h_start = h * factor
                        h_end = min((h + 1) * factor, H)
                        w_start = w * factor
                        w_end = min((w + 1) * factor, W)
                        
                        patch = grid.values[d_start:d_end, h_start:h_end, w_start:w_end]
                        new_values[d, h, w] = np.min(patch, axis=(0, 1, 2))
        
        # Downsample occupancy if present
        new_occupancy = None
        if grid.occupancy is not None:
            new_occupancy = np.zeros((new_D, new_H, new_W))
            
            for d in range(new_D):
                for h in range(new_H):
                    for w in range(new_W):
                        d_start = d * factor
                        d_end = min((d + 1) * factor, D)
                        h_start = h * factor
                        h_end = min((h + 1) * factor, H)
                        w_start = w * factor
                        w_end = min((w + 1) * factor, W)
                        
                        patch = grid.occupancy[d_start:d_end, h_start:h_end, w_start:w_end]
                        new_occupancy[d, h, w] = np.max(patch)
        
        return VoxelGrid(
            resolution=(new_D, new_H, new_W),
            voxel_size=grid.voxel_size * factor,
            origin=grid.origin,
            values=new_values,
            occupancy=new_occupancy
        )
    
    @staticmethod
    def upsample(
        grid: VoxelGrid,
        factor: int = 2,
        mode: str = 'nearest'
    ) -> VoxelGrid:
        """
        Upsample voxel grid.
        
        Args:
            grid: Input voxel grid
            factor: Upsampling factor
            mode: Interpolation mode ('nearest', 'trilinear')
            
        Returns:
            Upsampled voxel grid
        """
        D, H, W = grid.depth, grid.height, grid.width
        C = grid.channels
        
        new_D = D * factor
        new_H = H * factor
        new_W = W * factor
        
        if mode == 'nearest':
            # Nearest neighbor upsampling
            new_values = np.zeros((new_D, new_H, new_W, C))
            
            for d in range(new_D):
                for h in range(new_H):
                    for w in range(new_W):
                        src_d = d // factor
                        src_h = h // factor
                        src_w = w // factor
                        new_values[d, h, w] = grid.values[src_d, src_h, src_w]
        
        elif mode == 'trilinear':
            # Trilinear interpolation
            # Use scipy for interpolation
            from scipy.ndimage import zoom
            
            new_values = np.zeros((new_D, new_H, new_W, C))
            
            for c in range(C):
                channel_data = grid.values[..., c]
                zoom_factors = (factor, factor, factor)
                interpolated = zoom(channel_data, zoom_factors, order=1)
                new_values[..., c] = interpolated
        
        # Upsample occupancy if present
        new_occupancy = None
        if grid.occupancy is not None:
            if mode == 'nearest':
                new_occupancy = np.zeros((new_D, new_H, new_W))
                for d in range(new_D):
                    for h in range(new_H):
                        for w in range(new_W):
                            src_d = d // factor
                            src_h = h // factor
                            src_w = w // factor
                            new_occupancy[d, h, w] = grid.occupancy[src_d, src_h, src_w]
            else:
                from scipy.ndimage import zoom
                new_occupancy = zoom(grid.occupancy, (factor, factor, factor), order=1)
        
        return VoxelGrid(
            resolution=(new_D, new_H, new_W),
            voxel_size=grid.voxel_size / factor,
            origin=grid.origin,
            values=new_values,
            occupancy=new_occupancy
        )
    
    @staticmethod
    def apply_filter(
        grid: VoxelGrid,
        filter_type: str = 'gaussian',
        sigma: float = 1.0,
        radius: int = 1
    ) -> VoxelGrid:
        """
        Apply filter to voxel grid.
        
        Args:
            grid: Input voxel grid
            filter_type: Type of filter ('gaussian', 'median', 'bilateral')
            sigma: Sigma for Gaussian filter
            radius: Radius for median filter
            
        Returns:
            Filtered voxel grid
        """
        from scipy.ndimage import gaussian_filter, median_filter
        
        filtered_values = grid.values.copy()
        C = grid.channels
        
        if filter_type == 'gaussian':
            # Apply Gaussian filter to each channel
            for c in range(C):
                filtered_values[..., c] = gaussian_filter(
                    grid.values[..., c],
                    sigma=sigma
                )
        
        elif filter_type == 'median':
            # Apply median filter to each channel
            for c in range(C):
                filtered_values[..., c] = median_filter(
                    grid.values[..., c],
                    size=2 * radius + 1
                )
        
        elif filter_type == 'bilateral':
            # Simplified bilateral filter
            # Note: Full bilateral filter is computationally expensive for 3D
            warnings.warn("Bilateral filter is simplified and may be slow for large grids")
            
            for c in range(C):
                channel_data = grid.values[..., c]
                filtered_channel = np.zeros_like(channel_data)
                
                D, H, W = channel_data.shape
                
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            # Define neighborhood
                            d_min = max(0, d - radius)
                            d_max = min(D, d + radius + 1)
                            h_min = max(0, h - radius)
                            h_max = min(H, h + radius + 1)
                            w_min = max(0, w - radius)
                            w_max = min(W, w + radius + 1)
                            
                            # Extract neighborhood
                            neighborhood = channel_data[d_min:d_max, h_min:h_max, w_min:w_max]
                            center_val = channel_data[d, h, w]
                            
                            # Spatial weights (Gaussian)
                            d_idx, h_idx, w_idx = np.meshgrid(
                                np.arange(d_min, d_max) - d,
                                np.arange(h_min, h_max) - h,
                                np.arange(w_min, w_max) - w,
                                indexing='ij'
                            )
                            
                            spatial_dist = np.sqrt(d_idx**2 + h_idx**2 + w_idx**2)
                            spatial_weights = np.exp(-spatial_dist**2 / (2 * sigma**2))
                            
                            # Range weights
                            range_dist = np.abs(neighborhood - center_val)
                            range_weights = np.exp(-range_dist**2 / (2 * sigma**2))
                            
                            # Combined weights
                            weights = spatial_weights * range_weights
                            
                            # Weighted average
                            if weights.sum() > 0:
                                filtered_channel[d, h, w] = np.sum(neighborhood * weights) / weights.sum()
                            else:
                                filtered_channel[d, h, w] = center_val
                
                filtered_values[..., c] = filtered_channel
        
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Filter occupancy if present
        filtered_occupancy = None
        if grid.occupancy is not None:
            if filter_type == 'gaussian':
                filtered_occupancy = gaussian_filter(grid.occupancy, sigma=sigma)
            elif filter_type == 'median':
                filtered_occupancy = median_filter(grid.occupancy, size=2 * radius + 1)
            else:
                filtered_occupancy = grid.occupancy.copy()
        
        return VoxelGrid(
            resolution=grid.resolution,
            voxel_size=grid.voxel_size,
            origin=grid.origin,
            values=filtered_values,
            occupancy=filtered_occupancy
        )
    
    @staticmethod
    def compute_gradients(grid: VoxelGrid) -> VoxelGrid:
        """
        Compute gradients of voxel grid.
        
        Args:
            grid: Input voxel grid
            
        Returns:
            Voxel grid with gradient vectors
        """
        from scipy.ndimage import sobel
        
        D, H, W = grid.depth, grid.height, grid.width
        C = grid.channels
        
        # Compute gradients for each channel
        gradients = np.zeros((D, H, W, C, 3))  # [D, H, W, C, 3]
        
        for c in range(C):
            channel_data = grid.values[..., c]
            
            # Sobel filters for gradient
            grad_x = sobel(channel_data, axis=2)  # x gradient
            grad_y = sobel(channel_data, axis=1)  # y gradient
            grad_z = sobel(channel_data, axis=0)  # z gradient
            
            gradients[..., c, 0] = grad_x
            gradients[..., c, 1] = grad_y
            gradients[..., c, 2] = grad_z
        
        # For scalar fields, compute magnitude
        if C == 1:
            magnitude = np.linalg.norm(gradients.squeeze(axis=3), axis=-1)
            gradients = gradients.squeeze(axis=3)  # [D, H, W, 3]
        else:
            # For multi-channel, we might want per-channel gradients
            # or combined gradient magnitude
            pass
        
        # Create new grid with gradients
        # Note: This changes the channel dimension
        return VoxelGrid(
            resolution=grid.resolution,
            voxel_size=grid.voxel_size,
            origin=grid.origin,
            values=gradients,
            occupancy=grid.occupancy
        )
    
    @staticmethod
    def compute_sdf_from_occupancy(
        grid: VoxelGrid,
        inside_value: float = -1.0,
        outside_value: float = 1.0
    ) -> VoxelGrid:
        """
        Compute signed distance function from occupancy grid.
        
        Args:
            grid: Input occupancy grid (values assumed to be occupancy)
            inside_value: Value inside objects
            outside_value: Value outside objects
            
        Returns:
            SDF voxel grid
        """
        from scipy.ndimage import distance_transform_edt
        
        if grid.channels != 1:
            raise ValueError("SDF computation requires single-channel occupancy grid")
        
        # Extract occupancy
        occupancy = grid.values.squeeze()
        
        # Compute distance transform
        # For points outside objects (occupancy == 0)
        outside_dist = distance_transform_edt(occupancy == 0)
        
        # For points inside objects (occupancy > 0.5)
        inside_dist = distance_transform_edt(occupancy > 0.5)
        
        # Combine with signs
        sdf = outside_dist - inside_dist
        
        # Normalize to [inside_value, outside_value] range
        sdf_min = sdf.min()
        sdf_max = sdf.max()
        
        if sdf_max > sdf_min:
            sdf_normalized = (sdf - sdf_min) / (sdf_max - sdf_min)
            sdf_normalized = sdf_normalized * (outside_value - inside_value) + inside_value
        else:
            sdf_normalized = np.zeros_like(sdf)
        
        # Create SDF grid
        return VoxelGrid(
            resolution=grid.resolution,
            voxel_size=grid.voxel_size,
            origin=grid.origin,
            values=sdf_normalized[..., np.newaxis],
            occupancy=grid.occupancy
        )


class NeuralVoxelField(nn.Module):
    """
    Neural voxel field using MLP for continuous representation.
    
    Learns a continuous function f(x) -> value that can be queried
    at arbitrary resolutions.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 1,
        num_layers: int = 8,
        positional_encoding: bool = True,
        encoding_freq: int = 10,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.positional_encoding = positional_encoding
        self.encoding_freq = encoding_freq
        self.device = device
        
        # Positional encoding
        if positional_encoding:
            self.encoding_dim = input_dim * (2 * encoding_freq + 1)
            self.freq_bands = 2.0 ** torch.arange(encoding_freq, device=device)
        else:
            self.encoding_dim = input_dim
        
        # MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.encoding_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers).to(device)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def positional_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.
        
        Args:
            x: Input coordinates [..., input_dim]
            
        Returns:
            Encoded features [..., encoding_dim]
        """
        if not self.positional_encoding:
            return x
        
        # Compute sinusoidal encodings
        encodings = [x]
        
        for freq in self.freq_bands:
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))
        
        return torch.cat(encodings, dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input coordinates [..., input_dim]
            
        Returns:
            Output values [..., output_dim]
        """
        # Positional encoding
        if self.positional_encoding:
            x = self.positional_encode(x)
        
        # MLP
        return self.mlp(x)
    
    def train_on_voxel_grid(
        self,
        grid: VoxelGrid,
        batch_size: int = 1024,
        num_epochs: int = 100,
        lr: float = 1e-3,
        validation_split: float = 0.1
    ):
        """
        Train neural field on voxel grid data.
        
        Args:
            grid: Voxel grid to learn
            batch_size: Training batch size
            num_epochs: Number of training epochs
            lr: Learning rate
            validation_split: Fraction of data for validation
        """
        # Get occupied voxel coordinates and values
        occupied_coords = grid.get_occupied_voxels()
        occupied_values = grid.get_occupied_voxel_values()
        
        if len(occupied_coords) == 0:
            print("Warning: No occupied voxels to train on")
            return
        
        # Convert to normalized coordinates [0, 1]
        coords_norm = occupied_coords.astype(np.float32)
        coords_norm[:, 0] /= grid.depth - 1 if grid.depth > 1 else 1
        coords_norm[:, 1] /= grid.height - 1 if grid.height > 1 else 1
        coords_norm[:, 2] /= grid.width - 1 if grid.width > 1 else 1
        
        # Convert to tensors
        coords_tensor = torch.from_numpy(coords_norm).float().to(self.device)
        values_tensor = torch.from_numpy(occupied_values).float().to(self.device)
        
        # Split into train/validation
        num_samples = len(coords_tensor)
        num_val = int(num_samples * validation_split)
        num_train = num_samples - num_val
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        coords_train = coords_tensor[train_indices]
        values_train = values_tensor[train_indices]
        coords_val = coords_tensor[val_indices]
        values_val = values_tensor[val_indices]
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_loss = 0.0
            
            for i in range(0, num_train, batch_size):
                batch_coords = coords_train[i:i+batch_size]
                batch_values = values_train[i:i+batch_size]
                
                optimizer.zero_grad()
                pred_values = self.forward(batch_coords)
                loss = F.mse_loss(pred_values, batch_values)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_coords)
            
            train_loss /= num_train
            
            # Validation
            self.eval()
            with torch.no_grad():
                val_loss = 0.0
                
                for i in range(0, num_val, batch_size):
                    batch_coords = coords_val[i:i+batch_size]
                    batch_values = values_val[i:i+batch_size]
                    
                    pred_values = self.forward(batch_coords)
                    loss = F.mse_loss(pred_values, batch_values)
                    val_loss += loss.item() * len(batch_coords)
                
                val_loss /= num_val
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.state_dict().copy()
        
        # Load best model
        if 'best_state' in locals():
            self.load_state_dict(best_state)
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    def sample_grid(
        self,
        resolution: Tuple[int, int, int],
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        batch_size: int = 4096
    ) -> VoxelGrid:
        """
        Sample neural field on a regular grid.
        
        Args:
            resolution: Grid resolution (D, H, W)
            bounds_min: Minimum bounds
            bounds_max: Maximum bounds
            batch_size: Batch size for inference
            
        Returns:
            Sampled voxel grid
        """
        D, H, W = resolution
        
        # Create coordinate grid
        z_coords = np.linspace(bounds_min[0], bounds_max[0], D)
        y_coords = np.linspace(bounds_min[1], bounds_max[1], H)
        x_coords = np.linspace(bounds_min[2], bounds_max[2], W)
        
        grid_z, grid_y, grid_x = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
        grid_coords = np.stack([grid_z, grid_y, grid_x], axis=-1)  # [D, H, W, 3]
        
        # Normalize to [0, 1]
        grid_coords_norm = (grid_coords - bounds_min) / (bounds_max - bounds_min)
        
        # Reshape for batch processing
        coords_flat = grid_coords_norm.reshape(-1, 3)
        num_samples = len(coords_flat)
        
        # Convert to tensor
        coords_tensor = torch.from_numpy(coords_flat).float().to(self.device)
        
        # Sample in batches
        values_flat = np.zeros((num_samples, self.output_dim))
        
        self.eval()
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_coords = coords_tensor[i:i+batch_size]
                batch_values = self.forward(batch_coords)
                values_flat[i:i+batch_size] = batch_values.cpu().numpy()
        
        # Reshape back to grid
        values_grid = values_flat.reshape(D, H, W, self.output_dim)
        
        # Create voxel grid
        voxel_size = np.array([
            (bounds_max[0] - bounds_min[0]) / (D - 1) if D > 1 else 1.0,
            (bounds_max[1] - bounds_min[1]) / (H - 1) if H > 1 else 1.0,
            (bounds_max[2] - bounds_min[2]) / (W - 1) if W > 1 else 1.0
        ]).mean()
        
        return VoxelGrid(
            resolution=resolution,
            voxel_size=voxel_size,
            origin=bounds_min,
            values=values_grid
        )