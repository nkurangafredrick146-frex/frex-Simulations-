"""
Ray Sampling for NeRF.

This module implements ray sampling strategies for Neural Radiance Fields,
including hierarchical sampling, importance sampling, and different
sampling distributions.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
import math
from enum import Enum


class SamplingStrategy(Enum):
    """Enumeration of ray sampling strategies."""
    UNIFORM = "uniform"           # Uniform sampling along ray
    STRATIFIED = "stratified"     # Stratified sampling (jittered uniform)
    IMPORTANCE = "importance"     # Importance sampling based on density
    HIERARCHICAL = "hierarchical" # Hierarchical (coarse-to-fine) sampling
    INVERSE_SPHERE = "inverse_sphere"  # Inverse sphere for unbounded scenes
    NEAR_FAR = "near_far"         # Near-far plane sampling
    EXPONENTIAL = "exponential"   # Exponential sampling (for unbounded)


@dataclass
class SamplingConfig:
    """Configuration for ray sampling."""
    
    # Basic sampling
    near: float = 0.0
    far: float = 1.0
    num_coarse_samples: int = 64
    num_fine_samples: int = 128
    num_importance_samples: int = 64
    
    # Sampling strategy
    strategy: Union[str, SamplingStrategy] = SamplingStrategy.HIERARCHICAL
    perturb: bool = True  # Whether to jitter sample positions
    linear_disparity: bool = False  # Sample linearly in disparity instead of depth
    
    # For inverse sphere parameterization
    inverse_sphere_bg: bool = False
    sphere_radius: float = 1.0
    sphere_near: float = 0.0
    sphere_far: float = 1.0
    
    # For exponential sampling
    exponential_lambda: float = 1.0
    
    # For importance sampling
    importance_weight_threshold: float = 0.01
    max_importance_samples: int = 256
    
    # For near-far plane estimation
    auto_near_far: bool = False
    near_percentile: float = 0.01
    far_percentile: float = 0.99
    
    # Chunking
    chunk_size: int = 65536  # Process rays in chunks to avoid OOM
    
    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.strategy, str):
            self.strategy = SamplingStrategy(self.strategy.lower())
        
        if self.near >= self.far:
            raise ValueError(f"near ({self.near}) must be less than far ({self.far})")
        
        if self.num_coarse_samples <= 0:
            raise ValueError(f"num_coarse_samples must be positive, got {self.num_coarse_samples}")
        
        if self.num_fine_samples < 0:
            raise ValueError(f"num_fine_samples must be non-negative, got {self.num_fine_samples}")
        
        if self.num_importance_samples < 0:
            raise ValueError(f"num_importance_samples must be non-negative, got {self.num_importance_samples}")
    
    def total_samples(self, include_importance: bool = True) -> int:
        """Get total number of samples per ray."""
        total = self.num_coarse_samples
        if include_importance and self.strategy == SamplingStrategy.HIERARCHICAL:
            total += self.num_fine_samples
        if include_importance and self.strategy == SamplingStrategy.IMPORTANCE:
            total += self.num_importance_samples
        return total


@dataclass
class RaySamples:
    """Samples along a ray."""
    
    # Core data
    positions: torch.Tensor  # [N, S, 3] or [N, S1+S2, 3]
    view_directions: torch.Tensor  # [N, S, 3] or [N, S1+S2, 3]
    depths: torch.Tensor  # [N, S] or [N, S1+S2]
    deltas: torch.Tensor  # [N, S] or [N, S1+S2] distance between samples
    
    # Metadata
    origins: torch.Tensor  # [N, 3] ray origins
    directions: torch.Tensor  # [N, 3] ray directions
    near: torch.Tensor  # [N] or scalar
    far: torch.Tensor  # [N] or scalar
    
    # Additional info
    sample_type: Optional[torch.Tensor] = None  # 0: coarse, 1: fine, 2: importance
    weights: Optional[torch.Tensor] = None  # [N, S] sample weights
    valid_mask: Optional[torch.Tensor] = None  # [N, S] valid sample mask
    
    def __post_init__(self):
        """Validate shapes."""
        N = self.positions.shape[0]
        S = self.positions.shape[1]
        
        assert self.view_directions.shape == (N, S, 3), \
            f"view_directions shape {self.view_directions.shape} != ({N}, {S}, 3)"
        
        assert self.depths.shape == (N, S), \
            f"depths shape {self.depths.shape} != ({N}, {S})"
        
        assert self.deltas.shape == (N, S), \
            f"deltas shape {self.deltas.shape} != ({N}, {S})"
        
        assert self.origins.shape == (N, 3), \
            f"origins shape {self.origins.shape} != ({N}, 3)"
        
        assert self.directions.shape == (N, 3), \
            f"directions shape {self.directions.shape} != ({N}, 3)"
    
    @property
    def num_rays(self) -> int:
        """Get number of rays."""
        return self.positions.shape[0]
    
    @property
    def num_samples(self) -> int:
        """Get number of samples per ray."""
        return self.positions.shape[1]
    
    @property
    def device(self) -> torch.device:
        """Get device of samples."""
        return self.positions.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get data type of samples."""
        return self.positions.dtype
    
    def split_by_type(self) -> Dict[str, RaySamples]:
        """Split samples by type if sample_type is available."""
        if self.sample_type is None:
            return {'all': self}
        
        result = {}
        unique_types = torch.unique(self.sample_type)
        
        for type_val in unique_types:
            mask = self.sample_type == type_val
            type_name = {
                0: 'coarse',
                1: 'fine',
                2: 'importance'
            }.get(type_val.item(), f'type_{type_val.item()}')
            
            # For each ray, get samples of this type
            type_positions = []
            type_view_directions = []
            type_depths = []
            type_deltas = []
            type_weights = []
            
            for i in range(self.num_rays):
                ray_mask = mask[i]
                if ray_mask.any():
                    type_positions.append(self.positions[i, ray_mask])
                    type_view_directions.append(self.view_directions[i, ray_mask])
                    type_depths.append(self.depths[i, ray_mask])
                    type_deltas.append(self.deltas[i, ray_mask])
                    if self.weights is not None:
                        type_weights.append(self.weights[i, ray_mask])
            
            # Create new RaySamples object
            if type_positions:  # Check if we have any samples
                result[type_name] = RaySamples(
                    positions=torch.stack(type_positions),
                    view_directions=torch.stack(type_view_directions),
                    depths=torch.stack(type_depths),
                    deltas=torch.stack(type_deltas),
                    origins=self.origins,
                    directions=self.directions,
                    near=self.near,
                    far=self.far,
                    sample_type=torch.full((self.num_rays, len(type_positions[0])), 
                                          type_val, device=self.device),
                    weights=torch.stack(type_weights) if self.weights is not None else None,
                )
        
        return result
    
    def merge(self, other: RaySamples) -> RaySamples:
        """Merge with another RaySamples object (same rays)."""
        if self.num_rays != other.num_rays:
            raise ValueError(f"Cannot merge RaySamples with different num_rays: "
                           f"{self.num_rays} != {other.num_rays}")
        
        # Concatenate along sample dimension
        positions = torch.cat([self.positions, other.positions], dim=1)
        view_directions = torch.cat([self.view_directions, other.view_directions], dim=1)
        depths = torch.cat([self.depths, other.depths], dim=1)
        deltas = torch.cat([self.deltas, other.deltas], dim=1)
        
        # Merge sample types
        if self.sample_type is not None and other.sample_type is not None:
            sample_type = torch.cat([self.sample_type, other.sample_type], dim=1)
        elif self.sample_type is not None:
            sample_type = torch.cat([
                self.sample_type,
                torch.full(other.positions.shape[:2], -1, device=self.device)
            ], dim=1)
        elif other.sample_type is not None:
            sample_type = torch.cat([
                torch.full(self.positions.shape[:2], -1, device=self.device),
                other.sample_type
            ], dim=1)
        else:
            sample_type = None
        
        # Merge weights
        if self.weights is not None and other.weights is not None:
            weights = torch.cat([self.weights, other.weights], dim=1)
        elif self.weights is not None:
            weights = torch.cat([
                self.weights,
                torch.zeros(other.positions.shape[:2], device=self.device)
            ], dim=1)
        elif other.weights is not None:
            weights = torch.cat([
                torch.zeros(self.positions.shape[:2], device=self.device),
                other.weights
            ], dim=1)
        else:
            weights = None
        
        return RaySamples(
            positions=positions,
            view_directions=view_directions,
            depths=depths,
            deltas=deltas,
            origins=self.origins,
            directions=self.directions,
            near=self.near,
            far=self.far,
            sample_type=sample_type,
            weights=weights,
            valid_mask=None,  # Reset valid mask
        )
    
    def sort_by_depth(self) -> RaySamples:
        """Sort samples along each ray by depth."""
        # Get sort indices
        sorted_indices = torch.argsort(self.depths, dim=1)
        
        # Apply sorting to all sample dimensions
        batch_indices = torch.arange(self.num_rays, device=self.device).unsqueeze(1)
        
        positions_sorted = self.positions[batch_indices, sorted_indices]
        view_directions_sorted = self.view_directions[batch_indices, sorted_indices]
        depths_sorted = self.depths[batch_indices, sorted_indices]
        
        # Recompute deltas for sorted depths
        deltas_sorted = torch.zeros_like(depths_sorted)
        deltas_sorted[:, :-1] = depths_sorted[:, 1:] - depths_sorted[:, :-1]
        deltas_sorted[:, -1] = deltas_sorted[:, -2]  # Extend last delta
        
        # Sort other attributes if present
        sample_type_sorted = None
        if self.sample_type is not None:
            sample_type_sorted = self.sample_type[batch_indices, sorted_indices]
        
        weights_sorted = None
        if self.weights is not None:
            weights_sorted = self.weights[batch_indices, sorted_indices]
        
        return RaySamples(
            positions=positions_sorted,
            view_directions=view_directions_sorted,
            depths=depths_sorted,
            deltas=deltas_sorted,
            origins=self.origins,
            directions=self.directions,
            near=self.near,
            far=self.far,
            sample_type=sample_type_sorted,
            weights=weights_sorted,
            valid_mask=None,
        )
    
    def to(self, device: torch.device) -> RaySamples:
        """Move samples to device."""
        return RaySamples(
            positions=self.positions.to(device),
            view_directions=self.view_directions.to(device),
            depths=self.depths.to(device),
            deltas=self.deltas.to(device),
            origins=self.origins.to(device),
            directions=self.directions.to(device),
            near=self.near.to(device) if torch.is_tensor(self.near) else self.near,
            far=self.far.to(device) if torch.is_tensor(self.far) else self.far,
            sample_type=self.sample_type.to(device) if self.sample_type is not None else None,
            weights=self.weights.to(device) if self.weights is not None else None,
            valid_mask=self.valid_mask.to(device) if self.valid_mask is not None else None,
        )
    
    def detach(self) -> RaySamples:
        """Detach all tensors from computation graph."""
        return RaySamples(
            positions=self.positions.detach(),
            view_directions=self.view_directions.detach(),
            depths=self.depths.detach(),
            deltas=self.deltas.detach(),
            origins=self.origins.detach(),
            directions=self.directions.detach(),
            near=self.near.detach() if torch.is_tensor(self.near) else self.near,
            far=self.far.detach() if torch.is_tensor(self.far) else self.far,
            sample_type=self.sample_type.detach() if self.sample_type is not None else None,
            weights=self.weights.detach() if self.weights is not None else None,
            valid_mask=self.valid_mask.detach() if self.valid_mask is not None else None,
        )
    
    def slice(self, start: int, end: int) -> RaySamples:
        """Slice samples along ray dimension."""
        return RaySamples(
            positions=self.positions[:, start:end],
            view_directions=self.view_directions[:, start:end],
            depths=self.depths[:, start:end],
            deltas=self.deltas[:, start:end],
            origins=self.origins,
            directions=self.directions,
            near=self.near,
            far=self.far,
            sample_type=self.sample_type[:, start:end] if self.sample_type is not None else None,
            weights=self.weights[:, start:end] if self.weights is not None else None,
            valid_mask=self.valid_mask[:, start:end] if self.valid_mask is not None else None,
        )


@dataclass
class RayBundle:
    """Bundle of rays with sampling information."""
    
    # Ray data
    origins: torch.Tensor  # [N, 3] or [N, H*W, 3]
    directions: torch.Tensor  # [N, 3] or [N, H*W, 3]
    near: torch.Tensor  # [N] or scalar
    far: torch.Tensor  # [N] or scalar
    
    # Optional metadata
    pixel_coords: Optional[torch.Tensor] = None  # [N, 2] pixel coordinates
    image_indices: Optional[torch.Tensor] = None  # [N] image indices
    camera_poses: Optional[torch.Tensor] = None  # [N, 3, 4] or [N, 4, 4]
    intrinsics: Optional[torch.Tensor] = None  # [N, 3, 3] or [N, 4]
    
    # Samples (filled after sampling)
    samples: Optional[RaySamples] = None
    
    def __post_init__(self):
        """Validate shapes."""
        assert self.origins.shape == self.directions.shape, \
            f"origins shape {self.origins.shape} != directions shape {self.directions.shape}"
        
        # Ensure directions are normalized
        if not torch.allclose(self.directions.norm(dim=-1), torch.ones_like(self.directions.norm(dim=-1)), rtol=1e-3):
            self.directions = F.normalize(self.directions, dim=-1)
    
    @property
    def num_rays(self) -> int:
        """Get number of rays."""
        if self.origins.dim() == 2:
            return self.origins.shape[0]
        elif self.origins.dim() == 3:
            return self.origins.shape[0] * self.origins.shape[1]
        else:
            raise ValueError(f"Unexpected origins dimension: {self.origins.dim()}")
    
    @property
    def device(self) -> torch.device:
        """Get device of ray bundle."""
        return self.origins.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get data type of ray bundle."""
        return self.origins.dtype
    
    def reshape_for_image(self, height: int, width: int) -> RayBundle:
        """
        Reshape ray bundle for image rendering.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            Reshaped ray bundle
        """
        if self.origins.dim() != 3:
            return self
        
        batch_size = self.origins.shape[0]
        num_pixels = height * width
        
        if self.origins.shape[1] != num_pixels:
            raise ValueError(f"Expected {num_pixels} pixels, got {self.origins.shape[1]}")
        
        # Reshape to [batch_size, height, width, ...]
        origins_reshaped = self.origins.reshape(batch_size, height, width, 3)
        directions_reshaped = self.directions.reshape(batch_size, height, width, 3)
        
        # Handle near/far
        if torch.is_tensor(self.near) and self.near.dim() > 0:
            near_reshaped = self.near.reshape(batch_size, height, width)
        else:
            near_reshaped = self.near
        
        if torch.is_tensor(self.far) and self.far.dim() > 0:
            far_reshaped = self.far.reshape(batch_size, height, width)
        else:
            far_reshaped = self.far
        
        # Handle pixel coordinates
        pixel_coords_reshaped = None
        if self.pixel_coords is not None:
            pixel_coords_reshaped = self.pixel_coords.reshape(batch_size, height, width, 2)
        
        return RayBundle(
            origins=origins_reshaped,
            directions=directions_reshaped,
            near=near_reshaped,
            far=far_reshaped,
            pixel_coords=pixel_coords_reshaped,
            image_indices=self.image_indices,
            camera_poses=self.camera_poses,
            intrinsics=self.intrinsics,
            samples=self.samples,
        )
    
    def flatten(self) -> RayBundle:
        """Flatten ray bundle to 2D."""
        if self.origins.dim() == 2:
            return self
        
        batch_size = self.origins.shape[0]
        spatial_dims = self.origins.shape[1:-1]
        num_elements = np.prod(spatial_dims)
        
        origins_flat = self.origins.reshape(batch_size * num_elements, 3)
        directions_flat = self.directions.reshape(batch_size * num_elements, 3)
        
        # Handle near/far
        if torch.is_tensor(self.near) and self.near.dim() > 0:
            near_flat = self.near.reshape(batch_size * num_elements)
        else:
            near_flat = self.near
        
        if torch.is_tensor(self.far) and self.far.dim() > 0:
            far_flat = self.far.reshape(batch_size * num_elements)
        else:
            far_flat = self.far
        
        # Handle pixel coordinates
        pixel_coords_flat = None
        if self.pixel_coords is not None:
            pixel_coords_flat = self.pixel_coords.reshape(batch_size * num_elements, 2)
        
        # Handle image indices
        image_indices_flat = None
        if self.image_indices is not None:
            image_indices_flat = self.image_indices.unsqueeze(1).repeat(1, num_elements).reshape(-1)
        
        return RayBundle(
            origins=origins_flat,
            directions=directions_flat,
            near=near_flat,
            far=far_flat,
            pixel_coords=pixel_coords_flat,
            image_indices=image_indices_flat,
            camera_poses=self.camera_poses,
            intrinsics=self.intrinsics,
            samples=self.samples,
        )
    
    def to(self, device: torch.device) -> RayBundle:
        """Move ray bundle to device."""
        return RayBundle(
            origins=self.origins.to(device),
            directions=self.directions.to(device),
            near=self.near.to(device) if torch.is_tensor(self.near) else self.near,
            far=self.far.to(device) if torch.is_tensor(self.far) else self.far,
            pixel_coords=self.pixel_coords.to(device) if self.pixel_coords is not None else None,
            image_indices=self.image_indices.to(device) if self.image_indices is not None else None,
            camera_poses=self.camera_poses.to(device) if self.camera_poses is not None else None,
            intrinsics=self.intrinsics.to(device) if self.intrinsics is not None else None,
            samples=self.samples.to(device) if self.samples is not None else None,
        )
    
    def detach(self) -> RayBundle:
        """Detach all tensors from computation graph."""
        return RayBundle(
            origins=self.origins.detach(),
            directions=self.directions.detach(),
            near=self.near.detach() if torch.is_tensor(self.near) else self.near,
            far=self.far.detach() if torch.is_tensor(self.far) else self.far,
            pixel_coords=self.pixel_coords.detach() if self.pixel_coords is not None else None,
            image_indices=self.image_indices.detach() if self.image_indices is not None else None,
            camera_poses=self.camera_poses.detach() if self.camera_poses is not None else None,
            intrinsics=self.intrinsics.detach() if self.intrinsics is not None else None,
            samples=self.samples.detach() if self.samples is not None else None,
        )


class RaySampler(nn.Module):
    """
    Ray sampler for NeRF.
    
    This class handles sampling points along rays for volumetric rendering,
    including different sampling strategies and hierarchical sampling.
    """
    
    def __init__(
        self,
        config: Optional[SamplingConfig] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        num_coarse_samples: Optional[int] = None,
        num_fine_samples: Optional[int] = None,
        perturb: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize ray sampler.
        
        Args:
            config: Sampling configuration
            near: Near plane distance (overrides config)
            far: Far plane distance (overrides config)
            num_coarse_samples: Number of coarse samples (overrides config)
            num_fine_samples: Number of fine samples (overrides config)
            perturb: Whether to perturb samples (overrides config)
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        
        # Create or update config
        if config is None:
            config_dict = {}
            if near is not None:
                config_dict['near'] = near
            if far is not None:
                config_dict['far'] = far
            if num_coarse_samples is not None:
                config_dict['num_coarse_samples'] = num_coarse_samples
            if num_fine_samples is not None:
                config_dict['num_fine_samples'] = num_fine_samples
            if perturb is not None:
                config_dict['perturb'] = perturb
            config_dict.update(kwargs)
            config = SamplingConfig(**config_dict)
        else:
            # Update config with provided parameters
            config_dict = config.__dict__.copy()
            if near is not None:
                config_dict['near'] = near
            if far is not None:
                config_dict['far'] = far
            if num_coarse_samples is not None:
                config_dict['num_coarse_samples'] = num_coarse_samples
            if num_fine_samples is not None:
                config_dict['num_fine_samples'] = num_fine_samples
            if perturb is not None:
                config_dict['perturb'] = perturb
            config_dict.update(kwargs)
            config = SamplingConfig(**config_dict)
        
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        config = self.config
        
        if config.near >= config.far:
            raise ValueError(f"near ({config.near}) must be less than far ({config.far})")
        
        if config.num_coarse_samples <= 0:
            raise ValueError(f"num_coarse_samples must be positive, got {config.num_coarse_samples}")
        
        if config.num_fine_samples < 0:
            raise ValueError(f"num_fine_samples must be non-negative, got {config.num_fine_samples}")
        
        if config.num_importance_samples < 0:
            raise ValueError(f"num_importance_samples must be non-negative, got {config.num_importance_samples}")
    
    def forward(
        self,
        ray_bundle: RayBundle,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> RayBundle:
        """
        Sample points along rays.
        
        Args:
            ray_bundle: Input ray bundle
            weights: Optional weights for importance sampling [N, S]
            **kwargs: Additional sampling parameters
            
        Returns:
            Ray bundle with samples
        """
        strategy = kwargs.get('strategy', self.config.strategy)
        
        if strategy == SamplingStrategy.HIERARCHICAL:
            return self.hierarchical_sampling(ray_bundle, weights, **kwargs)
        elif strategy == SamplingStrategy.IMPORTANCE:
            return self.importance_sampling(ray_bundle, weights, **kwargs)
        elif strategy == SamplingStrategy.UNIFORM:
            return self.uniform_sampling(ray_bundle, **kwargs)
        elif strategy == SamplingStrategy.STRATIFIED:
            return self.stratified_sampling(ray_bundle, **kwargs)
        elif strategy == SamplingStrategy.INVERSE_SPHERE:
            return self.inverse_sphere_sampling(ray_bundle, **kwargs)
        elif strategy == SamplingStrategy.NEAR_FAR:
            return self.near_far_sampling(ray_bundle, **kwargs)
        elif strategy == SamplingStrategy.EXPONENTIAL:
            return self.exponential_sampling(ray_bundle, **kwargs)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def sample_along_rays(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        near: Optional[Union[float, torch.Tensor]] = None,
        far: Optional[Union[float, torch.Tensor]] = None,
        **kwargs
    ) -> RayBundle:
        """
        Sample points along rays (convenience method).
        
        Args:
            origins: Ray origins [N, 3] or [N, H*W, 3]
            directions: Ray directions [N, 3] or [N, H*W, 3]
            near: Near plane distance(s)
            far: Far plane distance(s)
            **kwargs: Additional sampling parameters
            
        Returns:
            Ray bundle with samples
        """
        # Use config values if not provided
        if near is None:
            near = self.config.near
        if far is None:
            far = self.config.far
        
        # Create ray bundle
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            near=near,
            far=far,
        )
        
        return self.forward(ray_bundle, **kwargs)
    
    def uniform_sampling(
        self,
        ray_bundle: RayBundle,
        **kwargs
    ) -> RayBundle:
        """
        Uniform sampling along rays.
        
        Args:
            ray_bundle: Input ray bundle
            **kwargs: Additional parameters
            
        Returns:
            Ray bundle with uniformly sampled points
        """
        num_samples = kwargs.get('num_samples', self.config.num_coarse_samples)
        perturb = kwargs.get('perturb', self.config.perturb)
        linear_disparity = kwargs.get('linear_disparity', self.config.linear_disparity)
        
        # Extract ray data
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        near = ray_bundle.near
        far = ray_bundle.far
        
        # Ensure tensors
        if not torch.is_tensor(near):
            near = torch.tensor(near, device=origins.device, dtype=origins.dtype)
        if not torch.is_tensor(far):
            far = torch.tensor(far, device=origins.device, dtype=origins.dtype)
        
        # Handle broadcasting
        if near.dim() == 0:
            near = near.expand(origins.shape[0])
        if far.dim() == 0:
            far = far.expand(origins.shape[0])
        
        # Sample depths
        depths = self._sample_uniform_depths(
            near, far, num_samples, perturb, linear_disparity
        )
        
        # Compute sample positions
        positions, deltas = self._compute_sample_positions(
            origins, directions, depths, near, far
        )
        
        # Create view directions (same for all samples along a ray)
        view_directions = directions.unsqueeze(1).expand(-1, num_samples, -1)
        
        # Create RaySamples
        samples = RaySamples(
            positions=positions,
            view_directions=view_directions,
            depths=depths,
            deltas=deltas,
            origins=origins,
            directions=directions,
            near=near,
            far=far,
            sample_type=torch.zeros(origins.shape[0], num_samples, device=origins.device, dtype=torch.long),
        )
        
        # Return updated ray bundle
        return RayBundle(
            origins=origins,
            directions=directions,
            near=near,
            far=far,
            pixel_coords=ray_bundle.pixel_coords,
            image_indices=ray_bundle.image_indices,
            camera_poses=ray_bundle.camera_poses,
            intrinsics=ray_bundle.intrinsics,
            samples=samples,
        )
    
    def stratified_sampling(
        self,
        ray_bundle: RayBundle,
        **kwargs
    ) -> RayBundle:
        """
        Stratified (jittered uniform) sampling along rays.
        
        Args:
            ray_bundle: Input ray bundle
            **kwargs: Additional parameters
            
        Returns:
            Ray bundle with stratified sampled points
        """
        # Stratified sampling is just uniform sampling with perturbation
        return self.uniform_sampling(ray_bundle, perturb=True, **kwargs)
    
    def hierarchical_sampling(
        self,
        ray_bundle: RayBundle,
        weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> RayBundle:
        """
        Hierarchical (coarse-to-fine) sampling.
        
        Args:
            ray_bundle: Input ray bundle
            weights: Weights from coarse sampling [N, S]
            **kwargs: Additional parameters
            
        Returns:
            Ray bundle with hierarchically sampled points
        """
        # First, do coarse sampling
        coarse_config = self.config.copy() if hasattr(self.config, 'copy') else SamplingConfig(
            near=self.config.near,
            far=self.config.far,
            num_coarse_samples=self.config.num_coarse_samples,
            perturb=self.config.perturb,
            linear_disparity=self.config.linear_disparity,
        )
        
        coarse_ray_bundle = self.uniform_sampling(
            ray_bundle,
            num_samples=coarse_config.num_coarse_samples,
            perturb=coarse_config.perturb,
            linear_disparity=coarse_config.linear_disparity,
        )
        
        # If no weights provided, assume uniform weights
        if weights is None:
            weights = torch.ones_like(coarse_ray_bundle.samples.depths)
        
        # Then, do fine sampling based on weights
        fine_samples = self._sample_fine_depths(
            coarse_ray_bundle.samples.depths,
            weights,
            self.config.num_fine_samples,
            self.config.perturb,
        )
        
        # Merge coarse and fine samples
        all_depths = torch.cat([coarse_ray_bundle.samples.depths, fine_samples], dim=1)
        
        # Sort depths
        sorted_indices = torch.argsort(all_depths, dim=1)
        batch_indices = torch.arange(all_depths.shape[0], device=all_depths.device).unsqueeze(1)
        depths_sorted = all_depths[batch_indices, sorted_indices]
        
        # Compute positions for all samples
        positions, deltas = self._compute_sample_positions(
            coarse_ray_bundle.origins,
            coarse_ray_bundle.directions,
            depths_sorted,
            coarse_ray_bundle.near,
            coarse_ray_bundle.far,
        )
        
        # Create sample types
        num_coarse = coarse_config.num_coarse_samples
        num_total = num_coarse + self.config.num_fine_samples
        
        sample_type = torch.zeros(positions.shape[0], num_total, device=positions.device, dtype=torch.long)
        sample_type[:, :num_coarse] = 0  # Coarse samples
        sample_type[:, num_coarse:] = 1  # Fine samples
        
        # Apply sorting to sample types
        sample_type_sorted = sample_type[batch_indices, sorted_indices]
        
        # Create view directions
        view_directions = coarse_ray_bundle.directions.unsqueeze(1).expand(-1, num_total, -1)
        
        # Create samples
        samples = RaySamples(
            positions=positions,
            view_directions=view_directions,
            depths=depths_sorted,
            deltas=deltas,
            origins=coarse_ray_bundle.origins,
            directions=coarse_ray_bundle.directions,
            near=coarse_ray_bundle.near,
            far=coarse_ray_bundle.far,
            sample_type=sample_type_sorted,
        )
        
        # Return updated ray bundle
        return RayBundle(
            origins=coarse_ray_bundle.origins,
            directions=coarse_ray_bundle.directions,
            near=coarse_ray_bundle.near,
            far=coarse_ray_bundle.far,
            pixel_coords=ray_bundle.pixel_coords,
            image_indices=ray_bundle.image_indices,
            camera_poses=ray_bundle.camera_poses,
            intrinsics=ray_bundle.intrinsics,
            samples=samples,
        )
    
    def importance_sampling(
        self,
        ray_bundle: RayBundle,
        weights: torch.Tensor,
        **kwargs
    ) -> RayBundle:
        """
        Importance sampling based on weights.
        
        Args:
            ray_bundle: Input ray bundle
            weights: Sample weights for importance sampling [N, S]
            **kwargs: Additional parameters
            
        Returns:
            Ray bundle with importance sampled points
        """
        num_samples = kwargs.get('num_importance_samples', self.config.num_importance_samples)
        perturb = kwargs.get('perturb', self.config.perturb)
        
        # Sample depths based on weights
        importance_depths = self._sample_importance_depths(
            ray_bundle.samples.depths,
            weights,
            num_samples,
            perturb,
        )
        
        # Merge with original samples
        all_depths = torch.cat([ray_bundle.samples.depths, importance_depths], dim=1)
        
        # Sort depths
        sorted_indices = torch.argsort(all_depths, dim=1)
        batch_indices = torch.arange(all_depths.shape[0], device=all_depths.device).unsqueeze(1)
        depths_sorted = all_depths[batch_indices, sorted_indices]
        
        # Compute positions for all samples
        positions, deltas = self._compute_sample_positions(
            ray_bundle.origins,
            ray_bundle.directions,
            depths_sorted,
            ray_bundle.near,
            ray_bundle.far,
        )
        
        # Create sample types
        num_original = ray_bundle.samples.depths.shape[1]
        num_total = num_original + num_samples
        
        sample_type = torch.zeros(positions.shape[0], num_total, device=positions.device, dtype=torch.long)
        sample_type[:, :num_original] = 0  # Original samples
        sample_type[:, num_original:] = 2  # Importance samples
        
        # Apply sorting to sample types
        sample_type_sorted = sample_type[batch_indices, sorted_indices]
        
        # Create view directions
        view_directions = ray_bundle.directions.unsqueeze(1).expand(-1, num_total, -1)
        
        # Create samples
        samples = RaySamples(
            positions=positions,
            view_directions=view_directions,
            depths=depths_sorted,
            deltas=deltas,
            origins=ray_bundle.origins,
            directions=ray_bundle.directions,
            near=ray_bundle.near,
            far=ray_bundle.far,
            sample_type=sample_type_sorted,
        )
        
        # Return updated ray bundle
        return RayBundle(
            origins=ray_bundle.origins,
            directions=ray_bundle.directions,
            near=ray_bundle.near,
            far=ray_bundle.far,
            pixel_coords=ray_bundle.pixel_coords,
            image_indices=ray_bundle.image_indices,
            camera_poses=ray_bundle.camera_poses,
            intrinsics=ray_bundle.intrinsics,
            samples=samples,
        )
    
    def inverse_sphere_sampling(
        self,
        ray_bundle: RayBundle,
        **kwargs
    ) -> RayBundle:
        """
        Inverse sphere sampling for unbounded scenes.
        
        Args:
            ray_bundle: Input ray bundle
            **kwargs: Additional parameters
            
        Returns:
            Ray bundle with inverse sphere sampled points
        """
        num_samples = kwargs.get('num_samples', self.config.num_coarse_samples)
        perturb = kwargs.get('perturb', self.config.perturb)
        sphere_radius = kwargs.get('sphere_radius', self.config.sphere_radius)
        
        # Extract ray data
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        near = ray_bundle.near
        far = ray_bundle.far
        
        # Ensure tensors
        if not torch.is_tensor(near):
            near = torch.tensor(near, device=origins.device, dtype=origins.dtype)
        if not torch.is_tensor(far):
            far = torch.tensor(far, device=origins.device, dtype=origins.dtype)
        
        # Handle broadcasting
        if near.dim() == 0:
            near = near.expand(origins.shape[0])
        if far.dim() == 0:
            far = far.expand(origins.shape[0])
        
        # Compute intersection with sphere
        # Solve quadratic equation: ||o + t*d||^2 = r^2
        a = torch.sum(directions * directions, dim=-1)
        b = 2 * torch.sum(origins * directions, dim=-1)
        c = torch.sum(origins * origins, dim=-1) - sphere_radius ** 2
        
        discriminant = b ** 2 - 4 * a * c
        
        # Find intersection points
        t1 = (-b - torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * a + 1e-8)
        t2 = (-b + torch.sqrt(torch.clamp(discriminant, min=0))) / (2 * a + 1e-8)
        
        # Determine near and far for inverse sphere parameterization
        sphere_near = torch.where(discriminant > 0, torch.maximum(near, t1), near)
        sphere_far = torch.where(discriminant > 0, torch.minimum(far, t2), far)
        
        # Sample in inverse depth space
        # Map t to s = 1/t for sphere region
        s_near = 1.0 / sphere_far
        s_far = 1.0 / sphere_near
        
        # Sample uniformly in inverse depth
        s_samples = self._sample_uniform_depths(
            s_near, s_far, num_samples, perturb, linear_disparity=False
        )
        
        # Map back to depth
        depths = 1.0 / s_samples
        
        # Also sample in foreground (inside sphere)
        foreground_depths = self._sample_uniform_depths(
            near, sphere_near, num_samples // 2, perturb, linear_disparity=False
        )
        
        # Combine samples
        all_depths = torch.cat([foreground_depths, depths], dim=1)
        
        # Sort depths
        sorted_indices = torch.argsort(all_depths, dim=1)
        batch_indices = torch.arange(all_depths.shape[0], device=all_depths.device).unsqueeze(1)
        depths_sorted = all_depths[batch_indices, sorted_indices]
        
        # Compute sample positions
        positions, deltas = self._compute_sample_positions(
            origins, directions, depths_sorted, near, far
        )
        
        # Create view directions
        num_total = num_samples + num_samples // 2
        view_directions = directions.unsqueeze(1).expand(-1, num_total, -1)
        
        # Create samples
        samples = RaySamples(
            positions=positions,
            view_directions=view_directions,
            depths=depths_sorted,
            deltas=deltas,
            origins=origins,
            directions=directions,
            near=near,
            far=far,
        )
        
        # Return updated ray bundle
        return RayBundle(
            origins=origins,
            directions=directions,
            near=near,
            far=far,
            pixel_coords=ray_bundle.pixel_coords,
            image_indices=ray_bundle.image_indices,
            camera_poses=ray_bundle.camera_poses,
            intrinsics=ray_bundle.intrinsics,
            samples=samples,
        )
    
    def near_far_sampling(
        self,
        ray_bundle: RayBundle,
        **kwargs
    ) -> RayBundle:
        """
        Sample based on estimated near and far planes.
        
        Args:
            ray_bundle: Input ray bundle
            **kwargs: Additional parameters
            
        Returns:
            Ray bundle with near-far sampled points
        """
        # This would use scene geometry to estimate near/far per ray
        # For now, fall back to uniform sampling
        return self.uniform_sampling(ray_bundle, **kwargs)
    
    def exponential_sampling(
        self,
        ray_bundle: RayBundle,
        **kwargs
    ) -> RayBundle:
        """
        Exponential sampling for unbounded scenes.
        
        Args:
            ray_bundle: Input ray bundle
            **kwargs: Additional parameters
            
        Returns:
            Ray bundle with exponentially sampled points
        """
        num_samples = kwargs.get('num_samples', self.config.num_coarse_samples)
        perturb = kwargs.get('perturb', self.config.perturb)
        lambd = kwargs.get('exponential_lambda', self.config.exponential_lambda)
        
        # Extract ray data
        origins = ray_bundle.origins
        directions = ray_bundle.directions
        near = ray_bundle.near
        far = ray_bundle.far
        
        # Ensure tensors
        if not torch.is_tensor(near):
            near = torch.tensor(near, device=origins.device, dtype=origins.dtype)
        if not torch.is_tensor(far):
            far = torch.tensor(far, device=origins.device, dtype=origins.dtype)
        
        # Handle broadcasting
        if near.dim() == 0:
            near = near.expand(origins.shape[0])
        if far.dim() == 0:
            far = far.expand(origins.shape[0])
        
        # Sample exponentially
        # t = near + (1/lambd) * log(1 + (exp(lambd*(far-near)) - 1) * u)
        u = torch.linspace(0, 1, num_samples, device=origins.device, dtype=origins.dtype)
        u = u.unsqueeze(0).expand(origins.shape[0], -1)
        
        if perturb:
            # Add random jitter
            u = u + torch.rand_like(u) / num_samples
            u = torch.clamp(u, 0, 1)
        
        depths = near.unsqueeze(1) + (1.0 / lambd) * torch.log(
            1.0 + (torch.exp(lambd * (far - near).unsqueeze(1)) - 1.0) * u
        )
        
        # Compute sample positions
        positions, deltas = self._compute_sample_positions(
            origins, directions, depths, near, far
        )
        
        # Create view directions
        view_directions = directions.unsqueeze(1).expand(-1, num_samples, -1)
        
        # Create samples
        samples = RaySamples(
            positions=positions,
            view_directions=view_directions,
            depths=depths,
            deltas=deltas,
            origins=origins,
            directions=directions,
            near=near,
            far=far,
        )
        
        # Return updated ray bundle
        return RayBundle(
            origins=origins,
            directions=directions,
            near=near,
            far=far,
            pixel_coords=ray_bundle.pixel_coords,
            image_indices=ray_bundle.image_indices,
            camera_poses=ray_bundle.camera_poses,
            intrinsics=ray_bundle.intrinsics,
            samples=samples,
        )
    
    def _sample_uniform_depths(
        self,
        near: torch.Tensor,
        far: torch.Tensor,
        num_samples: int,
        perturb: bool,
        linear_disparity: bool = False,
    ) -> torch.Tensor:
        """
        Sample depths uniformly along rays.
        
        Args:
            near: Near distances [N]
            far: Far distances [N]
            num_samples: Number of samples
            perturb: Whether to jitter samples
            linear_disparity: Sample linearly in disparity
            
        Returns:
            Sampled depths [N, S]
        """
        N = near.shape[0]
        
        # Create bin edges
        if linear_disparity:
            # Sample linearly in disparity (1/depth)
            start = 1.0 / far
            end = 1.0 / near
            t_vals = torch.linspace(0, 1, num_samples, device=near.device, dtype=near.dtype)
            t_vals = start.unsqueeze(1) + t_vals.unsqueeze(0) * (end - start).unsqueeze(1)
            depths = 1.0 / t_vals
        else:
            # Sample linearly in depth
            t_vals = torch.linspace(0, 1, num_samples, device=near.device, dtype=near.dtype)
            depths = near.unsqueeze(1) + t_vals.unsqueeze(0) * (far - near).unsqueeze(1)
        
        # Add jitter if requested
        if perturb and num_samples > 1:
            mids = 0.5 * (depths[:, 1:] + depths[:, :-1])
            upper = torch.cat([mids, depths[:, -1:]], dim=1)
            lower = torch.cat([depths[:, :1], mids], dim=1)
            t_rand = torch.rand(N, num_samples, device=near.device, dtype=near.dtype)
            depths = lower + (upper - lower) * t_rand
        
        return depths
    
    def _sample_fine_depths(
        self,
        coarse_depths: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int,
        perturb: bool,
    ) -> torch.Tensor:
        """
        Sample fine depths based on weights from coarse sampling.
        
        Args:
            coarse_depths: Coarse sample depths [N, S]
            weights: Sample weights [N, S]
            num_samples: Number of fine samples
            perturb: Whether to jitter samples
            
        Returns:
            Fine sample depths [N, S_fine]
        """
        if num_samples <= 0:
            return torch.empty(coarse_depths.shape[0], 0, device=coarse_depths.device, dtype=coarse_depths.dtype)
        
        N, S = coarse_depths.shape
        
        # Add small epsilon to avoid zero weights
        weights = weights + 1e-5
        
        # Normalize weights
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
        
        # Sample from distribution
        u = torch.linspace(0, 1, num_samples, device=coarse_depths.device, dtype=coarse_depths.dtype)
        u = u.unsqueeze(0).expand(N, -1)
        
        if perturb:
            u = u + torch.rand(N, num_samples, device=coarse_depths.device, dtype=coarse_depths.dtype) / num_samples
            u = torch.clamp(u, 0, 1)
        
        # Inverse transform sampling
        u = u.contiguous()
        cdf = cdf.contiguous()
        
        # Find indices where u would be inserted to maintain order
        indices = torch.searchsorted(cdf, u, right=True)
        
        # Clamp indices
        indices_below = torch.clamp(indices - 1, 0, S)
        indices_above = torch.clamp(indices, 0, S)
        
        # Gather cdf values
        cdf_below = torch.gather(cdf, 1, indices_below)
        cdf_above = torch.gather(cdf, 1, indices_above)
        
        # Gather depth values
        depths_below = torch.gather(coarse_depths, 1, torch.clamp(indices_below, 0, S-1))
        depths_above = torch.gather(coarse_depths, 1, torch.clamp(indices_above - 1, 0, S-1))
        
        # Avoid division by zero
        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        
        # Linear interpolation
        t = (u - cdf_below) / denom
        fine_depths = depths_below + t * (depths_above - depths_below)
        
        return fine_depths
    
    def _sample_importance_depths(
        self,
        depths: torch.Tensor,
        weights: torch.Tensor,
        num_samples: int,
        perturb: bool,
    ) -> torch.Tensor:
        """
        Sample depths based on importance weights.
        
        Args:
            depths: Existing sample depths [N, S]
            weights: Sample weights [N, S]
            num_samples: Number of importance samples
            perturb: Whether to jitter samples
            
        Returns:
            Importance sample depths [N, S_importance]
        """
        # This is similar to fine sampling but with different parameters
        return self._sample_fine_depths(depths, weights, num_samples, perturb)
    
    def _compute_sample_positions(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        depths: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sample positions and deltas from depths.
        
        Args:
            origins: Ray origins [N, 3]
            directions: Ray directions [N, 3]
            depths: Sample depths [N, S]
            near: Near distances [N]
            far: Far distances [N]
            
        Returns:
            Tuple of (positions [N, S, 3], deltas [N, S])
        """
        # Compute positions: o + t*d
        positions = origins.unsqueeze(1) + depths.unsqueeze(2) * directions.unsqueeze(1)
        
        # Compute deltas (distance between samples)
        deltas = torch.zeros_like(depths)
        deltas[:, :-1] = depths[:, 1:] - depths[:, :-1]
        
        # Set last delta to be same as second-to-last (or a large value)
        if deltas.shape[1] > 1:
            deltas[:, -1] = deltas[:, -2]
        else:
            # If only one sample, set delta to far - near
            deltas[:, -1] = far - near
        
        return positions, deltas
    
    def estimate_near_far(
        self,
        positions: torch.Tensor,
        directions: torch.Tensor,
        scene_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate near and far planes for rays.
        
        Args:
            positions: Ray origins [N, 3]
            directions: Ray directions [N, 3]
            scene_bounds: Optional scene bounds (min, max)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (near [N], far [N])
        """
        N = positions.shape[0]
        device = positions.device
        dtype = positions.dtype
        
        if scene_bounds is None:
            # Use default near/far
            near = torch.full((N,), self.config.near, device=device, dtype=dtype)
            far = torch.full((N,), self.config.far, device=device, dtype=dtype)
            return near, far
        
        # Compute ray-box intersection
        min_bound, max_bound = scene_bounds
        
        # Ray-box intersection algorithm (slab method)
        tmin = (min_bound - positions) / (directions + 1e-8)
        tmax = (max_bound - positions) / (directions + 1e-8)
        
        t1 = torch.minimum(tmin, tmax)
        t2 = torch.maximum(tmin, tmax)
        
        near = torch.max(t1, dim=-1)[0]
        far = torch.min(t2, dim=-1)[0]
        
        # Clamp to valid range
        near = torch.clamp(near, min=self.config.near)
        far = torch.clamp(far, max=self.config.far)
        
        # Ensure near < far
        valid = far > near
        near = torch.where(valid, near, torch.tensor(self.config.near, device=device, dtype=dtype))
        far = torch.where(valid, far, torch.tensor(self.config.far, device=device, dtype=dtype))
        
        return near, far
    
    def compute_sample_weights(
        self,
        densities: torch.Tensor,
        deltas: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute sample weights for volumetric rendering.
        
        Args:
            densities: Density values [N, S]
            deltas: Distance between samples [N, S]
            noise_std: Standard deviation of noise to add
            
        Returns:
            Sample weights [N, S]
        """
        # Add noise if training
        if noise_std > 0 and self.training:
            noise = torch.randn_like(densities) * noise_std
            densities = densities + noise
        
        # Compute alpha (opacity) from density
        alphas = 1.0 - torch.exp(-densities * deltas)
        
        # Compute transmittance and weights
        transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=1)
        transmittance = torch.cat([
            torch.ones_like(transmittance[:, :1]),
            transmittance[:, :-1]
        ], dim=1)
        
        weights = transmittance * alphas
        
        return weights
    
    def resample_rays(
        self,
        ray_bundle: RayBundle,
        weights: torch.Tensor,
        num_samples: int,
        strategy: Union[str, SamplingStrategy] = SamplingStrategy.IMPORTANCE,
        **kwargs
    ) -> RayBundle:
        """
        Resample rays based on weights.
        
        Args:
            ray_bundle: Input ray bundle with samples
            weights: Sample weights [N, S]
            num_samples: Number of samples for resampling
            strategy: Resampling strategy
            **kwargs: Additional parameters
            
        Returns:
            Resampled ray bundle
        """
        if strategy == SamplingStrategy.IMPORTANCE:
            return self.importance_sampling(ray_bundle, weights, num_importance_samples=num_samples, **kwargs)
        else:
            raise ValueError(f"Unsupported resampling strategy: {strategy}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_ray_sampler(
    config: Optional[SamplingConfig] = None,
    **kwargs
) -> RaySampler:
    """
    Create a ray sampler (convenience function).
    
    Args:
        config: Sampling configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Ray sampler
    """
    return RaySampler(config, **kwargs)


def generate_rays_from_camera(
    camera_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from camera pose and intrinsics.
    
    Args:
        camera_pose: Camera pose [3, 4] or [4, 4]
        intrinsics: Camera intrinsics [3, 3] or [4]
        height: Image height
        width: Image width
        device: Device for tensors
        dtype: Data type for tensors
        
    Returns:
        Tuple of (ray_origins [H*W, 3], ray_directions [H*W, 3])
    """
    if device is None:
        device = camera_pose.device if torch.is_tensor(camera_pose) else torch.device('cpu')
    
    if dtype is None:
        dtype = camera_pose.dtype if torch.is_tensor(camera_pose) else torch.float32
    
    # Ensure tensors
    if not torch.is_tensor(camera_pose):
        camera_pose = torch.tensor(camera_pose, device=device, dtype=dtype)
    
    if not torch.is_tensor(intrinsics):
        intrinsics = torch.tensor(intrinsics, device=device, dtype=dtype)
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij'
    )
    
    x = x.flatten()
    y = y.flatten()
    
    # Get camera parameters
    if camera_pose.shape == (4, 4):
        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]
    else:
        rotation = camera_pose[:3, :3]
        translation = camera_pose[:3, 3]
    
    # Get intrinsics parameters
    if intrinsics.dim() == 1 and intrinsics.shape[0] == 4:
        fx, fy, cx, cy = intrinsics
    else:
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
    
    # Convert to normalized device coordinates
    x_ndc = (x - cx) / fx
    y_ndc = (y - cy) / fy
    
    # Direction vectors in camera space
    directions_camera = torch.stack([x_ndc, y_ndc, torch.ones_like(x_ndc)], dim=-1)
    
    # Transform to world space
    directions_world = directions_camera @ rotation.T
    directions_world = F.normalize(directions_world, dim=-1)
    
    # Ray origins (camera position in world space)
    origins_world = translation.unsqueeze(0).expand(height * width, -1)
    
    return origins_world, directions_world


def generate_rays_from_cameras(
    camera_poses: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays from multiple cameras.
    
    Args:
        camera_poses: Camera poses [B, 3, 4] or [B, 4, 4]
        intrinsics: Camera intrinsics [B, 3, 3] or [B, 4]
        height: Image height
        width: Image width
        
    Returns:
        Tuple of (ray_origins [B, H*W, 3], ray_directions [B, H*W, 3])
    """
    batch_size = camera_poses.shape[0]
    num_pixels = height * width
    
    # Prepare output tensors
    origins = torch.zeros(batch_size, num_pixels, 3, device=camera_poses.device, dtype=camera_poses.dtype)
    directions = torch.zeros(batch_size, num_pixels, 3, device=camera_poses.device, dtype=camera_poses.dtype)
    
    # Generate rays for each camera
    for i in range(batch_size):
        cam_origin, cam_directions = generate_rays_from_camera(
            camera_poses[i],
            intrinsics[i] if intrinsics.dim() > 1 else intrinsics,
            height, width
        )
        
        origins[i] = cam_origin
        directions[i] = cam_directions
    
    return origins, directions