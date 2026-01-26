"""
Neural Radiance Fields (NeRF) Module.

This module implements Neural Radiance Fields for novel view synthesis,
including the core NeRF model, ray sampling, volume rendering, and
positional encoding components.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from pathlib import Path
import json

# Import base classes
from ..base import (
    BaseRepresentation, 
    RepresentationConfig, 
    RepresentationMetrics,
    RenderMode,
    SampleStrategy,
    ExportFormat
)

# Import other modules that will be defined
from .nerf_model import NeRFModel, NeRFConfig, NeRFOutput
from .ray_sampler import RaySampler, RayBundle, RaySamples, SamplingConfig
from .volume_renderer import VolumeRenderer, VolumeRenderingConfig, RenderingOutput
from .positional_encoding import PositionalEncoding, PositionalEncodingConfig

__all__ = [
    # Main classes
    'NeRFModel',
    'NeRFConfig',
    'NeRFOutput',
    
    # Ray sampling
    'RaySampler',
    'RayBundle',
    'RaySamples',
    'SamplingConfig',
    
    # Volume rendering
    'VolumeRenderer',
    'VolumeRenderingConfig',
    'RenderingOutput',
    
    # Positional encoding
    'PositionalEncoding',
    'PositionalEncodingConfig',
    
    # Enums and types
    'NeRFType',
    'ActivationType',
    
    # Factory and utilities
    'NeRFFactory',
    'create_nerf',
    'load_nerf',
    'train_nerf',
    'render_nerf_views',
]


class NeRFType(Enum):
    """Types of NeRF architectures."""
    VANILLA = "vanilla"  # Original NeRF
    MIP = "mip"          # Mip-NeRF
    INSTANT_NGP = "instant_ngp"  # Instant NGP
    TENSORF = "tensorf"  # TensoRF
    DVGO = "dvgo"        # Direct Voxel Grid Optimization
    PLENOXELS = "plenoxels"  # Plenoxels (without neural network)
    K_PLANES = "k_planes"  # K-planes
    HEX_PLANES = "hex_planes"  # HexPlanes


class ActivationType(Enum):
    """Types of activation functions for NeRF."""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    SINE = "sine"  # Periodic activation (SIREN)
    SWISH = "swish"
    GELU = "gelu"


@dataclass
class NeRFConfig(RepresentationConfig):
    """Configuration for NeRF model."""
    
    # Architecture type
    nerf_type: NeRFType = NeRFType.VANILLA
    
    # Network architecture
    num_layers: int = 8
    hidden_dim: int = 256
    skip_connection_at: int = 4
    num_encoding_functions_xyz: int = 10
    num_encoding_functions_dir: int = 4
    include_view_direction: bool = True
    use_importance_sampling: bool = True
    
    # Positional encoding
    positional_encoding_type: str = "frequency"  # "frequency", "hash", "spherical_harmonics"
    hash_table_size: int = 2**19
    hash_table_levels: int = 16
    hash_table_feature_dim: int = 2
    per_level_scale: float = 1.5
    
    # Ray sampling
    num_coarse_samples: int = 64
    num_fine_samples: int = 128
    num_importance_samples: int = 64
    perturb: bool = True
    inverse_sphere_bg: bool = False
    
    # Volume rendering
    noise_std: float = 0.0
    white_bkgd: bool = True
    raw_noise_std: float = 0.0
    min_alpha: float = 1e-5
    
    # Bounding and clipping
    near_distance: float = 0.0
    far_distance: float = 1.0
    bounding_sphere_radius: float = 1.0
    
    # Feature grid (for TensoRF/DVGO/etc)
    grid_resolution: List[int] = field(default_factory=lambda: [128, 128, 128])
    grid_feature_dim: int = 32
    grid_interpolation: str = "linear"  # "linear", "trilinear", "nearest"
    
    # Appearance embedding
    use_appearance_embedding: bool = False
    appearance_embedding_dim: int = 16
    num_appearance_embeddings: int = 0
    
    # Deformation field
    use_deformation_field: bool = False
    deformation_field_layers: int = 4
    deformation_field_dim: int = 128
    
    # Time conditioning (for dynamic scenes)
    use_time_conditioning: bool = False
    time_embedding_dim: int = 8
    
    # Activation functions
    activation_type: ActivationType = ActivationType.RELU
    activation_slope: float = 0.01  # For leaky ReLU
    sine_omega: float = 30.0  # For SIREN
    
    # Density bias (for better initialization)
    density_bias: float = -1.0
    
    def __post_init__(self):
        """Validate NeRF configuration."""
        super().__post_init__()
        
        # Convert string enums to Enum types if needed
        if isinstance(self.nerf_type, str):
            self.nerf_type = NeRFType(self.nerf_type.lower())
        
        if isinstance(self.activation_type, str):
            self.activation_type = ActivationType(self.activation_type.lower())
        
        # Validate network architecture
        if self.num_layers < 2:
            raise ValueError(f"num_layers must be at least 2, got {self.num_layers}")
        
        if self.hidden_dim < 16:
            raise ValueError(f"hidden_dim must be at least 16, got {self.hidden_dim}")
        
        # Validate sampling parameters
        if self.num_coarse_samples <= 0:
            raise ValueError(f"num_coarse_samples must be positive, got {self.num_coarse_samples}")
        
        if self.num_fine_samples < 0:
            raise ValueError(f"num_fine_samples must be non-negative, got {self.num_fine_samples}")
        
        if self.num_importance_samples < 0:
            raise ValueError(f"num_importance_samples must be non-negative, got {self.num_importance_samples}")
        
        # Validate positional encoding
        if self.num_encoding_functions_xyz < 0:
            raise ValueError(f"num_encoding_functions_xyz must be non-negative, got {self.num_encoding_functions_xyz}")
        
        if self.num_encoding_functions_dir < 0:
            raise ValueError(f"num_encoding_functions_dir must be non-negative, got {self.num_encoding_functions_dir}")
        
        # Validate grid resolution
        if len(self.grid_resolution) != 3:
            raise ValueError(f"grid_resolution must have 3 elements, got {len(self.grid_resolution)}")
        
        for res in self.grid_resolution:
            if res <= 0:
                raise ValueError(f"grid_resolution elements must be positive, got {res}")
    
    @property
    def input_dim_xyz(self) -> int:
        """Get dimension of positional encoded XYZ input."""
        if self.positional_encoding_type == "frequency":
            return 3 + 3 * 2 * self.num_encoding_functions_xyz
        elif self.positional_encoding_type == "hash":
            return self.hash_table_levels * self.hash_table_feature_dim
        else:
            return 3  # Raw coordinates
    
    @property
    def input_dim_dir(self) -> int:
        """Get dimension of positional encoded direction input."""
        if self.positional_encoding_type == "frequency":
            return 3 + 3 * 2 * self.num_encoding_functions_dir
        elif self.positional_encoding_type == "spherical_harmonics":
            # For spherical harmonics of degree 4
            return 16
        else:
            return 3  # Raw direction
    
    @property
    def total_samples(self) -> int:
        """Get total number of samples per ray."""
        total = self.num_coarse_samples
        if self.use_importance_sampling:
            total += self.num_fine_samples
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = super().to_dict()
        data.update({
            'nerf_type': self.nerf_type.value,
            'activation_type': self.activation_type.value,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NeRFConfig:
        """Create configuration from dictionary."""
        # Handle enum conversions
        if 'nerf_type' in data and isinstance(data['nerf_type'], str):
            data['nerf_type'] = NeRFType(data['nerf_type'])
        
        if 'activation_type' in data and isinstance(data['activation_type'], str):
            data['activation_type'] = ActivationType(data['activation_type'])
        
        return cls(**data)


@dataclass
class NeRFOutput:
    """Output of NeRF model for a set of points."""
    
    # Raw outputs
    density: torch.Tensor  # [N, 1] or [N, S, 1]
    color: torch.Tensor    # [N, 3] or [N, S, 3]
    
    # Optional features
    features: Optional[torch.Tensor] = None  # [N, F] or [N, S, F]
    normals: Optional[torch.Tensor] = None   # [N, 3] or [N, S, 3]
    semantic: Optional[torch.Tensor] = None  # [N, C] or [N, S, C]
    
    # Metadata
    positions: Optional[torch.Tensor] = None  # [N, 3] or [N, S, 3]
    view_directions: Optional[torch.Tensor] = None  # [N, 3] or [N, S, 3]
    
    def __post_init__(self):
        """Validate output shapes."""
        assert self.density.shape[:-1] == self.color.shape[:-1], \
            f"Density and color must have same batch shape, got {self.density.shape} and {self.color.shape}"
        
        if self.features is not None:
            assert self.density.shape[:-1] == self.features.shape[:-1], \
                f"Density and features must have same batch shape"
        
        if self.normals is not None:
            assert self.density.shape[:-1] == self.normals.shape[:-1], \
                f"Density and normals must have same batch shape"
    
    @property
    def batch_shape(self) -> torch.Size:
        """Get batch shape (excluding last dimension)."""
        return self.density.shape[:-1]
    
    @property
    def batch_size(self) -> int:
        """Get batch size (product of batch dimensions)."""
        return self.density.shape[:-1].numel()
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert output to dictionary."""
        result = {
            'density': self.density,
            'color': self.color,
        }
        
        if self.features is not None:
            result['features'] = self.features
        
        if self.normals is not None:
            result['normals'] = self.normals
        
        if self.semantic is not None:
            result['semantic'] = self.semantic
        
        if self.positions is not None:
            result['positions'] = self.positions
        
        if self.view_directions is not None:
            result['view_directions'] = self.view_directions
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> NeRFOutput:
        """Create output from dictionary."""
        return cls(
            density=data['density'],
            color=data['color'],
            features=data.get('features'),
            normals=data.get('normals'),
            semantic=data.get('semantic'),
            positions=data.get('positions'),
            view_directions=data.get('view_directions'),
        )
    
    def detach(self) -> NeRFOutput:
        """Detach all tensors from computation graph."""
        return NeRFOutput(
            density=self.density.detach(),
            color=self.color.detach(),
            features=self.features.detach() if self.features is not None else None,
            normals=self.normals.detach() if self.normals is not None else None,
            semantic=self.semantic.detach() if self.semantic is not None else None,
            positions=self.positions.detach() if self.positions is not None else None,
            view_directions=self.view_directions.detach() if self.view_directions is not None else None,
        )
    
    def to(self, device: torch.device) -> NeRFOutput:
        """Move all tensors to device."""
        return NeRFOutput(
            density=self.density.to(device),
            color=self.color.to(device),
            features=self.features.to(device) if self.features is not None else None,
            normals=self.normals.to(device) if self.normals is not None else None,
            semantic=self.semantic.to(device) if self.semantic is not None else None,
            positions=self.positions.to(device) if self.positions is not None else None,
            view_directions=self.view_directions.to(device) if self.view_directions is not None else None,
        )
    
    def cpu(self) -> NeRFOutput:
        """Move all tensors to CPU."""
        return self.to(torch.device('cpu'))


@dataclass
class TrainingData:
    """Data for training NeRF."""
    
    # Images and poses
    images: torch.Tensor  # [N, H, W, 3] or [N, C, H, W]
    camera_poses: torch.Tensor  # [N, 4, 4] or [N, 3, 4]
    intrinsics: torch.Tensor  # [N, 3, 3] or [N, 4] (fx, fy, cx, cy)
    
    # Optional data
    bounds: Optional[torch.Tensor] = None  # [N, 2] or [2]
    masks: Optional[torch.Tensor] = None  # [N, H, W, 1] or [N, 1, H, W]
    depths: Optional[torch.Tensor] = None  # [N, H, W, 1] or [N, 1, H, W]
    normals: Optional[torch.Tensor] = None  # [N, H, W, 3] or [N, 3, H, W]
    
    # Metadata
    image_ids: Optional[torch.Tensor] = None  # [N]
    timestamps: Optional[torch.Tensor] = None  # [N] for dynamic scenes
    appearance_ids: Optional[torch.Tensor] = None  # [N] for appearance conditioning
    
    def __post_init__(self):
        """Validate training data."""
        # Check dimensions
        num_images = self.images.shape[0]
        
        assert self.camera_poses.shape[0] == num_images, \
            f"Number of poses ({self.camera_poses.shape[0]}) must match number of images ({num_images})"
        
        assert self.intrinsics.shape[0] == num_images, \
            f"Number of intrinsics ({self.intrinsics.shape[0]}) must match number of images ({num_images})"
        
        # Convert images to channel-first format if needed
        if self.images.dim() == 4 and self.images.shape[-1] == 3:
            # [N, H, W, 3] -> [N, 3, H, W]
            self.images = self.images.permute(0, 3, 1, 2)
        
        # Convert other tensors if needed
        if self.masks is not None and self.masks.dim() == 4 and self.masks.shape[-1] == 1:
            self.masks = self.masks.permute(0, 3, 1, 2)
        
        if self.depths is not None and self.depths.dim() == 4 and self.depths.shape[-1] == 1:
            self.depths = self.depths.permute(0, 3, 1, 2)
        
        if self.normals is not None and self.normals.dim() == 4 and self.normals.shape[-1] == 3:
            self.normals = self.normals.permute(0, 3, 1, 2)
    
    @property
    def num_images(self) -> int:
        """Get number of images."""
        return self.images.shape[0]
    
    @property
    def image_size(self) -> Tuple[int, int]:
        """Get image size (height, width)."""
        return self.images.shape[2], self.images.shape[3]
    
    @property
    def device(self) -> torch.device:
        """Get device of data."""
        return self.images.device
    
    def to(self, device: torch.device) -> TrainingData:
        """Move all tensors to device."""
        return TrainingData(
            images=self.images.to(device),
            camera_poses=self.camera_poses.to(device),
            intrinsics=self.intrinsics.to(device),
            bounds=self.bounds.to(device) if self.bounds is not None else None,
            masks=self.masks.to(device) if self.masks is not None else None,
            depths=self.depths.to(device) if self.depths is not None else None,
            normals=self.normals.to(device) if self.normals is not None else None,
            image_ids=self.image_ids.to(device) if self.image_ids is not None else None,
            timestamps=self.timestamps.to(device) if self.timestamps is not None else None,
            appearance_ids=self.appearance_ids.to(device) if self.appearance_ids is not None else None,
        )
    
    def split(self, split_ratio: float = 0.8, shuffle: bool = True) -> Tuple[TrainingData, TrainingData]:
        """
        Split data into training and validation sets.
        
        Args:
            split_ratio: Ratio of training data
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_data, val_data)
        """
        num_images = self.num_images
        indices = torch.arange(num_images)
        
        if shuffle:
            indices = indices[torch.randperm(num_images)]
        
        split_idx = int(num_images * split_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        def select_data(data, idx):
            if data is None:
                return None
            return data[idx]
        
        train_data = TrainingData(
            images=select_data(self.images, train_indices),
            camera_poses=select_data(self.camera_poses, train_indices),
            intrinsics=select_data(self.intrinsics, train_indices),
            bounds=select_data(self.bounds, train_indices),
            masks=select_data(self.masks, train_indices),
            depths=select_data(self.depths, train_indices),
            normals=select_data(self.normals, train_indices),
            image_ids=select_data(self.image_ids, train_indices),
            timestamps=select_data(self.timestamps, train_indices),
            appearance_ids=select_data(self.appearance_ids, train_indices),
        )
        
        val_data = TrainingData(
            images=select_data(self.images, val_indices),
            camera_poses=select_data(self.camera_poses, val_indices),
            intrinsics=select_data(self.intrinsics, val_indices),
            bounds=select_data(self.bounds, val_indices),
            masks=select_data(self.masks, val_indices),
            depths=select_data(self.depths, val_indices),
            normals=select_data(self.normals, val_indices),
            image_ids=select_data(self.image_ids, val_indices),
            timestamps=select_data(self.timestamps, val_indices),
            appearance_ids=select_data(self.appearance_ids, val_indices),
        )
        
        return train_data, val_data
    
    def get_batch(self, batch_size: int, image_indices: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Get a batch of rays for training.
        
        Args:
            batch_size: Number of rays per batch
            image_indices: Specific image indices to sample from
            
        Returns:
            Dictionary with batch data
        """
        # Sample random image indices if not provided
        if image_indices is None:
            image_indices = torch.randint(0, self.num_images, (batch_size,), device=self.device)
        
        # Sample random pixel coordinates
        height, width = self.image_size
        x = torch.randint(0, width, (batch_size,), device=self.device)
        y = torch.randint(0, height, (batch_size,), device=self.device)
        
        # Get ray origins and directions
        rays_o, rays_d = self._get_rays(image_indices, x, y)
        
        # Get target colors
        colors = self.images[image_indices, :, y, x]  # [batch_size, 3]
        
        # Prepare batch
        batch = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            'target_colors': colors,
            'image_indices': image_indices,
            'pixel_coords': torch.stack([x, y], dim=-1),
        }
        
        # Add optional data
        if self.masks is not None:
            batch['masks'] = self.masks[image_indices, :, y, x]
        
        if self.depths is not None:
            batch['depths'] = self.depths[image_indices, :, y, x]
        
        if self.normals is not None:
            batch['normals'] = self.normals[image_indices, :, y, x]
        
        if self.timestamps is not None:
            batch['timestamps'] = self.timestamps[image_indices]
        
        if self.appearance_ids is not None:
            batch['appearance_ids'] = self.appearance_ids[image_indices]
        
        return batch
    
    def _get_rays(self, image_indices: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ray origins and directions for given image indices and pixel coordinates.
        
        Args:
            image_indices: Image indices [B]
            x: X coordinates [B]
            y: Y coordinates [B]
            
        Returns:
            Tuple of (ray_origins [B, 3], ray_directions [B, 3])
        """
        batch_size = image_indices.shape[0]
        
        # Get camera poses
        poses = self.camera_poses[image_indices]  # [B, 3, 4] or [B, 4, 4]
        
        # Extract rotation and translation
        if poses.shape[-2:] == (4, 4):
            rotation = poses[:, :3, :3]  # [B, 3, 3]
            translation = poses[:, :3, 3]  # [B, 3]
        else:
            rotation = poses[:, :3, :3]  # [B, 3, 3]
            translation = poses[:, :3, 3]  # [B, 3]
        
        # Get intrinsics
        intrinsics = self.intrinsics[image_indices]  # [B, 3, 3] or [B, 4]
        
        if intrinsics.dim() == 2 and intrinsics.shape[-1] == 4:
            # [fx, fy, cx, cy] format
            fx = intrinsics[:, 0].unsqueeze(-1)
            fy = intrinsics[:, 1].unsqueeze(-1)
            cx = intrinsics[:, 2].unsqueeze(-1)
            cy = intrinsics[:, 3].unsqueeze(-1)
        else:
            # [3, 3] matrix format
            fx = intrinsics[:, 0, 0].unsqueeze(-1)
            fy = intrinsics[:, 1, 1].unsqueeze(-1)
            cx = intrinsics[:, 0, 2].unsqueeze(-1)
            cy = intrinsics[:, 1, 2].unsqueeze(-1)
        
        # Convert pixel coordinates to normalized device coordinates
        x_ndc = (x.float() - cx) / fx
        y_ndc = (y.float() - cy) / fy
        
        # Create direction vectors in camera space
        directions = torch.stack([x_ndc, y_ndc, torch.ones_like(x_ndc)], dim=-1)  # [B, 3]
        
        # Transform directions to world space
        ray_directions = torch.bmm(rotation, directions.unsqueeze(-1)).squeeze(-1)
        ray_directions = F.normalize(ray_directions, dim=-1)
        
        # Ray origins are camera positions
        ray_origins = translation
        
        return ray_origins, ray_directions


class NeRFFactory:
    """Factory for creating NeRF models and components."""
    
    @staticmethod
    def create_model(
        config: Optional[NeRFConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> NeRFModel:
        """
        Create a NeRF model.
        
        Args:
            config: NeRF configuration
            device: PyTorch device
            **kwargs: Additional configuration parameters
            
        Returns:
            NeRF model
        """
        if config is None:
            config = NeRFConfig(**kwargs)
        
        return NeRFModel(config, device)
    
    @staticmethod
    def create_ray_sampler(
        near: float = 0.0,
        far: float = 1.0,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        perturb: bool = True,
        inverse_sphere_bg: bool = False,
        **kwargs
    ) -> RaySampler:
        """
        Create a ray sampler.
        
        Args:
            near: Near plane distance
            far: Far plane distance
            num_coarse_samples: Number of coarse samples
            num_fine_samples: Number of fine samples
            perturb: Whether to perturb sample positions
            inverse_sphere_bg: Whether to use inverse sphere background
            **kwargs: Additional parameters
            
        Returns:
            Ray sampler
        """
        sampling_config = SamplingConfig(
            near=near,
            far=far,
            num_coarse_samples=num_coarse_samples,
            num_fine_samples=num_fine_samples,
            perturb=perturb,
            inverse_sphere_bg=inverse_sphere_bg,
            **kwargs
        )
        
        return RaySampler(sampling_config)
    
    @staticmethod
    def create_volume_renderer(
        white_bkgd: bool = True,
        raw_noise_std: float = 0.0,
        min_alpha: float = 1e-5,
        **kwargs
    ) -> VolumeRenderer:
        """
        Create a volume renderer.
        
        Args:
            white_bkgd: Whether to use white background
            raw_noise_std: Standard deviation of noise added to raw density
            min_alpha: Minimum alpha value for stability
            **kwargs: Additional parameters
            
        Returns:
            Volume renderer
        """
        rendering_config = VolumeRenderingConfig(
            white_bkgd=white_bkgd,
            raw_noise_std=raw_noise_std,
            min_alpha=min_alpha,
            **kwargs
        )
        
        return VolumeRenderer(rendering_config)
    
    @staticmethod
    def create_positional_encoding(
        num_frequencies: int = 10,
        include_identity: bool = True,
        log_sampling: bool = True,
        **kwargs
    ) -> PositionalEncoding:
        """
        Create a positional encoding.
        
        Args:
            num_frequencies: Number of frequency bands
            include_identity: Whether to include raw coordinates
            log_sampling: Whether to use log sampling of frequencies
            **kwargs: Additional parameters
            
        Returns:
            Positional encoding
        """
        encoding_config = PositionalEncodingConfig(
            num_frequencies=num_frequencies,
            include_identity=include_identity,
            log_sampling=log_sampling,
            **kwargs
        )
        
        return PositionalEncoding(encoding_config)
    
    @staticmethod
    def create_default_nerf(
        device: Optional[torch.device] = None,
        nerf_type: Union[str, NeRFType] = NeRFType.VANILLA,
        **kwargs
    ) -> Tuple[NeRFModel, RaySampler, VolumeRenderer]:
        """
        Create a complete NeRF pipeline with default components.
        
        Args:
            device: PyTorch device
            nerf_type: Type of NeRF model
            **kwargs: Additional configuration parameters
            
        Returns:
            Tuple of (model, ray_sampler, volume_renderer)
        """
        if isinstance(nerf_type, str):
            nerf_type = NeRFType(nerf_type.lower())
        
        # Create configuration
        config = NeRFConfig(nerf_type=nerf_type, **kwargs)
        
        # Create components
        model = NeRFFactory.create_model(config, device)
        ray_sampler = NeRFFactory.create_ray_sampler(
            near=config.near_distance,
            far=config.far_distance,
            num_coarse_samples=config.num_coarse_samples,
            num_fine_samples=config.num_fine_samples,
            perturb=config.perturb,
            inverse_sphere_bg=config.inverse_sphere_bg,
        )
        volume_renderer = NeRFFactory.create_volume_renderer(
            white_bkgd=config.white_bkgd,
            raw_noise_std=config.raw_noise_std,
            min_alpha=config.min_alpha,
        )
        
        return model, ray_sampler, volume_renderer


# ============================================================================
# TRAINING AND RENDERING UTILITIES
# ============================================================================

def train_nerf(
    model: NeRFModel,
    train_data: TrainingData,
    val_data: Optional[TrainingData] = None,
    num_iterations: int = 10000,
    batch_size: int = 4096,
    learning_rate: float = 5e-4,
    lr_decay: float = 0.1,
    lr_decay_steps: int = 2500,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_freq: int = 1000,
    log_freq: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a NeRF model.
    
    Args:
        model: NeRF model to train
        train_data: Training data
        val_data: Validation data (optional)
        num_iterations: Number of training iterations
        batch_size: Batch size (number of rays per iteration)
        learning_rate: Learning rate
        lr_decay: Learning rate decay factor
        lr_decay_steps: Steps between learning rate decays
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Frequency of checkpoints (iterations)
        log_freq: Frequency of logging (iterations)
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary with training results
    """
    import time
    from tqdm import tqdm
    
    # Setup
    device = model.device
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=lr_decay_steps, 
        gamma=lr_decay
    )
    
    # Create ray sampler and volume renderer
    ray_sampler = NeRFFactory.create_ray_sampler(
        near=model.config.near_distance,
        far=model.config.far_distance,
        num_coarse_samples=model.config.num_coarse_samples,
        num_fine_samples=model.config.num_fine_samples,
        perturb=model.config.perturb,
    )
    
    volume_renderer = NeRFFactory.create_volume_renderer(
        white_bkgd=model.config.white_bkgd,
        raw_noise_std=model.config.raw_noise_std,
    )
    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    psnrs = []
    
    progress_bar = tqdm(range(num_iterations), desc="Training NeRF")
    
    for iteration in progress_bar:
        # Get training batch
        batch = train_data.get_batch(batch_size)
        
        # Extract batch data
        rays_o = batch['rays_o'].to(device)
        rays_d = batch['rays_d'].to(device)
        target_colors = batch['target_colors'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Sample points along rays
        ray_bundle = ray_sampler.sample_along_rays(rays_o, rays_d)
        
        # Query NeRF at sampled points
        nerf_output = model.query_points(
            ray_bundle.samples.positions,
            ray_bundle.samples.view_directions,
        )
        
        # Render colors
        rendering_output = volume_renderer.render(
            nerf_output.density,
            nerf_output.color,
            ray_bundle.samples.depths,
            ray_bundle.samples.deltas,
        )
        
        # Compute loss
        loss = F.mse_loss(rendering_output.colors, target_colors)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update statistics
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        # Compute PSNR
        with torch.no_grad():
            mse = F.mse_loss(rendering_output.colors, target_colors)
            psnr = -10.0 * torch.log10(mse)
            psnrs.append(psnr.item())
        
        # Validation
        if val_data is not None and iteration % log_freq == 0:
            model.eval()
            with torch.no_grad():
                val_batch = val_data.get_batch(min(batch_size, 1024))
                val_rays_o = val_batch['rays_o'].to(device)
                val_rays_d = val_batch['rays_d'].to(device)
                val_target_colors = val_batch['target_colors'].to(device)
                
                # Sample and render
                val_ray_bundle = ray_sampler.sample_along_rays(val_rays_o, val_rays_d)
                val_nerf_output = model.query_points(
                    val_ray_bundle.samples.positions,
                    val_ray_bundle.samples.view_directions,
                )
                val_rendering_output = volume_renderer.render(
                    val_nerf_output.density,
                    val_nerf_output.color,
                    val_ray_bundle.samples.depths,
                    val_ray_bundle.samples.deltas,
                )
                val_loss = F.mse_loss(val_rendering_output.colors, val_target_colors).item()
                val_losses.append(val_loss)
            
            model.train()
        
        # Logging
        if iteration % log_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{train_loss:.6f}',
                'psnr': f'{psnrs[-1]:.2f}',
                'lr': f'{current_lr:.2e}',
            })
        
        # Checkpoint
        if checkpoint_dir is not None and iteration % checkpoint_freq == 0:
            checkpoint_path = Path(checkpoint_dir) / f'checkpoint_{iteration:06d}.pt'
            model.save_checkpoint(checkpoint_path, optimizer=optimizer, scheduler=scheduler)
    
    # Training complete
    training_time = time.time() - start_time
    
    # Prepare results
    results = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'psnrs': psnrs,
        'training_time': training_time,
        'num_iterations': num_iterations,
        'final_loss': train_losses[-1] if train_losses else 0.0,
        'final_psnr': psnrs[-1] if psnrs else 0.0,
    }
    
    return results


def render_nerf_views(
    model: NeRFModel,
    camera_poses: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    batch_size: int = 4096,
    chunk_size: int = 65536,
    **kwargs
) -> torch.Tensor:
    """
    Render views from a trained NeRF model.
    
    Args:
        model: Trained NeRF model
        camera_poses: Camera poses [N, 3, 4] or [N, 4, 4]
        intrinsics: Camera intrinsics [N, 3, 3] or [N, 4]
        image_size: Output image size (height, width)
        batch_size: Batch size for rendering
        chunk_size: Chunk size for processing large numbers of rays
        **kwargs: Additional rendering parameters
        
    Returns:
        Rendered images [N, 3, H, W]
    """
    model.eval()
    device = model.device
    
    height, width = image_size
    num_views = camera_poses.shape[0]
    
    # Create ray sampler and volume renderer
    ray_sampler = NeRFFactory.create_ray_sampler(
        near=model.config.near_distance,
        far=model.config.far_distance,
        num_coarse_samples=model.config.num_coarse_samples,
        num_fine_samples=model.config.num_fine_samples,
        perturb=False,  # No perturbation for evaluation
    )
    
    volume_renderer = NeRFFactory.create_volume_renderer(
        white_bkgd=model.config.white_bkgd,
        raw_noise_std=0.0,  # No noise for evaluation
    )
    
    # Generate all pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    x = x.reshape(-1)
    y = y.reshape(-1)
    
    # Prepare output images
    images = torch.zeros((num_views, 3, height, width), device=device)
    
    with torch.no_grad():
        for view_idx in range(num_views):
            # Get camera pose and intrinsics for this view
            pose = camera_poses[view_idx].unsqueeze(0).to(device)
            K = intrinsics[view_idx].unsqueeze(0).to(device) if intrinsics.dim() > 1 else intrinsics.to(device)
            
            # Render in chunks
            num_pixels = height * width
            for chunk_start in range(0, num_pixels, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_pixels)
                chunk_size_current = chunk_end - chunk_start
                
                # Get pixel coordinates for this chunk
                x_chunk = x[chunk_start:chunk_end]
                y_chunk = y[chunk_start:chunk_end]
                
                # Repeat pose and intrinsics for chunk
                pose_chunk = pose.repeat(chunk_size_current, 1, 1)
                K_chunk = K.repeat(chunk_size_current, 1, 1) if K.dim() > 2 else K.repeat(chunk_size_current, 1)
                
                # Generate rays for chunk
                # This is simplified - in practice, you'd use a proper ray generation function
                rays_o, rays_d = _generate_rays_for_chunk(
                    pose_chunk, K_chunk, x_chunk, y_chunk, height, width
                )
                
                # Process rays in batches
                for batch_start in range(0, chunk_size_current, batch_size):
                    batch_end = min(batch_start + batch_size, chunk_size_current)
                    batch_size_current = batch_end - batch_start
                    
                    # Get batch rays
                    batch_rays_o = rays_o[batch_start:batch_end]
                    batch_rays_d = rays_d[batch_start:batch_end]
                    
                    # Sample points along rays
                    ray_bundle = ray_sampler.sample_along_rays(batch_rays_o, batch_rays_d)
                    
                    # Query NeRF
                    nerf_output = model.query_points(
                        ray_bundle.samples.positions,
                        ray_bundle.samples.view_directions,
                    )
                    
                    # Render colors
                    rendering_output = volume_renderer.render(
                        nerf_output.density,
                        nerf_output.color,
                        ray_bundle.samples.depths,
                        ray_bundle.samples.deltas,
                    )
                    
                    # Store rendered colors
                    chunk_start_global = chunk_start + batch_start
                    chunk_end_global = chunk_start + batch_end
                    
                    # Convert from flattened to 2D
                    for i in range(batch_size_current):
                        pixel_idx = chunk_start_global + i
                        y_idx = pixel_idx // width
                        x_idx = pixel_idx % width
                        
                        images[view_idx, :, y_idx, x_idx] = rendering_output.colors[i]
    
    return images


def _generate_rays_for_chunk(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    height: int,
    width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for a chunk of pixels.
    
    Args:
        poses: Camera poses [N, 3, 4]
        intrinsics: Camera intrinsics [N, 3, 3] or [N, 4]
        x: X coordinates [N]
        y: Y coordinates [N]
        height: Image height
        width: Image width
        
    Returns:
        Tuple of (ray_origins [N, 3], ray_directions [N, 3])
    """
    batch_size = poses.shape[0]
    
    # Extract rotation and translation
    rotation = poses[:, :3, :3]  # [N, 3, 3]
    translation = poses[:, :3, 3]  # [N, 3]
    
    # Get intrinsics parameters
    if intrinsics.dim() == 2 and intrinsics.shape[-1] == 4:
        # [fx, fy, cx, cy] format
        fx = intrinsics[:, 0].unsqueeze(-1)
        fy = intrinsics[:, 1].unsqueeze(-1)
        cx = intrinsics[:, 2].unsqueeze(-1)
        cy = intrinsics[:, 3].unsqueeze(-1)
    else:
        # [3, 3] matrix format
        fx = intrinsics[:, 0, 0].unsqueeze(-1)
        fy = intrinsics[:, 1, 1].unsqueeze(-1)
        cx = intrinsics[:, 0, 2].unsqueeze(-1)
        cy = intrinsics[:, 1, 2].unsqueeze(-1)
    
    # Convert pixel coordinates to normalized device coordinates
    x_ndc = (x.float() - cx) / fx
    y_ndc = (y.float() - cy) / fy
    
    # Create direction vectors in camera space
    directions = torch.stack([x_ndc, y_ndc, torch.ones_like(x_ndc)], dim=-1)  # [N, 3]
    
    # Transform directions to world space
    ray_directions = torch.bmm(rotation, directions.unsqueeze(-1)).squeeze(-1)
    ray_directions = F.normalize(ray_directions, dim=-1)
    
    # Ray origins are camera positions
    ray_origins = translation
    
    return ray_origins, ray_directions


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_nerf(
    nerf_type: Union[str, NeRFType] = NeRFType.VANILLA,
    config: Optional[Union[Dict[str, Any], NeRFConfig]] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[NeRFModel, RaySampler, VolumeRenderer]:
    """
    Create a complete NeRF pipeline.
    
    Args:
        nerf_type: Type of NeRF model
        config: Configuration dictionary or object
        device: PyTorch device
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (model, ray_sampler, volume_renderer)
    """
    return NeRFFactory.create_default_nerf(device, nerf_type, **(config or {}), **kwargs)


def load_nerf(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    load_optimizer: bool = False,
    **kwargs
) -> Tuple[NeRFModel, Optional[Dict[str, Any]]]:
    """
    Load a NeRF model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
        load_optimizer: Whether to load optimizer state
        **kwargs: Additional loading parameters
        
    Returns:
        Tuple of (model, checkpoint_data)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get configuration
    config_dict = checkpoint.get('config', {})
    config = NeRFConfig.from_dict(config_dict)
    
    # Create model
    model = NeRFFactory.create_model(config, device)
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Prepare return data
    checkpoint_data = {
        'iteration': checkpoint.get('iteration', 0),
        'loss_history': checkpoint.get('loss_history', []),
        'config': config,
    }
    
    if load_optimizer and 'optimizer_state_dict' in checkpoint:
        checkpoint_data['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
    
    return model, checkpoint_data


def extract_mesh_from_nerf(
    model: NeRFModel,
    resolution: int = 256,
    threshold: float = 25.0,
    chunk_size: int = 65536,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract mesh from NeRF using marching cubes.
    
    Args:
        model: Trained NeRF model
        resolution: Marching cubes resolution
        threshold: Density threshold for surface extraction
        chunk_size: Chunk size for processing
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (vertices [V, 3], faces [F, 3])
    """
    import mcubes
    
    model.eval()
    device = model.device
    
    # Create grid
    bounds = model.get_bounds()
    min_bound, max_bound = bounds
    
    # Generate grid coordinates
    x = torch.linspace(min_bound[0], max_bound[0], resolution, device=device)
    y = torch.linspace(min_bound[1], max_bound[1], resolution, device=device)
    z = torch.linspace(min_bound[2], max_bound[2], resolution, device=device)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)  # [N, 3]
    
    # Query density in chunks
    densities = torch.zeros(grid_points.shape[0], device=device)
    
    with torch.no_grad():
        for chunk_start in range(0, grid_points.shape[0], chunk_size):
            chunk_end = min(chunk_start + chunk_size, grid_points.shape[0])
            chunk_points = grid_points[chunk_start:chunk_end]
            
            # Query NeRF (no view dependence for density)
            nerf_output = model.query_points(chunk_points, None)
            densities[chunk_start:chunk_end] = nerf_output.density.squeeze(-1)
    
    # Reshape to 3D grid
    density_grid = densities.reshape(resolution, resolution, resolution).cpu().numpy()
    
    # Marching cubes
    vertices, faces = mcubes.marching_cubes(density_grid, threshold)
    
    # Scale vertices to world coordinates
    scale = (max_bound - min_bound).cpu().numpy()
    offset = min_bound.cpu().numpy()
    
    vertices = vertices / (resolution - 1) * scale + offset
    
    # Convert to torch tensors
    vertices = torch.from_numpy(vertices).float().to(device)
    faces = torch.from_numpy(faces.astype(np.int32)).long().to(device)
    
    return vertices, faces


def compute_nerf_metrics(
    model: NeRFModel,
    test_data: TrainingData,
    batch_size: int = 4096,
    **kwargs
) -> RepresentationMetrics:
    """
    Compute metrics for NeRF model.
    
    Args:
        model: Trained NeRF model
        test_data: Test data
        batch_size: Batch size for evaluation
        **kwargs: Additional parameters
        
    Returns:
        RepresentationMetrics object
    """
    import time
    
    model.eval()
    device = model.device
    
    # Create components
    ray_sampler = NeRFFactory.create_ray_sampler(
        near=model.config.near_distance,
        far=model.config.far_distance,
        num_coarse_samples=model.config.num_coarse_samples,
        num_fine_samples=model.config.num_fine_samples,
        perturb=False,
    )
    
    volume_renderer = NeRFFactory.create_volume_renderer(
        white_bkgd=model.config.white_bkgd,
        raw_noise_std=0.0,
    )
    
    # Initialize metrics
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    num_batches = 0
    
    # LPIPS model (if available)
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
    except ImportError:
        pass
    
    start_time = time.time()
    
    with torch.no_grad():
        # Process all test images
        for image_idx in range(test_data.num_images):
            # Get image
            image = test_data.images[image_idx:image_idx+1]  # [1, 3, H, W]
            pose = test_data.camera_poses[image_idx:image_idx+1]
            K = test_data.intrinsics[image_idx:image_idx+1]
            
            height, width = test_data.image_size
            
            # Generate rays for entire image (in chunks)
            for chunk_start in range(0, height * width, batch_size):
                chunk_end = min(chunk_start + batch_size, height * width)
                
                # Generate pixel coordinates for chunk
                y = torch.arange(height, device=device)
                x = torch.arange(width, device=device)
                y, x = torch.meshgrid(y, x, indexing='ij')
                x = x.flatten()[chunk_start:chunk_end]
                y = y.flatten()[chunk_start:chunk_end]
                
                # Generate rays
                rays_o, rays_d = test_data._get_rays(
                    torch.tensor([image_idx], device=device).repeat(len(x)),
                    x, y
                )
                
                # Sample and render
                ray_bundle = ray_sampler.sample_along_rays(rays_o, rays_d)
                nerf_output = model.query_points(
                    ray_bundle.samples.positions,
                    ray_bundle.samples.view_directions,
                )
                rendering_output = volume_renderer.render(
                    nerf_output.density,
                    nerf_output.color,
                    ray_bundle.samples.depths,
                    ray_bundle.samples.deltas,
                )
                
                # Get target colors
                target_colors = image[0, :, y, x].t()  # [N, 3]
                
                # Compute MSE and PSNR
                mse = F.mse_loss(rendering_output.colors, target_colors)
                psnr = -10.0 * torch.log10(mse)
                
                total_mse += mse.item()
                total_psnr += psnr.item()
                num_batches += 1
    
    inference_time = time.time() - start_time
    
    # Compute averages
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0.0
    
    # Create metrics
    metrics = RepresentationMetrics(
        mse=avg_mse,
        psnr_db=avg_psnr,
        inference_time_ms=inference_time * 1000 / num_batches if num_batches > 0 else 0.0,
    )
    
    return metrics


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Register NeRF type with the global registry
try:
    from ..base import RepresentationRegistry
    RepresentationRegistry.register('nerf', NeRFModel, aliases=['neural_radiance_field', 'radiance_field'])
except ImportError:
    pass