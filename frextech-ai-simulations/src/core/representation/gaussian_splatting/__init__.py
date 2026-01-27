"""
Gaussian Splatting Module.

This module implements 3D Gaussian Splatting for real-time novel view synthesis,
including Gaussian representation, optimization, and rasterization.
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
from .gaussian_model import GaussianModel, GaussianConfig, GaussianParameters
from .rasterizer import GaussianRasterizer, RasterizerConfig, RasterizerOutput
from .optimizer import GaussianOptimizer, OptimizationConfig, OptimizationState

__all__ = [
    # Main classes
    'GaussianModel',
    'GaussianConfig',
    'GaussianParameters',
    
    # Rasterization
    'GaussianRasterizer',
    'RasterizerConfig',
    'RasterizerOutput',
    
    # Optimization
    'GaussianOptimizer',
    'OptimizationConfig',
    'OptimizationState',
    
    # Enums and types
    'GaussianType',
    'ShadingType',
    'BlendingMode',
    
    # Factory and utilities
    'GaussianFactory',
    'create_gaussian_model',
    'load_gaussian_model',
    'train_gaussian_splatting',
    'render_gaussian_views',
]


class GaussianType(Enum):
    """Types of Gaussian representations."""
    STANDARD = "standard"          # Standard 3D Gaussian splatting
    ANISOTROPIC = "anisotropic"    # Anisotropic Gaussians
    SPHERICAL = "spherical"        # Spherical harmonics Gaussians
    DEFORMABLE = "deformable"      # Deformable Gaussians
    DYNAMIC = "dynamic"            # Dynamic Gaussians (for video)
    COMPRESSED = "compressed"      # Compressed Gaussians


class ShadingType(Enum):
    """Types of shading for Gaussians."""
    DIFFUSE = "diffuse"            # Diffuse shading
    LAMBERTIAN = "lambertian"      # Lambertian shading
    PHONG = "phong"                # Phong shading
    BLINN_PHONG = "blinn_phong"    # Blinn-Phong shading
    PBR = "pbr"                    # Physically-based rendering
    SPHERICAL_HARMONICS = "sh"     # Spherical harmonics


class BlendingMode(Enum):
    """Blending modes for Gaussian rasterization."""
    ALPHA_BLENDING = "alpha"       # Alpha blending
    ADDITIVE = "additive"          # Additive blending
    MAX = "max"                    # Max blending
    DEPTH_PEELING = "depth_peeling"  # Depth peeling
    WEIGHTED_AVERAGE = "weighted"  # Weighted average


@dataclass
class GaussianConfig(RepresentationConfig):
    """Configuration for Gaussian splatting model."""
    
    # Gaussian type
    gaussian_type: GaussianType = GaussianType.STANDARD
    
    # Gaussian parameters
    max_gaussians: int = 100000
    init_gaussians: int = 1000
    sh_degree: int = 3  # Spherical harmonics degree (0 = diffuse, 3 = full color)
    
    # Initialization
    init_from_points: bool = True
    init_scale: float = 0.01
    init_opacity: float = 0.1
    init_color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    
    # Optimization
    position_lr: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30000
    
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # Adaptive density control
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_grad_threshold: float = 0.0002
    densify_size_threshold: float = 0.01
    prune_opacity_threshold: float = 0.005
    prune_scale_threshold: float = 0.5
    
    # Cloning and splitting
    clone_threshold: float = 0.3
    split_threshold: float = 0.2
    max_clones: int = 2
    max_splits: int = 2
    
    # Regularization
    position_regularization: float = 0.0
    scale_regularization: float = 0.0
    rotation_regularization: float = 0.0
    opacity_regularization: float = 0.0
    
    # Shading
    shading_type: ShadingType = ShadingType.SPHERICAL_HARMONICS
    use_view_direction: bool = True
    ambient_light: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    diffuse_light: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8])
    specular_light: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    shininess: float = 32.0
    
    # Rendering
    blending_mode: BlendingMode = BlendingMode.ALPHA_BLENDING
    tile_size: int = 16  # For tile-based rasterization
    max_tiles: int = 1024
    depth_sorting: bool = True
    culling_radius: float = 0.0  # Cull Gaussians beyond this radius
    
    # Anti-aliasing
    use_antialiasing: bool = True
    antialiasing_sigma: float = 0.5
    supersampling: int = 1
    
    # Compression
    use_compression: bool = False
    compression_ratio: float = 0.5
    quantization_bits: int = 8
    
    # Dynamic properties (for video)
    use_temporal: bool = False
    temporal_lr: float = 0.0001
    motion_blur: bool = False
    
    def __post_init__(self):
        """Validate Gaussian configuration."""
        super().__post_init__()
        
        # Convert string enums
        if isinstance(self.gaussian_type, str):
            self.gaussian_type = GaussianType(self.gaussian_type.lower())
        
        if isinstance(self.shading_type, str):
            self.shading_type = ShadingType(self.shading_type.lower())
        
        if isinstance(self.blending_mode, str):
            self.blending_mode = BlendingMode(self.blending_mode.lower())
        
        # Validate parameters
        if self.max_gaussians <= 0:
            raise ValueError(f"max_gaussians must be positive, got {self.max_gaussians}")
        
        if self.sh_degree < 0 or self.sh_degree > 4:
            raise ValueError(f"sh_degree must be between 0 and 4, got {self.sh_degree}")
        
        if self.init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {self.init_scale}")
        
        if not 0 <= self.init_opacity <= 1:
            raise ValueError(f"init_opacity must be in [0, 1], got {self.init_opacity}")
    
    @property
    def sh_dim(self) -> int:
        """Get spherical harmonics dimension."""
        # SH coefficients: (degree + 1)^2 * 3 (for RGB)
        return (self.sh_degree + 1) ** 2 * 3
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension (color + opacity + scale + rotation)."""
        # Position: 3, Rotation: 4 (quaternion), Scale: 3, Opacity: 1, SH: sh_dim
        return 3 + 4 + 3 + 1 + self.sh_dim
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = super().to_dict()
        data.update({
            'gaussian_type': self.gaussian_type.value,
            'shading_type': self.shading_type.value,
            'blending_mode': self.blending_mode.value,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GaussianConfig:
        """Create configuration from dictionary."""
        # Handle enum conversions
        if 'gaussian_type' in data and isinstance(data['gaussian_type'], str):
            data['gaussian_type'] = GaussianType(data['gaussian_type'])
        
        if 'shading_type' in data and isinstance(data['shading_type'], str):
            data['shading_type'] = ShadingType(data['shading_type'])
        
        if 'blending_mode' in data and isinstance(data['blending_mode'], str):
            data['blending_mode'] = BlendingMode(data['blending_mode'])
        
        return cls(**data)


@dataclass
class GaussianParameters:
    """Parameters of a 3D Gaussian."""
    
    # Core parameters
    positions: torch.Tensor  # [N, 3] (x, y, z)
    rotations: torch.Tensor  # [N, 4] (quaternion: qx, qy, qz, qw)
    scales: torch.Tensor     # [N, 3] (sx, sy, sz)
    opacities: torch.Tensor  # [N, 1] (alpha)
    
    # Appearance parameters
    features: torch.Tensor   # [N, F] (colors or SH coefficients)
    
    # Optional parameters
    velocities: Optional[torch.Tensor] = None  # [N, 3] for dynamic scenes
    accelerations: Optional[torch.Tensor] = None  # [N, 3] for dynamic scenes
    timestamps: Optional[torch.Tensor] = None  # [N, 1] for temporal scenes
    
    # Metadata
    valid_mask: Optional[torch.Tensor] = None  # [N] boolean mask
    ids: Optional[torch.Tensor] = None  # [N] unique identifiers
    
    def __post_init__(self):
        """Validate parameter shapes."""
        N = self.positions.shape[0]
        
        assert self.rotations.shape == (N, 4), \
            f"rotations shape {self.rotations.shape} != ({N}, 4)"
        
        assert self.scales.shape == (N, 3), \
            f"scales shape {self.scales.shape} != ({N}, 3)"
        
        assert self.opacities.shape == (N, 1), \
            f"opacities shape {self.opacities.shape} != ({N}, 1)"
        
        assert self.features.shape[0] == N, \
            f"features batch size {self.features.shape[0]} != {N}"
    
    @property
    def num_gaussians(self) -> int:
        """Get number of Gaussians."""
        return self.positions.shape[0]
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        return self.features.shape[1]
    
    @property
    def device(self) -> torch.device:
        """Get device of parameters."""
        return self.positions.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get data type of parameters."""
        return self.positions.dtype
    
    def to(self, device: torch.device) -> GaussianParameters:
        """Move parameters to device."""
        return GaussianParameters(
            positions=self.positions.to(device),
            rotations=self.rotations.to(device),
            scales=self.scales.to(device),
            opacities=self.opacities.to(device),
            features=self.features.to(device),
            velocities=self.velocities.to(device) if self.velocities is not None else None,
            accelerations=self.accelerations.to(device) if self.accelerations is not None else None,
            timestamps=self.timestamps.to(device) if self.timestamps is not None else None,
            valid_mask=self.valid_mask.to(device) if self.valid_mask is not None else None,
            ids=self.ids.to(device) if self.ids is not None else None,
        )
    
    def detach(self) -> GaussianParameters:
        """Detach parameters from computation graph."""
        return GaussianParameters(
            positions=self.positions.detach(),
            rotations=self.rotations.detach(),
            scales=self.scales.detach(),
            opacities=self.opacities.detach(),
            features=self.features.detach(),
            velocities=self.velocities.detach() if self.velocities is not None else None,
            accelerations=self.accelerations.detach() if self.accelerations is not None else None,
            timestamps=self.timestamps.detach() if self.timestamps is not None else None,
            valid_mask=self.valid_mask.detach() if self.valid_mask is not None else None,
            ids=self.ids.detach() if self.ids is not None else None,
        )
    
    def clone(self) -> GaussianParameters:
        """Clone parameters."""
        return GaussianParameters(
            positions=self.positions.clone(),
            rotations=self.rotations.clone(),
            scales=self.scales.clone(),
            opacities=self.opacities.clone(),
            features=self.features.clone(),
            velocities=self.velocities.clone() if self.velocities is not None else None,
            accelerations=self.accelerations.clone() if self.accelerations is not None else None,
            timestamps=self.timestamps.clone() if self.timestamps is not None else None,
            valid_mask=self.valid_mask.clone() if self.valid_mask is not None else None,
            ids=self.ids.clone() if self.ids is not None else None,
        )
    
    def slice(self, start: int, end: int) -> GaussianParameters:
        """Slice a subset of Gaussians."""
        return GaussianParameters(
            positions=self.positions[start:end],
            rotations=self.rotations[start:end],
            scales=self.scales[start:end],
            opacities=self.opacities[start:end],
            features=self.features[start:end],
            velocities=self.velocities[start:end] if self.velocities is not None else None,
            accelerations=self.accelerations[start:end] if self.accelerations is not None else None,
            timestamps=self.timestamps[start:end] if self.timestamps is not None else None,
            valid_mask=self.valid_mask[start:end] if self.valid_mask is not None else None,
            ids=self.ids[start:end] if self.ids is not None else None,
        )
    
    def filter_by_mask(self, mask: torch.Tensor) -> GaussianParameters:
        """Filter Gaussians by boolean mask."""
        return GaussianParameters(
            positions=self.positions[mask],
            rotations=self.rotations[mask],
            scales=self.scales[mask],
            opacities=self.opacities[mask],
            features=self.features[mask],
            velocities=self.velocities[mask] if self.velocities is not None else None,
            accelerations=self.accelerations[mask] if self.accelerations is not None else None,
            timestamps=self.timestamps[mask] if self.timestamps is not None else None,
            valid_mask=self.valid_mask[mask] if self.valid_mask is not None else None,
            ids=self.ids[mask] if self.ids is not None else None,
        )
    
    def get_covariance_matrices(self) -> torch.Tensor:
        """Compute 3x3 covariance matrices for each Gaussian."""
        # Convert quaternions to rotation matrices
        R = self._quaternion_to_rotation_matrix(self.rotations)
        
        # Create diagonal scale matrices
        S = torch.diag_embed(self.scales)
        
        # Compute covariance: R @ S @ S^T @ R^T
        SS = torch.bmm(S, S.transpose(1, 2))
        covariance = torch.bmm(torch.bmm(R, SS), R.transpose(1, 2))
        
        return covariance
    
    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternions to rotation matrices."""
        # Normalize quaternions
        q = F.normalize(q, dim=-1)
        
        qx, qy, qz, qw = q.unbind(-1)
        
        # Compute rotation matrix elements
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        xw = qx * qw
        yw = qy * qw
        zw = qz * qw
        
        # Assemble rotation matrix
        R = torch.stack([
            1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
            2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
            2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
        ], dim=-1).reshape(-1, 3, 3)
        
        return R
    
    def get_bounding_spheres(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bounding spheres for each Gaussian (center and radius)."""
        # Center is the position
        centers = self.positions
        
        # Approximate radius as maximum scale
        radii = torch.max(self.scales, dim=-1)[0]
        
        return centers, radii
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert parameters to dictionary."""
        result = {
            'positions': self.positions,
            'rotations': self.rotations,
            'scales': self.scales,
            'opacities': self.opacities,
            'features': self.features,
        }
        
        if self.velocities is not None:
            result['velocities'] = self.velocities
        
        if self.accelerations is not None:
            result['accelerations'] = self.accelerations
        
        if self.timestamps is not None:
            result['timestamps'] = self.timestamps
        
        if self.valid_mask is not None:
            result['valid_mask'] = self.valid_mask
        
        if self.ids is not None:
            result['ids'] = self.ids
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> GaussianParameters:
        """Create parameters from dictionary."""
        return cls(
            positions=data['positions'],
            rotations=data['rotations'],
            scales=data['scales'],
            opacities=data['opacities'],
            features=data['features'],
            velocities=data.get('velocities'),
            accelerations=data.get('accelerations'),
            timestamps=data.get('timestamps'),
            valid_mask=data.get('valid_mask'),
            ids=data.get('ids'),
        )


@dataclass
class TrainingData:
    """Data for training Gaussian splatting."""
    
    # Images and poses
    images: torch.Tensor  # [N, H, W, 3] or [N, 3, H, W]
    camera_poses: torch.Tensor  # [N, 4, 4] or [N, 3, 4]
    intrinsics: torch.Tensor  # [N, 3, 3] or [N, 4]
    
    # Optional data
    depths: Optional[torch.Tensor] = None  # [N, H, W, 1] or [N, 1, H, W]
    masks: Optional[torch.Tensor] = None  # [N, H, W, 1] or [N, 1, H, W]
    point_clouds: Optional[torch.Tensor] = None  # List of point clouds
    
    # Metadata
    image_ids: Optional[torch.Tensor] = None  # [N]
    timestamps: Optional[torch.Tensor] = None  # [N] for dynamic scenes
    
    def __post_init__(self):
        """Validate training data."""
        num_images = self.images.shape[0]
        
        assert self.camera_poses.shape[0] == num_images, \
            f"Number of poses ({self.camera_poses.shape[0]}) must match images ({num_images})"
        
        assert self.intrinsics.shape[0] == num_images, \
            f"Number of intrinsics ({self.intrinsics.shape[0]}) must match images ({num_images})"
        
        # Convert to channel-first format if needed
        if self.images.dim() == 4 and self.images.shape[-1] == 3:
            self.images = self.images.permute(0, 3, 1, 2)
        
        if self.depths is not None and self.depths.dim() == 4 and self.depths.shape[-1] == 1:
            self.depths = self.depths.permute(0, 3, 1, 2)
        
        if self.masks is not None and self.masks.dim() == 4 and self.masks.shape[-1] == 1:
            self.masks = self.masks.permute(0, 3, 1, 2)
    
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
        """Move data to device."""
        return TrainingData(
            images=self.images.to(device),
            camera_poses=self.camera_poses.to(device),
            intrinsics=self.intrinsics.to(device),
            depths=self.depths.to(device) if self.depths is not None else None,
            masks=self.masks.to(device) if self.masks is not None else None,
            point_clouds=self.point_clouds.to(device) if self.point_clouds is not None else None,
            image_ids=self.image_ids.to(device) if self.image_ids is not None else None,
            timestamps=self.timestamps.to(device) if self.timestamps is not None else None,
        )
    
    def split(self, split_ratio: float = 0.8, shuffle: bool = True) -> Tuple[TrainingData, TrainingData]:
        """Split data into training and validation sets."""
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
            depths=select_data(self.depths, train_indices),
            masks=select_data(self.masks, train_indices),
            point_clouds=select_data(self.point_clouds, train_indices),
            image_ids=select_data(self.image_ids, train_indices),
            timestamps=select_data(self.timestamps, train_indices),
        )
        
        val_data = TrainingData(
            images=select_data(self.images, val_indices),
            camera_poses=select_data(self.camera_poses, val_indices),
            intrinsics=select_data(self.intrinsics, val_indices),
            depths=select_data(self.depths, val_indices),
            masks=select_data(self.masks, val_indices),
            point_clouds=select_data(self.point_clouds, val_indices),
            image_ids=select_data(self.image_ids, val_indices),
            timestamps=select_data(self.timestamps, val_indices),
        )
        
        return train_data, val_data


class GaussianFactory:
    """Factory for creating Gaussian splatting components."""
    
    @staticmethod
    def create_model(
        config: Optional[GaussianConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> GaussianModel:
        """
        Create a Gaussian splatting model.
        
        Args:
            config: Gaussian configuration
            device: PyTorch device
            **kwargs: Additional configuration parameters
            
        Returns:
            Gaussian model
        """
        if config is None:
            config = GaussianConfig(**kwargs)
        
        return GaussianModel(config, device)
    
    @staticmethod
    def create_rasterizer(
        tile_size: int = 16,
        max_tiles: int = 1024,
        blending_mode: Union[str, BlendingMode] = BlendingMode.ALPHA_BLENDING,
        depth_sorting: bool = True,
        **kwargs
    ) -> GaussianRasterizer:
        """
        Create a Gaussian rasterizer.
        
        Args:
            tile_size: Tile size for tile-based rasterization
            max_tiles: Maximum number of tiles
            blending_mode: Blending mode
            depth_sorting: Whether to sort Gaussians by depth
            **kwargs: Additional parameters
            
        Returns:
            Gaussian rasterizer
        """
        rasterizer_config = RasterizerConfig(
            tile_size=tile_size,
            max_tiles=max_tiles,
            blending_mode=blending_mode,
            depth_sorting=depth_sorting,
            **kwargs
        )
        
        return GaussianRasterizer(rasterizer_config)
    
    @staticmethod
    def create_optimizer(
        model: GaussianModel,
        learning_rates: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> GaussianOptimizer:
        """
        Create a Gaussian optimizer.
        
        Args:
            model: Gaussian model to optimize
            learning_rates: Learning rates for different parameters
            **kwargs: Additional parameters
            
        Returns:
            Gaussian optimizer
        """
        if learning_rates is None:
            learning_rates = {
                'position': model.config.position_lr,
                'feature': model.config.feature_lr,
                'opacity': model.config.opacity_lr,
                'scaling': model.config.scaling_lr,
                'rotation': model.config.rotation_lr,
            }
        
        optimization_config = OptimizationConfig(
            learning_rates=learning_rates,
            **kwargs
        )
        
        return GaussianOptimizer(model, optimization_config)
    
    @staticmethod
    def create_default_pipeline(
        device: Optional[torch.device] = None,
        gaussian_type: Union[str, GaussianType] = GaussianType.STANDARD,
        **kwargs
    ) -> Tuple[GaussianModel, GaussianRasterizer, GaussianOptimizer]:
        """
        Create a complete Gaussian splatting pipeline.
        
        Args:
            device: PyTorch device
            gaussian_type: Type of Gaussian model
            **kwargs: Additional configuration parameters
            
        Returns:
            Tuple of (model, rasterizer, optimizer)
        """
        if isinstance(gaussian_type, str):
            gaussian_type = GaussianType(gaussian_type.lower())
        
        # Create configuration
        config = GaussianConfig(gaussian_type=gaussian_type, **kwargs)
        
        # Create components
        model = GaussianFactory.create_model(config, device)
        rasterizer = GaussianFactory.create_rasterizer(
            tile_size=config.tile_size,
            max_tiles=config.max_tiles,
            blending_mode=config.blending_mode,
            depth_sorting=config.depth_sorting,
        )
        optimizer = GaussianFactory.create_optimizer(model)
        
        return model, rasterizer, optimizer


# ============================================================================
# TRAINING AND RENDERING UTILITIES
# ============================================================================

def train_gaussian_splatting(
    model: GaussianModel,
    train_data: TrainingData,
    val_data: Optional[TrainingData] = None,
    num_iterations: int = 30000,
    batch_size: int = 1,  # Usually 1 for Gaussian splatting
    checkpoint_dir: Optional[Union[str, Path]] = None,
    checkpoint_freq: int = 1000,
    log_freq: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Train a Gaussian splatting model.
    
    Args:
        model: Gaussian model to train
        train_data: Training data
        val_data: Validation data (optional)
        num_iterations: Number of training iterations
        batch_size: Batch size (usually 1)
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
    
    # Create rasterizer
    rasterizer = GaussianFactory.create_rasterizer(
        tile_size=model.config.tile_size,
        max_tiles=model.config.max_tiles,
        blending_mode=model.config.blending_mode,
        depth_sorting=model.config.depth_sorting,
    )
    
    # Create optimizer
    optimizer = GaussianFactory.create_optimizer(model)
    
    # Training loop
    start_time = time.time()
    train_losses = []
    val_losses = []
    psnrs = []
    
    progress_bar = tqdm(range(num_iterations), desc="Training Gaussian Splatting")
    
    for iteration in progress_bar:
        # Sample random image
        image_idx = torch.randint(0, train_data.num_images, (1,)).item()
        
        # Get image and camera
        image = train_data.images[image_idx:image_idx+1]  # [1, 3, H, W]
        pose = train_data.camera_poses[image_idx:image_idx+1]
        K = train_data.intrinsics[image_idx:image_idx+1]
        
        # Forward pass (render)
        optimizer.zero_grad()
        
        # Get Gaussian parameters
        gaussian_params = model.get_parameters()
        
        # Render image
        rendering_output = rasterizer.render(
            gaussian_params,
            pose,
            K,
            image.shape[2:]  # (H, W)
        )
        
        # Compute loss
        loss = F.mse_loss(rendering_output.image, image)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step(iteration)
        
        # Densification and pruning
        if iteration % model.config.densification_interval == 0:
            optimizer.densify_and_prune(iteration)
        
        # Update statistics
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        # Compute PSNR
        with torch.no_grad():
            mse = F.mse_loss(rendering_output.image, image)
            psnr = -10.0 * torch.log10(mse)
            psnrs.append(psnr.item())
        
        # Validation
        if val_data is not None and iteration % log_freq == 0:
            model.eval()
            with torch.no_grad():
                val_idx = torch.randint(0, val_data.num_images, (1,)).item()
                val_image = val_data.images[val_idx:val_idx+1]
                val_pose = val_data.camera_poses[val_idx:val_idx+1]
                val_K = val_data.intrinsics[val_idx:val_idx+1]
                
                # Render validation image
                val_params = model.get_parameters()
                val_output = rasterizer.render(
                    val_params,
                    val_pose,
                    val_K,
                    val_image.shape[2:]
                )
                val_loss = F.mse_loss(val_output.image, val_image).item()
                val_losses.append(val_loss)
            
            model.train()
        
        # Logging
        if iteration % log_freq == 0:
            num_gaussians = model.get_num_gaussians()
            progress_bar.set_postfix({
                'loss': f'{train_loss:.6f}',
                'psnr': f'{psnrs[-1]:.2f}',
                'gaussians': f'{num_gaussians:,}',
            })
        
        # Checkpoint
        if checkpoint_dir is not None and iteration % checkpoint_freq == 0:
            checkpoint_path = Path(checkpoint_dir) / f'checkpoint_{iteration:06d}.pt'
            model.save_checkpoint(checkpoint_path)
    
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
        'num_gaussians': model.get_num_gaussians(),
    }
    
    return results


def render_gaussian_views(
    model: GaussianModel,
    camera_poses: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Tuple[int, int],
    **kwargs
) -> torch.Tensor:
    """
    Render views from a trained Gaussian model.
    
    Args:
        model: Trained Gaussian model
        camera_poses: Camera poses [N, 4, 4] or [N, 3, 4]
        intrinsics: Camera intrinsics [N, 3, 3] or [N, 4]
        image_size: Output image size (height, width)
        **kwargs: Additional rendering parameters
        
    Returns:
        Rendered images [N, 3, H, W]
    """
    model.eval()
    device = model.device
    
    num_views = camera_poses.shape[0]
    height, width = image_size
    
    # Create rasterizer
    rasterizer = GaussianFactory.create_rasterizer(
        tile_size=model.config.tile_size,
        max_tiles=model.config.max_tiles,
        blending_mode=model.config.blending_mode,
        depth_sorting=model.config.depth_sorting,
    )
    
    # Get Gaussian parameters
    gaussian_params = model.get_parameters()
    
    # Render all views
    images = torch.zeros((num_views, 3, height, width), device=device)
    
    with torch.no_grad():
        for view_idx in range(num_views):
            pose = camera_poses[view_idx:view_idx+1]
            K = intrinsics[view_idx:view_idx+1] if intrinsics.dim() > 1 else intrinsics
            
            # Render
            rendering_output = rasterizer.render(
                gaussian_params,
                pose,
                K,
                (height, width),
                **kwargs
            )
            
            images[view_idx] = rendering_output.image.squeeze(0)
    
    return images


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_gaussian_model(
    gaussian_type: Union[str, GaussianType] = GaussianType.STANDARD,
    config: Optional[Union[Dict[str, Any], GaussianConfig]] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> GaussianModel:
    """
    Create a Gaussian model (convenience function).
    
    Args:
        gaussian_type: Type of Gaussian model
        config: Configuration dictionary or object
        device: PyTorch device
        **kwargs: Additional configuration parameters
        
    Returns:
        Gaussian model
    """
    if isinstance(gaussian_type, str):
        gaussian_type = GaussianType(gaussian_type.lower())
    
    if isinstance(config, dict):
        config_dict = {**config, **kwargs}
        config_dict['gaussian_type'] = gaussian_type
        config = GaussianConfig(**config_dict)
    elif config is None:
        config = GaussianConfig(gaussian_type=gaussian_type, **kwargs)
    
    return GaussianFactory.create_model(config, device)


def load_gaussian_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    **kwargs
) -> Tuple[GaussianModel, Optional[Dict[str, Any]]]:
    """
    Load Gaussian model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
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
    config = GaussianConfig.from_dict(config_dict)
    
    # Create model
    model = create_gaussian_model(config=config, device=device)
    
    # Load model state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Prepare return data
    checkpoint_data = {
        'iteration': checkpoint.get('iteration', 0),
        'loss_history': checkpoint.get('loss_history', []),
        'config': config,
    }
    
    return model, checkpoint_data


def extract_point_cloud_from_gaussians(
    model: GaussianModel,
    threshold: float = 0.01,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract point cloud from Gaussian model.
    
    Args:
        model: Gaussian model
        threshold: Opacity threshold for filtering
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (points [N, 3], colors [N, 3])
    """
    model.eval()
    
    # Get Gaussian parameters
    params = model.get_parameters()
    
    # Filter by opacity
    if threshold > 0:
        mask = params.opacities.squeeze() > threshold
        params = params.filter_by_mask(mask)
    
    # Points are Gaussian positions
    points = params.positions
    
    # Colors from spherical harmonics (simplified - use dominant color)
    # For simplicity, use the DC (degree 0) component
    if params.features.shape[1] > 3:  # Has SH coefficients
        # First 3 channels are DC component
        colors = params.features[:, :3]
        colors = torch.sigmoid(colors)  # SH coefficients are in logit space
    else:
        colors = params.features
    
    return points, colors


def compute_gaussian_metrics(
    model: GaussianModel,
    test_data: TrainingData,
    **kwargs
) -> RepresentationMetrics:
    """
    Compute metrics for Gaussian model.
    
    Args:
        model: Trained Gaussian model
        test_data: Test data
        **kwargs: Additional parameters
        
    Returns:
        RepresentationMetrics object
    """
    import time
    from ..base import RepresentationMetrics
    
    model.eval()
    device = model.device
    
    # Create rasterizer
    rasterizer = GaussianFactory.create_rasterizer(
        tile_size=model.config.tile_size,
        max_tiles=model.config.max_tiles,
        blending_mode=model.config.blending_mode,
        depth_sorting=model.config.depth_sorting,
    )
    
    # Get Gaussian parameters
    gaussian_params = model.get_parameters()
    
    # Initialize metrics
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        # Process all test images
        for image_idx in range(test_data.num_images):
            # Get image and camera
            image = test_data.images[image_idx:image_idx+1]
            pose = test_data.camera_poses[image_idx:image_idx+1]
            K = test_data.intrinsics[image_idx:image_idx+1]
            
            # Render
            rendering_output = rasterizer.render(
                gaussian_params,
                pose,
                K,
                image.shape[2:]
            )
            
            # Compute MSE and PSNR
            mse = F.mse_loss(rendering_output.image, image)
            psnr = -10.0 * torch.log10(mse)
            
            total_mse += mse.item()
            total_psnr += psnr.item()
            num_images += 1
    
    inference_time = time.time() - start_time
    
    # Compute averages
    avg_mse = total_mse / num_images if num_images > 0 else 0.0
    avg_psnr = total_psnr / num_images if num_images > 0 else 0.0
    
    # Create metrics
    metrics = RepresentationMetrics(
        mse=avg_mse,
        psnr_db=avg_psnr,
        inference_time_ms=inference_time * 1000 / num_images if num_images > 0 else 0.0,
        render_fps=num_images / inference_time if inference_time > 0 else 0.0,
    )
    
    return metrics


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Register Gaussian type with the global registry
try:
    from ..base import RepresentationRegistry
    RepresentationRegistry.register('gaussian', GaussianModel, 
                                  aliases=['gaussian_splatting', '3dgs', 'splatting'])
except ImportError:
    pass