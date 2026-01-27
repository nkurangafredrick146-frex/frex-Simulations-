"""
Volume Rendering for NeRF.

This module implements the volume rendering equations for Neural Radiance Fields,
including different rendering modes, compositing, and output processing.
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


class RenderingMode(Enum):
    """Enumeration of rendering modes."""
    STANDARD = "standard"        # Standard volume rendering
    ALPHA_COMPOSITING = "alpha"  # Alpha compositing
    MAX_COMPOSITING = "max"      # Max compositing (e.g., for depth)
    MEAN_COMPOSITING = "mean"    # Mean compositing
    MEDIAN_COMPOSITING = "median"  # Median compositing


class OutputType(Enum):
    """Enumeration of output types."""
    RGB = "rgb"                   # Color
    DEPTH = "depth"               # Depth
    NORMAL = "normal"             # Normal
    ALPHA = "alpha"               # Alpha (opacity)
    FEATURES = "features"         # Feature vectors
    SEMANTIC = "semantic"         # Semantic labels
    UNCERTAINTY = "uncertainty"   # Uncertainty


@dataclass
class VolumeRenderingConfig:
    """Configuration for volume rendering."""
    
    # Basic rendering
    white_bkgd: bool = True
    raw_noise_std: float = 0.0
    min_alpha: float = 1e-5
    max_alpha: float = 1.0 - 1e-5
    
    # Rendering mode
    rendering_mode: Union[str, RenderingMode] = RenderingMode.STANDARD
    
    # Output settings
    output_types: List[Union[str, OutputType]] = field(default_factory=lambda: [OutputType.RGB, OutputType.DEPTH])
    compute_normals: bool = False
    compute_uncertainty: bool = False
    
    # Depth rendering
    depth_mode: str = "expected"  # "expected", "median", "max"
    depth_scale: float = 1.0
    depth_offset: float = 0.0
    
    # Normal rendering
    normal_mode: str = "weighted"  # "weighted", "surface"
    normal_smoothing: float = 0.01
    
    # Alpha (opacity) threshold
    alpha_threshold: float = 0.01
    
    # Feature rendering
    feature_reduction: str = "weighted"  # "weighted", "max", "mean"
    
    # Anti-aliasing
    anti_aliasing: bool = False
    super_sampling: int = 1
    
    # Tone mapping
    tonemapping: bool = False
    exposure: float = 1.0
    gamma: float = 2.2
    
    # Post-processing
    denoising: bool = False
    denoising_strength: float = 0.1
    
    def __post_init__(self):
        """Validate configuration."""
        # Convert string enums
        if isinstance(self.rendering_mode, str):
            self.rendering_mode = RenderingMode(self.rendering_mode.lower())
        
        # Convert output types
        converted_outputs = []
        for output_type in self.output_types:
            if isinstance(output_type, str):
                converted_outputs.append(OutputType(output_type.lower()))
            else:
                converted_outputs.append(output_type)
        self.output_types = converted_outputs
        
        # Validate
        if self.raw_noise_std < 0:
            raise ValueError(f"raw_noise_std must be non-negative, got {self.raw_noise_std}")
        
        if self.min_alpha < 0 or self.min_alpha > 1:
            raise ValueError(f"min_alpha must be in [0, 1], got {self.min_alpha}")
        
        if self.max_alpha < 0 or self.max_alpha > 1:
            raise ValueError(f"max_alpha must be in [0, 1], got {self.max_alpha}")
        
        if self.min_alpha >= self.max_alpha:
            raise ValueError(f"min_alpha ({self.min_alpha}) must be less than max_alpha ({self.max_alpha})")
        
        if self.super_sampling < 1:
            raise ValueError(f"super_sampling must be >= 1, got {self.super_sampling}")
    
    def has_output_type(self, output_type: Union[str, OutputType]) -> bool:
        """Check if configuration includes specific output type."""
        if isinstance(output_type, str):
            output_type = OutputType(output_type.lower())
        return output_type in self.output_types


@dataclass
class RenderingOutput:
    """Output of volume rendering."""
    
    # Core outputs
    colors: torch.Tensor  # [N, 3] or [N, H, W, 3] or [B, H, W, 3]
    
    # Optional outputs
    depths: Optional[torch.Tensor] = None  # [N, 1] or [N, H, W, 1]
    alphas: Optional[torch.Tensor] = None  # [N, 1] or [N, H, W, 1]
    normals: Optional[torch.Tensor] = None  # [N, 3] or [N, H, W, 3]
    features: Optional[torch.Tensor] = None  # [N, F] or [N, H, W, F]
    semantic: Optional[torch.Tensor] = None  # [N, C] or [N, H, W, C]
    uncertainty: Optional[torch.Tensor] = None  # [N, 1] or [N, H, W, 1]
    
    # Intermediate values
    weights: Optional[torch.Tensor] = None  # [N, S] sample weights
    transmittance: Optional[torch.Tensor] = None  # [N, S] transmittance
    alpha: Optional[torch.Tensor] = None  # [N, S] per-sample alpha
    
    # Metadata
    ray_depths: Optional[torch.Tensor] = None  # [N, S] sample depths
    ray_deltas: Optional[torch.Tensor] = None  # [N, S] sample deltas
    
    def __post_init__(self):
        """Validate shapes."""
        # Check that all outputs have compatible shapes
        outputs = [
            self.depths, self.alphas, self.normals, 
            self.features, self.semantic, self.uncertainty
        ]
        
        for output in outputs:
            if output is not None:
                # Check batch dimensions match
                if output.shape[0] != self.colors.shape[0]:
                    raise ValueError(
                        f"Output batch size {output.shape[0]} != colors batch size {self.colors.shape[0]}"
                    )
                
                # For spatial outputs, check spatial dimensions match
                if output.dim() >= 3 and self.colors.dim() >= 3:
                    if output.shape[1:-1] != self.colors.shape[1:-1]:
                        raise ValueError(
                            f"Output spatial shape {output.shape[1:-1]} != colors spatial shape {self.colors.shape[1:-1]}"
                        )
    
    @property
    def batch_shape(self) -> torch.Size:
        """Get batch shape (excluding channel dimension)."""
        return self.colors.shape[:-1]
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.colors.shape[0]
    
    @property
    def device(self) -> torch.device:
        """Get device of outputs."""
        return self.colors.device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get data type of outputs."""
        return self.colors.dtype
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert output to dictionary."""
        result = {'colors': self.colors}
        
        if self.depths is not None:
            result['depths'] = self.depths
        
        if self.alphas is not None:
            result['alphas'] = self.alphas
        
        if self.normals is not None:
            result['normals'] = self.normals
        
        if self.features is not None:
            result['features'] = self.features
        
        if self.semantic is not None:
            result['semantic'] = self.semantic
        
        if self.uncertainty is not None:
            result['uncertainty'] = self.uncertainty
        
        if self.weights is not None:
            result['weights'] = self.weights
        
        if self.transmittance is not None:
            result['transmittance'] = self.transmittance
        
        if self.alpha is not None:
            result['alpha'] = self.alpha
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, torch.Tensor]) -> RenderingOutput:
        """Create output from dictionary."""
        return cls(
            colors=data['colors'],
            depths=data.get('depths'),
            alphas=data.get('alphas'),
            normals=data.get('normals'),
            features=data.get('features'),
            semantic=data.get('semantic'),
            uncertainty=data.get('uncertainty'),
            weights=data.get('weights'),
            transmittance=data.get('transmittance'),
            alpha=data.get('alpha'),
            ray_depths=data.get('ray_depths'),
            ray_deltas=data.get('ray_deltas'),
        )
    
    def detach(self) -> RenderingOutput:
        """Detach all tensors from computation graph."""
        return RenderingOutput(
            colors=self.colors.detach(),
            depths=self.depths.detach() if self.depths is not None else None,
            alphas=self.alphas.detach() if self.alphas is not None else None,
            normals=self.normals.detach() if self.normals is not None else None,
            features=self.features.detach() if self.features is not None else None,
            semantic=self.semantic.detach() if self.semantic is not None else None,
            uncertainty=self.uncertainty.detach() if self.uncertainty is not None else None,
            weights=self.weights.detach() if self.weights is not None else None,
            transmittance=self.transmittance.detach() if self.transmittance is not None else None,
            alpha=self.alpha.detach() if self.alpha is not None else None,
            ray_depths=self.ray_depths.detach() if self.ray_depths is not None else None,
            ray_deltas=self.ray_deltas.detach() if self.ray_deltas is not None else None,
        )
    
    def to(self, device: torch.device) -> RenderingOutput:
        """Move outputs to device."""
        return RenderingOutput(
            colors=self.colors.to(device),
            depths=self.depths.to(device) if self.depths is not None else None,
            alphas=self.alphas.to(device) if self.alphas is not None else None,
            normals=self.normals.to(device) if self.normals is not None else None,
            features=self.features.to(device) if self.features is not None else None,
            semantic=self.semantic.to(device) if self.semantic is not None else None,
            uncertainty=self.uncertainty.to(device) if self.uncertainty is not None else None,
            weights=self.weights.to(device) if self.weights is not None else None,
            transmittance=self.transmittance.to(device) if self.transmittance is not None else None,
            alpha=self.alpha.to(device) if self.alpha is not None else None,
            ray_depths=self.ray_depths.to(device) if self.ray_depths is not None else None,
            ray_deltas=self.ray_deltas.to(device) if self.ray_deltas is not None else None,
        )
    
    def cpu(self) -> RenderingOutput:
        """Move outputs to CPU."""
        return self.to(torch.device('cpu'))
    
    def apply_tonemapping(
        self, 
        exposure: float = 1.0, 
        gamma: float = 2.2
    ) -> RenderingOutput:
        """Apply tone mapping to colors."""
        # Reinhard tonemapping
        colors_tonemapped = self.colors * exposure
        colors_tonemapped = colors_tonemapped / (1.0 + colors_tonemapped)
        colors_tonemapped = torch.pow(colors_tonemapped, 1.0 / gamma)
        
        return RenderingOutput(
            colors=colors_tonemapped,
            depths=self.depths,
            alphas=self.alphas,
            normals=self.normals,
            features=self.features,
            semantic=self.semantic,
            uncertainty=self.uncertainty,
            weights=self.weights,
            transmittance=self.transmittance,
            alpha=self.alpha,
            ray_depths=self.ray_depths,
            ray_deltas=self.ray_deltas,
        )
    
    def apply_white_background(self) -> RenderingOutput:
        """Composite colors onto white background using alpha."""
        if self.alphas is None:
            # Compute alpha from colors if not provided
            alphas = self.colors.norm(dim=-1, keepdim=True).clamp(0, 1)
        else:
            alphas = self.alphas
        
        # Composite onto white background
        colors_bg = self.colors * alphas + (1.0 - alphas) * 1.0
        
        return RenderingOutput(
            colors=colors_bg,
            depths=self.depths,
            alphas=alphas,
            normals=self.normals,
            features=self.features,
            semantic=self.semantic,
            uncertainty=self.uncertainty,
            weights=self.weights,
            transmittance=self.transmittance,
            alpha=self.alpha,
            ray_depths=self.ray_depths,
            ray_deltas=self.ray_deltas,
        )


class VolumeRenderer(nn.Module):
    """
    Volume renderer for NeRF.
    
    This class implements the volume rendering equations for compositing
    samples along rays into final pixel colors and other outputs.
    """
    
    def __init__(
        self,
        config: Optional[VolumeRenderingConfig] = None,
        white_bkgd: Optional[bool] = None,
        raw_noise_std: Optional[float] = None,
        min_alpha: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize volume renderer.
        
        Args:
            config: Volume rendering configuration
            white_bkgd: Whether to use white background (overrides config)
            raw_noise_std: Noise std for raw density (overrides config)
            min_alpha: Minimum alpha value (overrides config)
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        
        # Create or update config
        if config is None:
            config_dict = {}
            if white_bkgd is not None:
                config_dict['white_bkgd'] = white_bkgd
            if raw_noise_std is not None:
                config_dict['raw_noise_std'] = raw_noise_std
            if min_alpha is not None:
                config_dict['min_alpha'] = min_alpha
            config_dict.update(kwargs)
            config = VolumeRenderingConfig(**config_dict)
        else:
            # Update config with provided parameters
            config_dict = config.__dict__.copy()
            if white_bkgd is not None:
                config_dict['white_bkgd'] = white_bkgd
            if raw_noise_std is not None:
                config_dict['raw_noise_std'] = raw_noise_std
            if min_alpha is not None:
                config_dict['min_alpha'] = min_alpha
            config_dict.update(kwargs)
            config = VolumeRenderingConfig(**config_dict)
        
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        config = self.config
        
        if config.raw_noise_std < 0:
            raise ValueError(f"raw_noise_std must be non-negative, got {config.raw_noise_std}")
        
        if config.min_alpha < 0 or config.min_alpha > 1:
            raise ValueError(f"min_alpha must be in [0, 1], got {config.min_alpha}")
    
    def forward(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        semantic: Optional[torch.Tensor] = None,
        mode: Optional[Union[str, RenderingMode]] = None,
        **kwargs
    ) -> RenderingOutput:
        """
        Perform volume rendering.
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            colors: Color values [N, S, 3]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            normals: Normal vectors [N, S, 3] (optional)
            features: Feature vectors [N, S, F] (optional)
            semantic: Semantic logits [N, S, C] (optional)
            mode: Rendering mode (overrides config)
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendering output
        """
        # Determine rendering mode
        if mode is None:
            mode = self.config.rendering_mode
        elif isinstance(mode, str):
            mode = RenderingMode(mode.lower())
        
        # Validate inputs
        self._validate_inputs(densities, colors, depths, deltas, normals, features, semantic)
        
        # Flatten if needed
        original_shape = densities.shape
        if densities.dim() == 4:
            # [B, H, W, S, ...] -> [B*H*W, S, ...]
            batch_dims = densities.shape[:-2]
            densities = densities.reshape(-1, *densities.shape[-2:])
            colors = colors.reshape(-1, *colors.shape[-2:])
            depths = depths.reshape(-1, depths.shape[-1])
            deltas = deltas.reshape(-1, deltas.shape[-1])
            
            if normals is not None:
                normals = normals.reshape(-1, *normals.shape[-2:])
            
            if features is not None:
                features = features.reshape(-1, *features.shape[-2:])
            
            if semantic is not None:
                semantic = semantic.reshape(-1, *semantic.shape[-2:])
        
        # Apply noise to densities if training
        if self.training and self.config.raw_noise_std > 0:
            noise = torch.randn_like(densities) * self.config.raw_noise_std
            densities = densities + noise
        
        # Clamp densities to avoid numerical issues
        densities = torch.clamp(densities, min=0.0)
        
        # Perform rendering based on mode
        if mode == RenderingMode.STANDARD:
            result = self._standard_rendering(
                densities, colors, depths, deltas, normals, features, semantic, **kwargs
            )
        elif mode == RenderingMode.ALPHA_COMPOSITING:
            result = self._alpha_compositing(
                densities, colors, depths, deltas, normals, features, semantic, **kwargs
            )
        elif mode == RenderingMode.MAX_COMPOSITING:
            result = self._max_compositing(
                densities, colors, depths, deltas, normals, features, semantic, **kwargs
            )
        elif mode == RenderingMode.MEAN_COMPOSITING:
            result = self._mean_compositing(
                densities, colors, depths, deltas, normals, features, semantic, **kwargs
            )
        elif mode == RenderingMode.MEDIAN_COMPOSITING:
            result = self._median_compositing(
                densities, colors, depths, deltas, normals, features, semantic, **kwargs
            )
        else:
            raise ValueError(f"Unknown rendering mode: {mode}")
        
        # Reshape back if needed
        if len(original_shape) == 4:
            # [B*H*W, ...] -> [B, H, W, ...]
            batch_dims = original_shape[:-2]
            result.colors = result.colors.reshape(*batch_dims, -1)
            
            if result.depths is not None:
                result.depths = result.depths.reshape(*batch_dims, -1)
            
            if result.alphas is not None:
                result.alphas = result.alphas.reshape(*batch_dims, -1)
            
            if result.normals is not None:
                result.normals = result.normals.reshape(*batch_dims, -1)
            
            if result.features is not None:
                result.features = result.features.reshape(*batch_dims, -1)
            
            if result.semantic is not None:
                result.semantic = result.semantic.reshape(*batch_dims, -1)
            
            if result.uncertainty is not None:
                result.uncertainty = result.uncertainty.reshape(*batch_dims, -1)
            
            if result.weights is not None:
                result.weights = result.weights.reshape(*batch_dims, -1)
            
            if result.transmittance is not None:
                result.transmittance = result.transmittance.reshape(*batch_dims, -1)
            
            if result.alpha is not None:
                result.alpha = result.alpha.reshape(*batch_dims, -1)
        
        # Apply post-processing
        result = self._apply_post_processing(result, **kwargs)
        
        return result
    
    def _validate_inputs(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        normals: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
        semantic: Optional[torch.Tensor],
    ) -> None:
        """Validate input tensors."""
        N, S = densities.shape[:2]
        
        # Check shapes
        assert colors.shape[:2] == (N, S), \
            f"colors shape {colors.shape[:2]} != ({N}, {S})"
        
        assert depths.shape == (N, S), \
            f"depths shape {depths.shape} != ({N}, {S})"
        
        assert deltas.shape == (N, S), \
            f"deltas shape {deltas.shape} != ({N}, {S})"
        
        if normals is not None:
            assert normals.shape[:2] == (N, S), \
                f"normals shape {normals.shape[:2]} != ({N}, {S})"
        
        if features is not None:
            assert features.shape[:2] == (N, S), \
                f"features shape {features.shape[:2]} != ({N}, {S})"
        
        if semantic is not None:
            assert semantic.shape[:2] == (N, S), \
                f"semantic shape {semantic.shape[:2]} != ({N}, {S})"
    
    def _standard_rendering(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        normals: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
        semantic: Optional[torch.Tensor],
        **kwargs
    ) -> RenderingOutput:
        """
        Standard volume rendering (NeRF's approach).
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            colors: Color values [N, S, 3]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            normals: Normal vectors [N, S, 3]
            features: Feature vectors [N, S, F]
            semantic: Semantic logits [N, S, C]
            **kwargs: Additional parameters
            
        Returns:
            Rendering output
        """
        # Ensure densities have correct shape
        if densities.dim() == 2:
            densities = densities.unsqueeze(-1)
        
        # Compute alpha (opacity) from density and delta
        alpha = 1.0 - torch.exp(-densities * deltas.unsqueeze(-1))
        
        # Clamp alpha for numerical stability
        alpha = torch.clamp(alpha, self.config.min_alpha, self.config.max_alpha)
        
        # Compute transmittance
        # T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)
        transmission = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
        transmittance = torch.cat([
            torch.ones_like(transmission[:, :1]),
            transmission[:, :-1]
        ], dim=1)
        
        # Compute weights
        weights = transmittance * alpha
        
        # Compute outputs using weighted sums
        # Colors
        rendered_colors = torch.sum(weights * colors, dim=1)
        
        # Depths (expected depth)
        rendered_depths = torch.sum(weights.squeeze(-1) * depths, dim=1, keepdim=True)
        
        # Alphas (total opacity)
        rendered_alphas = torch.sum(weights, dim=1)
        
        # Normals (weighted average)
        rendered_normals = None
        if normals is not None:
            rendered_normals = torch.sum(weights * normals, dim=1)
            rendered_normals = F.normalize(rendered_normals, dim=-1)
        
        # Features (weighted average)
        rendered_features = None
        if features is not None:
            if self.config.feature_reduction == "weighted":
                rendered_features = torch.sum(weights * features, dim=1)
            elif self.config.feature_reduction == "max":
                # Max pooling along ray
                max_weights, max_indices = torch.max(weights, dim=1, keepdim=True)
                rendered_features = torch.gather(features, 1, max_indices.expand(-1, -1, features.shape[-1])).squeeze(1)
            elif self.config.feature_reduction == "mean":
                rendered_features = torch.mean(features, dim=1)
            else:
                raise ValueError(f"Unknown feature reduction: {self.config.feature_reduction}")
        
        # Semantic (weighted average of logits, then softmax)
        rendered_semantic = None
        if semantic is not None:
            semantic_weights = torch.sum(weights * semantic, dim=1)
            rendered_semantic = F.softmax(semantic_weights, dim=-1)
        
        # Uncertainty (entropy of weights)
        rendered_uncertainty = None
        if self.config.compute_uncertainty:
            # Compute entropy of weight distribution
            weight_entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1)
            rendered_uncertainty = weight_entropy
        
        return RenderingOutput(
            colors=rendered_colors,
            depths=rendered_depths,
            alphas=rendered_alphas,
            normals=rendered_normals,
            features=rendered_features,
            semantic=rendered_semantic,
            uncertainty=rendered_uncertainty,
            weights=weights,
            transmittance=transmittance,
            alpha=alpha,
            ray_depths=depths,
            ray_deltas=deltas,
        )
    
    def _alpha_compositing(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        normals: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
        semantic: Optional[torch.Tensor],
        **kwargs
    ) -> RenderingOutput:
        """
        Alpha compositing (over operator).
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            colors: Color values [N, S, 3]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            normals: Normal vectors [N, S, 3]
            features: Feature vectors [N, S, F]
            semantic: Semantic logits [N, S, C]
            **kwargs: Additional parameters
            
        Returns:
            Rendering output
        """
        # Similar to standard rendering but with different weight computation
        result = self._standard_rendering(
            densities, colors, depths, deltas, normals, features, semantic, **kwargs
        )
        
        return result
    
    def _max_compositing(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        normals: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
        semantic: Optional[torch.Tensor],
        **kwargs
    ) -> RenderingOutput:
        """
        Max compositing (take sample with maximum density/weight).
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            colors: Color values [N, S, 3]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            normals: Normal vectors [N, S, 3]
            features: Feature vectors [N, S, F]
            semantic: Semantic logits [N, S, C]
            **kwargs: Additional parameters
            
        Returns:
            Rendering output
        """
        # Ensure densities have correct shape
        if densities.dim() == 2:
            densities = densities.unsqueeze(-1)
        
        # Compute weights based on density
        weights = densities.squeeze(-1)
        
        # Find index with maximum weight for each ray
        max_indices = torch.argmax(weights, dim=1, keepdim=True)
        
        # Gather outputs at max indices
        batch_indices = torch.arange(weights.shape[0], device=weights.device).unsqueeze(1)
        
        rendered_colors = colors[batch_indices, max_indices].squeeze(1)
        rendered_depths = depths[batch_indices, max_indices].unsqueeze(-1)
        
        # Alpha is 1.0 at max position
        rendered_alphas = torch.ones_like(rendered_depths)
        
        # Other outputs
        rendered_normals = None
        if normals is not None:
            rendered_normals = normals[batch_indices, max_indices].squeeze(1)
        
        rendered_features = None
        if features is not None:
            rendered_features = features[batch_indices, max_indices].squeeze(1)
        
        rendered_semantic = None
        if semantic is not None:
            semantic_max = semantic[batch_indices, max_indices].squeeze(1)
            rendered_semantic = F.softmax(semantic_max, dim=-1)
        
        # Create weight tensor (one-hot at max position)
        weight_mask = torch.zeros_like(weights)
        weight_mask.scatter_(1, max_indices, 1.0)
        weights = weight_mask.unsqueeze(-1)
        
        return RenderingOutput(
            colors=rendered_colors,
            depths=rendered_depths,
            alphas=rendered_alphas,
            normals=rendered_normals,
            features=rendered_features,
            semantic=rendered_semantic,
            uncertainty=None,
            weights=weights,
            transmittance=None,
            alpha=None,
            ray_depths=depths,
            ray_deltas=deltas,
        )
    
    def _mean_compositing(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        normals: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
        semantic: Optional[torch.Tensor],
        **kwargs
    ) -> RenderingOutput:
        """
        Mean compositing (average of all samples).
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            colors: Color values [N, S, 3]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            normals: Normal vectors [N, S, 3]
            features: Feature vectors [N, S, F]
            semantic: Semantic logits [N, S, C]
            **kwargs: Additional parameters
            
        Returns:
            Rendering output
        """
        # Simple mean across samples
        rendered_colors = torch.mean(colors, dim=1)
        rendered_depths = torch.mean(depths, dim=1, keepdim=True)
        rendered_alphas = torch.ones_like(rendered_depths)
        
        rendered_normals = None
        if normals is not None:
            rendered_normals = torch.mean(normals, dim=1)
            rendered_normals = F.normalize(rendered_normals, dim=-1)
        
        rendered_features = None
        if features is not None:
            rendered_features = torch.mean(features, dim=1)
        
        rendered_semantic = None
        if semantic is not None:
            semantic_mean = torch.mean(semantic, dim=1)
            rendered_semantic = F.softmax(semantic_mean, dim=-1)
        
        # Create uniform weights
        S = colors.shape[1]
        weights = torch.ones_like(densities) / S
        
        return RenderingOutput(
            colors=rendered_colors,
            depths=rendered_depths,
            alphas=rendered_alphas,
            normals=rendered_normals,
            features=rendered_features,
            semantic=rendered_semantic,
            uncertainty=None,
            weights=weights,
            transmittance=None,
            alpha=None,
            ray_depths=depths,
            ray_deltas=deltas,
        )
    
    def _median_compositing(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        normals: Optional[torch.Tensor],
        features: Optional[torch.Tensor],
        semantic: Optional[torch.Tensor],
        **kwargs
    ) -> RenderingOutput:
        """
        Median compositing (median of samples).
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            colors: Color values [N, S, 3]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            normals: Normal vectors [N, S, 3]
            features: Feature vectors [N, S, F]
            semantic: Semantic logits [N, S, C]
            **kwargs: Additional parameters
            
        Returns:
            Rendering output
        """
        # Find median index based on depths
        sorted_indices = torch.argsort(depths, dim=1)
        median_idx = depths.shape[1] // 2
        
        # Gather median indices
        batch_indices = torch.arange(depths.shape[0], device=depths.device).unsqueeze(1)
        median_indices = sorted_indices[:, median_idx:median_idx+1]
        
        rendered_colors = colors[batch_indices, median_indices].squeeze(1)
        rendered_depths = depths[batch_indices, median_indices].unsqueeze(-1)
        rendered_alphas = torch.ones_like(rendered_depths)
        
        rendered_normals = None
        if normals is not None:
            rendered_normals = normals[batch_indices, median_indices].squeeze(1)
        
        rendered_features = None
        if features is not None:
            rendered_features = features[batch_indices, median_indices].squeeze(1)
        
        rendered_semantic = None
        if semantic is not None:
            semantic_median = semantic[batch_indices, median_indices].squeeze(1)
            rendered_semantic = F.softmax(semantic_median, dim=-1)
        
        # Create weight tensor (one-hot at median position)
        weights = torch.zeros_like(densities)
        weights[batch_indices, median_indices] = 1.0
        
        return RenderingOutput(
            colors=rendered_colors,
            depths=rendered_depths,
            alphas=rendered_alphas,
            normals=rendered_normals,
            features=rendered_features,
            semantic=rendered_semantic,
            uncertainty=None,
            weights=weights,
            transmittance=None,
            alpha=None,
            ray_depths=depths,
            ray_deltas=deltas,
        )
    
    def _apply_post_processing(
        self,
        result: RenderingOutput,
        **kwargs
    ) -> RenderingOutput:
        """
        Apply post-processing to rendering results.
        
        Args:
            result: Raw rendering output
            **kwargs: Additional parameters
            
        Returns:
            Processed rendering output
        """
        # Apply white background if requested
        if self.config.white_bkgd and result.alphas is not None:
            result = result.apply_white_background()
        
        # Apply tone mapping if requested
        if self.config.tonemapping:
            result = result.apply_tonemapping(
                exposure=self.config.exposure,
                gamma=self.config.gamma
            )
        
        # Apply depth scaling
        if result.depths is not None:
            result.depths = result.depths * self.config.depth_scale + self.config.depth_offset
        
        # Apply alpha threshold
        if result.alphas is not None and self.config.alpha_threshold > 0:
            # Set colors to background where alpha is below threshold
            mask = result.alphas < self.config.alpha_threshold
            result.colors = torch.where(
                mask.expand_as(result.colors),
                torch.ones_like(result.colors),  # White background
                result.colors
            )
        
        # Apply normal smoothing
        if result.normals is not None and self.config.normal_smoothing > 0:
            # This would require spatial neighborhood information
            # For now, just normalize
            result.normals = F.normalize(result.normals, dim=-1)
        
        return result
    
    def render(
        self,
        densities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        mode: Union[str, RenderingMode] = RenderingMode.STANDARD,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Render with specific mode (convenience method).
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            colors: Color values [N, S, 3]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            mode: Rendering mode
            **kwargs: Additional parameters
            
        Returns:
            Rendered output(s)
        """
        result = self.forward(
            densities, colors, depths, deltas,
            mode=mode, **kwargs
        )
        
        # Return based on requested outputs
        output_types = kwargs.get('output_types', self.config.output_types)
        
        if len(output_types) == 1:
            output_type = output_types[0]
            if output_type == OutputType.RGB:
                return result.colors
            elif output_type == OutputType.DEPTH:
                return result.depths
            elif output_type == OutputType.NORMAL:
                return result.normals
            elif output_type == OutputType.ALPHA:
                return result.alphas
            elif output_type == OutputType.FEATURES:
                return result.features
            elif output_type == OutputType.SEMANTIC:
                return result.semantic
            elif output_type == OutputType.UNCERTAINTY:
                return result.uncertainty
            else:
                raise ValueError(f"Unknown output type: {output_type}")
        else:
            # Return dictionary
            outputs = {}
            for output_type in output_types:
                if output_type == OutputType.RGB and result.colors is not None:
                    outputs['colors'] = result.colors
                elif output_type == OutputType.DEPTH and result.depths is not None:
                    outputs['depths'] = result.depths
                elif output_type == OutputType.NORMAL and result.normals is not None:
                    outputs['normals'] = result.normals
                elif output_type == OutputType.ALPHA and result.alphas is not None:
                    outputs['alphas'] = result.alphas
                elif output_type == OutputType.FEATURES and result.features is not None:
                    outputs['features'] = result.features
                elif output_type == OutputType.SEMANTIC and result.semantic is not None:
                    outputs['semantic'] = result.semantic
                elif output_type == OutputType.UNCERTAINTY and result.uncertainty is not None:
                    outputs['uncertainty'] = result.uncertainty
            
            return outputs
    
    def compute_depth(
        self,
        densities: torch.Tensor,
        depths: torch.Tensor,
        deltas: torch.Tensor,
        mode: str = "expected",
        **kwargs
    ) -> torch.Tensor:
        """
        Compute depth from densities.
        
        Args:
            densities: Density values [N, S, 1] or [N, S]
            depths: Sample depths [N, S]
            deltas: Distance between samples [N, S]
            mode: Depth computation mode
            **kwargs: Additional parameters
            
        Returns:
            Depth values [N, 1]
        """
        if mode == "expected":
            # Expected depth (weighted average)
            if densities.dim() == 2:
                densities = densities.unsqueeze(-1)
            
            alpha = 1.0 - torch.exp(-densities * deltas.unsqueeze(-1))
            alpha = torch.clamp(alpha, self.config.min_alpha, self.config.max_alpha)
            
            transmission = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
            transmittance = torch.cat([
                torch.ones_like(transmission[:, :1]),
                transmission[:, :-1]
            ], dim=1)
            
            weights = transmittance * alpha
            depth = torch.sum(weights.squeeze(-1) * depths, dim=1, keepdim=True)
            
        elif mode == "median":
            # Median depth
            # First compute weights as above
            if densities.dim() == 2:
                densities = densities.unsqueeze(-1)
            
            alpha = 1.0 - torch.exp(-densities * deltas.unsqueeze(-1))
            alpha = torch.clamp(alpha, self.config.min_alpha, self.config.max_alpha)
            
            transmission = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
            transmittance = torch.cat([
                torch.ones_like(transmission[:, :1]),
                transmission[:, :-1]
            ], dim=1)
            
            weights = transmittance * alpha
            
            # Find cumulative weight distribution
            cum_weights = torch.cumsum(weights.squeeze(-1), dim=1)
            
            # Find depth where cumulative weight reaches 0.5
            # This is an approximation of median depth
            target = 0.5 * cum_weights[:, -1:]
            indices = torch.searchsorted(cum_weights, target, right=True)
            indices = torch.clamp(indices, 0, depths.shape[1] - 1)
            
            batch_indices = torch.arange(depths.shape[0], device=depths.device).unsqueeze(1)
            depth = depths[batch_indices, indices]
            
        elif mode == "max":
            # Depth at maximum density
            if densities.dim() == 3:
                densities = densities.squeeze(-1)
            
            max_indices = torch.argmax(densities, dim=1, keepdim=True)
            batch_indices = torch.arange(depths.shape[0], device=depths.device).unsqueeze(1)
            depth = depths[batch_indices, max_indices]
            
        else:
            raise ValueError(f"Unknown depth mode: {mode}")
        
        return depth
    
    def compute_normal(
        self,
        positions: torch.Tensor,
        densities: torch.Tensor,
        mode: str = "gradient",
        epsilon: float = 1e-3,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute normal vectors from density field.
        
        Args:
            positions: Sample positions [N, S, 3]
            densities: Density values [N, S, 1] or [N, S]
            mode: Normal computation mode
            epsilon: Finite difference epsilon
            **kwargs: Additional parameters
            
        Returns:
            Normal vectors [N, S, 3]
        """
        if mode == "gradient":
            # Compute gradient of density w.r.t. position
            positions.requires_grad_(True)
            
            # Ensure densities are computed from positions
            # This assumes densities is a function of positions
            # In practice, you'd recompute densities here
            if not positions.grad_fn:
                # Create dummy computation graph
                densities_computed = densities.clone()
            else:
                densities_computed = densities
            
            # Compute gradient
            grad_outputs = torch.ones_like(densities_computed)
            gradients = torch.autograd.grad(
                outputs=densities_computed,
                inputs=positions,
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            
            # Normalize to get normals
            normals = -F.normalize(gradients, dim=-1)  # Negative gradient points inward
            
        elif mode == "finite_difference":
            # Finite difference approximation
            if densities.dim() == 3:
                densities = densities.squeeze(-1)
            
            N, S, _ = positions.shape
            
            # Compute gradient using finite differences
            normals = torch.zeros_like(positions)
            
            for i in range(3):
                # Positive offset
                pos_offset = positions.clone()
                pos_offset[:, :, i] += epsilon
                
                # Negative offset (would need to recompute density)
                # This is a placeholder - actual implementation would query density
                neg_offset = positions.clone()
                neg_offset[:, :, i] -= epsilon
                
                # For now, return zeros (this is just a placeholder)
                # In practice, you'd need to recompute densities at offset positions
                pass
            
            # Placeholder: return normalized positions as normals
            normals = F.normalize(positions, dim=-1)
            
        else:
            raise ValueError(f"Unknown normal mode: {mode}")
        
        return normals
    
    def compute_uncertainty(
        self,
        weights: torch.Tensor,
        mode: str = "entropy",
        **kwargs
    ) -> torch.Tensor:
        """
        Compute uncertainty from sample weights.
        
        Args:
            weights: Sample weights [N, S, 1] or [N, S]
            mode: Uncertainty computation mode
            **kwargs: Additional parameters
            
        Returns:
            Uncertainty values [N, 1]
        """
        if weights.dim() == 3:
            weights = weights.squeeze(-1)
        
        if mode == "entropy":
            # Entropy of weight distribution
            # Add epsilon to avoid log(0)
            weight_entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=1, keepdim=True)
            
            # Normalize by maximum possible entropy (log(S))
            S = weights.shape[1]
            max_entropy = math.log(S)
            uncertainty = weight_entropy / max_entropy
            
        elif mode == "variance":
            # Variance of weight distribution
            weight_mean = torch.mean(weights, dim=1, keepdim=True)
            weight_var = torch.mean((weights - weight_mean) ** 2, dim=1, keepdim=True)
            uncertainty = weight_var
            
        elif mode == "max_min":
            # Difference between max and min weight
            weight_max = torch.max(weights, dim=1, keepdim=True)[0]
            weight_min = torch.min(weights, dim=1, keepdim=True)[0]
            uncertainty = weight_max - weight_min
            
        else:
            raise ValueError(f"Unknown uncertainty mode: {mode}")
        
        return uncertainty
    
    def alpha_composite(
        self,
        foreground: torch.Tensor,
        background: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alpha composite foreground over background.
        
        Args:
            foreground: Foreground colors [..., C]
            background: Background colors [..., C]
            alpha: Alpha values [..., 1] or [...]
            
        Returns:
            Composited colors [..., C]
        """
        if alpha.dim() == foreground.dim() - 1:
            alpha = alpha.unsqueeze(-1)
        
        return foreground * alpha + background * (1.0 - alpha)
    
    def apply_depth_visualization(
        self,
        depth: torch.Tensor,
        near: Optional[float] = None,
        far: Optional[float] = None,
        colormap: str = "viridis",
        invert: bool = False,
    ) -> torch.Tensor:
        """
        Apply colormap to depth for visualization.
        
        Args:
            depth: Depth values [..., 1]
            near: Near plane (optional)
            far: Far plane (optional)
            colormap: Colormap name
            invert: Whether to invert the colormap
            
        Returns:
            Colored depth [..., 3]
        """
        # Normalize depth to [0, 1]
        if near is not None and far is not None:
            depth_norm = (depth - near) / (far - near)
        else:
            # Auto-normalize
            depth_min = depth.min()
            depth_max = depth.max()
            depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        depth_norm = torch.clamp(depth_norm, 0, 1)
        
        if invert:
            depth_norm = 1.0 - depth_norm
        
        # Apply colormap (simplified - would use actual colormap in practice)
        if colormap == "viridis":
            # Simplified viridis approximation
            t = depth_norm
            r = 0.2627 + t * (0.0671 + t * (-1.6267 + t * (2.2597 + t * (-1.2181 + t * 0.2733))))
            g = -0.1217 + t * (1.7930 + t * (-4.1626 + t * (4.5531 + t * (-2.2343 + t * 0.4411))))
            b = 0.1532 + t * (2.6402 + t * (-9.4100 + t * (14.0800 + t * (-9.7550 + t * 2.7320))))
        elif colormap == "jet":
            # Simplified jet approximation
            t = depth_norm * 4.0
            r = torch.clamp(t - 1.5, 0, 1) - torch.clamp(t - 3.5, 0, 1)
            g = torch.clamp(t - 0.5, 0, 1) - torch.clamp(t - 2.5, 0, 1)
            b = torch.clamp(t + 0.5, 0, 1) - torch.clamp(t - 1.5, 0, 1)
        elif colormap == "plasma":
            # Simplified plasma approximation
            t = depth_norm
            r = 0.0580 + t * (1.2467 + t * (8.5656 + t * (-26.1050 + t * (33.8668 + t * (-16.7385 + t * 3.1971)))))
            g = 0.0287 + t * (1.0725 + t * (1.4326 + t * (-5.8341 + t * (6.1497 + t * (-2.7684 + t * 0.4919)))))
            b = 0.6271 + t * (1.6058 + t * (-5.1254 + t * (7.5248 + t * (-4.8328 + t * 1.2444))))
        else:
            # Grayscale
            r = g = b = depth_norm
        
        colored_depth = torch.cat([r, g, b], dim=-1)
        return colored_depth
    
    def render_image(
        self,
        ray_bundle: Any,
        nerf_output: Any,
        mode: Union[str, RenderingMode] = RenderingMode.STANDARD,
        **kwargs
    ) -> torch.Tensor:
        """
        Render an image from ray bundle and NeRF output.
        
        Args:
            ray_bundle: Ray bundle with samples
            nerf_output: NeRF output (density, color, etc.)
            mode: Rendering mode
            **kwargs: Additional parameters
            
        Returns:
            Rendered image [H, W, 3] or [B, H, W, 3]
        """
        # Extract densities and colors
        densities = nerf_output.density
        colors = nerf_output.color
        
        # Extract depths and deltas from ray bundle
        depths = ray_bundle.samples.depths
        deltas = ray_bundle.samples.deltas
        
        # Extract other outputs if available
        normals = nerf_output.normals if hasattr(nerf_output, 'normals') else None
        features = nerf_output.features if hasattr(nerf_output, 'features') else None
        semantic = nerf_output.semantic if hasattr(nerf_output, 'semantic') else None
        
        # Render
        result = self.forward(
            densities, colors, depths, deltas,
            normals=normals,
            features=features,
            semantic=semantic,
            mode=mode,
            **kwargs
        )
        
        return result.colors


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_volume_renderer(
    config: Optional[VolumeRenderingConfig] = None,
    **kwargs
) -> VolumeRenderer:
    """
    Create a volume renderer (convenience function).
    
    Args:
        config: Volume rendering configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Volume renderer
    """
    return VolumeRenderer(config, **kwargs)


def render_volume(
    densities: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    deltas: torch.Tensor,
    white_bkgd: bool = True,
    raw_noise_std: float = 0.0,
    min_alpha: float = 1e-5,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render volume (convenience function).
    
    Args:
        densities: Density values [N, S, 1] or [N, S]
        colors: Color values [N, S, 3]
        depths: Sample depths [N, S]
        deltas: Distance between samples [N, S]
        white_bkgd: Whether to use white background
        raw_noise_std: Noise std for raw density
        min_alpha: Minimum alpha value
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (colors, depths, weights)
    """
    renderer = VolumeRenderer(
        white_bkgd=white_bkgd,
        raw_noise_std=raw_noise_std,
        min_alpha=min_alpha,
    )
    
    result = renderer.forward(densities, colors, depths, deltas, **kwargs)
    
    return result.colors, result.depths, result.weights


def compute_absorption_weights(
    densities: torch.Tensor,
    deltas: torch.Tensor,
    min_alpha: float = 1e-5,
    max_alpha: float = 1.0 - 1e-5,
) -> torch.Tensor:
    """
    Compute absorption weights for volume rendering.
    
    Args:
        densities: Density values [N, S, 1] or [N, S]
        deltas: Distance between samples [N, S]
        min_alpha: Minimum alpha value
        max_alpha: Maximum alpha value
        
    Returns:
        Weights [N, S, 1]
    """
    if densities.dim() == 2:
        densities = densities.unsqueeze(-1)
    
    # Compute alpha (opacity) from density and delta
    alpha = 1.0 - torch.exp(-densities * deltas.unsqueeze(-1))
    alpha = torch.clamp(alpha, min_alpha, max_alpha)
    
    # Compute transmittance
    transmission = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
    transmittance = torch.cat([
        torch.ones_like(transmission[:, :1]),
        transmission[:, :-1]
    ], dim=1)
    
    # Compute weights
    weights = transmittance * alpha
    
    return weights


def alpha_composite_images(
    foreground: torch.Tensor,
    background: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Alpha composite foreground over background.
    
    Args:
        foreground: Foreground image [..., C]
        background: Background image [..., C]
        alpha: Alpha channel [..., 1] or [...]
        
    Returns:
        Composited image [..., C]
    """
    if alpha.dim() == foreground.dim() - 1:
        alpha = alpha.unsqueeze(-1)
    
    return foreground * alpha + background * (1.0 - alpha)