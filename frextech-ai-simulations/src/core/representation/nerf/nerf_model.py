"""
NeRF Model Implementation.

This module implements the core Neural Radiance Fields model,
including different architecture variants and query methods.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
import math
import warnings

from ..base import BaseRepresentation, RepresentationType, RenderMode, SampleStrategy
from . import NeRFConfig, NeRFOutput, NeRFType, ActivationType
from .positional_encoding import PositionalEncoding, PositionalEncodingConfig
from .ray_sampler import RaySampler
from .volume_renderer import VolumeRenderer


class NeRFModel(BaseRepresentation):
    """
    Neural Radiance Fields (NeRF) model.
    
    This implements the core NeRF architecture for representing
    3D scenes as continuous volumetric functions.
    """
    
    def __init__(
        self,
        config: Optional[NeRFConfig] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """
        Initialize NeRF model.
        
        Args:
            config: NeRF configuration
            device: PyTorch device
            **kwargs: Additional configuration parameters
        """
        if config is None:
            config = NeRFConfig(**kwargs)
        
        super().__init__(config, device)
        
        # Store config as NeRFConfig for type hints
        self.nerf_config = config
        
        # Initialize components based on NeRF type
        self._init_architecture()
        
        # Initialize appearance embeddings if needed
        if config.use_appearance_embedding:
            self.appearance_embeddings = nn.Embedding(
                config.num_appearance_embeddings or 1,
                config.appearance_embedding_dim
            )
        else:
            self.appearance_embeddings = None
        
        # Initialize time conditioning if needed
        if config.use_time_conditioning:
            self.time_embedding = nn.Linear(1, config.time_embedding_dim)
        else:
            self.time_embedding = None
        
        # Initialize deformation field if needed
        if config.use_deformation_field:
            self.deformation_field = self._build_deformation_field()
        else:
            self.deformation_field = None
        
        # Initialize feature grid if needed
        if config.nerf_type in [NeRFType.TENSORF, NeRFType.DVGO, NeRFType.PLENOXELS]:
            self._init_feature_grid()
        
        # Initialize activation function
        self.activation = self._get_activation_function()
        
        # Move to device
        self.to(self.device)
        
        self.logger.info(f"Initialized {config.nerf_type.value} NeRF model")
    
    def _init_parameters(self) -> None:
        """Initialize NeRF parameters."""
        # Most parameters are initialized in _init_architecture
        pass
    
    def _init_architecture(self) -> None:
        """Initialize NeRF architecture based on type."""
        config = self.nerf_config
        
        # Create positional encoding
        self.positional_encoding = self._create_positional_encoding()
        
        # Build MLP based on NeRF type
        if config.nerf_type == NeRFType.VANILLA:
            self.mlp = self._build_vanilla_mlp()
        elif config.nerf_type == NeRFType.MIP:
            self.mlp = self._build_mip_mlp()
        elif config.nerf_type == NeRFType.INSTANT_NGP:
            self.mlp = self._build_instant_ngp_mlp()
        elif config.nerf_type == NeRFType.TENSORF:
            self.mlp = self._build_tensorf_mlp()
        elif config.nerf_type == NeRFType.DVGO:
            self.mlp = self._build_dvgo_mlp()
        elif config.nerf_type == NeRFType.PLENOXELS:
            self.mlp = self._build_plenoxels_mlp()
        elif config.nerf_type == NeRFType.K_PLANES:
            self.mlp = self._build_k_planes_mlp()
        elif config.nerf_type == NeRFType.HEX_PLANES:
            self.mlp = self._build_hex_planes_mlp()
        else:
            raise ValueError(f"Unknown NeRF type: {config.nerf_type}")
    
    def _create_positional_encoding(self) -> PositionalEncoding:
        """Create positional encoding based on configuration."""
        config = self.nerf_config
        
        if config.positional_encoding_type == "frequency":
            encoding_config = PositionalEncodingConfig(
                num_frequencies=config.num_encoding_functions_xyz,
                include_identity=True,
                log_sampling=True,
            )
            return PositionalEncoding(encoding_config)
        
        elif config.positional_encoding_type == "hash":
            # Hash encoding (Instant NGP)
            from .hash_encoding import HashEncoding
            return HashEncoding(
                num_levels=config.hash_table_levels,
                hash_table_size=config.hash_table_size,
                feature_dim=config.hash_table_feature_dim,
                per_level_scale=config.per_level_scale,
                bounds=self.config.bounds,
            )
        
        elif config.positional_encoding_type == "spherical_harmonics":
            # Spherical harmonics for direction encoding
            from .spherical_harmonics import SphericalHarmonicsEncoding
            return SphericalHarmonicsEncoding(degree=4)
        
        else:
            # Identity encoding (no encoding)
            class IdentityEncoding(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.output_dim = 3
                
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x
            
            return IdentityEncoding()
    
    def _build_vanilla_mlp(self) -> nn.Module:
        """Build vanilla NeRF MLP."""
        config = self.nerf_config
        
        # Input dimension (position encoding + optional appearance/time)
        input_dim = self.positional_encoding.output_dim
        
        if config.include_view_direction:
            # Direction encoding will be concatenated later
            dir_encoding_dim = config.num_encoding_functions_dir * 2 * 3 + 3
        else:
            dir_encoding_dim = 0
        
        # Build MLP layers
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, config.hidden_dim))
        layers.append(self._get_activation_function())
        
        # Intermediate layers
        for i in range(1, config.num_layers):
            # Skip connection at specified layer
            if i == config.skip_connection_at:
                layers.append(nn.Linear(config.hidden_dim + input_dim, config.hidden_dim))
            else:
                layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            
            layers.append(self._get_activation_function())
        
        # Output layer for density
        layers.append(nn.Linear(config.hidden_dim, 1))
        
        # Branch for color (with view direction)
        if config.include_view_direction:
            color_layers = [
                nn.Linear(config.hidden_dim + dir_encoding_dim, config.hidden_dim // 2),
                self._get_activation_function(),
                nn.Linear(config.hidden_dim // 2, 3),
                nn.Sigmoid(),  # Color in [0, 1]
            ]
        else:
            color_layers = [
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                self._get_activation_function(),
                nn.Linear(config.hidden_dim // 2, 3),
                nn.Sigmoid(),
            ]
        
        return nn.ModuleDict({
            'density_mlp': nn.Sequential(*layers),
            'color_mlp': nn.Sequential(*color_layers),
        })
    
    def _build_mip_mlp(self) -> nn.Module:
        """Build Mip-NeRF MLP (supports conical frustums)."""
        # Mip-NeRF uses integrated positional encoding
        # This is a simplified implementation
        config = self.nerf_config
        
        # For Mip-NeRF, we need to handle positional encoding differently
        # Here we use the same architecture as vanilla but with different encoding
        return self._build_vanilla_mlp()
    
    def _build_instant_ngp_mlp(self) -> nn.Module:
        """Build Instant NGP MLP (with hash encoding)."""
        config = self.nerf_config
        
        # Instant NGP uses a small MLP with hash encoding
        hidden_dim = 64  # Smaller than vanilla NeRF
        
        layers = [
            nn.Linear(self.positional_encoding.output_dim + config.appearance_embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        
        # Density output
        density_layer = nn.Linear(hidden_dim, 1)
        
        # Color output (with view direction)
        color_input_dim = hidden_dim + config.appearance_embedding_dim
        color_layers = [
            nn.Linear(color_input_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        ]
        
        return nn.ModuleDict({
            'base_mlp': nn.Sequential(*layers),
            'density_layer': density_layer,
            'color_mlp': nn.Sequential(*color_layers),
        })
    
    def _build_tensorf_mlp(self) -> nn.Module:
        """Build TensoRF MLP (with tensor decomposition)."""
        config = self.nerf_config
        
        # TensoRF uses tensor decomposition of feature grids
        # This is a simplified placeholder
        from .tensorf import TensorFModel
        
        return TensorFModel(
            grid_resolution=config.grid_resolution,
            feature_dim=config.grid_feature_dim,
            bounds=self.config.bounds,
            device=self.device,
        )
    
    def _build_dvgo_mlp(self) -> nn.Module:
        """Build DVGO MLP (direct voxel grid optimization)."""
        config = self.nerf_config
        
        # DVGO uses explicit voxel grids
        from .dvgo import DVGOModel
        
        return DVGOModel(
            grid_resolution=config.grid_resolution,
            bounds=self.config.bounds,
            device=self.device,
        )
    
    def _build_plenoxels_mlp(self) -> nn.Module:
        """Build Plenoxels model (spherical harmonics in voxels)."""
        config = self.nerf_config
        
        # Plenoxels uses spherical harmonics in sparse voxel grid
        from .plenoxels import PlenoxelsModel
        
        return PlenoxelsModel(
            grid_resolution=config.grid_resolution,
            bounds=self.config.bounds,
            device=self.device,
        )
    
    def _build_k_planes_mlp(self) -> nn.Module:
        """Build K-Planes model (factorized planar features)."""
        config = self.nerf_config
        
        from .k_planes import KPlanesModel
        
        return KPlanesModel(
            grid_resolution=config.grid_resolution,
            feature_dim=config.grid_feature_dim,
            bounds=self.config.bounds,
            device=self.device,
        )
    
    def _build_hex_planes_mlp(self) -> nn.Module:
        """Build HexPlanes model (for dynamic scenes)."""
        config = self.nerf_config
        
        from .hex_planes import HexPlanesModel
        
        return HexPlanesModel(
            grid_resolution=config.grid_resolution,
            feature_dim=config.grid_feature_dim,
            bounds=self.config.bounds,
            device=self.device,
        )
    
    def _build_deformation_field(self) -> nn.Module:
        """Build deformation field for dynamic scenes."""
        config = self.nerf_config
        
        layers = []
        input_dim = 4 if config.use_time_conditioning else 3  # xyz + time
        
        # Input layer
        layers.append(nn.Linear(input_dim, config.deformation_field_dim))
        layers.append(self._get_activation_function())
        
        # Hidden layers
        for _ in range(config.deformation_field_layers - 2):
            layers.append(nn.Linear(config.deformation_field_dim, config.deformation_field_dim))
            layers.append(self._get_activation_function())
        
        # Output layer (3D deformation)
        layers.append(nn.Linear(config.deformation_field_dim, 3))
        
        return nn.Sequential(*layers)
    
    def _init_feature_grid(self) -> None:
        """Initialize feature grid for grid-based methods."""
        config = self.nerf_config
        
        if config.nerf_type == NeRFType.TENSORF:
            # TensoRF feature grid
            self.feature_grid = nn.Parameter(
                torch.randn(
                    1, config.grid_feature_dim,
                    *config.grid_resolution,
                    device=self.device
                ) * 0.01
            )
        
        elif config.nerf_type == NeRFType.DVGO:
            # DVGO density and color grids
            self.density_grid = nn.Parameter(
                torch.zeros(
                    1, 1, *config.grid_resolution,
                    device=self.device
                )
            )
            self.color_grid = nn.Parameter(
                torch.randn(
                    1, config.grid_feature_dim,
                    *config.grid_resolution,
                    device=self.device
                ) * 0.01
            )
        
        elif config.nerf_type == NeRFType.PLENOXELS:
            # Plenoxels spherical harmonics coefficients
            # 9 coefficients for 2nd degree SH (RGB * 9)
            self.sh_coeffs = nn.Parameter(
                torch.zeros(
                    1, 27, *config.grid_resolution,  # 3 colors * 9 coefficients
                    device=self.device
                )
            )
            self.density_grid = nn.Parameter(
                torch.zeros(
                    1, 1, *config.grid_resolution,
                    device=self.device
                )
            )
    
    def _get_activation_function(self) -> nn.Module:
        """Get activation function based on configuration."""
        config = self.nerf_config
        
        if config.activation_type == ActivationType.RELU:
            return nn.ReLU(inplace=True)
        elif config.activation_type == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU(negative_slope=config.activation_slope, inplace=True)
        elif config.activation_type == ActivationType.ELU:
            return nn.ELU(inplace=True)
        elif config.activation_type == ActivationType.SELU:
            return nn.SELU(inplace=True)
        elif config.activation_type == ActivationType.SINE:
            # SIREN activation
            class SineActivation(nn.Module):
                def __init__(self, omega=30.0):
                    super().__init__()
                    self.omega = omega
                
                def forward(self, x):
                    return torch.sin(self.omega * x)
            
            return SineActivation(config.sine_omega)
        elif config.activation_type == ActivationType.SWISH:
            return nn.SiLU(inplace=True)
        elif config.activation_type == ActivationType.GELU:
            return nn.GELU()
        else:
            warnings.warn(f"Unknown activation type: {config.activation_type}, using ReLU")
            return nn.ReLU(inplace=True)
    
    def forward(
        self,
        positions: torch.Tensor,
        view_directions: Optional[torch.Tensor] = None,
        appearance_ids: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        return_gradients: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, NeRFOutput]:
        """
        Forward pass through NeRF model.
        
        Args:
            positions: 3D positions [N, 3] or [N, S, 3]
            view_directions: View directions [N, 3] or [N, S, 3] (optional)
            appearance_ids: Appearance embedding indices [N] (optional)
            timestamps: Timestamps for dynamic scenes [N] or [N, 1] (optional)
            return_gradients: Whether to compute gradients
            **kwargs: Additional parameters
            
        Returns:
            NeRFOutput or tensor
        """
        # Handle batched inputs
        original_shape = positions.shape
        if positions.dim() == 3:
            # [N, S, 3] -> [N*S, 3]
            positions = positions.reshape(-1, 3)
            if view_directions is not None:
                view_directions = view_directions.reshape(-1, 3)
        
        # Apply deformation field if available
        if self.deformation_field is not None and timestamps is not None:
            # Concatenate positions and time
            if timestamps.dim() == 1:
                timestamps = timestamps.unsqueeze(-1)
            
            # Repeat timestamps to match positions
            if timestamps.shape[0] == 1 and positions.shape[0] > 1:
                timestamps = timestamps.repeat(positions.shape[0], 1)
            
            deformation_input = torch.cat([positions, timestamps], dim=-1)
            deformation = self.deformation_field(deformation_input)
            positions = positions + deformation
        
        # Encode positions
        encoded_positions = self.positional_encoding(positions)
        
        # Add appearance embedding if needed
        if self.appearance_embeddings is not None and appearance_ids is not None:
            if appearance_ids.dim() == 0:
                appearance_ids = appearance_ids.unsqueeze(0)
            
            # Get appearance embeddings
            appearance_embeds = self.appearance_embeddings(appearance_ids)
            
            # Repeat if needed
            if appearance_embeds.shape[0] == 1 and encoded_positions.shape[0] > 1:
                appearance_embeds = appearance_embeds.repeat(encoded_positions.shape[0], 1)
            
            encoded_positions = torch.cat([encoded_positions, appearance_embeds], dim=-1)
        
        # Add time conditioning if needed
        if self.time_embedding is not None and timestamps is not None:
            time_embeds = self.time_embedding(timestamps.float())
            encoded_positions = torch.cat([encoded_positions, time_embeds], dim=-1)
        
        # Query NeRF based on type
        config = self.nerf_config
        
        if config.nerf_type == NeRFType.VANILLA:
            output = self._forward_vanilla(
                encoded_positions, view_directions, return_gradients
            )
        elif config.nerf_type == NeRFType.INSTANT_NGP:
            output = self._forward_instant_ngp(
                encoded_positions, view_directions, appearance_ids, return_gradients
            )
        elif config.nerf_type in [NeRFType.TENSORF, NeRFType.DVGO, 
                                   NeRFType.PLENOXELS, NeRFType.K_PLANES, 
                                   NeRFType.HEX_PLANES]:
            output = self._forward_grid_based(
                positions, view_directions, return_gradients
            )
        else:
            # Default to vanilla
            output = self._forward_vanilla(
                encoded_positions, view_directions, return_gradients
            )
        
        # Reshape outputs if needed
        if len(original_shape) == 3:
            # Reshape back to [N, S, ...]
            batch_size, num_samples = original_shape[:2]
            output.density = output.density.reshape(batch_size, num_samples, -1)
            output.color = output.color.reshape(batch_size, num_samples, -1)
            
            if output.features is not None:
                output.features = output.features.reshape(batch_size, num_samples, -1)
            
            if output.normals is not None:
                output.normals = output.normals.reshape(batch_size, num_samples, -1)
        
        # Store positions and view directions
        output.positions = positions.reshape(original_shape)
        if view_directions is not None:
            output.view_directions = view_directions.reshape(original_shape)
        
        return output
    
    def _forward_vanilla(
        self,
        encoded_positions: torch.Tensor,
        view_directions: Optional[torch.Tensor],
        return_gradients: bool = False
    ) -> NeRFOutput:
        """Forward pass for vanilla NeRF."""
        config = self.nerf_config
        
        # Compute density
        x = encoded_positions
        
        # Pass through density MLP
        for i, layer in enumerate(self.mlp['density_mlp']):
            if i == config.skip_connection_at * 2:  # *2 because each layer has activation
                x = torch.cat([x, encoded_positions], dim=-1)
            x = layer(x)
        
        # Density output (apply bias and activation)
        density = x
        
        # Apply density bias and activation
        density = density + config.density_bias
        density = F.softplus(density)  # Ensure positive density
        
        # Compute color
        if config.include_view_direction and view_directions is not None:
            # Encode view directions
            if config.positional_encoding_type == "frequency":
                # Frequency encoding for directions
                dir_encoding = self._encode_frequencies(
                    view_directions, config.num_encoding_functions_dir
                )
            elif config.positional_encoding_type == "spherical_harmonics":
                # Spherical harmonics encoding
                dir_encoding = self.positional_encoding(view_directions)
            else:
                # Raw directions
                dir_encoding = view_directions
            
            # Concatenate with intermediate features
            color_input = torch.cat([x[:, :config.hidden_dim], dir_encoding], dim=-1)
        else:
            color_input = x[:, :config.hidden_dim]
        
        # Pass through color MLP
        color = self.mlp['color_mlp'](color_input)
        
        # Compute normals if requested
        normals = None
        if return_gradients:
            # Compute gradient of density w.r.t. position
            positions = encoded_positions[:, :3]  # Extract original positions
            positions.requires_grad_(True)
            
            # Recompute density with gradient tracking
            encoded_positions_grad = self.positional_encoding(positions)
            density_grad = self.mlp['density_mlp'](encoded_positions_grad)
            density_grad = density_grad + config.density_bias
            density_grad = F.softplus(density_grad)
            
            # Compute gradient
            grad_outputs = torch.ones_like(density_grad)
            gradients = torch.autograd.grad(
                outputs=density_grad,
                inputs=positions,
                grad_outputs=grad_outputs,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            
            # Normalize to get normals
            normals = -F.normalize(gradients, dim=-1)  # Negative gradient points inward
        
        return NeRFOutput(
            density=density,
            color=color,
            normals=normals,
        )
    
    def _forward_instant_ngp(
        self,
        encoded_positions: torch.Tensor,
        view_directions: Optional[torch.Tensor],
        appearance_ids: Optional[torch.Tensor],
        return_gradients: bool = False
    ) -> NeRFOutput:
        """Forward pass for Instant NGP."""
        # Base MLP
        x = self.mlp['base_mlp'](encoded_positions)
        
        # Density
        density = self.mlp['density_layer'](x)
        density = F.softplus(density + self.nerf_config.density_bias)
        
        # Color (with view direction)
        if view_directions is not None:
            # Encode view direction
            if self.nerf_config.include_view_direction:
                # Simple frequency encoding
                dir_encoding = self._encode_frequencies(
                    view_directions, self.nerf_config.num_encoding_functions_dir
                )
                
                # Add appearance embedding if available
                if appearance_ids is not None and self.appearance_embeddings is not None:
                    appearance_embeds = self.appearance_embeddings(appearance_ids)
                    if appearance_embeds.shape[0] == 1 and dir_encoding.shape[0] > 1:
                        appearance_embeds = appearance_embeds.repeat(dir_encoding.shape[0], 1)
                    dir_encoding = torch.cat([dir_encoding, appearance_embeds], dim=-1)
                
                color_input = torch.cat([x, dir_encoding], dim=-1)
            else:
                color_input = x
        else:
            color_input = x
        
        color = self.mlp['color_mlp'](color_input)
        
        return NeRFOutput(
            density=density,
            color=color,
        )
    
    def _forward_grid_based(
        self,
        positions: torch.Tensor,
        view_directions: Optional[torch.Tensor],
        return_gradients: bool = False
    ) -> NeRFOutput:
        """Forward pass for grid-based methods."""
        config = self.nerf_config
        
        if config.nerf_type == NeRFType.TENSORF:
            # TensoRF forward pass
            return self.mlp(positions, view_directions, return_gradients)
        
        elif config.nerf_type == NeRFType.DVGO:
            # DVGO forward pass
            return self.mlp(positions, view_directions, return_gradients)
        
        elif config.nerf_type == NeRFType.PLENOXELS:
            # Plenoxels forward pass
            return self.mlp(positions, view_directions, return_gradients)
        
        elif config.nerf_type == NeRFType.K_PLANES:
            # K-Planes forward pass
            return self.mlp(positions, view_directions, return_gradients)
        
        elif config.nerf_type == NeRFType.HEX_PLANES:
            # HexPlanes forward pass
            return self.mlp(positions, view_directions, return_gradients)
        
        else:
            raise ValueError(f"Unsupported grid-based NeRF type: {config.nerf_type}")
    
    def _encode_frequencies(
        self,
        x: torch.Tensor,
        num_frequencies: int,
        log_sampling: bool = True
    ) -> torch.Tensor:
        """
        Apply frequency encoding to input.
        
        Args:
            x: Input tensor [N, D]
            num_frequencies: Number of frequency bands
            log_sampling: Whether to use log sampling
            
        Returns:
            Encoded tensor [N, D * (2 * L + 1)]
        """
        if num_frequencies <= 0:
            return x
        
        # Generate frequencies
        if log_sampling:
            frequencies = 2.0 ** torch.arange(
                0, num_frequencies, dtype=torch.float32, device=x.device
            )
        else:
            frequencies = torch.arange(
                1, num_frequencies + 1, dtype=torch.float32, device=x.device
            )
        
        # Apply encoding
        encoded = [x]
        for freq in frequencies:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        return torch.cat(encoded, dim=-1)
    
    def query_points(
        self,
        positions: torch.Tensor,
        view_directions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Query NeRF at specific points.
        
        Args:
            positions: Query points [N, 3]
            view_directions: View directions [N, 3] (optional)
            **kwargs: Additional query parameters
            
        Returns:
            Dictionary with queried properties
        """
        output = self.forward(positions, view_directions, **kwargs)
        return output.to_dict()
    
    def render(
        self,
        cameras: Any,
        resolution: Tuple[int, int] = (512, 512),
        mode: Union[str, RenderMode] = RenderMode.RGB,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Render the NeRF from given camera viewpoints.
        
        Args:
            cameras: Camera parameters
            resolution: Output resolution (height, width)
            mode: Rendering mode
            **kwargs: Additional rendering parameters
            
        Returns:
            Rendered output(s)
        """
        # Create ray sampler and volume renderer
        ray_sampler = RaySampler(
            near=self.nerf_config.near_distance,
            far=self.nerf_config.far_distance,
            num_coarse_samples=self.nerf_config.num_coarse_samples,
            num_fine_samples=self.nerf_config.num_fine_samples,
            perturb=self.training and self.nerf_config.perturb,
        )
        
        volume_renderer = VolumeRenderer(
            white_bkgd=self.nerf_config.white_bkgd,
            raw_noise_std=self.nerf_config.raw_noise_std if self.training else 0.0,
            min_alpha=self.nerf_config.min_alpha,
        )
        
        # Generate rays from cameras
        # This is simplified - in practice, you'd parse camera parameters
        height, width = resolution
        
        if isinstance(cameras, dict):
            # Assume cameras dict contains poses and intrinsics
            poses = cameras.get('poses')
            intrinsics = cameras.get('intrinsics')
            
            if poses is None or intrinsics is None:
                raise ValueError("Cameras dict must contain 'poses' and 'intrinsics'")
            
            # Generate rays for all pixels
            # This is a simplified implementation
            rays_o, rays_d = self._generate_rays(poses, intrinsics, height, width)
        
        elif isinstance(cameras, torch.Tensor):
            # Assume cameras tensor is ray origins and directions
            if cameras.shape[-1] == 6:
                # [N, 6] where first 3 are origin, last 3 are direction
                rays_o = cameras[..., :3]
                rays_d = cameras[..., 3:]
            else:
                raise ValueError(f"Unexpected camera tensor shape: {cameras.shape}")
        
        else:
            raise ValueError(f"Unsupported camera type: {type(cameras)}")
        
        # Sample points along rays
        ray_bundle = ray_sampler.sample_along_rays(rays_o, rays_d)
        
        # Query NeRF
        nerf_output = self.forward(
            ray_bundle.samples.positions,
            ray_bundle.samples.view_directions,
            **kwargs
        )
        
        # Render
        rendering_output = volume_renderer.render(
            nerf_output.density,
            nerf_output.color,
            ray_bundle.samples.depths,
            ray_bundle.samples.deltas,
            mode=mode,
        )
        
        # Reshape to image
        if rays_o.dim() == 3:
            # Batched rays [B, H*W, 3] -> reshape to [B, H, W, ...]
            batch_size = rays_o.shape[0]
            images = {}
            
            for key, value in rendering_output.to_dict().items():
                if value is not None:
                    images[key] = value.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
        else:
            # Single image [H*W, ...] -> reshape to [H, W, ...]
            images = {}
            
            for key, value in rendering_output.to_dict().items():
                if value is not None:
                    images[key] = value.reshape(height, width, -1).permute(2, 0, 1)
        
        if mode == RenderMode.RGB:
            return images.get('colors', torch.zeros(3, height, width, device=self.device))
        elif mode == RenderMode.DEPTH:
            return images.get('depths', torch.zeros(1, height, width, device=self.device))
        elif mode == RenderMode.ALL:
            return images
        else:
            # Return specific output
            return images.get(mode.value, torch.zeros(3, height, width, device=self.device))
    
    def sample_points(
        self,
        num_points: int,
        strategy: Union[str, SampleStrategy] = SampleStrategy.UNIFORM,
        return_attributes: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Sample points from the NeRF representation.
        
        Args:
            num_points: Number of points to sample
            strategy: Sampling strategy
            return_attributes: Whether to return point attributes
            **kwargs: Additional sampling parameters
            
        Returns:
            Tuple of (positions, attributes)
        """
        strategy = SampleStrategy(strategy) if isinstance(strategy, str) else strategy
        
        # Get bounds
        min_bound, max_bound = self.get_bounds()
        
        if strategy == SampleStrategy.UNIFORM:
            # Sample uniformly within bounds
            positions = torch.rand(
                (num_points, 3),
                device=self.device,
                dtype=self.dtype
            )
            
            # Scale to bounds
            for i in range(3):
                positions[:, i] = positions[:, i] * (max_bound[i] - min_bound[i]) + min_bound[i]
        
        elif strategy == SampleStrategy.SURFACE:
            # Sample near surface (high density regions)
            # This is approximated by importance sampling
            positions = self._sample_surface_points(num_points, **kwargs)
        
        elif strategy == SampleStrategy.VOLUME:
            # Sample within volume with density-based importance
            positions = self._sample_volume_points(num_points, **kwargs)
        
        elif strategy == SampleStrategy.GRID:
            # Sample on a regular grid
            grid_res = int(num_points ** (1/3))
            positions = self._sample_grid_points(grid_res, **kwargs)
        
        else:
            # Default to uniform
            positions = torch.rand(
                (num_points, 3),
                device=self.device,
                dtype=self.dtype
            )
            
            for i in range(3):
                positions[:, i] = positions[:, i] * (max_bound[i] - min_bound[i]) + min_bound[i]
        
        # Get attributes if requested
        attributes = None
        if return_attributes:
            with torch.no_grad():
                output = self.forward(positions)
                # Combine color and density as attributes
                attributes = torch.cat([output.color, output.density], dim=-1)
        
        return positions, attributes
    
    def _sample_surface_points(self, num_points: int, **kwargs) -> torch.Tensor:
        """Sample points near the surface."""
        # Use ray marching to find surface points
        # This is a simplified implementation
        
        # Generate random rays
        bounds = self.get_bounds()
        center = (bounds[0] + bounds[1]) / 2
        radius = torch.norm(bounds[1] - bounds[0]) / 2
        
        # Random directions
        directions = F.normalize(
            torch.randn(num_points, 3, device=self.device),
            dim=-1
        )
        
        # Random origins on sphere
        origins = center + directions * radius * 1.5
        
        # March rays
        positions = []
        
        with torch.no_grad():
            for i in range(0, num_points, 1024):
                batch_size = min(1024, num_points - i)
                batch_origins = origins[i:i+batch_size]
                batch_directions = directions[i:i+batch_size]
                
                # Simple sphere tracing
                points = self._sphere_trace(
                    batch_origins, batch_directions, 
                    max_steps=64, threshold=0.01
                )
                
                positions.append(points)
        
        positions = torch.cat(positions, dim=0)
        
        # Filter out points that didn't converge
        mask = torch.isfinite(positions).all(dim=-1)
        positions = positions[mask]
        
        # Resample if we don't have enough points
        if positions.shape[0] < num_points:
            additional = self.sample_points(
                num_points - positions.shape[0],
                SampleStrategy.UNIFORM,
                return_attributes=False
            )[0]
            positions = torch.cat([positions, additional], dim=0)
        
        return positions[:num_points]
    
    def _sphere_trace(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        max_steps: int = 64,
        threshold: float = 0.01
    ) -> torch.Tensor:
        """Simple sphere tracing to find surface."""
        positions = origins.clone()
        
        for step in range(max_steps):
            # Query density at current positions
            with torch.no_grad():
                output = self.forward(positions)
                density = output.density.squeeze(-1)
            
            # Check convergence
            converged = density < threshold
            
            if converged.all():
                break
            
            # March forward
            step_size = torch.clamp(density / 10.0, 0.001, 0.1)
            positions = positions + directions * step_size.unsqueeze(-1)
        
        return positions
    
    def _sample_volume_points(self, num_points: int, **kwargs) -> torch.Tensor:
        """Sample points within volume with density-based importance."""
        # First sample uniformly
        uniform_points = self.sample_points(
            num_points * 10,  # Oversample
            SampleStrategy.UNIFORM,
            return_attributes=False
        )[0]
        
        # Query density
        with torch.no_grad():
            output = self.forward(uniform_points)
            density = output.density.squeeze(-1)
        
        # Importance sample based on density
        weights = density + 1e-6  # Add small epsilon
        weights = weights / weights.sum()
        
        # Sample indices
        indices = torch.multinomial(weights, num_points, replacement=True)
        
        return uniform_points[indices]
    
    def _sample_grid_points(self, resolution: int, **kwargs) -> torch.Tensor:
        """Sample points on a regular grid."""
        bounds = self.get_bounds()
        min_bound, max_bound = bounds
        
        # Create grid
        x = torch.linspace(min_bound[0], max_bound[0], resolution, device=self.device)
        y = torch.linspace(min_bound[1], max_bound[1], resolution, device=self.device)
        z = torch.linspace(min_bound[2], max_bound[2], resolution, device=self.device)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        positions = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        return positions
    
    def _generate_rays(
        self,
        poses: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays from camera poses and intrinsics.
        
        Args:
            poses: Camera poses [B, 3, 4] or [B, 4, 4]
            intrinsics: Camera intrinsics [B, 3, 3] or [B, 4]
            height: Image height
            width: Image width
            
        Returns:
            Tuple of (ray_origins, ray_directions)
        """
        batch_size = poses.shape[0]
        
        # Create pixel grid
        y, x = torch.meshgrid(
            torch.arange(height, device=self.device),
            torch.arange(width, device=self.device),
            indexing='ij'
        )
        
        x = x.flatten().float()
        y = y.flatten().float()
        
        # Repeat for batch
        x = x.repeat(batch_size)
        y = y.repeat(batch_size)
        
        # Repeat poses and intrinsics
        poses = poses.repeat_interleave(height * width, dim=0)
        
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.repeat_interleave(height * width, dim=0)
        else:
            intrinsics = intrinsics.repeat_interleave(height * width, dim=0)
        
        # Generate rays
        # This is a simplified implementation
        # Extract rotation and translation
        if poses.shape[-2:] == (4, 4):
            rotation = poses[:, :3, :3]
            translation = poses[:, :3, 3]
        else:
            rotation = poses[:, :3, :3]
            translation = poses[:, :3, 3]
        
        # Get intrinsics parameters
        if intrinsics.dim() == 2 and intrinsics.shape[-1] == 4:
            fx = intrinsics[:, 0]
            fy = intrinsics[:, 1]
            cx = intrinsics[:, 2]
            cy = intrinsics[:, 3]
        else:
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1]
            cx = intrinsics[:, 0, 2]
            cy = intrinsics[:, 1, 2]
        
        # Convert to normalized device coordinates
        x_ndc = (x - cx) / fx
        y_ndc = (y - cy) / fy
        
        # Direction vectors in camera space
        directions = torch.stack([x_ndc, y_ndc, torch.ones_like(x_ndc)], dim=-1)
        
        # Transform to world space
        ray_directions = torch.bmm(rotation, directions.unsqueeze(-1)).squeeze(-1)
        ray_directions = F.normalize(ray_directions, dim=-1)
        
        # Ray origins
        ray_origins = translation
        
        # Reshape to [B, H*W, 3]
        ray_origins = ray_origins.reshape(batch_size, height * width, 3)
        ray_directions = ray_directions.reshape(batch_size, height * width, 3)
        
        return ray_origins, ray_directions
    
    def get_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the axis-aligned bounding box of the representation.
        
        Returns:
            Tuple of (min_bounds, max_bounds)
        """
        bounds = self.config.bounds
        min_bound = torch.tensor([b[0] for b in bounds], device=self.device, dtype=torch.float32)
        max_bound = torch.tensor([b[1] for b in bounds], device=self.device, dtype=torch.float32)
        
        return min_bound, max_bound
    
    def get_type(self) -> RepresentationType:
        """
        Get the type of representation.
        
        Returns:
            Representation type
        """
        return RepresentationType.NERF
    
    def compute_metrics(
        self,
        reference: Any,
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> RepresentationMetrics:
        """
        Compute quality and performance metrics.
        
        Args:
            reference: Reference data (images, point cloud, etc.)
            metrics: List of metrics to compute (None for all)
            **kwargs: Additional parameters
            
        Returns:
            RepresentationMetrics object
        """
        # This would be implemented with specific metric computations
        # For now, return placeholder metrics
        from ..base import RepresentationMetrics
        
        return RepresentationMetrics(
            psnr=25.0,  # Example value
            ssim=0.9,   # Example value
            lpips=0.1,  # Example value
        )
    
    def optimize_for_inference(self) -> None:
        """Optimize NeRF for inference."""
        super().optimize_for_inference()
        
        # Additional NeRF-specific optimizations
        if hasattr(self, 'feature_grid'):
            # Convert feature grids to half precision if using GPU
            if self.device.type == 'cuda':
                self.feature_grid.data = self.feature_grid.data.half()
        
        # Fuse batch norm layers if any
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.eval()
        
        self.logger.info("NeRF optimized for inference")
    
    def _export_mesh(
        self,
        filepath: Path,
        format: str = "ply",
        quality: str = "high",
        **kwargs
    ) -> Path:
        """
        Export NeRF as mesh using marching cubes.
        
        Args:
            filepath: Path to export file
            format: Export format
            quality: Export quality
            **kwargs: Additional parameters
            
        Returns:
            Path to exported file
        """
        try:
            import mcubes
        except ImportError:
            raise ImportError("Please install PyMCubes: pip install PyMCubes")
        
        # Extract mesh
        resolution = {
            'low': 128,
            'medium': 256,
            'high': 512,
            'ultra': 1024,
        }.get(quality, 256)
        
        threshold = kwargs.get('threshold', 25.0)
        
        vertices, faces = self.extract_mesh(resolution, threshold)
        
        # Export based on format
        if format == "ply":
            self._export_ply_mesh(filepath, vertices, faces)
        elif format == "obj":
            self._export_obj_mesh(filepath, vertices, faces)
        else:
            # Default to PLY
            filepath = filepath.with_suffix('.ply')
            self._export_ply_mesh(filepath, vertices, faces)
        
        return filepath
    
    def extract_mesh(
        self,
        resolution: int = 256,
        threshold: float = 25.0,
        chunk_size: int = 65536,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mesh from NeRF using marching cubes.
        
        Args:
            resolution: Marching cubes resolution
            threshold: Density threshold for surface extraction
            chunk_size: Chunk size for processing
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (vertices, faces)
        """
        import mcubes
        
        self.eval()
        
        # Get bounds
        min_bound, max_bound = self.get_bounds()
        
        # Create grid
        x = torch.linspace(min_bound[0], max_bound[0], resolution, device=self.device)
        y = torch.linspace(min_bound[1], max_bound[1], resolution, device=self.device)
        z = torch.linspace(min_bound[2], max_bound[2], resolution, device=self.device)
        
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
        
        # Query density in chunks
        densities = torch.zeros(grid_points.shape[0], device=self.device)
        
        with torch.no_grad():
            for chunk_start in range(0, grid_points.shape[0], chunk_size):
                chunk_end = min(chunk_start + chunk_size, grid_points.shape[0])
                chunk_points = grid_points[chunk_start:chunk_end]
                
                output = self.forward(chunk_points)
                densities[chunk_start:chunk_end] = output.density.squeeze(-1)
        
        # Reshape to 3D grid
        density_grid = densities.reshape(resolution, resolution, resolution).cpu().numpy()
        
        # Marching cubes
        vertices, faces = mcubes.marching_cubes(density_grid, threshold)
        
        # Scale vertices to world coordinates
        scale = (max_bound - min_bound).cpu().numpy()
        offset = min_bound.cpu().numpy()
        
        vertices = vertices / (resolution - 1) * scale + offset
        
        # Convert to torch tensors
        vertices = torch.from_numpy(vertices).float()
        faces = torch.from_numpy(faces.astype(np.int32)).long()
        
        return vertices, faces
    
    def _export_ply_mesh(
        self,
        filepath: Path,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> None:
        """Export mesh as PLY file."""
        import plyfile
        
        vertices_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy()
        
        # Create vertex data
        vertex_data = np.zeros(
            vertices_np.shape[0],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        
        vertex_data['x'] = vertices_np[:, 0]
        vertex_data['y'] = vertices_np[:, 1]
        vertex_data['z'] = vertices_np[:, 2]
        
        # Create face data
        face_data = np.zeros(
            faces_np.shape[0],
            dtype=[('vertex_indices', 'i4', (3,))]
        )
        
        face_data['vertex_indices'] = faces_np
        
        # Create PLY elements
        vertex_element = plyfile.PlyElement.describe(vertex_data, 'vertex')
        face_element = plyfile.PlyElement.describe(face_data, 'face')
        
        # Write PLY file
        plyfile.PlyData([vertex_element, face_element], text=False).write(str(filepath))
    
    def _export_obj_mesh(
        self,
        filepath: Path,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> None:
        """Export mesh as OBJ file."""
        vertices_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy() + 1  # OBJ indices start at 1
        
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices_np:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces
            for face in faces_np:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    def __str__(self) -> str:
        """String representation."""
        base_str = super().__str__()
        
        # Add NeRF-specific information
        num_params = sum(p.numel() for p in self.parameters())
        
        return (
            f"{base_str}\n"
            f"  NeRF type: {self.nerf_config.nerf_type.value}\n"
            f"  Parameters: {num_params:,}\n"
            f"  Encoding: {self.nerf_config.positional_encoding_type}\n"
            f"  Samples per ray: {self.nerf_config.total_samples}\n"
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_nerf_model(
    config: Optional[NeRFConfig] = None,
    device: Optional[torch.device] = None,
    **kwargs
) -> NeRFModel:
    """
    Create a NeRF model (convenience function).
    
    Args:
        config: NeRF configuration
        device: PyTorch device
        **kwargs: Additional configuration parameters
        
    Returns:
        NeRF model
    """
    return NeRFModel(config, device, **kwargs)


def load_nerf_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    **kwargs
) -> NeRFModel:
    """
    Load NeRF model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
        **kwargs: Additional loading parameters
        
    Returns:
        Loaded NeRF model
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
    model = NeRFModel(config, device)
    
    # Load state
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load additional state
    if 'iteration' in checkpoint:
        model._iteration = checkpoint['iteration']
    
    if 'loss_history' in checkpoint:
        model._loss_history = checkpoint['loss_history']
    
    return model