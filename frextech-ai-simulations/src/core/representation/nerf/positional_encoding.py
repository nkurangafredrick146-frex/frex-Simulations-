"""
Positional Encoding for NeRF.

This module implements positional encoding (also known as Fourier feature mapping)
which maps input coordinates to higher-dimensional feature vectors to help
the model learn high-frequency functions.
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


class EncodingType(Enum):
    """Enumeration of positional encoding types."""
    FREQUENCY = "frequency"           # Fourier feature encoding
    HASH = "hash"                     # Hash encoding (Instant NGP)
    SPHERICAL_HARMONICS = "spherical_harmonics"  # Spherical harmonics
    IDENTITY = "identity"             # No encoding (identity)
    SINE = "sine"                     # Sine encoding (SIREN)
    GAUSSIAN = "gaussian"             # Gaussian encoding
    WAVELET = "wavelet"               # Wavelet encoding
    LEARNED = "learned"               # Learned encoding


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encoding."""
    
    # Basic parameters
    encoding_type: Union[str, EncodingType] = EncodingType.FREQUENCY
    input_dim: int = 3
    output_dim: Optional[int] = None
    
    # Frequency encoding parameters
    num_frequencies: int = 10
    include_identity: bool = True
    log_sampling: bool = True
    frequency_base: float = 2.0
    frequency_scale: float = 1.0
    max_frequency: Optional[float] = None
    min_frequency: Optional[float] = None
    
    # Hash encoding parameters (Instant NGP)
    hash_table_size: int = 2**19
    hash_table_levels: int = 16
    hash_table_feature_dim: int = 2
    per_level_scale: float = 1.5
    hash_table_init_scale: float = 0.0001
    
    # Spherical harmonics parameters
    sh_degree: int = 4  # Degree of spherical harmonics
    
    # Sine encoding parameters (SIREN)
    sine_omega: float = 30.0
    sine_layers: int = 1
    
    # Gaussian encoding parameters
    gaussian_bands: int = 10
    gaussian_scale: float = 1.0
    
    # Wavelet encoding parameters
    wavelet_levels: int = 4
    wavelet_type: str = "haar"
    
    # Learned encoding parameters
    learned_features: int = 64
    learned_resolution: int = 128
    
    # Normalization
    normalize_input: bool = False
    input_bounds: Optional[List[List[float]]] = None
    
    # Activation
    activation: Optional[str] = None  # 'sin', 'cos', 'relu', 'tanh', etc.
    activation_after_encoding: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        # Convert string enums
        if isinstance(self.encoding_type, str):
            self.encoding_type = EncodingType(self.encoding_type.lower())
        
        # Set output dimension if not specified
        if self.output_dim is None:
            if self.encoding_type == EncodingType.FREQUENCY:
                self.output_dim = self.input_dim * (2 * self.num_frequencies + (1 if self.include_identity else 0))
            elif self.encoding_type == EncodingType.HASH:
                self.output_dim = self.hash_table_levels * self.hash_table_feature_dim
            elif self.encoding_type == EncodingType.SPHERICAL_HARMONICS:
                # Spherical harmonics of degree L has (L+1)^2 coefficients
                self.output_dim = (self.sh_degree + 1) ** 2
            elif self.encoding_type == EncodingType.IDENTITY:
                self.output_dim = self.input_dim
            elif self.encoding_type == EncodingType.SINE:
                self.output_dim = self.input_dim * (2 * self.num_frequencies + (1 if self.include_identity else 0))
            elif self.encoding_type == EncodingType.GAUSSIAN:
                self.output_dim = self.input_dim * self.gaussian_bands
            elif self.encoding_type == EncodingType.WAVELET:
                self.output_dim = self.input_dim * (2 ** self.wavelet_levels)
            elif self.encoding_type == EncodingType.LEARNED:
                self.output_dim = self.learned_features
            else:
                raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Validate parameters
        if self.num_frequencies < 0:
            raise ValueError(f"num_frequencies must be non-negative, got {self.num_frequencies}")
        
        if self.hash_table_levels < 1:
            raise ValueError(f"hash_table_levels must be positive, got {self.hash_table_levels}")
        
        if self.hash_table_feature_dim < 1:
            raise ValueError(f"hash_table_feature_dim must be positive, got {self.hash_table_feature_dim}")
        
        if self.sh_degree < 0:
            raise ValueError(f"sh_degree must be non-negative, got {self.sh_degree}")
    
    @property
    def frequencies(self) -> torch.Tensor:
        """Get frequency bands for frequency encoding."""
        if self.log_sampling:
            # Log-linear sampling: 2^0, 2^1, ..., 2^(L-1)
            freqs = self.frequency_base ** torch.arange(
                0, self.num_frequencies, dtype=torch.float32
            )
        else:
            # Linear sampling: 1, 2, ..., L
            freqs = torch.arange(1, self.num_frequencies + 1, dtype=torch.float32)
        
        # Apply scaling
        freqs = freqs * self.frequency_scale
        
        # Apply min/max constraints
        if self.min_frequency is not None:
            freqs = torch.clamp(freqs, min=self.min_frequency)
        
        if self.max_frequency is not None:
            freqs = torch.clamp(freqs, max=self.max_frequency)
        
        return freqs


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for NeRF.
    
    This module maps input coordinates to higher-dimensional feature vectors
    using various encoding strategies.
    """
    
    def __init__(
        self,
        config: Optional[PositionalEncodingConfig] = None,
        input_dim: Optional[int] = None,
        num_frequencies: Optional[int] = None,
        encoding_type: Optional[Union[str, EncodingType]] = None,
        **kwargs
    ):
        """
        Initialize positional encoding.
        
        Args:
            config: Positional encoding configuration
            input_dim: Input dimension (overrides config)
            num_frequencies: Number of frequency bands (overrides config)
            encoding_type: Encoding type (overrides config)
            **kwargs: Additional configuration parameters
        """
        super().__init__()
        
        # Create or update config
        if config is None:
            config_dict = {}
            if input_dim is not None:
                config_dict['input_dim'] = input_dim
            if num_frequencies is not None:
                config_dict['num_frequencies'] = num_frequencies
            if encoding_type is not None:
                config_dict['encoding_type'] = encoding_type
            config_dict.update(kwargs)
            config = PositionalEncodingConfig(**config_dict)
        else:
            # Update config with provided parameters
            config_dict = config.__dict__.copy()
            if input_dim is not None:
                config_dict['input_dim'] = input_dim
            if num_frequencies is not None:
                config_dict['num_frequencies'] = num_frequencies
            if encoding_type is not None:
                config_dict['encoding_type'] = encoding_type
            config_dict.update(kwargs)
            config = PositionalEncodingConfig(**config_dict)
        
        self.config = config
        self._validate_config()
        
        # Initialize encoding based on type
        self.encoding_type = config.encoding_type
        
        if self.encoding_type == EncodingType.FREQUENCY:
            self._init_frequency_encoding()
        elif self.encoding_type == EncodingType.HASH:
            self._init_hash_encoding()
        elif self.encoding_type == EncodingType.SPHERICAL_HARMONICS:
            self._init_spherical_harmonics_encoding()
        elif self.encoding_type == EncodingType.SINE:
            self._init_sine_encoding()
        elif self.encoding_type == EncodingType.GAUSSIAN:
            self._init_gaussian_encoding()
        elif self.encoding_type == EncodingType.WAVELET:
            self._init_wavelet_encoding()
        elif self.encoding_type == EncodingType.LEARNED:
            self._init_learned_encoding()
        elif self.encoding_type == EncodingType.IDENTITY:
            self._init_identity_encoding()
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Initialize activation if specified
        self.activation = None
        if config.activation is not None:
            self.activation = self._get_activation(config.activation)
    
    def _validate_config(self) -> None:
        """Validate configuration."""
        config = self.config
        
        if config.num_frequencies < 0:
            raise ValueError(f"num_frequencies must be non-negative, got {config.num_frequencies}")
        
        if config.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {config.input_dim}")
    
    def _init_frequency_encoding(self) -> None:
        """Initialize frequency (Fourier feature) encoding."""
        config = self.config
        
        # Precompute frequencies
        self.frequencies = nn.Parameter(
            config.frequencies,
            requires_grad=False
        )
        
        # Store metadata
        self.include_identity = config.include_identity
        self.input_dim = config.input_dim
        
        # Output dimension is computed by config
        self.output_dim = config.output_dim
        
    def _init_hash_encoding(self) -> None:
        """Initialize hash encoding (Instant NGP)."""
        config = self.config
        
        # Parameters for hash encoding
        self.hash_table_size = config.hash_table_size
        self.hash_table_levels = config.hash_table_levels
        self.hash_table_feature_dim = config.hash_table_feature_dim
        self.per_level_scale = config.per_level_scale
        self.input_dim = config.input_dim
        
        # Output dimension
        self.output_dim = config.output_dim
        
        # Create hash tables for each level
        self.hash_tables = nn.ModuleList()
        for i in range(self.hash_table_levels):
            # Each level has its own hash table
            hash_table = nn.Embedding(
                self.hash_table_size,
                self.hash_table_feature_dim
            )
            
            # Initialize with small random values
            nn.init.uniform_(
                hash_table.weight,
                -config.hash_table_init_scale,
                config.hash_table_init_scale
            )
            
            self.hash_tables.append(hash_table)
        
        # Prime numbers for hash function
        self.primes = torch.tensor([
            1, 2654435761, 805459861, 3674653429, 2097192037, 
            1434869437, 2165219737, 1100000993
        ], dtype=torch.int64)
        
    def _init_spherical_harmonics_encoding(self) -> None:
        """Initialize spherical harmonics encoding."""
        config = self.config
        
        # Spherical harmonics are only defined for 3D unit vectors
        assert config.input_dim == 3, \
            f"Spherical harmonics require 3D input, got {config.input_dim}D"
        
        self.sh_degree = config.sh_degree
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
        # Precompute factorial values for normalization
        self._precompute_factorials()
        
    def _precompute_factorials(self) -> None:
        """Precompute factorial values for spherical harmonics normalization."""
        max_fact = 2 * self.sh_degree + 1
        fact = torch.ones(max_fact + 1, dtype=torch.float32)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i
        
        self.factorial = nn.Parameter(fact, requires_grad=False)
    
    def _init_sine_encoding(self) -> None:
        """Initialize sine encoding (SIREN)."""
        config = self.config
        
        # Similar to frequency encoding but with sine activation
        self.frequencies = nn.Parameter(
            config.frequencies,
            requires_grad=False
        )
        
        self.include_identity = config.include_identity
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.sine_omega = config.sine_omega
        
    def _init_gaussian_encoding(self) -> None:
        """Initialize Gaussian encoding."""
        config = self.config
        
        # Create Gaussian centers and scales
        self.gaussian_bands = config.gaussian_bands
        self.gaussian_scale = config.gaussian_scale
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
        # Create trainable Gaussian parameters
        self.centers = nn.Parameter(
            torch.randn(self.input_dim, self.gaussian_bands) * 0.1
        )
        self.scales = nn.Parameter(
            torch.ones(self.input_dim, self.gaussian_bands) * self.gaussian_scale
        )
        
    def _init_wavelet_encoding(self) -> None:
        """Initialize wavelet encoding."""
        config = self.config
        
        self.wavelet_levels = config.wavelet_levels
        self.wavelet_type = config.wavelet_type
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
        # Precompute wavelet filters
        self._init_wavelet_filters()
        
    def _init_wavelet_filters(self) -> None:
        """Initialize wavelet filters."""
        if self.wavelet_type == "haar":
            # Haar wavelet filters
            self.scaling_filter = torch.tensor([1.0, 1.0]) / math.sqrt(2)
            self.wavelet_filter = torch.tensor([1.0, -1.0]) / math.sqrt(2)
        elif self.wavelet_type == "db2":
            # Daubechies 2 wavelet filters
            self.scaling_filter = torch.tensor([
                0.4829629131445341, 0.8365163037378079,
                0.2241438680420134, -0.1294095225512604
            ])
            self.wavelet_filter = torch.tensor([
                -0.1294095225512604, -0.2241438680420134,
                0.8365163037378079, -0.4829629131445341
            ])
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
        
        # Register as buffers
        self.register_buffer('scaling_filter_buffer', self.scaling_filter)
        self.register_buffer('wavelet_filter_buffer', self.wavelet_filter)
        
    def _init_learned_encoding(self) -> None:
        """Initialize learned encoding."""
        config = self.config
        
        self.learned_features = config.learned_features
        self.learned_resolution = config.learned_resolution
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
        # Create learned feature grid
        if self.input_dim == 1:
            self.feature_grid = nn.Parameter(
                torch.randn(1, self.learned_features, self.learned_resolution) * 0.01
            )
        elif self.input_dim == 2:
            self.feature_grid = nn.Parameter(
                torch.randn(1, self.learned_features, 
                           self.learned_resolution, self.learned_resolution) * 0.01
            )
        elif self.input_dim == 3:
            self.feature_grid = nn.Parameter(
                torch.randn(1, self.learned_features,
                           self.learned_resolution, self.learned_resolution,
                           self.learned_resolution) * 0.01
            )
        else:
            raise ValueError(f"Learned encoding only supports 1D, 2D, or 3D input, got {self.input_dim}D")
        
    def _init_identity_encoding(self) -> None:
        """Initialize identity encoding (no transformation)."""
        config = self.config
        
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
    def _get_activation(self, activation_name: str) -> Callable:
        """Get activation function by name."""
        if activation_name == "sin":
            return torch.sin
        elif activation_name == "cos":
            return torch.cos
        elif activation_name == "relu":
            return F.relu
        elif activation_name == "tanh":
            return torch.tanh
        elif activation_name == "sigmoid":
            return torch.sigmoid
        elif activation_name == "exp":
            return torch.exp
        elif activation_name == "softplus":
            return F.softplus
        else:
            raise ValueError(f"Unknown activation: {activation_name}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Encoded tensor of shape [..., output_dim]
        """
        # Store original shape
        original_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        
        # Apply encoding based on type
        if self.encoding_type == EncodingType.FREQUENCY:
            encoded = self._frequency_encode(x_flat)
        elif self.encoding_type == EncodingType.HASH:
            encoded = self._hash_encode(x_flat)
        elif self.encoding_type == EncodingType.SPHERICAL_HARMONICS:
            encoded = self._spherical_harmonics_encode(x_flat)
        elif self.encoding_type == EncodingType.SINE:
            encoded = self._sine_encode(x_flat)
        elif self.encoding_type == EncodingType.GAUSSIAN:
            encoded = self._gaussian_encode(x_flat)
        elif self.encoding_type == EncodingType.WAVELET:
            encoded = self._wavelet_encode(x_flat)
        elif self.encoding_type == EncodingType.LEARNED:
            encoded = self._learned_encode(x_flat)
        elif self.encoding_type == EncodingType.IDENTITY:
            encoded = x_flat
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
        
        # Apply activation if specified
        if self.activation is not None and self.config.activation_after_encoding:
            encoded = self.activation(encoded)
        
        # Reshape back to original shape (except last dimension)
        output_shape = original_shape[:-1] + (self.output_dim,)
        encoded = encoded.reshape(output_shape)
        
        return encoded
    
    def _frequency_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency (Fourier feature) encoding.
        
        Args:
            x: Input tensor [N, D]
            
        Returns:
            Encoded tensor [N, output_dim]
        """
        encoded = []
        
        # Include identity (raw coordinates) if requested
        if self.include_identity:
            encoded.append(x)
        
        # Apply sine and cosine to each frequency
        for freq in self.frequencies:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        # Concatenate all features
        encoded = torch.cat(encoded, dim=-1)
        
        return encoded
    
    def _hash_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hash encoding (Instant NGP).
        
        Args:
            x: Input tensor [N, D] (assumed to be normalized to [0, 1])
            
        Returns:
            Encoded tensor [N, output_dim]
        """
        # Ensure input is in [0, 1]
        x = torch.clamp(x, 0, 1)
        
        encoded_features = []
        
        for level in range(self.hash_table_levels):
            # Scale for this level
            scale = self.per_level_scale ** level
            scaled_x = x * scale
            
            # Get integer and fractional parts
            xi = torch.floor(scaled_x).long()
            xf = scaled_x - xi
            
            # For each dimension, we need to interpolate between 2^D corners
            # For 3D input, we have 8 corners
            num_corners = 2 ** self.input_dim
            corner_features = []
            
            # Generate all corner indices
            for corner in range(num_corners):
                # Compute corner offset
                offset = torch.zeros_like(xi)
                for d in range(self.input_dim):
                    if (corner >> d) & 1:
                        offset[:, d] = 1
                
                # Compute corner index
                corner_idx = xi + offset
                
                # Hash the index to get table entry
                hash_idx = self._hash_function(corner_idx, level)
                
                # Look up feature from hash table
                corner_feature = self.hash_tables[level](hash_idx)
                corner_features.append(corner_feature.unsqueeze(1))
            
            # Stack corner features
            corner_features = torch.cat(corner_features, dim=1)  # [N, 8, F]
            
            # Compute interpolation weights (trilinear)
            weights = torch.ones(x.shape[0], num_corners, device=x.device)
            for d in range(self.input_dim):
                w = xf[:, d].unsqueeze(1)
                for corner in range(num_corners):
                    if (corner >> d) & 1:
                        weights[:, corner] *= w.squeeze()
                    else:
                        weights[:, corner] *= (1 - w.squeeze())
            
            # Weighted sum of corner features
            level_features = torch.sum(
                corner_features * weights.unsqueeze(-1), 
                dim=1
            )  # [N, F]
            
            encoded_features.append(level_features)
        
        # Concatenate features from all levels
        encoded = torch.cat(encoded_features, dim=-1)
        
        return encoded
    
    def _hash_function(self, indices: torch.Tensor, level: int) -> torch.Tensor:
        """
        Hash function for mapping indices to hash table entries.
        
        Args:
            indices: Integer indices [N, D]
            level: Hash table level
            
        Returns:
            Hash indices [N]
        """
        # Simple xor-based hash function
        hash_val = torch.zeros(indices.shape[0], dtype=torch.int64, device=indices.device)
        
        for d in range(self.input_dim):
            hash_val ^= indices[:, d].long() * self.primes[d]
        
        # Add level-specific offset
        hash_val ^= self.primes[self.input_dim] * level
        
        # Modulo hash table size
        hash_idx = hash_val % self.hash_table_size
        
        return hash_idx
    
    def _spherical_harmonics_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spherical harmonics encoding to unit vectors.
        
        Args:
            x: Input tensor [N, 3] (assumed to be normalized)
            
        Returns:
            Encoded tensor [N, output_dim]
        """
        # Ensure input is normalized
        x = F.normalize(x, dim=-1)
        
        # Convert to spherical coordinates
        theta = torch.acos(torch.clamp(x[:, 2], -1.0, 1.0))  # polar angle [0, π]
        phi = torch.atan2(x[:, 1], x[:, 0])  # azimuthal angle [0, 2π]
        
        # Compute spherical harmonics up to degree L
        sh_coeffs = []
        
        for l in range(self.sh_degree + 1):
            for m in range(-l, l + 1):
                # Compute associated Legendre polynomial
                P_lm = self._associated_legendre(l, m, torch.cos(theta))
                
                # Compute normalization factor
                norm = self._spherical_harmonics_normalization(l, m)
                
                # Compute spherical harmonic
                if m == 0:
                    Y_lm = norm * P_lm
                elif m > 0:
                    Y_lm = math.sqrt(2) * norm * P_lm * torch.cos(m * phi)
                else:  # m < 0
                    m_abs = abs(m)
                    Y_lm = math.sqrt(2) * norm * P_lm * torch.sin(m_abs * phi)
                
                sh_coeffs.append(Y_lm.unsqueeze(-1))
        
        # Concatenate all coefficients
        encoded = torch.cat(sh_coeffs, dim=-1)
        
        return encoded
    
    def _associated_legendre(self, l: int, m: int, x: torch.Tensor) -> torch.Tensor:
        """
        Compute associated Legendre polynomial P_l^m(x).
        
        Args:
            l: Degree
            m: Order
            x: Input tensor
            
        Returns:
            P_l^m(x)
        """
        # Use recursive computation
        m_abs = abs(m)
        
        # Base cases
        if l == m_abs:
            # P_m^m(x) = (-1)^m * (2m-1)!! * (1-x^2)^{m/2}
            factor = 1.0
            for i in range(1, 2*m_abs, 2):
                factor *= i
            
            if m_abs % 2 == 1:
                factor = -factor
            
            return factor * (1 - x**2) ** (m_abs / 2)
        
        elif l == m_abs + 1:
            # P_{m+1}^m(x) = x * (2m+1) * P_m^m(x)
            P_mm = self._associated_legendre(m_abs, m_abs, x)
            return x * (2 * m_abs + 1) * P_mm
        
        else:
            # Recurrence relation for l > m+1
            # P_l^m(x) = ((2l-1)*x*P_{l-1}^m(x) - (l+m-1)*P_{l-2}^m(x)) / (l-m)
            P_lm_1 = self._associated_legendre(l-1, m_abs, x)
            P_lm_2 = self._associated_legendre(l-2, m_abs, x)
            
            return ((2*l-1) * x * P_lm_1 - (l+m_abs-1) * P_lm_2) / (l-m_abs)
    
    def _spherical_harmonics_normalization(self, l: int, m: int) -> float:
        """
        Compute spherical harmonics normalization factor.
        
        Args:
            l: Degree
            m: Order
            
        Returns:
            Normalization factor
        """
        m_abs = abs(m)
        norm = math.sqrt(
            (2*l + 1) * self.factorial[l - m_abs] / 
            (4 * math.pi * self.factorial[l + m_abs])
        )
        
        return norm
    
    def _sine_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sine encoding (SIREN).
        
        Args:
            x: Input tensor [N, D]
            
        Returns:
            Encoded tensor [N, output_dim]
        """
        encoded = []
        
        # Include identity if requested
        if self.include_identity:
            encoded.append(x)
        
        # Apply sine activation with frequency scaling
        for freq in self.frequencies:
            encoded.append(torch.sin(self.sine_omega * freq * x))
        
        # Also include cosine for completeness
        for freq in self.frequencies:
            encoded.append(torch.cos(self.sine_omega * freq * x))
        
        # Concatenate all features
        encoded = torch.cat(encoded, dim=-1)
        
        return encoded
    
    def _gaussian_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian encoding.
        
        Args:
            x: Input tensor [N, D]
            
        Returns:
            Encoded tensor [N, output_dim]
        """
        # Reshape for broadcasting
        x_expanded = x.unsqueeze(-1)  # [N, D, 1]
        centers = self.centers.unsqueeze(0)  # [1, D, B]
        scales = self.scales.unsqueeze(0)    # [1, D, B]
        
        # Compute Gaussian functions
        gaussian = torch.exp(-0.5 * ((x_expanded - centers) / scales) ** 2)
        
        # Flatten across dimensions
        encoded = gaussian.reshape(x.shape[0], -1)
        
        return encoded
    
    def _wavelet_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet encoding.
        
        Args:
            x: Input tensor [N, D]
            
        Returns:
            Encoded tensor [N, output_dim]
        """
        # This is a simplified wavelet encoding
        # For each dimension, apply wavelet transform at multiple scales
        
        encoded_features = []
        
        for d in range(self.input_dim):
            x_d = x[:, d]
            
            # Apply wavelet decomposition at multiple levels
            for level in range(self.wavelet_levels):
                # Scale the input
                scaled_x = x_d * (2 ** level)
                
                # Apply wavelet and scaling functions
                wavelet_coeff = torch.sin(2 * math.pi * scaled_x) * torch.exp(-scaled_x**2)
                scaling_coeff = torch.cos(2 * math.pi * scaled_x) * torch.exp(-scaled_x**2)
                
                encoded_features.append(wavelet_coeff.unsqueeze(-1))
                encoded_features.append(scaling_coeff.unsqueeze(-1))
        
        # Concatenate all features
        encoded = torch.cat(encoded_features, dim=-1)
        
        return encoded
    
    def _learned_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learned encoding.
        
        Args:
            x: Input tensor [N, D] (assumed to be normalized to [0, 1])
            
        Returns:
            Encoded tensor [N, output_dim]
        """
        # Ensure input is in [0, 1]
        x = torch.clamp(x, 0, 1)
        
        # Scale to grid coordinates
        grid_coords = x * (self.learned_resolution - 1)
        
        # Reshape for grid sample
        if self.input_dim == 1:
            # 1D grid
            grid_coords = grid_coords.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 1]
            grid_coords = torch.cat([grid_coords, torch.zeros_like(grid_coords)], dim=-1)
            
            # Sample from feature grid
            encoded = F.grid_sample(
                self.feature_grid,
                grid_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(-1).squeeze(-1).transpose(1, 0)  # [N, F]
            
        elif self.input_dim == 2:
            # 2D grid
            grid_coords = grid_coords.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 2]
            grid_coords = grid_coords * 2 - 1  # Convert to [-1, 1] for grid_sample
            
            # Sample from feature grid
            encoded = F.grid_sample(
                self.feature_grid,
                grid_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(-1).squeeze(-1).transpose(1, 0)  # [N, F]
            
        elif self.input_dim == 3:
            # 3D grid
            grid_coords = grid_coords.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 1, 3]
            grid_coords = grid_coords * 2 - 1  # Convert to [-1, 1] for grid_sample
            
            # Sample from feature grid
            encoded = F.grid_sample(
                self.feature_grid,
                grid_coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            ).squeeze(-1).squeeze(-1).squeeze(-1).transpose(1, 0)  # [N, F]
        
        return encoded
    
    def encode_with_frequencies(
        self,
        x: torch.Tensor,
        frequencies: torch.Tensor,
        include_identity: bool = True
    ) -> torch.Tensor:
        """
        Encode with custom frequencies.
        
        Args:
            x: Input tensor [..., D]
            frequencies: Frequency bands [F]
            include_identity: Whether to include raw coordinates
            
        Returns:
            Encoded tensor [..., D * (2*F + (1 if include_identity else 0))]
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        
        encoded = []
        
        if include_identity:
            encoded.append(x_flat)
        
        for freq in frequencies:
            encoded.append(torch.sin(freq * x_flat))
            encoded.append(torch.cos(freq * x_flat))
        
        encoded = torch.cat(encoded, dim=-1)
        output_dim = encoded.shape[-1]
        
        # Reshape back
        output_shape = original_shape[:-1] + (output_dim,)
        encoded = encoded.reshape(output_shape)
        
        return encoded
    
    def compute_fourier_features(
        self,
        x: torch.Tensor,
        matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Fourier features using a learned/projected matrix.
        
        Args:
            x: Input tensor [..., D]
            matrix: Projection matrix [D, F]
            
        Returns:
            Fourier features [..., 2*F]
        """
        # Project input
        projected = x @ matrix  # [..., F]
        
        # Apply sine and cosine
        sin_features = torch.sin(2 * math.pi * projected)
        cos_features = torch.cos(2 * math.pi * projected)
        
        # Concatenate
        features = torch.cat([sin_features, cos_features], dim=-1)
        
        return features
    
    def get_encoding_matrix(self) -> Optional[torch.Tensor]:
        """
        Get encoding matrix if applicable.
        
        Returns:
            Encoding matrix or None
        """
        if self.encoding_type == EncodingType.FREQUENCY:
            # For frequency encoding, return frequency bands
            return self.frequencies
        elif self.encoding_type == EncodingType.HASH:
            # For hash encoding, return hash tables
            return self.hash_tables
        elif self.encoding_type in [EncodingType.SPHERICAL_HARMONICS, 
                                    EncodingType.SINE, 
                                    EncodingType.GAUSSIAN,
                                    EncodingType.WAVELET,
                                    EncodingType.LEARNED,
                                    EncodingType.IDENTITY]:
            # No single matrix for these encodings
            return None
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def get_output_dim(self) -> int:
        """Get output dimension of encoding."""
        return self.output_dim
    
    def get_input_dim(self) -> int:
        """Get input dimension of encoding."""
        return self.input_dim
    
    def normalize_inputs(
        self,
        x: torch.Tensor,
        bounds: Optional[List[List[float]]] = None
    ) -> torch.Tensor:
        """
        Normalize inputs to [0, 1] range.
        
        Args:
            x: Input tensor [..., D]
            bounds: Optional bounds for each dimension [[min1, max1], ...]
            
        Returns:
            Normalized tensor [..., D]
        """
        if bounds is None:
            # Use bounds from config if available
            bounds = self.config.input_bounds
        
        if bounds is None:
            # Auto-normalize based on input range
            min_val = x.min(dim=0)[0]
            max_val = x.max(dim=0)[0]
            x_norm = (x - min_val) / (max_val - min_val + 1e-8)
        else:
            # Use provided bounds
            bounds_tensor = torch.tensor(bounds, device=x.device, dtype=x.dtype)
            min_val = bounds_tensor[:, 0]
            max_val = bounds_tensor[:, 1]
            x_norm = (x - min_val) / (max_val - min_val + 1e-8)
        
        return x_norm


# ============================================================================
# SPECIALIZED ENCODINGS
# ============================================================================

class HashEncoding(nn.Module):
    """Hash encoding implementation (Instant NGP)."""
    
    def __init__(
        self,
        num_levels: int = 16,
        hash_table_size: int = 2**19,
        feature_dim: int = 2,
        per_level_scale: float = 1.5,
        bounds: Optional[List[List[float]]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.hash_table_size = hash_table_size
        self.feature_dim = feature_dim
        self.per_level_scale = per_level_scale
        
        # Input bounds for normalization
        self.bounds = bounds
        
        # Create hash tables
        self.hash_tables = nn.ModuleList()
        for i in range(num_levels):
            hash_table = nn.Embedding(hash_table_size, feature_dim)
            nn.init.uniform_(hash_table.weight, -1e-4, 1e-4)
            self.hash_tables.append(hash_table)
        
        # Prime numbers for hash function
        self.primes = torch.tensor([
            1, 2654435761, 805459861, 3674653429, 2097192037,
            1434869437, 2165219737, 1100000993
        ], dtype=torch.int64, device=device)
        
        # Output dimension
        self.output_dim = num_levels * feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hash encoding."""
        # Normalize to [0, 1]
        if self.bounds is not None:
            bounds_tensor = torch.tensor(self.bounds, device=x.device, dtype=x.dtype)
            min_val = bounds_tensor[:, 0]
            max_val = bounds_tensor[:, 1]
            x = (x - min_val) / (max_val - min_val + 1e-8)
        
        x = torch.clamp(x, 0, 1)
        
        encoded_features = []
        
        for level in range(self.num_levels):
            scale = self.per_level_scale ** level
            scaled_x = x * scale
            
            # Get integer and fractional parts
            xi = torch.floor(scaled_x).long()
            xf = scaled_x - xi
            
            # For 3D input, interpolate between 8 corners
            num_corners = 8
            corner_features = []
            
            for corner in range(num_corners):
                offset = torch.zeros_like(xi)
                for d in range(3):
                    if (corner >> d) & 1:
                        offset[:, d] = 1
                
                corner_idx = xi + offset
                hash_idx = self._hash_function(corner_idx, level)
                corner_feature = self.hash_tables[level](hash_idx)
                corner_features.append(corner_feature.unsqueeze(1))
            
            corner_features = torch.cat(corner_features, dim=1)
            
            # Trilinear interpolation weights
            weights = torch.ones(x.shape[0], num_corners, device=x.device)
            for d in range(3):
                w = xf[:, d].unsqueeze(1)
                for corner in range(num_corners):
                    if (corner >> d) & 1:
                        weights[:, corner] *= w.squeeze()
                    else:
                        weights[:, corner] *= (1 - w.squeeze())
            
            level_features = torch.sum(corner_features * weights.unsqueeze(-1), dim=1)
            encoded_features.append(level_features)
        
        encoded = torch.cat(encoded_features, dim=-1)
        return encoded
    
    def _hash_function(self, indices: torch.Tensor, level: int) -> torch.Tensor:
        """Hash indices to table entries."""
        hash_val = torch.zeros(indices.shape[0], dtype=torch.int64, device=indices.device)
        
        for d in range(3):
            hash_val ^= indices[:, d].long() * self.primes[d]
        
        hash_val ^= self.primes[3] * level
        hash_idx = hash_val % self.hash_table_size
        
        return hash_idx


class SphericalHarmonicsEncoding(nn.Module):
    """Spherical harmonics encoding for directional data."""
    
    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.output_dim = (degree + 1) ** 2
        
        # Precompute factorial table
        max_fact = 2 * degree + 1
        fact = torch.ones(max_fact + 1, dtype=torch.float32)
        for i in range(1, max_fact + 1):
            fact[i] = fact[i-1] * i
        
        self.register_buffer('factorial', fact)
    
    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonics for directions."""
        # Normalize directions
        directions = F.normalize(directions, dim=-1)
        
        # Convert to spherical coordinates
        theta = torch.acos(torch.clamp(directions[:, 2], -1.0, 1.0))
        phi = torch.atan2(directions[:, 1], directions[:, 0])
        
        sh_coeffs = []
        
        for l in range(self.degree + 1):
            for m in range(-l, l + 1):
                m_abs = abs(m)
                
                # Associated Legendre polynomial
                x = torch.cos(theta)
                P_lm = self._associated_legendre(l, m_abs, x)
                
                # Normalization
                norm = torch.sqrt(
                    (2*l + 1) * self.factorial[l - m_abs] / 
                    (4 * math.pi * self.factorial[l + m_abs])
                )
                
                # Spherical harmonic
                if m == 0:
                    Y_lm = norm * P_lm
                elif m > 0:
                    Y_lm = math.sqrt(2) * norm * P_lm * torch.cos(m * phi)
                else:
                    Y_lm = math.sqrt(2) * norm * P_lm * torch.sin(m_abs * phi)
                
                sh_coeffs.append(Y_lm.unsqueeze(-1))
        
        encoded = torch.cat(sh_coeffs, dim=-1)
        return encoded
    
    def _associated_legendre(self, l: int, m: int, x: torch.Tensor) -> torch.Tensor:
        """Compute associated Legendre polynomial recursively."""
        if l == m:
            factor = 1.0
            for i in range(1, 2*m, 2):
                factor *= i
            
            if m % 2 == 1:
                factor = -factor
            
            return factor * (1 - x**2) ** (m / 2)
        elif l == m + 1:
            P_mm = self._associated_legendre(m, m, x)
            return x * (2*m + 1) * P_mm
        else:
            P_lm_1 = self._associated_legendre(l-1, m, x)
            P_lm_2 = self._associated_legendre(l-2, m, x)
            return ((2*l-1) * x * P_lm_1 - (l+m-1) * P_lm_2) / (l-m)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_positional_encoding(
    config: Optional[PositionalEncodingConfig] = None,
    **kwargs
) -> PositionalEncoding:
    """
    Create a positional encoding (convenience function).
    
    Args:
        config: Positional encoding configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Positional encoding module
    """
    return PositionalEncoding(config, **kwargs)


def frequency_encode(
    x: torch.Tensor,
    num_frequencies: int = 10,
    include_identity: bool = True,
    log_sampling: bool = True,
    frequency_base: float = 2.0,
    frequency_scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply frequency encoding (convenience function).
    
    Args:
        x: Input tensor [..., D]
        num_frequencies: Number of frequency bands
        include_identity: Whether to include raw coordinates
        log_sampling: Whether to use log sampling of frequencies
        frequency_base: Base for log sampling
        frequency_scale: Frequency scaling factor
        
    Returns:
        Encoded tensor
    """
    config = PositionalEncodingConfig(
        encoding_type=EncodingType.FREQUENCY,
        input_dim=x.shape[-1],
        num_frequencies=num_frequencies,
        include_identity=include_identity,
        log_sampling=log_sampling,
        frequency_base=frequency_base,
        frequency_scale=frequency_scale,
    )
    
    encoder = PositionalEncoding(config)
    return encoder(x)


def get_positional_encoding_output_dim(
    encoding_type: Union[str, EncodingType],
    input_dim: int,
    **kwargs
) -> int:
    """
    Get output dimension for positional encoding.
    
    Args:
        encoding_type: Type of encoding
        input_dim: Input dimension
        **kwargs: Encoding parameters
        
    Returns:
        Output dimension
    """
    if isinstance(encoding_type, str):
        encoding_type = EncodingType(encoding_type.lower())
    
    config_dict = {'input_dim': input_dim}
    config_dict.update(kwargs)
    
    config = PositionalEncodingConfig(encoding_type=encoding_type, **config_dict)
    return config.output_dim