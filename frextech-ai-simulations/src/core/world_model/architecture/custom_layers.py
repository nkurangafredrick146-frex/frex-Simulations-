"""
Custom neural network layers for 3D world models
Specialized layers for 3D scene representation and generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
import numpy as np
from einops import rearrange, repeat

class PositionalEncoding3D(nn.Module):
    """3D positional encoding for neural fields"""
    
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.num_output_channels = (3 * num_frequencies * 2 + (3 if include_input else 0))
        
        # Create frequency bands
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates of shape [..., 3] (x, y, z)
        Returns:
            Positional encoding of shape [..., num_output_channels]
        """
        shape = x.shape
        x = x.view(-1, 3)
        
        # Apply positional encoding
        encoded = []
        
        if self.include_input:
            encoded.append(x)
            
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq))
            encoded.append(torch.cos(x * freq))
            
        encoded = torch.cat(encoded, dim=-1)
        return encoded.view(*shape[:-1], -1)

class FourierFeature3D(nn.Module):
    """Random Fourier features for 3D coordinates"""
    
    def __init__(self, num_features: int = 256, sigma: float = 10.0, learnable: bool = False):
        super().__init__()
        self.num_features = num_features
        
        # Random projection matrix
        if learnable:
            self.B = nn.Parameter(torch.randn(3, num_features) * sigma)
        else:
            self.register_buffer('B', torch.randn(3, num_features) * sigma)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates of shape [..., 3]
        Returns:
            Fourier features of shape [..., 2 * num_features]
        """
        shape = x.shape
        x = x.view(-1, 3)
        
        # Project to Fourier space
        proj = 2 * math.pi * x @ self.B
        features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
        return features.view(*shape[:-1], -1)

class AdaptiveInstanceNorm3D(nn.Module):
    """Adaptive Instance Normalization for 3D features"""
    
    def __init__(self, num_features: int, style_dim: int = 512):
        super().__init__()
        self.num_features = num_features
        
        # Style mapping network
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)
        
        # Learnable parameters for base style
        self.register_parameter('base_scale', nn.Parameter(torch.ones(1, num_features, 1, 1, 1)))
        self.register_parameter('base_shift', nn.Parameter(torch.zeros(1, num_features, 1, 1, 1)))
        
    def forward(self, x: torch.Tensor, style: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            style: Style vector of shape [B, style_dim]
        Returns:
            Stylized tensor
        """
        if style is not None:
            # Instance normalization
            mean = x.mean(dim=[2, 3, 4], keepdim=True)
            var = x.var(dim=[2, 3, 4], keepdim=True)
            x_normalized = (x - mean) / torch.sqrt(var + 1e-8)
            
            # Adaptive scaling and shifting
            scale = self.style_scale(style)[:, :, None, None, None]
            shift = self.style_shift(style)[:, :, None, None, None]
            
            return x_normalized * (self.base_scale + scale) + (self.base_shift + shift)
        else:
            # Fallback to regular normalization
            return F.instance_norm(x)

class SphericalHarmonicsEncoding(nn.Module):
    """Spherical harmonics encoding for directional information"""
    
    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
        self.num_coeffs = (degree + 1) ** 2
        
        # Precompute spherical harmonics coefficients
        self.register_buffer('coeffs', self._compute_coefficients())
        
    def _compute_coefficients(self) -> torch.Tensor:
        """Precompute spherical harmonics coefficients"""
        coeffs = []
        l_max = self.degree
        
        # Generate all (l, m) pairs
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                coeffs.append((l, m))
                
        return torch.tensor(coeffs, dtype=torch.float32)
    
    def forward(self, directions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            directions: Normalized direction vectors of shape [..., 3]
        Returns:
            Spherical harmonics encoding of shape [..., num_coeffs]
        """
        shape = directions.shape
        directions = directions.view(-1, 3)
        
        # Convert to spherical coordinates
        x, y, z = directions.unbind(-1)
        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)
        
        theta = torch.acos(z / r)  # Polar angle
        phi = torch.atan2(y, x)    # Azimuthal angle
        
        # Compute spherical harmonics
        harmonics = []
        coeffs = self.coeffs.to(directions.device)
        
        for l, m in coeffs:
            # Associated Legendre polynomials
            if l == 0:
                plm = torch.ones_like(theta)
            elif l == 1:
                if m == -1:
                    plm = torch.sin(theta)
                elif m == 0:
                    plm = torch.cos(theta)
                elif m == 1:
                    plm = -torch.sin(theta)
            else:
                # For higher degrees, use recursion formula
                plm = self._legendre_poly(l, m, torch.cos(theta))
            
            # Normalization constant
            norm = math.sqrt(((2 * l + 1) * math.factorial(l - abs(m))) / 
                            (4 * math.pi * math.factorial(l + abs(m))))
            
            # Spherical harmonic
            if m < 0:
                ylm = math.sqrt(2) * norm * plm * torch.sin(abs(m) * phi)
            elif m == 0:
                ylm = norm * plm
            else:
                ylm = math.sqrt(2) * norm * plm * torch.cos(m * phi)
                
            harmonics.append(ylm.unsqueeze(-1))
            
        harmonics = torch.cat(harmonics, dim=-1)
        return harmonics.view(*shape[:-1], -1)
    
    def _legendre_poly(self, l: int, m: int, x: torch.Tensor) -> torch.Tensor:
        """Compute associated Legendre polynomial P_l^m(x)"""
        # Using recursion relation
        if l == m:
            pmm = (-1)**m * math.factorial(2*m - 1) * (1 - x**2)**(m/2)
        else:
            pmm1 = x * (2*m + 1) * self._legendre_poly(l, m+1, x)
            if m + 2 <= l:
                pmm2 = self._legendre_poly(l, m+2, x)
                pmm = (pmm1 - pmm2) / (m - l)
            else:
                pmm = pmm1
        return pmm

class MultiHeadCrossAttention3D(nn.Module):
    """Multi-head cross attention for 3D features with key-value conditioning"""
    
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5
        
        # Projections
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query tensor of shape [B, N, query_dim]
            context: Context tensor of shape [B, M, context_dim]
            mask: Optional mask of shape [B, N, M]
        Returns:
            Attended tensor of shape [B, N, query_dim]
        """
        B, N, _ = x.shape
        _, M, _ = context.shape
        
        # Project to queries, keys, values
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)
        
        # Attention scores
        attn = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask[:, None, :, :]  # Add head dimension
            attn = attn.masked_fill(mask == 0, -1e9)
            
        # Attention weights
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class AdaptiveConv3D(nn.Module):
    """3D convolution with adaptive kernel weights based on conditioning"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        style_dim: int = 512,
        modulation: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.modulation = modulation
        
        # Base convolution weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        if modulation:
            # Style modulation network
            self.style_scale = nn.Linear(style_dim, in_channels)
            self.style_shift = nn.Linear(style_dim, in_channels)
            
            # Kernel modulation
            self.kernel_modulator = nn.Sequential(
                nn.Linear(style_dim, in_channels * kernel_size**3),
                nn.Unflatten(1, (in_channels, kernel_size, kernel_size, kernel_size))
            )
            
    def forward(self, x: torch.Tensor, style: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            style: Style vector of shape [B, style_dim]
        Returns:
            Convolved tensor
        """
        weight = self.weight
        bias = self.bias
        
        if self.modulation and style is not None:
            # Feature modulation
            scale = self.style_scale(style)[:, :, None, None, None]
            shift = self.style_shift(style)[:, :, None, None, None]
            
            x = x * (1 + scale) + shift
            
            # Kernel modulation
            kernel_mod = self.kernel_modulator(style)
            weight = weight * (1 + kernel_mod[:, None, :, :, :, :])
            
        # Apply convolution
        return F.conv3d(x, weight, bias, padding=self.kernel_size//2)

class MultiResolutionHashEncoding(nn.Module):
    """Multi-resolution hash encoding for instant neural graphics primitives"""
    
    def __init__(
        self,
        num_levels: int = 16,
        min_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        interpolation: str = 'linear'
    ):
        super().__init__()
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.interpolation = interpolation
        
        # Resolution for each level
        b = np.exp((np.log(max_res) - np.log(min_res)) / (num_levels - 1))
        self.resolutions = []
        for lvl in range(num_levels):
            resolution = int(np.floor(min_res * (b ** lvl)))
            self.resolutions.append(resolution)
            
        # Hash tables for each level
        self.hash_tables = nn.ModuleList()
        self.hashmap_size = 2 ** log2_hashmap_size
        
        for res in self.resolutions:
            # Number of vertices in grid
            table_size = min(res ** 3, self.hashmap_size)
            hash_table = nn.Embedding(table_size, features_per_level)
            nn.init.uniform_(hash_table.weight, -1e-4, 1e-4)
            self.hash_tables.append(hash_table)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinates in [0, 1] range, shape [..., 3]
        Returns:
            Hash-encoded features of shape [..., num_levels * features_per_level]
        """
        shape = x.shape
        x = x.view(-1, 3)
        B = x.shape[0]
        
        all_features = []
        
        for lvl, (res, hash_table) in enumerate(zip(self.resolutions, self.hash_tables)):
            # Scale coordinates to grid resolution
            scaled_coords = x * res
            
            # Get integer coordinates and fractional parts
            coords_floor = torch.floor(scaled_coords).long()
            coords_ceil = torch.ceil(scaled_coords).long()
            coords_frac = scaled_coords - coords_floor
            
            # Clamp coordinates
            coords_floor = torch.clamp(coords_floor, 0, res - 1)
            coords_ceil = torch.clamp(coords_ceil, 0, res - 1)
            
            # Get 8 corner vertices
            vertices = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        vert_coords = coords_floor + torch.tensor([dx, dy, dz], device=x.device)
                        vertices.append(vert_coords)
                        
            # Hash function to get indices
            indices = []
            for vert in vertices:
                # Simple spatial hash
                idx = vert[:, 0] * 73856093 ^ vert[:, 1] * 19349663 ^ vert[:, 2] * 83492791
                idx = idx % self.hashmap_size
                indices.append(idx)
                
            # Lookup features
            features = []
            for idx in indices:
                feat = hash_table(idx)
                features.append(feat)
                
            # Interpolate
            if self.interpolation == 'linear':
                # Trilinear interpolation
                f000, f001, f010, f011, f100, f101, f110, f111 = features
                
                # Interpolate along x
                fx00 = f000 * (1 - coords_frac[:, 0:1]) + f100 * coords_frac[:, 0:1]
                fx01 = f001 * (1 - coords_frac[:, 0:1]) + f101 * coords_frac[:, 0:1]
                fx10 = f010 * (1 - coords_frac[:, 0:1]) + f110 * coords_frac[:, 0:1]
                fx11 = f011 * (1 - coords_frac[:, 0:1]) + f111 * coords_frac[:, 0:1]
                
                # Interpolate along y
                fxy0 = fx00 * (1 - coords_frac[:, 1:2]) + fx10 * coords_frac[:, 1:2]
                fxy1 = fx01 * (1 - coords_frac[:, 1:2]) + fx11 * coords_frac[:, 1:2]
                
                # Interpolate along z
                interpolated = fxy0 * (1 - coords_frac[:, 2:3]) + fxy1 * coords_frac[:, 2:3]
                
            else:
                # Nearest neighbor
                interpolated = features[0]  # Use lower corner
                
            all_features.append(interpolated)
            
        # Concatenate all levels
        encoded = torch.cat(all_features, dim=-1)
        return encoded.view(*shape[:-1], -1)

class GeometryAwareConv3D(nn.Module):
    """3D convolution with geometric awareness using local coordinate frames"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_orientations: int = 6,
        use_attention: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        
        # Convolution kernels for different orientations
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
            for _ in range(num_orientations)
        ])
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Orientation prediction network
        self.orientation_predictor = nn.Sequential(
            nn.Conv3d(in_channels, 32, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv3d(32, num_orientations, 1)
        )
        
        # Optional attention for orientation mixing
        if use_attention:
            self.orientation_attention = nn.Sequential(
                nn.Conv3d(in_channels, num_orientations, 1),
                nn.Softmax(dim=1)
            )
        else:
            self.orientation_attention = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]
        Returns:
            Convolved tensor with geometric awareness
        """
        B, C, D, H, W = x.shape
        
        # Predict orientation weights for each position
        orientation_logits = self.orientation_predictor(x)  # [B, num_orientations, D, H, W]
        orientation_weights = F.softmax(orientation_logits, dim=1)
        
        # Apply convolutions with different orientations
        outputs = []
        for i, kernel in enumerate(self.kernels):
            # Apply convolution
            conv_out = F.conv3d(x, kernel, padding=self.kernel_size//2)
            
            # Weight by orientation
            weight = orientation_weights[:, i:i+1, :, :, :]
            weighted_out = conv_out * weight
            outputs.append(weighted_out)
            
        # Sum weighted outputs
        if self.orientation_attention is not None:
            # Additional attention-based mixing
            attention_weights = self.orientation_attention(x)
            out = sum(o * attention_weights[:, i:i+1] for i, o in enumerate(outputs))
        else:
            out = sum(outputs)
            
        # Add bias
        out = out + self.bias[None, :, None, None, None]
        
        return out

class SceneGraphAttention(nn.Module):
    """Attention over scene graph nodes for relational reasoning"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        heads: int = 8,
        dropout: float = 0.1,
        use_edge_features: bool = True
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.heads = heads
        self.use_edge_features = use_edge_features
        self.scale = (node_dim // heads) ** -0.5
        
        # Node projections
        self.to_q = nn.Linear(node_dim, node_dim, bias=False)
        self.to_k = nn.Linear(node_dim, node_dim, bias=False)
        self.to_v = nn.Linear(node_dim, node_dim, bias=False)
        
        # Edge projection (if using edge features)
        if use_edge_features:
            self.to_e = nn.Linear(edge_dim, heads, bias=False)
            
        # Output projection
        self.to_out = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        nodes: torch.Tensor,
        edges: Optional[torch.Tensor] = None,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            nodes: Node features of shape [B, N, node_dim]
            edges: Edge features of shape [B, N, N, edge_dim]
            adj_matrix: Adjacency matrix of shape [B, N, N]
        Returns:
            Updated node features
        """
        B, N, _ = nodes.shape
        
        # Project to queries, keys, values
        q = self.to_q(nodes)
        k = self.to_k(nodes)
        v = self.to_v(nodes)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Attention scores
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # Incorporate edge features
        if self.use_edge_features and edges is not None:
            edge_bias = self.to_e(edges)  # [B, N, N, heads]
            edge_bias = rearrange(edge_bias, 'b i j h -> b h i j')
            attn = attn + edge_bias
            
        # Apply adjacency mask
        if adj_matrix is not None:
            mask = adj_matrix[:, None, :, :]  # Add head dimension
            attn = attn.masked_fill(mask == 0, -1e9)
            
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Reshape back
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class AdaptivePooling3D(nn.Module):
    """Adaptive 3D pooling with learnable pooling regions"""
    
    def __init__(
        self,
        output_size: Tuple[int, int, int],
        channels: int,
        learnable_grid: bool = True
    ):
        super().__init__()
        self.output_size = output_size
        self.channels = channels
        
        if learnable_grid:
            # Learnable sampling grid
            self.sampling_grid = nn.Parameter(
                torch.randn(1, *output_size, 3) * 0.1  # [1, D', H', W', 3]
            )
        else:
            self.sampling_grid = None
            
        # Feature modulation
        self.feature_modulator = nn.Sequential(
            nn.Conv3d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv3d(channels // 2, channels, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]
        Returns:
            Pooled tensor of shape [B, C, D', H', W']
        """
        B, C, D, H, W = x.shape
        D_out, H_out, W_out = self.output_size
        
        if self.sampling_grid is not None:
            # Use learnable sampling grid
            grid = self.sampling_grid.repeat(B, 1, 1, 1, 1)
            
            # Normalize grid to [-1, 1]
            grid = grid * 2 - 1
            
            # Sample using grid
            pooled = F.grid_sample(
                x,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
        else:
            # Adaptive average pooling
            pooled = F.adaptive_avg_pool3d(x, self.output_size)
            
        # Feature modulation
        modulation = self.feature_modulator(pooled)
        pooled = pooled * torch.sigmoid(modulation)
        
        return pooled

class ConditionedNormalization(nn.Module):
    """Normalization layer conditioned on external inputs"""
    
    def __init__(
        self,
        num_features: int,
        condition_dim: int,
        norm_type: str = 'instance'
    ):
        super().__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.norm_type = norm_type
        
        # Condition projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, num_features * 2),
            nn.ReLU(),
            nn.Linear(num_features * 2, num_features * 2)
        )
        
        # Base normalization
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm3d(num_features, affine=False)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm3d(num_features, affine=False)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(num_features, elementwise_affine=False)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
            
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, ...]
            condition: Condition tensor of shape [B, condition_dim]
        Returns:
            Normalized and conditioned tensor
        """
        # Get normalization parameters from condition
        params = self.condition_proj(condition)
        gamma, beta = params.chunk(2, dim=1)
        
        # Apply normalization
        if self.norm_type in ['instance', 'batch']:
            # 3D normalization
            x_norm = self.norm(x)
            
            # Add dimensions for broadcasting
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]
        else:
            # Layer norm
            orig_shape = x.shape
            x = x.view(orig_shape[0], orig_shape[1], -1).transpose(1, 2)
            x_norm = self.norm(x)
            x_norm = x_norm.transpose(1, 2).view(orig_shape)
            
            gamma = gamma[:, :, None, None, None]
            beta = beta[:, :, None, None, None]
            
        # Apply conditioning
        return x_norm * (1 + gamma) + beta

# Utility functions
def init_weights_xavier(m: nn.Module):
    """Xavier initialization for custom layers"""
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def compute_gradient_penalty(real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
    """Compute gradient penalty for Wasserstein GAN"""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1, device=real_data.device)
    
    # Interpolate between real and fake data
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # Compute gradients
    grad_outputs = torch.ones_like(interpolated)
    gradients = torch.autograd.grad(
        outputs=interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def spectral_norm(module: nn.Module, n_power_iterations: int = 1) -> nn.Module:
    """Apply spectral normalization to module"""
    return nn.utils.spectral_norm(module, n_power_iterations=n_power_iterations)