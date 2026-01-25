"""
Advanced Attention Modules for World Model Architecture.

This module contains specialized attention mechanisms for 3D world generation,
including sparse attention, linear attention, and multimodal attention with
efficient implementations for long sequences and 3D data.
"""

import math
from typing import Optional, Tuple, Union, List, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat, einsum

from .custom_layers import RMSNorm, AdaptiveLayerNorm, GatedLinearUnit

class MultiScaleAttention(nn.Module):
    """
Multi-scale attention for capturing both local and global dependencies.

Combines:
- Local window attention for fine details
- Global attention for overall structure
- Dilated attention for intermediate scales
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    window_sizes: List[int] = [8, 16, 32],
    dilations: List[int] = [1, 2, 4],
    dropout: float = 0.0,
    qkv_bias: bool = False,
    use_flash: bool = True,
    norm_type: str = "rms",
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.window_sizes = window_sizes
    self.dilations = dilations
    self.scale = self.head_dim ** -0.5
    
    # Check if flash attention is available
    self.use_flash = use_flash
    if use_flash:
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.has_flash = True
        except ImportError:
            print("Flash attention not available, falling back to standard attention")
            self.has_flash = False
    else:
        self.has_flash = False
    
    # Linear projections for queries, keys, values
    self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
    
    # Output projection
    self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout),
    )
    
    # Normalization
    if norm_type == "rms":
        self.norm = RMSNorm(dim)
    elif norm_type == "adaptive":
        self.norm = AdaptiveLayerNorm(dim)
    else:
        self.norm = nn.LayerNorm(dim)
    
    # Learnable weights for combining different scales
    self.scale_weights = nn.Parameter(torch.ones(len(window_sizes) + len(dilations)))
    
    # Dropout
    self.attn_dropout = nn.Dropout(dropout)
    self.proj_dropout = nn.Dropout(dropout)
    
    self._init_weights()

def _init_weights(self):
    """Initialize attention weights."""
    nn.init.xavier_uniform_(self.to_q.weight)
    nn.init.xavier_uniform_(self.to_k.weight)
    nn.init.xavier_uniform_(self.to_v.weight)
    nn.init.xavier_uniform_(self.to_out[0].weight)
    
    if self.to_q.bias is not None:
        nn.init.zeros_(self.to_q.bias)
        nn.init.zeros_(self.to_k.bias)
        nn.init.zeros_(self.to_v.bias)
    
    nn.init.zeros_(self.to_out[0].bias)

def _local_window_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention within local windows."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Pad sequence to be divisible by window size
    pad_len = (window_size - seq_len % window_size) % window_size
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        padded_seq_len = seq_len + pad_len
    else:
        padded_seq_len = seq_len
    
    # Reshape into windows
    num_windows = padded_seq_len // window_size
    q_windows = q.view(batch_size, num_windows, window_size, num_heads, head_dim)
    k_windows = k.view(batch_size, num_windows, window_size, num_heads, head_dim)
    v_windows = v.view(batch_size, num_windows, window_size, num_heads, head_dim)
    
    # Compute attention within each window
    if self.has_flash and q.dtype in (torch.float16, torch.bfloat16):
        # Use flash attention for windows
        q_windows = q_windows.transpose(2, 3)  # (batch, num_windows, heads, window, dim)
        k_windows = k_windows.transpose(2, 3)
        v_windows = v_windows.transpose(2, 3)
        
        attn_output = self.flash_attn_func(
            q_windows.reshape(batch_size * num_windows, num_heads, window_size, head_dim),
            k_windows.reshape(batch_size * num_windows, num_heads, window_size, head_dim),
            v_windows.reshape(batch_size * num_windows, num_heads, window_size, head_dim),
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        ).reshape(batch_size, num_windows, num_heads, window_size, head_dim)
        
        attn_output = attn_output.transpose(2, 3)  # (batch, num_windows, window, heads, dim)
    else:
        # Standard attention for windows
        q_windows = q_windows * self.scale
        attn_scores = torch.einsum('bwqhd,bwkhd->bwhqk', q_windows, k_windows)
        
        if mask is not None:
            # Apply mask within windows
            mask_windows = mask.view(batch_size, num_windows, window_size)
            mask_scores = mask_windows.unsqueeze(2).unsqueeze(3)  # (batch, num_windows, 1, 1, window)
            attn_scores = attn_scores.masked_fill(mask_scores == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.einsum('bwhqk,bwkhd->bwqhd', attn_weights, v_windows)
    
    # Reshape back
    attn_output = attn_output.reshape(batch_size, padded_seq_len, num_heads, head_dim)
    
    # Remove padding
    if pad_len > 0:
        attn_output = attn_output[:, :seq_len]
    
    return attn_output

def _dilated_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dilation: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention with dilation for longer range dependencies."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Apply dilation
    if dilation > 1:
        # Select every dilation-th element
        q_dilated = q[:, ::dilation]
        k_dilated = k[:, ::dilation]
        v_dilated = v[:, ::dilation]
        seq_len_dilated = q_dilated.shape[1]
        
        if mask is not None:
            mask_dilated = mask[:, ::dilation]
    else:
        q_dilated, k_dilated, v_dilated = q, k, v
        seq_len_dilated = seq_len
        mask_dilated = mask
    
    # Compute attention on dilated sequence
    if self.has_flash and q.dtype in (torch.float16, torch.bfloat16):
        # Use flash attention
        q_dilated = q_dilated.transpose(1, 2)  # (batch, heads, seq_len_dilated, dim)
        k_dilated = k_dilated.transpose(1, 2)
        v_dilated = v_dilated.transpose(1, 2)
        
        attn_output_dilated = self.flash_attn_func(
            q_dilated,
            k_dilated,
            v_dilated,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        ).transpose(1, 2)  # (batch, seq_len_dilated, heads, dim)
    else:
        # Standard attention
        q_dilated = q_dilated * self.scale
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q_dilated, k_dilated)
        
        if mask_dilated is not None:
            mask_scores = mask_dilated.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len_dilated)
            attn_scores = attn_scores.masked_fill(mask_scores == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output_dilated = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v_dilated)
    
    # Interpolate back to original sequence length if needed
    if dilation > 1:
        attn_output = F.interpolate(
            attn_output_dilated.permute(0, 3, 1, 2),  # (batch, dim, seq_len_dilated, heads)
            size=(seq_len, num_heads),
            mode='nearest',
        ).permute(0, 2, 3, 1)  # (batch, seq_len, heads, dim)
    else:
        attn_output = attn_output_dilated
    
    return attn_output

def _global_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute global attention."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    if self.has_flash and q.dtype in (torch.float16, torch.bfloat16):
        # Use flash attention
        q_global = q.transpose(1, 2)  # (batch, heads, seq_len, dim)
        k_global = k.transpose(1, 2)
        v_global = v.transpose(1, 2)
        
        attn_output = self.flash_attn_func(
            q_global,
            k_global,
            v_global,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        ).transpose(1, 2)  # (batch, seq_len, heads, dim)
    else:
        # Standard attention
        q_global = q * self.scale
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q_global, k_global)
        
        if mask is not None:
            mask_scores = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask_scores == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
    
    return attn_output

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_attn: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Forward pass for multi-scale attention.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Optional attention mask
        return_attn: Whether to return attention weights
    
    Returns:
        Output tensor and optionally attention weights
    """
    batch_size, seq_len, _ = x.shape
    
    # Normalize input
    x_norm = self.norm(x)
    
    # Project to queries, keys, values
    q = self.to_q(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = self.to_k(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
    v = self.to_v(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
    
    # Compute attention at different scales
    attn_outputs = []
    attn_weights_list = []
    
    # Local window attention
    for window_size in self.window_sizes:
        output = self._local_window_attention(q, k, v, window_size, mask)
        attn_outputs.append(output)
    
    # Dilated attention
    for dilation in self.dilations:
        output = self._dilated_attention(q, k, v, dilation, mask)
        attn_outputs.append(output)
    
    # Global attention
    global_output = self._global_attention(q, k, v, mask)
    attn_outputs.append(global_output)
    
    # Combine outputs with learned weights
    scale_weights = F.softmax(self.scale_weights, dim=0)
    
    # Ensure we have the right number of weights
    num_scales = len(self.window_sizes) + len(self.dilations) + 1
    if len(scale_weights) < num_scales:
        # Pad with ones if needed
        pad_weights = torch.ones(num_scales - len(scale_weights), device=scale_weights.device)
        scale_weights = torch.cat([scale_weights, pad_weights])
    
    # Apply weights
    combined_output = torch.zeros_like(attn_outputs[0])
    for i, output in enumerate(attn_outputs):
        weight = scale_weights[i] if i < len(scale_weights) else 1.0
        combined_output = combined_output + weight * output
    
    # Reshape and project
    combined_output = combined_output.reshape(batch_size, seq_len, self.dim)
    output = self.to_out(combined_output)
    output = self.proj_dropout(output)
    
    # Residual connection
    output = x + output
    
    if return_attn:
        return output, attn_weights_list
    return output

class Sparse3DAttention(nn.Module):
    """
3D Sparse Attention for volumetric data.

Efficient attention for 3D representations using:
- Axial attention along each dimension
- Windowed attention in 3D space
- Sparse neighborhood attention
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    grid_size: Tuple[int, int, int] = (32, 32, 32),
    window_size: int = 8,
    dropout: float = 0.0,
    qkv_bias: bool = False,
    use_checkpoint: bool = False,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.grid_size = grid_size
    self.window_size = window_size
    self.use_checkpoint = use_checkpoint
    self.scale = self.head_dim ** -0.5
    
    # Linear projections
    self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout),
    )
    
    # Normalization
    self.norm = RMSNorm(dim)
    
    # Learnable relative position bias for windows
    self.relative_position_bias_table = nn.Parameter(
        torch.zeros((2 * window_size - 1) ** 3, num_heads)
    )
    
    # Generate relative position index
    self.register_buffer(
        "relative_position_index",
        self._generate_relative_position_index(window_size),
        persistent=False,
    )
    
    self._init_weights()

def _generate_relative_position_index(self, window_size: int) -> torch.Tensor:
    """Generate relative position index for 3D windows."""
    coords = torch.stack(torch.meshgrid([
        torch.arange(window_size),
        torch.arange(window_size),
        torch.arange(window_size),
    ])).flatten(1)  # (3, window_size^3)
    
    relative_coords = coords[:, :, None] - coords[:, None, :]  # (3, window_size^3, window_size^3)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (window_size^3, window_size^3, 3)
    
    # Shift to start from 0
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 2] += window_size - 1
    
    relative_coords[:, :, 0] *= (2 * window_size - 1) * (2 * window_size - 1)
    relative_coords[:, :, 1] *= (2 * window_size - 1)
    relative_index = relative_coords.sum(-1)  # (window_size^3, window_size^3)
    
    return relative_index

def _init_weights(self):
    """Initialize weights."""
    nn.init.xavier_uniform_(self.to_qkv.weight)
    if self.to_qkv.bias is not None:
        nn.init.zeros_(self.to_qkv.bias)
    
    nn.init.xavier_uniform_(self.to_out[0].weight)
    nn.init.zeros_(self.to_out[0].bias)
    
    nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

def _window_partition_3d(
    self,
    x: torch.Tensor,
    grid_size: Tuple[int, int, int],
    window_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int, int]]:
    """
    Partition 3D tensor into windows.
    
    Args:
        x: Tensor of shape (batch, depth, height, width, dim)
        grid_size: Original grid size (D, H, W)
        window_size: Window size
    
    Returns:
        Windowed tensor and window info
    """
    batch_size, D, H, W, dim = x.shape
    x = x.view(
        batch_size,
        D // window_size, window_size,
        H // window_size, window_size,
        W // window_size, window_size,
        dim,
    )
    
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, window_size, window_size, window_size, dim)
    
    return windows, (batch_size, D // window_size, H // window_size, W // window_size, window_size)

def _window_reverse_3d(
    self,
    windows: torch.Tensor,
    window_info: Tuple[int, int, int, int, int],
    grid_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Reverse window partitioning."""
    batch_size, num_windows_d, num_windows_h, num_windows_w, window_size = window_info
    D, H, W = grid_size
    
    x = windows.view(
        batch_size,
        num_windows_d, num_windows_h, num_windows_w,
        window_size, window_size, window_size,
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(batch_size, D, H, W, -1)
    
    return x

def _axial_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    axis: int,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention along a specific axis."""
    batch_size, D, H, W, num_heads, head_dim = q.shape
    
    # Permute based on axis
    if axis == 0:  # Depth axis
        q = q.permute(0, 2, 3, 1, 4, 5).contiguous()  # (batch, H, W, D, heads, dim)
        k = k.permute(0, 2, 3, 1, 4, 5).contiguous()
        v = v.permute(0, 2, 3, 1, 4, 5).contiguous()
        seq_len = D
    elif axis == 1:  # Height axis
        q = q.permute(0, 1, 3, 2, 4, 5).contiguous()  # (batch, D, W, H, heads, dim)
        k = k.permute(0, 1, 3, 2, 4, 5).contiguous()
        v = v.permute(0, 1, 3, 2, 4, 5).contiguous()
        seq_len = H
    else:  # Width axis
        q = q.permute(0, 1, 2, 3, 4, 5).contiguous()  # (batch, D, H, W, heads, dim)
        k = k.permute(0, 1, 2, 3, 4, 5).contiguous()
        v = v.permute(0, 1, 2, 3, 4, 5).contiguous()
        seq_len = W
    
    # Reshape for attention
    q = q.view(batch_size * H * W * D // seq_len, seq_len, num_heads, head_dim)
    k = k.view(batch_size * H * W * D // seq_len, seq_len, num_heads, head_dim)
    v = v.view(batch_size * H * W * D // seq_len, seq_len, num_heads, head_dim)
    
    # Compute attention
    q = q * self.scale
    attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
    
    if mask is not None:
        mask_reshaped = mask.view(batch_size * H * W * D // seq_len, seq_len)
        mask_scores = mask_reshaped.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(mask_scores == 0, -1e9)
    
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
    
    # Reshape back
    attn_output = attn_output.view(batch_size, D, H, W, num_heads, head_dim)
    
    # Permute back
    if axis == 0:
        attn_output = attn_output.permute(0, 3, 1, 2, 4, 5).contiguous()
    elif axis == 1:
        attn_output = attn_output.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    return attn_output

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for 3D sparse attention.
    
    Args:
        x: Input tensor of shape (batch, depth, height, width, dim)
        mask: Optional 3D mask
    
    Returns:
        Output tensor
    """
    batch_size, D, H, W, dim = x.shape
    
    # Normalize input
    x_norm = self.norm(x)
    
    # Project to queries, keys, values
    qkv = self.to_qkv(x_norm).view(batch_size, D, H, W, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(4)  # Each: (batch, D, H, W, heads, dim)
    
    # Compute axial attention along each dimension
    if self.use_checkpoint:
        # Use gradient checkpointing for memory efficiency
        axial_outputs = []
        for axis in range(3):
            axial_output = checkpoint.checkpoint(
                self._axial_attention,
                q, k, v, axis, mask,
                use_reentrant=False,
            )
            axial_outputs.append(axial_output)
    else:
        axial_outputs = [
            self._axial_attention(q, k, v, 0, mask),  # Depth axis
            self._axial_attention(q, k, v, 1, mask),  # Height axis
            self._axial_attention(q, k, v, 2, mask),  # Width axis
        ]
    
    # Average axial attention outputs
    axial_output = torch.stack(axial_outputs).mean(0)
    
    # Window attention
    windows, window_info = self._window_partition_3d(
        x_norm,
        (D, H, W),
        self.window_size,
    )
    
    window_batch = windows.shape[0]
    windows = windows.view(window_batch, -1, self.window_size**3, dim)
    
    # Project windows to QKV
    window_qkv = self.to_qkv(windows).view(
        window_batch, -1, self.window_size**3, 3, self.num_heads, self.head_dim
    )
    window_q, window_k, window_v = window_qkv.unbind(3)
    
    # Compute window attention
    window_q = window_q * self.scale
    window_attn_scores = torch.einsum('bqhd,bkhd->bhqk', window_q, window_k)
    
    # Add relative position bias
    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(self.window_size**3, self.window_size**3, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    window_attn_scores = window_attn_scores + relative_position_bias.unsqueeze(0)
    
    if mask is not None:
        # Apply mask to windows
        window_mask = mask.view(
            batch_size,
            D // self.window_size, self.window_size,
            H // self.window_size, self.window_size,
            W // self.window_size, self.window_size,
        ).permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        window_mask = window_mask.view(window_batch, -1, self.window_size**3)
        mask_scores = window_mask.unsqueeze(1).unsqueeze(2)
        window_attn_scores = window_attn_scores.masked_fill(mask_scores == 0, -1e9)
    
    window_attn_weights = F.softmax(window_attn_scores, dim=-1)
    window_output = torch.einsum('bhqk,bkhd->bqhd', window_attn_weights, window_v)
    window_output = window_output.view(window_batch, -1, self.window_size**3, dim)
    
    # Reverse window partitioning
    window_output = self._window_reverse_3d(
        window_output.view(window_batch, -1, self.window_size, self.window_size, self.window_size, dim),
        window_info,
        (D, H, W),
    )
    
    # Combine axial and window attention
    output = 0.5 * axial_output.view(batch_size, D, H, W, dim) + 0.5 * window_output
    
    # Project and add residual
    output = self.to_out(output)
    output = x + output
    
    return output

class LinearAttention3D(nn.Module):
    """
3D Linear Attention with O(n) complexity.

Efficient attention for volumetric data using linear attention kernels.
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    kernel: str = "elu",
    dropout: float = 0.0,
    qkv_bias: bool = False,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.kernel = kernel
    self.scale = self.head_dim ** -0.5
    
    # Linear projections
    self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout),
    )
    
    # Normalization
    self.norm = RMSNorm(dim)
    
    # Kernel activation
    if kernel == "elu":
        self.kernel_act = lambda x: F.elu(x) + 1
    elif kernel == "relu":
        self.kernel_act = F.relu
    elif kernel == "softplus":
        self.kernel_act = F.softplus
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    self._init_weights()

def _init_weights(self):
    """Initialize weights."""
    nn.init.xavier_uniform_(self.to_q.weight)
    nn.init.xavier_uniform_(self.to_k.weight)
    nn.init.xavier_uniform_(self.to_v.weight)
    nn.init.xavier_uniform_(self.to_out[0].weight)
    
    if self.to_q.bias is not None:
        nn.init.zeros_(self.to_q.bias)
        nn.init.zeros_(self.to_k.bias)
        nn.init.zeros_(self.to_v.bias)
    
    nn.init.zeros_(self.to_out[0].bias)

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for 3D linear attention.
    
    Args:
        x: Input tensor of shape (batch, depth, height, width, dim)
        mask: Optional 3D mask
    
    Returns:
        Output tensor
    """
    batch_size, D, H, W, dim = x.shape
    total_elements = D * H * W
    
    # Normalize input
    x_norm = self.norm(x)
    
    # Project to queries, keys, values
    q = self.to_q(x_norm).view(batch_size, total_elements, self.num_heads, self.head_dim)
    k = self.to_k(x_norm).view(batch_size, total_elements, self.num_heads, self.head_dim)
    v = self.to_v(x_norm).view(batch_size, total_elements, self.num_heads, self.head_dim)
    
    # Apply kernel to keys
    k = self.kernel_act(k)
    
    # Apply scaling to queries
    q = q * self.scale
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.view(batch_size, total_elements, 1, 1)
        k = k * mask_flat
        v = v * mask_flat
    
    # Linear attention computation
    # K^T V: (batch, heads, dim, dim)
    k_t_v = torch.einsum('bnhd,bnhv->bhvd', k, v)
    
    # Denominator: sum(K) for each query
    k_sum = k.sum(dim=1, keepdim=True)  # (batch, 1, heads, dim)
    
    # Attention output
    numerator = torch.einsum('bqhd,bhvd->bqhv', q, k_t_v)
    denominator = torch.einsum('bqhd,bhvd->bqh', q, k_sum.transpose(1, 2))
    
    # Avoid division by zero
    output = numerator / (denominator.unsqueeze(-1) + 1e-8)
    
    # Reshape back to 3D
    output = output.view(batch_size, D, H, W, self.num_heads, self.head_dim)
    output = output.permute(0, 1, 2, 3, 4, 5).contiguous()
    output = output.view(batch_size, D, H, W, dim)
    
    # Project and add residual
    output = self.to_out(output)
    output = x + output
    
    return output

class MultimodalCrossAttention(nn.Module):
    """
Cross-modal attention for fusing information from different modalities.

Supports attention between:
- Text and images
- Text and 3D representations
- Images and 3D representations
"""

def __init__(
    self,
    query_dim: int,
    context_dim: int,
    num_heads: int = 8,
    dropout: float = 0.0,
    gated: bool = True,
    use_flash: bool = True,
    norm_type: str = "rms",
):
    super().__init__()
    
    self.query_dim = query_dim
    self.context_dim = context_dim
    self.num_heads = num_heads
    self.head_dim = query_dim // num_heads
    self.scale = self.head_dim ** -0.5
    
    # Normalization layers
    if norm_type == "rms":
        self.query_norm = RMSNorm(query_dim)
        self.context_norm = RMSNorm(context_dim)
    elif norm_type == "adaptive":
        self.query_norm = AdaptiveLayerNorm(query_dim)
        self.context_norm = AdaptiveLayerNorm(context_dim)
    else:
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
    
    # Projection layers
    self.to_q = nn.Linear(query_dim, query_dim)
    self.to_k = nn.Linear(context_dim, query_dim)
    self.to_v = nn.Linear(context_dim, query_dim)
    
    # Output projection
    self.to_out = nn.Sequential(
        nn.Linear(query_dim, query_dim),
        nn.Dropout(dropout),
    )
    
    # Gating mechanism
    if gated:
        self.gate = nn.Sequential(
            nn.Linear(query_dim + context_dim, query_dim),
            nn.Sigmoid(),
        )
    else:
        self.gate = None
    
    # Flash attention support
    self.use_flash = use_flash
    if use_flash:
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.has_flash = True
        except ImportError:
            print("Flash attention not available for cross attention")
            self.has_flash = False
    else:
        self.has_flash = False
    
    # Dropout
    self.attn_dropout = nn.Dropout(dropout)
    
    self._init_weights()

def _init_weights(self):
    """Initialize weights."""
    nn.init.xavier_uniform_(self.to_q.weight)
    nn.init.xavier_uniform_(self.to_k.weight)
    nn.init.xavier_uniform_(self.to_v.weight)
    nn.init.xavier_uniform_(self.to_out[0].weight)
    
    nn.init.zeros_(self.to_q.bias)
    nn.init.zeros_(self.to_k.bias)
    nn.init.zeros_(self.to_v.bias)
    nn.init.zeros_(self.to_out[0].bias)
    
    if self.gate is not None:
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)

def forward(
    self,
    query: torch.Tensor,
    context: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
    return_attn: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Forward pass for multimodal cross attention.
    
    Args:
        query: Query tensor of shape (batch, query_len, query_dim)
        context: Context tensor of shape (batch, context_len, context_dim)
        query_mask: Optional mask for query
        context_mask: Optional mask for context
        return_attn: Whether to return attention weights
    
    Returns:
        Output tensor and optionally attention weights
    """
    batch_size, query_len, _ = query.shape
    context_len = context.shape[1]
    
    # Normalize inputs
    query_norm = self.query_norm(query)
    context_norm = self.context_norm(context)
    
    # Project to queries, keys, values
    q = self.to_q(query_norm).view(batch_size, query_len, self.num_heads, self.head_dim)
    k = self.to_k(context_norm).view(batch_size, context_len, self.num_heads, self.head_dim)
    v = self.to_v(context_norm).view(batch_size, context_len, self.num_heads, self.head_dim)
    
    # Compute attention
    if self.has_flash and q.dtype in (torch.float16, torch.bfloat16):
        # Use flash attention
        q = q.transpose(1, 2)  # (batch, heads, query_len, dim)
        k = k.transpose(1, 2)  # (batch, heads, context_len, dim)
        v = v.transpose(1, 2)  # (batch, heads, context_len, dim)
        
        # Combine masks if provided
        attn_mask = None
        if query_mask is not None and context_mask is not None:
            # For cross attention, we need a mask of shape (batch, query_len, context_len)
            attn_mask = query_mask.unsqueeze(-1) * context_mask.unsqueeze(1)
        
        attn_output = self.flash_attn_func(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        ).transpose(1, 2)  # (batch, query_len, heads, dim)
        
        attn_weights = None
    
    else:
        # Standard attention
        q = q * self.scale
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
        
        # Apply masks if provided
        if query_mask is not None and context_mask is not None:
            mask = query_mask.unsqueeze(-1) * context_mask.unsqueeze(1)  # (batch, query_len, context_len)
            mask = mask.unsqueeze(1)  # (batch, 1, query_len, context_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        elif context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, context_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
    
    # Reshape attention output
    attn_output = attn_output.reshape(batch_size, query_len, self.query_dim)
    
    # Apply gating if enabled
    if self.gate is not None:
        # Compute gate value based on query and context
        context_mean = context_norm.mean(dim=1, keepdim=True)  # (batch, 1, context_dim)
        gate_input = torch.cat([query_norm, context_mean.expand(-1, query_len, -1)], dim=-1)
        gate_value = self.gate(gate_input)
        attn_output = attn_output * gate_value
    
    # Project and add residual
    output = self.to_out(attn_output)
    output = query + output
    
    if return_attn:
        return output, attn_weights
    return output

class MemoryEfficientAttention(nn.Module):
    """
Memory efficient attention using gradient checkpointing and chunking.

Designed for very long sequences or large 3D grids.
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    chunk_size: int = 1024,
    checkpoint: bool = True,
    dropout: float = 0.0,
    qkv_bias: bool = False,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.chunk_size = chunk_size
    self.checkpoint = checkpoint
    self.scale = self.head_dim ** -0.5
    
    # Linear projections
    self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
    self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout),
    )
    
    # Normalization
    self.norm = RMSNorm(dim)
    
    self._init_weights()

def _init_weights(self):
    """Initialize weights."""
    nn.init.xavier_uniform_(self.to_q.weight)
    nn.init.xavier_uniform_(self.to_k.weight)
    nn.init.xavier_uniform_(self.to_v.weight)
    nn.init.xavier_uniform_(self.to_out[0].weight)
    
    if self.to_q.bias is not None:
        nn.init.zeros_(self.to_q.bias)
        nn.init.zeros_(self.to_k.bias)
        nn.init.zeros_(self.to_v.bias)
    
    nn.init.zeros_(self.to_out[0].bias)

def _attention_chunk(
    self,
    q_chunk: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute attention for a chunk of queries."""
    batch_size, chunk_len, num_heads, head_dim = q_chunk.shape
    seq_len = k.shape[1]
    
    # Compute attention scores for this chunk
    q_chunk = q_chunk * self.scale
    attn_scores = torch.einsum('bqhd,bkhd->bhqk', q_chunk, k)
    
    if mask is not None:
        # Apply mask for this chunk
        mask_chunk = mask[:, :chunk_len].unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, chunk_len, seq_len)
        attn_scores = attn_scores.masked_fill(mask_chunk == 0, -1e9)
    
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
    
    return attn_output

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for memory efficient attention.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Optional attention mask
    
    Returns:
        Output tensor
    """
    batch_size, seq_len, _ = x.shape
    
    # Normalize input
    x_norm = self.norm(x)
    
    # Project to queries, keys, values
    q = self.to_q(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
    k = self.to_k(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
    v = self.to_v(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim)
    
    # Process in chunks
    num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
    attn_outputs = []
    
    for i in range(num_chunks):
        start_idx = i * self.chunk_size
        end_idx = min((i + 1) * self.chunk_size, seq_len)
        
        q_chunk = q[:, start_idx:end_idx]
        
        if self.checkpoint:
            # Use gradient checkpointing for this chunk
            attn_chunk = checkpoint.checkpoint(
                self._attention_chunk,
                q_chunk, k, v, mask,
                use_reentrant=False,
            )
        else:
            attn_chunk = self._attention_chunk(q_chunk, k, v, mask)
        
        attn_outputs.append(attn_chunk)
    
    # Combine chunks
    attn_output = torch.cat(attn_outputs, dim=1)
    attn_output = attn_output.reshape(batch_size, seq_len, self.dim)
    
    # Project and add residual
    output = self.to_out(attn_output)
    output = x + output
    
    return output

class AdaptiveAttention(nn.Module):
    """
Adaptive attention that dynamically selects attention mechanism based on input.

Can switch between:
- Full attention for short sequences
- Sparse attention for medium sequences
- Linear attention for very long sequences
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    threshold_short: int = 256,
    threshold_medium: int = 1024,
    dropout: float = 0.0,
    qkv_bias: bool = False,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.threshold_short = threshold_short
    self.threshold_medium = threshold_medium
    
    # Different attention mechanisms
    self.full_attention = MultiScaleAttention(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        qkv_bias=qkv_bias,
    )
    
    self.sparse_attention = Sparse3DAttention(
        dim=dim,
        num_heads=num_heads,
        grid_size=(32, 32, 32),  # Will be adjusted dynamically
        dropout=dropout,
        qkv_bias=qkv_bias,
    )
    
    self.linear_attention = LinearAttention3D(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        qkv_bias=qkv_bias,
    )
    
    # Gating network to choose attention mechanism
    self.gate_network = nn.Sequential(
        nn.Linear(dim + 3, 64),  # +3 for sequence length features
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, 3),  # 3 attention mechanisms
        nn.Softmax(dim=-1),
    )
    
    # Normalization
    self.norm = RMSNorm(dim)
    
def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for adaptive attention.
    
    Args:
        x: Input tensor
        mask: Optional attention mask
    
    Returns:
        Output tensor
    """
    batch_size, *spatial_dims, dim = x.shape
    seq_len = math.prod(spatial_dims)
    
    # Extract features for gating
    x_norm = self.norm(x)
    x_mean = x_norm.mean(dim=tuple(range(1, len(spatial_dims) + 1)), keepdim=True)
    
    # Add sequence length features
    seq_len_feat = torch.tensor([
        seq_len / self.threshold_short,
        seq_len / self.threshold_medium,
        math.log(seq_len + 1)
    ], device=x.device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
    
    gate_input = torch.cat([x_mean, seq_len_feat], dim=-1)
    gate_weights = self.gate_network(gate_input)  # (batch, 1, 3)
    
    # Reshape for 3D attention if needed
    if len(spatial_dims) == 3:
        x_3d = x.view(batch_size, *spatial_dims, dim)
        
        # Update sparse attention grid size
        D, H, W = spatial_dims
        self.sparse_attention.grid_size = (D, H, W)
        
        # Compute outputs from each attention mechanism
        full_out = self.full_attention(x_3d.view(batch_size, seq_len, dim), mask)
        full_out = full_out.view(batch_size, D, H, W, dim)
        
        sparse_out = self.sparse_attention(x_3d, mask)
        linear_out = self.linear_attention(x_3d, mask)
        
        # Combine with gate weights
        output = (
            gate_weights[..., 0] * full_out +
            gate_weights[..., 1] * sparse_out +
            gate_weights[..., 2] * linear_out
        )
        
        # Add residual
        output = x + output
        
    else:
        # 1D or 2D input
        x_flat = x.view(batch_size, seq_len, dim)
        
        # Only use full attention for non-3D inputs
        output = self.full_attention(x_flat, mask)
        output = output.view(batch_size, *spatial_dims, dim)
    
    return output

#Factory function for creating attention modules#

def create_attention_module(
module_type: str = "multiscale",
**kwargs,
) -> nn.Module:
    """
Factory function for creating attention modules.

Args:
    module_type: Type of attention module
        ("multiscale", "sparse3d", "linear3d", "cross", "memory", "adaptive")
    **kwargs: Arguments for the module constructor

Returns:
    Attention module instance
"""
module_registry = {
    "multiscale": MultiScaleAttention,
    "sparse3d": Sparse3DAttention,
    "linear3d": LinearAttention3D,
    "cross": MultimodalCrossAttention,
    "memory": MemoryEfficientAttention,
    "adaptive": AdaptiveAttention,
}

if module_type not in module_registry:
    raise ValueError(
        f"Unknown attention module type: {module_type}. "
        f"Available types: {list(module_registry.keys())}"
    )

return module_registry[module_type](**kwargs)

#Utility functions#

def compute_attention_scores(
q: torch.Tensor,
k: torch.Tensor,
scale: float = 1.0,
mask: Optional[torch.Tensor] = None,
causal: bool = False,
) -> torch.Tensor:
    """
Compute attention scores between queries and keys.

Args:
    q: Query tensor
    k: Key tensor
    scale: Scaling factor
    mask: Optional attention mask
    causal: Whether to apply causal masking

Returns:
    Attention scores
"""
scores = torch.einsum('bqhd,bkhd->bhqk', q * scale, k)

if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)

if causal:
    seq_len = q.shape[1]
    causal_mask = torch.ones(seq_len, seq_len, device=scores.device).triu(diagonal=1).bool()
    scores = scores.masked_fill(causal_mask, -1e9)

return scores

def apply_attention_weights(
attn_weights: torch.Tensor,
v: torch.Tensor,
) -> torch.Tensor:
    """
Apply attention weights to values.

Args:
    attn_weights: Attention weights
    v: Value tensor

Returns:
    Attention output
"""
return torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
Create causal attention mask.

Args:
    seq_len: Sequence length
    device: Device for the mask

Returns:
    Causal mask tensor
"""
if device is None:
    device = torch.device("cpu")

mask = torch.ones(seq_len, seq_len, device=device).triu(diagonal=1).bool()
return mask

def create_3d_positional_encoding(
grid_size: Tuple[int, int, int],
dim: int,
device: torch.device = None,
) -> torch.Tensor:
  """
Create 3D positional encoding for volumetric data.

Args:
    grid_size: (depth, height, width) of the grid
    dim: Dimension of positional encoding
    device: Device for the encoding

Returns:
    Positional encoding tensor
"""
if device is None:
    device = torch.device("cpu")

D, H, W = grid_size
total_elements = D * H * W

# Create grid coordinates
coords_d = torch.arange(D, device=device).float()
coords_h = torch.arange(H, device=device).float()
coords_w = torch.arange(W, device=device).float()

# Normalize coordinates
coords_d = 2 * (coords_d / (D - 1)) - 1 if D > 1 else torch.zeros(1, device=device)
coords_h = 2 * (coords_h / (H - 1)) - 1 if H > 1 else torch.zeros(1, device=device)
coords_w = 2 * (coords_w / (W - 1)) - 1 if W > 1 else torch.zeros(1, device=device)

# Create meshgrid
grid_d, grid_h, grid_w = torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij')

# Flatten coordinates
pos_d = grid_d.reshape(-1, 1)
pos_h = grid_h.reshape(-1, 1)
pos_w = grid_w.reshape(-1, 1)

# Compute positional encoding
position = torch.cat([pos_d, pos_h, pos_w], dim=1)

# Create frequency bands
num_bands = dim // 6  # 3 dimensions * 2 (sin/cos)
if num_bands == 0:
    num_bands = 1

freqs = torch.exp(
    torch.linspace(
        math.log(1.0),
        math.log(1000.0),
        num_bands,
        device=device,
    )
).unsqueeze(0)

# Compute sin/cos encodings
pos_enc = []
for i in range(3):  # For each dimension
    angles = position[:, i:i+1] * freqs * math.pi
    pos_enc.append(torch.sin(angles))
    pos_enc.append(torch.cos(angles))

pos_enc = torch.cat(pos_enc, dim=1)

# Pad or truncate to desired dimension
if pos_enc.shape[1] < dim:
    pad = torch.zeros(total_elements, dim - pos_enc.shape[1], device=device)
    pos_enc = torch.cat([pos_enc, pad], dim=1)
elif pos_enc.shape[1] > dim:
    pos_enc = pos_enc[:, :dim]

# Reshape to 3D
pos_enc = pos_enc.view(D, H, W, dim)

return pos_enc

#Test the module#

if __name__ == "__main__":
    print("Testing Attention Modules...")

    # Test configurations
batch_size = 2
seq_len = 128
dim = 512
num_heads = 8

# Test MultiScaleAttention
print("\n1. Testing MultiScaleAttention:")
x = torch.randn(batch_size, seq_len, dim)
attn = MultiScaleAttention(dim=dim, num_heads=num_heads)
output = attn(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")

# Test Sparse3DAttention
print("\n2. Testing Sparse3DAttention:")
D, H, W = 16, 16, 16
x_3d = torch.randn(batch_size, D, H, W, dim)
attn_3d = Sparse3DAttention(dim=dim, num_heads=num_heads, grid_size=(D, H, W))
output_3d = attn_3d(x_3d)
print(f"   Input shape: {x_3d.shape}")
print(f"   Output shape: {output_3d.shape}")

# Test LinearAttention3D
print("\n3. Testing LinearAttention3D:")
linear_attn = LinearAttention3D(dim=dim, num_heads=num_heads)
output_linear = linear_attn(x_3d)
print(f"   Input shape: {x_3d.shape}")
print(f"   Output shape: {output_linear.shape}")

# Test MultimodalCrossAttention
print("\n4. Testing MultimodalCrossAttention:")
query_len = 32
context_len = 64
query = torch.randn(batch_size, query_len, dim)
context = torch.randn(batch_size, context_len, dim * 2)  # Different dimension
cross_attn = MultimodalCrossAttention(
    query_dim=dim,
    context_dim=dim * 2,
    num_heads=num_heads,
)
output_cross = cross_attn(query, context)
print(f"   Query shape: {query.shape}")
print(f"   Context shape: {context.shape}")
print(f"   Output shape: {output_cross.shape}")

# Test MemoryEfficientAttention
print("\n5. Testing MemoryEfficientAttention:")
mem_attn = MemoryEfficientAttention(dim=dim, num_heads=num_heads)
output_mem = mem_attn(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output_mem.shape}")

# Test AdaptiveAttention
print("\n6. Testing AdaptiveAttention:")
adaptive_attn = AdaptiveAttention(dim=dim, num_heads=num_heads)
output_adaptive = adaptive_attn(x_3d)
print(f"   Input shape: {x_3d.shape}")
print(f"   Output shape: {output_adaptive.shape}")

# Test utility functions
print("\n7. Testing Utility Functions:")
mask = create_causal_mask(seq_len)
print(f"   Causal mask shape: {mask.shape}")

pos_enc = create_3d_positional_encoding((8, 8, 8), dim=64)
print(f"   3D positional encoding shape: {pos_enc.shape}")

print("\nAll attention module tests passed!")