"""
Transformer Blocks for World Model Architecture.

This module contains various transformer block implementations optimized for
3D world generation, including standard attention blocks, cross-attention blocks,
and specialized blocks for diffusion models.
"""

import math
from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .custom_layers import AdaptiveLayerNorm, RMSNorm, GatedLinearUnit

class MultiHeadAttention(nn.Module):
    """
Multi-head attention with optional memory efficient attention and flash attention.

Supports:
- Standard attention
- Memory efficient attention (for long sequences)
- Flash attention (if available)
- Cross attention between two modalities
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    dim_head: int = 64,
    dropout: float = 0.0,
    use_flash: bool = True,
    qkv_bias: bool = False,
    causal: bool = False,
    rotary_emb: bool = False,
    max_seq_len: int = 2048,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.dim_head = dim_head
    self.inner_dim = dim_head * num_heads
    self.dropout = dropout
    self.use_flash = use_flash
    self.causal = causal
    
    # Check if flash attention is available
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
    
    # Linear projections
    self.to_q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
    self.to_k = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
    self.to_v = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
    self.to_out = nn.Sequential(
        nn.Linear(self.inner_dim, dim),
        nn.Dropout(dropout),
    )
    
    # Rotary embeddings for positional encoding
    if rotary_emb:
        self.rotary_emb = RotaryEmbedding(dim_head, max_seq_len=max_seq_len)
    else:
        self.rotary_emb = None
    
    # Scale for dot product attention
    self.scale = dim_head ** -0.5
    
    # Initialize weights
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

def forward(
    self,
    x: torch.Tensor,
    context: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    return_attn: bool = False,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Forward pass for multi-head attention.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        context: Optional context tensor for cross-attention
        mask: Optional attention mask of shape (batch, seq_len) or (batch, seq_len, seq_len)
        return_attn: Whether to return attention weights
        **kwargs: Additional arguments
    
    Returns:
        Output tensor and optionally attention weights
    """
    batch_size, seq_len, _ = x.shape
    
    # Project to queries, keys, values
    q = self.to_q(x)
    
    if context is not None:
        k = self.to_k(context)
        v = self.to_v(context)
        k_seq_len = context.shape[1]
    else:
        k = self.to_k(x)
        v = self.to_v(x)
        k_seq_len = seq_len
    
    # Reshape to (batch, heads, seq_len, dim_head)
    q = q.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
    k = k.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
    v = v.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
    
    # Apply rotary embeddings if enabled
    if self.rotary_emb is not None:
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
    
    # Use flash attention if available and conditions are met
    if self.has_flash and q.dtype in (torch.float16, torch.bfloat16):
        # Flash attention expects (batch, seq_len, heads, dim_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Convert mask format if provided
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                attn_mask = mask.unsqueeze(1)
        
        output = self.flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=self.causal,
            window_size=(-1, -1),  # disable local attention
        )
        attn_weights = None
    
    else:
        # Standard attention implementation
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.ones(
                seq_len, k_seq_len, device=scores.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch, seq_len, heads, dim_head)
        output = output.transpose(1, 2)
    
    # Reshape output
    output = output.reshape(batch_size, seq_len, self.inner_dim)
    
    # Apply output projection
    output = self.to_out(output)
    
    if return_attn:
        return output, attn_weights
    return output

class CrossAttention(nn.Module):
    """
Cross attention block for fusing information from two modalities.

Specifically designed for text-visual fusion in world generation.
"""

def __init__(
    self,
    dim: int,
    context_dim: Optional[int] = None,
    num_heads: int = 8,
    dim_head: int = 64,
    dropout: float = 0.0,
    gated: bool = True,
    use_flash: bool = True,
):
    super().__init__()
    
    context_dim = context_dim or dim
    
    self.norm = AdaptiveLayerNorm(dim)
    self.context_norm = AdaptiveLayerNorm(context_dim) if context_dim != dim else self.norm
    
    self.attn = MultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        dim_head=dim_head,
        dropout=dropout,
        use_flash=use_flash,
    )
    
    # Gating mechanism for adaptive fusion
    if gated:
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
    else:
        self.gate = None
    
    # Feed-forward network for post-attention processing
    self.ff = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * 4, dim),
        nn.Dropout(dropout),
    )
    
    # Skip connection scaling (Adaptive from LLaMA)
    self.alpha = nn.Parameter(torch.ones(1))

def forward(
    self,
    x: torch.Tensor,
    context: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for cross attention.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        context: Context tensor of shape (batch, context_len, context_dim)
        mask: Optional mask for x
        context_mask: Optional mask for context
    
    Returns:
        Output tensor
    """
    # Normalize inputs
    x_norm = self.norm(x)
    context_norm = self.context_norm(context)
    
    # Apply attention
    attn_output = self.attn(x_norm, context_norm, mask=context_mask)
    
    # Apply gating if enabled
    if self.gate is not None:
        gate_value = self.gate(x_norm)
        attn_output = attn_output * gate_value
    
    # Add skip connection with learned scaling
    x = x + self.alpha * attn_output
    
    # Apply feed-forward network
    ff_output = self.ff(self.norm(x))
    x = x + ff_output
    
    return x

class TransformerBlock(nn.Module):
    """
Standard transformer block with pre-normalization.

Based on the architecture from LLaMA and GPT-3, with optional improvements
for stability and efficiency.
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    dim_head: int = 64,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    activation: str = "gelu",
    norm_type: str = "rms",
    use_flash: bool = True,
    fused_mlp: bool = True,
    rotary_emb: bool = True,
    max_seq_len: int = 2048,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    
    # Normalization layers
    if norm_type == "rms":
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    elif norm_type == "adaptive":
        self.norm1 = AdaptiveLayerNorm(dim)
        self.norm2 = AdaptiveLayerNorm(dim)
    else:
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    # Self-attention
    self.attn = MultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        dim_head=dim_head,
        dropout=dropout,
        use_flash=use_flash,
        rotary_emb=rotary_emb,
        max_seq_len=max_seq_len,
    )
    
    # Feed-forward network
    mlp_dim = int(dim * mlp_ratio)
    
    if fused_mlp:
        # Fused MLP with Gated Linear Units (GLU)
        self.mlp = GatedFeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            activation=activation,
        )
    else:
        # Standard MLP
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=mlp_dim,
            dropout=dropout,
            activation=activation,
        )
    
    # Dropout for residual connections
    self.dropout = nn.Dropout(dropout)
    
    # Skip connection scaling (Adaptive from LLaMA)
    self.alpha1 = nn.Parameter(torch.ones(1))
    self.alpha2 = nn.Parameter(torch.ones(1))

def forward(
    self,
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    return_attn: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Forward pass for transformer block.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Optional attention mask
        return_attn: Whether to return attention weights
    
    Returns:
        Output tensor and optionally attention weights
    """
    # Self-attention with pre-normalization
    x_norm = self.norm1(x)
    
    if return_attn:
        attn_output, attn_weights = self.attn(x_norm, mask=mask, return_attn=True)
    else:
        attn_output = self.attn(x_norm, mask=mask)
    
    # First residual connection with learned scaling
    x = x + self.alpha1 * self.dropout(attn_output)
    
    # Feed-forward network with pre-normalization
    mlp_input = self.norm2(x)
    mlp_output = self.mlp(mlp_input)
    
    # Second residual connection with learned scaling
    x = x + self.alpha2 * self.dropout(mlp_output)
    
    if return_attn:
        return x, attn_weights
    return x

class DiffusionTransformerBlock(nn.Module):
    """
Transformer block specialized for diffusion models.

Incorporates timestep conditioning and optional class conditioning.
Based on DiT (Diffusion Transformers) architecture.
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    activation: str = "gelu",
    norm_type: str = "adaptive",
    cond_dim: int = 256,
    use_adaLN: bool = True,
    use_flash: bool = True,
):
    super().__init__()
    
    self.dim = dim
    self.cond_dim = cond_dim
    self.use_adaLN = use_adaLN
    
    # Adaptive Layer Normalization with conditioning
    if use_adaLN:
        self.norm1 = AdaptiveConditionalNorm(dim, cond_dim)
        self.norm2 = AdaptiveConditionalNorm(dim, cond_dim)
    else:
        if norm_type == "rms":
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
    
    # Self-attention
    self.attn = MultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        use_flash=use_flash,
    )
    
    # Feed-forward network
    mlp_dim = int(dim * mlp_ratio)
    self.mlp = FeedForward(
        dim=dim,
        hidden_dim=mlp_dim,
        dropout=dropout,
        activation=activation,
    )
    
    # Modulation layers for conditioning (from DiT)
    if use_adaLN:
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim),
        )
    
    # Dropout for residual connections
    self.dropout = nn.Dropout(dropout)

def forward(
    self,
    x: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for diffusion transformer block.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        cond: Conditioning tensor of shape (batch, cond_dim)
        mask: Optional attention mask
    
    Returns:
        Output tensor
    """
    batch_size, seq_len, _ = x.shape
    
    # Apply adaptive modulation if using AdaLN
    if self.use_adaLN and cond is not None:
        # Get modulation parameters from conditioning
        modulation = self.adaLN_modulation(cond)
        
        # Split into scale and shift parameters for each norm
        scale1, shift1, gate1, scale2, shift2, gate2 = modulation.chunk(6, dim=1)
        
        # Apply first normalization with modulation
        x_norm = self.norm1(x, scale=scale1, shift=shift1)
        
        # Self-attention
        attn_output = self.attn(x_norm, mask=mask)
        
        # Apply gating to attention output
        attn_output = attn_output * gate1.unsqueeze(1)
        
        # First residual connection
        x = x + self.dropout(attn_output)
        
        # Apply second normalization with modulation
        mlp_input = self.norm2(x, scale=scale2, shift=shift2)
        
        # Feed-forward network
        mlp_output = self.mlp(mlp_input)
        
        # Apply gating to MLP output
        mlp_output = mlp_output * gate2.unsqueeze(1)
        
        # Second residual connection
        x = x + self.dropout(mlp_output)
        
    else:
        # Standard transformer block without conditioning
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm, mask=mask)
        x = x + self.dropout(attn_output)
        
        mlp_input = self.norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + self.dropout(mlp_output)
    
    return x

class SparseTransformerBlock(nn.Module):
    """
Sparse transformer block with local and global attention.

Efficient for long sequences by combining:
- Local windowed attention
- Global attention with striding
- Linear attention for efficiency
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    window_size: int = 64,
    global_stride: int = 8,
    dropout: float = 0.0,
    mlp_ratio: float = 4.0,
    use_linear: bool = True,
):
    super().__init__()
    
    self.dim = dim
    self.window_size = window_size
    self.global_stride = global_stride
    
    # Normalization
    self.norm1 = AdaptiveLayerNorm(dim)
    self.norm2 = AdaptiveLayerNorm(dim)
    
    # Attention mechanisms
    self.local_attn = LocalWindowAttention(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        dropout=dropout,
    )
    
    self.global_attn = StridedAttention(
        dim=dim,
        num_heads=num_heads,
        stride=global_stride,
        dropout=dropout,
    )
    
    # Linear attention for efficiency (optional)
    if use_linear:
        self.linear_attn = LinearAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
        )
    else:
        self.linear_attn = None
    
    # Feed-forward network
    mlp_dim = int(dim * mlp_ratio)
    self.mlp = FeedForward(
        dim=dim,
        hidden_dim=mlp_dim,
        dropout=dropout,
    )
    
    # Gating for combining attention outputs
    self.attention_gate = nn.Sequential(
        nn.Linear(dim * 3 if use_linear else dim * 2, 3 if use_linear else 2),
        nn.Softmax(dim=-1),
    )
    
    # Dropout
    self.dropout = nn.Dropout(dropout)

def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward pass for sparse transformer block.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Optional attention mask
    
    Returns:
        Output tensor
    """
    batch_size, seq_len, _ = x.shape
    
    # Normalize input
    x_norm = self.norm1(x)
    
    # Compute different attention outputs
    local_out = self.local_attn(x_norm, mask=mask)
    global_out = self.global_attn(x_norm, mask=mask)
    
    if self.linear_attn is not None:
        linear_out = self.linear_attn(x_norm, mask=mask)
        attention_outputs = torch.stack([local_out, global_out, linear_out], dim=1)
    else:
        attention_outputs = torch.stack([local_out, global_out], dim=1)
    
    # Compute attention weights
    gate_input = attention_outputs.mean(dim=2)  # (batch, num_attentions, dim)
    gate_weights = self.attention_gate(gate_input)  # (batch, num_attentions)
    
    # Combine attention outputs
    gate_weights = gate_weights.unsqueeze(-1)  # (batch, num_attentions, 1)
    combined_attn = (attention_outputs * gate_weights).sum(dim=1)
    
    # First residual connection
    x = x + self.dropout(combined_attn)
    
    # Feed-forward network
    mlp_input = self.norm2(x)
    mlp_output = self.mlp(mlp_input)
    
    # Second residual connection
    x = x + self.dropout(mlp_output)
    
    return x

class LocalWindowAttention(nn.Module):
    """
Local windowed attention for efficient long sequence processing.
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    window_size: int = 64,
    dropout: float = 0.0,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.window_size = window_size
    
    self.attn = MultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout,
        use_flash=False,  # Windowed attention doesn't benefit from flash
    )

def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward pass for local window attention.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Optional attention mask
    
    Returns:
        Output tensor
    """
    batch_size, seq_len, _ = x.shape
    
    # Pad sequence to be divisible by window size
    padding = (self.window_size - seq_len % self.window_size) % self.window_size
    if padding > 0:
        x = F.pad(x, (0, 0, 0, padding))
        if mask is not None:
            mask = F.pad(mask, (0, padding), value=0)
        seq_len = x.shape[1]
    
    # Reshape into windows
    num_windows = seq_len // self.window_size
    x = x.view(batch_size, num_windows, self.window_size, self.dim)
    
    if mask is not None:
        mask = mask.view(batch_size, num_windows, self.window_size)
    
    # Apply attention within each window
    output_windows = []
    for w in range(num_windows):
        window_x = x[:, w]
        window_mask = mask[:, w] if mask is not None else None
        window_out = self.attn(window_x, mask=window_mask)
        output_windows.append(window_out)
    
    # Combine windows
    x = torch.stack(output_windows, dim=1)
    x = x.view(batch_size, seq_len, self.dim)
    
    # Remove padding
    if padding > 0:
        x = x[:, :seq_len - padding]
    
    return x

class StridedAttention(nn.Module):
    """
Strided attention for capturing global context efficiently.
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    stride: int = 8,
    dropout: float = 0.0,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.stride = stride
    
    self.attn = MultiHeadAttention(
        dim=dim,
        num_heads=num_heads,
        dropout=dropout,
    )

def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward pass for strided attention.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Optional attention mask
    
    Returns:
        Output tensor
    """
    batch_size, seq_len, _ = x.shape
    
    # Create strided sequence
    if seq_len % self.stride != 0:
        padding = self.stride - (seq_len % self.stride)
        x = F.pad(x, (0, 0, 0, padding))
        if mask is not None:
            mask = F.pad(mask, (0, padding), value=0)
        seq_len = x.shape[1]
    
    # Reshape to (batch, stride, seq_len/stride, dim)
    x_strided = x.view(batch_size, self.stride, seq_len // self.stride, self.dim)
    
    if mask is not None:
        mask_strided = mask.view(batch_size, self.stride, seq_len // self.stride)
    
    # Apply attention across strided dimension
    output = []
    for s in range(self.stride):
        stride_x = x_strided[:, s]
        stride_mask = mask_strided[:, s] if mask is not None else None
        stride_out = self.attn(stride_x, mask=stride_mask)
        output.append(stride_out)
    
    # Combine strides
    x = torch.stack(output, dim=1)
    x = x.view(batch_size, seq_len, self.dim)
    
    # Remove padding
    if seq_len > x.shape[1]:
        x = x[:, :seq_len]
    
    return x

class LinearAttention(nn.Module):
    """
Linear attention for O(n) complexity.

Based on "Transformers are RNNs" by Katharopoulos et al.
"""

def __init__(
    self,
    dim: int,
    num_heads: int = 8,
    dropout: float = 0.0,
):
    super().__init__()
    
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    
    # Linear projections
    self.to_q = nn.Linear(dim, dim)
    self.to_k = nn.Linear(dim, dim)
    self.to_v = nn.Linear(dim, dim)
    self.to_out = nn.Linear(dim, dim)
    
    # Activation for linear attention
    self.activation = nn.ReLU()
    
    # Dropout
    self.dropout = nn.Dropout(dropout)
    
    # Scale factor
    self.scale = self.head_dim ** -0.5
    
    # Epsilon for numerical stability
    self.eps = 1e-6

def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward pass for linear attention.
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Optional attention mask
    
    Returns:
        Output tensor
    """
    batch_size, seq_len, _ = x.shape
    
    # Project to queries, keys, values
    q = self.to_q(x) * self.scale
    k = self.to_k(x)
    v = self.to_v(x)
    
    # Reshape to multi-head
    q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    # Apply activation to keys
    k = self.activation(k)
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        k = k.masked_fill(mask == 0, 0)
        v = v.masked_fill(mask == 0, 0)
    
    # Linear attention computation
    # kv = k^T * v
    kv = torch.einsum('bhsd,bhsv->bhvd', k, v)
    
    # Denominator = sum(k) + eps
    z = 1 / (torch.einsum('bhld,bhld->bhl', q, k.sum(dim=2, keepdim=True)) + self.eps)
    
    # Output = q * kv / z
    output = torch.einsum('bhld,bhvd,bhl->bhld', q, kv, z)
    
    # Reshape back
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
    
    # Apply output projection
    output = self.to_out(output)
    output = self.dropout(output)
    
    return output

class FeedForward(nn.Module):
    """
Standard feed-forward network with GELU activation.
"""

def __init__(
    self,
    dim: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation: str = "gelu",
):
    super().__init__()
    
    hidden_dim = hidden_dim or dim * 4
    
    # Activation function
    if activation == "gelu":
        self.activation = nn.GELU()
    elif activation == "relu":
        self.activation = nn.ReLU()
    elif activation == "silu":
        self.activation = nn.SiLU()
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    # MLP layers
    self.net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        self.activation,
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for feed-forward network."""
    return self.net(x)

class GatedFeedForward(nn.Module):
    """
Gated feed-forward network with GLU activation.

More expressive than standard FFN, used in LLaMA and GPT-2.
"""

def __init__(
    self,
    dim: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation: str = "silu",
):
    super().__init__()
    
    hidden_dim = hidden_dim or dim * 4
    
    # Gated Linear Unit
    self.glu = GatedLinearUnit(
        dim=dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        activation=activation,
    )
    
    # Output projection
    self.out_proj = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
    )

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass for gated feed-forward network."""
    return self.out_proj(self.glu(x))

class AdaptiveConditionalNorm(nn.Module):
    """
Adaptive Layer Normalization with conditioning.

Used in diffusion transformers for timestep conditioning.
"""

def __init__(self, dim: int, cond_dim: int):
    super().__init__()
    
    self.dim = dim
    self.cond_dim = cond_dim
    
    # Standard layer norm
    self.norm = nn.LayerNorm(dim, elementwise_affine=False)
    
    # Learnable parameters (will be modulated by conditioning)
    self.weight = None
    self.bias = None

def forward(
    self,
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass with optional scale and shift.
    
    Args:
        x: Input tensor
        scale: Optional scale parameter
        shift: Optional shift parameter
    
    Returns:
        Normalized tensor
    """
    # Apply layer norm
    x = self.norm(x)
    
    # Apply adaptive scale and shift if provided
    if scale is not None and shift is not None:
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
    
    return x

class RotaryEmbedding(nn.Module):
    """
Rotary positional embeddings (RoPE).

Used for relative positional encoding in transformers.
"""

def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
    super().__init__()
    
    self.dim = dim
    self.max_seq_len = max_seq_len
    self.theta = theta
    
    # Precompute frequencies
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    # Cache for cos/sin values
    self._seq_len_cached = None
    self._cos_cached = None
    self._sin_cached = None

def _update_cos_sin_cache(self, x: torch.Tensor, seq_len: int):
    """Update cached cos/sin values for given sequence length."""
    if (
        self._seq_len_cached is None
        or seq_len > self._seq_len_cached
        or self._cos_cached.device != x.device
    ):
        self._seq_len_cached = seq_len
        
        # Create position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor.
    
    Args:
        x: Input tensor of shape (batch, seq_len, heads, dim_head)
    
    Returns:
        Tensor with rotary embeddings applied
    """
    batch_size, seq_len, num_heads, dim_head = x.shape
    
    # Update cache
    self._update_cos_sin_cache(x, seq_len)
    
    # Get cos and sin for current sequence length
    cos = self._cos_cached[:seq_len].to(x.device)
    sin = self._sin_cached[:seq_len].to(x.device)
    
    # Reshape cos and sin for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
    sin = sin.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim)
    
    # Split input into two halves for rotation
    x1 = x[..., : dim_head // 2]
    x2 = x[..., dim_head // 2 :]
    
    # Apply rotary embeddings
    rotated = torch.cat(
        [
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ],
        dim=-1,
    )
    
    return rotated

#Factory function for creating transformer blocks#

def create_transformer_block(
block_type: str = "standard",
**kwargs,
) -> nn.Module:
    """
Factory function for creating transformer blocks.

Args:
    block_type: Type of transformer block
        ("standard", "diffusion", "sparse", "cross", "local", "linear")
    **kwargs: Arguments for the block constructor

Returns:
    Transformer block instance
"""
block_registry = {
    "standard": TransformerBlock,
    "diffusion": DiffusionTransformerBlock,
    "sparse": SparseTransformerBlock,
    "cross": CrossAttention,
    "local": LocalWindowAttention,
    "linear": LinearAttention,
}

if block_type not in block_registry:
    raise ValueError(
        f"Unknown block type: {block_type}. "
        f"Available types: {list(block_registry.keys())}"
    )

return block_registry[block_type](**kwargs)

#utility functions#

def get_attention_mask(
seq_len: int,
batch_size: int = 1,
causal: bool = False,
device: torch.device = None,
) -> torch.Tensor:
    """
Generate attention mask.

Args:
    seq_len: Sequence length
    batch_size: Batch size
    causal: Whether to generate causal mask
    device: Device for the mask

Returns:
    Attention mask tensor
"""
if device is None:
    device = torch.device("cpu")

if causal:
    # Causal mask (upper triangular)
    mask = torch.ones(seq_len, seq_len, device=device).triu(diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    mask = mask.repeat(batch_size, 1, 1, 1)
else:
    # Full attention mask
    mask = torch.ones(batch_size, 1, seq_len, seq_len, device=device).bool()

return mask

def apply_rotary_embeddings(
x: torch.Tensor,
cos: torch.Tensor,
sin: torch.Tensor,
) -> torch.Tensor:
   """

Apply rotary embeddings to input tensor.

Args:
    x: Input tensor of shape (..., dim)
    cos: Cosine values
    sin: Sine values

Returns:
    Tensor with rotary embeddings applied
"""
# Split input into two halves
x1, x2 = x.chunk(2, dim=-1)

# Apply rotation
rotated = torch.cat(
    [
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ],
    dim=-1,
)

return rotated

#Test the module#

if __name__ == "__main__":
    print("Testing Transformer Blocks...")

# Test configurations
batch_size = 2
seq_len = 128
dim = 512
num_heads = 8

# Create test input
x = torch.randn(batch_size, seq_len, dim)

# Test standard transformer block
print("\n1. Testing Standard Transformer Block:")
block = TransformerBlock(dim=dim, num_heads=num_heads)
output = block(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
print(f"   Output mean: {output.mean().item():.4f}")
print(f"   Output std: {output.std().item():.4f}")

# Test diffusion transformer block
print("\n2. Testing Diffusion Transformer Block:")
cond = torch.randn(batch_size, 256)  # Conditioning vector
block = DiffusionTransformerBlock(dim=dim, num_heads=num_heads, cond_dim=256)
output = block(x, cond=cond)
print(f"   Input shape: {x.shape}")
print(f"   Cond shape: {cond.shape}")
print(f"   Output shape: {output.shape}")

# Test cross attention block
print("\n3. Testing Cross Attention Block:")
context = torch.randn(batch_size, 64, dim)  # Context sequence
block = CrossAttention(dim=dim, num_heads=num_heads)
output = block(x, context=context)
print(f"   Input shape: {x.shape}")
print(f"   Context shape: {context.shape}")
print(f"   Output shape: {output.shape}")

# Test sparse transformer block
print("\n4. Testing Sparse Transformer Block:")
block = SparseTransformerBlock(
    dim=dim,
    num_heads=num_heads,
    window_size=32,
    global_stride=8,
)
output = block(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")

# Test multi-head attention
print("\n5. Testing Multi-Head Attention:")
attn = MultiHeadAttention(dim=dim, num_heads=num_heads)
output = attn(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")

# Test local window attention
print("\n6. Testing Local Window Attention:")
local_attn = LocalWindowAttention(dim=dim, num_heads=num_heads, window_size=32)
output = local_attn(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")

# Test linear attention
print("\n7. Testing Linear Attention:")
linear_attn = LinearAttention(dim=dim, num_heads=num_heads)
output = linear_attn(x)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")

print("\nAll tests passed successfully!")