"""
3D Diffusion Models for World Generation
Implements diffusion processes for 3D scene representation learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
from einops import rearrange, repeat

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Batch of timesteps, shape [batch_size]
        Returns:
            time embeddings, shape [batch_size, dim]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding

class AdaptiveGroupNorm(nn.Module):
    """GroupNorm with adaptive scaling from time embeddings"""
    
    def __init__(self, num_groups: int, num_channels: int, time_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=1e-6)
        self.adaptive_scale = nn.Linear(time_dim, num_channels)
        self.adaptive_shift = nn.Linear(time_dim, num_channels)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Normalize
        x = self.norm(x)
        
        # Adaptive scaling
        scale = self.adaptive_scale(t_emb)[:, :, None, None, None]  # 5D: [B, C, D, H, W]
        shift = self.adaptive_shift(t_emb)[:, :, None, None, None]
        
        return x * (1 + scale) + shift

class ResidualBlock3D(nn.Module):
    """3D Residual block with time conditioning"""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaptiveGroupNorm(8, out_channels, time_dim)
        
        # Second convolution
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveGroupNorm(8, out_channels, time_dim)
        
        # Time projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection if dimensions change
        if in_channels != out_channels:
            self.skip_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
            
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = x
        B, C, D, H, W = x.shape
        
        # Time conditioning
        t_emb = self.time_mlp(t_emb)
        scale, shift = t_emb.chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        
        # First convolution
        h = self.conv1(h)
        h = self.norm1(h, t_emb)
        h = F.silu(h)
        h = h * (1 + scale) + shift
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h, t_emb)
        h = self.dropout(h)
        
        # Skip connection
        skip = self.skip_conv(x)
        
        return F.silu(skip + h)

class AttentionBlock3D(nn.Module):
    """3D Attention block for diffusion models"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # Normalization
        self.norm = nn.GroupNorm(8, channels)
        
        # Attention layers
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)
        
        # Relative positional encoding
        self.rel_pos_encoding = RelativePositionalEncoding3D(self.head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(h)
        q, k, v = rearrange(qkv, 'b (three c) d h w -> three b c d h w', three=3)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b (heads d) depth height width -> b heads (depth height width) d', 
                     heads=self.num_heads)
        k = rearrange(k, 'b (heads d) depth height width -> b heads (depth height width) d',
                     heads=self.num_heads)
        v = rearrange(v, 'b (heads d) depth height width -> b heads (depth height width) d',
                     heads=self.num_heads)
        
        # Attention with relative positional encoding
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) / math.sqrt(self.head_dim)
        
        # Add relative positional bias
        rel_pos_bias = self.rel_pos_encoding(D, H, W, device=x.device)
        attn = attn + rel_pos_bias
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Reshape back
        out = rearrange(out, 'b heads (depth height width) d -> b (heads d) depth height width',
                       depth=D, height=H, width=W)
        
        # Project out
        out = self.proj_out(out)
        
        return x + out

class RelativePositionalEncoding3D(nn.Module):
    """Relative positional encoding for 3D attention"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.num_buckets = 32
        self.max_distance = 128
        
        self.rel_pos_bias = nn.Embedding(self.num_buckets, 1)
        
    def forward(self, D: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        # Create grid for relative positions
        d_coords = torch.arange(D, device=device)
        h_coords = torch.arange(H, device=device)
        w_coords = torch.arange(W, device=device)
        
        # 3D meshgrid
        d1, h1, w1 = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        d2, h2, w2 = torch.meshgrid(d_coords, h_coords, w_coords, indexing='ij')
        
        # Flatten and compute relative distances
        d1_flat = d1.flatten()[:, None]
        h1_flat = h1.flatten()[:, None]
        w1_flat = w1.flatten()[:, None]
        d2_flat = d2.flatten()[None, :]
        h2_flat = h2.flatten()[None, :]
        w2_flat = w2.flatten()[None, :]
        
        # 3D relative distances
        rel_d = d1_flat - d2_flat
        rel_h = h1_flat - h2_flat
        rel_w = w1_flat - w2_flat
        
        # Map to buckets
        def _bucketize(x, num_buckets, max_distance):
            ret = 0
            x = torch.abs(x)
            for i in range(num_buckets):
                if x <= (max_distance * (i + 1) / num_buckets):
                    ret = i
                    break
            return ret
        
        # Vectorized bucketization
        abs_rel_d = torch.abs(rel_d)
        abs_rel_h = torch.abs(rel_h)
        abs_rel_w = torch.abs(rel_w)
        
        # Combine 3D distances
        combined_dist = torch.sqrt(rel_d.float()**2 + rel_h.float()**2 + rel_w.float()**2)
        
        # Bucketize
        buckets = torch.floor(combined_dist / (self.max_distance / self.num_buckets)).long()
        buckets = torch.clamp(buckets, 0, self.num_buckets - 1)
        
        # Get bias
        bias = self.rel_pos_bias(buckets).squeeze(-1)
        
        return bias

class Downsample3D(nn.Module):
    """3D Downsampling layer"""
    
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        if with_conv:
            self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.AvgPool3d(2)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample3D(nn.Module):
    """3D Upsampling layer"""
    
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1) if with_conv else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.conv is not None:
            x = self.conv(x)
        return x

class UNet3D(nn.Module):
    """3D U-Net for diffusion models"""
    
    def __init__(
        self,
        in_channels: int = 4,  # RGB + density
        out_channels: int = 4,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.0,
        time_dim: int = 256
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input convolution
        self.input_conv = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [ch]
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock3D(ch, mult * model_channels, time_dim, dropout),
                    AttentionBlock3D(mult * model_channels) if 2**level in attention_resolutions else nn.Identity()
                ]
                ch = mult * model_channels
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample3D(ch)]))
                input_block_chans.append(ch)
                
        # Middle block
        self.middle_block = nn.ModuleList([
            ResidualBlock3D(ch, ch, time_dim, dropout),
            AttentionBlock3D(ch),
            ResidualBlock3D(ch, ch, time_dim, dropout)
        ])
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock3D(
                        ch + input_block_chans.pop(),
                        mult * model_channels if level > 0 else model_channels,
                        time_dim,
                        dropout
                    ),
                    AttentionBlock3D(mult * model_channels if level > 0 else model_channels) 
                    if 2**level in attention_resolutions else nn.Identity()
                ]
                ch = mult * model_channels if level > 0 else model_channels
                self.up_blocks.append(nn.ModuleList(layers))
                
            if level > 0:
                self.up_blocks.append(nn.ModuleList([Upsample3D(ch)]))
                
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv3d(ch, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            t: Timestep tensor of shape [B]
        Returns:
            Output tensor of same shape as input
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input convolution
        h = self.input_conv(x)
        
        # Skip connections
        hs = [h]
        
        # Downsample path
        for module_list in self.down_blocks:
            if len(module_list) == 1:  # Downsample layer
                h = module_list[0](h)
                hs.append(h)
            else:  # Residual + Attention blocks
                for layer in module_list:
                    if isinstance(layer, ResidualBlock3D):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                hs.append(h)
                
        # Middle path
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock3D):
                h = layer(h, t_emb)
            else:
                h = layer(h)
                
        # Upsample path
        for module_list in self.up_blocks:
            if len(module_list) == 1:  # Upsample layer
                h = module_list[0](h)
            else:  # Residual + Attention blocks
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in module_list:
                    if isinstance(layer, ResidualBlock3D):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                        
        # Output
        return self.output_conv(h)

class DiffusionModel3D(nn.Module):
    """Main 3D diffusion model class"""
    
    def __init__(
        self,
        unet_config: Dict[str, Any],
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        timesteps: int = 1000,
        loss_type: str = "l2",
        objective: str = "pred_noise"
    ):
        super().__init__()
        
        self.unet = UNet3D(**unet_config)
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.objective = objective
        
        # Setup noise schedule
        self.register_buffer('betas', self.get_beta_schedule(beta_schedule, beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))
        
    def get_beta_schedule(self, schedule: str, beta_start: float, beta_end: float, timesteps: int) -> torch.Tensor:
        """Get noise schedule"""
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif schedule == "cosine":
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")
            
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute diffusion loss"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        model_output = self.unet(x_t, t)
        
        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
            
        # Compute loss
        if self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction='none')
        elif self.loss_type == "l2":
            loss = F.mse_loss(model_output, target, reduction='none')
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights[:, None, None, None, None]
            
        return loss.mean()
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        guidance_scale: float = 1.0,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Reverse diffusion step: p(x_{t-1} | x_t)"""
        betas_t = self.betas[t][:, None, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None, None, None]
        
        # Predict noise
        model_output = self.unet(x, t)
        
        if guidance_scale != 1.0 and conditioning is not None:
            # Classifier-free guidance
            uncond_output = self.unet(x, torch.zeros_like(t))
            model_output = uncond_output + guidance_scale * (model_output - uncond_output)
        
        # Calculate x_{t-1}
        pred_xstart = self.predict_start_from_noise(x, t, model_output)
        pred_xstart = torch.clamp(pred_xstart, -1., 1.)
        
        mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        
        if t_index == 0:
            return mean
        else:
            posterior_variance = self.get_variance(t)
            noise = torch.randn_like(x)
            return mean + torch.sqrt(posterior_variance) * noise
    
    def get_variance(self, t: torch.Tensor) -> torch.Tensor:
        """Get variance for reverse process"""
        alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None, None]
        alphas_cumprod_t_prev = self.alphas_cumprod_prev[t][:, None, None, None, None]
        betas_t = self.betas[t][:, None, None, None, None]
        
        # Clip variance
        variance = betas_t * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
        variance = torch.clamp(variance, min=1e-20)
        return variance
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        guidance_scale: float = 1.0,
        conditioning: Optional[torch.Tensor] = None,
        timesteps: Optional[int] = None,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Generate samples from noise"""
        b, c, d, h, w = shape
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Sample loop
        timesteps = timesteps or self.timesteps
        for i in reversed(range(0, timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, i, guidance_scale, conditioning)
            
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> torch.Tensor:
        """Forward pass"""
        if return_loss:
            # Training mode
            if t is None:
                t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
            return self.p_losses(x, t)
        else:
            # Inference mode
            return self.unet(x, t)