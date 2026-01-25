"""
World Model Module for FrexTech AI Simulations.

This module contains the core world generation model based on diffusion transformers
and Gaussian splatting. It provides a unified interface for generating, editing,
and representing 3D worlds from multimodal inputs.
"""

from attrs import Factory
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import numpy as np

from src.core.world_model.architecture.transformer_blocks import TransformerBlock
from src.core.world_model.architecture.attention_modules import MultiHeadAttention
from src.core.world_model.architecture.diffusion_models import DiffusionTransformer
from src.core.world_model.architecture.custom_layers import AdaptiveLayerNorm

from src.core.world_model.training.trainer import WorldModelTrainer
from src.core.world_model.training.loss_functions import WorldModelLoss
from src.core.world_model.training.optimizer_scheduler import OptimizerScheduler
from src.core.world_model.training.checkpoint_manager import CheckpointManager

from src.core.world_model.inference.generator import WorldGenerator
from src.core.world_model.inference.sampler import DiffusionSampler
from src.core.world_model.inference.post_processor import PostProcessor

version = "1.0.0"
author = "FrexTech AI Research Team"
email = "world-model@frextech-sim.com"

all = [
# Main model
"WorldModel",
"WorldModelConfig",

# Architecture components
"TransformerBlock",
"MultiHeadAttention",
"DiffusionTransformer",
"AdaptiveLayerNorm",

# Training
"WorldModelTrainer",
"WorldModelLoss",
"OptimizerScheduler",
"CheckpointManager",

# Inference
"WorldGenerator",
"DiffusionSampler",
"PostProcessor",

# Utilities
"create_world_model",
"load_pretrained",
"generate_world",
"edit_world",
]

@dataclass
class WorldModelConfig:
    """Configuration for the World Model."""

# Model architecture
latent_dim: int = 768
num_layers: int = 24
num_heads: int = 16
hidden_dim: int = 3072
dropout: float = 0.1
activation: str = "gelu"

# Diffusion parameters
diffusion_steps: int = 1000
beta_schedule: str = "cosine"
variance_type: str = "fixed_small"
guidance_scale: float = 7.5

# Gaussian splatting decoder
gaussian_decoder: Dict[str, Any] = field(default_factory=lambda: {
    "max_gaussians": 1000000,
    "sh_degree": 3,
    "opacity_activation": "sigmoid",
    "scale_activation": "exp",
    "rotation_activation": "normalize",
})

# NeRF decoder
nerf_decoder: Dict[str, Any] = field(default_factory=lambda: {
    "num_samples_per_ray": 128,
    "num_importance_samples": 0,
    "positional_encoding_freqs": 10,
    "density_activation": "softplus",
    "color_activation": "sigmoid",
})

# Training parameters
batch_size: int = 32
learning_rate: float = 1e-4
weight_decay: float = 0.01
gradient_clip: float = 1.0
warmup_steps: int = 10000

# Mixed precision
use_amp: bool = True
amp_dtype: str = "float16"

# Checkpointing
save_every: int = 1000
keep_checkpoints: int = 5

def validate(self):
    """Validate configuration parameters."""
    assert self.latent_dim > 0, "latent_dim must be positive"
    assert self.num_layers > 0, "num_layers must be positive"
    assert self.num_heads > 0 and self.latent_dim % self.num_heads == 0, \
        "latent_dim must be divisible by num_heads"
    assert self.hidden_dim > 0, "hidden_dim must be positive"
    assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
    assert self.diffusion_steps > 0, "diffusion_steps must be positive"
    assert self.guidance_scale >= 1.0, "guidance_scale must be >= 1.0"
    
    valid_activations = ["gelu", "relu", "silu", "leaky_relu"]
    assert self.activation in valid_activations, \
        f"activation must be one of {valid_activations}"
    
    valid_schedules = ["linear", "cosine", "sigmoid"]
    assert self.beta_schedule in valid_schedules, \
        f"beta_schedule must be one of {valid_schedules}"
    
    valid_variance_types = ["fixed_small", "fixed_large", "learned", "learned_range"]
    assert self.variance_type in valid_variance_types, \
        f"variance_type must be one of {valid_variance_types}"
    
    return True

def to_dict(self) -> Dict[str, Any]:
    """Convert configuration to dictionary."""
    return {
        k: v if not isinstance(v, dict) else v.copy()
        for k, v in self.__dict__.items()
    }

@classmethod
def from_dict(cls, config_dict: Dict[str, Any]):
    """Create configuration from dictionary."""
    return cls(**config_dict)

def __str__(self) -> str:
    """String representation of configuration."""
    lines = ["WorldModelConfig:"]
    for key, value in self.__dict__.items():
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            for subkey, subvalue in value.items():
                lines.append(f"    {subkey}: {subvalue}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)

class WorldModel(nn.Module):
    """
Main World Model for 3D world generation.
This model takes multimodal inputs (text, images, videos) and generates
3D world representations using diffusion transformers and Gaussian splatting.

Architecture:
    1. Multimodal encoders → Cross-attention fusion
    2. Diffusion transformer in latent space
    3. Parallel decoders for Gaussian/NeRF/Mesh representations
"""

def __init__(self, config: Union[WorldModelConfig, Dict[str, Any]]):
    """Initialize the World Model."""
    super().__init__()
    
    # Convert dict to config if needed
    if isinstance(config, dict):
        config = WorldModelConfig.from_dict(config)
    self.config = config
    self.config.validate()
    
    # Store device for later use
    self._device = None
    
    # Initialize submodules
    self._init_encoders()
    self._init_fusion_layers()
    self._init_diffusion_model()
    self._init_decoders()
    
    # Initialize weights
    self.apply(self._init_weights)
    
    # Mixed precision
    self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
    
def _init_encoders(self):
    """Initialize multimodal encoders."""
    from src.core.multimodal.encoders.text_encoder import TextEncoder
    from src.core.multimodal.encoders.vision_encoder import VisionEncoder
    from src.core.multimodal.encoders.video_encoder import VideoEncoder
    
    # Text encoder (CLIP or similar)
    self.text_encoder = TextEncoder(
        model_type="clip",
        output_dim=self.config.latent_dim,
        freeze_backbone=True,
    )
    
    # Vision encoder (ViT)
    self.vision_encoder = VisionEncoder(
        model_type="vit_large",
        output_dim=self.config.latent_dim,
        freeze_backbone=True,
    )
    
    # Video encoder (S3D or similar)
    self.video_encoder = VideoEncoder(
        model_type="s3d",
        output_dim=self.config.latent_dim,
        freeze_backbone=True,
    )
    
    # Projection layers to align dimensions
    self.text_proj = nn.Linear(
        self.text_encoder.output_dim,
        self.config.latent_dim
    )
    self.vision_proj = nn.Linear(
        self.vision_encoder.output_dim,
        self.config.latent_dim
    )
    self.video_proj = nn.Linear(
        self.video_encoder.output_dim,
        self.config.latent_dim
    )
    
def _init_fusion_layers(self):
    """Initialize cross-attention fusion layers."""
    from src.core.multimodal.fusion.cross_attention import CrossAttentionFusion
    
    # Cross-attention fusion for text-visual alignment
    self.text_vision_fusion = CrossAttentionFusion(
        dim=self.config.latent_dim,
        num_heads=self.config.num_heads,
        dropout=self.config.dropout,
    )
    
    # Temporal fusion for video inputs
    self.temporal_fusion = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=self.config.latent_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.hidden_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            batch_first=True,
        ),
        num_layers=2,
    )
    
    # Final projection to latent space
    self.latent_proj = nn.Sequential(
        nn.Linear(self.config.latent_dim * 2, self.config.latent_dim),
        AdaptiveLayerNorm(self.config.latent_dim),
        nn.Dropout(self.config.dropout),
        nn.Linear(self.config.latent_dim, self.config.latent_dim),
    )
    
def _init_diffusion_model(self):
    """Initialize diffusion transformer."""
    self.diffusion_model = DiffusionTransformer(
        latent_dim=self.config.latent_dim,
        num_layers=self.config.num_layers,
        num_heads=self.config.num_heads,
        hidden_dim=self.config.hidden_dim,
        dropout=self.config.dropout,
        activation=self.config.activation,
        num_timesteps=self.config.diffusion_steps,
        guidance_dim=self.config.latent_dim,
    )
    
def _init_decoders(self):
    """Initialize representation decoders."""
    from src.core.representation.gaussian_splatting import GaussianSplattingModel
    from src.core.representation.nerf import NeuralRadianceField
    from src.core.representation.mesh import MeshGenerator
    
    # Gaussian splatting decoder
    self.gaussian_decoder = GaussianSplattingModel(
        latent_dim=self.config.latent_dim,
        **self.config.gaussian_decoder,
    )
    
    # NeRF decoder
    self.nerf_decoder = NeuralRadianceField(
        latent_dim=self.config.latent_dim,
        **self.config.nerf_decoder,
    )
    
    # Mesh decoder
    self.mesh_decoder = MeshGenerator(
        latent_dim=self.config.latent_dim,
        **self.config.mesh_config if hasattr(self.config, 'mesh_config') else {},
    )
    
def _init_weights(self, module):
    """Initialize model weights."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

@property
def device(self):
    """Get the device of the model."""
    if self._device is None:
        # Get device from first parameter
        return next(self.parameters()).device
    return self._device

@device.setter
def device(self, value):
    """Set the device of the model."""
    self._device = value

def encode_text(self, text: Union[str, List[str]]) -> torch.Tensor:
    """
    Encode text input.
    
    Args:
        text: String or list of strings
    
    Returns:
        Text embeddings of shape (batch, latent_dim)
    """
    if isinstance(text, str):
        text = [text]
    
    with torch.no_grad():
        text_features = self.text_encoder(text)
        text_features = self.text_proj(text_features)
    
    return text_features

def encode_image(self, images: torch.Tensor) -> torch.Tensor:
    """
    Encode image input.
    
    Args:
        images: Tensor of shape (batch, channels, height, width)
    
    Returns:
        Image embeddings of shape (batch, latent_dim)
    """
    with torch.no_grad():
        image_features = self.vision_encoder(images)
        image_features = self.vision_proj(image_features)
    
    return image_features

def encode_video(self, videos: torch.Tensor) -> torch.Tensor:
    """
    Encode video input.
    
    Args:
        videos: Tensor of shape (batch, frames, channels, height, width)
    
    Returns:
        Video embeddings of shape (batch, latent_dim)
    """
    batch_size, num_frames = videos.shape[:2]
    
    # Encode each frame
    video_features = []
    for t in range(num_frames):
        frame_features = self.vision_encoder(videos[:, t])
        frame_features = self.vision_proj(frame_features)
        video_features.append(frame_features)
    
    video_features = torch.stack(video_features, dim=1)  # (batch, frames, latent_dim)
    
    # Temporal fusion
    with torch.no_grad():
        video_features = self.temporal_fusion(video_features)
        # Mean pool over time
        video_features = video_features.mean(dim=1)
    
    return video_features

def fuse_modalities(
    self,
    text_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fuse text and visual embeddings using cross-attention.
    
    Args:
        text_embeddings: Text embeddings of shape (batch, seq_len, latent_dim)
        visual_embeddings: Visual embeddings of shape (batch, latent_dim)
        attention_mask: Optional attention mask for text
    
    Returns:
        Fused embeddings of shape (batch, latent_dim)
    """
    # Expand visual embeddings to match text sequence length
    visual_embeddings = visual_embeddings.unsqueeze(1)  # (batch, 1, latent_dim)
    
    # Cross-attention fusion
    fused = self.text_vision_fusion(
        text_embeddings,
        visual_embeddings,
        attention_mask=attention_mask,
    )
    
    # Mean pool over sequence dimension
    if attention_mask is not None:
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        fused = (fused * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    else:
        fused = fused.mean(dim=1)
    
    # Project to latent space
    fused = self.latent_proj(fused)
    
    return fused

def forward(
    self,
    text_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    timesteps: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    guidance_scale: Optional[float] = None,
    return_latents: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Forward pass through the diffusion model.
    
    Args:
        text_embeddings: Text embeddings of shape (batch, latent_dim)
        visual_embeddings: Visual embeddings of shape (batch, latent_dim)
        timesteps: Diffusion timesteps of shape (batch,)
        noise: Optional noise input of shape (batch, latent_dim)
        guidance_scale: Optional classifier-free guidance scale
        return_latents: Whether to return intermediate latents
    
    Returns:
        Predicted noise or (predicted noise, intermediate latents)
    """
    # Fuse modalities
    fused_embeddings = self.fuse_modalities(
        text_embeddings.unsqueeze(1),  # Add sequence dimension
        visual_embeddings,
    )
    
    # Generate noise if not provided
    if noise is None:
        noise = torch.randn_like(fused_embeddings)
    
    # Apply diffusion model
    if guidance_scale is None:
        guidance_scale = self.config.guidance_scale
    
    # Classifier-free guidance
    if guidance_scale != 1.0 and self.training:
        # Randomly drop text embeddings for unconditional training
        drop_mask = torch.rand(text_embeddings.shape[0]) < 0.1
        text_embeddings[drop_mask] = 0
        
        # Forward pass with unconditional guidance
        model_output = self.diffusion_model(
            noise,
            timesteps,
            text_embeddings=text_embeddings,
            visual_embeddings=visual_embeddings,
            guidance_scale=guidance_scale,
        )
    else:
        # Standard forward pass
        model_output = self.diffusion_model(
            noise,
            timesteps,
            text_embeddings=text_embeddings,
            visual_embeddings=visual_embeddings,
        )
    
    if return_latents:
        return model_output, fused_embeddings
    return model_output

def decode_to_gaussian(
    self,
    latents: torch.Tensor,
    camera_positions: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Decode latents to Gaussian splatting representation.
    
    Args:
        latents: Latent vectors of shape (batch, latent_dim)
        camera_positions: Optional camera positions for view-dependent effects
        **kwargs: Additional decoder parameters
    
    Returns:
        Dictionary with Gaussian parameters
    """
    return self.gaussian_decoder(latents, camera_positions, **kwargs)

def decode_to_nerf(
    self,
    latents: torch.Tensor,
    rays: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Decode latents to NeRF representation.
    
    Args:
        latents: Latent vectors of shape (batch, latent_dim)
        rays: Optional ray origins and directions
        **kwargs: Additional decoder parameters
    
    Returns:
        Dictionary with NeRF parameters
    """
    return self.nerf_decoder(latents, rays, **kwargs)

def decode_to_mesh(
    self,
    latents: torch.Tensor,
    resolution: int = 256,
    **kwargs,
) -> Dict[str, Any]:
    """
    Decode latents to mesh representation.
    
    Args:
        latents: Latent vectors of shape (batch, latent_dim)
        resolution: Marching cubes resolution
        **kwargs: Additional decoder parameters
    
    Returns:
        Dictionary with mesh vertices and faces
    """
    return self.mesh_decoder(latents, resolution, **kwargs)

def generate_latents(
    self,
    text_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    num_samples: int = 1,
    guidance_scale: Optional[float] = None,
    seed: Optional[int] = None,
    progress_callback = None,
) -> torch.Tensor:
    """
    Generate latents via diffusion sampling.
    
    Args:
        text_embeddings: Text embeddings of shape (batch, latent_dim)
        visual_embeddings: Visual embeddings of shape (batch, latent_dim)
        num_samples: Number of samples to generate
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        progress_callback: Callback for progress updates
    
    Returns:
        Generated latents of shape (batch * num_samples, latent_dim)
    """
    from src.core.world_model.inference.sampler import DiffusionSampler
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create sampler
    sampler = DiffusionSampler(
        diffusion_model=self.diffusion_model,
        num_timesteps=self.config.diffusion_steps,
        beta_schedule=self.config.beta_schedule,
        variance_type=self.config.variance_type,
        device=self.device,
    )
    
    # Prepare inputs
    batch_size = text_embeddings.shape[0]
    text_embeddings = text_embeddings.repeat(num_samples, 1)
    visual_embeddings = visual_embeddings.repeat(num_samples, 1)
    
    if guidance_scale is None:
        guidance_scale = self.config.guidance_scale
    
    # Sample latents
    latents = sampler.sample(
        text_embeddings=text_embeddings,
        visual_embeddings=visual_embeddings,
        guidance_scale=guidance_scale,
        progress_callback=progress_callback,
    )
    
    return latents

@torch.no_grad()
def generate_world(
    self,
    prompt: Union[str, List[str]],
    images: Optional[torch.Tensor] = None,
    videos: Optional[torch.Tensor] = None,
    format: str = "gaussian",
    num_samples: int = 1,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    progress_callback = None,
    **decoder_kwargs,
) -> Dict[str, Any]:
    """
    Generate a complete 3D world from inputs.
    
    Args:
        prompt: Text description or list of descriptions
        images: Optional image tensor of shape (batch, channels, height, width)
        videos: Optional video tensor of shape (batch, frames, channels, height, width)
        format: Output format ("gaussian", "nerf", "mesh")
        num_samples: Number of samples to generate
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        progress_callback: Callback for progress updates
        **decoder_kwargs: Additional decoder parameters
    
    Returns:
        Dictionary containing the generated world representation
    """
    # Encode inputs
    if isinstance(prompt, str):
        prompt = [prompt]
    
    text_embeddings = self.encode_text(prompt)
    
    if images is not None:
        visual_embeddings = self.encode_image(images)
    elif videos is not None:
        visual_embeddings = self.encode_video(videos)
    else:
        # Use zero embeddings if no visual input
        batch_size = len(prompt)
        visual_embeddings = torch.zeros(
            batch_size, self.config.latent_dim, device=self.device
        )
    
    # Generate latents
    latents = self.generate_latents(
        text_embeddings=text_embeddings,
        visual_embeddings=visual_embeddings,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        seed=seed,
        progress_callback=progress_callback,
    )
    
    # Decode to specified format
    if format == "gaussian":
        result = self.decode_to_gaussian(latents, **decoder_kwargs)
    elif format == "nerf":
        result = self.decode_to_nerf(latents, **decoder_kwargs)
    elif format == "mesh":
        result = self.decode_to_mesh(latents, **decoder_kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Add metadata
    result["metadata"] = {
        "prompt": prompt,
        "format": format,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "latent_shape": latents.shape,
        "model_version": __version__,
    }
    
    return result

def edit_world(
    self,
    world_latents: torch.Tensor,
    edit_prompt: Union[str, List[str]],
    strength: float = 0.7,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Edit existing world latents with a new prompt.
    
    Args:
        world_latents: Original world latents of shape (batch, latent_dim)
        edit_prompt: New text description for editing
        strength: Editing strength (0-1)
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
        **kwargs: Additional sampling parameters
    
    Returns:
        Edited latents
    """
    from src.core.world_model.inference.sampler import DiffusionSampler
    
    if isinstance(edit_prompt, str):
        edit_prompt = [edit_prompt]
    
    # Encode edit prompt
    edit_embeddings = self.encode_text(edit_prompt)
    
    # Create sampler
    sampler = DiffusionSampler(
        diffusion_model=self.diffusion_model,
        num_timesteps=self.config.diffusion_steps,
        beta_schedule=self.config.beta_schedule,
        variance_type=self.config.variance_type,
        device=self.device,
    )
    
    # Determine starting timestep based on strength
    start_timestep = int(self.config.diffusion_steps * (1 - strength))
    
    # Add noise to latents based on strength
    if seed is not None:
        torch.manual_seed(seed)
    
    noise = torch.randn_like(world_latents)
    noisy_latents = sampler.add_noise(
        world_latents,
        noise,
        torch.tensor([start_timestep], device=self.device),
    )
    
    # Sample from noisy latents
    edited_latents = sampler.sample_from(
        noisy_latents,
        start_timestep=start_timestep,
        text_embeddings=edit_embeddings,
        visual_embeddings=torch.zeros_like(edit_embeddings),  # No visual guidance for edits
        guidance_scale=guidance_scale,
        **kwargs,
    )
    
    return edited_latents

def compute_loss(
    self,
    text_embeddings: torch.Tensor,
    visual_embeddings: torch.Tensor,
    target_latents: torch.Tensor,
    timesteps: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Compute training loss.
    
    Args:
        text_embeddings: Text embeddings
        visual_embeddings: Visual embeddings
        target_latents: Target latent vectors
        timesteps: Optional diffusion timesteps
        noise: Optional noise input
        **kwargs: Additional loss parameters
    
    Returns:
        Dictionary of loss values
    """
    from src.core.world_model.training.loss_functions import WorldModelLoss
    
    # Create loss function
    loss_fn = WorldModelLoss(
        diffusion_steps=self.config.diffusion_steps,
        loss_type="l2",
        **kwargs,
    )
    
    # Sample timesteps if not provided
    if timesteps is None:
        batch_size = text_embeddings.shape[0]
        timesteps = torch.randint(
            0, self.config.diffusion_steps,
            (batch_size,), device=self.device,
        )
    
    # Sample noise if not provided
    if noise is None:
        noise = torch.randn_like(target_latents)
    
    # Add noise to targets
    noisy_targets, actual_noise = self.diffusion_model.add_noise(
        target_latents, noise, timesteps
    )
    
    # Predict noise
    pred_noise = self.forward(
        text_embeddings,
        visual_embeddings,
        timesteps,
        noisy_targets,
    )
    
    # Compute losses
    losses = loss_fn(
        pred_noise=pred_noise,
        target_noise=actual_noise,
        latents=target_latents,
        timesteps=timesteps,
    )
    
    return losses

def save_checkpoint(
    self,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    step: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save model checkpoint.
    
    Args:
        path: Path to save checkpoint
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        step: Current training step
        metadata: Optional metadata to save
    """
    checkpoint = {
        "model_state_dict": self.state_dict(),
        "config": self.config.to_dict(),
        "epoch": epoch,
        "step": step,
        "version": __version__,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

@classmethod
def load_checkpoint(
    cls,
    path: str,
    device: str = None,
    map_location: str = "cpu",
    strict: bool = True,
    **kwargs,
) -> "WorldModel":
    """
    Load model from checkpoint.
    
    Args:
        path: Path to checkpoint file
        device: Device to load model on
        map_location: How to remap storage locations
        strict: Whether to strictly enforce state_dict keys match
        **kwargs: Additional arguments to pass to model constructor
    
    Returns:
        Loaded WorldModel instance
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=map_location)
    
    # Create config
    config_dict = checkpoint.get("config", {})
    config_dict.update(kwargs)  # Override with kwargs
    config = WorldModelConfig.from_dict(config_dict)
    
    # Create model
    model = cls(config)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Move to device
    if device is not None:
        model.to(device)
    
    print(f"Loaded checkpoint from {path}")
    print(f"  Version: {checkpoint.get('version', 'unknown')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    
    return model

def get_num_parameters(self, trainable_only: bool = False) -> int:
    """
    Get the number of parameters in the model.
    
    Args:
        trainable_only: Whether to count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in self.parameters())

def get_memory_usage(self) -> Dict[str, float]:
    """
    Get memory usage of the model.
    
    Returns:
        Dictionary with memory usage in GB
    """
    param_memory = sum(
        p.numel() * p.element_size()
        for p in self.parameters()
    ) / 1e9  # GB
    
    buffer_memory = sum(
        b.numel() * b.element_size()
        for b in self.buffers()
    ) / 1e9  # GB
    
    return {
        "parameters_gb": param_memory,
        "buffers_gb": buffer_memory,
        "total_gb": param_memory + buffer_memory,
    }

def summary(self) -> str:
    """
    Get a summary of the model architecture.
    
    Returns:
        Formatted summary string
    """
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        print("=" * 80)
        print("World Model Summary")
        print("=" * 80)
        print(f"Version: {__version__}")
        print(f"Author: {__author__}")
        print()
        
        print("Configuration:")
        print(self.config)
        print()
        
        print("Model Architecture:")
        print(f"  Total parameters: {self.get_num_parameters():,}")
        print(f"  Trainable parameters: {self.get_num_parameters(trainable_only=True):,}")
        print()
        
        print("Submodules:")
        print(f"  Text Encoder: {self.text_encoder.__class__.__name__}")
        print(f"  Vision Encoder: {self.vision_encoder.__class__.__name__}")
        print(f"  Video Encoder: {self.video_encoder.__class__.__name__}")
        print(f"  Diffusion Model: {self.diffusion_model.__class__.__name__}")
        print(f"  Gaussian Decoder: {self.gaussian_decoder.__class__.__name__}")
        print(f"  NeRF Decoder: {self.nerf_decoder.__class__.__name__}")
        print(f"  Mesh Decoder: {self.mesh_decoder.__class__.__name__}")
        print()
        
        print("Memory Usage:")
        memory = self.get_memory_usage()
        for key, value in memory.items():
            print(f"  {key}: {value:.2f} GB")
        print("=" * 80)
    
    return f.getvalue()


#Factory functions#

def create_world_model(
config: Union[WorldModelConfig, Dict[str, Any]] = None,
pretrained: bool = True,
device: str = None,
**kwargs,
) -> WorldModel:
    """
Create a WorldModel instance.

Args:
    config: Model configuration
    pretrained: Whether to load pretrained weights
    device: Device to load model on
    **kwargs: Additional configuration parameters

Returns:
    WorldModel instance
"""
# Set default device
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Create config
if config is None:
    config = WorldModelConfig()

# Update config with kwargs
if kwargs:
    if isinstance(config, dict):
        config.update(kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

# Create model
model = WorldModel(config)

# Load pretrained weights if requested
if pretrained:
    try:
        model = load_pretrained(model, device)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
        print("Using randomly initialized model.")

# Move to device
model.to(device)
model.device = device

return model

def load_pretrained(
model: WorldModel = None,
device: str = None,
version: str = "v1.0",
) -> WorldModel:
  """
Load pretrained weights for the WorldModel.

Args:
    model: Existing model instance (optional)
    device: Device to load model on
    version: Model version to load

Returns:
    Model with pretrained weights
"""
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Model URLs (these would be actual URLs in production)
model_urls = {
    "v1.0": "https://models.frextech-sim.com/world_model_v1.0.pt",
    "v1.1": "https://models.frextech-sim.com/world_model_v1.1.pt",
    "latest": "https://models.frextech-sim.com/world_model_v1.1.pt",
}

if version not in model_urls:
    raise ValueError(f"Unknown model version: {version}. Available: {list(model_urls.keys())}")

# Download or load pretrained weights
import tempfile
import requests
from pathlib import Path

# Check cache first
cache_dir = Path.home() / ".frextech" / "models"
cache_dir.mkdir(parents=True, exist_ok=True)

model_filename = f"world_model_{version}.pt"
model_path = cache_dir / model_filename

if not model_path.exists():
    print(f"Downloading pretrained model {version}...")
    url = model_urls[version]
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as f:
        from tqdm import tqdm
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

# Load checkpoint
checkpoint = torch.load(model_path, map_location=device)

# Create or load model
if model is None:
    # Create new model from checkpoint config
    model = WorldModel.load_checkpoint(str(model_path), device=device)
else:
    # Load weights into existing model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.device = device

print(f"Loaded pretrained model {version}")
return model

High-level API functions

def generate_world(
prompt: Union[str, List[str]],
model: Optional[WorldModel] = None,
images: Optional[torch.Tensor] = None,
videos: Optional[torch.Tensor] = None,
format: str = "gaussian",
**kwargs,
) -> Dict[str, Any]:
   """
#High-level function to generate a 3D world.#

Args:
    prompt: Text description or list of descriptions
    model: Optional WorldModel instance (will load default if not provided)
    images: Optional input images
    videos: Optional input videos
    format: Output format ("gaussian", "nerf", "mesh")
    **kwargs: Additional generation parameters

Returns:
    Generated world representation
"""
# Load default model if not provided
if model is None:
    model = create_world_model(pretrained=True)

# Generate world
result = model.generate_world(
    prompt=prompt,
    images=images,
    videos=videos,
    format=format,
    **kwargs,
)

return result

def edit_world(
world_latents: torch.Tensor,
edit_prompt: Union[str, List[str]],
model: Optional[WorldModel] = None,
**kwargs,
) -> torch.Tensor:
    """
High-level function to edit existing world latents.

Args:
    world_latents: Original world latents
    edit_prompt: New text description for editing
    model: Optional WorldModel instance
    **kwargs: Additional editing parameters

Returns:
    Edited latents
"""
# Load default model if not provided
if model is None:
    model = create_world_model(pretrained=True)

# Edit world
edited_latents = model.edit_world(
    world_latents=world_latents,
    edit_prompt=edit_prompt,
    **kwargs,
)

return edited_latents

#Utility functions#

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_available_models() -> List[str]:
    """Get list of available pretrained models."""
return ["v1.0", "v1.1", "latest"]

#Test the module#

if __name__ == "__main__":
    print("Testing World Model module...")

    # Create a minimal config for testing
test_config = WorldModelConfig(
    latent_dim=128,
    num_layers=4,
    num_heads=8,
    hidden_dim=512,
    diffusion_steps=100,
)

print("Configuration:")
print(test_config)

# Create model
model = WorldModel(test_config)
print(f"\nModel created with {model.get_num_parameters():,} parameters")

# Test forward pass
batch_size = 2
text_embeddings = torch.randn(batch_size, test_config.latent_dim)
visual_embeddings = torch.randn(batch_size, test_config.latent_dim)
timesteps = torch.randint(0, test_config.diffusion_steps, (batch_size,))

output = model(text_embeddings, visual_embeddings, timesteps)
print(f"\nForward pass output shape: {output.shape}")

# Test generation (without actual decoding)
print("\nTesting generation...")
prompt = ["A beautiful mountain landscape"]
try:
    # This will fail if encoders aren't available, but we'll catch it
    result = model.generate_world(prompt, format="gaussian", num_samples=1)
    print("Generation successful!")
except Exception as e:
    print(f"Generation test failed (expected for minimal test): {e}")

print("\nWorld Model module test complete!")