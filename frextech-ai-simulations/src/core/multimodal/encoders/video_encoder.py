"""
Video encoders for processing video inputs.
Includes CNN3D, Video Transformer, TimeSformer, and VideoMAE encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
import logging
import warnings
from transformers import (
    TimesformerModel,
    TimesformerImageProcessor,
    VideoMAEModel,
    VideoMAEImageProcessor,
    AutoModel,
    AutoImageProcessor,
    AutoFeatureExtractor
)
import torchvision
from torchvision import models, transforms

from ..alignment import EncoderConfig, EncoderOutput

logger = logging.getLogger(__name__)

@dataclass
class VideoEncoderConfig(EncoderConfig):
    """Configuration for video encoders."""
    
    # Video-specific parameters
    num_frames: int = 16
    frame_size: Tuple[int, int] = (224, 224)  # (height, width)
    frame_rate: int = 30
    
    # CNN parameters
    cnn_arch: str = "r3d_18"  # "r3d_18", "mc3_18", "r2plus1d_18"
    pretrained_cnn: bool = True
    cnn_pooling: str = "avg"  # "avg", "max", "adaptive"
    
    # Transformer parameters
    patch_size: Tuple[int, int] = (16, 16)
    num_patches: int = 196  # (224/16)^2
    
    # Temporal parameters
    temporal_stride: int = 1
    temporal_pooling: str = "mean"  # "mean", "max", "attention", "last"
    
    # Feature extraction
    feature_layer: int = -1  # Which layer to extract features from
    extract_spatial_features: bool = False
    
    # Augmentation
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    random_crop: bool = True
    random_flip: bool = True
    
    def __post_init__(self):
        """Set encoder type."""
        self.encoder_type = "video"
        
        # Calculate num_patches if not provided
        if self.num_patches is None:
            h, w = self.frame_size
            patch_h, patch_w = self.patch_size
            self.num_patches = (h // patch_h) * (w // patch_w)

class VideoEncoder(nn.Module):
    """Base class for video encoders."""
    
    def __init__(self, config: VideoEncoderConfig):
        super().__init__()
        self.config = config
        self.preprocessor = None
        self.model = None
        self.output_dim = config.output_dim
        
        # Initialize preprocessing transforms
        self._init_transforms()
        
        # Initialize model
        self._initialize_model()
        
        # Freeze parameters if not trainable
        if not config.trainable:
            self._freeze_parameters()
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {config.model_name}")
    
    def _init_transforms(self):
        """Initialize video preprocessing transforms."""
        # Training transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.config.frame_size),
            transforms.RandomHorizontalFlip() if self.config.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
        
        # Validation transforms
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.config.frame_size),
            transforms.CenterCrop(self.config.frame_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            )
        ])
    
    def _initialize_model(self):
        """Initialize model - to be implemented by subclasses."""
        raise NotImplementedError
    
    def _freeze_parameters(self):
        """Freeze model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Froze all model parameters")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
    
    def preprocess(self,
                  video: Union[torch.Tensor, np.ndarray, List, str],
                  is_training: bool = False,
                  **kwargs) -> torch.Tensor:
        """
        Preprocess video for model input.
        
        Args:
            video: Input video - can be tensor, numpy array, list of frames, or file path
            is_training: Whether to use training transforms
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed video tensor
        """
        transforms_to_use = self.train_transforms if is_training else self.val_transforms
        
        # Handle different input types
        if isinstance(video, str):
            # Load video from file
            try:
                video = self._load_video_file(video)
            except Exception as e:
                logger.error(f"Failed to load video file {video}: {e}")
                raise
        
        if isinstance(video, np.ndarray):
            # Convert numpy array to tensor
            video = torch.from_numpy(video).float()
        
        if isinstance(video, list):
            # List of frames (PIL Images or tensors)
            frames = []
            for frame in video:
                if isinstance(frame, torch.Tensor):
                    # Assume already preprocessed
                    frames.append(frame)
                else:
                    # Apply transforms
                    frame_transformed = transforms_to_use(frame)
                    frames.append(frame_transformed)
            video = torch.stack(frames)
        
        # Ensure video has correct dimensions
        # Expected: [batch_size, num_frames, channels, height, width] or [num_frames, channels, height, width]
        if video.dim() == 4:
            # Single video: [num_frames, channels, height, width]
            video = video.unsqueeze(0)  # Add batch dimension
        
        elif video.dim() == 5:
            # Batch of videos: [batch_size, num_frames, channels, height, width]
            pass
        
        else:
            raise ValueError(f"Video must be 4D or 5D tensor, got shape: {video.shape}")
        
        # Use preprocessor if available
        if self.preprocessor is not None:
            try:
                # Convert to numpy for HuggingFace processors
                video_np = video.numpy() if video.device == torch.device('cpu') else video.cpu().numpy()
                
                # Process with HuggingFace preprocessor
                processed = self.preprocessor(
                    video_np,
                    return_tensors="pt",
                    **kwargs
                )
                return processed['pixel_values']
            except Exception as e:
                logger.warning(f"Preprocessor failed: {e}, using manual preprocessing")
        
        return video
    
    def _load_video_file(self, filepath: str) -> torch.Tensor:
        """Load video from file."""
        try:
            # Try torchvision first
            import torchvision.io as io
            video, _, info = io.read_video(filepath, pts_unit='sec')
            return video.float() / 255.0  # Normalize to [0, 1]
        except:
            # Fallback: use OpenCV
            import cv2
            cap = cv2.VideoCapture(filepath)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).float() / 255.0
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames read from video file: {filepath}")
            
            return torch.stack(frames)
    
    def forward(self,
                video: Union[torch.Tensor, np.ndarray, List, str],
                is_training: bool = False,
                return_dict: bool = True,
                **kwargs) -> EncoderOutput:
        """
        Forward pass through video encoder.
        
        Args:
            video: Input video
            is_training: Whether to use training mode
            return_dict: Whether to return EncoderOutput dict
            **kwargs: Additional arguments for model
            
        Returns:
            EncoderOutput with video features
        """
        # Preprocess video
        pixel_values = self.preprocess(video, is_training, **kwargs)
        pixel_values = pixel_values.to(self.model.device)
        
        # Model forward pass
        model_kwargs = {
            'pixel_values': pixel_values,
            'output_hidden_states': True,
            'output_attentions': True,
            'return_dict': True
        }
        
        # Update with any additional kwargs
        model_kwargs.update(kwargs)
        
        outputs = self.model(**model_kwargs)
        
        # Extract features
        if hasattr(outputs, 'last_hidden_state'):
            # Transformer model
            features = outputs.last_hidden_state
            
            # Handle temporal dimension
            if features.dim() == 4:
                # [batch_size, num_frames * num_patches, hidden_dim]
                # Reshape to separate temporal and spatial dimensions
                batch_size, total_tokens, hidden_dim = features.shape
                num_frames = self.config.num_frames
                num_patches = total_tokens // num_frames
                
                features = features.view(batch_size, num_frames, num_patches, hidden_dim)
            
            # Pool over temporal dimension
            pooled_temporal = self._temporal_pooling(features)
            
            # Pool over spatial dimension (if needed)
            if not self.config.extract_spatial_features:
                # Mean pool over patches
                pooled = pooled_temporal.mean(dim=1) if pooled_temporal.dim() == 3 else pooled_temporal
            else:
                # Keep spatial features
                pooled = pooled_temporal
            
        elif hasattr(outputs, 'pooler_output'):
            # Some models have pooler_output
            pooled = outputs.pooler_output
            features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else pooled.unsqueeze(1)
        
        elif hasattr(outputs, 'features'):
            # CNN models
            features = outputs.features
            pooled = self._pool_cnn_features(features)
        
        else:
            # Last resort
            if hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1]
                pooled = self._temporal_pooling(features)
            else:
                raise ValueError("Could not extract features from model output")
        
        # Normalize if requested
        if self.config.normalize_features:
            if pooled.dim() == 3:
                # Normalize each frame separately
                pooled = F.normalize(pooled, p=2, dim=-1)
            else:
                pooled = F.normalize(pooled, p=2, dim=-1)
        
        if return_dict:
            return EncoderOutput(
                features=features,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                pooled_features=pooled,
                mask=None,
                metadata={
                    'pixel_values_shape': pixel_values.shape,
                    'model_name': self.config.model_name,
                    'num_frames': self.config.num_frames,
                    'frame_size': self.config.frame_size
                }
            )
        else:
            return features, pooled
    
    def _temporal_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool features over temporal dimension.
        
        Args:
            features: Features [batch_size, num_frames, ...]
            
        Returns:
            Temporally pooled features
        """
        if features.dim() == 3:
            # [batch_size, num_frames, hidden_dim]
            if self.config.temporal_pooling == "mean":
                pooled = torch.mean(features, dim=1)
            elif self.config.temporal_pooling == "max":
                pooled = torch.max(features, dim=1)[0]
            elif self.config.temporal_pooling == "attention":
                pooled = self._temporal_attention_pooling(features)
            elif self.config.temporal_pooling == "last":
                pooled = features[:, -1, :]
            else:
                raise ValueError(f"Unknown temporal pooling: {self.config.temporal_pooling}")
        
        elif features.dim() == 4:
            # [batch_size, num_frames, num_patches, hidden_dim]
            if self.config.temporal_pooling == "mean":
                pooled = torch.mean(features, dim=1)  # [batch_size, num_patches, hidden_dim]
            elif self.config.temporal_pooling == "max":
                pooled = torch.max(features, dim=1)[0]
            elif self.config.temporal_pooling == "attention":
                # Apply attention over temporal dimension for each patch
                batch_size, num_frames, num_patches, hidden_dim = features.shape
                features_flat = features.view(batch_size * num_patches, num_frames, hidden_dim)
                pooled_flat = self._temporal_attention_pooling(features_flat)
                pooled = pooled_flat.view(batch_size, num_patches, hidden_dim)
            elif self.config.temporal_pooling == "last":
                pooled = features[:, -1, :, :]
            else:
                raise ValueError(f"Unknown temporal pooling: {self.config.temporal_pooling}")
        
        else:
            raise ValueError(f"Unexpected feature dimension: {features.dim()}")
        
        return pooled
    
    def _temporal_attention_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Attention-based temporal pooling."""
        batch_size, num_frames, hidden_dim = features.shape
        
        # Learnable attention weights
        if not hasattr(self, 'temporal_attention'):
            self.temporal_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            ).to(features.device)
        
        # Compute attention scores
        attention_scores = self.temporal_attention(features).squeeze(-1)  # [batch_size, num_frames]
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        
        # Weighted sum
        pooled = torch.sum(features * attention_weights, dim=1)
        
        return pooled
    
    def _pool_cnn_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pool CNN features."""
        if self.config.cnn_pooling == "avg":
            # Global average pooling
            if features.dim() == 5:
                # 3D CNN output: [batch_size, channels, depth, height, width]
                pooled = F.adaptive_avg_pool3d(features, (1, 1, 1)).squeeze(-1).squeeze(-1).squeeze(-1)
            else:
                pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        
        elif self.config.cnn_pooling == "max":
            # Global max pooling
            if features.dim() == 5:
                pooled = F.adaptive_max_pool3d(features, (1, 1, 1)).squeeze(-1).squeeze(-1).squeeze(-1)
            else:
                pooled = F.adaptive_max_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        
        elif self.config.cnn_pooling == "adaptive":
            # Adaptive pooling to fixed size
            target_size = (self.config.output_dim // 8, 7, 7) if features.dim() == 5 else (7, 7)
            if features.dim() == 5:
                pooled = F.adaptive_avg_pool3d(features, target_size)
            else:
                pooled = F.adaptive_avg_pool2d(features, target_size)
            pooled = pooled.view(pooled.size(0), -1)
        
        else:
            raise ValueError(f"Unknown CNN pooling: {self.config.cnn_pooling}")
        
        return pooled
    
    def extract_spatial_features(self,
                                video: Union[torch.Tensor, np.ndarray, List, str],
                                frame_idx: Optional[int] = None,
                                is_training: bool = False) -> torch.Tensor:
        """
        Extract spatial features from specific frame(s).
        
        Args:
            video: Input video
            frame_idx: Specific frame index (None for all frames)
            is_training: Whether to use training mode
            
        Returns:
            Spatial features
        """
        # Preprocess video
        pixel_values = self.preprocess(video, is_training)
        pixel_values = pixel_values.to(self.model.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(pixel_values, output_hidden_states=True, return_dict=True)
        
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
            
            # Reshape to separate spatial and temporal dimensions
            batch_size, total_tokens, hidden_dim = features.shape
            num_frames = self.config.num_frames
            num_patches = total_tokens // num_frames
            
            features = features.view(batch_size, num_frames, num_patches, hidden_dim)
            
            # Select specific frame if requested
            if frame_idx is not None:
                features = features[:, frame_idx, :, :]
            
            return features
        
        else:
            raise ValueError("Model does not support spatial feature extraction")
    
    def extract_temporal_features(self,
                                 video: Union[torch.Tensor, np.ndarray, List, str],
                                 is_training: bool = False) -> torch.Tensor:
        """
        Extract temporal features.
        
        Args:
            video: Input video
            is_training: Whether to use training mode
            
        Returns:
            Temporal features
        """
        output = self.forward(video, is_training=is_training, return_dict=False)
        features, _ = output
        
        # If features have spatial dimension, pool over it
        if features.dim() == 4:
            # [batch_size, num_frames, num_patches, hidden_dim]
            features = features.mean(dim=2)  # Mean pool over patches
        
        return features
    
    def batch_encode(self,
                    video_list: List[Union[torch.Tensor, np.ndarray, List, str]],
                    batch_size: int = 2,
                    is_training: bool = False,
                    **kwargs) -> List[EncoderOutput]:
        """
        Encode a batch of videos.
        
        Args:
            video_list: List of video inputs
            batch_size: Batch size for processing
            is_training: Whether to use training mode
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of encoder outputs
        """
        outputs = []
        
        for i in range(0, len(video_list), batch_size):
            batch_videos = video_list[i:i + batch_size]
            output = self(batch_videos, is_training=is_training, **kwargs)
            outputs.append(output)
        
        # Concatenate if multiple batches
        if len(outputs) > 1:
            # Concatenate features
            all_features = torch.cat([out.features for out in outputs], dim=0)
            all_pooled = torch.cat([out.pooled_features for out in outputs], dim=0)
            
            # Handle other outputs
            if outputs[0].hidden_states is not None:
                all_hidden = []
                num_layers = len(outputs[0].hidden_states)
                for layer_idx in range(num_layers):
                    layer_hidden = torch.cat(
                        [out.hidden_states[layer_idx] for out in outputs], dim=0
                    )
                    all_hidden.append(layer_hidden)
            else:
                all_hidden = None
            
            if outputs[0].attentions is not None:
                all_attentions = []
                num_layers = len(outputs[0].attentions)
                for layer_idx in range(num_layers):
                    layer_attn = torch.cat(
                        [out.attentions[layer_idx] for out in outputs], dim=0
                    )
                    all_attentions.append(layer_attn)
            else:
                all_attentions = None
            
            output = EncoderOutput(
                features=all_features,
                hidden_states=all_hidden,
                attentions=all_attentions,
                pooled_features=all_pooled,
                mask=None,
                metadata={'batch_size': len(video_list)}
            )
            return [output]
        
        return outputs
    
    def save(self, path: str):
        """Save encoder to file."""
        torch.save({
            'config': self.config,
            'model_state_dict': self.model.state_dict()
        }, path)
        logger.info(f"Saved encoder to {path}")
    
    def load(self, path: str):
        """Load encoder from file."""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded encoder from {path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        stats = {
            'model_name': self.config.model_name,
            'output_dim': self.output_dim,
            'trainable': self.config.trainable,
            'num_frames': self.config.num_frames,
            'frame_size': self.config.frame_size,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
        
        # Add model-specific stats
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'hidden_size'):
                stats['hidden_size'] = config.hidden_size
            if hasattr(config, 'num_hidden_layers'):
                stats['num_layers'] = config.num_hidden_layers
            if hasattr(config, 'num_attention_heads'):
                stats['num_heads'] = config.num_attention_heads
        
        return stats

class VideoTransformerEncoder(VideoEncoder):
    """Generic video transformer encoder."""
    
    def _initialize_model(self):
        """Initialize video transformer model."""
        try:
            self.model = AutoModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            if hasattr(self.model.config, 'hidden_size'):
                self.output_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, 'd_model'):
                self.output_dim = self.model.config.d_model
            else:
                self.output_dim = self.config.output_dim
            
        except Exception as e:
            logger.error(f"Failed to load video transformer model {self.config.model_name}: {e}")
            raise

class TimesformerEncoder(VideoEncoder):
    """TimeSformer video encoder."""
    
    def _initialize_model(self):
        """Initialize TimeSformer model and preprocessor."""
        try:
            self.preprocessor = TimesformerImageProcessor.from_pretrained(
                self.config.model_name
            )
            self.model = TimesformerModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load TimeSformer model {self.config.model_name}: {e}")
            raise

class VideoMAEEncoder(VideoEncoder):
    """VideoMAE encoder (masked autoencoder for video)."""
    
    def _initialize_model(self):
        """Initialize VideoMAE model and preprocessor."""
        try:
            self.preprocessor = VideoMAEImageProcessor.from_pretrained(
                self.config.model_name
            )
            self.model = VideoMAEModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load VideoMAE model {self.config.model_name}: {e}")
            raise

class CNN3DEncoder(VideoEncoder):
    """3D CNN video encoder."""
    
    def _initialize_model(self):
        """Initialize 3D CNN model."""
        try:
            import torchvision.models.video as video_models
            
            # Map architecture names to model classes
            model_map = {
                'r3d_18': video_models.r3d_18,
                'mc3_18': video_models.mc3_18,
                'r2plus1d_18': video_models.r2plus1d_18,
                'r3d_50': video_models.r3d_50,
                'r2plus1d_50': video_models.r2plus1d_50,
            }
            
            if self.config.cnn_arch not in model_map:
                raise ValueError(f"Unsupported 3D CNN architecture: {self.config.cnn_arch}")
            
            # Load pretrained model
            self.model = model_map[self.config.cnn_arch](pretrained=self.config.pretrained_cnn)
            
            # Remove classification head
            if hasattr(self.model, 'fc'):
                # Store output dimension before removing fc
                self.output_dim = self.model.fc.in_features
                self.model.fc = nn.Identity()
            elif hasattr(self.model, 'head'):
                self.output_dim = self.model.head.in_features
                self.model.head = nn.Identity()
            else:
                # Try to infer output dimension
                self.output_dim = 512  # Default
            
        except Exception as e:
            logger.error(f"Failed to load 3D CNN model {self.config.cnn_arch}: {e}")
            raise
    
    def forward(self, video, is_training=False, **kwargs):
        """3D CNN forward pass."""
        # Preprocess video
        pixel_values = self.preprocess(video, is_training, **kwargs)
        pixel_values = pixel_values.to(self.model.device)
        
        # 3D CNN expects [batch_size, channels, depth, height, width]
        # Our preprocessing returns [batch_size, depth, channels, height, width]
        # So we need to permute dimensions
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)  # [batch_size, channels, num_frames, height, width]
        
        # Forward through model
        features = self.model(pixel_values)
        
        # For 3D CNN, features are already pooled
        pooled = features
        
        # Create dummy sequence features for consistency
        features_seq = pooled.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Normalize if requested
        if self.config.normalize_features:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return EncoderOutput(
            features=features_seq,
            hidden_states=None,
            attentions=None,
            pooled_features=pooled,
            mask=None,
            metadata={
                'model_name': self.config.cnn_arch,
                'num_frames': self.config.num_frames,
                'frame_size': self.config.frame_size
            }
        )

# Example usage
if __name__ == "__main__":
    # Test different video encoders
    encoders_to_test = [
        ('facebook/timesformer-base-finetuned-k400', TimesformerEncoder),
        ('MCG-NJU/videomae-base', VideoMAEEncoder),
        ('r3d_18', CNN3DEncoder)
    ]
    
    # Create a dummy video
    num_frames = 8
    height, width = 224, 224
    dummy_video = torch.randn(num_frames, 3, height, width)  # [num_frames, channels, height, width]
    
    for model_name, encoder_class in encoders_to_test:
        try:
            print(f"\nTesting {encoder_class.__name__} with {model_name}:")
            
            config = VideoEncoderConfig(
                model_name=model_name,
                pretrained=True,
                trainable=False,
                num_frames=num_frames,
                frame_size=(height, width)
            )
            
            encoder = encoder_class(config)
            
            # Test video encoding
            output = encoder(dummy_video, is_training=False)
            print(f"  Output shape: {output.features.shape}")
            print(f"  Pooled shape: {output.pooled_features.shape}")
            
            # Test spatial features
            if hasattr(encoder, 'extract_spatial_features'):
                spatial_features = encoder.extract_spatial_features(dummy_video)
                print(f"  Spatial features shape: {spatial_features.shape}")
            
            # Test temporal features
            if hasattr(encoder, 'extract_temporal_features'):
                temporal_features = encoder.extract_temporal_features(dummy_video)
                print(f"  Temporal features shape: {temporal_features.shape}")
            
            # Get statistics
            stats = encoder.get_statistics()
            print(f"  Parameters: {stats['num_parameters']:,}")
            print(f"  Output dim: {stats['output_dim']}")
            
        except Exception as e:
            print(f"  Error: {e}")