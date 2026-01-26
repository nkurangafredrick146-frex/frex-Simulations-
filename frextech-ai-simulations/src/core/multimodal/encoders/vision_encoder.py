"""
Vision encoders for processing image inputs.
Includes CNN-based (ResNet, EfficientNet) and transformer-based (ViT, CLIP, DINO) encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
import logging
from PIL import Image
import torchvision.transforms as transforms
from transformers import (
    ViTModel,
    ViTImageProcessor,
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoImageProcessor,
    AutoModel
)

from ..alignment import EncoderConfig, EncoderOutput

logger = logging.getLogger(__name__)

@dataclass
class VisionEncoderConfig(EncoderConfig):
    """Configuration for vision encoders."""
    
    # Vision-specific parameters
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_patches: int = 196  # (224/16)^2
    
    # CNN parameters
    cnn_arch: str = "resnet50"
    pretrained_cnn: bool = True
    remove_fc: bool = True
    
    # Transformer parameters
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    
    # Augmentation
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Feature extraction
    feature_layer: int = -1  # Which layer to extract features from
    return_patch_features: bool = False
    
    def __post_init__(self):
        """Set encoder type."""
        self.encoder_type = "vision"
        
        # Calculate num_patches if not provided
        if self.num_patches is None:
            self.num_patches = (self.image_size // self.patch_size) ** 2

class VisionEncoder(nn.Module):
    """Base class for vision encoders."""
    
    def __init__(self, config: VisionEncoderConfig):
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
        """Initialize image preprocessing transforms."""
        self.transforms = transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
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
                  images: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray],
                  **kwargs) -> torch.Tensor:
        """
        Preprocess image(s) for model input.
        
        Args:
            images: Input image(s) in various formats
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed tensor [batch_size, channels, height, width]
        """
        if self.preprocessor is not None:
            # Use HuggingFace preprocessor if available
            return self.preprocessor(images, return_tensors="pt")["pixel_values"]
        
        # Manual preprocessing
        if isinstance(images, Image.Image):
            images = [images]
        
        if isinstance(images, list):
            # List of PIL Images or tensors
            processed = []
            for img in images:
                if isinstance(img, Image.Image):
                    processed.append(self.transforms(img))
                elif isinstance(img, torch.Tensor):
                    # Assume already preprocessed
                    processed.append(img)
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                    processed.append(self.transforms(img))
                else:
                    raise TypeError(f"Unsupported image type: {type(img)}")
            
            if processed:
                return torch.stack(processed)
            else:
                raise ValueError("Empty image list")
        
        elif isinstance(images, torch.Tensor):
            # Assume batch of tensors
            if images.dim() == 3:
                # Single image [C, H, W]
                images = images.unsqueeze(0)
            return images
        
        elif isinstance(images, np.ndarray):
            # Numpy array
            if images.ndim == 3:
                # Single image [H, W, C] or [C, H, W]
                images = torch.from_numpy(images).float()
                if images.shape[0] != 3:  # Assume [H, W, C]
                    images = images.permute(2, 0, 1)
                images = images.unsqueeze(0)
            elif images.ndim == 4:
                # Batch of images
                images = torch.from_numpy(images).float()
                if images.shape[1] != 3:  # Assume [N, H, W, C]
                    images = images.permute(0, 3, 1, 2)
            return images
        
        else:
            raise TypeError(f"Unsupported input type: {type(images)}")
    
    def forward(self,
                images: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray],
                return_dict: bool = True,
                **kwargs) -> EncoderOutput:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Input image(s)
            return_dict: Whether to return EncoderOutput dict
            **kwargs: Additional arguments for model
            
        Returns:
            EncoderOutput with image features
        """
        # Preprocess images
        pixel_values = self.preprocess(images, **kwargs)
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
            # Transformer model (ViT, CLIP)
            features = outputs.last_hidden_state
            
            # Get CLS token or pool patch tokens
            if not self.config.return_patch_features:
                # Use CLS token (first token)
                cls_features = features[:, 0, :]
                patch_features = features[:, 1:, :] if features.shape[1] > 1 else None
            else:
                cls_features = features[:, 0, :]
                patch_features = features[:, 1:, :] if features.shape[1] > 1 else None
            
        elif hasattr(outputs, 'pooler_output'):
            # Some models have pooler_output (e.g., some ViT variants)
            cls_features = outputs.pooler_output
            patch_features = None
            
        elif hasattr(outputs, 'feature_maps'):
            # CNN model with feature maps
            feature_maps = outputs.feature_maps
            cls_features = F.adaptive_avg_pool2d(feature_maps, (1, 1)).squeeze(-1).squeeze(-1)
            patch_features = feature_maps
            
        else:
            # Last resort: try to get hidden states
            if hasattr(outputs, 'hidden_states'):
                features = outputs.hidden_states[-1]
                cls_features = features[:, 0, :] if features.dim() == 3 else features.mean(dim=1)
                patch_features = features[:, 1:, :] if features.dim() == 3 and features.shape[1] > 1 else None
            else:
                raise ValueError("Could not extract features from model output")
        
        # Pool features if needed
        pooled_features = cls_features
        if self.config.normalize_features:
            pooled_features = F.normalize(pooled_features, p=2, dim=-1)
        
        if return_dict:
            return EncoderOutput(
                features=cls_features if not self.config.return_patch_features else features,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                pooled_features=pooled_features,
                mask=None,
                metadata={
                    'pixel_values_shape': pixel_values.shape,
                    'model_name': self.config.model_name,
                    'patch_features': patch_features
                }
            )
        else:
            return cls_features if not self.config.return_patch_features else features, pooled_features
    
    def extract_features(self,
                        images: Union[Image.Image, List[Image.Image], torch.Tensor, np.ndarray],
                        layer: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract features from specific layer(s).
        
        Args:
            images: Input image(s)
            layer: Specific layer to extract from (None for all layers)
            
        Returns:
            Dictionary of features from different layers
        """
        pixel_values = self.preprocess(images)
        pixel_values = pixel_values.to(self.model.device)
        
        # Forward pass with all hidden states
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True
        )
        
        if not hasattr(outputs, 'hidden_states'):
            raise ValueError("Model does not return hidden states")
        
        features = {}
        for i, hidden_state in enumerate(outputs.hidden_states):
            if layer is None or i == layer:
                # Extract CLS token or average pool
                if hidden_state.dim() == 3:  # Transformer output
                    cls_token = hidden_state[:, 0, :]
                    features[f'layer_{i}'] = cls_token
                else:  # CNN or other
                    features[f'layer_{i}'] = hidden_state
        
        return features
    
    def get_patch_embeddings(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Get patch embeddings from transformer-based models.
        
        Args:
            images: Input image(s)
            
        Returns:
            Patch embeddings [batch_size, num_patches, hidden_dim]
        """
        if not hasattr(self.model, 'embeddings'):
            raise ValueError("Model does not have patch embeddings")
        
        pixel_values = self.preprocess(images)
        pixel_values = pixel_values.to(self.model.device)
        
        # Get patch embeddings
        embeddings = self.model.embeddings(pixel_values)
        return embeddings
    
    def get_attention_maps(self,
                          images: Union[Image.Image, List[Image.Image]],
                          layer: Optional[int] = None,
                          head: Optional[int] = None) -> torch.Tensor:
        """
        Get attention maps from transformer-based models.
        
        Args:
            images: Input image(s)
            layer: Specific layer to get attention from (None for all)
            head: Specific head to get attention from (None for all)
            
        Returns:
            Attention maps
        """
        pixel_values = self.preprocess(images)
        pixel_values = pixel_values.to(self.model.device)
        
        outputs = self.model(
            pixel_values=pixel_values,
            output_attentions=True,
            return_dict=True
        )
        
        if not hasattr(outputs, 'attentions'):
            raise ValueError("Model does not return attention maps")
        
        attentions = outputs.attentions
        
        if layer is not None:
            if layer >= len(attentions):
                raise ValueError(f"Layer {layer} not available. Model has {len(attentions)} layers.")
            attentions = [attentions[layer]]
        
        # Stack attentions
        if len(attentions) == 1:
            attention = attentions[0]
        else:
            attention = torch.stack(attentions, dim=0)
        
        # Select head if specified
        if head is not None:
            if head >= attention.shape[2]:
                raise ValueError(f"Head {head} not available. Model has {attention.shape[2]} heads.")
            attention = attention[..., head, :, :]
        
        return attention
    
    def batch_encode(self,
                    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
                    batch_size: int = 32,
                    **kwargs) -> List[EncoderOutput]:
        """
        Encode a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of encoder outputs
        """
        outputs = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            output = self(batch_images, **kwargs)
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
                metadata={'batch_size': len(images)}
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
            'image_size': self.config.image_size,
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

class ViTEncoder(VisionEncoder):
    """Vision Transformer (ViT) encoder."""
    
    def _initialize_model(self):
        """Initialize ViT model and preprocessor."""
        try:
            self.preprocessor = ViTImageProcessor.from_pretrained(
                self.config.model_name,
                size={"height": self.config.image_size, "width": self.config.image_size}
            )
            self.model = ViTModel.from_pretrained(
                self.config.model_name,
                add_pooling_layer=True
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
            # Update config with model-specific values
            self.config.patch_size = self.model.config.patch_size
            self.config.num_patches = (self.config.image_size // self.config.patch_size) ** 2
            
        except Exception as e:
            logger.error(f"Failed to load ViT model {self.config.model_name}: {e}")
            raise
    
    def forward(self, images, **kwargs):
        """ViT-specific forward pass."""
        # ViT expects pixel_values
        return super().forward(images, **kwargs)

class CLIPVisionEncoder(VisionEncoder):
    """CLIP vision encoder."""
    
    def _initialize_model(self):
        """Initialize CLIP vision model and preprocessor."""
        try:
            self.preprocessor = CLIPImageProcessor.from_pretrained(
                self.config.model_name
            )
            self.model = CLIPVisionModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
            # Update config with model-specific values
            self.config.patch_size = self.model.config.patch_size
            self.config.num_patches = (self.config.image_size // self.config.patch_size) ** 2
            
        except Exception as e:
            logger.error(f"Failed to load CLIP vision model {self.config.model_name}: {e}")
            raise

class ResNetEncoder(VisionEncoder):
    """ResNet encoder (CNN-based)."""
    
    def _initialize_model(self):
        """Initialize ResNet model."""
        try:
            import torchvision.models as models
            
            # Get ResNet model
            if self.config.model_name.lower() == 'resnet18':
                model_fn = models.resnet18
            elif self.config.model_name.lower() == 'resnet34':
                model_fn = models.resnet34
            elif self.config.model_name.lower() == 'resnet50':
                model_fn = models.resnet50
            elif self.config.model_name.lower() == 'resnet101':
                model_fn = models.resnet101
            elif self.config.model_name.lower() == 'resnet152':
                model_fn = models.resnet152
            else:
                raise ValueError(f"Unsupported ResNet model: {self.config.model_name}")
            
            # Load pretrained model
            self.model = model_fn(pretrained=self.config.pretrained_cnn)
            
            # Remove final fully connected layer if requested
            if self.config.remove_fc:
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                self.output_dim = 512 if '18' in self.config.model_name or '34' in self.config.model_name else 2048
            else:
                self.output_dim = 1000  # ImageNet classes
            
            # Modify first conv layer if needed
            if self.config.num_channels != 3:
                # Change first conv layer to accept different number of channels
                original_conv = self.model.conv1
                self.model.conv1 = nn.Conv2d(
                    self.config.num_channels,
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=original_conv.bias is not None
                )
                
                # Copy weights for first 3 channels, initialize rest
                with torch.no_grad():
                    if self.config.num_channels > 3:
                        self.model.conv1.weight[:, :3] = original_conv.weight
                        # Initialize remaining channels
                        nn.init.kaiming_normal_(self.model.conv1.weight[:, 3:])
                    else:
                        self.model.conv1.weight = original_conv.weight[:, :self.config.num_channels]
            
        except Exception as e:
            logger.error(f"Failed to load ResNet model {self.config.model_name}: {e}")
            raise
    
    def forward(self, images, **kwargs):
        """ResNet-specific forward pass."""
        # Preprocess images
        pixel_values = self.preprocess(images, **kwargs)
        pixel_values = pixel_values.to(next(self.model.parameters()).device)
        
        # Forward pass
        features = self.model(pixel_values)
        
        # ResNet outputs [batch_size, channels, 1, 1] after avgpool
        if features.dim() == 4:
            features = features.squeeze(-1).squeeze(-1)
        
        # Normalize if requested
        pooled_features = features
        if self.config.normalize_features:
            pooled_features = F.normalize(pooled_features, p=2, dim=-1)
        
        return EncoderOutput(
            features=features,
            hidden_states=None,
            attentions=None,
            pooled_features=pooled_features,
            mask=None,
            metadata={
                'pixel_values_shape': pixel_values.shape,
                'model_name': self.config.model_name
            }
        )

class EfficientNetEncoder(VisionEncoder):
    """EfficientNet encoder."""
    
    def _initialize_model(self):
        """Initialize EfficientNet model."""
        try:
            from efficientnet_pytorch import EfficientNet
            
            # Map model names
            model_map = {
                'efficientnet-b0': 'efficientnet-b0',
                'efficientnet-b1': 'efficientnet-b1',
                'efficientnet-b2': 'efficientnet-b2',
                'efficientnet-b3': 'efficientnet-b3',
                'efficientnet-b4': 'efficientnet-b4',
                'efficientnet-b5': 'efficientnet-b5',
                'efficientnet-b6': 'efficientnet-b6',
                'efficientnet-b7': 'efficientnet-b7'
            }
            
            if self.config.model_name.lower() not in model_map:
                raise ValueError(f"Unsupported EfficientNet model: {self.config.model_name}")
            
            # Load pretrained model
            self.model = EfficientNet.from_pretrained(model_map[self.config.model_name.lower()])
            
            # Remove final fully connected layer
            if self.config.remove_fc:
                self.model._fc = nn.Identity()
                # Get output dimension from model
                self.output_dim = self.model._conv_head.out_channels
            else:
                self.output_dim = 1000  # ImageNet classes
            
        except ImportError:
            logger.error("efficientnet-pytorch not installed. Install with: pip install efficientnet-pytorch")
            raise
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model {self.config.model_name}: {e}")
            raise

class DINOEncoder(VisionEncoder):
    """DINO (self-supervised ViT) encoder."""
    
    def _initialize_model(self):
        """Initialize DINO model."""
        try:
            # DINO uses ViT architecture
            self.preprocessor = ViTImageProcessor.from_pretrained(
                'facebook/dino-vitb16',
                size={"height": self.config.image_size, "width": self.config.image_size}
            )
            
            # Load DINO model
            self.model = ViTModel.from_pretrained('facebook/dino-vitb16')
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
            # Update config
            self.config.patch_size = self.model.config.patch_size
            self.config.num_patches = (self.config.image_size // self.config.patch_size) ** 2
            
        except Exception as e:
            logger.error(f"Failed to load DINO model: {e}")
            raise
    
    def get_self_attention(self, images, threshold=0.6):
        """
        Get self-attention maps from DINO.
        
        Args:
            images: Input images
            threshold: Threshold for binarizing attention
            
        Returns:
            Self-attention maps
        """
        # Get patch embeddings
        pixel_values = self.preprocess(images)
        pixel_values = pixel_values.to(self.model.device)
        
        # Forward pass
        outputs = self.model(pixel_values, output_attentions=True)
        
        # Get attention from last layer
        attentions = outputs.attentions[-1]
        
        # Average over heads
        attentions = attentions.mean(dim=1)
        
        # Keep only the [CLS] token attention
        cls_attention = attentions[:, 0, 1:]  # [batch_size, num_patches]
        
        # Reshape to 2D
        grid_size = int(np.sqrt(cls_attention.shape[-1]))
        cls_attention = cls_attention.reshape(-1, grid_size, grid_size)
        
        # Binarize if threshold provided
        if threshold is not None:
            cls_attention = (cls_attention > threshold).float()
        
        return cls_attention

# Example usage
if __name__ == "__main__":
    # Test different vision encoders
    encoders_to_test = [
        ('google/vit-base-patch16-224', ViTEncoder),
        ('openai/clip-vit-base-patch32', CLIPVisionEncoder),
        ('resnet50', ResNetEncoder),
        ('efficientnet-b0', EfficientNetEncoder)
    ]
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    
    for model_name, encoder_class in encoders_to_test:
        try:
            print(f"\nTesting {encoder_class.__name__} with {model_name}:")
            
            config = VisionEncoderConfig(
                model_name=model_name,
                pretrained=True,
                trainable=False,
                image_size=224
            )
            
            encoder = encoder_class(config)
            
            # Test single image
            output = encoder(dummy_image)
            print(f"  Output shape: {output.features.shape}")
            print(f"  Pooled shape: {output.pooled_features.shape}")
            
            # Test batch
            batch_output = encoder([dummy_image, dummy_image])
            print(f"  Batch output shape: {batch_output.features.shape}")
            
            # Get statistics
            stats = encoder.get_statistics()
            print(f"  Parameters: {stats['num_parameters']:,}")
            print(f"  Output dim: {stats['output_dim']}")
            
            # Test feature extraction
            features = encoder.extract_features(dummy_image)
            print(f"  Extracted features from {len(features)} layers")
            
        except Exception as e:
            print(f"  Error: {e}")