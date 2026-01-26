"""
Multimodal encoders for processing different input modalities.
Includes text, vision, video, and audio encoders with support for pre-trained models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings

from .text_encoder import (
    TextEncoder,
    TransformerTextEncoder,
    CLIPTextEncoder,
    BERTEncoder,
    T5Encoder,
    DistilBERTEncoder,
    RoBERTaEncoder,
    TextEncoderConfig
)

from .vision_encoder import (
    VisionEncoder,
    ResNetEncoder,
    ViTEncoder,
    CLIPVisionEncoder,
    DINOEncoder,
    EfficientNetEncoder,
    VisionEncoderConfig
)

from .video_encoder import (
    VideoEncoder,
    VideoTransformerEncoder,
    CNN3DEncoder,
    TimesformerEncoder,
    VideoMAEEncoder,
    VideoEncoderConfig
)

from .audio_encoder import (
    AudioEncoder,
    Wav2Vec2Encoder,
    HuBERTEncoder,
    WhisperEncoder,
    AudioEncoderConfig
)

from .multimodal_encoder import (
    MultiModalEncoder,
    ModalityProjection,
    MultiModalEncoderConfig
)

from .utils import (
    encoder_utils,
    preprocessing_utils,
    feature_extraction_utils,
    model_loading_utils
)

__all__ = [
    # Text encoders
    'TextEncoder',
    'TransformerTextEncoder',
    'CLIPTextEncoder',
    'BERTEncoder',
    'T5Encoder',
    'DistilBERTEncoder',
    'RoBERTaEncoder',
    'TextEncoderConfig',
    
    # Vision encoders
    'VisionEncoder',
    'ResNetEncoder',
    'ViTEncoder',
    'CLIPVisionEncoder',
    'DINOEncoder',
    'EfficientNetEncoder',
    'VisionEncoderConfig',
    
    # Video encoders
    'VideoEncoder',
    'VideoTransformerEncoder',
    'CNN3DEncoder',
    'TimesformerEncoder',
    'VideoMAEEncoder',
    'VideoEncoderConfig',
    
    # Audio encoders
    'AudioEncoder',
    'Wav2Vec2Encoder',
    'HuBERTEncoder',
    'WhisperEncoder',
    'AudioEncoderConfig',
    
    # Multimodal encoders
    'MultiModalEncoder',
    'ModalityProjection',
    'MultiModalEncoderConfig',
    
    # Utilities
    'encoder_utils',
    'preprocessing_utils',
    'feature_extraction_utils',
    'model_loading_utils'
]

# Version
__version__ = '1.0.0'

# Initialize logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Encoder types
class EncoderType(Enum):
    """Types of encoders."""
    TEXT = "text"
    VISION = "vision"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

@dataclass
class EncoderConfig:
    """Base configuration for encoders."""
    
    # General parameters
    encoder_type: EncoderType = EncoderType.TEXT
    model_name: str = "bert-base-uncased"
    pretrained: bool = True
    trainable: bool = True
    output_dim: int = 768
    hidden_dim: int = 768
    dropout: float = 0.1
    
    # Feature extraction
    pool_method: str = "mean"  # "mean", "max", "cls", "last"
    normalize_features: bool = True
    return_all_tokens: bool = False
    
    # Input preprocessing
    max_length: int = 512
    image_size: int = 224
    num_frames: int = 16
    audio_length: int = 16000
    
    # Model-specific
    gradient_checkpointing: bool = False
    use_fp16: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EncoderConfig':
        """Create config from dictionary."""
        # Handle encoder_type conversion
        if 'encoder_type' in config_dict and isinstance(config_dict['encoder_type'], str):
            config_dict['encoder_type'] = EncoderType(config_dict['encoder_type'])
        return cls(**config_dict)

@dataclass
class EncoderOutput:
    """Output from encoder forward pass."""
    
    features: torch.Tensor
    hidden_states: Optional[List[torch.Tensor]] = None
    attentions: Optional[List[torch.Tensor]] = None
    pooled_features: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EncoderFactory:
    """Factory class for creating encoders."""
    
    @staticmethod
    def create_encoder(
        encoder_type: Union[str, EncoderType],
        model_name: Optional[str] = None,
        config: Optional[EncoderConfig] = None,
        **kwargs
    ) -> nn.Module:
        """
        Create an encoder instance.
        
        Args:
            encoder_type: Type of encoder (text, vision, video, audio, multimodal)
            model_name: Name of pre-trained model
            config: Encoder configuration
            **kwargs: Additional arguments
            
        Returns:
            Encoder instance
        """
        if isinstance(encoder_type, str):
            encoder_type = EncoderType(encoder_type.lower())
        
        # Use provided config or create default
        if config is None:
            config = EncoderConfig(encoder_type=encoder_type)
        
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                warnings.warn(f"Config has no attribute '{key}', ignoring")
        
        # Set model name if provided
        if model_name is not None:
            config.model_name = model_name
        
        # Create encoder based on type
        if encoder_type == EncoderType.TEXT:
            return EncoderFactory._create_text_encoder(config)
        elif encoder_type == EncoderType.VISION:
            return EncoderFactory._create_vision_encoder(config)
        elif encoder_type == EncoderType.VIDEO:
            return EncoderFactory._create_video_encoder(config)
        elif encoder_type == EncoderType.AUDIO:
            return EncoderFactory._create_audio_encoder(config)
        elif encoder_type == EncoderType.MULTIMODAL:
            return EncoderFactory._create_multimodal_encoder(config)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    @staticmethod
    def _create_text_encoder(config: EncoderConfig) -> TextEncoder:
        """Create text encoder."""
        model_name = config.model_name.lower()
        
        if 'clip' in model_name:
            return CLIPTextEncoder(config)
        elif 'bert' in model_name:
            if 'distil' in model_name:
                return DistilBERTEncoder(config)
            elif 'roberta' in model_name:
                return RoBERTaEncoder(config)
            else:
                return BERTEncoder(config)
        elif 't5' in model_name:
            return T5Encoder(config)
        else:
            # Default to transformer encoder
            return TransformerTextEncoder(config)
    
    @staticmethod
    def _create_vision_encoder(config: EncoderConfig) -> VisionEncoder:
        """Create vision encoder."""
        model_name = config.model_name.lower()
        
        if 'clip' in model_name:
            return CLIPVisionEncoder(config)
        elif 'vit' in model_name or 'vision' in model_name:
            return ViTEncoder(config)
        elif 'resnet' in model_name:
            return ResNetEncoder(config)
        elif 'efficientnet' in model_name:
            return EfficientNetEncoder(config)
        elif 'dino' in model_name:
            return DINOEncoder(config)
        else:
            # Default to ResNet
            return ResNetEncoder(config)
    
    @staticmethod
    def _create_video_encoder(config: EncoderConfig) -> VideoEncoder:
        """Create video encoder."""
        model_name = config.model_name.lower()
        
        if 'timesformer' in model_name:
            return TimesformerEncoder(config)
        elif 'videomae' in model_name:
            return VideoMAEEncoder(config)
        elif '3d' in model_name or 'cnn3d' in model_name:
            return CNN3DEncoder(config)
        else:
            # Default to video transformer
            return VideoTransformerEncoder(config)
    
    @staticmethod
    def _create_audio_encoder(config: EncoderConfig) -> AudioEncoder:
        """Create audio encoder."""
        model_name = config.model_name.lower()
        
        if 'wav2vec' in model_name:
            return Wav2Vec2Encoder(config)
        elif 'hubert' in model_name:
            return HuBERTEncoder(config)
        elif 'whisper' in model_name:
            return WhisperEncoder(config)
        else:
            # Default to Wav2Vec2
            return Wav2Vec2Encoder(config)
    
    @staticmethod
    def _create_multimodal_encoder(config: EncoderConfig) -> MultiModalEncoder:
        """Create multimodal encoder."""
        return MultiModalEncoder(config)

class MultiEncoderSystem:
    """
    System for managing multiple encoders and extracting features from different modalities.
    """
    
    def __init__(self, configs: Optional[Dict[str, EncoderConfig]] = None):
        """
        Initialize multi-encoder system.
        
        Args:
            configs: Dictionary mapping modality names to encoder configs
        """
        self.configs = configs or {}
        self.encoders = {}
        self.feature_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize encoders from configs
        for modality, config in self.configs.items():
            self.add_encoder(modality, config)
        
        self.logger.info(f"MultiEncoderSystem initialized with {len(self.encoders)} encoders")
    
    def add_encoder(self, modality: str, config: EncoderConfig) -> bool:
        """
        Add an encoder for a modality.
        
        Args:
            modality: Modality name (e.g., 'text', 'image', 'video')
            config: Encoder configuration
            
        Returns:
            Success status
        """
        try:
            encoder = EncoderFactory.create_encoder(
                encoder_type=config.encoder_type,
                config=config
            )
            
            self.encoders[modality] = encoder
            self.configs[modality] = config
            
            self.logger.info(f"Added {config.encoder_type.value} encoder for modality '{modality}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add encoder for modality '{modality}': {e}")
            return False
    
    def remove_encoder(self, modality: str) -> bool:
        """
        Remove an encoder.
        
        Args:
            modality: Modality name
            
        Returns:
            Success status
        """
        if modality in self.encoders:
            del self.encoders[modality]
            del self.configs[modality]
            
            # Clear cache for this modality
            cache_keys = [k for k in self.feature_cache.keys() if k.startswith(f"{modality}_")]
            for key in cache_keys:
                del self.feature_cache[key]
            
            self.logger.info(f"Removed encoder for modality '{modality}'")
            return True
        else:
            self.logger.warning(f"No encoder found for modality '{modality}'")
            return False
    
    def encode(self,
               modality: str,
               inputs: Any,
               cache_key: Optional[str] = None,
               use_cache: bool = True,
               **kwargs) -> EncoderOutput:
        """
        Encode inputs for a specific modality.
        
        Args:
            modality: Modality name
            inputs: Input data (format depends on modality)
            cache_key: Optional cache key for storing/retrieving features
            use_cache: Whether to use feature cache
            **kwargs: Additional arguments for encoder
            
        Returns:
            Encoder output
        """
        if modality not in self.encoders:
            raise ValueError(
                f"No encoder found for modality '{modality}'. "
                f"Available modalities: {list(self.encoders.keys())}"
            )
        
        encoder = self.encoders[modality]
        
        # Check cache
        if use_cache and cache_key is not None:
            cache_key = f"{modality}_{cache_key}"
            if cache_key in self.feature_cache:
                self.logger.debug(f"Using cached features for {cache_key}")
                return self.feature_cache[cache_key]
        
        try:
            # Encode inputs
            output = encoder(inputs, **kwargs)
            
            # Cache results if requested
            if use_cache and cache_key is not None:
                self.feature_cache[cache_key] = output
                self.logger.debug(f"Cached features for {cache_key}")
            
            return output
            
        except Exception as e:
            self.logger.error(f"Failed to encode {modality} inputs: {e}")
            raise
    
    def encode_batch(self,
                    modality: str,
                    batch_inputs: List[Any],
                    batch_size: int = 32,
                    **kwargs) -> List[EncoderOutput]:
        """
        Encode a batch of inputs.
        
        Args:
            modality: Modality name
            batch_inputs: List of input data
            batch_size: Batch size for processing
            **kwargs: Additional arguments for encoder
            
        Returns:
            List of encoder outputs
        """
        outputs = []
        
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            
            # Process batch
            try:
                encoder = self.encoders[modality]
                batch_output = encoder.batch_encode(batch, **kwargs)
                outputs.extend(batch_output)
            except AttributeError:
                # Encoder doesn't have batch_encode, process individually
                for item in batch:
                    output = self.encode(modality, item, use_cache=False, **kwargs)
                    outputs.append(output)
        
        return outputs
    
    def encode_multimodal(self,
                         modality_inputs: Dict[str, Any],
                         cache_keys: Optional[Dict[str, str]] = None,
                         **kwargs) -> Dict[str, EncoderOutput]:
        """
        Encode inputs from multiple modalities.
        
        Args:
            modality_inputs: Dictionary mapping modality names to inputs
            cache_keys: Optional cache keys for each modality
            **kwargs: Additional arguments for encoders
            
        Returns:
            Dictionary mapping modality names to encoder outputs
        """
        outputs = {}
        
        for modality, inputs in modality_inputs.items():
            cache_key = None
            if cache_keys and modality in cache_keys:
                cache_key = cache_keys[modality]
            
            try:
                output = self.encode(modality, inputs, cache_key=cache_key, **kwargs)
                outputs[modality] = output
            except Exception as e:
                self.logger.error(f"Failed to encode {modality}: {e}")
                # Continue with other modalities
                continue
        
        return outputs
    
    def get_embedding_dim(self, modality: str) -> int:
        """
        Get embedding dimension for a modality.
        
        Args:
            modality: Modality name
            
        Returns:
            Embedding dimension
        """
        if modality not in self.encoders:
            raise ValueError(f"No encoder found for modality '{modality}'")
        
        encoder = self.encoders[modality]
        if hasattr(encoder, 'output_dim'):
            return encoder.output_dim
        elif hasattr(encoder, 'config') and hasattr(encoder.config, 'output_dim'):
            return encoder.config.output_dim
        else:
            # Try to infer from model
            return 768  # Default
    
    def get_all_embedding_dims(self) -> Dict[str, int]:
        """Get embedding dimensions for all encoders."""
        dims = {}
        for modality in self.encoders.keys():
            dims[modality] = self.get_embedding_dim(modality)
        return dims
    
    def project_to_common_space(self,
                               modality_outputs: Dict[str, EncoderOutput],
                               target_dim: int = 512,
                               normalize: bool = True) -> Dict[str, torch.Tensor]:
        """
        Project features from different modalities to a common embedding space.
        
        Args:
            modality_outputs: Dictionary of encoder outputs
            target_dim: Target embedding dimension
            normalize: Whether to normalize embeddings
            
        Returns:
            Dictionary of projected embeddings
        """
        projected = {}
        
        for modality, output in modality_outputs.items():
            features = output.features
            
            # Get or create projection layer
            proj_key = f"{modality}_to_{target_dim}"
            if not hasattr(self, '_projection_layers'):
                self._projection_layers = {}
            
            if proj_key not in self._projection_layers:
                input_dim = features.shape[-1]
                self._projection_layers[proj_key] = nn.Linear(input_dim, target_dim)
                # Initialize projection layer
                nn.init.xavier_uniform_(self._projection_layers[proj_key].weight)
                nn.init.zeros_(self._projection_layers[proj_key].bias)
            
            projection = self._projection_layers[proj_key]
            projected_features = projection(features)
            
            if normalize:
                projected_features = nn.functional.normalize(projected_features, p=2, dim=-1)
            
            projected[modality] = projected_features
        
        return projected
    
    def clear_cache(self, modality: Optional[str] = None):
        """
        Clear feature cache.
        
        Args:
            modality: If provided, clear only cache for this modality
        """
        if modality is None:
            self.feature_cache.clear()
            self.logger.info("Cleared all feature cache")
        else:
            # Clear cache for specific modality
            cache_keys = [k for k in self.feature_cache.keys() if k.startswith(f"{modality}_")]
            for key in cache_keys:
                del self.feature_cache[key]
            self.logger.info(f"Cleared cache for modality '{modality}' ({len(cache_keys)} entries)")
    
    def save_encoders(self, directory: str):
        """
        Save all encoders to directory.
        
        Args:
            directory: Directory to save encoders
        """
        import os
        import json
        
        os.makedirs(directory, exist_ok=True)
        
        # Save configs
        configs_dict = {}
        for modality, config in self.configs.items():
            configs_dict[modality] = config.to_dict()
        
        configs_path = os.path.join(directory, 'encoder_configs.json')
        with open(configs_path, 'w') as f:
            json.dump(configs_dict, f, indent=2)
        
        # Save models
        for modality, encoder in self.encoders.items():
            model_path = os.path.join(directory, f"{modality}_encoder.pt")
            torch.save(encoder.state_dict(), model_path)
        
        self.logger.info(f"Saved encoders to {directory}")
    
    def load_encoders(self, directory: str):
        """
        Load encoders from directory.
        
        Args:
            directory: Directory containing saved encoders
        """
        import os
        import json
        
        # Load configs
        configs_path = os.path.join(directory, 'encoder_configs.json')
        if os.path.exists(configs_path):
            with open(configs_path, 'r') as f:
                configs_dict = json.load(f)
            
            # Recreate encoders from configs
            for modality, config_dict in configs_dict.items():
                config = EncoderConfig.from_dict(config_dict)
                self.add_encoder(modality, config)
                
                # Load model weights
                model_path = os.path.join(directory, f"{modality}_encoder.pt")
                if os.path.exists(model_path):
                    self.encoders[modality].load_state_dict(torch.load(model_path))
                    self.logger.info(f"Loaded weights for {modality} encoder")
        
        self.logger.info(f"Loaded encoders from {directory}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the encoder system."""
        stats = {
            'num_encoders': len(self.encoders),
            'modalities': list(self.encoders.keys()),
            'cache_size': len(self.feature_cache),
            'embedding_dims': self.get_all_embedding_dims()
        }
        
        # Add encoder-specific stats
        for modality, encoder in self.encoders.items():
            if hasattr(encoder, 'get_statistics'):
                stats[f"{modality}_stats"] = encoder.get_statistics()
        
        return stats

# Global encoder system instance
_encoder_system = None

def get_encoder_system(configs: Optional[Dict[str, EncoderConfig]] = None) -> MultiEncoderSystem:
    """Get or create the global encoder system."""
    global _encoder_system
    if _encoder_system is None:
        _encoder_system = MultiEncoderSystem(configs)
    return _encoder_system

def reset_encoder_system():
    """Reset the global encoder system."""
    global _encoder_system
    _encoder_system = None

# Example usage
if __name__ == "__main__":
    # Create encoder system
    configs = {
        'text': EncoderConfig(
            encoder_type=EncoderType.TEXT,
            model_name='bert-base-uncased',
            output_dim=768
        ),
        'image': EncoderConfig(
            encoder_type=EncoderType.VISION,
            model_name='resnet50',
            output_dim=2048
        )
    }
    
    encoder_system = MultiEncoderSystem(configs)
    
    # Encode some data (example)
    # text_output = encoder_system.encode('text', "Hello, world!")
    # image_output = encoder_system.encode('image', torch.randn(3, 224, 224))
    
    # Get statistics
    stats = encoder_system.get_statistics()
    print(f"Encoder system statistics: {stats}")