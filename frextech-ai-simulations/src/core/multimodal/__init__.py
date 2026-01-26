"""
Multimodal module for handling multiple input modalities (text, image, video, audio).
This module provides encoders, fusion mechanisms, and alignment strategies for multimodal data.
"""

from .encoders import (
    TextEncoder,
    VisionEncoder,
    VideoEncoder,
    AudioEncoder,
    MultiModalEncoder
)
from .fusion import (
    CrossAttentionFusion,
    ConcatenationFusion,
    TransformerFusion,
    MultiModalFusion
)
from .alignment import (
    ContrastiveAlignment,
    AlignmentTrainer,
    MultiModalAlignment
)

from .embeddings import MultiModalEmbeddings
from .preprocessor import MultiModalPreprocessor
from .utils import modality_utils

__all__ = [
    # Encoders
    'TextEncoder',
    'VisionEncoder',
    'VideoEncoder',
    'AudioEncoder',
    'MultiModalEncoder',
    
    # Fusion
    'CrossAttentionFusion',
    'ConcatenationFusion',
    'TransformerFusion',
    'MultiModalFusion',
    
    # Alignment
    'ContrastiveAlignment',
    'AlignmentTrainer',
    'MultiModalAlignment',
    
    # Utilities
    'MultiModalEmbeddings',
    'MultiModalPreprocessor',
    'modality_utils'
]

# Version
__version__ = '1.0.0'

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Configuration defaults
DEFAULT_CONFIG = {
    'embedding_dim': 512,
    'projection_dim': 256,
    'fusion_method': 'cross_attention',
    'normalize_embeddings': True,
    'use_pretrained': True,
    'trainable': True
}

def get_encoder(encoder_type: str, **kwargs):
    """
    Factory function to get an encoder instance.
    
    Args:
        encoder_type: Type of encoder ('text', 'vision', 'video', 'audio', 'multimodal')
        **kwargs: Encoder-specific parameters
        
    Returns:
        Encoder instance
        
    Raises:
        ValueError: If encoder_type is invalid
    """
    encoder_map = {
        'text': TextEncoder,
        'vision': VisionEncoder,
        'video': VideoEncoder,
        'audio': AudioEncoder,
        'multimodal': MultiModalEncoder
    }
    
    if encoder_type not in encoder_map:
        raise ValueError(
            f"Invalid encoder_type: {encoder_type}. "
            f"Must be one of: {list(encoder_map.keys())}"
        )
    
    return encoder_map[encoder_type](**kwargs)

def get_fusion(fusion_type: str, **kwargs):
    """
    Factory function to get a fusion module.
    
    Args:
        fusion_type: Type of fusion ('cross_attention', 'concat', 'transformer')
        **kwargs: Fusion-specific parameters
        
    Returns:
        Fusion module instance
        
    Raises:
        ValueError: If fusion_type is invalid
    """
    fusion_map = {
        'cross_attention': CrossAttentionFusion,
        'concat': ConcatenationFusion,
        'transformer': TransformerFusion
    }
    
    if fusion_type not in fusion_map:
        raise ValueError(
            f"Invalid fusion_type: {fusion_type}. "
            f"Must be one of: {list(fusion_map.keys())}"
        )
    
    return fusion_map[fusion_type](**kwargs)

class MultiModalConfig:
    """Configuration class for multimodal modules."""
    
    def __init__(self, **kwargs):
        self.embedding_dim = kwargs.get('embedding_dim', 512)
        self.projection_dim = kwargs.get('projection_dim', 256)
        self.fusion_method = kwargs.get('fusion_method', 'cross_attention')
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        self.use_pretrained = kwargs.get('use_pretrained', True)
        self.trainable = kwargs.get('trainable', True)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        self.attention_heads = kwargs.get('attention_heads', 8)
        self.hidden_dim = kwargs.get('hidden_dim', 1024)
        
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute: {key}")

# Initialize global configuration
_config = MultiModalConfig()

def get_config():
    """Get the global configuration."""
    return _config

def set_config(**kwargs):
    """Update the global configuration."""
    _config.update(**kwargs)