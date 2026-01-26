"""
Fusion modules for combining information from multiple modalities.
Includes cross-attention, concatenation, transformer fusion, and adaptive fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
import math

from .cross_attention import (
    CrossAttentionFusion,
    MultiHeadCrossAttention,
    CrossAttentionBlock,
    CrossAttentionConfig
)

from .concatenation import (
    ConcatenationFusion,
    WeightedConcatenation,
    GatedConcatenation,
    ConcatenationConfig
)

from .transformer_fusion import (
    TransformerFusion,
    MultiModalTransformer,
    TransformerFusionConfig
)

from .adaptive_fusion import (
    AdaptiveFusion,
    DynamicFusion,
    AdaptiveFusionConfig
)

from .utils import (
    fusion_utils,
    attention_utils,
    gating_utils,
    normalization_utils
)

__all__ = [
    # Cross-attention fusion
    'CrossAttentionFusion',
    'MultiHeadCrossAttention',
    'CrossAttentionBlock',
    'CrossAttentionConfig',
    
    # Concatenation fusion
    'ConcatenationFusion',
    'WeightedConcatenation',
    'GatedConcatenation',
    'ConcatenationConfig',
    
    # Transformer fusion
    'TransformerFusion',
    'MultiModalTransformer',
    'TransformerFusionConfig',
    
    # Adaptive fusion
    'AdaptiveFusion',
    'DynamicFusion',
    'AdaptiveFusionConfig',
    
    # Utilities
    'fusion_utils',
    'attention_utils',
    'gating_utils',
    'normalization_utils'
]

# Version
__version__ = '1.0.0'

# Initialize logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Fusion types
class FusionType(Enum):
    """Types of multimodal fusion."""
    CROSS_ATTENTION = "cross_attention"
    CONCATENATION = "concatenation"
    TRANSFORMER = "transformer"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"
    TENSOR_FUSION = "tensor_fusion"

@dataclass
class FusionConfig:
    """Configuration for fusion modules."""
    
    # General parameters
    fusion_type: FusionType = FusionType.CROSS_ATTENTION
    input_dims: Dict[str, int] = field(default_factory=dict)  # Modality -> dimension
    output_dim: int = 512
    hidden_dim: int = 256
    
    # Cross-attention parameters
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_bias: bool = True
    
    # Concatenation parameters
    concatenation_dim: int = -1  # Dimension to concatenate along
    normalize_inputs: bool = True
    project_before_concat: bool = False
    
    # Transformer parameters
    num_layers: int = 4
    mlp_ratio: int = 4
    activation: str = "gelu"
    pre_norm: bool = True
    
    # Adaptive parameters
    temperature: float = 1.0
    use_gating: bool = True
    gating_type: str = "softmax"  # "softmax", "sigmoid", "sparsemax"
    
    # Output
    output_projection: bool = True
    output_activation: str = "none"  # "none", "relu", "tanh", "sigmoid"
    residual_connection: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, FusionType):
                config_dict[k] = v.value
            else:
                config_dict[k] = v
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FusionConfig':
        """Create config from dictionary."""
        # Handle FusionType conversion
        if 'fusion_type' in config_dict and isinstance(config_dict['fusion_type'], str):
            config_dict['fusion_type'] = FusionType(config_dict['fusion_type'])
        return cls(**config_dict)

@dataclass
class FusionOutput:
    """Output from fusion module."""
    
    fused_features: torch.Tensor
    attention_weights: Optional[Dict[str, torch.Tensor]] = None
    gate_values: Optional[Dict[str, torch.Tensor]] = None
    intermediate_features: Optional[Dict[str, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiModalFusion(nn.Module):
    """
    Main multimodal fusion module supporting multiple fusion strategies.
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.fusion_modules = nn.ModuleDict()
        self.logger = logging.getLogger(__name__)
        
        # Initialize fusion module based on type
        self._initialize_fusion()
        
        # Initialize input projections if needed
        self._initialize_projections()
        
        # Initialize output projection
        if config.output_projection:
            self.output_projection = nn.Sequential(
                nn.Linear(config.hidden_dim, config.output_dim),
                self._get_activation(config.output_activation)
            )
        else:
            self.output_projection = nn.Identity()
        
        self.logger.info(f"Initialized {config.fusion_type.value} fusion module")
    
    def _initialize_fusion(self):
        """Initialize the fusion module based on config."""
        fusion_type = self.config.fusion_type
        
        if fusion_type == FusionType.CROSS_ATTENTION:
            self.fusion_module = CrossAttentionFusion(
                CrossAttentionConfig(
                    input_dims=self.config.input_dims,
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                    attention_dropout=self.config.attention_dropout
                )
            )
            
        elif fusion_type == FusionType.CONCATENATION:
            self.fusion_module = ConcatenationFusion(
                ConcatenationConfig(
                    input_dims=self.config.input_dims,
                    output_dim=self.config.hidden_dim,
                    normalize_inputs=self.config.normalize_inputs,
                    project_before_concat=self.config.project_before_concat
                )
            )
            
        elif fusion_type == FusionType.TRANSFORMER:
            self.fusion_module = TransformerFusion(
                TransformerFusionConfig(
                    input_dims=self.config.input_dims,
                    hidden_dim=self.config.hidden_dim,
                    num_layers=self.config.num_layers,
                    num_heads=self.config.num_heads,
                    mlp_ratio=self.config.mlp_ratio,
                    dropout=self.config.dropout,
                    activation=self.config.activation,
                    pre_norm=self.config.pre_norm
                )
            )
            
        elif fusion_type == FusionType.ADAPTIVE:
            self.fusion_module = AdaptiveFusion(
                AdaptiveFusionConfig(
                    input_dims=self.config.input_dims,
                    output_dim=self.config.hidden_dim,
                    temperature=self.config.temperature,
                    use_gating=self.config.use_gating,
                    gating_type=self.config.gating_type
                )
            )
            
        elif fusion_type == FusionType.HIERARCHICAL:
            # Hierarchical fusion (combines multiple fusion types)
            self.fusion_module = self._create_hierarchical_fusion()
            
        elif fusion_type == FusionType.TENSOR_FUSION:
            # Tensor fusion network
            self.fusion_module = self._create_tensor_fusion()
            
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def _initialize_projections(self):
        """Initialize input projections for each modality."""
        self.input_projections = nn.ModuleDict()
        
        for modality, input_dim in self.config.input_dims.items():
            if input_dim != self.config.hidden_dim:
                self.input_projections[modality] = nn.Linear(input_dim, self.config.hidden_dim)
            else:
                self.input_projections[modality] = nn.Identity()
    
    def _create_hierarchical_fusion(self) -> nn.Module:
        """Create hierarchical fusion combining multiple strategies."""
        # This is a simplified hierarchical fusion
        # In practice, you might want a more complex architecture
        
        class HierarchicalFusion(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # First level: cross-attention between modalities
                self.cross_attention = CrossAttentionFusion(
                    CrossAttentionConfig(
                        input_dims=config.input_dims,
                        hidden_dim=config.hidden_dim,
                        num_heads=config.num_heads
                    )
                )
                
                # Second level: transformer fusion
                self.transformer_fusion = TransformerFusion(
                    TransformerFusionConfig(
                        input_dims={mod: config.hidden_dim for mod in config.input_dims},
                        hidden_dim=config.hidden_dim,
                        num_layers=2,
                        num_heads=config.num_heads
                    )
                )
                
                # Third level: adaptive fusion
                self.adaptive_fusion = AdaptiveFusion(
                    AdaptiveFusionConfig(
                        input_dims={mod: config.hidden_dim for mod in config.input_dims},
                        output_dim=config.hidden_dim
                    )
                )
            
            def forward(self, modality_features, **kwargs):
                # Level 1: Cross-attention
                cross_output = self.cross_attention(modality_features)
                
                # Level 2: Transformer fusion
                transformer_output = self.transformer_fusion(cross_output.fused_features)
                
                # Level 3: Adaptive fusion
                # Prepare features for adaptive fusion
                adaptive_features = {
                    'cross': cross_output.fused_features,
                    'transformer': transformer_output.fused_features
                }
                final_output = self.adaptive_fusion(adaptive_features)
                
                return FusionOutput(
                    fused_features=final_output.fused_features,
                    attention_weights={
                        'cross': cross_output.attention_weights,
                        'transformer': transformer_output.attention_weights,
                        'adaptive': final_output.attention_weights
                    },
                    metadata={
                        'fusion_type': 'hierarchical',
                        'levels': 3
                    }
                )
        
        return HierarchicalFusion(self.config)
    
    def _create_tensor_fusion(self) -> nn.Module:
        """Create tensor fusion network."""
        class TensorFusion(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
                # Calculate tensor fusion dimension
                self.num_modalities = len(config.input_dims)
                self.input_dims = list(config.input_dims.values())
                
                # For tensor fusion, we need to handle outer products
                # This is a simplified version
                self.fusion_dim = np.prod(self.input_dims)
                if self.fusion_dim > 1000000:  # Limit size
                    self.fusion_dim = config.hidden_dim
                    self.use_compressed = True
                else:
                    self.use_compressed = False
                
                if self.use_compressed:
                    # Compressed tensor fusion
                    self.compression = nn.ModuleDict()
                    for modality, dim in config.input_dims.items():
                        self.compression[modality] = nn.Linear(dim, config.hidden_dim // self.num_modalities)
                    
                    self.fusion = nn.Linear(
                        (config.hidden_dim // self.num_modalities) * self.num_modalities,
                        config.hidden_dim
                    )
                else:
                    # Full tensor fusion (memory intensive)
                    self.fusion = nn.Linear(self.fusion_dim, config.hidden_dim)
            
            def forward(self, modality_features, **kwargs):
                if self.use_compressed:
                    # Compress each modality
                    compressed = []
                    for modality, features in modality_features.items():
                        if modality in self.compression:
                            compressed.append(self.compression[modality](features))
                        else:
                            compressed.append(features)
                    
                    # Concatenate compressed features
                    fused = torch.cat(compressed, dim=-1)
                    fused = self.fusion(fused)
                else:
                    # Compute outer product (simplified)
                    features_list = list(modality_features.values())
                    
                    # Start with ones for bias term
                    fused = torch.ones(features_list[0].shape[0], 1, device=features_list[0].device)
                    
                    for features in features_list:
                        # Add modality and outer product with existing tensor
                        ones = torch.ones(features.shape[0], 1, device=features.device)
                        features_with_bias = torch.cat([ones, features], dim=1)
                        
                        # Outer product (Kronecker product for vectors)
                        # For efficiency, we approximate with concatenation of outer products
                        fused = torch.einsum('bi,bj->bij', fused, features_with_bias)
                        fused = fused.reshape(fused.shape[0], -1)
                    
                    # Project to hidden dimension
                    fused = self.fusion(fused[:, :self.fusion_dim])  # Trim to expected size
                
                return FusionOutput(
                    fused_features=fused,
                    metadata={'fusion_type': 'tensor_fusion', 'compressed': self.use_compressed}
                )
        
        return TensorFusion(self.config)
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function module."""
        activations = {
            'none': nn.Identity(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU()
        }
        
        if activation_name not in activations:
            self.logger.warning(f"Unknown activation {activation_name}, using ReLU")
            return nn.ReLU()
        
        return activations[activation_name]
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                attention_mask: Optional[Dict[str, torch.Tensor]] = None,
                return_dict: bool = True,
                **kwargs) -> Union[torch.Tensor, FusionOutput]:
        """
        Fuse features from multiple modalities.
        
        Args:
            modality_features: Dictionary mapping modality names to features
            attention_mask: Optional attention masks for each modality
            return_dict: Whether to return FusionOutput dict
            **kwargs: Additional arguments for fusion module
            
        Returns:
            Fused features or FusionOutput
        """
        # Validate inputs
        if not modality_features:
            raise ValueError("No modality features provided")
        
        # Project inputs to common dimension if needed
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.input_projections:
                projected_features[modality] = self.input_projections[modality](features)
            else:
                projected_features[modality] = features
        
        # Apply fusion
        fusion_output = self.fusion_module(projected_features, attention_mask=attention_mask, **kwargs)
        
        # Apply output projection
        fused_features = fusion_output.fused_features
        projected_fused = self.output_projection(fused_features)
        
        # Add residual connection if enabled
        if self.config.residual_connection and fused_features.shape == projected_fused.shape:
            projected_fused = projected_fused + fused_features
        
        if return_dict:
            return FusionOutput(
                fused_features=projected_fused,
                attention_weights=fusion_output.attention_weights,
                gate_values=fusion_output.gate_values,
                intermediate_features=fusion_output.intermediate_features,
                metadata={
                    **fusion_output.metadata,
                    'output_dim': projected_fused.shape[-1],
                    'num_modalities': len(modality_features)
                }
            )
        else:
            return projected_fused
    
    def fuse_modalities(self,
                       modality_outputs: Dict[str, Any],
                       feature_key: str = 'pooled_features',
                       **kwargs) -> FusionOutput:
        """
        Convenience method to fuse encoder outputs.
        
        Args:
            modality_outputs: Dictionary of encoder outputs
            feature_key: Key to extract from encoder outputs ('features' or 'pooled_features')
            **kwargs: Additional arguments for forward pass
            
        Returns:
            Fusion output
        """
        # Extract features from encoder outputs
        modality_features = {}
        for modality, output in modality_outputs.items():
            if isinstance(output, dict):
                if feature_key in output:
                    modality_features[modality] = output[feature_key]
                elif 'features' in output:
                    modality_features[modality] = output['features']
                else:
                    raise ValueError(f"No features found in {modality} output")
            elif isinstance(output, torch.Tensor):
                modality_features[modality] = output
            else:
                raise TypeError(f"Unsupported output type for {modality}: {type(output)}")
        
        return self.forward(modality_features, **kwargs)
    
    def compute_modality_importance(self,
                                  modality_features: Dict[str, torch.Tensor],
                                  method: str = 'attention') -> Dict[str, float]:
        """
        Compute importance scores for each modality.
        
        Args:
            modality_features: Dictionary of modality features
            method: Method to compute importance ('attention', 'gating', 'variance')
            
        Returns:
            Dictionary mapping modality names to importance scores
        """
        # Forward pass to get attention/gating weights
        output = self.forward(modality_features, return_dict=True)
        
        importance_scores = {}
        
        if method == 'attention' and output.attention_weights is not None:
            # Use attention weights
            for modality, weights in output.attention_weights.items():
                if weights is not None:
                    # Average attention weights
                    importance_scores[modality] = weights.mean().item()
        
        elif method == 'gating' and output.gate_values is not None:
            # Use gating values
            for modality, gates in output.gate_values.items():
                if gates is not None:
                    importance_scores[modality] = gates.mean().item()
        
        elif method == 'variance':
            # Use feature variance as proxy for importance
            for modality, features in modality_features.items():
                importance_scores[modality] = features.var().item()
        
        else:
            # Equal importance as fallback
            num_modalities = len(modality_features)
            for modality in modality_features.keys():
                importance_scores[modality] = 1.0 / num_modalities
        
        # Normalize scores to sum to 1
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v / total for k, v in importance_scores.items()}
        
        return importance_scores
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fusion module."""
        stats = {
            'fusion_type': self.config.fusion_type.value,
            'input_modalities': list(self.config.input_dims.keys()),
            'input_dims': self.config.input_dims,
            'output_dim': self.config.output_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        }
        
        # Add fusion module specific stats
        if hasattr(self.fusion_module, 'get_statistics'):
            stats['fusion_module'] = self.fusion_module.get_statistics()
        
        return stats
    
    def save(self, path: str):
        """Save fusion module to file."""
        torch.save({
            'config': self.config.to_dict(),
            'state_dict': self.state_dict()
        }, path)
        self.logger.info(f"Saved fusion module to {path}")
    
    def load(self, path: str):
        """Load fusion module from file."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        self.logger.info(f"Loaded fusion module from {path}")

# Factory function
def create_fusion_module(fusion_type: Union[str, FusionType],
                        input_dims: Dict[str, int],
                        output_dim: int = 512,
                        **kwargs) -> MultiModalFusion:
    """
    Create a fusion module.
    
    Args:
        fusion_type: Type of fusion
        input_dims: Dictionary mapping modality names to input dimensions
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
        
    Returns:
        Fusion module
    """
    if isinstance(fusion_type, str):
        fusion_type = FusionType(fusion_type.lower())
    
    config = FusionConfig(
        fusion_type=fusion_type,
        input_dims=input_dims,
        output_dim=output_dim,
        **kwargs
    )
    
    return MultiModalFusion(config)

# Example usage
if __name__ == "__main__":
    # Test different fusion methods
    input_dims = {
        'text': 768,
        'image': 2048,
        'audio': 1024
    }
    
    batch_size = 4
    seq_len = 16
    
    # Create dummy features
    dummy_features = {
        'text': torch.randn(batch_size, seq_len, input_dims['text']),
        'image': torch.randn(batch_size, input_dims['image']),
        'audio': torch.randn(batch_size, seq_len, input_dims['audio'])
    }
    
    fusion_types = [
        FusionType.CROSS_ATTENTION,
        FusionType.CONCATENATION,
        FusionType.TRANSFORMER,
        FusionType.ADAPTIVE
    ]
    
    for fusion_type in fusion_types:
        print(f"\nTesting {fusion_type.value} fusion:")
        
        try:
            # Create fusion module
            fusion = create_fusion_module(
                fusion_type=fusion_type,
                input_dims=input_dims,
                output_dim=512,
                hidden_dim=256
            )
            
            # Test fusion
            output = fusion(dummy_features)
            
            if isinstance(output, FusionOutput):
                print(f"  Output shape: {output.fused_features.shape}")
                print(f"  Has attention weights: {output.attention_weights is not None}")
                print(f"  Has gate values: {output.gate_values is not None}")
            else:
                print(f"  Output shape: {output.shape}")
            
            # Test modality importance
            importance = fusion.compute_modality_importance(dummy_features, method='attention')
            print(f"  Modality importance: {importance}")
            
            # Get statistics
            stats = fusion.get_fusion_statistics()
            print(f"  Parameters: {stats['num_parameters']:,}")
            
        except Exception as e:
            print(f"  Error: {e}")