"""
Concatenation-based fusion modules for multimodal integration.
Includes simple concatenation, weighted concatenation, and gated concatenation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
import logging
import math

from ..alignment import FusionConfig, FusionOutput

logger = logging.getLogger(__name__)

@dataclass
class ConcatenationConfig:
    """Configuration for concatenation fusion."""
    
    # Input parameters
    input_dims: Dict[str, int] = field(default_factory=dict)
    output_dim: int = 512
    
    # Concatenation parameters
    concatenation_dim: int = -1  # Dimension to concatenate along
    normalize_inputs: bool = True
    project_before_concat: bool = True
    
    # Weighted concatenation
    use_weighted_concat: bool = False
    learnable_weights: bool = True
    temperature: float = 1.0
    
    # Gated concatenation
    use_gated_concat: bool = False
    gating_type: str = "sigmoid"  # "sigmoid", "softmax", "sparsemax"
    
    # Output
    output_activation: str = "none"  # "none", "relu", "tanh", "sigmoid"
    dropout: float = 0.1
    residual_connection: bool = False

class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion module.
    Concatenates features from multiple modalities along specified dimension.
    """
    
    def __init__(self, config: ConcatenationConfig):
        super().__init__()
        self.config = config
        self.modality_dims = config.input_dims
        self.modalities = list(config.input_dims.keys())
        self.num_modalities = len(self.modalities)
        
        # Input projections (optional)
        if config.project_before_concat:
            self.input_projections = nn.ModuleDict()
            for modality, dim in self.modality_dims.items():
                # Project to a common dimension or keep original
                proj_dim = config.output_dim // self.num_modalities
                self.input_projections[modality] = nn.Linear(dim, proj_dim)
        else:
            self.input_projections = None
        
        # Weighted concatenation (optional)
        if config.use_weighted_concat:
            self.weighted_concat = WeightedConcatenation(
                num_modalities=self.num_modalities,
                learnable_weights=config.learnable_weights,
                temperature=config.temperature
            )
        else:
            self.weighted_concat = None
        
        # Gated concatenation (optional)
        if config.use_gated_concat:
            self.gated_concat = GatedConcatenation(
                input_dims=self.modality_dims,
                gating_type=config.gating_type
            )
        else:
            self.gated_concat = None
        
        # Calculate total concatenated dimension
        if config.project_before_concat:
            self.concat_dim = config.output_dim
        else:
            self.concat_dim = sum(self.modality_dims.values())
        
        # Output projection (if needed)
        if self.concat_dim != config.output_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(self.concat_dim, config.output_dim),
                self._get_activation(config.output_activation),
                nn.Dropout(config.dropout)
            )
        else:
            self.output_projection = nn.Identity()
        
        # Residual connection (if enabled and dimensions match)
        self.residual_connection = config.residual_connection
        if self.residual_connection:
            # We'll use the first modality as residual
            self.residual_proj = None
            first_modality = self.modalities[0]
            first_dim = self.modality_dims[first_modality]
            if first_dim != config.output_dim:
                self.residual_proj = nn.Linear(first_dim, config.output_dim)
        
        logger.info(f"Initialized ConcatenationFusion with {self.num_modalities} modalities")
        logger.info(f"Concatenation dim: {self.config.concatenation_dim}")
        logger.info(f"Output dim: {config.output_dim}")
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'none': nn.Identity(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation_name, nn.Identity())
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                attention_mask: Optional[Dict[str, torch.Tensor]] = None,
                return_dict: bool = True,
                **kwargs) -> Union[torch.Tensor, FusionOutput]:
        """
        Forward pass for concatenation fusion.
        
        Args:
            modality_features: Dictionary of modality features
            attention_mask: Optional attention masks (not used in simple concat)
            return_dict: Whether to return FusionOutput
            **kwargs: Additional arguments
            
        Returns:
            Fused features or FusionOutput
        """
        # Store original features for residual connection
        original_features = modality_features.copy()
        
        # Project inputs if needed
        if self.input_projections is not None:
            projected_features = {}
            for modality, feat in modality_features.items():
                if modality in self.input_projections:
                    projected_features[modality] = self.input_projections[modality](feat)
                else:
                    projected_features[modality] = feat
            modality_features = projected_features
        
        # Normalize inputs if requested
        if self.config.normalize_inputs:
            normalized_features = {}
            for modality, feat in modality_features.items():
                # Handle 2D and 3D tensors
                if feat.dim() == 3:
                    # Normalize each position independently
                    normalized = F.normalize(feat, p=2, dim=-1)
                else:
                    normalized = F.normalize(feat, p=2, dim=-1)
                normalized_features[modality] = normalized
            modality_features = normalized_features
        
        # Apply weighted concatenation if enabled
        if self.weighted_concat is not None:
            modality_features, weights = self.weighted_concat(modality_features)
            gate_values = {'weights': weights}
        else:
            gate_values = None
        
        # Apply gated concatenation if enabled
        if self.gated_concat is not None:
            modality_features, gates = self.gated_concat(modality_features)
            gate_values = gates if gate_values is None else {**gate_values, **gates}
        
        # Extract features in order of modalities
        features_list = []
        for modality in self.modalities:
            if modality in modality_features:
                feat = modality_features[modality]
                
                # Handle sequence features (3D tensors)
                if feat.dim() == 3:
                    # Pool over sequence dimension
                    if attention_mask is not None and modality in attention_mask:
                        mask = attention_mask[modality].unsqueeze(-1).float()
                        feat_pooled = (feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    else:
                        feat_pooled = feat.mean(dim=1)
                else:
                    feat_pooled = feat
                
                features_list.append(feat_pooled)
        
        # Concatenate along specified dimension
        if features_list:
            if self.config.concatenation_dim == -1:
                # Concatenate along last dimension
                concatenated = torch.cat(features_list, dim=-1)
            else:
                # Concatenate along specified dimension
                concatenated = torch.cat(features_list, dim=self.config.concatenation_dim)
        else:
            # Fallback: use zeros
            batch_size = next(iter(modality_features.values())).shape[0]
            concatenated = torch.zeros(batch_size, self.concat_dim, 
                                     device=next(iter(modality_features.values())).device)
        
        # Apply output projection
        fused = self.output_projection(concatenated)
        
        # Add residual connection if enabled
        if self.residual_connection and self.modalities:
            first_modality = self.modalities[0]
            if first_modality in original_features:
                residual = original_features[first_modality]
                # Pool if needed
                if residual.dim() == 3:
                    residual = residual.mean(dim=1)
                
                # Project if dimensions don't match
                if self.residual_proj is not None:
                    residual = self.residual_proj(residual)
                
                # Add residual
                fused = fused + residual
        
        if return_dict:
            return FusionOutput(
                fused_features=fused,
                attention_weights=None,  # Concatenation doesn't use attention
                gate_values=gate_values,
                intermediate_features=modality_features,
                metadata={
                    'fusion_type': 'concatenation',
                    'num_modalities': self.num_modalities,
                    'modalities': self.modalities,
                    'weighted': self.weighted_concat is not None,
                    'gated': self.gated_concat is not None
                }
            )
        else:
            return fused
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fusion module."""
        stats = {
            'fusion_type': 'concatenation',
            'modalities': self.modalities,
            'input_dims': self.modality_dims,
            'output_dim': self.config.output_dim,
            'concat_dim': self.concat_dim,
            'weighted': self.weighted_concat is not None,
            'gated': self.gated_concat is not None,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'num_trainable_parameters': sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        }
        return stats

class WeightedConcatenation(nn.Module):
    """
    Weighted concatenation fusion.
    Learns weights for each modality during concatenation.
    """
    
    def __init__(self,
                 num_modalities: int,
                 learnable_weights: bool = True,
                 temperature: float = 1.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.learnable_weights = learnable_weights
        self.temperature = temperature
        
        if learnable_weights:
            # Learnable weights for each modality
            self.weights = nn.Parameter(torch.ones(num_modalities))
            # Initialize with uniform weights
            nn.init.constant_(self.weights, 1.0 / num_modalities)
        else:
            self.weights = None
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply weighted concatenation.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Tuple of (weighted_features, weights)
        """
        if self.weights is not None:
            # Use learnable weights
            weights = F.softmax(self.weights / self.temperature, dim=0)
        else:
            # Uniform weights
            weights = torch.ones(self.num_modalities, 
                               device=next(iter(modality_features.values())).device)
            weights = weights / self.num_modalities
        
        # Apply weights to features
        weighted_features = {}
        for i, (modality, feat) in enumerate(modality_features.items()):
            if i < len(weights):
                weighted_features[modality] = feat * weights[i]
            else:
                weighted_features[modality] = feat
        
        return weighted_features, weights
    
    def get_weights(self) -> torch.Tensor:
        """Get current modality weights."""
        if self.weights is not None:
            return F.softmax(self.weights / self.temperature, dim=0)
        else:
            return torch.ones(self.num_modalities) / self.num_modalities

class GatedConcatenation(nn.Module):
    """
    Gated concatenation fusion.
    Uses gating mechanisms to control information flow from each modality.
    """
    
    def __init__(self,
                 input_dims: Dict[str, int],
                 gating_type: str = "sigmoid"):
        super().__init__()
        self.input_dims = input_dims
        self.modalities = list(input_dims.keys())
        self.gating_type = gating_type
        
        # Create gating networks for each modality
        self.gate_networks = nn.ModuleDict()
        for modality, dim in input_dims.items():
            # Simple MLP for gating
            self.gate_networks[modality] = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(),
                nn.Linear(dim // 2, 1),
                self._get_gating_activation(gating_type)
            )
    
    def _get_gating_activation(self, gating_type: str) -> nn.Module:
        """Get gating activation function."""
        if gating_type == "sigmoid":
            return nn.Sigmoid()
        elif gating_type == "softmax":
            return nn.Softmax(dim=-1)
        elif gating_type == "sparsemax":
            return Sparsemax(dim=-1)
        else:
            raise ValueError(f"Unknown gating type: {gating_type}")
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Apply gated concatenation.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Tuple of (gated_features, gate_values)
        """
        gated_features = {}
        gate_values = {}
        
        for modality, feat in modality_features.items():
            if modality in self.gate_networks:
                # Compute gate value
                if feat.dim() == 3:
                    # For sequence features, compute gate per position and average
                    batch_size, seq_len, feat_dim = feat.shape
                    feat_flat = feat.view(batch_size * seq_len, feat_dim)
                    gate_flat = self.gate_networks[modality](feat_flat)
                    gate = gate_flat.view(batch_size, seq_len, 1)
                else:
                    # For pooled features
                    gate = self.gate_networks[modality](feat)
                
                # Apply gate
                gated_feat = feat * gate
                
                gated_features[modality] = gated_feat
                gate_values[modality] = gate.squeeze(-1) if gate.dim() > 1 else gate
            else:
                gated_features[modality] = feat
        
        return gated_features, gate_values

class Sparsemax(nn.Module):
    """
    Sparsemax activation function.
    Produces sparse probability distributions.
    """
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sparsemax forward pass."""
        # Implementation of Sparsemax
        # For simplicity, we'll use a soft approximation
        # In production, you might want a proper sparsemax implementation
        return F.softmax(x, dim=self.dim)
    
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        """Sparsemax backward pass."""
        # Simplified gradient
        return grad_output

# Example usage
if __name__ == "__main__":
    # Test concatenation fusion
    config = ConcatenationConfig(
        input_dims={
            'text': 768,
            'image': 2048,
            'audio': 1024
        },
        output_dim=512,
        project_before_concat=True,
        use_weighted_concat=True,
        use_gated_concat=False
    )
    
    fusion = ConcatenationFusion(config)
    
    # Create dummy inputs
    batch_size = 2
    dummy_features = {
        'text': torch.randn(batch_size, 768),
        'image': torch.randn(batch_size, 2048),
        'audio': torch.randn(batch_size, 1024)
    }
    
    # Test forward pass
    output = fusion(dummy_features, return_dict=True)
    
    print(f"Input modalities: {list(dummy_features.keys())}")
    print(f"Output shape: {output.fused_features.shape}")
    print(f"Has gate values: {output.gate_values is not None}")
    print(f"Metadata: {output.metadata}")
    
    # Test statistics
    stats = fusion.get_statistics()
    print(f"\nFusion statistics:")
    print(f"  Parameters: {stats['num_parameters']:,}")
    print(f"  Trainable parameters: {stats['num_trainable_parameters']:,}")