"""
Adaptive fusion modules for multimodal integration.
Dynamically weights modalities based on input content and context.
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
class AdaptiveFusionConfig:
    """Configuration for adaptive fusion."""
    
    # Input parameters
    input_dims: Dict[str, int] = field(default_factory=dict)
    hidden_dim: int = 512
    output_dim: int = 512
    
    # Adaptive parameters
    temperature: float = 1.0
    use_gating: bool = True
    gating_type: str = "softmax"  # "softmax", "sigmoid", "sparsemax", "gumbel_softmax"
    learnable_temperature: bool = False
    
    # Gating network
    gate_hidden_dim: int = 256
    gate_num_layers: int = 2
    gate_dropout: float = 0.1
    
    # Attention-based adaptation
    use_attention: bool = False
    attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Mixture of experts
    use_moe: bool = False
    num_experts: int = 4
    top_k: int = 2
    
    # Output
    output_activation: str = "none"
    dropout: float = 0.1
    residual_connection: bool = True

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module.
    Dynamically weights modalities based on input features.
    """
    
    def __init__(self, config: AdaptiveFusionConfig):
        super().__init__()
        self.config = config
        self.modality_dims = config.input_dims
        self.modalities = list(config.input_dims.keys())
        self.num_modalities = len(self.modalities)
        
        # Input projections
        self.input_projections = nn.ModuleDict()
        for modality, dim in self.modality_dims.items():
            self.input_projections[modality] = nn.Linear(dim, config.hidden_dim)
        
        # Temperature parameter (learnable or fixed)
        if config.learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(config.temperature))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature))
        
        # Gating network
        if config.use_gating:
            self.gate_network = GatingNetwork(
                input_dims={mod: config.hidden_dim for mod in self.modalities},
                num_modalities=self.num_modalities,
                hidden_dim=config.gate_hidden_dim,
                num_layers=config.gate_num_layers,
                dropout=config.gate_dropout,
                gating_type=config.gating_type
            )
        else:
            self.gate_network = None
        
        # Attention-based adaptation
        if config.use_attention:
            self.attention_adaptation = AttentionAdaptation(
                hidden_dim=config.hidden_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout
            )
        else:
            self.attention_adaptation = None
        
        # Mixture of experts
        if config.use_moe:
            self.mixture_of_experts = MixtureOfExperts(
                input_dim=config.hidden_dim * self.num_modalities,
                hidden_dim=config.hidden_dim,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout=config.dropout
            )
        else:
            self.mixture_of_experts = None
        
        # Output projection
        if config.output_dim != config.hidden_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(config.hidden_dim, config.output_dim),
                self._get_activation(config.output_activation),
                nn.Dropout(config.dropout)
            )
        else:
            self.output_projection = nn.Identity()
        
        # Residual connection
        self.residual_connection = config.residual_connection
        if self.residual_connection:
            self.residual_proj = None
            first_modality = self.modalities[0]
            first_dim = self.modality_dims[first_modality]
            if first_dim != config.output_dim:
                self.residual_proj = nn.Linear(first_dim, config.output_dim)
        
        logger.info(f"Initialized AdaptiveFusion with {self.num_modalities} modalities")
        logger.info(f"Gating: {config.use_gating}")
        logger.info(f"Attention adaptation: {config.use_attention}")
        logger.info(f"Mixture of experts: {config.use_moe}")
    
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
    
    def _compute_modality_weights(self,
                                 modality_features: Dict[str, torch.Tensor],
                                 method: str = "gating") -> Dict[str, torch.Tensor]:
        """
        Compute weights for each modality.
        
        Args:
            modality_features: Dictionary of modality features
            method: Weight computation method
            
        Returns:
            Dictionary of modality weights
        """
        weights = {}
        
        if method == "gating" and self.gate_network is not None:
            # Use gating network
            gate_values = self.gate_network(modality_features)
            for modality, gate in gate_values.items():
                weights[modality] = gate
        
        elif method == "attention" and self.attention_adaptation is not None:
            # Use attention-based weighting
            attention_weights = self.attention_adaptation(modality_features)
            weights.update(attention_weights)
        
        elif method == "uniform":
            # Uniform weights
            for modality in self.modalities:
                if modality in modality_features:
                    batch_size = modality_features[modality].shape[0]
                    weights[modality] = torch.ones(batch_size, 1, 
                                                  device=modality_features[modality].device) / self.num_modalities
        
        elif method == "variance":
            # Weight by feature variance (higher variance = more important)
            total_variance = 0
            modality_variances = {}
            
            for modality, feat in modality_features.items():
                # Compute variance
                if feat.dim() == 3:
                    var = feat.var(dim=[1, 2]).mean()
                else:
                    var = feat.var(dim=1).mean()
                modality_variances[modality] = var
                total_variance += var
            
            # Normalize
            if total_variance > 0:
                for modality, var in modality_variances.items():
                    weights[modality] = (var / total_variance).unsqueeze(0).unsqueeze(0)
            else:
                # Fallback to uniform
                for modality in self.modalities:
                    if modality in modality_features:
                        batch_size = modality_features[modality].shape[0]
                        weights[modality] = torch.ones(batch_size, 1,
                                                      device=modality_features[modality].device) / self.num_modalities
        
        else:
            # Default to uniform
            for modality in self.modalities:
                if modality in modality_features:
                    batch_size = modality_features[modality].shape[0]
                    weights[modality] = torch.ones(batch_size, 1,
                                                  device=modality_features[modality].device) / self.num_modalities
        
        return weights
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                attention_mask: Optional[Dict[str, torch.Tensor]] = None,
                return_dict: bool = True,
                **kwargs) -> Union[torch.Tensor, FusionOutput]:
        """
        Forward pass for adaptive fusion.
        
        Args:
            modality_features: Dictionary of modality features
            attention_mask: Optional attention masks
            return_dict: Whether to return FusionOutput
            **kwargs: Additional arguments
            
        Returns:
            Fused features or FusionOutput
        """
        # Store original features for residual connection
        original_features = modality_features.copy()
        
        # Project inputs to common dimension
        projected_features = {}
        for modality, feat in modality_features.items():
            if modality in self.input_projections:
                projected = self.input_projections[modality](feat)
                projected_features[modality] = projected
            else:
                projected_features[modality] = feat
        
        # Compute modality weights
        modality_weights = self._compute_modality_weights(projected_features, method="gating")
        
        # Apply weights to features
        weighted_features = {}
        for modality, feat in projected_features.items():
            if modality in modality_weights:
                weight = modality_weights[modality]
                # Expand weight to match feature dimensions
                if feat.dim() == 3:
                    weight = weight.unsqueeze(1)  # [batch_size, 1, 1] for sequence features
                weighted_features[modality] = feat * weight
            else:
                weighted_features[modality] = feat
        
        # Apply attention adaptation if enabled
        if self.attention_adaptation is not None:
            adapted_features = self.attention_adaptation(weighted_features)
        else:
            adapted_features = weighted_features
        
        # Pool features (handle sequence dimensions)
        pooled_features = []
        for modality in self.modalities:
            if modality in adapted_features:
                feat = adapted_features[modality]
                
                # Pool over sequence dimension if present
                if feat.dim() == 3:
                    if attention_mask is not None and modality in attention_mask:
                        mask = attention_mask[modality].unsqueeze(-1).float()
                        feat_pooled = (feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    else:
                        feat_pooled = feat.mean(dim=1)
                else:
                    feat_pooled = feat
                
                pooled_features.append(feat_pooled)
        
        # Concatenate pooled features
        if pooled_features:
            concatenated = torch.cat(pooled_features, dim=-1)
        else:
            # Fallback: zeros
            batch_size = next(iter(projected_features.values())).shape[0]
            concatenated = torch.zeros(batch_size, self.config.hidden_dim * self.num_modalities,
                                     device=next(iter(projected_features.values())).device)
        
        # Apply mixture of experts if enabled
        if self.mixture_of_experts is not None:
            fused, expert_weights = self.mixture_of_experts(concatenated)
            gate_values = {'expert_weights': expert_weights, 'modality_weights': modality_weights}
        else:
            # Simple weighted sum
            # Sum weighted features (already pooled)
            fused = sum(pooled_features) if pooled_features else concatenated
            gate_values = {'modality_weights': modality_weights}
        
        # Apply output projection
        fused = self.output_projection(fused)
        
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
                attention_weights=None,  # Adaptive fusion doesn't use attention
                gate_values=gate_values,
                intermediate_features={
                    'projected': projected_features,
                    'weighted': weighted_features,
                    'adapted': adapted_features
                },
                metadata={
                    'fusion_type': 'adaptive',
                    'num_modalities': self.num_modalities,
                    'modalities': self.modalities,
                    'gating': self.gate_network is not None,
                    'attention': self.attention_adaptation is not None,
                    'moe': self.mixture_of_experts is not None
                }
            )
        else:
            return fused
    
    def get_modality_importance(self,
                               modality_features: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute importance scores for each modality.
        
        Args:
            modality_features: Input features
            
        Returns:
            Dictionary of importance scores
        """
        # Compute weights
        weights = self._compute_modality_weights(modality_features, method="gating")
        
        # Average over batch
        importance = {}
        for modality, weight in weights.items():
            if weight is not None:
                importance[modality] = weight.mean().item()
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance

class GatingNetwork(nn.Module):
    """Gating network for adaptive fusion."""
    
    def __init__(self,
                 input_dims: Dict[str, int],
                 num_modalities: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 gating_type: str = "softmax"):
        super().__init__()
        
        self.input_dims = input_dims
        self.num_modalities = num_modalities
        self.gating_type = gating_type
        
        # Feature extractors for each modality
        self.feature_extractors = nn.ModuleDict()
        for modality, dim in input_dims.items():
            layers = []
            current_dim = dim
            
            for i in range(num_layers):
                next_dim = hidden_dim if i < num_layers - 1 else 1
                layers.append(nn.Linear(current_dim, next_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
                current_dim = next_dim
            
            self.feature_extractors[modality] = nn.Sequential(*layers)
        
        # Temperature for gating
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Gating activation
        self.gate_activation = self._get_gating_activation(gating_type)
    
    def _get_gating_activation(self, gating_type: str):
        """Get gating activation function."""
        if gating_type == "softmax":
            return lambda x: F.softmax(x / self.temperature, dim=-1)
        elif gating_type == "sigmoid":
            return nn.Sigmoid()
        elif gating_type == "sparsemax":
            return lambda x: self._sparsemax(x / self.temperature)
        elif gating_type == "gumbel_softmax":
            return lambda x: F.gumbel_softmax(x / self.temperature, tau=self.temperature, hard=False)
        else:
            raise ValueError(f"Unknown gating type: {gating_type}")
    
    def _sparsemax(self, x: torch.Tensor) -> torch.Tensor:
        """Sparsemax activation (simplified)."""
        # Simplified sparsemax implementation
        # In production, use a proper sparsemax implementation
        return F.softmax(x, dim=-1)
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute gating values for each modality.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Dictionary of gating values
        """
        # Extract features for each modality
        gate_logits = []
        modality_list = []
        
        for modality, feat in modality_features.items():
            if modality in self.feature_extractors:
                # Pool if sequence features
                if feat.dim() == 3:
                    feat_pooled = feat.mean(dim=1)  # Mean pool over sequence
                else:
                    feat_pooled = feat
                
                # Extract gate logits
                logits = self.feature_extractors[modality](feat_pooled)
                gate_logits.append(logits)
                modality_list.append(modality)
        
        # Concatenate logits
        if gate_logits:
            all_logits = torch.cat(gate_logits, dim=-1)  # [batch_size, num_modalities]
            
            # Apply gating activation
            gates = self.gate_activation(all_logits)
            
            # Split back to modalities
            gate_values = {}
            for i, modality in enumerate(modality_list):
                gate_values[modality] = gates[:, i:i+1]  # Keep dimension
        else:
            # Fallback: uniform gates
            batch_size = next(iter(modality_features.values())).shape[0]
            gate_values = {}
            for modality in modality_features.keys():
                gate_values[modality] = torch.ones(batch_size, 1,
                                                 device=next(iter(modality_features.values())).device) / len(modality_features)
        
        return gate_values

class AttentionAdaptation(nn.Module):
    """Attention-based adaptation for multimodal fusion."""
    
    def __init__(self,
                 hidden_dim: int = 512,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-attention for cross-modal interaction
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply attention-based adaptation.
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            Adapted features with attention weights
        """
        # Stack features from all modalities
        modality_names = list(modality_features.keys())
        stacked_features = torch.stack(
            [modality_features[name] for name in modality_names],
            dim=1  # [batch_size, num_modalities, seq_len, hidden_dim] or [batch_size, num_modalities, hidden_dim]
        )
        
        # Handle different dimensionalities
        if stacked_features.dim() == 4:
            # Sequence features: [batch_size, num_modalities, seq_len, hidden_dim]
            batch_size, num_modalities, seq_len, hidden_dim = stacked_features.shape
            # Reshape for attention: [batch_size * num_modalities, seq_len, hidden_dim]
            features_flat = stacked_features.view(batch_size * num_modalities, seq_len, hidden_dim)
            
            # Apply self-attention
            attended, attn_weights = self.self_attn(features_flat, features_flat, features_flat)
            
            # Reshape back
            attended = attended.view(batch_size, num_modalities, seq_len, hidden_dim)
            
            # Apply feed-forward
            attended_flat = attended.view(batch_size * num_modalities, seq_len, hidden_dim)
            ffn_output = self.ffn(attended_flat)
            ffn_output = ffn_output.view(batch_size, num_modalities, seq_len, hidden_dim)
            
            # Add residual and normalize
            output = self.norm(attended + ffn_output)
            
        else:
            # Pooled features: [batch_size, num_modalities, hidden_dim]
            batch_size, num_modalities, hidden_dim = stacked_features.shape
            
            # Apply self-attention
            attended, attn_weights = self.self_attn(stacked_features, stacked_features, stacked_features)
            
            # Apply feed-forward
            ffn_output = self.ffn(attended)
            
            # Add residual and normalize
            output = self.norm(attended + ffn_output)
        
        # Split back to modalities
        adapted_features = {}
        for i, name in enumerate(modality_names):
            adapted_features[name] = output[:, i, ...]
        
        return adapted_features

class MixtureOfExperts(nn.Module):
    """Mixture of experts for adaptive fusion."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_experts: int = 4,
                 top_k: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Temperature for gating
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for mixture of experts.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (output, expert_weights)
        """
        # Compute gate logits
        gate_logits = self.gate(x) / self.temperature
        
        # Select top-k experts
        if self.top_k < self.num_experts:
            # Get top-k expert indices
            top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
            
            # Compute softmax over top-k
            top_k_gates = F.softmax(top_k_logits, dim=-1)
            
            # Create sparse gate values
            gates = torch.zeros_like(gate_logits)
            gates.scatter_(-1, top_k_indices, top_k_gates)
        else:
            # Use all experts
            gates = F.softmax(gate_logits, dim=-1)
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
        
        # Weighted sum of expert outputs
        gates_expanded = gates.unsqueeze(-1)  # [batch_size, num_experts, 1]
        output = torch.sum(expert_outputs * gates_expanded, dim=1)  # [batch_size, hidden_dim]
        
        return output, gates

# Example usage
if __name__ == "__main__":
    # Test adaptive fusion
    config = AdaptiveFusionConfig(
        input_dims={
            'text': 768,
            'image': 2048,
            'audio': 1024
        },
        hidden_dim=512,
        use_gating=True,
        use_attention=True,
        use_moe=False
    )
    
    fusion = AdaptiveFusion(config)
    
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
    
    # Test modality importance
    importance = fusion.get_modality_importance(dummy_features)
    print(f"\nModality importance: {importance}")