"""
Utility modules for multimodal fusion.
Includes attention utilities, gating mechanisms, normalization, and fusion helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import math

# Attention utilities
class AttentionUtils:
    """Utilities for attention mechanisms."""
    
    @staticmethod
    def scaled_dot_product_attention(query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None,
                                     dropout: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, dim]
            key: Key tensor [batch_size, seq_len_k, dim]
            value: Value tensor [batch_size, seq_len_v, dim]
            mask: Attention mask
            dropout: Dropout probability
            
        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if dropout > 0:
            attn_weights = F.dropout(attn_weights, p=dropout, training=True)
        
        output = torch.matmul(attn_weights, value)
        return output, attn_weights
    
    @staticmethod
    def multi_head_attention_forward(query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     num_heads: int,
                                     mask: Optional[torch.Tensor] = None,
                                     dropout: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass.
        
        Args:
            query: Query tensor [batch_size, seq_len, dim]
            key: Key tensor [batch_size, seq_len, dim]
            value: Value tensor [batch_size, seq_len, dim]
            num_heads: Number of attention heads
            mask: Attention mask
            dropout: Dropout probability
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, dim = query.shape
        head_dim = dim // num_heads
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Apply attention
        output, attn_weights = AttentionUtils.scaled_dot_product_attention(
            query, key, value, mask, dropout
        )
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        
        return output, attn_weights
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    @staticmethod
    def create_padding_mask(seq_lengths: List[int], max_len: int, device: torch.device = None) -> torch.Tensor:
        """Create padding mask from sequence lengths."""
        batch_size = len(seq_lengths)
        mask = torch.zeros(batch_size, max_len, device=device)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = 1
        return mask.bool()

# Gating utilities
class GatingUtils:
    """Utilities for gating mechanisms."""
    
    @staticmethod
    def sigmoid_gate(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sigmoid gating function."""
        return torch.sigmoid(x / temperature)
    
    @staticmethod
    def softmax_gate(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Softmax gating function."""
        return F.softmax(x / temperature, dim=-1)
    
    @staticmethod
    def sparsemax_gate(x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sparsemax gating function (approximation)."""
        # Simplified sparsemax - in production use proper implementation
        return F.softmax(x / temperature, dim=-1)
    
    @staticmethod
    def gumbel_softmax_gate(logits: torch.Tensor,
                           temperature: float = 1.0,
                           hard: bool = False) -> torch.Tensor:
        """Gumbel softmax gating function."""
        return F.gumbel_softmax(logits, tau=temperature, hard=hard)
    
    @staticmethod
    def compute_gate_entropy(gates: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute entropy of gate distribution."""
        return -torch.sum(gates * torch.log(gates + eps), dim=-1)
    
    @staticmethod
    def compute_gate_sparsity(gates: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Compute sparsity of gate distribution."""
        return (gates < threshold).float().mean(dim=-1)

# Normalization utilities
class NormalizationUtils:
    """Utilities for normalization operations."""
    
    @staticmethod
    def layer_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
        """Layer normalization."""
        mean = x.mean(dim=dim, keepdim=True)
        std = x.std(dim=dim, keepdim=True)
        return (x - mean) / (std + eps)
    
    @staticmethod
    def instance_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
        """Instance normalization."""
        if x.dim() != 3:
            raise ValueError("Instance norm expects 3D input [batch_size, seq_len, dim]")
        
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)
    
    @staticmethod
    def batch_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
        """Batch normalization."""
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return (x - mean) / (std + eps)
    
    @staticmethod
    def normalize_features(features: Dict[str, torch.Tensor],
                          norm_type: str = "layer",
                          dim: int = -1) -> Dict[str, torch.Tensor]:
        """Normalize features from multiple modalities."""
        normalized = {}
        
        for modality, feat in features.items():
            if norm_type == "layer":
                normalized[modality] = NormalizationUtils.layer_norm(feat, dim)
            elif norm_type == "instance":
                normalized[modality] = NormalizationUtils.instance_norm(feat, dim)
            elif norm_type == "batch":
                normalized[modality] = NormalizationUtils.batch_norm(feat, dim)
            elif norm_type == "l2":
                normalized[modality] = F.normalize(feat, p=2, dim=dim)
            else:
                normalized[modality] = feat
        
        return normalized

# Fusion utilities
class FusionUtils:
    """Utilities for fusion operations."""
    
    @staticmethod
    def concatenate_features(features: Dict[str, torch.Tensor],
                           dim: int = -1,
                           modalities: Optional[List[str]] = None) -> torch.Tensor:
        """Concatenate features from multiple modalities."""
        if modalities is None:
            modalities = list(features.keys())
        
        feature_list = []
        for modality in modalities:
            if modality in features:
                feature_list.append(features[modality])
        
        if not feature_list:
            raise ValueError("No features to concatenate")
        
        return torch.cat(feature_list, dim=dim)
    
    @staticmethod
    def weighted_sum(features: Dict[str, torch.Tensor],
                    weights: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Weighted sum of features."""
        if weights is None:
            # Uniform weights
            weights = {mod: torch.ones_like(feat) for mod, feat in features.items()}
        
        weighted_sum = None
        for modality, feat in features.items():
            if modality in weights:
                weight = weights[modality]
                # Ensure weight has correct dimensions
                if weight.dim() < feat.dim():
                    weight = weight.unsqueeze(-1)
                
                weighted_feat = feat * weight
                if weighted_sum is None:
                    weighted_sum = weighted_feat
                else:
                    weighted_sum = weighted_sum + weighted_feat
        
        if weighted_sum is None:
            raise ValueError("No features to sum")
        
        return weighted_sum
    
    @staticmethod
    def elementwise_product(features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Element-wise product of features."""
        product = None
        for modality, feat in features.items():
            if product is None:
                product = feat
            else:
                product = product * feat
        
        if product is None:
            raise ValueError("No features to multiply")
        
        return product
    
    @staticmethod
    def compute_feature_similarity(features1: torch.Tensor,
                                  features2: torch.Tensor,
                                  similarity_type: str = "cosine") -> torch.Tensor:
        """Compute similarity between two feature sets."""
        if similarity_type == "cosine":
            # Cosine similarity
            features1_norm = F.normalize(features1, p=2, dim=-1)
            features2_norm = F.normalize(features2, p=2, dim=-1)
            similarity = torch.matmul(features1_norm, features2_norm.transpose(-2, -1))
        
        elif similarity_type == "dot":
            # Dot product similarity
            similarity = torch.matmul(features1, features2.transpose(-2, -1))
        
        elif similarity_type == "euclidean":
            # Euclidean distance (converted to similarity)
            # similarity = 1 / (1 + distance)
            batch_size1, seq_len1, dim = features1.shape
            batch_size2, seq_len2, _ = features2.shape
            
            features1_exp = features1.unsqueeze(2)  # [batch_size1, seq_len1, 1, dim]
            features2_exp = features2.unsqueeze(1)  # [batch_size2, 1, seq_len2, dim]
            
            # Expand for broadcasting
            features1_exp = features1_exp.expand(-1, -1, seq_len2, -1)
            features2_exp = features2_exp.expand(batch_size1, seq_len1, -1, -1)
            
            distances = torch.sqrt(((features1_exp - features2_exp) ** 2).sum(dim=-1))
            similarity = 1.0 / (1.0 + distances)
        
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        return similarity
    
    @staticmethod
    def compute_modality_correlation(features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute correlation matrix between modalities."""
        modality_names = list(features.keys())
        num_modalities = len(modality_names)
        
        # Stack features
        feature_list = []
        for name in modality_names:
            feat = features[name]
            # Pool if sequence features
            if feat.dim() == 3:
                feat = feat.mean(dim=1)
            feature_list.append(feat)
        
        stacked_features = torch.stack(feature_list, dim=0)  # [num_modalities, batch_size, dim]
        stacked_features = stacked_features.transpose(0, 1)  # [batch_size, num_modalities, dim]
        
        # Compute correlation
        batch_size, num_modalities, dim = stacked_features.shape
        
        # Center features
        features_centered = stacked_features - stacked_features.mean(dim=2, keepdim=True)
        
        # Compute covariance
        covariance = torch.matmul(
            features_centered.transpose(1, 2),  # [batch_size, dim, num_modalities]
            features_centered  # [batch_size, num_modalities, dim]
        ) / (dim - 1)
        
        # Compute correlation from covariance
        std = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2))  # [batch_size, num_modalities]
        std_prod = torch.matmul(std.unsqueeze(2), std.unsqueeze(1))  # [batch_size, num_modalities, num_modalities]
        correlation = covariance / (std_prod + 1e-8)
        
        # Average over batch
        correlation = correlation.mean(dim=0)
        
        # Create dictionary
        correlation_dict = {}
        for i, mod_i in enumerate(modality_names):
            for j, mod_j in enumerate(modality_names):
                correlation_dict[f"{mod_i}_{mod_j}"] = correlation[i, j]
        
        return correlation_dict

# Positional encoding utilities
class PositionalEncodingUtils:
    """Utilities for positional encoding."""
    
    @staticmethod
    def sinusoidal_positional_encoding(seq_len: int,
                                      d_model: int,
                                      device: torch.device = None) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # [1, seq_len, d_model]
    
    @staticmethod
    def learned_positional_encoding(seq_len: int,
                                   d_model: int,
                                   max_len: int = 5000) -> nn.Embedding:
        """Create learned positional encoding."""
        return nn.Embedding(max_len, d_model)
    
    @staticmethod
    def relative_positional_encoding(seq_len: int,
                                    d_model: int,
                                    max_relative_position: int = 64) -> nn.Embedding:
        """Create relative positional encoding."""
        return nn.Embedding(2 * max_relative_position + 1, d_model)
    
    @staticmethod
    def rotary_positional_encoding(seq_len: int,
                                  d_model: int,
                                  theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rotary positional encoding (RoPE)."""
        # Simplified RoPE implementation
        position = torch.arange(seq_len, dtype=torch.float, device=None).unsqueeze(1)
        dim_idx = torch.arange(0, d_model, 2, dtype=torch.float, device=None).unsqueeze(0)
        
        div_term = torch.pow(theta, -2 * dim_idx / d_model)
        angle = position * div_term
        
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        
        return cos, sin

# Export utilities
__all__ = [
    'AttentionUtils',
    'GatingUtils',
    'NormalizationUtils',
    'FusionUtils',
    'PositionalEncodingUtils'
]

# Example usage
if __name__ == "__main__":
    # Test utilities
    batch_size = 2
    seq_len = 10
    dim = 512
    
    # Create dummy features
    features = {
        'text': torch.randn(batch_size, seq_len, dim),
        'image': torch.randn(batch_size, dim),
        'audio': torch.randn(batch_size, seq_len, dim // 2)
    }
    
    # Test concatenation
    concatenated = FusionUtils.concatenate_features(features, dim=-1)
    print(f"Concatenated shape: {concatenated.shape}")
    
    # Test normalization
    normalized = NormalizationUtils.normalize_features(features, norm_type="layer")
    print(f"Normalized features keys: {list(normalized.keys())}")
    
    # Test attention
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)
    
    output, attn_weights = AttentionUtils.scaled_dot_product_attention(query, key, value)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Test positional encoding
    pe = PositionalEncodingUtils.sinusoidal_positional_encoding(seq_len, dim)
    print(f"Positional encoding shape: {pe.shape}")