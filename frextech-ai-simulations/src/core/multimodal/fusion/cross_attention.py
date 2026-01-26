"""
Cross-attention fusion modules for multimodal integration.
Includes multi-head cross-attention and cross-attention blocks.
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
class CrossAttentionConfig:
    """Configuration for cross-attention fusion."""
    
    # Attention parameters
    input_dims: Dict[str, int] = field(default_factory=dict)
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_bias: bool = True
    
    # Positional encoding
    use_positional_encoding: bool = True
    max_seq_length: int = 512
    positional_encoding_type: str = "learned"  # "learned", "sinusoidal"
    
    # Layer parameters
    num_layers: int = 4
    mlp_ratio: int = 4
    activation: str = "gelu"
    pre_norm: bool = True
    
    # Cross-attention direction
    query_modality: Optional[str] = None  # If None, uses all modalities as both query and key/value
    bidirectional: bool = True
    
    # Output
    output_projection: bool = True
    residual_connection: bool = True

class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention layer."""
    
    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 value_dim: int,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1):
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for proj in [self.query_proj, self.key_proj, self.value_proj]:
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for cross-attention.
        
        Args:
            query: Query tensor [batch_size, query_len, query_dim]
            key: Key tensor [batch_size, key_len, key_dim]
            value: Value tensor [batch_size, value_len, value_dim]
            attention_mask: Attention mask [batch_size, query_len, key_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Attention output [batch_size, query_len, hidden_dim]
            (Optional) Attention weights [batch_size, num_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # Project inputs
        q = self.query_proj(query)  # [batch_size, query_len, hidden_dim]
        k = self.key_proj(key)      # [batch_size, key_len, hidden_dim]
        v = self.value_proj(value)  # [batch_size, value_len, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for heads
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, key_len]
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.num_heads * self.head_dim
        )
        
        # Output projection
        output = self.output_proj(attn_output)
        output = self.proj_dropout(output)
        
        if return_attention:
            return output, attn_weights
        else:
            return output

class CrossAttentionBlock(nn.Module):
    """Cross-attention block with feed-forward network."""
    
    def __init__(self,
                 query_dim: int,
                 key_dim: int,
                 value_dim: int,
                 hidden_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 activation: str = "gelu",
                 pre_norm: bool = True):
        super().__init__()
        
        self.pre_norm = pre_norm
        self.hidden_dim = hidden_dim
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Cross-attention
        self.cross_attention = MultiHeadCrossAttention(
            query_dim=query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh()
        }
        return activations.get(activation_name, nn.GELU())
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for cross-attention block.
        
        Args:
            query: Query features
            key: Key features
            value: Value features
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output features (and optionally attention weights)
        """
        residual = query
        
        # Apply layer norm before attention if using pre-norm
        if self.pre_norm:
            query_norm = self.norm1(query)
        else:
            query_norm = query
        
        # Cross-attention
        if return_attention:
            attn_output, attention_weights = self.cross_attention(
                query_norm, key, value, attention_mask, return_attention=True
            )
        else:
            attn_output = self.cross_attention(
                query_norm, key, value, attention_mask, return_attention=False
            )
        
        # Add residual
        if self.pre_norm:
            # Project residual if dimensions don't match
            if residual.shape[-1] != attn_output.shape[-1]:
                residual = nn.Linear(residual.shape[-1], attn_output.shape[-1])(residual)
            
            output = residual + self.dropout(attn_output)
            
            # Apply layer norm after attention if not using pre-norm
            if not self.pre_norm:
                output = self.norm1(output)
        else:
            output = self.norm1(residual + self.dropout(attn_output))
        
        # Feed-forward network
        ff_residual = output
        
        if self.pre_norm:
            output_norm = self.norm2(output)
            ff_output = self.ffn(output_norm)
            output = ff_residual + self.dropout(ff_output)
        else:
            ff_output = self.ffn(output)
            output = self.norm2(ff_residual + self.dropout(ff_output))
        
        if return_attention:
            return output, attention_weights
        else:
            return output

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion for multiple modalities.
    Supports bidirectional attention between modalities.
    """
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.config = config
        self.modality_dims = config.input_dims
        self.modalities = list(config.input_dims.keys())
        self.num_modalities = len(self.modalities)
        
        # Create modality embeddings
        self.modality_embeddings = nn.ParameterDict()
        for modality in self.modalities:
            self.modality_embeddings[modality] = nn.Parameter(
                torch.randn(1, 1, config.hidden_dim) * 0.02
            )
        
        # Input projections
        self.input_projections = nn.ModuleDict()
        for modality, dim in self.modality_dims.items():
            self.input_projections[modality] = nn.Linear(dim, config.hidden_dim)
        
        # Positional encoding (if needed)
        if config.use_positional_encoding:
            if config.positional_encoding_type == "learned":
                self.positional_encoding = nn.Embedding(
                    config.max_seq_length, config.hidden_dim
                )
            else:  # sinusoidal
                self.positional_encoding = SinusoidalPositionalEncoding(
                    config.hidden_dim, max_len=config.max_seq_length
                )
        else:
            self.positional_encoding = None
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList()
        
        for _ in range(config.num_layers):
            layer = nn.ModuleDict()
            
            if config.query_modality is not None:
                # Single query modality
                query_mod = config.query_modality
                for key_mod in self.modalities:
                    if key_mod != query_mod or config.bidirectional:
                        layer_name = f"{query_mod}_to_{key_mod}"
                        layer[layer_name] = CrossAttentionBlock(
                            query_dim=config.hidden_dim,
                            key_dim=config.hidden_dim,
                            value_dim=config.hidden_dim,
                            hidden_dim=config.hidden_dim,
                            num_heads=config.num_heads,
                            mlp_ratio=config.mlp_ratio,
                            dropout=config.dropout,
                            attention_dropout=config.attention_dropout,
                            activation=config.activation,
                            pre_norm=config.pre_norm
                        )
            else:
                # All-to-all attention
                for query_mod in self.modalities:
                    for key_mod in self.modalities:
                        if query_mod != key_mod or config.bidirectional:
                            layer_name = f"{query_mod}_to_{key_mod}"
                            layer[layer_name] = CrossAttentionBlock(
                                query_dim=config.hidden_dim,
                                key_dim=config.hidden_dim,
                                value_dim=config.hidden_dim,
                                hidden_dim=config.hidden_dim,
                                num_heads=config.num_heads,
                                mlp_ratio=config.mlp_ratio,
                                dropout=config.dropout,
                                attention_dropout=config.attention_dropout,
                                activation=config.activation,
                                pre_norm=config.pre_norm
                            )
            
            self.cross_attention_layers.append(layer)
        
        # Output projection
        if config.output_projection:
            self.output_projection = nn.Linear(
                config.hidden_dim * self.num_modalities,
                config.hidden_dim
            )
        else:
            self.output_projection = nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized CrossAttentionFusion with {self.num_modalities} modalities")
        logger.info(f"Modalities: {self.modalities}")
        logger.info(f"Hidden dim: {config.hidden_dim}")
        logger.info(f"Num layers: {config.num_layers}")
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize modality embeddings
        for param in self.modality_embeddings.values():
            nn.init.normal_(param, mean=0.0, std=0.02)
        
        # Initialize input projections
        for proj in self.input_projections.values():
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        # Initialize output projection
        if hasattr(self.output_projection, 'weight'):
            nn.init.xavier_uniform_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)
    
    def _add_modality_embeddings(self,
                                features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add modality-specific embeddings to features."""
        embedded_features = {}
        
        for modality, feat in features.items():
            if modality in self.modality_embeddings:
                # Expand modality embedding to match batch size
                batch_size = feat.shape[0]
                modality_emb = self.modality_embeddings[modality].expand(batch_size, -1, -1)
                
                # Add to features
                if feat.dim() == 2:
                    # [batch_size, hidden_dim] -> add to all positions
                    embedded = feat.unsqueeze(1) + modality_emb
                else:
                    # [batch_size, seq_len, hidden_dim] -> add to each position
                    seq_len = feat.shape[1]
                    modality_emb = modality_emb.expand(-1, seq_len, -1)
                    embedded = feat + modality_emb
                
                embedded_features[modality] = embedded
            else:
                embedded_features[modality] = feat
        
        return embedded_features
    
    def _add_positional_encoding(self,
                                features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add positional encoding to features."""
        if self.positional_encoding is None:
            return features
        
        encoded_features = {}
        
        for modality, feat in features.items():
            if feat.dim() == 3:  # Has sequence dimension
                seq_len = feat.shape[1]
                positions = torch.arange(seq_len, device=feat.device).unsqueeze(0)
                
                if isinstance(self.positional_encoding, nn.Embedding):
                    pos_enc = self.positional_encoding(positions)
                else:  # Sinusoidal
                    pos_enc = self.positional_encoding(feat)
                
                encoded_features[modality] = feat + pos_enc
            else:
                encoded_features[modality] = feat
        
        return encoded_features
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                attention_mask: Optional[Dict[str, torch.Tensor]] = None,
                return_dict: bool = True,
                **kwargs) -> Union[torch.Tensor, FusionOutput]:
        """
        Forward pass for cross-attention fusion.
        
        Args:
            modality_features: Dictionary of modality features
            attention_mask: Optional attention masks
            return_dict: Whether to return FusionOutput
            **kwargs: Additional arguments
            
        Returns:
            Fused features or FusionOutput
        """
        # Project inputs to common dimension
        projected_features = {}
        for modality, feat in modality_features.items():
            if modality in self.input_projections:
                projected = self.input_projections[modality](feat)
                projected_features[modality] = projected
            else:
                projected_features[modality] = feat
        
        # Add modality embeddings
        embedded_features = self._add_modality_embeddings(projected_features)
        
        # Add positional encoding
        encoded_features = self._add_positional_encoding(embedded_features)
        
        # Initialize attention weights storage
        all_attention_weights = {}
        
        # Apply cross-attention layers
        current_features = encoded_features.copy()
        
        for layer_idx, layer in enumerate(self.cross_attention_layers):
            # Store updated features
            updated_features = current_features.copy()
            layer_attention_weights = {}
            
            # Apply all cross-attention operations in this layer
            for attn_name, attn_block in layer.items():
                # Parse attention direction
                if '_to_' in attn_name:
                    query_mod, key_mod = attn_name.split('_to_')
                else:
                    # Handle other naming conventions
                    query_mod, key_mod = attn_name.split('_query_')[0], attn_name.split('_key_')[1]
                
                if query_mod not in current_features or key_mod not in current_features:
                    continue
                
                # Get features
                query = current_features[query_mod]
                key = current_features[key_mod]
                value = current_features[key_mod]  # Usually same as key
                
                # Get attention mask if provided
                attn_mask = None
                if attention_mask is not None and key_mod in attention_mask:
                    attn_mask = attention_mask[key_mod]
                
                # Apply cross-attention
                output = attn_block(
                    query, key, value,
                    attention_mask=attn_mask,
                    return_attention=False
                )
                
                # Store updated features for query modality
                if self.config.residual_connection:
                    updated_features[query_mod] = updated_features[query_mod] + output
                else:
                    updated_features[query_mod] = output
            
            # Update features for next layer
            current_features = updated_features
        
        # Combine modalities
        combined_features = []
        for modality in self.modalities:
            if modality in current_features:
                feat = current_features[modality]
                
                # Pool if needed (e.g., mean pool over sequence dimension)
                if feat.dim() == 3:
                    # Mean pool over sequence
                    if attention_mask is not None and modality in attention_mask:
                        mask = attention_mask[modality].unsqueeze(-1).float()
                        feat_pooled = (feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    else:
                        feat_pooled = feat.mean(dim=1)
                else:
                    feat_pooled = feat
                
                combined_features.append(feat_pooled)
        
        # Concatenate and project
        if combined_features:
            fused = torch.cat(combined_features, dim=-1)
            fused = self.output_projection(fused)
        else:
            # Fallback: use first modality
            first_mod = list(current_features.keys())[0]
            fused = current_features[first_mod]
            if fused.dim() == 3:
                fused = fused.mean(dim=1)
        
        if return_dict:
            return FusionOutput(
                fused_features=fused,
                attention_weights=all_attention_weights,
                gate_values=None,
                intermediate_features=current_features,
                metadata={
                    'fusion_type': 'cross_attention',
                    'num_modalities': self.num_modalities,
                    'modalities': self.modalities,
                    'num_layers': len(self.cross_attention_layers)
                }
            )
        else:
            return fused
    
    def get_attention_weights(self,
                            modality_features: Dict[str, torch.Tensor],
                            layer_idx: int = -1) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from a specific layer.
        
        Args:
            modality_features: Input features
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Dictionary of attention weights
        """
        if layer_idx < 0:
            layer_idx = len(self.cross_attention_layers) + layer_idx
        
        if layer_idx >= len(self.cross_attention_layers):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        # Project inputs
        projected_features = {}
        for modality, feat in modality_features.items():
            if modality in self.input_projections:
                projected_features[modality] = self.input_projections[modality](feat)
            else:
                projected_features[modality] = feat
        
        # Get the specified layer
        layer = self.cross_attention_layers[layer_idx]
        attention_weights = {}
        
        # Forward through layer with return_attention=True
        for attn_name, attn_block in layer.items():
            if '_to_' in attn_name:
                query_mod, key_mod = attn_name.split('_to_')
            else:
                continue
            
            if query_mod not in projected_features or key_mod not in projected_features:
                continue
            
            query = projected_features[query_mod]
            key = projected_features[key_mod]
            value = projected_features[key_mod]
            
            # Get attention weights
            _, weights = attn_block(
                query, key, value,
                return_attention=True
            )
            
            attention_weights[attn_name] = weights
        
        return attention_weights

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]

# Example usage
if __name__ == "__main__":
    # Test cross-attention fusion
    config = CrossAttentionConfig(
        input_dims={
            'text': 768,
            'image': 2048,
            'audio': 1024
        },
        hidden_dim=512,
        num_heads=8,
        num_layers=2,
        bidirectional=True
    )
    
    fusion = CrossAttentionFusion(config)
    
    # Create dummy inputs
    batch_size = 2
    text_len = 16
    audio_len = 20
    
    dummy_features = {
        'text': torch.randn(batch_size, text_len, 768),
        'image': torch.randn(batch_size, 2048),
        'audio': torch.randn(batch_size, audio_len, 1024)
    }
    
    # Test forward pass
    output = fusion(dummy_features, return_dict=True)
    
    print(f"Input modalities: {list(dummy_features.keys())}")
    print(f"Output shape: {output.fused_features.shape}")
    print(f"Has attention weights: {output.attention_weights is not None}")
    print(f"Metadata: {output.metadata}")
    
    # Test getting attention weights
    attn_weights = fusion.get_attention_weights(dummy_features, layer_idx=0)
    print(f"\nAttention weights from layer 0:")
    for name, weights in attn_weights.items():
        print(f"  {name}: {weights.shape}")