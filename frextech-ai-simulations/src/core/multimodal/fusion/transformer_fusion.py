"""
Transformer-based fusion modules for multimodal integration.
Uses transformer encoders to fuse information across modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
import logging
import math
from einops import rearrange, repeat

from ..alignment import FusionConfig, FusionOutput

logger = logging.getLogger(__name__)

@dataclass
class TransformerFusionConfig:
    """Configuration for transformer fusion."""
    
    # Input parameters
    input_dims: Dict[str, int] = field(default_factory=dict)
    hidden_dim: int = 512
    
    # Transformer parameters
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation: str = "gelu"
    pre_norm: bool = True
    
    # Positional encoding
    use_positional_encoding: bool = True
    max_seq_length: int = 512
    positional_encoding_type: str = "learned"  # "learned", "sinusoidal", "relative"
    
    # Modality handling
    modality_embedding: bool = True
    modality_embedding_dim: int = 64
    
    # Output
    output_projection: bool = True
    pool_method: str = "cls"  # "cls", "mean", "max"
    residual_connection: bool = True

class TransformerFusion(nn.Module):
    """
    Transformer-based fusion module.
    Treats each modality as a sequence and uses transformer encoder for fusion.
    """
    
    def __init__(self, config: TransformerFusionConfig):
        super().__init__()
        self.config = config
        self.modality_dims = config.input_dims
        self.modalities = list(config.input_dims.keys())
        self.num_modalities = len(self.modalities)
        
        # Input projections
        self.input_projections = nn.ModuleDict()
        for modality, dim in self.modality_dims.items():
            self.input_projections[modality] = nn.Linear(dim, config.hidden_dim)
        
        # Modality embeddings
        if config.modality_embedding:
            self.modality_embeddings = nn.Embedding(self.num_modalities, config.modality_embedding_dim)
            # Project modality embeddings to hidden dim
            self.modality_projection = nn.Linear(config.modality_embedding_dim, config.hidden_dim)
        else:
            self.modality_embeddings = None
        
        # Positional encoding
        if config.use_positional_encoding:
            if config.positional_encoding_type == "learned":
                self.positional_encoding = nn.Embedding(config.max_seq_length, config.hidden_dim)
            elif config.positional_encoding_type == "sinusoidal":
                self.positional_encoding = SinusoidalPositionalEncoding(config.hidden_dim, config.max_seq_length)
            elif config.positional_encoding_type == "relative":
                self.positional_encoding = RelativePositionalEncoding(config.hidden_dim, config.max_seq_length)
            else:
                raise ValueError(f"Unknown positional encoding type: {config.positional_encoding_type}")
        else:
            self.positional_encoding = None
        
        # Transformer encoder
        self.transformer = MultiModalTransformer(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation=config.activation,
            pre_norm=config.pre_norm
        )
        
        # CLS token (optional, for pooling)
        if config.pool_method == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        else:
            self.cls_token = None
        
        # Output projection
        if config.output_projection:
            self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim)
        else:
            self.output_projection = nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized TransformerFusion with {self.num_modalities} modalities")
        logger.info(f"Hidden dim: {config.hidden_dim}")
        logger.info(f"Num layers: {config.num_layers}")
        logger.info(f"Pool method: {config.pool_method}")
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize input projections
        for proj in self.input_projections.values():
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        
        # Initialize CLS token
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        
        # Initialize output projection
        if hasattr(self.output_projection, 'weight'):
            nn.init.xavier_uniform_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)
    
    def _prepare_inputs(self,
                       modality_features: Dict[str, torch.Tensor],
                       attention_mask: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for transformer.
        
        Args:
            modality_features: Dictionary of modality features
            attention_mask: Optional attention masks
            
        Returns:
            Tuple of (tokens, attention_mask)
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        
        # Project all features to hidden dimension
        projected_features = []
        sequence_lengths = []
        
        for i, modality in enumerate(self.modalities):
            if modality in modality_features:
                feat = modality_features[modality]
                
                # Project to hidden dim
                if modality in self.input_projections:
                    feat = self.input_projections[modality](feat)
                
                # Add modality embedding if enabled
                if self.modality_embeddings is not None:
                    modality_emb = self.modality_embeddings(
                        torch.tensor(i, device=feat.device)
                    )
                    modality_emb = self.modality_projection(modality_emb)
                    
                    # Expand to match batch size and sequence length
                    if feat.dim() == 3:
                        # [batch_size, seq_len, hidden_dim]
                        seq_len = feat.shape[1]
                        modality_emb = modality_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                        modality_emb = modality_emb.expand(batch_size, seq_len, -1)
                    else:
                        # [batch_size, hidden_dim]
                        modality_emb = modality_emb.unsqueeze(0).expand(batch_size, -1)
                    
                    feat = feat + modality_emb
                
                # Reshape if needed (for 2D features, add sequence dimension)
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                
                projected_features.append(feat)
                sequence_lengths.append(feat.shape[1])
        
        # Concatenate along sequence dimension
        tokens = torch.cat(projected_features, dim=1)  # [batch_size, total_seq_len, hidden_dim]
        total_seq_len = tokens.shape[1]
        
        # Add CLS token if using CLS pooling
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            total_seq_len += 1
        
        # Add positional encoding if enabled
        if self.positional_encoding is not None:
            if isinstance(self.positional_encoding, nn.Embedding):
                positions = torch.arange(total_seq_len, device=tokens.device).unsqueeze(0)
                pos_emb = self.positional_encoding(positions)
            else:
                pos_emb = self.positional_encoding(tokens)
            
            tokens = tokens + pos_emb
        
        # Create attention mask if not provided
        if attention_mask is None:
            # All tokens are valid
            attention_mask = torch.ones(batch_size, total_seq_len, device=tokens.device)
        else:
            # Need to combine modality-specific masks
            # For simplicity, assume all tokens are valid
            attention_mask = torch.ones(batch_size, total_seq_len, device=tokens.device)
        
        return tokens, attention_mask
    
    def _pool_features(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence features.
        
        Args:
            features: Sequence features [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled features [batch_size, hidden_dim]
        """
        if self.config.pool_method == "cls":
            # Use CLS token (first token)
            pooled = features[:, 0, :]
        
        elif self.config.pool_method == "mean":
            # Mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            sum_features = torch.sum(features * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1)
            pooled = sum_features / torch.clamp(sum_mask, min=1e-9)
        
        elif self.config.pool_method == "max":
            # Max pooling
            mask = attention_mask.unsqueeze(-1)
            features = features.masked_fill(~mask.bool(), float('-inf'))
            pooled = torch.max(features, dim=1)[0]
            pooled[torch.isinf(pooled)] = 0
        
        else:
            raise ValueError(f"Unknown pool method: {self.config.pool_method}")
        
        return pooled
    
    def forward(self,
                modality_features: Dict[str, torch.Tensor],
                attention_mask: Optional[Dict[str, torch.Tensor]] = None,
                return_dict: bool = True,
                **kwargs) -> Union[torch.Tensor, FusionOutput]:
        """
        Forward pass for transformer fusion.
        
        Args:
            modality_features: Dictionary of modality features
            attention_mask: Optional attention masks
            return_dict: Whether to return FusionOutput
            **kwargs: Additional arguments
            
        Returns:
            Fused features or FusionOutput
        """
        # Prepare inputs
        tokens, attn_mask = self._prepare_inputs(modality_features, attention_mask)
        
        # Apply transformer
        transformer_output = self.transformer(
            tokens,
            attention_mask=attn_mask,
            return_attention=True
        )
        
        # Get features and attention weights
        features = transformer_output['features']
        attention_weights = transformer_output['attention_weights']
        
        # Pool features
        pooled = self._pool_features(features, attn_mask)
        
        # Apply output projection
        fused = self.output_projection(pooled)
        
        # Add residual connection if enabled
        if self.config.residual_connection and self.modalities:
            # Use first modality as residual
            first_modality = self.modalities[0]
            if first_modality in modality_features:
                residual = modality_features[first_modality]
                if residual.dim() == 3:
                    residual = residual.mean(dim=1)
                
                # Project if dimensions don't match
                if residual.shape[-1] != fused.shape[-1]:
                    residual = nn.Linear(residual.shape[-1], fused.shape[-1])(residual)
                
                fused = fused + residual
        
        if return_dict:
            return FusionOutput(
                fused_features=fused,
                attention_weights=attention_weights,
                gate_values=None,
                intermediate_features={'tokens': tokens, 'transformer_features': features},
                metadata={
                    'fusion_type': 'transformer',
                    'num_modalities': self.num_modalities,
                    'modalities': self.modalities,
                    'pool_method': self.config.pool_method,
                    'num_layers': self.config.num_layers
                }
            )
        else:
            return fused
    
    def get_attention_maps(self,
                          modality_features: Dict[str, torch.Tensor],
                          layer_idx: int = -1,
                          head_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get attention maps from transformer.
        
        Args:
            modality_features: Input features
            layer_idx: Layer index (-1 for last layer)
            head_idx: Specific head index (None for all)
            
        Returns:
            Attention maps
        """
        # Prepare inputs
        tokens, attn_mask = self._prepare_inputs(modality_features)
        
        # Forward through transformer with attention output
        output = self.transformer(
            tokens,
            attention_mask=attn_mask,
            return_attention=True,
            return_all_layers=True
        )
        
        # Get attention from specified layer
        if layer_idx < 0:
            layer_idx = len(output['all_attention_weights']) + layer_idx
        
        if layer_idx >= len(output['all_attention_weights']):
            raise ValueError(f"Layer index {layer_idx} out of range")
        
        attention = output['all_attention_weights'][layer_idx]
        
        # Select specific head if requested
        if head_idx is not None:
            if head_idx >= attention.shape[1]:
                raise ValueError(f"Head index {head_idx} out of range")
            attention = attention[:, head_idx, :, :]
        
        return attention

class MultiModalTransformer(nn.Module):
    """
    Transformer encoder for multimodal fusion.
    """
    
    def __init__(self,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 activation: str = "gelu",
                 pre_norm: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pre_norm = pre_norm
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim) if pre_norm else nn.Identity()
    
    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False,
                return_all_layers: bool = False) -> Dict[str, Any]:
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            return_all_layers: Whether to return features from all layers
            
        Returns:
            Dictionary with features and optionally attention weights
        """
        all_features = [] if return_all_layers else None
        all_attention_weights = [] if return_attention else None
        
        # Apply transformer layers
        for layer in self.layers:
            if return_attention:
                x, attn_weights = layer(x, attention_mask, return_attention=True)
                if all_attention_weights is not None:
                    all_attention_weights.append(attn_weights)
            else:
                x = layer(x, attention_mask)
            
            if return_all_layers:
                all_features.append(x)
        
        # Apply final layer norm
        if self.pre_norm:
            x = self.norm(x)
        
        # Prepare output
        output = {'features': x}
        
        if return_attention:
            output['attention_weights'] = all_attention_weights[-1] if all_attention_weights else None
            if return_all_layers:
                output['all_attention_weights'] = all_attention_weights
        
        if return_all_layers:
            output['all_features'] = all_features
        
        return output

class TransformerLayer(nn.Module):
    """Single transformer layer."""
    
    def __init__(self,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 activation: str = "gelu",
                 pre_norm: bool = True):
        super().__init__()
        
        self.pre_norm = pre_norm
        self.hidden_dim = hidden_dim
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attention_dropout
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            activation=activation
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for transformer layer.
        
        Args:
            x: Input tensor
            attention_mask: Attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (and optionally attention weights)
        """
        residual = x
        
        # Self-attention with pre-norm or post-norm
        if self.pre_norm:
            x_norm = self.norm1(x)
            if return_attention:
                attn_output, attn_weights = self.self_attn(x_norm, x_norm, x_norm, 
                                                          attention_mask, return_attention=True)
            else:
                attn_output = self.self_attn(x_norm, x_norm, x_norm, attention_mask)
            x = residual + self.dropout(attn_output)
        else:
            if return_attention:
                attn_output, attn_weights = self.self_attn(x, x, x, 
                                                          attention_mask, return_attention=True)
            else:
                attn_output = self.self_attn(x, x, x, attention_mask)
            x = self.norm1(residual + self.dropout(attn_output))
        
        # Feed-forward network
        residual = x
        
        if self.pre_norm:
            x_norm = self.norm2(x)
            ffn_output = self.ffn(x_norm)
            x = residual + self.dropout(ffn_output)
        else:
            ffn_output = self.ffn(x)
            x = self.norm2(residual + self.dropout(ffn_output))
        
        if return_attention:
            return x, attn_weights
        else:
            return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""
    
    def __init__(self,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
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
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attention_mask: Attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Attention output (and optionally attention weights)
        """
        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        
        # Project inputs
        q = self.q_proj(query)  # [batch_size, query_len, hidden_dim]
        k = self.k_proj(key)    # [batch_size, key_len, hidden_dim]
        v = self.v_proj(value)  # [batch_size, value_len, hidden_dim]
        
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

class FeedForwardNetwork(nn.Module):
    """Feed-forward network for transformer."""
    
    def __init__(self,
                 hidden_dim: int = 512,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.mlp_dim = hidden_dim * mlp_ratio
        
        self.fc1 = nn.Linear(hidden_dim, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, hidden_dim)
        
        self.activation = self._get_activation(activation)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feed-forward network."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for transformer."""
    
    def __init__(self, hidden_dim: int, max_len: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        # Learnable relative position embeddings
        self.embeddings = nn.Embedding(2 * max_len - 1, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add relative positional encoding."""
        batch_size, seq_len, _ = x.shape
        
        # Generate relative position indices
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]
        relative_positions = relative_positions + self.max_len - 1  # Shift to positive indices
        
        # Get embeddings
        pos_emb = self.embeddings(relative_positions)  # [seq_len, seq_len, hidden_dim]
        
        # Add to input (simplified - in practice might integrate differently)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return pos_emb.mean(dim=2)  # Simplified: average over relative positions

# Reuse SinusoidalPositionalEncoding from cross_attention.py
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
    # Test transformer fusion
    config = TransformerFusionConfig(
        input_dims={
            'text': 768,
            'image': 2048,
            'audio': 1024
        },
        hidden_dim=512,
        num_layers=2,
        num_heads=8,
        pool_method='cls'
    )
    
    fusion = TransformerFusion(config)
    
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
    
    # Test attention maps
    attention_maps = fusion.get_attention_maps(dummy_features, layer_idx=-1)
    print(f"\nAttention maps shape: {attention_maps.shape}")