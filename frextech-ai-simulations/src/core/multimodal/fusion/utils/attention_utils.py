"""
Attention utilities for multimodal fusion.
Includes various attention mechanisms and helper functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
import math
import numpy as np

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
            mask: Attention mask (0 for masked positions)
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
    
    @staticmethod
    def create_2d_mask(height: int, width: int, device: torch.device = None) -> torch.Tensor:
        """Create 2D attention mask for images."""
        mask = torch.ones(height * width, height * width, device=device)
        return mask
    
    @staticmethod
    def compute_attention_entropy(attention_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute entropy of attention distribution."""
        return -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
    
    @staticmethod
    def compute_attention_sparsity(attention_weights: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Compute sparsity of attention distribution."""
        return (attention_weights < threshold).float().mean(dim=-1)
    
    @staticmethod
    def apply_local_attention(attention_weights: torch.Tensor,
                            window_size: int,
                            causal: bool = False) -> torch.Tensor:
        """Apply local attention within a window."""
        seq_len = attention_weights.size(-1)
        mask = torch.ones(seq_len, seq_len, device=attention_weights.device)
        
        for i in range(seq_len):
            if causal:
                # Causal local attention
                start = max(0, i - window_size + 1)
                mask[i, :start] = 0
                mask[i, i+1:] = 0
            else:
                # Bidirectional local attention
                start = max(0, i - window_size // 2)
                end = min(seq_len, i + window_size // 2 + 1)
                mask[i, :start] = 0
                mask[i, end:] = 0
        
        attention_weights = attention_weights * mask
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        return attention_weights
    
    @staticmethod
    def compute_attention_distance(attention_weights: torch.Tensor) -> torch.Tensor:
        """Compute average attention distance."""
        seq_len = attention_weights.size(-1)
        positions = torch.arange(seq_len, device=attention_weights.device).float()
        
        # Expected position for each query
        expected_positions = torch.matmul(attention_weights, positions)
        
        # Compute distance from query position
        query_positions = positions.unsqueeze(0).expand_as(attention_weights)
        distances = torch.abs(expected_positions - query_positions)
        
        return distances
    
    @staticmethod
    def visualize_attention(attention_weights: torch.Tensor,
                          cmap: str = 'viridis',
                          figsize: Tuple[int, int] = (10, 8)) -> None:
        """Visualize attention weights (for debugging)."""
        import matplotlib.pyplot as plt
        
        if attention_weights.dim() == 4:
            # Multi-head attention: average over heads
            attention_weights = attention_weights.mean(dim=1)
        
        if attention_weights.dim() == 3:
            # Batch dimension: take first element
            attention_weights = attention_weights[0]
        
        plt.figure(figsize=figsize)
        plt.imshow(attention_weights.cpu().numpy(), cmap=cmap, aspect='auto')
        plt.colorbar()
        plt.title('Attention Weights')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        plt.show()
    
    @staticmethod
    def efficient_attention(query: torch.Tensor,
                          key: torch.Tensor,
                          value: torch.Tensor,
                          chunk_size: int = 64) -> torch.Tensor:
        """Memory-efficient attention for long sequences."""
        batch_size, seq_len, dim = query.shape
        
        # Process in chunks
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, chunk_size):
            q_chunk = query[:, i:i+chunk_size, :]
            
            # Compute attention for this chunk
            scores = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(dim)
            attn_weights = F.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn_weights, value)
            
            output[:, i:i+chunk_size, :] = chunk_output
        
        return output
    
    @staticmethod
    def linear_attention(query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        feature_map: str = 'elu') -> torch.Tensor:
        """Linear attention approximation for efficient computation."""
        # Apply feature map
        if feature_map == 'elu':
            phi = lambda x: F.elu(x) + 1
        elif feature_map == 'relu':
            phi = lambda x: F.relu(x)
        else:
            phi = lambda x: x
        
        phi_q = phi(query)
        phi_k = phi(key)
        
        # Compute using associativity property
        kv = torch.matmul(phi_k.transpose(-2, -1), value)
        z = 1.0 / (torch.matmul(phi_q, phi_k.sum(dim=1, keepdim=True).transpose(-2, -1)) + 1e-8)
        
        output = torch.matmul(phi_q, kv) * z
        
        return output
    
    @staticmethod
    def relative_positional_attention(query: torch.Tensor,
                                     key: torch.Tensor,
                                     value: torch.Tensor,
                                     relative_bias: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention with relative positional bias."""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Add relative positional bias
        seq_len = query.size(1)
        relative_indices = torch.arange(seq_len, device=query.device)
        relative_indices = relative_indices.unsqueeze(1) - relative_indices.unsqueeze(0)
        relative_indices = relative_indices + seq_len - 1  # Shift to positive indices
        
        # Gather relative biases
        bias = relative_bias[relative_indices]
        scores = scores + bias.unsqueeze(0)  # Add batch dimension
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        return output
    
    @staticmethod
    def sparse_attention(query: torch.Tensor,
                        key: torch.Tensor,
                        value: torch.Tensor,
                        sparsity_pattern: str = 'strided',
                        num_blocks: int = 4) -> torch.Tensor:
        """Sparse attention with different patterns."""
        batch_size, seq_len, dim = query.shape
        
        if sparsity_pattern == 'strided':
            # Strided attention
            stride = seq_len // num_blocks
            mask = torch.zeros(seq_len, seq_len, device=query.device)
            
            for i in range(seq_len):
                block_idx = i // stride
                mask[i, block_idx*stride:(block_idx+1)*stride] = 1
                # Also attend to previous block
                if block_idx > 0:
                    mask[i, (block_idx-1)*stride:block_idx*stride] = 1
        
        elif sparsity_pattern == 'fixed':
            # Fixed block attention
            block_size = seq_len // num_blocks
            mask = torch.zeros(seq_len, seq_len, device=query.device)
            
            for i in range(0, seq_len, block_size):
                mask[i:i+block_size, i:i+block_size] = 1
        
        elif sparsity_pattern == 'random':
            # Random attention
            mask = torch.rand(seq_len, seq_len, device=query.device) > 0.5
            mask = mask.float()
        
        else:
            raise ValueError(f"Unknown sparsity pattern: {sparsity_pattern}")
        
        # Apply mask
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)
        
        return output

# Example usage
if __name__ == "__main__":
    # Test attention utilities
    batch_size = 2
    seq_len = 16
    hidden_dim = 64
    
    query = torch.randn(batch_size, seq_len, hidden_dim)
    key = torch.randn(batch_size, seq_len, hidden_dim)
    value = torch.randn(batch_size, seq_len, hidden_dim)
    
    print("Testing AttentionUtils:")
    
    # Test scaled dot-product attention
    output, weights = AttentionUtils.scaled_dot_product_attention(query, key, value)
    print(f"Scaled dot-product attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Test multi-head attention
    num_heads = 4
    output_mh, weights_mh = AttentionUtils.multi_head_attention_forward(
        query, key, value, num_heads
    )
    print(f"\nMulti-head attention output shape: {output_mh.shape}")
    
    # Test causal mask
    causal_mask = AttentionUtils.create_causal_mask(seq_len)
    print(f"\nCausal mask shape: {causal_mask.shape}")
    
    # Test attention entropy
    entropy = AttentionUtils.compute_attention_entropy(weights)
    print(f"\nAttention entropy shape: {entropy.shape}")
    
    # Test efficient attention
    output_eff = AttentionUtils.efficient_attention(query, key, value, chunk_size=8)
    print(f"\nEfficient attention output shape: {output_eff.shape}")
    
    # Test linear attention
    output_lin = AttentionUtils.linear_attention(query, key, value)
    print(f"\nLinear attention output shape: {output_lin.shape}")