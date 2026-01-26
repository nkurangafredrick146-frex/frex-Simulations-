"""
Contrastive loss functions for multimodal alignment.
Includes InfoNCE, triplet loss, circle loss, and other contrastive variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import math

class ContrastiveLoss(nn.Module):
    """
    Standard contrastive loss (also known as NT-Xent or SimCLR loss).
    Maximizes similarity between positive pairs, minimizes for negative pairs.
    """
    
    def __init__(self,
                 temperature: float = 0.07,
                 margin: float = 1.0,
                 similarity_type: str = 'cosine',
                 reduction: str = 'mean'):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature scaling parameter
            margin: Margin for contrastive loss
            similarity_type: Type of similarity ('cosine', 'dot', 'euclidean')
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.similarity_type = similarity_type
        self.reduction = reduction
        
        # Validate parameters
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if margin < 0:
            raise ValueError(f"Margin must be non-negative, got {margin}")
        if similarity_type not in ['cosine', 'dot', 'euclidean']:
            raise ValueError(
                f"Similarity type must be one of ['cosine', 'dot', 'euclidean'], "
                f"got {similarity_type}"
            )
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(
                f"Reduction must be one of ['mean', 'sum', 'none'], got {reduction}"
            )
    
    def forward(self,
                embeddings_a: torch.Tensor,
                embeddings_b: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings_a: First set of embeddings [batch_size, embedding_dim]
            embeddings_b: Second set of embeddings [batch_size, embedding_dim]
            labels: Optional labels for supervised contrastive [batch_size]
            mask: Optional mask for valid pairs [batch_size, batch_size]
            
        Returns:
            Contrastive loss
        """
        batch_size = embeddings_a.shape[0]
        
        # Normalize embeddings for cosine similarity
        if self.similarity_type == 'cosine':
            embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Compute similarity matrix
        if self.similarity_type == 'cosine' or self.similarity_type == 'dot':
            similarity = torch.mm(embeddings_a, embeddings_b.t()) / self.temperature
        elif self.similarity_type == 'euclidean':
            # Convert Euclidean distance to similarity
            expanded_a = embeddings_a.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            expanded_b = embeddings_b.unsqueeze(0)  # [1, batch_size, embedding_dim]
            distances = torch.sqrt(
                ((expanded_a - expanded_b) ** 2).sum(dim=2)
            )
            similarity = -distances / self.temperature
        
        # Create labels if not provided (assume corresponding pairs)
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings_a.device)
        
        # Create positive/negative masks
        if mask is None:
            # Default: diagonal is positive, others are negative
            positive_mask = torch.eye(batch_size, device=embeddings_a.device).bool()
            negative_mask = ~positive_mask
        else:
            positive_mask = mask.bool()
            negative_mask = ~positive_mask
        
        # Extract positive and negative similarities
        positive_sim = similarity[positive_mask].view(batch_size, -1)
        negative_sim = similarity[negative_mask].view(batch_size, -1)
        
        # Compute contrastive loss
        # For each positive pair, contrast with all negatives
        losses = []
        for i in range(batch_size):
            # Positive similarity for this example
            pos = positive_sim[i]
            
            # Negative similarities for this example
            neg = negative_sim[i]
            
            # Compute logits: concatenate positive and negatives
            logits = torch.cat([pos.unsqueeze(0), neg], dim=0)
            
            # Labels: first is positive (0), rest are negative
            target = torch.zeros(1, dtype=torch.long, device=embeddings_a.device)
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            losses.append(loss)
        
        losses = torch.stack(losses)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses
    
    def compute_similarity_matrix(self,
                                embeddings_a: torch.Tensor,
                                embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between embeddings.
        
        Args:
            embeddings_a: First set of embeddings
            embeddings_b: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        if self.similarity_type == 'cosine':
            embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
            similarity = torch.mm(embeddings_a, embeddings_b.t())
        elif self.similarity_type == 'dot':
            similarity = torch.mm(embeddings_a, embeddings_b.t())
        elif self.similarity_type == 'euclidean':
            n_a, n_b = embeddings_a.shape[0], embeddings_b.shape[0]
            expanded_a = embeddings_a.unsqueeze(1).expand(n_a, n_b, -1)
            expanded_b = embeddings_b.unsqueeze(0).expand(n_a, n_b, -1)
            distances = torch.sqrt(((expanded_a - expanded_b) ** 2).sum(dim=2))
            similarity = -distances
        
        return similarity

class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss.
    A specific form of contrastive loss used in self-supervised learning.
    """
    
    def __init__(self,
                 temperature: float = 0.07,
                 reduction: str = 'mean'):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter
            reduction: Loss reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
    
    def forward(self,
                query: torch.Tensor,
                positive_key: torch.Tensor,
                negative_keys: Optional[torch.Tensor] = None,
                batch_negatives: bool = True) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings [batch_size, embedding_dim]
            positive_key: Positive key embeddings [batch_size, embedding_dim]
            negative_keys: Negative key embeddings [num_negatives, embedding_dim]
            batch_negatives: Whether to use in-batch negatives
            
        Returns:
            InfoNCE loss
        """
        batch_size = query.shape[0]
        
        # Normalize embeddings
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)
        
        # Positive similarity
        positive_sim = torch.sum(query * positive_key, dim=1) / self.temperature
        positive_sim = positive_sim.unsqueeze(1)  # [batch_size, 1]
        
        # Negative similarities
        if batch_negatives and negative_keys is None:
            # Use all other positives as negatives (self-contrastive)
            negative_keys = positive_key
        
        if negative_keys is not None:
            negative_keys = F.normalize(negative_keys, p=2, dim=1)
            
            if batch_negatives:
                # All other examples in batch are negatives
                negative_sim = torch.mm(query, negative_keys.t()) / self.temperature
                
                # Mask out positives (diagonal)
                mask = torch.eye(batch_size, device=query.device).bool()
                negative_sim = negative_sim[~mask].view(batch_size, batch_size - 1)
            else:
                # Provided negative keys
                negative_sim = torch.mm(query, negative_keys.t()) / self.temperature
        else:
            # No explicit negatives (unlikely but handle)
            negative_sim = torch.zeros(batch_size, 1, device=query.device)
        
        # Concatenate positive and negatives
        logits = torch.cat([positive_sim, negative_sim], dim=1)
        
        # Labels: 0 for positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        
        return loss

class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    Pulls positive pairs together and pushes negative pairs apart.
    """
    
    def __init__(self,
                 margin: float = 1.0,
                 distance_type: str = 'euclidean',
                 mining_type: str = 'hard',
                 reduction: str = 'mean'):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_type: Type of distance ('euclidean', 'cosine')
            mining_type: Type of triplet mining ('hard', 'semi-hard', 'all')
            reduction: Loss reduction method
        """
        super().__init__()
        self.margin = margin
        self.distance_type = distance_type
        self.mining_type = mining_type
        self.reduction = reduction
        
        if margin < 0:
            raise ValueError(f"Margin must be non-negative, got {margin}")
        if distance_type not in ['euclidean', 'cosine']:
            raise ValueError(
                f"Distance type must be 'euclidean' or 'cosine', got {distance_type}"
            )
        if mining_type not in ['hard', 'semi-hard', 'all']:
            raise ValueError(
                f"Mining type must be one of ['hard', 'semi-hard', 'all'], "
                f"got {mining_type}"
            )
    
    def forward(self,
                anchors: torch.Tensor,
                positives: torch.Tensor,
                negatives: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchors: Anchor embeddings [batch_size, embedding_dim]
            positives: Positive embeddings [batch_size, embedding_dim]
            negatives: Negative embeddings [batch_size, embedding_dim]
            weights: Optional weights for each triplet
            
        Returns:
            Triplet loss
        """
        batch_size = anchors.shape[0]
        
        # Compute distances
        if self.distance_type == 'euclidean':
            pos_dist = F.pairwise_distance(anchors, positives, p=2)
            neg_dist = F.pairwise_distance(anchors, negatives, p=2)
        else:  # cosine
            # Cosine distance = 1 - cosine similarity
            pos_sim = F.cosine_similarity(anchors, positives)
            neg_sim = F.cosine_similarity(anchors, negatives)
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim
        
        # Compute triplet loss
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Apply triplet mining if needed
        if self.mining_type != 'all':
            losses = self._apply_mining(losses, pos_dist, neg_dist)
        
        # Apply weights if provided
        if weights is not None:
            losses = losses * weights
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses
    
    def _apply_mining(self,
                     losses: torch.Tensor,
                     pos_dist: torch.Tensor,
                     neg_dist: torch.Tensor) -> torch.Tensor:
        """
        Apply triplet mining.
        
        Args:
            losses: Raw triplet losses
            pos_dist: Positive distances
            neg_dist: Negative distances
            
        Returns:
            Mined losses
        """
        if self.mining_type == 'hard':
            # Only keep hard triplets (positive distance > negative distance)
            mask = pos_dist > neg_dist
            losses = losses[mask]
            
            if losses.numel() == 0:
                # No hard triplets, return zero loss
                return torch.tensor(0.0, device=losses.device)
                
        elif self.mining_type == 'semi-hard':
            # Keep semi-hard triplets (positive distance < negative distance < positive distance + margin)
            mask = (pos_dist < neg_dist) & (neg_dist < pos_dist + self.margin)
            losses = losses[mask]
            
            if losses.numel() == 0:
                # No semi-hard triplets, fall back to random
                return losses  # Will be empty, handled by reduction
        
        return losses
    
    def generate_triplets(self,
                         embeddings: torch.Tensor,
                         labels: torch.Tensor,
                         num_triplets: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate triplets from embeddings and labels.
        
        Args:
            embeddings: All embeddings [n_samples, embedding_dim]
            labels: Corresponding labels [n_samples]
            num_triplets: Number of triplets to generate (None for all possible)
            
        Returns:
            Tuple of (anchors, positives, negatives)
        """
        n_samples = embeddings.shape[0]
        device = embeddings.device
        
        # Get unique labels
        unique_labels = torch.unique(labels)
        
        anchors = []
        positives = []
        negatives = []
        
        for label in unique_labels:
            # Indices for this class
            pos_indices = torch.where(labels == label)[0]
            neg_indices = torch.where(labels != label)[0]
            
            if len(pos_indices) < 2 or len(neg_indices) == 0:
                continue
            
            # Create pairs within class
            for i in pos_indices:
                for j in pos_indices:
                    if i != j:
                        # For each positive pair, sample a negative
                        for k in neg_indices:
                            anchors.append(embeddings[i])
                            positives.append(embeddings[j])
                            negatives.append(embeddings[k])
        
        # Convert to tensors
        if anchors:
            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            negatives = torch.stack(negatives)
            
            # Sample if too many
            if num_triplets is not None and len(anchors) > num_triplets:
                indices = torch.randperm(len(anchors))[:num_triplets]
                anchors = anchors[indices]
                positives = positives[indices]
                negatives = negatives[indices]
        else:
            # No valid triplets
            anchors = torch.empty((0, embeddings.shape[1]), device=device)
            positives = torch.empty((0, embeddings.shape[1]), device=device)
            negatives = torch.empty((0, embeddings.shape[1]), device=device)
        
        return anchors, positives, negatives

class CircleLoss(nn.Module):
    """
    Circle loss from "Circle Loss: A Unified Perspective of Pair Similarity Optimization".
    Provides more flexible optimization with separate margins for positive and negative pairs.
    """
    
    def __init__(self,
                 margin: float = 0.25,
                 gamma: float = 256.0,
                 reduction: str = 'mean'):
        """
        Initialize circle loss.
        
        Args:
            margin: Margin parameter
            gamma: Scaling factor
            reduction: Loss reduction method
        """
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        
        # Derived parameters
        self.positive_margin = margin
        self.negative_margin = -margin
        self.positive_optimal = 1 - margin
        self.negative_optimal = margin
        
        if margin <= 0:
            raise ValueError(f"Margin must be positive, got {margin}")
        if gamma <= 0:
            raise ValueError(f"Gamma must be positive, got {gamma}")
    
    def forward(self,
                embeddings_a: torch.Tensor,
                embeddings_b: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute circle loss.
        
        Args:
            embeddings_a: First set of embeddings
            embeddings_b: Second set of embeddings
            labels: Optional labels for supervised setting
            
        Returns:
            Circle loss
        """
        batch_size = embeddings_a.shape[0]
        
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(embeddings_a, embeddings_b.t())  # [batch_size, batch_size]
        
        # Create masks
        if labels is None:
            # Self-supervised: diagonal is positive
            positive_mask = torch.eye(batch_size, device=embeddings_a.device).bool()
            negative_mask = ~positive_mask
        else:
            # Supervised: same label is positive
            label_expanded = labels.unsqueeze(1) == labels.unsqueeze(0)
            positive_mask = label_expanded.bool()
            negative_mask = ~positive_mask
        
        # Extract positive and negative similarities
        positive_sim = similarity[positive_mask]
        negative_sim = similarity[negative_mask]
        
        if positive_sim.numel() == 0 or negative_sim.numel() == 0:
            return torch.tensor(0.0, device=embeddings_a.device)
        
        # Reshape to [batch_size, num_positives] and [batch_size, num_negatives]
        positive_sim = positive_sim.view(batch_size, -1)
        negative_sim = negative_sim.view(batch_size, -1)
        
        # Compute circle loss
        losses = []
        for i in range(batch_size):
            pos = positive_sim[i]
            neg = negative_sim[i]
            
            # Skip if no positives or negatives
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            
            # Compute losses for positive pairs
            alpha_pos = torch.clamp(self.positive_optimal - pos, min=0)
            logit_pos = -self.gamma * alpha_pos * (pos - self.positive_margin)
            
            # Compute losses for negative pairs
            alpha_neg = torch.clamp(neg - self.negative_optimal, min=0)
            logit_neg = self.gamma * alpha_neg * (neg - self.negative_margin)
            
            # Combine losses
            loss_pos = torch.logsumexp(logit_pos, dim=0)
            loss_neg = torch.logsumexp(logit_neg, dim=0)
            
            loss = F.softplus(loss_pos + loss_neg)
            losses.append(loss)
        
        if not losses:
            return torch.tensor(0.0, device=embeddings_a.device)
        
        losses = torch.stack(losses)
        
        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss from "Supervised Contrastive Learning".
    Extends contrastive loss to leverage label information.
    """
    
    def __init__(self,
                 temperature: float = 0.07,
                 contrast_mode: str = 'all',
                 reduction: str = 'mean'):
        """
        Initialize supervised contrastive loss.
        
        Args:
            temperature: Temperature parameter
            contrast_mode: Contrast mode ('all', 'one')
            reduction: Loss reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.reduction = reduction
        
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if contrast_mode not in ['all', 'one']:
            raise ValueError(
                f"Contrast mode must be 'all' or 'one', got {contrast_mode}"
            )
    
    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings [batch_size, embedding_dim]
            labels: Corresponding labels [batch_size]
            mask: Optional boolean mask [batch_size, batch_size]
            
        Returns:
            Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.mm(features, features.t()) / self.temperature
        
        # Create mask for positive pairs (same label)
        label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Mask out self-contrast (diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        label_matrix = label_matrix & ~self_mask
        
        # Combine with optional mask
        if mask is not None:
            label_matrix = label_matrix & mask
        
        # Compute logits
        exp_sim = torch.exp(similarity)
        
        # Sum of similarities for positive pairs
        pos_sum = torch.sum(exp_sim * label_matrix, dim=1)
        
        # Sum of similarities for all pairs (excluding self)
        all_sum = torch.sum(exp_sim, dim=1) - torch.exp(torch.diag(similarity))
        
        # Compute loss
        losses = -torch.log(pos_sum / all_sum)
        
        # Handle cases with no positive pairs
        losses = torch.where(
            label_matrix.any(dim=1),
            losses,
            torch.tensor(0.0, device=device)
        )
        
        # Apply reduction
        if self.reduction == 'mean':
            # Average over samples with positive pairs
            valid_losses = losses[losses > 0]
            if valid_losses.numel() > 0:
                return valid_losses.mean()
            else:
                return torch.tensor(0.0, device=device)
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            return losses

class AlignUniformLoss(nn.Module):
    """
    Align-Uniform loss from "Understanding Contrastive Representation Learning
    through Alignment and Uniformity on the Hypersphere".
    
    Alignment: Positive pairs should have similar features
    Uniformity: Features should be uniformly distributed on hypersphere
    """
    
    def __init__(self,
                 align_weight: float = 1.0,
                 uniform_weight: float = 1.0,
                 align_alpha: float = 2.0,
                 uniform_t: float = 2.0,
                 reduction: str = 'mean'):
        """
        Initialize align-uniform loss.
        
        Args:
            align_weight: Weight for alignment loss
            uniform_weight: Weight for uniformity loss
            align_alpha: Exponent for alignment loss
            uniform_t: Temperature for uniformity loss
            reduction: Loss reduction method
        """
        super().__init__()
        self.align_weight = align_weight
        self.uniform_weight = uniform_weight
        self.align_alpha = align_alpha
        self.uniform_t = uniform_t
        self.reduction = reduction
        
        if align_weight < 0 or uniform_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if align_alpha <= 0:
            raise ValueError(f"align_alpha must be positive, got {align_alpha}")
        if uniform_t <= 0:
            raise ValueError(f"uniform_t must be positive, got {uniform_t}")
    
    def forward(self,
                embeddings_a: torch.Tensor,
                embeddings_b: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute align-uniform loss.
        
        Args:
            embeddings_a: First set of embeddings
            embeddings_b: Second set of embeddings
            mask: Optional mask for valid pairs
            
        Returns:
            Align-uniform loss
        """
        batch_size = embeddings_a.shape[0]
        
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Compute alignment loss
        if mask is None:
            # Use corresponding pairs
            align_loss = torch.norm(embeddings_a - embeddings_b, p=2, dim=1) ** self.align_alpha
            align_loss = align_loss.mean()
        else:
            # Use masked pairs
            masked_a = embeddings_a[mask.any(dim=1)]
            masked_b = embeddings_b[mask.any(dim=0)]
            if masked_a.numel() > 0 and masked_b.numel() > 0:
                align_loss = torch.norm(masked_a - masked_b, p=2, dim=1) ** self.align_alpha
                align_loss = align_loss.mean()
            else:
                align_loss = torch.tensor(0.0, device=embeddings_a.device)
        
        # Compute uniformity loss
        # Concatenate all embeddings
        all_embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)
        n_all = all_embeddings.shape[0]
        
        # Compute pairwise distances
        pairwise_dist = torch.pdist(all_embeddings, p=2)  # [n_all * (n_all - 1) / 2]
        
        # Compute uniformity loss (Gaussian potential)
        uniform_loss = torch.exp(-self.uniform_t * pairwise_dist ** 2).mean()
        
        # Combine losses
        total_loss = (
            self.align_weight * align_loss +
            self.uniform_weight * uniform_loss
        )
        
        return total_loss
    
    def compute_alignment(self,
                         embeddings_a: torch.Tensor,
                         embeddings_b: torch.Tensor) -> torch.Tensor:
        """
        Compute only the alignment component.
        
        Args:
            embeddings_a: First set of embeddings
            embeddings_b: Second set of embeddings
            
        Returns:
            Alignment loss
        """
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        align_loss = torch.norm(embeddings_a - embeddings_b, p=2, dim=1) ** self.align_alpha
        return align_loss.mean()
    
    def compute_uniformity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute only the uniformity component.
        
        Args:
            embeddings: Embeddings to check uniformity
            
        Returns:
            Uniformity loss
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        pairwise_dist = torch.pdist(embeddings, p=2)
        uniform_loss = torch.exp(-self.uniform_t * pairwise_dist ** 2).mean()
        
        return uniform_loss

# Example usage
if __name__ == "__main__":
    # Test contrastive losses
    batch_size = 32
    embedding_dim = 128
    
    # Create random embeddings
    embeddings_a = torch.randn(batch_size, embedding_dim)
    embeddings_b = torch.randn(batch_size, embedding_dim)
    
    # Test each loss
    contrastive = ContrastiveLoss(temperature=0.07)
    infonce = InfoNCELoss(temperature=0.07)
    triplet = TripletLoss(margin=1.0)
    circle = CircleLoss(margin=0.25, gamma=256.0)
    supcon = SupConLoss(temperature=0.07)
    align_uniform = AlignUniformLoss(align_weight=1.0, uniform_weight=1.0)
    
    # Compute losses
    loss_contrastive = contrastive(embeddings_a, embeddings_b)
    loss_infonce = infonce(embeddings_a, embeddings_b)
    
    # For triplet loss, need triplets
    anchors = embeddings_a
    positives = embeddings_b
    negatives = torch.randn(batch_size, embedding_dim)  # Random negatives
    loss_triplet = triplet(anchors, positives, negatives)
    
    loss_circle = circle(embeddings_a, embeddings_b)
    loss_align_uniform = align_uniform(embeddings_a, embeddings_b)
    
    print(f"Contrastive Loss: {loss_contrastive.item():.4f}")
    print(f"InfoNCE Loss: {loss_infonce.item():.4f}")
    print(f"Triplet Loss: {loss_triplet.item():.4f}")
    print(f"Circle Loss: {loss_circle.item():.4f}")
    print(f"Align-Uniform Loss: {loss_align_uniform.item():.4f}")