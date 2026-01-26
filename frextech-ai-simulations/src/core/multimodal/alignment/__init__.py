"""
Multimodal alignment module for aligning different modalities in a shared embedding space.
Includes contrastive learning, alignment training, and cross-modal retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

from .contrastive_loss import (
    ContrastiveLoss,
    InfoNCELoss,
    TripletLoss,
    CircleLoss,
    SupConLoss,
    AlignUniformLoss
)

from .alignment_trainer import (
    AlignmentTrainer,
    AlignmentConfig,
    AlignmentMetrics,
    AlignmentDataset,
    AlignmentModel
)

from .cross_modal_retrieval import (
    CrossModalRetriever,
    RetrievalMetrics,
    RetrievalDataset,
    EmbeddingIndex
)

from .consistency_checker import (
    ConsistencyChecker,
    ConsistencyMetrics,
    ConsistencyConfig
)

from .utils import (
    alignment_utils,
    embedding_utils,
    similarity_utils,
    evaluation_utils
)

__all__ = [
    # Losses
    'ContrastiveLoss',
    'InfoNCELoss',
    'TripletLoss',
    'CircleLoss',
    'SupConLoss',
    'AlignUniformLoss',
    
    # Alignment training
    'AlignmentTrainer',
    'AlignmentConfig',
    'AlignmentMetrics',
    'AlignmentDataset',
    'AlignmentModel',
    
    # Cross-modal retrieval
    'CrossModalRetriever',
    'RetrievalMetrics',
    'RetrievalDataset',
    'EmbeddingIndex',
    
    # Consistency checking
    'ConsistencyChecker',
    'ConsistencyMetrics',
    'ConsistencyConfig',
    
    # Utilities
    'alignment_utils',
    'embedding_utils',
    'similarity_utils',
    'evaluation_utils'
]

# Version
__version__ = '1.0.0'

# Initialize logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Alignment types
class AlignmentType(Enum):
    """Types of multimodal alignment."""
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    CROSS_ATTENTION = "cross_attention"
    CYCLE_CONSISTENCY = "cycle_consistency"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"

@dataclass
class AlignmentResult:
    """Result of alignment operation."""
    success: bool
    alignment_score: float
    confidence: float
    aligned_embeddings: Optional[torch.Tensor] = None
    similarity_matrix: Optional[torch.Tensor] = None
    alignment_loss: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None

@dataclass
class AlignmentPair:
    """Pair of modalities for alignment."""
    modality_a: str
    modality_b: str
    embeddings_a: torch.Tensor
    embeddings_b: torch.Tensor
    similarity_matrix: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None

class MultiModalAlignment:
    """
    Main class for multimodal alignment operations.
    Supports multiple alignment strategies and loss functions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.loss_functions = {
            'contrastive': ContrastiveLoss(**self.config.get('contrastive', {})),
            'infonce': InfoNCELoss(**self.config.get('infonce', {})),
            'triplet': TripletLoss(**self.config.get('triplet', {})),
            'circle': CircleLoss(**self.config.get('circle', {})),
            'supcon': SupConLoss(**self.config.get('supcon', {})),
            'align_uniform': AlignUniformLoss(**self.config.get('align_uniform', {}))
        }
        
        self.trainer = AlignmentTrainer(
            config=self.config.get('trainer', {})
        )
        
        self.retriever = CrossModalRetriever(
            config=self.config.get('retriever', {})
        )
        
        self.consistency_checker = ConsistencyChecker(
            config=self.config.get('consistency', {})
        )
        
        # State
        self.alignment_models = {}
        self.alignment_history = []
        self.embedding_cache = {}
        
        # Metrics
        self.metrics = {
            'total_alignments': 0,
            'successful_alignments': 0,
            'average_alignment_score': 0.0,
            'total_loss': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def align_modalities(self,
                        modality_a: str,
                        modality_b: str,
                        embeddings_a: torch.Tensor,
                        embeddings_b: torch.Tensor,
                        alignment_type: Union[str, AlignmentType] = 'contrastive',
                        **kwargs) -> AlignmentResult:
        """
        Align two modalities using specified alignment method.
        
        Args:
            modality_a: Name of first modality
            modality_b: Name of second modality
            embeddings_a: Embeddings for modality A [batch_size, embedding_dim]
            embeddings_b: Embeddings for modality B [batch_size, embedding_dim]
            alignment_type: Type of alignment to use
            **kwargs: Additional alignment parameters
            
        Returns:
            Alignment result
        """
        try:
            # Validate inputs
            if embeddings_a.shape[0] != embeddings_b.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: {embeddings_a.shape[0]} != {embeddings_b.shape[0]}"
                )
            
            if embeddings_a.shape[1] != embeddings_b.shape[1]:
                # Project to same dimension if needed
                embeddings_b = self._project_embeddings(
                    embeddings_b, embeddings_a.shape[1]
                )
            
            # Normalize embeddings
            embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
            
            # Create alignment pair
            pair = AlignmentPair(
                modality_a=modality_a,
                modality_b=modality_b,
                embeddings_a=embeddings_a,
                embeddings_b=embeddings_b
            )
            
            # Calculate similarity matrix
            similarity = self.calculate_similarity(embeddings_a, embeddings_b)
            pair.similarity_matrix = similarity
            
            # Perform alignment based on type
            if isinstance(alignment_type, str):
                alignment_type = AlignmentType(alignment_type)
            
            if alignment_type == AlignmentType.CONTRASTIVE:
                result = self._align_contrastive(pair, **kwargs)
            elif alignment_type == AlignmentType.TRIPLET:
                result = self._align_triplet(pair, **kwargs)
            elif alignment_type == AlignmentType.CROSS_ATTENTION:
                result = self._align_cross_attention(pair, **kwargs)
            elif alignment_type == AlignmentType.CYCLE_CONSISTENCY:
                result = self._align_cycle_consistency(pair, **kwargs)
            elif alignment_type == AlignmentType.ADAPTIVE:
                result = self._align_adaptive(pair, **kwargs)
            elif alignment_type == AlignmentType.HIERARCHICAL:
                result = self._align_hierarchical(pair, **kwargs)
            else:
                raise ValueError(f"Unsupported alignment type: {alignment_type}")
            
            # Update metrics
            self._update_metrics(result)
            
            # Cache aligned embeddings
            if result.aligned_embeddings is not None:
                cache_key = f"{modality_a}_{modality_b}"
                self.embedding_cache[cache_key] = {
                    'embeddings_a': embeddings_a,
                    'embeddings_b': embeddings_b,
                    'aligned_embeddings': result.aligned_embeddings,
                    'similarity': result.similarity_matrix,
                    'timestamp': time.time()
                }
            
            # Log alignment
            self.alignment_history.append({
                'modality_a': modality_a,
                'modality_b': modality_b,
                'alignment_type': alignment_type.value,
                'timestamp': time.time(),
                'result': result,
                'kwargs': kwargs
            })
            
            self.logger.info(
                f"Aligned {modality_a} and {modality_b} using {alignment_type.value}. "
                f"Score: {result.alignment_score:.4f}, Confidence: {result.confidence:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to align modalities: {e}")
            return AlignmentResult(
                success=False,
                alignment_score=0.0,
                confidence=0.0,
                metrics={'error': str(e)}
            )
    
    def _align_contrastive(self, pair: AlignmentPair, **kwargs) -> AlignmentResult:
        """Align using contrastive learning."""
        # Get contrastive loss
        temperature = kwargs.get('temperature', 0.07)
        loss_fn = self.loss_functions['contrastive']
        
        # Calculate loss
        loss = loss_fn(pair.embeddings_a, pair.embeddings_b)
        
        # Calculate alignment score (1 - normalized loss)
        alignment_score = 1.0 - min(loss.item() / 10.0, 1.0)
        
        # Calculate confidence based on similarity consistency
        similarity_std = pair.similarity_matrix.std().item()
        confidence = 1.0 / (1.0 + similarity_std)
        
        return AlignmentResult(
            success=True,
            alignment_score=alignment_score,
            confidence=confidence,
            aligned_embeddings=torch.cat([pair.embeddings_a, pair.embeddings_b], dim=1),
            similarity_matrix=pair.similarity_matrix,
            alignment_loss=loss.item(),
            metrics={
                'contrastive_loss': loss.item(),
                'similarity_mean': pair.similarity_matrix.mean().item(),
                'similarity_std': similarity_std
            }
        )
    
    def _align_triplet(self, pair: AlignmentPair, **kwargs) -> AlignmentResult:
        """Align using triplet loss."""
        # Create triplets (anchor, positive, negative)
        batch_size = pair.embeddings_a.shape[0]
        
        # Use embeddings_a as anchors, embeddings_b as positives
        anchors = pair.embeddings_a
        positives = pair.embeddings_b
        
        # Create negatives by shuffling positives
        negatives = positives[torch.randperm(batch_size)]
        
        # Calculate triplet loss
        margin = kwargs.get('margin', 1.0)
        loss_fn = self.loss_functions['triplet']
        loss = loss_fn(anchors, positives, negatives)
        
        # Calculate alignment score
        alignment_score = max(0.0, 1.0 - loss.item() / margin)
        
        # Calculate confidence
        pos_dist = F.pairwise_distance(anchors, positives).mean().item()
        neg_dist = F.pairwise_distance(anchors, negatives).mean().item()
        confidence = min(pos_dist / neg_dist, 1.0) if neg_dist > 0 else 0.0
        
        return AlignmentResult(
            success=True,
            alignment_score=alignment_score,
            confidence=confidence,
            aligned_embeddings=anchors,  # Return anchors as aligned
            similarity_matrix=pair.similarity_matrix,
            alignment_loss=loss.item(),
            metrics={
                'triplet_loss': loss.item(),
                'positive_distance': pos_dist,
                'negative_distance': neg_dist,
                'margin': margin
            }
        )
    
    def _align_cross_attention(self, pair: AlignmentPair, **kwargs) -> AlignmentResult:
        """Align using cross-attention."""
        batch_size, embed_dim = pair.embeddings_a.shape
        
        # Create cross-attention layer
        cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=kwargs.get('num_heads', 8),
            batch_first=True
        )
        
        # Apply cross-attention
        attended_a, _ = cross_attn(
            pair.embeddings_a.unsqueeze(1),
            pair.embeddings_b.unsqueeze(1),
            pair.embeddings_b.unsqueeze(1)
        )
        
        attended_b, _ = cross_attn(
            pair.embeddings_b.unsqueeze(1),
            pair.embeddings_a.unsqueeze(1),
            pair.embeddings_a.unsqueeze(1)
        )
        
        attended_a = attended_a.squeeze(1)
        attended_b = attended_b.squeeze(1)
        
        # Calculate alignment
        similarity = F.cosine_similarity(attended_a, attended_b).mean()
        alignment_score = similarity.item()
        
        # Calculate reconstruction loss
        recon_loss_a = F.mse_loss(attended_a, pair.embeddings_a).item()
        recon_loss_b = F.mse_loss(attended_b, pair.embeddings_b).item()
        confidence = 1.0 / (1.0 + (recon_loss_a + recon_loss_b) / 2.0)
        
        return AlignmentResult(
            success=True,
            alignment_score=alignment_score,
            confidence=confidence,
            aligned_embeddings=torch.cat([attended_a, attended_b], dim=1),
            similarity_matrix=similarity.unsqueeze(0).unsqueeze(0),
            alignment_loss=(recon_loss_a + recon_loss_b) / 2.0,
            metrics={
                'cross_attention_similarity': alignment_score,
                'reconstruction_loss_a': recon_loss_a,
                'reconstruction_loss_b': recon_loss_b
            }
        )
    
    def _align_cycle_consistency(self, pair: AlignmentPair, **kwargs) -> AlignmentResult:
        """Align using cycle consistency."""
        # Calculate forward and backward mappings
        similarity = pair.similarity_matrix
        
        # Find nearest neighbors
        _, nn_ab = similarity.max(dim=1)  # A -> B
        _, nn_ba = similarity.max(dim=0)  # B -> A
        
        # Calculate cycle consistency
        cycle_scores = []
        for i in range(similarity.shape[0]):
            j = nn_ab[i]
            i_prime = nn_ba[j]
            cycle_scores.append(1.0 if i == i_prime else 0.0)
        
        cycle_consistency = torch.tensor(cycle_scores).float().mean()
        alignment_score = cycle_consistency.item()
        
        # Calculate confidence based on cycle lengths
        confidence = alignment_score
        
        return AlignmentResult(
            success=True,
            alignment_score=alignment_score,
            confidence=confidence,
            aligned_embeddings=pair.embeddings_a,  # Return A as reference
            similarity_matrix=similarity,
            alignment_loss=1.0 - alignment_score,
            metrics={
                'cycle_consistency': alignment_score,
                'forward_matches': nn_ab.tolist(),
                'backward_matches': nn_ba.tolist()
            }
        )
    
    def _align_adaptive(self, pair: AlignmentPair, **kwargs) -> AlignmentResult:
        """Adaptive alignment combining multiple methods."""
        # Try multiple alignment methods
        methods = ['contrastive', 'triplet', 'cross_attention']
        results = []
        
        for method in methods:
            if method == 'contrastive':
                result = self._align_contrastive(pair, **kwargs)
            elif method == 'triplet':
                result = self._align_triplet(pair, **kwargs)
            elif method == 'cross_attention':
                result = self._align_cross_attention(pair, **kwargs)
            results.append(result)
        
        # Weighted combination based on confidence
        total_confidence = sum(r.confidence for r in results)
        if total_confidence > 0:
            weights = [r.confidence / total_confidence for r in results]
            alignment_score = sum(r.alignment_score * w for r, w in zip(results, weights))
            combined_confidence = sum(r.confidence * w for r, w in zip(results, weights))
        else:
            alignment_score = sum(r.alignment_score for r in results) / len(results)
            combined_confidence = sum(r.confidence for r in results) / len(results)
        
        # Combine aligned embeddings (weighted average)
        aligned_embeddings = torch.zeros_like(results[0].aligned_embeddings)
        for r, w in zip(results, weights):
            if r.aligned_embeddings is not None:
                aligned_embeddings += r.aligned_embeddings * w
        
        return AlignmentResult(
            success=True,
            alignment_score=alignment_score,
            confidence=combined_confidence,
            aligned_embeddings=aligned_embeddings,
            similarity_matrix=pair.similarity_matrix,
            alignment_loss=sum(r.alignment_loss or 0.0 for r in results) / len(results),
            metrics={
                'adaptive_score': alignment_score,
                'method_weights': dict(zip(methods, weights)),
                'individual_scores': {m: r.alignment_score for m, r in zip(methods, results)}
            }
        )
    
    def _align_hierarchical(self, pair: AlignmentPair, **kwargs) -> AlignmentResult:
        """Hierarchical alignment at multiple scales."""
        # This is a simplified version - in practice would use multi-scale embeddings
        batch_size, embed_dim = pair.embeddings_a.shape
        
        # Create hierarchical clusters
        n_clusters = kwargs.get('n_clusters', min(8, batch_size // 4))
        
        if n_clusters > 1:
            # Cluster embeddings
            from sklearn.cluster import KMeans
            
            kmeans_a = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans_b = KMeans(n_clusters=n_clusters, random_state=42)
            
            clusters_a = kmeans_a.fit_predict(pair.embeddings_a.cpu().numpy())
            clusters_b = kmeans_b.fit_predict(pair.embeddings_b.cpu().numpy())
            
            # Calculate cluster alignment
            cluster_similarity = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    mask_a = clusters_a == i
                    mask_b = clusters_b == j
                    if mask_a.any() and mask_b.any():
                        cluster_emb_a = pair.embeddings_a[mask_a].mean(dim=0)
                        cluster_emb_b = pair.embeddings_b[mask_b].mean(dim=0)
                        sim = F.cosine_similarity(
                            cluster_emb_a.unsqueeze(0),
                            cluster_emb_b.unsqueeze(0)
                        ).item()
                        cluster_similarity[i, j] = sim
            
            # Calculate hierarchical alignment score
            hierarchical_score = cluster_similarity.mean()
            
            # Assign cluster-aware embeddings
            cluster_embeddings_a = torch.stack([
                pair.embeddings_a[clusters_a == i].mean(dim=0) 
                for i in range(n_clusters) if (clusters_a == i).any()
            ])
            
            cluster_embeddings_b = torch.stack([
                pair.embeddings_b[clusters_b == i].mean(dim=0) 
                for i in range(n_clusters) if (clusters_b == i).any()
            ])
            
            # Align cluster centroids
            cluster_result = self._align_contrastive(
                AlignmentPair(
                    modality_a=pair.modality_a,
                    modality_b=pair.modality_b,
                    embeddings_a=cluster_embeddings_a,
                    embeddings_b=cluster_embeddings_b
                ),
                **kwargs
            )
            
            alignment_score = (hierarchical_score + cluster_result.alignment_score) / 2
            confidence = cluster_result.confidence
            
        else:
            # Fall back to standard contrastive alignment
            result = self._align_contrastive(pair, **kwargs)
            alignment_score = result.alignment_score
            confidence = result.confidence
        
        return AlignmentResult(
            success=True,
            alignment_score=alignment_score,
            confidence=confidence,
            aligned_embeddings=pair.embeddings_a,  # Return original
            similarity_matrix=pair.similarity_matrix,
            alignment_loss=1.0 - alignment_score,
            metrics={
                'hierarchical_score': alignment_score,
                'n_clusters': n_clusters
            }
        )
    
    def calculate_similarity(self,
                           embeddings_a: torch.Tensor,
                           embeddings_b: torch.Tensor,
                           similarity_type: str = 'cosine') -> torch.Tensor:
        """
        Calculate similarity matrix between two sets of embeddings.
        
        Args:
            embeddings_a: First set of embeddings [N, D]
            embeddings_b: Second set of embeddings [N, D]
            similarity_type: Type of similarity ('cosine', 'dot', 'euclidean', 'mahalanobis')
            
        Returns:
            Similarity matrix [N, N]
        """
        if similarity_type == 'cosine':
            # Normalize embeddings
            embeddings_a_norm = F.normalize(embeddings_a, p=2, dim=1)
            embeddings_b_norm = F.normalize(embeddings_b, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.mm(embeddings_a_norm, embeddings_b_norm.t())
            
        elif similarity_type == 'dot':
            # Dot product similarity
            similarity = torch.mm(embeddings_a, embeddings_b.t())
            
        elif similarity_type == 'euclidean':
            # Convert Euclidean distance to similarity
            # similarity = 1 / (1 + distance)
            n_a, n_b = embeddings_a.shape[0], embeddings_b.shape[0]
            expanded_a = embeddings_a.unsqueeze(1).expand(n_a, n_b, -1)
            expanded_b = embeddings_b.unsqueeze(0).expand(n_a, n_b, -1)
            distances = torch.sqrt(((expanded_a - expanded_b) ** 2).sum(dim=2))
            similarity = 1.0 / (1.0 + distances)
            
        elif similarity_type == 'mahalanobis':
            # Mahalanobis distance (simplified - assumes identity covariance)
            # In practice, would need to compute/estimate covariance matrix
            n_a, n_b = embeddings_a.shape[0], embeddings_b.shape[0]
            expanded_a = embeddings_a.unsqueeze(1).expand(n_a, n_b, -1)
            expanded_b = embeddings_b.unsqueeze(0).expand(n_a, n_b, -1)
            differences = expanded_a - expanded_b
            # Assuming identity covariance for simplicity
            mahalanobis = torch.sqrt((differences ** 2).sum(dim=2))
            similarity = 1.0 / (1.0 + mahalanobis)
            
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        return similarity
    
    def train_alignment(self,
                       dataset: Union[AlignmentDataset, Any],
                       modality_pairs: List[Tuple[str, str]],
                       config: Optional[AlignmentConfig] = None) -> Dict[str, Any]:
        """
        Train alignment models for multiple modality pairs.
        
        Args:
            dataset: Dataset containing multimodal examples
            modality_pairs: List of (modality_a, modality_b) pairs to align
            config: Training configuration
            
        Returns:
            Training results
        """
        return self.trainer.train(
            dataset=dataset,
            modality_pairs=modality_pairs,
            config=config
        )
    
    def retrieve_cross_modal(self,
                           query_modality: str,
                           query_embeddings: torch.Tensor,
                           target_modality: str,
                           k: int = 10,
                           **kwargs) -> Dict[str, Any]:
        """
        Retrieve across modalities.
        
        Args:
            query_modality: Modality of query embeddings
            query_embeddings: Query embeddings [N, D]
            target_modality: Modality to retrieve from
            k: Number of results to retrieve
            
        Returns:
            Retrieval results
        """
        return self.retriever.retrieve(
            query_modality=query_modality,
            query_embeddings=query_embeddings,
            target_modality=target_modality,
            k=k,
            **kwargs
        )
    
    def check_consistency(self,
                         modalities: List[str],
                         embeddings: Dict[str, torch.Tensor],
                         **kwargs) -> Dict[str, Any]:
        """
        Check consistency across multiple modalities.
        
        Args:
            modalities: List of modality names
            embeddings: Dictionary mapping modality to embeddings
            
        Returns:
            Consistency check results
        """
        return self.consistency_checker.check(
            modalities=modalities,
            embeddings=embeddings,
            **kwargs
        )
    
    def get_alignment_model(self,
                          modality_a: str,
                          modality_b: str) -> Optional[nn.Module]:
        """
        Get trained alignment model for modality pair.
        
        Args:
            modality_a: First modality
            modality_b: Second modality
            
        Returns:
            Alignment model if exists
        """
        model_key = f"{modality_a}_{modality_b}"
        return self.alignment_models.get(model_key)
    
    def save_alignment_model(self,
                           modality_a: str,
                           modality_b: str,
                           path: str) -> bool:
        """
        Save alignment model to disk.
        
        Args:
            modality_a: First modality
            modality_b: Second modality
            path: Path to save model
            
        Returns:
            Success status
        """
        model = self.get_alignment_model(modality_a, modality_b)
        if model is None:
            self.logger.warning(f"No alignment model for {modality_a}-{modality_b}")
            return False
        
        try:
            torch.save(model.state_dict(), path)
            self.logger.info(f"Saved alignment model to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save alignment model: {e}")
            return False
    
    def load_alignment_model(self,
                           modality_a: str,
                           modality_b: str,
                           path: str,
                           model_class: Optional[nn.Module] = None) -> bool:
        """
        Load alignment model from disk.
        
        Args:
            modality_a: First modality
            modality_b: Second modality
            path: Path to load model from
            model_class: Model class to instantiate
            
        Returns:
            Success status
        """
        try:
            if model_class is None:
                # Default to simple projection model
                model_class = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256)
                )
            
            model = model_class()
            model.load_state_dict(torch.load(path))
            
            model_key = f"{modality_a}_{modality_b}"
            self.alignment_models[model_key] = model
            
            self.logger.info(f"Loaded alignment model from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load alignment model: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get alignment metrics.
        
        Returns:
            Alignment metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate success rate
        total = metrics['total_alignments']
        if total > 0:
            metrics['success_rate'] = metrics['successful_alignments'] / total
        else:
            metrics['success_rate'] = 0.0
        
        # Add current state
        metrics['alignment_models'] = len(self.alignment_models)
        metrics['cached_embeddings'] = len(self.embedding_cache)
        metrics['alignment_history'] = len(self.alignment_history)
        
        return metrics
    
    def _project_embeddings(self,
                           embeddings: torch.Tensor,
                           target_dim: int) -> torch.Tensor:
        """Project embeddings to target dimension."""
        if embeddings.shape[1] == target_dim:
            return embeddings
        
        # Create projection layer if needed
        proj_key = f"proj_{embeddings.shape[1]}_to_{target_dim}"
        if proj_key not in self.alignment_models:
            self.alignment_models[proj_key] = nn.Linear(
                embeddings.shape[1], target_dim
            )
        
        projection = self.alignment_models[proj_key]
        return projection(embeddings)
    
    def _update_metrics(self, result: AlignmentResult):
        """Update alignment metrics."""
        self.metrics['total_alignments'] += 1
        
        if result.success:
            self.metrics['successful_alignments'] += 1
        
        # Update average alignment score
        total_score = self.metrics['average_alignment_score'] * (self.metrics['total_alignments'] - 1)
        self.metrics['average_alignment_score'] = (
            total_score + result.alignment_score
        ) / self.metrics['total_alignments']
        
        # Update total loss
        if result.alignment_loss is not None:
            self.metrics['total_loss'] += result.alignment_loss

# Import time for timestamps
import time
from sklearn.cluster import KMeans