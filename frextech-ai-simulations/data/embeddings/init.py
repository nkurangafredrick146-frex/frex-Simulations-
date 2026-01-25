"""
Embeddings Module
Management and processing of precomputed embeddings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import json
from datetime import datetime
import hashlib

import numpy as np
import torch
from tqdm import tqdm

from ..utils.file_io import (
    ensure_directory, save_numpy, load_numpy,
    save_tensor, load_tensor, write_json, read_json,
    get_file_size, calculate_md5
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingManager:
    """
    Manager for embedding storage and retrieval.
    
    Features:
    - Embedding storage and organization
    - Embedding similarity search
    - Embedding visualization
    - Embedding versioning
    - Batch embedding operations
    """
    
    def __init__(self, embeddings_root: Union[str, Path] = "./data/embeddings"):
        """
        Initialize embedding manager.
        
        Args:
            embeddings_root: Root directory for embeddings
        """
        self.embeddings_root = Path(embeddings_root)
        ensure_directory(self.embeddings_root)
        
        # Metadata and index
        self.metadata_file = self.embeddings_root / "metadata.json"
        self.index_file = self.embeddings_root / "index.json"
        
        self.metadata = self._load_metadata()
        self.index = self._load_index()
        
        logger.info(f"Initialized EmbeddingManager at {self.embeddings_root}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                return read_json(self.metadata_file)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {'embeddings': {}, 'models': {}, 'stats': {}}
        else:
            return {'embeddings': {}, 'models': {}, 'stats': {}}
    
    def _load_index(self) -> Dict[str, Any]:
        """Load index from file."""
        if self.index_file.exists():
            try:
                return read_json(self.index_file)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                return {'by_dataset': {}, 'by_model': {}, 'by_feature': {}}
        else:
            return {'by_dataset': {}, 'by_model': {}, 'by_feature': {}}
    
    def _save_metadata(self):
        """Save metadata to file."""
        write_json(self.metadata, self.metadata_file)
    
    def _save_index(self):
        """Save index to file."""
        write_json(self.index, self.index_file)
    
    def store_embeddings(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        dataset_name: str,
        model_name: str,
        sample_ids: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        overwrite: bool = False
    ) -> str:
        """
        Store embeddings in the embedding store.
        
        Args:
            embeddings: Embeddings array/tensor
            dataset_name: Name of the dataset
            model_name: Name of the model that generated embeddings
            sample_ids: List of sample IDs (optional)
            features: List of feature names (optional)
            metadata: Additional metadata
            version: Embedding version
            overwrite: Whether to overwrite existing embeddings
            
        Returns:
            Embedding ID
        """
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Create embedding ID
        embedding_id = self._generate_embedding_id(
            dataset_name, model_name, version, embeddings_np
        )
        
        # Check if already exists
        if embedding_id in self.metadata['embeddings'] and not overwrite:
            logger.info(f"Embeddings '{embedding_id}' already exist. Returning existing ID.")
            return embedding_id
        
        # Create embedding directory
        embedding_dir = self.embeddings_root / embedding_id
        if embedding_dir.exists() and overwrite:
            import shutil
            shutil.rmtree(embedding_dir)
        
        ensure_directory(embedding_dir)
        
        # Save embeddings
        embeddings_file = embedding_dir / "embeddings.npy"
        save_numpy(embeddings_np, embeddings_file)
        
        # Save metadata
        embedding_metadata = {
            'embedding_id': embedding_id,
            'dataset_name': dataset_name,
            'model_name': model_name,
            'version': version,
            'shape': embeddings_np.shape,
            'dtype': str(embeddings_np.dtype),
            'sample_count': embeddings_np.shape[0],
            'embedding_dim': embeddings_np.shape[1],
            'sample_ids': sample_ids,
            'features': features,
            'created_date': datetime.now().isoformat(),
            'embeddings_file': str(embeddings_file.relative_to(self.embeddings_root)),
            'metadata': metadata or {}
        }
        
        # Update global metadata
        self.metadata['embeddings'][embedding_id] = embedding_metadata
        
        # Update model metadata
        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {
                'name': model_name,
                'embeddings': [],
                'total_samples': 0,
                'last_updated': datetime.now().isoformat()
            }
        
        self.metadata['models'][model_name]['embeddings'].append(embedding_id)
        self.metadata['models'][model_name]['total_samples'] += embeddings_np.shape[0]
        self.metadata['models'][model_name]['last_updated'] = datetime.now().isoformat()
        
        # Update index
        self._update_index(embedding_id, embedding_metadata)
        
        # Update statistics
        self._update_stats()
        
        # Save metadata and index
        self._save_metadata()
        self._save_index()
        
        logger.info(f"Stored embeddings '{embedding_id}': {embeddings_np.shape}")
        return embedding_id
    
    def _generate_embedding_id(
        self,
        dataset_name: str,
        model_name: str,
        version: str,
        embeddings: np.ndarray
    ) -> str:
        """Generate unique embedding ID."""
        # Create hash from embeddings
        embeddings_hash = hashlib.md5(embeddings.tobytes()).hexdigest()[:16]
        
        # Create ID
        base_id = f"{dataset_name}_{model_name}_v{version}"
        embedding_id = f"{base_id}_{embeddings_hash}"
        
        return embedding_id
    
    def _update_index(self, embedding_id: str, metadata: Dict[str, Any]):
        """Update search index."""
        # Index by dataset
        dataset_name = metadata['dataset_name']
        if dataset_name not in self.index['by_dataset']:
            self.index['by_dataset'][dataset_name] = []
        
        if embedding_id not in self.index['by_dataset'][dataset_name]:
            self.index['by_dataset'][dataset_name].append(embedding_id)
        
        # Index by model
        model_name = metadata['model_name']
        if model_name not in self.index['by_model']:
            self.index['by_model'][model_name] = []
        
        if embedding_id not in self.index['by_model'][model_name]:
            self.index['by_model'][model_name].append(embedding_id)
        
        # Index by features if available
        if metadata.get('features'):
            for feature in metadata['features']:
                if feature not in self.index['by_feature']:
                    self.index['by_feature'][feature] = []
                
                if embedding_id not in self.index['by_feature'][feature]:
                    self.index['by_feature'][feature].append(embedding_id)
    
    def _update_stats(self):
        """Update global statistics."""
        total_embeddings = len(self.metadata['embeddings'])
        total_samples = sum(
            meta['sample_count'] 
            for meta in self.metadata['embeddings'].values()
        )
        total_size = sum(
            get_file_size(self.embeddings_root / meta['embeddings_file'])
            for meta in self.metadata['embeddings'].values()
        )
        
        self.metadata['stats'] = {
            'total_embeddings': total_embeddings,
            'total_samples': total_samples,
            'total_size_bytes': total_size,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_embeddings(self, embedding_id: str) -> Optional[np.ndarray]:
        """
        Retrieve embeddings by ID.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Embeddings array or None if not found
        """
        if embedding_id not in self.metadata['embeddings']:
            logger.error(f"Embedding '{embedding_id}' not found")
            return None
        
        metadata = self.metadata['embeddings'][embedding_id]
        embeddings_file = self.embeddings_root / metadata['embeddings_file']
        
        try:
            embeddings = load_numpy(embeddings_file)
            logger.debug(f"Loaded embeddings '{embedding_id}': {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings '{embedding_id}': {e}")
            return None
    
    def get_metadata(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for embeddings.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata['embeddings'].get(embedding_id)
    
    def search_embeddings(
        self,
        dataset_name: Optional[str] = None,
        model_name: Optional[str] = None,
        feature: Optional[str] = None,
        min_samples: Optional[int] = None,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for embeddings by criteria.
        
        Args:
            dataset_name: Filter by dataset name
            model_name: Filter by model name
            feature: Filter by feature
            min_samples: Minimum number of samples
            max_samples: Maximum number of samples
            
        Returns:
            List of matching embedding metadata
        """
        # Start with all embeddings
        candidate_ids = set(self.metadata['embeddings'].keys())
        
        # Apply filters
        if dataset_name:
            dataset_ids = set(self.index['by_dataset'].get(dataset_name, []))
            candidate_ids &= dataset_ids
        
        if model_name:
            model_ids = set(self.index['by_model'].get(model_name, []))
            candidate_ids &= model_ids
        
        if feature:
            feature_ids = set(self.index['by_feature'].get(feature, []))
            candidate_ids &= feature_ids
        
        # Get metadata for candidates
        results = []
        for embedding_id in candidate_ids:
            metadata = self.metadata['embeddings'][embedding_id]
            
            # Apply sample count filters
            sample_count = metadata['sample_count']
            
            if min_samples is not None and sample_count < min_samples:
                continue
            
            if max_samples is not None and sample_count > max_samples:
                continue
            
            results.append(metadata)
        
        # Sort by creation date (newest first)
        results.sort(key=lambda x: x['created_date'], reverse=True)
        
        return results
    
    def compute_similarity(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor],
        embedding_id: str,
        metric: str = "cosine",
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Compute similarity between query and stored embeddings.
        
        Args:
            query_embedding: Query embedding(s)
            embedding_id: Target embedding ID
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity) pairs
        """
        # Get target embeddings
        target_embeddings = self.get_embeddings(embedding_id)
        if target_embeddings is None:
            return []
        
        # Convert query to numpy
        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.detach().cpu().numpy()
        else:
            query_np = query_embedding
        
        # Ensure query is 2D
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        
        # Compute similarities
        similarities = []
        
        if metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(query_np, target_embeddings)
        
        elif metric == "euclidean":
            from sklearn.metrics.pairwise import euclidean_distances
            dist_matrix = euclidean_distances(query_np, target_embeddings)
            # Convert distance to similarity (higher is more similar)
            sim_matrix = 1.0 / (1.0 + dist_matrix)
        
        elif metric == "dot":
            sim_matrix = np.dot(query_np, target_embeddings.T)
        
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
        
        # Get