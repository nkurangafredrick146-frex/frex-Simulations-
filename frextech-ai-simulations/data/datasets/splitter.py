"""
Dataset Splitter Module
Utilities for splitting datasets into train/validation/test sets.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.cluster import KMeans
import pandas as pd

from .base_dataset import BaseDataset, DatasetConfig, DatasetPhase
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

class SplitStrategy(Enum):
    """Strategies for splitting datasets."""
    RANDOM = "random"
    STRATIFIED = "stratified"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    CLUSTER = "cluster"
    CUSTOM = "custom"
    K_FOLD = "k_fold"
    TIME_SERIES = "time_series"

@dataclass
class SplitConfig:
    """
    Configuration for dataset splitting.
    
    Attributes:
        strategy: Splitting strategy
        ratios: Dictionary of split names to ratios (must sum to 1.0)
        random_state: Random seed for reproducibility
        stratify_by: Column name for stratification
        temporal_column: Column name for temporal ordering
        spatial_columns: Column names for spatial coordinates
        n_clusters: Number of clusters for cluster-based splitting
        n_folds: Number of folds for k-fold splitting
        test_size: Size of test set (alternative to ratios)
        val_size: Size of validation set (alternative to ratios)
        shuffle: Whether to shuffle before splitting
        group_by: Column name for group-based splitting
    """
    strategy: SplitStrategy = SplitStrategy.RANDOM
    ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    random_state: int = 42
    stratify_by: Optional[str] = None
    temporal_column: Optional[str] = None
    spatial_columns: Optional[List[str]] = None
    n_clusters: int = 5
    n_folds: int = 5
    test_size: Optional[float] = None
    val_size: Optional[float] = None
    shuffle: bool = True
    group_by: Optional[str] = None
    
    def __post_init__(self):
        """Validate split configuration."""
        # Convert string to enum if needed
        if isinstance(self.strategy, str):
            self.strategy = SplitStrategy(self.strategy)
        
        # Validate ratios
        if self.ratios:
            total = sum(self.ratios.values())
            if abs(total - 1.0) > 1e-6:
                warnings.warn(f"Split ratios sum to {total}, not 1.0. Normalizing.")
                # Normalize ratios
                self.ratios = {k: v/total for k, v in self.ratios.items()}
        
        # Validate that split names are valid
        valid_names = {"train", "val", "validation", "test", "eval"}
        for name in self.ratios.keys():
            if name not in valid_names:
                warnings.warn(f"Unconventional split name: {name}")
        
        # Set default test/val sizes if using alternative specification
        if self.test_size is not None and self.val_size is not None:
            self.ratios = {
                "train": 1.0 - self.test_size - self.val_size,
                "val": self.val_size,
                "test": self.test_size
            }


class DatasetSplitter:
    """
    Split datasets into train/validation/test sets using various strategies.
    """
    
    def __init__(self, dataset: BaseDataset, config: Optional[SplitConfig] = None, **kwargs):
        """
        Initialize dataset splitter.
        
        Args:
            dataset: Dataset to split
            config: Split configuration
            **kwargs: Configuration overrides
        """
        self.dataset = dataset
        self.config = config or SplitConfig(**kwargs)
        
        # Validate dataset has samples
        if len(dataset) == 0:
            raise ValueError("Cannot split empty dataset")
        
        # Extract sample information
        self.sample_indices = list(range(len(dataset)))
        self.sample_metadata = self._extract_sample_metadata()
        
        logger.info(f"Initialized splitter for dataset with {len(dataset)} samples")
    
    def _extract_sample_metadata(self) -> List[Dict[str, Any]]:
        """Extract metadata from samples for splitting."""
        metadata = []
        
        for idx in range(min(10000, len(self.dataset))):  # Limit for efficiency
            try:
                sample = self.dataset.get_sample(idx, apply_transform=False)
                metadata.append(sample)
            except Exception as e:
                logger.warning(f"Failed to get sample {idx}: {e}")
                metadata.append({})
        
        # Pad if necessary
        if len(metadata) < len(self.dataset):
            metadata.extend([{}] * (len(self.dataset) - len(metadata)))
        
        return metadata
    
    def split(self) -> Dict[str, BaseDataset]:
        """
        Split dataset according to configuration.
        
        Returns:
            Dictionary mapping split names to dataset subsets
        """
        logger.info(f"Splitting dataset using {self.config.strategy.value} strategy")
        
        # Get indices for each split
        split_indices = self._compute_split_indices()
        
        # Create subset datasets
        splits = {}
        for split_name, indices in split_indices.items():
            if indices:  # Only create split if there are samples
                subset_dataset = self._create_subset_dataset(indices, split_name)
                splits[split_name] = subset_dataset
        
        # Log split statistics
        self._log_split_statistics(split_indices)
        
        return splits
    
    def _compute_split_indices(self) -> Dict[str, List[int]]:
        """Compute indices for each split based on strategy."""
        strategy = self.config.strategy
        
        if strategy == SplitStrategy.RANDOM:
            return self._random_split()
        
        elif strategy == SplitStrategy.STRATIFIED:
            return self._stratified_split()
        
        elif strategy == SplitStrategy.TEMPORAL:
            return self._temporal_split()
        
        elif strategy == SplitStrategy.SPATIAL:
            return self._spatial_split()
        
        elif strategy == SplitStrategy.CLUSTER:
            return self._cluster_split()
        
        elif strategy == SplitStrategy.K_FOLD:
            return self._kfold_split()
        
        elif strategy == SplitStrategy.TIME_SERIES:
            return self._time_series_split()
        
        elif strategy == SplitStrategy.CUSTOM:
            return self._custom_split()
        
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
    
    def _random_split(self) -> Dict[str, List[int]]:
        """Random split."""
        indices = np.array(self.sample_indices)
        
        if self.config.shuffle:
            rng = np.random.RandomState(self.config.random_state)
            rng.shuffle(indices)
        
        # Calculate split points
        split_points = self._calculate_split_points(len(indices))
        
        # Split indices
        split_indices = {}
        start = 0
        for split_name, split_size in split_points.items():
            end = start + split_size
            split_indices[split_name] = indices[start:end].tolist()
            start = end
        
        return split_indices
    
    def _stratified_split(self) -> Dict[str, List[int]]:
        """Stratified split based on labels."""
        # Extract labels for stratification
        labels = self._extract_stratification_labels()
        
        if labels is None:
            logger.warning("No stratification labels found, falling back to random split")
            return self._random_split()
        
        indices = np.array(self.sample_indices)
        
        # Split into train and temp (val+test)
        train_ratio = self.config.ratios.get('train', 0.7)
        temp_ratio = 1.0 - train_ratio
        
        train_indices, temp_indices, train_labels, _ = train_test_split(
            indices, labels,
            test_size=temp_ratio,
            stratify=labels,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle
        )
        
        # Split temp into val and test
        val_ratio = self.config.ratios.get('val', 0.15) / temp_ratio
        temp_labels = labels[temp_indices]
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=1.0 - val_ratio,
            stratify=temp_labels,
            random_state=self.config.random_state,
            shuffle=self.config.shuffle
        )
        
        return {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist()
        }
    
    def _extract_stratification_labels(self) -> Optional[np.ndarray]:
        """Extract labels for stratification."""
        if self.config.stratify_by:
            # Use specified column
            labels = []
            for metadata in self.sample_metadata:
                label = metadata.get(self.config.stratify_by)
                if label is None:
                    return None
                labels.append(str(label))
            return np.array(labels)
        else:
            # Try to find labels automatically
            for idx, metadata in enumerate(self.sample_metadata[:100]):  # Check first 100 samples
                for key, value in metadata.items():
                    if 'label' in key.lower() and value is not None:
                        # Found label column
                        labels = []
                        for md in self.sample_metadata:
                            label = md.get(key)
                            labels.append(str(label) if label is not None else 'unknown')
                        return np.array(labels)
        
        return None
    
    def _temporal_split(self) -> Dict[str, List[int]]:
        """Temporal split based on time column."""
        if not self.config.temporal_column:
            # Try to find temporal column
            for idx, metadata in enumerate(self.sample_metadata[:100]):
                for key, value in metadata.items():
                    if any(time_word in key.lower() for time_word in ['time', 'date', 'timestamp', 'year']):
                        self.config.temporal_column = key
                        break
                if self.config.temporal_column:
                    break
        
        if not self.config.temporal_column:
            logger.warning("No temporal column found, falling back to random split")
            return self._random_split()
        
        # Extract temporal values
        temporal_values = []
        for metadata in self.sample_metadata:
            value = metadata.get(self.config.temporal_column)
            if value is None:
                temporal_values.append(0)
            else:
                # Try to convert to numeric
                try:
                    temporal_values.append(float(value))
                except:
                    temporal_values.append(0)
        
        # Sort by temporal value
        sorted_indices = np.array(self.sample_indices)[np.argsort(temporal_values)]
        
        # Split based on temporal order
        split_points = self._calculate_split_points(len(sorted_indices))
        
        split_indices = {}
        start = 0
        for split_name, split_size in split_points.items():
            end = start + split_size
            split_indices[split_name] = sorted_indices[start:end].tolist()
            start = end
        
        return split_indices
    
    def _spatial_split(self) -> Dict[str, List[int]]:
        """Spatial split based on coordinates."""
        if not self.config.spatial_columns:
            # Try to find spatial columns
            spatial_cols = []
            for idx, metadata in enumerate(self.sample_metadata[:100]):
                for key in metadata.keys():
                    if any(coord in key.lower() for coord in ['x', 'y', 'z', 'lat', 'lon', 'coord']):
                        if key not in spatial_cols:
                            spatial_cols.append(key)
                
                if len(spatial_cols) >= 2:
                    break
            
            self.config.spatial_columns = spatial_cols[:2]  # Use first 2 spatial columns
        
        if len(self.config.spatial_columns) < 2:
            logger.warning("Insufficient spatial columns found, falling back to random split")
            return self._random_split()
        
        # Extract spatial coordinates
        coordinates = []
        for metadata in self.sample_metadata:
            coord = []
            for col in self.config.spatial_columns[:2]:  # Use at most 2 dimensions
                value = metadata.get(col, 0)
                try:
                    coord.append(float(value))
                except:
                    coord.append(0.0)
            
            # Pad if only 1 coordinate
            if len(coord) == 1:
                coord.append(0.0)
            
            coordinates.append(coord)
        
        coordinates = np.array(coordinates)
        
        # Use KMeans for spatial clustering
        n_clusters = min(self.config.n_clusters, len(coordinates))
        if n_clusters < 2:
            logger.warning("Too few samples for spatial clustering, falling back to random split")
            return self._random_split()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Split clusters among splits
        cluster_to_split = self._assign_clusters_to_splits(cluster_labels)
        
        # Create split indices
        split_indices = {name: [] for name in self.config.ratios.keys()}
        for idx, cluster in enumerate(cluster_labels):
            split_name = cluster_to_split[cluster]
            split_indices[split_name].append(idx)
        
        return split_indices
    
    def _assign_clusters_to_splits(self, cluster_labels: np.ndarray) -> Dict[int, str]:
        """Assign clusters to splits proportionally."""
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)
        
        # Calculate target number of clusters per split
        split_names = list(self.config.ratios.keys())
        split_ratios = list(self.config.ratios.values())
        
        target_clusters = [int(ratio * n_clusters) for ratio in split_ratios]
        
        # Adjust for rounding errors
        while sum(target_clusters) < n_clusters:
            target_clusters[np.argmax(split_ratios)] += 1
        
        # Assign clusters to splits
        cluster_to_split = {}
        cluster_idx = 0
        
        for split_name, n_target in zip(split_names, target_clusters):
            for _ in range(n_target):
                if cluster_idx < n_clusters:
                    cluster_to_split[unique_clusters[cluster_idx]] = split_name
                    cluster_idx += 1
        
        # Assign remaining clusters to largest split
        remaining_clusters = [c for c in unique_clusters if c not in cluster_to_split]
        largest_split = split_names[np.argmax(split_ratios)]
        for cluster in remaining_clusters:
            cluster_to_split[cluster] = largest_split
        
        return cluster_to_split
    
    def _cluster_split(self) -> Dict[str, List[int]]:
        """Cluster-based split using sample features."""
        # Extract features for clustering
        features = self._extract_clustering_features()
        
        if features is None or len(features) < 2:
            logger.warning("Could not extract features for clustering, falling back to random split")
            return self._random_split()
        
        # Apply KMeans clustering
        n_clusters = min(self.config.n_clusters, len(features))
        if n_clusters < 2:
            return self._random_split()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.random_state)
        cluster_labels = kmeans.fit_predict(features)
        
        # Split clusters among splits
        cluster_to_split = self._assign_clusters_to_splits(cluster_labels)
        
        # Create split indices
        split_indices = {name: [] for name in self.config.ratios.keys()}
        for idx, cluster in enumerate(cluster_labels):
            split_name = cluster_to_split[cluster]
            split_indices[split_name].append(idx)
        
        return split_indices
    
    def _extract_clustering_features(self) -> Optional[np.ndarray]:
        """Extract features for clustering."""
        features = []
        
        for metadata in self.sample_metadata[:1000]:  # Limit for efficiency
            feature_vector = []
            
            # Try to extract numeric features
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, str):
                    # Simple string encoding: length and hash
                    feature_vector.append(float(len(value)))
                    feature_vector.append(float(hash(value) % 1000) / 1000.0)
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    # Use first element if numeric
                    if isinstance(value[0], (int, float)):
                        feature_vector.append(float(value[0]))
            
            if feature_vector:
                features.append(feature_vector)
        
        if not features:
            return None
        
        # Pad features to same length
        max_len = max(len(f) for f in features)
        padded_features = []
        for f in features:
            if len(f) < max_len:
                f = f + [0.0] * (max_len - len(f))
            padded_features.append(f)
        
        return np.array(padded_features)
    
    def _kfold_split(self) -> Dict[str, List[int]]:
        """K-fold cross-validation split."""
        indices = np.array(self.sample_indices)
        
        if self.config.shuffle:
            rng = np.random.RandomState(self.config.random_state)
            rng.shuffle(indices)
        
        # Create k folds
        kfold = KFold(n_splits=self.config.n_folds, 
                     shuffle=False,  # Already shuffled
                     random_state=None)
        
        splits = {}
        for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(indices)):
            # Further split train_val into train and val
            train_val_indices = indices[train_val_idx]
            n_train_val = len(train_val_indices)
            
            # Use 80/20 split for train/val within train_val
            val_size = int(0.2 * n_train_val)
            
            if self.config.shuffle:
                rng = np.random.RandomState(self.config.random_state + fold_idx)
                shuffle_idx = rng.permutation(n_train_val)
                train_val_indices = train_val_indices[shuffle_idx]
            
            val_indices = train_val_indices[:val_size]
            train_indices = train_val_indices[val_size:]
            
            splits[f'fold_{fold_idx}_train'] = train_indices.tolist()
            splits[f'fold_{fold_idx}_val'] = val_indices.tolist()
            splits[f'fold_{fold_idx}_test'] = indices[test_idx].tolist()
        
        return splits
    
    def _time_series_split(self) -> Dict[str, List[int]]:
        """Time series split (sequential)."""
        indices = np.array(self.sample_indices)
        
        # Sort if temporal column available
        if self.config.temporal_column:
            temporal_values = []
            for metadata in self.sample_metadata:
                value = metadata.get(self.config.temporal_column, 0)
                try:
                    temporal_values.append(float(value))
                except:
                    temporal_values.append(0)
            
            indices = indices[np.argsort(temporal_values)]
        
        # Calculate split points for time series
        # Typically: train (oldest), val (middle), test (newest)
        split_points = self._calculate_split_points(len(indices))
        
        # For time series, we want contiguous blocks
        split_indices = {}
        start = 0
        for split_name, split_size in split_points.items():
            end = start + split_size
            split_indices[split_name] = indices[start:end].tolist()
            start = end
        
        return split_indices
    
    def _custom_split(self) -> Dict[str, List[int]]:
        """Custom split based on user-defined function."""
        # This would require a custom split function from the user
        # For now, fall back to random split
        logger.warning("Custom split strategy not implemented, using random split")
        return self._random_split()
    
    def _calculate_split_points(self, total_samples: int) -> Dict[str, int]:
        """Calculate number of samples for each split."""
        ratios = self.config.ratios
        
        # Calculate sample counts
        split_points = {}
        remaining = total_samples
        
        for split_name, ratio in ratios.items():
            if split_name == list(ratios.keys())[-1]:  # Last split gets remainder
                n_samples = remaining
            else:
                n_samples = int(ratio * total_samples)
                remaining -= n_samples
            
            split_points[split_name] = n_samples
        
        # Validate
        total_allocated = sum(split_points.values())
        if total_allocated != total_samples:
            # Adjust last split
            last_split = list(ratios.keys())[-1]
            split_points[last_split] += total_samples - total_allocated
        
        return split_points
    
    def _create_subset_dataset(self, indices: List[int], split_name: str) -> BaseDataset:
        """Create a subset dataset."""
        # Create new config for subset
        subset_config = self.dataset.config
        
        # Convert to dict for modification
        if hasattr(subset_config, 'to_dict'):
            config_dict = subset_config.to_dict()
        else:
            config_dict = subset_config.__dict__.copy()
        
        # Update config for subset
        config_dict['name'] = f"{self.dataset.name}_{split_name}"
        config_dict['phase'] = self._get_phase_from_split_name(split_name)
        config_dict['indices'] = indices
        
        # Create new dataset instance
        dataset_class = type(self.dataset)
        
        try:
            # Try to create with modified config
            subset_dataset = dataset_class(config=config_dict)
            logger.info(f"Created {split_name} subset with {len(subset_dataset)} samples")
            return subset_dataset
        except Exception as e:
            logger.error(f"Failed to create subset dataset: {e}")
            
            # Fallback: create a wrapper dataset
            from .subset_dataset import SubsetDataset
            return SubsetDataset(self.dataset, indices, name=f"{self.dataset.name}_{split_name}")
    
    def _get_phase_from_split_name(self, split_name: str) -> DatasetPhase:
        """Convert split name to dataset phase."""
        split_name_lower = split_name.lower()
        
        if 'train' in split_name_lower:
            return DatasetPhase.TRAIN
        elif 'val' in split_name_lower or 'validation' in split_name_lower:
            return DatasetPhase.VALIDATION
        elif 'test' in split_name_lower or 'eval' in split_name_lower:
            return DatasetPhase.TEST
        else:
            return DatasetPhase.TRAIN  # Default
    
    def _log_split_statistics(self, split_indices: Dict[str, List[int]]) -> None:
        """Log statistics about the splits."""
        total_samples = len(self.dataset)
        
        logger.info("Split Statistics:")
        logger.info("-" * 40)
        
        for split_name, indices in split_indices.items():
            n_samples = len(indices)
            percentage = (n_samples / total_samples) * 100 if total_samples > 0 else 0
            
            logger.info(f"{split_name:15} {n_samples:8d} samples ({percentage:6.2f}%)")
        
        logger.info("-" * 40)
        logger.info(f"{'Total':15} {total_samples:8d} samples (100.00%)")
    
    def save_splits(self, output_dir: str, format: str = 'json') -> None:
        """
        Save split indices to files.
        
        Args:
            output_dir: Directory to save split files
            format: File format ('json', 'txt', 'csv')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute splits
        splits = self.split()
        
        # Save each split
        for split_name, dataset in splits.items():
            indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
            
            if format == 'json':
                output_path = output_dir / f"{split_name}_indices.json"
                with open(output_path, 'w') as f:
                    json.dump(indices, f)
            
            elif format == 'txt':
                output_path = output_dir / f"{split_name}_indices.txt"
                with open(output_path, 'w') as f:
                    f.write('\n'.join(map(str, indices)))
            
            elif format == 'csv':
                output_path = output_dir / f"{split_name}_indices.csv"
                df = pd.DataFrame({'index': indices})
                df.to_csv(output_path, index=False)
        
        # Save split configuration
        config_path = output_dir / "split_config.json"
        config_dict = {
            'strategy': self.config.strategy.value,
            'ratios': self.config.ratios,
            'random_state': self.config.random_state,
            'total_samples': len(self.dataset)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved splits to {output_dir}")
    
    def load_splits(self, input_dir: str) -> Dict[str, BaseDataset]:
        """
        Load splits from saved files.
        
        Args:
            input_dir: Directory containing split files
            
        Returns:
            Dictionary of split datasets
        """
        input_dir = Path(input_dir)
        
        # Find split files
        split_files = list(input_dir.glob("*_indices.*"))
        
        splits = {}
        for split_file in split_files:
            # Extract split name
            split_name = split_file.stem.replace('_indices', '')
            
            # Load indices
            if split_file.suffix == '.json':
                with open(split_file, 'r') as f:
                    indices = json.load(f)
            elif split_file.suffix == '.txt':
                with open(split_file, 'r') as f:
                    indices = [int(line.strip()) for line in f if line.strip()]
            elif split_file.suffix == '.csv':
                df = pd.read_csv(split_file)
                indices = df['index'].tolist()
            else:
                continue
            
            # Create subset dataset
            subset_dataset = self._create_subset_dataset(indices, split_name)
            splits[split_name] = subset_dataset
        
        return splits

# Convenience functions
def create_splits(dataset: BaseDataset, 
                 strategy: Union[str, SplitStrategy] = SplitStrategy.RANDOM,
                 ratios: Optional[Dict[str, float]] = None,
                 **kwargs) -> Dict[str, BaseDataset]:
    """
    Create dataset splits.
    
    Args:
        dataset: Dataset to split
        strategy: Splitting strategy
        ratios: Split ratios
        **kwargs: Additional split configuration
        
    Returns:
        Dictionary of split datasets
    """
    config_kwargs = {'strategy': strategy}
    if ratios:
        config_kwargs['ratios'] = ratios
    config_kwargs.update(kwargs)
    
    splitter = DatasetSplitter(dataset, **config_kwargs)
    return splitter.split()

def train_val_test_split(dataset: BaseDataset,
                        train_size: float = 0.7,
                        val_size: float = 0.15,
                        test_size: float = 0.15,
                        **kwargs) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_size: Training set size ratio
        val_size: Validation set size ratio
        test_size: Test set size ratio
        **kwargs: Additional split configuration
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    ratios = {'train': train_size, 'val': val_size, 'test': test_size}
    
    splits = create_splits(dataset, ratios=ratios, **kwargs)
    
    return splits.get('train'), splits.get('val'), splits.get('test')

def kfold_split(dataset: BaseDataset,
               n_folds: int = 5,
               **kwargs) -> List[Tuple[BaseDataset, BaseDataset]]:
    """
    Create k-fold cross-validation splits.
    
    Args:
        dataset: Dataset to split
        n_folds: Number of folds
        **kwargs: Additional split configuration
        
    Returns:
        List of (train_dataset, val_dataset) tuples for each fold
    """
    splitter = DatasetSplitter(
        dataset,
        strategy=SplitStrategy.K_FOLD,
        n_folds=n_folds,
        **kwargs
    )
    
    splits = splitter.split()
    
    # Organize by fold
    folds = []
    for fold_idx in range(n_folds):
        train_key = f'fold_{fold_idx}_train'
        val_key = f'fold_{fold_idx}_val'
        
        if train_key in splits and val_key in splits:
            folds.append((splits[train_key], splits[val_key]))
    
    return folds