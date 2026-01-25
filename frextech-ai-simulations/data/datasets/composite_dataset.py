"""
Composite Dataset Module
Combine multiple datasets into a single dataset with weighted sampling.
"""

import random
import math
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset

from .base_dataset import BaseDataset, DatasetConfig, DatasetPhase
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

@dataclass
class CompositeDatasetConfig(DatasetConfig):
    """
    Configuration for composite dataset.
    
    Attributes:
        datasets: List of datasets to combine
        sampling_weights: Weights for sampling from each dataset
        sampling_strategy: Sampling strategy ('weighted', 'sequential', 'round_robin')
        max_samples_per_dataset: Maximum samples to take from each dataset
        shuffle_datasets: Whether to shuffle dataset order
        balance_datasets: Whether to balance datasets by class
        dataset_phases: Phase for each dataset
        dataset_transforms: Transforms for each dataset
    """
    datasets: List[BaseDataset] = field(default_factory=list)
    sampling_weights: Optional[List[float]] = None
    sampling_strategy: str = "weighted"  # "weighted", "sequential", "round_robin"
    max_samples_per_dataset: Optional[int] = None
    shuffle_datasets: bool = True
    balance_datasets: bool = False
    dataset_phases: Optional[List[DatasetPhase]] = None
    dataset_transforms: Optional[List[Callable]] = None
    
    def __post_init__(self):
        """Validate composite dataset configuration."""
        super().__post_init__()
        
        # Set dataset type
        self.type = "composite"
        
        # Validate datasets
        if not self.datasets:
            raise ValueError("composite dataset must have at least one dataset")
        
        # Set default sampling weights
        if self.sampling_weights is None:
            self.sampling_weights = [1.0] * len(self.datasets)
        
        # Validate weights
        if len(self.sampling_weights) != len(self.datasets):
            raise ValueError(f"sampling_weights length ({len(self.sampling_weights)}) "
                           f"must match datasets length ({len(self.datasets)})")
        
        # Normalize weights
        total_weight = sum(self.sampling_weights)
        if total_weight > 0:
            self.sampling_weights = [w / total_weight for w in self.sampling_weights]
        
        # Validate sampling strategy
        valid_strategies = ["weighted", "sequential", "round_robin"]
        if self.sampling_strategy not in valid_strategies:
            raise ValueError(f"sampling_strategy must be one of {valid_strategies}, "
                           f"got {self.sampling_strategy}")
        
        # Set dataset phases if not provided
        if self.dataset_phases is None:
            self.dataset_phases = [dataset.phase for dataset in self.datasets]
        elif len(self.dataset_phases) != len(self.datasets):
            raise ValueError(f"dataset_phases length ({len(self.dataset_phases)}) "
                           f"must match datasets length ({len(self.datasets)})")


class CompositeDataset(BaseDataset):
    """
    Composite dataset that combines multiple datasets.
    
    Features:
    - Weighted sampling from multiple datasets
    - Different sampling strategies
    - Dataset balancing
    - Per-dataset transforms
    """
    
    def __init__(self, config: CompositeDatasetConfig):
        """
        Initialize composite dataset.
        
        Args:
            config: Composite dataset configuration
        """
        if not isinstance(config, CompositeDatasetConfig):
            config = CompositeDatasetConfig.from_dict(config)
        
        # Initialize base dataset
        super().__init__(config)
        
        # Store datasets
        self.datasets = config.datasets
        self.num_datasets = len(self.datasets)
        
        # Sampling configuration
        self.sampling_weights = config.sampling_weights
        self.sampling_strategy = config.sampling_strategy
        self.max_samples_per_dataset = config.max_samples_per_dataset
        self.shuffle_datasets = config.shuffle_datasets
        self.balance_datasets = config.balance_datasets
        
        # Dataset phases and transforms
        self.dataset_phases = config.dataset_phases
        self.dataset_transforms = config.dataset_transforms or [None] * self.num_datasets
        
        # Build dataset index
        self._build_dataset_index()
        
        # Balance datasets if requested
        if self.balance_datasets:
            self._balance_datasets()
        
        logger.info(f"Created composite dataset with {self.num_datasets} datasets, "
                   f"total {len(self)} samples")
    
    def _build_dataset_index(self):
        """Build index mapping from global index to dataset and local index."""
        self.global_to_local = []
        self.dataset_offsets = []
        self.dataset_sizes = []
        
        current_offset = 0
        
        for i, dataset in enumerate(self.datasets):
            dataset_size = len(dataset)
            
            # Apply max samples limit
            if self.max_samples_per_dataset is not None:
                dataset_size = min(dataset_size, self.max_samples_per_dataset)
            
            self.dataset_sizes.append(dataset_size)
            self.dataset_offsets.append(current_offset)
            
            # Create mapping for this dataset
            for j in range(dataset_size):
                self.global_to_local.append((i, j))
            
            current_offset += dataset_size
        
        # Store total size
        self.total_size = current_offset
        
        # Shuffle if requested
        if self.shuffle_datasets:
            random.shuffle(self.global_to_local)
    
    def _balance_datasets(self):
        """Balance datasets by adjusting sampling weights."""
        # Calculate class distribution for each dataset
        class_distributions = []
        
        for dataset in self.datasets:
            # Try to get class distribution
            try:
                if hasattr(dataset, 'compute_statistics'):
                    stats = dataset.compute_statistics()
                    class_counts = stats.get('class_distribution', {}).get('counts', {})
                    
                    if class_counts:
                        # Convert to normalized distribution
                        total = sum(class_counts.values())
                        distribution = {cls: count/total for cls, count in class_counts.items()}
                        class_distributions.append(distribution)
                    else:
                        class_distributions.append(None)
                else:
                    class_distributions.append(None)
            except Exception as e:
                logger.warning(f"Failed to get class distribution for dataset: {e}")
                class_distributions.append(None)
        
        # Adjust weights based on class diversity
        new_weights = []
        
        for i, distribution in enumerate(class_distributions):
            if distribution is None:
                # No class information, use original weight
                new_weights.append(self.sampling_weights[i])
            else:
                # Weight inversely proportional to class imbalance
                # More balanced datasets get higher weight
                counts = list(distribution.values())
                if counts:
                    # Use entropy as measure of balance
                    from scipy.stats import entropy
                    ent = entropy(counts)
                    max_ent = math.log(len(counts)) if len(counts) > 0 else 0
                    
                    if max_ent > 0:
                        balance_score = ent / max_ent
                    else:
                        balance_score = 1.0
                    
                    # Adjust weight
                    new_weight = self.sampling_weights[i] * balance_score
                    new_weights.append(new_weight)
                else:
                    new_weights.append(self.sampling_weights[i])
        
        # Normalize new weights
        total = sum(new_weights)
        if total > 0:
            self.sampling_weights = [w/total for w in new_weights]
        
        logger.info(f"Balanced dataset weights: {self.sampling_weights}")
    
    def _load_dataset(self) -> None:
        """Load composite dataset (already handled by _build_dataset_index)."""
        # Samples are loaded on-demand from component datasets
        self.samples = [{}] * self.total_size  # Placeholder
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for composite samples."""
        # Union of required fields from all datasets
        required_fields = set()
        
        for dataset in self.datasets:
            if hasattr(dataset, '_get_required_fields'):
                required_fields.update(dataset._get_required_fields())
        
        return list(required_fields)
    
    def __len__(self) -> int:
        """Get total number of samples."""
        return self.total_size
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get sample by global index.
        
        Args:
            index: Global sample index
            
        Returns:
            Sample dictionary
        """
        if index < 0:
            index = self.total_size + index
        
        if index >= self.total_size:
            raise IndexError(f"Index {index} out of range for dataset with {self.total_size} samples")
        
        # Get dataset and local index
        dataset_idx, local_idx = self.global_to_local[index]
        
        # Get sample from dataset
        dataset = self.datasets[dataset_idx]
        sample = dataset[local_idx]
        
        # Apply dataset-specific transform if available
        transform = self.dataset_transforms[dataset_idx]
        if transform is not None:
            sample = transform(sample)
        
        # Add dataset metadata
        sample['dataset_index'] = dataset_idx
        sample['dataset_name'] = dataset.name
        sample['original_index'] = local_idx
        
        return sample
    
    def get_sample(self, index: int, apply_transform: bool = True) -> Dict[str, Any]:
        """
        Get sample without applying transforms.
        
        Args:
            index: Global sample index
            apply_transform: Whether to apply transforms
            
        Returns:
            Sample dictionary
        """
        # Get dataset and local index
        dataset_idx, local_idx = self.global_to_local[index]
        
        # Get sample from dataset
        dataset = self.datasets[dataset_idx]
        
        if hasattr(dataset, 'get_sample'):
            sample = dataset.get_sample(local_idx, apply_transform)
        else:
            # Fall back to regular getitem
            original_transform = dataset.transform
            original_target_transform = dataset.target_transform
            
            if not apply_transform:
                dataset.transform = None
                dataset.target_transform = None
            
            try:
                sample = dataset[local_idx]
            finally:
                if not apply_transform:
                    dataset.transform = original_transform
                    dataset.target_transform = original_target_transform
        
        # Add dataset metadata
        sample['dataset_index'] = dataset_idx
        sample['dataset_name'] = dataset.name
        sample['original_index'] = local_idx
        
        return sample
    
    def create_sampler(self, batch_size: int = 32) -> 'CompositeSampler':
        """
        Create a sampler for the composite dataset.
        
        Args:
            batch_size: Batch size
            
        Returns:
            CompositeSampler instance
        """
        return CompositeSampler(
            dataset=self,
            batch_size=batch_size,
            sampling_strategy=self.sampling_strategy,
            sampling_weights=self.sampling_weights,
            shuffle=self.config.shuffle
        )
    
    def get_dataset_statistics(self, dataset_idx: int) -> Dict[str, Any]:
        """
        Get statistics for a specific dataset.
        
        Args:
            dataset_idx: Dataset index
            
        Returns:
            Statistics dictionary
        """
        if dataset_idx < 0 or dataset_idx >= self.num_datasets:
            raise IndexError(f"Dataset index {dataset_idx} out of range")
        
        dataset = self.datasets[dataset_idx]
        
        if hasattr(dataset, 'compute_statistics'):
            return dataset.compute_statistics()
        else:
            # Basic statistics
            return {
                'name': dataset.name,
                'type': str(dataset.type.value) if hasattr(dataset, 'type') else 'unknown',
                'size': len(dataset),
                'phase': str(dataset.phase.value) if hasattr(dataset, 'phase') else 'unknown'
            }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all datasets."""
        stats = {
            'total_samples': self.total_size,
            'num_datasets': self.num_datasets,
            'datasets': []
        }
        
        for i, dataset in enumerate(self.datasets):
            dataset_stats = self.get_dataset_statistics(i)
            dataset_stats['index'] = i
            dataset_stats['weight'] = self.sampling_weights[i]
            dataset_stats['size_in_composite'] = self.dataset_sizes[i]
            
            stats['datasets'].append(dataset_stats)
        
        return stats
    
    def split_by_dataset(self) -> Dict[str, BaseDataset]:
        """
        Split composite dataset into individual datasets.
        
        Returns:
            Dictionary mapping dataset names to datasets
        """
        datasets_dict = {}
        
        for i, dataset in enumerate(self.datasets):
            dataset_name = f"{dataset.name}_from_composite"
            datasets_dict[dataset_name] = dataset
        
        return datasets_dict
    
    def filter_by_dataset(self, dataset_indices: List[int]) -> 'CompositeDataset':
        """
        Create a new composite dataset with only specified datasets.
        
        Args:
            dataset_indices: List of dataset indices to include
            
        Returns:
            New CompositeDataset instance
        """
        # Filter datasets
        filtered_datasets = [self.datasets[i] for i in dataset_indices]
        filtered_weights = [self.sampling_weights[i] for i in dataset_indices]
        
        # Create new config
        new_config = CompositeDatasetConfig(
            name=f"{self.name}_filtered",
            datasets=filtered_datasets,
            sampling_weights=filtered_weights,
            sampling_strategy=self.sampling_strategy,
            max_samples_per_dataset=self.max_samples_per_dataset,
            shuffle_datasets=self.shuffle_datasets,
            balance_datasets=self.balance_datasets
        )
        
        return CompositeDataset(new_config)
    
    def add_dataset(self, dataset: BaseDataset, weight: float = 1.0):
        """
        Add a dataset to the composite.
        
        Args:
            dataset: Dataset to add
            weight: Sampling weight for the dataset
        """
        self.datasets.append(dataset)
        self.sampling_weights.append(weight)
        self.num_datasets += 1
        
        # Rebuild index
        self._build_dataset_index()
        
        # Update total size
        self.total_size = len(self.global_to_local)
        
        logger.info(f"Added dataset {dataset.name} with weight {weight}")
    
    def remove_dataset(self, dataset_idx: int):
        """
        Remove a dataset from the composite.
        
        Args:
            dataset_idx: Index of dataset to remove
        """
        if dataset_idx < 0 or dataset_idx >= self.num_datasets:
            raise IndexError(f"Dataset index {dataset_idx} out of range")
        
        removed_name = self.datasets[dataset_idx].name
        
        # Remove dataset
        del self.datasets[dataset_idx]
        del self.sampling_weights[dataset_idx]
        self.num_datasets -= 1
        
        # Rebuild index
        self._build_dataset_index()
        
        # Update total size
        self.total_size = len(self.global_to_local)
        
        # Normalize weights
        total_weight = sum(self.sampling_weights)
        if total_weight > 0:
            self.sampling_weights = [w/total_weight for w in self.sampling_weights]
        
        logger.info(f"Removed dataset {removed_name}")
    
    def update_weights(self, new_weights: List[float]):
        """
        Update sampling weights.
        
        Args:
            new_weights: New sampling weights
        """
        if len(new_weights) != self.num_datasets:
            raise ValueError(f"new_weights length ({len(new_weights)}) "
                           f"must match number of datasets ({self.num_datasets})")
        
        # Normalize weights
        total = sum(new_weights)
        if total > 0:
            self.sampling_weights = [w/total for w in new_weights]
        
        logger.info(f"Updated sampling weights: {self.sampling_weights}")


class CompositeSampler:
    """
    Sampler for composite dataset with different sampling strategies.
    """
    
    def __init__(self, dataset: CompositeDataset, batch_size: int = 32,
                 sampling_strategy: str = "weighted",
                 sampling_weights: Optional[List[float]] = None,
                 shuffle: bool = True, seed: int = 42):
        """
        Initialize composite sampler.
        
        Args:
            dataset: Composite dataset
            batch_size: Batch size
            sampling_strategy: Sampling strategy
            sampling_weights: Weights for each dataset
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.sampling_weights = sampling_weights or dataset.sampling_weights
        self.shuffle = shuffle
        self.seed = seed
        
        self.rng = np.random.RandomState(seed)
        
        # Build dataset indices
        self.dataset_indices = []
        for i in range(dataset.num_datasets):
            indices = list(range(dataset.dataset_offsets[i], 
                               dataset.dataset_offsets[i] + dataset.dataset_sizes[i]))
            if shuffle:
                self.rng.shuffle(indices)
            self.dataset_indices.append(indices)
        
        # Initialize pointers for each dataset
        self.dataset_pointers = [0] * dataset.num_datasets
        
        logger.info(f"Initialized CompositeSampler with {sampling_strategy} strategy")
    
    def __iter__(self):
        """Create iterator over batches."""
        # Reset pointers
        self.dataset_pointers = [0] * self.dataset.num_datasets
        
        # Create batch indices
        batch_indices = []
        
        while True:
            # Get next sample based on strategy
            if self.sampling_strategy == "weighted":
                # Weighted random sampling
                dataset_idx = self.rng.choice(
                    self.dataset.num_datasets,
                    p=self.sampling_weights
                )
            elif self.sampling_strategy == "sequential":
                # Sequential sampling from each dataset
                dataset_idx = self._get_next_sequential_dataset()
            elif self.sampling_strategy == "round_robin":
                # Round-robin sampling
                dataset_idx = self._get_next_round_robin_dataset()
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
            
            # Check if dataset has more samples
            if self.dataset_pointers[dataset_idx] >= len(self.dataset_indices[dataset_idx]):
                # This dataset is exhausted
                # Check if all datasets are exhausted
                if all(ptr >= len(indices) for ptr, indices in zip(self.dataset_pointers, self.dataset_indices)):
                    break
                else:
                    # Try another dataset
                    continue
            
            # Get next sample from dataset
            sample_idx = self.dataset_indices[dataset_idx][self.dataset_pointers[dataset_idx]]
            batch_indices.append(sample_idx)
            
            # Move pointer
            self.dataset_pointers[dataset_idx] += 1
            
            # Check if batch is complete
            if len(batch_indices) >= self.batch_size:
                yield batch_indices
                batch_indices = []
        
        # Yield remaining samples
        if batch_indices:
            yield batch_indices
    
    def _get_next_sequential_dataset(self) -> int:
        """Get next dataset for sequential sampling."""
        # Find dataset with samples remaining
        for i in range(self.dataset.num_datasets):
            if self.dataset_pointers[i] < len(self.dataset_indices[i]):
                return i
        
        # All datasets exhausted, reset
        self.dataset_pointers = [0] * self.dataset.num_datasets
        return 0
    
    def _get_next_round_robin_dataset(self) -> int:
        """Get next dataset for round-robin sampling."""
        # Start from last dataset + 1
        for offset in range(self.dataset.num_datasets):
            dataset_idx = (self.current_dataset + offset + 1) % self.dataset.num_datasets
            
            if self.dataset_pointers[dataset_idx] < len(self.dataset_indices[dataset_idx]):
                self.current_dataset = dataset_idx
                return dataset_idx
        
        # All datasets exhausted, reset
        self.dataset_pointers = [0] * self.dataset.num_datasets
        self.current_dataset = 0
        return 0
    
    def __len__(self) -> int:
        """Get number of batches."""
        total_samples = self.dataset.total_size
        return (total_samples + self.batch_size - 1) // self.batch_size


def create_composite_dataset(datasets: List[BaseDataset], 
                           weights: Optional[List[float]] = None,
                           name: str = "composite",
                           **kwargs) -> CompositeDataset:
    """
    Create a composite dataset.
    
    Args:
        datasets: List of datasets to combine
        weights: Sampling weights for each dataset
        name: Dataset name
        **kwargs: Additional configuration
        
    Returns:
        CompositeDataset instance
    """
    config = CompositeDatasetConfig(
        name=name,
        datasets=datasets,
        sampling_weights=weights,
        **kwargs
    )
    
    return CompositeDataset(config)


def merge_datasets(datasets: List[BaseDataset], 
                  strategy: str = "concat",
                  **kwargs) -> BaseDataset:
    """
    Merge multiple datasets using different strategies.
    
    Args:
        datasets: List of datasets to merge
        strategy: Merge strategy ('concat', 'composite', 'union')
        **kwargs: Additional configuration
        
    Returns:
        Merged dataset
    """
    if strategy == "concat":
        # Simple concatenation
        return ConcatDataset(datasets)
    
    elif strategy == "composite":
        # Weighted composite
        return create_composite_dataset(datasets, **kwargs)
    
    elif strategy == "union":
        # Union of datasets (remove duplicates)
        # This would require more sophisticated implementation
        logger.warning("Union strategy not implemented, using composite")
        return create_composite_dataset(datasets, **kwargs)
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")