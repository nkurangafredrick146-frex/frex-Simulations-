"""
Base Dataset Class
Defines the interface and common functionality for all datasets in FrexTech AI.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import yaml

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

class DatasetPhase(Enum):
    """Dataset phases for training, validation, and testing."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    INFERENCE = "inference"

class DatasetType(Enum):
    """Types of datasets."""
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    MULTIMODAL = "multimodal"
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"
    AUDIO = "audio"

@dataclass
class DatasetConfig:
    """
    Configuration for a dataset.
    
    Attributes:
        name: Name of the dataset
        type: Type of dataset (image, video, multimodal, etc.)
        phase: Phase of dataset (train, validation, test)
        data_dir: Root directory containing data
        metadata_file: Path to metadata file (JSON, CSV, etc.)
        transform: Transformation to apply to samples
        target_transform: Transformation to apply to targets
        cache: Whether to cache samples in memory
        cache_dir: Directory for caching
        shuffle: Whether to shuffle the dataset
        seed: Random seed for reproducibility
        max_samples: Maximum number of samples to load (None for all)
        sample_rate: Rate at which to sample from dataset (1.0 for all)
        num_workers: Number of workers for data loading
        batch_size: Batch size for data loading
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch
        persistent_workers: Whether to persist workers
        prefetch_factor: Prefetch factor for data loading
        collate_fn: Custom collate function
    """
    name: str = "dataset"
    type: DatasetType = DatasetType.IMAGE
    phase: DatasetPhase = DatasetPhase.TRAIN
    data_dir: str = "./data"
    metadata_file: Optional[str] = None
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None
    cache: bool = False
    cache_dir: Optional[str] = None
    shuffle: bool = True
    seed: int = 42
    max_samples: Optional[int] = None
    sample_rate: float = 1.0
    num_workers: int = 4
    batch_size: int = 32
    pin_memory: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    collate_fn: Optional[Callable] = None
    
    # Dataset statistics
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    
    # Dataset splitting
    split_ratio: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    
    # Data augmentation
    augment: bool = False
    augmentation_config: Optional[Dict[str, Any]] = None
    
    # Filtering
    filter_conditions: Optional[Dict[str, Any]] = None
    filter_func: Optional[Callable] = None
    
    # Sampling weights
    sample_weights: Optional[List[float]] = None
    
    # Subset selection
    indices: Optional[List[int]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not isinstance(self.type, DatasetType):
            self.type = DatasetType(self.type)
        if not isinstance(self.phase, DatasetPhase):
            self.phase = DatasetPhase(self.phase)
        
        # Validate sample rate
        if not 0 < self.sample_rate <= 1:
            raise ValueError(f"sample_rate must be between 0 and 1, got {self.sample_rate}")
        
        # Validate split ratios
        if self.split_ratio:
            total = sum(self.split_ratio.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DatasetConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DatasetConfig':
        """Create configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'DatasetConfig':
        """Create configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def schema() -> Dict[str, Any]:
        """Get JSON schema for validation."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": [t.value for t in DatasetType]},
                "phase": {"type": "string", "enum": [p.value for p in DatasetPhase]},
                "data_dir": {"type": "string"},
                "metadata_file": {"type": ["string", "null"]},
                "cache": {"type": "boolean"},
                "cache_dir": {"type": ["string", "null"]},
                "shuffle": {"type": "boolean"},
                "seed": {"type": "integer", "minimum": 0},
                "max_samples": {"type": ["integer", "null"], "minimum": 1},
                "sample_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "num_workers": {"type": "integer", "minimum": 0},
                "batch_size": {"type": "integer", "minimum": 1},
                "pin_memory": {"type": "boolean"},
                "drop_last": {"type": "boolean"},
                "persistent_workers": {"type": "boolean"},
                "prefetch_factor": {"type": "integer", "minimum": 1},
                "split_ratio": {
                    "type": "object",
                    "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "augment": {"type": "boolean"},
                "augmentation_config": {"type": ["object", "null"]},
                "filter_conditions": {"type": ["object", "null"]},
                "sample_weights": {"type": ["array", "null"], "items": {"type": "number"}},
                "indices": {"type": ["array", "null"], "items": {"type": "integer"}}
            },
            "required": ["name", "type", "data_dir"]
        }


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all datasets.
    
    This class provides common functionality for data loading, caching,
    transformations, and metadata handling.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize dataset.
        
        Args:
            config: Dataset configuration
        """
        super().__init__()
        
        self.config = config
        self.name = config.name
        self.type = config.type
        self.phase = config.phase
        self.data_dir = Path(config.data_dir).expanduser().resolve()
        self.metadata_file = Path(config.metadata_file) if config.metadata_file else None
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = config.cache
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        self._cached_samples = {}
        
        if self.cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        self.seed = config.seed
        self.rng = np.random.RandomState(self.seed)
        
        # Transformations
        self.transform = config.transform
        self.target_transform = config.target_transform
        
        # Data structures
        self.samples = []
        self.metadata = {}
        self.class_names = config.class_names or []
        self.num_classes = config.num_classes or 0
        
        # Statistics
        self.mean = config.mean
        self.std = config.std
        self._statistics = {}
        
        # Filtering
        self.filter_conditions = config.filter_conditions
        self.filter_func = config.filter_func
        
        # Sampling weights
        self.sample_weights = config.sample_weights
        
        # Subset indices
        self.indices = config.indices
        
        # Load dataset
        self._load_dataset()
        
        # Apply filters
        if self.filter_conditions or self.filter_func:
            self._apply_filters()
        
        # Limit samples if specified
        if config.max_samples is not None and len(self.samples) > config.max_samples:
            self.samples = self.samples[:config.max_samples]
        
        # Apply sampling rate
        if config.sample_rate < 1.0:
            num_samples = int(len(self.samples) * config.sample_rate)
            indices = self.rng.choice(len(self.samples), num_samples, replace=False)
            self.samples = [self.samples[i] for i in indices]
        
        # Validate dataset
        self._validate_dataset()
        
        logger.info(f"Loaded {self.__class__.__name__} with {len(self)} samples")
    
    @abstractmethod
    def _load_dataset(self) -> None:
        """
        Load dataset samples and metadata.
        
        This method should populate self.samples and self.metadata.
        """
        pass
    
    def _apply_filters(self) -> None:
        """Apply filters to samples."""
        if not self.samples:
            return
        
        filtered_samples = []
        
        for sample in self.samples:
            include_sample = True
            
            # Apply filter conditions
            if self.filter_conditions:
                for key, value in self.filter_conditions.items():
                    if key in sample:
                        if isinstance(value, list):
                            if sample[key] not in value:
                                include_sample = False
                                break
                        elif sample[key] != value:
                            include_sample = False
                            break
            
            # Apply filter function
            if include_sample and self.filter_func:
                include_sample = self.filter_func(sample)
            
            if include_sample:
                filtered_samples.append(sample)
        
        self.samples = filtered_samples
        logger.info(f"Filtered dataset: {len(self.samples)} samples remaining")
    
    def _validate_dataset(self) -> None:
        """Validate dataset integrity."""
        if not self.samples:
            raise ValueError(f"Dataset {self.name} is empty")
        
        # Check sample structure
        first_sample = self.samples[0]
        required_fields = self._get_required_fields()
        
        for field in required_fields:
            if field not in first_sample:
                logger.warning(f"Sample missing field {field}")
        
        logger.info(f"Dataset validation passed: {len(self.samples)} samples")
    
    @abstractmethod
    def _get_required_fields(self) -> List[str]:
        """Get list of required fields for samples."""
        return []
    
    def __len__(self) -> int:
        """Get number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample by index.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        # Handle negative indexing
        if index < 0:
            index = len(self) + index
        
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} samples")
        
        # Check cache
        cache_key = f"{self.name}_{index}"
        if self.cache and cache_key in self._cached_samples:
            return self._cached_samples[cache_key]
        
        # Get sample
        sample = self._load_sample(index)
        
        # Apply transformations
        sample = self._apply_transformations(sample)
        
        # Cache if enabled
        if self.cache:
            self._cached_samples[cache_key] = sample
        
        return sample
    
    def _load_sample(self, index: int) -> Dict[str, Any]:
        """
        Load a single sample.
        
        Args:
            index: Sample index
            
        Returns:
            Sample dictionary
        """
        sample_info = self.samples[index]
        
        # This method should be overridden by subclasses
        # to load actual data (images, videos, etc.)
        return sample_info.copy()
    
    def _apply_transformations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transformations to sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Transformed sample
        """
        # Apply main transformation
        if self.transform is not None:
            sample = self.transform(sample)
        
        # Apply target transformation if specified
        if self.target_transform is not None and 'target' in sample:
            sample['target'] = self.target_transform(sample['target'])
        
        return sample
    
    def get_sample(self, index: int, apply_transform: bool = True) -> Dict[str, Any]:
        """
        Get sample without applying transformations.
        
        Args:
            index: Sample index
            apply_transform: Whether to apply transformations
            
        Returns:
            Raw sample data
        """
        original_transform = self.transform
        original_target_transform = self.target_transform
        
        if not apply_transform:
            self.transform = None
            self.target_transform = None
        
        try:
            sample = self[index]
        finally:
            if not apply_transform:
                self.transform = original_transform
                self.target_transform = original_target_transform
        
        return sample
    
    def get_batch(self, indices: List[int]) -> Dict[str, Any]:
        """
        Get a batch of samples.
        
        Args:
            indices: List of sample indices
            
        Returns:
            Batch dictionary
        """
        samples = [self[i] for i in indices]
        
        # Custom collate function if provided
        if self.config.collate_fn is not None:
            return self.config.collate_fn(samples)
        
        # Default collate: stack tensors, keep lists
        batch = {}
        for key in samples[0].keys():
            values = [sample[key] for sample in samples]
            
            # Stack tensors
            if isinstance(values[0], torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(values[0], np.ndarray):
                batch[key] = np.stack(values)
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values)
            else:
                batch[key] = values
        
        return batch
    
    def create_data_loader(self, **kwargs) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            **kwargs: Override DataLoader parameters
            
        Returns:
            DataLoader instance
        """
        loader_kwargs = {
            "batch_size": self.config.batch_size,
            "shuffle": self.config.shuffle and self.phase == DatasetPhase.TRAIN,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "drop_last": self.config.drop_last,
            "persistent_workers": self.config.persistent_workers,
            "prefetch_factor": self.config.prefetch_factor,
        }
        
        # Override with provided kwargs
        loader_kwargs.update(kwargs)
        
        # Use custom collate function if provided
        if self.config.collate_fn is not None:
            loader_kwargs["collate_fn"] = self.config.collate_fn
        
        return DataLoader(self, **loader_kwargs)
    
    def split(self, ratios: Dict[str, float], seed: Optional[int] = None) -> Dict[str, 'BaseDataset']:
        """
        Split dataset into subsets.
        
        Args:
            ratios: Dictionary mapping subset names to ratios
            seed: Random seed for splitting
            
        Returns:
            Dictionary of dataset subsets
        """
        from .splitter import DatasetSplitter
        
        splitter = DatasetSplitter(
            dataset=self,
            ratios=ratios,
            seed=seed or self.seed,
            stratify_by=None  # Can be overridden by subclasses
        )
        
        return splitter.split()
    
    def compute_statistics(self, force: bool = False) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Args:
            force: Whether to recompute statistics
            
        Returns:
            Dictionary of statistics
        """
        if self._statistics and not force:
            return self._statistics
        
        from .dataset_stats import DatasetStatistics
        stats_calculator = DatasetStatistics(self)
        self._statistics = stats_calculator.compute()
        
        # Update configuration
        if 'mean' in self._statistics:
            self.mean = self._statistics['mean']
        if 'std' in self._statistics:
            self.std = self._statistics['std']
        if 'num_classes' in self._statistics:
            self.num_classes = self._statistics['num_classes']
        
        return self._statistics
    
    def save_statistics(self, output_path: str) -> None:
        """Save dataset statistics to file."""
        stats = self.compute_statistics()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        elif output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"Saved statistics to {output_path}")
    
    def get_sample_indices_by_class(self, class_label: Any) -> List[int]:
        """
        Get indices of samples with given class label.
        
        Args:
            class_label: Class label to filter by
            
        Returns:
            List of indices
        """
        indices = []
        for i, sample in enumerate(self.samples):
            if 'label' in sample and sample['label'] == class_label:
                indices.append(i)
        return indices
    
    def balance_classes(self, max_samples_per_class: Optional[int] = None) -> None:
        """
        Balance dataset by class.
        
        Args:
            max_samples_per_class: Maximum samples per class
        """
        if 'label' not in self.samples[0]:
            logger.warning("Dataset does not have labels, skipping balancing")
            return
        
        # Group samples by class
        class_indices = {}
        for i, sample in enumerate(self.samples):
            label = sample['label']
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(i)
        
        # Determine target samples per class
        if max_samples_per_class is None:
            # Use minimum class size
            min_size = min(len(indices) for indices in class_indices.values())
            target_samples = min_size
        else:
            target_samples = max_samples_per_class
        
        # Sample from each class
        balanced_indices = []
        for label, indices in class_indices.items():
            if len(indices) > target_samples:
                # Randomly sample
                sampled = self.rng.choice(indices, target_samples, replace=False)
                balanced_indices.extend(sampled.tolist())
            else:
                # Use all samples, optionally with repetition
                balanced_indices.extend(indices)
        
        # Update samples
        self.samples = [self.samples[i] for i in balanced_indices]
        logger.info(f"Balanced dataset: {len(self.samples)} samples, {len(class_indices)} classes")
    
    def save_dataset_info(self, output_path: str) -> None:
        """Save dataset information to file."""
        info = {
            "name": self.name,
            "type": self.type.value,
            "phase": self.phase.value,
            "num_samples": len(self),
            "samples": self.samples[:10],  # First 10 samples as example
            "config": self.config.to_dict(),
            "statistics": self._statistics,
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        logger.info(f"Saved dataset info to {output_path}")
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        return (f"{self.__class__.__name__}(name={self.name}, "
                f"type={self.type.value}, phase={self.phase.value}, "
                f"samples={len(self)})")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset."""
        for i in range(len(self)):
            yield self[i]