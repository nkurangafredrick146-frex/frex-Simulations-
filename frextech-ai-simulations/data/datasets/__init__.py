"""
Dataset module for FrexTech AI Simulations.

This module provides dataset classes for various data types including images,
videos, 3D models, and multimodal combinations.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base_dataset import BaseDataset
from .video_dataset import VideoDataset
from .image_dataset import ImageDataset
from .multimodal_dataset import MultimodalDataset

__version__ = "1.0.0"
__all__ = [
    "BaseDataset",
    "VideoDataset", 
    "ImageDataset",
    "MultimodalDataset",
    "get_dataset",
    "list_datasets",
    "register_dataset",
]

# Dataset registry
_DATASET_REGISTRY = {
    "base": BaseDataset,
    "video": VideoDataset,
    "image": ImageDataset,
    "multimodal": MultimodalDataset,
}

logger = logging.getLogger(__name__)


def get_dataset(
    dataset_type: str,
    data_dir: Union[str, Path],
    split: str = "train",
    **kwargs
) -> BaseDataset:
    """
    Factory function to get dataset instance.
    
    Args:
        dataset_type: Type of dataset ("video", "image", "multimodal")
        data_dir: Directory containing the dataset
        split: Dataset split ("train", "val", "test")
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Dataset instance
        
    Raises:
        ValueError: If dataset type is not registered
        FileNotFoundError: If data directory doesn't exist
    """
    if dataset_type not in _DATASET_REGISTRY:
        available = list(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available types: {available}"
        )
    
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    dataset_class = _DATASET_REGISTRY[dataset_type]
    
    logger.info(
        f"Loading {dataset_type} dataset from {data_dir} "
        f"(split: {split})"
    )
    
    return dataset_class(data_dir=data_dir, split=split, **kwargs)


def list_datasets() -> List[str]:
    """
    List all registered dataset types.
    
    Returns:
        List of dataset type names
    """
    return list(_DATASET_REGISTRY.keys())


def register_dataset(
    name: str,
    dataset_class: type,
    overwrite: bool = False
) -> None:
    """
    Register a new dataset class.
    
    Args:
        name: Name of the dataset type
        dataset_class: Dataset class (must inherit from BaseDataset)
        overwrite: Whether to overwrite existing registration
        
    Raises:
        ValueError: If name already registered and overwrite is False
        TypeError: If dataset_class is not a subclass of BaseDataset
    """
    if name in _DATASET_REGISTRY and not overwrite:
        raise ValueError(f"Dataset '{name}' already registered")
    
    if not issubclass(dataset_class, BaseDataset):
        raise TypeError(
            f"Dataset class must inherit from BaseDataset, "
            f"got {type(dataset_class)}"
        )
    
    _DATASET_REGISTRY[name] = dataset_class
    logger.info(f"Registered dataset: {name} -> {dataset_class.__name__}")


def load_dataset_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dataset configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_dataset_from_config(
    config_path: Union[str, Path]
) -> BaseDataset:
    """
    Create dataset from configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dataset instance
    """
    config = load_dataset_config(config_path)
    
    # Extract dataset parameters
    dataset_type = config.get("type", "multimodal")
    data_dir = config.get("data_dir", "./data")
    split = config.get("split", "train")
    
    # Get dataset-specific parameters
    dataset_params = config.get("params", {})
    
    return get_dataset(
        dataset_type=dataset_type,
        data_dir=data_dir,
        split=split,
        **dataset_params
    )


def get_dataset_stats(
    dataset: BaseDataset,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        dataset: Dataset instance
        sample_size: Number of samples to use for statistics.
                    If None, use entire dataset.
                    
    Returns:
        Dictionary with dataset statistics
    """
    import numpy as np
    from tqdm import tqdm
    
    stats = {
        "total_samples": len(dataset),
        "sample_size": sample_size or len(dataset),
        "modalities": dataset.modalities,
    }
    
    # Sample indices
    if sample_size and sample_size < len(dataset):
        indices = np.random.choice(len(dataset), sample_size, replace=False)
    else:
        indices = range(len(dataset))
    
    # Collect statistics
    image_shapes = []
    video_lengths = []
    text_lengths = []
    
    for idx in tqdm(indices, desc="Computing dataset stats"):
        sample = dataset[idx]
        
        if "image" in sample and sample["image"] is not None:
            if hasattr(sample["image"], "shape"):
                image_shapes.append(sample["image"].shape)
        
        if "video" in sample and sample["video"] is not None:
            if hasattr(sample["video"], "shape"):
                video_lengths.append(sample["video"].shape[0])
        
        if "text" in sample and sample["text"] is not None:
            text_lengths.append(len(sample["text"]))
    
    # Compute statistics
    if image_shapes:
        shapes = np.array(image_shapes)
        stats["image"] = {
            "count": len(image_shapes),
            "height_mean": float(np.mean(shapes[:, 1])),
            "height_std": float(np.std(shapes[:, 1])),
            "width_mean": float(np.mean(shapes[:, 2])),
            "width_std": float(np.std(shapes[:, 2])),
            "channels_mean": float(np.mean(shapes[:, 0])),
            "channels_std": float(np.std(shapes[:, 0])),
        }
    
    if video_lengths:
        lengths = np.array(video_lengths)
        stats["video"] = {
            "count": len(video_lengths),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "length_min": int(np.min(lengths)),
            "length_max": int(np.max(lengths)),
        }
    
    if text_lengths:
        lengths = np.array(text_lengths)
        stats["text"] = {
            "count": len(text_lengths),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "length_min": int(np.min(lengths)),
            "length_max": int(np.max(lengths)),
        }
    
    return stats


def split_dataset(
    dataset: BaseDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[BaseDataset, BaseDataset, BaseDataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle the dataset before splitting
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    import numpy as np
    
    # Set random seed
    np.random.seed(seed)
    
    # Get indices
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    
    # Calculate split points
    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    logger.info(
        f"Split dataset: "
        f"train={len(train_dataset)} ({train_ratio:.1%}), "
        f"val={len(val_dataset)} ({val_ratio:.1%}), "
        f"test={len(test_dataset)} ({test_ratio:.1%})"
    )
    
    return train_dataset, val_dataset, test_dataset


def visualize_dataset_samples(
    dataset: BaseDataset,
    num_samples: int = 5,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Visualize samples from dataset.
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to visualize
        save_path: Path to save visualization. If None, display interactively.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure
    fig, axes = plt.subplots(
        num_samples,
        len(dataset.modalities),
        figsize=(4 * len(dataset.modalities), 4 * num_samples)
    )
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Get random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        for j, modality in enumerate(dataset.modalities):
            ax = axes[i, j]
            
            if modality == "image" and sample.get("image") is not None:
                img = sample["image"]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                
                # Handle different image formats
                if img.shape[0] in [1, 3, 4]:  # CHW format
                    img = np.transpose(img, (1, 2, 0))
                
                # Normalize for display
                if img.dtype == np.float32:
                    img = np.clip(img, 0, 1)
                elif img.dtype in [np.uint16, np.int16]:
                    img = img.astype(np.float32) / 65535.0
                
                ax.imshow(img)
                ax.set_title(f"Image {idx}")
                ax.axis("off")
            
            elif modality == "video" and sample.get("video") is not None:
                video = sample["video"]
                if hasattr(video, "numpy"):
                    video = video.numpy()
                
                # Show first frame
                frame = video[0]
                if frame.shape[0] in [1, 3, 4]:  # CHW format
                    frame = np.transpose(frame, (1, 2, 0))
                
                ax.imshow(frame)
                ax.set_title(f"Video {idx} (frame 0)")
                ax.axis("off")
            
            elif modality == "text" and sample.get("text") is not None:
                text = sample["text"]
                if isinstance(text, str):
                    display_text = text[:100] + "..." if len(text) > 100 else text
                else:
                    display_text = str(text)[:100] + "..."
                
                ax.text(0.5, 0.5, display_text,
                       ha='center', va='center',
                       wrap=True, fontsize=8)
                ax.set_title(f"Text {idx}")
                ax.axis("off")
            
            else:
                ax.text(0.5, 0.5, f"No {modality}",
                       ha='center', va='center')
                ax.set_title(f"{modality.capitalize()} {idx}")
                ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[callable] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> Any:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (for GPU training)
        drop_last: Whether to drop last incomplete batch
        collate_fn: Custom collate function
        prefetch_factor: Number of batches to prefetch
        persistent_workers: Whether to keep workers alive
        
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    
    if collate_fn is None:
        # Use default collate function
        from torch.utils.data._utils.collate import default_collate
        collate_fn = default_collate
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )


def check_dataset_integrity(
    dataset: BaseDataset,
    num_samples: int = 100
) -> Dict[str, Any]:
    """
    Check dataset integrity by sampling and validating.
    
    Args:
        dataset: Dataset to check
        num_samples: Number of samples to check
        
    Returns:
        Dictionary with integrity check results
    """
    import numpy as np
    from tqdm import tqdm
    
    results = {
        "total_samples": len(dataset),
        "checked_samples": min(num_samples, len(dataset)),
        "errors": [],
        "warnings": [],
        "valid_samples": 0,
        "invalid_samples": 0,
    }
    
    # Sample indices
    indices = np.random.choice(
        len(dataset),
        min(num_samples, len(dataset)),
        replace=False
    )
    
    for idx in tqdm(indices, desc="Checking dataset integrity"):
        try:
            sample = dataset[idx]
            
            # Check each modality
            for modality in dataset.modalities:
                if modality in sample:
                    data = sample[modality]
                    
                    # Check for None
                    if data is None:
                        results["warnings"].append(
                            f"Sample {idx}: {modality} is None"
                        )
                        continue
                    
                    # Check for NaN/Inf
                    if hasattr(data, "numpy"):
                        data_np = data.numpy()
                        if np.any(np.isnan(data_np)):
                            results["errors"].append(
                                f"Sample {idx}: {modality} contains NaN"
                            )
                        if np.any(np.isinf(data_np)):
                            results["errors"].append(
                                f"Sample {idx}: {modality} contains Inf"
                            )
                    
                    # Check shape/dimensions
                    if hasattr(data, "shape"):
                        if len(data.shape) == 0:
                            results["warnings"].append(
                                f"Sample {idx}: {modality} has empty shape"
                            )
            
            results["valid_samples"] += 1
            
        except Exception as e:
            results["invalid_samples"] += 1
            results["errors"].append(f"Sample {idx}: {str(e)}")
    
    # Summary
    results["valid_percentage"] = (
        results["valid_samples"] / results["checked_samples"] * 100
    )
    
    logger.info(f"Dataset integrity check: {results['valid_percentage']:.1f}% valid")
    
    if results["errors"]:
        logger.warning(f"Found {len(results['errors'])} errors")
        for error in results["errors"][:5]:  # Show first 5 errors
            logger.warning(f"  - {error}")
    
    if results["warnings"]:
        logger.info(f"Found {len(results['warnings'])} warnings")
        for warning in results["warnings"][:5]:  # Show first 5 warnings
            logger.info(f"  - {warning}")
    
    return results


# Export functions
__all__.extend([
    "get_dataset_stats",
    "split_dataset",
    "visualize_dataset_samples",
    "create_dataloader",
    "check_dataset_integrity",
    "load_dataset_config",
    "create_dataset_from_config",
])