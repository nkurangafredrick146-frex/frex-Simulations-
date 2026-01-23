"""
Data Loader Module
Utilities for creating data loaders with advanced features.
"""

import os
import math
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
from torch.utils.data._utils.collate import default_collate
import torch.distributed as dist

from .base_dataset import BaseDataset, DatasetPhase
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for batches with mixed data types.
    
    Args:
        batch: List of samples
        
    Returns:
        Collated batch
    """
    if not batch:
        return {}
    
    # Get all keys from first sample
    keys = batch[0].keys()
    
    collated_batch = {}
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        # Handle None values
        if all(v is None for v in values):
            collated_batch[key] = None
            continue
        
        # Filter out None values for stacking
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            collated_batch[key] = None
            continue
        
        # Check data type of first valid value
        first_value = valid_values[0]
        
        if isinstance(first_value, torch.Tensor):
            # Stack tensors
            try:
                # Handle tensors with different shapes
                if all(v.shape == first_value.shape for v in valid_values):
                    collated_batch[key] = torch.stack(valid_values)
                else:
                    # Pad tensors to same size
                    collated_batch[key] = pad_tensors(valid_values)
            except Exception as e:
                logger.warning(f"Failed to stack tensors for key {key}: {e}")
                collated_batch[key] = valid_values
        
        elif isinstance(first_value, np.ndarray):
            # Stack numpy arrays
            try:
                if all(v.shape == first_value.shape for v in valid_values):
                    collated_batch[key] = np.stack(valid_values)
                else:
                    # Convert to list
                    collated_batch[key] = valid_values
            except Exception as e:
                logger.warning(f"Failed to stack arrays for key {key}: {e}")
                collated_batch[key] = valid_values
        
        elif isinstance(first_value, (int, float, bool)):
            # Convert to tensor
            collated_batch[key] = torch.tensor(values)
        
        elif isinstance(first_value, str):
            # Keep as list
            collated_batch[key] = values
        
        elif isinstance(first_value, dict):
            # Recursively collate dictionaries
            collated_batch[key] = collate_fn(values)
        
        elif isinstance(first_value, list):
            # Keep as list of lists
            collated_batch[key] = values
        
        else:
            # Keep as list for other types
            collated_batch[key] = values
    
    return collated_batch

def pad_tensors(tensors: List[torch.Tensor], padding_value: float = 0.0) -> torch.Tensor:
    """
    Pad tensors to same size.
    
    Args:
        tensors: List of tensors
        padding_value: Value for padding
        
    Returns:
        Padded tensor
    """
    # Get max dimensions
    max_dims = []
    for tensor in tensors:
        for i, dim in enumerate(tensor.shape):
            if i >= len(max_dims):
                max_dims.append(dim)
            else:
                max_dims[i] = max(max_dims[i], dim)
    
    # Pad tensors
    padded_tensors = []
    for tensor in tensors:
        padding = []
        for i, (dim, max_dim) in enumerate(zip(tensor.shape, max_dims)):
            if dim < max_dim:
                padding.append((0, max_dim - dim))
            else:
                padding.append((0, 0))
        
        # Reverse padding for F.pad (starts from last dimension)
        padding = tuple(p for pair in reversed(padding) for p in pair)
        
        if padding != (0,) * len(padding):
            padded = torch.nn.functional.pad(tensor, padding, value=padding_value)
        else:
            padded = tensor
        
        padded_tensors.append(padded)
    
    return torch.stack(padded_tensors)

def create_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    collate_fn: Optional[Callable] = None,
    sampler=None,
    batch_sampler=None,
    timeout: int = 0,
    worker_init_fn: Optional[Callable] = None,
    multiprocessing_context=None,
    generator=None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults and error handling.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True for training, False otherwise)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch
        persistent_workers: Whether to persist workers
        prefetch_factor: Prefetch factor
        collate_fn: Custom collate function
        sampler: Custom sampler
        batch_sampler: Custom batch sampler
        timeout: Timeout for worker processes
        worker_init_fn: Worker initialization function
        multiprocessing_context: Multiprocessing context
        generator: Random generator
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    # Determine shuffle if not specified
    if shuffle is None:
        if hasattr(dataset, 'phase'):
            shuffle = (dataset.phase == DatasetPhase.TRAIN)
        else:
            shuffle = True
    
    # Set default collate function
    if collate_fn is None:
        collate_fn = globals()['collate_fn']
    
    # Validate parameters
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    
    if prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
    
    # Adjust num_workers based on dataset size
    if num_workers > 0 and len(dataset) < batch_size * num_workers:
        num_workers = max(1, len(dataset) // batch_size)
        logger.warning(f"Reduced num_workers to {num_workers} due to small dataset size")
    
    # Create loader
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        sampler=sampler,
        batch_sampler=batch_sampler,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
        generator=generator,
        **kwargs
    )
    
    logger.info(f"Created DataLoader: batch_size={batch_size}, "
               f"num_workers={num_workers}, shuffle={shuffle}, "
               f"samples={len(dataset)}")
    
    return loader

def create_distributed_sampler(
    dataset: Dataset,
    num_replicas: Optional[int] = None,
    rank: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = False
) -> DistributedSampler:
    """
    Create a distributed sampler for multi-GPU training.
    
    Args:
        dataset: Dataset instance
        num_replicas: Number of processes
        rank: Rank of current process
        shuffle: Whether to shuffle
        seed: Random seed
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        DistributedSampler instance
    """
    if num_replicas is None:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        num_replicas = dist.get_world_size()
    
    if rank is None:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available")
        rank = dist.get_rank()
    
    sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last
    )
    
    logger.info(f"Created DistributedSampler: num_replicas={num_replicas}, "
               f"rank={rank}, shuffle={shuffle}, drop_last={drop_last}")
    
    return sampler

def create_distributed_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a distributed DataLoader for multi-GPU training.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size per GPU
        num_workers: Number of worker processes per GPU
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    
    # Create distributed sampler
    sampler = create_distributed_sampler(
        dataset=dataset,
        shuffle=True,
        drop_last=drop_last
    )
    
    # Adjust batch size for each replica
    effective_batch_size = batch_size
    
    # Create data loader
    loader = create_data_loader(
        dataset=dataset,
        batch_size=effective_batch_size,
        shuffle=False,  # Sampler handles shuffling
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        **kwargs
    )
    
    return loader

def create_infinite_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    **kwargs
) -> Iterator[Dict[str, Any]]:
    """
    Create an infinite data loader that cycles through dataset.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        **kwargs: Additional arguments for DataLoader
        
    Yields:
        Batches indefinitely
    """
    loader = create_data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    
    while True:
        for batch in loader:
            yield batch

def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move batch to specified device.
    
    Args:
        batch: Batch dictionary
        device: Target device
        
    Returns:
        Batch on target device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    
    if isinstance(batch, dict):
        return {key: batch_to_device(value, device) for key, value in batch.items()}
    
    if isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(item, device) for item in batch)
    
    return batch

def split_batch(batch: Dict[str, Any], split_size: int) -> List[Dict[str, Any]]:
    """
    Split a batch into smaller batches.
    
    Args:
        batch: Batch dictionary
        split_size: Size of each split
        
    Returns:
        List of smaller batches
    """
    if not batch:
        return []
    
    # Get batch size from first tensor
    batch_size = None
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            batch_size = value.size(0)
            break
    
    if batch_size is None:
        return [batch]
    
    # Calculate number of splits
    num_splits = math.ceil(batch_size / split_size)
    
    splits = []
    for i in range(num_splits):
        start = i * split_size
        end = min((i + 1) * split_size, batch_size)
        
        split_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                split_batch[key] = value[start:end]
            elif isinstance(value, (list, tuple)):
                split_batch[key] = value[start:end]
            elif isinstance(value, dict):
                # Recursively split dictionaries
                split_batch[key] = split_batch(value, split_size)[0]
            else:
                split_batch[key] = value
        
        splits.append(split_batch)
    
    return splits

def merge_batches(batches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple batches into one.
    
    Args:
        batches: List of batches
        
    Returns:
        Merged batch
    """
    if not batches:
        return {}
    
    merged = {}
    
    # Get all keys from first batch
    keys = batches[0].keys()
    
    for key in keys:
        values = []
        for batch in batches:
            if key in batch:
                values.append(batch[key])
        
        if not values:
            continue
        
        # Merge based on type
        first_value = values[0]
        
        if isinstance(first_value, torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        
        elif isinstance(first_value, np.ndarray):
            merged[key] = np.concatenate(values, axis=0)
        
        elif isinstance(first_value, (list, tuple)):
            # Flatten lists
            flat_list = []
            for value in values:
                flat_list.extend(value)
            merged[key] = flat_list
        
        elif isinstance(first_value, dict):
            # Recursively merge dictionaries
            merged[key] = merge_batches(values)
        
        else:
            # Keep as list
            merged[key] = values
    
    return merged

def create_balanced_sampler(
    dataset: Dataset,
    labels: List[Any],
    replacement: bool = True,
    num_samples: Optional[int] = None
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a balanced sampler for imbalanced datasets.
    
    Args:
        dataset: Dataset instance
        labels: List of labels for each sample
        replacement: Whether to sample with replacement
        num_samples: Number of samples to draw
        
    Returns:
        WeightedRandomSampler
    """
    # Count samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Calculate weights
    weights = 1.0 / counts
    sample_weights = weights[np.searchsorted(unique_labels, labels)]
    
    # Convert to tensor
    sample_weights = torch.from_numpy(sample_weights).float()
    
    # Set num_samples if not provided
    if num_samples is None:
        num_samples = len(dataset)
    
    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement
    )
    
    logger.info(f"Created balanced sampler: {len(unique_labels)} classes, "
               f"weights={weights.tolist()}")
    
    return sampler

def prefetch_data_loader(
    loader: DataLoader,
    device: torch.device,
    prefetch_steps: int = 2
) -> Iterator[Dict[str, Any]]:
    """
    Prefetch data loader batches to device.
    
    Args:
        loader: DataLoader instance
        device: Target device
        prefetch_steps: Number of batches to prefetch
        
    Yields:
        Prefetched batches
    """
    import queue
    import threading
    
    def _worker(loader_iter, output_queue, device):
        try:
            for batch in loader_iter:
                batch = batch_to_device(batch, device)
                output_queue.put(batch)
        except Exception as e:
            output_queue.put(e)
        finally:
            output_queue.put(None)  # Sentinel
    
    # Create iterator and queue
    loader_iter = iter(loader)
    output_queue = queue.Queue(maxsize=prefetch_steps)
    
    # Start worker thread
    worker_thread = threading.Thread(
        target=_worker,
        args=(loader_iter, output_queue, device),
        daemon=True
    )
    worker_thread.start()
    
    # Yield prefetched batches
    while True:
        item = output_queue.get()
        
        if item is None:
            break
        
        if isinstance(item, Exception):
            raise item
        
        yield item

def get_dataloader_stats(loader: DataLoader) -> Dict[str, Any]:
    """
    Get statistics about a DataLoader.
    
    Args:
        loader: DataLoader instance
        
    Returns:
        Statistics dictionary
    """
    stats = {
        'batch_size': loader.batch_size,
        'num_batches': len(loader),
        'num_workers': loader.num_workers,
        'pin_memory': loader.pin_memory,
        'drop_last': loader.drop_last,
        'prefetch_factor': loader.prefetch_factor,
        'persistent_workers': loader.persistent_workers,
    }
    
    # Get batch shape info
    try:
        first_batch = next(iter(loader))
        stats['batch_keys'] = list(first_batch.keys())
        
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                stats[f'{key}_shape'] = value.shape
                stats[f'{key}_dtype'] = str(value.dtype)
            elif isinstance(value, np.ndarray):
                stats[f'{key}_shape'] = value.shape
                stats[f'{key}_dtype'] = str(value.dtype)
    except Exception as e:
        logger.warning(f"Could not get batch info: {e}")
    
    return stats

def profile_dataloader(loader: DataLoader, num_batches: int = 10) -> Dict[str, float]:
    """
    Profile DataLoader performance.
    
    Args:
        loader: DataLoader instance
        num_batches: Number of batches to profile
        
    Returns:
        Performance metrics
    """
    import time
    
    metrics = {
        'batch_times': [],
        'data_transfer_times': [],
        'batch_sizes': []
    }
    
    start_time = time.time()
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Measure data transfer if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            transfer_start = time.time()
            batch = batch_to_device(batch, torch.device('cuda'))
            torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start
            metrics['data_transfer_times'].append(transfer_time)
        
        batch_time = time.time() - batch_start
        metrics['batch_times'].append(batch_time)
        
        # Get batch size
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                metrics['batch_sizes'].append(value.size(0))
                break
    
    total_time = time.time() - start_time
    
    # Compute statistics
    if metrics['batch_times']:
        metrics['avg_batch_time'] = np.mean(metrics['batch_times'])
        metrics['std_batch_time'] = np.std(metrics['batch_times'])
        metrics['min_batch_time'] = np.min(metrics['batch_times'])
        metrics['max_batch_time'] = np.max(metrics['batch_times'])
        
        # Throughput
        total_samples = sum(metrics['batch_sizes'])
        metrics['samples_per_second'] = total_samples / total_time
        metrics['batches_per_second'] = len(metrics['batch_times']) / total_time
    
    if metrics['data_transfer_times']:
        metrics['avg_transfer_time'] = np.mean(metrics['data_transfer_times'])
    
    metrics['total_time'] = total_time
    metrics['num_batches_profiled'] = len(metrics['batch_times'])
    
    return metrics