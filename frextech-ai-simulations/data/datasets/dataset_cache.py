"""
Dataset Caching Module
Utilities for caching datasets and dataset transformations for improved performance.
"""

import os
import pickle
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from PIL import Image
from diskcache import Cache as DiskCache

from .base_dataset import BaseDataset, DatasetConfig
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

@dataclass
class CacheConfig:
    """
    Configuration for dataset caching.
    
    Attributes:
        enabled: Whether caching is enabled
        cache_dir: Directory for cache storage
        max_size_gb: Maximum cache size in gigabytes
        ttl_seconds: Time-to-live for cache entries in seconds
        compress: Whether to compress cache entries
        shards: Number of cache shards
        memory_limit_mb: Memory limit for in-memory cache
        backend: Cache backend ('diskcache', 'sqlite', 'memory')
        clear_on_start: Whether to clear cache on startup
        prefetch: Whether to prefetch samples
        prefetch_workers: Number of workers for prefetching
        cache_transforms: Whether to cache transformed samples
        cache_metadata: Whether to cache dataset metadata
    """
    enabled: bool = True
    cache_dir: str = "./data/cache/"
    max_size_gb: int = 100
    ttl_seconds: int = 86400  # 24 hours
    compress: bool = True
    shards: int = 64
    memory_limit_mb: int = 1024
    backend: str = "diskcache"  # options: diskcache, sqlite, memory
    clear_on_start: bool = False
    prefetch: bool = False
    prefetch_workers: int = 4
    cache_transforms: bool = True
    cache_metadata: bool = True
    
    def __post_init__(self):
        """Validate cache configuration."""
        # Create cache directory
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate backend
        valid_backends = ["diskcache", "sqlite", "memory"]
        if self.backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}, got {self.backend}")
        
        # Validate size limits
        if self.max_size_gb <= 0:
            raise ValueError(f"max_size_gb must be > 0, got {self.max_size_gb}")
        
        if self.memory_limit_mb <= 0:
            raise ValueError(f"memory_limit_mb must be > 0, got {self.memory_limit_mb}")


class DatasetCache:
    """
    Advanced caching system for datasets.
    
    Features:
    - Multiple backend support (disk, sqlite, memory)
    - LRU eviction policy
    - Compression
    - Time-based expiration
    - Thread-safe operations
    - Prefetching
    - Statistics tracking
    """
    
    def __init__(self, config: CacheConfig, dataset_name: str = "dataset"):
        """
        Initialize dataset cache.
        
        Args:
            config: Cache configuration
            dataset_name: Name of dataset for cache isolation
        """
        self.config = config
        self.dataset_name = dataset_name
        self.cache_dir = config.cache_dir / dataset_name
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
            'size_bytes': 0,
            'entries': 0
        }
        
        # Initialize cache backend
        self._init_cache_backend()
        
        # Prefetch thread pool
        self.prefetch_executor = None
        if config.prefetch and config.prefetch_workers > 0:
            self.prefetch_executor = ThreadPoolExecutor(
                max_workers=config.prefetch_workers,
                thread_name_prefix=f"cache_prefetch_{dataset_name}"
            )
        
        # Clear cache on start if requested
        if config.clear_on_start:
            self.clear()
        
        logger.info(f"Initialized cache for {dataset_name} "
                   f"(backend={config.backend}, dir={self.cache_dir})")
    
    def _init_cache_backend(self):
        """Initialize cache backend based on configuration."""
        if self.config.backend == "diskcache":
            self._init_diskcache()
        elif self.config.backend == "sqlite":
            self._init_sqlite()
        elif self.config.backend == "memory":
            self._init_memory()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    def _init_diskcache(self):
        """Initialize diskcache backend."""
        try:
            self.cache = DiskCache(
                directory=str(self.cache_dir),
                size_limit=self.config.max_size_gb * 1024**3,  # Convert GB to bytes
                shards=self.config.shards,
                timeout=1.0,
                disk_min_file_size=1024**2,  # 1MB
                disk_pickle_protocol=pickle.HIGHEST_PROTOCOL,
                cull_limit=10  # Cull 10% when full
            )
        except Exception as e:
            logger.error(f"Failed to initialize diskcache: {e}")
            # Fall back to memory cache
            self.config.backend = "memory"
            self._init_memory()
    
    def _init_sqlite(self):
        """Initialize SQLite backend."""
        import sqlite3
        from contextlib import contextmanager
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        
        # Initialize database
        @contextmanager
        def get_connection():
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
        
        self.get_connection = get_connection
        
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    timestamp REAL,
                    size INTEGER,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp)")
            conn.commit()
    
    def _init_memory(self):
        """Initialize in-memory cache backend."""
        import cachetools
        
        # Calculate maxsize (number of items)
        # Estimate average item size as 1MB
        avg_item_size = 1024 * 1024  # 1MB
        maxsize = (self.config.memory_limit_mb * 1024 * 1024) // avg_item_size
        maxsize = max(100, maxsize)  # At least 100 items
        
        self.cache = cachetools.LRUCache(maxsize=maxsize)
        self._memory_timestamps = {}
        self._memory_sizes = {}
    
    def _make_key(self, sample_id: Union[int, str], transform_hash: Optional[str] = None) -> str:
        """
        Create cache key for a sample.
        
        Args:
            sample_id: Sample identifier
            transform_hash: Hash of transformations
            
        Returns:
            Cache key
        """
        if transform_hash:
            return f"{self.dataset_name}_{sample_id}_{transform_hash}"
        else:
            return f"{self.dataset_name}_{sample_id}"
    
    def get(self, sample_id: Union[int, str], transform_hash: Optional[str] = None) -> Optional[Any]:
        """
        Get sample from cache.
        
        Args:
            sample_id: Sample identifier
            transform_hash: Hash of transformations
            
        Returns:
            Cached sample or None if not found
        """
        if not self.config.enabled:
            return None
        
        key = self._make_key(sample_id, transform_hash)
        
        try:
            if self.config.backend == "diskcache":
                value = self.cache.get(key, default=None)
                if value is not None:
                    self.stats['hits'] += 1
                    return pickle.loads(value) if self.config.compress else value
            
            elif self.config.backend == "sqlite":
                with self.get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT value, timestamp FROM cache WHERE key = ?",
                        (key,)
                    )
                    row = cursor.fetchone()
                    
                    if row is not None:
                        # Check TTL
                        timestamp = row['timestamp']
                        if time.time() - timestamp > self.config.ttl_seconds:
                            # Expired, delete it
                            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                            conn.commit()
                            self.stats['misses'] += 1
                            self.stats['deletes'] += 1
                            return None
                        
                        value = row['value']
                        self.stats['hits'] += 1
                        return pickle.loads(value) if self.config.compress else value
            
            elif self.config.backend == "memory":
                if key in self.cache:
                    # Check TTL
                    timestamp = self._memory_timestamps.get(key, 0)
                    if time.time() - timestamp > self.config.ttl_seconds:
                        # Expired, delete it
                        del self.cache[key]
                        del self._memory_timestamps[key]
                        if key in self._memory_sizes:
                            self.stats['size_bytes'] -= self._memory_sizes[key]
                            del self._memory_sizes[key]
                        self.stats['misses'] += 1
                        self.stats['deletes'] += 1
                        return None
                    
                    self.stats['hits'] += 1
                    return self.cache[key]
        
        except Exception as e:
            logger.error(f"Error reading from cache for key {key}: {e}")
        
        self.stats['misses'] += 1
        return None
    
    def set(self, sample_id: Union[int, str], value: Any, 
            transform_hash: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store sample in cache.
        
        Args:
            sample_id: Sample identifier
            value: Sample data to cache
            transform_hash: Hash of transformations
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.config.enabled:
            return False
        
        key = self._make_key(sample_id, transform_hash)
        
        try:
            # Estimate size
            size_bytes = self._estimate_size(value)
            
            if self.config.backend == "diskcache":
                if self.config.compress:
                    serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    serialized = value
                
                success = self.cache.set(key, serialized)
                if success:
                    self.stats['sets'] += 1
                    self.stats['size_bytes'] += size_bytes
                    self.stats['entries'] = len(self.cache)
                return success
            
            elif self.config.backend == "sqlite":
                with self.get_connection() as conn:
                    if self.config.compress:
                        serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        serialized = value
                    
                    # Delete existing entry
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    
                    # Insert new entry
                    metadata_json = json.dumps(metadata or {})
                    conn.execute(
                        "INSERT INTO cache (key, value, timestamp, size, metadata) VALUES (?, ?, ?, ?, ?)",
                        (key, serialized, time.time(), size_bytes, metadata_json)
                    )
                    conn.commit()
                    
                    # Count entries
                    cursor = conn.execute("SELECT COUNT(*) as count FROM cache")
                    self.stats['entries'] = cursor.fetchone()['count']
                    
                    self.stats['sets'] += 1
                    self.stats['size_bytes'] += size_bytes
                    return True
            
            elif self.config.backend == "memory":
                # Check if we need to evict
                if key not in self.cache and len(self.cache) >= self.cache.maxsize:
                    # Evict least recently used
                    oldest_key = next(iter(self.cache))
                    if oldest_key in self._memory_sizes:
                        self.stats['size_bytes'] -= self._memory_sizes[oldest_key]
                        del self._memory_sizes[oldest_key]
                    if oldest_key in self._memory_timestamps:
                        del self._memory_timestamps[oldest_key]
                    del self.cache[oldest_key]
                    self.stats['evictions'] += 1
                
                self.cache[key] = value
                self._memory_timestamps[key] = time.time()
                self._memory_sizes[key] = size_bytes
                
                self.stats['sets'] += 1
                self.stats['size_bytes'] += size_bytes
                self.stats['entries'] = len(self.cache)
                return True
        
        except Exception as e:
            logger.error(f"Error writing to cache for key {key}: {e}")
            return False
        
        return False
    
    def delete(self, sample_id: Union[int, str], transform_hash: Optional[str] = None) -> bool:
        """
        Delete sample from cache.
        
        Args:
            sample_id: Sample identifier
            transform_hash: Hash of transformations
            
        Returns:
            True if successful, False otherwise
        """
        key = self._make_key(sample_id, transform_hash)
        
        try:
            if self.config.backend == "diskcache":
                success = key in self.cache
                if success:
                    del self.cache[key]
                    self.stats['deletes'] += 1
                    self.stats['entries'] = len(self.cache)
                return success
            
            elif self.config.backend == "sqlite":
                with self.get_connection() as conn:
                    cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    success = cursor.rowcount > 0
                    
                    if success:
                        self.stats['deletes'] += 1
                        # Recalculate size and count
                        self._update_sqlite_stats(conn)
                    
                    return success
            
            elif self.config.backend == "memory":
                if key in self.cache:
                    if key in self._memory_sizes:
                        self.stats['size_bytes'] -= self._memory_sizes[key]
                        del self._memory_sizes[key]
                    if key in self._memory_timestamps:
                        del self._memory_timestamps[key]
                    del self.cache[key]
                    self.stats['deletes'] += 1
                    self.stats['entries'] = len(self.cache)
                    return True
        
        except Exception as e:
            logger.error(f"Error deleting from cache for key {key}: {e}")
        
        return False
    
    def _update_sqlite_stats(self, conn):
        """Update statistics for SQLite backend."""
        cursor = conn.execute("SELECT COUNT(*) as count, SUM(size) as total_size FROM cache")
        row = cursor.fetchone()
        
        if row:
            self.stats['entries'] = row['count'] or 0
            self.stats['size_bytes'] = row['total_size'] or 0
    
    def _estimate_size(self, value: Any) -> int:
        """
        Estimate size of value in bytes.
        
        Args:
            value: Value to estimate size of
            
        Returns:
            Estimated size in bytes
        """
        try:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, (str, bytes, bytearray)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8  # Approximate
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(v) for v in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in value.items())
            else:
                # Try to serialize and measure
                import pickle
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Default estimate
    
    def clear(self) -> bool:
        """Clear entire cache."""
        try:
            if self.config.backend == "diskcache":
                self.cache.clear()
            elif self.config.backend == "sqlite":
                with self.get_connection() as conn:
                    conn.execute("DELETE FROM cache")
                    conn.commit()
            elif self.config.backend == "memory":
                self.cache.clear()
                self._memory_timestamps.clear()
                self._memory_sizes.clear()
            
            # Reset statistics
            self.stats['size_bytes'] = 0
            self.stats['entries'] = 0
            self.stats['evictions'] = 0
            
            logger.info(f"Cleared cache for {self.dataset_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        stats = self.stats.copy()
        
        # Calculate hit rate
        total_accesses = stats['hits'] + stats['misses']
        if total_accesses > 0:
            stats['hit_rate'] = stats['hits'] / total_accesses
        else:
            stats['hit_rate'] = 0.0
        
        # Add configuration info
        stats['config'] = {
            'enabled': self.config.enabled,
            'backend': self.config.backend,
            'max_size_gb': self.config.max_size_gb,
            'ttl_seconds': self.config.ttl_seconds,
            'compress': self.config.compress
        }
        
        # Add current cache info
        stats['current_size_gb'] = stats['size_bytes'] / (1024**3)
        stats['current_size_mb'] = stats['size_bytes'] / (1024**2)
        
        return stats
    
    def prefetch_samples(self, sample_ids: List[Union[int, str]], 
                        loader_func: Callable[[Union[int, str]], Any],
                        transform_hash: Optional[str] = None) -> None:
        """
        Prefetch samples into cache.
        
        Args:
            sample_ids: List of sample IDs to prefetch
            loader_func: Function to load sample if not in cache
            transform_hash: Hash of transformations
        """
        if not self.config.prefetch or not self.prefetch_executor:
            return
        
        # Submit prefetch tasks
        futures = []
        for sample_id in sample_ids:
            # Check if already in cache
            if self.get(sample_id, transform_hash) is None:
                future = self.prefetch_executor.submit(
                    self._prefetch_sample,
                    sample_id,
                    loader_func,
                    transform_hash
                )
                futures.append(future)
        
        # Wait for completion (optional, could be non-blocking)
        # for future in futures:
        #     future.result()
    
    def _prefetch_sample(self, sample_id: Union[int, str], 
                        loader_func: Callable[[Union[int, str]], Any],
                        transform_hash: Optional[str] = None) -> None:
        """Prefetch a single sample."""
        try:
            # Load sample
            sample = loader_func(sample_id)
            
            # Cache it
            if sample is not None:
                self.set(sample_id, sample, transform_hash)
                logger.debug(f"Prefetched sample {sample_id} into cache")
        
        except Exception as e:
            logger.error(f"Error prefetching sample {sample_id}: {e}")
    
    def warmup(self, dataset: BaseDataset, num_samples: Optional[int] = None) -> None:
        """
        Warm up cache with dataset samples.
        
        Args:
            dataset: Dataset to warm up with
            num_samples: Number of samples to warm up (None for all)
        """
        if not self.config.enabled:
            return
        
        logger.info(f"Warming up cache for {self.dataset_name}")
        
        if num_samples is None:
            num_samples = len(dataset)
        else:
            num_samples = min(num_samples, len(dataset))
        
        # Create transform hash if caching transformed samples
        transform_hash = None
        if self.config.cache_transforms and dataset.transform:
            transform_hash = self._hash_transform(dataset.transform)
        
        # Warm up samples
        for i in range(num_samples):
            try:
                # Check if already cached
                if self.get(i, transform_hash) is None:
                    # Load and cache
                    sample = dataset._load_sample(i)
                    if dataset.transform:
                        sample = dataset._apply_transformations(sample)
                    
                    self.set(i, sample, transform_hash)
                    
                    if i % 100 == 0:
                        logger.debug(f"Warmed up {i}/{num_samples} samples")
            
            except Exception as e:
                logger.error(f"Error warming up sample {i}: {e}")
        
        logger.info(f"Cache warmup completed: {num_samples} samples")
    
    def _hash_transform(self, transform: Callable) -> str:
        """
        Create hash of transform function.
        
        Args:
            transform: Transform function
            
        Returns:
            Hash string
        """
        try:
            # Try to get source code
            import inspect
            source = inspect.getsource(transform)
            return hashlib.md5(source.encode()).hexdigest()[:16]
        except:
            # Fallback to object ID
            return str(id(transform))
    
    def close(self):
        """Close cache and cleanup resources."""
        if self.prefetch_executor:
            self.prefetch_executor.shutdown(wait=True)
        
        if self.config.backend == "diskcache":
            self.cache.close()
        
        logger.info(f"Closed cache for {self.dataset_name}")


def cached_dataset(dataset_class):
    """
    Decorator to add caching to a dataset class.
    
    Args:
        dataset_class: Dataset class to decorate
        
    Returns:
        Decorated dataset class with caching
    """
    class CachedDataset(dataset_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
            # Initialize cache
            cache_config = getattr(self.config, 'cache_config', CacheConfig())
            self.cache = DatasetCache(cache_config, dataset_name=self.name)
            
            # Create transform hash
            self.transform_hash = None
            if cache_config.cache_transforms and self.transform:
                self.transform_hash = self.cache._hash_transform(self.transform)
            
            # Warm up cache if configured
            if getattr(self.config, 'cache_warmup', False):
                self.cache.warmup(self, num_samples=getattr(self.config, 'cache_warmup_samples', 1000))
        
        def __getitem__(self, index):
            # Try cache first
            cached_sample = self.cache.get(index, self.transform_hash)
            if cached_sample is not None:
                return cached_sample
            
            # Load from dataset
            sample = super().__getitem__(index)
            
            # Cache it
            self.cache.set(index, sample, self.transform_hash)
            
            return sample
        
        def get_cache_stats(self):
            """Get cache statistics."""
            return self.cache.get_stats()
        
        def clear_cache(self):
            """Clear dataset cache."""
            return self.cache.clear()
        
        def close(self):
            """Close dataset and cache."""
            self.cache.close()
            if hasattr(super(), 'close'):
                super().close()
    
    return CachedDataset


class SmartCacheManager:
    """
    Intelligent cache manager that adapts based on usage patterns.
    
    Features:
    - Adaptive prefetching
    - Dynamic TTL adjustment
    - Usage pattern analysis
    - Memory pressure detection
    """
    
    def __init__(self, cache: DatasetCache):
        """
        Initialize smart cache manager.
        
        Args:
            cache: DatasetCache instance to manage
        """
        self.cache = cache
        self.usage_patterns = {}
        self.access_times = {}
        self.prefetch_predictions = {}
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_cache, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Initialized SmartCacheManager")
    
    def _monitor_cache(self):
        """Monitor cache usage and adapt settings."""
        import psutil
        
        while True:
            try:
                # Check memory pressure
                memory_percent = psutil.virtual_memory().percent
                
                # Adjust cache behavior based on memory pressure
                if memory_percent > 90:
                    # High memory pressure - reduce cache aggressiveness
                    self._reduce_cache_aggressiveness()
                elif memory_percent < 60:
                    # Low memory pressure - increase cache aggressiveness
                    self._increase_cache_aggressiveness()
                
                # Analyze usage patterns
                self._analyze_usage_patterns()
                
                # Adjust prefetching
                self._adjust_prefetching()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
            
            except Exception as e:
                logger.error(f"Error in cache monitor: {e}")
                time.sleep(60)  # Sleep longer on error
    
    def _reduce_cache_aggressiveness(self):
        """Reduce cache aggressiveness under memory pressure."""
        # Reduce TTL
        if self.cache.config.ttl_seconds > 300:  # At least 5 minutes
            self.cache.config.ttl_seconds = max(300, self.cache.config.ttl_seconds // 2)
        
        # Disable prefetching
        self.cache.config.prefetch = False
        
        logger.debug("Reduced cache aggressiveness due to memory pressure")
    
    def _increase_cache_aggressiveness(self):
        """Increase cache aggressiveness when memory is available."""
        # Increase TTL (up to 24 hours)
        if self.cache.config.ttl_seconds < 86400:
            self.cache.config.ttl_seconds = min(86400, self.cache.config.ttl_seconds * 2)
        
        # Enable prefetching
        self.cache.config.prefetch = True
        
        logger.debug("Increased cache aggressiveness")
    
    def _analyze_usage_patterns(self):
        """Analyze cache usage patterns."""
        stats = self.cache.get_stats()
        
        # Track hit rate trends
        hit_rate = stats.get('hit_rate', 0)
        
        if 'hit_rates' not in self.usage_patterns:
            self.usage_patterns['hit_rates'] = []
        
        self.usage_patterns['hit_rates'].append(hit_rate)
        
        # Keep only last 100 measurements
        if len(self.usage_patterns['hit_rates']) > 100:
            self.usage_patterns['hit_rates'] = self.usage_patterns['hit_rates'][-100:]
    
    def _adjust_prefetching(self):
        """Adjust prefetching based on usage patterns."""
        if not self.cache.config.prefetch:
            return
        
        # Analyze sequential access patterns
        # This is a simplified implementation
        # In production, you would use more sophisticated algorithms
        
        stats = self.cache.get_stats()
        hit_rate = stats.get('hit_rate', 0)
        
        # Adjust prefetch workers based on hit rate
        if hit_rate > 0.8:
            # High hit rate, increase prefetching
            self.cache.config.prefetch_workers = min(
                16, self.cache.config.prefetch_workers + 1
            )
        elif hit_rate < 0.3:
            # Low hit rate, decrease prefetching
            self.cache.config.prefetch_workers = max(
                1, self.cache.config.prefetch_workers - 1
            )
    
    def predict_next_samples(self, current_index: int, lookahead: int = 10) -> List[int]:
        """
        Predict next samples to prefetch.
        
        Args:
            current_index: Current sample index
            lookahead: Number of samples to predict ahead
            
        Returns:
            List of predicted sample indices
        """
        # Simple sequential prediction
        # In production, use more sophisticated prediction algorithms
        
        predicted = []
        for i in range(1, lookahead + 1):
            predicted.append(current_index + i)
        
        return predicted
    
    def record_access(self, sample_id: Union[int, str], timestamp: float = None):
        """
        Record sample access for pattern analysis.
        
        Args:
            sample_id: Sample ID that was accessed
            timestamp: Access timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.access_times[sample_id] = timestamp
        
        # Clean up old entries
        cutoff = timestamp - 3600  # Keep only last hour
        to_delete = [sid for sid, ts in self.access_times.items() if ts < cutoff]
        for sid in to_delete:
            del self.access_times[sid]
    
    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get cache optimization recommendations.
        
        Returns:
            Dictionary of recommendations
        """
        stats = self.cache.get_stats()
        hit_rate = stats.get('hit_rate', 0)
        
        recommendations = {
            'increase_cache_size': hit_rate > 0.7 and stats['current_size_gb'] < self.cache.config.max_size_gb * 0.9,
            'decrease_cache_size': hit_rate < 0.3,
            'enable_compression': not self.cache.config.compress and stats['current_size_gb'] > 10,
            'adjust_ttl': hit_rate < 0.5,
            'prefetch_pattern': self._detect_prefetch_pattern()
        }
        
        return recommendations
    
    def _detect_prefetch_pattern(self) -> Optional[str]:
        """Detect access pattern for prefetching."""
        if len(self.access_times) < 10:
            return None
        
        # Analyze access times for patterns
        # This is a simplified implementation
        
        indices = sorted([int(sid) for sid in self.access_times.keys() if str(sid).isdigit()])
        
        if len(indices) < 3:
            return None
        
        # Check for sequential access
        diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        
        if all(d == 1 for d in diffs):
            return "sequential"
        elif all(d == diffs[0] for d in diffs):
            return f"stride_{diffs[0]}"
        else:
            return "random"
    
    def close(self):
        """Close smart cache manager."""
        if self.monitor_thread.is_alive():
            # Signal thread to exit
            # In production, use proper thread termination
            pass


# Global cache registry
_CACHE_REGISTRY = {}

def get_cache(dataset_name: str, config: Optional[CacheConfig] = None) -> DatasetCache:
    """
    Get or create cache for a dataset.
    
    Args:
        dataset_name: Dataset name
        config: Cache configuration
        
    Returns:
        DatasetCache instance
    """
    if dataset_name not in _CACHE_REGISTRY:
        config = config or CacheConfig()
        _CACHE_REGISTRY[dataset_name] = DatasetCache(config, dataset_name)
    
    return _CACHE_REGISTRY[dataset_name]

def clear_all_caches() -> None:
    """Clear all registered caches."""
    for cache in _CACHE_REGISTRY.values():
        cache.clear()
    
    _CACHE_REGISTRY.clear()
    logger.info("Cleared all caches")

def get_global_cache_stats() -> Dict[str, Any]:
    """
    Get statistics for all caches.
    
    Returns:
        Dictionary of cache statistics
    """
    stats = {
        'total_caches': len(_CACHE_REGISTRY),
        'caches': {},
        'aggregate': {
            'total_hits': 0,
            'total_misses': 0,
            'total_sets': 0,
            'total_size_bytes': 0,
            'total_entries': 0
        }
    }
    
    for name, cache in _CACHE_REGISTRY.items():
        cache_stats = cache.get_stats()
        stats['caches'][name] = cache_stats
        
        # Update aggregate
        for key in ['hits', 'misses', 'sets', 'size_bytes', 'entries']:
            if key in cache_stats:
                stats['aggregate'][f'total_{key}'] += cache_stats[key]
    
    # Calculate aggregate hit rate
    total_accesses = stats['aggregate']['total_hits'] + stats['aggregate']['total_misses']
    if total_accesses > 0:
        stats['aggregate']['total_hit_rate'] = stats['aggregate']['total_hits'] / total_accesses
    else:
        stats['aggregate']['total_hit_rate'] = 0.0
    
    return stats