"""
Advanced file I/O utilities with support for multiple formats, compression, and efficient data handling.
"""

import os
import json
import pickle
import yaml
import h5py
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Tuple, Union, Optional, BinaryIO, TextIO, Callable
from pathlib import Path
import io
import gzip
import bz2
import lzma
import lz4.frame
import zstandard as zstd
import struct
import hashlib
import tempfile
import shutil
import tarfile
import zipfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import logging
from contextlib import contextmanager
import gc
import warnings
import inspect
import sys

# Image/video handling
try:
    import cv2
    import PIL.Image
    import imageio
    HAS_IMAGE = True
except ImportError:
    HAS_IMAGE = False

# Audio handling
try:
    import soundfile as sf
    import librosa
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Point cloud handling
try:
    import open3d as o3d
    HAS_POINT_CLOUD = True
except ImportError:
    HAS_POINT_CLOUD = False


class FileType(Enum):
    """Supported file types."""
    TEXT = auto()
    JSON = auto()
    YAML = auto()
    PICKLE = auto()
    NUMPY = auto()
    HDF5 = auto()
    TORCH = auto()
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    CSV = auto()
    EXCEL = auto()
    PARQUET = auto()
    FEATHER = auto()
    POINT_CLOUD = auto()
    CUSTOM = auto()


class CompressionType(Enum):
    """Compression types."""
    NONE = auto()
    GZIP = auto()
    BZIP2 = auto()
    LZMA = auto()
    LZ4 = auto()
    ZSTD = auto()


@dataclass
class FileInfo:
    """File information."""
    path: Path
    size: int
    created: datetime
    modified: datetime
    accessed: datetime
    file_type: Optional[FileType] = None
    compression: Optional[CompressionType] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Cache entry for file data."""
    data: Any
    timestamp: float
    size: int
    access_count: int = 0
    last_accessed: float = field(default_factory=lambda: time.time())


class LRUCache:
    """LRU cache for file data."""
    
    def __init__(self, max_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.max_size = max_size
        self.current_size = 0
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            # Move to end of access order (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return entry.data
    
    def put(self, key: str, data: Any, size: int):
        """Put item in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            size: Estimated size in bytes
        """
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.current_size -= old_entry.size
                if key in self.access_order:
                    self.access_order.remove(key)
            
            # Make space if needed
            while self.current_size + size > self.max_size and self.access_order:
                lru_key = self.access_order.pop(0)
                if lru_key in self.cache:
                    lru_entry = self.cache.pop(lru_key)
                    self.current_size -= lru_entry.size
            
            # Add new entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                size=size,
                access_count=1,
                last_accessed=time.time()
            )
            self.cache[key] = entry
            self.current_size += size
            self.access_order.append(key)
    
    def remove(self, key: str):
        """Remove item from cache.
        
        Args:
            key: Cache key
        """
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_size -= entry.size
                if key in self.access_order:
                    self.access_order.remove(key)
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.lock:
            return {
                "max_size": self.max_size,
                "current_size": self.current_size,
                "num_items": len(self.cache),
                "hit_rate": self._calculate_hit_rate(),
                "access_order_length": len(self.access_order)
            }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate.
        
        Returns:
            Hit rate (0-1)
        """
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        if total_accesses == 0:
            return 0.0
        
        hits = sum(1 for entry in self.cache.values() if entry.access_count > 0)
        return hits / len(self.cache) if self.cache else 0.0


class FileManager:
    """Advanced file manager with caching, compression, and format detection."""
    
    def __init__(self, 
                 base_path: Union[str, Path] = None,
                 cache_size: int = 1024 * 1024 * 1024,  # 1GB
                 enable_cache: bool = True,
                 enable_compression: bool = True,
                 thread_pool_size: int = 4,
                 process_pool_size: int = 2):
        """Initialize file manager.
        
        Args:
            base_path: Base directory for file operations
            cache_size: Cache size in bytes
            enable_cache: Whether to enable caching
            enable_compression: Whether to enable compression
            thread_pool_size: Thread pool size for I/O operations
            process_pool_size: Process pool size for CPU-intensive operations
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.cache = LRUCache(cache_size) if enable_cache else None
        self.enable_compression = enable_compression
        
        # Thread/process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self.process_pool = ProcessPoolExecutor(max_workers=process_pool_size)
        
        # File type detection
        self.file_extensions = {
            # Text
            '.txt': FileType.TEXT, '.md': FileType.TEXT, '.log': FileType.TEXT,
            '.xml': FileType.TEXT, '.html': FileType.TEXT, '.css': FileType.TEXT,
            '.js': FileType.TEXT, '.py': FileType.TEXT, '.json': FileType.JSON,
            '.yaml': FileType.YAML, '.yml': FileType.YAML,
            
            # Binary
            '.pkl': FileType.PICKLE, '.pickle': FileType.PICKLE,
            '.npy': FileType.NUMPY, '.npz': FileType.NUMPY,
            '.h5': FileType.HDF5, '.hdf5': FileType.HDF5,
            '.pt': FileType.TORCH, '.pth': FileType.TORCH,
            
            # Images
            '.png': FileType.IMAGE, '.jpg': FileType.IMAGE, '.jpeg': FileType.IMAGE,
            '.bmp': FileType.IMAGE, '.tiff': FileType.IMAGE, '.tif': FileType.IMAGE,
            '.gif': FileType.IMAGE, '.webp': FileType.IMAGE,
            
            # Video
            '.mp4': FileType.VIDEO, '.avi': FileType.VIDEO, '.mov': FileType.VIDEO,
            '.mkv': FileType.VIDEO, '.webm': FileType.VIDEO,
            
            # Audio
            '.wav': FileType.AUDIO, '.mp3': FileType.AUDIO, '.flac': FileType.AUDIO,
            '.ogg': FileType.AUDIO,
            
            # Data
            '.csv': FileType.CSV, '.tsv': FileType.CSV,
            '.xlsx': FileType.EXCEL, '.xls': FileType.EXCEL,
            '.parquet': FileType.PARQUET,
            '.feather': FileType.FEATHER,
            
            # Point clouds
            '.ply': FileType.POINT_CLOUD, '.pcd': FileType.POINT_CLOUD,
            '.obj': FileType.POINT_CLOUD,
        }
        
        # Compression detection
        self.compression_extensions = {
            '.gz': CompressionType.GZIP,
            '.bz2': CompressionType.BZIP2,
            '.xz': CompressionType.LZMA,
            '.lz4': CompressionType.LZ4,
            '.zst': CompressionType.ZSTD,
        }
        
        # Serialization functions
        self.serializers = {}
        self.deserializers = {}
        self._register_default_serializers()
        
        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'bytes_read': 0,
            'bytes_written': 0,
            'compression_ratio': 0.0,
        }
        
        # Locks for thread safety
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _register_default_serializers(self):
        """Register default serializers for file types."""
        # JSON
        self.register_serializer(FileType.JSON, self._serialize_json, self._deserialize_json)
        
        # YAML
        self.register_serializer(FileType.YAML, self._serialize_yaml, self._deserialize_yaml)
        
        # Pickle
        self.register_serializer(FileType.PICKLE, self._serialize_pickle, self._deserialize_pickle)
        
        # NumPy
        self.register_serializer(FileType.NUMPY, self._serialize_numpy, self._deserialize_numpy)
        
        # PyTorch
        self.register_serializer(FileType.TORCH, self._serialize_torch, self._deserialize_torch)
        
        # Text
        self.register_serializer(FileType.TEXT, self._serialize_text, self._deserialize_text)
        
        # Image
        if HAS_IMAGE:
            self.register_serializer(FileType.IMAGE, self._serialize_image, self._deserialize_image)
        
        # CSV
        self.register_serializer(FileType.CSV, self._serialize_csv, self._deserialize_csv)
        
        # HDF5
        self.register_serializer(FileType.HDF5, self._serialize_hdf5, self._deserialize_hdf5)
    
    def register_serializer(self, 
                           file_type: FileType,
                           serialize_func: Callable[[Any, Union[str, Path]], None],
                           deserialize_func: Callable[[Union[str, Path]], Any]):
        """Register custom serializer/deserializer.
        
        Args:
            file_type: File type
            serialize_func: Serialization function
            deserialize_func: Deserialization function
        """
        self.serializers[file_type] = serialize_func
        self.deserializers[file_type] = deserialize_func
    
    def detect_file_type(self, filepath: Union[str, Path]) -> Tuple[FileType, CompressionType]:
        """Detect file type and compression.
        
        Args:
            filepath: Path to file
            
        Returns:
            Tuple of (file_type, compression_type)
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        # Check compression
        compression = CompressionType.NONE
        for comp_ext, comp_type in self.compression_extensions.items():
            if str(path).endswith(comp_ext):
                compression = comp_type
                # Remove compression extension for file type detection
                stem = path.stem
                while Path(stem).suffix.lower() in self.compression_extensions:
                    stem = Path(stem).stem
                suffix = Path(stem).suffix.lower()
                break
        
        # Detect file type
        file_type = FileType.CUSTOM
        if suffix in self.file_extensions:
            file_type = self.file_extensions[suffix]
        else:
            # Try to detect by content
            file_type = self._detect_by_content(path)
        
        return file_type, compression
    
    def _detect_by_content(self, filepath: Path) -> FileType:
        """Detect file type by content.
        
        Args:
            filepath: Path to file
            
        Returns:
            Detected file type
        """
        if not filepath.exists():
            return FileType.CUSTOM
        
        try:
            with open(filepath, 'rb') as f:
                # Read first few bytes
                header = f.read(1024)
                
                # Check for common file signatures
                if header.startswith(b'\x89PNG\r\n\x1a\n'):
                    return FileType.IMAGE
                elif header.startswith(b'\xff\xd8\xff'):
                    return FileType.IMAGE
                elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                    return FileType.IMAGE
                elif header.startswith(b'BM'):
                    return FileType.IMAGE
                elif header.startswith(b'RIFF') and header[8:12] == b'WAVE':
                    return FileType.AUDIO
                elif header.startswith(b'ID3'):
                    return FileType.AUDIO
                elif header.startswith(b'OggS'):
                    return FileType.AUDIO
                elif header.startswith(b'\x1aE\xdf\xa3'):
                    return FileType.VIDEO  # WebM
                elif header.startswith(b'\x00\x00\x00 ftyp'):
                    return FileType.VIDEO  # MP4
                elif header.startswith(b'{\n') or header.startswith(b'{\r\n'):
                    # Check if it's valid JSON
                    try:
                        json.loads(header.decode('utf-8', errors='ignore').split('\n')[0] + '}')
                        return FileType.JSON
                    except:
                        pass
                elif header.startswith(b'---\n') or b'\n---\n' in header:
                    return FileType.YAML
                elif header.startswith(b'\x93NUMPY'):
                    return FileType.NUMPY
                elif b'<html' in header.lower() or b'<!doctype html' in header.lower():
                    return FileType.TEXT
                
                # Check for text vs binary
                try:
                    header.decode('utf-8')
                    return FileType.TEXT
                except UnicodeDecodeError:
                    return FileType.CUSTOM
                    
        except Exception:
            return FileType.CUSTOM
    
    def get_file_info(self, filepath: Union[str, Path]) -> FileInfo:
        """Get detailed file information.
        
        Args:
            filepath: Path to file
            
        Returns:
            FileInfo object
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = path.stat()
        file_type, compression = self.detect_file_type(path)
        
        # Calculate checksum
        checksum = self._calculate_checksum(path)
        
        return FileInfo(
            path=path,
            size=stat.st_size,
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            accessed=datetime.fromtimestamp(stat.st_atime),
            file_type=file_type,
            compression=compression,
            checksum=checksum
        )
    
    def _calculate_checksum(self, filepath: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum.
        
        Args:
            filepath: Path to file
            algorithm: Hash algorithm
            
        Returns:
            Checksum string
        """
        hash_func = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    def read(self, 
             filepath: Union[str, Path],
             file_type: Optional[FileType] = None,
             cache: bool = True,
             decompress: bool = True) -> Any:
        """Read file with automatic type detection.
        
        Args:
            filepath: Path to file
            file_type: Optional file type (auto-detected if None)
            cache: Whether to use cache
            decompress: Whether to decompress automatically
            
        Returns:
            File contents
        """
        path = Path(filepath)
        cache_key = str(path.absolute())
        
        # Check cache
        if cache and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self.stats['cache_hits'] += 1
                self.logger.debug(f"Cache hit: {filepath}")
                return cached
        
        self.stats['cache_misses'] += 1
        
        # Detect file type if not provided
        if file_type is None:
            detected_type, compression = self.detect_file_type(path)
            file_type = detected_type
        else:
            _, compression = self.detect_file_type(path)
        
        # Open file with appropriate compression
        if decompress and compression != CompressionType.NONE:
            data = self._read_compressed(path, compression, file_type)
        else:
            # Use appropriate deserializer
            if file_type in self.deserializers:
                data = self.deserializers[file_type](path)
            else:
                # Fallback to binary read
                with open(path, 'rb') as f:
                    data = f.read()
        
        # Update statistics
        self.stats['reads'] += 1
        self.stats['bytes_read'] += path.stat().st_size
        
        # Cache if enabled
        if cache and self.cache:
            # Estimate size (approximate)
            size = self._estimate_size(data)
            self.cache.put(cache_key, data, size)
        
        return data
    
    def _read_compressed(self, 
                        filepath: Path, 
                        compression: CompressionType,
                        file_type: FileType) -> Any:
        """Read compressed file.
        
        Args:
            filepath: Path to compressed file
            compression: Compression type
            file_type: File type after decompression
            
        Returns:
            Decompressed data
        """
        # Open with appropriate decompression
        if compression == CompressionType.GZIP:
            with gzip.open(filepath, 'rb') as f:
                decompressed = f.read()
        elif compression == CompressionType.BZIP2:
            with bz2.open(filepath, 'rb') as f:
                decompressed = f.read()
        elif compression == CompressionType.LZMA:
            with lzma.open(filepath, 'rb') as f:
                decompressed = f.read()
        elif compression == CompressionType.LZ4:
            with open(filepath, 'rb') as f:
                compressed_data = f.read()
            decompressed = lz4.frame.decompress(compressed_data)
        elif compression == CompressionType.ZSTD:
            with open(filepath, 'rb') as f:
                compressed_data = f.read()
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(compressed_data)
        else:
            raise ValueError(f"Unsupported compression: {compression}")
        
        # Deserialize based on file type
        if file_type in self.deserializers:
            # Write to temporary file for deserialization
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(decompressed)
                tmp_path = Path(tmp.name)
            
            try:
                data = self.deserializers[file_type](tmp_path)
            finally:
                tmp_path.unlink()
        else:
            data = decompressed
        
        return data
    
    def write(self, 
              data: Any,
              filepath: Union[str, Path],
              file_type: Optional[FileType] = None,
              compress: Optional[CompressionType] = None,
              cache: bool = True,
              metadata: Optional[Dict[str, Any]] = None):
        """Write data to file.
        
        Args:
            data: Data to write
            filepath: Path to write to
            file_type: File type (auto-detected from extension if None)
            compress: Compression type (None for auto)
            cache: Whether to update cache
            metadata: Optional metadata to store
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file type
        if file_type is None:
            file_type, detected_compression = self.detect_file_type(path)
            if compress is None and self.enable_compression:
                compress = detected_compression
        else:
            if compress is None and self.enable_compression:
                _, detected_compression = self.detect_file_type(path)
                compress = detected_compression
        
        # Serialize data
        if file_type in self.serializers:
            # Create temporary file for serialization
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = Path(tmp.name)
            
            try:
                # Serialize to temporary file
                self.serializers[file_type](data, tmp_path)
                
                # Read serialized data
                with open(tmp_path, 'rb') as f:
                    serialized = f.read()
            finally:
                tmp_path.unlink()
        else:
            # Assume data is already bytes
            if isinstance(data, bytes):
                serialized = data
            else:
                serialized = str(data).encode('utf-8')
        
        # Apply compression if requested
        if compress and compress != CompressionType.NONE:
            serialized = self._compress_data(serialized, compress)
            # Update extension if needed
            if compress == CompressionType.GZIP and not str(path).endswith('.gz'):
                path = path.with_suffix(path.suffix + '.gz')
            elif compress == CompressionType.BZIP2 and not str(path).endswith('.bz2'):
                path = path.with_suffix(path.suffix + '.bz2')
            elif compress == CompressionType.LZMA and not str(path).endswith('.xz'):
                path = path.with_suffix(path.suffix + '.xz')
            elif compress == CompressionType.LZ4 and not str(path).endswith('.lz4'):
                path = path.with_suffix(path.suffix + '.lz4')
            elif compress == CompressionType.ZSTD and not str(path).endswith('.zst'):
                path = path.with_suffix(path.suffix + '.zst')
        
        # Write to file
        with open(path, 'wb') as f:
            f.write(serialized)
        
        # Update statistics
        self.stats['writes'] += 1
        self.stats['bytes_written'] += len(serialized)
        if metadata:
            self._write_metadata(path, metadata)
        
        # Update cache
        if cache and self.cache:
            cache_key = str(path.absolute())
            size = self._estimate_size(data)
            self.cache.put(cache_key, data, size)
    
    def _compress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Compress data.
        
        Args:
            data: Data to compress
            compression: Compression type
            
        Returns:
            Compressed data
        """
        if compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=9)
        elif compression == CompressionType.BZIP2:
            return bz2.compress(data, compresslevel=9)
        elif compression == CompressionType.LZMA:
            return lzma.compress(data, preset=9)
        elif compression == CompressionType.LZ4:
            return lz4.frame.compress(data, compression_level=12)
        elif compression == CompressionType.ZSTD:
            cctx = zstd.ZstdCompressor(level=22)
            return cctx.compress(data)
        else:
            return data
    
    def _write_metadata(self, filepath: Path, metadata: Dict[str, Any]):
        """Write metadata to sidecar file.
        
        Args:
            filepath: Main file path
            metadata: Metadata to write
        """
        metadata_path = filepath.with_suffix(filepath.suffix + '.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def read_metadata(self, filepath: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read metadata from sidecar file.
        
        Args:
            filepath: Main file path
            
        Returns:
            Metadata dictionary or None
        """
        path = Path(filepath)
        metadata_path = path.with_suffix(path.suffix + '.meta.json')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def _serialize_json(self, data: Any, filepath: Path):
        """Serialize data as JSON.
        
        Args:
            data: Data to serialize
            filepath: Path to write to
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, (dict, list, tuple, str, int, float, bool, type(None))):
                json.dump(data, f, indent=2, default=str)
            else:
                # Try to convert to dict
                try:
                    data_dict = asdict(data) if hasattr(data, '__dataclass_fields__') else vars(data)
                    json.dump(data_dict, f, indent=2, default=str)
                except:
                    # Fallback to string representation
                    json.dump(str(data), f, indent=2)
    
    def _deserialize_json(self, filepath: Path) -> Any:
        """Deserialize JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Deserialized data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _serialize_yaml(self, data: Any, filepath: Path):
        """Serialize data as YAML.
        
        Args:
            data: Data to serialize
            filepath: Path to write to
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def _deserialize_yaml(self, filepath: Path) -> Any:
        """Deserialize YAML file.
        
        Args:
            filepath: Path to YAML file
            
        Returns:
            Deserialized data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _serialize_pickle(self, data: Any, filepath: Path):
        """Serialize data with pickle.
        
        Args:
            data: Data to serialize
            filepath: Path to write to
        """
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_pickle(self, filepath: Path) -> Any:
        """Deserialize pickle file.
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            Deserialized data
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _serialize_numpy(self, data: Any, filepath: Path):
        """Serialize data with NumPy.
        
        Args:
            data: Data to serialize
            filepath: Path to write to
        """
        if isinstance(data, np.ndarray):
            np.save(filepath, data, allow_pickle=False)
        elif isinstance(data, dict):
            np.savez_compressed(filepath, **data)
        else:
            # Convert to numpy array if possible
            try:
                np.save(filepath, np.array(data), allow_pickle=False)
            except:
                # Fallback to pickle
                self._serialize_pickle(data, filepath)
    
    def _deserialize_numpy(self, filepath: Path) -> Any:
        """Deserialize NumPy file.
        
        Args:
            filepath: Path to NumPy file
            
        Returns:
            Deserialized data
        """
        if filepath.suffix == '.npz':
            return dict(np.load(filepath, allow_pickle=False))
        else:
            return np.load(filepath, allow_pickle=False)
    
    def _serialize_torch(self, data: Any, filepath: Path):
        """Serialize data with PyTorch.
        
        Args:
            data: Data to serialize
            filepath: Path to write to
        """
        torch.save(data, filepath)
    
    def _deserialize_torch(self, filepath: Path) -> Any:
        """Deserialize PyTorch file.
        
        Args:
            filepath: Path to PyTorch file
            
        Returns:
            Deserialized data
        """
        return torch.load(filepath, map_location='cpu')
    
    def _serialize_text(self, data: Any, filepath: Path):
        """Serialize data as text.
        
        Args:
            data: Data to serialize
            filepath: Path to write to
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            if isinstance(data, (list, tuple)):
                for item in data:
                    f.write(str(item) + '\n')
            elif isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(str(data))
    
    def _deserialize_text(self, filepath: Path) -> str:
        """Deserialize text file.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Text content
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _serialize_image(self, data: Any, filepath: Path):
        """Serialize image data.
        
        Args:
            data: Image data (numpy array, PIL Image, etc.)
            filepath: Path to write to
        """
        if isinstance(data, np.ndarray):
            # Convert to PIL Image
            if data.dtype == np.float32 or data.dtype == np.float64:
                data = (data * 255).astype(np.uint8)
            
            if len(data.shape) == 3 and data.shape[2] == 3:
                mode = 'RGB'
            elif len(data.shape) == 3 and data.shape[2] == 4:
                mode = 'RGBA'
            elif len(data.shape) == 2:
                mode = 'L'
            else:
                raise ValueError(f"Unsupported image shape: {data.shape}")
            
            image = PIL.Image.fromarray(data, mode=mode)
            image.save(filepath)
        elif isinstance(data, PIL.Image.Image):
            data.save(filepath)
        else:
            raise TypeError(f"Unsupported image type: {type(data)}")
    
    def _deserialize_image(self, filepath: Path) -> np.ndarray:
        """Deserialize image file.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Image as numpy array
        """
        image = PIL.Image.open(filepath)
        return np.array(image)
    
    def _serialize_csv(self, data: Any, filepath: Path):
        """Serialize data as CSV.
        
        Args:
            data: Data to serialize (DataFrame, list of lists, dict)
            filepath: Path to write to
        """
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, list):
            # Assume list of lists or list of dicts
            if data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False)
            elif data and isinstance(data[0], (list, tuple)):
                df = pd.DataFrame(data)
                df.to_csv(filepath, index=False, header=False)
            else:
                with open(filepath, 'w') as f:
                    for item in data:
                        f.write(str(item) + '\n')
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
            df.to_csv(filepath, index=False)
        else:
            raise TypeError(f"Unsupported data type for CSV: {type(data)}")
    
    def _deserialize_csv(self, filepath: Path) -> pd.DataFrame:
        """Deserialize CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame
        """
        return pd.read_csv(filepath)
    
    def _serialize_hdf5(self, data: Any, filepath: Path):
        """Serialize data as HDF5.
        
        Args:
            data: Data to serialize (dict of arrays, single array)
            filepath: Path to write to
        """
        with h5py.File(filepath, 'w') as f:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value, compression='gzip')
                    elif isinstance(value, (int, float, str)):
                        f.attrs[key] = value
                    else:
                        # Try to convert to numpy array
                        try:
                            arr = np.array(value)
                            f.create_dataset(key, data=arr, compression='gzip')
                        except:
                            # Store as string
                            f.attrs[key] = str(value)
            elif isinstance(data, np.ndarray):
                f.create_dataset('data', data=data, compression='gzip')
            else:
                raise TypeError(f"Unsupported data type for HDF5: {type(data)}")
    
    def _deserialize_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """Deserialize HDF5 file.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            Dictionary of datasets and attributes
        """
        result = {}
        with h5py.File(filepath, 'r') as f:
            # Read datasets
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    result[key] = f[key][:]
            
            # Read attributes
            for key, value in f.attrs.items():
                result[key] = value
        
        return result
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes.
        
        Args:
            data: Data to estimate size of
            
        Returns:
            Estimated size in bytes
        """
        if isinstance(data, bytes):
            return len(data)
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        elif isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.element_size() * data.nelement()
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
        elif hasattr(data, '__sizeof__'):
            return sys.getsizeof(data)
        else:
            # Rough estimate
            return 1024  # 1KB default
    
    def batch_read(self, 
                   filepaths: List[Union[str, Path]],
                   file_type: Optional[FileType] = None,
                   cache: bool = True,
                   parallel: bool = True) -> List[Any]:
        """Read multiple files in batch.
        
        Args:
            filepaths: List of file paths
            file_type: Optional file type
            cache: Whether to use cache
            parallel: Whether to read in parallel
            
        Returns:
            List of file contents
        """
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.read, fp, file_type, cache)
                    for fp in filepaths
                ]
                return [future.result() for future in futures]
        else:
            return [self.read(fp, file_type, cache) for fp in filepaths]
    
    def batch_write(self, 
                    data_list: List[Any],
                    filepaths: List[Union[str, Path]],
                    file_type: Optional[FileType] = None,
                    compress: Optional[CompressionType] = None,
                    cache: bool = True,
                    parallel: bool = True):
        """Write multiple files in batch.
        
        Args:
            data_list: List of data to write
            filepaths: List of file paths
            file_type: Optional file type
            compress: Compression type
            cache: Whether to update cache
            parallel: Whether to write in parallel
        """
        if len(data_list) != len(filepaths):
            raise ValueError("data_list and filepaths must have same length")
        
        if parallel:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.write, data, fp, file_type, compress, cache)
                    for data, fp in zip(data_list, filepaths)
                ]
                # Wait for all to complete
                for future in futures:
                    future.result()
        else:
            for data, filepath in zip(data_list, filepaths):
                self.write(data, filepath, file_type, compress, cache)
    
    def copy(self, 
             src: Union[str, Path],
             dst: Union[str, Path],
             overwrite: bool = False):
        """Copy file with metadata preservation.
        
        Args:
            src: Source path
            dst: Destination path
            overwrite: Whether to overwrite existing file
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        
        # Copy main file
        shutil.copy2(src_path, dst_path)
        
        # Copy metadata if exists
        metadata_src = src_path.with_suffix(src_path.suffix + '.meta.json')
        if metadata_src.exists():
            metadata_dst = dst_path.with_suffix(dst_path.suffix + '.meta.json')
            shutil.copy2(metadata_src, metadata_dst)
        
        # Update cache
        if self.cache:
            src_key = str(src_path.absolute())
            dst_key = str(dst_path.absolute())
            
            if src_key in self.cache.cache:
                entry = self.cache.cache[src_key]
                self.cache.cache[dst_key] = CacheEntry(
                    data=entry.data,
                    timestamp=entry.timestamp,
                    size=entry.size,
                    access_count=0,
                    last_accessed=time.time()
                )
    
    def move(self, 
             src: Union[str, Path],
             dst: Union[str, Path],
             overwrite: bool = False):
        """Move file with cache update.
        
        Args:
            src: Source path
            dst: Destination path
            overwrite: Whether to overwrite existing file
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        
        # Move main file
        shutil.move(src_path, dst_path)
        
        # Move metadata if exists
        metadata_src = src_path.with_suffix(src_path.suffix + '.meta.json')
        if metadata_src.exists():
            metadata_dst = dst_path.with_suffix(dst_path.suffix + '.meta.json')
            shutil.move(metadata_src, metadata_dst)
        
        # Update cache
        if self.cache:
            src_key = str(src_path.absolute())
            dst_key = str(dst_path.absolute())
            
            if src_key in self.cache.cache:
                entry = self.cache.cache.pop(src_key)
                self.cache.cache[dst_key] = entry
                # Update access order
                if src_key in self.cache.access_order:
                    idx = self.cache.access_order.index(src_key)
                    self.cache.access_order[idx] = dst_key
    
    def delete(self, filepath: Union[str, Path], missing_ok: bool = False):
        """Delete file and its metadata.
        
        Args:
            filepath: Path to delete
            missing_ok: Whether to ignore missing files
        """
        path = Path(filepath)
        
        # Delete main file
        if path.exists():
            path.unlink()
        elif not missing_ok:
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Delete metadata if exists
        metadata_path = path.with_suffix(path.suffix + '.meta.json')
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from cache
        if self.cache:
            cache_key = str(path.absolute())
            self.cache.remove(cache_key)
    
    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check if file exists.
        
        Args:
            filepath: Path to check
            
        Returns:
            True if file exists
        """
        path = Path(filepath)
        return path.exists()
    
    def get_size(self, filepath: Union[str, Path]) -> int:
        """Get file size in bytes.
        
        Args:
            filepath: Path to file
            
        Returns:
            File size in bytes
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        return path.stat().st_size
    
    def list_files(self, 
                   directory: Union[str, Path],
                   pattern: str = "*",
                   recursive: bool = False,
                   include_dirs: bool = False) -> List[Path]:
        """List files in directory.
        
        Args:
            directory: Directory to list
            pattern: Glob pattern
            recursive: Whether to search recursively
            include_dirs: Whether to include directories
            
        Returns:
            List of file paths
        """
        dir_path = Path(directory)
        
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
        
        if not include_dirs:
            files = [f for f in files if f.is_file()]
        
        return files
    
    def find_files(self, 
                   directory: Union[str, Path],
                   patterns: List[str],
                   recursive: bool = True) -> List[Path]:
        """Find files matching patterns.
        
        Args:
            directory: Directory to search
            patterns: List of glob patterns
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        files = set()
        
        for pattern in patterns:
            if recursive:
                matches = set(dir_path.rglob(pattern))
            else:
                matches = set(dir_path.glob(pattern))
            files.update(matches)
        
        return sorted(files)
    
    def walk(self, 
             directory: Union[str, Path],
             topdown: bool = True,
             onerror: Optional[Callable] = None):
        """Walk directory tree.
        
        Args:
            directory: Root directory
            topdown: Whether to yield directories before contents
            onerror: Error handler
            
        Yields:
            (dirpath, dirnames, filenames) tuples
        """
        dir_path = Path(directory)
        for root, dirs, files in os.walk(dir_path, topdown=topdown, onerror=onerror):
            yield Path(root), [Path(d) for d in dirs], [Path(f) for f in files]
    
    def create_directory(self, 
                        directory: Union[str, Path],
                        parents: bool = True,
                        exist_ok: bool = True):
        """Create directory.
        
        Args:
            directory: Directory path
            parents: Whether to create parent directories
            exist_ok: Whether to ignore existing directory
        """
        path = Path(directory)
        path.mkdir(parents=parents, exist_ok=exist_ok)
    
    def delete_directory(self, 
                        directory: Union[str, Path],
                        recursive: bool = True):
        """Delete directory.
        
        Args:
            directory: Directory path
            recursive: Whether to delete recursively
        """
        path = Path(directory)
        
        if recursive:
            shutil.rmtree(path)
        else:
            path.rmdir()
        
        # Clean cache entries for files in this directory
        if self.cache:
            dir_str = str(path.absolute())
            keys_to_remove = [
                key for key in self.cache.cache.keys()
                if key.startswith(dir_str)
            ]
            for key in keys_to_remove:
                self.cache.remove(key)
    
    def get_temp_file(self, 
                     suffix: str = None,
                     prefix: str = None,
                     dir: Union[str, Path] = None) -> Path:
        """Get temporary file path.
        
        Args:
            suffix: File suffix
            prefix: File prefix
            dir: Directory for temp file
            
        Returns:
            Path to temporary file
        """
        if dir:
            dir = Path(dir)
            dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            prefix=prefix,
            dir=dir,
            delete=False
        ) as f:
            return Path(f.name)
    
    def get_temp_directory(self, 
                          suffix: str = None,
                          prefix: str = None,
                          dir: Union[str, Path] = None) -> Path:
        """Get temporary directory path.
        
        Args:
            suffix: Directory suffix
            prefix: Directory prefix
            dir: Parent directory
            
        Returns:
            Path to temporary directory
        """
        if dir:
            dir = Path(dir)
            dir.mkdir(parents=True, exist_ok=True)
        
        return Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir))
    
    def archive(self, 
               source: Union[str, Path],
               archive_path: Union[str, Path],
               format: str = 'zip',
               compression: str = 'deflate'):
        """Create archive of files/directory.
        
        Args:
            source: Source path (file or directory)
            archive_path: Archive output path
            format: Archive format ('zip', 'tar', 'gztar', 'bztar', 'xztar')
            compression: Compression level for zip
        """
        source_path = Path(source)
        archive_path = Path(archive_path)
        
        if format == 'zip':
            with zipfile.ZipFile(archive_path, 'w', compression=getattr(zipfile, compression.upper())) as zf:
                if source_path.is_file():
                    zf.write(source_path, source_path.name)
                else:
                    for root, dirs, files in os.walk(source_path):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(source_path.parent if source_path.is_file() else source_path)
                            zf.write(file_path, arcname)
        else:
            # Use shutil for other formats
            shutil.make_archive(
                str(archive_path.with_suffix('')),
                format,
                root_dir=source_path.parent if source_path.is_file() else source_path,
                base_dir=source_path.name if source_path.is_file() else '.'
            )
    
    def extract(self, 
               archive_path: Union[str, Path],
               extract_dir: Union[str, Path],
               format: str = None):
        """Extract archive.
        
        Args:
            archive_path: Archive path
            extract_dir: Extraction directory
            format: Archive format (auto-detected if None)
        """
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        if format is None:
            # Auto-detect format
            if archive_path.suffix == '.zip':
                format = 'zip'
            elif archive_path.suffix in ['.tar', '.gz', '.bz2', '.xz']:
                format = 'tar'
            else:
                raise ValueError(f"Cannot detect archive format: {archive_path}")
        
        if format == 'zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        else:
            shutil.unpack_archive(archive_path, extract_dir, format)
    
    def sync(self, 
            source: Union[str, Path],
            destination: Union[str, Path],
            overwrite: bool = False,
            delete: bool = False):
        """Sync files from source to destination.
        
        Args:
            source: Source directory
            destination: Destination directory
            overwrite: Whether to overwrite existing files
            delete: Whether to delete extra files in destination
        """
        source_path = Path(source)
        dest_path = Path(destination)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        dest_path.mkdir(parents=True, exist_ok=True)
        
        # Walk source directory
        for root, dirs, files in os.walk(source_path):
            rel_path = Path(root).relative_to(source_path)
            dest_dir = dest_path / rel_path
            
            # Create destination directory
            dest_dir.mkdir(exist_ok=True)
            
            # Copy files
            for file in files:
                src_file = Path(root) / file
                dst_file = dest_dir / file
                
                if not dst_file.exists() or overwrite:
                    # Check if file needs updating
                    if dst_file.exists():
                        src_mtime = src_file.stat().st_mtime
                        dst_mtime = dst_file.stat().st_mtime
                        if src_mtime <= dst_mtime and not overwrite:
                            continue
                    
                    self.copy(src_file, dst_file, overwrite=True)
        
        # Delete extra files if requested
        if delete:
            for root, dirs, files in os.walk(dest_path):
                rel_path = Path(root).relative_to(dest_path)
                src_dir = source_path / rel_path
                
                if not src_dir.exists():
                    # Delete entire directory
                    shutil.rmtree(root)
                    continue
                
                # Check files in destination
                for file in files:
                    dst_file = Path(root) / file
                    src_file = src_dir / file
                    
                    if not src_file.exists():
                        dst_file.unlink()
                
                # Check directories in destination
                for dir in dirs:
                    dst_dir = Path(root) / dir
                    src_dir_sub = src_dir / dir
                    
                    if not src_dir_sub.exists():
                        shutil.rmtree(dst_dir)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get file manager statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            stats['cache'] = cache_stats
            stats['cache_hit_rate'] = self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        
        stats['compression_enabled'] = self.enable_compression
        stats['base_path'] = str(self.base_path)
        
        return stats
    
    def clear_cache(self):
        """Clear file cache."""
        if self.cache:
            self.cache.clear()
        
        # Clear statistics
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0
    
    def cleanup(self):
        """Cleanup resources."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.clear_cache()
        gc.collect()
    
    @contextmanager
    def atomic_write(self, 
                    filepath: Union[str, Path],
                    file_type: Optional[FileType] = None,
                    compress: Optional[CompressionType] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """Context manager for atomic file writing.
        
        Args:
            filepath: Target file path
            file_type: File type
            compress: Compression type
            metadata: Optional metadata
            
        Yields:
            Write function for data
        """
        path = Path(filepath)
        temp_file = self.get_temp_file(suffix=path.suffix, dir=path.parent)
        
        def write_func(data):
            self.write(data, temp_file, file_type, compress, cache=False, metadata=metadata)
        
        try:
            yield write_func
            # Atomically replace target file
            temp_file.replace(path)
            
            # Update cache
            if self.cache:
                cache_key = str(path.absolute())
                # We don't have the data here, so remove from cache
                # The next read will populate it
                self.cache.remove(cache_key)
                
        except Exception:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class DataLoader:
    """Efficient data loader with preprocessing and batching."""
    
    def __init__(self, 
                 file_manager: FileManager,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 drop_last: bool = False,
                 collate_fn: Optional[Callable] = None):
        """Initialize data loader.
        
        Args:
            file_manager: File manager instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker threads
            prefetch_factor: Prefetch factor for workers
            drop_last: Whether to drop last incomplete batch
            collate_fn: Custom collate function
        """
        self.file_manager = file_manager
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.collate_fn = collate_fn or self.default_collate
        
        self.filepaths: List[Path] = []
        self.indices: List[int] = []
        self.current_index = 0
        
        # Worker pool
        self.worker_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.prefetch_queue = Queue(maxsize=prefetch_factor * batch_size)
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'total_batches': 0,
            'current_epoch': 0,
            'samples_seen': 0,
            'worker_utilization': 0.0,
        }
    
    def add_files(self, filepaths: List[Union[str, Path]]):
        """Add files to loader.
        
        Args:
            filepaths: List of file paths
        """
        new_paths = [Path(fp) for fp in filepaths]
        self.filepaths.extend(new_paths)
        self.stats['total_files'] = len(self.filepaths)
        
        # Update indices
        self.indices = list(range(len(self.filepaths)))
        
        # Reset state
        self.current_index = 0
        self.stats['total_batches'] = len(self) if self.batch_size > 0 else 0
    
    def add_directory(self, 
                     directory: Union[str, Path],
                     pattern: str = "*",
                     recursive: bool = True):
        """Add files from directory.
        
        Args:
            directory: Directory path
            pattern: Glob pattern
            recursive: Whether to search recursively
        """
        files = self.file_manager.list_files(directory, pattern, recursive)
        self.add_files(files)
    
    def set_transforms(self, 
                      transform: Optional[Callable] = None,
                      target_transform: Optional[Callable] = None):
        """Set data transforms.
        
        Args:
            transform: Input transform function
            target_transform: Target transform function
        """
        self.transform = transform
        self.target_transform = target_transform
    
    def _worker_fn(self, indices_batch: List[int]) -> Any:
        """Worker function for loading data.
        
        Args:
            indices_batch: Batch of indices
            
        Returns:
            Loaded and processed data
        """
        batch_data = []
        
        for idx in indices_batch:
            if idx >= len(self.filepaths):
                continue
            
            filepath = self.filepaths[idx]
            
            try:
                # Load data
                data = self.file_manager.read(filepath, cache=True)
                
                # Apply transforms if provided
                if self.transform:
                    if isinstance(data, tuple):
                        # Assume (input, target) tuple
                        inputs, targets = data
                        inputs = self.transform(inputs)
                        if self.target_transform:
                            targets = self.target_transform(targets)
                        data = (inputs, targets)
                    else:
                        data = self.transform(data)
                
                batch_data.append(data)
                
            except Exception as e:
                self.file_manager.logger.warning(f"Failed to load {filepath}: {e}")
                continue
        
        # Collate batch
        return self.collate_fn(batch_data)
    
    def _prefetch_worker(self):
        """Prefetch worker for async loading."""
        while True:
            # Get next batch indices
            batch_indices = self._get_next_batch_indices()
            if batch_indices is None:
                break
            
            # Submit to worker pool
            future = self.worker_pool.submit(self._worker_fn, batch_indices)
            self.prefetch_queue.put(future)
    
    def _get_next_batch_indices(self) -> Optional[List[int]]:
        """Get indices for next batch.
        
        Returns:
            List of indices or None if done
        """
        with threading.Lock():
            if self.current_index >= len(self.indices):
                return None
            
            start = self.current_index
            end = min(start + self.batch_size, len(self.indices))
            
            # Check if we should drop last incomplete batch
            if self.drop_last and (end - start) < self.batch_size:
                return None
            
            batch_indices = [self.indices[i] for i in range(start, end)]
            self.current_index = end
            
            return batch_indices
    
    def __iter__(self):
        """Iterator for data loader.
        
        Returns:
            Iterator yielding batches
        """
        # Reset for new epoch
        self.current_index = 0
        
        # Shuffle if requested
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        # Start prefetch workers
        self.prefetch_queue = Queue(maxsize=self.prefetch_factor * self.batch_size)
        prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        prefetch_thread.start()
        
        return self
    
    def __next__(self) -> Any:
        """Get next batch.
        
        Returns:
            Batch data
        """
        # Check if we have more data
        if self.current_index >= len(self.indices) and self.prefetch_queue.empty():
            self.stats['current_epoch'] += 1
            raise StopIteration
        
        try:
            # Get next batch from prefetch queue
            future = self.prefetch_queue.get(timeout=30.0)
            batch = future.result()
            
            # Update statistics
            self.stats['samples_seen'] += len(batch) if hasattr(batch, '__len__') else 1
            
            return batch
            
        except Exception as e:
            self.file_manager.logger.error(f"Error getting next batch: {e}")
            raise StopIteration
    
    def __len__(self) -> int:
        """Get number of batches.
        
        Returns:
            Number of batches
        """
        if self.batch_size <= 0:
            return 0
        
        num_samples = len(self.indices)
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size
    
    @staticmethod
    def default_collate(batch: List[Any]) -> Any:
        """Default collate function.
        
        Args:
            batch: List of samples
            
        Returns:
            Collated batch
        """
        elem = batch[0]
        
        if isinstance(elem, torch.Tensor):
            return torch.stack(batch, dim=0)
        elif isinstance(elem, np.ndarray):
            return np.stack(batch, axis=0)
        elif isinstance(elem, (int, float)):
            return torch.tensor(batch)
        elif isinstance(elem, (list, tuple)):
            # Recursively collate
            transposed = list(zip(*batch))
            return type(elem)([DataLoader.default_collate(samples) for samples in transposed])
        elif isinstance(elem, dict):
            return {key: DataLoader.default_collate([d[key] for d in batch]) for key in elem}
        else:
            return batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data loader statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        stats['batch_size'] = self.batch_size
        stats['shuffle'] = self.shuffle
        stats['num_workers'] = self.num_workers
        stats['prefetch_factor'] = self.prefetch_factor
        stats['drop_last'] = self.drop_last
        
        return stats
    
    def cleanup(self):
        """Cleanup resources."""
        self.worker_pool.shutdown(wait=True)
        self.prefetch_queue = Queue()


# Convenience functions
def save_json(data: Any, 
             filepath: Union[str, Path],
             indent: int = 2,
             sort_keys: bool = False,
             **kwargs):
    """Save data as JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save to
        indent: JSON indentation
        sort_keys: Whether to sort keys
        **kwargs: Additional kwargs for json.dump
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys, default=str, **kwargs)


def load_json(filepath: Union[str, Path], **kwargs) -> Any:
    """Load JSON file.
    
    Args:
        filepath: Path to JSON file
        **kwargs: Additional kwargs for json.load
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f, **kwargs)


def save_pickle(data: Any, 
               filepath: Union[str, Path],
               protocol: int = pickle.HIGHEST_PROTOCOL):
    """Save data with pickle.
    
    Args:
        data: Data to save
        filepath: Path to save to
        protocol: Pickle protocol
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)


def load_pickle(filepath: Union[str, Path], **kwargs) -> Any:
    """Load pickle file.
    
    Args:
        filepath: Path to pickle file
        **kwargs: Additional kwargs for pickle.load
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f, **kwargs)


def save_numpy(data: np.ndarray, 
              filepath: Union[str, Path],
              allow_pickle: bool = False,
              fix_imports: bool = True):
    """Save numpy array.
    
    Args:
        data: Array to save
        filepath: Path to save to
        allow_pickle: Whether to allow pickling
        fix_imports: Whether to fix imports for Python 2 compatibility
    """
    np.save(filepath, data, allow_pickle=allow_pickle, fix_imports=fix_imports)


def load_numpy(filepath: Union[str, Path], 
              allow_pickle: bool = False,
              fix_imports: bool = True) -> np.ndarray:
    """Load numpy array.
    
    Args:
        filepath: Path to numpy file
        allow_pickle: Whether to allow pickling
        fix_imports: Whether to fix imports for Python 2 compatibility
        
    Returns:
        Loaded array
    """
    return np.load(filepath, allow_pickle=allow_pickle, fix_imports=fix_imports)


def save_image(data: Union[np.ndarray, PIL.Image.Image], 
              filepath: Union[str, Path],
              quality: int = 95,
              **kwargs):
    """Save image.
    
    Args:
        data: Image data
        filepath: Path to save to
        quality: Image quality (for JPEG)
        **kwargs: Additional kwargs for PIL.Image.save
    """
    if isinstance(data, np.ndarray):
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        
        if len(data.shape) == 3 and data.shape[2] == 3:
            mode = 'RGB'
        elif len(data.shape) == 3 and data.shape[2] == 4:
            mode = 'RGBA'
        elif len(data.shape) == 2:
            mode = 'L'
        else:
            raise ValueError(f"Unsupported image shape: {data.shape}")
        
        image = PIL.Image.fromarray(data, mode=mode)
    else:
        image = data
    
    image.save(filepath, quality=quality, **kwargs)


def load_image(filepath: Union[str, Path], 
              mode: str = None,
              **kwargs) -> np.ndarray:
    """Load image.
    
    Args:
        filepath: Path to image file
        mode: PIL mode (None for auto)
        **kwargs: Additional kwargs for PIL.Image.open
        
    Returns:
        Image as numpy array
    """
    image = PIL.Image.open(filepath, **kwargs)
    if mode:
        image = image.convert(mode)
    return np.array(image)


def save_video(frames: List[np.ndarray], 
              filepath: Union[str, Path],
              fps: int = 30,
              codec: str = 'mp4v',
              **kwargs):
    """Save video from frames.
    
    Args:
        frames: List of video frames
        filepath: Path to save to
        fps: Frames per second
        codec: Video codec
        **kwargs: Additional kwargs for VideoWriter
    """
    if not HAS_IMAGE:
        raise ImportError("OpenCV required for video saving")
    
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))
    
    for frame in frames:
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame)
    
    out.release()


def load_video(filepath: Union[str, Path],
              start_frame: int = 0,
              end_frame: Optional[int] = None,
              step: int = 1) -> List[np.ndarray]:
    """Load video as frames.
    
    Args:
        filepath: Path to video file
        start_frame: Starting frame index
        end_frame: Ending frame index (None for all)
        step: Frame step
        
    Returns:
        List of video frames
    """
    if not HAS_IMAGE:
        raise ImportError("OpenCV required for video loading")
    
    cap = cv2.VideoCapture(str(filepath))
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= start_frame and (end_frame is None or frame_idx < end_frame):
            if (frame_idx - start_frame) % step == 0:
                # Convert BGR to RGB
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        frame_idx += 1
        if end_frame is not None and frame_idx >= end_frame:
            break
    
    cap.release()
    return frames


def compress_data(data: bytes, 
                 compression: Union[str, CompressionType] = 'gzip',
                 level: int = 9) -> bytes:
    """Compress data.
    
    Args:
        data: Data to compress
        compression: Compression type
        level: Compression level
        
    Returns:
        Compressed data
    """
    if isinstance(compression, str):
        compression = CompressionType[compression.upper()]
    
    if compression == CompressionType.GZIP:
        return gzip.compress(data, compresslevel=level)
    elif compression == CompressionType.BZIP2:
        return bz2.compress(data, compresslevel=level)
    elif compression == CompressionType.LZMA:
        return lzma.compress(data, preset=level)
    elif compression == CompressionType.LZ4:
        return lz4.frame.compress(data, compression_level=level)
    elif compression == CompressionType.ZSTD:
        cctx = zstd.ZstdCompressor(level=level)
        return cctx.compress(data)
    else:
        return data


def decompress_data(data: bytes, 
                   compression: Union[str, CompressionType] = 'gzip') -> bytes:
    """Decompress data.
    
    Args:
        data: Compressed data
        compression: Compression type
        
    Returns:
        Decompressed data
    """
    if isinstance(compression, str):
        compression = CompressionType[compression.upper()]
    
    if compression == CompressionType.GZIP:
        return gzip.decompress(data)
    elif compression == CompressionType.BZIP2:
        return bz2.decompress(data)
    elif compression == CompressionType.LZMA:
        return lzma.decompress(data)
    elif compression == CompressionType.LZ4:
        return lz4.frame.decompress(data)
    elif compression == CompressionType.ZSTD:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    else:
        return data


def calculate_checksum(filepath: Union[str, Path], 
                      algorithm: str = 'sha256',
                      chunk_size: int = 4096) -> str:
    """Calculate file checksum.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm
        chunk_size: Read chunk size
        
    Returns:
        Checksum string
    """
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def verify_checksum(filepath: Union[str, Path], 
                   expected_checksum: str,
                   algorithm: str = 'sha256') -> bool:
    """Verify file checksum.
    
    Args:
        filepath: Path to file
        expected_checksum: Expected checksum
        algorithm: Hash algorithm
        
    Returns:
        True if checksum matches
    """
    actual = calculate_checksum(filepath, algorithm)
    return actual == expected_checksum


def get_file_info(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Get file information.
    
    Args:
        filepath: Path to file
        
    Returns:
        File information dictionary
    """
    path = Path(filepath)
    stat = path.stat()
    
    return {
        'path': str(path.absolute()),
        'size': stat.st_size,
        'created': datetime.fromtimestamp(stat.st_ctime),
        'modified': datetime.fromtimestamp(stat.st_mtime),
        'accessed': datetime.fromtimestamp(stat.st_atime),
        'is_file': path.is_file(),
        'is_dir': path.is_dir(),
        'is_symlink': path.is_symlink(),
        'suffix': path.suffix,
        'stem': path.stem,
        'name': path.name,
        'parent': str(path.parent),
    }


def create_temp_file(suffix: str = None,
                    prefix: str = None,
                    dir: Union[str, Path] = None,
                    delete: bool = True) -> Path:
    """Create temporary file.
    
    Args:
        suffix: File suffix
        prefix: File prefix
        dir: Directory for temp file
        delete: Whether to delete on close
        
    Returns:
        Path to temporary file
    """
    if dir:
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix=prefix,
        dir=dir,
        delete=delete
    ) as f:
        if delete:
            return Path(f.name)
        else:
            f.close()
            return Path(f.name)


def create_temp_dir(suffix: str = None,
                   prefix: str = None,
                   dir: Union[str, Path] = None) -> Path:
    """Create temporary directory.
    
    Args:
        suffix: Directory suffix
        prefix: Directory prefix
        dir: Parent directory
        
    Returns:
        Path to temporary directory
    """
    if dir:
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
    
    return Path(tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=dir))


@contextmanager
def atomic_save(filepath: Union[str, Path], mode: str = 'wb'):
    """Context manager for atomic file saving.
    
    Args:
        filepath: Target file path
        mode: File mode
        
    Yields:
        File object for writing
    """
    path = Path(filepath)
    temp_file = create_temp_file(suffix=path.suffix, dir=path.parent, delete=False)
    
    try:
        with open(temp_file, mode) as f:
            yield f
        
        # Atomically replace target file
        temp_file.replace(path)
        
    except Exception:
        # Clean up temp file on error
        if temp_file.exists():
            temp_file.unlink()
        raise
