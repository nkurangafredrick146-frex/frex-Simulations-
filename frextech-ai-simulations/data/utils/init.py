"""
Data Utilities Module
Collection of utilities for data processing, validation, and management.
"""

# Core utilities
from .logging_config import setup_logger, get_logger, log_execution_time
from .file_io import (
    read_json, write_json, read_yaml, write_yaml,
    read_csv, write_csv, read_pickle, write_pickle,
    load_image, save_image, load_video, save_video,
    load_numpy, save_numpy, load_tensor, save_tensor,
    ensure_directory, list_files, find_files, get_file_size,
    calculate_md5, copy_file, move_file, delete_file,
    compress_file, decompress_file
)

from .data_validator import DataValidator, ValidationResult, ValidationRule
from .data_preprocessor import DataPreprocessor, PreprocessingConfig
from .embedding_manager import EmbeddingManager, EmbeddingConfig
from .data_augmentor import DataAugmentor, AugmentationConfig
from .data_analyzer import DataAnalyzer, DataStatistics

# Version
__version__ = "1.0.0"
__author__ = "FrexTech AI Team"

# Exports
__all__ = [
    # Logging
    "setup_logger", "get_logger", "log_execution_time",
    
    # File I/O
    "read_json", "write_json", "read_yaml", "write_yaml",
    "read_csv", "write_csv", "read_pickle", "write_pickle",
    "load_image", "save_image", "load_video", "save_video",
    "load_numpy", "save_numpy", "load_tensor", "save_tensor",
    "ensure_directory", "list_files", "find_files", "get_file_size",
    "calculate_md5", "copy_file", "move_file", "delete_file",
    "compress_file", "decompress_file",
    
    # Data Processing
    "DataValidator", "ValidationResult", "ValidationRule",
    "DataPreprocessor", "PreprocessingConfig",
    "EmbeddingManager", "EmbeddingConfig",
    "DataAugmentor", "AugmentationConfig",
    "DataAnalyzer", "DataStatistics",
]

# Convenience functions
def validate_data_path(path: str, required_extensions: list = None) -> bool:
    """Validate data path exists and has required extensions."""
    from pathlib import Path
    path_obj = Path(path)
    
    if not path_obj.exists():
        return False
    
    if required_extensions:
        if path_obj.is_file():
            return path_obj.suffix.lower() in required_extensions
        elif path_obj.is_dir():
            files = list(path_obj.rglob('*'))
            return any(f.suffix.lower() in required_extensions for f in files)
    
    return True

def get_data_type(path: str) -> str:
    """Infer data type from file extension."""
    from pathlib import Path
    suffix = Path(path).suffix.lower()
    
    image_exts = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff']
    video_exts = ['.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv']
    audio_exts = ['.wav', '.mp3', '.flac', '.aac', '.ogg']
    text_exts = ['.txt', '.json', '.xml', '.csv', '.md']
    model_exts = ['.pt', '.pth', '.ckpt', '.onnx', '.h5']
    point_cloud_exts = ['.ply', '.obj', '.stl', '.xyz']
    mesh_exts = ['.obj', '.glb', '.fbx', '.dae']
    
    if suffix in image_exts:
        return 'image'
    elif suffix in video_exts:
        return 'video'
    elif suffix in audio_exts:
        return 'audio'
    elif suffix in text_exts:
        return 'text'
    elif suffix in model_exts:
        return 'model'
    elif suffix in point_cloud_exts:
        return 'point_cloud'
    elif suffix in mesh_exts:
        return 'mesh'
    else:
        return 'unknown'

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def calculate_directory_size(directory: str) -> int:
    """Calculate total size of directory in bytes."""
    from pathlib import Path
    total_size = 0
    directory_path = Path(directory)
    
    for file_path in directory_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size