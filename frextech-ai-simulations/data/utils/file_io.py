"""
File I/O Utilities Module
Comprehensive utilities for reading, writing, and managing files of various formats.
"""

import os
import json
import pickle
import csv
import shutil
import hashlib
import zipfile
import tarfile
import gzip
import bz2
import lzma
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, BinaryIO, TextIO
from datetime import datetime
import warnings

import yaml
import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
import cv2
import pandas as pd
from tqdm import tqdm

from .logging_config import get_logger

logger = get_logger(__name__)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Supported file extensions
SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
SUPPORTED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.webm', '.mkv', '.flv', '.wmv'}
SUPPORTED_AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
SUPPORTED_TEXT_EXTS = {'.txt', '.json', '.yaml', '.yml', '.csv', '.xml', '.md'}
SUPPORTED_MODEL_EXTS = {'.pt', '.pth', '.ckpt', '.onnx', '.h5', '.hdf5'}
SUPPORTED_POINT_CLOUD_EXTS = {'.ply', '.obj', '.stl', '.xyz', '.pcd'}
SUPPORTED_MESH_EXTS = {'.obj', '.glb', '.fbx', '.dae', '.stl'}


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
    sort_by: str = "name",
    reverse: bool = False
) -> List[Path]:
    """
    List files in directory with optional filtering and sorting.
    
    Args:
        directory: Directory path
        pattern: File pattern to match
        recursive: Whether to search recursively
        sort_by: Sort by 'name', 'size', 'mtime', or 'ctime'
        reverse: Whether to reverse sort order
        
    Returns:
        List of file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))
    
    # Filter out directories
    files = [f for f in files if f.is_file()]
    
    # Sort files
    if sort_by == "name":
        files.sort(key=lambda x: x.name, reverse=reverse)
    elif sort_by == "size":
        files.sort(key=lambda x: x.stat().st_size, reverse=reverse)
    elif sort_by == "mtime":
        files.sort(key=lambda x: x.stat().st_mtime, reverse=reverse)
    elif sort_by == "ctime":
        files.sort(key=lambda x: x.stat().st_ctime, reverse=reverse)
    
    return files


def find_files(
    directory: Union[str, Path],
    extensions: List[str],
    recursive: bool = True,
    case_sensitive: bool = False
) -> List[Path]:
    """
    Find files with specified extensions.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (with or without dot)
        recursive: Whether to search recursively
        case_sensitive: Whether extension matching is case-sensitive
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    # Normalize extensions
    normalized_exts = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        if not case_sensitive:
            ext = ext.lower()
        normalized_exts.append(ext)
    
    # Find files
    all_files = []
    if recursive:
        all_files = list(directory.rglob('*'))
    else:
        all_files = list(directory.glob('*'))
    
    # Filter by extension
    matching_files = []
    for file_path in all_files:
        if file_path.is_file():
            suffix = file_path.suffix
            if not case_sensitive:
                suffix = suffix.lower()
            
            if suffix in normalized_exts:
                matching_files.append(file_path)
    
    return matching_files


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return 0
    
    return file_path.stat().st_size


def calculate_md5(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """
    Calculate MD5 hash of file.
    
    Args:
        file_path: Path to file
        chunk_size: Chunk size for reading
        
    Returns:
        MD5 hash string
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_md5 = hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False,
    preserve_metadata: bool = False
) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing file
        preserve_metadata: Whether to preserve file metadata
        
    Returns:
        True if successful, False otherwise
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        logger.error(f"Source file does not exist: {src_path}")
        return False
    
    if dst_path.exists() and not overwrite:
        logger.warning(f"Destination file exists and overwrite=False: {dst_path}")
        return False
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if preserve_metadata:
            shutil.copy2(src_path, dst_path)
        else:
            shutil.copy(src_path, dst_path)
        
        logger.debug(f"Copied {src_path} to {dst_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to copy {src_path} to {dst_path}: {e}")
        return False


def move_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False
) -> bool:
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        logger.error(f"Source file does not exist: {src_path}")
        return False
    
    if dst_path.exists() and not overwrite:
        logger.warning(f"Destination file exists and overwrite=False: {dst_path}")
        return False
    
    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.move(src_path, dst_path)
        logger.debug(f"Moved {src_path} to {dst_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to move {src_path} to {dst_path}: {e}")
        return False


def delete_file(file_path: Union[str, Path]) -> bool:
    """
    Delete a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    try:
        file_path.unlink()
        logger.debug(f"Deleted file: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return False


def compress_file(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    compression: str = "gzip",
    compression_level: int = 6
) -> bool:
    """
    Compress a file.
    
    Args:
        input_file: Input file path
        output_file: Output file path (None for automatic naming)
        compression: Compression type: 'gzip', 'bz2', 'lzma', 'zip', 'tar'
        compression_level: Compression level (1-9 for gzip/bz2)
        
    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return False
    
    if output_file is None:
        if compression in ["gzip", "bz2", "lzma"]:
            output_path = input_path.with_suffix(input_path.suffix + f".{compression}")
        elif compression == "zip":
            output_path = input_path.with_suffix(".zip")
        elif compression == "tar":
            output_path = input_path.with_suffix(".tar.gz")
        else:
            logger.error(f"Unsupported compression type: {compression}")
            return False
    else:
        output_path = Path(output_file)
    
    try:
        if compression == "gzip":
            with open(input_path, 'rb') as f_in:
                with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == "bz2":
            with open(input_path, 'rb') as f_in:
                with bz2.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == "lzma":
            with open(input_path, 'rb') as f_in:
                with lzma.open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == "zip":
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(input_path, arcname=input_path.name)
        
        elif compression == "tar":
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(input_path, arcname=input_path.name)
        
        else:
            logger.error(f"Unsupported compression type: {compression}")
            return False
        
        logger.debug(f"Compressed {input_path} to {output_path} using {compression}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to compress {input_path}: {e}")
        return False


def decompress_file(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    compression: Optional[str] = None
) -> bool:
    """
    Decompress a file.
    
    Args:
        input_file: Input file path
        output_dir: Output directory (None for same directory as input)
        compression: Compression type (None for auto-detection)
        
    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        return False
    
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect compression type from extension
    if compression is None:
        suffix = input_path.suffix.lower()
        if suffix == '.gz' or suffix == '.gzip':
            compression = 'gzip'
        elif suffix == '.bz2':
            compression = 'bz2'
        elif suffix == '.xz' or suffix == '.lzma':
            compression = 'lzma'
        elif suffix == '.zip':
            compression = 'zip'
        elif suffix in ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2']:
            compression = 'tar'
        else:
            logger.error(f"Could not detect compression type from extension: {suffix}")
            return False
    
    try:
        if compression == "gzip":
            output_path = output_dir / input_path.stem
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == "bz2":
            output_path = output_dir / input_path.stem
            with bz2.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == "lzma":
            output_path = output_dir / input_path.stem
            with lzma.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        elif compression == "zip":
            with zipfile.ZipFile(input_path, 'r') as zipf:
                zipf.extractall(output_dir)
        
        elif compression == "tar":
            with tarfile.open(input_path, 'r:*') as tar:
                tar.extractall(output_dir)
        
        else:
            logger.error(f"Unsupported compression type: {compression}")
            return False
        
        logger.debug(f"Decompressed {input_path} to {output_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to decompress {input_path}: {e}")
        return False


def read_json(file_path: Union[str, Path], **kwargs) -> Any:
    """
    Read JSON file.
    
    Args:
        file_path: Path to JSON file
        **kwargs: Additional arguments for json.load
        
    Returns:
        Parsed JSON data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f, **kwargs)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read JSON file {file_path}: {e}")
        raise


def write_json(
    data: Any,
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
    **kwargs
) -> bool:
    """
    Write data to JSON file.
    
    Args:
        data: Data to write
        file_path: Path to output file
        indent: Indentation level
        ensure_ascii: Whether to ensure ASCII output
        **kwargs: Additional arguments for json.dump
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
        logger.debug(f"Wrote JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}: {e}")
        return False


def read_yaml(file_path: Union[str, Path], **kwargs) -> Any:
    """
    Read YAML file.
    
    Args:
        file_path: Path to YAML file
        **kwargs: Additional arguments for yaml.safe_load
        
    Returns:
        Parsed YAML data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f, **kwargs)
        return data
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read YAML file {file_path}: {e}")
        raise


def write_yaml(
    data: Any,
    file_path: Union[str, Path],
    default_flow_style: bool = False,
    **kwargs
) -> bool:
    """
    Write data to YAML file.
    
    Args:
        data: Data to write
        file_path: Path to output file
        default_flow_style: YAML flow style
        **kwargs: Additional arguments for yaml.dump
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=default_flow_style, **kwargs)
        logger.debug(f"Wrote YAML to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write YAML to {file_path}: {e}")
        return False


def read_csv(
    file_path: Union[str, Path],
    delimiter: str = ',',
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Read CSV file.
    
    Args:
        file_path: Path to CSV file
        delimiter: Field delimiter
        **kwargs: Additional arguments for csv.DictReader
        
    Returns:
        List of dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter=delimiter, **kwargs)
            data = list(reader)
        return data
    except Exception as e:
        logger.error(f"Failed to read CSV file {file_path}: {e}")
        raise


def write_csv(
    data: List[Dict[str, Any]],
    file_path: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ',',
    **kwargs
) -> bool:
    """
    Write data to CSV file.
    
    Args:
        data: List of dictionaries to write
        file_path: Path to output file
        fieldnames: List of field names (None to infer from data)
        delimiter: Field delimiter
        **kwargs: Additional arguments for csv.DictWriter
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    if not data:
        logger.warning("No data to write to CSV")
        return False
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get fieldnames if not provided
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, **kwargs)
            writer.writeheader()
            writer.writerows(data)
        
        logger.debug(f"Wrote CSV to {file_path} with {len(data)} rows")
        return True
    except Exception as e:
        logger.error(f"Failed to write CSV to {file_path}: {e}")
        return False


def read_pickle(file_path: Union[str, Path], **kwargs) -> Any:
    """
    Read pickle file.
    
    Args:
        file_path: Path to pickle file
        **kwargs: Additional arguments for pickle.load
        
    Returns:
        Unpickled data
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, **kwargs)
        return data
    except Exception as e:
        logger.error(f"Failed to read pickle file {file_path}: {e}")
        raise


def write_pickle(
    data: Any,
    file_path: Union[str, Path],
    protocol: int = pickle.HIGHEST_PROTOCOL,
    **kwargs
) -> bool:
    """
    Write data to pickle file.
    
    Args:
        data: Data to pickle
        file_path: Path to output file
        protocol: Pickle protocol
        **kwargs: Additional arguments for pickle.dump
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=protocol, **kwargs)
        logger.debug(f"Wrote pickle to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write pickle to {file_path}: {e}")
        return False


def load_image(
    file_path: Union[str, Path],
    mode: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None,
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LANCZOS4
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Load image from file.
    
    Args:
        file_path: Path to image file
        mode: Image mode (None for auto, 'pil', 'numpy', 'tensor')
        target_size: Target size (width, height)
        keep_aspect_ratio: Whether to keep aspect ratio when resizing
        interpolation: Interpolation method for resizing
        
    Returns:
        Loaded image in specified format
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    # Auto-detect mode from file extension if not specified
    if mode is None:
        suffix = file_path.suffix.lower()
        if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            mode = 'pil'
        else:
            mode = 'numpy'
    
    try:
        if mode == 'pil':
            image = Image.open(file_path)
            if target_size is not None:
                if keep_aspect_ratio:
                    image.thumbnail(target_size, Image.Resampling.LANCZOS)
                else:
                    image = image.resize(target_size, Image.Resampling.LANCZOS)
            return image
        
        elif mode == 'numpy':
            image = cv2.imread(str(file_path))
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if target_size is not None:
                if keep_aspect_ratio:
                    h, w = image.shape[:2]
                    target_w, target_h = target_size
                    
                    # Calculate aspect ratio
                    aspect = w / h
                    target_aspect = target_w / target_h
                    
                    if aspect > target_aspect:
                        # Image is wider
                        new_w = target_w
                        new_h = int(target_w / aspect)
                    else:
                        # Image is taller
                        new_h = target_h
                        new_w = int(target_h * aspect)
                    
                    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
                else:
                    image = cv2.resize(image, target_size, interpolation=interpolation)
            
            return image
        
        elif mode == 'tensor':
            # Load as numpy first, then convert to tensor
            image_np = load_image(file_path, mode='numpy', target_size=target_size,
                                 keep_aspect_ratio=keep_aspect_ratio, interpolation=interpolation)
            image_tensor = torch.from_numpy(image_np).float()
            # Convert HWC to CHW
            image_tensor = image_tensor.permute(2, 0, 1) / 255.0
            return image_tensor
        
        else:
            raise ValueError(f"Unsupported image mode: {mode}")
    
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        raise


def save_image(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    file_path: Union[str, Path],
    quality: int = 95,
    **kwargs
) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        file_path: Path to output file
        quality: JPEG quality (1-100)
        **kwargs: Additional arguments for save
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if isinstance(image, Image.Image):
            image.save(file_path, quality=quality, **kwargs)
        
        elif isinstance(image, np.ndarray):
            # Convert to PIL Image
            if image.ndim == 3 and image.shape[2] == 3:
                # RGB
                image_pil = Image.fromarray(image.astype(np.uint8))
            elif image.ndim == 2:
                # Grayscale
                image_pil = Image.fromarray(image.astype(np.uint8))
            else:
                raise ValueError(f"Unsupported numpy array shape: {image.shape}")
            
            image_pil.save(file_path, quality=quality, **kwargs)
        
        elif isinstance(image, torch.Tensor):
            # Convert to numpy
            if image.ndim == 3:
                # CHW to HWC
                if image.shape[0] <= 3:
                    image_np = image.permute(1, 2, 0).numpy()
                else:
                    image_np = image.numpy()
                
                # Denormalize if needed
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                save_image(image_np, file_path, quality=quality, **kwargs)
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        logger.debug(f"Saved image to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save image to {file_path}: {e}")
        return False


def load_video(
    file_path: Union[str, Path],
    max_frames: Optional[int] = None,
    target_size: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None
) -> np.ndarray:
    """
    Load video from file.
    
    Args:
        file_path: Path to video file
        max_frames: Maximum number of frames to load
        target_size: Target frame size (width, height)
        fps: Target frames per second (None for original)
        
    Returns:
        Video as numpy array (T, H, W, C)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    try:
        cap = cv2.VideoCapture(str(file_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {file_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine number of frames to read
        if max_frames is not None:
            frames_to_read = min(max_frames, total_frames)
        else:
            frames_to_read = total_frames
        
        # Calculate skip factor if fps is specified
        skip_factor = 1
        if fps is not None and original_fps > 0:
            skip_factor = max(1, int(original_fps / fps))
        
        frames = []
        frame_count = 0
        
        while frame_count < frames_to_read:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Skip frames if needed
            if frame_count % skip_factor == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if needed
                if target_size is not None:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
                
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames read from video: {file_path}")
        
        video_array = np.stack(frames)
        logger.debug(f"Loaded video {file_path}: {video_array.shape}")
        
        return video_array
    
    except Exception as e:
        logger.error(f"Failed to load video {file_path}: {e}")
        raise


def save_video(
    frames: Union[np.ndarray, List[np.ndarray]],
    file_path: Union[str, Path],
    fps: float = 30.0,
    codec: str = 'mp4v',
    **kwargs
) -> bool:
    """
    Save video to file.
    
    Args:
        frames: List of frames or numpy array (T, H, W, C)
        file_path: Path to output file
        fps: Frames per second
        codec: Video codec
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    if not frames:
        logger.warning("No frames to save")
        return False
    
    # Convert to list if numpy array
    if isinstance(frames, np.ndarray):
        frames = [frames[i] for i in range(frames.shape[0])]
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Determine file extension
        suffix = file_path.suffix.lower()
        if suffix == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif suffix == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*codec)
        elif suffix == '.mov':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create video writer
        out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
        
        out.release()
        
        logger.debug(f"Saved video to {file_path}: {len(frames)} frames, {fps} FPS")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save video to {file_path}: {e}")
        return False


def load_numpy(file_path: Union[str, Path], **kwargs) -> np.ndarray:
    """
    Load numpy array from file.
    
    Args:
        file_path: Path to numpy file
        **kwargs: Additional arguments for np.load
        
    Returns:
        Numpy array
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Numpy file not found: {file_path}")
    
    try:
        data = np.load(file_path, **kwargs)
        return data
    except Exception as e:
        logger.error(f"Failed to load numpy file {file_path}: {e}")
        raise


def save_numpy(
    array: np.ndarray,
    file_path: Union[str, Path],
    **kwargs
) -> bool:
    """
    Save numpy array to file.
    
    Args:
        array: Numpy array to save
        file_path: Path to output file
        **kwargs: Additional arguments for np.save
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        np.save(file_path, array, **kwargs)
        logger.debug(f"Saved numpy array to {file_path}: {array.shape}")
        return True
    except Exception as e:
        logger.error(f"Failed to save numpy array to {file_path}: {e}")
        return False


def load_tensor(
    file_path: Union[str, Path],
    map_location: Optional[str] = None,
    **kwargs
) -> torch.Tensor:
    """
    Load tensor from file.
    
    Args:
        file_path: Path to tensor file
        map_location: Device to map tensor to
        **kwargs: Additional arguments for torch.load
        
    Returns:
        PyTorch tensor
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {file_path}")
    
    try:
        data = torch.load(file_path, map_location=map_location, **kwargs)
        return data
    except Exception as e:
        logger.error(f"Failed to load tensor file {file_path}: {e}")
        raise


def save_tensor(
    tensor: torch.Tensor,
    file_path: Union[str, Path],
    **kwargs
) -> bool:
    """
    Save tensor to file.
    
    Args:
        tensor: Tensor to save
        file_path: Path to output file
        **kwargs: Additional arguments for torch.save
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(tensor, file_path, **kwargs)
        logger.debug(f"Saved tensor to {file_path}: {tensor.shape}")
        return True
    except Exception as e:
        logger.error(f"Failed to save tensor to {file_path}: {e}")
        return False


def batch_process_files(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    process_func: callable,
    file_pattern: str = "*",
    recursive: bool = False,
    max_files: Optional[int] = None,
    skip_existing: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Batch process files in directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        process_func: Function to process each file
        file_pattern: File pattern to match
        recursive: Whether to search recursively
        max_files: Maximum number of files to process
        skip_existing: Whether to skip existing output files
        **kwargs: Additional arguments for process_func
        
    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find files
    if recursive:
        files = list(input_dir.rglob(file_pattern))
    else:
        files = list(input_dir.glob(file_pattern))
    
    # Filter files
    files = [f for f in files if f.is_file()]
    
    # Limit number of files
    if max_files is not None:
        files = files[:max_files]
    
    # Statistics
    stats = {
        'total': len(files),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    # Process files
    for file_path in tqdm(files, desc="Processing files"):
        try:
            # Determine output path
            rel_path = file_path.relative_to(input_dir)
            output_path = output_dir / rel_path
            
            # Skip if output exists and skip_existing is True
            if skip_existing and output_path.exists():
                stats['skipped'] += 1
                continue
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Process file
            result = process_func(file_path, output_path, **kwargs)
            
            if result:
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append(str(e))
            logger.error(f"Failed to process {file_path}: {e}")
    
    logger.info(f"Batch processing completed: {stats}")
    return stats


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get detailed information about a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = file_path.stat()
    
    info = {
        'path': str(file_path),
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'parent': str(file_path.parent),
        'size_bytes': stat.st_size,
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir(),
        'is_symlink': file_path.is_symlink(),
        'md5': calculate_md5(file_path) if file_path.is_file() else None
    }
    
    return info


def create_file_index(directory: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Create index of all files in directory.
    
    Args:
        directory: Directory to index
        output_file: Optional output CSV file
        
    Returns:
        DataFrame with file information
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    files = []
    
    for file_path in tqdm(directory.rglob('*'), desc="Indexing files"):
        if file_path.is_file():
            try:
                info = get_file_info(file_path)
                files.append(info)
            except Exception as e:
                logger.warning(f"Failed to get info for {file_path}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(files)
    
    # Save to file if requested
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"File index saved to {output_file}: {len(df)} files")
    
    return df