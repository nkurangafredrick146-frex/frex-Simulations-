"""
Multimodal Dataset Module
Handles datasets with multiple modalities (image, text, 3D, etc.) for world model training.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
import trimesh

from .base_dataset import BaseDataset, DatasetConfig, DatasetPhase, DatasetType
from .image_dataset import ImageDataset
from .video_dataset import VideoDataset

class ModalityType(Enum):
    """Types of modalities supported."""
    IMAGE = "image"
    TEXT = "text"
    VIDEO = "video"
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"
    VOXEL = "voxel"
    DEPTH = "depth"
    NORMAL = "normal"
    AUDIO = "audio"
    EMBEDDING = "embedding"

@dataclass
class MultimodalSample:
    """
    Container for multimodal sample data.
    
    Attributes:
        id: Unique sample identifier
        modalities: Dictionary mapping modality type to data
        metadata: Additional metadata
    """
    id: str
    modalities: Dict[ModalityType, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert modality keys to ModalityType enum."""
        converted_modalities = {}
        for key, value in self.modalities.items():
            if isinstance(key, str):
                converted_modalities[ModalityType(key)] = value
            else:
                converted_modalities[key] = value
        self.modalities = converted_modalities
    
    def add_modality(self, modality_type: Union[ModalityType, str], data: Any) -> None:
        """Add a modality to the sample."""
        if isinstance(modality_type, str):
            modality_type = ModalityType(modality_type)
        self.modalities[modality_type] = data
    
    def get_modality(self, modality_type: Union[ModalityType, str]) -> Optional[Any]:
        """Get data for a specific modality."""
        if isinstance(modality_type, str):
            modality_type = ModalityType(modality_type)
        return self.modalities.get(modality_type)
    
    def has_modality(self, modality_type: Union[ModalityType, str]) -> bool:
        """Check if sample has specific modality."""
        if isinstance(modality_type, str):
            modality_type = ModalityType(modality_type)
        return modality_type in self.modalities
    
    def get_modality_types(self) -> Set[ModalityType]:
        """Get set of available modality types."""
        return set(self.modalities.keys())


@dataclass
class MultimodalDatasetConfig(DatasetConfig):
    """
    Configuration for multimodal datasets.
    
    Attributes:
        modalities: List of modality types to include
        modality_configs: Configuration for each modality
        alignment_strategy: How to align different modalities
        require_all_modalities: Whether all modalities must be present
        missing_modality_handling: How to handle missing modalities
        cross_modal_augmentation: Whether to apply cross-modal augmentations
        modality_weights: Weights for each modality in loss
        embedding_cache: Whether to cache precomputed embeddings
        embedding_dir: Directory for cached embeddings
    """
    modalities: List[Union[str, ModalityType]] = field(default_factory=lambda: [
        ModalityType.IMAGE, ModalityType.TEXT
    ])
    modality_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alignment_strategy: str = "sample_id"  # "sample_id", "temporal", "spatial"
    require_all_modalities: bool = False
    missing_modality_handling: str = "skip"  # "skip", "fill_zeros", "fill_random"
    cross_modal_augmentation: bool = False
    modality_weights: Dict[str, float] = field(default_factory=dict)
    embedding_cache: bool = True
    embedding_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate multimodal dataset configuration."""
        super().__post_init__()
        
        # Set dataset type
        self.type = DatasetType.MULTIMODAL
        
        # Convert modality strings to ModalityType
        converted_modalities = []
        for modality in self.modalities:
            if isinstance(modality, str):
                converted_modalities.append(ModalityType(modality))
            else:
                converted_modalities.append(modality)
        self.modalities = converted_modalities
        
        # Validate alignment strategy
        valid_strategies = ["sample_id", "temporal", "spatial"]
        if self.alignment_strategy not in valid_strategies:
            raise ValueError(f"alignment_strategy must be one of {valid_strategies}, "
                           f"got {self.alignment_strategy}")
        
        # Validate missing modality handling
        valid_handling = ["skip", "fill_zeros", "fill_random"]
        if self.missing_modality_handling not in valid_handling:
            raise ValueError(f"missing_modality_handling must be one of {valid_handling}, "
                           f"got {self.missing_modality_handling}")
        
        # Set default modality weights
        if not self.modality_weights:
            self.modality_weights = {modality.value: 1.0 for modality in self.modalities}
        
        # Ensure embedding directory exists
        if self.embedding_cache and self.embedding_dir:
            Path(self.embedding_dir).mkdir(parents=True, exist_ok=True)


class MultimodalDataset(BaseDataset):
    """
    Dataset for handling multimodal data with multiple modality types.
    
    Supports alignment and fusion of different data types.
    """
    
    def __init__(self, config: MultimodalDatasetConfig):
        """
        Initialize multimodal dataset.
        
        Args:
            config: Multimodal dataset configuration
        """
        if not isinstance(config, MultimodalDatasetConfig):
            config = MultimodalDatasetConfig.from_dict(config)
        
        super().__init__(config)
        
        # Modality handling
        self.modalities = config.modalities
        self.modality_configs = config.modality_configs
        self.alignment_strategy = config.alignment_strategy
        self.require_all_modalities = config.require_all_modalities
        self.missing_modality_handling = config.missing_modality_handling
        self.modality_weights = config.modality_weights
        
        # Cross-modal augmentation
        self.cross_modal_augmentation = config.cross_modal_augmentation
        
        # Embedding cache
        self.embedding_cache = config.embedding_cache
        self.embedding_dir = Path(config.embedding_dir) if config.embedding_dir else None
        self.embedding_cache_dict = {}
        
        # Modality-specific loaders
        self.modality_loaders = self._initialize_modality_loaders()
        
        # Alignment data
        self.alignment_map = {}
        
        logger.info(f"Initialized multimodal dataset with modalities: "
                   f"{[m.value for m in self.modalities]}")
    
    def _initialize_modality_loaders(self) -> Dict[ModalityType, Callable]:
        """Initialize loaders for each modality type."""
        loaders = {}
        
        for modality in self.modalities:
            if modality == ModalityType.IMAGE:
                loaders[modality] = self._load_image_modality
            elif modality == ModalityType.TEXT:
                loaders[modality] = self._load_text_modality
            elif modality == ModalityType.VIDEO:
                loaders[modality] = self._load_video_modality
            elif modality == ModalityType.POINT_CLOUD:
                loaders[modality] = self._load_point_cloud_modality
            elif modality == ModalityType.MESH:
                loaders[modality] = self._load_mesh_modality
            elif modality == ModalityType.VOXEL:
                loaders[modality] = self._load_voxel_modality
            elif modality == ModalityType.DEPTH:
                loaders[modality] = self._load_depth_modality
            elif modality == ModalityType.NORMAL:
                loaders[modality] = self._load_normal_modality
            elif modality == ModalityType.AUDIO:
                loaders[modality] = self._load_audio_modality
            elif modality == ModalityType.EMBEDDING:
                loaders[modality] = self._load_embedding_modality
            else:
                raise ValueError(f"Unsupported modality: {modality}")
        
        return loaders
    
    def _load_dataset(self) -> None:
        """Load multimodal dataset samples."""
        # Check if metadata file exists
        if self.metadata_file and self.metadata_file.exists():
            self._load_from_metadata()
        else:
            self._load_from_directory_structure()
        
        # Build alignment map
        self._build_alignment_map()
        
        # Filter samples based on modality requirements
        self._filter_samples_by_modality()
        
        logger.info(f"Loaded {len(self.samples)} multimodal samples")
    
    def _load_from_metadata(self) -> None:
        """Load dataset from metadata file."""
        if self.metadata_file.suffix == '.json':
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if isinstance(metadata, list):
                # List of samples
                for sample_data in metadata:
                    sample = MultimodalSample(
                        id=sample_data.get('id', str(len(self.samples))),
                        metadata=sample_data.get('metadata', {})
                    )
                    
                    # Add modalities
                    for modality_str, modality_data in sample_data.get('modalities', {}).items():
                        try:
                            modality_type = ModalityType(modality_str)
                            sample.add_modality(modality_type, modality_data)
                        except ValueError:
                            logger.warning(f"Unknown modality: {modality_str}")
                    
                    self.samples.append(sample)
            
            elif isinstance(metadata, dict):
                # Dictionary format
                if 'samples' in metadata:
                    samples_data = metadata['samples']
                    self.metadata = {k: v for k, v in metadata.items() if k != 'samples'}
                    
                    for sample_data in samples_data:
                        sample = MultimodalSample(
                            id=sample_data.get('id', str(len(self.samples))),
                            metadata=sample_data.get('metadata', {})
                        )
                        
                        for modality_str, modality_data in sample_data.get('modalities', {}).items():
                            try:
                                modality_type = ModalityType(modality_str)
                                sample.add_modality(modality_type, modality_data)
                            except ValueError:
                                logger.warning(f"Unknown modality: {modality_str}")
                        
                        self.samples.append(sample)
                else:
                    raise ValueError("Invalid metadata format: missing 'samples' key")
    
    def _load_from_directory_structure(self) -> None:
        """Load dataset from directory structure organized by modality."""
        # Find all modality directories
        modality_dirs = {}
        for modality in self.modalities:
            modality_dir = self.data_dir / modality.value
            if modality_dir.exists():
                modality_dirs[modality] = modality_dir
        
        if not modality_dirs:
            raise ValueError(f"No modality directories found in {self.data_dir}")
        
        # Build sample map
        sample_map = defaultdict(dict)
        
        for modality, modality_dir in modality_dirs.items():
            # Find files in modality directory
            extensions = self._get_modality_extensions(modality)
            
            for ext in extensions:
                files = list(modality_dir.rglob(f'*{ext}'))
                
                for file_path in files:
                    # Extract sample ID from filename
                    sample_id = self._extract_sample_id(file_path, modality)
                    
                    # Store file path
                    sample_map[sample_id][modality] = str(file_path.relative_to(self.data_dir))
        
        # Create samples
        for sample_id, modalities_data in sample_map.items():
            sample = MultimodalSample(id=sample_id)
            
            for modality, rel_path in modalities_data.items():
                sample.add_modality(modality, rel_path)
            
            self.samples.append(sample)
    
    def _extract_sample_id(self, file_path: Path, modality: ModalityType) -> str:
        """
        Extract sample ID from file path.
        
        Args:
            file_path: Path to file
            modality: Modality type
            
        Returns:
            Sample ID
        """
        # Remove extension
        stem = file_path.stem
        
        # Remove modality suffix if present
        modality_suffix = f"_{modality.value}"
        if stem.endswith(modality_suffix):
            stem = stem[:-len(modality_suffix)]
        
        # Remove additional suffixes
        for suffix in ["_mask", "_depth", "_normal", "_albedo"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        
        return stem
    
    def _get_modality_extensions(self, modality: ModalityType) -> List[str]:
        """Get file extensions for a modality type."""
        if modality == ModalityType.IMAGE:
            return ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        elif modality == ModalityType.TEXT:
            return ['.txt', '.json']
        elif modality == ModalityType.VIDEO:
            return ['.mp4', '.avi', '.mov', '.webm']
        elif modality == ModalityType.POINT_CLOUD:
            return ['.ply', '.obj', '.npy', '.npz']
        elif modality == ModalityType.MESH:
            return ['.obj', '.glb', '.fbx', '.ply']
        elif modality == ModalityType.VOXEL:
            return ['.npy', '.npz', '.h5']
        elif modality == ModalityType.DEPTH:
            return ['.png', '.npy', '.exr']
        elif modality == ModalityType.NORMAL:
            return ['.png', '.npy', '.exr']
        elif modality == ModalityType.AUDIO:
            return ['.wav', '.mp3', '.flac']
        elif modality == ModalityType.EMBEDDING:
            return ['.npy', '.npz', '.pt', '.pth']
        else:
            return ['']
    
    def _build_alignment_map(self) -> None:
        """Build alignment map based on alignment strategy."""
        if self.alignment_strategy == "sample_id":
            # Samples are already aligned by ID
            self.alignment_map = {sample.id: {modality: sample.id for modality in self.modalities} 
                                for sample in self.samples}
        
        elif self.alignment_strategy == "temporal":
            # Align by temporal sequence (e.g., video frames)
            # This would require additional metadata
            logger.warning("Temporal alignment not implemented, using sample_id")
            self.alignment_strategy = "sample_id"
            self._build_alignment_map()
        
        elif self.alignment_strategy == "spatial":
            # Align by spatial coordinates
            logger.warning("Spatial alignment not implemented, using sample_id")
            self.alignment_strategy = "sample_id"
            self._build_alignment_map()
    
    def _filter_samples_by_modality(self) -> None:
        """Filter samples based on modality requirements."""
        if not self.require_all_modalities:
            return
        
        filtered_samples = []
        
        for sample in self.samples:
            sample_modalities = sample.get_modality_types()
            
            # Check if sample has all required modalities
            has_all_modalities = all(modality in sample_modalities 
                                   for modality in self.modalities)
            
            if has_all_modalities:
                filtered_samples.append(sample)
            else:
                missing = [m.value for m in self.modalities if m not in sample_modalities]
                logger.debug(f"Sample {sample.id} missing modalities: {missing}")
        
        self.samples = filtered_samples
        logger.info(f"Filtered to {len(self.samples)} samples with all modalities")
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for multimodal samples."""
        return ['id']
    
    def _load_sample(self, index: int) -> Dict[str, Any]:
        """Load a multimodal sample."""
        sample = self.samples[index]
        sample_id = sample.id
        
        result = {
            'id': sample_id,
            'modalities': {},
            'modality_weights': self.modality_weights.copy(),
            'metadata': sample.metadata.copy()
        }
        
        # Load data for each modality
        for modality in self.modalities:
            if sample.has_modality(modality):
                # Get modality data (could be path or actual data)
                modality_data = sample.get_modality(modality)
                
                # Load using modality-specific loader
                loader = self.modality_loaders[modality]
                loaded_data = loader(modality_data)
                
                result['modalities'][modality.value] = loaded_data
            else:
                # Handle missing modality
                result['modalities'][modality.value] = self._handle_missing_modality(modality)
        
        return result
    
    def _handle_missing_modality(self, modality: ModalityType) -> Any:
        """Handle missing modality data."""
        if self.missing_modality_handling == "skip":
            return None
        
        elif self.missing_modality_handling == "fill_zeros":
            # Return zero tensor with appropriate shape
            if modality == ModalityType.IMAGE:
                return torch.zeros((3, 256, 256))
            elif modality == ModalityType.TEXT:
                return ""
            elif modality == ModalityType.VIDEO:
                return torch.zeros((16, 3, 256, 256))
            elif modality == ModalityType.POINT_CLOUD:
                return torch.zeros((1024, 3))
            elif modality == ModalityType.MESH:
                # Return empty mesh
                return trimesh.Trimesh()
            else:
                return None
        
        elif self.missing_modality_handling == "fill_random":
            # Return random data
            if modality == ModalityType.IMAGE:
                return torch.randn((3, 256, 256))
            elif modality == ModalityType.TEXT:
                return "random text"
            elif modality == ModalityType.VIDEO:
                return torch.randn((16, 3, 256, 256))
            elif modality == ModalityType.POINT_CLOUD:
                return torch.randn((1024, 3))
            elif modality == ModalityType.MESH:
                # Return simple cube
                return trimesh.creation.box()
            else:
                return None
    
    def _load_image_modality(self, image_data: Any) -> torch.Tensor:
        """Load image modality."""
        if isinstance(image_data, torch.Tensor):
            return image_data
        elif isinstance(image_data, np.ndarray):
            return torch.from_numpy(image_data).float()
        elif isinstance(image_data, str):
            # Load from file
            image_path = self.data_dir / image_data
            return self._load_image_file(image_path)
        elif isinstance(image_data, Image.Image):
            return torch.from_numpy(np.array(image_data)).float()
        else:
            raise TypeError(f"Unsupported image data type: {type(image_data)}")
    
    def _load_image_file(self, image_path: Path) -> torch.Tensor:
        """Load image from file."""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                # Get image config
                config = self.modality_configs.get('image', {})
                size = config.get('size', (256, 256))
                
                # Resize
                img = img.resize(size, Image.Resampling.LANCZOS)
                
                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(img)).float()
                img_tensor = img_tensor.permute(2, 0, 1) / 255.0
                
                # Normalize
                if config.get('normalize', True):
                    mean = config.get('mean', [0.485, 0.456, 0.406])
                    std = config.get('std', [0.229, 0.224, 0.225])
                    mean = torch.tensor(mean).view(3, 1, 1)
                    std = torch.tensor(std).view(3, 1, 1)
                    img_tensor = (img_tensor - mean) / std
                
                return img_tensor
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return torch.zeros((3, 256, 256))
    
    def _load_text_modality(self, text_data: Any) -> str:
        """Load text modality."""
        if isinstance(text_data, str):
            if text_data.endswith('.txt'):
                # Load from file
                text_path = self.data_dir / text_data
                try:
                    with open(text_path, 'r') as f:
                        return f.read().strip()
                except Exception as e:
                    logger.error(f"Failed to load text {text_path}: {e}")
                    return ""
            else:
                return text_data
        else:
            return str(text_data)
    
    def _load_video_modality(self, video_data: Any) -> torch.Tensor:
        """Load video modality."""
        if isinstance(video_data, torch.Tensor):
            return video_data
        elif isinstance(video_data, np.ndarray):
            return torch.from_numpy(video_data).float()
        elif isinstance(video_data, str):
            # Load from file
            video_path = self.data_dir / video_data
            
            # Use VideoDataset's loading logic
            config = self.modality_configs.get('video', {})
            num_frames = config.get('num_frames', 16)
            frame_size = config.get('frame_size', (256, 256))
            
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                
                for _ in range(num_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize
                    frame = cv2.resize(frame, frame_size)
                    # BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Normalize
                    frame = frame / 255.0
                    
                    frames.append(frame)
                
                cap.release()
                
                if frames:
                    frames_tensor = torch.from_numpy(np.stack(frames)).float()
                    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # T, C, H, W
                    return frames_tensor
                else:
                    return torch.zeros((num_frames, 3, *frame_size))
            
            except Exception as e:
                logger.error(f"Failed to load video {video_path}: {e}")
                return torch.zeros((num_frames, 3, *frame_size))
        else:
            raise TypeError(f"Unsupported video data type: {type(video_data)}")
    
    def _load_point_cloud_modality(self, point_cloud_data: Any) -> torch.Tensor:
        """Load point cloud modality."""
        if isinstance(point_cloud_data, torch.Tensor):
            return point_cloud_data
        elif isinstance(point_cloud_data, np.ndarray):
            return torch.from_numpy(point_cloud_data).float()
        elif isinstance(point_cloud_data, str):
            # Load from file
            pc_path = self.data_dir / point_cloud_data
            
            try:
                if pc_path.suffix == '.npy':
                    points = np.load(pc_path)
                elif pc_path.suffix == '.npz':
                    data = np.load(pc_path)
                    points = data['points'] if 'points' in data else data['arr_0']
                elif pc_path.suffix == '.ply':
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(str(pc_path))
                    points = np.asarray(pcd.points)
                else:
                    raise ValueError(f"Unsupported point cloud format: {pc_path.suffix}")
                
                # Ensure correct shape
                if points.ndim == 2 and points.shape[1] == 3:
                    return torch.from_numpy(points).float()
                else:
                    raise ValueError(f"Invalid point cloud shape: {points.shape}")
            
            except Exception as e:
                logger.error(f"Failed to load point cloud {pc_path}: {e}")
                return torch.zeros((1024, 3))
        else:
            raise TypeError(f"Unsupported point cloud data type: {type(point_cloud_data)}")
    
    def _load_mesh_modality(self, mesh_data: Any) -> Any:
        """Load mesh modality."""
        if isinstance(mesh_data, trimesh.Trimesh):
            return mesh_data
        elif isinstance(mesh_data, str):
            # Load from file
            mesh_path = self.data_dir / mesh_data
            
            try:
                mesh = trimesh.load(str(mesh_path))
                return mesh
            except Exception as e:
                logger.error(f"Failed to load mesh {mesh_path}: {e}")
                return trimesh.creation.box()
        else:
            raise TypeError(f"Unsupported mesh data type: {type(mesh_data)}")
    
    def _load_voxel_modality(self, voxel_data: Any) -> torch.Tensor:
        """Load voxel modality."""
        if isinstance(voxel_data, torch.Tensor):
            return voxel_data
        elif isinstance(voxel_data, np.ndarray):
            return torch.from_numpy(voxel_data).float()
        elif isinstance(voxel_data, str):
            # Load from file
            voxel_path = self.data_dir / voxel_data
            
            try:
                if voxel_path.suffix == '.npy':
                    voxels = np.load(voxel_path)
                elif voxel_path.suffix == '.npz':
                    data = np.load(voxel_path)
                    voxels = data['voxels'] if 'voxels' in data else data['arr_0']
                else:
                    raise ValueError(f"Unsupported voxel format: {voxel_path.suffix}")
                
                return torch.from_numpy(voxels).float()
            
            except Exception as e:
                logger.error(f"Failed to load voxels {voxel_path}: {e}")
                return torch.zeros((32, 32, 32))
        else:
            raise TypeError(f"Unsupported voxel data type: {type(voxel_data)}")
    
    def _load_depth_modality(self, depth_data: Any) -> torch.Tensor:
        """Load depth modality."""
        # Similar to image loading but single channel
        if isinstance(depth_data, torch.Tensor):
            return depth_data
        elif isinstance(depth_data, np.ndarray):
            return torch.from_numpy(depth_data).float()
        elif isinstance(depth_data, str):
            depth_path = self.data_dir / depth_data
            
            try:
                if depth_path.suffix == '.npy':
                    depth = np.load(depth_path)
                else:
                    # Load as image
                    with Image.open(depth_path) as img:
                        depth = np.array(img.convert('F'))  # Convert to float
                
                # Add channel dimension if needed
                if depth.ndim == 2:
                    depth = depth[np.newaxis, ...]
                
                return torch.from_numpy(depth).float()
            
            except Exception as e:
                logger.error(f"Failed to load depth {depth_path}: {e}")
                return torch.zeros((1, 256, 256))
        else:
            raise TypeError(f"Unsupported depth data type: {type(depth_data)}")
    
    def _load_normal_modality(self, normal_data: Any) -> torch.Tensor:
        """Load normal modality."""
        # Similar to image loading but 3 channels
        return self._load_image_modality(normal_data)
    
    def _load_audio_modality(self, audio_data: Any) -> torch.Tensor:
        """Load audio modality."""
        if isinstance(audio_data, torch.Tensor):
            return audio_data
        elif isinstance(audio_data, np.ndarray):
            return torch.from_numpy(audio_data).float()
        elif isinstance(audio_data, str):
            audio_path = self.data_dir / audio_data
            
            try:
                import librosa
                audio, sr = librosa.load(str(audio_path), sr=None)
                return torch.from_numpy(audio).float()
            except Exception as e:
                logger.error(f"Failed to load audio {audio_path}: {e}")
                return torch.zeros(16000)
        else:
            raise TypeError(f"Unsupported audio data type: {type(audio_data)}")
    
    def _load_embedding_modality(self, embedding_data: Any) -> torch.Tensor:
        """Load embedding modality."""
        if isinstance(embedding_data, torch.Tensor):
            return embedding_data
        elif isinstance(embedding_data, np.ndarray):
            return torch.from_numpy(embedding_data).float()
        elif isinstance(embedding_data, str):
            embedding_path = self.data_dir / embedding_data
            
            try:
                if embedding_path.suffix == '.pt' or embedding_path.suffix == '.pth':
                    embedding = torch.load(embedding_path)
                elif embedding_path.suffix == '.npy':
                    embedding = np.load(embedding_path)
                    embedding = torch.from_numpy(embedding).float()
                else:
                    raise ValueError(f"Unsupported embedding format: {embedding_path.suffix}")
                
                return embedding
            
            except Exception as e:
                logger.error(f"Failed to load embedding {embedding_path}: {e}")
                return torch.zeros(768)
        else:
            raise TypeError(f"Unsupported embedding data type: {type(embedding_data)}")
    
    def get_modality_statistics(self, modality: ModalityType) -> Dict[str, Any]:
        """
        Compute statistics for a specific modality.
        
        Args:
            modality: Modality type
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'count': 0,
            'shape': None,
            'dtype': None,
            'mean': None,
            'std': None,
            'min': None,
            'max': None
        }
        
        values = []
        
        for sample in self.samples:
            if sample.has_modality(modality):
                modality_data = sample.get_modality(modality)
                
                if isinstance(modality_data, (torch.Tensor, np.ndarray)):
                    values.append(modality_data)
                    stats['count'] += 1
                    
                    if stats['shape'] is None:
                        stats['shape'] = modality_data.shape
                        stats['dtype'] = str(modality_data.dtype)
        
        if values:
            # Convert to single array/tensor for statistics
            if isinstance(values[0], torch.Tensor):
                all_values = torch.cat([v.flatten() for v in values])
                stats['mean'] = all_values.mean().item()
                stats['std'] = all_values.std().item()
                stats['min'] = all_values.min().item()
                stats['max'] = all_values.max().item()
            else:
                all_values = np.concatenate([v.flatten() for v in values])
                stats['mean'] = float(all_values.mean())
                stats['std'] = float(all_values.std())
                stats['min'] = float(all_values.min())
                stats['max'] = float(all_values.max())
        
        return stats
    
    def get_all_modality_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Compute statistics for all modalities."""
        stats = {}
        
        for modality in self.modalities:
            stats[modality.value] = self.get_modality_statistics(modality)
        
        return stats
    
    def visualize_sample(self, index: int, save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize multimodal sample.
        
        Args:
            index: Sample index
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        sample = self.get_sample(index, apply_transform=False)
        
        # Determine grid size based on modalities
        modalities = [m for m in self.modalities if sample['modalities'].get(m.value) is not None]
        num_modalities = len(modalities)
        
        if num_modalities == 0:
            return Image.new('RGB', (400, 400), color='white')
        
        # Create subplot grid
        cols = min(3, num_modalities)
        rows = (num_modalities + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(-1)
        else:
            axes = axes.flatten()
        
        for i, (ax, modality) in enumerate(zip(axes, modalities)):
            modality_data = sample['modalities'][modality.value]
            ax.set_title(f"{modality.value}")
            
            if modality == ModalityType.IMAGE:
                if isinstance(modality_data, torch.Tensor):
                    img = modality_data.detach().cpu()
                    if img.shape[0] == 3:  # C, H, W
                        img = img.permute(1, 2, 0)
                    img = img.numpy()
                    ax.imshow(img)
                elif isinstance(modality_data, np.ndarray):
                    ax.imshow(modality_data)
            
            elif modality == ModalityType.TEXT:
                text = str(modality_data)[:100] + "..." if len(str(modality_data)) > 100 else str(modality_data)
                ax.text(0.5, 0.5, text, ha='center', va='center', wrap=True)
                ax.axis('off')
            
            elif modality == ModalityType.VIDEO:
                if isinstance(modality_data, torch.Tensor):
                    # Show first frame
                    frame = modality_data[0].detach().cpu()
                    if frame.shape[0] == 3:  # C, H, W
                        frame = frame.permute(1, 2, 0)
                    frame = frame.numpy()
                    ax.imshow(frame)
                    ax.set_title(f"{modality.value} (frame 0/{len(modality_data)})")
            
            elif modality == ModalityType.POINT_CLOUD:
                if isinstance(modality_data, torch.Tensor):
                    points = modality_data.detach().cpu().numpy()
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
            
            elif modality == ModalityType.MESH:
                if hasattr(modality_data, 'vertices'):
                    vertices = modality_data.vertices
                    faces = modality_data.faces
                    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                                  triangles=faces)
            
            else:
                ax.text(0.5, 0.5, f"Modality: {modality.value}\nType: {type(modality_data)}",
                       ha='center', va='center')
                ax.axis('off')
        
        # Hide unused axes
        for i in range(len(modalities), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Multimodal Sample: {sample['id']}", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            return Image.open(save_path)
        else:
            # Convert to PIL Image
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return Image.fromarray(image_array)