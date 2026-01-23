"""
Video Dataset Module
Handles loading and processing of video datasets for temporal generation.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import torch
import numpy as np
import cv2
from PIL import Image

from .base_dataset import BaseDataset, DatasetConfig, DatasetPhase, DatasetType

class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    WEBM = "webm"
    GIF = "gif"
    MKV = "mkv"

class FrameSamplingStrategy(Enum):
    """Strategies for sampling frames from videos."""
    UNIFORM = "uniform"
    RANDOM = "random"
    STRIDE = "stride"
    ALL = "all"
    KEYFRAME = "keyframe"

@dataclass
class VideoDatasetConfig(DatasetConfig):
    """
    Configuration for video datasets.
    
    Attributes:
        frame_size: Target frame size (width, height)
        num_frames: Number of frames to sample per video
        frame_sampling: Frame sampling strategy
        frame_stride: Stride for frame sampling
        fps: Target frames per second (None for original)
        max_frames: Maximum number of frames to load per video
        channels: Number of color channels (3 for RGB)
        normalize: Whether to normalize frames
        normalize_range: Range for normalization
        load_as_tensor: Whether to load frames as tensors
        cache_videos: Whether to cache videos in memory
        video_format: Expected video format
        include_audio: Whether to include audio tracks
        audio_sample_rate: Target audio sample rate
        temporal_augmentation: Whether to apply temporal augmentations
        temporal_stride_range: Range for random temporal stride
        reverse_probability: Probability of reversing video
    """
    frame_size: Tuple[int, int] = (256, 256)
    num_frames: int = 16
    frame_sampling: FrameSamplingStrategy = FrameSamplingStrategy.UNIFORM
    frame_stride: int = 1
    fps: Optional[int] = None
    max_frames: Optional[int] = None
    channels: int = 3
    normalize: bool = True
    normalize_range: str = "-1_1"
    load_as_tensor: bool = True
    cache_videos: bool = False
    video_format: Optional[str] = None
    include_audio: bool = False
    audio_sample_rate: int = 16000
    temporal_augmentation: bool = False
    temporal_stride_range: Tuple[int, int] = (1, 4)
    reverse_probability: float = 0.0
    
    def __post_init__(self):
        """Validate video dataset configuration."""
        super().__post_init__()
        
        # Set dataset type
        self.type = DatasetType.VIDEO
        
        # Validate frame size
        if isinstance(self.frame_size, int):
            self.frame_size = (self.frame_size, self.frame_size)
        
        # Validate frame sampling
        if isinstance(self.frame_sampling, str):
            self.frame_sampling = FrameSamplingStrategy(self.frame_sampling)
        
        # Validate num_frames
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        
        # Validate frame_stride
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {self.frame_stride}")
        
        # Validate normalize_range
        if self.normalize_range not in ["0_1", "-1_1"]:
            raise ValueError(f"normalize_range must be '0_1' or '-1_1', got {self.normalize_range}")
        
        # Validate reverse_probability
        if not 0 <= self.reverse_probability <= 1:
            raise ValueError(f"reverse_probability must be between 0 and 1, got {self.reverse_probability}")


class VideoDataset(BaseDataset):
    """
    Dataset for handling video data with frame sampling and temporal processing.
    """
    
    def __init__(self, config: VideoDatasetConfig):
        """
        Initialize video dataset.
        
        Args:
            config: Video dataset configuration
        """
        if not isinstance(config, VideoDatasetConfig):
            config = VideoDatasetConfig.from_dict(config)
        
        super().__init__(config)
        
        # Video cache
        self.video_cache = {}
        self.cache_videos = config.cache_videos
        
        # Supported formats
        self.supported_formats = [fmt.value for fmt in VideoFormat]
        
        # Audio handling
        self.include_audio = config.include_audio
        if self.include_audio:
            try:
                import librosa
                self.librosa = librosa
            except ImportError:
                logger.warning("librosa not installed, audio support disabled")
                self.include_audio = False
        
        # Video readers cache
        self.video_readers = {}
    
    def _load_dataset(self) -> None:
        """Load video dataset samples."""
        # Check if metadata file exists
        if self.metadata_file and self.metadata_file.exists():
            self._load_from_metadata()
        else:
            self._load_from_directory()
        
        logger.info(f"Loaded {len(self.samples)} video samples")
    
    def _load_from_metadata(self) -> None:
        """Load dataset from metadata file."""
        if self.metadata_file.suffix == '.json':
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if isinstance(metadata, list):
                self.samples = metadata
            elif isinstance(metadata, dict) and 'samples' in metadata:
                self.samples = metadata['samples']
                self.metadata = {k: v for k, v in metadata.items() if k != 'samples'}
            else:
                raise ValueError(f"Invalid metadata format in {self.metadata_file}")
        
        # Convert paths to absolute
        for sample in self.samples:
            if 'video_path' in sample:
                sample['video_path'] = str(self.data_dir / sample['video_path'])
            elif 'path' in sample:
                sample['video_path'] = str(self.data_dir / sample['path'])
    
    def _load_from_directory(self) -> None:
        """Load dataset from directory structure."""
        self.samples = []
        
        # Supported video extensions
        extensions = {f'.{fmt.value}' for fmt in VideoFormat}
        
        # Recursively find video files
        for ext in extensions:
            video_files = list(self.data_dir.rglob(f'*{ext}'))
            
            for video_path in video_files:
                # Skip hidden files
                if video_path.name.startswith('.'):
                    continue
                
                # Get video info
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = frame_count / fps if fps > 0 else 0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    cap.release()
                    
                    sample = {
                        'video_path': str(video_path),
                        'filename': video_path.name,
                        'frame_count': frame_count,
                        'fps': fps,
                        'duration': duration,
                        'resolution': (width, height),
                        'relative_path': str(video_path.relative_to(self.data_dir))
                    }
                    
                    self.samples.append(sample)
                
                except Exception as e:
                    logger.warning(f"Failed to read video {video_path}: {e}")
        
        # Sort by filename
        self.samples.sort(key=lambda x: x['video_path'])
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for video samples."""
        return ['video_path']
    
    def _load_sample(self, index: int) -> Dict[str, Any]:
        """Load a single video sample."""
        sample_info = self.samples[index].copy()
        video_path = sample_info['video_path']
        
        # Check video cache
        if self.cache_videos and video_path in self.video_cache:
            frames = self.video_cache[video_path]
        else:
            # Load video frames
            frames = self._load_video_frames(video_path)
            
            # Cache if enabled
            if self.cache_videos:
                self.video_cache[video_path] = frames
        
        # Add frames to sample
        sample_info['frames'] = frames
        
        # Load audio if requested
        if self.include_audio:
            audio = self._load_audio(video_path)
            sample_info['audio'] = audio
        
        return sample_info
    
    def _load_video_frames(self, video_path: str) -> Union[torch.Tensor, np.ndarray]:
        """
        Load frames from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Loaded frames as tensor or array
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Apply max_frames limit
            if self.config.max_frames is not None:
                frame_count = min(frame_count, self.config.max_frames)
            
            # Sample frames
            frame_indices = self._sample_frame_indices(frame_count)
            
            # Read frames
            frames = []
            for idx in frame_indices:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    # If frame can't be read, use previous frame or zeros
                    if frames:
                        frame = frames[-1].copy()
                    else:
                        frame = np.zeros((*self.config.frame_size[::-1], self.config.channels), dtype=np.uint8)
                    logger.warning(f"Failed to read frame {idx} from {video_path}")
                
                # Process frame
                frame = self._process_frame(frame)
                frames.append(frame)
            
            cap.release()
            
            # Stack frames
            if self.config.load_as_tensor:
                frames_tensor = torch.stack(frames)
                
                # Normalize if requested
                if self.config.normalize:
                    if self.config.normalize_range == "0_1":
                        frames_tensor = frames_tensor / 255.0
                    else:  # "-1_1"
                        frames_tensor = (frames_tensor / 127.5) - 1.0
                
                return frames_tensor
            else:
                return np.stack(frames)
        
        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")
            # Return blank frames as fallback
            if self.config.load_as_tensor:
                return torch.zeros((self.config.num_frames, self.config.channels, 
                                   *self.config.frame_size[::-1]))
            else:
                return np.zeros((self.config.num_frames, *self.config.frame_size[::-1], 
                                self.config.channels), dtype=np.uint8)
    
    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """
        Sample frame indices based on sampling strategy.
        
        Args:
            total_frames: Total number of frames in video
            
        Returns:
            List of frame indices
        """
        if total_frames <= 0:
            return [0] * self.config.num_frames
        
        # Apply temporal augmentation if enabled
        if self.config.temporal_augmentation and self.phase == DatasetPhase.TRAIN:
            stride = self.rng.randint(*self.config.temporal_stride_range)
        else:
            stride = self.config.frame_stride
        
        # Determine available frames after stride
        available_frames = total_frames // stride
        if available_frames == 0:
            # If stride is too large, use all frames with stride 1
            stride = 1
            available_frames = total_frames
        
        # Apply sampling strategy
        if self.config.frame_sampling == FrameSamplingStrategy.ALL:
            # Use all available frames
            num_frames_to_sample = min(available_frames, self.config.num_frames)
            if available_frames <= self.config.num_frames:
                indices = list(range(0, available_frames * stride, stride))
                # Pad if needed
                while len(indices) < self.config.num_frames:
                    indices.append(indices[-1])
            else:
                # Sample uniformly
                step = available_frames / self.config.num_frames
                indices = [int(i * step) * stride for i in range(self.config.num_frames)]
        
        elif self.config.frame_sampling == FrameSamplingStrategy.UNIFORM:
            # Sample uniformly
            step = available_frames / self.config.num_frames
            indices = [int(i * step) * stride for i in range(self.config.num_frames)]
        
        elif self.config.frame_sampling == FrameSamplingStrategy.RANDOM:
            # Sample random frames
            if available_frames >= self.config.num_frames:
                sampled = self.rng.choice(available_frames, self.config.num_frames, replace=False)
                sampled.sort()
                indices = [idx * stride for idx in sampled]
            else:
                # With replacement if not enough frames
                sampled = self.rng.choice(available_frames, self.config.num_frames, replace=True)
                sampled.sort()
                indices = [idx * stride for idx in sampled]
        
        elif self.config.frame_sampling == FrameSamplingStrategy.STRIDE:
            # Fixed stride from start
            start_idx = 0
            if self.config.temporal_augmentation and self.phase == DatasetPhase.TRAIN:
                max_start = max(0, total_frames - self.config.num_frames * stride)
                start_idx = self.rng.randint(0, max_start)
            
            indices = list(range(start_idx, start_idx + self.config.num_frames * stride, stride))
            indices = indices[:self.config.num_frames]
        
        elif self.config.frame_sampling == FrameSamplingStrategy.KEYFRAME:
            # Try to detect keyframes (simplified)
            # In production, use proper keyframe detection
            step = max(1, total_frames // self.config.num_frames)
            indices = list(range(0, total_frames, step))[:self.config.num_frames]
        
        else:
            raise ValueError(f"Unknown frame sampling strategy: {self.config.frame_sampling}")
        
        # Ensure indices are within bounds
        indices = [min(idx, total_frames - 1) for idx in indices]
        
        # Apply reverse with probability
        if (self.config.reverse_probability > 0 and 
            self.phase == DatasetPhase.TRAIN and 
            self.rng.random() < self.config.reverse_probability):
            indices = indices[::-1]
        
        return indices
    
    def _process_frame(self, frame: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        target_width, target_height = self.config.frame_size
        if frame.shape[:2] != (target_height, target_width):
            frame = cv2.resize(frame, (target_width, target_height), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor if requested
        if self.config.load_as_tensor:
            frame_tensor = torch.from_numpy(frame).float()
            
            # Rearrange to (C, H, W)
            if len(frame_tensor.shape) == 3:
                frame_tensor = frame_tensor.permute(2, 0, 1)
            
            return frame_tensor
        else:
            return frame
    
    def _load_audio(self, video_path: str) -> Optional[torch.Tensor]:
        """
        Load audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Audio tensor or None
        """
        if not self.include_audio:
            return None
        
        try:
            # Load audio using librosa
            audio, sr = self.librosa.load(video_path, sr=self.config.audio_sample_rate)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Normalize
            if audio_tensor.abs().max() > 0:
                audio_tensor = audio_tensor / audio_tensor.abs().max()
            
            return audio_tensor
        
        except Exception as e:
            logger.warning(f"Failed to load audio from {video_path}: {e}")
            return None
    
    def get_video_info(self, index: int) -> Dict[str, Any]:
        """
        Get video information without loading frames.
        
        Args:
            index: Sample index
            
        Returns:
            Video information dictionary
        """
        sample = self.samples[index]
        
        info = {
            'path': sample['video_path'],
            'frame_count': sample.get('frame_count', 0),
            'fps': sample.get('fps', 0),
            'duration': sample.get('duration', 0),
            'resolution': sample.get('resolution', (0, 0)),
        }
        
        return info
    
    def extract_frame(self, video_path: str, frame_index: int) -> Image.Image:
        """
        Extract a single frame from video.
        
        Args:
            video_path: Path to video file
            frame_index: Frame index
            
        Returns:
            Extracted frame as PIL Image
        """
        try:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
            else:
                return Image.new('RGB', self.config.frame_size)
        
        except Exception as e:
            logger.error(f"Failed to extract frame {frame_index} from {video_path}: {e}")
            return Image.new('RGB', self.config.frame_size)
    
    def create_video_from_frames(self, frames: Union[torch.Tensor, np.ndarray], 
                                output_path: str, fps: int = 30) -> None:
        """
        Create video from frames.
        
        Args:
            frames: Sequence of frames
            output_path: Output video path
            fps: Frames per second
        """
        # Convert frames to numpy array
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        
        # Ensure correct shape and type
        if frames.ndim == 4:  # (T, C, H, W)
            frames = frames.transpose(0, 2, 3, 1)  # (T, H, W, C)
        
        # Denormalize if needed
        if self.config.normalize:
            if self.config.normalize_range == "-1_1":
                frames = (frames + 1) * 127.5
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if frames.shape[-1] == 3:
            frames = frames[..., ::-1]
        
        # Get video properties
        height, width = frames.shape[1:3]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Created video: {output_path}")
    
    def visualize_sample(self, index: int, save_path: Optional[str] = None, 
                        grid_size: Tuple[int, int] = (4, 4)) -> Image.Image:
        """
        Visualize video sample as frame grid.
        
        Args:
            index: Sample index
            save_path: Optional path to save visualization
            grid_size: Grid dimensions (rows, cols)
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        sample = self.get_sample(index, apply_transform=False)
        frames = sample['frames']
        
        # Convert to numpy for visualization
        if isinstance(frames, torch.Tensor):
            # Denormalize if needed
            if self.config.normalize:
                if self.config.normalize_range == "-1_1":
                    frames = (frames + 1) / 2
                frames = frames * 255
            
            # Convert to HWC format
            if frames.ndim == 4:  # T, C, H, W
                frames = frames.permute(0, 2, 3, 1)
            frames = frames.byte().numpy()
        
        # Select frames to display
        num_frames = frames.shape[0]
        rows, cols = grid_size
        max_frames = rows * cols
        
        if num_frames > max_frames:
            # Sample uniformly
            indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
            frames = frames[indices]
        else:
            # Pad with last frame
            rows = int(np.ceil(np.sqrt(num_frames)))
            cols = rows
            max_frames = rows * cols
            
            if num_frames < max_frames:
                padding = np.repeat(frames[-1:], max_frames - num_frames, axis=0)
                frames = np.concatenate([frames, padding], axis=0)
        
        # Create grid
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < frames.shape[0]:
                frame = frames[i]
                ax.imshow(frame)
            ax.axis('off')
        
        plt.suptitle(f"Video: {sample.get('filename', '')}", fontsize=14)
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