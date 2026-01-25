"""
Transforms Module
Image, video, and multimodal transformations for data augmentation and preprocessing.
"""

import random
import math
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

class InterpolationMode(Enum):
    """Interpolation modes for resizing."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"
    BOX = "box"
    HAMMING = "hamming"

class PaddingMode(Enum):
    """Padding modes."""
    CONSTANT = "constant"
    EDGE = "edge"
    REFLECT = "reflect"
    SYMMETRIC = "symmetric"

@dataclass
class TransformConfig:
    """
    Configuration for transformations.
    
    Attributes:
        resize: Target size for resizing (width, height)
        crop: Crop size (width, height) or crop ratio
        normalize: Whether to normalize
        mean: Mean for normalization
        std: Standard deviation for normalization
        color_jitter: Color jitter parameters
        random_erasing: Random erasing parameters
        random_flip: Random flip probability
        random_rotate: Random rotation range in degrees
        gaussian_blur: Gaussian blur parameters
        random_perspective: Random perspective parameters
        random_affine: Random affine parameters
        random_crop: Random crop parameters
        center_crop: Whether to center crop
        to_tensor: Whether to convert to tensor
        interpolation: Interpolation mode
        padding_mode: Padding mode
    """
    resize: Optional[Tuple[int, int]] = None
    crop: Optional[Union[int, Tuple[int, int], float]] = None
    normalize: bool = True
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    color_jitter: Optional[Dict[str, float]] = None
    random_erasing: Optional[Dict[str, Any]] = None
    random_flip: float = 0.5
    random_rotate: Optional[Tuple[float, float]] = None
    gaussian_blur: Optional[Dict[str, Any]] = None
    random_perspective: Optional[Dict[str, float]] = None
    random_affine: Optional[Dict[str, Any]] = None
    random_crop: Optional[Dict[str, Any]] = None
    center_crop: bool = False
    to_tensor: bool = True
    interpolation: InterpolationMode = InterpolationMode.BILINEAR
    padding_mode: PaddingMode = PaddingMode.CONSTANT
    
    def __post_init__(self):
        """Validate transform configuration."""
        # Convert interpolation string to enum if needed
        if isinstance(self.interpolation, str):
            self.interpolation = InterpolationMode(self.interpolation)
        
        # Convert padding mode string to enum if needed
        if isinstance(self.padding_mode, str):
            self.padding_mode = PaddingMode(self.padding_mode)
        
        # Validate mean and std
        if self.normalize:
            if len(self.mean) != len(self.std):
                raise ValueError(f"mean and std must have same length, got {len(self.mean)} and {len(self.std)}")


class Compose:
    """
    Compose multiple transforms together.
    
    Similar to torchvision.transforms.Compose but with additional features.
    """
    
    def __init__(self, transforms: List[Callable]):
        """
        Initialize compose transform.
        
        Args:
            transforms: List of transforms to compose
        """
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Transformed sample
        """
        for transform in self.transforms:
            sample = transform(sample)
        return sample
    
    def __repr__(self) -> str:
        """String representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
    def add_transform(self, transform: Callable):
        """Add a transform to the composition."""
        self.transforms.append(transform)
    
    def insert_transform(self, index: int, transform: Callable):
        """Insert a transform at specific index."""
        self.transforms.insert(index, transform)
    
    def remove_transform(self, index: int):
        """Remove transform at index."""
        if 0 <= index < len(self.transforms):
            del self.transforms[index]


class ImageTransform:
    """
    Image transformation pipeline.
    
    Supports:
    - Resizing and cropping
    - Color jitter
    - Random erasing
    - Gaussian blur
    - Random perspective
    - Random affine
    - Normalization
    """
    
    def __init__(self, config: TransformConfig, training: bool = True):
        """
        Initialize image transform.
        
        Args:
            config: Transform configuration
            training: Whether in training mode (affects random transforms)
        """
        self.config = config
        self.training = training
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> Compose:
        """Build transform pipeline."""
        transforms = []
        
        # Resize
        if self.config.resize is not None:
            transforms.append(Resize(self.config.resize, interpolation=self.config.interpolation))
        
        # Random crop (training) or center crop (validation)
        if self.config.crop is not None:
            if self.training and self.config.random_crop:
                transforms.append(RandomCrop(
                    size=self.config.crop,
                    padding=self.config.random_crop.get('padding', 4),
                    pad_if_needed=self.config.random_crop.get('pad_if_needed', True),
                    fill=self.config.random_crop.get('fill', 0),
                    padding_mode=self.config.padding_mode
                ))
            elif self.config.center_crop:
                transforms.append(CenterCrop(self.config.crop))
        
        # Random flip (training only)
        if self.training and self.config.random_flip > 0:
            transforms.append(RandomHorizontalFlip(p=self.config.random_flip))
            transforms.append(RandomVerticalFlip(p=self.config.random_flip/2))
        
        # Color jitter (training only)
        if self.training and self.config.color_jitter:
            transforms.append(ColorJitter(
                brightness=self.config.color_jitter.get('brightness', 0.2),
                contrast=self.config.color_jitter.get('contrast', 0.2),
                saturation=self.config.color_jitter.get('saturation', 0.2),
                hue=self.config.color_jitter.get('hue', 0.1)
            ))
        
        # Random rotate (training only)
        if self.training and self.config.random_rotate:
            transforms.append(RandomRotation(
                degrees=self.config.random_rotate,
                interpolation=self.config.interpolation,
                expand=False
            ))
        
        # Gaussian blur (training only)
        if self.training and self.config.gaussian_blur:
            transforms.append(GaussianBlur(
                kernel_size=self.config.gaussian_blur.get('kernel_size', 3),
                sigma=self.config.gaussian_blur.get('sigma', (0.1, 2.0))
            ))
        
        # Random perspective (training only)
        if self.training and self.config.random_perspective:
            transforms.append(RandomPerspective(
                distortion_scale=self.config.random_perspective.get('distortion_scale', 0.5),
                p=self.config.random_perspective.get('probability', 0.5),
                interpolation=self.config.interpolation
            ))
        
        # Random affine (training only)
        if self.training and self.config.random_affine:
            transforms.append(RandomAffine(
                degrees=self.config.random_affine.get('degrees', 0),
                translate=self.config.random_affine.get('translate', None),
                scale=self.config.random_affine.get('scale', None),
                shear=self.config.random_affine.get('shear', None),
                interpolation=self.config.interpolation
            ))
        
        # Convert to tensor
        if self.config.to_tensor:
            transforms.append(ToTensor())
        
        # Normalize
        if self.config.normalize:
            transforms.append(Normalize(mean=self.config.mean, std=self.config.std))
        
        # Random erasing (training only)
        if self.training and self.config.random_erasing:
            transforms.append(RandomErasing(
                p=self.config.random_erasing.get('probability', 0.5),
                scale=self.config.random_erasing.get('scale', (0.02, 0.33)),
                ratio=self.config.random_erasing.get('ratio', (0.3, 3.3)),
                value=self.config.random_erasing.get('value', 0)
            ))
        
        return Compose(transforms)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to sample.
        
        Args:
            sample: Input sample with 'image' key
            
        Returns:
            Transformed sample
        """
        # Apply transforms to image
        if 'image' in sample:
            sample['image'] = self.transforms(sample['image'])
        
        # Apply same transforms to mask if present
        if 'mask' in sample and isinstance(sample['mask'], (Image.Image, np.ndarray)):
            # For masks, use nearest neighbor interpolation
            mask_transform = self._build_mask_transform()
            sample['mask'] = mask_transform(sample['mask'])
        
        return sample
    
    def _build_mask_transform(self) -> Compose:
        """Build transform pipeline for masks (uses nearest neighbor)."""
        transforms = []
        
        # Resize
        if self.config.resize is not None:
            transforms.append(Resize(self.config.resize, interpolation=InterpolationMode.NEAREST))
        
        # Crop
        if self.config.crop is not None:
            if self.training and self.config.random_crop:
                transforms.append(RandomCrop(
                    size=self.config.crop,
                    padding=self.config.random_crop.get('padding', 4),
                    pad_if_needed=self.config.random_crop.get('pad_if_needed', True),
                    fill=0,  # Fill with 0 for masks
                    padding_mode=self.config.padding_mode
                ))
            elif self.config.center_crop:
                transforms.append(CenterCrop(self.config.crop))
        
        # Random flip (training only)
        if self.training and self.config.random_flip > 0:
            transforms.append(RandomHorizontalFlip(p=self.config.random_flip))
            transforms.append(RandomVerticalFlip(p=self.config.random_flip/2))
        
        # Random rotate (training only)
        if self.training and self.config.random_rotate:
            transforms.append(RandomRotation(
                degrees=self.config.random_rotate,
                interpolation=InterpolationMode.NEAREST,
                expand=False
            ))
        
        # Random perspective (training only)
        if self.training and self.config.random_perspective:
            transforms.append(RandomPerspective(
                distortion_scale=self.config.random_perspective.get('distortion_scale', 0.5),
                p=self.config.random_perspective.get('probability', 0.5),
                interpolation=InterpolationMode.NEAREST
            ))
        
        # Convert mask to tensor (as long)
        transforms.append(ToTensor(dtype=torch.long))
        
        return Compose(transforms)
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        self.transforms = self._build_transforms()


class VideoTransform:
    """
    Video transformation pipeline.
    
    Applies transforms to each frame in a video.
    """
    
    def __init__(self, config: TransformConfig, training: bool = True):
        """
        Initialize video transform.
        
        Args:
            config: Transform configuration
            training: Whether in training mode
        """
        self.config = config
        self.training = training
        self.frame_transform = ImageTransform(config, training)
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to video sample.
        
        Args:
            sample: Input sample with 'frames' key
            
        Returns:
            Transformed sample
        """
        if 'frames' not in sample:
            return sample
        
        frames = sample['frames']
        
        # Apply transforms to each frame
        if isinstance(frames, torch.Tensor):
            # Tensor shape: (T, C, H, W) or (T, H, W, C)
            transformed_frames = []
            for i in range(frames.shape[0]):
                frame = frames[i]
                
                # Convert to dict for transform
                frame_dict = {'image': frame}
                transformed = self.frame_transform(frame_dict)
                transformed_frames.append(transformed['image'])
            
            sample['frames'] = torch.stack(transformed_frames)
        
        elif isinstance(frames, list):
            # List of frames
            transformed_frames = []
            for frame in frames:
                frame_dict = {'image': frame}
                transformed = self.frame_transform(frame_dict)
                transformed_frames.append(transformed['image'])
            
            sample['frames'] = transformed_frames
        
        return sample
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training
        self.frame_transform.set_training(training)


class PointCloudTransform:
    """
    Point cloud transformation pipeline.
    
    Supports:
    - Random rotation
    - Random scaling
    - Random translation
    - Random jitter
    - Normalization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, training: bool = True):
        """
        Initialize point cloud transform.
        
        Args:
            config: Transform configuration
            training: Whether in training mode
        """
        self.config = config or {}
        self.training = training
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to point cloud.
        
        Args:
            sample: Input sample with 'points' key
            
        Returns:
            Transformed sample
        """
        if 'points' not in sample:
            return sample
        
        points = sample['points']
        
        if isinstance(points, torch.Tensor):
            points = points.clone()
        elif isinstance(points, np.ndarray):
            points = points.copy()
            points = torch.from_numpy(points).float()
        else:
            return sample
        
        # Apply transforms
        if self.training:
            # Random rotation
            if self.config.get('random_rotation', True):
                points = self._random_rotation(points)
            
            # Random scaling
            if self.config.get('random_scaling', True):
                points = self._random_scaling(points)
            
            # Random translation
            if self.config.get('random_translation', False):
                points = self._random_translation(points)
            
            # Random jitter
            if self.config.get('random_jitter', False):
                points = self._random_jitter(points)
        
        # Normalize
        if self.config.get('normalize', True):
            points = self._normalize_points(points)
        
        sample['points'] = points
        return sample
    
    def _random_rotation(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random rotation to point cloud."""
        angle = random.uniform(0, 2 * math.pi)
        
        # Create rotation matrix
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Rotate around Z-axis
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=points.dtype, device=points.device)
        
        # Apply rotation
        points = torch.matmul(points, rotation_matrix.T)
        
        return points
    
    def _random_scaling(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random scaling to point cloud."""
        scale = random.uniform(0.8, 1.2)
        points = points * scale
        return points
    
    def _random_translation(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random translation to point cloud."""
        translation = torch.randn(3) * 0.01  # Small translation
        points = points + translation
        return points
    
    def _random_jitter(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random jitter to point cloud."""
        jitter = torch.randn_like(points) * 0.01  # Small jitter
        points = points + jitter
        return points
    
    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        """Normalize point cloud to unit sphere."""
        # Center
        centroid = points.mean(dim=0)
        points = points - centroid
        
        # Scale to unit sphere
        max_dist = torch.sqrt(torch.max(torch.sum(points ** 2, dim=1)))
        if max_dist > 0:
            points = points / max_dist
        
        return points
    
    def set_training(self, training: bool):
        """Set training mode."""
        self.training = training


class TextTransform:
    """
    Text transformation pipeline.
    
    Supports:
    - Tokenization
    - Padding/truncation
    - Lowercasing
    - Stop word removal
    - Stemming/Lemmatization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text transform.
        
        Args:
            config: Transform configuration
        """
        self.config = config or {}
        
        # Initialize tokenizer if needed
        self.tokenizer = None
        if self.config.get('tokenizer') == 'spacy':
            try:
                import spacy
                self.tokenizer = spacy.load('en_core_web_sm')
            except ImportError:
                logger.warning("spaCy not installed, using simple tokenizer")
        
        # Initialize stemmer/lemmatizer if needed
        self.stemmer = None
        self.lemmatizer = None
        if self.config.get('stemming', False):
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                logger.warning("nltk not installed, stemming disabled")
        
        if self.config.get('lemmatization', False):
            try:
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
            except ImportError:
                logger.warning("nltk not installed, lemmatization disabled")
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to text.
        
        Args:
            sample: Input sample with 'text' key
            
        Returns:
            Transformed sample
        """
        if 'text' not in sample:
            return sample
        
        text = sample['text']
        
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Tokenize
        tokens = self._tokenize(text)
        
        # Remove stop words
        if self.config.get('remove_stopwords', False):
            tokens = self._remove_stopwords(tokens)
        
        # Stemming
        if self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Lemmatization
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Truncate/pad
        max_length = self.config.get('max_length')
        if max_length:
            tokens = self._truncate_or_pad(tokens, max_length)
        
        # Convert to tensor if requested
        if self.config.get('to_tensor', False):
            # Create vocabulary index
            vocab = self.config.get('vocabulary', {})
            if not vocab:
                # Create from tokens
                unique_tokens = set(tokens)
                vocab = {token: idx for idx, token in enumerate(unique_tokens)}
            
            # Convert tokens to indices
            indices = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
            tokens = torch.tensor(indices, dtype=torch.long)
        
        sample['text'] = tokens
        sample['text_processed'] = True
        
        return sample
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        if self.tokenizer:
            # Use spaCy tokenizer
            doc = self.tokenizer(text)
            return [token.text for token in doc]
        else:
            # Simple whitespace tokenizer
            return text.split()
    
    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from tokens."""
        # Basic English stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
            'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        return [token for token in tokens if token.lower() not in stop_words]
    
    def _truncate_or_pad(self, tokens: List[str], max_length: int) -> List[str]:
        """Truncate or pad tokens to max_length."""
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif len(tokens) < max_length:
            tokens = tokens + ['<PAD>'] * (max_length - len(tokens))
        
        return tokens


# Individual transform classes

class Resize:
    """Resize transform."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], 
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        """
        Initialize resize transform.
        
        Args:
            size: Target size (width, height) or single dimension
            interpolation: Interpolation mode
        """
        if isinstance(size, int):
            size = (size, size)
        
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):
        """Resize image."""
        if isinstance(img, Image.Image):
            return TF.resize(img, self.size, interpolation=self._get_pil_interpolation())
        elif isinstance(img, torch.Tensor):
            # Use torchvision functional
            return TF.resize(img, self.size, interpolation=self._get_torch_interpolation())
        else:
            # Assume numpy array
            import cv2
            interpolation = self._get_cv2_interpolation()
            return cv2.resize(img, self.size, interpolation=interpolation)
    
    def _get_pil_interpolation(self):
        """Get PIL interpolation constant."""
        if self.interpolation == InterpolationMode.NEAREST:
            return Image.NEAREST
        elif self.interpolation == InterpolationMode.BILINEAR:
            return Image.BILINEAR
        elif self.interpolation == InterpolationMode.BICUBIC:
            return Image.BICUBIC
        elif self.interpolation == InterpolationMode.LANCZOS:
            return Image.LANCZOS
        elif self.interpolation == InterpolationMode.BOX:
            return Image.BOX
        elif self.interpolation == InterpolationMode.HAMMING:
            return Image.HAMMING
        else:
            return Image.BILINEAR
    
    def _get_torch_interpolation(self):
        """Get torchvision interpolation string."""
        return self.interpolation.value
    
    def _get_cv2_interpolation(self):
        """Get OpenCV interpolation constant."""
        if self.interpolation == InterpolationMode.NEAREST:
            return cv2.INTER_NEAREST
        elif self.interpolation == InterpolationMode.BILINEAR:
            return cv2.INTER_LINEAR
        elif self.interpolation == InterpolationMode.BICUBIC:
            return cv2.INTER_CUBIC
        elif self.interpolation == InterpolationMode.LANCZOS:
            return cv2.INTER_LANCZOS4
        else:
            return cv2.INTER_LINEAR


class RandomCrop:
    """Random crop transform."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], padding: int = None,
                 pad_if_needed: bool = False, fill: int = 0, 
                 padding_mode: PaddingMode = PaddingMode.CONSTANT):
        """
        Initialize random crop transform.
        
        Args:
            size: Crop size (width, height)
            padding: Optional padding amount
            pad_if_needed: Pad if image is smaller than crop size
            fill: Fill value for padding
            padding_mode: Padding mode
        """
        if isinstance(size, int):
            size = (size, size)
        
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, img):
        """Apply random crop."""
        if isinstance(img, Image.Image):
            # Apply padding if specified
            if self.padding is not None:
                img = TF.pad(img, self.padding, fill=self.fill, 
                           padding_mode=self.padding_mode.value)
            
            # Pad if needed
            if self.pad_if_needed:
                width, height = img.size
                if width < self.size[0]:
                    padding = [self.size[0] - width, 0]
                    img = TF.pad(img, padding, fill=self.fill,
                               padding_mode=self.padding_mode.value)
                if height < self.size[1]:
                    padding = [0, self.size[1] - height]
                    img = TF.pad(img, padding, fill=self.fill,
                               padding_mode=self.padding_mode.value)
            
            # Get crop parameters
            width, height = img.size
            top = random.randint(0, height - self.size[1])
            left = random.randint(0, width - self.size[0])
            
            return TF.crop(img, top, left, self.size[1], self.size[0])
        
        elif isinstance(img, torch.Tensor):
            # Use torchvision functional
            if self.padding is not None:
                img = TF.pad(img, self.padding, fill=self.fill,
                           padding_mode=self.padding_mode.value)
            
            # Get crop parameters
            _, height, width = img.shape
            top = random.randint(0, height - self.size[1])
            left = random.randint(0, width - self.size[0])
            
            return TF.crop(img, top, left, self.size[1], self.size[0])
        
        else:
            # Assume numpy array
            height, width = img.shape[:2]
            
            # Apply padding if needed
            if self.pad_if_needed and (height < self.size[1] or width < self.size[0]):
                import cv2
                pad_h = max(0, self.size[1] - height)
                pad_w = max(0, self.size[0] - width)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                padding_mode = self._get_cv2_padding_mode()
                img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                       padding_mode, value=self.fill)
            
            # Get crop parameters
            height, width = img.shape[:2]
            top = random.randint(0, height - self.size[1])
            left = random.randint(0, width - self.size[0])
            
            return img[top:top+self.size[1], left:left+self.size[0]]
    
    def _get_cv2_padding_mode(self):
        """Get OpenCV padding constant."""
        if self.padding_mode == PaddingMode.CONSTANT:
            return cv2.BORDER_CONSTANT
        elif self.padding_mode == PaddingMode.EDGE:
            return cv2.BORDER_REPLICATE
        elif self.padding_mode == PaddingMode.REFLECT:
            return cv2.BORDER_REFLECT
        elif self.padding_mode == PaddingMode.SYMMETRIC:
            return cv2.BORDER_REFLECT_101
        else:
            return cv2.BORDER_CONSTANT


class CenterCrop:
    """Center crop transform."""
    
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Initialize center crop transform.
        
        Args:
            size: Crop size (width, height)
        """
        if isinstance(size, int):
            size = (size, size)
        
        self.size = size
    
    def __call__(self, img):
        """Apply center crop."""
        if isinstance(img, Image.Image):
            return TF.center_crop(img, self.size)
        elif isinstance(img, torch.Tensor):
            return TF.center_crop(img, self.size)
        else:
            # Assume numpy array
            height, width = img.shape[:2]
            top = (height - self.size[1]) // 2
            left = (width - self.size[0]) // 2
            return img[top:top+self.size[1], left:left+self.size[0]]


class RandomHorizontalFlip:
    """Random horizontal flip transform."""
    
    def __init__(self, p: float = 0.5):
        """
        Initialize random horizontal flip.
        
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, img):
        """Apply random horizontal flip."""
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                return TF.hflip(img)
            elif isinstance(img, torch.Tensor):
                return TF.hflip(img)
            else:
                # Assume numpy array
                return np.fliplr(img)
        return img


class RandomVerticalFlip:
    """Random vertical flip transform."""
    
    def __init__(self, p: float = 0.5):
        """
        Initialize random vertical flip.
        
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, img):
        """Apply random vertical flip."""
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                return TF.vflip(img)
            elif isinstance(img, torch.Tensor):
                return TF.vflip(img)
            else:
                # Assume numpy array
                return np.flipud(img)
        return img


class ColorJitter:
    """Color jitter transform."""
    
    def __init__(self, brightness: float = 0, contrast: float = 0,
                 saturation: float = 0, hue: float = 0):
        """
        Initialize color jitter.
        
        Args:
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            hue: Hue jitter factor
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img):
        """Apply color jitter."""
        if isinstance(img, Image.Image):
            # Apply each transformation
            if self.brightness > 0:
                factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(factor)
            
            if self.contrast > 0:
                factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(factor)
            
            if self.saturation > 0:
                factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(factor)
            
            if self.hue > 0:
                # Convert to HSV, adjust hue, convert back
                import colorsys
                img = img.convert('RGB')
                data = np.array(img) / 255.0
                
                # Convert to HSV
                hsv = np.apply_along_axis(colorsys.rgb_to_hsv, 2, data)
                
                # Adjust hue
                hue_shift = random.uniform(-self.hue, self.hue)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 1.0
                
                # Convert back to RGB
                rgb = np.apply_along_axis(colorsys.hsv_to_rgb, 2, hsv)
                rgb = (rgb * 255).astype(np.uint8)
                
                img = Image.fromarray(rgb)
            
            return img
        
        elif isinstance(img, torch.Tensor):
            # Use torchvision functional
            return TF.adjust_brightness(img, random.uniform(max(0, 1 - self.brightness), 1 + self.brightness))
            # Note: Full color jitter implementation for tensors would be more complex
        
        else:
            # Assume numpy array
            import cv2
            img = img.copy()
            
            if self.brightness > 0:
                factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            
            if self.contrast > 0:
                factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            
            # Saturation and hue adjustment for numpy arrays would require HSV conversion
            # Similar to PIL implementation but with OpenCV
            
            return img


class RandomRotation:
    """Random rotation transform."""
    
    def __init__(self, degrees: Union[float, Tuple[float, float]],
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 expand: bool = False):
        """
        Initialize random rotation.
        
        Args:
            degrees: Rotation degrees or range
            interpolation: Interpolation mode
            expand: Whether to expand image to fit rotation
        """
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)
        
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand
    
    def __call__(self, img):
        """Apply random rotation."""
        angle = random.uniform(self.degrees[0], self.degrees[1])
        
        if isinstance(img, Image.Image):
            return TF.rotate(img, angle, interpolation=self._get_pil_interpolation(),
                           expand=self.expand)
        elif isinstance(img, torch.Tensor):
            return TF.rotate(img, angle, interpolation=self._get_torch_interpolation(),
                           expand=self.expand)
        else:
            # Assume numpy array
            import cv2
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            if self.expand:
                # Calculate new image dimensions
                cos = abs(rotation_matrix[0, 0])
                sin = abs(rotation_matrix[0, 1])
                
                new_width = int((height * sin) + (width * cos))
                new_height = int((height * cos) + (width * sin))
                
                # Adjust rotation matrix
                rotation_matrix[0, 2] += (new_width / 2) - center[0]
                rotation_matrix[1, 2] += (new_height / 2) - center[1]
                
                width, height = new_width, new_height
            
            interpolation = self._get_cv2_interpolation()
            return cv2.warpAffine(img, rotation_matrix, (width, height),
                                flags=interpolation)
    
    def _get_pil_interpolation(self):
        """Get PIL interpolation constant."""
        if self.interpolation == InterpolationMode.NEAREST:
            return Image.NEAREST
        elif self.interpolation == InterpolationMode.BILINEAR:
            return Image.BILINEAR
        else:
            return Image.BILINEAR
    
    def _get_torch_interpolation(self):
        """Get torchvision interpolation string."""
        return self.interpolation.value
    
    def _get_cv2_interpolation(self):
        """Get OpenCV interpolation constant."""
        if self.interpolation == InterpolationMode.NEAREST:
            return cv2.INTER_NEAREST
        else:
            return cv2.INTER_LINEAR


class GaussianBlur:
    """Gaussian blur transform."""
    
    def __init__(self, kernel_size: int = 3, sigma: Union[float, Tuple[float, float]] = (0.1, 2.0)):
        """
        Initialize Gaussian blur.
        
        Args:
            kernel_size: Kernel size (must be odd)
            sigma: Sigma value or range
        """
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        """Apply Gaussian blur."""
        if isinstance(img, Image.Image):
            # PIL implementation
            if isinstance(self.sigma, (tuple, list)):
                sigma = random.uniform(self.sigma[0], self.sigma[1])
            else:
                sigma = self.sigma
            
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        elif isinstance(img, torch.Tensor):
            # Use torchvision functional
            if isinstance(self.sigma, (tuple, list)):
                sigma = random.uniform(self.sigma[0], self.sigma[1])
            else:
                sigma = self.sigma
            
            return TF.gaussian_blur(img, kernel_size=self.kernel_size, sigma=sigma)
        
        else:
            # Assume numpy array
            import cv2
            
            if isinstance(self.sigma, (tuple, list)):
                sigma = random.uniform(self.sigma[0], self.sigma[1])
            else:
                sigma = self.sigma
            
            # Ensure kernel size is odd
            kernel_size = self.kernel_size
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


class RandomPerspective:
    """Random perspective transform."""
    
    def __init__(self, distortion_scale: float = 0.5, p: float = 0.5,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        """
        Initialize random perspective.
        
        Args:
            distortion_scale: Distortion scale
            p: Probability of applying transform
            interpolation: Interpolation mode
        """
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation
    
    def __call__(self, img):
        """Apply random perspective."""
        if random.random() < self.p:
            if isinstance(img, Image.Image):
                return TF.perspective(img, self._get_perspective_coefficients(img.size),
                                    interpolation=self._get_pil_interpolation())
            elif isinstance(img, torch.Tensor):
                return TF.perspective(img, self._get_perspective_coefficients(img.shape[-2:]),
                                    interpolation=self._get_torch_interpolation())
        
        return img
    
    def _get_perspective_coefficients(self, size):
        """Generate perspective transform coefficients."""
        width, height = size if isinstance(size, tuple) else (size[1], size[0])
        
        half_height = height // 2
        half_width = width // 2
        
        topleft = [
            random.randint(0, int(self.distortion_scale * half_width)),
            random.randint(0, int(self.distortion_scale * half_height))
        ]
        topright = [
            width - random.randint(0, int(self.distortion_scale * half_width)),
            random.randint(0, int(self.distortion_scale * half_height))
        ]
        botright = [
            width - random.randint(0, int(self.distortion_scale * half_width)),
            height - random.randint(0, int(self.distortion_scale * half_height))
        ]
        botleft = [
            random.randint(0, int(self.distortion_scale * half_width)),
            height - random.randint(0, int(self.distortion_scale * half_height))
        ]
        
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        
        return startpoints, endpoints
    
    def _get_pil_interpolation(self):
        """Get PIL interpolation constant."""
        if self.interpolation == InterpolationMode.NEAREST:
            return Image.NEAREST
        else:
            return Image.BILINEAR
    
    def _get_torch_interpolation(self):
        """Get torchvision interpolation string."""
        return self.interpolation.value


class RandomAffine:
    """Random affine transform."""
    
    def __init__(self, degrees: float = 0, translate: Optional[Tuple[float, float]] = None,
                 scale: Optional[Tuple[float, float]] = None, shear: Optional[float] = None,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        """
        Initialize random affine.
        
        Args:
            degrees: Rotation degrees
            translate: Translation range
            scale: Scale range
            shear: Shear range
            interpolation: Interpolation mode
        """
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
    
    def __call__(self, img):
        """Apply random affine."""
        if isinstance(img, Image.Image):
            return TF.affine(img, self._get_affine_parameters(img.size),
                           interpolation=self._get_pil_interpolation())
        elif isinstance(img, torch.Tensor):
            return TF.affine(img, self._get_affine_parameters(img.shape[-2:]),
                           interpolation=self._get_torch_interpolation())
        return img
    
    def _get_affine_parameters(self, size):
        """Generate affine transform parameters."""
        angle = random.uniform(-self.degrees, self.degrees) if self.degrees else 0
        
        if self.translate:
            max_dx = self.translate[0] * size[0] if isinstance(size, tuple) else self.translate[0] * size[1]
            max_dy = self.translate[1] * size[1] if isinstance(size, tuple) else self.translate[1] * size[0]
            translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
        else:
            translations = (0, 0)
        
        scale = random.uniform(self.scale[0], self.scale[1]) if self.scale else 1.0
        
        shear_x = shear_y = 0
        if self.shear:
            shear_x = random.uniform(-self.shear, self.shear)
            if isinstance(self.shear, (tuple, list)):
                shear_y = random.uniform(self.shear[0], self.shear[1])
        
        return angle, translations, scale, (shear_x, shear_y)
    
    def _get_pil_interpolation(self):
        """Get PIL interpolation constant."""
        if self.interpolation == InterpolationMode.NEAREST:
            return Image.NEAREST
        else:
            return Image.BILINEAR
    
    def _get_torch_interpolation(self):
        """Get torchvision interpolation string."""
        return self.interpolation.value


class ToTensor:
    """Convert to tensor transform."""
    
    def __init__(self, dtype: Optional[torch.dtype] = None):
        """
        Initialize to tensor transform.
        
        Args:
            dtype: Target dtype
        """
        self.dtype = dtype
    
    def __call__(self, img):
        """Convert to tensor."""
        if isinstance(img, Image.Image):
            tensor = TF.to_tensor(img)
        elif isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)  # Add channel dimension
            elif tensor.ndim == 3 and tensor.shape[2] <= 4:
                # HWC to CHW
                tensor = tensor.permute(2, 0, 1)
        elif isinstance(img, torch.Tensor):
            tensor = img
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        
        if self.dtype:
            tensor = tensor.to(self.dtype)
        
        return tensor


class Normalize:
    """Normalize transform."""
    
    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        """
        Initialize normalize transform.
        
        Args:
            mean: Mean values
            std: Standard deviation values
            inplace: Whether to modify in place
        """
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, tensor):
        """Normalize tensor."""
        return TF.normalize(tensor, self.mean, self.std, self.inplace)


class RandomErasing:
    """Random erasing transform."""
    
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3), value: Union[float, List[float]] = 0):
        """
        Initialize random erasing.
        
        Args:
            p: Probability of erasing
            scale: Range of area proportion of erased region
            ratio: Range of aspect ratio of erased region
            value: Erasing value
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        """Apply random erasing."""
        if random.random() < self.p:
            if isinstance(img, torch.Tensor):
                return TF.erase(img, *self._get_erasing_parameters(img.shape))
        return img
    
    def _get_erasing_parameters(self, shape):
        """Get erasing parameters for tensor."""
        _, h, w = shape
        
        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erase_w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        if erase_h < h and erase_w < w:
            i = random.randint(0, h - erase_h)
            j = random.randint(0, w - erase_w)
            
            return i, j, erase_h, erase_w, self.value
        
        return 0, 0, h, w, self.value


# Factory functions

def create_image_transform(config: TransformConfig, training: bool = True) -> ImageTransform:
    """
    Create image transform pipeline.
    
    Args:
        config: Transform configuration
        training: Whether in training mode
        
    Returns:
        ImageTransform instance
    """
    return ImageTransform(config, training)

def create_video_transform(config: TransformConfig, training: bool = True) -> VideoTransform:
    """
    Create video transform pipeline.
    
    Args:
        config: Transform configuration
        training: Whether in training mode
        
    Returns:
        VideoTransform instance
    """
    return VideoTransform(config, training)

def create_point_cloud_transform(config: Optional[Dict[str, Any]] = None, 
                                training: bool = True) -> PointCloudTransform:
    """
    Create point cloud transform pipeline.
    
    Args:
        config: Transform configuration
        training: Whether in training mode
        
    Returns:
        PointCloudTransform instance
    """
    return PointCloudTransform(config, training)

def create_text_transform(config: Optional[Dict[str, Any]] = None) -> TextTransform:
    """
    Create text transform pipeline.
    
    Args:
        config: Transform configuration
        
    Returns:
        TextTransform instance
    """
    return TextTransform(config)

def create_multimodal_transform(modality_transforms: Dict[str, Callable]) -> Compose:
    """
    Create multimodal transform pipeline.
    
    Args:
        modality_transforms: Dictionary mapping modality names to transforms
        
    Returns:
        Compose transform
    """
    def multimodal_transform(sample):
        for modality, transform in modality_transforms.items():
            if modality in sample:
                sample[modality] = transform(sample[modality])
        return sample
    
    return multimodal_transform