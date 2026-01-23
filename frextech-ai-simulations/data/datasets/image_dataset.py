"""
Image Dataset Module
Handles loading and processing of image datasets for 2D image generation.
"""

import os
import json
import csv
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import torch
import numpy as np
from PIL import Image, ImageFile
import cv2

from .base_dataset import BaseDataset, DatasetConfig, DatasetPhase, DatasetType

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFormat(Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    GIF = "gif"

@dataclass
class ImageDatasetConfig(DatasetConfig):
    """
    Configuration for image datasets.
    
    Attributes:
        image_size: Target image size (width, height) or single dimension for square
        channels: Number of image channels (1 for grayscale, 3 for RGB, 4 for RGBA)
        normalize: Whether to normalize images to [0, 1] or [-1, 1]
        normalize_range: Range for normalization: "0_1" or "-1_1"
        resize_mode: Resize mode: "crop", "pad", "stretch", "center_crop"
        color_mode: Color mode: "rgb", "bgr", "grayscale", "rgba"
        exif_orientation: Whether to apply EXIF orientation
        load_as_tensor: Whether to load images as PyTorch tensors
        cache_images: Whether to cache images in memory
        image_format: Expected image format
        annotation_type: Type of annotations: "classification", "detection", "segmentation", "caption"
        annotation_file: Path to annotation file
        captions_field: Field name for captions in metadata
        labels_field: Field name for labels in metadata
        bounding_boxes_field: Field name for bounding boxes
        segmentation_masks_field: Field name for segmentation masks
    """
    image_size: Union[int, Tuple[int, int]] = 256
    channels: int = 3
    normalize: bool = True
    normalize_range: str = "-1_1"  # "0_1" or "-1_1"
    resize_mode: str = "center_crop"  # "crop", "pad", "stretch", "center_crop"
    color_mode: str = "rgb"  # "rgb", "bgr", "grayscale", "rgba"
    exif_orientation: bool = True
    load_as_tensor: bool = True
    cache_images: bool = False
    image_format: Optional[str] = None
    annotation_type: str = "classification"
    annotation_file: Optional[str] = None
    captions_field: str = "caption"
    labels_field: str = "label"
    bounding_boxes_field: str = "bbox"
    segmentation_masks_field: str = "mask"
    
    def __post_init__(self):
        """Validate image dataset configuration."""
        super().__post_init__()
        
        # Set dataset type
        self.type = DatasetType.IMAGE
        
        # Validate image size
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        
        # Validate channels
        if self.channels not in [1, 3, 4]:
            raise ValueError(f"channels must be 1, 3, or 4, got {self.channels}")
        
        # Validate color mode
        valid_color_modes = ["rgb", "bgr", "grayscale", "rgba"]
        if self.color_mode not in valid_color_modes:
            raise ValueError(f"color_mode must be one of {valid_color_modes}, got {self.color_mode}")
        
        # Validate resize mode
        valid_resize_modes = ["crop", "pad", "stretch", "center_crop"]
        if self.resize_mode not in valid_resize_modes:
            raise ValueError(f"resize_mode must be one of {valid_resize_modes}, got {self.resize_mode}")
        
        # Validate normalize range
        if self.normalize_range not in ["0_1", "-1_1"]:
            raise ValueError(f"normalize_range must be '0_1' or '-1_1', got {self.normalize_range}")
        
        # Validate annotation type
        valid_annotation_types = ["classification", "detection", "segmentation", "caption", "none"]
        if self.annotation_type not in valid_annotation_types:
            raise ValueError(f"annotation_type must be one of {valid_annotation_types}, got {self.annotation_type}")


class ImageDataset(BaseDataset):
    """
    Dataset for handling image data with various annotation types.
    
    Supports:
    - Classification: Single label per image
    - Detection: Bounding boxes with labels
    - Segmentation: Pixel-wise masks
    - Captioning: Text descriptions
    """
    
    def __init__(self, config: ImageDatasetConfig):
        """
        Initialize image dataset.
        
        Args:
            config: Image dataset configuration
        """
        if not isinstance(config, ImageDatasetConfig):
            config = ImageDatasetConfig.from_dict(config)
        
        super().__init__(config)
        
        # Image cache
        self.image_cache = {}
        self.cache_images = config.cache_images
        
        # Image format detection
        self.supported_formats = [fmt.value for fmt in ImageFormat]
        if config.image_format:
            if config.image_format not in self.supported_formats:
                logger.warning(f"Image format {config.image_format} not in supported formats: {self.supported_formats}")
        
        # Annotation handling
        self.annotation_type = config.annotation_type
        self.captions_field = config.captions_field
        self.labels_field = config.labels_field
        self.bounding_boxes_field = config.bounding_boxes_field
        self.segmentation_masks_field = config.segmentation_masks_field
        
        # Load annotations if provided
        if config.annotation_file:
            self._load_annotations()
    
    def _load_dataset(self) -> None:
        """Load image dataset samples."""
        # Check if metadata file exists
        if self.metadata_file and self.metadata_file.exists():
            self._load_from_metadata()
        else:
            self._load_from_directory()
        
        logger.info(f"Loaded {len(self.samples)} image samples")
    
    def _load_from_metadata(self) -> None:
        """Load dataset from metadata file."""
        suffix = self.metadata_file.suffix.lower()
        
        try:
            if suffix == '.json':
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                if isinstance(metadata, list):
                    self.samples = metadata
                elif isinstance(metadata, dict) and 'samples' in metadata:
                    self.samples = metadata['samples']
                    self.metadata = {k: v for k, v in metadata.items() if k != 'samples'}
                else:
                    raise ValueError(f"Invalid metadata format in {self.metadata_file}")
            
            elif suffix == '.csv':
                with open(self.metadata_file, 'r') as f:
                    reader = csv.DictReader(f)
                    self.samples = list(reader)
            
            else:
                raise ValueError(f"Unsupported metadata format: {suffix}")
        
        except Exception as e:
            logger.error(f"Failed to load metadata from {self.metadata_file}: {e}")
            raise
        
        # Convert paths to absolute paths relative to data_dir
        for sample in self.samples:
            if 'image_path' in sample:
                sample['image_path'] = str(self.data_dir / sample['image_path'])
            elif 'path' in sample:
                sample['image_path'] = str(self.data_dir / sample['path'])
    
    def _load_from_directory(self) -> None:
        """Load dataset from directory structure."""
        self.samples = []
        
        # Supported image extensions
        extensions = {f'.{fmt.value}' for fmt in ImageFormat}
        
        # Recursively find image files
        for ext in extensions:
            image_files = list(self.data_dir.rglob(f'*{ext}'))
            
            for image_path in image_files:
                # Skip hidden files and directories
                if image_path.name.startswith('.'):
                    continue
                
                sample = {
                    'image_path': str(image_path),
                    'filename': image_path.name,
                    'relative_path': str(image_path.relative_to(self.data_dir))
                }
                
                # Try to extract label from directory structure
                parent_dir = image_path.parent
                if parent_dir != self.data_dir:
                    sample[self.labels_field] = parent_dir.name
                
                self.samples.append(sample)
        
        # Sort by filename for reproducibility
        self.samples.sort(key=lambda x: x['image_path'])
    
    def _load_annotations(self) -> None:
        """Load additional annotations from annotation file."""
        annotation_path = Path(self.config.annotation_file)
        
        if not annotation_path.exists():
            logger.warning(f"Annotation file not found: {annotation_path}")
            return
        
        # Create mapping from image path to annotations
        annotations_map = {}
        
        try:
            if annotation_path.suffix == '.json':
                with open(annotation_path, 'r') as f:
                    annotations = json.load(f)
                
                if isinstance(annotations, list):
                    for ann in annotations:
                        img_path = str(self.data_dir / ann['image_id'])
                        annotations_map[img_path] = ann
                elif isinstance(annotations, dict):
                    # COCO format
                    if 'images' in annotations and 'annotations' in annotations:
                        # Map image id to file name
                        image_id_to_path = {}
                        for img in annotations['images']:
                            img_path = str(self.data_dir / img['file_name'])
                            image_id_to_path[img['id']] = img_path
                        
                        # Group annotations by image
                        for ann in annotations['annotations']:
                            img_path = image_id_to_path[ann['image_id']]
                            if img_path not in annotations_map:
                                annotations_map[img_path] = []
                            annotations_map[img_path].append(ann)
                
            elif annotation_path.suffix == '.csv':
                with open(annotation_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_path = str(self.data_dir / row['image_path'])
                        annotations_map[img_path] = row
            
            else:
                logger.warning(f"Unsupported annotation format: {annotation_path.suffix}")
                return
        
        except Exception as e:
            logger.error(f"Failed to load annotations: {e}")
            return
        
        # Merge annotations with samples
        for sample in self.samples:
            img_path = sample['image_path']
            if img_path in annotations_map:
                sample.update(annotations_map[img_path])
    
    def _get_required_fields(self) -> List[str]:
        """Get required fields for image samples."""
        return ['image_path']
    
    def _load_sample(self, index: int) -> Dict[str, Any]:
        """Load a single image sample."""
        sample_info = self.samples[index].copy()
        image_path = sample_info['image_path']
        
        # Check image cache
        if self.cache_images and image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            # Load image
            image = self._load_image(image_path)
            
            # Cache if enabled
            if self.cache_images:
                self.image_cache[image_path] = image
        
        # Add image to sample
        sample_info['image'] = image
        
        # Load additional data based on annotation type
        if self.annotation_type == "segmentation" and self.segmentation_masks_field in sample_info:
            mask_path = sample_info[self.segmentation_masks_field]
            mask = self._load_mask(mask_path)
            sample_info['mask'] = mask
        
        return sample_info
    
    def _load_image(self, image_path: str) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Loaded image
        """
        try:
            # Open image
            with Image.open(image_path) as img:
                # Apply EXIF orientation if requested
                if self.config.exif_orientation:
                    img = self._apply_exif_orientation(img)
                
                # Convert color mode
                img = self._convert_color_mode(img)
                
                # Resize/crop
                img = self._resize_image(img)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Convert to tensor if requested
                if self.config.load_as_tensor:
                    img_tensor = torch.from_numpy(img_array).float()
                    
                    # Normalize
                    if self.config.normalize:
                        if self.config.normalize_range == "0_1":
                            img_tensor = img_tensor / 255.0
                        else:  # "-1_1"
                            img_tensor = (img_tensor / 127.5) - 1.0
                    
                    # Rearrange dimensions to (C, H, W)
                    if len(img_tensor.shape) == 3:  # H, W, C
                        img_tensor = img_tensor.permute(2, 0, 1)
                    elif len(img_tensor.shape) == 2:  # H, W
                        img_tensor = img_tensor.unsqueeze(0)  # Add channel dimension
                    
                    return img_tensor
                else:
                    return img_array
        
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return blank image as fallback
            if self.config.load_as_tensor:
                return torch.zeros((self.config.channels, *self.config.image_size))
            else:
                return np.zeros((*self.config.image_size, self.config.channels), dtype=np.uint8)
    
    def _apply_exif_orientation(self, image: Image.Image) -> Image.Image:
        """Apply EXIF orientation to image."""
        try:
            exif = image._getexif()
            if exif is None:
                return image
            
            orientation = exif.get(0x0112, 1)
            
            if orientation == 2:
                # Mirror horizontally
                return image.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # Rotate 180 degrees
                return image.transpose(Image.ROTATE_180)
            elif orientation == 4:
                # Mirror vertically
                return image.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                # Mirror horizontally and rotate 270 CW
                return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
            elif orientation == 6:
                # Rotate 270 CW
                return image.transpose(Image.ROTATE_270)
            elif orientation == 7:
                # Mirror horizontally and rotate 90 CW
                return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
            elif orientation == 8:
                # Rotate 90 CW
                return image.transpose(Image.ROTATE_90)
            
            return image
        except Exception:
            return image
    
    def _convert_color_mode(self, image: Image.Image) -> Image.Image:
        """Convert image to specified color mode."""
        if self.config.color_mode == "grayscale":
            if image.mode != "L":
                image = image.convert("L")
        elif self.config.color_mode == "rgb":
            if image.mode != "RGB":
                image = image.convert("RGB")
        elif self.config.color_mode == "rgba":
            if image.mode != "RGBA":
                image = image.convert("RGBA")
        elif self.config.color_mode == "bgr":
            # PIL doesn't have BGR mode, convert to RGB and swap channels later
            if image.mode != "RGB":
                image = image.convert("RGB")
        
        return image
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image according to configuration."""
        target_width, target_height = self.config.image_size
        img_width, img_height = image.size
        
        if self.config.resize_mode == "stretch":
            # Simple resize
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        elif self.config.resize_mode == "center_crop":
            # Center crop to target aspect ratio, then resize
            aspect_ratio = target_width / target_height
            img_aspect_ratio = img_width / img_height
            
            if img_aspect_ratio > aspect_ratio:
                # Image is wider than target
                new_width = int(img_height * aspect_ratio)
                left = (img_width - new_width) // 2
                image = image.crop((left, 0, left + new_width, img_height))
            else:
                # Image is taller than target
                new_height = int(img_width / aspect_ratio)
                top = (img_height - new_height) // 2
                image = image.crop((0, top, img_width, top + new_height))
            
            return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        elif self.config.resize_mode == "crop":
            # Random crop during training, center crop during validation/test
            if self.phase == DatasetPhase.TRAIN and self.config.augment:
                # Random crop
                left = self.rng.randint(0, img_width - target_width)
                top = self.rng.randint(0, img_height - target_height)
                image = image.crop((left, top, left + target_width, top + target_height))
            else:
                # Center crop
                left = (img_width - target_width) // 2
                top = (img_height - target_height) // 2
                image = image.crop((left, top, left + target_width, top + target_height))
            
            return image
        
        elif self.config.resize_mode == "pad":
            # Pad to target size
            new_image = Image.new(image.mode, (target_width, target_height), (0, 0, 0))
            left = (target_width - img_width) // 2
            top = (target_height - img_height) // 2
            new_image.paste(image, (left, top))
            return new_image
        
        else:
            raise ValueError(f"Unknown resize mode: {self.config.resize_mode}")
    
    def _load_mask(self, mask_path: str) -> torch.Tensor:
        """Load segmentation mask."""
        try:
            # Load mask as grayscale
            with Image.open(mask_path) as mask_img:
                mask_img = mask_img.convert("L")
                mask_img = mask_img.resize(self.config.image_size, Image.Resampling.NEAREST)
                mask_array = np.array(mask_img)
                
                # Convert to tensor
                mask_tensor = torch.from_numpy(mask_array).long()
                return mask_tensor
        
        except Exception as e:
            logger.error(f"Failed to load mask {mask_path}: {e}")
            return torch.zeros(self.config.image_size, dtype=torch.long)
    
    def get_image_stats(self) -> Dict[str, Any]:
        """Compute image statistics (mean, std) across dataset."""
        if self.mean is not None and self.std is not None:
            return {"mean": self.mean, "std": self.std}
        
        # Compute statistics
        mean = np.zeros(self.config.channels)
        std = np.zeros(self.config.channels)
        count = 0
        
        for i in range(len(self)):
            sample = self.get_sample(i, apply_transform=False)
            image = sample['image']
            
            if isinstance(image, torch.Tensor):
                img_np = image.numpy()
            elif isinstance(image, np.ndarray):
                img_np = image
            else:
                continue
            
            # Reshape to (C, H*W)
            if len(img_np.shape) == 3:
                img_flat = img_np.reshape(img_np.shape[0], -1)
                mean += img_flat.mean(axis=1)
                std += img_flat.std(axis=1)
                count += 1
        
        if count > 0:
            mean /= count
            std /= count
        
        self.mean = mean.tolist()
        self.std = std.tolist()
        
        return {"mean": self.mean, "std": self.std}
    
    def visualize_sample(self, index: int, save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize a sample.
        
        Args:
            index: Sample index
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        sample = self.get_sample(index, apply_transform=False)
        image = sample['image']
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            # Denormalize if needed
            if self.config.normalize:
                if self.config.normalize_range == "-1_1":
                    image = (image + 1) / 2
                image = image * 255
            
            # Convert to HWC format
            if len(image.shape) == 3:
                image = image.permute(1, 2, 0)
            image = image.byte().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        
        if len(image.shape) == 3 and image.shape[2] == 1:
            axes.imshow(image[:, :, 0], cmap='gray')
        elif len(image.shape) == 3:
            # Convert BGR to RGB if needed
            if self.config.color_mode == "bgr":
                image = image[:, :, ::-1]
            axes.imshow(image)
        else:
            axes.imshow(image, cmap='gray')
        
        axes.axis('off')
        
        # Add caption if available
        if self.captions_field in sample:
            axes.set_title(sample[self.captions_field], fontsize=12)
        
        # Add label if available
        if self.labels_field in sample:
            axes.text(0.5, -0.1, f"Label: {sample[self.labels_field]}", 
                     transform=axes.transAxes, ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            return Image.open(save_path)
        else:
            # Convert to PIL Image
            plt.tight_layout()
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return Image.fromarray(image_array)
    
    def create_augmentation_transform(self) -> Callable:
        """Create augmentation transform for training."""
        from torchvision import transforms
        
        augmentation_config = self.config.augmentation_config or {}
        
        transform_list = []
        
        # Random horizontal flip
        if augmentation_config.get('random_flip', True):
            transform_list.append(transforms.RandomHorizontalFlip())
        
        # Random rotation
        if 'random_rotate' in augmentation_config:
            angle = augmentation_config.get('rotation_range', 30)
            transform_list.append(transforms.RandomRotation(angle))
        
        # Color jitter
        if augmentation_config.get('color_jitter', True):
            brightness = augmentation_config.get('brightness', 0.2)
            contrast = augmentation_config.get('contrast', 0.2)
            saturation = augmentation_config.get('saturation', 0.2)
            hue = augmentation_config.get('hue', 0.1)
            transform_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                      saturation=saturation, hue=hue)
            )
        
        # Random affine
        if 'random_affine' in augmentation_config:
            translate = augmentation_config.get('translate', (0.1, 0.1))
            scale = augmentation_config.get('scale', (0.9, 1.1))
            shear = augmentation_config.get('shear', 10)
            transform_list.append(
                transforms.RandomAffine(degrees=0, translate=translate, scale=scale, shear=shear)
            )
        
        # Random perspective
        if augmentation_config.get('random_perspective', False):
            distortion_scale = augmentation_config.get('distortion_scale', 0.5)
            transform_list.append(
                transforms.RandomPerspective(distortion_scale=distortion_scale, p=0.5)
            )
        
        # Gaussian blur
        if augmentation_config.get('gaussian_blur', False):
            kernel_size = augmentation_config.get('gaussian_kernel_size', 3)
            sigma = augmentation_config.get('gaussian_sigma', (0.1, 2.0))
            transform_list.append(
                transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            )
        
        return transforms.Compose(transform_list)