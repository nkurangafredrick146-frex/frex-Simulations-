"""
Style Transfer Module
Transfers artistic styles between 3D scenes and objects
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
import hashlib
from collections import defaultdict

# Image processing
import cv2
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

# Local imports
from ...utils.metrics import Timer
from ...utils.file_io import save_json, load_json, save_image, load_image

logger = logging.getLogger(__name__)


class StyleTransferMethod(Enum):
    """Methods for style transfer"""
    NEURAL_STYLE = "neural_style"
    ADAIN = "adain"
    WCT = "wct"
    PATCH_BASED = "patch_based"
    COLOR_TRANSFER = "color_transfer"


class StyleDomain(Enum):
    """Domains for style transfer"""
    REALISTIC = "realistic"
    PAINTING = "painting"
    SKETCH = "sketch"
    CARTOON = "cartoon"
    ANIME = "anime"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"
    SCIFI = "scifi"


@dataclass
class StylePreset:
    """Style preset configuration"""
    name: str
    domain: StyleDomain
    parameters: Dict[str, Any]
    color_palette: List[Tuple[int, int, int]]
    texture_patterns: List[str] = field(default_factory=list)
    lighting_conditions: Dict[str, float] = field(default_factory=dict)
    material_properties: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Generate hash for style preset"""
        content = f"{self.name}_{self.domain.value}_{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class NeuralStyleTransfer:
    """Neural Style Transfer using VGG19"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = self._build_vgg19().to(self.device)
        self.model.eval()
        
        # Style layers and weights
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        self.style_weights = [1.0, 0.8, 0.5, 0.3, 0.1]
        
        # Content layer
        self.content_layer = 'conv4_2'
        
        # Normalization mean and std
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        
        logger.info(f"NeuralStyleTransfer initialized on {device}")
    
    def _build_vgg19(self) -> nn.Module:
        """Build VGG19 model with feature extraction"""
        vgg = models.vgg19(pretrained=True).features
        
        # Freeze all parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        return vgg
    
    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from image at different layers"""
        features = {}
        x = image
        
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.style_layers or name == self.content_layer:
                features[name] = x
        
        return features
    
    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        """Calculate Gram matrix for style representation"""
        b, c, h, w = tensor.size()
        features = tensor.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def transfer(
        self,
        content_image: np.ndarray,
        style_image: np.ndarray,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        num_steps: int = 500,
        learning_rate: float = 0.01
    ) -> np.ndarray:
        """
        Perform neural style transfer
        
        Args:
            content_image: Content image (H, W, C)
            style_image: Style image (H, W, C)
            content_weight: Weight for content loss
            style_weight: Weight for style loss
            num_steps: Number of optimization steps
            learning_rate: Learning rate
            
        Returns:
            Stylized image
        """
        timer = Timer()
        
        # Preprocess images
        content_tensor = self._preprocess_image(content_image).to(self.device)
        style_tensor = self._preprocess_image(style_image).to(self.device)
        
        # Initialize output image as content image
        input_img = content_tensor.clone().requires_grad_(True)
        
        # Extract features
        content_features = self.extract_features(content_tensor)
        style_features = self.extract_features(style_tensor)
        
        # Calculate style Gram matrices
        style_grams = {layer: self.gram_matrix(style_features[layer]) 
                      for layer in self.style_layers}
        
        # Optimizer
        optimizer = torch.optim.Adam([input_img], lr=learning_rate)
        
        logger.info(f"Starting style transfer with {num_steps} steps")
        
        for step in range(num_steps):
            # Zero gradients
            optimizer.zero_grad()
            
            # Extract features from current image
            current_features = self.extract_features(input_img)
            
            # Content loss
            content_loss = F.mse_loss(
                current_features[self.content_layer],
                content_features[self.content_layer]
            )
            
            # Style loss
            style_loss = 0
            for layer, weight in zip(self.style_layers, self.style_weights):
                current_gram = self.gram_matrix(current_features[layer])
                style_gram = style_grams[layer]
                style_loss += weight * F.mse_loss(current_gram, style_gram)
            
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clamp values
            input_img.data.clamp_(0, 1)
            
            if step % 100 == 0:
                logger.debug(f"Step {step}: Loss={total_loss.item():.4f}")
        
        # Convert back to numpy
        output_image = self._deprocess_image(input_img)
        
        logger.info(f"Style transfer completed in {timer.elapsed():.2f}s")
        
        return output_image
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for VGG"""
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize
        image = (image - self.normalization_mean.view(1, 3, 1, 1)) / self.normalization_std.view(1, 3, 1, 1)
        
        return image.float()
    
    def _deprocess_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy image"""
        tensor = tensor.squeeze(0).cpu()
        
        # Denormalize
        tensor = tensor * self.normalization_std.view(3, 1, 1) + self.normalization_mean.view(3, 1, 1)
        
        # Clip and convert
        tensor = torch.clamp(tensor, 0, 1)
        image = tensor.permute(1, 2, 0).numpy()
        
        return (image * 255).astype(np.uint8)


class AdaptiveInstanceNormalization:
    """Adaptive Instance Normalization (AdaIN) for style transfer"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.encoder = self._build_encoder().to(self.device)
        self.decoder = self._build_decoder().to(self.device)
        
        # Load pretrained weights
        self._load_pretrained_weights()
        
        self.encoder.eval()
        self.decoder.eval()
        
        logger.info(f"AdaIN initialized on {device}")
    
    def _build_encoder(self) -> nn.Module:
        """Build encoder network"""
        encoder_layers = [
            nn.Conv2d(3, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]
        
        return nn.Sequential(*encoder_layers)
    
    def _build_decoder(self) -> nn.Module:
        """Build decoder network"""
        decoder_layers = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        ]
        
        return nn.Sequential(*decoder_layers)
    
    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights for AdaIN"""
        # In production, load from checkpoint
        pass
    
    def adain(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """Apply Adaptive Instance Normalization"""
        size = content_feat.size()
        
        # Calculate mean and std
        content_mean = content_feat.view(size[0], size[1], -1).mean(dim=2)
        content_std = content_feat.view(size[0], size[1], -1).std(dim=2) + 1e-8
        
        style_mean = style_feat.view(size[0], size[1], -1).mean(dim=2)
        style_std = style_feat.view(size[0], size[1], -1).std(dim=2) + 1e-8
        
        # Normalize content
        normalized = (content_feat - content_mean.view(size[0], size[1], 1, 1)) / \
                    content_std.view(size[0], size[1], 1, 1)
        
        # Apply style statistics
        stylized = normalized * style_std.view(size[0], size[1], 1, 1) + \
                  style_mean.view(size[0], size[1], 1, 1)
        
        return stylized
    
    def transfer(
        self,
        content_image: np.ndarray,
        style_image: np.ndarray,
        alpha: float = 1.0
    ) -> np.ndarray:
        """
        Transfer style using AdaIN
        
        Args:
            content_image: Content image
            style_image: Style image
            alpha: Style strength (0-1)
            
        Returns:
            Stylized image
        """
        timer = Timer()
        
        # Preprocess images
        content_tensor = self._preprocess_image(content_image).to(self.device)
        style_tensor = self._preprocess_image(style_image).to(self.device)
        
        # Encode images
        with torch.no_grad():
            content_feat = self.encoder(content_tensor)
            style_feat = self.encoder(style_tensor)
            
            # Apply AdaIN
            if alpha < 1.0:
                # Interpolate between content and style features
                target_feat = self.adain(content_feat, style_feat)
                target_feat = alpha * target_feat + (1 - alpha) * content_feat
            else:
                target_feat = self.adain(content_feat, style_feat)
            
            # Decode
            output_tensor = self.decoder(target_feat)
        
        # Convert to numpy
        output_image = self._tensor_to_image(output_tensor)
        
        logger.info(f"AdaIN style transfer completed in {timer.elapsed():.2f}s")
        
        return output_image
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for network"""
        if isinstance(image, np.ndarray):
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image.float()
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image"""
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        image = tensor.permute(1, 2, 0).numpy()
        
        return (image * 255).astype(np.uint8)


class ColorTransfer:
    """Color transfer between images"""
    
    @staticmethod
    def reinhard_transfer(
        source: np.ndarray,
        target: np.ndarray,
        clip: bool = True
    ) -> np.ndarray:
        """
        Color transfer using Reinhard's method
        
        Args:
            source: Source image
            target: Target image
            clip: Whether to clip values to [0, 255]
            
        Returns:
            Color-transferred image
        """
        # Convert to Lab color space
        source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate mean and std
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)
        
        # Normalize
        source_lab -= source_mean
        source_lab /= source_std
        source_lab *= target_std
        source_lab += target_mean
        
        # Clip values
        if clip:
            source_lab = np.clip(source_lab, 0, 255)
        
        # Convert back to RGB
        result = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result
    
    @staticmethod
    def histogram_matching(
        source: np.ndarray,
        target: np.ndarray,
        channels: List[int] = [0, 1, 2]
    ) -> np.ndarray:
        """
        Histogram matching between images
        
        Args:
            source: Source image
            target: Target image
            channels: Channels to match
            
        Returns:
            Histogram-matched image
        """
        result = source.copy()
        
        for channel in channels:
            source_hist, _ = np.histogram(
                source[:, :, channel].flatten(), 256, [0, 256]
            )
            target_hist, _ = np.histogram(
                target[:, :, channel].flatten(), 256, [0, 256]
            )
            
            # Calculate CDF
            source_cdf = source_hist.cumsum()
            source_cdf = source_cdf / source_cdf[-1]
            
            target_cdf = target_hist.cumsum()
            target_cdf = target_cdf / target_cdf[-1]
            
            # Create mapping
            mapping = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                j = 0
                while target_cdf[j] < source_cdf[i] and j < 255:
                    j += 1
                mapping[i] = j
            
            # Apply mapping
            result[:, :, channel] = mapping[source[:, :, channel]]
        
        return result
    
    @staticmethod
    def color_palette_transfer(
        image: np.ndarray,
        palette: List[Tuple[int, int, int]],
        method: str = "kmeans"
    ) -> np.ndarray:
        """
        Transfer color palette to image
        
        Args:
            image: Input image
            palette: Target color palette
            method: Transfer method ('kmeans' or 'simple')
            
        Returns:
            Image with transferred palette
        """
        if method == "simple":
            return ColorTransfer._simple_palette_transfer(image, palette)
        else:
            return ColorTransfer._kmeans_palette_transfer(image, palette)
    
    @staticmethod
    def _simple_palette_transfer(
        image: np.ndarray,
        palette: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Simple palette transfer using nearest colors"""
        h, w = image.shape[:2]
        reshaped = image.reshape(-1, 3)
        
        # Convert palette to numpy
        palette_array = np.array(palette, dtype=np.float32)
        
        # Find nearest palette color for each pixel
        distances = np.linalg.norm(
            reshaped[:, np.newaxis, :] - palette_array[np.newaxis, :, :],
            axis=2
        )
        nearest = np.argmin(distances, axis=1)
        
        # Replace colors
        result = palette_array[nearest].reshape(h, w, 3)
        
        return result.astype(np.uint8)
    
    @staticmethod
    def _kmeans_palette_transfer(
        image: np.ndarray,
        palette: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """Palette transfer using K-means clustering"""
        # Flatten image
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Perform K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, len(palette), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Replace cluster centers with palette colors
        new_centers = np.array(palette, dtype=np.float32)
        
        # Map labels to new colors
        result = new_centers[labels.flatten()].reshape(image.shape)
        
        return result.astype(np.uint8)


class StyleTransferEngine:
    """
    Main style transfer engine supporting multiple methods
    """
    
    def __init__(
        self,
        method: StyleTransferMethod = StyleTransferMethod.NEURAL_STYLE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize style transfer engine
        
        Args:
            method: Style transfer method
            device: Device for computation
        """
        self.method = method
        self.device = device
        
        # Initialize transfer modules
        self.neural_style = None
        self.adain = None
        self.color_transfer = ColorTransfer()
        
        # Load method-specific modules
        if method == StyleTransferMethod.NEURAL_STYLE:
            self.neural_style = NeuralStyleTransfer(device)
        elif method == StyleTransferMethod.ADAIN:
            self.adain = AdaptiveInstanceNormalization(device)
        
        # Style presets
        self.presets = self._load_default_presets()
        
        # Cache for transferred styles
        self.style_cache = {}
        
        logger.info(f"StyleTransferEngine initialized with method={method.value}")
    
    def _load_default_presets(self) -> Dict[str, StylePreset]:
        """Load default style presets"""
        presets = {}
        
        # Realistic preset
        presets["realistic"] = StylePreset(
            name="realistic",
            domain=StyleDomain.REALISTIC,
            parameters={
                "saturation": 1.0,
                "contrast": 1.0,
                "sharpness": 1.0,
                "color_variance": 0.1
            },
            color_palette=[
                (50, 50, 50),   # Dark gray
                (150, 150, 150), # Medium gray
                (255, 255, 255), # White
                (200, 150, 100), # Wood
                (100, 150, 200)  # Sky
            ],
            lighting_conditions={
                "ambient": 0.3,
                "diffuse": 0.7,
                "specular": 0.1,
                "shadows": True
            }
        )
        
        # Painting preset
        presets["painting"] = StylePreset(
            name="painting",
            domain=StyleDomain.PAINTING,
            parameters={
                "brush_stroke": 0.8,
                "color_blending": 0.6,
                "texture_emphasis": 0.7,
                "edge_softness": 0.4
            },
            color_palette=[
                (255, 200, 150), # Warm light
                (100, 150, 200), # Cool shadow
                (255, 255, 200), # Highlight
                (150, 100, 50),  # Earth tone
                (50, 100, 150)   # Deep tone
            ],
            texture_patterns=["brush_strokes", "canvas_weave"],
            material_properties={
                "reflectivity": 0.2,
                "roughness": 0.8,
                "metallic": 0.0
            }
        )
        
        # Cyberpunk preset
        presets["cyberpunk"] = StylePreset(
            name="cyberpunk",
            domain=StyleDomain.CYBERPUNK,
            parameters={
                "neon_intensity": 0.9,
                "glow_effect": 0.8,
                "grain_amount": 0.3,
                "scanlines": 0.5
            },
            color_palette=[
                (255, 0, 255),   # Magenta
                (0, 255, 255),   # Cyan
                (255, 255, 0),   # Yellow
                (0, 0, 0),       # Black
                (255, 255, 255)  # White
            ],
            lighting_conditions={
                "ambient": 0.1,
                "emissive": 0.9,
                "bloom": 0.8,
                "fog": 0.3
            }
        )
        
        # Fantasy preset
        presets["fantasy"] = StylePreset(
            name="fantasy",
            domain=StyleDomain.FANTASY,
            parameters={
                "saturation": 1.2,
                "contrast": 0.8,
                "glow": 0.6,
                "magic_effects": 0.7
            },
            color_palette=[
                (255, 200, 255), # Magical light
                (200, 255, 200), # Nature glow
                (150, 200, 255), # Mystical
                (255, 255, 150), # Golden
                (100, 50, 150)   # Deep purple
            ],
            material_properties={
                "iridescence": 0.7,
                "translucency": 0.4,
                "glow_strength": 0.6
            }
        )
        
        return presets
    
    def transfer_style(
        self,
        content: np.ndarray,
        style_source: Union[np.ndarray, str],
        strength: float = 1.0,
        preserve_content: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Transfer style to content
        
        Args:
            content: Content image or scene data
            style_source: Style image or preset name
            strength: Style strength (0-1)
            preserve_content: Whether to preserve content structure
            **kwargs: Additional parameters
            
        Returns:
            Stylized content
        """
        timer = Timer()
        
        # Check cache
        cache_key = self._generate_cache_key(content, style_source, strength, kwargs)
        if cache_key in self.style_cache:
            logger.debug(f"Using cached style transfer result")
            return self.style_cache[cache_key]
        
        # Handle style preset
        if isinstance(style_source, str) and style_source in self.presets:
            style_image = self._generate_style_from_preset(style_source, content.shape)
        else:
            style_image = style_source
        
        # Apply style transfer based on method
        if self.method == StyleTransferMethod.NEURAL_STYLE and self.neural_style:
            result = self.neural_style.transfer(
                content, style_image,
                style_weight=strength * 1e6,
                **kwargs
            )
        
        elif self.method == StyleTransferMethod.ADAIN and self.adain:
            result = self.adain.transfer(content, style_image, alpha=strength)
        
        elif self.method == StyleTransferMethod.COLOR_TRANSFER:
            result = self.color_transfer.reinhard_transfer(content, style_image)
        
        elif self.method == StyleTransferMethod.PATCH_BASED:
            result = self._patch_based_transfer(content, style_image, strength)
        
        else:
            raise ValueError(f"Unsupported style transfer method: {self.method}")
        
        # Preserve content if requested
        if preserve_content and strength < 1.0:
            result = self._blend_images(content, result, strength)
        
        # Apply post-processing
        result = self._post_process(result, style_source if isinstance(style_source, str) else None)
        
        # Cache result
        self.style_cache[cache_key] = result
        
        logger.info(f"Style transfer completed in {timer.elapsed():.2f}s")
        
        return result
    
    def _generate_style_from_preset(
        self,
        preset_name: str,
        target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Generate style image from preset"""
        preset = self.presets[preset_name]
        
        # Create base noise pattern
        h, w = target_shape[:2]
        style_image = np.random.randn(h, w, 3).astype(np.float32)
        
        # Apply color palette
        if preset.color_palette:
            style_image = self.color_transfer.color_palette_transfer(
                (style_image * 255).astype(np.uint8),
                preset.color_palette,
                method="kmeans"
            ).astype(np.float32) / 255.0
        
        # Apply texture patterns
        for pattern in preset.texture_patterns:
            style_image = self._apply_texture_pattern(style_image, pattern)
        
        # Apply lighting effects
        if preset.lighting_conditions:
            style_image = self._apply_lighting_effects(style_image, preset.lighting_conditions)
        
        return (style_image * 255).astype(np.uint8)
    
    def _apply_texture_pattern(self, image: np.ndarray, pattern: str) -> np.ndarray:
        """Apply texture pattern to image"""
        h, w = image.shape[:2]
        
        if pattern == "brush_strokes":
            # Create brush stroke pattern
            strokes = np.zeros((h, w), dtype=np.float32)
            for _ in range(100):
                x = np.random.randint(0, w)
                y = np.random.randint(0, h)
                length = np.random.randint(20, 100)
                angle = np.random.uniform(0, 2 * np.pi)
                
                for i in range(length):
                    px = int(x + i * np.cos(angle))
                    py = int(y + i * np.sin(angle))
                    
                    if 0 <= px < w and 0 <= py < h:
                        strokes[py, px] = 1.0
            
            # Blend with image
            for c in range(3):
                image[:, :, c] = image[:, :, c] * 0.7 + strokes * 0.3
        
        elif pattern == "canvas_weave":
            # Create canvas weave pattern
            weave = np.zeros((h, w), dtype=np.float32)
            grid_size = 10
            
            for i in range(0, h, grid_size):
                for j in range(0, w, grid_size):
                    weave[i:i+grid_size//2, j:j+grid_size] = 1.0
                    weave[i+grid_size//2:i+grid_size, j:j+grid_size//2] = 1.0
            
            # Blend with image
            for c in range(3):
                image[:, :, c] = image[:, :, c] * 0.8 + weave * 0.2
        
        return image
    
    def _apply_lighting_effects(
        self,
        image: np.ndarray,
        lighting: Dict[str, float]
    ) -> np.ndarray:
        """Apply lighting effects to image"""
        result = image.copy()
        
        # Ambient lighting
        if "ambient" in lighting:
            result = result * lighting["ambient"]
        
        # Add glow effect
        if "glow" in lighting:
            blurred = cv2.GaussianBlur(result, (0, 0), lighting["glow"] * 10)
            result = cv2.addWeighted(result, 0.7, blurred, 0.3, 0)
        
        # Add bloom
        if "bloom" in lighting:
            bright = np.where(result > 0.7, result, 0)
            bloom = cv2.GaussianBlur(bright, (0, 0), lighting["bloom"] * 20)
            result = cv2.addWeighted(result, 1.0, bloom, lighting["bloom"], 0)
        
        return np.clip(result, 0, 1)
    
    def _patch_based_transfer(
        self,
        content: np.ndarray,
        style: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Patch-based style transfer"""
        h, w = content.shape[:2]
        patch_size = 32
        result = np.zeros_like(content)
        
        # Extract patches from style
        style_patches = []
        for i in range(0, style.shape[0] - patch_size, patch_size//2):
            for j in range(0, style.shape[1] - patch_size, patch_size//2):
                patch = style[i:i+patch_size, j:j+patch_size]
                style_patches.append(patch)
        
        if not style_patches:
            return content
        
        # Transfer patches to content
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                content_patch = content[i:i+patch_size, j:j+patch_size]
                
                # Find best matching style patch
                best_patch = self._find_best_patch(content_patch, style_patches)
                
                # Blend with content based on strength
                blended = cv2.addWeighted(
                    content_patch, 1 - strength,
                    best_patch, strength, 0
                )
                
                result[i:i+patch_size, j:j+patch_size] = blended
        
        return result.astype(np.uint8)
    
    def _find_best_patch(
        self,
        content_patch: np.ndarray,
        style_patches: List[np.ndarray]
    ) -> np.ndarray:
        """Find best matching style patch for content patch"""
        best_score = float('inf')
        best_patch = style_patches[0]
        
        content_features = self._extract_patch_features(content_patch)
        
        for style_patch in style_patches:
            style_features = self._extract_patch_features(style_patch)
            score = np.linalg.norm(content_features - style_features)
            
            if score < best_score:
                best_score = score
                best_patch = style_patch
        
        return best_patch
    
    def _extract_patch_features(self, patch: np.ndarray) -> np.ndarray:
        """Extract features from patch"""
        # Simple color histogram features
        hist = []
        for c in range(3):
            hist_channel = cv2.calcHist([patch], [c], None, [16], [0, 256])
            hist.append(hist_channel.flatten())
        
        return np.concatenate(hist)
    
    def _blend_images(
        self,
        content: np.ndarray,
        style: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Blend content and style images"""
        return cv2.addWeighted(content, 1 - strength, style, strength, 0)
    
    def _post_process(
        self,
        image: np.ndarray,
        preset_name: Optional[str] = None
    ) -> np.ndarray:
        """Apply post-processing to stylized image"""
        result = image.copy()
        
        if preset_name and preset_name in self.presets:
            preset = self.presets[preset_name]
            
            # Apply preset-specific post-processing
            params = preset.parameters
            
            if "saturation" in params:
                result = self._adjust_saturation(result, params["saturation"])
            
            if "contrast" in params:
                result = self._adjust_contrast(result, params["contrast"])
            
            if "sharpness" in params:
                result = self._adjust_sharpness(result, params["sharpness"])
        
        return result
    
    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast"""
        mean = np.mean(image, axis=(0, 1))
        result = (image - mean) * factor + mean
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _adjust_sharpness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image sharpness"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * factor
        return cv2.filter2D(image, -1, kernel)
    
    def _generate_cache_key(
        self,
        content: np.ndarray,
        style_source: Union[np.ndarray, str],
        strength: float,
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for style transfer"""
        # Create hash from inputs
        content_hash = hashlib.md5(content.tobytes()).hexdigest()[:8]
        
        if isinstance(style_source, np.ndarray):
            style_hash = hashlib.md5(style_source.tobytes()).hexdigest()[:8]
        else:
            style_hash = style_source
        
        param_str = json.dumps(kwargs, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        key = f"{self.method.value}_{content_hash}_{style_hash}_{strength:.2f}_{param_hash}"
        return key
    
    def apply_to_scene(
        self,
        scene_data: Dict[str, Any],
        style_preset: str,
        components: List[str] = None
    ) -> Dict[str, Any]:
        """
        Apply style to 3D scene
        
        Args:
            scene_data: Scene data dictionary
            style_preset: Style preset name
            components: List of components to style (None for all)
            
        Returns:
            Styled scene data
        """
        if style_preset not in self.presets:
            raise ValueError(f"Unknown style preset: {style_preset}")
        
        preset = self.presets[style_preset]
        styled_scene = scene_data.copy()
        
        # Apply to materials
        if "materials" in styled_scene:
            for mat_name, material in styled_scene["materials"].items():
                if components and mat_name not in components:
                    continue
                
                # Update material properties
                material.update(preset.material_properties)
                
                # Apply color adjustments
                if "color" in material:
                    material["color"] = self._adjust_color_for_style(
                        material["color"], preset
                    )
        
        # Apply to lighting
        if "lighting" in styled_scene:
            styled_scene["lighting"].update(preset.lighting_conditions)
        
        # Apply to post-processing effects
        if "post_processing" not in styled_scene:
            styled_scene["post_processing"] = {}
        
        styled_scene["post_processing"].update(preset.parameters)
        
        # Add style metadata
        styled_scene["metadata"] = styled_scene.get("metadata", {})
        styled_scene["metadata"]["style"] = {
            "preset": preset.name,
            "domain": preset.domain.value,
            "hash": preset.hash
        }
        
        return styled_scene
    
    def _adjust_color_for_style(
        self,
        color: Union[List[float], Tuple[float, ...]],
        preset: StylePreset
    ) -> List[float]:
        """Adjust color according to style preset"""
        if len(color) == 3:  # RGB
            r, g, b = color
            
            # Convert to target palette space
            color_array = np.array([[r, g, b]], dtype=np.float32)
            palette_array = np.array(preset.color_palette, dtype=np.float32) / 255.0
            
            # Find nearest palette color
            distances = np.linalg.norm(color_array - palette_array, axis=1)
            nearest_idx = np.argmin(distances)
            
            # Blend with original color
            target_color = palette_array[nearest_idx]
            blended = color_array * 0.3 + target_color * 0.7
            
            return blended[0].tolist()
        
        return list(color)
    
    def get_available_presets(self) -> List[str]:
        """Get list of available style presets"""
        return list(self.presets.keys())
    
    def get_preset_details(self, preset_name: str) -> Dict[str, Any]:
        """Get details for a style preset"""
        if preset_name not in self.presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        preset = self.presets[preset_name]
        
        return {
            "name": preset.name,
            "domain": preset.domain.value,
            "parameters": preset.parameters,
            "color_palette": preset.color_palette,
            "texture_patterns": preset.texture_patterns,
            "lighting_conditions": preset.lighting_conditions,
            "material_properties": preset.material_properties,
            "hash": preset.hash
        }
    
    def create_custom_preset(
        self,
        name: str,
        domain: str,
        color_palette: List[Tuple[int, int, int]],
        parameters: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Create custom style preset
        
        Args:
            name: Preset name
            domain: Style domain
            color_palette: Color palette
            parameters: Style parameters
            **kwargs: Additional preset properties
            
        Returns:
            Preset hash
        """
        # Validate domain
        try:
            style_domain = StyleDomain(domain)
        except ValueError:
            raise ValueError(f"Invalid domain: {domain}. "
                           f"Valid domains: {[d.value for d in StyleDomain]}")
        
        # Create preset
        preset = StylePreset(
            name=name,
            domain=style_domain,
            parameters=parameters,
            color_palette=color_palette,
            texture_patterns=kwargs.get("texture_patterns", []),
            lighting_conditions=kwargs.get("lighting_conditions", {}),
            material_properties=kwargs.get("material_properties", {})
        )
        
        # Add to presets
        self.presets[name] = preset
        
        # Save to file
        self._save_preset(preset)
        
        logger.info(f"Created custom preset: {name} ({preset.hash})")
        
        return preset.hash
    
    def _save_preset(self, preset: StylePreset) -> None:
        """Save preset to file"""
        preset_dir = Path("data/style_presets")
        preset_dir.mkdir(parents=True, exist_ok=True)
        
        preset_data = {
            "name": preset.name,
            "domain": preset.domain.value,
            "parameters": preset.parameters,
            "color_palette": preset.color_palette,
            "texture_patterns": preset.texture_patterns,
            "lighting_conditions": preset.lighting_conditions,
            "material_properties": preset.material_properties
        }
        
        save_json(preset_data, preset_dir / f"{preset.name}_{preset.hash}.json")
    
    def clear_cache(self) -> None:
        """Clear style transfer cache"""
        self.style_cache.clear()
        logger.info("Style transfer cache cleared")
    
    def __str__(self) -> str:
        """String representation"""
        return f"StyleTransferEngine(method={self.method.value}, presets={len(self.presets)})"


# Factory function for creating style transfer engines
def create_style_transfer_engine(
    method: str = "neural_style",
    device: str = "auto",
    **kwargs
) -> StyleTransferEngine:
    """
    Factory function to create style transfer engines
    
    Args:
        method: Style transfer method
        device: Computation device
        **kwargs: Additional arguments
        
    Returns:
        StyleTransferEngine instance
    """
    method_map = {
        "neural_style": StyleTransferMethod.NEURAL_STYLE,
        "adain": StyleTransferMethod.ADAIN,
        "wct": StyleTransferMethod.WCT,
        "patch_based": StyleTransferMethod.PATCH_BASED,
        "color_transfer": StyleTransferMethod.COLOR_TRANSFER
    }
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    style_method = method_map.get(method, StyleTransferMethod.NEURAL_STYLE)
    
    return StyleTransferEngine(method=style_method, device=device)