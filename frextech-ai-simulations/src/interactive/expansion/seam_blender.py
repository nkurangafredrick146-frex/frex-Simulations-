"""
Seam Blender Module
Blends seams between expanded scenes and existing content
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
from collections import defaultdict, deque
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Image processing and computer vision
import cv2
from PIL import Image, ImageFilter, ImageChops
import scipy.ndimage as ndi
from scipy import signal
from skimage import filters, restoration, segmentation

# Local imports
from ...utils.metrics import Timer, PerformanceMetrics
from ...utils.file_io import save_json, load_json, save_image, load_image
from .boundary_detector import BoundaryDetector, BoundarySegment

logger = logging.getLogger(__name__)


class BlendMethod(Enum):
    """Seam blending methods"""
    LINEAR = "linear"
    GRADIENT = "gradient"
    MULTIBAND = "multiband"
    POISSON = "poisson"
    FEATHER = "feather"
    GRAPH_CUT = "graph_cut"
    DEEP_BLEND = "deep_blend"
    CONTENT_AWARE = "content_aware"


class SeamType(Enum):
    """Types of seams"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIAGONAL = "diagonal"
    IRREGULAR = "irregular"
    OBJECT_BOUNDARY = "object_boundary"
    TEXTURE_BOUNDARY = "texture_boundary"


@dataclass
class SeamRegion:
    """Represents a seam region for blending"""
    region_id: str
    seam_type: SeamType
    mask: np.ndarray  # Binary mask of seam region
    source_region: np.ndarray  # Source image region
    target_region: np.ndarray  # Target image region
    overlap_mask: np.ndarray  # Overlap region mask
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def width(self) -> int:
        """Get width of seam region"""
        return self.mask.shape[1] if self.mask is not None else 0
    
    @property
    def height(self) -> int:
        """Get height of seam region"""
        return self.mask.shape[0] if self.mask is not None else 0
    
    @property
    def area(self) -> int:
        """Get area of seam region"""
        return int(np.sum(self.mask > 0)) if self.mask is not None else 0


@dataclass
class BlendResult:
    """Result of seam blending"""
    blend_id: str
    source_image: np.ndarray
    target_image: np.ndarray
    blended_image: np.ndarray
    blend_mask: np.ndarray
    seam_regions: List[SeamRegion]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def blend_quality(self) -> float:
        """Calculate blend quality score"""
        if self.blended_image is None:
            return 0.0
        
        # Simple quality metric based on gradient smoothness
        try:
            # Calculate gradients
            grad_source = np.gradient(self.source_image.astype(float))
            grad_target = np.gradient(self.target_image.astype(float))
            grad_blended = np.gradient(self.blended_image.astype(float))
            
            # Calculate gradient differences
            diff_source = np.mean(np.abs(grad_blended[0] - grad_source[0]))
            diff_target = np.mean(np.abs(grad_blended[0] - grad_target[0]))
            
            # Quality is inverse of average gradient difference
            max_diff = max(diff_source, diff_target)
            quality = 1.0 - min(max_diff / 255.0, 1.0)
            
            return float(quality)
        except:
            return 0.5  # Default quality


class SeamBlender:
    """
    Main seam blender for seamless integration of expanded content
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        boundary_detector: Optional[BoundaryDetector] = None,
        max_workers: int = 4
    ):
        """
        Initialize seam blender
        
        Args:
            config: Configuration dictionary
            boundary_detector: Optional boundary detector
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.boundary_detector = boundary_detector
        self.max_workers = max_workers
        
        # Blend methods registry
        self.blend_methods = self._initialize_blend_methods()
        
        # Blending history
        self.blend_history: Dict[str, BlendResult] = {}
        self.blend_queue: deque = deque(maxlen=50)
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info("SeamBlender initialized")
    
    def _initialize_blend_methods(self) -> Dict[BlendMethod, Callable]:
        """Initialize blend methods registry"""
        return {
            BlendMethod.LINEAR: self._blend_linear,
            BlendMethod.GRADIENT: self._blend_gradient,
            BlendMethod.MULTIBAND: self._blend_multiband,
            BlendMethod.POISSON: self._blend_poisson,
            BlendMethod.FEATHER: self._blend_feather,
            BlendMethod.GRAPH_CUT: self._blend_graph_cut,
            BlendMethod.DEEP_BLEND: self._blend_deep,
            BlendMethod.CONTENT_AWARE: self._blend_content_aware
        }
    
    def detect_seams(
        self,
        source: np.ndarray,
        target: np.ndarray,
        overlap_region: Optional[Tuple[int, int, int, int]] = None,
        detect_method: str = "gradient"
    ) -> List[SeamRegion]:
        """
        Detect seams between source and target images
        
        Args:
            source: Source image
            target: Target image
            overlap_region: Overlap region (x, y, width, height)
            detect_method: Seam detection method
            
        Returns:
            List of detected seam regions
        """
        timer = Timer()
        
        if source is None or target is None:
            logger.error("Source or target image is None")
            return []
        
        if source.shape != target.shape:
            logger.error(f"Image shape mismatch: source={source.shape}, target={target.shape}")
            return []
        
        logger.info(f"Detecting seams between images of shape {source.shape}")
        
        # Convert to grayscale for detection if needed
        if len(source.shape) == 3:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        else:
            source_gray = source.copy()
            target_gray = target.copy()
        
        # Calculate difference
        diff = cv2.absdiff(source_gray, target_gray)
        
        # Apply threshold to find significant differences
        _, binary_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours of differences
        contours, _ = cv2.findContours(
            binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        seam_regions = []
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 100:  # Skip small contours
                continue
            
            # Create mask from contour
            mask = np.zeros_like(source_gray, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Determine seam type
            seam_type = self._classify_seam_type(contour, diff)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract regions with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(source.shape[1], x + w + padding)
            y2 = min(source.shape[0], y + h + padding)
            
            source_region = source[y1:y2, x1:x2]
            target_region = target[y1:y2, x1:x2]
            region_mask = mask[y1:y2, x1:x2]
            
            # Create overlap mask (area where both images have content)
            overlap_mask = self._create_overlap_mask(source_region, target_region)
            
            # Create seam region
            region_id = f"seam_{i}_{int(time.time()*1000)}"
            
            seam_region = SeamRegion(
                region_id=region_id,
                seam_type=seam_type,
                mask=region_mask,
                source_region=source_region,
                target_region=target_region,
                overlap_mask=overlap_mask,
                metadata={
                    "contour_area": float(cv2.contourArea(contour)),
                    "bounding_box": (x, y, w, h),
                    "detection_method": detect_method
                }
            )
            
            seam_regions.append(seam_region)
        
        # Update metrics
        self.metrics.record_operation("detect_seams", timer.elapsed())
        
        logger.info(f"Detected {len(seam_regions)} seam regions in {timer.elapsed():.2f}s")
        
        return seam_regions
    
    def _classify_seam_type(
        self,
        contour: np.ndarray,
        diff_image: np.ndarray
    ) -> SeamType:
        """Classify seam type based on contour shape"""
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Calculate orientation
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]  # Rotation angle
        else:
            angle = 0
        
        # Classify based on shape and orientation
        if aspect_ratio > 3.0:
            return SeamType.HORIZONTAL
        elif aspect_ratio < 0.33:
            return SeamType.VERTICAL
        elif 45 < angle < 135:
            return SeamType.DIAGONAL
        else:
            # Analyze texture around seam
            mask = np.zeros_like(diff_image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Check if seam is along object boundary
            if self._is_object_boundary(diff_image, mask):
                return SeamType.OBJECT_BOUNDARY
            elif self._is_texture_boundary(diff_image, mask):
                return SeamType.TEXTURE_BOUNDARY
            else:
                return SeamType.IRREGULAR
    
    def _is_object_boundary(
        self,
        diff_image: np.ndarray,
        mask: np.ndarray
    ) -> bool:
        """Check if seam is along object boundary"""
        # Object boundaries typically have strong, continuous edges
        edges = cv2.Canny(diff_image, 50, 150)
        
        # Calculate edge density in mask area
        edge_pixels = np.sum(edges[mask > 0] > 0)
        total_pixels = np.sum(mask > 0)
        
        if total_pixels > 0:
            edge_density = edge_pixels / total_pixels
            return edge_density > 0.3
        
        return False
    
    def _is_texture_boundary(
        self,
        diff_image: np.ndarray,
        mask: np.ndarray
    ) -> bool:
        """Check if seam is along texture boundary"""
        # Texture boundaries have repetitive patterns
        # Calculate local variance
        variance = ndi.generic_filter(
            diff_image.astype(float), 
            np.var, 
            size=3
        )
        
        # Calculate average variance in mask area
        mask_area = mask > 0
        if np.any(mask_area):
            avg_variance = np.mean(variance[mask_area])
            return avg_variance < 100.0  # Low variance indicates texture
        
        return False
    
    def _create_overlap_mask(
        self,
        source_region: np.ndarray,
        target_region: np.ndarray
    ) -> np.ndarray:
        """Create mask of overlapping valid regions"""
        if source_region.shape != target_region.shape:
            logger.error("Region shape mismatch")
            return np.ones_like(source_region[:, :, 0], dtype=np.uint8) * 255
        
        # Check for valid pixels (non-zero)
        if len(source_region.shape) == 3:
            source_valid = np.all(source_region > 0, axis=2)
            target_valid = np.all(target_region > 0, axis=2)
        else:
            source_valid = source_region > 0
            target_valid = target_region > 0
        
        # Overlap is where both have valid pixels
        overlap = np.logical_and(source_valid, target_valid)
        
        return overlap.astype(np.uint8) * 255
    
    def blend_seams(
        self,
        source: np.ndarray,
        target: np.ndarray,
        method: Union[str, BlendMethod] = BlendMethod.MULTIBAND,
        seam_regions: Optional[List[SeamRegion]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[BlendResult]:
        """
        Blend seams between source and target images
        
        Args:
            source: Source image
            target: Target image
            method: Blending method
            seam_regions: Pre-detected seam regions
            parameters: Blending parameters
            
        Returns:
            Blend result or None if failed
        """
        timer = Timer()
        
        if source is None or target is None:
            logger.error("Source or target image is None")
            return None
        
        if source.shape != target.shape:
            logger.error(f"Image shape mismatch: source={source.shape}, target={target.shape}")
            return None
        
        try:
            # Parse method
            if isinstance(method, str):
                try:
                    blend_method = BlendMethod(method)
                except ValueError:
                    logger.error(f"Unknown blend method: {method}")
                    blend_method = BlendMethod.MULTIBAND
            else:
                blend_method = method
            
            # Merge parameters
            merged_params = self._get_default_parameters(blend_method)
            if parameters:
                merged_params.update(parameters)
            
            # Detect seams if not provided
            if seam_regions is None:
                seam_regions = self.detect_seams(
                    source, target,
                    detect_method=merged_params.get("detect_method", "gradient")
                )
            
            blend_id = self._generate_blend_id(source, target, blend_method)
            
            logger.info(f"Blending seams {blend_id} using {blend_method.value} "
                       f"with {len(seam_regions)} seam regions")
            
            # Get blend function
            blend_func = self.blend_methods.get(blend_method)
            if blend_func is None:
                logger.error(f"Blend method not implemented: {blend_method}")
                return None
            
            # Apply blending
            blended_image, blend_mask = blend_func(
                source, target, seam_regions, merged_params
            )
            
            if blended_image is None:
                logger.error("Blending function failed")
                return None
            
            # Create blend result
            result = BlendResult(
                blend_id=blend_id,
                source_image=source.copy(),
                target_image=target.copy(),
                blended_image=blended_image,
                blend_mask=blend_mask,
                seam_regions=seam_regions,
                metadata={
                    "method": blend_method.value,
                    "parameters": merged_params,
                    "num_seams": len(seam_regions),
                    "processing_time": timer.elapsed(),
                    "timestamp": time.time()
                }
            )
            
            # Store in history
            self.blend_history[blend_id] = result
            self.blend_queue.append(blend_id)
            
            # Update metrics
            self.metrics.record_operation(f"blend_seams_{blend_method.value}", timer.elapsed())
            
            logger.info(f"Seam blending completed in {timer.elapsed():.2f}s: "
                       f"quality={result.blend_quality:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error blending seams: {e}")
            return None
    
    def _get_default_parameters(
        self,
        method: BlendMethod
    ) -> Dict[str, Any]:
        """Get default parameters for blend method"""
        defaults = {
            BlendMethod.LINEAR: {
                "blend_width": 20,
                "feather_edges": True
            },
            BlendMethod.GRADIENT: {
                "blend_width": 30,
                "smoothness": 0.5,
                "iterations": 3
            },
            BlendMethod.MULTIBAND: {
                "num_bands": 5,
                "band_scale": 2.0,
                "blend_width": 40
            },
            BlendMethod.POISSON: {
                "iterations": 1000,
                "tolerance": 1e-6,
                "method": "normal"
            },
            BlendMethod.FEATHER: {
                "feather_amount": 0.5,
                "smoothness": 2.0
            },
            BlendMethod.GRAPH_CUT: {
                "cost_function": "gradient",
                "smoothness_weight": 1.0,
                "data_weight": 1.0
            },
            BlendMethod.DEEP_BLEND: {
                "model_path": None,
                "confidence_threshold": 0.7
            },
            BlendMethod.CONTENT_AWARE: {
                "patch_size": 15,
                "search_window": 30,
                "iterations": 5
            }
        }
        
        return defaults.get(method, {}).copy()
    
    def _generate_blend_id(
        self,
        source: np.ndarray,
        target: np.ndarray,
        method: BlendMethod
    ) -> str:
        """Generate unique blend ID"""
        timestamp = int(time.time() * 1000)
        
        # Create hash from image data and method
        source_hash = hashlib.md5(source.tobytes()).hexdigest()[:8]
        target_hash = hashlib.md5(target.tobytes()).hexdigest()[:8]
        
        content = f"{source_hash}_{target_hash}_{method.value}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    # Blend method implementations
    def _blend_linear(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Linear blending (alpha blending)"""
        blend_width = parameters.get("blend_width", 20)
        feather_edges = parameters.get("feather_edges", True)
        
        # Create blend mask
        blend_mask = np.zeros(source.shape[:2], dtype=np.float32)
        
        # For each seam region, create linear blend
        for region in seam_regions:
            if region.mask is not None:
                # Create linear gradient across seam
                region_mask = region.mask.astype(np.float32) / 255.0
                
                # Apply distance transform for smooth gradient
                if feather_edges:
                    dist = cv2.distanceTransform(
                        (region_mask > 0).astype(np.uint8),
                        cv2.DIST_L2, 3
                    )
                    region_gradient = dist / (blend_width / 2)
                    region_gradient = np.clip(region_gradient, 0, 1)
                else:
                    # Binary mask
                    region_gradient = region_mask
                
                # Update blend mask
                y, x = np.where(region.mask > 0)
                if len(y) > 0 and len(x) > 0:
                    y_start = np.min(y)
                    x_start = np.min(x)
                    region_height = np.max(y) - y_start + 1
                    region_width = np.max(x) - x_start + 1
                    
                    if region_height > 0 and region_width > 0:
                        # Ensure we don't go out of bounds
                        y_end = min(y_start + region_height, blend_mask.shape[0])
                        x_end = min(x_start + region_width, blend_mask.shape[1])
                        
                        region_slice = (
                            slice(y_start, y_end),
                            slice(x_start, x_end)
                        )
                        
                        blend_mask[region_slice] = np.maximum(
                            blend_mask[region_slice],
                            region_gradient[:y_end-y_start, :x_end-x_start]
                        )
        
        # If no seams detected, create default blend mask
        if np.sum(blend_mask) == 0:
            # Create horizontal blend gradient
            h, w = source.shape[:2]
            blend_mask = np.zeros((h, w), dtype=np.float32)
            
            # Simple left-to-right blend
            for i in range(w):
                blend_mask[:, i] = min(1.0, i / blend_width)
        
        # Ensure mask is in [0, 1] range
        blend_mask = np.clip(blend_mask, 0, 1)
        
        # Apply linear blending
        if len(source.shape) == 3:
            # RGB image
            blended = np.zeros_like(source, dtype=np.float32)
            for c in range(3):
                blended[:, :, c] = (
                    source[:, :, c].astype(np.float32) * (1 - blend_mask) +
                    target[:, :, c].astype(np.float32) * blend_mask
                )
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            # Grayscale image
            blended = (
                source.astype(np.float32) * (1 - blend_mask) +
                target.astype(np.float32) * blend_mask
            )
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended, (blend_mask * 255).astype(np.uint8)
    
    def _blend_gradient(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Gradient domain blending"""
        blend_width = parameters.get("blend_width", 30)
        smoothness = parameters.get("smoothness", 0.5)
        iterations = parameters.get("iterations", 3)
        
        # Calculate gradients
        if len(source.shape) == 3:
            # Process each channel separately
            blended_channels = []
            gradient_masks = []
            
            for c in range(3):
                source_channel = source[:, :, c].astype(np.float32)
                target_channel = target[:, :, c].astype(np.float32)
                
                # Calculate gradients
                grad_source_x = np.gradient(source_channel, axis=1)
                grad_source_y = np.gradient(source_channel, axis=0)
                grad_target_x = np.gradient(target_channel, axis=1)
                grad_target_y = np.gradient(target_channel, axis=0)
                
                # Create blend mask for this channel
                blend_mask = np.zeros_like(source_channel, dtype=np.float32)
                
                for region in seam_regions:
                    if region.mask is not None:
                        region_mask = region.mask.astype(np.float32) / 255.0
                        
                        # Apply Gaussian smoothing to mask
                        sigma = blend_width * smoothness
                        if sigma > 0:
                            region_mask = cv2.GaussianBlur(
                                region_mask, (0, 0), sigma
                            )
                        
                        # Update blend mask
                        y, x = np.where(region.mask > 0)
                        if len(y) > 0 and len(x) > 0:
                            y_start = np.min(y)
                            x_start = np.min(x)
                            region_height = np.max(y) - y_start + 1
                            region_width = np.max(x) - x_start + 1
                            
                            if region_height > 0 and region_width > 0:
                                y_end = min(y_start + region_height, blend_mask.shape[0])
                                x_end = min(x_start + region_width, blend_mask.shape[1])
                                
                                region_slice = (
                                    slice(y_start, y_end),
                                    slice(x_start, x_end)
                                )
                                
                                blend_mask[region_slice] = np.maximum(
                                    blend_mask[region_slice],
                                    region_mask[:y_end-y_start, :x_end-x_start]
                                )
                
                # If no seams detected, create default mask
                if np.sum(blend_mask) == 0:
                    h, w = source_channel.shape
                    blend_mask = np.zeros((h, w), dtype=np.float32)
                    for i in range(w):
                        blend_mask[:, i] = min(1.0, i / blend_width)
                
                blend_mask = np.clip(blend_mask, 0, 1)
                gradient_masks.append(blend_mask)
                
                # Blend gradients
                grad_blended_x = (
                    grad_source_x * (1 - blend_mask) +
                    grad_target_x * blend_mask
                )
                grad_blended_y = (
                    grad_source_y * (1 - blend_mask) +
                    grad_target_y * blend_mask
                )
                
                # Reconstruct image from blended gradients
                blended_channel = self._reconstruct_from_gradient(
                    grad_blended_x, grad_blended_y,
                    source_channel, target_channel,
                    blend_mask, iterations
                )
                
                blended_channels.append(blended_channel)
            
            # Combine channels
            blended = np.stack(blended_channels, axis=2)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Create combined blend mask
            blend_mask = np.mean(np.array(gradient_masks), axis=0)
        
        else:
            # Grayscale image
            source_f = source.astype(np.float32)
            target_f = target.astype(np.float32)
            
            # Calculate gradients
            grad_source_x = np.gradient(source_f, axis=1)
            grad_source_y = np.gradient(source_f, axis=0)
            grad_target_x = np.gradient(target_f, axis=1)
            grad_target_y = np.gradient(target_f, axis=0)
            
            # Create blend mask
            blend_mask = np.zeros_like(source_f, dtype=np.float32)
            
            for region in seam_regions:
                if region.mask is not None:
                    region_mask = region.mask.astype(np.float32) / 255.0
                    
                    # Apply smoothing
                    sigma = blend_width * smoothness
                    if sigma > 0:
                        region_mask = cv2.GaussianBlur(region_mask, (0, 0), sigma)
                    
                    # Update blend mask
                    y, x = np.where(region.mask > 0)
                    if len(y) > 0 and len(x) > 0:
                        y_start = np.min(y)
                        x_start = np.min(x)
                        y_end = min(np.max(y) + 1, blend_mask.shape[0])
                        x_end = min(np.max(x) + 1, blend_mask.shape[1])
                        
                        blend_mask[y_start:y_end, x_start:x_end] = np.maximum(
                            blend_mask[y_start:y_end, x_start:x_end],
                            region_mask[:y_end-y_start, :x_end-x_start]
                        )
            
            # If no seams detected
            if np.sum(blend_mask) == 0:
                h, w = source_f.shape
                blend_mask = np.zeros((h, w), dtype=np.float32)
                for i in range(w):
                    blend_mask[:, i] = min(1.0, i / blend_width)
            
            blend_mask = np.clip(blend_mask, 0, 1)
            
            # Blend gradients
            grad_blended_x = (
                grad_source_x * (1 - blend_mask) +
                grad_target_x * blend_mask
            )
            grad_blended_y = (
                grad_source_y * (1 - blend_mask) +
                grad_target_y * blend_mask
            )
            
            # Reconstruct
            blended = self._reconstruct_from_gradient(
                grad_blended_x, grad_blended_y,
                source_f, target_f,
                blend_mask, iterations
            )
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended, (blend_mask * 255).astype(np.uint8)
    
    def _reconstruct_from_gradient(
        self,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        source: np.ndarray,
        target: np.ndarray,
        blend_mask: np.ndarray,
        iterations: int
    ) -> np.ndarray:
        """Reconstruct image from gradient field using Poisson equation"""
        h, w = grad_x.shape
        
        # Initialize with weighted average
        reconstructed = (
            source * (1 - blend_mask) +
            target * blend_mask
        )
        
        # Simple iterative solver (Gauss-Seidel)
        for _ in range(iterations):
            # Update interior pixels
            for i in range(1, h-1):
                for j in range(1, w-1):
                    # Use discrete Laplacian
                    laplacian = (
                        reconstructed[i-1, j] +
                        reconstructed[i+1, j] +
                        reconstructed[i, j-1] +
                        reconstructed[i, j+1]
                    ) / 4.0
                    
                    # Add gradient constraint
                    grad_constraint = (
                        (grad_x[i, j] - grad_x[i, j-1]) +
                        (grad_y[i, j] - grad_y[i-1, j])
                    ) / 2.0
                    
                    reconstructed[i, j] = laplacian + grad_constraint
        
        return reconstructed
    
    def _blend_multiband(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Multiband blending (Laplacian pyramid blending)"""
        num_bands = parameters.get("num_bands", 5)
        band_scale = parameters.get("band_scale", 2.0)
        blend_width = parameters.get("blend_width", 40)
        
        if source.shape != target.shape:
            logger.error("Image shapes must match for multiband blending")
            return None, None
        
        # Create blend mask
        blend_mask = self._create_blend_mask(source, seam_regions, blend_width)
        
        # Build Gaussian pyramids
        gaussian_source = [source.astype(np.float32)]
        gaussian_target = [target.astype(np.float32)]
        gaussian_mask = [blend_mask.astype(np.float32)]
        
        for i in range(1, num_bands):
            # Downsample
            source_down = cv2.pyrDown(gaussian_source[-1])
            target_down = cv2.pyrDown(gaussian_target[-1])
            mask_down = cv2.pyrDown(gaussian_mask[-1])
            
            gaussian_source.append(source_down)
            gaussian_target.append(target_down)
            gaussian_mask.append(mask_down)
        
        # Build Laplacian pyramids
        laplacian_source = [gaussian_source[-1]]
        laplacian_target = [gaussian_target[-1]]
        
        for i in range(num_bands - 2, -1, -1):
            # Upsample and subtract
            source_up = cv2.pyrUp(gaussian_source[i + 1])
            target_up = cv2.pyrUp(gaussian_target[i + 1])
            
            # Ensure sizes match
            h, w = gaussian_source[i].shape[:2]
            if source_up.shape[:2] != (h, w):
                source_up = cv2.resize(source_up, (w, h))
                target_up = cv2.resize(target_up, (w, h))
            
            laplacian_s = gaussian_source[i] - source_up
            laplacian_t = gaussian_target[i] - target_up
            
            laplacian_source.append(laplacian_s)
            laplacian_target.append(laplacian_t)
        
        # Reverse to have coarse to fine
        laplacian_source.reverse()
        laplacian_target.reverse()
        
        # Blend each level
        blended_pyramid = []
        
        for i in range(num_bands):
            # Get mask for this level
            if i < len(gaussian_mask):
                mask_level = gaussian_mask[i]
            else:
                # Use smallest mask
                mask_level = gaussian_mask[-1]
                # Resize if needed
                h, w = laplacian_source[i].shape[:2]
                mask_level = cv2.resize(mask_level, (w, h))
            
            # Ensure mask is 2D for single channel blending
            if len(mask_level.shape) == 2 and len(laplacian_source[i].shape) == 3:
                # Expand mask for RGB
                mask_level = np.stack([mask_level] * 3, axis=2)
            
            # Blend Laplacian coefficients
            blended_level = (
                laplacian_source[i] * (1 - mask_level) +
                laplacian_target[i] * mask_level
            )
            blended_pyramid.append(blended_level)
        
        # Reconstruct from pyramid
        blended = blended_pyramid[0]
        
        for i in range(1, num_bands):
            # Upsample
            upsampled = cv2.pyrUp(blended)
            
            # Ensure sizes match
            h, w = blended_pyramid[i].shape[:2]
            if upsampled.shape[:2] != (h, w):
                upsampled = cv2.resize(upsampled, (w, h))
            
            # Add Laplacian
            blended = upsampled + blended_pyramid[i]
        
        # Clip and convert to uint8
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended, (blend_mask * 255).astype(np.uint8)
    
    def _blend_poisson(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Poisson blending"""
        iterations = parameters.get("iterations", 1000)
        tolerance = parameters.get("tolerance", 1e-6)
        method = parameters.get("method", "normal")
        
        # Create blend mask
        blend_mask = self._create_blend_mask(source, seam_regions, 20)
        
        # Convert to float
        source_f = source.astype(np.float32)
        target_f = target.astype(np.float32)
        
        # For each channel
        if len(source.shape) == 3:
            blended_channels = []
            
            for c in range(3):
                source_channel = source_f[:, :, c]
                target_channel = target_f[:, :, c]
                
                # Create guidance field (gradient of source)
                grad_x = cv2.Sobel(source_channel, cv2.CV_32F, 1, 0)
                grad_y = cv2.Sobel(source_channel, cv2.CV_32F, 0, 1)
                
                # Solve Poisson equation
                blended_channel = self._solve_poisson(
                    target_channel, grad_x, grad_y,
                    blend_mask, iterations, tolerance, method
                )
                
                blended_channels.append(blended_channel)
            
            blended = np.stack(blended_channels, axis=2)
        
        else:
            # Grayscale
            grad_x = cv2.Sobel(source_f, cv2.CV_32F, 1, 0)
            grad_y = cv2.Sobel(source_f, cv2.CV_32F, 0, 1)
            
            blended = self._solve_poisson(
                target_f, grad_x, grad_y,
                blend_mask, iterations, tolerance, method
            )
        
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended, (blend_mask * 255).astype(np.uint8)
    
    def _solve_poisson(
        self,
        target: np.ndarray,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        mask: np.ndarray,
        iterations: int,
        tolerance: float,
        method: str
    ) -> np.ndarray:
        """Solve Poisson equation using Jacobi iteration"""
        h, w = target.shape
        
        # Initialize solution with target
        solution = target.copy()
        
        # Create mask indicating pixels to update
        update_mask = mask > 0.5
        
        # Jacobi iteration
        for it in range(iterations):
            solution_old = solution.copy()
            max_change = 0.0
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if update_mask[i, j]:
                        # Discrete Laplacian
                        laplacian = (
                            solution[i-1, j] +
                            solution[i+1, j] +
                            solution[i, j-1] +
                            solution[i, j+1]
                        ) / 4.0
                        
                        # Add gradient guidance
                        grad_term = (
                            (grad_x[i, j] - grad_x[i, j-1]) +
                            (grad_y[i, j] - grad_y[i-1, j])
                        ) / 2.0
                        
                        solution[i, j] = laplacian + grad_term
                        
                        # Track convergence
                        change = abs(solution[i, j] - solution_old[i, j])
                        max_change = max(max_change, change)
            
            # Check convergence
            if max_change < tolerance:
                logger.debug(f"Poisson solver converged after {it} iterations")
                break
        
        return solution
    
    def _blend_feather(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Feather blending"""
        feather_amount = parameters.get("feather_amount", 0.5)
        smoothness = parameters.get("smoothness", 2.0)
        
        # Create blend mask with feathered edges
        blend_mask = np.zeros(source.shape[:2], dtype=np.float32)
        
        for region in seam_regions:
            if region.mask is not None:
                region_mask = region.mask.astype(np.float32) / 255.0
                
                # Apply distance transform
                dist = cv2.distanceTransform(
                    (region_mask > 0).astype(np.uint8),
                    cv2.DIST_L2, 3
                )
                
                # Create feathered gradient
                max_dist = np.max(dist)
                if max_dist > 0:
                    gradient = dist / max_dist
                    
                    # Apply feathering function
                    if feather_amount > 0:
                        gradient = gradient ** (1.0 / feather_amount)
                    
                    # Apply smoothing
                    if smoothness > 0:
                        gradient = cv2.GaussianBlur(
                            gradient, (0, 0), smoothness
                        )
                    
                    # Update blend mask
                    y, x = np.where(region.mask > 0)
                    if len(y) > 0 and len(x) > 0:
                        y_start = np.min(y)
                        x_start = np.min(x)
                        y_end = min(np.max(y) + 1, blend_mask.shape[0])
                        x_end = min(np.max(x) + 1, blend_mask.shape[1])
                        
                        blend_mask[y_start:y_end, x_start:x_end] = np.maximum(
                            blend_mask[y_start:y_end, x_start:x_end],
                            gradient[:y_end-y_start, :x_end-x_start]
                        )
        
        # If no seams detected
        if np.sum(blend_mask) == 0:
            h, w = source.shape[:2]
            blend_mask = np.zeros((h, w), dtype=np.float32)
            
            # Create simple gradient
            for i in range(w):
                blend_mask[:, i] = min(1.0, i / w)
            
            # Apply feathering
            if feather_amount > 0:
                blend_mask = blend_mask ** (1.0 / feather_amount)
        
        blend_mask = np.clip(blend_mask, 0, 1)
        
        # Apply blending
        if len(source.shape) == 3:
            blended = np.zeros_like(source, dtype=np.float32)
            for c in range(3):
                blended[:, :, c] = (
                    source[:, :, c].astype(np.float32) * (1 - blend_mask) +
                    target[:, :, c].astype(np.float32) * blend_mask
                )
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            blended = (
                source.astype(np.float32) * (1 - blend_mask) +
                target.astype(np.float32) * blend_mask
            )
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended, (blend_mask * 255).astype(np.uint8)
    
    def _blend_graph_cut(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Graph cut based blending"""
        cost_function = parameters.get("cost_function", "gradient")
        smoothness_weight = parameters.get("smoothness_weight", 1.0)
        data_weight = parameters.get("data_weight", 1.0)
        
        # This is a simplified implementation
        # In production, would use actual graph cut algorithm
        
        # Create initial mask using simple difference
        diff = cv2.absdiff(source, target)
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)
        
        # Threshold difference
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Refine mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Create smooth blend mask from binary mask
        blend_mask = mask.astype(np.float32) / 255.0
        
        # Apply Gaussian smoothing
        blend_mask = cv2.GaussianBlur(blend_mask, (0, 0), 10)
        
        # Apply blending
        if len(source.shape) == 3:
            blended = np.zeros_like(source, dtype=np.float32)
            for c in range(3):
                blended[:, :, c] = (
                    source[:, :, c].astype(np.float32) * (1 - blend_mask) +
                    target[:, :, c].astype(np.float32) * blend_mask
                )
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            blended = (
                source.astype(np.float32) * (1 - blend_mask) +
                target.astype(np.float32) * blend_mask
            )
            blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        return blended, (blend_mask * 255).astype(np.uint8)
    
    def _blend_deep(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Deep learning based blending (placeholder)"""
        # This is a placeholder for deep learning based blending
        # In production, would load and run a neural network
        
        logger.info("Using deep blend (placeholder - falling back to multiband)")
        
        # Fall back to multiband blending
        return self._blend_multiband(
            source, target, seam_regions,
            {"num_bands": 5, "blend_width": 30}
        )
    
    def _blend_content_aware(
        self,
        source: np.ndarray,
        target: np.ndarray,
        seam_regions: List[SeamRegion],
        parameters: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Content-aware fill blending"""
        patch_size = parameters.get("patch_size", 15)
        search_window = parameters.get("search_window", 30)
        iterations = parameters.get("iterations", 5)
        
        # Create initial blend using linear blending
        blended, blend_mask = self._blend_linear(
            source, target, seam_regions,
            {"blend_width": 20, "feather_edges": True}
        )
        
        if blended is None:
            return None, None
        
        # Refine using patch-based synthesis (simplified)
        h, w = source.shape[:2]
        
        for _ in range(iterations):
            # Find high-error regions
            error = cv2.absdiff(blended, source) + cv2.absdiff(blended, target)
            if len(error.shape) == 3:
                error = np.mean(error, axis=2)
            
            # Threshold to find problematic areas
            _, error_mask = cv2.threshold(error, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours of high-error regions
            contours, _ = cv2.findContours(
                error_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Get bounding box
                x, y, w_contour, h_contour = cv2.boundingRect(contour)
                
                # Extract patch
                patch_x = max(0, x - patch_size)
                patch_y = max(0, y - patch_size)
                patch_w = min(w, x + w_contour + patch_size) - patch_x
                patch_h = min(h, y + h_contour + patch_size) - patch_y
                
                if patch_w <= 0 or patch_h <= 0:
                    continue
                
                # Extract source and target patches
                source_patch = source[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
                target_patch = target[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w]
                
                # Find best matching patch from surroundings
                best_patch = self._find_best_patch(
                    source_patch, target_patch,
                    patch_x, patch_y, patch_w, patch_h,
                    source, target, search_window
                )
                
                if best_patch is not None:
                    # Blend patch into result
                    patch_mask = np.ones((patch_h, patch_w), dtype=np.float32) * 0.5
                    blended_patch = (
                        best_patch.astype(np.float32) * patch_mask +
                        blended[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w].astype(np.float32) * (1 - patch_mask)
                    )
                    blended[patch_y:patch_y+patch_h, patch_x:patch_x+patch_w] = blended_patch.astype(np.uint8)
        
        return blended, blend_mask
    
    def _find_best_patch(
        self,
        source_patch: np.ndarray,
        target_patch: np.ndarray,
        patch_x: int,
        patch_y: int,
        patch_w: int,
        patch_h: int,
        source: np.ndarray,
        target: np.ndarray,
        search_window: int
    ) -> Optional[np.ndarray]:
        """Find best matching patch from surrounding area"""
        h, w = source.shape[:2]
        
        best_score = float('inf')
        best_patch = None
        
        # Search in surrounding area
        search_min_x = max(0, patch_x - search_window)
        search_max_x = min(w - patch_w, patch_x + search_window)
        search_min_y = max(0, patch_y - search_window)
        search_max_y = min(h - patch_h, patch_y + search_window)
        
        for y in range(search_min_y, search_max_y, patch_h // 2):
            for x in range(search_min_x, search_max_x, patch_w // 2):
                if x == patch_x and y == patch_y:
                    continue
                
                # Extract candidate patches
                candidate_source = source[y:y+patch_h, x:x+patch_w]
                candidate_target = target[y:y+patch_h, x:x+patch_w]
                
                if candidate_source.shape != source_patch.shape:
                    continue
                
                # Calculate match score
                source_diff = np.mean(np.abs(candidate_source - source_patch))
                target_diff = np.mean(np.abs(candidate_target - target_patch))
                score = (source_diff + target_diff) / 2.0
                
                if score < best_score:
                    best_score = score
                    # Blend candidate patches
                    if len(source_patch.shape) == 3:
                        best_patch = (
                            candidate_source.astype(np.float32) * 0.5 +
                            candidate_target.astype(np.float32) * 0.5
                        ).astype(np.uint8)
                    else:
                        best_patch = (
                            candidate_source.astype(np.float32) * 0.5 +
                            candidate_target.astype(np.float32) * 0.5
                        ).astype(np.uint8)
        
        return best_patch
    
    def _create_blend_mask(
        self,
        image: np.ndarray,
        seam_regions: List[SeamRegion],
        blend_width: int
    ) -> np.ndarray:
        """Create blend mask from seam regions"""
        h, w = image.shape[:2]
        blend_mask = np.zeros((h, w), dtype=np.float32)
        
        for region in seam_regions:
            if region.mask is not None:
                region_mask = region.mask.astype(np.float32) / 255.0
                
                # Apply distance transform for smooth gradient
                dist = cv2.distanceTransform(
                    (region_mask > 0).astype(np.uint8),
                    cv2.DIST_L2, 3
                )
                
                # Normalize and create gradient
                max_dist = np.max(dist)
                if max_dist > 0:
                    gradient = dist / max_dist
                    
                    # Update blend mask
                    y, x = np.where(region.mask > 0)
                    if len(y) > 0 and len(x) > 0:
                        y_start = np.min(y)
                        x_start = np.min(x)
                        y_end = min(np.max(y) + 1, h)
                        x_end = min(np.max(x) + 1, w)
                        
                        blend_mask[y_start:y_end, x_start:x_end] = np.maximum(
                            blend_mask[y_start:y_end, x_start:x_end],
                            gradient[:y_end-y_start, :x_end-x_start]
                        )
        
        # If no seams detected, create default mask
        if np.sum(blend_mask) == 0:
            for i in range(w):
                blend_mask[:, i] = min(1.0, i / blend_width)
        
        return np.clip(blend_mask, 0, 1)
    
    def blend_scenes_3d(
        self,
        scene1: Dict[str, Any],
        scene2: Dict[str, Any],
        method: Union[str, BlendMethod] = BlendMethod.MULTIBAND,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Blend 3D scenes at their boundaries
        
        Args:
            scene1: First scene
            scene2: Second scene
            method: Blending method
            parameters: Blending parameters
            
        Returns:
            Blended scene or None if failed
        """
        # This is a simplified implementation
        # In production, would handle 3D geometry blending
        
        logger.info("Blending 3D scenes (simplified implementation)")
        
        # Merge objects from both scenes
        blended_scene = scene1.copy()
        
        if "objects" not in blended_scene:
            blended_scene["objects"] = {}
        
        # Add objects from scene2 with modified positions
        if "objects" in scene2:
            for obj_id, obj_data in scene2["objects"].items():
                # Create new ID to avoid conflicts
                new_id = f"{obj_id}_blended"
                
                # Adjust position to avoid overlaps
                if "position" in obj_data:
                    pos = obj_data["position"].copy()
                    pos[0] += 5.0  # Simple offset
                    obj_data["position"] = pos
                
                blended_scene["objects"][new_id] = obj_data
        
        # Update metadata
        if "metadata" not in blended_scene:
            blended_scene["metadata"] = {}
        
        blended_scene["metadata"]["blended"] = {
            "method": method.value if isinstance(method, BlendMethod) else method,
            "source_scenes": [scene1.get("metadata", {}).get("name", "scene1"),
                            scene2.get("metadata", {}).get("name", "scene2")],
            "timestamp": time.time()
        }
        
        return blended_scene
    
    def get_blend_history(self) -> List[Dict[str, Any]]:
        """Get blend history"""
        history = []
        
        for blend_id, result in self.blend_history.items():
            history.append({
                "id": blend_id,
                "method": result.metadata.get("method", "unknown"),
                "quality": result.blend_quality,
                "timestamp": result.metadata.get("timestamp", time.time()),
                "num_seams": result.metadata.get("num_seams", 0),
                "processing_time": result.metadata.get("processing_time", 0.0)
            })
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return history
    
    def export_blend_result(
        self,
        result: BlendResult,
        output_path: Union[str, Path],
        format: str = "image"
    ) -> bool:
        """
        Export blend result to file
        
        Args:
            result: Blend result
            output_path: Output file path
            format: Export format
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            
            if format == "image":
                # Export blended image
                if result.blended_image is not None:
                    cv2.imwrite(str(output_path), result.blended_image)
                else:
                    logger.error("No blended image to export")
                    return False
            
            elif format == "mask":
                # Export blend mask
                if result.blend_mask is not None:
                    cv2.imwrite(str(output_path), result.blend_mask)
                else:
                    logger.error("No blend mask to export")
                    return False
            
            elif format == "json":
                # Export metadata and statistics
                data = {
                    "blend_id": result.blend_id,
                    "metadata": result.metadata,
                    "statistics": {
                        "quality": result.blend_quality,
                        "image_shape": result.blended_image.shape if result.blended_image is not None else None,
                        "mask_shape": result.blend_mask.shape if result.blend_mask is not None else None
                    },
                    "exported": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                save_json(data, output_path.with_suffix('.json'))
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported blend result to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting blend result: {e}")
            return False
    
    def clear_blends(self) -> None:
        """Clear all blends"""
        self.blend_history.clear()
        self.blend_queue.clear()
        logger.info("Cleared all blends")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("SeamBlender cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        num_blends = len(self.blend_history)
        avg_quality = np.mean([r.blend_quality for r in self.blend_history.values()]) if self.blend_history else 0.0
        
        return (f"SeamBlender(blends={num_blends}, "
                f"avg_quality={avg_quality:.3f})")


# Factory function for creating seam blenders
def create_seam_blender(
    config: Optional[Dict[str, Any]] = None,
    boundary_detector: Optional[BoundaryDetector] = None,
    max_workers: int = 4
) -> SeamBlender:
    """
    Factory function to create seam blenders
    
    Args:
        config: Configuration dictionary
        boundary_detector: Optional boundary detector
        max_workers: Maximum worker threads
        
    Returns:
        SeamBlender instance
    """
    return SeamBlender(
        config=config,
        boundary_detector=boundary_detector,
        max_workers=max_workers
    )