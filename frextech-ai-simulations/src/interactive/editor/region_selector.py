"""
Region Selector Module
Advanced region selection tools for scene editing
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

# Image processing
import cv2
from PIL import Image, ImageDraw
import scipy.ndimage as ndi
from skimage import measure, segmentation, filters

# Local imports
from ...utils.metrics import Timer, PerformanceMetrics
from ...utils.file_io import save_json, load_json, save_image, load_image

logger = logging.getLogger(__name__)


class SelectionMode(Enum):
    """Selection modes"""
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    LASSO = "lasso"
    MAGIC_WAND = "magic_wand"
    SMART = "smart"
    SEMANTIC = "semantic"
    DEPTH = "depth"


class SelectionAction(Enum):
    """Selection actions"""
    NEW = "new"
    ADD = "add"
    SUBTRACT = "subtract"
    INTERSECT = "intersect"
    INVERT = "invert"


@dataclass
class SelectionRegion:
    """Represents a selected region"""
    region_id: str
    mask: np.ndarray  # Binary mask
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    mode: SelectionMode
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def area(self) -> int:
        """Get area of selection in pixels"""
        return int(np.sum(self.mask))
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center of selection"""
        y, x = np.where(self.mask > 0)
        if len(x) > 0 and len(y) > 0:
            return (float(np.mean(x)), float(np.mean(y)))
        return (self.bounds[0] + self.bounds[2] / 2, 
                self.bounds[1] + self.bounds[3] / 2)
    
    @property
    def perimeter(self) -> float:
        """Get perimeter of selection"""
        contours, _ = cv2.findContours(
            self.mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            return cv2.arcLength(contours[0], True)
        return 0.0
    
    def to_polygon(self, simplify: bool = True) -> List[Tuple[float, float]]:
        """Convert mask to polygon"""
        contours, _ = cv2.findContours(
            self.mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Simplify if requested
        if simplify:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert to list of points
        polygon = [(float(point[0][0]), float(point[0][1])) 
                  for point in contour]
        
        return polygon


class RegionSelector:
    """
    Main region selector for advanced selection tools
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4
    ):
        """
        Initialize region selector
        
        Args:
            config: Configuration dictionary
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.max_workers = max_workers
        
        # Current selection state
        self.current_selection: Optional[SelectionRegion] = None
        self.selection_history: Dict[str, SelectionRegion] = {}
        self.selection_stack: deque = deque(maxlen=20)
        
        # Reference image for selection
        self.reference_image: Optional[np.ndarray] = None
        self.image_metadata: Dict[str, Any] = {}
        
        # Selection tools
        self.active_mode: SelectionMode = SelectionMode.RECTANGLE
        self.tool_settings: Dict[str, Any] = {
            "feather": 10,
            "smoothness": 5,
            "threshold": 30,
            "tolerance": 20,
            "smart_sensitivity": 0.5
        }
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info("RegionSelector initialized")
    
    def set_reference_image(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set reference image for selection
        
        Args:
            image: Reference image
            metadata: Image metadata
            
        Returns:
            True if successful
        """
        if image is None or len(image.shape) not in [2, 3]:
            logger.error("Invalid image format")
            return False
        
        self.reference_image = image.copy()
        self.image_metadata = metadata or {}
        
        # Clear current selection
        self.current_selection = None
        
        logger.info(f"Set reference image: {image.shape}")
        return True
    
    def load_reference_image(
        self,
        image_path: Union[str, Path]
    ) -> bool:
        """
        Load reference image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if successful
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return False
            
            image = load_image(str(image_path))
            
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            metadata = {
                "path": str(image_path),
                "dimensions": image.shape[:2],
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "loaded": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return self.set_reference_image(image, metadata)
            
        except Exception as e:
            logger.error(f"Error loading reference image: {e}")
            return False
    
    def create_selection(
        self,
        mode: Union[str, SelectionMode],
        points: List[Tuple[int, int]],
        action: Union[str, SelectionAction] = SelectionAction.NEW,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a selection region
        
        Args:
            mode: Selection mode
            points: Selection points (interpretation depends on mode)
            action: Selection action
            parameters: Selection parameters
            
        Returns:
            Selection ID if successful, None otherwise
        """
        if self.reference_image is None:
            logger.error("No reference image set")
            return None
        
        timer = Timer()
        
        try:
            # Convert mode if string
            if isinstance(mode, str):
                try:
                    selection_mode = SelectionMode(mode)
                except ValueError:
                    logger.error(f"Unknown selection mode: {mode}")
                    return None
            else:
                selection_mode = mode
            
            # Convert action if string
            if isinstance(action, str):
                try:
                    selection_action = SelectionAction(action)
                except ValueError:
                    logger.error(f"Unknown selection action: {action}")
                    return None
            else:
                selection_action = action
            
            # Merge parameters
            merged_params = self.tool_settings.copy()
            if parameters:
                merged_params.update(parameters)
            
            # Create mask based on mode
            mask = self._create_mask_from_points(
                selection_mode, points, merged_params
            )
            
            if mask is None:
                logger.error("Failed to create selection mask")
                return None
            
            # Apply selection action
            final_mask = self._apply_selection_action(
                mask, selection_action, merged_params
            )
            
            if final_mask is None:
                logger.error("Failed to apply selection action")
                return None
            
            # Calculate bounds
            bounds = self._calculate_mask_bounds(final_mask)
            
            # Generate selection ID
            selection_id = self._generate_selection_id(
                selection_mode, bounds, merged_params
            )
            
            # Create selection region
            selection = SelectionRegion(
                region_id=selection_id,
                mask=final_mask,
                bounds=bounds,
                mode=selection_mode,
                parameters=merged_params,
                metadata={
                    "action": selection_action.value,
                    "points": points,
                    "reference_image": self.image_metadata
                }
            )
            
            # Update current selection
            self.current_selection = selection
            
            # Store in history
            self.selection_history[selection_id] = selection
            self.selection_stack.append(selection_id)
            
            # Update metrics
            self.metrics.record_operation(
                f"create_selection_{selection_mode.value}", 
                timer.elapsed()
            )
            
            logger.info(f"Created selection {selection_id}: "
                       f"{selection_mode.value}, area={selection.area} pixels")
            
            return selection_id
            
        except Exception as e:
            logger.error(f"Error creating selection: {e}")
            return None
    
    def _create_mask_from_points(
        self,
        mode: SelectionMode,
        points: List[Tuple[int, int]],
        parameters: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Create mask from points based on selection mode"""
        if self.reference_image is None:
            return None
        
        h, w = self.reference_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if mode == SelectionMode.RECTANGLE:
            if len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                y_min = min(y1, y2)
                y_max = max(y1, y2)
                
                # Clamp to image bounds
                x_min = max(0, min(x_min, w-1))
                x_max = max(0, min(x_max, w-1))
                y_min = max(0, min(y_min, h-1))
                y_max = max(0, min(y_max, h-1))
                
                mask[y_min:y_max+1, x_min:x_max+1] = 255
        
        elif mode == SelectionMode.ELLIPSE:
            if len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius_x = abs(x2 - x1) // 2
                radius_y = abs(y2 - y1) // 2
                
                cv2.ellipse(
                    mask, 
                    (center_x, center_y), 
                    (radius_x, radius_y), 
                    0, 0, 360, 255, -1
                )
        
        elif mode == SelectionMode.POLYGON:
            if len(points) >= 3:
                polygon_points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon_points], 255)
        
        elif mode == SelectionMode.LASSO:
            if len(points) >= 3:
                polygon_points = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon_points], 255)
        
        elif mode == SelectionMode.MAGIC_WAND:
            if len(points) >= 1:
                seed_point = points[0]
                mask = self._create_magic_wand_mask(seed_point, parameters)
        
        elif mode == SelectionMode.SMART:
            if len(points) >= 1:
                seed_point = points[0]
                mask = self._create_smart_mask(seed_point, parameters)
        
        elif mode == SelectionMode.SEMANTIC:
            if len(points) >= 1:
                seed_point = points[0]
                mask = self._create_semantic_mask(seed_point, parameters)
        
        elif mode == SelectionMode.DEPTH:
            if len(points) >= 1:
                seed_point = points[0]
                mask = self._create_depth_mask(seed_point, parameters)
        
        else:
            logger.error(f"Unsupported selection mode: {mode}")
            return None
        
        return mask
    
    def _create_magic_wand_mask(
        self,
        seed_point: Tuple[int, int],
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Create magic wand selection mask"""
        if self.reference_image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        
        tolerance = parameters.get("tolerance", 20)
        feather = parameters.get("feather", 10)
        
        h, w = self.reference_image.shape[:2]
        
        # Convert to grayscale if needed
        if len(self.reference_image.shape) == 3:
            gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.reference_image
        
        # Get seed value
        x, y = seed_point
        if x < 0 or x >= w or y < 0 or y >= h:
            return np.zeros((h, w), dtype=np.uint8)
        
        seed_value = gray[y, x]
        
        # Create mask using flood fill
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        
        cv2.floodFill(
            gray.copy(), mask, 
            (x, y), 255,
            tolerance, tolerance,
            cv2.FLOODFILL_MASK_ONLY
        )
        
        # Remove padding
        mask = mask[1:-1, 1:-1]
        
        # Apply feathering
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), feather / 2)
            mask = (mask > 127).astype(np.uint8) * 255
        
        return mask
    
    def _create_smart_mask(
        self,
        seed_point: Tuple[int, int],
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Create smart selection mask using edge detection"""
        if self.reference_image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        
        sensitivity = parameters.get("smart_sensitivity", 0.5)
        smoothness = parameters.get("smoothness", 5)
        
        h, w = self.reference_image.shape[:2]
        
        # Convert to grayscale
        if len(self.reference_image.shape) == 3:
            gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.reference_image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply Gaussian blur for smoothness
        if smoothness > 0:
            edges = cv2.GaussianBlur(edges, (0, 0), smoothness)
        
        # Threshold based on sensitivity
        threshold = int(255 * sensitivity)
        _, binary = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
        
        # Find connected components near seed point
        x, y = seed_point
        if 0 <= x < w and 0 <= y < h:
            # Use flood fill to select region
            mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            
            # Invert binary image for flood fill
            binary_inv = 255 - binary
            
            cv2.floodFill(
                binary_inv.copy(), mask, 
                (x, y), 255,
                50, 50,
                cv2.FLOODFILL_MASK_ONLY
            )
            
            mask = mask[1:-1, 1:-1]
            
            # Combine with edge information
            mask = cv2.bitwise_and(mask, 255 - binary)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        return mask
    
    def _create_semantic_mask(
        self,
        seed_point: Tuple[int, int],
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Create semantic selection mask (placeholder)"""
        if self.reference_image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        
        # For now, use magic wand as placeholder
        # In production, would use semantic segmentation model
        return self._create_magic_wand_mask(seed_point, parameters)
    
    def _create_depth_mask(
        self,
        seed_point: Tuple[int, int],
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Create depth-based selection mask (placeholder)"""
        if self.reference_image is None:
            return np.zeros((1, 1), dtype=np.uint8)
        
        # For now, use magic wand as placeholder
        # In production, would use depth information
        return self._create_magic_wand_mask(seed_point, parameters)
    
    def _apply_selection_action(
        self,
        new_mask: np.ndarray,
        action: SelectionAction,
        parameters: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Apply selection action to combine masks"""
        feather = parameters.get("feather", 10)
        
        # Apply feathering to new mask
        if feather > 0:
            new_mask = cv2.GaussianBlur(new_mask, (0, 0), feather / 2)
            new_mask = (new_mask > 127).astype(np.uint8) * 255
        
        if action == SelectionAction.NEW:
            return new_mask
        
        if self.current_selection is None:
            return new_mask
        
        current_mask = self.current_selection.mask
        
        if action == SelectionAction.ADD:
            # Union of masks
            result = cv2.bitwise_or(current_mask, new_mask)
        
        elif action == SelectionAction.SUBTRACT:
            # Subtract new mask from current
            result = cv2.subtract(current_mask, new_mask)
        
        elif action == SelectionAction.INTERSECT:
            # Intersection of masks
            result = cv2.bitwise_and(current_mask, new_mask)
        
        elif action == SelectionAction.INVERT:
            # Invert current selection
            result = cv2.bitwise_not(current_mask)
            
            # Also apply new mask if provided
            if np.any(new_mask > 0):
                result = cv2.bitwise_and(result, new_mask)
        
        else:
            logger.error(f"Unknown selection action: {action}")
            return None
        
        return result
    
    def _calculate_mask_bounds(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate bounding box of mask"""
        if mask is None or mask.size == 0:
            return (0, 0, 0, 0)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        return (x_min, y_min, width, height)
    
    def _generate_selection_id(
        self,
        mode: SelectionMode,
        bounds: Tuple[int, int, int, int],
        parameters: Dict[str, Any]
    ) -> str:
        """Generate unique selection ID"""
        timestamp = int(time.time() * 1000)
        content = f"{mode.value}_{bounds}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def modify_selection(
        self,
        selection_id: str,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """
        Modify existing selection
        
        Args:
            selection_id: ID of selection to modify
            operation: Modification operation
            parameters: Operation parameters
            
        Returns:
            New selection ID if successful, None otherwise
        """
        if selection_id not in self.selection_history:
            logger.error(f"Selection not found: {selection_id}")
            return None
        
        original = self.selection_history[selection_id]
        
        timer = Timer()
        
        try:
            # Apply modification
            if operation == "expand":
                modified_mask = self._expand_selection(
                    original.mask, parameters
                )
            
            elif operation == "contract":
                modified_mask = self._contract_selection(
                    original.mask, parameters
                )
            
            elif operation == "smooth":
                modified_mask = self._smooth_selection(
                    original.mask, parameters
                )
            
            elif operation == "feather":
                modified_mask = self._feather_selection(
                    original.mask, parameters
                )
            
            elif operation == "border":
                modified_mask = self._border_selection(
                    original.mask, parameters
                )
            
            elif operation == "transform":
                modified_mask = self._transform_selection(
                    original.mask, parameters
                )
            
            else:
                logger.error(f"Unknown modification operation: {operation}")
                return None
            
            if modified_mask is None:
                logger.error(f"Modification {operation} failed")
                return None
            
            # Calculate new bounds
            bounds = self._calculate_mask_bounds(modified_mask)
            
            # Generate new selection ID
            new_id = self._generate_selection_id(
                original.mode, bounds, parameters
            )
            
            # Create modified selection
            modified = SelectionRegion(
                region_id=new_id,
                mask=modified_mask,
                bounds=bounds,
                mode=original.mode,
                parameters={**original.parameters, **parameters},
                metadata={
                    **original.metadata,
                    "modified_from": selection_id,
                    "modification": operation,
                    "parameters": parameters
                }
            )
            
            # Update current selection
            self.current_selection = modified
            
            # Store in history
            self.selection_history[new_id] = modified
            self.selection_stack.append(new_id)
            
            # Update metrics
            self.metrics.record_operation(f"modify_selection_{operation}", timer.elapsed())
            
            logger.info(f"Modified selection {selection_id} -> {new_id}: "
                       f"{operation}, area={modified.area} pixels")
            
            return new_id
            
        except Exception as e:
            logger.error(f"Error modifying selection: {e}")
            return None
    
    def _expand_selection(
        self,
        mask: np.ndarray,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Expand selection by specified amount"""
        pixels = parameters.get("pixels", 10)
        iterations = parameters.get("iterations", 1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
        expanded = cv2.dilate(mask, kernel, iterations=iterations)
        
        return expanded
    
    def _contract_selection(
        self,
        mask: np.ndarray,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Contract selection by specified amount"""
        pixels = parameters.get("pixels", 10)
        iterations = parameters.get("iterations", 1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels*2+1, pixels*2+1))
        contracted = cv2.erode(mask, kernel, iterations=iterations)
        
        return contracted
    
    def _smooth_selection(
        self,
        mask: np.ndarray,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Smooth selection boundaries"""
        iterations = parameters.get("iterations", 3)
        
        smoothed = mask.copy()
        
        for _ in range(iterations):
            # Opening to remove small protrusions
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            # Closing to fill small holes
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        return smoothed
    
    def _feather_selection(
        self,
        mask: np.ndarray,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Feather selection edges"""
        radius = parameters.get("radius", 10)
        
        if radius <= 0:
            return mask
        
        # Convert to float for smooth blending
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(mask_float, (0, 0), radius / 2)
        
        # Threshold to maintain core selection
        threshold = parameters.get("threshold", 0.1)
        feathered = (blurred > threshold).astype(np.uint8) * 255
        
        return feathered
    
    def _border_selection(
        self,
        mask: np.ndarray,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Create border selection"""
        width = parameters.get("width", 5)
        
        # Get inner border by eroding
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width*2+1, width*2+1))
        inner = cv2.erode(mask, kernel)
        
        # Border is difference between original and inner
        border = cv2.subtract(mask, inner)
        
        return border
    
    def _transform_selection(
        self,
        mask: np.ndarray,
        parameters: Dict[str, Any]
    ) -> np.ndarray:
        """Transform selection (scale, rotate, translate)"""
        operation = parameters.get("operation", "scale")
        
        h, w = mask.shape
        
        if operation == "scale":
            scale = parameters.get("scale", 1.5)
            new_h, new_w = int(h * scale), int(w * scale)
            
            if new_h > 0 and new_w > 0:
                scaled = cv2.resize(mask, (new_w, new_h), 
                                   interpolation=cv2.INTER_NEAREST)
                
                # Center in original dimensions
                result = np.zeros((h, w), dtype=np.uint8)
                y_offset = max(0, (h - new_h) // 2)
                x_offset = max(0, (w - new_w) // 2)
                
                y_end = min(h, y_offset + new_h)
                x_end = min(w, x_offset + new_w)
                
                result[y_offset:y_end, x_offset:x_end] = \
                    scaled[:y_end-y_offset, :x_end-x_offset]
                
                return result
        
        elif operation == "rotate":
            angle = parameters.get("angle", 45)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(mask, matrix, (w, h), 
                                    flags=cv2.INTER_NEAREST)
            return rotated
        
        elif operation == "translate":
            tx = parameters.get("tx", 0)
            ty = parameters.get("ty", 0)
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            translated = cv2.warpAffine(mask, matrix, (w, h), 
                                       flags=cv2.INTER_NEAREST)
            return translated
        
        return mask
    
    def get_selection_preview(
        self,
        selection_id: Optional[str] = None,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        overlay_alpha: float = 0.3,
        show_bounds: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get preview of selection over reference image
        
        Args:
            selection_id: Selection ID (None for current)
            overlay_color: Overlay color
            overlay_alpha: Overlay transparency
            show_bounds: Whether to show bounding box
            
        Returns:
            Preview image
        """
        if self.reference_image is None:
            return None
        
        # Get selection
        if selection_id is None:
            selection = self.current_selection
        else:
            selection = self.selection_history.get(selection_id)
        
        if selection is None:
            # Return reference image without overlay
            if len(self.reference_image.shape) == 2:
                return cv2.cvtColor(self.reference_image, cv2.COLOR_GRAY2BGR)
            else:
                return self.reference_image.copy()
        
        # Create color image from reference
        if len(self.reference_image.shape) == 2:
            preview = cv2.cvtColor(self.reference_image, cv2.COLOR_GRAY2BGR)
        else:
            preview = self.reference_image.copy()
        
        # Create selection overlay
        overlay = np.zeros_like(preview)
        overlay[:, :] = overlay_color
        
        # Apply mask to overlay
        mask = selection.mask > 0
        overlay[~mask] = 0
        
        # Blend overlay with preview
        preview = cv2.addWeighted(preview, 1, overlay, overlay_alpha, 0)
        
        # Draw bounding box if requested
        if show_bounds:
            x, y, w, h = selection.bounds
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Draw center
            center_x, center_y = int(selection.center[0]), int(selection.center[1])
            cv2.circle(preview, (center_x, center_y), 5, (0, 0, 255), -1)
        
        return preview
    
    def get_selection_stats(self, selection_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for selection"""
        selection = self._get_selection(selection_id)
        
        if selection is None:
            return {}
        
        # Calculate various statistics
        mask = selection.mask > 0
        
        # Basic stats
        area = selection.area
        perimeter = selection.perimeter
        bounds = selection.bounds
        
        # Shape statistics
        if area > 0:
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Aspect ratio
            _, _, w, h = bounds
            aspect_ratio = w / h if h > 0 else 0
            
            # Solidity (area / convex hull area)
            contours, _ = cv2.findContours(
                selection.mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
            else:
                solidity = 0
            
            # Extent (area / bounding box area)
            bbox_area = w * h
            extent = area / bbox_area if bbox_area > 0 else 0
            
        else:
            circularity = aspect_ratio = solidity = extent = 0
        
        return {
            "id": selection.region_id,
            "area": area,
            "perimeter": perimeter,
            "bounds": bounds,
            "center": selection.center,
            "circularity": float(circularity),
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity),
            "extent": float(extent),
            "mode": selection.mode.value,
            "timestamp": selection.timestamp
        }
    
    def export_selection(
        self,
        selection_id: Optional[str] = None,
        output_path: Union[str, Path],
        format: str = "png"
    ) -> bool:
        """
        Export selection to file
        
        Args:
            selection_id: Selection ID (None for current)
            output_path: Output file path
            format: Output format
            
        Returns:
            True if successful
        """
        selection = self._get_selection(selection_id)
        
        if selection is None:
            logger.error("No selection to export")
            return False
        
        try:
            output_path = Path(output_path)
            
            if format == "mask":
                # Export mask as binary image
                cv2.imwrite(str(output_path), selection.mask)
            
            elif format == "polygon":
                # Export as polygon JSON
                polygon = selection.to_polygon(simplify=True)
                
                data = {
                    "selection_id": selection.region_id,
                    "bounds": selection.bounds,
                    "polygon": polygon,
                    "area": selection.area,
                    "center": selection.center,
                    "metadata": selection.metadata
                }
                
                save_json(data, output_path.with_suffix('.json'))
            
            elif format == "preview":
                # Export preview image
                preview = self.get_selection_preview(selection_id)
                
                if preview is not None:
                    cv2.imwrite(str(output_path), preview)
                else:
                    logger.error("Failed to create preview")
                    return False
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported selection to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting selection: {e}")
            return False
    
    def _get_selection(self, selection_id: Optional[str] = None) -> Optional[SelectionRegion]:
        """Get selection by ID or current selection"""
        if selection_id is None:
            return self.current_selection
        
        return self.selection_history.get(selection_id)
    
    def clear_selection(self) -> None:
        """Clear current selection"""
        self.current_selection = None
        logger.info("Cleared current selection")
    
    def clear_all_selections(self) -> None:
        """Clear all selections"""
        self.current_selection = None
        self.selection_history.clear()
        self.selection_stack.clear()
        logger.info("Cleared all selections")
    
    def get_selection_list(self) -> List[Dict[str, Any]]:
        """Get list of all selections"""
        selections = []
        
        for selection_id, selection in self.selection_history.items():
            selections.append({
                "id": selection_id,
                "mode": selection.mode.value,
                "bounds": selection.bounds,
                "area": selection.area,
                "timestamp": selection.timestamp,
                "is_current": (self.current_selection and 
                              self.current_selection.region_id == selection_id)
            })
        
        # Sort by timestamp (newest first)
        selections.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return selections
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("RegionSelector cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        if self.reference_image is None:
            return "RegionSelector(no reference image)"
        
        h, w = self.reference_image.shape[:2]
        num_selections = len(self.selection_history)
        current_area = self.current_selection.area if self.current_selection else 0
        
        return (f"RegionSelector({w}x{h}, "
                f"selections={num_selections}, "
                f"current_area={current_area})")


# Factory function for creating region selectors
def create_region_selector(
    config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4
) -> RegionSelector:
    """
    Factory function to create region selectors
    
    Args:
        config: Configuration dictionary
        max_workers: Maximum worker threads
        
    Returns:
        RegionSelector instance
    """
    return RegionSelector(config=config, max_workers=max_workers)