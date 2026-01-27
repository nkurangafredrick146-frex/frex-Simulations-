"""
Panorama Editor Module
Specialized editor for panoramic scene editing and manipulation
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
from PIL import Image, ImageFilter, ImageOps
import imageio

# Local imports
from ...utils.metrics import Timer, PerformanceMetrics
from ...utils.file_io import save_json, load_json, save_image, load_image
from .consistency_checker import ConsistencyChecker
from .edit_propagator import EditPropagator

logger = logging.getLogger(__name__)


class PanoProjection(Enum):
    """Panorama projection types"""
    EQUIRECTANGULAR = "equirectangular"
    CUBEMAP = "cubemap"
    FISHEYE = "fisheye"
    LITTLE_PLANET = "little_planet"
    STEREOGRAPHIC = "stereographic"


class EditTool(Enum):
    """Panorama editing tools"""
    BRUSH = "brush"
    CLONE = "clone"
    HEAL = "heal"
    WARP = "warp"
    FILTER = "filter"
    SELECT = "select"
    TRANSFORM = "transform"


@dataclass
class PanoEdit:
    """Record of a panorama edit"""
    edit_id: str
    tool: EditTool
    region: Tuple[int, int, int, int]  # x, y, width, height
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    layer: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Generate hash for edit"""
        content = f"{self.edit_id}_{self.tool.value}_{self.layer}_{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class PanoramaEditor:
    """
    Main panorama editor for panoramic scene editing
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        consistency_checker: Optional[ConsistencyChecker] = None,
        edit_propagator: Optional[EditPropagator] = None,
        max_workers: int = 4
    ):
        """
        Initialize panorama editor
        
        Args:
            config: Configuration dictionary
            consistency_checker: Optional consistency checker
            edit_propagator: Optional edit propagator
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.consistency_checker = consistency_checker
        self.edit_propagator = edit_propagator
        self.max_workers = max_workers
        
        # Current panorama
        self.current_pano: Optional[np.ndarray] = None
        self.pano_projection: PanoProjection = PanoProjection.EQUIRECTANGULAR
        self.pano_metadata: Dict[str, Any] = {}
        
        # Edit tracking
        self.edit_history: Dict[str, PanoEdit] = {}
        self.edit_stack: deque = deque(maxlen=50)
        self.edit_layers: Dict[str, np.ndarray] = {}
        
        # Tools and brushes
        self.active_tool: EditTool = EditTool.BRUSH
        self.brush_settings: Dict[str, Any] = {
            "size": 20,
            "hardness": 0.8,
            "opacity": 1.0,
            "flow": 0.5,
            "color": [255, 255, 255]
        }
        
        # Selection state
        self.selection_mask: Optional[np.ndarray] = None
        self.selection_active: bool = False
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize tools
        self._initialize_tools()
        
        logger.info("PanoramaEditor initialized")
    
    def _initialize_tools(self) -> None:
        """Initialize editing tools"""
        self.tools = {
            EditTool.BRUSH: self._apply_brush_tool,
            EditTool.CLONE: self._apply_clone_tool,
            EditTool.HEAL: self._apply_heal_tool,
            EditTool.WARP: self._apply_warp_tool,
            EditTool.FILTER: self._apply_filter_tool,
            EditTool.SELECT: self._apply_select_tool,
            EditTool.TRANSFORM: self._apply_transform_tool
        }
    
    def load_panorama(
        self,
        image_path: Union[str, Path],
        projection: Union[str, PanoProjection] = PanoProjection.EQUIRECTANGULAR
    ) -> bool:
        """
        Load panorama image
        
        Args:
            image_path: Path to panorama image
            projection: Panorama projection type
            
        Returns:
            True if successful
        """
        timer = Timer()
        
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                logger.error(f"Panorama file not found: {image_path}")
                return False
            
            # Load image
            self.current_pano = load_image(str(image_path))
            
            if self.current_pano is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Set projection
            if isinstance(projection, str):
                try:
                    self.pano_projection = PanoProjection(projection)
                except ValueError:
                    logger.warning(f"Unknown projection: {projection}, using equirectangular")
                    self.pano_projection = PanoProjection.EQUIRECTANGULAR
            else:
                self.pano_projection = projection
            
            # Initialize metadata
            self.pano_metadata = {
                "path": str(image_path),
                "dimensions": self.current_pano.shape[:2],
                "projection": self.pano_projection.value,
                "loaded": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Initialize edit layers
            self._initialize_layers()
            
            # Clear edit history
            self.edit_history.clear()
            self.edit_stack.clear()
            
            logger.info(f"Loaded panorama: {image_path.name} "
                       f"({self.current_pano.shape[1]}x{self.current_pano.shape[0]}) "
                       f"in {timer.elapsed():.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading panorama: {e}")
            return False
    
    def _initialize_layers(self) -> None:
        """Initialize edit layers"""
        if self.current_pano is None:
            return
        
        h, w = self.current_pano.shape[:2]
        
        # Base layer (original panorama)
        self.edit_layers["base"] = self.current_pano.copy()
        
        # Edit layer (current edits)
        self.edit_layers["edits"] = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Mask layer (selection masks)
        self.edit_layers["mask"] = np.zeros((h, w), dtype=np.uint8)
        
        # Temporary layer (preview)
        self.edit_layers["preview"] = np.zeros((h, w, 4), dtype=np.uint8)
    
    def create_panorama(
        self,
        width: int = 4096,
        height: int = 2048,
        color: Tuple[int, int, int] = (0, 0, 0)
    ) -> bool:
        """
        Create a new blank panorama
        
        Args:
            width: Panorama width
            height: Panorama height
            color: Background color
            
        Returns:
            True if successful
        """
        try:
            # Create blank panorama
            self.current_pano = np.full((height, width, 3), color, dtype=np.uint8)
            
            # Set metadata
            self.pano_metadata = {
                "dimensions": (height, width),
                "projection": self.pano_projection.value,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "color": color
            }
            
            # Initialize layers
            self._initialize_layers()
            
            # Clear edit history
            self.edit_history.clear()
            self.edit_stack.clear()
            
            logger.info(f"Created new panorama: {width}x{height}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating panorama: {e}")
            return False
    
    def apply_edit(
        self,
        tool: Union[str, EditTool],
        region: Tuple[int, int, int, int],
        parameters: Optional[Dict[str, Any]] = None,
        layer: str = "edits"
    ) -> Optional[str]:
        """
        Apply an edit to the panorama
        
        Args:
            tool: Editing tool to use
            region: Edit region (x, y, width, height)
            parameters: Tool parameters
            layer: Target layer
            
        Returns:
            Edit ID if successful, None otherwise
        """
        if self.current_pano is None:
            logger.error("No panorama loaded")
            return None
        
        timer = Timer()
        
        try:
            # Convert tool if string
            if isinstance(tool, str):
                try:
                    edit_tool = EditTool(tool)
                except ValueError:
                    logger.error(f"Unknown tool: {tool}")
                    return None
            else:
                edit_tool = tool
            
            if edit_tool not in self.tools:
                logger.error(f"Tool not implemented: {edit_tool}")
                return None
            
            # Validate region
            x, y, w, h = region
            pano_h, pano_w = self.current_pano.shape[:2]
            
            if x < 0 or y < 0 or x + w > pano_w or y + h > pano_h:
                logger.error(f"Edit region out of bounds: {region}")
                return None
            
            # Merge tool parameters with defaults
            merged_params = self._merge_tool_parameters(edit_tool, parameters or {})
            
            # Generate edit ID
            edit_id = self._generate_edit_id(edit_tool, region)
            
            # Apply tool
            result = self.tools[edit_tool](region, merged_params, layer)
            
            if result is None:
                logger.error(f"Tool {edit_tool.value} failed")
                return None
            
            # Record edit
            edit = PanoEdit(
                edit_id=edit_id,
                tool=edit_tool,
                region=region,
                parameters=merged_params,
                layer=layer,
                metadata={
                    "tool": edit_tool.value,
                    "region": region,
                    "parameters": merged_params
                }
            )
            
            self.edit_history[edit_id] = edit
            self.edit_stack.append(edit_id)
            
            # Update metrics
            self.metrics.record_operation(f"apply_edit_{edit_tool.value}", timer.elapsed())
            
            logger.debug(f"Applied edit {edit_id}: {edit_tool.value} "
                        f"to region {region} in {timer.elapsed():.2f}s")
            
            return edit_id
            
        except Exception as e:
            logger.error(f"Error applying edit: {e}")
            return None
    
    def _merge_tool_parameters(
        self,
        tool: EditTool,
        user_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge user parameters with tool defaults"""
        defaults = {
            EditTool.BRUSH: self.brush_settings.copy(),
            EditTool.CLONE: {
                "source_region": None,
                "alignment": True,
                "blending": 0.7
            },
            EditTool.HEAL: {
                "method": "telea",
                "radius": 10
            },
            EditTool.WARP: {
                "strength": 0.5,
                "method": "grid",
                "grid_size": 20
            },
            EditTool.FILTER: {
                "filter_type": "gaussian",
                "strength": 1.0,
                "radius": 5.0
            },
            EditTool.SELECT: {
                "mode": "rectangle",
                "feather": 10,
                "invert": False
            },
            EditTool.TRANSFORM: {
                "operation": "scale",
                "scale_factor": 1.0,
                "rotation": 0.0,
                "translation": [0, 0]
            }
        }
        
        merged = defaults.get(tool, {}).copy()
        merged.update(user_params)
        
        return merged
    
    def _generate_edit_id(
        self,
        tool: EditTool,
        region: Tuple[int, int, int, int]
    ) -> str:
        """Generate unique edit ID"""
        timestamp = int(time.time() * 1000)
        content = f"{tool.value}_{region}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    # Tool implementations
    def _apply_brush_tool(
        self,
        region: Tuple[int, int, int, int],
        parameters: Dict[str, Any],
        layer: str
    ) -> Optional[np.ndarray]:
        """Apply brush tool"""
        if self.current_pano is None:
            return None
        
        x, y, w, h = region
        brush_size = parameters.get("size", 20)
        hardness = parameters.get("hardness", 0.8)
        opacity = parameters.get("opacity", 1.0)
        color = parameters.get("color", [255, 255, 255])
        
        # Create brush mask
        mask = self._create_brush_mask(w, h, brush_size, hardness)
        
        # Apply to layer
        if layer in self.edit_layers:
            layer_data = self.edit_layers[layer]
            
            # Convert color to array
            color_array = np.array(color, dtype=np.uint8)
            
            # Apply brush strokes
            if len(layer_data.shape) == 3:  # RGB or RGBA
                if layer_data.shape[2] == 3:  # RGB
                    for c in range(3):
                        layer_data[y:y+h, x:x+w, c] = np.where(
                            mask > 0,
                            color_array[c] * opacity + layer_data[y:y+h, x:x+w, c] * (1 - opacity),
                            layer_data[y:y+h, x:x+w, c]
                        )
                elif layer_data.shape[2] == 4:  # RGBA
                    alpha_mask = (mask * 255 * opacity).astype(np.uint8)
                    
                    # Create brush overlay
                    overlay = np.zeros((h, w, 4), dtype=np.uint8)
                    overlay[:, :, :3] = color_array
                    overlay[:, :, 3] = alpha_mask
                    
                    # Blend with layer
                    self._blend_layers(layer_data[y:y+h, x:x+w], overlay)
            
            self.edit_layers[layer] = layer_data
        
        return self._get_layer_region(layer, region)
    
    def _create_brush_mask(
        self,
        width: int,
        height: int,
        brush_size: int,
        hardness: float
    ) -> np.ndarray:
        """Create brush stroke mask"""
        # Create circular brush
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[-center_y:height-center_y, -center_x:width-center_x]
        distance = np.sqrt(x*x + y*y)
        
        # Create soft brush
        mask = np.clip(1 - distance / (brush_size / 2), 0, 1)
        
        # Apply hardness
        mask = mask ** (2 / (hardness + 0.1))
        
        return mask
    
    def _apply_clone_tool(
        self,
        region: Tuple[int, int, int, int],
        parameters: Dict[str, Any],
        layer: str
    ) -> Optional[np.ndarray]:
        """Apply clone tool"""
        if self.current_pano is None:
            return None
        
        x, y, w, h = region
        source_region = parameters.get("source_region")
        blending = parameters.get("blending", 0.7)
        
        if source_region is None:
            logger.error("Clone tool requires source region")
            return None
        
        sx, sy, sw, sh = source_region
        
        # Validate source region
        pano_h, pano_w = self.current_pano.shape[:2]
        if sx < 0 or sy < 0 or sx + sw > pano_w or sy + sh > pano_w:
            logger.error("Source region out of bounds")
            return None
        
        # Ensure regions have same size
        if sw != w or sh != h:
            # Resize source to match target
            source_data = self.current_pano[sy:sy+sh, sx:sx+sw]
            source_data = cv2.resize(source_data, (w, h))
        else:
            source_data = self.current_pano[sy:sy+h, sx:sx+w]
        
        # Apply to layer
        if layer in self.edit_layers:
            layer_data = self.edit_layers[layer]
            
            if len(layer_data.shape) == 3:
                # Blend source with target
                layer_data[y:y+h, x:x+w] = cv2.addWeighted(
                    layer_data[y:y+h, x:x+w], 1 - blending,
                    source_data, blending, 0
                )
            
            self.edit_layers[layer] = layer_data
        
        return self._get_layer_region(layer, region)
    
    def _apply_heal_tool(
        self,
        region: Tuple[int, int, int, int],
        parameters: Dict[str, Any],
        layer: str
    ) -> Optional[np.ndarray]:
        """Apply heal/inpainting tool"""
        if self.current_pano is None:
            return None
        
        x, y, w, h = region
        method = parameters.get("method", "telea")
        radius = parameters.get("radius", 10)
        
        # Get region data
        if layer in self.edit_layers:
            layer_data = self.edit_layers[layer]
            region_data = layer_data[y:y+h, x:x+w].copy()
        else:
            region_data = self.current_pano[y:y+h, x:x+w].copy()
        
        # Create mask (center of region)
        mask = np.zeros((h, w), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Apply inpainting
        if method == "telea":
            healed = cv2.inpaint(region_data, mask, radius, cv2.INPAINT_TELEA)
        else:  # navier-stokes
            healed = cv2.inpaint(region_data, mask, radius, cv2.INPAINT_NS)
        
        # Apply to layer
        if layer in self.edit_layers:
            layer_data[y:y+h, x:x+w] = healed
            self.edit_layers[layer] = layer_data
        
        return healed
    
    def _apply_warp_tool(
        self,
        region: Tuple[int, int, int, int],
        parameters: Dict[str, Any],
        layer: str
    ) -> Optional[np.ndarray]:
        """Apply warp/distortion tool"""
        if self.current_pano is None:
            return None
        
        x, y, w, h = region
        strength = parameters.get("strength", 0.5)
        method = parameters.get("method", "grid")
        grid_size = parameters.get("grid_size", 20)
        
        # Get region data
        if layer in self.edit_layers:
            layer_data = self.edit_layers[layer]
            region_data = layer_data[y:y+h, x:x+w].copy()
        else:
            region_data = self.current_pano[y:y+h, x:x+w].copy()
        
        if method == "grid":
            # Create grid warp
            warped = self._apply_grid_warp(region_data, strength, grid_size)
        elif method == "bulge":
            # Create bulge effect
            warped = self._apply_bulge_warp(region_data, strength)
        elif method == "pinch":
            # Create pinch effect
            warped = self._apply_pinch_warp(region_data, strength)
        else:
            logger.warning(f"Unknown warp method: {method}")
            return None
        
        # Apply to layer
        if layer in self.edit_layers:
            layer_data[y:y+h, x:x+w] = warped
            self.edit_layers[layer] = layer_data
        
        return warped
    
    def _apply_grid_warp(
        self,
        image: np.ndarray,
        strength: float,
        grid_size: int
    ) -> np.ndarray:
        """Apply grid-based warp"""
        h, w = image.shape[:2]
        
        # Create grid
        grid_x = np.linspace(0, w, grid_size)
        grid_y = np.linspace(0, h, grid_size)
        
        # Create distortion
        distortion = (np.random.randn(grid_size, grid_size, 2) * strength * 10).astype(np.float32)
        
        # Apply remapping
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                # Define source and destination quadrilaterals
                src_pts = np.array([
                    [grid_x[i], grid_y[j]],
                    [grid_x[i+1], grid_y[j]],
                    [grid_x[i], grid_y[j+1]],
                    [grid_x[i+1], grid_y[j+1]]
                ], dtype=np.float32)
                
                dst_pts = src_pts + distortion[j:j+2, i:i+2].reshape(4, 2)
                
                # Create transformation
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                
                # Apply to grid cell
                cell_h = int(grid_y[j+1] - grid_y[j])
                cell_w = int(grid_x[i+1] - grid_x[i])
                
                if cell_h > 0 and cell_w > 0:
                    cell_map_x, cell_map_y = np.meshgrid(
                        np.arange(cell_w), np.arange(cell_h)
                    )
                    
                    # Transform coordinates
                    ones = np.ones(cell_h * cell_w)
                    coords = np.vstack([cell_map_x.flatten(), cell_map_y.flatten(), ones])
                    transformed = matrix @ coords
                    
                    transformed_x = transformed[0] / transformed[2] + grid_x[i]
                    transformed_y = transformed[1] / transformed[2] + grid_y[j]
                    
                    # Update maps
                    start_y = int(grid_y[j])
                    start_x = int(grid_x[i])
                    map_x[start_y:start_y+cell_h, start_x:start_x+cell_w] = transformed_x.reshape(cell_h, cell_w)
                    map_y[start_y:start_y+cell_h, start_x:start_x+cell_w] = transformed_y.reshape(cell_h, cell_w)
        
        # Apply remapping
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return warped
    
    def _apply_bulge_warp(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply bulge warp effect"""
        h, w = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        
        # Calculate distances
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx*dx + dy*dy)
        max_distance = np.sqrt(center_x*center_x + center_y*center_y)
        
        # Create bulge effect
        scale = 1 + strength * (1 - distance / max_distance)
        scale = np.clip(scale, 0.5, 2.0)
        
        # Calculate new coordinates
        new_x = center_x + dx * scale
        new_y = center_y + dy * scale
        
        # Create maps
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        # Apply remapping
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return warped
    
    def _apply_pinch_warp(
        self,
        image: np.ndarray,
        strength: float
    ) -> np.ndarray:
        """Apply pinch warp effect"""
        h, w = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        
        # Calculate distances
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx*dx + dy*dy)
        max_distance = np.sqrt(center_x*center_x + center_y*center_y)
        
        # Create pinch effect
        scale = 1 - strength * (distance / max_distance)
        scale = np.clip(scale, 0.5, 1.5)
        
        # Calculate new coordinates
        new_x = center_x + dx * scale
        new_y = center_y + dy * scale
        
        # Create maps
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        # Apply remapping
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return warped
    
    def _apply_filter_tool(
        self,
        region: Tuple[int, int, int, int],
        parameters: Dict[str, Any],
        layer: str
    ) -> Optional[np.ndarray]:
        """Apply filter tool"""
        if self.current_pano is None:
            return None
        
        x, y, w, h = region
        filter_type = parameters.get("filter_type", "gaussian")
        strength = parameters.get("strength", 1.0)
        radius = parameters.get("radius", 5.0)
        
        # Get region data
        if layer in self.edit_layers:
            layer_data = self.edit_layers[layer]
            region_data = layer_data[y:y+h, x:x+w].copy()
        else:
            region_data = self.current_pano[y:y+h, x:x+w].copy()
        
        # Apply filter
        if filter_type == "gaussian":
            filtered = cv2.GaussianBlur(region_data, (0, 0), radius * strength)
        
        elif filter_type == "blur":
            kernel_size = int(radius * strength * 2) * 2 + 1
            filtered = cv2.blur(region_data, (kernel_size, kernel_size))
        
        elif filter_type == "sharpen":
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) * strength
            filtered = cv2.filter2D(region_data, -1, kernel)
        
        elif filter_type == "edge":
            filtered = cv2.Canny(region_data, 100, 200)
            filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
        
        elif filter_type == "emboss":
            kernel = np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]]) * strength
            filtered = cv2.filter2D(region_data, -1, kernel)
            filtered = cv2.addWeighted(region_data, 0.5, filtered, 0.5, 0)
        
        else:
            logger.warning(f"Unknown filter type: {filter_type}")
            return None
        
        # Apply to layer
        if layer in self.edit_layers:
            layer_data[y:y+h, x:x+w] = filtered
            self.edit_layers[layer] = layer_data
        
        return filtered
    
    def _apply_select_tool(
        self,
        region: Tuple[int, int, int, int],
        parameters: Dict[str, Any],
        layer: str
    ) -> Optional[np.ndarray]:
        """Apply selection tool"""
        x, y, w, h = region
        mode = parameters.get("mode", "rectangle")
        feather = parameters.get("feather", 10)
        invert = parameters.get("invert", False)
        
        # Create selection mask
        if mode == "rectangle":
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        elif mode == "ellipse":
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 255, -1)
        
        elif mode == "lasso":
            # For now, use rectangle
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        else:
            logger.warning(f"Unknown selection mode: {mode}")
            return None
        
        # Apply feathering
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), feather / 2)
        
        # Invert if requested
        if invert:
            mask = 255 - mask
        
        # Store selection
        if self.selection_mask is None:
            pano_h, pano_w = self.current_pano.shape[:2]
            self.selection_mask = np.zeros((pano_h, pano_w), dtype=np.uint8)
        
        self.selection_mask[y:y+h, x:x+w] = np.maximum(
            self.selection_mask[y:y+h, x:x+w], mask
        )
        
        self.selection_active = True
        
        return self.selection_mask[y:y+h, x:x+w]
    
    def _apply_transform_tool(
        self,
        region: Tuple[int, int, int, int],
        parameters: Dict[str, Any],
        layer: str
    ) -> Optional[np.ndarray]:
        """Apply transform tool"""
        if self.current_pano is None:
            return None
        
        x, y, w, h = region
        operation = parameters.get("operation", "scale")
        scale_factor = parameters.get("scale_factor", 1.0)
        rotation = parameters.get("rotation", 0.0)
        translation = parameters.get("translation", [0, 0])
        
        # Get region data
        if layer in self.edit_layers:
            layer_data = self.edit_layers[layer]
            region_data = layer_data[y:y+h, x:x+w].copy()
        else:
            region_data = self.current_pano[y:y+h, x:x+w].copy()
        
        transformed = region_data.copy()
        
        if operation == "scale":
            # Scale region
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            
            if new_w > 0 and new_h > 0:
                transformed = cv2.resize(region_data, (new_w, new_h))
                
                # For now, just resize in-place
                # In full implementation, would need to handle boundary
                if new_w <= w and new_h <= h:
                    # Center in original region
                    offset_x = (w - new_w) // 2
                    offset_y = (h - new_h) // 2
                    
                    # Clear region
                    layer_data[y:y+h, x:x+w] = 0
                    
                    # Place transformed
                    layer_data[y+offset_y:y+offset_y+new_h, x+offset_x:x+offset_x+new_w] = transformed
                else:
                    logger.warning("Scaling beyond region bounds not implemented")
        
        elif operation == "rotate":
            # Rotate region
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            transformed = cv2.warpAffine(region_data, matrix, (w, h))
            
            # Apply to layer
            layer_data[y:y+h, x:x+w] = transformed
        
        elif operation == "translate":
            # Translate region
            tx, ty = translation
            
            # Create translation matrix
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            transformed = cv2.warpAffine(region_data, matrix, (w, h))
            
            # Apply to layer
            layer_data[y:y+h, x:x+w] = transformed
        
        # Update layer
        if layer in self.edit_layers:
            self.edit_layers[layer] = layer_data
        
        return transformed
    
    def _blend_layers(self, background: np.ndarray, foreground: np.ndarray) -> None:
        """Blend foreground layer onto background layer"""
        if background.shape[2] == 4 and foreground.shape[2] == 4:
            # Alpha blending for RGBA
            alpha_f = foreground[:, :, 3] / 255.0
            alpha_b = background[:, :, 3] / 255.0
            
            for c in range(3):
                background[:, :, c] = (
                    foreground[:, :, c] * alpha_f +
                    background[:, :, c] * alpha_b * (1 - alpha_f)
                ) / (alpha_f + alpha_b * (1 - alpha_f) + 1e-8)
            
            background[:, :, 3] = (alpha_f + alpha_b * (1 - alpha_f)) * 255
    
    def _get_layer_region(
        self,
        layer: str,
        region: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Get region from layer"""
        if layer not in self.edit_layers:
            return None
        
        x, y, w, h = region
        layer_data = self.edit_layers[layer]
        
        if (y + h <= layer_data.shape[0] and 
            x + w <= layer_data.shape[1]):
            return layer_data[y:y+h, x:x+w].copy()
        
        return None
    
    def get_composite_preview(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        show_layers: List[str] = None
    ) -> Optional[np.ndarray]:
        """
        Get composite preview of panorama with edits
        
        Args:
            region: Optional region to preview
            show_layers: List of layers to include
            
        Returns:
            Composite image
        """
        if self.current_pano is None:
            return None
        
        if show_layers is None:
            show_layers = ["base", "edits", "mask"]
        
        # Start with base layer
        composite = self.edit_layers["base"].copy()
        
        # Apply edits layer
        if "edits" in show_layers and "edits" in self.edit_layers:
            edits_layer = self.edit_layers["edits"]
            if edits_layer.shape[2] == 4:  # RGBA
                # Alpha blend edits
                alpha = edits_layer[:, :, 3:] / 255.0
                composite = composite * (1 - alpha) + edits_layer[:, :, :3] * alpha
            else:  # RGB
                composite = edits_layer
        
        # Apply mask layer
        if "mask" in show_layers and "mask" in self.edit_layers:
            mask = self.edit_layers["mask"]
            if len(mask.shape) == 2:
                # Convert mask to color overlay
                mask_rgb = np.zeros_like(composite)
                mask_rgb[:, :, 0] = mask  # Red channel
                
                # Blend mask
                composite = cv2.addWeighted(composite, 0.7, mask_rgb, 0.3, 0)
        
        # Apply selection
        if self.selection_active and self.selection_mask is not None:
            # Create selection overlay
            selection_overlay = np.zeros_like(composite)
            selection_overlay[:, :, 1] = self.selection_mask  # Green channel
            
            # Blend selection
            composite = cv2.addWeighted(composite, 0.8, selection_overlay, 0.2, 0)
        
        # Extract region if specified
        if region is not None:
            x, y, w, h = region
            if (y + h <= composite.shape[0] and 
                x + w <= composite.shape[1]):
                composite = composite[y:y+h, x:x+w]
            else:
                logger.warning("Preview region out of bounds")
        
        return composite
    
    def undo_last_edit(self) -> bool:
        """Undo last edit"""
        if not self.edit_stack:
            logger.warning("No edits to undo")
            return False
        
        last_edit_id = self.edit_stack.pop()
        
        if last_edit_id in self.edit_history:
            edit = self.edit_history[last_edit_id]
            
            # For now, just remove from history
            # In full implementation, would revert the edit
            del self.edit_history[last_edit_id]
            
            logger.info(f"Undid edit: {last_edit_id} ({edit.tool.value})")
            return True
        
        return False
    
    def save_panorama(
        self,
        output_path: Union[str, Path],
        format: str = "png",
        quality: int = 95
    ) -> bool:
        """
        Save panorama to file
        
        Args:
            output_path: Output file path
            format: Image format
            quality: JPEG quality (if applicable)
            
        Returns:
            True if successful
        """
        if self.current_pano is None:
            logger.error("No panorama to save")
            return False
        
        try:
            output_path = Path(output_path)
            
            # Get composite
            composite = self.get_composite_preview()
            
            if composite is None:
                logger.error("Failed to create composite")
                return False
            
            # Save image
            if format.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(
                    str(output_path), 
                    composite,
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
            else:
                cv2.imwrite(str(output_path), composite)
            
            logger.info(f"Saved panorama to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving panorama: {e}")
            return False
    
    def save_edit_history(
        self,
        output_path: Union[str, Path]
    ) -> bool:
        """Save edit history to file"""
        try:
            output_path = Path(output_path)
            
            history_data = {
                "panorama": self.pano_metadata,
                "edits": [
                    {
                        "id": edit.edit_id,
                        "tool": edit.tool.value,
                        "region": edit.region,
                        "layer": edit.layer,
                        "timestamp": edit.timestamp,
                        "metadata": edit.metadata
                    }
                    for edit in self.edit_history.values()
                ],
                "exported": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            save_json(history_data, output_path.with_suffix('.json'))
            
            logger.info(f"Saved edit history to {output_path.with_suffix('.json')}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving edit history: {e}")
            return False
    
    def load_edit_history(
        self,
        history_path: Union[str, Path]
    ) -> bool:
        """Load edit history from file"""
        try:
            history_path = Path(history_path)
            
            if not history_path.exists():
                logger.error(f"History file not found: {history_path}")
                return False
            
            history_data = load_json(history_path)
            
            # Clear current history
            self.edit_history.clear()
            self.edit_stack.clear()
            
            # Load edits
            for edit_data in history_data.get("edits", []):
                edit = PanoEdit(
                    edit_id=edit_data["id"],
                    tool=EditTool(edit_data["tool"]),
                    region=tuple(edit_data["region"]),
                    parameters=edit_data.get("parameters", {}),
                    layer=edit_data.get("layer", "default"),
                    timestamp=edit_data["timestamp"],
                    metadata=edit_data.get("metadata", {})
                )
                
                self.edit_history[edit.edit_id] = edit
                self.edit_stack.append(edit.edit_id)
            
            logger.info(f"Loaded {len(self.edit_history)} edits from history")
            return True
            
        except Exception as e:
            logger.error(f"Error loading edit history: {e}")
            return False
    
    def clear_edits(self) -> None:
        """Clear all edits"""
        self.edit_history.clear()
        self.edit_stack.clear()
        self._initialize_layers()
        
        logger.info("Cleared all edits")
    
    def get_edit_stats(self) -> Dict[str, Any]:
        """Get statistics about edits"""
        total_edits = len(self.edit_history)
        
        # Count by tool
        tool_counts = defaultdict(int)
        for edit in self.edit_history.values():
            tool_counts[edit.tool.value] += 1
        
        return {
            "total_edits": total_edits,
            "tool_counts": dict(tool_counts),
            "panorama": self.pano_metadata,
            "selection_active": self.selection_active
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("PanoramaEditor cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        if self.current_pano is None:
            return "PanoramaEditor(no panorama loaded)"
        
        h, w = self.current_pano.shape[:2]
        return (f"PanoramaEditor({w}x{h}, "
                f"projection={self.pano_projection.value}, "
                f"edits={len(self.edit_history)})")


# Factory function for creating panorama editors
def create_panorama_editor(
    config: Optional[Dict[str, Any]] = None,
    consistency_checker: Optional[ConsistencyChecker] = None,
    edit_propagator: Optional[EditPropagator] = None,
    max_workers: int = 4
) -> PanoramaEditor:
    """
    Factory function to create panorama editors
    
    Args:
        config: Configuration dictionary
        consistency_checker: Optional consistency checker
        edit_propagator: Optional edit propagator
        max_workers: Maximum worker threads
        
    Returns:
        PanoramaEditor instance
    """
    return PanoramaEditor(
        config=config,
        consistency_checker=consistency_checker,
        edit_propagator=edit_propagator,
        max_workers=max_workers
    )