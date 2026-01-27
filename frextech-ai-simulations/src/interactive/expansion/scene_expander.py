"""
Scene Expander Module
Expands existing scenes by generating new content at boundaries
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

# Image and geometry processing
import cv2
from PIL import Image, ImageFilter
import trimesh
from scipy.spatial import KDTree
from scipy.ndimage import binary_dilation, binary_erosion

# Local imports
from ...utils.metrics import Timer, PerformanceMetrics
from ...utils.file_io import save_json, load_json, save_image, load_image
from ...interactive.composition.scene_composer import SceneComposer, SceneObject
from .boundary_detector import BoundaryDetector, BoundarySegment

logger = logging.getLogger(__name__)


class ExpansionDirection(Enum):
    """Directions for scene expansion"""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    FORWARD = "forward"
    BACKWARD = "backward"
    ALL = "all"
    CUSTOM = "custom"


class ExpansionMethod(Enum):
    """Methods for scene expansion"""
    GENERATIVE = "generative"      # Use AI to generate new content
    MIRROR = "mirror"              # Mirror existing content
    TILE = "tile"                  # Tile existing patterns
    EXTRAPOLATE = "extrapolate"    # Extrapolate from existing
    BLEND = "blend"                # Blend with new content
    COMPOSITION = "composition"    # Compose from existing objects


@dataclass
class ExpansionRequest:
    """Request for scene expansion"""
    direction: ExpansionDirection
    distance: float
    method: ExpansionMethod
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Generate hash for expansion request"""
        content = f"{self.direction.value}_{self.distance}_{self.method.value}_{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class ExpansionResult:
    """Result of scene expansion"""
    expansion_id: str
    original_scene: Dict[str, Any]
    expanded_scene: Dict[str, Any]
    new_objects: List[Dict[str, Any]]
    expansion_mask: Optional[np.ndarray] = None
    boundaries: List[BoundarySegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def expansion_area(self) -> float:
        """Calculate total expansion area"""
        if self.expansion_mask is not None:
            return float(np.sum(self.expansion_mask > 0))
        return 0.0
    
    @property
    def num_new_objects(self) -> int:
        """Get number of new objects"""
        return len(self.new_objects)


class SceneExpander:
    """
    Main scene expander for generating new content at scene boundaries
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        boundary_detector: Optional[BoundaryDetector] = None,
        scene_composer: Optional[SceneComposer] = None,
        max_workers: int = 4
    ):
        """
        Initialize scene expander
        
        Args:
            config: Configuration dictionary
            boundary_detector: Optional boundary detector
            scene_composer: Optional scene composer
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.boundary_detector = boundary_detector
        self.scene_composer = scene_composer
        self.max_workers = max_workers
        
        # Expansion methods registry
        self.expansion_methods = self._initialize_expansion_methods()
        
        # Expansion history
        self.expansion_history: Dict[str, ExpansionResult] = {}
        self.expansion_queue: deque = deque(maxlen=50)
        
        # Scene state
        self.current_scene: Optional[Dict[str, Any]] = None
        self.scene_bounds: Optional[Tuple[float, float, float, float]] = None
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Content generation models (would be loaded in production)
        self.generation_models = {}
        
        logger.info("SceneExpander initialized")
    
    def _initialize_expansion_methods(self) -> Dict[ExpansionMethod, Callable]:
        """Initialize expansion methods registry"""
        return {
            ExpansionMethod.GENERATIVE: self._expand_generative,
            ExpansionMethod.MIRROR: self._expand_mirror,
            ExpansionMethod.TILE: self._expand_tile,
            ExpansionMethod.EXTRAPOLATE: self._expand_extrapolate,
            ExpansionMethod.BLEND: self._expand_blend,
            ExpansionMethod.COMPOSITION: self._expand_composition
        }
    
    def load_scene(
        self,
        scene_data: Dict[str, Any],
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> bool:
        """
        Load scene for expansion
        
        Args:
            scene_data: Scene data dictionary
            bounds: Scene bounds (x_min, x_max, z_min, z_max)
            
        Returns:
            True if successful
        """
        if not scene_data or "objects" not in scene_data:
            logger.error("Invalid scene data")
            return False
        
        self.current_scene = scene_data.copy()
        
        # Calculate bounds if not provided
        if bounds is None:
            self.scene_bounds = self._calculate_scene_bounds(scene_data)
        else:
            self.scene_bounds = bounds
        
        logger.info(f"Loaded scene with {len(scene_data['objects'])} objects, "
                   f"bounds: {self.scene_bounds}")
        
        return True
    
    def _calculate_scene_bounds(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[float, float, float, float]:
        """Calculate scene bounds from objects"""
        objects = scene_data.get("objects", {})
        
        if not objects:
            return (-10.0, 10.0, -10.0, 10.0)
        
        min_x, max_x = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        
        for obj_id, obj_data in objects.items():
            if "position" in obj_data:
                pos = obj_data["position"]
                min_x = min(min_x, pos[0])
                max_x = max(max_x, pos[0])
                min_z = min(min_z, pos[2])
                max_z = max(max_z, pos[2])
        
        # Add padding
        padding = 5.0
        min_x -= padding
        max_x += padding
        min_z -= padding
        max_z += padding
        
        return (min_x, max_x, min_z, max_z)
    
    def expand_scene(
        self,
        request: Union[ExpansionRequest, Dict[str, Any]],
        generate_content: bool = True
    ) -> Optional[ExpansionResult]:
        """
        Expand scene according to expansion request
        
        Args:
            request: Expansion request or parameters dictionary
            generate_content: Whether to generate new content
            
        Returns:
            Expansion result or None if failed
        """
        if self.current_scene is None:
            logger.error("No scene loaded")
            return None
        
        timer = Timer()
        
        try:
            # Parse request
            if isinstance(request, dict):
                expansion_request = self._parse_expansion_request(request)
            else:
                expansion_request = request
            
            expansion_id = self._generate_expansion_id(expansion_request)
            
            logger.info(f"Expanding scene {expansion_id}: "
                       f"{expansion_request.direction.value} by "
                       f"{expansion_request.distance} units using "
                       f"{expansion_request.method.value}")
            
            # Get expansion method
            expansion_method = self.expansion_methods.get(expansion_request.method)
            if expansion_method is None:
                logger.error(f"Unknown expansion method: {expansion_request.method}")
                return None
            
            # Detect boundaries if needed
            boundaries = []
            if self.boundary_detector:
                boundaries = self._detect_expansion_boundaries(expansion_request)
            
            # Apply expansion
            expansion_result = expansion_method(
                expansion_request, 
                boundaries,
                generate_content
            )
            
            if expansion_result is None:
                logger.error("Expansion method failed")
                return None
            
            # Set expansion ID
            expansion_result.expansion_id = expansion_id
            
            # Update scene state
            self.current_scene = expansion_result.expanded_scene
            
            # Update bounds
            self.scene_bounds = self._update_scene_bounds(
                expansion_result.expanded_scene,
                expansion_request
            )
            
            # Store in history
            self.expansion_history[expansion_id] = expansion_result
            self.expansion_queue.append(expansion_id)
            
            # Update metrics
            self.metrics.record_operation(
                f"expand_scene_{expansion_request.method.value}", 
                timer.elapsed()
            )
            
            logger.info(f"Scene expansion completed in {timer.elapsed():.2f}s: "
                       f"{expansion_result.num_new_objects} new objects, "
                       f"area={expansion_result.expansion_area:.1f}")
            
            return expansion_result
            
        except Exception as e:
            logger.error(f"Error expanding scene: {e}")
            return None
    
    def _parse_expansion_request(self, params: Dict[str, Any]) -> ExpansionRequest:
        """Parse expansion request from dictionary"""
        # Get direction
        direction_str = params.get("direction", "right")
        try:
            direction = ExpansionDirection(direction_str)
        except ValueError:
            logger.warning(f"Unknown direction: {direction_str}, using RIGHT")
            direction = ExpansionDirection.RIGHT
        
        # Get method
        method_str = params.get("method", "composition")
        try:
            method = ExpansionMethod(method_str)
        except ValueError:
            logger.warning(f"Unknown method: {method_str}, using COMPOSITION")
            method = ExpansionMethod.COMPOSITION
        
        # Get distance
        distance = float(params.get("distance", 10.0))
        
        # Get parameters
        expansion_params = params.get("parameters", {})
        
        # Get constraints
        constraints = params.get("constraints", {})
        
        return ExpansionRequest(
            direction=direction,
            distance=distance,
            method=method,
            parameters=expansion_params,
            constraints=constraints
        )
    
    def _generate_expansion_id(self, request: ExpansionRequest) -> str:
        """Generate unique expansion ID"""
        timestamp = int(time.time() * 1000)
        content = f"{request.direction.value}_{request.distance}_{request.method.value}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _detect_expansion_boundaries(
        self,
        request: ExpansionRequest
    ) -> List[BoundarySegment]:
        """Detect boundaries for expansion direction"""
        if self.boundary_detector is None:
            return []
        
        # Get scene preview for boundary detection
        scene_preview = self._create_scene_preview()
        
        if scene_preview is None:
            return []
        
        # Set reference image
        self.boundary_detector.set_reference_image(scene_preview)
        
        # Detect boundaries based on direction
        detection_params = {}
        
        if request.direction in [ExpansionDirection.LEFT, ExpansionDirection.RIGHT]:
            # Focus on vertical boundaries
            detection_params = {
                "method": "canny",
                "parameters": {"sigma": 2.0}
            }
        elif request.direction in [ExpansionDirection.UP, ExpansionDirection.DOWN]:
            # Focus on horizontal boundaries
            detection_params = {
                "method": "sobel",
                "parameters": {"kernel_size": 5}
            }
        else:
            # Detect all boundaries
            detection_params = {
                "method": "watershed",
                "parameters": {"compactness": 0.01}
            }
        
        # Run boundary detection
        results = self.boundary_detector.detect_boundaries(**detection_params)
        
        if "error" in results:
            logger.warning(f"Boundary detection failed: {results['error']}")
            return []
        
        # Filter boundaries by direction
        boundaries = []
        for segment in self.boundary_detector.boundary_segments.values():
            if self._is_boundary_in_direction(segment, request.direction):
                boundaries.append(segment)
        
        logger.info(f"Detected {len(boundaries)} boundaries for {request.direction.value} expansion")
        
        return boundaries
    
    def _is_boundary_in_direction(
        self,
        segment: BoundarySegment,
        direction: ExpansionDirection
    ) -> bool:
        """Check if boundary is in expansion direction"""
        if not segment.points:
            return False
        
        # Calculate boundary orientation
        points = np.array(segment.points)
        if len(points) < 2:
            return False
        
        # Calculate principal direction
        diff = points[-1] - points[0]
        angle = np.arctan2(diff[1], diff[0])
        
        # Convert to degrees
        angle_deg = np.degrees(angle)
        
        # Check if boundary aligns with expansion direction
        if direction == ExpansionDirection.RIGHT:
            # Right expansion: vertical boundaries
            return abs(angle_deg) < 30 or abs(angle_deg) > 150
        elif direction == ExpansionDirection.LEFT:
            # Left expansion: vertical boundaries
            return abs(angle_deg) < 30 or abs(angle_deg) > 150
        elif direction == ExpansionDirection.UP:
            # Up expansion: horizontal boundaries
            return 60 < abs(angle_deg) < 120
        elif direction == ExpansionDirection.DOWN:
            # Down expansion: horizontal boundaries
            return 60 < abs(angle_deg) < 120
        else:
            # All directions: include all boundaries
            return True
    
    def _create_scene_preview(self) -> Optional[np.ndarray]:
        """Create 2D preview of 3D scene for boundary detection"""
        if self.current_scene is None:
            return None
        
        # Create top-down view of scene
        preview_size = 512
        preview = np.zeros((preview_size, preview_size, 3), dtype=np.uint8)
        
        objects = self.current_scene.get("objects", {})
        
        if not objects:
            return preview
        
        # Get scene bounds
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        for obj_id, obj_data in objects.items():
            if "position" in obj_data:
                pos = obj_data["position"]
                
                # Map 3D position to 2D preview coordinates
                x_norm = (pos[0] - min_x) / (max_x - min_x)
                z_norm = (pos[2] - min_z) / (max_z - min_z)
                
                x_pix = int(x_norm * (preview_size - 1))
                y_pix = int(z_norm * (preview_size - 1))
                
                # Draw object as circle
                obj_type = obj_data.get("type", "unknown")
                if obj_type == "tree":
                    color = (0, 255, 0)  # Green for trees
                    radius = 5
                elif obj_type == "building":
                    color = (255, 0, 0)  # Red for buildings
                    radius = 8
                elif obj_type == "road":
                    color = (100, 100, 100)  # Gray for roads
                    radius = 3
                else:
                    color = (200, 200, 200)  # Gray for other objects
                    radius = 4
                
                cv2.circle(preview, (x_pix, y_pix), radius, color, -1)
        
        return preview
    
    # Expansion method implementations
    def _expand_generative(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        generate_content: bool
    ) -> Optional[ExpansionResult]:
        """
        Expand scene using generative AI
        
        Args:
            request: Expansion request
            boundaries: Detected boundaries
            generate_content: Whether to generate new content
            
        Returns:
            Expansion result
        """
        if self.current_scene is None:
            return None
        
        logger.info("Using generative expansion method")
        
        # Create expansion mask
        expansion_mask = self._create_expansion_mask(request, boundaries)
        
        if expansion_mask is None:
            return None
        
        # Generate new content
        new_objects = []
        
        if generate_content:
            new_objects = self._generate_new_content(
                request, boundaries, expansion_mask
            )
        
        # Create expanded scene
        expanded_scene = self.current_scene.copy()
        
        # Add new objects to scene
        if "objects" not in expanded_scene:
            expanded_scene["objects"] = {}
        
        for obj in new_objects:
            obj_id = obj.get("id", f"generated_{len(expanded_scene['objects'])}")
            expanded_scene["objects"][obj_id] = obj
        
        # Update scene metadata
        if "metadata" not in expanded_scene:
            expanded_scene["metadata"] = {}
        
        expanded_scene["metadata"]["expansions"] = expanded_scene["metadata"].get("expansions", []) + [{
            "method": "generative",
            "direction": request.direction.value,
            "distance": request.distance,
            "timestamp": time.time()
        }]
        
        return ExpansionResult(
            expansion_id="",  # Will be set by caller
            original_scene=self.current_scene.copy(),
            expanded_scene=expanded_scene,
            new_objects=new_objects,
            expansion_mask=expansion_mask,
            boundaries=boundaries,
            metadata={
                "method": "generative",
                "parameters": request.parameters,
                "num_generated": len(new_objects)
            }
        )
    
    def _expand_mirror(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        generate_content: bool
    ) -> Optional[ExpansionResult]:
        """
        Expand scene by mirroring existing content
        
        Args:
            request: Expansion request
            boundaries: Detected boundaries
            generate_content: Whether to generate new content
            
        Returns:
            Expansion result
        """
        if self.current_scene is None:
            return None
        
        logger.info("Using mirror expansion method")
        
        # Get objects to mirror
        objects_to_mirror = self._select_objects_for_mirroring(request)
        
        if not objects_to_mirror:
            logger.warning("No objects selected for mirroring")
            return self._expand_composition(request, boundaries, generate_content)
        
        # Create mirrored objects
        new_objects = []
        
        for obj_id, obj_data in objects_to_mirror.items():
            mirrored_obj = self._mirror_object(obj_data, request)
            if mirrored_obj:
                new_objects.append(mirrored_obj)
        
        # Create expansion mask
        expansion_mask = self._create_expansion_mask(request, boundaries)
        
        # Create expanded scene
        expanded_scene = self.current_scene.copy()
        
        # Add mirrored objects
        for obj in new_objects:
            obj_id = obj.get("id", f"mirrored_{len(expanded_scene.get('objects', {}))}")
            if "objects" not in expanded_scene:
                expanded_scene["objects"] = {}
            expanded_scene["objects"][obj_id] = obj
        
        return ExpansionResult(
            expansion_id="",  # Will be set by caller
            original_scene=self.current_scene.copy(),
            expanded_scene=expanded_scene,
            new_objects=new_objects,
            expansion_mask=expansion_mask,
            boundaries=boundaries,
            metadata={
                "method": "mirror",
                "mirrored_objects": len(objects_to_mirror),
                "parameters": request.parameters
            }
        )
    
    def _expand_tile(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        generate_content: bool
    ) -> Optional[ExpansionResult]:
        """
        Expand scene by tiling existing patterns
        
        Args:
            request: Expansion request
            boundaries: Detected boundaries
            generate_content: Whether to generate new content
            
        Returns:
            Expansion result
        """
        if self.current_scene is None:
            return None
        
        logger.info("Using tile expansion method")
        
        # Extract patterns from existing scene
        patterns = self._extract_patterns(request)
        
        if not patterns:
            logger.warning("No patterns found for tiling")
            return self._expand_composition(request, boundaries, generate_content)
        
        # Create tiled objects
        new_objects = self._tile_patterns(patterns, request)
        
        # Create expansion mask
        expansion_mask = self._create_expansion_mask(request, boundaries)
        
        # Create expanded scene
        expanded_scene = self.current_scene.copy()
        
        # Add tiled objects
        for obj in new_objects:
            obj_id = obj.get("id", f"tiled_{len(expanded_scene.get('objects', {}))}")
            if "objects" not in expanded_scene:
                expanded_scene["objects"] = {}
            expanded_scene["objects"][obj_id] = obj
        
        return ExpansionResult(
            expansion_id="",  # Will be set by caller
            original_scene=self.current_scene.copy(),
            expanded_scene=expanded_scene,
            new_objects=new_objects,
            expansion_mask=expansion_mask,
            boundaries=boundaries,
            metadata={
                "method": "tile",
                "patterns_found": len(patterns),
                "tiles_created": len(new_objects),
                "parameters": request.parameters
            }
        )
    
    def _expand_extrapolate(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        generate_content: bool
    ) -> Optional[ExpansionResult]:
        """
        Expand scene by extrapolating from existing content
        
        Args:
            request: Expansion request
            boundaries: Detected boundaries
            generate_content: Whether to generate new content
            
        Returns:
            Expansion result
        """
        if self.current_scene is None:
            return None
        
        logger.info("Using extrapolation expansion method")
        
        # Analyze scene structure
        scene_structure = self._analyze_scene_structure()
        
        # Extrapolate new content
        new_objects = self._extrapolate_content(scene_structure, request)
        
        # Create expansion mask
        expansion_mask = self._create_expansion_mask(request, boundaries)
        
        # Create expanded scene
        expanded_scene = self.current_scene.copy()
        
        # Add extrapolated objects
        for obj in new_objects:
            obj_id = obj.get("id", f"extrapolated_{len(expanded_scene.get('objects', {}))}")
            if "objects" not in expanded_scene:
                expanded_scene["objects"] = {}
            expanded_scene["objects"][obj_id] = obj
        
        return ExpansionResult(
            expansion_id="",  # Will be set by caller
            original_scene=self.current_scene.copy(),
            expanded_scene=expanded_scene,
            new_objects=new_objects,
            expansion_mask=expansion_mask,
            boundaries=boundaries,
            metadata={
                "method": "extrapolate",
                "structure_analyzed": bool(scene_structure),
                "extrapolated_objects": len(new_objects),
                "parameters": request.parameters
            }
        )
    
    def _expand_blend(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        generate_content: bool
    ) -> Optional[ExpansionResult]:
        """
        Expand scene by blending with new content
        
        Args:
            request: Expansion request
            boundaries: Detected boundaries
            generate_content: Whether to generate new content
            
        Returns:
            Expansion result
        """
        if self.current_scene is None:
            return None
        
        logger.info("Using blend expansion method")
        
        # Get blending parameters
        blend_strength = request.parameters.get("blend_strength", 0.5)
        source_scene = request.parameters.get("source_scene")
        
        # If no source scene, use generative method
        if source_scene is None:
            return self._expand_generative(request, boundaries, generate_content)
        
        # Blend scenes
        blended_objects = self._blend_scenes(
            self.current_scene, 
            source_scene, 
            blend_strength,
            request
        )
        
        # Create expansion mask
        expansion_mask = self._create_expansion_mask(request, boundaries)
        
        # Create expanded scene (blended with current)
        expanded_scene = self.current_scene.copy()
        
        # Add blended objects
        for obj in blended_objects:
            obj_id = obj.get("id", f"blended_{len(expanded_scene.get('objects', {}))}")
            if "objects" not in expanded_scene:
                expanded_scene["objects"] = {}
            expanded_scene["objects"][obj_id] = obj
        
        return ExpansionResult(
            expansion_id="",  # Will be set by caller
            original_scene=self.current_scene.copy(),
            expanded_scene=expanded_scene,
            new_objects=blended_objects,
            expansion_mask=expansion_mask,
            boundaries=boundaries,
            metadata={
                "method": "blend",
                "blend_strength": blend_strength,
                "blended_objects": len(blended_objects),
                "parameters": request.parameters
            }
        )
    
    def _expand_composition(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        generate_content: bool
    ) -> Optional[ExpansionResult]:
        """
        Expand scene by composing from existing objects
        
        Args:
            request: Expansion request
            boundaries: Detected boundaries
            generate_content: Whether to generate new content
            
        Returns:
            Expansion result
        """
        if self.current_scene is None:
            return None
        
        logger.info("Using composition expansion method")
        
        # Use scene composer if available
        if self.scene_composer is not None:
            return self._expand_with_composer(request, boundaries, generate_content)
        
        # Fallback to simple composition
        new_objects = self._compose_from_existing(request)
        
        # Create expansion mask
        expansion_mask = self._create_expansion_mask(request, boundaries)
        
        # Create expanded scene
        expanded_scene = self.current_scene.copy()
        
        # Add composed objects
        for obj in new_objects:
            obj_id = obj.get("id", f"composed_{len(expanded_scene.get('objects', {}))}")
            if "objects" not in expanded_scene:
                expanded_scene["objects"] = {}
            expanded_scene["objects"][obj_id] = obj
        
        return ExpansionResult(
            expansion_id="",  # Will be set by caller
            original_scene=self.current_scene.copy(),
            expanded_scene=expanded_scene,
            new_objects=new_objects,
            expansion_mask=expansion_mask,
            boundaries=boundaries,
            metadata={
                "method": "composition",
                "composed_objects": len(new_objects),
                "parameters": request.parameters
            }
        )
    
    def _expand_with_composer(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        generate_content: bool
    ) -> Optional[ExpansionResult]:
        """Expand scene using scene composer"""
        if self.scene_composer is None or self.current_scene is None:
            return None
        
        # Extract objects from current scene
        scene_objects = []
        objects_data = self.current_scene.get("objects", {})
        
        for obj_id, obj_data in objects_data.items():
            # Convert to SceneObject (simplified)
            # In production, would need proper conversion
            mesh = trimesh.creation.box([1, 1, 1])  # Placeholder
            
            scene_obj = SceneObject(
                id=obj_id,
                mesh=mesh,
                position=np.array(obj_data.get("position", [0, 0, 0])),
                rotation=np.array(obj_data.get("rotation", np.eye(3))),
                scale=np.array(obj_data.get("scale", [1, 1, 1])),
                semantic_label=obj_data.get("type", "unknown")
            )
            scene_objects.append(scene_obj)
        
        # Create composition constraints based on expansion direction
        constraints = self._create_composition_constraints(request)
        
        # Calculate new bounds for composition
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        if request.direction == ExpansionDirection.RIGHT:
            new_bounds = (max_x, max_x + request.distance, min_z, max_z)
        elif request.direction == ExpansionDirection.LEFT:
            new_bounds = (min_x - request.distance, min_x, min_z, max_z)
        elif request.direction == ExpansionDirection.UP:
            new_bounds = (min_x, max_x, max_z, max_z + request.distance)
        elif request.direction == ExpansionDirection.DOWN:
            new_bounds = (min_x, max_x, min_z - request.distance, min_z)
        else:
            # Expand in all directions
            padding = request.distance / 2
            new_bounds = (
                min_x - padding, max_x + padding,
                min_z - padding, max_z + padding
            )
        
        # Compose new scene
        composition_result = self.scene_composer.compose_scene(
            objects=scene_objects,
            constraints=constraints,
            bounds=new_bounds
        )
        
        # Convert composed objects back to scene format
        new_objects = []
        for obj in composition_result.get("objects", []):
            obj_dict = {
                "id": obj.id,
                "type": obj.semantic_label,
                "position": obj.position.tolist(),
                "rotation": obj.rotation.tolist(),
                "scale": obj.scale.tolist(),
                "metadata": obj.metadata
            }
            new_objects.append(obj_dict)
        
        # Create expansion mask
        expansion_mask = self._create_expansion_mask(request, boundaries)
        
        # Create expanded scene
        expanded_scene = self.current_scene.copy()
        
        # Add composed objects
        for obj in new_objects:
            expanded_scene["objects"][obj["id"]] = obj
        
        return ExpansionResult(
            expansion_id="",  # Will be set by caller
            original_scene=self.current_scene.copy(),
            expanded_scene=expanded_scene,
            new_objects=new_objects,
            expansion_mask=expansion_mask,
            boundaries=boundaries,
            metadata={
                "method": "composition",
                "composer_used": True,
                "composition_stats": composition_result.get("stats", {}),
                "parameters": request.parameters
            }
        )
    
    def _create_expansion_mask(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment]
    ) -> Optional[np.ndarray]:
        """Create mask showing expansion area"""
        # Create 2D representation of expansion area
        mask_size = 256
        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        
        # Get scene bounds
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        # Calculate expansion area
        if request.direction == ExpansionDirection.RIGHT:
            # Right expansion
            expansion_min_x = max_x
            expansion_max_x = max_x + request.distance
            expansion_min_z = min_z
            expansion_max_z = max_z
        elif request.direction == ExpansionDirection.LEFT:
            # Left expansion
            expansion_min_x = min_x - request.distance
            expansion_max_x = min_x
            expansion_min_z = min_z
            expansion_max_z = max_z
        elif request.direction == ExpansionDirection.UP:
            # Up expansion (forward in Z)
            expansion_min_x = min_x
            expansion_max_x = max_x
            expansion_min_z = max_z
            expansion_max_z = max_z + request.distance
        elif request.direction == ExpansionDirection.DOWN:
            # Down expansion (backward in Z)
            expansion_min_x = min_x
            expansion_max_x = max_x
            expansion_min_z = min_z - request.distance
            expansion_max_z = min_z
        else:
            # All directions
            padding = request.distance / 2
            expansion_min_x = min_x - padding
            expansion_max_x = max_x + padding
            expansion_min_z = min_z - padding
            expansion_max_z = max_z + padding
        
        # Convert to mask coordinates
        def to_mask_coords(x, z):
            x_norm = (x - min_x) / (max_x - min_x)
            z_norm = (z - min_z) / (max_z - min_z)
            x_pix = int(x_norm * (mask_size - 1))
            y_pix = int(z_norm * (mask_size - 1))
            return x_pix, y_pix
        
        # Draw expansion area
        x1, y1 = to_mask_coords(expansion_min_x, expansion_min_z)
        x2, y2 = to_mask_coords(expansion_max_x, expansion_max_z)
        
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Draw boundaries if available
        for boundary in boundaries:
            if len(boundary.points) >= 2:
                points = []
                for point in boundary.points:
                    # Convert boundary points (in image coordinates) to mask coordinates
                    # This is simplified - in production would need proper mapping
                    bx = int(point[0] * mask_size / 512)  # Assuming 512x512 boundary detection
                    by = int(point[1] * mask_size / 512)
                    points.append((bx, by))
                
                if len(points) >= 2:
                    pts = np.array(points, dtype=np.int32)
                    cv2.polylines(mask, [pts], False, 128, 2)
        
        return mask
    
    def _select_objects_for_mirroring(
        self,
        request: ExpansionRequest
    ) -> Dict[str, Any]:
        """Select objects to mirror based on expansion direction"""
        if self.current_scene is None:
            return {}
        
        objects = self.current_scene.get("objects", {})
        
        if not objects:
            return {}
        
        # Select objects near the boundary
        selected = {}
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        for obj_id, obj_data in objects.items():
            if "position" not in obj_data:
                continue
            
            pos = obj_data["position"]
            x, z = pos[0], pos[2]
            
            # Check if object is near boundary in expansion direction
            if request.direction == ExpansionDirection.RIGHT:
                if abs(x - max_x) < 5.0:  # Near right boundary
                    selected[obj_id] = obj_data
            elif request.direction == ExpansionDirection.LEFT:
                if abs(x - min_x) < 5.0:  # Near left boundary
                    selected[obj_id] = obj_data
            elif request.direction == ExpansionDirection.UP:
                if abs(z - max_z) < 5.0:  # Near top boundary
                    selected[obj_id] = obj_data
            elif request.direction == ExpansionDirection.DOWN:
                if abs(z - min_z) < 5.0:  # Near bottom boundary
                    selected[obj_id] = obj_data
            else:
                # For all directions, select random objects
                if np.random.random() < 0.3:  # 30% chance
                    selected[obj_id] = obj_data
        
        return selected
    
    def _mirror_object(
        self,
        obj_data: Dict[str, Any],
        request: ExpansionRequest
    ) -> Optional[Dict[str, Any]]:
        """Create mirrored version of object"""
        if "position" not in obj_data:
            return None
        
        mirrored = obj_data.copy()
        
        # Generate new ID
        mirrored["id"] = f"{obj_data.get('id', 'object')}_mirrored"
        
        # Mirror position
        pos = np.array(obj_data["position"])
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        if request.direction == ExpansionDirection.RIGHT:
            # Mirror across right boundary
            mirrored_x = max_x + (max_x - pos[0])
            pos[0] = mirrored_x
        elif request.direction == ExpansionDirection.LEFT:
            # Mirror across left boundary
            mirrored_x = min_x - (pos[0] - min_x)
            pos[0] = mirrored_x
        elif request.direction == ExpansionDirection.UP:
            # Mirror across top boundary
            mirrored_z = max_z + (max_z - pos[2])
            pos[2] = mirrored_z
        elif request.direction == ExpansionDirection.DOWN:
            # Mirror across bottom boundary
            mirrored_z = min_z - (pos[2] - min_z)
            pos[2] = mirrored_z
        
        mirrored["position"] = pos.tolist()
        
        # Add variance to avoid exact copies
        if "scale" in mirrored:
            scale = np.array(mirrored["scale"])
            variance = np.random.uniform(0.8, 1.2, 3)
            mirrored["scale"] = (scale * variance).tolist()
        
        # Randomize color slightly
        if "color" in mirrored:
            color = np.array(mirrored["color"])
            variance = np.random.uniform(0.9, 1.1, 3)
            mirrored["color"] = np.clip(color * variance, 0, 255).astype(int).tolist()
        
        # Add metadata
        mirrored["metadata"] = mirrored.get("metadata", {})
        mirrored["metadata"]["mirrored_from"] = obj_data.get("id", "unknown")
        mirrored["metadata"]["mirror_direction"] = request.direction.value
        
        return mirrored
    
    def _extract_patterns(
        self,
        request: ExpansionRequest
    ) -> List[Dict[str, Any]]:
        """Extract repeating patterns from scene"""
        if self.current_scene is None:
            return []
        
        objects = self.current_scene.get("objects", {})
        
        if not objects:
            return []
        
        # Group objects by type and position patterns
        patterns = []
        
        # Simple pattern extraction: find clusters of similar objects
        obj_positions = []
        obj_types = []
        
        for obj_id, obj_data in objects.items():
            if "position" in obj_data and "type" in obj_data:
                pos = obj_data["position"]
                obj_positions.append([pos[0], pos[2]])  # X, Z coordinates
                obj_types.append(obj_data["type"])
        
        if len(obj_positions) < 2:
            return []
        
        # Use KDTree to find spatial patterns
        positions_array = np.array(obj_positions)
        tree = KDTree(positions_array)
        
        # Find neighbors within pattern radius
        pattern_radius = request.parameters.get("pattern_radius", 5.0)
        
        for i, pos in enumerate(positions_array):
            neighbors = tree.query_ball_point(pos, pattern_radius)
            
            if len(neighbors) >= 3:  # Minimum for a pattern
                pattern = {
                    "center": pos.tolist(),
                    "object_type": obj_types[i],
                    "neighbors": len(neighbors),
                    "positions": positions_array[neighbors].tolist(),
                    "types": [obj_types[j] for j in neighbors]
                }
                patterns.append(pattern)
        
        # Remove duplicate patterns
        unique_patterns = []
        pattern_centers = []
        
        for pattern in patterns:
            center = tuple(pattern["center"])
            if center not in pattern_centers:
                pattern_centers.append(center)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _tile_patterns(
        self,
        patterns: List[Dict[str, Any]],
        request: ExpansionRequest
    ) -> List[Dict[str, Any]]:
        """Tile patterns in expansion area"""
        if not patterns:
            return []
        
        new_objects = []
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        # Calculate tiling parameters
        tile_spacing = request.parameters.get("tile_spacing", 10.0)
        num_tiles = int(request.distance / tile_spacing)
        
        # Select patterns to tile
        patterns_to_tile = patterns[:3]  # Use first 3 patterns
        
        for pattern in patterns_to_tile:
            pattern_center = np.array(pattern["center"])
            pattern_positions = np.array(pattern["positions"])
            
            # Calculate offset from center for each position
            offsets = pattern_positions - pattern_center
            
            # Tile in expansion direction
            for tile_idx in range(1, num_tiles + 1):
                # Calculate tile center
                if request.direction == ExpansionDirection.RIGHT:
                    tile_center = pattern_center + [tile_idx * tile_spacing, 0]
                elif request.direction == ExpansionDirection.LEFT:
                    tile_center = pattern_center + [-tile_idx * tile_spacing, 0]
                elif request.direction == ExpansionDirection.UP:
                    tile_center = pattern_center + [0, tile_idx * tile_spacing]
                elif request.direction == ExpansionDirection.DOWN:
                    tile_center = pattern_center + [0, -tile_idx * tile_spacing]
                else:
                    # Tile in random direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    tile_center = pattern_center + [
                        np.cos(angle) * tile_idx * tile_spacing,
                        np.sin(angle) * tile_idx * tile_spacing
                    ]
                
                # Create objects for this tile
                for offset, obj_type in zip(offsets, pattern["types"]):
                    obj_pos = tile_center + offset
                    
                    # Create object
                    obj = {
                        "id": f"tiled_{obj_type}_{len(new_objects)}",
                        "type": obj_type,
                        "position": [float(obj_pos[0]), 0.0, float(obj_pos[1])],
                        "scale": [1.0, 1.0, 1.0],
                        "metadata": {
                            "pattern_index": len(new_objects),
                            "tile_index": tile_idx,
                            "source_pattern": pattern["object_type"]
                        }
                    }
                    
                    new_objects.append(obj)
        
        return new_objects
    
    def _analyze_scene_structure(self) -> Dict[str, Any]:
        """Analyze scene structure for extrapolation"""
        if self.current_scene is None:
            return {}
        
        objects = self.current_scene.get("objects", {})
        
        if not objects:
            return {}
        
        # Analyze object distribution
        positions = []
        types = []
        scales = []
        
        for obj_data in objects.values():
            if "position" in obj_data:
                pos = obj_data["position"]
                positions.append([pos[0], pos[2]])  # X, Z
                
                if "type" in obj_data:
                    types.append(obj_data["type"])
                
                if "scale" in obj_data:
                    scales.append(obj_data["scale"])
        
        if not positions:
            return {}
        
        positions_array = np.array(positions)
        
        # Calculate density map
        min_x, max_x, min_z, max_z = self.scene_bounds
        grid_size = 20
        
        density = np.zeros((grid_size, grid_size))
        
        for pos in positions_array:
            x_idx = int((pos[0] - min_x) / (max_x - min_x) * (grid_size - 1))
            z_idx = int((pos[1] - min_z) / (max_z - min_z) * (grid_size - 1))
            
            if 0 <= x_idx < grid_size and 0 <= z_idx < grid_size:
                density[z_idx, x_idx] += 1
        
        # Calculate type frequencies
        type_counts = {}
        for obj_type in types:
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        # Calculate average scale per type
        type_scales = defaultdict(list)
        for obj_type, scale in zip(types, scales):
            if scale:
                type_scales[obj_type].append(np.mean(scale))
        
        avg_scales = {}
        for obj_type, scale_list in type_scales.items():
            if scale_list:
                avg_scales[obj_type] = np.mean(scale_list)
        
        return {
            "object_count": len(objects),
            "type_distribution": dict(type_counts),
            "density_map": density.tolist(),
            "average_scales": avg_scales,
            "bounds": self.scene_bounds,
            "position_range": {
                "min_x": float(np.min(positions_array[:, 0])),
                "max_x": float(np.max(positions_array[:, 0])),
                "min_z": float(np.min(positions_array[:, 1])),
                "max_z": float(np.max(positions_array[:, 1]))
            }
        }
    
    def _extrapolate_content(
        self,
        scene_structure: Dict[str, Any],
        request: ExpansionRequest
    ) -> List[Dict[str, Any]]:
        """Extrapolate new content based on scene structure"""
        if not scene_structure:
            return []
        
        new_objects = []
        
        # Get structure parameters
        type_dist = scene_structure.get("type_distribution", {})
        density_map = scene_structure.get("density_map", [])
        avg_scales = scene_structure.get("average_scales", {})
        pos_range = scene_structure.get("position_range", {})
        
        if not type_dist:
            return []
        
        # Calculate number of new objects based on density
        total_objects = scene_structure.get("object_count", 0)
        expansion_factor = request.parameters.get("expansion_factor", 0.3)
        num_new_objects = int(total_objects * expansion_factor)
        
        # Generate objects
        for i in range(num_new_objects):
            # Select object type based on distribution
            types = list(type_dist.keys())
            weights = list(type_dist.values())
            weights_sum = sum(weights)
            
            if weights_sum > 0:
                normalized_weights = [w / weights_sum for w in weights]
                obj_type = np.random.choice(types, p=normalized_weights)
            else:
                obj_type = "unknown"
            
            # Generate position in expansion area
            min_x, max_x, min_z, max_z = self.scene_bounds
            
            if request.direction == ExpansionDirection.RIGHT:
                x = np.random.uniform(max_x, max_x + request.distance)
                z = np.random.uniform(min_z, max_z)
            elif request.direction == ExpansionDirection.LEFT:
                x = np.random.uniform(min_x - request.distance, min_x)
                z = np.random.uniform(min_z, max_z)
            elif request.direction == ExpansionDirection.UP:
                x = np.random.uniform(min_x, max_x)
                z = np.random.uniform(max_z, max_z + request.distance)
            elif request.direction == ExpansionDirection.DOWN:
                x = np.random.uniform(min_x, max_x)
                z = np.random.uniform(min_z - request.distance, min_z)
            else:
                # Random in expanded area
                padding = request.distance / 2
                x = np.random.uniform(min_x - padding, max_x + padding)
                z = np.random.uniform(min_z - padding, max_z + padding)
            
            # Get scale
            if obj_type in avg_scales:
                base_scale = avg_scales[obj_type]
                scale_variance = np.random.uniform(0.8, 1.2)
                scale = [base_scale * scale_variance] * 3
            else:
                scale = [1.0, 1.0, 1.0]
            
            # Create object
            obj = {
                "id": f"extrapolated_{obj_type}_{i}",
                "type": obj_type,
                "position": [float(x), 0.0, float(z)],
                "scale": scale,
                "metadata": {
                    "generation_method": "extrapolation",
                    "source_type_distribution": type_dist.get(obj_type, 0)
                }
            }
            
            new_objects.append(obj)
        
        return new_objects
    
    def _blend_scenes(
        self,
        scene1: Dict[str, Any],
        scene2: Dict[str, Any],
        blend_strength: float,
        request: ExpansionRequest
    ) -> List[Dict[str, Any]]:
        """Blend two scenes together"""
        blended_objects = []
        
        # Get objects from both scenes
        objects1 = scene1.get("objects", {})
        objects2 = scene2.get("objects", {})
        
        # Take objects from scene2 and adjust based on blend strength
        for obj_id, obj_data in objects2.items():
            blended_obj = obj_data.copy()
            
            # Adjust position based on expansion direction
            if "position" in blended_obj:
                pos = np.array(blended_obj["position"])
                
                if request.direction == ExpansionDirection.RIGHT:
                    pos[0] += request.distance * blend_strength
                elif request.direction == ExpansionDirection.LEFT:
                    pos[0] -= request.distance * blend_strength
                elif request.direction == ExpansionDirection.UP:
                    pos[2] += request.distance * blend_strength
                elif request.direction == ExpansionDirection.DOWN:
                    pos[2] -= request.distance * blend_strength
                
                blended_obj["position"] = pos.tolist()
            
            # Adjust scale based on blend strength
            if "scale" in blended_obj:
                scale = np.array(blended_obj["scale"])
                blended_scale = scale * (0.5 + blend_strength * 0.5)
                blended_obj["scale"] = blended_scale.tolist()
            
            # Generate new ID
            blended_obj["id"] = f"blended_{obj_id}_{len(blended_objects)}"
            
            # Add metadata
            blended_obj["metadata"] = blended_obj.get("metadata", {})
            blended_obj["metadata"]["blend_strength"] = blend_strength
            blended_obj["metadata"]["source_scene"] = "scene2"
            
            blended_objects.append(blended_obj)
        
        return blended_objects
    
    def _compose_from_existing(
        self,
        request: ExpansionRequest
    ) -> List[Dict[str, Any]]:
        """Compose new objects from existing ones"""
        if self.current_scene is None:
            return []
        
        objects = self.current_scene.get("objects", {})
        
        if not objects:
            return []
        
        new_objects = []
        
        # Select template objects
        template_objects = list(objects.values())[:5]  # Use first 5 as templates
        
        # Number of new objects to create
        num_new = request.parameters.get("num_new_objects", 10)
        
        for i in range(num_new):
            # Select random template
            template = np.random.choice(template_objects)
            
            # Create variant
            variant = template.copy()
            
            # Generate new ID
            variant["id"] = f"composed_{template.get('type', 'object')}_{i}"
            
            # Adjust position for expansion direction
            if "position" in variant:
                pos = np.array(variant["position"])
                min_x, max_x, min_z, max_z = self.scene_bounds
                
                if request.direction == ExpansionDirection.RIGHT:
                    pos[0] = np.random.uniform(max_x, max_x + request.distance)
                    pos[2] = np.random.uniform(min_z, max_z)
                elif request.direction == ExpansionDirection.LEFT:
                    pos[0] = np.random.uniform(min_x - request.distance, min_x)
                    pos[2] = np.random.uniform(min_z, max_z)
                elif request.direction == ExpansionDirection.UP:
                    pos[0] = np.random.uniform(min_x, max_x)
                    pos[2] = np.random.uniform(max_z, max_z + request.distance)
                elif request.direction == ExpansionDirection.DOWN:
                    pos[0] = np.random.uniform(min_x, max_x)
                    pos[2] = np.random.uniform(min_z - request.distance, min_z)
                
                variant["position"] = pos.tolist()
            
            # Add random variations
            if "scale" in variant:
                scale = np.array(variant["scale"])
                variance = np.random.uniform(0.7, 1.3, 3)
                variant["scale"] = (scale * variance).tolist()
            
            if "color" in variant:
                color = np.array(variant["color"])
                variance = np.random.uniform(0.8, 1.2, 3)
                variant["color"] = np.clip(color * variance, 0, 255).astype(int).tolist()
            
            # Add metadata
            variant["metadata"] = variant.get("metadata", {})
            variant["metadata"]["composed_from"] = template.get("id", "unknown")
            variant["metadata"]["variant_index"] = i
            
            new_objects.append(variant)
        
        return new_objects
    
    def _generate_new_content(
        self,
        request: ExpansionRequest,
        boundaries: List[BoundarySegment],
        expansion_mask: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Generate new content using AI models"""
        # This is a placeholder for AI content generation
        # In production, would use actual AI models
        
        logger.info("Generating new content (placeholder implementation)")
        
        new_objects = []
        
        # Simple procedural generation based on request
        num_objects = request.parameters.get("num_objects", 15)
        
        for i in range(num_objects):
            obj_type = self._select_object_type(request)
            position = self._generate_position(request, i, num_objects)
            scale = self._generate_scale(obj_type)
            color = self._generate_color(obj_type)
            
            obj = {
                "id": f"generated_{obj_type}_{i}",
                "type": obj_type,
                "position": position,
                "scale": scale,
                "color": color,
                "metadata": {
                    "generation_method": "AI",
                    "request_id": request.hash,
                    "boundary_count": len(boundaries)
                }
            }
            
            new_objects.append(obj)
        
        return new_objects
    
    def _select_object_type(self, request: ExpansionRequest) -> str:
        """Select object type based on expansion parameters"""
        # Object type probabilities
        type_probs = {
            "tree": 0.3,
            "bush": 0.2,
            "rock": 0.15,
            "flower": 0.1,
            "grass": 0.1,
            "building": 0.05,
            "road": 0.05,
            "water": 0.05
        }
        
        # Adjust based on direction
        if request.direction in [ExpansionDirection.UP, ExpansionDirection.DOWN]:
            # More natural objects for vertical expansion
            type_probs["tree"] = 0.4
            type_probs["building"] = 0.02
        elif request.direction in [ExpansionDirection.LEFT, ExpansionDirection.RIGHT]:
            # More structural objects for horizontal expansion
            type_probs["building"] = 0.1
            type_probs["road"] = 0.1
        
        # Select type
        types = list(type_probs.keys())
        probs = list(type_probs.values())
        return np.random.choice(types, p=probs)
    
    def _generate_position(
        self,
        request: ExpansionRequest,
        index: int,
        total: int
    ) -> List[float]:
        """Generate position for new object"""
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        if request.direction == ExpansionDirection.RIGHT:
            x = np.random.uniform(max_x, max_x + request.distance)
            z = np.random.uniform(min_z, max_z)
        elif request.direction == ExpansionDirection.LEFT:
            x = np.random.uniform(min_x - request.distance, min_x)
            z = np.random.uniform(min_z, max_z)
        elif request.direction == ExpansionDirection.UP:
            x = np.random.uniform(min_x, max_x)
            z = np.random.uniform(max_z, max_z + request.distance)
        elif request.direction == ExpansionDirection.DOWN:
            x = np.random.uniform(min_x, max_x)
            z = np.random.uniform(min_z - request.distance, min_z)
        else:
            # All directions - use grid pattern
            grid_size = int(np.sqrt(total))
            row = index // grid_size
            col = index % grid_size
            
            padding = request.distance / 2
            cell_width = (max_x - min_x + 2 * padding) / grid_size
            cell_height = (max_z - min_z + 2 * padding) / grid_size
            
            x = min_x - padding + (col + 0.5) * cell_width
            z = min_z - padding + (row + 0.5) * cell_height
        
        y = 0.0  # Ground level
        
        # Add some randomness
        x += np.random.uniform(-1.0, 1.0)
        z += np.random.uniform(-1.0, 1.0)
        
        return [float(x), float(y), float(z)]
    
    def _generate_scale(self, obj_type: str) -> List[float]:
        """Generate scale for object based on type"""
        base_scales = {
            "tree": [2.0, 3.0, 2.0],
            "bush": [1.0, 0.5, 1.0],
            "rock": [0.5, 0.3, 0.5],
            "flower": [0.2, 0.3, 0.2],
            "grass": [0.1, 0.2, 0.1],
            "building": [5.0, 10.0, 5.0],
            "road": [3.0, 0.1, 10.0],
            "water": [10.0, 0.1, 10.0]
        }
        
        base = base_scales.get(obj_type, [1.0, 1.0, 1.0])
        variance = np.random.uniform(0.8, 1.2, 3)
        
        return (np.array(base) * variance).tolist()
    
    def _generate_color(self, obj_type: str) -> List[int]:
        """Generate color for object based on type"""
        base_colors = {
            "tree": [34, 139, 34],      # Forest green
            "bush": [0, 100, 0],        # Dark green
            "rock": [128, 128, 128],    # Gray
            "flower": [255, 105, 180],  # Pink
            "grass": [124, 252, 0],     # Lawn green
            "building": [192, 192, 192], # Silver
            "road": [105, 105, 105],    # Dim gray
            "water": [30, 144, 255]     # Dodger blue
        }
        
        base = base_colors.get(obj_type, [200, 200, 200])
        variance = np.random.uniform(0.9, 1.1, 3)
        
        return np.clip((np.array(base) * variance), 0, 255).astype(int).tolist()
    
    def _create_composition_constraints(
        self,
        request: ExpansionRequest
    ) -> Dict[str, Any]:
        """Create composition constraints for scene composer"""
        constraints = {
            "spatial_constraints": {},
            "semantic_constraints": {},
            "density_constraints": {},
            "symmetry_constraints": {},
            "adjacency_constraints": {},
            "exclusion_zones": []
        }
        
        # Add direction-specific constraints
        if request.direction == ExpansionDirection.RIGHT:
            constraints["spatial_constraints"]["all"] = [
                (self.scene_bounds[1], self.scene_bounds[1] + request.distance),
                (0, 100),  # Y range
                (self.scene_bounds[2], self.scene_bounds[3])  # Z range
            ]
        elif request.direction == ExpansionDirection.LEFT:
            constraints["spatial_constraints"]["all"] = [
                (self.scene_bounds[0] - request.distance, self.scene_bounds[0]),
                (0, 100),
                (self.scene_bounds[2], self.scene_bounds[3])
            ]
        
        # Add density constraint
        constraints["density_constraints"] = {
            "tree": 0.1,
            "building": 0.05,
            "road": 0.02
        }
        
        return constraints
    
    def _update_scene_bounds(
        self,
        scene: Dict[str, Any],
        request: ExpansionRequest
    ) -> Tuple[float, float, float, float]:
        """Update scene bounds after expansion"""
        min_x, max_x, min_z, max_z = self.scene_bounds
        
        # Calculate bounds from objects
        objects = scene.get("objects", {})
        
        if objects:
            obj_min_x = min_x
            obj_max_x = max_x
            obj_min_z = min_z
            obj_max_z = max_z
            
            for obj_data in objects.values():
                if "position" in obj_data:
                    pos = obj_data["position"]
                    obj_min_x = min(obj_min_x, pos[0])
                    obj_max_x = max(obj_max_x, pos[0])
                    obj_min_z = min(obj_min_z, pos[2])
                    obj_max_z = max(obj_max_z, pos[2])
            
            # Add padding
            padding = 5.0
            return (
                obj_min_x - padding,
                obj_max_x + padding,
                obj_min_z - padding,
                obj_max_z + padding
            )
        
        # If no objects, expand based on request
        if request.direction == ExpansionDirection.RIGHT:
            return (min_x, max_x + request.distance, min_z, max_z)
        elif request.direction == ExpansionDirection.LEFT:
            return (min_x - request.distance, max_x, min_z, max_z)
        elif request.direction == ExpansionDirection.UP:
            return (min_x, max_x, min_z, max_z + request.distance)
        elif request.direction == ExpansionDirection.DOWN:
            return (min_x, max_x, min_z - request.distance, max_z)
        else:
            padding = request.distance / 2
            return (
                min_x - padding,
                max_x + padding,
                min_z - padding,
                max_z + padding
            )
    
    def expand_multiple(
        self,
        requests: List[Union[ExpansionRequest, Dict[str, Any]]],
        sequential: bool = True
    ) -> List[ExpansionResult]:
        """
        Expand scene multiple times
        
        Args:
            requests: List of expansion requests
            sequential: Whether to expand sequentially
            
        Returns:
            List of expansion results
        """
        results = []
        
        if sequential:
            # Expand sequentially, each expansion builds on previous
            for request in requests:
                result = self.expand_scene(request)
                if result:
                    results.append(result)
        else:
            # Expand in parallel from original scene
            futures = {}
            
            for i, request in enumerate(requests):
                # Save current scene
                original_scene = self.current_scene.copy()
                
                # Submit expansion task
                future = self.executor.submit(
                    self._expand_from_scene,
                    original_scene,
                    request,
                    i
                )
                futures[future] = i
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Parallel expansion failed: {e}")
            
            # Sort by request order
            results.sort(key=lambda x: x.metadata.get("request_index", 0))
        
        return results
    
    def _expand_from_scene(
        self,
        scene: Dict[str, Any],
        request: Union[ExpansionRequest, Dict[str, Any]],
        request_index: int
    ) -> Optional[ExpansionResult]:
        """Expand from a specific scene (for parallel processing)"""
        # Create temporary expander
        temp_expander = SceneExpander(
            config=self.config,
            boundary_detector=self.boundary_detector,
            scene_composer=self.scene_composer,
            max_workers=1
        )
        
        # Load scene
        temp_expander.load_scene(scene, self.scene_bounds)
        
        # Expand
        result = temp_expander.expand_scene(request)
        
        if result:
            result.metadata["request_index"] = request_index
        
        return result
    
    def get_expansion_history(self) -> List[Dict[str, Any]]:
        """Get expansion history"""
        history = []
        
        for expansion_id, result in self.expansion_history.items():
            history.append({
                "id": expansion_id,
                "method": result.metadata.get("method", "unknown"),
                "direction": result.metadata.get("direction", "unknown"),
                "timestamp": result.metadata.get("timestamp", time.time()),
                "new_objects": result.num_new_objects,
                "expansion_area": result.expansion_area
            })
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return history
    
    def export_expansion_result(
        self,
        result: ExpansionResult,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """
        Export expansion result to file
        
        Args:
            result: Expansion result
            output_path: Output file path
            format: Export format
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            
            if format == "json":
                data = {
                    "expansion_id": result.expansion_id,
                    "metadata": result.metadata,
                    "original_scene": result.original_scene,
                    "expanded_scene": result.expanded_scene,
                    "new_objects": result.new_objects,
                    "statistics": {
                        "num_new_objects": result.num_new_objects,
                        "expansion_area": result.expansion_area,
                        "boundary_count": len(result.boundaries)
                    },
                    "exported": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                save_json(data, output_path.with_suffix('.json'))
                
            elif format == "scene":
                # Export only expanded scene
                save_json(result.expanded_scene, output_path.with_suffix('.json'))
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported expansion result to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting expansion result: {e}")
            return False
    
    def clear_expansions(self) -> None:
        """Clear all expansions and reset to original scene"""
        if self.expansion_history:
            # Restore first original scene if available
            first_result = list(self.expansion_history.values())[0]
            if first_result:
                self.current_scene = first_result.original_scene.copy()
            
            self.expansion_history.clear()
            self.expansion_queue.clear()
            
            logger.info("Cleared all expansions")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("SceneExpander cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        if self.current_scene is None:
            return "SceneExpander(no scene loaded)"
        
        num_objects = len(self.current_scene.get("objects", {}))
        num_expansions = len(self.expansion_history)
        
        return (f"SceneExpander(objects={num_objects}, "
                f"expansions={num_expansions}, "
                f"bounds={self.scene_bounds})")


# Factory function for creating scene expanders
def create_scene_expander(
    config: Optional[Dict[str, Any]] = None,
    boundary_detector: Optional[BoundaryDetector] = None,
    scene_composer: Optional[SceneComposer] = None,
    max_workers: int = 4
) -> SceneExpander:
    """
    Factory function to create scene expanders
    
    Args:
        config: Configuration dictionary
        boundary_detector: Optional boundary detector
        scene_composer: Optional scene composer
        max_workers: Maximum worker threads
        
    Returns:
        SceneExpander instance
    """
    return SceneExpander(
        config=config,
        boundary_detector=boundary_detector,
        scene_composer=scene_composer,
        max_workers=max_workers
    )