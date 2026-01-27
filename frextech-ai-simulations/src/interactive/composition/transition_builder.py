"""
Transition Builder Module
Creates smooth transitions between scenes and world states
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from ...utils.metrics import Timer, PerformanceMetrics
from ...utils.file_io import save_json, load_json, save_pickle, load_pickle

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Types of transitions"""
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    SLIDE = "slide"
    ZOOM = "zoom"
    ROTATE = "rotate"
    MORPH = "morph"
    SEAMLESS = "seamless"
    CUSTOM = "custom"


class TransitionCurve(Enum):
    """Transition timing curves"""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    SMOOTHSTEP = "smoothstep"
    BOUNCE = "bounce"
    ELASTIC = "elastic"


@dataclass
class TransitionConfig:
    """Configuration for transitions"""
    transition_type: TransitionType = TransitionType.FADE
    duration: float = 2.0  # seconds
    curve: TransitionCurve = TransitionCurve.EASE_IN_OUT
    parameters: Dict[str, Any] = field(default_factory=dict)
    pre_transition_delay: float = 0.0
    post_transition_delay: float = 0.0
    sync_points: List[float] = field(default_factory=list)
    
    @property
    def total_duration(self) -> float:
        """Get total transition duration including delays"""
        return self.pre_transition_delay + self.duration + self.post_transition_delay


@dataclass
class SceneState:
    """Represents a scene state for transitions"""
    id: str
    objects: Dict[str, Any]
    camera: Dict[str, Any]
    lighting: Dict[str, Any]
    environment: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Generate hash for scene state"""
        import hashlib
        content = json.dumps(self.objects, sort_keys=True) + \
                 json.dumps(self.camera, sort_keys=True) + \
                 json.dumps(self.lighting, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]


class TransitionBuilder:
    """
    Main transition builder for creating smooth transitions between scenes
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4
    ):
        """
        Initialize transition builder
        
        Args:
            config: Configuration dictionary
            max_workers: Maximum worker threads for parallel processing
        """
        self.config = config or {}
        self.max_workers = max_workers
        
        # Transition registry
        self.transition_registry = self._build_transition_registry()
        
        # State management
        self.current_state: Optional[SceneState] = None
        self.target_state: Optional[SceneState] = None
        self.transition_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Cache for precomputed transitions
        self.transition_cache = {}
        
        logger.info(f"TransitionBuilder initialized with {max_workers} workers")
    
    def _build_transition_registry(self) -> Dict[TransitionType, Callable]:
        """Build registry of transition functions"""
        return {
            TransitionType.FADE: self._create_fade_transition,
            TransitionType.DISSOLVE: self._create_dissolve_transition,
            TransitionType.WIPE: self._create_wipe_transition,
            TransitionType.SLIDE: self._create_slide_transition,
            TransitionType.ZOOM: self._create_zoom_transition,
            TransitionType.ROTATE: self._create_rotate_transition,
            TransitionType.MORPH: self._create_morph_transition,
            TransitionType.SEAMLESS: self._create_seamless_transition
        }
    
    def create_transition(
        self,
        from_state: SceneState,
        to_state: SceneState,
        config: TransitionConfig,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Create transition between two scene states
        
        Args:
            from_state: Starting scene state
            to_state: Target scene state
            config: Transition configuration
            progress_callback: Optional progress callback
            
        Returns:
            Transition data including intermediate states
        """
        timer = Timer()
        transition_id = self._generate_transition_id(from_state, to_state, config)
        
        # Check cache
        if transition_id in self.transition_cache:
            logger.debug(f"Using cached transition: {transition_id}")
            return self.transition_cache[transition_id].copy()
        
        logger.info(f"Creating transition {transition_id} ({config.transition_type.value})")
        
        # Validate states
        self._validate_states(from_state, to_state)
        
        # Get transition function
        transition_func = self.transition_registry.get(config.transition_type)
        if not transition_func:
            raise ValueError(f"Unsupported transition type: {config.transition_type}")
        
        # Calculate timing
        num_frames = int(config.duration * self.config.get("fps", 30))
        time_points = np.linspace(0, 1, num_frames)
        
        # Apply timing curve
        curved_points = self._apply_timing_curve(time_points, config.curve)
        
        # Generate intermediate states
        intermediate_states = []
        
        for i, t in enumerate(curved_points):
            # Create intermediate state
            intermediate = self._interpolate_states(from_state, to_state, t, config)
            
            # Apply transition-specific effects
            intermediate = transition_func(intermediate, t, config)
            
            intermediate_states.append(intermediate)
            
            # Update progress
            if progress_callback:
                progress_callback(i / len(curved_points))
        
        # Add delays if specified
        if config.pre_transition_delay > 0:
            pre_delay_frames = int(config.pre_transition_delay * self.config.get("fps", 30))
            intermediate_states = [from_state] * pre_delay_frames + intermediate_states
        
        if config.post_transition_delay > 0:
            post_delay_frames = int(config.post_transition_delay * self.config.get("fps", 30))
            intermediate_states = intermediate_states + [to_state] * post_delay_frames
        
        # Create transition result
        transition_data = {
            "id": transition_id,
            "from_state": from_state.id,
            "to_state": to_state.id,
            "type": config.transition_type.value,
            "duration": config.total_duration,
            "num_frames": len(intermediate_states),
            "fps": self.config.get("fps", 30),
            "intermediate_states": intermediate_states,
            "sync_points": config.sync_points,
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": config.__dict__
            }
        }
        
        # Cache transition
        self.transition_cache[transition_id] = transition_data.copy()
        
        # Update metrics
        self.metrics.record_operation("create_transition", timer.elapsed())
        
        # Log completion
        logger.info(f"Transition created in {timer.elapsed():.2f}s "
                   f"({len(intermediate_states)} frames)")
        
        # Update history
        self.transition_history.append({
            "timestamp": time.time(),
            "transition_id": transition_id,
            "duration": timer.elapsed(),
            "num_frames": len(intermediate_states)
        })
        
        # Limit history size
        if len(self.transition_history) > 100:
            self.transition_history = self.transition_history[-100:]
        
        return transition_data
    
    def _generate_transition_id(
        self,
        from_state: SceneState,
        to_state: SceneState,
        config: TransitionConfig
    ) -> str:
        """Generate unique ID for transition"""
        import hashlib
        
        content = f"{from_state.hash}_{to_state.hash}_{config.transition_type.value}_{config.duration}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _validate_states(self, from_state: SceneState, to_state: SceneState) -> None:
        """Validate scene states for transition"""
        if not from_state.objects or not to_state.objects:
            raise ValueError("Scene states must have objects")
        
        # Check for compatible object structures
        from_keys = set(from_state.objects.keys())
        to_keys = set(to_state.objects.keys())
        
        common_keys = from_keys.intersection(to_keys)
        if len(common_keys) < min(len(from_keys), len(to_keys)) * 0.5:
            logger.warning(f"Low object overlap between states: "
                          f"{len(common_keys)}/{len(from_keys)} common objects")
    
    def _apply_timing_curve(
        self,
        time_points: np.ndarray,
        curve: TransitionCurve
    ) -> np.ndarray:
        """Apply timing curve to time points"""
        if curve == TransitionCurve.LINEAR:
            return time_points
        
        elif curve == TransitionCurve.EASE_IN:
            return time_points ** 2
        
        elif curve == TransitionCurve.EASE_OUT:
            return 1 - (1 - time_points) ** 2
        
        elif curve == TransitionCurve.EASE_IN_OUT:
            return np.where(
                time_points < 0.5,
                2 * time_points ** 2,
                1 - (-2 * time_points + 2) ** 2 / 2
            )
        
        elif curve == TransitionCurve.SMOOTHSTEP:
            return time_points ** 2 * (3 - 2 * time_points)
        
        elif curve == TransitionCurve.BOUNCE:
            # Simplified bounce curve
            return np.where(
                time_points < 0.5,
                time_points * 2,
                1 - (time_points - 0.5) * 2
            )
        
        elif curve == TransitionCurve.ELASTIC:
            # Simplified elastic curve
            c4 = (2 * np.pi) / 3
            return np.where(
                time_points == 0,
                0,
                np.where(
                    time_points == 1,
                    1,
                    -np.power(2, 10 * time_points - 10) * np.sin((time_points * 10 - 10.75) * c4)
                )
            )
        
        return time_points
    
    def _interpolate_states(
        self,
        from_state: SceneState,
        to_state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """
        Interpolate between two scene states
        
        Args:
            from_state: Starting state
            to_state: Target state
            t: Interpolation factor (0-1)
            config: Transition configuration
            
        Returns:
            Interpolated scene state
        """
        # Create new state ID
        state_id = f"intermediate_{from_state.id}_{to_state.id}_{t:.3f}"
        
        # Interpolate objects
        interpolated_objects = self._interpolate_objects(
            from_state.objects, to_state.objects, t, config
        )
        
        # Interpolate camera
        interpolated_camera = self._interpolate_camera(
            from_state.camera, to_state.camera, t, config
        )
        
        # Interpolate lighting
        interpolated_lighting = self._interpolate_lighting(
            from_state.lighting, to_state.lighting, t, config
        )
        
        # Interpolate environment
        interpolated_environment = self._interpolate_environment(
            from_state.environment, to_state.environment, t, config
        )
        
        return SceneState(
            id=state_id,
            objects=interpolated_objects,
            camera=interpolated_camera,
            lighting=interpolated_lighting,
            environment=interpolated_environment,
            metadata={
                "interpolation_factor": t,
                "from_state": from_state.id,
                "to_state": to_state.id
            }
        )
    
    def _interpolate_objects(
        self,
        from_objects: Dict[str, Any],
        to_objects: Dict[str, Any],
        t: float,
        config: TransitionConfig
    ) -> Dict[str, Any]:
        """Interpolate object properties"""
        interpolated = {}
        
        # Get all object IDs
        all_object_ids = set(from_objects.keys()) | set(to_objects.keys())
        
        for obj_id in all_object_ids:
            from_obj = from_objects.get(obj_id)
            to_obj = to_objects.get(obj_id)
            
            if from_obj and to_obj:
                # Both states have this object - interpolate
                interpolated[obj_id] = self._interpolate_object(from_obj, to_obj, t)
            
            elif from_obj and not to_obj:
                # Object disappears in target state
                if t < 0.5:
                    # Fade out
                    interpolated[obj_id] = self._fade_out_object(from_obj, t * 2)
            
            elif not from_obj and to_obj:
                # Object appears in target state
                if t > 0.5:
                    # Fade in
                    interpolated[obj_id] = self._fade_in_object(to_obj, (t - 0.5) * 2)
        
        return interpolated
    
    def _interpolate_object(
        self,
        from_obj: Dict[str, Any],
        to_obj: Dict[str, Any],
        t: float
    ) -> Dict[str, Any]:
        """Interpolate single object"""
        interpolated = from_obj.copy()
        
        # Interpolate position
        if "position" in from_obj and "position" in to_obj:
            from_pos = np.array(from_obj["position"])
            to_pos = np.array(to_obj["position"])
            interpolated["position"] = (from_pos * (1 - t) + to_pos * t).tolist()
        
        # Interpolate rotation
        if "rotation" in from_obj and "rotation" in to_obj:
            from_rot = np.array(from_obj["rotation"])
            to_rot = np.array(to_obj["rotation"])
            
            # Spherical linear interpolation for quaternions
            if len(from_rot) == 4 and len(to_rot) == 4:
                interpolated["rotation"] = self._slerp_quaternion(from_rot, to_rot, t)
            else:
                interpolated["rotation"] = (from_rot * (1 - t) + to_rot * t).tolist()
        
        # Interpolate scale
        if "scale" in from_obj and "scale" in to_obj:
            from_scale = np.array(from_obj["scale"])
            to_scale = np.array(to_obj["scale"])
            interpolated["scale"] = (from_scale * (1 - t) + to_scale * t).tolist()
        
        # Interpolate color
        if "color" in from_obj and "color" in to_obj:
            from_color = np.array(from_obj["color"])
            to_color = np.array(to_obj["color"])
            interpolated["color"] = (from_color * (1 - t) + to_color * t).tolist()
        
        # Interpolate opacity
        if "opacity" in from_obj and "opacity" in to_obj:
            from_opacity = from_obj["opacity"]
            to_opacity = to_obj["opacity"]
            interpolated["opacity"] = from_opacity * (1 - t) + to_opacity * t
        
        return interpolated
    
    def _fade_out_object(self, obj: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Fade out object"""
        faded = obj.copy()
        
        if "opacity" in faded:
            faded["opacity"] = faded["opacity"] * (1 - t)
        else:
            faded["opacity"] = 1.0 - t
        
        # Scale down
        if "scale" in faded:
            scale = np.array(faded["scale"])
            faded["scale"] = (scale * (1 - t)).tolist()
        
        return faded
    
    def _fade_in_object(self, obj: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Fade in object"""
        faded = obj.copy()
        
        if "opacity" in faded:
            faded["opacity"] = faded["opacity"] * t
        else:
            faded["opacity"] = t
        
        # Scale up
        if "scale" in faded:
            scale = np.array(faded["scale"])
            faded["scale"] = (scale * t).tolist()
        
        return faded
    
    def _slerp_quaternion(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        t: float
    ) -> np.ndarray:
        """Spherical linear interpolation for quaternions"""
        # Normalize quaternions
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate dot product
        dot = np.dot(q1, q2)
        
        # If dot is negative, negate one quaternion for shortest path
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q1 * (1 - t) + q2 * t
            return result / np.linalg.norm(result)
        
        # Calculate angle
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        
        # Calculate coefficients
        q2 = q2 - q1 * dot
        q2 = q2 / np.linalg.norm(q2)
        
        return q1 * np.cos(theta) + q2 * np.sin(theta)
    
    def _interpolate_camera(
        self,
        from_camera: Dict[str, Any],
        to_camera: Dict[str, Any],
        t: float,
        config: TransitionConfig
    ) -> Dict[str, Any]:
        """Interpolate camera properties"""
        interpolated = from_camera.copy()
        
        # Interpolate position
        if "position" in from_camera and "position" in to_camera:
            from_pos = np.array(from_camera["position"])
            to_pos = np.array(to_camera["position"])
            interpolated["position"] = (from_pos * (1 - t) + to_pos * t).tolist()
        
        # Interpolate target
        if "target" in from_camera and "target" in to_camera:
            from_target = np.array(from_camera["target"])
            to_target = np.array(to_camera["target"])
            interpolated["target"] = (from_target * (1 - t) + to_target * t).tolist()
        
        # Interpolate field of view
        if "fov" in from_camera and "fov" in to_camera:
            from_fov = from_camera["fov"]
            to_fov = to_camera["fov"]
            interpolated["fov"] = from_fov * (1 - t) + to_fov * t
        
        return interpolated
    
    def _interpolate_lighting(
        self,
        from_lighting: Dict[str, Any],
        to_lighting: Dict[str, Any],
        t: float,
        config: TransitionConfig
    ) -> Dict[str, Any]:
        """Interpolate lighting properties"""
        interpolated = from_lighting.copy()
        
        # Interpolate ambient color
        if "ambient_color" in from_lighting and "ambient_color" in to_lighting:
            from_color = np.array(from_lighting["ambient_color"])
            to_color = np.array(to_lighting["ambient_color"])
            interpolated["ambient_color"] = (from_color * (1 - t) + to_color * t).tolist()
        
        # Interpolate ambient intensity
        if "ambient_intensity" in from_lighting and "ambient_intensity" in to_lighting:
            from_intensity = from_lighting["ambient_intensity"]
            to_intensity = to_lighting["ambient_intensity"]
            interpolated["ambient_intensity"] = from_intensity * (1 - t) + to_intensity * t
        
        # Interpolate directional light
        if "directional_light" in from_lighting and "directional_light" in to_lighting:
            from_light = from_lighting["directional_light"]
            to_light = to_lighting["directional_light"]
            
            interpolated_light = {}
            
            if "direction" in from_light and "direction" in to_light:
                from_dir = np.array(from_light["direction"])
                to_dir = np.array(to_light["direction"])
                interpolated_light["direction"] = (from_dir * (1 - t) + to_dir * t).tolist()
            
            if "color" in from_light and "color" in to_light:
                from_color = np.array(from_light["color"])
                to_color = np.array(to_light["color"])
                interpolated_light["color"] = (from_color * (1 - t) + to_color * t).tolist()
            
            if "intensity" in from_light and "intensity" in to_light:
                from_intensity = from_light["intensity"]
                to_intensity = to_light["intensity"]
                interpolated_light["intensity"] = from_intensity * (1 - t) + to_intensity * t
            
            interpolated["directional_light"] = interpolated_light
        
        return interpolated
    
    def _interpolate_environment(
        self,
        from_env: Dict[str, Any],
        to_env: Dict[str, Any],
        t: float,
        config: TransitionConfig
    ) -> Dict[str, Any]:
        """Interpolate environment properties"""
        interpolated = from_env.copy()
        
        # Interpolate background color
        if "background_color" in from_env and "background_color" in to_env:
            from_color = np.array(from_env["background_color"])
            to_color = np.array(to_env["background_color"])
            interpolated["background_color"] = (from_color * (1 - t) + to_color * t).tolist()
        
        # Interpolate fog
        if "fog" in from_env and "fog" in to_env:
            from_fog = from_env["fog"]
            to_fog = to_env["fog"]
            
            interpolated_fog = {}
            
            if "color" in from_fog and "color" in to_fog:
                from_color = np.array(from_fog["color"])
                to_color = np.array(to_fog["color"])
                interpolated_fog["color"] = (from_color * (1 - t) + to_color * t).tolist()
            
            if "density" in from_fog and "density" in to_fog:
                from_density = from_fog["density"]
                to_density = to_fog["density"]
                interpolated_fog["density"] = from_density * (1 - t) + to_density * t
            
            interpolated["fog"] = interpolated_fog
        
        return interpolated
    
    # Transition effect functions
    def _create_fade_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create fade transition effect"""
        # Fade applies globally to all objects
        modified_state = SceneState(
            id=state.id + f"_fade_{t:.3f}",
            objects=state.objects.copy(),
            camera=state.camera.copy(),
            lighting=state.lighting.copy(),
            environment=state.environment.copy(),
            metadata=state.metadata.copy()
        )
        
        # Adjust global opacity based on fade direction
        fade_type = config.parameters.get("fade_type", "crossfade")
        
        if fade_type == "crossfade":
            # Crossfade: from_state fades out while to_state fades in
            if t < 0.5:
                # First half: fade out from_state
                opacity = 1.0 - (t * 2)
            else:
                # Second half: fade in to_state
                opacity = (t - 0.5) * 2
        
        elif fade_type == "fade_out_in":
            # Complete fade out then fade in
            if t < 0.5:
                opacity = 1.0 - (t * 2)
            else:
                opacity = (t - 0.5) * 2
        
        else:  # simple fade
            opacity = t
        
        # Apply opacity to environment
        if "background_color" in modified_state.environment:
            bg_color = np.array(modified_state.environment["background_color"])
            modified_state.environment["background_color"] = (bg_color * opacity).tolist()
        
        return modified_state
    
    def _create_dissolve_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create dissolve transition effect"""
        modified_state = SceneState(
            id=state.id + f"_dissolve_{t:.3f}",
            objects=state.objects.copy(),
            camera=state.camera.copy(),
            lighting=state.lighting.copy(),
            environment=state.environment.copy(),
            metadata=state.metadata.copy()
        )
        
        # Apply dissolve pattern
        pattern = config.parameters.get("pattern", "random")
        cell_size = config.parameters.get("cell_size", 0.1)
        
        for obj_id, obj in modified_state.objects.items():
            # Generate dissolve pattern based on object position
            if "position" in obj:
                pos = np.array(obj["position"])
                
                if pattern == "random":
                    # Use object ID for deterministic randomness
                    seed = hash(obj_id) % 10000
                    np.random.seed(seed)
                    dissolve_value = np.random.random()
                elif pattern == "grid":
                    # Grid-based pattern
                    grid_pos = (pos / cell_size).astype(int)
                    dissolve_value = (hash(tuple(grid_pos)) % 1000) / 1000.0
                else:  # distance-based
                    center = config.parameters.get("center", [0, 0, 0])
                    distance = np.linalg.norm(pos - np.array(center))
                    dissolve_value = (distance % cell_size) / cell_size
                
                # Apply dissolve threshold
                if t < dissolve_value:
                    # Object is dissolved
                    if "opacity" in obj:
                        obj["opacity"] = 0.0
                    else:
                        obj["opacity"] = 0.0
        
        return modified_state
    
    def _create_wipe_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create wipe transition effect"""
        modified_state = SceneState(
            id=state.id + f"_wipe_{t:.3f}",
            objects=state.objects.copy(),
            camera=state.camera.copy(),
            lighting=state.lighting.copy(),
            environment=state.environment.copy(),
            metadata=state.metadata.copy()
        )
        
        # Wipe direction and parameters
        direction = config.parameters.get("direction", "right")
        softness = config.parameters.get("softness", 0.1)
        
        for obj_id, obj in modified_state.objects.items():
            if "position" in obj:
                pos = np.array(obj["position"])
                
                # Calculate wipe progress for this object
                if direction == "right":
                    wipe_value = (pos[0] + 10) / 20  # Map from -10,10 to 0,1
                elif direction == "left":
                    wipe_value = (10 - pos[0]) / 20
                elif direction == "up":
                    wipe_value = (pos[1] + 10) / 20
                elif direction == "down":
                    wipe_value = (10 - pos[1]) / 20
                elif direction == "center":
                    distance = np.linalg.norm(pos[:2])  # Ignore Z
                    wipe_value = 1.0 - (distance / 10)
                else:
                    wipe_value = t
                
                # Apply wipe with soft edges
                if t < wipe_value - softness:
                    # Fully visible
                    opacity = 1.0
                elif t > wipe_value + softness:
                    # Fully hidden
                    opacity = 0.0
                else:
                    # Soft edge
                    opacity = 1.0 - (t - (wipe_value - softness)) / (2 * softness)
                
                if "opacity" in obj:
                    obj["opacity"] = min(obj.get("opacity", 1.0), opacity)
                else:
                    obj["opacity"] = opacity
        
        return modified_state
    
    def _create_slide_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create slide transition effect"""
        modified_state = SceneState(
            id=state.id + f"_slide_{t:.3f}",
            objects=state.objects.copy(),
            camera=state.camera.copy(),
            lighting=state.lighting.copy(),
            environment=state.environment.copy(),
            metadata=state.metadata.copy()
        )
        
        # Slide parameters
        direction = config.parameters.get("direction", "right")
        distance = config.parameters.get("distance", 20.0)
        
        # Calculate offset based on progress
        offset = distance * (1 - t) if t < 0.5 else distance * (t - 0.5) * 2
        
        # Apply offset to objects
        for obj_id, obj in modified_state.objects.items():
            if "position" in obj:
                pos = np.array(obj["position"])
                
                if direction == "right":
                    pos[0] += offset
                elif direction == "left":
                    pos[0] -= offset
                elif direction == "up":
                    pos[1] += offset
                elif direction == "down":
                    pos[1] -= offset
                elif direction == "forward":
                    pos[2] += offset
                elif direction == "backward":
                    pos[2] -= offset
                
                obj["position"] = pos.tolist()
        
        return modified_state
    
    def _create_zoom_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create zoom transition effect"""
        modified_state = SceneState(
            id=state.id + f"_zoom_{t:.3f}",
            objects=state.objects.copy(),
            camera=state.camera.copy(),
            lighting=state.lighting.copy(),
            environment=state.environment.copy(),
            metadata=state.metadata.copy()
        )
        
        # Zoom parameters
        zoom_type = config.parameters.get("zoom_type", "in_out")
        intensity = config.parameters.get("intensity", 2.0)
        
        # Calculate zoom factor
        if zoom_type == "in_out":
            # Zoom in then out
            if t < 0.5:
                zoom_factor = 1.0 + (intensity - 1.0) * (t * 2)
            else:
                zoom_factor = intensity - (intensity - 1.0) * ((t - 0.5) * 2)
        elif zoom_type == "out_in":
            # Zoom out then in
            if t < 0.5:
                zoom_factor = 1.0 - (1.0 - 1.0/intensity) * (t * 2)
            else:
                zoom_factor = 1.0/intensity + (1.0 - 1.0/intensity) * ((t - 0.5) * 2)
        else:  # simple zoom
            zoom_factor = 1.0 + (intensity - 1.0) * t
        
        # Apply zoom to camera
        if "fov" in modified_state.camera:
            modified_state.camera["fov"] = modified_state.camera["fov"] / zoom_factor
        
        # Apply zoom to objects (scale them)
        for obj_id, obj in modified_state.objects.items():
            if "scale" in obj:
                scale = np.array(obj["scale"])
                obj["scale"] = (scale * zoom_factor).tolist()
        
        return modified_state
    
    def _create_rotate_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create rotate transition effect"""
        modified_state = SceneState(
            id=state.id + f"_rotate_{t:.3f}",
            objects=state.objects.copy(),
            camera=state.camera.copy(),
            lighting=state.lighting.copy(),
            environment=state.environment.copy(),
            metadata=state.metadata.copy()
        )
        
        # Rotation parameters
        axis = config.parameters.get("axis", "y")
        angle = config.parameters.get("angle", 360.0)  # degrees
        
        # Calculate rotation angle
        rotation_angle = angle * t
        
        # Create rotation matrix
        if axis == "x":
            rot_matrix = self._rotation_matrix_x(np.radians(rotation_angle))
        elif axis == "y":
            rot_matrix = self._rotation_matrix_y(np.radians(rotation_angle))
        elif axis == "z":
            rot_matrix = self._rotation_matrix_z(np.radians(rotation_angle))
        else:
            rot_matrix = np.eye(3)
        
        # Apply rotation to camera
        if "position" in modified_state.camera:
            pos = np.array(modified_state.camera["position"])
            modified_state.camera["position"] = (rot_matrix @ pos).tolist()
        
        if "target" in modified_state.camera:
            target = np.array(modified_state.camera["target"])
            modified_state.camera["target"] = (rot_matrix @ target).tolist()
        
        # Apply rotation to objects
        for obj_id, obj in modified_state.objects.items():
            if "position" in obj:
                pos = np.array(obj["position"])
                obj["position"] = (rot_matrix @ pos).tolist()
            
            if "rotation" in obj:
                # Combine rotations
                current_rot = np.array(obj["rotation"])
                if len(current_rot) == 9:  # 3x3 matrix
                    obj_rot = current_rot.reshape(3, 3)
                    new_rot = rot_matrix @ obj_rot
                    obj["rotation"] = new_rot.flatten().tolist()
        
        return modified_state
    
    def _create_morph_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create morph transition effect"""
        # Morph transition requires special handling of geometry
        # For now, implement as a combination of other effects
        modified_state = SceneState(
            id=state.id + f"_morph_{t:.3f}",
            objects=state.objects.copy(),
            camera=state.camera.copy(),
            lighting=state.lighting.copy(),
            environment=state.environment.copy(),
            metadata=state.metadata.copy()
        )
        
        # Apply warping effect
        warp_intensity = config.parameters.get("warp_intensity", 0.5)
        
        for obj_id, obj in modified_state.objects.items():
            if "position" in obj:
                pos = np.array(obj["position"])
                
                # Simple sinusoidal warp
                warp = np.sin(t * np.pi * 2) * warp_intensity
                pos += np.random.randn(3) * warp
                
                obj["position"] = pos.tolist()
            
            # Morph scale
            if "scale" in obj:
                scale = np.array(obj["scale"])
                
                # Pulsating effect
                pulse = 1.0 + 0.2 * np.sin(t * np.pi * 4)
                scale *= pulse
                
                obj["scale"] = scale.tolist()
        
        return modified_state
    
    def _create_seamless_transition(
        self,
        state: SceneState,
        t: float,
        config: TransitionConfig
    ) -> SceneState:
        """Create seamless transition (no visible transition)"""
        # Seamless transition just returns the state as-is
        # The seamless nature comes from careful state preparation
        return state
    
    def _rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Create rotation matrix around X axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    
    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Y axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    
    def _rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Create rotation matrix around Z axis"""
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    
    def create_transition_sequence(
        self,
        states: List[SceneState],
        configs: List[TransitionConfig],
        loop: bool = False
    ) -> Dict[str, Any]:
        """
        Create sequence of transitions
        
        Args:
            states: List of scene states
            configs: List of transition configurations
            loop: Whether to loop back to first state
            
        Returns:
            Transition sequence data
        """
        if len(states) < 2:
            raise ValueError("Need at least 2 states for sequence")
        
        if len(configs) != len(states) - 1:
            raise ValueError(f"Need {len(states)-1} configs for {len(states)} states")
        
        timer = Timer()
        sequence_id = f"sequence_{int(time.time())}"
        
        logger.info(f"Creating transition sequence {sequence_id} with {len(states)} states")
        
        # Create individual transitions
        transitions = []
        total_duration = 0.0
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            config = configs[i]
            
            transition = self.create_transition(from_state, to_state, config)
            transitions.append(transition)
            total_duration += config.total_duration
        
        # Add loop transition if requested
        if loop and len(states) >= 2:
            loop_config = TransitionConfig(
                transition_type=TransitionType.SEAMLESS,
                duration=1.0
            )
            loop_transition = self.create_transition(
                states[-1], states[0], loop_config
            )
            transitions.append(loop_transition)
            total_duration += loop_config.total_duration
        
        # Create sequence data
        sequence_data = {
            "id": sequence_id,
            "num_states": len(states),
            "num_transitions": len(transitions),
            "total_duration": total_duration,
            "loop": loop,
            "transitions": transitions,
            "states": [s.id for s in states],
            "metadata": {
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_frames": sum(t["num_frames"] for t in transitions)
            }
        }
        
        logger.info(f"Transition sequence created in {timer.elapsed():.2f}s")
        
        return sequence_data
    
    def export_transition(
        self,
        transition_data: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Export transition to file
        
        Args:
            transition_data: Transition data
            output_path: Output file path
            format: Export format
        """
        output_path = Path(output_path)
        
        if format == "json":
            save_json(transition_data, output_path.with_suffix('.json'))
            logger.info(f"Exported transition to {output_path.with_suffix('.json')}")
        
        elif format == "video":
            # Would require video encoding library
            logger.warning("Video export not implemented")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def preview_transition(
        self,
        transition_data: Dict[str, Any],
        frame_callback: Callable[[Dict[str, Any], int], None],
        fps: int = 30
    ) -> None:
        """
        Preview transition by calling callback for each frame
        
        Args:
            transition_data: Transition data
            frame_callback: Callback for each frame
            fps: Frames per second
        """
        frames = transition_data["intermediate_states"]
        
        logger.info(f"Previewing transition with {len(frames)} frames")
        
        for i, state in enumerate(frames):
            frame_callback(state, i)
            
            # Simulate frame timing
            time.sleep(1.0 / fps)
    
    def get_transition_info(self, transition_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a transition"""
        if transition_id in self.transition_cache:
            data = self.transition_cache[transition_id]
            
            return {
                "id": data["id"],
                "type": data["type"],
                "duration": data["duration"],
                "num_frames": data["num_frames"],
                "from_state": data["from_state"],
                "to_state": data["to_state"]
            }
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def clear_cache(self) -> None:
        """Clear transition cache"""
        self.transition_cache.clear()
        logger.info("Transition cache cleared")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("TransitionBuilder cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        return (f"TransitionBuilder(cache={len(self.transition_cache)}, "
                f"history={len(self.transition_history)})")


# Factory function for creating transition builders
def create_transition_builder(
    config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4
) -> TransitionBuilder:
    """
    Factory function to create transition builders
    
    Args:
        config: Configuration dictionary
        max_workers: Maximum worker threads
        
    Returns:
        TransitionBuilder instance
    """
    return TransitionBuilder(config=config, max_workers=max_workers)