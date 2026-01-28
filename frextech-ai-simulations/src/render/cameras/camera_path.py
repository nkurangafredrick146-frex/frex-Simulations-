"""
Camera path animation system for creating smooth camera movements.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import json
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import math
from scipy.interpolate import CubicSpline, interp1d


class InterpolationMode(Enum):
    """Interpolation modes for camera paths."""
    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    BEZIER = "bezier"
    CATMULL_ROM = "catmull_rom"
    SMOOTH_STEP = "smooth_step"
    EXPONENTIAL = "exponential"


class EasingFunction(Enum):
    """Easing functions for animation."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    SMOOTH = "smooth"
    BOUNCE = "bounce"
    ELASTIC = "elastic"


@dataclass
class Keyframe:
    """Camera keyframe for animation."""
    time: float  # in seconds
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    fov: float = 60.0
    interpolation: InterpolationMode = InterpolationMode.CUBIC_SPLINE
    easing: EasingFunction = EasingFunction.EASE_IN_OUT
    label: str = ""
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.float32)
        if not isinstance(self.target, np.ndarray):
            self.target = np.array(self.target, dtype=np.float32)
        if not isinstance(self.up, np.ndarray):
            self.up = np.array(self.up, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "time": self.time,
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "up": self.up.tolist(),
            "fov": self.fov,
            "interpolation": self.interpolation.value,
            "easing": self.easing.value,
            "label": self.label,
            "custom_data": self.custom_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Keyframe":
        """Create from dictionary."""
        return cls(
            time=data["time"],
            position=np.array(data["position"], dtype=np.float32),
            target=np.array(data["target"], dtype=np.float32),
            up=np.array(data.get("up", [0, 1, 0]), dtype=np.float32),
            fov=data.get("fov", 60.0),
            interpolation=InterpolationMode(data.get("interpolation", "cubic_spline")),
            easing=EasingFunction(data.get("easing", "ease_in_out")),
            label=data.get("label", ""),
            custom_data=data.get("custom_data", {})
        )


class CameraPath:
    """Camera path for animation sequences."""
    
    def __init__(self, name: str = "CameraPath"):
        """Initialize camera path.
        
        Args:
            name: Name of the camera path
        """
        self.name = name
        self.keyframes: List[Keyframe] = []
        self.loop: bool = False
        self.speed: float = 1.0
        self.duration: float = 0.0
        
        # Precomputed interpolation data
        self._position_splines: Optional[List[CubicSpline]] = None
        self._target_splines: Optional[List[CubicSpline]] = None
        self._fov_spline: Optional[CubicSpline] = None
        self._times: Optional[np.ndarray] = None
        
    def add_keyframe(self, keyframe: Keyframe):
        """Add a keyframe to the path.
        
        Args:
            keyframe: Keyframe to add
        """
        self.keyframes.append(keyframe)
        self.keyframes.sort(key=lambda k: k.time)
        self.duration = max(k.time for k in self.keyframes) if self.keyframes else 0.0
        self._invalidate_cache()
    
    def remove_keyframe(self, index: int):
        """Remove a keyframe by index.
        
        Args:
            index: Index of keyframe to remove
        """
        if 0 <= index < len(self.keyframes):
            del self.keyframes[index]
            self.duration = max(k.time for k in self.keyframes) if self.keyframes else 0.0
            self._invalidate_cache()
    
    def clear_keyframes(self):
        """Remove all keyframes."""
        self.keyframes.clear()
        self.duration = 0.0
        self._invalidate_cache()
    
    def get_keyframe_at_time(self, time: float) -> Optional[Keyframe]:
        """Get keyframe at or near specified time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Keyframe at time, or None if not found
        """
        for kf in self.keyframes:
            if abs(kf.time - time) < 0.001:
                return kf
        return None
    
    def interpolate(self, time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Interpolate camera state at given time.
        
        Args:
            time: Time in seconds (will be clamped to path duration)
            
        Returns:
            Tuple of (position, target, up, fov)
        """
        if not self.keyframes:
            return (np.zeros(3), np.array([0, 0, -1]), np.array([0, 1, 0]), 60.0)
        
        # Handle looping
        if self.loop and self.duration > 0:
            time = time % self.duration
        else:
            time = max(0.0, min(time, self.duration))
        
        # Find surrounding keyframes
        prev_idx, next_idx = self._find_keyframe_indices(time)
        
        if prev_idx == next_idx:
            # Exact match or single keyframe
            kf = self.keyframes[prev_idx]
            return kf.position, kf.target, kf.up, kf.fov
        
        prev_kf = self.keyframes[prev_idx]
        next_kf = self.keyframes[next_idx]
        
        # Calculate interpolation factor
        t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
        
        # Apply easing
        t = self._apply_easing(t, prev_kf.easing)
        
        # Interpolate based on mode
        if prev_kf.interpolation == InterpolationMode.LINEAR:
            return self._linear_interpolate(prev_kf, next_kf, t)
        elif prev_kf.interpolation == InterpolationMode.CUBIC_SPLINE:
            return self._cubic_spline_interpolate(time)
        elif prev_kf.interpolation == InterpolationMode.BEZIER:
            return self._bezier_interpolate(prev_kf, next_kf, t)
        elif prev_kf.interpolation == InterpolationMode.CATMULL_ROM:
            return self._catmull_rom_interpolate(prev_idx, next_idx, t)
        elif prev_kf.interpolation == InterpolationMode.SMOOTH_STEP:
            return self._smooth_step_interpolate(prev_kf, next_kf, t)
        elif prev_kf.interpolation == InterpolationMode.EXPONENTIAL:
            return self._exponential_interpolate(prev_kf, next_kf, t)
        else:
            return self._linear_interpolate(prev_kf, next_kf, t)
    
    def _find_keyframe_indices(self, time: float) -> Tuple[int, int]:
        """Find indices of keyframes surrounding given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Tuple of (prev_index, next_index)
        """
        if not self.keyframes:
            return 0, 0
        
        # Binary search for efficiency
        low, high = 0, len(self.keyframes) - 1
        
        while low <= high:
            mid = (low + high) // 2
            if self.keyframes[mid].time < time:
                low = mid + 1
            elif self.keyframes[mid].time > time:
                high = mid - 1
            else:
                return mid, mid
        
        # At this point, low > high
        prev_idx = max(0, min(low - 1, len(self.keyframes) - 1))
        next_idx = min(len(self.keyframes) - 1, max(high + 1, 0))
        
        return prev_idx, next_idx
    
    def _apply_easing(self, t: float, easing: EasingFunction) -> float:
        """Apply easing function to interpolation factor.
        
        Args:
            t: Raw interpolation factor (0-1)
            easing: Easing function to apply
            
        Returns:
            Eased interpolation factor
        """
        if easing == EasingFunction.LINEAR:
            return t
        elif easing == EasingFunction.EASE_IN:
            return t * t
        elif easing == EasingFunction.EASE_OUT:
            return t * (2 - t)
        elif easing == EasingFunction.EASE_IN_OUT:
            return t * t * (3 - 2 * t)
        elif easing == EasingFunction.SMOOTH:
            return t * t * t * (t * (t * 6 - 15) + 10)
        elif easing == EasingFunction.BOUNCE:
            if t < 4/11:
                return (121 * t * t) / 16
            elif t < 8/11:
                return (363/40 * t * t) - (99/10 * t) + 17/5
            elif t < 9/10:
                return (4356/361 * t * t) - (35442/1805 * t) + 16061/1805
            else:
                return (54/5 * t * t) - (513/25 * t) + 268/25
        elif easing == EasingFunction.ELASTIC:
            if t == 0 or t == 1:
                return t
            return math.pow(2, -10 * t) * math.sin((t - 0.075) * (2 * math.pi) / 0.3) + 1
        else:
            return t
    
    def _linear_interpolate(self, prev: Keyframe, next: Keyframe, t: float
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Linear interpolation between keyframes.
        
        Args:
            prev: Previous keyframe
            next: Next keyframe
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated camera state
        """
        position = prev.position * (1 - t) + next.position * t
        target = prev.target * (1 - t) + next.target * t
        fov = prev.fov * (1 - t) + next.fov * t
        
        # Spherical linear interpolation for up vector
        up = self._slerp_vectors(prev.up, next.up, t)
        
        return position, target, up, fov
    
    def _cubic_spline_interpolate(self, time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Cubic spline interpolation.
        
        Args:
            time: Time in seconds
            
        Returns:
            Interpolated camera state
        """
        self._ensure_splines_cached()
        
        if self._position_splines is None or self._target_splines is None or self._fov_spline is None:
            return self._linear_interpolate(self.keyframes[0], self.keyframes[-1], 0.5)
        
        position = np.array([
            self._position_splines[0](time),
            self._position_splines[1](time),
            self._position_splines[2](time)
        ], dtype=np.float32)
        
        target = np.array([
            self._target_splines[0](time),
            self._target_splines[1](time),
            self._target_splines[2](time)
        ], dtype=np.float32)
        
        fov = float(self._fov_spline(time))
        
        # For up vector, find nearest keyframes and slerp
        prev_idx, next_idx = self._find_keyframe_indices(time)
        if prev_idx == next_idx:
            up = self.keyframes[prev_idx].up
        else:
            prev_kf = self.keyframes[prev_idx]
            next_kf = self.keyframes[next_idx]
            t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
            up = self._slerp_vectors(prev_kf.up, next_kf.up, t)
        
        return position, target, up, fov
    
    def _bezier_interpolate(self, prev: Keyframe, next: Keyframe, t: float
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Bezier curve interpolation.
        
        Args:
            prev: Previous keyframe
            next: Next keyframe
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated camera state
        """
        # Extract control points from custom data
        p0 = prev.position
        p3 = next.position
        
        # Get control points or use defaults
        p1 = np.array(prev.custom_data.get("control_point_out", p0 + (p3 - p0) * 0.33))
        p2 = np.array(next.custom_data.get("control_point_in", p0 + (p3 - p0) * 0.66))
        
        # Cubic Bezier interpolation
        position = self._cubic_bezier(p0, p1, p2, p3, t)
        
        # Similar for target
        t0 = prev.target
        t3 = next.target
        t1 = np.array(prev.custom_data.get("target_control_out", t0 + (t3 - t0) * 0.33))
        t2 = np.array(next.custom_data.get("target_control_in", t0 + (t3 - t0) * 0.66))
        target = self._cubic_bezier(t0, t1, t2, t3, t)
        
        # Linear interpolation for fov and slerp for up
        fov = prev.fov * (1 - t) + next.fov * t
        up = self._slerp_vectors(prev.up, next.up, t)
        
        return position, target, up, fov
    
    def _catmull_rom_interpolate(self, prev_idx: int, next_idx: int, t: float
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Catmull-Rom spline interpolation.
        
        Args:
            prev_idx: Previous keyframe index
            next_idx: Next keyframe index
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated camera state
        """
        # Need four points for Catmull-Rom
        p0_idx = max(0, prev_idx - 1)
        p1_idx = prev_idx
        p2_idx = next_idx
        p3_idx = min(len(self.keyframes) - 1, next_idx + 1)
        
        p0 = self.keyframes[p0_idx].position
        p1 = self.keyframes[p1_idx].position
        p2 = self.keyframes[p2_idx].position
        p3 = self.keyframes[p3_idx].position
        
        position = self._catmull_rom(p0, p1, p2, p3, t)
        
        # Similar for target
        t0 = self.keyframes[p0_idx].target
        t1 = self.keyframes[p1_idx].target
        t2 = self.keyframes[p2_idx].target
        t3 = self.keyframes[p3_idx].target
        target = self._catmull_rom(t0, t1, t2, t3, t)
        
        # Interpolate fov and up from surrounding keyframes
        prev_kf = self.keyframes[prev_idx]
        next_kf = self.keyframes[next_idx]
        fov = prev_kf.fov * (1 - t) + next_kf.fov * t
        up = self._slerp_vectors(prev_kf.up, next_kf.up, t)
        
        return position, target, up, fov
    
    def _smooth_step_interpolate(self, prev: Keyframe, next: Keyframe, t: float
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Smooth step interpolation.
        
        Args:
            prev: Previous keyframe
            next: Next keyframe
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated camera state
        """
        # Smooth step function
        smooth_t = t * t * (3 - 2 * t)
        
        position = prev.position * (1 - smooth_t) + next.position * smooth_t
        target = prev.target * (1 - smooth_t) + next.target * smooth_t
        fov = prev.fov * (1 - smooth_t) + next.fov * smooth_t
        up = self._slerp_vectors(prev.up, next.up, smooth_t)
        
        return position, target, up, fov
    
    def _exponential_interpolate(self, prev: Keyframe, next: Keyframe, t: float
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Exponential interpolation.
        
        Args:
            prev: Previous keyframe
            next: Next keyframe
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated camera state
        """
        # Exponential easing
        exp_t = math.exp(5 * (t - 1)) if t < 0.8 else 1 - math.exp(-5 * (t - 0.8))
        exp_t = max(0.0, min(1.0, exp_t))
        
        position = prev.position * (1 - exp_t) + next.position * exp_t
        target = prev.target * (1 - exp_t) + next.target * exp_t
        fov = prev.fov * (1 - exp_t) + next.fov * exp_t
        up = self._slerp_vectors(prev.up, next.up, exp_t)
        
        return position, target, up, fov
    
    def _cubic_bezier(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
        """Calculate point on cubic Bezier curve.
        
        Args:
            p0, p1, p2, p3: Control points
            t: Parameter (0-1)
            
        Returns:
            Point on curve
        """
        u = 1 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t
        
        return (uuu * p0 + 
                3 * uu * t * p1 + 
                3 * u * tt * p2 + 
                ttt * p3)
    
    def _catmull_rom(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: float) -> np.ndarray:
        """Calculate point on Catmull-Rom spline.
        
        Args:
            p0, p1, p2, p3: Control points
            t: Parameter (0-1)
            
        Returns:
            Point on spline
        """
        # Catmull-Rom spline formula
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
        )
    
    def _slerp_vectors(self, v1: np.ndarray, v2: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            t: Interpolation factor (0-1)
            
        Returns:
            Interpolated vector
        """
        # Normalize inputs
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Calculate dot product
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        
        # If vectors are nearly identical, use linear interpolation
        if dot > 0.9995:
            result = v1_norm * (1 - t) + v2_norm * t
            return result / np.linalg.norm(result)
        
        # Calculate angle
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        
        if abs(sin_theta) < 1e-10:
            return v1_norm
        
        # Slerp formula
        a = math.sin((1 - t) * theta) / sin_theta
        b = math.sin(t * theta) / sin_theta
        
        return a * v1_norm + b * v2_norm
    
    def _ensure_splines_cached(self):
        """Ensure spline interpolation data is cached."""
        if (self._position_splines is not None and 
            self._target_splines is not None and 
            self._fov_spline is not None and
            self._times is not None):
            return
        
        if len(self.keyframes) < 2:
            return
        
        # Extract times and values
        self._times = np.array([kf.time for kf in self.keyframes])
        
        # Position splines (x, y, z components separately)
        positions = np.array([kf.position for kf in self.keyframes])
        self._position_splines = [
            CubicSpline(self._times, positions[:, 0]),
            CubicSpline(self._times, positions[:, 1]),
            CubicSpline(self._times, positions[:, 2])
        ]
        
        # Target splines
        targets = np.array([kf.target for kf in self.keyframes])
        self._target_splines = [
            CubicSpline(self._times, targets[:, 0]),
            CubicSpline(self._times, targets[:, 1]),
            CubicSpline(self._times, targets[:, 2])
        ]
        
        # FOV spline
        fovs = np.array([kf.fov for kf in self.keyframes])
        self._fov_spline = CubicSpline(self._times, fovs)
    
    def _invalidate_cache(self):
        """Invalidate cached interpolation data."""
        self._position_splines = None
        self._target_splines = None
        self._fov_spline = None
        self._times = None
    
    def sample_path(self, num_samples: int = 100) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Sample the path at regular intervals.
        
        Args:
            num_samples: Number of samples to take
            
        Returns:
            List of sampled camera states
        """
        samples = []
        if self.duration == 0:
            return samples
        
        for i in range(num_samples):
            time = (i / (num_samples - 1)) * self.duration
            samples.append(self.interpolate(time))
        
        return samples
    
    def calculate_velocity(self, time: float) -> Tuple[float, np.ndarray]:
        """Calculate camera velocity at given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Tuple of (speed, velocity_vector)
        """
        if not self.keyframes or self.duration == 0:
            return 0.0, np.zeros(3)
        
        # Use finite difference to estimate velocity
        eps = 0.001
        t1 = max(0.0, time - eps)
        t2 = min(self.duration, time + eps)
        
        if t1 == t2:
            return 0.0, np.zeros(3)
        
        pos1, _, _, _ = self.interpolate(t1)
        pos2, _, _, _ = self.interpolate(t2)
        
        velocity = (pos2 - pos1) / (t2 - t1)
        speed = np.linalg.norm(velocity)
        
        return speed, velocity
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save camera path to file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            "name": self.name,
            "loop": self.loop,
            "speed": self.speed,
            "duration": self.duration,
            "keyframes": [kf.to_dict() for kf in self.keyframes]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filepath: Union[str, Path]):
        """Load camera path from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.name = data.get("name", "CameraPath")
        self.loop = data.get("loop", False)
        self.speed = data.get("speed", 1.0)
        self.duration = data.get("duration", 0.0)
        self.keyframes = [Keyframe.from_dict(kf) for kf in data.get("keyframes", [])]
        self._invalidate_cache()
    
    def create_from_points(self, points: List[np.ndarray], 
                          targets: Optional[List[np.ndarray]] = None,
                          duration_per_segment: float = 1.0):
        """Create camera path from list of points.
        
        Args:
            points: List of camera positions
            targets: Optional list of camera targets
            duration_per_segment: Duration for each segment
        """
        self.clear_keyframes()
        
        if not points:
            return
        
        # If no targets provided, look slightly forward from position
        if targets is None:
            targets = []
            for i in range(len(points)):
                if i < len(points) - 1:
                    # Look toward next point
                    target = points[i] + (points[i + 1] - points[i]) * 0.5
                else:
                    # Last point looks forward from its position
                    target = points[i] + np.array([0, 0, -1])
                targets.append(target)
        
        # Create keyframes
        for i, (pos, tgt) in enumerate(zip(points, targets)):
            time = i * duration_per_segment
            kf = Keyframe(
                time=time,
                position=pos,
                target=tgt,
                interpolation=InterpolationMode.CUBIC_SPLINE,
                easing=EasingFunction.EASE_IN_OUT
            )
            self.add_keyframe(kf)
        
        self.duration = (len(points) - 1) * duration_per_segment
