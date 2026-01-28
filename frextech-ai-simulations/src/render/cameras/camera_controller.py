"""
Camera controller module for interactive 3D camera manipulation.
Supports multiple camera types and control schemes.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
import math
from enum import Enum
import json
from dataclasses import dataclass, field
from pathlib import Path

class CameraProjection(Enum):
    """Camera projection types."""
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"
    PANORAMIC = "panoramic"

class CameraControlMode(Enum):
    """Camera control modes."""
    ORBIT = "orbit"
    FLY = "fly"
    FPS = "fps"
    DOLLY = "dolly"
    TRACK = "track"

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fov: float = 60.0  # degrees
    aspect_ratio: float = 16.0 / 9.0
    near_plane: float = 0.1
    far_plane: float = 1000.0
    focal_length: float = 35.0  # mm
    sensor_width: float = 36.0  # mm
    principal_point: Tuple[float, float] = (0.0, 0.0)
    skew: float = 0.0
    
    def get_projection_matrix(self) -> np.ndarray:
        """Calculate projection matrix."""
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        aspect = self.aspect_ratio
        near, far = self.near_plane, self.far_plane
        
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fov": self.fov,
            "aspect_ratio": self.aspect_ratio,
            "near_plane": self.near_plane,
            "far_plane": self.far_plane,
            "focal_length": self.focal_length,
            "sensor_width": self.sensor_width,
            "principal_point": list(self.principal_point),
            "skew": self.skew
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraIntrinsics":
        """Create from dictionary."""
        return cls(
            fov=data.get("fov", 60.0),
            aspect_ratio=data.get("aspect_ratio", 16.0/9.0),
            near_plane=data.get("near_plane", 0.1),
            far_plane=data.get("far_plane", 1000.0),
            focal_length=data.get("focal_length", 35.0),
            sensor_width=data.get("sensor_width", 36.0),
            principal_point=tuple(data.get("principal_point", (0.0, 0.0))),
            skew=data.get("skew", 0.0)
        )

@dataclass
class CameraState:
    """Camera state including position and orientation."""
    position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 5], dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32))  # quaternion
    projection_type: CameraProjection = CameraProjection.PERSPECTIVE
    
    def get_view_matrix(self) -> np.ndarray:
        """Calculate view matrix."""
        # Forward vector
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        # Right vector
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        # Recalculate up vector
        up = np.cross(right, forward)
        
        # Create view matrix
        rotation = np.eye(4, dtype=np.float32)
        rotation[:3, 0] = right
        rotation[:3, 1] = up
        rotation[:3, 2] = -forward
        
        translation = np.eye(4, dtype=np.float32)
        translation[3, :3] = -self.position
        
        return rotation @ translation
    
    def get_view_projection_matrix(self, intrinsics: CameraIntrinsics) -> np.ndarray:
        """Calculate combined view-projection matrix."""
        view = self.get_view_matrix()
        projection = intrinsics.get_projection_matrix()
        return projection @ view
    
    def look_at(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = None):
        """Set camera to look at target."""
        self.position = position
        self.target = target
        if up is not None:
            self.up = up
        
        # Calculate rotation from look-at
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up if up is None else up)
        right = right / np.linalg.norm(right)
        
        new_up = np.cross(right, forward)
        
        # Convert to quaternion
        rot_matrix = np.eye(3, dtype=np.float32)
        rot_matrix[:, 0] = right
        rot_matrix[:, 1] = new_up
        rot_matrix[:, 2] = forward
        
        self.rotation = self._matrix_to_quaternion(rot_matrix)
    
    def rotate(self, yaw: float, pitch: float, roll: float = 0.0):
        """Rotate camera by Euler angles (degrees)."""
        # Convert to quaternion
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)
        
        q = np.array([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ], dtype=np.float32)
        
        # Apply rotation
        self.rotation = self._quaternion_multiply(q, self.rotation)
        
        # Update target based on new orientation
        forward = self._quaternion_rotate(self.rotation, np.array([0, 0, -1], dtype=np.float32))
        self.target = self.position + forward
    
    def translate(self, delta: np.ndarray, local_space: bool = True):
        """Translate camera position."""
        if local_space:
            # Rotate delta by camera orientation
            delta_rotated = self._quaternion_rotate(self.rotation, delta)
            self.position += delta_rotated
            self.target += delta_rotated
        else:
            self.position += delta
            self.target += delta
    
    def zoom(self, amount: float):
        """Zoom camera (dolly zoom)."""
        forward = self.target - self.position
        distance = np.linalg.norm(forward)
        
        if distance > 0.1 or amount < 0:  # Prevent going through target
            new_distance = distance * (1.0 - amount * 0.1)
            new_distance = max(0.1, min(1000.0, new_distance))
            
            direction = forward / distance
            self.position = self.target - direction * new_distance
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
        
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ], dtype=np.float32)
    
    def _quaternion_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector by quaternion."""
        qv = np.array([q[0], q[1], q[2]], dtype=np.float32)
        qw = q[3]
        
        uv = np.cross(qv, v)
        uuv = np.cross(qv, uv)
        
        return v + 2.0 * (qw * uv + uuv)
    
    def _matrix_to_quaternion(self, m: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion."""
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        
        return np.array([x, y, z, w], dtype=np.float32)


class CameraController:
    """Main camera controller class."""
    
    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        state: Optional[CameraState] = None,
        control_mode: CameraControlMode = CameraControlMode.ORBIT,
        sensitivity: float = 0.1,
        move_speed: float = 2.0,
        zoom_speed: float = 1.0
    ):
        """Initialize camera controller.
        
        Args:
            intrinsics: Camera intrinsic parameters
            state: Initial camera state
            control_mode: Control mode
            sensitivity: Mouse sensitivity
            move_speed: Movement speed
            zoom_speed: Zoom speed
        """
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.state = state or CameraState()
        self.control_mode = control_mode
        self.sensitivity = sensitivity
        self.move_speed = move_speed
        self.zoom_speed = zoom_speed
        
        # Input state
        self.mouse_position = (0, 0)
        self.mouse_delta = (0, 0)
        self.keys_pressed = set()
        self.is_mouse_dragging = False
        
        # Constraints
        self.min_distance = 0.1
        self.max_distance = 1000.0
        self.min_pitch = -89.0
        self.max_pitch = 89.0
        
        # Smoothing
        self.position_velocity = np.zeros(3, dtype=np.float32)
        self.rotation_velocity = np.zeros(3, dtype=np.float32)
        self.smooth_factor = 0.9
        
    def handle_mouse_move(self, x: float, y: float, dx: float, dy: float):
        """Handle mouse movement.
        
        Args:
            x: Current mouse x
            y: Current mouse y
            dx: Delta x
            dy: Delta y
        """
        self.mouse_position = (x, y)
        self.mouse_delta = (dx, dy)
        
        if self.is_mouse_dragging:
            if self.control_mode == CameraControlMode.ORBIT:
                self._handle_orbit_rotation(dx, dy)
            elif self.control_mode == CameraControlMode.FLY:
                self._handle_fly_rotation(dx, dy)
    
    def handle_mouse_drag(self, x: float, y: float, button: int, is_dragging: bool):
        """Handle mouse drag.
        
        Args:
            x: Mouse x
            y: Mouse y
            button: Mouse button
            is_dragging: Whether dragging
        """
        self.is_mouse_dragging = is_dragging
        self.mouse_position = (x, y)
        
        if button == 0:  # Left button - rotation
            self.control_mode = CameraControlMode.ORBIT
        elif button == 2:  # Middle button - pan
            self.control_mode = CameraControlMode.TRACK
    
    def handle_mouse_wheel(self, delta: float):
        """Handle mouse wheel.
        
        Args:
            delta: Wheel delta
        """
        if self.control_mode == CameraControlMode.ORBIT:
            self.state.zoom(delta * self.zoom_speed)
        elif self.control_mode == CameraControlMode.DOLLY:
            forward = self.state.target - self.state.position
            distance = np.linalg.norm(forward)
            new_distance = distance + delta * self.move_speed * 0.1
            new_distance = np.clip(new_distance, self.min_distance, self.max_distance)
            
            if distance > 0:
                direction = forward / distance
                self.state.position = self.state.target - direction * new_distance
    
    def handle_key_press(self, key: str, is_pressed: bool):
        """Handle key press.
        
        Args:
            key: Key identifier
            is_pressed: Whether key is pressed
        """
        if is_pressed:
            self.keys_pressed.add(key)
        else:
            self.keys_pressed.discard(key)
    
    def update(self, delta_time: float):
        """Update camera state.
        
        Args:
            delta_time: Time since last update in seconds
        """
        # Handle keyboard movement
        if self.control_mode in [CameraControlMode.FLY, CameraControlMode.FPS]:
            self._handle_keyboard_movement(delta_time)
        
        # Apply smoothing
        self._apply_smoothing(delta_time)
        
        # Constrain camera
        self._constrain_camera()
    
    def _handle_orbit_rotation(self, dx: float, dy: float):
        """Handle orbit rotation."""
        # Convert to yaw and pitch
        yaw = dx * self.sensitivity
        pitch = dy * self.sensitivity
        
        # Get current spherical coordinates
        forward = self.state.target - self.state.position
        distance = np.linalg.norm(forward)
        
        if distance > 0:
            # Convert to spherical coordinates
            forward = forward / distance
            
            # Calculate current angles
            current_yaw = math.atan2(forward[0], forward[2])
            current_pitch = math.asin(forward[1])
            
            # Apply rotation
            new_yaw = current_yaw - math.radians(yaw)
            new_pitch = current_pitch - math.radians(pitch)
            new_pitch = np.clip(new_pitch, 
                               math.radians(self.min_pitch),
                               math.radians(self.max_pitch))
            
            # Convert back to cartesian
            new_forward = np.array([
                math.sin(new_yaw) * math.cos(new_pitch),
                math.sin(new_pitch),
                math.cos(new_yaw) * math.cos(new_pitch)
            ], dtype=np.float32)
            
            # Update position
            self.state.position = self.state.target - new_forward * distance
            
            # Update up vector (keep it mostly upward)
            right = np.cross(new_forward, np.array([0, 1, 0], dtype=np.float32))
            if np.linalg.norm(right) > 0:
                right = right / np.linalg.norm(right)
                self.state.up = np.cross(right, new_forward)
    
    def _handle_fly_rotation(self, dx: float, dy: float):
        """Handle fly/FPS rotation."""
        yaw = dx * self.sensitivity
        pitch = dy * self.sensitivity
        
        # Rotate camera
        self.state.rotate(yaw, pitch)
    
    def _handle_keyboard_movement(self, delta_time: float):
        """Handle keyboard-based movement."""
        move_vector = np.zeros(3, dtype=np.float32)
        
        # Forward/backward
        if 'w' in self.keys_pressed or 'W' in self.keys_pressed:
            move_vector[2] -= 1
        if 's' in self.keys_pressed or 'S' in self.keys_pressed:
            move_vector[2] += 1
        
        # Left/right
        if 'a' in self.keys_pressed or 'A' in self.keys_pressed:
            move_vector[0] -= 1
        if 'd' in self.keys_pressed or 'D' in self.keys_pressed:
            move_vector[0] += 1
        
        # Up/down
        if 'e' in self.keys_pressed or 'E' in self.keys_pressed:
            move_vector[1] += 1
        if 'q' in self.keys_pressed or 'Q' in self.keys_pressed:
            move_vector[1] -= 1
        
        # Normalize if moving diagonally
        norm = np.linalg.norm(move_vector)
        if norm > 0:
            move_vector = move_vector / norm
        
        # Apply movement
        if np.any(move_vector != 0):
            speed = self.move_speed * delta_time
            self.state.translate(move_vector * speed, local_space=True)
            
            if self.control_mode == CameraControlMode.FPS:
                # Keep target in front
                forward = self.state._quaternion_rotate(
                    self.state.rotation, 
                    np.array([0, 0, -1], dtype=np.float32)
                )
                self.state.target = self.state.position + forward
    
    def _apply_smoothing(self, delta_time: float):
        """Apply smoothing to camera movement."""
        # Exponential moving average for smoothing
        if delta_time > 0:
            alpha = 1.0 - math.exp(-delta_time * 10.0)  # Adjust 10.0 for smoothing strength
            self.position_velocity = self.position_velocity * (1.0 - alpha)
            self.rotation_velocity = self.rotation_velocity * (1.0 - alpha)
    
    def _constrain_camera(self):
        """Apply constraints to camera."""
        # Constrain pitch
        forward = self.state.target - self.state.position
        distance = np.linalg.norm(forward)
        
        if distance > 0:
            forward = forward / distance
            pitch = math.asin(forward[1])
            
            if pitch < math.radians(self.min_pitch):
                forward[1] = math.sin(math.radians(self.min_pitch))
                forward = forward / np.linalg.norm(forward)
                self.state.position = self.state.target - forward * distance
            elif pitch > math.radians(self.max_pitch):
                forward[1] = math.sin(math.radians(self.max_pitch))
                forward = forward / np.linalg.norm(forward)
                self.state.position = self.state.target - forward * distance
        
        # Constrain distance
        distance = np.linalg.norm(self.state.target - self.state.position)
        if distance < self.min_distance:
            direction = (self.state.target - self.state.position) / distance
            self.state.position = self.state.target - direction * self.min_distance
        elif distance > self.max_distance:
            direction = (self.state.target - self.state.position) / distance
            self.state.position = self.state.target - direction * self.max_distance
    
    def get_view_matrix(self) -> np.ndarray:
        """Get current view matrix."""
        return self.state.get_view_matrix()
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get current projection matrix."""
        return self.intrinsics.get_projection_matrix()
    
    def get_view_projection_matrix(self) -> np.ndarray:
        """Get combined view-projection matrix."""
        return self.state.get_view_projection_matrix(self.intrinsics)
    
    def screen_to_world_ray(self, screen_x: float, screen_y: float, 
                           screen_width: float, screen_height: float) -> Tuple[np.ndarray, np.ndarray]:
        """Convert screen coordinates to world space ray.
        
        Args:
            screen_x: Screen x coordinate (0 to width)
            screen_y: Screen y coordinate (0 to height)
            screen_width: Screen width
            screen_height: Screen height
            
        Returns:
            Tuple of (ray_origin, ray_direction)
        """
        # Normalize device coordinates
        x = (2.0 * screen_x) / screen_width - 1.0
        y = 1.0 - (2.0 * screen_y) / screen_height
        
        # Create ray in clip space
        ray_clip = np.array([x, y, -1.0, 1.0], dtype=np.float32)
        
        # Transform to eye space
        inv_projection = np.linalg.inv(self.get_projection_matrix())
        ray_eye = inv_projection @ ray_clip
        ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0], dtype=np.float32)
        
        # Transform to world space
        inv_view = np.linalg.inv(self.get_view_matrix())
        ray_world = inv_view @ ray_eye
        ray_dir = np.array([ray_world[0], ray_world[1], ray_world[2]], dtype=np.float32)
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        return self.state.position, ray_dir
    
    def save_state(self, filepath: Union[str, Path]):
        """Save camera state to file.
        
        Args:
            filepath: Path to save file
        """
        state = {
            "intrinsics": self.intrinsics.to_dict(),
            "state": {
                "position": self.state.position.tolist(),
                "target": self.state.target.tolist(),
                "up": self.state.up.tolist(),
                "rotation": self.state.rotation.tolist(),
                "projection_type": self.state.projection_type.value
            },
            "control_mode": self.control_mode.value,
            "sensitivity": self.sensitivity,
            "move_speed": self.move_speed,
            "zoom_speed": self.zoom_speed
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: Union[str, Path]):
        """Load camera state from file.
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.intrinsics = CameraIntrinsics.from_dict(state["intrinsics"])
        
        camera_state = state["state"]
        self.state = CameraState(
            position=np.array(camera_state["position"], dtype=np.float32),
            target=np.array(camera_state["target"], dtype=np.float32),
            up=np.array(camera_state["up"], dtype=np.float32),
            rotation=np.array(camera_state["rotation"], dtype=np.float32),
            projection_type=CameraProjection(camera_state["projection_type"])
        )
        
        self.control_mode = CameraControlMode(state["control_mode"])
        self.sensitivity = state.get("sensitivity", 0.1)
        self.move_speed = state.get("move_speed", 2.0)
        self.zoom_speed = state.get("zoom_speed", 1.0)
    
    def reset(self):
        """Reset camera to default state."""
        self.intrinsics = CameraIntrinsics()
        self.state = CameraState()
        self.control_mode = CameraControlMode.ORBIT
        self.sensitivity = 0.1
        self.move_speed = 2.0
        self.zoom_speed = 1.0
        
        # Clear input state
        self.mouse_position = (0, 0)
        self.mouse_delta = (0, 0)
        self.keys_pressed.clear()
        self.is_mouse_dragging = False
