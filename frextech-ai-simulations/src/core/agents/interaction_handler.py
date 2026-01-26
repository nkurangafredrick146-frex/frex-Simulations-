"""
Interaction Handler for agent-object and agent-agent interactions in 3D environments.
Supports manipulation, modification, and complex interaction behaviors.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

# Configure logging
logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """Types of interactions."""
    GRAB = "grab"
    PUSH = "push"
    PULL = "pull"
    ROTATE = "rotate"
    SCALE = "scale"
    DEFORM = "deform"
    ATTACH = "attach"
    DETACH = "detach"
    COMBINE = "combine"
    SEPARATE = "separate"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"

class InteractionState(Enum):
    """States of an interaction."""
    IDLE = "idle"
    PREPARING = "preparing"
    EXECUTING = "executing"
    HOLDING = "holding"
    RELEASING = "releasing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class InteractionResult:
    """Result of an interaction."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    energy_cost: float = 0.0

@dataclass
class InteractionConstraint:
    """Constraints for interactions."""
    max_force: float = 100.0  # N
    max_torque: float = 50.0  # N·m
    max_distance: float = 2.0  # m
    min_distance: float = 0.1  # m
    max_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 180.0  # deg/s
    requires_line_of_sight: bool = True
    requires_proximity: bool = True

class ObjectManipulator:
    """Handles object manipulation operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.grasped_objects = {}
        self.manipulation_history = []
        
        # Physical properties
        self.max_grasp_force = self.config.get('max_grasp_force', 50.0)
        self.grasp_stiffness = self.config.get('grasp_stiffness', 1000.0)
        self.grasp_damping = self.config.get('grasp_damping', 100.0)
        
        # Precision
        self.position_tolerance = self.config.get('position_tolerance', 0.01)  # m
        self.rotation_tolerance = self.config.get('rotation_tolerance', 1.0)  # deg
        
    def grasp_object(self, 
                    object_id: str, 
                    position: np.ndarray,
                    orientation: np.ndarray,
                    object_properties: Dict[str, Any]) -> InteractionResult:
        """
        Grasp an object.
        
        Args:
            object_id: Unique identifier for the object
            position: Grasp position in world coordinates (3,)
            orientation: Grasp orientation as quaternion (4,) or rotation matrix (3,3)
            object_properties: Physical properties of the object
            
        Returns:
            Interaction result
        """
        try:
            # Check if object is already grasped
            if object_id in self.grasped_objects:
                return InteractionResult(
                    success=False,
                    message=f"Object {object_id} is already grasped"
                )
            
            # Calculate required grasp force based on object properties
            mass = object_properties.get('mass', 1.0)
            friction = object_properties.get('friction', 0.5)
            required_force = mass * 9.81 / friction
            
            if required_force > self.max_grasp_force:
                return InteractionResult(
                    success=False,
                    message=f"Insufficient grasp force. Required: {required_force:.1f}N, Max: {self.max_grasp_force:.1f}N"
                )
            
            # Store grasp information
            grasp_info = {
                'position': np.array(position, dtype=np.float32),
                'orientation': self._normalize_orientation(orientation),
                'grasp_force': required_force,
                'object_properties': object_properties,
                'grasp_time': time.time(),
                'initial_object_position': object_properties.get('position', position),
                'initial_object_orientation': object_properties.get('orientation', orientation)
            }
            
            self.grasped_objects[object_id] = grasp_info
            
            # Log the grasp
            self.manipulation_history.append({
                'action': 'grasp',
                'object_id': object_id,
                'timestamp': time.time(),
                'parameters': grasp_info
            })
            
            logger.info(f"Successfully grasped object {object_id} with force {required_force:.1f}N")
            
            return InteractionResult(
                success=True,
                message=f"Successfully grasped object {object_id}",
                data={'grasp_force': required_force, 'grasp_info': grasp_info}
            )
            
        except Exception as e:
            logger.error(f"Failed to grasp object {object_id}: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to grasp object {object_id}: {str(e)}"
            )
    
    def release_object(self, object_id: str, release_velocity: np.ndarray = None) -> InteractionResult:
        """
        Release a grasped object.
        
        Args:
            object_id: ID of object to release
            release_velocity: Optional release velocity (3,)
            
        Returns:
            Interaction result
        """
        try:
            if object_id not in self.grasped_objects:
                return InteractionResult(
                    success=False,
                    message=f"Object {object_id} is not grasped"
                )
            
            grasp_info = self.grasped_objects.pop(object_id)
            
            # Calculate final state
            final_position = grasp_info['position']
            final_orientation = grasp_info['orientation']
            
            if release_velocity is not None:
                # Apply release velocity
                release_info = {
                    'velocity': np.array(release_velocity, dtype=np.float32),
                    'position': final_position,
                    'orientation': final_orientation
                }
            else:
                release_info = None
            
            # Log the release
            self.manipulation_history.append({
                'action': 'release',
                'object_id': object_id,
                'timestamp': time.time(),
                'release_info': release_info
            })
            
            logger.info(f"Released object {object_id}")
            
            return InteractionResult(
                success=True,
                message=f"Released object {object_id}",
                data={'release_info': release_info}
            )
            
        except Exception as e:
            logger.error(f"Failed to release object {object_id}: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to release object {object_id}: {str(e)}"
            )
    
    def move_object(self, 
                   object_id: str, 
                   target_position: np.ndarray,
                   target_orientation: np.ndarray = None,
                   max_speed: float = 1.0) -> InteractionResult:
        """
        Move a grasped object to target pose.
        
        Args:
            object_id: ID of object to move
            target_position: Target position (3,)
            target_orientation: Target orientation
            max_speed: Maximum movement speed (m/s)
            
        Returns:
            Interaction result
        """
        try:
            if object_id not in self.grasped_objects:
                return InteractionResult(
                    success=False,
                    message=f"Object {object_id} is not grasped"
                )
            
            grasp_info = self.grasped_objects[object_id]
            current_position = grasp_info['position']
            current_orientation = grasp_info['orientation']
            
            # Calculate movement
            displacement = np.array(target_position) - current_position
            distance = np.linalg.norm(displacement)
            
            if distance > 0:
                # Calculate direction and speed
                direction = displacement / distance
                speed = min(max_speed, distance * 10)  # Simple control
                
                # Update position
                new_position = current_position + direction * speed * 0.033  # Assuming 30Hz
                grasp_info['position'] = new_position
                
                # Update orientation if provided
                if target_orientation is not None:
                    target_orient_norm = self._normalize_orientation(target_orientation)
                    # Interpolate orientation
                    grasp_info['orientation'] = self._interpolate_orientation(
                        current_orientation, target_orient_norm, 0.1
                    )
                
                # Update grasp force based on movement
                object_mass = grasp_info['object_properties'].get('mass', 1.0)
                acceleration = speed * 10  # Approximate
                required_force = object_mass * acceleration
                grasp_info['grasp_force'] = max(grasp_info['grasp_force'], required_force)
                
                # Log movement
                self.manipulation_history.append({
                    'action': 'move',
                    'object_id': object_id,
                    'timestamp': time.time(),
                    'from_position': current_position,
                    'to_position': new_position,
                    'distance': distance
                })
                
                logger.debug(f"Moved object {object_id} by {distance:.3f}m")
                
                return InteractionResult(
                    success=True,
                    message=f"Moved object {object_id}",
                    data={
                        'new_position': new_position,
                        'new_orientation': grasp_info['orientation'],
                        'distance_moved': distance,
                        'grasp_force': grasp_info['grasp_force']
                    }
                )
            else:
                return InteractionResult(
                    success=True,
                    message=f"Object {object_id} already at target position"
                )
                
        except Exception as e:
            logger.error(f"Failed to move object {object_id}: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to move object {object_id}: {str(e)}"
            )
    
    def apply_force(self, 
                   object_id: str, 
                   force: np.ndarray,
                   application_point: np.ndarray = None,
                   duration: float = 0.1) -> InteractionResult:
        """
        Apply force to an object.
        
        Args:
            object_id: ID of object to apply force to
            force: Force vector in world coordinates (3,) [N]
            application_point: Point of application in world coordinates (3,)
            duration: Duration to apply force (s)
            
        Returns:
            Interaction result
        """
        try:
            # Calculate resulting motion
            if object_id in self.grasped_objects:
                grasp_info = self.grasped_objects[object_id]
                object_mass = grasp_info['object_properties'].get('mass', 1.0)
            else:
                # Assume some default properties for non-grasped objects
                object_mass = 1.0
            
            # Calculate acceleration (F = ma)
            acceleration = force / object_mass
            
            # Calculate velocity change (Δv = a * t)
            velocity_change = acceleration * duration
            
            # Calculate displacement (s = v₀t + ½at²)
            # Assuming initial velocity is 0
            displacement = 0.5 * acceleration * duration**2
            
            result_data = {
                'applied_force': force.tolist(),
                'acceleration': acceleration.tolist(),
                'velocity_change': velocity_change.tolist(),
                'displacement': displacement.tolist(),
                'duration': duration
            }
            
            # Log force application
            self.manipulation_history.append({
                'action': 'apply_force',
                'object_id': object_id,
                'timestamp': time.time(),
                'force': force.tolist(),
                'result': result_data
            })
            
            logger.info(f"Applied force {force} to object {object_id}")
            
            return InteractionResult(
                success=True,
                message=f"Applied force to object {object_id}",
                data=result_data,
                duration=duration,
                energy_cost=np.linalg.norm(force) * duration
            )
            
        except Exception as e:
            logger.error(f"Failed to apply force to object {object_id}: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to apply force to object {object_id}: {str(e)}"
            )
    
    def _normalize_orientation(self, orientation: Union[np.ndarray, List]) -> np.ndarray:
        """Normalize orientation to quaternion."""
        orientation = np.array(orientation, dtype=np.float32)
        
        if orientation.shape == (3, 3):
            # Rotation matrix to quaternion
            rot = R.from_matrix(orientation.reshape(3, 3))
            return rot.as_quat()
        elif orientation.shape == (4,):
            # Already quaternion, ensure normalized
            norm = np.linalg.norm(orientation)
            if norm > 0:
                return orientation / norm
            return np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            raise ValueError(f"Invalid orientation shape: {orientation.shape}")
    
    def _interpolate_orientation(self, 
                                q1: np.ndarray, 
                                q2: np.ndarray, 
                                t: float) -> np.ndarray:
        """Spherical linear interpolation between two quaternions."""
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Dot product
        dot = np.dot(q1, q2)
        
        # If dot is negative, quaternions have opposite handedness
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        
        # Clamp dot to [-1, 1]
        dot = np.clip(dot, -1.0, 1.0)
        
        # Calculate interpolation factor
        theta = np.arccos(dot) * t
        q3 = q2 - q1 * dot
        q3 = q3 / np.linalg.norm(q3)
        
        return q1 * np.cos(theta) + q3 * np.sin(theta)

class SceneModifier:
    """Modifies scene geometry and properties."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.modification_history = []
        
        # Modification parameters
        self.max_deformation = self.config.get('max_deformation', 0.1)  # m
        self.modification_strength = self.config.get('modification_strength', 1.0)
        self.smooth_radius = self.config.get('smooth_radius', 0.1)  # m
        
    def deform_surface(self, 
                      surface_points: np.ndarray,
                      deformation_center: np.ndarray,
                      deformation_vector: np.ndarray,
                      radius: float = 0.1) -> Tuple[np.ndarray, InteractionResult]:
        """
        Deform a surface by applying displacement.
        
        Args:
            surface_points: Array of surface points (N, 3)
            deformation_center: Center of deformation (3,)
            deformation_vector: Deformation direction and magnitude (3,)
            radius: Influence radius of deformation
            
        Returns:
            Tuple of (modified_points, interaction_result)
        """
        try:
            modified_points = surface_points.copy()
            deformation_magnitude = np.linalg.norm(deformation_vector)
            
            if deformation_magnitude == 0:
                return modified_points, InteractionResult(
                    success=False,
                    message="Zero deformation vector"
                )
            
            # Limit deformation magnitude
            deformation_magnitude = min(deformation_magnitude, self.max_deformation)
            deformation_vector = deformation_vector / np.linalg.norm(deformation_vector) * deformation_magnitude
            
            # Calculate deformation for each point
            for i in range(len(surface_points)):
                point = surface_points[i]
                distance = np.linalg.norm(point - deformation_center)
                
                if distance < radius:
                    # Calculate deformation weight (Gaussian falloff)
                    weight = np.exp(-(distance**2) / (2 * (radius/3)**2))
                    
                    # Apply deformation
                    modified_points[i] += deformation_vector * weight * self.modification_strength
            
            # Log deformation
            self.modification_history.append({
                'action': 'deform_surface',
                'timestamp': time.time(),
                'deformation_center': deformation_center.tolist(),
                'deformation_vector': deformation_vector.tolist(),
                'radius': radius,
                'num_points_modified': len(surface_points)
            })
            
            logger.info(f"Deformed surface with {len(surface_points)} points")
            
            return modified_points, InteractionResult(
                success=True,
                message=f"Deformed surface with {len(surface_points)} points",
                data={
                    'num_points_modified': len(surface_points),
                    'max_deformation': deformation_magnitude,
                    'influence_radius': radius
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to deform surface: {e}")
            return surface_points, InteractionResult(
                success=False,
                message=f"Failed to deform surface: {str(e)}"
            )
    
    def add_object(self, 
                  object_template: Dict[str, Any],
                  position: np.ndarray,
                  orientation: np.ndarray,
                  scale: float = 1.0) -> InteractionResult:
        """
        Add a new object to the scene.
        
        Args:
            object_template: Template defining object properties
            position: Object position (3,)
            orientation: Object orientation
            scale: Object scale
            
        Returns:
            Interaction result
        """
        try:
            # Create object instance from template
            object_id = object_template.get('id', f"object_{len(self.modification_history)}")
            
            # Apply transformations
            object_instance = {
                'id': object_id,
                'type': object_template['type'],
                'position': np.array(position, dtype=np.float32).tolist(),
                'orientation': self._normalize_orientation(orientation).tolist(),
                'scale': scale,
                'properties': object_template.get('properties', {}),
                'geometry': self._scale_geometry(object_template['geometry'], scale)
            }
            
            # Log addition
            self.modification_history.append({
                'action': 'add_object',
                'timestamp': time.time(),
                'object_id': object_id,
                'object_type': object_template['type'],
                'position': position.tolist(),
                'scale': scale
            })
            
            logger.info(f"Added object {object_id} of type {object_template['type']}")
            
            return InteractionResult(
                success=True,
                message=f"Added object {object_id}",
                data={'object_instance': object_instance}
            )
            
        except Exception as e:
            logger.error(f"Failed to add object: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to add object: {str(e)}"
            )
    
    def remove_object(self, object_id: str) -> InteractionResult:
        """
        Remove an object from the scene.
        
        Args:
            object_id: ID of object to remove
            
        Returns:
            Interaction result
        """
        try:
            # Log removal
            self.modification_history.append({
                'action': 'remove_object',
                'timestamp': time.time(),
                'object_id': object_id
            })
            
            logger.info(f"Removed object {object_id}")
            
            return InteractionResult(
                success=True,
                message=f"Removed object {object_id}",
                data={'object_id': object_id}
            )
            
        except Exception as e:
            logger.error(f"Failed to remove object {object_id}: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to remove object {object_id}: {str(e)}"
            )
    
    def modify_material(self, 
                       object_id: str,
                       material_properties: Dict[str, Any]) -> InteractionResult:
        """
        Modify material properties of an object.
        
        Args:
            object_id: ID of object to modify
            material_properties: New material properties
            
        Returns:
            Interaction result
        """
        try:
            # Log material modification
            self.modification_history.append({
                'action': 'modify_material',
                'timestamp': time.time(),
                'object_id': object_id,
                'material_properties': material_properties
            })
            
            logger.info(f"Modified material of object {object_id}")
            
            return InteractionResult(
                success=True,
                message=f"Modified material of object {object_id}",
                data={'material_properties': material_properties}
            )
            
        except Exception as e:
            logger.error(f"Failed to modify material of object {object_id}: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to modify material of object {object_id}: {str(e)}"
            )
    
    def _scale_geometry(self, geometry: Dict[str, Any], scale: float) -> Dict[str, Any]:
        """Scale geometry by given factor."""
        scaled_geometry = geometry.copy()
        
        if 'vertices' in geometry:
            vertices = np.array(geometry['vertices'])
            scaled_geometry['vertices'] = (vertices * scale).tolist()
        
        if 'bounding_box' in geometry:
            bbox = geometry['bounding_box']
            scaled_geometry['bounding_box'] = {
                'min': [coord * scale for coord in bbox['min']],
                'max': [coord * scale for coord in bbox['max']]
            }
        
        return scaled_geometry
    
    def _normalize_orientation(self, orientation):
        """Normalize orientation (reuse from ObjectManipulator)."""
        from scipy.spatial.transform import Rotation as R
        
        orientation = np.array(orientation, dtype=np.float32)
        
        if orientation.shape == (3, 3):
            rot = R.from_matrix(orientation.reshape(3, 3))
            return rot.as_quat()
        elif orientation.shape == (4,):
            norm = np.linalg.norm(orientation)
            if norm > 0:
                return orientation / norm
            return np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            return np.array([1, 0, 0, 0], dtype=np.float32)

class AgentController:
    """Controls agent behaviors and interactions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agents = {}
        self.behaviors = {}
        self.interaction_queue = []
        
        # Control parameters
        self.update_rate = self.config.get('update_rate', 30.0)  # Hz
        self.max_queue_size = self.config.get('max_queue_size', 100)
        self.priority_threshold = self.config.get('priority_threshold', 0.5)
        
    def register_agent(self, agent_id: str, agent_properties: Dict[str, Any]) -> bool:
        """
        Register an agent with the controller.
        
        Args:
            agent_id: Unique agent identifier
            agent_properties: Agent capabilities and properties
            
        Returns:
            Success status
        """
        try:
            self.agents[agent_id] = {
                'properties': agent_properties,
                'state': 'idle',
                'current_interaction': None,
                'capabilities': agent_properties.get('capabilities', []),
                'position': agent_properties.get('position', [0, 0, 0]),
                'orientation': agent_properties.get('orientation', [1, 0, 0, 0])
            }
            
            logger.info(f"Registered agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    def queue_interaction(self, 
                         interaction_type: InteractionType,
                         agent_id: str,
                         target_id: str,
                         parameters: Dict[str, Any],
                         priority: float = 0.5) -> str:
        """
        Queue an interaction for execution.
        
        Args:
            interaction_type: Type of interaction
            agent_id: ID of agent performing interaction
            target_id: ID of target (object or agent)
            parameters: Interaction parameters
            priority: Interaction priority (0.0 to 1.0)
            
        Returns:
            Interaction ID
        """
        try:
            # Check if agent exists
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not registered")
            
            # Check if agent has required capabilities
            agent_capabilities = self.agents[agent_id]['capabilities']
            required_capability = interaction_type.value
            
            if required_capability not in agent_capabilities:
                raise ValueError(
                    f"Agent {agent_id} lacks capability '{required_capability}'. "
                    f"Available: {agent_capabilities}"
                )
            
            # Create interaction
            interaction_id = f"interaction_{len(self.interaction_queue)}_{interaction_type.value}"
            
            interaction = {
                'id': interaction_id,
                'type': interaction_type,
                'agent_id': agent_id,
                'target_id': target_id,
                'parameters': parameters,
                'priority': max(0.0, min(1.0, priority)),
                'status': 'queued',
                'timestamp': time.time(),
                'attempts': 0,
                'max_attempts': 3
            }
            
            # Add to queue
            self.interaction_queue.append(interaction)
            
            # Sort by priority (highest first)
            self.interaction_queue.sort(key=lambda x: x['priority'], reverse=True)
            
            # Trim queue if too large
            if len(self.interaction_queue) > self.max_queue_size:
                self.interaction_queue = self.interaction_queue[:self.max_queue_size]
            
            logger.info(f"Queued interaction {interaction_id} with priority {priority:.2f}")
            
            return interaction_id
            
        except Exception as e:
            logger.error(f"Failed to queue interaction: {e}")
            return None
    
    def execute_interaction(self, interaction_id: str) -> InteractionResult:
        """
        Execute a queued interaction.
        
        Args:
            interaction_id: ID of interaction to execute
            
        Returns:
            Interaction result
        """
        try:
            # Find interaction
            interaction = None
            for i, inter in enumerate(self.interaction_queue):
                if inter['id'] == interaction_id:
                    interaction = inter
                    interaction_index = i
                    break
            
            if interaction is None:
                return InteractionResult(
                    success=False,
                    message=f"Interaction {interaction_id} not found in queue"
                )
            
            # Check if agent is available
            agent_id = interaction['agent_id']
            agent = self.agents[agent_id]
            
            if agent['state'] != 'idle':
                return InteractionResult(
                    success=False,
                    message=f"Agent {agent_id} is busy (state: {agent['state']})"
                )
            
            # Update interaction and agent state
            interaction['status'] = 'executing'
            interaction['attempts'] += 1
            interaction['start_time'] = time.time()
            
            agent['state'] = 'busy'
            agent['current_interaction'] = interaction_id
            
            # Execute based on interaction type
            result = self._execute_interaction_type(interaction)
            
            # Update states
            interaction['status'] = 'completed' if result.success else 'failed'
            interaction['end_time'] = time.time()
            interaction['duration'] = interaction['end_time'] - interaction['start_time']
            interaction['result'] = result
            
            agent['state'] = 'idle'
            agent['current_interaction'] = None
            
            # Remove from queue if completed or max attempts reached
            if result.success or interaction['attempts'] >= interaction['max_attempts']:
                self.interaction_queue.pop(interaction_index)
            
            logger.info(f"Executed interaction {interaction_id}: {result.message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute interaction {interaction_id}: {e}")
            
            # Reset agent state
            if 'agent_id' in locals() and agent_id in self.agents:
                self.agents[agent_id]['state'] = 'idle'
                self.agents[agent_id]['current_interaction'] = None
            
            return InteractionResult(
                success=False,
                message=f"Failed to execute interaction {interaction_id}: {str(e)}"
            )
    
    def _execute_interaction_type(self, interaction: Dict[str, Any]) -> InteractionResult:
        """Execute specific interaction type."""
        interaction_type = interaction['type']
        parameters = interaction['parameters']
        
        # This would be connected to actual manipulation/modification systems
        # For now, return simulated results
        
        if interaction_type == InteractionType.GRAB:
            return InteractionResult(
                success=True,
                message=f"Grabbed {interaction['target_id']}",
                data={'force_applied': 25.0, 'grip_strength': 0.8}
            )
        elif interaction_type == InteractionType.PUSH:
            return InteractionResult(
                success=True,
                message=f"Pushed {interaction['target_id']}",
                data={'force': parameters.get('force', 10.0), 'distance': 0.5}
            )
        elif interaction_type == InteractionType.ROTATE:
            return InteractionResult(
                success=True,
                message=f"Rotated {interaction['target_id']}",
                data={'angle': parameters.get('angle', 45.0), 'axis': parameters.get('axis', [0, 1, 0])}
            )
        # Add more interaction types...
        
        return InteractionResult(
            success=False,
            message=f"Unsupported interaction type: {interaction_type}"
        )
    
    def update_agent_state(self, 
                          agent_id: str, 
                          position: np.ndarray,
                          orientation: np.ndarray,
                          velocity: np.ndarray = None) -> bool:
        """
        Update agent's state.
        
        Args:
            agent_id: Agent identifier
            position: New position (3,)
            orientation: New orientation
            velocity: Current velocity (3,)
            
        Returns:
            Success status
        """
        try:
            if agent_id not in self.agents:
                return False
            
            self.agents[agent_id]['position'] = np.array(position, dtype=np.float32)
            self.agents[agent_id]['orientation'] = np.array(orientation, dtype=np.float32)
            
            if velocity is not None:
                self.agents[agent_id]['velocity'] = np.array(velocity, dtype=np.float32)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update agent {agent_id} state: {e}")
            return False
    
    def get_agent_interactions(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get all interactions for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            List of interactions
        """
        return [
            inter for inter in self.interaction_queue 
            if inter['agent_id'] == agent_id
        ]
    
    def clear_queue(self, agent_id: str = None) -> int:
        """
        Clear interaction queue.
        
        Args:
            agent_id: If provided, only clear this agent's interactions
            
        Returns:
            Number of interactions cleared
        """
        if agent_id is None:
            count = len(self.interaction_queue)
            self.interaction_queue = []
            return count
        else:
            initial_count = len(self.interaction_queue)
            self.interaction_queue = [
                inter for inter in self.interaction_queue 
                if inter['agent_id'] != agent_id
            ]
            return initial_count - len(self.interaction_queue)

class InteractionHandler:
    """
    Main interaction handler coordinating object manipulation, scene modification,
    and agent control.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.manipulator = ObjectManipulator(
            self.config.get('manipulation', {})
        )
        self.modifier = SceneModifier(
            self.config.get('modification', {})
        )
        self.controller = AgentController(
            self.config.get('control', {})
        )
        
        # Interaction state
        self.active_interactions = {}
        self.interaction_history = []
        self.collision_handler = CollisionHandler()
        
        # Performance metrics
        self.metrics = {
            'total_interactions': 0,
            'successful_interactions': 0,
            'failed_interactions': 0,
            'average_duration': 0.0,
            'total_energy': 0.0
        }
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("InteractionHandler initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def register_agent(self, agent_id: str, agent_properties: Dict[str, Any]) -> bool:
        """
        Register an agent with the interaction system.
        
        Args:
            agent_id: Unique agent identifier
            agent_properties: Agent capabilities and properties
            
        Returns:
            Success status
        """
        return self.controller.register_agent(agent_id, agent_properties)
    
    def perform_interaction(self,
                          interaction_type: InteractionType,
                          agent_id: str,
                          target_id: str,
                          parameters: Dict[str, Any],
                          priority: float = 0.5) -> InteractionResult:
        """
        Perform an interaction immediately (blocks until complete).
        
        Args:
            interaction_type: Type of interaction
            agent_id: ID of agent performing interaction
            target_id: ID of target (object or agent)
            parameters: Interaction parameters
            priority: Interaction priority
            
        Returns:
            Interaction result
        """
        try:
            # Queue interaction
            interaction_id = self.controller.queue_interaction(
                interaction_type, agent_id, target_id, parameters, priority
            )
            
            if interaction_id is None:
                return InteractionResult(
                    success=False,
                    message="Failed to queue interaction"
                )
            
            # Execute immediately
            result = self.controller.execute_interaction(interaction_id)
            
            # Update metrics
            self._update_metrics(result)
            
            # Log to history
            self.interaction_history.append({
                'id': interaction_id,
                'type': interaction_type,
                'agent_id': agent_id,
                'target_id': target_id,
                'timestamp': time.time(),
                'result': result,
                'parameters': parameters
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to perform interaction: {e}")
            return InteractionResult(
                success=False,
                message=f"Failed to perform interaction: {str(e)}"
            )
    
    def schedule_interaction(self,
                           interaction_type: InteractionType,
                           agent_id: str,
                           target_id: str,
                           parameters: Dict[str, Any],
                           priority: float = 0.5) -> str:
        """
        Schedule an interaction for later execution.
        
        Args:
            interaction_type: Type of interaction
            agent_id: ID of agent performing interaction
            target_id: ID of target (object or agent)
            parameters: Interaction parameters
            priority: Interaction priority
            
        Returns:
            Interaction ID
        """
        return self.controller.queue_interaction(
            interaction_type, agent_id, target_id, parameters, priority
        )
    
    def execute_scheduled(self, max_interactions: int = 5) -> List[InteractionResult]:
        """
        Execute scheduled interactions.
        
        Args:
            max_interactions: Maximum number of interactions to execute
            
        Returns:
            List of interaction results
        """
        results = []
        
        for _ in range(min(max_interactions, len(self.controller.interaction_queue))):
            # Get highest priority interaction
            if not self.controller.interaction_queue:
                break
            
            interaction = self.controller.interaction_queue[0]
            
            # Execute if priority threshold met
            if interaction['priority'] >= self.controller.priority_threshold:
                result = self.controller.execute_interaction(interaction['id'])
                results.append(result)
                
                # Update metrics
                self._update_metrics(result)
                
                # Log to history
                self.interaction_history.append({
                    'id': interaction['id'],
                    'type': interaction['type'],
                    'agent_id': interaction['agent_id'],
                    'target_id': interaction['target_id'],
                    'timestamp': time.time(),
                    'result': result,
                    'parameters': interaction['parameters']
                })
        
        return results
    
    def grasp_object(self, 
                    agent_id: str,
                    object_id: str,
                    grasp_position: np.ndarray,
                    grasp_orientation: np.ndarray,
                    object_properties: Dict[str, Any]) -> InteractionResult:
        """
        Grasp an object using an agent.
        
        Args:
            agent_id: ID of agent performing grasp
            object_id: ID of object to grasp
            grasp_position: Grasp position
            grasp_orientation: Grasp orientation
            object_properties: Object properties
            
        Returns:
            Interaction result
        """
        # First, check if agent is in range
        agent_info = self.controller.agents.get(agent_id)
        if not agent_info:
            return InteractionResult(
                success=False,
                message=f"Agent {agent_id} not found"
            )
        
        # Check distance
        agent_position = agent_info['position']
        distance = np.linalg.norm(agent_position - grasp_position)
        
        if distance > self.manipulator.config.get('max_grasp_distance', 1.0):
            return InteractionResult(
                success=False,
                message=f"Object too far away: {distance:.2f}m > 1.0m"
            )
        
        # Perform grasp
        result = self.manipulator.grasp_object(
            object_id, grasp_position, grasp_orientation, object_properties
        )
        
        if result.success:
            # Register grasp with agent
            self.controller.update_agent_state(
                agent_id,
                position=agent_position,
                orientation=agent_info['orientation'],
                velocity=None
            )
            
            # Add to active interactions
            self.active_interactions[f"{agent_id}_{object_id}"] = {
                'type': 'grasp',
                'agent_id': agent_id,
                'object_id': object_id,
                'start_time': time.time()
            }
        
        return result
    
    def release_object(self, agent_id: str, object_id: str) -> InteractionResult:
        """
        Release a grasped object.
        
        Args:
            agent_id: ID of agent releasing object
            object_id: ID of object to release
            
        Returns:
            Interaction result
        """
        # Check if this grasp is active
        interaction_key = f"{agent_id}_{object_id}"
        if interaction_key not in self.active_interactions:
            return InteractionResult(
                success=False,
                message=f"No active grasp for object {object_id} by agent {agent_id}"
            )
        
        # Perform release
        result = self.manipulator.release_object(object_id)
        
        if result.success:
            # Remove from active interactions
            del self.active_interactions[interaction_key]
        
        return result
    
    def modify_scene(self,
                   modification_type: str,
                   parameters: Dict[str, Any],
                   agent_id: str = None) -> InteractionResult:
        """
        Modify the scene geometry or properties.
        
        Args:
            modification_type: Type of modification
            parameters: Modification parameters
            agent_id: Optional agent ID performing modification
            
        Returns:
            Interaction result
        """
        if modification_type == 'deform_surface':
            if 'surface_points' not in parameters or 'deformation_center' not in parameters:
                return InteractionResult(
                    success=False,
                    message="Missing required parameters for surface deformation"
                )
            
            # Call surface deformation
            modified_points, result = self.modifier.deform_surface(
                parameters['surface_points'],
                parameters['deformation_center'],
                parameters.get('deformation_vector', [0, 0.1, 0]),
                parameters.get('radius', 0.1)
            )
            
            if result.success:
                result.data['modified_points'] = modified_points
            
            return result
            
        elif modification_type == 'add_object':
            required = ['object_template', 'position']
            if not all(req in parameters for req in required):
                return InteractionResult(
                    success=False,
                    message=f"Missing required parameters for add_object: {required}"
                )
            
            return self.modifier.add_object(
                parameters['object_template'],
                parameters['position'],
                parameters.get('orientation', [1, 0, 0, 0]),
                parameters.get('scale', 1.0)
            )
            
        elif modification_type == 'remove_object':
            if 'object_id' not in parameters:
                return InteractionResult(
                    success=False,
                    message="Missing object_id parameter for remove_object"
                )
            
            return self.modifier.remove_object(parameters['object_id'])
            
        elif modification_type == 'modify_material':
            required = ['object_id', 'material_properties']
            if not all(req in parameters for req in required):
                return InteractionResult(
                    success=False,
                    message=f"Missing required parameters for modify_material: {required}"
                )
            
            return self.modifier.modify_material(
                parameters['object_id'],
                parameters['material_properties']
            )
            
        else:
            return InteractionResult(
                success=False,
                message=f"Unknown modification type: {modification_type}"
            )
    
    def check_collisions(self, 
                        agent_id: str = None,
                        object_id: str = None) -> List[Dict[str, Any]]:
        """
        Check for collisions.
        
        Args:
            agent_id: Optional specific agent to check
            object_id: Optional specific object to check
            
        Returns:
            List of collision information
        """
        return self.collision_handler.check_collisions(
            self.controller.agents,
            self.manipulator.grasped_objects,
            agent_id,
            object_id
        )
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get status of an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent status information
        """
        if agent_id not in self.controller.agents:
            return {}
        
        agent = self.controller.agents[agent_id]
        
        # Find active interactions for this agent
        active_interactions = [
            key for key in self.active_interactions.keys()
            if key.startswith(f"{agent_id}_")
        ]
        
        # Get queued interactions
        queued_interactions = self.controller.get_agent_interactions(agent_id)
        
        return {
            'id': agent_id,
            'state': agent['state'],
            'position': agent['position'],
            'orientation': agent['orientation'],
            'capabilities': agent['capabilities'],
            'active_interactions': active_interactions,
            'queued_interactions': queued_interactions,
            'current_interaction': agent['current_interaction']
        }
    
    def get_interaction_history(self, 
                               limit: int = 100,
                               agent_id: str = None,
                               interaction_type: str = None) -> List[Dict[str, Any]]:
        """
        Get interaction history with filters.
        
        Args:
            limit: Maximum number of entries to return
            agent_id: Filter by agent ID
            interaction_type: Filter by interaction type
            
        Returns:
            Filtered interaction history
        """
        filtered = self.interaction_history
        
        if agent_id:
            filtered = [h for h in filtered if h['agent_id'] == agent_id]
        
        if interaction_type:
            filtered = [h for h in filtered if h['type'].value == interaction_type]
        
        # Sort by timestamp (newest first)
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return filtered[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get interaction performance metrics.
        
        Returns:
            Performance metrics
        """
        metrics = self.metrics.copy()
        
        # Calculate success rate
        total = metrics['total_interactions']
        if total > 0:
            metrics['success_rate'] = metrics['successful_interactions'] / total
        else:
            metrics['success_rate'] = 0.0
        
        # Add current stats
        metrics['active_interactions'] = len(self.active_interactions)
        metrics['queued_interactions'] = len(self.controller.interaction_queue)
        metrics['registered_agents'] = len(self.controller.agents)
        
        return metrics
    
    def reset(self, clear_history: bool = False):
        """
        Reset the interaction handler.
        
        Args:
            clear_history: Whether to clear interaction history
        """
        # Reset components
        self.manipulator.grasped_objects.clear()
        self.manipulator.manipulation_history.clear()
        
        self.modifier.modification_history.clear()
        
        self.controller.agents.clear()
        self.controller.behaviors.clear()
        self.controller.interaction_queue.clear()
        
        # Reset state
        self.active_interactions.clear()
        
        if clear_history:
            self.interaction_history.clear()
            self.metrics = {
                'total_interactions': 0,
                'successful_interactions': 0,
                'failed_interactions': 0,
                'average_duration': 0.0,
                'total_energy': 0.0
            }
        
        logger.info("InteractionHandler reset")
    
    def _update_metrics(self, result: InteractionResult):
        """Update performance metrics with interaction result."""
        self.metrics['total_interactions'] += 1
        
        if result.success:
            self.metrics['successful_interactions'] += 1
        else:
            self.metrics['failed_interactions'] += 1
        
        # Update average duration
        total_duration = self.metrics['average_duration'] * (self.metrics['total_interactions'] - 1)
        self.metrics['average_duration'] = (total_duration + result.duration) / self.metrics['total_interactions']
        
        # Update total energy
        self.metrics['total_energy'] += result.energy_cost

class CollisionHandler:
    """Handles collision detection and resolution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.collision_threshold = self.config.get('collision_threshold', 0.01)  # m
        
    def check_collisions(self,
                        agents: Dict[str, Any],
                        grasped_objects: Dict[str, Any],
                        agent_id: str = None,
                        object_id: str = None) -> List[Dict[str, Any]]:
        """
        Check for collisions between agents and objects.
        
        Args:
            agents: Dictionary of agents
            grasped_objects: Dictionary of grasped objects
            agent_id: Optional specific agent to check
            object_id: Optional specific object to check
            
        Returns:
            List of collision information
        """
        collisions = []
        
        # Extract positions
        agent_positions = {}
        object_positions = {}
        
        for a_id, agent in agents.items():
            if agent_id is None or a_id == agent_id:
                agent_positions[a_id] = np.array(agent['position'])
        
        for o_id, obj in grasped_objects.items():
            if object_id is None or o_id == object_id:
                object_positions[o_id] = np.array(obj['position'])
        
        # Check agent-agent collisions
        agent_ids = list(agent_positions.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                pos1 = agent_positions[agent_ids[i]]
                pos2 = agent_positions[agent_ids[j]]
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < self.collision_threshold * 2:  # Agent radius approximation
                    collisions.append({
                        'type': 'agent-agent',
                        'participants': [agent_ids[i], agent_ids[j]],
                        'distance': distance,
                        'position': ((pos1 + pos2) / 2).tolist()
                    })
        
        # Check agent-object collisions
        for a_id, a_pos in agent_positions.items():
            for o_id, o_pos in object_positions.items():
                distance = np.linalg.norm(a_pos - o_pos)
                
                if distance < self.collision_threshold * 1.5:  # Smaller threshold
                    collisions.append({
                        'type': 'agent-object',
                        'agent_id': a_id,
                        'object_id': o_id,
                        'distance': distance,
                        'position': o_pos.tolist()
                    })
        
        return collisions
    
    def resolve_collision(self, 
                         collision: Dict[str, Any],
                         agents: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Resolve a collision by calculating repulsion forces.
        
        Args:
            collision: Collision information
            agents: Dictionary of agents
            
        Returns:
            Dictionary of forces to apply
        """
        forces = {}
        
        if collision['type'] == 'agent-agent':
            a1_id, a2_id = collision['participants']
            
            if a1_id in agents and a2_id in agents:
                pos1 = np.array(agents[a1_id]['position'])
                pos2 = np.array(agents[a2_id]['position'])
                
                # Calculate repulsion direction
                direction = pos1 - pos2
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    direction = direction / distance
                    
                    # Calculate repulsion force (inverse square)
                    min_distance = self.collision_threshold * 2
                    force_magnitude = 100.0 * (min_distance / distance)**2
                    
                    forces[a1_id] = direction * force_magnitude
                    forces[a2_id] = -direction * force_magnitude
        
        elif collision['type'] == 'agent-object':
            a_id = collision['agent_id']
            o_id = collision['object_id']
            
            if a_id in agents:
                a_pos = np.array(agents[a_id]['position'])
                o_pos = np.array(collision['position'])
                
                # Calculate repulsion direction
                direction = a_pos - o_pos
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    direction = direction / distance
                    
                    # Calculate repulsion force
                    min_distance = self.collision_threshold * 1.5
                    force_magnitude = 50.0 * (min_distance / distance)**2
                    
                    forces[a_id] = direction * force_magnitude
        
        return forces

# Import time for timestamps
import time

# Example usage
if __name__ == "__main__":
    # Create interaction handler
    handler = InteractionHandler()
    
    # Register an agent
    agent_properties = {
        'capabilities': ['grab', 'push', 'pull', 'rotate'],
        'position': [0, 0, 0],
        'orientation': [1, 0, 0, 0],
        'mass': 70.0,
        'max_force': 200.0
    }
    handler.register_agent('agent_1', agent_properties)
    
    # Perform an interaction
    result = handler.perform_interaction(
        interaction_type=InteractionType.GRAB,
        agent_id='agent_1',
        target_id='object_1',
        parameters={
            'grasp_position': [0.5, 1.0, 0.5],
            'grasp_force': 30.0
        }
    )
    
    print(f"Interaction result: {result.success} - {result.message}")
    
    # Get metrics
    metrics = handler.get_metrics()
    print(f"Metrics: {metrics}")