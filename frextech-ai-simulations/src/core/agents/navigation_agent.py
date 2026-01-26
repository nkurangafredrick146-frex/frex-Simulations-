"""
Navigation Agent for 3D Worlds
Intelligent navigation and pathfinding in complex 3D environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Set
from dataclasses import dataclass, field
import logging
from enum import Enum
import heapq
from collections import defaultdict, deque
from scipy.spatial import KDTree
from scipy import ndimage
import networkx as nx
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class NavigationState(Enum):
    """Navigation agent states"""
    IDLE = "idle"
    PLANNING = "planning"
    MOVING = "moving"
    AVOIDING = "avoiding"
    STUCK = "stuck"
    COMPLETE = "complete"

class MovementType(Enum):
    """Types of movement capabilities"""
    WALKING = "walking"
    FLYING = "flying"
    SWIMMING = "swimming"
    CLIMBING = "climbing"
    TELEPORTING = "teleporting"
    VEHICLE = "vehicle"

@dataclass
class NavigationConfig:
    """Configuration for navigation agent"""
    # Movement parameters
    movement_speed: float = 2.0  # m/s
    rotation_speed: float = 90.0  # degrees/s
    acceleration: float = 2.0  # m/s²
    deceleration: float = 3.0
    
    # Navigation parameters
    planning_timeout: float = 5.0
    replanning_interval: float = 1.0
    goal_tolerance: float = 0.5
    lookahead_distance: float = 3.0
    
    # Pathfinding
    pathfinding_algorithm: str = "astar"  # "astar", "theta", "rrt", "rrt*"
    heuristic_weight: float = 1.0
    max_search_nodes: int = 10000
    smooth_path: bool = True
    path_smoothing_iterations: int = 10
    
    # Obstacle avoidance
    obstacle_margin: float = 0.5
    dynamic_avoidance: bool = True
    prediction_horizon: float = 2.0
    social_navigation: bool = True
    
    # Memory
    use_memory: bool = True
    memory_decay: float = 0.99
    max_memory_items: int = 1000
    
    # Perception
    perception_range: float = 10.0
    perception_update_rate: float = 10.0  # Hz
    
    # Advanced
    enable_learning: bool = True
    use_neural_planner: bool = False
    exploration_bonus: float = 0.1

@dataclass
class NavigationGoal:
    """Navigation goal specification"""
    position: np.ndarray  # Target position [x, y, z]
    orientation: Optional[np.ndarray] = None  # Target orientation (quaternion)
    tolerance: float = 0.5
    priority: int = 1
    timeout: Optional[float] = None
    requirements: Optional[Dict[str, Any]] = None
    callback: Optional[callable] = None
    
    def __post_init__(self):
        """Initialize goal"""
        if self.requirements is None:
            self.requirements = {}
        if isinstance(self.position, list):
            self.position = np.array(self.position)

@dataclass 
class NavigationPath:
    """Navigation path with waypoints"""
    waypoints: List[np.ndarray]
    costs: List[float]
    total_cost: float
    feasible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_current_segment(self, current_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get current segment of path"""
        if len(self.waypoints) < 2:
            return current_pos, current_pos
        
        # Find closest waypoint
        distances = [np.linalg.norm(wp - current_pos) for wp in self.waypoints]
        closest_idx = np.argmin(distances)
        
        # Get next waypoint (skip if too close to current)
        next_idx = min(closest_idx + 1, len(self.waypoints) - 1)
        
        return self.waypoints[closest_idx], self.waypoints[next_idx]
    
    def is_complete(self, current_pos: np.ndarray, tolerance: float = 0.5) -> bool:
        """Check if path is complete"""
        if not self.waypoints:
            return True
        
        final_waypoint = self.waypoints[-1]
        distance = np.linalg.norm(final_waypoint - current_pos)
        
        return distance <= tolerance

class NavigationAgent:
    """Intelligent navigation agent for 3D environments"""
    
    def __init__(
        self,
        agent_id: str,
        config: Optional[NavigationConfig] = None,
        movement_type: MovementType = MovementType.WALKING
    ):
        self.agent_id = agent_id
        self.config = config or NavigationConfig()
        self.movement_type = movement_type
        
        # Agent state
        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion
        self.velocity = np.zeros(3)
        self.state = NavigationState.IDLE
        
        # Navigation state
        self.current_goal: Optional[NavigationGoal] = None
        self.current_path: Optional[NavigationPath] = None
        self.waypoint_index = 0
        
        # Environment knowledge
        self.navigation_mesh: Optional[NavigationMesh] = None
        self.obstacles: List[Dict[str, Any]] = []
        self.dynamic_objects: List[Dict[str, Any]] = []
        
        # Memory and learning
        self.memory = NavigationMemory(
            decay=self.config.memory_decay,
            max_items=self.config.max_memory_items
        )
        
        if self.config.enable_learning:
            self.learning_model = NavigationLearningModel()
        
        # Perception system
        self.perception = PerceptionSystem(
            range=self.config.perception_range,
            update_rate=self.config.perception_update_rate
        )
        
        # Path planner
        self.planner = self._create_path_planner()
        
        # Control system
        self.controller = NavigationController(
            movement_speed=self.config.movement_speed,
            rotation_speed=self.config.rotation_speed,
            acceleration=self.config.acceleration,
            deceleration=self.config.deceleration
        )
        
        # Timing and statistics
        self.start_time = 0.0
        self.path_time = 0.0
        self.statistics = defaultdict(float)
        
        logger.info(f"Navigation agent {agent_id} initialized")
    
    def _create_path_planner(self):
        """Create appropriate path planner"""
        if self.config.pathfinding_algorithm == "astar":
            return AStarPlanner(
                heuristic_weight=self.config.heuristic_weight,
                max_nodes=self.config.max_search_nodes
            )
        elif self.config.pathfinding_algorithm == "theta":
            return ThetaStarPlanner(
                heuristic_weight=self.config.heuristic_weight,
                max_nodes=self.config.max_search_nodes
            )
        elif self.config.pathfinding_algorithm == "rrt":
            return RRTPlanner(
                max_nodes=self.config.max_search_nodes,
                goal_bias=0.1
            )
        elif self.config.pathfinding_algorithm == "rrt*":
            return RRTStarPlanner(
                max_nodes=self.config.max_search_nodes,
                goal_bias=0.1,
                rewire_radius=2.0
            )
        else:
            raise ValueError(f"Unknown pathfinding algorithm: {self.config.pathfinding_algorithm}")
    
    def set_navigation_mesh(self, navmesh: "NavigationMesh"):
        """Set navigation mesh for pathfinding"""
        self.navigation_mesh = navmesh
        self.planner.set_navigation_mesh(navmesh)
        logger.debug(f"Navigation mesh set for agent {self.agent_id}")
    
    def set_position(self, position: np.ndarray, orientation: Optional[np.ndarray] = None):
        """Set agent position and orientation"""
        self.position = np.array(position)
        
        if orientation is not None:
            self.orientation = np.array(orientation)
            if len(self.orientation) == 3:  # Convert from Euler angles
                self.orientation = self._euler_to_quaternion(self.orientation)
    
    def set_goal(self, goal: NavigationGoal):
        """Set navigation goal"""
        self.current_goal = goal
        self.state = NavigationState.PLANNING
        self.start_time = time.time()
        self.waypoint_index = 0
        
        logger.info(f"Agent {self.agent_id} received new goal at {goal.position}")
        
        # Start planning
        self.plan_path()
    
    def plan_path(self) -> bool:
        """Plan path to current goal"""
        if self.current_goal is None:
            logger.warning("No goal set for path planning")
            return False
        
        if self.navigation_mesh is None:
            logger.warning("No navigation mesh set")
            return False
        
        # Check if goal is reachable
        if not self._is_goal_reachable(self.current_goal.position):
            logger.warning(f"Goal {self.current_goal.position} is not reachable")
            self.state = NavigationState.STUCK
            return False
        
        # Plan path
        start_time = time.time()
        
        path = self.planner.plan(
            start=self.position,
            goal=self.current_goal.position,
            obstacles=self.obstacles
        )
        
        planning_time = time.time() - start_time
        
        if path is None or not path.waypoints:
            logger.warning(f"Failed to find path to {self.current_goal.position}")
            self.state = NavigationState.STUCK
            return False
        
        # Smooth path if requested
        if self.config.smooth_path:
            path = self._smooth_path(path)
        
        self.current_path = path
        self.state = NavigationState.MOVING
        self.waypoint_index = 0
        
        # Update statistics
        self.statistics["planning_time"] += planning_time
        self.statistics["planning_count"] += 1
        
        logger.info(f"Path planned with {len(path.waypoints)} waypoints, cost: {path.total_cost:.2f}")
        
        return True
    
    def _is_goal_reachable(self, goal_position: np.ndarray) -> bool:
        """Check if goal position is reachable"""
        # Check navigation mesh
        if self.navigation_mesh:
            start_face = self.navigation_mesh.find_containing_face(self.position)
            goal_face = self.navigation_mesh.find_containing_face(goal_position)
            
            if start_face is None or goal_face is None:
                return False
            
            # Check if faces are connected
            return self.navigation_mesh.are_faces_connected(start_face, goal_face)
        
        return True
    
    def _smooth_path(self, path: NavigationPath) -> NavigationPath:
        """Smooth navigation path"""
        if len(path.waypoints) < 3:
            return path
        
        smoothed_waypoints = [path.waypoints[0]]
        
        for i in range(1, len(path.waypoints) - 1):
            prev = smoothed_waypoints[-1]
            current = path.waypoints[i]
            next_wp = path.waypoints[i + 1]
            
            # Check if we can skip current waypoint
            if self._line_of_sight(prev, next_wp):
                continue  # Skip current waypoint
            else:
                smoothed_waypoints.append(current)
        
        smoothed_waypoints.append(path.waypoints[-1])
        
        # Recompute costs
        costs = [0.0]
        for i in range(1, len(smoothed_waypoints)):
            cost = np.linalg.norm(smoothed_waypoints[i] - smoothed_waypoints[i-1])
            costs.append(cost)
        
        total_cost = sum(costs)
        
        return NavigationPath(
            waypoints=smoothed_waypoints,
            costs=costs,
            total_cost=total_cost,
            feasible=path.feasible,
            metadata=path.metadata
        )
    
    def _line_of_sight(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        """Check line of sight between two points"""
        if not self.navigation_mesh:
            return True
        
        # Sample points along line
        direction = point2 - point1
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return True
        
        direction = direction / distance
        step_size = 0.5  # meters
        num_steps = int(distance / step_size)
        
        for i in range(num_steps + 1):
            t = i * step_size / distance
            if t > 1.0:
                t = 1.0
            
            point = point1 + direction * t * distance
            
            # Check if point is in navigation mesh
            face = self.navigation_mesh.find_containing_face(point)
            if face is None:
                return False
        
        return True
    
    def update(self, dt: float) -> bool:
        """Update navigation agent"""
        if self.state == NavigationState.IDLE:
            return False
        
        # Check timeout
        if self.current_goal and self.current_goal.timeout:
            elapsed = time.time() - self.start_time
            if elapsed > self.current_goal.timeout:
                logger.warning(f"Navigation timeout for agent {self.agent_id}")
                self.state = NavigationState.STUCK
                return False
        
        # Update perception
        self._update_perception(dt)
        
        # Check if path needs replanning
        if self._needs_replanning():
            logger.debug(f"Replanning path for agent {self.agent_id}")
            self.plan_path()
        
        # Execute navigation
        if self.state == NavigationState.MOVING and self.current_path:
            self._follow_path(dt)
        
        # Check goal completion
        if self._check_goal_completion():
            self._complete_goal()
            return True
        
        return False
    
    def _update_perception(self, dt: float):
        """Update perception of environment"""
        # Update perception system
        self.perception.update(self.position, self.orientation, dt)
        
        # Get perceived obstacles
        perceived_obstacles = self.perception.get_obstacles()
        
        # Update dynamic objects
        self.dynamic_objects = []
        for obj in perceived_obstacles:
            if obj.get("dynamic", False):
                self.dynamic_objects.append(obj)
        
        # Update memory with perceived information
        if self.config.use_memory:
            for obj in perceived_obstacles:
                self.memory.remember(
                    location=obj["position"],
                    event="obstacle",
                    data=obj
                )
    
    def _needs_replanning(self) -> bool:
        """Check if path needs replanning"""
        if self.state != NavigationState.MOVING:
            return False
        
        # Check replanning interval
        current_time = time.time()
        if current_time - self.path_time < self.config.replanning_interval:
            return False
        
        # Check for dynamic obstacles
        if self.config.dynamic_avoidance and self.dynamic_objects:
            for obj in self.dynamic_objects:
                # Predict future position
                if "velocity" in obj:
                    future_pos = obj["position"] + obj["velocity"] * self.config.prediction_horizon
                    
                    # Check if obstacle will intersect path
                    if self._path_obstructed(future_pos, obj.get("radius", 0.5)):
                        return True
        
        # Check for static obstacles not in original planning
        if self.current_path:
            # Sample points along path
            for i in range(self.waypoint_index, len(self.current_path.waypoints) - 1):
                segment_start = self.current_path.waypoints[i]
                segment_end = self.current_path.waypoints[i + 1]
                
                # Check for obstacles along segment
                if self._segment_obstructed(segment_start, segment_end):
                    return True
        
        return False
    
    def _path_obstructed(self, obstacle_position: np.ndarray, obstacle_radius: float) -> bool:
        """Check if obstacle obstructs path"""
        if not self.current_path:
            return False
        
        # Find closest point on path to obstacle
        min_distance = float('inf')
        closest_point = None
        
        for i in range(self.waypoint_index, len(self.current_path.waypoints) - 1):
            start = self.current_path.waypoints[i]
            end = self.current_path.waypoints[i + 1]
            
            # Find closest point on segment
            t = np.clip(np.dot(obstacle_position - start, end - start) / 
                       np.dot(end - start, end - start), 0, 1)
            point = start + t * (end - start)
            distance = np.linalg.norm(point - obstacle_position)
            
            if distance < min_distance:
                min_distance = distance
                closest_point = point
        
        return min_distance < (obstacle_radius + self.config.obstacle_margin)
    
    def _segment_obstructed(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if segment is obstructed by obstacles"""
        # Check against known obstacles
        for obstacle in self.obstacles:
            obstacle_pos = obstacle["position"]
            obstacle_radius = obstacle.get("radius", 0.5)
            
            # Find closest point on segment to obstacle
            t = np.clip(np.dot(obstacle_pos - start, end - start) / 
                       np.dot(end - start, end - start), 0, 1)
            closest_point = start + t * (end - start)
            distance = np.linalg.norm(closest_point - obstacle_pos)
            
            if distance < (obstacle_radius + self.config.obstacle_margin):
                return True
        
        return False
    
    def _follow_path(self, dt: float):
        """Follow current path"""
        if not self.current_path or self.waypoint_index >= len(self.current_path.waypoints):
            return
        
        # Get current and next waypoint
        if self.waypoint_index == 0:
            current_wp = self.position
        else:
            current_wp = self.current_path.waypoints[self.waypoint_index - 1]
        
        next_wp = self.current_path.waypoints[self.waypoint_index]
        
        # Compute desired velocity
        to_next = next_wp - self.position
        distance = np.linalg.norm(to_next)
        
        if distance < self.config.goal_tolerance:
            # Reached waypoint, move to next
            self.waypoint_index += 1
            
            if self.waypoint_index >= len(self.current_path.waypoints):
                # Reached final waypoint
                return
            
            # Update next waypoint
            next_wp = self.current_path.waypoints[self.waypoint_index]
            to_next = next_wp - self.position
            distance = np.linalg.norm(to_next)
        
        # Compute desired direction
        if distance > 1e-6:
            desired_direction = to_next / distance
        else:
            desired_direction = np.zeros(3)
        
        # Avoid obstacles
        if self.config.dynamic_avoidance and self.dynamic_objects:
            avoidance_force = self._compute_avoidance_force()
            if np.linalg.norm(avoidance_force) > 0:
                # Blend desired direction with avoidance
                avoidance_weight = min(1.0, np.linalg.norm(avoidance_force) / self.config.movement_speed)
                desired_direction = (1 - avoidance_weight) * desired_direction + \
                                   avoidance_weight * avoidance_force / np.linalg.norm(avoidance_force)
        
        # Get control command
        control = self.controller.compute_control(
            current_position=self.position,
            current_velocity=self.velocity,
            target_position=next_wp,
            target_direction=desired_direction,
            dt=dt
        )
        
        # Apply control
        self.velocity = control.velocity
        self.position += self.velocity * dt
        
        # Update orientation
        if np.linalg.norm(self.velocity) > 0.1:
            forward = self.velocity / np.linalg.norm(self.velocity)
            self.orientation = self._look_at(forward, np.array([0.0, 1.0, 0.0]))
    
    def _compute_avoidance_force(self) -> np.ndarray:
        """Compute obstacle avoidance force"""
        avoidance_force = np.zeros(3)
        
        for obj in self.dynamic_objects:
            obj_pos = obj["position"]
            obj_vel = obj.get("velocity", np.zeros(3))
            obj_radius = obj.get("radius", 0.5)
            
            # Relative position and velocity
            relative_pos = obj_pos - self.position
            relative_vel = obj_vel - self.velocity
            
            distance = np.linalg.norm(relative_pos)
            
            if distance < (obj_radius + self.config.obstacle_margin) * 2:
                # Compute repulsive force
                # Use velocity obstacle method
                time_to_collision = self._compute_time_to_collision(
                    self.position, self.velocity,
                    obj_pos, obj_vel,
                    obj_radius + self.config.obstacle_margin
                )
                
                if time_to_collision > 0 and time_to_collision < self.config.prediction_horizon:
                    # Compute avoidance direction
                    avoidance_dir = -relative_pos / distance
                    avoidance_strength = 1.0 / (time_to_collision + 1e-6)
                    
                    avoidance_force += avoidance_dir * avoidance_strength
        
        return avoidance_force
    
    def _compute_time_to_collision(
        self,
        pos1: np.ndarray, vel1: np.ndarray,
        pos2: np.ndarray, vel2: np.ndarray,
        radius_sum: float
    ) -> float:
        """Compute time to collision between two moving spheres"""
        # Relative position and velocity
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        
        # Solve quadratic equation for collision time
        a = np.dot(rel_vel, rel_vel)
        b = 2 * np.dot(rel_pos, rel_vel)
        c = np.dot(rel_pos, rel_pos) - radius_sum**2
        
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return float('inf')
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # Return smallest positive time
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
        elif t1 > 0:
            return t1
        elif t2 > 0:
            return t2
        else:
            return float('inf')
    
    def _look_at(self, forward: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
        """Create quaternion looking in direction"""
        forward = forward / np.linalg.norm(forward)
        
        # Compute right vector
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        # Recompute up vector
        up = np.cross(forward, right)
        
        # Create rotation matrix
        R = np.column_stack([right, up, forward])
        
        # Convert to quaternion
        trace = R[0,0] + R[1,1] + R[2,2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        
        return np.array([qw, qx, qy, qz])
    
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to quaternion"""
        roll, pitch, yaw = euler
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qw, qx, qy, qz])
    
    def _check_goal_completion(self) -> bool:
        """Check if current goal is completed"""
        if self.current_goal is None:
            return False
        
        # Check position
        distance = np.linalg.norm(self.position - self.current_goal.position)
        
        if distance > self.current_goal.tolerance:
            return False
        
        # Check orientation if specified
        if self.current_goal.orientation is not None:
            # Compute angular difference
            dot = np.abs(np.dot(self.orientation, self.current_goal.orientation))
            angle = 2 * np.arccos(np.clip(dot, -1.0, 1.0))
            
            if angle > 0.1:  # ~5.7 degrees
                return False
        
        # Check requirements if specified
        if self.current_goal.requirements:
            for req_name, req_value in self.current_goal.requirements.items():
                # Check various requirements
                if req_name == "height" and self.position[1] < req_value:
                    return False
                # Add more requirement checks as needed
        
        return True
    
    def _complete_goal(self):
        """Complete current goal"""
        logger.info(f"Agent {self.agent_id} reached goal at {self.current_goal.position}")
        
        # Call callback if specified
        if self.current_goal.callback:
            self.current_goal.callback(self.agent_id, self.current_goal)
        
        # Update statistics
        elapsed = time.time() - self.start_time
        self.statistics["total_navigation_time"] += elapsed
        self.statistics["goals_completed"] += 1
        
        # Reset state
        self.current_goal = None
        self.current_path = None
        self.state = NavigationState.COMPLETE
    
    def add_obstacle(self, obstacle: Dict[str, Any]):
        """Add obstacle to navigation system"""
        self.obstacles.append(obstacle)
        
        # Update navigation mesh if available
        if self.navigation_mesh:
            self.navigation_mesh.add_obstacle(obstacle)
    
    def remove_obstacle(self, obstacle_id: str):
        """Remove obstacle from navigation system"""
        self.obstacles = [o for o in self.obstacles if o.get("id") != obstacle_id]
        
        if self.navigation_mesh:
            self.navigation_mesh.remove_obstacle(obstacle_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get navigation statistics"""
        stats = dict(self.statistics)
        
        if self.statistics.get("goals_completed", 0) > 0:
            stats["average_time_per_goal"] = \
                stats["total_navigation_time"] / stats["goals_completed"]
        
        if self.statistics.get("planning_count", 0) > 0:
            stats["average_planning_time"] = \
                stats["planning_time"] / stats["planning_count"]
        
        stats.update({
            "agent_id": self.agent_id,
            "state": self.state.value,
            "position": self.position.tolist(),
            "has_goal": self.current_goal is not None,
            "has_path": self.current_path is not None
        })
        
        return stats
    
    def save_state(self, filepath: Union[str, Path]):
        """Save navigation agent state"""
        state = {
            "agent_id": self.agent_id,
            "position": self.position.tolist(),
            "orientation": self.orientation.tolist(),
            "velocity": self.velocity.tolist(),
            "state": self.state.value,
            "statistics": self.statistics,
            "memory": self.memory.get_state() if self.config.use_memory else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved navigation state to {filepath}")
    
    def load_state(self, filepath: Union[str, Path]):
        """Load navigation agent state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.position = np.array(state["position"])
        self.orientation = np.array(state["orientation"])
        self.velocity = np.array(state["velocity"])
        self.state = NavigationState(state["state"])
        self.statistics = state["statistics"]
        
        if self.config.use_memory and state["memory"]:
            self.memory.load_state(state["memory"])
        
        logger.info(f"Loaded navigation state from {filepath}")

class NavigationMesh:
    """Navigation mesh for efficient pathfinding"""
    
    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces
        
        # Build adjacency graph
        self.graph = self._build_adjacency_graph()
        
        # Build face connections
        self.face_graph = self._build_face_graph()
        
        # Build spatial acceleration structure
        self.kdtree = KDTree(vertices)
        
        logger.info(f"Navigation mesh built with {len(vertices)} vertices, {len(faces)} faces")
    
    def _build_adjacency_graph(self) -> nx.Graph:
        """Build graph of walkable areas"""
        graph = nx.Graph()
        
        # Add vertices
        for i, vertex in enumerate(self.vertices):
            graph.add_node(i, pos=vertex)
        
        # Add edges from faces
        for face in self.faces:
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                
                # Add edge with weight as distance
                distance = np.linalg.norm(self.vertices[v1] - self.vertices[v2])
                graph.add_edge(v1, v2, weight=distance)
        
        return graph
    
    def _build_face_graph(self) -> nx.Graph:
        """Build graph of connected faces"""
        face_graph = nx.Graph()
        
        # Map edges to faces
        edge_to_faces = defaultdict(list)
        
        for face_idx, face in enumerate(self.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edge_to_faces[edge].append(face_idx)
        
        # Connect faces sharing edges
        for edge, face_indices in edge_to_faces.items():
            if len(face_indices) == 2:
                # Two faces share this edge (walkable connection)
                f1, f2 = face_indices
                
                # Compute edge midpoint
                v1, v2 = self.vertices[edge[0]], self.vertices[edge[1]]
                midpoint = (v1 + v2) / 2
                
                # Add edge between faces
                face_graph.add_edge(f1, f2, midpoint=midpoint, edge=edge)
        
        return face_graph
    
    def find_containing_face(self, point: np.ndarray) -> Optional[int]:
        """Find face containing point"""
        # Simple test: find closest face
        # In production, would use proper point-in-polygon test
        
        for face_idx, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face]
            
            # Barycentric coordinates test
            if self._point_in_triangle(point, v0, v1, v2):
                return face_idx
        
        return None
    
    def _point_in_triangle(self, p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
        """Test if point is in triangle using barycentric coordinates"""
        # Compute vectors
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        
        # Compute dot products
        dot00 = np.dot(v0v1, v0v1)
        dot01 = np.dot(v0v1, v0v2)
        dot02 = np.dot(v0v1, p - v0)
        dot11 = np.dot(v0v2, v0v2)
        dot12 = np.dot(v0v2, p - v0)
        
        # Compute barycentric coordinates
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v < 1)
    
    def are_faces_connected(self, face1: int, face2: int) -> bool:
        """Check if two faces are connected in the navigation mesh"""
        return nx.has_path(self.face_graph, face1, face2)
    
    def add_obstacle(self, obstacle: Dict[str, Any]):
        """Add obstacle to navigation mesh"""
        # Mark faces intersecting obstacle as unwalkable
        obstacle_pos = obstacle["position"]
        obstacle_radius = obstacle.get("radius", 0.5)
        
        for face_idx, face in enumerate(self.faces):
            face_center = np.mean(self.vertices[face], axis=0)
            distance = np.linalg.norm(face_center - obstacle_pos)
            
            if distance < obstacle_radius:
                # Remove face from graph
                self.face_graph.remove_node(face_idx)
    
    def remove_obstacle(self, obstacle_id: str):
        """Remove obstacle from navigation mesh"""
        # Rebuild face graph (simplified)
        # In production, would track which faces were removed
        self.face_graph = self._build_face_graph()

class NavigationMemory:
    """Memory system for navigation learning"""
    
    def __init__(self, decay: float = 0.99, max_items: int = 1000):
        self.decay = decay
        self.max_items = max_items
        self.memories: List[Dict[str, Any]] = []
        self.spatial_index = KDTree([])
    
    def remember(self, location: np.ndarray, event: str, data: Dict[str, Any]):
        """Remember an event at a location"""
        memory = {
            "location": location,
            "event": event,
            "data": data,
            "timestamp": time.time(),
            "strength": 1.0
        }
        
        self.memories.append(memory)
        
        # Apply decay to old memories
        self._apply_decay()
        
        # Trim if too many memories
        if len(self.memories) > self.max_items:
            self.memories = self.memories[-self.max_items:]
        
        # Update spatial index
        if len(self.memories) > 0:
            locations = np.array([m["location"] for m in self.memories])
            self.spatial_index = KDTree(locations)
    
    def recall(self, location: np.ndarray, radius: float = 5.0) -> List[Dict[str, Any]]:
        """Recall memories near a location"""
        if len(self.memories) == 0:
            return []
        
        # Find memories within radius
        indices = self.spatial_index.query_ball_point(location, radius)
        
        recalled = []
        for idx in indices:
            memory = self.memories[idx]
            
            # Apply recency and relevance weighting
            age = time.time() - memory["timestamp"]
            recency_weight = np.exp(-age / 3600)  # Decay over hours
            
            distance = np.linalg.norm(location - memory["location"])
            relevance_weight = 1.0 / (1.0 + distance)
            
            memory_weight = memory["strength"] * recency_weight * relevance_weight
            
            recalled.append({
                **memory,
                "weight": memory_weight
            })
        
        # Sort by weight
        recalled.sort(key=lambda x: x["weight"], reverse=True)
        
        return recalled
    
    def _apply_decay(self):
        """Apply decay to memory strengths"""
        current_time = time.time()
        
        for memory in self.memories:
            age = current_time - memory["timestamp"]
            memory["strength"] *= self.decay ** (age / 3600)  # Decay per hour
    
    def get_state(self) -> Dict[str, Any]:
        """Get memory state for saving"""
        return {
            "memories": self.memories,
            "decay": self.decay,
            "max_items": self.max_items
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load memory state"""
        self.memories = state["memories"]
        self.decay = state.get("decay", self.decay)
        self.max_items = state.get("max_items", self.max_items)
        
        # Rebuild spatial index
        if len(self.memories) > 0:
            locations = np.array([m["location"] for m in self.memories])
            self.spatial_index = KDTree(locations)

class NavigationLearningModel:
    """Learning model for navigation improvement"""
    
    def __init__(self):
        # Neural network for path prediction
        self.path_predictor = self._build_path_predictor()
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Learning parameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        
        logger.info("Navigation learning model initialized")
    
    def _build_path_predictor(self) -> nn.Module:
        """Build neural network for path prediction"""
        class PathPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(6, 64),  # start + goal
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU()
                )
                
                self.decoder = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 3)  # next waypoint
                )
            
            def forward(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
                x = torch.cat([start, goal], dim=-1)
                encoded = self.encoder(x)
                return self.decoder(encoded)
        
        return PathPredictor()
    
    def predict_next_waypoint(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Predict next waypoint using learning model"""
        with torch.no_grad():
            start_tensor = torch.FloatTensor(start).unsqueeze(0)
            goal_tensor = torch.FloatTensor(goal).unsqueeze(0)
            
            prediction = self.path_predictor(start_tensor, goal_tensor)
            return prediction.squeeze(0).numpy()
    
    def learn_from_experience(self, experience: Dict[str, Any]):
        """Learn from navigation experience"""
        self.experience_buffer.append(experience)
        
        # TODO: Implement reinforcement learning
        # This would include Q-learning, policy gradients, etc.
        
        pass

class PerceptionSystem:
    """Perception system for navigation agent"""
    
    def __init__(self, range: float = 10.0, update_rate: float = 10.0):
        self.range = range
        self.update_rate = update_rate
        self.last_update = 0.0
        
        # Detected objects
        self.objects: List[Dict[str, Any]] = []
        
        # Field of view
        self.fov_horizontal = 120.0  # degrees
        self.fov_vertical = 90.0  # degrees
    
    def update(self, position: np.ndarray, orientation: np.ndarray, dt: float):
        """Update perception"""
        current_time = time.time()
        
        if current_time - self.last_update < 1.0 / self.update_rate:
            return
        
        self.last_update = current_time
        
        # Clear previous detections
        self.objects.clear()
        
        # In production, this would interface with actual sensors
        # For now, return empty list
        # self.objects = self._simulate_perception(position, orientation)
    
    def get_obstacles(self) -> List[Dict[str, Any]]:
        """Get perceived obstacles"""
        return [obj for obj in self.objects if obj.get("type") == "obstacle"]
    
    def _simulate_perception(self, position: np.ndarray, orientation: np.ndarray) -> List[Dict[str, Any]]:
        """Simulate perception (for testing)"""
        # This would be replaced with actual sensor data
        return []

class NavigationController:
    """Controller for smooth navigation"""
    
    def __init__(
        self,
        movement_speed: float = 2.0,
        rotation_speed: float = 90.0,
        acceleration: float = 2.0,
        deceleration: float = 3.0
    ):
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed
        self.acceleration = acceleration
        self.deceleration = deceleration
        
        # PID controllers
        self.position_pid = PIDController(kp=2.0, ki=0.0, kd=0.5)
        self.rotation_pid = PIDController(kp=1.0, ki=0.0, kd=0.1)
    
    @dataclass
    class ControlOutput:
        velocity: np.ndarray
        angular_velocity: np.ndarray
    
    def compute_control(
        self,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        target_position: np.ndarray,
        target_direction: np.ndarray,
        dt: float
    ) -> "ControlOutput":
        """Compute control output"""
        # Position control
        position_error = target_position - current_position
        distance = np.linalg.norm(position_error)
        
        # Desired speed based on distance
        if distance < 1.0:
            desired_speed = self.movement_speed * distance  # Slow down near target
        else:
            desired_speed = self.movement_speed
        
        # Compute desired velocity
        if distance > 1e-6:
            desired_direction = position_error / distance
        else:
            desired_direction = target_direction
        
        desired_velocity = desired_direction * desired_speed
        
        # Apply acceleration limits
        velocity_change = desired_velocity - current_velocity
        max_change = self.acceleration * dt
        
        if np.linalg.norm(velocity_change) > max_change:
            velocity_change = velocity_change / np.linalg.norm(velocity_change) * max_change
        
        new_velocity = current_velocity + velocity_change
        
        # Limit maximum speed
        current_speed = np.linalg.norm(new_velocity)
        if current_speed > self.movement_speed:
            new_velocity = new_velocity / current_speed * self.movement_speed
        
        # Rotation control
        current_forward = current_velocity / (current_speed + 1e-6)
        desired_forward = desired_direction
        
        # Compute rotation error
        cross = np.cross(current_forward, desired_forward)
        rotation_error = np.arctan2(np.linalg.norm(cross), np.dot(current_forward, desired_forward))
        
        if rotation_error > 1e-3:
            rotation_axis = cross / np.linalg.norm(cross)
            angular_velocity = rotation_axis * rotation_error * self.rotation_speed * np.pi / 180.0
        else:
            angular_velocity = np.zeros(3)
        
        return self.ControlOutput(
            velocity=new_velocity,
            angular_velocity=angular_velocity
        )

class PIDController:
    """PID controller for smooth control"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.previous_error = 0.0
    
    def compute(self, error: float, dt: float) -> float:
        """Compute PID output"""
        self.integral += error * dt
        
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        self.previous_error = error
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Path planning algorithms

class AStarPlanner:
    """A* path planning algorithm"""
    
    def __init__(self, heuristic_weight: float = 1.0, max_nodes: int = 10000):
        self.heuristic_weight = heuristic_weight
        self.max_nodes = max_nodes
    
    def set_navigation_mesh(self, navmesh: NavigationMesh):
        self.navmesh = navmesh
    
    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Dict[str, Any]]) -> Optional[NavigationPath]:
        """Plan path using A*"""
        if self.navmesh is None:
            return None
        
        # Find containing faces
        start_face = self.navmesh.find_containing_face(start)
        goal_face = self.navmesh.find_containing_face(goal)
        
        if start_face is None or goal_face is None:
            return None
        
        # Use face graph for planning
        try:
            face_path = nx.astar_path(
                self.navmesh.face_graph,
                start_face,
                goal_face,
                heuristic=lambda u, v: self._heuristic(u, v, goal),
                weight='weight'
            )
        except nx.NetworkXNoPath:
            return None
        
        # Convert face path to waypoints
        waypoints = [start]
        
        for face_idx in face_path:
            face = self.navmesh.faces[face_idx]
            face_center = np.mean(self.navmesh.vertices[face], axis=0)
            waypoints.append(face_center)
        
        waypoints.append(goal)
        
        # Compute costs
        costs = [0.0]
        for i in range(1, len(waypoints)):
            cost = np.linalg.norm(waypoints[i] - waypoints[i-1])
            costs.append(cost)
        
        total_cost = sum(costs)
        
        return NavigationPath(
            waypoints=waypoints,
            costs=costs,
            total_cost=total_cost,
            metadata={"algorithm": "astar", "face_path": face_path}
        )
    
    def _heuristic(self, face1: int, face2: int, goal: np.ndarray) -> float:
        """Heuristic function for A*"""
        face1_center = np.mean(self.navmesh.vertices[self.navmesh.faces[face1]], axis=0)
        face2_center = np.mean(self.navmesh.vertices[self.navmesh.faces[face2]], axis=0)
        
        distance_to_goal1 = np.linalg.norm(face1_center - goal)
        distance_to_goal2 = np.linalg.norm(face2_center - goal)
        
        return self.heuristic_weight * abs(distance_to_goal1 - distance_to_goal2)

class ThetaStarPlanner(AStarPlanner):
    """Theta* path planning algorithm (any-angle A*)"""
    
    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Dict[str, Any]]) -> Optional[NavigationPath]:
        """Plan path using Theta*"""
        # Theta* extends A* with line-of-sight optimization
        path = super().plan(start, goal, obstacles)
        
        if path is None:
            return None
        
        # Post-process with line-of-sight checks
        optimized_waypoints = self._optimize_with_los(path.waypoints)
        
        # Recompute costs
        costs = [0.0]
        for i in range(1, len(optimized_waypoints)):
            cost = np.linalg.norm(optimized_waypoints[i] - optimized_waypoints[i-1])
            costs.append(cost)
        
        total_cost = sum(costs)
        
        return NavigationPath(
            waypoints=optimized_waypoints,
            costs=costs,
            total_cost=total_cost,
            metadata={"algorithm": "theta*", "optimized": True}
        )
    
    def _optimize_with_los(self, waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize path with line-of-sight checks"""
        if len(waypoints) < 3:
            return waypoints
        
        optimized = [waypoints[0]]
        i = 0
        
        while i < len(waypoints) - 1:
            j = len(waypoints) - 1
            
            while j > i + 1:
                if self._line_of_sight(optimized[-1], waypoints[j]):
                    optimized.append(waypoints[j])
                    i = j
                    break
                j -= 1
            
            if j == i + 1:
                optimized.append(waypoints[i + 1])
                i += 1
        
        return optimized
    
    def _line_of_sight(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check line of sight between points"""
        # Simplified check - in production would use navigation mesh
        return True

class RRTPlanner:
    """Rapidly-exploring Random Tree planner"""
    
    def __init__(self, max_nodes: int = 10000, goal_bias: float = 0.1):
        self.max_nodes = max_nodes
        self.goal_bias = goal_bias
    
    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Dict[str, Any]]) -> Optional[NavigationPath]:
        """Plan path using RRT"""
        # RRT implementation
        nodes = [start]
        parents = [-1]
        
        for i in range(self.max_nodes):
            # Sample random point with goal bias
            if np.random.random() < self.goal_bias:
                target = goal
            else:
                target = self._sample_random_point()
            
            # Find nearest node
            nearest_idx = self._find_nearest_node(nodes, target)
            nearest = nodes[nearest_idx]
            
            # Extend toward target
            new_node = self._extend(nearest, target)
            
            # Check collision
            if not self._collision_free(nearest, new_node, obstacles):
                continue
            
            # Add to tree
            nodes.append(new_node)
            parents.append(nearest_idx)
            
            # Check if goal reached
            if np.linalg.norm(new_node - goal) < 1.0:
                # Reconstruct path
                path = self._reconstruct_path(nodes, parents, goal)
                return self._create_path_from_points(path)
        
        return None
    
    def _sample_random_point(self) -> np.ndarray:
        """Sample random point in configuration space"""
        return np.random.uniform(-10, 10, size=3)
    
    def _find_nearest_node(self, nodes: List[np.ndarray], target: np.ndarray) -> int:
        """Find nearest node to target"""
        distances = [np.linalg.norm(node - target) for node in nodes]
        return np.argmin(distances)
    
    def _extend(self, from_node: np.ndarray, to_node: np.ndarray, step_size: float = 0.5) -> np.ndarray:
        """Extend from node toward target"""
        direction = to_node - from_node
        distance = np.linalg.norm(direction)
        
        if distance < step_size:
            return to_node
        
        direction = direction / distance
        return from_node + direction * step_size
    
    def _collision_free(self, p1: np.ndarray, p2: np.ndarray, obstacles: List[Dict[str, Any]]) -> bool:
        """Check if segment is collision-free"""
        # Simplified collision checking
        for obstacle in obstacles:
            obstacle_pos = obstacle["position"]
            obstacle_radius = obstacle.get("radius", 0.5)
            
            # Find closest point on segment to obstacle
            t = np.clip(np.dot(obstacle_pos - p1, p2 - p1) / 
                       np.dot(p2 - p1, p2 - p1), 0, 1)
            closest = p1 + t * (p2 - p1)
            distance = np.linalg.norm(closest - obstacle_pos)
            
            if distance < obstacle_radius:
                return False
        
        return True
    
    def _reconstruct_path(self, nodes: List[np.ndarray], parents: List[int], goal: np.ndarray) -> List[np.ndarray]:
        """Reconstruct path from tree"""
        # Find node closest to goal
        distances = [np.linalg.norm(node - goal) for node in nodes]
        current_idx = np.argmin(distances)
        
        # Reconstruct path
        path = [goal]
        while current_idx != -1:
            path.append(nodes[current_idx])
            current_idx = parents[current_idx]
        
        return list(reversed(path))
    
    def _create_path_from_points(self, points: List[np.ndarray]) -> NavigationPath:
        """Create NavigationPath from points"""
        costs = [0.0]
        for i in range(1, len(points)):
            cost = np.linalg.norm(points[i] - points[i-1])
            costs.append(cost)
        
        total_cost = sum(costs)
        
        return NavigationPath(
            waypoints=points,
            costs=costs,
            total_cost=total_cost,
            metadata={"algorithm": "rrt"}
        )

class RRTStarPlanner(RRTPlanner):
    """RRT* planner with optimal path refinement"""
    
    def __init__(self, max_nodes: int = 10000, goal_bias: float = 0.1, rewire_radius: float = 2.0):
        super().__init__(max_nodes, goal_bias)
        self.rewire_radius = rewire_radius
    
    def plan(self, start: np.ndarray, goal: np.ndarray, obstacles: List[Dict[str, Any]]) -> Optional[NavigationPath]:
        """Plan path using RRT*"""
        # RRT* implementation with rewiring
        nodes = [start]
        parents = [-1]
        costs = [0.0]
        
        for i in range(self.max_nodes):
            # Sample random point
            if np.random.random() < self.goal_bias:
                target = goal
            else:
                target = self._sample_random_point()
            
            # Find nearest node
            nearest_idx = self._find_nearest_node(nodes, target)
            nearest = nodes[nearest_idx]
            nearest_cost = costs[nearest_idx]
            
            # Extend toward target
            new_node = self._extend(nearest, target)
            new_cost = nearest_cost + np.linalg.norm(new_node - nearest)
            
            # Find nearby nodes for potential rewiring
            nearby_indices = self._find_nearby_nodes(nodes, new_node, self.rewire_radius)
            
            # Choose best parent
            best_parent = nearest_idx
            best_cost = new_cost
            
            for nearby_idx in nearby_indices:
                nearby = nodes[nearby_idx]
                nearby_cost = costs[nearby_idx]
                
                # Check if path through nearby is better
                tentative_cost = nearby_cost + np.linalg.norm(new_node - nearby)
                
                if tentative_cost < best_cost and self._collision_free(nearby, new_node, obstacles):
                    best_parent = nearby_idx
                    best_cost = tentative_cost
            
            # Add node to tree
            nodes.append(new_node)
            parents.append(best_parent)
            costs.append(best_cost)
            
            # Rewire nearby nodes
            self._rewire_nearby_nodes(nodes, parents, costs, nearby_indices, len(nodes) - 1, obstacles)
        
        # Find best path to goal region
        goal_nodes = []
        for i, node in enumerate(nodes):
            if np.linalg.norm(node - goal) < 1.0:
                goal_nodes.append((i, costs[i]))
        
        if not goal_nodes:
            return None
        
        # Get node with lowest cost to goal
        goal_nodes.sort(key=lambda x: x[1])
        best_goal_idx = goal_nodes[0][0]
        
        # Reconstruct path
        path = self._reconstruct_path_with_costs(nodes, parents, costs, best_goal_idx, goal)
        return self._create_path_from_points(path)
    
    def _find_nearby_nodes(self, nodes: List[np.ndarray], target: np.ndarray, radius: float) -> List[int]:
        """Find nodes within radius of target"""
        nearby = []
        
        for i, node in enumerate(nodes):
            if np.linalg.norm(node - target) <= radius:
                nearby.append(i)
        
        return nearby
    
    def _rewire_nearby_nodes(
        self,
        nodes: List[np.ndarray],
        parents: List[int],
        costs: List[float],
        nearby_indices: List[int],
        new_idx: int,
        obstacles: List[Dict[str, Any]]
    ):
        """Rewire nearby nodes through new node if beneficial"""
        new_node = nodes[new_idx]
        new_cost = costs[new_idx]
        
        for nearby_idx in nearby_indices:
            if nearby_idx == new_idx:
                continue
            
            nearby = nodes[nearby_idx]
            nearby_cost = costs[nearby_idx]
            
            # Check if path through new node is better
            tentative_cost = new_cost + np.linalg.norm(nearby - new_node)
            
            if tentative_cost < nearby_cost and self._collision_free(new_node, nearby, obstacles):
                # Rewire
                parents[nearby_idx] = new_idx
                costs[nearby_idx] = tentative_cost
    
    def _reconstruct_path_with_costs(
        self,
        nodes: List[np.ndarray],
        parents: List[int],
        costs: List[float],
        goal_idx: int,
        final_goal: np.ndarray
    ) -> List[np.ndarray]:
        """Reconstruct path with cost information"""
        path = [final_goal]
        current_idx = goal_idx
        
        while current_idx != -1:
            path.append(nodes[current_idx])
            current_idx = parents[current_idx]
        
        return list(reversed(path))