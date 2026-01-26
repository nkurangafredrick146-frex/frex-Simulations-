"""
Physics Simulator for 3D Worlds
Advanced physics simulation for dynamic 3D scenes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Set
from dataclasses import dataclass, field
import logging
from enum import Enum
from scipy.spatial import KDTree
import trimesh
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PhysicsMaterial(Enum):
    """Material properties for physics simulation"""
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    ELASTIC = "elastic"
    RIGID = "rigid"
    DEFORMABLE = "deformable"

@dataclass
class PhysicsConfig:
    """Configuration for physics simulation"""
    # Simulation parameters
    time_step: float = 0.016  # 60 FPS
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    max_iterations: int = 100
    tolerance: float = 1e-4
    
    # Material properties
    default_density: float = 1000.0  # kg/m³
    default_friction: float = 0.5
    default_restitution: float = 0.3
    default_youngs_modulus: float = 1e6  # Pa
    default_poisson_ratio: float = 0.3
    
    # Fluid dynamics
    fluid_viscosity: float = 0.01
    fluid_density: float = 997.0
    surface_tension: float = 0.072
    
    # Collision detection
    collision_margin: float = 0.01
    broad_phase_type: str = "grid"  # "grid", "tree", "hash"
    narrow_phase_type: str = "gjk"  # "gjk", "sat", "mpr"
    
    # Performance
    use_gpu: bool = True
    parallel_processing: bool = True
    max_particles: int = 10000
    max_contacts: int = 1000
    
    # Advanced
    enable_wind: bool = True
    wind_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    enable_thermal: bool = False
    thermal_expansion: float = 0.0001

@dataclass
class PhysicsBody:
    """Represents a physical body in the simulation"""
    id: str
    vertices: np.ndarray  # Shape: [N, 3]
    faces: Optional[np.ndarray] = None  # Shape: [M, 3] for meshes
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))  # Quaternion
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = 1.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    material: PhysicsMaterial = PhysicsMaterial.RIGID
    density: float = 1000.0
    friction: float = 0.5
    restitution: float = 0.3
    youngs_modulus: float = 1e6
    poisson_ratio: float = 0.3
    is_static: bool = False
    is_kinematic: bool = False
    collision_mask: int = 0xFFFF
    user_data: Any = None
    
    def __post_init__(self):
        """Initialize derived properties"""
        if self.faces is not None and self.inertia is None:
            self.inertia = self._compute_inertia_tensor()
    
    def _compute_inertia_tensor(self) -> np.ndarray:
        """Compute inertia tensor for mesh"""
        # Simplified inertia tensor computation
        # In production, would use proper mesh inertia calculation
        volume = self._compute_volume()
        radius = np.cbrt(volume * 3 / (4 * np.pi))
        
        # Sphere approximation
        I = 0.4 * self.mass * radius**2
        return np.diag([I, I, I])
    
    def _compute_volume(self) -> float:
        """Compute mesh volume"""
        if self.faces is None:
            # Bounding box volume
            bounds = np.ptp(self.vertices, axis=0)
            return np.prod(bounds)
        
        # Mesh volume using signed tetrahedra
        volume = 0.0
        for face in self.faces:
            v1, v2, v3 = self.vertices[face]
            volume += np.dot(v1, np.cross(v2, v3)) / 6.0
        
        return abs(volume)

@dataclass
class ContactPoint:
    """Contact point between two bodies"""
    point: np.ndarray  # World-space contact point
    normal: np.ndarray  # Contact normal (from body1 to body2)
    penetration: float  # Penetration depth
    friction: Tuple[float, float] = (0.5, 0.5)  # Static and dynamic friction
    restitution: float = 0.3
    impulse: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tangent_impulse: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def get_relative_velocity(self, body1: PhysicsBody, body2: PhysicsBody) -> np.ndarray:
        """Get relative velocity at contact point"""
        # Body1 velocity at contact point
        r1 = self.point - body1.position
        v1 = body1.velocity + np.cross(body1.angular_velocity, r1)
        
        # Body2 velocity at contact point
        r2 = self.point - body2.position
        v2 = body2.velocity + np.cross(body2.angular_velocity, r2)
        
        return v1 - v2

class PhysicsSimulator:
    """Main physics simulator class"""
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        self.bodies: Dict[str, PhysicsBody] = {}
        self.constraints: List[Any] = []
        self.contact_points: List[ContactPoint] = []
        self.time = 0.0
        self.frame = 0
        
        # Initialize physics engine
        self._init_physics_engine()
        
        # Performance tracking
        self.timings = {}
        self.collision_stats = {}
        
        logger.info("Physics simulator initialized")
    
    def _init_physics_engine(self):
        """Initialize physics engine components"""
        # Initialize collision detection
        self.broad_phase = self._create_broad_phase()
        self.narrow_phase = self._create_narrow_phase()
        
        # Initialize constraint solver
        self.constraint_solver = ConstraintSolver(
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance
        )
        
        # Initialize spatial acceleration structures
        self.spatial_hash = SpatialHash(cell_size=1.0)
        
        # GPU acceleration if available
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu")
        if self.device.type == "cuda":
            logger.info(f"Using GPU acceleration: {torch.cuda.get_device_name(self.device)}")
        
        # Initialize particle system for fluids
        self.particle_system = ParticleSystem(
            max_particles=self.config.max_particles,
            device=self.device
        )
    
    def _create_broad_phase(self):
        """Create broad phase collision detector"""
        if self.config.broad_phase_type == "grid":
            return GridBroadPhase(cell_size=2.0)
        elif self.config.broad_phase_type == "tree":
            return AABBTree()
        elif self.config.broad_phase_type == "hash":
            return SpatialHash(cell_size=1.0)
        else:
            raise ValueError(f"Unknown broad phase type: {self.config.broad_phase_type}")
    
    def _create_narrow_phase(self):
        """Create narrow phase collision detector"""
        if self.config.narrow_phase_type == "gjk":
            return GJKCollisionDetector()
        elif self.config.narrow_phase_type == "sat":
            return SATCollisionDetector()
        elif self.config.narrow_phase_type == "mpr":
            return MPRCollisionDetector()
        else:
            raise ValueError(f"Unknown narrow phase type: {self.config.narrow_phase_type}")
    
    def add_body(self, body: PhysicsBody):
        """Add a body to the simulation"""
        self.bodies[body.id] = body
        self.broad_phase.add_body(body)
        
        # Add to particle system if fluid
        if body.material in [PhysicsMaterial.LIQUID, PhysicsMaterial.GAS]:
            self.particle_system.add_particles_from_body(body)
        
        logger.debug(f"Added body {body.id} to simulation")
    
    def remove_body(self, body_id: str):
        """Remove a body from the simulation"""
        if body_id in self.bodies:
            body = self.bodies[body_id]
            self.broad_phase.remove_body(body)
            
            if body.material in [PhysicsMaterial.LIQUID, PhysicsMaterial.GAS]:
                self.particle_system.remove_particles(body_id)
            
            del self.bodies[body_id]
            logger.debug(f"Removed body {body_id} from simulation")
    
    def step(self, dt: Optional[float] = None):
        """Advance simulation by one time step"""
        dt = dt or self.config.time_step
        start_time = time.time()
        
        # Clear previous contacts
        self.contact_points.clear()
        
        # Step 1: Apply external forces
        self._apply_external_forces(dt)
        
        # Step 2: Detect collisions
        self._detect_collisions()
        
        # Step 3: Solve constraints and contacts
        self._solve_constraints(dt)
        
        # Step 4: Update particle systems
        self._update_particles(dt)
        
        # Step 5: Integrate positions
        self._integrate_positions(dt)
        
        # Step 6: Update broad phase
        self._update_broad_phase()
        
        # Update time
        self.time += dt
        self.frame += 1
        
        # Record timing
        elapsed = time.time() - start_time
        self.timings[self.frame] = elapsed
        
        if self.frame % 100 == 0:
            logger.debug(f"Physics step {self.frame}: {elapsed*1000:.1f}ms")
    
    def _apply_external_forces(self, dt: float):
        """Apply gravity and other external forces"""
        for body in self.bodies.values():
            if body.is_static or body.is_kinematic:
                continue
            
            # Apply gravity
            if not body.is_kinematic:
                body.velocity += np.array(self.config.gravity) * dt
            
            # Apply wind if enabled
            if self.config.enable_wind:
                wind_force = self._compute_wind_force(body)
                body.velocity += wind_force * dt / body.mass
            
            # Apply damping
            body.velocity *= 0.999
            body.angular_velocity *= 0.995
    
    def _compute_wind_force(self, body: PhysicsBody) -> np.ndarray:
        """Compute wind force on body"""
        # Simplified wind model
        relative_velocity = np.array(self.config.wind_velocity) - body.velocity
        
        # Drag force: F = 0.5 * ρ * v² * A * Cd
        rho = 1.225  # Air density at sea level
        A = self._compute_cross_sectional_area(body)
        Cd = 0.47  # Sphere drag coefficient
        
        v_squared = np.dot(relative_velocity, relative_velocity)
        drag_magnitude = 0.5 * rho * v_squared * A * Cd
        
        if v_squared > 0:
            direction = relative_velocity / np.sqrt(v_squared)
            return drag_magnitude * direction
        
        return np.zeros(3)
    
    def _compute_cross_sectional_area(self, body: PhysicsBody) -> float:
        """Compute approximate cross-sectional area for wind resistance"""
        # Project vertices onto wind direction
        wind_dir = np.array(self.config.wind_velocity)
        if np.linalg.norm(wind_dir) < 1e-6:
            return 0.0
        
        wind_dir = wind_dir / np.linalg.norm(wind_dir)
        
        # Project all vertices
        projections = body.vertices @ wind_dir
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        
        # Get vertices in the projection slice
        center = (min_proj + max_proj) / 2
        thickness = max_proj - min_proj
        
        # Find vertices near the center plane
        mask = np.abs(projections - center) < thickness * 0.1
        
        if np.sum(mask) < 3:
            return 0.0
        
        # Compute convex hull area of projected points
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(body.vertices[mask, :2])  # Project onto XY plane
            return hull.volume  # Actually area in 2D
        except:
            # Fallback: bounding box area
            bounds = np.ptp(body.vertices, axis=0)
            return bounds[0] * bounds[2]  # X * Z area
    
    def _detect_collisions(self):
        """Detect collisions between bodies"""
        # Broad phase: find potential collisions
        potential_pairs = self.broad_phase.get_potential_pairs()
        
        # Narrow phase: detailed collision detection
        for body1_id, body2_id in potential_pairs:
            if body1_id not in self.bodies or body2_id not in self.bodies:
                continue
            
            body1 = self.bodies[body1_id]
            body2 = self.bodies[body2_id]
            
            # Skip if collision masks don't overlap
            if not (body1.collision_mask & body2.collision_mask):
                continue
            
            # Check collision
            contacts = self.narrow_phase.detect_collision(body1, body2)
            
            if contacts:
                self.contact_points.extend(contacts)
                
                # Update collision statistics
                pair_key = f"{body1_id}_{body2_id}"
                self.collision_stats[pair_key] = self.collision_stats.get(pair_key, 0) + 1
    
    def _solve_constraints(self, dt: float):
        """Solve constraints and contact forces"""
        # Prepare constraint data
        constraints = []
        
        # Add contact constraints
        for contact in self.contact_points:
            # Find bodies for this contact
            body1, body2 = self._find_contact_bodies(contact)
            if body1 is None or body2 is None:
                continue
            
            constraint = ContactConstraint(contact, body1, body2)
            constraints.append(constraint)
        
        # Add other constraints
        constraints.extend(self.constraints)
        
        # Solve constraints
        if constraints:
            self.constraint_solver.solve(constraints, dt)
    
    def _find_contact_bodies(self, contact: ContactPoint) -> Tuple[Optional[PhysicsBody], Optional[PhysicsBody]]:
        """Find bodies involved in a contact"""
        # This is simplified - in production would use proper contact body tracking
        for body1 in self.bodies.values():
            for body2 in self.bodies.values():
                if body1.id >= body2.id:
                    continue
                
                # Check if contact point is near either body
                dist1 = np.min(np.linalg.norm(body1.vertices - contact.point, axis=1))
                dist2 = np.min(np.linalg.norm(body2.vertices - contact.point, axis=1))
                
                if dist1 < 1.0 and dist2 < 1.0:
                    return body1, body2
        
        return None, None
    
    def _update_particles(self, dt: float):
        """Update particle systems (fluids, gases)"""
        if self.particle_system.has_particles():
            self.particle_system.step(
                dt=dt,
                gravity=self.config.gravity,
                viscosity=self.config.fluid_viscosity,
                surface_tension=self.config.surface_tension
            )
    
    def _integrate_positions(self, dt: float):
        """Integrate positions and orientations"""
        for body in self.bodies.values():
            if body.is_static:
                continue
            
            # Update position
            body.position += body.velocity * dt
            
            # Update orientation (quaternion integration)
            if not body.is_kinematic:
                # Angular velocity quaternion
                omega = np.array([0.0, *body.angular_velocity])
                q = body.orientation
                
                # Quaternion derivative: dq/dt = 0.5 * omega * q
                dq = 0.5 * self._quaternion_multiply(omega, q)
                body.orientation += dq * dt
                body.orientation /= np.linalg.norm(body.orientation)  # Normalize
            
            # Update vertex positions for dynamic bodies
            if not body.is_static:
                self._update_body_vertices(body)
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def _update_body_vertices(self, body: PhysicsBody):
        """Update vertex positions for transformed body"""
        # Create rotation matrix from quaternion
        q = body.orientation
        R = np.array([
            [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
            [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
            [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)]
        ])
        
        # Transform vertices
        body.vertices = (body.vertices @ R.T) + body.position
    
    def _update_broad_phase(self):
        """Update broad phase acceleration structure"""
        for body in self.bodies.values():
            if not body.is_static:
                self.broad_phase.update_body(body)
    
    def add_force(self, body_id: str, force: np.ndarray, position: Optional[np.ndarray] = None):
        """Add force to a body at a specific position"""
        if body_id not in self.bodies:
            logger.warning(f"Body {body_id} not found")
            return
        
        body = self.bodies[body_id]
        
        if body.is_static:
            return
        
        # Linear acceleration
        body.velocity += force / body.mass * self.config.time_step
        
        # Torque if position specified
        if position is not None:
            r = position - body.position
            torque = np.cross(r, force)
            angular_acc = np.linalg.solve(body.inertia, torque)
            body.angular_velocity += angular_acc * self.config.time_step
    
    def add_impulse(self, body_id: str, impulse: np.ndarray, position: Optional[np.ndarray] = None):
        """Add impulse to a body at a specific position"""
        if body_id not in self.bodies:
            logger.warning(f"Body {body_id} not found")
            return
        
        body = self.bodies[body_id]
        
        if body.is_static:
            return
        
        # Linear impulse
        body.velocity += impulse / body.mass
        
        # Angular impulse if position specified
        if position is not None:
            r = position - body.position
            angular_impulse = np.cross(r, impulse)
            body.angular_velocity += np.linalg.solve(body.inertia, angular_impulse)
    
    def raycast(self, origin: np.ndarray, direction: np.ndarray, max_distance: float = 100.0) -> Optional[Dict[str, Any]]:
        """Cast a ray and return hit information"""
        direction = direction / np.linalg.norm(direction)
        end = origin + direction * max_distance
        
        closest_hit = None
        closest_distance = max_distance
        
        for body in self.bodies.values():
            # Check ray against body's AABB first
            if not self._ray_aabb_intersect(origin, direction, body):
                continue
            
            # Detailed ray-mesh intersection
            hit = self._ray_mesh_intersect(origin, direction, body)
            
            if hit and hit["distance"] < closest_distance:
                closest_hit = hit
                closest_distance = hit["distance"]
        
        return closest_hit
    
    def _ray_aabb_intersect(self, origin: np.ndarray, direction: np.ndarray, body: PhysicsBody) -> bool:
        """Check ray against axis-aligned bounding box"""
        # Compute AABB
        min_bounds = np.min(body.vertices, axis=0)
        max_bounds = np.max(body.vertices, axis=0)
        
        # Ray-AABB intersection (slab method)
        tmin = 0.0
        tmax = float('inf')
        
        for i in range(3):
            if abs(direction[i]) < 1e-6:
                # Ray is parallel to slab
                if origin[i] < min_bounds[i] or origin[i] > max_bounds[i]:
                    return False
            else:
                t1 = (min_bounds[i] - origin[i]) / direction[i]
                t2 = (max_bounds[i] - origin[i]) / direction[i]
                
                if t1 > t2:
                    t1, t2 = t2, t1
                
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                
                if tmin > tmax:
                    return False
        
        return tmax > 0
    
    def _ray_mesh_intersect(self, origin: np.ndarray, direction: np.ndarray, body: PhysicsBody) -> Optional[Dict[str, Any]]:
        """Ray-mesh intersection test"""
        if body.faces is None:
            return None
        
        closest_hit = None
        closest_distance = float('inf')
        
        # Check each triangle
        for face in body.faces:
            v0, v1, v2 = body.vertices[face]
            
            # Möller–Trumbore intersection algorithm
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(direction, edge2)
            a = np.dot(edge1, h)
            
            if abs(a) < 1e-6:
                continue  # Ray parallel to triangle
            
            f = 1.0 / a
            s = origin - v0
            u = f * np.dot(s, h)
            
            if u < 0.0 or u > 1.0:
                continue
            
            q = np.cross(s, edge1)
            v = f * np.dot(direction, q)
            
            if v < 0.0 or u + v > 1.0:
                continue
            
            t = f * np.dot(edge2, q)
            
            if t > 1e-6 and t < closest_distance:
                closest_distance = t
                hit_point = origin + direction * t
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                closest_hit = {
                    "body": body,
                    "point": hit_point,
                    "normal": normal,
                    "distance": t,
                    "face": face,
                    "uv": (u, v)
                }
        
        return closest_hit
    
    def query_sphere(self, center: np.ndarray, radius: float) -> List[PhysicsBody]:
        """Find all bodies within a sphere"""
        results = []
        
        for body in self.bodies.values():
            # Compute distance to body's AABB
            min_bounds = np.min(body.vertices, axis=0)
            max_bounds = np.max(body.vertices, axis=0)
            
            # Closest point on AABB to sphere center
            closest = np.clip(center, min_bounds, max_bounds)
            distance = np.linalg.norm(center - closest)
            
            if distance <= radius:
                results.append(body)
        
        return results
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        state = {
            "time": self.time,
            "frame": self.frame,
            "bodies": {},
            "contacts": len(self.contact_points),
            "performance": self.timings
        }
        
        for body_id, body in self.bodies.items():
            state["bodies"][body_id] = {
                "position": body.position.tolist(),
                "orientation": body.orientation.tolist(),
                "velocity": body.velocity.tolist(),
                "angular_velocity": body.angular_velocity.tolist()
            }
        
        return state
    
    def save_state(self, filepath: Union[str, Path]):
        """Save simulation state to file"""
        state = self.get_state()
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved physics state to {filepath}")
    
    def load_state(self, filepath: Union[str, Path]):
        """Load simulation state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.time = state["time"]
        self.frame = state["frame"]
        
        for body_id, body_state in state["bodies"].items():
            if body_id in self.bodies:
                body = self.bodies[body_id]
                body.position = np.array(body_state["position"])
                body.orientation = np.array(body_state["orientation"])
                body.velocity = np.array(body_state["velocity"])
                body.angular_velocity = np.array(body_state["angular_velocity"])
                
                # Update vertex positions
                self._update_body_vertices(body)
        
        logger.info(f"Loaded physics state from {filepath}")
    
    def reset(self):
        """Reset simulation to initial state"""
        self.bodies.clear()
        self.constraints.clear()
        self.contact_points.clear()
        self.time = 0.0
        self.frame = 0
        self.timings.clear()
        self.collision_stats.clear()
        
        # Reset acceleration structures
        self.broad_phase.clear()
        self.particle_system.clear()
        
        logger.info("Physics simulation reset")

class ConstraintSolver:
    """Solver for physics constraints"""
    
    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def solve(self, constraints: List[Any], dt: float):
        """Solve constraints using iterative method"""
        if not constraints:
            return
        
        # Warm starting
        for constraint in constraints:
            constraint.warm_start(dt)
        
        # Iterative solver
        for iteration in range(self.max_iterations):
            max_error = 0.0
            
            for constraint in constraints:
                error = constraint.solve(dt)
                max_error = max(max_error, abs(error))
            
            # Early convergence
            if max_error < self.tolerance:
                break

class ContactConstraint:
    """Constraint for contact resolution"""
    
    def __init__(self, contact: ContactPoint, body1: PhysicsBody, body2: PhysicsBody):
        self.contact = contact
        self.body1 = body1
        self.body2 = body2
        
        # Precompute Jacobian
        self._compute_jacobian()
    
    def _compute_jacobian(self):
        """Compute constraint Jacobian"""
        r1 = self.contact.point - self.body1.position
        r2 = self.contact.point - self.body2.position
        
        # Normal Jacobian
        self.normal_jacobian1 = np.hstack([-self.contact.normal, -np.cross(r1, self.contact.normal)])
        self.normal_jacobian2 = np.hstack([self.contact.normal, np.cross(r2, self.contact.normal)])
        
        # Tangent Jacobians (for friction)
        tangent1 = self._compute_tangent_vectors(self.contact.normal)
        self.tangent_jacobian1 = [np.hstack([-t, -np.cross(r1, t)]) for t in tangent1]
        self.tangent_jacobian2 = [np.hstack([t, np.cross(r2, t)]) for t in tangent1]
    
    def _compute_tangent_vectors(self, normal: np.ndarray) -> List[np.ndarray]:
        """Compute orthogonal tangent vectors"""
        # Choose arbitrary vector not parallel to normal
        if abs(normal[0]) > 0.1:
            tangent1 = np.array([normal[1], -normal[0], 0.0])
        else:
            tangent1 = np.array([0.0, normal[2], -normal[1]])
        
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        
        return [tangent1, tangent2]
    
    def warm_start(self, dt: float):
        """Warm start the constraint"""
        # Apply previous impulses
        pass  # Implement warm starting
    
    def solve(self, dt: float) -> float:
        """Solve contact constraint"""
        # Compute relative velocity
        v_rel = self.contact.get_relative_velocity(self.body1, self.body2)
        vn = np.dot(v_rel, self.contact.normal)
        
        # Restitution
        if vn < -1.0:  # Colliding
            desired_vn = -self.contact.restitution * vn
        else:  # Resting contact
            desired_vn = 0.0
        
        # Constraint error
        error = desired_vn - vn
        
        # Compute effective mass
        inv_mass1 = 1.0 / self.body1.mass if not self.body1.is_static else 0.0
        inv_mass2 = 1.0 / self.body2.mass if not self.body2.is_static else 0.0
        
        inv_inertia1 = np.linalg.inv(self.body1.inertia) if not self.body1.is_static else np.zeros((3, 3))
        inv_inertia2 = np.linalg.inv(self.body2.inertia) if not self.body2.is_static else np.zeros((3, 3))
        
        # Effective mass for normal direction
        r1 = self.contact.point - self.body1.position
        r2 = self.contact.point - self.body2.position
        
        k_normal = inv_mass1 + inv_mass2
        k_normal += np.dot(np.cross(r1, self.contact.normal), inv_inertia1 @ np.cross(r1, self.contact.normal))
        k_normal += np.dot(np.cross(r2, self.contact.normal), inv_inertia2 @ np.cross(r2, self.contact.normal))
        
        if k_normal < 1e-6:
            return 0.0
        
        # Impulse
        lambda_n = error / k_normal
        
        # Apply impulse
        impulse = lambda_n * self.contact.normal
        
        if not self.body1.is_static:
            self.body1.velocity -= impulse / self.body1.mass
            self.body1.angular_velocity -= inv_inertia1 @ np.cross(r1, impulse)
        
        if not self.body2.is_static:
            self.body2.velocity += impulse / self.body2.mass
            self.body2.angular_velocity += inv_inertia2 @ np.cross(r2, impulse)
        
        return error

class ParticleSystem:
    """Particle system for fluid and gas simulation"""
    
    def __init__(self, max_particles: int = 10000, device: torch.device = torch.device("cpu")):
        self.max_particles = max_particles
        self.device = device
        
        # Particle buffers
        self.positions = torch.zeros((max_particles, 3), device=device)
        self.velocities = torch.zeros((max_particles, 3), device=device)
        self.forces = torch.zeros((max_particles, 3), device=device)
        self.densities = torch.zeros(max_particles, device=device)
        self.pressures = torch.zeros(max_particles, device=device)
        
        # Particle properties
        self.masses = torch.ones(max_particles, device=device)
        self.radii = torch.ones(max_particles, device=device) * 0.1
        
        # Body mapping
        self.body_ids = [""] * max_particles
        self.particle_count = 0
        
        # Spatial acceleration
        self.grid = SpatialHash(cell_size=0.2)
    
    def add_particles_from_body(self, body: PhysicsBody):
        """Add particles from a physics body"""
        if body.material not in [PhysicsMaterial.LIQUID, PhysicsMaterial.GAS]:
            return
        
        # Convert mesh vertices to particles
        vertices = torch.from_numpy(body.vertices).float().to(self.device)
        num_particles = min(len(vertices), self.max_particles - self.particle_count)
        
        if num_particles <= 0:
            return
        
        start_idx = self.particle_count
        end_idx = start_idx + num_particles
        
        self.positions[start_idx:end_idx] = vertices[:num_particles]
        self.velocities[start_idx:end_idx] = torch.from_numpy(body.velocity).float().to(self.device)
        self.masses[start_idx:end_idx] = body.mass / num_particles
        self.radii[start_idx:end_idx] = 0.1
        
        # Set body ID
        for i in range(start_idx, end_idx):
            self.body_ids[i] = body.id
        
        self.particle_count = end_idx
        
        # Update spatial grid
        self._update_spatial_grid()
    
    def step(self, dt: float, gravity: Tuple[float, float, float], viscosity: float, surface_tension: float):
        """Advance particle system by one time step"""
        if self.particle_count == 0:
            return
        
        # Convert to tensors
        gravity_tensor = torch.tensor(gravity, device=self.device).float()
        
        # Step 1: Update neighborhoods
        neighborhoods = self._find_neighborhoods()
        
        # Step 2: Compute densities and pressures
        self._compute_densities_pressures(neighborhoods)
        
        # Step 3: Compute forces
        self._compute_forces(neighborhoods, viscosity, surface_tension)
        
        # Step 4: Apply gravity
        self.forces[:self.particle_count] += gravity_tensor * self.masses[:self.particle_count].unsqueeze(1)
        
        # Step 5: Integrate
        self._integrate_particles(dt)
        
        # Step 6: Handle collisions
        self._handle_particle_collisions()
        
        # Step 7: Update spatial grid
        self._update_spatial_grid()
    
    def _find_neighborhoods(self) -> List[List[int]]:
        """Find neighboring particles using spatial grid"""
        neighborhoods = [[] for _ in range(self.particle_count)]
        
        for i in range(self.particle_count):
            pos_i = self.positions[i].cpu().numpy()
            neighbors = self.grid.query_sphere(pos_i, 0.2)  # Interaction radius
            
            for j in neighbors:
                if i != j:
                    neighborhoods[i].append(j)
        
        return neighborhoods
    
    def _compute_densities_pressures(self, neighborhoods: List[List[int]]):
        """Compute densities and pressures using SPH"""
        kernel_radius = 0.2
        
        for i in range(self.particle_count):
            density = 0.0
            
            # Self contribution
            density += self.masses[i] * self._sph_kernel(0.0, kernel_radius)
            
            # Neighbor contributions
            for j in neighborhoods[i]:
                r_ij = self.positions[i] - self.positions[j]
                dist = torch.norm(r_ij)
                
                if dist < kernel_radius:
                    density += self.masses[j] * self._sph_kernel(dist, kernel_radius)
            
            self.densities[i] = density
            
            # Compute pressure (ideal gas equation of state)
            rest_density = 1000.0
            stiffness = 100.0
            self.pressures[i] = stiffness * (self.densities[i] - rest_density)
    
    def _sph_kernel(self, r: float, h: float) -> float:
        """SPH cubic spline kernel"""
        q = r / h
        
        if q <= 1.0:
            return (1.0 - 1.5 * q**2 + 0.75 * q**3) / (np.pi * h**3)
        elif q <= 2.0:
            return 0.25 * (2.0 - q)**3 / (np.pi * h**3)
        else:
            return 0.0
    
    def _compute_forces(self, neighborhoods: List[List[int]], viscosity: float, surface_tension: float):
        """Compute forces between particles"""
        kernel_radius = 0.2
        
        # Reset forces
        self.forces.zero_()
        
        for i in range(self.particle_count):
            pressure_force = torch.zeros(3, device=self.device)
            viscosity_force = torch.zeros(3, device=self.device)
            
            for j in neighborhoods[i]:
                if i == j:
                    continue
                
                r_ij = self.positions[i] - self.positions[j]
                dist = torch.norm(r_ij)
                
                if dist < 1e-6:
                    continue
                
                # Pressure force
                if dist < kernel_radius:
                    grad_kernel = self._sph_kernel_gradient(r_ij, dist, kernel_radius)
                    pressure_force += -self.masses[j] * (self.pressures[i] + self.pressures[j]) / \
                                     (2.0 * self.densities[j]) * grad_kernel
                
                # Viscosity force
                v_ij = self.velocities[i] - self.velocities[j]
                viscosity_force += viscosity * self.masses[j] * v_ij / self.densities[j] * \
                                 self._sph_kernel_laplacian(dist, kernel_radius)
            
            # Surface tension (simplified)
            if len(neighborhoods[i]) > 0:
                # Compute color field gradient
                color_gradient = torch.zeros(3, device=self.device)
                for j in neighborhoods[i]:
                    r_ij = self.positions[i] - self.positions[j]
                    dist = torch.norm(r_ij)
                    color_gradient += self.masses[j] / self.densities[j] * \
                                     self._sph_kernel_gradient(r_ij, dist, kernel_radius)
                
                # Surface tension force
                color_laplacian = 0.0
                for j in neighborhoods[i]:
                    r_ij = self.positions[i] - self.positions[j]
                    dist = torch.norm(r_ij)
                    color_laplacian += self.masses[j] / self.densities[j] * \
                                      self._sph_kernel_laplacian(dist, kernel_radius)
                
                tension_force = -surface_tension * color_laplacian * color_gradient / torch.norm(color_gradient)
            else:
                tension_force = torch.zeros(3, device=self.device)
            
            # Combine forces
            self.forces[i] = pressure_force + viscosity_force + tension_force
    
    def _sph_kernel_gradient(self, r_ij: torch.Tensor, r: float, h: float) -> torch.Tensor:
        """Gradient of SPH kernel"""
        if r < 1e-6:
            return torch.zeros(3, device=r_ij.device)
        
        q = r / h
        grad = torch.zeros(3, device=r_ij.device)
        
        if q <= 1.0:
            grad = r_ij * (-3.0 * q + 2.25 * q**2) / (np.pi * h**5)
        elif q <= 2.0:
            grad = r_ij * -0.75 * (2.0 - q)**2 / (r * np.pi * h**4)
        
        return grad
    
    def _sph_kernel_laplacian(self, r: float, h: float) -> float:
        """Laplacian of SPH kernel"""
        q = r / h
        
        if q <= 1.0:
            return (9.0 / (4.0 * np.pi * h**5)) * (1.0 - q)
        elif q <= 2.0:
            return (3.0 / (4.0 * np.pi * h**5)) * (2.0 - q)
        else:
            return 0.0
    
    def _integrate_particles(self, dt: float):
        """Integrate particle positions and velocities"""
        # Update velocities
        self.velocities[:self.particle_count] += self.forces[:self.particle_count] / \
                                                 self.masses[:self.particle_count].unsqueeze(1) * dt
        
        # Update positions
        self.positions[:self.particle_count] += self.velocities[:self.particle_count] * dt
        
        # Apply damping
        self.velocities[:self.particle_count] *= 0.99
    
    def _handle_particle_collisions(self):
        """Handle particle collisions with boundaries"""
        # Simple boundary constraints
        min_bound = torch.tensor([-10.0, 0.0, -10.0], device=self.device)
        max_bound = torch.tensor([10.0, 10.0, 10.0], device=self.device)
        
        for i in range(self.particle_count):
            # Check boundaries
            for dim in range(3):
                if self.positions[i, dim] < min_bound[dim]:
                    self.positions[i, dim] = min_bound[dim]
                    self.velocities[i, dim] = -self.velocities[i, dim] * 0.5
                elif self.positions[i, dim] > max_bound[dim]:
                    self.positions[i, dim] = max_bound[dim]
                    self.velocities[i, dim] = -self.velocities[i, dim] * 0.5
    
    def _update_spatial_grid(self):
        """Update spatial hash grid for acceleration"""
        self.grid.clear()
        
        for i in range(self.particle_count):
            pos = self.positions[i].cpu().numpy()
            self.grid.insert(i, pos)
    
    def has_particles(self) -> bool:
        """Check if system has particles"""
        return self.particle_count > 0
    
    def clear(self):
        """Clear all particles"""
        self.positions.zero_()
        self.velocities.zero_()
        self.forces.zero_()
        self.densities.zero_()
        self.pressures.zero_()
        self.particle_count = 0
        self.body_ids = [""] * self.max_particles
        self.grid.clear()
    
    def remove_particles(self, body_id: str):
        """Remove particles belonging to a body"""
        indices_to_keep = []
        
        for i in range(self.particle_count):
            if self.body_ids[i] != body_id:
                indices_to_keep.append(i)
        
        # Reorganize arrays
        new_count = len(indices_to_keep)
        
        self.positions[:new_count] = self.positions[indices_to_keep]
        self.velocities[:new_count] = self.velocities[indices_to_keep]
        self.forces[:new_count] = self.forces[indices_to_keep]
        self.densities[:new_count] = self.densities[indices_to_keep]
        self.pressures[:new_count] = self.pressures[indices_to_keep]
        self.masses[:new_count] = self.masses[indices_to_keep]
        self.radii[:new_count] = self.radii[indices_to_keep]
        
        # Update body IDs
        self.body_ids = [self.body_ids[i] for i in indices_to_keep] + \
                       [""] * (self.max_particles - new_count)
        
        self.particle_count = new_count
        
        # Update spatial grid
        self._update_spatial_grid()

class SpatialHash:
    """Spatial hash for broad phase collision detection"""
    
    def __init__(self, cell_size: float = 1.0):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], Set[int]] = {}
        self.objects: Dict[int, Tuple[float, float, float]] = {}
    
    def insert(self, obj_id: int, position: np.ndarray):
        """Insert object into spatial hash"""
        cell = self._get_cell(position)
        
        if cell not in self.grid:
            self.grid[cell] = set()
        
        self.grid[cell].add(obj_id)
        self.objects[obj_id] = position
    
    def query_sphere(self, center: np.ndarray, radius: float) -> List[int]:
        """Query objects within sphere"""
        results = []
        
        # Get range of cells to check
        min_cell = self._get_cell(center - radius)
        max_cell = self._get_cell(center + radius)
        
        for x in range(min_cell[0], max_cell[0] + 1):
            for y in range(min_cell[1], max_cell[1] + 1):
                for z in range(min_cell[2], max_cell[2] + 1):
                    cell = (x, y, z)
                    
                    if cell in self.grid:
                        for obj_id in self.grid[cell]:
                            pos = self.objects[obj_id]
                            if np.linalg.norm(pos - center) <= radius:
                                results.append(obj_id)
        
        return results
    
    def _get_cell(self, position: np.ndarray) -> Tuple[int, int, int]:
        """Convert position to cell coordinates"""
        return (
            int(position[0] // self.cell_size),
            int(position[1] // self.cell_size),
            int(position[2] // self.cell_size)
        )
    
    def clear(self):
        """Clear spatial hash"""
        self.grid.clear()
        self.objects.clear()

# Additional helper classes for collision detection

class GridBroadPhase:
    """Grid-based broad phase collision detection"""
    pass  # Implementation would go here

class AABBTree:
    """AABB tree for broad phase collision detection"""
    pass  # Implementation would go here

class GJKCollisionDetector:
    """GJK algorithm for narrow phase collision detection"""
    pass  # Implementation would go here

class SATCollisionDetector:
    """Separating Axis Theorem for narrow phase collision detection"""
    pass  # Implementation would go here

class MPRCollisionDetector:
    """MPR (Minkowski Portal Refinement) for narrow phase collision detection"""
    pass  # Implementation would go here