"""
Scene Composition Module
Composes and arranges 3D scenes from generated elements
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
from collections import defaultdict
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

# Local imports
from ...utils.metrics import Timer
from ...utils.file_io import save_json, load_json, save_pickle, load_pickle

logger = logging.getLogger(__name__)


class CompositionMode(Enum):
    """Modes for scene composition"""
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"


class SceneLayout(Enum):
    """Types of scene layouts"""
    OPEN_WORLD = "open_world"
    INTERIOR = "interior"
    CITYSCAPE = "cityscape"
    NATURAL = "natural"
    ABSTRACT = "abstract"
    MIXED = "mixed"


@dataclass
class SceneObject:
    """Represents an object in the scene"""
    id: str
    mesh: trimesh.Trimesh
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    semantic_label: str = ""
    material: Dict[str, Any] = field(default_factory=dict)
    physics_properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def transform(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        transform = np.eye(4)
        transform[:3, :3] = self.rotation @ np.diag(self.scale)
        transform[:3, 3] = self.position
        return transform
    
    @property
    def bounds(self) -> np.ndarray:
        """Get axis-aligned bounding box in world coordinates"""
        local_bounds = self.mesh.bounds
        transformed_vertices = (self.rotation @ (local_bounds * self.scale).T).T + self.position
        return np.array([
            transformed_vertices.min(axis=0),
            transformed_vertices.max(axis=0)
        ])
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside object bounds"""
        return self.mesh.contains([np.linalg.inv(self.transform) @ np.append(point, 1)])[0]


@dataclass
class CompositionConstraints:
    """Constraints for scene composition"""
    spatial_constraints: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    semantic_constraints: Dict[str, List[str]] = field(default_factory=dict)
    density_constraints: Dict[str, float] = field(default_factory=dict)
    symmetry_constraints: Dict[str, str] = field(default_factory=dict)
    adjacency_constraints: Dict[str, List[str]] = field(default_factory=dict)
    exclusion_zones: List[Dict[str, Any]] = field(default_factory=list)


class SceneComposer:
    """
    Main scene composition engine
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mode: CompositionMode = CompositionMode.AUTOMATIC,
        layout: SceneLayout = SceneLayout.OPEN_WORLD
    ):
        """
        Initialize scene composer
        
        Args:
            config: Configuration dictionary
            mode: Composition mode
            layout: Scene layout type
        """
        self.config = config or {}
        self.mode = mode
        self.layout = layout
        self.scene_objects: Dict[str, SceneObject] = {}
        self.constraints = CompositionConstraints()
        self.spatial_tree: Optional[KDTree] = None
        self.semantic_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "total_objects": 0,
            "composition_time": 0.0,
            "collision_checks": 0,
            "iterations": 0
        }
        
        # Initialize composition rules based on layout
        self._init_composition_rules()
        
        logger.info(f"SceneComposer initialized with mode={mode}, layout={layout}")
    
    def _init_composition_rules(self) -> None:
        """Initialize composition rules based on layout"""
        self.rules = {
            SceneLayout.OPEN_WORLD: {
                "max_density": 0.1,
                "object_distribution": "clustered",
                "height_variation": 0.3,
                "scale_variation": 0.5
            },
            SceneLayout.INTERIOR: {
                "max_density": 0.7,
                "object_distribution": "structured",
                "height_variation": 0.1,
                "scale_variation": 0.2
            },
            SceneLayout.CITYSCAPE: {
                "max_density": 0.4,
                "object_distribution": "grid",
                "height_variation": 0.5,
                "scale_variation": 0.3
            },
            SceneLayout.NATURAL: {
                "max_density": 0.2,
                "object_distribution": "random",
                "height_variation": 0.4,
                "scale_variation": 0.6
            }
        }.get(self.layout, {})
    
    def compose_scene(
        self,
        objects: List[SceneObject],
        constraints: Optional[CompositionConstraints] = None,
        bounds: Tuple[float, float, float, float] = (-100, 100, -100, 100),
        max_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Compose a scene from a list of objects
        
        Args:
            objects: List of scene objects to compose
            constraints: Optional composition constraints
            bounds: Scene bounds (x_min, x_max, z_min, z_max)
            max_iterations: Maximum iterations for optimization
            
        Returns:
            Dictionary containing composed scene
        """
        timer = Timer()
        self.stats["iterations"] = 0
        
        if constraints:
            self.constraints = constraints
        
        logger.info(f"Starting scene composition with {len(objects)} objects")
        
        # Pre-process objects
        processed_objects = self._preprocess_objects(objects)
        
        # Apply composition strategy based on mode
        if self.mode == CompositionMode.AUTOMATIC:
            composed_objects = self._automatic_composition(
                processed_objects, bounds, max_iterations
            )
        elif self.mode == CompositionMode.SEMI_AUTOMATIC:
            composed_objects = self._semi_automatic_composition(
                processed_objects, bounds
            )
        else:
            composed_objects = self._manual_composition(processed_objects)
        
        # Build spatial index
        self._build_spatial_index(composed_objects)
        
        # Apply post-processing
        final_objects = self._postprocess_composition(composed_objects)
        
        # Calculate statistics
        self.stats["total_objects"] = len(final_objects)
        self.stats["composition_time"] = timer.elapsed()
        
        logger.info(f"Scene composition completed in {self.stats['composition_time']:.2f}s")
        
        return {
            "objects": final_objects,
            "bounds": bounds,
            "layout": self.layout.value,
            "mode": self.mode.value,
            "stats": self.stats.copy(),
            "spatial_tree": self.spatial_tree,
            "semantic_graph": dict(self.semantic_graph)
        }
    
    def _preprocess_objects(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Pre-process objects before composition"""
        processed = []
        
        for obj in objects:
            # Normalize scale if needed
            if self.config.get("normalize_scales", False):
                obj.scale = self._normalize_scale(obj)
            
            # Assign semantic labels if missing
            if not obj.semantic_label:
                obj.semantic_label = self._infer_semantic_label(obj)
            
            # Calculate physics properties
            if not obj.physics_properties:
                obj.physics_properties = self._calculate_physics_properties(obj)
            
            processed.append(obj)
        
        return processed
    
    def _automatic_composition(
        self,
        objects: List[SceneObject],
        bounds: Tuple[float, float, float, float],
        max_iterations: int
    ) -> List[SceneObject]:
        """Automatic scene composition using optimization"""
        x_min, x_max, z_min, z_max = bounds
        
        # Sort objects by importance (size, semantic priority)
        sorted_objects = sorted(
            objects,
            key=lambda obj: (
                -np.prod(obj.scale),  # Larger objects first
                self._get_semantic_priority(obj.semantic_label)
            )
        )
        
        composed = []
        placed_positions = []
        
        for i, obj in enumerate(sorted_objects):
            best_position = None
            best_score = -float('inf')
            best_rotation = np.eye(3)
            
            for iteration in range(max_iterations // len(sorted_objects)):
                # Generate candidate position
                if self.rules.get("object_distribution") == "grid":
                    position = self._generate_grid_position(
                        i, len(sorted_objects), bounds
                    )
                elif self.rules.get("object_distribution") == "clustered":
                    position = self._generate_clustered_position(
                        placed_positions, bounds
                    )
                else:
                    position = self._generate_random_position(bounds)
                
                # Generate candidate rotation
                rotation = self._generate_rotation(obj)
                
                # Check constraints
                if not self._check_constraints(obj, position, rotation, composed):
                    continue
                
                # Calculate placement score
                score = self._calculate_placement_score(
                    obj, position, rotation, composed
                )
                
                if score > best_score:
                    best_score = score
                    best_position = position
                    best_rotation = rotation
            
            if best_position is not None:
                obj.position = best_position
                obj.rotation = best_rotation
                composed.append(obj)
                placed_positions.append(best_position)
                
                # Update semantic graph
                self._update_semantic_graph(obj, composed)
            
            self.stats["iterations"] += max_iterations // len(sorted_objects)
        
        return composed
    
    def _semi_automatic_composition(
        self,
        objects: List[SceneObject],
        bounds: Tuple[float, float, float, float]
    ) -> List[SceneObject]:
        """Semi-automatic composition with user guidance"""
        composed = []
        
        # Group objects by semantic category
        semantic_groups = defaultdict(list)
        for obj in objects:
            semantic_groups[obj.semantic_label].append(obj)
        
        # Place objects by category
        for category, group in semantic_groups.items():
            # Get placement strategy for category
            strategy = self._get_placement_strategy(category)
            
            if strategy == "cluster":
                positions = self._generate_cluster_positions(
                    len(group), bounds
                )
            elif strategy == "line":
                positions = self._generate_line_positions(
                    len(group), bounds
                )
            else:
                positions = self._generate_random_positions(
                    len(group), bounds
                )
            
            for obj, position in zip(group, positions):
                obj.position = position
                obj.rotation = self._generate_rotation(obj)
                composed.append(obj)
        
        return composed
    
    def _manual_composition(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Manual composition (no automatic placement)"""
        return objects  # Return objects as-is, assuming positions are already set
    
    def _generate_random_position(
        self,
        bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Generate random position within bounds"""
        x_min, x_max, z_min, z_max = bounds
        x = np.random.uniform(x_min, x_max)
        z = np.random.uniform(z_min, z_max)
        y = 0  # Ground level, can be modified for terrain
        
        # Add height variation based on rules
        if "height_variation" in self.rules:
            y += np.random.uniform(-self.rules["height_variation"], self.rules["height_variation"])
        
        return np.array([x, y, z])
    
    def _generate_grid_position(
        self,
        index: int,
        total: int,
        bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Generate position in a grid layout"""
        x_min, x_max, z_min, z_max = bounds
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(total)))
        cell_width = (x_max - x_min) / grid_size
        cell_depth = (z_max - z_min) / grid_size
        
        row = index // grid_size
        col = index % grid_size
        
        x = x_min + (col + 0.5) * cell_width
        z = z_min + (row + 0.5) * cell_depth
        
        # Add small random offset
        x += np.random.uniform(-cell_width * 0.2, cell_width * 0.2)
        z += np.random.uniform(-cell_depth * 0.2, cell_depth * 0.2)
        
        return np.array([x, 0, z])
    
    def _generate_clustered_position(
        self,
        existing_positions: List[np.ndarray],
        bounds: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Generate position that clusters with existing positions"""
        if not existing_positions:
            return self._generate_random_position(bounds)
        
        # Choose a random existing position as cluster center
        center = existing_positions[np.random.randint(len(existing_positions))]
        
        # Generate position around center with Gaussian distribution
        std_dev = 5.0  # Cluster radius
        position = center + np.random.normal(0, std_dev, 3)
        position[1] = 0  # Keep on ground
        
        # Ensure within bounds
        x_min, x_max, z_min, z_max = bounds
        position[0] = np.clip(position[0], x_min, x_max)
        position[2] = np.clip(position[2], z_min, z_max)
        
        return position
    
    def _generate_rotation(self, obj: SceneObject) -> np.ndarray:
        """Generate rotation matrix for object"""
        # Align with ground plane
        rotation = R.from_euler('z', np.random.uniform(0, 360), degrees=True).as_matrix()
        
        # Apply semantic-specific rotations
        if obj.semantic_label in ["tree", "plant"]:
            # Keep upright
            pass
        elif obj.semantic_label in ["chair", "table"]:
            # Align with floor
            rotation = R.from_euler('y', np.random.choice([0, 90, 180, 270]), degrees=True).as_matrix()
        
        return rotation
    
    def _check_constraints(
        self,
        obj: SceneObject,
        position: np.ndarray,
        rotation: np.ndarray,
        existing_objects: List[SceneObject]
    ) -> bool:
        """Check if placement satisfies all constraints"""
        # Check spatial constraints
        if obj.semantic_label in self.constraints.spatial_constraints:
            constraints = self.constraints.spatial_constraints[obj.semantic_label]
            for i, (min_val, max_val) in enumerate(constraints):
                if not (min_val <= position[i] <= max_val):
                    return False
        
        # Check semantic constraints
        if obj.semantic_label in self.constraints.semantic_constraints:
            allowed_categories = self.constraints.semantic_constraints[obj.semantic_label]
            # Check adjacency to allowed categories
            pass
        
        # Check density constraints
        density = self._calculate_local_density(position, existing_objects)
        max_density = self.constraints.density_constraints.get(
            obj.semantic_label,
            self.rules.get("max_density", 0.5)
        )
        if density > max_density:
            return False
        
        # Check collision with existing objects
        temp_obj = SceneObject(
            id=obj.id + "_temp",
            mesh=obj.mesh,
            position=position,
            rotation=rotation,
            scale=obj.scale
        )
        
        for existing in existing_objects:
            if self._check_collision(temp_obj, existing):
                return False
        
        # Check exclusion zones
        for zone in self.constraints.exclusion_zones:
            if self._point_in_zone(position, zone):
                return False
        
        self.stats["collision_checks"] += 1
        return True
    
    def _calculate_placement_score(
        self,
        obj: SceneObject,
        position: np.ndarray,
        rotation: np.ndarray,
        existing_objects: List[SceneObject]
    ) -> float:
        """Calculate score for a placement position"""
        score = 0.0
        
        # Distance to nearest object (prefer moderate spacing)
        if existing_objects:
            distances = [np.linalg.norm(position - obj2.position) 
                        for obj2 in existing_objects]
            min_distance = min(distances)
            
            if min_distance < 1.0:  # Too close
                score -= 10.0
            elif 1.0 <= min_distance <= 5.0:  # Good spacing
                score += 5.0
            else:  # Too far
                score += 1.0
        
        # Alignment with semantic neighbors
        for existing in existing_objects:
            if existing.semantic_label == obj.semantic_label:
                distance = np.linalg.norm(position - existing.position)
                if distance < 10.0:  # Prefer clustering of same category
                    score += 2.0
        
        # Orientation score (face outward in clusters)
        if len(existing_objects) >= 3:
            # Calculate centroid of existing objects
            centroid = np.mean([obj2.position for obj2 in existing_objects], axis=0)
            direction_to_centroid = centroid - position
            direction_to_centroid[1] = 0  # Ignore vertical
            
            if np.linalg.norm(direction_to_centroid) > 0:
                forward = rotation @ np.array([0, 0, 1])  # Object forward vector
                dot_product = np.dot(forward, direction_to_centroid / np.linalg.norm(direction_to_centroid))
                
                if dot_product < -0.5:  # Facing away from centroid (good)
                    score += 3.0
        
        return score
    
    def _check_collision(
        self,
        obj1: SceneObject,
        obj2: SceneObject
    ) -> bool:
        """Check collision between two objects"""
        # Simple AABB collision check
        bounds1 = obj1.bounds
        bounds2 = obj2.bounds
        
        return (
            bounds1[0][0] < bounds2[1][0] and
            bounds1[1][0] > bounds2[0][0] and
            bounds1[0][1] < bounds2[1][1] and
            bounds1[1][1] > bounds2[0][1] and
            bounds1[0][2] < bounds2[1][2] and
            bounds1[1][2] > bounds2[0][2]
        )
    
    def _calculate_local_density(
        self,
        position: np.ndarray,
        objects: List[SceneObject],
        radius: float = 10.0
    ) -> float:
        """Calculate object density around a position"""
        if not objects:
            return 0.0
        
        count = 0
        for obj in objects:
            distance = np.linalg.norm(position - obj.position)
            if distance <= radius:
                count += 1
        
        area = np.pi * radius * radius
        return count / area if area > 0 else 0.0
    
    def _point_in_zone(self, point: np.ndarray, zone: Dict[str, Any]) -> bool:
        """Check if point is inside exclusion zone"""
        zone_type = zone.get("type", "circle")
        
        if zone_type == "circle":
            center = np.array(zone["center"])
            radius = zone["radius"]
            return np.linalg.norm(point - center) <= radius
        elif zone_type == "rectangle":
            min_corner = np.array(zone["min"])
            max_corner = np.array(zone["max"])
            return (
                min_corner[0] <= point[0] <= max_corner[0] and
                min_corner[2] <= point[2] <= max_corner[2]
            )
        
        return False
    
    def _normalize_scale(self, obj: SceneObject) -> np.ndarray:
        """Normalize object scale"""
        bounds_size = obj.mesh.bounds[1] - obj.mesh.bounds[0]
        max_dim = np.max(bounds_size)
        
        if max_dim > 0:
            target_size = 2.0  # Normalize to 2 units max dimension
            scale_factor = target_size / max_dim
            
            # Apply scale variation from rules
            if "scale_variation" in self.rules:
                variation = np.random.uniform(
                    1 - self.rules["scale_variation"],
                    1 + self.rules["scale_variation"]
                )
                scale_factor *= variation
            
            return np.ones(3) * scale_factor
        
        return np.ones(3)
    
    def _infer_semantic_label(self, obj: SceneObject) -> str:
        """Infer semantic label from mesh properties"""
        # Simple inference based on mesh characteristics
        bounds_size = obj.mesh.bounds[1] - obj.mesh.bounds[0]
        volume = np.prod(bounds_size)
        aspect_ratio = bounds_size[1] / max(bounds_size[0], bounds_size[2], 1e-6)
        
        if aspect_ratio > 3.0:
            return "tree"
        elif volume < 0.1:
            return "small_object"
        elif volume < 1.0:
            return "medium_object"
        else:
            return "large_object"
    
    def _get_semantic_priority(self, label: str) -> int:
        """Get placement priority for semantic label"""
        priority_map = {
            "building": 0,
            "tree": 1,
            "road": 2,
            "vehicle": 3,
            "person": 4,
            "small_object": 5
        }
        return priority_map.get(label, 6)
    
    def _calculate_physics_properties(self, obj: SceneObject) -> Dict[str, Any]:
        """Calculate physics properties for object"""
        volume = obj.mesh.volume
        bounds_size = obj.mesh.bounds[1] - obj.mesh.bounds[0]
        
        return {
            "mass": volume * 1000,  # Assume density of water
            "inertia": self._calculate_inertia_tensor(obj.mesh),
            "friction": 0.5,
            "restitution": 0.2,
            "collision_shape": "convex_hull"
        }
    
    def _calculate_inertia_tensor(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Calculate inertia tensor for mesh"""
        # Simplified calculation
        volume = mesh.volume
        if volume == 0:
            return np.eye(3)
        
        bounds_size = mesh.bounds[1] - mesh.bounds[0]
        mass = volume * 1000
        
        # Approximate as box
        Ixx = mass * (bounds_size[1]**2 + bounds_size[2]**2) / 12
        Iyy = mass * (bounds_size[0]**2 + bounds_size[2]**2) / 12
        Izz = mass * (bounds_size[0]**2 + bounds_size[1]**2) / 12
        
        return np.diag([Ixx, Iyy, Izz])
    
    def _build_spatial_index(self, objects: List[SceneObject]) -> None:
        """Build spatial index for fast queries"""
        if not objects:
            self.spatial_tree = None
            return
        
        positions = np.array([obj.position for obj in objects])
        self.spatial_tree = KDTree(positions)
    
    def _update_semantic_graph(
        self,
        new_object: SceneObject,
        existing_objects: List[SceneObject]
    ) -> None:
        """Update semantic relationship graph"""
        for obj in existing_objects:
            if obj.id == new_object.id:
                continue
            
            distance = np.linalg.norm(new_object.position - obj.position)
            if distance < 5.0:  # Neighbors within 5 units
                self.semantic_graph[new_object.id].append(obj.id)
                self.semantic_graph[obj.id].append(new_object.id)
    
    def _postprocess_composition(self, objects: List[SceneObject]) -> List[SceneObject]:
        """Apply post-processing to composed scene"""
        # Apply ground alignment
        for obj in objects:
            if obj.semantic_label not in ["tree", "plant"]:
                # Ensure object is on ground
                obj.position[1] = 0
        
        # Apply symmetry if specified
        if "symmetry" in self.constraints.symmetry_constraints:
            self._apply_symmetry(objects)
        
        return objects
    
    def _apply_symmetry(self, objects: List[SceneObject]) -> None:
        """Apply symmetry to scene"""
        symmetry_type = self.constraints.symmetry_constraints.get("symmetry", "none")
        
        if symmetry_type == "reflection":
            axis = self.constraints.symmetry_constraints.get("axis", "x")
            reflection_point = self.constraints.symmetry_constraints.get("point", 0.0)
            
            mirrored_objects = []
            for obj in objects:
                if axis == "x":
                    mirrored_pos = np.array([
                        2 * reflection_point - obj.position[0],
                        obj.position[1],
                        obj.position[2]
                    ])
                elif axis == "z":
                    mirrored_pos = np.array([
                        obj.position[0],
                        obj.position[1],
                        2 * reflection_point - obj.position[2]
                    ])
                else:
                    continue
                
                mirrored_obj = SceneObject(
                    id=obj.id + "_mirrored",
                    mesh=obj.mesh,
                    position=mirrored_pos,
                    rotation=obj.rotation,
                    scale=obj.scale,
                    semantic_label=obj.semantic_label,
                    material=obj.material,
                    physics_properties=obj.physics_properties,
                    metadata=obj.metadata
                )
                mirrored_objects.append(mirrored_obj)
            
            objects.extend(mirrored_objects)
    
    def query_objects(
        self,
        position: np.ndarray,
        radius: float,
        semantic_filter: Optional[List[str]] = None
    ) -> List[SceneObject]:
        """Query objects within radius of position"""
        if self.spatial_tree is None:
            return []
        
        indices = self.spatial_tree.query_ball_point(position, radius)
        
        if not indices:
            return []
        
        all_objects = list(self.scene_objects.values())
        results = [all_objects[i] for i in indices]
        
        if semantic_filter:
            results = [obj for obj in results if obj.semantic_label in semantic_filter]
        
        return results
    
    def export_scene(
        self,
        output_path: Union[str, Path],
        format: str = "gltf"
    ) -> None:
        """Export composed scene to file"""
        output_path = Path(output_path)
        
        if format == "gltf":
            self._export_gltf(output_path)
        elif format == "obj":
            self._export_obj(output_path)
        elif format == "json":
            self._export_json(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_gltf(self, output_path: Path) -> None:
        """Export scene as GLTF"""
        import trimesh.exchange.gltf
        
        # Create a scene with all objects
        scene = trimesh.Scene()
        
        for obj in self.scene_objects.values():
            # Apply transformation to mesh
            transformed_mesh = obj.mesh.copy()
            transformed_mesh.apply_transform(obj.transform)
            scene.add_geometry(transformed_mesh, node_name=obj.id)
        
        # Export
        scene.export(output_path.with_suffix('.glb'))
        logger.info(f"Exported scene to {output_path.with_suffix('.glb')}")
    
    def _export_obj(self, output_path: Path) -> None:
        """Export scene as OBJ with MTL"""
        import trimesh.exchange.obj
        
        scene = trimesh.Scene()
        
        for obj in self.scene_objects.values():
            transformed_mesh = obj.mesh.copy()
            transformed_mesh.apply_transform(obj.transform)
            scene.add_geometry(transformed_mesh, node_name=obj.id)
        
        scene.export(output_path.with_suffix('.obj'))
        logger.info(f"Exported scene to {output_path.with_suffix('.obj')}")
    
    def _export_json(self, output_path: Path) -> None:
        """Export scene as JSON metadata"""
        scene_data = {
            "metadata": {
                "version": "1.0",
                "generator": "FrexTech SceneComposer",
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "layout": self.layout.value,
                "mode": self.mode.value
            },
            "objects": [],
            "statistics": self.stats,
            "constraints": {
                "spatial": self.constraints.spatial_constraints,
                "semantic": self.constraints.semantic_constraints,
                "density": self.constraints.density_constraints
            }
        }
        
        for obj in self.scene_objects.values():
            obj_data = {
                "id": obj.id,
                "position": obj.position.tolist(),
                "rotation": obj.rotation.tolist(),
                "scale": obj.scale.tolist(),
                "semantic_label": obj.semantic_label,
                "bounds": obj.bounds.tolist(),
                "physics": obj.physics_properties,
                "metadata": obj.metadata
            }
            scene_data["objects"].append(obj_data)
        
        save_json(scene_data, output_path.with_suffix('.json'))
        logger.info(f"Exported scene metadata to {output_path.with_suffix('.json')}")
    
    def load_scene(self, filepath: Union[str, Path]) -> None:
        """Load scene from file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            self._load_json_scene(filepath)
        else:
            raise ValueError(f"Unsupported scene format: {filepath.suffix}")
    
    def _load_json_scene(self, filepath: Path) -> None:
        """Load scene from JSON"""
        scene_data = load_json(filepath)
        
        # Clear existing objects
        self.scene_objects.clear()
        
        # Load objects
        for obj_data in scene_data.get("objects", []):
            # Note: Meshes need to be loaded separately
            obj = SceneObject(
                id=obj_data["id"],
                mesh=trimesh.creation.box([1, 1, 1]),  # Placeholder
                position=np.array(obj_data["position"]),
                rotation=np.array(obj_data["rotation"]),
                scale=np.array(obj_data["scale"]),
                semantic_label=obj_data["semantic_label"],
                physics_properties=obj_data.get("physics", {}),
                metadata=obj_data.get("metadata", {})
            )
            self.scene_objects[obj.id] = obj
        
        # Load constraints
        constraints_data = scene_data.get("constraints", {})
        self.constraints = CompositionConstraints(
            spatial_constraints=constraints_data.get("spatial", {}),
            semantic_constraints=constraints_data.get("semantic", {}),
            density_constraints=constraints_data.get("density", {})
        )
        
        # Rebuild spatial index
        self._build_spatial_index(list(self.scene_objects.values()))
        
        logger.info(f"Loaded scene from {filepath} with {len(self.scene_objects)} objects")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get composition statistics"""
        return self.stats.copy()
    
    def clear(self) -> None:
        """Clear all scene objects and reset state"""
        self.scene_objects.clear()
        self.spatial_tree = None
        self.semantic_graph.clear()
        self.stats = {
            "total_objects": 0,
            "composition_time": 0.0,
            "collision_checks": 0,
            "iterations": 0
        }
        logger.info("Scene composer cleared")
    
    def __str__(self) -> str:
        """String representation"""
        return (f"SceneComposer(mode={self.mode.value}, "
                f"layout={self.layout.value}, "
                f"objects={len(self.scene_objects)})")
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"SceneComposer(mode={self.mode}, layout={self.layout}, "
                f"constraints={self.constraints}, "
                f"stats={self.stats})")


# Factory function for creating composers
def create_scene_composer(
    composer_type: str = "automatic",
    layout: str = "open_world",
    **kwargs
) -> SceneComposer:
    """
    Factory function to create scene composers
    
    Args:
        composer_type: Type of composer ('automatic', 'semi_automatic', 'manual')
        layout: Scene layout type
        **kwargs: Additional arguments for composer
    
    Returns:
        SceneComposer instance
    """
    mode_map = {
        "automatic": CompositionMode.AUTOMATIC,
        "semi_automatic": CompositionMode.SEMI_AUTOMATIC,
        "manual": CompositionMode.MANUAL,
        "hybrid": CompositionMode.HYBRID
    }
    
    layout_map = {
        "open_world": SceneLayout.OPEN_WORLD,
        "interior": SceneLayout.INTERIOR,
        "cityscape": SceneLayout.CITYSCAPE,
        "natural": SceneLayout.NATURAL,
        "abstract": SceneLayout.ABSTRACT,
        "mixed": SceneLayout.MIXED
    }
    
    mode = mode_map.get(composer_type, CompositionMode.AUTOMATIC)
    scene_layout = layout_map.get(layout, SceneLayout.OPEN_WORLD)
    
    return SceneComposer(
        config=kwargs.get("config", {}),
        mode=mode,
        layout=scene_layout
    )