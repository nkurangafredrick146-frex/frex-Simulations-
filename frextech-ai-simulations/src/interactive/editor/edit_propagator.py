"""
Edit Propagator Module
Propagates edits across scenes while maintaining consistency
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
from collections import defaultdict, deque
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local imports
from ...utils.metrics import Timer, PerformanceMetrics
from ...utils.file_io import save_json, load_json, save_pickle, load_pickle
from .consistency_checker import ConsistencyChecker, ConsistencyViolation

logger = logging.getLogger(__name__)


class PropagationMode(Enum):
    """Modes for edit propagation"""
    LOCAL = "local"           # Propagate to nearby regions
    GLOBAL = "global"         # Propagate to entire scene
    SEMANTIC = "semantic"     # Propagate based on semantic similarity
    STRUCTURAL = "structural" # Propagate based on structural similarity
    MANUAL = "manual"         # Manual propagation control


class PropagationMethod(Enum):
    """Methods for edit propagation"""
    DIFFUSION = "diffusion"       # Gradually spread edits
    REPLICATION = "replication"   # Copy edits to similar areas
    TRANSFORMATION = "transformation" # Transform edits to fit new context
    NEURAL = "neural"             # Use neural networks for propagation


@dataclass
class EditOperation:
    """Record of an edit operation"""
    edit_id: str
    operation_type: str
    target_object: str
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Generate hash for edit"""
        content = f"{self.edit_id}_{self.operation_type}_{self.target_object}_{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class PropagationRule:
    """Rule for edit propagation"""
    rule_id: str
    source_pattern: Dict[str, Any]
    target_pattern: Dict[str, Any]
    transformation: Callable[[Dict[str, Any]], Dict[str, Any]]
    priority: int = 0
    conditions: List[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    
    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if rule matches current context"""
        for key, value in self.source_pattern.items():
            if key not in context or context[key] != value:
                return False
        
        # Check additional conditions
        for condition in self.conditions:
            if not condition(context):
                return False
        
        return True


class EditPropagator:
    """
    Main edit propagator for spreading edits across scenes
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        consistency_checker: Optional[ConsistencyChecker] = None,
        max_workers: int = 4
    ):
        """
        Initialize edit propagator
        
        Args:
            config: Configuration dictionary
            consistency_checker: Optional consistency checker
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.consistency_checker = consistency_checker
        self.max_workers = max_workers
        
        # Edit tracking
        self.edit_history: Dict[str, EditOperation] = {}
        self.edit_groups: Dict[str, List[str]] = defaultdict(list)
        
        # Propagation rules
        self.propagation_rules: Dict[str, PropagationRule] = {}
        self.propagation_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Scene state
        self.scene_state: Optional[Dict[str, Any]] = None
        self.scene_graph: Optional[Dict[str, Any]] = None
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Load default propagation rules
        self._load_default_rules()
        
        logger.info(f"EditPropagator initialized with {len(self.propagation_rules)} rules")
    
    def _load_default_rules(self) -> None:
        """Load default propagation rules"""
        # Color propagation rule
        self.add_propagation_rule(PropagationRule(
            rule_id="color_propagation",
            source_pattern={"operation_type": "color_change"},
            target_pattern={"operation_type": "color_change"},
            transformation=self._transform_color_edit,
            priority=5
        ))
        
        # Scale propagation rule
        self.add_propagation_rule(PropagationRule(
            rule_id="scale_propagation",
            source_pattern={"operation_type": "scale_change"},
            target_pattern={"operation_type": "scale_change"},
            transformation=self._transform_scale_edit,
            priority=4
        ))
        
        # Position propagation rule
        self.add_propagation_rule(PropagationRule(
            rule_id="position_propagation",
            source_pattern={"operation_type": "position_change"},
            target_pattern={"operation_type": "position_change"},
            transformation=self._transform_position_edit,
            priority=6
        ))
        
        # Material propagation rule
        self.add_propagation_rule(PropagationRule(
            rule_id="material_propagation",
            source_pattern={"operation_type": "material_change"},
            target_pattern={"operation_type": "material_change"},
            transformation=self._transform_material_edit,
            priority=3
        ))
        
        # Structural propagation rule
        self.add_propagation_rule(PropagationRule(
            rule_id="structural_propagation",
            source_pattern={"operation_type": "structural_change"},
            target_pattern={"operation_type": "structural_change"},
            transformation=self._transform_structural_edit,
            priority=7,
            conditions=[self._check_structural_similarity]
        ))
    
    def add_propagation_rule(self, rule: PropagationRule) -> None:
        """Add a propagation rule"""
        self.propagation_rules[rule.rule_id] = rule
        logger.debug(f"Added propagation rule: {rule.rule_id}")
    
    def remove_propagation_rule(self, rule_id: str) -> None:
        """Remove a propagation rule"""
        if rule_id in self.propagation_rules:
            del self.propagation_rules[rule_id]
            logger.debug(f"Removed propagation rule: {rule_id}")
    
    def record_edit(
        self,
        operation_type: str,
        target_object: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record an edit operation
        
        Args:
            operation_type: Type of edit operation
            target_object: ID of target object
            parameters: Edit parameters
            metadata: Additional metadata
            
        Returns:
            Edit ID
        """
        edit_id = self._generate_edit_id(operation_type, target_object)
        
        edit = EditOperation(
            edit_id=edit_id,
            operation_type=operation_type,
            target_object=target_object,
            parameters=parameters,
            metadata=metadata or {}
        )
        
        self.edit_history[edit_id] = edit
        
        # Group edits by type
        self.edit_groups[operation_type].append(edit_id)
        
        logger.debug(f"Recorded edit {edit_id}: {operation_type} on {target_object}")
        
        return edit_id
    
    def propagate_edit(
        self,
        source_edit_id: str,
        scene_data: Dict[str, Any],
        mode: PropagationMode = PropagationMode.LOCAL,
        method: PropagationMethod = PropagationMethod.REPLICATION,
        radius: float = 10.0,
        max_propagations: int = 20,
        consistency_check: bool = True
    ) -> Dict[str, Any]:
        """
        Propagate an edit to other parts of the scene
        
        Args:
            source_edit_id: ID of source edit to propagate
            scene_data: Scene data
            mode: Propagation mode
            method: Propagation method
            radius: Propagation radius for local mode
            max_propagations: Maximum number of propagations
            consistency_check: Whether to check consistency
            
        Returns:
            Updated scene data with propagated edits
        """
        timer = Timer()
        
        if source_edit_id not in self.edit_history:
            raise ValueError(f"Unknown edit ID: {source_edit_id}")
        
        source_edit = self.edit_history[source_edit_id]
        logger.info(f"Propagating edit {source_edit_id} ({source_edit.operation_type})")
        
        # Update scene state
        self.scene_state = scene_data.copy()
        self._build_scene_graph(scene_data)
        
        # Find propagation targets
        targets = self._find_propagation_targets(
            source_edit, scene_data, mode, radius, max_propagations
        )
        
        logger.info(f"Found {len(targets)} propagation targets")
        
        # Apply propagations
        propagated_scene = scene_data.copy()
        applied_edits = []
        
        for target in targets:
            # Generate edit for target
            target_edit = self._generate_propagated_edit(
                source_edit, target, method
            )
            
            if target_edit:
                # Apply edit to scene
                propagated_scene = self._apply_edit(
                    propagated_scene, target_edit, method
                )
                
                # Record propagated edit
                propagated_id = self.record_edit(
                    operation_type=target_edit["operation_type"],
                    target_object=target_edit["target_object"],
                    parameters=target_edit["parameters"],
                    metadata={
                        "source_edit": source_edit_id,
                        "propagation_method": method.value,
                        "propagation_mode": mode.value
                    }
                )
                
                applied_edits.append(propagated_id)
        
        # Check consistency if requested
        if consistency_check and self.consistency_checker:
            violations = self._check_propagated_consistency(propagated_scene)
            
            if violations:
                logger.warning(f"Found {len(violations)} consistency violations after propagation")
                
                # Attempt to auto-fix violations
                if self.config.get("auto_fix_violations", True):
                    propagated_scene = self.consistency_checker.auto_fix_violations(
                        propagated_scene, violations
                    )
        
        # Update metrics
        self.metrics.record_operation("propagate_edit", timer.elapsed())
        
        logger.info(f"Edit propagation completed in {timer.elapsed():.2f}s: "
                   f"{len(applied_edits)} edits applied")
        
        return {
            "scene_data": propagated_scene,
            "source_edit": source_edit_id,
            "propagated_edits": applied_edits,
            "propagation_mode": mode.value,
            "propagation_method": method.value,
            "metrics": {
                "time": timer.elapsed(),
                "edits_applied": len(applied_edits),
                "targets_found": len(targets)
            }
        }
    
    def _generate_edit_id(
        self,
        operation_type: str,
        target_object: str
    ) -> str:
        """Generate unique edit ID"""
        timestamp = int(time.time() * 1000)
        content = f"{operation_type}_{target_object}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _build_scene_graph(self, scene_data: Dict[str, Any]) -> None:
        """Build scene graph for efficient queries"""
        objects = scene_data.get("objects", {})
        
        self.scene_graph = {
            "objects": objects,
            "spatial_index": {},
            "semantic_index": defaultdict(list),
            "structural_index": defaultdict(list)
        }
        
        # Build spatial index (simple grid)
        grid_size = 5.0
        for obj_id, obj_data in objects.items():
            if "position" in obj_data:
                pos = obj_data["position"]
                grid_key = (
                    int(pos[0] / grid_size),
                    int(pos[1] / grid_size),
                    int(pos[2] / grid_size)
                )
                
                if grid_key not in self.scene_graph["spatial_index"]:
                    self.scene_graph["spatial_index"][grid_key] = []
                self.scene_graph["spatial_index"][grid_key].append(obj_id)
            
            # Build semantic index
            obj_type = obj_data.get("type", "unknown")
            self.scene_graph["semantic_index"][obj_type].append(obj_id)
            
            # Build structural index
            if "structure" in obj_data:
                structure = obj_data["structure"]
                self.scene_graph["structural_index"][structure].append(obj_id)
    
    def _find_propagation_targets(
        self,
        source_edit: EditOperation,
        scene_data: Dict[str, Any],
        mode: PropagationMode,
        radius: float,
        max_targets: int
    ) -> List[Dict[str, Any]]:
        """Find suitable targets for propagation"""
        source_obj_id = source_edit.target_object
        objects = scene_data.get("objects", {})
        
        if source_obj_id not in objects:
            return []
        
        source_obj = objects[source_obj_id]
        targets = []
        
        if mode == PropagationMode.LOCAL:
            # Find objects within radius
            if "position" in source_obj:
                source_pos = np.array(source_obj["position"])
                
                for obj_id, obj in objects.items():
                    if obj_id == source_obj_id:
                        continue
                    
                    if "position" in obj:
                        target_pos = np.array(obj["position"])
                        distance = np.linalg.norm(source_pos - target_pos)
                        
                        if distance <= radius:
                            targets.append({
                                "object_id": obj_id,
                                "object_data": obj,
                                "distance": distance,
                                "similarity": self._calculate_similarity(source_obj, obj)
                            })
            
            # Sort by distance
            targets.sort(key=lambda x: x["distance"])
        
        elif mode == PropagationMode.SEMANTIC:
            # Find objects with similar semantic type
            source_type = source_obj.get("type", "unknown")
            
            for obj_id, obj in objects.items():
                if obj_id == source_obj_id:
                    continue
                
                obj_type = obj.get("type", "unknown")
                
                if obj_type == source_type:
                    similarity = self._calculate_semantic_similarity(source_obj, obj)
                    
                    targets.append({
                        "object_id": obj_id,
                        "object_data": obj,
                        "similarity": similarity,
                        "semantic_match": True
                    })
            
            # Sort by similarity
            targets.sort(key=lambda x: -x["similarity"])
        
        elif mode == PropagationMode.STRUCTURAL:
            # Find objects with similar structure
            source_structure = source_obj.get("structure", {})
            
            for obj_id, obj in objects.items():
                if obj_id == source_obj_id:
                    continue
                
                obj_structure = obj.get("structure", {})
                
                if self._check_structural_similarity({
                    "source": source_structure,
                    "target": obj_structure
                }):
                    similarity = self._calculate_structural_similarity(
                        source_structure, obj_structure
                    )
                    
                    targets.append({
                        "object_id": obj_id,
                        "object_data": obj,
                        "similarity": similarity,
                        "structural_match": True
                    })
            
            # Sort by similarity
            targets.sort(key=lambda x: -x["similarity"])
        
        elif mode == PropagationMode.GLOBAL:
            # Propagate to all objects (with some filtering)
            for obj_id, obj in objects.items():
                if obj_id == source_obj_id:
                    continue
                
                # Filter out inappropriate objects
                if self._is_valid_target(source_obj, obj):
                    similarity = self._calculate_similarity(source_obj, obj)
                    
                    targets.append({
                        "object_id": obj_id,
                        "object_data": obj,
                        "similarity": similarity
                    })
            
            # Sort by similarity
            targets.sort(key=lambda x: -x["similarity"])
        
        # Limit number of targets
        return targets[:max_targets]
    
    def _calculate_similarity(
        self,
        source_obj: Dict[str, Any],
        target_obj: Dict[str, Any]
    ) -> float:
        """Calculate overall similarity between objects"""
        similarity = 0.0
        factors = 0
        
        # Type similarity
        if source_obj.get("type") == target_obj.get("type"):
            similarity += 0.3
        factors += 1
        
        # Scale similarity
        if "scale" in source_obj and "scale" in target_obj:
            source_scale = np.array(source_obj["scale"])
            target_scale = np.array(target_obj["scale"])
            
            if np.all(source_scale > 0) and np.all(target_scale > 0):
                scale_ratio = source_scale / target_scale
                scale_similarity = 1.0 - np.mean(np.abs(np.log(scale_ratio))) / 5.0
                similarity += max(0, scale_similarity) * 0.3
                factors += 1
        
        # Color similarity
        if "color" in source_obj and "color" in target_obj:
            source_color = np.array(source_obj["color"])
            target_color = np.array(target_obj["color"])
            
            color_distance = np.linalg.norm(source_color - target_color) / 441.67  # max distance
            color_similarity = 1.0 - color_distance
            similarity += max(0, color_similarity) * 0.2
            factors += 1
        
        # Position similarity (relative to scene center)
        if "position" in source_obj and "position" in target_obj:
            source_pos = np.array(source_obj["position"])
            target_pos = np.array(target_obj["position"])
            
            # Consider objects in similar areas of scene
            source_region = np.sign(source_pos)
            target_region = np.sign(target_pos)
            
            region_match = np.mean(source_region == target_region)
            similarity += region_match * 0.2
            factors += 1
        
        return similarity / factors if factors > 0 else 0.0
    
    def _calculate_semantic_similarity(
        self,
        source_obj: Dict[str, Any],
        target_obj: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity between objects"""
        # Simple implementation - could be enhanced with semantic embeddings
        source_type = source_obj.get("type", "").lower()
        target_type = target_obj.get("type", "").lower()
        
        if source_type == target_type:
            return 1.0
        
        # Check for semantic categories
        categories = {
            "furniture": ["chair", "table", "bed", "sofa", "desk"],
            "vegetation": ["tree", "bush", "plant", "flower", "grass"],
            "vehicle": ["car", "truck", "bus", "bicycle", "motorcycle"],
            "building": ["house", "building", "tower", "wall", "roof"]
        }
        
        for category, types in categories.items():
            if source_type in types and target_type in types:
                return 0.8
        
        return 0.0
    
    def _calculate_structural_similarity(
        self,
        source_structure: Dict[str, Any],
        target_structure: Dict[str, Any]
    ) -> float:
        """Calculate structural similarity"""
        if not source_structure or not target_structure:
            return 0.0
        
        # Compare key structural properties
        similarity = 0.0
        factors = 0
        
        # Compare primitive counts
        for key in ["vertices", "faces", "edges"]:
            if key in source_structure and key in target_structure:
                source_val = source_structure[key]
                target_val = target_structure[key]
                
                if source_val > 0 and target_val > 0:
                    ratio = min(source_val, target_val) / max(source_val, target_val)
                    similarity += ratio
                    factors += 1
        
        # Compare bounding box proportions
        if "bounds" in source_structure and "bounds" in target_structure:
            source_bounds = source_structure["bounds"]
            target_bounds = target_structure["bounds"]
            
            if len(source_bounds) == 2 and len(target_bounds) == 2:
                source_size = np.array(source_bounds[1]) - np.array(source_bounds[0])
                target_size = np.array(target_bounds[1]) - np.array(target_bounds[0])
                
                if np.all(source_size > 0) and np.all(target_size > 0):
                    source_proportions = source_size / np.max(source_size)
                    target_proportions = target_size / np.max(target_size)
                    
                    proportion_similarity = 1.0 - np.mean(np.abs(source_proportions - target_proportions))
                    similarity += max(0, proportion_similarity)
                    factors += 1
        
        return similarity / factors if factors > 0 else 0.0
    
    def _check_structural_similarity(self, context: Dict[str, Any]) -> bool:
        """Check if objects are structurally similar"""
        source_structure = context.get("source", {})
        target_structure = context.get("target", {})
        
        if not source_structure or not target_structure:
            return False
        
        # Check if both have similar primitive types
        source_type = source_structure.get("primitive_type", "")
        target_type = target_structure.get("primitive_type", "")
        
        if source_type and target_type and source_type != target_type:
            return False
        
        # Check if face counts are within reasonable range
        source_faces = source_structure.get("faces", 0)
        target_faces = target_structure.get("faces", 0)
        
        if source_faces > 0 and target_faces > 0:
            ratio = max(source_faces, target_faces) / min(source_faces, target_faces)
            if ratio > 10:  # Too different in complexity
                return False
        
        return True
    
    def _is_valid_target(
        self,
        source_obj: Dict[str, Any],
        target_obj: Dict[str, Any]
    ) -> bool:
        """Check if target object is valid for propagation"""
        # Skip if target is very different
        similarity = self._calculate_similarity(source_obj, target_obj)
        if similarity < 0.3:
            return False
        
        # Skip if target has been recently edited
        recent_edits = list(self.edit_history.values())[-10:]
        for edit in recent_edits:
            if edit.target_object == target_obj.get("id"):
                return False
        
        return True
    
    def _generate_propagated_edit(
        self,
        source_edit: EditOperation,
        target: Dict[str, Any],
        method: PropagationMethod
    ) -> Optional[Dict[str, Any]]:
        """Generate propagated edit for target"""
        # Find matching propagation rule
        context = {
            "source_edit": source_edit,
            "target_object": target["object_data"],
            "method": method
        }
        
        for rule in self.propagation_rules.values():
            if rule.matches(context):
                try:
                    # Apply transformation
                    transformed_params = rule.transformation({
                        "source_params": source_edit.parameters,
                        "target_object": target["object_data"],
                        "similarity": target.get("similarity", 0.5)
                    })
                    
                    return {
                        "operation_type": source_edit.operation_type,
                        "target_object": target["object_id"],
                        "parameters": transformed_params,
                        "propagation_rule": rule.rule_id
                    }
                
                except Exception as e:
                    logger.error(f"Error applying rule {rule.rule_id}: {e}")
                    continue
        
        # No matching rule found
        logger.warning(f"No propagation rule found for edit type: {source_edit.operation_type}")
        return None
    
    def _transform_color_edit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform color edit for propagation"""
        source_params = context["source_params"]
        target_object = context["target_object"]
        similarity = context["similarity"]
        
        transformed = source_params.copy()
        
        # Adjust color based on target object's current color
        if "color" in target_object and "color" in source_params:
            current_color = np.array(target_object["color"])
            source_color = np.array(source_params["color"])
            
            # Blend based on similarity
            blend_factor = 0.5 + similarity * 0.5
            new_color = current_color * (1 - blend_factor) + source_color * blend_factor
            
            transformed["color"] = new_color.tolist()
        
        return transformed
    
    def _transform_scale_edit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform scale edit for propagation"""
        source_params = context["source_params"]
        target_object = context["target_object"]
        similarity = context["similarity"]
        
        transformed = source_params.copy()
        
        if "scale" in source_params and "scale" in target_object:
            source_scale = np.array(source_params["scale"])
            current_scale = np.array(target_object.get("scale", [1, 1, 1]))
            
            # Calculate relative scale change
            if np.all(current_scale > 0):
                scale_factor = source_scale / current_scale
                
                # Adjust factor based on similarity
                adjusted_factor = 1.0 + (scale_factor - 1.0) * similarity
                
                new_scale = current_scale * adjusted_factor
                transformed["scale"] = new_scale.tolist()
        
        return transformed
    
    def _transform_position_edit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform position edit for propagation"""
        source_params = context["source_params"]
        target_object = context["target_object"]
        
        transformed = source_params.copy()
        
        if "position" in source_params and "position" in target_object:
            source_pos = np.array(source_params["position"])
            current_pos = np.array(target_object["position"])
            
            # Calculate offset from source edit
            # For now, just use the same offset
            offset = source_pos - np.array([0, 0, 0])  # Would need reference
            
            new_position = current_pos + offset
            transformed["position"] = new_position.tolist()
        
        return transformed
    
    def _transform_material_edit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform material edit for propagation"""
        # Simple replication for material edits
        return context["source_params"].copy()
    
    def _transform_structural_edit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Transform structural edit for propagation"""
        source_params = context["source_params"]
        similarity = context["similarity"]
        
        transformed = source_params.copy()
        
        # Adjust structural parameters based on similarity
        if "structure_params" in source_params:
            params = source_params["structure_params"]
            
            # Scale parameters based on similarity
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    params[key] = value * similarity
            
            transformed["structure_params"] = params
        
        return transformed
    
    def _apply_edit(
        self,
        scene_data: Dict[str, Any],
        edit: Dict[str, Any],
        method: PropagationMethod
    ) -> Dict[str, Any]:
        """Apply edit to scene data"""
        updated_scene = scene_data.copy()
        objects = updated_scene.get("objects", {})
        
        target_id = edit["target_object"]
        
        if target_id not in objects:
            logger.warning(f"Target object {target_id} not found in scene")
            return updated_scene
        
        # Apply edit based on operation type
        operation_type = edit["operation_type"]
        parameters = edit["parameters"]
        
        if operation_type == "color_change":
            if "color" in parameters:
                objects[target_id]["color"] = parameters["color"]
        
        elif operation_type == "scale_change":
            if "scale" in parameters:
                objects[target_id]["scale"] = parameters["scale"]
        
        elif operation_type == "position_change":
            if "position" in parameters:
                objects[target_id]["position"] = parameters["position"]
        
        elif operation_type == "material_change":
            if "material" in parameters:
                objects[target_id]["material"] = parameters["material"]
        
        elif operation_type == "structural_change":
            if "structure" in parameters:
                objects[target_id]["structure"] = parameters["structure"]
        
        # Mark as edited
        if "metadata" not in objects[target_id]:
            objects[target_id]["metadata"] = {}
        
        objects[target_id]["metadata"]["last_edit"] = {
            "type": operation_type,
            "method": method.value,
            "timestamp": time.time()
        }
        
        return updated_scene
    
    def _check_propagated_consistency(
        self,
        scene_data: Dict[str, Any]
    ) -> List[ConsistencyViolation]:
        """Check consistency after propagation"""
        if not self.consistency_checker:
            return []
        
        # Run consistency check
        report = self.consistency_checker.check_scene(scene_data)
        
        # Filter for violations likely caused by propagation
        propagation_violations = []
        
        for violation in report.get("violations", []):
            # Check if violation involves recently edited objects
            if "objects" in violation:
                for obj_id in violation["objects"]:
                    # Check if this object was recently edited
                    recent_edits = list(self.edit_history.values())[-20:]
                    for edit in recent_edits:
                        if edit.target_object == obj_id:
                            propagation_violations.append(violation)
                            break
        
        return propagation_violations
    
    def propagate_edit_group(
        self,
        group_id: str,
        scene_data: Dict[str, Any],
        mode: PropagationMode = PropagationMode.SEMANTIC,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Propagate a group of related edits
        
        Args:
            group_id: Edit group ID
            scene_data: Scene data
            mode: Propagation mode
            **kwargs: Additional propagation parameters
            
        Returns:
            Updated scene data
        """
        if group_id not in self.edit_groups:
            raise ValueError(f"Unknown edit group: {group_id}")
        
        edit_ids = self.edit_groups[group_id]
        
        if not edit_ids:
            return scene_data
        
        logger.info(f"Propagating edit group {group_id} with {len(edit_ids)} edits")
        
        # Propagate each edit in the group
        current_scene = scene_data.copy()
        
        for edit_id in edit_ids:
            if edit_id in self.edit_history:
                result = self.propagate_edit(
                    edit_id, current_scene, mode, **kwargs
                )
                current_scene = result["scene_data"]
        
        return current_scene
    
    def create_edit_group(
        self,
        edit_ids: List[str],
        name: Optional[str] = None
    ) -> str:
        """
        Create a group of related edits
        
        Args:
            edit_ids: List of edit IDs to group
            name: Optional group name
            
        Returns:
            Group ID
        """
        group_id = f"group_{int(time.time())}_{hashlib.md5(str(edit_ids).encode()).hexdigest()[:6]}"
        
        # Validate edits exist
        valid_edits = []
        for edit_id in edit_ids:
            if edit_id in self.edit_history:
                valid_edits.append(edit_id)
            else:
                logger.warning(f"Edit {edit_id} not found, skipping")
        
        self.edit_groups[group_id] = valid_edits
        
        if name:
            # Store group metadata
            if "metadata" not in self.__dict__:
                self.metadata = {}
            
            self.metadata[group_id] = {"name": name}
        
        logger.info(f"Created edit group {group_id} with {len(valid_edits)} edits")
        
        return group_id
    
    def undo_propagation(
        self,
        scene_data: Dict[str, Any],
        propagation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Undo a propagation operation
        
        Args:
            scene_data: Current scene data
            propagation_result: Original propagation result
            
        Returns:
            Scene data with propagation undone
        """
        # This would restore the scene to its pre-propagation state
        # For now, return original scene data
        logger.info("Undoing propagation")
        return scene_data
    
    def get_edit_stats(self) -> Dict[str, Any]:
        """Get statistics about edits"""
        total_edits = len(self.edit_history)
        
        # Count by operation type
        type_counts = defaultdict(int)
        for edit in self.edit_history.values():
            type_counts[edit.operation_type] += 1
        
        # Count propagations
        propagation_count = sum(
            1 for edit in self.edit_history.values()
            if "propagation" in edit.metadata
        )
        
        return {
            "total_edits": total_edits,
            "edit_groups": len(self.edit_groups),
            "type_counts": dict(type_counts),
            "propagation_count": propagation_count,
            "recent_edits": len(list(self.edit_history.values())[-100:])
        }
    
    def export_edit_history(
        self,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """Export edit history to file"""
        output_path = Path(output_path)
        
        if format == "json":
            data = {
                "edits": [
                    {
                        "id": edit.edit_id,
                        "operation_type": edit.operation_type,
                        "target_object": edit.target_object,
                        "parameters": edit.parameters,
                        "timestamp": edit.timestamp,
                        "metadata": edit.metadata
                    }
                    for edit in self.edit_history.values()
                ],
                "groups": {
                    group_id: edits
                    for group_id, edits in self.edit_groups.items()
                },
                "summary": self.get_edit_stats(),
                "exported": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            save_json(data, output_path.with_suffix('.json'))
            logger.info(f"Exported edit history to {output_path.with_suffix('.json')}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_edit_history(self, keep_recent: int = 100) -> None:
        """Clear edit history, optionally keeping recent edits"""
        if keep_recent > 0 and len(self.edit_history) > keep_recent:
            # Keep only recent edits
            all_edits = list(self.edit_history.items())
            recent_edits = dict(all_edits[-keep_recent:])
            
            self.edit_history = recent_edits
            logger.info(f"Cleared edit history, kept {len(recent_edits)} recent edits")
        else:
            self.edit_history.clear()
            self.edit_groups.clear()
            logger.info("Cleared all edit history")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("EditPropagator cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        return (f"EditPropagator(edits={len(self.edit_history)}, "
                f"groups={len(self.edit_groups)}, "
                f"rules={len(self.propagation_rules)})")


# Factory function for creating edit propagators
def create_edit_propagator(
    config: Optional[Dict[str, Any]] = None,
    consistency_checker: Optional[ConsistencyChecker] = None,
    max_workers: int = 4
) -> EditPropagator:
    """
    Factory function to create edit propagators
    
    Args:
        config: Configuration dictionary
        consistency_checker: Optional consistency checker
        max_workers: Maximum worker threads
        
    Returns:
        EditPropagator instance
    """
    return EditPropagator(
        config=config,
        consistency_checker=consistency_checker,
        max_workers=max_workers
    )