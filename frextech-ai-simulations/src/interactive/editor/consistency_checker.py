"""
Consistency Checker Module
Ensures consistency across scene edits and modifications
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

logger = logging.getLogger(__name__)


class ConsistencyRuleType(Enum):
    """Types of consistency rules"""
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    PHYSICAL = "physical"
    VISUAL = "visual"
    LOGICAL = "logical"


class ConsistencyLevel(Enum):
    """Levels of consistency enforcement"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    WARNING = "warning"


@dataclass
class ConsistencyRule:
    """Definition of a consistency rule"""
    rule_id: str
    rule_type: ConsistencyRuleType
    description: str
    check_function: Callable[[Dict[str, Any]], Tuple[bool, str]]
    level: ConsistencyLevel = ConsistencyLevel.MODERATE
    priority: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def hash(self) -> str:
        """Generate hash for rule"""
        content = f"{self.rule_id}_{self.rule_type.value}_{self.level.value}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class ConsistencyViolation:
    """Record of a consistency violation"""
    violation_id: str
    rule_id: str
    severity: str
    description: str
    location: Optional[Tuple[float, float, float]] = None
    objects: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_critical(self) -> bool:
        """Check if violation is critical"""
        return self.severity in ["critical", "high"]


class SceneGraphNode:
    """Node in scene graph for consistency checking"""
    
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        self.id = node_id
        self.type = node_type
        self.properties = properties
        self.children: List[SceneGraphNode] = []
        self.parent: Optional[SceneGraphNode] = None
        self.metadata: Dict[str, Any] = {}
    
    def add_child(self, child: 'SceneGraphNode') -> None:
        """Add child node"""
        child.parent = self
        self.children.append(child)
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """Get property with default"""
        return self.properties.get(key, default)
    
    def update_property(self, key: str, value: Any) -> None:
        """Update property"""
        self.properties[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "type": self.type,
            "properties": self.properties,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneGraphNode':
        """Create from dictionary"""
        node = cls(data["id"], data["type"], data["properties"])
        node.metadata = data.get("metadata", {})
        
        for child_data in data.get("children", []):
            child_node = cls.from_dict(child_data)
            node.add_child(child_node)
        
        return node


class ConsistencyChecker:
    """
    Main consistency checker for ensuring scene consistency
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4
    ):
        """
        Initialize consistency checker
        
        Args:
            config: Configuration dictionary
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.max_workers = max_workers
        
        # Rules registry
        self.rules: Dict[str, ConsistencyRule] = {}
        self.rule_groups: Dict[ConsistencyRuleType, List[str]] = defaultdict(list)
        
        # Violation tracking
        self.violations: Dict[str, ConsistencyViolation] = {}
        self.violation_history: deque = deque(maxlen=1000)
        
        # Scene graph
        self.scene_graph: Optional[SceneGraphNode] = None
        self.scene_snapshot: Optional[Dict[str, Any]] = None
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Load default rules
        self._load_default_rules()
        
        logger.info(f"ConsistencyChecker initialized with {len(self.rules)} rules")
    
    def _load_default_rules(self) -> None:
        """Load default consistency rules"""
        # Spatial rules
        self.add_rule(ConsistencyRule(
            rule_id="no_intersections",
            rule_type=ConsistencyRuleType.SPATIAL,
            description="Objects should not intersect",
            check_function=self._check_object_intersections,
            level=ConsistencyLevel.STRICT,
            priority=10
        ))
        
        self.add_rule(ConsistencyRule(
            rule_id="ground_contact",
            rule_type=ConsistencyRuleType.SPATIAL,
            description="Objects should be in contact with ground or supported",
            check_function=self._check_ground_contact,
            level=ConsistencyLevel.MODERATE,
            priority=5
        ))
        
        # Semantic rules
        self.add_rule(ConsistencyRule(
            rule_id="semantic_placement",
            rule_type=ConsistencyRuleType.SEMANTIC,
            description="Objects should be placed in semantically appropriate locations",
            check_function=self._check_semantic_placement,
            level=ConsistencyLevel.MODERATE,
            priority=3
        ))
        
        # Physical rules
        self.add_rule(ConsistencyRule(
            rule_id="scale_realism",
            rule_type=ConsistencyRuleType.PHYSICAL,
            description="Objects should have realistic scales",
            check_function=self._check_scale_realism,
            level=ConsistencyLevel.LENIENT,
            priority=2
        ))
        
        self.add_rule(ConsistencyRule(
            rule_id="physics_stability",
            rule_type=ConsistencyRuleType.PHYSICAL,
            description="Objects should be physically stable",
            check_function=self._check_physics_stability,
            level=ConsistencyLevel.MODERATE,
            priority=7
        ))
        
        # Visual rules
        self.add_rule(ConsistencyRule(
            rule_id="texture_consistency",
            rule_type=ConsistencyRuleType.VISUAL,
            description="Textures should be consistent across objects",
            check_function=self._check_texture_consistency,
            level=ConsistencyLevel.LENIENT,
            priority=1
        ))
        
        self.add_rule(ConsistencyRule(
            rule_id="lighting_consistency",
            rule_type=ConsistencyRuleType.VISUAL,
            description="Lighting should be consistent",
            check_function=self._check_lighting_consistency,
            level=ConsistencyLevel.MODERATE,
            priority=4
        ))
        
        # Temporal rules
        self.add_rule(ConsistencyRule(
            rule_id="temporal_consistency",
            rule_type=ConsistencyRuleType.TEMPORAL,
            description="Scene changes should be temporally consistent",
            check_function=self._check_temporal_consistency,
            level=ConsistencyLevel.MODERATE,
            priority=6
        ))
        
        # Logical rules
        self.add_rule(ConsistencyRule(
            rule_id="logical_constraints",
            rule_type=ConsistencyRuleType.LOGICAL,
            description="Scene should respect logical constraints",
            check_function=self._check_logical_constraints,
            level=ConsistencyLevel.MODERATE,
            priority=8
        ))
    
    def add_rule(self, rule: ConsistencyRule) -> None:
        """Add a consistency rule"""
        self.rules[rule.rule_id] = rule
        self.rule_groups[rule.rule_type].append(rule.rule_id)
        logger.debug(f"Added rule: {rule.rule_id} ({rule.rule_type.value})")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove a consistency rule"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            self.rule_groups[rule.rule_type].remove(rule_id)
            del self.rules[rule_id]
            logger.debug(f"Removed rule: {rule_id}")
    
    def check_scene(
        self,
        scene_data: Dict[str, Any],
        rule_types: Optional[List[ConsistencyRuleType]] = None,
        rule_ids: Optional[List[str]] = None,
        level_filter: Optional[ConsistencyLevel] = None
    ) -> Dict[str, Any]:
        """
        Check scene for consistency violations
        
        Args:
            scene_data: Scene data dictionary
            rule_types: Specific rule types to check
            rule_ids: Specific rule IDs to check
            level_filter: Filter violations by level
            
        Returns:
            Dictionary with check results
        """
        timer = Timer()
        scene_id = scene_data.get("id", "unknown")
        
        logger.info(f"Checking scene {scene_id} for consistency")
        
        # Update scene graph
        self._update_scene_graph(scene_data)
        
        # Determine which rules to check
        rules_to_check = self._get_rules_to_check(rule_types, rule_ids, level_filter)
        
        # Run checks in parallel
        violations = []
        warnings = []
        
        futures = {}
        for rule_id in rules_to_check:
            rule = self.rules[rule_id]
            future = self.executor.submit(
                self._run_single_check,
                rule,
                scene_data
            )
            futures[future] = rule_id
        
        # Collect results
        for future in as_completed(futures):
            rule_id = futures[future]
            try:
                result = future.result()
                if result:
                    is_violation, message, severity = result
                    
                    if is_violation:
                        violation = ConsistencyViolation(
                            violation_id=self._generate_violation_id(rule_id, scene_id),
                            rule_id=rule_id,
                            severity=severity,
                            description=message,
                            metadata={
                                "scene_id": scene_id,
                                "timestamp": time.time()
                            }
                        )
                        violations.append(violation)
                        
                        # Store in violations dict
                        self.violations[violation.violation_id] = violation
                        self.violation_history.append(violation)
                    else:
                        warnings.append({
                            "rule_id": rule_id,
                            "message": message,
                            "severity": severity
                        })
            except Exception as e:
                logger.error(f"Error checking rule {rule_id}: {e}")
        
        # Generate report
        report = self._generate_report(violations, warnings, scene_id)
        
        # Update metrics
        self.metrics.record_operation("check_scene", timer.elapsed())
        
        logger.info(f"Consistency check completed in {timer.elapsed():.2f}s: "
                   f"{len(violations)} violations, {len(warnings)} warnings")
        
        return report
    
    def _get_rules_to_check(
        self,
        rule_types: Optional[List[ConsistencyRuleType]],
        rule_ids: Optional[List[str]],
        level_filter: Optional[ConsistencyLevel]
    ) -> List[str]:
        """Get list of rules to check based on filters"""
        if rule_ids:
            # Check specific rules
            return [rid for rid in rule_ids if rid in self.rules]
        
        if rule_types:
            # Check rules of specific types
            rules = []
            for rule_type in rule_types:
                rules.extend(self.rule_groups.get(rule_type, []))
        else:
            # Check all rules
            rules = list(self.rules.keys())
        
        # Filter by level if specified
        if level_filter:
            rules = [
                rid for rid in rules
                if self.rules[rid].level == level_filter
            ]
        
        # Sort by priority
        rules.sort(key=lambda rid: -self.rules[rid].priority)
        
        return rules
    
    def _run_single_check(
        self,
        rule: ConsistencyRule,
        scene_data: Dict[str, Any]
    ) -> Optional[Tuple[bool, str, str]]:
        """
        Run a single consistency check
        
        Returns:
            Tuple of (is_violation, message, severity) or None if check passed
        """
        try:
            is_violation, message = rule.check_function(scene_data)
            
            if is_violation:
                severity = self._determine_severity(rule.level, message)
                return True, message, severity
            else:
                if message:  # Warning message
                    return False, message, "warning"
        
        except Exception as e:
            logger.error(f"Error in rule {rule.rule_id}: {e}")
            return True, f"Rule check failed: {str(e)}", "high"
        
        return None
    
    def _determine_severity(
        self,
        rule_level: ConsistencyLevel,
        message: str
    ) -> str:
        """Determine violation severity"""
        if rule_level == ConsistencyLevel.STRICT:
            return "critical"
        elif rule_level == ConsistencyLevel.MODERATE:
            return "high"
        elif rule_level == ConsistencyLevel.LENIENT:
            return "medium"
        else:
            return "low"
    
    def _generate_report(
        self,
        violations: List[ConsistencyViolation],
        warnings: List[Dict[str, Any]],
        scene_id: str
    ) -> Dict[str, Any]:
        """Generate consistency check report"""
        # Count violations by severity
        severity_counts = defaultdict(int)
        rule_violations = defaultdict(list)
        
        for violation in violations:
            severity_counts[violation.severity] += 1
            rule_violations[violation.rule_id].append(violation.violation_id)
        
        # Calculate scores
        total_violations = len(violations)
        critical_violations = severity_counts.get("critical", 0)
        
        # Base score (100 = perfect)
        base_score = 100
        
        # Penalize based on violations
        if critical_violations > 0:
            base_score -= critical_violations * 20
        base_score -= severity_counts.get("high", 0) * 10
        base_score -= severity_counts.get("medium", 0) * 5
        base_score -= severity_counts.get("low", 0) * 2
        
        # Clamp score
        consistency_score = max(0, min(100, base_score))
        
        # Determine overall status
        if critical_violations > 0:
            status = "critical"
        elif total_violations > 5:
            status = "poor"
        elif total_violations > 0:
            status = "fair"
        else:
            status = "good"
        
        report = {
            "scene_id": scene_id,
            "timestamp": time.time(),
            "status": status,
            "consistency_score": consistency_score,
            "summary": {
                "total_violations": total_violations,
                "total_warnings": len(warnings),
                "severity_counts": dict(severity_counts)
            },
            "violations": [
                {
                    "id": v.violation_id,
                    "rule_id": v.rule_id,
                    "severity": v.severity,
                    "description": v.description,
                    "location": v.location,
                    "objects": v.objects,
                    "metadata": v.metadata
                }
                for v in violations
            ],
            "warnings": warnings,
            "rule_summary": {
                rule_id: len(violations)
                for rule_id, violations in rule_violations.items()
            },
            "recommendations": self._generate_recommendations(violations)
        }
        
        return report
    
    def _generate_recommendations(
        self,
        violations: List[ConsistencyViolation]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for fixing violations"""
        recommendations = []
        
        # Group violations by type
        spatial_violations = [
            v for v in violations 
            if self.rules[v.rule_id].rule_type == ConsistencyRuleType.SPATIAL
        ]
        physical_violations = [
            v for v in violations 
            if self.rules[v.rule_id].rule_type == ConsistencyRuleType.PHYSICAL
        ]
        semantic_violations = [
            v for v in violations 
            if self.rules[v.rule_id].rule_type == ConsistencyRuleType.SEMANTIC
        ]
        
        # Generate recommendations
        if spatial_violations:
            recommendations.append({
                "type": "spatial",
                "priority": "high" if any(v.is_critical for v in spatial_violations) else "medium",
                "description": "Fix object intersections and positioning",
                "actions": [
                    "Run automatic collision resolution",
                    "Adjust object positions manually",
                    "Use grid snapping for alignment"
                ]
            })
        
        if physical_violations:
            recommendations.append({
                "type": "physical",
                "priority": "medium",
                "description": "Improve physical realism",
                "actions": [
                    "Adjust object scales to realistic values",
                    "Ensure proper support for floating objects",
                    "Check mass and density values"
                ]
            })
        
        if semantic_violations:
            recommendations.append({
                "type": "semantic",
                "priority": "low",
                "description": "Improve semantic consistency",
                "actions": [
                    "Move objects to more appropriate locations",
                    "Add missing contextual objects",
                    "Remove semantically inappropriate objects"
                ]
            })
        
        return recommendations
    
    def _update_scene_graph(self, scene_data: Dict[str, Any]) -> None:
        """Update scene graph from scene data"""
        try:
            # Create root node
            root = SceneGraphNode(
                node_id="root",
                node_type="scene",
                properties={"name": scene_data.get("name", "Untitled Scene")}
            )
            
            # Add objects as children
            if "objects" in scene_data:
                for obj_id, obj_data in scene_data["objects"].items():
                    obj_node = SceneGraphNode(
                        node_id=obj_id,
                        node_type="object",
                        properties=obj_data
                    )
                    root.add_child(obj_node)
            
            self.scene_graph = root
            
            # Save snapshot for temporal consistency
            self.scene_snapshot = self._create_scene_snapshot(scene_data)
            
        except Exception as e:
            logger.error(f"Failed to update scene graph: {e}")
    
    def _create_scene_snapshot(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create snapshot of scene for temporal comparison"""
        snapshot = {
            "timestamp": time.time(),
            "objects": {},
            "metadata": scene_data.get("metadata", {})
        }
        
        if "objects" in scene_data:
            for obj_id, obj_data in scene_data["objects"].items():
                # Store essential properties
                snapshot["objects"][obj_id] = {
                    "position": obj_data.get("position", [0, 0, 0]),
                    "rotation": obj_data.get("rotation", [0, 0, 0, 1]),
                    "scale": obj_data.get("scale", [1, 1, 1]),
                    "type": obj_data.get("type", "unknown")
                }
        
        return snapshot
    
    def _generate_violation_id(self, rule_id: str, scene_id: str) -> str:
        """Generate unique violation ID"""
        timestamp = int(time.time() * 1000)
        content = f"{rule_id}_{scene_id}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    # Rule check functions
    def _check_object_intersections(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check for object intersections"""
        objects = scene_data.get("objects", {})
        
        if len(objects) < 2:
            return False, ""
        
        violations = []
        
        # Simple AABB intersection check
        object_bounds = {}
        
        # Calculate bounds for each object
        for obj_id, obj_data in objects.items():
            if "position" in obj_data and "scale" in obj_data:
                pos = np.array(obj_data["position"])
                scale = np.array(obj_data.get("scale", [1, 1, 1]))
                
                # Assume unit cube centered at origin
                min_bound = pos - scale / 2
                max_bound = pos + scale / 2
                
                object_bounds[obj_id] = (min_bound, max_bound)
        
        # Check intersections
        obj_ids = list(objects.keys())
        for i in range(len(obj_ids)):
            for j in range(i + 1, len(obj_ids)):
                obj1_id = obj_ids[i]
                obj2_id = obj_ids[j]
                
                if obj1_id in object_bounds and obj2_id in object_bounds:
                    min1, max1 = object_bounds[obj1_id]
                    min2, max2 = object_bounds[obj2_id]
                    
                    # Check AABB intersection
                    if (max1[0] > min2[0] and min1[0] < max2[0] and
                        max1[1] > min2[1] and min1[1] < max2[1] and
                        max1[2] > min2[2] and min1[2] < max2[2]):
                        
                        violations.append((obj1_id, obj2_id))
        
        if violations:
            message = f"Objects intersecting: {', '.join(f'{a}-{b}' for a, b in violations[:5])}"
            if len(violations) > 5:
                message += f" and {len(violations) - 5} more"
            return True, message
        
        return False, ""
    
    def _check_ground_contact(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if objects are properly grounded"""
        objects = scene_data.get("objects", {})
        ground_level = scene_data.get("environment", {}).get("ground_level", 0)
        
        floating_objects = []
        
        for obj_id, obj_data in objects.items():
            if "position" in obj_data:
                pos_y = obj_data["position"][1]
                scale_y = obj_data.get("scale", [1, 1, 1])[1]
                
                # Object is floating if bottom is above ground
                bottom = pos_y - scale_y / 2
                if bottom > ground_level + 0.1:  # Small tolerance
                    
                    # Check if object is supposed to float (like clouds, birds)
                    obj_type = obj_data.get("type", "").lower()
                    if obj_type not in ["cloud", "bird", "airplane", "light", "ceiling"]:
                        floating_objects.append(obj_id)
        
        if floating_objects:
            message = f"Objects floating above ground: {', '.join(floating_objects[:5])}"
            if len(floating_objects) > 5:
                message += f" and {len(floating_objects) - 5} more"
            return True, message
        
        return False, ""
    
    def _check_semantic_placement(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check semantic appropriateness of object placement"""
        objects = scene_data.get("objects", {})
        
        violations = []
        
        # Simple semantic rules
        semantic_rules = {
            "tree": {"min_y": 0, "max_y": 100, "allowed_environments": ["outdoor", "forest", "park"]},
            "car": {"min_y": 0, "max_y": 0.5, "allowed_environments": ["street", "parking", "garage"]},
            "chair": {"min_y": 0, "max_y": 1, "allowed_environments": ["indoor", "office", "home"]},
            "bed": {"min_y": 0, "max_y": 0.5, "allowed_environments": ["indoor", "bedroom"]},
            "sink": {"min_y": 0.8, "max_y": 1.2, "allowed_environments": ["indoor", "kitchen", "bathroom"]}
        }
        
        scene_environment = scene_data.get("metadata", {}).get("environment", "mixed")
        
        for obj_id, obj_data in objects.items():
            obj_type = obj_data.get("type", "").lower()
            
            if obj_type in semantic_rules:
                rules = semantic_rules[obj_type]
                
                # Check height
                if "position" in obj_data:
                    pos_y = obj_data["position"][1]
                    
                    if pos_y < rules["min_y"] or pos_y > rules["max_y"]:
                        violations.append(f"{obj_id} at inappropriate height")
                
                # Check environment
                if scene_environment not in rules["allowed_environments"]:
                    if "mixed" not in rules["allowed_environments"]:
                        violations.append(f"{obj_id} in inappropriate environment")
        
        if violations:
            message = f"Semantic placement issues: {', '.join(violations[:5])}"
            return True, message
        
        return False, ""
    
    def _check_scale_realism(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if objects have realistic scales"""
        objects = scene_data.get("objects", {})
        
        unrealistic_objects = []
        
        # Expected scales for common objects (in meters)
        expected_scales = {
            "tree": (2, 20),
            "car": (3, 5),
            "chair": (0.5, 1),
            "table": (0.7, 1.5),
            "person": (1.5, 2),
            "house": (5, 20),
            "book": (0.1, 0.3),
            "cup": (0.05, 0.15)
        }
        
        for obj_id, obj_data in objects.items():
            obj_type = obj_data.get("type", "").lower()
            
            if obj_type in expected_scales:
                scale = obj_data.get("scale", [1, 1, 1])
                avg_scale = np.mean(scale)
                min_scale, max_scale = expected_scales[obj_type]
                
                if avg_scale < min_scale * 0.1 or avg_scale > max_scale * 10:
                    unrealistic_objects.append(f"{obj_id} (scale: {avg_scale:.1f}m)")
        
        if unrealistic_objects:
            message = f"Unrealistic scales: {', '.join(unrealistic_objects[:5])}"
            return True, message
        
        return False, ""
    
    def _check_physics_stability(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if objects are physically stable"""
        objects = scene_data.get("objects", {})
        
        unstable_objects = []
        
        for obj_id, obj_data in objects.items():
            # Check for extremely high aspect ratios (likely to tip over)
            if "scale" in obj_data:
                scale = np.array(obj_data["scale"])
                aspect_ratio = max(scale) / min(scale) if min(scale) > 0 else 1
                
                if aspect_ratio > 10:  # Very tall and thin
                    unstable_objects.append(f"{obj_id} (aspect ratio: {aspect_ratio:.1f})")
            
            # Check for objects balanced on edges
            # This would require more complex physics simulation
        
        if unstable_objects:
            message = f"Physically unstable objects: {', '.join(unstable_objects[:5])}"
            return True, message
        
        return False, ""
    
    def _check_texture_consistency(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check texture consistency across objects"""
        objects = scene_data.get("objects", {})
        
        # Group objects by material/texture
        texture_groups = defaultdict(list)
        
        for obj_id, obj_data in objects.items():
            texture = obj_data.get("texture", "default")
            texture_groups[texture].append(obj_id)
        
        # Check for texture variety
        num_textures = len(texture_groups)
        num_objects = len(objects)
        
        if num_objects > 10 and num_textures < 3:
            return False, "Consider adding more texture variety for visual interest"
        
        return False, ""
    
    def _check_lighting_consistency(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check lighting consistency"""
        lighting = scene_data.get("lighting", {})
        
        issues = []
        
        # Check for multiple conflicting light sources
        light_sources = lighting.get("sources", [])
        if len(light_sources) > 4:
            issues.append("Too many light sources, may cause visual confusion")
        
        # Check for unrealistic lighting
        ambient = lighting.get("ambient_intensity", 0.3)
        if ambient > 0.8:
            issues.append("Ambient lighting too bright, may wash out scene")
        
        if issues:
            message = f"Lighting issues: {', '.join(issues)}"
            return True, message
        
        return False, ""
    
    def _check_temporal_consistency(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check temporal consistency with previous state"""
        if not self.scene_snapshot:
            return False, ""
        
        current_objects = scene_data.get("objects", {})
        previous_objects = self.scene_snapshot.get("objects", {})
        
        major_changes = []
        
        # Check for objects that moved too far
        for obj_id in set(current_objects.keys()) & set(previous_objects.keys()):
            current_pos = np.array(current_objects[obj_id].get("position", [0, 0, 0]))
            previous_pos = np.array(previous_objects[obj_id].get("position", [0, 0, 0]))
            
            distance = np.linalg.norm(current_pos - previous_pos)
            if distance > 10:  # Objects moved more than 10 units
                major_changes.append(f"{obj_id} moved {distance:.1f} units")
        
        # Check for objects that changed scale dramatically
        for obj_id in set(current_objects.keys()) & set(previous_objects.keys()):
            current_scale = np.array(current_objects[obj_id].get("scale", [1, 1, 1]))
            previous_scale = np.array(previous_objects[obj_id].get("scale", [1, 1, 1]))
            
            scale_ratio = np.max(current_scale / previous_scale)
            if scale_ratio > 2 or scale_ratio < 0.5:
                major_changes.append(f"{obj_id} changed scale by {scale_ratio:.1f}x")
        
        if major_changes:
            message = f"Major temporal changes: {', '.join(major_changes[:3])}"
            return False, message  # Warning, not violation
        
        return False, ""
    
    def _check_logical_constraints(
        self,
        scene_data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check logical constraints"""
        objects = scene_data.get("objects", {})
        
        violations = []
        
        # Check for contradictory objects
        object_types = [obj.get("type", "").lower() for obj in objects.values()]
        
        if "snowman" in object_types and "palm_tree" in object_types:
            violations.append("Snowman and palm tree in same scene (climate contradiction)")
        
        if "swimming_pool" in object_types and "desert" in scene_data.get("environment", {}):
            violations.append("Swimming pool in desert environment")
        
        if "indoor" in scene_data.get("metadata", {}).get("environment", ""):
            if "sun" in object_types or "moon" in object_types:
                violations.append("Celestial objects in indoor scene")
        
        if violations:
            message = f"Logical constraints violated: {', '.join(violations)}"
            return True, message
        
        return False, ""
    
    def auto_fix_violations(
        self,
        scene_data: Dict[str, Any],
        violations: List[ConsistencyViolation],
        max_fixes: int = 10
    ) -> Dict[str, Any]:
        """
        Automatically fix consistency violations
        
        Args:
            scene_data: Original scene data
            violations: List of violations to fix
            max_fixes: Maximum number of fixes to attempt
            
        Returns:
            Fixed scene data
        """
        fixed_scene = scene_data.copy()
        
        if not violations:
            return fixed_scene
        
        logger.info(f"Attempting to auto-fix {len(violations)} violations")
        
        fixes_applied = 0
        
        for violation in violations[:max_fixes]:
            rule_id = violation.rule_id
            
            if rule_id == "no_intersections":
                fixed_scene = self._fix_intersections(fixed_scene, violation)
                fixes_applied += 1
            
            elif rule_id == "ground_contact":
                fixed_scene = self._fix_ground_contact(fixed_scene, violation)
                fixes_applied += 1
            
            elif rule_id == "scale_realism":
                fixed_scene = self._fix_scale_realism(fixed_scene, violation)
                fixes_applied += 1
        
        logger.info(f"Applied {fixes_applied} automatic fixes")
        
        return fixed_scene
    
    def _fix_intersections(
        self,
        scene_data: Dict[str, Any],
        violation: ConsistencyViolation
    ) -> Dict[str, Any]:
        """Fix object intersections"""
        fixed_scene = scene_data.copy()
        objects = fixed_scene.get("objects", {})
        
        # Simple fix: move intersecting objects apart
        for obj_id in violation.objects:
            if obj_id in objects:
                pos = np.array(objects[obj_id].get("position", [0, 0, 0]))
                # Move object up slightly
                pos[1] += 0.5
                objects[obj_id]["position"] = pos.tolist()
        
        return fixed_scene
    
    def _fix_ground_contact(
        self,
        scene_data: Dict[str, Any],
        violation: ConsistencyViolation
    ) -> Dict[str, Any]:
        """Fix floating objects"""
        fixed_scene = scene_data.copy()
        objects = fixed_scene.get("objects", {})
        ground_level = scene_data.get("environment", {}).get("ground_level", 0)
        
        for obj_id in violation.objects:
            if obj_id in objects:
                pos = objects[obj_id].get("position", [0, 0, 0])
                scale = objects[obj_id].get("scale", [1, 1, 1])
                
                # Calculate required Y position to touch ground
                required_y = ground_level + scale[1] / 2
                pos[1] = required_y
                objects[obj_id]["position"] = pos
        
        return fixed_scene
    
    def _fix_scale_realism(
        self,
        scene_data: Dict[str, Any],
        violation: ConsistencyViolation
    ) -> Dict[str, Any]:
        """Fix unrealistic scales"""
        fixed_scene = scene_data.copy()
        objects = fixed_scene.get("objects", {})
        
        # Expected scales for common objects
        expected_scales = {
            "tree": 5.0,
            "car": 4.0,
            "chair": 0.8,
            "table": 1.0,
            "person": 1.7,
            "house": 10.0,
            "book": 0.2,
            "cup": 0.1
        }
        
        for obj_id in violation.objects:
            if obj_id in objects:
                obj_type = objects[obj_id].get("type", "").lower()
                
                if obj_type in expected_scales:
                    target_scale = expected_scales[obj_type]
                    current_scale = np.mean(objects[obj_id].get("scale", [1, 1, 1]))
                    
                    if current_scale > 0:
                        scale_factor = target_scale / current_scale
                        new_scale = [s * scale_factor for s in objects[obj_id].get("scale", [1, 1, 1])]
                        objects[obj_id]["scale"] = new_scale
        
        return fixed_scene
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get statistics about violations"""
        total_violations = len(self.violations)
        critical_violations = sum(1 for v in self.violations.values() if v.is_critical)
        
        # Count by rule type
        rule_type_counts = defaultdict(int)
        for violation in self.violations.values():
            rule = self.rules.get(violation.rule_id)
            if rule:
                rule_type_counts[rule.rule_type.value] += 1
        
        return {
            "total_violations": total_violations,
            "critical_violations": critical_violations,
            "rule_type_counts": dict(rule_type_counts),
            "recent_violations": len(self.violation_history)
        }
    
    def export_violations(
        self,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """Export violations to file"""
        output_path = Path(output_path)
        
        if format == "json":
            data = {
                "violations": [
                    {
                        "id": v.violation_id,
                        "rule_id": v.rule_id,
                        "severity": v.severity,
                        "description": v.description,
                        "location": v.location,
                        "objects": v.objects,
                        "timestamp": v.timestamp,
                        "metadata": v.metadata
                    }
                    for v in self.violations.values()
                ],
                "summary": self.get_violation_stats(),
                "exported": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            save_json(data, output_path.with_suffix('.json'))
            logger.info(f"Exported violations to {output_path.with_suffix('.json')}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_violations(self) -> None:
        """Clear all violations"""
        self.violations.clear()
        self.violation_history.clear()
        logger.info("All violations cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("ConsistencyChecker cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        return (f"ConsistencyChecker(rules={len(self.rules)}, "
                f"violations={len(self.violations)})")


# Factory function for creating consistency checkers
def create_consistency_checker(
    config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4
) -> ConsistencyChecker:
    """
    Factory function to create consistency checkers
    
    Args:
        config: Configuration dictionary
        max_workers: Maximum worker threads
        
    Returns:
        ConsistencyChecker instance
    """
    return ConsistencyChecker(config=config, max_workers=max_workers)