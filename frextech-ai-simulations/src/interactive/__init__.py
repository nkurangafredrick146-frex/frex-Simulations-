"""
Interactive Module Package
Tools for interactive scene editing, composition, and expansion
"""

from .composition.scene_composer import SceneComposer, create_scene_composer
from .composition.style_transfer import StyleTransferEngine, create_style_transfer_engine
from .composition.transition_builder import TransitionBuilder, create_transition_builder

from .editor.consistency_checker import ConsistencyChecker, create_consistency_checker
from .editor.edit_propagator import EditPropagator, create_edit_propagator
from .editor.pano_editor import PanoramaEditor, create_panorama_editor
from .editor.region_selector import RegionSelector, create_region_selector

from .expansion.boundary_detector import BoundaryDetector, create_boundary_detector
from .expansion.scene_expander import SceneExpander, create_scene_expander
from .expansion.seam_blender import SeamBlender, create_seam_blender

__version__ = "1.0.0"
__author__ = "FrexTech AI Team"
__license__ = "Proprietary"

# Export main classes
__all__ = [
    # Composition
    "SceneComposer",
    "create_scene_composer",
    "StyleTransferEngine",
    "create_style_transfer_engine",
    "TransitionBuilder",
    "create_transition_builder",
    
    # Editor
    "ConsistencyChecker",
    "create_consistency_checker",
    "EditPropagator",
    "create_edit_propagator",
    "PanoramaEditor",
    "create_panorama_editor",
    "RegionSelector",
    "create_region_selector",
    
    # Expansion
    "BoundaryDetector",
    "create_boundary_detector",
    "SceneExpander",
    "create_scene_expander",
    "SeamBlender",
    "create_seam_blender",
]


class InteractivePipeline:
    """
    Complete interactive pipeline for scene manipulation
    
    This class provides a unified interface for all interactive tools.
    """
    
    def __init__(
        self,
        config: dict = None,
        max_workers: int = 4
    ):
        """
        Initialize interactive pipeline
        
        Args:
            config: Configuration dictionary
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.max_workers = max_workers
        
        # Initialize all components
        self._initialize_components()
        
        # Pipeline state
        self.current_scene = None
        self.history = []
        self.undo_stack = []
        
    def _initialize_components(self):
        """Initialize all interactive components"""
        # Composition tools
        self.scene_composer = create_scene_composer(
            config=self.config.get("composition", {}),
            max_workers=self.max_workers
        )
        
        self.style_transfer = create_style_transfer_engine(
            method="neural_style",
            device="auto"
        )
        
        self.transition_builder = create_transition_builder(
            config=self.config.get("transitions", {}),
            max_workers=self.max_workers
        )
        
        # Editor tools
        self.consistency_checker = create_consistency_checker(
            config=self.config.get("consistency", {}),
            max_workers=self.max_workers
        )
        
        self.edit_propagator = create_edit_propagator(
            config=self.config.get("propagation", {}),
            consistency_checker=self.consistency_checker,
            max_workers=self.max_workers
        )
        
        self.pano_editor = create_panorama_editor(
            config=self.config.get("panorama", {}),
            consistency_checker=self.consistency_checker,
            edit_propagator=self.edit_propagator,
            max_workers=self.max_workers
        )
        
        self.region_selector = create_region_selector(
            config=self.config.get("selection", {}),
            max_workers=self.max_workers
        )
        
        # Expansion tools
        self.boundary_detector = create_boundary_detector(
            config=self.config.get("boundaries", {}),
            max_workers=self.max_workers
        )
        
        self.scene_expander = create_scene_expander(
            config=self.config.get("expansion", {}),
            boundary_detector=self.boundary_detector,
            scene_composer=self.scene_composer,
            max_workers=self.max_workers
        )
        
        self.seam_blender = create_seam_blender(
            config=self.config.get("blending", {}),
            boundary_detector=self.boundary_detector,
            max_workers=self.max_workers
        )
        
    def load_scene(self, scene_data: dict) -> bool:
        """
        Load a scene into the pipeline
        
        Args:
            scene_data: Scene data dictionary
            
        Returns:
            True if successful
        """
        try:
            self.current_scene = scene_data.copy()
            
            # Load into relevant components
            if self.scene_composer:
                # Would need to convert scene data to SceneComposer format
                pass
            
            if self.scene_expander:
                self.scene_expander.load_scene(scene_data)
            
            # Record in history
            self.history.append({
                "action": "load_scene",
                "timestamp": time.time(),
                "scene_id": scene_data.get("id", "unknown")
            })
            
            return True
            
        except Exception as e:
            print(f"Error loading scene: {e}")
            return False
    
    def compose_scene(
        self,
        objects: list,
        layout: str = "open_world",
        constraints: dict = None
    ) -> dict:
        """
        Compose a scene from objects
        
        Args:
            objects: List of scene objects
            layout: Scene layout type
            constraints: Composition constraints
            
        Returns:
            Composed scene
        """
        # Convert objects to SceneComposer format
        scene_objects = []
        # ... conversion logic ...
        
        result = self.scene_composer.compose_scene(
            objects=scene_objects,
            constraints=constraints,
            layout=layout
        )
        
        # Update current scene
        self.current_scene = result
        
        # Record action
        self.history.append({
            "action": "compose_scene",
            "timestamp": time.time(),
            "num_objects": len(objects),
            "layout": layout
        })
        
        return result
    
    def expand_scene(
        self,
        direction: str = "right",
        distance: float = 10.0,
        method: str = "composition"
    ) -> dict:
        """
        Expand the current scene
        
        Args:
            direction: Expansion direction
            distance: Expansion distance
            method: Expansion method
            
        Returns:
            Expanded scene
        """
        if self.current_scene is None:
            raise ValueError("No scene loaded")
        
        request = {
            "direction": direction,
            "distance": distance,
            "method": method
        }
        
        result = self.scene_expander.expand_scene(request)
        
        if result:
            self.current_scene = result.expanded_scene
            
            self.history.append({
                "action": "expand_scene",
                "timestamp": time.time(),
                "direction": direction,
                "distance": distance,
                "method": method,
                "new_objects": result.num_new_objects
            })
            
            return result.expanded_scene
        
        return self.current_scene
    
    def apply_style(
        self,
        style_preset: str,
        components: list = None
    ) -> dict:
        """
        Apply style to current scene
        
        Args:
            style_preset: Style preset name
            components: List of components to style
            
        Returns:
            Styled scene
        """
        if self.current_scene is None:
            raise ValueError("No scene loaded")
        
        styled_scene = self.style_transfer.apply_to_scene(
            self.current_scene,
            style_preset,
            components
        )
        
        self.current_scene = styled_scene
        
        self.history.append({
            "action": "apply_style",
            "timestamp": time.time(),
            "style_preset": style_preset,
            "components": components
        })
        
        return styled_scene
    
    def check_consistency(self, rule_types: list = None) -> dict:
        """
        Check scene consistency
        
        Args:
            rule_types: List of rule types to check
            
        Returns:
            Consistency report
        """
        if self.current_scene is None:
            raise ValueError("No scene loaded")
        
        report = self.consistency_checker.check_scene(
            self.current_scene,
            rule_types=rule_types
        )
        
        return report
    
    def create_transition(
        self,
        target_scene: dict,
        transition_type: str = "fade",
        duration: float = 2.0
    ) -> dict:
        """
        Create transition to target scene
        
        Args:
            target_scene: Target scene data
            transition_type: Type of transition
            duration: Transition duration
            
        Returns:
            Transition data
        """
        if self.current_scene is None:
            raise ValueError("No scene loaded")
        
        # Create scene states
        from_state = self.transition_builder.SceneState(
            id="current",
            objects=self.current_scene.get("objects", {}),
            camera=self.current_scene.get("camera", {}),
            lighting=self.current_scene.get("lighting", {}),
            environment=self.current_scene.get("environment", {})
        )
        
        to_state = self.transition_builder.SceneState(
            id="target",
            objects=target_scene.get("objects", {}),
            camera=target_scene.get("camera", {}),
            lighting=target_scene.get("lighting", {}),
            environment=target_scene.get("environment", {})
        )
        
        # Create transition
        config = self.transition_builder.TransitionConfig(
            transition_type=self.transition_builder.TransitionType(transition_type),
            duration=duration
        )
        
        transition = self.transition_builder.create_transition(
            from_state, to_state, config
        )
        
        return transition
    
    def undo_last_action(self) -> bool:
        """
        Undo last action
        
        Returns:
            True if successful
        """
        if not self.history:
            return False
        
        last_action = self.history.pop()
        self.undo_stack.append(last_action)
        
        # In a full implementation, would restore previous state
        print(f"Undid action: {last_action['action']}")
        
        return True
    
    def redo_last_action(self) -> bool:
        """
        Redo last undone action
        
        Returns:
            True if successful
        """
        if not self.undo_stack:
            return False
        
        last_undone = self.undo_stack.pop()
        self.history.append(last_undone)
        
        # In a full implementation, would reapply the action
        print(f"Redid action: {last_undone['action']}")
        
        return True
    
    def export_pipeline_state(self, output_path: str) -> bool:
        """
        Export pipeline state to file
        
        Args:
            output_path: Output file path
            
        Returns:
            True if successful
        """
        try:
            state = {
                "current_scene": self.current_scene,
                "history": self.history,
                "undo_stack": self.undo_stack,
                "config": self.config,
                "export_timestamp": time.time()
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting pipeline state: {e}")
            return False
    
    def import_pipeline_state(self, input_path: str) -> bool:
        """
        Import pipeline state from file
        
        Args:
            input_path: Input file path
            
        Returns:
            True if successful
        """
        try:
            import json
            with open(input_path, 'r') as f:
                state = json.load(f)
            
            self.current_scene = state.get("current_scene")
            self.history = state.get("history", [])
            self.undo_stack = state.get("undo_stack", [])
            
            return True
            
        except Exception as e:
            print(f"Error importing pipeline state: {e}")
            return False
    
    def get_statistics(self) -> dict:
        """Get pipeline statistics"""
        return {
            "components_initialized": {
                "composition": self.scene_composer is not None,
                "style_transfer": self.style_transfer is not None,
                "transitions": self.transition_builder is not None,
                "consistency": self.consistency_checker is not None,
                "propagation": self.edit_propagator is not None,
                "panorama": self.pano_editor is not None,
                "selection": self.region_selector is not None,
                "boundaries": self.boundary_detector is not None,
                "expansion": self.scene_expander is not None,
                "blending": self.seam_blender is not None
            },
            "history_length": len(self.history),
            "undo_stack_length": len(self.undo_stack),
            "current_scene": {
                "has_scene": self.current_scene is not None,
                "object_count": len(self.current_scene.get("objects", {})) if self.current_scene else 0
            }
        }
    
    def cleanup(self):
        """Clean up all pipeline components"""
        components = [
            self.scene_composer,
            self.style_transfer,
            self.transition_builder,
            self.consistency_checker,
            self.edit_propagator,
            self.pano_editor,
            self.region_selector,
            self.boundary_detector,
            self.scene_expander,
            self.seam_blender
        ]
        
        for component in components:
            if hasattr(component, 'cleanup'):
                try:
                    component.cleanup()
                except:
                    pass
        
        print("Interactive pipeline cleaned up")


# Convenience function to create a complete interactive pipeline
def create_interactive_pipeline(
    config: dict = None,
    max_workers: int = 4
) -> InteractivePipeline:
    """
    Create a complete interactive pipeline
    
    Args:
        config: Configuration dictionary
        max_workers: Maximum worker threads
        
    Returns:
        InteractivePipeline instance
    """
    return InteractivePipeline(config=config, max_workers=max_workers)


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = create_interactive_pipeline()
    
    # Example scene data
    example_scene = {
        "id": "example_scene",
        "objects": {
            "tree_1": {
                "type": "tree",
                "position": [0, 0, 0],
                "scale": [1, 1, 1]
            }
        },
        "metadata": {
            "created": "2024-01-01"
        }
    }
    
    # Load scene
    pipeline.load_scene(example_scene)
    
    # Expand scene
    expanded = pipeline.expand_scene(direction="right", distance=20.0)
    
    # Check consistency
    report = pipeline.check_consistency()
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"Pipeline statistics: {stats}")
    
    # Cleanup
    pipeline.cleanup()