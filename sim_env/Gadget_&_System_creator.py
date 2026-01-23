"""
Advanced Gadget & System Creator
Advanced system for creating, customizing, and integrating custom gadgets, tools, and simulation systems
with drag-and-drop interface, visual programming, and AI-assisted design.
"""

import numpy as np
import pygame
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
import inspect
import ast
import hashlib
import pickle
from pathlib import Path
import logging
from abc import ABC, abstractmethod

class GadgetType(Enum):
    """Types of gadgets that can be created"""
    SENSOR = "sensor"
    ACTUATOR = "actuator"
    PROCESSOR = "processor"
    DISPLAY = "display"
    CONTROLLER = "controller"
    GENERATOR = "generator"
    ANALYZER = "analyzer"
    SIMULATION = "simulation"
    CUSTOM = "custom"

class ComponentCategory(Enum):
    """Categories of gadget components"""
    INPUT = "input"
    OUTPUT = "output"
    PROCESSING = "processing"
    STORAGE = "storage"
    VISUALIZATION = "visualization"
    PHYSICS = "physics"
    AI = "ai"
    NETWORK = "network"

class ConnectionType(Enum):
    """Types of connections between components"""
    DATA_FLOW = "data_flow"
    CONTROL_FLOW = "control_flow"
    PHYSICAL = "physical"
    ENERGY = "energy"
    SIGNAL = "signal"

@dataclass
class ComponentParameter:
    """Parameter definition for gadget components"""
    name: str
    data_type: str
    default_value: Any
    description: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    options: List[Any] = field(default_factory=list)

@dataclass
class ComponentInterface:
    """Interface definition for gadget components"""
    inputs: List[ComponentParameter] = field(default_factory=list)
    outputs: List[ComponentParameter] = field(default_factory=list)
    parameters: List[ComponentParameter] = field(default_factory=list)

@dataclass
class GadgetComponent:
    """Base gadget component"""
    component_id: str
    name: str
    category: ComponentCategory
    interface: ComponentInterface
    position: Tuple[float, float] = (0, 0)
    size: Tuple[float, float] = (100, 80)
    configuration: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentConnection:
    """Connection between gadget components"""
    connection_id: str
    source_component: str
    source_output: str
    target_component: str
    target_input: str
    connection_type: ConnectionType
    data_type: str
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GadgetDefinition:
    """Complete gadget definition"""
    gadget_id: str
    name: str
    description: str
    gadget_type: GadgetType
    components: Dict[str, GadgetComponent]
    connections: List[ComponentConnection]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    author: str = "unknown"

@dataclass
class GadgetInstance:
    """Running instance of a gadget"""
    instance_id: str
    gadget_definition: GadgetDefinition
    component_states: Dict[str, Dict[str, Any]]
    connections: List[ComponentConnection]
    is_running: bool = False
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class ComponentBase(ABC):
    """Base class for all gadget components"""
    
    def __init__(self, component_id: str, name: str, category: ComponentCategory):
        self.component_id = component_id
        self.name = name
        self.category = category
        self.interface = ComponentInterface()
        self.configuration = {}
        self.state = {}
        self.input_values = {}
        self.output_values = {}
        self.connected_inputs = set()
        self.connected_outputs = set()
        
    @abstractmethod
    def initialize(self):
        """Initialize the component"""
        pass
        
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs"""
        pass
        
    @abstractmethod
    def cleanup(self):
        """Clean up resources"""
        pass
        
    def validate_connection(self, output_name: str, target_input: str, 
                          target_component: 'ComponentBase') -> bool:
        """Validate if connection is possible"""
        # Check if output exists
        output_param = next((p for p in self.interface.outputs if p.name == output_name), None)
        if not output_param:
            return False
            
        # Check if target input exists and types match
        target_input_param = next((p for p in target_component.interface.inputs 
                                 if p.name == target_input), None)
        if not target_input_param:
            return False
            
        # Check type compatibility
        return self._types_compatible(output_param.data_type, target_input_param.data_type)
        
    def _types_compatible(self, source_type: str, target_type: str) -> bool:
        """Check if data types are compatible"""
        type_hierarchy = {
            'any': ['number', 'string', 'boolean', 'vector', 'matrix', 'any'],
            'number': ['number'],
            'string': ['string'],
            'boolean': ['boolean'],
            'vector': ['vector', 'matrix'],
            'matrix': ['matrix']
        }
        
        source_types = type_hierarchy.get(source_type, [source_type])
        return target_type in source_types

class SensorComponent(ComponentBase):
    """Base class for sensor components"""
    
    def __init__(self, component_id: str, name: str):
        super().__init__(component_id, name, ComponentCategory.INPUT)
        self.sampling_rate = 1.0
        self.accuracy = 1.0
        self.last_reading_time = 0
        
    @abstractmethod
    def read_sensor(self) -> Dict[str, Any]:
        """Read sensor data"""
        pass

class ProcessorComponent(ComponentBase):
    """Base class for processor components"""
    
    def __init__(self, component_id: str, name: str):
        super().__init__(component_id, name, ComponentCategory.PROCESSING)
        self.processing_time = 0.0
        self.cache_enabled = False
        
    @abstractmethod
    def transform_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform input data"""
        pass

class ActuatorComponent(ComponentBase):
    """Base class for actuator components"""
    
    def __init__(self, component_id: str, name: str):
        super().__init__(component_id, name, ComponentCategory.OUTPUT)
        self.actuation_power = 1.0
        self.response_time = 0.1
        
    @abstractmethod
    def actuate(self, inputs: Dict[str, Any]) -> bool:
        """Perform actuation based on inputs"""
        pass

class ParticleSensor(SensorComponent):
    """Sensor for detecting particle properties"""
    
    def __init__(self, component_id: str):
        super().__init__(component_id, "Particle Sensor")
        
        # Define interface
        self.interface.outputs = [
            ComponentParameter("particle_count", "number", 0, "Number of particles detected"),
            ComponentParameter("average_velocity", "number", 0, "Average particle velocity"),
            ComponentParameter("density", "number", 0, "Particle density"),
            ComponentParameter("temperature", "number", 0, "System temperature")
        ]
        
        self.interface.parameters = [
            ComponentParameter("detection_range", "number", 10.0, "Sensor detection range"),
            ComponentParameter("sampling_rate", "number", 30.0, "Samples per second")
        ]
        
    def initialize(self):
        """Initialize particle sensor"""
        self.sampling_rate = self.configuration.get("sampling_rate", 30.0)
        self.detection_range = self.configuration.get("detection_range", 10.0)
        self.state["calibration_factor"] = 1.0
        logging.info(f"Particle Sensor {self.component_id} initialized")
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor reading"""
        current_time = time.time()
        
        # Only read at specified sampling rate
        if current_time - self.last_reading_time < 1.0 / self.sampling_rate:
            return self.output_values
            
        # Simulate particle detection (in real implementation, this would access simulation data)
        particle_data = inputs.get("particle_data", {})
        particles = particle_data.get("particles", [])
        
        if particles:
            positions = np.array([p.get("position", [0, 0, 0]) for p in particles])
            velocities = np.array([p.get("velocity", [0, 0, 0]) for p in particles])
            
            # Calculate metrics
            particle_count = len(particles)
            avg_velocity = np.mean(np.linalg.norm(velocities, axis=1))
            
            # Calculate density (particles per unit volume)
            if len(positions) > 0:
                bbox = np.ptp(positions, axis=0)
                volume = np.prod(bbox) if np.all(bbox > 0) else 1.0
                density = particle_count / volume
            else:
                density = 0
                
            # Estimate temperature from velocity variance
            temperature = np.var(np.linalg.norm(velocities, axis=1)) * 100
            
            self.output_values = {
                "particle_count": particle_count,
                "average_velocity": avg_velocity,
                "density": density,
                "temperature": temperature
            }
        else:
            self.output_values = {
                "particle_count": 0,
                "average_velocity": 0,
                "density": 0,
                "temperature": 0
            }
            
        self.last_reading_time = current_time
        return self.output_values
        
    def read_sensor(self) -> Dict[str, Any]:
        """Read sensor data"""
        return self.process({})
        
    def cleanup(self):
        """Clean up sensor"""
        logging.info(f"Particle Sensor {self.component_id} cleaned up")

class DataProcessor(ProcessorComponent):
    """Data processing component with various operations"""
    
    def __init__(self, component_id: str):
        super().__init__(component_id, "Data Processor")
        
        # Define interface
        self.interface.inputs = [
            ComponentParameter("input_data", "any", None, "Input data to process")
        ]
        
        self.interface.outputs = [
            ComponentParameter("processed_data", "any", None, "Processed output data"),
            ComponentParameter("statistics", "any", {}, "Processing statistics"),
            ComponentParameter("error", "string", "", "Processing errors")
        ]
        
        self.interface.parameters = [
            ComponentParameter("operation", "string", "filter", "Processing operation", 
                             options=["filter", "transform", "analyze", "aggregate"]),
            ComponentParameter("filter_threshold", "number", 0.5, "Filter threshold"),
            ComponentParameter("transform_function", "string", "x * 2", "Transformation function")
        ]
        
    def initialize(self):
        """Initialize data processor"""
        self.operation = self.configuration.get("operation", "filter")
        self.filter_threshold = self.configuration.get("filter_threshold", 0.5)
        self.transform_function = self.configuration.get("transform_function", "x * 2")
        self.state["processed_count"] = 0
        logging.info(f"Data Processor {self.component_id} initialized")
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data"""
        input_data = inputs.get("input_data")
        error = ""
        processed_data = None
        statistics = {}
        
        try:
            if input_data is not None:
                if self.operation == "filter":
                    processed_data = self._apply_filter(input_data)
                elif self.operation == "transform":
                    processed_data = self._apply_transform(input_data)
                elif self.operation == "analyze":
                    processed_data, statistics = self._analyze_data(input_data)
                elif self.operation == "aggregate":
                    processed_data = self._aggregate_data(input_data)
                    
                self.state["processed_count"] += 1
                statistics["processed_count"] = self.state["processed_count"]
                
        except Exception as e:
            error = str(e)
            logging.error(f"Data processing error: {e}")
            
        self.output_values = {
            "processed_data": processed_data,
            "statistics": statistics,
            "error": error
        }
        
        return self.output_values
        
    def _apply_filter(self, data: Any) -> Any:
        """Apply filter to data"""
        if isinstance(data, (list, np.ndarray)):
            return [x for x in data if abs(x) > self.filter_threshold]
        elif isinstance(data, (int, float)):
            return data if abs(data) > self.filter_threshold else 0
        else:
            return data
            
    def _apply_transform(self, data: Any) -> Any:
        """Apply transformation function"""
        try:
            # Simple transformation using eval (in production, use safer methods)
            if isinstance(data, (list, np.ndarray)):
                return [eval(self.transform_function, {"x": x, "np": np}) for x in data]
            else:
                return eval(self.transform_function, {"x": data, "np": np})
        except:
            return data
            
    def _analyze_data(self, data: Any) -> Tuple[Any, Dict[str, Any]]:
        """Analyze data and return statistics"""
        stats = {}
        
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            data_array = np.array(data)
            stats = {
                "mean": float(np.mean(data_array)),
                "std": float(np.std(data_array)),
                "min": float(np.min(data_array)),
                "max": float(np.max(data_array)),
                "count": len(data_array)
            }
            
        return data, stats
        
    def _aggregate_data(self, data: Any) -> Any:
        """Aggregate data"""
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            return np.sum(data)
        else:
            return data
            
    def transform_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transform input data"""
        return self.process(inputs)
        
    def cleanup(self):
        """Clean up processor"""
        logging.info(f"Data Processor {self.component_id} cleaned up")

class ForceActuator(ActuatorComponent):
    """Actuator for applying forces in simulation"""
    
    def __init__(self, component_id: str):
        super().__init__(component_id, "Force Actuator")
        
        # Define interface
        self.interface.inputs = [
            ComponentParameter("force_magnitude", "number", 1.0, "Magnitude of force to apply"),
            ComponentParameter("force_direction", "vector", [1, 0, 0], "Direction of force"),
            ComponentParameter("target_position", "vector", [0, 0, 0], "Position to apply force"),
            ComponentParameter("enabled", "boolean", True, "Whether actuator is enabled")
        ]
        
        self.interface.outputs = [
            ComponentParameter("force_applied", "boolean", False, "Whether force was applied"),
            ComponentParameter("actual_force", "vector", [0, 0, 0], "Actual force applied")
        ]
        
        self.interface.parameters = [
            ComponentParameter("max_force", "number", 10.0, "Maximum force magnitude"),
            ComponentParameter("force_type", "string", "constant", "Type of force",
                             options=["constant", "impulse", "field"]),
            ComponentParameter("falloff_radius", "number", 5.0, "Force field falloff radius")
        ]
        
    def initialize(self):
        """Initialize force actuator"""
        self.max_force = self.configuration.get("max_force", 10.0)
        self.force_type = self.configuration.get("force_type", "constant")
        self.falloff_radius = self.configuration.get("falloff_radius", 5.0)
        self.state["total_force_applied"] = 0.0
        logging.info(f"Force Actuator {self.component_id} initialized")
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process actuator inputs"""
        enabled = inputs.get("enabled", True)
        force_magnitude = inputs.get("force_magnitude", 1.0)
        force_direction = inputs.get("force_direction", [1, 0, 0])
        target_position = inputs.get("target_position", [0, 0, 0])
        
        force_applied = False
        actual_force = [0, 0, 0]
        
        if enabled and force_magnitude > 0:
            # Clamp force magnitude
            clamped_magnitude = min(abs(force_magnitude), self.max_force)
            
            # Normalize direction
            direction_array = np.array(force_direction)
            if np.linalg.norm(direction_array) > 0:
                normalized_direction = direction_array / np.linalg.norm(direction_array)
                actual_force = (normalized_direction * clamped_magnitude).tolist()
                
                # Apply force in simulation (this would interface with physics engine)
                self._apply_force_in_simulation(actual_force, target_position)
                
                force_applied = True
                self.state["total_force_applied"] += clamped_magnitude
                
        self.output_values = {
            "force_applied": force_applied,
            "actual_force": actual_force
        }
        
        return self.output_values
        
    def _apply_force_in_simulation(self, force: List[float], position: List[float]):
        """Apply force in the simulation (placeholder implementation)"""
        # In real implementation, this would interface with the physics engine
        logging.info(f"Applying force {force} at position {position}")
        
    def actuate(self, inputs: Dict[str, Any]) -> bool:
        """Perform actuation"""
        result = self.process(inputs)
        return result.get("force_applied", False)
        
    def cleanup(self):
        """Clean up actuator"""
        logging.info(f"Force Actuator {self.component_id} cleaned up")

class VisualizerComponent(ComponentBase):
    """Component for data visualization"""
    
    def __init__(self, component_id: str):
        super().__init__(component_id, "Data Visualizer", ComponentCategory.VISUALIZATION)
        
        # Define interface
        self.interface.inputs = [
            ComponentParameter("data", "any", None, "Data to visualize"),
            ComponentParameter("visualization_type", "string", "plot", "Type of visualization")
        ]
        
        self.interface.outputs = [
            ComponentParameter("visualization_data", "any", None, "Processed visualization data"),
            ComponentParameter("rendering_instructions", "any", {}, "Rendering instructions")
        ]
        
        self.interface.parameters = [
            ComponentParameter("color_scheme", "string", "viridis", "Color scheme"),
            ComponentParameter("width", "number", 800, "Visualization width"),
            ComponentParameter("height", "number", 600, "Visualization height")
        ]
        
    def initialize(self):
        """Initialize visualizer"""
        self.color_scheme = self.configuration.get("color_scheme", "viridis")
        self.width = self.configuration.get("width", 800)
        self.height = self.configuration.get("height", 600)
        self.state["visualizations_generated"] = 0
        logging.info(f"Visualizer {self.component_id} initialized")
        
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process visualization data"""
        data = inputs.get("data")
        viz_type = inputs.get("visualization_type", "plot")
        
        visualization_data = None
        rendering_instructions = {}
        
        if data is not None:
            if viz_type == "plot" and isinstance(data, (list, np.ndarray)):
                visualization_data = self._create_plot(data)
            elif viz_type == "histogram" and isinstance(data, (list, np.ndarray)):
                visualization_data = self._create_histogram(data)
            elif viz_type == "heatmap" and isinstance(data, (list, np.ndarray)):
                visualization_data = self._create_heatmap(data)
                
            rendering_instructions = {
                "type": viz_type,
                "width": self.width,
                "height": self.height,
                "color_scheme": self.color_scheme
            }
            
            self.state["visualizations_generated"] += 1
            
        self.output_values = {
            "visualization_data": visualization_data,
            "rendering_instructions": rendering_instructions
        }
        
        return self.output_values
        
    def _create_plot(self, data: Any) -> Dict[str, Any]:
        """Create plot data"""
        if isinstance(data, (list, np.ndarray)):
            return {
                "type": "line_plot",
                "x_values": list(range(len(data))),
                "y_values": data.tolist() if isinstance(data, np.ndarray) else data,
                "title": "Data Plot"
            }
        return {}
        
    def _create_histogram(self, data: Any) -> Dict[str, Any]:
        """Create histogram data"""
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            hist, bin_edges = np.histogram(data, bins=20)
            return {
                "type": "histogram",
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "title": "Data Histogram"
            }
        return {}
        
    def _create_heatmap(self, data: Any) -> Dict[str, Any]:
        """Create heatmap data"""
        if isinstance(data, (list, np.ndarray)):
            # Ensure 2D data for heatmap
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim > 2:
                data = data.reshape(data.shape[0], -1)
                
            return {
                "type": "heatmap",
                "matrix": data.tolist(),
                "title": "Data Heatmap"
            }
        return {}
        
    def cleanup(self):
        """Clean up visualizer"""
        logging.info(f"Visualizer {self.component_id} cleaned up")

class ComponentLibrary:
    """Library of available gadget components"""
    
    def __init__(self):
        self.components = {}
        self._register_builtin_components()
        
    def _register_builtin_components(self):
        """Register built-in components"""
        # Sensor components
        self.register_component("particle_sensor", ParticleSensor)
        self.register_component("temperature_sensor", 
                              lambda cid: self._create_sensor_template(cid, "Temperature Sensor", "temperature"))
        self.register_component("pressure_sensor", 
                              lambda cid: self._create_sensor_template(cid, "Pressure Sensor", "pressure"))
        
        # Processor components
        self.register_component("data_processor", DataProcessor)
        self.register_component("filter", 
                              lambda cid: self._create_processor_template(cid, "Filter", "filter"))
        self.register_component("calculator", 
                              lambda cid: self._create_processor_template(cid, "Calculator", "calculate"))
        
        # Actuator components
        self.register_component("force_actuator", ForceActuator)
        self.register_component("emitter", 
                              lambda cid: self._create_actuator_template(cid, "Particle Emitter", "emit"))
        self.register_component("field_generator", 
                              lambda cid: self._create_actuator_template(cid, "Field Generator", "generate_field"))
        
        # Visualization components
        self.register_component("visualizer", VisualizerComponent)
        self.register_component("plotter", 
                              lambda cid: self._create_visualizer_template(cid, "Plotter", "plot"))
        self.register_component("display", 
                              lambda cid: self._create_visualizer_template(cid, "Display", "display"))
        
    def _create_sensor_template(self, component_id: str, name: str, sensor_type: str) -> SensorComponent:
        """Create a template sensor component"""
        sensor = SensorComponent(component_id, name)
        sensor.interface.outputs = [
            ComponentParameter(f"{sensor_type}_reading", "number", 0, f"{sensor_type} reading"),
            ComponentParameter("accuracy", "number", 1.0, "Measurement accuracy")
        ]
        return sensor
        
    def _create_processor_template(self, component_id: str, name: str, operation: str) -> ProcessorComponent:
        """Create a template processor component"""
        processor = ProcessorComponent(component_id, name)
        processor.interface.inputs = [
            ComponentParameter("input_data", "any", None, "Input data")
        ]
        processor.interface.outputs = [
            ComponentParameter("output_data", "any", None, "Processed output data")
        ]
        return processor
        
    def _create_actuator_template(self, component_id: str, name: str, action: str) -> ActuatorComponent:
        """Create a template actuator component"""
        actuator = ActuatorComponent(component_id, name)
        actuator.interface.inputs = [
            ComponentParameter("intensity", "number", 1.0, f"{action} intensity"),
            ComponentParameter("target", "any", None, f"Target for {action}")
        ]
        actuator.interface.outputs = [
            ComponentParameter("success", "boolean", False, f"{action} success status")
        ]
        return actuator
        
    def _create_visualizer_template(self, component_id: str, name: str, viz_type: str) -> VisualizerComponent:
        """Create a template visualizer component"""
        visualizer = VisualizerComponent(component_id)
        visualizer.name = name
        visualizer.interface.inputs = [
            ComponentParameter("data", "any", None, "Data to visualize")
        ]
        visualizer.interface.outputs = [
            ComponentParameter("visualization", "any", None, f"{viz_type} visualization")
        ]
        return visualizer
        
    def register_component(self, component_type: str, component_class: Callable[[str], ComponentBase]):
        """Register a new component type"""
        self.components[component_type] = component_class
        
    def create_component(self, component_type: str, component_id: str) -> Optional[ComponentBase]:
        """Create a component instance"""
        if component_type in self.components:
            return self.components[component_type](component_id)
        return None
        
    def get_available_components(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available components"""
        components_info = {}
        
        for comp_type, comp_class in self.components.items():
            # Create a temporary instance to get interface info
            temp_component = self.create_component(comp_type, "temp")
            if temp_component:
                components_info[comp_type] = {
                    "name": temp_component.name,
                    "category": temp_component.category.value,
                    "inputs": [p.__dict__ for p in temp_component.interface.inputs],
                    "outputs": [p.__dict__ for p in temp_component.interface.outputs],
                    "parameters": [p.__dict__ for p in temp_component.interface.parameters]
                }
                
        return components_info

class GadgetEngine:
    """Engine for executing gadget definitions"""
    
    def __init__(self):
        self.component_library = ComponentLibrary()
        self.running_gadgets: Dict[str, GadgetInstance] = {}
        self.execution_thread = None
        self.is_running = False
        self.execution_interval = 0.016  # ~60 FPS
        
    def create_gadget(self, gadget_definition: GadgetDefinition) -> str:
        """Create a new gadget instance"""
        instance_id = f"gadget_{len(self.running_gadgets)}_{int(time.time())}"
        
        # Create component instances
        component_instances = {}
        component_states = {}
        
        for comp_id, comp_def in gadget_definition.components.items():
            component = self.component_library.create_component(
                comp_def.name.lower().replace(' ', '_'), comp_id
            )
            if component:
                component.configuration = comp_def.configuration
                component_instances[comp_id] = component
                component_states[comp_id] = comp_def.state
                
        # Create gadget instance
        gadget_instance = GadgetInstance(
            instance_id=instance_id,
            gadget_definition=gadget_definition,
            component_states=component_states,
            connections=gadget_definition.connections
        )
        
        # Store instance
        self.running_gadgets[instance_id] = gadget_instance
        
        # Initialize components
        for component in component_instances.values():
            component.initialize()
            
        logging.info(f"Gadget {instance_id} created with {len(component_instances)} components")
        return instance_id
        
    def start_gadget(self, instance_id: str):
        """Start a gadget instance"""
        if instance_id in self.running_gadgets:
            self.running_gadgets[instance_id].is_running = True
            logging.info(f"Gadget {instance_id} started")
            
    def stop_gadget(self, instance_id: str):
        """Stop a gadget instance"""
        if instance_id in self.running_gadgets:
            self.running_gadgets[instance_id].is_running = False
            logging.info(f"Gadget {instance_id} stopped")
            
    def start_engine(self):
        """Start the gadget execution engine"""
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        logging.info("Gadget engine started")
        
    def stop_engine(self):
        """Stop the gadget execution engine"""
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=2.0)
            
        # Stop all gadgets
        for instance_id in self.running_gadgets:
            self.stop_gadget(instance_id)
            
        logging.info("Gadget engine stopped")
        
    def _execution_loop(self):
        """Main execution loop for running gadgets"""
        while self.is_running:
            start_time = time.time()
            
            # Execute all running gadgets
            for instance_id, gadget_instance in self.running_gadgets.items():
                if gadget_instance.is_running:
                    self._execute_gadget(gadget_instance)
                    
            # Maintain execution rate
            execution_time = time.time() - start_time
            sleep_time = max(0, self.execution_interval - execution_time)
            time.sleep(sleep_time)
            
    def _execute_gadget(self, gadget_instance: GadgetInstance):
        """Execute a single gadget instance"""
        try:
            # This would execute the gadget's data flow
            # For now, we'll just update performance metrics
            gadget_instance.performance_metrics = {
                "execution_count": gadget_instance.performance_metrics.get("execution_count", 0) + 1,
                "last_execution": time.time(),
                "average_execution_time": 0.001  # Placeholder
            }
            
        except Exception as e:
            logging.error(f"Error executing gadget {gadget_instance.instance_id}: {e}")
            
    def get_gadget_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a gadget instance"""
        if instance_id in self.running_gadgets:
            gadget = self.running_gadgets[instance_id]
            return {
                "instance_id": gadget.instance_id,
                "is_running": gadget.is_running,
                "component_count": len(gadget.gadget_definition.components),
                "connection_count": len(gadget.connections),
                "performance_metrics": gadget.performance_metrics
            }
        return None

class VisualGadgetBuilder:
    """Visual interface for building gadgets"""
    
    def __init__(self, gadget_engine: GadgetEngine):
        self.gadget_engine = gadget_engine
        self.current_gadget: Optional[GadgetDefinition] = None
        self.components: Dict[str, GadgetComponent] = {}
        self.connections: List[ComponentConnection] = []
        self.selected_component: Optional[str] = None
        self.dragging_component: Optional[str] = None
        self.connection_in_progress: Optional[Tuple[str, str]] = None
        
    def create_new_gadget(self, name: str, description: str, gadget_type: GadgetType) -> str:
        """Create a new gadget definition"""
        gadget_id = f"gadget_{int(time.time())}"
        
        self.current_gadget = GadgetDefinition(
            gadget_id=gadget_id,
            name=name,
            description=description,
            gadget_type=gadget_type,
            components={},
            connections=[],
            metadata={"created": time.time()}
        )
        
        self.components = {}
        self.connections = []
        
        logging.info(f"Created new gadget: {name} ({gadget_id})")
        return gadget_id
        
    def add_component(self, component_type: str, position: Tuple[float, float]) -> str:
        """Add a component to the current gadget"""
        if not self.current_gadget:
            raise ValueError("No gadget currently being edited")
            
        component_id = f"comp_{len(self.components)}_{int(time.time())}"
        
        # Create component definition
        component_info = self.gadget_engine.component_library.get_available_components().get(component_type)
        if not component_info:
            raise ValueError(f"Unknown component type: {component_type}")
            
        component = GadgetComponent(
            component_id=component_id,
            name=component_info["name"],
            category=ComponentCategory(component_info["category"]),
            interface=ComponentInterface(
                inputs=[ComponentParameter(**p) for p in component_info["inputs"]],
                outputs=[ComponentParameter(**p) for p in component_info["outputs"]],
                parameters=[ComponentParameter(**p) for p in component_info["parameters"]]
            ),
            position=position,
            configuration={}
        )
        
        self.components[component_id] = component
        self.current_gadget.components[component_id] = component
        
        logging.info(f"Added component {component_id} to gadget {self.current_gadget.gadget_id}")
        return component_id
        
    def remove_component(self, component_id: str):
        """Remove a component from the current gadget"""
        if component_id in self.components:
            # Remove associated connections
            self.connections = [conn for conn in self.connections 
                              if conn.source_component != component_id 
                              and conn.target_component != component_id]
            
            del self.components[component_id]
            if self.current_gadget:
                del self.current_gadget.components[component_id]
                
            logging.info(f"Removed component {component_id}")
            
    def connect_components(self, source_component: str, source_output: str,
                          target_component: str, target_input: str) -> str:
        """Connect two components"""
        if source_component not in self.components or target_component not in self.components:
            raise ValueError("Invalid component IDs")
            
        # Check if connection is valid
        source_comp = self.components[source_component]
        target_comp = self.components[target_component]
        
        # Find output parameter
        output_param = next((p for p in source_comp.interface.outputs if p.name == source_output), None)
        if not output_param:
            raise ValueError(f"Invalid output: {source_output}")
            
        # Find input parameter
        input_param = next((p for p in target_comp.interface.inputs if p.name == target_input), None)
        if not input_param:
            raise ValueError(f"Invalid input: {target_input}")
            
        # Create connection
        connection_id = f"conn_{len(self.connections)}_{int(time.time())}"
        connection = ComponentConnection(
            connection_id=connection_id,
            source_component=source_component,
            source_output=source_output,
            target_component=target_component,
            target_input=target_input,
            connection_type=ConnectionType.DATA_FLOW,
            data_type=output_param.data_type
        )
        
        self.connections.append(connection)
        if self.current_gadget:
            self.current_gadget.connections.append(connection)
            
        logging.info(f"Connected {source_component}.{source_output} -> {target_component}.{target_input}")
        return connection_id
        
    def disconnect_components(self, connection_id: str):
        """Disconnect components"""
        self.connections = [conn for conn in self.connections if conn.connection_id != connection_id]
        if self.current_gadget:
            self.current_gadget.connections = [conn for conn in self.current_gadget.connections 
                                             if conn.connection_id != connection_id]
            
    def set_component_parameter(self, component_id: str, parameter_name: str, value: Any):
        """Set a component parameter"""
        if component_id in self.components:
            self.components[component_id].configuration[parameter_name] = value
            if self.current_gadget:
                self.current_gadget.components[component_id].configuration[parameter_name] = value
                
    def move_component(self, component_id: str, new_position: Tuple[float, float]):
        """Move a component to a new position"""
        if component_id in self.components:
            self.components[component_id].position = new_position
            if self.current_gadget:
                self.current_gadget.components[component_id].position = new_position
                
    def save_gadget(self, file_path: str) -> bool:
        """Save gadget to file"""
        if not self.current_gadget:
            return False
            
        try:
            with open(file_path, 'w') as f:
                gadget_dict = {
                    'gadget_id': self.current_gadget.gadget_id,
                    'name': self.current_gadget.name,
                    'description': self.current_gadget.description,
                    'gadget_type': self.current_gadget.gadget_type.value,
                    'components': {
                        comp_id: {
                            'component_id': comp.component_id,
                            'name': comp.name,
                            'category': comp.category.value,
                            'position': comp.position,
                            'configuration': comp.configuration,
                            'state': comp.state
                        } for comp_id, comp in self.current_gadget.components.items()
                    },
                    'connections': [conn.__dict__ for conn in self.current_gadget.connections],
                    'metadata': self.current_gadget.metadata,
                    'version': self.current_gadget.version,
                    'author': self.current_gadget.author
                }
                json.dump(gadget_dict, f, indent=2)
                
            logging.info(f"Saved gadget to {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving gadget: {e}")
            return False
            
    def load_gadget(self, file_path: str) -> bool:
        """Load gadget from file"""
        try:
            with open(file_path, 'r') as f:
                gadget_dict = json.load(f)
                
            # Recreate components
            components = {}
            for comp_id, comp_data in gadget_dict['components'].items():
                component = GadgetComponent(
                    component_id=comp_data['component_id'],
                    name=comp_data['name'],
                    category=ComponentCategory(comp_data['category']),
                    interface=ComponentInterface(),  # Would need to recreate from component library
                    position=tuple(comp_data['position']),
                    configuration=comp_data['configuration'],
                    state=comp_data['state']
                )
                components[comp_id] = component
                
            # Recreate connections
            connections = []
            for conn_data in gadget_dict['connections']:
                connection = ComponentConnection(
                    connection_id=conn_data['connection_id'],
                    source_component=conn_data['source_component'],
                    source_output=conn_data['source_output'],
                    target_component=conn_data['target_component'],
                    target_input=conn_data['target_input'],
                    connection_type=ConnectionType(conn_data['connection_type']),
                    data_type=conn_data['data_type'],
                    properties=conn_data.get('properties', {})
                )
                connections.append(connection)
                
            # Create gadget definition
            self.current_gadget = GadgetDefinition(
                gadget_id=gadget_dict['gadget_id'],
                name=gadget_dict['name'],
                description=gadget_dict['description'],
                gadget_type=GadgetType(gadget_dict['gadget_type']),
                components=components,
                connections=connections,
                metadata=gadget_dict.get('metadata', {}),
                version=gadget_dict.get('version', '1.0'),
                author=gadget_dict.get('author', 'unknown')
            )
            
            self.components = components
            self.connections = connections
            
            logging.info(f"Loaded gadget from {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading gadget: {e}")
            return False

class AIGadgetDesigner:
    """AI-assisted gadget designer"""
    
    def __init__(self, gadget_engine: GadgetEngine):
        self.gadget_engine = gadget_engine
        self.design_patterns = self._load_design_patterns()
        
    def _load_design_patterns(self) -> Dict[str, Any]:
        """Load common gadget design patterns"""
        return {
            "data_pipeline": {
                "description": "Linear data processing pipeline",
                "components": ["sensor", "processor", "visualizer"],
                "connections": ["sensor.output -> processor.input", "processor.output -> visualizer.input"]
            },
            "feedback_controller": {
                "description": "Closed-loop feedback control system",
                "components": ["sensor", "controller", "actuator"],
                "connections": ["sensor.output -> controller.input", "controller.output -> actuator.input"]
            },
            "monitoring_system": {
                "description": "Multi-sensor monitoring system",
                "components": ["sensor", "sensor", "processor", "visualizer"],
                "connections": ["sensor1.output -> processor.input1", "sensor2.output -> processor.input2", 
                              "processor.output -> visualizer.input"]
            }
        }
        
    def generate_gadget_design(self, requirements: Dict[str, Any]) -> GadgetDefinition:
        """Generate gadget design based on requirements"""
        gadget_type = requirements.get("type", "custom")
        functionality = requirements.get("functionality", "")
        complexity = requirements.get("complexity", "medium")
        
        # Generate gadget ID and name
        gadget_id = f"ai_designed_{int(time.time())}"
        name = f"AI-Designed {functionality.title()} System"
        
        # Select appropriate design pattern
        design_pattern = self._select_design_pattern(functionality, complexity)
        
        # Create components based on pattern
        components = {}
        component_positions = self._generate_component_layout(len(design_pattern["components"]))
        
        for i, comp_type in enumerate(design_pattern["components"]):
            comp_id = f"comp_{i}"
            position = component_positions[i]
            
            # Get component info from library
            comp_info = self.gadget_engine.component_library.get_available_components().get(comp_type)
            if comp_info:
                component = GadgetComponent(
                    component_id=comp_id,
                    name=comp_info["name"],
                    category=ComponentCategory(comp_info["category"]),
                    interface=ComponentInterface(
                        inputs=[ComponentParameter(**p) for p in comp_info["inputs"]],
                        outputs=[ComponentParameter(**p) for p in comp_info["outputs"]],
                        parameters=[ComponentParameter(**p) for p in comp_info["parameters"]]
                    ),
                    position=position
                )
                components[comp_id] = component
                
        # Create connections based on pattern
        connections = []
        for conn_pattern in design_pattern["connections"]:
            # Parse connection pattern: "source.output -> target.input"
            parts = conn_pattern.split(" -> ")
            if len(parts) == 2:
                source_part = parts[0].split(".")
                target_part = parts[1].split(".")
                
                if len(source_part) == 2 and len(target_part) == 2:
                    source_comp = source_part[0]
                    source_output = source_part[1]
                    target_comp = target_part[0]
                    target_input = target_part[1]
                    
                    # Map pattern names to actual component IDs
                    actual_source = self._map_pattern_to_component(source_comp, components)
                    actual_target = self._map_pattern_to_component(target_comp, components)
                    
                    if actual_source and actual_target:
                        connection = ComponentConnection(
                            connection_id=f"conn_{len(connections)}",
                            source_component=actual_source,
                            source_output=source_output,
                            target_component=actual_target,
                            target_input=target_input,
                            connection_type=ConnectionType.DATA_FLOW,
                            data_type="any"
                        )
                        connections.append(connection)
                        
        # Create gadget definition
        gadget_def = GadgetDefinition(
            gadget_id=gadget_id,
            name=name,
            description=f"AI-generated gadget for {functionality}",
            gadget_type=GadgetType.CUSTOM,
            components=components,
            connections=connections,
            metadata={
                "ai_designed": True,
                "requirements": requirements,
                "generation_timestamp": time.time()
            }
        )
        
        return gadget_def
        
    def _select_design_pattern(self, functionality: str, complexity: str) -> Dict[str, Any]:
        """Select appropriate design pattern based on functionality"""
        functionality_lower = functionality.lower()
        
        if any(word in functionality_lower for word in ["monitor", "sense", "detect"]):
            return self.design_patterns["monitoring_system"]
        elif any(word in functionality_lower for word in ["control", "regulate", "maintain"]):
            return self.design_patterns["feedback_controller"]
        else:
            return self.design_patterns["data_pipeline"]
            
    def _generate_component_layout(self, num_components: int) -> List[Tuple[float, float]]:
        """Generate positions for components in a logical layout"""
        positions = []
        
        if num_components == 1:
            positions = [(0.5, 0.5)]
        elif num_components == 2:
            positions = [(0.3, 0.5), (0.7, 0.5)]
        elif num_components == 3:
            positions = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]
        else:
            # Grid layout for more components
            cols = int(np.ceil(np.sqrt(num_components)))
            rows = int(np.ceil(num_components / cols))
            
            for i in range(num_components):
                col = i % cols
                row = i // cols
                x = (col + 0.5) / cols
                y = (row + 0.5) / rows
                positions.append((x, y))
                
        return positions
        
    def _map_pattern_to_component(self, pattern_name: str, components: Dict[str, GadgetComponent]) -> Optional[str]:
        """Map pattern component name to actual component ID"""
        # Simple mapping - in practice, this would be more sophisticated
        for comp_id, component in components.items():
            if pattern_name in component.name.lower():
                return comp_id
            elif pattern_name.replace('1', '').replace('2', '') in component.name.lower():
                return comp_id
                
        # Return first component if no match found
        return list(components.keys())[0] if components else None

class AdvancedGadgetSystemCreator:
    """
    Advanced Gadget & System Creator
    Complete system for designing, building, and deploying custom gadgets and simulation systems
    """
    
    def __init__(self, simulation_system: Any = None):
        self.simulation_system = simulation_system
        self.gadget_engine = GadgetEngine()
        self.visual_builder = VisualGadgetBuilder(self.gadget_engine)
        self.ai_designer = AIGadgetDesigner(self.gadget_engine)
        
        # Gadget management
        self.available_gadgets: Dict[str, GadgetDefinition] = {}
        self.running_instances: Dict[str, str] = {}  # instance_id -> gadget_id
        
        # User interface state
        self.current_mode = "design"  # "design", "test", "deploy"
        self.selected_gadget: Optional[str] = None
        
        # Start gadget engine
        self.gadget_engine.start_engine()
        
        logging.info("Advanced Gadget System Creator initialized")
        
    def create_gadget_from_scratch(self, name: str, description: str, gadget_type: GadgetType) -> str:
        """Create a new gadget from scratch using visual builder"""
        return self.visual_builder.create_new_gadget(name, description, gadget_type)
        
    def create_gadget_with_ai(self, requirements: Dict[str, Any]) -> str:
        """Create a gadget using AI-assisted design"""
        gadget_def = self.ai_designer.generate_gadget_design(requirements)
        self.available_gadgets[gadget_def.gadget_id] = gadget_def
        
        # Load into visual builder for further editing
        self.visual_builder.current_gadget = gadget_def
        self.visual_builder.components = gadget_def.components
        self.visual_builder.connections = gadget_def.connections
        
        logging.info(f"AI-designed gadget created: {gadget_def.gadget_id}")
        return gadget_def.gadget_id
        
    def deploy_gadget(self, gadget_id: str) -> str:
        """Deploy a gadget as a running instance"""
        if gadget_id not in self.available_gadgets:
            raise ValueError(f"Gadget {gadget_id} not found")
            
        gadget_def = self.available_gadgets[gadget_id]
        instance_id = self.gadget_engine.create_gadget(gadget_def)
        self.gadget_engine.start_gadget(instance_id)
        
        self.running_instances[instance_id] = gadget_id
        
        logging.info(f"Gadget {gadget_id} deployed as instance {instance_id}")
        return instance_id
        
    def stop_gadget_instance(self, instance_id: str):
        """Stop a running gadget instance"""
        self.gadget_engine.stop_gadget(instance_id)
        if instance_id in self.running_instances:
            del self.running_instances[instance_id]
            
        logging.info(f"Gadget instance {instance_id} stopped")
        
    def save_gadget_design(self, file_path: str) -> bool:
        """Save current gadget design to file"""
        return self.visual_builder.save_gadget(file_path)
        
    def load_gadget_design(self, file_path: str) -> bool:
        """Load gadget design from file"""
        success = self.visual_builder.load_gadget(file_path)
        if success and self.visual_builder.current_gadget:
            gadget_def = self.visual_builder.current_gadget
            self.available_gadgets[gadget_def.gadget_id] = gadget_def
        return success
        
    def get_gadget_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a gadget instance"""
        return self.gadget_engine.get_gadget_status(instance_id)
        
    def get_available_components(self) -> Dict[str, Any]:
        """Get list of available components"""
        return self.gadget_engine.component_library.get_available_components()
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            "mode": self.current_mode,
            "available_gadgets": len(self.available_gadgets),
            "running_instances": len(self.running_instances),
            "gadget_engine_running": self.gadget_engine.is_running,
            "current_gadget": self.visual_builder.current_gadget.gadget_id if self.visual_builder.current_gadget else None
        }
        
    def cleanup(self):
        """Clean up the gadget system"""
        self.gadget_engine.stop_engine()
        logging.info("Advanced Gadget System Creator cleaned up")

# Example usage and demonstration
def demo_gadget_system():
    """Demonstrate the advanced gadget system creator"""
    creator = AdvancedGadgetSystemCreator()
    
    print("Advanced Gadget System Creator Demo")
    print("=" * 50)
    
    # Show available components
    components = creator.get_available_components()
    print(f"Available components: {len(components)}")
    for comp_type in list(components.keys())[:5]:  # Show first 5
        print(f"  - {comp_type}: {components[comp_type]['name']}")
    
    # Create a gadget using AI design
    print("\n1. Creating gadget with AI design...")
    requirements = {
        "type": "monitoring",
        "functionality": "particle system monitor",
        "complexity": "medium"
    }
    
    gadget_id = creator.create_gadget_with_ai(requirements)
    print(f"Created gadget: {gadget_id}")
    
    # Deploy the gadget
    print("\n2. Deploying gadget...")
    instance_id = creator.deploy_gadget(gadget_id)
    print(f"Deployed as instance: {instance_id}")
    
    # Check gadget status
    print("\n3. Checking gadget status...")
    status = creator.get_gadget_status(instance_id)
    if status:
        print(f"Instance {instance_id}:")
        print(f"  Running: {status['is_running']}")
        print(f"  Components: {status['component_count']}")
        print(f"  Connections: {status['connection_count']}")
    
    # Get system status
    print("\n4. System status:")
    system_status = creator.get_system_status()
    for key, value in system_status.items():
        print(f"  {key}: {value}")
    
    # Create a gadget from scratch
    print("\n5. Creating gadget from scratch...")
    scratch_id = creator.create_gadget_from_scratch(
        "Custom Particle Analyzer",
        "Custom gadget for advanced particle analysis",
        GadgetType.ANALYZER
    )
    print(f"Created gadget from scratch: {scratch_id}")
    
    # Save gadget design
    print("\n6. Saving gadget design...")
    success = creator.save_gadget_design("test_gadget.json")
    print(f"Save successful: {success}")
    
    # Cleanup
    print("\n7. Cleaning up...")
    creator.stop_gadget_instance(instance_id)
    creator.cleanup()
    
    return creator

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    gadget_creator = demo_gadget_system()