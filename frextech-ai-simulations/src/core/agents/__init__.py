"""
Agents module for autonomous navigation, physics simulation, and interaction.
"""

from .physics_simulator import PhysicsSimulator, RigidBody, SoftBody, FluidSimulator
from .navigation_agent import NavigationAgent, PathPlanner, ObstacleAvoidance
from .interaction_handler import (
    InteractionHandler,
    ObjectManipulator,
    SceneModifier,
    AgentController
)
from .sensors import (
    DepthSensor,
    RGBDSensor,
    LiDAR,
    ProximitySensor,
    MultiModalSensor
)
from .behaviors import (
    BehaviorTree,
    FiniteStateMachine,
    ReinforcementLearningAgent,
    ScriptedBehavior
)
from .utils import (
    agent_utils,
    physics_utils,
    navigation_utils,
    interaction_utils
)

__all__ = [
    # Core agents
    'PhysicsSimulator',
    'NavigationAgent',
    'InteractionHandler',
    
    # Physics
    'RigidBody',
    'SoftBody',
    'FluidSimulator',
    
    # Navigation
    'PathPlanner',
    'ObstacleAvoidance',
    
    # Interaction
    'ObjectManipulator',
    'SceneModifier',
    'AgentController',
    
    # Sensors
    'DepthSensor',
    'RGBDSensor',
    'LiDAR',
    'ProximitySensor',
    'MultiModalSensor',
    
    # Behaviors
    'BehaviorTree',
    'FiniteStateMachine',
    'ReinforcementLearningAgent',
    'ScriptedBehavior',
    
    # Utilities
    'agent_utils',
    'physics_utils',
    'navigation_utils',
    'interaction_utils'
]

# Version
__version__ = '1.0.0'

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Configuration
class AgentConfig:
    """Configuration for agent systems."""
    
    def __init__(self, **kwargs):
        # General
        self.agent_type = kwargs.get('agent_type', 'navigation')
        self.update_frequency = kwargs.get('update_frequency', 30.0)  # Hz
        self.max_velocity = kwargs.get('max_velocity', 2.0)  # m/s
        self.max_acceleration = kwargs.get('max_acceleration', 5.0)  # m/s²
        
        # Physics
        self.gravity = kwargs.get('gravity', 9.81)  # m/s²
        self.time_step = kwargs.get('time_step', 1.0/30.0)  # s
        self.solver_iterations = kwargs.get('solver_iterations', 10)
        self.enable_collision_detection = kwargs.get('enable_collision_detection', True)
        
        # Navigation
        self.pathfinding_algorithm = kwargs.get('pathfinding_algorithm', 'astar')
        self.avoidance_distance = kwargs.get('avoidance_distance', 0.5)  # m
        self.navigation_mesh_resolution = kwargs.get('navigation_mesh_resolution', 0.1)  # m
        
        # Interaction
        self.interaction_range = kwargs.get('interaction_range', 2.0)  # m
        self.max_interaction_force = kwargs.get('max_interaction_force', 100.0)  # N
        self.enable_tactile_feedback = kwargs.get('enable_tactile_feedback', False)
        
        # Sensors
        self.sensor_update_rate = kwargs.get('sensor_update_rate', 30.0)  # Hz
        self.sensor_noise_level = kwargs.get('sensor_noise_level', 0.01)
        self.fusion_method = kwargs.get('fusion_method', 'kalman')
        
        # Behavior
        self.behavior_update_interval = kwargs.get('behavior_update_interval', 0.1)  # s
        self.enable_learning = kwargs.get('enable_learning', False)
        self.memory_capacity = kwargs.get('memory_capacity', 10000)
        
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def update(self, **kwargs):
        """Update configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute: {key}")

# Factory functions
def create_agent(agent_type: str, config: dict = None, **kwargs):
    """
    Create an agent instance.
    
    Args:
        agent_type: Type of agent ('navigation', 'physics', 'interaction', 'composite')
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Agent instance
    """
    if config is None:
        config = {}
    
    config.update(kwargs)
    agent_config = AgentConfig(**config)
    
    if agent_type == 'navigation':
        return NavigationAgent(agent_config)
    elif agent_type == 'physics':
        return PhysicsSimulator(agent_config)
    elif agent_type == 'interaction':
        return InteractionHandler(agent_config)
    elif agent_type == 'composite':
        # Create a composite agent with all capabilities
        from .composite_agent import CompositeAgent
        return CompositeAgent(agent_config)
    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Must be one of: ['navigation', 'physics', 'interaction', 'composite']"
        )

def create_sensor(sensor_type: str, config: dict = None, **kwargs):
    """
    Create a sensor instance.
    
    Args:
        sensor_type: Type of sensor ('depth', 'rgbd', 'lidar', 'proximity', 'multimodal')
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Sensor instance
    """
    if config is None:
        config = {}
    
    config.update(kwargs)
    
    if sensor_type == 'depth':
        return DepthSensor(**config)
    elif sensor_type == 'rgbd':
        return RGBDSensor(**config)
    elif sensor_type == 'lidar':
        return LiDAR(**config)
    elif sensor_type == 'proximity':
        return ProximitySensor(**config)
    elif sensor_type == 'multimodal':
        return MultiModalSensor(**config)
    else:
        raise ValueError(
            f"Unknown sensor type: {sensor_type}. "
            f"Must be one of: ['depth', 'rgbd', 'lidar', 'proximity', 'multimodal']"
        )

def create_behavior(behavior_type: str, config: dict = None, **kwargs):
    """
    Create a behavior system.
    
    Args:
        behavior_type: Type of behavior ('tree', 'fsm', 'rl', 'scripted')
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Behavior system instance
    """
    if config is None:
        config = {}
    
    config.update(kwargs)
    
    if behavior_type == 'tree':
        return BehaviorTree(**config)
    elif behavior_type == 'fsm':
        return FiniteStateMachine(**config)
    elif behavior_type == 'rl':
        return ReinforcementLearningAgent(**config)
    elif behavior_type == 'scripted':
        return ScriptedBehavior(**config)
    else:
        raise ValueError(
            f"Unknown behavior type: {behavior_type}. "
            f"Must be one of: ['tree', 'fsm', 'rl', 'scripted']"
        )

# Global state management
class AgentManager:
    """Manages multiple agents in a scene."""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.agents = {}
        self.sensors = {}
        self.behaviors = {}
        self._is_running = False
        
    def add_agent(self, agent_id: str, agent_type: str, **kwargs):
        """Add an agent to the manager."""
        agent = create_agent(agent_type, **kwargs)
        self.agents[agent_id] = agent
        
        # Create default sensor and behavior
        sensor_id = f"{agent_id}_sensor"
        behavior_id = f"{agent_id}_behavior"
        
        self.sensors[sensor_id] = create_sensor('multimodal')
        self.behaviors[behavior_id] = create_behavior('fsm')
        
        return agent
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the manager."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
            # Remove associated sensors and behaviors
            sensor_id = f"{agent_id}_sensor"
            behavior_id = f"{agent_id}_behavior"
            
            if sensor_id in self.sensors:
                del self.sensors[sensor_id]
            if behavior_id in self.behaviors:
                del self.behaviors[behavior_id]
    
    def update(self, dt: float):
        """Update all agents."""
        if not self._is_running:
            return
        
        # Update sensors
        for sensor in self.sensors.values():
            sensor.update(dt)
        
        # Update behaviors
        for behavior_id, behavior in self.behaviors.items():
            agent_id = behavior_id.replace('_behavior', '')
            if agent_id in self.agents:
                sensor_data = self.get_sensor_data(agent_id)
                action = behavior.update(sensor_data)
                self.agents[agent_id].apply_action(action)
        
        # Update agents
        for agent in self.agents.values():
            agent.update(dt)
    
    def get_sensor_data(self, agent_id: str):
        """Get sensor data for an agent."""
        sensor_id = f"{agent_id}_sensor"
        if sensor_id in self.sensors:
            return self.sensors[sensor_id].get_data()
        return {}
    
    def start(self):
        """Start all agents."""
        self._is_running = True
        for agent in self.agents.values():
            if hasattr(agent, 'start'):
                agent.start()
    
    def stop(self):
        """Stop all agents."""
        self._is_running = False
        for agent in self.agents.values():
            if hasattr(agent, 'stop'):
                agent.stop()
    
    def get_agent_positions(self):
        """Get positions of all agents."""
        positions = {}
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_position'):
                positions[agent_id] = agent.get_position()
        return positions
    
    def check_collisions(self):
        """Check for collisions between agents."""
        collisions = []
        agent_ids = list(self.agents.keys())
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                agent1 = self.agents[agent_ids[i]]
                agent2 = self.agents[agent_ids[j]]
                
                if hasattr(agent1, 'check_collision') and hasattr(agent2, 'check_collision'):
                    if agent1.check_collision(agent2):
                        collisions.append((agent_ids[i], agent_ids[j]))
        
        return collisions

# Global agent manager instance
_agent_manager = None

def get_agent_manager(config: AgentConfig = None):
    """Get or create the global agent manager."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager(config)
    return _agent_manager

def reset_agent_manager():
    """Reset the global agent manager."""
    global _agent_manager
    _agent_manager = None