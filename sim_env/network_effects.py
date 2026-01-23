#!/usr/bin/env python3
"""
Network Effects & Multi-particle Interactions Module
Advanced particle networking, force propagation, and collective behaviors
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree, distance
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Callable
import random
import math
from enum import Enum
import json
from collections import defaultdict, deque

class InteractionType(Enum):
    """Types of particle interactions"""
    SPRING = "spring"
    REPULSION = "repulsion"
    ATTRACTION = "attraction"
    ALIGNMENT = "alignment"
    COHESION = "cohesion"
    SEPARATION = "separation"
    GRAVITATIONAL = "gravitational"
    ELECTROMAGNETIC = "electromagnetic"
    CHEMICAL = "chemical"
    NEURAL = "neural"
    SOCIAL = "social"
    INFORMATION = "information"

class NetworkTopology(Enum):
    """Types of network topologies"""
    RANDOM = "random"
    RING = "ring"
    STAR = "star"
    MESH = "mesh"
    SCALE_FREE = "scale_free"
    SMALL_WORLD = "small_world"
    SPATIAL = "spatial"
    HIERARCHICAL = "hierarchical"

@dataclass
class ParticleNode:
    """Node in the particle interaction network"""
    particle_id: int
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    charge: float
    radius: float
    node_type: str
    properties: Dict
    
    def __post_init__(self):
        self.neighbors: Set[int] = set()
        self.incoming_signals: List[Dict] = []
        self.outgoing_signals: List[Dict] = []
        self.energy_level: float = 1.0
        self.information: Dict = {}
        self.state_history: deque = deque(maxlen=100)

@dataclass
class InteractionEdge:
    """Edge in the particle interaction network"""
    source_id: int
    target_id: int
    interaction_type: InteractionType
    strength: float
    distance: float
    properties: Dict
    
    def __post_init__(self):
        self.age: float = 0.0
        self.strength_history: deque = deque(maxlen=50)
        self.last_activity: float = 0.0

class NetworkEffects:
    """Main class for managing particle networks and interactions"""
    
    def __init__(self, max_particles: int = 10000):
        # Network structure
        self.network_graph = nx.Graph()
        self.particle_nodes: Dict[int, ParticleNode] = {}
        self.interaction_edges: Dict[Tuple[int, int], InteractionEdge] = {}
        
        # Spatial indexing
        self.kd_tree: Optional[KDTree] = None
        self.positions_array: Optional[np.ndarray] = None
        
        # Network parameters
        self.max_interaction_distance = 5.0
        self.min_interaction_distance = 0.1
        self.max_neighbors = 8
        
        # Interaction strengths
        self.interaction_strengths = {
            InteractionType.SPRING: 1.0,
            InteractionType.REPULSION: 2.0,
            InteractionType.ATTRACTION: 0.5,
            InteractionType.ALIGNMENT: 0.8,
            InteractionType.COHESION: 0.6,
            InteractionType.SEPARATION: 1.2,
            InteractionType.GRAVITATIONAL: 0.1,
            InteractionType.ELECTROMAGNETIC: 0.3,
            InteractionType.CHEMICAL: 0.4,
            InteractionType.NEURAL: 0.7,
            InteractionType.SOCIAL: 0.9,
            InteractionType.INFORMATION: 0.2
        }
        
        # Network dynamics
        self.network_dynamics_enabled = True
        self.adaptive_connections = True
        self.signal_propagation = True
        
        # Signal system
        self.active_signals: List[Dict] = []
        self.signal_speed = 10.0
        self.signal_decay_rate = 0.1
        
        # Collective behavior parameters
        self.flocking_enabled = True
        self.swarm_intelligence = True
        self.emergence_detection = True
        
        # Performance optimization
        self.update_frequency = 1.0
        self.last_update_time = 0.0
        self.spatial_update_threshold = 0.1
        
        # Statistics
        self.network_statistics = {
            'average_degree': 0.0,
            'clustering_coefficient': 0.0,
            'average_path_length': 0.0,
            'network_density': 0.0,
            'connected_components': 0
        }
        
        print("Network Effects system initialized")

    def add_particle(self, particle_id: int, position: np.ndarray, 
                    velocity: np.ndarray, mass: float = 1.0, 
                    charge: float = 0.0, radius: float = 0.1,
                    node_type: str = "standard", properties: Dict = None):
        """Add a particle to the network"""
        if properties is None:
            properties = {}
            
        node = ParticleNode(
            particle_id=particle_id,
            position=np.array(position, dtype=np.float32),
            velocity=np.array(velocity, dtype=np.float32),
            mass=mass,
            charge=charge,
            radius=radius,
            node_type=node_type,
            properties=properties
        )
        
        self.particle_nodes[particle_id] = node
        self.network_graph.add_node(particle_id)
        
        # Update spatial indexing if needed
        self._update_spatial_indexing()

    def remove_particle(self, particle_id: int):
        """Remove a particle from the network"""
        if particle_id in self.particle_nodes:
            # Remove all connections
            neighbors = list(self.particle_nodes[particle_id].neighbors)
            for neighbor_id in neighbors:
                self.remove_connection(particle_id, neighbor_id)
            
            # Remove node
            del self.particle_nodes[particle_id]
            self.network_graph.remove_node(particle_id)
            
            self._update_spatial_indexing()

    def update_particle_position(self, particle_id: int, position: np.ndarray, velocity: np.ndarray):
        """Update particle position and velocity"""
        if particle_id in self.particle_nodes:
            self.particle_nodes[particle_id].position = np.array(position, dtype=np.float32)
            self.particle_nodes[particle_id].velocity = np.array(velocity, dtype=np.float32)
            
            # Record state history
            self.particle_nodes[particle_id].state_history.append({
                'position': position.copy(),
                'velocity': velocity.copy(),
                'timestamp': self.last_update_time
            })

    def create_connection(self, source_id: int, target_id: int, 
                         interaction_type: InteractionType, 
                         strength: float = None,
                         properties: Dict = None):
        """Create a connection between two particles"""
        if source_id not in self.particle_nodes or target_id not in self.particle_nodes:
            return False
            
        if properties is None:
            properties = {}
            
        if strength is None:
            strength = self.interaction_strengths.get(interaction_type, 1.0)
            
        # Calculate distance
        pos1 = self.particle_nodes[source_id].position
        pos2 = self.particle_nodes[target_id].position
        dist = np.linalg.norm(pos1 - pos2)
        
        # Create edge
        edge = InteractionEdge(
            source_id=source_id,
            target_id=target_id,
            interaction_type=interaction_type,
            strength=strength,
            distance=dist,
            properties=properties
        )
        
        edge_key = (min(source_id, target_id), max(source_id, target_id))
        self.interaction_edges[edge_key] = edge
        
        # Update network graph
        self.network_graph.add_edge(source_id, target_id, 
                                   interaction_type=interaction_type.value,
                                   strength=strength)
        
        # Update neighbor lists
        self.particle_nodes[source_id].neighbors.add(target_id)
        self.particle_nodes[target_id].neighbors.add(source_id)
        
        return True

    def remove_connection(self, source_id: int, target_id: int):
        """Remove connection between two particles"""
        edge_key = (min(source_id, target_id), max(source_id, target_id))
        
        if edge_key in self.interaction_edges:
            del self.interaction_edges[edge_key]
            
        # Update network graph
        if self.network_graph.has_edge(source_id, target_id):
            self.network_graph.remove_edge(source_id, target_id)
            
        # Update neighbor lists
        if source_id in self.particle_nodes:
            self.particle_nodes[source_id].neighbors.discard(target_id)
        if target_id in self.particle_nodes:
            self.particle_nodes[target_id].neighbors.discard(source_id)

    def update_network(self, current_time: float, dt: float):
        """Update the entire network state"""
        self.last_update_time = current_time
        
        # Update spatial relationships periodically
        if self._should_update_spatial_index():
            self._update_spatial_connections()
            
        # Update all interactions
        self._update_interactions(dt)
        
        # Handle signal propagation
        if self.signal_propagation:
            self._propagate_signals(dt)
            
        # Adaptive network restructuring
        if self.adaptive_connections:
            self._adaptive_restructuring()
            
        # Update network statistics
        self._update_network_statistics()
        
        # Detect emergent patterns
        if self.emergence_detection:
            self._detect_emergence()

    def _should_update_spatial_index(self) -> bool:
        """Check if spatial index needs updating"""
        if self.kd_tree is None:
            return True
            
        # Check if significant movement occurred
        total_movement = 0.0
        for node in self.particle_nodes.values():
            if len(node.state_history) >= 2:
                recent_pos = node.state_history[-1]['position']
                older_pos = node.state_history[0]['position']
                movement = np.linalg.norm(recent_pos - older_pos)
                total_movement += movement
                
        return total_movement > self.spatial_update_threshold * len(self.particle_nodes)

    def _update_spatial_indexing(self):
        """Update KD-tree for spatial queries"""
        if not self.particle_nodes:
            self.kd_tree = None
            return
            
        positions = []
        self.particle_indices = []
        
        for particle_id, node in self.particle_nodes.items():
            positions.append(node.position)
            self.particle_indices.append(particle_id)
            
        self.positions_array = np.array(positions)
        self.kd_tree = KDTree(self.positions_array)

    def _update_spatial_connections(self):
        """Update connections based on spatial proximity"""
        if self.kd_tree is None or len(self.particle_nodes) < 2:
            return
            
        # Find neighbors within interaction distance
        neighbor_pairs = self.kd_tree.query_pairs(self.max_interaction_distance)
        
        # Create connections for nearby particles
        for i, j in neighbor_pairs:
            particle_id_i = self.particle_indices[i]
            particle_id_j = self.particle_indices[j]
            
            # Skip if already connected
            edge_key = (min(particle_id_i, particle_id_j), max(particle_id_i, particle_id_j))
            if edge_key in self.interaction_edges:
                continue
                
            # Determine interaction type based on particle properties
            node_i = self.particle_nodes[particle_id_i]
            node_j = self.particle_nodes[particle_id_j]
            
            interaction_type = self._determine_interaction_type(node_i, node_j)
            
            # Create connection
            self.create_connection(particle_id_i, particle_id_j, interaction_type)

    def _determine_interaction_type(self, node1: ParticleNode, node2: ParticleNode) -> InteractionType:
        """Determine appropriate interaction type between two nodes"""
        # Charge-based interactions
        if node1.charge != 0 or node2.charge != 0:
            product = node1.charge * node2.charge
            if product < 0:
                return InteractionType.ATTRACTION
            else:
                return InteractionType.REPULSION
                
        # Mass-based interactions
        if node1.mass > 2.0 or node2.mass > 2.0:
            return InteractionType.GRAVITATIONAL
            
        # Type-based interactions
        if node1.node_type == "neural" or node2.node_type == "neural":
            return InteractionType.NEURAL
            
        if node1.node_type == "social" or node2.node_type == "social":
            return InteractionType.SOCIAL
            
        # Default to spring-like behavior
        return InteractionType.SPRING

    def _update_interactions(self, dt: float):
        """Update all active interactions"""
        for edge_key, edge in list(self.interaction_edges.items()):
            source_id, target_id = edge_key
            source = self.particle_nodes.get(source_id)
            target = self.particle_nodes.get(target_id)
            
            if not source or not target:
                self.remove_connection(source_id, target_id)
                continue
                
            # Update edge properties
            edge.age += dt
            current_distance = np.linalg.norm(source.position - target.position)
            edge.distance = current_distance
            edge.strength_history.append(edge.strength)
            
            # Apply interaction forces
            force = self._calculate_interaction_force(edge, source, target)
            
            # Apply force to particles (this would typically update particle velocities)
            if force is not None:
                self._apply_force_to_particles(source, target, force, dt)
                
            # Remove connections that are too far
            if current_distance > self.max_interaction_distance * 2:
                self.remove_connection(source_id, target_id)

    def _calculate_interaction_force(self, edge: InteractionEdge, 
                                   source: ParticleNode, 
                                   target: ParticleNode) -> Optional[np.ndarray]:
        """Calculate force based on interaction type"""
        direction = target.position - source.position
        distance = np.linalg.norm(direction)
        
        if distance < self.min_interaction_distance:
            return None
            
        direction_normalized = direction / distance
        
        if edge.interaction_type == InteractionType.SPRING:
            return self._spring_force(edge, distance, direction_normalized)
        elif edge.interaction_type == InteractionType.REPULSION:
            return self._repulsion_force(edge, distance, direction_normalized)
        elif edge.interaction_type == InteractionType.ATTRACTION:
            return self._attraction_force(edge, distance, direction_normalized)
        elif edge.interaction_type == InteractionType.ALIGNMENT:
            return self._alignment_force(edge, source, target)
        elif edge.interaction_type == InteractionType.COHESION:
            return self._cohesion_force(edge, source, target)
        elif edge.interaction_type == InteractionType.SEPARATION:
            return self._separation_force(edge, distance, direction_normalized)
        elif edge.interaction_type == InteractionType.GRAVITATIONAL:
            return self._gravitational_force(edge, source, target, distance, direction_normalized)
        elif edge.interaction_type == InteractionType.ELECTROMAGNETIC:
            return self._electromagnetic_force(edge, source, target, distance, direction_normalized)
        elif edge.interaction_type == InteractionType.NEURAL:
            return self._neural_force(edge, source, target)
        elif edge.interaction_type == InteractionType.SOCIAL:
            return self._social_force(edge, source, target)
        elif edge.interaction_type == InteractionType.INFORMATION:
            return self._information_force(edge, source, target)
        else:
            return self._chemical_force(edge, source, target, distance, direction_normalized)

    def _spring_force(self, edge: InteractionEdge, distance: float, direction: np.ndarray) -> np.ndarray:
        """Calculate spring force"""
        rest_length = edge.properties.get('rest_length', 1.0)
        k = edge.strength
        
        displacement = distance - rest_length
        force_magnitude = -k * displacement
        return force_magnitude * direction

    def _repulsion_force(self, edge: InteractionEdge, distance: float, direction: np.ndarray) -> np.ndarray:
        """Calculate repulsion force"""
        force_magnitude = edge.strength / (distance ** 2 + 0.1)
        return -force_magnitude * direction  # Repel

    def _attraction_force(self, edge: InteractionEdge, distance: float, direction: np.ndarray) -> np.ndarray:
        """Calculate attraction force"""
        force_magnitude = edge.strength * distance
        return force_magnitude * direction  # Attract

    def _alignment_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode) -> np.ndarray:
        """Calculate velocity alignment force"""
        velocity_diff = target.velocity - source.velocity
        return velocity_diff * edge.strength * 0.1

    def _cohesion_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode) -> np.ndarray:
        """Calculate cohesion force towards group center"""
        # This would typically use neighborhood information
        direction = target.position - source.position
        return direction * edge.strength * 0.05

    def _separation_force(self, edge: InteractionEdge, distance: float, direction: np.ndarray) -> np.ndarray:
        """Calculate separation force"""
        desired_separation = edge.properties.get('desired_separation', 1.0)
        if distance < desired_separation:
            force_magnitude = edge.strength * (desired_separation - distance)
            return -force_magnitude * direction
        return np.zeros(3)

    def _gravitational_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode,
                           distance: float, direction: np.ndarray) -> np.ndarray:
        """Calculate gravitational force"""
        G = 6.67430e-11 * 1e9  # Scaled for simulation
        force_magnitude = G * source.mass * target.mass / (distance ** 2 + 0.01)
        return force_magnitude * direction

    def _electromagnetic_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode,
                             distance: float, direction: np.ndarray) -> np.ndarray:
        """Calculate electromagnetic force"""
        k = 8.9875517873681764e9 * 1e6  # Scaled Coulomb constant
        force_magnitude = k * source.charge * target.charge / (distance ** 2 + 0.01)
        return force_magnitude * direction  # Attractive for opposite charges

    def _neural_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode) -> np.ndarray:
        """Calculate neural network-inspired force"""
        # Simple neural activation based on distance and velocities
        activation = np.tanh(np.dot(source.velocity, target.velocity) * 0.1)
        direction = target.position - source.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction_normalized = direction / distance
            return activation * edge.strength * direction_normalized
        return np.zeros(3)

    def _social_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode) -> np.ndarray:
        """Calculate social behavior force"""
        # Combine separation, alignment, and cohesion
        separation = self._separation_force(edge, 
                                          np.linalg.norm(target.position - source.position),
                                          (target.position - source.position))
        alignment = self._alignment_force(edge, source, target)
        cohesion = self._cohesion_force(edge, source, target)
        
        return (separation * 1.5 + alignment * 1.0 + cohesion * 1.0) * edge.strength

    def _information_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode) -> np.ndarray:
        """Calculate information-based force"""
        # Particles move towards information-rich areas
        info_gradient = target.information.get('value', 0) - source.information.get('value', 0)
        direction = target.position - source.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction_normalized = direction / distance
            return info_gradient * edge.strength * direction_normalized
        return np.zeros(3)

    def _chemical_force(self, edge: InteractionEdge, source: ParticleNode, target: ParticleNode,
                       distance: float, direction: np.ndarray) -> np.ndarray:
        """Calculate chemical gradient force"""
        # Simulate chemotaxis-like behavior
        concentration_diff = (target.properties.get('concentration', 0) - 
                            source.properties.get('concentration', 0))
        force_magnitude = edge.strength * concentration_diff / (distance + 0.1)
        return force_magnitude * direction

    def _apply_force_to_particles(self, source: ParticleNode, target: ParticleNode, 
                                force: np.ndarray, dt: float):
        """Apply calculated force to particles"""
        # This is a simplified implementation
        # In a full physics system, this would update velocities based on mass
        
        acceleration_source = force / (source.mass + 0.001)
        acceleration_target = -force / (target.mass + 0.001)
        
        source.velocity += acceleration_source * dt
        target.velocity += acceleration_target * dt

    def _propagate_signals(self, dt: float):
        """Propagate signals through the network"""
        new_signals = []
        
        for signal in self.active_signals:
            signal['age'] += dt
            signal['strength'] *= (1 - self.signal_decay_rate * dt)
            
            if signal['strength'] < 0.01:
                continue
                
            current_node_id = signal['current_node']
            if current_node_id not in self.particle_nodes:
                continue
                
            current_node = self.particle_nodes[current_node_id]
            
            # Process signal at current node
            self._process_signal_at_node(signal, current_node)
            
            # Propagate to neighbors
            for neighbor_id in current_node.neighbors:
                if neighbor_id not in signal['visited_nodes']:
                    new_signal = signal.copy()
                    new_signal['current_node'] = neighbor_id
                    new_signal['visited_nodes'].add(neighbor_id)
                    new_signal['distance_traveled'] += self._get_distance(current_node_id, neighbor_id)
                    new_signals.append(new_signal)
                    
                    # Update node information based on signal
                    neighbor_node = self.particle_nodes[neighbor_id]
                    neighbor_node.incoming_signals.append(new_signal)
        
        self.active_signals = new_signals

    def _process_signal_at_node(self, signal: Dict, node: ParticleNode):
        """Process a signal when it reaches a node"""
        signal_type = signal.get('type', 'information')
        
        if signal_type == 'information':
            node.information.update(signal.get('content', {}))
        elif signal_type == 'activation':
            node.energy_level += signal['strength'] * 0.1
        elif signal_type == 'inhibition':
            node.energy_level -= signal['strength'] * 0.1
            
        node.energy_level = np.clip(node.energy_level, 0.0, 2.0)

    def _get_distance(self, node1_id: int, node2_id: int) -> float:
        """Get distance between two nodes"""
        if node1_id in self.particle_nodes and node2_id in self.particle_nodes:
            pos1 = self.particle_nodes[node1_id].position
            pos2 = self.particle_nodes[node2_id].position
            return np.linalg.norm(pos1 - pos2)
        return 0.0

    def _adaptive_restructuring(self):
        """Adaptively restructure the network based on activity"""
        for edge_key, edge in list(self.interaction_edges.items()):
            # Strengthen frequently used connections
            if edge.last_activity > self.last_update_time - 1.0:  # Active in last second
                edge.strength = min(edge.strength * 1.01, 5.0)
            else:
                # Weaken unused connections
                edge.strength = max(edge.strength * 0.99, 0.1)
                
            # Remove very weak connections
            if edge.strength < 0.2:
                self.remove_connection(edge.source_id, edge.target_id)

    def _update_network_statistics(self):
        """Update network-wide statistics"""
        if len(self.network_graph) == 0:
            return
            
        try:
            self.network_statistics['average_degree'] = np.mean([d for _, d in self.network_graph.degree()])
            self.network_statistics['clustering_coefficient'] = nx.average_clustering(self.network_graph)
            self.network_statistics['network_density'] = nx.density(self.network_graph)
            self.network_statistics['connected_components'] = nx.number_connected_components(self.network_graph)
            
            # Only calculate average path length for connected graphs
            if nx.is_connected(self.network_graph):
                self.network_statistics['average_path_length'] = nx.average_shortest_path_length(self.network_graph)
            else:
                self.network_statistics['average_path_length'] = float('inf')
                
        except Exception as e:
            print(f"Error calculating network statistics: {e}")

    def _detect_emergence(self):
        """Detect emergent patterns in the network"""
        # This is a simplified emergence detection
        # In practice, this would use more sophisticated pattern recognition
        
        patterns = {}
        
        # Detect clusters
        if len(self.network_graph) > 10:
            clusters = list(nx.find_cliques(self.network_graph))
            large_clusters = [c for c in clusters if len(c) >= 3]
            if large_clusters:
                patterns['clusters'] = large_clusters
                
        # Detect chains/linear structures
        degrees = dict(self.network_graph.degree())
        chain_nodes = [node for node, degree in degrees.items() if degree == 2]
        if len(chain_nodes) >= 3:
            patterns['chains'] = chain_nodes
            
        # Detect hubs
        hub_nodes = [node for node, degree in degrees.items() if degree >= 5]
        if hub_nodes:
            patterns['hubs'] = hub_nodes
            
        return patterns

    def create_signal(self, source_id: int, signal_type: str, strength: float = 1.0, 
                     content: Dict = None) -> bool:
        """Create and broadcast a signal from a source node"""
        if source_id not in self.particle_nodes:
            return False
            
        if content is None:
            content = {}
            
        signal = {
            'source_node': source_id,
            'current_node': source_id,
            'type': signal_type,
            'strength': strength,
            'content': content,
            'age': 0.0,
            'distance_traveled': 0.0,
            'visited_nodes': {source_id}
        }
        
        self.active_signals.append(signal)
        return True

    def get_node_centrality(self, node_id: int) -> Dict[str, float]:
        """Calculate centrality measures for a node"""
        if node_id not in self.network_graph:
            return {}
            
        try:
            centrality = {
                'degree': nx.degree_centrality(self.network_graph).get(node_id, 0),
                'betweenness': nx.betweenness_centrality(self.network_graph).get(node_id, 0),
                'closeness': nx.closeness_centrality(self.network_graph).get(node_id, 0),
                'eigenvector': nx.eigenvector_centrality(self.network_graph, max_iter=1000).get(node_id, 0)
            }
            return centrality
        except:
            return {}

    def export_network_data(self, filename: str):
        """Export network data to JSON file"""
        export_data = {
            'nodes': [],
            'edges': [],
            'statistics': self.network_statistics,
            'timestamp': self.last_update_time
        }
        
        for node_id, node in self.particle_nodes.items():
            node_data = {
                'id': node_id,
                'position': node.position.tolist(),
                'velocity': node.velocity.tolist(),
                'mass': node.mass,
                'charge': node.charge,
                'type': node.node_type,
                'properties': node.properties,
                'energy_level': node.energy_level,
                'neighbors': list(node.neighbors)
            }
            export_data['nodes'].append(node_data)
            
        for edge_key, edge in self.interaction_edges.items():
            edge_data = {
                'source': edge.source_id,
                'target': edge.target_id,
                'type': edge.interaction_type.value,
                'strength': edge.strength,
                'distance': edge.distance,
                'age': edge.age,
                'properties': edge.properties
            }
            export_data['edges'].append(edge_data)
            
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Network data exported to {filename}")
        except Exception as e:
            print(f"Error exporting network data: {e}")

    def create_network_topology(self, topology: NetworkTopology, **kwargs):
        """Create a specific network topology"""
        num_particles = kwargs.get('num_particles', len(self.particle_nodes))
        
        if topology == NetworkTopology.RANDOM:
            self._create_random_topology(num_particles, kwargs.get('connection_probability', 0.1))
        elif topology == NetworkTopology.RING:
            self._create_ring_topology(num_particles)
        elif topology == NetworkTopology.STAR:
            self._create_star_topology(num_particles)
        elif topology == NetworkTopology.SCALE_FREE:
            self._create_scale_free_topology(num_particles)
        elif topology == NetworkTopology.SMALL_WORLD:
            self._create_small_world_topology(num_particles, 
                                            kwargs.get('k', 4),
                                            kwargs.get('p', 0.1))

    def _create_random_topology(self, num_particles: int, connection_probability: float):
        """Create random network topology"""
        # Clear existing connections
        for edge_key in list(self.interaction_edges.keys()):
            self.remove_connection(edge_key[0], edge_key[1])
            
        # Create random connections
        particle_ids = list(self.particle_nodes.keys())[:num_particles]
        
        for i, id1 in enumerate(particle_ids):
            for j, id2 in enumerate(particle_ids[i+1:], i+1):
                if random.random() < connection_probability:
                    interaction_type = random.choice(list(InteractionType))
                    self.create_connection(id1, id2, interaction_type)

    def _create_ring_topology(self, num_particles: int):
        """Create ring network topology"""
        particle_ids = list(self.particle_nodes.keys())[:num_particles]
        
        for i in range(len(particle_ids)):
            id1 = particle_ids[i]
            id2 = particle_ids[(i + 1) % len(particle_ids)]
            self.create_connection(id1, id2, InteractionType.SPRING)

    def _create_star_topology(self, num_particles: int):
        """Create star network topology"""
        if num_particles < 2:
            return
            
        particle_ids = list(self.particle_nodes.keys())[:num_particles]
        center_id = particle_ids[0]
        
        for peripheral_id in particle_ids[1:]:
            self.create_connection(center_id, peripheral_id, InteractionType.SPRING)

    def _create_scale_free_topology(self, num_particles: int):
        """Create scale-free network topology using Barabási-Albert model"""
        # Simplified implementation
        particle_ids = list(self.particle_nodes.keys())[:num_particles]
        
        # Start with connected pair
        if len(particle_ids) >= 2:
            self.create_connection(particle_ids[0], particle_ids[1], InteractionType.SPRING)
            
        # Add remaining nodes with preferential attachment
        for i in range(2, len(particle_ids)):
            new_id = particle_ids[i]
            
            # Calculate attachment probabilities based on degree
            degrees = []
            for existing_id in particle_ids[:i]:
                degree = len(self.particle_nodes[existing_id].neighbors)
                degrees.append(degree)
                
            total_degree = sum(degrees)
            if total_degree > 0:
                probabilities = [d / total_degree for d in degrees]
                
                # Connect to existing nodes based on probability
                targets = np.random.choice(particle_ids[:i], size=min(2, i), p=probabilities, replace=False)
                for target_id in targets:
                    self.create_connection(new_id, target_id, InteractionType.SPRING)

    def _create_small_world_topology(self, num_particles: int, k: int, p: float):
        """Create small-world network topology using Watts-Strogatz model"""
        particle_ids = list(self.particle_nodes.keys())[:num_particles]
        
        # Create ring lattice
        for i in range(len(particle_ids)):
            for j in range(1, k // 2 + 1):
                neighbor_idx = (i + j) % len(particle_ids)
                self.create_connection(particle_ids[i], particle_ids[neighbor_idx], InteractionType.SPRING)
                
        # Rewire edges with probability p
        for edge_key in list(self.interaction_edges.keys()):
            if random.random() < p:
                self.remove_connection(edge_key[0], edge_key[1])
                # Connect to random node
                available_nodes = [pid for pid in particle_ids 
                                 if pid != edge_key[0] and pid not in self.particle_nodes[edge_key[0]].neighbors]
                if available_nodes:
                    new_target = random.choice(available_nodes)
                    self.create_connection(edge_key[0], new_target, InteractionType.SPRING)

    def get_network_analysis(self) -> Dict:
        """Get comprehensive network analysis"""
        analysis = {
            'basic_statistics': self.network_statistics,
            'degree_distribution': self._get_degree_distribution(),
            'community_structure': self._detect_communities(),
            'centrality_measures': self._get_centrality_measures(),
            'resilience_metrics': self._calculate_resilience()
        }
        return analysis

    def _get_degree_distribution(self) -> Dict:
        """Get degree distribution of the network"""
        degrees = [d for _, d in self.network_graph.degree()]
        return {
            'min': min(degrees) if degrees else 0,
            'max': max(degrees) if degrees else 0,
            'mean': np.mean(degrees) if degrees else 0,
            'distribution': np.histogram(degrees, bins=10)[0].tolist()
        }

    def _detect_communities(self) -> Dict:
        """Detect communities in the network using Louvain method"""
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.network_graph)
            
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
                
            return {
                'number_of_communities': len(set(partition.values())),
                'modularity': community_louvain.modularity(partition, self.network_graph),
                'communities': communities
            }
        except ImportError:
            return {'error': 'python-louvain package not installed'}

    def _get_centrality_measures(self) -> Dict:
        """Calculate various centrality measures for the network"""
        try:
            return {
                'degree_centrality': nx.degree_centrality(self.network_graph),
                'betweenness_centrality': nx.betweenness_centrality(self.network_graph),
                'closeness_centrality': nx.closeness_centrality(self.network_graph),
                'eigenvector_centrality': nx.eigenvector_centrality(self.network_graph, max_iter=1000)
            }
        except:
            return {}

    def _calculate_resilience(self) -> Dict:
        """Calculate network resilience metrics"""
        try:
            # Calculate robustness to random failures and targeted attacks
            original_components = nx.number_connected_components(self.network_graph)
            
            # Random failure simulation
            nodes = list(self.network_graph.nodes())
            random.shuffle(nodes)
            failed_components = []
            
            for i in range(min(10, len(nodes))):
                test_graph = self.network_graph.copy()
                test_graph.remove_nodes_from(nodes[:i+1])
                failed_components.append(nx.number_connected_components(test_graph))
                
            return {
                'original_components': original_components,
                'failure_robustness': failed_components,
                'average_degree': self.network_statistics['average_degree'],
                'clustering': self.network_statistics['clustering_coefficient']
            }
        except:
            return {}

    def cleanup(self):
        """Clean up network resources"""
        self.particle_nodes.clear()
        self.interaction_edges.clear()
        self.network_graph.clear()
        self.active_signals.clear()
        print("Network Effects system cleaned up")

# Example usage and testing
if __name__ == "__main__":
    # Test the network effects system
    network = NetworkEffects()
    
    # Create some test particles
    for i in range(20):
        position = np.random.uniform(-5, 5, 3)
        velocity = np.random.uniform(-1, 1, 3)
        network.add_particle(i, position, velocity)
    
    # Create a scale-free network
    network.create_network_topology(NetworkTopology.SCALE_FREE, num_particles=20)
    
    # Run some updates
    for i in range(100):
        network.update_network(i * 0.016, 0.016)
    
    # Export network data
    network.export_network_data("test_network.json")
    
    # Get analysis
    analysis = network.get_network_analysis()
    print("Network analysis:", analysis)
    
    network.cleanup()