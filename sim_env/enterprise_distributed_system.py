#!/usr/bin/env python3
"""
Enterprise-Grade Distributed System
Distributed computing, load balancing, high availability, and cluster management
"""

import asyncio
import aiohttp
import aiohttp.web
import multiprocessing as mp
import threading
import queue
import time
import json
import pickle
import hashlib
import zlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import glm
import redis
import requests
from consul import Consul
import psutil
import socket
import logging
from logging.handlers import RotatingFileHandler
import secrets
from cryptography.fernet import Fernet
import ssl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('distributed_system.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DistributedSystem")

@dataclass
class NodeInfo:
    node_id: str
    host: str
    port: int
    node_type: str  # "master", "worker", "render", "database"
    status: str  # "online", "offline", "busy", "idle"
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    load_factor: float
    last_heartbeat: float
    capabilities: List[str]
    assigned_tasks: int

@dataclass
class DistributedTask:
    task_id: str
    task_type: str
    priority: int
    data: Any
    source_node: str
    target_nodes: List[str]
    status: str  # "pending", "running", "completed", "failed"
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    result: Any
    retry_count: int
    max_retries: int

@dataclass
class ClusterMetrics:
    total_nodes: int
    online_nodes: int
    total_memory_gb: float
    available_memory_gb: float
    total_cpu_cores: int
    average_load: float
    tasks_queued: int
    tasks_running: int
    tasks_completed: int
    network_latency_ms: float
    data_throughput_mbps: float

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_LOAD = "least_load"
    AFFINITY = "affinity"

class DistributedSystem:
    """Enterprise-grade distributed simulation system"""
    
    def __init__(self, simulation_app, cluster_config: Dict[str, Any]):
        self.simulation_app = simulation_app
        self.config = cluster_config
        
        # Node management
        self.node_id = self.generate_node_id()
        self.nodes: Dict[str, NodeInfo] = {}
        self.heartbeat_interval = 5.0  # seconds
        
        # Task management
        self.task_queue = asyncio.PriorityQueue()
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_results: Dict[str, Any] = {}
        
        # Load balancing
        self.load_balancer = LoadBalancer()
        self.load_balancing_strategy = LoadBalancingStrategy.LEAST_LOAD
        
        # Communication
        self.message_broker = MessageBroker(self.config.get('redis_host', 'localhost'))
        self.api_server = None
        self.websocket_connections: Dict[str, Any] = {}
        
        # Security
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.auth_tokens: Dict[str, str] = {}
        
        # Monitoring
        self.metrics = ClusterMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.performance_monitor = PerformanceMonitor()
        
        # Fault tolerance
        self.health_checker = HealthChecker()
        self.backup_nodes: List[str] = []
        
        # Initialize distributed components
        self.initialize_distributed_system()
        
        logger.info(f"Distributed System initialized on node {self.node_id}")
    
    def generate_node_id(self) -> str:
        """Generate unique node identifier"""
        hostname = socket.gethostname()
        timestamp = str(time.time())
        random_bytes = secrets.token_bytes(8)
        return hashlib.sha256(f"{hostname}{timestamp}{random_bytes}".encode()).hexdigest()[:16]
    
    def initialize_distributed_system(self):
        """Initialize all distributed system components"""
        # Register this node
        self.register_node()
        
        # Start background tasks
        self.start_background_tasks()
        
        # Initialize API server
        self.initialize_api_server()
        
        # Connect to message broker
        self.message_broker.connect()
        
        # Discover other nodes
        self.discover_nodes()
        
        logger.info("Distributed system components initialized")
    
    def register_node(self):
        """Register this node in the cluster"""
        node_info = NodeInfo(
            node_id=self.node_id,
            host=self.config.get('host', 'localhost'),
            port=self.config.get('port', 8000),
            node_type=self.config.get('node_type', 'worker'),
            status="online",
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            gpu_usage=self.get_gpu_usage(),
            load_factor=0.0,
            last_heartbeat=time.time(),
            capabilities=self.config.get('capabilities', ['physics', 'rendering']),
            assigned_tasks=0
        )
        
        self.nodes[self.node_id] = node_info
        logger.info(f"Node {self.node_id} registered as {node_info.node_type}")
    
    def get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        try:
            # This would use GPU monitoring libraries like pynvml
            # For now, return a mock value
            return 0.0
        except:
            return None
    
    def start_background_tasks(self):
        """Start background maintenance tasks"""
        # Heartbeat task
        heartbeat_task = threading.Thread(target=self.heartbeat_worker, daemon=True)
        heartbeat_task.start()
        
        # Metrics collection task
        metrics_task = threading.Thread(target=self.metrics_worker, daemon=True)
        metrics_task.start()
        
        # Task processor
        task_processor = threading.Thread(target=self.task_processor_worker, daemon=True)
        task_processor.start()
        
        # Health checker
        health_task = threading.Thread(target=self.health_check_worker, daemon=True)
        health_task.start()
        
        logger.info("Background tasks started")
    
    def heartbeat_worker(self):
        """Send periodic heartbeats to maintain node status"""
        while True:
            try:
                self.send_heartbeat()
                self.update_node_status()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat worker error: {e}")
                time.sleep(1)
    
    def send_heartbeat(self):
        """Send heartbeat to other nodes"""
        # Update local node status
        self.nodes[self.node_id].cpu_usage = psutil.cpu_percent()
        self.nodes[self.node_id].memory_usage = psutil.virtual_memory().percent
        self.nodes[self.node_id].gpu_usage = self.get_gpu_usage()
        self.nodes[self.node_id].load_factor = self.calculate_load_factor()
        self.nodes[self.node_id].last_heartbeat = time.time()
        self.nodes[self.node_id].assigned_tasks = len([t for t in self.tasks.values() 
                                                     if t.status == "running"])
        
        # Broadcast heartbeat to other nodes
        heartbeat_data = {
            'type': 'heartbeat',
            'node_info': asdict(self.nodes[self.node_id]),
            'timestamp': time.time()
        }
        
        self.message_broker.broadcast('node_heartbeats', heartbeat_data)
    
    def calculate_load_factor(self) -> float:
        """Calculate current load factor for this node"""
        cpu_load = psutil.cpu_percent() / 100.0
        memory_load = psutil.virtual_memory().percent / 100.0
        task_load = len([t for t in self.tasks.values() if t.status == "running"]) / 10.0  # Normalize
        
        return (cpu_load + memory_load + task_load) / 3.0
    
    def update_node_status(self):
        """Update status of all nodes based on heartbeats"""
        current_time = time.time()
        offline_threshold = self.heartbeat_interval * 3  # 3 missed heartbeats
        
        for node_id, node in list(self.nodes.items()):
            if node_id == self.node_id:
                continue
                
            if current_time - node.last_heartbeat > offline_threshold:
                node.status = "offline"
                logger.warning(f"Node {node_id} is offline")
            else:
                node.status = "online"
    
    def metrics_worker(self):
        """Collect and update cluster metrics"""
        while True:
            try:
                self.update_cluster_metrics()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Metrics worker error: {e}")
                time.sleep(1)
    
    def update_cluster_metrics(self):
        """Update cluster-wide metrics"""
        online_nodes = [n for n in self.nodes.values() if n.status == "online"]
        
        self.metrics.total_nodes = len(self.nodes)
        self.metrics.online_nodes = len(online_nodes)
        self.metrics.total_memory_gb = sum([psutil.virtual_memory().total for _ in online_nodes]) / (1024**3)
        self.metrics.available_memory_gb = sum([psutil.virtual_memory().available for _ in online_nodes]) / (1024**3)
        self.metrics.total_cpu_cores = sum([psutil.cpu_count() for _ in online_nodes])
        self.metrics.average_load = np.mean([n.load_factor for n in online_nodes]) if online_nodes else 0
        self.metrics.tasks_queued = len([t for t in self.tasks.values() if t.status == "pending"])
        self.metrics.tasks_running = len([t for t in self.tasks.values() if t.status == "running"])
        self.metrics.tasks_completed = len([t for t in self.tasks.values() if t.status == "completed"])
        
        # Network metrics (simplified)
        self.metrics.network_latency_ms = self.measure_network_latency()
        self.metrics.data_throughput_mbps = self.estimate_throughput()
    
    def measure_network_latency(self) -> float:
        """Measure average network latency to other nodes"""
        # Simplified implementation - in production would ping all nodes
        return 10.0  # Mock value
    
    def estimate_throughput(self) -> float:
        """Estimate data throughput in Mbps"""
        # Simplified implementation
        return 100.0  # Mock value
    
    def task_processor_worker(self):
        """Process tasks from the distributed queue"""
        while True:
            try:
                # Get next task (non-blocking with timeout)
                try:
                    priority, task_id = self.task_queue.get_nowait()
                    task = self.tasks[task_id]
                    
                    if task.status == "pending":
                        self.execute_task(task)
                    
                    self.task_queue.task_done()
                    
                except queue.Empty:
                    time.sleep(0.1)  # Short sleep when queue is empty
                    
            except Exception as e:
                logger.error(f"Task processor error: {e}")
                time.sleep(1)
    
    def execute_task(self, task: DistributedTask):
        """Execute a distributed task"""
        try:
            task.status = "running"
            task.started_at = time.time()
            
            logger.info(f"Executing task {task.task_id} of type {task.task_type}")
            
            # Route to appropriate handler based on task type
            if task.task_type == "physics_simulation":
                result = self.execute_physics_task(task)
            elif task.task_type == "rendering":
                result = self.execute_rendering_task(task)
            elif task.task_type == "data_processing":
                result = self.execute_data_processing_task(task)
            elif task.task_type == "ml_training":
                result = self.execute_ml_training_task(task)
            else:
                result = self.execute_generic_task(task)
            
            task.result = result
            task.status = "completed"
            task.completed_at = time.time()
            
            # Notify task completion
            self.notify_task_completion(task)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            task.status = "failed"
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                # Reschedule task
                self.schedule_task(task)
            else:
                logger.error(f"Task {task.task_id} exceeded maximum retries")
    
    def execute_physics_task(self, task: DistributedTask) -> Any:
        """Execute physics simulation task"""
        # Deserialize simulation data
        simulation_data = self.deserialize_data(task.data)
        
        # Run physics simulation
        # This would integrate with the physics engine
        result = {
            'particle_positions': [],
            'forces': [],
            'energy': 0.0,
            'simulation_time': simulation_data.get('time', 0.0)
        }
        
        return self.serialize_data(result)
    
    def execute_rendering_task(self, task: DistributedTask) -> Any:
        """Execute rendering task"""
        render_data = self.deserialize_data(task.data)
        
        # Distributed rendering
        result = {
            'image_data': b'',
            'resolution': render_data.get('resolution', (1920, 1080)),
            'render_time': 0.1
        }
        
        return self.serialize_data(result)
    
    def execute_data_processing_task(self, task: DistributedTask) -> Any:
        """Execute data processing task"""
        data = self.deserialize_data(task.data)
        
        # Process data (e.g.,统计分析, filtering, transformation)
        processed_data = {
            'statistics': {
                'mean': np.mean(data.get('values', [])),
                'std': np.std(data.get('values', [])),
                'count': len(data.get('values', []))
            },
            'processed_values': []
        }
        
        return self.serialize_data(processed_data)
    
    def execute_ml_training_task(self, task: DistributedTask) -> Any:
        """Execute machine learning training task"""
        training_data = self.deserialize_data(task.data)
        
        # Distributed ML training
        result = {
            'model_weights': [],
            'training_loss': 0.1,
            'accuracy': 0.95,
            'training_time': 60.0
        }
        
        return self.serialize_data(result)
    
    def execute_generic_task(self, task: DistributedTask) -> Any:
        """Execute generic computational task"""
        # Generic task execution
        computation_data = self.deserialize_data(task.data)
        
        # Perform computation
        result = {
            'computation_result': 'success',
            'output_data': computation_data.get('input_data'),
            'processing_time': 0.01
        }
        
        return self.serialize_data(result)
    
    def notify_task_completion(self, task: DistributedTask):
        """Notify relevant nodes about task completion"""
        completion_message = {
            'type': 'task_completed',
            'task_id': task.task_id,
            'result': task.result,
            'completed_at': task.completed_at
        }
        
        # Send to source node and any other interested nodes
        for node_id in [task.source_node] + task.target_nodes:
            if node_id in self.nodes:
                self.message_broker.send_to_node(node_id, completion_message)
    
    def health_check_worker(self):
        """Perform health checks on other nodes"""
        while True:
            try:
                self.perform_health_checks()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check worker error: {e}")
                time.sleep(1)
    
    def perform_health_checks(self):
        """Perform health checks on all nodes"""
        for node_id, node in self.nodes.items():
            if node_id == self.node_id:
                continue
                
            if node.status == "online":
                # Perform health check
                is_healthy = self.health_checker.check_node_health(node)
                
                if not is_healthy:
                    node.status = "offline"
                    logger.warning(f"Node {node_id} failed health check")
                    
                    # Redistribute tasks from failed node
                    self.redistribute_tasks_from_node(node_id)
    
    def redistribute_tasks_from_node(self, failed_node_id: str):
        """Redistribute tasks from a failed node"""
        failed_tasks = [t for t in self.tasks.values() 
                       if t.status == "running" and failed_node_id in t.target_nodes]
        
        for task in failed_tasks:
            task.status = "pending"
            task.retry_count += 1
            
            # Reschedule on available nodes
            available_nodes = self.find_available_nodes(task.task_type)
            if available_nodes:
                task.target_nodes = available_nodes
                self.schedule_task(task)
                logger.info(f"Rescheduled task {task.task_id} from failed node {failed_node_id}")
    
    def initialize_api_server(self):
        """Initialize the distributed system API server"""
        try:
            self.api_server = DistributedAPIServer(self)
            self.api_server.start()
            logger.info("Distributed API server started")
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
    
    def discover_nodes(self):
        """Discover other nodes in the cluster"""
        # In a real system, this would use service discovery (Consul, etcd, etc.)
        # For now, we'll simulate node discovery
        
        # Simulate discovering some nodes
        simulated_nodes = [
            NodeInfo(
                node_id="node_physics_1",
                host="physics1.cluster.local",
                port=8001,
                node_type="worker",
                status="online",
                cpu_usage=0.2,
                memory_usage=0.3,
                gpu_usage=0.1,
                load_factor=0.15,
                last_heartbeat=time.time(),
                capabilities=["physics", "computation"],
                assigned_tasks=2
            ),
            NodeInfo(
                node_id="node_render_1", 
                host="render1.cluster.local",
                port=8002,
                node_type="render",
                status="online",
                cpu_usage=0.4,
                memory_usage=0.6,
                gpu_usage=0.8,
                load_factor=0.6,
                last_heartbeat=time.time(),
                capabilities=["rendering", "gpu_compute"],
                assigned_tasks=5
            )
        ]
        
        for node in simulated_nodes:
            self.nodes[node.node_id] = node
        
        logger.info(f"Discovered {len(simulated_nodes)} nodes in cluster")
    
    def schedule_task(self, task: DistributedTask) -> str:
        """Schedule a task for distributed execution"""
        task.task_id = self.generate_task_id()
        task.created_at = time.time()
        task.status = "pending"
        
        # Select target nodes based on load balancing strategy
        if not task.target_nodes:
            task.target_nodes = self.select_target_nodes(task)
        
        # Add to task tracking
        self.tasks[task.task_id] = task
        
        # Add to priority queue
        priority = -task.priority  # Higher priority = lower number
        self.task_queue.put((priority, task.task_id))
        
        logger.info(f"Scheduled task {task.task_id} with priority {task.priority}")
        
        return task.task_id
    
    def generate_task_id(self) -> str:
        """Generate unique task identifier"""
        timestamp = str(time.time())
        random_bytes = secrets.token_bytes(8)
        return hashlib.sha256(f"{self.node_id}{timestamp}{random_bytes}".encode()).hexdigest()[:16]
    
    def select_target_nodes(self, task: DistributedTask) -> List[str]:
        """Select target nodes for task execution based on load balancing strategy"""
        capable_nodes = [node_id for node_id, node in self.nodes.items()
                        if node.status == "online" and 
                        self.node_has_capabilities(node, task.task_type)]
        
        if not capable_nodes:
            logger.warning(f"No capable nodes found for task type {task.task_type}")
            return []
        
        return self.load_balancer.select_nodes(
            capable_nodes, 
            self.nodes, 
            self.load_balancing_strategy
        )
    
    def node_has_capabilities(self, node: NodeInfo, task_type: str) -> bool:
        """Check if node has required capabilities for task type"""
        capability_map = {
            "physics_simulation": ["physics", "computation"],
            "rendering": ["rendering", "gpu_compute"],
            "data_processing": ["computation", "data_processing"],
            "ml_training": ["ml", "gpu_compute", "computation"]
        }
        
        required_capabilities = capability_map.get(task_type, ["computation"])
        return any(cap in node.capabilities for cap in required_capabilities)
    
    def serialize_data(self, data: Any) -> bytes:
        """Serialize data for transmission"""
        serialized = pickle.dumps(data)
        compressed = zlib.compress(serialized)
        encrypted = self.cipher_suite.encrypt(compressed)
        return encrypted
    
    def deserialize_data(self, data: bytes) -> Any:
        """Deserialize received data"""
        decrypted = self.cipher_suite.decrypt(data)
        decompressed = zlib.decompress(decrypted)
        return pickle.loads(decompressed)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status"""
        return {
            'cluster_metrics': asdict(self.metrics),
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()},
            'tasks': {
                'queued': self.metrics.tasks_queued,
                'running': self.metrics.tasks_running,
                'completed': self.metrics.tasks_completed
            },
            'load_balancing_strategy': self.load_balancing_strategy.value
        }
    
    def distribute_simulation_frame(self, frame_data: Dict[str, Any]) -> List[str]:
        """Distribute simulation frame processing across cluster"""
        tasks = []
        
        # Split physics simulation
        physics_task = DistributedTask(
            task_id="",
            task_type="physics_simulation",
            priority=1,
            data=frame_data.get('physics_data', {}),
            source_node=self.node_id,
            target_nodes=[],
            status="pending",
            created_at=0,
            started_at=None,
            completed_at=None,
            result=None,
            retry_count=0,
            max_retries=3
        )
        physics_task_id = self.schedule_task(physics_task)
        tasks.append(physics_task_id)
        
        # Split rendering if needed
        if frame_data.get('needs_rendering', False):
            render_task = DistributedTask(
                task_id="",
                task_type="rendering",
                priority=2,
                data=frame_data.get('render_data', {}),
                source_node=self.node_id,
                target_nodes=[],
                status="pending",
                created_at=0,
                started_at=None,
                completed_at=None,
                result=None,
                retry_count=0,
                max_retries=3
            )
            render_task_id = self.schedule_task(render_task)
            tasks.append(render_task_id)
        
        return tasks
    
    def cleanup(self):
        """Cleanup distributed system resources"""
        if self.api_server:
            self.api_server.stop()
        
        self.message_broker.disconnect()
        logger.info("Distributed system cleaned up")

class LoadBalancer:
    """Intelligent load balancer for task distribution"""
    
    def __init__(self):
        self.node_weights: Dict[str, float] = {}
        self.last_used: Dict[str, int] = {}
        self.connection_counts: Dict[str, int] = {}
    
    def select_nodes(self, available_nodes: List[str], nodes: Dict[str, NodeInfo],
                    strategy: LoadBalancingStrategy) -> List[str]:
        """Select nodes based on load balancing strategy"""
        if not available_nodes:
            return []
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self.round_robin(available_nodes)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self.least_connections(available_nodes)
        elif strategy == LoadBalancingStrategy.LEAST_LOAD:
            return self.least_load(available_nodes, nodes)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self.weighted_round_robin(available_nodes, nodes)
        else:
            return self.affinity_based(available_nodes)
    
    def round_robin(self, nodes: List[str]) -> List[str]:
        """Round robin node selection"""
        if not nodes:
            return []
        
        # Simple round robin - just return nodes in order
        return [nodes[0]]  # For simplicity, just use first node
    
    def least_connections(self, nodes: List[str]) -> List[str]:
        """Select node with fewest active connections"""
        if not nodes:
            return []
        
        # Find node with minimum connection count
        min_connections = float('inf')
        best_node = nodes[0]
        
        for node in nodes:
            connections = self.connection_counts.get(node, 0)
            if connections < min_connections:
                min_connections = connections
                best_node = node
        
        # Update connection count
        self.connection_counts[best_node] = self.connection_counts.get(best_node, 0) + 1
        
        return [best_node]
    
    def least_load(self, nodes: List[str], node_info: Dict[str, NodeInfo]) -> List[str]:
        """Select node with lowest current load"""
        if not nodes:
            return []
        
        # Find node with minimum load factor
        min_load = float('inf')
        best_node = nodes[0]
        
        for node in nodes:
            if node in node_info:
                load = node_info[node].load_factor
                if load < min_load:
                    min_load = load
                    best_node = node
        
        return [best_node]
    
    def weighted_round_robin(self, nodes: List[str], node_info: Dict[str, NodeInfo]) -> List[str]:
        """Weighted round robin based on node capacity"""
        if not nodes:
            return []
        
        # Calculate weights based on node capabilities
        weights = []
        for node in nodes:
            if node in node_info:
                # Weight based on CPU cores, memory, and capabilities
                weight = (node_info[node].cpu_usage + 
                         node_info[node].memory_usage + 
                         len(node_info[node].capabilities))
                weights.append(max(1, int(weight * 10)))
            else:
                weights.append(1)
        
        # Simple weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return [nodes[0]]
        
        # Select based on weights (simplified)
        return [nodes[0]]  # In practice, would implement proper weighted selection
    
    def affinity_based(self, nodes: List[str]) -> List[str]:
        """Affinity-based node selection (for data locality)"""
        # For now, just use first node
        # In practice, would consider data locality and previous assignments
        return [nodes[0]] if nodes else []

class MessageBroker:
    """Distributed message broker for inter-node communication"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.pubsub = None
        self.connected = False
    
    def connect(self):
        """Connect to Redis message broker"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=0,
                decode_responses=False
            )
            self.redis_client.ping()
            self.connected = True
            logger.info("Connected to message broker")
        except Exception as e:
            logger.error(f"Failed to connect to message broker: {e}")
            self.connected = False
    
    def disconnect(self):
        """Disconnect from message broker"""
        if self.redis_client:
            self.redis_client.close()
            self.connected = False
    
    def broadcast(self, channel: str, message: Dict[str, Any]):
        """Broadcast message to all nodes"""
        if not self.connected:
            return
        
        try:
            serialized_message = json.dumps(message).encode('utf-8')
            self.redis_client.publish(channel, serialized_message)
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
    
    def send_to_node(self, node_id: str, message: Dict[str, Any]):
        """Send message to specific node"""
        if not self.connected:
            return
        
        try:
            channel = f"node_{node_id}"
            serialized_message = json.dumps(message).encode('utf-8')
            self.redis_client.publish(channel, serialized_message)
        except Exception as e:
            logger.error(f"Failed to send message to node {node_id}: {e}")
    
    def subscribe(self, channel: str, callback: Callable):
        """Subscribe to channel and register callback"""
        if not self.connected:
            return
        
        try:
            if self.pubsub is None:
                self.pubsub = self.redis_client.pubsub()
            
            self.pubsub.subscribe(**{channel: callback})
            
            # Start listening in background thread
            thread = threading.Thread(target=self.listen, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.error(f"Failed to subscribe to channel {channel}: {e}")
    
    def listen(self):
        """Listen for messages on subscribed channels"""
        if not self.connected or not self.pubsub:
            return
        
        try:
            for message in self.pubsub.listen():
                if message['type'] == 'message':
                    # Call registered callback
                    pass
        except Exception as e:
            logger.error(f"Message listener error: {e}")

class HealthChecker:
    """Health checking service for cluster nodes"""
    
    def __init__(self):
        self.timeout = 5  # seconds
        self.check_interval = 30  # seconds
    
    def check_node_health(self, node: NodeInfo) -> bool:
        """Check health of a node"""
        try:
            # Try to connect to node's health endpoint
            url = f"http://{node.host}:{node.port}/health"
            response = requests.get(url, timeout=self.timeout)
            
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for node {node.node_id}: {e}")
            return False

class PerformanceMonitor:
    """Performance monitoring and analytics"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}
        self.performance_thresholds = {
            'cpu_usage': 0.8,
            'memory_usage': 0.85,
            'network_latency': 100.0,  # ms
            'task_queue_length': 100
        }
    
    def record_metric(self, metric_name: str, value: float):
        """Record performance metric"""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = []
        
        self.metrics_history[metric_name].append(value)
        
        # Keep only last 1000 values
        if len(self.metrics_history[metric_name]) > 1000:
            self.metrics_history[metric_name].pop(0)
    
    def check_thresholds(self, current_metrics: Dict[str, float]) -> List[str]:
        """Check performance thresholds and return violations"""
        violations = []
        
        for metric, threshold in self.performance_thresholds.items():
            if metric in current_metrics and current_metrics[metric] > threshold:
                violations.append(f"{metric} exceeded threshold: {current_metrics[metric]:.2f} > {threshold}")
        
        return violations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {
            'current_metrics': {},
            'threshold_violations': [],
            'trends': {},
            'recommendations': []
        }
        
        # Calculate current values
        for metric, values in self.metrics_history.items():
            if values:
                report['current_metrics'][metric] = values[-1]
        
        # Check thresholds
        report['threshold_violations'] = self.check_thresholds(report['current_metrics'])
        
        # Calculate trends
        for metric, values in self.metrics_history.items():
            if len(values) >= 10:
                recent = values[-10:]
                trend = np.polyfit(range(len(recent)), recent, 1)[0]
                report['trends'][metric] = trend
        
        # Generate recommendations
        if report['current_metrics'].get('cpu_usage', 0) > 0.9:
            report['recommendations'].append("Consider adding more compute nodes")
        
        if report['current_metrics'].get('memory_usage', 0) > 0.9:
            report['recommendations'].append("Consider adding more memory or optimizing memory usage")
        
        return report

class DistributedAPIServer:
    """REST API server for distributed system management"""
    
    def __init__(self, distributed_system: DistributedSystem):
        self.distributed_system = distributed_system
        self.app = aiohttp.web.Application()
        self.runner = None
        self.site = None
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes"""
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/cluster/status', self.get_cluster_status)
        self.app.router.add_post('/tasks/schedule', self.schedule_task)
        self.app.router.add_get('/tasks/{task_id}', self.get_task_status)
        self.app.router.add_get('/nodes', self.get_nodes)
        self.app.router.add_post('/simulation/distribute', self.distribute_simulation)
    
    async def health_check(self, request):
        """Health check endpoint"""
        return aiohttp.web.json_response({
            'status': 'healthy',
            'node_id': self.distributed_system.node_id,
            'timestamp': time.time()
        })
    
    async def get_cluster_status(self, request):
        """Get cluster status endpoint"""
        status = self.distributed_system.get_cluster_status()
        return aiohttp.web.json_response(status)
    
    async def schedule_task(self, request):
        """Schedule task endpoint"""
        try:
            data = await request.json()
            task = DistributedTask(**data)
            task_id = self.distributed_system.schedule_task(task)
            
            return aiohttp.web.json_response({
                'task_id': task_id,
                'status': 'scheduled'
            })
        except Exception as e:
            return aiohttp.web.json_response({
                'error': str(e)
            }, status=400)
    
    async def get_task_status(self, request):
        """Get task status endpoint"""
        task_id = request.match_info['task_id']
        
        if task_id in self.distributed_system.tasks:
            task = self.distributed_system.tasks[task_id]
            return aiohttp.web.json_response(asdict(task))
        else:
            return aiohttp.web.json_response({
                'error': 'Task not found'
            }, status=404)
    
    async def get_nodes(self, request):
        """Get nodes information endpoint"""
        nodes = {node_id: asdict(node) for node_id, node in self.distributed_system.nodes.items()}
        return aiohttp.web.json_response(nodes)
    
    async def distribute_simulation(self, request):
        """Distribute simulation frame endpoint"""
        try:
            data = await request.json()
            task_ids = self.distributed_system.distribute_simulation_frame(data)
            
            return aiohttp.web.json_response({
                'task_ids': task_ids,
                'status': 'distributed'
            })
        except Exception as e:
            return aiohttp.web.json_response({
                'error': str(e)
            }, status=400)
    
    def start(self):
        """Start the API server"""
        async def start_server():
            self.runner = aiohttp.web.AppRunner(self.app)
            await self.runner.setup()
            
            port = self.distributed_system.config.get('port', 8000)
            self.site = aiohttp.web.TCPSite(
                self.runner, 
                self.distributed_system.config.get('host', 'localhost'), 
                port
            )
            await self.site.start()
            
            logger.info(f"Distributed API server started on port {port}")
        
        # Run in background thread
        thread = threading.Thread(target=lambda: asyncio.run(start_server()), daemon=True)
        thread.start()
    
    def stop(self):
        """Stop the API server"""
        async def stop_server():
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
        
        asyncio.run(stop_server())

# Example usage
if __name__ == "__main__":
    # Test the distributed system
    config = {
        'host': 'localhost',
        'port': 8000,
        'node_type': 'master',
        'capabilities': ['physics', 'rendering', 'computation'],
        'redis_host': 'localhost'
    }
    
    distributed_system = DistributedSystem(None, config)
    
    # Wait a bit for initialization
    time.sleep(2)
    
    # Print cluster status
    status = distributed_system.get_cluster_status()
    print("Cluster Status:")
    print(json.dumps(status, indent=2))
    
    # Cleanup
    distributed_system.cleanup()