"""
Distributed computing utilities for multi-GPU and multi-node training.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from typing import Optional, List, Dict, Any, Union, Callable
import socket
import os
import time
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager
import warnings


class NodeRole(Enum):
    """Node roles in distributed setup."""
    MASTER = "master"
    WORKER = "worker"
    EVALUATOR = "evaluator"
    PARAMETER_SERVER = "parameter_server"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # or "gloo", "mpi"
    init_method: str = "env://"  # or "tcp://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    node_role: NodeRole = NodeRole.WORKER
    timeout: int = 1800  # seconds
    sync_batch_norm: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    def __post_init__(self):
        """Set defaults from environment if available."""
        # Read from environment variables
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        if "MASTER_ADDR" in os.environ:
            self.master_addr = os.environ["MASTER_ADDR"]
        if "MASTER_PORT" in os.environ:
            self.master_port = int(os.environ["MASTER_PORT"])


class DistributedManager:
    """Manager for distributed training setup."""
    
    _instance: Optional["DistributedManager"] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config: Optional[DistributedConfig] = None
            self.process_group: Optional[dist.ProcessGroup] = None
            self.device: Optional[torch.device] = None
            self.logger: Optional[logging.Logger] = None
            self._barrier_counter = 0
            self._initialized = True
    
    def initialize(self, config: Optional[DistributedConfig] = None) -> bool:
        """Initialize distributed training.
        
        Args:
            config: Distributed configuration
            
        Returns:
            True if initialization successful
        """
        if self.is_initialized():
            warnings.warn("DistributedManager already initialized")
            return True
        
        self.config = config or DistributedConfig()
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f"cuda:{self.config.local_rank}")
        else:
            self.device = torch.device("cpu")
        
        # Initialize process group
        try:
            if self.config.world_size > 1:
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=self.config.init_method,
                    world_size=self.config.world_size,
                    rank=self.config.rank,
                    timeout=datetime.timedelta(seconds=self.config.timeout)
                )
                
                self.process_group = dist.group.WORLD
                self.logger.info(f"Initialized process group: rank={self.config.rank}, "
                               f"world_size={self.config.world_size}, "
                               f"backend={self.config.backend}")
            else:
                self.logger.info("Running in single process mode")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            return False
    
    def _setup_logger(self) -> logging.Logger:
        """Setup distributed-aware logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(f"distributed_rank_{self.config.rank}")
        
        if not logger.handlers:
            formatter = logging.Formatter(
                f'[Rank {self.config.rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            logger.setLevel(logging.INFO)
        
        return logger
    
    def is_initialized(self) -> bool:
        """Check if distributed training is initialized.
        
        Returns:
            True if initialized
        """
        return dist.is_initialized() if dist.is_available() else False
    
    def get_rank(self) -> int:
        """Get global rank.
        
        Returns:
            Global rank
        """
        return dist.get_rank() if self.is_initialized() else 0
    
    def get_local_rank(self) -> int:
        """Get local rank.
        
        Returns:
            Local rank
        """
        return self.config.local_rank if self.config else 0
    
    def get_world_size(self) -> int:
        """Get world size.
        
        Returns:
            World size
        """
        return dist.get_world_size() if self.is_initialized() else 1
    
    def is_master(self) -> bool:
        """Check if current process is master (rank 0).
        
        Returns:
            True if master
        """
        return self.get_rank() == 0
    
    def barrier(self):
        """Synchronize all processes."""
        if self.is_initialized() and self.get_world_size() > 1:
            dist.barrier()
            self._barrier_counter += 1
    
    def broadcast(self, data: Any, src: int = 0) -> Any:
        """Broadcast data from source to all processes.
        
        Args:
            data: Data to broadcast
            src: Source rank
            
        Returns:
            Broadcasted data
        """
        if not self.is_initialized() or self.get_world_size() == 1:
            return data
        
        # Handle different data types
        if torch.is_tensor(data):
            dist.broadcast(data, src=src)
            return data
        else:
            # Convert to tensor for broadcasting
            if self.get_rank() == src:
                # Serialize and send
                serialized = pickle.dumps(data)
                length_tensor = torch.tensor([len(serialized)], dtype=torch.long)
                dist.broadcast(length_tensor, src=src)
                
                # Convert to tensor and broadcast
                data_tensor = torch.ByteTensor(list(serialized))
                dist.broadcast(data_tensor, src=src)
                return data
            else:
                # Receive length
                length_tensor = torch.tensor([0], dtype=torch.long)
                dist.broadcast(length_tensor, src=src)
                length = length_tensor.item()
                
                # Receive data
                data_tensor = torch.ByteTensor([0] * length)
                dist.broadcast(data_tensor, src=src)
                
                # Deserialize
                serialized = bytes(data_tensor.tolist())
                return pickle.loads(serialized)
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce operation across all processes.
        
        Args:
            tensor: Input tensor
            op: Reduction operation ("sum", "mean", "prod", "min", "max")
            
        Returns:
            Reduced tensor
        """
        if not self.is_initialized() or self.get_world_size() == 1:
            return tensor
        
        # Map operation string to dist.ReduceOp
        op_map = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,  # Divide by world size after
            "prod": dist.ReduceOp.PRODUCT,
            "min": dist.ReduceOp.MIN,
            "max": dist.ReduceOp.MAX
        }
        
        if op not in op_map:
            raise ValueError(f"Unsupported reduction operation: {op}")
        
        # Perform all-reduce
        dist.all_reduce(tensor, op=op_map[op])
        
        # Handle mean operation
        if op == "mean":
            tensor = tensor / self.get_world_size()
        
        return tensor
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensors from all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            List of gathered tensors from all processes
        """
        if not self.is_initialized() or self.get_world_size() == 1:
            return [tensor]
        
        world_size = self.get_world_size()
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        
        return tensor_list
    
    def gather(self, data: Any, dst: int = 0) -> Optional[List[Any]]:
        """Gather data from all processes to destination.
        
        Args:
            data: Data to gather
            dst: Destination rank
            
        Returns:
            List of gathered data (only at destination), None otherwise
        """
        if not self.is_initialized() or self.get_world_size() == 1:
            return [data]
        
        rank = self.get_rank()
        world_size = self.get_world_size()
        
        if torch.is_tensor(data):
            # Tensor gathering
            if rank == dst:
                # Destination: gather all tensors
                gathered = [torch.zeros_like(data) for _ in range(world_size)]
                dist.gather(data, gathered, dst=dst)
                return gathered
            else:
                # Source: send tensor
                dist.gather(data, dst=dst)
                return None
        else:
            # Object gathering
            # Serialize to tensor
            serialized = pickle.dumps(data)
            length = len(serialized)
            
            # Gather lengths first
            length_tensor = torch.tensor([length], dtype=torch.long)
            gathered_lengths = self.gather(length_tensor, dst=dst)
            
            if rank == dst:
                # Allocate buffers
                max_length = max(t.item() for t in gathered_lengths)
                data_tensor = torch.ByteTensor([0] * max_length)
                gathered_data = []
                
                # Gather data from each rank
                for i in range(world_size):
                    if i == dst:
                        # Local data
                        gathered_data.append(data)
                    else:
                        # Receive from other ranks
                        recv_tensor = torch.ByteTensor([0] * gathered_lengths[i].item())
                        dist.recv(recv_tensor, src=i)
                        gathered_data.append(pickle.loads(bytes(recv_tensor.tolist())))
                
                return gathered_data
            else:
                # Send data to destination
                data_tensor = torch.ByteTensor(list(serialized))
                dist.send(data_tensor, dst=dst)
                return None
    
    def scatter(self, data_list: List[Any], src: int = 0) -> Any:
        """Scatter data from source to all processes.
        
        Args:
            data_list: List of data to scatter (length must equal world_size)
            src: Source rank
            
        Returns:
            Scattered data for current process
        """
        if not self.is_initialized() or self.get_world_size() == 1:
            return data_list[0]
        
        rank = self.get_rank()
        world_size = self.get_world_size()
        
        if len(data_list) != world_size:
            raise ValueError(f"data_list length ({len(data_list)}) must equal world_size ({world_size})")
        
        if torch.is_tensor(data_list[0]):
            # Tensor scattering
            if rank == src:
                # Source: scatter tensors
                dist.scatter(data_list[0], data_list, src=src)
                return data_list[0]
            else:
                # Destination: receive tensor
                recv_tensor = torch.zeros_like(data_list[0])
                dist.scatter(recv_tensor, src=src)
                return recv_tensor
        else:
            # Object scattering
            if rank == src:
                # Serialize and send
                for i in range(world_size):
                    if i == src:
                        continue
                    
                    serialized = pickle.dumps(data_list[i])
                    length_tensor = torch.tensor([len(serialized)], dtype=torch.long)
                    dist.send(length_tensor, dst=i)
                    
                    data_tensor = torch.ByteTensor(list(serialized))
                    dist.send(data_tensor, dst=i)
                
                return data_list[src]
            else:
                # Receive from source
                length_tensor = torch.tensor([0], dtype=torch.long)
                dist.recv(length_tensor, src=src)
                length = length_tensor.item()
                
                data_tensor = torch.ByteTensor([0] * length)
                dist.recv(data_tensor, src=src)
                
                return pickle.loads(bytes(data_tensor.tolist()))
    
    def create_ddp_model(self, model: torch.nn.Module, **kwargs) -> torch.nn.Module:
        """Wrap model with DistributedDataParallel.
        
        Args:
            model: Model to wrap
            **kwargs: Additional arguments for DDP
            
        Returns:
            DDP-wrapped model
        """
        if not self.is_initialized() or self.get_world_size() == 1:
            return model
        
        # Merge with default config
        ddp_kwargs = {
            "device_ids": [self.config.local_rank] if torch.cuda.is_available() else None,
            "output_device": self.config.local_rank if torch.cuda.is_available() else None,
            "find_unused_parameters": self.config.find_unused_parameters,
            "gradient_as_bucket_view": self.config.gradient_as_bucket_view,
            "static_graph": self.config.static_graph,
            **kwargs
        }
        
        # Sync batch norm if requested
        if self.config.sync_batch_norm and isinstance(model, torch.nn.SyncBatchNorm):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        return DDP(model, **ddp_kwargs)
    
    def create_distributed_sampler(self, dataset, **kwargs):
        """Create distributed sampler for dataset.
        
        Args:
            dataset: Dataset to sample from
            **kwargs: Additional arguments for DistributedSampler
            
        Returns:
            DistributedSampler
        """
        if not self.is_initialized() or self.get_world_size() == 1:
            return None
        
        return DistributedSampler(
            dataset,
            num_replicas=self.get_world_size(),
            rank=self.get_rank(),
            **kwargs
        )
    
    @contextmanager
    def distributed_context(self, model: torch.nn.Module = None):
        """Context manager for distributed operations.
        
        Args:
            model: Optional model to wrap with DDP
            
        Yields:
            Distributed context
        """
        if model is not None:
            model = self.create_ddp_model(model)
        
        try:
            yield model
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup distributed resources."""
        if self.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
            self.logger.info("Cleaned up distributed resources")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed training statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "initialized": self.is_initialized(),
            "rank": self.get_rank(),
            "local_rank": self.get_local_rank(),
            "world_size": self.get_world_size(),
            "is_master": self.is_master(),
            "device": str(self.device) if self.device else None,
            "barrier_count": self._barrier_counter,
            "backend": self.config.backend if self.config else None
        }


# Global manager instance
_dist_manager = DistributedManager()


# Convenience functions
def init_distributed(config: Optional[DistributedConfig] = None) -> bool:
    """Initialize distributed training.
    
    Args:
        config: Distributed configuration
        
    Returns:
        True if successful
    """
    return _dist_manager.initialize(config)


def get_rank() -> int:
    """Get global rank.
    
    Returns:
        Global rank
    """
    return _dist_manager.get_rank()


def get_local_rank() -> int:
    """Get local rank.
    
    Returns:
        Local rank
    """
    return _dist_manager.get_local_rank()


def get_world_size() -> int:
    """Get world size.
    
    Returns:
        World size
    """
    return _dist_manager.get_world_size()


def is_master() -> bool:
    """Check if current process is master.
    
    Returns:
        True if master
    """
    return _dist_manager.is_master()


def barrier():
    """Synchronize all processes."""
    _dist_manager.barrier()


def broadcast(data: Any, src: int = 0) -> Any:
    """Broadcast data from source to all processes.
    
    Args:
        data: Data to broadcast
        src: Source rank
        
    Returns:
        Broadcasted data
    """
    return _dist_manager.broadcast(data, src)


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """All-reduce operation across all processes.
    
    Args:
        tensor: Input tensor
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    return _dist_manager.all_reduce(tensor, op)


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        List of gathered tensors
    """
    return _dist_manager.all_gather(tensor)


def gather(data: Any, dst: int = 0) -> Optional[List[Any]]:
    """Gather data from all processes to destination.
    
    Args:
        data: Data to gather
        dst: Destination rank
        
    Returns:
        List of gathered data (only at destination)
    """
    return _dist_manager.gather(data, dst)


def scatter(data_list: List[Any], src: int = 0) -> Any:
    """Scatter data from source to all processes.
    
    Args:
        data_list: List of data to scatter
        src: Source rank
        
    Returns:
        Scattered data for current process
    """
    return _dist_manager.scatter(data_list, src)


def cleanup():
    """Cleanup distributed resources."""
    _dist_manager.cleanup()


def get_distributed_manager() -> DistributedManager:
    """Get the global distributed manager instance.
    
    Returns:
        DistributedManager instance
    """
    return _dist_manager
