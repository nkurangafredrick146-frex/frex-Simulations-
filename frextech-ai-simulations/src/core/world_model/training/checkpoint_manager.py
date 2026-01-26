"""
Checkpoint Manager for World Model Training
Handles checkpoint saving, loading, and management
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from pathlib import Path
import json
import shutil
import logging
import time
from datetime import datetime
import hashlib
import zipfile
import pickle
import yaml
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint"""
    timestamp: str
    epoch: int
    step: int
    loss: float
    val_loss: Optional[float] = None
    lr: Optional[float] = None
    model_hash: Optional[str] = None
    config_hash: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None

class CheckpointManager:
    """Manages checkpoint saving, loading, and organization"""
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        max_checkpoints: int = 10,
        keep_best: int = 3,
        compression: bool = True,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_scaler: bool = True,
        save_metadata: bool = True
    ):
        self.save_dir = Path(save_dir)
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.compression = compression
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_scaler = save_scaler
        self.save_metadata = save_metadata
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = self.save_dir / "checkpoints_metadata.json"
        self.metadata = self._load_metadata()
        
        # Track best checkpoints
        self.best_checkpoints = []
        
        logger.info(f"Checkpoint manager initialized at {self.save_dir}")
    
    def save_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        name: Optional[str] = None,
        is_best: bool = False,
        additional_files: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Path:
        """Save checkpoint with metadata"""
        # Generate checkpoint name
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            epoch = checkpoint_data.get("epoch", 0)
            step = checkpoint_data.get("step", 0)
            name = f"checkpoint_epoch{epoch:04d}_step{step:08d}_{timestamp}"
        
        # Create checkpoint directory
        checkpoint_dir = self.save_dir / name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save main checkpoint file
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        
        # Compress checkpoint data
        if self.compression:
            checkpoint_data = self._compress_checkpoint(checkpoint_data)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save additional files
        if additional_files:
            for filename, data in additional_files.items():
                filepath = checkpoint_dir / filename
                
                if isinstance(data, (dict, list)):
                    with open(filepath.with_suffix('.json'), 'w') as f:
                        json.dump(data, f, indent=2)
                elif isinstance(data, np.ndarray):
                    np.save(filepath.with_suffix('.npy'), data)
                elif isinstance(data, str):
                    with open(filepath, 'w') as f:
                        f.write(data)
                else:
                    logger.warning(f"Unsupported data type for {filename}: {type(data)}")
        
        # Create metadata
        metadata = self._create_metadata(checkpoint_data, name, **kwargs)
        
        # Save metadata
        self._save_metadata_entry(name, metadata, is_best)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return checkpoint_path
    
    def _compress_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress checkpoint data to save space"""
        compressed = {}
        
        for key, value in checkpoint_data.items():
            if isinstance(value, torch.Tensor):
                # Use half precision for tensors
                if value.is_floating_point():
                    compressed[key] = value.half()
                else:
                    compressed[key] = value
            elif isinstance(value, dict) and 'state_dict' in value:
                # Compress model state dict
                compressed[key] = {
                    k: v.half() if v.is_floating_point() else v
                    for k, v in value['state_dict'].items()
                }
            else:
                compressed[key] = value
        
        return compressed
    
    def _create_metadata(
        self,
        checkpoint_data: Dict[str, Any],
        name: str,
        **kwargs
    ) -> CheckpointMetadata:
        """Create metadata for checkpoint"""
        # Compute model hash
        model_state = checkpoint_data.get("model_state_dict")
        model_hash = None
        if model_state:
            model_hash = self._compute_state_hash(model_state)
        
        # Compute config hash
        config = checkpoint_data.get("config")
        config_hash = None
        if config:
            config_hash = self._compute_dict_hash(config)
        
        # Extract metrics
        metrics = checkpoint_data.get("metrics", {})
        if not metrics:
            # Try to extract from checkpoint data
            metrics = {
                "loss": checkpoint_data.get("loss"),
                "val_loss": checkpoint_data.get("val_loss"),
                "accuracy": checkpoint_data.get("accuracy"),
                "psnr": checkpoint_data.get("psnr"),
                "ssim": checkpoint_data.get("ssim")
            }
        
        # Create metadata
        metadata = CheckpointMetadata(
            timestamp=datetime.now().isoformat(),
            epoch=checkpoint_data.get("epoch", 0),
            step=checkpoint_data.get("step", 0),
            loss=checkpoint_data.get("loss", 0.0),
            val_loss=checkpoint_data.get("val_loss"),
            lr=checkpoint_data.get("lr"),
            model_hash=model_hash,
            config_hash=config_hash,
            metrics=metrics,
            tags=kwargs.get("tags", []),
            description=kwargs.get("description")
        )
        
        return metadata
    
    def _compute_state_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Compute hash of model state dict"""
        # Convert to bytes
        bytes_data = b""
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            if tensor.is_floating_point():
                # Use half precision for hash computation
                tensor = tensor.half()
            bytes_data += tensor.cpu().numpy().tobytes()
        
        # Compute hash
        return hashlib.md5(bytes_data).hexdigest()
    
    def _compute_dict_hash(self, data_dict: Dict[str, Any]) -> str:
        """Compute hash of dictionary"""
        # Convert to JSON string
        json_str = json.dumps(data_dict, sort_keys=True)
        
        # Compute hash
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _save_metadata_entry(
        self,
        checkpoint_name: str,
        metadata: CheckpointMetadata,
        is_best: bool
    ):
        """Save metadata entry"""
        # Update metadata dictionary
        self.metadata[checkpoint_name] = asdict(metadata)
        
        # Update best checkpoints
        if is_best:
            self.best_checkpoints.append((checkpoint_name, metadata.val_loss or metadata.loss))
            # Sort by loss (ascending)
            self.best_checkpoints.sort(key=lambda x: x[1])
            # Keep only top N
            self.best_checkpoints = self.best_checkpoints[:self.keep_best]
        
        # Save to file
        with open(self.metadata_file, 'w') as f:
            json.dump({
                "checkpoints": self.metadata,
                "best_checkpoints": self.best_checkpoints,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.best_checkpoints = data.get("best_checkpoints", [])
                    return data.get("checkpoints", {})
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        return {}
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max limit"""
        # Get all checkpoints sorted by timestamp
        checkpoints = []
        for checkpoint_dir in self.save_dir.iterdir():
            if checkpoint_dir.is_dir() and (checkpoint_dir / "checkpoint.pth").exists():
                if checkpoint_dir.name in self.metadata:
                    timestamp = self.metadata[checkpoint_dir.name].get("timestamp", "")
                    checkpoints.append((checkpoint_dir, timestamp))
        
        # Sort by timestamp (oldest first)
        checkpoints.sort(key=lambda x: x[1])
        
        # Remove checkpoints beyond limit, except best ones
        best_names = [name for name, _ in self.best_checkpoints]
        
        for i, (checkpoint_dir, _) in enumerate(checkpoints):
            if checkpoint_dir.name in best_names:
                continue  # Don't remove best checkpoints
            
            if i < len(checkpoints) - self.max_checkpoints:
                logger.info(f"Removing old checkpoint: {checkpoint_dir.name}")
                shutil.rmtree(checkpoint_dir)
                
                # Remove from metadata
                if checkpoint_dir.name in self.metadata:
                    del self.metadata[checkpoint_dir.name]
    
    def load_checkpoint(
        self,
        checkpoint_name: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        map_location: Optional[Union[str, torch.device]] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_scaler: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load checkpoint"""
        # Determine checkpoint path
        if checkpoint_path is None:
            if checkpoint_name is None:
                # Load latest checkpoint
                checkpoint_name = self.get_latest_checkpoint()
            
            checkpoint_path = self.save_dir / checkpoint_name / "checkpoint.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
        
        # Decompress if needed
        checkpoint_data = self._decompress_checkpoint(checkpoint_data)
        
        # Validate checkpoint
        self._validate_checkpoint(checkpoint_data)
        
        # Load additional files
        checkpoint_dir = checkpoint_path.parent
        additional_data = self._load_additional_files(checkpoint_dir)
        
        # Merge with checkpoint data
        checkpoint_data.update(additional_data)
        
        # Log checkpoint info
        metadata = self.metadata.get(checkpoint_dir.name, {})
        logger.info(f"Checkpoint info: epoch={metadata.get('epoch', 'N/A')}, "
                   f"step={metadata.get('step', 'N/A')}, "
                   f"loss={metadata.get('loss', 'N/A'):.6f}")
        
        return checkpoint_data
    
    def _decompress_checkpoint(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress checkpoint data"""
        if not self.compression:
            return checkpoint_data
        
        decompressed = {}
        
        for key, value in checkpoint_data.items():
            if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                # Decompress model state dict
                decompressed[key] = {
                    k: v.float() if v.is_floating_point() else v
                    for k, v in value.items()
                }
            elif isinstance(value, torch.Tensor) and value.is_floating_point():
                # Decompress tensor
                decompressed[key] = value.float()
            else:
                decompressed[key] = value
        
        return decompressed
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Validate checkpoint data"""
        required_keys = ["model_state_dict", "epoch", "step"]
        
        for key in required_keys:
            if key not in checkpoint_data:
                logger.warning(f"Checkpoint missing required key: {key}")
        
        # Validate model state dict
        model_state = checkpoint_data.get("model_state_dict")
        if model_state:
            if isinstance(model_state, dict):
                logger.info(f"Model state dict contains {len(model_state)} parameters")
            else:
                logger.warning("Model state dict is not a dictionary")
    
    def _load_additional_files(self, checkpoint_dir: Path) -> Dict[str, Any]:
        """Load additional files from checkpoint directory"""
        additional_data = {}
        
        for filepath in checkpoint_dir.iterdir():
            if filepath.name == "checkpoint.pth":
                continue
            
            if filepath.suffix == '.json':
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    additional_data[filepath.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
            
            elif filepath.suffix == '.npy':
                try:
                    data = np.load(filepath)
                    additional_data[filepath.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
            
            elif filepath.suffix == '.pkl':
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    additional_data[filepath.stem] = data
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
        
        return additional_data
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get name of latest checkpoint"""
        if not self.metadata:
            return None
        
        # Find checkpoint with latest timestamp
        latest_checkpoint = None
        latest_timestamp = ""
        
        for checkpoint_name, metadata in self.metadata.items():
            timestamp = metadata.get("timestamp", "")
            if timestamp > latest_timestamp:
                latest_timestamp = timestamp
                latest_checkpoint = checkpoint_name
        
        return latest_checkpoint
    
    def get_best_checkpoint(self, metric: str = "val_loss", mode: str = "min") -> Optional[str]:
        """Get name of best checkpoint based on metric"""
        if not self.metadata:
            return None
        
        # Find best checkpoint based on metric
        best_checkpoint = None
        best_value = float('inf') if mode == "min" else float('-inf')
        
        for checkpoint_name, metadata in self.metadata.items():
            value = metadata.get("metrics", {}).get(metric)
            if value is not None:
                if (mode == "min" and value < best_value) or (mode == "max" and value > best_value):
                    best_value = value
                    best_checkpoint = checkpoint_name
        
        return best_checkpoint
    
    def list_checkpoints(self, sort_by: str = "timestamp", reverse: bool = True) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for checkpoint_name, metadata in self.metadata.items():
            checkpoint_dir = self.save_dir / checkpoint_name
            checkpoint_path = checkpoint_dir / "checkpoint.pth"
            
            if checkpoint_path.exists():
                checkpoints.append({
                    "name": checkpoint_name,
                    "path": str(checkpoint_path),
                    "metadata": metadata,
                    "size": checkpoint_path.stat().st_size / (1024 * 1024),  # MB
                    "exists": True
                })
        
        # Sort checkpoints
        if sort_by == "timestamp":
            checkpoints.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=reverse)
        elif sort_by == "epoch":
            checkpoints.sort(key=lambda x: x["metadata"].get("epoch", 0), reverse=reverse)
        elif sort_by == "loss":
            checkpoints.sort(key=lambda x: x["metadata"].get("loss", float('inf')), reverse=not reverse)
        elif sort_by == "val_loss":
            checkpoints.sort(key=lambda x: x["metadata"].get("val_loss", float('inf')), reverse=not reverse)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_name: str):
        """Delete checkpoint"""
        checkpoint_dir = self.save_dir / checkpoint_name
        
        if checkpoint_dir.exists():
            logger.info(f"Deleting checkpoint: {checkpoint_name}")
            shutil.rmtree(checkpoint_dir)
            
            # Remove from metadata
            if checkpoint_name in self.metadata:
                del self.metadata[checkpoint_name]
            
            # Remove from best checkpoints
            self.best_checkpoints = [(name, loss) for name, loss in self.best_checkpoints if name != checkpoint_name]
            
            # Save updated metadata
            self._save_metadata_file()
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_name}")
    
    def export_checkpoint(
        self,
        checkpoint_name: str,
        export_path: Path,
        include_metadata: bool = True,
        include_config: bool = True,
        include_samples: bool = False
    ) -> Path:
        """Export checkpoint to external location"""
        checkpoint_dir = self.save_dir / checkpoint_name
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
        
        # Create export directory
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint files
        checkpoint_files = ["checkpoint.pth"]
        
        if include_metadata:
            checkpoint_files.append("metadata.json")
        
        for filename in checkpoint_files:
            src = checkpoint_dir / filename
            dst = export_path / filename
            
            if src.exists():
                shutil.copy2(src, dst)
        
        # Copy config if available
        if include_config:
            config_files = ["config.json", "config.yaml", "config.yml"]
            for config_file in config_files:
                src = self.save_dir.parent / config_file
                if src.exists():
                    shutil.copy2(src, export_path / config_file)
        
        # Copy sample images if requested
        if include_samples:
            samples_dir = checkpoint_dir / "samples"
            if samples_dir.exists():
                shutil.copytree(samples_dir, export_path / "samples", dirs_exist_ok=True)
        
        # Create archive
        archive_path = export_path.parent / f"{checkpoint_name}.zip"
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in export_path.rglob("*"):
                if filepath.is_file():
                    arcname = filepath.relative_to(export_path)
                    zipf.write(filepath, arcname)
        
        logger.info(f"Checkpoint exported to: {archive_path}")
        
        return archive_path
    
    def import_checkpoint(
        self,
        import_path: Union[str, Path],
        checkpoint_name: Optional[str] = None
    ) -> str:
        """Import checkpoint from external location"""
        import_path = Path(import_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import path not found: {import_path}")
        
        # Extract if zip file
        if import_path.suffix == '.zip':
            extract_dir = self.save_dir / "import_temp"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(import_path, 'r') as zipf:
                zipf.extractall(extract_dir)
            
            # Find checkpoint file
            checkpoint_files = list(extract_dir.rglob("checkpoint.pth"))
            if not checkpoint_files:
                raise FileNotFoundError("No checkpoint.pth found in archive")
            
            import_dir = checkpoint_files[0].parent
        else:
            import_dir = import_path
        
        # Determine checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"imported_{timestamp}"
        
        # Copy to checkpoint directory
        dest_dir = self.save_dir / checkpoint_name
        dest_dir.mkdir(exist_ok=True)
        
        # Copy all files
        for filepath in import_dir.iterdir():
            if filepath.is_file():
                shutil.copy2(filepath, dest_dir / filepath.name)
        
        # Load and save metadata
        checkpoint_path = dest_dir / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Create metadata
            metadata = self._create_metadata(checkpoint_data, checkpoint_name)
            self._save_metadata_entry(checkpoint_name, metadata, is_best=False)
        
        # Cleanup temp directory
        if 'extract_dir' in locals():
            shutil.rmtree(extract_dir)
        
        logger.info(f"Checkpoint imported as: {checkpoint_name}")
        
        return checkpoint_name
    
    def create_snapshot(self, snapshot_name: Optional[str] = None) -> Path:
        """Create snapshot of all checkpoints"""
        if snapshot_name is None:
            snapshot_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        snapshot_dir = self.save_dir.parent / "snapshots" / snapshot_name
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all checkpoints
        for checkpoint_dir in self.save_dir.iterdir():
            if checkpoint_dir.is_dir():
                dest_dir = snapshot_dir / checkpoint_dir.name
                shutil.copytree(checkpoint_dir, dest_dir)
        
        # Copy metadata
        shutil.copy2(self.metadata_file, snapshot_dir / "checkpoints_metadata.json")
        
        # Create archive
        snapshot_archive = snapshot_dir.parent / f"{snapshot_name}.tar.gz"
        
        import tarfile
        with tarfile.open(snapshot_archive, "w:gz") as tar:
            tar.add(snapshot_dir, arcname=snapshot_dir.name)
        
        # Cleanup
        shutil.rmtree(snapshot_dir)
        
        logger.info(f"Snapshot created: {snapshot_archive}")
        
        return snapshot_archive
    
    def verify_checkpoint(self, checkpoint_name: str) -> Dict[str, Any]:
        """Verify checkpoint integrity"""
        checkpoint_dir = self.save_dir / checkpoint_name
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")
        
        results = {
            "name": checkpoint_name,
            "exists": True,
            "size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
            "files": [],
            "integrity": {},
            "metadata": {}
        }
        
        # Check all files
        for filepath in checkpoint_dir.iterdir():
            if filepath.is_file():
                file_info = {
                    "name": filepath.name,
                    "size_mb": filepath.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                }
                results["files"].append(file_info)
        
        # Load and verify checkpoint
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            results["integrity"]["load_success"] = True
            
            # Check model state dict
            if "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
                results["integrity"]["model_params"] = len(state_dict)
                
                # Check for NaN or Inf
                has_nan = False
                has_inf = False
                for tensor in state_dict.values():
                    if torch.is_tensor(tensor) and tensor.is_floating_point():
                        if torch.isnan(tensor).any():
                            has_nan = True
                        if torch.isinf(tensor).any():
                            has_inf = True
                
                results["integrity"]["has_nan"] = has_nan
                results["integrity"]["has_inf"] = has_inf
            
            # Check metadata
            if checkpoint_name in self.metadata:
                results["metadata"] = self.metadata[checkpoint_name]
            
        except Exception as e:
            results["integrity"]["load_success"] = False
            results["integrity"]["error"] = str(e)
        
        return results
    
    def _save_metadata_file(self):
        """Save metadata to file"""
        with open(self.metadata_file, 'w') as f:
            json.dump({
                "checkpoints": self.metadata,
                "best_checkpoints": self.best_checkpoints,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

class ModelEMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: Optional[torch.device] = None):
        self.model = model
        self.decay = decay
        self.device = device
        
        # Create shadow model
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state dict"""
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']

class CheckpointCallback:
    """Callback for checkpoint saving during training"""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        save_interval: int = 1000,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: bool = True
    ):
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.best_epoch = 0
    
    def __call__(
        self,
        epoch: int,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object],
        scaler: Optional[GradScaler],
        metrics: Dict[str, float]
    ):
        """Callback function called during training"""
        should_save = False
        is_best = False
        
        # Check interval
        if step % self.save_interval == 0:
            should_save = True
        
        # Check if best
        if self.save_best and self.monitor in metrics:
            current_value = metrics[self.monitor]
            
            if (self.mode == "min" and current_value < self.best_value) or \
               (self.mode == "max" and current_value > self.best_value):
                self.best_value = current_value
                self.best_epoch = epoch
                is_best = True
                should_save = True
        
        # Save checkpoint
        if should_save:
            checkpoint_data = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "metrics": metrics,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch
            }
            
            # Add loss to metrics if not present
            if "loss" in metrics and "loss" not in checkpoint_data:
                checkpoint_data["loss"] = metrics["loss"]
            
            name = f"epoch{epoch:04d}_step{step:08d}"
            
            self.checkpoint_manager.save_checkpoint(
                checkpoint_data,
                name=name,
                is_best=is_best,
                tags=["interval"] if not is_best else ["interval", "best"]
            )
            
            if self.verbose:
                logger.info(f"Checkpoint saved at epoch {epoch}, step {step}")
                if is_best:
                    logger.info(f"New best {self.monitor}: {self.best_value:.6f}")