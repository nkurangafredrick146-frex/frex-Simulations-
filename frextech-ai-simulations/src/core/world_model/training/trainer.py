"""
World Model Trainer
Main training loop for 3D diffusion world models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import logging
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import wandb
import copy
import random

from ..architecture.diffusion_models import DiffusionModel3D
from ..architecture.custom_layers import PositionalEncoding3D, FourierFeature3D
from ...multimodal.encoders.text_encoder import TextEncoder
from ...multimodal.encoders.vision_encoder import VisionEncoder
from .loss_functions import WorldLoss, PerceptualLoss
from .optimizer_scheduler import OptimizerManager, get_scheduler
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    batch_size: int = 4
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Model
    model_name: str = "world_diffusion_3d"
    latent_channels: int = 4
    model_channels: int = 128
    num_res_blocks: int = 2
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    attention_resolutions: Tuple[int, ...] = (16, 8)
    
    # Training
    epochs: int = 1000
    steps_per_epoch: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Diffusion
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    loss_type: str = "l2"
    objective: str = "pred_noise"
    
    # Mixed Precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16", "bfloat16"
    
    # Distributed
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1
    sync_bn: bool = False
    
    # Logging
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    sample_interval: int = 1000
    log_to_wandb: bool = True
    wandb_project: str = "frextech-world-model"
    wandb_entity: Optional[str] = None
    
    # Checkpoints
    checkpoint_dir: Path = Path("checkpoints")
    max_checkpoints: int = 10
    resume_from: Optional[str] = None
    
    # Validation
    validation_split: float = 0.1
    validation_steps: int = 100
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Data Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    
    # Early Stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001

class WorldModelTrainer:
    """Main trainer for 3D world diffusion models"""
    
    def __init__(
        self,
        model: DiffusionModel3D,
        config: TrainingConfig,
        train_dataset,
        val_dataset = None,
        text_encoder: Optional[TextEncoder] = None,
        vision_encoder: Optional[VisionEncoder] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        
        # Setup device
        self.device = self._setup_device()
        
        # Setup distributed training
        self.is_distributed = config.distributed and torch.cuda.device_count() > 1
        if self.is_distributed:
            self._setup_distributed()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                find_unused_parameters=True
            )
            if config.sync_bn:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Setup training components
        self._setup_training_components()
        
        # Metrics
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Timing
        self.start_time = time.time()
        self.epoch_times = []
        
        # Logging
        self._setup_logging()
        
        # Load checkpoint if resuming
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.local_rank}")
            torch.cuda.set_device(device)
            
            # Enable TF32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set deterministic for reproducibility
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        return device
    
    def _setup_distributed(self):
        """Setup distributed training"""
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.config.world_size,
            rank=self.config.local_rank
        )
        logger.info(f"Initialized distributed training: rank {self.config.local_rank}, world size {self.config.world_size}")
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, loss, etc."""
        # Optimizer
        self.optimizer_manager = OptimizerManager(self.model)
        self.optimizer = self.optimizer_manager.create_optimizer(
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = get_scheduler(
            "cosine_warmup",
            optimizer=self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.epochs * self.config.steps_per_epoch
        )
        
        # Loss function
        self.loss_fn = WorldLoss(
            diffusion_loss_type=self.config.loss_type,
            perceptual_weight=0.1,
            consistency_weight=0.01,
            kl_weight=0.0001
        )
        
        # Perceptual loss for validation
        self.perceptual_loss = PerceptualLoss().to(self.device)
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Dataloaders
        self.train_loader = self._create_dataloader(self.train_dataset, train=True)
        self.val_loader = self._create_dataloader(self.val_dataset, train=False) if self.val_dataset else None
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config.checkpoint_dir,
            max_checkpoints=self.config.max_checkpoints
        )
    
    def _create_dataloader(self, dataset, train: bool = True) -> DataLoader:
        """Create dataloader"""
        if dataset is None:
            return None
        
        sampler = None
        if self.is_distributed and train:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.local_rank,
                shuffle=True
            )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None and train),
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True,
            drop_last=train
        )
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        # TensorBoard
        if self.config.local_rank == 0:
            self.writer = SummaryWriter(log_dir=self.config.checkpoint_dir / "tensorboard")
        
        # Weights & Biases
        if self.config.log_to_wandb and self.config.local_rank == 0:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=asdict(self.config),
                name=f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            wandb.watch(self.model, log="all", log_freq=self.config.log_interval)
        
        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        if self.config.local_rank == 0:
            config_path = self.config.checkpoint_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                epoch_loss = self.train_epoch()
                
                # Validate
                if self.val_loader and (epoch % self.config.eval_interval == 0 or epoch == self.config.epochs - 1):
                    val_loss = self.validate()
                    
                    # Check for improvement
                    if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        
                        # Save best model
                        if self.config.local_rank == 0:
                            self.save_checkpoint("best_model")
                    else:
                        self.patience_counter += 1
                
                # Early stopping check
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
                
                # Log epoch metrics
                self.log_epoch_metrics(epoch, epoch_loss)
            
            # Training complete
            logger.info("Training completed!")
            
            # Save final model
            if self.config.local_rank == 0:
                self.save_checkpoint("final_model")
            
            # Cleanup
            self.cleanup()
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            if self.config.local_rank == 0:
                self.save_checkpoint("interrupted")
            self.cleanup()
            raise
        
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            if self.config.local_rank == 0:
                self.save_checkpoint("failed")
            self.cleanup()
            raise
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        
        if self.is_distributed and self.train_loader.sampler:
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            disable=self.config.local_rank != 0
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Stop if we've reached steps per epoch
            if batch_idx >= self.config.steps_per_epoch:
                break
            
            # Prepare batch
            batch = self._prepare_batch(batch)
            
            # Training step
            loss, metrics = self.train_step(batch, batch_idx)
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log step metrics
            if self.global_step % self.config.log_interval == 0:
                self.log_step_metrics(metrics, "train")
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0 and self.config.local_rank == 0:
                self.save_checkpoint(f"step_{self.global_step}")
            
            # Generate samples
            if self.global_step % self.config.sample_interval == 0 and self.config.local_rank == 0:
                self.generate_samples()
            
            self.global_step += 1
        
        # Average loss for epoch
        epoch_loss /= min(num_batches, self.config.steps_per_epoch)
        
        return epoch_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step"""
        # Unpack batch
        x = batch["voxels"]  # Shape: [B, C, D, H, W]
        text_embeddings = batch.get("text_embeddings", None)
        images = batch.get("images", None)
        
        # Sample timesteps
        t = torch.randint(0, self.config.timesteps, (x.shape[0],), device=self.device).long()
        
        # Prepare condition
        condition = None
        if text_embeddings is not None:
            condition = text_embeddings
        elif images is not None and self.vision_encoder is not None:
            with torch.no_grad():
                condition = self.vision_encoder(images)
        
        # Mixed precision training
        with autocast(enabled=self.config.use_amp, dtype=self._get_amp_dtype()):
            # Forward pass
            loss = self.model.p_losses(x, t, conditioning=condition)
            
            # Add perceptual loss if images available
            if images is not None and self.perceptual_loss is not None:
                # Generate reconstruction for perceptual loss
                with torch.no_grad():
                    noise = torch.randn_like(x)
                    x_t = self.model.q_sample(x_start=x, t=t, noise=noise)
                    pred = self.model(x_t, t, condition)
                
                # Compute perceptual loss
                perc_loss = self.perceptual_loss(pred, x)
                loss = loss + 0.1 * perc_loss
        
        # Backward pass with gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if accumulation steps reached
        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.gradient_clip > 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            
            # Optimizer step
            if self.config.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        # Metrics
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "lr": self.optimizer.param_groups[0]["lr"]
        }
        
        return loss * self.config.gradient_accumulation_steps, metrics
    
    def validate(self) -> float:
        """Validation loop"""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc="Validation",
                disable=self.config.local_rank != 0
            )
            
            for batch in pbar:
                batch = self._prepare_batch(batch)
                x = batch["voxels"]
                
                # Sample timesteps
                t = torch.randint(0, self.config.timesteps, (x.shape[0],), device=self.device).long()
                
                # Get condition
                condition = None
                if "text_embeddings" in batch:
                    condition = batch["text_embeddings"]
                
                # Compute loss
                loss = self.model.p_losses(x, t, conditioning=condition)
                total_loss += loss.item()
                
                pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        
        # Log validation metrics
        if self.config.local_rank == 0:
            metrics = {
                "val_loss": avg_loss,
                "val_psnr": self._compute_psnr(avg_loss),
                "val_ssim": self._compute_ssim_estimate(avg_loss)
            }
            self.log_step_metrics(metrics, "val")
        
        return avg_loss
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training"""
        prepared = {}
        
        for key, value in batch.items():
            if torch.is_tensor(value):
                prepared[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                # Convert to tensor
                tensor = torch.from_numpy(np.array(value)) if isinstance(value, np.ndarray) else torch.tensor(value)
                prepared[key] = tensor.to(self.device, non_blocking=True)
            else:
                prepared[key] = value
        
        # Apply data augmentation if needed
        if self.config.use_augmentation and "voxels" in prepared:
            if random.random() < self.config.augmentation_prob:
                prepared["voxels"] = self._augment_voxels(prepared["voxels"])
        
        return prepared
    
    def _augment_voxels(self, voxels: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to voxels"""
        B, C, D, H, W = voxels.shape
        
        # Random rotation (90 degree increments)
        if random.random() < 0.3:
            k = random.randint(0, 3)
            voxels = torch.rot90(voxels, k, dims=[-2, -1])
        
        # Random flip
        if random.random() < 0.3:
            if random.random() < 0.5:
                voxels = torch.flip(voxels, dims=[-1])  # Horizontal flip
            else:
                voxels = torch.flip(voxels, dims=[-2])  # Vertical flip
        
        # Random scaling (zoom)
        if random.random() < 0.2:
            scale = random.uniform(0.8, 1.2)
            new_size = [max(1, int(s * scale)) for s in [D, H, W]]
            
            # Upsample or downsample
            voxels = F.interpolate(
                voxels,
                size=tuple(new_size),
                mode='trilinear',
                align_corners=False
            )
            
            # Pad or crop to original size
            if new_size[0] < D or new_size[1] < H or new_size[2] < W:
                pad_d = (D - new_size[0]) // 2
                pad_h = (H - new_size[1]) // 2
                pad_w = (W - new_size[2]) // 2
                
                voxels = F.pad(
                    voxels,
                    (pad_w, W - new_size[2] - pad_w,
                     pad_h, H - new_size[1] - pad_h,
                     pad_d, D - new_size[0] - pad_d),
                    mode='constant',
                    value=0
                )
            else:
                # Crop
                start_d = (new_size[0] - D) // 2
                start_h = (new_size[1] - H) // 2
                start_w = (new_size[2] - W) // 2
                
                voxels = voxels[
                    :, :,
                    start_d:start_d + D,
                    start_h:start_h + H,
                    start_w:start_w + W
                ]
        
        # Add noise
        if random.random() < 0.1:
            noise = torch.randn_like(voxels) * 0.02
            voxels = voxels + noise
        
        # Random brightness/contrast
        if random.random() < 0.2 and C >= 3:  # RGB channels
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            
            # Apply to RGB channels only
            voxels[:, :3] = (voxels[:, :3] - 0.5) * contrast + 0.5 * brightness
            voxels[:, :3] = torch.clamp(voxels[:, :3], 0, 1)
        
        return voxels
    
    def generate_samples(self, num_samples: int = 4):
        """Generate sample worlds during training"""
        if self.text_encoder is None:
            return
        
        self.model.eval()
        
        with torch.no_grad():
            # Sample prompts
            prompts = [
                "A futuristic city with tall glass buildings",
                "A dense forest with ancient trees",
                "An underwater coral reef with colorful fish",
                "A medieval castle on a mountain"
            ][:num_samples]
            
            # Encode prompts
            text_embeddings = self.text_encoder(prompts).to(self.device)
            
            # Generate samples
            shape = (num_samples, self.config.latent_channels, 32, 32, 32)
            samples = self.model.sample(
                shape,
                guidance_scale=7.5,
                conditioning=text_embeddings,
                timesteps=50,
                device=self.device
            )
            
            # Convert to images for logging
            if self.config.local_rank == 0:
                self._log_samples(samples, prompts)
        
        self.model.train()
    
    def _log_samples(self, samples: torch.Tensor, prompts: List[str]):
        """Log generated samples to tensorboard/wandb"""
        # Take slices for visualization
        slices = []
        
        for i in range(min(4, samples.shape[0])):
            # Get middle slices
            depth_slice = samples[i, :3, samples.shape[2]//2, :, :]  # RGB, middle depth
            height_slice = samples[i, :3, :, samples.shape[3]//2, :]  # RGB, middle height
            width_slice = samples[i, :3, :, :, samples.shape[4]//2]  # RGB, middle width
            
            # Stack slices
            slice_img = torch.cat([depth_slice, height_slice, width_slice], dim=1)
            slices.append(slice_img)
        
        if slices:
            grid = torch.cat(slices, dim=2)  # Combine samples
            grid = torch.clamp(grid, 0, 1)
            
            # Convert to numpy for logging
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            
            # Log to tensorboard
            self.writer.add_image("samples", grid_np, self.global_step)
            
            # Log to wandb
            if self.config.log_to_wandb:
                wandb.log({
                    "samples": wandb.Image(grid_np, caption=", ".join(prompts[:4]))
                }, step=self.global_step)
    
    def _compute_psnr(self, mse_loss: float) -> float:
        """Compute PSNR from MSE loss"""
        if mse_loss <= 0:
            return 100.0
        return 20 * np.log10(1.0 / np.sqrt(mse_loss))
    
    def _compute_ssim_estimate(self, mse_loss: float) -> float:
        """Estimate SSIM from MSE (approximation)"""
        # This is a rough approximation
        sigma = 0.5
        return np.exp(-mse_loss / (2 * sigma**2))
    
    def _get_amp_dtype(self):
        """Get mixed precision dtype"""
        if self.config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    
    def log_step_metrics(self, metrics: Dict[str, float], phase: str = "train"):
        """Log metrics for current step"""
        if self.config.local_rank != 0:
            return
        
        # Add prefix for validation
        prefix = "" if phase == "train" else "val_"
        
        # Log to tensorboard
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}{key}", value, self.global_step)
        
        # Log to wandb
        if self.config.log_to_wandb:
            wandb_metrics = {f"{prefix}{key}": value for key, value in metrics.items()}
            wandb.log(wandb_metrics, step=self.global_step)
    
    def log_epoch_metrics(self, epoch: int, epoch_loss: float):
        """Log metrics for completed epoch"""
        if self.config.local_rank != 0:
            return
        
        # Compute epoch time
        if len(self.epoch_times) > 0:
            avg_epoch_time = np.mean(self.epoch_times)
            eta = avg_epoch_time * (self.config.epochs - epoch - 1)
            eta_str = str(timedelta(seconds=int(eta)))
        else:
            eta_str = "Calculating..."
        
        # Log to console
        logger.info(
            f"Epoch {epoch:04d} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Best Val Loss: {self.best_val_loss:.6f} | "
            f"ETA: {eta_str}"
        )
        
        # Log to tensorboard
        self.writer.add_scalar("epoch_loss", epoch_loss, epoch)
        self.writer.add_scalar("best_val_loss", self.best_val_loss, epoch)
        
        # Log to wandb
        if self.config.log_to_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "best_val_loss": self.best_val_loss
            })
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        if self.config.local_rank != 0:
            return
        
        # Prepare checkpoint data
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save using checkpoint manager
        save_path = self.checkpoint_manager.save_checkpoint(
            checkpoint,
            name=name,
            is_best=(name == "best_model")
        )
        
        logger.info(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint"""
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load scaler state
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def cleanup(self):
        """Cleanup training resources"""
        if self.is_distributed:
            dist.destroy_process_group()
        
        if self.config.local_rank == 0:
            self.writer.close()
            if self.config.log_to_wandb:
                wandb.finish()
        
        logger.info("Training cleanup completed")

class MultiGPUTrainer(WorldModelTrainer):
    """Trainer with multi-GPU optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional multi-GPU optimizations
        if self.is_distributed:
            self._setup_multi_gpu_optimizations()
    
    def _setup_multi_gpu_optimizations(self):
        """Setup multi-GPU specific optimizations"""
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'module'):
            self.model.module.enable_gradient_checkpointing()
        else:
            self.model.enable_gradient_checkpointing()
        
        # Set different gradient accumulation for each GPU based on memory
        gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        
        if gpu_memory < 16:  # Less than 16GB
            self.config.gradient_accumulation_steps = max(4, self.config.gradient_accumulation_steps)
            logger.info(f"Low GPU memory ({gpu_memory:.1f}GB), increasing gradient accumulation to {self.config.gradient_accumulation_steps}")
        
        # Use async data loading
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    def train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Optimized training step for multi-GPU"""
        # Unpack batch
        x = batch["voxels"]
        
        # Distributed: split batch across GPUs
        if self.is_distributed:
            batch_size = x.shape[0]
            local_batch_size = batch_size // self.config.world_size
            start_idx = self.config.local_rank * local_batch_size
            end_idx = start_idx + local_batch_size
            
            x = x[start_idx:end_idx]
            if "text_embeddings" in batch:
                batch["text_embeddings"] = batch["text_embeddings"][start_idx:end_idx]
            if "images" in batch:
                batch["images"] = batch["images"][start_idx:end_idx]
        
        return super().train_step(batch, batch_idx)

class CurriculumTrainer(WorldModelTrainer):
    """Trainer with curriculum learning"""
    
    def __init__(self, *args, curriculum_config: Optional[Dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Curriculum learning configuration
        self.curriculum_config = curriculum_config or {
            "stages": [
                {"epochs": 50, "resolution": (16, 16, 16), "lr_multiplier": 0.5},
                {"epochs": 100, "resolution": (32, 32, 32), "lr_multiplier": 1.0},
                {"epochs": 150, "resolution": (64, 64, 64), "lr_multiplier": 0.5},
                {"epochs": float('inf'), "resolution": (128, 128, 128), "lr_multiplier": 0.25}
            ]
        }
        
        self.current_stage = 0
        self.stage_start_epoch = 0
        
    def train_epoch(self) -> float:
        """Train epoch with curriculum learning"""
        # Update curriculum stage if needed
        self._update_curriculum_stage()
        
        # Adjust learning rate for current stage
        self._adjust_learning_rate()
        
        # Train epoch
        return super().train_epoch()
    
    def _update_curriculum_stage(self):
        """Update current curriculum stage"""
        for i, stage in enumerate(self.curriculum_config["stages"]):
            if self.current_epoch - self.stage_start_epoch < stage["epochs"]:
                if i != self.current_stage:
                    self.current_stage = i
                    self.stage_start_epoch = self.current_epoch
                    logger.info(f"Entering curriculum stage {i}: resolution {stage['resolution']}")
                break
    
    def _adjust_learning_rate(self):
        """Adjust learning rate based on curriculum stage"""
        stage = self.curriculum_config["stages"][self.current_stage]
        lr_multiplier = stage.get("lr_multiplier", 1.0)
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.config.learning_rate * lr_multiplier
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare batch with curriculum resolution"""
        batch = super()._prepare_batch(batch)
        
        if "voxels" in batch:
            # Get target resolution for current stage
            stage = self.curriculum_config["stages"][self.current_stage]
            target_res = stage["resolution"]
            
            # Resize voxels to target resolution
            voxels = batch["voxels"]
            if voxels.shape[2:] != target_res:
                voxels = F.interpolate(
                    voxels,
                    size=target_res,
                    mode='trilinear',
                    align_corners=False
                )
                batch["voxels"] = voxels
        
        return batch