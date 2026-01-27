"""
Optimizer for Gaussian Splatting model.
Implements specialized optimization strategies for 3D Gaussians.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import json


@dataclass
class OptimizerConfig:
    """Configuration for Gaussian optimizer."""
    
    # Learning rates
    lr_position: float = 0.00016
    lr_rotation: float = 0.001
    lr_scale: float = 0.005
    lr_opacity: float = 0.05
    lr_sh: float = 0.0025
    lr_feature: float = 0.0025
    
    # Learning rate scheduling
    lr_scheduler: str = 'exponential'  # 'exponential', 'cosine', 'step', 'plateau'
    lr_decay_rate: float = 0.99
    lr_decay_steps: int = 1000
    warmup_steps: int = 100
    
    # Optimization parameters
    max_iterations: int = 30000
    iterations_per_density_update: int = 100
    iterations_per_prune: int = 1000
    iterations_per_save: int = 1000
    
    # Density control
    opacity_reset_interval: int = 3000
    densification_interval: int = 100
    prune_interval: int = 1000
    
    # Gradient thresholds
    position_lr_max: float = 0.00016
    position_lr_min: float = 1.6e-06
    scaling_lr_max: float = 0.005
    scaling_lr_min: float = 5e-06
    
    # Regularization
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    
    # Adaptive learning rates
    adaptive_lr: bool = True
    lr_scale_factor: float = 0.01
    
    # Monitoring
    log_interval: int = 100
    vis_interval: int = 1000
    save_interval: int = 1000
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    keep_checkpoints: int = 5
    
    # Early stopping
    patience: int = 1000
    min_delta: float = 1e-6
    
    # Debug
    debug: bool = False
    profile: bool = False


class GaussianOptimizer:
    """
    Optimizer for Gaussian Splatting model.
    
    Implements:
    - Separate learning rates for different parameters
    - Adaptive density control
    - Learning rate scheduling
    - Checkpointing and resuming
    - Loss monitoring and visualization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[OptimizerConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.config = config or OptimizerConfig()
        self.device = device
        
        # Initialize optimizers for different parameter groups
        self.optimizer = None
        self.scheduler = None
        
        # Tracking
        self.iteration = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.loss_history = []
        self.lr_history = []
        self.gaussian_count_history = []
        self.timing_history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize
        self._init_optimizer()
        self._init_scheduler()
        
        # Profiling
        self.profiler = None
        if self.config.profile:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
    
    def _init_optimizer(self):
        """Initialize optimizer with separate learning rates."""
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': [self.model.positions],
                'lr': self.config.lr_position,
                'name': 'position'
            },
            {
                'params': [self.model.rotations],
                'lr': self.config.lr_rotation,
                'name': 'rotation'
            },
            {
                'params': [self.model.scales],
                'lr': self.config.lr_scale,
                'name': 'scale'
            },
            {
                'params': [self.model.opacities],
                'lr': self.config.lr_opacity,
                'name': 'opacity'
            },
            {
                'params': [self.model.sh_coeffs],
                'lr': self.config.lr_sh,
                'name': 'sh'
            }
        ]
        
        # Add features if present
        if hasattr(self.model, 'features') and self.model.features is not None:
            param_groups.append({
                'params': [self.model.features],
                'lr': self.config.lr_feature,
                'name': 'feature'
            })
        
        # Create optimizer
        self.optimizer = optim.Adam(
            param_groups,
            lr=self.config.lr_position,  # Base LR, will be overridden
            weight_decay=self.config.weight_decay
        )
        
        print(f"Initialized optimizer with {len(param_groups)} parameter groups")
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.lr_scheduler == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.lr_decay_rate
            )
        elif self.config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_iterations
            )
        elif self.config.lr_scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_steps,
                gamma=self.config.lr_decay_rate
            )
        elif self.config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=100,
                verbose=True
            )
        else:
            self.scheduler = None
    
    def compute_adaptive_lr(self, grad_norm: float, param_name: str) -> float:
        """
        Compute adaptive learning rate based on gradient statistics.
        
        Args:
            grad_norm: Gradient norm
            param_name: Parameter name
            
        Returns:
            Adaptive learning rate
        """
        if not self.config.adaptive_lr:
            return getattr(self.config, f'lr_{param_name}')
        
        # Get base LR
        base_lr = getattr(self.config, f'lr_{param_name}')
        
        # Adaptive scaling based on gradient
        if grad_norm > 0:
            adaptive_factor = self.config.lr_scale_factor / (grad_norm + 1e-8)
            adaptive_factor = np.clip(adaptive_factor, 0.1, 10.0)
            return base_lr * adaptive_factor
        else:
            return base_lr
    
    def update_learning_rates(self):
        """Update learning rates adaptively based on gradient statistics."""
        if not self.config.adaptive_lr:
            return
        
        for param_group in self.optimizer.param_groups:
            param_name = param_group.get('name', 'unknown')
            params = param_group['params'][0]
            
            if params.grad is not None:
                grad_norm = params.grad.norm().item()
                new_lr = self.compute_adaptive_lr(grad_norm, param_name)
                
                # Clamp learning rate
                if param_name == 'position':
                    new_lr = np.clip(new_lr, self.config.position_lr_min, self.config.position_lr_max)
                elif param_name == 'scale':
                    new_lr = np.clip(new_lr, self.config.scaling_lr_min, self.config.scaling_lr_max)
                
                param_group['lr'] = new_lr
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization losses.
        
        Returns:
            Total regularization loss
        """
        reg_loss = torch.tensor(0.0, device=self.device)
        
        # Scale regularization (prevent too large/small scales)
        scales = torch.exp(self.model.scales[self.model.active_mask])
        scale_reg = torch.mean(torch.abs(scales - 0.1))
        reg_loss += 0.01 * scale_reg
        
        # Opacity regularization (encourage reasonable opacities)
        opacities = torch.sigmoid(self.model.opacities[self.model.active_mask])
        opacity_reg = torch.mean((opacities - 0.5) ** 2)
        reg_loss += 0.001 * opacity_reg
        
        # Rotation regularization (keep quaternions normalized)
        rotations = self.model.rotations[self.model.active_mask]
        quat_norm = torch.norm(rotations, dim=1)
        quat_reg = torch.mean((quat_norm - 1.0) ** 2)
        reg_loss += 0.01 * quat_reg
        
        return reg_loss
    
    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        Compute sparsity loss to encourage compact representation.
        
        Returns:
            Sparsity loss
        """
        if not self.model.active_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        # Encourage fewer Gaussians by penalizing total "presence"
        opacities = torch.sigmoid(self.model.opacities[self.model.active_mask])
        sparsity_loss = torch.mean(opacities)
        
        return 0.001 * sparsity_loss
    
    def gradient_clipping(self):
        """Apply gradient clipping."""
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
    
    def density_control_step(self):
        """Perform density control (clone, split, prune)."""
        if (self.iteration % self.config.densification_interval == 0 and 
            self.iteration > self.config.warmup_steps):
            
            # Get gradient statistics
            grad_norms = {}
            for param_group in self.optimizer.param_groups:
                param_name = param_group.get('name', 'unknown')
                params = param_group['params'][0]
                
                if params.grad is not None:
                    grad_norms[param_name] = params.grad.norm().item()
            
            # Perform density control
            self.model.density_control(
                grad_threshold=0.0002,
                density_threshold=0.01,
                max_grad=0.5,
                scene_extent=1.0
            )
            
            # Reset optimizer for new parameters
            self._init_optimizer()
    
    def prune_step(self):
        """Prune Gaussians with low opacity."""
        if self.iteration % self.config.prune_interval == 0:
            with torch.no_grad():
                opacities = torch.sigmoid(self.model.opacities)
                prune_mask = (opacities.squeeze() < 0.01) & self.model.active_mask
                
                if torch.any(prune_mask):
                    self.model.active_mask[prune_mask] = False
                    
                    # Reset gradients for pruned Gaussians
                    if hasattr(self.model, 'grad_accum'):
                        self.model.grad_accum[prune_mask] = 0.0
                    
                    print(f"Pruned {prune_mask.sum().item()} Gaussians at iteration {self.iteration}")
    
    def reset_opacity(self):
        """Reset opacity for under-reconstructed areas."""
        if self.iteration % self.config.opacity_reset_interval == 0:
            with torch.no_grad():
                # Reset opacity for all active Gaussians
                self.model.opacities[self.model.active_mask] = 0.0
                print(f"Reset opacity at iteration {self.iteration}")
    
    def step(
        self,
        loss: torch.Tensor,
        loss_components: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Perform optimization step.
        
        Args:
            loss: Total loss
            loss_components: Optional dictionary of loss components
            
        Returns:
            Dictionary with loss information
        """
        start_time = time.time()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        self.gradient_clipping()
        
        # Update learning rates adaptively
        self.update_learning_rates()
        
        # Optimization step
        self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(loss.item())
            else:
                self.scheduler.step()
        
        # Density control
        self.density_control_step()
        
        # Prune
        self.prune_step()
        
        # Reset opacity periodically
        self.reset_opacity()
        
        # Update iteration
        self.iteration += 1
        
        # Record timing
        step_time = time.time() - start_time
        self.timing_history.append(step_time)
        
        # Prepare return dictionary
        result = {
            'total_loss': loss.item(),
            'iteration': self.iteration,
            'step_time': step_time,
            'num_gaussians': self.model.active_mask.sum().item()
        }
        
        # Add loss components
        if loss_components is not None:
            for name, component in loss_components.items():
                result[name] = component.item()
        
        # Add learning rates
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_name = param_group.get('name', f'group_{i}')
            result[f'lr_{param_name}'] = param_group['lr']
        
        # Logging
        if self.iteration % self.config.log_interval == 0:
            self._log_step(result)
        
        # Visualization
        if self.iteration % self.config.vis_interval == 0:
            self._visualize_step()
        
        # Checkpointing
        if self.iteration % self.config.save_interval == 0:
            self.save_checkpoint()
        
        # Early stopping check
        if self._check_early_stopping(loss.item()):
            print(f"Early stopping triggered at iteration {self.iteration}")
        
        # Update profiler
        if self.profiler is not None:
            self.profiler.step()
        
        return result
    
    def _log_step(self, result: Dict[str, float]):
        """Log optimization step."""
        log_str = f"Iteration {self.iteration:06d}: "
        log_str += f"Loss = {result['total_loss']:.6f}, "
        log_str += f"Gaussians = {result['num_gaussians']}, "
        log_str += f"Time = {result['step_time']:.3f}s"
        
        # Add loss components if available
        for key, value in result.items():
            if key.startswith('loss_'):
                log_str += f", {key} = {value:.6f}"
        
        print(log_str)
        
        # Update history
        self.loss_history.append(result['total_loss'])
        self.gaussian_count_history.append(result['num_gaussians'])
        
        # Save learning rates
        current_lrs = {}
        for param_group in self.optimizer.param_groups:
            param_name = param_group.get('name', 'unknown')
            current_lrs[param_name] = param_group['lr']
        self.lr_history.append(current_lrs)
    
    def _visualize_step(self):
        """Create visualization of optimization progress."""
        if len(self.loss_history) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curve
        axes[0, 0].plot(self.loss_history)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gaussian count
        axes[0, 1].plot(self.gaussian_count_history)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Number of Gaussians')
        axes[0, 1].set_title('Gaussian Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rates
        if self.lr_history:
            lr_data = {}
            for lr_dict in self.lr_history:
                for key, value in lr_dict.items():
                    if key not in lr_data:
                        lr_data[key] = []
                    lr_data[key].append(value)
            
            for key, values in lr_data.items():
                axes[1, 0].plot(values, label=key)
            
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rates')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Step time
        if self.timing_history:
            axes[1, 1].plot(self.timing_history)
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].set_title('Step Time')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        vis_path = self.checkpoint_dir / f'progress_iter_{self.iteration:06d}.png'
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Save data as JSON
        data_path = self.checkpoint_dir / f'progress_iter_{self.iteration:06d}.json'
        with open(data_path, 'w') as f:
            json.dump({
                'iteration': self.iteration,
                'loss_history': self.loss_history,
                'gaussian_count_history': self.gaussian_count_history,
                'lr_history': self.lr_history,
                'timing_history': self.timing_history
            }, f, indent=2)
    
    def _check_early_stopping(self, current_loss: float) -> bool:
        """
        Check if early stopping criteria are met.
        
        Args:
            current_loss: Current loss value
            
        Returns:
            True if should stop early
        """
        if current_loss < self.best_loss - self.config.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience
    
    def save_checkpoint(self, name: Optional[str] = None):
        """
        Save checkpoint.
        
        Args:
            name: Optional checkpoint name
        """
        if name is None:
            name = f'checkpoint_iter_{self.iteration:06d}.pt'
        
        checkpoint_path = self.checkpoint_dir / name
        
        # Prepare checkpoint data
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter,
            'loss_history': self.loss_history,
            'lr_history': self.lr_history,
            'gaussian_count_history': self.gaussian_count_history,
            'timing_history': self.timing_history,
            'config': self.config.__dict__
        }
        
        # Save
        torch.save(checkpoint, checkpoint_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_iter_*.pt'))
        checkpoints.sort()
        
        # Keep only the most recent N checkpoints
        if len(checkpoints) > self.config.keep_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_checkpoints]:
                checkpoint.unlink()
    
    def load_checkpoint(self, path: str) -> bool:
        """
        Load checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            True if successful
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.iteration = checkpoint['iteration']
            self.best_loss = checkpoint['best_loss']
            self.patience_counter = checkpoint['patience_counter']
            
            # Load history
            self.loss_history = checkpoint.get('loss_history', [])
            self.lr_history = checkpoint.get('lr_history', [])
            self.gaussian_count_history = checkpoint.get('gaussian_count_history', [])
            self.timing_history = checkpoint.get('timing_history', [])
            
            print(f"Loaded checkpoint from {path} at iteration {self.iteration}")
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint from {path}: {e}")
            return False
    
    def train(
        self,
        loss_fn: Callable[[], Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
        max_iterations: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            loss_fn: Function that returns (loss, loss_components)
            max_iterations: Maximum number of iterations
        """
        max_iterations = max_iterations or self.config.max_iterations
        
        print(f"Starting training for {max_iterations} iterations")
        print(f"Initial Gaussian count: {self.model.active_mask.sum().item()}")
        
        # Start profiler if enabled
        if self.profiler is not None:
            self.profiler.start()
        
        try:
            while self.iteration < max_iterations:
                # Compute loss
                loss, loss_components = loss_fn()
                
                # Optimization step
                step_result = self.step(loss, loss_components)
                
                # Check for early stopping
                if self._check_early_stopping(step_result['total_loss']):
                    print(f"Early stopping at iteration {self.iteration}")
                    break
                
        except KeyboardInterrupt:
            print("Training interrupted by user")
        
        finally:
            # Stop profiler if enabled
            if self.profiler is not None:
                self.profiler.stop()
            
            # Save final checkpoint
            self.save_checkpoint('final_checkpoint.pt')
            
            # Save final visualization
            self._visualize_step()
            
            print(f"Training completed at iteration {self.iteration}")
            print(f"Final Gaussian count: {self.model.active_mask.sum().item()}")
            print(f"Final loss: {self.loss_history[-1] if self.loss_history else 'N/A'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.loss_history:
            return {}
        
        stats = {
            'total_iterations': self.iteration,
            'final_loss': self.loss_history[-1] if self.loss_history else float('inf'),
            'best_loss': self.best_loss,
            'final_gaussian_count': self.model.active_mask.sum().item(),
            'avg_step_time': np.mean(self.timing_history) if self.timing_history else 0,
            'total_training_time': np.sum(self.timing_history) if self.timing_history else 0,
            'converged': self.patience_counter >= self.config.patience
        }
        
        # Loss reduction
        if len(self.loss_history) > 1:
            stats['initial_loss'] = self.loss_history[0]
            stats['loss_reduction'] = (self.loss_history[0] - stats['final_loss']) / self.loss_history[0]
        
        return stats
    
    def create_summary_report(self) -> str:
        """Create a summary report of the optimization."""
        stats = self.get_statistics()
        
        report = "=" * 80 + "\n"
        report += "GAUSSIAN SPLATTING OPTIMIZATION SUMMARY\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Total Iterations: {stats.get('total_iterations', 0)}\n"
        report += f"Final Loss: {stats.get('final_loss', 0):.6f}\n"
        report += f"Best Loss: {stats.get('best_loss', 0):.6f}\n"
        report += f"Final Gaussian Count: {stats.get('final_gaussian_count', 0)}\n"
        report += f"Average Step Time: {stats.get('avg_step_time', 0):.3f}s\n"
        report += f"Total Training Time: {stats.get('total_training_time', 0):.1f}s\n"
        
        if 'loss_reduction' in stats:
            report += f"Loss Reduction: {stats['loss_reduction']*100:.1f}%\n"
        
        report += f"Converged: {stats.get('converged', False)}\n\n"
        
        report += "Learning Rate History:\n"
        if self.lr_history and len(self.lr_history) > 0:
            latest_lrs = self.lr_history[-1]
            for param_name, lr in latest_lrs.items():
                report += f"  {param_name}: {lr:.2e}\n"
        
        report += "\n" + "=" * 80 + "\n"
        
        return report