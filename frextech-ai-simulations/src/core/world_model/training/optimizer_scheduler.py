"""
Optimizers and Schedulers for 3D World Model Training
Advanced optimization strategies for large-scale 3D model training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LambdaLR, StepLR, MultiStepLR, ExponentialLR,
    CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
)
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = True
    amsgrad: bool = False
    centered: bool = False
    
    # Learning rate scheduler
    scheduler_type: str = "cosine_warmup"
    warmup_steps: int = 1000
    total_steps: int = 1000000
    min_lr: float = 1e-6
    decay_rate: float = 0.9
    decay_steps: int = 10000
    cycle_length: int = 1000
    cycle_mult: float = 1.0
    
    # Gradient clipping
    gradient_clip: float = 1.0
    gradient_clip_type: str = "norm"  # "norm", "value", "adaptive"
    
    # Mixed precision
    use_amp: bool = True
    
    # Layer-wise learning rate
    layerwise_lr: bool = False
    layerwise_decay: float = 0.95

class OptimizerManager:
    """Manages optimizer creation and configuration"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.param_groups = []
        
    def create_optimizer(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        optimizer_type: str = "adamw",
        **kwargs
    ) -> optim.Optimizer:
        """Create optimizer with parameter groups"""
        
        # Get parameter groups
        self._create_parameter_groups(lr, weight_decay)
        
        # Create optimizer
        if optimizer_type == "adamw":
            optimizer = optim.AdamW(
                self.param_groups,
                lr=lr,
                betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
                eps=kwargs.get('epsilon', 1e-8),
                weight_decay=weight_decay,
                amsgrad=kwargs.get('amsgrad', False)
            )
        elif optimizer_type == "adam":
            optimizer = optim.Adam(
                self.param_groups,
                lr=lr,
                betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
                eps=kwargs.get('epsilon', 1e-8),
                weight_decay=weight_decay,
                amsgrad=kwargs.get('amsgrad', False)
            )
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                self.param_groups,
                lr=lr,
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=kwargs.get('nesterov', True)
            )
        elif optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(
                self.param_groups,
                lr=lr,
                alpha=kwargs.get('alpha', 0.99),
                eps=kwargs.get('epsilon', 1e-8),
                weight_decay=weight_decay,
                momentum=kwargs.get('momentum', 0.9),
                centered=kwargs.get('centered', False)
            )
        elif optimizer_type == "adagrad":
            optimizer = optim.Adagrad(
                self.param_groups,
                lr=lr,
                lr_decay=kwargs.get('lr_decay', 0),
                weight_decay=weight_decay,
                eps=kwargs.get('epsilon', 1e-10)
            )
        elif optimizer_type == "lion":
            optimizer = Lion(
                self.param_groups,
                lr=lr,
                betas=(kwargs.get('beta1', 0.9), kwargs.get('beta2', 0.999)),
                weight_decay=weight_decay
            )
        elif optimizer_type == "adafactor":
            optimizer = Adafactor(
                self.param_groups,
                lr=lr,
                eps=kwargs.get('eps', (1e-30, 1e-3)),
                clip_threshold=kwargs.get('clip_threshold', 1.0),
                decay_rate=kwargs.get('decay_rate', -0.8),
                beta1=kwargs.get('beta1', None),
                weight_decay=kwargs.get('weight_decay', 0.0),
                scale_parameter=kwargs.get('scale_parameter', True),
                relative_step=kwargs.get('relative_step', True),
                warmup_init=kwargs.get('warmup_init', False)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizer
    
    def _create_parameter_groups(self, lr: float, weight_decay: float):
        """Create parameter groups with different learning rates"""
        self.param_groups = []
        
        # Layer-wise parameter grouping
        if hasattr(self.model, 'named_parameters'):
            params_dict = {}
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Determine parameter group based on layer type
                if 'text_encoder' in name:
                    group_name = 'text_encoder'
                    group_lr = lr * 0.1  # Lower LR for text encoder
                elif 'vision_encoder' in name:
                    group_name = 'vision_encoder'
                    group_lr = lr * 0.1  # Lower LR for vision encoder
                elif 'unet' in name:
                    # Different LR for different parts of UNet
                    if 'input_conv' in name or 'output_conv' in name:
                        group_name = 'unet_io'
                        group_lr = lr
                    elif 'down_blocks' in name:
                        group_name = 'unet_down'
                        group_lr = lr
                    elif 'up_blocks' in name:
                        group_name = 'unet_up'
                        group_lr = lr
                    elif 'middle_block' in name:
                        group_name = 'unet_middle'
                        group_lr = lr
                    else:
                        group_name = 'unet_other'
                        group_lr = lr
                elif 'positional' in name or 'embedding' in name:
                    group_name = 'embeddings'
                    group_lr = lr * 2.0  # Higher LR for embeddings
                else:
                    group_name = 'other'
                    group_lr = lr
                
                # Add to parameter group
                if group_name not in params_dict:
                    params_dict[group_name] = {
                        'params': [],
                        'lr': group_lr,
                        'weight_decay': weight_decay if 'bias' not in name else 0.0
                    }
                
                params_dict[group_name]['params'].append(param)
            
            # Convert to list
            self.param_groups = list(params_dict.values())
        else:
            # Simple parameter grouping
            self.param_groups = [
                {
                    'params': [p for p in self.model.parameters() if p.requires_grad],
                    'lr': lr,
                    'weight_decay': weight_decay
                }
            ]
    
    def apply_gradient_clipping(
        self,
        optimizer: optim.Optimizer,
        gradient_clip: float,
        clip_type: str = "norm"
    ):
        """Apply gradient clipping"""
        
        def clip_gradients():
            if clip_type == "norm":
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'],
                    gradient_clip
                )
            elif clip_type == "value":
                torch.nn.utils.clip_grad_value_(
                    optimizer.param_groups[0]['params'],
                    gradient_clip
                )
            elif clip_type == "adaptive":
                self._adaptive_gradient_clipping(optimizer, gradient_clip)
            else:
                raise ValueError(f"Unknown clip type: {clip_type}")
        
        return clip_gradients
    
    def _adaptive_gradient_clipping(self, optimizer: optim.Optimizer, max_norm: float):
        """Adaptive gradient clipping based on parameter norms"""
        total_norm = 0.0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.data.norm(2)
                    grad_norm = p.grad.data.norm(2)
                    
                    # Clip if gradient norm is much larger than parameter norm
                    if grad_norm > max_norm * param_norm:
                        clip_coef = max_norm * param_norm / (grad_norm + 1e-6)
                        p.grad.data.mul_(clip_coef)
                    
                    total_norm += grad_norm.item() ** 2
        
        return total_norm ** 0.5

def get_scheduler(
    scheduler_type: str,
    optimizer: optim.Optimizer,
    warmup_steps: int = 1000,
    total_steps: int = 1000000,
    **kwargs
) -> Optional[object]:
    """Get learning rate scheduler"""
    
    if scheduler_type == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    
    elif scheduler_type == "linear_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == "cosine_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    elif scheduler_type == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10000),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type == "multi_step":
        return MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [30000, 60000, 90000]),
            gamma=kwargs.get('gamma', 0.1)
        )
    
    elif scheduler_type == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.9999)
        )
    
    elif scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == "cyclic":
        return CyclicLR(
            optimizer,
            base_lr=kwargs.get('base_lr', 1e-5),
            max_lr=kwargs.get('max_lr', 1e-3),
            step_size_up=kwargs.get('step_size_up', 2000),
            mode=kwargs.get('mode', 'triangular')
        )
    
    elif scheduler_type == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=total_steps,
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos')
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class Lion(optim.Optimizer):
    """
    Lion optimizer from "Symbolic Discovery of Optimization Algorithms"
    https://arxiv.org/abs/2302.06675
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update parameters
                update = exp_avg.sign().mul_(group['lr'])
                p.add_(-update)
        
        return loss

class Adafactor(optim.Optimizer):
    """
    Adafactor optimizer from "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    https://arxiv.org/abs/1804.04235
    """
    
    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")
        
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored = len(grad_shape) >= 2
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=grad.device)
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=grad.device)
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad, dtype=torch.float32)
                    
                    if group['beta1'] is not None:
                        state['exp_avg'] = torch.zeros_like(grad, dtype=torch.float32)
                
                state['step'] += 1
                lr = self._get_lr(group, state)
                beta2 = 1.0 - group['decay_rate'] ** state['step']
                
                if factored:
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    # Update row and column moving averages
                    exp_avg_sq_row.mul_(beta2).add_(
                        grad.pow(2).mean(dim=-1), alpha=1 - beta2
                    )
                    exp_avg_sq_col.mul_(beta2).add_(
                        grad.pow(2).mean(dim=-2), alpha=1 - beta2
                    )
                    
                    # Compute RMS
                    row_rms = exp_avg_sq_row.rsqrt().unsqueeze(-1)
                    col_rms = exp_avg_sq_col.rsqrt().unsqueeze(-2)
                    denom = row_rms.mul(col_rms).clamp(max=group['clip_threshold'])
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
                    denom = exp_avg_sq.rsqrt().clamp(max=group['clip_threshold'])
                
                update = grad * denom
                
                if group['beta1'] is not None:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg
                
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - lr * group['weight_decay'])
                
                update.mul_(lr)
                
                if group['scale_parameter']:
                    param_rms = p.data.pow(2).mean().sqrt().clamp(min=group['eps'][0])
                    update.div_(param_rms.clamp(min=group['eps'][1]))
                
                p.data.add_(-update)
        
        return loss
    
    def _get_lr(self, group, state):
        """Get learning rate for current step"""
        if group['relative_step']:
            min_step = 1e-6 * state['step'] ** -0.5 if group['warmup_init'] else 1e-2
            lr = min(min_step, state['step'] ** -0.5)
        else:
            lr = group['lr']
        
        return lr

class OneCycleLR:
    """One-cycle learning rate scheduler"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.step_num = 0
        self.base_lr = max_lr / div_factor
        self.final_lr = self.base_lr / final_div_factor
        
        # Phase lengths
        self.phase1_steps = int(total_steps * pct_start)
        self.phase2_steps = total_steps - self.phase1_steps
    
    def step(self):
        """Update learning rate"""
        self.step_num += 1
        
        if self.step_num <= self.phase1_steps:
            # Phase 1: Increase from base_lr to max_lr
            progress = self.step_num / self.phase1_steps
            lr = self._annealing(progress, self.base_lr, self.max_lr)
        else:
            # Phase 2: Decrease from max_lr to final_lr
            progress = (self.step_num - self.phase1_steps) / self.phase2_steps
            lr = self._annealing(progress, self.max_lr, self.final_lr)
        
        # Update optimizer learning rates
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _annealing(self, progress: float, start: float, end: float) -> float:
        """Compute annealed value"""
        if self.anneal_strategy == 'linear':
            return start + (end - start) * progress
        elif self.anneal_strategy == 'cos':
            return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2
        else:
            raise ValueError(f"Unknown anneal strategy: {self.anneal_strategy}")

class Lookahead(optim.Optimizer):
    """
    Lookahead optimizer from "Lookahead Optimizer: k steps forward, 1 step back"
    https://arxiv.org/abs/1907.08610
    """
    
    def __init__(self, optimizer, k: int = 5, alpha: float = 0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
    
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = self.optimizer.step(closure)
        
        for group in self.param_groups:
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
                
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    
                    param_state = self.state[p]
                    if "slow_param" not in param_state:
                        param_state["slow_param"] = p.data.clone()
                    
                    slow = param_state["slow_param"]
                    p.data.mul_(self.alpha).add_(slow, alpha=1.0 - self.alpha)
                    slow.copy_(p.data)
        
        return loss
    
    def zero_grad(self):
        """Clears the gradients of all optimized parameters"""
        self.optimizer.zero_grad()

class GradientAccumulator:
    """Gradient accumulation for large batch training"""
    
    def __init__(self, model: nn.Module, accumulation_steps: int = 1):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
        
    def accumulate(self, loss: torch.Tensor):
        """Accumulate gradients"""
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.step_counter += 1
        
        if self.step_counter % self.accumulation_steps == 0:
            return True  # Ready for optimizer step
        return False
    
    def zero_grad(self):
        """Zero gradients or prepare for accumulation"""
        if self.step_counter % self.accumulation_steps == 0:
            self.model.zero_grad()

class SAM(optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer
    https://arxiv.org/abs/2010.01412
    """
    
    def __init__(self, params, base_optimizer, rho: float = 0.05, adaptive: bool = False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """First step: compute gradient at current point"""
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                
                if group['adaptive']:
                    e_w *= p.data.norm() / (p.grad.norm() + 1e-12)
                
                p.add_(e_w)  # climb to the local maximum
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Second step: update parameters from sharp point"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                p.data = self.state[p]["old_p"]  # get back to original point
        
        self.base_optimizer.step()  # do the actual update
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        """Combined step for SAM"""
        raise NotImplementedError("SAM doesn't work like other optimizers!")
    
    def _grad_norm(self):
        """Compute gradient norm"""
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

class LearningRateFinder:
    """Learning rate finder for optimal learning rate selection"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion,
        device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        self.losses = []
        self.lrs = []
    
    def range_test(
        self,
        train_loader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5
    ):
        """Perform learning rate range test"""
        self.model.train()
        
        # Save original state
        original_state = {
            'model': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict())
        }
        
        # Set up learning rate scheduler
        lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # Initialize
        avg_loss = 0.0
        best_loss = float('inf')
        
        for iteration, batch in enumerate(train_loader):
            if iteration >= num_iter:
                break
            
            # Get learning rate
            lr = scheduler.get_last_lr()[0]
            self.lrs.append(lr)
            
            # Training step
            batch = self._prepare_batch(batch)
            loss = self._training_step(batch)
            
            # Compute smoothed loss
            avg_loss = smooth_f * loss + (1 - smooth_f) * avg_loss
            smoothed_loss = avg_loss / (1 - (1 - smooth_f) ** (iteration + 1))
            
            self.losses.append(smoothed_loss)
            
            # Check for divergence
            if smoothed_loss > diverge_th * best_loss:
                print(f"Divergence detected at LR {lr:.2e}")
                break
            
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Update learning rate
            scheduler.step()
        
        # Restore original state
        self.model.load_state_dict(original_state['model'])
        self.optimizer.load_state_dict(original_state['optimizer'])
        
        return self.lrs, self.losses
    
    def _prepare_batch(self, batch):
        """Prepare batch for training"""
        if isinstance(batch, (list, tuple)):
            return [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
        elif torch.is_tensor(batch):
            return batch.to(self.device)
        else:
            raise TypeError(f"Batch type {type(batch)} not supported")
    
    def _training_step(self, batch):
        """Single training step"""
        # Forward pass
        if isinstance(batch, (list, tuple)):
            output = self.model(*batch[:-1])
            target = batch[-1]
        else:
            output = self.model(batch)
            target = batch  # Assuming self-supervised
        
        # Compute loss
        loss = self.criterion(output, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def find_optimal_lr(self, skip_start: int = 10, skip_end: int = 5):
        """Find optimal learning rate from range test results"""
        if len(self.losses) == 0:
            raise ValueError("Run range_test first")
        
        # Skip beginning and end
        losses = self.losses[skip_start:-skip_end] if skip_end > 0 else self.losses[skip_start:]
        lrs = self.lrs[skip_start:-skip_end] if skip_end > 0 else self.lrs[skip_start:]
        
        # Find minimum gradient point
        gradients = np.gradient(losses)
        min_grad_idx = np.argmin(gradients)
        
        optimal_lr = lrs[min_grad_idx]
        
        return optimal_lr, min_grad_idx