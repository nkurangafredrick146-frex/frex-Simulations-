"""
Loss Functions for 3D World Model Training
Specialized loss functions for 3D scene generation and reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import lpips
from kornia.losses import SSIMLoss, TotalVariation

class WorldLoss(nn.Module):
    """Combined loss function for 3D world generation"""
    
    def __init__(
        self,
        diffusion_loss_type: str = "l2",
        perceptual_weight: float = 0.1,
        consistency_weight: float = 0.01,
        kl_weight: float = 0.0001,
        style_weight: float = 0.0,
        temporal_weight: float = 0.0,
        enable_progressive: bool = True
    ):
        super().__init__()
        
        self.diffusion_loss_type = diffusion_loss_type
        self.perceptual_weight = perceptual_weight
        self.consistency_weight = consistency_weight
        self.kl_weight = kl_weight
        self.style_weight = style_weight
        self.temporal_weight = temporal_weight
        
        # Diffusion loss
        if diffusion_loss_type == "l1":
            self.diffusion_loss = nn.L1Loss(reduction='mean')
        elif diffusion_loss_type == "l2":
            self.diffusion_loss = nn.MSELoss(reduction='mean')
        elif diffusion_loss_type == "huber":
            self.diffusion_loss = nn.SmoothL1Loss(reduction='mean')
        elif diffusion_loss_type == "vlb":  # Variational Lower Bound
            self.diffusion_loss = self._vlb_loss
        else:
            raise ValueError(f"Unknown diffusion loss type: {diffusion_loss_type}")
        
        # Perceptual loss
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
        
        # Consistency loss
        if consistency_weight > 0:
            self.consistency_loss = ConsistencyLoss3D()
        
        # Style loss (for style transfer)
        if style_weight > 0:
            self.style_loss = StyleLoss3D()
        
        # Temporal loss (for video/sequential generation)
        if temporal_weight > 0:
            self.temporal_loss = TemporalConsistencyLoss()
        
        # Progressive weighting
        self.enable_progressive = enable_progressive
        if enable_progressive:
            self.progressive_weights = self._compute_progressive_weights()
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timestep: Optional[int] = None,
        masks: Optional[torch.Tensor] = None,
        style_features: Optional[torch.Tensor] = None,
        temporal_sequence: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss"""
        losses = {}
        
        # Base diffusion loss
        if self.diffusion_loss_type == "vlb":
            losses["diffusion"] = self.diffusion_loss(prediction, target, timestep, **kwargs)
        else:
            losses["diffusion"] = self.diffusion_loss(prediction, target)
        
        # Weight by timestep if progressive weighting enabled
        if self.enable_progressive and timestep is not None:
            weight = self.progressive_weights[timestep] if timestep < len(self.progressive_weights) else 1.0
            losses["diffusion"] = losses["diffusion"] * weight
        
        # Apply mask if provided
        if masks is not None:
            losses["diffusion"] = (losses["diffusion"] * masks).sum() / (masks.sum() + 1e-8)
        
        total_loss = losses["diffusion"]
        
        # Perceptual loss
        if self.perceptual_weight > 0 and hasattr(self, 'perceptual_loss'):
            # Extract RGB channels for perceptual loss
            if prediction.shape[1] >= 3 and target.shape[1] >= 3:
                pred_rgb = prediction[:, :3]
                target_rgb = target[:, :3]
                
                perc_loss = self.perceptual_loss(pred_rgb, target_rgb)
                losses["perceptual"] = perc_loss
                total_loss = total_loss + self.perceptual_weight * perc_loss
        
        # Consistency loss (3D specific)
        if self.consistency_weight > 0 and hasattr(self, 'consistency_loss'):
            cons_loss = self.consistency_loss(prediction, target)
            losses["consistency"] = cons_loss
            total_loss = total_loss + self.consistency_weight * cons_loss
        
        # Style loss
        if self.style_weight > 0 and style_features is not None and hasattr(self, 'style_loss'):
            style_loss = self.style_loss(prediction, style_features)
            losses["style"] = style_loss
            total_loss = total_loss + self.style_weight * style_loss
        
        # Temporal loss
        if self.temporal_weight > 0 and temporal_sequence is not None and hasattr(self, 'temporal_loss'):
            temp_loss = self.temporal_loss(prediction, temporal_sequence)
            losses["temporal"] = temp_loss
            total_loss = total_loss + self.temporal_weight * temp_loss
        
        # KL divergence (for VAE components)
        if self.kl_weight > 0 and "kl_divergence" in kwargs:
            kl_loss = kwargs["kl_divergence"]
            losses["kl"] = kl_loss
            total_loss = total_loss + self.kl_weight * kl_loss
        
        losses["total"] = total_loss
        
        return losses
    
    def _vlb_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        timestep: int,
        alphas_cumprod: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Variational Lower Bound loss for diffusion"""
        # Compute SNR
        alpha_bar_t = alphas_cumprod[timestep]
        sigma2_t = 1 - alpha_bar_t
        
        # VLB weighting
        weighting = sigma2_t / (alpha_bar_t * (1 - alpha_bar_t))
        weighting = weighting.view(-1, 1, 1, 1, 1)
        
        # MSE loss with weighting
        mse_loss = F.mse_loss(prediction, target, reduction='none')
        vlb_loss = weighting * mse_loss
        
        return vlb_loss.mean()
    
    def _compute_progressive_weights(self, max_timesteps: int = 1000) -> torch.Tensor:
        """Compute progressive weights for different timesteps"""
        # More weight on mid-timesteps, less on very early/late
        weights = torch.ones(max_timesteps)
        
        # Create bell curve weights
        mu = max_timesteps / 2
        sigma = max_timesteps / 6
        
        for t in range(max_timesteps):
            weights[t] = torch.exp(-0.5 * ((t - mu) / sigma) ** 2)
        
        # Normalize
        weights = weights / weights.max()
        
        # Ensure minimum weight
        weights = torch.clamp(weights, 0.1, 1.0)
        
        return weights

class PerceptualLoss(nn.Module):
    """Perceptual loss using pre-trained networks for 3D scenes"""
    
    def __init__(
        self,
        network: str = "vgg16",
        layers: List[str] = None,
        weights: List[float] = None,
        spatial_reduction: str = "mean",
        normalize: bool = True
    ):
        super().__init__()
        
        self.network = network
        self.normalize = normalize
        self.spatial_reduction = spatial_reduction
        
        # Default layers and weights
        if layers is None:
            if network == "vgg16":
                layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
                weights = [1.0, 1.0, 1.0, 1.0]
            elif network == "resnet50":
                layers = ['layer1', 'layer2', 'layer3', 'layer4']
                weights = [1.0, 1.0, 1.0, 1.0]
            else:
                raise ValueError(f"Unsupported network: {network}")
        
        self.layers = layers
        self.weights = weights
        
        # Load pre-trained model
        self.model = self._load_pretrained_model(network)
        self.model.eval()
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Register hooks to extract features
        self.features = {}
        self._register_hooks()
    
    def _load_pretrained_model(self, network: str) -> nn.Module:
        """Load pre-trained model"""
        if network == "vgg16":
            model = models.vgg16(pretrained=True).features
        elif network == "resnet50":
            model = models.resnet50(pretrained=True)
        elif network == "inception_v3":
            model = models.inception_v3(pretrained=True, aux_logits=False)
        else:
            raise ValueError(f"Unsupported network: {network}")
        
        return model
    
    def _register_hooks(self):
        """Register hooks to extract features from specific layers"""
        
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        if self.network == "vgg16":
            # VGG16 layer indices
            layer_indices = {
                'relu1_2': 4,
                'relu2_2': 9,
                'relu3_3': 16,
                'relu4_3': 23,
                'relu5_3': 30
            }
            
            for layer_name, layer_idx in layer_indices.items():
                if layer_name in self.layers:
                    self.model[layer_idx].register_forward_hook(get_activation(layer_name))
        
        elif self.network == "resnet50":
            # ResNet50 layers
            layer_mapping = {
                'layer1': self.model.layer1,
                'layer2': self.model.layer2,
                'layer3': self.model.layer3,
                'layer4': self.model.layer4
            }
            
            for layer_name, layer_module in layer_mapping.items():
                if layer_name in self.layers:
                    layer_module.register_forward_hook(get_activation(layer_name))
    
    def _extract_3d_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from 3D volume using 2D slices"""
        B, C, D, H, W = x.shape
        
        # Process each depth slice independently
        features = {}
        
        for d in range(D):
            # Extract 2D slice
            slice_2d = x[:, :, d, :, :]  # [B, C, H, W]
            
            # Normalize if needed
            if self.normalize:
                slice_2d = self._normalize_imagenet(slice_2d)
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(slice_2d)
            
            # Store features
            for layer_name in self.layers:
                if layer_name in self.features:
                    feat = self.features[layer_name]
                    
                    if layer_name not in features:
                        features[layer_name] = []
                    
                    features[layer_name].append(feat)
        
        # Stack depth dimension
        for layer_name in features:
            # Stack along new dimension (depth)
            features[layer_name] = torch.stack(features[layer_name], dim=2)  # [B, C, D, H', W']
        
        return features
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between prediction and target"""
        # Extract features
        pred_features = self._extract_3d_features(pred)
        target_features = self._extract_3d_features(target)
        
        # Compute loss for each layer
        total_loss = 0.0
        
        for i, layer_name in enumerate(self.layers):
            if layer_name in pred_features and layer_name in target_features:
                pred_feat = pred_features[layer_name]
                target_feat = target_features[layer_name]
                
                # Compute L2 loss between features
                if self.spatial_reduction == "mean":
                    layer_loss = F.mse_loss(pred_feat, target_feat)
                elif self.spatial_reduction == "sum":
                    layer_loss = F.mse_loss(pred_feat, target_feat, reduction='sum')
                else:
                    raise ValueError(f"Unknown spatial reduction: {self.spatial_reduction}")
                
                # Apply layer weight
                total_loss = total_loss + self.weights[i] * layer_loss
        
        # Average over layers
        total_loss = total_loss / len(self.layers)
        
        return total_loss
    
    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize images for ImageNet-trained models"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        return (x - mean) / std

class ConsistencyLoss3D(nn.Module):
    """3D consistency loss for ensuring spatial coherence"""
    
    def __init__(
        self,
        consistency_type: str = "multiview",
        weight_scheme: str = "uniform",
        epsilon: float = 1e-6
    ):
        super().__init__()
        
        self.consistency_type = consistency_type
        self.weight_scheme = weight_scheme
        self.epsilon = epsilon
        
        if consistency_type == "multiview":
            self.loss_fn = self._multiview_consistency
        elif consistency_type == "symmetry":
            self.loss_fn = self._symmetry_consistency
        elif consistency_type == "smoothness":
            self.loss_fn = self._smoothness_consistency
        else:
            raise ValueError(f"Unknown consistency type: {consistency_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 3D consistency loss"""
        return self.loss_fn(pred, target)
    
    def _multiview_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ensure consistency between different 2D views of 3D scene"""
        B, C, D, H, W = pred.shape
        
        # Extract orthogonal views
        views = []
        
        # XY plane (front view)
        xy_view = pred[:, :, D//2, :, :]  # Middle depth slice
        
        # XZ plane (top view)
        xz_view = pred[:, :, :, H//2, :]  # Middle height slice
        
        # YZ plane (side view)
        yz_view = pred[:, :, :, :, W//2]  # Middle width slice
        
        views = [xy_view, xz_view, yz_view]
        
        # Compute pairwise consistency
        total_loss = 0.0
        count = 0
        
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                # Project view j to view i's coordinates
                # This is simplified - in practice would use actual projection
                view_i = views[i]
                view_j = views[j]
                
                # Resize to same dimensions
                if view_i.shape[-2:] != view_j.shape[-2:]:
                    view_j = F.interpolate(
                        view_j,
                        size=view_i.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Compute consistency loss
                loss = F.mse_loss(view_i, view_j)
                total_loss += loss
                count += 1
        
        return total_loss / count if count > 0 else torch.tensor(0.0, device=pred.device)
    
    def _symmetry_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ensure symmetry in generated scenes"""
        B, C, D, H, W = pred.shape
        
        # Check symmetry along different axes
        
        # X-axis symmetry (left-right)
        pred_left = pred[:, :, :, :, :W//2]
        pred_right = pred[:, :, :, :, W//2:][:, :, :, :, ::-1]  # Flip
        
        # Y-axis symmetry (top-bottom)
        pred_top = pred[:, :, :, :H//2, :]
        pred_bottom = pred[:, :, :, H//2:, :][:, :, :, ::-1, :]  # Flip
        
        # Z-axis symmetry (front-back)
        pred_front = pred[:, :, :D//2, :, :]
        pred_back = pred[:, :, D//2:, :, :][:, :, ::-1, :, :]  # Flip
        
        # Compute symmetry losses
        loss_x = F.mse_loss(pred_left, pred_right)
        loss_y = F.mse_loss(pred_top, pred_bottom)
        loss_z = F.mse_loss(pred_front, pred_back)
        
        # Weighted sum
        weights = self._get_symmetry_weights()
        total_loss = weights[0] * loss_x + weights[1] * loss_y + weights[2] * loss_z
        
        return total_loss / sum(weights)
    
    def _get_symmetry_weights(self) -> List[float]:
        """Get weights for different symmetry axes"""
        if self.weight_scheme == "uniform":
            return [1.0, 1.0, 1.0]
        elif self.weight_scheme == "vertical_emphasis":
            return [1.0, 0.5, 1.0]  # Less emphasis on top-bottom symmetry
        elif self.weight_scheme == "horizontal_emphasis":
            return [0.5, 1.0, 1.0]  # Less emphasis on left-right symmetry
        else:
            return [1.0, 1.0, 1.0]
    
    def _smoothness_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Ensure smooth transitions in 3D space"""
        # Compute gradients in 3D
        grad_x = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        grad_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_z = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        
        # Compute gradient magnitude
        grad_mag_x = torch.abs(grad_x).mean()
        grad_mag_y = torch.abs(grad_y).mean()
        grad_mag_z = torch.abs(grad_z).mean()
        
        # Total variation loss
        tv_loss = grad_mag_x + grad_mag_y + grad_mag_z
        
        return tv_loss

class StyleLoss3D(nn.Module):
    """Style loss for 3D scenes using Gram matrices"""
    
    def __init__(self):
        super().__init__()
        
        # Use VGG for style features
        self.vgg = models.vgg19(pretrained=True).features
        self.vgg.eval()
        
        # Freeze VGG
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Style layers
        self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.style_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        
        # Register hooks
        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to extract style features"""
        
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        # VGG19 layer indices
        layer_indices = {
            'relu1_1': 1,
            'relu2_1': 6,
            'relu3_1': 11,
            'relu4_1': 20,
            'relu5_1': 29
        }
        
        for layer_name, layer_idx in layer_indices.items():
            if layer_name in self.style_layers:
                self.vgg[layer_idx].register_forward_hook(get_activation(layer_name))
    
    def _compute_gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix for style features"""
        B, C, H, W = x.shape
        features = x.view(B, C, H * W)
        
        # Compute Gram matrix
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by number of elements
        gram = gram / (C * H * W)
        
        return gram
    
    def _extract_style_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract style features from 3D volume"""
        B, C, D, H, W = x.shape
        
        style_features = {layer: [] for layer in self.style_layers}
        
        for d in range(D):
            # Extract 2D slice
            slice_2d = x[:, :, d, :, :]
            
            # Normalize for VGG
            slice_2d = self._normalize_imagenet(slice_2d)
            
            # Forward pass
            with torch.no_grad():
                _ = self.vgg(slice_2d)
            
            # Extract features for each style layer
            for layer_name in self.style_layers:
                if layer_name in self.features:
                    feat = self.features[layer_name]
                    style_features[layer_name].append(feat)
        
        # Average over depth dimension
        for layer_name in style_features:
            if style_features[layer_name]:
                # Average features across depth
                style_features[layer_name] = torch.stack(style_features[layer_name], dim=0).mean(dim=0)
        
        return style_features
    
    def forward(self, pred: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        """Compute style loss between prediction and style target"""
        # Extract style features
        pred_features = self._extract_style_features(pred)
        style_features = self._extract_style_features(style_target)
        
        # Compute style loss
        total_loss = 0.0
        
        for i, layer_name in enumerate(self.style_layers):
            if layer_name in pred_features and layer_name in style_features:
                pred_feat = pred_features[layer_name]
                style_feat = style_features[layer_name]
                
                # Compute Gram matrices
                pred_gram = self._compute_gram_matrix(pred_feat)
                style_gram = self._compute_gram_matrix(style_feat)
                
                # Compute MSE between Gram matrices
                layer_loss = F.mse_loss(pred_gram, style_gram)
                
                # Apply layer weight
                total_loss = total_loss + self.style_weights[i] * layer_loss
        
        return total_loss / len(self.style_layers)
    
    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize for ImageNet-trained VGG"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        
        return (x - mean) / std

class TemporalConsistencyLoss(nn.Module):
    """Loss for temporal consistency in sequential generation"""
    
    def __init__(self, method: str = "flow", alpha: float = 0.5):
        super().__init__()
        
        self.method = method
        self.alpha = alpha
        
        if method == "flow":
            self.loss_fn = self._optical_flow_loss
        elif method == "warp":
            self.loss_fn = self._warping_loss
        elif method == "recurrent":
            self.loss_fn = self._recurrent_consistency_loss
        else:
            raise ValueError(f"Unknown temporal method: {method}")
    
    def forward(self, pred_sequence: torch.Tensor, target_sequence: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss"""
        return self.loss_fn(pred_sequence, target_sequence)
    
    def _optical_flow_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss based on optical flow consistency"""
        # Assuming sequence dimension is first: [T, B, C, D, H, W]
        T = pred.shape[0]
        
        total_loss = 0.0
        
        for t in range(T - 1):
            # Current and next frames
            frame_t = pred[t]
            frame_t1 = pred[t + 1]
            target_t = target[t]
            target_t1 = target[t + 1]
            
            # Compute flow consistency (simplified)
            # In practice, would use actual optical flow estimation
            flow_pred = frame_t1 - frame_t
            flow_target = target_t1 - target_t
            
            # Flow consistency loss
            flow_loss = F.mse_loss(flow_pred, flow_target)
            
            # Brightness constancy (simplified)
            brightness_loss = F.mse_loss(frame_t1, frame_t + flow_target)
            
            total_loss += self.alpha * flow_loss + (1 - self.alpha) * brightness_loss
        
        return total_loss / (T - 1)
    
    def _warping_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss based on frame warping consistency"""
        T = pred.shape[0]
        
        total_loss = 0.0
        
        for t in range(1, T):
            # Warp previous prediction to current time
            # Simplified: just compare differences
            warp_loss = F.mse_loss(pred[t], pred[t-1])
            total_loss += warp_loss
        
        return total_loss / (T - 1)
    
    def _recurrent_consistency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss for recurrent network consistency"""
        # Assuming we have hidden states or recurrent outputs
        # This is a placeholder for actual recurrent consistency loss
        
        T = pred.shape[0]
        
        # Compute variance across time (should be low for consistency)
        time_variance = torch.var(pred, dim=0).mean()
        
        return time_variance

class LPIPSLoss3D(nn.Module):
    """Learned Perceptual Image Patch Similarity for 3D"""
    
    def __init__(self, net_type: str = 'alex', spatial: bool = False):
        super().__init__()
        
        # Use 2D LPIPS and extend to 3D
        self.lpips_2d = lpips.LPIPS(net=net_type, spatial=spatial)
        
        # Freeze LPIPS
        for param in self.lpips_2d.parameters():
            param.requires_grad = False
        
        self.lpips_2d.eval()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS loss for 3D volume"""
        B, C, D, H, W = pred.shape
        
        total_loss = 0.0
        
        # Process each depth slice
        for d in range(D):
            pred_slice = pred[:, :, d, :, :]  # [B, C, H, W]
            target_slice = target[:, :, d, :, :]  # [B, C, H, W]
            
            # Compute LPIPS for this slice
            with torch.no_grad():
                loss_slice = self.lpips_2d(pred_slice, target_slice)
            
            total_loss += loss_slice.mean()
        
        return total_loss / D

class GradientMatchingLoss(nn.Module):
    """Match gradients between prediction and target"""
    
    def __init__(self, order: int = 1, mode: str = 'sobel'):
        super().__init__()
        
        self.order = order
        self.mode = mode
        
        if mode == 'sobel':
            self._create_sobel_filters()
    
    def _create_sobel_filters(self):
        """Create Sobel filters for gradient computation"""
        # 3D Sobel filters
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32) / 32.0
        
        sobel_y = sobel_x.transpose(1, 2)
        sobel_z = sobel_x.transpose(0, 2)
        
        self.sobel_x = nn.Parameter(sobel_x.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.sobel_y = nn.Parameter(sobel_y.unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.sobel_z = nn.Parameter(sobel_z.unsqueeze(0).unsqueeze(0), requires_grad=False)
    
    def _compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute gradients using Sobel filters"""
        grad_x = F.conv3d(x, self.sobel_x.to(x.device), padding=1)
        grad_y = F.conv3d(x, self.sobel_y.to(x.device), padding=1)
        grad_z = F.conv3d(x, self.sobel_z.to(x.device), padding=1)
        
        return grad_x, grad_y, grad_z
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient matching loss"""
        # Compute gradients
        pred_grad_x, pred_grad_y, pred_grad_z = self._compute_gradients(pred)
        target_grad_x, target_grad_y, target_grad_z = self._compute_gradients(target)
        
        # Compute losses
        loss_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_y = F.mse_loss(pred_grad_y, target_grad_y)
        loss_z = F.mse_loss(pred_grad_z, target_grad_z)
        
        return (loss_x + loss_y + loss_z) / 3

class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training"""
    
    def __init__(
        self,
        gan_mode: str = 'hinge',
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0
    ):
        super().__init__()
        
        self.gan_mode = gan_mode
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        
        if gan_mode == 'hinge':
            self.loss_fn = self._hinge_loss
        elif gan_mode == 'wasserstein':
            self.loss_fn = self._wasserstein_loss
        elif gan_mode == 'lsgan':
            self.loss_fn = self._lsgan_loss
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")
    
    def forward(
        self,
        discriminator_output: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """Compute adversarial loss"""
        if target_is_real:
            target_tensor = torch.tensor(
                self.target_real_label,
                device=discriminator_output.device
            ).expand_as(discriminator_output)
        else:
            target_tensor = torch.tensor(
                self.target_fake_label,
                device=discriminator_output.device
            ).expand_as(discriminator_output)
        
        return self.loss_fn(discriminator_output, target_tensor)
    
    def _hinge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Hinge loss for GANs"""
        if target.mean() > 0:  # Real
            loss = F.relu(1 - pred).mean()
        else:  # Fake
            loss = F.relu(1 + pred).mean()
        
        return loss
    
    def _wasserstein_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Wasserstein loss for GANs"""
        return (pred * target).mean()
    
    def _lsgan_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Least Squares GAN loss"""
        return F.mse_loss(pred, target)

class FrequencyLoss(nn.Module):
    """Loss in frequency domain for 3D scenes"""
    
    def __init__(self, low_freq_weight: float = 1.0, high_freq_weight: float = 0.5):
        super().__init__()
        
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute frequency domain loss"""
        # Compute FFT
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1))
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1))
        
        # Get magnitude
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Get phase
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        
        # Split into low and high frequency components
        D, H, W = pred.shape[-3:]
        
        # Create frequency mask
        d_freq = torch.fft.fftfreq(D, device=pred.device).abs()
        h_freq = torch.fft.fftfreq(H, device=pred.device).abs()
        w_freq = torch.fft.fftfreq(W, device=pred.device).abs()
        
        # 3D frequency grid
        d_grid, h_grid, w_grid = torch.meshgrid(d_freq, h_freq, w_freq, indexing='ij')
        freq_mag = torch.sqrt(d_grid**2 + h_grid**2 + w_grid**2)
        
        # Normalize frequencies
        freq_mag = freq_mag / freq_mag.max()
        
        # Low frequency mask
        low_freq_mask = (freq_mag < 0.3).float()
        high_freq_mask = (freq_mag >= 0.3).float()
        
        # Compute losses
        low_freq_loss = F.mse_loss(pred_mag * low_freq_mask, target_mag * low_freq_mask)
        high_freq_loss = F.mse_loss(pred_mag * high_freq_mask, target_mag * high_freq_mask)
        phase_loss = F.mse_loss(pred_phase, target_phase)
        
        # Weighted sum
        total_loss = (
            self.low_freq_weight * low_freq_loss +
            self.high_freq_weight * high_freq_loss +
            0.1 * phase_loss
        )
        
        return total_loss