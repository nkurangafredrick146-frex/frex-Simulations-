"""
Unit tests for world model components.
Tests transformer blocks, attention modules, diffusion models, and training components.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.world_model.architecture.transformer_blocks import TransformerBlock
from src.core.world_model.architecture.attention_modules import MultiHeadAttention, CrossAttention
from src.core.world_model.architecture.diffusion_models import DiffusionModel
from src.core.world_model.training.trainer import WorldModelTrainer
from src.core.world_model.training.loss_functions import WorldModelLoss
from src.core.world_model.inference.generator import WorldGenerator
from src.core.world_model.inference.sampler import Sampler


class TestTransformerBlock:
    """Tests for transformer blocks"""
    
    @pytest.fixture
    def transformer_block(self):
        """Create TransformerBlock instance"""
        return TransformerBlock(
            dim=512,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_sequence(self):
        """Create sample sequence for testing"""
        batch_size = 2
        seq_len = 10
        dim = 512
        
        return torch.randn(batch_size, seq_len, dim)
    
    def test_transformer_block_initialization(self, transformer_block):
        """Test TransformerBlock initialization"""
        assert transformer_block.dim == 512
        assert transformer_block.num_heads == 8
        assert transformer_block.mlp_ratio == 4
        assert transformer_block.dropout == 0.1
        assert transformer_block.device == "cpu"
        
        # Check components
        assert hasattr(transformer_block, 'attention')
        assert hasattr(transformer_block, 'mlp')
        assert hasattr(transformer_block, 'norm1')
        assert hasattr(transformer_block, 'norm2')
    
    def test_forward_pass(self, transformer_block, sample_sequence):
        """Test forward pass through transformer block"""
        output = transformer_block(sample_sequence)
        
        # Verify output shape matches input
        assert output.shape == sample_sequence.shape
        
        # Output should be different from input (transformation occurred)
        assert not torch.allclose(output, sample_sequence)
    
    def test_attention_masks(self, transformer_block):
        """Test attention with different mask types"""
        batch_size = 2
        seq_len = 10
        dim = 512
        
        sequence = torch.randn(batch_size, seq_len, dim)
        
        # Test with causal mask (for autoregressive generation)
        causal_mask = transformer_block.create_causal_mask(seq_len)
        output_causal = transformer_block(sequence, attention_mask=causal_mask)
        
        # Test with padding mask
        padding_mask = torch.ones(batch_size, seq_len)
        padding_mask[:, 5:] = 0  # Last 5 tokens are padding
        output_padding = transformer_block(sequence, attention_mask=padding_mask)
        
        # Both should produce valid output
        assert output_causal.shape == sequence.shape
        assert output_padding.shape == sequence.shape
    
    def test_residual_connections(self, transformer_block, sample_sequence):
        """Test residual connections"""
        # Forward pass
        output = transformer_block(sample_sequence)
        
        # Manually compute with residual to verify
        # This test would need access to intermediate activations
        # For now, just verify output is reasonable
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))
    
    def test_gradient_flow(self, transformer_block, sample_sequence):
        """Test gradient flow through transformer block"""
        # Create loss function
        target = torch.randn_like(sample_sequence)
        
        # Forward and backward pass
        output = transformer_block(sample_sequence)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check gradients
        for name, param in transformer_block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.any(torch.isnan(param.grad))
    
    def test_dropout_training_eval(self, transformer_block, sample_sequence):
        """Test dropout behavior in training vs evaluation mode"""
        # Set to training mode
        transformer_block.train()
        output_train = transformer_block(sample_sequence)
        
        # Set to evaluation mode
        transformer_block.eval()
        output_eval = transformer_block(sample_sequence)
        
        # Outputs should be different due to dropout
        # (but could be same if dropout is 0)
        if transformer_block.dropout > 0:
            assert not torch.allclose(output_train, output_eval)


class TestMultiHeadAttention:
    """Tests for multi-head attention"""
    
    @pytest.fixture
    def multi_head_attention(self):
        """Create MultiHeadAttention instance"""
        return MultiHeadAttention(
            dim=512,
            num_heads=8,
            dropout=0.1,
            bias=True
        )
    
    @pytest.fixture
    def sample_attention_input(self):
        """Create sample input for attention"""
        batch_size = 2
        seq_len = 10
        dim = 512
        
        return torch.randn(batch_size, seq_len, dim)
    
    def test_multi_head_attention_initialization(self, multi_head_attention):
        """Test MultiHeadAttention initialization"""
        assert multi_head_attention.dim == 512
        assert multi_head_attention.num_heads == 8
        assert multi_head_attention.dropout == 0.1
        assert multi_head_attention.bias == True
        
        # Check that projection layers are created
        assert hasattr(multi_head_attention, 'q_proj')
        assert hasattr(multi_head_attention, 'k_proj')
        assert hasattr(multi_head_attention, 'v_proj')
        assert hasattr(multi_head_attention, 'out_proj')
    
    def test_attention_forward(self, multi_head_attention, sample_attention_input):
        """Test attention forward pass"""
        output = multi_head_attention(
            query=sample_attention_input,
            key=sample_attention_input,
            value=sample_attention_input
        )
        
        # Verify output
        assert "output" in output
        assert "attention_weights" in output
        
        # Output shape should match input
        assert output["output"].shape == sample_attention_input.shape
        
        # Attention weights should be probabilities
        weights = output["attention_weights"]
        assert weights.shape == (2, 8, 10, 10)  # Batch, Heads, Query, Key
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)
    
    def test_attention_with_mask(self, multi_head_attention):
        """Test attention with mask"""
        batch_size = 2
        seq_len = 10
        dim = 512
        
        x = torch.randn(batch_size, seq_len, dim)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # 1, 1, Seq, Seq
        mask = mask.repeat(batch_size, multi_head_attention.num_heads, 1, 1)
        
        output = multi_head_attention(x, x, x, attention_mask=mask)
        
        # With causal mask, attention weights should be upper triangular
        weights = output["attention_weights"]
        for b in range(batch_size):
            for h in range(multi_head_attention.num_heads):
                # Check that upper triangular part is zero
                upper_tri = torch.triu(weights[b, h], diagonal=1)
                assert torch.all(upper_tri < 1e-6)
    
    def test_attention_scaling(self, multi_head_attention):
        """Test attention scaling factor"""
        # Create small and large inputs
        x_small = torch.randn(1, 5, 512) * 0.1
        x_large = torch.randn(1, 5, 512) * 10.0
        
        output_small = multi_head_attention(x_small, x_small, x_small)
        output_large = multi_head_attention(x_large, x_large, x_large)
        
        # Both should produce valid output
        assert not torch.any(torch.isnan(output_small["output"]))
        assert not torch.any(torch.isnan(output_large["output"]))
    
    def test_multi_head_projection(self, multi_head_attention, sample_attention_input):
        """Test multi-head projection and reshaping"""
        batch_size, seq_len, dim = sample_attention_input.shape
        
        # Project queries
        q_projected = multi_head_attention.q_proj(sample_attention_input)
        
        # Reshape to multi-head format
        head_dim = dim // multi_head_attention.num_heads
        q_reshaped = q_projected.view(batch_size, seq_len, multi_head_attention.num_heads, head_dim)
        q_reshaped = q_reshaped.transpose(1, 2)  # Batch, Heads, Seq, HeadDim
        
        # Verify reshaping
        assert q_reshaped.shape == (batch_size, multi_head_attention.num_heads, seq_len, head_dim)
        
        # Data should be preserved
        original_flat = q_projected.view(-1)
        reshaped_flat = q_reshaped.contiguous().view(-1)
        assert torch.allclose(original_flat, reshaped_flat)


class TestCrossAttention:
    """Tests for cross-attention"""
    
    @pytest.fixture
    def cross_attention(self):
        """Create CrossAttention instance"""
        return CrossAttention(
            query_dim=512,
            context_dim=768,
            num_heads=8,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_cross_attention_input(self):
        """Create sample input for cross-attention"""
        batch_size = 2
        
        # Queries (e.g., image features)
        queries = torch.randn(batch_size, 196, 512)  # Batch, QuerySeq, Dim
        
        # Context (e.g., text embeddings)
        context = torch.randn(batch_size, 10, 768)   # Batch, ContextSeq, Dim
        
        return queries, context
    
    def test_cross_attention_initialization(self, cross_attention):
        """Test CrossAttention initialization"""
        assert cross_attention.query_dim == 512
        assert cross_attention.context_dim == 768
        assert cross_attention.num_heads == 8
        assert cross_attention.dropout == 0.1
        
        # Should have separate projections for queries and context
        assert hasattr(cross_attention, 'q_proj')
        assert hasattr(cross_attention, 'k_proj')
        assert hasattr(cross_attention, 'v_proj')
        assert hasattr(cross_attention, 'out_proj')
    
    def test_cross_attention_forward(self, cross_attention, sample_cross_attention_input):
        """Test cross-attention forward pass"""
        queries, context = sample_cross_attention_input
        
        output = cross_attention(queries=queries, context=context)
        
        # Verify output
        assert "output" in output
        assert "attention_weights" in output
        
        # Output should have same shape as queries
        assert output["output"].shape == queries.shape
        
        # Attention weights should relate queries to context
        weights = output["attention_weights"]
        assert weights.shape == (2, 8, 196, 10)  # Batch, Heads, QuerySeq, ContextSeq
        
        # Should be probability distributions over context
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)
    
    def test_cross_attention_without_context(self, cross_attention):
        """Test cross-attention without context (self-attention mode)"""
        batch_size = 2
        seq_len = 10
        dim = 512
        
        queries = torch.randn(batch_size, seq_len, dim)
        
        # When context is None, should use queries as context
        output = cross_attention(queries=queries, context=None)
        
        # Should still produce valid output
        assert output["output"].shape == queries.shape
        
        # Attention weights should be square
        assert output["attention_weights"].shape == (batch_size, 8, seq_len, seq_len)
    
    def test_cross_attention_gating(self, cross_attention, sample_cross_attention_input):
        """Test cross-attention with gating mechanism"""
        queries, context = sample_cross_attention_input
        
        # Add gating
        output = cross_attention(
            queries=queries,
            context=context,
            use_gating=True,
            gate_bias=1.0
        )
        
        # Should have gate values in output
        assert "gate_values" in output
        gate_values = output["gate_values"]
        
        # Gate values should be between 0 and 1
        assert torch.all(gate_values >= 0) and torch.all(gate_values <= 1)
    
    def test_cross_attention_cache(self, cross_attention):
        """Test cross-attention with KV caching"""
        batch_size = 2
        seq_len = 10
        context_len = 5
        
        # Initial queries and context
        queries1 = torch.randn(batch_size, 1, 512)  # Single query (autoregressive)
        context1 = torch.randn(batch_size, context_len, 768)
        
        # First forward pass
        output1 = cross_attention(queries=queries1, context=context1)
        
        # Extract KV cache
        kv_cache = output1.get("kv_cache")
        
        # Second forward pass with cache
        queries2 = torch.randn(batch_size, 1, 512)  # Next query
        output2 = cross_attention(queries=queries2, context=None, kv_cache=kv_cache)
        
        # Should produce output without full context
        assert output2["output"].shape == queries2.shape


class TestDiffusionModel:
    """Tests for diffusion models"""
    
    @pytest.fixture
    def diffusion_model(self):
        """Create DiffusionModel instance"""
        return DiffusionModel(
            in_channels=4,
            model_channels=128,
            num_res_blocks=2,
            attention_resolutions=[16, 8],
            dropout=0.1,
            num_heads=4,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_diffusion_input(self):
        """Create sample input for diffusion model"""
        batch_size = 2
        channels = 4
        height = 32
        width = 32
        
        # Noisy input
        x = torch.randn(batch_size, channels, height, width)
        
        # Timestep
        t = torch.randint(0, 1000, (batch_size,))
        
        # Optional conditioning
        cond = torch.randn(batch_size, 512)
        
        return x, t, cond
    
    def test_diffusion_model_initialization(self, diffusion_model):
        """Test DiffusionModel initialization"""
        assert diffusion_model.in_channels == 4
        assert diffusion_model.model_channels == 128
        assert diffusion_model.num_res_blocks == 2
        assert diffusion_model.attention_resolutions == [16, 8]
        assert diffusion_model.dropout == 0.1
        assert diffusion_model.num_heads == 4
        assert diffusion_model.device == "cpu"
        
        # Check components
        assert hasattr(diffusion_model, 'time_embed')
        assert hasattr(diffusion_model, 'input_blocks')
        assert hasattr(diffusion_model, 'middle_block')
        assert hasattr(diffusion_model, 'output_blocks')
    
    def test_forward_pass(self, diffusion_model, sample_diffusion_input):
        """Test forward pass through diffusion model"""
        x, t, cond = sample_diffusion_input
        
        output = diffusion_model(x, t, cond)
        
        # Verify output shape
        assert output.shape == x.shape
        
        # Output should be noise prediction
        # (same shape as input)
    
    def test_timestep_embedding(self, diffusion_model):
        """Test timestep embedding"""
        batch_size = 4
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        # Get embedding
        embedding = diffusion_model.time_embed(timesteps)
        
        # Verify embedding
        assert embedding.shape == (batch_size, diffusion_model.model_channels * 4)
        
        # Different timesteps should have different embeddings
        if batch_size > 1:
            assert not torch.allclose(embedding[0], embedding[1])
    
    def test_conditioning(self, diffusion_model, sample_diffusion_input):
        """Test model with conditioning"""
        x, t, cond = sample_diffusion_input
        
        # With conditioning
        output_with_cond = diffusion_model(x, t, cond)
        
        # Without conditioning
        output_without_cond = diffusion_model(x, t, None)
        
        # Outputs should be different
        assert not torch.allclose(output_with_cond, output_without_cond)
    
    def test_diffusion_sampling(self, diffusion_model):
        """Test diffusion sampling process"""
        batch_size = 2
        shape = (4, 32, 32)  # Channels, Height, Width
        
        # Mock the model to return simple predictions
        with patch.object(diffusion_model, 'forward') as mock_forward:
            # Mock forward to return scaled input
            mock_forward.side_effect = lambda x, t, cond: -x  # Predict negative of input
            
            # Test different samplers
            for sampler_type in ['ddpm', 'ddim', 'plms']:
                samples = diffusion_model.sample(
                    shape=shape,
                    batch_size=batch_size,
                    sampler_type=sampler_type,
                    num_steps=10,
                    conditioning=None
                )
                
                # Should generate samples
                assert samples.shape == (batch_size, *shape)
    
    def test_noise_schedule(self, diffusion_model):
        """Test noise schedule computation"""
        num_timesteps = 1000
        
        # Get noise schedule
        schedule = diffusion_model.get_noise_schedule(num_timesteps, schedule_type='linear')
        
        # Verify schedule
        assert 'alphas' in schedule
        assert 'alphas_cumprod' in schedule
        assert 'betas' in schedule
        
        # Should have values for each timestep
        assert len(schedule['betas']) == num_timesteps
        
        # Betas should increase over time
        for i in range(1, num_timesteps):
            assert schedule['betas'][i] >= schedule['betas'][i-1]
        
        # Alphas should decrease over time
        for i in range(1, num_timesteps):
            assert schedule['alphas'][i] <= schedule['alphas'][i-1]


class TestWorldModelLoss:
    """Tests for world model loss functions"""
    
    @pytest.fixture
    def world_model_loss(self):
        """Create WorldModelLoss instance"""
        return WorldModelLoss(
            reconstruction_weight=1.0,
            perceptual_weight=0.1,
            adversarial_weight=0.01,
            kl_weight=0.001,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_loss_input(self):
        """Create sample input for loss computation"""
        batch_size = 2
        channels = 3
        height = 64
        width = 64
        
        # Predictions and targets
        pred = torch.rand(batch_size, channels, height, width)
        target = torch.rand(batch_size, channels, height, width)
        
        # Latent distributions (for VAE loss)
        mu = torch.randn(batch_size, 512)
        logvar = torch.randn(batch_size, 512)
        
        # Discriminator outputs (for adversarial loss)
        real_score = torch.rand(batch_size, 1)
        fake_score = torch.rand(batch_size, 1)
        
        return {
            "pred": pred,
            "target": target,
            "mu": mu,
            "logvar": logvar,
            "real_score": real_score,
            "fake_score": fake_score
        }
    
    def test_world_model_loss_initialization(self, world_model_loss):
        """Test WorldModelLoss initialization"""
        assert world_model_loss.reconstruction_weight == 1.0
        assert world_model_loss.perceptual_weight == 0.1
        assert world_model_loss.adversarial_weight == 0.01
        assert world_model_loss.kl_weight == 0.001
        assert world_model_loss.device == "cpu"
    
    def test_reconstruction_loss(self, world_model_loss, sample_loss_input):
        """Test reconstruction loss"""
        loss = world_model_loss.reconstruction_loss(
            pred=sample_loss_input["pred"],
            target=sample_loss_input["target"]
        )
        
        # Loss should be scalar
        assert loss.shape == ()
        assert loss >= 0
        
        # Same prediction and target should give zero loss
        zero_loss = world_model_loss.reconstruction_loss(
            pred=sample_loss_input["pred"],
            target=sample_loss_input["pred"]
        )
        assert torch.allclose(zero_loss, torch.tensor(0.0), atol=1e-6)
    
    def test_perceptual_loss(self, world_model_loss, sample_loss_input):
        """Test perceptual loss"""
        with patch.object(world_model_loss, 'perceptual_net') as mock_net:
            # Mock feature extraction
            mock_features = torch.randn(2, 256, 16, 16)
            mock_net.return_value = mock_features
            
            loss = world_model_loss.perceptual_loss(
                pred=sample_loss_input["pred"],
                target=sample_loss_input["target"]
            )
            
            # Loss should be scalar and non-negative
            assert loss.shape == ()
            assert loss >= 0
    
    def test_kl_divergence_loss(self, world_model_loss, sample_loss_input):
        """Test KL divergence loss"""
        loss = world_model_loss.kl_loss(
            mu=sample_loss_input["mu"],
            logvar=sample_loss_input["logvar"]
        )
        
        # KL loss should be scalar and non-negative
        assert loss.shape == ()
        assert loss >= 0
        
        # KL to standard normal should be positive unless mu=0, logvar=0
        assert loss > 0 or torch.allclose(sample_loss_input["mu"], torch.zeros_like(sample_loss_input["mu"]))
    
    def test_adversarial_loss(self, world_model_loss, sample_loss_input):
        """Test adversarial loss"""
        # Generator loss (trying to fool discriminator)
        gen_loss = world_model_loss.generator_adversarial_loss(
            fake_score=sample_loss_input["fake_score"]
        )
        
        # Discriminator loss
        disc_loss = world_model_loss.discriminator_loss(
            real_score=sample_loss_input["real_score"],
            fake_score=sample_loss_input["fake_score"]
        )
        
        # Both should be scalar
        assert gen_loss.shape == ()
        assert disc_loss.shape == ()
        
        # Discriminator loss should have real and fake components
        assert "real_loss" in disc_loss
        assert "fake_loss" in disc_loss
        assert "total_loss" in disc_loss
    
    def test_composite_loss(self, world_model_loss, sample_loss_input):
        """Test composite loss computation"""
        total_loss, components = world_model_loss.compute_total_loss(
            pred=sample_loss_input["pred"],
            target=sample_loss_input["target"],
            mu=sample_loss_input["mu"],
            logvar=sample_loss_input["logvar"],
            real_score=sample_loss_input["real_score"],
            fake_score=sample_loss_input["fake_score"]
        )
        
        # Verify components
        assert "total" in components
        assert "reconstruction" in components
        assert "perceptual" in components
        assert "kl" in components
        assert "adversarial" in components
        
        # Total loss should be weighted sum of components
        expected_total = (
            world_model_loss.reconstruction_weight * components["reconstruction"] +
            world_model_loss.perceptual_weight * components["perceptual"] +
            world_model_loss.kl_weight * components["kl"] +
            world_model_loss.adversarial_weight * components["adversarial"]
        )
        
        assert torch.allclose(total_loss, expected_total)


class TestWorldModelTrainer:
    """Tests for world model trainer"""
    
    @pytest.fixture
    def world_model_trainer(self):
        """Create WorldModelTrainer instance"""
        return WorldModelTrainer(
            model=None,  # Will be mocked
            optimizer="adam",
            learning_rate=1e-4,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_training_batch(self):
        """Create sample training batch"""
        batch_size = 4
        
        # Video frames
        frames = torch.randn(batch_size, 16, 3, 64, 64)  # Batch, Time, Channels, H, W
        
        # Text descriptions
        texts = ["description"] * batch_size
        
        # Camera poses
        poses = torch.randn(batch_size, 16, 4, 4)
        
        return {
            "frames": frames,
            "texts": texts,
            "poses": poses
        }
    
    def test_world_model_trainer_initialization(self, world_model_trainer):
        """Test WorldModelTrainer initialization"""
        assert world_model_trainer.optimizer_name == "adam"
        assert world_model_trainer.learning_rate == 1e-4
        assert world_model_trainer.device == "cpu"
        
        # Optimizer should be created when model is set
        assert world_model_trainer.optimizer is None
        assert world_model_trainer.scheduler is None
    
    def test_training_step(self, world_model_trainer, sample_training_batch):
        """Test training step"""
        # Mock model
        mock_model = Mock()
        mock_model.return_value = {
            "predicted_frames": torch.randn_like(sample_training_batch["frames"]),
            "mu": torch.randn(sample_training_batch["frames"].shape[0], 512),
            "logvar": torch.randn(sample_training_batch["frames"].shape[0], 512)
        }
        
        # Mock discriminator
        mock_discriminator = Mock()
        mock_discriminator.return_value = torch.rand(sample_training_batch["frames"].shape[0], 1)
        
        world_model_trainer.model = mock_model
        world_model_trainer.discriminator = mock_discriminator
        
        # Mock loss computation
        with patch.object(world_model_trainer, 'loss_function') as mock_loss:
            mock_loss.return_value = (torch.tensor(0.1), {
                "total": torch.tensor(0.1),
                "reconstruction": torch.tensor(0.05),
                "perceptual": torch.tensor(0.03),
                "kl": torch.tensor(0.01),
                "adversarial": torch.tensor(0.01)
            })
            
            # Create optimizer
            world_model_trainer.optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
            
            # Training step
            result = world_model_trainer.training_step(sample_training_batch, 0)
            
            # Verify training step result
            assert "loss" in result
            assert "loss_components" in result
            assert "grad_norm" in result
            assert "learning_rate" in result
    
    def test_validation_step(self, world_model_trainer, sample_training_batch):
        """Test validation step"""
        # Mock model (same as training)
        mock_model = Mock()
        mock_model.return_value = {
            "predicted_frames": torch.randn_like(sample_training_batch["frames"]),
            "mu": torch.randn(sample_training_batch["frames"].shape[0], 512),
            "logvar": torch.randn(sample_training_batch["frames"].shape[0], 512)
        }
        
        world_model_trainer.model = mock_model
        
        # Mock metrics computation
        with patch.object(world_model_trainer, 'compute_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "psnr": 25.0,
                "ssim": 0.85,
                "lpips": 0.15,
                "fid": 30.0
            }
            
            # Validation step
            result = world_model_trainer.validation_step(sample_training_batch, 0)
            
            # Verify validation result
            assert "val_loss" in result
            assert "metrics" in result
    
    def test_learning_rate_scheduling(self, world_model_trainer):
        """Test learning rate scheduling"""
        # Create mock model and optimizer
        mock_model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=1e-4)
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        world_model_trainer.optimizer = optimizer
        world_model_trainer.scheduler = scheduler
        
        # Check initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        assert initial_lr == 1e-4
        
        # Step scheduler
        scheduler.step()
        
        # Learning rate should be reduced
        new_lr = optimizer.param_groups[0]['lr']
        assert new_lr == 1e-4 * 0.5
    
    def test_gradient_clipping(self, world_model_trainer):
        """Test gradient clipping"""
        # Create model with large gradients
        model = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        world_model_trainer.optimizer = optimizer
        
        # Create large loss
        x = torch.randn(2, 10)
        y = torch.randn(2, 10)
        
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        
        # Backward with large loss
        loss.backward()
        
        # Check gradients before clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.norm().item() ** 2
        total_norm_before = total_norm_before ** 0.5
        
        # Apply gradient clipping
        world_model_trainer.clip_gradients(max_norm=1.0)
        
        # Check gradients after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.norm().item() ** 2
        total_norm_after = total_norm_after ** 0.5
        
        # Gradient norm should be clipped
        if total_norm_before > 1.0:
            assert total_norm_after <= 1.0 + 1e-6


class TestWorldGenerator:
    """Tests for world generator"""
    
    @pytest.fixture
    def world_generator(self):
        """Create WorldGenerator instance"""
        return WorldGenerator(
            model_path="models/world_model/checkpoint.pt",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_generation_input(self):
        """Create sample input for generation"""
        return {
            "prompt": "A beautiful mountain landscape",
            "num_frames": 30,
            "resolution": (256, 256),
            "seed": 42
        }
    
    def test_world_generator_initialization(self, world_generator):
        """Test WorldGenerator initialization"""
        assert world_generator.model_path == "models/world_model/checkpoint.pt"
        assert world_generator.device == "cpu"
        
        # Components should be initialized (may be None until loaded)
        assert hasattr(world_generator, 'text_encoder')
        assert hasattr(world_generator, 'world_model')
        assert hasattr(world_generator, 'renderer')
    
    def test_generate_world(self, world_generator, sample_generation_input):
        """Test world generation"""
        with patch.object(world_generator, 'text_encoder') as mock_text_encoder:
            with patch.object(world_generator, 'world_model') as mock_world_model:
                with patch.object(world_generator, 'renderer') as mock_renderer:
                    
                    # Mock components
                    mock_text_encoder.encode.return_value = {
                        "pooled_embedding": torch.randn(1, 512)
                    }
                    
                    mock_world_model.generate.return_value = {
                        "latent_representation": torch.randn(1, 256, 32, 32, 32),
                        "camera_params": {
                            "positions": torch.randn(30, 3),
                            "rotations": torch.randn(30, 4)
                        }
                    }
                    
                    mock_renderer.render.return_value = {
                        "video": torch.randn(30, 256, 256, 3),
                        "depth": torch.randn(30, 256, 256),
                        "normals": torch.randn(30, 256, 256, 3)
                    }
                    
                    # Generate world
                    result = world_generator.generate(**sample_generation_input)
                    
                    # Verify result structure
                    assert "video" in result
                    assert "depth" in result
                    assert "normals" in result
                    assert "metadata" in result
                    
                    # Verify shapes
                    assert result["video"].shape == (30, 256, 256, 3)
                    assert result["depth"].shape == (30, 256, 256)
                    assert result["normals"].shape == (30, 256, 256, 3)
                    
                    # Verify metadata
                    metadata = result["metadata"]
                    assert metadata["prompt"] == sample_generation_input["prompt"]
                    assert metadata["num_frames"] == sample_generation_input["num_frames"]
                    assert metadata["resolution"] == sample_generation_input["resolution"]
    
    def test_batch_generation(self, world_generator):
        """Test batch generation"""
        prompts = [
            "A mountain landscape",
            "An ocean view",
            "A forest path"
        ]
        
        with patch.object(world_generator, 'generate') as mock_generate:
            # Mock single generation
            mock_result = {
                "video": torch.randn(10, 128, 128, 3),
                "metadata": {"prompt": "test"}
            }
            mock_generate.return_value = mock_result
            
            # Batch generate
            results = world_generator.batch_generate(
                prompts=prompts,
                num_frames=10,
                resolution=(128, 128)
            )
            
            # Should generate for each prompt
            assert len(results) == len(prompts)
            
            # Each result should have video
            for result in results:
                assert "video" in result
                assert result["video"].shape == (10, 128, 128, 3)
    
    def test_generation_with_guidance(self, world_generator, sample_generation_input):
        """Test generation with guidance scale"""
        with patch.object(world_generator, 'world_model') as mock_world_model:
            mock_world_model.generate.return_value = {
                "latent_representation": torch.randn(1, 256, 32, 32, 32)
            }
            
            # Test different guidance scales
            for guidance_scale in [1.0, 3.0, 7.0, 15.0]:
                result = world_generator.generate(
                    **sample_generation_input,
                    guidance_scale=guidance_scale
                )
                
                # Should complete without error
                assert result is not None
    
    def test_interpolation(self, world_generator):
        """Test interpolation between prompts"""
        prompt_a = "A sunny beach"
        prompt_b = "A snowy mountain"
        
        with patch.object(world_generator, 'text_encoder') as mock_text_encoder:
            # Mock embeddings
            embedding_a = torch.randn(1, 512)
            embedding_b = torch.randn(1, 512)
            
            mock_text_encoder.encode.side_effect = lambda prompts: {
                "pooled_embedding": embedding_a if prompts[0] == prompt_a else embedding_b
            }
            
            # Interpolate
            results = world_generator.interpolate(
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                num_steps=5,
                num_frames=10,
                resolution=(128, 128)
            )
            
            # Should generate interpolation sequence
            assert len(results) == 5
            
            # Each step should have video
            for result in results:
                assert "video" in result
                assert "interpolation_weight" in result


class TestSampler:
    """Tests for diffusion sampler"""
    
    @pytest.fixture
    def sampler(self):
        """Create Sampler instance"""
        return Sampler(
            model=None,  # Will be mocked
            sampler_type="ddim",
            num_steps=50,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_sampler_input(self):
        """Create sample input for sampling"""
        batch_size = 2
        shape = (4, 32, 32)  # Channels, Height, Width
        
        # Initial noise
        x_start = torch.randn(batch_size, *shape)
        
        # Conditioning
        cond = torch.randn(batch_size, 512)
        
        return x_start, cond
    
    def test_sampler_initialization(self, sampler):
        """Test Sampler initialization"""
        assert sampler.sampler_type == "ddim"
        assert sampler.num_steps == 50
        assert sampler.device == "cpu"
        
        # Sampler-specific parameters
        assert hasattr(sampler, 'sampling_method')
    
    def test_ddim_sampling(self, sampler, sample_sampler_input):
        """Test DDIM sampling"""
        x_start, cond = sample_sampler_input
        batch_size = x_start.shape[0]
        
        # Mock model
        mock_model = Mock()
        
        def model_forward(x, t, cond):
            # Simple mock: predict -x (denoising towards zero)
            return -x
        
        mock_model.side_effect = model_forward
        sampler.model = mock_model
        
        # DDIM sampling
        samples = sampler.ddim_sample(
            shape=x_start.shape[1:],
            batch_size=batch_size,
            conditioning=cond
        )
        
        # Should generate samples
        assert samples.shape == x_start.shape
    
    def test_ddpm_sampling(self, sampler, sample_sampler_input):
        """Test DDPM sampling"""
        x_start, cond = sample_sampler_input
        batch_size = x_start.shape[0]
        
        # Mock model
        mock_model = Mock()
        mock_model.side_effect = lambda x, t, cond: -x
        
        sampler.model = mock_model
        sampler.sampler_type = "ddpm"
        
        # DDPM sampling
        samples = sampler.ddpm_sample(
            shape=x_start.shape[1:],
            batch_size=batch_size,
            conditioning=cond
        )
        
        # Should generate samples
        assert samples.shape == x_start.shape
    
    def test_plms_sampling(self, sampler, sample_sampler_input):
        """Test PLMS sampling"""
        x_start, cond = sample_sampler_input
        batch_size = x_start.shape[0]
        
        # Mock model
        mock_model = Mock()
        mock_model.side_effect = lambda x, t, cond: -x
        
        sampler.model = mock_model
        sampler.sampler_type = "plms"
        
        # PLMS sampling
        samples = sampler.plms_sample(
            shape=x_start.shape[1:],
            batch_size=batch_size,
            conditioning=cond
        )
        
        # Should generate samples
        assert samples.shape == x_start.shape
    
    def test_cfg_sampling(self, sampler, sample_sampler_input):
        """Test classifier-free guidance sampling"""
        _, cond = sample_sampler_input
        batch_size = cond.shape[0]
        shape = (4, 32, 32)
        
        # Mock model that returns different outputs for conditional/unconditional
        mock_model = Mock()
        
        def model_forward(x, t, cond):
            if cond is None:
                return -x * 0.5  # Unconditional prediction
            else:
                return -x  # Conditional prediction
        
        mock_model.side_effect = model_forward
        sampler.model = mock_model
        
        # Sampling with classifier-free guidance
        samples = sampler.sample_with_cfg(
            shape=shape,
            batch_size=batch_size,
            conditioning=cond,
            guidance_scale=7.5
        )
        
        # Should generate samples
        assert samples.shape == (batch_size, *shape)
    
    def test_sampling_with_latent(self, sampler):
        """Test sampling in latent space"""
        batch_size = 2
        latent_shape = (4, 32, 32)  # Latent dimensions
        
        # Mock model
        mock_model = Mock()
        mock_model.side_effect = lambda x, t, cond: -x
        
        sampler.model = mock_model
        
        # Sample in latent space
        latents = sampler.sample_latents(
            shape=latent_shape,
            batch_size=batch_size,
            conditioning=None
        )
        
        # Should generate latents
        assert latents.shape == (batch_size, *latent_shape)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
