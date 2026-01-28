"""
Unit tests for multimodal encoders.
Tests text, vision, video, and audio encoders.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.multimodal.encoders.text_encoder import TextEncoder
from src.core.multimodal.encoders.vision_encoder import VisionEncoder
from src.core.multimodal.encoders.video_encoder import VideoEncoder
from src.core.multimodal.encoders.audio_encoder import AudioEncoder
from src.core.multimodal.fusion.cross_attention import CrossAttentionFusion
from src.core.multimodal.fusion.transformer_fusion import TransformerFusion
from src.core.multimodal.alignment.contrastive_loss import ContrastiveLoss


class TestTextEncoder:
    """Tests for text encoder"""
    
    @pytest.fixture
    def text_encoder(self):
        """Create TextEncoder instance"""
        return TextEncoder(
            model_name="bert-base-uncased",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_texts(self):
        """Create sample texts for testing"""
        return [
            "A beautiful mountain landscape with flowing rivers",
            "Urban cityscape at night with neon lights",
            "Underwater coral reef with colorful fish",
            "Desert oasis with palm trees and camels",
            "Ancient forest with giant mushrooms"
        ]
    
    def test_text_encoder_initialization(self, text_encoder):
        """Test TextEncoder initialization"""
        assert text_encoder.model_name == "bert-base-uncased"
        assert text_encoder.device == "cpu"
        assert text_encoder.max_length == 512
        assert text_encoder.embedding_dim > 0
    
    def test_tokenization(self, text_encoder, sample_texts):
        """Test text tokenization"""
        tokens = text_encoder.tokenize(sample_texts[0])
        
        # Verify token structure
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert "token_type_ids" in tokens if hasattr(text_encoder.tokenizer, 'token_type_ids') else True
        
        # Input IDs should be integers
        assert tokens["input_ids"].dtype in [torch.int32, torch.int64]
        
        # Attention mask should be binary
        assert set(tokens["attention_mask"].unique().tolist()).issubset({0, 1})
    
    def test_single_text_encoding(self, text_encoder):
        """Test encoding single text"""
        text = "A beautiful landscape"
        
        with patch.object(text_encoder, 'model') as mock_model:
            # Mock model output
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(1, 10, text_encoder.embedding_dim)
            mock_output.pooler_output = torch.randn(1, text_encoder.embedding_dim)
            mock_model.return_value = mock_output
            
            encoding = text_encoder.encode([text], normalize=True)
            
            # Verify encoding structure
            assert "embeddings" in encoding
            assert "pooled_embedding" in encoding
            assert "hidden_states" in encoding if text_encoder.return_hidden_states else True
            
            # Verify shapes
            assert encoding["embeddings"].shape == (1, 10, text_encoder.embedding_dim)
            assert encoding["pooled_embedding"].shape == (1, text_encoder.embedding_dim)
            
            # Verify normalization
            if normalize:
                norm = torch.norm(encoding["pooled_embedding"], dim=-1)
                assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)
    
    def test_batch_text_encoding(self, text_encoder, sample_texts):
        """Test batch text encoding"""
        batch_size = len(sample_texts)
        
        with patch.object(text_encoder, 'model') as mock_model:
            # Mock model output for batch
            mock_output = Mock()
            mock_output.last_hidden_state = torch.randn(batch_size, 15, text_encoder.embedding_dim)
            mock_output.pooler_output = torch.randn(batch_size, text_encoder.embedding_dim)
            mock_model.return_value = mock_output
            
            encoding = text_encoder.encode(sample_texts, normalize=False)
            
            # Verify batch dimensions
            assert encoding["embeddings"].shape[0] == batch_size
            assert encoding["pooled_embedding"].shape[0] == batch_size
    
    def test_text_similarity(self, text_encoder):
        """Test text similarity computation"""
        texts1 = ["a cat sitting on a mat", "a dog playing in the park"]
        texts2 = ["a feline resting on a carpet", "a canine running in a field"]
        
        with patch.object(text_encoder, 'encode') as mock_encode:
            # Mock embeddings
            mock_embeddings1 = torch.randn(2, text_encoder.embedding_dim)
            mock_embeddings2 = torch.randn(2, text_encoder.embedding_dim)
            
            def encode_side_effect(texts, **kwargs):
                if texts == texts1:
                    return {"pooled_embedding": mock_embeddings1}
                else:
                    return {"pooled_embedding": mock_embeddings2}
            
            mock_encode.side_effect = encode_side_effect
            
            similarity = text_encoder.compute_similarity(texts1, texts2)
            
            # Verify similarity matrix
            assert similarity.shape == (2, 2)
            assert torch.all(similarity >= -1) and torch.all(similarity <= 1)
            
            # Diagonal should have higher similarity (same semantic meaning)
            assert similarity[0, 0] > similarity[0, 1]  # Cat vs cat > cat vs dog
            assert similarity[1, 1] > similarity[1, 0]  # Dog vs dog > dog vs cat
    
    def test_text_classification(self, text_encoder):
        """Test text classification"""
        texts = ["positive review", "negative review", "neutral review"]
        labels = ["positive", "negative", "neutral"]
        
        with patch.object(text_encoder, 'model') as mock_model:
            # Mock classifier head
            mock_output = Mock()
            mock_output.logits = torch.randn(3, len(labels))
            mock_model.return_value = mock_output
            
            predictions = text_encoder.classify(texts, labels)
            
            # Verify predictions
            assert "predictions" in predictions
            assert "probabilities" in predictions
            assert "confidences" in predictions
            
            # Should have prediction for each text
            assert len(predictions["predictions"]) == len(texts)
            
            # Probabilities should sum to 1 for each sample
            probs = predictions["probabilities"]
            assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)
    
    def test_text_generation(self, text_encoder):
        """Test text generation capabilities"""
        prompt = "Once upon a time"
        
        with patch.object(text_encoder, 'generation_model') as mock_generator:
            # Mock generation output
            mock_output = ["Once upon a time in a faraway land."]
            mock_generator.generate.return_value = mock_output
            
            generated = text_encoder.generate_text(
                prompt=prompt,
                max_length=50,
                temperature=0.7
            )
            
            # Verify generation
            assert "generated_text" in generated
            assert "tokens" in generated
            assert "logits" in generated
            
            # Generated text should contain prompt
            assert prompt.lower() in generated["generated_text"].lower()
    
    def test_attention_visualization(self, text_encoder):
        """Test attention visualization"""
        text = "The quick brown fox jumps over the lazy dog"
        
        with patch.object(text_encoder, 'model') as mock_model:
            # Mock attention weights
            mock_output = Mock()
            mock_output.attentions = [torch.randn(1, 12, 10, 10) for _ in range(12)]  # 12 layers
            mock_model.return_value = mock_output
            
            attention = text_encoder.get_attention(text, layer=6, head=3)
            
            # Verify attention structure
            assert "attention_weights" in attention
            assert "tokens" in attention
            assert "visualization" in attention
            
            # Attention weights should be probabilities
            weights = attention["attention_weights"]
            assert torch.all(weights >= 0) and torch.all(weights <= 1)
            assert torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[:-1]), atol=1e-6)


class TestVisionEncoder:
    """Tests for vision encoder"""
    
    @pytest.fixture
    def vision_encoder(self):
        """Create VisionEncoder instance"""
        return VisionEncoder(
            model_name="resnet50",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing"""
        return [
            np.random.rand(224, 224, 3).astype(np.float32) for _ in range(3)
        ]
    
    def test_vision_encoder_initialization(self, vision_encoder):
        """Test VisionEncoder initialization"""
        assert vision_encoder.model_name == "resnet50"
        assert vision_encoder.device == "cpu"
        assert vision_encoder.input_size == (224, 224)
        assert vision_encoder.embedding_dim > 0
    
    def test_image_preprocessing(self, vision_encoder):
        """Test image preprocessing"""
        image = np.random.rand(300, 400, 3).astype(np.uint8)
        
        processed = vision_encoder.preprocess(image)
        
        # Verify preprocessing
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (3, 224, 224)  # CHW format
        assert processed.dtype == torch.float32
        
        # Values should be normalized
        assert torch.all(processed >= -3) and torch.all(processed <= 3)  # After normalization
    
    def test_single_image_encoding(self, vision_encoder):
        """Test encoding single image"""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        
        with patch.object(vision_encoder, 'model') as mock_model:
            # Mock model output
            mock_output = torch.randn(1, vision_encoder.embedding_dim)
            mock_model.return_value = mock_output
            
            encoding = vision_encoder.encode([image], normalize=True)
            
            # Verify encoding
            assert "embeddings" in encoding
            assert "features" in encoding if vision_encoder.return_features else True
            
            # Verify shapes
            assert encoding["embeddings"].shape == (1, vision_encoder.embedding_dim)
            
            # Verify normalization
            if normalize:
                norm = torch.norm(encoding["embeddings"], dim=-1)
                assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)
    
    def test_batch_image_encoding(self, vision_encoder, sample_images):
        """Test batch image encoding"""
        batch_size = len(sample_images)
        
        with patch.object(vision_encoder, 'model') as mock_model:
            # Mock model output
            mock_output = torch.randn(batch_size, vision_encoder.embedding_dim)
            mock_model.return_value = mock_output
            
            encoding = vision_encoder.encode(sample_images, normalize=False)
            
            # Verify batch dimensions
            assert encoding["embeddings"].shape[0] == batch_size
    
    def test_feature_extraction(self, vision_encoder):
        """Test feature extraction at different layers"""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        
        with patch.object(vision_encoder, 'model') as mock_model:
            # Mock feature extraction
            mock_features = {
                'layer1': torch.randn(1, 256, 56, 56),
                'layer2': torch.randn(1, 512, 28, 28),
                'layer3': torch.randn(1, 1024, 14, 14),
                'layer4': torch.randn(1, 2048, 7, 7)
            }
            
            def hook_fn(module, input, output):
                layer_name = module._get_name()
                if layer_name in mock_features:
                    return mock_features[layer_name]
                return output
            
            # Apply hooks
            for name, module in vision_encoder.model.named_modules():
                if name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    module.register_forward_hook(hook_fn)
            
            features = vision_encoder.extract_features(image, layers=['layer3', 'layer4'])
            
            # Verify feature extraction
            assert "layer3" in features
            assert "layer4" in features
            assert features["layer3"].shape == (1, 1024, 14, 14)
            assert features["layer4"].shape == (1, 2048, 7, 7)
    
    def test_image_similarity(self, vision_encoder):
        """Test image similarity computation"""
        images1 = [np.random.rand(224, 224, 3) for _ in range(2)]
        images2 = [np.random.rand(224, 224, 3) for _ in range(2)]
        
        with patch.object(vision_encoder, 'encode') as mock_encode:
            # Mock embeddings
            mock_embeddings1 = torch.randn(2, vision_encoder.embedding_dim)
            mock_embeddings2 = torch.randn(2, vision_encoder.embedding_dim)
            
            def encode_side_effect(images, **kwargs):
                if images == images1:
                    return {"embeddings": mock_embeddings1}
                else:
                    return {"embeddings": mock_embeddings2}
            
            mock_encode.side_effect = encode_side_effect
            
            similarity = vision_encoder.compute_similarity(images1, images2)
            
            # Verify similarity matrix
            assert similarity.shape == (2, 2)
            assert torch.all(similarity >= -1) and torch.all(similarity <= 1)
    
    def test_zero_shot_classification(self, vision_encoder):
        """Test zero-shot image classification"""
        image = np.random.rand(224, 224, 3).astype(np.float32)
        class_names = ["cat", "dog", "bird", "car", "tree"]
        
        with patch.object(vision_encoder, 'encode') as mock_encode:
            with patch.object(TextEncoder, 'encode') as mock_text_encode:
                # Mock image embedding
                mock_image_embedding = torch.randn(1, vision_encoder.embedding_dim)
                
                # Mock text embeddings for classes
                mock_text_embeddings = torch.randn(len(class_names), vision_encoder.embedding_dim)
                
                mock_encode.return_value = {"embeddings": mock_image_embedding}
                mock_text_encode.return_value = {"pooled_embedding": mock_text_embeddings}
                
                predictions = vision_encoder.zero_shot_classify(image, class_names)
                
                # Verify predictions
                assert "predictions" in predictions
                assert "probabilities" in predictions
                assert "class_scores" in predictions
                
                # Should have prediction for each class
                assert len(predictions["predictions"]) == len(class_names)
                
                # Probabilities should sum to 1
                probs = predictions["probabilities"]
                assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


class TestVideoEncoder:
    """Tests for video encoder"""
    
    @pytest.fixture
    def video_encoder(self):
        """Create VideoEncoder instance"""
        return VideoEncoder(
            model_name="slowfast",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_video(self):
        """Create sample video for testing"""
        # 16 frames of 224x224 video
        return np.random.rand(16, 224, 224, 3).astype(np.float32)
    
    @pytest.fixture
    def sample_video_batch(self):
        """Create batch of sample videos"""
        return [
            np.random.rand(8, 224, 224, 3).astype(np.float32) for _ in range(2)
        ]
    
    def test_video_encoder_initialization(self, video_encoder):
        """Test VideoEncoder initialization"""
        assert video_encoder.model_name == "slowfast"
        assert video_encoder.device == "cpu"
        assert video_encoder.temporal_stride > 0
        assert video_encoder.num_frames > 0
    
    def test_video_preprocessing(self, video_encoder, sample_video):
        """Test video preprocessing"""
        processed = video_encoder.preprocess(sample_video)
        
        # Verify preprocessing
        assert isinstance(processed, torch.Tensor)
        assert len(processed.shape) == 5  # BTCHW format
        assert processed.shape[1] == 3  # RGB channels
        assert processed.shape[2] == video_encoder.num_frames or processed.shape[2] <= sample_video.shape[0]
    
    def test_video_encoding(self, video_encoder, sample_video):
        """Test video encoding"""
        with patch.object(video_encoder, 'model') as mock_model:
            # Mock model output
            mock_output = torch.randn(1, video_encoder.embedding_dim)
            mock_model.return_value = mock_output
            
            encoding = video_encoder.encode(sample_video)
            
            # Verify encoding structure
            assert "video_embedding" in encoding
            assert "temporal_features" in encoding if video_encoder.return_temporal else True
            assert "spatial_features" in encoding if video_encoder.return_spatial else True
            
            # Verify embedding shape
            assert encoding["video_embedding"].shape == (1, video_encoder.embedding_dim)
    
    def test_temporal_feature_extraction(self, video_encoder, sample_video):
        """Test temporal feature extraction"""
        with patch.object(video_encoder, 'model') as mock_model:
            # Mock temporal features
            mock_features = torch.randn(1, 16, 512)  # Batch, Time, Features
            mock_model.extract_temporal_features.return_value = mock_features
            
            features = video_encoder.extract_temporal_features(sample_video)
            
            # Verify temporal features
            assert features.shape == (1, 16, 512)
            
            # Features should vary over time
            temporal_var = features.var(dim=1)  # Variance over time dimension
            assert temporal_var.mean() > 0
    
    def test_action_recognition(self, video_encoder, sample_video):
        """Test action recognition"""
        action_classes = ["walking", "running", "jumping", "sitting", "standing"]
        
        with patch.object(video_encoder, 'model') as mock_model:
            # Mock action recognition output
            mock_logits = torch.randn(1, len(action_classes))
            mock_model.classify.return_value = mock_logits
            
            predictions = video_encoder.recognize_actions(
                video=sample_video,
                action_classes=action_classes
            )
            
            # Verify predictions
            assert "actions" in predictions
            assert "confidences" in predictions
            assert "temporal_segments" in predictions
            
            # Should have predictions
            assert len(predictions["actions"]) > 0
            
            # Confidences should be probabilities
            confidences = predictions["confidences"]
            assert torch.all(confidences >= 0) and torch.all(confidences <= 1)
    
    def test_optical_flow_computation(self, video_encoder, sample_video):
        """Test optical flow computation"""
        flow = video_encoder.compute_optical_flow(sample_video)
        
        # Verify optical flow
        assert flow.shape == (sample_video.shape[0] - 1, *sample_video.shape[1:3], 2)
        assert flow.dtype == np.float32
        
        # Flow values should be reasonable
        assert np.abs(flow).max() < 100  # Pixels per frame
    
    def test_temporal_consistency_check(self, video_encoder, sample_video):
        """Test temporal consistency check"""
        consistency = video_encoder.check_temporal_consistency(sample_video)
        
        # Verify consistency metrics
        assert "consistency_score" in consistency
        assert "inconsistency_frames" in consistency
        assert "flow_consistency" in consistency
        
        # Score should be between 0 and 1
        assert 0 <= consistency["consistency_score"] <= 1


class TestAudioEncoder:
    """Tests for audio encoder (future expansion)"""
    
    @pytest.fixture
    def audio_encoder(self):
        """Create AudioEncoder instance"""
        return AudioEncoder(
            model_name="wav2vec2",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio for testing"""
        # 3 seconds of audio at 16kHz
        return np.random.randn(48000).astype(np.float32)
    
    def test_audio_encoder_initialization(self, audio_encoder):
        """Test AudioEncoder initialization"""
        assert audio_encoder.model_name == "wav2vec2"
        assert audio_encoder.device == "cpu"
        assert audio_encoder.sample_rate == 16000
        assert audio_encoder.embedding_dim > 0
    
    def test_audio_preprocessing(self, audio_encoder, sample_audio):
        """Test audio preprocessing"""
        processed = audio_encoder.preprocess(sample_audio)
        
        # Verify preprocessing
        assert isinstance(processed, torch.Tensor)
        assert len(processed.shape) == 2  # Batch, Time
        assert processed.shape[0] == 1  # Single audio sample
        assert processed.dtype == torch.float32
    
    def test_audio_encoding(self, audio_encoder, sample_audio):
        """Test audio encoding"""
        with patch.object(audio_encoder, 'model') as mock_model:
            # Mock model output
            mock_features = torch.randn(1, 100, audio_encoder.embedding_dim)
            mock_model.return_value = mock_features
            
            encoding = audio_encoder.encode(sample_audio)
            
            # Verify encoding
            assert "audio_features" in encoding
            assert "pooled_features" in encoding
            
            # Verify feature shape
            assert encoding["audio_features"].shape == (1, 100, audio_encoder.embedding_dim)
    
    def test_speech_recognition(self, audio_encoder, sample_audio):
        """Test speech recognition"""
        with patch.object(audio_encoder, 'model') as mock_model:
            # Mock transcription
            mock_transcription = "this is a test transcription"
            mock_model.transcribe.return_value = mock_transcription
            
            result = audio_encoder.transcribe(sample_audio)
            
            # Verify transcription
            assert "transcription" in result
            assert "confidence" in result
            assert "timestamps" in result
            
            assert result["transcription"] == mock_transcription
    
    def test_audio_classification(self, audio_encoder, sample_audio):
        """Test audio classification"""
        sound_classes = ["speech", "music", "noise", "silence"]
        
        with patch.object(audio_encoder, 'model') as mock_model:
            # Mock classification
            mock_logits = torch.randn(1, len(sound_classes))
            mock_model.classify.return_value = mock_logits
            
            predictions = audio_encoder.classify_sound(sample_audio, sound_classes)
            
            # Verify predictions
            assert "predictions" in predictions
            assert "probabilities" in predictions
            
            # Should have prediction for each class
            assert len(predictions["predictions"]) == len(sound_classes)


class TestCrossAttentionFusion:
    """Tests for cross-attention fusion"""
    
    @pytest.fixture
    def cross_attention(self):
        """Create CrossAttentionFusion instance"""
        return CrossAttentionFusion(
            text_dim=512,
            vision_dim=768,
            hidden_dim=1024,
            num_heads=8
        )
    
    @pytest.fixture
    def sample_modalities(self):
        """Create sample modality embeddings"""
        batch_size = 2
        
        text_embeddings = torch.randn(batch_size, 10, 512)  # Batch, SeqLen, Dim
        vision_embeddings = torch.randn(batch_size, 196, 768)  # Batch, Patches, Dim
        
        return {
            "text": text_embeddings,
            "vision": vision_embeddings
        }
    
    def test_cross_attention_initialization(self, cross_attention):
        """Test CrossAttentionFusion initialization"""
        assert cross_attention.text_dim == 512
        assert cross_attention.vision_dim == 768
        assert cross_attention.hidden_dim == 1024
        assert cross_attention.num_heads == 8
        
        # Check that attention layers are created
        assert hasattr(cross_attention, 'text_to_vision_attention')
        assert hasattr(cross_attention, 'vision_to_text_attention')
    
    def test_text_to_vision_attention(self, cross_attention, sample_modalities):
        """Test text-to-vision attention"""
        result = cross_attention.text_to_vision_attention(
            text_embeddings=sample_modalities["text"],
            vision_embeddings=sample_modalities["vision"]
        )
        
        # Verify result structure
        assert "attended_vision" in result
        assert "attention_weights" in result
        assert "cross_modal_features" in result
        
        # Verify shapes
        batch_size = sample_modalities["text"].shape[0]
        assert result["attended_vision"].shape == (batch_size, 196, 1024)
        assert result["attention_weights"].shape == (batch_size, 8, 10, 196)  # Heads, TextSeq, VisionSeq
    
    def test_vision_to_text_attention(self, cross_attention, sample_modalities):
        """Test vision-to-text attention"""
        result = cross_attention.vision_to_text_attention(
            vision_embeddings=sample_modalities["vision"],
            text_embeddings=sample_modalities["text"]
        )
        
        # Verify result structure
        assert "attended_text" in result
        assert "attention_weights" in result
        
        # Verify shapes
        batch_size = sample_modalities["vision"].shape[0]
        assert result["attended_text"].shape == (batch_size, 10, 1024)
        assert result["attention_weights"].shape == (batch_size, 8, 196, 10)  # Heads, VisionSeq, TextSeq
    
    def test_bidirectional_attention(self, cross_attention, sample_modalities):
        """Test bidirectional attention fusion"""
        result = cross_attention.bidirectional_attention(
            text_embeddings=sample_modalities["text"],
            vision_embeddings=sample_modalities["vision"]
        )
        
        # Verify result structure
        assert "fused_features" in result
        assert "text_attended_vision" in result
        assert "vision_attended_text" in result
        assert "attention_weights" in result
        
        # Fused features should combine both modalities
        assert result["fused_features"].shape[0] == sample_modalities["text"].shape[0]
        assert result["fused_features"].shape[-1] == cross_attention.hidden_dim
    
    def test_multi_head_attention_visualization(self, cross_attention, sample_modalities):
        """Test multi-head attention visualization"""
        result = cross_attention.text_to_vision_attention(
            text_embeddings=sample_modalities["text"],
            vision_embeddings=sample_modalities["vision"]
        )
        
        # Extract attention weights for visualization
        attention_weights = result["attention_weights"]  # Batch, Heads, TextSeq, VisionSeq
        
        # Test visualization generation
        visualization = cross_attention.visualize_attention(
            attention_weights=attention_weights[0],  # First batch
            text_tokens=["token"] * 10,
            vision_patches=196
        )
        
        # Verify visualization
        assert "attention_maps" in visualization
        assert "head_importance" in visualization
        assert "aggregated_attention" in visualization
        
        # Should have visualization for each head
        assert len(visualization["attention_maps"]) == cross_attention.num_heads


class TestTransformerFusion:
    """Tests for transformer fusion"""
    
    @pytest.fixture
    def transformer_fusion(self):
        """Create TransformerFusion instance"""
        return TransformerFusion(
            input_dims={"text": 512, "vision": 768},
            hidden_dim=1024,
            num_layers=4,
            num_heads=8
        )
    
    @pytest.fixture
    def sample_multimodal_input(self):
        """Create sample multimodal input"""
        batch_size = 2
        
        return {
            "text": torch.randn(batch_size, 10, 512),
            "vision": torch.randn(batch_size, 196, 768),
            "audio": torch.randn(batch_size, 100, 256)  # Optional third modality
        }
    
    def test_transformer_fusion_initialization(self, transformer_fusion):
        """Test TransformerFusion initialization"""
        assert transformer_fusion.hidden_dim == 1024
        assert transformer_fusion.num_layers == 4
        assert transformer_fusion.num_heads == 8
        
        # Check transformer layers
        assert hasattr(transformer_fusion, 'input_projections')
        assert hasattr(transformer_fusion, 'transformer')
        assert hasattr(transformer_fusion, 'output_projection')
    
    def test_input_projection(self, transformer_fusion, sample_multimodal_input):
        """Test input projection to common dimension"""
        projected = transformer_fusion.project_inputs(sample_multimodal_input)
        
        # Verify all modalities projected to hidden_dim
        for modality, embedding in projected.items():
            assert embedding.shape[-1] == transformer_fusion.hidden_dim
        
        # Verify batch size preserved
        assert projected["text"].shape[0] == sample_multimodal_input["text"].shape[0]
    
    def test_multimodal_fusion(self, transformer_fusion, sample_multimodal_input):
        """Test multimodal fusion with transformer"""
        result = transformer_fusion.fuse(sample_multimodal_input)
        
        # Verify result structure
        assert "fused_embeddings" in result
        assert "modality_embeddings" in result
        assert "attention_weights" in result
        
        # Fused embeddings should have same hidden dimension
        assert result["fused_embeddings"].shape[-1] == transformer_fusion.hidden_dim
        
        # Should have embeddings for each modality
        for modality in sample_multimodal_input.keys():
            assert modality in result["modality_embeddings"]
    
    def test_cross_modal_interaction(self, transformer_fusion, sample_multimodal_input):
        """Test cross-modal interaction in transformer"""
        result = transformer_fusion.fuse(sample_multimodal_input)
        
        # Extract attention weights
        attention_weights = result["attention_weights"]
        
        # Verify attention structure
        assert len(attention_weights) == transformer_fusion.num_layers
        
        # Each layer should have attention for each head
        for layer_weights in attention_weights:
            assert layer_weights.shape[1] == transformer_fusion.num_heads  # Number of heads
    
    def test_modality_dropout(self, transformer_fusion, sample_multimodal_input):
        """Test modality dropout for robustness"""
        # Apply modality dropout
        transformer_fusion.modality_dropout_rate = 0.5
        
        result = transformer_fusion.fuse_with_dropout(
            sample_multimodal_input,
            dropout_rate=0.5
        )
        
        # Verify fusion still works with dropped modalities
        assert "fused_embeddings" in result
        
        # Fused embeddings should have correct shape even with dropout
        assert result["fused_embeddings"].shape[-1] == transformer_fusion.hidden_dim


class TestContrastiveLoss:
    """Tests for contrastive loss"""
    
    @pytest.fixture
    def contrastive_loss(self):
        """Create ContrastiveLoss instance"""
        return ContrastiveLoss(
            temperature=0.07,
            similarity_metric="cosine"
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for contrastive loss"""
        batch_size = 4
        
        # Positive pairs: text_i matches vision_i
        text_embeddings = torch.randn(batch_size, 512)
        vision_embeddings = torch.randn(batch_size, 512)
        
        # Normalize embeddings
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        vision_embeddings = torch.nn.functional.normalize(vision_embeddings, dim=-1)
        
        return text_embeddings, vision_embeddings
    
    def test_contrastive_loss_initialization(self, contrastive_loss):
        """Test ContrastiveLoss initialization"""
        assert contrastive_loss.temperature == 0.07
        assert contrastive_loss.similarity_metric == "cosine"
        
        # Temperature should be positive
        assert contrastive_loss.temperature > 0
    
    def test_similarity_computation(self, contrastive_loss, sample_embeddings):
        """Test similarity computation"""
        text_embeddings, vision_embeddings = sample_embeddings
        
        similarity = contrastive_loss.compute_similarity(text_embeddings, vision_embeddings)
        
        # Verify similarity matrix
        batch_size = text_embeddings.shape[0]
        assert similarity.shape == (batch_size, batch_size)
        
        # Cosine similarity should be between -1 and 1
        assert torch.all(similarity >= -1) and torch.all(similarity <= 1)
        
        # Diagonal should have high similarity (positive pairs)
        for i in range(batch_size):
            assert similarity[i, i] > similarity[i, (i + 1) % batch_size]  # Should be higher than random pair
    
    def test_contrastive_loss_calculation(self, contrastive_loss, sample_embeddings):
        """Test contrastive loss calculation"""
        text_embeddings, vision_embeddings = sample_embeddings
        
        loss = contrastive_loss(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings
        )
        
        # Verify loss components
        assert "loss" in loss
        assert "text_to_vision_loss" in loss
        assert "vision_to_text_loss" in loss
        assert "accuracy" in loss
        
        # Loss should be positive
        assert loss["loss"] > 0
        
        # Accuracy should be between 0 and 1
        assert 0 <= loss["accuracy"] <= 1
    
    def test_hard_negative_mining(self, contrastive_loss):
        """Test hard negative mining"""
        batch_size = 8
        
        # Create embeddings with hard negatives
        text_embeddings = torch.randn(batch_size, 512)
        vision_embeddings = torch.randn(batch_size, 512)
        
        # Make some negatives very similar (hard negatives)
        vision_embeddings[2] = text_embeddings[0] + torch.randn(512) * 0.1  # Hard negative for text_embeddings[0]
        
        # Normalize
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        vision_embeddings = torch.nn.functional.normalize(vision_embeddings, dim=-1)
        
        loss_with_hard_negatives = contrastive_loss(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            use_hard_negatives=True
        )
        
        # Loss with hard negatives should be higher
        loss_without = contrastive_loss(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings,
            use_hard_negatives=False
        )
        
        assert loss_with_hard_negatives["loss"] >= loss_without["loss"]
    
    def test_temperature_effect(self, contrastive_loss, sample_embeddings):
        """Test effect of temperature on loss"""
        text_embeddings, vision_embeddings = sample_embeddings
        
        # Test with different temperatures
        temperatures = [0.01, 0.07, 0.5, 1.0]
        losses = []
        
        for temp in temperatures:
            contrastive_loss.temperature = temp
            loss = contrastive_loss(text_embeddings, vision_embeddings)
            losses.append(loss["loss"])
        
        # Lower temperature should give higher loss (sharper distribution)
        assert losses[0] > losses[1]  # 0.01 > 0.07
    
    def test_multi_modal_contrastive_loss(self, contrastive_loss):
        """Test multi-modal contrastive loss with more than 2 modalities"""
        batch_size = 4
        
        # Three modalities
        text_embeddings = torch.randn(batch_size, 512)
        vision_embeddings = torch.randn(batch_size, 512)
        audio_embeddings = torch.randn(batch_size, 512)
        
        # Normalize
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        vision_embeddings = torch.nn.functional.normalize(vision_embeddings, dim=-1)
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1)
        
        loss = contrastive_loss.multi_modal_contrastive(
            embeddings={
                "text": text_embeddings,
                "vision": vision_embeddings,
                "audio": audio_embeddings
            }
        )
        
        # Verify multi-modal loss
        assert "total_loss" in loss
        assert "pairwise_losses" in loss
        
        # Should have loss for each pair of modalities
        assert len(loss["pairwise_losses"]) == 3  # text-vision, text-audio, vision-audio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
