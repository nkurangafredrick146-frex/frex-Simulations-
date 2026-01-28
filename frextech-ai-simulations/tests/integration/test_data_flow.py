"""
Integration tests for data flow through the system.
Tests data ingestion, preprocessing, model input/output, and storage.
"""

import pytest
import asyncio
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.file_io import DataLoader, DataWriter
from src.data.datasets.video_dataset import VideoDataset
from src.data.datasets.image_dataset import ImageDataset
from src.data.datasets.multimodal_dataset import MultimodalDataset
from src.core.multimodal.encoders.text_encoder import TextEncoder
from src.core.multimodal.encoders.vision_encoder import VisionEncoder
from src.core.world_model.inference.generator import WorldGenerator
from src.core.world_model.inference.sampler import Sampler
from src.api.utils.cache_manager import CacheManager
from src.api.utils.async_processor import AsyncProcessor


class TestDataFlow:
    """Test suite for data flow through the system"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_video_data(self, temp_data_dir):
        """Create sample video data for testing"""
        # Create a mock video file structure
        video_dir = temp_data_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        
        # Create metadata file
        metadata = {
            "video_id": "test_video_001",
            "duration": 10.0,
            "frame_rate": 30,
            "resolution": [1280, 720],
            "format": "mp4",
            "tags": ["landscape", "nature", "test"]
        }
        
        metadata_file = video_dir / "test_video_001.json"
        metadata_file.write_text(json.dumps(metadata))
        
        # Create dummy video file
        video_file = video_dir / "test_video_001.mp4"
        video_file.write_bytes(b"fake video data")
        
        return {
            "path": video_file,
            "metadata": metadata
        }
    
    @pytest.fixture
    def sample_image_data(self, temp_data_dir):
        """Create sample image data for testing"""
        image_dir = temp_data_dir / "images"
        image_dir.mkdir(exist_ok=True)
        
        # Create sample images
        images = []
        for i in range(5):
            img_path = image_dir / f"image_{i:03d}.jpg"
            img_path.write_bytes(b"fake image data")
            
            metadata = {
                "image_id": f"image_{i:03d}",
                "resolution": [1024, 768],
                "format": "jpg",
                "caption": f"Test image {i}",
                "tags": ["test", f"category_{i % 3}"]
            }
            
            images.append({
                "path": img_path,
                "metadata": metadata
            })
        
        return images
    
    @pytest.fixture
    def sample_3d_data(self, temp_data_dir):
        """Create sample 3D data for testing"""
        model_dir = temp_data_dir / "3d_models"
        model_dir.mkdir(exist_ok=True)
        
        # Create sample 3D model files
        model_files = []
        for ext in ['.obj', '.glb', '.ply']:
            model_path = model_dir / f"model{ext}"
            model_path.write_bytes(b"fake 3d model data")
            
            metadata = {
                "model_id": f"model_{ext[1:]}",
                "format": ext[1:],
                "vertex_count": 1000,
                "face_count": 2000,
                "textures": True,
                "bounding_box": [[-1, -1, -1], [1, 1, 1]]
            }
            
            model_files.append({
                "path": model_path,
                "metadata": metadata
            })
        
        return model_files
    
    @pytest.fixture
    def sample_text_data(self):
        """Create sample text data for testing"""
        return [
            {
                "text": "A beautiful mountain landscape with snow-capped peaks",
                "metadata": {
                    "source": "caption",
                    "language": "en",
                    "length": 10
                }
            },
            {
                "text": "A serene lake reflecting the surrounding forest",
                "metadata": {
                    "source": "caption",
                    "language": "en",
                    "length": 8
                }
            },
            {
                "text": "Urban cityscape at night with neon lights",
                "metadata": {
                    "source": "caption",
                    "language": "en",
                    "length": 7
                }
            }
        ]
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization and configuration"""
        config = {
            "data_dir": "/tmp/test_data",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 4,
            "prefetch_factor": 2
        }
        
        loader = DataLoader(config)
        assert loader.data_dir == Path(config["data_dir"])
        assert loader.batch_size == config["batch_size"]
        assert loader.shuffle == config["shuffle"]
        assert loader.num_workers == config["num_workers"]
    
    def test_video_dataset_loading(self, sample_video_data):
        """Test loading and processing video data"""
        dataset = VideoDataset(
            data_dir=sample_video_data["path"].parent,
            max_frames=100,
            frame_size=(256, 256)
        )
        
        # Test dataset properties
        assert len(dataset) == 1  # Only one video in test data
        assert dataset[0]["video_id"] == "test_video_001"
        assert dataset[0]["frame_rate"] == 30
        assert dataset[0]["duration"] == 10.0
        
        # Test frame extraction (mocked)
        with patch('cv2.VideoCapture') as mock_capture:
            mock_capture.return_value.isOpened.return_value = True
            mock_capture.return_value.get.return_value = 30  # Frame count
            mock_capture.return_value.read.return_value = (True, np.zeros((720, 1280, 3)))
            
            frames = dataset.extract_frames("test_video_001", max_frames=10)
            assert isinstance(frames, list)
            assert len(frames) <= 10
    
    def test_image_dataset_loading(self, sample_image_data):
        """Test loading and processing image data"""
        dataset = ImageDataset(
            data_dir=sample_image_data[0]["path"].parent,
            image_size=(512, 512),
            augment=True
        )
        
        # Test dataset properties
        assert len(dataset) == 5  # Five test images
        
        # Test image loading
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img.size = (1024, 768)
            mock_img.resize.return_value = Mock()
            mock_img.convert.return_value = Mock()
            mock_open.return_value = mock_img
            
            item = dataset[0]
            assert "image_id" in item
            assert "image" in item
            assert "metadata" in item
    
    def test_multimodal_dataset_creation(self, sample_image_data, sample_text_data):
        """Test multimodal dataset creation and alignment"""
        # Create aligned multimodal data
        multimodal_data = []
        for img, text in zip(sample_image_data[:3], sample_text_data):
            multimodal_data.append({
                "image_path": img["path"],
                "image_metadata": img["metadata"],
                "text": text["text"],
                "text_metadata": text["metadata"],
                "alignment_score": 0.95
            })
        
        dataset = MultimodalDataset(
            data=multimodal_data,
            image_size=(512, 512),
            text_max_length=128
        )
        
        assert len(dataset) == 3
        
        # Test batch creation
        batch = dataset.collate_fn([dataset[i] for i in range(2)])
        assert "images" in batch
        assert "texts" in batch
        assert "metadata" in batch
    
    def test_text_encoder_data_flow(self):
        """Test data flow through text encoder"""
        encoder = TextEncoder(model_name="clip-text")
        
        # Mock the actual model
        with patch.object(encoder, 'model') as mock_model:
            mock_model.encode.return_value = np.random.randn(4, 512)
            
            texts = [
                "A beautiful landscape",
                "An urban cityscape"
            ]
            
            embeddings = encoder.encode(texts, normalize=True)
            
            assert embeddings.shape == (2, 512)
            assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)
    
    def test_vision_encoder_data_flow(self):
        """Test data flow through vision encoder"""
        encoder = VisionEncoder(model_name="clip-vision")
        
        # Mock image processing
        with patch.object(encoder, 'processor') as mock_processor:
            with patch.object(encoder, 'model') as mock_model:
                # Mock processor output
                mock_processor.return_value = {
                    'pixel_values': torch.randn(2, 3, 224, 224)
                }
                
                # Mock model output
                mock_model.return_value.pooler_output = torch.randn(2, 768)
                
                # Create dummy images
                dummy_images = [np.random.rand(256, 256, 3) for _ in range(2)]
                
                embeddings = encoder.encode(dummy_images)
                
                assert embeddings.shape == (2, 768)
    
    def test_world_generation_data_flow(self):
        """Test complete data flow for world generation"""
        generator = WorldGenerator(model_path="models/world_model/checkpoint.pt")
        
        # Mock all components
        with patch.object(generator, 'text_encoder') as mock_text_encoder:
            with patch.object(generator, 'diffusion_model') as mock_diffusion:
                with patch.object(generator, 'renderer') as mock_renderer:
                    # Setup mocks
                    mock_text_encoder.encode.return_value = torch.randn(1, 512)
                    mock_diffusion.generate.return_value = {
                        'latents': torch.randn(1, 256, 256, 256),
                        'timesteps': 100,
                        'guidance_scale': 7.5
                    }
                    mock_renderer.render.return_value = {
                        'video': np.random.rand(30, 256, 256, 3),
                        'depth': np.random.rand(30, 256, 256),
                        'normals': np.random.rand(30, 256, 256, 3)
                    }
                    
                    # Generate world
                    prompt = "A beautiful mountain landscape"
                    result = generator.generate(
                        prompt=prompt,
                        num_frames=30,
                        resolution=(256, 256)
                    )
                    
                    # Verify result structure
                    assert 'video' in result
                    assert 'depth' in result
                    assert 'normals' in result
                    assert 'metadata' in result
                    
                    # Verify metadata
                    assert result['metadata']['prompt'] == prompt
                    assert result['metadata']['resolution'] == (256, 256)
                    assert result['metadata']['num_frames'] == 30
    
    def test_sampler_data_flow(self):
        """Test data flow through sampler"""
        sampler = Sampler(
            model="models/world_model/checkpoint.pt",
            sampler_type="ddim",
            num_steps=50
        )
        
        # Mock diffusion model
        with patch.object(sampler, 'model') as mock_model:
            # Setup mock responses
            noise = torch.randn(1, 4, 32, 32)
            mock_model.apply_model.return_value = torch.randn_like(noise)
            
            # Test sampling
            latents = sampler.sample(
                condition=torch.randn(1, 512),
                batch_size=1,
                shape=(4, 32, 32)
            )
            
            assert latents.shape == (1, 4, 32, 32)
    
    def test_cache_manager_data_flow(self):
        """Test data flow through cache manager"""
        cache_manager = CacheManager(
            cache_dir="/tmp/test_cache",
            max_size_gb=10
        )
        
        # Test cache storage and retrieval
        key = "test_embedding_001"
        data = {
            "embedding": np.random.randn(512),
            "metadata": {"source": "test", "timestamp": 1234567890}
        }
        
        # Store data
        cache_manager.set(key, data, ttl=3600)
        
        # Retrieve data
        retrieved = cache_manager.get(key)
        
        assert retrieved is not None
        assert np.array_equal(data["embedding"], retrieved["embedding"])
        assert data["metadata"] == retrieved["metadata"]
        
        # Test cache miss
        assert cache_manager.get("non_existent_key") is None
    
    def test_async_processor_data_flow(self):
        """Test data flow through async processor"""
        processor = AsyncProcessor(
            max_workers=2,
            queue_size=100
        )
        
        # Test function to process
        def process_item(item):
            return {"processed": True, "value": item * 2}
        
        # Submit tasks
        futures = []
        for i in range(5):
            future = processor.submit(process_item, i)
            futures.append(future)
        
        # Collect results
        results = [f.result() for f in futures]
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["processed"] == True
            assert result["value"] == i * 2
    
    def test_end_to_end_data_pipeline(self, temp_data_dir, sample_text_data):
        """Test complete end-to-end data pipeline"""
        # Create data writer
        writer = DataWriter(output_dir=temp_data_dir / "processed")
        
        # Simulate processing pipeline
        processed_data = []
        for i, text_item in enumerate(sample_text_data):
            # Simulate embedding generation
            embedding = np.random.randn(512)
            
            # Simulate metadata enrichment
            metadata = {
                **text_item["metadata"],
                "embedding_shape": embedding.shape,
                "embedding_norm": float(np.linalg.norm(embedding)),
                "processing_timestamp": time.time()
            }
            
            processed_data.append({
                "id": f"processed_{i:03d}",
                "text": text_item["text"],
                "embedding": embedding,
                "metadata": metadata
            })
        
        # Write processed data
        output_file = writer.write_batch(
            data=processed_data,
            filename="processed_embeddings",
            format="hdf5"
        )
        
        assert output_file.exists()
        assert output_file.suffix == ".h5"
        
        # Read back and verify
        loader = DataLoader(config={"data_dir": output_file.parent})
        loaded_data = loader.load(output_file.name)
        
        assert len(loaded_data) == len(processed_data)
        for original, loaded in zip(processed_data, loaded_data):
            assert original["id"] == loaded["id"]
            assert original["text"] == loaded["text"]
            assert np.allclose(original["embedding"], loaded["embedding"])
    
    def test_data_validation_pipeline(self):
        """Test data validation at each pipeline stage"""
        from src.utils.validation import DataValidator
        
        validator = DataValidator()
        
        # Test text validation
        valid_text = "A valid text description"
        invalid_text = ""  # Empty
        
        assert validator.validate_text(valid_text) == True
        assert validator.validate_text(invalid_text) == False
        
        # Test image validation
        valid_image = np.random.rand(256, 256, 3).astype(np.uint8)
        invalid_image = np.random.rand(10, 10, 3)  # Too small
        
        assert validator.validate_image(valid_image, min_size=(64, 64)) == True
        assert validator.validate_image(invalid_image, min_size=(64, 64)) == False
        
        # Test metadata validation
        valid_metadata = {
            "source": "test",
            "timestamp": 1234567890,
            "format": "jpg",
            "resolution": [256, 256]
        }
        
        invalid_metadata = {
            "source": "test"
            # Missing required fields
        }
        
        required_fields = ["source", "timestamp", "format", "resolution"]
        assert validator.validate_metadata(valid_metadata, required_fields) == True
        assert validator.validate_metadata(invalid_metadata, required_fields) == False
    
    def test_error_handling_in_data_flow(self):
        """Test error handling throughout data flow"""
        from src.utils.error_handling import DataFlowErrorHandler
        
        error_handler = DataFlowErrorHandler()
        
        # Test with function that might fail
        def risky_operation(data):
            if "fail" in data:
                raise ValueError("Intentional failure")
            return {"success": True, "data": data}
        
        # Test successful operation
        result = error_handler.execute_with_retry(
            risky_operation,
            args=["safe_data"],
            max_retries=3,
            retry_delay=0.1
        )
        
        assert result["success"] == True
        
        # Test failing operation
        with pytest.raises(ValueError):
            error_handler.execute_with_retry(
                risky_operation,
                args=["fail_data"],
                max_retries=2,
                retry_delay=0.1
            )
    
    def test_data_transformation_pipeline(self):
        """Test data transformation pipeline"""
        from src.utils.transforms import DataTransformer
        
        transformer = DataTransformer()
        
        # Create sample data
        sample_data = {
            "images": [np.random.rand(256, 256, 3) for _ in range(3)],
            "texts": ["sample text 1", "sample text 2", "sample text 3"],
            "metadata": [
                {"id": i, "source": "test"} for i in range(3)
            ]
        }
        
        # Apply transformations
        transformed = transformer.transform_batch(
            data=sample_data,
            image_transforms=["resize", "normalize"],
            text_transforms=["tokenize", "pad"],
            image_target_size=(128, 128),
            text_max_length=128
        )
        
        # Verify transformations
        assert "transformed_images" in transformed
        assert "transformed_texts" in transformed
        assert "transformed_metadata" in transformed
        
        # Check image transformation
        assert transformed["transformed_images"].shape == (3, 128, 128, 3)
        
        # Check text transformation
        assert "input_ids" in transformed["transformed_texts"]
        assert "attention_mask" in transformed["transformed_texts"]
        
        # Check metadata preservation
        assert len(transformed["transformed_metadata"]) == 3
    
    def test_data_augmentation_pipeline(self):
        """Test data augmentation pipeline"""
        from src.utils.augmentation import DataAugmentor
        
        augmentor = DataAugmentor()
        
        # Create sample image
        image = np.random.rand(256, 256, 3).astype(np.uint8)
        
        # Apply augmentations
        augmented = augmentor.augment_image(
            image=image,
            augmentations=[
                "random_crop",
                "random_flip",
                "color_jitter",
                "random_rotation"
            ],
            crop_size=(224, 224),
            flip_probability=0.5
        )
        
        assert augmented.shape == (224, 224, 3)
        assert augmented.dtype == np.uint8
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency"""
        from src.utils.batch_processor import BatchProcessor
        
        processor = BatchProcessor(
            batch_size=32,
            max_queue_size=1000,
            num_workers=4
        )
        
        # Create mock processing function
        def mock_process(batch):
            # Simulate processing time
            time.sleep(0.01)
            return [{"processed": True} for _ in batch]
        
        # Generate test data
        test_data = [f"item_{i}" for i in range(1000)]
        
        # Process in batches
        start_time = time.time()
        results = processor.process(
            data=test_data,
            process_func=mock_process,
            description="Test processing"
        )
        end_time = time.time()
        
        # Verify all items processed
        assert len(results) == len(test_data)
        
        # Check processing time is reasonable
        processing_time = end_time - start_time
        # 1000 items / 32 batch size = ~32 batches
        # 0.01s per batch * 32 batches = 0.32s ideal
        # Add overhead for multiprocessing
        assert processing_time < 2.0  # Should be less than 2 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
