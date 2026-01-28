"""
Unit tests for interactive editing components.
Tests panorama editing, region selection, edit propagation, and consistency checking.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.interactive.editor.pano_editor import PanoEditor
from src.interactive.editor.region_selector import RegionSelector
from src.interactive.editor.edit_propagator import EditPropagator
from src.interactive.editor.consistency_checker import ConsistencyChecker
from src.interactive.editor.style_transfer import StyleTransferEditor
from src.interactive.editor.object_editor import ObjectEditor


class TestPanoEditor:
    """Tests for panorama editor"""
    
    @pytest.fixture
    def pano_editor(self):
        """Create PanoEditor instance"""
        return PanoEditor(
            model_path="models/editor/checkpoint.pt",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_panorama(self):
        """Create sample panorama image"""
        return np.random.rand(512, 1024, 3).astype(np.float32)
    
    @pytest.fixture
    def sample_edit_mask(self):
        """Create sample edit mask"""
        mask = np.zeros((512, 1024), dtype=np.float32)
        mask[200:300, 400:600] = 1.0  # Rectangle in center
        return mask
    
    def test_pano_editor_initialization(self, pano_editor):
        """Test PanoEditor initialization"""
        assert pano_editor.device == "cpu"
        assert pano_editor.model_path == "models/editor/checkpoint.pt"
        assert pano_editor.edit_history is not None
    
    def test_apply_global_edit(self, pano_editor, sample_panorama):
        """Test applying global edit to panorama"""
        with patch.object(pano_editor, 'model') as mock_model:
            # Mock model prediction
            mock_output = torch.randn(1, 3, 512, 1024)
            mock_model.return_value = mock_output
            
            edit_params = {
                "edit_type": "style_transfer",
                "style_prompt": "make it look like a painting",
                "strength": 0.7
            }
            
            result = pano_editor.apply_global_edit(
                panorama=sample_panorama,
                edit_params=edit_params
            )
            
            # Verify result structure
            assert "edited_panorama" in result
            assert "edit_mask" in result
            assert "edit_metadata" in result
            
            # Verify edited panorama shape
            assert result["edited_panorama"].shape == sample_panorama.shape
            
            # Verify edit mask is all ones for global edit
            assert np.allclose(result["edit_mask"], 1.0)
    
    def test_apply_regional_edit(self, pano_editor, sample_panorama, sample_edit_mask):
        """Test applying regional edit to panorama"""
        with patch.object(pano_editor, 'model') as mock_model:
            # Mock model prediction for region
            mock_output = torch.randn(1, 3, 512, 1024)
            mock_model.return_value = mock_output
            
            edit_params = {
                "edit_type": "object_replacement",
                "object_prompt": "replace tree with rock",
                "region_mask": sample_edit_mask,
                "blend_edges": True
            }
            
            result = pano_editor.apply_regional_edit(
                panorama=sample_panorama,
                edit_params=edit_params
            )
            
            # Verify result structure
            assert "edited_panorama" in result
            assert "edit_mask" in result
            assert "blended" in result
            assert "edit_metadata" in result
            
            # Verify shapes
            assert result["edited_panorama"].shape == sample_panorama.shape
            assert result["edit_mask"].shape == sample_edit_mask.shape
            
            # Verify blend flag
            assert result["blended"] == True
    
    def test_apply_temporal_edit(self, pano_editor):
        """Test applying temporal edit to video"""
        # Create sample video frames
        sample_frames = [np.random.rand(256, 256, 3).astype(np.float32) for _ in range(10)]
        
        with patch.object(pano_editor, 'model') as mock_model:
            # Mock model prediction for each frame
            mock_output = torch.randn(1, 3, 256, 256)
            mock_model.return_value = mock_output
            
            edit_params = {
                "edit_type": "temporal_consistency",
                "edit_prompt": "add snow to the ground",
                "propagate_over_time": True,
                "keyframes": [0, 5, 9]
            }
            
            result = pano_editor.apply_temporal_edit(
                frames=sample_frames,
                edit_params=edit_params
            )
            
            # Verify result structure
            assert "edited_frames" in result
            assert "consistency_scores" in result
            assert "propagation_path" in result
            
            # Verify number of frames
            assert len(result["edited_frames"]) == len(sample_frames)
            
            # Verify consistency scores
            assert len(result["consistency_scores"]) == len(sample_frames) - 1
    
    def test_blend_edit_boundary(self, pano_editor, sample_panorama):
        """Test blending edit boundaries"""
        # Create edited region
        edited_region = np.random.rand(100, 100, 3).astype(np.float32)
        
        # Create mask with soft edges
        mask = np.zeros((512, 1024), dtype=np.float32)
        mask[200:300, 200:300] = 1.0
        
        # Apply Gaussian blur to mask edges
        from scipy.ndimage import gaussian_filter
        soft_mask = gaussian_filter(mask, sigma=5)
        
        result = pano_editor.blend_edit_boundary(
            original=sample_panorama,
            edited=edited_region,
            edit_mask=soft_mask,
            blend_radius=10
        )
        
        # Verify result
        assert result.shape == sample_panorama.shape
        assert result.dtype == np.float32
        
        # Verify blending in boundary region
        boundary_region = result[195:205, 195:205]  # Just outside edit boundary
        assert not np.allclose(boundary_region, sample_panorama[195:205, 195:205])
        assert not np.allclose(boundary_region, edited_region[195:205, 195:205])
    
    def test_undo_edit(self, pano_editor, sample_panorama):
        """Test undo functionality"""
        # Apply multiple edits
        edit_results = []
        for i in range(3):
            with patch.object(pano_editor, 'model'):
                edit_params = {
                    "edit_type": f"test_edit_{i}",
                    "strength": 0.5
                }
                
                result = pano_editor.apply_global_edit(sample_panorama, edit_params)
                edit_results.append(result)
        
        # Verify edit history
        assert len(pano_editor.edit_history) == 3
        
        # Undo last edit
        undone = pano_editor.undo()
        assert undone is not None
        assert len(pano_editor.edit_history) == 2
        
        # Undo all edits
        while pano_editor.undo():
            pass
        
        assert len(pano_editor.edit_history) == 0
    
    def test_redo_edit(self, pano_editor, sample_panorama):
        """Test redo functionality"""
        # Apply and undo edits
        with patch.object(pano_editor, 'model'):
            edit_params = {"edit_type": "test_edit", "strength": 0.5}
            pano_editor.apply_global_edit(sample_panorama, edit_params)
            pano_editor.undo()
        
        # Redo the edit
        redone = pano_editor.redo()
        assert redone is not None
        assert len(pano_editor.edit_history) == 1
    
    def test_compute_edit_distance(self, pano_editor):
        """Test computing edit distance between images"""
        img1 = np.random.rand(256, 256, 3).astype(np.float32)
        img2 = np.random.rand(256, 256, 3).astype(np.float32)
        
        distance = pano_editor.compute_edit_distance(img1, img2)
        
        # Verify distance metrics
        assert "mse" in distance
        assert "psnr" in distance
        assert "ssim" in distance
        assert "lpips" in distance if hasattr(pano_editor, 'lpips_model') else True
        
        # MSE should be positive
        assert distance["mse"] >= 0
        
        # PSNR should be finite
        assert np.isfinite(distance["psnr"])


class TestRegionSelector:
    """Tests for region selection"""
    
    @pytest.fixture
    def region_selector(self):
        """Create RegionSelector instance"""
        return RegionSelector(
            selection_method="interactive",
            refinement_steps=3
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image"""
        return np.random.rand(512, 512, 3).astype(np.float32)
    
    def test_rectangle_selection(self, region_selector, sample_image):
        """Test rectangle region selection"""
        rect_params = {
            "x": 100,
            "y": 100,
            "width": 200,
            "height": 150,
            "rotation": 0
        }
        
        mask = region_selector.select_rectangle(
            image_shape=sample_image.shape,
            params=rect_params
        )
        
        # Verify mask shape
        assert mask.shape == sample_image.shape[:2]
        
        # Verify selected region
        assert mask[100, 100] == 1.0  # Inside
        assert mask[299, 249] == 1.0  # Bottom-right corner
        assert mask[99, 100] == 0.0  # Outside (top)
        assert mask[100, 99] == 0.0  # Outside (left)
    
    def test_polygon_selection(self, region_selector, sample_image):
        """Test polygon region selection"""
        polygon_points = [
            [100, 100],
            [200, 50],
            [300, 100],
            [250, 200],
            [150, 200]
        ]
        
        mask = region_selector.select_polygon(
            image_shape=sample_image.shape,
            points=polygon_points
        )
        
        # Verify mask shape
        assert mask.shape == sample_image.shape[:2]
        
        # Verify center point is selected (convex polygon)
        assert mask[125, 200] == 1.0
        
        # Verify points outside polygon are not selected
        assert mask[0, 0] == 0.0
    
    def test_freeform_selection(self, region_selector, sample_image):
        """Test freeform (brush) selection"""
        # Create brush strokes
        strokes = [
            {"points": [(100, 100), (150, 150)], "radius": 10},
            {"points": [(200, 200), (250, 250)], "radius": 15}
        ]
        
        mask = region_selector.select_freeform(
            image_shape=sample_image.shape,
            strokes=strokes
        )
        
        # Verify mask shape
        assert mask.shape == sample_image.shape[:2]
        
        # Verify stroke areas are selected
        assert mask[120, 120] == 1.0  # Near first stroke
        assert mask[220, 220] == 1.0  # Near second stroke
    
    def test_semantic_selection(self, region_selector, sample_image):
        """Test semantic (object-based) selection"""
        with patch.object(region_selector, 'segmentation_model') as mock_model:
            # Mock segmentation model
            mock_output = torch.zeros(1, 21, *sample_image.shape[:2])
            mock_output[0, 15] = 1.0  # Class 15 = "person"
            mock_model.return_value = mock_output
            
            mask = region_selector.select_semantic(
                image=sample_image,
                class_name="person"
            )
            
            # Verify mask shape
            assert mask.shape == sample_image.shape[:2]
            
            # In mock, all pixels should be selected for class 15
            assert np.all(mask == 1.0)
    
    def test_mask_refinement(self, region_selector):
        """Test mask refinement"""
        # Create rough mask
        rough_mask = np.zeros((256, 256), dtype=np.float32)
        rough_mask[100:150, 100:150] = 1.0
        rough_mask[120:130, 120:130] = 0.5  # Some uncertainty
        
        refined_mask = region_selector.refine_mask(
            mask=rough_mask,
            refinement_steps=3
        )
        
        # Verify refinement
        assert refined_mask.shape == rough_mask.shape
        assert refined_mask.dtype == np.float32
        
        # Refined mask should be binary or near-binary
        unique_values = np.unique(refined_mask)
        assert len(unique_values) <= 256  # Some continuous values allowed
    
    def test_mask_operations(self, region_selector):
        """Test mask operations (union, intersection, difference)"""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[30:70, 30:70] = True  # Square
        
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:90, 50:90] = True  # Overlapping square
        
        # Union
        union = region_selector.mask_union(mask1, mask2)
        assert union[40, 40]  # In mask1 only
        assert union[80, 80]  # In mask2 only
        assert union[60, 60]  # In both
        
        # Intersection
        intersection = region_selector.mask_intersection(mask1, mask2)
        assert not intersection[40, 40]  # Not in intersection
        assert intersection[60, 60]  # In intersection
        
        # Difference
        difference = region_selector.mask_difference(mask1, mask2)
        assert difference[40, 40]  # In mask1 but not mask2
        assert not difference[80, 80]  # Not in mask1
        assert not difference[60, 60]  # In both


class TestEditPropagator:
    """Tests for edit propagation"""
    
    @pytest.fixture
    def edit_propagator(self):
        """Create EditPropagator instance"""
        return EditPropagator(
            propagation_method="diffusion",
            consistency_weight=0.8
        )
    
    @pytest.fixture
    def sample_video_frames(self):
        """Create sample video frames"""
        return [np.random.rand(256, 256, 3).astype(np.float32) for _ in range(10)]
    
    @pytest.fixture
    def sample_edit_on_frame(self):
        """Create sample edit on a single frame"""
        return {
            "frame_index": 0,
            "edit_mask": np.zeros((256, 256), dtype=np.float32),
            "edited_content": np.random.rand(256, 256, 3).astype(np.float32)
        }
    
    def test_temporal_propagation(self, edit_propagator, sample_video_frames, sample_edit_on_frame):
        """Test temporal propagation of edits"""
        # Set edit on frame 0
        sample_edit_on_frame["edit_mask"][100:150, 100:150] = 1.0
        
        result = edit_propagator.propagate_temporal(
            frames=sample_video_frames,
            initial_edit=sample_edit_on_frame,
            propagate_directions=["forward", "backward"]
        )
        
        # Verify result structure
        assert "propagated_frames" in result
        assert "propagation_masks" in result
        assert "consistency_scores" in result
        
        # Verify all frames have edits
        assert len(result["propagated_frames"]) == len(sample_video_frames)
        assert len(result["propagation_masks"]) == len(sample_video_frames)
        
        # Verify initial edit preserved
        assert np.allclose(
            result["propagated_frames"][0][100:150, 100:150],
            sample_edit_on_frame["edited_content"][100:150, 100:150]
        )
    
    def test_spatial_propagation(self, edit_propagator):
        """Test spatial propagation within a frame"""
        frame = np.random.rand(256, 256, 3).astype(np.float32)
        edit_mask = np.zeros((256, 256), dtype=np.float32)
        edit_mask[120:130, 120:130] = 1.0  # Small region
        
        edited_region = np.ones((10, 10, 3), dtype=np.float32)  # White square
        
        result = edit_propagator.propagate_spatial(
            frame=frame,
            edit_mask=edit_mask,
            edited_content=edited_region,
            propagation_distance=50
        )
        
        # Verify result
        assert "propagated_frame" in result
        assert "propagation_mask" in result
        
        # Propagation mask should be larger than original edit mask
        original_area = np.sum(edit_mask)
        propagated_area = np.sum(result["propagation_mask"] > 0)
        assert propagated_area >= original_area
    
    def test_view_propagation(self, edit_propagator):
        """Test propagation across different views"""
        views = {
            "front": np.random.rand(256, 256, 3).astype(np.float32),
            "left": np.random.rand(256, 256, 3).astype(np.float32),
            "right": np.random.rand(256, 256, 3).astype(np.float32)
        }
        
        camera_poses = {
            "front": np.eye(4),
            "left": np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "right": np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        }
        
        # Edit on front view
        edit_mask = np.zeros((256, 256), dtype=np.float32)
        edit_mask[100:150, 100:150] = 1.0
        
        result = edit_propagator.propagate_views(
            views=views,
            camera_poses=camera_poses,
            initial_view="front",
            edit_mask=edit_mask,
            edited_content=np.random.rand(50, 50, 3)
        )
        
        # Verify all views have propagated edits
        assert "propagated_views" in result
        assert "propagation_masks" in result
        
        for view_name in views.keys():
            assert view_name in result["propagated_views"]
            assert view_name in result["propagation_masks"]
    
    def test_consistency_weighted_propagation(self, edit_propagator, sample_video_frames):
        """Test consistency-weighted propagation"""
        # Create optical flow (mock)
        optical_flows = []
        for i in range(len(sample_video_frames) - 1):
            flow = np.random.randn(256, 256, 2).astype(np.float32) * 0.1
            optical_flows.append(flow)
        
        # Initial edit
        initial_edit = {
            "frame_index": 0,
            "edit_mask": np.zeros((256, 256), dtype=np.float32),
            "edited_content": np.random.rand(256, 256, 3)
        }
        initial_edit["edit_mask"][100:150, 100:150] = 1.0
        
        result = edit_propagator.consistency_weighted_propagation(
            frames=sample_video_frames,
            optical_flows=optical_flows,
            initial_edit=initial_edit,
            consistency_threshold=0.7
        )
        
        # Verify result
        assert "propagated_frames" in result
        assert "consistency_weights" in result
        assert "reliable_propagation" in result
        
        # Consistency weights should decrease over time
        weights = result["consistency_weights"]
        if len(weights) > 1:
            assert weights[0] >= weights[-1]  # First frame has highest confidence


class TestConsistencyChecker:
    """Tests for consistency checking"""
    
    @pytest.fixture
    def consistency_checker(self):
        """Create ConsistencyChecker instance"""
        return ConsistencyChecker(
            metrics=["psnr", "ssim", "lpips", "flow_consistency"],
            thresholds={
                "psnr": 25,
                "ssim": 0.8,
                "lpips": 0.3
            }
    )
    
    @pytest.fixture
    def sample_frame_pair(self):
        """Create sample frame pair"""
        frame1 = np.random.rand(256, 256, 3).astype(np.float32)
        frame2 = frame1.copy()
        # Add some noise to frame2
        frame2 += np.random.randn(*frame2.shape) * 0.1
        frame2 = np.clip(frame2, 0, 1)
        return frame1, frame2
    
    @pytest.fixture
    def sample_video_sequence(self):
        """Create sample video sequence"""
        return [np.random.rand(256, 256, 3).astype(np.float32) for _ in range(5)]
    
    def test_frame_consistency(self, consistency_checker, sample_frame_pair):
        """Test consistency between two frames"""
        frame1, frame2 = sample_frame_pair
        
        result = consistency_checker.check_frame_consistency(frame1, frame2)
        
        # Verify metrics
        assert "psnr" in result
        assert "ssim" in result
        assert "lpips" in result if "lpips" in consistency_checker.metrics else True
        
        # PSNR should be reasonable for slightly noisy images
        assert 20 < result["psnr"] < 50
        
        # SSIM should be close to 1
        assert 0.8 < result["ssim"] <= 1.0
        
        # Check thresholds
        assert result["is_consistent"] == (
            result["psnr"] >= consistency_checker.thresholds["psnr"] and
            result["ssim"] >= consistency_checker.thresholds["ssim"]
        )
    
    def test_temporal_consistency(self, consistency_checker, sample_video_sequence):
        """Test temporal consistency across sequence"""
        result = consistency_checker.check_temporal_consistency(sample_video_sequence)
        
        # Verify result structure
        assert "pairwise_consistency" in result
        assert "average_consistency" in result
        assert "inconsistency_frames" in result
        
        # Check pairwise consistency matrix
        n_frames = len(sample_video_sequence)
        pairwise = result["pairwise_consistency"]
        assert pairwise.shape == (n_frames, n_frames)
        
        # Diagonal should be perfect consistency
        for i in range(n_frames):
            assert pairwise[i, i] == 1.0
    
    def test_edit_consistency(self, consistency_checker):
        """Test consistency of edits"""
        original_frames = [np.random.rand(256, 256, 3) for _ in range(3)]
        edited_frames = [frame.copy() for frame in original_frames]
        
        # Apply consistent edit to all frames
        for i, frame in enumerate(edited_frames):
            frame[100:150, 100:150] = 0.5  # Same edit to all frames
        
        result = consistency_checker.check_edit_consistency(
            original_frames=original_frames,
            edited_frames=edited_frames
        )
        
        # Verify metrics
        assert "edit_consistency_score" in result
        assert "temporal_consistency_score" in result
        assert "spatial_consistency_score" in result
        
        # Consistent edits should have high score
        assert result["edit_consistency_score"] > 0.8
    
    def test_inconsistent_edit_detection(self, consistency_checker):
        """Test detection of inconsistent edits"""
        original_frames = [np.random.rand(256, 256, 3) for _ in range(3)]
        edited_frames = [frame.copy() for frame in original_frames]
        
        # Apply inconsistent edits
        edited_frames[0][100:150, 100:150] = 0.5  # Edit in frame 0
        edited_frames[1][100:150, 100:150] = 0.7  # Different edit in frame 1
        edited_frames[2][100:150, 100:150] = 0.5  # Same as frame 0
        
        result = consistency_checker.check_edit_consistency(
            original_frames=original_frames,
            edited_frames=edited_frames
        )
        
        # Should detect inconsistency between frame 0 and 1
        pairwise = result["pairwise_edit_consistency"]
        assert pairwise[0, 1] < pairwise[0, 2]  # 0-1 less consistent than 0-2
    
    def test_geometric_consistency(self, consistency_checker):
        """Test geometric consistency with camera poses"""
        # Create sample frames and poses
        frames = [np.random.rand(256, 256, 3) for _ in range(2)]
        
        # Camera poses (slightly different)
        poses = [
            np.eye(4),  # Identity
            np.array([  # Translated
                [1, 0, 0, 0.1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        ]
        
        # Mock depth maps
        depth_maps = [
            np.random.rand(256, 256) * 10,
            np.random.rand(256, 256) * 10
        ]
        
        result = consistency_checker.check_geometric_consistency(
            frames=frames,
            camera_poses=poses,
            depth_maps=depth_maps
        )
        
        # Verify geometric consistency metrics
        assert "reprojection_error" in result
        assert "depth_consistency" in result
        assert "normal_consistency" in result
    
    def test_semantic_consistency(self, consistency_checker):
        """Test semantic consistency"""
        frames = [np.random.rand(256, 256, 3) for _ in range(2)]
        
        with patch.object(consistency_checker, 'segmentation_model') as mock_model:
            # Mock semantic segmentation
            mock_output = torch.zeros(2, 21, 256, 256)
            mock_output[:, 15] = 1.0  # Class 15 = "person"
            mock_model.return_value = mock_output
            
            result = consistency_checker.check_semantic_consistency(frames)
            
            # Verify semantic consistency
            assert "semantic_similarity" in result
            assert "class_distribution" in result
            assert "object_persistence" in result
            
            # Should be perfectly consistent with same mock segmentation
            assert result["semantic_similarity"] == 1.0


class TestStyleTransferEditor:
    """Tests for style transfer editing"""
    
    @pytest.fixture
    def style_editor(self):
        """Create StyleTransferEditor instance"""
        return StyleTransferEditor(
            style_model="models/style_transfer/checkpoint.pt",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_content_image(self):
        """Create sample content image"""
        return np.random.rand(256, 256, 3).astype(np.float32)
    
    @pytest.fixture
    def sample_style_image(self):
        """Create sample style image"""
        return np.random.rand(256, 256, 3).astype(np.float32)
    
    def test_apply_style_transfer(self, style_editor, sample_content_image, sample_style_image):
        """Test applying style transfer"""
        with patch.object(style_editor, 'model') as mock_model:
            # Mock style transfer output
            mock_output = torch.randn(1, 3, 256, 256)
            mock_model.return_value = mock_output
            
            result = style_editor.apply_style_transfer(
                content_image=sample_content_image,
                style_image=sample_style_image,
                style_weight=0.7,
                content_weight=0.3
            )
            
            # Verify result
            assert "styled_image" in result
            assert "style_loss" in result
            assert "content_loss" in result
            assert "total_loss" in result
            
            # Verify image shape
            assert result["styled_image"].shape == sample_content_image.shape
    
    def test_apply_text_guided_style(self, style_editor, sample_content_image):
        """Test text-guided style transfer"""
        with patch.object(style_editor, 'text_to_style_model') as mock_model:
            # Mock text-guided style transfer
            mock_output = torch.randn(1, 3, 256, 256)
            mock_model.return_value = mock_output
            
            result = style_editor.apply_text_guided_style(
                content_image=sample_content_image,
                style_text="oil painting, impressionist style",
                style_strength=0.8
            )
            
            # Verify result
            assert "styled_image" in result
            assert "style_adherence" in result
            assert "content_preservation" in result
    
    def test_extract_style_features(self, style_editor, sample_style_image):
        """Test style feature extraction"""
        with patch.object(style_editor, 'extract_features') as mock_extract:
            # Mock feature extraction
            mock_features = {
                'gram_matrices': [torch.randn(64, 64) for _ in range(5)],
                'style_embeddings': torch.randn(512)
            }
            mock_extract.return_value = mock_features
            
            features = style_editor.extract_style_features(sample_style_image)
            
            # Verify feature structure
            assert "gram_matrices" in features
            assert "style_embeddings" in features
            assert "style_statistics" in features
            
            # Gram matrices should be square
            for gram in features["gram_matrices"]:
                assert gram.shape[0] == gram.shape[1]
    
    def test_style_interpolation(self, style_editor, sample_content_image):
        """Test interpolation between styles"""
        style_images = [
            np.random.rand(256, 256, 3).astype(np.float32) for _ in range(3)
        ]
        
        with patch.object(style_editor, 'model'):
            result = style_editor.interpolate_styles(
                content_image=sample_content_image,
                style_images=style_images,
                interpolation_weights=[0.5, 0.3, 0.2]
            )
            
            # Verify interpolation result
            assert "interpolated_image" in result
            assert "interpolation_weights" in result
            assert "style_contributions" in result


class TestObjectEditor:
    """Tests for object editing"""
    
    @pytest.fixture
    def object_editor(self):
        """Create ObjectEditor instance"""
        return ObjectEditor(
            segmentation_model="models/segmentation/checkpoint.pt",
            inpainting_model="models/inpainting/checkpoint.pt",
            device="cpu"
        )
    
    @pytest.fixture
    def sample_image_with_objects(self):
        """Create sample image with objects"""
        return np.random.rand(512, 512, 3).astype(np.float32)
    
    def test_object_removal(self, object_editor, sample_image_with_objects):
        """Test object removal"""
        with patch.object(object_editor, 'segmentation_model') as mock_seg:
            with patch.object(object_editor, 'inpainting_model') as mock_inpaint:
                # Mock segmentation
                mock_seg_mask = torch.zeros(1, 21, 512, 512)
                mock_seg_mask[0, 15] = 1.0  # Class 15 = "person"
                mock_seg.return_value = mock_seg_mask
                
                # Mock inpainting
                mock_inpaint_output = torch.randn(1, 3, 512, 512)
                mock_inpaint.return_value = mock_inpaint_output
                
                result = object_editor.remove_object(
                    image=sample_image_with_objects,
                    object_class="person"
                )
                
                # Verify result
                assert "edited_image" in result
                assert "removal_mask" in result
                assert "inpainted_region" in result
                
                # Verify mask shape
                assert result["removal_mask"].shape == sample_image_with_objects.shape[:2]
    
    def test_object_replacement(self, object_editor, sample_image_with_objects):
        """Test object replacement"""
        replacement_object = np.random.rand(100, 100, 3).astype(np.float32)
        
        with patch.object(object_editor, 'segmentation_model'):
            with patch.object(object_editor, 'blend_object') as mock_blend:
                # Mock blending
                mock_blend.return_value = np.random.rand(512, 512, 3)
                
                result = object_editor.replace_object(
                    image=sample_image_with_objects,
                    object_mask=np.zeros((512, 512), dtype=bool),
                    replacement_object=replacement_object,
                    blend_edges=True
                )
                
                # Verify result
                assert "edited_image" in result
                assert "replacement_mask" in result
                assert "blend_success" in result
    
    def test_object_insertion(self, object_editor, sample_image_with_objects):
        """Test object insertion"""
        new_object = np.random.rand(150, 150, 3).astype(np.float32)
        
        result = object_editor.insert_object(
            image=sample_image_with_objects,
            new_object=new_object,
            position=(200, 200),
            scale=1.0,
            rotation=0
        )
        
        # Verify result
        assert "edited_image" in result
        assert "insertion_mask" in result
        assert "shadow_added" in result
        assert "reflections_added" in result
        
        # Object should be inserted at specified position
        assert result["insertion_mask"][200, 200] == 1.0
    
    def test_object_recoloring(self, object_editor, sample_image_with_objects):
        """Test object recoloring"""
        object_mask = np.zeros((512, 512), dtype=bool)
        object_mask[200:250, 200:250] = True
        
        result = object_editor.recolor_object(
            image=sample_image_with_objects,
            object_mask=object_mask,
            new_color=(0.8, 0.2, 0.2),  # Reddish
            preserve_texture=True
        )
        
        # Verify result
        assert "edited_image" in result
        assert "recolored_mask" in result
        assert "color_transfer" in result
        
        # Recolored area should be different from original
        original_region = sample_image_with_objects[200:250, 200:250]
        edited_region = result["edited_image"][200:250, 200:250]
        assert not np.allclose(original_region, edited_region)
    
    def test_object_tracking(self, object_editor):
        """Test object tracking across frames"""
        frames = [np.random.rand(256, 256, 3) for _ in range(5)]
        initial_mask = np.zeros((256, 256), dtype=bool)
        initial_mask[100:120, 100:120] = True
        
        with patch.object(object_editor, 'track_object') as mock_track:
            # Mock tracking results
            tracked_masks = [
                initial_mask.copy(),
                np.zeros((256, 256), dtype=bool),
                np.zeros((256, 256), dtype=bool),
                np.zeros((256, 256), dtype=bool),
                np.zeros((256, 256), dtype=bool)
            ]
            # Simulate movement
            for i in range(1, 5):
                tracked_masks[i][100+i*10:120+i*10, 100+i*10:120+i*10] = True
            
            mock_track.return_value = tracked_masks
            
            result = object_editor.track_object_across_frames(
                frames=frames,
                initial_mask=initial_mask
            )
            
            # Verify tracking
            assert "tracked_masks" in result
            assert "trajectory" in result
            assert "tracking_confidence" in result
            
            # Should track object across all frames
            assert len(result["tracked_masks"]) == len(frames)
            
            # Trajectory should show movement
            trajectory = result["trajectory"]
            if len(trajectory) > 1:
                assert trajectory[-1][0] > trajectory[0][0]  # X coordinate increased


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
