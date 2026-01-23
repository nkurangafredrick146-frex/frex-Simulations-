"""
Dataset Statistics Module
Utilities for computing and analyzing dataset statistics.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .base_dataset import BaseDataset
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

@dataclass
class DatasetStatistics:
    """
    Compute and store statistics for a dataset.
    """
    dataset: BaseDataset
    stats: Dict[str, Any] = field(default_factory=dict)
    computed: bool = False
    
    def compute(self, force: bool = False) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Args:
            force: Whether to recompute statistics
            
        Returns:
            Statistics dictionary
        """
        if self.computed and not force:
            return self.stats
        
        logger.info(f"Computing statistics for dataset: {self.dataset.name}")
        
        # Reset stats
        self.stats = {
            'basic': {},
            'modality': {},
            'class_distribution': {},
            'quality_metrics': {},
            'temporal_stats': {},
            'spatial_stats': {},
            'metadata_stats': {}
        }
        
        # Compute basic statistics
        self._compute_basic_stats()
        
        # Compute modality-specific statistics
        self._compute_modality_stats()
        
        # Compute class distribution
        self._compute_class_distribution()
        
        # Compute quality metrics
        self._compute_quality_metrics()
        
        # Compute temporal statistics (if applicable)
        self._compute_temporal_stats()
        
        # Compute spatial statistics (if applicable)
        self._compute_spatial_stats()
        
        # Compute metadata statistics
        self._compute_metadata_stats()
        
        self.computed = True
        logger.info(f"Completed statistics computation for {self.dataset.name}")
        
        return self.stats
    
    def _compute_basic_stats(self) -> None:
        """Compute basic dataset statistics."""
        basic = self.stats['basic']
        
        basic['num_samples'] = len(self.dataset)
        basic['num_classes'] = self.dataset.num_classes
        basic['class_names'] = self.dataset.class_names
        
        # Sample size distribution
        sample_sizes = []
        for i in range(min(1000, len(self.dataset))):
            try:
                sample = self.dataset.get_sample(i, apply_transform=False)
                # Estimate sample size (bytes)
                size_bytes = self._estimate_sample_size(sample)
                sample_sizes.append(size_bytes)
            except Exception as e:
                logger.debug(f"Failed to get sample {i}: {e}")
        
        if sample_sizes:
            basic['sample_size_bytes'] = {
                'mean': np.mean(sample_sizes),
                'std': np.std(sample_sizes),
                'min': np.min(sample_sizes),
                'max': np.max(sample_sizes),
                'percentiles': {
                    '25': np.percentile(sample_sizes, 25),
                    '50': np.percentile(sample_sizes, 50),
                    '75': np.percentile(sample_sizes, 75),
                    '95': np.percentile(sample_sizes, 95)
                }
            }
    
    def _compute_modality_stats(self) -> None:
        """Compute statistics for each modality."""
        modality_stats = self.stats['modality']
        
        # Get a few samples to determine modalities
        if not self.dataset.samples:
            return
        
        first_sample = self.dataset.samples[0]
        
        # Check for different data types in samples
        for key, value in first_sample.items():
            if isinstance(value, str) and any(ext in value.lower() for ext in ['.jpg', '.png', '.jpeg', '.mp4', '.txt']):
                # Likely a file path
                modality_type = self._infer_modality_from_path(value)
                if modality_type:
                    modality_stats[modality_type] = self._compute_modality_specific_stats(modality_type)
        
        # If no modalities detected, try to infer from dataset type
        if not modality_stats and hasattr(self.dataset, 'type'):
            modality_type = str(self.dataset.type.value)
            modality_stats[modality_type] = self._compute_modality_specific_stats(modality_type)
    
    def _infer_modality_from_path(self, path: str) -> Optional[str]:
        """Infer modality type from file path."""
        path_lower = path.lower()
        
        if any(ext in path_lower for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']):
            return 'image'
        elif any(ext in path_lower for ext in ['.mp4', '.avi', '.mov', '.webm']):
            return 'video'
        elif any(ext in path_lower for ext in ['.txt', '.json', '.xml']):
            return 'text'
        elif any(ext in path_lower for ext in ['.ply', '.obj', '.stl']):
            return '3d'
        elif any(ext in path_lower for ext in ['.wav', '.mp3', '.flac']):
            return 'audio'
        
        return None
    
    def _compute_modality_specific_stats(self, modality: str) -> Dict[str, Any]:
        """Compute statistics for a specific modality."""
        stats = {
            'count': 0,
            'samples': [],
            'quality_metrics': {}
        }
        
        # Sample subset for analysis
        sample_indices = np.linspace(0, len(self.dataset) - 1, min(100, len(self.dataset)), dtype=int)
        
        for idx in tqdm(sample_indices, desc=f"Analyzing {modality}"):
            try:
                sample = self.dataset.get_sample(idx, apply_transform=False)
                
                if modality == 'image':
                    img_stats = self._analyze_image(sample)
                    stats['samples'].append(img_stats)
                    stats['count'] += 1
                
                elif modality == 'video':
                    video_stats = self._analyze_video(sample)
                    stats['samples'].append(video_stats)
                    stats['count'] += 1
                
                elif modality == 'text':
                    text_stats = self._analyze_text(sample)
                    stats['samples'].append(text_stats)
                    stats['count'] += 1
                
                elif modality == '3d':
                    mesh_stats = self._analyze_3d(sample)
                    stats['samples'].append(mesh_stats)
                    stats['count'] += 1
            
            except Exception as e:
                logger.debug(f"Failed to analyze sample {idx} for modality {modality}: {e}")
        
        # Aggregate statistics
        if stats['samples']:
            stats = self._aggregate_modality_stats(stats, modality)
        
        return stats
    
    def _analyze_image(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image sample."""
        stats = {}
        
        # Find image data
        image = None
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)) and (len(value.shape) in [2, 3]):
                image = value
                break
            elif isinstance(value, Image.Image):
                image = np.array(value)
                break
        
        if image is not None:
            # Convert to numpy array
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            
            # Basic image stats
            if len(image.shape) == 3:
                stats['shape'] = image.shape  # H, W, C or C, H, W
                stats['channels'] = image.shape[2] if image.shape[2] <= 4 else image.shape[0]
            else:
                stats['shape'] = image.shape
                stats['channels'] = 1
            
            # Convert to float for analysis
            img_float = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image.astype(np.float32)
            
            # Compute statistics
            if len(img_float.shape) == 3:
                # For multi-channel images
                for c in range(min(3, img_float.shape[2] if img_float.shape[2] <= 4 else img_float.shape[0])):
                    if img_float.shape[2] <= 4:
                        channel_data = img_float[:, :, c]
                    else:
                        channel_data = img_float[c, :, :]
                    
                    stats[f'channel_{c}_mean'] = float(channel_data.mean())
                    stats[f'channel_{c}_std'] = float(channel_data.std())
                    stats[f'channel_{c}_min'] = float(channel_data.min())
                    stats[f'channel_{c}_max'] = float(channel_data.max())
            else:
                # For single channel
                stats['mean'] = float(img_float.mean())
                stats['std'] = float(img_float.std())
                stats['min'] = float(img_float.min())
                stats['max'] = float(img_float.max())
            
            # Image quality metrics
            if len(img_float.shape) == 3 and (img_float.shape[2] == 3 or img_float.shape[0] == 3):
                # Compute contrast
                if img_float.shape[2] == 3:
                    gray = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
                else:
                    gray = 0.299 * img_float[0, :, :] + 0.587 * img_float[1, :, :] + 0.114 * img_float[2, :, :]
                
                stats['contrast'] = float(gray.std())
                
                # Compute brightness
                stats['brightness'] = float(gray.mean())
        
        return stats
    
    def _analyze_video(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video sample."""
        stats = {}
        
        # Find video data
        video = None
        for key, value in sample.items():
            if isinstance(value, torch.Tensor) and len(value.shape) == 4:  # T, C, H, W
                video = value
                break
            elif isinstance(value, np.ndarray) and len(value.shape) == 4:
                video = value
                break
        
        if video is not None:
            # Basic stats
            stats['shape'] = video.shape
            stats['num_frames'] = video.shape[0]
            stats['channels'] = video.shape[1]
            stats['resolution'] = (video.shape[2], video.shape[3])
            
            # Convert to float
            if isinstance(video, torch.Tensor):
                video_np = video.numpy()
            else:
                video_np = video
            
            if video_np.dtype == np.uint8:
                video_float = video_np.astype(np.float32) / 255.0
            else:
                video_float = video_np.astype(np.float32)
            
            # Temporal statistics
            frame_means = []
            frame_stds = []
            
            for t in range(min(10, video_float.shape[0])):  # Sample first 10 frames
                frame = video_float[t]
                frame_means.append(frame.mean())
                frame_stds.append(frame.std())
            
            stats['temporal_mean'] = float(np.mean(frame_means))
            stats['temporal_std'] = float(np.std(frame_means))
            stats['temporal_variation'] = float(np.mean(frame_stds))
        
        return stats
    
    def _analyze_text(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text sample."""
        stats = {}
        
        # Find text data
        text = None
        for key, value in sample.items():
            if isinstance(value, str):
                text = value
                break
        
        if text:
            stats['length'] = len(text)
            stats['word_count'] = len(text.split())
            stats['char_count'] = len(text)
            stats['line_count'] = len(text.split('\n'))
            
            # Basic text statistics
            words = text.split()
            if words:
                stats['avg_word_length'] = np.mean([len(word) for word in words])
                stats['unique_words'] = len(set(words))
                stats['vocabulary_ratio'] = stats['unique_words'] / len(words)
        
        return stats
    
    def _analyze_3d(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze 3D sample."""
        stats = {}
        
        # Look for 3D data (point clouds, meshes, etc.)
        point_cloud = None
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)) and value.shape[-1] == 3:
                point_cloud = value
                break
        
        if point_cloud is not None:
            stats['num_points'] = point_cloud.shape[0]
            stats['bounds'] = {
                'min': point_cloud.min(axis=0).tolist(),
                'max': point_cloud.max(axis=0).tolist(),
                'center': point_cloud.mean(axis=0).tolist()
            }
            
            # Compute density
            bounds = point_cloud.max(axis=0) - point_cloud.min(axis=0)
            volume = np.prod(bounds)
            if volume > 0:
                stats['density'] = point_cloud.shape[0] / volume
            else:
                stats['density'] = 0.0
        
        return stats
    
    def _aggregate_modality_stats(self, stats: Dict[str, Any], modality: str) -> Dict[str, Any]:
        """Aggregate statistics across samples for a modality."""
        if not stats['samples']:
            return stats
        
        samples = stats['samples']
        
        # Collect all keys
        all_keys = set()
        for sample in samples:
            all_keys.update(sample.keys())
        
        # Aggregate numeric statistics
        numeric_stats = {}
        for key in all_keys:
            values = []
            for sample in samples:
                if key in sample and isinstance(sample[key], (int, float)):
                    values.append(sample[key])
            
            if values:
                numeric_stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'count': len(values)
                }
        
        stats['aggregated'] = numeric_stats
        
        # Modality-specific aggregations
        if modality == 'image':
            # Common image resolutions
            shapes = [s.get('shape') for s in samples if 'shape' in s]
            if shapes:
                unique_shapes = Counter([str(s) for s in shapes])
                stats['common_resolutions'] = dict(unique_shapes.most_common(10))
        
        elif modality == 'text':
            # Text length distribution
            lengths = [s.get('length', 0) for s in samples if 'length' in s]
            if lengths:
                stats['length_distribution'] = {
                    'mean': float(np.mean(lengths)),
                    'std': float(np.std(lengths)),
                    'histogram': np.histogram(lengths, bins=20)[0].tolist()
                }
        
        elif modality == 'video':
            # Video length distribution
            lengths = [s.get('num_frames', 0) for s in samples if 'num_frames' in s]
            if lengths:
                stats['frame_count_distribution'] = {
                    'mean': float(np.mean(lengths)),
                    'std': float(np.std(lengths))
                }
        
        return stats
    
    def _compute_class_distribution(self) -> None:
        """Compute class distribution statistics."""
        class_dist = self.stats['class_distribution']
        
        # Count samples per class
        class_counts = Counter()
        label_field = None
        
        # Try to find label field
        for i in range(min(1000, len(self.dataset))):
            try:
                sample = self.dataset.get_sample(i, apply_transform=False)
                for key, value in sample.items():
                    if 'label' in key.lower() and isinstance(value, (str, int, float)):
                        label_field = key
                        break
                if label_field:
                    break
            except Exception:
                continue
        
        if label_field:
            # Count all samples
            for i in tqdm(range(len(self.dataset)), desc="Counting classes"):
                try:
                    sample = self.dataset.get_sample(i, apply_transform=False)
                    if label_field in sample:
                        label = sample[label_field]
                        class_counts[label] += 1
                except Exception:
                    continue
            
            if class_counts:
                class_dist['counts'] = dict(class_counts)
                class_dist['total'] = sum(class_counts.values())
                class_dist['num_classes'] = len(class_counts)
                
                # Compute balance metrics
                counts = list(class_counts.values())
                class_dist['balance_ratio'] = min(counts) / max(counts) if max(counts) > 0 else 0
                class_dist['entropy'] = stats.entropy(counts)
                
                # Most common classes
                class_dist['most_common'] = dict(class_counts.most_common(10))
        
        else:
            class_dist['note'] = 'No label field found'
    
    def _compute_quality_metrics(self) -> None:
        """Compute dataset quality metrics."""
        quality = self.stats['quality_metrics']
        
        # Sample subset for quality analysis
        sample_indices = np.linspace(0, len(self.dataset) - 1, min(500, len(self.dataset)), dtype=int)
        
        # Metrics
        quality['missing_data'] = 0
        quality['corrupt_samples'] = 0
        quality['duplicates'] = 0
        
        # Check for missing/corrupt data
        for idx in tqdm(sample_indices, desc="Checking quality"):
            try:
                sample = self.dataset.get_sample(idx, apply_transform=False)
                
                # Check for missing values
                if self._has_missing_values(sample):
                    quality['missing_data'] += 1
                
                # Check for obviously corrupt data
                if self._is_corrupt_sample(sample):
                    quality['corrupt_samples'] += 1
            
            except Exception as e:
                quality['corrupt_samples'] += 1
                logger.debug(f"Sample {idx} failed: {e}")
        
        # Normalize counts
        total = len(sample_indices)
        if total > 0:
            quality['missing_data_pct'] = quality['missing_data'] / total * 100
            quality['corrupt_samples_pct'] = quality['corrupt_samples'] / total * 100
        
        # Dataset diversity (estimated)
        quality['estimated_diversity'] = self._estimate_diversity()
    
    def _has_missing_values(self, sample: Dict[str, Any]) -> bool:
        """Check if sample has missing values."""
        for key, value in sample.items():
            if value is None:
                return True
            if isinstance(value, (torch.Tensor, np.ndarray)):
                if torch.isnan(value).any() if isinstance(value, torch.Tensor) else np.isnan(value).any():
                    return True
        return False
    
    def _is_corrupt_sample(self, sample: Dict[str, Any]) -> bool:
        """Check if sample is corrupt."""
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                # Check for all zeros or all same value
                if isinstance(value, torch.Tensor):
                    if torch.all(value == 0) or torch.all(value == value[0]):
                        return True
                else:
                    if np.all(value == 0) or np.all(value == value.flat[0]):
                        return True
                
                # Check for extreme values
                if isinstance(value, torch.Tensor):
                    if torch.any(torch.isinf(value)):
                        return True
                else:
                    if np.any(np.isinf(value)):
                        return True
        return False
    
    def _estimate_diversity(self) -> float:
        """Estimate dataset diversity."""
        # Simple diversity estimation based on sample variation
        sample_subset = []
        
        for i in range(min(100, len(self.dataset))):
            try:
                sample = self.dataset.get_sample(i, apply_transform=False)
                # Extract a simple feature for comparison
                feature = self._extract_diversity_feature(sample)
                if feature is not None:
                    sample_subset.append(feature)
            except Exception:
                continue
        
        if len(sample_subset) < 2:
            return 0.0
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist, squareform
        
        try:
            distances = pdist(np.array(sample_subset))
            diversity = distances.mean()
            return float(diversity)
        except Exception:
            return 0.0
    
    def _extract_diversity_feature(self, sample: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract a simple feature vector for diversity estimation."""
        # Look for image data
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                if len(value.shape) in [2, 3]:
                    # Use flattened version (limited size)
                    if isinstance(value, torch.Tensor):
                        value_np = value.numpy()
                    else:
                        value_np = value
                    
                    # Flatten and sample
                    flat = value_np.flatten()
                    if len(flat) > 1000:
                        flat = flat[:1000]
                    return flat
        
        # Look for text data
        for key, value in sample.items():
            if isinstance(value, str):
                # Use character counts as feature
                chars = list(value[:100]) if len(value) > 100 else list(value)
                # One-hot encoding of first 10 chars
                feature = np.zeros(256 * 10)  # ASCII * 10 positions
                for i, char in enumerate(chars[:10]):
                    feature[i * 256 + ord(char) % 256] = 1
                return feature
        
        return None
    
    def _compute_temporal_stats(self) -> None:
        """Compute temporal statistics for sequential data."""
        temporal = self.stats['temporal_stats']
        
        # Check if dataset has temporal dimension
        has_temporal = False
        for i in range(min(10, len(self.dataset))):
            try:
                sample = self.dataset.get_sample(i, apply_transform=False)
                for key, value in sample.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)) and len(value.shape) >= 3:
                        has_temporal = True
                        break
                if has_temporal:
                    break
            except Exception:
                continue
        
        if has_temporal:
            # Sample temporal statistics
            temporal['has_temporal_dimension'] = True
            
            # Collect frame/step counts
            lengths = []
            for i in range(min(100, len(self.dataset))):
                try:
                    sample = self.dataset.get_sample(i, apply_transform=False)
                    for key, value in sample.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)) and len(value.shape) >= 3:
                            lengths.append(value.shape[0])  # Assume first dimension is time
                            break
                except Exception:
                    continue
            
            if lengths:
                temporal['sequence_lengths'] = {
                    'mean': float(np.mean(lengths)),
                    'std': float(np.std(lengths)),
                    'min': int(np.min(lengths)),
                    'max': int(np.max(lengths))
                }
        else:
            temporal['has_temporal_dimension'] = False
    
    def _compute_spatial_stats(self) -> None:
        """Compute spatial statistics for image/video/3D data."""
        spatial = self.stats['spatial_stats']
        
        # Collect spatial dimensions
        resolutions = []
        for i in range(min(100, len(self.dataset))):
            try:
                sample = self.dataset.get_sample(i, apply_transform=False)
                for key, value in sample.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)):
                        if len(value.shape) >= 2:
                            # Get spatial dimensions
                            if len(value.shape) == 2:  # H, W
                                h, w = value.shape
                            elif len(value.shape) == 3:  # C, H, W or H, W, C
                                if value.shape[0] <= 4:  # Assume C, H, W
                                    h, w = value.shape[1], value.shape[2]
                                else:  # Assume H, W, C
                                    h, w = value.shape[0], value.shape[1]
                            elif len(value.shape) == 4:  # T, C, H, W or T, H, W, C
                                if value.shape[1] <= 4:  # Assume T, C, H, W
                                    h, w = value.shape[2], value.shape[3]
                                else:  # Assume T, H, W, C
                                    h, w = value.shape[1], value.shape[2]
                            else:
                                continue
                            
                            resolutions.append((h, w))
                            break
            except Exception:
                continue
        
        if resolutions:
            spatial['resolutions'] = {
                'unique': list(set(resolutions)),
                'counts': Counter(resolutions),
                'common': dict(Counter(resolutions).most_common(5))
            }
            
            # Aspect ratios
            aspect_ratios = [w/h for h, w in resolutions if h > 0]
            if aspect_ratios:
                spatial['aspect_ratios'] = {
                    'mean': float(np.mean(aspect_ratios)),
                    'std': float(np.std(aspect_ratios)),
                    'common': dict(Counter([round(ar, 2) for ar in aspect_ratios]).most_common(5))
                }
    
    def _compute_metadata_stats(self) -> None:
        """Compute statistics on dataset metadata."""
        metadata_stats = self.stats['metadata_stats']
        
        # Collect metadata fields
        metadata_fields = set()
        for i in range(min(100, len(self.dataset))):
            try:
                sample = self.dataset.get_sample(i, apply_transform=False)
                metadata_fields.update(sample.keys())
            except Exception:
                continue
        
        metadata_stats['fields'] = list(metadata_fields)
        metadata_stats['num_fields'] = len(metadata_fields)
        
        # Field frequency
        field_counts = Counter()
        for i in range(min(100, len(self.dataset))):
            try:
                sample = self.dataset.get_sample(i, apply_transform=False)
                for field in sample.keys():
                    field_counts[field] += 1
            except Exception:
                continue
        
        metadata_stats['field_frequency'] = dict(field_counts)
    
    def _estimate_sample_size(self, sample: Dict[str, Any]) -> int:
        """Estimate sample size in bytes."""
        total_bytes = 0
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                total_bytes += value.nelement() * value.element_size()
            elif isinstance(value, np.ndarray):
                total_bytes += value.nbytes
            elif isinstance(value, str):
                total_bytes += len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                total_bytes += 8  # Approximate
            elif isinstance(value, dict):
                total_bytes += self._estimate_sample_size(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, (torch.Tensor, np.ndarray, str, int, float)):
                        # Approximate
                        total_bytes += 100
            else:
                # Unknown type, approximate
                total_bytes += 100
        
        return total_bytes
    
    def save_report(self, output_path: str, format: str = 'json') -> None:
        """
        Save statistics report to file.
        
        Args:
            output_path: Path to save report
            format: Report format ('json', 'yaml', 'html', 'txt')
        """
        if not self.computed:
            self.compute()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
        
        elif format == 'yaml':
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(self.stats, f, default_flow_style=False)
        
        elif format == 'html':
            self._save_html_report(output_path)
        
        elif format == 'txt':
            self._save_text_report(output_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved statistics report to {output_path}")
    
    def _save_html_report(self, output_path: Path) -> None:
        """Save report as HTML with visualizations."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dataset Statistics: {self.dataset.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; margin-top: 30px; }}
                .card {{ background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                .stat {{ margin: 10px 0; }}
                .value {{ font-weight: bold; color: #0066cc; }}
            </style>
        </head>
        <body>
            <h1>Dataset Statistics Report</h1>
            <div class="card">
                <h2>Dataset: {self.dataset.name}</h2>
                <div class="stat">Total Samples: <span class="value">{self.stats['basic'].get('num_samples', 'N/A')}</span></div>
                <div class="stat">Number of Classes: <span class="value">{self.stats['basic'].get('num_classes', 'N/A')}</span></div>
            </div>
        """
        
        # Add more sections based on computed statistics
        for section_name, section_data in self.stats.items():
            if section_data:
                html_content += f"""
                <div class="card">
                    <h2>{section_name.replace('_', ' ').title()}</h2>
                """
                
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        html_content += f"<div class='stat'><strong>{key}:</strong><br>"
                        for subkey, subvalue in value.items():
                            html_content += f"&nbsp;&nbsp;{subkey}: <span class='value'>{subvalue}</span><br>"
                        html_content += "</div>"
                    else:
                        html_content += f"<div class='stat'>{key}: <span class='value'>{value}</span></div>"
                
                html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _save_text_report(self, output_path: Path) -> None:
        """Save report as plain text."""
        lines = []
        lines.append(f"Dataset Statistics Report: {self.dataset.name}")
        lines.append("=" * 60)
        
        for section_name, section_data in self.stats.items():
            if section_data:
                lines.append(f"\n{section_name.upper()}:")
                lines.append("-" * 40)
                
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        lines.append(f"  {key}:")
                        for subkey, subvalue in value.items():
                            lines.append(f"    {subkey}: {subvalue}")
                    else:
                        lines.append(f"  {key}: {value}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def visualize(self, output_dir: Optional[str] = None) -> None:
        """
        Create visualizations of dataset statistics.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not self.computed:
            self.compute()
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations for different statistics
        try:
            self._visualize_class_distribution(output_dir)
            self._visualize_sample_sizes(output_dir)
            self._visualize_modality_stats(output_dir)
            self._visualize_quality_metrics(output_dir)
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    def _visualize_class_distribution(self, output_dir: Optional[Path]) -> None:
        """Visualize class distribution."""
        class_dist = self.stats.get('class_distribution', {})
        counts = class_dist.get('counts', {})
        
        if counts and len(counts) > 1:
            plt.figure(figsize=(12, 6))
            
            # Bar plot
            labels = list(counts.keys())
            values = list(counts.values())
            
            # Truncate if too many classes
            if len(labels) > 20:
                labels = labels[:20]
                values = values[:20]
                labels[-1] = f"Other ({len(counts) - 19} classes)"
                other_count = sum(list(counts.values())[20:])
                values[-1] = other_count
            
            plt.bar(range(len(labels)), values)
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.xlabel('Classes')
            plt.ylabel('Count')
            plt.title(f'Class Distribution (Balance Ratio: {class_dist.get("balance_ratio", 0):.3f})')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _visualize_sample_sizes(self, output_dir: Optional[Path]) -> None:
        """Visualize sample size distribution."""
        basic_stats = self.stats.get('basic', {})
        size_stats = basic_stats.get('sample_size_bytes', {})
        
        if size_stats and 'percentiles' in size_stats:
            # Create histogram
            plt.figure(figsize=(10, 6))
            
            # Simulate distribution based on percentiles
            import numpy as np
            from scipy.stats import norm
            
            # Generate synthetic data for visualization
            mean = size_stats.get('mean', 0)
            std = size_stats.get('std', 1)
            
            if std > 0:
                x = np.linspace(mean - 3*std, mean + 3*std, 100)
                y = norm.pdf(x, mean, std)
                
                plt.plot(x, y, 'b-', linewidth=2)
                plt.fill_between(x, y, alpha=0.3)
                
                # Add percentiles
                percentiles = size_stats['percentiles']
                colors = ['r', 'g', 'b', 'y']
                for i, (pct_name, pct_value) in enumerate(percentiles.items()):
                    plt.axvline(x=pct_value, color=colors[i % len(colors)], 
                              linestyle='--', label=f'{pct_name}%: {pct_value:.0f} bytes')
                
                plt.xlabel('Sample Size (bytes)')
                plt.ylabel('Density')
                plt.title('Sample Size Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if output_dir:
                    plt.savefig(output_dir / 'sample_size_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    def _visualize_modality_stats(self, output_dir: Optional[Path]) -> None:
        """Visualize modality statistics."""
        modality_stats = self.stats.get('modality', {})
        
        if modality_stats:
            plt.figure(figsize=(10, 6))
            
            modalities = list(modality_stats.keys())
            counts = [modality_stats[m].get('count', 0) for m in modalities]
            
            plt.bar(modalities, counts)
            plt.xlabel('Modality')
            plt.ylabel('Sample Count')
            plt.title('Modality Distribution')
            
            for i, count in enumerate(counts):
                plt.text(i, count + max(counts)*0.01, str(count), ha='center')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'modality_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _visualize_quality_metrics(self, output_dir: Optional[Path]) -> None:
        """Visualize quality metrics."""
        quality_metrics = self.stats.get('quality_metrics', {})
        
        if quality_metrics:
            plt.figure(figsize=(8, 6))
            
            metrics_to_plot = []
            values = []
            
            for key, value in quality_metrics.items():
                if isinstance(value, (int, float)) and key.endswith('_pct'):
                    metrics_to_plot.append(key.replace('_pct', '').replace('_', ' ').title())
                    values.append(value)
            
            if metrics_to_plot:
                plt.bar(metrics_to_plot, values)
                plt.xlabel('Quality Metric')
                plt.ylabel('Percentage (%)')
                plt.title('Dataset Quality Metrics')
                plt.ylim(0, 100)
                
                for i, v in enumerate(values):
                    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
                
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(output_dir / 'quality_metrics.png', dpi=150, bbox_inches='tight')
                plt.close()

# Convenience function
def compute_dataset_statistics(dataset: BaseDataset, 
                             force: bool = False,
                             visualize: bool = False,
                             output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        dataset: Dataset instance
        force: Whether to recompute statistics
        visualize: Whether to create visualizations
        output_dir: Directory for output files
        
    Returns:
        Statistics dictionary
    """
    stats_calculator = DatasetStatistics(dataset)
    statistics = stats_calculator.compute(force=force)
    
    if visualize:
        stats_calculator.visualize(output_dir)
    
    return statistics