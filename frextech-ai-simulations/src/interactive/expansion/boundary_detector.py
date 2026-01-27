"""
Boundary Detector Module
Detects and analyzes boundaries in scenes for expansion
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
import time
from collections import defaultdict, deque
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Image processing and computer vision
import cv2
from PIL import Image, ImageFilter
import scipy.ndimage as ndi
from skimage import measure, filters, feature, segmentation
import networkx as nx

# Local imports
from ...utils.metrics import Timer, PerformanceMetrics
from ...utils.file_io import save_json, load_json, save_image, load_image

logger = logging.getLogger(__name__)


class BoundaryType(Enum):
    """Types of boundaries"""
    EDGE = "edge"             # Simple intensity edges
    SEMANTIC = "semantic"     # Semantic boundaries
    DEPTH = "depth"           # Depth discontinuities
    TEXTURE = "texture"       # Texture boundaries
    OBJECT = "object"         # Object boundaries
    SHADOW = "shadow"         # Shadow boundaries
    OCCLUSION = "occlusion"   # Occlusion boundaries


class DetectionMethod(Enum):
    """Boundary detection methods"""
    CANNY = "canny"
    SOBEL = "sobel"
    LAPLACIAN = "laplacian"
    HED = "hed"              # Holistically-nested edge detection
    DEXTR = "dextr"          # Deep extreme cut
    WATERSHED = "watershed"
    GRABCUT = "grabcut"
    SEMANTIC_SEG = "semantic_seg"


@dataclass
class BoundarySegment:
    """Represents a boundary segment"""
    segment_id: str
    points: List[Tuple[float, float]]  # List of (x, y) points
    boundary_type: BoundaryType
    confidence: float
    length: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def start_point(self) -> Tuple[float, float]:
        """Get start point of segment"""
        return self.points[0] if self.points else (0.0, 0.0)
    
    @property
    def end_point(self) -> Tuple[float, float]:
        """Get end point of segment"""
        return self.points[-1] if self.points else (0.0, 0.0)
    
    @property
    def is_closed(self) -> bool:
        """Check if segment forms a closed loop"""
        if len(self.points) < 3:
            return False
        
        # Check if start and end points are close
        start = np.array(self.start_point)
        end = np.array(self.end_point)
        distance = np.linalg.norm(start - end)
        
        return distance < 5.0  # Threshold for closure


@dataclass
class BoundaryRegion:
    """Represents a region defined by boundaries"""
    region_id: str
    boundary_segments: List[BoundarySegment]
    mask: np.ndarray
    bounds: Tuple[int, int, int, int]
    area: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BoundaryDetector:
    """
    Main boundary detector for scene analysis and expansion
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4
    ):
        """
        Initialize boundary detector
        
        Args:
            config: Configuration dictionary
            max_workers: Maximum worker threads
        """
        self.config = config or {}
        self.max_workers = max_workers
        
        # Detection settings
        self.detection_methods: Dict[DetectionMethod, Callable] = {}
        self._initialize_detection_methods()
        
        # Detection results
        self.boundary_segments: Dict[str, BoundarySegment] = {}
        self.boundary_regions: Dict[str, BoundaryRegion] = {}
        self.boundary_graph: Optional[nx.Graph] = None
        
        # Reference data
        self.reference_image: Optional[np.ndarray] = None
        self.reference_depth: Optional[np.ndarray] = None
        self.reference_semantic: Optional[np.ndarray] = None
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info("BoundaryDetector initialized")
    
    def _initialize_detection_methods(self) -> None:
        """Initialize boundary detection methods"""
        self.detection_methods = {
            DetectionMethod.CANNY: self._detect_canny_edges,
            DetectionMethod.SOBEL: self._detect_sobel_edges,
            DetectionMethod.LAPLACIAN: self._detect_laplacian_edges,
            DetectionMethod.WATERSHED: self._detect_watershed_boundaries,
            DetectionMethod.GRABCUT: self._detect_grabcut_boundaries
        }
    
    def set_reference_data(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None
    ) -> bool:
        """
        Set reference data for boundary detection
        
        Args:
            image: Reference image
            depth: Optional depth map
            semantic: Optional semantic segmentation
            
        Returns:
            True if successful
        """
        if image is None or len(image.shape) not in [2, 3]:
            logger.error("Invalid image format")
            return False
        
        self.reference_image = image.copy()
        
        if depth is not None:
            self.reference_depth = depth.copy()
        
        if semantic is not None:
            self.reference_semantic = semantic.copy()
        
        # Clear previous results
        self.boundary_segments.clear()
        self.boundary_regions.clear()
        self.boundary_graph = None
        
        logger.info(f"Set reference data: image={image.shape}, "
                   f"depth={depth.shape if depth is not None else 'None'}, "
                   f"semantic={semantic.shape if semantic is not None else 'None'}")
        
        return True
    
    def detect_boundaries(
        self,
        method: Union[str, DetectionMethod] = DetectionMethod.CANNY,
        parameters: Optional[Dict[str, Any]] = None,
        boundary_types: Optional[List[Union[str, BoundaryType]]] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Detect boundaries in reference data
        
        Args:
            method: Detection method
            parameters: Method-specific parameters
            boundary_types: Types of boundaries to detect
            roi: Region of interest (x, y, width, height)
            
        Returns:
            Dictionary with detection results
        """
        if self.reference_image is None:
            logger.error("No reference data set")
            return {"error": "No reference data"}
        
        timer = Timer()
        
        try:
            # Convert method if string
            if isinstance(method, str):
                try:
                    detection_method = DetectionMethod(method)
                except ValueError:
                    logger.error(f"Unknown detection method: {method}")
                    return {"error": f"Unknown method: {method}"}
            else:
                detection_method = method
            
            # Convert boundary types if strings
            types_to_detect = []
            if boundary_types:
                for bt in boundary_types:
                    if isinstance(bt, str):
                        try:
                            types_to_detect.append(BoundaryType(bt))
                        except ValueError:
                            logger.warning(f"Unknown boundary type: {bt}")
                    else:
                        types_to_detect.append(bt)
            else:
                # Default to edge detection
                types_to_detect = [BoundaryType.EDGE]
            
            # Merge parameters
            merged_params = self._get_default_parameters(detection_method)
            if parameters:
                merged_params.update(parameters)
            
            logger.info(f"Detecting boundaries using {detection_method.value} "
                       f"for types: {[t.value for t in types_to_detect]}")
            
            # Extract ROI if specified
            image_to_process = self.reference_image.copy()
            depth_to_process = self.reference_depth.copy() if self.reference_depth is not None else None
            semantic_to_process = self.reference_semantic.copy() if self.reference_semantic is not None else None
            
            if roi is not None:
                x, y, w, h = roi
                image_to_process = image_to_process[y:y+h, x:x+w]
                if depth_to_process is not None:
                    depth_to_process = depth_to_process[y:y+h, x:x+w]
                if semantic_to_process is not None:
                    semantic_to_process = semantic_to_process[y:y+h, x:x+w]
            
            # Detect boundaries using specified method
            if detection_method in self.detection_methods:
                detection_func = self.detection_methods[detection_method]
                results = detection_func(
                    image_to_process, depth_to_process, semantic_to_process,
                    types_to_detect, merged_params
                )
            else:
                logger.error(f"Detection method not implemented: {detection_method}")
                return {"error": f"Method not implemented: {detection_method}"}
            
            # Process results
            if results and "segments" in results:
                segments = results["segments"]
                
                # Adjust coordinates if ROI was used
                if roi is not None:
                    x_offset, y_offset, _, _ = roi
                    for segment in segments:
                        adjusted_points = []
                        for point in segment.points:
                            adjusted_points.append((
                                point[0] + x_offset,
                                point[1] + y_offset
                            ))
                        segment.points = adjusted_points
                
                # Store segments
                for segment in segments:
                    self.boundary_segments[segment.segment_id] = segment
                
                # Build boundary graph
                self._build_boundary_graph(segments)
                
                # Extract regions if possible
                if results.get("regions"):
                    regions = results["regions"]
                    
                    # Adjust region coordinates if ROI was used
                    if roi is not None:
                        for region in regions:
                            # Adjust mask bounds
                            if region.mask is not None:
                                # This would require more complex adjustment
                                pass
                    
                    for region in regions:
                        self.boundary_regions[region.region_id] = region
            
            # Update metrics
            self.metrics.record_operation(
                f"detect_boundaries_{detection_method.value}", 
                timer.elapsed()
            )
            
            # Prepare response
            response = {
                "method": detection_method.value,
                "parameters": merged_params,
                "boundary_types": [t.value for t in types_to_detect],
                "num_segments": len(results.get("segments", [])),
                "num_regions": len(results.get("regions", [])),
                "processing_time": timer.elapsed(),
                "timestamp": time.time()
            }
            
            if roi is not None:
                response["roi"] = roi
            
            logger.info(f"Boundary detection completed in {timer.elapsed():.2f}s: "
                       f"{response['num_segments']} segments, "
                       f"{response['num_regions']} regions")
            
            return response
            
        except Exception as e:
            logger.error(f"Error detecting boundaries: {e}")
            return {"error": str(e)}
    
    def _get_default_parameters(
        self,
        method: DetectionMethod
    ) -> Dict[str, Any]:
        """Get default parameters for detection method"""
        defaults = {
            DetectionMethod.CANNY: {
                "low_threshold": 50,
                "high_threshold": 150,
                "sigma": 1.0,
                "aperture_size": 3
            },
            DetectionMethod.SOBEL: {
                "kernel_size": 3,
                "scale": 1.0,
                "delta": 0.0,
                "border_type": cv2.BORDER_DEFAULT
            },
            DetectionMethod.LAPLACIAN: {
                "kernel_size": 3,
                "scale": 1.0,
                "delta": 0.0
            },
            DetectionMethod.WATERSHED: {
                "marker_method": "gradient",
                "compactness": 0.01,
                "min_size": 100
            },
            DetectionMethod.GRABCUT: {
                "iterations": 5,
                "mode": cv2.GC_INIT_WITH_RECT
            }
        }
        
        return defaults.get(method, {}).copy()
    
    # Detection method implementations
    def _detect_canny_edges(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        semantic: Optional[np.ndarray],
        boundary_types: List[BoundaryType],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect edges using Canny edge detector"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        sigma = parameters.get("sigma", 1.0)
        if sigma > 0:
            gray = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # Apply Canny edge detection
        low_threshold = parameters.get("low_threshold", 50)
        high_threshold = parameters.get("high_threshold", 150)
        aperture_size = parameters.get("aperture_size", 3)
        
        edges = cv2.Canny(gray, low_threshold, high_threshold, 
                         apertureSize=aperture_size)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to segments
        segments = []
        for i, contour in enumerate(contours):
            if len(contour) >= 2:
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Convert to list of points
                points = [(float(point[0][0]), float(point[0][1])) 
                         for point in approx]
                
                if points:
                    segment_id = f"canny_{i}_{int(time.time()*1000)}"
                    
                    # Calculate confidence based on edge strength
                    # For Canny, we can use the fact that edges are binary
                    # and all detected edges have high confidence
                    confidence = 0.9
                    
                    # Calculate length
                    length = cv2.arcLength(contour, False)
                    
                    segment = BoundarySegment(
                        segment_id=segment_id,
                        points=points,
                        boundary_type=BoundaryType.EDGE,
                        confidence=confidence,
                        length=length,
                        metadata={
                            "method": "canny",
                            "contour_length": len(contour),
                            "simplified_points": len(points)
                        }
                    )
                    
                    segments.append(segment)
        
        return {"segments": segments}
    
    def _detect_sobel_edges(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        semantic: Optional[np.ndarray],
        boundary_types: List[BoundaryType],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect edges using Sobel operator"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Sobel operator
        kernel_size = parameters.get("kernel_size", 3)
        scale = parameters.get("scale", 1.0)
        delta = parameters.get("delta", 0.0)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size, 
                          scale=scale, delta=delta)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size, 
                          scale=scale, delta=delta)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Normalize magnitude
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        magnitude_norm = magnitude_norm.astype(np.uint8)
        
        # Threshold to get edges
        _, edges = cv2.threshold(magnitude_norm, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert to segments
        segments = []
        for i, contour in enumerate(contours):
            if len(contour) >= 2:
                points = [(float(point[0][0]), float(point[0][1])) 
                         for point in contour]
                
                if points:
                    segment_id = f"sobel_{i}_{int(time.time()*1000)}"
                    
                    # Calculate average gradient magnitude along contour
                    # as confidence measure
                    contour_mask = np.zeros_like(gray)
                    cv2.drawContours(contour_mask, [contour], -1, 1, 1)
                    contour_magnitude = np.mean(magnitude_norm[contour_mask > 0])
                    confidence = min(contour_magnitude / 255.0, 1.0)
                    
                    length = cv2.arcLength(contour, False)
                    
                    segment = BoundarySegment(
                        segment_id=segment_id,
                        points=points,
                        boundary_type=BoundaryType.EDGE,
                        confidence=float(confidence),
                        length=length,
                        metadata={
                            "method": "sobel",
                            "avg_gradient": float(contour_magnitude)
                        }
                    )
                    
                    segments.append(segment)
        
        return {"segments": segments}
    
    def _detect_laplacian_edges(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        semantic: Optional[np.ndarray],
        boundary_types: List[BoundaryType],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect edges using Laplacian operator"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Laplacian operator
        kernel_size = parameters.get("kernel_size", 3)
        scale = parameters.get("scale", 1.0)
        delta = parameters.get("delta", 0.0)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size, 
                                 scale=scale, delta=delta)
        
        # Take absolute value and normalize
        laplacian_abs = np.abs(laplacian)
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)
        laplacian_norm = laplacian_norm.astype(np.uint8)
        
        # Zero-crossing detection (simplified)
        laplacian_sgn = np.sign(laplacian)
        zero_crossing = np.zeros_like(laplacian_sgn, dtype=np.uint8)
        
        # Simple zero-crossing detection
        for i in range(1, laplacian_sgn.shape[0] - 1):
            for j in range(1, laplacian_sgn.shape[1] - 1):
                if (laplacian_sgn[i, j] == 0 or
                    laplacian_sgn[i, j] * laplacian_sgn[i-1, j] < 0 or
                    laplacian_sgn[i, j] * laplacian_sgn[i+1, j] < 0 or
                    laplacian_sgn[i, j] * laplacian_sgn[i, j-1] < 0 or
                    laplacian_sgn[i, j] * laplacian_sgn[i, j+1] < 0):
                    zero_crossing[i, j] = 255
        
        # Find contours
        contours, _ = cv2.findContours(
            zero_crossing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert to segments
        segments = []
        for i, contour in enumerate(contours):
            if len(contour) >= 2:
                points = [(float(point[0][0]), float(point[0][1])) 
                         for point in contour]
                
                if points:
                    segment_id = f"laplacian_{i}_{int(time.time()*1000)}"
                    
                    # Calculate average Laplacian magnitude along contour
                    contour_mask = np.zeros_like(gray)
                    cv2.drawContours(contour_mask, [contour], -1, 1, 1)
                    contour_laplacian = np.mean(laplacian_norm[contour_mask > 0])
                    confidence = min(contour_laplacian / 255.0, 1.0)
                    
                    length = cv2.arcLength(contour, False)
                    
                    segment = BoundarySegment(
                        segment_id=segment_id,
                        points=points,
                        boundary_type=BoundaryType.EDGE,
                        confidence=float(confidence),
                        length=length,
                        metadata={
                            "method": "laplacian",
                            "avg_laplacian": float(contour_laplacian)
                        }
                    )
                    
                    segments.append(segment)
        
        return {"segments": segments}
    
    def _detect_watershed_boundaries(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        semantic: Optional[np.ndarray],
        boundary_types: List[BoundaryType],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect boundaries using watershed segmentation"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            rgb = image.copy()
        
        # Convert to grayscale for gradient calculation
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply watershed segmentation
        marker_method = parameters.get("marker_method", "gradient")
        
        if marker_method == "gradient":
            # Use gradient magnitude for markers
            _, markers = cv2.threshold(
                magnitude.astype(np.uint8), 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Distance transform
            dist_transform = cv2.distanceTransform(markers, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(
                dist_transform, 0.7 * dist_transform.max(), 255, 0
            )
            
            # Marker labeling
            sure_fg = np.uint8(sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            
            # Add one to all labels so that sure background is 1
            markers = markers + 1
            
            # Mark the unknown region
            markers[markers == 0] = 0
        
        else:
            # Simple marker generation
            ret, markers = cv2.threshold(gray, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply watershed
        markers = cv2.watershed(rgb, markers)
        
        # Extract boundaries
        boundaries = np.zeros_like(gray, dtype=np.uint8)
        boundaries[markers == -1] = 255  # Watershed boundaries are marked with -1
        
        # Find contours
        contours, _ = cv2.findContours(
            boundaries, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert to segments
        segments = []
        regions = []
        
        for i, contour in enumerate(contours):
            if len(contour) >= 2:
                points = [(float(point[0][0]), float(point[0][1])) 
                         for point in contour]
                
                if points:
                    segment_id = f"watershed_{i}_{int(time.time()*1000)}"
                    
                    # Calculate confidence based on gradient magnitude
                    contour_mask = np.zeros_like(gray)
                    cv2.drawContours(contour_mask, [contour], -1, 1, 1)
                    contour_gradient = np.mean(magnitude[contour_mask > 0])
                    confidence = min(contour_gradient / magnitude.max(), 1.0)
                    
                    length = cv2.arcLength(contour, False)
                    
                    segment = BoundarySegment(
                        segment_id=segment_id,
                        points=points,
                        boundary_type=BoundaryType.OBJECT,
                        confidence=float(confidence),
                        length=length,
                        metadata={
                            "method": "watershed",
                            "avg_gradient": float(contour_gradient)
                        }
                    )
                    
                    segments.append(segment)
        
        # Extract regions from markers
        unique_markers = np.unique(markers)
        for marker in unique_markers:
            if marker > 0:  # Skip background and boundaries
                region_mask = (markers == marker).astype(np.uint8) * 255
                
                if np.sum(region_mask) > 0:
                    # Find contours of region
                    region_contours, _ = cv2.findContours(
                        region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if region_contours:
                        contour = region_contours[0]
                        bounds = cv2.boundingRect(contour)
                        area = cv2.contourArea(contour)
                        
                        region_id = f"watershed_region_{marker}_{int(time.time()*1000)}"
                        
                        region = BoundaryRegion(
                            region_id=region_id,
                            boundary_segments=[],
                            mask=region_mask,
                            bounds=bounds,
                            area=area,
                            metadata={
                                "method": "watershed",
                                "marker_id": int(marker),
                                "contour_area": float(area)
                            }
                        )
                        
                        regions.append(region)
        
        return {"segments": segments, "regions": regions}
    
    def _detect_grabcut_boundaries(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        semantic: Optional[np.ndarray],
        boundary_types: List[BoundaryType],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect boundaries using GrabCut segmentation"""
        # GrabCut requires RGB image
        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            rgb = image.copy()
        
        h, w = rgb.shape[:2]
        
        # Create initial mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Define initial rectangle (center 70% of image)
        rect_x = int(w * 0.15)
        rect_y = int(h * 0.15)
        rect_w = int(w * 0.7)
        rect_h = int(h * 0.7)
        
        # Initialize mask
        mask[:] = cv2.GC_PR_BGD  # Probable background
        
        # Set rectangle area to probable foreground
        mask[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w] = cv2.GC_PR_FGD
        
        # Set border to definite background
        border_size = 5
        mask[:border_size, :] = cv2.GC_BGD
        mask[-border_size:, :] = cv2.GC_BGD
        mask[:, :border_size] = cv2.GC_BGD
        mask[:, -border_size:] = cv2.GC_BGD
        
        # Create background and foreground models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        # Run GrabCut
        iterations = parameters.get("iterations", 5)
        mode = parameters.get("mode", cv2.GC_INIT_WITH_MASK)
        
        cv2.grabCut(rgb, mask, None, bgd_model, fgd_model, 
                   iterations, mode)
        
        # Extract foreground mask
        foreground_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 
                                  255, 0).astype(np.uint8)
        
        # Find boundaries
        contours, _ = cv2.findContours(
            foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert to segments
        segments = []
        
        for i, contour in enumerate(contours):
            if len(contour) >= 2:
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                points = [(float(point[0][0]), float(point[0][1])) 
                         for point in approx]
                
                if points:
                    segment_id = f"grabcut_{i}_{int(time.time()*1000)}"
                    
                    # Calculate confidence based on GrabCut mask values
                    # Areas marked as GC_FGD have higher confidence
                    contour_mask = np.zeros_like(mask)
                    cv2.drawContours(contour_mask, [contour], -1, 1, 1)
                    
                    # Calculate ratio of definite foreground in contour area
                    fgd_pixels = np.sum((mask[contour_mask > 0] == cv2.GC_FGD).astype(int))
                    total_pixels = np.sum(contour_mask > 0)
                    confidence = fgd_pixels / total_pixels if total_pixels > 0 else 0.5
                    
                    length = cv2.arcLength(contour, False)
                    
                    segment = BoundarySegment(
                        segment_id=segment_id,
                        points=points,
                        boundary_type=BoundaryType.OBJECT,
                        confidence=float(confidence),
                        length=length,
                        metadata={
                            "method": "grabcut",
                            "fgd_ratio": float(confidence),
                            "total_pixels": int(total_pixels)
                        }
                    )
                    
                    segments.append(segment)
        
        # Create region from foreground mask
        region_id = f"grabcut_region_{int(time.time()*1000)}"
        bounds = cv2.boundingRect(foreground_mask)
        area = np.sum(foreground_mask > 0)
        
        region = BoundaryRegion(
            region_id=region_id,
            boundary_segments=segments,
            mask=foreground_mask,
            bounds=bounds,
            area=float(area),
            metadata={
                "method": "grabcut",
                "iterations": iterations,
                "mode": mode
            }
        )
        
        return {"segments": segments, "regions": [region]}
    
    def _build_boundary_graph(self, segments: List[BoundarySegment]) -> None:
        """Build graph representation of boundary segments"""
        self.boundary_graph = nx.Graph()
        
        # Add segments as nodes
        for segment in segments:
            self.boundary_graph.add_node(
                segment.segment_id,
                segment=segment,
                type=segment.boundary_type.value,
                confidence=segment.confidence,
                length=segment.length,
                points=segment.points
            )
        
        # Add edges between connected segments
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments[i+1:], i+1):
                if self._are_segments_connected(seg1, seg2):
                    self.boundary_graph.add_edge(
                        seg1.segment_id, seg2.segment_id,
                        connection_type="adjacent",
                        distance=self._segment_distance(seg1, seg2)
                    )
    
    def _are_segments_connected(
        self,
        seg1: BoundarySegment,
        seg2: BoundarySegment,
        distance_threshold: float = 10.0
    ) -> bool:
        """Check if two segments are connected"""
        # Check end points proximity
        for point1 in [seg1.start_point, seg1.end_point]:
            for point2 in [seg2.start_point, seg2.end_point]:
                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                if dist < distance_threshold:
                    return True
        
        return False
    
    def _segment_distance(
        self,
        seg1: BoundarySegment,
        seg2: BoundarySegment
    ) -> float:
        """Calculate minimum distance between two segments"""
        min_distance = float('inf')
        
        for point1 in seg1.points:
            for point2 in seg2.points:
                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                if dist < min_distance:
                    min_distance = dist
        
        return min_distance
    
    def analyze_boundaries(
        self,
        analysis_type: str = "connectivity"
    ) -> Dict[str, Any]:
        """
        Analyze detected boundaries
        
        Args:
            analysis_type: Type of analysis
            
        Returns:
            Analysis results
        """
        if not self.boundary_segments:
            logger.warning("No boundaries to analyze")
            return {"error": "No boundaries detected"}
        
        timer = Timer()
        
        try:
            if analysis_type == "connectivity":
                results = self._analyze_connectivity()
            elif analysis_type == "strength":
                results = self._analyze_strength()
            elif analysis_type == "closure":
                results = self._analyze_closure()
            elif analysis_type == "hierarchy":
                results = self._analyze_hierarchy()
            else:
                logger.error(f"Unknown analysis type: {analysis_type}")
                return {"error": f"Unknown analysis type: {analysis_type}"}
            
            # Update metrics
            self.metrics.record_operation(f"analyze_{analysis_type}", timer.elapsed())
            
            logger.info(f"Boundary analysis ({analysis_type}) "
                       f"completed in {timer.elapsed():.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing boundaries: {e}")
            return {"error": str(e)}
    
    def _analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze boundary connectivity"""
        if self.boundary_graph is None:
            return {"error": "Boundary graph not built"}
        
        # Calculate graph metrics
        num_nodes = self.boundary_graph.number_of_nodes()
        num_edges = self.boundary_graph.number_of_edges()
        
        # Find connected components
        components = list(nx.connected_components(self.boundary_graph))
        num_components = len(components)
        
        # Calculate component sizes
        component_sizes = [len(comp) for comp in components]
        
        # Calculate average degree
        degrees = [deg for _, deg in self.boundary_graph.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        
        return {
            "analysis_type": "connectivity",
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_components": num_components,
            "component_sizes": component_sizes,
            "avg_degree": float(avg_degree),
            "max_component_size": max(component_sizes) if component_sizes else 0,
            "min_component_size": min(component_sizes) if component_sizes else 0
        }
    
    def _analyze_strength(self) -> Dict[str, Any]:
        """Analyze boundary strength"""
        if not self.boundary_segments:
            return {"error": "No boundary segments"}
        
        confidences = [seg.confidence for seg in self.boundary_segments.values()]
        lengths = [seg.length for seg in self.boundary_segments.values()]
        
        # Group by boundary type
        type_confidences = defaultdict(list)
        type_lengths = defaultdict(list)
        
        for seg in self.boundary_segments.values():
            type_confidences[seg.boundary_type.value].append(seg.confidence)
            type_lengths[seg.boundary_type.value].append(seg.length)
        
        # Calculate statistics
        stats = {
            "analysis_type": "strength",
            "total_segments": len(confidences),
            "overall": {
                "avg_confidence": float(np.mean(confidences)) if confidences else 0,
                "std_confidence": float(np.std(confidences)) if len(confidences) > 1 else 0,
                "avg_length": float(np.mean(lengths)) if lengths else 0,
                "total_length": float(np.sum(lengths)) if lengths else 0
            },
            "by_type": {}
        }
        
        for boundary_type, conf_list in type_confidences.items():
            stats["by_type"][boundary_type] = {
                "count": len(conf_list),
                "avg_confidence": float(np.mean(conf_list)) if conf_list else 0,
                "avg_length": float(np.mean(type_lengths[boundary_type])) 
                            if type_lengths[boundary_type] else 0
            }
        
        return stats
    
    def _analyze_closure(self) -> Dict[str, Any]:
        """Analyze boundary closure (closed loops)"""
        if not self.boundary_segments:
            return {"error": "No boundary segments"}
        
        closed_segments = []
        open_segments = []
        
        for seg in self.boundary_segments.values():
            if seg.is_closed:
                closed_segments.append(seg)
            else:
                open_segments.append(seg)
        
        # Calculate statistics for closed segments
        closed_areas = []
        closed_perimeters = []
        
        for seg in closed_segments:
            # Convert points to contour
            points = np.array(seg.points, dtype=np.int32).reshape((-1, 1, 2))
            area = cv2.contourArea(points)
            perimeter = cv2.arcLength(points, True)
            
            closed_areas.append(area)
            closed_perimeters.append(perimeter)
        
        return {
            "analysis_type": "closure",
            "total_segments": len(self.boundary_segments),
            "closed_segments": len(closed_segments),
            "open_segments": len(open_segments),
            "closure_ratio": len(closed_segments) / len(self.boundary_segments) 
                           if self.boundary_segments else 0,
            "closed_areas": {
                "count": len(closed_areas),
                "total": float(np.sum(closed_areas)) if closed_areas else 0,
                "avg": float(np.mean(closed_areas)) if closed_areas else 0,
                "min": float(np.min(closed_areas)) if closed_areas else 0,
                "max": float(np.max(closed_areas)) if closed_areas else 0
            },
            "closed_perimeters": {
                "total": float(np.sum(closed_perimeters)) if closed_perimeters else 0,
                "avg": float(np.mean(closed_perimeters)) if closed_perimeters else 0
            }
        }
    
    def _analyze_hierarchy(self) -> Dict[str, Any]:
        """Analyze boundary hierarchy (nested boundaries)"""
        if not self.boundary_segments:
            return {"error": "No boundary segments"}
        
        # Find closed segments
        closed_segments = [seg for seg in self.boundary_segments.values() 
                          if seg.is_closed]
        
        # Build hierarchy tree
        hierarchy = []
        
        for i, seg1 in enumerate(closed_segments):
            containment_count = 0
            
            for j, seg2 in enumerate(closed_segments):
                if i != j:
                    if self._segment_contains(seg1, seg2):
                        containment_count += 1
            
            hierarchy.append({
                "segment_id": seg1.segment_id,
                "area": cv2.contourArea(
                    np.array(seg1.points, dtype=np.int32).reshape((-1, 1, 2))
                ),
                "contains": containment_count,
                "confidence": seg1.confidence
            })
        
        # Sort by area (largest first)
        hierarchy.sort(key=lambda x: x["area"], reverse=True)
        
        return {
            "analysis_type": "hierarchy",
            "total_closed": len(closed_segments),
            "hierarchy_levels": self._calculate_hierarchy_levels(hierarchy),
            "segments": hierarchy[:10]  # Top 10 largest
        }
    
    def _segment_contains(
        self,
        outer_seg: BoundarySegment,
        inner_seg: BoundarySegment
    ) -> bool:
        """Check if outer segment contains inner segment"""
        if not outer_seg.is_closed or not inner_seg.is_closed:
            return False
        
        # Convert to contours
        outer_points = np.array(outer_seg.points, dtype=np.int32).reshape((-1, 1, 2))
        inner_points = np.array(inner_seg.points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Check if center of inner segment is inside outer segment
        inner_center = np.mean(inner_points, axis=0).flatten()
        
        return cv2.pointPolygonTest(outer_points, tuple(inner_center), False) > 0
    
    def _calculate_hierarchy_levels(
        self,
        hierarchy: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate hierarchy levels from containment relationships"""
        # This is a simplified implementation
        # In production, would build a proper tree structure
        
        levels = defaultdict(list)
        
        for seg in hierarchy:
            if seg["contains"] == 0:
                levels["root"].append(seg["segment_id"])
            elif seg["contains"] == 1:
                levels["level1"].append(seg["segment_id"])
            elif seg["contains"] == 2:
                levels["level2"].append(seg["segment_id"])
            else:
                levels["deep"].append(seg["segment_id"])
        
        return dict(levels)
    
    def get_boundary_preview(
        self,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_segments: bool = True,
        show_regions: bool = False
    ) -> Optional[np.ndarray]:
        """
        Get preview of detected boundaries
        
        Args:
            overlay_color: Boundary color
            thickness: Line thickness
            show_segments: Whether to show boundary segments
            show_regions: Whether to show boundary regions
            
        Returns:
            Preview image
        """
        if self.reference_image is None:
            return None
        
        # Create base image
        if len(self.reference_image.shape) == 2:
            preview = cv2.cvtColor(self.reference_image, cv2.COLOR_GRAY2BGR)
        else:
            preview = self.reference_image.copy()
        
        # Draw boundary segments
        if show_segments:
            for segment in self.boundary_segments.values():
                if len(segment.points) >= 2:
                    # Convert points to integer coordinates
                    points = np.array(segment.points, dtype=np.int32)
                    
                    # Draw polyline
                    if len(points) > 1:
                        cv2.polylines(preview, [points], False, overlay_color, thickness)
                    
                    # Draw start and end points
                    cv2.circle(preview, tuple(map(int, segment.start_point)), 
                              thickness * 2, (255, 0, 0), -1)
                    cv2.circle(preview, tuple(map(int, segment.end_point)), 
                              thickness * 2, (0, 0, 255), -1)
        
        # Draw boundary regions
        if show_regions:
            for region in self.boundary_regions.values():
                if region.mask is not None:
                    # Create region overlay
                    overlay = np.zeros_like(preview)
                    overlay[region.mask > 0] = [0, 0, 255]  # Red overlay
                    
                    # Blend with preview
                    preview = cv2.addWeighted(preview, 0.7, overlay, 0.3, 0)
                    
                    # Draw bounding box
                    x, y, w, h = region.bounds
                    cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        return preview
    
    def export_boundaries(
        self,
        output_path: Union[str, Path],
        format: str = "json"
    ) -> bool:
        """
        Export detected boundaries to file
        
        Args:
            output_path: Output file path
            format: Export format
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            
            if format == "json":
                # Prepare data
                data = {
                    "metadata": {
                        "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "num_segments": len(self.boundary_segments),
                        "num_regions": len(self.boundary_regions),
                        "reference_image": self.image_metadata if hasattr(self, 'image_metadata') else {}
                    },
                    "segments": [
                        {
                            "id": seg.segment_id,
                            "type": seg.boundary_type.value,
                            "points": seg.points,
                            "confidence": seg.confidence,
                            "length": seg.length,
                            "is_closed": seg.is_closed,
                            "metadata": seg.metadata
                        }
                        for seg in self.boundary_segments.values()
                    ],
                    "regions": [
                        {
                            "id": reg.region_id,
                            "bounds": reg.bounds,
                            "area": reg.area,
                            "metadata": reg.metadata
                        }
                        for reg in self.boundary_regions.values()
                    ]
                }
                
                save_json(data, output_path.with_suffix('.json'))
                
            elif format == "image":
                # Export preview image
                preview = self.get_boundary_preview()
                
                if preview is not None:
                    cv2.imwrite(str(output_path), preview)
                else:
                    logger.error("Failed to create preview")
                    return False
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported boundaries to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting boundaries: {e}")
            return False
    
    def clear_boundaries(self) -> None:
        """Clear all detected boundaries"""
        self.boundary_segments.clear()
        self.boundary_regions.clear()
        self.boundary_graph = None
        logger.info("Cleared all boundaries")
    
    def get_boundary_stats(self) -> Dict[str, Any]:
        """Get statistics about detected boundaries"""
        return {
            "num_segments": len(self.boundary_segments),
            "num_regions": len(self.boundary_regions),
            "total_length": sum(seg.length for seg in self.boundary_segments.values()),
            "avg_confidence": np.mean([seg.confidence for seg in self.boundary_segments.values()]) 
                            if self.boundary_segments else 0,
            "closed_segments": sum(1 for seg in self.boundary_segments.values() 
                                  if seg.is_closed)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.get_summary()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        logger.info("BoundaryDetector cleaned up")
    
    def __str__(self) -> str:
        """String representation"""
        if self.reference_image is None:
            return "BoundaryDetector(no reference data)"
        
        h, w = self.reference_image.shape[:2]
        num_segments = len(self.boundary_segments)
        num_regions = len(self.boundary_regions)
        
        return (f"BoundaryDetector({w}x{h}, "
                f"segments={num_segments}, "
                f"regions={num_regions})")


# Factory function for creating boundary detectors
def create_boundary_detector(
    config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4
) -> BoundaryDetector:
    """
    Factory function to create boundary detectors
    
    Args:
        config: Configuration dictionary
        max_workers: Maximum worker threads
        
    Returns:
        BoundaryDetector instance
    """
    return BoundaryDetector(config=config, max_workers=max_workers)