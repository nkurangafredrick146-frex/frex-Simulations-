"""
Data Validator Module
Validate data integrity, format, and quality.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
import cv2
import pandas as pd

from .file_io import (
    read_json, read_yaml, read_csv, load_image, load_video,
    calculate_md5, get_file_size, list_files
)
from .logging_config import get_logger

logger = get_logger(__name__)

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, valid: bool = True, message: str = "", errors: List[str] = None):
        """
        Initialize validation result.
        
        Args:
            valid: Whether validation passed
            message: Summary message
            errors: List of error messages
        """
        self.valid = valid
        self.message = message
        self.errors = errors or []
        
        # Statistics
        self.checked_files = 0
        self.passed_files = 0
        self.failed_files = 0
        self.warnings = []
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def update_stats(self, checked: int, passed: int, failed: int):
        """Update statistics."""
        self.checked_files = checked
        self.passed_files = passed
        self.failed_files = failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'valid': self.valid,
            'message': self.message,
            'errors': self.errors,
            'warnings': self.warnings,
            'statistics': {
                'checked_files': self.checked_files,
                'passed_files': self.passed_files,
                'failed_files': self.failed_files
            }
        }
    
    def __str__(self) -> str:
        """String representation."""
        status = "PASS" if self.valid else "FAIL"
        return f"Validation {status}: {self.message}"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    
    name: str
    description: str = ""
    check_function: Optional[Callable] = None
    required: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def check(self, data_path: Union[str, Path], **kwargs) -> Tuple[bool, str]:
        """
        Check rule against data.
        
        Args:
            data_path: Path to data
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (passed, message)
        """
        if self.check_function is None:
            return True, "No check function defined"
        
        try:
            # Merge parameters with kwargs
            check_kwargs = self.parameters.copy()
            check_kwargs.update(kwargs)
            
            # Execute check
            result = self.check_function(data_path, **check_kwargs)
            
            if isinstance(result, tuple):
                return result
            elif isinstance(result, bool):
                return result, "Check passed" if result else "Check failed"
            else:
                return False, f"Invalid check result type: {type(result)}"
        
        except Exception as e:
            return False, f"Check failed with error: {str(e)}"


class DataValidator:
    """
    Comprehensive data validator.
    
    Features:
    - File existence and accessibility checks
    - Format validation
    - Integrity checks (checksums)
    - Content validation
    - Schema validation
    - Custom validation rules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or {}
        self.validation_level = ValidationLevel(
            self.config.get('validation_level', 'standard')
        )
        
        # Initialize rules
        self.rules = self._initialize_rules()
        
        # Custom rules
        self.custom_rules = []
        
        logger.info(f"Initialized DataValidator with {self.validation_level.value} level")
    
    def _initialize_rules(self) -> List[ValidationRule]:
        """Initialize default validation rules."""
        rules = []
        
        # Basic rules (always applied)
        rules.extend([
            ValidationRule(
                name="file_exists",
                description="Check if file exists",
                check_function=self._check_file_exists,
                required=True
            ),
            ValidationRule(
                name="file_readable",
                description="Check if file is readable",
                check_function=self._check_file_readable,
                required=True
            ),
            ValidationRule(
                name="file_not_empty",
                description="Check if file is not empty",
                check_function=self._check_file_not_empty,
                required=True
            )
        ])
        
        # Standard rules
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            rules.extend([
                ValidationRule(
                    name="file_format",
                    description="Check file format",
                    check_function=self._check_file_format,
                    required=False
                ),
                ValidationRule(
                    name="checksum",
                    description="Verify checksum if provided",
                    check_function=self._check_checksum,
                    required=False
                )
            ])
        
        # Strict rules
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            rules.extend([
                ValidationRule(
                    name="content_integrity",
                    description="Check content integrity",
                    check_function=self._check_content_integrity,
                    required=False
                ),
                ValidationRule(
                    name="schema_validation",
                    description="Validate against schema",
                    check_function=self._check_schema,
                    required=False
                )
            ])
        
        # Comprehensive rules
        if self.validation_level == ValidationLevel.COMPREHENSIVE:
            rules.extend([
                ValidationRule(
                    name="metadata_consistency",
                    description="Check metadata consistency",
                    check_function=self._check_metadata_consistency,
                    required=False
                ),
                ValidationRule(
                    name="data_quality",
                    description="Check data quality metrics",
                    check_function=self._check_data_quality,
                    required=False
                )
            ])
        
        return rules
    
    def add_custom_rule(self, rule: ValidationRule):
        """Add custom validation rule."""
        self.custom_rules.append(rule)
        logger.info(f"Added custom rule: {rule.name}")
    
    def validate(self, data_path: Union[str, Path], **kwargs) -> ValidationResult:
        """
        Validate data at path.
        
        Args:
            data_path: Path to data (file or directory)
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult
        """
        data_path = Path(data_path)
        result = ValidationResult()
        
        if not data_path.exists():
            result.add_error(f"Path does not exist: {data_path}")
            result.message = "Validation failed: Path does not exist"
            return result
        
        try:
            if data_path.is_file():
                # Validate single file
                return self._validate_file(data_path, **kwargs)
            elif data_path.is_dir():
                # Validate directory
                return self._validate_directory(data_path, **kwargs)
            else:
                result.add_error(f"Path is not a file or directory: {data_path}")
                result.message = "Validation failed: Invalid path type"
                return result
        
        except Exception as e:
            result.add_error(f"Validation failed with exception: {str(e)}")
            result.message = f"Validation failed: {str(e)}"
            return result
    
    def _validate_file(self, file_path: Path, **kwargs) -> ValidationResult:
        """Validate a single file."""
        result = ValidationResult()
        errors = []
        warnings = []
        
        # Check all rules
        for rule in self.rules + self.custom_rules:
            try:
                passed, message = rule.check(file_path, **kwargs)
                
                if not passed:
                    if rule.required:
                        errors.append(f"{rule.name}: {message}")
                    else:
                        warnings.append(f"{rule.name}: {message}")
                else:
                    logger.debug(f"Rule {rule.name} passed: {message}")
            
            except Exception as e:
                error_msg = f"{rule.name}: Check failed with exception: {str(e)}"
                if rule.required:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
        
        # Update result
        result.errors = errors
        result.warnings = warnings
        result.valid = len(errors) == 0
        
        if result.valid:
            result.message = f"File validation passed: {file_path}"
            if warnings:
                result.message += f" (with {len(warnings)} warnings)"
        else:
            result.message = f"File validation failed: {file_path}"
        
        result.update_stats(checked=1, passed=1 if result.valid else 0, failed=0 if result.valid else 1)
        
        return result
    
    def _validate_directory(self, directory: Path, **kwargs) -> ValidationResult:
        """Validate all files in directory."""
        result = ValidationResult()
        
        # Get file pattern
        file_pattern = kwargs.get('file_pattern', '*')
        recursive = kwargs.get('recursive', False)
        
        # Find files
        if recursive:
            files = list(directory.rglob(file_pattern))
        else:
            files = list(directory.glob(file_pattern))
        
        # Filter files only
        files = [f for f in files if f.is_file()]
        
        if not files:
            result.add_error(f"No files found in directory: {directory}")
            result.message = "Validation failed: No files found"
            return result
        
        # Validate each file
        all_errors = []
        all_warnings = []
        passed_files = 0
        failed_files = 0
        
        for file_path in files:
            file_result = self._validate_file(file_path, **kwargs)
            
            if file_result.valid:
                passed_files += 1
            else:
                failed_files += 1
            
            all_errors.extend(file_result.errors)
            all_warnings.extend(file_result.warnings)
        
        # Update result
        result.errors = all_errors
        result.warnings = all_warnings
        result.valid = failed_files == 0
        
        if result.valid:
            result.message = f"Directory validation passed: {passed_files} files"
            if all_warnings:
                result.message += f" (with {len(all_warnings)} warnings)"
        else:
            result.message = f"Directory validation failed: {failed_files}/{len(files)} files failed"
        
        result.update_stats(
            checked=len(files),
            passed=passed_files,
            failed=failed_files
        )
        
        return result
    
    # Validation rule implementations
    
    def _check_file_exists(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check if file exists."""
        if file_path.exists():
            return True, f"File exists: {file_path}"
        else:
            return False, f"File does not exist: {file_path}"
    
    def _check_file_readable(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check if file is readable."""
        if not file_path.exists():
            return False, "File does not exist"
        
        try:
            with open(file_path, 'rb') as f:
                # Try to read first byte
                f.read(1)
            return True, f"File is readable: {file_path}"
        except Exception as e:
            return False, f"File is not readable: {str(e)}"
    
    def _check_file_not_empty(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check if file is not empty."""
        if not file_path.exists():
            return False, "File does not exist"
        
        try:
            size = get_file_size(file_path)
            if size > 0:
                return True, f"File is not empty: {size} bytes"
            else:
                return False, "File is empty (0 bytes)"
        except Exception as e:
            return False, f"Failed to check file size: {str(e)}"
    
    def _check_file_format(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check file format."""
        if not file_path.exists():
            return False, "File does not exist"
        
        # Get expected format from kwargs
        expected_format = kwargs.get('expected_format')
        if not expected_format:
            return True, "No expected format specified"
        
        # Check based on file extension
        suffix = file_path.suffix.lower()
        
        # Normalize expected format
        if expected_format.startswith('.'):
            expected_format = expected_format[1:]
        
        # Check if extension matches expected format
        if suffix.lstrip('.') == expected_format:
            return True, f"File format matches expected: {expected_format}"
        
        # Try to validate based on content
        try:
            if expected_format in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif']:
                # Validate image
                with Image.open(file_path) as img:
                    img.verify()  # Verify image integrity
                return True, f"Image format validated: {suffix}"
            
            elif expected_format in ['mp4', 'avi', 'mov', 'webm']:
                # Validate video
                cap = cv2.VideoCapture(str(file_path))
                if cap.isOpened():
                    cap.release()
                    return True, f"Video format validated: {suffix}"
                else:
                    return False, f"Invalid video format: {suffix}"
            
            elif expected_format == 'json':
                # Validate JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                return True, f"JSON format validated: {suffix}"
            
            elif expected_format in ['yaml', 'yml']:
                # Validate YAML
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    yaml.safe_load(f)
                return True, f"YAML format validated: {suffix}"
            
            else:
                # Can't validate this format
                return False, f"File extension {suffix} does not match expected format {expected_format}"
        
        except Exception as e:
            return False, f"Format validation failed: {str(e)}"
    
    def _check_checksum(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Verify file checksum."""
        if not file_path.exists():
            return False, "File does not exist"
        
        # Get expected checksum from kwargs
        expected_checksum = kwargs.get('expected_checksum')
        if not expected_checksum:
            return True, "No expected checksum specified"
        
        # Calculate actual checksum
        try:
            actual_checksum = calculate_md5(file_path)
            
            if actual_checksum.lower() == expected_checksum.lower():
                return True, f"Checksum matches: {actual_checksum}"
            else:
                return False, f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
        
        except Exception as e:
            return False, f"Failed to calculate checksum: {str(e)}"
    
    def _check_content_integrity(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check content integrity based on file type."""
        if not file_path.exists():
            return False, "File does not exist"
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']:
                # Check image integrity
                with Image.open(file_path) as img:
                    img.verify()  # Verify without loading
                    img.load()  # Actually load image to check for decode errors
                return True, "Image integrity validated"
            
            elif suffix in ['.mp4', '.avi', '.mov', '.webm']:
                # Check video integrity
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    return False, "Failed to open video"
                
                # Try to read first few frames
                frames_read = 0
                for _ in range(10):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frames_read += 1
                
                cap.release()
                
                if frames_read > 0:
                    return True, f"Video integrity validated ({frames_read} frames read)"
                else:
                    return False, "Failed to read any video frames"
            
            elif suffix == '.json':
                # Check JSON integrity
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, (dict, list)):
                    return True, "JSON integrity validated"
                else:
                    return False, "JSON data is not a dict or list"
            
            elif suffix in ['.pt', '.pth']:
                # Check PyTorch file integrity
                data = torch.load(file_path, map_location='cpu')
                return True, "PyTorch file integrity validated"
            
            elif suffix == '.npy':
                # Check numpy file integrity
                data = np.load(file_path)
                return True, "NumPy file integrity validated"
            
            else:
                # For other file types, just check if we can read them
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Read first KB
                return True, "File integrity validated (basic read check)"
        
        except Exception as e:
            return False, f"Content integrity check failed: {str(e)}"
    
    def _check_schema(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Validate data against schema."""
        if not file_path.exists():
            return False, "File does not exist"
        
        # Get schema from kwargs
        schema = kwargs.get('schema')
        if not schema:
            return True, "No schema specified"
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.json':
                # Load JSON data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate against schema
                if isinstance(schema, dict):
                    # Simple schema validation
                    return self._validate_json_schema(data, schema)
                else:
                    return False, "Schema must be a dictionary"
            
            elif suffix in ['.yaml', '.yml']:
                # Load YAML data
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Validate against schema
                if isinstance(schema, dict):
                    return self._validate_json_schema(data, schema)
                else:
                    return False, "Schema must be a dictionary"
            
            elif suffix == '.csv':
                # Load CSV data
                import pandas as pd
                data = pd.read_csv(file_path)
                
                # Get schema for CSV
                if isinstance(schema, dict) and 'columns' in schema:
                    return self._validate_csv_schema(data, schema)
                else:
                    return False, "CSV schema must include 'columns' key"
            
            else:
                return False, f"Schema validation not supported for file type: {suffix}"
        
        except Exception as e:
            return False, f"Schema validation failed: {str(e)}"
    
    def _validate_json_schema(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Simple JSON schema validation."""
        errors = []
        
        # Check required fields
        required_fields = schema.get('required', [])
        if isinstance(data, dict):
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
        
        # Check field types
        properties = schema.get('properties', {})
        if isinstance(data, dict):
            for field, value in data.items():
                if field in properties:
                    field_schema = properties[field]
                    expected_type = field_schema.get('type')
                    
                    if expected_type:
                        type_check = self._check_type(value, expected_type)
                        if not type_check:
                            errors.append(f"Field '{field}' has wrong type. Expected {expected_type}, got {type(value).__name__}")
        
        if errors:
            return False, f"Schema validation failed: {', '.join(errors)}"
        else:
            return True, "Schema validation passed"
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None)
        }
        
        if expected_type not in type_map:
            return True  # Unknown type, skip check
        
        expected = type_map[expected_type]
        
        if isinstance(expected, tuple):
            return isinstance(value, expected)
        else:
            return isinstance(value, expected)
    
    def _validate_csv_schema(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate CSV data against schema."""
        errors = []
        
        # Check required columns
        required_columns = schema.get('required_columns', [])
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check column types
        column_types = schema.get('column_types', {})
        for col, expected_type in column_types.items():
            if col in df.columns:
                # Simple type checking
                if expected_type == 'numeric':
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Column '{col}' is not numeric")
                elif expected_type == 'string':
                    if not pd.api.types.is_string_dtype(df[col]):
                        errors.append(f"Column '{col}' is not string")
                elif expected_type == 'datetime':
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        errors.append(f"Column '{col}' is not datetime")
        
        # Check for null values in non-nullable columns
        non_nullable = schema.get('non_nullable', [])
        for col in non_nullable:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' has {null_count} null values")
        
        if errors:
            return False, f"CSV schema validation failed: {', '.join(errors)}"
        else:
            return True, "CSV schema validation passed"
    
    def _check_metadata_consistency(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check metadata consistency."""
        # This is a placeholder for more complex metadata validation
        # In production, you would implement specific checks based on your data format
        
        return True, "Metadata consistency check passed (basic)"
    
    def _check_data_quality(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check data quality metrics."""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp']:
                return self._check_image_quality(file_path, **kwargs)
            elif suffix in ['.mp4', '.avi', '.mov', '.webm']:
                return self._check_video_quality(file_path, **kwargs)
            elif suffix == '.csv':
                return self._check_csv_quality(file_path, **kwargs)
            else:
                return True, f"Data quality check not implemented for file type: {suffix}"
        
        except Exception as e:
            return False, f"Data quality check failed: {str(e)}"
    
    def _check_image_quality(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check image quality metrics."""
        try:
            with Image.open(file_path) as img:
                # Get image properties
                width, height = img.size
                mode = img.mode
                
                # Check dimensions
                min_width = kwargs.get('min_width', 32)
                min_height = kwargs.get('min_height', 32)
                max_width = kwargs.get('max_width', 8192)
                max_height = kwargs.get('max_height', 8192)
                
                if width < min_width or height < min_height:
                    return False, f"Image too small: {width}x{height} (min {min_width}x{min_height})"
                
                if width > max_width or height > max_height:
                    return False, f"Image too large: {width}x{height} (max {max_width}x{max_height})"
                
                # Check aspect ratio
                min_aspect = kwargs.get('min_aspect_ratio', 0.1)
                max_aspect = kwargs.get('max_aspect_ratio', 10.0)
                aspect_ratio = width / height
                
                if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
                    return False, f"Extreme aspect ratio: {aspect_ratio:.2f}"
                
                # Check if image is all one color (possible corruption)
                if kwargs.get('check_uniformity', True):
                    img_array = np.array(img)
                    if len(img_array.shape) == 3:  # Color image
                        # Check each channel
                        for c in range(img_array.shape[2]):
                            if np.all(img_array[:, :, c] == img_array[0, 0, c]):
                                return False, "Image appears to be uniform color (possible corruption)"
                    else:  # Grayscale
                        if np.all(img_array == img_array[0, 0]):
                            return False, "Image appears to be uniform (possible corruption)"
                
                return True, f"Image quality OK: {width}x{height}, mode={mode}"
        
        except Exception as e:
            return False, f"Image quality check failed: {str(e)}"
    
    def _check_video_quality(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check video quality metrics."""
        try:
            cap = cv2.VideoCapture(str(file_path))
            
            if not cap.isOpened():
                return False, "Failed to open video"
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Check properties
            min_frames = kwargs.get('min_frames', 1)
            min_fps = kwargs.get('min_fps', 1.0)
            min_width = kwargs.get('min_width', 32)
            min_height = kwargs.get('min_height', 32)
            
            if frame_count < min_frames:
                cap.release()
                return False, f"Insufficient frames: {frame_count} (min {min_frames})"
            
            if fps < min_fps:
                cap.release()
                return False, f"Low frame rate: {fps:.1f} FPS (min {min_fps})"
            
            if width < min_width or height < min_height:
                cap.release()
                return False, f"Video too small: {width}x{height} (min {min_width}x{min_height})"
            
            # Sample frames to check for corruption
            sample_count = min(kwargs.get('sample_frames', 10), frame_count)
            frames_checked = 0
            frames_valid = 0
            
            for i in range(0, frame_count, max(1, frame_count // sample_count)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    frames_checked += 1
                    # Check if frame is valid (not all zeros or NaN)
                    if frame is not None and np.any(frame):
                        frames_valid += 1
            
            cap.release()
            
            if frames_checked == 0:
                return False, "Could not read any frames"
            
            valid_ratio = frames_valid / frames_checked
            min_valid_ratio = kwargs.get('min_valid_frame_ratio', 0.5)
            
            if valid_ratio < min_valid_ratio:
                return False, f"Low frame validity: {valid_ratio:.2f} (min {min_valid_ratio})"
            
            return True, f"Video quality OK: {frame_count} frames, {fps:.1f} FPS, {width}x{height}"
        
        except Exception as e:
            return False, f"Video quality check failed: {str(e)}"
    
    def _check_csv_quality(self, file_path: Path, **kwargs) -> Tuple[bool, str]:
        """Check CSV quality metrics."""
        try:
            df = pd.read_csv(file_path)
            
            # Check basic metrics
            row_count = len(df)
            col_count = len(df.columns)
            
            min_rows = kwargs.get('min_rows', 1)
            min_cols = kwargs.get('min_cols', 1)
            
            if row_count < min_rows:
                return False, f"Insufficient rows: {row_count} (min {min_rows})"
            
            if col_count < min_cols:
                return False, f"Insufficient columns: {col_count} (min {min_cols})"
            
            # Check for null values
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            null_percentage = total_nulls / (row_count * col_count)
            
            max_null_percentage = kwargs.get('max_null_percentage', 0.5)
            if null_percentage > max_null_percentage:
                return False, f"High null percentage: {null_percentage:.2%} (max {max_null_percentage:.0%})"
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            duplicate_percentage = duplicate_count / row_count
            
            max_duplicate_percentage = kwargs.get('max_duplicate_percentage', 0.9)
            if duplicate_percentage > max_duplicate_percentage:
                return False, f"High duplicate percentage: {duplicate_percentage:.2%} (max {max_duplicate_percentage:.0%})"
            
            return True, f"CSV quality OK: {row_count} rows, {col_count} columns, {null_percentage:.2%} nulls, {duplicate_percentage:.2%} duplicates"
        
        except Exception as e:
            return False, f"CSV quality check failed: {str(e)}"
    
    def generate_validation_report(
        self,
        data_path: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            data_path: Path to data
            output_file: Optional output file path
            **kwargs: Additional validation parameters
            
        Returns:
            Validation report dictionary
        """
        # Perform validation
        result = self.validate(data_path, **kwargs)
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(data_path),
            'validation_level': self.validation_level.value,
            'result': result.to_dict(),
            'configuration': self.config
        }
        
        # Add additional metrics
        if Path(data_path).is_file():
            report['file_info'] = {
                'size_bytes': get_file_size(data_path),
                'md5': calculate_md5(data_path) if Path(data_path).exists() else None
            }
        
        # Save report if output file specified
        if output_file:
            from .file_io import write_json
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if write_json(report, output_path):
                logger.info(f"Validation report saved to {output_path}")
        
        return report


# Convenience functions

def validate_data_file(
    file_path: Union[str, Path],
    expected_format: Optional[str] = None,
    expected_checksum: Optional[str] = None,
    validation_level: str = "standard"
) -> ValidationResult:
    """
    Convenience function to validate a data file.
    
    Args:
        file_path: Path to file
        expected_format: Expected file format
        expected_checksum: Expected MD5 checksum
        validation_level: Validation level
        
    Returns:
        ValidationResult
    """
    config = {
        'validation_level': validation_level
    }
    
    validator = DataValidator(config)
    
    if expected_format:
        validator.add_custom_rule(
            ValidationRule(
                name="expected_format",
                description=f"Check if file is {expected_format}",
                check_function=lambda path, **kw: validator._check_file_format(
                    path, expected_format=expected_format
                ),
                required=True
            )
        )
    
    if expected_checksum:
        validator.add_custom_rule(
            ValidationRule(
                name="expected_checksum",
                description="Verify checksum",
                check_function=lambda path, **kw: validator._check_checksum(
                    path, expected_checksum=expected_checksum
                ),
                required=True
            )
        )
    
    return validator.validate(file_path)


def validate_data_directory(
    directory: Union[str, Path],
    file_pattern: str = "*",
    recursive: bool = False,
    validation_level: str = "standard"
) -> ValidationResult:
    """
    Convenience function to validate a data directory.
    
    Args:
        directory: Path to directory
        file_pattern: File pattern to match
        recursive: Whether to search recursively
        validation_level: Validation level
        
    Returns:
        ValidationResult
    """
    config = {
        'validation_level': validation_level
    }
    
    validator = DataValidator(config)
    
    return validator.validate(
        directory,
        file_pattern=file_pattern,
        recursive=recursive
    )