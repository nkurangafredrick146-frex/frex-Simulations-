"""
Validation utilities and custom validators for API schemas
"""

import re
import uuid
import base64
import mimetypes
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
from datetime import datetime, timedelta

from pydantic import validator, root_validator
import numpy as np

from .request_models import (
    GenerationRequest,
    EditRequest,
    RenderRequest,
    ConversionRequest,
    validate_world_id,
    validate_session_id
)


# Custom validators
def validate_resolution(value: str) -> str:
    """Validate resolution string format (WIDTHxHEIGHT)"""
    if not isinstance(value, str):
        raise ValueError("Resolution must be a string")
    
    if 'x' not in value:
        raise ValueError('Resolution must be in format "WIDTHxHEIGHT"')
    
    try:
        width, height = map(int, value.split('x'))
    except ValueError:
        raise ValueError('Resolution must contain integers')
    
    if width < 64 or height < 64:
        raise ValueError('Minimum resolution is 64x64')
    
    if width > 16384 or height > 16384:
        raise ValueError('Maximum resolution is 16384x16384')
    
    # Check aspect ratio
    aspect_ratio = width / height
    if aspect_ratio < 0.1 or aspect_ratio > 10:
        raise ValueError('Aspect ratio must be between 0.1 and 10')
    
    return value


def validate_prompt(value: str) -> str:
    """Validate prompt string"""
    if not isinstance(value, str):
        raise ValueError("Prompt must be a string")
    
    value = value.strip()
    if not value:
        raise ValueError("Prompt cannot be empty")
    
    if len(value) > 10000:
        raise ValueError("Prompt cannot exceed 10000 characters")
    
    # Check for potentially harmful content
    harmful_patterns = [
        r'<\s*script\s*>',
        r'javascript:',
        r'on\w+\s*=',
        r'<\s*iframe\s*>',
        r'<\s*object\s*>',
        r'<\s*embed\s*>'
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, value, re.IGNORECASE):
            raise ValueError("Prompt contains potentially harmful content")
    
    return value


def validate_api_key(value: str) -> str:
    """Validate API key format"""
    if not isinstance(value, str):
        raise ValueError("API key must be a string")
    
    if len(value) < 32:
        raise ValueError("API key must be at least 32 characters")
    
    # Check for invalid characters
    if not re.match(r'^[A-Za-z0-9\-_]+$', value):
        raise ValueError("API key contains invalid characters")
    
    return value


def validate_timestamp(value: Union[str, datetime]) -> datetime:
    """Validate and parse timestamp"""
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, str):
        try:
            # Try ISO format
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            try:
                # Try Unix timestamp
                return datetime.fromtimestamp(float(value))
            except ValueError:
                raise ValueError("Invalid timestamp format")
    
    raise ValueError("Timestamp must be string or datetime object")


def validate_duration(value: Union[int, float, str]) -> float:
    """Validate duration value"""
    if isinstance(value, (int, float)):
        duration = float(value)
    elif isinstance(value, str):
        try:
            duration = float(value)
        except ValueError:
            raise ValueError("Duration must be a number")
    else:
        raise ValueError("Duration must be a number")
    
    if duration <= 0:
        raise ValueError("Duration must be positive")
    
    if duration > 3600:  # 1 hour
        raise ValueError("Duration cannot exceed 1 hour")
    
    return duration


def validate_float_range(value: float, min_val: float, max_val: float, field_name: str) -> float:
    """Validate float value within range"""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    
    if value < min_val or value > max_val:
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
    
    return float(value)


def validate_integer_range(value: int, min_val: int, max_val: int, field_name: str) -> int:
    """Validate integer value within range"""
    if not isinstance(value, int):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValueError(f"{field_name} must be an integer")
    
    if value < min_val or value > max_val:
        raise ValueError(f"{field_name} must be between {min_val} and {max_val}")
    
    return value


def validate_coordinates(coords: List[List[float]], min_points: int = 2, max_points: int = 100) -> List[List[float]]:
    """Validate coordinate array"""
    if not isinstance(coords, list):
        raise ValueError("Coordinates must be a list")
    
    if len(coords) < min_points:
        raise ValueError(f"At least {min_points} points are required")
    
    if len(coords) > max_points:
        raise ValueError(f"Maximum {max_points} points allowed")
    
    for i, point in enumerate(coords):
        if not isinstance(point, list):
            raise ValueError(f"Point {i} must be a list")
        
        if len(point) not in [2, 3]:
            raise ValueError(f"Point {i} must have 2 or 3 coordinates")
        
        for j, coord in enumerate(point):
            if not isinstance(coord, (int, float)):
                raise ValueError(f"Coordinate {j} of point {i} must be a number")
            
            if abs(coord) > 1e6:  # Prevent excessively large coordinates
                raise ValueError(f"Coordinate {j} of point {i} is too large")
    
    return coords


def validate_image_data(value: str) -> Tuple[str, Optional[str]]:
    """Validate image data (URL or base64)"""
    if not isinstance(value, str):
        raise ValueError("Image data must be a string")
    
    # Check if it's a URL
    try:
        result = urlparse(value)
        if result.scheme and result.netloc:
            # Validate URL
            allowed_schemes = ['http', 'https', 'data']
            if result.scheme not in allowed_schemes:
                raise ValueError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
            
            # For data URLs, validate base64
            if result.scheme == 'data':
                if not value.startswith('data:image/'):
                    raise ValueError("Data URL must be an image")
                
                # Extract and validate base64
                header, data = value.split(',', 1)
                if 'base64' not in header:
                    raise ValueError("Data URL must be base64 encoded")
                
                try:
                    base64.b64decode(data, validate=True)
                except Exception:
                    raise ValueError("Invalid base64 data in URL")
            
            return value, 'url'
    except ValueError:
        pass
    
    # Check if it's base64
    try:
        # Try to decode as base64
        decoded = base64.b64decode(value, validate=True)
        
        # Check if it looks like an image (basic check)
        if len(decoded) < 100:  # Very small images might not have headers
            # Still accept it, but with warning
            return value, 'base64'
        
        # Check common image magic numbers
        magic_numbers = {
            b'\xff\xd8\xff': 'jpeg',
            b'\x89PNG\r\n\x1a\n': 'png',
            b'GIF87a': 'gif',
            b'GIF89a': 'gif',
            b'BM': 'bmp',
            b'II*\x00': 'tiff',
            b'MM\x00*': 'tiff',
            b'RIFF': 'webp',
        }
        
        for magic, format in magic_numbers.items():
            if decoded.startswith(magic):
                return value, 'base64'
        
        # If we get here, it's base64 but not a recognized image format
        raise ValueError("Base64 data does not appear to be a valid image")
    
    except Exception:
        pass
    
    raise ValueError("Image data must be a valid URL or base64 encoded string")


def validate_color(value: Union[str, List[float]]) -> List[float]:
    """Validate color value (hex, rgb, rgba)"""
    if isinstance(value, str):
        # Hex color
        if value.startswith('#'):
            if len(value) not in [4, 5, 7, 9]:  # #RGB, #RGBA, #RRGGBB, #RRGGBBAA
                raise ValueError("Invalid hex color format")
            
            try:
                hex_value = value[1:]
                if len(hex_value) == 3:
                    hex_value = ''.join(c*2 for c in hex_value)
                elif len(hex_value) == 4:
                    hex_value = ''.join(c*2 for c in hex_value)
                
                # Parse hex to rgb/rgba
                if len(hex_value) == 6:
                    r = int(hex_value[0:2], 16) / 255.0
                    g = int(hex_value[2:4], 16) / 255.0
                    b = int(hex_value[4:6], 16) / 255.0
                    return [r, g, b, 1.0]
                elif len(hex_value) == 8:
                    r = int(hex_value[0:2], 16) / 255.0
                    g = int(hex_value[2:4], 16) / 255.0
                    b = int(hex_value[4:6], 16) / 255.0
                    a = int(hex_value[6:8], 16) / 255.0
                    return [r, g, b, a]
            except ValueError:
                raise ValueError("Invalid hex color value")
        
        # RGB/RGBA string
        elif value.startswith('rgb') or value.startswith('rgba'):
            import re
            pattern = r'rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)'
            match = re.match(pattern, value)
            
            if not match:
                raise ValueError("Invalid RGB/RGBA format")
            
            r = int(match.group(1)) / 255.0
            g = int(match.group(2)) / 255.0
            b = int(match.group(3)) / 255.0
            a = float(match.group(4)) if match.group(4) else 1.0
            
            return [r, g, b, a]
        
        else:
            raise ValueError("Invalid color format. Use hex (#RRGGBB) or rgb/rgba()")
    
    elif isinstance(value, list):
        if len(value) not in [3, 4]:
            raise ValueError("Color list must have 3 (RGB) or 4 (RGBA) values")
        
        for i, component in enumerate(value):
            if not isinstance(component, (int, float)):
                raise ValueError(f"Color component {i} must be a number")
            
            if i < 3:  # RGB components
                if component < 0 or component > 1:
                    raise ValueError(f"RGB component {i} must be between 0 and 1")
            else:  # Alpha component
                if component < 0 or component > 1:
                    raise ValueError("Alpha component must be between 0 and 1")
        
        # Ensure 4 components (add alpha if missing)
        if len(value) == 3:
            return value + [1.0]
        return value
    
    else:
        raise ValueError("Color must be string or list")


# Composite validators
def validate_generation_request(request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate generation request comprehensively"""
    errors = []
    
    try:
        # Validate prompt
        if 'prompt' not in request:
            errors.append("Prompt is required")
        else:
            try:
                validate_prompt(request['prompt'])
            except ValueError as e:
                errors.append(f"Invalid prompt: {e}")
        
        # Validate resolution if provided
        if 'parameters' in request and 'resolution' in request['parameters']:
            try:
                validate_resolution(request['parameters']['resolution'])
            except ValueError as e:
                errors.append(f"Invalid resolution: {e}")
        
        # Validate steps
        if 'parameters' in request and 'steps' in request['parameters']:
            try:
                validate_integer_range(
                    request['parameters']['steps'],
                    1, 500, "Steps"
                )
            except ValueError as e:
                errors.append(f"Invalid steps: {e}")
        
        # Validate guidance scale
        if 'parameters' in request and 'guidance_scale' in request['parameters']:
            try:
                validate_float_range(
                    request['parameters']['guidance_scale'],
                    1.0, 20.0, "Guidance scale"
                )
            except ValueError as e:
                errors.append(f"Invalid guidance scale: {e}")
        
        # Validate reference image
        if 'reference_image' in request and request['reference_image']:
            try:
                validate_image_data(request['reference_image'])
            except ValueError as e:
                errors.append(f"Invalid reference image: {e}")
    
    except Exception as e:
        errors.append(f"Validation error: {e}")
    
    return len(errors) == 0, errors


def validate_edit_request(request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate edit request comprehensively"""
    errors = []
    
    try:
        # Validate world ID
        if 'world_id' not in request:
            errors.append("World ID is required")
        elif not validate_world_id(request['world_id']):
            errors.append("Invalid world ID format")
        
        # Validate operations
        if 'operations' not in request or not request['operations']:
            errors.append("At least one edit operation is required")
        else:
            if len(request['operations']) > 20:
                errors.append("Maximum 20 operations per edit")
            
            for i, op in enumerate(request['operations']):
                # Validate operation type
                if 'type' not in op:
                    errors.append(f"Operation {i}: Type is required")
                
                # Validate prompt for operations that need it
                if op.get('type') in ['text_prompt', 'object_addition']:
                    if 'prompt' not in op or not op['prompt']:
                        errors.append(f"Operation {i}: Prompt is required for {op['type']}")
                    else:
                        try:
                            validate_prompt(op['prompt'])
                        except ValueError as e:
                            errors.append(f"Operation {i}: Invalid prompt: {e}")
        
        # Validate region if provided
        if 'region' in request and request['region']:
            region = request['region']
            if 'coordinates' not in region:
                errors.append("Region coordinates are required")
            else:
                try:
                    validate_coordinates(region['coordinates'])
                except ValueError as e:
                    errors.append(f"Invalid region coordinates: {e}")
    
    except Exception as e:
        errors.append(f"Validation error: {e}")
    
    return len(errors) == 0, errors


def validate_render_request(request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate render request comprehensively"""
    errors = []
    
    try:
        # Validate world ID
        if 'world_id' not in request:
            errors.append("World ID is required")
        elif not validate_world_id(request['world_id']):
            errors.append("Invalid world ID format")
        
        # Validate render type
        valid_render_types = ['image', 'panorama', 'video', 'interactive', 'point_cloud']
        if 'render_type' not in request:
            errors.append("Render type is required")
        elif request['render_type'] not in valid_render_types:
            errors.append(f"Invalid render type. Must be one of: {', '.join(valid_render_types)}")
        
        # Validate resolution
        if 'resolution' in request:
            try:
                validate_resolution(request['resolution'])
            except ValueError as e:
                errors.append(f"Invalid resolution: {e}")
        
        # Validate output format
        if 'output_format' not in request:
            errors.append("Output format is required")
        else:
            # Validate format based on render type
            format = request['output_format'].lower()
            render_type = request.get('render_type')
            
            format_validation = {
                'image': ['png', 'jpg', 'jpeg', 'webp', 'exr', 'hdr'],
                'panorama': ['png', 'jpg', 'jpeg', 'webp', 'exr', 'hdr'],
                'video': ['mp4', 'webm', 'gif', 'mov'],
                'interactive': ['html', 'gltf', 'glb'],
                'point_cloud': ['ply', 'xyz', 'pcd']
            }
            
            if render_type in format_validation and format not in format_validation[render_type]:
                errors.append(
                    f"Invalid format for {render_type}. "
                    f"Must be one of: {', '.join(format_validation[render_type])}"
                )
        
        # Validate camera if provided
        if 'camera' in request and request['camera']:
            camera = request['camera']
            
            # Validate position
            if 'position' in camera:
                try:
                    validate_coordinates([camera['position']], 1, 1)
                except ValueError as e:
                    errors.append(f"Invalid camera position: {e}")
            
            # Validate FOV
            if 'fov' in camera:
                try:
                    validate_float_range(camera['fov'], 1.0, 180.0, "FOV")
                except ValueError as e:
                    errors.append(f"Invalid FOV: {e}")
        
        # Validate parameters
        if 'parameters' in request and request['parameters']:
            params = request['parameters']
            
            # Validate FPS for video
            if request.get('render_type') == 'video' and 'fps' in params:
                try:
                    validate_integer_range(params['fps'], 1, 120, "FPS")
                except ValueError as e:
                    errors.append(f"Invalid FPS: {e}")
            
            # Validate duration for video
            if request.get('render_type') == 'video' and 'duration' in params:
                try:
                    validate_duration(params['duration'])
                except ValueError as e:
                    errors.append(f"Invalid duration: {e}")
    
    except Exception as e:
        errors.append(f"Validation error: {e}")
    
    return len(errors) == 0, errors


def validate_conversion_request(request: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate conversion request comprehensively"""
    errors = []
    
    try:
        # Validate world ID
        if 'world_id' not in request:
            errors.append("World ID is required")
        elif not validate_world_id(request['world_id']):
            errors.append("Invalid world ID format")
        
        # Validate target format
        valid_formats = ['nerf', 'gaussian', 'mesh', 'voxel', 'point_cloud', 'gltf', 'usd', 'fbx']
        if 'target_format' not in request:
            errors.append("Target format is required")
        elif request['target_format'] not in valid_formats:
            errors.append(f"Invalid target format. Must be one of: {', '.join(valid_formats)}")
    
    except Exception as e:
        errors.append(f"Validation error: {e}")
    
    return len(errors) == 0, errors


# Schema validators (to be used as decorators)
def validate_generation_parameters(cls, values):
    """Pydantic validator for generation parameters"""
    if 'parameters' in values and values['parameters']:
        params = values['parameters']
        
        # Validate resolution
        if hasattr(params, 'resolution'):
            try:
                params.resolution = validate_resolution(params.resolution)
            except ValueError as e:
                raise ValueError(f"Invalid resolution: {e}")
        
        # Validate steps
        if hasattr(params, 'steps'):
            try:
                params.steps = validate_integer_range(params.steps, 1, 500, "Steps")
            except ValueError as e:
                raise ValueError(f"Invalid steps: {e}")
        
        # Validate guidance scale
        if hasattr(params, 'guidance_scale'):
            try:
                params.guidance_scale = validate_float_range(
                    params.guidance_scale, 1.0, 20.0, "Guidance scale"
                )
            except ValueError as e:
                raise ValueError(f"Invalid guidance scale: {e}")
    
    return values


def validate_camera_parameters(cls, values):
    """Pydantic validator for camera parameters"""
    # Validate position
    if 'position' in values:
        try:
            validate_coordinates([values['position']], 1, 1)
        except ValueError as e:
            raise ValueError(f"Invalid camera position: {e}")
    
    # Validate FOV
    if 'fov' in values:
        try:
            values['fov'] = validate_float_range(values['fov'], 1.0, 180.0, "FOV")
        except ValueError as e:
            raise ValueError(f"Invalid FOV: {e}")
    
    # Validate near/far planes
    if 'near' in values and values['near'] <= 0:
        raise ValueError("Near plane must be positive")
    
    if 'far' in values and values['far'] <= 0:
        raise ValueError("Far plane must be positive")
    
    if 'near' in values and 'far' in values and values['near'] >= values['far']:
        raise ValueError("Near plane must be less than far plane")
    
    return values


def validate_region_selection(cls, values):
    """Pydantic validator for region selection"""
    if 'coordinates' in values:
        try:
            values['coordinates'] = validate_coordinates(values['coordinates'])
        except ValueError as e:
            raise ValueError(f"Invalid region coordinates: {e}")
    
    # Validate confidence
    if 'confidence' in values:
        try:
            values['confidence'] = validate_float_range(
                values['confidence'], 0.0, 1.0, "Confidence"
            )
        except ValueError as e:
            raise ValueError(f"Invalid confidence: {e}")
    
    return values


# Utility functions for validation
def sanitize_input(data: Any) -> Any:
    """Sanitize input data to prevent injection attacks"""
    if isinstance(data, str):
        # Remove potentially dangerous characters/patterns
        dangerous_patterns = [
            (r'<\s*script\s*>', ''),
            (r'javascript:', ''),
            (r'on\w+\s*=', ''),
            (r'<\s*iframe\s*>', ''),
            (r'<\s*object\s*>', ''),
            (r'<\s*embed\s*>', ''),
            (r'eval\s*\(', ''),
            (r'alert\s*\(', ''),
            (r'document\.', ''),
            (r'window\.', ''),
            (r'\.\./', ''),
            (r'\.\.\\', ''),
        ]
        
        sanitized = data
        for pattern, replacement in dangerous_patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        # Limit length
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized
    
    elif isinstance(data, dict):
        return {k: sanitize_input(v) for k, v in data.items()}
    
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    
    else:
        return data


def validate_file_upload(
    filename: str,
    content_type: str,
    max_size: int = 100 * 1024 * 1024  # 100MB
) -> Tuple[bool, Optional[str]]:
    """Validate file upload"""
    # Check file extension
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    file_ext = filename.lower()[-4:]
    
    if file_ext not in allowed_extensions and filename.lower()[-5:] not in allowed_extensions:
        return False, f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
    
    # Check content type
    allowed_mime_types = [
        'image/png', 'image/jpeg', 'image/gif', 'image/bmp',
        'image/tiff', 'image/webp'
    ]
    
    if content_type not in allowed_mime_types:
        return False, f"Invalid content type. Allowed: {', '.join(allowed_mime_types)}"
    
    # File would be checked for size when uploaded
    # This is just for metadata validation
    
    return True, None


def validate_numeric_array(
    data: List[Any],
    expected_length: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """Validate numeric array"""
    if not isinstance(data, list):
        return False, "Data must be a list"
    
    if expected_length is not None and len(data) != expected_length:
        return False, f"Array must have {expected_length} elements"
    
    for i, item in enumerate(data):
        if not isinstance(item, (int, float)):
            return False, f"Element {i} must be a number"
        
        if min_value is not None and item < min_value:
            return False, f"Element {i} must be >= {min_value}"
        
        if max_value is not None and item > max_value:
            return False, f"Element {i} must be <= {max_value}"
    
    return True, None


def validate_bbox(bbox: List[float]) -> Tuple[bool, Optional[str]]:
    """Validate bounding box [x_min, y_min, z_min, x_max, y_max, z_max]"""
    valid, msg = validate_numeric_array(bbox, expected_length=6)
    if not valid:
        return False, msg
    
    # Check that min <= max for each dimension
    for i in range(3):
        if bbox[i] > bbox[i + 3]:
            return False, f"Min coordinate {i} must be <= max coordinate"
    
    return True, None


def validate_transform_matrix(matrix: List[List[float]]) -> Tuple[bool, Optional[str]]:
    """Validate transformation matrix (4x4)"""
    if not isinstance(matrix, list) or len(matrix) != 4:
        return False, "Matrix must be 4x4"
    
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != 4:
            return False, f"Row {i} must have 4 elements"
        
        for j, value in enumerate(row):
            if not isinstance(value, (int, float)):
                return False, f"Element ({i},{j}) must be a number"
    
    return True, None


# Validation result class
class ValidationResult:
    """Result of validation operation"""
    
    def __init__(self, valid: bool = True, errors: List[str] = None, warnings: List[str] = None):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, error: str):
        """Add error to result"""
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str):
        """Add warning to result"""
        self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings
        }
    
    def __bool__(self):
        return self.valid
    
    def __str__(self):
        result = "VALID" if self.valid else "INVALID"
        if self.errors:
            result += f"\nErrors: {', '.join(self.errors)}"
        if self.warnings:
            result += f"\nWarnings: {', '.join(self.warnings)}"
        return result


# Validation registry for custom validators
VALIDATORS_REGISTRY = {
    'resolution': validate_resolution,
    'prompt': validate_prompt,
    'api_key': validate_api_key,
    'timestamp': validate_timestamp,
    'duration': validate_duration,
    'coordinates': validate_coordinates,
    'image_data': validate_image_data,
    'color': validate_color,
    'generation_request': validate_generation_request,
    'edit_request': validate_edit_request,
    'render_request': validate_render_request,
    'conversion_request': validate_conversion_request,
    'file_upload': validate_file_upload,
    'numeric_array': validate_numeric_array,
    'bbox': validate_bbox,
    'transform_matrix': validate_transform_matrix,
}


def get_validator(name: str):
    """Get validator function by name"""
    return VALIDATORS_REGISTRY.get(name)


# Export all validators and utilities
__all__ = [
    # Validator functions
    'validate_resolution',
    'validate_prompt',
    'validate_api_key',
    'validate_timestamp',
    'validate_duration',
    'validate_coordinates',
    'validate_image_data',
    'validate_color',
    
    # Composite validators
    'validate_generation_request',
    'validate_edit_request',
    'validate_render_request',
    'validate_conversion_request',
    
    # Pydantic validators
    'validate_generation_parameters',
    'validate_camera_parameters',
    'validate_region_selection',
    
    # Utility functions
    'sanitize_input',
    'validate_file_upload',
    'validate_numeric_array',
    'validate_bbox',
    'validate_transform_matrix',
    
    # Validation result class
    'ValidationResult',
    
    # Validator registry
    'VALIDATORS_REGISTRY',
    'get_validator'
]