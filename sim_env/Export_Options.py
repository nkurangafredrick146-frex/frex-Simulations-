"""
Complete Enhanced Export Options Module
Advanced export system for simulations including video, images, 3D models, and data formats
"""

import pygame
import numpy as np
import glm
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import imageio
import cv2
import json
import pickle
import h5py
import csv
from PIL import Image
import subprocess
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import tempfile
from pathlib import Path

class ExportFormat(Enum):
    """Supported export formats"""
    PNG = "png"
    JPEG = "jpeg"
    BMP = "bmp"
    TIFF = "tiff"
    MP4 = "mp4"
    AVI = "avi"
    GIF = "gif"
    OBJ = "obj"
    PLY = "ply"
    STL = "stl"
    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    PICKLE = "pickle"

@dataclass
class ExportConfig:
    """Configuration for export operations"""
    format: ExportFormat
    quality: int = 95
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 60
    duration: float = 10.0
    codec: str = "libx264"
    compression: str = "none"
    include_metadata: bool = True
    include_timestamps: bool = True
    parallel_export: bool = True
    
class ExportMetadata:
    """Metadata for exported simulations"""
    
    def __init__(self):
        self.simulation_type = ""
        self.particle_count = 0
        self.duration = 0.0
        self.timestamp = time.time()
        self.frame_count = 0
        self.physics_parameters = {}
        self.rendering_settings = {}
        self.custom_data = {}
        
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        return {
            'simulation_type': self.simulation_type,
            'particle_count': self.particle_count,
            'duration': self.duration,
            'timestamp': self.timestamp,
            'frame_count': self.frame_count,
            'physics_parameters': self.physics_parameters,
            'rendering_settings': self.rendering_settings,
            'custom_data': self.custom_data,
            'export_version': '1.0'
        }

class FrameBuffer:
    """High-quality frame buffer for capturing frames"""
    
    def __init__(self, width: int, height: int, multisample: bool = True):
        self.width = width
        self.height = height
        self.multisample = multisample
        
        # Create framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        if multisample:
            # Multisampled color texture
            self.color_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, self.color_texture)
            glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGB, width, height, GL_TRUE)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, self.color_texture, 0)
            
            # Multisampled depth buffer
            self.depth_buffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, width, height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer)
        else:
            # Regular color texture
            self.color_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.color_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.color_texture, 0)
            
            # Regular depth buffer
            self.depth_buffer = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer)
        
        # Check framebuffer completeness
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer not complete!")
            
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Resolve framebuffer for reading
        self.resolve_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.resolve_fbo)
        
        self.resolve_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.resolve_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.resolve_texture, 0)
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
    def bind(self):
        """Bind framebuffer for rendering"""
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)
        
    def unbind(self):
        """Unbind framebuffer"""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
    def capture_frame(self) -> np.ndarray:
        """Capture current frame as numpy array"""
        if self.multisample:
            # Resolve multisampling
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.resolve_fbo)
            glBlitFramebuffer(0, 0, self.width, self.height, 0, 0, self.width, self.height, 
                             GL_COLOR_BUFFER_BIT, GL_LINEAR)
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.resolve_fbo)
        else:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.fbo)
            
        # Read pixels
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        pixels = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and flip vertically
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 3)
        image = np.flipud(image)
        
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)
        
        return image
        
    def cleanup(self):
        """Clean up OpenGL resources"""
        glDeleteFramebuffers(1, [self.fbo])
        glDeleteFramebuffers(1, [self.resolve_fbo])
        glDeleteTextures(1, [self.color_texture])
        glDeleteTextures(1, [self.resolve_texture])
        glDeleteRenderbuffers(1, [self.depth_buffer])

class VideoEncoder:
    """High-quality video encoding with multiple codec support"""
    
    def __init__(self, output_path: str, resolution: Tuple[int, int], fps: int = 60, 
                 codec: str = "libx264", quality: int = 95):
        self.output_path = output_path
        self.resolution = resolution
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.writer = None
        self.frame_count = 0
        
        # Video writer parameters based on codec
        self.codec_params = self.get_codec_parameters()
        
    def get_codec_parameters(self) -> Dict:
        """Get encoding parameters for specific codec"""
        if self.codec == "libx264":
            return {
                'codec': 'libx264',
                'pixel_format': 'yuv420p',
                'crf': str(51 - int(self.quality * 0.5)),  # Convert quality to CRF
                'preset': 'medium'
            }
        elif self.codec == "libx265":
            return {
                'codec': 'libx265',
                'pixel_format': 'yuv420p',
                'crf': str(51 - int(self.quality * 0.5)),
                'preset': 'medium'
            }
        elif self.codec == "prores":
            return {
                'codec': 'prores',
                'pixel_format': 'yuv422p10le',
                'profile': 3
            }
        else:  # Default to MPEG-4
            return {
                'codec': 'mpeg4',
                'pixel_format': 'yuv420p',
                'qscale': str(31 - int(self.quality * 0.3))
            }
            
    def start_encoding(self):
        """Start video encoding"""
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.codec_params['codec'])
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                self.resolution
            )
            
            if not self.writer.isOpened():
                raise Exception(f"Failed to open video writer for {self.output_path}")
                
        except Exception as e:
            print(f"Video encoding error: {e}")
            self.writer = None
            
    def add_frame(self, frame: np.ndarray):
        """Add frame to video"""
        if self.writer is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.writer.write(frame_bgr)
            self.frame_count += 1
            
    def finish_encoding(self):
        """Finish video encoding and release resources"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            
    def get_progress(self) -> float:
        """Get encoding progress"""
        # This would be more meaningful with a target frame count
        return min(self.frame_count / (self.fps * 10), 1.0)  # Estimate based on 10s default

class ImageSequenceExporter:
    """Export image sequences with various formats and quality settings"""
    
    def __init__(self, output_dir: str, format: ExportFormat, quality: int = 95):
        self.output_dir = Path(output_dir)
        self.format = format
        self.quality = quality
        self.frame_count = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_frame(self, frame: np.ndarray, frame_number: int = None):
        """Export single frame"""
        if frame_number is None:
            frame_number = self.frame_count
            
        filename = f"frame_{frame_number:06d}.{self.format.value}"
        filepath = self.output_dir / filename
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Save with quality settings
        if self.format == ExportFormat.JPEG:
            pil_image.save(filepath, quality=self.quality, optimize=True)
        elif self.format == ExportFormat.PNG:
            pil_image.save(filepath, optimize=True)
        else:
            pil_image.save(filepath)
            
        self.frame_count += 1
        
    def export_metadata(self, metadata: ExportMetadata):
        """Export metadata file"""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

class ThreeDModelExporter:
    """Export 3D models of simulation state"""
    
    def __init__(self):
        self.supported_formats = [ExportFormat.OBJ, ExportFormat.PLY, ExportFormat.STL]
        
    def export_simulation_state(self, simulation, filepath: str, format: ExportFormat):
        """Export current simulation state as 3D model"""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported 3D format: {format}")
            
        if format == ExportFormat.OBJ:
            self.export_obj(simulation, filepath)
        elif format == ExportFormat.PLY:
            self.export_ply(simulation, filepath)
        elif format == ExportFormat.STL:
            self.export_stl(simulation, filepath)
            
    def export_obj(self, simulation, filepath: str):
        """Export as Wavefront OBJ format"""
        particles = getattr(simulation, 'particles', [])
        
        with open(filepath, 'w') as f:
            # Write header
            f.write("# Simulation Export - OBJ Format\n")
            f.write(f"# Particle Count: {len(particles)}\n")
            f.write(f"# Export Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write vertices (particle positions)
            for i, particle in enumerate(particles):
                pos = particle.position
                f.write(f"v {pos.x} {pos.y} {pos.z}\n")
                
            # Write vertex colors (if available)
            for i, particle in enumerate(particles):
                if hasattr(particle, 'color'):
                    color = particle.color
                    f.write(f"vc {color.x} {color.y} {color.z}\n")
                else:
                    f.write(f"vc 1.0 1.0 1.0\n")
                    
            # Write points
            f.write("\n")
            for i in range(len(particles)):
                f.write(f"p {i+1}\n")
                
    def export_ply(self, simulation, filepath: str):
        """Export as Stanford PLY format"""
        particles = getattr(simulation, 'particles', [])
        
        with open(filepath, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(particles)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("element face 0\n")
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")
            
            # Write vertices with colors
            for particle in particles:
                pos = particle.position
                if hasattr(particle, 'color'):
                    color = particle.color
                    r = int(color.x * 255)
                    g = int(color.y * 255)
                    b = int(color.z * 255)
                else:
                    r = g = b = 255
                    
                f.write(f"{pos.x} {pos.y} {pos.z} {r} {g} {b}\n")
                
    def export_stl(self, simulation, filepath: str):
        """Export as STL format (simplified - would normally include triangles)"""
        # STL requires triangular mesh data
        # For particles, we'll create simple spheres or just export positions
        particles = getattr(simulation, 'particles', [])
        
        with open(filepath, 'w') as f:
            f.write("solid simulation_export\n")
            
            # Create simple cube for each particle (simplified)
            for particle in particles:
                pos = particle.position
                size = getattr(particle, 'radius', 0.1)
                
                # Create a simple cube around each particle
                self.write_stl_cube(f, pos, size)
                
            f.write("endsolid simulation_export\n")
            
    def write_stl_cube(self, file, position: glm.vec3, size: float):
        """Write a simple cube to STL file"""
        # This is a simplified cube representation
        # A real implementation would include proper triangle data
        half_size = size / 2
        
        # Define cube vertices relative to particle position
        vertices = [
            glm.vec3(-half_size, -half_size, -half_size) + position,
            glm.vec3(half_size, -half_size, -half_size) + position,
            glm.vec3(half_size, half_size, -half_size) + position,
            glm.vec3(-half_size, half_size, -half_size) + position,
            glm.vec3(-half_size, -half_size, half_size) + position,
            glm.vec3(half_size, -half_size, half_size) + position,
            glm.vec3(half_size, half_size, half_size) + position,
            glm.vec3(-half_size, half_size, half_size) + position
        ]
        
        # Define cube faces (triangles)
        faces = [
            [0, 1, 2], [0, 2, 3],  # front
            [4, 5, 6], [4, 6, 7],  # back
            [0, 4, 7], [0, 7, 3],  # left
            [1, 5, 6], [1, 6, 2],  # right
            [3, 2, 6], [3, 6, 7],  # top
            [0, 1, 5], [0, 5, 4]   # bottom
        ]
        
        # Write faces to STL
        for face in faces:
            # Calculate face normal (simplified)
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            normal = glm.normalize(glm.cross(v1 - v0, v2 - v0))
            
            file.write(f"facet normal {normal.x} {normal.y} {normal.z}\n")
            file.write("  outer loop\n")
            for vertex_idx in face:
                v = vertices[vertex_idx]
                file.write(f"    vertex {v.x} {v.y} {v.z}\n")
            file.write("  endloop\n")
            file.write("endfacet\n")

class DataExporter:
    """Export simulation data in various formats"""
    
    def __init__(self):
        self.supported_formats = [ExportFormat.CSV, ExportFormat.JSON, ExportFormat.HDF5, ExportFormat.PICKLE]
        
    def export_simulation_data(self, simulation, filepath: str, format: ExportFormat):
        """Export simulation data"""
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported data format: {format}")
            
        data = self.collect_simulation_data(simulation)
        
        if format == ExportFormat.CSV:
            self.export_csv(data, filepath)
        elif format == ExportFormat.JSON:
            self.export_json(data, filepath)
        elif format == ExportFormat.HDF5:
            self.export_hdf5(data, filepath)
        elif format == ExportFormat.PICKLE:
            self.export_pickle(data, filepath)
            
    def collect_simulation_data(self, simulation) -> Dict:
        """Collect comprehensive simulation data"""
        particles = getattr(simulation, 'particles', [])
        physics_module = getattr(simulation, 'physics_module', None)
        
        data = {
            'metadata': {
                'export_time': time.time(),
                'particle_count': len(particles),
                'simulation_time': getattr(simulation, 'simulation_time', 0.0)
            },
            'particles': [],
            'physics_parameters': {}
        }
        
        # Collect particle data
        for i, particle in enumerate(particles):
            particle_data = {
                'id': i,
                'position': [particle.position.x, particle.position.y, particle.position.z],
                'velocity': [particle.velocity.x, particle.velocity.y, particle.velocity.z],
                'mass': getattr(particle, 'mass', 1.0),
                'radius': getattr(particle, 'radius', 0.1)
            }
            
            if hasattr(particle, 'color'):
                particle_data['color'] = [particle.color.x, particle.color.y, particle.color.z]
                
            if hasattr(particle, 'density'):
                particle_data['density'] = particle.density
                
            if hasattr(particle, 'pressure'):
                particle_data['pressure'] = particle.pressure
                
            data['particles'].append(particle_data)
            
        # Collect physics parameters
        if physics_module:
            data['physics_parameters'] = {
                'gravity': [physics_module.settings.gravity.x, 
                           physics_module.settings.gravity.y, 
                           physics_module.settings.gravity.z],
                'time_scale': physics_module.settings.time_scale,
                'viscosity': getattr(physics_module.settings, 'viscosity', 0.0)
            }
            
        return data
        
    def export_csv(self, data: Dict, filepath: str):
        """Export data as CSV"""
        particles = data['particles']
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['id', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 
                           'mass', 'radius', 'density', 'pressure'])
            
            # Write particle data
            for particle in particles:
                writer.writerow([
                    particle['id'],
                    particle['position'][0], particle['position'][1], particle['position'][2],
                    particle['velocity'][0], particle['velocity'][1], particle['velocity'][2],
                    particle.get('mass', 1.0),
                    particle.get('radius', 0.1),
                    particle.get('density', 0.0),
                    particle.get('pressure', 0.0)
                ])
                
        # Export metadata as separate file
        metadata_path = filepath.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(data['metadata'], f, indent=2)
            
    def export_json(self, data: Dict, filepath: str):
        """Export data as JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=self.json_serializer)
            
    def export_hdf5(self, data: Dict, filepath: str):
        """Export data as HDF5"""
        with h5py.File(filepath, 'w') as f:
            # Create groups
            metadata_group = f.create_group('metadata')
            particles_group = f.create_group('particles')
            physics_group = f.create_group('physics_parameters')
            
            # Store metadata
            for key, value in data['metadata'].items():
                metadata_group.attrs[key] = value
                
            # Store particle data as datasets
            if data['particles']:
                positions = np.array([p['position'] for p in data['particles']])
                velocities = np.array([p['velocity'] for p in data['particles']])
                masses = np.array([p.get('mass', 1.0) for p in data['particles']])
                
                particles_group.create_dataset('positions', data=positions)
                particles_group.create_dataset('velocities', data=velocities)
                particles_group.create_dataset('masses', data=masses)
                
            # Store physics parameters
            for key, value in data['physics_parameters'].items():
                physics_group.attrs[key] = value
                
    def export_pickle(self, data: Dict, filepath: str):
        """Export data as pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

class ExportManager:
    """Main export manager coordinating all export operations"""
    
    def __init__(self, application):
        self.application = application
        self.is_exporting = False
        self.export_queue = Queue()
        self.export_thread = None
        self.current_export = None
        
        # Export components
        self.frame_buffer = None
        self.video_encoder = None
        self.image_exporter = None
        self.model_exporter = ThreeDModelExporter()
        self.data_exporter = DataExporter()
        
        # Export state
        self.export_start_time = 0
        self.exported_frames = 0
        self.total_frames = 0
        self.export_progress = 0.0
        
        # Temporary directory for export operations
        self.temp_dir = tempfile.mkdtemp(prefix="simulation_export_")
        
    def initialize_export(self, config: ExportConfig):
        """Initialize export operation"""
        if self.is_exporting:
            raise Exception("Export already in progress")
            
        self.is_exporting = True
        self.export_start_time = time.time()
        self.exported_frames = 0
        self.total_frames = int(config.duration * config.fps)
        self.export_progress = 0.0
        
        # Create frame buffer for high-quality capture
        self.frame_buffer = FrameBuffer(config.resolution[0], config.resolution[1], multisample=True)
        
        # Initialize appropriate exporter based on format
        if config.format in [ExportFormat.MP4, ExportFormat.AVI]:
            output_path = self.get_export_path(config.format)
            self.video_encoder = VideoEncoder(
                output_path, config.resolution, config.fps, config.codec, config.quality
            )
            self.video_encoder.start_encoding()
            
        elif config.format in [ExportFormat.PNG, ExportFormat.JPEG, ExportFormat.BMP, ExportFormat.TIFF]:
            output_dir = self.get_export_directory()
            self.image_exporter = ImageSequenceExporter(output_dir, config.format, config.quality)
            
        print(f"Export initialized: {config.format.value} at {config.resolution}")
        
    def capture_frame(self):
        """Capture current frame for export"""
        if not self.is_exporting or not self.frame_buffer:
            return
            
        try:
            # Bind frame buffer and render scene
            self.frame_buffer.bind()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Render simulation (this would call the application's render method)
            if self.application and hasattr(self.application, 'render'):
                self.application.render()
                
            # Capture frame
            frame_data = self.frame_buffer.capture_frame()
            self.frame_buffer.unbind()
            
            # Add frame to export
            self.add_frame_to_export(frame_data)
            self.exported_frames += 1
            self.export_progress = self.exported_frames / self.total_frames
            
        except Exception as e:
            print(f"Frame capture error: {e}")
            
    def add_frame_to_export(self, frame: np.ndarray):
        """Add captured frame to current export"""
        if self.video_encoder:
            self.video_encoder.add_frame(frame)
        elif self.image_exporter:
            self.image_exporter.export_frame(frame, self.exported_frames)
            
    def finalize_export(self, metadata: ExportMetadata = None):
        """Finalize export operation"""
        if not self.is_exporting:
            return
            
        try:
            # Finish video encoding if active
            if self.video_encoder:
                self.video_encoder.finish_encoding()
                
            # Export metadata if provided
            if metadata and self.image_exporter:
                self.image_exporter.export_metadata(metadata)
                
            # Clean up resources
            if self.frame_buffer:
                self.frame_buffer.cleanup()
                self.frame_buffer = None
                
            export_time = time.time() - self.export_start_time
            print(f"Export completed: {self.exported_frames} frames in {export_time:.2f}s")
            
        except Exception as e:
            print(f"Export finalization error: {e}")
            
        finally:
            self.is_exporting = False
            self.export_progress = 1.0
            self.video_encoder = None
            self.image_exporter = None
            
    def export_snapshot(self, filepath: str, format: ExportFormat = ExportFormat.PNG, 
                       quality: int = 95):
        """Export single snapshot of current simulation state"""
        try:
            # Create temporary frame buffer
            width, height = self.application.config["window_width"], self.application.config["window_height"]
            temp_buffer = FrameBuffer(width, height, multisample=False)
            
            # Capture frame
            temp_buffer.bind()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.application.render()
            frame_data = temp_buffer.capture_frame()
            temp_buffer.unbind()
            temp_buffer.cleanup()
            
            # Save image
            pil_image = Image.fromarray(frame_data)
            
            if format == ExportFormat.JPEG:
                pil_image.save(filepath, quality=quality, optimize=True)
            else:
                pil_image.save(filepath)
                
            print(f"Snapshot exported: {filepath}")
            
        except Exception as e:
            print(f"Snapshot export error: {e}")
            
    def export_3d_model(self, filepath: str, format: ExportFormat):
        """Export current simulation state as 3D model"""
        try:
            if self.application and hasattr(self.application, 'current_simulation'):
                self.model_exporter.export_simulation_state(
                    self.application.current_simulation, filepath, format
                )
                print(f"3D model exported: {filepath}")
                
        except Exception as e:
            print(f"3D model export error: {e}")
            
    def export_simulation_data(self, filepath: str, format: ExportFormat):
        """Export simulation data"""
        try:
            if self.application and hasattr(self.application, 'current_simulation'):
                self.data_exporter.export_simulation_data(
                    self.application.current_simulation, filepath, format
                )
                print(f"Simulation data exported: {filepath}")
                
        except Exception as e:
            print(f"Data export error: {e}")
            
    def get_export_path(self, format: ExportFormat, prefix: str = "simulation") -> str:
        """Generate export file path"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.{format.value}"
        return os.path.join(self.temp_dir, filename)
        
    def get_export_directory(self, prefix: str = "simulation") -> str:
        """Generate export directory path"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dirname = f"{prefix}_{timestamp}"
        return os.path.join(self.temp_dir, dirname)
        
    def get_export_progress(self) -> Dict:
        """Get current export progress"""
        return {
            'is_exporting': self.is_exporting,
            'progress': self.export_progress,
            'exported_frames': self.exported_frames,
            'total_frames': self.total_frames,
            'elapsed_time': time.time() - self.export_start_time if self.is_exporting else 0
        }
        
    def create_metadata(self) -> ExportMetadata:
        """Create metadata for current simulation"""
        metadata = ExportMetadata()
        
        if self.application:
            sim = getattr(self.application, 'current_simulation', None)
            if sim:
                metadata.simulation_type = getattr(sim, 'config', type(sim).__name__)
                metadata.particle_count = len(getattr(sim, 'particles', []))
                metadata.duration = getattr(sim, 'simulation_time', 0.0)
                metadata.frame_count = getattr(sim, 'frame_count', 0)
                
                # Add physics parameters
                physics_module = getattr(sim, 'physics_module', None)
                if physics_module:
                    metadata.physics_parameters = {
                        'gravity': [physics_module.settings.gravity.x,
                                   physics_module.settings.gravity.y,
                                   physics_module.settings.gravity.z],
                        'time_scale': physics_module.settings.time_scale
                    }
                    
        return metadata
        
    def cleanup(self):
        """Clean up export resources"""
        if self.is_exporting:
            self.finalize_export()
            
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except:
            pass

class BatchExporter:
    """Batch export multiple simulations or configurations"""
    
    def __init__(self, export_manager: ExportManager):
        self.export_manager = export_manager
        self.batch_queue = []
        self.current_batch = 0
        self.total_batches = 0
        
    def add_export_job(self, config: ExportConfig, simulation_config: Dict = None):
        """Add export job to batch"""
        self.batch_queue.append({
            'export_config': config,
            'simulation_config': simulation_config
        })
        
    def start_batch_export(self):
        """Start batch export process"""
        self.total_batches = len(self.batch_queue)
        self.current_batch = 0
        
        # Start exporting in a separate thread
        export_thread = threading.Thread(target=self._run_batch_export)
        export_thread.daemon = True
        export_thread.start()
        
    def _run_batch_export(self):
        """Run batch export in background thread"""
        for i, job in enumerate(self.batch_queue):
            self.current_batch = i + 1
            print(f"Processing batch {self.current_batch}/{self.total_batches}")
            
            try:
                # Configure simulation if specified
                if job['simulation_config'] and self.export_manager.application:
                    self.configure_simulation(job['simulation_config'])
                    
                # Perform export
                self.export_manager.initialize_export(job['export_config'])
                
                # Simulate export process (in real implementation, this would capture actual frames)
                duration = job['export_config'].duration
                fps = job['export_config'].fps
                total_frames = int(duration * fps)
                
                for frame in range(total_frames):
                    time.sleep(1.0 / fps)  # Simulate frame timing
                    self.export_manager.capture_frame()
                    
                # Finalize export
                metadata = self.export_manager.create_metadata()
                self.export_manager.finalize_export(metadata)
                
            except Exception as e:
                print(f"Batch export error: {e}")
                
        print("Batch export completed")
        
    def configure_simulation(self, config: Dict):
        """Configure simulation based on batch job"""
        # This would configure the simulation based on the provided configuration
        pass
        
    def get_batch_progress(self) -> Dict:
        """Get batch export progress"""
        return {
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'progress': self.current_batch / self.total_batches if self.total_batches > 0 else 0
        }

# Demo and testing
if __name__ == "__main__":
    print("Testing Export Options...")
    
    # Test frame buffer
    try:
        frame_buffer = FrameBuffer(800, 600, multisample=True)
        print("✓ Frame buffer test passed")
        frame_buffer.cleanup()
    except Exception as e:
        print(f"✗ Frame buffer test failed: {e}")
        
    # Test data exporter
    data_exporter = DataExporter()
    test_data = {
        'metadata': {'test': True},
        'particles': [
            {'position': [1, 2, 3], 'velocity': [0, 1, 0], 'mass': 1.0, 'radius': 0.1}
        ],
        'physics_parameters': {'gravity': [0, -9.81, 0]}
    }
    
    try:
        data_exporter.export_json(test_data, 'test_export.json')
        print("✓ Data exporter test passed")
        os.remove('test_export.json')
    except Exception as e:
        print(f"✗ Data exporter test failed: {e}")
        
    # Test 3D model exporter
    model_exporter = ThreeDModelExporter()
    
    class TestParticle:
        def __init__(self):
            self.position = glm.vec3(1, 2, 3)
            self.velocity = glm.vec3(0, 1, 0)
            self.color = glm.vec3(1, 0, 0)
            self.radius = 0.1
            
    class TestSimulation:
        def __init__(self):
            self.particles = [TestParticle() for _ in range(5)]
            
    test_sim = TestSimulation()
    
    try:
        model_exporter.export_obj(test_sim, 'test_export.obj')
        print("✓ 3D model exporter test passed")
        os.remove('test_export.obj')
    except Exception as e:
        print(f"✗ 3D model exporter test failed: {e}")
        
    print("All export option tests completed!")