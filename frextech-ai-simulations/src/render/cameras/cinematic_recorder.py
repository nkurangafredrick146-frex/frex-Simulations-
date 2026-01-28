"""
Cinematic recorder for capturing camera animations and rendering sequences.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import cv2
import imageio
from PIL import Image
import subprocess
import tempfile
import shutil
from enum import Enum


class RecordingFormat(Enum):
    """Video recording formats."""
    MP4 = "mp4"
    AVI = "avi"
    WEBM = "webm"
    GIF = "gif"
    IMAGE_SEQUENCE = "image_sequence"


class RecordingQuality(Enum):
    """Recording quality presets."""
    LOW = "low"      # 480p, low bitrate
    MEDIUM = "medium" # 720p, medium bitrate
    HIGH = "high"    # 1080p, high bitrate
    ULTRA = "ultra"  # 4K, very high bitrate


@dataclass
class RecordingSettings:
    """Recording settings for cinematic capture."""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    format: RecordingFormat = RecordingFormat.MP4
    quality: RecordingQuality = RecordingQuality.HIGH
    bitrate: Optional[str] = None
    codec: str = "libx264"
    preset: str = "medium"
    crf: int = 18  # Constant Rate Factor (0-51, lower is better)
    
    def __post_init__(self):
        """Set default bitrate based on quality."""
        if self.bitrate is None:
            if self.quality == RecordingQuality.LOW:
                self.bitrate = "1000k"
            elif self.quality == RecordingQuality.MEDIUM:
                self.bitrate = "5000k"
            elif self.quality == RecordingQuality.HIGH:
                self.bitrate = "10000k"
            else:  # ULTRA
                self.bitrate = "50000k"
    
    def get_video_writer_params(self) -> Dict[str, Any]:
        """Get parameters for video writer.
        
        Returns:
            Dictionary of writer parameters
        """
        params = {
            "fps": self.fps,
            "codec": self.codec,
            "bitrate": self.bitrate
        }
        
        # Format-specific parameters
        if self.format == RecordingFormat.MP4:
            params["codec"] = "libx264"
            params["pix_fmt"] = "yuv420p"
        elif self.format == RecordingFormat.WEBM:
            params["codec"] = "libvpx-vp9"
        elif self.format == RecordingFormat.AVI:
            params["codec"] = "MJPG"
        
        return params
    
    def get_file_extension(self) -> str:
        """Get file extension for current format.
        
        Returns:
            File extension
        """
        return {
            RecordingFormat.MP4: ".mp4",
            RecordingFormat.AVI: ".avi",
            RecordingFormat.WEBM: ".webm",
            RecordingFormat.GIF: ".gif",
            RecordingFormat.IMAGE_SEQUENCE: ""
        }[self.format]


@dataclass
class RecordingMetadata:
    """Metadata for recorded cinematic."""
    title: str = ""
    description: str = ""
    author: str = ""
    created: str = ""
    duration: float = 0.0
    frame_count: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)
    camera_path: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Set creation time if not provided."""
        if not self.created:
            self.created = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "title": self.title,
            "description": self.description,
            "author": self.author,
            "created": self.created,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "settings": self.settings,
            "camera_path": self.camera_path,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordingMetadata":
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            RecordingMetadata instance
        """
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            created=data.get("created", ""),
            duration=data.get("duration", 0.0),
            frame_count=data.get("frame_count", 0),
            settings=data.get("settings", {}),
            camera_path=data.get("camera_path"),
            tags=data.get("tags", [])
        )


class FrameBuffer:
    """Thread-safe frame buffer for recording."""
    
    def __init__(self, max_size: int = 100):
        """Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to buffer
        """
        self.buffer = Queue(maxsize=max_size)
        self.frame_count = 0
        self.total_size_bytes = 0
        self.lock = threading.RLock()
    
    def put_frame(self, frame: np.ndarray, timestamp: float):
        """Put frame into buffer.
        
        Args:
            frame: Frame image data
            timestamp: Frame timestamp
        """
        with self.lock:
            self.buffer.put((frame, timestamp))
            self.frame_count += 1
            self.total_size_bytes += frame.nbytes
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Get frame from buffer.
        
        Returns:
            Tuple of (frame, timestamp) or (None, None) if empty
        """
        try:
            with self.lock:
                return self.buffer.get_nowait()
        except:
            return None, None
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            while not self.buffer.empty():
                try:
                    self.buffer.get_nowait()
                except:
                    break
            self.frame_count = 0
            self.total_size_bytes = 0
    
    def is_empty(self) -> bool:
        """Check if buffer is empty.
        
        Returns:
            True if buffer is empty
        """
        return self.buffer.empty()
    
    def size(self) -> int:
        """Get buffer size.
        
        Returns:
            Number of frames in buffer
        """
        return self.buffer.qsize()


class CinematicRecorder:
    """Main cinematic recorder class."""
    
    def __init__(self, settings: Optional[RecordingSettings] = None):
        """Initialize cinematic recorder.
        
        Args:
            settings: Recording settings
        """
        self.settings = settings or RecordingSettings()
        self.metadata = RecordingMetadata()
        
        # Recording state
        self.is_recording = False
        self.is_paused = False
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self.frame_counter = 0
        
        # Frame buffer
        self.frame_buffer = FrameBuffer(max_size=200)
        
        # Writers
        self.video_writer = None
        self.image_sequence_dir = None
        
        # Threading
        self.recording_thread = None
        self.writer_thread = None
        self.stop_event = threading.Event()
        self.writer_executor = ThreadPoolExecutor(max_workers=2)
        
        # Callbacks
        self.on_frame_captured = None
        self.on_recording_start = None
        self.on_recording_stop = None
        self.on_error = None
        
        # Camera path (if recording with animation)
        self.camera_path = None
        self.camera_controller = None
        
    def start_recording(self, output_path: Optional[Union[str, Path]] = None):
        """Start recording.
        
        Args:
            output_path: Output file path
        """
        if self.is_recording:
            self._log_warning("Recording already in progress")
            return
        
        # Reset state
        self.is_recording = True
        self.is_paused = False
        self.start_time = time.time()
        self.elapsed_time = 0.0
        self.frame_counter = 0
        self.frame_buffer.clear()
        self.stop_event.clear()
        
        # Setup output
        self.output_path = self._setup_output_path(output_path)
        
        # Initialize writer
        self._initialize_writer()
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True
        )
        self.recording_thread.start()
        
        # Start writer thread
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True
        )
        self.writer_thread.start()
        
        # Call callback
        if self.on_recording_start:
            self.on_recording_start()
        
        self._log_info(f"Started recording to {self.output_path}")
    
    def stop_recording(self) -> Optional[Path]:
        """Stop recording and save file.
        
        Returns:
            Path to saved recording, or None if error
        """
        if not self.is_recording:
            self._log_warning("No recording in progress")
            return None
        
        self._log_info("Stopping recording...")
        
        # Signal threads to stop
        self.stop_event.set()
        self.is_recording = False
        
        # Wait for threads to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)
        if self.writer_thread:
            self.writer_thread.join(timeout=5.0)
        
        # Finalize writer
        self._finalize_writer()
        
        # Update metadata
        self.metadata.duration = self.elapsed_time
        self.metadata.frame_count = self.frame_counter
        self.metadata.settings = {
            "width": self.settings.width,
            "height": self.settings.height,
            "fps": self.settings.fps,
            "format": self.settings.format.value,
            "quality": self.settings.quality.value
        }
        
        # Save metadata
        self._save_metadata()
        
        # Call callback
        if self.on_recording_stop:
            self.on_recording_stop(self.output_path)
        
        self._log_info(f"Recording saved to {self.output_path}")
        
        return self.output_path
    
    def pause_recording(self):
        """Pause recording."""
        if not self.is_recording:
            self._log_warning("No recording in progress")
            return
        
        self.is_paused = True
        self._log_info("Recording paused")
    
    def resume_recording(self):
        """Resume recording."""
        if not self.is_recording:
            self._log_warning("No recording in progress")
            return
        
        self.is_paused = False
        self._log_info("Recording resumed")
    
    def capture_frame(self, frame: np.ndarray):
        """Capture a frame for recording.
        
        Args:
            frame: Frame image data (HxWxC, uint8)
        """
        if not self.is_recording or self.is_paused:
            return
        
        # Check frame dimensions
        if frame.shape[0] != self.settings.height or frame.shape[1] != self.settings.width:
            frame = self._resize_frame(frame)
        
        # Convert to BGR if needed (OpenCV format)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume RGB, convert to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add timestamp
        timestamp = time.time() - self.start_time
        
        # Put in buffer
        self.frame_buffer.put_frame(frame, timestamp)
        
        # Update counters
        self.frame_counter += 1
        self.elapsed_time = timestamp
        
        # Call callback
        if self.on_frame_captured:
            self.on_frame_captured(self.frame_counter, timestamp)
    
    def record_camera_path(self, camera_path, camera_controller, 
                          render_callback: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]):
        """Record a camera path animation.
        
        Args:
            camera_path: CameraPath instance
            camera_controller: CameraController instance
            render_callback: Function that renders frame given camera state
        """
        if self.is_recording:
            self._log_warning("Recording already in progress")
            return
        
        self.camera_path = camera_path
        self.camera_controller = camera_controller
        
        # Start recording
        self.start_recording()
        
        # Calculate frame times
        total_frames = int(camera_path.duration * self.settings.fps)
        frame_time = 1.0 / self.settings.fps
        
        # Record animation
        for frame_idx in range(total_frames):
            if not self.is_recording or self.stop_event.is_set():
                break
            
            time_in_animation = frame_idx * frame_time
            
            # Get camera state at this time
            position, target, up, fov = camera_path.interpolate(time_in_animation)
            
            # Set camera controller state
            camera_controller.state.position = position
            camera_controller.state.target = target
            camera_controller.state.up = up
            camera_controller.intrinsics.fov = fov
            
            # Render frame
            frame = render_callback(position, target, up, fov)
            
            # Capture frame
            self.capture_frame(frame)
            
            # Sleep to maintain frame rate
            time.sleep(frame_time)
        
        # Stop recording
        return self.stop_recording()
    
    def _recording_loop(self):
        """Main recording loop."""
        try:
            while self.is_recording and not self.stop_event.is_set():
                # This loop is driven by external frame capture
                # For camera path recording, see record_camera_path method
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except Exception as e:
            self._log_error(f"Recording loop error: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    def _writer_loop(self):
        """Writer loop that processes frames from buffer."""
        try:
            while (self.is_recording or not self.frame_buffer.is_empty()) and not self.stop_event.is_set():
                # Get frame from buffer
                frame, timestamp = self.frame_buffer.get_frame()
                
                if frame is None:
                    time.sleep(0.001)  # Small sleep if buffer empty
                    continue
                
                # Write frame
                self._write_frame(frame)
                
        except Exception as e:
            self._log_error(f"Writer loop error: {e}")
            if self.on_error:
                self.on_error(str(e))
    
    def _write_frame(self, frame: np.ndarray):
        """Write frame to output.
        
        Args:
            frame: Frame to write
        """
        try:
            if self.settings.format == RecordingFormat.IMAGE_SEQUENCE:
                self._write_image_sequence_frame(frame)
            else:
                self._write_video_frame(frame)
                
        except Exception as e:
            self._log_error(f"Frame write error: {e}")
    
    def _write_video_frame(self, frame: np.ndarray):
        """Write frame to video file.
        
        Args:
            frame: Frame to write
        """
        if self.video_writer is None:
            self._log_error("Video writer not initialized")
            return
        
        self.video_writer.write(frame)
    
    def _write_image_sequence_frame(self, frame: np.ndarray):
        """Write frame to image sequence.
        
        Args:
            frame: Frame to write
        """
        if self.image_sequence_dir is None:
            self._log_error("Image sequence directory not initialized")
            return
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create filename with frame number
        filename = f"frame_{self.frame_counter:06d}.png"
        filepath = self.image_sequence_dir / filename
        
        # Save image
        image = Image.fromarray(frame_rgb)
        image.save(filepath, format='PNG', optimize=True)
    
    def _setup_output_path(self, output_path: Optional[Union[str, Path]]) -> Path:
        """Setup output path for recording.
        
        Args:
            output_path: Desired output path
            
        Returns:
            Final output path
        """
        if output_path is None:
            # Create default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}{self.settings.get_file_extension()}"
            output_path = Path.cwd() / "recordings" / filename
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def _initialize_writer(self):
        """Initialize video or image writer."""
        try:
            if self.settings.format == RecordingFormat.IMAGE_SEQUENCE:
                self._initialize_image_sequence_writer()
            else:
                self._initialize_video_writer()
                
        except Exception as e:
            self._log_error(f"Failed to initialize writer: {e}")
            raise
    
    def _initialize_video_writer(self):
        """Initialize video writer."""
        # FFmpeg parameters
        params = self.settings.get_video_writer_params()
        
        # Create temporary file for encoding
        temp_dir = tempfile.mkdtemp(prefix="frextech_recording_")
        temp_file = Path(temp_dir) / "temp_video.mp4"
        
        # FFmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.settings.width}x{self.settings.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.settings.fps),
            '-i', '-',  # Read from stdin
            '-an',  # No audio
            '-vcodec', params['codec'],
            '-b:v', params['bitrate'],
            '-preset', self.settings.preset,
            '-crf', str(self.settings.crf),
            str(self.output_path)
        ]
        
        # Start FFmpeg process
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Store temp directory for cleanup
        self.temp_dir = temp_dir
        
        self._log_info(f"Initialized video writer with codec {params['codec']}")
    
    def _initialize_image_sequence_writer(self):
        """Initialize image sequence writer."""
        # Create directory for image sequence
        sequence_dir = self.output_path.parent / f"{self.output_path.stem}_frames"
        sequence_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_sequence_dir = sequence_dir
        self._log_info(f"Initialized image sequence writer in {sequence_dir}")
    
    def _finalize_writer(self):
        """Finalize writer and cleanup."""
        try:
            if self.settings.format == RecordingFormat.IMAGE_SEQUENCE:
                self._finalize_image_sequence()
            else:
                self._finalize_video()
                
            # Cleanup temp directory
            if hasattr(self, 'temp_dir'):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                
        except Exception as e:
            self._log_error(f"Error finalizing writer: {e}")
    
    def _finalize_video(self):
        """Finalize video recording."""
        if hasattr(self, 'ffmpeg_process'):
            # Close stdin to signal EOF
            self.ffmpeg_process.stdin.close()
            
            # Wait for process to finish
            self.ffmpeg_process.wait(timeout=10.0)
            
            # Check return code
            if self.ffmpeg_process.returncode != 0:
                self._log_warning(f"FFmpeg exited with code {self.ffmpeg_process.returncode}")
    
    def _finalize_image_sequence(self):
        """Finalize image sequence recording."""
        # For image sequences, we might want to create a video from them
        if self.settings.format == RecordingFormat.IMAGE_SEQUENCE:
            self._create_video_from_image_sequence()
    
    def _create_video_from_image_sequence(self):
        """Create video from image sequence."""
        if self.image_sequence_dir is None:
            return
        
        # Check if FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         check=True)
        except:
            self._log_warning("FFmpeg not available, skipping video creation")
            return
        
        # Create video from image sequence
        input_pattern = str(self.image_sequence_dir / "frame_%06d.png")
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(self.settings.fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(self.output_path.with_suffix('.mp4'))
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL,
                         check=True)
            self._log_info(f"Created video from image sequence: {self.output_path.with_suffix('.mp4')}")
        except subprocess.CalledProcessError as e:
            self._log_error(f"Failed to create video from image sequence: {e}")
    
    def _save_metadata(self):
        """Save recording metadata."""
        metadata_path = self.output_path.with_suffix('.json')
        
        try:
            metadata_dict = self.metadata.to_dict()
            
            # Add camera path if available
            if self.camera_path:
                metadata_dict['camera_path'] = {
                    'name': self.camera_path.name,
                    'duration': self.camera_path.duration,
                    'keyframe_count': len(self.camera_path.keyframes),
                    'loop': self.camera_path.loop
                }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
                
            self._log_info(f"Saved metadata to {metadata_path}")
            
        except Exception as e:
            self._log_error(f"Failed to save metadata: {e}")
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to target dimensions.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, (self.settings.width, self.settings.height), 
                         interpolation=cv2.INTER_LINEAR)
    
    def _log_info(self, message: str):
        """Log info message.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [INFO] {message}")
    
    def _log_warning(self, message: str):
        """Log warning message.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [WARNING] {message}")
    
    def _log_error(self, message: str):
        """Log error message.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [ERROR] {message}")
    
    def get_recording_status(self) -> Dict[str, Any]:
        """Get current recording status.
        
        Returns:
            Dictionary with recording status
        """
        return {
            "is_recording": self.is_recording,
            "is_paused": self.is_paused,
            "elapsed_time": self.elapsed_time,
            "frame_count": self.frame_counter,
            "buffer_size": self.frame_buffer.size(),
            "output_path": str(self.output_path) if hasattr(self, 'output_path') else None
        }
    
    def set_metadata(self, **kwargs):
        """Set metadata fields.
        
        Args:
            **kwargs: Metadata fields to set
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
    
    def save_thumbnail(self, frame: Optional[np.ndarray] = None, 
                      thumbnail_size: Tuple[int, int] = (320, 180)) -> Optional[Path]:
        """Save thumbnail image for recording.
        
        Args:
            frame: Frame to use for thumbnail (uses last frame if None)
            thumbnail_size: Thumbnail dimensions
            
        Returns:
            Path to thumbnail file, or None if error
        """
        try:
            # Get frame for thumbnail
            if frame is None:
                # Try to get last frame from buffer
                # In practice, you might want to store the first frame
                frame = np.zeros((self.settings.height, self.settings.width, 3), dtype=np.uint8)
            
            # Resize for thumbnail
            thumbnail = cv2.resize(frame, thumbnail_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            
            # Save thumbnail
            thumbnail_path = self.output_path.with_suffix('.jpg')
            image = Image.fromarray(thumbnail_rgb)
            image.save(thumbnail_path, format='JPEG', quality=85)
            
            self._log_info(f"Saved thumbnail to {thumbnail_path}")
            return thumbnail_path
            
        except Exception as e:
            self._log_error(f"Failed to save thumbnail: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Shutdown executor
        if hasattr(self, 'writer_executor'):
            self.writer_executor.shutdown(wait=False)
        
        # Clear buffer
        self.frame_buffer.clear()
        
        self._log_info("Cinematic recorder cleaned up")
