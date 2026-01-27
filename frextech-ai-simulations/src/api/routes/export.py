"""
Export routes for rendering and converting worlds
"""

import uuid
import asyncio
import tempfile
import os
from typing import List, Optional, Dict, Any, BinaryIO
from datetime import datetime
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    FileResponse,
    Response
)
from pydantic import BaseModel, Field

from ..middleware.authentication import require_auth
from ..schemas.request_models import (
    ExportRequest,
    RenderRequest,
    ConversionRequest,
    BatchExportRequest
)
from ..schemas.response_models import (
    ExportResponse,
    RenderResponse,
    ConversionResponse,
    BatchExportResponse,
    ProgressUpdate,
    APIError,
    SuccessResponse
)
from ..utils.async_processor import AsyncProcessor
from ..utils.cache_manager import CacheManager
from ..utils.file_handler import FileHandler
from src.render.engines.webgl_engine import WebGLEngine
from src.render.engines.ray_tracing_engine import RayTracingEngine
from src.render.cameras.camera_controller import CameraController
from src.render.cameras.cinematic_recorder import CinematicRecorder
from src.core.representation.nerf.nerf_model import NeRFModel
from src.core.representation.gaussian_splatting.gaussian_model import GaussianModel
from src.core.representation.mesh.mesh_generator import MeshGenerator
from src.utils.logging_config import setup_logging
from configs.model.inference import InferenceConfig

logger = setup_logging("export_routes")

router = APIRouter()

# Global components
webgl_engine: Optional[WebGLEngine] = None
ray_tracing_engine: Optional[RayTracingEngine] = None
camera_controller: Optional[CameraController] = None
cinematic_recorder: Optional[CinematicRecorder] = None
nerf_model: Optional[NeRFModel] = None
gaussian_model: Optional[GaussianModel] = None
mesh_generator: Optional[MeshGenerator] = None
file_handler: Optional[FileHandler] = None
cache_manager: Optional[CacheManager] = None
async_processor: Optional[AsyncProcessor] = None


class ExportSession:
    """Manages export sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.exports: Dict[str, Dict] = {}
        self.output_files: Dict[str, str] = {}  # session_id -> file_path
    
    def create_session(
        self,
        user_id: str,
        world_id: str,
        export_type: str,
        format: str,
        parameters: Dict
    ) -> str:
        """Create a new export session"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "world_id": world_id,
            "export_type": export_type,
            "format": format,
            "parameters": parameters,
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
            "output_path": None,
            "file_size": None,
            "download_url": None
        }
        
        return session_id
    
    def update_session(
        self,
        session_id: str,
        status: str = None,
        progress: float = None,
        error: str = None,
        result: Dict = None,
        output_path: str = None,
        file_size: int = None,
        download_url: str = None
    ):
        """Update session status"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if status:
            session["status"] = status
            if status == "processing" and not session["started_at"]:
                session["started_at"] = datetime.utcnow()
            elif status in ["completed", "failed"]:
                session["completed_at"] = datetime.utcnow()
        
        if progress is not None:
            session["progress"] = max(0.0, min(1.0, progress))
        
        if error is not None:
            session["error"] = error
        
        if result is not None:
            session["result"] = result
        
        if output_path is not None:
            session["output_path"] = output_path
            self.output_files[session_id] = output_path
        
        if file_size is not None:
            session["file_size"] = file_size
        
        if download_url is not None:
            session["download_url"] = download_url
        
        return True
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def list_user_sessions(self, user_id: str) -> List[Dict]:
        """List all export sessions for a user"""
        return [
            session for session in self.sessions.values()
            if session["user_id"] == user_id
        ]
    
    def get_output_file(self, session_id: str) -> Optional[str]:
        """Get output file path for session"""
        return self.output_files.get(session_id)
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old export files"""
        cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        sessions_to_clean = []
        for session_id, session in self.sessions.items():
            if session["created_at"].timestamp() < cutoff:
                sessions_to_clean.append(session_id)
        
        for session_id in sessions_to_clean:
            file_path = self.output_files.get(session_id)
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up export file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
            
            if session_id in self.sessions:
                del self.sessions[session_id]
            if session_id in self.output_files:
                del self.output_files[session_id]


# Initialize session manager
session_manager = ExportSession()


@router.on_event("startup")
async def startup_event():
    """Initialize export components"""
    global webgl_engine, ray_tracing_engine, camera_controller, cinematic_recorder
    global nerf_model, gaussian_model, mesh_generator, file_handler
    global cache_manager, async_processor
    
    logger.info("Initializing export components...")
    
    try:
        # Load configuration
        config = InferenceConfig()
        
        # Initialize render engines
        webgl_engine = WebGLEngine(config=config)
        ray_tracing_engine = RayTracingEngine(config=config)
        
        # Initialize camera components
        camera_controller = CameraController(config=config)
        cinematic_recorder = CinematicRecorder(config=config)
        
        # Initialize representation models
        nerf_model = NeRFModel(config=config)
        await nerf_model.load()
        
        gaussian_model = GaussianModel(config=config)
        await gaussian_model.load()
        
        mesh_generator = MeshGenerator(config=config)
        await mesh_generator.load()
        
        # Initialize file handler
        file_handler = FileHandler(
            base_path=config.export_base_path,
            max_file_size=config.max_export_size
        )
        
        logger.info("Export components initialized successfully")
        
        # Start cleanup task
        asyncio.create_task(periodic_cleanup())
    
    except Exception as e:
        logger.error(f"Failed to initialize export components: {e}")
        raise


async def periodic_cleanup():
    """Periodically clean up old export files"""
    while True:
        try:
            session_manager.cleanup_old_files()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        
        # Run every hour
        await asyncio.sleep(3600)


@router.post("/render", response_model=RenderResponse)
async def render_world(
    request: RenderRequest,
    background_tasks: BackgroundTasks,
    api_key=Depends(require_auth("read"))
):
    """
    Render a world to images or video
    
    - **world_id**: ID of world to render
    - **render_type**: Type of render (image, panorama, video, interactive)
    - **camera**: Camera parameters
    - **resolution**: Output resolution
    - **quality**: Render quality
    - **output_format**: Output file format
    - **async_mode**: Whether to process asynchronously
    """
    
    logger.info(f"Render request for world {request.world_id}, type: {request.render_type}")
    
    # Validate input
    valid_render_types = ["image", "panorama", "video", "interactive", "point_cloud"]
    if request.render_type not in valid_render_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid render type. Must be one of: {', '.join(valid_render_types)}"
        )
    
    valid_formats = {
        "image": ["png", "jpg", "jpeg", "webp", "exr", "hdr"],
        "panorama": ["png", "jpg", "jpeg", "webp", "exr", "hdr"],
        "video": ["mp4", "webm", "gif", "mov"],
        "interactive": ["html", "gltf", "glb"],
        "point_cloud": ["ply", "xyz", "pcd"]
    }
    
    if request.output_format not in valid_formats.get(request.render_type, []):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format for {request.render_type}. Must be one of: {', '.join(valid_formats[request.render_type])}"
        )
    
    # Create export session
    session_id = session_manager.create_session(
        user_id=api_key.key_id,
        world_id=request.world_id,
        export_type="render",
        format=request.output_format,
        parameters=request.dict()
    )
    
    if request.async_mode:
        if async_processor:
            task_id = await async_processor.submit_task(
                process_render,
                session_id=session_id,
                request=request,
                api_key=api_key.key_id
            )
            
            return RenderResponse(
                session_id=session_id,
                task_id=task_id,
                status="queued",
                progress=0.0,
                message="Render queued for processing",
                created_at=datetime.utcnow()
            )
        else:
            background_tasks.add_task(
                process_render_sync,
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return RenderResponse(
                session_id=session_id,
                status="processing",
                progress=0.0,
                message="Render started in background",
                created_at=datetime.utcnow()
            )
    else:
        try:
            result = await process_render_sync(
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return RenderResponse(
                session_id=session_id,
                status="completed",
                progress=1.0,
                output_url=result["download_url"],
                file_size=result["file_size"],
                message="Render completed successfully",
                created_at=datetime.utcnow(),
                metadata=result.get("metadata", {})
            )
        
        except Exception as e:
            session_manager.update_session(
                session_id=session_id,
                status="failed",
                error=str(e)
            )
            
            logger.error(f"Render failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Render failed: {str(e)}"
            )


async def process_render(
    session_id: str,
    request: RenderRequest,
    api_key: str
) -> Dict:
    """Process render operation"""
    logger.info(f"Processing render session: {session_id}")
    
    try:
        session_manager.update_session(
            session_id=session_id,
            status="processing",
            progress=0.1
        )
        
        # Load world data
        world_data = await load_world_data(request.world_id)
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.2
        )
        
        # Determine which render engine to use
        if request.render_type == "interactive":
            render_engine = webgl_engine
        elif request.quality == "high" and request.render_type in ["image", "panorama"]:
            render_engine = ray_tracing_engine
        else:
            render_engine = webgl_engine
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.3
        )
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(
            suffix=f".{request.output_format}",
            delete=False
        ) as tmp_file:
            output_path = tmp_file.name
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.4
        )
        
        # Perform render based on type
        if request.render_type == "image":
            # Single image render
            logger.debug("Rendering single image...")
            await render_engine.render_image(
                world_data=world_data,
                camera_params=request.camera.dict() if request.camera else None,
                output_path=output_path,
                resolution=request.resolution,
                quality=request.quality
            )
        
        elif request.render_type == "panorama":
            # Panorama render
            logger.debug("Rendering panorama...")
            await render_engine.render_panorama(
                world_data=world_data,
                output_path=output_path,
                resolution=request.resolution,
                quality=request.quality,
                stereo=request.parameters.get("stereo", False) if request.parameters else False
            )
        
        elif request.render_type == "video":
            # Video render
            logger.debug("Rendering video...")
            
            # Setup camera path if provided
            camera_path = None
            if request.camera and request.camera.path:
                camera_path = await camera_controller.create_path(
                    points=request.camera.path.points,
                    duration=request.camera.path.duration,
                    easing=request.camera.path.easing
                )
            
            await cinematic_recorder.record_video(
                world_data=world_data,
                camera_path=camera_path,
                output_path=output_path,
                resolution=request.resolution,
                fps=request.parameters.get("fps", 30) if request.parameters else 30,
                duration=request.parameters.get("duration", 10) if request.parameters else 10,
                quality=request.quality
            )
        
        elif request.render_type == "interactive":
            # Interactive export (WebGL/Three.js)
            logger.debug("Creating interactive export...")
            await webgl_engine.export_interactive(
                world_data=world_data,
                output_path=output_path,
                format=request.output_format,
                quality=request.quality,
                include_controls=request.parameters.get("include_controls", True) if request.parameters else True
            )
        
        elif request.render_type == "point_cloud":
            # Point cloud export
            logger.debug("Exporting point cloud...")
            
            # Convert world to point cloud representation
            point_cloud_data = await convert_to_point_cloud(world_data)
            
            # Save point cloud
            await file_handler.save_point_cloud(
                point_cloud_data=point_cloud_data,
                output_path=output_path,
                format=request.output_format
            )
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.8
        )
        
        # Get file size
        file_size = os.path.getsize(output_path)
        
        # Generate download URL (in production, this would be a signed URL)
        # For now, we'll use a direct download endpoint
        download_url = f"/api/v1/export/download/{session_id}"
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.9
        )
        
        # Prepare result
        result = {
            "output_path": output_path,
            "file_size": file_size,
            "download_url": download_url,
            "render_type": request.render_type,
            "format": request.output_format,
            "resolution": request.resolution,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store in session manager
        session_manager.update_session(
            session_id=session_id,
            status="completed",
            progress=1.0,
            result=result,
            output_path=output_path,
            file_size=file_size,
            download_url=download_url
        )
        
        logger.info(f"Render completed: {session_id}, size: {file_size} bytes")
        
        return result
    
    except Exception as e:
        logger.error(f"Render error in session {session_id}: {e}")
        session_manager.update_session(
            session_id=session_id,
            status="failed",
            error=str(e)
        )
        
        # Clean up temporary file if it exists
        if 'output_path' in locals() and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
        raise


async def process_render_sync(
    session_id: str,
    request: RenderRequest,
    api_key
) -> Dict:
    """Process render synchronously"""
    return await process_render(session_id, request, api_key.key_id)


async def convert_to_point_cloud(world_data: Dict) -> Dict:
    """Convert world data to point cloud representation"""
    # This is a simplified implementation
    # In reality, you would extract points from the world representation
    
    # Check world format
    world_format = world_data.get("format", "unknown")
    
    if world_format == "gaussian_splatting":
        return await gaussian_model.to_point_cloud(world_data)
    elif world_format == "nerf":
        return await nerf_model.to_point_cloud(world_data)
    elif world_format == "mesh":
        return await mesh_generator.to_point_cloud(world_data)
    else:
        # Default: generate synthetic point cloud
        return {
            "points": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "colors": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
            "normals": [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
        }


@router.post("/convert", response_model=ConversionResponse)
async def convert_world(
    request: ConversionRequest,
    background_tasks: BackgroundTasks,
    api_key=Depends(require_auth("write"))
):
    """
    Convert world between different representations
    
    - **world_id**: ID of world to convert
    - **target_format**: Target format (nerf, gaussian, mesh, voxel, point_cloud)
    - **parameters**: Conversion parameters
    - **async_mode**: Whether to process asynchronously
    """
    
    logger.info(f"Convert request for world {request.world_id} to {request.target_format}")
    
    # Validate input
    valid_formats = ["nerf", "gaussian", "mesh", "voxel", "point_cloud", "gltf", "usd", "fbx"]
    if request.target_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target format. Must be one of: {', '.join(valid_formats)}"
        )
    
    # Create export session
    session_id = session_manager.create_session(
        user_id=api_key.key_id,
        world_id=request.world_id,
        export_type="convert",
        format=request.target_format,
        parameters=request.dict()
    )
    
    if request.async_mode:
        if async_processor:
            task_id = await async_processor.submit_task(
                process_conversion,
                session_id=session_id,
                request=request,
                api_key=api_key.key_id
            )
            
            return ConversionResponse(
                session_id=session_id,
                task_id=task_id,
                status="queued",
                progress=0.0,
                message="Conversion queued for processing",
                created_at=datetime.utcnow()
            )
        else:
            background_tasks.add_task(
                process_conversion_sync,
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return ConversionResponse(
                session_id=session_id,
                status="processing",
                progress=0.0,
                message="Conversion started in background",
                created_at=datetime.utcnow()
            )
    else:
        try:
            result = await process_conversion_sync(
                session_id=session_id,
                request=request,
                api_key=api_key
            )
            
            return ConversionResponse(
                session_id=session_id,
                status="completed",
                progress=1.0,
                new_world_id=result["new_world_id"],
                world_data=result.get("world_data"),
                message="World converted successfully",
                created_at=datetime.utcnow(),
                metadata=result.get("metadata", {})
            )
        
        except Exception as e:
            session_manager.update_session(
                session_id=session_id,
                status="failed",
                error=str(e)
            )
            
            logger.error(f"Conversion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Conversion failed: {str(e)}"
            )


async def process_conversion(
    session_id: str,
    request: ConversionRequest,
    api_key: str
) -> Dict:
    """Process conversion operation"""
    logger.info(f"Processing conversion session: {session_id}")
    
    try:
        session_manager.update_session(
            session_id=session_id,
            status="processing",
            progress=0.1
        )
        
        # Load world data
        world_data = await load_world_data(request.world_id)
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.3
        )
        
        # Get source format
        source_format = world_data.get("format", "unknown")
        
        logger.debug(f"Converting from {source_format} to {request.target_format}")
        
        # Perform conversion
        converted_data = None
        
        if request.target_format == "nerf":
            # Convert to NeRF
            if source_format == "gaussian":
                converted_data = await gaussian_model.to_nerf(world_data)
            elif source_format == "mesh":
                converted_data = await mesh_generator.to_nerf(world_data)
            else:
                # Use NeRF model to create from scratch
                converted_data = await nerf_model.from_world_data(world_data)
        
        elif request.target_format == "gaussian":
            # Convert to Gaussian Splatting
            if source_format == "nerf":
                converted_data = await nerf_model.to_gaussian(world_data)
            elif source_format == "mesh":
                converted_data = await mesh_generator.to_gaussian(world_data)
            else:
                # Use Gaussian model to create from scratch
                converted_data = await gaussian_model.from_world_data(world_data)
        
        elif request.target_format == "mesh":
            # Convert to Mesh
            if source_format == "nerf":
                converted_data = await nerf_model.to_mesh(world_data)
            elif source_format == "gaussian":
                converted_data = await gaussian_model.to_mesh(world_data)
            else:
                # Use mesh generator to create from scratch
                converted_data = await mesh_generator.from_world_data(world_data)
        
        elif request.target_format == "gltf":
            # Convert to glTF/glb
            converted_data = await convert_to_gltf(world_data)
        
        elif request.target_format == "usd":
            # Convert to USD
            converted_data = await convert_to_usd(world_data)
        
        elif request.target_format == "fbx":
            # Convert to FBX
            converted_data = await convert_to_fbx(world_data)
        
        elif request.target_format == "point_cloud":
            # Convert to point cloud
            converted_data = await convert_to_point_cloud(world_data)
        
        else:
            raise ValueError(f"Unsupported target format: {request.target_format}")
        
        session_manager.update_session(
            session_id=session_id,
            progress=0.8
        )
        
        # Create new world ID
        new_world_id = str(uuid.uuid4())
        
        # Update format in converted data
        converted_data["format"] = request.target_format
        converted_data["converted_from"] = source_format
        converted_data["conversion_parameters"] = request.parameters.dict() if request.parameters else {}
        
        # Prepare result
        result = {
            "new_world_id": new_world_id,
            "world_data": converted_data,
            "source_world_id": request.world_id,
            "source_format": source_format,
            "target_format": request.target_format,
            "created_at": datetime.utcnow().isoformat(),
            "size": len(str(converted_data))
        }
        
        # Store in session manager
        session_manager.update_session(
            session_id=session_id,
            status="completed",
            progress=1.0,
            result=result
        )
        
        logger.info(f"Conversion completed: {session_id} -> {new_world_id}")
        
        return result
    
    except Exception as e:
        logger.error(f"Conversion error in session {session_id}: {e}")
        session_manager.update_session(
            session_id=session_id,
            status="failed",
            error=str(e)
        )
        raise


async def process_conversion_sync(
    session_id: str,
    request: ConversionRequest,
    api_key
) -> Dict:
    """Process conversion synchronously"""
    return await process_conversion(session_id, request, api_key.key_id)


async def convert_to_gltf(world_data: Dict) -> Dict:
    """Convert world data to glTF format"""
    # This would be implemented with a proper glTF exporter
    # For now, return a placeholder
    return {
        "format": "gltf",
        "data": {
            "scenes": [],
            "nodes": [],
            "meshes": [],
            "materials": [],
            "textures": [],
            "images": []
        },
        "metadata": {
            "generator": "FrexTech AI Simulations",
            "version": "2.0"
        }
    }


async def convert_to_usd(world_data: Dict) -> Dict:
    """Convert world data to USD format"""
    # This would be implemented with a proper USD exporter
    return {
        "format": "usd",
        "data": "#usda 1.0\n(\n    doc = \"FrexTech AI Simulation\"\n    metersPerUnit = 1\n    upAxis = \"Y\"\n)\n\ndef Xform \"World\"\n{\n}",
        "metadata": {
            "format": "usda",
            "version": "1.0"
        }
    }


async def convert_to_fbx(world_data: Dict) -> Dict:
    """Convert world data to FBX format"""
    # This would be implemented with a proper FBX exporter
    return {
        "format": "fbx",
        "data": b"",  # Binary FBX data
        "metadata": {
            "format": "binary",
            "version": "2020"
        }
    }


@router.post("/batch", response_model=BatchExportResponse)
async def batch_export(
    request: BatchExportRequest,
    api_key=Depends(require_auth("write"))
):
    """
    Batch export multiple worlds
    
    - **exports**: List of export specifications
    - **common_parameters**: Common parameters for all exports
    - **max_concurrent**: Maximum concurrent exports
    """
    
    logger.info(f"Batch export request for {len(request.exports)} exports")
    
    if len(request.exports) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 exports per batch"
        )
    
    if not request.exports:
        raise HTTPException(
            status_code=400,
            detail="At least one export is required"
        )
    
    # Create batch session
    batch_id = str(uuid.uuid4())
    session_ids = []
    
    for export_spec in request.exports:
        # Create individual session for each export
        session_id = session_manager.create_session(
            user_id=api_key.key_id,
            world_id=export_spec.world_id,
            export_type=export_spec.export_type,
            format=export_spec.format,
            parameters={
                **export_spec.parameters.dict() if export_spec.parameters else {},
                **request.common_parameters.dict() if request.common_parameters else {}
            }
        )
        session_ids.append(session_id)
        
        # Submit for processing
        if async_processor:
            # Determine processing function based on export type
            if export_spec.export_type == "render":
                process_func = process_render
            elif export_spec.export_type == "convert":
                process_func = process_conversion
            else:
                process_func = None
            
            if process_func:
                await async_processor.submit_task(
                    process_func,
                    session_id=session_id,
                    request=export_spec,
                    api_key=api_key.key_id
                )
    
    return BatchExportResponse(
        batch_id=batch_id,
        session_ids=session_ids,
        total_tasks=len(session_ids),
        status="queued",
        message=f"Batch export started with {len(session_ids)} tasks"
    )


@router.get("/download/{session_id}")
async def download_export(
    session_id: str,
    api_key=Depends(require_auth("read")),
    as_attachment: bool = Query(True, description="Download as attachment")
):
    """Download exported file"""
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Export session not found"
        )
    
    # Check authorization
    if session["user_id"] != api_key.key_id and "admin" not in api_key.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to download this export"
        )
    
    # Check status
    if session["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Export not ready. Status: {session['status']}"
        )
    
    # Get output file path
    output_path = session_manager.get_output_file(session_id)
    
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(
            status_code=404,
            detail="Export file not found"
        )
    
    # Determine filename
    filename = f"export_{session_id}.{session['format']}"
    
    # Determine media type based on format
    media_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "exr": "image/x-exr",
        "hdr": "image/vnd.radiance",
        "mp4": "video/mp4",
        "webm": "video/webm",
        "gif": "image/gif",
        "mov": "video/quicktime",
        "html": "text/html",
        "gltf": "model/gltf+json",
        "glb": "model/gltf-binary",
        "ply": "application/octet-stream",
        "xyz": "text/plain",
        "pcd": "application/octet-stream",
        "usd": "model/vnd.usd+zip",
        "fbx": "application/octet-stream"
    }
    
    media_type = media_types.get(session['format'].lower(), "application/octet-stream")
    
    # Return file response
    return FileResponse(
        path=output_path,
        filename=filename if as_attachment else None,
        media_type=media_type,
        headers={
            "Content-Disposition": f"{'attachment' if as_attachment else 'inline'}; filename=\"{filename}\"",
            "X-Export-Session": session_id,
            "X-File-Size": str(session.get("file_size", 0))
        }
    )


@router.get("/sessions/{session_id}", response_model=Dict)
async def get_export_status(
    session_id: str,
    api_key=Depends(require_auth("read"))
):
    """Get status of an export session"""
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    # Check authorization
    if session["user_id"] != api_key.key_id and "admin" not in api_key.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to access this session"
        )
    
    response = {
        "session_id": session_id,
        "status": session["status"],
        "progress": session["progress"],
        "world_id": session["world_id"],
        "export_type": session["export_type"],
        "format": session["format"],
        "created_at": session["created_at"],
        "started_at": session.get("started_at"),
        "completed_at": session.get("completed_at"),
        "error": session.get("error"),
        "file_size": session.get("file_size"),
        "download_url": session.get("download_url")
    }
    
    return response


@router.get("/sessions", response_model=List[Dict])
async def list_export_sessions(
    status: Optional[str] = Query(None, regex="^(pending|processing|completed|failed)$"),
    export_type: Optional[str] = Query(None, regex="^(render|convert)$"),
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    api_key=Depends(require_auth("read"))
):
    """List export sessions for the authenticated user"""
    
    sessions = session_manager.list_user_sessions(api_key.key_id)
    
    # Filter by status if provided
    if status:
        sessions = [s for s in sessions if s["status"] == status]
    
    # Filter by export type if provided
    if export_type:
        sessions = [s for s in sessions if s["export_type"] == export_type]
    
    # Apply pagination
    paginated = sessions[offset:offset + limit]
    
    return [
        {
            "session_id": s["session_id"],
            "status": s["status"],
            "progress": s["progress"],
            "world_id": s["world_id"],
            "export_type": s["export_type"],
            "format": s["format"],
            "created_at": s["created_at"],
            "started_at": s.get("started_at"),
            "completed_at": s.get("completed_at"),
            "error": s.get("error"),
            "file_size": s.get("file_size"),
            "download_url": s.get("download_url")
        }
        for s in paginated
    ]


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def cancel_export(
    session_id: str,
    api_key=Depends(require_auth("write"))
):
    """Cancel an export session"""
    
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    
    # Check authorization
    if session["user_id"] != api_key.key_id and "admin" not in api_key.permissions:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to cancel this session"
        )
    
    # Only cancel if not already completed/failed
    if session["status"] in ["completed", "failed"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel session with status: {session['status']}"
        )
    
    # Update session status
    session_manager.update_session(
        session_id=session_id,
        status="cancelled",
        error="Cancelled by user"
    )
    
    # Clean up output file if it exists
    output_path = session_manager.get_output_file(session_id)
    if output_path and os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception as e:
            logger.warning(f"Failed to clean up file {output_path}: {e}")
    
    return SuccessResponse(
        success=True,
        message="Export session cancelled",
        data={"session_id": session_id}
    )


@router.get("/capabilities")
async def get_export_capabilities(
    api_key=Depends(require_auth("read"))
):
    """Get export capabilities and limits"""
    
    return {
        "render": {
            "types": [
                {
                    "type": "image",
                    "description": "Single image render",
                    "formats": ["png", "jpg", "jpeg", "webp", "exr", "hdr"],
                    "max_resolution": "8192x8192",
                    "quality_levels": ["low", "medium", "high", "ultra"]
                },
                {
                    "type": "panorama",
                    "description": "360-degree panorama",
                    "formats": ["png", "jpg", "jpeg", "webp", "exr", "hdr"],
                    "max_resolution": "16384x8192",
                    "stereo_support": True
                },
                {
                    "type": "video",
                    "description": "Animated video",
                    "formats": ["mp4", "webm", "gif", "mov"],
                    "max_resolution": "4096x2160",
                    "max_duration": 300,  # seconds
                    "max_fps": 60
                },
                {
                    "type": "interactive",
                    "description": "Interactive WebGL export",
                    "formats": ["html", "gltf", "glb"],
                    "features": ["camera_controls", "lighting_controls", "animation"]
                },
                {
                    "type": "point_cloud",
                    "description": "3D point cloud",
                    "formats": ["ply", "xyz", "pcd"],
                    "max_points": 10000000
                }
            ],
            "camera": {
                "types": ["perspective", "orthographic", "panoramic"],
                "controls": ["position", "rotation", "fov", "near", "far"],
                "animation": ["path", "keyframe", "orbit"]
            }
        },
        "conversion": {
            "source_formats": ["nerf", "gaussian", "mesh", "voxel", "point_cloud"],
            "target_formats": [
                "nerf",
                "gaussian",
                "mesh",
                "voxel",
                "point_cloud",
                "gltf",
                "usd",
                "fbx"
            ],
            "quality_presets": ["fast", "balanced", "high", "lossless"]
        },
        "limits": {
            "max_file_size": 10737418240,  # 10GB
            "max_batch_size": 100,
            "max_concurrent_exports": 10,
            "retention_period": 24  # hours
        }
    }


async def load_world_data(world_id: str) -> Dict:
    """Load world data from storage"""
    # In a real implementation, this would load from database or file system
    # For now, return a placeholder
    return {
        "world_id": world_id,
        "format": "gaussian",
        "data": {
            "position": [0, 0, 0],
            "scale": 1.0,
            "rotation": [0, 0, 0],
            "gaussians": []
        },
        "created_at": datetime.utcnow().isoformat()
    }