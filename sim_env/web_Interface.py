#!/usr/bin/env python3
"""
Real-time Web Interface with FastAPI
Advanced web interface for remote simulation control, real-time visualization, and collaborative features
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import uuid
import time
import base64
import io
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
import aiofiles
from PIL import Image
import logging
from logging.handlers import RotatingFileHandler
import redis.asyncio as redis
import jwt
from passlib.context import CryptContext
from sse_starlette.sse import EventSourceResponse
import websockets
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_interface")

class UserRole(Enum):
    VIEWER = "viewer"
    RESEARCHER = "researcher"
    ADMIN = "admin"

class SimulationStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class WebSocketMessageType(Enum):
    SIMULATION_UPDATE = "simulation_update"
    CONTROL_COMMAND = "control_command"
    VISUALIZATION_DATA = "visualization_data"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CHAT_MESSAGE = "chat_message"
    ERROR = "error"

@dataclass
class User:
    """User session information"""
    user_id: str
    username: str
    role: UserRole
    connection_time: datetime
    last_activity: datetime
    websocket: Optional[WebSocket] = None

@dataclass
class SimulationSession:
    """Simulation session management"""
    session_id: str
    name: str
    description: str
    simulation_type: str
    status: SimulationStatus
    created_by: str
    created_at: datetime
    participants: List[str]
    simulation_data: Dict[str, Any]
    settings: Dict[str, Any]

class ControlCommand(BaseModel):
    """Control command model"""
    command: str
    parameters: Dict[str, Any] = {}
    user_id: str
    session_id: str

class SimulationSettings(BaseModel):
    """Simulation settings model"""
    resolution: str = Field(default="1920x1080", regex=r'^\d+x\d+$')
    quality: str = Field(default="high", pattern="^(low|medium|high|ultra)$")
    physics_accuracy: str = Field(default="high", pattern="^(low|medium|high|precise)$")
    max_particles: int = Field(default=100000, ge=1000, le=10000000)
    real_time_rendering: bool = True
    data_streaming: bool = True

class UserCredentials(BaseModel):
    """User credentials model"""
    username: str
    password: str

class ChatMessage(BaseModel):
    """Chat message model"""
    user_id: str
    username: str
    message: str
    timestamp: datetime

class WebInterfaceManager:
    """Main web interface manager"""
    
    def __init__(self):
        # FastAPI app
        self.app = FastAPI(
            title="frex Simulations Web Interface",
            description="Real-time web interface for advanced physics simulations",
            version="1.0.0",
            docs_url="/api/docs",
            redoc_url="/api/redoc"
        )
        
        # Security
        self.security = HTTPBearer()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.secret_key = "your-secret-key-change-in-production"  # In production, use environment variable
        self.algorithm = "HS256"
        
        # Session management
        self.active_sessions: Dict[str, SimulationSession] = {}
        self.connected_users: Dict[str, User] = {}
        self.websocket_connections: Dict[str, List[WebSocket]] = {}
        
        # Simulation integration
        self.simulation_managers: Dict[str, Any] = {}  # Will hold simulation instances
        
        # Real-time data streaming
        self.data_streams: Dict[str, asyncio.Queue] = {}
        self.video_streams: Dict[str, asyncio.Queue] = {}
        
        # Redis for distributed sessions (optional)
        self.redis_client: Optional[redis.Redis] = None
        
        # Background tasks
        self.background_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize the application
        self._setup_middleware()
        self._setup_routes()
        self._setup_event_handlers()
        
        logger.info("Web Interface Manager initialized")

    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, restrict to specific domains
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add custom middleware for request logging
        @self.app.middleware("http")
        async def log_requests(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
            return response

    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        # Static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Authentication routes
        @self.app.post("/api/auth/login")
        async def login(credentials: UserCredentials):
            return await self._authenticate_user(credentials)
        
        @self.app.post("/api/auth/register")
        async def register(credentials: UserCredentials):
            return await self._register_user(credentials)
        
        @self.app.post("/api/auth/refresh")
        async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._refresh_token(credentials)
        
        # Simulation management routes
        @self.app.get("/api/simulations")
        async def list_simulations(_: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._get_simulation_list()
        
        @self.app.post("/api/simulations")
        async def create_simulation(
            simulation_data: dict,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._create_simulation(simulation_data, credentials)
        
        @self.app.get("/api/simulations/{session_id}")
        async def get_simulation(
            session_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._get_simulation(session_id, credentials)
        
        @self.app.put("/api/simulations/{session_id}")
        async def update_simulation(
            session_id: str,
            update_data: dict,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._update_simulation(session_id, update_data, credentials)
        
        @self.app.delete("/api/simulations/{session_id}")
        async def delete_simulation(
            session_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._delete_simulation(session_id, credentials)
        
        # Real-time data routes
        @self.app.get("/api/simulations/{session_id}/stream")
        async def stream_simulation_data(session_id: str):
            return await self._stream_simulation_data(session_id)
        
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await self._handle_websocket_connection(websocket, session_id)
        
        # File upload and download
        @self.app.post("/api/simulations/{session_id}/upload")
        async def upload_file(
            session_id: str,
            file_data: dict,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._handle_file_upload(session_id, file_data, credentials)
        
        @self.app.get("/api/simulations/{session_id}/export")
        async def export_simulation_data(
            session_id: str,
            format: str = "json",
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._export_simulation_data(session_id, format, credentials)
        
        # Collaboration features
        @self.app.get("/api/simulations/{session_id}/chat")
        async def get_chat_history(
            session_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._get_chat_history(session_id, credentials)
        
        @self.app.post("/api/simulations/{session_id}/chat")
        async def send_chat_message(
            session_id: str,
            message: dict,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            return await self._send_chat_message(session_id, message, credentials)
        
        # System monitoring
        @self.app.get("/api/system/status")
        async def get_system_status(_: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._get_system_status()
        
        @self.app.get("/api/system/metrics")
        async def get_system_metrics(_: HTTPAuthorizationCredentials = Depends(self.security)):
            return await self._get_system_metrics()
        
        # Root endpoint
        @self.app.get("/")
        async def root():
            return await self._serve_web_interface()

    def _setup_event_handlers(self):
        """Setup event handlers for simulation integration"""
        
        @self.app.on_event("startup")
        async def startup_event():
            await self._initialize_services()
            logger.info("Web Interface started successfully")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            await self._cleanup_services()
            logger.info("Web Interface shutdown complete")

    async def _initialize_services(self):
        """Initialize background services"""
        try:
            # Initialize Redis client
            self.redis_client = await redis.Redis(
                host='localhost', port=6379, db=0, decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None
        
        # Start background tasks
        self.background_tasks["session_cleanup"] = asyncio.create_task(
            self._session_cleanup_task()
        )
        self.background_tasks["system_monitoring"] = asyncio.create_task(
            self._system_monitoring_task()
        )

    async def _cleanup_services(self):
        """Cleanup background services"""
        # Cancel background tasks
        for task_name, task in self.background_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Background task {task_name} cancelled")
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
        
        # Close all WebSocket connections
        for session_id, connections in self.websocket_connections.items():
            for websocket in connections:
                await websocket.close()
        logger.info("All WebSocket connections closed")

    async def _session_cleanup_task(self):
        """Background task to clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # Remove sessions inactive for more than 24 hours
                    if (current_time - session.created_at) > timedelta(hours=24):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self._cleanup_session(session_id)
                    logger.info(f"Cleaned up expired session: {session_id}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup task error: {e}")

    async def _system_monitoring_task(self):
        """Background task to monitor system health"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Monitor system resources
                system_metrics = await self._collect_system_metrics()
                
                # Broadcast system status to admin users
                await self._broadcast_system_metrics(system_metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring task error: {e}")

    async def _authenticate_user(self, credentials: UserCredentials) -> Dict[str, Any]:
        """Authenticate user and return JWT token"""
        # In a real application, this would verify against a database
        # For demonstration, we'll use a simple in-memory user store
        
        users = {
            "admin": {"password": self._hash_password("admin123"), "role": UserRole.ADMIN},
            "researcher": {"password": self._hash_password("research123"), "role": UserRole.RESEARCHER},
            "viewer": {"password": self._hash_password("viewer123"), "role": UserRole.VIEWER}
        }
        
        if credentials.username not in users:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_data = users[credentials.username]
        if not self._verify_password(credentials.password, user_data["password"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate JWT token
        token_data = {
            "user_id": str(uuid.uuid4()),
            "username": credentials.username,
            "role": user_data["role"].value,
            "exp": datetime.now() + timedelta(hours=24)
        }
        
        token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "user_id": token_data["user_id"],
                "username": credentials.username,
                "role": user_data["role"].value
            }
        }

    async def _register_user(self, credentials: UserCredentials) -> Dict[str, Any]:
        """Register a new user"""
        # In a real application, this would store user in database
        # For demonstration, we'll just return success
        
        if len(credentials.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        
        return {
            "message": "User registered successfully",
            "user": {
                "username": credentials.username,
                "role": UserRole.VIEWER.value
            }
        }

    async def _refresh_token(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Refresh JWT token"""
        try:
            payload = jwt.decode(credentials.credentials, self.secret_key, algorithms=[self.algorithm])
            
            # Generate new token
            token_data = {
                "user_id": payload["user_id"],
                "username": payload["username"],
                "role": payload["role"],
                "exp": datetime.now() + timedelta(hours=24)
            }
            
            new_token = jwt.encode(token_data, self.secret_key, algorithm=self.algorithm)
            
            return {
                "access_token": new_token,
                "token_type": "bearer"
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    async def _get_simulation_list(self) -> List[Dict[str, Any]]:
        """Get list of available simulations"""
        simulations = []
        for session_id, session in self.active_sessions.items():
            simulations.append({
                "session_id": session_id,
                "name": session.name,
                "description": session.description,
                "simulation_type": session.simulation_type,
                "status": session.status.value,
                "created_by": session.created_by,
                "created_at": session.created_at.isoformat(),
                "participants": len(session.participants),
                "settings": session.settings
            })
        return simulations

    async def _create_simulation(self, simulation_data: dict, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Create a new simulation session"""
        try:
            # Verify user permissions
            user_info = await self._get_user_from_token(credentials.credentials)
            if user_info["role"] not in [UserRole.RESEARCHER.value, UserRole.ADMIN.value]:
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            
            # Create simulation session
            session = SimulationSession(
                session_id=session_id,
                name=simulation_data.get("name", "Unnamed Simulation"),
                description=simulation_data.get("description", ""),
                simulation_type=simulation_data.get("type", "generic"),
                status=SimulationStatus.STOPPED,
                created_by=user_info["username"],
                created_at=datetime.now(),
                participants=[],
                simulation_data={},
                settings=simulation_data.get("settings", {})
            )
            
            # Store session
            self.active_sessions[session_id] = session
            self.websocket_connections[session_id] = []
            self.data_streams[session_id] = asyncio.Queue()
            self.video_streams[session_id] = asyncio.Queue()
            
            # Initialize simulation manager
            await self._initialize_simulation_manager(session_id, simulation_data)
            
            logger.info(f"Created simulation session: {session_id} by {user_info['username']}")
            
            return {
                "session_id": session_id,
                "message": "Simulation created successfully",
                "session": {
                    "name": session.name,
                    "description": session.description,
                    "type": session.simulation_type,
                    "status": session.status.value
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating simulation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _get_simulation(self, session_id: str, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Get simulation session details"""
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Simulation session not found")
        
        session = self.active_sessions[session_id]
        user_info = await self._get_user_from_token(credentials.credentials)
        
        return {
            "session_id": session_id,
            "name": session.name,
            "description": session.description,
            "simulation_type": session.simulation_type,
            "status": session.status.value,
            "created_by": session.created_by,
            "created_at": session.created_at.isoformat(),
            "participants": session.participants,
            "settings": session.settings,
            "simulation_data": await self._get_safe_simulation_data(session_id, user_info)
        }

    async def _update_simulation(self, session_id: str, update_data: dict, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Update simulation session"""
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Simulation session not found")
        
        session = self.active_sessions[session_id]
        user_info = await self._get_user_from_token(credentials.credentials)
        
        # Check permissions
        if user_info["role"] not in [UserRole.RESEARCHER.value, UserRole.ADMIN.value] and user_info["username"] != session.created_by:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Update session properties
        if "name" in update_data:
            session.name = update_data["name"]
        if "description" in update_data:
            session.description = update_data["description"]
        if "settings" in update_data:
            session.settings.update(update_data["settings"])
        
        # Handle control commands
        if "command" in update_data:
            await self._handle_control_command(session_id, update_data["command"], update_data.get("parameters", {}), user_info)
        
        logger.info(f"Updated simulation session: {session_id} by {user_info['username']}")
        
        return {
            "message": "Simulation updated successfully",
            "session": {
                "name": session.name,
                "description": session.description,
                "status": session.status.value,
                "settings": session.settings
            }
        }

    async def _delete_simulation(self, session_id: str, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Delete simulation session"""
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Simulation session not found")
        
        session = self.active_sessions[session_id]
        user_info = await self._get_user_from_token(credentials.credentials)
        
        # Check permissions
        if user_info["role"] != UserRole.ADMIN.value and user_info["username"] != session.created_by:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        await self._cleanup_session(session_id)
        
        logger.info(f"Deleted simulation session: {session_id} by {user_info['username']}")
        
        return {"message": "Simulation deleted successfully"}

    async def _stream_simulation_data(self, session_id: str) -> EventSourceResponse:
        """Stream simulation data using Server-Sent Events"""
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Simulation session not found")
        
        async def event_generator():
            try:
                while True:
                    # Wait for new data
                    data = await self.data_streams[session_id].get()
                    yield {
                        "event": "simulation_update",
                        "data": json.dumps(data)
                    }
            except asyncio.CancelledError:
                logger.info(f"SSE connection closed for session: {session_id}")
        
        return EventSourceResponse(event_generator())

    async def _handle_websocket_connection(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection for real-time communication"""
        await websocket.accept()
        
        # Add to connection list
        if session_id not in self.websocket_connections:
            self.websocket_connections[session_id] = []
        self.websocket_connections[session_id].append(websocket)
        
        user_id = str(uuid.uuid4())
        logger.info(f"WebSocket connection established: {user_id} for session: {session_id}")
        
        try:
            while True:
                # Receive message from client
                message_data = await websocket.receive_text()
                message = json.loads(message_data)
                
                # Handle different message types
                await self._handle_websocket_message(session_id, user_id, message, websocket)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket connection closed: {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Remove from connection list
            if session_id in self.websocket_connections:
                self.websocket_connections[session_id].remove(websocket)
            
            # Notify other users
            await self._broadcast_to_session(session_id, {
                "type": WebSocketMessageType.USER_LEFT.value,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            })

    async def _handle_websocket_message(self, session_id: str, user_id: str, message: dict, websocket: WebSocket):
        """Handle WebSocket messages"""
        message_type = message.get("type")
        
        if message_type == WebSocketMessageType.CONTROL_COMMAND.value:
            # Handle control command
            command = ControlCommand(
                command=message["command"],
                parameters=message.get("parameters", {}),
                user_id=user_id,
                session_id=session_id
            )
            await self._handle_control_command(session_id, command.command, command.parameters, {"user_id": user_id})
            
        elif message_type == WebSocketMessageType.CHAT_MESSAGE.value:
            # Handle chat message
            chat_message = ChatMessage(
                user_id=user_id,
                username=message.get("username", "Anonymous"),
                message=message["message"],
                timestamp=datetime.now()
            )
            await self._broadcast_chat_message(session_id, chat_message)
            
        elif message_type == "join_session":
            # Handle user joining session
            user_info = {
                "user_id": user_id,
                "username": message.get("username", "Anonymous"),
                "role": message.get("role", UserRole.VIEWER.value)
            }
            await self._handle_user_join(session_id, user_info, websocket)
            
        else:
            logger.warning(f"Unknown WebSocket message type: {message_type}")

    async def _handle_user_join(self, session_id: str, user_info: dict, websocket: WebSocket):
        """Handle user joining a simulation session"""
        if session_id not in self.active_sessions:
            await websocket.send_text(json.dumps({
                "type": WebSocketMessageType.ERROR.value,
                "message": "Session not found"
            }))
            return
        
        session = self.active_sessions[session_id]
        session.participants.append(user_info["user_id"])
        
        # Send current simulation state to new user
        await websocket.send_text(json.dumps({
            "type": "session_state",
            "session": {
                "name": session.name,
                "description": session.description,
                "status": session.status.value,
                "participants": len(session.participants)
            },
            "simulation_data": await self._get_safe_simulation_data(session_id, user_info)
        }))
        
        # Notify other users
        await self._broadcast_to_session(session_id, {
            "type": WebSocketMessageType.USER_JOINED.value,
            "user_id": user_info["user_id"],
            "username": user_info["username"],
            "timestamp": datetime.now().isoformat()
        })

    async def _handle_control_command(self, session_id: str, command: str, parameters: dict, user_info: dict):
        """Handle simulation control commands"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        try:
            if command == "start":
                session.status = SimulationStatus.RUNNING
                await self._start_simulation(session_id)
                
            elif command == "stop":
                session.status = SimulationStatus.STOPPED
                await self._stop_simulation(session_id)
                
            elif command == "pause":
                session.status = SimulationStatus.PAUSED
                await self._pause_simulation(session_id)
                
            elif command == "reset":
                await self._reset_simulation(session_id)
                
            elif command == "update_parameters":
                await self._update_simulation_parameters(session_id, parameters)
                
            elif command == "capture_frame":
                frame_data = await self._capture_simulation_frame(session_id)
                await self._broadcast_to_session(session_id, {
                    "type": "frame_captured",
                    "frame_data": frame_data,
                    "user_id": user_info["user_id"],
                    "timestamp": datetime.now().isoformat()
                })
                
            else:
                logger.warning(f"Unknown control command: {command}")
                return
            
            # Broadcast status update
            await self._broadcast_to_session(session_id, {
                "type": WebSocketMessageType.SIMULATION_UPDATE.value,
                "status": session.status.value,
                "command": command,
                "user_id": user_info["user_id"],
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"Control command executed: {command} for session: {session_id} by user: {user_info.get('username', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error executing control command {command}: {e}")
            await self._broadcast_to_session(session_id, {
                "type": WebSocketMessageType.ERROR.value,
                "message": f"Error executing command: {str(e)}",
                "command": command,
                "timestamp": datetime.now().isoformat()
            })

    async def _broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast message to all WebSocket connections in a session"""
        if session_id not in self.websocket_connections:
            return
        
        message_json = json.dumps(message)
        disconnected = []
        
        for websocket in self.websocket_connections[session_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected WebSockets
        for websocket in disconnected:
            self.websocket_connections[session_id].remove(websocket)

    async def _broadcast_chat_message(self, session_id: str, chat_message: ChatMessage):
        """Broadcast chat message to session"""
        message = {
            "type": WebSocketMessageType.CHAT_MESSAGE.value,
            "user_id": chat_message.user_id,
            "username": chat_message.username,
            "message": chat_message.message,
            "timestamp": chat_message.timestamp.isoformat()
        }
        await self._broadcast_to_session(session_id, message)

    async def _broadcast_system_metrics(self, metrics: dict):
        """Broadcast system metrics to admin users"""
        message = {
            "type": "system_metrics",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to all sessions where admin users are present
        for session_id in self.active_sessions:
            await self._broadcast_to_session(session_id, message)

    async def _handle_file_upload(self, session_id: str, file_data: dict, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Handle file upload for simulation"""
        user_info = await self._get_user_from_token(credentials.credentials)
        
        if user_info["role"] not in [UserRole.RESEARCHER.value, UserRole.ADMIN.value]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        
        # Process uploaded file
        file_content = base64.b64decode(file_data["content"])
        file_type = file_data.get("type", "unknown")
        
        # Store file (in production, use proper file storage)
        filename = f"uploads/{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_data['filename']}"
        
        async with aiofiles.open(filename, 'wb') as f:
            await f.write(file_content)
        
        # Process based on file type
        if file_type == "initial_conditions":
            await self._load_initial_conditions(session_id, filename)
        elif file_type == "configuration":
            await self._load_configuration(session_id, filename)
        
        return {
            "message": "File uploaded successfully",
            "filename": filename,
            "size": len(file_content)
        }

    async def _export_simulation_data(self, session_id: str, format: str, credentials: HTTPAuthorizationCredentials) -> StreamingResponse:
        """Export simulation data in specified format"""
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Simulation session not found")
        
        user_info = await self._get_user_from_token(credentials.credentials)
        
        # Generate export data
        export_data = await self._generate_export_data(session_id, format, user_info)
        
        if format == "json":
            return StreamingResponse(
                io.BytesIO(json.dumps(export_data).encode()),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=simulation_{session_id}.json"}
            )
        elif format == "csv":
            # Convert to CSV format
            csv_data = await self._convert_to_csv(export_data)
            return StreamingResponse(
                io.BytesIO(csv_data.encode()),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=simulation_{session_id}.csv"}
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported export format")

    async def _get_chat_history(self, session_id: str, credentials: HTTPAuthorizationCredentials) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Simulation session not found")
        
        # In production, this would fetch from database
        # For demonstration, return empty list
        return []

    async def _send_chat_message(self, session_id: str, message: dict, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Send chat message to session"""
        if session_id not in self.active_sessions:
            raise HTTPException(status_code=404, detail="Simulation session not found")
        
        user_info = await self._get_user_from_token(credentials.credentials)
        
        chat_message = ChatMessage(
            user_id=user_info["user_id"],
            username=user_info["username"],
            message=message["message"],
            timestamp=datetime.now()
        )
        
        await self._broadcast_chat_message(session_id, chat_message)
        
        return {
            "message": "Chat message sent",
            "timestamp": chat_message.timestamp.isoformat()
        }

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(self.active_sessions),
            "connected_users": sum(len(connections) for connections in self.websocket_connections.values()),
            "system_metrics": await self._collect_system_metrics()
        }

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get detailed system metrics"""
        return await self._collect_system_metrics()

    async def _serve_web_interface(self) -> HTMLResponse:
        """Serve the main web interface"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
                <title>frex Simulations - Web Interface</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: rgba(255,255,255,0.1); 
                    padding: 20px; 
                    border-radius: 10px; 
                    backdrop-filter: blur(10px);
                }
                h1 { text-align: center; margin-bottom: 30px; }
                .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
                .status.running { background: #4CAF50; }
                .status.stopped { background: #f44336; }
                .status.paused { background: #ff9800; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>frex Simulations - Web Interface</h1>
                <div id="status" class="status stopped">Status: Loading...</div>
                <div id="content">
                    <p>Web interface is running. Use the API endpoints to interact with simulations.</p>
                    <p>Visit <a href="/api/docs" style="color: #4FC3F7;">/api/docs</a> for API documentation.</p>
                </div>
            </div>
            <script>
                // Simple status monitoring
                async function updateStatus() {
                    try {
                        const response = await fetch('/api/system/status');
                        const data = await response.json();
                        document.getElementById('status').textContent = `Status: ${data.status} | Active Sessions: ${data.active_sessions} | Connected Users: ${data.connected_users}`;
                        document.getElementById('status').className = `status ${data.status === 'operational' ? 'running' : 'stopped'}`;
                    } catch (error) {
                        document.getElementById('status').textContent = 'Status: Error connecting to server';
                        document.getElementById('status').className = 'status stopped';
                    }
                }
                updateStatus();
                setInterval(updateStatus, 5000);
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)

    # Simulation integration methods (to be implemented based on specific simulation requirements)
    
    async def _initialize_simulation_manager(self, session_id: str, simulation_data: dict):
        """Initialize simulation manager for a session"""
        # This would integrate with the specific simulation modules
        # For now, create a placeholder simulation manager
        self.simulation_managers[session_id] = {
            "type": simulation_data.get("type", "generic"),
            "status": "initialized",
            "data": {}
        }
        
        logger.info(f"Initialized simulation manager for session: {session_id}")

    async def _start_simulation(self, session_id: str):
        """Start simulation for a session"""
        if session_id in self.simulation_managers:
            self.simulation_managers[session_id]["status"] = "running"
            
            # Start background data streaming
            asyncio.create_task(self._simulation_data_streaming_task(session_id))
            
            logger.info(f"Started simulation for session: {session_id}")

    async def _stop_simulation(self, session_id: str):
        """Stop simulation for a session"""
        if session_id in self.simulation_managers:
            self.simulation_managers[session_id]["status"] = "stopped"
            logger.info(f"Stopped simulation for session: {session_id}")

    async def _pause_simulation(self, session_id: str):
        """Pause simulation for a session"""
        if session_id in self.simulation_managers:
            self.simulation_managers[session_id]["status"] = "paused"
            logger.info(f"Paused simulation for session: {session_id}")

    async def _reset_simulation(self, session_id: str):
        """Reset simulation for a session"""
        if session_id in self.simulation_managers:
            self.simulation_managers[session_id]["data"] = {}
            logger.info(f"Reset simulation for session: {session_id}")

    async def _update_simulation_parameters(self, session_id: str, parameters: dict):
        """Update simulation parameters"""
        if session_id in self.simulation_managers:
            self.simulation_managers[session_id]["data"].update(parameters)
            logger.info(f"Updated parameters for session: {session_id}")

    async def _capture_simulation_frame(self, session_id: str) -> str:
        """Capture simulation frame and return as base64"""
        # This would capture an actual frame from the simulation
        # For demonstration, create a placeholder image
        width, height = 320, 240
        image = Image.new('RGB', (width, height), color='blue')
        
        # Add some random "simulation" data
        for _ in range(100):
            x, y = np.random.randint(0, width), np.random.randint(0, height)
            color = tuple(np.random.randint(0, 255, 3))
            image.putpixel((x, y), color)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/jpeg;base64,{img_str}"

    async def _simulation_data_streaming_task(self, session_id: str):
        """Background task to stream simulation data"""
        while (session_id in self.simulation_managers and 
               self.simulation_managers[session_id]["status"] == "running"):
            try:
                # Generate simulation data
                simulation_data = await self._generate_simulation_data(session_id)
                
                # Send to data stream
                if session_id in self.data_streams:
                    await self.data_streams[session_id].put(simulation_data)
                
                # Broadcast via WebSocket
                await self._broadcast_to_session(session_id, {
                    "type": WebSocketMessageType.SIMULATION_UPDATE.value,
                    "data": simulation_data,
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(0.1)  # 10 FPS
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Simulation data streaming error for {session_id}: {e}")
                await asyncio.sleep(1)

    async def _generate_simulation_data(self, session_id: str) -> Dict[str, Any]:
        """Generate simulation data for streaming"""
        # This would generate actual simulation data
        # For demonstration, create placeholder data
        return {
            "timestamp": time.time(),
            "particles": np.random.randint(1000, 10000),
            "energy": np.random.uniform(0, 100),
            "temperature": np.random.uniform(0, 1000),
            "pressure": np.random.uniform(0, 100),
            "custom_metrics": {
                "metric1": np.random.random(),
                "metric2": np.random.random(),
                "metric3": np.random.random()
            }
        }

    async def _get_safe_simulation_data(self, session_id: str, user_info: dict) -> Dict[str, Any]:
        """Get simulation data with appropriate access controls"""
        if session_id not in self.simulation_managers:
            return {}
        
        base_data = self.simulation_managers[session_id].get("data", {})
        
        # Apply access controls based on user role
        if user_info["role"] == UserRole.VIEWER.value:
            # Viewers get limited data
            safe_data = {k: v for k, v in base_data.items() if not k.startswith("internal_")}
        elif user_info["role"] == UserRole.RESEARCHER.value:
            # Researchers get most data
            safe_data = base_data
        else:  # Admin
            # Admins get all data
            safe_data = base_data
        
        return safe_data

    async def _load_initial_conditions(self, session_id: str, filename: str):
        """Load initial conditions from file"""
        # Implementation would depend on simulation type
        logger.info(f"Loading initial conditions from {filename} for session {session_id}")

    async def _load_configuration(self, session_id: str, filename: str):
        """Load configuration from file"""
        # Implementation would depend on simulation type
        logger.info(f"Loading configuration from {filename} for session {session_id}")

    async def _generate_export_data(self, session_id: str, format: str, user_info: dict) -> Dict[str, Any]:
        """Generate export data for simulation"""
        simulation_data = await self._get_safe_simulation_data(session_id, user_info)
        session = self.active_sessions[session_id]
        
        return {
            "session_info": {
                "session_id": session_id,
                "name": session.name,
                "description": session.description,
                "type": session.simulation_type,
                "created_by": session.created_by,
                "created_at": session.created_at.isoformat()
            },
            "simulation_data": simulation_data,
            "export_info": {
                "format": format,
                "exported_at": datetime.now().isoformat(),
                "exported_by": user_info["username"]
            }
        }

    async def _convert_to_csv(self, data: dict) -> str:
        """Convert data to CSV format"""
        # Simple CSV conversion for demonstration
        csv_lines = []
        
        # Add headers
        csv_lines.append("key,value")
        
        # Flatten data structure
        def flatten_dict(d, prefix=""):
            items = []
            for k, v in d.items():
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, f"{prefix}{k}."))
                else:
                    items.append((f"{prefix}{k}", str(v)))
            return items
        
        flat_data = flatten_dict(data)
        for key, value in flat_data:
            csv_lines.append(f'"{key}","{value}"')
        
        return "\n".join(csv_lines)

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics"""
        # In production, this would collect actual system metrics
        # For demonstration, return placeholder metrics
        return {
            "cpu_usage": np.random.uniform(0, 100),
            "memory_usage": np.random.uniform(0, 100),
            "disk_usage": np.random.uniform(0, 100),
            "network_throughput": np.random.uniform(0, 1000),
            "active_tasks": len(asyncio.all_tasks()),
            "timestamp": datetime.now().isoformat()
        }

    async def _get_user_from_token(self, token: str) -> Dict[str, Any]:
        """Extract user information from JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return {
                "user_id": payload["user_id"],
                "username": payload["username"],
                "role": payload["role"]
            }
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def _cleanup_session(self, session_id: str):
        """Clean up a simulation session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        if session_id in self.websocket_connections:
            # Close all WebSocket connections
            for websocket in self.websocket_connections[session_id]:
                await websocket.close()
            del self.websocket_connections[session_id]
        
        if session_id in self.data_streams:
            del self.data_streams[session_id]
        
        if session_id in self.video_streams:
            del self.video_streams[session_id]
        
        if session_id in self.simulation_managers:
            del self.simulation_managers[session_id]
        
        logger.info(f"Cleaned up session: {session_id}")

# Create and export the FastAPI application
web_interface = WebInterfaceManager()
app = web_interface.app

# Run the application directly for testing
if __name__ == "__main__":
    uvicorn.run(
        "web_interface:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )