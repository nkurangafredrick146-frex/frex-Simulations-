"""
Authentication middleware for API security
"""

import os
import json
import hashlib
import hmac
import secrets
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ...utils.logging_config import setup_logging

logger = setup_logging("auth_middleware")

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory storage for API keys (in production, use Redis or database)
_api_keys: Dict[str, Dict] = {}
_api_key_hashes: Dict[str, str] = {}
_rate_limits: Dict[str, Dict] = {}


class APIKey:
    """API Key model"""
    
    def __init__(
        self,
        key_id: str,
        name: str,
        permissions: List[str],
        rate_limit: int = 100,
        expires_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None
    ):
        self.key_id = key_id
        self.name = name
        self.permissions = permissions
        self.rate_limit = rate_limit
        self.expires_at = expires_at
        self.created_at = created_at or datetime.utcnow()
        self.last_used = None
        self.usage_count = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "permissions": self.permissions,
            "rate_limit": self.rate_limit,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count
        }
    
    def is_valid(self) -> bool:
        """Check if API key is valid"""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def has_permission(self, permission: str) -> bool:
        """Check if key has specific permission"""
        return permission in self.permissions or "admin" in self.permissions


def generate_api_key() -> Tuple[str, str]:
    """Generate a new API key pair (key_id and secret)"""
    key_id = secrets.token_urlsafe(16)
    secret = secrets.token_urlsafe(32)
    return key_id, secret


def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage"""
    salt = os.getenv("API_KEY_SALT", "frextech-ai-simulations")
    return hashlib.sha256(f"{api_key}:{salt}".encode()).hexdigest()


def setup_api_keys(keys_file: Optional[str] = None):
    """Setup API keys from file or environment"""
    global _api_keys, _api_key_hashes
    
    # Load from file if provided
    if keys_file and os.path.exists(keys_file):
        try:
            with open(keys_file, 'r') as f:
                keys_data = json.load(f)
            
            for key_data in keys_data:
                api_key = APIKey(
                    key_id=key_data["key_id"],
                    name=key_data["name"],
                    permissions=key_data.get("permissions", ["read"]),
                    rate_limit=key_data.get("rate_limit", 100),
                    expires_at=datetime.fromisoformat(key_data["expires_at"]) if key_data.get("expires_at") else None,
                    created_at=datetime.fromisoformat(key_data["created_at"]) if key_data.get("created_at") else None
                )
                
                # Store hashed version of the key
                if "secret" in key_data:
                    hashed_key = hash_api_key(key_data["secret"])
                    _api_key_hashes[hashed_key] = api_key.key_id
                    _api_keys[api_key.key_id] = api_key
                    
            logger.info(f"Loaded {len(_api_keys)} API keys from {keys_file}")
        
        except Exception as e:
            logger.error(f"Failed to load API keys from {keys_file}: {e}")
    
    # Load from environment (for development)
    env_api_key = os.getenv("API_KEY")
    if env_api_key:
        # Create a default admin key
        api_key = APIKey(
            key_id="dev_admin",
            name="Development Admin",
            permissions=["admin", "read", "write", "delete"],
            rate_limit=1000,
            expires_at=datetime.utcnow() + timedelta(days=365)
        )
        
        hashed_key = hash_api_key(env_api_key)
        _api_key_hashes[hashed_key] = api_key.key_id
        _api_keys[api_key.key_id] = api_key
        
        logger.info("Loaded API key from environment")


async def authenticate_token(api_key: Optional[str] = Depends(api_key_header)) -> APIKey:
    """Authenticate API token"""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # Hash the provided key
    hashed_key = hash_api_key(api_key)
    
    # Look up the key
    key_id = _api_key_hashes.get(hashed_key)
    if not key_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Get the API key object
    api_key_obj = _api_keys.get(key_id)
    if not api_key_obj:
        raise HTTPException(
            status_code=401,
            detail="API key not found"
        )
    
    # Check if key is valid
    if not api_key_obj.is_valid():
        raise HTTPException(
            status_code=401,
            detail="API key expired"
        )
    
    # Update usage statistics
    api_key_obj.last_used = datetime.utcnow()
    api_key_obj.usage_count += 1
    
    logger.debug(f"Authenticated API key: {api_key_obj.name} ({key_id})")
    
    return api_key_obj


def require_auth(permission: Optional[str] = None):
    """Dependency for requiring authentication with optional permission check"""
    async def dependency(api_key: APIKey = Depends(authenticate_token)):
        if permission and not api_key.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        return api_key
    return dependency


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public endpoints
        public_paths = ["/health", "/docs", "/redoc", "/openapi.json", "/"]
        if request.url.path in public_paths:
            return await call_next(request)
        
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            # Allow OPTIONS requests for CORS
            if request.method == "OPTIONS":
                return await call_next(request)
            
            return JSONResponse(
                status_code=401,
                content={"detail": "API key required"}
            )
        
        try:
            # Authenticate the key
            api_key_obj = await authenticate_token(api_key)
            
            # Add API key info to request state
            request.state.api_key = api_key_obj
            
            # Continue with the request
            response = await call_next(request)
            
            # Add API key usage header
            response.headers["X-API-Key-Usage"] = str(api_key_obj.usage_count)
            
            return response
        
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal authentication error"}
            )


# Utility functions for API key management
def create_api_key(
    name: str,
    permissions: List[str],
    rate_limit: int = 100,
    expires_in_days: int = 30
) -> Tuple[str, str, Dict]:
    """Create a new API key"""
    key_id, secret = generate_api_key()
    
    api_key = APIKey(
        key_id=key_id,
        name=name,
        permissions=permissions,
        rate_limit=rate_limit,
        expires_at=datetime.utcnow() + timedelta(days=expires_in_days)
    )
    
    # Store hashed version
    hashed_key = hash_api_key(secret)
    _api_key_hashes[hashed_key] = key_id
    _api_keys[key_id] = api_key
    
    logger.info(f"Created API key: {name} ({key_id})")
    
    return key_id, secret, api_key.to_dict()


def revoke_api_key(key_id: str):
    """Revoke an API key"""
    if key_id in _api_keys:
        # Find and remove the hashed key
        hashed_keys_to_remove = [
            hashed for hashed, k_id in _api_key_hashes.items()
            if k_id == key_id
        ]
        
        for hashed_key in hashed_keys_to_remove:
            del _api_key_hashes[hashed_key]
        
        del _api_keys[key_id]
        logger.info(f"Revoked API key: {key_id}")
        return True
    
    return False


def list_api_keys() -> List[Dict]:
    """List all API keys (without secrets)"""
    return [
        {
            **api_key.to_dict(),
            "has_secret": False  # Never expose the secret
        }
        for api_key in _api_keys.values()
    ]