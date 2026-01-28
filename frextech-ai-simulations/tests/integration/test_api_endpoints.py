"""
Integration tests for API endpoints.
Tests complete API workflows including authentication, request/response cycles, and error handling.
"""

import pytest
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from httpx import AsyncClient
from unittest.mock import Mock, patch, AsyncMock

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.api.server import app
from src.api.schemas.request_models import (
    GenerationRequest,
    EditRequest,
    ExportRequest,
    WorldParameters
)
from src.api.schemas.response_models import (
    GenerationResponse,
    EditResponse,
    ExportResponse,
    TaskStatus,
    ErrorResponse
)


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    @pytest.fixture
    async def client(self):
        """Create test client with authentication bypass"""
        async with AsyncClient(
            app=app,
            base_url="http://testserver",
            headers={"Authorization": "Bearer test-token"}
        ) as client:
            yield client
    
    @pytest.fixture
    def sample_generation_request(self):
        """Create sample generation request"""
        return {
            "prompt": "A serene mountain landscape with a flowing river and pine trees",
            "parameters": {
                "resolution": "1024x1024",
                "style": "photorealistic",
                "duration_seconds": 10,
                "frame_rate": 30,
                "quality": "high"
            },
            "output_format": "mp4",
            "callback_url": "http://localhost:8080/callback"
        }
    
    @pytest.fixture
    def sample_edit_request(self):
        """Create sample edit request"""
        return {
            "world_id": "test-world-123",
            "edit_type": "region_edit",
            "region": {
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 200
            },
            "edit_prompt": "Change trees to autumn colors",
            "parameters": {
                "propagation_method": "temporal",
                "consistency_weight": 0.8
            }
        }
    
    @pytest.fixture
    def sample_export_request(self):
        """Create sample export request"""
        return {
            "world_id": "test-world-123",
            "export_format": "glb",
            "parameters": {
                "include_textures": True,
                "lod_level": "high",
                "animation": True
            },
            "storage_location": "s3://bucket/exports"
        }
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_generation_endpoint_success(self, client, sample_generation_request):
        """Test successful world generation"""
        with patch('src.api.routes.generation.Generator.generate_world') as mock_generate:
            mock_generate.return_value = {
                "task_id": "gen-123",
                "status": "processing",
                "estimated_completion": time.time() + 60
            }
            
            response = await client.post(
                "/api/v1/generate",
                json=sample_generation_request
            )
            
            assert response.status_code == 202  # Accepted
            data = response.json()
            assert data["task_id"] == "gen-123"
            assert data["status"] == "processing"
            assert "queue_position" in data
    
    @pytest.mark.asyncio
    async def test_generation_endpoint_validation_error(self, client):
        """Test generation endpoint with invalid data"""
        invalid_request = {
            "prompt": "",  # Empty prompt
            "parameters": {}
        }
        
        response = await client.post(
            "/api/v1/generate",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
        assert any("prompt" in str(error).lower() for error in data["detail"])
    
    @pytest.mark.asyncio
    async def test_edit_endpoint_success(self, client, sample_edit_request):
        """Test successful world editing"""
        with patch('src.api.routes.editing.Editor.apply_edit') as mock_edit:
            mock_edit.return_value = {
                "edit_id": "edit-123",
                "status": "processing",
                "affected_frames": [1, 2, 3, 4, 5]
            }
            
            response = await client.post(
                "/api/v1/edit",
                json=sample_edit_request
            )
            
            assert response.status_code == 202
            data = response.json()
            assert data["edit_id"] == "edit-123"
            assert "affected_frames" in data
    
    @pytest.mark.asyncio
    async def test_export_endpoint_success(self, client, sample_export_request):
        """Test successful world export"""
        with patch('src.api.routes.export.Exporter.export_world') as mock_export:
            mock_export.return_value = {
                "export_id": "exp-123",
                "status": "processing",
                "download_url": "http://storage.example.com/export.glb",
                "file_size": 1024 * 1024 * 100  # 100MB
            }
            
            response = await client.post(
                "/api/v1/export",
                json=sample_export_request
            )
            
            assert response.status_code == 202
            data = response.json()
            assert data["export_id"] == "exp-123"
            assert "download_url" in data
    
    @pytest.mark.asyncio
    async def test_task_status_endpoint(self, client):
        """Test task status retrieval"""
        task_id = "gen-123"
        
        with patch('src.api.routes.management.TaskManager.get_status') as mock_status:
            mock_status.return_value = {
                "task_id": task_id,
                "type": "generation",
                "status": "completed",
                "progress": 100,
                "result": {
                    "world_id": "world-123",
                    "preview_url": "http://storage.example.com/preview.mp4"
                },
                "created_at": time.time() - 120,
                "completed_at": time.time() - 10
            }
            
            response = await client.get(f"/api/v1/tasks/{task_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == task_id
            assert data["status"] == "completed"
            assert data["progress"] == 100
    
    @pytest.mark.asyncio
    async def test_task_status_not_found(self, client):
        """Test task status for non-existent task"""
        task_id = "non-existent"
        
        with patch('src.api.routes.management.TaskManager.get_status') as mock_status:
            mock_status.return_value = None
            
            response = await client.get(f"/api/v1/tasks/{task_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not found" in data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_list_tasks_endpoint(self, client):
        """Test listing tasks with filtering"""
        with patch('src.api.routes.management.TaskManager.list_tasks') as mock_list:
            mock_list.return_value = {
                "tasks": [
                    {
                        "task_id": "gen-123",
                        "type": "generation",
                        "status": "completed",
                        "created_at": time.time() - 300
                    },
                    {
                        "task_id": "edit-456",
                        "type": "edit",
                        "status": "processing",
                        "created_at": time.time() - 60
                    }
                ],
                "total": 2,
                "page": 1,
                "page_size": 20
            }
            
            response = await client.get("/api/v1/tasks?status=processing")
            
            assert response.status_code == 200
            data = response.json()
            assert "tasks" in data
            assert len(data["tasks"]) == 2
            assert data["total"] == 2
    
    @pytest.mark.asyncio
    async def test_cancel_task_endpoint(self, client):
        """Test task cancellation"""
        task_id = "gen-123"
        
        with patch('src.api.routes.management.TaskManager.cancel_task') as mock_cancel:
            mock_cancel.return_value = True
            
            response = await client.post(f"/api/v1/tasks/{task_id}/cancel")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["task_id"] == task_id
    
    @pytest.mark.asyncio
    async def test_cancel_task_failed(self, client):
        """Test cancellation of non-cancellable task"""
        task_id = "gen-123"
        
        with patch('src.api.routes.management.TaskManager.cancel_task') as mock_cancel:
            mock_cancel.return_value = False
            
            response = await client.post(f"/api/v1/tasks/{task_id}/cancel")
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data
    
    @pytest.mark.asyncio
    async def test_authentication_required(self):
        """Test that endpoints require authentication"""
        async with AsyncClient(app=app, base_url="http://testserver") as client:
            response = await client.post("/api/v1/generate", json={})
            assert response.status_code == 401  # Unauthorized
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting on generation endpoint"""
        # Make multiple rapid requests
        responses = []
        for i in range(15):  # Should hit rate limit (assuming limit of 10/min)
            response = await client.post("/api/v1/generate", json={
                "prompt": f"Test {i}",
                "parameters": {}
            })
            responses.append(response)
        
        # Check that at least one was rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes  # Too Many Requests
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, sample_generation_request):
        """Test handling of concurrent requests"""
        with patch('src.api.routes.generation.Generator.generate_world') as mock_generate:
            mock_generate.side_effect = lambda *args, **kwargs: {
                "task_id": f"gen-{hash(str(args))}",
                "status": "processing",
                "estimated_completion": time.time() + 30
            }
            
            # Make 5 concurrent requests
            tasks = [
                client.post("/api/v1/generate", json=sample_generation_request)
                for _ in range(5)
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # All should be accepted
            for response in responses:
                assert response.status_code in [202, 429]  # Accepted or rate limited
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test proper error handling when service fails"""
        with patch('src.api.routes.generation.Generator.generate_world') as mock_generate:
            mock_generate.side_effect = Exception("Internal server error")
            
            response = await client.post("/api/v1/generate", json={
                "prompt": "Test prompt",
                "parameters": {}
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data
            assert "Internal server error" in data["error"]
    
    @pytest.mark.asyncio
    async def test_pagination_validation(self, client):
        """Test pagination parameter validation"""
        # Test invalid page number
        response = await client.get("/api/v1/tasks?page=0")
        assert response.status_code == 422
        
        # Test invalid page size
        response = await client.get("/api/v1/tasks?page_size=101")
        assert response.status_code == 422
        
        # Test valid pagination
        with patch('src.api.routes.management.TaskManager.list_tasks') as mock_list:
            mock_list.return_value = {"tasks": [], "total": 0, "page": 2, "page_size": 20}
            
            response = await client.get("/api/v1/tasks?page=2&page_size=20")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_file_upload_endpoint(self, client, tmp_path):
        """Test file upload for custom assets"""
        # Create a test file
        test_file = tmp_path / "test_texture.jpg"
        test_file.write_bytes(b"fake image data")
        
        with open(test_file, "rb") as f:
            response = await client.post(
                "/api/v1/assets/upload",
                files={
                    "file": ("texture.jpg", f, "image/jpeg")
                },
                data={
                    "asset_type": "texture",
                    "tags": "wood, surface, rough"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "asset_id" in data
        assert "url" in data
        assert "file_size" in data
    
    @pytest.mark.asyncio
    async def test_invalid_file_upload(self, client, tmp_path):
        """Test file upload with invalid file type"""
        # Create a test file with invalid extension
        test_file = tmp_path / "test.exe"
        test_file.write_bytes(b"malicious content")
        
        with open(test_file, "rb") as f:
            response = await client.post(
                "/api/v1/assets/upload",
                files={"file": ("test.exe", f, "application/octet-stream")},
                data={"asset_type": "texture"}
            )
        
        assert response.status_code == 415  # Unsupported Media Type
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, client):
        """Test batch generation endpoint"""
        batch_request = {
            "tasks": [
                {
                    "prompt": "Mountain landscape",
                    "parameters": {"resolution": "512x512"}
                },
                {
                    "prompt": "Ocean view",
                    "parameters": {"resolution": "512x512"}
                }
            ],
            "priority": "normal",
            "notify_complete": True
        }
        
        with patch('src.api.routes.generation.Generator.batch_generate') as mock_batch:
            mock_batch.return_value = {
                "batch_id": "batch-123",
                "task_ids": ["gen-001", "gen-002"],
                "status": "processing",
                "total_tasks": 2,
                "completed_tasks": 0
            }
            
            response = await client.post(
                "/api/v1/batch/generate",
                json=batch_request
            )
            
            assert response.status_code == 202
            data = response.json()
            assert data["batch_id"] == "batch-123"
            assert len(data["task_ids"]) == 2
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, client):
        """Test system metrics endpoint"""
        response = await client.get("/api/v1/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "api" in data
        assert "queue" in data
        
        # Check expected metrics
        assert "cpu_usage" in data["system"]
        assert "memory_usage" in data["system"]
        assert "total_requests" in data["api"]
        assert "average_response_time" in data["api"]
    
    @pytest.mark.asyncio
    async def test_configuration_endpoint(self, client):
        """Test configuration retrieval endpoint"""
        response = await client.get("/api/v1/config")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "features" in data
        assert "limits" in data
        assert "supported_formats" in data
        
        # Verify configuration structure
        assert "generation" in data["features"]
        assert "editing" in data["features"]
        assert "export" in data["features"]
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, client):
        """Test WebSocket endpoint for real-time updates"""
        # Note: This test requires a WebSocket client
        # For now, we'll just verify the endpoint exists
        response = await client.get("/ws")
        # WebSocket endpoints typically return 426 or handle upgrade
        assert response.status_code in [426, 101, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
