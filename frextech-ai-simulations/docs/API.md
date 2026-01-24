FrexTech AI Simulations - API Documentation

Overview

The FrexTech AI Simulations API provides programmatic access to our world generation, editing, and rendering capabilities. This RESTful API allows developers to integrate AI-powered 3D world creation into their applications.

Base URLs

· Development: http://localhost:8000
· Staging: https://api-staging.frextech-sim.com
· Production: https://api.frextech-sim.com

Authentication

API Keys

All endpoints require authentication using API keys passed in the request header:

```http
Authorization: Bearer your_api_key_here
```

Rate Limiting

· Free Tier: 100 requests/hour
· Pro Tier: 10,000 requests/hour
· Enterprise: Custom limits

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625097600
```

Versioning

The API uses URL versioning. Current version: v1

```http
https://api.frextech-sim.com/v1/generate
```

Endpoints

World Generation

Generate from Text

```http
POST /v1/generate/text
```

Generates a 3D world from a text description.

Request Body:

```json
{
  "prompt": "A serene mountain lake at sunset with pine trees",
  "quality": "standard", // "draft", "standard", "premium"
  "format": "gaussian", // "gaussian", "nerf", "mesh"
  "resolution": "2048x2048",
  "seed": 42,
  "metadata": {
    "style": "photorealistic",
    "season": "autumn"
  }
}
```

Response:

```json
{
  "job_id": "gen_1234567890abcdef",
  "status": "processing",
  "estimated_time": 45,
  "webhook_url": "https://your-domain.com/webhook",
  "poll_url": "/v1/jobs/gen_1234567890abcdef"
}
```

Generate from Image

```http
POST /v1/generate/image
```

Generates a 3D world from a 2D image.

Request Body (multipart/form-data):

· image: Image file (PNG, JPG, WebP)
· prompt (optional): Additional guidance
· remove_background: boolean
· extend_borders: boolean

Response: Same as text generation

Generate from Video

```http
POST /v1/generate/video
```

Generates a navigable 3D world from a video.

Request Body (multipart/form-data):

· video: Video file (MP4, MOV, AVI)
· extract_keyframes: boolean
· temporal_consistency: "strict" | "medium" | "loose"

World Editing

Select Region

```http
POST /v1/edit/select
```

Selects a region in an existing world for editing.

Request Body:

```json
{
  "world_id": "world_abcdef123456",
  "selection_type": "rectangle", // "rectangle", "polygon", "sampling"
  "coordinates": {
    "center": [0.5, 0.5, 0.5],
    "radius": 0.1
  },
  "prompt": "Replace trees with palm trees"
}
```

Apply Edit

```http
POST /v1/edit/apply
```

Applies edits to a selected region.

Request Body:

```json
{
  "edit_id": "edit_123456",
  "operation": "replace", // "replace", "add", "remove", "modify"
  "parameters": {
    "blend_strength": 0.7,
    "preserve_geometry": true,
    "style_match": "aggressive"
  }
}
```

Expand World

```http
POST /v1/edit/expand
```

Expands the boundaries of an existing world.

Request Body:

```json
{
  "world_id": "world_abcdef123456",
  "direction": "north", // "north", "south", "east", "west", "up", "down"
  "distance": 50, // meters
  "prompt": "Continue the forest with more variety"
}
```

Export & Rendering

Export World

```http
POST /v1/export
```

Exports a world in various formats.

Request Body:

```json
{
  "world_id": "world_abcdef123456",
  "format": "gltf", // "gltf", "fbx", "obj", "usd", "blend"
  "include_textures": true,
  "lod": "high", // "low", "medium", "high"
  "compression": true
}
```

Response:

```json
{
  "download_url": "https://storage.frextech-sim.com/export/world_abcdef123456.glb",
  "file_size": 15674329,
  "expires_at": "2024-01-01T00:00:00Z"
}
```

Render Preview

```http
POST /v1/render/preview
```

Renders a preview image or video of the world.

Request Body:

```json
{
  "world_id": "world_abcdef123456",
  "type": "image", // "image", "video", "panorama"
  "camera": {
    "position": [0, 1.7, 5],
    "target": [0, 1.7, 0],
    "fov": 60
  },
  "resolution": "1920x1080",
  "samples": 256,
  "denoise": true
}
```

Job Management

Get Job Status

```http
GET /v1/jobs/{job_id}
```

Response:

```json
{
  "job_id": "gen_1234567890abcdef",
  "type": "generation",
  "status": "completed", // "queued", "processing", "completed", "failed"
  "progress": 100,
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:01Z",
  "completed_at": "2024-01-01T12:00:45Z",
  "result": {
    "world_id": "world_abcdef123456",
    "preview_url": "https://storage.frextech-sim.com/previews/world_abcdef123456.jpg",
    "metadata": {
      "bounding_box": [[-10, -10, -5], [10, 10, 15]],
      "vertex_count": 1250000,
      "texture_size": "4K"
    }
  },
  "error": null
}
```

List Jobs

```http
GET /v1/jobs
```

Query Parameters:

· limit: Number of jobs to return (default: 50, max: 500)
· offset: Pagination offset
· type: Filter by job type
· status: Filter by status
· from_date: Filter jobs created after this date
· to_date: Filter jobs created before this date

Cancel Job

```http
POST /v1/jobs/{job_id}/cancel
```

World Management

List Worlds

```http
GET /v1/worlds
```

Query Parameters:

· limit, offset: Pagination
· sort: "newest", "oldest", "size"
· search: Search in metadata

Get World Details

```http
GET /v1/worlds/{world_id}
```

Response:

```json
{
  "world_id": "world_abcdef123456",
  "name": "Mountain Lake Sunset",
  "created_at": "2024-01-01T12:00:45Z",
  "modified_at": "2024-01-01T12:00:45Z",
  "format": "gaussian",
  "size_bytes": 256789123,
  "metadata": {
    "prompt": "A serene mountain lake at sunset with pine trees",
    "dimensions": [100, 50, 100],
    "bounding_box": [[-50, 0, -50], [50, 50, 50]]
  },
  "preview_url": "https://storage.frextech-sim.com/previews/world_abcdef123456.jpg",
  "thumbnail_url": "https://storage.frextech-sim.com/thumbnails/world_abcdef123456.jpg"
}
```

Delete World

```http
DELETE /v1/worlds/{world_id}
```

Response:

```json
{
  "deleted": true,
  "world_id": "world_abcdef123456"
}
```

System Status

Health Check

```http
GET /v1/health
```

Response:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": "healthy",
    "storage": "healthy",
    "gpu_workers": 4,
    "queue_size": 12
  }
}
```

Usage Statistics

```http
GET /v1/usage
```

Response:

```json
{
  "period": "2024-01",
  "total_requests": 1250,
  "generations": 45,
  "edits": 120,
  "exports": 85,
  "compute_hours": 12.5,
  "storage_used": 156743290,
  "remaining_credits": 8750
}
```

Webhooks

Configure webhooks to receive notifications when jobs complete.

Webhook Events

· generation.completed
· generation.failed
· edit.completed
· export.ready

Webhook Payload

```json
{
  "event": "generation.completed",
  "timestamp": "2024-01-01T12:00:45Z",
  "data": {
    "job_id": "gen_1234567890abcdef",
    "world_id": "world_abcdef123456",
    "preview_url": "https://storage.frextech-sim.com/previews/world_abcdef123456.jpg"
  }
}
```

Error Handling

Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "invalid_prompt",
    "message": "The provided prompt contains prohibited content",
    "details": {
      "violation": "explicit_content",
      "suggestion": "Modify your prompt to comply with our content policy"
    },
    "request_id": "req_1234567890abcdef"
  }
}
```

Common Error Codes

Code HTTP Status Description
authentication_required 401 Valid API key is required
invalid_api_key 401 Provided API key is invalid
rate_limit_exceeded 429 Rate limit exceeded
invalid_prompt 400 Prompt violates content policy
invalid_format 400 Unsupported file format
file_too_large 400 File exceeds size limit
insufficient_credits 402 Not enough credits
job_not_found 404 Job ID doesn't exist
world_not_found 404 World ID doesn't exist
internal_error 500 Server error

File Formats

Supported Input Formats

Images:

· JPEG/JPG (.jpg, .jpeg)
· PNG (.png)
· WebP (.webp)
· TIFF (.tif, .tiff)

Videos:

· MP4 (.mp4)
· MOV (.mov)
· AVI (.avi)
· WebM (.webm)

3D Models:

· glTF/GLB (.gltf, .glb)
· OBJ (.obj)
· FBX (.fbx)

Supported Output Formats

3D Formats:

· glTF 2.0 (.gltf, .glb)
· FBX (.fbx)
· OBJ (.obj)
· USD (.usd, .usda, .usdc)
· Blender (.blend)

Preview Formats:

· JPEG (.jpg)
· PNG (.png)
· WebM (.webm)
· MP4 (.mp4)

SDK Examples

Python SDK

```python
from frextech import FrexTechClient

client = FrexTechClient(api_key="your_api_key")

# Generate from text
job = client.generate.from_text(
    prompt="A futuristic city at night",
    quality="premium",
    format="gaussian"
)

# Wait for completion
world = job.wait_for_completion()

# Export to glTF
export = client.export(world.id, format="gltf")
export.download("city.glb")
```

JavaScript SDK

```javascript
import { FrexTechClient } from '@frextech/sdk';

const client = new FrexTechClient({ apiKey: 'your_api_key' });

// Generate from image
const formData = new FormData();
formData.append('image', imageFile);
formData.append('prompt', 'Make it look like winter');

const job = await client.generate.fromImage(formData);

// Poll for completion
const world = await job.pollUntilComplete();

// Get preview
const previewUrl = world.preview_url;
```

cURL Examples

```bash
# Generate from text
curl -X POST https://api.frextech-sim.com/v1/generate/text \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Ancient ruins in a jungle",
    "quality": "standard"
  }'

# Get job status
curl https://api.frextech-sim.com/v1/jobs/gen_1234567890abcdef \
  -H "Authorization: Bearer YOUR_API_KEY"

# Download export
curl https://storage.frextech-sim.com/export/world_abcdef123456.glb \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -o world.glb
```

Best Practices

Prompt Engineering

1. Be specific: "A medieval castle on a cliff overlooking a stormy sea at sunset" is better than "a castle"
2. Include details: Mention materials, lighting, weather, and scale
3. Use style references: "in the style of Studio Ghibli" or "photorealistic"
4. Consider composition: Specify foreground, midground, background elements

Performance Tips

1. Use webhooks: Polling is less efficient than webhook notifications
2. Cache results: Generated worlds can be reused
3. Batch operations: Combine edits before applying
4. Choose appropriate quality: Use "draft" for prototyping

Cost Optimization

1. Start with drafts: Use lower quality for concept validation
2. Reuse worlds: Edit existing worlds instead of generating new ones
3. Monitor usage: Check /v1/usage regularly
4. Clean up old data: Delete unused worlds to reduce storage costs

Support

· Documentation: https://docs.frextech-sim.com
· API Reference: https://api.frextech-sim.com/docs
· Community Forum: https://community.frextech-sim.com
· Support Email: api-support@frextech-sim.com
· Status Page: https://status.frextech-sim.com

Changelog

v1.0.0 (2024-01-01)

· Initial release
· World generation from text, images, and videos
· World editing and expansion
· Multiple export formats
· Job management system

---

Last Updated: January 1, 2024
API Version: 1.0.0