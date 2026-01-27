"""
File handling utilities for uploads, downloads, and storage management
"""

import os
import io
import shutil
import hashlib
import mimetypes
import tempfile
import uuid
import asyncio
from typing import Dict, List, Optional, Tuple, BinaryIO, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import zipfile
import tarfile
import aiofiles
import aiofiles.os
from fastapi import UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse

from ...utils.logging_config import setup_logging

logger = setup_logging("file_handler")


class FileHandler:
    """
    File handling utility for managing uploads, downloads, and storage
    """
    
    def __init__(
        self,
        base_path: str = "data/uploads",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        allowed_extensions: List[str] = None,
        allowed_mime_types: List[str] = None,
        chunk_size: int = 8192,  # 8KB chunks
        cleanup_interval: int = 3600,  # 1 hour
        max_temp_age: int = 24 * 3600,  # 24 hours
        use_checksum: bool = True,
        create_dirs: bool = True
    ):
        """
        Initialize file handler
        
        Args:
            base_path: Base directory for file storage
            max_file_size: Maximum file size in bytes
            allowed_extensions: List of allowed file extensions
            allowed_mime_types: List of allowed MIME types
            chunk_size: Chunk size for streaming
            cleanup_interval: Interval for cleaning up temporary files (seconds)
            max_temp_age: Maximum age of temporary files (seconds)
            use_checksum: Whether to compute file checksums
            create_dirs: Whether to create directories if they don't exist
        """
        self.base_path = Path(base_path).resolve()
        self.max_file_size = max_file_size
        self.chunk_size = chunk_size
        self.cleanup_interval = cleanup_interval
        self.max_temp_age = max_temp_age
        self.use_checksum = use_checksum
        self.create_dirs = create_dirs
        
        # Default allowed extensions and MIME types
        self.allowed_extensions = allowed_extensions or [
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp',
            '.mp4', '.webm', '.mov', '.avi', '.mkv',
            '.gltf', '.glb', '.obj', '.fbx', '.stl', '.ply',
            '.json', '.txt', '.csv', '.xml',
            '.zip', '.tar', '.gz'
        ]
        
        self.allowed_mime_types = allowed_mime_types or [
            'image/png', 'image/jpeg', 'image/gif', 'image/bmp',
            'image/tiff', 'image/webp',
            'video/mp4', 'video/webm', 'video/quicktime',
            'application/json', 'text/plain', 'text/csv', 'application/xml',
            'application/zip', 'application/x-tar', 'application/gzip',
            'model/gltf+json', 'model/gltf-binary',
            'application/octet-stream'  # For binary files
        ]
        
        # Create base directory if needed
        if self.create_dirs:
            self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.dirs = {
            'uploads': self.base_path / 'uploads',
            'exports': self.base_path / 'exports',
            'temp': self.base_path / 'temp',
            'cache': self.base_path / 'cache',
            'backup': self.base_path / 'backup'
        }
        
        # Create subdirectories
        for dir_path in self.dirs.values():
            if self.create_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # File metadata storage
        self.metadata_file = self.base_path / 'metadata.json'
        self.metadata: Dict[str, Dict] = self._load_metadata()
        
        # Statistics
        self.stats = {
            'uploads': 0,
            'downloads': 0,
            'deletes': 0,
            'errors': 0,
            'total_size': 0,
            'start_time': datetime.utcnow().isoformat()
        }
        
        # Start cleanup task
        self.cleanup_task = None
        self.is_running = False
        
        logger.info(f"FileHandler initialized with base path: {self.base_path}")
    
    async def start(self):
        """Start file handler (start cleanup task)"""
        if self.is_running:
            return
        
        self.is_running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("FileHandler started")
    
    async def shutdown(self):
        """Shutdown file handler"""
        self.is_running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save metadata
        self._save_metadata()
        
        logger.info("FileHandler shutdown complete")
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load file metadata from disk"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
        
        return {}
    
    def _save_metadata(self):
        """Save file metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def save_upload(
        self,
        upload_file: UploadFile,
        category: str = "uploads",
        custom_filename: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Save uploaded file
        
        Args:
            upload_file: FastAPI UploadFile object
            category: File category (uploads, exports, temp, cache, backup)
            custom_filename: Custom filename (without extension)
            metadata: Additional file metadata
        
        Returns:
            Dictionary with file information
        """
        if category not in self.dirs:
            raise ValueError(f"Invalid category: {category}. Must be one of: {list(self.dirs.keys())}")
        
        # Validate file
        await self._validate_file(upload_file)
        
        # Generate filename
        original_filename = upload_file.filename or "unknown"
        file_ext = Path(original_filename).suffix.lower()
        
        if not custom_filename:
            custom_filename = str(uuid.uuid4())
        
        filename = f"{custom_filename}{file_ext}"
        file_path = self.dirs[category] / filename
        
        # Ensure unique filename
        counter = 1
        while file_path.exists():
            filename = f"{custom_filename}_{counter}{file_ext}"
            file_path = self.dirs[category] / filename
            counter += 1
        
        # Save file
        file_size = 0
        checksum = None
        
        try:
            # Read file in chunks and save
            async with aiofiles.open(file_path, 'wb') as f:
                if self.use_checksum:
                    hash_md5 = hashlib.md5()
                
                # Read file in chunks
                while True:
                    chunk = await upload_file.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    await f.write(chunk)
                    file_size += len(chunk)
                    
                    if self.use_checksum:
                        hash_md5.update(chunk)
                    
                    # Check file size limit
                    if file_size > self.max_file_size:
                        await f.close()
                        await aiofiles.os.remove(file_path)
                        raise ValueError(f"File size exceeds limit: {self.max_file_size} bytes")
                
                if self.use_checksum:
                    checksum = hash_md5.hexdigest()
        
        except Exception as e:
            # Clean up if error occurred
            if file_path.exists():
                await aiofiles.os.remove(file_path)
            raise
        
        finally:
            await upload_file.seek(0)  # Reset file pointer
        
        # Create file info
        file_info = {
            'id': str(uuid.uuid4()),
            'original_filename': original_filename,
            'filename': filename,
            'path': str(file_path),
            'category': category,
            'size': file_size,
            'content_type': upload_file.content_type or mimetypes.guess_type(filename)[0] or 'application/octet-stream',
            'checksum': checksum,
            'uploaded_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {},
            'accessed_at': datetime.utcnow().isoformat(),
            'download_count': 0
        }
        
        # Store metadata
        self.metadata[file_info['id']] = file_info
        self._save_metadata()
        
        # Update statistics
        self.stats['uploads'] += 1
        self.stats['total_size'] += file_size
        
        logger.info(f"File saved: {filename} ({file_size} bytes) in category: {category}")
        
        return file_info
    
    async def save_bytes(
        self,
        data: bytes,
        filename: str,
        category: str = "exports",
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Save bytes data to file
        
        Args:
            data: Bytes data to save
            filename: Output filename
            category: File category
            metadata: Additional file metadata
        
        Returns:
            Dictionary with file information
        """
        if category not in self.dirs:
            raise ValueError(f"Invalid category: {category}. Must be one of: {list(self.dirs.keys())}")
        
        # Validate file size
        if len(data) > self.max_file_size:
            raise ValueError(f"Data size exceeds limit: {self.max_file_size} bytes")
        
        file_path = self.dirs[category] / filename
        
        # Ensure unique filename
        counter = 1
        original_filename = filename
        while file_path.exists():
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            file_path = self.dirs[category] / filename
            counter += 1
        
        # Save file
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
        except Exception as e:
            logger.error(f"Failed to save bytes to file: {e}")
            raise
        
        # Compute checksum
        checksum = None
        if self.use_checksum:
            checksum = hashlib.md5(data).hexdigest()
        
        # Create file info
        file_info = {
            'id': str(uuid.uuid4()),
            'original_filename': filename,
            'filename': filename,
            'path': str(file_path),
            'category': category,
            'size': len(data),
            'content_type': mimetypes.guess_type(filename)[0] or 'application/octet-stream',
            'checksum': checksum,
            'uploaded_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {},
            'accessed_at': datetime.utcnow().isoformat(),
            'download_count': 0
        }
        
        # Store metadata
        self.metadata[file_info['id']] = file_info
        self._save_metadata()
        
        # Update statistics
        self.stats['uploads'] += 1
        self.stats['total_size'] += len(data)
        
        logger.info(f"Bytes saved to file: {filename} ({len(data)} bytes)")
        
        return file_info
    
    async def get_file(
        self,
        file_id: str,
        as_download: bool = False,
        custom_filename: Optional[str] = None
    ) -> Optional[FileResponse]:
        """
        Get file as FastAPI FileResponse
        
        Args:
            file_id: File ID from metadata
            as_download: Whether to force download
            custom_filename: Custom filename for download
        
        Returns:
            FileResponse or None if file not found
        """
        file_info = self.metadata.get(file_id)
        if not file_info:
            return None
        
        file_path = Path(file_info['path'])
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        # Update access metadata
        file_info['accessed_at'] = datetime.utcnow().isoformat()
        file_info['download_count'] = file_info.get('download_count', 0) + 1
        self._save_metadata()
        
        # Update statistics
        self.stats['downloads'] += 1
        
        # Determine filename
        filename = custom_filename or file_info['original_filename']
        
        # Create FileResponse
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=file_info['content_type'],
            headers={
                'Content-Disposition': f"{'attachment' if as_download else 'inline'}; filename=\"{filename}\"",
                'X-File-ID': file_id,
                'X-File-Size': str(file_info['size']),
                'X-File-Checksum': file_info.get('checksum', ''),
                'X-File-Uploaded': file_info['uploaded_at']
            }
        )
    
    async def get_file_stream(
        self,
        file_id: str,
        chunk_size: Optional[int] = None
    ) -> Optional[StreamingResponse]:
        """
        Get file as streaming response
        
        Args:
            file_id: File ID from metadata
            chunk_size: Chunk size for streaming
        
        Returns:
            StreamingResponse or None if file not found
        """
        file_info = self.metadata.get(file_id)
        if not file_info:
            return None
        
        file_path = Path(file_info['path'])
        if not file_path.exists():
            return None
        
        # Update access metadata
        file_info['accessed_at'] = datetime.utcnow().isoformat()
        file_info['download_count'] = file_info.get('download_count', 0) + 1
        self._save_metadata()
        
        # Update statistics
        self.stats['downloads'] += 1
        
        chunk_size = chunk_size or self.chunk_size
        
        async def file_stream():
            """Stream file in chunks"""
            async with aiofiles.open(file_path, 'rb') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        return StreamingResponse(
            file_stream(),
            media_type=file_info['content_type'],
            headers={
                'Content-Disposition': f'inline; filename="{file_info["original_filename"]}"',
                'X-File-ID': file_id,
                'X-File-Size': str(file_info['size']),
                'X-File-Checksum': file_info.get('checksum', ''),
                'X-File-Uploaded': file_info['uploaded_at']
            }
        )
    
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file information by ID"""
        file_info = self.metadata.get(file_id)
        if not file_info:
            return None
        
        # Check if file still exists
        file_path = Path(file_info['path'])
        if not file_path.exists():
            # Remove from metadata
            del self.metadata[file_id]
            self._save_metadata()
            return None
        
        # Update size if changed
        try:
            file_size = file_path.stat().st_size
            if file_size != file_info['size']:
                file_info['size'] = file_size
                self._save_metadata()
        except:
            pass
        
        return file_info
    
    async def delete_file(self, file_id: str) -> bool:
        """Delete file by ID"""
        file_info = self.metadata.get(file_id)
        if not file_info:
            return False
        
        file_path = Path(file_info['path'])
        
        try:
            # Delete file
            if file_path.exists():
                await aiofiles.os.remove(file_path)
            
            # Remove from metadata
            del self.metadata[file_id]
            self._save_metadata()
            
            # Update statistics
            self.stats['deletes'] += 1
            self.stats['total_size'] -= file_info['size']
            
            logger.info(f"File deleted: {file_info['filename']}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            self.stats['errors'] += 1
            return False
    
    async def list_files(
        self,
        category: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        min_age: Optional[int] = None,
        max_age: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List files with optional filtering"""
        filtered_files = []
        
        for file_info in self.metadata.values():
            # Apply filters
            if category and file_info['category'] != category:
                continue
            
            if min_size is not None and file_info['size'] < min_size:
                continue
            
            if max_size is not None and file_info['size'] > max_size:
                continue
            
            if min_age is not None or max_age is not None:
                uploaded_at = datetime.fromisoformat(file_info['uploaded_at'].replace('Z', '+00:00'))
                age = (datetime.utcnow() - uploaded_at).total_seconds()
                
                if min_age is not None and age < min_age:
                    continue
                
                if max_age is not None and age > max_age:
                    continue
            
            # Check if file exists
            file_path = Path(file_info['path'])
            if not file_path.exists():
                continue
            
            filtered_files.append(file_info)
        
        # Sort by upload time (newest first)
        filtered_files.sort(
            key=lambda x: datetime.fromisoformat(x['uploaded_at'].replace('Z', '+00:00')),
            reverse=True
        )
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        
        return filtered_files[start_idx:end_idx]
    
    async def cleanup_category(
        self,
        category: str,
        max_age: Optional[int] = None,
        max_count: Optional[int] = None
    ) -> int:
        """
        Clean up files in a category
        
        Args:
            category: Category to clean up
            max_age: Maximum file age in seconds
            max_count: Maximum number of files to keep
        
        Returns:
            Number of files deleted
        """
        if category not in self.dirs:
            raise ValueError(f"Invalid category: {category}")
        
        files = await self.list_files(category=category)
        
        # Sort by upload time (oldest first for age-based cleanup)
        files.sort(
            key=lambda x: datetime.fromisoformat(x['uploaded_at'].replace('Z', '+00:00'))
        )
        
        deleted_count = 0
        
        # Age-based cleanup
        if max_age is not None:
            cutoff_time = datetime.utcnow() - timedelta(seconds=max_age)
            
            for file_info in files[:]:  # Copy list for iteration
                uploaded_at = datetime.fromisoformat(file_info['uploaded_at'].replace('Z', '+00:00'))
                
                if uploaded_at < cutoff_time:
                    if await self.delete_file(file_info['id']):
                        deleted_count += 1
                        files.remove(file_info)
        
        # Count-based cleanup
        if max_count is not None and len(files) > max_count:
            # Keep only the newest max_count files
            files_to_delete = files[:-max_count] if max_count > 0 else files
            
            for file_info in files_to_delete:
                if await self.delete_file(file_info['id']):
                    deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} files from category: {category}")
        
        return deleted_count
    
    async def _cleanup_loop(self):
        """Periodic cleanup loop"""
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_temp_files()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = self.dirs['temp']
        
        if not temp_dir.exists():
            return
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.max_temp_age)
        deleted_count = 0
        
        try:
            for item in temp_dir.iterdir():
                try:
                    # Get file age
                    stat = await aiofiles.os.stat(item)
                    created_time = datetime.fromtimestamp(stat.st_ctime)
                    
                    if created_time < cutoff_time:
                        if item.is_file():
                            await aiofiles.os.remove(item)
                        elif item.is_dir():
                            shutil.rmtree(item)
                        
                        deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to clean up {item}: {e}")
        
        except Exception as e:
            logger.error(f"Temp cleanup error: {e}")
        
        if deleted_count > 0:
            logger.debug(f"Cleaned up {deleted_count} temporary files")
    
    async def _validate_file(self, upload_file: UploadFile):
        """Validate uploaded file"""
        # Check file size (approximate)
        if hasattr(upload_file.file, 'size'):
            if upload_file.file.size > self.max_file_size:
                raise ValueError(f"File size exceeds limit: {self.max_file_size} bytes")
        
        # Check filename
        if not upload_file.filename:
            raise ValueError("Filename is required")
        
        # Check extension
        file_ext = Path(upload_file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise ValueError(f"File extension not allowed: {file_ext}. Allowed: {self.allowed_extensions}")
        
        # Check MIME type
        content_type = upload_file.content_type
        if content_type and content_type not in self.allowed_mime_types:
            # Try to guess MIME type from extension
            guessed_type = mimetypes.guess_type(upload_file.filename)[0]
            if not guessed_type or guessed_type not in self.allowed_mime_types:
                raise ValueError(f"Content type not allowed: {content_type}. Allowed: {self.allowed_mime_types}")
    
    async def create_temp_file(
        self,
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        content: Optional[bytes] = None
    ) -> Tuple[str, Path]:
        """
        Create a temporary file
        
        Args:
            suffix: File suffix
            prefix: File prefix
            content: Initial content
        
        Returns:
            Tuple of (file_id, file_path)
        """
        temp_dir = self.dirs['temp']
        
        # Create unique filename
        file_id = str(uuid.uuid4())
        filename = f"{prefix or 'temp'}_{file_id}{suffix or '.tmp'}"
        file_path = temp_dir / filename
        
        # Write content if provided
        if content:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
        
        # Create metadata entry
        file_info = {
            'id': file_id,
            'original_filename': filename,
            'filename': filename,
            'path': str(file_path),
            'category': 'temp',
            'size': len(content) if content else 0,
            'content_type': 'application/octet-stream',
            'checksum': hashlib.md5(content).hexdigest() if content else None,
            'uploaded_at': datetime.utcnow().isoformat(),
            'metadata': {'temporary': True},
            'accessed_at': datetime.utcnow().isoformat(),
            'download_count': 0
        }
        
        self.metadata[file_id] = file_info
        self._save_metadata()
        
        logger.debug(f"Temporary file created: {filename}")
        
        return file_id, file_path
    
    async def compress_files(
        self,
        file_ids: List[str],
        output_filename: str,
        format: str = 'zip'
    ) -> Dict[str, Any]:
        """
        Compress multiple files into an archive
        
        Args:
            file_ids: List of file IDs to compress
            output_filename: Output archive filename
            format: Archive format ('zip' or 'tar')
        
        Returns:
            Dictionary with archive information
        """
        if format not in ['zip', 'tar']:
            raise ValueError(f"Unsupported format: {format}. Must be 'zip' or 'tar'")
        
        # Get file information
        files_to_compress = []
        total_size = 0
        
        for file_id in file_ids:
            file_info = await self.get_file_info(file_id)
            if not file_info:
                raise ValueError(f"File not found: {file_id}")
            
            files_to_compress.append(file_info)
            total_size += file_info['size']
        
        # Create output file path
        output_path = self.dirs['exports'] / output_filename
        
        # Ensure unique filename
        counter = 1
        original_output_path = output_path
        while output_path.exists():
            name, ext = os.path.splitext(output_filename)
            output_filename = f"{name}_{counter}{ext}"
            output_path = self.dirs['exports'] / output_filename
            counter += 1
        
        # Create archive
        try:
            if format == 'zip':
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_info in files_to_compress:
                        file_path = Path(file_info['path'])
                        if file_path.exists():
                            arcname = file_info['original_filename']
                            zipf.write(file_path, arcname)
            
            elif format == 'tar':
                with tarfile.open(output_path, 'w:gz') as tarf:
                    for file_info in files_to_compress:
                        file_path = Path(file_info['path'])
                        if file_path.exists():
                            arcname = file_info['original_filename']
                            tarf.add(file_path, arcname=arcname)
        
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            if output_path.exists():
                await aiofiles.os.remove(output_path)
            raise
        
        # Create archive metadata
        archive_info = await self.save_bytes(
            data=b'',  # Will be filled with actual size
            filename=output_filename,
            category='exports',
            metadata={
                'format': format,
                'file_count': len(files_to_compress),
                'total_size': total_size,
                'contents': [{
                    'id': f['id'],
                    'filename': f['original_filename'],
                    'size': f['size']
                } for f in files_to_compress]
            }
        )
        
        # Update actual file size
        archive_size = output_path.stat().st_size
        archive_info['size'] = archive_size
        self.metadata[archive_info['id']]['size'] = archive_size
        self._save_metadata()
        
        logger.info(f"Archive created: {output_filename} with {len(files_to_compress)} files")
        
        return archive_info
    
    async def save_point_cloud(
        self,
        point_cloud_data: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = 'ply'
    ) -> bool:
        """
        Save point cloud data to file
        
        Args:
            point_cloud_data: Dictionary with point cloud data
            output_path: Output file path
            format: Output format ('ply', 'xyz', 'pcd')
        
        Returns:
            True if successful, False otherwise
        """
        if format not in ['ply', 'xyz', 'pcd']:
            raise ValueError(f"Unsupported point cloud format: {format}")
        
        output_path = Path(output_path)
        
        try:
            if format == 'ply':
                await self._save_ply(point_cloud_data, output_path)
            elif format == 'xyz':
                await self._save_xyz(point_cloud_data, output_path)
            elif format == 'pcd':
                await self._save_pcd(point_cloud_data, output_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to save point cloud: {e}")
            return False
    
    async def _save_ply(self, data: Dict[str, Any], output_path: Path):
        """Save point cloud as PLY file"""
        points = data.get('points', [])
        colors = data.get('colors', [])
        normals = data.get('normals', [])
        
        num_points = len(points)
        
        # Create PLY header
        header = [
            'ply',
            'format ascii 1.0',
            f'element vertex {num_points}',
            'property float x',
            'property float y',
            'property float z'
        ]
        
        if colors and len(colors) == num_points:
            header.extend([
                'property uchar red',
                'property uchar green',
                'property uchar blue'
            ])
        
        if normals and len(normals) == num_points:
            header.extend([
                'property float nx',
                'property float ny',
                'property float nz'
            ])
        
        header.extend([
            'end_header',
            ''
        ])
        
        # Write file
        async with aiofiles.open(output_path, 'w') as f:
            # Write header
            await f.write('\n'.join(header))
            
            # Write points
            for i in range(num_points):
                point = points[i]
                line = f"{point[0]} {point[1]} {point[2]}"
                
                if colors and len(colors) == num_points:
                    color = colors[i]
                    # Convert 0-1 float to 0-255 integer
                    r = int(color[0] * 255) if len(color) > 0 else 0
                    g = int(color[1] * 255) if len(color) > 1 else 0
                    b = int(color[2] * 255) if len(color) > 2 else 0
                    line += f" {r} {g} {b}"
                
                if normals and len(normals) == num_points:
                    normal = normals[i]
                    line += f" {normal[0]} {normal[1]} {normal[2]}"
                
                await f.write(line + '\n')
    
    async def _save_xyz(self, data: Dict[str, Any], output_path: Path):
        """Save point cloud as XYZ file"""
        points = data.get('points', [])
        
        async with aiofiles.open(output_path, 'w') as f:
            for point in points:
                line = f"{point[0]} {point[1]} {point[2]}"
                
                # XYZ format can include RGB as additional columns
                if 'colors' in data:
                    idx = points.index(point)
                    colors = data['colors']
                    if idx < len(colors):
                        color = colors[idx]
                        if len(color) >= 3:
                            line += f" {color[0]} {color[1]} {color[2]}"
                
                await f.write(line + '\n')
    
    async def _save_pcd(self, data: Dict[str, Any], output_path: Path):
        """Save point cloud as PCD file"""
        points = data.get('points', [])
        colors = data.get('colors', [])
        normals = data.get('normals', [])
        
        num_points = len(points)
        
        # Create PCD header
        header = [
            '# .PCD v0.7 - Point Cloud Data file format',
            'VERSION 0.7',
            f'FIELDS x y z'
        ]
        
        has_color = colors and len(colors) == num_points
        has_normal = normals and len(normals) == num_points
        
        if has_color:
            header[2] += ' rgb'
        
        if has_normal:
            header[2] += ' normal_x normal_y normal_z'
        
        header.extend([
            'SIZE 4 4 4' + (' 4' if has_color else '') + (' 4 4 4' if has_normal else ''),
            'TYPE F F F' + (' F' if has_color else '') + (' F F F' if has_normal else ''),
            'COUNT 1 1 1' + (' 1' if has_color else '') + (' 1 1 1' if has_normal else ''),
            f'WIDTH {num_points}',
            'HEIGHT 1',
            'VIEWPOINT 0 0 0 1 0 0 0',
            f'POINTS {num_points}',
            'DATA ascii',
            ''
        ])
        
        # Write file
        async with aiofiles.open(output_path, 'w') as f:
            # Write header
            await f.write('\n'.join(header))
            
            # Write points
            for i in range(num_points):
                point = points[i]
                line = f"{point[0]} {point[1]} {point[2]}"
                
                if has_color:
                    color = colors[i]
                    # Pack RGB into float
                    if len(color) >= 3:
                        r = int(color[0] * 255)
                        g = int(color[1] * 255)
                        b = int(color[2] * 255)
                        rgb_packed = (r << 16) | (g << 8) | b
                        line += f" {rgb_packed}"
                    else:
                        line += " 0"
                
                if has_normal:
                    normal = normals[i]
                    line += f" {normal[0]} {normal[1]} {normal[2]}"
                
                await f.write(line + '\n')
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get file handler statistics"""
        # Count files by category
        files_by_category = {}
        for file_info in self.metadata.values():
            category = file_info['category']
            files_by_category[category] = files_by_category.get(category, 0) + 1
        
        # Calculate disk usage
        disk_usage = {}
        for category, dir_path in self.dirs.items():
            if dir_path.exists():
                total_size = 0
                file_count = 0
                
                for item in dir_path.rglob('*'):
                    if item.is_file():
                        try:
                            total_size += item.stat().st_size
                            file_count += 1
                        except:
                            pass
                
                disk_usage[category] = {
                    'size': total_size,
                    'file_count': file_count
                }
        
        return {
            **self.stats,
            'files_by_category': files_by_category,
            'disk_usage': disk_usage,
            'metadata_count': len(self.metadata),
            'max_file_size': self.max_file_size,
            'allowed_extensions': self.allowed_extensions,
            'base_path': str(self.base_path),
            'timestamp': datetime.utcnow().isoformat()
        }


# Utility functions for file operations
async def read_file_chunks(file_path: Path, chunk_size: int = 8192):
    """Read file in chunks asynchronously"""
    async with aiofiles.open(file_path, 'rb') as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk


async def compute_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """Compute file hash asynchronously"""
    hash_func = hashlib.new(algorithm)
    
    async with aiofiles.open(file_path, 'rb') as f:
        while True:
            chunk = await f.read(8192)
            if not chunk:
                break
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


async def safe_remove(file_path: Path, max_attempts: int = 3) -> bool:
    """Safely remove file with retry"""
    for attempt in range(max_attempts):
        try:
            await aiofiles.os.remove(file_path)
            return True
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"Failed to remove file {file_path} after {max_attempts} attempts: {e}")
                return False
            await asyncio.sleep(0.1 * (attempt + 1))
    
    return False


async def create_directory_structure(base_path: Path, structure: Dict[str, Any]):
    """Create directory structure asynchronously"""
    for name, content in structure.items():
        path = base_path / name
        
        if isinstance(content, dict):
            # Directory
            await aiofiles.os.makedirs(path, exist_ok=True)
            await create_directory_structure(path, content)
        else:
            # File with content
            await aiofiles.os.makedirs(path.parent, exist_ok=True)
            async with aiofiles.open(path, 'w') as f:
                await f.write(str(content))


# Export
__all__ = [
    'FileHandler',
    'read_file_chunks',
    'compute_file_hash',
    'safe_remove',
    'create_directory_structure'
]