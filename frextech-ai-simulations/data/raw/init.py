"""
Raw Data Module
Handling and management of raw datasets before preprocessing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from ..utils.file_io import (
    ensure_directory, list_files, find_files,
    get_file_size, calculate_md5, copy_file,
    move_file, delete_file, compress_file,
    decompress_file, get_file_info
)
from ..utils.data_validator import DataValidator, validate_data_file
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class RawDataManager:
    """
    Manager for raw data operations.
    
    Features:
    - Raw data organization
    - Data validation and verification
    - Data ingestion from external sources
    - Metadata management
    - Data versioning
    """
    
    def __init__(self, data_root: Union[str, Path] = "./data/raw"):
        """
        Initialize raw data manager.
        
        Args:
            data_root: Root directory for raw data
        """
        self.data_root = Path(data_root)
        ensure_directory(self.data_root)
        
        # Metadata file
        self.metadata_file = self.data_root / "metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized RawDataManager at {self.data_root}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {}
        else:
            return {
                'datasets': {},
                'last_updated': None,
                'total_size_bytes': 0,
                'file_count': 0
            }
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.debug(f"Saved metadata to {self.metadata_file}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def add_dataset(
        self,
        name: str,
        source_path: Union[str, Path],
        description: str = "",
        tags: List[str] = None,
        overwrite: bool = False
    ) -> bool:
        """
        Add a dataset to raw data storage.
        
        Args:
            name: Dataset name
            source_path: Path to source data
            description: Dataset description
            tags: List of tags
            overwrite: Whether to overwrite existing dataset
            
        Returns:
            True if successful, False otherwise
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            logger.error(f"Source path does not exist: {source_path}")
            return False
        
        # Check if dataset already exists
        if name in self.metadata['datasets'] and not overwrite:
            logger.warning(f"Dataset '{name}' already exists. Use overwrite=True to replace.")
            return False
        
        # Create dataset directory
        dataset_dir = self.data_root / name
        if dataset_dir.exists() and overwrite:
            # Clean up existing directory
            import shutil
            shutil.rmtree(dataset_dir)
        
        ensure_directory(dataset_dir)
        
        # Copy or move data
        if source_path.is_file():
            # Single file
            dest_path = dataset_dir / source_path.name
            if not copy_file(source_path, dest_path, overwrite=True):
                return False
            
            file_count = 1
            total_size = get_file_size(dest_path)
            
        elif source_path.is_dir():
            # Directory
            import shutil
            try:
                # Copy directory contents
                for item in source_path.iterdir():
                    dest_item = dataset_dir / item.name
                    if item.is_file():
                        shutil.copy2(item, dest_item)
                    elif item.is_dir():
                        shutil.copytree(item, dest_item)
                
                # Count files and calculate size
                file_count = sum(1 for _ in dataset_dir.rglob('*') if _.is_file())
                total_size = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
            
            except Exception as e:
                logger.error(f"Failed to copy directory: {e}")
                return False
        
        else:
            logger.error(f"Invalid source path type: {source_path}")
            return False
        
        # Update metadata
        self.metadata['datasets'][name] = {
            'name': name,
            'description': description,
            'tags': tags or [],
            'path': str(dataset_dir.relative_to(self.data_root)),
            'file_count': file_count,
            'total_size_bytes': total_size,
            'added_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'files': self._index_files(dataset_dir)
        }
        
        # Update global stats
        self._update_global_stats()
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Added dataset '{name}' with {file_count} files ({total_size} bytes)")
        return True
    
    def _index_files(self, directory: Path) -> List[Dict[str, Any]]:
        """Create index of files in directory."""
        files = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                files.append({
                    'path': str(file_path.relative_to(directory)),
                    'size_bytes': file_path.stat().st_size,
                    'md5': calculate_md5(file_path),
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })
        
        return files
    
    def _update_global_stats(self):
        """Update global statistics."""
        total_size = 0
        total_files = 0
        
        for dataset_info in self.metadata['datasets'].values():
            total_size += dataset_info['total_size_bytes']
            total_files += dataset_info['file_count']
        
        self.metadata['total_size_bytes'] = total_size
        self.metadata['file_count'] = total_files
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets."""
        datasets = []
        
        for name, info in self.metadata['datasets'].items():
            datasets.append({
                'name': name,
                'description': info.get('description', ''),
                'tags': info.get('tags', []),
                'file_count': info.get('file_count', 0),
                'size_bytes': info.get('total_size_bytes', 0),
                'added_date': info.get('added_date'),
                'last_updated': info.get('last_updated')
            })
        
        return datasets
    
    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a dataset."""
        return self.metadata['datasets'].get(name)
    
    def remove_dataset(self, name: str, keep_files: bool = False) -> bool:
        """
        Remove a dataset.
        
        Args:
            name: Dataset name
            keep_files: Whether to keep the data files
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.metadata['datasets']:
            logger.warning(f"Dataset '{name}' not found")
            return False
        
        # Remove files if requested
        if not keep_files:
            dataset_info = self.metadata['datasets'][name]
            dataset_path = self.data_root / dataset_info['path']
            
            if dataset_path.exists():
                import shutil
                try:
                    shutil.rmtree(dataset_path)
                    logger.info(f"Removed dataset files: {dataset_path}")
                except Exception as e:
                    logger.error(f"Failed to remove dataset files: {e}")
                    # Continue to remove metadata even if file removal fails
        
        # Remove from metadata
        del self.metadata['datasets'][name]
        
        # Update global stats
        self._update_global_stats()
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Removed dataset '{name}' from metadata")
        return True
    
    def validate_dataset(self, name: str, validation_level: str = "standard") -> Dict[str, Any]:
        """
        Validate a dataset.
        
        Args:
            name: Dataset name
            validation_level: Validation level
            
        Returns:
            Validation report
        """
        if name not in self.metadata['datasets']:
            logger.error(f"Dataset '{name}' not found")
            return {'valid': False, 'error': 'Dataset not found'}
        
        dataset_info = self.metadata['datasets'][name]
        dataset_path = self.data_root / dataset_info['path']
        
        # Create validator
        validator = DataValidator({'validation_level': validation_level})
        
        # Validate
        result = validator.validate(dataset_path)
        
        # Create report
        report = {
            'dataset': name,
            'validation_level': validation_level,
            'result': result.to_dict(),
            'dataset_info': dataset_info
        }
        
        return report
    
    def search_datasets(self, query: str, search_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search datasets by query.
        
        Args:
            query: Search query
            search_fields: Fields to search in
            
        Returns:
            List of matching datasets
        """
        if search_fields is None:
            search_fields = ['name', 'description', 'tags']
        
        query_lower = query.lower()
        results = []
        
        for name, info in self.metadata['datasets'].items():
            match_found = False
            
            for field in search_fields:
                if field == 'name':
                    if query_lower in name.lower():
                        match_found = True
                        break
                elif field == 'description':
                    description = info.get('description', '').lower()
                    if query_lower in description:
                        match_found = True
                        break
                elif field == 'tags':
                    tags = [tag.lower() for tag in info.get('tags', [])]
                    if any(query_lower in tag for tag in tags):
                        match_found = True
                        break
            
            if match_found:
                results.append({
                    'name': name,
                    'description': info.get('description', ''),
                    'tags': info.get('tags', []),
                    'file_count': info.get('file_count', 0),
                    'size_bytes': info.get('total_size_bytes', 0)
                })
        
        return results
    
    def export_dataset(self, name: str, output_path: Union[str, Path], 
                      format: str = "directory") -> bool:
        """
        Export dataset to external location.
        
        Args:
            name: Dataset name
            output_path: Output path
            format: Export format ('directory', 'zip', 'tar')
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.metadata['datasets']:
            logger.error(f"Dataset '{name}' not found")
            return False
        
        dataset_info = self.metadata['datasets'][name]
        dataset_path = self.data_root / dataset_info['path']
        
        output_path = Path(output_path)
        
        try:
            if format == "directory":
                # Copy directory
                import shutil
                if output_path.exists():
                    shutil.rmtree(output_path)
                shutil.copytree(dataset_path, output_path)
            
            elif format == "zip":
                # Create zip archive
                import shutil
                shutil.make_archive(
                    str(output_path.with_suffix('')),  # Remove .zip suffix
                    'zip',
                    dataset_path
                )
            
            elif format == "tar":
                # Create tar archive
                import tarfile
                with tarfile.open(output_path.with_suffix('.tar.gz'), 'w:gz') as tar:
                    tar.add(dataset_path, arcname=name)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported dataset '{name}' to {output_path} in {format} format")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export dataset '{name}': {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get raw data storage statistics."""
        return {
            'total_datasets': len(self.metadata['datasets']),
            'total_files': self.metadata.get('file_count', 0),
            'total_size_bytes': self.metadata.get('total_size_bytes', 0),
            'last_updated': self.metadata.get('last_updated')
        }


# Convenience functions

def get_raw_data_manager(data_root: Optional[Union[str, Path]] = None) -> RawDataManager:
    """Get or create raw data manager."""
    if data_root is None:
        data_root = Path("./data/raw")
    
    return RawDataManager(data_root)


def import_raw_data(
    source_path: Union[str, Path],
    dataset_name: str,
    data_root: Optional[Union[str, Path]] = None,
    **kwargs
) -> bool:
    """
    Import raw data into storage.
    
    Args:
        source_path: Path to source data
        dataset_name: Name for the dataset
        data_root: Raw data root directory
        **kwargs: Additional arguments for add_dataset
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_raw_data_manager(data_root)
    return manager.add_dataset(dataset_name, source_path, **kwargs)


def validate_raw_data(
    dataset_name: str,
    data_root: Optional[Union[str, Path]] = None,
    validation_level: str = "standard"
) -> Dict[str, Any]:
    """
    Validate raw dataset.
    
    Args:
        dataset_name: Dataset name
        data_root: Raw data root directory
        validation_level: Validation level
        
    Returns:
        Validation report
    """
    manager = get_raw_data_manager(data_root)
    return manager.validate_dataset(dataset_name, validation_level)