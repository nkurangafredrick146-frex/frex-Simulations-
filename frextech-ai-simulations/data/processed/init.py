"""
Processed Data Module
Handling and management of processed/preprocessed datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
from datetime import datetime

from ..utils.file_io import (
    ensure_directory, list_files, get_file_size,
    calculate_md5, copy_file, read_json, write_json
)
from ..utils.logging_config import get_logger
from ..utils.data_validator import DataValidator

logger = get_logger(__name__)

class ProcessedDataManager:
    """
    Manager for processed data operations.
    
    Features:
    - Processed data organization
    - Data versioning and lineage tracking
    - Quality assurance
    - Data splitting and sampling
    - Caching and optimization
    """
    
    def __init__(self, data_root: Union[str, Path] = "./data/processed"):
        """
        Initialize processed data manager.
        
        Args:
            data_root: Root directory for processed data
        """
        self.data_root = Path(data_root)
        ensure_directory(self.data_root)
        
        # Metadata and lineage tracking
        self.metadata_file = self.data_root / "metadata.json"
        self.lineage_file = self.data_root / "lineage.json"
        
        self.metadata = self._load_metadata()
        self.lineage = self._load_lineage()
        
        logger.info(f"Initialized ProcessedDataManager at {self.data_root}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            try:
                return read_json(self.metadata_file)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                return {'datasets': {}, 'versions': {}}
        else:
            return {'datasets': {}, 'versions': {}}
    
    def _load_lineage(self) -> Dict[str, Any]:
        """Load lineage data from file."""
        if self.lineage_file.exists():
            try:
                return read_json(self.lineage_file)
            except Exception as e:
                logger.warning(f"Failed to load lineage: {e}")
                return {'transformations': []}
        else:
            return {'transformations': []}
    
    def _save_metadata(self):
        """Save metadata to file."""
        write_json(self.metadata, self.metadata_file)
    
    def _save_lineage(self):
        """Save lineage data to file."""
        write_json(self.lineage, self.lineage_file)
    
    def create_dataset(
        self,
        name: str,
        raw_dataset: str,
        processing_pipeline: Dict[str, Any],
        description: str = "",
        tags: List[str] = None,
        version: str = "1.0.0",
        overwrite: bool = False
    ) -> str:
        """
        Create a processed dataset.
        
        Args:
            name: Dataset name
            raw_dataset: Source raw dataset name
            processing_pipeline: Processing pipeline configuration
            description: Dataset description
            tags: List of tags
            version: Version string
            overwrite: Whether to overwrite existing version
            
        Returns:
            Dataset version ID
        """
        # Create version ID
        version_id = f"{name}_v{version}"
        
        # Check if version already exists
        if version_id in self.metadata['versions'] and not overwrite:
            logger.warning(f"Dataset version '{version_id}' already exists. Use overwrite=True to replace.")
            return version_id
        
        # Create dataset directory
        dataset_dir = self.data_root / name / f"v{version}"
        if dataset_dir.exists() and overwrite:
            import shutil
            shutil.rmtree(dataset_dir)
        
        ensure_directory(dataset_dir)
        
        # Store processing configuration
        pipeline_file = dataset_dir / "processing_pipeline.json"
        write_json(processing_pipeline, pipeline_file)
        
        # Create dataset metadata
        dataset_metadata = {
            'name': name,
            'version': version,
            'version_id': version_id,
            'raw_dataset': raw_dataset,
            'description': description,
            'tags': tags or [],
            'processing_pipeline': str(pipeline_file.relative_to(self.data_root)),
            'created_date': datetime.now().isoformat(),
            'dataset_path': str(dataset_dir.relative_to(self.data_root)),
            'status': 'created'
        }
        
        # Update metadata
        if name not in self.metadata['datasets']:
            self.metadata['datasets'][name] = {
                'name': name,
                'description': description,
                'tags': tags or [],
                'versions': []
            }
        
        # Add version to dataset
        self.metadata['datasets'][name]['versions'].append(version_id)
        
        # Add version metadata
        self.metadata['versions'][version_id] = dataset_metadata
        
        # Add to lineage
        self.lineage['transformations'].append({
            'timestamp': datetime.now().isoformat(),
            'operation': 'create_dataset',
            'source': raw_dataset,
            'target': version_id,
            'pipeline': processing_pipeline
        })
        
        # Save metadata and lineage
        self._save_metadata()
        self._save_lineage()
        
        logger.info(f"Created processed dataset '{version_id}'")
        return version_id
    
    def add_data(
        self,
        version_id: str,
        data_type: str,
        data: Any,
        file_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add data to processed dataset.
        
        Args:
            version_id: Dataset version ID
            data_type: Type of data ('images', 'labels', 'features', 'embeddings', 'splits')
            data: Data to add
            file_name: Output file name
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if version_id not in self.metadata['versions']:
            logger.error(f"Dataset version '{version_id}' not found")
            return False
        
        version_info = self.metadata['versions'][version_id]
        dataset_dir = self.data_root / version_info['dataset_path']
        
        # Create data type directory
        data_dir = dataset_dir / data_type
        ensure_directory(data_dir)
        
        # Save data based on type
        file_path = data_dir / file_name
        
        try:
            from ..utils.file_io import (
                save_numpy, save_tensor, write_json,
                write_pickle, save_image, save_video
            )
            
            # Determine how to save based on data type and file extension
            suffix = file_path.suffix.lower()
            
            if suffix == '.npy' or isinstance(data, np.ndarray):
                save_numpy(data if isinstance(data, np.ndarray) else np.array(data), file_path)
            
            elif suffix in ['.pt', '.pth'] or isinstance(data, torch.Tensor):
                save_tensor(data if isinstance(data, torch.Tensor) else torch.tensor(data), file_path)
            
            elif suffix == '.json':
                write_json(data, file_path)
            
            elif suffix in ['.pkl', '.pickle']:
                write_pickle(data, file_path)
            
            elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                save_image(data, file_path)
            
            elif suffix in ['.mp4', '.avi', '.mov']:
                save_video(data, file_path)
            
            else:
                # Default to pickle
                write_pickle(data, file_path)
            
            # Update dataset metadata
            if 'data_files' not in version_info:
                version_info['data_files'] = {}
            
            if data_type not in version_info['data_files']:
                version_info['data_files'][data_type] = []
            
            file_info = {
                'file_name': file_name,
                'path': str(file_path.relative_to(self.data_root)),
                'size_bytes': get_file_size(file_path),
                'md5': calculate_md5(file_path),
                'added_date': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            version_info['data_files'][data_type].append(file_info)
            
            # Add to lineage
            self.lineage['transformations'].append({
                'timestamp': datetime.now().isoformat(),
                'operation': 'add_data',
                'dataset': version_id,
                'data_type': data_type,
                'file_name': file_name,
                'metadata': metadata
            })
            
            # Save metadata and lineage
            self._save_metadata()
            self._save_lineage()
            
            logger.debug(f"Added data to '{version_id}': {data_type}/{file_name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add data to '{version_id}': {e}")
            return False
    
    def get_dataset(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processed dataset information.
        
        Args:
            version_id: Dataset version ID
            
        Returns:
            Dataset metadata or None if not found
        """
        return self.metadata['versions'].get(version_id)
    
    def list_datasets(self, name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all processed datasets.
        
        Args:
            name_filter: Optional name filter
            
        Returns:
            List of dataset information
        """
        datasets = []
        
        for version_id, version_info in self.metadata['versions'].items():
            if name_filter and name_filter not in version_id:
                continue
            
            datasets.append({
                'version_id': version_id,
                'name': version_info['name'],
                'version': version_info['version'],
                'raw_dataset': version_info['raw_dataset'],
                'description': version_info.get('description', ''),
                'tags': version_info.get('tags', []),
                'created_date': version_info.get('created_date'),
                'status': version_info.get('status', 'unknown'),
                'data_types': list(version_info.get('data_files', {}).keys())
            })
        
        return datasets
    
    def load_data(
        self,
        version_id: str,
        data_type: str,
        file_name: Optional[str] = None,
        index: Optional[int] = None
    ) -> Any:
        """
        Load data from processed dataset.
        
        Args:
            version_id: Dataset version ID
            data_type: Type of data to load
            file_name: Specific file to load (None for all files of type)
            index: Index of file to load (if file_name not specified)
            
        Returns:
            Loaded data
        """
        if version_id not in self.metadata['versions']:
            logger.error(f"Dataset version '{version_id}' not found")
            return None
        
        version_info = self.metadata['versions'][version_id]
        
        if data_type not in version_info.get('data_files', {}):
            logger.error(f"Data type '{data_type}' not found in dataset '{version_id}'")
            return None
        
        files = version_info['data_files'][data_type]
        
        if not files:
            logger.error(f"No files found for data type '{data_type}'")
            return None
        
        # Determine which file to load
        if file_name:
            # Load specific file
            file_info = next((f for f in files if f['file_name'] == file_name), None)
            if not file_info:
                logger.error(f"File '{file_name}' not found in data type '{data_type}'")
                return None
            file_path = self.data_root / file_info['path']
        
        elif index is not None:
            # Load by index
            if index < 0 or index >= len(files):
                logger.error(f"Index {index} out of range for {len(files)} files")
                return None
            file_info = files[index]
            file_path = self.data_root / file_info['path']
        
        else:
            # Load all files
            all_data = []
            for file_info in files:
                file_path = self.data_root / file_info['path']
                data = self._load_file(file_path)
                if data is not None:
                    all_data.append(data)
            
            return all_data
        
        # Load single file
        return self._load_file(file_path)
    
    def _load_file(self, file_path: Path) -> Any:
        """Load a file based on its extension."""
        try:
            from ..utils.file_io import (
                load_numpy, load_tensor, read_json,
                read_pickle, load_image, load_video
            )
            
            suffix = file_path.suffix.lower()
            
            if suffix == '.npy':
                return load_numpy(file_path)
            elif suffix in ['.pt', '.pth']:
                return load_tensor(file_path, map_location='cpu')
            elif suffix == '.json':
                return read_json(file_path)
            elif suffix in ['.pkl', '.pickle']:
                return read_pickle(file_path)
            elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return load_image(file_path)
            elif suffix in ['.mp4', '.avi', '.mov']:
                return load_video(file_path)
            else:
                # Try pickle as default
                return read_pickle(file_path)
        
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return None
    
    def create_data_split(
        self,
        version_id: str,
        split_name: str,
        split_config: Dict[str, Any],
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a data split for a dataset.
        
        Args:
            version_id: Dataset version ID
            split_name: Name of the split
            split_config: Split configuration
            train_indices: Training set indices
            val_indices: Validation set indices
            test_indices: Test set indices
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        split_data = {
            'split_name': split_name,
            'split_config': split_config,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'created_date': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        return self.add_data(
            version_id=version_id,
            data_type='splits',
            data=split_data,
            file_name=f"{split_name}_split.json",
            metadata=metadata
        )
    
    def get_split(
        self,
        version_id: str,
        split_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a data split for a dataset.
        
        Args:
            version_id: Dataset version ID
            split_name: Name of the split
            
        Returns:
            Split data or None if not found
        """
        splits = self.load_data(version_id, 'splits')
        
        if not splits:
            return None
        
        if isinstance(splits, list):
            # Find split by name
            for split in splits:
                if split.get('split_name') == split_name:
                    return split
        
        elif isinstance(splits, dict) and splits.get('split_name') == split_name:
            return splits
        
        return None
    
    def validate_dataset(self, version_id: str, validation_level: str = "standard") -> Dict[str, Any]:
        """
        Validate a processed dataset.
        
        Args:
            version_id: Dataset version ID
            validation_level: Validation level
            
        Returns:
            Validation report
        """
        if version_id not in self.metadata['versions']:
            return {'valid': False, 'error': f"Dataset '{version_id}' not found"}
        
        version_info = self.metadata['versions'][version_id]
        dataset_dir = self.data_root / version_info['dataset_path']
        
        # Create validator
        validator = DataValidator({'validation_level': validation_level})
        
        # Validate dataset structure
        result = validator.validate(dataset_dir)
        
        # Additional checks for processed data
        additional_checks = []
        
        # Check if required files exist
        required_files = ['processing_pipeline.json']
        for req_file in required_files:
            req_path = dataset_dir / req_file
            if not req_path.exists():
                additional_checks.append(f"Missing required file: {req_file}")
        
        # Check if any data files exist
        data_files_found = False
        for data_dir in dataset_dir.iterdir():
            if data_dir.is_dir():
                if any(data_dir.iterdir()):
                    data_files_found = True
                    break
        
        if not data_files_found:
            additional_checks.append("No data files found in dataset")
        
        # Create report
        report = {
            'dataset': version_id,
            'validation_level': validation_level,
            'basic_validation': result.to_dict(),
            'additional_checks': additional_checks,
            'valid': result.valid and len(additional_checks) == 0
        }
        
        return report
    
    def export_dataset(
        self,
        version_id: str,
        output_path: Union[str, Path],
        format: str = "directory",
        include_splits: bool = True
    ) -> bool:
        """
        Export processed dataset.
        
        Args:
            version_id: Dataset version ID
            output_path: Output path
            format: Export format ('directory', 'zip', 'tar')
            include_splits: Whether to include split information
            
        Returns:
            True if successful, False otherwise
        """
        if version_id not in self.metadata['versions']:
            logger.error(f"Dataset version '{version_id}' not found")
            return False
        
        version_info = self.metadata['versions'][version_id]
        dataset_dir = self.data_root / version_info['dataset_path']
        
        output_path = Path(output_path)
        
        try:
            if format == "directory":
                import shutil
                if output_path.exists():
                    shutil.rmtree(output_path)
                shutil.copytree(dataset_dir, output_path)
            
            elif format == "zip":
                import shutil
                shutil.make_archive(
                    str(output_path.with_suffix('')),  # Remove .zip suffix
                    'zip',
                    dataset_dir
                )
            
            elif format == "tar":
                import tarfile
                with tarfile.open(output_path.with_suffix('.tar.gz'), 'w:gz') as tar:
                    tar.add(dataset_dir, arcname=version_id)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported dataset '{version_id}' to {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to export dataset '{version_id}': {e}")
            return False
    
    def get_lineage(self, version_id: str) -> List[Dict[str, Any]]:
        """
        Get lineage information for a dataset.
        
        Args:
            version_id: Dataset version ID
            
        Returns:
            List of lineage transformations
        """
        transformations = []
        
        for transform in self.lineage.get('transformations', []):
            if transform.get('target') == version_id or transform.get('dataset') == version_id:
                transformations.append(transform)
        
        return transformations


# Convenience functions

def get_processed_data_manager(data_root: Optional[Union[str, Path]] = None) -> ProcessedDataManager:
    """Get or create processed data manager."""
    if data_root is None:
        data_root = Path("./data/processed")
    
    return ProcessedDataManager(data_root)


def create_processed_dataset(
    name: str,
    raw_dataset: str,
    processing_config: Dict[str, Any],
    data_root: Optional[Union[str, Path]] = None,
    **kwargs
) -> str:
    """
    Create a processed dataset.
    
    Args:
        name: Dataset name
        raw_dataset: Source raw dataset
        processing_config: Processing configuration
        data_root: Processed data root directory
        **kwargs: Additional arguments for create_dataset
        
    Returns:
        Dataset version ID
    """
    manager = get_processed_data_manager(data_root)
    return manager.create_dataset(name, raw_dataset, processing_config, **kwargs)


def load_processed_data(
    version_id: str,
    data_type: str,
    data_root: Optional[Union[str, Path]] = None,
    **kwargs
) -> Any:
    """
    Load data from processed dataset.
    
    Args:
        version_id: Dataset version ID
        data_type: Type of data to load
        data_root: Processed data root directory
        **kwargs: Additional arguments for load_data
        
    Returns:
        Loaded data
    """
    manager = get_processed_data_manager(data_root)
    return manager.load_data(version_id, data_type, **kwargs)