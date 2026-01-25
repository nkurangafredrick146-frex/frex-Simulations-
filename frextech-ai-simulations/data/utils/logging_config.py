"""
Logging Configuration Module
Centralized logging setup for data processing pipeline.
"""

import os
import sys
import logging
import logging.config
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import time
from datetime import datetime
from contextlib import contextmanager

# Default logging configuration
DEFAULT_LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '[%(levelname)s] %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': 'logs/data_processing.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'verbose',
            'filename': 'logs/errors.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf8'
        }
    },
    'loggers': {
        'data': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False
        },
        'data.datasets': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'data.utils': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}


def setup_logger(
    name: str = 'data',
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    error_file: Optional[str] = None,
    console: bool = True,
    json_format: bool = False,
    propagate: bool = False
) -> logging.Logger:
    """
    Set up a logger with specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (if None, no file logging)
        error_file: Path to error log file
        console: Whether to log to console
        json_format: Whether to use JSON format
        propagate: Whether to propagate to parent loggers
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if error_file:
        error_path = Path(error_file)
        error_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create custom config
    config = DEFAULT_LOG_CONFIG.copy()
    
    # Update formatter based on json_format
    formatter_name = 'json' if json_format else 'verbose'
    
    # Update handlers
    handlers = []
    
    if console:
        config['handlers']['console']['formatter'] = formatter_name
        handlers.append('console')
    
    if log_file:
        config['handlers']['file']['filename'] = log_file
        config['handlers']['file']['formatter'] = formatter_name
        handlers.append('file')
    
    if error_file:
        config['handlers']['error_file']['filename'] = error_file
        config['handlers']['error_file']['formatter'] = formatter_name
        handlers.append('error_file')
    
    # Update logger config
    if name not in config['loggers']:
        config['loggers'][name] = {
            'level': level if isinstance(level, str) else logging.getLevelName(level),
            'handlers': handlers,
            'propagate': propagate
        }
    else:
        config['loggers'][name]['level'] = level if isinstance(level, str) else logging.getLevelName(level)
        config['loggers'][name]['handlers'] = handlers
        config['loggers'][name]['propagate'] = propagate
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get logger
    logger = logging.getLogger(name)
    
    # Set level
    if isinstance(level, str):
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(level)
    
    return logger


def get_logger(name: str = 'data') -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class DataLogger:
    """
    Enhanced logger for data processing with additional features.
    
    Features:
    - Structured logging
    - Performance timing
    - Data validation logging
    - Progress tracking
    - Statistics collection
    """
    
    def __init__(self, name: str = 'data.processor', **kwargs):
        """
        Initialize data logger.
        
        Args:
            name: Logger name
            **kwargs: Additional logger configuration
        """
        self.logger = setup_logger(name, **kwargs)
        self.name = name
        self.metrics = {}
        self.timers = {}
        
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        self.metrics[name] = value
        
        if step is not None:
            self.logger.info(f"Metric [{name}] at step {step}: {value:.6f}")
        else:
            self.logger.info(f"Metric [{name}]: {value:.6f}")
    
    def start_timer(self, name: str):
        """Start a timer."""
        self.timers[name] = time.time()
        self.logger.debug(f"Timer [{name}] started")
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.timers:
            self.logger.warning(f"Timer [{name}] not found")
            return 0.0
        
        elapsed = time.time() - self.timers[name]
        del self.timers[name]
        
        self.logger.debug(f"Timer [{name}] stopped: {elapsed:.4f}s")
        return elapsed
    
    def log_data_info(self, data_path: str, data_type: str, count: int, size_bytes: int):
        """Log information about data."""
        from .file_io import format_file_size
        
        size_str = format_file_size(size_bytes)
        
        self.logger.info(f"Data Info - Path: {data_path}")
        self.logger.info(f"  Type: {data_type}")
        self.logger.info(f"  Count: {count}")
        self.logger.info(f"  Size: {size_str}")
    
    def log_validation_result(self, result: Dict[str, Any]):
        """Log data validation results."""
        if result.get('valid', False):
            self.logger.info(f"Validation PASSED: {result.get('message', 'No issues found')}")
        else:
            self.logger.error(f"Validation FAILED: {result.get('message', 'Validation failed')}")
            
            errors = result.get('errors', [])
            for error in errors:
                self.logger.error(f"  - {error}")
    
    def log_progress(self, current: int, total: int, prefix: str = "Progress"):
        """Log progress."""
        percentage = (current / total) * 100
        self.logger.info(f"{prefix}: {current}/{total} ({percentage:.1f}%)")
    
    def log_dataset_stats(self, stats: Dict[str, Any]):
        """Log dataset statistics."""
        self.logger.info("Dataset Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    self.logger.info(f"    {subkey}: {subvalue}")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all logged metrics."""
        return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()
    
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying logger."""
        return getattr(self.logger, name)


@contextmanager
def log_execution_time(operation: str, logger: Optional[logging.Logger] = None):
    """
    Context manager to log execution time of an operation.
    
    Args:
        operation: Name of operation being timed
        logger: Logger to use (creates new one if None)
    """
    if logger is None:
        logger = get_logger()
    
    start_time = time.time()
    logger.info(f"Starting {operation}")
    
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed {operation} in {elapsed:.4f} seconds")


class ProgressLogger:
    """
    Progress logger with visual indicators.
    """
    
    def __init__(self, total: int, desc: str = "Processing", logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.
        
        Args:
            total: Total number of items
            desc: Description of operation
            logger: Logger instance
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()
        self.logger = logger or get_logger('data.progress')
        
        # Progress bar characters
        self.bar_length = 50
        
        self.logger.info(f"Starting {desc} ({total} items)")
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """
        Update progress.
        
        Args:
            increment: Number of items completed
            message: Optional message to log
        """
        self.current += increment
        self.current = min(self.current, self.total)
        
        # Calculate progress
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Calculate ETA
        if self.current > 0:
            items_per_second = self.current / elapsed
            eta = (self.total - self.current) / items_per_second if items_per_second > 0 else 0
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: Calculating..."
        
        # Create progress bar
        filled_length = int(self.bar_length * self.current // self.total)
        bar = '█' * filled_length + '░' * (self.bar_length - filled_length)
        
        # Log progress
        log_message = f"\r{self.desc} |{bar}| {self.current}/{self.total} ({percentage:.1f}%) {eta_str}"
        
        if message:
            log_message += f" | {message}"
        
        # Use logger if available, else print
        if self.logger:
            self.logger.info(log_message)
        else:
            print(log_message, end='')
    
    def finish(self, message: Optional[str] = None):
        """Finish progress logging."""
        elapsed = time.time() - self.start_time
        
        if message:
            self.logger.info(f"Completed {self.desc}: {message}")
        else:
            self.logger.info(f"Completed {self.desc} in {elapsed:.2f} seconds")
        
        # Print final newline if using print
        if not self.logger:
            print()


def configure_logging_from_file(config_file: str):
    """
    Configure logging from a JSON or YAML configuration file.
    
    Args:
        config_file: Path to configuration file
    """
    from .file_io import read_json, read_yaml
    
    config_path = Path(config_file)
    
    if config_path.suffix == '.json':
        config = read_json(config_file)
    elif config_path.suffix in ['.yaml', '.yml']:
        config = read_yaml(config_file)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    logging.config.dictConfig(config)


def set_logging_level(level: Union[str, int], logger_names: Optional[list] = None):
    """
    Set logging level for specific loggers.
    
    Args:
        level: Logging level
        logger_names: List of logger names (None for all)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    if logger_names is None:
        logging.getLogger().setLevel(level)
    else:
        for name in logger_names:
            logging.getLogger(name).setLevel(level)


def create_log_file_handler(
    log_file: str,
    level: Union[str, int] = logging.DEBUG,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    formatter: Optional[logging.Formatter] = None
) -> logging.Handler:
    """
    Create a rotating file handler.
    
    Args:
        log_file: Path to log file
        level: Logging level
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
        formatter: Log formatter
        
    Returns:
        Configured file handler
    """
    from logging.handlers import RotatingFileHandler
    
    # Ensure directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf8'
    )
    
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    handler.setLevel(level)
    
    if formatter is None:
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    
    return handler