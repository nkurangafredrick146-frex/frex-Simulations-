"""
Advanced logging configuration for FrexTech AI Simulations
Provides structured logging, distributed logging, and monitoring integration
"""

import os
import sys
import json
import logging
import logging.config
import logging.handlers
import threading
import queue
import socket
import time
import inspect
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

# Third-party imports (optional)
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class LogLevel(str, Enum):
    """Log levels enum"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log format enum"""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


class LogDestination(str, Enum):
    """Log destination enum"""
    CONSOLE = "console"
    FILE = "file"
    SYSLOG = "syslog"
    HTTP = "http"
    KAFKA = "kafka"


@dataclass
class LogRecord:
    """Structured log record"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    thread_name: str
    process_id: int
    hostname: str
    ip_address: str
    correlation_id: str = None
    request_id: str = None
    user_id: str = None
    session_id: str = None
    duration_ms: float = None
    extra: Dict[str, Any] = field(default_factory=dict)
    exception: str = None
    stack_trace: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = '%',
        json_indent: Optional[int] = None,
        include_extra: bool = True
    ):
        super().__init__(fmt, datefmt, style)
        self.json_indent = json_indent
        self.include_extra = include_extra
        self.hostname = socket.gethostname()
        self.ip_address = self._get_ip_address()
    
    def _get_ip_address(self) -> str:
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        # Get exception info if available
        exception_info = None
        stack_trace = None
        
        if record.exc_info:
            exception_info = str(record.exc_info[1])
            stack_trace = self.formatException(record.exc_info)
        
        # Create structured log record
        log_record = LogRecord(
            timestamp=self.formatTime(record, self.datefmt),
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            thread_name=record.threadName,
            process_id=record.process,
            hostname=self.hostname,
            ip_address=self.ip_address,
            correlation_id=getattr(record, 'correlation_id', None),
            request_id=getattr(record, 'request_id', None),
            user_id=getattr(record, 'user_id', None),
            session_id=getattr(record, 'session_id', None),
            duration_ms=getattr(record, 'duration_ms', None),
            exception=exception_info,
            stack_trace=stack_trace
        )
        
        # Add extra fields
        if self.include_extra and hasattr(record, 'extra'):
            log_record.extra.update(record.extra)
        
        # Add any additional attributes from record
        for attr in ['pathname', 'filename', 'processName']:
            if hasattr(record, attr):
                log_record.extra[attr] = getattr(record, attr)
        
        # Convert to JSON
        return json.dumps(
            log_record.to_dict(),
            default=str,
            ensure_ascii=False,
            indent=self.json_indent
        )


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records"""
    
    def __init__(self, name: str = ''):
        super().__init__(name)
        self.context = threading.local()
        self.context.correlation_id = None
        self.context.request_id = None
        self.context.user_id = None
        self.context.session_id = None
        self.context.start_time = None
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record"""
        # Add correlation ID
        if hasattr(self.context, 'correlation_id') and self.context.correlation_id:
            record.correlation_id = self.context.correlation_id
        
        # Add request ID
        if hasattr(self.context, 'request_id') and self.context.request_id:
            record.request_id = self.context.request_id
        
        # Add user ID
        if hasattr(self.context, 'user_id') and self.context.user_id:
            record.user_id = self.context.user_id
        
        # Add session ID
        if hasattr(self.context, 'session_id') and self.context.session_id:
            record.session_id = self.session_id
        
        # Calculate duration if start time is set
        if hasattr(self.context, 'start_time') and self.context.start_time:
            duration = (datetime.utcnow() - self.context.start_time).total_seconds() * 1000
            record.duration_ms = round(duration, 2)
        
        return True
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID in context"""
        self.context.correlation_id = correlation_id
    
    def set_request_id(self, request_id: str):
        """Set request ID in context"""
        self.context.request_id = request_id
    
    def set_user_id(self, user_id: str):
        """Set user ID in context"""
        self.context.user_id = user_id
    
    def set_session_id(self, session_id: str):
        """Set session ID in context"""
        self.context.session_id = session_id
    
    def start_timer(self):
        """Start timing for duration calculation"""
        self.context.start_time = datetime.utcnow()
    
    def clear_context(self):
        """Clear all context"""
        self.context.correlation_id = None
        self.context.request_id = None
        self.context.user_id = None
        self.context.session_id = None
        self.context.start_time = None


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler using queue"""
    
    def __init__(
        self,
        target_handler: logging.Handler,
        queue_size: int = 10000,
        worker_count: int = 1
    ):
        super().__init__()
        self.target_handler = target_handler
        self.queue = queue.Queue(maxsize=queue_size)
        self.worker_count = worker_count
        self.workers = []
        self._stop_event = threading.Event()
        
        # Start workers
        for i in range(worker_count):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"log_worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to queue"""
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            # If queue is full, drop the record but log error
            print(f"Log queue full, dropping record: {record.getMessage()}")
    
    def _worker_loop(self):
        """Worker loop to process log records"""
        while not self._stop_event.is_set():
            try:
                record = self.queue.get(timeout=1)
                self.target_handler.emit(record)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Log worker error: {e}")
    
    def flush(self):
        """Flush all pending log records"""
        self.queue.join()
        self.target_handler.flush()
    
    def close(self):
        """Close handler and stop workers"""
        self._stop_event.set()
        for worker in self.workers:
            worker.join(timeout=5)
        self.target_handler.close()
        super().close()


class MetricsLogHandler(logging.Handler):
    """Log handler that exports metrics to Prometheus"""
    
    def __init__(self, namespace: str = "frextech"):
        super().__init__()
        self.namespace = namespace
        
        if PROMETHEUS_AVAILABLE:
            self._init_metrics()
        else:
            self.metrics = None
    
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'log_messages_total': prometheus_client.Counter(
                f'{self.namespace}_log_messages_total',
                'Total number of log messages',
                ['level', 'logger', 'module']
            ),
            'log_message_bytes': prometheus_client.Counter(
                f'{self.namespace}_log_message_bytes',
                'Total bytes of log messages',
                ['level', 'logger']
            ),
            'log_latency_seconds': prometheus_client.Histogram(
                f'{self.namespace}_log_latency_seconds',
                'Log processing latency',
                ['level'],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            )
        }
    
    def emit(self, record: logging.LogRecord):
        """Emit log record and record metrics"""
        start_time = time.time()
        
        # Calculate message size
        message = self.format(record)
        message_size = len(message.encode('utf-8'))
        
        # Update metrics if available
        if self.metrics and PROMETHEUS_AVAILABLE:
            with self.metrics['log_latency_seconds'].labels(level=record.levelname).time():
                # Record metrics
                self.metrics['log_messages_total'].labels(
                    level=record.levelname,
                    logger=record.name,
                    module=record.module
                ).inc()
                
                self.metrics['log_message_bytes'].labels(
                    level=record.levelname,
                    logger=record.name
                ).inc(message_size)
        
        latency = time.time() - start_time
        
        # Store latency in record for potential use
        record.latency_seconds = latency


class RotatingFileHandlerWithCompression(logging.handlers.RotatingFileHandler):
    """Rotating file handler with optional compression"""
    
    def __init__(
        self,
        filename: str,
        mode: str = 'a',
        maxBytes: int = 10 * 1024 * 1024,  # 10MB
        backupCount: int = 10,
        encoding: Optional[str] = None,
        delay: bool = False,
        compress: bool = True,
        compresslevel: int = 9
    ):
        super().__init__(
            filename, mode, maxBytes, backupCount,
            encoding, delay
        )
        self.compress = compress
        self.compresslevel = compresslevel
    
    def doRollover(self):
        """Do a rollover, optionally compressing the old log file"""
        super().doRollover()
        
        if self.compress and self.backupCount > 0:
            # Find the oldest backup file (highest number)
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.baseFilename}.{i}"
                if os.path.exists(sfn):
                    # Compress this file
                    import gzip
                    compressed_file = f"{sfn}.gz"
                    with open(sfn, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb', compresslevel=self.compresslevel) as f_out:
                            f_out.write(f_in.read())
                    # Remove uncompressed file
                    os.remove(sfn)
                    break


class LogManager:
    """Central log manager for the application"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = None
        self.context_filter = ContextFilter()
        self.loggers = {}
        self.default_logger = None
        self._initialized = True
    
    def setup_logging(
        self,
        app_name: str = "frextech",
        log_level: Union[str, int] = logging.INFO,
        log_format: LogFormat = LogFormat.STRUCTURED,
        destinations: List[LogDestination] = None,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = "logs",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,
        enable_async: bool = True,
        enable_metrics: bool = True,
        enable_context: bool = True,
        correlation_id: Optional[str] = None
    ):
        """
        Setup logging configuration
        
        Args:
            app_name: Application name for logging
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Log format (text, json, structured)
            destinations: Where to send logs (console, file, syslog, http, kafka)
            log_file: Log file path (if not provided, generated from app_name)
            log_dir: Log directory for file logging
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            enable_async: Enable asynchronous logging
            enable_metrics: Enable metrics collection
            enable_context: Enable contextual logging
            correlation_id: Initial correlation ID
        """
        # Set default destinations
        if destinations is None:
            destinations = [LogDestination.CONSOLE, LogDestination.FILE]
        
        # Create log directory if needed
        if LogDestination.FILE in destinations:
            os.makedirs(log_dir, exist_ok=True)
            if log_file is None:
                log_file = os.path.join(log_dir, f"{app_name}.log")
        
        # Convert string log level to logging constant
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create handlers based on destinations
        handlers = []
        
        for destination in destinations:
            handler = self._create_handler(
                destination=destination,
                log_format=log_format,
                log_file=log_file,
                max_file_size=max_file_size,
                backup_count=backup_count,
                app_name=app_name
            )
            
            if handler:
                handlers.append(handler)
        
        # Add metrics handler if enabled
        if enable_metrics and PROMETHEUS_AVAILABLE:
            metrics_handler = MetricsLogHandler(namespace=app_name)
            metrics_handler.setLevel(log_level)
            handlers.append(metrics_handler)
        
        # Wrap handlers with async handler if enabled
        if enable_async and handlers:
            async_handlers = []
            for handler in handlers:
                async_handler = AsyncLogHandler(
                    target_handler=handler,
                    queue_size=10000,
                    worker_count=2
                )
                async_handler.setLevel(log_level)
                async_handlers.append(async_handler)
            
            handlers = async_handlers
        
        # Add context filter if enabled
        if enable_context:
            for handler in handlers:
                handler.addFilter(self.context_filter)
        
        # Set formatters
        formatter = self._create_formatter(log_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
        
        # Set correlation ID if provided
        if correlation_id:
            self.context_filter.set_correlation_id(correlation_id)
        
        # Store configuration
        self.config = {
            'app_name': app_name,
            'log_level': log_level,
            'log_format': log_format,
            'destinations': destinations,
            'log_file': log_file,
            'log_dir': log_dir,
            'enable_async': enable_async,
            'enable_metrics': enable_metrics,
            'enable_context': enable_context
        }
        
        # Create default logger
        self.default_logger = self.get_logger(app_name)
        
        # Log startup message
        self.default_logger.info(
            "Logging initialized",
            extra={
                'config': self.config,
                'hostname': socket.gethostname(),
                'python_version': sys.version
            }
        )
    
    def _create_handler(
        self,
        destination: LogDestination,
        log_format: LogFormat,
        log_file: Optional[str],
        max_file_size: int,
        backup_count: int,
        app_name: str
    ) -> Optional[logging.Handler]:
        """Create log handler based on destination"""
        
        if destination == LogDestination.CONSOLE:
            handler = logging.StreamHandler(sys.stdout)
        
        elif destination == LogDestination.FILE and log_file:
            handler = RotatingFileHandlerWithCompression(
                filename=log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                compress=True
            )
        
        elif destination == LogDestination.SYSLOG:
            # Syslog handler for Unix/Linux
            if hasattr(logging.handlers, 'SysLogHandler'):
                handler = logging.handlers.SysLogHandler(
                    address='/dev/log',
                    facility=logging.handlers.SysLogHandler.LOG_USER
                )
            else:
                return None
        
        elif destination == LogDestination.HTTP:
            # HTTP handler for sending logs to remote server
            handler = logging.handlers.HTTPHandler(
                host='localhost:8000',
                url='/logs',
                method='POST'
            )
        
        elif destination == LogDestination.KAFKA:
            # Kafka handler (requires kafka-python)
            try:
                from kafka import KafkaProducer
                from kafka_logger.handlers import KafkaHandler
                
                producer = KafkaProducer(
                    bootstrap_servers=['localhost:9092'],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                handler = KafkaHandler(
                    producer=producer,
                    topic=f'{app_name}_logs'
                )
            except ImportError:
                print("Kafka handler requires kafka-python and kafka-logger")
                return None
        
        else:
            return None
        
        return handler
    
    def _create_formatter(self, log_format: LogFormat) -> logging.Formatter:
        """Create formatter based on log format"""
        
        if log_format == LogFormat.TEXT:
            return logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        elif log_format == LogFormat.JSON:
            return StructuredFormatter(
                json_indent=None,
                include_extra=True
            )
        
        elif log_format == LogFormat.STRUCTURED:
            return StructuredFormatter(
                json_indent=None,
                include_extra=True
            )
        
        else:
            # Default to structured
            return StructuredFormatter()
    
    def get_logger(
        self,
        name: str,
        level: Optional[Union[str, int]] = None,
        propagate: bool = True
    ) -> logging.Logger:
        """
        Get or create a logger with the given name
        
        Args:
            name: Logger name
            level: Logger level (overrides root level)
            propagate: Whether to propagate to parent loggers
        
        Returns:
            Configured logger
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        
        if level is not None:
            if isinstance(level, str):
                level = getattr(logging, level.upper())
            logger.setLevel(level)
        
        logger.propagate = propagate
        self.loggers[name] = logger
        
        return logger
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        return {
            'correlation_id': self.context_filter.context.correlation_id,
            'request_id': self.context_filter.context.request_id,
            'user_id': self.context_filter.context.user_id,
            'session_id': self.context_filter.context.session_id
        }
    
    def set_context(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Set logging context"""
        if correlation_id:
            self.context_filter.set_correlation_id(correlation_id)
        if request_id:
            self.context_filter.set_request_id(request_id)
        if user_id:
            self.context_filter.set_user_id(user_id)
        if session_id:
            self.context_filter.set_session_id(session_id)
    
    def clear_context(self):
        """Clear logging context"""
        self.context_filter.clear_context()
    
    def start_timer(self):
        """Start timer for duration calculation"""
        self.context_filter.start_timer()
    
    def log_exception(
        self,
        exception: Exception,
        logger_name: str = None,
        level: str = "ERROR",
        extra: Dict[str, Any] = None
    ):
        """Log exception with full traceback"""
        logger = self.get_logger(logger_name) if logger_name else self.default_logger
        
        log_method = getattr(logger, level.lower())
        
        exc_info = (type(exception), exception, exception.__traceback__)
        log_method(
            f"Exception occurred: {str(exception)}",
            exc_info=exc_info,
            extra=extra or {}
        )
    
    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        logger_name: str = None,
        extra: Dict[str, Any] = None
    ):
        """Log performance metrics"""
        logger = self.get_logger(logger_name) if logger_name else self.default_logger
        
        log_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'performance': 'slow' if duration_ms > 1000 else 'normal' if duration_ms > 100 else 'fast'
        }
        
        if extra:
            log_data.update(extra)
        
        if duration_ms > 5000:
            logger.warning(f"Slow operation: {operation}", extra=log_data)
        elif duration_ms > 1000:
            logger.info(f"Operation completed: {operation}", extra=log_data)
        else:
            logger.debug(f"Operation completed: {operation}", extra=log_data)
    
    def flush(self):
        """Flush all log handlers"""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.flush()
        
        if logging.root.handlers:
            for handler in logging.root.handlers:
                handler.flush()
    
    def shutdown(self):
        """Shutdown logging system"""
        self.flush()
        
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
        
        if logging.root.handlers:
            for handler in logging.root.handlers:
                handler.close()
        
        self.loggers.clear()
        self.default_logger = None
        self.config = None


# Convenience functions
def setup_logging(
    name: str = "frextech",
    level: str = "INFO",
    config_file: Optional[str] = None
) -> logging.Logger:
    """
    Convenience function to setup logging
    
    Args:
        name: Logger name
        level: Log level
        config_file: Optional logging config file
    
    Returns:
        Configured logger
    """
    log_manager = LogManager()
    
    if config_file and os.path.exists(config_file):
        # Load logging config from file
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        log_manager.setup_logging(**config)
    else:
        # Default configuration
        log_manager.setup_logging(
            app_name=name,
            log_level=level,
            log_format=LogFormat.STRUCTURED,
            destinations=[LogDestination.CONSOLE, LogDestination.FILE],
            log_dir="logs",
            enable_async=True,
            enable_metrics=True,
            enable_context=True
        )
    
    return log_manager.get_logger(name)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance
    
    Args:
        name: Logger name (defaults to caller's module)
    
    Returns:
        Logger instance
    """
    if name is None:
        # Get caller's module name
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else "__main__"
    
    return LogManager().get_logger(name)


def log_with_context(
    correlation_id: str = None,
    request_id: str = None,
    user_id: str = None,
    session_id: str = None
):
    """
    Decorator to add logging context to function
    
    Args:
        correlation_id: Correlation ID
        request_id: Request ID
        user_id: User ID
        session_id: Session ID
    
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            log_manager = LogManager()
            
            # Store original context
            original_context = log_manager.get_context()
            
            try:
                # Set new context
                log_manager.set_context(
                    correlation_id=correlation_id,
                    request_id=request_id,
                    user_id=user_id,
                    session_id=session_id
                )
                
                # Start timer
                log_manager.start_timer()
                
                # Call function
                result = func(*args, **kwargs)
                
                return result
            
            finally:
                # Restore original context
                log_manager.set_context(**original_context)
        
        return wrapper
    
    return decorator


class LoggingContext:
    """Context manager for logging"""
    
    def __init__(
        self,
        correlation_id: str = None,
        request_id: str = None,
        user_id: str = None,
        session_id: str = None,
        logger_name: str = None
    ):
        self.correlation_id = correlation_id
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self.logger_name = logger_name
        self.log_manager = LogManager()
        self.original_context = None
        self.start_time = None
    
    def __enter__(self):
        # Store original context
        self.original_context = self.log_manager.get_context()
        
        # Set new context
        self.log_manager.set_context(
            correlation_id=self.correlation_id,
            request_id=self.request_id,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        # Start timer
        self.start_time = datetime.utcnow()
        self.log_manager.start_timer()
        
        # Get logger
        self.logger = self.log_manager.get_logger(
            self.logger_name or __name__
        )
        
        self.logger.debug(
            "Entering logging context",
            extra={
                'correlation_id': self.correlation_id,
                'request_id': self.request_id
            }
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        
        # Log exit
        if exc_type is None:
            self.logger.debug(
                "Exiting logging context",
                extra={'duration_ms': duration}
            )
        else:
            self.logger.error(
                f"Exiting logging context with exception: {exc_val}",
                extra={
                    'duration_ms': duration,
                    'exception_type': exc_type.__name__,
                    'exception': str(exc_val)
                },
                exc_info=(exc_type, exc_val, exc_tb)
            )
        
        # Restore original context
        self.log_manager.set_context(**self.original_context)


# Export
__all__ = [
    'LogLevel',
    'LogFormat',
    'LogDestination',
    'LogRecord',
    'StructuredFormatter',
    'ContextFilter',
    'AsyncLogHandler',
    'MetricsLogHandler',
    'RotatingFileHandlerWithCompression',
    'LogManager',
    'setup_logging',
    'get_logger',
    'log_with_context',
    'LoggingContext'
]
