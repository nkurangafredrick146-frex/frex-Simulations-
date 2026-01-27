"""
Asynchronous task processor for handling long-running operations
"""

import asyncio
import uuid
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Coroutine
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import inspect

from ...utils.logging_config import setup_logging

logger = setup_logging("async_processor")


class TaskStatus(str, Enum):
    """Task status enum"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """Task priority enum"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ProcessingTask:
    """Processing task data class"""
    
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    progress: float = 0.0
    progress_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    callback: Optional[Callable] = None
    future: Optional[asyncio.Future] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "priority": self.priority.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timeout": self.timeout,
            "has_callback": self.callback is not None
        }


class AsyncProcessor:
    """
    Asynchronous task processor with priority queue, timeouts, and progress tracking
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 1000,
        default_timeout: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # Clean up completed tasks every 60 seconds
        max_task_age: float = 3600.0,  # Keep tasks for 1 hour
        use_thread_pool: bool = True,
        use_process_pool: bool = False
    ):
        """
        Initialize async processor
        
        Args:
            max_workers: Maximum number of concurrent workers
            queue_size: Maximum queue size
            default_timeout: Default task timeout in seconds
            cleanup_interval: Interval for cleaning up old tasks
            max_task_age: Maximum age of completed tasks to keep
            use_thread_pool: Use thread pool for I/O bound tasks
            use_process_pool: Use process pool for CPU bound tasks
        """
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.default_timeout = default_timeout
        self.cleanup_interval = cleanup_interval
        self.max_task_age = max_task_age
        
        # Task storage
        self.tasks: Dict[str, ProcessingTask] = {}
        self.priority_queues: Dict[TaskPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=queue_size // 4)
            for priority in TaskPriority
        }
        
        # Execution pools
        self.use_thread_pool = use_thread_pool
        self.use_process_pool = use_process_pool
        
        if use_thread_pool:
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="async_worker")
        
        if use_process_pool:
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        
        # Worker tasks
        self.worker_tasks: List[asyncio.Task] = []
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
            "tasks_timed_out": 0,
            "queue_size": 0,
            "active_workers": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        # Locks
        self._stats_lock = asyncio.Lock()
        self._tasks_lock = asyncio.Lock()
        
        logger.info(f"AsyncProcessor initialized with {max_workers} workers, queue size: {queue_size}")
    
    async def start(self):
        """Start the async processor"""
        if self.is_running:
            logger.warning("AsyncProcessor already running")
            return
        
        logger.info("Starting AsyncProcessor...")
        
        # Start workers
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(
                self._worker_loop(),
                name=f"async_worker_{i}"
            )
            self.worker_tasks.append(worker_task)
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="async_cleanup"
        )
        
        self.is_running = True
        logger.info(f"AsyncProcessor started with {self.max_workers} workers")
    
    async def shutdown(self, graceful: bool = True, timeout: float = 30.0):
        """Shutdown the async processor"""
        if not self.is_running:
            logger.warning("AsyncProcessor not running")
            return
        
        logger.info(f"Shutting down AsyncProcessor (graceful={graceful})...")
        self.is_running = False
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await asyncio.wait_for(self.cleanup_task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Cancel all worker tasks
        for worker_task in self.worker_tasks:
            worker_task.cancel()
        
        if graceful:
            # Wait for workers to finish current tasks
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.worker_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("Some workers did not finish gracefully")
        
        # Shutdown thread pools
        if self.use_thread_pool:
            self.thread_pool.shutdown(wait=graceful)
        
        if self.use_process_pool:
            self.process_pool.shutdown(wait=graceful)
        
        # Clear task queues
        for queue in self.priority_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except:
                    pass
        
        self.worker_tasks.clear()
        logger.info("AsyncProcessor shutdown complete")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for asynchronous processing
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority
            timeout: Task timeout in seconds
            metadata: Task metadata
            user_id: User ID associated with task
            session_id: Session ID associated with task
            callback: Callback function to call when task completes
            **kwargs: Function keyword arguments
        
        Returns:
            Task ID
        """
        if not self.is_running:
            raise RuntimeError("AsyncProcessor not running. Call start() first.")
        
        # Check queue capacity
        total_queue_size = sum(q.qsize() for q in self.priority_queues.values())
        if total_queue_size >= self.queue_size:
            raise RuntimeError(f"Task queue is full (size: {self.queue_size})")
        
        # Create task
        task_id = str(uuid.uuid4())
        timeout = timeout or self.default_timeout
        
        task = ProcessingTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id,
            callback=callback
        )
        
        # Store task
        async with self._tasks_lock:
            self.tasks[task_id] = task
        
        # Update statistics
        async with self._stats_lock:
            self.stats["tasks_submitted"] += 1
            self.stats["queue_size"] = total_queue_size + 1
        
        # Add to priority queue
        queue = self.priority_queues[priority]
        await queue.put(task)
        
        logger.debug(f"Task submitted: {task_id}, priority: {priority.name}, queue: {queue.qsize()}")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by ID"""
        async with self._tasks_lock:
            task = self.tasks.get(task_id)
        
        if not task:
            return None
        
        return task.to_dict()
    
    async def cancel_task(self, task_id: str, force: bool = False) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: Task ID to cancel
            force: Whether to force cancellation (may not work for running tasks)
        
        Returns:
            True if task was cancelled, False otherwise
        """
        async with self._tasks_lock:
            task = self.tasks.get(task_id)
            
            if not task:
                return False
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False  # Already in terminal state
            
            # Try to cancel future if it exists
            if task.future and not task.future.done():
                if force:
                    task.future.cancel()
                else:
                    # Only cancel if not running
                    if task.status != TaskStatus.RUNNING:
                        task.future.cancel()
            
            # Update task status
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            
            # Update statistics
            async with self._stats_lock:
                self.stats["tasks_cancelled"] += 1
        
        logger.info(f"Task cancelled: {task_id}")
        return True
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1
    ) -> Optional[Any]:
        """
        Wait for a task to complete and return its result
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds
        
        Returns:
            Task result if completed, None if timeout or error
        """
        start_time = datetime.utcnow()
        
        while True:
            # Check timeout
            if timeout:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None
            
            # Get task status
            async with self._tasks_lock:
                task = self.tasks.get(task_id)
                
                if not task:
                    return None
                
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                    if task.status == TaskStatus.COMPLETED:
                        return task.result
                    else:
                        return None
            
            # Wait before polling again
            await asyncio.sleep(poll_interval)
    
    async def update_task_progress(
        self,
        task_id: str,
        progress: float,
        message: Optional[str] = None
    ) -> bool:
        """
        Update task progress
        
        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            message: Progress message
        
        Returns:
            True if progress was updated, False otherwise
        """
        async with self._tasks_lock:
            task = self.tasks.get(task_id)
            
            if not task:
                return False
            
            if task.status != TaskStatus.RUNNING:
                return False
            
            task.progress = max(0.0, min(1.0, progress))
            if message:
                task.progress_message = message
        
        logger.debug(f"Task progress updated: {task_id}, progress: {progress:.2f}")
        return True
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        async with self._stats_lock:
            stats = self.stats.copy()
        
        # Add current queue sizes
        stats["priority_queues"] = {
            priority.name: queue.qsize()
            for priority, queue in self.priority_queues.items()
        }
        
        # Add task counts by status
        async with self._tasks_lock:
            status_counts = {}
            for task in self.tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
        
        stats["tasks_by_status"] = status_counts
        stats["total_tasks"] = len(self.tasks)
        stats["is_running"] = self.is_running
        stats["timestamp"] = datetime.utcnow().isoformat()
        
        return stats
    
    async def _worker_loop(self):
        """Worker loop that processes tasks from queues"""
        worker_id = asyncio.current_task().get_name()
        logger.debug(f"Worker started: {worker_id}")
        
        while self.is_running:
            try:
                # Get next task from priority queues (highest priority first)
                task = None
                for priority in sorted(TaskPriority, reverse=True):
                    queue = self.priority_queues[priority]
                    if not queue.empty():
                        try:
                            task = queue.get_nowait()
                            break
                        except asyncio.QueueEmpty:
                            continue
                
                if not task:
                    # No tasks available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the task
                await self._process_task(task)
                
                # Mark task as done in queue
                queue.task_done()
                
            except asyncio.CancelledError:
                logger.debug(f"Worker cancelled: {worker_id}")
                break
            except Exception as e:
                logger.error(f"Worker error in {worker_id}: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight error loop
        
        logger.debug(f"Worker stopped: {worker_id}")
    
    async def _process_task(self, task: ProcessingTask):
        """Process a single task"""
        logger.info(f"Processing task: {task.task_id}, func: {task.func.__name__}")
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        async with self._stats_lock:
            self.stats["active_workers"] += 1
        
        try:
            # Determine execution method
            is_coroutine = inspect.iscoroutinefunction(task.func)
            is_generator = inspect.isgeneratorfunction(task.func) or inspect.isasyncgenfunction(task.func)
            
            # Create future for timeout handling
            task.future = asyncio.Future()
            
            # Execute task with timeout
            if is_coroutine:
                # Async function
                coro = task.func(*task.args, **task.kwargs)
                result = await asyncio.wait_for(coro, timeout=task.timeout)
            elif is_generator:
                # Generator or async generator
                result = []
                if inspect.isgeneratorfunction(task.func):
                    gen = task.func(*task.args, **task.kwargs)
                    for item in gen:
                        result.append(item)
                else:  # async generator
                    gen = task.func(*task.args, **task.kwargs)
                    async for item in gen:
                        result.append(item)
            elif self.use_process_pool and self._is_cpu_bound(task.func):
                # CPU-bound function in process pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.process_pool,
                    task.func,
                    *task.args,
                    **task.kwargs
                )
            elif self.use_thread_pool:
                # I/O bound function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: task.func(*task.args, **task.kwargs)
                )
            else:
                # Run in current thread (should be fast)
                result = task.func(*task.args, **task.kwargs)
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.progress = 1.0
            task.progress_message = "Completed"
            
            # Update statistics
            async with self._stats_lock:
                self.stats["tasks_completed"] += 1
                self.stats["active_workers"] -= 1
                
                # Update processing time stats
                if task.started_at and task.completed_at:
                    processing_time = (task.completed_at - task.started_at).total_seconds()
                    self.stats["total_processing_time"] += processing_time
                    self.stats["avg_processing_time"] = (
                        self.stats["total_processing_time"] / self.stats["tasks_completed"]
                    )
            
            logger.info(f"Task completed: {task.task_id}")
        
        except asyncio.TimeoutError:
            # Task timeout
            task.status = TaskStatus.TIMEOUT
            task.error = f"Task timeout after {task.timeout} seconds"
            task.traceback = "Timeout"
            
            async with self._stats_lock:
                self.stats["tasks_timed_out"] += 1
                self.stats["active_workers"] -= 1
            
            logger.warning(f"Task timeout: {task.task_id}")
        
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.traceback = traceback.format_exc()
            
            async with self._stats_lock:
                self.stats["tasks_failed"] += 1
                self.stats["active_workers"] -= 1
            
            logger.error(f"Task failed: {task.task_id}, error: {e}", exc_info=True)
        
        finally:
            # Ensure completion time is set
            task.completed_at = datetime.utcnow()
            
            # Call callback if provided
            if task.callback and callable(task.callback):
                try:
                    if inspect.iscoroutinefunction(task.callback):
                        await task.callback(task)
                    else:
                        task.callback(task)
                except Exception as e:
                    logger.error(f"Callback error for task {task.task_id}: {e}", exc_info=True)
    
    def _is_cpu_bound(self, func: Callable) -> bool:
        """Check if a function is likely CPU-bound"""
        # This is a simple heuristic - in practice, you might want a more sophisticated approach
        func_name = func.__name__.lower()
        cpu_keywords = ['compute', 'calculate', 'process', 'train', 'encode', 'decode', 'transform']
        
        return any(keyword in func_name for keyword in cpu_keywords)
    
    async def _cleanup_loop(self):
        """Cleanup loop for removing old tasks"""
        logger.debug("Cleanup loop started")
        
        while self.is_running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}", exc_info=True)
        
        logger.debug("Cleanup loop stopped")
    
    async def _cleanup_old_tasks(self):
        """Remove old completed tasks"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.max_task_age)
        tasks_to_remove = []
        
        async with self._tasks_lock:
            for task_id, task in self.tasks.items():
                if task.completed_at and task.completed_at < cutoff_time:
                    tasks_to_remove.append(task_id)
            
            # Remove old tasks
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
        
        if tasks_to_remove:
            logger.debug(f"Cleaned up {len(tasks_to_remove)} old tasks")
    
    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering"""
        filtered_tasks = []
        
        async with self._tasks_lock:
            for task in self.tasks.values():
                # Apply filters
                if status and task.status != status:
                    continue
                
                if user_id and task.user_id != user_id:
                    continue
                
                if session_id and task.session_id != session_id:
                    continue
                
                filtered_tasks.append(task.to_dict())
        
        # Apply pagination
        start_idx = offset
        end_idx = offset + limit
        return filtered_tasks[start_idx:end_idx]
    
    async def purge_tasks(
        self,
        status: Optional[TaskStatus] = None,
        older_than: Optional[float] = None
    ) -> int:
        """
        Purge tasks matching criteria
        
        Args:
            status: Only purge tasks with this status
            older_than: Only purge tasks older than this many seconds
        
        Returns:
            Number of tasks purged
        """
        cutoff_time = None
        if older_than:
            cutoff_time = datetime.utcnow() - timedelta(seconds=older_than)
        
        tasks_to_purge = []
        
        async with self._tasks_lock:
            for task_id, task in self.tasks.items():
                # Apply status filter
                if status and task.status != status:
                    continue
                
                # Apply age filter
                if cutoff_time and task.created_at > cutoff_time:
                    continue
                
                tasks_to_purge.append(task_id)
            
            # Purge tasks
            for task_id in tasks_to_purge:
                del self.tasks[task_id]
        
        logger.info(f"Purged {len(tasks_to_purge)} tasks")
        return len(tasks_to_purge)


# Task progress context manager
class TaskProgress:
    """Context manager for task progress reporting"""
    
    def __init__(self, processor: AsyncProcessor, task_id: str):
        self.processor = processor
        self.task_id = task_id
        self.progress = 0.0
    
    async def update(self, progress: float, message: Optional[str] = None):
        """Update task progress"""
        self.progress = progress
        await self.processor.update_task_progress(self.task_id, progress, message)
    
    async def increment(self, amount: float = 0.1, message: Optional[str] = None):
        """Increment task progress"""
        self.progress = min(1.0, self.progress + amount)
        await self.update(self.progress, message)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Ensure progress is 100% on successful completion
        if exc_type is None and self.progress < 1.0:
            await self.update(1.0, "Completed")


# Utility functions for common task patterns
async def run_with_progress(
    processor: AsyncProcessor,
    func: Callable,
    *args,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    **kwargs
) -> Any:
    """
    Run a function with progress reporting
    
    Args:
        processor: AsyncProcessor instance
        func: Function to execute
        *args: Function arguments
        progress_callback: Optional callback for progress updates
        **kwargs: Function keyword arguments
    
    Returns:
        Function result
    """
    task_id = await processor.submit_task(func, *args, **kwargs)
    
    # Monitor progress
    while True:
        status = await processor.get_task_status(task_id)
        if not status:
            break
        
        if status["status"] in ["completed", "failed", "cancelled", "timeout"]:
            break
        
        # Report progress if callback provided
        if progress_callback and status["progress"] is not None:
            progress_callback(status["progress"], status.get("progress_message", ""))
        
        await asyncio.sleep(0.5)
    
    # Get final result
    final_status = await processor.get_task_status(task_id)
    if final_status and final_status["status"] == "completed":
        return final_status["result"]
    
    return None


def create_task_decorator(processor: AsyncProcessor):
    """
    Create a decorator for running functions as async tasks
    
    Args:
        processor: AsyncProcessor instance
    
    Returns:
        Decorator function
    """
    def task_decorator(
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        **task_kwargs
    ):
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract task metadata from kwargs
                task_metadata = {
                    "function": func.__name__,
                    "module": func.__module__
                }
                
                # Submit task
                task_id = await processor.submit_task(
                    func,
                    *args,
                    priority=priority,
                    timeout=timeout,
                    metadata=task_metadata,
                    **task_kwargs,
                    **kwargs
                )
                
                return task_id
            
            # Preserve original function attributes
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            wrapper.__module__ = func.__module__
            
            return wrapper
        
        return decorator
    
    return task_decorator


# Export
__all__ = [
    'TaskStatus',
    'TaskPriority',
    'ProcessingTask',
    'AsyncProcessor',
    'TaskProgress',
    'run_with_progress',
    'create_task_decorator'
]