"""
Metrics collection and monitoring for FrexTech AI Simulations
Provides comprehensive metrics for performance monitoring, business insights, and system health
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import json
import psutil
import socket
import platform
from contextlib import contextmanager, asynccontextmanager
from functools import wraps

# Third-party imports (optional)
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from influxdb import InfluxDBClient
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False


class MetricType(str, Enum):
    """Metric types enum"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AggregationMethod(str, Enum):
    """Aggregation methods for metrics"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[List[float]] = None  # For summaries
    namespace: str = "frextech"
    subsystem: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class MetricValue:
    """Single metric value with timestamp and labels"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricAggregation:
    """Aggregated metric values"""
    metric_name: str
    aggregation: AggregationMethod
    value: float
    timestamp: datetime
    window_seconds: int
    sample_count: int
    labels: Dict[str, str] = field(default_factory=dict)


class BaseMetric:
    """Base class for all metrics"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        namespace: str = "frextech"
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.namespace = namespace
        self.full_name = f"{namespace}_{name}"
        self.values = deque(maxlen=10000)  # Store recent values
        self._lock = threading.Lock()
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        with self._lock:
            metric_value = MetricValue(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            )
            self.values.append(metric_value)
            self._on_record(metric_value)
    
    def _on_record(self, metric_value: MetricValue):
        """Hook called when a value is recorded"""
        pass
    
    def get_values(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> List[MetricValue]:
        """Get metric values within time range and matching labels"""
        with self._lock:
            filtered = []
            
            for value in self.values:
                # Filter by time
                if start_time and value.timestamp < start_time:
                    continue
                if end_time and value.timestamp > end_time:
                    continue
                
                # Filter by labels
                if labels:
                    if not all(value.labels.get(k) == v for k, v in labels.items()):
                        continue
                
                filtered.append(value)
            
            return filtered
    
    def aggregate(
        self,
        aggregation: AggregationMethod,
        window_seconds: int = 60,
        labels: Optional[Dict[str, str]] = None
    ) -> MetricAggregation:
        """Aggregate metric values"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(seconds=window_seconds)
        
        values = self.get_values(start_time, end_time, labels)
        
        if not values:
            return MetricAggregation(
                metric_name=self.name,
                aggregation=aggregation,
                value=0.0,
                timestamp=end_time,
                window_seconds=window_seconds,
                sample_count=0,
                labels=labels or {}
            )
        
        numeric_values = [v.value for v in values]
        
        if aggregation == AggregationMethod.SUM:
            result = sum(numeric_values)
        elif aggregation == AggregationMethod.AVG:
            result = statistics.mean(numeric_values)
        elif aggregation == AggregationMethod.MIN:
            result = min(numeric_values)
        elif aggregation == AggregationMethod.MAX:
            result = max(numeric_values)
        elif aggregation == AggregationMethod.COUNT:
            result = len(numeric_values)
        elif aggregation == AggregationMethod.P50:
            result = statistics.quantiles(numeric_values, n=100)[49]
        elif aggregation == AggregationMethod.P90:
            result = statistics.quantiles(numeric_values, n=100)[89]
        elif aggregation == AggregationMethod.P95:
            result = statistics.quantiles(numeric_values, n=100)[94]
        elif aggregation == AggregationMethod.P99:
            result = statistics.quantiles(numeric_values, n=100)[98]
        else:
            result = statistics.mean(numeric_values)
        
        return MetricAggregation(
            metric_name=self.name,
            aggregation=aggregation,
            value=result,
            timestamp=end_time,
            window_seconds=window_seconds,
            sample_count=len(values),
            labels=labels or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'labels': self.labels,
            'namespace': self.namespace,
            'type': self.__class__.__name__
        }


class CounterMetric(BaseMetric):
    """Counter metric (monotonically increasing)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = 0
    
    def record(self, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Record a counter increment"""
        with self._lock:
            self.total += value
        super().record(value, labels)
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter"""
        self.record(amount, labels)


class GaugeMetric(BaseMetric):
    """Gauge metric (can go up and down)"""
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge value"""
        self.record(value, labels)
    
    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment gauge"""
        current = self.get_current_value(labels)
        self.record(current + amount, labels)
    
    def dec(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Decrement gauge"""
        current = self.get_current_value(labels)
        self.record(current - amount, labels)
    
    def get_current_value(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value"""
        values = self.get_values(labels=labels)
        return values[-1].value if values else 0.0


class HistogramMetric(BaseMetric):
    """Histogram metric with configurable buckets"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        namespace: str = "frextech",
        buckets: Optional[List[float]] = None
    ):
        super().__init__(name, description, labels, namespace)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.bucket_counts = defaultdict(int)
    
    def record(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        super().record(value, labels)
        
        # Update bucket counts
        with self._lock:
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] += 1
    
    def get_bucket_counts(self) -> Dict[float, int]:
        """Get bucket counts"""
        with self._lock:
            return dict(self.bucket_counts)


class SummaryMetric(BaseMetric):
    """Summary metric (calculates quantiles over sliding time window)"""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        namespace: str = "frextech",
        quantiles: Optional[List[float]] = None,
        max_age_seconds: int = 600,
        age_buckets: int = 5
    ):
        super().__init__(name, description, labels, namespace)
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self.max_age_seconds = max_age_seconds
        self.age_buckets = age_buckets


class MetricsRegistry:
    """Registry for managing all metrics"""
    
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
        
        self.metrics: Dict[str, BaseMetric] = {}
        self.definitions: Dict[str, MetricDefinition] = {}
        self.prometheus_metrics = {}
        self.system_metrics_enabled = False
        self.exporters = []
        self._lock = threading.Lock()
        
        # Start system metrics collection if enabled
        self.system_metrics_task = None
        self.collection_interval = 60  # seconds
        
        self._initialized = True
    
    def register(
        self,
        metric_def: MetricDefinition,
        prometheus_metric: Any = None
    ) -> BaseMetric:
        """Register a new metric"""
        with self._lock:
            if metric_def.name in self.metrics:
                return self.metrics[metric_def.name]
            
            # Create metric instance
            if metric_def.type == MetricType.COUNTER:
                metric = CounterMetric(
                    name=metric_def.name,
                    description=metric_def.description,
                    labels=metric_def.labels,
                    namespace=metric_def.namespace
                )
            elif metric_def.type == MetricType.GAUGE:
                metric = GaugeMetric(
                    name=metric_def.name,
                    description=metric_def.description,
                    labels=metric_def.labels,
                    namespace=metric_def.namespace
                )
            elif metric_def.type == MetricType.HISTOGRAM:
                metric = HistogramMetric(
                    name=metric_def.name,
                    description=metric_def.description,
                    labels=metric_def.labels,
                    namespace=metric_def.namespace,
                    buckets=metric_def.buckets
                )
            elif metric_def.type == MetricType.SUMMARY:
                metric = SummaryMetric(
                    name=metric_def.name,
                    description=metric_def.description,
                    labels=metric_def.labels,
                    namespace=metric_def.namespace,
                    quantiles=metric_def.quantiles
                )
            else:
                raise ValueError(f"Unknown metric type: {metric_def.type}")
            
            # Store metric
            self.metrics[metric_def.name] = metric
            self.definitions[metric_def.name] = metric_def
            
            # Store Prometheus metric if provided
            if prometheus_metric:
                self.prometheus_metrics[metric_def.name] = prometheus_metric
            
            return metric
    
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get metric by name"""
        with self._lock:
            return self.metrics.get(name)
    
    def get_or_create_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        namespace: str = "frextech"
    ) -> CounterMetric:
        """Get or create a counter metric"""
        metric_def = MetricDefinition(
            name=name,
            type=MetricType.COUNTER,
            description=description,
            labels=labels or [],
            namespace=namespace
        )
        
        metric = self.get_metric(name)
        if metric:
            return metric
        
        return self.register(metric_def)
    
    def get_or_create_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        namespace: str = "frextech"
    ) -> GaugeMetric:
        """Get or create a gauge metric"""
        metric_def = MetricDefinition(
            name=name,
            type=MetricType.GAUGE,
            description=description,
            labels=labels or [],
            namespace=namespace
        )
        
        metric = self.get_metric(name)
        if metric:
            return metric
        
        return self.register(metric_def)
    
    def get_or_create_histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        namespace: str = "frextech",
        buckets: Optional[List[float]] = None
    ) -> HistogramMetric:
        """Get or create a histogram metric"""
        metric_def = MetricDefinition(
            name=name,
            type=MetricType.HISTOGRAM,
            description=description,
            labels=labels or [],
            namespace=namespace,
            buckets=buckets
        )
        
        metric = self.get_metric(name)
        if metric:
            return metric
        
        return self.register(metric_def)
    
    def record_timing(
        self,
        name: str,
        duration_seconds: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record timing metric"""
        metric = self.get_or_create_histogram(
            name=f"{name}_duration_seconds",
            description=f"Duration of {name} in seconds",
            labels=list(labels.keys()) if labels else [],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        metric.record(duration_seconds, labels)
    
    def record_error(
        self,
        name: str,
        error_type: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record error metric"""
        metric = self.get_or_create_counter(
            name=f"{name}_errors_total",
            description=f"Total errors for {name}",
            labels=["error_type"] + (list(labels.keys()) if labels else [])
        )
        
        metric_labels = {"error_type": error_type}
        if labels:
            metric_labels.update(labels)
        
        metric.inc(labels=metric_labels)
    
    def enable_system_metrics(self, interval_seconds: int = 60):
        """Enable system metrics collection"""
        if self.system_metrics_enabled:
            return
        
        self.collection_interval = interval_seconds
        self.system_metrics_enabled = True
        
        # Create system metrics
        self._create_system_metrics()
        
        # Start collection task
        self.system_metrics_task = threading.Thread(
            target=self._collect_system_metrics_loop,
            daemon=True
        )
        self.system_metrics_task.start()
    
    def _create_system_metrics(self):
        """Create system metrics definitions"""
        # CPU metrics
        self.get_or_create_gauge(
            name="system_cpu_percent",
            description="System CPU usage percentage",
            labels=["cpu"],
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="system_cpu_count",
            description="Number of CPU cores",
            namespace="frextech"
        )
        
        # Memory metrics
        self.get_or_create_gauge(
            name="system_memory_total_bytes",
            description="Total system memory in bytes",
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="system_memory_available_bytes",
            description="Available system memory in bytes",
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="system_memory_used_bytes",
            description="Used system memory in bytes",
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="system_memory_percent",
            description="System memory usage percentage",
            namespace="frextech"
        )
        
        # Disk metrics
        self.get_or_create_gauge(
            name="system_disk_total_bytes",
            description="Total disk space in bytes",
            labels=["device", "mountpoint"],
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="system_disk_used_bytes",
            description="Used disk space in bytes",
            labels=["device", "mountpoint"],
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="system_disk_free_bytes",
            description="Free disk space in bytes",
            labels=["device", "mountpoint"],
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="system_disk_percent",
            description="Disk usage percentage",
            labels=["device", "mountpoint"],
            namespace="frextech"
        )
        
        # Network metrics
        self.get_or_create_counter(
            name="system_network_bytes_sent",
            description="Total bytes sent over network",
            labels=["interface"],
            namespace="frextech"
        )
        
        self.get_or_create_counter(
            name="system_network_bytes_received",
            description="Total bytes received over network",
            labels=["interface"],
            namespace="frextech"
        )
        
        self.get_or_create_counter(
            name="system_network_packets_sent",
            description="Total packets sent over network",
            labels=["interface"],
            namespace="frextech"
        )
        
        self.get_or_create_counter(
            name="system_network_packets_received",
            description="Total packets received over network",
            labels=["interface"],
            namespace="frextech"
        )
        
        # Process metrics
        self.get_or_create_gauge(
            name="process_memory_rss_bytes",
            description="Process RSS memory in bytes",
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="process_memory_vms_bytes",
            description="Process VMS memory in bytes",
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="process_cpu_percent",
            description="Process CPU usage percentage",
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="process_thread_count",
            description="Number of threads in process",
            namespace="frextech"
        )
        
        self.get_or_create_gauge(
            name="process_open_files",
            description="Number of open files by process",
            namespace="frextech"
        )
    
    def _collect_system_metrics_loop(self):
        """Collect system metrics periodically"""
        while self.system_metrics_enabled:
            try:
                self._collect_system_metrics()
            except Exception as e:
                print(f"Error collecting system metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_count = psutil.cpu_count()
        
        cpu_gauge = self.get_metric("system_cpu_percent")
        for i, percent in enumerate(cpu_percent):
            cpu_gauge.record(percent, labels={"cpu": str(i)})
        
        self.get_metric("system_cpu_count").record(cpu_count)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.get_metric("system_memory_total_bytes").record(memory.total)
        self.get_metric("system_memory_available_bytes").record(memory.available)
        self.get_metric("system_memory_used_bytes").record(memory.used)
        self.get_metric("system_memory_percent").record(memory.percent)
        
        # Disk metrics
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                
                labels = {
                    "device": partition.device,
                    "mountpoint": partition.mountpoint
                }
                
                self.get_metric("system_disk_total_bytes").record(usage.total, labels)
                self.get_metric("system_disk_used_bytes").record(usage.used, labels)
                self.get_metric("system_disk_free_bytes").record(usage.free, labels)
                self.get_metric("system_disk_percent").record(usage.percent, labels)
            except:
                pass
        
        # Network metrics
        net_io = psutil.net_io_counters(pernic=True)
        for interface, counters in net_io.items():
            labels = {"interface": interface}
            
            self.get_metric("system_network_bytes_sent").record(counters.bytes_sent, labels)
            self.get_metric("system_network_bytes_received").record(counters.bytes_recv, labels)
            self.get_metric("system_network_packets_sent").record(counters.packets_sent, labels)
            self.get_metric("system_network_packets_received").record(counters.packets_recv, labels)
        
        # Process metrics
        process = psutil.Process()
        
        memory_info = process.memory_info()
        self.get_metric("process_memory_rss_bytes").record(memory_info.rss)
        self.get_metric("process_memory_vms_bytes").record(memory_info.vms)
        
        self.get_metric("process_cpu_percent").record(process.cpu_percent())
        self.get_metric("process_thread_count").record(process.num_threads())
        
        try:
            open_files = len(process.open_files())
            self.get_metric("process_open_files").record(open_files)
        except:
            pass
    
    def add_exporter(self, exporter: 'MetricsExporter'):
        """Add metrics exporter"""
        self.exporters.append(exporter)
    
    def remove_exporter(self, exporter: 'MetricsExporter'):
        """Remove metrics exporter"""
        if exporter in self.exporters:
            self.exporters.remove(exporter)
    
    def export_metrics(self):
        """Export metrics to all registered exporters"""
        for exporter in self.exporters:
            try:
                exporter.export(self.metrics)
            except Exception as e:
                print(f"Error exporting metrics with {exporter.__class__.__name__}: {e}")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics as dictionary"""
        result = {}
        
        with self._lock:
            for name, metric in self.metrics.items():
                definition = self.definitions.get(name)
                
                # Get recent values
                values = list(metric.values)[-100:]  # Last 100 values
                
                # Calculate aggregations
                aggregations = {}
                for agg in AggregationMethod:
                    try:
                        aggregation = metric.aggregate(agg, window_seconds=300)
                        aggregations[agg.value] = aggregation.value
                    except:
                        pass
                
                result[name] = {
                    'definition': definition.to_dict() if definition else None,
                    'recent_values': [asdict(v) for v in values],
                    'aggregations': aggregations,
                    'current_value': values[-1].value if values else None
                }
        
        return result
    
    def clear_old_metrics(self, max_age_seconds: int = 3600):
        """Clear metrics older than specified age"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        
        with self._lock:
            for metric in self.metrics.values():
                with metric._lock:
                    # Remove old values
                    metric.values = deque(
                        [v for v in metric.values if v.timestamp > cutoff_time],
                        maxlen=10000
                    )
    
    def shutdown(self):
        """Shutdown metrics registry"""
        self.system_metrics_enabled = False
        
        if self.system_metrics_task:
            self.system_metrics_task.join(timeout=5)
        
        # Export final metrics
        self.export_metrics()
        
        # Clear all metrics
        with self._lock:
            self.metrics.clear()
            self.definitions.clear()
            self.prometheus_metrics.clear()
            self.exporters.clear()


class MetricsExporter:
    """Base class for metrics exporters"""
    
    def export(self, metrics: Dict[str, BaseMetric]):
        """Export metrics"""
        raise NotImplementedError


class JSONExporter(MetricsExporter):
    """Export metrics to JSON format"""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
    
    def export(self, metrics: Dict[str, BaseMetric]):
        """Export metrics to JSON"""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': {}
        }
        
        for name, metric in metrics.items():
            values = list(metric.values)[-100:]  # Last 100 values
            
            data['metrics'][name] = {
                'type': metric.__class__.__name__,
                'description': metric.description,
                'recent_values': [asdict(v) for v in values],
                'total_values': len(metric.values)
            }
        
        json_data = json.dumps(data, default=str, indent=2)
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(json_data)
        
        return json_data


class PrometheusExporter(MetricsExporter):
    """Export metrics to Prometheus format"""
    
    def __init__(self, registry=None):
        self.registry = registry or REGISTRY
    
    def export(self, metrics: Dict[str, BaseMetric]):
        """Export metrics to Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return ""
        
        # Collect metrics from Prometheus registry
        from prometheus_client import generate_latest
        return generate_latest(self.registry).decode('utf-8')


class InfluxDBExporter(MetricsExporter):
    """Export metrics to InfluxDB"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8086,
        database: str = 'metrics',
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        if not INFLUXDB_AVAILABLE:
            raise ImportError("InfluxDB client not installed")
        
        self.client = InfluxDBClient(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database
        )
        self.database = database
    
    def export(self, metrics: Dict[str, BaseMetric]):
        """Export metrics to InfluxDB"""
        points = []
        
        for name, metric in metrics.items():
            for value in metric.values[-100:]:  # Last 100 values
                point = {
                    "measurement": name,
                    "tags": value.labels,
                    "time": value.timestamp.isoformat(),
                    "fields": {
                        "value": value.value
                    }
                }
                points.append(point)
        
        if points:
            self.client.write_points(points)


# Decorators for metrics
def measure_time(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
    registry: Optional[MetricsRegistry] = None
):
    """
    Decorator to measure function execution time
    
    Args:
        metric_name: Name of the metric
        labels: Additional labels for the metric
        registry: Metrics registry to use
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            registry_instance = registry or MetricsRegistry()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success timing
                metric_labels = (labels or {}).copy()
                metric_labels["status"] = "success"
                registry_instance.record_timing(metric_name, duration, metric_labels)
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error timing
                metric_labels = (labels or {}).copy()
                metric_labels["status"] = "error"
                metric_labels["error_type"] = e.__class__.__name__
                registry_instance.record_timing(metric_name, duration, metric_labels)
                
                # Record error count
                registry_instance.record_error(metric_name, e.__class__.__name__, labels)
                
                raise
        
        return wrapper
    
    return decorator


def measure_time_async(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
    registry: Optional[MetricsRegistry] = None
):
    """
    Decorator to measure async function execution time
    
    Args:
        metric_name: Name of the metric
        labels: Additional labels for the metric
        registry: Metrics registry to use
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            registry_instance = registry or MetricsRegistry()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success timing
                metric_labels = (labels or {}).copy()
                metric_labels["status"] = "success"
                registry_instance.record_timing(metric_name, duration, metric_labels)
                
                return result
            
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error timing
                metric_labels = (labels or {}).copy()
                metric_labels["status"] = "error"
                metric_labels["error_type"] = e.__class__.__name__
                registry_instance.record_timing(metric_name, duration, metric_labels)
                
                # Record error count
                registry_instance.record_error(metric_name, e.__class__.__name__, labels)
                
                raise
        
        return wrapper
    
    return decorator


def count_calls(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
    registry: Optional[MetricsRegistry] = None
):
    """
    Decorator to count function calls
    
    Args:
        metric_name: Name of the metric
        labels: Additional labels for the metric
        registry: Metrics registry to use
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            registry_instance = registry or MetricsRegistry()
            
            # Record call
            counter = registry_instance.get_or_create_counter(
                name=f"{metric_name}_calls_total",
                description=f"Total calls to {metric_name}",
                labels=list(labels.keys()) if labels else []
            )
            counter.inc(labels=labels)
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                success_counter = registry_instance.get_or_create_counter(
                    name=f"{metric_name}_success_total",
                    description=f"Successful calls to {metric_name}",
                    labels=list(labels.keys()) if labels else []
                )
                success_counter.inc(labels=labels)
                
                return result
            
            except Exception as e:
                # Record failure
                failure_counter = registry_instance.get_or_create_counter(
                    name=f"{metric_name}_failure_total",
                    description=f"Failed calls to {metric_name}",
                    labels=["error_type"] + (list(labels.keys()) if labels else [])
                )
                
                failure_labels = {"error_type": e.__class__.__name__}
                if labels:
                    failure_labels.update(labels)
                
                failure_counter.inc(labels=failure_labels)
                raise
        
        return wrapper
    
    return decorator


# Context managers for metrics
@contextmanager
def measure_context(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
    registry: Optional[MetricsRegistry] = None
):
    """
    Context manager to measure execution time
    
    Args:
        metric_name: Name of the metric
        labels: Additional labels for the metric
        registry: Metrics registry to use
    """
    registry_instance = registry or MetricsRegistry()
    start_time = time.time()
    
    try:
        yield
        duration = time.time() - start_time
        
        # Record success timing
        metric_labels = (labels or {}).copy()
        metric_labels["status"] = "success"
        registry_instance.record_timing(metric_name, duration, metric_labels)
    
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error timing
        metric_labels = (labels or {}).copy()
        metric_labels["status"] = "error"
        metric_labels["error_type"] = e.__class__.__name__
        registry_instance.record_timing(metric_name, duration, metric_labels)
        
        # Record error count
        registry_instance.record_error(metric_name, e.__class__.__name__, labels)
        
        raise


@asynccontextmanager
async def measure_context_async(
    metric_name: str,
    labels: Optional[Dict[str, str]] = None,
    registry: Optional[MetricsRegistry] = None
):
    """
    Async context manager to measure execution time
    
    Args:
        metric_name: Name of the metric
        labels: Additional labels for the metric
        registry: Metrics registry to use
    """
    registry_instance = registry or MetricsRegistry()
    start_time = time.time()
    
    try:
        yield
        duration = time.time() - start_time
        
        # Record success timing
        metric_labels = (labels or {}).copy()
        metric_labels["status"] = "success"
        registry_instance.record_timing(metric_name, duration, metric_labels)
    
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error timing
        metric_labels = (labels or {}).copy()
        metric_labels["status"] = "error"
        metric_labels["error_type"] = e.__class__.__name__
        registry_instance.record_timing(metric_name, duration, metric_labels)
        
        # Record error count
        registry_instance.record_error(metric_name, e.__class__.__name__, labels)
        
        raise


# Utility functions
def get_metrics_registry() -> MetricsRegistry:
    """Get the global metrics registry"""
    return MetricsRegistry()


def setup_default_metrics(namespace: str = "frextech"):
    """Setup default metrics for the application"""
    registry = MetricsRegistry()
    
    # API metrics
    registry.get_or_create_counter(
        name="api_requests_total",
        description="Total API requests",
        labels=["method", "endpoint", "status_code"],
        namespace=namespace
    )
    
    registry.get_or_create_histogram(
        name="api_request_duration_seconds",
        description="API request duration in seconds",
        labels=["method", "endpoint"],
        namespace=namespace,
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    # Model metrics
    registry.get_or_create_counter(
        name="model_inferences_total",
        description="Total model inferences",
        labels=["model_type", "model_name"],
        namespace=namespace
    )
    
    registry.get_or_create_histogram(
        name="model_inference_duration_seconds",
        description="Model inference duration in seconds",
        labels=["model_type", "model_name"],
        namespace=namespace,
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    
    registry.get_or_create_gauge(
        name="model_memory_usage_bytes",
        description="Model memory usage in bytes",
        labels=["model_type", "model_name"],
        namespace=namespace
    )
    
    # Cache metrics
    registry.get_or_create_counter(
        name="cache_hits_total",
        description="Total cache hits",
        labels=["cache_name"],
        namespace=namespace
    )
    
    registry.get_or_create_counter(
        name="cache_misses_total",
        description="Total cache misses",
        labels=["cache_name"],
        namespace=namespace
    )
    
    registry.get_or_create_gauge(
        name="cache_size_bytes",
        description="Cache size in bytes",
        labels=["cache_name"],
        namespace=namespace
    )
    
    # Enable system metrics
    registry.enable_system_metrics()
    
    return registry


# Export
__all__ = [
    'MetricType',
    'AggregationMethod',
    'MetricDefinition',
    'MetricValue',
    'MetricAggregation',
    'BaseMetric',
    'CounterMetric',
    'GaugeMetric',
    'HistogramMetric',
    'SummaryMetric',
    'MetricsRegistry',
    'MetricsExporter',
    'JSONExporter',
    'PrometheusExporter',
    'InfluxDBExporter',
    'measure_time',
    'measure_time_async',
    'count_calls',
    'measure_context',
    'measure_context_async',
    'get_metrics_registry',
    'setup_default_metrics'
]
