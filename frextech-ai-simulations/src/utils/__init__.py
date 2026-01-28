"""
FrexTech AI Simulations - Utility Module
Common utilities for logging, file I/O, distributed computing, and visualization.
"""

from .logging_config import (
    setup_logging, get_logger,
    LogLevel, LogFormatter, JSONFormatter,
    FileHandler, ConsoleHandler, RotatingFileHandler
)

from .distributed import (
    DistributedManager, NodeRole,
    init_distributed, get_rank, get_world_size,
    broadcast, all_reduce, all_gather,
    DistributedSampler, DistributedDataParallel
)

from .file_io import (
    FileManager, DataLoader,
    save_json, load_json,
    save_pickle, load_pickle,
    save_numpy, load_numpy,
    save_image, load_image,
    save_video, load_video,
    compress_data, decompress_data,
    FileType, CompressionType
)

from .metrics import (
    MetricsCollector, MetricType,
    compute_psnr, compute_ssim, compute_lpips,
    compute_fid, compute_accuracy, compute_precision_recall,
    Timer, Profiler, MemoryTracker
)

from .visualization import (
    VisualizationEngine, PlotType,
    create_heatmap, create_scatter_plot,
    create_line_plot, create_bar_plot,
    create_3d_plot, create_animation,
    export_to_html, export_to_pdf,
    ColorMap, PlotStyle
)

__version__ = "1.0.0"
__author__ = "FrexTech AI Simulations Team"
__all__ = [
    # Logging
    "setup_logging", "get_logger",
    "LogLevel", "LogFormatter", "JSONFormatter",
    "FileHandler", "ConsoleHandler", "RotatingFileHandler",
    
    # Distributed Computing
    "DistributedManager", "NodeRole",
    "init_distributed", "get_rank", "get_world_size",
    "broadcast", "all_reduce", "all_gather",
    "DistributedSampler", "DistributedDataParallel",
    
    # File I/O
    "FileManager", "DataLoader",
    "save_json", "load_json",
    "save_pickle", "load_pickle",
    "save_numpy", "load_numpy",
    "save_image", "load_image",
    "save_video", "load_video",
    "compress_data", "decompress_data",
    "FileType", "CompressionType",
    
    # Metrics
    "MetricsCollector", "MetricType",
    "compute_psnr", "compute_ssim", "compute_lpips",
    "compute_fid", "compute_accuracy", "compute_precision_recall",
    "Timer", "Profiler", "MemoryTracker",
    
    # Visualization
    "VisualizationEngine", "PlotType",
    "create_heatmap", "create_scatter_plot",
    "create_line_plot", "create_bar_plot",
    "create_3d_plot", "create_animation",
    "export_to_html", "export_to_pdf",
    "ColorMap", "PlotStyle",
    
    # Version
    "__version__", "__author__"
]
