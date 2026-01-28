"""
Visualization utilities for FrexTech AI Simulations
Provides tools for visualizing worlds, metrics, and system data
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
import base64
import io
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import colorsys
from enum import Enum
import warnings

# Third-party imports (optional)
try:
    import trimesh
    import pyvista as pv
    TRIMESH_AVAILABLE = True
    PYVISTA_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    PYVISTA_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False


class ColorMap(str, Enum):
    """Color map types"""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    TWILIGHT = "twilight"
    TWILIGHT_SHIFTED = "twilight_shifted"
    TURBO = "turbo"
    HSV = "hsv"
    HOT = "hot"
    COOL = "cool"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"
    BONE = "bone"
    COPPER = "copper"
    GREYS = "greys"
    JET = "jet"
    RAINBOW = "rainbow"


class VisualizationType(str, Enum):
    """Visualization types"""
    LINE = "line"
    SCATTER = "scatter"
    BAR = "bar"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    HEATMAP = "heatmap"
    CONTOUR = "contour"
    SURFACE = "surface"
    VOLUME = "volume"
    MESH = "mesh"
    POINT_CLOUD = "point_cloud"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    width: int = 800
    height: int = 600
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""
    color_map: ColorMap = ColorMap.VIRIDIS
    background_color: str = "white"
    grid: bool = True
    legend: bool = True
    interactive: bool = True
    dark_mode: bool = False
    font_size: int = 12
    dpi: int = 100
    save_format: str = "png"
    transparent: bool = False


@dataclass
class PointCloudData:
    """Point cloud data structure"""
    points: np.ndarray  # Shape: (N, 3)
    colors: Optional[np.ndarray] = None  # Shape: (N, 3) or (N, 4)
    normals: Optional[np.ndarray] = None  # Shape: (N, 3)
    intensities: Optional[np.ndarray] = None  # Shape: (N,)
    
    def validate(self):
        """Validate point cloud data"""
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")
        
        if self.colors is not None:
            if self.colors.ndim != 2 or self.colors.shape[1] not in [3, 4]:
                raise ValueError("Colors must have shape (N, 3) or (N, 4)")
            if len(self.colors) != len(self.points):
                raise ValueError("Colors must have same length as points")
        
        if self.normals is not None:
            if self.normals.ndim != 2 or self.normals.shape[1] != 3:
                raise ValueError("Normals must have shape (N, 3)")
            if len(self.normals) != len(self.points):
                raise ValueError("Normals must have same length as points")
        
        if self.intensities is not None:
            if self.intensities.ndim != 1:
                raise ValueError("Intensities must have shape (N,)")
            if len(self.intensities) != len(self.points):
                raise ValueError("Intensities must have same length as points")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'points': self.points.tolist(),
            'num_points': len(self.points)
        }
        
        if self.colors is not None:
            result['colors'] = self.colors.tolist()
        
        if self.normals is not None:
            result['normals'] = self.normals.tolist()
        
        if self.intensities is not None:
            result['intensities'] = self.intensities.tolist()
        
        return result


@dataclass
class MeshData:
    """Mesh data structure"""
    vertices: np.ndarray  # Shape: (N, 3)
    faces: np.ndarray  # Shape: (M, 3) for triangles
    vertex_colors: Optional[np.ndarray] = None  # Shape: (N, 3) or (N, 4)
    vertex_normals: Optional[np.ndarray] = None  # Shape: (N, 3)
    face_normals: Optional[np.ndarray] = None  # Shape: (M, 3)
    texture_coords: Optional[np.ndarray] = None  # Shape: (N, 2)
    texture: Optional[np.ndarray] = None  # Image data
    
    def validate(self):
        """Validate mesh data"""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3)")
        
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("Faces must have shape (M, 3)")
        
        if self.vertex_colors is not None:
            if self.vertex_colors.ndim != 2 or self.vertex_colors.shape[1] not in [3, 4]:
                raise ValueError("Vertex colors must have shape (N, 3) or (N, 4)")
            if len(self.vertex_colors) != len(self.vertices):
                raise ValueError("Vertex colors must have same length as vertices")
        
        if self.vertex_normals is not None:
            if self.vertex_normals.ndim != 2 or self.vertex_normals.shape[1] != 3:
                raise ValueError("Vertex normals must have shape (N, 3)")
            if len(self.vertex_normals) != len(self.vertices):
                raise ValueError("Vertex normals must have same length as vertices")
        
        if self.face_normals is not None:
            if self.face_normals.ndim != 2 or self.face_normals.shape[1] != 3:
                raise ValueError("Face normals must have shape (M, 3)")
            if len(self.face_normals) != len(self.faces):
                raise ValueError("Face normals must have same length as faces")
        
        if self.texture_coords is not None:
            if self.texture_coords.ndim != 2 or self.texture_coords.shape[1] != 2:
                raise ValueError("Texture coordinates must have shape (N, 2)")
            if len(self.texture_coords) != len(self.vertices):
                raise ValueError("Texture coordinates must have same length as vertices")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'vertices': self.vertices.tolist(),
            'faces': self.faces.tolist(),
            'num_vertices': len(self.vertices),
            'num_faces': len(self.faces)
        }
        
        if self.vertex_colors is not None:
            result['vertex_colors'] = self.vertex_colors.tolist()
        
        if self.vertex_normals is not None:
            result['vertex_normals'] = self.vertex_normals.tolist()
        
        if self.face_normals is not None:
            result['face_normals'] = self.face_normals.tolist()
        
        if self.texture_coords is not None:
            result['texture_coords'] = self.texture_coords.tolist()
        
        if self.texture is not None:
            result['texture_shape'] = self.texture.shape
        
        return result


class VisualizationEngine:
    """Main visualization engine for creating various visualizations"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Set matplotlib style
        if SEABORN_AVAILABLE:
            sns.set_style("darkgrid" if self.config.dark_mode else "whitegrid")
            if self.config.dark_mode:
                plt.style.use('dark_background')
    
    def create_line_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create a line plot
        
        Args:
            x_data: X-axis data (can be 1D or 2D for multiple lines)
            y_data: Y-axis data (can be 1D or 2D for multiple lines)
            labels: Labels for each line
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        
        # Handle multiple lines
        if x_data.ndim == 1:
            x_data = np.expand_dims(x_data, axis=0)
            y_data = np.expand_dims(y_data, axis=0)
        
        num_lines = x_data.shape[0]
        
        # Create color palette
        colors = self._get_color_palette(num_lines)
        
        for i in range(num_lines):
            label = labels[i] if labels and i < len(labels) else None
            ax.plot(x_data[i], y_data[i], color=colors[i], label=label, linewidth=2)
        
        # Set labels and title
        ax.set_xlabel(x_label or self.config.x_label, fontsize=self.config.font_size)
        ax.set_ylabel(y_label or self.config.y_label, fontsize=self.config.font_size)
        ax.set_title(title or self.config.title, fontsize=self.config.font_size + 2)
        
        # Add grid and legend
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        if self.config.legend and labels:
            ax.legend(fontsize=self.config.font_size - 2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def create_scatter_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create a scatter plot
        
        Args:
            x_data: X coordinates
            y_data: Y coordinates
            colors: Point colors (can be single color, array of colors, or array of values)
            sizes: Point sizes
            labels: Labels for different groups
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        
        # Handle colors
        if colors is None:
            colors = 'blue'
        elif isinstance(colors, np.ndarray):
            if colors.ndim == 1:
                # Color values, use colormap
                scatter = ax.scatter(x_data, y_data, c=colors, cmap=self.config.color_map.value,
                                    s=sizes or 20, alpha=0.7)
                # Add colorbar
                plt.colorbar(scatter, ax=ax)
                colors = None
            elif colors.ndim == 2 and colors.shape[1] in [3, 4]:
                # RGB/RGBA colors
                colors = colors
        
        # Plot scatter
        if isinstance(colors, (list, np.ndarray)) and not isinstance(colors, str):
            if labels and len(labels) == len(x_data):
                # Group by labels
                unique_labels = np.unique(labels)
                for label in unique_labels:
                    mask = labels == label
                    ax.scatter(x_data[mask], y_data[mask],
                              color=colors[mask][0] if colors.ndim == 2 else colors,
                              s=sizes[mask] if sizes is not None else 20,
                              label=label, alpha=0.7)
            else:
                ax.scatter(x_data, y_data, c=colors, s=sizes or 20, alpha=0.7)
        else:
            ax.scatter(x_data, y_data, c=colors, s=sizes or 20, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel(x_label or self.config.x_label, fontsize=self.config.font_size)
        ax.set_ylabel(y_label or self.config.y_label, fontsize=self.config.font_size)
        ax.set_title(title or self.config.title, fontsize=self.config.font_size + 2)
        
        # Add grid and legend
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        if self.config.legend and labels is not None:
            ax.legend(fontsize=self.config.font_size - 2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def create_3d_scatter(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Union[go.Figure, Tuple[Figure, Axes]]:
        """
        Create 3D scatter plot
        
        Args:
            points: 3D points (N, 3)
            colors: Point colors
            sizes: Point sizes
            title: Plot title
            save_path: Path to save the plot
            interactive: Whether to use Plotly for interactive plot
        
        Returns:
            Plotly figure or matplotlib figure and axes
        """
        if interactive:
            return self._create_3d_scatter_plotly(points, colors, sizes, title, save_path)
        else:
            return self._create_3d_scatter_matplotlib(points, colors, sizes, title, save_path)
    
    def _create_3d_scatter_plotly(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create 3D scatter plot using Plotly"""
        # Prepare colors
        if colors is None:
            colors = np.linalg.norm(points - points.mean(axis=0), axis=1)
            color_continuous_scale = self.config.color_map.value
        elif colors.ndim == 1:
            color_continuous_scale = self.config.color_map.value
        else:
            # RGB colors
            colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
            color_continuous_scale = None
        
        # Prepare sizes
        marker_size = sizes if sizes is not None else 2
        if isinstance(marker_size, np.ndarray):
            marker_size = marker_size.tolist()
        
        # Create plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=colors,
                    colorscale=color_continuous_scale,
                    opacity=0.7,
                    colorbar=dict(title='Value' if colors.ndim == 1 else None)
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=title or self.config.title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                bgcolor=self.config.background_color
            ),
            width=self.config.width,
            height=self.config.height,
            template='plotly_dark' if self.config.dark_mode else 'plotly_white'
        )
        
        # Save if requested
        if save_path:
            fig.write_image(save_path, width=self.config.width, height=self.config.height)
        
        return fig
    
    def _create_3d_scatter_matplotlib(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """Create 3D scatter plot using Matplotlib"""
        fig = plt.figure(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare colors
        if colors is None:
            colors = np.linalg.norm(points - points.mean(axis=0), axis=1)
        
        # Plot
        scatter = ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors,
            cmap=self.config.color_map.value,
            s=sizes or 2,
            alpha=0.7
        )
        
        # Add colorbar
        if colors.ndim == 1:
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels and title
        ax.set_xlabel('X', fontsize=self.config.font_size)
        ax.set_ylabel('Y', fontsize=self.config.font_size)
        ax.set_zlabel('Z', fontsize=self.config.font_size)
        ax.set_title(title or self.config.title, fontsize=self.config.font_size + 2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def create_heatmap(
        self,
        data: np.ndarray,
        x_labels: Optional[List] = None,
        y_labels: Optional[List] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        annotate: bool = False,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create a heatmap
        
        Args:
            data: 2D data array
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            annotate: Whether to annotate cells with values
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        
        # Create heatmap
        im = ax.imshow(data, cmap=self.config.color_map.value, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        if x_labels is not None:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        if y_labels is not None:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)
        
        ax.set_xlabel(x_label or self.config.x_label, fontsize=self.config.font_size)
        ax.set_ylabel(y_label or self.config.y_label, fontsize=self.config.font_size)
        ax.set_title(title or self.config.title, fontsize=self.config.font_size + 2)
        
        # Annotate cells
        if annotate:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = ax.text(j, i, f'{data[i, j]:.2f}',
                                  ha="center", va="center",
                                  color="white" if data[i, j] > data.mean() else "black",
                                  fontsize=self.config.font_size - 4)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def create_histogram(
        self,
        data: Union[np.ndarray, List[np.ndarray]],
        bins: int = 30,
        density: bool = False,
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create histogram
        
        Args:
            data: Data to histogram (can be multiple datasets)
            bins: Number of bins
            density: Whether to normalize to form probability density
            labels: Labels for each dataset
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        
        # Handle multiple datasets
        if isinstance(data, list) or (isinstance(data, np.ndarray) and data.ndim > 1):
            # Multiple datasets
            colors = self._get_color_palette(len(data))
            
            for i, dataset in enumerate(data):
                label = labels[i] if labels and i < len(labels) else None
                ax.hist(dataset, bins=bins, density=density, alpha=0.7,
                       color=colors[i], label=label, edgecolor='black')
        else:
            # Single dataset
            ax.hist(data, bins=bins, density=density, alpha=0.7,
                   color='blue', edgecolor='black')
        
        # Set labels and title
        ax.set_xlabel(x_label or self.config.x_label, fontsize=self.config.font_size)
        ax.set_ylabel(y_label or 'Density' if density else 'Count', fontsize=self.config.font_size)
        ax.set_title(title or self.config.title, fontsize=self.config.font_size + 2)
        
        # Add grid and legend
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        if self.config.legend and labels:
            ax.legend(fontsize=self.config.font_size - 2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def create_box_plot(
        self,
        data: List[np.ndarray],
        labels: Optional[List[str]] = None,
        title: Optional[str] = None,
        y_label: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create box plot
        
        Args:
            data: List of datasets
            labels: Labels for each dataset
            title: Plot title
            y_label: Y-axis label
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        
        # Create box plot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = self._get_color_palette(len(data))
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(colors[i])
            box.set_alpha(0.7)
        
        # Color medians
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)
        
        # Set labels and title
        ax.set_ylabel(y_label or self.config.y_label, fontsize=self.config.font_size)
        ax.set_title(title or self.config.title, fontsize=self.config.font_size + 2)
        
        # Add grid
        if self.config.grid:
            ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def create_time_series(
        self,
        timestamps: List[datetime],
        values: List[float],
        title: Optional[str] = None,
        y_label: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """
        Create time series plot
        
        Args:
            timestamps: List of datetime objects
            values: Corresponding values
            title: Plot title
            y_label: Y-axis label
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        
        # Plot time series
        ax.plot(timestamps, values, 'b-', linewidth=2, alpha=0.8)
        
        # Fill under curve
        ax.fill_between(timestamps, values, alpha=0.3, color='blue')
        
        # Format x-axis for dates
        fig.autofmt_xdate()
        
        # Set labels and title
        ax.set_ylabel(y_label or self.config.y_label, fontsize=self.config.font_size)
        ax.set_title(title or self.config.title, fontsize=self.config.font_size + 2)
        
        # Add grid
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def visualize_point_cloud(
        self,
        point_cloud: PointCloudData,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Union[go.Figure, Tuple[Figure, Axes]]:
        """
        Visualize point cloud
        
        Args:
            point_cloud: Point cloud data
            title: Plot title
            save_path: Path to save the visualization
            interactive: Whether to use interactive visualization
        
        Returns:
            Visualization figure
        """
        point_cloud.validate()
        
        # Prepare colors
        if point_cloud.colors is not None:
            colors = point_cloud.colors
            if colors.shape[1] == 3:
                # RGB to Plotly format
                colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in colors]
        elif point_cloud.intensities is not None:
            colors = point_cloud.intensities
        else:
            # Use height for color
            colors = point_cloud.points[:, 2]
        
        if interactive:
            # Use Plotly for interactive visualization
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=point_cloud.points[:, 0],
                    y=point_cloud.points[:, 1],
                    z=point_cloud.points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=colors,
                        colorscale=self.config.color_map.value if isinstance(colors, np.ndarray) and colors.ndim == 1 else None,
                        opacity=0.7
                    )
                )
            ])
            
            fig.update_layout(
                title=title or "Point Cloud",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    bgcolor=self.config.background_color
                ),
                width=self.config.width,
                height=self.config.height,
                template='plotly_dark' if self.config.dark_mode else 'plotly_white'
            )
            
            if save_path:
                fig.write_image(save_path, width=self.config.width, height=self.config.height)
            
            return fig
        
        else:
            # Use matplotlib
            fig = plt.figure(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                point_cloud.points[:, 0],
                point_cloud.points[:, 1],
                point_cloud.points[:, 2],
                c=colors if isinstance(colors, np.ndarray) and colors.ndim == 1 else 'blue',
                cmap=self.config.color_map.value,
                s=1,
                alpha=0.7
            )
            
            ax.set_xlabel('X', fontsize=self.config.font_size)
            ax.set_ylabel('Y', fontsize=self.config.font_size)
            ax.set_zlabel('Z', fontsize=self.config.font_size)
            ax.set_title(title or "Point Cloud", fontsize=self.config.font_size + 2)
            
            if isinstance(colors, np.ndarray) and colors.ndim == 1:
                plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
            
            plt.tight_layout()
            
            if save_path:
                self._save_figure(fig, save_path)
            
            return fig, ax
    
    def visualize_mesh(
        self,
        mesh: MeshData,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Optional[go.Figure]:
        """
        Visualize 3D mesh
        
        Args:
            mesh: Mesh data
            title: Plot title
            save_path: Path to save the visualization
            interactive: Whether to use interactive visualization
        
        Returns:
            Plotly figure if interactive, None otherwise
        """
        mesh.validate()
        
        if not interactive or not TRIMESH_AVAILABLE:
            # Fall back to simple visualization
            return self._visualize_mesh_simple(mesh, title, save_path)
        
        # Use Plotly for interactive visualization
        fig = go.Figure(data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color='lightblue',
                opacity=0.7,
                flatshading=True
            )
        ])
        
        # Add vertex colors if available
        if mesh.vertex_colors is not None:
            if mesh.vertex_colors.shape[1] == 3:
                colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' 
                         for c in mesh.vertex_colors]
                fig.data[0].vertexcolor = colors
        
        fig.update_layout(
            title=title or "3D Mesh",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                bgcolor=self.config.background_color
            ),
            width=self.config.width,
            height=self.config.height,
            template='plotly_dark' if self.config.dark_mode else 'plotly_white'
        )
        
        if save_path:
            fig.write_image(save_path, width=self.config.width, height=self.config.height)
        
        return fig
    
    def _visualize_mesh_simple(
        self,
        mesh: MeshData,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Figure, Axes]:
        """Simple mesh visualization using matplotlib"""
        fig = plt.figure(figsize=(self.config.width/100, self.config.height/100), dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot wireframe
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            # Close the triangle
            vertices = np.vstack([vertices, vertices[0]])
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'b-', alpha=0.3)
        
        # Plot vertices
        ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                  c='red', s=10, alpha=0.7)
        
        ax.set_xlabel('X', fontsize=self.config.font_size)
        ax.set_ylabel('Y', fontsize=self.config.font_size)
        ax.set_zlabel('Z', fontsize=self.config.font_size)
        ax.set_title(title or "3D Mesh", fontsize=self.config.font_size + 2)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig, ax
    
    def create_dashboard(
        self,
        plots: List[Tuple[str, Figure]],
        title: Optional[str] = None,
        layout: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create a dashboard with multiple plots
        
        Args:
            plots: List of (title, figure) tuples
            title: Dashboard title
            layout: Grid layout (rows, cols)
            save_path: Path to save the dashboard
        
        Returns:
            Combined figure
        """
        num_plots = len(plots)
        
        # Determine layout
        if layout is None:
            cols = min(3, num_plots)
            rows = (num_plots + cols - 1) // cols
        else:
            rows, cols = layout
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, 
                                figsize=(self.config.width/100 * cols/3, 
                                        self.config.height/100 * rows/2),
                                dpi=self.config.dpi,
                                squeeze=False)
        
        # Flatten axes array
        axes_flat = axes.flatten()
        
        # Add plots
        for i, (plot_title, plot_fig) in enumerate(plots):
            if i >= len(axes_flat):
                break
            
            # Get the plot from the figure
            plot_ax = plot_fig.axes[0]
            
            # Clear target axis
            axes_flat[i].clear()
            
            # Copy plot content
            for line in plot_ax.get_lines():
                axes_flat[i].plot(line.get_xdata(), line.get_ydata(),
                                 color=line.get_color(),
                                 linewidth=line.get_linewidth(),
                                 label=line.get_label())
            
            for collection in plot_ax.collections:
                # This is simplified - in practice, you'd need to handle different types
                pass
            
            # Copy labels and title
            axes_flat[i].set_xlabel(plot_ax.get_xlabel())
            axes_flat[i].set_ylabel(plot_ax.get_ylabel())
            axes_flat[i].set_title(plot_title)
            
            # Copy grid
            if plot_ax.get_xgridlines():
                axes_flat[i].grid(True, alpha=0.3)
            
            # Copy legend
            if plot_ax.get_legend():
                axes_flat[i].legend()
        
        # Hide unused axes
        for i in range(len(plots), len(axes_flat)):
            axes_flat[i].axis('off')
        
        # Set main title
        if title:
            fig.suptitle(title, fontsize=self.config.font_size + 4, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def figure_to_base64(self, fig: Figure) -> str:
        """
        Convert matplotlib figure to base64 string
        
        Args:
            fig: Matplotlib figure
        
        Returns:
            Base64 encoded image string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.config.dpi,
                   bbox_inches='tight', transparent=self.config.transparent)
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        
        return img_base64
    
    def plotly_to_html(self, fig: go.Figure, include_plotlyjs: bool = True) -> str:
        """
        Convert Plotly figure to HTML
        
        Args:
            fig: Plotly figure
            include_plotlyjs: Whether to include Plotly.js library
        
        Returns:
            HTML string
        """
        return fig.to_html(include_plotlyjs=include_plotlyjs,
                          full_html=False,
                          default_width=f'{self.config.width}px',
                          default_height=f'{self.config.height}px')
    
    def _get_color_palette(self, n_colors: int) -> List[str]:
        """Get color palette for given number of colors"""
        if SEABORN_AVAILABLE:
            return sns.color_palette(self.config.color_map.value, n_colors)
        else:
            cmap = cm.get_cmap(self.config.color_map.value)
            return [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
    
    def _save_figure(self, fig: Figure, path: str):
        """Save figure to file"""
        fig.savefig(path, format=self.config.save_format,
                   dpi=self.config.dpi,
                   bbox_inches='tight',
                   transparent=self.config.transparent,
                   facecolor='black' if self.config.dark_mode else 'white')


class WorldVisualizer:
    """Specialized visualizer for AI-generated worlds"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.engine = VisualizationEngine(config)
        self.config = config or VisualizationConfig()
    
    def visualize_world_comparison(
        self,
        world_data_before: Dict[str, Any],
        world_data_after: Dict[str, Any],
        title: str = "World Comparison",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize comparison between two worlds
        
        Args:
            world_data_before: World data before changes
            world_data_after: World data after changes
            title: Plot title
            save_path: Path to save the visualization
        
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Before - Top View', 'After - Top View',
                           'Before - 3D View', 'After - 3D View'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}],
                   [{'type': 'scene'}, {'type': 'scene'}]]
        )
        
        # Extract point clouds or mesh data
        # This is simplified - in practice, you'd extract actual data
        points_before = self._extract_points(world_data_before)
        points_after = self._extract_points(world_data_after)
        
        # Top view (2D projection)
        fig.add_trace(
            go.Scatter(
                x=points_before[:, 0],
                y=points_before[:, 1],
                mode='markers',
                marker=dict(size=1, color='blue', opacity=0.5),
                name='Before'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=points_after[:, 0],
                y=points_after[:, 1],
                mode='markers',
                marker=dict(size=1, color='red', opacity=0.5),
                name='After'
            ),
            row=1, col=2
        )
        
        # 3D view
        fig.add_trace(
            go.Scatter3d(
                x=points_before[:, 0],
                y=points_before[:, 1],
                z=points_before[:, 2],
                mode='markers',
                marker=dict(size=1, color='blue', opacity=0.5),
                name='Before'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter3d(
                x=points_after[:, 0],
                y=points_after[:, 1],
                z=points_after[:, 2],
                mode='markers',
                marker=dict(size=1, color='red', opacity=0.5),
                name='After'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            width=self.config.width * 1.5,
            height=self.config.height * 1.5,
            template='plotly_dark' if self.config.dark_mode else 'plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="X", row=1, col=1)
        fig.update_yaxes(title_text="Y", row=1, col=1)
        fig.update_xaxes(title_text="X", row=1, col=2)
        fig.update_yaxes(title_text="Y", row=1, col=2)
        
        fig.update_scenes(
            dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            row=2, col=1
        )
        fig.update_scenes(
            dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
            row=2, col=2
        )
        
        if save_path:
            fig.write_image(save_path, width=fig.layout.width, height=fig.layout.height)
        
        return fig
    
    def visualize_world_evolution(
        self,
        world_sequence: List[Dict[str, Any]],
        title: str = "World Evolution",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Visualize evolution of a world over time
        
        Args:
            world_sequence: Sequence of world states
            title: Plot title
            save_path: Path to save the visualization
        
        Returns:
            Plotly figure
        """
        num_steps = len(world_sequence)
        
        # Create subplots
        cols = min(4, num_steps)
        rows = (num_steps + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"Step {i+1}" for i in range(num_steps)],
            specs=[[{'type': 'scene'} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Add each world state
        for i, world_data in enumerate(world_sequence):
            row = i // cols + 1
            col = i % cols + 1
            
            points = self._extract_points(world_data)
            
            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=points[:, 2],  # Color by height
                        colorscale=self.config.color_map.value,
                        opacity=0.7
                    ),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=self.config.width * cols / 2,
            height=self.config.height * rows / 2,
            template='plotly_dark' if self.config.dark_mode else 'plotly_white'
        )
        
        # Update all scenes
        for i in range(1, num_steps + 1):
            fig.update_scenes(
                dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                row=(i-1)//cols + 1, col=(i-1)%cols + 1
            )
        
        if save_path:
            fig.write_image(save_path, width=fig.layout.width, height=fig.layout.height)
        
        return fig
    
    def create_world_statistics(
        self,
        world_data: Dict[str, Any],
        title: str = "World Statistics",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Create statistics visualization for a world
        
        Args:
            world_data: World data
            title: Plot title
            save_path: Path to save the visualization
        
        Returns:
            Matplotlib figure
        """
        points = self._extract_points(world_data)
        
        # Calculate statistics
        x_values = points[:, 0]
        y_values = points[:, 1]
        z_values = points[:, 2]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 3D scatter plot
        ax = axes[0, 0]
        scatter = ax.scatter(x_values, y_values, c=z_values, cmap=self.config.color_map.value, s=1, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Top View')
        plt.colorbar(scatter, ax=ax)
        
        # 2. Height distribution
        ax = axes[0, 1]
        ax.hist(z_values, bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Height (Z)')
        ax.set_ylabel('Count')
        ax.set_title('Height Distribution')
        ax.grid(True, alpha=0.3)
        
        # 3. Density heatmap
        ax = axes[0, 2]
        h = ax.hist2d(x_values, y_values, bins=50, cmap=self.config.color_map.value)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Density Heatmap')
        plt.colorbar(h[3], ax=ax)
        
        # 4. X-Y distribution
        ax = axes[1, 0]
        ax.boxplot([x_values, y_values], labels=['X', 'Y'])
        ax.set_ylabel('Value')
        ax.set_title('X-Y Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 5. Cumulative distribution
        ax = axes[1, 1]
        for values, label in zip([x_values, y_values, z_values], ['X', 'Y', 'Z']):
            sorted_values = np.sort(values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax.plot(sorted_values, cumulative, label=label, linewidth=2)
        ax.set_xlabel('Value')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Statistics table
        ax = axes[1, 2]
        ax.axis('off')
        
        stats_text = [
            f'Total Points: {len(points):,}',
            f'X Range: [{x_values.min():.2f}, {x_values.max():.2f}]',
            f'Y Range: [{y_values.min():.2f}, {y_values.max():.2f}]',
            f'Z Range: [{z_values.min():.2f}, {z_values.max():.2f}]',
            f'Mean X: {x_values.mean():.2f}',
            f'Mean Y: {y_values.mean():.2f}',
            f'Mean Z: {z_values.mean():.2f}',
            f'Std X: {x_values.std():.2f}',
            f'Std Y: {y_values.std():.2f}',
            f'Std Z: {z_values.std():.2f}'
        ]
        
        ax.text(0.1, 0.5, '\n'.join(stats_text),
                fontsize=10, family='monospace',
                verticalalignment='center')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def _extract_points(self, world_data: Dict[str, Any]) -> np.ndarray:
        """Extract points from world data (simplified)"""
        # This is a simplified implementation
        # In practice, you'd extract based on world format
        if 'points' in world_data:
            return np.array(world_data['points'])
        elif 'vertices' in world_data:
            return np.array(world_data['vertices'])
        else:
            # Generate random points for demonstration
            return np.random.randn(1000, 3)


# Utility functions
def create_color_gradient(
    n_colors: int,
    colormap: ColorMap = ColorMap.VIRIDIS,
    start: float = 0.0,
    end: float = 1.0
) -> np.ndarray:
    """Create color gradient"""
    cmap = plt.cm.get_cmap(colormap.value)
    positions = np.linspace(start, end, n_colors)
    return cmap(positions)[:, :3]  # RGB only, no alpha


def normalize_colors(colors: np.ndarray) -> np.ndarray:
    """Normalize colors to [0, 1] range"""
    if colors.dtype == np.uint8:
        return colors.astype(np.float32) / 255.0
    return colors


def calculate_bounding_box(points: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate bounding box of points"""
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    extent = max_coords - min_coords
    
    return {
        'min': min_coords,
        'max': max_coords,
        'center': center,
        'extent': extent,
        'diagonal': np.linalg.norm(extent)
    }


def downsample_point_cloud(
    points: np.ndarray,
    target_count: int,
    method: str = 'random'
) -> np.ndarray:
    """Downsample point cloud"""
    if len(points) <= target_count:
        return points
    
    if method == 'random':
        indices = np.random.choice(len(points), target_count, replace=False)
        return points[indices]
    elif method == 'uniform':
        # Simple uniform sampling (could be improved with voxel grid)
        step = len(points) // target_count
        return points[::step][:target_count]
    else:
        raise ValueError(f"Unknown downsampling method: {method}")


# Export
__all__ = [
    'ColorMap',
    'VisualizationType',
    'VisualizationConfig',
    'PointCloudData',
    'MeshData',
    'VisualizationEngine',
    'WorldVisualizer',
    'create_color_gradient',
    'normalize_colors',
    'calculate_bounding_box',
    'downsample_point_cloud'
]
