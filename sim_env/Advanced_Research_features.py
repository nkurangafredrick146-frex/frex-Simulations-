#!/usr/bin/env python3
"""
Advanced Research Features
Scientific research tools, experimental controls, and advanced analysis capabilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize, signal
import sklearn
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import umap
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
import pickle
import hashlib
import warnings
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict, deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResearchFeatures")

@dataclass
class ExperimentParameters:
    """Parameters for scientific experiments"""
    name: str
    variables: Dict[str, Any]
    constants: Dict[str, Any]
    measurement_interval: float
    duration: float
    repetitions: int
    control_conditions: List[str]
    randomization: bool
    blinding: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentParameters':
        return cls(**data)

@dataclass
class ResearchDataPoint:
    """Single data point in a research experiment"""
    timestamp: float
    experiment_id: str
    trial_number: int
    conditions: Dict[str, Any]
    measurements: Dict[str, float]
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]

@dataclass
class StatisticalResults:
    """Results of statistical analysis"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_sizes: Tuple[int, int]
    assumptions_met: bool
    interpretation: str

class ResearchExperiment:
    """Base class for scientific experiments"""
    
    def __init__(self, experiment_id: str, parameters: ExperimentParameters):
        self.experiment_id = experiment_id
        self.parameters = parameters
        self.data: List[ResearchDataPoint] = []
        self.current_trial = 0
        self.start_time = time.time()
        self.completed = False
        self.quality_control_passed = True
        
        # Analysis results
        self.statistical_tests: Dict[str, StatisticalResults] = {}
        self.visualizations: Dict[str, Any] = {}
        self.insights: List[str] = []
        
        # Initialize experiment
        self.initialize_experiment()
        
        logger.info(f"Initialized experiment: {experiment_id}")
    
    def initialize_experiment(self):
        """Initialize experiment-specific settings"""
        # Set random seed for reproducibility if randomization is disabled
        if not self.parameters.randomization:
            np.random.seed(42)
    
    def run_experiment(self, simulation_app) -> List[ResearchDataPoint]:
        """Run the complete experiment"""
        logger.info(f"Starting experiment: {self.experiment_id}")
        
        for repetition in range(self.parameters.repetitions):
            if not self.quality_control_passed:
                logger.warning("Quality control failed, stopping experiment")
                break
                
            self.run_trial(repetition, simulation_app)
            
            # Check intermediate results
            if repetition % 5 == 0:
                self.perform_quality_control()
        
        self.completed = True
        self.analyze_results()
        
        logger.info(f"Completed experiment: {self.experiment_id}")
        return self.data
    
    def run_trial(self, trial_number: int, simulation_app):
        """Run a single trial of the experiment"""
        trial_start = time.time()
        
        # Set up experimental conditions
        conditions = self.determine_conditions(trial_number)
        
        # Configure simulation based on conditions
        self.configure_simulation(simulation_app, conditions)
        
        # Run simulation and collect data
        measurements = self.collect_data(simulation_app, trial_number)
        
        # Create data point
        data_point = ResearchDataPoint(
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            trial_number=trial_number,
            conditions=conditions,
            measurements=measurements,
            metadata=self.collect_metadata(simulation_app),
            quality_metrics=self.calculate_quality_metrics(measurements)
        )
        
        self.data.append(data_point)
        self.current_trial += 1
        
        trial_duration = time.time() - trial_start
        logger.debug(f"Trial {trial_number} completed in {trial_duration:.2f}s")
    
    def determine_conditions(self, trial_number: int) -> Dict[str, Any]:
        """Determine experimental conditions for this trial"""
        conditions = self.parameters.constants.copy()
        
        # Add variable conditions
        for var_name, var_values in self.parameters.variables.items():
            if self.parameters.randomization:
                # Random selection
                conditions[var_name] = np.random.choice(var_values)
            else:
                # Systematic variation
                idx = trial_number % len(var_values)
                conditions[var_name] = var_values[idx]
        
        return conditions
    
    def configure_simulation(self, simulation_app, conditions: Dict[str, Any]):
        """Configure simulation based on experimental conditions"""
        # This method should be overridden by specific experiment types
        for key, value in conditions.items():
            if hasattr(simulation_app, key):
                setattr(simulation_app, key, value)
            elif hasattr(simulation_app, 'config') and key in simulation_app.config:
                simulation_app.config[key] = value
    
    def collect_data(self, simulation_app, trial_number: int) -> Dict[str, float]:
        """Collect measurement data from simulation"""
        # This method should be overridden by specific experiment types
        measurements = {}
        
        # Example measurements (should be customized per experiment)
        if hasattr(simulation_app, 'particle_system'):
            particles = simulation_app.particle_system.particles
            if particles:
                positions = np.array([[p.position.x, p.position.y, p.position.z] for p in particles])
                velocities = np.array([[p.velocity.x, p.velocity.y, p.velocity.z] for p in particles])
                
                measurements['particle_count'] = len(particles)
                measurements['mean_velocity'] = np.mean(np.linalg.norm(velocities, axis=1))
                measurements['position_variance'] = np.var(positions)
                measurements['energy'] = self.calculate_energy(particles)
        
        measurements['simulation_time'] = simulation_app.simulation_time
        measurements['fps'] = simulation_app.fps
        
        return measurements
    
    def calculate_energy(self, particles) -> float:
        """Calculate total energy of particle system"""
        total_energy = 0.0
        for particle in particles:
            kinetic = 0.5 * particle.mass * glm.length2(particle.velocity)
            potential = particle.mass * 9.8 * particle.position.y  # gravitational potential
            total_energy += kinetic + potential
        return total_energy
    
    def collect_metadata(self, simulation_app) -> Dict[str, Any]:
        """Collect metadata about the simulation state"""
        return {
            'frame_count': simulation_app.frame_count,
            'simulation_time': simulation_app.simulation_time,
            'performance_metrics': {
                'fps': simulation_app.fps,
                'frame_time': simulation_app.average_frame_time
            },
            'system_info': {
                'timestamp': time.time(),
                'experiment_stage': self.current_trial
            }
        }
    
    def calculate_quality_metrics(self, measurements: Dict[str, float]) -> Dict[str, float]:
        """Calculate quality control metrics for data point"""
        metrics = {}
        
        # Check for reasonable values
        for key, value in measurements.items():
            if np.isfinite(value):
                metrics[f'{key}_valid'] = 1.0
            else:
                metrics[f'{key}_valid'] = 0.0
        
        # Overall data quality score
        valid_metrics = [m for m in metrics.values() if m > 0]
        metrics['overall_quality'] = np.mean(valid_metrics) if valid_metrics else 0.0
        
        return metrics
    
    def perform_quality_control(self):
        """Perform quality control on collected data"""
        if len(self.data) < 2:
            return
        
        recent_data = self.data[-10:]  # Check last 10 data points
        
        quality_scores = [dp.quality_metrics['overall_quality'] for dp in recent_data]
        avg_quality = np.mean(quality_scores)
        
        if avg_quality < 0.8:
            self.quality_control_passed = False
            logger.warning(f"Quality control failed: average quality = {avg_quality:.3f}")
    
    def analyze_results(self):
        """Perform comprehensive analysis of experiment results"""
        logger.info("Analyzing experiment results...")
        
        # Convert to pandas DataFrame for analysis
        df = self.to_dataframe()
        
        # Statistical tests
        self.perform_statistical_tests(df)
        
        # Dimensionality reduction
        self.perform_dimensionality_reduction(df)
        
        # Clustering analysis
        self.perform_clustering_analysis(df)
        
        # Time series analysis
        self.perform_time_series_analysis(df)
        
        # Generate insights
        self.generate_insights()
        
        logger.info("Analysis completed")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert experiment data to pandas DataFrame"""
        rows = []
        for data_point in self.data:
            row = {
                'timestamp': data_point.timestamp,
                'trial_number': data_point.trial_number,
                **data_point.conditions,
                **data_point.measurements,
                **{f'meta_{k}': v for k, v in data_point.metadata.items()},
                **{f'quality_{k}': v for k, v in data_point.quality_metrics.items()}
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def perform_statistical_tests(self, df: pd.DataFrame):
        """Perform various statistical tests on the data"""
        if len(df) < 2:
            return
        
        # Test for normality
        for measurement in self.get_measurement_columns(df):
            if len(df[measurement]) > 3:
                stat, p_value = stats.normaltest(df[measurement])
                self.statistical_tests[f'normality_{measurement}'] = StatisticalResults(
                    test_name="D'Agostino's normality test",
                    statistic=stat,
                    p_value=p_value,
                    effect_size=0.0,
                    confidence_interval=(0, 0),
                    sample_sizes=(len(df[measurement]), 0),
                    assumptions_met=p_value > 0.05,
                    interpretation="Data is normally distributed" if p_value > 0.05 else "Data is not normally distributed"
                )
        
        # Correlation analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            correlation_matrix = df[numeric_columns].corr()
            self.visualizations['correlation_matrix'] = correlation_matrix
            
            # Find significant correlations
            for i, col1 in enumerate(numeric_columns):
                for j, col2 in enumerate(numeric_columns):
                    if i < j:  # Avoid duplicates
                        corr, p_value = stats.pearsonr(df[col1], df[col2])
                        if abs(corr) > 0.5 and p_value < 0.05:
                            test_name = f"correlation_{col1}_{col2}"
                            self.statistical_tests[test_name] = StatisticalResults(
                                test_name="Pearson correlation",
                                statistic=corr,
                                p_value=p_value,
                                effect_size=corr,
                                confidence_interval=stats.pearsonr_ci(corr, len(df)),
                                sample_sizes=(len(df), len(df)),
                                assumptions_met=True,
                                interpretation=f"Significant correlation between {col1} and {col2}"
                            )
    
    def perform_dimensionality_reduction(self, df: pd.DataFrame):
        """Perform dimensionality reduction for visualization"""
        numeric_data = df.select_dtypes(include=[np.number])
        
        if len(numeric_data) < 2:
            return
        
        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.std() > 0]
        
        if len(numeric_data.columns) < 2:
            return
        
        # Standardize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        self.visualizations['pca'] = {
            'components': pca_result,
            'explained_variance': pca.explained_variance_ratio_,
            'feature_importance': pca.components_
        }
        
        # t-SNE if we have enough samples
        if len(scaled_data) > 10:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(scaled_data)
            self.visualizations['tsne'] = tsne_result
        
        # UMAP if we have enough samples
        if len(scaled_data) > 15:
            reducer = umap.UMAP(random_state=42)
            umap_result = reducer.fit_transform(scaled_data)
            self.visualizations['umap'] = umap_result
    
    def perform_clustering_analysis(self, df: pd.DataFrame):
        """Perform clustering analysis on the data"""
        numeric_data = df.select_dtypes(include=[np.number])
        
        if len(numeric_data) < 5:
            return
        
        # Remove columns with zero variance
        numeric_data = numeric_data.loc[:, numeric_data.std() > 0]
        
        if len(numeric_data.columns) < 2:
            return
        
        # Standardize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=min(5, len(scaled_data)), random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        self.visualizations['kmeans_clusters'] = clusters
        self.visualizations['kmeans_centers'] = kmeans.cluster_centers_
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(scaled_data, clusters)
            self.statistical_tests['clustering_quality'] = StatisticalResults(
                test_name="Silhouette score",
                statistic=silhouette_avg,
                p_value=0.0,
                effect_size=silhouette_avg,
                confidence_interval=(0, 0),
                sample_sizes=(len(clusters), 0),
                assumptions_met=silhouette_avg > 0.5,
                interpretation="Good clustering structure" if silhouette_avg > 0.5 else "Weak clustering structure"
            )
    
    def perform_time_series_analysis(self, df: pd.DataFrame):
        """Perform time series analysis on sequential data"""
        if len(df) < 10:
            return
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        # Analyze each measurement over time
        for measurement in self.get_measurement_columns(df):
            time_series = df_sorted[measurement].values
            
            # Stationarity test (Augmented Dickey-Fuller)
            try:
                adf_result = stats.adfuller(time_series)
                self.statistical_tests[f'stationarity_{measurement}'] = StatisticalResults(
                    test_name="Augmented Dickey-Fuller",
                    statistic=adf_result[0],
                    p_value=adf_result[1],
                    effect_size=0.0,
                    confidence_interval=adf_result[4],
                    sample_sizes=(len(time_series), 0),
                    assumptions_met=adf_result[1] < 0.05,
                    interpretation="Stationary time series" if adf_result[1] < 0.05 else "Non-stationary time series"
                )
            except:
                pass
            
            # Autocorrelation
            try:
                autocorr = signal.correlate(time_series, time_series, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                self.visualizations[f'autocorrelation_{measurement}'] = autocorr
            except:
                pass
    
    def get_measurement_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that contain measurement data"""
        measurement_cols = []
        for col in df.columns:
            if col not in ['timestamp', 'trial_number'] and not col.startswith(('meta_', 'quality_')):
                if col not in self.parameters.variables and col not in self.parameters.constants:
                    measurement_cols.append(col)
        return measurement_cols
    
    def generate_insights(self):
        """Generate human-readable insights from analysis"""
        self.insights = []
        
        # Add insights based on statistical tests
        for test_name, result in self.statistical_tests.items():
            if result.p_value < 0.05:
                if 'correlation' in test_name:
                    self.insights.append(f"Found significant correlation: {result.interpretation}")
                elif 'stationarity' in test_name:
                    self.insights.append(f"Time series analysis: {result.interpretation}")
        
        # Add insights based on clustering
        if 'clustering_quality' in self.statistical_tests:
            cluster_result = self.statistical_tests['clustering_quality']
            if cluster_result.effect_size > 0.5:
                self.insights.append("Clear clustering patterns detected in the data")
        
        # Add general insights
        if len(self.data) > 0:
            avg_quality = np.mean([dp.quality_metrics['overall_quality'] for dp in self.data])
            if avg_quality > 0.9:
                self.insights.append("High data quality throughout experiment")
            elif avg_quality < 0.7:
                self.insights.append("Data quality concerns detected")
        
        if not self.insights:
            self.insights.append("No strong patterns detected in the data")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report"""
        return {
            'experiment_id': self.experiment_id,
            'parameters': self.parameters.to_dict(),
            'summary': {
                'total_trials': len(self.data),
                'completion_status': 'completed' if self.completed else 'incomplete',
                'quality_control_passed': self.quality_control_passed,
                'total_duration': time.time() - self.start_time
            },
            'statistical_results': {k: asdict(v) for k, v in self.statistical_tests.items()},
            'insights': self.insights,
            'recommendations': self.generate_recommendations(),
            'visualization_data': self.prepare_visualization_data()
        }
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations for future experiments"""
        recommendations = []
        
        # Based on statistical power
        if len(self.data) < 30:
            recommendations.append("Consider increasing sample size for better statistical power")
        
        # Based on effect sizes
        strong_effects = [r for r in self.statistical_tests.values() 
                         if hasattr(r, 'effect_size') and abs(r.effect_size) > 0.8]
        if not strong_effects:
            recommendations.append("No strong effects detected; consider varying parameters more widely")
        
        # Based on data quality
        quality_scores = [dp.quality_metrics['overall_quality'] for dp in self.data]
        if np.mean(quality_scores) < 0.8:
            recommendations.append("Improve data collection methods to increase data quality")
        
        return recommendations
    
    def prepare_visualization_data(self) -> Dict[str, Any]:
        """Prepare data for visualization"""
        df = self.to_dataframe()
        
        visualization_data = {
            'scatter_data': {},
            'time_series_data': {},
            'distribution_data': {},
            'correlation_data': self.visualizations.get('correlation_matrix', None)
        }
        
        # Scatter plots
        measurement_cols = self.get_measurement_columns(df)
        if len(measurement_cols) >= 2:
            visualization_data['scatter_data'] = {
                'x': df[measurement_cols[0]].tolist(),
                'y': df[measurement_cols[1]].tolist(),
                'labels': df['trial_number'].tolist()
            }
        
        # Time series
        for measurement in measurement_cols[:3]:  # First 3 measurements
            visualization_data['time_series_data'][measurement] = {
                'time': df['timestamp'].tolist(),
                'values': df[measurement].tolist()
            }
        
        # Distributions
        for measurement in measurement_cols[:4]:  # First 4 measurements
            visualization_data['distribution_data'][measurement] = df[measurement].tolist()
        
        return visualization_data
    
    def save_experiment(self, filename: str):
        """Save experiment data and results to file"""
        experiment_data = {
            'experiment_id': self.experiment_id,
            'parameters': self.parameters.to_dict(),
            'data': [asdict(dp) for dp in self.data],
            'statistical_tests': {k: asdict(v) for k, v in self.statistical_tests.items()},
            'visualizations': self.visualizations,
            'insights': self.insights,
            'metadata': {
                'save_timestamp': time.time(),
                'total_trials': len(self.data),
                'completion_status': self.completed
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(experiment_data, f)
        
        logger.info(f"Experiment saved to {filename}")
    
    @classmethod
    def load_experiment(cls, filename: str) -> 'ResearchExperiment':
        """Load experiment from file"""
        with open(filename, 'rb') as f:
            experiment_data = pickle.load(f)
        
        parameters = ExperimentParameters.from_dict(experiment_data['parameters'])
        experiment = cls(experiment_data['experiment_id'], parameters)
        
        # Restore data
        experiment.data = [ResearchDataPoint(**dp) for dp in experiment_data['data']]
        experiment.statistical_tests = {k: StatisticalResults(**v) 
                                      for k, v in experiment_data['statistical_tests'].items()}
        experiment.visualizations = experiment_data['visualizations']
        experiment.insights = experiment_data['insights']
        experiment.completed = experiment_data['metadata']['completion_status']
        
        logger.info(f"Experiment loaded from {filename}")
        return experiment

class PhysicsParameterSweep(ResearchExperiment):
    """Experiment for sweeping physics parameters"""
    
    def __init__(self, experiment_id: str, parameters: Dict[str, Any]):
        exp_params = ExperimentParameters(
            name=experiment_id,
            variables=parameters.get('variables', {}),
            constants=parameters.get('constants', {}),
            measurement_interval=parameters.get('measurement_interval', 1.0),
            duration=parameters.get('duration', 60.0),
            repetitions=parameters.get('repetitions', 10),
            control_conditions=parameters.get('control_conditions', []),
            randomization=parameters.get('randomization', False),
            blinding=parameters.get('blinding', False)
        )
        super().__init__(experiment_id, exp_params)
    
    def configure_simulation(self, simulation_app, conditions: Dict[str, Any]):
        """Configure physics parameters in simulation"""
        # Set gravity
        if 'gravity' in conditions:
            if hasattr(simulation_app, 'physics_module'):
                simulation_app.physics_module.gravity = glm.vec3(0, conditions['gravity'], 0)
        
        # Set particle properties
        if 'particle_mass' in conditions and hasattr(simulation_app, 'particle_system'):
            for particle in simulation_app.particle_system.particles:
                particle.mass = conditions['particle_mass']
        
        # Set simulation parameters
        for key, value in conditions.items():
            if hasattr(simulation_app, key):
                setattr(simulation_app, key, value)

class EmergentBehaviorStudy(ResearchExperiment):
    """Study of emergent behaviors in complex systems"""
    
    def __init__(self, experiment_id: str, parameters: Dict[str, Any]):
        exp_params = ExperimentParameters(
            name=experiment_id,
            variables=parameters.get('variables', {}),
            constants=parameters.get('constants', {}),
            measurement_interval=parameters.get('measurement_interval', 0.5),
            duration=parameters.get('duration', 120.0),
            repetitions=parameters.get('repetitions', 20),
            control_conditions=parameters.get('control_conditions', []),
            randomization=parameters.get('randomization', True),
            blinding=parameters.get('blinding', False)
        )
        super().__init__(experiment_id, exp_params)
        self.behavior_metrics = []
    
    def collect_data(self, simulation_app, trial_number: int) -> Dict[str, float]:
        """Collect data about emergent behaviors"""
        measurements = super().collect_data(simulation_app, trial_number)
        
        # Additional behavior-specific measurements
        if hasattr(simulation_app, 'particle_system'):
            particles = simulation_app.particle_system.particles
            
            # Calculate flocking/swarming metrics
            if len(particles) > 1:
                positions = np.array([[p.position.x, p.position.y, p.position.z] for p in particles])
                velocities = np.array([[p.velocity.x, p.velocity.y, p.velocity.z] for p in particles])
                
                # Alignment (mean velocity direction similarity)
                mean_velocity = np.mean(velocities, axis=0)
                velocity_norms = np.linalg.norm(velocities, axis=1)
                normalized_velocities = velocities / velocity_norms[:, np.newaxis]
                alignment = np.mean([np.dot(v, mean_velocity/np.linalg.norm(mean_velocity)) 
                                   for v in normalized_velocities])
                
                # Cohesion (average distance to center)
                center = np.mean(positions, axis=0)
                distances = np.linalg.norm(positions - center, axis=1)
                cohesion = 1.0 / (1.0 + np.mean(distances))
                
                # Separation (average minimum distance between particles)
                from scipy.spatial.distance import pdist
                if len(positions) > 1:
                    pairwise_distances = pdist(positions)
                    separation = np.mean(pairwise_distances)
                else:
                    separation = 0.0
                
                measurements['alignment'] = alignment
                measurements['cohesion'] = cohesion
                measurements['separation'] = separation
                measurements['swarm_stability'] = alignment * cohesion / (1.0 + separation)
        
        return measurements

class ResearchManager:
    """Manager for research experiments and studies"""
    
    def __init__(self, simulation_app):
        self.simulation_app = simulation_app
        self.experiments: Dict[str, ResearchExperiment] = {}
        self.active_experiment: Optional[ResearchExperiment] = None
        self.experiment_queue = deque()
        self.results_database = ResultsDatabase()
        
        # Visualization
        self.live_plotting_enabled = True
        self.current_visualization = None
        
        logger.info("Research Manager initialized")
    
    def create_experiment(self, experiment_type: str, experiment_id: str, 
                         parameters: Dict[str, Any]) -> ResearchExperiment:
        """Create a new research experiment"""
        if experiment_type == "parameter_sweep":
            experiment = PhysicsParameterSweep(experiment_id, parameters)
        elif experiment_type == "emergent_behavior":
            experiment = EmergentBehaviorStudy(experiment_id, parameters)
        else:
            experiment = ResearchExperiment(experiment_id, 
                                          ExperimentParameters(**parameters))
        
        self.experiments[experiment_id] = experiment
        logger.info(f"Created experiment: {experiment_id} of type {experiment_type}")
        
        return experiment
    
    def queue_experiment(self, experiment: ResearchExperiment):
        """Add experiment to execution queue"""
        self.experiment_queue.append(experiment)
        logger.info(f"Queued experiment: {experiment.experiment_id}")
    
    def run_next_experiment(self) -> Optional[ResearchExperiment]:
        """Run the next experiment in the queue"""
        if not self.experiment_queue:
            return None
        
        self.active_experiment = self.experiment_queue.popleft()
        results = self.active_experiment.run_experiment(self.simulation_app)
        
        # Store results
        self.results_database.store_experiment(self.active_experiment)
        
        # Generate report
        report = self.active_experiment.generate_report()
        self.results_database.store_report(self.active_experiment.experiment_id, report)
        
        logger.info(f"Completed experiment: {self.active_experiment.experiment_id}")
        
        completed_experiment = self.active_experiment
        self.active_experiment = None
        
        return completed_experiment
    
    def run_all_queued_experiments(self):
        """Run all experiments in the queue"""
        while self.experiment_queue:
            self.run_next_experiment()
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of an experiment"""
        if experiment_id in self.experiments:
            experiment = self.experiments[experiment_id]
            return {
                'experiment_id': experiment_id,
                'status': 'completed' if experiment.completed else 'running',
                'progress': f"{len(experiment.data)}/{experiment.parameters.repetitions}",
                'quality_control': experiment.quality_control_passed
            }
        return {'error': 'Experiment not found'}
    
    def generate_comparative_analysis(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Generate comparative analysis across multiple experiments"""
        if not experiment_ids:
            return {}
        
        # Load experiment data
        experiments_data = []
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                experiments_data.append(self.experiments[exp_id].to_dataframe())
        
        if not experiments_data:
            return {}
        
        # Combine data with experiment labels
        combined_data = pd.concat(experiments_data, keys=experiment_ids, names=['experiment_id'])
        
        # Comparative statistics
        comparative_stats = {}
        measurement_cols = [col for col in combined_data.columns 
                           if col not in ['timestamp', 'trial_number'] and 
                           not col.startswith(('meta_', 'quality_'))]
        
        for measurement in measurement_cols:
            group_means = combined_data.groupby('experiment_id')[measurement].mean()
            group_stds = combined_data.groupby('experiment_id')[measurement].std()
            
            comparative_stats[measurement] = {
                'means': group_means.to_dict(),
                'standard_deviations': group_stds.to_dict(),
                'anova_result': self.perform_anova(combined_data, 'experiment_id', measurement)
            }
        
        return {
            'comparative_statistics': comparative_stats,
            'experiment_count': len(experiment_ids),
            'total_data_points': len(combined_data)
        }
    
    def perform_anova(self, data: pd.DataFrame, group_col: str, measurement_col: str) -> Dict[str, Any]:
        """Perform ANOVA test between groups"""
        try:
            groups = [group[measurement_col].values for name, group in data.groupby(group_col)]
            f_stat, p_value = stats.f_oneway(*groups)
            
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': 'Significant differences between groups' if p_value < 0.05 
                                else 'No significant differences between groups'
            }
        except:
            return {'error': 'ANOVA test failed'}
    
    def export_research_data(self, format: str = 'csv') -> str:
        """Export all research data in specified format"""
        if format == 'csv':
            return self.export_to_csv()
        elif format == 'json':
            return self.export_to_json()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_to_csv(self) -> str:
        """Export research data to CSV format"""
        all_data = []
        for exp_id, experiment in self.experiments.items():
            df = experiment.to_dataframe()
            df['experiment_id'] = exp_id
            all_data.append(df)
        
        if not all_data:
            return ""
        
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df.to_csv(index=False)
    
    def export_to_json(self) -> str:
        """Export research data to JSON format"""
        export_data = {}
        for exp_id, experiment in self.experiments.items():
            export_data[exp_id] = {
                'parameters': experiment.parameters.to_dict(),
                'data': [asdict(dp) for dp in experiment.data],
                'statistical_results': {k: asdict(v) for k, v in experiment.statistical_tests.items()},
                'insights': experiment.insights
            }
        
        return json.dumps(export_data, indent=2)
    
    def get_research_dashboard_data(self) -> Dict[str, Any]:
        """Get data for research dashboard"""
        dashboard_data = {
            'total_experiments': len(self.experiments),
            'completed_experiments': len([e for e in self.experiments.values() if e.completed]),
            'queued_experiments': len(self.experiment_queue),
            'active_experiment': self.active_experiment.experiment_id if self.active_experiment else None,
            'recent_insights': [],
            'data_quality_metrics': self.calculate_overall_data_quality()
        }
        
        # Collect recent insights from completed experiments
        for experiment in self.experiments.values():
            if experiment.completed and experiment.insights:
                dashboard_data['recent_insights'].extend(experiment.insights[-3:])  # Last 3 insights
        
        return dashboard_data
    
    def calculate_overall_data_quality(self) -> Dict[str, float]:
        """Calculate overall data quality metrics"""
        if not self.experiments:
            return {}
        
        all_quality_scores = []
        for experiment in self.experiments.values():
            for data_point in experiment.data:
                all_quality_scores.append(data_point.quality_metrics['overall_quality'])
        
        if not all_quality_scores:
            return {}
        
        return {
            'mean_quality': np.mean(all_quality_scores),
            'std_quality': np.std(all_quality_scores),
            'min_quality': np.min(all_quality_scores),
            'max_quality': np.max(all_quality_scores),
            'completeness': len(all_quality_scores) / sum([e.parameters.repetitions for e in self.experiments.values()])
        }

class ResultsDatabase:
    """Database for storing and querying research results"""
    
    def __init__(self, db_path: str = "research_results.db"):
        self.db_path = db_path
        self.experiments: Dict[str, ResearchExperiment] = {}
        self.reports: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with sample data structure
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize the results database"""
        # In a real implementation, this would connect to a proper database
        # For now, we'll use in-memory storage
        pass
    
    def store_experiment(self, experiment: ResearchExperiment):
        """Store an experiment in the database"""
        self.experiments[experiment.experiment_id] = experiment
    
    def store_report(self, experiment_id: str, report: Dict[str, Any]):
        """Store an experiment report in the database"""
        self.reports[experiment_id] = report
    
    def query_experiments(self, filters: Dict[str, Any] = None) -> List[ResearchExperiment]:
        """Query experiments based on filters"""
        if not filters:
            return list(self.experiments.values())
        
        filtered_experiments = []
        for experiment in self.experiments.values():
            matches = True
            
            for key, value in filters.items():
                if hasattr(experiment, key):
                    attr_value = getattr(experiment, key)
                    if attr_value != value:
                        matches = False
                        break
            
            if matches:
                filtered_experiments.append(experiment)
        
        return filtered_experiments
    
    def get_statistical_summary(self) -> Dict[str, Any]:
        """Get statistical summary of all stored experiments"""
        if not self.experiments:
            return {}
        
        summary = {
            'total_experiments': len(self.experiments),
            'total_data_points': sum([len(exp.data) for exp in self.experiments.values()]),
            'experiment_types': defaultdict(int),
            'significant_findings': 0,
            'data_quality_stats': {}
        }
        
        # Count experiment types and significant findings
        for experiment in self.experiments.values():
            summary['experiment_types'][experiment.__class__.__name__] += 1
            
            # Count significant statistical tests
            significant_tests = [t for t in experiment.statistical_tests.values() 
                               if hasattr(t, 'p_value') and t.p_value < 0.05]
            summary['significant_findings'] += len(significant_tests)
        
        # Calculate data quality statistics
        all_quality_scores = []
        for experiment in self.experiments.values():
            for data_point in experiment.data:
                all_quality_scores.append(data_point.quality_metrics['overall_quality'])
        
        if all_quality_scores:
            summary['data_quality_stats'] = {
                'mean': np.mean(all_quality_scores),
                'std': np.std(all_quality_scores),
                'min': np.min(all_quality_scores),
                'max': np.max(all_quality_scores)
            }
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Create a mock simulation app for testing
    class MockSimulationApp:
        def __init__(self):
            self.frame_count = 0
            self.simulation_time = 0.0
            self.fps = 60.0
            self.average_frame_time = 0.016
            self.config = {}
            
            class MockParticle:
                def __init__(self):
                    self.position = glm.vec3(np.random.random(3))
                    self.velocity = glm.vec3(np.random.random(3) - 0.5)
                    self.mass = 1.0
            
            self.particle_system = type('MockParticleSystem', (), {
                'particles': [MockParticle() for _ in range(100)]
            })()
    
    # Test the research features
    simulation_app = MockSimulationApp()
    research_manager = ResearchManager(simulation_app)
    
    # Create a parameter sweep experiment
    parameters = {
        'variables': {
            'gravity': [-5.0, -9.8, -15.0],
            'particle_mass': [0.5, 1.0, 2.0]
        },
        'constants': {'simulation_speed': 1.0},
        'repetitions': 5,
        'duration': 10.0,
        'randomization': False
    }
    
    experiment = research_manager.create_experiment(
        "parameter_sweep", 
        "gravity_mass_study", 
        parameters
    )
    
    # Run the experiment
    research_manager.queue_experiment(experiment)
    completed_experiment = research_manager.run_next_experiment()
    
    if completed_experiment:
        # Generate report
        report = completed_experiment.generate_report()
        print("Experiment completed successfully!")
        print(f"Insights: {report['insights']}")
        print(f"Statistical tests performed: {len(report['statistical_results'])}")
    
    # Get research dashboard data
    dashboard_data = research_manager.get_research_dashboard_data()
    print(f"\nResearch Dashboard:")
    print(f"Total experiments: {dashboard_data['total_experiments']}")
    print(f"Data quality: {dashboard_data['data_quality_metrics']}")