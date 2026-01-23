#!/usr/bin/env python3
"""
Astrophysics & Cosmic Simulations Module
Advanced simulations of cosmic phenomena, general relativity, and large-scale structure formation
"""

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import scipy.constants as const
import numba
from numba import jit, cuda, prange
import torch
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import *
import glm
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Callable
import math
import random
from enum import Enum
import time
from collections import deque
import h5py
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

class CosmicScale(Enum):
    """Cosmic scale enumerations"""
    QUANTUM = "quantum"
    PLANETARY = "planetary"
    STELLAR = "stellar"
    GALACTIC = "galactic"
    CLUSTER = "cluster"
    COSMIC = "cosmic"
    MULTIVERSAL = "multiversal"

class PhysicsModel(Enum):
    """Physics models for simulations"""
    NEWTONIAN = "newtonian"
    SPECIAL_RELATIVITY = "special_relativity"
    GENERAL_RELATIVITY = "general_relativity"
    QUANTUM_GRAVITY = "quantum_gravity"
    COSMOLOGICAL = "cosmological"

class CosmicPhenomena(Enum):
    """Cosmic phenomena to simulate"""
    BLACK_HOLE = "black_hole"
    NEUTRON_STAR = "neutron_star"
    SUPERNOVA = "supernova"
    GALAXY_FORMATION = "galaxy_formation"
    COSMIC_INFLATION = "cosmic_inflation"
    DARK_MATER_HALO = "dark_matter_halo"
    GRAVITATIONAL_LENSING = "gravitational_lensing"
    COSMIC_STRINGS = "cosmic_strings"

@dataclass
class CosmologicalParameters:
    """Cosmological parameters from Planck satellite"""
    H0: float = 67.4  # Hubble constant [km/s/Mpc]
    Omega_m: float = 0.315  # Matter density parameter
    Omega_b: float = 0.049  # Baryon density parameter
    Omega_lambda: float = 0.685  # Dark energy density parameter
    Omega_r: float = 9.24e-5  # Radiation density parameter
    sigma8: float = 0.811  # Matter fluctuation amplitude
    n_s: float = 0.965  # Scalar spectral index
    T_CMB: float = 2.7255  # CMB temperature [K]

@dataclass
class Particle:
    """Astrophysical particle for N-body simulations"""
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    particle_type: str  # 'dark_matter', 'gas', 'star', 'black_hole'
    formation_time: float = 0.0
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
        self.acceleration = np.zeros(3)
        self.potential = 0.0
        self.temperature = 0.0
        self.metallicity = 0.0

@dataclass
class BlackHole:
    """Kerr black hole with general relativistic properties"""
    position: np.ndarray
    mass: float  # Solar masses
    spin: float  # Dimensionless spin parameter (0 <= a <= 1)
    charge: float = 0.0  # Usually negligible
    accretion_rate: float = 0.0  # Solar masses per year
    
    @property
    def schwarzschild_radius(self) -> float:
        """Schwarzschild radius in meters"""
        return 2 * const.G * self.mass * const.M_sun / const.c**2
    
    @property
    def event_horizon_radius(self) -> float:
        """Event horizon radius for Kerr black hole"""
        r_s = self.schwarzschild_radius
        return 0.5 * (r_s + math.sqrt(r_s**2 - (self.spin * r_s)**2))
    
    @property
    def ergosphere_radius(self) -> float:
        """Ergosphere radius"""
        r_s = self.schwarzschild_radius
        return 0.5 * (r_s + math.sqrt(r_s**2 - (self.spin * r_s * math.cos(0))**2))

class AstrophysicsSimulator:
    """Advanced astrophysics and cosmic phenomena simulator"""
    
    def __init__(self, scale: CosmicScale, physics_model: PhysicsModel):
        self.scale = scale
        self.physics_model = physics_model
        self.cosmological_params = CosmologicalParameters()
        
        # Simulation parameters
        self.box_size = self._get_default_box_size()
        self.resolution = 256
        self.current_time = 0.0
        self.time_step = self._get_default_time_step()
        self.redshift = 0.0
        
        # Physical constants
        self.G = const.G  # Gravitational constant
        self.c = const.c  # Speed of light
        self.h = const.h  # Planck constant
        self.k_B = const.k  # Boltzmann constant
        self.M_sun = const.M_sun  # Solar mass
        
        # Simulation data
        self.particles: List[Particle] = []
        self.black_holes: List[BlackHole] = []
        self.density_field: Optional[np.ndarray] = None
        self.velocity_field: Optional[np.ndarray] = None
        self.temperature_field: Optional[np.ndarray] = None
        self.magnetic_field: Optional[np.ndarray] = None
        
        # Numerical methods
        self.gravity_solver = "tree"
        self.hydro_solver = "sph"
        self.radiative_transfer = "ray_tracing"
        
        # Performance optimization
        self.softening_length = self.box_size / self.resolution / 20
        self.theta_tree = 0.7  # Barnes-Hut opening angle
        self.use_gpu = torch.cuda.is_available()
        self.use_numba = True
        
        # Visualization
        self.visualization_scale = 1.0
        self.color_maps = {}
        self._initialize_color_maps()
        
        # Statistics
        self.statistics = {
            'total_mass': 0.0,
            'kinetic_energy': 0.0,
            'potential_energy': 0.0,
            'virial_ratio': 0.0,
            'star_formation_rate': 0.0
        }
        
        print(f"Astrophysics Simulator initialized: {scale.value} scale, {physics_model.value} physics")

    def _get_default_box_size(self) -> float:
        """Get default simulation box size based on scale"""
        sizes = {
            CosmicScale.PLANETARY: 1e12,  # 1 AU in meters
            CosmicScale.STELLAR: 1e16,    # ~1 light-year
            CosmicScale.GALACTIC: 1e21,   # ~100 kpc
            CosmicScale.CLUSTER: 1e24,    # ~30 Mpc
            CosmicScale.COSMIC: 1e26,     # ~3 Gpc
        }
        return sizes.get(self.scale, 1e21)

    def _get_default_time_step(self) -> float:
        """Get default time step based on scale"""
        time_steps = {
            CosmicScale.PLANETARY: 86400,      # 1 day
            CosmicScale.STELLAR: 3.154e7,      # 1 year
            CosmicScale.GALACTIC: 3.154e13,    # 1 million years
            CosmicScale.CLUSTER: 3.154e16,     # 1 billion years
            CosmicScale.COSMIC: 3.154e17,      # 10 billion years
        }
        return time_steps.get(self.scale, 3.154e13)

    def _initialize_color_maps(self):
        """Initialize astrophysical color maps"""
        # Temperature color map (for stars, gas)
        self.color_maps['temperature'] = {
            'range': (10, 1e8),  # Kelvin
            'colors': ['blue', 'white', 'yellow', 'red']
        }
        
        # Density color map
        self.color_maps['density'] = {
            'range': (1e-30, 1e-18),  # kg/m³
            'colors': ['black', 'purple', 'blue', 'green', 'yellow', 'red', 'white']
        }
        
        # Velocity color map
        self.color_maps['velocity'] = {
            'range': (0, 3e5),  # km/s
            'colors': ['blue', 'cyan', 'green', 'yellow', 'red']
        }

    def set_initial_conditions(self, condition_type: str, **kwargs):
        """Set initial conditions for simulation"""
        if condition_type == "cosmological":
            self._set_cosmological_initial_conditions(**kwargs)
        elif condition_type == "galaxy_collision":
            self._set_galaxy_collision_initial_conditions(**kwargs)
        elif condition_type == "cluster_formation":
            self._set_cluster_formation_initial_conditions(**kwargs)
        elif condition_type == "black_hole_binary":
            self._set_black_hole_binary_initial_conditions(**kwargs)
        else:
            self._set_uniform_initial_conditions(**kwargs)

    def _set_cosmological_initial_conditions(self, num_particles: int = 1000000, power_spectrum_index: float = -2.0):
        """Set cosmological initial conditions with Gaussian random field"""
        print("Generating cosmological initial conditions...")
        
        # Generate Gaussian random field for density fluctuations
        k_grid = np.fft.fftfreq(self.resolution) * 2 * np.pi / (self.box_size / self.resolution)
        k_magnitude = np.sqrt(sum(ki**2 for ki in np.meshgrid(*[k_grid]*3, indexing='ij')))
        
        # Power spectrum P(k) ~ k^n
        power_spectrum = np.where(k_magnitude > 0, k_magnitude**power_spectrum_index, 0)
        power_spectrum[0, 0, 0] = 0  # Remove zero mode
        
        # Generate Gaussian random field
        random_phase = np.exp(2j * np.pi * np.random.random(power_spectrum.shape))
        density_fluctuations = np.fft.ifftn(np.sqrt(power_spectrum) * random_phase).real
        density_fluctuations = (density_fluctuations - density_fluctuations.mean()) / density_fluctuations.std()
        
        # Set density field
        self.density_field = np.exp(density_fluctuations)  # Lognormal distribution
        
        # Generate particles using Poisson sampling
        self._generate_particles_from_density_field(num_particles)
        
        # Set initial velocities using Zel'dovich approximation
        self._set_zeldoovich_velocities()
        
        print(f"Generated {len(self.particles)} particles with cosmological initial conditions")

    def _generate_particles_from_density_field(self, num_particles: int):
        """Generate particles from density field using Poisson sampling"""
        if self.density_field is None:
            raise ValueError("Density field not initialized")
        
        # Normalize density to probability distribution
        density_normalized = self.density_field / self.density_field.sum()
        
        # Sample particle positions
        flat_indices = np.random.choice(self.resolution**3, size=num_particles, p=density_normalized.flatten())
        positions_flat = np.unravel_index(flat_indices, self.density_field.shape)
        
        # Convert to physical coordinates
        cell_size = self.box_size / self.resolution
        for i in range(num_particles):
            pos = np.array([positions_flat[0][i], positions_flat[1][i], positions_flat[2][i]])
            pos_physical = (pos + np.random.random(3)) * cell_size - self.box_size / 2
            
            # Assign mass based on local density
            local_density = self.density_field[tuple(pos.astype(int))]
            mass = local_density * (self.box_size**3) / num_particles
            
            particle = Particle(
                position=pos_physical,
                velocity=np.zeros(3),
                mass=mass,
                particle_type='dark_matter'
            )
            self.particles.append(particle)

    def _set_zeldoovich_velocities(self):
        """Set initial velocities using Zel'dovich approximation"""
        if self.density_field is None:
            return
        
        # Compute displacement potential from density field
        delta_k = np.fft.fftn(self.density_field - 1.0)  # Density contrast
        k_grid = np.fft.fftfreq(self.resolution) * 2 * np.pi / (self.box_size / self.resolution)
        
        k_mesh = np.meshgrid(*[k_grid]*3, indexing='ij')
        k_squared = sum(ki**2 for ki in k_mesh)
        k_squared[0, 0, 0] = 1  # Avoid division by zero
        
        # Zel'dovich displacement in Fourier space
        displacement_potential_k = -delta_k / k_squared
        
        # Velocity is gradient of displacement potential
        velocity_field_k = []
        for i in range(3):
            velocity_component_k = 1j * k_mesh[i] * displacement_potential_k
            velocity_field_k.append(np.fft.ifftn(velocity_component_k).real)
        
        self.velocity_field = np.stack(velocity_field_k, axis=-1)
        
        # Assign velocities to particles
        cell_size = self.box_size / self.resolution
        for particle in self.particles:
            grid_pos = ((particle.position + self.box_size / 2) / cell_size).astype(int)
            grid_pos = np.clip(grid_pos, 0, self.resolution - 1)
            particle.velocity = self.velocity_field[tuple(grid_pos)]

    def _set_galaxy_collision_initial_conditions(self, galaxy1_mass: float = 1e12, galaxy2_mass: float = 1e11,
                                               separation: float = 100e3 * const.parsec, relative_velocity: float = 200e3):
        """Set initial conditions for galaxy collision (Antennae galaxies)"""
        # Galaxy 1 (primary)
        self._generate_galaxy(galaxy1_mass, np.array([-separation/2, 0, 0]), np.array([relative_velocity/2, 0, 0]))
        
        # Galaxy 2 (secondary)
        self._generate_galaxy(galaxy2_mass, np.array([separation/2, 0, 0]), np.array([-relative_velocity/2, 0, 0]))
        
        print(f"Generated galaxy collision scenario with {len(self.particles)} particles")

    def _generate_galaxy(self, total_mass: float, center: np.ndarray, bulk_velocity: np.ndarray,
                        num_particles: int = 50000, bulge_fraction: float = 0.2):
        """Generate a realistic galaxy with bulge, disk, and dark matter halo"""
        # Dark matter halo (NFW profile)
        halo_mass = total_mass * 0.85
        halo_particles = int(num_particles * 0.7)
        self._generate_nfw_halo(halo_mass, center, bulk_velocity, halo_particles)
        
        # Stellar bulge (Hernquist profile)
        bulge_mass = total_mass * bulge_fraction
        bulge_particles = int(num_particles * 0.1)
        self._generate_hernquist_bulge(bulge_mass, center, bulk_velocity, bulge_particles)
        
        # Stellar disk (exponential profile)
        disk_mass = total_mass * (0.15 - bulge_fraction)
        disk_particles = num_particles - halo_particles - bulge_particles
        self._generate_exponential_disk(disk_mass, center, bulk_velocity, disk_particles)

    def _generate_nfw_halo(self, mass: float, center: np.ndarray, bulk_velocity: np.ndarray, num_particles: int):
        """Generate Navarro-Frenk-White dark matter halo"""
        # NFW profile parameters
        r_s = 20 * const.kilo * const.parsec  # Scale radius
        r_max = 10 * r_s  # Truncation radius
        
        # Generate positions using inverse transform sampling
        u = np.random.random(num_particles)
        radii = r_s * (1 / (1 - u) - 1)  # Inverse CDF for NFW
        radii = np.clip(radii, 0, r_max)
        
        # Spherical coordinates
        theta = np.arccos(2 * np.random.random(num_particles) - 1)
        phi = 2 * np.pi * np.random.random(num_particles)
        
        # Convert to Cartesian coordinates
        x = radii * np.sin(theta) * np.cos(phi)
        y = radii * np.sin(theta) * np.sin(phi)
        z = radii * np.cos(theta)
        
        positions = np.column_stack([x, y, z]) + center
        
        # Circular velocities for NFW profile
        v_circ = self._nfw_circular_velocity(radii, mass, r_s)
        
        # Random orbital directions
        v_phi = 2 * np.pi * np.random.random(num_particles)
        v_x = -v_circ * np.sin(v_phi)
        v_y = v_circ * np.cos(v_phi)
        v_z = np.zeros(num_particles)
        
        velocities = np.column_stack([v_x, v_y, v_z]) + bulk_velocity
        
        # Create particles
        particle_mass = mass / num_particles
        for i in range(num_particles):
            particle = Particle(
                position=positions[i],
                velocity=velocities[i],
                mass=particle_mass,
                particle_type='dark_matter'
            )
            self.particles.append(particle)

    def _nfw_circular_velocity(self, r: np.ndarray, M_vir: float, r_s: float) -> np.ndarray:
        """Circular velocity for NFW profile"""
        c = 10.0  # Concentration parameter
        r_vir = r_s * c
        
        # Virial velocity
        V_vir = np.sqrt(self.G * M_vir * const.M_sun / r_vir)
        
        # Circular velocity
        x = r / r_s
        V_circ = V_vir * np.sqrt((np.log(1 + x) - x/(1 + x)) / (x * (np.log(1 + c) - c/(1 + c))))
        
        return V_circ

    def _generate_hernquist_bulge(self, mass: float, center: np.ndarray, bulk_velocity: np.ndarray, num_particles: int):
        """Generate Hernquist profile bulge"""
        a = 1 * const.kilo * const.parsec  # Scale radius
        
        # Generate positions
        u = np.random.random(num_particles)
        radii = a * np.sqrt(u) / (1 - np.sqrt(u))  # Inverse CDF for Hernquist
        
        # Spherical coordinates
        theta = np.arccos(2 * np.random.random(num_particles) - 1)
        phi = 2 * np.pi * np.random.random(num_particles)
        
        # Convert to Cartesian
        x = radii * np.sin(theta) * np.cos(phi)
        y = radii * np.sin(theta) * np.sin(phi)
        z = radii * np.cos(theta)
        
        positions = np.column_stack([x, y, z]) + center
        
        # Isotropic velocity dispersion
        sigma = self._hernquist_velocity_dispersion(radii, mass, a)
        velocities = np.random.normal(0, sigma, (num_particles, 3)) + bulk_velocity
        
        # Create particles
        particle_mass = mass / num_particles
        for i in range(num_particles):
            particle = Particle(
                position=positions[i],
                velocity=velocities[i],
                mass=particle_mass,
                particle_type='star'
            )
            self.particles.append(particle)

    def _hernquist_velocity_dispersion(self, r: np.ndarray, M: float, a: float) -> np.ndarray:
        """Velocity dispersion for Hernquist profile"""
        # Simplified approximation
        V_circ_max = np.sqrt(0.5 * self.G * M * const.M_sun / a)
        return V_circ_max / np.sqrt(2) * np.sqrt(a / (r + a))

    def _generate_exponential_disk(self, mass: float, center: np.ndarray, bulk_velocity: np.ndarray, num_particles: int):
        """Generate exponential disk"""
        R_d = 3 * const.kilo * const.parsec  # Disk scale length
        z_d = 0.1 * const.kilo * const.parsec  # Disk scale height
        
        # Radial distribution (exponential)
        u = np.random.random(num_particles)
        radii = -R_d * np.log(1 - u)
        
        # Azimuthal distribution
        phi = 2 * np.pi * np.random.random(num_particles)
        
        # Vertical distribution (sech²)
        z = z_d * np.arcsinh(2 * np.random.random(num_particles) - 1)
        
        # Convert to Cartesian
        x = radii * np.cos(phi)
        y = radii * np.sin(phi)
        
        positions = np.column_stack([x, y, z]) + center
        
        # Circular velocities
        V_circ = self._exponential_disk_circular_velocity(radii, mass, R_d)
        
        # Velocities with dispersion
        v_R = np.random.normal(0, 0.1 * V_circ, num_particles)
        v_phi = V_circ + np.random.normal(0, 0.15 * V_circ, num_particles)
        v_z = np.random.normal(0, 0.05 * V_circ, num_particles)
        
        # Convert to Cartesian velocities
        v_x = v_R * np.cos(phi) - v_phi * np.sin(phi)
        v_y = v_R * np.sin(phi) + v_phi * np.cos(phi)
        
        velocities = np.column_stack([v_x, v_y, v_z]) + bulk_velocity
        
        # Create particles
        particle_mass = mass / num_particles
        for i in range(num_particles):
            particle = Particle(
                position=positions[i],
                velocity=velocities[i],
                mass=particle_mass,
                particle_type='star'
            )
            self.particles.append(particle)

    def _exponential_disk_circular_velocity(self, R: np.ndarray, M: float, R_d: float) -> np.ndarray:
        """Circular velocity for exponential disk"""
        # Approximation using Bessel functions
        y = R / (2 * R_d)
        V_circ = np.sqrt(self.G * M * const.M_sun * R**2 / (R_d**3) * 
                        (special.i0(y) * special.k0(y) - special.i1(y) * special.k1(y)))
        return np.sqrt(V_circ)

    def evolve(self, time_step: Optional[float] = None):
        """Evolve the simulation by one time step"""
        if time_step is None:
            time_step = self.time_step
        
        start_time = time.time()
        
        # Update forces
        self._compute_gravitational_forces()
        
        # Update positions and velocities
        self._integrate_motion(time_step)
        
        # Handle astrophysical processes
        self._process_astrophysical_physics(time_step)
        
        # Update time
        self.current_time += time_step
        self.redshift = self._time_to_redshift(self.current_time)
        
        # Update statistics
        self._update_statistics()
        
        compute_time = time.time() - start_time
        print(f"Time: {self.current_time:.2e} s, Redshift: {self.redshift:.2f}, Compute: {compute_time:.2f} s")

    def _compute_gravitational_forces(self):
        """Compute gravitational forces using selected method"""
        if self.gravity_solver == "tree":
            self._compute_tree_gravity()
        elif self.gravity_solver == "pm":
            self._compute_pm_gravity()
        elif self.gravity_solver == "direct":
            self._compute_direct_gravity()
        else:
            self._compute_tree_gravity()  # Default

    def _compute_tree_gravity(self):
        """Compute gravity using Barnes-Hut tree algorithm"""
        if self.use_numba and len(self.particles) > 1000:
            self._compute_tree_gravity_numba()
        else:
            self._compute_tree_gravity_python()

    def _compute_tree_gravity_numba(self):
        """Numba-accelerated tree gravity computation"""
        positions = np.array([p.position for p in self.particles])
        masses = np.array([p.mass for p in self.particles])
        accelerations = np.zeros_like(positions)
        
        @jit(nopython=True, parallel=True)
        def compute_accelerations(positions, masses, accelerations, theta, softening):
            n = len(positions)
            for i in prange(n):
                for j in range(n):
                    if i == j:
                        continue
                    r = positions[j] - positions[i]
                    r_mag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2 + softening**2)
                    force_mag = const.G * masses[j] / (r_mag**3)
                    accelerations[i] += force_mag * r
            return accelerations
        
        accelerations = compute_accelerations(positions, masses, accelerations, self.theta_tree, self.softening_length)
        
        for i, particle in enumerate(self.particles):
            particle.acceleration = accelerations[i]

    def _compute_tree_gravity_python(self):
        """Python implementation of tree gravity (simplified)"""
        # Simplified version - full Barnes-Hut would be more complex
        for i, particle_i in enumerate(self.particles):
            total_force = np.zeros(3)
            for j, particle_j in enumerate(self.particles):
                if i == j:
                    continue
                r = particle_j.position - particle_i.position
                r_mag = np.linalg.norm(r) + self.softening_length
                force_mag = self.G * particle_j.mass / (r_mag**3)
                total_force += force_mag * r
            particle_i.acceleration = total_force

    def _compute_pm_gravity(self):
        """Compute gravity using Particle-Mesh method"""
        if self.density_field is None:
            self.density_field = np.zeros((self.resolution, self.resolution, self.resolution))
        
        # Deposit mass to grid
        self._deposit_mass_to_grid()
        
        # Solve Poisson equation
        potential_field = self._solve_poisson_equation(self.density_field)
        
        # Compute gravitational field
        gravitational_field = np.gradient(potential_field)
        
        # Interpolate forces to particles
        self._interpolate_forces_to_particles(gravitational_field)

    def _deposit_mass_to_grid(self):
        """Deposit particle masses to density grid using cloud-in-cell"""
        self.density_field.fill(0.0)
        cell_size = self.box_size / self.resolution
        
        for particle in self.particles:
            # Normalized position in grid coordinates
            pos_norm = (particle.position + self.box_size / 2) / cell_size
            
            # Grid indices
            i, j, k = np.floor(pos_norm).astype(int)
            dx, dy, dz = pos_norm - np.floor(pos_norm)
            
            # CIC weights
            weights = [
                (1 - dx) * (1 - dy) * (1 - dz),
                dx * (1 - dy) * (1 - dz),
                (1 - dx) * dy * (1 - dz),
                dx * dy * (1 - dz),
                (1 - dx) * (1 - dy) * dz,
                dx * (1 - dy) * dz,
                (1 - dx) * dy * dz,
                dx * dy * dz
            ]
            
            # Deposit to 8 surrounding cells
            for di in [0, 1]:
                for dj in [0, 1]:
                    for dk in [0, 1]:
                        idx_i = (i + di) % self.resolution
                        idx_j = (j + dj) % self.resolution
                        idx_k = (k + dk) % self.resolution
                        weight_index = di + 2 * dj + 4 * dk
                        self.density_field[idx_i, idx_j, idx_k] += (
                            particle.mass * weights[weight_index] / cell_size**3
                        )

    def _solve_poisson_equation(self, density_field: np.ndarray) -> np.ndarray:
        """Solve Poisson equation ∇²Φ = 4πGρ using FFT"""
        # Fourier transform density
        rho_k = np.fft.fftn(density_field)
        
        # k-space grid
        k_grid = np.fft.fftfreq(self.resolution) * 2 * np.pi / (self.box_size / self.resolution)
        kx, ky, kz = np.meshgrid(k_grid, k_grid, k_grid, indexing='ij')
        k_squared = kx**2 + ky**2 + kz**2
        
        # Avoid division by zero
        k_squared[0, 0, 0] = 1.0
        
        # Potential in Fourier space: Φ_k = -4πG ρ_k / k²
        potential_k = -4 * np.pi * self.G * rho_k / k_squared
        potential_k[0, 0, 0] = 0  # Set zero mode to zero
        
        # Inverse Fourier transform
        potential = np.fft.ifftn(potential_k).real
        
        return potential

    def _interpolate_forces_to_particles(self, gravitational_field: np.ndarray):
        """Interpolate gravitational forces from grid to particles"""
        cell_size = self.box_size / self.resolution
        
        for particle in self.particles:
            # Normalized position in grid coordinates
            pos_norm = (particle.position + self.box_size / 2) / cell_size
            
            # Trilinear interpolation
            force = np.zeros(3)
            for comp in range(3):
                # Interpolate each component separately
                force[comp] = ndimage.map_coordinates(
                    gravitational_field[comp], 
                    [pos_norm[0], pos_norm[1], pos_norm[2]], 
                    order=1, mode='wrap'
                )
            
            particle.acceleration = force

    def _integrate_motion(self, time_step: float):
        """Integrate particle motion using leapfrog method"""
        for particle in self.particles:
            # Kick: update velocities
            particle.velocity += particle.acceleration * time_step / 2
            
            # Drift: update positions
            particle.position += particle.velocity * time_step
            
            # Kick: update velocities with new accelerations
            particle.velocity += particle.acceleration * time_step / 2

    def _process_astrophysical_physics(self, time_step: float):
        """Process astrophysical physics (star formation, feedback, etc.)"""
        # Star formation
        self._process_star_formation(time_step)
        
        # Supernova feedback
        self._process_supernova_feedback(time_step)
        
        # Black hole accretion and feedback
        self._process_black_hole_physics(time_step)
        
        # Radiative cooling
        self._process_radiative_cooling(time_step)

    def _process_star_formation(self, time_step: float):
        """Process star formation based on local conditions"""
        star_formation_rate = 0.0
        
        for particle in self.particles:
            if particle.particle_type == 'gas':
                # Simple star formation recipe
                density_threshold = 1e-21  # kg/m³
                free_fall_time = 1 / np.sqrt(self.G * particle.mass)
                
                if particle.mass > density_threshold and np.random.random() < time_step / free_fall_time:
                    # Convert gas particle to star particle
                    particle.particle_type = 'star'
                    particle.formation_time = self.current_time
                    particle.temperature = 1e4  # Typical stellar temperature
                    
                    star_formation_rate += particle.mass / time_step
        
        self.statistics['star_formation_rate'] = star_formation_rate

    def _process_supernova_feedback(self, time_step: float):
        """Process supernova feedback from massive stars"""
        supernova_energy_per_event = 1e44  # Joules
        supernova_rate = 1e-12  # Supernovae per solar mass per year
        
        for particle in self.particles:
            if particle.particle_type == 'star':
                star_age = self.current_time - particle.formation_time
                
                # Massive stars explode as supernovae after ~10 million years
                if 1e6 * const.year < star_age < 5e7 * const.year:
                    supernova_probability = particle.mass * supernova_rate * time_step
                    
                    if np.random.random() < supernova_probability:
                        # Add energy to surrounding gas
                        explosion_energy = supernova_energy_per_event
                        self._distribute_supernova_energy(particle.position, explosion_energy)
                        
                        # Convert star to gas or remove it
                        particle.particle_type = 'gas'
                        particle.temperature = 1e6  # Hot gas from supernova

    def _distribute_supernova_energy(self, position: np.ndarray, energy: float):
        """Distribute supernova energy to nearby gas particles"""
        for particle in self.particles:
            if particle.particle_type == 'gas':
                distance = np.linalg.norm(particle.position - position)
                if distance < 100 * const.parsec:
                    # Add thermal energy
                    specific_energy = energy / particle.mass
                    particle.temperature += specific_energy * particle.mass / (1.5 * self.k_B)

    def _process_black_hole_physics(self, time_step: float):
        """Process black hole accretion and feedback"""
        for black_hole in self.black_holes:
            # Bondi-Hoyle accretion
            accretion_rate = self._compute_bondi_accretion(black_hole)
            black_hole.accretion_rate = accretion_rate
            
            # Radiative feedback
            luminosity = 0.1 * accretion_rate * const.c**2  # 10% efficiency
            self._apply_black_hole_feedback(black_hole, luminosity, time_step)

    def _compute_bondi_accretion(self, black_hole: BlackHole) -> float:
        """Compute Bondi-Hoyle accretion rate"""
        # Find nearby gas particles
        total_density = 0.0
        sound_speed = 0.0
        relative_velocity = 0.0
        
        for particle in self.particles:
            if particle.particle_type == 'gas':
                distance = np.linalg.norm(particle.position - black_hole.position)
                if distance < 100 * const.parsec:
                    total_density += particle.mass
                    # Simplified sound speed estimate
                    sound_speed = np.sqrt(self.k_B * particle.temperature / (1.67e-27))  # Proton mass
                    relative_velocity = np.linalg.norm(particle.velocity)
        
        # Bondi-Hoyle formula
        bondi_radius = 2 * self.G * black_hole.mass * const.M_sun / (sound_speed**2 + relative_velocity**2)
        accretion_rate = 4 * np.pi * bondi_radius**2 * total_density * sound_speed
        
        return accretion_rate

    def _apply_black_hole_feedback(self, black_hole: BlackHole, luminosity: float, time_step: float):
        """Apply black hole feedback to surrounding gas"""
        feedback_energy = luminosity * time_step
        
        for particle in self.particles:
            if particle.particle_type == 'gas':
                distance = np.linalg.norm(particle.position - black_hole.position)
                if distance < 1 * const.kilo * const.parsec:
                    # Add energy inversely proportional to distance
                    energy_fraction = 1.0 / (1.0 + distance / (100 * const.parsec))
                    particle.temperature += feedback_energy * energy_fraction / particle.mass

    def _process_radiative_cooling(self, time_step: float):
        """Process radiative cooling of gas"""
        for particle in self.particles:
            if particle.particle_type == 'gas':
                # Simple cooling function (approximate)
                cooling_rate = 1e-22  # W/m³ (approximate for interstellar medium)
                cooling_energy = cooling_rate * time_step / particle.mass
                particle.temperature = max(10.0, particle.temperature - cooling_energy)

    def _time_to_redshift(self, cosmic_time: float) -> float:
        """Convert cosmic time to redshift using ΛCDM cosmology"""
        # Numerical integration of Friedmann equation
        def friedmann_integrand(z):
            H = self.cosmological_params.H0 * 1000 / const.parsec  # Convert to 1/s
            E_z = np.sqrt(self.cosmological_params.Omega_m * (1 + z)**3 + 
                         self.cosmological_params.Omega_r * (1 + z)**4 + 
                         self.cosmological_params.Omega_lambda)
            return 1 / (H * E_z * (1 + z))
        
        # Find redshift that gives the correct lookback time
        target_time = cosmic_time
        z_guess = 10.0
        
        for _ in range(10):  # Newton-Raphson iteration
            t, _ = integrate.quad(friedmann_integrand, 0, z_guess)
            dt_dz = friedmann_integrand(z_guess)
            z_guess = z_guess - (t - target_time) / dt_dz
        
        return z_guess

    def _update_statistics(self):
        """Update simulation statistics"""
        total_mass = 0.0
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        for particle in self.particles:
            total_mass += particle.mass
            kinetic_energy += 0.5 * particle.mass * np.linalg.norm(particle.velocity)**2
        
        # Approximate potential energy (simplified)
        potential_energy = -self.G * total_mass**2 / self.box_size
        
        self.statistics.update({
            'total_mass': total_mass,
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'virial_ratio': -2 * kinetic_energy / potential_energy if potential_energy < 0 else 0.0
        })

    def add_black_hole(self, position: np.ndarray, mass: float, spin: float = 0.0):
        """Add a black hole to the simulation"""
        black_hole = BlackHole(
            position=position,
            mass=mass,
            spin=spin
        )
        self.black_holes.append(black_hole)
        
        # Also add as a particle for gravity computation
        particle = Particle(
            position=position,
            velocity=np.zeros(3),
            mass=mass * const.M_sun,
            particle_type='black_hole'
        )
        self.particles.append(particle)

    def simulate_gravitational_lensing(self, source_positions: np.ndarray, observer_position: np.ndarray) -> np.ndarray:
        """Simulate gravitational lensing by massive objects"""
        lensed_positions = []
        
        for source_pos in source_positions:
            def null_geodesic_derivatives(lambda_, y):
                # y = [t, x, y, z, dt/dλ, dx/dλ, dy/dλ, dz/dλ]
                # Simplified for weak field
                r = np.linalg.norm(y[1:4])
                if r == 0:
                    return np.zeros(8)
                
                # Schwarzschild metric perturbation
                phi = -self.G * self.statistics['total_mass'] / r
                
                # Geodesic equations (Post-Newtonian approximation)
                d2t = 0
                acceleration = -2 * np.gradient(phi, y[1:4])
                
                return np.concatenate([[y[4], y[5], y[6], y[7]], [d2t, acceleration[0], acceleration[1], acceleration[2]]])
            
            # Solve geodesic equation
            y0 = np.concatenate([observer_position, source_pos - observer_position, [1, 1, 1, 1]])
            solution = integrate.solve_ivp(null_geodesic_derivatives, [0, 1], y0, method='RK45')
            
            lensed_pos = solution.y[1:4, -1]
            lensed_positions.append(lensed_pos)
        
        return np.array(lensed_positions)

    def get_visualization_data(self, data_type: str = 'density') -> np.ndarray:
        """Get data for visualization"""
        if data_type == 'density':
            if self.density_field is not None:
                return self.density_field
            else:
                return self._compute_projection('mass')
        elif data_type == 'temperature':
            return self._compute_projection('temperature')
        elif data_type == 'velocity':
            return self._compute_projection('velocity_magnitude')
        else:
            return self._compute_projection('mass')

    def _compute_projection(self, quantity: str) -> np.ndarray:
        """Compute 2D projection of 3D data"""
        projection = np.zeros((self.resolution, self.resolution))
        cell_size = self.box_size / self.resolution
        
        for particle in self.particles:
            pos_norm = (particle.position + self.box_size / 2) / cell_size
            i, j = int(pos_norm[0]), int(pos_norm[1])
            
            if 0 <= i < self.resolution and 0 <= j < self.resolution:
                if quantity == 'mass':
                    projection[i, j] += particle.mass
                elif quantity == 'temperature' and hasattr(particle, 'temperature'):
                    projection[i, j] += particle.temperature * particle.mass
                elif quantity == 'velocity_magnitude':
                    projection[i, j] += np.linalg.norm(particle.velocity) * particle.mass
        
        # Normalize if needed
        if quantity in ['temperature', 'velocity_magnitude']:
            mass_projection = self._compute_projection('mass')
            projection = np.divide(projection, mass_projection, 
                                 out=np.zeros_like(projection), 
                                 where=mass_projection > 0)
        
        return projection

    def save_snapshot(self, filename: str):
        """Save simulation snapshot to HDF5 file"""
        with h5py.File(filename, 'w') as f:
            # Save parameters
            params_group = f.create_group('parameters')
            params_group.attrs['box_size'] = self.box_size
            params_group.attrs['current_time'] = self.current_time
            params_group.attrs['redshift'] = self.redshift
            
            # Save particles
            particles_group = f.create_group('particles')
            positions = np.array([p.position for p in self.particles])
            velocities = np.array([p.velocity for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            types = np.array([p.particle_type for p in self.particles], dtype='S')
            
            particles_group.create_dataset('positions', data=positions)
            particles_group.create_dataset('velocities', data=velocities)
            particles_group.create_dataset('masses', data=masses)
            particles_group.create_dataset('types', data=types)
            
            # Save black holes
            if self.black_holes:
                bh_group = f.create_group('black_holes')
                bh_positions = np.array([bh.position for bh in self.black_holes])
                bh_masses = np.array([bh.mass for bh in self.black_holes])
                bh_spins = np.array([bh.spin for bh in self.black_holes])
                
                bh_group.create_dataset('positions', data=bh_positions)
                bh_group.create_dataset('masses', data=bh_masses)
                bh_group.create_dataset('spins', data=bh_spins)
            
            # Save statistics
            stats_group = f.create_group('statistics')
            for key, value in self.statistics.items():
                stats_group.attrs[key] = value
        
        print(f"Simulation snapshot saved to {filename}")

    def load_snapshot(self, filename: str):
        """Load simulation snapshot from HDF5 file"""
        with h5py.File(filename, 'r') as f:
            # Load parameters
            params_group = f['parameters']
            self.box_size = params_group.attrs['box_size']
            self.current_time = params_group.attrs['current_time']
            self.redshift = params_group.attrs['redshift']
            
            # Load particles
            particles_group = f['particles']
            positions = particles_group['positions'][:]
            velocities = particles_group['velocities'][:]
            masses = particles_group['masses'][:]
            types = particles_group['types'][:]
            
            self.particles.clear()
            for i in range(len(positions)):
                particle = Particle(
                    position=positions[i],
                    velocity=velocities[i],
                    mass=masses[i],
                    particle_type=types[i].decode()
                )
                self.particles.append(particle)
            
            # Load black holes
            if 'black_holes' in f:
                bh_group = f['black_holes']
                bh_positions = bh_group['positions'][:]
                bh_masses = bh_group['masses'][:]
                bh_spins = bh_group['spins'][:]
                
                self.black_holes.clear()
                for i in range(len(bh_positions)):
                    black_hole = BlackHole(
                        position=bh_positions[i],
                        mass=bh_masses[i],
                        spin=bh_spins[i]
                    )
                    self.black_holes.append(black_hole)
            
            # Load statistics
            stats_group = f['statistics']
            for key in stats_group.attrs:
                self.statistics[key] = stats_group.attrs[key]
        
        print(f"Simulation snapshot loaded from {filename}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'num_particles': len(self.particles),
            'num_black_holes': len(self.black_holes),
            'current_time': self.current_time,
            'redshift': self.redshift,
            'total_mass': self.statistics['total_mass'],
            'virial_ratio': self.statistics['virial_ratio'],
            'star_formation_rate': self.statistics['star_formation_rate']
        }

    def cleanup(self):
        """Clean up resources"""
        self.particles.clear()
        self.black_holes.clear()
        self.density_field = None
        self.velocity_field = None
        self.temperature_field = None
        print("Astrophysics simulator cleaned up")

class CosmicVisualization:
    """Advanced cosmic phenomena visualization system"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        # Visualization settings
        self.camera_position = glm.vec3(0, 0, 5)
        self.camera_target = glm.vec3(0, 0, 0)
        self.camera_up = glm.vec3(0, 1, 0)
        self.field_of_view = 45.0
        
        # Rendering resources
        self.shader_programs = {}
        self.textures = {}
        self.vertex_buffers = {}
        
        # Color mapping
        self.color_transfer_functions = {}
        self._initialize_transfer_functions()
        
        # Time and animation
        self.animation_speed = 1.0
        self.current_frame = 0
        
        print(f"Cosmic Visualization initialized: {width}x{height}")

    def _initialize_transfer_functions(self):
        """Initialize color transfer functions for different physical quantities"""
        # Density transfer function
        self.color_transfer_functions['density'] = {
            'range': (1e-30, 1e-18),
            'colors': [
                (0.0, 0.0, 0.0, 0.0),      # Transparent black
                (0.1, 0.0, 0.3, 0.3),      # Dark purple
                (0.3, 0.2, 0.5, 0.6),      # Purple
                (0.5, 0.4, 0.7, 0.8),      # Light purple
                (0.7, 0.8, 0.9, 1.0),      # Blue-white
                (1.0, 1.0, 1.0, 1.0)       # White
            ]
        }
        
        # Temperature transfer function
        self.color_transfer_functions['temperature'] = {
            'range': (10, 1e8),
            'colors': [
                (0.0, 0.0, 0.0, 1.0),      # Black
                (0.2, 0.0, 0.0, 1.0),      # Dark blue
                (0.4, 0.0, 1.0, 1.0),      # Cyan
                (0.6, 0.0, 1.0, 0.0),      # Green
                (0.8, 1.0, 1.0, 0.0),      # Yellow
                (1.0, 1.0, 0.0, 0.0)       # Red
            ]
        }

    def initialize_opengl(self):
        """Initialize OpenGL resources for cosmic visualization"""
        self._initialize_shaders()
        self._initialize_geometry()
        print("OpenGL resources initialized for cosmic visualization")

    def _initialize_shaders(self):
        """Initialize shader programs for different rendering techniques"""
        # Volume rendering shader
        volume_vertex = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        out vec3 TexCoords;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            TexCoords = aPos;
        }
        """
        
        volume_fragment = """
        #version 330 core
        in vec3 TexCoords;
        out vec4 FragColor;
        uniform sampler3D volumeTexture;
        uniform sampler1D transferFunction;
        uniform vec2 dataRange;
        void main() {
            // Sample volume data
            float value = texture(volumeTexture, TexCoords).r;
            
            // Normalize to [0, 1] range
            value = (value - dataRange.x) / (dataRange.y - dataRange.x);
            value = clamp(value, 0.0, 1.0);
            
            // Apply transfer function
            vec4 color = texture(transferFunction, value);
            
            // Simple emission-absorption model
            FragColor = color;
        }
        """
        
        self.shader_programs['volume'] = compileProgram(
            compileShader(volume_vertex, GL_VERTEX_SHADER),
            compileShader(volume_fragment, GL_FRAGMENT_SHADER)
        )
        
        # Particle rendering shader
        particle_vertex = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aColor;
        layout (location = 2) in float aSize;
        out vec3 Color;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(aPos, 1.0);
            Color = aColor;
            gl_PointSize = aSize;
        }
        """
        
        particle_fragment = """
        #version 330 core
        in vec3 Color;
        out vec4 FragColor;
        void main() {
            // Circular point sprites
            vec2 coord = gl_PointCoord - vec2(0.5);
            if(length(coord) > 0.5)
                discard;
            FragColor = vec4(Color, 1.0);
        }
        """
        
        self.shader_programs['particle'] = compileProgram(
            compileShader(particle_vertex, GL_VERTEX_SHADER),
            compileShader(particle_fragment, GL_FRAGMENT_SHADER)
        )

    def _initialize_geometry(self):
        """Initialize geometric primitives for rendering"""
        # Fullscreen quad for volume rendering
        quad_vertices = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0,  1.0, 0.0,
        ], dtype=np.float32)
        
        self.vertex_buffers['fullscreen_quad'] = quad_vertices

    def render_simulation(self, simulator: AstrophysicsSimulator, render_mode: str = 'volume'):
        """Render astrophysics simulation"""
        if render_mode == 'volume':
            self._render_volume(simulator)
        elif render_mode == 'particles':
            self._render_particles(simulator)
        elif render_mode == 'black_holes':
            self._render_black_holes(simulator)
        else:
            self._render_volume(simulator)  # Default

    def _render_volume(self, simulator: AstrophysicsSimulator):
        """Render simulation data as volume"""
        glUseProgram(self.shader_programs['volume'])
        
        # Get volume data
        volume_data = simulator.get_visualization_data('density')
        if volume_data is None:
            return
        
        # Create 3D texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, texture_id)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, volume_data.shape[0], volume_data.shape[1], volume_data.shape[2], 
                     0, GL_RED, GL_FLOAT, volume_data)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        
        # Set up transformation matrices
        model = glm.mat4(1.0)
        view = glm.lookAt(self.camera_position, self.camera_target, self.camera_up)
        projection = glm.perspective(glm.radians(self.field_of_view), self.width/self.height, 0.1, 100.0)
        
        # Pass uniforms
        glUniformMatrix4fv(glGetUniformLocation(self.shader_programs['volume'], "model"), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_programs['volume'], "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_programs['volume'], "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        
        # Render fullscreen quad
        self._render_fullscreen_quad()
        
        glDeleteTextures(1, [texture_id])

    def _render_particles(self, simulator: AstrophysicsSimulator):
        """Render particles as point sprites"""
        glUseProgram(self.shader_programs['particle'])
        
        # Prepare particle data
        positions = []
        colors = []
        sizes = []
        
        for particle in simulator.particles:
            positions.append(particle.position / simulator.box_size)  # Normalize
            
            # Color by particle type
            if particle.particle_type == 'dark_matter':
                color = (0.3, 0.3, 0.8)  # Blue
                size = 2.0
            elif particle.particle_type == 'gas':
                color = (0.8, 0.5, 0.2)  # Orange
                size = 3.0
            elif particle.particle_type == 'star':
                color = (1.0, 1.0, 0.8)  # Yellow-white
                size = 4.0
            else:  # black_hole
                color = (0.0, 0.0, 0.0)  # Black
                size = 8.0
            
            colors.append(color)
            sizes.append(size)
        
        if not positions:
            return
        
        positions = np.array(positions, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)
        
        # Set up transformation matrices
        model = glm.mat4(1.0)
        view = glm.lookAt(self.camera_position, self.camera_target, self.camera_up)
        projection = glm.perspective(glm.radians(self.field_of_view), self.width/self.height, 0.1, 100.0)
        
        glUniformMatrix4fv(glGetUniformLocation(self.shader_programs['particle'], "model"), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_programs['particle'], "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(self.shader_programs['particle'], "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        
        # Render particles
        self._render_point_cloud(positions, colors, sizes)

    def _render_black_holes(self, simulator: AstrophysicsSimulator):
        """Render black holes with accretion disks"""
        for black_hole in simulator.black_holes:
            # Render black hole as dark sphere
            # Render accretion disk as bright ring
            # Render jet as particle streams
            pass  # Implementation would be complex

    def _render_fullscreen_quad(self):
        """Render fullscreen quad for volume rendering"""
        vertices = self.vertex_buffers['fullscreen_quad']
        
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo])

    def _render_point_cloud(self, positions: np.ndarray, colors: np.ndarray, sizes: np.ndarray):
        """Render point cloud with colors and sizes"""
        vao = glGenVertexArrays(1)
        
        # Position buffer
        vbo_positions = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
        
        # Color buffer
        vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        
        # Size buffer
        vbo_sizes = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sizes)
        glBufferData(GL_ARRAY_BUFFER, sizes.nbytes, sizes, GL_STATIC_DRAW)
        
        glBindVertexArray(vao)
        
        # Position attribute
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
        
        # Size attribute
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sizes)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)
        
        # Enable point sprites
        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_POINT_SPRITE)
        
        glDrawArrays(GL_POINTS, 0, len(positions))
        
        glDeleteVertexArrays(1, [vao])
        glDeleteBuffers(1, [vbo_positions])
        glDeleteBuffers(1, [vbo_colors])
        glDeleteBuffers(1, [vbo_sizes])

    def set_camera(self, position: glm.vec3, target: glm.vec3, up: glm.vec3 = None):
        """Set camera position and orientation"""
        self.camera_position = position
        self.camera_target = target
        if up is not None:
            self.camera_up = up

    def set_field_of_view(self, fov: float):
        """Set camera field of view"""
        self.field_of_view = fov

    def cleanup(self):
        """Clean up visualization resources"""
        for program in self.shader_programs.values():
            glDeleteProgram(program)
        self.shader_programs.clear()
        self.vertex_buffers.clear()
        print("Cosmic visualization cleaned up")

# Example usage and testing
if __name__ == "__main__":
    # Test astrophysics simulator
    print("Testing Astrophysics & Cosmic Simulations...")
    
    # Create galaxy collision simulation
    simulator = AstrophysicsSimulator(CosmicScale.GALACTIC, PhysicsModel.NEWTONIAN)
    simulator.set_initial_conditions("galaxy_collision")
    
    # Add a supermassive black hole
    simulator.add_black_hole(np.array([0, 0, 0]), mass=1e6)
    
    # Run simulation for a few steps
    for step in range(10):
        simulator.evolve(time_step=1e15)  # ~30,000 years
        stats = simulator.get_performance_stats()
        print(f"Step {step}: {stats['num_particles']} particles, virial ratio: {stats['virial_ratio']:.3f}")
    
    # Save snapshot
    simulator.save_snapshot("galaxy_collision.h5")
    
    # Test visualization
    visualization = CosmicVisualization(800, 600)
    visualization.initialize_opengl()
    
    print("Astrophysics simulation test completed")
    
    # Cleanup
    simulator.cleanup()
    visualization.cleanup()