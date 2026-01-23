"""
Anatomy Engine Module
Physics-based rendering of organs and tissues with realistic biomechanics
"""

import numpy as np
import pywavefront
from typing import Dict, List, Tuple, Optional, Any
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Optional OpenGL imports
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL not available, using software rendering")

class TissueType(Enum):
    """Types of biological tissues"""
    SKIN = "skin"
    MUSCLE = "muscle"
    BONE = "bone"
    FAT = "fat"
    LIGAMENT = "ligament"
    TENDON = "tendon"
    CARTILAGE = "cartilage"
    BLOOD_VESSEL = "blood_vessel"
    NERVE = "nerve"
    ORGAN = "organ"

class MaterialProperty:
    """Material properties for biological tissues"""
    
    def __init__(self, tissue_type: TissueType):
        self.tissue_type = tissue_type
        self.properties = self._get_default_properties()
    
    def _get_default_properties(self):
        """Get default physical properties for tissue type"""
        properties = {
            TissueType.SKIN: {
                'density': 1100,  # kg/m³
                'youngs_modulus': 0.42e6,  # Pa
                'poissons_ratio': 0.48,
                'shear_modulus': 0.15e6,
                'damping_coefficient': 0.3,
                'friction_coefficient': 0.6,
                'thermal_conductivity': 0.37,  # W/(m·K)
                'specific_heat': 3500,  # J/(kg·K)
                'perfusion_rate': 0.0012,  # 1/s
                'color': [1.0, 0.9, 0.8, 1.0]  # RGBA
            },
            TissueType.MUSCLE: {
                'density': 1050,
                'youngs_modulus': 0.12e6,
                'poissons_ratio': 0.49,
                'shear_modulus': 0.04e6,
                'damping_coefficient': 0.4,
                'friction_coefficient': 0.8,
                'thermal_conductivity': 0.42,
                'specific_heat': 3800,
                'perfusion_rate': 0.008,
                'color': [0.8, 0.2, 0.2, 1.0]
            },
            TissueType.BONE: {
                'density': 1900,
                'youngs_modulus': 15e9,
                'poissons_ratio': 0.3,
                'shear_modulus': 5.8e9,
                'damping_coefficient': 0.1,
                'friction_coefficient': 0.3,
                'thermal_conductivity': 0.4,
                'specific_heat': 1300,
                'perfusion_rate': 0.0001,
                'color': [0.96, 0.96, 0.96, 1.0]
            },
            TissueType.FAT: {
                'density': 900,
                'youngs_modulus': 0.02e6,
                'poissons_ratio': 0.49,
                'shear_modulus': 0.0067e6,
                'damping_coefficient': 0.5,
                'friction_coefficient': 0.9,
                'thermal_conductivity': 0.21,
                'specific_heat': 2300,
                'perfusion_rate': 0.003,
                'color': [1.0, 0.9, 0.6, 1.0]
            },
            TissueType.BLOOD_VESSEL: {
                'density': 1060,
                'youngs_modulus': 0.6e6,
                'poissons_ratio': 0.45,
                'shear_modulus': 0.21e6,
                'damping_coefficient': 0.35,
                'friction_coefficient': 0.7,
                'thermal_conductivity': 0.5,
                'specific_heat': 3600,
                'perfusion_rate': 0.05,
                'color': [0.8, 0.1, 0.1, 1.0]
            }
        }
        return properties.get(self.tissue_type, {
            'density': 1000,
            'youngs_modulus': 1e6,
            'poissons_ratio': 0.45,
            'shear_modulus': 0.345e6,
            'damping_coefficient': 0.3,
            'friction_coefficient': 0.5,
            'thermal_conductivity': 0.5,
            'specific_heat': 3500,
            'perfusion_rate': 0.001,
            'color': [0.7, 0.7, 0.7, 1.0]
        })
    
    def get_property(self, property_name: str) -> float:
        """Get specific property value"""
        return self.properties.get(property_name, 0.0)
    
    def set_property(self, property_name: str, value: float):
        """Set property value"""
        self.properties[property_name] = value

@dataclass
class AnatomicalVertex:
    """Vertex in anatomical mesh with physical properties"""
    position: np.ndarray  # [x, y, z]
    normal: np.ndarray    # [nx, ny, nz]
    tissue_type: TissueType
    material_id: int
    temperature: float = 37.0  # Celsius
    blood_flow: float = 0.0    # ml/s
    deformation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    stress: np.ndarray = field(default_factory=lambda: np.zeros(6))  # stress tensor
    
    def apply_force(self, force: np.ndarray, dt: float, properties: Dict):
        """Apply force to vertex and update deformation"""
        # Simplified viscoelastic deformation
        youngs_modulus = properties.get('youngs_modulus', 1e6)
        damping = properties.get('damping_coefficient', 0.3)
        
        # Hooke's law: stress = E * strain
        strain = force / youngs_modulus
        
        # Add damping
        strain *= (1.0 - damping)
        
        # Update deformation
        self.deformation += strain * dt
        
        # Update stress (simplified)
        self.stress[:3] = force * youngs_modulus
    
    def reset_deformation(self):
        """Reset deformation to zero"""
        self.deformation = np.zeros(3)
        self.stress = np.zeros(6)

@dataclass
class AnatomicalFace:
    """Face/triangle in anatomical mesh"""
    vertices: List[int]  # indices of vertices
    normal: np.ndarray
    area: float
    tissue_thickness: float = 1.0
    vascularity: float = 0.5  # 0-1 scale
    innervation: float = 0.5   # 0-1 scale
    
    def calculate_normal(self, vertices: List[AnatomicalVertex]) -> np.ndarray:
        """Calculate face normal from vertices"""
        if len(self.vertices) < 3:
            return np.array([0, 0, 1])
        
        v0 = vertices[self.vertices[0]].position
        v1 = vertices[self.vertices[1]].position
        v2 = vertices[self.vertices[2]].position
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        
        if norm > 0:
            return normal / norm
        return np.array([0, 0, 1])
    
    def calculate_area(self, vertices: List[AnatomicalVertex]) -> float:
        """Calculate face area"""
        if len(self.vertices) < 3:
            return 0.0
        
        v0 = vertices[self.vertices[0]].position
        v1 = vertices[self.vertices[1]].position
        v2 = vertices[self.vertices[2]].position
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        cross_product = np.cross(edge1, edge2)
        return 0.5 * np.linalg.norm(cross_product)

class OrganModel:
    """3D model of an organ with physical properties"""
    
    def __init__(self, organ_name: str, detail_level: str = 'high'):
        self.organ_name = organ_name
        self.detail_level = detail_level
        self.vertices: List[AnatomicalVertex] = []
        self.faces: List[AnatomicalFace] = []
        self.materials: Dict[int, MaterialProperty] = {}
        self.textures = {}
        self.animations = {}
        self.physiology_models = {}
        
        # Physical state
        self.temperature = 37.0  # Celsius
        self.blood_flow = 0.0    # ml/s
        self.oxygenation = 0.98   # 0-1 scale
        self.metabolic_rate = 1.0 # relative to baseline
        
        # Deformation state
        self.deformation_enabled = True
        self.max_deformation = 0.1  # maximum deformation factor
        
        # Rendering
        self.display_list = None
        self.initialized = False
    
    def load_from_file(self, filepath: str):
        """Load organ model from file (OBJ, STL, etc.)"""
        print(f"Loading organ model: {self.organ_name} from {filepath}")
        
        try:
            # Try to load Wavefront OBJ file
            scene = pywavefront.Wavefront(
                filepath,
                collect_faces=True,
                create_materials=True
            )
            
            # Extract vertices and faces
            for name, mesh in scene.meshes.items():
                for vertex in mesh.vertices:
                    pos = np.array(vertex[:3])
                    normal = np.array(vertex[3:6]) if len(vertex) > 3 else np.array([0, 0, 1])
                    
                    # Determine tissue type from mesh name
                    tissue_type = self._infer_tissue_type(name)
                    
                    vertex_obj = AnatomicalVertex(
                        position=pos,
                        normal=normal,
                        tissue_type=tissue_type,
                        material_id=0
                    )
                    self.vertices.append(vertex_obj)
            
            print(f"Loaded {len(self.vertices)} vertices")
            
        except Exception as e:
            print(f"Error loading model file: {e}")
            # Generate simple geometric model as fallback
            self._generate_fallback_model()
    
    def _infer_tissue_type(self, mesh_name: str) -> TissueType:
        """Infer tissue type from mesh name"""
        mesh_name_lower = mesh_name.lower()
        
        if 'skin' in mesh_name_lower:
            return TissueType.SKIN
        elif 'muscle' in mesh_name_lower:
            return TissueType.MUSCLE
        elif 'bone' in mesh_name_lower:
            return TissueType.BONE
        elif 'fat' in mesh_name_lower or 'adipose' in mesh_name_lower:
            return TissueType.FAT
        elif 'vessel' in mesh_name_lower or 'artery' in mesh_name_lower or 'vein' in mesh_name_lower:
            return TissueType.BLOOD_VESSEL
        elif 'nerve' in mesh_name_lower:
            return TissueType.NERVE
        elif 'cartilage' in mesh_name_lower:
            return TissueType.CARTILAGE
        elif 'tendon' in mesh_name_lower:
            return TissueType.TENDON
        elif 'ligament' in mesh_name_lower:
            return TissueType.LIGAMENT
        else:
            return TissueType.ORGAN
    
    def _generate_fallback_model(self):
        """Generate fallback geometric model"""
        print(f"Generating fallback model for {self.organ_name}")
        
        # Generate sphere for organ
        subdivisions = 3 if self.detail_level == 'high' else 2 if self.detail_level == 'medium' else 1
        
        # Create icosphere
        vertices, faces = self._create_icosphere(subdivisions, 1.0)
        
        for pos, normal in vertices:
            vertex = AnatomicalVertex(
                position=pos,
                normal=normal,
                tissue_type=TissueType.ORGAN,
                material_id=0
            )
            self.vertices.append(vertex)
        
        for face_indices in faces:
            face = AnatomicalFace(
                vertices=list(face_indices),
                normal=np.array([0, 0, 1]),
                area=0.1
            )
            self.faces.append(face)
        
        # Calculate normals and areas
        for face in self.faces:
            face.normal = face.calculate_normal(self.vertices)
            face.area = face.calculate_area(self.vertices)
    
    def _create_icosphere(self, subdivisions: int, radius: float) -> Tuple[List, List]:
        """Create icosphere mesh"""
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Initial vertices of icosahedron
        vertices = [
            np.array([-1, phi, 0]),
            np.array([1, phi, 0]),
            np.array([-1, -phi, 0]),
            np.array([1, -phi, 0]),
            
            np.array([0, -1, phi]),
            np.array([0, 1, phi]),
            np.array([0, -1, -phi]),
            np.array([0, 1, -phi]),
            
            np.array([phi, 0, -1]),
            np.array([phi, 0, 1]),
            np.array([-phi, 0, -1]),
            np.array([-phi, 0, 1])
        ]
        
        # Normalize to sphere
        vertices = [(v / np.linalg.norm(v) * radius, v / np.linalg.norm(v)) 
                   for v in vertices]
        
        # Initial faces
        faces = [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1)
        ]
        
        # Subdivide
        for _ in range(subdivisions):
            new_faces = []
            edge_divisions = {}
            
            for face in faces:
                # Get edge midpoints
                edges = [
                    tuple(sorted((face[0], face[1]))),
                    tuple(sorted((face[1], face[2]))),
                    tuple(sorted((face[2], face[0])))
                ]
                
                midpoints = []
                for edge in edges:
                    if edge in edge_divisions:
                        midpoints.append(edge_divisions[edge])
                    else:
                        v1 = vertices[edge[0]][0]
                        v2 = vertices[edge[1]][0]
                        midpoint = (v1 + v2) / 2
                        midpoint = midpoint / np.linalg.norm(midpoint) * radius
                        normal = midpoint / np.linalg.norm(midpoint)
                        
                        vertices.append((midpoint, normal))
                        mid_idx = len(vertices) - 1
                        edge_divisions[edge] = mid_idx
                        midpoints.append(mid_idx)
                
                # Create 4 new faces
                a, b, c = face
                d, e, f = midpoints
                
                new_faces.extend([
                    (a, d, f),
                    (d, b, e),
                    (f, e, c),
                    (d, e, f)
                ])
            
            faces = new_faces
        
        return vertices, faces
    
    def initialize_physiology(self):
        """Initialize physiological models for organ"""
        print(f"Initializing physiology for {self.organ_name}")
        
        # Temperature regulation model
        self.physiology_models['temperature'] = {
            'basal_rate': 1.0,  # W
            'perfusion_effect': 0.8,
            'metabolic_heat': 0.5  # W/kg
        }
        
        # Blood flow model
        self.physiology_models['blood_flow'] = {
            'basal_flow': 100.0,  # ml/min
            'regulation_factor': 1.0,
            'resistance': 1.0
        }
        
        # Metabolism model
        self.physiology_models['metabolism'] = {
            'basal_rate': 1.0,
            'oxygen_consumption': 0.003,  # ml O2/g/min
            'glucose_uptake': 0.01  # mmol/g/min
        }
        
        self.initialized = True
    
    def update_physiology(self, dt: float, conditions: Dict = None):
        """Update physiological state"""
        if not self.initialized:
            self.initialize_physiology()
        
        if conditions is None:
            conditions = {}
        
        # Update temperature
        ambient_temp = conditions.get('ambient_temperature', 25.0)
        blood_temp = conditions.get('blood_temperature', 37.0)
        
        # Heat transfer equation
        heat_production = self.physiology_models['temperature']['metabolic_heat'] * self.metabolic_rate
        heat_loss = 0.1 * (self.temperature - ambient_temp)  # Simplified
        
        perfusion_effect = self.physiology_models['temperature']['perfusion_effect'] * self.blood_flow / 100.0
        heat_transfer = perfusion_effect * (blood_temp - self.temperature)
        
        delta_temp = (heat_production - heat_loss + heat_transfer) * dt / 1000.0
        self.temperature += delta_temp
        
        # Update blood flow based on metabolic demand
        metabolic_demand = self.metabolic_rate * self.physiology_models['blood_flow']['basal_flow']
        flow_change = (metabolic_demand - self.blood_flow) * 0.1 * dt
        self.blood_flow += flow_change
        
        # Update oxygenation
        oxygen_supply = self.blood_flow * 0.2  # Simplified
        oxygen_consumption = self.metabolic_rate * self.physiology_models['metabolism']['oxygen_consumption']
        
        delta_oxygen = (oxygen_supply - oxygen_consumption) * dt
        self.oxygenation = max(0.7, min(1.0, self.oxygenation + delta_oxygen / 100.0))
        
        # Update vertex properties based on physiology
        for vertex in self.vertices:
            # Update blood flow at vertex (simplified perfusion model)
            vertex.blood_flow = self.blood_flow * vertex.material_id / 100.0
            
            # Update temperature gradient
            tissue_props = self.materials.get(vertex.material_id, MaterialProperty(vertex.tissue_type))
            thermal_conductivity = tissue_props.get_property('thermal_conductivity')
            
            # Simple heat conduction
            vertex.temperature = self.temperature + np.random.normal(0, 0.1)
    
    def apply_force(self, force_position: np.ndarray, force_vector: np.ndarray, 
                   radius: float = 0.1, dt: float = 0.016):
        """Apply force to organ surface"""
        if not self.deformation_enabled:
            return
        
        for i, vertex in enumerate(self.vertices):
            # Calculate distance to force application point
            distance = np.linalg.norm(vertex.position - force_position)
            
            if distance < radius:
                # Calculate force magnitude (inverse square law)
                force_magnitude = np.linalg.norm(force_vector)
                distance_factor = 1.0 - (distance / radius) ** 2
                
                # Apply force to vertex
                local_force = force_vector * distance_factor
                
                # Get material properties
                material = self.materials.get(vertex.material_id, 
                                            MaterialProperty(vertex.tissue_type))
                properties = material.properties
                
                # Apply force with physical properties
                vertex.apply_force(local_force, dt, properties)
    
    def update_deformation(self, dt: float):
        """Update deformation recovery (elastic rebound)"""
        for vertex in self.vertices:
            # Simple elastic recovery
            recovery_factor = 0.1 * dt
            
            # Gradually reduce deformation
            vertex.deformation *= (1.0 - recovery_factor)
            
            # Limit maximum deformation
            deformation_magnitude = np.linalg.norm(vertex.deformation)
            if deformation_magnitude > self.max_deformation:
                vertex.deformation = vertex.deformation / deformation_magnitude * self.max_deformation
    
    def reset_deformation(self):
        """Reset all deformation"""
        for vertex in self.vertices:
            vertex.reset_deformation()
    
    def get_visualization_data(self) -> Dict:
        """Get data for visualization"""
        visualization = {
            'vertices': [],
            'faces': [],
            'colors': [],
            'normals': [],
            'deformations': [],
            'temperatures': [],
            'blood_flows': []
        }
        
        for vertex in self.vertices:
            visualization['vertices'].append(vertex.position.tolist())
            visualization['normals'].append(vertex.normal.tolist())
            visualization['deformations'].append(vertex.deformation.tolist())
            visualization['temperatures'].append(float(vertex.temperature))
            visualization['blood_flows'].append(float(vertex.blood_flow))
            
            # Get color from material
            material = self.materials.get(vertex.material_id, 
                                        MaterialProperty(vertex.tissue_type))
            color = material.get_property('color')
            visualization['colors'].append(color)
        
        for face in self.faces:
            visualization['faces'].append(face.vertices)
        
        return visualization
    
    def get_physical_properties(self) -> Dict:
        """Get physical properties of organ"""
        total_volume = self._calculate_volume()
        total_surface_area = sum(face.area for face in self.faces)
        
        # Calculate average tissue properties
        tissue_counts = {}
        for vertex in self.vertices:
            tissue_type = vertex.tissue_type
            tissue_counts[tissue_type] = tissue_counts.get(tissue_type, 0) + 1
        
        return {
            'organ_name': self.organ_name,
            'vertex_count': len(self.vertices),
            'face_count': len(self.faces),
            'estimated_volume': total_volume,
            'surface_area': total_surface_area,
            'temperature': self.temperature,
            'blood_flow': self.blood_flow,
            'oxygenation': self.oxygenation,
            'metabolic_rate': self.metabolic_rate,
            'tissue_distribution': {t.name: c for t, c in tissue_counts.items()},
            'physiology_models': self.physiology_models
        }
    
    def _calculate_volume(self) -> float:
        """Calculate approximate volume using tetrahedron method"""
        if len(self.faces) == 0 or len(self.vertices) == 0:
            return 0.0
        
        volume = 0.0
        origin = np.array([0.0, 0.0, 0.0])
        
        for face in self.faces:
            if len(face.vertices) >= 3:
                v0 = self.vertices[face.vertices[0]].position
                v1 = self.vertices[face.vertices[1]].position
                v2 = self.vertices[face.vertices[2]].position
                
                # Tetrahedron volume (signed)
                tetra_vol = np.dot(v0, np.cross(v1, v2)) / 6.0
                volume += tetra_vol
        
        return abs(volume)

class AnatomyEngine:
    """Main engine for anatomy simulation and rendering"""
    
    def __init__(self, detail_level: str = 'high', enable_physics: bool = True):
        """
        Initialize anatomy engine
        
        Args:
            detail_level: Level of anatomical detail ('low', 'medium', 'high')
            enable_physics: Enable physics-based simulation
        """
        self.detail_level = detail_level
        self.enable_physics = enable_physics
        
        # Organ models
        self.organs: Dict[str, OrganModel] = {}
        self.active_organ = None
        
        # Skeletal system
        self.skeleton = {}
        self.joints = {}
        
        # Muscular system
        self.muscles = {}
        
        # Vascular system
        self.vascular_network = {}
        
        # Nervous system
        self.nervous_system = {}
        
        # Physiology models
        self.physiology = {
            'cardiac_output': 5.0,  # L/min
            'blood_pressure': [120, 80],  # mmHg [systolic, diastolic]
            'respiratory_rate': 12,  # breaths/min
            'body_temperature': 37.0,  # Celsius
            'metabolic_rate': 1.0  # Basal metabolic rate multiplier
        }
        
        # Rendering state
        self.view_mode = 'anatomical'  # 'anatomical', 'physiological', 'thermal', 'mechanical'
        self.show_labels = True
        self.show_grid = True
        self.transparency = 0.7
        
        # Simulation state
        self.simulation_time = 0.0
        self.time_scale = 1.0
        self.paused = False
        
        # Performance
        self.frame_count = 0
        self.fps = 0
        self.last_update = datetime.now()
        
        # Initialize OpenGL if available
        if OPENGL_AVAILABLE and self.enable_physics:
            self._initialize_opengl()
        
        print(f"Anatomy Engine initialized (Detail: {detail_level}, Physics: {enable_physics})")
    
    def _initialize_opengl(self):
        """Initialize OpenGL for rendering"""
        try:
            # Initialize display lists
            self.display_lists = {}
            
            # Set up lighting
            self._setup_lighting()
            
            print("OpenGL initialized for anatomy rendering")
        except Exception as e:
            print(f"OpenGL initialization failed: {e}")
            self.enable_physics = False
    
    def _setup_lighting(self):
        """Setup OpenGL lighting"""
        if not OPENGL_AVAILABLE:
            return
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Light properties
        light_position = [10.0, 10.0, 10.0, 1.0]
        light_ambient = [0.2, 0.2, 0.2, 1.0]
        light_diffuse = [0.8, 0.8, 0.8, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
        
        # Material properties
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)
    
    def load_anatomy_model(self, model_name: str, model_path: str = None) -> OrganModel:
        """
        Load anatomical model
        
        Args:
            model_name: Name of organ/system
            model_path: Path to model file (optional)
            
        Returns:
            Loaded organ model
        """
        print(f"Loading anatomy model: {model_name}")
        
        # Check if already loaded
        if model_name in self.organs:
            print(f"Model {model_name} already loaded")
            return self.organs[model_name]
        
        # Create organ model
        organ = OrganModel(model_name, self.detail_level)
        
        # Load from file if provided
        if model_path:
            organ.load_from_file(model_path)
        else:
            # Generate based on name
            organ._generate_fallback_model()
        
        # Initialize physiology
        organ.initialize_physiology()
        
        # Store model
        self.organs[model_name] = organ
        
        # Set as active if first organ
        if self.active_organ is None:
            self.active_organ = model_name
        
        print(f"Anatomy model loaded: {model_name} ({len(organ.vertices)} vertices)")
        return organ
    
    def load_full_body(self, gender: str = 'male', age: int = 30):
        """Load full human body model"""
        print(f"Loading full body model (Gender: {gender}, Age: {age})")
        
        # Major organs
        organs_to_load = [
            'heart', 'lungs', 'liver', 'kidneys', 'brain', 
            'stomach', 'intestines', 'pancreas', 'spleen',
            'bladder', 'reproductive_organs'
        ]
        
        # Skeletal system
        self._load_skeletal_system(gender, age)
        
        # Muscular system
        self._load_muscular_system(gender, age)
        
        # Vascular system
        self._load_vascular_system()
        
        # Nervous system
        self._load_nervous_system()
        
        # Load each organ
        for organ_name in organs_to_load:
            self.load_anatomy_model(organ_name)
        
        print(f"Full body model loaded with {len(self.organs)} organs")
    
    def _load_skeletal_system(self, gender: str, age: int):
        """Load skeletal system model"""
        print("Loading skeletal system...")
        
        self.skeleton = {
            'bones': [],
            'joints': [],
            'gender': gender,
            'age': age,
            'bone_density': self._calculate_bone_density(gender, age)
        }
        
        # Major bones (simplified)
        major_bones = [
            'skull', 'mandible', 'cervical_spine', 'thoracic_spine', 
            'lumbar_spine', 'sacrum', 'coccyx', 'clavicle', 'scapula',
            'humerus', 'radius', 'ulna', 'carpals', 'metacarpals', 'phalanges',
            'ribs', 'sternum', 'pelvis', 'femur', 'patella', 'tibia', 
            'fibula', 'tarsals', 'metatarsals', 'phalanges_feet'
        ]
        
        for bone_name in major_bones:
            bone_model = OrganModel(bone_name, self.detail_level)
            bone_model.tissue_type = TissueType.BONE
            self.skeleton['bones'].append(bone_name)
            
            # Store in organs dict for unified access
            self.organs[f"bone_{bone_name}"] = bone_model
        
        # Joint definitions
        self.joints = {
            'shoulder': {'type': 'ball_socket', 'range': {'flexion': 180, 'extension': 60}},
            'elbow': {'type': 'hinge', 'range': {'flexion': 150, 'extension': 0}},
            'wrist': {'type': 'condyloid', 'range': {'flexion': 80, 'extension': 70}},
            'hip': {'type': 'ball_socket', 'range': {'flexion': 120, 'extension': 30}},
            'knee': {'type': 'hinge', 'range': {'flexion': 135, 'extension': 0}},
            'ankle': {'type': 'hinge', 'range': {'dorsiflexion': 20, 'plantarflexion': 50}}
        }
    
    def _calculate_bone_density(self, gender: str, age: int) -> float:
        """Calculate bone density based on gender and age"""
        # Simplified model
        if gender.lower() == 'male':
            base_density = 1.2
            age_factor = max(0.7, 1.0 - (age - 30) * 0.005)
        else:
            base_density = 1.1
            age_factor = max(0.6, 1.0 - (age - 30) * 0.006)
        
        return base_density * age_factor
    
    def _load_muscular_system(self, gender: str, age: int):
        """Load muscular system model"""
        print("Loading muscular system...")
        
        self.muscles = {
            'major_groups': [],
            'muscle_fibers': {},
            'contraction_state': {},
            'strength_factor': self._calculate_muscle_strength(gender, age)
        }
        
        # Major muscle groups
        muscle_groups = [
            'deltoids', 'pectorals', 'biceps', 'triceps', 'forearms',
            'abdominals', 'obliques', 'trapezius', 'latissimus_dorsi',
            'erector_spinae', 'gluteals', 'quadriceps', 'hamstrings',
            'calves', 'soleus'
        ]
        
        for muscle_name in muscle_groups:
            muscle_model = OrganModel(muscle_name, self.detail_level)
            muscle_model.tissue_type = TissueType.MUSCLE
            self.muscles['major_groups'].append(muscle_name)
            self.muscles['contraction_state'][muscle_name] = 0.0  # 0-1 scale
            
            # Store in organs dict
            self.organs[f"muscle_{muscle_name}"] = muscle_model
    
    def _calculate_muscle_strength(self, gender: str, age: int) -> float:
        """Calculate muscle strength factor"""
        # Simplified model
        if gender.lower() == 'male':
            base_strength = 1.0
            age_factor = max(0.5, 1.0 - (age - 30) * 0.008)
        else:
            base_strength = 0.8
            age_factor = max(0.5, 1.0 - (age - 30) * 0.007)
        
        return base_strength * age_factor
    
    def _load_vascular_system(self):
        """Load vascular system model"""
        print("Loading vascular system...")
        
        self.vascular_network = {
            'arteries': [],
            'veins': [],
            'capillaries': [],
            'blood_volume': 5.0,  # liters
            'hematocrit': 0.45,
            'blood_viscosity': 3.5  # cP
        }
        
        # Major arteries
        major_arteries = [
            'aorta', 'carotid', 'subclavian', 'brachial', 'radial',
            'ulnar', 'iliac', 'femoral', 'popliteal', 'tibial'
        ]
        
        # Major veins
        major_veins = [
            'superior_vena_cava', 'inferior_vena_cava', 'jugular',
            'subclavian_vein', 'brachial_vein', 'femoral_vein'
        ]
        
        for vessel_name in major_arteries + major_veins:
            vessel_model = OrganModel(vessel_name, 'medium')
            vessel_model.tissue_type = TissueType.BLOOD_VESSEL
            
            if vessel_name in major_arteries:
                self.vascular_network['arteries'].append(vessel_name)
            else:
                self.vascular_network['veins'].append(vessel_name)
            
            self.organs[f"vessel_{vessel_name}"] = vessel_model
    
    def _load_nervous_system(self):
        """Load nervous system model"""
        print("Loading nervous system...")
        
        self.nervous_system = {
            'central': [],
            'peripheral': [],
            'autonomic': [],
            'nerve_conduction_velocity': 50.0,  # m/s
            'synaptic_delay': 0.5  # ms
        }
        
        # Central nervous system
        cns_components = ['brain', 'spinal_cord']
        
        # Peripheral nerves (major)
        pns_nerves = [
            'median_nerve', 'ulnar_nerve', 'radial_nerve',
            'sciatic_nerve', 'femoral_nerve', 'tibial_nerve'
        ]
        
        for nerve_name in cns_components + pns_nerves:
            nerve_model = OrganModel(nerve_name, 'medium')
            nerve_model.tissue_type = TissueType.NERVE
            
            if nerve_name in cns_components:
                self.nervous_system['central'].append(nerve_name)
            else:
                self.nervous_system['peripheral'].append(nerve_name)
            
            self.organs[f"nerve_{nerve_name}"] = nerve_model
    
    def set_active_organ(self, organ_name: str):
        """Set active organ for detailed viewing/manipulation"""
        if organ_name in self.organs:
            self.active_organ = organ_name
            print(f"Active organ set to: {organ_name}")
        else:
            print(f"Organ {organ_name} not found")
    
    def update(self, dt: float):
        """Update anatomy simulation"""
        if self.paused:
            return
        
        self.simulation_time += dt * self.time_scale
        self.frame_count += 1
        
        # Update physiology
        self._update_physiology(dt)
        
        # Update all organs
        for organ_name, organ in self.organs.items():
            # Update organ physiology
            conditions = {
                'ambient_temperature': 25.0,
                'blood_temperature': self.physiology['body_temperature'],
                'blood_pressure': self.physiology['blood_pressure'][0],
                'cardiac_output': self.physiology['cardiac_output']
            }
            
            organ.update_physiology(dt, conditions)
            
            # Update deformation recovery
            organ.update_deformation(dt)
        
        # Update FPS counter
        current_time = datetime.now()
        time_diff = (current_time - self.last_update).total_seconds()
        
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_update = current_time
    
    def _update_physiology(self, dt: float):
        """Update physiological parameters"""
        # Simulate basic physiological cycles
        time = self.simulation_time
        
        # Respiratory cycle
        respiratory_rate = self.physiology['respiratory_rate'] / 60.0  # Hz
        breath_phase = np.sin(2 * np.pi * respiratory_rate * time)
        
        # Cardiac cycle
        heart_rate = self.physiology['cardiac_output'] * 15.0  # Simplified
        cardiac_phase = np.sin(2 * np.pi * heart_rate * time)
        
        # Update blood pressure with cardiac cycle
        systolic = 120 + 20 * cardiac_phase
        diastolic = 80 + 10 * cardiac_phase
        self.physiology['blood_pressure'] = [max(80, systolic), max(60, diastolic)]
        
        # Update cardiac output with respiratory modulation
        self.physiology['cardiac_output'] = 5.0 + 0.5 * breath_phase
        
        # Update body temperature (homeostasis)
        target_temp = 37.0
        current_temp = self.physiology['body_temperature']
        temp_diff = target_temp - current_temp
        self.physiology['body_temperature'] += temp_diff * 0.01 * dt
    
    def apply_surgical_action(self, action_type: str, position: np.ndarray, 
                            parameters: Dict = None):
        """
        Apply surgical action to anatomy
        
        Args:
            action_type: Type of surgical action ('incision', 'cautery', 'suture', etc.)
            position: 3D position of action
            parameters: Additional parameters
        """
        if parameters is None:
            parameters = {}
        
        print(f"Applying surgical action: {action_type} at {position}")
        
        # Find affected organ(s)
        affected_organs = self._find_organs_at_position(position)
        
        for organ_name in affected_organs:
            organ = self.organs[organ_name]
            
            if action_type == 'incision':
                self._apply_incision(organ, position, parameters)
            elif action_type == 'cautery':
                self._apply_cautery(organ, position, parameters)
            elif action_type == 'suture':
                self._apply_suture(organ, position, parameters)
            elif action_type == 'clamp':
                self._apply_clamp(organ, position, parameters)
            elif action_type == 'dissection':
                self._apply_dissection(organ, position, parameters)
    
    def _find_organs_at_position(self, position: np.ndarray, radius: float = 0.05) -> List[str]:
        """Find organs at or near given position"""
        affected_organs = []
        
        for organ_name, organ in self.organs.items():
            # Simplified collision detection
            # In production, would use spatial partitioning (octree, BVH)
            for vertex in organ.vertices:
                distance = np.linalg.norm(vertex.position - position)
                if distance < radius:
                    affected_organs.append(organ_name)
                    break
        
        return affected_organs
    
    def _apply_incision(self, organ: OrganModel, position: np.ndarray, parameters: Dict):
        """Apply incision to organ"""
        depth = parameters.get('depth', 0.01)  # meters
        length = parameters.get('length', 0.02)
        width = parameters.get('width', 0.001)
        
        # Apply force along incision line
        direction = parameters.get('direction', np.array([1, 0, 0]))
        force_magnitude = parameters.get('force', 10.0)
        
        # Simulate cutting by applying forces
        for i in range(int(length / 0.001)):  # Sample points along incision
            point_pos = position + direction * i * 0.001
            force = direction * force_magnitude
            
            organ.apply_force(point_pos, force, radius=width/2)
    
    def _apply_cautery(self, organ: OrganModel, position: np.ndarray, parameters: Dict):
        """Apply cautery/thermal effect"""
        temperature = parameters.get('temperature', 100.0)  # Celsius
        duration = parameters.get('duration', 1.0)  # seconds
        radius = parameters.get('radius', 0.005)
        
        # Apply thermal damage
        for vertex in organ.vertices:
            distance = np.linalg.norm(vertex.position - position)
            if distance < radius:
                # Calculate thermal damage
                heat_transfer = (temperature - vertex.temperature) * 0.1
                vertex.temperature += heat_transfer
                
                # Thermal damage model (Arrhenius type)
                if vertex.temperature > 45.0:  # Threshold for protein denaturation
                    # Reduce blood flow (coagulation)
                    vertex.blood_flow *= 0.1
                    
                    # Change tissue properties
                    vertex.tissue_type = TissueType.SKIN  # Changed to scar-like tissue
    
    def _apply_suture(self, organ: OrganModel, position: np.ndarray, parameters: Dict):
        """Apply suture/closure"""
        # In production, this would create connection points
        # For simulation, reduce deformation in area
        radius = parameters.get('radius', 0.005)
        strength = parameters.get('strength', 1.0)
        
        for vertex in organ.vertices:
            distance = np.linalg.norm(vertex.position - position)
            if distance < radius:
                # Increase stiffness in sutured area
                vertex.deformation *= 0.5  # Reduce existing deformation
                
                # Limit future deformation
                deformation_magnitude = np.linalg.norm(vertex.deformation)
                max_deformation = organ.max_deformation * (1.0 - 0.5 * strength)
                if deformation_magnitude > max_deformation:
                    vertex.deformation = vertex.deformation / deformation_magnitude * max_deformation
    
    def _apply_clamp(self, organ: OrganModel, position: np.ndarray, parameters: Dict):
        """Apply vascular clamp"""
        pressure = parameters.get('pressure', 50.0)  # mmHg
        radius = parameters.get('radius', 0.003)
        
        # Stop blood flow in clamped area
        for vertex in organ.vertices:
            distance = np.linalg.norm(vertex.position - position)
            if distance < radius and vertex.tissue_type == TissueType.BLOOD_VESSEL:
                vertex.blood_flow = 0.0
                
                # Apply compressive force
                force_direction = np.array([0, -1, 0])  # Downward force
                force_magnitude = pressure * 133.32 * 0.0001  # Convert mmHg to Pa * area
                force = force_direction * force_magnitude
                
                vertex.apply_force(force, 0.1, {})  # dt=0.1
    
    def _apply_dissection(self, organ: OrganModel, position: np.ndarray, parameters: Dict):
        """Apply tissue dissection"""
        # Similar to incision but with tissue separation
        self._apply_incision(organ, position, parameters)
        
        # Additional tissue separation
        separation_force = parameters.get('separation_force', 5.0)
        direction = parameters.get('direction', np.array([0, 1, 0]))
        
        # Apply forces to separate tissue
        organ.apply_force(position, direction * separation_force, radius=0.01)
    
    def render(self, view_matrix=None, projection_matrix=None, 
               render_mode: str = 'opengl'):
        """
        Render anatomy
        
        Args:
            view_matrix: Camera view matrix
            projection_matrix: Camera projection matrix
            render_mode: Rendering mode ('opengl', 'software', 'wireframe')
        """
        if render_mode == 'opengl' and OPENGL_AVAILABLE:
            self._render_opengl(view_matrix, projection_matrix)
        else:
            self._render_software()
    
    def _render_opengl(self, view_matrix, projection_matrix):
        """Render using OpenGL"""
        if not OPENGL_AVAILABLE:
            return
        
        # Set up matrices
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if projection_matrix is not None:
            glMultMatrixf(projection_matrix.flatten())
        else:
            gluPerspective(45, 1.33, 0.1, 100.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        if view_matrix is not None:
            glMultMatrixf(view_matrix.flatten())
        else:
            gluLookAt(3, 3, 3, 0, 0, 0, 0, 1, 0)
        
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render each organ
        for organ_name, organ in self.organs.items():
            self._render_organ_opengl(organ)
        
        # Render grid if enabled
        if self.show_grid:
            self._render_grid()
        
        # Render labels if enabled
        if self.show_labels and self.active_organ:
            self._render_labels()
    
    def _render_organ_opengl(self, organ: OrganModel):
        """Render single organ using OpenGL"""
        if not OPENGL_AVAILABLE or len(organ.vertices) == 0:
            return
        
        # Set material properties based on view mode
        if self.view_mode == 'anatomical':
            glDisable(GL_BLEND)
            glDepthMask(GL_TRUE)
        elif self.view_mode == 'physiological':
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
        
        # Begin rendering
        glBegin(GL_TRIANGLES)
        
        for face in organ.faces:
            if len(face.vertices) >= 3:
                # Get vertices
                v0 = organ.vertices[face.vertices[0]]
                v1 = organ.vertices[face.vertices[1]]
                v2 = organ.vertices[face.vertices[2]]
                
                # Apply deformation
                pos0 = v0.position + v0.deformation
                pos1 = v1.position + v1.deformation
                pos2 = v2.position + v2.deformation
                
                # Get color based on view mode
                if self.view_mode == 'anatomical':
                    color = self._get_anatomical_color(v0.tissue_type)
                elif self.view_mode == 'thermal':
                    color = self._get_thermal_color(v0.temperature)
                elif self.view_mode == 'physiological':
                    color = self._get_physiological_color(v0.blood_flow, v0.temperature)
                else:
                    color = [0.7, 0.7, 0.7, self.transparency]
                
                # Set color
                glColor4f(*color)
                
                # Set normal
                glNormal3f(*face.normal)
                
                # Draw triangle
                glVertex3f(*pos0)
                glVertex3f(*pos1)
                glVertex3f(*pos2)
        
        glEnd()
    
    def _get_anatomical_color(self, tissue_type: TissueType) -> List[float]:
        """Get color for anatomical view"""
        colors = {
            TissueType.SKIN: [1.0, 0.9, 0.8, 1.0],
            TissueType.MUSCLE: [0.8, 0.2, 0.2, 1.0],
            TissueType.BONE: [0.96, 0.96, 0.96, 1.0],
            TissueType.FAT: [1.0, 0.9, 0.6, 1.0],
            TissueType.BLOOD_VESSEL: [0.8, 0.1, 0.1, 0.8],
            TissueType.NERVE: [1.0, 1.0, 0.0, 1.0],
            TissueType.ORGAN: [0.2, 0.8, 0.2, 1.0]
        }
        return colors.get(tissue_type, [0.7, 0.7, 0.7, 1.0])
    
    def _get_thermal_color(self, temperature: float) -> List[float]:
        """Get color for thermal view"""
        # Map temperature to color (blue = cold, red = hot)
        if temperature < 35.0:
            # Cool colors
            t = (temperature - 30.0) / 5.0
            return [0.0, 0.0, max(0.0, min(1.0, t)), 0.8]
        elif temperature > 40.0:
            # Hot colors
            t = min(1.0, (temperature - 40.0) / 20.0)
            return [1.0, max(0.0, 1.0 - t), 0.0, 0.8]
        else:
            # Normal range
            t = (temperature - 35.0) / 5.0
            return [t, 0.0, 1.0 - t, 0.8]
    
    def _get_physiological_color(self, blood_flow: float, temperature: float) -> List[float]:
        """Get color for physiological view"""
        # Combine blood flow (red) and temperature (blue)
        flow_factor = min(1.0, blood_flow / 10.0)
        temp_factor = (temperature - 30.0) / 20.0  # 30-50°C range
        
        return [
            flow_factor,  # Red = blood flow
            0.2,
            max(0.0, min(1.0, temp_factor)),  # Blue = temperature
            0.7
        ]
    
    def _render_grid(self):
        """Render 3D grid"""
        if not OPENGL_AVAILABLE:
            return
        
        glColor3f(0.3, 0.3, 0.3)
        glLineWidth(1.0)
        
        glBegin(GL_LINES)
        
        # X-axis lines
        for x in np.arange(-5, 5.1, 1):
            glVertex3f(x, 0, -5)
            glVertex3f(x, 0, 5)
        
        # Z-axis lines
        for z in np.arange(-5, 5.1, 1):
            glVertex3f(-5, 0, z)
            glVertex3f(5, 0, z)
        
        glEnd()
    
    def _render_labels(self):
        """Render organ labels"""
        if not OPENGL_AVAILABLE or not self.active_organ:
            return
        
        # Switch to 2D orthographic projection for text
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 800, 0, 600)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Disable depth test for text
        glDisable(GL_DEPTH_TEST)
        
        # Render active organ label
        glColor3f(1.0, 1.0, 1.0)
        glRasterPos2f(10, 580)
        
        # In production, use proper font rendering
        # For now, just print to console
        print(f"Active Organ: {self.active_organ}")
        
        # Restore state
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    
    def _render_software(self):
        """Software rendering fallback"""
        # This would use a software renderer like Pygame or matplotlib
        # For now, just update visualization data
        for organ in self.organs.values():
            organ.get_visualization_data()
    
    def get_diagnostics(self) -> Dict:
        """Get engine diagnostics"""
        total_vertices = sum(len(organ.vertices) for organ in self.organs.values())
        total_faces = sum(len(organ.faces) for organ in self.organs.values())
        
        return {
            'organ_count': len(self.organs),
            'total_vertices': total_vertices,
            'total_faces': total_faces,
            'active_organ': self.active_organ,
            'simulation_time': self.simulation_time,
            'fps': self.fps,
            'physiology': self.physiology,
            'memory_usage': total_vertices * 100 / 1e6,  # Approximate MB
            'render_mode': 'opengl' if OPENGL_AVAILABLE else 'software'
        }
    
    def export_model(self, organ_name: str, format: str = 'obj') -> str:
        """Export organ model to file format"""
        if organ_name not in self.organs:
            return f"Organ {organ_name} not found"
        
        organ = self.organs[organ_name]
        
        if format.lower() == 'obj':
            return self._export_to_obj(organ)
        elif format.lower() == 'stl':
            return self._export_to_stl(organ)
        elif format.lower() == 'json':
            return self._export_to_json(organ)
        else:
            return f"Format {format} not supported"
    
    def _export_to_obj(self, organ: OrganModel) -> str:
        """Export to Wavefront OBJ format"""
        obj_lines = [f"# Anatomy Engine Export: {organ.organ_name}"]
        obj_lines.append(f"# Vertices: {len(organ.vertices)}, Faces: {len(organ.faces)}")
        obj_lines.append(f"# Generated: {datetime.now()}")
        
        # Write vertices
        obj_lines.append("")
        obj_lines.append("# Vertices")
        for i, vertex in enumerate(organ.vertices):
            pos = vertex.position + vertex.deformation
            obj_lines.append(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        
        # Write vertex normals
        obj_lines.append("")
        obj_lines.append("# Normals")
        for vertex in organ.vertices:
            obj_lines.append(f"vn {vertex.normal[0]:.6f} {vertex.normal[1]:.6f} {vertex.normal[2]:.6f}")
        
        # Write faces
        obj_lines.append("")
        obj_lines.append("# Faces")
        for face in organ.faces:
            if len(face.vertices) >= 3:
                # OBJ uses 1-based indexing
                v1 = face.vertices[0] + 1
                v2 = face.vertices[1] + 1
                v3 = face.vertices[2] + 1
                obj_lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")
        
        return "\n".join(obj_lines)
    
    def _export_to_stl(self, organ: OrganModel) -> str:
        """Export to STL format (ASCII)"""
        stl_lines = [f"solid {organ.organ_name}"]
        
        for face in organ.faces:
            if len(face.vertices) >= 3:
                v0 = organ.vertices[face.vertices[0]]
                v1 = organ.vertices[face.vertices[1]]
                v2 = organ.vertices[face.vertices[2]]
                
                pos0 = v0.position + v0.deformation
                pos1 = v1.position + v1.deformation
                pos2 = v2.position + v2.deformation
                
                stl_lines.append("  facet normal {} {} {}".format(
                    face.normal[0], face.normal[1], face.normal[2]
                ))
                stl_lines.append("    outer loop")
                stl_lines.append("      vertex {} {} {}".format(pos0[0], pos0[1], pos0[2]))
                stl_lines.append("      vertex {} {} {}".format(pos1[0], pos1[1], pos1[2]))
                stl_lines.append("      vertex {} {} {}".format(pos2[0], pos2[1], pos2[2]))
                stl_lines.append("    endloop")
                stl_lines.append("  endfacet")
        
        stl_lines.append(f"endsolid {organ.organ_name}")
        return "\n".join(stl_lines)
    
    def _export_to_json(self, organ: OrganModel) -> str:
        """Export to JSON format"""
        import json
        
        data = {
            'organ_name': organ.organ_name,
            'vertices': [],
            'faces': [],
            'tissue_types': [],
            'physiology': {
                'temperature': organ.temperature,
                'blood_flow': organ.blood_flow,
                'oxygenation': organ.oxygenation
            }
        }
        
        for vertex in organ.vertices:
            pos = (vertex.position + vertex.deformation).tolist()
            data['vertices'].append(pos)
            data['tissue_types'].append(vertex.tissue_type.value)
        
        for face in organ.faces:
            data['faces'].append(face.vertices)
        
        return json.dumps(data, indent=2)
    
    def set_view_mode(self, mode: str):
        """Set view mode"""
        valid_modes = ['anatomical', 'physiological', 'thermal', 'mechanical', 'wireframe']
        if mode in valid_modes:
            self.view_mode = mode
            print(f"View mode set to: {mode}")
        else:
            print(f"Invalid view mode: {mode}. Valid modes: {valid_modes}")
    
    def toggle_pause(self):
        """Toggle simulation pause"""
        self.paused = not self.paused
        print(f"Simulation {'paused' if self.paused else 'resumed'}")
    
    def reset_simulation(self):
        """Reset simulation state"""
        self.simulation_time = 0.0
        
        for organ in self.organs.values():
            organ.reset_deformation()
            organ.temperature = 37.0
            organ.blood_flow = 0.0
        
        print("Simulation reset")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up Anatomy Engine...")
        
        # Clear OpenGL resources
        if OPENGL_AVAILABLE:
            glDeleteLists(list(self.display_lists.values()), 1)
        
        # Clear organ models
        self.organs.clear()
        
        # Reset state
        self.active_organ = None
        self.skeleton.clear()
        self.muscles.clear()
        self.vascular_network.clear()
        self.nervous_system.clear()

# Utility functions for external use

def create_organ_primitive(organ_type: str, size: float = 1.0, 
                          detail: int = 2) -> OrganModel:
    """Create primitive organ model"""
    primitives = {
        'heart': _create_heart_model,
        'lung': _create_lung_model,
        'liver': _create_liver_model,
        'kidney': _create_kidney_model,
        'brain': _create_brain_model,
        'stomach': _create_stomach_model
    }
    
    if organ_type in primitives:
        return primitives[organ_type](size, detail)
    else:
        # Default sphere
        organ = OrganModel(organ_type, 'medium')
        organ._generate_fallback_model()
        return organ

def _create_heart_model(size: float, detail: int) -> OrganModel:
    """Create simplified heart model"""
    heart = OrganModel('heart', 'high')
    
    # Create base shape (simplified)
    vertices, faces = heart._create_icosphere(detail, size)
    
    for i, (pos, normal) in enumerate(vertices):
        # Modify shape to be more heart-like
        if i % 2 == 0:
            pos = pos * 1.2  # Make some parts larger
        
        vertex = AnatomicalVertex(
            position=pos,
            normal=normal,
            tissue_type=TissueType.MUSCLE,
            material_id=0
        )
        heart.vertices.append(vertex)
    
    for face_indices in faces:
        face = AnatomicalFace(
            vertices=list(face_indices),
            normal=np.array([0, 0, 1]),
            area=0.1
        )
        heart.faces.append(face)
    
    # Calculate normals and areas
    for face in heart.faces:
        face.normal = face.calculate_normal(heart.vertices)
        face.area = face.calculate_area(heart.vertices)
    
    # Set heart-specific properties
    heart.temperature = 37.5  # Slightly warmer
    heart.blood_flow = 250.0  # High blood flow
    heart.oxygenation = 0.95
    
    return heart

def _create_lung_model(size: float, detail: int) -> OrganModel:
    """Create simplified lung model"""
    lung = OrganModel('lung', 'medium')
    
    # Create elongated sphere
    vertices, faces = lung._create_icosphere(detail, size)
    
    for pos, normal in vertices:
        # Elongate in one direction
        pos = np.array([pos[0] * 0.8, pos[1] * 1.5, pos[2] * 0.8])
        
        vertex = AnatomicalVertex(
            position=pos,
            normal=normal,
            tissue_type=TissueType.ORGAN,
            material_id=0
        )
        lung.vertices.append(vertex)
    
    for face_indices in faces:
        face = AnatomicalFace(
            vertices=list(face_indices),
            normal=np.array([0, 0, 1]),
            area=0.1
        )
        lung.faces.append(face)
    
    # Calculate normals and areas
    for face in lung.faces:
        face.normal = face.calculate_normal(lung.vertices)
        face.area = face.calculate_area(lung.vertices)
    
    # Set lung-specific properties
    lung.temperature = 37.0
    lung.blood_flow = 100.0
    lung.oxygenation = 0.98
    
    return lung

def _create_liver_model(size: float, detail: int) -> OrganModel:
    """Create simplified liver model"""
    liver = OrganModel('liver', 'medium')
    
    # Create irregular shape
    vertices, faces = liver._create_icosphere(detail, size)
    
    for pos, normal in vertices:
        # Make shape irregular
        distortion = np.random.normal(1.0, 0.1, 3)
        pos = pos * distortion
        
        vertex = AnatomicalVertex(
            position=pos,
            normal=normal,
            tissue_type=TissueType.ORGAN,
            material_id=0
        )
        liver.vertices.append(vertex)
    
    for face_indices in faces:
        face = AnatomicalFace(
            vertices=list(face_indices),
            normal=np.array([0, 0, 1]),
            area=0.1
        )
        liver.faces.append(face)
    
    # Calculate normals and areas
    for face in liver.faces:
        face.normal = face.calculate_normal(liver.vertices)
        face.area = face.calculate_area(liver.vertices)
    
    # Set liver-specific properties
    liver.temperature = 37.2
    liver.blood_flow = 150.0
    liver.oxygenation = 0.85  # Lower due to metabolic activity
    
    return liver

def _create_kidney_model(size: float, detail: int) -> OrganModel:
    """Create simplified kidney model"""
    kidney = OrganModel('kidney', 'medium')
    
    # Create bean-shaped object
    vertices, faces = kidney._create_icosphere(detail, size)
    
    for pos, normal in vertices:
        # Shape into kidney bean form
        # Flatten on one side, indent on opposite
        if pos[0] > 0:
            pos = pos * np.array([0.7, 1.0, 1.0])
        else:
            pos = pos * np.array([1.2, 1.0, 1.0])
            # Add indentation
            if abs(pos[1]) < 0.3:
                pos[0] *= 0.8
        
        vertex = AnatomicalVertex(
            position=pos,
            normal=normal,
            tissue_type=TissueType.ORGAN,
            material_id=0
        )
        kidney.vertices.append(vertex)
    
    for face_indices in faces:
        face = AnatomicalFace(
            vertices=list(face_indices),
            normal=np.array([0, 0, 1]),
            area=0.1
        )
        kidney.faces.append(face)
    
    # Calculate normals and areas
    for face in kidney.faces:
        face.normal = face.calculate_normal(kidney.vertices)
        face.area = face.calculate_area(kidney.vertices)
    
    # Set kidney-specific properties
    kidney.temperature = 37.0
    kidney.blood_flow = 120.0
    kidney.oxygenation = 0.90
    
    return kidney

def _create_brain_model(size: float, detail: int) -> OrganModel:
    """Create simplified brain model"""
    brain = OrganModel('brain', 'high')
    
    # Create folded structure
    vertices, faces = brain._create_icosphere(detail + 1, size)
    
    for pos, normal in vertices:
        # Add cortical folding
        radius = np.linalg.norm(pos)
        if radius > 0:
            # Create folds using sine waves
            theta = np.arctan2(pos[1], pos[0])
            phi = np.arccos(pos[2] / radius)
            
            fold_depth = 0.1 * np.sin(8 * theta) * np.sin(6 * phi)
            pos = pos * (1.0 + fold_depth)
        
        vertex = AnatomicalVertex(
            position=pos,
            normal=normal,
            tissue_type=TissueType.ORGAN,
            material_id=0
        )
        brain.vertices.append(vertex)
    
    for face_indices in faces:
        face = AnatomicalFace(
            vertices=list(face_indices),
            normal=np.array([0, 0, 1]),
            area=0.05  # Smaller faces for detail
        )
        brain.faces.append(face)
    
    # Calculate normals and areas
    for face in brain.faces:
        face.normal = face.calculate_normal(brain.vertices)
        face.area = face.calculate_area(brain.vertices)
    
    # Set brain-specific properties
    brain.temperature = 37.0
    brain.blood_flow = 75.0  # High per weight
    brain.oxygenation = 0.99  # Very high
    
    return brain

def _create_stomach_model(size: float, detail: int) -> OrganModel:
    """Create simplified stomach model"""
    stomach = OrganModel('stomach', 'medium')
    
    # Create bag-like shape
    vertices, faces = stomach._create_icosphere(detail, size)
    
    for pos, normal in vertices:
        # Stretch and distort for stomach shape
        stretch = np.array([1.5, 1.0, 0.8])  # Longer in x, flatter in z
        pos = pos * stretch
        
        # Add irregularity
        if pos[0] > 0:  # Fundus region
            pos[1] *= 1.2
        
        vertex = AnatomicalVertex(
            position=pos,
            normal=normal,
            tissue_type=TissueType.MUSCLE,  # Stomach has muscular walls
            material_id=0
        )
        stomach.vertices.append(vertex)
    
    for face_indices in faces:
        face = AnatomicalFace(
            vertices=list(face_indices),
            normal=np.array([0, 0, 1]),
            area=0.1
        )
        stomach.faces.append(face)
    
    # Calculate normals and areas
    for face in stomach.faces:
        face.normal = face.calculate_normal(stomach.vertices)
        face.area = face.calculate_area(stomach.vertices)
    
    # Set stomach-specific properties
    stomach.temperature = 37.0
    stomach.blood_flow = 50.0
    stomach.oxygenation = 0.92
    
    return stomach

# Example usage
def run_anatomy_demo():
    """Run anatomy engine demonstration"""
    print("Running Anatomy Engine Demo...")
    
    # Initialize engine
    engine = AnatomyEngine(detail_level='medium', enable_physics=True)
    
    # Load some organs
    heart = engine.load_anatomy_model('heart')
    liver = engine.load_anatomy_model('liver')
    brain = engine.load_anatomy_model('brain')
    
    # Set active organ
    engine.set_active_organ('heart')
    
    # Apply some surgical actions
    print("\nApplying surgical actions...")
    
    # Make an incision
    engine.apply_surgical_action(
        'incision',
        position=np.array([0.1, 0.0, 0.0]),
        parameters={'depth': 0.02, 'length': 0.03, 'force': 15.0}
    )
    
    # Apply cautery
    engine.apply_surgical_action(
        'cautery',
        position=np.array([-0.1, 0.0, 0.0]),
        parameters={'temperature': 80.0, 'duration': 2.0}
    )
    
    # Update simulation
    print("\nUpdating simulation...")
    for i in range(10):
        engine.update(0.1)  # 100ms time step
    
    # Get diagnostics
    diagnostics = engine.get_diagnostics()
    print(f"\nEngine Diagnostics:")
    print(f"  Organ count: {diagnostics['organ_count']}")
    print(f"  Total vertices: {diagnostics['total_vertices']}")
    print(f"  Simulation time: {diagnostics['simulation_time']:.1f}s")
    print(f"  FPS: {diagnostics['fps']:.1f}")
    
    # Get organ properties
    heart_props = heart.get_physical_properties()
    print(f"\nHeart Properties:")
    print(f"  Temperature: {heart_props['temperature']:.1f}°C")
    print(f"  Blood flow: {heart_props['blood_flow']:.1f} ml/s")
    print(f"  Oxygenation: {heart_props['oxygenation']:.2f}")
    
    # Export model
    print("\nExporting heart model...")
    obj_data = engine.export_model('heart', 'obj')
    print(f"Exported {len(obj_data.splitlines())} lines of OBJ data")
    
    # Cleanup
    engine.cleanup()
    print("\nDemo completed!")

if __name__ == "__main__":
    run_anatomy_demo()