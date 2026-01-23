"""
Surgical Training Module
VR/AR integration for practicing surgical procedures
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Pygame not available for VR simulation")

try:
    import openvr
    OPENVR_AVAILABLE = True
except ImportError:
    OPENVR_AVAILABLE = False
    print("OpenVR not available, using simulated VR")

class SurgicalInstrument(Enum):
    """Types of surgical instruments"""
    SCALPEL = "scalpel"
    FORCEPS = "forceps"
    SCISSORS = "scissors"
    RETRACTOR = "retractor"
    CLAMP = "clamp"
    SUTURE_NEEDLE = "suture_needle"
    ELECTROCAUTERY = "electrocautery"
    LAPAROSCOPE = "laparoscope"
    SUCTION = "suction"
    DRILL = "drill"

class SurgicalStep(Enum):
    """Steps in surgical procedures"""
    PREPARATION = "preparation"
    INCISION = "incision"
    DISSECTION = "dissection"
    HEMOSTASIS = "hemostasis"
    REPAIR = "repair"
    CLOSURE = "closure"
    DRESSING = "dressing"

@dataclass
class InstrumentState:
    """State of surgical instrument"""
    instrument_type: SurgicalInstrument
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # quaternion [x, y, z, w]
    velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [ωx, ωy, ωz]
    grip_strength: float = 0.0  # 0-1
    is_active: bool = False
    temperature: float = 20.0  # Celsius
    force_feedback: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def update(self, dt: float, new_position: np.ndarray, new_rotation: np.ndarray):
        """Update instrument state"""
        # Calculate velocity
        position_diff = new_position - self.position
        self.velocity = position_diff / dt if dt > 0 else np.zeros(3)
        
        # Update position and rotation
        self.position = new_position.copy()
        self.rotation = new_rotation.copy()

@dataclass
class SurgicalAction:
    """Recorded surgical action"""
    step_type: SurgicalStep
    instrument: SurgicalInstrument
    position: np.ndarray
    orientation: np.ndarray
    force_applied: float
    time_stamp: float
    duration: float
    success_score: float = 0.0
    error_type: str = None
    error_severity: float = 0.0

class HapticFeedback:
    """Haptic feedback system for surgical simulation"""
    
    def __init__(self, enable_haptics: bool = True):
        self.enable_haptics = enable_haptics
        self.devices = {}
        self.feedback_patterns = self._initialize_patterns()
        
        if enable_haptics:
            self._initialize_haptic_devices()
    
    def _initialize_patterns(self):
        """Initialize haptic feedback patterns"""
        return {
            'tissue_cutting': {
                'frequency': 100,  # Hz
                'amplitude': 0.7,
                'duration': 0.1,
                'pattern': 'vibration'
            },
            'bone_contact': {
                'frequency': 200,
                'amplitude': 0.9,
                'duration': 0.05,
                'pattern': 'sharp_pulse'
            },
            'vessel_pulsation': {
                'frequency': 2,  # Hz (heart rate)
                'amplitude': 0.3,
                'duration': 0.5,
                'pattern': 'rhythmic'
            },
            'instrument_slip': {
                'frequency': 50,
                'amplitude': 0.5,
                'duration': 0.2,
                'pattern': 'sliding'
            },
            'error_warning': {
                'frequency': 10,
                'amplitude': 1.0,
                'duration': 0.3,
                'pattern': 'repeating_pulse'
            }
        }
    
    def _initialize_haptic_devices(self):
        """Initialize haptic feedback devices"""
        # In production, this would interface with actual haptic devices
        # For simulation, create virtual devices
        self.devices = {
            'right_hand': {'connected': True, 'vibration_capable': True},
            'left_hand': {'connected': True, 'vibration_capable': True},
            'foot_pedal': {'connected': False, 'vibration_capable': False}
        }
    
    def provide_feedback(self, feedback_type: str, intensity: float = 1.0, 
                        device: str = 'right_hand'):
        """Provide haptic feedback"""
        if not self.enable_haptics or device not in self.devices:
            return
        
        pattern = self.feedback_patterns.get(feedback_type)
        if not pattern:
            return
        
        # Scale intensity
        amplitude = pattern['amplitude'] * intensity
        duration = pattern['duration']
        
        # In production, send command to haptic device
        # For simulation, just log
        print(f"Haptic feedback: {feedback_type} (intensity: {intensity:.2f})")
    
    def update_force_feedback(self, force_vector: np.ndarray, 
                            device: str = 'right_hand'):
        """Update force feedback (resistance)"""
        if not self.enable_haptics:
            return
        
        # Calculate force magnitude and direction
        force_magnitude = np.linalg.norm(force_vector)
        if force_magnitude > 0:
            direction = force_vector / force_magnitude
            
            # Apply force feedback based on magnitude
            # In production, this would control force feedback devices
            print(f"Force feedback: {force_magnitude:.2f}N in direction {direction}")

class VRController:
    """Virtual Reality controller interface"""
    
    def __init__(self, vr_enabled: bool = False):
        self.vr_enabled = vr_enabled and OPENVR_AVAILABLE
        self.controllers = {}
        self.hmd_position = np.zeros(3)
        self.hmd_orientation = np.array([0, 0, 0, 1])  # quaternion
        
        if self.vr_enabled:
            self._initialize_openvr()
        else:
            self._initialize_simulated_controllers()
    
    def _initialize_openvr(self):
        """Initialize OpenVR for actual VR headsets"""
        try:
            self.vr_system = openvr.init(openvr.VRApplication_Scene)
            
            # Get controller indices
            for i in range(openvr.k_unMaxTrackedDeviceCount):
                device_class = self.vr_system.getTrackedDeviceClass(i)
                if device_class == openvr.TrackedDeviceClass_Controller:
                    role = self.vr_system.getControllerRoleForTrackedDeviceIndex(i)
                    if role == openvr.TrackedControllerRole_RightHand:
                        self.controllers['right'] = i
                    elif role == openvr.TrackedControllerRole_LeftHand:
                        self.controllers['left'] = i
            
            print(f"OpenVR initialized with {len(self.controllers)} controllers")
            
        except Exception as e:
            print(f"OpenVR initialization failed: {e}")
            self.vr_enabled = False
            self._initialize_simulated_controllers()
    
    def _initialize_simulated_controllers(self):
        """Initialize simulated controllers for non-VR mode"""
        print("Using simulated VR controllers")
        self.controllers = {
            'right': 'simulated_right',
            'left': 'simulated_left'
        }
    
    def get_controller_pose(self, controller: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get controller position and orientation"""
        if not self.vr_enabled or controller not in self.controllers:
            # Return simulated data
            if controller == 'right':
                position = np.array([0.3, -0.5, -0.2])
                orientation = np.array([0, 0, 0, 1])
            else:  # left
                position = np.array([-0.3, -0.5, -0.2])
                orientation = np.array([0, 0, 0, 1])
            
            # Add slight movement for simulation
            import time
            t = time.time()
            position[1] += 0.1 * np.sin(t)
            
            return position, orientation
        
        # Get actual VR controller pose
        try:
            controller_index = self.controllers[controller]
            poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
            
            pose = poses[controller_index]
            if pose.bPoseIsValid:
                matrix = pose.mDeviceToAbsoluteTracking
                
                # Extract position
                position = np.array([matrix[0][3], matrix[1][3], matrix[2][3]])
                
                # Extract orientation (matrix to quaternion)
                orientation = self._matrix_to_quaternion(matrix)
                
                return position, orientation
            
        except Exception as e:
            print(f"Error getting controller pose: {e}")
        
        return np.zeros(3), np.array([0, 0, 0, 1])
    
    def _matrix_to_quaternion(self, matrix) -> np.ndarray:
        """Convert 3x4 matrix to quaternion"""
        # Simplified conversion
        return np.array([0, 0, 0, 1])  # Identity quaternion
    
    def get_button_state(self, controller: str, button: str) -> Dict:
        """Get button state"""
        # Simulated button states
        states = {
            'trigger': {'pressed': False, 'value': 0.0},
            'grip': {'pressed': False, 'value': 0.0},
            'trackpad': {'pressed': False, 'x': 0.0, 'y': 0.0},
            'menu': {'pressed': False}
        }
        
        # In production, would get actual button states from VR system
        return states.get(button, {})
    
    def update(self):
        """Update VR system"""
        if self.vr_enabled:
            # Process VR events
            event = openvr.VREvent_t()
            while self.vr_system.pollNextEvent(event):
                self._process_vr_event(event)
    
    def _process_vr_event(self, event):
        """Process VR event"""
        # Handle VR events
        pass
    
    def cleanup(self):
        """Clean up VR resources"""
        if self.vr_enabled:
            openvr.shutdown()

class SurgicalProcedure:
    """Definition of a surgical procedure"""
    
    def __init__(self, procedure_name: str, difficulty: str = 'medium'):
        self.procedure_name = procedure_name
        self.difficulty = difficulty
        self.steps = []
        self.instruments_required = []
        self.time_estimate = 0  # minutes
        self.success_criteria = {}
        self.common_errors = []
        
        self._load_procedure_definition()
    
    def _load_procedure_definition(self):
        """Load procedure definition"""
        # Procedure database
        procedures = {
            'appendectomy': {
                'difficulty': 'medium',
                'steps': [
                    {'type': SurgicalStep.PREPARATION, 'duration': 5, 'critical': False},
                    {'type': SurgicalStep.INCISION, 'duration': 3, 'critical': True},
                    {'type': SurgicalStep.DISSECTION, 'duration': 10, 'critical': True},
                    {'type': SurgicalStep.HEMOSTASIS, 'duration': 5, 'critical': True},
                    {'type': SurgicalStep.REPAIR, 'duration': 15, 'critical': True},
                    {'type': SurgicalStep.CLOSURE, 'duration': 10, 'critical': False},
                    {'type': SurgicalStep.DRESSING, 'duration': 2, 'critical': False}
                ],
                'instruments': [
                    SurgicalInstrument.SCALPEL,
                    SurgicalInstrument.FORCEPS,
                    SurgicalInstrument.SCISSORS,
                    SurgicalInstrument.RETRACTOR,
                    SurgicalInstrument.CLAMP,
                    SurgicalInstrument.SUTURE_NEEDLE
                ],
                'time_estimate': 60,
                'success_criteria': {
                    'completion_time': 90,  # max minutes
                    'blood_loss': 100,  # max ml
                    'instrument_errors': 5,  # max count
                    'critical_errors': 0  # max count
                },
                'common_errors': [
                    'incision_too_large',
                    'vascular_injury',
                    'bowel_perforation',
                    'inadequate_hemostasis'
                ]
            },
            'cataract_surgery': {
                'difficulty': 'high',
                'steps': [
                    {'type': SurgicalStep.PREPARATION, 'duration': 10, 'critical': False},
                    {'type': SurgicalStep.INCISION, 'duration': 5, 'critical': True},
                    {'type': SurgicalStep.DISSECTION, 'duration': 20, 'critical': True},
                    {'type': SurgicalStep.REPAIR, 'duration': 25, 'critical': True},
                    {'type': SurgicalStep.CLOSURE, 'duration': 5, 'critical': False}
                ],
                'instruments': [
                    SurgicalInstrument.SCALPEL,
                    SurgicalInstrument.FORCEPS,
                    SurgicalInstrument.SUCTION,
                    SurgicalInstrument.DRILL
                ],
                'time_estimate': 65,
                'success_criteria': {
                    'completion_time': 90,
                    'tissue_damage': 0.1,  # max area (mm²)
                    'lens_position_error': 0.5,  # max mm
                    'visual_outcome': 0.8  # min score
                },
                'common_errors': [
                    'capsular_rupture',
                    'vitreous_loss',
                    'incorrect_lens_power',
                    'corneal_damage'
                ]
            },
            'knee_arthroscopy': {
                'difficulty': 'medium',
                'steps': [
                    {'type': SurgicalStep.PREPARATION, 'duration': 15, 'critical': False},
                    {'type': SurgicalStep.INCISION, 'duration': 5, 'critical': True},
                    {'type': SurgicalStep.DISSECTION, 'duration': 30, 'critical': True},
                    {'type': SurgicalStep.REPAIR, 'duration': 40, 'critical': True},
                    {'type': SurgicalStep.CLOSURE, 'duration': 10, 'critical': False}
                ],
                'instruments': [
                    SurgicalInstrument.SCALPEL,
                    SurgicalInstrument.FORCEPS,
                    SurgicalInstrument.SCISSORS,
                    SurgicalInstrument.LAPAROSCOPE,
                    SurgicalInstrument.DRILL
                ],
                'time_estimate': 100,
                'success_criteria': {
                    'completion_time': 120,
                    'cartilage_damage': 5.0,  # max area (mm²)
                    'range_of_motion': 120,  # min degrees
                    'pain_score': 3.0  # max (0-10)
                },
                'common_errors': [
                    'meniscal_damage',
                    'ligament_injury',
                    'infection_risk',
                    'nerve_damage'
                ]
            }
        }
        
        if self.procedure_name in procedures:
            proc_data = procedures[self.procedure_name]
            
            self.difficulty = proc_data['difficulty']
            self.steps = proc_data['steps']
            self.instruments_required = proc_data['instruments']
            self.time_estimate = proc_data['time_estimate']
            self.success_criteria = proc_data['success_criteria']
            self.common_errors = proc_data['common_errors']
        else:
            # Default procedure
            self.steps = [
                {'type': SurgicalStep.PREPARATION, 'duration': 5, 'critical': False},
                {'type': SurgicalStep.INCISION, 'duration': 2, 'critical': True},
                {'type': SurgicalStep.CLOSURE, 'duration': 3, 'critical': False}
            ]
            self.instruments_required = [SurgicalInstrument.SCALPEL, SurgicalInstrument.SUTURE_NEEDLE]
            self.time_estimate = 10
            self.success_criteria = {'completion_time': 15, 'critical_errors': 0}
            self.common_errors = ['incision_error', 'suture_error']
    
    def get_current_step(self, elapsed_time: float) -> Dict:
        """Get current step based on elapsed time"""
        cumulative_time = 0
        for step in self.steps:
            cumulative_time += step['duration']
            if elapsed_time <= cumulative_time:
                return step
        
        # Return last step if time exceeded
        return self.steps[-1] if self.steps else None
    
    def get_step_progress(self, elapsed_time: float) -> Tuple[int, float]:
        """Get current step index and progress"""
        cumulative_time = 0
        for i, step in enumerate(self.steps):
            step_duration = step['duration']
            if elapsed_time <= cumulative_time + step_duration:
                step_progress = (elapsed_time - cumulative_time) / step_duration
                return i, max(0, min(1, step_progress))
            cumulative_time += step_duration
        
        return len(self.steps) - 1, 1.0
    
    def validate_action(self, step_index: int, action: SurgicalAction) -> Dict:
        """Validate surgical action for current step"""
        if step_index >= len(self.steps):
            return {'valid': False, 'error': 'Step index out of range'}
        
        current_step = self.steps[step_index]
        validation = {
            'valid': True,
            'step_type': current_step['type'].value,
            'score': 0.0,
            'errors': [],
            'warnings': []
        }
        
        # Check if instrument is appropriate for step
        appropriate_instruments = {
            SurgicalStep.INCISION: [SurgicalInstrument.SCALPEL],
            SurgicalStep.DISSECTION: [SurgicalInstrument.SCALPEL, SurgicalInstrument.SCISSORS, 
                                     SurgicalInstrument.FORCEPS],
            SurgicalStep.HEMOSTASIS: [SurgicalInstrument.ELECTROCAUTERY, SurgicalInstrument.CLAMP],
            SurgicalStep.REPAIR: [SurgicalInstrument.SUTURE_NEEDLE, SurgicalInstrument.FORCEPS],
            SurgicalStep.CLOSURE: [SurgicalInstrument.SUTURE_NEEDLE]
        }
        
        step_type = current_step['type']
        if step_type in appropriate_instruments:
            if action.instrument not in appropriate_instruments[step_type]:
                validation['warnings'].append(f"Instrument {action.instrument.value} "
                                            f"not ideal for {step_type.value}")
        
        # Score the action
        validation['score'] = self._score_action(action, current_step)
        
        return validation
    
    def _score_action(self, action: SurgicalAction, step: Dict) -> float:
        """Score surgical action"""
        score = 1.0
        
        # Deduct for errors
        if action.error_type:
            score -= action.error_severity
        
        # Adjust based on force
        ideal_force_ranges = {
            SurgicalInstrument.SCALPEL: (0.5, 2.0),  # Newtons
            SurgicalInstrument.SUTURE_NEEDLE: (0.2, 1.0),
            SurgicalInstrument.ELECTROCAUTERY: (0.1, 0.5)
        }
        
        if action.instrument in ideal_force_ranges:
            min_f, max_f = ideal_force_ranges[action.instrument]
            if action.force_applied < min_f:
                score -= 0.2
            elif action.force_applied > max_f:
                score -= 0.3
        
        # Critical step penalty
        if step.get('critical', False) and action.error_type:
            score -= 0.5
        
        return max(0.0, min(1.0, score))

class SurgicalSimulator:
    """Main surgical simulation engine"""
    
    def __init__(self, vr_enabled: bool = False, haptics_enabled: bool = True,
                difficulty: str = 'medium'):
        """
        Initialize surgical simulator
        
        Args:
            vr_enabled: Enable VR mode
            haptics_enabled: Enable haptic feedback
            difficulty: Simulation difficulty
        """
        self.vr_enabled = vr_enabled
        self.haptics_enabled = haptics_enabled
        self.difficulty = difficulty
        
        # VR and haptics
        self.vr_controller = VRController(vr_enabled)
        self.haptic_feedback = HapticFeedback(haptics_enabled)
        
        # Surgical state
        self.current_procedure = None
        self.current_instruments = {}
        self.surgical_actions = []
        self.patient_anatomy = None
        
        # Simulation state
        self.simulation_time = 0.0
        self.is_running = False
        self.is_paused = False
        self.current_step_index = 0
        self.step_start_time = 0.0
        
        # Performance metrics
        self.performance_metrics = {
            'total_errors': 0,
            'critical_errors': 0,
            'blood_loss': 0.0,  # ml
            'procedure_time': 0.0,
            'instrument_changes': 0,
            'efficiency_score': 0.0
        }
        
        # Visualization
        self.visualization_data = {}
        
        # Initialize instrument states
        self._initialize_instruments()
        
        print(f"Surgical Simulator initialized (VR: {vr_enabled}, Haptics: {haptics_enabled})")
    
    def _initialize_instruments(self):
        """Initialize surgical instruments"""
        for instrument in SurgicalInstrument:
            self.current_instruments[instrument] = InstrumentState(
                instrument_type=instrument,
                position=np.zeros(3),
                rotation=np.array([0, 0, 0, 1]),
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3)
            )
    
    def load_procedure(self, procedure_name: str, patient_anatomy = None):
        """Load surgical procedure"""
        print(f"Loading procedure: {procedure_name}")
        
        self.current_procedure = SurgicalProcedure(procedure_name, self.difficulty)
        self.patient_anatomy = patient_anatomy
        
        # Reset simulation state
        self.simulation_time = 0.0
        self.current_step_index = 0
        self.step_start_time = 0.0
        self.surgical_actions.clear()
        self._reset_metrics()
        
        print(f"Procedure loaded: {procedure_name} "
              f"({len(self.current_procedure.steps)} steps, "
              f"{self.current_procedure.time_estimate} min estimated)")
    
    def _reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = {
            'total_errors': 0,
            'critical_errors': 0,
            'blood_loss': 0.0,
            'procedure_time': 0.0,
            'instrument_changes': 0,
            'efficiency_score': 0.0
        }
    
    def start_simulation(self):
        """Start surgical simulation"""
        if not self.current_procedure:
            print("No procedure loaded. Call load_procedure() first.")
            return False
        
        self.is_running = True
        self.is_paused = False
        self.simulation_time = 0.0
        self.current_step_index = 0
        self.step_start_time = 0.0
        
        print(f"Surgical simulation started: {self.current_procedure.procedure_name}")
        return True
    
    def pause_simulation(self):
        """Pause simulation"""
        self.is_paused = not self.is_paused
        print(f"Simulation {'paused' if self.is_paused else 'resumed'}")
    
    def stop_simulation(self):
        """Stop simulation"""
        self.is_running = False
        self.calculate_final_score()
        print("Surgical simulation stopped")
    
    def update(self, dt: float):
        """Update simulation"""
        if not self.is_running or self.is_paused:
            return
        
        # Update time
        self.simulation_time += dt
        self.performance_metrics['procedure_time'] = self.simulation_time
        
        # Update VR controllers
        self.vr_controller.update()
        
        # Update current step
        self._update_current_step()
        
        # Update instrument states from VR controllers
        self._update_instrument_states(dt)
        
        # Check for collisions and interactions
        self._check_interactions(dt)
        
        # Update performance metrics
        self._update_metrics(dt)
    
    def _update_current_step(self):
        """Update current surgical step"""
        step_index, step_progress = self.current_procedure.get_step_progress(self.simulation_time)
        
        if step_index != self.current_step_index:
            # Step changed
            self.current_step_index = step_index
            self.step_start_time = self.simulation_time
            
            current_step = self.current_procedure.steps[step_index]
            print(f"Step changed: {current_step['type'].value} "
                  f"(Duration: {current_step['duration']} min)")
    
    def _update_instrument_states(self, dt: float):
        """Update instrument states from VR controllers"""
        # Get controller poses
        right_pos, right_rot = self.vr_controller.get_controller_pose('right')
        left_pos, left_rot = self.vr_controller.get_controller_pose('left')
        
        # Get button states to determine active instrument
        right_trigger = self.vr_controller.get_button_state('right', 'trigger')
        left_trigger = self.vr_controller.get_button_state('left', 'trigger')
        
        # Update primary instrument (right hand)
        primary_instrument = self._get_active_instrument()
        if primary_instrument in self.current_instruments:
            instrument_state = self.current_instruments[primary_instrument]
            instrument_state.update(dt, right_pos, right_rot)
            instrument_state.grip_strength = right_trigger.get('value', 0.0)
            instrument_state.is_active = right_trigger.get('pressed', False)
        
        # Update secondary instrument (left hand)
        secondary_instrument = self._get_secondary_instrument()
        if secondary_instrument in self.current_instruments:
            instrument_state = self.current_instruments[secondary_instrument]
            instrument_state.update(dt, left_pos, left_rot)
            instrument_state.grip_strength = left_trigger.get('value', 0.0)
            instrument_state.is_active = left_trigger.get('pressed', False)
    
    def _get_active_instrument(self) -> SurgicalInstrument:
        """Get currently active instrument based on procedure step"""
        if not self.current_procedure or not self.current_procedure.steps:
            return SurgicalInstrument.SCALPEL
        
        current_step = self.current_procedure.steps[self.current_step_index]
        step_type = current_step['type']
        
        # Map step to instrument
        step_instrument_map = {
            SurgicalStep.INCISION: SurgicalInstrument.SCALPEL,
            SurgicalStep.DISSECTION: SurgicalInstrument.SCISSORS,
            SurgicalStep.HEMOSTASIS: SurgicalInstrument.ELECTROCAUTERY,
            SurgicalStep.REPAIR: SurgicalInstrument.SUTURE_NEEDLE,
            SurgicalStep.CLOSURE: SurgicalInstrument.SUTURE_NEEDLE
        }
        
        return step_instrument_map.get(step_type, SurgicalInstrument.SCALPEL)
    
    def _get_secondary_instrument(self) -> SurgicalInstrument:
        """Get secondary instrument (usually forceps or retractor)"""
        return SurgicalInstrument.FORCEPS
    
    def _check_interactions(self, dt: float):
        """Check for instrument-tissue interactions"""
        if not self.patient_anatomy:
            return
        
        # Check each active instrument
        for instrument_type, instrument_state in self.current_instruments.items():
            if not instrument_state.is_active:
                continue
            
            # Check collision with anatomy
            collision_data = self._check_collision(instrument_state)
            
            if collision_data['collision']:
                # Handle interaction based on instrument type
                self._handle_interaction(instrument_type, instrument_state, 
                                       collision_data, dt)
    
    def _check_collision(self, instrument_state: InstrumentState) -> Dict:
        """Check collision with patient anatomy"""
        # Simplified collision detection
        # In production, use spatial partitioning and detailed anatomy
        collision_data = {
            'collision': False,
            'position': np.zeros(3),
            'normal': np.zeros(3),
            'depth': 0.0,
            'tissue_type': None,
            'resistance': 0.0
        }
        
        if not self.patient_anatomy:
            return collision_data
        
        # For simulation, create synthetic collision data
        # Based on procedure step and instrument position
        current_step = self.current_procedure.steps[self.current_step_index]
        
        # Simulate different tissue resistances based on step
        step_resistance = {
            SurgicalStep.INCISION: 0.3,
            SurgicalStep.DISSECTION: 0.5,
            SurgicalStep.HEMOSTASIS: 0.2,
            SurgicalStep.REPAIR: 0.4,
            SurgicalStep.CLOSURE: 0.3
        }
        
        resistance = step_resistance.get(current_step['type'], 0.3)
        
        # Add some randomness
        resistance += np.random.normal(0, 0.05)
        
        collision_data.update({
            'collision': True,
            'position': instrument_state.position,
            'normal': np.array([0, 1, 0]),  # Upward normal
            'depth': 0.01,  # 1 cm penetration
            'tissue_type': 'simulated_tissue',
            'resistance': max(0.1, min(1.0, resistance))
        })
        
        return collision_data
    
    def _handle_interaction(self, instrument_type: SurgicalInstrument,
                          instrument_state: InstrumentState,
                          collision_data: Dict, dt: float):
        """Handle instrument-tissue interaction"""
        
        # Create surgical action
        action = SurgicalAction(
            step_type=self.current_procedure.steps[self.current_step_index]['type'],
            instrument=instrument_type,
            position=instrument_state.position.copy(),
            orientation=instrument_state.rotation.copy(),
            force_applied=collision_data['resistance'] * 10.0,  # Scale resistance to force
            time_stamp=self.simulation_time,
            duration=dt
        )
        
        # Validate action
        validation = self.current_procedure.validate_action(
            self.current_step_index, action
        )
        
        # Update action with validation results
        action.success_score = validation['score']
        
        if validation['errors']:
            action.error_type = validation['errors'][0]
            action.error_severity = 0.5  # Default severity
        
        # Record action
        self.surgical_actions.append(action)
        
        # Provide haptic feedback
        self._provide_interaction_feedback(instrument_type, collision_data, 
                                         validation, action)
        
        # Update anatomy if available
        if self.patient_anatomy and hasattr(self.patient_anatomy, 'apply_surgical_action'):
            # Map instrument to surgical action type
            action_type_map = {
                SurgicalInstrument.SCALPEL: 'incision',
                SurgicalInstrument.ELECTROCAUTERY: 'cautery',
                SurgicalInstrument.SUTURE_NEEDLE: 'suture',
                SurgicalInstrument.CLAMP: 'clamp',
                SurgicalInstrument.SCISSORS: 'dissection'
            }
            
            action_type = action_type_map.get(instrument_type, 'incision')
            
            # Apply to anatomy
            self.patient_anatomy.apply_surgical_action(
                action_type=action_type,
                position=instrument_state.position,
                parameters={
                    'force': action.force_applied,
                    'depth': collision_data['depth'],
                    'temperature': instrument_state.temperature
                }
            )
        
        # Update metrics
        if action.error_type:
            self.performance_metrics['total_errors'] += 1
            if self.current_procedure.steps[self.current_step_index].get('critical', False):
                self.performance_metrics['critical_errors'] += 1
        
        # Simulate blood loss for certain errors
        if action.error_type in ['vascular_injury', 'inadequate_hemostasis']:
            blood_loss_rate = 10.0 * action.error_severity  # ml per error
            self.performance_metrics['blood_loss'] += blood_loss_rate * dt
    
    def _provide_interaction_feedback(self, instrument_type: SurgicalInstrument,
                                    collision_data: Dict, validation: Dict,
                                    action: SurgicalAction):
        """Provide haptic and visual feedback for interaction"""
        
        # Haptic feedback based on tissue resistance
        resistance = collision_data['resistance']
        self.haptic_feedback.provide_feedback('tissue_cutting', resistance)
        
        # Force feedback (resistance)
        if collision_data['normal'] is not None:
            force_vector = -collision_data['normal'] * resistance * 5.0
            self.haptic_feedback.update_force_feedback(force_vector)
        
        # Error feedback
        if action.error_type:
            self.haptic_feedback.provide_feedback('error_warning', 1.0)
            
            # Visual/audio error indication
            print(f"Error: {action.error_type} (severity: {action.error_severity})")
    
    def _update_metrics(self, dt: float):
        """Update performance metrics"""
        # Calculate efficiency score
        if self.current_procedure:
            time_ratio = self.simulation_time / (self.current_procedure.time_estimate * 60)
            efficiency = max(0, 1.0 - time_ratio)  # Penalize for taking too long
            
            error_penalty = self.performance_metrics['total_errors'] * 0.1
            blood_loss_penalty = self.performance_metrics['blood_loss'] / 100.0
            
            self.performance_metrics['efficiency_score'] = max(0, 
                efficiency - error_penalty - blood_loss_penalty)
    
    def switch_instrument(self, new_instrument: SurgicalInstrument):
        """Switch to different surgical instrument"""
        if new_instrument in self.current_instruments:
            print(f"Switching to {new_instrument.value}")
            self.performance_metrics['instrument_changes'] += 1
            
            # In production, would update VR controller model
            return True
        
        print(f"Instrument {new_instrument.value} not available")
        return False
    
    def record_voice_command(self, command: str):
        """Record and process voice command"""
        commands = {
            'scalpel': SurgicalInstrument.SCALPEL,
            'forceps': SurgicalInstrument.FORCEPS,
            'scissors': SurgicalInstrument.SCISSORS,
            'cautery': SurgicalInstrument.ELECTROCAUTERY,
            'suture': SurgicalInstrument.SUTURE_NEEDLE,
            'suction': SurgicalInstrument.SUCTION,
            'retractor': SurgicalInstrument.RETRACTOR
        }
        
        command_lower = command.lower().strip()
        
        if command_lower in commands:
            self.switch_instrument(commands[command_lower])
        elif 'help' in command_lower:
            self._provide_guidance()
        elif 'pause' in command_lower:
            self.pause_simulation()
        elif 'next step' in command_lower:
            self._advance_step()
    
    def _provide_guidance(self):
        """Provide guidance for current step"""
        if not self.current_procedure:
            return
        
        current_step = self.current_procedure.steps[self.current_step_index]
        step_type = current_step['type'].value
        
        guidance = {
            SurgicalStep.INCISION.value: "Make a clean, controlled incision along the marked line.",
            SurgicalStep.DISSECTION.value: "Carefully dissect through tissue planes, avoiding major vessels.",
            SurgicalStep.HEMOSTASIS.value: "Control bleeding using cautery or clamps as needed.",
            SurgicalStep.REPAIR.value: "Repair or remove the target tissue with precision.",
            SurgicalStep.CLOSURE.value: "Close the wound in layers with appropriate suture technique."
        }
        
        message = guidance.get(step_type, "Proceed with the current surgical step.")
        print(f"Guidance: {message}")
        
        # In VR, this would be displayed in headset
        return message
    
    def _advance_step(self):
        """Manually advance to next step"""
        if not self.current_procedure:
            return
        
        if self.current_step_index < len(self.current_procedure.steps) - 1:
            self.current_step_index += 1
            self.step_start_time = self.simulation_time
            print(f"Advanced to step {self.current_step_index + 1}")
        else:
            print("Already at final step")
    
    def get_current_status(self) -> Dict:
        """Get current simulation status"""
        if not self.current_procedure:
            return {'status': 'no_procedure_loaded'}
        
        step_index, step_progress = self.current_procedure.get_step_progress(
            self.simulation_time
        )
        
        current_step = self.current_procedure.steps[step_index]
        time_elapsed = self.simulation_time
        time_remaining = max(0, (self.current_procedure.time_estimate * 60) - time_elapsed)
        
        return {
            'status': 'running' if self.is_running else 'paused' if self.is_paused else 'stopped',
            'procedure': self.current_procedure.procedure_name,
            'current_step': {
                'index': step_index,
                'type': current_step['type'].value,
                'progress': step_progress,
                'critical': current_step.get('critical', False),
                'duration': current_step['duration']
            },
            'time': {
                'elapsed': time_elapsed,
                'remaining': time_remaining,
                'total_estimated': self.current_procedure.time_estimate * 60
            },
            'performance': self.performance_metrics.copy(),
            'active_instrument': self._get_active_instrument().value,
            'vr_enabled': self.vr_enabled,
            'haptics_enabled': self.haptics_enabled
        }
    
    def calculate_final_score(self) -> Dict:
        """Calculate final performance score"""
        if not self.current_procedure:
            return {'error': 'No procedure loaded'}
        
        # Base score components
        time_score = self._calculate_time_score()
        error_score = self._calculate_error_score()
        technique_score = self._calculate_technique_score()
        efficiency_score = self.performance_metrics['efficiency_score']
        
        # Weighted final score
        weights = {
            'time': 0.25,
            'errors': 0.35,
            'technique': 0.25,
            'efficiency': 0.15
        }
        
        final_score = (
            time_score * weights['time'] +
            error_score * weights['errors'] +
            technique_score * weights['technique'] +
            efficiency_score * weights['efficiency']
        )
        
        # Letter grade
        if final_score >= 0.9:
            grade = 'A'
        elif final_score >= 0.8:
            grade = 'B'
        elif final_score >= 0.7:
            grade = 'C'
        elif final_score >= 0.6:
            grade = 'D'
        else:
            grade = 'F'
        
        # Recommendations
        recommendations = self._generate_recommendations(
            time_score, error_score, technique_score
        )
        
        return {
            'final_score': final_score,
            'letter_grade': grade,
            'component_scores': {
                'time': time_score,
                'errors': error_score,
                'technique': technique_score,
                'efficiency': efficiency_score
            },
            'performance_metrics': self.performance_metrics.copy(),
            'recommendations': recommendations,
            'procedure_completed': self.simulation_time > 0,
            'completion_time': self.simulation_time
        }
    
    def _calculate_time_score(self) -> float:
        """Calculate score based on time"""
        if not self.current_procedure:
            return 0.0
        
        estimated_time = self.current_procedure.time_estimate * 60  # Convert to seconds
        actual_time = self.simulation_time
        
        if actual_time <= estimated_time:
            return 1.0
        elif actual_time <= estimated_time * 1.5:
            return 0.7
        elif actual_time <= estimated_time * 2.0:
            return 0.4
        else:
            return 0.1
    
    def _calculate_error_score(self) -> float:
        """Calculate score based on errors"""
        total_errors = self.performance_metrics['total_errors']
        critical_errors = self.performance_metrics['critical_errors']
        
        # Base error score
        if total_errors == 0:
            error_score = 1.0
        elif total_errors <= 3:
            error_score = 0.7
        elif total_errors <= 6:
            error_score = 0.4
        else:
            error_score = 0.1
        
        # Penalize critical errors
        critical_penalty = critical_errors * 0.3
        error_score = max(0, error_score - critical_penalty)
        
        return error_score
    
    def _calculate_technique_score(self) -> float:
        """Calculate score based on surgical technique"""
        if not self.surgical_actions:
            return 0.5  # Default
        
        # Average success score of all actions
        total_score = sum(action.success_score for action in self.surgical_actions)
        avg_score = total_score / len(self.surgical_actions)
        
        # Adjust based on action consistency
        score_variance = np.var([action.success_score for action in self.surgical_actions])
        consistency_bonus = max(0, 0.2 - score_variance)  # Reward consistent performance
        
        return min(1.0, avg_score + consistency_bonus)
    
    def _generate_recommendations(self, time_score: float, error_score: float, 
                                technique_score: float) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if time_score < 0.7:
            recommendations.append("Work on improving procedural efficiency and time management")
        
        if error_score < 0.7:
            recommendations.append("Focus on error reduction, especially avoiding critical errors")
        
        if technique_score < 0.7:
            recommendations.append("Practice instrument handling and surgical technique")
        
        if self.performance_metrics['blood_loss'] > 50:
            recommendations.append("Improve hemostasis techniques to reduce blood loss")
        
        if self.performance_metrics['instrument_changes'] > 10:
            recommendations.append("Plan instrument usage more efficiently to reduce changes")
        
        # Add specific error recommendations
        error_types = {}
        for action in self.surgical_actions:
            if action.error_type:
                error_types[action.error_type] = error_types.get(action.error_type, 0) + 1
        
        for error_type, count in error_types.items():
            if count >= 3:
                recommendations.append(f"Address recurring error: {error_type}")
        
        return recommendations
    
    def get_training_report(self, include_details: bool = True) -> Dict:
        """Generate comprehensive training report"""
        final_score = self.calculate_final_score()
        current_status = self.get_current_status()
        
        report = {
            'report_id': f"SR_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
            'generated_at': datetime.now().isoformat(),
            'procedure': self.current_procedure.procedure_name if self.current_procedure else 'Unknown',
            'difficulty': self.current_procedure.difficulty if self.current_procedure else 'medium',
            'summary': final_score,
            'status': current_status,
            'simulation_parameters': {
                'vr_enabled': self.vr_enabled,
                'haptics_enabled': self.haptics_enabled,
                'difficulty': self.difficulty
            }
        }
        
        if include_details:
            report['detailed_actions'] = [
                {
                    'step': action.step_type.value,
                    'instrument': action.instrument.value,
                    'time': action.time_stamp,
                    'score': action.success_score,
                    'error': action.error_type,
                    'force': action.force_applied
                }
                for action in self.surgical_actions
            ]
            
            report['step_analysis'] = self._analyze_step_performance()
        
        return report
    
    def _analyze_step_performance(self) -> Dict:
        """Analyze performance for each step"""
        if not self.current_procedure:
            return {}
        
        step_analysis = {}
        
        for i, step in enumerate(self.current_procedure.steps):
            step_actions = [a for a in self.surgical_actions 
                          if a.step_type == step['type']]
            
            if step_actions:
                avg_score = np.mean([a.success_score for a in step_actions])
                error_count = sum(1 for a in step_actions if a.error_type)
                
                step_analysis[i] = {
                    'step_type': step['type'].value,
                    'action_count': len(step_actions),
                    'average_score': avg_score,
                    'error_count': error_count,
                    'critical_step': step.get('critical', False)
                }
        
        return step_analysis
    
    def export_training_data(self, format: str = 'json') -> str:
        """Export training data for analysis"""
        report = self.get_training_report(include_details=True)
        
        if format.lower() == 'json':
            import json
            return json.dumps(report, indent=2, default=str)
        elif format.lower() == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Action_Index', 'Step', 'Instrument', 'Time', 
                           'Score', 'Error', 'Force', 'Position_X', 'Position_Y', 'Position_Z'])
            
            # Write data
            for i, action in enumerate(self.surgical_actions):
                writer.writerow([
                    i,
                    action.step_type.value,
                    action.instrument.value,
                    action.time_stamp,
                    action.success_score,
                    action.error_type or '',
                    action.force_applied,
                    action.position[0],
                    action.position[1],
                    action.position[2]
                ])
            
            return output.getvalue()
        else:
            return f"Format {format} not supported"
    
    def replay_simulation(self, speed: float = 1.0):
        """Replay simulation for review"""
        print(f"Replaying simulation at {speed}x speed")
        
        # In production, this would replay the surgical actions
        # For now, just print summary
        for i, action in enumerate(self.surgical_actions[:10]):  # First 10 actions
            print(f"Action {i}: {action.instrument.value} at {action.time_stamp:.1f}s - "
                  f"Score: {action.success_score:.2f}")
    
    def reset_simulation(self):
        """Reset simulation to initial state"""
        self.simulation_time = 0.0
        self.current_step_index = 0
        self.step_start_time = 0.0
        self.surgical_actions.clear()
        self._reset_metrics()
        
        # Reset instrument positions
        self._initialize_instruments()
        
        print("Simulation reset")
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up Surgical Simulator...")
        
        self.vr_controller.cleanup()
        self.is_running = False
        self.current_procedure = None
        self.surgical_actions.clear()

# Training scenarios

class SurgicalScenario:
    """Pre-defined surgical training scenario"""
    
    def __init__(self, scenario_name: str, difficulty: str = 'medium'):
        self.scenario_name = scenario_name
        self.difficulty = difficulty
        self.complications = []
        self.time_pressure = False
        self.resource_limits = {}
        self.learning_objectives = []
        
        self._load_scenario()
    
    def _load_scenario(self):
        """Load scenario definition"""
        scenarios = {
            'emergency_appendectomy': {
                'difficulty': 'high',
                'complications': ['bleeding', 'perforation', 'adhesions'],
                'time_pressure': True,
                'resource_limits': {'blood_units': 2, 'suture_material': 1},
                'learning_objectives': [
                    'Manage intraoperative complications',
                    'Work under time pressure',
                    'Resource conservation'
                ]
            },
            'elective_cholecystectomy': {
                'difficulty': 'medium',
                'complications': ['gallbladder_perforation', 'bile_leak'],
                'time_pressure': False,
                'resource_limits': {},
                'learning_objectives': [
                    'Laparoscopic skills',
                    'Anatomical identification',
                    'Safe dissection'
                ]
            },
            'trauma_exploration': {
                'difficulty': 'very_high',
                'complications': ['massive_hemorrhage', 'organ_injury', 'shock'],
                'time_pressure': True,
                'resource_limits': {'time': 30, 'assistance': 1},
                'learning_objectives': [
                    'Damage control surgery',
                    'Rapid decision making',
                    'Team communication'
                ]
            },
            'pediatric_hernia_repair': {
                'difficulty': 'medium',
                'complications': ['vessel_injury', 'nerve_damage'],
                'time_pressure': False,
                'resource_limits': {'instrument_size': 'pediatric'},
                'learning_objectives': [
                    'Delicate tissue handling',
                    'Precision suturing',
                    'Pediatric considerations'
                ]
            }
        }
        
        if self.scenario_name in scenarios:
            scenario_data = scenarios[self.scenario_name]
            self.difficulty = scenario_data['difficulty']
            self.complications = scenario_data['complications']
            self.time_pressure = scenario_data['time_pressure']
            self.resource_limits = scenario_data['resource_limits']
            self.learning_objectives = scenario_data['learning_objectives']
        else:
            # Default scenario
            self.complications = []
            self.time_pressure = False
            self.resource_limits = {}
            self.learning_objectives = ['Basic surgical technique']

class SurgicalTrainingManager:
    """Manager for surgical training programs"""
    
    def __init__(self):
        self.trainees = {}
        self.training_programs = {}
        self.completed_sessions = []
        self.performance_history = {}
        
        self._initialize_training_programs()
    
    def _initialize_training_programs(self):
        """Initialize training programs"""
        self.training_programs = {
            'beginner': {
                'procedures': ['basic_incision', 'simple_suture'],
                'scenarios': [],
                'prerequisites': [],
                'completion_criteria': {'average_score': 0.7, 'sessions': 5}
            },
            'intermediate': {
                'procedures': ['appendectomy', 'cholecystectomy'],
                'scenarios': ['elective_cholecystectomy'],
                'prerequisites': ['beginner'],
                'completion_criteria': {'average_score': 0.8, 'sessions': 10}
            },
            'advanced': {
                'procedures': ['trauma_exploration', 'vascular_repair'],
                'scenarios': ['emergency_appendectomy', 'trauma_exploration'],
                'prerequisites': ['intermediate'],
                'completion_criteria': {'average_score': 0.85, 'sessions': 15}
            },
            'expert': {
                'procedures': ['complex_reconstruction', 'transplant_surgery'],
                'scenarios': ['complex_trauma', 'mass_casualty'],
                'prerequisites': ['advanced'],
                'completion_criteria': {'average_score': 0.9, 'sessions': 20}
            }
        }
    
    def register_trainee(self, trainee_id: str, name: str, 
                        current_level: str = 'beginner'):
        """Register a new trainee"""
        self.trainees[trainee_id] = {
            'name': name,
            'level': current_level,
            'start_date': datetime.now().isoformat(),
            'sessions_completed': 0,
            'average_score': 0.0,
            'skills': {},
            'weak_areas': []
        }
        
        print(f"Trainee registered: {name} (ID: {trainee_id}, Level: {current_level})")
    
    def record_session(self, trainee_id: str, simulator: SurgicalSimulator):
        """Record training session results"""
        if trainee_id not in self.trainees:
            print(f"Trainee {trainee_id} not found")
            return
        
        # Get simulation results
        final_score = simulator.calculate_final_score()
        status = simulator.get_current_status()
        
        # Update trainee record
        trainee = self.trainees[trainee_id]
        trainee['sessions_completed'] += 1
        
        # Update average score
        current_avg = trainee['average_score']
        session_score = final_score.get('final_score', 0.0)
        new_avg = (current_avg * (trainee['sessions_completed'] - 1) + session_score) / trainee['sessions_completed']
        trainee['average_score'] = new_avg
        
        # Update skills assessment
        self._update_skills_assessment(trainee_id, final_score, status)
        
        # Record session
        session_record = {
            'trainee_id': trainee_id,
            'procedure': status.get('procedure', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'score': session_score,
            'grade': final_score.get('letter_grade', 'F'),
            'duration': status['time']['elapsed'],
            'errors': status['performance']['total_errors'],
            'critical_errors': status['performance']['critical_errors']
        }
        
        self.completed_sessions.append(session_record)
        
        # Check for level progression
        self._check_level_progression(trainee_id)
        
        print(f"Session recorded for {trainee['name']}: Score={session_score:.2f}, Grade={session_record['grade']}")
        
        return session_record
    
    def _update_skills_assessment(self, trainee_id: str, final_score: Dict, 
                                status: Dict):
        """Update skills assessment based on performance"""
        trainee = self.trainees[trainee_id]
        
        # Extract skill metrics
        component_scores = final_score.get('component_scores', {})
        recommendations = final_score.get('recommendations', [])
        
        # Define skill categories
        skill_categories = {
            'technical_skill': ['technique', 'efficiency'],
            'cognitive_skill': ['errors', 'decision_making'],
            'non_technical_skill': ['time_management', 'resource_management']
        }
        
        # Update skill scores
        for category, components in skill_categories.items():
            if category not in trainee['skills']:
                trainee['skills'][category] = {'score': 0.0, 'sessions': 0}
            
            # Calculate category score from components
            category_score = 0.0
            valid_components = 0
            
            for component in components:
                if component in component_scores:
                    category_score += component_scores[component]
                    valid_components += 1
            
            if valid_components > 0:
                avg_score = category_score / valid_components
                
                # Update weighted average
                current = trainee['skills'][category]
                new_score = (current['score'] * current['sessions'] + avg_score) / (current['sessions'] + 1)
                
                trainee['skills'][category]['score'] = new_score
                trainee['skills'][category]['sessions'] += 1
        
        # Update weak areas from recommendations
        for recommendation in recommendations:
            if 'efficiency' in recommendation.lower():
                weak_area = 'time_management'
            elif 'error' in recommendation.lower():
                weak_area = 'technical_precision'
            elif 'technique' in recommendation.lower():
                weak_area = 'instrument_handling'
            else:
                weak_area = 'general_technique'
            
            if weak_area not in trainee['weak_areas']:
                trainee['weak_areas'].append(weak_area)
    
    def _check_level_progression(self, trainee_id: str):
        """Check if trainee can progress to next level"""
        trainee = self.trainees[trainee_id]
        current_level = trainee['level']
        
        if current_level not in self.training_programs:
            return
        
        program = self.training_programs[current_level]
        completion_criteria = program['completion_criteria']
        
        # Check if criteria met
        score_met = trainee['average_score'] >= completion_criteria['average_score']
        sessions_met = trainee['sessions_completed'] >= completion_criteria['sessions']
        
        if score_met and sessions_met:
            # Find next level
            levels = list(self.training_programs.keys())
            current_index = levels.index(current_level)
            
            if current_index < len(levels) - 1:
                next_level = levels[current_index + 1]
                
                # Check prerequisites
                next_program = self.training_programs[next_level]
                if current_level in next_program['prerequisites']:
                    trainee['level'] = next_level
                    print(f"{trainee['name']} has progressed to {next_level} level!")
                    
                    # Generate certificate
                    self._generate_certificate(trainee_id, current_level)
    
    def _generate_certificate(self, trainee_id: str, completed_level: str):
        """Generate training completion certificate"""
        trainee = self.trainees[trainee_id]
        
        certificate = {
            'certificate_id': f"CERT_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
            'trainee_name': trainee['name'],
            'trainee_id': trainee_id,
            'level_completed': completed_level,
            'completion_date': datetime.now().isoformat(),
            'average_score': trainee['average_score'],
            'sessions_completed': trainee['sessions_completed'],
            'skills_summary': trainee['skills']
        }
        
        print(f"Certificate generated for {trainee['name']}: {completed_level} level")
        return certificate
    
    def get_trainee_report(self, trainee_id: str) -> Dict:
        """Get comprehensive report for trainee"""
        if trainee_id not in self.trainees:
            return {'error': 'Trainee not found'}
        
        trainee = self.trainees[trainee_id]
        
        # Get trainee's sessions
        trainee_sessions = [s for s in self.completed_sessions 
                          if s['trainee_id'] == trainee_id]
        
        # Calculate statistics
        if trainee_sessions:
            recent_sessions = trainee_sessions[-5:]  # Last 5 sessions
            recent_avg = np.mean([s['score'] for s in recent_sessions])
            
            # Performance trend
            session_scores = [s['score'] for s in trainee_sessions]
            if len(session_scores) >= 3:
                trend = np.polyfit(range(len(session_scores)), session_scores, 1)[0]
            else:
                trend = 0.0
        else:
            recent_avg = 0.0
            trend = 0.0
        
        # Generate recommendations for improvement
        improvement_plan = self._generate_improvement_plan(trainee_id)
        
        report = {
            'trainee_info': trainee.copy(),
            'statistics': {
                'total_sessions': len(trainee_sessions),
                'average_score': trainee['average_score'],
                'recent_average': recent_avg,
                'performance_trend': trend,
                'current_level': trainee['level'],
                'next_level_requirements': self._get_next_level_requirements(trainee['level'])
            },
            'session_history': trainee_sessions[-10:],  # Last 10 sessions
            'skills_assessment': trainee['skills'],
            'weak_areas': trainee['weak_areas'],
            'improvement_plan': improvement_plan,
            'certificates_earned': []
        }
        
        return report
    
    def _generate_improvement_plan(self, trainee_id: str) -> Dict:
        """Generate personalized improvement plan"""
        trainee = self.trainees[trainee_id]
        
        plan = {
            'focus_areas': [],
            'recommended_procedures': [],
            'training_goals': [],
            'timeline': '4 weeks'
        }
        
        # Identify weakest skills
        skills = trainee['skills']
        if skills:
            weakest_category = min(skills.items(), key=lambda x: x[1]['score'])[0]
            
            focus_map = {
                'technical_skill': ['Basic suturing drills', 'Instrument handling exercises'],
                'cognitive_skill': ['Surgical decision making scenarios', 'Complication management'],
                'non_technical_skill': ['Time management exercises', 'Resource allocation scenarios']
            }
            
            plan['focus_areas'] = focus_map.get(weakest_category, ['General surgical practice'])
        
        # Recommend procedures based on weak areas
        weak_areas = trainee['weak_areas']
        if 'technical_precision' in weak_areas:
            plan['recommended_procedures'].append('microsurgery_practice')
        if 'time_management' in weak_areas:
            plan['recommended_procedures'].append('timed_procedures')
        if 'instrument_handling' in weak_areas:
            plan['recommended_procedures'].append('basic_instrument_drills')
        
        # Set goals
        current_score = trainee['average_score']
        goal_score = min(1.0, current_score + 0.1)  # Aim for 10% improvement
        
        plan['training_goals'] = [
            f"Achieve average score of {goal_score:.2f}",
            f"Reduce errors by 20%",
            f"Complete 3 sessions focusing on {', '.join(plan['focus_areas'][:2])}"
        ]
        
        return plan
    
    def _get_next_level_requirements(self, current_level: str) -> Dict:
        """Get requirements for next level"""
        levels = list(self.training_programs.keys())
        current_index = levels.index(current_level)
        
        if current_index < len(levels) - 1:
            next_level = levels[current_index + 1]
            program = self.training_programs[next_level]
            return {
                'next_level': next_level,
                'prerequisites': program['prerequisites'],
                'completion_criteria': program['completion_criteria']
            }
        
        return {'next_level': None, 'message': 'Maximum level achieved'}
    
    def export_training_program(self, format: str = 'json') -> str:
        """Export training program structure"""
        import json
        
        program_data = {
            'training_programs': self.training_programs,
            'trainee_count': len(self.trainees),
            'total_sessions': len(self.completed_sessions),
            'generated_at': datetime.now().isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(program_data, indent=2)
        else:
            return f"Format {format} not supported"
    
    def cleanup(self):
        """Clean up training manager"""
        print("Cleaning up Surgical Training Manager...")
        self.trainees.clear()
        self.completed_sessions.clear()

# Example usage
def run_surgical_training_demo():
    """Run surgical training demonstration"""
    print("Running Surgical Training Demo...")
    
    # Initialize simulator
    simulator = SurgicalSimulator(vr_enabled=False, haptics_enabled=True)
    
    # Load a procedure
    simulator.load_procedure('appendectomy')
    
    # Start simulation
    simulator.start_simulation()
    
    # Simulate some surgical actions
    print("\nSimulating surgical actions...")
    
    # Update simulation with some actions
    for i in range(5):
        simulator.update(0.5)  # 500ms time steps
        
        # Get current status
        status = simulator.get_current_status()
        print(f"Time: {status['time']['elapsed']:.1f}s - "
              f"Step: {status['current_step']['type']} - "
              f"Progress: {status['current_step']['progress']:.1%}")
    
    # Stop simulation
    simulator.stop_simulation()
    
    # Get final score
    final_score = simulator.calculate_final_score()
    print(f"\nFinal Score: {final_score['final_score']:.2f} ({final_score['letter_grade']})")
    
    # Get training report
    report = simulator.get_training_report()
    print(f"\nTraining Report ID: {report['report_id']}")
    
    # Initialize training manager
    training_manager = SurgicalTrainingManager()
    
    # Register a trainee
    training_manager.register_trainee('T001', 'Dr. John Smith', 'beginner')
    
    # Record session
    session_record = training_manager.record_session('T001', simulator)
    print(f"\nSession Recorded:")
    print(f"  Procedure: {session_record['procedure']}")
    print(f"  Score: {session_record['score']:.2f}")
    print(f"  Grade: {session_record['grade']}")
    
    # Get trainee report
    trainee_report = training_manager.get_trainee_report('T001')
    print(f"\nTrainee Report:")
    print(f"  Name: {trainee_report['trainee_info']['name']}")
    print(f"  Level: {trainee_report['statistics']['current_level']}")
    print(f"  Average Score: {trainee_report['statistics']['average_score']:.2f}")
    
    # Cleanup
    simulator.cleanup()
    training_manager.cleanup()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    run_surgical_training_demo()