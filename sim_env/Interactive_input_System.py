"""
Advanced Interactive Input System
Multi-modal input system supporting touch, voice, gesture, eye-tracking, BCI, and AI-powered interaction
"""

import numpy as np
import pygame
import speech_recognition as sr
import threading
import time
import json
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import mediapipe as mp
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class InputModality(Enum):
    """Supported input modalities"""
    TOUCH = "touch"
    VOICE = "voice"
    GESTURE = "gesture"
    EYE_TRACKING = "eye_tracking"
    BCI = "bci"
    MULTI_MODAL = "multi_modal"
    AI_PREDICTIVE = "ai_predictive"

class GestureType(Enum):
    """Recognized gesture types"""
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH_IN = "pinch_in"
    PINCH_OUT = "pinch_out"
    CIRCLE_CLOCKWISE = "circle_clockwise"
    CIRCLE_COUNTERCLOCKWISE = "circle_counterclockwise"
    WAVE = "wave"
    FIST = "fist"
    POINT = "point"
    GRAB = "grab"

class VoiceCommandType(Enum):
    """Supported voice command types"""
    SIMULATION_CONTROL = "simulation_control"
    NAVIGATION = "navigation"
    SYSTEM_COMMAND = "system_command"
    DATA_QUERY = "data_query"
    CREATION = "creation"
    MODIFICATION = "modification"

@dataclass
class InputEvent:
    """Universal input event representation"""
    event_id: str
    modality: InputModality
    timestamp: float
    data: Dict[str, Any]
    confidence: float = 1.0
    source: str = "unknown"
    processed: bool = False

@dataclass
class GestureData:
    """Gesture recognition data"""
    gesture_type: GestureType
    hand_landmarks: List[Tuple[float, float, float]]
    bounding_box: Tuple[float, float, float, float]  # x, y, w, h
    velocity: Tuple[float, float, float]  # x, y, z velocity
    duration: float
    confidence: float

@dataclass
class VoiceCommand:
    """Voice command data"""
    command_type: VoiceCommandType
    text: str
    intent: str
    parameters: Dict[str, Any]
    confidence: float

@dataclass
class GazeData:
    """Eye tracking gaze data"""
    gaze_point: Tuple[float, float]  # x, y coordinates
    gaze_origin: Tuple[float, float, float]  # 3D origin point
    gaze_direction: Tuple[float, float, float]  # 3D direction vector
    pupil_diameter: float
    blink_detected: bool
    confidence: float

@dataclass
class BCIData:
    """Brain-Computer Interface data"""
    signal_type: str
    signal_data: np.ndarray
    mental_command: str
    intensity: float
    confidence: float

class AdvancedGestureRecognizer:
    """Advanced gesture recognition using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.gesture_history = deque(maxlen=10)
        self.current_gestures = []
        
    def process_frame(self, frame: np.ndarray) -> List[GestureData]:
        """Process video frame for gesture recognition"""
        gestures = []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    gesture = self._analyze_hand_landmarks(hand_landmarks)
                    if gesture:
                        gestures.append(gesture)
                        
            self.current_gestures = gestures
            self.gesture_history.extend(gestures)
            
        except Exception as e:
            print(f"Gesture recognition error: {e}")
            
        return gestures
        
    def _analyze_hand_landmarks(self, hand_landmarks) -> Optional[GestureData]:
        """Analyze hand landmarks to detect gestures"""
        try:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
                
            # Calculate hand bounding box
            x_coords = [lm[0] for lm in landmarks]
            y_coords = [lm[1] for lm in landmarks]
            bbox = (min(x_coords), min(y_coords), 
                   max(x_coords) - min(x_coords), 
                   max(y_coords) - min(y_coords))
            
            # Detect specific gestures
            gesture_type, confidence = self._detect_gesture_type(landmarks)
            
            if gesture_type:
                return GestureData(
                    gesture_type=gesture_type,
                    hand_landmarks=landmarks,
                    bounding_box=bbox,
                    velocity=(0, 0, 0),  # Would need temporal data for velocity
                    duration=0.1,
                    confidence=confidence
                )
                
        except Exception as e:
            print(f"Hand landmark analysis error: {e}")
            
        return None
        
    def _detect_gesture_type(self, landmarks: List[Tuple[float, float, float]]) -> Tuple[Optional[GestureType], float]:
        """Detect specific gesture types from hand landmarks"""
        try:
            # Get key landmark indices
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            wrist = landmarks[0]
            
            # Calculate distances for pinch detection
            thumb_index_dist = np.sqrt(
                (thumb_tip[0] - index_tip[0])**2 + 
                (thumb_tip[1] - index_tip[1])**2
            )
            
            # Pinch gestures
            if thumb_index_dist < 0.05:
                return GestureType.PINCH_IN, 0.9
                
            # Fist detection (all fingers curled)
            if self._is_fist(landmarks):
                return GestureType.FIST, 0.85
                
            # Pointing gesture
            if self._is_pointing(landmarks):
                return GestureType.POINT, 0.8
                
            # Open hand (could be wave or grab)
            if self._is_open_hand(landmarks):
                return GestureType.WAVE, 0.7
                
        except Exception as e:
            print(f"Gesture type detection error: {e}")
            
        return None, 0.0
        
    def _is_fist(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """Check if hand is making a fist"""
        try:
            # Simplified fist detection
            finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]  # Index, middle, ring, pinky tips
            finger_bases = [landmarks[i] for i in [5, 9, 13, 17]]  # Corresponding bases
            
            for tip, base in zip(finger_tips, finger_bases):
                if tip[1] > base[1]:  # Tip below base (fingers curled)
                    continue
                else:
                    return False
            return True
        except:
            return False
            
    def _is_pointing(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """Check if hand is pointing"""
        try:
            index_tip = landmarks[8]
            index_base = landmarks[5]
            other_tips = [landmarks[12], landmarks[16], landmarks[20]]  # Middle, ring, pinky
            
            # Index finger extended, others curled
            if index_tip[1] < index_base[1]:  # Index tip above base
                for tip in other_tips:
                    if tip[1] < index_base[1]:  # Other finger also extended
                        return False
                return True
        except:
            pass
        return False
        
    def _is_open_hand(self, landmarks: List[Tuple[float, float, float]]) -> bool:
        """Check if hand is open"""
        try:
            finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]
            finger_bases = [landmarks[i] for i in [5, 9, 13, 17]]
            
            extended_fingers = 0
            for tip, base in zip(finger_tips, finger_bases):
                if tip[1] < base[1]:  # Tip above base (finger extended)
                    extended_fingers += 1
                    
            return extended_fingers >= 3  # At least 3 fingers extended
        except:
            return False

class AdvancedVoiceRecognizer:
    """Advanced voice recognition with intent parsing"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.command_history = deque(maxlen=50)
        
        # Voice command patterns
        self.command_patterns = {
            VoiceCommandType.SIMULATION_CONTROL: [
                r"(start|stop|pause|resume|reset).*simulation",
                r"(run|halt).*experiment",
                r"(add|remove).*particle",
                r"(increase|decrease).*(speed|size|force)"
            ],
            VoiceCommandType.NAVIGATION: [
                r"(zoom|pan|rotate).*(in|out|left|right|up|down)",
                r"(move|go).*(forward|backward|left|right)",
                r"(focus|look).*at",
                r"(camera|view).*(change|switch)"
            ],
            VoiceCommandType.SYSTEM_COMMAND: [
                r"(save|load|export).*(project|data)",
                r"(open|close).*(menu|panel|window)",
                r"(enable|disable).*(feature|mode)",
                r"(show|hide).*(interface|gui)"
            ],
            VoiceCommandType.DATA_QUERY: [
                r"(what|show).*(data|results|analysis)",
                r"(how many).*(particles|measurements)",
                r"(what is).*(value|parameter|setting)",
                r"(analyze|compute).*(data|results)"
            ],
            VoiceCommandType.CREATION: [
                r"(create|make|add).*(new|object|element)",
                r"(build|construct).*(model|system)",
                r"(generate).*(pattern|structure)"
            ],
            VoiceCommandType.MODIFICATION: [
                r"(change|modify|adjust).*(parameter|setting|value)",
                r"(set).*(to|value)",
                r"(update).*(property|attribute)"
            ]
        }
        
        # Calibrate microphone
        self._calibrate_microphone()
        
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            print("Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Microphone calibration complete")
        except Exception as e:
            print(f"Microphone calibration failed: {e}")
            
    def start_listening(self, callback: Callable[[VoiceCommand], None]):
        """Start continuous voice listening"""
        self.is_listening = True
        self.listening_thread = threading.Thread(
            target=self._listening_loop,
            args=(callback,),
            daemon=True
        )
        self.listening_thread.start()
        
    def stop_listening(self):
        """Stop voice listening"""
        self.is_listening = False
        if hasattr(self, 'listening_thread'):
            self.listening_thread.join(timeout=1.0)
            
    def _listening_loop(self, callback: Callable[[VoiceCommand], None]):
        """Continuous listening loop"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                if text:
                    command = self._parse_voice_command(text)
                    if command:
                        callback(command)
                        
            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except sr.UnknownValueError:
                # Speech not understood
                continue
            except Exception as e:
                print(f"Voice recognition error: {e}")
                time.sleep(0.1)  # Prevent tight loop on error
                
    def _parse_voice_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse voice text into structured command"""
        try:
            text_lower = text.lower()
            
            # Determine command type
            command_type = None
            intent = ""
            parameters = {}
            
            for cmd_type, patterns in self.command_patterns.items():
                for pattern in patterns:
                    if self._matches_pattern(text_lower, pattern):
                        command_type = cmd_type
                        intent = self._extract_intent(text_lower, pattern)
                        parameters = self._extract_parameters(text_lower)
                        break
                if command_type:
                    break
                    
            if command_type:
                confidence = self._calculate_confidence(text_lower, command_type)
                command = VoiceCommand(
                    command_type=command_type,
                    text=text,
                    intent=intent,
                    parameters=parameters,
                    confidence=confidence
                )
                self.command_history.append(command)
                return command
                
        except Exception as e:
            print(f"Voice command parsing error: {e}")
            
        return None
        
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches command pattern"""
        # Simple keyword matching - in practice, use regex or NLP
        keywords = pattern.replace('(', '').replace(')', '').replace('.*', ' ').split()
        return any(keyword in text for keyword in keywords if len(keyword) > 2)
        
    def _extract_intent(self, text: str, pattern: str) -> str:
        """Extract intent from voice command"""
        # Simplified intent extraction
        if "simulation" in text:
            return "simulation_control"
        elif "camera" in text or "view" in text:
            return "view_control"
        elif "data" in text or "results" in text:
            return "data_access"
        elif "create" in text or "add" in text:
            return "object_creation"
        else:
            return "general_command"
            
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract parameters from voice command"""
        parameters = {}
        
        # Extract numerical values
        words = text.split()
        for i, word in enumerate(words):
            try:
                # Look for numbers
                if word.isdigit():
                    parameters['value'] = int(word)
                elif word.replace('.', '').isdigit():
                    parameters['value'] = float(word)
                    
                # Look for directions
                if word in ['left', 'right', 'up', 'down', 'forward', 'backward']:
                    parameters['direction'] = word
                    
                # Look for quantities
                if word in ['more', 'less', 'increase', 'decrease']:
                    parameters['modification'] = word
                    
            except:
                continue
                
        return parameters
        
    def _calculate_confidence(self, text: str, command_type: VoiceCommandType) -> float:
        """Calculate confidence score for voice command"""
        base_confidence = 0.7
        
        # Increase confidence for longer, more specific commands
        word_count = len(text.split())
        if word_count > 3:
            base_confidence += 0.1
            
        # Increase confidence for recent command patterns
        recent_commands = list(self.command_history)[-5:]
        if any(cmd.text.lower() in text.lower() for cmd in recent_commands):
            base_confidence += 0.1
            
        return min(0.95, base_confidence)

class EyeTrackingSystem:
    """Advanced eye tracking system"""
    
    def __init__(self):
        self.is_tracking = False
        self.gaze_data = None
        self.calibration_points = []
        self.calibration_data = {}
        self.calibrated = False
        
    def start_tracking(self):
        """Start eye tracking"""
        self.is_tracking = True
        # In a real implementation, this would initialize camera and tracking
        print("Eye tracking started (simulated)")
        
    def stop_tracking(self):
        """Stop eye tracking"""
        self.is_tracking = False
        print("Eye tracking stopped")
        
    def get_gaze_data(self) -> Optional[GazeData]:
        """Get current gaze data"""
        if not self.is_tracking:
            return None
            
        # Simulated gaze data - in practice, this would come from eye tracker
        gaze_point = (np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7))
        gaze_origin = (0, 0, 0.5)  # Simplified
        gaze_direction = (gaze_point[0] - 0.5, gaze_point[1] - 0.5, -1)
        
        return GazeData(
            gaze_point=gaze_point,
            gaze_origin=gaze_origin,
            gaze_direction=gaze_direction,
            pupil_diameter=3.5 + np.random.normal(0, 0.1),
            blink_detected=np.random.random() < 0.01,  # 1% chance of blink
            confidence=0.8 if self.calibrated else 0.5
        )
        
    def calibrate(self, calibration_points: List[Tuple[float, float]]):
        """Calibrate eye tracking system"""
        print(f"Calibrating eye tracking with {len(calibration_points)} points")
        self.calibration_points = calibration_points
        
        # Simulate calibration process
        time.sleep(2)
        self.calibrated = True
        print("Eye tracking calibration complete")
        
    def is_looking_at(self, region: Tuple[float, float, float, float], 
                     gaze_data: GazeData) -> bool:
        """Check if user is looking at specific screen region"""
        x, y, w, h = region
        gaze_x, gaze_y = gaze_data.gaze_point
        
        return (x <= gaze_x <= x + w and y <= gaze_y <= y + h)

class BCISimulator:
    """Brain-Computer Interface simulator for development"""
    
    def __init__(self):
        self.is_connected = False
        self.signal_types = ['EEG', 'fNIRS', 'EMG', 'ECG']
        self.mental_commands = [
            'focus', 'relax', 'push', 'pull', 'left', 'right', 'up', 'down'
        ]
        
    def connect(self):
        """Connect to BCI device (simulated)"""
        self.is_connected = True
        print("BCI connected (simulated)")
        
    def disconnect(self):
        """Disconnect from BCI device"""
        self.is_connected = False
        print("BCI disconnected")
        
    def get_bci_data(self) -> Optional[BCIData]:
        """Get BCI data (simulated)"""
        if not self.is_connected:
            return None
            
        # Simulate BCI data
        signal_type = np.random.choice(self.signal_types)
        mental_command = np.random.choice(self.mental_commands)
        intensity = np.random.uniform(0.1, 1.0)
        
        # Generate simulated signal data
        signal_data = np.random.normal(0, 1, 100)  # 100 samples
        
        return BCIData(
            signal_type=signal_type,
            signal_data=signal_data,
            mental_command=mental_command,
            intensity=intensity,
            confidence=np.random.uniform(0.6, 0.9)
        )

class MultiModalInputFusion:
    """Fusion engine for multi-modal input data"""
    
    def __init__(self):
        self.modality_weights = {
            InputModality.TOUCH: 0.9,
            InputModality.VOICE: 0.8,
            InputModality.GESTURE: 0.7,
            InputModality.EYE_TRACKING: 0.6,
            InputModality.BCI: 0.5,
            InputModality.AI_PREDICTIVE: 0.85
        }
        
        self.fusion_history = deque(maxlen=100)
        self.confidence_threshold = 0.6
        
    def fuse_inputs(self, input_events: List[InputEvent]) -> InputEvent:
        """Fuse multiple input events into a single coherent event"""
        if not input_events:
            raise ValueError("No input events to fuse")
            
        if len(input_events) == 1:
            return input_events[0]
            
        # Group events by temporal proximity (within 0.5 seconds)
        current_time = time.time()
        recent_events = [e for e in input_events 
                        if current_time - e.timestamp < 0.5]
        
        if not recent_events:
            return input_events[0]  # Return first event if no recent ones
            
        # Calculate fused confidence
        total_confidence = 0
        total_weight = 0
        
        for event in recent_events:
            weight = self.modality_weights.get(event.modality, 0.5)
            total_confidence += event.confidence * weight
            total_weight += weight
            
        fused_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        
        # Select highest confidence event as base
        base_event = max(recent_events, key=lambda e: e.confidence)
        
        # Create fused event
        fused_data = base_event.data.copy()
        
        # Enhance data with information from other modalities
        for event in recent_events:
            if event != base_event:
                fused_data.update({
                    f"{event.modality.value}_enhanced": event.data,
                    f"{event.modality.value}_confidence": event.confidence
                })
                
        fused_event = InputEvent(
            event_id=f"fused_{base_event.event_id}",
            modality=InputModality.MULTI_MODAL,
            timestamp=current_time,
            data=fused_data,
            confidence=fused_confidence,
            source="fusion_engine"
        )
        
        self.fusion_history.append(fused_event)
        return fused_event
        
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get statistics about input fusion"""
        if not self.fusion_history:
            return {}
            
        confidences = [event.confidence for event in self.fusion_history]
        modalities_used = {}
        
        for event in self.fusion_history:
            for key in event.data:
                if key.endswith('_enhanced'):
                    modality = key.replace('_enhanced', '')
                    modalities_used[modality] = modalities_used.get(modality, 0) + 1
                    
        return {
            'total_fusions': len(self.fusion_history),
            'average_confidence': np.mean(confidences),
            'modality_usage': modalities_used,
            'success_rate': np.mean([1 if c > self.confidence_threshold else 0 
                                   for c in confidences])
        }

class AIPredictiveInput:
    """AI-powered predictive input system"""
    
    def __init__(self):
        self.prediction_history = deque(maxlen=1000)
        self.user_behavior_model = {}
        self.prediction_horizon = 2.0  # seconds
        
    def predict_next_input(self, current_context: Dict[str, Any]) -> List[InputEvent]:
        """Predict likely next inputs based on current context and history"""
        predictions = []
        
        try:
            # Analyze current context
            simulation_state = current_context.get('simulation_state', {})
            user_activity = current_context.get('user_activity', 'idle')
            recent_inputs = current_context.get('recent_inputs', [])
            
            # Generate predictions based on patterns
            if user_activity == 'navigating':
                predictions.extend(self._predict_navigation_inputs(current_context))
            elif user_activity == 'creating':
                predictions.extend(self._predict_creation_inputs(current_context))
            elif user_activity == 'analyzing':
                predictions.extend(self._predict_analysis_inputs(current_context))
                
            # Add general predictions
            predictions.extend(self._predict_general_inputs(current_context))
            
            # Update behavior model
            self._update_behavior_model(current_context, predictions)
            
        except Exception as e:
            print(f"AI prediction error: {e}")
            
        return predictions
        
    def _predict_navigation_inputs(self, context: Dict[str, Any]) -> List[InputEvent]:
        """Predict navigation-related inputs"""
        predictions = []
        
        # Predict camera movements
        if context.get('camera_moving', False):
            predictions.append(InputEvent(
                event_id=f"pred_nav_{int(time.time()*1000)}",
                modality=InputModality.AI_PREDICTIVE,
                timestamp=time.time() + 0.5,  # 500ms in future
                data={
                    'type': 'camera_movement',
                    'direction': context.get('last_movement_direction', 'forward'),
                    'intensity': 0.7,
                    'duration': 1.0
                },
                confidence=0.75,
                source='ai_navigation_predictor'
            ))
            
        return predictions
        
    def _predict_creation_inputs(self, context: Dict[str, Any]) -> List[InputEvent]:
        """Predict creation-related inputs"""
        predictions = []
        
        # Predict object creation patterns
        creation_history = context.get('creation_history', [])
        if creation_history:
            last_creation = creation_history[-1]
            predictions.append(InputEvent(
                event_id=f"pred_create_{int(time.time()*1000)}",
                modality=InputModality.AI_PREDICTIVE,
                timestamp=time.time() + 1.0,
                data={
                    'type': 'object_creation',
                    'object_type': last_creation.get('type', 'particle'),
                    'position': self._predict_next_position(context),
                    'parameters': last_creation.get('parameters', {})
                },
                confidence=0.65,
                source='ai_creation_predictor'
            ))
            
        return predictions
        
    def _predict_analysis_inputs(self, context: Dict[str, Any]) -> List[InputEvent]:
        """Predict analysis-related inputs"""
        predictions = []
        
        # Predict data query patterns
        analysis_focus = context.get('analysis_focus', {})
        if analysis_focus:
            predictions.append(InputEvent(
                event_id=f"pred_analysis_{int(time.time()*1000)}",
                modality=InputModality.AI_PREDICTIVE,
                timestamp=time.time() + 2.0,
                data={
                    'type': 'data_query',
                    'query_type': 'performance_metrics',
                    'time_range': 'recent',
                    'confidence': 0.7
                },
                confidence=0.8,
                source='ai_analysis_predictor'
            ))
            
        return predictions
        
    def _predict_general_inputs(self, context: Dict[str, Any]) -> List[InputEvent]:
        """Predict general interaction patterns"""
        predictions = []
        
        # Predict interface interactions based on gaze
        gaze_data = context.get('gaze_data')
        if gaze_data and hasattr(gaze_data, 'gaze_point'):
            gaze_x, gaze_y = gaze_data.gaze_point
            
            # Predict button clicks based on gaze dwell time
            if context.get('gaze_dwell_time', 0) > 1.0:  # 1 second dwell
                predictions.append(InputEvent(
                    event_id=f"pred_click_{int(time.time()*1000)}",
                    modality=InputModality.AI_PREDICTIVE,
                    timestamp=time.time() + 0.3,
                    data={
                        'type': 'mouse_click',
                        'button': 'left',
                        'position': (gaze_x, gaze_y),
                        'confidence': 0.85
                    },
                    confidence=0.8,
                    source='ai_gaze_predictor'
                ))
                
        return predictions
        
    def _predict_next_position(self, context: Dict[str, Any]) -> Tuple[float, float, float]:
        """Predict next object creation position"""
        # Simple linear prediction based on recent positions
        recent_positions = context.get('recent_positions', [])
        if len(recent_positions) < 2:
            return (0, 0, 0)
            
        # Calculate average movement
        movements = []
        for i in range(1, len(recent_positions)):
            move = np.array(recent_positions[i]) - np.array(recent_positions[i-1])
            movements.append(move)
            
        if movements:
            avg_movement = np.mean(movements, axis=0)
            last_position = recent_positions[-1]
            return tuple(np.array(last_position) + avg_movement)
            
        return (0, 0, 0)
        
    def _update_behavior_model(self, context: Dict[str, Any], predictions: List[InputEvent]):
        """Update user behavior model based on actual inputs"""
        user_id = context.get('user_id', 'default')
        if user_id not in self.user_behavior_model:
            self.user_behavior_model[user_id] = {
                'input_patterns': {},
                'preferences': {},
                'skill_level': 'intermediate'
            }
            
        # Record prediction for later accuracy assessment
        for prediction in predictions:
            self.prediction_history.append({
                'timestamp': time.time(),
                'prediction': prediction,
                'context': context,
                'user_id': user_id
            })

class HapticFeedbackSystem:
    """Advanced haptic feedback system"""
    
    def __init__(self):
        self.feedback_devices = {}
        self.feedback_patterns = {
            'click': {'duration': 0.1, 'intensity': 0.7},
            'confirm': {'duration': 0.2, 'intensity': 0.8},
            'warning': {'duration': 0.5, 'intensity': 0.9, 'pattern': [0.1, 0.1, 0.1]},
            'error': {'duration': 1.0, 'intensity': 1.0, 'pattern': [0.2, 0.1, 0.2, 0.1]},
            'notification': {'duration': 0.3, 'intensity': 0.6}
        }
        
    def register_device(self, device_id: str, device_type: str, capabilities: Dict[str, Any]):
        """Register a haptic feedback device"""
        self.feedback_devices[device_id] = {
            'type': device_type,
            'capabilities': capabilities,
            'connected': True
        }
        print(f"Haptic device registered: {device_id} ({device_type})")
        
    def provide_feedback(self, feedback_type: str, intensity: float = None, 
                        device_id: str = None):
        """Provide haptic feedback"""
        if feedback_type not in self.feedback_patterns:
            print(f"Unknown feedback type: {feedback_type}")
            return
            
        pattern = self.feedback_patterns[feedback_type].copy()
        if intensity is not None:
            pattern['intensity'] = intensity
            
        # Send to all devices if none specified
        devices = [device_id] if device_id else list(self.feedback_devices.keys())
        
        for dev_id in devices:
            if dev_id in self.feedback_devices and self.feedback_devices[dev_id]['connected']:
                self._send_feedback_to_device(dev_id, pattern)
                
    def _send_feedback_to_device(self, device_id: str, pattern: Dict[str, Any]):
        """Send feedback pattern to specific device"""
        # In a real implementation, this would communicate with the hardware
        print(f"Sending haptic feedback to {device_id}: {pattern}")

class AdvancedInteractiveInputSystem:
    """
    Advanced Interactive Input System
    Multi-modal input system with fusion, prediction, and haptic feedback
    """
    
    def __init__(self, simulation_system: Any = None):
        self.simulation_system = simulation_system
        self.is_active = False
        
        # Initialize input subsystems
        self.gesture_recognizer = AdvancedGestureRecognizer()
        self.voice_recognizer = AdvancedVoiceRecognizer()
        self.eye_tracker = EyeTrackingSystem()
        self.bci_simulator = BCISimulator()
        self.input_fusion = MultiModalInputFusion()
        self.ai_predictor = AIPredictiveInput()
        self.haptic_feedback = HapticFeedbackSystem()
        
        # Input processing
        self.input_queue = deque()
        self.processing_thread = None
        self.event_handlers = {}
        
        # State tracking
        self.current_context = {}
        self.user_preferences = {}
        self.input_statistics = {
            'total_events': 0,
            'modality_usage': {},
            'average_confidence': 0.0,
            'prediction_accuracy': 0.0
        }
        
        # Register default event handlers
        self._register_default_handlers()
        
        print("Advanced Interactive Input System initialized")
        
    def _register_default_handlers(self):
        """Register default input event handlers"""
        self.register_event_handler('mouse_click', self._handle_mouse_click)
        self.register_event_handler('keyboard_input', self._handle_keyboard_input)
        self.register_event_handler('voice_command', self._handle_voice_command)
        self.register_event_handler('gesture', self._handle_gesture)
        self.register_event_handler('gaze_input', self._handle_gaze_input)
        self.register_event_handler('bci_command', self._handle_bci_command)
        self.register_event_handler('ai_prediction', self._handle_ai_prediction)
        
    def start_system(self):
        """Start the complete input system"""
        if self.is_active:
            print("Input system already active")
            return
            
        self.is_active = True
        
        # Start voice recognition
        self.voice_recognizer.start_listening(self._on_voice_command)
        
        # Start eye tracking
        self.eye_tracker.start_tracking()
        
        # Start BCI (simulated)
        self.bci_simulator.connect()
        
        # Start input processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        # Initialize haptic devices
        self._initialize_haptic_devices()
        
        print("Advanced Interactive Input System started")
        
    def stop_system(self):
        """Stop the input system"""
        self.is_active = False
        
        # Stop subsystems
        self.voice_recognizer.stop_listening()
        self.eye_tracker.stop_tracking()
        self.bci_simulator.disconnect()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
        print("Advanced Interactive Input System stopped")
        
    def register_event_handler(self, event_type: str, handler: Callable[[InputEvent], None]):
        """Register handler for specific event types"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        
    def process_input_event(self, event: InputEvent):
        """Process a single input event"""
        if not self.is_active:
            return
            
        self.input_queue.append(event)
        self.input_statistics['total_events'] += 1
        
        # Update modality usage statistics
        modality = event.modality.value
        self.input_statistics['modality_usage'][modality] = \
            self.input_statistics['modality_usage'].get(modality, 0) + 1
            
    def _processing_loop(self):
        """Main input processing loop"""
        while self.is_active:
            try:
                # Process all queued events
                while self.input_queue:
                    event = self.input_queue.popleft()
                    self._process_single_event(event)
                    
                # Update context for AI prediction
                self._update_current_context()
                
                # Generate AI predictions
                predictions = self.ai_predictor.predict_next_input(self.current_context)
                for prediction in predictions:
                    self.process_input_event(prediction)
                    
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Input processing error: {e}")
                time.sleep(0.1)
                
    def _process_single_event(self, event: InputEvent):
        """Process a single input event"""
        try:
            # Update event confidence based on context
            event.confidence = self._adjust_confidence(event)
            
            # Apply multi-modal fusion if multiple recent events
            recent_events = self._get_recent_events(0.5)  # Last 500ms
            if len(recent_events) > 1:
                event = self.input_fusion.fuse_inputs(recent_events + [event])
                
            # Route to appropriate handlers
            event_type = self._determine_event_type(event)
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"Event handler error: {e}")
                        
            # Provide haptic feedback if appropriate
            self._provide_contextual_feedback(event)
            
            # Mark event as processed
            event.processed = True
            
        except Exception as e:
            print(f"Event processing error: {e}")
            
    def _adjust_confidence(self, event: InputEvent) -> float:
        """Adjust event confidence based on context"""
        base_confidence = event.confidence
        
        # Reduce confidence for improbable events
        if event.modality == InputModality.BCI and event.data.get('intensity', 0) < 0.3:
            base_confidence *= 0.7
            
        # Increase confidence for consistent patterns
        recent_similar = self._count_similar_events(event, 2.0)  # Last 2 seconds
        if recent_similar > 0:
            base_confidence = min(0.95, base_confidence * (1 + 0.1 * recent_similar))
            
        return base_confidence
        
    def _count_similar_events(self, event: InputEvent, time_window: float) -> int:
        """Count similar events in recent history"""
        count = 0
        current_time = time.time()
        
        for past_event in self.input_queue:
            if current_time - past_event.timestamp > time_window:
                continue
                
            if (past_event.modality == event.modality and 
                self._events_similar(past_event, event)):
                count += 1
                
        return count
        
    def _events_similar(self, event1: InputEvent, event2: InputEvent) -> bool:
        """Check if two events are similar"""
        # Simple similarity check - could be enhanced
        if event1.modality != event2.modality:
            return False
            
        # Check if data structures are similar
        keys1 = set(event1.data.keys())
        keys2 = set(event2.data.keys())
        return len(keys1.intersection(keys2)) > 0
        
    def _get_recent_events(self, time_window: float) -> List[InputEvent]:
        """Get events from recent time window"""
        current_time = time.time()
        return [e for e in self.input_queue 
                if current_time - e.timestamp <= time_window]
                
    def _determine_event_type(self, event: InputEvent) -> str:
        """Determine the type of input event"""
        data = event.data
        
        if event.modality == InputModality.VOICE:
            return 'voice_command'
        elif event.modality == InputModality.GESTURE:
            return 'gesture'
        elif event.modality == InputModality.EYE_TRACKING:
            return 'gaze_input'
        elif event.modality == InputModality.BCI:
            return 'bci_command'
        elif event.modality == InputModality.AI_PREDICTIVE:
            return 'ai_prediction'
        elif 'click' in data or 'button' in data:
            return 'mouse_click'
        elif 'key' in data:
            return 'keyboard_input'
        else:
            return 'generic_input'
            
    def _update_current_context(self):
        """Update current interaction context"""
        self.current_context = {
            'timestamp': time.time(),
            'recent_inputs': list(self.input_queue),
            'input_statistics': self.input_statistics.copy(),
            'simulation_state': self._get_simulation_state(),
            'user_activity': self._infer_user_activity(),
            'gaze_data': self.eye_tracker.get_gaze_data(),
            'bci_data': self.bci_simulator.get_bci_data()
        }
        
    def _get_simulation_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        if self.simulation_system:
            return {
                'running': getattr(self.simulation_system, 'running', False),
                'simulation_type': getattr(self.simulation_system, 'current_simulation', None),
                'particle_count': 0  # Would be actual count from simulation
            }
        return {}
        
    def _infer_user_activity(self) -> str:
        """Infer current user activity from input patterns"""
        recent_events = self._get_recent_events(5.0)  # Last 5 seconds
        
        if not recent_events:
            return 'idle'
            
        # Analyze event patterns
        gesture_count = sum(1 for e in recent_events if e.modality == InputModality.GESTURE)
        voice_count = sum(1 for e in recent_events if e.modality == InputModality.VOICE)
        navigation_count = sum(1 for e in recent_events if 'navigation' in str(e.data))
        
        if gesture_count > 3:
            return 'gesturing'
        elif voice_count > 2:
            return 'commanding'
        elif navigation_count > 2:
            return 'navigating'
        else:
            return 'interacting'
            
    def _provide_contextual_feedback(self, event: InputEvent):
        """Provide contextual haptic feedback"""
        if event.confidence > 0.8:
            self.haptic_feedback.provide_feedback('confirm')
        elif event.confidence < 0.4:
            self.haptic_feedback.provide_feedback('warning')
        elif event.modality == InputModality.VOICE:
            self.haptic_feedback.provide_feedback('notification')
            
    def _initialize_haptic_devices(self):
        """Initialize haptic feedback devices"""
        # Simulate device registration
        self.haptic_feedback.register_device(
            'controller_1', 'vibration_controller', 
            {'vibration_intensity': True, 'pattern_playback': True}
        )
        self.haptic_feedback.register_device(
            'wearable_1', 'haptic_suit', 
            {'localized_feedback': True, 'temperature': False}
        )
        
    def _on_voice_command(self, voice_command: VoiceCommand):
        """Callback for voice commands"""
        event = InputEvent(
            event_id=f"voice_{int(time.time()*1000)}",
            modality=InputModality.VOICE,
            timestamp=time.time(),
            data={
                'command_type': voice_command.command_type.value,
                'text': voice_command.text,
                'intent': voice_command.intent,
                'parameters': voice_command.parameters
            },
            confidence=voice_command.confidence,
            source='voice_recognizer'
        )
        self.process_input_event(event)
        
    # Event handler implementations
    def _handle_mouse_click(self, event: InputEvent):
        """Handle mouse click events"""
        position = event.data.get('position', (0, 0))
        button = event.data.get('button', 'left')
        
        print(f"Mouse {button} click at {position} (confidence: {event.confidence:.2f})")
        
        if self.simulation_system:
            # Convert to simulation coordinates and add particle
            world_pos = self.simulation_system.screen_to_world(position)
            self.simulation_system.current_simulation.add_particle_at_position(world_pos)
            
    def _handle_keyboard_input(self, event: InputEvent):
        """Handle keyboard input events"""
        key = event.data.get('key', '')
        modifiers = event.data.get('modifiers', [])
        
        print(f"Keyboard input: {key} with modifiers {modifiers}")
        
        # Map keys to simulation actions
        key_actions = {
            'space': 'pause_toggle',
            'r': 'reset',
            'c': 'clear',
            '1': 'switch_basic',
            '2': 'switch_fluid',
            '3': 'switch_quantum'
        }
        
        if key in key_actions and self.simulation_system:
            action = key_actions[key]
            if hasattr(self.simulation_system, action):
                getattr(self.simulation_system, action)()
                
    def _handle_voice_command(self, event: InputEvent):
        """Handle voice command events"""
        command_type = event.data.get('command_type')
        intent = event.data.get('intent')
        parameters = event.data.get('parameters', {})
        
        print(f"Voice command: {intent} with parameters {parameters}")
        
        # Execute voice command in simulation
        if self.simulation_system and intent == 'simulation_control':
            action = parameters.get('modification', 'toggle')
            if action == 'start':
                self.simulation_system.running = True
            elif action == 'stop':
                self.simulation_system.running = False
                
    def _handle_gesture(self, event: InputEvent):
        """Handle gesture events"""
        gesture_type = event.data.get('gesture_type')
        print(f"Gesture detected: {gesture_type}")
        
        # Map gestures to simulation actions
        gesture_actions = {
            'swipe_left': 'camera_left',
            'swipe_right': 'camera_right',
            'swipe_up': 'camera_up',
            'swipe_down': 'camera_down',
            'pinch_in': 'zoom_out',
            'pinch_out': 'zoom_in'
        }
        
        if gesture_type in gesture_actions and self.simulation_system:
            action = gesture_actions[gesture_type]
            if hasattr(self.simulation_system, action):
                getattr(self.simulation_system, action)()
                
    def _handle_gaze_input(self, event: InputEvent):
        """Handle gaze input events"""
        gaze_point = event.data.get('gaze_point', (0, 0))
        print(f"Gaze at: {gaze_point}")
        
        # Implement gaze-based interaction
        if self.simulation_system:
            # Could use gaze for attention-based rendering or automatic camera control
            pass
            
    def _handle_bci_command(self, event: InputEvent):
        """Handle BCI command events"""
        mental_command = event.data.get('mental_command')
        intensity = event.data.get('intensity', 0)
        
        print(f"BCI command: {mental_command} (intensity: {intensity:.2f})")
        
        # Map BCI commands to simulation actions
        bci_actions = {
            'focus': 'increase_speed',
            'relax': 'decrease_speed',
            'push': 'add_force',
            'pull': 'reverse_force'
        }
        
        if mental_command in bci_actions and self.simulation_system:
            action = bci_actions[mental_command]
            if hasattr(self.simulation_system, action):
                getattr(self.simulation_system, action)(intensity)
                
    def _handle_ai_prediction(self, event: InputEvent):
        """Handle AI-predicted input events"""
        prediction_type = event.data.get('type')
        print(f"AI prediction: {prediction_type}")
        
        # Use predictions to pre-compute or optimize simulation
        if prediction_type == 'object_creation' and self.simulation_system:
            # Pre-allocate resources for predicted object creation
            position = event.data.get('position', (0, 0, 0))
            # Could prepare particle system for new objects
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_active': self.is_active,
            'input_statistics': self.input_statistics,
            'fusion_statistics': self.input_fusion.get_fusion_statistics(),
            'active_modalities': [
                modality for modality, count in self.input_statistics['modality_usage'].items()
                if count > 0
            ],
            'user_activity': self.current_context.get('user_activity', 'unknown'),
            'haptic_devices': list(self.haptic_feedback.feedback_devices.keys())
        }

# Example usage and integration
def demo_advanced_input_system():
    """Demonstrate the advanced input system"""
    input_system = AdvancedInteractiveInputSystem()
    
    print("Starting Advanced Interactive Input System...")
    input_system.start_system()
    
    # Simulate various input events
    print("\nSimulating multi-modal inputs...")
    
    # Simulate voice command
    voice_event = InputEvent(
        event_id="test_voice_1",
        modality=InputModality.VOICE,
        timestamp=time.time(),
        data={
            'command_type': 'simulation_control',
            'text': 'start simulation',
            'intent': 'start_simulation',
            'parameters': {'action': 'start'}
        },
        confidence=0.9,
        source='test'
    )
    input_system.process_input_event(voice_event)
    
    # Simulate gesture
    gesture_event = InputEvent(
        event_id="test_gesture_1",
        modality=InputModality.GESTURE,
        timestamp=time.time(),
        data={
            'gesture_type': 'swipe_right',
            'hand_landmarks': [(0.1, 0.2, 0.3)] * 21,  # 21 hand landmarks
            'bounding_box': (0.1, 0.1, 0.2, 0.3),
            'velocity': (0.5, 0.1, 0.0)
        },
        confidence=0.8,
        source='test'
    )
    input_system.process_input_event(gesture_event)
    
    # Simulate mouse click
    mouse_event = InputEvent(
        event_id="test_mouse_1",
        modality=InputModality.TOUCH,
        timestamp=time.time(),
        data={
            'type': 'mouse_click',
            'button': 'left',
            'position': (0.5, 0.5),
            'modifiers': []
        },
        confidence=1.0,
        source='test'
    )
    input_system.process_input_event(mouse_event)
    
    # Let system process events
    time.sleep(2)
    
    # Get system status
    status = input_system.get_system_status()
    print(f"\nInput System Status:")
    print(f"Active: {status['system_active']}")
    print(f"Total Events: {status['input_statistics']['total_events']}")
    print(f"Active Modalities: {status['active_modalities']}")
    print(f"User Activity: {status['user_activity']}")
    
    # Stop system
    print("\nStopping input system...")
    input_system.stop_system()
    
    return input_system

if __name__ == "__main__":
    # Run demonstration
    demo_system = demo_advanced_input_system()