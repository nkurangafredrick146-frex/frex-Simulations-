#!/usr/bin/env python3
"""
Audio Integration Module for frex Simulations
Real-time audio synthesis and synchronization with physics simulations
"""

import pygame
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import math

@dataclass
class AudioEvent:
    """Represents a single audio event to be synthesized"""
    event_type: str  # 'collision', 'explosion', 'fluid', 'custom'
    position: np.ndarray
    intensity: float
    frequency: float
    duration: float
    timestamp: float
    properties: Dict

class AudioParticle:
    """Audio representation of a particle"""
    def __init__(self, particle_id):
        self.particle_id = particle_id
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.audio_frequency = 440.0
        self.audio_volume = 0.0
        self.life_time = 0.0
        self.is_active = False

class AudioSystem:
    """Main audio system handling real-time synthesis and playback"""
    
    def __init__(self):
        # Audio configuration
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.num_channels = 2
        
        # Audio state
        self.master_volume = 0.5
        self.is_initialized = False
        self.is_playing = False
        
        # Event management
        self.audio_events = queue.Queue()
        self.active_sounds = []
        self.particle_audio_map = {}
        
        # Synthesis parameters
        self.time = 0.0
        self.phase = 0.0
        
        # Frequency ranges for different event types
        self.frequency_ranges = {
            'collision': (200, 800),
            'explosion': (50, 400),
            'fluid': (100, 600),
            'custom': (100, 1000)
        }
        
        # Audio buffers
        self.output_buffer = deque(maxlen=10)
        self.current_chunk = np.zeros((self.chunk_size, self.num_channels), dtype=np.float32)
        
        # Threading
        self.audio_thread = None
        self.shutdown_flag = threading.Event()
        
        # Audio effects
        self.reverb_buffer = deque(maxlen=int(self.sample_rate * 0.5))  # 0.5 second reverb
        self.lowpass_cutoff = 20000
        self.highpass_cutoff = 20
        
        # Real-time analysis
        self.spectrum_analyzer = SpectrumAnalyzer(self.sample_rate, self.chunk_size)
        self.audio_features = {}
        
        print("Audio System initialized")

    def initialize(self):
        """Initialize the audio system with pygame mixer"""
        try:
            pygame.mixer.init(
                frequency=self.sample_rate,
                size=-16,
                channels=self.num_channels,
                buffer=self.chunk_size
            )
            
            # Start audio generation thread
            self.audio_thread = threading.Thread(target=self._audio_generation_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            self.is_initialized = True
            self.is_playing = True
            
            print("Audio System fully initialized and running")
            
        except pygame.error as e:
            print(f"Failed to initialize audio: {e}")
            self.is_initialized = False

    def update(self, simulation):
        """Update audio based on simulation state"""
        if not self.is_initialized or not self.is_playing:
            return
            
        # Process simulation events for audio generation
        self._process_simulation_events(simulation)
        
        # Generate continuous audio based on particle states
        self._generate_particle_audio(simulation)
        
        # Update audio features for visualization
        self._update_audio_features()

    def _process_simulation_events(self, simulation):
        """Process simulation events and create audio events"""
        if hasattr(simulation, 'get_audio_events'):
            events = simulation.get_audio_events()
            for event in events:
                self.add_audio_event(event)

    def _generate_particle_audio(self, simulation):
        """Generate continuous audio from particle states"""
        if not hasattr(simulation, 'particles'):
            return
            
        # Process each particle for audio contribution
        for particle in simulation.particles:
            particle_id = id(particle)
            
            # Create or update audio particle
            if particle_id not in self.particle_audio_map:
                self.particle_audio_map[particle_id] = AudioParticle(particle_id)
            
            audio_particle = self.particle_audio_map[particle_id]
            
            # Update particle state
            if hasattr(particle, 'position'):
                audio_particle.position = np.array(particle.position)
            if hasattr(particle, 'velocity'):
                audio_particle.velocity = np.array(particle.velocity)
                
            # Calculate audio parameters from physics
            speed = np.linalg.norm(audio_particle.velocity)
            audio_particle.audio_volume = min(speed * 0.1, 1.0)
            
            # Map speed to frequency (Doppler-like effect)
            base_freq = 220.0
            audio_particle.audio_frequency = base_freq * (1.0 + speed * 0.5)
            
            audio_particle.life_time += 1/60.0  # Assuming 60 FPS
            audio_particle.is_active = True
            
            # Create continuous audio event for active particles
            if audio_particle.audio_volume > 0.01:
                event = AudioEvent(
                    event_type='custom',
                    position=audio_particle.position,
                    intensity=audio_particle.audio_volume,
                    frequency=audio_particle.audio_frequency,
                    duration=0.1,
                    timestamp=time.time(),
                    properties={'particle_id': particle_id}
                )
                self.add_audio_event(event)

    def add_audio_event(self, event: AudioEvent):
        """Add an audio event to the processing queue"""
        if self.is_initialized and self.is_playing:
            self.audio_events.put(event)

    def _audio_generation_loop(self):
        """Main audio generation loop running in separate thread"""
        print("Audio generation thread started")
        
        while not self.shutdown_flag.is_set():
            try:
                # Generate audio chunk
                audio_chunk = self._generate_audio_chunk()
                
                # Convert to pygame sound and play
                if self.is_playing:
                    self._play_audio_chunk(audio_chunk)
                    
                # Small sleep to prevent excessive CPU usage
                time.sleep(self.chunk_size / self.sample_rate)
                
            except Exception as e:
                print(f"Audio generation error: {e}")
                time.sleep(0.1)

    def _generate_audio_chunk(self) -> np.ndarray:
        """Generate one chunk of audio data"""
        chunk = np.zeros((self.chunk_size, self.num_channels), dtype=np.float32)
        
        # Process audio events
        self._process_audio_events(chunk)
        
        # Generate continuous tones
        self._generate_continuous_audio(chunk)
        
        # Apply audio effects
        self._apply_audio_effects(chunk)
        
        # Update time
        self.time += self.chunk_size / self.sample_rate
        
        return chunk

    def _process_audio_events(self, chunk: np.ndarray):
        """Process all pending audio events"""
        processed_events = 0
        max_events_per_chunk = 10  # Limit for performance
        
        while (not self.audio_events.empty() and 
               processed_events < max_events_per_chunk):
            try:
                event = self.audio_events.get_nowait()
                self._synthesize_event(chunk, event)
                processed_events += 1
            except queue.Empty:
                break

    def _synthesize_event(self, chunk: np.ndarray, event: AudioEvent):
        """Synthesize audio for a specific event"""
        samples = np.arange(chunk.shape[0])
        time_points = samples / self.sample_rate
        
        # Generate base waveform based on event type
        if event.event_type == 'collision':
            waveform = self._generate_collision_sound(time_points, event)
        elif event.event_type == 'explosion':
            waveform = self._generate_explosion_sound(time_points, event)
        elif event.event_type == 'fluid':
            waveform = self._generate_fluid_sound(time_points, event)
        else:
            waveform = self._generate_custom_sound(time_points, event)
        
        # Apply stereo panning based on position
        stereo_mix = self._apply_stereo_panning(waveform, event.position)
        
        # Add to output chunk
        chunk += stereo_mix * event.intensity * self.master_volume

    def _generate_collision_sound(self, time_points: np.ndarray, event: AudioEvent) -> np.ndarray:
        """Generate collision impact sound"""
        # Short impact noise burst
        noise = np.random.normal(0, 1, len(time_points))
        
        # Envelope - quick attack, fast decay
        envelope = np.exp(-time_points * 50)  # Fast decay
        
        # Filter the noise
        b, a = signal.butter(2, event.frequency / (self.sample_rate / 2), 'low')
        filtered_noise = signal.lfilter(b, a, noise)
        
        return filtered_noise * envelope

    def _generate_explosion_sound(self, time_points: np.ndarray, event: AudioEvent) -> np.ndarray:
        """Generate explosion sound with low frequency content"""
        # Multiple frequency components
        fundamental = np.sin(2 * np.pi * event.frequency * 0.5 * time_points)
        harmonic1 = np.sin(2 * np.pi * event.frequency * time_points) * 0.5
        harmonic2 = np.sin(2 * np.pi * event.frequency * 2 * time_points) * 0.25
        
        # Noise component
        noise = np.random.normal(0, 0.3, len(time_points))
        
        # Complex envelope - slow attack, long decay
        attack = np.minimum(time_points * 100, 1.0)
        decay = np.exp(-time_points * 10)
        envelope = attack * decay
        
        combined = (fundamental + harmonic1 + harmonic2 + noise) * envelope
        return combined

    def _generate_fluid_sound(self, time_points: np.ndarray, event: AudioEvent) -> np.ndarray:
        """Generate fluid bubbling/waves sound"""
        # Multiple sine waves with slightly detuned frequencies
        base_freq = event.frequency
        waves = []
        
        for i in range(5):
            detune = 1.0 + (i - 2) * 0.02  # Slight detuning
            wave = np.sin(2 * np.pi * base_freq * detune * time_points)
            waves.append(wave * (1.0 / (1 + i)))
        
        combined = sum(waves) / len(waves)
        
        # Modulate amplitude for bubbling effect
        bubble_rate = 10  # bubbles per second
        bubble_mod = (1 + np.sin(2 * np.pi * bubble_rate * time_points)) * 0.5
        
        # Filter to make it more fluid-like
        b, a = signal.butter(4, 1000 / (self.sample_rate / 2), 'low')
        filtered = signal.lfilter(b, a, combined * bubble_mod)
        
        return filtered

    def _generate_custom_sound(self, time_points: np.ndarray, event: AudioEvent) -> np.ndarray:
        """Generate custom sound based on event properties"""
        # Basic sine wave with modulation
        base_wave = np.sin(2 * np.pi * event.frequency * time_points)
        
        # Frequency modulation for more interesting sound
        fm_depth = event.properties.get('fm_depth', 0.1)
        fm_rate = event.properties.get('fm_rate', 5.0)
        fm = fm_depth * np.sin(2 * np.pi * fm_rate * time_points)
        
        modulated_wave = np.sin(2 * np.pi * (event.frequency + fm) * time_points)
        
        # Amplitude envelope
        attack = np.minimum(time_points * 100, 1.0)
        decay = np.exp(-time_points * 20)
        envelope = attack * decay
        
        return modulated_wave * envelope

    def _generate_continuous_audio(self, chunk: np.ndarray):
        """Generate continuous background/ambient audio"""
        time_points = np.arange(chunk.shape[0]) / self.sample_rate + self.time
        
        # Low-frequency ambient drone
        drone_freq = 55.0
        drone = np.sin(2 * np.pi * drone_freq * time_points) * 0.05
        
        # Gentle noise for atmosphere
        noise = np.random.normal(0, 0.02, chunk.shape[0])
        
        # Filter noise
        b, a = signal.butter(2, 500 / (self.sample_rate / 2), 'low')
        filtered_noise = signal.lfilter(b, a, noise)
        
        # Combine and add to both channels
        ambient = (drone + filtered_noise) * self.master_volume * 0.3
        chunk[:, 0] += ambient
        chunk[:, 1] += ambient

    def _apply_stereo_panning(self, mono_signal: np.ndarray, position: np.ndarray) -> np.ndarray:
        """Convert mono signal to stereo with panning based on position"""
        if len(position) < 2:
            return np.column_stack([mono_signal, mono_signal])
        
        # Simple left-right panning based on x position
        # Normalize x position to [-1, 1] range
        pan = np.clip(position[0] / 10.0, -1.0, 1.0)  # Assuming world coordinates ±10
        
        # Calculate left/right gains
        left_gain = max(0, 1 - pan)  # 1 when pan = -1 (full left), 0 when pan = 1 (full right)
        right_gain = max(0, 1 + pan)  # 0 when pan = -1, 1 when pan = 1
        
        # Apply gains
        stereo = np.column_stack([
            mono_signal * left_gain,
            mono_signal * right_gain
        ])
        
        return stereo

    def _apply_audio_effects(self, chunk: np.ndarray):
        """Apply audio effects like reverb and filtering"""
        # Simple reverb effect
        if len(self.reverb_buffer) > 0:
            reverb_mix = np.array(list(self.reverb_buffer)) * 0.3
            # Ensure same length
            min_len = min(len(chunk), len(reverb_mix))
            chunk[:min_len] += reverb_mix[:min_len]
        
        # Update reverb buffer
        self.reverb_buffer.extend(chunk.flatten())
        
        # Apply lowpass filter
        b, a = signal.butter(2, self.lowpass_cutoff / (self.sample_rate / 2), 'low')
        chunk[:, 0] = signal.lfilter(b, a, chunk[:, 0])
        chunk[:, 1] = signal.lfilter(b, a, chunk[:, 1])
        
        # Apply highpass filter to remove DC offset
        b, a = signal.butter(1, self.highpass_cutoff / (self.sample_rate / 2), 'high')
        chunk[:, 0] = signal.lfilter(b, a, chunk[:, 0])
        chunk[:, 1] = signal.lfilter(b, a, chunk[:, 1])
        
        # Clipping protection
        chunk = np.clip(chunk, -1.0, 1.0)

    def _play_audio_chunk(self, chunk: np.ndarray):
        """Play an audio chunk using pygame mixer"""
        try:
            # Convert to int16 for pygame
            int_chunk = (chunk * 32767).astype(np.int16)
            
            # Create pygame sound and play
            sound = pygame.sndarray.make_sound(int_chunk)
            sound.play()
            
        except Exception as e:
            print(f"Audio playback error: {e}")

    def _update_audio_features(self):
        """Update real-time audio analysis features"""
        if len(self.current_chunk) > 0:
            mono_signal = np.mean(self.current_chunk, axis=1)
            self.audio_features = self.spectrum_analyzer.analyze(mono_signal)

    def set_master_volume(self, volume: float):
        """Set master volume (0.0 to 1.0)"""
        self.master_volume = max(0.0, min(1.0, volume))

    def pause(self):
        """Pause audio playback"""
        self.is_playing = False
        pygame.mixer.pause()

    def resume(self):
        """Resume audio playback"""
        self.is_playing = True
        pygame.mixer.unpause()

    def export_audio(self, filename: str, duration: float = 10.0):
        """Export current audio to WAV file"""
        try:
            total_samples = int(duration * self.sample_rate)
            export_buffer = np.zeros((total_samples, self.num_channels), dtype=np.float32)
            
            # Generate audio for export
            for i in range(0, total_samples, self.chunk_size):
                chunk_samples = min(self.chunk_size, total_samples - i)
                chunk = self._generate_audio_chunk()
                export_buffer[i:i+chunk_samples] = chunk[:chunk_samples]
            
            # Convert to int16 and save
            int_buffer = (export_buffer * 32767).astype(np.int16)
            wavfile.write(filename, self.sample_rate, int_buffer)
            
            print(f"Audio exported to {filename}")
            
        except Exception as e:
            print(f"Audio export failed: {e}")

    def get_audio_features(self) -> Dict:
        """Get current audio analysis features"""
        return self.audio_features.copy()

    def cleanup(self):
        """Clean up audio system resources"""
        print("Cleaning up audio system...")
        self.shutdown_flag.set()
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
            
        pygame.mixer.quit()
        self.is_initialized = False
        print("Audio system cleaned up")

class SpectrumAnalyzer:
    """Real-time audio spectrum analysis"""
    
    def __init__(self, sample_rate: int, chunk_size: int):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Frequency bands for analysis
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }

    def analyze(self, signal: np.ndarray) -> Dict:
        """Analyze audio signal and extract features"""
        if len(signal) == 0:
            return {}
            
        # Compute FFT
        fft = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(len(signal), 1/self.sample_rate)
        magnitudes = np.abs(fft)
        
        features = {}
        
        # Calculate energy in each frequency band
        for band_name, (low_freq, high_freq) in self.bands.items():
            mask = (frequencies >= low_freq) & (frequencies < high_freq)
            if np.any(mask):
                band_energy = np.mean(magnitudes[mask])
                features[f'{band_name}_energy'] = float(band_energy)
        
        # Overall features
        features['rms'] = float(np.sqrt(np.mean(signal**2)))
        features['peak'] = float(np.max(np.abs(signal)))
        
        # Spectral centroid (brightness)
        if np.sum(magnitudes) > 0:
            spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
            features['spectral_centroid'] = float(spectral_centroid)
        
        return features

# Example usage and testing
if __name__ == "__main__":
    # Test the audio system
    audio = AudioSystem()
    audio.initialize()
    
    # Test events
    test_event = AudioEvent(
        event_type='collision',
        position=np.array([1.0, 0.0, 0.0]),
        intensity=0.8,
        frequency=440.0,
        duration=0.5,
        timestamp=time.time(),
        properties={}
    )
    
    audio.add_audio_event(test_event)
    
    # Let it play for a bit
    time.sleep(2)
    
    audio.cleanup()