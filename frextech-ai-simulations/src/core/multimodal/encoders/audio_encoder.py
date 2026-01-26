"""
Audio encoders for processing audio inputs.
Includes Wav2Vec2, HuBERT, Whisper, and custom audio encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
import logging
import warnings
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    HubertModel,
    HubertProcessor,
    WhisperModel,
    WhisperProcessor,
    WhisperFeatureExtractor,
    AutoModel,
    AutoProcessor,
    AutoFeatureExtractor
)
import librosa
import soundfile as sf

from ..alignment import EncoderConfig, EncoderOutput

logger = logging.getLogger(__name__)

@dataclass
class AudioEncoderConfig(EncoderConfig):
    """Configuration for audio encoders."""
    
    # Audio-specific parameters
    sample_rate: int = 16000
    num_mel_bins: int = 80
    num_mfcc: int = 13
    hop_length: int = 160
    win_length: int = 400
    n_fft: int = 512
    fmin: float = 0.0
    fmax: float = 8000.0
    
    # Feature extraction
    feature_type: str = "wav2vec2"  # "wav2vec2", "hubert", "whisper", "mfcc", "mel", "spectrogram"
    use_preprocessor: bool = True
    return_attention_mask: bool = True
    
    # Audio augmentation
    noise_level: float = 0.0
    time_stretch: float = 1.0
    pitch_shift: float = 0.0
    
    # Model-specific
    layer_to_extract: int = -1  # Which layer to extract features from
    pooling_method: str = "mean"  # "mean", "max", "attention", "last"
    
    def __post_init__(self):
        """Set encoder type."""
        self.encoder_type = "audio"

class AudioEncoder(nn.Module):
    """Base class for audio encoders."""
    
    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.preprocessor = None
        self.feature_extractor = None
        self.model = None
        self.output_dim = config.output_dim
        
        # Initialize preprocessing
        self._init_preprocessing()
        
        # Initialize model
        self._initialize_model()
        
        # Freeze parameters if not trainable
        if not config.trainable:
            self._freeze_parameters()
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {config.model_name}")
    
    def _init_preprocessing(self):
        """Initialize audio preprocessing."""
        # Basic audio preprocessing transforms
        self._init_audio_transforms()
    
    def _init_audio_transforms(self):
        """Initialize audio transforms for feature extraction."""
        # These will be used for custom feature extraction
        self.mel_transform = None
        self.mfcc_transform = None
        
        if self.config.feature_type in ["mel", "mfcc", "spectrogram"]:
            try:
                import torchaudio
                # Mel spectrogram transform
                self.mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.config.sample_rate,
                    n_fft=self.config.n_fft,
                    win_length=self.config.win_length,
                    hop_length=self.config.hop_length,
                    n_mels=self.config.num_mel_bins,
                    f_min=self.config.fmin,
                    f_max=self.config.fmax
                )
                
                # MFCC transform
                if self.config.feature_type == "mfcc":
                    self.mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=self.config.sample_rate,
                        n_mfcc=self.config.num_mfcc,
                        melkwargs={
                            'n_fft': self.config.n_fft,
                            'win_length': self.config.win_length,
                            'hop_length': self.config.hop_length,
                            'n_mels': self.config.num_mel_bins,
                            'f_min': self.config.fmin,
                            'f_max': self.config.fmax
                        }
                    )
            except ImportError:
                logger.warning("torchaudio not installed, using librosa for feature extraction")
    
    def _initialize_model(self):
        """Initialize model - to be implemented by subclasses."""
        raise NotImplementedError
    
    def _freeze_parameters(self):
        """Freeze model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Froze all model parameters")
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
    
    def preprocess(self,
                  audio: Union[np.ndarray, torch.Tensor, str, List],
                  sample_rate: Optional[int] = None,
                  **kwargs) -> Dict[str, torch.Tensor]:
        """
        Preprocess audio for model input.
        
        Args:
            audio: Input audio - can be numpy array, torch tensor, file path, or list of these
            sample_rate: Sample rate of input audio (None to use config sample_rate)
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Dictionary of preprocessed audio features
        """
        if sample_rate is None:
            sample_rate = self.config.sample_rate
        
        # Handle different input types
        if isinstance(audio, str):
            # Load audio file
            try:
                waveform, sr = sf.read(audio)
                if sr != sample_rate:
                    # Resample if needed
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
                audio = torch.from_numpy(waveform).float()
            except Exception as e:
                logger.error(f"Failed to load audio file {audio}: {e}")
                raise
        
        elif isinstance(audio, np.ndarray):
            # Convert numpy array to tensor
            audio = torch.from_numpy(audio).float()
        
        elif isinstance(audio, list):
            # Process list of audio inputs
            processed = []
            for item in audio:
                item_processed = self.preprocess(item, sample_rate, **kwargs)
                if isinstance(item_processed, dict) and 'input_values' in item_processed:
                    processed.append(item_processed['input_values'])
                else:
                    processed.append(item_processed)
            
            # Stack if all have same shape
            if all(p.shape == processed[0].shape for p in processed):
                return {'input_values': torch.stack(processed)}
            else:
                return {'input_values': processed}
        
        # Ensure waveform is 2D: [batch_size, sequence_length] or 1D: [sequence_length]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Use preprocessor if available
        if self.preprocessor is not None and self.config.use_preprocessor:
            try:
                # Convert to numpy for HuggingFace processors
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.numpy() if audio.device == torch.device('cpu') else audio.cpu().numpy()
                else:
                    audio_np = audio
                
                # Process with HuggingFace preprocessor
                processed = self.preprocessor(
                    audio_np,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=self.config.return_attention_mask,
                    **kwargs
                )
                return processed
            except Exception as e:
                logger.warning(f"Preprocessor failed: {e}, falling back to manual preprocessing")
        
        # Manual preprocessing
        if self.config.feature_type == "raw":
            # Use raw waveform
            return {'input_values': audio}
        
        elif self.config.feature_type in ["mel", "mfcc", "spectrogram"]:
            # Extract spectral features
            features = self._extract_spectral_features(audio, sample_rate)
            return {'input_values': features}
        
        else:
            # Default: use raw waveform
            return {'input_values': audio}
    
    def _extract_spectral_features(self,
                                  audio: torch.Tensor,
                                  sample_rate: int) -> torch.Tensor:
        """Extract spectral features from audio."""
        if self.config.feature_type == "mel":
            if self.mel_transform is not None:
                # Use torchaudio
                features = self.mel_transform(audio)
                # Log compression
                features = torch.log(features + 1e-8)
            else:
                # Use librosa
                import librosa
                features = []
                for waveform in audio:
                    waveform_np = waveform.numpy() if waveform.device == torch.device('cpu') else waveform.cpu().numpy()
                    mel = librosa.feature.melspectrogram(
                        y=waveform_np,
                        sr=sample_rate,
                        n_fft=self.config.n_fft,
                        hop_length=self.config.hop_length,
                        win_length=self.config.win_length,
                        n_mels=self.config.num_mel_bins,
                        fmin=self.config.fmin,
                        fmax=self.config.fmax
                    )
                    mel = torch.from_numpy(np.log(mel + 1e-8)).float()
                    features.append(mel)
                features = torch.stack(features)
        
        elif self.config.feature_type == "mfcc":
            if self.mfcc_transform is not None:
                # Use torchaudio
                features = self.mfcc_transform(audio)
            else:
                # Use librosa
                import librosa
                features = []
                for waveform in audio:
                    waveform_np = waveform.numpy() if waveform.device == torch.device('cpu') else waveform.cpu().numpy()
                    mfcc = librosa.feature.mfcc(
                        y=waveform_np,
                        sr=sample_rate,
                        n_mfcc=self.config.num_mfcc,
                        n_fft=self.config.n_fft,
                        hop_length=self.config.hop_length,
                        n_mels=self.config.num_mel_bins,
                        fmin=self.config.fmin,
                        fmax=self.config.fmax
                    )
                    mfcc = torch.from_numpy(mfcc).float()
                    features.append(mfcc)
                features = torch.stack(features)
        
        elif self.config.feature_type == "spectrogram":
            # Compute spectrogram
            import librosa
            features = []
            for waveform in audio:
                waveform_np = waveform.numpy() if waveform.device == torch.device('cpu') else waveform.cpu().numpy()
                spectrogram = librosa.stft(
                    waveform_np,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.win_length
                )
                spectrogram = torch.from_numpy(np.abs(spectrogram)).float()
                features.append(spectrogram)
            features = torch.stack(features)
        
        else:
            raise ValueError(f"Unknown feature type: {self.config.feature_type}")
        
        return features
    
    def forward(self,
                audio: Union[np.ndarray, torch.Tensor, str, List],
                sample_rate: Optional[int] = None,
                return_dict: bool = True,
                **kwargs) -> EncoderOutput:
        """
        Forward pass through audio encoder.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate of input audio
            return_dict: Whether to return EncoderOutput dict
            **kwargs: Additional arguments for model
            
        Returns:
            EncoderOutput with audio features
        """
        # Preprocess audio
        processed = self.preprocess(audio, sample_rate, **kwargs)
        
        if 'input_values' not in processed:
            raise ValueError("Preprocessing did not produce 'input_values'")
        
        input_values = processed['input_values']
        attention_mask = processed.get('attention_mask', None)
        
        # Move to model device
        input_values = input_values.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        # Model forward pass
        model_kwargs = {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'output_hidden_states': True,
            'output_attentions': True,
            'return_dict': True
        }
        
        # Update with any additional kwargs
        model_kwargs.update(kwargs)
        
        outputs = self.model(**model_kwargs)
        
        # Extract features
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            # Extract from specific layer if specified
            if self.config.layer_to_extract >= 0:
                layer_idx = min(self.config.layer_to_extract, len(outputs.hidden_states) - 1)
                features = outputs.hidden_states[layer_idx]
            else:
                features = outputs.hidden_states[-1]
        else:
            raise ValueError("Model output doesn't contain hidden states")
        
        # Pool features
        pooled_features = self.pool_features(features, attention_mask)
        
        if return_dict:
            return EncoderOutput(
                features=features,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
                pooled_features=pooled_features,
                mask=attention_mask,
                metadata={
                    'input_shape': input_values.shape,
                    'sample_rate': sample_rate or self.config.sample_rate,
                    'model_name': self.config.model_name,
                    'feature_type': self.config.feature_type
                }
            )
        else:
            return features, pooled_features
    
    def pool_features(self,
                     features: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool sequence features into a single vector.
        
        Args:
            features: Sequence features [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Pooled features [batch_size, hidden_dim]
        """
        batch_size = features.shape[0]
        
        if self.config.pooling_method == "mean":
            # Mean pooling
            if attention_mask is not None:
                # Expand mask for broadcasting
                mask = attention_mask.unsqueeze(-1).float()
                sum_features = torch.sum(features * mask, dim=1)
                sum_mask = torch.sum(mask, dim=1)
                pooled = sum_features / torch.clamp(sum_mask, min=1e-9)
            else:
                pooled = torch.mean(features, dim=1)
        
        elif self.config.pooling_method == "max":
            # Max pooling
            if attention_mask is not None:
                # Set masked positions to -inf
                mask = attention_mask.unsqueeze(-1)
                features = features.masked_fill(~mask.bool(), float('-inf'))
                pooled = torch.max(features, dim=1)[0]
                # Replace -inf with 0
                pooled[torch.isinf(pooled)] = 0
            else:
                pooled = torch.max(features, dim=1)[0]
        
        elif self.config.pooling_method == "attention":
            # Attention pooling
            pooled = self._attention_pooling(features, attention_mask)
        
        elif self.config.pooling_method == "last":
            # Last token pooling
            if attention_mask is not None:
                # Get indices of last non-padded tokens
                seq_lengths = torch.sum(attention_mask, dim=1) - 1
                batch_indices = torch.arange(batch_size, device=features.device)
                pooled = features[batch_indices, seq_lengths, :]
            else:
                pooled = features[:, -1, :]
        
        else:
            raise ValueError(f"Unknown pooling method: {self.config.pooling_method}")
        
        # Normalize if requested
        if self.config.normalize_features:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled
    
    def _attention_pooling(self,
                          features: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention-based pooling."""
        batch_size, seq_len, hidden_dim = features.shape
        
        # Learnable attention weights
        if not hasattr(self, 'attention_pool'):
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            ).to(features.device)
        
        # Compute attention scores
        attention_scores = self.attention_pool(features).squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask == 0, float('-inf')
            )
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
        
        # Weighted sum
        pooled = torch.sum(features * attention_weights, dim=1)
        
        return pooled
    
    def extract_features(self,
                        audio: Union[np.ndarray, torch.Tensor, str],
                        layers: Optional[List[int]] = None,
                        sample_rate: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Extract features from specific layer(s).
        
        Args:
            audio: Input audio
            layers: List of layer indices to extract (None for all)
            sample_rate: Sample rate of input audio
            
        Returns:
            Dictionary of features from different layers
        """
        processed = self.preprocess(audio, sample_rate)
        input_values = processed['input_values'].to(self.model.device)
        attention_mask = processed.get('attention_mask', None)
        
        # Forward pass with all hidden states
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True
        )
        
        if not hasattr(outputs, 'hidden_states'):
            raise ValueError("Model does not return hidden states")
        
        features = {}
        for i, hidden_state in enumerate(outputs.hidden_states):
            if layers is None or i in layers:
                # Pool the features
                pooled = self.pool_features(hidden_state, attention_mask)
                features[f'layer_{i}'] = pooled
        
        return features
    
    def get_embeddings(self,
                      audio: Union[np.ndarray, torch.Tensor, str],
                      sample_rate: Optional[int] = None) -> torch.Tensor:
        """Get audio embeddings (convenience method)."""
        output = self(audio, sample_rate=sample_rate)
        return output.pooled_features
    
    def batch_encode(self,
                    audio_list: List[Union[np.ndarray, torch.Tensor, str]],
                    batch_size: int = 4,
                    sample_rate: Optional[int] = None,
                    **kwargs) -> List[EncoderOutput]:
        """
        Encode a batch of audio inputs.
        
        Args:
            audio_list: List of audio inputs
            batch_size: Batch size for processing
            sample_rate: Sample rate of input audio
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of encoder outputs
        """
        outputs = []
        
        for i in range(0, len(audio_list), batch_size):
            batch_audio = audio_list[i:i + batch_size]
            output = self(batch_audio, sample_rate=sample_rate, **kwargs)
            outputs.append(output)
        
        # Concatenate if multiple batches
        if len(outputs) > 1:
            # Concatenate features
            all_features = torch.cat([out.features for out in outputs], dim=0)
            all_pooled = torch.cat([out.pooled_features for out in outputs], dim=0)
            
            # Handle other outputs
            if outputs[0].hidden_states is not None:
                all_hidden = []
                num_layers = len(outputs[0].hidden_states)
                for layer_idx in range(num_layers):
                    layer_hidden = torch.cat(
                        [out.hidden_states[layer_idx] for out in outputs], dim=0
                    )
                    all_hidden.append(layer_hidden)
            else:
                all_hidden = None
            
            if outputs[0].attentions is not None:
                all_attentions = []
                num_layers = len(outputs[0].attentions)
                for layer_idx in range(num_layers):
                    layer_attn = torch.cat(
                        [out.attentions[layer_idx] for out in outputs], dim=0
                    )
                    all_attentions.append(layer_attn)
            else:
                all_attentions = None
            
            output = EncoderOutput(
                features=all_features,
                hidden_states=all_hidden,
                attentions=all_attentions,
                pooled_features=all_pooled,
                mask=None,  # Can't concatenate masks easily
                metadata={'batch_size': len(audio_list)}
            )
            return [output]
        
        return outputs
    
    def save(self, path: str):
        """Save encoder to file."""
        torch.save({
            'config': self.config,
            'model_state_dict': self.model.state_dict()
        }, path)
        logger.info(f"Saved encoder to {path}")
    
    def load(self, path: str):
        """Load encoder from file."""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded encoder from {path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        stats = {
            'model_name': self.config.model_name,
            'output_dim': self.output_dim,
            'trainable': self.config.trainable,
            'sample_rate': self.config.sample_rate,
            'feature_type': self.config.feature_type,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
        
        # Add model-specific stats
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'hidden_size'):
                stats['hidden_size'] = config.hidden_size
            if hasattr(config, 'num_hidden_layers'):
                stats['num_layers'] = config.num_hidden_layers
            if hasattr(config, 'num_attention_heads'):
                stats['num_heads'] = config.num_attention_heads
        
        return stats

class Wav2Vec2Encoder(AudioEncoder):
    """Wav2Vec2 audio encoder."""
    
    def _initialize_model(self):
        """Initialize Wav2Vec2 model and preprocessor."""
        try:
            # Load processor/feature extractor
            if self.config.use_preprocessor:
                self.preprocessor = Wav2Vec2Processor.from_pretrained(
                    self.config.model_name
                )
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                    self.config.model_name
                )
            
            # Load model
            self.model = Wav2Vec2Model.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model {self.config.model_name}: {e}")
            raise
    
    def forward(self, audio, sample_rate=None, **kwargs):
        """Wav2Vec2-specific forward pass."""
        # Wav2Vec2 expects input_values
        return super().forward(audio, sample_rate, **kwargs)

class HuBERTEncoder(AudioEncoder):
    """HuBERT audio encoder."""
    
    def _initialize_model(self):
        """Initialize HuBERT model and preprocessor."""
        try:
            # Load processor/feature extractor
            if self.config.use_preprocessor:
                self.preprocessor = HubertProcessor.from_pretrained(
                    self.config.model_name
                )
            
            # Load model
            self.model = HubertModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load HuBERT model {self.config.model_name}: {e}")
            raise

class WhisperEncoder(AudioEncoder):
    """Whisper audio encoder."""
    
    def _initialize_model(self):
        """Initialize Whisper model and preprocessor."""
        try:
            # Load processor/feature extractor
            if self.config.use_preprocessor:
                self.preprocessor = WhisperProcessor.from_pretrained(
                    self.config.model_name
                )
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    self.config.model_name
                )
            
            # Load model (encoder only)
            whisper_model = WhisperModel.from_pretrained(
                self.config.model_name
            )
            self.model = whisper_model.encoder
            
            # Set output dimension
            self.output_dim = whisper_model.config.d_model
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model {self.config.model_name}: {e}")
            raise
    
    def forward(self, audio, sample_rate=None, **kwargs):
        """Whisper-specific forward pass."""
        # Whisper expects specific preprocessing
        processed = self.preprocess(audio, sample_rate)
        
        if 'input_features' not in processed:
            # Convert to input_features format
            input_values = processed['input_values']
            # Compute log-mel spectrogram
            import librosa
            input_features = []
            for waveform in input_values:
                waveform_np = waveform.numpy() if waveform.device == torch.device('cpu') else waveform.cpu().numpy()
                mel = librosa.feature.melspectrogram(
                    y=waveform_np,
                    sr=sample_rate or self.config.sample_rate,
                    n_mels=80,
                    n_fft=400,
                    hop_length=160
                )
                mel = torch.from_numpy(np.log(mel + 1e-8)).float()
                input_features.append(mel)
            input_features = torch.stack(input_features)
        else:
            input_features = processed['input_features']
        
        input_features = input_features.to(self.model.device)
        
        # Forward pass
        outputs = self.model(input_features)
        
        features = outputs.last_hidden_state
        pooled = self.pool_features(features)
        
        return EncoderOutput(
            features=features,
            hidden_states=None,
            attentions=None,
            pooled_features=pooled,
            mask=None,
            metadata={'model_name': self.config.model_name}
        )

# Custom audio encoder for spectral features
class SpectralEncoder(AudioEncoder):
    """Custom encoder for spectral features (MFCC, Mel, etc.)."""
    
    def _initialize_model(self):
        """Initialize custom spectral encoder."""
        # Simple CNN for spectral features
        if self.config.feature_type == "mfcc":
            input_channels = self.config.num_mfcc
        elif self.config.feature_type == "mel":
            input_channels = self.config.num_mel_bins
        elif self.config.feature_type == "spectrogram":
            input_channels = self.config.n_fft // 2 + 1
        else:
            raise ValueError(f"Unsupported feature type for SpectralEncoder: {self.config.feature_type}")
        
        self.model = nn.Sequential(
            # Conv layers for spectral features
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(128, self.config.output_dim)
        )
        
        self.output_dim = self.config.output_dim
    
    def forward(self, audio, sample_rate=None, **kwargs):
        """Spectral encoder forward pass."""
        # Extract spectral features
        processed = self.preprocess(audio, sample_rate)
        features = processed['input_values']
        
        # Add channel dimension for CNN
        if features.dim() == 3:
            features = features.unsqueeze(1)  # [batch_size, 1, n_mels, time]
        
        features = features.to(next(self.model.parameters()).device)
        
        # Forward through model
        pooled = self.model(features)
        
        # For spectral encoder, we don't have sequence features
        # Return pooled features as both features and pooled_features
        features_expanded = pooled.unsqueeze(1)  # Add sequence dimension
        
        return EncoderOutput(
            features=features_expanded,
            hidden_states=None,
            attentions=None,
            pooled_features=pooled,
            mask=None,
            metadata={
                'model_name': self.config.model_name,
                'feature_type': self.config.feature_type
            }
        )

# Example usage
if __name__ == "__main__":
    # Test different audio encoders
    encoders_to_test = [
        ('facebook/wav2vec2-base-960h', Wav2Vec2Encoder),
        ('facebook/hubert-base-ls960', HuBERTEncoder),
        ('openai/whisper-tiny', WhisperEncoder)
    ]
    
    # Create a dummy audio signal
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    dummy_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    for model_name, encoder_class in encoders_to_test:
        try:
            print(f"\nTesting {encoder_class.__name__} with {model_name}:")
            
            config = AudioEncoderConfig(
                model_name=model_name,
                pretrained=True,
                trainable=False,
                sample_rate=sample_rate,
                feature_type="wav2vec2" if "wav2vec2" in model_name else 
                           "hubert" if "hubert" in model_name else "whisper"
            )
            
            encoder = encoder_class(config)
            
            # Test audio encoding
            output = encoder(dummy_audio, sample_rate=sample_rate)
            print(f"  Output shape: {output.features.shape}")
            print(f"  Pooled shape: {output.pooled_features.shape}")
            
            # Test batch
            batch_output = encoder([dummy_audio, dummy_audio], sample_rate=sample_rate)
            print(f"  Batch output shape: {batch_output.features.shape}")
            
            # Get statistics
            stats = encoder.get_statistics()
            print(f"  Parameters: {stats['num_parameters']:,}")
            print(f"  Output dim: {stats['output_dim']}")
            
        except Exception as e:
            print(f"  Error: {e}")