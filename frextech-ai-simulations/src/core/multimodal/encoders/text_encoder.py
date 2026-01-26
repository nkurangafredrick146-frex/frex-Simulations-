"""
Text encoders for processing natural language inputs.
Includes transformer-based encoders (BERT, RoBERTa, T5, CLIP) and custom text encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModel,
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    T5Model,
    T5Tokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import logging
import warnings

from ..alignment import EncoderConfig, EncoderOutput

logger = logging.getLogger(__name__)

@dataclass
class TextEncoderConfig(EncoderConfig):
    """Configuration for text encoders."""
    
    # Text-specific parameters
    vocab_size: int = 30522
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    
    # Tokenization
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    
    # Special tokens
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = True
    
    # Positional embeddings
    max_position_embeddings: int = 512
    position_embedding_type: str = "absolute"
    
    def __post_init__(self):
        """Set encoder type."""
        self.encoder_type = "text"

class TextEncoder(nn.Module):
    """Base class for text encoders."""
    
    def __init__(self, config: TextEncoderConfig):
        super().__init__()
        self.config = config
        self.tokenizer = None
        self.model = None
        self.output_dim = config.output_dim
        
        # Initialize tokenizer and model
        self._initialize_model()
        
        # Freeze parameters if not trainable
        if not config.trainable:
            self._freeze_parameters()
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {config.model_name}")
    
    def _initialize_model(self):
        """Initialize tokenizer and model - to be implemented by subclasses."""
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
    
    def forward(self,
                text: Union[str, List[str], torch.Tensor],
                return_dict: bool = True,
                **kwargs) -> EncoderOutput:
        """
        Forward pass through text encoder.
        
        Args:
            text: Input text(s) - can be string, list of strings, or tokenized tensor
            return_dict: Whether to return EncoderOutput dict
            **kwargs: Additional arguments for tokenizer/model
            
        Returns:
            EncoderOutput with text features
        """
        # Tokenize if input is text
        if isinstance(text, (str, list)):
            tokenized = self.tokenize(text, **kwargs)
        else:
            # Assume already tokenized
            tokenized = {'input_ids': text}
            if 'attention_mask' not in tokenized:
                tokenized['attention_mask'] = torch.ones_like(text)
        
        # Move to model device
        input_ids = tokenized['input_ids'].to(self.model.device)
        attention_mask = tokenized.get('attention_mask', None)
        
        # Model forward pass
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_hidden_states': True,
            'output_attentions': True,
            'return_dict': True
        }
        
        # Add token_type_ids if model expects it
        if 'token_type_ids' in tokenized:
            model_kwargs['token_type_ids'] = tokenized['token_type_ids']
        
        # Update with any additional kwargs
        model_kwargs.update(kwargs)
        
        outputs = self.model(**model_kwargs)
        
        # Extract features
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
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
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'model_name': self.config.model_name
                }
            )
        else:
            return features, pooled_features
    
    def tokenize(self,
                 text: Union[str, List[str]],
                 **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenize text inputs.
        
        Args:
            text: Input text(s)
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary of tokenized inputs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        # Use config parameters as defaults
        tokenizer_kwargs = {
            'padding': self.config.padding,
            'truncation': self.config.truncation,
            'max_length': self.config.max_length,
            'return_tensors': self.config.return_tensors,
            'add_special_tokens': self.config.add_special_tokens,
            'return_attention_mask': self.config.return_attention_mask,
            'return_token_type_ids': self.config.return_token_type_ids
        }
        
        # Update with any kwargs
        tokenizer_kwargs.update(kwargs)
        
        # Tokenize
        return self.tokenizer(text, **tokenizer_kwargs)
    
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
        
        if self.config.pool_method == "cls":
            # Use [CLS] token
            pooled = features[:, 0, :]
        
        elif self.config.pool_method == "mean":
            # Mean pooling
            if attention_mask is not None:
                # Expand mask for broadcasting
                mask = attention_mask.unsqueeze(-1).float()
                sum_features = torch.sum(features * mask, dim=1)
                sum_mask = torch.sum(mask, dim=1)
                pooled = sum_features / torch.clamp(sum_mask, min=1e-9)
            else:
                pooled = torch.mean(features, dim=1)
        
        elif self.config.pool_method == "max":
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
        
        elif self.config.pool_method == "last":
            # Last token pooling (useful for decoder models)
            if attention_mask is not None:
                # Get indices of last non-padded tokens
                seq_lengths = torch.sum(attention_mask, dim=1) - 1
                batch_indices = torch.arange(batch_size, device=features.device)
                pooled = features[batch_indices, seq_lengths, :]
            else:
                pooled = features[:, -1, :]
        
        else:
            raise ValueError(f"Unknown pool method: {self.config.pool_method}")
        
        # Normalize if requested
        if self.config.normalize_features:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled
    
    def batch_encode(self,
                     texts: List[str],
                     batch_size: int = 32,
                     **kwargs) -> List[EncoderOutput]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            **kwargs: Additional arguments for encoding
            
        Returns:
            List of encoder outputs
        """
        outputs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            output = self(batch_texts, **kwargs)
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
                mask=None,  # Can't concatenate masks easily due to variable lengths
                metadata={'batch_size': len(texts)}
            )
            return [output]
        
        return outputs
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get text embeddings (convenience method)."""
        output = self(text)
        return output.pooled_features
    
    def get_similarity(self,
                      text1: str,
                      text2: str,
                      similarity_type: str = 'cosine') -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            similarity_type: Type of similarity ('cosine', 'dot')
            
        Returns:
            Similarity score
        """
        emb1 = self.get_embeddings(text1)
        emb2 = self.get_embeddings(text2)
        
        if similarity_type == 'cosine':
            similarity = F.cosine_similarity(emb1, emb2).item()
        elif similarity_type == 'dot':
            similarity = torch.dot(emb1.squeeze(), emb2.squeeze()).item()
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
        
        return similarity
    
    def save(self, path: str):
        """Save encoder to file."""
        torch.save({
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer.__class__.__name__
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
            'vocab_size': getattr(self.model.config, 'vocab_size', None),
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'num_trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
        }
        return stats

class TransformerTextEncoder(TextEncoder):
    """Generic transformer-based text encoder."""
    
    def _initialize_model(self):
        """Initialize transformer model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            self.model = AutoModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension from model config
            if hasattr(self.model.config, 'hidden_size'):
                self.output_dim = self.model.config.hidden_size
            elif hasattr(self.model.config, 'd_model'):
                self.output_dim = self.model.config.d_model
            else:
                self.output_dim = self.config.output_dim
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise

class BERTEncoder(TextEncoder):
    """BERT text encoder."""
    
    def _initialize_model(self):
        """Initialize BERT model and tokenizer."""
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.config.model_name
            )
            self.model = BertModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load BERT model {self.config.model_name}: {e}")
            raise
    
    def forward(self, text: Union[str, List[str]], **kwargs) -> EncoderOutput:
        """BERT-specific forward pass."""
        # BERT expects token_type_ids
        tokenized = self.tokenize(text, **kwargs)
        
        # Ensure token_type_ids
        if 'token_type_ids' not in tokenized:
            tokenized['token_type_ids'] = torch.zeros_like(tokenized['input_ids'])
        
        return super().forward(tokenized, **kwargs)

class RoBERTaEncoder(TextEncoder):
    """RoBERTa text encoder."""
    
    def _initialize_model(self):
        """Initialize RoBERTa model and tokenizer."""
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                self.config.model_name
            )
            self.model = RobertaModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load RoBERTa model {self.config.model_name}: {e}")
            raise

class DistilBERTEncoder(TextEncoder):
    """DistilBERT text encoder (lighter version of BERT)."""
    
    def _initialize_model(self):
        """Initialize DistilBERT model and tokenizer."""
        try:
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                self.config.model_name
            )
            self.model = DistilBertModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load DistilBERT model {self.config.model_name}: {e}")
            raise

class T5Encoder(TextEncoder):
    """T5 text encoder (encoder-decoder model)."""
    
    def _initialize_model(self):
        """Initialize T5 model and tokenizer."""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.config.model_name
            )
            self.model = T5Model.from_pretrained(
                self.config.model_name
            )
            
            # T5 uses encoder outputs
            self.model = self.model.encoder
            
            # Set output dimension
            self.output_dim = self.model.config.d_model
            
        except Exception as e:
            logger.error(f"Failed to load T5 model {self.config.model_name}: {e}")
            raise
    
    def forward(self, text: Union[str, List[str]], **kwargs) -> EncoderOutput:
        """T5-specific forward pass."""
        # T5 has different tokenization
        tokenized = self.tokenize(text, **kwargs)
        
        # T5 doesn't use token_type_ids
        if 'token_type_ids' in tokenized:
            del tokenized['token_type_ids']
        
        # Add decoder_input_ids for T5
        tokenized['decoder_input_ids'] = tokenized['input_ids']
        
        return super().forward(tokenized, **kwargs)
    
    def pool_features(self, features, attention_mask=None):
        """T5 often uses mean pooling or last token."""
        if self.config.pool_method == "last":
            return super().pool_features(features, attention_mask)
        else:
            # Default to mean pooling for T5
            return super().pool_features(features, attention_mask)

class CLIPTextEncoder(TextEncoder):
    """CLIP text encoder (part of CLIP multimodal model)."""
    
    def _initialize_model(self):
        """Initialize CLIP text model and tokenizer."""
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.config.model_name
            )
            self.model = CLIPTextModel.from_pretrained(
                self.config.model_name
            )
            
            # Set output dimension
            self.output_dim = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model {self.config.model_name}: {e}")
            raise
    
    def forward(self, text: Union[str, List[str]], **kwargs) -> EncoderOutput:
        """CLIP-specific forward pass."""
        tokenized = self.tokenize(text, **kwargs)
        
        # CLIP doesn't use token_type_ids
        if 'token_type_ids' in tokenized:
            del tokenized['token_type_ids']
        
        # Get position_ids for CLIP
        input_ids = tokenized['input_ids']
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        tokenized['position_ids'] = position_ids
        
        return super().forward(tokenized, **kwargs)
    
    def pool_features(self, features, attention_mask=None):
        """CLIP uses EOS token for pooling."""
        # CLIP uses the EOS token (last hidden state) for text features
        if attention_mask is not None:
            # Get EOS token positions
            eos_positions = torch.argmax(attention_mask, dim=1)
            batch_indices = torch.arange(features.shape[0], device=features.device)
            pooled = features[batch_indices, eos_positions]
        else:
            pooled = features[:, -1, :]
        
        if self.config.normalize_features:
            pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled

# Custom text encoder for specialized tasks
class CustomTextEncoder(TextEncoder):
    """Custom text encoder with configurable architecture."""
    
    def _initialize_model(self):
        """Initialize custom transformer model."""
        from transformers import BertConfig, BertModel
        
        # Create custom config
        config = BertConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_dim,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=self.config.intermediate_size,
            hidden_act=self.config.hidden_act,
            max_position_embeddings=self.config.max_position_embeddings,
            position_embedding_type=self.config.position_embedding_type
        )
        
        # Initialize model
        self.model = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        self.output_dim = self.config.hidden_dim
    
    def save(self, path: str):
        """Save custom encoder."""
        torch.save({
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'tokenizer_config': self.tokenizer.__class__.__name__
        }, path)
    
    def load(self, path: str):
        """Load custom encoder."""
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

# Example usage
if __name__ == "__main__":
    # Test different text encoders
    encoders_to_test = [
        ('bert-base-uncased', BERTEncoder),
        ('roberta-base', RoBERTaEncoder),
        ('distilbert-base-uncased', DistilBERTEncoder),
        ('t5-small', T5Encoder),
        ('openai/clip-vit-base-patch32', CLIPTextEncoder)
    ]
    
    test_texts = [
        "Hello, world!",
        "This is a test sentence.",
        "Multimodal AI is fascinating."
    ]
    
    for model_name, encoder_class in encoders_to_test:
        try:
            print(f"\nTesting {encoder_class.__name__} with {model_name}:")
            
            config = TextEncoderConfig(
                model_name=model_name,
                pretrained=True,
                trainable=False,
                pool_method='mean'
            )
            
            encoder = encoder_class(config)
            
            # Test single text
            output = encoder(test_texts[0])
            print(f"  Output shape: {output.features.shape}")
            print(f"  Pooled shape: {output.pooled_features.shape}")
            
            # Test batch
            batch_output = encoder(test_texts)
            print(f"  Batch output shape: {batch_output.features.shape}")
            
            # Get statistics
            stats = encoder.get_statistics()
            print(f"  Parameters: {stats['num_parameters']:,}")
            print(f"  Output dim: {stats['output_dim']}")
            
        except Exception as e:
            print(f"  Error: {e}")