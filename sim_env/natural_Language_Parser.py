#!/usr/bin/env python3
"""
Natural Language Simulation Descriptor & Parser
Convert natural language descriptions into executable simulation configurations
"""

import re
import json
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sem import extract_rels, rtuple
import threading
import time
import glm
from collections import defaultdict, OrderedDict
import logging
import hashlib

# Download required NLTK data (would be done once on first run)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLParser")

@dataclass
class NLParseResult:
    """Result of natural language parsing"""
    success: bool
    simulation_type: str
    parameters: Dict[str, Any]
    confidence: float
    parsed_elements: List[Dict[str, Any]]
    warnings: List[str]
    suggestions: List[str]
    raw_input: str

@dataclass
class SemanticFrame:
    """Semantic frame representing a simulation concept"""
    frame_type: str
    elements: Dict[str, Any]
    confidence: float
    source_text: str

class SimulationIntent(Enum):
    CREATE = "create"
    MODIFY = "modify"
    QUERY = "query"
    CONTROL = "control"
    ANALYZE = "analyze"
    COMPARE = "compare"

class NaturalLanguageParser:
    """Natural language parser for simulation descriptions"""
    
    def __init__(self, simulation_app):
        self.simulation_app = simulation_app
        
        # Load spaCy model for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
            logger.info("spaCy model loaded successfully")
        except OSError:
            self.nlp = None
            self.spacy_available = False
            logger.warning("spaCy model not available, using basic NLTK parsing")
        
        # Knowledge base of simulation concepts
        self.knowledge_base = self.initialize_knowledge_base()
        
        # Pattern matching rules
        self.patterns = self.initialize_patterns()
        
        # Entity recognition patterns
        self.entity_patterns = self.initialize_entity_patterns()
        
        # Context management
        self.conversation_context = []
        self.max_context_length = 10
        
        # Learning from user corrections
        self.learning_enabled = True
        self.correction_history = []
        
        logger.info("Natural Language Parser initialized")
    
    def initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base of simulation concepts"""
        return {
            "simulation_types": {
                "fluid": {
                    "keywords": ["fluid", "liquid", "water", "flow", "wave", "splash", "ocean", "river"],
                    "parameters": ["viscosity", "density", "surface_tension", "resolution"],
                    "defaults": {"viscosity": 0.01, "density": 1.0, "resolution": 64}
                },
                "particle": {
                    "keywords": ["particle", "point", "dot", "spark", "dust", "smoke", "cloud"],
                    "parameters": ["count", "size", "lifetime", "gravity", "wind"],
                    "defaults": {"count": 1000, "size": 0.1, "gravity": -9.8}
                },
                "fire": {
                    "keywords": ["fire", "flame", "burn", "smoke", "spark", "ember", "blaze"],
                    "parameters": ["intensity", "fuel", "oxygen", "temperature", "smoke_density"],
                    "defaults": {"intensity": 1.0, "temperature": 1000.0, "smoke_density": 0.5}
                },
                "fountain": {
                    "keywords": ["fountain", "waterfall", "spray", "jet", "sprinkle", "cascade"],
                    "parameters": ["height", "pressure", "particle_count", "gravity"],
                    "defaults": {"height": 5.0, "pressure": 10.0, "particle_count": 500}
                },
                "quantum": {
                    "keywords": ["quantum", "wave", "probability", "uncertainty", "superposition", "entanglement"],
                    "parameters": ["wave_function", "potential", "particle_count", "uncertainty"],
                    "defaults": {"particle_count": 100, "uncertainty": 0.1}
                },
                "astrophysics": {
                    "keywords": ["star", "planet", "galaxy", "orbit", "gravity", "cosmic", "universe", "black hole"],
                    "parameters": ["mass", "gravity_strength", "time_scale", "particle_count"],
                    "defaults": {"gravity_strength": 6.674e-11, "time_scale": 1e6, "particle_count": 100}
                }
            },
            "physical_properties": {
                "size": {"keywords": ["size", "large", "small", "big", "tiny", "huge", "massive"],
                        "values": {"tiny": 0.1, "small": 0.5, "medium": 1.0, "large": 2.0, "huge": 5.0, "massive": 10.0}},
                "speed": {"keywords": ["speed", "fast", "slow", "velocity", "quick", "rapid"],
                         "values": {"slow": 0.5, "medium": 1.0, "fast": 2.0, "very fast": 5.0}},
                "intensity": {"keywords": ["intensity", "strong", "weak", "powerful", "mild"],
                             "values": {"weak": 0.3, "medium": 1.0, "strong": 2.0, "very strong": 5.0}},
                "color": {"keywords": ["color", "red", "blue", "green", "yellow", "purple", "orange", "white", "black"],
                         "values": {
                             "red": (1.0, 0.0, 0.0), "blue": (0.0, 0.0, 1.0), "green": (0.0, 1.0, 0.0),
                             "yellow": (1.0, 1.0, 0.0), "purple": (0.5, 0.0, 0.5), "orange": (1.0, 0.5, 0.0),
                             "white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0)
                         }}
            },
            "actions": {
                "create": ["create", "make", "build", "generate", "start", "initialize"],
                "modify": ["change", "modify", "adjust", "set", "update", "alter"],
                "add": ["add", "insert", "include", "put"],
                "remove": ["remove", "delete", "eliminate", "take away"],
                "control": ["start", "stop", "pause", "resume", "reset", "run"]
            }
        }
    
    def initialize_patterns(self) -> List[Dict[str, Any]]:
        """Initialize regex patterns for parsing"""
        return [
            # Number patterns
            {
                "name": "number",
                "pattern": r'\b(\d+\.?\d*)\b',
                "type": "value"
            },
            # Unit patterns
            {
                "name": "units",
                "pattern": r'\b(m/s|m/s²|kg|N|Pa|K|°C|°F)\b',
                "type": "unit"
            },
            # Simulation type patterns
            {
                "name": "sim_type",
                "pattern": r'\b(fluid|particle|fire|fountain|quantum|astrophysics|gravity)\s+(simulation|system|effect)\b',
                "type": "simulation_type"
            },
            # Parameter assignment patterns
            {
                "name": "parameter_assignment",
                "pattern": r'\b(set|make|with)\s+(\w+)\s+(to|at)\s+([\d\.]+)\b',
                "type": "assignment"
            },
            # Color patterns
            {
                "name": "color",
                "pattern": r'\b(red|blue|green|yellow|purple|orange|white|black)\b',
                "type": "color"
            }
        ]
    
    def initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for entity recognition"""
        return {
            "quantities": [
                r'\b\d+\s+particles?\b',
                r'\b\d+\s+points?\b',
                r'\b\d+\s+units?\b'
            ],
            "properties": [
                r'\b(high|low)\s+(viscosity|density|pressure|temperature)\b',
                r'\b(large|small)\s+(size|amount|number)\b',
                r'\b(fast|slow)\s+(speed|velocity|movement)\b'
            ],
            "conditions": [
                r'\bunder\s+(gravity|pressure|force)\b',
                r'\bwith\s+(wind|friction|resistance)\b',
                r'\bin\s+(vacuum|air|water)\b'
            ]
        }
    
    def parse_input(self, text: str, context: List[str] = None) -> NLParseResult:
        """Parse natural language input and generate simulation configuration"""
        logger.info(f"Parsing input: {text}")
        
        # Update conversation context
        self.update_context(text, context)
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Extract intent
        intent = self.extract_intent(cleaned_text)
        
        # Extract entities and parameters
        entities = self.extract_entities(cleaned_text)
        parameters = self.extract_parameters(cleaned_text, entities)
        
        # Determine simulation type
        sim_type, sim_confidence = self.determine_simulation_type(cleaned_text, entities)
        
        # Apply context from previous interactions
        parameters = self.apply_context(parameters)
        
        # Generate confidence score
        confidence = self.calculate_confidence(sim_confidence, parameters, entities)
        
        # Generate suggestions for clarification
        suggestions = self.generate_suggestions(parameters, sim_type, confidence)
        
        # Create result
        result = NLParseResult(
            success=confidence > 0.3,  # Minimum confidence threshold
            simulation_type=sim_type,
            parameters=parameters,
            confidence=confidence,
            parsed_elements=entities,
            warnings=self.generate_warnings(parameters, sim_type),
            suggestions=suggestions,
            raw_input=text
        )
        
        logger.info(f"Parse result: {result.simulation_type} with confidence {confidence:.2f}")
        
        return result
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text for parsing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Expand contractions
        contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "it's": "it is",
            "i'm": "i am",
            "you're": "you are",
            "they're": "they are",
            "we're": "we are",
            "that's": "that is",
            "what's": "what is",
            "where's": "where is",
            "how's": "how is",
            "why's": "why is",
            "let's": "let us"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def extract_intent(self, text: str) -> SimulationIntent:
        """Extract user intent from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in self.knowledge_base["actions"]["create"]):
            return SimulationIntent.CREATE
        elif any(word in text_lower for word in self.knowledge_base["actions"]["modify"]):
            return SimulationIntent.MODIFY
        elif any(word in text_lower for word in self.knowledge_base["actions"]["add"]):
            return SimulationIntent.MODIFY
        elif any(word in text_lower for word in self.knowledge_base["actions"]["remove"]):
            return SimulationIntent.MODIFY
        elif any(word in text_lower for word in self.knowledge_base["actions"]["control"]):
            return SimulationIntent.CONTROL
        elif "what" in text_lower or "how" in text_lower or "show" in text_lower:
            return SimulationIntent.QUERY
        elif "compare" in text_lower or "difference" in text_lower:
            return SimulationIntent.COMPARE
        else:
            return SimulationIntent.CREATE  # Default intent
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using multiple methods"""
        entities = []
        
        # Method 1: Regex pattern matching
        entities.extend(self.extract_entities_with_patterns(text))
        
        # Method 2: NLTK POS tagging and chunking
        entities.extend(self.extract_entities_with_nltk(text))
        
        # Method 3: spaCy processing (if available)
        if self.spacy_available:
            entities.extend(self.extract_entities_with_spacy(text))
        
        # Remove duplicates
        unique_entities = []
        seen = set()
        for entity in entities:
            entity_key = f"{entity['type']}_{entity['value']}"
            if entity_key not in seen:
                unique_entities.append(entity)
                seen.add(entity_key)
        
        return unique_entities
    
    def extract_entities_with_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns"""
        entities = []
        
        for pattern in self.patterns:
            matches = re.finditer(pattern["pattern"], text)
            for match in matches:
                entity = {
                    "type": pattern["type"],
                    "value": match.group(),
                    "groups": match.groups(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.8
                }
                entities.append(entity)
        
        # Extract quantities
        quantity_patterns = [
            (r'\b(\d+)\s+particles?\b', 'particle_count'),
            (r'\b(\d+)\s+points?\b', 'point_count'),
            (r'\b(\d+)\s+units?\b', 'unit_count'),
            (r'\bgravity\s+of\s+([\d\.\-]+)\b', 'gravity'),
            (r'\bvelocity\s+of\s+([\d\.\-]+)\b', 'velocity'),
            (r'\bmass\s+of\s+([\d\.\-]+)\b', 'mass')
        ]
        
        for pattern, param_name in quantity_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    value = float(match.group(1))
                    entity = {
                        "type": "parameter",
                        "name": param_name,
                        "value": value,
                        "confidence": 0.9,
                        "source": match.group()
                    }
                    entities.append(entity)
                except ValueError:
                    continue
        
        return entities
    
    def extract_entities_with_nltk(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using NLTK"""
        entities = []
        
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity recognition
            tree = ne_chunk(pos_tags)
            
            # Extract noun phrases and other patterns
            for i, (word, pos) in enumerate(pos_tags):
                # Numbers
                if pos == 'CD':
                    # Look for unit or context
                    unit = None
                    if i > 0 and pos_tags[i-1][1] in ['NN', 'NNS']:
                        unit = pos_tags[i-1][0]
                    
                    entity = {
                        "type": "quantity",
                        "value": word,
                        "unit": unit,
                        "pos": pos,
                        "confidence": 0.7
                    }
                    entities.append(entity)
                
                # Adjectives that might indicate properties
                elif pos == 'JJ':
                    property_map = {
                        'large': 'size', 'small': 'size', 'big': 'size', 'tiny': 'size',
                        'fast': 'speed', 'slow': 'speed', 'quick': 'speed',
                        'strong': 'intensity', 'weak': 'intensity', 'powerful': 'intensity',
                        'hot': 'temperature', 'cold': 'temperature', 'warm': 'temperature'
                    }
                    
                    if word in property_map:
                        entity = {
                            "type": "property",
                            "property": property_map[word],
                            "value": word,
                            "confidence": 0.6
                        }
                        entities.append(entity)
            
            # Extract verb-object patterns
            for i in range(len(pos_tags) - 1):
                if pos_tags[i][1].startswith('VB') and pos_tags[i+1][1] in ['NN', 'NNS']:
                    entity = {
                        "type": "action",
                        "verb": pos_tags[i][0],
                        "object": pos_tags[i+1][0],
                        "confidence": 0.5
                    }
                    entities.append(entity)
                    
        except Exception as e:
            logger.warning(f"NLTK entity extraction failed: {e}")
        
        return entities
    
    def extract_entities_with_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy"""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity = {
                    "type": "named_entity",
                    "value": ent.text,
                    "label": ent.label_,
                    "confidence": 0.8
                }
                entities.append(entity)
            
            # Extract noun chunks
            for chunk in doc.noun_chunks:
                entity = {
                    "type": "noun_chunk",
                    "value": chunk.text,
                    "root": chunk.root.text,
                    "confidence": 0.6
                }
                entities.append(entity)
            
            # Extract adjectives and their modified nouns
            for token in doc:
                if token.pos_ == 'ADJ' and token.head.pos_ == 'NOUN':
                    entity = {
                        "type": "property_assignment",
                        "property": token.head.text,
                        "value": token.text,
                        "confidence": 0.7
                    }
                    entities.append(entity)
                    
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}")
        
        return entities
    
    def extract_parameters(self, text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract simulation parameters from entities and text"""
        parameters = {}
        
        # Extract from direct parameter entities
        for entity in entities:
            if entity["type"] == "parameter":
                parameters[entity["name"]] = entity["value"]
        
        # Extract particle count
        particle_matches = re.findall(r'\b(\d+)\s+particles?\b', text)
        if particle_matches:
            parameters["particle_count"] = int(particle_matches[0])
        
        # Extract gravity
        gravity_matches = re.findall(r'\bgravity\s+of\s+([\d\.\-]+)\b', text)
        if gravity_matches:
            parameters["gravity"] = float(gravity_matches[0])
        
        # Extract size descriptions
        size_descriptions = {
            "tiny": 0.1, "small": 0.5, "medium": 1.0, 
            "large": 2.0, "big": 2.0, "huge": 5.0, "massive": 10.0
        }
        
        for desc, value in size_descriptions.items():
            if desc in text:
                parameters["size"] = value
                break
        
        # Extract speed descriptions
        speed_descriptions = {
            "slow": 0.5, "medium": 1.0, "fast": 2.0, "very fast": 5.0, "rapid": 3.0
        }
        
        for desc, value in speed_descriptions.items():
            if desc in text:
                parameters["speed"] = value
                break
        
        # Extract colors
        color_matches = re.findall(r'\b(red|blue|green|yellow|purple|orange|white|black)\b', text)
        if color_matches:
            color_name = color_matches[0]
            color_values = self.knowledge_base["physical_properties"]["color"]["values"]
            if color_name in color_values:
                parameters["color"] = color_values[color_name]
        
        # Extract intensity
        intensity_words = ["weak", "mild", "medium", "strong", "intense", "powerful"]
        for word in intensity_words:
            if word in text:
                intensity_map = {"weak": 0.3, "mild": 0.6, "medium": 1.0, "strong": 2.0, "intense": 3.0, "powerful": 5.0}
                parameters["intensity"] = intensity_map.get(word, 1.0)
                break
        
        return parameters
    
    def determine_simulation_type(self, text: str, entities: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Determine the most likely simulation type"""
        scores = {}
        
        # Score based on keyword matching
        for sim_type, sim_info in self.knowledge_base["simulation_types"].items():
            score = 0.0
            keywords = sim_info["keywords"]
            
            # Direct keyword matches
            for keyword in keywords:
                if keyword in text:
                    score += 1.0
            
            # Contextual matches
            if any(entity.get("value", "").lower() in keywords for entity in entities):
                score += 0.5
            
            scores[sim_type] = score
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            return "particle", 0.1  # Default fallback
        
        best_type = max(scores.keys(), key=lambda k: scores[k])
        max_score = scores[best_type]
        
        # Normalize confidence (0.0 to 1.0)
        confidence = min(max_score / 3.0, 1.0)
        
        return best_type, confidence
    
    def apply_context(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply context from previous interactions"""
        if not self.conversation_context:
            return parameters
        
        # Merge parameters with context from recent interactions
        context_parameters = {}
        for context_item in self.conversation_context[-3:]:  # Last 3 items
            if isinstance(context_item, dict) and 'parameters' in context_item:
                context_parameters.update(context_item['parameters'])
        
        # Update current parameters with context (current overrides context)
        merged_parameters = {**context_parameters, **parameters}
        
        return merged_parameters
    
    def calculate_confidence(self, sim_confidence: float, parameters: Dict[str, Any], 
                           entities: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the parse result"""
        confidence = sim_confidence
        
        # Boost confidence for specific parameters
        if parameters:
            confidence += min(len(parameters) * 0.1, 0.3)
        
        # Boost for entity count
        if entities:
            confidence += min(len(entities) * 0.05, 0.2)
        
        # Penalize for ambiguous terms
        ambiguous_terms = ["it", "that", "something", "thing"]
        text_lower = self.conversation_context[-1] if self.conversation_context else ""
        for term in ambiguous_terms:
            if term in text_lower:
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def generate_suggestions(self, parameters: Dict[str, Any], sim_type: str, 
                           confidence: float) -> List[str]:
        """Generate suggestions for clarification or improvement"""
        suggestions = []
        
        if confidence < 0.5:
            suggestions.append("Could you provide more details about what kind of simulation you want?")
        
        if not parameters:
            suggestions.append("Please specify parameters like particle count, size, or speed.")
        
        # Type-specific suggestions
        if sim_type == "fluid" and "viscosity" not in parameters:
            suggestions.append("Would you like to specify the fluid viscosity?")
        elif sim_type == "fire" and "temperature" not in parameters:
            suggestions.append("What temperature would you like for the fire?")
        elif sim_type == "particle" and "particle_count" not in parameters:
            suggestions.append("How many particles would you like?")
        
        return suggestions
    
    def generate_warnings(self, parameters: Dict[str, Any], sim_type: str) -> List[str]:
        """Generate warnings about potential issues"""
        warnings = []
        
        # Performance warnings
        if parameters.get("particle_count", 0) > 10000:
            warnings.append("High particle count may affect performance")
        
        # Physical plausibility warnings
        if sim_type == "fluid" and parameters.get("viscosity", 1.0) < 0.001:
            warnings.append("Very low viscosity may cause numerical instability")
        
        if parameters.get("gravity", -9.8) > 100:
            warnings.append("Extreme gravity values may cause instability")
        
        return warnings
    
    def update_context(self, text: str, context: List[str] = None):
        """Update conversation context"""
        if context:
            self.conversation_context = context[-self.max_context_length:]
        
        self.conversation_context.append(text)
        
        # Trim context if too long
        if len(self.conversation_context) > self.max_context_length:
            self.conversation_context = self.conversation_context[-self.max_context_length:]
    
    def learn_from_correction(self, original_parse: NLParseResult, corrected_parameters: Dict[str, Any]):
        """Learn from user corrections to improve future parsing"""
        if not self.learning_enabled:
            return
        
        correction = {
            "original_input": original_parse.raw_input,
            "original_parameters": original_parse.parameters,
            "corrected_parameters": corrected_parameters,
            "timestamp": time.time()
        }
        
        self.correction_history.append(correction)
        
        # Simple learning: adjust pattern weights based on corrections
        logger.info("Learned from user correction")
    
    def generate_simulation_config(self, parse_result: NLParseResult) -> Dict[str, Any]:
        """Generate complete simulation configuration from parse result"""
        if not parse_result.success:
            return {"error": "Parsing failed", "confidence": parse_result.confidence}
        
        # Get default parameters for simulation type
        sim_type = parse_result.simulation_type
        defaults = self.knowledge_base["simulation_types"].get(sim_type, {}).get("defaults", {})
        
        # Merge defaults with parsed parameters (parsed parameters override defaults)
        config = {**defaults, **parse_result.parameters}
        
        # Add simulation metadata
        config["_metadata"] = {
            "simulation_type": sim_type,
            "source": "natural_language",
            "parse_confidence": parse_result.confidence,
            "timestamp": time.time(),
            "warnings": parse_result.warnings
        }
        
        return config
    
    def execute_parsed_simulation(self, parse_result: NLParseResult) -> bool:
        """Execute simulation based on parse result"""
        if not parse_result.success:
            logger.error("Cannot execute failed parse result")
            return False
        
        try:
            # Generate configuration
            config = self.generate_simulation_config(parse_result)
            
            # Switch to appropriate simulation type
            sim_type = parse_result.simulation_type
            if sim_type in self.simulation_app.simulation_types:
                self.simulation_app.switch_simulation(sim_type)
            else:
                logger.warning(f"Unknown simulation type: {sim_type}")
                return False
            
            # Apply parameters to current simulation
            current_sim = self.simulation_app.current_simulation
            if current_sim:
                self.apply_parameters_to_simulation(current_sim, config)
                return True
            else:
                logger.error("No current simulation available")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute parsed simulation: {e}")
            return False
    
    def apply_parameters_to_simulation(self, simulation, parameters: Dict[str, Any]):
        """Apply parsed parameters to simulation object"""
        # Remove metadata
        clean_params = {k: v for k, v in parameters.items() if not k.startswith('_')}
        
        for param_name, param_value in clean_params.items():
            if hasattr(simulation, param_name):
                setattr(simulation, param_name, param_value)
            elif hasattr(simulation, 'config') and param_name in simulation.config:
                simulation.config[param_name] = param_value
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation"""
        return {
            "turn_count": len(self.conversation_context),
            "recent_turns": self.conversation_context[-3:],
            "learning_examples": len(self.correction_history),
            "active_context": bool(self.conversation_context)
        }
    
    def reset_conversation(self):
        """Reset conversation context"""
        self.conversation_context = []
        logger.info("Conversation context reset")

class InteractiveNLInterface:
    """Interactive natural language interface for real-time parsing"""
    
    def __init__(self, parser: NaturalLanguageParser):
        self.parser = parser
        self.is_active = False
        self.last_result = None
        self.user_feedback = {}
        
    def start_interactive_mode(self):
        """Start interactive natural language mode"""
        self.is_active = True
        print("Natural Language Interface Active")
        print("Describe the simulation you want to create...")
        print("Type 'help' for assistance or 'exit' to quit\n")
        
        while self.is_active:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'stop']:
                    self.stop_interactive_mode()
                    break
                elif user_input.lower() in ['help', '?']:
                    self.show_help()
                elif user_input.lower() == 'reset':
                    self.parser.reset_conversation()
                    print("Conversation reset")
                elif user_input.lower() == 'status':
                    self.show_status()
                else:
                    self.process_user_input(user_input)
                    
            except KeyboardInterrupt:
                self.stop_interactive_mode()
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def stop_interactive_mode(self):
        """Stop interactive mode"""
        self.is_active = False
        print("\nNatural Language Interface stopped")
    
    def show_help(self):
        """Show help information"""
        help_text = """
Available Commands:
  - Describe a simulation: "Create a fountain with 1000 particles"
  - Modify parameters: "Make the gravity stronger" 
  - Control simulation: "Start the simulation", "Pause", "Reset"
  - Ask questions: "What is the current particle count?"
  
Special Commands:
  - help: Show this help message
  - reset: Reset conversation context
  - status: Show current simulation status
  - exit: Exit natural language mode

Examples:
  - "Create a fire simulation with high temperature"
  - "Make a fluid system with low viscosity"
  - "Show me 500 particles with blue color"
  - "Increase the particle size"
        """
        print(help_text)
    
    def show_status(self):
        """Show current simulation status"""
        summary = self.parser.get_conversation_summary()
        print(f"Conversation turns: {summary['turn_count']}")
        print(f"Learning examples: {summary['learning_examples']}")
        
        if self.last_result:
            print(f"Last simulation: {self.last_result.simulation_type}")
            print(f"Confidence: {self.last_result.confidence:.2f}")
    
    def process_user_input(self, user_input: str):
        """Process user input and execute commands"""
        # Parse the input
        result = self.parser.parse_input(user_input)
        self.last_result = result
        
        # Display results
        print(f"\nParser: {result.simulation_type} simulation (confidence: {result.confidence:.2f})")
        
        if result.parameters:
            print("Parameters:", result.parameters)
        
        if result.warnings:
            print("Warnings:", ", ".join(result.warnings))
        
        if result.suggestions and result.confidence < 0.7:
            print("Suggestions:", ", ".join(result.suggestions))
        
        # Execute if confident enough
        if result.confidence > 0.4:
            success = self.parser.execute_parsed_simulation(result)
            if success:
                print("✓ Simulation configured successfully")
            else:
                print("✗ Failed to configure simulation")
        else:
            print("? Low confidence - please provide more details")
        
        print()  # Empty line for readability

# Example usage and testing
if __name__ == "__main__":
    # Create a mock simulation app for testing
    class MockSimulationApp:
        def __init__(self):
            self.simulation_types = {
                "fluid": "FluidDynamicsSimulation",
                "particle": "BasicParticleSimulation", 
                "fire": "FireSimulation",
                "fountain": "FountainSimulation",
                "quantum": "QuantumPhysicsSimulation",
                "astrophysics": "AstrophysicsSimulation"
            }
            self.current_simulation = type('MockSimulation', (), {
                'config': {},
                'particle_count': 100
            })()
        
        def switch_simulation(self, sim_type):
            print(f"Switching to {sim_type} simulation")
            return True
    
    # Test the natural language parser
    simulation_app = MockSimulationApp()
    parser = NaturalLanguageParser(simulation_app)
    
    # Test cases
    test_cases = [
        "Create a fountain with 1000 particles",
        "Make a fire simulation with high temperature",
        "Show me 500 blue particles with medium speed",
        "Create a fluid system with low viscosity and high density",
        "Make gravity stronger and add more particles"
    ]
    
    print("Testing Natural Language Parser:")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nInput: '{test_case}'")
        result = parser.parse_input(test_case)
        
        print(f"Simulation: {result.simulation_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Parameters: {result.parameters}")
        print(f"Success: {result.success}")
        
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        
        print("-" * 30)
    
    # Test interactive mode
    print("\nStarting Interactive Mode Test:")
    print("=" * 50)
    
    interface = InteractiveNLInterface(parser)
    interface.start_interactive_mode()