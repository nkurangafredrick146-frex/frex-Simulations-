"""
AI Screening Module
Machine learning pipeline for drug efficacy and toxicity prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class MolecularDataset(Dataset):
    """Dataset for molecular machine learning"""
    
    def __init__(self, features, labels, compound_ids=None):
        """
        Initialize molecular dataset
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Target labels (n_samples, n_targets)
            compound_ids: List of compound IDs
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.compound_ids = compound_ids if compound_ids else []
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.compound_ids:
            return self.features[idx], self.labels[idx], self.compound_ids[idx]
        return self.features[idx], self.labels[idx]

class AIScreeningPipeline:
    """AI pipeline for drug screening with multiple ML models"""
    
    def __init__(self, model_type='ensemble', use_gpu=True):
        """
        Initialize AI screening pipeline
        
        Args:
            model_type: Type of model ('ensemble', 'gnn', 'transformer', 'xgboost')
            use_gpu: Whether to use GPU acceleration
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Model components
        self.efficacy_model = None
        self.toxicity_model = None
        self.adme_model = None
        self.scaler = StandardScaler()
        self.feature_encoder = None
        
        # Training history
        self.training_history = {
            'efficacy': [],
            'toxicity': [],
            'adme': []
        }
        
        # Model configuration
        self.config = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 10,
            'hidden_dims': [1024, 512, 256, 128],
            'dropout_rate': 0.3
        }
        
        # Feature dimensions
        self.num_features = 2048  # Default molecular fingerprint size
        self.num_efficacy_targets = 1  # Efficacy score
        self.num_toxicity_classes = 10  # 10 toxicity endpoints
        self.num_adme_properties = 5   # ADME properties
        
        self.initialized = False
        
    def initialize(self, config_path=None):
        """Initialize the AI screening pipeline"""
        print(f"Initializing AI Screening Pipeline ({self.model_type})...")
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
        
        # Initialize models based on type
        self._initialize_models()
        
        # Load pre-trained models if available
        self._load_pretrained_models()
        
        # Initialize feature encoder
        self._initialize_feature_encoder()
        
        self.initialized = True
        print(f"AI Screening Pipeline initialized on {self.device}")
        
    def _initialize_models(self):
        """Initialize machine learning models"""
        if self.model_type == 'ensemble':
            self._initialize_ensemble_models()
        elif self.model_type == 'gnn':
            self._initialize_gnn_models()
        elif self.model_type == 'transformer':
            self._initialize_transformer_models()
        elif self.model_type == 'xgboost':
            self._initialize_xgboost_models()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _initialize_ensemble_models(self):
        """Initialize ensemble of neural networks"""
        # Efficacy prediction model
        self.efficacy_model = EfficacyPredictionModel(
            input_dim=self.num_features,
            hidden_dims=self.config['hidden_dims'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Toxicity prediction model
        self.toxicity_model = ToxicityPredictionModel(
            input_dim=self.num_features,
            hidden_dims=self.config['hidden_dims'],
            num_classes=self.num_toxicity_classes,
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # ADME properties model
        self.adme_model = ADMEPredictionModel(
            input_dim=self.num_features,
            hidden_dims=self.config['hidden_dims'],
            num_properties=self.num_adme_properties,
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Initialize optimizers
        self.efficacy_optimizer = optim.Adam(
            self.efficacy_model.parameters(),
            lr=self.config['learning_rate']
        )
        self.toxicity_optimizer = optim.Adam(
            self.toxicity_model.parameters(),
            lr=self.config['learning_rate']
        )
        self.adme_optimizer = optim.Adam(
            self.adme_model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Loss functions
        self.efficacy_loss_fn = nn.MSELoss()  # Regression
        self.toxicity_loss_fn = nn.CrossEntropyLoss()  # Multi-class classification
        self.adme_loss_fn = nn.MSELoss()  # Regression for each property
    
    def _initialize_gnn_models(self):
        """Initialize Graph Neural Network models"""
        # GNN for molecular graph representation
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        class MolecularGNN(nn.Module):
            def __init__(self, node_dim=78, edge_dim=4, hidden_dim=256, output_dim=128):
                super(MolecularGNN, self).__init__()
                self.conv1 = GCNConv(node_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.conv3 = GCNConv(hidden_dim, hidden_dim)
                self.lin = nn.Linear(hidden_dim, output_dim)
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x, edge_index, batch):
                x = self.conv1(x, edge_index)
                x = self.activation(x)
                x = self.dropout(x)
                
                x = self.conv2(x, edge_index)
                x = self.activation(x)
                x = self.dropout(x)
                
                x = self.conv3(x, edge_index)
                x = self.activation(x)
                
                # Global pooling
                x = global_mean_pool(x, batch)
                x = self.lin(x)
                
                return x
        
        self.gnn_encoder = MolecularGNN().to(self.device)
        
        # Task-specific heads
        self.efficacy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        self.toxicity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, self.num_toxicity_classes)
        ).to(self.device)
    
    def _initialize_transformer_models(self):
        """Initialize Transformer-based models"""
        # Transformer for sequence-based molecular representation
        from transformers import AutoModel, AutoTokenizer
        
        try:
            # Use pre-trained chemical transformer
            self.tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
            self.transformer = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
            
            # Freeze transformer layers
            for param in self.transformer.parameters():
                param.requires_grad = False
            
            # Task-specific heads
            self.efficacy_head = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ).to(self.device)
            
        except ImportError:
            print("Transformers library not available, using simple transformer")
            self._initialize_simple_transformer()
    
    def _initialize_simple_transformer(self):
        """Initialize simple transformer model"""
        class SimpleTransformer(nn.Module):
            def __init__(self, input_dim=512, num_heads=8, num_layers=3, hidden_dim=256):
                super(SimpleTransformer, self).__init__()
                self.embedding = nn.Linear(input_dim, hidden_dim)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim*4,
                    dropout=0.1
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.pool = nn.AdaptiveAvgPool1d(1)
            
            def forward(self, x):
                # x shape: (batch_size, seq_len, input_dim)
                x = self.embedding(x)
                x = x.transpose(0, 1)  # Transformer expects (seq_len, batch_size, hidden_dim)
                x = self.transformer(x)
                x = x.transpose(0, 1)  # Back to (batch_size, seq_len, hidden_dim)
                x = self.pool(x.transpose(1, 2)).squeeze(2)
                return x
        
        self.transformer = SimpleTransformer().to(self.device)
    
    def _initialize_xgboost_models(self):
        """Initialize XGBoost models"""
        try:
            import xgboost as xgb
            
            # Efficacy model
            self.efficacy_xgb = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='reg:squarederror'
            )
            
            # Toxicity model
            self.toxicity_xgb = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=self.num_toxicity_classes
            )
            
        except ImportError:
            print("XGBoost not available, using neural network fallback")
            self.model_type = 'ensemble'
            self._initialize_ensemble_models()
    
    def _initialize_feature_encoder(self):
        """Initialize molecular feature encoder"""
        # This would use RDKit or other cheminformatics libraries
        # For simulation, create a mock encoder
        
        class MockFeatureEncoder:
            def __init__(self):
                self.fingerprint_size = self.num_features
            
            def encode(self, smiles):
                """Encode SMILES string to molecular fingerprint"""
                # Generate deterministic fingerprint from SMILES hash
                hash_val = hashlib.md5(smiles.encode()).hexdigest()
                
                # Convert hash to binary array
                fingerprint = np.zeros(self.num_features)
                
                # Use hash to set bits in fingerprint
                for i in range(0, len(hash_val), 2):
                    idx = int(hash_val[i:i+2], 16) % self.num_features
                    fingerprint[idx] = 1
                
                # Add some noise
                noise = np.random.randn(self.num_features) * 0.1
                fingerprint = np.clip(fingerprint + noise, 0, 1)
                
                return fingerprint
            
            def encode_batch(self, smiles_list):
                """Encode batch of SMILES strings"""
                fingerprints = []
                for smiles in smiles_list:
                    fingerprints.append(self.encode(smiles))
                return np.array(fingerprints)
        
        self.feature_encoder = MockFeatureEncoder()
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        import json
        
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        import os
        from pathlib import Path
        
        model_dir = Path('./models/ai_screening/')
        
        if model_dir.exists():
            # Load efficacy model
            efficacy_path = model_dir / 'efficacy_model.pth'
            if efficacy_path.exists() and self.efficacy_model:
                self.efficacy_model.load_state_dict(
                    torch.load(efficacy_path, map_location=self.device)
                )
                print("Loaded pre-trained efficacy model")
            
            # Load toxicity model
            toxicity_path = model_dir / 'toxicity_model.pth'
            if toxicity_path.exists() and self.toxicity_model:
                self.toxicity_model.load_state_dict(
                    torch.load(toxicity_path, map_location=self.device)
                )
                print("Loaded pre-trained toxicity model")
            
            # Load scaler
            scaler_path = model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("Loaded feature scaler")
    
    def screen_compound(self, compound_data: Dict) -> Dict:
        """
        Screen a single compound using AI models
        
        Args:
            compound_data: Dictionary containing compound information
            
        Returns:
            Dictionary with screening predictions
        """
        if not self.initialized:
            self.initialize()
        
        print(f"Screening compound: {compound_data.get('name', 'Unknown')}")
        
        # Extract compound information
        smiles = compound_data.get('smiles', '')
        compound_id = compound_data.get('compound_id', 'unknown')
        
        if not smiles:
            raise ValueError("SMILES string required for screening")
        
        # Generate molecular features
        features = self._extract_features(compound_data)
        
        # Make predictions
        predictions = {
            'compound_id': compound_id,
            'compound_name': compound_data.get('name', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'features_used': features.shape[-1] if hasattr(features, 'shape') else len(features)
        }
        
        # Efficacy prediction
        efficacy_pred = self._predict_efficacy(features)
        predictions['efficacy'] = efficacy_pred
        
        # Toxicity prediction
        toxicity_pred = self._predict_toxicity(features)
        predictions['toxicity'] = toxicity_pred
        
        # ADME prediction
        adme_pred = self._predict_adme(features)
        predictions['adme'] = adme_pred
        
        # Overall drug-likeness score
        drug_likeness = self._calculate_drug_likeness(efficacy_pred, toxicity_pred, adme_pred)
        predictions['drug_likeness'] = drug_likeness
        
        # Risk assessment
        risk_assessment = self._assess_risks(efficacy_pred, toxicity_pred)
        predictions['risk_assessment'] = risk_assessment
        
        # Recommendation
        predictions['recommendation'] = self._generate_recommendation(predictions)
        
        return predictions
    
    def screen_compounds_batch(self, compounds_list: List[Dict]) -> List[Dict]:
        """
        Screen multiple compounds in batch
        
        Args:
            compounds_list: List of compound dictionaries
            
        Returns:
            List of screening results
        """
        if not self.initialized:
            self.initialize()
        
        print(f"Batch screening {len(compounds_list)} compounds...")
        
        results = []
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(compounds_list), batch_size):
            batch = compounds_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(compounds_list) + batch_size - 1)//batch_size}")
            
            for compound in batch:
                try:
                    result = self.screen_compound(compound)
                    results.append(result)
                except Exception as e:
                    print(f"Error screening compound {compound.get('name', 'Unknown')}: {e}")
                    results.append({
                        'compound_id': compound.get('compound_id', 'unknown'),
                        'error': str(e)
                    })
        
        # Sort by drug-likeness score
        results.sort(key=lambda x: x.get('drug_likeness', {}).get('overall_score', 0), reverse=True)
        
        return results
    
    def _extract_features(self, compound_data: Dict) -> np.ndarray:
        """Extract molecular features for ML models"""
        features = []
        
        # 1. Molecular fingerprint
        smiles = compound_data.get('smiles', '')
        if smiles and self.feature_encoder:
            fingerprint = self.feature_encoder.encode(smiles)
            features.extend(fingerprint)
        
        # 2. Molecular descriptors
        descriptors = self._calculate_molecular_descriptors(compound_data)
        features.extend(descriptors)
        
        # 3. Pharmacophore features
        pharmacophore = self._extract_pharmacophore_features(compound_data)
        features.extend(pharmacophore)
        
        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)
        
        # Scale features
        if hasattr(self.scaler, 'transform'):
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)
            features_array = self.scaler.transform(features_array)
            if features_array.shape[0] == 1:
                features_array = features_array.flatten()
        
        return features_array
    
    def _calculate_molecular_descriptors(self, compound_data: Dict) -> List[float]:
        """Calculate molecular descriptors"""
        descriptors = []
        
        # Basic properties
        descriptors.append(compound_data.get('molecular_weight', 300.0) / 1000.0)  # Normalized
        descriptors.append(compound_data.get('logp', 2.0))
        descriptors.append(compound_data.get('h_bond_donors', 2) / 10.0)  # Normalized
        descriptors.append(compound_data.get('h_bond_acceptors', 5) / 20.0)  # Normalized
        descriptors.append(compound_data.get('rotatable_bonds', 5) / 20.0)  # Normalized
        descriptors.append(compound_data.get('polar_surface_area', 100.0) / 200.0)  # Normalized
        
        # Advanced descriptors (simulated)
        descriptors.append(self._calculate_molecular_complexity(compound_data))
        descriptors.append(self._calculate_flexibility_index(compound_data))
        descriptors.append(self._calculate_polarity_index(compound_data))
        descriptors.append(self._calculate_aromatic_ratio(compound_data))
        
        return descriptors
    
    def _calculate_molecular_complexity(self, compound_data: Dict) -> float:
        """Calculate molecular complexity score"""
        # Simplified complexity calculation
        mw = compound_data.get('molecular_weight', 300.0)
        rot_bonds = compound_data.get('rotatable_bonds', 5)
        rings = compound_data.get('num_rings', 2)
        
        complexity = (mw / 500.0) * 0.4 + (rot_bonds / 20.0) * 0.3 + (rings / 10.0) * 0.3
        return complexity
    
    def _calculate_flexibility_index(self, compound_data: Dict) -> float:
        """Calculate molecular flexibility index"""
        rot_bonds = compound_data.get('rotatable_bonds', 5)
        mw = compound_data.get('molecular_weight', 300.0)
        
        # Flexibility increases with rotatable bonds relative to size
        flexibility = rot_bonds / (mw / 100.0)
        return min(flexibility, 1.0)
    
    def _calculate_polarity_index(self, compound_data: Dict) -> float:
        """Calculate molecular polarity index"""
        hba = compound_data.get('h_bond_acceptors', 5)
        hbd = compound_data.get('h_bond_donors', 2)
        psa = compound_data.get('polar_surface_area', 100.0)
        
        polarity = (hba + hbd) / 20.0 * 0.5 + (psa / 200.0) * 0.5
        return polarity
    
    def _calculate_aromatic_ratio(self, compound_data: Dict) -> float:
        """Calculate aromatic ratio"""
        rings = compound_data.get('num_rings', 2)
        aromatic = compound_data.get('aromatic', False)
        
        if rings == 0:
            return 0.0
        
        aromatic_ratio = (1.0 if aromatic else 0.5) * (min(rings, 5) / 5.0)
        return aromatic_ratio
    
    def _extract_pharmacophore_features(self, compound_data: Dict) -> List[float]:
        """Extract pharmacophore features"""
        # Simplified pharmacophore features
        features = [0.0] * 20  # 20 pharmacophore features
        
        smiles = compound_data.get('smiles', '')
        if not smiles:
            return features
        
        # Simulate presence of common pharmacophore features
        # Based on SMILES string patterns
        if 'N' in smiles and 'C=O' in smiles:
            features[0] = 1.0  # Amide bond
        if 'O' in smiles.count('O') > 2:
            features[1] = 1.0  # Multiple oxygen atoms
        if 'N' in smiles.count('N') > 1:
            features[2] = 1.0  # Multiple nitrogen atoms
        if 'S' in smiles:
            features[3] = 1.0  # Sulfur atom
        if 'P' in smiles:
            features[4] = 1.0  # Phosphorus atom
        if 'F' in smiles or 'Cl' in smiles or 'Br' in smiles or 'I' in smiles:
            features[5] = 1.0  # Halogen atom
        if 'C1=CC=CC=C1' in smiles or smiles.count('c') > 3:
            features[6] = 1.0  # Aromatic ring
        
        # Hydrogen bond donors/acceptors
        hbd = compound_data.get('h_bond_donors', 0)
        hba = compound_data.get('h_bond_acceptors', 0)
        
        features[7] = min(hbd / 5.0, 1.0)  # Normalized HBD count
        features[8] = min(hba / 10.0, 1.0)  # Normalized HBA count
        
        # Charge features
        features[9] = 0.2 if 'N+' in smiles else 0.0  # Positive charge
        features[10] = 0.2 if 'O-' in smiles or 'C(=O)[O-]' in smiles else 0.0  # Negative charge
        
        # Hydrophobic features
        long_chain = any(pattern in smiles for pattern in ['CCCC', 'CCCCC', 'CCCCCC'])
        features[11] = 1.0 if long_chain else 0.0
        
        return features
    
    def _predict_efficacy(self, features: np.ndarray) -> Dict:
        """Predict drug efficacy"""
        if self.model_type == 'xgboost':
            return self._predict_efficacy_xgboost(features)
        else:
            return self._predict_efficacy_nn(features)
    
    def _predict_efficacy_nn(self, features: np.ndarray) -> Dict:
        """Predict efficacy using neural network"""
        if not self.efficacy_model:
            return {'error': 'Efficacy model not initialized'}
        
        # Prepare input
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Make prediction
        self.efficacy_model.eval()
        with torch.no_grad():
            prediction = self.efficacy_model(features_tensor)
        
        # Convert to probability
        efficacy_score = prediction.cpu().numpy().flatten()[0]
        
        # Calculate confidence
        confidence = self._calculate_confidence(features, 'efficacy')
        
        return {
            'score': float(efficacy_score),
            'confidence': confidence,
            'interpretation': self._interpret_efficacy_score(efficacy_score),
            'targets': self._predict_efficacy_targets(features_tensor)
        }
    
    def _predict_efficacy_xgboost(self, features: np.ndarray) -> Dict:
        """Predict efficacy using XGBoost"""
        if not hasattr(self, 'efficacy_xgb'):
            return {'error': 'XGBoost efficacy model not initialized'}
        
        # Prepare input
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        efficacy_score = self.efficacy_xgb.predict(features)[0]
        
        return {
            'score': float(efficacy_score),
            'confidence': 0.8,  # XGBoost confidence estimation
            'interpretation': self._interpret_efficacy_score(efficacy_score)
        }
    
    def _interpret_efficacy_score(self, score: float) -> str:
        """Interpret efficacy score"""
        if score >= 0.8:
            return "High potential - Strong predicted efficacy"
        elif score >= 0.6:
            return "Moderate potential - Good predicted efficacy"
        elif score >= 0.4:
            return "Low potential - Marginal predicted efficacy"
        else:
            return "Poor potential - Weak predicted efficacy"
    
    def _predict_efficacy_targets(self, features_tensor: torch.Tensor) -> Dict:
        """Predict efficacy against specific targets"""
        # In production, this would be a multi-task model
        # For simulation, generate plausible predictions
        
        common_targets = [
            'GPCR', 'Kinase', 'Protease', 'Nuclear Receptor',
            'Ion Channel', 'Enzyme', 'Transporter'
        ]
        
        predictions = {}
        for target in common_targets:
            # Generate target-specific prediction
            target_hash = hash(target) % 1000 / 1000.0
            feature_mean = features_tensor.mean().item()
            
            # Combine with some randomness
            target_score = 0.5 + 0.3 * np.sin(target_hash * 10) + 0.2 * feature_mean
            target_score = np.clip(target_score, 0.0, 1.0)
            
            predictions[target] = {
                'score': float(target_score),
                'confidence': float(0.7 + 0.2 * np.cos(target_hash * 5)),
                'mechanism': self._predict_mechanism(target, target_score)
            }
        
        return predictions
    
    def _predict_mechanism(self, target: str, score: float) -> str:
        """Predict mechanism of action"""
        mechanisms = {
            'GPCR': ['Agonist', 'Antagonist', 'Inverse Agonist', 'Allosteric Modulator'],
            'Kinase': ['Inhibitor', 'Activator', 'ATP-competitive', 'Allosteric'],
            'Protease': ['Inhibitor', 'Activator', 'Substrate Mimetic'],
            'Nuclear Receptor': ['Agonist', 'Antagonist', 'Partial Agonist']
        }
        
        if target in mechanisms:
            available_mechanisms = mechanisms[target]
            idx = int(score * len(available_mechanisms)) % len(available_mechanisms)
            return available_mechanisms[idx]
        
        return 'Unknown'
    
    def _predict_toxicity(self, features: np.ndarray) -> Dict:
        """Predict toxicity endpoints"""
        if self.model_type == 'xgboost':
            return self._predict_toxicity_xgboost(features)
        else:
            return self._predict_toxicity_nn(features)
    
    def _predict_toxicity_nn(self, features: np.ndarray) -> Dict:
        """Predict toxicity using neural network"""
        if not self.toxicity_model:
            return {'error': 'Toxicity model not initialized'}
        
        # Prepare input
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Make prediction
        self.toxicity_model.eval()
        with torch.no_grad():
            predictions = self.toxicity_model(features_tensor)
        
        # Get probabilities
        probabilities = torch.softmax(predictions, dim=1).cpu().numpy().flatten()
        
        # Toxicity endpoints
        endpoints = [
            'Hepatotoxicity', 'Nephrotoxicity', 'Cardiotoxicity', 
            'Neurotoxicity', 'Mutagenicity', 'Carcinogenicity',
            'Reproductive Toxicity', 'Immunotoxicity', 
            'Skin Sensitization', 'Respiratory Toxicity'
        ]
        
        # Calculate overall toxicity score
        weights = np.array([0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.075, 0.075, 0.05, 0.05])
        overall_toxicity = np.sum(probabilities * weights)
        
        # Get top toxicities
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_toxicities = [
            {
                'endpoint': endpoints[idx],
                'probability': float(probabilities[idx]),
                'risk_level': self._get_toxicity_risk_level(probabilities[idx])
            }
            for idx in top_indices
        ]
        
        # Calculate confidence
        confidence = self._calculate_confidence(features, 'toxicity')
        
        return {
            'overall_score': float(overall_toxicity),
            'confidence': confidence,
            'endpoint_predictions': {
                endpoints[i]: {
                    'probability': float(prob),
                    'risk_level': self._get_toxicity_risk_level(prob)
                }
                for i, prob in enumerate(probabilities)
            },
            'top_toxicities': top_toxicities,
            'safety_margin': self._calculate_safety_margin(overall_toxicity),
            'warning_flags': self._identify_toxicity_flags(probabilities, endpoints)
        }
    
    def _predict_toxicity_xgboost(self, features: np.ndarray) -> Dict:
        """Predict toxicity using XGBoost"""
        if not hasattr(self, 'toxicity_xgb'):
            return {'error': 'XGBoost toxicity model not initialized'}
        
        # Prepare input
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Make prediction
        probabilities = self.toxicity_xgb.predict_proba(features)[0]
        
        endpoints = [
            'Hepatotoxicity', 'Nephrotoxicity', 'Cardiotoxicity', 
            'Neurotoxicity', 'Mutagenicity', 'Carcinogenicity',
            'Reproductive Toxicity', 'Immunotoxicity', 
            'Skin Sensitization', 'Respiratory Toxicity'
        ]
        
        overall_toxicity = np.mean(probabilities)
        
        return {
            'overall_score': float(overall_toxicity),
            'confidence': 0.8,
            'endpoint_predictions': {
                endpoints[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        }
    
    def _get_toxicity_risk_level(self, probability: float) -> str:
        """Get risk level from toxicity probability"""
        if probability >= 0.7:
            return "High Risk"
        elif probability >= 0.4:
            return "Moderate Risk"
        elif probability >= 0.2:
            return "Low Risk"
        else:
            return "Minimal Risk"
    
    def _calculate_safety_margin(self, toxicity_score: float) -> float:
        """Calculate safety margin"""
        # Safety margin is inverse of toxicity
        safety_margin = max(0.0, 1.0 - toxicity_score)
        
        # Adjust based on typical drug profiles
        if safety_margin > 0.8:
            safety_margin *= 1.1  # Bonus for very safe compounds
        elif safety_margin < 0.3:
            safety_margin *= 0.8  # Penalty for unsafe compounds
        
        return round(safety_margin, 3)
    
    def _identify_toxicity_flags(self, probabilities: np.ndarray, endpoints: List[str]) -> List[str]:
        """Identify toxicity warning flags"""
        flags = []
        
        # Check for high-risk toxicities
        high_risk_threshold = 0.7
        for i, prob in enumerate(probabilities):
            if prob >= high_risk_threshold:
                flags.append(f"High {endpoints[i]} risk ({prob:.1%})")
        
        # Check for multiple moderate risks
        moderate_count = sum(1 for prob in probabilities if 0.4 <= prob < 0.7)
        if moderate_count >= 3:
            flags.append(f"Multiple moderate toxicity risks ({moderate_count} endpoints)")
        
        # Check for specific concerning combinations
        if (probabilities[0] > 0.5 and probabilities[2] > 0.5):  # Hepatotoxicity + Cardiotoxicity
            flags.append("Concerning hepatocardiotoxicity profile")
        
        if probabilities[4] > 0.6 or probabilities[5] > 0.6:  # Mutagenicity or Carcinogenicity
            flags.append("Genotoxicity concern")
        
        return flags
    
    def _predict_adme(self, features: np.ndarray) -> Dict:
        """Predict ADME properties"""
        if self.model_type == 'xgboost':
            return self._predict_adme_simulated(features)  # XGBoost ADME not implemented
        else:
            return self._predict_adme_nn(features)
    
    def _predict_adme_nn(self, features: np.ndarray) -> Dict:
        """Predict ADME properties using neural network"""
        if not self.adme_model:
            return self._predict_adme_simulated(features)
        
        # Prepare input
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Make prediction
        self.adme_model.eval()
        with torch.no_grad():
            predictions = self.adme_model(features_tensor)
        
        predictions = predictions.cpu().numpy().flatten()
        
        adme_properties = [
            'Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Bioavailability'
        ]
        
        adme_predictions = {}
        for i, prop in enumerate(adme_properties):
            score = float(predictions[i])
            adme_predictions[prop] = {
                'score': score,
                'prediction': self._interpret_adme_property(prop, score),
                'optimal_range': self._get_adme_optimal_range(prop)
            }
        
        # Calculate overall ADME score
        weights = np.array([0.25, 0.2, 0.25, 0.15, 0.15])  # Emphasize Absorption, Metabolism, Bioavailability
        overall_adme = np.sum(predictions * weights)
        
        # Identify ADME issues
        issues = self._identify_adme_issues(adme_predictions)
        
        return {
            'overall_score': float(overall_adme),
            'properties': adme_predictions,
            'issues': issues,
            'adme_class': self._classify_adme_profile(adme_predictions)
        }
    
    def _predict_adme_simulated(self, features: np.ndarray) -> Dict:
        """Generate simulated ADME predictions"""
        # Use features to generate deterministic but plausible ADME predictions
        feature_hash = hash(tuple(features[:10])) % 1000 / 1000.0
        
        adme_properties = [
            'Absorption', 'Distribution', 'Metabolism', 'Excretion', 'Bioavailability'
        ]
        
        # Generate scores based on feature hash
        base_scores = []
        for i in range(5):
            score = 0.5 + 0.3 * np.sin(feature_hash * (i+1) * 10) + 0.2 * (features[i] if i < len(features) else 0)
            base_scores.append(np.clip(score, 0.0, 1.0))
        
        adme_predictions = {}
        for i, prop in enumerate(adme_properties):
            score = float(base_scores[i])
            adme_predictions[prop] = {
                'score': score,
                'prediction': self._interpret_adme_property(prop, score),
                'optimal_range': self._get_adme_optimal_range(prop)
            }
        
        # Calculate overall ADME
        weights = np.array([0.25, 0.2, 0.25, 0.15, 0.15])
        overall_adme = np.sum(base_scores * weights)
        
        return {
            'overall_score': float(overall_adme),
            'properties': adme_predictions,
            'issues': ['Simulated predictions - use neural network for accurate results'],
            'adme_class': 'Simulated'
        }
    
    def _interpret_adme_property(self, property_name: str, score: float) -> str:
        """Interpret ADME property score"""
        interpretations = {
            'Absorption': {
                (0.8, 1.0): "Excellent absorption",
                (0.6, 0.8): "Good absorption",
                (0.4, 0.6): "Moderate absorption",
                (0.0, 0.4): "Poor absorption"
            },
            'Distribution': {
                (0.7, 1.0): "Good tissue distribution",
                (0.4, 0.7): "Moderate distribution",
                (0.0, 0.4): "Limited distribution"
            },
            'Metabolism': {
                (0.6, 1.0): "Slow metabolism (long half-life)",
                (0.4, 0.6): "Moderate metabolism",
                (0.0, 0.4): "Rapid metabolism (short half-life)"
            },
            'Excretion': {
                (0.6, 1.0): "Efficient excretion",
                (0.4, 0.6): "Moderate excretion",
                (0.0, 0.4): "Poor excretion (accumulation risk)"
            },
            'Bioavailability': {
                (0.8, 1.0): "High bioavailability",
                (0.6, 0.8): "Good bioavailability",
                (0.4, 0.6): "Moderate bioavailability",
                (0.0, 0.4): "Low bioavailability"
            }
        }
        
        if property_name in interpretations:
            for (low, high), interpretation in interpretations[property_name].items():
                if low <= score < high:
                    return interpretation
        
        return "Unknown"
    
    def _get_adme_optimal_range(self, property_name: str) -> Dict:
        """Get optimal range for ADME property"""
        ranges = {
            'Absorption': {'min': 0.6, 'max': 0.9, 'ideal': 0.8},
            'Distribution': {'min': 0.5, 'max': 0.8, 'ideal': 0.65},
            'Metabolism': {'min': 0.4, 'max': 0.7, 'ideal': 0.55},
            'Excretion': {'min': 0.5, 'max': 0.8, 'ideal': 0.65},
            'Bioavailability': {'min': 0.6, 'max': 0.9, 'ideal': 0.75}
        }
        return ranges.get(property_name, {'min': 0.0, 'max': 1.0, 'ideal': 0.5})
    
    def _identify_adme_issues(self, adme_predictions: Dict) -> List[str]:
        """Identify ADME issues"""
        issues = []
        
        for prop_name, prop_data in adme_predictions.items():
            score = prop_data['score']
            optimal_range = prop_data['optimal_range']
            
            if score < optimal_range['min']:
                issues.append(f"Low {prop_name} ({score:.2f} < {optimal_range['min']})")
            elif score > optimal_range['max']:
                issues.append(f"High {prop_name} ({score:.2f} > {optimal_range['max']})")
        
        # Specific issue combinations
        if ('Low Absorption' in issues or 'Low Bioavailability' in issues) and 'Rapid metabolism' in prop_data.get('prediction', ''):
            issues.append("Poor oral bioavailability due to absorption and metabolism issues")
        
        return issues
    
    def _classify_adme_profile(self, adme_predictions: Dict) -> str:
        """Classify overall ADME profile"""
        scores = [data['score'] for data in adme_predictions.values()]
        avg_score = np.mean(scores)
        
        if avg_score >= 0.75:
            return "Excellent ADME Profile"
        elif avg_score >= 0.6:
            return "Good ADME Profile"
        elif avg_score >= 0.45:
            return "Moderate ADME Profile"
        else:
            return "Poor ADME Profile"
    
    def _calculate_drug_likeness(self, efficacy_pred: Dict, toxicity_pred: Dict, adme_pred: Dict) -> Dict:
        """Calculate overall drug-likeness score"""
        # Extract scores
        efficacy_score = efficacy_pred.get('score', 0.5)
        toxicity_score = toxicity_pred.get('overall_score', 0.5)
        adme_score = adme_pred.get('overall_score', 0.5)
        
        # Calculate safety margin
        safety_margin = toxicity_pred.get('safety_margin', 1.0 - toxicity_score)
        
        # Weighted combination
        weights = {
            'efficacy': 0.4,
            'safety': 0.35,
            'adme': 0.25
        }
        
        overall_score = (
            weights['efficacy'] * efficacy_score +
            weights['safety'] * safety_margin +
            weights['adme'] * adme_score
        )
        
        # Apply rule-based adjustments
        adjustments = []
        
        # Penalize high toxicity
        if toxicity_score > 0.7:
            overall_score *= 0.7
            adjustments.append("High toxicity penalty")
        
        # Penalize low efficacy
        if efficacy_score < 0.3:
            overall_score *= 0.8
            adjustments.append("Low efficacy penalty")
        
        # Bonus for excellent ADME
        if adme_score > 0.8:
            overall_score *= 1.1
            adjustments.append("Excellent ADME bonus")
        
        # Bonus for high safety margin
        if safety_margin > 0.8:
            overall_score *= 1.05
            adjustments.append("High safety margin bonus")
        
        # Ensure score is in [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Classification
        if overall_score >= 0.75:
            classification = "Excellent Drug Candidate"
            recommendation = "Proceed to experimental validation"
        elif overall_score >= 0.6:
            classification = "Good Drug Candidate"
            recommendation = "Consider for further optimization"
        elif overall_score >= 0.45:
            classification = "Marginal Candidate"
            recommendation = "Requires significant optimization"
        else:
            classification = "Poor Candidate"
            recommendation = "Consider alternative compounds"
        
        return {
            'overall_score': round(overall_score, 3),
            'component_scores': {
                'efficacy': round(efficacy_score, 3),
                'safety_margin': round(safety_margin, 3),
                'adme': round(adme_score, 3)
            },
            'classification': classification,
            'adjustments': adjustments,
            'recommendation': recommendation
        }
    
    def _assess_risks(self, efficacy_pred: Dict, toxicity_pred: Dict) -> Dict:
        """Assess overall risks"""
        efficacy_score = efficacy_pred.get('score', 0.5)
        toxicity_score = toxicity_pred.get('overall_score', 0.5)
        
        # Calculate risk-benefit ratio
        if efficacy_score > 0:
            risk_benefit_ratio = toxicity_score / efficacy_score
        else:
            risk_benefit_ratio = float('inf')
        
        # Risk classification
        if risk_benefit_ratio < 0.5:
            risk_level = "Low Risk"
        elif risk_benefit_ratio < 1.0:
            risk_level = "Moderate Risk"
        elif risk_benefit_ratio < 2.0:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        # Identify specific risks
        specific_risks = []
        
        # Efficacy risks
        if efficacy_score < 0.4:
            specific_risks.append("Low predicted efficacy")
        
        # Toxicity risks
        top_toxicities = toxicity_pred.get('top_toxicities', [])
        for toxicity in top_toxicities:
            if toxicity.get('probability', 0) > 0.6:
                specific_risks.append(f"High {toxicity['endpoint']} risk")
        
        return {
            'risk_benefit_ratio': round(risk_benefit_ratio, 2),
            'risk_level': risk_level,
            'specific_risks': specific_risks,
            'monitoring_recommendations': self._generate_monitoring_recommendations(risk_level, specific_risks)
        }
    
    def _generate_monitoring_recommendations(self, risk_level: str, specific_risks: List[str]) -> List[str]:
        """Generate monitoring recommendations based on risks"""
        recommendations = []
        
        if risk_level in ["High Risk", "Very High Risk"]:
            recommendations.append("Close monitoring in preclinical studies")
            recommendations.append("Consider lower starting dose in clinical trials")
        
        if any("Hepatotoxicity" in risk for risk in specific_risks):
            recommendations.append("Monitor liver function tests")
            recommendations.append("Consider hepatoprotective agents")
        
        if any("Cardiotoxicity" in risk for risk in specific_risks):
            recommendations.append("Monitor ECG and cardiac biomarkers")
            recommendations.append("Consider cardioprotective measures")
        
        if any("Nephrotoxicity" in risk for risk in specific_risks):
            recommendations.append("Monitor renal function")
            recommendations.append("Ensure adequate hydration")
        
        return recommendations
    
    def _generate_recommendation(self, predictions: Dict) -> Dict:
        """Generate overall recommendation"""
        drug_likeness = predictions.get('drug_likeness', {})
        risk_assessment = predictions.get('risk_assessment', {})
        
        overall_score = drug_likeness.get('overall_score', 0.5)
        risk_level = risk_assessment.get('risk_level', 'Moderate Risk')
        
        # Decision logic
        if overall_score >= 0.7 and risk_level in ["Low Risk", "Moderate Risk"]:
            decision = "APPROVE - Proceed to next phase"
            priority = "High"
            next_steps = ["Preclinical validation", "Formulation development", "Toxicology studies"]
        elif overall_score >= 0.5:
            decision = "CONSIDER - Requires optimization"
            priority = "Medium"
            next_steps = ["Lead optimization", "SAR studies", "Improve ADME properties"]
        else:
            decision = "REJECT - Poor candidate profile"
            priority = "Low"
            next_steps = ["Consider alternative scaffolds", "Re-evaluate screening criteria"]
        
        return {
            'decision': decision,
            'priority': priority,
            'next_steps': next_steps,
            'rationale': self._generate_recommendation_rationale(predictions),
            'confidence': self._calculate_overall_confidence(predictions)
        }
    
    def _generate_recommendation_rationale(self, predictions: Dict) -> str:
        """Generate rationale for recommendation"""
        efficacy_score = predictions.get('efficacy', {}).get('score', 0.5)
        toxicity_score = predictions.get('toxicity', {}).get('overall_score', 0.5)
        adme_score = predictions.get('adme', {}).get('overall_score', 0.5)
        
        rationales = []
        
        if efficacy_score >= 0.7:
            rationales.append(f"Strong efficacy prediction ({efficacy_score:.2f})")
        elif efficacy_score <= 0.3:
            rationales.append(f"Weak efficacy prediction ({efficacy_score:.2f})")
        
        if toxicity_score >= 0.7:
            rationales.append(f"High toxicity risk ({toxicity_score:.2f})")
        elif toxicity_score <= 0.3:
            rationales.append(f"Low toxicity risk ({toxicity_score:.2f})")
        
        if adme_score >= 0.7:
            rationales.append(f"Good ADME profile ({adme_score:.2f})")
        elif adme_score <= 0.3:
            rationales.append(f"Poor ADME profile ({adme_score:.2f})")
        
        return "; ".join(rationales)
    
    def _calculate_overall_confidence(self, predictions: Dict) -> float:
        """Calculate overall confidence in predictions"""
        confidences = []
        
        # Efficacy confidence
        if 'efficacy' in predictions:
            confidences.append(predictions['efficacy'].get('confidence', 0.7))
        
        # Toxicity confidence
        if 'toxicity' in predictions:
            confidences.append(predictions['toxicity'].get('confidence', 0.7))
        
        # ADME confidence
        if 'adme' in predictions:
            confidences.append(0.6)  # ADME predictions typically less confident
        
        if confidences:
            overall_confidence = np.mean(confidences)
            
            # Adjust based on consistency
            efficacy_score = predictions.get('efficacy', {}).get('score', 0.5)
            toxicity_score = predictions.get('toxicity', {}).get('overall_score', 0.5)
            
            # High confidence if predictions are clear (very high or very low scores)
            if efficacy_score > 0.8 or efficacy_score < 0.2:
                overall_confidence *= 1.1
            
            if toxicity_score > 0.8 or toxicity_score < 0.2:
                overall_confidence *= 1.1
            
            return min(1.0, overall_confidence)
        
        return 0.7  # Default confidence
    
    def _calculate_confidence(self, features: np.ndarray, model_type: str) -> float:
        """Calculate prediction confidence"""
        # Confidence based on feature quality and model uncertainty
        base_confidence = 0.7
        
        # Adjust based on feature vector
        if len(features) > 100:
            base_confidence += 0.1  # More features = more confidence
        
        # Adjust based on feature variability
        feature_variance = np.var(features)
        if feature_variance > 0.1:
            base_confidence += 0.05  # Good feature variance
        
        # Model-specific adjustments
        if model_type == 'efficacy':
            # Efficacy models typically have higher confidence
            base_confidence += 0.05
        elif model_type == 'toxicity':
            # Toxicity predictions are more uncertain
            base_confidence -= 0.05
        
        # Ensure within bounds
        return max(0.5, min(0.95, base_confidence))
    
    def train_models(self, training_data: pd.DataFrame, 
                    validation_data: pd.DataFrame = None,
                    save_path: str = './models/ai_screening/') -> Dict:
        """
        Train AI models on provided data
        
        Args:
            training_data: DataFrame with training data
            validation_data: DataFrame with validation data
            save_path: Path to save trained models
            
        Returns:
            Training results and metrics
        """
        print("Training AI screening models...")
        
        # Prepare data
        X_train, y_train = self._prepare_training_data(training_data)
        
        if validation_data is not None:
            X_val, y_val = self._prepare_training_data(validation_data)
        else:
            # Split training data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Train efficacy model
        efficacy_metrics = self._train_efficacy_model(X_train, y_train['efficacy'], 
                                                     X_val, y_val['efficacy'])
        
        # Train toxicity model
        toxicity_metrics = self._train_toxicity_model(X_train, y_train['toxicity'],
                                                     X_val, y_val['toxicity'])
        
        # Train ADME model
        adme_metrics = self._train_adme_model(X_train, y_train['adme'],
                                             X_val, y_val['adme'])
        
        # Save models
        self._save_models(save_path)
        
        return {
            'efficacy_metrics': efficacy_metrics,
            'toxicity_metrics': toxicity_metrics,
            'adme_metrics': adme_metrics,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Prepare training data from DataFrame"""
        # Extract features and labels
        X = []
        y_efficacy = []
        y_toxicity = []
        y_adme = []
        
        for _, row in data.iterrows():
            # Extract features from row
            features = self._extract_features_from_row(row)
            X.append(features)
            
            # Extract labels (assuming specific column names)
            y_efficacy.append(row.get('efficacy_score', 0.5))
            
            # Toxicity labels (multi-hot encoding)
            toxicity_labels = np.zeros(self.num_toxicity_classes)
            for i in range(self.num_toxicity_classes):
                col_name = f'toxicity_{i}'
                if col_name in row:
                    toxicity_labels[i] = row[col_name]
            y_toxicity.append(toxicity_labels)
            
            # ADME labels
            adme_labels = []
            for prop in ['absorption', 'distribution', 'metabolism', 'excretion', 'bioavailability']:
                adme_labels.append(row.get(f'{prop}_score', 0.5))
            y_adme.append(adme_labels)
        
        X = np.array(X)
        y_efficacy = np.array(y_efficacy)
        y_toxicity = np.array(y_toxicity)
        y_adme = np.array(y_adme)
        
        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, {
            'efficacy': y_efficacy,
            'toxicity': y_toxicity,
            'adme': y_adme
        }
    
    def _extract_features_from_row(self, row: pd.Series) -> np.ndarray:
        """Extract features from DataFrame row"""
        features = []
        
        # Molecular descriptors
        descriptors = [
            row.get('molecular_weight', 300.0),
            row.get('logp', 2.0),
            row.get('h_bond_donors', 2),
            row.get('h_bond_acceptors', 5),
            row.get('rotatable_bonds', 5),
            row.get('polar_surface_area', 100.0)
        ]
        features.extend(descriptors)
        
        # Add random features for simulation
        # In production, these would be real molecular features
        num_additional_features = self.num_features - len(features)
        if num_additional_features > 0:
            # Generate deterministic random features based on row data
            seed = hash(tuple(row.values)) % 2**32
            np.random.seed(seed)
            random_features = np.random.randn(num_additional_features)
            features.extend(random_features)
        
        return np.array(features[:self.num_features], dtype=np.float32)
    
    def _train_efficacy_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train efficacy prediction model"""
        if self.model_type == 'xgboost':
            return self._train_efficacy_xgboost(X_train, y_train, X_val, y_val)
        else:
            return self._train_efficacy_nn(X_train, y_train, X_val, y_val)
    
    def _train_efficacy_nn(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train neural network efficacy model"""
        if not self.efficacy_model:
            return {'error': 'Efficacy model not initialized'}
        
        print("Training efficacy model...")
        
        # Create datasets
        train_dataset = MolecularDataset(X_train, y_train)
        val_dataset = MolecularDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.efficacy_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.efficacy_optimizer.zero_grad()
                predictions = self.efficacy_model(batch_X)
                loss = self.efficacy_loss_fn(predictions.squeeze(), batch_y)
                loss.backward()
                self.efficacy_optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.efficacy_model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    predictions = self.efficacy_model(batch_X)
                    loss = self.efficacy_loss_fn(predictions.squeeze(), batch_y)
                    val_loss += loss.item()
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            val_predictions = np.array(val_predictions).flatten()
            val_targets = np.array(val_targets)
            
            # Calculate R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(val_targets, val_predictions)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.efficacy_model.state_dict(), 
                          f'./models/ai_screening/efficacy_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Store history
            self.training_history['efficacy'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'r2_score': r2
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R²: {r2:.4f}")
        
        # Load best model
        self.efficacy_model.load_state_dict(
            torch.load('./models/ai_screening/efficacy_best.pth', 
                      map_location=self.device)
        )
        
        # Final evaluation
        final_metrics = self.training_history['efficacy'][-1]
        
        return {
            'final_val_loss': final_metrics['val_loss'],
            'final_r2_score': final_metrics['r2_score'],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.training_history['efficacy']),
            'training_history': self.training_history['efficacy']
        }
    
    def _train_efficacy_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train XGBoost efficacy model"""
        if not hasattr(self, 'efficacy_xgb'):
            return {'error': 'XGBoost efficacy model not initialized'}
        
        print("Training XGBoost efficacy model...")
        
        # Train model
        self.efficacy_xgb.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.efficacy_xgb.predict(X_train)
        val_pred = self.efficacy_xgb.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        # Save model
        import joblib
        joblib.dump(self.efficacy_xgb, './models/ai_screening/efficacy_xgb.pkl')
        
        return {
            'train_r2_score': train_r2,
            'val_r2_score': val_r2,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'model_type': 'xgboost'
        }
    
    def _train_toxicity_model(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train toxicity prediction model"""
        if self.model_type == 'xgboost':
            return self._train_toxicity_xgboost(X_train, y_train, X_val, y_val)
        else:
            return self._train_toxicity_nn(X_train, y_train, X_val, y_val)
    
    def _train_toxicity_nn(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train neural network toxicity model"""
        if not self.toxicity_model:
            return {'error': 'Toxicity model not initialized'}
        
        print("Training toxicity model...")
        
        # Convert to class labels for classification
        # Assuming y_train contains multi-label binary classifications
        y_train_labels = np.argmax(y_train, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        
        # Create datasets
        train_dataset = MolecularDataset(X_train, y_train_labels)
        val_dataset = MolecularDataset(X_val, y_val_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.toxicity_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.toxicity_optimizer.zero_grad()
                predictions = self.toxicity_model(batch_X)
                loss = self.toxicity_loss_fn(predictions, batch_y)
                loss.backward()
                self.toxicity_optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(predictions, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            self.toxicity_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    predictions = self.toxicity_model(batch_X)
                    loss = self.toxicity_loss_fn(predictions, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(predictions, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_accuracy = train_correct / train_total
            val_accuracy = val_correct / val_total
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.toxicity_model.state_dict(),
                          f'./models/ai_screening/toxicity_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Store history
            self.training_history['toxicity'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Load best model
        self.toxicity_model.load_state_dict(
            torch.load('./models/ai_screening/toxicity_best.pth',
                      map_location=self.device)
        )
        
        # Final evaluation
        final_metrics = self.training_history['toxicity'][-1]
        
        return {
            'final_val_loss': final_metrics['val_loss'],
            'final_val_accuracy': final_metrics['val_accuracy'],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.training_history['toxicity']),
            'training_history': self.training_history['toxicity']
        }
    
    def _train_toxicity_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train XGBoost toxicity model"""
        if not hasattr(self, 'toxicity_xgb'):
            return {'error': 'XGBoost toxicity model not initialized'}
        
        print("Training XGBoost toxicity model...")
        
        # Convert to class labels
        y_train_labels = np.argmax(y_train, axis=1)
        y_val_labels = np.argmax(y_val, axis=1)
        
        # Train model
        self.toxicity_xgb.fit(X_train, y_train_labels)
        
        # Make predictions
        train_pred = self.toxicity_xgb.predict(X_train)
        val_pred = self.toxicity_xgb.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        
        train_acc = accuracy_score(y_train_labels, train_pred)
        val_acc = accuracy_score(y_val_labels, val_pred)
        
        train_f1 = f1_score(y_train_labels, train_pred, average='weighted')
        val_f1 = f1_score(y_val_labels, val_pred, average='weighted')
        
        # Save model
        import joblib
        joblib.dump(self.toxicity_xgb, './models/ai_screening/toxicity_xgb.pkl')
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_f1_score': train_f1,
            'val_f1_score': val_f1,
            'model_type': 'xgboost'
        }
    
    def _train_adme_model(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train ADME prediction model"""
        if self.model_type == 'xgboost':
            return self._train_adme_simulated()
        else:
            return self._train_adme_nn(X_train, y_train, X_val, y_val)
    
    def _train_adme_nn(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train neural network ADME model"""
        if not self.adme_model:
            return self._train_adme_simulated()
        
        print("Training ADME model...")
        
        # Create datasets
        train_dataset = MolecularDataset(X_train, y_train)
        val_dataset = MolecularDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training
            self.adme_model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.adme_optimizer.zero_grad()
                predictions = self.adme_model(batch_X)
                loss = self.adme_loss_fn(predictions, batch_y)
                loss.backward()
                self.adme_optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.adme_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    predictions = self.adme_model(batch_X)
                    loss = self.adme_loss_fn(predictions, batch_y)
                    val_loss += loss.item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Calculate R² for each ADME property
            self.adme_model.eval()
            with torch.no_grad():
                val_predictions = []
                val_targets = []
                
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    predictions = self.adme_model(batch_X)
                    val_predictions.append(predictions.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())
                
                val_predictions = np.vstack(val_predictions)
                val_targets = np.vstack(val_targets)
                
                from sklearn.metrics import r2_score
                r2_scores = []
                for i in range(val_predictions.shape[1]):
                    r2 = r2_score(val_targets[:, i], val_predictions[:, i])
                    r2_scores.append(r2)
            
            avg_r2 = np.mean(r2_scores)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.adme_model.state_dict(),
                          f'./models/ai_screening/adme_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Store history
            self.training_history['adme'].append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'avg_r2_score': avg_r2,
                'property_r2_scores': r2_scores
            })
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Avg R²: {avg_r2:.4f}")
        
        # Load best model
        self.adme_model.load_state_dict(
            torch.load('./models/ai_screening/adme_best.pth',
                      map_location=self.device)
        )
        
        # Final evaluation
        final_metrics = self.training_history['adme'][-1]
        
        return {
            'final_val_loss': final_metrics['val_loss'],
            'final_avg_r2': final_metrics['avg_r2_score'],
            'property_r2_scores': final_metrics['property_r2_scores'],
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.training_history['adme']),
            'training_history': self.training_history['adme']
        }
    
    def _train_adme_simulated(self) -> Dict:
        """Simulated ADME training (for when model not available)"""
        print("ADME model not available, using simulated training results")
        
        return {
            'simulated': True,
            'message': 'ADME model requires specialized training data',
            'recommendation': 'Collect ADME experimental data or use pre-trained models'
        }
    
    def _save_models(self, save_path: str):
        """Save trained models"""
        import os
        from pathlib import Path
        
        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save neural network models
        if self.efficacy_model and hasattr(self.efficacy_model, 'state_dict'):
            torch.save(self.efficacy_model.state_dict(),
                      os.path.join(save_path, 'efficacy_model.pth'))
        
        if self.toxicity_model and hasattr(self.toxicity_model, 'state_dict'):
            torch.save(self.toxicity_model.state_dict(),
                      os.path.join(save_path, 'toxicity_model.pth'))
        
        if self.adme_model and hasattr(self.adme_model, 'state_dict'):
            torch.save(self.adme_model.state_dict(),
                      os.path.join(save_path, 'adme_model.pth'))
        
        # Save XGBoost models
        if hasattr(self, 'efficacy_xgb'):
            import joblib
            joblib.dump(self.efficacy_xgb, 
                       os.path.join(save_path, 'efficacy_xgb.pkl'))
        
        if hasattr(self, 'toxicity_xgb'):
            import joblib
            joblib.dump(self.toxicity_xgb,
                       os.path.join(save_path, 'toxicity_xgb.pkl'))
        
        # Save scaler
        if hasattr(self.scaler, 'transform'):
            joblib.dump(self.scaler,
                       os.path.join(save_path, 'scaler.pkl'))
        
        # Save configuration
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'config': self.config,
                'num_features': self.num_features,
                'num_toxicity_classes': self.num_toxicity_classes,
                'num_adme_properties': self.num_adme_properties
            }, f, indent=2)
        
        print(f"Models saved to {save_path}")
    
    def get_status(self) -> Dict:
        """Get current status of AI pipeline"""
        status = {
            'initialized': self.initialized,
            'model_type': self.model_type,
            'device': str(self.device),
            'models_available': {
                'efficacy': self.efficacy_model is not None or hasattr(self, 'efficacy_xgb'),
                'toxicity': self.toxicity_model is not None or hasattr(self, 'toxicity_xgb'),
                'adme': self.adme_model is not None
            },
            'feature_encoder': self.feature_encoder is not None,
            'training_history': {
                'efficacy_samples': len(self.training_history['efficacy']),
                'toxicity_samples': len(self.training_history['toxicity']),
                'adme_samples': len(self.training_history['adme'])
            }
        }
        
        return status
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up AI Screening Pipeline...")
        
        # Clear GPU cache if using GPU
        if self.use_gpu:
            torch.cuda.empty_cache()
        
        # Clear model references
        self.efficacy_model = None
        self.toxicity_model = None
        self.adme_model = None
        
        if hasattr(self, 'efficacy_xgb'):
            del self.efficacy_xgb
        if hasattr(self, 'toxicity_xgb'):
            del self.toxicity_xgb

# Neural Network Model Definitions

class EfficacyPredictionModel(nn.Module):
    """Neural network for efficacy prediction"""
    
    def __init__(self, input_dim=2048, hidden_dims=[1024, 512, 256, 128], dropout_rate=0.3):
        super(EfficacyPredictionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ToxicityPredictionModel(nn.Module):
    """Neural network for toxicity prediction"""
    
    def __init__(self, input_dim=2048, hidden_dims=[1024, 512, 256, 128], 
                 num_classes=10, dropout_rate=0.3):
        super(ToxicityPredictionModel, self).__init__()
        
        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return self.classifier(encoded)

class ADMEPredictionModel(nn.Module):
    """Neural network for ADME prediction"""
    
    def __init__(self, input_dim=2048, hidden_dims=[1024, 512, 256, 128],
                 num_properties=5, dropout_rate=0.3):
        super(ADMEPredictionModel, self).__init__()
        
        # Shared encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Multi-task regression heads
        self.property_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 1)
            )
            for _ in range(num_properties)
        ])
    
    def forward(self, x):
        encoded = self.encoder(x)
        
        # Get predictions from each head
        predictions = []
        for head in self.property_heads:
            pred = head(encoded)
            predictions.append(pred)
        
        # Concatenate predictions
        return torch.cat(predictions, dim=1)

# Example usage and helper functions

def generate_sample_compounds(num_compounds: int = 10) -> List[Dict]:
    """Generate sample compounds for testing"""
    import random
    
    compounds = []
    
    for i in range(num_compounds):
        compound = {
            'compound_id': f'TEST_{i:03d}',
            'name': f'Test Compound {i}',
            'smiles': f'CC(C)CC{"C" * random.randint(5, 10)}O{"N" * random.randint(1, 3)}',
            'molecular_weight': random.uniform(200.0, 600.0),
            'logp': random.uniform(-1.0, 5.0),
            'h_bond_donors': random.randint(0, 5),
            'h_bond_acceptors': random.randint(2, 10),
            'rotatable_bonds': random.randint(3, 15),
            'polar_surface_area': random.uniform(50.0, 150.0)
        }
        compounds.append(compound)
    
    return compounds

def run_demo():
    """Run a demonstration of the AI screening pipeline"""
    print("Running AI Screening Pipeline Demo...")
    
    # Generate sample compounds
    compounds = generate_sample_compounds(5)
    
    # Initialize pipeline
    pipeline = AIScreeningPipeline(model_type='ensemble')
    pipeline.initialize()
    
    # Screen each compound
    for compound in compounds:
        print(f"\n{'='*60}")
        print(f"Screening: {compound['name']}")
        print(f"{'='*60}")
        
        try:
            results = pipeline.screen_compound(compound)
            
            # Print summary
            print(f"Efficacy Score: {results['efficacy']['score']:.3f} "
                  f"({results['efficacy']['interpretation']})")
            print(f"Toxicity Score: {results['toxicity']['overall_score']:.3f} "
                  f"(Safety Margin: {results['toxicity']['safety_margin']:.3f})")
            print(f"ADME Score: {results['adme']['overall_score']:.3f}")
            print(f"Drug Likeness: {results['drug_likeness']['overall_score']:.3f} "
                  f"({results['drug_likeness']['classification']})")
            print(f"Recommendation: {results['recommendation']['decision']}")
            
            # Print top toxicities
            print("\nTop Toxicity Risks:")
            for toxicity in results['toxicity']['top_toxicities']:
                print(f"  - {toxicity['endpoint']}: {toxicity['probability']:.3f} "
                      f"({toxicity['risk_level']})")
            
        except Exception as e:
            print(f"Error screening compound: {e}")
    
    pipeline.cleanup()
    print("\nDemo completed!")

if __name__ == "__main__":
    run_demo()