"""
Risk Modeling Module
Predict patient responses to treatments and surgical outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using simplified models")

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class ComplicationType(Enum):
    """Types of surgical/medical complications"""
    INFECTION = "infection"
    BLEEDING = "bleeding"
    THROMBOSIS = "thrombosis"
    ORGAN_FAILURE = "organ_failure"
    NEUROLOGICAL = "neurological"
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    RENAL = "renal"
    HEPATIC = "hepatic"
    WOUND_DEHISCENCE = "wound_dehiscence"
    ANASTOMOTIC_LEAK = "anastomotic_leak"
    SEPSIS = "sepsis"

@dataclass
class PatientRiskProfile:
    """Comprehensive patient risk profile"""
    patient_id: str
    age: int
    gender: str
    bmi: float
    comorbidities: List[str]
    medications: List[str]
    allergies: List[str]
    surgical_history: List[Dict]
    lab_values: Dict[str, float]
    vital_signs: Dict[str, float]
    genetic_factors: Dict[str, str]
    lifestyle_factors: Dict[str, Any]
    
    # Calculated scores
    asa_score: int = 1  # American Society of Anesthesiologists score
    charlson_index: float = 0.0  # Charlson Comorbidity Index
    surgical_risk_score: float = 0.0
    mortality_risk: float = 0.0
    complication_risk: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_high_risk(self) -> bool:
        """Check if patient is high risk"""
        return self.asa_score >= 3 or self.surgical_risk_score >= 0.7
    
    @property
    def needs_special_monitoring(self) -> bool:
        """Check if patient needs special monitoring"""
        return (self.age >= 70 or 
                self.bmi >= 40 or 
                len(self.comorbidities) >= 3 or
                self.mortality_risk >= 0.1)

class RiskPredictionModel:
    """Base class for risk prediction models"""
    
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.feature_importance = {}
        self.calibration_curve = {}
        
    def predict(self, patient_data: PatientRiskProfile, 
                procedure_data: Dict) -> Dict:
        """Predict risks for patient and procedure"""
        raise NotImplementedError
    
    def calibrate(self, calibration_data: List[Tuple]):
        """Calibrate model predictions"""
        pass
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores"""
        return self.feature_importance

class StatisticalRiskModel(RiskPredictionModel):
    """Statistical risk prediction models"""
    
    def __init__(self):
        super().__init__('statistical')
        self.risk_scores = self._initialize_risk_scores()
    
    def _initialize_risk_scores(self):
        """Initialize statistical risk scores"""
        return {
            'surgical_risk': {
                'age': lambda age: min(1.0, age / 100.0),
                'bmi': lambda bmi: 0 if bmi < 30 else (bmi - 30) / 30.0,
                'asa_score': lambda score: (score - 1) / 4.0,
                'comorbidity_count': lambda count: min(1.0, count / 10.0)
            },
            'infection_risk': {
                'diabetes': 0.3,
                'immunosuppression': 0.4,
                'smoking': 0.2,
                'wound_class': {'clean': 0.01, 'clean_contaminated': 0.05, 
                               'contaminated': 0.15, 'dirty': 0.30}
            },
            'bleeding_risk': {
                'anticoagulants': 0.4,
                'liver_disease': 0.3,
                'platelet_count': lambda count: 0.5 if count < 100 else 0.1,
                'procedure_type': {'vascular': 0.4, 'orthopedic': 0.3, 'general': 0.1}
            }
        }
    
    def predict(self, patient_data: PatientRiskProfile, 
                procedure_data: Dict) -> Dict:
        """Predict risks using statistical models"""
        
        predictions = {
            'overall_surgical_risk': self._calculate_surgical_risk(patient_data, procedure_data),
            'mortality_risk': self._calculate_mortality_risk(patient_data, procedure_data),
            'complication_risks': {},
            'length_of_stay_prediction': self._predict_length_of_stay(patient_data, procedure_data),
            'readmission_risk': self._calculate_readmission_risk(patient_data, procedure_data)
        }
        
        # Calculate specific complication risks
        for complication in ComplicationType:
            risk = self._calculate_complication_risk(patient_data, procedure_data, complication)
            predictions['complication_risks'][complication.value] = risk
        
        # Calculate risk level
        predictions['risk_level'] = self._determine_risk_level(predictions['overall_surgical_risk'])
        
        # Generate recommendations
        predictions['recommendations'] = self._generate_recommendations(patient_data, predictions)
        
        return predictions
    
    def _calculate_surgical_risk(self, patient: PatientRiskProfile, 
                               procedure: Dict) -> float:
        """Calculate overall surgical risk"""
        risk_factors = []
        
        # Age factor
        age_factor = min(1.0, patient.age / 100.0)
        risk_factors.append(age_factor * 0.2)
        
        # BMI factor
        if patient.bmi < 18.5:
            bmi_factor = 0.2  # Underweight
        elif patient.bmi < 25:
            bmi_factor = 0.0  # Normal
        elif patient.bmi < 30:
            bmi_factor = 0.1  # Overweight
        elif patient.bmi < 40:
            bmi_factor = 0.3  # Obese
        else:
            bmi_factor = 0.5  # Morbidly obese
        risk_factors.append(bmi_factor * 0.15)
        
        # ASA score
        asa_factor = (patient.asa_score - 1) / 4.0
        risk_factors.append(asa_factor * 0.25)
        
        # Comorbidity count
        comorbidity_factor = min(1.0, len(patient.comorbidities) / 10.0)
        risk_factors.append(comorbidity_factor * 0.2)
        
        # Procedure complexity
        complexity = procedure.get('complexity', 'medium')
        complexity_factors = {'low': 0.1, 'medium': 0.3, 'high': 0.5, 'very_high': 0.8}
        procedure_factor = complexity_factors.get(complexity, 0.3)
        risk_factors.append(procedure_factor * 0.2)
        
        # Calculate weighted sum
        overall_risk = sum(risk_factors)
        
        # Adjust for emergency status
        if procedure.get('emergency', False):
            overall_risk *= 1.5
        
        return min(1.0, overall_risk)
    
    def _calculate_mortality_risk(self, patient: PatientRiskProfile,
                                procedure: Dict) -> float:
        """Calculate mortality risk"""
        # Base mortality risk
        base_risk = 0.01  # 1% base risk
        
        # Age adjustment
        age_adjustment = 0.005 * max(0, patient.age - 40)
        
        # Comorbidity adjustment
        comorbidity_adjustment = 0.02 * len(patient.comorbidities)
        
        # ASA score adjustment
        asa_adjustment = 0.03 * (patient.asa_score - 1)
        
        # Procedure risk adjustment
        procedure_risk = procedure.get('mortality_risk', 0.02)
        
        # Calculate total risk
        total_risk = (base_risk + age_adjustment + comorbidity_adjustment + 
                     asa_adjustment + procedure_risk)
        
        # Emergency procedure multiplier
        if procedure.get('emergency', False):
            total_risk *= 2.0
        
        return min(0.5, total_risk)  # Cap at 50%
    
    def _calculate_complication_risk(self, patient: PatientRiskProfile,
                                   procedure: Dict, 
                                   complication: ComplicationType) -> float:
        """Calculate risk for specific complication"""
        
        complication_models = {
            ComplicationType.INFECTION: self._calculate_infection_risk,
            ComplicationType.BLEEDING: self._calculate_bleeding_risk,
            ComplicationType.THROMBOSIS: self._calculate_thrombosis_risk,
            ComplicationType.ORGAN_FAILURE: self._calculate_organ_failure_risk,
            ComplicationType.CARDIAC: self._calculate_cardiac_risk,
            ComplicationType.RESPIRATORY: self._calculate_respiratory_risk
        }
        
        if complication in complication_models:
            return complication_models[complication](patient, procedure)
        
        # Default risk calculation
        base_risk = 0.05
        risk_multiplier = 1.0
        
        # Adjust based on patient factors
        if complication == ComplicationType.NEUROLOGICAL:
            if 'stroke' in patient.comorbidities or 'neurological_disorder' in patient.comorbidities:
                risk_multiplier *= 2.0
        
        elif complication == ComplicationType.RENAL:
            if 'kidney_disease' in patient.comorbidities:
                risk_multiplier *= 3.0
            creatinine = patient.lab_values.get('creatinine', 1.0)
            if creatinine > 1.5:
                risk_multiplier *= 1.5
        
        elif complication == ComplicationType.HEPATIC:
            if 'liver_disease' in patient.comorbidities:
                risk_multiplier *= 2.5
        
        return min(1.0, base_risk * risk_multiplier)
    
    def _calculate_infection_risk(self, patient: PatientRiskProfile,
                                procedure: Dict) -> float:
        """Calculate surgical site infection risk"""
        base_risk = 0.03  # 3% baseline
        
        risk_factors = 0.0
        
        # Diabetes
        if 'diabetes' in patient.comorbidities:
            hba1c = patient.lab_values.get('hba1c', 5.0)
            if hba1c > 7.0:
                risk_factors += 0.15
            else:
                risk_factors += 0.08
        
        # Immunosuppression
        if 'immunosuppression' in patient.comorbidities:
            risk_factors += 0.12
        
        # Smoking
        if patient.lifestyle_factors.get('smoking', False):
            risk_factors += 0.10
        
        # BMI
        if patient.bmi >= 30:
            risk_factors += 0.08
        
        # Age
        if patient.age >= 70:
            risk_factors += 0.05
        
        # Procedure duration
        duration = procedure.get('estimated_duration', 60)  # minutes
        if duration > 120:
            risk_factors += 0.07
        
        # Wound class
        wound_class = procedure.get('wound_classification', 'clean')
        wound_class_risk = {'clean': 0.01, 'clean_contaminated': 0.03,
                           'contaminated': 0.10, 'dirty': 0.20}
        base_risk = wound_class_risk.get(wound_class, 0.03)
        
        total_risk = base_risk + risk_factors
        return min(0.5, total_risk)
    
    def _calculate_bleeding_risk(self, patient: PatientRiskProfile,
                               procedure: Dict) -> float:
        """Calculate bleeding risk"""
        base_risk = 0.02
        
        risk_factors = 0.0
        
        # Anticoagulant use
        anticoagulants = ['warfarin', 'heparin', 'enoxaparin', 'apixaban', 'rivaroxaban']
        for med in patient.medications:
            if any(anticoag in med.lower() for anticoag in anticoagulants):
                risk_factors += 0.25
                break
        
        # Liver disease
        if 'liver_disease' in patient.comorbidities:
            risk_factors += 0.15
        
        # Platelet count
        platelet_count = patient.lab_values.get('platelets', 250)
        if platelet_count < 100:
            risk_factors += 0.20
        elif platelet_count < 150:
            risk_factors += 0.10
        
        # INR (coagulation)
        inr = patient.lab_values.get('inr', 1.0)
        if inr > 1.5:
            risk_factors += 0.15 * (inr - 1.0)
        
        # Procedure type
        procedure_type = procedure.get('type', 'general')
        if procedure_type in ['vascular', 'cardiothoracic']:
            risk_factors += 0.10
        
        total_risk = base_risk + risk_factors
        return min(0.4, total_risk)
    
    def _calculate_thrombosis_risk(self, patient: PatientRiskProfile,
                                 procedure: Dict) -> float:
        """Calculate deep vein thrombosis/pulmonary embolism risk"""
        # Calculate Padua Prediction Score
        padua_score = 0
        
        # Active cancer
        if 'cancer' in patient.comorbidities:
            padua_score += 3
        
        # Previous VTE
        if any('thrombosis' in h or 'embolism' in h for h in patient.surgical_history):
            padua_score += 3
        
        # Reduced mobility
        if patient.lifestyle_factors.get('reduced_mobility', False):
            padua_score += 3
        
        # Thrombophilia
        if 'thrombophilia' in patient.comorbidities:
            padua_score += 3
        
        # Recent trauma/surgery
        if len(patient.surgical_history) > 0:
            padua_score += 2
        
        # Age >= 70
        if patient.age >= 70:
            padua_score += 1
        
        # Heart/respiratory failure
        if 'heart_failure' in patient.comorbidities or 'copd' in patient.comorbidities:
            padua_score += 1
        
        # Acute MI/stroke
        if 'myocardial_infarction' in patient.comorbidities or 'stroke' in patient.comorbidities:
            padua_score += 1
        
        # Infection/rheumatologic
        if 'infection' in patient.comorbidities or 'rheumatologic' in patient.comorbidities:
            padua_score += 1
        
        # Obesity
        if patient.bmi >= 30:
            padua_score += 1
        
        # Hormonal treatment
        if any('estrogen' in med.lower() or 'contraceptive' in med.lower() 
              for med in patient.medications):
            padua_score += 1
        
        # Convert Padua score to risk probability
        if padua_score >= 4:
            risk = 0.11  # High risk
        else:
            risk = 0.02  # Low risk
        
        # Adjust for procedure duration
        duration = procedure.get('estimated_duration', 60)
        if duration > 120:
            risk *= 1.5
        
        return min(0.3, risk)
    
    def _calculate_organ_failure_risk(self, patient: PatientRiskProfile,
                                    procedure: Dict) -> float:
        """Calculate risk of organ failure"""
        risk = 0.01  # Base risk
        
        # Age factor
        if patient.age >= 75:
            risk += 0.05
        elif patient.age >= 65:
            risk += 0.03
        
        # Comorbidity factors
        organ_comorbidities = ['heart_failure', 'kidney_disease', 'liver_disease',
                              'copd', 'diabetes']
        
        for comorbidity in organ_comorbidities:
            if comorbidity in patient.comorbidities:
                risk += 0.04
        
        # ASA score
        risk += 0.02 * (patient.asa_score - 1)
        
        # Emergency procedure
        if procedure.get('emergency', False):
            risk *= 2.0
        
        # Procedure complexity
        complexity = procedure.get('complexity', 'medium')
        if complexity == 'high':
            risk *= 1.5
        elif complexity == 'very_high':
            risk *= 2.0
        
        return min(0.4, risk)
    
    def _calculate_cardiac_risk(self, patient: PatientRiskProfile,
                              procedure: Dict) -> float:
        """Calculate cardiac complication risk"""
        # Revised Cardiac Risk Index (RCRI)
        rcri_score = 0
        
        # High-risk surgery
        high_risk_procedures = ['vascular', 'intraperitoneal', 'intrathoracic']
        procedure_type = procedure.get('type', '')
        if any(high_risk in procedure_type for high_risk in high_risk_procedures):
            rcri_score += 1
        
        # Ischemic heart disease
        if any(cond in patient.comorbidities for cond in 
              ['coronary_artery_disease', 'myocardial_infarction', 'angina']):
            rcri_score += 1
        
        # Heart failure
        if 'heart_failure' in patient.comorbidities:
            rcri_score += 1
        
        # Cerebrovascular disease
        if any(cond in patient.comorbidities for cond in 
              ['stroke', 'tia', 'carotid_disease']):
            rcri_score += 1
        
        # Diabetes requiring insulin
        if 'diabetes' in patient.comorbidities:
            # Check if insulin is in medications
            if any('insulin' in med.lower() for med in patient.medications):
                rcri_score += 1
        
        # Renal insufficiency
        creatinine = patient.lab_values.get('creatinine', 1.0)
        if creatinine > 2.0:
            rcri_score += 1
        
        # Convert RCRI score to risk
        rcri_risk_map = {0: 0.004, 1: 0.01, 2: 0.07, 3: 0.11, 4: 0.26}
        risk = rcri_risk_map.get(rcri_score, 0.3)
        
        # Age adjustment
        if patient.age >= 70:
            risk *= 1.5
        
        return min(0.4, risk)
    
    def _calculate_respiratory_risk(self, patient: PatientRiskProfile,
                                  procedure: Dict) -> float:
        """Calculate respiratory complication risk"""
        risk = 0.02  # Base risk
        
        # COPD/asthma
        if 'copd' in patient.comorbidities or 'asthma' in patient.comorbidities:
            risk += 0.08
        
        # Smoking
        if patient.lifestyle_factors.get('smoking', False):
            risk += 0.05
        
        # Age
        if patient.age >= 70:
            risk += 0.04
        
        # Obesity
        if patient.bmi >= 35:
            risk += 0.06
        
        # Procedure factors
        procedure_type = procedure.get('type', '')
        if 'thoracic' in procedure_type or 'abdominal' in procedure_type:
            risk += 0.04
        
        # Emergency procedure
        if procedure.get('emergency', False):
            risk *= 1.5
        
        return min(0.3, risk)
    
    def _predict_length_of_stay(self, patient: PatientRiskProfile,
                              procedure: Dict) -> float:
        """Predict hospital length of stay in days"""
        base_los = procedure.get('typical_los', 3.0)  # days
        
        # Adjust for patient factors
        adjustments = 0.0
        
        # Age adjustment
        if patient.age >= 80:
            adjustments += 2.0
        elif patient.age >= 65:
            adjustments += 1.0
        
        # Comorbidity adjustment
        adjustments += len(patient.comorbidities) * 0.5
        
        # ASA score adjustment
        adjustments += (patient.asa_score - 1) * 0.8
        
        # BMI adjustment
        if patient.bmi >= 40:
            adjustments += 1.5
        elif patient.bmi >= 30:
            adjustments += 0.5
        
        # Emergency adjustment
        if procedure.get('emergency', False):
            adjustments += 1.0
        
        predicted_los = base_los + adjustments
        
        # Add random variation
        predicted_los += np.random.normal(0, 0.5)
        
        return max(1.0, predicted_los)
    
    def _calculate_readmission_risk(self, patient: PatientRiskProfile,
                                  procedure: Dict) -> float:
        """Calculate 30-day readmission risk"""
        base_risk = 0.08  # 8% baseline
        
        risk_factors = 0.0
        
        # Age
        if patient.age >= 75:
            risk_factors += 0.04
        
        # Comorbidity count
        comorbidity_count = len(patient.comorbidities)
        risk_factors += comorbidity_count * 0.02
        
        # Previous admissions
        previous_admissions = sum(1 for h in patient.surgical_history 
                                if h.get('admitted', False))
        risk_factors += previous_admissions * 0.03
        
        # Social factors
        if patient.lifestyle_factors.get('lives_alone', False):
            risk_factors += 0.02
        if patient.lifestyle_factors.get('low_socioeconomic', False):
            risk_factors += 0.03
        
        # Procedure complexity
        complexity = procedure.get('complexity', 'medium')
        if complexity == 'high':
            risk_factors += 0.04
        elif complexity == 'very_high':
            risk_factors += 0.08
        
        total_risk = base_risk + risk_factors
        return min(0.3, total_risk)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        if risk_score < 0.1:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.2:
            return RiskLevel.LOW
        elif risk_score < 0.4:
            return RiskLevel.MODERATE
        elif risk_score < 0.6:
            return RiskLevel.HIGH
        elif risk_score < 0.8:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _generate_recommendations(self, patient: PatientRiskProfile,
                                predictions: Dict) -> List[Dict]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        # Overall risk recommendations
        risk_level = predictions['risk_level']
        
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
            recommendations.append({
                'category': 'general',
                'recommendation': 'Consider preoperative optimization clinic',
                'priority': 'high',
                'rationale': f'High overall surgical risk ({risk_level.value})'
            })
        
        # Complication-specific recommendations
        complication_risks = predictions['complication_risks']
        
        # Infection prevention
        if complication_risks.get('infection', 0) > 0.1:
            recommendations.append({
                'category': 'infection',
                'recommendation': 'Administer preoperative antibiotics',
                'priority': 'medium',
                'rationale': f'Infection risk: {complication_risks["infection"]:.1%}'
            })
        
        # Bleeding prevention
        if complication_risks.get('bleeding', 0) > 0.15:
            recommendations.append({
                'category': 'bleeding',
                'recommendation': 'Consider preoperative transfusion if indicated',
                'priority': 'medium',
                'rationale': f'Bleeding risk: {complication_risks["bleeding"]:.1%}'
            })
        
        # Thrombosis prevention
        if complication_risks.get('thrombosis', 0) > 0.05:
            recommendations.append({
                'category': 'thrombosis',
                'recommendation': 'Implement VTE prophylaxis protocol',
                'priority': 'high',
                'rationale': f'VTE risk: {complication_risks["thrombosis"]:.1%}'
            })
        
        # Cardiac risk management
        if complication_risks.get('cardiac', 0) > 0.05:
            recommendations.append({
                'category': 'cardiac',
                'recommendation': 'Cardiology consultation and optimization',
                'priority': 'high',
                'rationale': f'Cardiac risk: {complication_risks["cardiac"]:.1%}'
            })
        
        # Patient-specific recommendations
        if 'diabetes' in patient.comorbidities:
            hba1c = patient.lab_values.get('hba1c', 5.0)
            if hba1c > 8.0:
                recommendations.append({
                    'category': 'diabetes',
                    'recommendation': 'Optimize glycemic control preoperatively',
                    'priority': 'medium',
                    'rationale': f'Elevated HbA1c: {hba1c:.1f}%'
                })
        
        if patient.bmi >= 40:
            recommendations.append({
                'category': 'obesity',
                'recommendation': 'Consider bariatric surgery consultation',
                'priority': 'medium',
                'rationale': f'Morbid obesity (BMI: {patient.bmi:.1f})'
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations

class MLRiskModel(RiskPredictionModel):
    """Machine learning based risk prediction model"""
    
    def __init__(self, model_architecture: str = 'ensemble'):
        super().__init__(model_architecture)
        
        if TORCH_AVAILABLE:
            self._initialize_neural_networks()
        else:
            print("PyTorch not available, using simplified ML models")
            self._initialize_simplified_models()
    
    def _initialize_neural_networks(self):
        """Initialize neural network models"""
        # Mortality prediction model
        self.models['mortality'] = MortalityPredictionNN(
            input_size=50,
            hidden_sizes=[256, 128, 64],
            dropout_rate=0.3
        )
        
        # Complication prediction model (multi-task)
        self.models['complications'] = ComplicationPredictionNN(
            input_size=50,
            hidden_sizes=[256, 128],
            num_complications=len(ComplicationType),
            dropout_rate=0.3
        )
        
        # Length of stay prediction model
        self.models['los'] = LengthOfStayNN(
            input_size=50,
            hidden_sizes=[128, 64],
            dropout_rate=0.2
        )
    
    def _initialize_simplified_models(self):
        """Initialize simplified ML models"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Simplified models using scikit-learn
        self.models['mortality'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.models['complications'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.models['los'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def extract_features(self, patient: PatientRiskProfile, 
                        procedure: Dict) -> np.ndarray:
        """Extract features for ML models"""
        features = []
        
        # Demographic features
        features.append(patient.age / 100.0)  # Normalized age
        features.append(1.0 if patient.gender.lower() == 'male' else 0.0)
        features.append(min(1.0, patient.bmi / 50.0))  # Normalized BMI
        
        # Comorbidity features (one-hot encoded)
        comorbidity_list = [
            'hypertension', 'diabetes', 'coronary_artery_disease',
            'heart_failure', 'copd', 'asthma', 'kidney_disease',
            'liver_disease', 'cancer', 'stroke', 'obesity'
        ]
        
        for comorbidity in comorbidity_list:
            features.append(1.0 if comorbidity in patient.comorbidities else 0.0)
        
        # Lab values (normalized)
        lab_tests = ['creatinine', 'hba1c', 'platelets', 'inr', 'albumin']
        for test in lab_tests:
            value = patient.lab_values.get(test, 0.0)
            # Normalize based on typical ranges
            if test == 'creatinine':
                features.append(min(1.0, value / 5.0))
            elif test == 'hba1c':
                features.append(min(1.0, value / 15.0))
            elif test == 'platelets':
                features.append(max(0.0, min(1.0, (value - 50) / 450)))
            elif test == 'inr':
                features.append(min(1.0, value / 3.0))
            elif test == 'albumin':
                features.append(min(1.0, value / 5.0))
        
        # Vital signs (normalized)
        vital_signs = ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate']
        for vs in vital_signs:
            value = patient.vital_signs.get(vs, 0.0)
            if vs in ['systolic_bp', 'diastolic_bp']:
                features.append(min(1.0, value / 200.0))
            elif vs == 'heart_rate':
                features.append(min(1.0, value / 150.0))
            elif vs == 'respiratory_rate':
                features.append(min(1.0, value / 40.0))
        
        # ASA score
        features.append((patient.asa_score - 1) / 4.0)
        
        # Charlson index (normalized)
        features.append(min(1.0, patient.charlson_index / 10.0))
        
        # Lifestyle factors
        features.append(1.0 if patient.lifestyle_factors.get('smoking', False) else 0.0)
        features.append(1.0 if patient.lifestyle_factors.get('alcohol', False) else 0.0)
        
        # Procedure features
        features.append(1.0 if procedure.get('emergency', False) else 0.0)
        
        complexity_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'very_high': 1.0}
        features.append(complexity_map.get(procedure.get('complexity', 'medium'), 0.5))
        
        # Estimated duration (normalized)
        duration = procedure.get('estimated_duration', 60)
        features.append(min(1.0, duration / 240.0))  # 4 hours max
        
        # Pad with zeros if needed
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50], dtype=np.float32)
    
    def predict(self, patient: PatientRiskProfile, procedure: Dict) -> Dict:
        """Predict risks using ML models"""
        # Extract features
        features = self.extract_features(patient, procedure)
        
        predictions = {
            'overall_surgical_risk': 0.0,
            'mortality_risk': 0.0,
            'complication_risks': {},
            'length_of_stay_prediction': 0.0,
            'risk_level': RiskLevel.MODERATE
        }
        
        if TORCH_AVAILABLE and 'mortality' in self.models:
            # Use neural networks
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Mortality prediction
            mortality_model = self.models['mortality']
            mortality_model.eval()
            with torch.no_grad():
                mortality_risk = mortality_model(features_tensor).item()
            predictions['mortality_risk'] = max(0.0, min(1.0, mortality_risk))
            
            # Complication predictions
            complication_model = self.models['complications']
            complication_model.eval()
            with torch.no_grad():
                complication_probs = torch.sigmoid(complication_model(features_tensor)).squeeze().numpy()
            
            for i, complication in enumerate(ComplicationType):
                if i < len(complication_probs):
                    predictions['complication_risks'][complication.value] = float(complication_probs[i])
            
            # Length of stay prediction
            los_model = self.models['los']
            los_model.eval()
            with torch.no_grad():
                los_pred = los_model(features_tensor).item()
            predictions['length_of_stay_prediction'] = max(1.0, los_pred)
            
        else:
            # Use simplified models
            if 'mortality' in self.models:
                # For demo purposes, generate simulated predictions
                predictions['mortality_risk'] = self._simulate_ml_prediction(features, 'mortality')
                
                # Complication risks
                for complication in ComplicationType:
                    risk = self._simulate_ml_prediction(features, 'complication')
                    predictions['complication_risks'][complication.value] = risk
                
                # Length of stay
                predictions['length_of_stay_prediction'] = self._simulate_ml_prediction(features, 'los', regression=True)
        
        # Calculate overall surgical risk (weighted average)
        complication_risks = list(predictions['complication_risks'].values())
        if complication_risks:
            predictions['overall_surgical_risk'] = (
                predictions['mortality_risk'] * 0.4 +
                np.mean(complication_risks) * 0.6
            )
        
        # Determine risk level
        predictions['risk_level'] = self._determine_risk_level(predictions['overall_surgical_risk'])
        
        # Generate recommendations
        recommendations_model = StatisticalRiskModel()
        predictions['recommendations'] = recommendations_model._generate_recommendations(
            patient, predictions
        )
        
        return predictions
    
    def _simulate_ml_prediction(self, features: np.ndarray, 
                              prediction_type: str,
                              regression: bool = False) -> float:
        """Simulate ML prediction for demo purposes"""
        # Use features to generate deterministic but realistic predictions
        feature_hash = hash(tuple(features[:10].tolist())) % 1000 / 1000.0
        
        if prediction_type == 'mortality':
            # Mortality risk simulation
            base_risk = 0.05
            feature_influence = np.mean(features[:20]) * 0.3
            return min(0.5, base_risk + feature_influence + feature_hash * 0.1)
        
        elif prediction_type == 'complication':
            # Complication risk simulation
            base_risk = 0.1
            feature_influence = np.mean(features[10:30]) * 0.4
            return min(0.8, base_risk + feature_influence + feature_hash * 0.15)
        
        elif prediction_type == 'los' and regression:
            # Length of stay simulation
            base_los = 5.0
            feature_influence = np.sum(features[:15]) * 3.0
            return max(1.0, base_los + feature_influence + feature_hash * 4.0)
        
        return 0.0
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        # Same as statistical model
        if risk_score < 0.1:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.2:
            return RiskLevel.LOW
        elif risk_score < 0.4:
            return RiskLevel.MODERATE
        elif risk_score < 0.6:
            return RiskLevel.HIGH
        elif risk_score < 0.8:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL
    
    def train(self, training_data: List[Tuple], validation_data: List[Tuple] = None):
        """Train ML models on data"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available for training")
            return
        
        print("Training ML risk models...")
        
        # Prepare data
        X_train = []
        y_mortality_train = []
        y_complications_train = []
        y_los_train = []
        
        for patient, procedure, outcomes in training_data:
            features = self.extract_features(patient, procedure)
            X_train.append(features)
            y_mortality_train.append(outcomes.get('mortality', 0))
            y_complications_train.append(outcomes.get('complications', []))
            y_los_train.append(outcomes.get('length_of_stay', 0.0))
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_train)
        y_mortality_tensor = torch.FloatTensor(y_mortality_train)
        y_los_tensor = torch.FloatTensor(y_los_train)
        
        # Training loop for mortality model
        mortality_model = self.models['mortality']
        optimizer = torch.optim.Adam(mortality_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        epochs = 50
        batch_size = 32
        
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[indices]
            y_shuffled = y_mortality_tensor[indices]
            
            epoch_loss = 0.0
            
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions = mortality_model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        print("Training completed")

# Neural Network Definitions (if PyTorch available)
if TORCH_AVAILABLE:
    class MortalityPredictionNN(nn.Module):
        """Neural network for mortality prediction"""
        
        def __init__(self, input_size=50, hidden_sizes=[256, 128, 64], dropout_rate=0.3):
            super(MortalityPredictionNN, self).__init__()
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.Sigmoid())
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)
    
    class ComplicationPredictionNN(nn.Module):
        """Neural network for complication prediction (multi-task)"""
        
        def __init__(self, input_size=50, hidden_sizes=[256, 128], 
                    num_complications=10, dropout_rate=0.3):
            super(ComplicationPredictionNN, self).__init__()
            
            # Shared encoder
            encoder_layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                encoder_layers.append(nn.Linear(prev_size, hidden_size))
                encoder_layers.append(nn.BatchNorm1d(hidden_size))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            self.encoder = nn.Sequential(*encoder_layers)
            
            # Task-specific heads
            self.complication_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(prev_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(64, 1)
                )
                for _ in range(num_complications)
            ])
        
        def forward(self, x):
            encoded = self.encoder(x)
            
            # Get predictions from each head
            predictions = []
            for head in self.complication_heads:
                pred = head(encoded)
                predictions.append(pred)
            
            # Concatenate predictions
            return torch.cat(predictions, dim=1)
    
    class LengthOfStayNN(nn.Module):
        """Neural network for length of stay prediction"""
        
        def __init__(self, input_size=50, hidden_sizes=[128, 64], dropout_rate=0.2):
            super(LengthOfStayNN, self).__init__()
            
            layers = []
            prev_size = input_size
            
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            layers.append(nn.Linear(prev_size, 1))
            layers.append(nn.ReLU())  # LOS can't be negative
            
            self.model = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)

class RiskModeler:
    """Main risk modeling engine with multiple prediction methods"""
    
    def __init__(self, model_type: str = 'ensemble', calibration_data=None):
        """
        Initialize risk modeler
        
        Args:
            model_type: Type of model ('statistical', 'ml', 'ensemble')
            calibration_data: Data for model calibration
        """
        self.model_type = model_type
        self.calibration_data = calibration_data
        
        # Initialize models
        self.statistical_model = StatisticalRiskModel()
        
        if model_type in ['ml', 'ensemble']:
            self.ml_model = MLRiskModel()
        else:
            self.ml_model = None
        
        # Risk thresholds
        self.risk_thresholds = {
            'mortality': {'low': 0.01, 'medium': 0.05, 'high': 0.10},
            'infection': {'low': 0.03, 'medium': 0.08, 'high': 0.15},
            'bleeding': {'low': 0.02, 'medium': 0.06, 'high': 0.12},
            'thrombosis': {'low': 0.02, 'medium': 0.05, 'high': 0.10}
        }
        
        # Performance tracking
        self.prediction_history = []
        self.model_performance = {}
        
        print(f"Risk Modeler initialized (Model type: {model_type})")
    
    def assess_patient_risk(self, patient: PatientRiskProfile, 
                           procedure: Dict, 
                           use_ml: bool = None) -> Dict:
        """
        Assess patient risk for procedure
        
        Args:
            patient: Patient risk profile
            procedure: Procedure details
            use_ml: Whether to use ML model (None = auto based on model_type)
            
        Returns:
            Comprehensive risk assessment
        """
        if use_ml is None:
            use_ml = self.model_type in ['ml', 'ensemble']
        
        print(f"Assessing risk for patient {patient.patient_id} "
              f"undergoing {procedure.get('name', 'procedure')}")
        
        # Calculate basic risk scores
        self._calculate_basic_scores(patient)
        
        # Get predictions
        if use_ml and self.ml_model:
            predictions = self.ml_model.predict(patient, procedure)
        else:
            predictions = self.statistical_model.predict(patient, procedure)
        
        # If ensemble model, combine predictions
        if self.model_type == 'ensemble' and self.ml_model:
            statistical_predictions = self.statistical_model.predict(patient, procedure)
            predictions = self._combine_predictions(predictions, statistical_predictions)
        
        # Add patient information
        predictions['patient_info'] = {
            'patient_id': patient.patient_id,
            'age': patient.age,
            'gender': patient.gender,
            'bmi': patient.bmi,
            'asa_score': patient.asa_score,
            'charlson_index': patient.charlson_index,
            'comorbidity_count': len(patient.comorbidities)
        }
        
        # Add procedure information
        predictions['procedure_info'] = {
            'name': procedure.get('name', 'Unknown'),
            'type': procedure.get('type', 'general'),
            'complexity': procedure.get('complexity', 'medium'),
            'emergency': procedure.get('emergency', False),
            'estimated_duration': procedure.get('estimated_duration', 60)
        }
        
        # Calculate risk summary
        predictions['risk_summary'] = self._generate_risk_summary(predictions)
        
        # Generate detailed report
        predictions['detailed_report'] = self._generate_detailed_report(patient, predictions)
        
        # Record prediction
        self._record_prediction(patient, procedure, predictions)
        
        return predictions
    
    def _calculate_basic_scores(self, patient: PatientRiskProfile):
        """Calculate basic risk scores"""
        # ASA score (simplified calculation)
        if not hasattr(patient, 'asa_score') or patient.asa_score == 1:
            patient.asa_score = self._calculate_asa_score(patient)
        
        # Charlson Comorbidity Index
        if not hasattr(patient, 'charlson_index') or patient.charlson_index == 0:
            patient.charlson_index = self._calculate_charlson_index(patient)
    
    def _calculate_asa_score(self, patient: PatientRiskProfile) -> int:
        """Calculate ASA (American Society of Anesthesiologists) score"""
        # Simplified ASA scoring
        if len(patient.comorbidities) == 0 and patient.age < 50:
            return 1  # Healthy patient
        elif len(patient.comorbidities) == 0:
            return 2  # Mild systemic disease
        elif len(patient.comorbidities) <= 2:
            return 3  # Severe systemic disease
        elif len(patient.comorbidities) <= 4:
            return 4  # Severe systemic disease that is a constant threat to life
        else:
            return 5  # Moribund patient not expected to survive without operation
    
    def _calculate_charlson_index(self, patient: PatientRiskProfile) -> float:
        """Calculate Charlson Comorbidity Index"""
        charlson_weights = {
            'myocardial_infarction': 1,
            'heart_failure': 1,
            'peripheral_vascular_disease': 1,
            'cerebrovascular_disease': 1,
            'dementia': 1,
            'copd': 1,
            'connective_tissue_disease': 1,
            'peptic_ulcer': 1,
            'mild_liver_disease': 1,
            'diabetes': 1,
            'hemiplegia': 2,
            'moderate_severe_renal_disease': 2,
            'diabetes_with_end_organ_damage': 2,
            'any_tumor': 2,
            'leukemia': 2,
            'lymphoma': 2,
            'moderate_severe_liver_disease': 3,
            'metastatic_solid_tumor': 6,
            'aids': 6
        }
        
        score = 0
        for comorbidity in patient.comorbidities:
            for key, weight in charlson_weights.items():
                if key in comorbidity.lower():
                    score += weight
                    break
        
        # Age adjustment
        if patient.age >= 50:
            score += 1
        if patient.age >= 60:
            score += 1
        if patient.age >= 70:
            score += 1
        if patient.age >= 80:
            score += 2
        
        return score
    
    def _combine_predictions(self, ml_predictions: Dict, 
                           statistical_predictions: Dict) -> Dict:
        """Combine ML and statistical predictions (ensemble)"""
        combined = ml_predictions.copy()
        
        # Weighted combination
        ml_weight = 0.6
        statistical_weight = 0.4
        
        # Combine overall surgical risk
        combined['overall_surgical_risk'] = (
            ml_predictions['overall_surgical_risk'] * ml_weight +
            statistical_predictions['overall_surgical_risk'] * statistical_weight
        )
        
        # Combine mortality risk
        combined['mortality_risk'] = (
            ml_predictions['mortality_risk'] * ml_weight +
            statistical_predictions['mortality_risk'] * statistical_weight
        )
        
        # Combine complication risks
        for complication in ComplicationType:
            ml_risk = ml_predictions['complication_risks'].get(complication.value, 0.0)
            stat_risk = statistical_predictions['complication_risks'].get(complication.value, 0.0)
            
            combined_risk = ml_risk * ml_weight + stat_risk * statistical_weight
            combined['complication_risks'][complication.value] = combined_risk
        
        # Recalculate risk level
        combined['risk_level'] = self._determine_risk_level(combined['overall_surgical_risk'])
        
        return combined
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        # Use same thresholds as statistical model
        return self.statistical_model._determine_risk_level(risk_score)
    
    def _generate_risk_summary(self, predictions: Dict) -> Dict:
        """Generate risk summary"""
        risk_level = predictions['risk_level']
        mortality_risk = predictions['mortality_risk']
        
        # Get top 3 complications by risk
        complication_risks = predictions['complication_risks']
        top_complications = sorted(
            complication_risks.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        summary = {
            'overall_risk_level': risk_level.value,
            'mortality_risk_percentage': f"{mortality_risk:.1%}",
            'key_risks': [
                {
                    'complication': comp[0],
                    'risk': f"{comp[1]:.1%}",
                    'level': self._get_risk_level_for_complication(comp[0], comp[1])
                }
                for comp in top_complications
            ],
            'length_of_stay_estimate': f"{predictions['length_of_stay_prediction']:.1f} days",
            'readmission_risk': f"{predictions.get('readmission_risk', 0.0):.1%}",
            'interpretation': self._interpret_risk_level(risk_level)
        }
        
        return summary
    
    def _get_risk_level_for_complication(self, complication: str, 
                                       risk: float) -> str:
        """Get risk level for specific complication"""
        if complication in self.risk_thresholds:
            thresholds = self.risk_thresholds[complication]
            if risk < thresholds['low']:
                return 'low'
            elif risk < thresholds['medium']:
                return 'medium'
            else:
                return 'high'
        else:
            # Default thresholds
            if risk < 0.05:
                return 'low'
            elif risk < 0.15:
                return 'medium'
            else:
                return 'high'
    
    def _interpret_risk_level(self, risk_level: RiskLevel) -> str:
        """Generate interpretation of risk level"""
        interpretations = {
            RiskLevel.VERY_LOW: "Minimal risk - Standard precautions sufficient",
            RiskLevel.LOW: "Low risk - Routine monitoring recommended",
            RiskLevel.MODERATE: "Moderate risk - Enhanced monitoring and precautions advised",
            RiskLevel.HIGH: "High risk - Consider preoperative optimization and specialized care",
            RiskLevel.VERY_HIGH: "Very high risk - Requires multidisciplinary team and intensive monitoring",
            RiskLevel.CRITICAL: "Critical risk - Consider alternative treatments or palliative care"
        }
        return interpretations.get(risk_level, "Risk assessment not available")
    
    def _generate_detailed_report(self, patient: PatientRiskProfile, 
                                predictions: Dict) -> Dict:
        """Generate detailed risk report"""
        report = {
            'patient_summary': {
                'demographics': {
                    'age': patient.age,
                    'gender': patient.gender,
                    'bmi': patient.bmi,
                    'bmi_category': self._get_bmi_category(patient.bmi)
                },
                'comorbidities': patient.comorbidities,
                'medications': patient.medications[:10],  # Top 10
                'key_lab_values': self._extract_key_labs(patient.lab_values),
                'vital_signs': patient.vital_signs
            },
            'risk_analysis': {
                'overall_risk_score': predictions['overall_surgical_risk'],
                'risk_level': predictions['risk_level'].value,
                'mortality_risk': predictions['mortality_risk'],
                'complication_risk_breakdown': predictions['complication_risks'],
                'comparative_analysis': self._compare_to_population(predictions)
            },
            'predictive_metrics': {
                'length_of_stay': predictions['length_of_stay_prediction'],
                'icu_admission_probability': self._calculate_icu_probability(predictions),
                'reoperation_probability': self._calculate_reoperation_probability(predictions),
                'discharge_disposition': self._predict_discharge_disposition(predictions)
            },
            'temporal_analysis': self._analyze_temporal_risks(predictions)
        }
        
        return report
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        elif bmi < 35:
            return "Obese I"
        elif bmi < 40:
            return "Obese II"
        else:
            return "Obese III"
    
    def _extract_key_labs(self, lab_values: Dict) -> Dict:
        """Extract key laboratory values"""
        key_labs = {}
        
        important_tests = ['creatinine', 'hba1c', 'platelets', 'inr', 'albumin', 
                          'hemoglobin', 'white_blood_cells', 'sodium', 'potassium']
        
        for test in important_tests:
            if test in lab_values:
                key_labs[test] = {
                    'value': lab_values[test],
                    'unit': self._get_lab_unit(test),
                    'interpretation': self._interpret_lab_value(test, lab_values[test])
                }
        
        return key_labs
    
    def _get_lab_unit(self, test: str) -> str:
        """Get unit for lab test"""
        units = {
            'creatinine': 'mg/dL',
            'hba1c': '%',
            'platelets': 'K/μL',
            'inr': 'ratio',
            'albumin': 'g/dL',
            'hemoglobin': 'g/dL',
            'white_blood_cells': 'K/μL',
            'sodium': 'mEq/L',
            'potassium': 'mEq/L'
        }
        return units.get(test, '')
    
    def _interpret_lab_value(self, test: str, value: float) -> str:
        """Interpret lab value"""
        ranges = {
            'creatinine': {'normal': (0.6, 1.2), 'high': (1.3, 2.0), 'critical': (2.1, float('inf'))},
            'hba1c': {'normal': (0, 5.6), 'prediabetes': (5.7, 6.4), 'diabetes': (6.5, float('inf'))},
            'platelets': {'low': (0, 149), 'normal': (150, 450), 'high': (451, float('inf'))},
            'inr': {'normal': (0.8, 1.2), 'elevated': (1.3, 2.0), 'high': (2.1, float('inf'))},
            'albumin': {'low': (0, 3.4), 'normal': (3.5, 5.0), 'high': (5.1, float('inf'))}
        }
        
        if test in ranges:
            test_ranges = ranges[test]
            for category, (low, high) in test_ranges.items():
                if low <= value <= high:
                    return category
        
        return "unknown"
    
    def _compare_to_population(self, predictions: Dict) -> Dict:
        """Compare patient risks to population averages"""
        population_averages = {
            'mortality': 0.02,  # 2%
            'infection': 0.03,  # 3%
            'bleeding': 0.02,   # 2%
            'thrombosis': 0.02, # 2%
            'length_of_stay': 4.5  # days
        }
        
        comparison = {}
        for metric, avg in population_averages.items():
            if metric in predictions:
                patient_value = predictions[metric]
                ratio = patient_value / avg if avg > 0 else 1.0
                comparison[metric] = {
                    'patient_value': patient_value,
                    'population_average': avg,
                    'ratio': ratio,
                    'interpretation': 'higher' if ratio > 1.2 else 
                                     'lower' if ratio < 0.8 else 'similar'
                }
        
        return comparison
    
    def _calculate_icu_probability(self, predictions: Dict) -> float:
        """Calculate probability of ICU admission"""
        base_prob = 0.05  # 5% baseline
        
        # Increase based on risk factors
        risk_factors = 0.0
        
        if predictions['mortality_risk'] > 0.1:
            risk_factors += 0.15
        
        complication_risks = predictions['complication_risks']
        high_risk_complications = ['cardiac', 'respiratory', 'organ_failure', 'sepsis']
        
        for complication in high_risk_complications:
            if complication in complication_risks:
                risk = complication_risks[complication]
                if risk > 0.1:
                    risk_factors += risk * 0.3
        
        total_prob = min(0.8, base_prob + risk_factors)
        return total_prob
    
    def _calculate_reoperation_probability(self, predictions: Dict) -> float:
        """Calculate probability of reoperation"""
        base_prob = 0.03  # 3% baseline
        
        # Increase based on bleeding and infection risks
        complication_risks = predictions['complication_risks']
        
        risk_factors = 0.0
        if 'bleeding' in complication_risks:
            risk_factors += complication_risks['bleeding'] * 0.4
        
        if 'infection' in complication_risks:
            risk_factors += complication_risks['infection'] * 0.3
        
        if 'anastomotic_leak' in complication_risks:
            risk_factors += complication_risks['anastomotic_leak'] * 0.5
        
        total_prob = min(0.5, base_prob + risk_factors)
        return total_prob
    
    def _predict_discharge_disposition(self, predictions: Dict) -> Dict:
        """Predict discharge disposition"""
        # Simplified model
        mortality_risk = predictions['mortality_risk']
        los = predictions['length_of_stay_prediction']
        
        if mortality_risk > 0.3:
            return {'primary': 'hospice', 'probability': 0.4, 'alternatives': ['death']}
        elif los > 14:
            return {'primary': 'rehabilitation', 'probability': 0.6, 'alternatives': ['skilled_nursing']}
        elif los > 7:
            return {'primary': 'skilled_nursing', 'probability': 0.5, 'alternatives': ['home_with_services']}
        else:
            return {'primary': 'home', 'probability': 0.8, 'alternatives': ['home_with_services']}
    
    def _analyze_temporal_risks(self, predictions: Dict) -> Dict:
        """Analyze risks over time"""
        temporal_risks = {
            'immediate_intraoperative': {
                'timeframe': '0-6 hours',
                'primary_risks': ['bleeding', 'anesthesia_complications'],
                'estimated_risk': predictions['complication_risks'].get('bleeding', 0.0) * 0.8
            },
            'early_postoperative': {
                'timeframe': '6-48 hours',
                'primary_risks': ['infection', 'respiratory', 'cardiac'],
                'estimated_risk': predictions['complication_risks'].get('infection', 0.0) * 0.6
            },
            'intermediate_postoperative': {
                'timeframe': '2-7 days',
                'primary_risks': ['thrombosis', 'organ_failure', 'sepsis'],
                'estimated_risk': predictions['complication_risks'].get('thrombosis', 0.0) * 0.7
            },
            'late_postoperative': {
                'timeframe': '7-30 days',
                'primary_risks': ['wound_dehiscence', 'readmission', 'mortality'],
                'estimated_risk': predictions.get('readmission_risk', 0.0)
            }
        }
        
        return temporal_risks
    
    def _record_prediction(self, patient: PatientRiskProfile, 
                          procedure: Dict, predictions: Dict):
        """Record prediction for performance tracking"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient.patient_id,
            'procedure': procedure.get('name', 'Unknown'),
            'predictions': predictions.copy(),
            'model_type': self.model_type
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only last 1000 predictions
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def validate_with_outcomes(self, patient_id: str, actual_outcomes: Dict):
        """Validate predictions with actual outcomes"""
        # Find prediction for this patient
        for i, record in enumerate(reversed(self.prediction_history)):
            if record['patient_id'] == patient_id:
                predictions = record['predictions']
                
                # Calculate accuracy metrics
                validation = {
                    'patient_id': patient_id,
                    'prediction_date': record['timestamp'],
                    'actual_outcomes': actual_outcomes,
                    'predicted_outcomes': {
                        'mortality_risk': predictions['mortality_risk'],
                        'complication_risks': predictions['complication_risks'],
                        'length_of_stay': predictions['length_of_stay_prediction']
                    },
                    'accuracy_metrics': self._calculate_accuracy_metrics(predictions, actual_outcomes)
                }
                
                # Update model performance
                self._update_model_performance(validation)
                
                return validation
        
        return {'error': 'No prediction found for patient'}
    
    def _calculate_accuracy_metrics(self, predictions: Dict, 
                                  actual_outcomes: Dict) -> Dict:
        """Calculate prediction accuracy metrics"""
        metrics = {
            'mortality_prediction': {
                'predicted': predictions['mortality_risk'],
                'actual': 1.0 if actual_outcomes.get('died', False) else 0.0,
                'error': abs(predictions['mortality_risk'] - 
                           (1.0 if actual_outcomes.get('died', False) else 0.0))
            },
            'complication_prediction': {},
            'los_prediction': {
                'predicted': predictions['length_of_stay_prediction'],
                'actual': actual_outcomes.get('length_of_stay', 0.0),
                'error': abs(predictions['length_of_stay_prediction'] - 
                           actual_outcomes.get('length_of_stay', 0.0))
            }
        }
        
        # Calculate complication prediction accuracy
        actual_complications = actual_outcomes.get('complications', {})
        for complication, predicted_risk in predictions['complication_risks'].items():
            actual_occurred = 1.0 if complication in actual_complications else 0.0
            metrics['complication_prediction'][complication] = {
                'predicted': predicted_risk,
                'actual': actual_occurred,
                'error': abs(predicted_risk - actual_occurred)
            }
        
        # Calculate Brier score for mortality prediction
        mortality_error = metrics['mortality_prediction']['error']
        metrics['brier_score'] = mortality_error ** 2
        
        # Calculate calibration
        metrics['calibration'] = self._assess_calibration(predictions, actual_outcomes)
        
        return metrics
    
    def _assess_calibration(self, predictions: Dict, 
                          actual_outcomes: Dict) -> Dict:
        """Assess prediction calibration"""
        # Simplified calibration assessment
        predicted_mortality = predictions['mortality_risk']
        actual_mortality = 1.0 if actual_outcomes.get('died', False) else 0.0
        
        calibration_error = abs(predicted_mortality - actual_mortality)
        
        return {
            'predicted_probability': predicted_mortality,
            'actual_outcome': actual_mortality,
            'calibration_error': calibration_error,
            'well_calibrated': calibration_error < 0.1
        }
    
    def _update_model_performance(self, validation: Dict):
        """Update model performance statistics"""
        metrics = validation['accuracy_metrics']
        
        if 'performance_history' not in self.model_performance:
            self.model_performance['performance_history'] = []
        
        self.model_performance['performance_history'].append(validation)
        
        # Calculate aggregate statistics
        if len(self.model_performance['performance_history']) > 0:
            history = self.model_performance['performance_history']
            
            # Calculate average errors
            mortality_errors = [v['accuracy_metrics']['mortality_prediction']['error'] 
                              for v in history]
            los_errors = [v['accuracy_metrics']['los_prediction']['error'] 
                         for v in history]
            
            self.model_performance['aggregate_stats'] = {
                'total_validations': len(history),
                'mean_mortality_error': np.mean(mortality_errors) if mortality_errors else 0.0,
                'mean_los_error': np.mean(los_errors) if los_errors else 0.0,
                'brier_score': np.mean([v['accuracy_metrics']['brier_score'] 
                                      for v in history]) if history else 0.0,
                'calibration_rate': np.mean([1.0 if v['accuracy_metrics']['calibration']['well_calibrated'] 
                                           else 0.0 for v in history]) if history else 0.0
            }
    
    def get_model_performance(self) -> Dict:
        """Get model performance statistics"""
        if 'aggregate_stats' not in self.model_performance:
            return {'error': 'No performance data available'}
        
        return self.model_performance
    
    def generate_risk_report(self, patient: PatientRiskProfile, 
                           procedure: Dict, format: str = 'json') -> Union[str, Dict]:
        """Generate comprehensive risk report"""
        # Get risk assessment
        risk_assessment = self.assess_patient_risk(patient, procedure)
        
        # Create report
        report = {
            'report_id': f"RR_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}",
            'generated_at': datetime.now().isoformat(),
            'model_type': self.model_type,
            'patient_summary': risk_assessment['patient_info'],
            'procedure_summary': risk_assessment['procedure_info'],
            'executive_summary': risk_assessment['risk_summary'],
            'detailed_risk_analysis': risk_assessment['detailed_report'],
            'recommendations': risk_assessment['recommendations'],
            'risk_mitigation_strategies': self._generate_mitigation_strategies(risk_assessment),
            'monitoring_plan': self._generate_monitoring_plan(risk_assessment),
            'discharge_planning': self._generate_discharge_plan(risk_assessment)
        }
        
        if format.lower() == 'json':
            import json
            return json.dumps(report, indent=2, default=str)
        elif format.lower() == 'html':
            return self._generate_html_report(report)
        else:
            return report
    
    def _generate_mitigation_strategies(self, risk_assessment: Dict) -> List[Dict]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        risk_level = RiskLevel(risk_assessment['risk_summary']['overall_risk_level'])
        
        # General strategies based on risk level
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
            strategies.append({
                'category': 'preoperative',
                'strategy': 'Multidisciplinary team consultation',
                'rationale': 'High complexity case requiring multiple specialties',
                'priority': 'high'
            })
            
            strategies.append({
                'category': 'intraoperative',
                'strategy': 'Advanced hemodynamic monitoring',
                'rationale': 'High risk of cardiovascular complications',
                'priority': 'high'
            })
        
        # Complication-specific strategies
        complication_risks = risk_assessment['complication_risks']
        
        if complication_risks.get('infection', 0) > 0.1:
            strategies.append({
                'category': 'infection',
                'strategy': 'Extended antibiotic prophylaxis',
                'rationale': f"Elevated infection risk ({complication_risks['infection']:.1%})",
                'priority': 'medium'
            })
        
        if complication_risks.get('thrombosis', 0) > 0.05:
            strategies.append({
                'category': 'thrombosis',
                'strategy': 'Mechanical and pharmacological VTE prophylaxis',
                'rationale': f"Elevated VTE risk ({complication_risks['thrombosis']:.1%})",
                'priority': 'high'
            })
        
        if complication_risks.get('bleeding', 0) > 0.1:
            strategies.append({
                'category': 'bleeding',
                'strategy': 'Preoperative correction of coagulopathy if present',
                'rationale': f"Elevated bleeding risk ({complication_risks['bleeding']:.1%})",
                'priority': 'medium'
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        strategies.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return strategies
    
    def _generate_monitoring_plan(self, risk_assessment: Dict) -> Dict:
        """Generate postoperative monitoring plan"""
        risk_level = RiskLevel(risk_assessment['risk_summary']['overall_risk_level'])
        
        base_monitoring = {
            'vital_signs': {
                'frequency': 'q4h' if risk_level.value in ['low', 'moderate'] else 'q1h',
                'parameters': ['blood_pressure', 'heart_rate', 'respiratory_rate', 'temperature', 'spo2']
            },
            'laboratory_tests': {
                'frequency': 'daily',
                'tests': ['complete_blood_count', 'basic_metabolic_panel']
            },
            'clinical_assessment': {
                'frequency': 'shift' if risk_level.value in ['low', 'moderate'] else 'q4h',
                'components': ['wound_assessment', 'pain_assessment', 'neurological_assessment']
            }
        }
        
        # Enhanced monitoring for high risk
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
            base_monitoring['vital_signs']['frequency'] = 'continuous'
            base_monitoring['laboratory_tests']['frequency'] = 'q12h'
            base_monitoring['clinical_assessment']['frequency'] = 'q2h'
            
            # Add specialized monitoring
            base_monitoring['specialized_monitoring'] = {
                'arterial_line': risk_assessment['complication_risks'].get('cardiac', 0) > 0.1,
                'central_venous_pressure': risk_assessment['complication_risks'].get('organ_failure', 0) > 0.1,
                'urine_output': 'hourly',
                'continuous_cardiac_monitoring': True
            }
        
        return base_monitoring
    
    def _generate_discharge_plan(self, risk_assessment: Dict) -> Dict:
        """Generate discharge planning recommendations"""
        los_prediction = risk_assessment['length_of_stay_prediction']
        readmission_risk = risk_assessment.get('readmission_risk', 0.0)
        
        plan = {
            'estimated_los': los_prediction,
            'readmission_risk': f"{readmission_risk:.1%}",
            'discharge_criteria': [
                'Afebrile for 24 hours',
                'Adequate pain control on oral medications',
                'Tolerating diet',
                'Ambulating independently or with assistance',
                'Stable wound without signs of infection'
            ],
            'follow_up_plan': {
                'primary_care': 'Within 1 week of discharge',
                'surgical_follow_up': 'Within 2 weeks of discharge'
            }
        }
        
        # Additional services based on risk
        if readmission_risk > 0.15:
            plan['additional_services'] = [
                'Home health nursing',
                'Physical therapy at home',
                'Medication management service'
            ]
        
        if los_prediction > 7:
            plan['rehabilitation_recommendation'] = 'Consider inpatient rehabilitation'
        
        return plan
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML version of risk report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Risk Assessment Report - {report['report_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .risk-high {{ color: #d9534f; font-weight: bold; }}
                .risk-medium {{ color: #f0ad4e; }}
                .risk-low {{ color: #5cb85c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Patient Risk Assessment Report</h1>
                <p><strong>Report ID:</strong> {report['report_id']}</p>
                <p><strong>Generated:</strong> {report['generated_at']}</p>
                <p><strong>Model Type:</strong> {report['model_type']}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Overall Risk Level:</strong> 
                   <span class="risk-{report['executive_summary']['overall_risk_level']}">
                   {report['executive_summary']['overall_risk_level'].upper()}
                   </span>
                </p>
                <p><strong>Mortality Risk:</strong> {report['executive_summary']['mortality_risk_percentage']}</p>
                <p><strong>Interpretation:</strong> {report['executive_summary']['interpretation']}</p>
            </div>
            
            <div class="section">
                <h2>Key Risks</h2>
                <table>
                    <tr>
                        <th>Complication</th>
                        <th>Risk</th>
                        <th>Level</th>
                    </tr>
        """
        
        for risk in report['executive_summary']['key_risks']:
            html += f"""
                    <tr>
                        <td>{risk['complication'].replace('_', ' ').title()}</td>
                        <td>{risk['risk']}</td>
                        <td class="risk-{risk['level']}">{risk['level'].upper()}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in report['recommendations']:
            html += f"""
                    <li><strong>{rec['category'].title()}:</strong> {rec['recommendation']} 
                    <br><em>Rationale: {rec['rationale']}</em></li>
            """
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def batch_assess_risks(self, patients: List[PatientRiskProfile], 
                          procedure: Dict) -> List[Dict]:
        """Assess risks for multiple patients"""
        assessments = []
        
        for patient in patients:
            assessment = self.assess_patient_risk(patient, procedure)
            assessments.append(assessment)
        
        return assessments
    
    def compare_patients(self, patient1: PatientRiskProfile, 
                        patient2: PatientRiskProfile, 
                        procedure: Dict) -> Dict:
        """Compare risks between two patients"""
        assessment1 = self.assess_patient_risk(patient1, procedure)
        assessment2 = self.assess_patient_risk(patient2, procedure)
        
        comparison = {
            'patient1': {
                'id': patient1.patient_id,
                'assessment': assessment1
            },
            'patient2': {
                'id': patient2.patient_id,
                'assessment': assessment2
            },
            'comparison': {
                'mortality_risk_difference': abs(assessment1['mortality_risk'] - 
                                               assessment2['mortality_risk']),
                'overall_risk_difference': abs(assessment1['overall_surgical_risk'] - 
                                             assessment2['overall_surgical_risk']),
                'higher_risk_patient': patient1.patient_id if 
                    assessment1['overall_surgical_risk'] > assessment2['overall_surgical_risk'] 
                    else patient2.patient_id
            }
        }
        
        return comparison
    
    def export_predictions(self, format: str = 'csv') -> str:
        """Export prediction history"""
        if format.lower() == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Timestamp', 'Patient_ID', 'Procedure', 'Model_Type',
                           'Mortality_Risk', 'Overall_Risk', 'Risk_Level',
                           'LOS_Prediction', 'Readmission_Risk'])
            
            # Write data
            for record in self.prediction_history[-100:]:  # Last 100 predictions
                predictions = record['predictions']
                writer.writerow([
                    record['timestamp'],
                    record['patient_id'],
                    record['procedure'],
                    record['model_type'],
                    f"{predictions['mortality_risk']:.3f}",
                    f"{predictions['overall_surgical_risk']:.3f}",
                    predictions['risk_level'].value,
                    f"{predictions['length_of_stay_prediction']:.1f}",
                    f"{predictions.get('readmission_risk', 0.0):.3f}"
                ])
            
            return output.getvalue()
        
        elif format.lower() == 'json':
            import json
            return json.dumps(self.prediction_history[-100:], indent=2, default=str)
        
        else:
            return f"Format {format} not supported"
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up Risk Modeler...")
        self.prediction_history.clear()
        self.model_performance.clear()

# Utility functions

def create_patient_profile(patient_data: Dict) -> PatientRiskProfile:
    """Create patient risk profile from data"""
    return PatientRiskProfile(
        patient_id=patient_data.get('patient_id', 'unknown'),
        age=patient_data.get('age', 50),
        gender=patient_data.get('gender', 'unknown'),
        bmi=patient_data.get('bmi', 25.0),
        comorbidities=patient_data.get('comorbidities', []),
        medications=patient_data.get('medications', []),
        allergies=patient_data.get('allergies', []),
        surgical_history=patient_data.get('surgical_history', []),
        lab_values=patient_data.get('lab_values', {}),
        vital_signs=patient_data.get('vital_signs', {}),
        genetic_factors=patient_data.get('genetic_factors', {}),
        lifestyle_factors=patient_data.get('lifestyle_factors', {})
    )

def calculate_simple_risk_score(age: int, comorbidities: List[str], 
                              procedure_complexity: str) -> float:
    """Calculate simple risk score for quick assessment"""
    score = 0.0
    
    # Age component
    if age < 40:
        score += 0.1
    elif age < 60:
        score += 0.2
    elif age < 75:
        score += 0.3
    else:
        score += 0.4
    
    # Comorbidity component
    score += min(0.4, len(comorbidities) * 0.1)
    
    # Procedure complexity component
    complexity_scores = {'low': 0.1, 'medium': 0.2, 'high': 0.3, 'very_high': 0.4}
    score += complexity_scores.get(procedure_complexity, 0.2)
    
    return min(1.0, score)

# Example usage
def run_risk_modeling_demo():
    """Run risk modeling demonstration"""
    print("Running Risk Modeling Demo...")
    
    # Create sample patients
    patient1 = PatientRiskProfile(
        patient_id="P001",
        age=65,
        gender="male",
        bmi=28.5,
        comorbidities=["hypertension", "diabetes", "coronary_artery_disease"],
        medications=["lisinopril", "metformin", "atorvastatin"],
        allergies=["penicillin"],
        surgical_history=[{"procedure": "appendectomy", "year": 2005}],
        lab_values={
            'creatinine': 1.2,
            'hba1c': 7.2,
            'platelets': 220,
            'inr': 1.1
        },
        vital_signs={
            'systolic_bp': 140,
            'diastolic_bp': 85,
            'heart_rate': 72,
            'respiratory_rate': 16
        },
        genetic_factors={'CYP2C9': '*1/*1'},
        lifestyle_factors={'smoking': True, 'alcohol': False}
    )
    
    patient2 = PatientRiskProfile(
        patient_id="P002",
        age=45,
        gender="female",
        bmi=22.0,
        comorbidities=["hypertension"],
        medications=["amlodipine"],
        allergies=[],
        surgical_history=[],
        lab_values={
            'creatinine': 0.9,
            'hba1c': 5.8,
            'platelets': 250,
            'inr': 1.0
        },
        vital_signs={
            'systolic_bp': 120,
            'diastolic_bp': 78,
            'heart_rate': 68,
            'respiratory_rate': 14
        },
        genetic_factors={},
        lifestyle_factors={'smoking': False, 'alcohol': False}
    )
    
    # Procedure definition
    procedure = {
        'name': 'laparoscopic_cholecystectomy',
        'type': 'general',
        'complexity': 'medium',
        'emergency': False,
        'estimated_duration': 90,
        'typical_los': 2.0,
        'wound_classification': 'clean_contaminated'
    }
    
    # Initialize risk modeler
    modeler = RiskModeler(model_type='ensemble')
    
    # Assess risks for patient 1
    print(f"\nAssessing risks for Patient {patient1.patient_id}...")
    assessment1 = modeler.assess_patient_risk(patient1, procedure)
    
    print(f"Overall Risk Level: {assessment1['risk_summary']['overall_risk_level'].upper()}")
    print(f"Mortality Risk: {assessment1['risk_summary']['mortality_risk_percentage']}")
    print(f"Length of Stay Estimate: {assessment1['risk_summary']['length_of_stay_estimate']}")
    
    # Assess risks for patient 2
    print(f"\nAssessing risks for Patient {patient2.patient_id}...")
    assessment2 = modeler.assess_patient_risk(patient2, procedure)
    
    print(f"Overall Risk Level: {assessment2['risk_summary']['overall_risk_level'].upper()}")
    print(f"Mortality Risk: {assessment2['risk_summary']['mortality_risk_percentage']}")
    
    # Compare patients
    comparison = modeler.compare_patients(patient1, patient2, procedure)
    print(f"\nHigher Risk Patient: {comparison['comparison']['higher_risk_patient']}")
    print(f"Risk Difference: {comparison['comparison']['overall_risk_difference']:.3f}")
    
    # Generate detailed report
    report = modeler.generate_risk_report(patient1, procedure)
    if isinstance(report, dict):
        print(f"\nReport ID: {report['report_id']}")
        print(f"Key Recommendations:")
        for rec in report['recommendations'][:3]:
            print(f"  - {rec['recommendation']}")
    
    # Get model performance
    performance = modeler.get_model_performance()
    print(f"\nModel Performance:")
    print(f"  Total Validations: {performance.get('aggregate_stats', {}).get('total_validations', 0)}")
    print(f"  Mean Mortality Error: {performance.get('aggregate_stats', {}).get('mean_mortality_error', 0):.3f}")
    
    # Cleanup
    modeler.cleanup()
    
    print("\nDemo completed!")

if __name__ == "__main__":
    run_risk_modeling_demo()