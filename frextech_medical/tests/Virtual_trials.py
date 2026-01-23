#!/usr/bin/env python3
"""
Virtual Clinical Trials Module for FrexTech Medical
Patient population simulations for dosage optimization and outcome prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import time
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
import warnings
from collections import defaultdict

# Statistics and ML
from scipy import stats
from scipy.stats import norm, expon, poisson, beta, gamma
import pymc3 as pm
import arviz as az
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatientGender(Enum):
    """Patient gender"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class TrialPhase(Enum):
    """Clinical trial phases"""
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"
    PHASE_4 = "phase_4"

class DiseaseSeverity(Enum):
    """Disease severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class PatientProfile:
    """Virtual patient profile"""
    patient_id: str
    age: int
    gender: PatientGender
    weight_kg: float
    height_cm: float
    bmi: float
    disease_status: DiseaseSeverity
    comorbidities: List[str]
    genetic_markers: Dict[str, Any]
    biomarkers: Dict[str, float]
    medication_history: List[Dict[str, Any]]
    demographics: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_bmi(self) -> float:
        """Calculate BMI"""
        height_m = self.height_cm / 100
        return self.weight_kg / (height_m ** 2)
    
    def get_age_group(self) -> str:
        """Get age group"""
        if self.age < 18:
            return "pediatric"
        elif self.age < 65:
            return "adult"
        else:
            return "geriatric"

@dataclass
class TreatmentArm:
    """Treatment arm in virtual trial"""
    arm_id: str
    treatment_name: str
    dosage_mg: float
    frequency_days: int
    administration_route: str
    duration_days: int
    inclusion_criteria: Dict[str, Any]
    exclusion_criteria: Dict[str, Any]
    expected_mechanism: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrialOutcome:
    """Outcome of virtual trial"""
    outcome_id: str
    trial_id: str
    patient_id: str
    treatment_arm: str
    primary_endpoint: float
    secondary_endpoints: Dict[str, float]
    adverse_events: List[Dict[str, Any]]
    biomarker_changes: Dict[str, float]
    efficacy_score: float
    safety_score: float
    quality_of_life_score: float
    response_category: str
    dropout_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PopulationStatistics:
    """Population-level statistics"""
    trial_id: str
    n_patients: int
    response_rates: Dict[str, float]  # by category
    mean_primary_endpoint: float
    std_primary_endpoint: float
    adverse_event_rates: Dict[str, float]
    dropout_rate: float
    subgroup_analyses: Dict[str, Dict[str, float]]
    power_analysis: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

class VirtualTrialSimulator:
    """Virtual clinical trial simulator"""
    
    def __init__(self, cache_dir: str = "./virtual_trials_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Storage
        self.patient_populations: Dict[str, List[PatientProfile]] = {}
        self.treatment_arms: Dict[str, List[TreatmentArm]] = {}
        self.trial_results: Dict[str, List[TrialOutcome]] = {}
        self.population_stats: Dict[str, PopulationStatistics] = {}
        
        # Pharmacokinetic/Pharmacodynamic models
        self.pkpd_models: Dict[str, Any] = {}
        
        # ML models for outcome prediction
        self.outcome_predictors: Dict[str, Any] = {}
        
        # Configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "default_trial_duration_days": 90,
            "default_n_patients": 1000,
            "dropout_rate": 0.1,
            "adverse_event_rate": 0.15,
            "response_variability": 0.3,
            "age_distribution_mean": 45,
            "age_distribution_std": 15,
            "gender_distribution": {"male": 0.5, "female": 0.5},
            "bmi_distribution_mean": 26,
            "bmi_distribution_std": 5,
            "pk_model_type": "one_compartment",
            "pd_model_type": "emax"
        }
        
        config_file = self.cache_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}")
        
        return default_config
    
    def generate_patient_population(self, 
                                   population_id: str,
                                   n_patients: int,
                                   disease_type: str,
                                   age_range: Tuple[int, int] = (18, 80),
                                   gender_ratio: float = 0.5,
                                   comorbidity_prevalence: Dict[str, float] = None) -> List[PatientProfile]:
        """Generate a virtual patient population"""
        
        patients = []
        
        try:
            np.random.seed(42)  # For reproducibility
            
            for i in range(n_patients):
                # Generate patient characteristics
                age = int(np.random.uniform(age_range[0], age_range[1]))
                
                gender_choice = np.random.choice(
                    [PatientGender.MALE, PatientGender.FEMALE],
                    p=[gender_ratio, 1 - gender_ratio]
                )
                
                # Generate height and weight with correlation
                height = np.random.normal(170, 10)  # cm
                weight = np.random.normal(70, 15)  # kg
                
                # Ensure realistic values
                height = max(140, min(210, height))
                weight = max(40, min(150, weight))
                
                bmi = weight / ((height / 100) ** 2)
                
                # Disease severity (based on age and comorbidities)
                severity_probs = [0.3, 0.4, 0.2, 0.1]  # mild, moderate, severe, critical
                disease_severity = np.random.choice(
                    [DiseaseSeverity.MILD, DiseaseSeverity.MODERATE, 
                     DiseaseSeverity.SEVERE, DiseaseSeverity.CRITICAL],
                    p=severity_probs
                )
                
                # Generate comorbidities
                comorbidity_list = []
                if comorbidity_prevalence:
                    for comorbidity, prevalence in comorbidity_prevalence.items():
                        if np.random.random() < prevalence:
                            comorbidity_list.append(comorbidity)
                
                # Generate genetic markers
                genetic_markers = {
                    "CYP2D6": np.random.choice(["extensive", "intermediate", "poor"], 
                                              p=[0.7, 0.25, 0.05]),
                    "CYP3A4": np.random.choice(["normal", "reduced"], 
                                              p=[0.9, 0.1]),
                    "TPMT": np.random.choice(["normal", "deficient"], 
                                            p=[0.95, 0.05]),
                    "HLA-B*5701": np.random.choice(["positive", "negative"], 
                                                  p=[0.05, 0.95])
                }
                
                # Generate biomarkers
                biomarkers = {
                    "creatinine": np.random.normal(0.9, 0.2),
                    "ALT": np.random.normal(25, 10),
                    "AST": np.random.normal(22, 8),
                    "CRP": np.random.lognormal(1, 0.5),
                    "IL6": np.random.lognormal(1.5, 0.7)
                }
                
                # Generate medication history
                medication_history = []
                n_medications = np.random.poisson(2)  # Average 2 medications
                common_meds = ["Metformin", "Lisinopril", "Atorvastatin", 
                              "Levothyroxine", "Metoprolol"]
                
                for _ in range(n_medications):
                    med = np.random.choice(common_meds)
                    medication_history.append({
                        "medication": med,
                        "duration_days": int(np.random.exponential(365)),
                        "dose_mg": np.random.choice([5, 10, 20, 40, 80])
                    })
                
                # Create patient profile
                patient_id = f"{population_id}_PAT_{i:04d}"
                
                patient = PatientProfile(
                    patient_id=patient_id,
                    age=age,
                    gender=gender_choice,
                    weight_kg=weight,
                    height_cm=height,
                    bmi=bmi,
                    disease_status=disease_severity,
                    comorbidities=comorbidity_list,
                    genetic_markers=genetic_markers,
                    biomarkers=biomarkers,
                    medication_history=medication_history,
                    demographics={
                        "population_id": population_id,
                        "disease_type": disease_type,
                        "smoking_status": np.random.choice(["never", "former", "current"], 
                                                          p=[0.5, 0.3, 0.2]),
                        "alcohol_use": np.random.choice(["none", "moderate", "heavy"], 
                                                       p=[0.3, 0.6, 0.1]),
                        "ethnicity": np.random.choice(["Caucasian", "African", "Asian", "Hispanic"], 
                                                     p=[0.6, 0.15, 0.15, 0.1])
                    }
                )
                
                patients.append(patient)
            
            # Store population
            self.patient_populations[population_id] = patients
            
            logger.info(f"Generated {n_patients} virtual patients for population {population_id}")
            
            return patients
            
        except Exception as e:
            logger.error(f"Error generating patient population: {e}")
            return []
    
    def create_treatment_arms(self, 
                             trial_id: str,
                             treatment_configs: List[Dict[str, Any]]) -> List[TreatmentArm]:
        """Create treatment arms for virtual trial"""
        
        arms = []
        
        try:
            for i, config in enumerate(treatment_configs):
                arm_id = f"{trial_id}_ARM_{i:02d}"
                
                arm = TreatmentArm(
                    arm_id=arm_id,
                    treatment_name=config.get("name", f"Treatment_{i}"),
                    dosage_mg=config.get("dosage_mg", 100),
                    frequency_days=config.get("frequency_days", 1),
                    administration_route=config.get("administration_route", "oral"),
                    duration_days=config.get("duration_days", self.config["default_trial_duration_days"]),
                    inclusion_criteria=config.get("inclusion_criteria", {}),
                    exclusion_criteria=config.get("exclusion_criteria", {}),
                    expected_mechanism=config.get("mechanism", "unknown"),
                    metadata={
                        "trial_id": trial_id,
                        "arm_index": i,
                        "is_control": config.get("is_control", False),
                        "randomization_ratio": config.get("randomization_ratio", 1.0)
                    }
                )
                
                arms.append(arm)
            
            # Store arms
            self.treatment_arms[trial_id] = arms
            
            logger.info(f"Created {len(arms)} treatment arms for trial {trial_id}")
            
            return arms
            
        except Exception as e:
            logger.error(f"Error creating treatment arms: {e}")
            return []
    
    def simulate_pk_profile(self, 
                           patient: PatientProfile,
                           treatment: TreatmentArm,
                           time_points: List[float]) -> Dict[str, List[float]]:
        """Simulate pharmacokinetic profile"""
        
        try:
            # One-compartment model with first-order absorption
            # C(t) = (Dose * F * ka) / (V * (ka - ke)) * (e^(-ke*t) - e^(-ka*t))
            
            # Model parameters (individualized based on patient characteristics)
            dose = treatment.dosage_mg
            bioavailability = 0.8  # F
            absorption_rate = 1.0  # ka, 1/h
            elimination_rate = 0.1  # ke, 1/h
            volume = 50  # V, L
            
            # Adjust based on patient factors
            if patient.age > 65:
                elimination_rate *= 0.8  # Reduced clearance in elderly
                volume *= 0.9
            
            if patient.weight_kg > 100:
                volume *= 1.2  # Larger volume in heavier patients
            
            if "renal_impairment" in patient.comorbidities:
                elimination_rate *= 0.6  # Reduced renal clearance
            
            # Genetic polymorphisms affecting metabolism
            if patient.genetic_markers.get("CYP2D6") == "poor":
                elimination_rate *= 0.5
            elif patient.genetic_markers.get("CYP2D6") == "extensive":
                elimination_rate *= 1.2
            
            # Calculate concentration at each time point
            concentrations = []
            for t in time_points:
                if t <= 0:
                    c = 0
                else:
                    c = (dose * bioavailability * absorption_rate) / \
                        (volume * (absorption_rate - elimination_rate)) * \
                        (np.exp(-elimination_rate * t) - np.exp(-absorption_rate * t))
                
                # Add some variability
                c *= np.random.lognormal(0, 0.2)  # 20% CV
                concentrations.append(max(0, c))
            
            # Calculate AUC (Area Under Curve)
            auc = np.trapz(concentrations, time_points)
            
            # Calculate Cmax and Tmax
            cmax = max(concentrations)
            tmax = time_points[concentrations.index(cmax)] if cmax > 0 else 0
            
            # Calculate half-life
            if elimination_rate > 0:
                half_life = np.log(2) / elimination_rate
            else:
                half_life = 0
            
            pk_profile = {
                "time_points": time_points,
                "concentrations": concentrations,
                "auc": auc,
                "cmax": cmax,
                "tmax": tmax,
                "half_life": half_life,
                "parameters": {
                    "dose_mg": dose,
                    "bioavailability": bioavailability,
                    "absorption_rate": absorption_rate,
                    "elimination_rate": elimination_rate,
                    "volume_l": volume
                }
            }
            
            return pk_profile
            
        except Exception as e:
            logger.error(f"Error simulating PK profile: {e}")
            return {}
    
    def simulate_pd_response(self, 
                            pk_profile: Dict[str, Any],
                            patient: PatientProfile,
                            treatment: TreatmentArm,
                            time_points: List[float]) -> Dict[str, List[float]]:
        """Simulate pharmacodynamic response"""
        
        try:
            # Emax model: E = E0 + (Emax * C) / (EC50 + C)
            
            # Baseline effect
            e0 = 0.0
            
            # Maximum effect
            emax = 1.0  # Normalized to 0-1 scale
            
            # Concentration for 50% effect
            ec50 = 5.0  # mg/L
            
            # Individual variability
            if patient.disease_status == DiseaseSeverity.SEVERE:
                emax *= 1.2  # Greater effect in severe disease
            elif patient.disease_status == DiseaseSeverity.MILD:
                emax *= 0.8  # Smaller effect in mild disease
            
            # Genetic factors
            if patient.genetic_markers.get("target_receptor") == "high_affinity":
                ec50 *= 0.7  # Lower EC50 for high affinity
            
            # Calculate effect at each time point
            effects = []
            concentrations = pk_profile.get("concentrations", [0] * len(time_points))
            
            for c in concentrations:
                if c <= 0:
                    effect = e0
                else:
                    effect = e0 + (emax * c) / (ec50 + c)
                
                # Add variability
                effect *= np.random.normal(1, 0.1)  # 10% variability
                effects.append(max(0, min(1, effect)))
            
            # Calculate response metrics
            max_effect = max(effects) if effects else 0
            time_to_response = None
            
            for t, e in zip(time_points, effects):
                if e > 0.5 * max_effect and time_to_response is None:
                    time_to_response = t
            
            pd_response = {
                "time_points": time_points,
                "effects": effects,
                "max_effect": max_effect,
                "time_to_response": time_to_response,
                "parameters": {
                    "e0": e0,
                    "emax": emax,
                    "ec50": ec50,
                    "patient_sensitivity": emax / ec50  # Sensitivity index
                }
            }
            
            return pd_response
            
        except Exception as e:
            logger.error(f"Error simulating PD response: {e}")
            return {}
    
    def simulate_adverse_events(self, 
                               patient: PatientProfile,
                               treatment: TreatmentArm,
                               pk_profile: Dict[str, Any],
                               duration_days: int) -> List[Dict[str, Any]]:
        """Simulate adverse events"""
        
        adverse_events = []
        
        try:
            # Base rates for common adverse events
            base_rates = {
                "nausea": 0.15,
                "headache": 0.10,
                "fatigue": 0.08,
                "rash": 0.05,
                "diarrhea": 0.07,
                "elevated_liver_enzymes": 0.03,
                "renal_impairment": 0.02
            }
            
            # Adjust based on treatment
            if treatment.dosage_mg > 100:
                for ae in base_rates:
                    base_rates[ae] *= 1.5  # Higher dose = higher AE risk
            
            # Adjust based on patient factors
            if patient.age > 65:
                base_rates["renal_impairment"] *= 2.0
                base_rates["fatigue"] *= 1.5
            
            if "liver_disease" in patient.comorbidities:
                base_rates["elevated_liver_enzymes"] *= 3.0
            
            # Genetic factors
            if patient.genetic_markers.get("HLA-B*5701") == "positive":
                base_rates["rash"] *= 10.0  # Hypersensitivity reaction
            
            # Simulate events
            for ae_name, base_rate in base_rates.items():
                # Daily probability
                daily_prob = base_rate / duration_days
                
                # Simulate occurrence
                if np.random.random() < daily_prob * duration_days:
                    # Random onset day
                    onset_day = int(np.random.uniform(1, duration_days))
                    
                    # Severity (1=mild, 2=moderate, 3=severe, 4=life-threatening)
                    severity = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.15, 0.05])
                    
                    # Duration
                    duration = int(np.random.exponential(7))  # Average 7 days
                    
                    # Action taken
                    actions = ["none", "dose_reduction", "treatment_interruption", "treatment_discontinuation"]
                    action_probs = [0.3, 0.3, 0.2, 0.2]
                    action_taken = np.random.choice(actions, p=action_probs)
                    
                    adverse_events.append({
                        "event": ae_name,
                        "onset_day": onset_day,
                        "severity": severity,
                        "duration_days": duration,
                        "action_taken": action_taken,
                        "related_to_treatment": True,
                        "serious": severity >= 3
                    })
            
            return adverse_events
            
        except Exception as e:
            logger.error(f"Error simulating adverse events: {e}")
            return []
    
    def run_virtual_trial(self, 
                         trial_id: str,
                         population_id: str,
                         treatment_configs: List[Dict[str, Any]],
                         primary_endpoint: str,
                         secondary_endpoints: List[str],
                         n_patients_per_arm: int = 100) -> List[TrialOutcome]:
        """Run virtual clinical trial"""
        
        outcomes = []
        
        try:
            # Get patient population
            if population_id not in self.patient_populations:
                logger.error(f"Population {population_id} not found")
                return []
            
            patients = self.patient_populations[population_id]
            
            # Create treatment arms
            treatment_arms = self.create_treatment_arms(trial_id, treatment_configs)
            
            # Randomize patients to treatment arms
            np.random.shuffle(patients)
            
            # Assign patients to arms
            assignments = []
            arm_index = 0
            for patient in patients[:n_patients_per_arm * len(treatment_arms)]:
                arm = treatment_arms[arm_index % len(treatment_arms)]
                assignments.append((patient, arm))
                arm_index += 1
            
            logger.info(f"Running virtual trial {trial_id} with {len(assignments)} patients")
            
            # Simulate outcomes for each patient
            for patient, treatment in assignments:
                try:
                    # Check inclusion/exclusion criteria
                    if not self._check_eligibility(patient, treatment):
                        continue
                    
                    # Simulate PK profile
                    time_points = list(range(0, treatment.duration_days + 1, 7))  # Weekly
                    pk_profile = self.simulate_pk_profile(patient, treatment, time_points)
                    
                    # Simulate PD response
                    pd_response = self.simulate_pd_response(pk_profile, patient, treatment, time_points)
                    
                    # Simulate adverse events
                    adverse_events = self.simulate_adverse_events(patient, treatment, pk_profile, treatment.duration_days)
                    
                    # Calculate primary endpoint (e.g., disease activity score)
                    baseline_score = 5.0  # Example baseline
                    
                    # Treatment effect
                    treatment_effect = pd_response.get("max_effect", 0)
                    
                    # Add variability
                    variability = np.random.normal(0, self.config["response_variability"])
                    
                    # Final score (lower is better)
                    final_score = baseline_score * (1 - treatment_effect) + variability
                    final_score = max(0, final_score)
                    
                    # Calculate secondary endpoints
                    secondary_endpoint_values = {}
                    for endpoint in secondary_endpoints:
                        if endpoint == "quality_of_life":
                            # Improve with treatment effect, worsen with adverse events
                            qol = 0.7 + (treatment_effect * 0.3) - (len(adverse_events) * 0.05)
                            secondary_endpoint_values[endpoint] = max(0, min(1, qol))
                        elif endpoint == "biomarker_change":
                            # CRP reduction
                            crp_reduction = treatment_effect * 0.5
                            secondary_endpoint_values[endpoint] = crp_reduction
                        else:
                            secondary_endpoint_values[endpoint] = np.random.random()
                    
                    # Calculate efficacy score
                    efficacy_score = treatment_effect
                    
                    # Calculate safety score (1 - normalized AE burden)
                    ae_burden = sum(ae.get("severity", 1) for ae in adverse_events) / max(1, len(adverse_events))
                    safety_score = max(0, 1 - (ae_burden / 4))  # Normalize to 0-1
                    
                    # Quality of life score
                    quality_of_life_score = secondary_endpoint_values.get("quality_of_life", 0.5)
                    
                    # Determine response category
                    if efficacy_score > 0.7 and safety_score > 0.7:
                        response_category = "EXCELLENT_RESPONSE"
                    elif efficacy_score > 0.5 and safety_score > 0.5:
                        response_category = "GOOD_RESPONSE"
                    elif efficacy_score > 0.3:
                        response_category = "MODERATE_RESPONSE"
                    else:
                        response_category = "POOR_RESPONSE"
                    
                    # Check for dropout
                    dropout_reason = None
                    dropout_prob = self.config["dropout_rate"]
                    
                    # Increase dropout if severe adverse events
                    severe_aes = [ae for ae in adverse_events if ae.get("severity", 0) >= 3]
                    if severe_aes:
                        dropout_prob *= 2
                    
                    if np.random.random() < dropout_prob:
                        dropout_reason = np.random.choice([
                            "adverse_events", "lack_of_efficacy", 
                            "patient_decision", "protocol_violation"
                        ])
                    
                    # Biomarker changes
                    biomarker_changes = {}
                    for biomarker, baseline in patient.biomarkers.items():
                        change = np.random.normal(treatment_effect * 0.5, 0.2)
                        biomarker_changes[biomarker] = baseline * (1 - change)
                    
                    # Create outcome
                    outcome_id = f"{trial_id}_{patient.patient_id}"
                    
                    outcome = TrialOutcome(
                        outcome_id=outcome_id,
                        trial_id=trial_id,
                        patient_id=patient.patient_id,
                        treatment_arm=treatment.arm_id,
                        primary_endpoint=final_score,
                        secondary_endpoints=secondary_endpoint_values,
                        adverse_events=adverse_events,
                        biomarker_changes=biomarker_changes,
                        efficacy_score=efficacy_score,
                        safety_score=safety_score,
                        quality_of_life_score=quality_of_life_score,
                        response_category=response_category,
                        dropout_reason=dropout_reason,
                        metadata={
                            "treatment_duration": treatment.duration_days,
                            "pk_auc": pk_profile.get("auc", 0),
                            "pk_cmax": pk_profile.get("cmax", 0),
                            "pd_max_effect": pd_response.get("max_effect", 0),
                            "num_adverse_events": len(adverse_events),
                            "severe_adverse_events": len(severe_aes)
                        }
                    )
                    
                    outcomes.append(outcome)
                    
                except Exception as e:
                    logger.error(f"Error simulating outcome for patient {patient.patient_id}: {e}")
                    continue
            
            # Store results
            self.trial_results[trial_id] = outcomes
            
            # Calculate population statistics
            self._calculate_population_statistics(trial_id, outcomes)
            
            logger.info(f"Virtual trial {trial_id} completed: {len(outcomes)} outcomes")
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error running virtual trial: {e}")
            return []
    
    def _check_eligibility(self, patient: PatientProfile, treatment: TreatmentArm) -> bool:
        """Check if patient meets eligibility criteria"""
        
        # Check inclusion criteria
        inclusion = treatment.inclusion_criteria
        
        if inclusion.get("min_age", 0) > patient.age:
            return False
        
        if inclusion.get("max_age", 150) < patient.age:
            return False
        
        if inclusion.get("disease_severity", []):
            if patient.disease_status.value not in inclusion["disease_severity"]:
                return False
        
        # Check exclusion criteria
        exclusion = treatment.exclusion_criteria
        
        if exclusion.get("comorbidities", []):
            for comorbidity in exclusion["comorbidities"]:
                if comorbidity in patient.comorbidities:
                    return False
        
        if exclusion.get("age_range", (0, 150)):
            min_age, max_age = exclusion["age_range"]
            if min_age <= patient.age <= max_age:
                return False
        
        return True
    
    def _calculate_population_statistics(self, trial_id: str, outcomes: List[TrialOutcome]):
        """Calculate population-level statistics"""
        
        try:
            if not outcomes:
                return
            
            # Group by treatment arm
            outcomes_by_arm = {}
            for outcome in outcomes:
                arm = outcome.treatment_arm
                if arm not in outcomes_by_arm:
                    outcomes_by_arm[arm] = []
                outcomes_by_arm[arm].append(outcome)
            
            # Calculate statistics for each arm
            population_stats = {}
            
            for arm, arm_outcomes in outcomes_by_arm.items():
                n_patients = len(arm_outcomes)
                
                # Response rates
                response_counts = {}
                for outcome in arm_outcomes:
                    category = outcome.response_category
                    response_counts[category] = response_counts.get(category, 0) + 1
                
                response_rates = {cat: count / n_patients for cat, count in response_counts.items()}
                
                # Primary endpoint statistics
                primary_endpoints = [o.primary_endpoint for o in arm_outcomes]
                mean_primary = np.mean(primary_endpoints)
                std_primary = np.std(primary_endpoints)
                
                # Adverse event rates
                adverse_event_counts = {}
                for outcome in arm_outcomes:
                    for ae in outcome.adverse_events:
                        event_type = ae.get("event", "unknown")
                        adverse_event_counts[event_type] = adverse_event_counts.get(event_type, 0) + 1
                
                adverse_event_rates = {ae: count / n_patients for ae, count in adverse_event_counts.items()}
                
                # Dropout rate
                dropouts = sum(1 for o in arm_outcomes if o.dropout_reason)
                dropout_rate = dropouts / n_patients
                
                # Subgroup analyses (simplified)
                subgroup_analyses = {
                    "by_age": {
                        "<65": np.mean([o.efficacy_score for o in arm_outcomes if getattr(o, '_patient_age', 45) < 65]),
                        ">=65": np.mean([o.efficacy_score for o in arm_outcomes if getattr(o, '_patient_age', 45) >= 65])
                    },
                    "by_gender": {
                        "male": np.mean([o.efficacy_score for o in arm_outcomes if getattr(o, '_patient_gender', 'male') == 'male']),
                        "female": np.mean([o.efficacy_score for o in arm_outcomes if getattr(o, '_patient_gender', 'female') == 'female'])
                    }
                }
                
                # Power analysis (simplified)
                # Assuming we want to detect effect size of 0.5 with 80% power
                effect_size = mean_primary if arm_outcomes else 0
                power_analysis = {
                    "detectable_effect": 0.5,
                    "achieved_power": min(0.99, effect_size * 2),  # Simplified
                    "required_sample_size": int(100 / (effect_size + 0.1))  # Simplified
                }
                
                # Confidence intervals
                if n_patients > 1:
                    ci_low, ci_high = stats.t.interval(
                        0.95, 
                        len(primary_endpoints)-1, 
                        loc=mean_primary, 
                        scale=stats.sem(primary_endpoints)
                    )
                    confidence_intervals = {
                        "primary_endpoint": (float(ci_low), float(ci_high)),
                        "efficacy_score": (0.5, 0.8),  # Placeholder
                        "safety_score": (0.6, 0.9)     # Placeholder
                    }
                else:
                    confidence_intervals = {}
                
                # Create population statistics
                stats = PopulationStatistics(
                    trial_id=trial_id,
                    n_patients=n_patients,
                    response_rates=response_rates,
                    mean_primary_endpoint=mean_primary,
                    std_primary_endpoint=std_primary,
                    adverse_event_rates=adverse_event_rates,
                    dropout_rate=dropout_rate,
                    subgroup_analyses=subgroup_analyses,
                    power_analysis=power_analysis,
                    confidence_intervals=confidence_intervals,
                    metadata={
                        "treatment_arm": arm,
                        "calculation_date": datetime.now().isoformat()
                    }
                )
                
                population_stats[arm] = stats
            
            # Store statistics
            self.population_stats[trial_id] = population_stats
            
        except Exception as e:
            logger.error(f"Error calculating population statistics: {e}")
    
    def optimize_dosage(self, 
                       trial_id: str,
                       target_efficacy: float = 0.7,
                       max_toxicity: float = 0.3,
                       dosage_range: Tuple[float, float] = (10, 500),
                       n_simulations: int = 100) -> Dict[str, Any]:
        """Optimize dosage using simulation"""
        
        try:
            # Get trial results
            if trial_id not in self.trial_results:
                logger.error(f"Trial {trial_id} not found")
                return {}
            
            outcomes = self.trial_results[trial_id]
            
            # Extract dosage-response relationship
            dosages = []
            efficacies = []
            toxicities = []
            
            for outcome in outcomes:
                # Get treatment arm info
                for arm_id, stats in self.population_stats.get(trial_id, {}).items():
                    if arm_id == outcome.treatment_arm:
                        # Extract dosage from arm ID or metadata
                        dosage = float(arm_id.split('_')[-1]) if '_' in arm_id else 100
                        dosages.append(dosage)
                        efficacies.append(outcome.efficacy_score)
                        
                        # Calculate toxicity score from adverse events
                        ae_severity = sum(ae.get("severity", 1) for ae in outcome.adverse_events)
                        toxicity = min(1.0, ae_severity / (len(outcome.adverse_events) + 1) / 4)
                        toxicities.append(toxicity)
                        break
            
            if not dosages:
                return {}
            
            # Fit dose-response curve
            dosages = np.array(dosages)
            efficacies = np.array(efficacies)
            toxicities = np.array(toxicities)
            
            # Simple polynomial fit
            efficacy_coeffs = np.polyfit(dosages, efficacies, 2)
            toxicity_coeffs = np.polyfit(dosages, toxicities, 2)
            
            efficacy_poly = np.poly1d(efficacy_coeffs)
            toxicity_poly = np.poly1d(toxicity_coeffs)
            
            # Find optimal dosage
            test_dosages = np.linspace(dosage_range[0], dosage_range[1], n_simulations)
            test_efficacies = efficacy_poly(test_dosages)
            test_toxicities = toxicity_poly(test_dosages)
            
            # Calculate utility function
            utility = test_efficacies - test_toxicities
            
            # Find dosage with maximum utility within constraints
            valid_indices = np.where((test_efficacies >= target_efficacy) & 
                                    (test_toxicities <= max_toxicity))[0]
            
            if len(valid_indices) > 0:
                best_idx = valid_indices[np.argmax(utility[valid_indices])]
                optimal_dosage = test_dosages[best_idx]
                predicted_efficacy = test_efficacies[best_idx]
                predicted_toxicity = test_toxicities[best_idx]
            else:
                # If no dosage meets constraints, find best compromise
                best_idx = np.argmax(utility)
                optimal_dosage = test_dosages[best_idx]
                predicted_efficacy = test_efficacies[best_idx]
                predicted_toxicity = test_toxicities[best_idx]
            
            # Calculate therapeutic window
            therapeutic_index = predicted_efficacy / (predicted_toxicity + 0.001)
            
            result = {
                "optimal_dosage_mg": float(optimal_dosage),
                "predicted_efficacy": float(predicted_efficacy),
                "predicted_toxicity": float(predicted_toxicity),
                "therapeutic_index": float(therapeutic_index),
                "meets_efficacy_target": predicted_efficacy >= target_efficacy,
                "meets_toxicity_limit": predicted_toxicity <= max_toxicity,
                "dosage_response_curve": {
                    "dosages": test_dosages.tolist(),
                    "efficacies": test_efficacies.tolist(),
                    "toxicities": test_toxicities.tolist(),
                    "utilities": utility.tolist()
                },
                "fitting_parameters": {
                    "efficacy_coefficients": efficacy_coeffs.tolist(),
                    "toxicity_coefficients": toxicity_coeffs.tolist()
                }
            }
            
            logger.info(f"Dosage optimization: {optimal_dosage:.1f} mg, Efficacy: {predicted_efficacy:.3f}, Toxicity: {predicted_toxicity:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing dosage: {e}")
            return {}
    
    def create_trial_report(self, 
                           trial_id: str,
                           output_file: str = "trial_report.html") -> bool:
        """Create comprehensive trial report"""
        
        try:
            if trial_id not in self.trial_results:
                logger.error(f"Trial {trial_id} not found")
                return False
            
            outcomes = self.trial_results[trial_id]
            stats = self.population_stats.get(trial_id, {})
            
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Virtual Trial Report: {trial_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                    .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .stat {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Virtual Clinical Trial Report</h1>
                    <h2>Trial ID: {trial_id}</h2>
                    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h3>Executive Summary</h3>
                    <p>Total Patients: {len(outcomes)}</p>
                    <p>Treatment Arms: {len(stats)}</p>
                    <p>Simulation Date: {datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
            """
            
            # Add statistics for each treatment arm
            for arm_id, arm_stats in stats.items():
                html_content += f"""
                <div class="section">
                    <h3>Treatment Arm: {arm_id}</h3>
                    
                    <div class="stat">
                        <strong>Patients:</strong> {arm_stats.n_patients}
                    </div>
                    <div class="stat">
                        <strong>Mean Efficacy:</strong> {arm_stats.mean_primary_endpoint:.3f}
                    </div>
                    <div class="stat">
                        <strong>Dropout Rate:</strong> {arm_stats.dropout_rate:.1%}
                    </div>
                    
                    <h4>Response Rates</h4>
                    <table>
                        <tr>
                            <th>Response Category</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                """
                
                for category, rate in arm_stats.response_rates.items():
                    count = int(rate * arm_stats.n_patients)
                    html_content += f"""
                        <tr>
                            <td>{category}</td>
                            <td>{count}</td>
                            <td>{rate:.1%}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                    
                    <h4>Adverse Events</h4>
                    <table>
                        <tr>
                            <th>Event Type</th>
                            <th>Rate</th>
                        </tr>
                """
                
                for event_type, rate in arm_stats.adverse_event_rates.items():
                    html_content += f"""
                        <tr>
                            <td>{event_type}</td>
                            <td>{rate:.1%}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
                <div class="section">
                    <h3>Conclusions</h3>
                    <p>This virtual trial simulation provides insights into expected outcomes for the tested treatment regimens. The results should be validated with real-world data when available.</p>
                </div>
            </body>
            </html>
            """
            
            # Write HTML file
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Trial report saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating trial report: {e}")
            return False
    
    def predict_real_world_outcomes(self, 
                                  trial_id: str,
                                  real_world_population: List[PatientProfile]) -> List[TrialOutcome]:
        """Predict outcomes for real-world population based on virtual trial"""
        
        try:
            # This would use the virtual trial results to build a prediction model
            # and apply it to the real-world population
            
            logger.info(f"Predicting outcomes for {len(real_world_population)} real-world patients")
            
            # For now, return empty list
            # In production, implement actual prediction model
            
            return []
            
        except Exception as e:
            logger.error(f"Error predicting real-world outcomes: {e}")
            return []

# Factory function
def create_virtual_trial_simulator(config_file: str = "virtual_trials_config.json") -> VirtualTrialSimulator:
    """Create and configure virtual trial simulator"""
    simulator = VirtualTrialSimulator()
    
    try:
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                simulator.config.update(config)
        
        logger.info("Virtual trial simulator initialized")
        
    except Exception as e:
        logger.warning(f"Error loading config: {e}")
    
    return simulator

# Example usage
def example_virtual_trial():
    """Example virtual trial workflow"""
    
    # Create simulator
    simulator = create_virtual_trial_simulator()
    
    print("Virtual Clinical Trial Simulation")
    print("=" * 50)
    
    # Generate patient population
    print("\n1. Generating virtual patient population...")
    patients = simulator.generate_patient_population(
        population_id="RHEUMATOID_ARTHRITIS_2024",
        n_patients=500,
        disease_type="Rheumatoid Arthritis",
        age_range=(30, 75),
        gender_ratio=0.7,  # 70% female (common in RA)
        comorbidity_prevalence={
            "hypertension": 0.3,
            "diabetes": 0.15,
            "osteoporosis": 0.2,
            "liver_disease": 0.05
        }
    )
    
    print(f"Generated {len(patients)} virtual patients")
    
    # Define treatment arms
    print("\n2. Defining treatment arms...")
    treatment_configs = [
        {
            "name": "Placebo",
            "dosage_mg": 0,
            "frequency_days": 1,
            "is_control": True,
            "inclusion_criteria": {"min_age": 18, "max_age": 75},
            "exclusion_criteria": {"comorbidities": ["severe_liver_disease"]}
        },
        {
            "name": "Standard Dose",
            "dosage_mg": 50,
            "frequency_days": 1,
            "inclusion_criteria": {"min_age": 18, "max_age": 75},
            "exclusion_criteria": {"comorbidities": ["severe_liver_disease"]}
        },
        {
            "name": "High Dose",
            "dosage_mg": 100,
            "frequency_days": 1,
            "inclusion_criteria": {"min_age": 18, "max_age": 75},
            "exclusion_criteria": {"comorbidities": ["severe_liver_disease", "renal_impairment"]}
        }
    ]
    
    # Run virtual trial
    print("\n3. Running virtual trial...")
    trial_id = "RA_TRIAL_2024_001"
    outcomes = simulator.run_virtual_trial(
        trial_id=trial_id,
        population_id="RHEUMATOID_ARTHRITIS_2024",
        treatment_configs=treatment_configs,
        primary_endpoint="disease_activity_score",
        secondary_endpoints=["quality_of_life", "biomarker_change"],
        n_patients_per_arm=50
    )
    
    print(f"Trial completed: {len(outcomes)} outcomes")
    
    # Analyze results
    print("\n4. Analyzing results...")
    if trial_id in simulator.population_stats:
        for arm_id, stats in simulator.population_stats[trial_id].items():
            print(f"\n{arm_id}:")
            print(f"  Patients: {stats.n_patients}")
            print(f"  Mean Efficacy Score: {stats.mean_primary_endpoint:.3f}")
            print(f"  Dropout Rate: {stats.dropout_rate:.1%}")
            
            # Show response rates
            for category, rate in stats.response_rates.items():
                print(f"  {category}: {rate:.1%}")
    
    # Optimize dosage
    print("\n5. Optimizing dosage...")
    dosage_opt = simulator.optimize_dosage(
        trial_id=trial_id,
        target_efficacy=0.7,
        max_toxicity=0.3,
        dosage_range=(10, 150),
        n_simulations=50
    )
    
    if dosage_opt:
        print(f"Optimal Dosage: {dosage_opt['optimal_dosage_mg']:.1f} mg")
        print(f"Predicted Efficacy: {dosage_opt['predicted_efficacy']:.3f}")
        print(f"Predicted Toxicity: {dosage_opt['predicted_toxicity']:.3f}")
        print(f"Therapeutic Index: {dosage_opt['therapeutic_index']:.2f}")
    
    # Create report
    print("\n6. Creating trial report...")
    simulator.create_trial_report(trial_id, "ra_trial_report.html")
    print("Report saved to ra_trial_report.html")
    
    print("\n✅ Virtual trial example completed")

if __name__ == "__main__":
    example_virtual_trial()