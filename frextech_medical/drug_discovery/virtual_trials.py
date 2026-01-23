"""
Virtual Trials Module
Simulate patient populations and clinical trial outcomes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TrialPhase(Enum):
    """Clinical trial phases"""
    PHASE_1 = "Phase I"
    PHASE_2 = "Phase II"
    PHASE_3 = "Phase III"
    PHASE_4 = "Phase IV"

class PatientStatus(Enum):
    """Patient status in trial"""
    SCREENED = "Screened"
    RANDOMIZED = "Randomized"
    COMPLETED = "Completed"
    DROPPED_OUT = "Dropped Out"
    ADVERSE_EVENT = "Adverse Event"

@dataclass
class PatientProfile:
    """Individual patient profile"""
    patient_id: str
    age: int
    gender: str
    weight: float  # kg
    height: float  # cm
    bmi: float
    genotype: Dict[str, str]  # Genetic markers
    phenotype: Dict[str, float]  # Physiological parameters
    comorbidities: List[str]
    medication_history: List[str]
    inclusion_criteria: Dict[str, bool]
    exclusion_criteria: Dict[str, bool]
    
    @property
    def bsa(self) -> float:
        """Body Surface Area (Mosteller formula)"""
        return np.sqrt((self.height * self.weight) / 3600)
    
    @property
    def creatinine_clearance(self) -> float:
        """Estimated creatinine clearance (Cockcroft-Gault)"""
        if self.gender.lower() == 'male':
            return ((140 - self.age) * self.weight) / (72 * 1.0)
        else:
            return ((140 - self.age) * self.weight) / (72 * 1.0) * 0.85

class VirtualTrialSimulator:
    """Simulate virtual clinical trials with patient populations"""
    
    def __init__(self, simulation_fidelity='high', random_seed=42):
        """
        Initialize virtual trial simulator
        
        Args:
            simulation_fidelity: Level of detail ('low', 'medium', 'high')
            random_seed: Random seed for reproducibility
        """
        self.simulation_fidelity = simulation_fidelity
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Trial parameters
        self.trial_phase = TrialPhase.PHASE_2
        self.trial_design = {
            'design': 'randomized_double_blind',
            'arms': ['treatment', 'control', 'placebo'],
            'randomization_ratio': [2, 1, 1],
            'duration_days': 180,
            'visits': [0, 7, 14, 30, 60, 90, 180],
            'primary_endpoint': 'efficacy_score',
            'secondary_endpoints': [
                'safety_score', 'quality_of_life', 'biomarker_change'
            ]
        }
        
        # Patient population database
        self.patient_database = []
        self.disease_models = {}
        
        # Pharmacokinetic/Pharmacodynamic (PK/PD) models
        self.pk_models = {}
        self.pd_models = {}
        
        # Statistical models
        self.statistical_models = {
            'efficacy': self._simulate_efficacy_response,
            'safety': self._simulate_safety_profile,
            'dropout': self._simulate_dropout_rates,
            'adverse_events': self._simulate_adverse_events
        }
        
        # Results storage
        self.trial_results = {}
        self.simulation_history = []
        
        self.initialized = False
        
    def initialize(self, disease_type=None, population_size=1000):
        """Initialize the virtual trial simulator"""
        print(f"Initializing Virtual Trial Simulator ({self.simulation_fidelity} fidelity)...")
        
        # Load disease models
        self._load_disease_models(disease_type)
        
        # Generate virtual patient population
        self._generate_patient_population(population_size)
        
        # Initialize PK/PD models
        self._initialize_pkpd_models()
        
        # Initialize statistical power calculator
        self._initialize_statistical_power()
        
        self.initialized = True
        print(f"Virtual Trial Simulator initialized with {population_size} virtual patients")
    
    def _load_disease_models(self, disease_type=None):
        """Load disease progression models"""
        # Default disease models
        self.disease_models = {
            'hypertension': {
                'baseline_sbp': lambda age: 120 + 0.5 * (age - 50),  # mmHg
                'baseline_dbp': lambda age: 80 + 0.3 * (age - 50),   # mmHg
                'progression_rate': 0.01,  # mmHg per day
                'variability': 5.0,  # mmHg standard deviation
                'comorbidities': ['obesity', 'diabetes', 'kidney_disease']
            },
            'diabetes_type2': {
                'baseline_hba1c': lambda age: 6.5 + 0.02 * (age - 50),  # %
                'baseline_fasting_glucose': lambda age: 110 + 0.5 * (age - 50),  # mg/dL
                'progression_rate': 0.001,  % per day
                'variability': 0.5,
                'comorbidities': ['hypertension', 'obesity', 'cardiovascular_disease']
            },
            'arthritis': {
                'baseline_pain_score': lambda age: 4.0 + 0.01 * (age - 50),  # 0-10 scale
                'baseline_inflammation': lambda age: 2.0 + 0.005 * (age - 50),  # CRP mg/L
                'progression_rate': 0.002,
                'variability': 0.3,
                'comorbidities': ['obesity', 'osteoporosis']
            }
        }
        
        if disease_type and disease_type in self.disease_models:
            self.current_disease = disease_type
        else:
            self.current_disease = list(self.disease_models.keys())[0]
    
    def _generate_patient_population(self, population_size=1000):
        """Generate virtual patient population"""
        print(f"Generating {population_size} virtual patients...")
        
        disease_model = self.disease_models[self.current_disease]
        
        for i in range(population_size):
            # Generate patient demographics
            age = int(np.random.normal(55, 15))
            age = max(18, min(85, age))
            
            gender = np.random.choice(['male', 'female'], p=[0.48, 0.52])
            weight = np.random.normal(75 if gender == 'male' else 65, 15)
            height = np.random.normal(175 if gender == 'male' else 165, 10)
            bmi = weight / ((height/100) ** 2)
            
            # Generate genotype (simplified)
            genotype = {
                'CYP2C9': np.random.choice(['*1/*1', '*1/*2', '*2/*2'], p=[0.7, 0.25, 0.05]),
                'CYP2D6': np.random.choice(['EM', 'IM', 'PM'], p=[0.7, 0.25, 0.05]),
                'VKORC1': np.random.choice(['GG', 'GA', 'AA'], p=[0.4, 0.4, 0.2]),
                'TPMT': np.random.choice(['*1/*1', '*1/*3', '*3/*3'], p=[0.86, 0.13, 0.01])
            }
            
            # Generate phenotype based on disease
            phenotype = {}
            for param, func in disease_model.items():
                if callable(func):
                    baseline = func(age)
                    phenotype[param] = np.random.normal(baseline, disease_model['variability'])
            
            # Add physiological parameters
            phenotype.update({
                'heart_rate': np.random.normal(72, 10),
                'respiratory_rate': np.random.normal(16, 3),
                'body_temperature': np.random.normal(36.8, 0.4),
                'blood_pressure_systolic': np.random.normal(120, 15),
                'blood_pressure_diastolic': np.random.normal(80, 10)
            })
            
            # Generate comorbidities
            comorbidity_probabilities = {
                'hypertension': 0.3,
                'diabetes': 0.2,
                'obesity': 0.25,
                'cardiovascular_disease': 0.15,
                'kidney_disease': 0.1,
                'liver_disease': 0.05
            }
            
            comorbidities = []
            for disease, prob in comorbidity_probabilities.items():
                if np.random.random() < prob:
                    comorbidities.append(disease)
            
            # Medication history
            common_medications = [
                'aspirin', 'metformin', 'lisinopril', 'atorvastatin',
                'metoprolol', 'losartan', 'levothyroxine', 'omeprazole'
            ]
            
            medication_history = []
            for med in common_medications:
                if np.random.random() < 0.2:
                    medication_history.append(med)
            
            # Inclusion/exclusion criteria
            inclusion_criteria = {
                'age_18_75': 18 <= age <= 75,
                'bmi_18_40': 18 <= bmi <= 40,
                'disease_severity': phenotype.get('baseline_pain_score', 3) >= 4,
                'no_pregnancy': gender == 'male' or np.random.random() > 0.1,
                'informed_consent': True
            }
            
            exclusion_criteria = {
                'severe_comorbidities': len([c for c in comorbidities if c in ['liver_disease', 'kidney_disease']]) > 0,
                'concurrent_medications': 'warfarin' in medication_history,
                'allergy_to_treatment': np.random.random() < 0.05,
                'recent_surgery': np.random.random() < 0.1
            }
            
            # Create patient profile
            patient = PatientProfile(
                patient_id=f"PAT_{i:06d}",
                age=age,
                gender=gender,
                weight=weight,
                height=height,
                bmi=bmi,
                genotype=genotype,
                phenotype=phenotype,
                comorbidities=comorbidities,
                medication_history=medication_history,
                inclusion_criteria=inclusion_criteria,
                exclusion_criteria=exclusion_criteria
            )
            
            self.patient_database.append(patient)
        
        print(f"Generated {len(self.patient_database)} virtual patients")
    
    def _initialize_pkpd_models(self):
        """Initialize pharmacokinetic/pharmacodynamic models"""
        
        class PKModel:
            """One-compartment PK model with first-order absorption"""
            def __init__(self, ka=1.0, ke=0.1, vd=1.0):
                self.ka = ka  # Absorption rate constant (1/hr)
                self.ke = ke  # Elimination rate constant (1/hr)
                self.vd = vd  # Volume of distribution (L/kg)
            
            def simulate(self, dose, dosing_interval, duration, patient_weight):
                """Simulate plasma concentration over time"""
                # Convert to patient-specific parameters
                vd_patient = self.vd * patient_weight
                
                # Time points
                times = np.arange(0, duration * 24, 0.1)  # hours
                concentrations = np.zeros_like(times)
                
                # Simulate multiple doses
                for t in times:
                    # Calculate contributions from all previous doses
                    conc = 0
                    dose_time = 0
                    while dose_time <= t:
                        # One-compartment model with first-order absorption
                        if t >= dose_time:
                            conc += (dose * self.ka / vd_patient / (self.ka - self.ke)) * \
                                   (np.exp(-self.ke * (t - dose_time)) - 
                                    np.exp(-self.ka * (t - dose_time)))
                        dose_time += dosing_interval
                    
                    concentrations[np.where(times == t)[0][0]] = conc
                
                return times, concentrations
        
        class PDModel:
            """Simple Emax model for drug effect"""
            def __init__(self, emax=1.0, ec50=10.0, hill_coefficient=1.0):
                self.emax = emax  # Maximum effect
                self.ec50 = ec50  # Concentration for 50% effect
                self.hill = hill_coefficient  # Hill coefficient
            
            def effect(self, concentration):
                """Calculate drug effect at given concentration"""
                return (self.emax * concentration**self.hill) / \
                       (self.ec50**self.hill + concentration**self.hill)
        
        # Store models
        self.pk_models['standard'] = PKModel(ka=0.8, ke=0.15, vd=0.7)
        self.pd_models['efficacy'] = PDModel(emax=0.8, ec50=5.0, hill_coefficient=1.2)
        self.pd_models['toxicity'] = PDModel(emax=1.0, ec50=20.0, hill_coefficient=1.5)
    
    def _initialize_statistical_power(self):
        """Initialize statistical power calculator"""
        self.power_calculator = {
            'alpha': 0.05,
            'beta': 0.2,
            'power': 0.8,
            'effect_size_calculator': self._calculate_effect_size,
            'sample_size_calculator': self._calculate_sample_size
        }
    
    def simulate_trial(self, compound_data: Dict, trial_config: Dict = None) -> Dict:
        """
        Simulate a complete clinical trial
        
        Args:
            compound_data: Compound information and predicted properties
            trial_config: Trial configuration (overrides defaults)
            
        Returns:
            Complete trial simulation results
        """
        if not self.initialized:
            self.initialize()
        
        print(f"Simulating virtual trial for {compound_data.get('name', 'Unknown Compound')}")
        
        # Merge trial configuration
        config = self.trial_design.copy()
        if trial_config:
            config.update(trial_config)
        
        # Generate trial ID
        trial_id = self._generate_trial_id(compound_data)
        
        # Screen and randomize patients
        screened_patients = self._screen_patients(config.get('screening_criteria', {}))
        randomized_patients = self._randomize_patients(screened_patients, config['arms'], 
                                                      config['randomization_ratio'])
        
        # Simulate treatment
        treatment_results = self._simulate_treatment(
            randomized_patients, compound_data, config
        )
        
        # Analyze results
        statistical_analysis = self._analyze_trial_results(treatment_results, config)
        
        # Generate report
        trial_report = self._generate_trial_report(
            trial_id, compound_data, config, treatment_results, statistical_analysis
        )
        
        # Store results
        self.trial_results[trial_id] = {
            'trial_id': trial_id,
            'compound_data': compound_data,
            'config': config,
            'treatment_results': treatment_results,
            'statistical_analysis': statistical_analysis,
            'report': trial_report,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to simulation history
        self.simulation_history.append(trial_id)
        
        return trial_report
    
    def _generate_trial_id(self, compound_data: Dict) -> str:
        """Generate unique trial ID"""
        compound_name = compound_data.get('name', 'unknown').replace(' ', '_').lower()
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"VT_{compound_name}_{timestamp}"
    
    def _screen_patients(self, screening_criteria: Dict) -> List[PatientProfile]:
        """Screen patients based on inclusion/exclusion criteria"""
        print(f"Screening {len(self.patient_database)} patients...")
        
        screened = []
        excluded_counts = {
            'inclusion_criteria': 0,
            'exclusion_criteria': 0,
            'other': 0
        }
        
        for patient in self.patient_database:
            # Check inclusion criteria
            inclusion_passed = all(
                patient.inclusion_criteria.get(criterion, False)
                for criterion in screening_criteria.get('inclusion', [])
            )
            
            if not inclusion_passed:
                excluded_counts['inclusion_criteria'] += 1
                continue
            
            # Check exclusion criteria
            exclusion_failed = any(
                patient.exclusion_criteria.get(criterion, False)
                for criterion in screening_criteria.get('exclusion', [])
            )
            
            if exclusion_failed:
                excluded_counts['exclusion_criteria'] += 1
                continue
            
            # Additional screening
            if self._additional_screening(patient, screening_criteria):
                screened.append(patient)
            else:
                excluded_counts['other'] += 1
        
        print(f"Screened: {len(screened)} patients")
        print(f"Excluded: {excluded_counts}")
        
        return screened
    
    def _additional_screening(self, patient: PatientProfile, criteria: Dict) -> bool:
        """Perform additional patient screening"""
        # Screen based on disease severity
        if 'disease_severity_min' in criteria:
            min_severity = criteria['disease_severity_min']
            # Assuming disease severity is in phenotype
            severity = patient.phenotype.get('baseline_pain_score', 0)
            if severity < min_severity:
                return False
        
        # Screen based on comorbidities
        if 'allowed_comorbidities' in criteria:
            allowed = criteria['allowed_comorbidities']
            for comorbidity in patient.comorbidities:
                if comorbidity not in allowed:
                    return False
        
        # Screen based on concomitant medications
        if 'prohibited_medications' in criteria:
            prohibited = criteria['prohibited_medications']
            for medication in patient.medication_history:
                if medication in prohibited:
                    return False
        
        return True
    
    def _randomize_patients(self, patients: List[PatientProfile], 
                          arms: List[str], ratios: List[int]) -> Dict[str, List[PatientProfile]]:
        """Randomize patients to trial arms"""
        print(f"Randomizing {len(patients)} patients to {len(arms)} arms...")
        
        # Convert ratios to probabilities
        total = sum(ratios)
        probabilities = [r/total for r in ratios]
        
        # Shuffle patients
        shuffled_patients = patients.copy()
        np.random.shuffle(shuffled_patients)
        
        # Assign to arms
        randomized = {arm: [] for arm in arms}
        current_idx = 0
        
        for i, arm in enumerate(arms):
            n_arm = int(len(shuffled_patients) * probabilities[i])
            randomized[arm] = shuffled_patients[current_idx:current_idx + n_arm]
            current_idx += n_arm
        
        # Assign remaining patients
        remaining = shuffled_patients[current_idx:]
        for patient in remaining:
            arm = np.random.choice(arms, p=probabilities)
            randomized[arm].append(patient)
        
        # Print allocation
        for arm in arms:
            print(f"  {arm}: {len(randomized[arm])} patients")
        
        return randomized
    
    def _simulate_treatment(self, randomized_patients: Dict[str, List[PatientProfile]],
                          compound_data: Dict, config: Dict) -> Dict:
        """Simulate treatment for all patients"""
        print("Simulating treatment...")
        
        treatment_results = {
            'arms': {},
            'patient_data': [],
            'visit_data': [],
            'adverse_events': [],
            'dropouts': []
        }
        
        # Treatment parameters from compound data
        treatment_params = {
            'dose': compound_data.get('recommended_dose', 100),  # mg
            'dosing_interval': 24,  # hours
            'duration': config['duration_days']
        }
        
        # Simulate each arm
        for arm, patients in randomized_patients.items():
            arm_results = {
                'patients': [],
                'efficacy_over_time': [],
                'safety_over_time': [],
                'summary': {}
            }
            
            for patient in patients:
                # Simulate patient journey
                patient_result = self._simulate_patient_journey(
                    patient, arm, treatment_params, config
                )
                
                arm_results['patients'].append(patient_result)
                
                # Collect visit data
                for visit in patient_result.get('visits', []):
                    treatment_results['visit_data'].append({
                        'patient_id': patient.patient_id,
                        'arm': arm,
                        'visit_day': visit['day'],
                        'efficacy': visit.get('efficacy', 0),
                        'safety': visit.get('safety', 0),
                        'adverse_events': visit.get('adverse_events', []),
                        'compliance': visit.get('compliance', 1.0)
                    })
                
                # Record adverse events
                for ae in patient_result.get('adverse_events', []):
                    treatment_results['adverse_events'].append({
                        'patient_id': patient.patient_id,
                        'arm': arm,
                        'event': ae['event'],
                        'severity': ae['severity'],
                        'day': ae['day'],
                        'related_to_treatment': ae.get('related', True)
                    })
                
                # Record dropouts
                if patient_result.get('status') == PatientStatus.DROPPED_OUT:
                    treatment_results['dropouts'].append({
                        'patient_id': patient.patient_id,
                        'arm': arm,
                        'day': patient_result.get('dropout_day', 0),
                        'reason': patient_result.get('dropout_reason', 'Unknown')
                    })
            
            # Calculate arm-level summaries
            arm_results['summary'] = self._summarize_arm_results(arm_results['patients'])
            treatment_results['arms'][arm] = arm_results
        
        return treatment_results
    
    def _simulate_patient_journey(self, patient: PatientProfile, arm: str,
                                treatment_params: Dict, config: Dict) -> Dict:
        """Simulate individual patient journey through trial"""
        
        # Initialize patient result
        patient_result = {
            'patient_id': patient.patient_id,
            'arm': arm,
            'status': PatientStatus.RANDOMIZED,
            'visits': [],
            'adverse_events': [],
            'pk_profile': {},
            'compliance': self._simulate_compliance(),
            'genotype_effects': self._simulate_genotype_effects(patient.genotype)
        }
        
        # Get baseline measurements
        baseline = self._get_baseline_measurements(patient)
        
        # Determine treatment effect based on arm
        if arm == 'treatment':
            treatment_effect = self._simulate_treatment_effect(patient, treatment_params)
            placebo_effect = 0.0
        elif arm == 'control':
            treatment_effect = 0.0  # Active control
            placebo_effect = 0.0
        else:  # placebo
            treatment_effect = 0.0
            placebo_effect = self._simulate_placebo_effect(patient)
        
        # Simulate visits
        dropout_day = None
        dropout_reason = None
        
        for visit_day in config['visits']:
            if dropout_day is not None and visit_day > dropout_day:
                break
            
            # Simulate visit outcomes
            visit_result = self._simulate_visit(
                patient, visit_day, baseline, treatment_effect, placebo_effect,
                treatment_params, arm
            )
            
            patient_result['visits'].append(visit_result)
            
            # Check for dropout
            if dropout_day is None and self._check_for_dropout(visit_result, visit_day):
                dropout_day = visit_day
                dropout_reason = self._determine_dropout_reason(visit_result)
                patient_result['status'] = PatientStatus.DROPPED_OUT
                patient_result['dropout_day'] = dropout_day
                patient_result['dropout_reason'] = dropout_reason
        
        # If completed all visits
        if dropout_day is None:
            patient_result['status'] = PatientStatus.COMPLETED
        
        # Calculate final outcomes
        if patient_result['visits']:
            last_visit = patient_result['visits'][-1]
            patient_result['final_efficacy'] = last_visit.get('efficacy', 0)
            patient_result['final_safety'] = last_visit.get('safety', 0)
            patient_result['response_category'] = self._categorize_response(
                baseline.get('efficacy', 0),
                patient_result['final_efficacy']
            )
        
        return patient_result
    
    def _simulate_compliance(self) -> float:
        """Simulate patient compliance"""
        # Normal distribution around 90% compliance
        compliance = np.random.normal(0.9, 0.1)
        return max(0.5, min(1.0, compliance))
    
    def _simulate_genotype_effects(self, genotype: Dict) -> Dict:
        """Simulate effects of genetic polymorphisms"""
        effects = {}
        
        # CYP2C9 effects (affects metabolism of many drugs)
        if genotype['CYP2C9'] == '*2/*2':
            effects['metabolism_rate'] = 0.5  # Poor metabolizer
        elif genotype['CYP2C9'] == '*1/*2':
            effects['metabolism_rate'] = 0.75  # Intermediate metabolizer
        else:
            effects['metabolism_rate'] = 1.0  # Extensive metabolizer
        
        # VKORC1 effects (affects warfarin sensitivity)
        if genotype['VKORC1'] == 'AA':
            effects['drug_sensitivity'] = 1.5  # More sensitive
        elif genotype['VKORC1'] == 'GA':
            effects['drug_sensitivity'] = 1.2  # Moderately sensitive
        else:
            effects['drug_sensitivity'] = 1.0  # Normal sensitivity
        
        return effects
    
    def _get_baseline_measurements(self, patient: PatientProfile) -> Dict:
        """Get baseline measurements for patient"""
        disease_model = self.disease_models[self.current_disease]
        
        baseline = {
            'efficacy': patient.phenotype.get('baseline_pain_score', 4.0),
            'safety': 0.0,  # Baseline safety score (0 = no issues)
            'quality_of_life': np.random.uniform(0.6, 0.9),
            'biomarkers': {}
        }
        
        # Add disease-specific biomarkers
        for param in disease_model:
            if param.startswith('baseline_'):
                biomarker_name = param.replace('baseline_', '')
                baseline['biomarkers'][biomarker_name] = patient.phenotype.get(param, 0)
        
        return baseline
    
    def _simulate_treatment_effect(self, patient: PatientProfile, 
                                 treatment_params: Dict) -> Dict:
        """Simulate treatment effect for a patient"""
        
        # Get PK profile
        pk_model = self.pk_models['standard']
        times, concentrations = pk_model.simulate(
            dose=treatment_params['dose'],
            dosing_interval=treatment_params['dosing_interval'],
            duration=treatment_params['duration'],
            patient_weight=patient.weight
        )
        
        # Adjust for genotype effects
        genotype_effects = self._simulate_genotype_effects(patient.genotype)
        concentrations = concentrations * genotype_effects.get('metabolism_rate', 1.0)
        
        # Calculate PD effects
        pd_model_efficacy = self.pd_models['efficacy']
        pd_model_toxicity = self.pd_models['toxicity']
        
        # Calculate average effects over treatment period
        efficacy_effects = []
        toxicity_effects = []
        
        for conc in concentrations:
            efficacy_effects.append(pd_model_efficacy.effect(conc))
            toxicity_effects.append(pd_model_toxicity.effect(conc))
        
        avg_efficacy_effect = np.mean(efficacy_effects) if efficacy_effects else 0
        avg_toxicity_effect = np.mean(toxicity_effects) if toxicity_effects else 0
        
        # Adjust for patient factors
        patient_factor = self._calculate_patient_factor(patient)
        efficacy_effect = avg_efficacy_effect * patient_factor
        
        return {
            'efficacy': efficacy_effect,
            'toxicity': avg_toxicity_effect,
            'concentration_profile': {
                'times': times.tolist(),
                'concentrations': concentrations.tolist()
            },
            'patient_factor': patient_factor
        }
    
    def _calculate_patient_factor(self, patient: PatientProfile) -> float:
        """Calculate patient-specific response factor"""
        factor = 1.0
        
        # Age effect (younger patients may respond better)
        if patient.age < 40:
            factor *= 1.1
        elif patient.age > 70:
            factor *= 0.9
        
        # BMI effect
        if patient.bmi < 18.5:
            factor *= 0.9  # Underweight
        elif patient.bmi > 30:
            factor *= 1.05  # Obesity may enhance some drug effects
        
        # Comorbidity effects
        comorbidity_effects = {
            'kidney_disease': 0.8,
            'liver_disease': 0.7,
            'diabetes': 0.95,
            'hypertension': 1.0
        }
        
        for comorbidity in patient.comorbidities:
            if comorbidity in comorbidity_effects:
                factor *= comorbidity_effects[comorbidity]
        
        # Medication interactions
        interacting_meds = ['rifampin', 'phenytoin', 'carbamazepine']
        for med in patient.medication_history:
            if med in interacting_meds:
                factor *= 0.8  # Reduce effect due to interaction
        
        return factor
    
    def _simulate_placebo_effect(self, patient: PatientProfile) -> float:
        """Simulate placebo effect"""
        # Placebo effect varies by patient and condition
        base_effect = np.random.normal(0.3, 0.1)  # 30% average placebo effect
        
        # Adjust based on patient factors
        if patient.age > 65:
            base_effect *= 0.9  # Older patients may have reduced placebo effect
        
        # Condition-specific placebo effects
        condition_effects = {
            'pain': 0.35,
            'depression': 0.4,
            'anxiety': 0.45,
            'hypertension': 0.2
        }
        
        # Use current disease or default
        disease_effect = condition_effects.get(self.current_disease, 0.3)
        
        return base_effect * disease_effect
    
    def _simulate_visit(self, patient: PatientProfile, visit_day: int,
                       baseline: Dict, treatment_effect: Dict, 
                       placebo_effect: float, treatment_params: Dict,
                       arm: str) -> Dict:
        """Simulate outcomes at a specific visit"""
        
        # Calculate time-dependent effects
        time_factor = min(1.0, visit_day / 30.0)  # Effects increase over first month
        
        # Calculate efficacy
        baseline_efficacy = baseline.get('efficacy', 0)
        
        if arm == 'treatment':
            efficacy_improvement = treatment_effect['efficacy'] * time_factor
            # Add natural progression of disease
            disease_progression = self._simulate_disease_progression(visit_day)
            final_efficacy = baseline_efficacy - efficacy_improvement + disease_progression
        elif arm == 'placebo':
            placebo_improvement = placebo_effect * time_factor
            disease_progression = self._simulate_disease_progression(visit_day)
            final_efficacy = baseline_efficacy - placebo_improvement + disease_progression
        else:  # control
            # Active control has fixed effect
            control_effect = 0.4 * time_factor  # Assume 40% effect for control
            disease_progression = self._simulate_disease_progression(visit_day)
            final_efficacy = baseline_efficacy - control_effect + disease_progression
        
        # Ensure efficacy stays in reasonable range
        final_efficacy = max(0.0, min(10.0, final_efficacy))
        
        # Calculate safety score
        if arm == 'treatment':
            safety_score = treatment_effect.get('toxicity', 0) * time_factor
            # Add random safety events
            safety_score += np.random.exponential(0.1)
        else:
            safety_score = np.random.exponential(0.05)  # Lower for non-treatment
        
        # Simulate adverse events
        adverse_events = self._simulate_adverse_events_at_visit(
            patient, visit_day, safety_score, arm
        )
        
        # Quality of life improvement
        qol_improvement = 0.0
        if arm == 'treatment':
            # QoL improves with efficacy improvement
            qol_improvement = efficacy_improvement * 0.1  # Scale factor
        elif arm == 'placebo':
            qol_improvement = placebo_effect * 0.08
        
        final_qol = baseline.get('quality_of_life', 0.7) + qol_improvement
        final_qol = max(0.0, min(1.0, final_qol))
        
        # Biomarker changes
        biomarker_changes = {}
        for biomarker, baseline_value in baseline.get('biomarkers', {}).items():
            if arm == 'treatment':
                # Biomarkers improve with treatment
                improvement = treatment_effect['efficacy'] * 0.5  # Scale factor
                biomarker_changes[biomarker] = baseline_value - improvement
            else:
                # Small random changes for control/placebo
                change = np.random.normal(0, baseline_value * 0.05)
                biomarker_changes[biomarker] = baseline_value + change
        
        return {
            'day': visit_day,
            'efficacy': float(final_efficacy),
            'safety': float(safety_score),
            'quality_of_life': float(final_qol),
            'biomarker_changes': biomarker_changes,
            'adverse_events': adverse_events,
            'compliance': self._simulate_compliance()
        }
    
    def _simulate_disease_progression(self, days: int) -> float:
        """Simulate natural disease progression"""
        disease_model = self.disease_models[self.current_disease]
        progression_rate = disease_model.get('progression_rate', 0.001)
        
        # Linear progression
        progression = progression_rate * days
        
        # Add random variation
        variation = np.random.normal(0, disease_model.get('variability', 0.1) * 0.1)
        
        return progression + variation
    
    def _simulate_adverse_events_at_visit(self, patient: PatientProfile,
                                        visit_day: int, safety_score: float,
                                        arm: str) -> List[Dict]:
        """Simulate adverse events at a specific visit"""
        adverse_events = []
        
        # Base probability of adverse event
        base_probability = 0.05
        
        # Increase probability with safety score
        ae_probability = base_probability + safety_score * 0.2
        
        # Treatment arm has higher probability
        if arm == 'treatment':
            ae_probability *= 1.5
        
        # Patient factors affecting AE probability
        if 'kidney_disease' in patient.comorbidities:
            ae_probability *= 1.3
        if 'liver_disease' in patient.comorbidities:
            ae_probability *= 1.4
        if patient.age > 70:
            ae_probability *= 1.2
        
        # Check for adverse events
        if np.random.random() < ae_probability:
            # Select adverse event type
            ae_types = [
                ('headache', 'mild', 0.3),
                ('nausea', 'mild', 0.25),
                ('dizziness', 'mild', 0.2),
                ('rash', 'moderate', 0.1),
                ('elevated_liver_enzymes', 'moderate', 0.08),
                ('hypertension', 'moderate', 0.05),
                ('severe_allergic_reaction', 'severe', 0.02)
            ]
            
            # Weighted random selection
            ae_names, ae_severities, ae_probs = zip(*ae_types)
            ae_probs = np.array(ae_probs)
            ae_probs = ae_probs / ae_probs.sum()  # Normalize
            
            idx = np.random.choice(len(ae_names), p=ae_probs)
            
            adverse_events.append({
                'event': ae_names[idx],
                'severity': ae_severities[idx],
                'day': visit_day,
                'related': arm == 'treatment'  # Assume related if in treatment arm
            })
        
        return adverse_events
    
    def _check_for_dropout(self, visit_result: Dict, visit_day: int) -> bool:
        """Check if patient should drop out at this visit"""
        dropout_probability = 0.0
        
        # High adverse events increase dropout probability
        severe_ae_count = sum(1 for ae in visit_result.get('adverse_events', [])
                            if ae['severity'] in ['severe', 'moderate'])
        
        dropout_probability += severe_ae_count * 0.3
        
        # Poor efficacy may lead to dropout
        if visit_result.get('efficacy', 0) > 8.0:  # High pain score = poor efficacy
            dropout_probability += 0.2
        
        # Non-compliance
        if visit_result.get('compliance', 1.0) < 0.7:
            dropout_probability += 0.15
        
        # Random life events
        dropout_probability += 0.02  # Base rate
        
        # Check dropout
        return np.random.random() < dropout_probability
    
    def _determine_dropout_reason(self, visit_result: Dict) -> str:
        """Determine reason for dropout"""
        # Check for adverse events first
        adverse_events = visit_result.get('adverse_events', [])
        if adverse_events:
            severe_events = [ae for ae in adverse_events if ae['severity'] == 'severe']
            if severe_events:
                return f"Severe adverse event: {severe_events[0]['event']}"
            moderate_events = [ae for ae in adverse_events if ae['severity'] == 'moderate']
            if moderate_events:
                return f"Moderate adverse event: {moderate_events[0]['event']}"
        
        # Check efficacy
        if visit_result.get('efficacy', 0) > 8.0:
            return "Lack of efficacy"
        
        # Check compliance
        if visit_result.get('compliance', 1.0) < 0.7:
            return "Poor compliance"
        
        # Default reasons
        reasons = [
            "Withdrew consent",
            "Lost to follow-up",
            "Protocol violation",
            "Intercurrent illness",
            "Personal reasons"
        ]
        
        return np.random.choice(reasons)
    
    def _categorize_response(self, baseline_efficacy: float, final_efficacy: float) -> str:
        """Categorize patient response"""
        improvement = baseline_efficacy - final_efficacy
        
        if improvement >= 5.0:
            return "Complete Response"
        elif improvement >= 3.0:
            return "Partial Response"
        elif improvement >= 1.0:
            return "Minimal Response"
        else:
            return "No Response"
    
    def _summarize_arm_results(self, patient_results: List[Dict]) -> Dict:
        """Summarize results for a treatment arm"""
        if not patient_results:
            return {}
        
        # Extract final outcomes
        final_efficacies = []
        final_safeties = []
        response_categories = []
        completion_statuses = []
        adverse_event_counts = []
        
        for result in patient_results:
            final_efficacies.append(result.get('final_efficacy', 0))
            final_safeties.append(result.get('final_safety', 0))
            response_categories.append(result.get('response_category', 'No Response'))
            completion_statuses.append(result.get('status', PatientStatus.DROPPED_OUT))
            
            # Count adverse events
            ae_count = 0
            for visit in result.get('visits', []):
                ae_count += len(visit.get('adverse_events', []))
            adverse_event_counts.append(ae_count)
        
        # Calculate statistics
        n_patients = len(patient_results)
        n_completed = sum(1 for status in completion_statuses 
                         if status == PatientStatus.COMPLETED)
        completion_rate = n_completed / n_patients if n_patients > 0 else 0
        
        # Response rates
        response_counts = {}
        for category in response_categories:
            response_counts[category] = response_counts.get(category, 0) + 1
        
        response_rates = {
            category: count / n_patients 
            for category, count in response_counts.items()
        }
        
        return {
            'n_patients': n_patients,
            'completion_rate': completion_rate,
            'mean_efficacy': np.mean(final_efficacies) if final_efficacies else 0,
            'std_efficacy': np.std(final_efficacies) if final_efficacies else 0,
            'mean_safety': np.mean(final_safeties) if final_safeties else 0,
            'mean_adverse_events': np.mean(adverse_event_counts) if adverse_event_counts else 0,
            'response_rates': response_rates,
            'efficacy_95ci': self._calculate_confidence_interval(final_efficacies),
            'safety_95ci': self._calculate_confidence_interval(final_safeties)
        }
    
    def _calculate_confidence_interval(self, data: List[float], confidence: float = 0.95) -> Dict:
        """Calculate confidence interval for data"""
        if not data or len(data) < 2:
            return {'lower': 0, 'upper': 0, 'mean': 0}
        
        from scipy import stats
        
        data_array = np.array(data)
        mean = np.mean(data_array)
        sem = stats.sem(data_array)
        
        # Calculate confidence interval
        ci = stats.t.interval(confidence, len(data_array)-1, loc=mean, scale=sem)
        
        return {
            'lower': float(ci[0]),
            'upper': float(ci[1]),
            'mean': float(mean)
        }
    
    def _analyze_trial_results(self, treatment_results: Dict, config: Dict) -> Dict:
        """Perform statistical analysis of trial results"""
        print("Analyzing trial results...")
        
        analysis = {
            'primary_endpoint': {},
            'secondary_endpoints': {},
            'safety_analysis': {},
            'subgroup_analyses': {},
            'power_analysis': {},
            'conclusions': []
        }
        
        # Primary endpoint analysis
        primary_endpoint = config['primary_endpoint']
        analysis['primary_endpoint'] = self._analyze_endpoint(
            treatment_results, primary_endpoint, config
        )
        
        # Secondary endpoints
        for endpoint in config['secondary_endpoints']:
            analysis['secondary_endpoints'][endpoint] = self._analyze_endpoint(
                treatment_results, endpoint, config
            )
        
        # Safety analysis
        analysis['safety_analysis'] = self._analyze_safety(treatment_results)
        
        # Subgroup analyses
        analysis['subgroup_analyses'] = self._analyze_subgroups(treatment_results)
        
        # Power analysis
        analysis['power_analysis'] = self._analyze_statistical_power(
            treatment_results, config
        )
        
        # Generate conclusions
        analysis['conclusions'] = self._generate_conclusions(analysis)
        
        return analysis
    
    def _analyze_endpoint(self, treatment_results: Dict, endpoint: str, 
                         config: Dict) -> Dict:
        """Analyze a specific endpoint"""
        
        # Extract data by arm
        arm_data = {}
        for arm, arm_results in treatment_results['arms'].items():
            data = []
            for patient_result in arm_results['patients']:
                if patient_result.get('status') == PatientStatus.COMPLETED:
                    # Get endpoint value from last visit
                    last_visit = patient_result.get('visits', [])[-1] if patient_result.get('visits') else {}
                    value = last_visit.get(endpoint, 0)
                    data.append(value)
            
            if data:
                arm_data[arm] = {
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'n': len(data),
                    'data': data
                }
        
        # Perform statistical tests
        statistical_tests = {}
        
        if 'treatment' in arm_data and 'placebo' in arm_data:
            # Compare treatment vs placebo
            from scipy import stats
            
            treatment_data = arm_data['treatment']['data']
            placebo_data = arm_data['placebo']['data']
            
            # T-test
            t_stat, p_value = stats.ttest_ind(treatment_data, placebo_data, 
                                             equal_var=False)
            
            statistical_tests['treatment_vs_placebo'] = {
                'test': 'independent_t_test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size': self._calculate_effect_size(treatment_data, placebo_data)
            }
        
        if 'treatment' in arm_data and 'control' in arm_data:
            # Compare treatment vs active control
            treatment_data = arm_data['treatment']['data']
            control_data = arm_data['control']['data']
            
            t_stat, p_value = stats.ttest_ind(treatment_data, control_data,
                                             equal_var=False)
            
            statistical_tests['treatment_vs_control'] = {
                'test': 'independent_t_test',
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'effect_size': self._calculate_effect_size(treatment_data, control_data)
            }
        
        return {
            'arm_statistics': arm_data,
            'statistical_tests': statistical_tests,
            'interpretation': self._interpret_endpoint_results(endpoint, arm_data, statistical_tests)
        }
    
    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if not group1 or not group2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _interpret_endpoint_results(self, endpoint: str, arm_data: Dict, 
                                   statistical_tests: Dict) -> str:
        """Interpret endpoint analysis results"""
        
        if 'treatment_vs_placebo' not in statistical_tests:
            return "Insufficient data for analysis"
        
        test_result = statistical_tests['treatment_vs_placebo']
        p_value = test_result['p_value']
        effect_size = test_result['effect_size']
        
        if p_value < 0.05:
            if effect_size > 0.8:
                magnitude = "large"
            elif effect_size > 0.5:
                magnitude = "medium"
            else:
                magnitude = "small"
            
            return (f"Statistically significant difference (p={p_value:.4f}) with "
                   f"{magnitude} effect size (d={effect_size:.2f})")
        else:
            return (f"No statistically significant difference (p={p_value:.4f}), "
                   f"effect size d={effect_size:.2f}")
    
    def _analyze_safety(self, treatment_results: Dict) -> Dict:
        """Analyze safety data"""
        safety_data = {
            'adverse_events_by_arm': {},
            'serious_adverse_events': {},
            'dropout_rates': {},
            'safety_signals': []
        }
        
        # Analyze by arm
        for arm, arm_results in treatment_results['arms'].items():
            # Adverse events
            ae_counts = {}
            sae_counts = {}
            
            for patient_result in arm_results['patients']:
                for visit in patient_result.get('visits', []):
                    for ae in visit.get('adverse_events', []):
                        event_name = ae['event']
                        severity = ae['severity']
                        
                        # Count adverse events
                        ae_counts[event_name] = ae_counts.get(event_name, 0) + 1
                        
                        # Count serious adverse events
                        if severity == 'severe':
                            sae_counts[event_name] = sae_counts.get(event_name, 0) + 1
            
            safety_data['adverse_events_by_arm'][arm] = {
                'total_patients': len(arm_results['patients']),
                'events': ae_counts,
                'event_rates': {event: count/len(arm_results['patients']) 
                               for event, count in ae_counts.items()}
            }
            
            safety_data['serious_adverse_events'][arm] = sae_counts
            
            # Dropout rates
            dropouts = sum(1 for p in arm_results['patients'] 
                          if p.get('status') == PatientStatus.DROPPED_OUT)
            safety_data['dropout_rates'][arm] = dropouts / len(arm_results['patients'])
        
        # Identify safety signals
        safety_data['safety_signals'] = self._identify_safety_signals(safety_data)
        
        return safety_data
    
    def _identify_safety_signals(self, safety_data: Dict) -> List[Dict]:
        """Identify potential safety signals"""
        signals = []
        
        # Compare adverse event rates between treatment and placebo
        if 'treatment' in safety_data['adverse_events_by_arm'] and \
           'placebo' in safety_data['adverse_events_by_arm']:
            
            treatment_rates = safety_data['adverse_events_by_arm']['treatment']['event_rates']
            placebo_rates = safety_data['adverse_events_by_arm']['placebo']['event_rates']
            
            for event, treatment_rate in treatment_rates.items():
                placebo_rate = placebo_rates.get(event, 0)
                
                # Calculate risk ratio
                if placebo_rate > 0:
                    risk_ratio = treatment_rate / placebo_rate
                    
                    # Flag if risk ratio > 2
                    if risk_ratio > 2.0 and treatment_rate > 0.05:
                        signals.append({
                            'event': event,
                            'treatment_rate': treatment_rate,
                            'placebo_rate': placebo_rate,
                            'risk_ratio': risk_ratio,
                            'signal_strength': 'strong' if risk_ratio > 3 else 'moderate'
                        })
        
        return signals
    
    def _analyze_subgroups(self, treatment_results: Dict) -> Dict:
        """Analyze treatment effects in subgroups"""
        subgroups = {
            'by_age': {'<50': [], '50-65': [], '>65': []},
            'by_gender': {'male': [], 'female': []},
            'by_genotype': {},
            'by_comorbidity': {}
        }
        
        # Collect data for subgroups
        for arm, arm_results in treatment_results['arms'].items():
            for patient_result in arm_results['patients']:
                # Get patient from database
                patient = next((p for p in self.patient_database 
                              if p.patient_id == patient_result['patient_id']), None)
                
                if not patient:
                    continue
                
                # Age subgroups
                if patient.age < 50:
                    subgroups['by_age']['<50'].append(patient_result)
                elif patient.age <= 65:
                    subgroups['by_age']['50-65'].append(patient_result)
                else:
                    subgroups['by_age']['>65'].append(patient_result)
                
                # Gender subgroups
                subgroups['by_gender'][patient.gender].append(patient_result)
                
                # Genotype subgroups
                genotype_key = f"{patient.genotype['CYP2C9']}_{patient.genotype['CYP2D6']}"
                if genotype_key not in subgroups['by_genotype']:
                    subgroups['by_genotype'][genotype_key] = []
                subgroups['by_genotype'][genotype_key].append(patient_result)
                
                # Comorbidity subgroups
                for comorbidity in patient.comorbidities:
                    if comorbidity not in subgroups['by_comorbidity']:
                        subgroups['by_comorbidity'][comorbidity] = []
                    subgroups['by_comorbidity'][comorbidity].append(patient_result)
        
        # Analyze each subgroup
        subgroup_analyses = {}
        
        for subgroup_name, subgroup_data in subgroups.items():
            subgroup_analyses[subgroup_name] = {}
            
            for subgroup_key, patient_results in subgroup_data.items():
                if patient_results:
                    # Calculate response rate in this subgroup
                    responses = [r.get('response_category', 'No Response') 
                               for r in patient_results]
                    response_rate = sum(1 for r in responses 
                                      if r in ['Complete Response', 'Partial Response']) / len(responses)
                    
                    subgroup_analyses[subgroup_name][subgroup_key] = {
                        'n_patients': len(patient_results),
                        'response_rate': response_rate,
                        'mean_efficacy': np.mean([r.get('final_efficacy', 0) 
                                                for r in patient_results]) if patient_results else 0
                    }
        
        return subgroup_analyses
    
    def _analyze_statistical_power(self, treatment_results: Dict, config: Dict) -> Dict:
        """Analyze statistical power of the trial"""
        
        # Calculate achieved power
        if 'treatment' in treatment_results['arms'] and 'placebo' in treatment_results['arms']:
            treatment_data = []
            placebo_data = []
            
            for arm in ['treatment', 'placebo']:
                for patient_result in treatment_results['arms'][arm]['patients']:
                    if patient_result.get('status') == PatientStatus.COMPLETED:
                        last_visit = patient_result.get('visits', [])[-1]
                        efficacy = last_visit.get('efficacy', 0)
                        
                        if arm == 'treatment':
                            treatment_data.append(efficacy)
                        else:
                            placebo_data.append(efficacy)
            
            if treatment_data and placebo_data:
                effect_size = self._calculate_effect_size(treatment_data, placebo_data)
                n1, n2 = len(treatment_data), len(placeholder_data)
                
                # Calculate achieved power
                from statsmodels.stats.power import TTestIndPower
                power_analysis = TTestIndPower()
                achieved_power = power_analysis.solve_power(
                    effect_size=effect_size,
                    nobs1=n1,
                    alpha=0.05,
                    ratio=n2/n1 if n1 > 0 else 1
                )
                
                return {
                    'effect_size': float(effect_size),
                    'sample_sizes': {'treatment': n1, 'placebo': n2},
                    'achieved_power': float(achieved_power) if achieved_power else 0,
                    'adequate_power': achieved_power >= 0.8 if achieved_power else False,
                    'required_sample_size': self._calculate_required_sample_size(effect_size)
                }
        
        return {
            'effect_size': 0,
            'sample_sizes': {},
            'achieved_power': 0,
            'adequate_power': False,
            'required_sample_size': {'per_group': 0, 'total': 0}
        }
    
    def _calculate_required_sample_size(self, effect_size: float, 
                                       power: float = 0.8, alpha: float = 0.05) -> Dict:
        """Calculate required sample size for given effect size"""
        from statsmodels.stats.power import TTestIndPower
        
        power_analysis = TTestIndPower()
        
        if effect_size <= 0:
            return {'per_group': 0, 'total': 0}
        
        try:
            n_per_group = power_analysis.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=alpha,
                ratio=1.0
            )
            
            if n_per_group:
                return {
                    'per_group': int(np.ceil(n_per_group)),
                    'total': int(np.ceil(n_per_group * 2))
                }
        except:
            pass
        
        return {'per_group': 0, 'total': 0}
    
    def _generate_conclusions(self, analysis: Dict) -> List[str]:
        """Generate conclusions from trial analysis"""
        conclusions = []
        
        # Primary endpoint conclusion
        primary = analysis['primary_endpoint']
        if 'treatment_vs_placebo' in primary.get('statistical_tests', {}):
            test = primary['statistical_tests']['treatment_vs_placebo']
            
            if test.get('significant', False):
                conclusions.append(
                    f"Primary endpoint met: Treatment showed statistically significant "
                    f"improvement over placebo (p={test['p_value']:.4f}, "
                    f"effect size d={test['effect_size']:.2f})"
                )
            else:
                conclusions.append(
                    f"Primary endpoint not met: No statistically significant difference "
                    f"between treatment and placebo (p={test['p_value']:.4f})"
                )
        
        # Safety conclusions
        safety = analysis['safety_analysis']
        if safety.get('safety_signals'):
            conclusions.append(
                f"Safety signals identified: {len(safety['safety_signals'])} potential "
                f"safety concerns requiring further investigation"
            )
        else:
            conclusions.append(
                "No major safety signals identified: Treatment appears safe and well-tolerated"
            )
        
        # Power conclusion
        power = analysis['power_analysis']
        if power.get('adequate_power', False):
            conclusions.append(
                f"Trial achieved adequate statistical power ({power['achieved_power']:.2%})"
            )
        else:
            conclusions.append(
                f"Trial underpowered ({power['achieved_power']:.2%} power). "
                f"Consider larger sample size for future studies"
            )
        
        # Overall recommendation
        primary_success = any('Primary endpoint met' in c for c in conclusions)
        safety_acceptable = 'No major safety signals' in conclusions[-1]  # Last safety conclusion
        
        if primary_success and safety_acceptable:
            conclusions.append(
                "RECOMMENDATION: Proceed to next phase of clinical development"
            )
        elif primary_success and not safety_acceptable:
            conclusions.append(
                "RECOMMENDATION: Further safety evaluation required before proceeding"
            )
        elif not primary_success and safety_acceptable:
            conclusions.append(
                "RECOMMENDATION: Consider reformulation or different patient population"
            )
        else:
            conclusions.append(
                "RECOMMENDATION: Discontinue development due to lack of efficacy and safety concerns"
            )
        
        return conclusions
    
    def _generate_trial_report(self, trial_id: str, compound_data: Dict,
                             config: Dict, treatment_results: Dict,
                             statistical_analysis: Dict) -> Dict:
        """Generate comprehensive trial report"""
        
        report = {
            'trial_id': trial_id,
            'compound_name': compound_data.get('name', 'Unknown'),
            'simulation_date': datetime.now().isoformat(),
            'trial_phase': self.trial_phase.value,
            'disease': self.current_disease,
            'simulation_fidelity': self.simulation_fidelity,
            'executive_summary': self._generate_executive_summary(statistical_analysis),
            'trial_design': config,
            'patient_population': {
                'total_screened': len(self.patient_database),
                'total_randomized': sum(len(arm['patients']) 
                                      for arm in treatment_results['arms'].values()),
                'demographics': self._summarize_demographics(treatment_results)
            },
            'efficacy_results': statistical_analysis['primary_endpoint'],
            'safety_results': statistical_analysis['safety_analysis'],
            'secondary_endpoints': statistical_analysis['secondary_endpoints'],
            'subgroup_analyses': statistical_analysis['subgroup_analyses'],
            'statistical_power': statistical_analysis['power_analysis'],
            'conclusions': statistical_analysis['conclusions'],
            'recommendations': self._generate_recommendations(statistical_analysis),
            'limitations': self._acknowledge_limitations(),
            'appendices': {
                'patient_level_data': self._prepare_patient_level_data(treatment_results),
                'visit_schedule': config['visits'],
                'simulation_parameters': {
                    'random_seed': self.random_seed,
                    'fidelity': self.simulation_fidelity,
                    'disease_model': self.current_disease
                }
            }
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict) -> str:
        """Generate executive summary"""
        
        primary = analysis['primary_endpoint']
        safety = analysis['safety_analysis']
        power = analysis['power_analysis']
        conclusions = analysis['conclusions']
        
        summary_parts = []
        
        # Primary endpoint result
        if 'treatment_vs_placebo' in primary.get('statistical_tests', {}):
            test = primary['statistical_tests']['treatment_vs_placebo']
            if test.get('significant', False):
                summary_parts.append(
                    f"The trial successfully met its primary endpoint with statistical "
                    f"significance (p={test['p_value']:.4f}) and a {self._describe_effect_size(test['effect_size'])} "
                    f"effect size."
                )
            else:
                summary_parts.append(
                    f"The trial did not meet its primary endpoint (p={test['p_value']:.4f})."
                )
        
        # Safety summary
        safety_signals = safety.get('safety_signals', [])
        if safety_signals:
            summary_parts.append(
                f"{len(safety_signals)} safety signal(s) were identified requiring further investigation."
            )
        else:
            summary_parts.append(
                "The treatment was safe and well-tolerated with no major safety concerns."
            )
        
        # Power summary
        if power.get('adequate_power', False):
            summary_parts.append(
                f"The trial achieved adequate statistical power ({power['achieved_power']:.1%})."
            )
        else:
            summary_parts.append(
                f"The trial was underpowered ({power['achieved_power']:.1%} power)."
            )
        
        # Overall recommendation
        if conclusions:
            last_conclusion = conclusions[-1]
            if 'RECOMMENDATION:' in last_conclusion:
                recommendation = last_conclusion.split('RECOMMENDATION:')[1].strip()
                summary_parts.append(f"Recommendation: {recommendation}")
        
        return " ".join(summary_parts)
    
    def _describe_effect_size(self, d: float) -> str:
        """Describe effect size in words"""
        if d >= 0.8:
            return "large"
        elif d >= 0.5:
            return "medium"
        elif d >= 0.2:
            return "small"
        else:
            return "negligible"
    
    def _summarize_demographics(self, treatment_results: Dict) -> Dict:
        """Summarize patient demographics"""
        demographics = {
            'total_patients': 0,
            'by_arm': {},
            'age_distribution': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'gender_distribution': {'male': 0, 'female': 0},
            'bmi_distribution': {'mean': 0, 'std': 0}
        }
        
        all_ages = []
        all_bmis = []
        
        for arm, arm_results in treatment_results['arms'].items():
            arm_demo = {
                'n_patients': len(arm_results['patients']),
                'ages': [],
                'genders': {'male': 0, 'female': 0},
                'bmis': []
            }
            
            for patient_result in arm_results['patients']:
                patient = next((p for p in self.patient_database 
                              if p.patient_id == patient_result['patient_id']), None)
                
                if patient:
                    arm_demo['ages'].append(patient.age)
                    arm_demo['genders'][patient.gender] += 1
                    arm_demo['bmis'].append(patient.bmi)
                    
                    all_ages.append(patient.age)
                    all_bmis.append(patient.bmi)
                    
                    if patient.gender == 'male':
                        demographics['gender_distribution']['male'] += 1
                    else:
                        demographics['gender_distribution']['female'] += 1
            
            demographics['by_arm'][arm] = {
                'n_patients': arm_demo['n_patients'],
                'mean_age': np.mean(arm_demo['ages']) if arm_demo['ages'] else 0,
                'gender_ratio': arm_demo['genders'],
                'mean_bmi': np.mean(arm_demo['bmis']) if arm_demo['bmis'] else 0
            }
            
            demographics['total_patients'] += arm_demo['n_patients']
        
        if all_ages:
            demographics['age_distribution'] = {
                'mean': float(np.mean(all_ages)),
                'std': float(np.std(all_ages)),
                'min': float(np.min(all_ages)),
                'max': float(np.max(all_ages))
            }
        
        if all_bmis:
            demographics['bmi_distribution'] = {
                'mean': float(np.mean(all_bmis)),
                'std': float(np.std(all_bmis))
            }
        
        return demographics
    
    def _generate_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate recommendations based on trial results"""
        recommendations = []
        
        # Primary endpoint recommendations
        primary = analysis['primary_endpoint']
        if 'treatment_vs_placebo' in primary.get('statistical_tests', {}):
            test = primary['statistical_tests']['treatment_vs_placebo']
            
            if test.get('significant', False):
                recommendations.append({
                    'category': 'Development',
                    'recommendation': 'Proceed to next clinical phase',
                    'priority': 'High',
                    'rationale': 'Primary endpoint met with statistical significance'
                })
            else:
                recommendations.append({
                    'category': 'Development',
                    'recommendation': 'Consider reformulation or new indication',
                    'priority': 'Medium',
                    'rationale': 'Primary endpoint not met'
                })
        
        # Safety recommendations
        safety = analysis['safety_analysis']
        safety_signals = safety.get('safety_signals', [])
        
        if safety_signals:
            for signal in safety_signals[:3]:  # Top 3 signals
                recommendations.append({
                    'category': 'Safety',
                    'recommendation': f"Monitor {signal['event']} in future studies",
                    'priority': 'High' if signal['signal_strength'] == 'strong' else 'Medium',
                    'rationale': f"Elevated risk ratio: {signal['risk_ratio']:.1f}"
                })
        
        # Subgroup recommendations
        subgroups = analysis['subgroup_analyses']
        if 'by_age' in subgroups:
            age_groups = subgroups['by_age']
            
            # Find best responding age group
            best_group = max(age_groups.items(), 
                           key=lambda x: x[1].get('response_rate', 0))
            
            if best_group[1].get('response_rate', 0) > 0.5:
                recommendations.append({
                    'category': 'Target Population',
                    'recommendation': f"Focus on {best_group[0]} age group",
                    'priority': 'Medium',
                    'rationale': f"Highest response rate: {best_group[1]['response_rate']:.1%}"
                })
        
        # Statistical power recommendations
        power = analysis['power_analysis']
        if not power.get('adequate_power', False):
            required_size = power.get('required_sample_size', {}).get('total', 0)
            recommendations.append({
                'category': 'Study Design',
                'recommendation': f"Increase sample size to {required_size} in future studies",
                'priority': 'Medium',
                'rationale': f"Inadequate statistical power ({power.get('achieved_power', 0):.1%})"
            })
        
        return recommendations
    
    def _acknowledge_limitations(self) -> List[str]:
        """Acknowledge simulation limitations"""
        limitations = [
            "Virtual patients are simulated based on statistical distributions and may not capture full biological complexity",
            "Drug-drug interactions are simplified and may not reflect all real-world scenarios",
            "Placebo effects are modeled statistically and may vary in actual clinical settings",
            "Long-term effects and rare adverse events may not be fully captured in simulation",
            "Patient compliance is modeled as a constant factor rather than dynamic behavior"
        ]
        
        if self.simulation_fidelity == 'low':
            limitations.append("Low-fidelity simulation: Results should be interpreted with caution")
        elif self.simulation_fidelity == 'medium':
            limitations.append("Medium-fidelity simulation: Results provide reasonable estimates but require validation")
        else:
            limitations.append("High-fidelity simulation: Results are detailed but still require experimental confirmation")
        
        return limitations
    
    def _prepare_patient_level_data(self, treatment_results: Dict) -> List[Dict]:
        """Prepare anonymized patient-level data"""
        patient_data = []
        
        for arm, arm_results in treatment_results['arms'].items():
            for patient_result in arm_results['patients']:
                patient = next((p for p in self.patient_database 
                              if p.patient_id == patient_result['patient_id']), None)
                
                if patient:
                    # Anonymize patient ID
                    anonymized_id = f"P{hash(patient.patient_id) % 1000000:06d}"
                    
                    patient_data.append({
                        'anonymized_id': anonymized_id,
                        'arm': arm,
                        'age_group': self._get_age_group(patient.age),
                        'gender': patient.gender,
                        'bmi_category': self._get_bmi_category(patient.bmi),
                        'final_efficacy': patient_result.get('final_efficacy', 0),
                        'final_safety': patient_result.get('final_safety', 0),
                        'response_category': patient_result.get('response_category', 'No Response'),
                        'completion_status': patient_result.get('status', '').value,
                        'adverse_event_count': sum(len(v.get('adverse_events', [])) 
                                                 for v in patient_result.get('visits', []))
                    })
        
        return patient_data
    
    def _get_age_group(self, age: int) -> str:
        """Categorize age into groups"""
        if age < 40:
            return "18-39"
        elif age < 60:
            return "40-59"
        elif age < 75:
            return "60-74"
        else:
            return "75+"
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Categorize BMI"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def visualize_results(self, trial_id: str = None, save_path: str = None):
        """Visualize trial results"""
        if not trial_id and self.trial_results:
            trial_id = list(self.trial_results.keys())[-1]
        
        if trial_id not in self.trial_results:
            print(f"Trial {trial_id} not found")
            return
        
        trial_data = self.trial_results[trial_id]
        treatment_results = trial_data['treatment_results']
        statistical_analysis = trial_data['statistical_analysis']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Virtual Trial Results: {trial_data['compound_data'].get('name', 'Unknown')}", 
                    fontsize=16, fontweight='bold')
        
        # 1. Efficacy over time by arm
        ax1 = axes[0, 0]
        self._plot_efficacy_over_time(ax1, treatment_results)
        
        # 2. Response rates by arm
        ax2 = axes[0, 1]
        self._plot_response_rates(ax2, treatment_results)
        
        # 3. Adverse events by arm
        ax3 = axes[0, 2]
        self._plot_adverse_events(ax3, treatment_results)
        
        # 4. Subgroup analysis - Age
        ax4 = axes[1, 0]
        self._plot_subgroup_analysis(ax4, statistical_analysis, 'by_age')
        
        # 5. Subgroup analysis - Gender
        ax5 = axes[1, 1]
        self._plot_subgroup_analysis(ax5, statistical_analysis, 'by_gender')
        
        # 6. Power analysis
        ax6 = axes[1, 2]
        self._plot_power_analysis(ax6, statistical_analysis)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def _plot_efficacy_over_time(self, ax, treatment_results: Dict):
        """Plot efficacy over time by arm"""
        # Collect data by visit and arm
        visit_data = {}
        
        for arm, arm_results in treatment_results['arms'].items():
            visit_data[arm] = {}
            for patient_result in arm_results['patients']:
                for visit in patient_result.get('visits', []):
                    visit_day = visit['day']
                    efficacy = visit.get('efficacy', 0)
                    
                    if visit_day not in visit_data[arm]:
                        visit_data[arm][visit_day] = []
                    visit_data[arm][visit_day].append(efficacy)
        
        # Plot
        colors = {'treatment': 'blue', 'control': 'green', 'placebo': 'red'}
        
        for arm, arm_data in visit_data.items():
            days = sorted(arm_data.keys())
            means = [np.mean(arm_data[day]) for day in days]
            stds = [np.std(arm_data[day]) for day in days]
            
            ax.plot(days, means, 'o-', label=arm.capitalize(), color=colors.get(arm, 'black'))
            ax.fill_between(days, 
                           [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.2, color=colors.get(arm, 'black'))
        
        ax.set_xlabel('Study Day')
        ax.set_ylabel('Efficacy Score')
        ax.set_title('Efficacy Over Time by Treatment Arm')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_response_rates(self, ax, treatment_results: Dict):
        """Plot response rates by arm"""
        response_categories = ['Complete Response', 'Partial Response', 
                              'Minimal Response', 'No Response']
        
        arm_response_rates = {}
        
        for arm, arm_results in treatment_results['arms'].items():
            response_counts = {category: 0 for category in response_categories}
            
            for patient_result in arm_results['patients']:
                category = patient_result.get('response_category', 'No Response')
                if category in response_counts:
                    response_counts[category] += 1
            
            total_patients = len(arm_results['patients'])
            if total_patients > 0:
                arm_response_rates[arm] = {
                    category: count/total_patients 
                    for category, count in response_counts.items()
                }
        
        # Plot as stacked bar chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(response_categories)))
        bottom = np.zeros(len(arm_response_rates))
        
        for i, category in enumerate(response_categories):
            rates = [arm_response_rates[arm].get(category, 0) 
                    for arm in arm_response_rates.keys()]
            ax.bar(arm_response_rates.keys(), rates, bottom=bottom, 
                  label=category, color=colors[i])
            bottom += rates
        
        ax.set_ylabel('Proportion of Patients')
        ax.set_title('Response Rates by Treatment Arm')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)
    
    def _plot_adverse_events(self, ax, treatment_results: Dict):
        """Plot adverse events by arm"""
        safety_analysis = self._analyze_safety(treatment_results)
        
        arms = list(safety_analysis['adverse_events_by_arm'].keys())
        events_data = {}
        
        for arm in arms:
            event_rates = safety_analysis['adverse_events_by_arm'][arm]['event_rates']
            for event, rate in event_rates.items():
                if event not in events_data:
                    events_data[event] = {}
                events_data[event][arm] = rate
        
        # Get top 5 events by average rate
        avg_rates = {
            event: np.mean(list(rates.values())) 
            for event, rates in events_data.items()
        }
        top_events = sorted(avg_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Plot
        x = np.arange(len(arms))
        width = 0.15
        
        for i, (event, _) in enumerate(top_events):
            rates = [events_data[event].get(arm, 0) for arm in arms]
            ax.bar(x + (i - 2) * width, rates, width, label=event)
        
        ax.set_xlabel('Treatment Arm')
        ax.set_ylabel('Event Rate')
        ax.set_title('Top 5 Adverse Events by Arm')
        ax.set_xticks(x)
        ax.set_xticklabels([arm.capitalize() for arm in arms])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_subgroup_analysis(self, ax, statistical_analysis: Dict, subgroup: str):
        """Plot subgroup analysis results"""
        if subgroup not in statistical_analysis['subgroup_analyses']:
            ax.text(0.5, 0.5, 'No subgroup data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{subgroup.replace("_", " ").title()} Analysis')
            return
        
        subgroup_data = statistical_analysis['subgroup_analyses'][subgroup]
        
        categories = list(subgroup_data.keys())
        response_rates = [subgroup_data[cat].get('response_rate', 0) 
                         for cat in categories]
        
        ax.bar(categories, response_rates, color='skyblue')
        ax.set_xlabel(subgroup.replace('_', ' ').title())
        ax.set_ylabel('Response Rate')
        ax.set_title(f'{subgroup.replace("_", " ").title()} Analysis')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(response_rates):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center')
    
    def _plot_power_analysis(self, ax, statistical_analysis: Dict):
        """Plot power analysis results"""
        power_data = statistical_analysis['power_analysis']
        
        if not power_data.get('sample_sizes'):
            ax.text(0.5, 0.5, 'No power analysis data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Statistical Power Analysis')
            return
        
        # Create power analysis visualization
        labels = ['Achieved Power', 'Target Power (0.8)']
        values = [power_data.get('achieved_power', 0), 0.8]
        colors = ['green' if power_data.get('adequate_power', False) else 'red', 'gray']
        
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel('Power')
        ax.set_title('Statistical Power Analysis')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, v in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{v:.1%}', ha='center', va='bottom')
        
        # Add effect size annotation
        effect_size = power_data.get('effect_size', 0)
        ax.text(0.5, 0.5, f'Effect Size (d): {effect_size:.2f}',
               transform=ax.transAxes, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def export_report(self, trial_id: str = None, format: str = 'json',
                     save_path: str = None) -> Union[str, None]:
        """Export trial report in specified format"""
        if not trial_id and self.trial_results:
            trial_id = list(self.trial_results.keys())[-1]
        
        if trial_id not in self.trial_results:
            print(f"Trial {trial_id} not found")
            return None
        
        report = self.trial_results[trial_id]['report']
        
        if format.lower() == 'json':
            output = json.dumps(report, indent=2, default=str)
        elif format.lower() == 'csv':
            # Convert to DataFrame for CSV export
            import pandas as pd
            
            # Create flattened version for CSV
            flattened = self._flatten_report(report)
            df = pd.DataFrame([flattened])
            output = df.to_csv(index=False)
        else:
            print(f"Format {format} not supported")
            return None
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(output)
            print(f"Report saved to {save_path}")
        
        return output
    
    def _flatten_report(self, report: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in report.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_report(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to strings
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def get_status(self) -> Dict:
        """Get current status of virtual trial simulator"""
        return {
            'initialized': self.initialized,
            'simulation_fidelity': self.simulation_fidelity,
            'current_disease': self.current_disease,
            'patient_database_size': len(self.patient_database),
            'completed_trials': len(self.trial_results),
            'disease_models_loaded': list(self.disease_models.keys())
        }
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up Virtual Trial Simulator...")
        
        self.patient_database.clear()
        self.disease_models.clear()
        self.trial_results.clear()
        self.simulation_history.clear()

# Example usage and helper functions

def run_virtual_trial_demo():
    """Run a demonstration of virtual trial simulation"""
    print("Running Virtual Trial Simulation Demo...")
    
    # Create sample compound
    compound_data = {
        'name': 'TestDrug-123',
        'recommended_dose': 100,  # mg
        'predicted_efficacy': 0.7,
        'predicted_toxicity': 0.3,
        'pk_parameters': {
            'half_life': 12,  # hours
            'bioavailability': 0.8,
            'protein_binding': 0.9
        }
    }
    
    # Initialize simulator
    simulator = VirtualTrialSimulator(simulation_fidelity='medium')
    simulator.initialize(disease_type='arthritis', population_size=500)
    
    # Configure trial
    trial_config = {
        'design': 'randomized_double_blind',
        'arms': ['treatment', 'placebo'],
        'randomization_ratio': [1, 1],
        'duration_days': 90,
        'visits': [0, 7, 14, 28, 56, 90],
        'primary_endpoint': 'efficacy',
        'screening_criteria': {
            'inclusion': ['age_18_75', 'bmi_18_40', 'disease_severity'],
            'exclusion': ['severe_comorbidities', 'concurrent_medications']
        }
    }
    
    # Run simulation
    trial_report = simulator.simulate_trial(compound_data, trial_config)
    
    # Print summary
    print(f"\n{'='*60}")
    print("VIRTUAL TRIAL SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"Trial ID: {trial_report['trial_id']}")
    print(f"Compound: {trial_report['compound_name']}")
    print(f"Disease: {trial_report['disease']}")
    print(f"\nExecutive Summary:")
    print(trial_report['executive_summary'])
    
    # Print key results
    primary = trial_report['efficacy_results']
    if 'treatment_vs_placebo' in primary.get('statistical_tests', {}):
        test = primary['statistical_tests']['treatment_vs_placebo']
        print(f"\nPrimary Endpoint Analysis:")
        print(f"  p-value: {test['p_value']:.4f}")
        print(f"  Effect Size (Cohen's d): {test['effect_size']:.2f}")
        print(f"  Significant: {'Yes' if test['significant'] else 'No'}")
    
    # Print safety summary
    safety = trial_report['safety_results']
    print(f"\nSafety Summary:")
    print(f"  Safety Signals: {len(safety['safety_signals'])}")
    for signal in safety['safety_signals'][:3]:
        print(f"    - {signal['event']}: RR={signal['risk_ratio']:.1f}")
    
    # Print recommendations
    print(f"\nRecommendations:")
    for rec in trial_report['recommendations'][:3]:
        print(f"  [{rec['priority']}] {rec['recommendation']}")
    
    # Visualize results
    simulator.visualize_results(trial_report['trial_id'])
    
    # Cleanup
    simulator.cleanup()
    
    return trial_report

if __name__ == "__main__":
    report = run_virtual_trial_demo()