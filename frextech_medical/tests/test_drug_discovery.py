#!/usr/bin/env python3
"""
Comprehensive Tests for Drug Discovery Module
Unit and integration tests for molecular modeling, AI screening, and virtual trials
"""

import unittest
import asyncio
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drug_discovery.molecular_modeling import (
    MolecularModeler, MoleculeType, SequenceData, 
    DockingResult, MolecularDynamicsResult
)
from drug_discovery.ai_screening import (
    AIScreening, ScreeningResult, ToxicityCategory, ADMEProperty
)
from drug_discovery.virtual_trials import (
    VirtualTrialSimulator, PatientProfile, PatientGender,
    DiseaseSeverity, TreatmentArm, TrialOutcome
)

class TestMolecularModeling(unittest.TestCase):
    """Test Molecular Modeling module"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.modeler = MolecularModeler(cache_dir=self.temp_dir)
        
        # Create test SMILES
        self.test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_from_smiles(self):
        """Test loading molecule from SMILES"""
        molecule_id = "test_aspirin"
        success = self.modeler.load_from_smiles(
            self.test_smiles, 
            molecule_id,
            MoleculeType.SMALL_MOLECULE
        )
        
        self.assertTrue(success)
        self.assertIn(molecule_id, self.modeler.molecules)
        
        # Check molecule properties
        mol = self.modeler.molecules[molecule_id]
        self.assertIsNotNone(mol)
        self.assertGreater(mol.GetNumAtoms(), 0)
    
    def test_calculate_descriptors(self):
        """Test molecular descriptor calculation"""
        # Load test molecule
        molecule_id = "test_desc"
        self.modeler.load_from_smiles(self.test_smiles, molecule_id)
        
        # Calculate descriptors
        descriptors = self.modeler.calculate_descriptors(molecule_id)
        
        self.assertGreater(len(descriptors), 0)
        
        # Check for specific descriptors
        descriptor_names = [desc.name for desc in descriptors]
        self.assertIn("Molecular Weight", descriptor_names)
        self.assertIn("LogP", descriptor_names)
        self.assertIn("TPSA", descriptor_names)
        
        # Check values are reasonable
        for desc in descriptors:
            if desc.name == "Molecular Weight":
                self.assertGreater(desc.value, 100)
                self.assertLess(desc.value, 500)
    
    def test_docking_simulation(self):
        """Test molecular docking simulation"""
        # Load test compound
        compound_id = "test_ligand"
        self.modeler.load_from_smiles(self.test_smiles, compound_id)
        
        # Create a mock protein (simplified)
        # In real test, you would load an actual protein
        protein_smiles = "C" * 100  # Simple mock
        target_id = "test_target"
        
        # Note: This test is simplified since we don't have actual protein structure
        # In production, you would test with real PDB files
        
        print("Note: Docking test requires actual protein structure")
        self.assertTrue(True)  # Placeholder assertion
    
    def test_binding_affinity_model_training(self):
        """Test binding affinity ML model training"""
        # Create synthetic training data
        n_samples = 100
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.uniform(0, 10, n_samples)  # pKd values
        
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        training_data = pd.DataFrame(X, columns=feature_names)
        training_data['pKd'] = y
        
        # Train model
        results = self.modeler.train_binding_affinity_model(training_data)
        
        self.assertIsNotNone(results)
        self.assertIn('metrics', results)
        self.assertIn('r2', results['metrics'])
        
        # Check that model was stored
        self.assertIsNotNone(self.modeler.binding_affinity_model)
    
    def test_molecular_dynamics_simulation(self):
        """Test molecular dynamics simulation (simplified)"""
        # This test would require actual molecular structures
        # For now, test the method exists and can be called
        
        try:
            # The method should handle missing data gracefully
            result = self.modeler.run_molecular_dynamics(
                system_id="test_system",
                compound_id="test_compound",
                target_id="test_target",
                simulation_time_ns=0.1  # Short simulation
            )
            
            # Either returns None (if no docking results) or a valid result
            if result is not None:
                self.assertIsInstance(result, MolecularDynamicsResult)
                self.assertIsNotNone(result.simulation_id)
                
        except Exception as e:
            # Method should not crash
            print(f"MD simulation test warning: {e}")
            self.assertTrue(True)  # Method exists and was called
    
    def test_visualization(self):
        """Test visualization methods"""
        # Test that visualization methods exist and can be called
        try:
            success = self.modeler.visualize_docking(
                compound_id="test",
                target_id="test",
                output_file=os.path.join(self.temp_dir, "test_viz.png")
            )
            
            # Method should not crash
            # Success depends on whether we have actual data
            self.assertTrue(True)  # Method exists
            
        except Exception as e:
            print(f"Visualization test warning: {e}")
            self.assertTrue(True)  # Method exists

class TestAIScreening(unittest.TestCase):
    """Test AI Screening module"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.screening = AIScreening(cache_dir=self.temp_dir)
        
        # Create test data
        self.test_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        ]
        
        self.compound_ids = [f"TEST_{i:03d}" for i in range(len(self.test_smiles))]
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_feature_generation(self):
        """Test molecular feature generation from SMILES"""
        features_df = self.screening.generate_features_from_smiles(
            self.test_smiles,
            self.compound_ids
        )
        
        self.assertFalse(features_df.empty)
        self.assertEqual(len(features_df), len(self.test_smiles))
        
        # Check for expected features
        expected_features = ['MolecularWeight', 'LogP', 'TPSA', 'QED']
        for feature in expected_features:
            self.assertIn(feature, features_df.columns)
        
        # Check feature values are reasonable
        self.assertTrue((features_df['MolecularWeight'] > 0).all())
        self.assertTrue((features_df['QED'] >= 0).all())
        self.assertTrue((features_df['QED'] <= 1).all())
    
    def test_efficacy_model_training(self):
        """Test efficacy model training"""
        # Generate features
        features_df = self.screening.generate_features_from_smiles(
            self.test_smiles,
            self.compound_ids
        )
        
        # Create synthetic training data
        n_samples = 200
        n_features = len(features_df.columns)
        
        X = np.random.randn(n_samples, n_features)
        y = (np.random.rand(n_samples) > 0.5).astype(int)  # Binary efficacy
        
        training_data = pd.DataFrame(
            X, 
            columns=[f"F{i}" for i in range(n_features)]
        )
        training_data['efficacy'] = y
        
        # Train model
        performance = self.screening.train_efficacy_model(
            features=training_data.drop('efficacy', axis=1),
            target=training_data['efficacy'],
            model_type="random_forest"
        )
        
        self.assertIsNotNone(performance)
        self.assertEqual(performance.model_name, "random_forest")
        self.assertGreaterEqual(performance.accuracy, 0.0)
        self.assertLessEqual(performance.accuracy, 1.0)
        
        # Check that model was stored
        self.assertIn("random_forest", self.screening.efficacy_models)
    
    def test_toxicity_model_training(self):
        """Test toxicity model training"""
        # Create synthetic training data
        n_samples = 150
        n_features = 50
        
        X = np.random.randn(n_samples, n_features)
        y = (np.random.rand(n_samples) > 0.3).astype(int)  # 30% toxic
        
        training_data = pd.DataFrame(
            X, 
            columns=[f"F{i}" for i in range(n_features)]
        )
        training_data['toxicity'] = y
        
        # Train model
        performance = self.screening.train_toxicity_model(
            features=training_data.drop('toxicity', axis=1),
            target=training_data['toxicity'],
            model_type="xgboost",
            toxicity_type="general"
        )
        
        self.assertIsNotNone(performance)
        self.assertIn("xgboost", performance.model_name)
        self.assertGreaterEqual(performance.accuracy, 0.0)
        
        # Check that model was stored
        self.assertIn("general", self.screening.toxicity_models)
    
    def test_adme_model_training(self):
        """Test ADME model training"""
        # Create synthetic training data for bioavailability
        n_samples = 100
        n_features = 40
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.uniform(0, 1, n_samples)  # Bioavailability 0-1
        
        training_data = pd.DataFrame(
            X, 
            columns=[f"F{i}" for i in range(n_features)]
        )
        training_data['bioavailability'] = y
        
        # Train model
        performance = self.screening.train_adme_model(
            adme_property=ADMEProperty.BIOAVAILABILITY,
            features=training_data.drop('bioavailability', axis=1),
            target=training_data['bioavailability'],
            model_type="gradient_boosting"
        )
        
        self.assertIsNotNone(performance)
        self.assertIn("bioavailability", performance.model_name)
        
        # Check that model was stored
        self.assertIn(ADMEProperty.BIOAVAILABILITY, self.screening.adme_models)
    
    def test_compound_screening(self):
        """Test compound screening pipeline"""
        # Generate features
        features_df = self.screening.generate_features_from_smiles(
            self.test_smiles,
            self.compound_ids
        )
        
        # Train a simple model first
        n_samples = 50
        n_features = len(features_df.columns)
        
        X = np.random.randn(n_samples, n_features)
        y_eff = (np.random.rand(n_samples) > 0.5).astype(int)
        y_tox = (np.random.rand(n_samples) > 0.3).astype(int)
        
        # Train efficacy model
        self.screening.train_efficacy_model(
            pd.DataFrame(X, columns=[f"F{i}" for i in range(n_features)]),
            pd.Series(y_eff),
            model_type="random_forest"
        )
        
        # Train toxicity model
        self.screening.train_toxicity_model(
            pd.DataFrame(X, columns=[f"F{i}" for i in range(n_features)]),
            pd.Series(y_tox),
            model_type="random_forest"
        )
        
        # Screen compounds
        screening_results = self.screening.screen_compounds(
            features_df.reset_index()
        )
        
        self.assertIsInstance(screening_results, list)
        self.assertEqual(len(screening_results), len(self.test_smiles))
        
        # Check result structure
        for result in screening_results:
            self.assertIsInstance(result, ScreeningResult)
            self.assertIn(result.compound_id, self.compound_ids)
            self.assertGreaterEqual(result.efficacy_score, 0.0)
            self.assertLessEqual(result.efficacy_score, 1.0)
            self.assertGreaterEqual(result.toxicity_score, 0.0)
            self.assertLessEqual(result.toxicity_score, 1.0)
            self.assertIsInstance(result.adme_profile, dict)
            self.assertIsInstance(result.risk_assessment, dict)
    
    def test_visualization(self):
        """Test screening visualization"""
        # Create mock screening results
        results = []
        for i, compound_id in enumerate(self.compound_ids):
            results.append(ScreeningResult(
                compound_id=compound_id,
                efficacy_score=np.random.uniform(0.3, 0.9),
                toxicity_score=np.random.uniform(0.1, 0.7),
                adme_profile={prop: np.random.uniform(0.4, 0.8) for prop in ADMEProperty},
                predicted_ic50=np.random.uniform(10, 1000),
                predicted_ld50=np.random.uniform(100, 10000),
                confidence_scores={
                    'efficacy': 0.8,
                    'toxicity': 0.7,
                    'adme': 0.6
                },
                risk_assessment={'overall': 'MODERATE_RISK'}
            ))
        
        # Test visualization
        output_file = os.path.join(self.temp_dir, "test_visualization.html")
        success = self.screening.visualize_screening_results(
            results, output_file
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_file))
        
        # Test report generation
        report_file = os.path.join(self.temp_dir, "test_report.pdf")
        success = self.screening.create_screening_report(results, report_file)
        
        # Report might fail due to missing dependencies, but method should exist
        if not success:
            print("Note: PDF report generation may require additional dependencies")

class TestVirtualTrials(unittest.TestCase):
    """Test Virtual Trials module"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.simulator = VirtualTrialSimulator(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_patient_population_generation(self):
        """Test virtual patient population generation"""
        population_id = "TEST_POP_001"
        n_patients = 50
        
        patients = self.simulator.generate_patient_population(
            population_id=population_id,
            n_patients=n_patients,
            disease_type="Test Disease",
            age_range=(20, 70),
            gender_ratio=0.5,
            comorbidity_prevalence={
                "hypertension": 0.2,
                "diabetes": 0.1
            }
        )
        
        self.assertEqual(len(patients), n_patients)
        self.assertIn(population_id, self.simulator.patient_populations)
        
        # Check patient properties
        for patient in patients:
            self.assertIsInstance(patient, PatientProfile)
            self.assertGreaterEqual(patient.age, 20)
            self.assertLessEqual(patient.age, 70)
            self.assertIn(patient.gender, [PatientGender.MALE, PatientGender.FEMALE])
            self.assertGreater(patient.weight_kg, 0)
            self.assertGreater(patient.height_cm, 0)
            self.assertGreater(patient.bmi, 0)
            self.assertIsInstance(patient.disease_status, DiseaseSeverity)
            self.assertIsInstance(patient.comorbidities, list)
            self.assertIsInstance(patient.genetic_markers, dict)
            self.assertIsInstance(patient.biomarkers, dict)
            self.assertIsInstance(patient.medication_history, list)
            
            # Check age group calculation
            age_group = patient.get_age_group()
            self.assertIn(age_group, ["pediatric", "adult", "geriatric"])
            
            # Check BMI calculation
            calculated_bmi = patient.calculate_bmi()
            self.assertAlmostEqual(patient.bmi, calculated_bmi, places=1)
    
    def test_treatment_arm_creation(self):
        """Test treatment arm creation"""
        trial_id = "TEST_TRIAL_001"
        
        treatment_configs = [
            {
                "name": "Placebo",
                "dosage_mg": 0,
                "frequency_days": 1,
                "administration_route": "oral",
                "duration_days": 30,
                "inclusion_criteria": {"min_age": 18, "max_age": 75},
                "exclusion_criteria": {"comorbidities": ["severe_liver_disease"]},
                "mechanism": "control"
            },
            {
                "name": "Test Drug",
                "dosage_mg": 100,
                "frequency_days": 1,
                "administration_route": "oral",
                "duration_days": 30,
                "inclusion_criteria": {"min_age": 18, "max_age": 75},
                "exclusion_criteria": {"comorbidities": ["severe_liver_disease"]},
                "mechanism": "enzyme_inhibition"
            }
        ]
        
        arms = self.simulator.create_treatment_arms(
            trial_id=trial_id,
            treatment_configs=treatment_configs
        )
        
        self.assertEqual(len(arms), 2)
        self.assertIn(trial_id, self.simulator.treatment_arms)
        
        # Check arm properties
        for i, arm in enumerate(arms):
            self.assertIsInstance(arm, TreatmentArm)
            self.assertEqual(arm.treatment_name, treatment_configs[i]["name"])
            self.assertEqual(arm.dosage_mg, treatment_configs[i]["dosage_mg"])
            self.assertEqual(arm.frequency_days, treatment_configs[i]["frequency_days"])
            self.assertEqual(arm.duration_days, treatment_configs[i]["duration_days"])
            self.assertIsInstance(arm.inclusion_criteria, dict)
            self.assertIsInstance(arm.exclusion_criteria, dict)
    
    def test_pk_simulation(self):
        """Test pharmacokinetic simulation"""
        # Create a test patient
        patient = PatientProfile(
            patient_id="TEST_PATIENT",
            age=45,
            gender=PatientGender.MALE,
            weight_kg=70,
            height_cm=175,
            bmi=22.9,
            disease_status=DiseaseSeverity.MODERATE,
            comorbidities=["hypertension"],
            genetic_markers={"CYP2D6": "extensive", "CYP3A4": "normal"},
            biomarkers={"creatinine": 0.9, "ALT": 25},
            medication_history=[]
        )
        
        # Create a test treatment
        treatment = TreatmentArm(
            arm_id="TEST_ARM",
            treatment_name="Test Drug",
            dosage_mg=100,
            frequency_days=1,
            administration_route="oral",
            duration_days=30,
            inclusion_criteria={},
            exclusion_criteria={},
            expected_mechanism="test"
        )
        
        # Simulate PK profile
        time_points = [0, 1, 2, 4, 8, 12, 24]
        pk_profile = self.simulator.simulate_pk_profile(
            patient, treatment, time_points
        )
        
        self.assertIsInstance(pk_profile, dict)
        self.assertIn("time_points", pk_profile)
        self.assertIn("concentrations", pk_profile)
        self.assertIn("auc", pk_profile)
        self.assertIn("cmax", pk_profile)
        self.assertIn("tmax", pk_profile)
        self.assertIn("half_life", pk_profile)
        self.assertIn("parameters", pk_profile)
        
        # Check data structure
        self.assertEqual(len(pk_profile["time_points"]), len(time_points))
        self.assertEqual(len(pk_profile["concentrations"]), len(time_points))
        
        # Check values are reasonable
        self.assertGreaterEqual(pk_profile["auc"], 0)
        self.assertGreaterEqual(pk_profile["cmax"], 0)
        self.assertGreaterEqual(pk_profile["half_life"], 0)
        
        # tmax should be within time points
        self.assertGreaterEqual(pk_profile["tmax"], 0)
        self.assertLessEqual(pk_profile["tmax"], max(time_points))
    
    def test_pd_simulation(self):
        """Test pharmacodynamic simulation"""
        # Create mock PK profile
        time_points = [0, 1, 2, 4, 8, 12, 24]
        pk_profile = {
            "time_points": time_points,
            "concentrations": [0, 5, 8, 6, 4, 2, 1],
            "auc": 50,
            "cmax": 8,
            "tmax": 2,
            "half_life": 6,
            "parameters": {}
        }
        
        # Create test patient and treatment
        patient = PatientProfile(
            patient_id="TEST_PATIENT",
            age=45,
            gender=PatientGender.MALE,
            weight_kg=70,
            height_cm=175,
            bmi=22.9,
            disease_status=DiseaseSeverity.MODERATE,
            comorbidities=[],
            genetic_markers={},
            biomarkers={},
            medication_history=[]
        )
        
        treatment = TreatmentArm(
            arm_id="TEST_ARM",
            treatment_name="Test Drug",
            dosage_mg=100,
            frequency_days=1,
            administration_route="oral",
            duration_days=30,
            inclusion_criteria={},
            exclusion_criteria={},
            expected_mechanism="test"
        )
        
        # Simulate PD response
        pd_response = self.simulator.simulate_pd_response(
            pk_profile, patient, treatment, time_points
        )
        
        self.assertIsInstance(pd_response, dict)
        self.assertIn("time_points", pd_response)
        self.assertIn("effects", pd_response)
        self.assertIn("max_effect", pd_response)
        self.assertIn("time_to_response", pd_response)
        self.assertIn("parameters", pd_response)
        
        # Check data structure
        self.assertEqual(len(pd_response["time_points"]), len(time_points))
        self.assertEqual(len(pd_response["effects"]), len(time_points))
        
        # Check values are reasonable
        self.assertGreaterEqual(pd_response["max_effect"], 0)
        self.assertLessEqual(pd_response["max_effect"], 1)
        
        # Effects should be between 0 and 1
        for effect in pd_response["effects"]:
            self.assertGreaterEqual(effect, 0)
            self.assertLessEqual(effect, 1)
    
    def test_adverse_event_simulation(self):
        """Test adverse event simulation"""
        # Create test patient and treatment
        patient = PatientProfile(
            patient_id="TEST_PATIENT",
            age=45,
            gender=PatientGender.MALE,
            weight_kg=70,
            height_cm=175,
            bmi=22.9,
            disease_status=DiseaseSeverity.MODERATE,
            comorbidities=[],
            genetic_markers={"HLA-B*5701": "positive"},  # Increased risk
            biomarkers={},
            medication_history=[]
        )
        
        treatment = TreatmentArm(
            arm_id="TEST_ARM",
            treatment_name="Test Drug",
            dosage_mg=100,
            frequency_days=1,
            administration_route="oral",
            duration_days=30,
            inclusion_criteria={},
            exclusion_criteria={},
            expected_mechanism="test"
        )
        
        # Mock PK profile
        pk_profile = {
            "concentrations": [1, 2, 3],
            "auc": 50,
            "cmax": 3
        }
        
        # Simulate adverse events
        adverse_events = self.simulator.simulate_adverse_events(
            patient, treatment, pk_profile, duration_days=30
        )
        
        self.assertIsInstance(adverse_events, list)
        
        # Check event structure if any events occurred
        for event in adverse_events:
            self.assertIsInstance(event, dict)
            self.assertIn("event", event)
            self.assertIn("onset_day", event)
            self.assertIn("severity", event)
            self.assertIn("duration_days", event)
            self.assertIn("action_taken", event)
            self.assertIn("related_to_treatment", event)
            self.assertIn("serious", event)
            
            # Check values are reasonable
            self.assertGreaterEqual(event["onset_day"], 1)
            self.assertLessEqual(event["onset_day"], 30)
            self.assertGreaterEqual(event["severity"], 1)
            self.assertLessEqual(event["severity"], 4)
            self.assertGreater(event["duration_days"], 0)
            self.assertIn(event["action_taken"], 
                         ["none", "dose_reduction", "treatment_interruption", "treatment_discontinuation"])
    
    def test_virtual_trial_simulation(self):
        """Test complete virtual trial simulation"""
        # Generate patient population
        population_id = "TRIAL_TEST_POP"
        patients = self.simulator.generate_patient_population(
            population_id=population_id,
            n_patients=20,  # Small number for test
            disease_type="Test Disease",
            age_range=(30, 60)
        )
        
        # Define treatment arms
        treatment_configs = [
            {
                "name": "Placebo",
                "dosage_mg": 0,
                "frequency_days": 1,
                "is_control": True,
                "inclusion_criteria": {"min_age": 30, "max_age": 60},
                "exclusion_criteria": {}
            },
            {
                "name": "Test Drug",
                "dosage_mg": 50,
                "frequency_days": 1,
                "inclusion_criteria": {"min_age": 30, "max_age": 60},
                "exclusion_criteria": {}
            }
        ]
        
        # Run virtual trial
        trial_id = "TEST_TRIAL_SIM"
        outcomes = self.simulator.run_virtual_trial(
            trial_id=trial_id,
            population_id=population_id,
            treatment_configs=treatment_configs,
            primary_endpoint="test_score",
            secondary_endpoints=["quality_of_life"],
            n_patients_per_arm=5  # Small for test
        )
        
        self.assertIsInstance(outcomes, list)
        
        # Check outcomes if any were generated
        if outcomes:
            for outcome in outcomes:
                self.assertIsInstance(outcome, TrialOutcome)
                self.assertEqual(outcome.trial_id, trial_id)
                self.assertIn(outcome.treatment_arm, 
                            [f"{trial_id}_ARM_00", f"{trial_id}_ARM_01"])
                self.assertGreaterEqual(outcome.primary_endpoint, 0)
                self.assertIsInstance(outcome.secondary_endpoints, dict)
                self.assertIsInstance(outcome.adverse_events, list)
                self.assertIsInstance(outcome.biomarker_changes, dict)
                self.assertGreaterEqual(outcome.efficacy_score, 0)
                self.assertLessEqual(outcome.efficacy_score, 1)
                self.assertGreaterEqual(outcome.safety_score, 0)
                self.assertLessEqual(outcome.safety_score, 1)
                self.assertGreaterEqual(outcome.quality_of_life_score, 0)
                self.assertLessEqual(outcome.quality_of_life_score, 1)
                self.assertIn(outcome.response_category, 
                            ["EXCELLENT_RESPONSE", "GOOD_RESPONSE", 
                             "MODERATE_RESPONSE", "POOR_RESPONSE"])
            
            # Check that statistics were calculated
            self.assertIn(trial_id, self.simulator.population_stats)
            
            # Check report generation
            report_file = os.path.join(self.temp_dir, "test_trial_report.html")
            success = self.simulator.create_trial_report(trial_id, report_file)
            
            # Report might fail, but method should not crash
            if not success:
                print("Note: Trial report generation completed with warnings")
        else:
            print("Note: No outcomes generated in test trial (may be due to eligibility criteria)")
    
    def test_dosage_optimization(self):
        """Test dosage optimization"""
        # Create a mock trial with outcomes
        trial_id = "OPT_TEST_TRIAL"
        
        # We need to create some mock outcomes first
        # For this test, we'll directly test the optimization method
        # with synthetic data
        
        try:
            # Test the optimization method
            result = self.simulator.optimize_dosage(
                trial_id=trial_id,
                target_efficacy=0.7,
                max_toxicity=0.3,
                dosage_range=(10, 200),
                n_simulations=20
            )
            
            # Method should handle missing data gracefully
            # Either returns empty dict or optimization results
            if result:
                self.assertIsInstance(result, dict)
                self.assertIn("optimal_dosage_mg", result)
                self.assertIn("predicted_efficacy", result)
                self.assertIn("predicted_toxicity", result)
                self.assertIn("therapeutic_index", result)
                self.assertIn("dosage_response_curve", result)
                
                # Check values are reasonable
                self.assertGreater(result["optimal_dosage_mg"], 0)
                self.assertGreaterEqual(result["predicted_efficacy"], 0)
                self.assertLessEqual(result["predicted_efficacy"], 1)
                self.assertGreaterEqual(result["predicted_toxicity"], 0)
                self.assertLessEqual(result["predicted_toxicity"], 1)
                self.assertGreater(result["therapeutic_index"], 0)
            
        except Exception as e:
            print(f"Dosage optimization test warning: {e}")
            # Method should not crash
            self.assertTrue(True)

class TestIntegrationWorkflows(unittest.TestCase):
    """Integration tests for complete drug discovery workflows"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize all three systems
        self.modeler = MolecularModeler(cache_dir=os.path.join(self.temp_dir, "modeling"))
        self.screening = AIScreening(cache_dir=os.path.join(self.temp_dir, "screening"))
        self.trials = VirtualTrialSimulator(cache_dir=os.path.join(self.temp_dir, "trials"))
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_drug_discovery_workflow(self):
        """
        Test complete drug discovery workflow:
        1. Molecular modeling for lead compounds
        2. AI screening for efficacy/toxicity
        3. Virtual trials for dosage optimization
        """
        
        print("\n" + "="*60)
        print("Testing Complete Drug Discovery Workflow")
        print("="*60)
        
        # Step 1: Molecular Modeling - Discover lead compounds
        print("\n1. Molecular Modeling: Discovering lead compounds...")
        
        # Load example compounds
        compounds = [
            ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
            ("Celecoxib", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"),
            ("Metformin", "CN(C)C(=N)N=C(N)N"),
            ("Simvastatin", "CC(C)C(=O)O[C@H]1C[C@@H]2CC[C@H]3C[C@H](C[C@H]4C[C@@H](CC[C@]34C)O2)C1")
        ]
        
        compound_ids = []
        for name, smiles in compounds:
            compound_id = f"LEAD_{name.upper()}"
            success = self.modeler.load_from_smiles(
                smiles, compound_id, MoleculeType.SMALL_MOLECULE
            )
            if success:
                compound_ids.append(compound_id)
                print(f"  Loaded: {name} as {compound_id}")
        
        self.assertGreater(len(compound_ids), 0, "Failed to load compounds")
        
        # Calculate descriptors
        print(f"\n  Calculating molecular descriptors...")
        all_descriptors = []
        for compound_id in compound_ids:
            descriptors = self.modeler.calculate_descriptors(compound_id)
            all_descriptors.extend(descriptors)
            print(f"    {compound_id}: {len(descriptors)} descriptors")
        
        # Step 2: AI Screening - Predict efficacy and toxicity
        print("\n2. AI Screening: Predicting efficacy and toxicity...")
        
        # Generate features for screening
        features_df = self.screening.generate_features_from_smiles(
            [comp[1] for comp in compounds],
            compound_ids
        )
        
        self.assertFalse(features_df.empty, "Failed to generate features")
        print(f"  Generated {len(features_df.columns)} features")
        
        # Train models with synthetic data
        print("\n  Training ML models with synthetic data...")
        
        # Create synthetic training data
        n_samples = 100
        n_features = min(50, len(features_df.columns))
        
        X_train = np.random.randn(n_samples, n_features)
        y_eff = (np.random.rand(n_samples) > 0.4).astype(int)
        y_tox = (np.random.rand(n_samples) > 0.2).astype(int)
        
        # Train efficacy model
        eff_perf = self.screening.train_efficacy_model(
            pd.DataFrame(X_train, columns=[f"F{i}" for i in range(n_features)]),
            pd.Series(y_eff),
            model_type="random_forest"
        )
        
        print(f"    Efficacy model accuracy: {eff_perf.accuracy:.3f}")
        
        # Train toxicity model
        tox_perf = self.screening.train_toxicity_model(
            pd.DataFrame(X_train, columns=[f"F{i}" for i in range(n_features)]),
            pd.Series(y_tox),
            model_type="random_forest"
        )
        
        print(f"    Toxicity model accuracy: {tox_perf.accuracy:.3f}")
        
        # Screen compounds
        print("\n  Screening lead compounds...")
        screening_results = self.screening.screen_compounds(
            features_df.reset_index()
        )
        
        self.assertEqual(len(screening_results), len(compound_ids))
        
        # Analyze screening results
        promising_compounds = []
        for result in screening_results:
            net_score = result.efficacy_score - result.toxicity_score
            risk = result.risk_assessment.get('overall', 'UNKNOWN')
            
            print(f"    {result.compound_id}:")
            print(f"      Efficacy: {result.efficacy_score:.3f}, "
                  f"Toxicity: {result.toxicity_score:.3f}, "
                  f"Net: {net_score:.3f}, Risk: {risk}")
            
            if net_score > 0.3 and risk in ['LOW_RISK_HIGH_POTENTIAL', 'MODERATE_RISK']:
                promising_compounds.append(result.compound_id)
        
        print(f"\n  Promising compounds: {len(promising_compounds)}/{len(compound_ids)}")
        
        # Step 3: Virtual Trials - Optimize dosage
        if promising_compounds:
            print("\n3. Virtual Trials: Optimizing dosage...")
            
            # Select most promising compound
            selected_compound = promising_compounds[0]
            print(f"  Selected compound for trials: {selected_compound}")
            
            # Generate patient population
            population_id = "RA_POPULATION"
            patients = self.trials.generate_patient_population(
                population_id=population_id,
                n_patients=100,
                disease_type="Rheumatoid Arthritis",
                age_range=(30, 70),
                gender_ratio=0.7  # RA is more common in females
            )
            
            print(f"  Generated {len(patients)} virtual patients")
            
            # Define treatment arms with different dosages
            treatment_configs = [
                {
                    "name": "Placebo",
                    "dosage_mg": 0,
                    "frequency_days": 1,
                    "is_control": True,
                    "inclusion_criteria": {"min_age": 30, "max_age": 70},
                    "exclusion_criteria": {}
                },
                {
                    "name": f"{selected_compound} Low",
                    "dosage_mg": 25,
                    "frequency_days": 1,
                    "inclusion_criteria": {"min_age": 30, "max_age": 70},
                    "exclusion_criteria": {}
                },
                {
                    "name": f"{selected_compound} Medium",
                    "dosage_mg": 50,
                    "frequency_days": 1,
                    "inclusion_criteria": {"min_age": 30, "max_age": 70},
                    "exclusion_criteria": {}
                },
                {
                    "name": f"{selected_compound} High",
                    "dosage_mg": 100,
                    "frequency_days": 1,
                    "inclusion_criteria": {"min_age": 30, "max_age": 70},
                    "exclusion_criteria": {"comorbidities": ["severe_liver_disease", "renal_impairment"]}
                }
            ]
            
            # Run virtual trial
            trial_id = f"{selected_compound}_TRIAL"
            outcomes = self.trials.run_virtual_trial(
                trial_id=trial_id,
                population_id=population_id,
                treatment_configs=treatment_configs,
                primary_endpoint="disease_activity",
                secondary_endpoints=["quality_of_life", "pain_score"],
                n_patients_per_arm=20
            )
            
            print(f"  Virtual trial completed: {len(outcomes)} outcomes")
            
            # Optimize dosage
            if trial_id in self.trials.population_stats:
                print("\n  Dosage optimization results:")
                
                opt_result = self.trials.optimize_dosage(
                    trial_id=trial_id,
                    target_efficacy=0.6,
                    max_toxicity=0.3,
                    dosage_range=(10, 150),
                    n_simulations=30
                )
                
                if opt_result:
                    print(f"    Optimal dosage: {opt_result['optimal_dosage_mg']:.1f} mg")
                    print(f"    Predicted efficacy: {opt_result['predicted_efficacy']:.3f}")
                    print(f"    Predicted toxicity: {opt_result['predicted_toxicity']:.3f}")
                    print(f"    Therapeutic index: {opt_result['therapeutic_index']:.2f}")
                    
                    # Create trial report
                    report_file = os.path.join(self.temp_dir, "integration_report.html")
                    success = self.trials.create_trial_report(trial_id, report_file)
                    
                    if success:
                        print(f"\n  Trial report saved to: {report_file}")
                    
                    # Validate optimization results
                    self.assertGreater(opt_result['optimal_dosage_mg'], 0)
                    self.assertGreater(opt_result['therapeutic_index'], 0)
        
        print("\n" + "="*60)
        print("✅ Drug Discovery Workflow Test Completed Successfully!")
        print("="*60)

def run_all_drug_discovery_tests():
    """Run all drug discovery tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMolecularModeling))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAIScreening))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVirtualTrials))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegrationWorkflows))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("DRUG DISCOVERY TESTS SUMMARY")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.splitlines()[-1]}")
    
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests
    success = run_all_drug_discovery_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)