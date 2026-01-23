#!/usr/bin/env python3
"""
Unit and Integration Tests for Pharma Workflows
Comprehensive test suite covering lab automation, bioinformatics, and compliance
"""

import unittest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pharma_workflows.lab_automation import (
    LabAutomationManager,
    LabAutomationProtocol,
    ProtocolStep,
    ExperimentRecord,
    TecanEVO
)
from pharma_workflows.bioinformatics_pipeline import (
    BioinformaticsPipeline,
    SequenceData,
    VariantData,
    OmicsDataType,
    DataSource
)
from pharma_workflows.compliance_logging import (
    ComplianceLogger,
    ComplianceStandard,
    EventSeverity,
    ElectronicRecord,
    AuditTrail
)

class TestProtocolStep(unittest.TestCase):
    """Test ProtocolStep model"""
    
    def test_valid_protocol_step(self):
        """Test creating a valid protocol step"""
        step = ProtocolStep(
            step_id="aspirate_001",
            action="pipette",
            parameters={
                "source_plate": "MasterMix_Plate",
                "source_well": "A1",
                "volume_ul": 100
            },
            duration_seconds=5.0,
            notes="Aspirate master mix"
        )
        
        self.assertEqual(step.step_id, "aspirate_001")
        self.assertEqual(step.action, "pipette")
        self.assertEqual(step.parameters["volume_ul"], 100)
        self.assertEqual(step.duration_seconds, 5.0)
    
    def test_invalid_action(self):
        """Test validation of invalid action"""
        with self.assertRaises(ValueError):
            ProtocolStep(
                step_id="invalid_step",
                action="invalid_action",  # This should raise ValueError
                parameters={}
            )
    
    def test_dependencies(self):
        """Test step dependencies"""
        step1 = ProtocolStep(step_id="step1", action="pipette", parameters={})
        step2 = ProtocolStep(
            step_id="step2",
            action="incubate",
            parameters={},
            dependencies=["step1"]
        )
        
        self.assertEqual(step2.dependencies, ["step1"])
        self.assertTrue("step1" in step2.dependencies)

class TestLabAutomationProtocol(unittest.TestCase):
    """Test LabAutomationProtocol model"""
    
    def setUp(self):
        """Set up test protocol"""
        self.steps = [
            ProtocolStep(step_id="step1", action="pipette", parameters={"volume_ul": 100}),
            ProtocolStep(step_id="step2", action="incubate", parameters={"temperature_c": 37}),
            ProtocolStep(
                step_id="step3",
                action="pipette",
                parameters={"volume_ul": 50},
                dependencies=["step1", "step2"]
            )
        ]
        
        self.protocol = LabAutomationProtocol(
            protocol_id="TEST_PROTOCOL_001",
            name="Test Protocol",
            description="Test protocol for unit testing",
            steps=self.steps,
            compliance_tags=["FDA", "GLP"]
        )
    
    def test_protocol_creation(self):
        """Test protocol creation and properties"""
        self.assertEqual(self.protocol.protocol_id, "TEST_PROTOCOL_001")
        self.assertEqual(self.protocol.name, "Test Protocol")
        self.assertEqual(len(self.protocol.steps), 3)
        self.assertIn("FDA", self.protocol.compliance_tags)
    
    def test_protocol_validation(self):
        """Test protocol validation"""
        is_valid, errors = self.protocol.validate_protocol()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_protocol_with_circular_dependency(self):
        """Test protocol validation with circular dependency"""
        steps_with_circular = [
            ProtocolStep(
                step_id="step1",
                action="pipette",
                parameters={},
                dependencies=["step3"]  # Circular: step1 depends on step3
            ),
            ProtocolStep(
                step_id="step2",
                action="incubate",
                parameters={},
                dependencies=["step1"]
            ),
            ProtocolStep(
                step_id="step3",
                action="pipette",
                parameters={},
                dependencies=["step2"]  # Circular: step3 depends on step2
            )
        ]
        
        protocol = LabAutomationProtocol(
            protocol_id="CIRCULAR_PROTOCOL",
            name="Circular Protocol",
            steps=steps_with_circular
        )
        
        is_valid, errors = protocol.validate_protocol()
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

class TestLabAutomationManager(unittest.IsolatedAsyncioTestCase):
    """Test LabAutomationManager with async support"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"sqlite:///{self.temp_dir}/test.db"
        
        # Create manager with test database
        self.manager = LabAutomationManager(db_url=self.db_path)
        
        # Create test protocol
        self.protocol = LabAutomationProtocol(
            protocol_id="TEST_ASYNC_PROTOCOL",
            name="Async Test Protocol",
            steps=[
                ProtocolStep(
                    step_id="test_step",
                    action="pipette",
                    parameters={"volume_ul": 100}
                )
            ]
        )
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch('pharma_workflows.lab_automation.aiohttp.ClientSession')
    async def test_robot_registration(self, mock_session_class):
        """Test robot registration"""
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Register a mock robot
        self.manager.register_robot(
            robot_id="test_robot",
            robot_type="tecan_evo",
            base_url="http://test-robot.local",
            api_key="test_key"
        )
        
        self.assertIn("test_robot", self.manager.robots)
        robot = self.manager.robots["test_robot"]
        self.assertEqual(robot.robot_id, "test_robot")
    
    async def test_experiment_recording(self):
        """Test experiment recording to database"""
        # Run a mock experiment
        experiment_id = "TEST_EXP_001"
        
        # Mock robot execution
        with patch.object(self.manager, 'robots', {}):
            # Since no robots are actually registered, this should fail gracefully
            results = await self.manager.run_protocol(
                protocol=self.protocol,
                experiment_id=experiment_id,
                robot_id="non_existent_robot",
                operator="test@example.com"
            )
            
            # Should have failed but still created a record
            self.assertFalse(results["overall_success"])
        
        # Check database record
        report = self.manager.get_experiment_report(experiment_id)
        self.assertIsNotNone(report)
        self.assertEqual(report["experiment_id"], experiment_id)
        self.assertEqual(report["protocol"], "Async Test Protocol")
    
    def test_audit_hash_generation(self):
        """Test audit hash generation for compliance"""
        test_data = {
            "experiment_id": "HASH_TEST",
            "timestamp": "2024-01-01T00:00:00",
            "data": "test data"
        }
        
        secret = "test_secret"
        hash1 = self.manager.create_audit_hash(test_data, secret)
        hash2 = self.manager.create_audit_hash(test_data, secret)
        
        # Same data and secret should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different secret should produce different hash
        hash3 = self.manager.create_audit_hash(test_data, "different_secret")
        self.assertNotEqual(hash1, hash3)
        
        # Different data should produce different hash
        different_data = test_data.copy()
        different_data["data"] = "different data"
        hash4 = self.manager.create_audit_hash(different_data, secret)
        self.assertNotEqual(hash1, hash4)

class TestSequenceData(unittest.TestCase):
    """Test SequenceData model"""
    
    def setUp(self):
        """Set up test sequence"""
        self.dna_sequence = SequenceData(
            sequence_id="TEST_DNA",
            sequence="ATCGATCGATCG",
            sequence_type="DNA",
            description="Test DNA sequence"
        )
        
        self.protein_sequence = SequenceData(
            sequence_id="TEST_PROTEIN",
            sequence="MAGTESTSEQ",
            sequence_type="PROTEIN",
            description="Test protein sequence"
        )
    
    def test_gc_content(self):
        """Test GC content calculation"""
        # Sequence: ATCGATCGATCG has 6 GC out of 12 bases = 50%
        gc_content = self.dna_sequence.gc_content()
        self.assertEqual(gc_content, 50.0)
        
        # Test with empty sequence
        empty_seq = SequenceData(sequence_id="EMPTY", sequence="", sequence_type="DNA")
        self.assertEqual(empty_seq.gc_content(), 0.0)
    
    def test_translation(self):
        """Test DNA to protein translation"""
        # Simple DNA sequence that translates to known protein
        dna_seq = SequenceData(
            sequence_id="TRANSLATE_TEST",
            sequence="ATGGCGAAATAA",  # Start-Met-Ala-Lys-Stop
            sequence_type="DNA"
        )
        
        protein = dna_seq.translate()
        self.assertEqual(protein, "MAK*")  # * represents stop codon
        
        # Test RNA translation
        rna_seq = SequenceData(
            sequence_id="RNA_TEST",
            sequence="AUGGCGAAAUAA",
            sequence_type="RNA"
        )
        
        protein_from_rna = rna_seq.translate()
        self.assertEqual(protein_from_rna, "MAK*")
    
    def test_reverse_complement(self):
        """Test reverse complement calculation"""
        # Sequence: ATCG should have reverse complement: CGAT
        rev_comp = self.dna_sequence.reverse_complement()
        self.assertEqual(rev_comp, "CGATCGATCGAT")
        
        # Test with non-DNA sequence
        with self.assertRaises(ValueError):
            self.protein_sequence.reverse_complement()

class TestBioinformaticsPipeline(unittest.IsolatedAsyncioTestCase):
    """Test BioinformaticsPipeline with async support"""
    
    async def asyncSetUp(self):
        """Set up test pipeline"""
        # Create temporary cache directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create pipeline
        self.pipeline = BioinformaticsPipeline(cache_dir=self.temp_dir)
        
        # Configure mock NCBI
        self.pipeline.configure_ncbi(email="test@example.com")
    
    async def asyncTearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    @patch('pharma_workflows.bioinformatics_pipeline.Entrez.efetch')
    async def test_fetch_sequence_ncbi(self, mock_efetch):
        """Test fetching sequence from NCBI"""
        # Mock NCBI response
        mock_handle = Mock()
        mock_handle.read.return_value = """>test_id Test Sequence\nATCGATCG"
        mock_efetch.return_value = mock_handle
        
        # Mock SeqIO.read
        with patch('pharma_workflows.bioinformatics_pipeline.SeqIO.read') as mock_read:
            mock_record = Mock()
            mock_record.id = "test_id"
            mock_record.seq = "ATCGATCG"
            mock_record.description = "Test Sequence"
            mock_read.return_value = mock_record
            
            # Fetch sequence
            sequence = await self.pipeline.fetch_sequence("test_id", DataSource.NCBI)
            
            self.assertIsNotNone(sequence)
            self.assertEqual(sequence.sequence_id, "test_id")
            self.assertEqual(sequence.sequence, "ATCGATCG")
            self.assertEqual(sequence.description, "Test Sequence")
    
    def test_differential_expression(self):
        """Test differential expression analysis"""
        # Create mock expression data
        np.random.seed(42)
        n_genes = 100
        n_samples = 20
        
        # Generate data with some differentially expressed genes
        data = {}
        for i in range(n_genes):
            if i < 10:  # First 10 genes are differentially expressed
                control_mean = np.random.uniform(5, 10)
                treated_mean = control_mean * 2.0  # 2-fold change
            else:
                control_mean = np.random.uniform(5, 10)
                treated_mean = control_mean * np.random.uniform(0.8, 1.2)
            
            control_vals = np.random.normal(control_mean, 1, n_samples//2)
            treated_vals = np.random.normal(treated_mean, 1, n_samples//2)
            
            gene_id = f"GENE_{i:03d}"
            data[gene_id] = np.concatenate([control_vals, treated_vals])
        
        df = pd.DataFrame(data, index=[f"SAMPLE_{i}" for i in range(n_samples)])
        self.pipeline.expression_data = df
        
        # Define groups
        control_samples = [f"SAMPLE_{i}" for i in range(n_samples//2)]
        treated_samples = [f"SAMPLE_{i}" for i in range(n_samples//2, n_samples)]
        
        # Run differential expression
        de_results = self.pipeline.differential_expression(
            group_a=control_samples,
            group_b=treated_samples,
            method="t-test",
            pvalue_threshold=0.05,
            log2fc_threshold=1.0
        )
        
        # Check results
        self.assertEqual(len(de_results), n_genes)
        
        # First 10 genes should be significant (we made them DE)
        significant_genes = de_results[de_results["significant"]]
        self.assertGreaterEqual(len(significant_genes), 5)  # At least 5 should be significant
        
        # Check that significant genes have high fold change
        for _, row in significant_genes.iterrows():
            self.assertGreaterEqual(abs(row["log2_fold_change"]), 0.5)
    
    def test_pathway_enrichment(self):
        """Test pathway enrichment analysis"""
        # Define test gene list
        gene_list = ["CDK1", "CDK2", "TP53", "KRAS", "CASP3", "BAX"]
        
        # Run enrichment analysis
        enrichment_results = self.pipeline.pathway_enrichment(
            gene_list=gene_list,
            database="KEGG"
        )
        
        # Check results
        self.assertIsInstance(enrichment_results, pd.DataFrame)
        self.assertGreater(len(enrichration_results), 0)
        
        # Should have columns we expect
        expected_columns = ["pathway_id", "pathway_name", "genes_in_pathway", 
                           "n_genes_in_pathway", "enrichment_ratio", "p_value"]
        
        for col in expected_columns:
            self.assertIn(col, enrichment_results.columns)
    
    def test_save_load_analysis(self):
        """Test saving and loading analysis"""
        # Add some test data
        self.pipeline.sequences["test_seq"] = SequenceData(
            sequence_id="test",
            sequence="ATCG",
            sequence_type="DNA"
        )
        
        self.pipeline.variants.append(
            VariantData(
                chromosome="1",
                position=1000,
                reference="A",
                alternate="G"
            )
        )
        
        # Save analysis
        save_path = os.path.join(self.temp_dir, "test_analysis.json")
        success = self.pipeline.save_analysis(save_path, format="json")
        self.assertTrue(success)
        self.assertTrue(os.path.exists(save_path))
        
        # Create new pipeline and load analysis
        new_pipeline = BioinformaticsPipeline()
        load_success = new_pipeline.load_analysis(save_path, format="json")
        
        self.assertTrue(load_success)
        self.assertIn("test_seq", new_pipeline.sequences)
        self.assertEqual(len(new_pipeline.variants), 1)
        self.assertEqual(new_pipeline.variants[0].position, 1000)

class TestVariantData(unittest.TestCase):
    """Test VariantData model"""
    
    def setUp(self):
        """Set up test variant"""
        self.snv = VariantData(
            chromosome="chr1",
            position=1000,
            reference="A",
            alternate="G",
            variant_id="rs123456",
            quality=100.0,
            filter="PASS"
        )
        
        self.indel = VariantData(
            chromosome="chr2",
            position=2000,
            reference="AT",
            alternate="A",
            variant_id="indel_001"
        )
    
    def test_variant_types(self):
        """Test variant type detection"""
        self.assertTrue(self.snv.is_snv())
        self.assertFalse(self.snv.is_indel())
        
        self.assertTrue(self.indel.is_indel())
        self.assertFalse(self.indel.is_snv())
    
    def test_vcf_format(self):
        """Test VCF format generation"""
        vcf_line = self.snv.vcf_format()
        parts = vcf_line.split('\t')
        
        self.assertEqual(len(parts), 8)
        self.assertEqual(parts[0], "chr1")
        self.assertEqual(parts[1], "1000")
        self.assertEqual(parts[2], "rs123456")
        self.assertEqual(parts[3], "A")
        self.assertEqual(parts[4], "G")
        self.assertEqual(parts[5], "100.0")
        self.assertEqual(parts[6], "PASS")

class TestComplianceLogger(unittest.TestCase):
    """Test ComplianceLogger"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"sqlite:///{self.temp_dir}/compliance_test.db"
        
        # Create logger
        self.logger = ComplianceLogger(db_url=self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_event_logging(self):
        """Test basic event logging"""
        event_id = self.logger.log_event(
            event_type="TEST_EVENT",
            user_id="test_user",
            action="Test action performed",
            entity_type="TEST_ENTITY",
            entity_id="test_123",
            severity=EventSeverity.INFO,
            immediate=True  # Force synchronous logging for test
        )
        
        self.assertNotEqual(event_id, "QUEUED")
        
        # Verify event was logged
        with self.logger.session_scope() as session:
            event = session.query(AuditTrail).filter_by(id=event_id).first()
            self.assertIsNotNone(event)
            self.assertEqual(event.user_id, "test_user")
            self.assertEqual(event.event_type, "TEST_EVENT")
    
    def test_electronic_record_creation(self):
        """Test electronic record creation"""
        record_content = {
            "test_field": "test_value",
            "numeric_field": 123
        }
        
        record_id = self.logger.create_electronic_record(
            record_id="TEST_RECORD_001",
            record_type="TEST_TYPE",
            content=record_content,
            created_by="test_creator",
            retention_days=365
        )
        
        self.assertEqual(record_id, "TEST_RECORD_001")
        
        # Verify record was created
        with self.logger.session_scope() as session:
            record = session.query(ElectronicRecord).filter_by(record_id=record_id).first()
            self.assertIsNotNone(record)
            self.assertEqual(record.created_by, "test_creator")
            self.assertEqual(record.content["test_field"], "test_value")
            self.assertIsNotNone(record.digital_signature)
    
    def test_record_update(self):
        """Test electronic record update with audit trail"""
        # First create a record
        initial_content = {"field": "initial"}
        record_id = self.logger.create_electronic_record(
            record_id="UPDATE_TEST",
            record_type="TEST",
            content=initial_content,
            created_by="creator"
        )
        
        # Update the record
        updated_content = {"field": "updated", "new_field": "new_value"}
        success = self.logger.update_electronic_record(
            record_id=record_id,
            new_content=updated_content,
            modified_by="modifier",
            change_reason="Test update"
        )
        
        self.assertTrue(success)
        
        # Verify update
        with self.logger.session_scope() as session:
            record = session.query(ElectronicRecord).filter_by(record_id=record_id).first()
            self.assertEqual(record.version, 2)  # Should be version 2 after update
            self.assertEqual(record.modified_by, "modifier")
            self.assertEqual(record.content["field"], "updated")
    
    def test_digital_signature(self):
        """Test digital signature application"""
        # Create and sign a record
        record_id = self.logger.create_electronic_record(
            record_id="SIGN_TEST",
            record_type="TEST",
            content={"data": "to_sign"},
            created_by="creator"
        )
        
        success = self.logger.sign_record(
            record_id=record_id,
            signer_id="signer",
            signer_role="QA",
            signing_reason="Test signature",
            valid_until=datetime.utcnow() + timedelta(days=365)
        )
        
        self.assertTrue(success)
        
        # Verify signature
        with self.logger.session_scope() as session:
            record = session.query(ElectronicRecord).filter_by(record_id=record_id).first()
            self.assertEqual(record.status, "SIGNED")
            self.assertEqual(record.approved_by, "signer")
            
            # Check signature record exists
            signature = session.query(SignatureRecord).filter_by(record_id=record_id).first()
            self.assertIsNotNone(signature)
            self.assertEqual(signature.signer_id, "signer")
    
    def test_audit_trail_validation(self):
        """Test audit trail validation"""
        # Log some events
        for i in range(5):
            self.logger.log_event(
                event_type=f"EVENT_{i}",
                user_id="test_user",
                action=f"Action {i}",
                immediate=True
            )
        
        # Run validation
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        validation_results = self.logger.validate_audit_trail(start_date, end_date)
        
        # Should have validation results
        self.assertIsInstance(validation_results, pd.DataFrame)
        self.assertGreater(len(validation_results), 0)
        
        # Check columns
        expected_columns = ["event_id", "timestamp", "event_type", "user_id",
                          "checksum_valid", "signature_valid", "overall_valid"]
        
        for col in expected_columns:
            self.assertIn(col, validation_results.columns)
    
    def test_compliance_report(self):
        """Test compliance report generation"""
        # Generate some test data
        for i in range(10):
            self.logger.log_event(
                event_type="TEST" if i % 2 == 0 else "OTHER",
                user_id=f"user_{i % 3}",
                action=f"Test action {i}",
                immediate=True
            )
        
        # Generate report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        report = self.logger.generate_compliance_report(start_date, end_date)
        
        # Check report structure
        self.assertIn("report_period", report)
        self.assertIn("audit_trail_metrics", report)
        self.assertIn("user_activity", report)
        self.assertIn("compliance_violations", report)
        self.assertIn("electronic_records", report)
        self.assertIn("recommendations", report)
        
        # Check metrics
        self.assertGreater(report["audit_trail_metrics"]["total_events"], 0)
        self.assertIn("TEST", report["audit_trail_metrics"]["event_types"])

class TestIntegrationWorkflows(unittest.IsolatedAsyncioTestCase):
    """Integration tests for complete workflows"""
    
    async def asyncSetUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create all three systems
        self.lab_manager = LabAutomationManager(
            db_url=f"sqlite:///{self.temp_dir}/lab.db"
        )
        
        self.bio_pipeline = BioinformaticsPipeline(
            cache_dir=os.path.join(self.temp_dir, "bio_cache")
        )
        
        self.compliance_logger = ComplianceLogger(
            db_url=f"sqlite:///{self.temp_dir}/compliance.db"
        )
    
    async def asyncTearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir)
    
    async def test_complete_drug_discovery_workflow(self):
        """
        Test complete workflow from lab automation to bioinformatics to compliance
        """
        # Step 1: Lab Automation - Create and validate protocol
        protocol = LabAutomationProtocol(
            protocol_id="INTEGRATION_TEST_PROTOCOL",
            name="Integration Test Protocol",
            steps=[
                ProtocolStep(
                    step_id="compound_addition",
                    action="pipette",
                    parameters={
                        "compound_id": "TEST_COMPOUND_001",
                        "volume_ul": 50,
                        "concentration_um": 10
                    }
                ),
                ProtocolStep(
                    step_id="incubation",
                    action="incubate",
                    parameters={
                        "temperature_c": 37.0,
                        "duration_minutes": 60
                    },
                    dependencies=["compound_addition"]
                )
            ],
            compliance_tags=["FDA", "GLP"]
        )
        
        # Validate protocol
        is_valid, errors = protocol.validate_protocol()
        self.assertTrue(is_valid, f"Protocol validation failed: {errors}")
        
        # Step 2: Bioinformatics - Analyze molecular data
        # Create mock sequence data for target protein
        target_protein = SequenceData(
            sequence_id="TARGET_PROTEIN",
            sequence="MAGTESTSEQUENCEFORINTEGRATIONTEST",
            sequence_type="PROTEIN",
            description="Target protein for drug binding"
        )
        
        self.bio_pipeline.sequences[target_protein.sequence_id] = target_protein
        
        # Analyze protein properties
        protein_length = len(target_protein.sequence)
        self.assertGreater(protein_length, 0)
        
        # Step 3: Compliance - Log the entire workflow
        # Log protocol creation
        protocol_event_id = self.compliance_logger.log_event(
            event_type="PROTOCOL_CREATION",
            user_id="researcher@company.com",
            action="Created drug screening protocol",
            entity_type="PROTOCOL",
            entity_id=protocol.protocol_id,
            after_state=protocol.dict(),
            severity=EventSeverity.INFO,
            immediate=True
        )
        
        self.assertNotEqual(protocol_event_id, "QUEUED")
        
        # Create electronic record for experiment
        experiment_record = {
            "protocol_id": protocol.protocol_id,
            "compound_id": "TEST_COMPOUND_001",
            "target_protein": target_protein.sequence_id,
            "parameters": {
                "concentration_um": 10,
                "incubation_time_min": 60,
                "temperature_c": 37.0
            },
            "expected_analysis": "Binding affinity measurement"
        }
        
        record_id = self.compliance_logger.create_electronic_record(
            record_id=f"EXP_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            record_type="DRUG_SCREENING_EXPERIMENT",
            content=experiment_record,
            created_by="researcher@company.com",
            retention_days=365 * 10
        )
        
        # Step 4: Simulate experiment results
        # Mock results from lab automation
        experiment_results = {
            "experiment_id": record_id,
            "results": {
                "binding_affinity_kd_nm": 150.5,
                "specificity_score": 0.85,
                "toxicity_indicator": "LOW",
                "efficacy_prediction": "PROMISING"
            },
            "quality_metrics": {
                "signal_to_noise": 12.5,
                "cv_percent": 8.2,
                "z_prime": 0.65
            }
        }
        
        # Update electronic record with results
        updated_content = experiment_record.copy()
        updated_content["results"] = experiment_results["results"]
        updated_content["quality_metrics"] = experiment_results["quality_metrics"]
        updated_content["completion_timestamp"] = datetime.utcnow().isoformat()
        
        update_success = self.compliance_logger.update_electronic_record(
            record_id=record_id,
            new_content=updated_content,
            modified_by="researcher@company.com",
            change_reason="Added experiment results"
        )
        
        self.assertTrue(update_success)
        
        # Step 5: Sign the results
        sign_success = self.compliance_logger.sign_record(
            record_id=record_id,
            signer_id="qa_manager@company.com",
            signer_role="QUALITY_ASSURANCE",
            signing_reason="Verified experiment results meet quality standards",
            valid_until=datetime.utcnow() + timedelta(days=365 * 5)
        )
        
        self.assertTrue(sign_success)
        
        # Step 6: Generate compliance report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        report = self.compliance_logger.generate_compliance_report(start_date, end_date)
        
        # Verify report contains our activity
        self.assertGreater(report["audit_trail_metrics"]["total_events"], 0)
        self.assertEqual(report["electronic_records"]["signed_records"], 1)
        
        # Step 7: Bioinformatics - Pathway analysis on results
        # Create mock gene expression data based on results
        if experiment_results["results"]["efficacy_prediction"] == "PROMISING":
            # Simulate analysis of affected pathways
            affected_genes = ["CDK1", "TP53", "EGFR", "MAPK1"]
            
            enrichment = self.bio_pipeline.pathway_enrichment(
                gene_list=affected_genes,
                database="KEGG"
            )
            
            self.assertGreater(len(enrichment), 0)
            
            # Log bioinformatics analysis
            bio_event_id = self.compliance_logger.log_event(
                event_type="BIOINFORMATICS_ANALYSIS",
                user_id="bioinformatician@company.com",
                action=f"Pathway analysis for {record_id}",
                entity_type="ANALYSIS",
                entity_id=record_id,
                after_state={
                    "affected_genes": affected_genes,
                    "top_pathways": enrichment["pathway_name"].tolist()[:3]
                },
                severity=EventSeverity.INFO,
                immediate=True
            )
            
            self.assertNotEqual(bio_event_id, "QUEUED")
        
        print(f"\n✅ Integration test completed successfully!")
        print(f"   Experiment Record: {record_id}")
        print(f"   Target Protein: {target_protein.sequence_id}")
        print(f"   Compliance Events: {report['audit_trail_metrics']['total_events']}")

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProtocolStep))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLabAutomationProtocol))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLabAutomationManager))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSequenceData))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBioinformaticsPipeline))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVariantData))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestComplianceLogger))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegrationWorkflows))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)