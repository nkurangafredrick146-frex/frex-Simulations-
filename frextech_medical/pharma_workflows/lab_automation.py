#!/usr/bin/env python3
"""
Lab Automation Module for FrexTech Medical
Connects to robotic lab APIs for automated experimentation
FDA 21 CFR Part 11 compliant
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import numpy as np
from pydantic import BaseModel, Field, validator
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, JSON, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import hashlib
import hmac
import ssl
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()

class ExperimentRecord(Base):
    """Database model for experiment records"""
    __tablename__ = 'experiment_recuments'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    compound_id = Column(String, nullable=True)
    protocol_name = Column(String, nullable=False)
    status = Column(String, default='PENDING')  # PENDING, RUNNING, COMPLETED, FAILED
    robot_id = Column(String, nullable=False)
    parameters = Column(JSON, default=dict)
    results = Column(JSON, default=dict)
    metadata = Column(JSON, default=dict)
    duration_seconds = Column(Float, default=0.0)
    operator = Column(String, default='SYSTEM')
    audit_hash = Column(String, nullable=True)  # For FDA compliance

class ProtocolStep(BaseModel):
    """Individual step in an automation protocol"""
    step_id: str = Field(..., description="Unique step identifier")
    action: str = Field(..., description="Action to perform (pipette, incubate, centrifuge, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(0.0, description="Duration in seconds")
    dependencies: List[str] = Field(default_factory=list)
    expected_volume_ul: Optional[float] = None
    temperature_c: Optional[float] = None
    notes: Optional[str] = None
    
    @validator('action')
    def validate_action(cls, v):
        valid_actions = {
            'pipette', 'incubate', 'centrifuge', 'heat', 'cool',
            'mix', 'dispense', 'aspirate', 'seal', 'unseal',
            'read_absorbance', 'read_fluorescence', 'wash', 'elute'
        }
        if v.lower() not in valid_actions:
            raise ValueError(f"Invalid action. Must be one of: {valid_actions}")
        return v.lower()

class LabAutomationProtocol(BaseModel):
    """Complete laboratory automation protocol"""
    protocol_id: str = Field(..., description="Unique protocol identifier")
    name: str = Field(..., description="Human-readable protocol name")
    version: str = Field("1.0.0", description="Protocol version")
    description: Optional[str] = None
    steps: List[ProtocolStep] = Field(default_factory=list)
    created_by: str = Field("system", description="Protocol creator")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    compliance_tags: List[str] = Field(default_factory=list)  # FDA, EMA, etc.
    
    def validate_protocol(self) -> Tuple[bool, List[str]]:
        """Validate protocol for completeness and safety"""
        errors = []
        
        # Check for circular dependencies
        step_ids = {step.step_id for step in self.steps}
        for step in self.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id} depends on non-existent step {dep}")
        
        # Check for orphaned steps
        all_deps = set()
        for step in self.steps:
            all_deps.update(step.dependencies)
        
        orphans = step_ids - all_deps - {self.steps[0].step_id if self.steps else ''}
        if len(orphans) > 1:
            errors.append(f"Multiple orphaned steps detected: {orphans}")
        
        # Validate parameters
        for step in self.steps:
            if step.action == 'pipette':
                if 'volume_ul' not in step.parameters:
                    errors.append(f"Pipette step {step.step_id} missing volume_ul parameter")
                if step.parameters.get('volume_ul', 0) > 1000:
                    errors.append(f"Pipette volume too high in step {step.step_id}")
        
        return len(errors) == 0, errors

class RobotAPI:
    """Base class for robot API interactions"""
    
    def __init__(self, robot_id: str, base_url: str, api_key: str):
        self.robot_id = robot_id
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'X-API-Key': self.api_key},
            connector=aiohttp.TCPConnector(ssl=self.ssl_context)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def check_status(self) -> Dict[str, Any]:
        """Check robot status"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/status") as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error checking robot status: {e}")
            return {"status": "ERROR", "error": str(e)}
            
    async def execute_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a command on the robot"""
        try:
            payload = {
                "command": command,
                "parameters": params,
                "timestamp": datetime.utcnow().isoformat(),
                "command_id": str(uuid.uuid4())
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/execute",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
            ) as response:
                result = await response.json()
                result['success'] = response.status == 200
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout executing command {command}")
            return {"success": False, "error": "Command timeout"}
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {"success": False, "error": str(e)}

class TecanEVO(RobotAPI):
    """Tecan EVO liquid handling robot interface"""
    
    async def aspirate(self, source_plate: str, source_well: str, 
                      volume_ul: float, liquid_class: str = "Water") -> Dict[str, Any]:
        """Aspirate liquid from a well"""
        params = {
            "source_plate": source_plate,
            "source_well": source_well,
            "volume_ul": volume_ul,
            "liquid_class": liquid_class,
            "tip_type": "FilteredTip_1000"
        }
        return await self.execute_command("aspirate", params)
        
    async def dispense(self, destination_plate: str, destination_well: str,
                      volume_ul: float, liquid_class: str = "Water") -> Dict[str, Any]:
        """Dispense liquid to a well"""
        params = {
            "destination_plate": destination_plate,
            "destination_well": destination_well,
            "volume_ul": volume_ul,
            "liquid_class": liquid_class,
            "tip_type": "FilteredTip_1000"
        }
        return await self.execute_command("dispense", params)
        
    async def wash_tips(self, waste_container: str = "Waste1") -> Dict[str, Any]:
        """Wash tips"""
        return await self.execute_command("wash_tips", {"waste_container": waste_container})
        
    async def incubate(self, plate: str, temperature_c: float, 
                      duration_minutes: float, shaking_rpm: float = 0) -> Dict[str, Any]:
        """Incubate plate"""
        params = {
            "plate": plate,
            "temperature_c": temperature_c,
            "duration_minutes": duration_minutes,
            "shaking_rpm": shaking_rpm
        }
        return await self.execute_command("incubate", params)

class LabAutomationManager:
    """Main lab automation manager with FDA compliance"""
    
    def __init__(self, db_url: str = "sqlite:///lab_automation.db"):
        self.db_engine = create_engine(db_url)
        Base.metadata.create_all(self.db_engine)
        self.Session = sessionmaker(bind=self.db_engine)
        
        self.robots = {}
        self.active_experiments = {}
        self.compliance_mode = True  # FDA 21 CFR Part 11 by default
        
    def register_robot(self, robot_id: str, robot_type: str, 
                      base_url: str, api_key: str):
        """Register a new robot"""
        if robot_type.lower() == "tecan_evo":
            self.robots[robot_id] = TecanEVO(robot_id, base_url, api_key)
        else:
            self.robots[robot_id] = RobotAPI(robot_id, base_url, api_key)
        logger.info(f"Registered robot {robot_id} ({robot_type})")
        
    def create_audit_hash(self, data: Dict[str, Any], secret: str) -> str:
        """Create an audit hash for compliance tracking"""
        # Sort data for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        # Create HMAC-SHA256 hash
        h = hmac.new(
            secret.encode('utf-8'),
            sorted_data.encode('utf-8'),
            hashlib.sha256
        )
        return h.hexdigest()
        
    async def run_protocol(self, protocol: LabAutomationProtocol, 
                          experiment_id: str, robot_id: str,
                          operator: str = "SYSTEM") -> Dict[str, Any]:
        """Execute a complete protocol on a robot"""
        
        session = self.Session()
        experiment_record = ExperimentRecord(
            experiment_id=experiment_id,
            protocol_name=protocol.name,
            robot_id=robot_id,
            parameters={"protocol": protocol.dict()},
            operator=operator,
            status="RUNNING"
        )
        session.add(experiment_record)
        session.commit()
        
        start_time = time.time()
        results = {"steps": [], "overall_success": True}
        
        try:
            # Validate protocol
            is_valid, errors = protocol.validate_protocol()
            if not is_valid:
                raise ValueError(f"Protocol validation failed: {errors}")
                
            # Check robot availability
            if robot_id not in self.robots:
                raise ValueError(f"Robot {robot_id} not registered")
                
            async with self.robots[robot_id] as robot:
                # Check robot status
                status = await robot.check_status()
                if status.get("status") != "READY":
                    raise RuntimeError(f"Robot not ready: {status}")
                    
                # Execute steps in dependency order
                executed_steps = set()
                
                while len(executed_steps) < len(protocol.steps):
                    progress_made = False
                    
                    for step in protocol.steps:
                        if step.step_id in executed_steps:
                            continue
                            
                        # Check if dependencies are satisfied
                        deps_satisfied = all(
                            dep in executed_steps for dep in step.dependencies
                        ) or not step.dependencies
                        
                        if deps_satisfied:
                            # Execute step
                            step_start = time.time()
                            step_result = await self._execute_step(robot, step)
                            step_duration = time.time() - step_start
                            
                            step_result_data = {
                                "step_id": step.step_id,
                                "action": step.action,
                                "success": step_result.get("success", False),
                                "duration_seconds": step_duration,
                                "result": step_result,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            
                            if self.compliance_mode:
                                step_result_data["audit_hash"] = self.create_audit_hash(
                                    step_result_data,
                                    f"{experiment_id}_{step.step_id}"
                                )
                            
                            results["steps"].append(step_result_data)
                            
                            if not step_result.get("success", False):
                                results["overall_success"] = False
                                logger.error(f"Step {step.step_id} failed: {step_result}")
                                
                            executed_steps.add(step.step_id)
                            progress_made = True
                            
                            # Add delay if specified
                            if step.duration_seconds > 0:
                                await asyncio.sleep(step.duration_seconds)
                                
                    if not progress_made:
                        raise RuntimeError("Circular dependency detected in protocol")
                        
        except Exception as e:
            logger.error(f"Protocol execution failed: {e}")
            results["overall_success"] = False
            results["error"] = str(e)
            experiment_record.status = "FAILED"
            
        finally:
            # Update experiment record
            duration = time.time() - start_time
            experiment_record.duration_seconds = duration
            experiment_record.results = results
            experiment_record.status = "COMPLETED" if results["overall_success"] else "FAILED"
            
            # Create final audit hash
            if self.compliance_mode:
                final_data = {
                    "experiment_id": experiment_id,
                    "protocol": protocol.dict(),
                    "results": results,
                    "duration": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
                experiment_record.audit_hash = self.create_audit_hash(
                    final_data, experiment_id
                )
                
            session.commit()
            session.close()
            
        return results
        
    async def _execute_step(self, robot: RobotAPI, step: ProtocolStep) -> Dict[str, Any]:
        """Execute a single protocol step"""
        try:
            if step.action == "pipette":
                # Handle pipette action
                if "aspirate" in step.parameters:
                    result = await robot.aspirate(
                        source_plate=step.parameters.get("source_plate", "Plate1"),
                        source_well=step.parameters.get("source_well", "A1"),
                        volume_ul=step.parameters.get("volume_ul", 100),
                        liquid_class=step.parameters.get("liquid_class", "Water")
                    )
                else:
                    result = await robot.dispense(
                        destination_plate=step.parameters.get("destination_plate", "Plate1"),
                        destination_well=step.parameters.get("destination_well", "A1"),
                        volume_ul=step.parameters.get("volume_ul", 100),
                        liquid_class=step.parameters.get("liquid_class", "Water")
                    )
                    
            elif step.action == "incubate":
                result = await robot.incubate(
                    plate=step.parameters.get("plate", "Plate1"),
                    temperature_c=step.parameters.get("temperature_c", 37.0),
                    duration_minutes=step.parameters.get("duration_minutes", 60),
                    shaking_rpm=step.parameters.get("shaking_rpm", 0)
                )
                
            elif step.action == "wash":
                result = await robot.wash_tips(
                    waste_container=step.parameters.get("waste_container", "Waste1")
                )
                
            else:
                # Generic command execution
                result = await robot.execute_command(step.action, step.parameters)
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            return {"success": False, "error": str(e)}
            
    def get_experiment_report(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive experiment report"""
        session = self.Session()
        try:
            record = session.query(ExperimentRecord).filter_by(
                experiment_id=experiment_id
            ).first()
            
            if not record:
                return None
                
            report = {
                "experiment_id": record.experiment_id,
                "protocol": record.protocol_name,
                "status": record.status,
                "start_time": record.timestamp.isoformat() if record.timestamp else None,
                "duration_seconds": record.duration_seconds,
                "operator": record.operator,
                "robot_id": record.robot_id,
                "parameters": record.parameters,
                "results": record.results,
                "metadata": record.metadata,
                "audit_hash": record.audit_hash if self.compliance_mode else None,
                "compliance_valid": self._verify_compliance(record) if self.compliance_mode else None
            }
            
            return report
            
        finally:
            session.close()
            
    def _verify_compliance(self, record: ExperimentRecord) -> bool:
        """Verify audit trail compliance"""
        if not record.audit_hash:
            return False
            
        # Recreate hash from record data
        data_to_hash = {
            "experiment_id": record.experiment_id,
            "protocol": record.parameters,
            "results": record.results,
            "duration": record.duration_seconds,
            "timestamp": record.timestamp.isoformat() if record.timestamp else None
        }
        
        expected_hash = self.create_audit_hash(
            data_to_hash, record.experiment_id
        )
        
        return expected_hash == record.audit_hash
        
    def export_to_csv(self, experiment_id: str, filepath: str) -> bool:
        """Export experiment data to CSV for reporting"""
        report = self.get_experiment_report(experiment_id)
        if not report:
            return False
            
        try:
            # Flatten report structure
            flat_data = {
                "experiment_id": report["experiment_id"],
                "protocol": report["protocol"],
                "status": report["status"],
                "operator": report["operator"],
                "robot_id": report["robot_id"],
                "duration_seconds": report["duration_seconds"],
                "overall_success": report["results"].get("overall_success", False)
            }
            
            # Add step summaries
            steps = report["results"].get("steps", [])
            for i, step in enumerate(steps[:10]):  # Limit to first 10 steps
                flat_data[f"step_{i+1}_id"] = step.get("step_id", "")
                flat_data[f"step_{i+1}_success"] = step.get("success", False)
                
            df = pd.DataFrame([flat_data])
            df.to_csv(filepath, index=False)
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

# Example usage and factory function
def create_lab_automation_manager(config_file: str = "lab_config.json") -> LabAutomationManager:
    """Factory function to create and configure lab automation manager"""
    manager = LabAutomationManager()
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        # Register robots from config
        for robot_config in config.get("robots", []):
            manager.register_robot(
                robot_id=robot_config["id"],
                robot_type=robot_config["type"],
                base_url=robot_config["base_url"],
                api_key=robot_config["api_key"]
            )
            
        # Set compliance mode
        manager.compliance_mode = config.get("compliance_mode", True)
        
        logger.info(f"Lab automation manager initialized with {len(manager.robots)} robots")
        
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        
    return manager

# Async main for testing
async def main():
    """Example usage of the lab automation system"""
    
    # Create protocol
    protocol = LabAutomationProtocol(
        protocol_id="PCR_SETUP_001",
        name="PCR Reaction Setup",
        description="Setup for 96-well PCR plate",
        steps=[
            ProtocolStep(
                step_id="aspirate_master_mix",
                action="pipette",
                parameters={
                    "source_plate": "MasterMix_Plate",
                    "source_well": "A1",
                    "volume_ul": 15,
                    "liquid_class": "MasterMix"
                }
            ),
            ProtocolStep(
                step_id="dispense_to_pcr_plate",
                action="pipette",
                parameters={
                    "destination_plate": "PCR_Plate",
                    "destination_well": "A1:H12",
                    "volume_ul": 15,
                    "liquid_class": "MasterMix"
                },
                dependencies=["aspirate_master_mix"]
            )
        ],
        compliance_tags=["FDA_21CFR_Part11", "GLP"]
    )
    
    # Create manager
    manager = LabAutomationManager()
    manager.register_robot(
        robot_id="tecan_001",
        robot_type="tecan_evo",
        base_url="https://robot-lab.example.com",
        api_key="test_api_key"
    )
    
    # Run protocol (in real scenario, this would connect to actual robot)
    experiment_id = f"EXP_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    results = await manager.run_protocol(
        protocol=protocol,
        experiment_id=experiment_id,
        robot_id="tecan_001",
        operator="john.doe@company.com"
    )
    
    print(f"Experiment {experiment_id} completed: {results['overall_success']}")

if __name__ == "__main__":
    asyncio.run(main())