#!/usr/bin/env python3
"""
Compliance Logging System for FrexTech Medical
FDA 21 CFR Part 11, EMA Annex 11, HIPAA compliant
"""

import logging
import json
import hashlib
import hmac
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import ssl
import base64
import cryptography
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, Boolean, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError
import pandas as pd
from pathlib import Path
import threading
import queue
import asyncio
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

class AuditTrail(Base):
    """FDA 21 CFR Part 11 compliant audit trail"""
    __tablename__ = 'audit_trail'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    event_subtype = Column(String(100))
    user_id = Column(String(100), nullable=False, index=True)
    user_role = Column(String(100))
    action = Column(Text, nullable=False)
    entity_type = Column(String(100))
    entity_id = Column(String(100), index=True)
    before_state = Column(JSON)  # State before change
    after_state = Column(JSON)   # State after change
    digital_signature = Column(LargeBinary)  # Cryptographic signature
    checksum = Column(String(64))  # SHA-256 hash of record
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    session_id = Column(String(100))
    compliance_standard = Column(String(100), default="FDA_21CFR_Part11")
    validated = Column(Boolean, default=False)
    validation_timestamp = Column(DateTime)
    metadata = Column(JSON)

class ElectronicRecord(Base):
    """Electronic records for FDA compliance"""
    __tablename__ = 'electronic_records'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    record_id = Column(String(100), nullable=False, unique=True, index=True)
    record_type = Column(String(100), nullable=False, index=True)
    version = Column(Integer, nullable=False, default=1)
    content = Column(JSON, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA-256 of content
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    modified_by = Column(String(100))
    modified_at = Column(DateTime)
    approved_by = Column(String(100))
    approved_at = Column(DateTime)
    status = Column(String(50), default="DRAFT")  # DRAFT, PENDING_REVIEW, APPROVED, REJECTED, ARCHIVED
    digital_signature = Column(LargeBinary)
    parent_record_id = Column(String(100), index=True)
    metadata = Column(JSON)
    retention_period_days = Column(Integer, default=365 * 10)  # 10 years default
    deletion_scheduled = Column(DateTime)

class SignatureRecord(Base):
    """Digital signatures for electronic records"""
    __tablename__ = 'digital_signatures'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    record_id = Column(String(100), nullable=False, index=True)
    record_version = Column(Integer, nullable=False)
    signer_id = Column(String(100), nullable=False)
    signer_role = Column(String(100), nullable=False)
    signature_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    signature_data = Column(LargeBinary, nullable=False)  # Encrypted signature
    signature_hash = Column(String(64), nullable=False)  # Hash of signature data
    certificate_thumbprint = Column(String(64))  # SHA-1 thumbprint of certificate
    signing_reason = Column(String(500))
    valid_until = Column(DateTime)
    revoked = Column(Boolean, default=False)
    revocation_reason = Column(String(500))
    metadata = Column(JSON)

class ComplianceStandard(Enum):
    """Supported compliance standards"""
    FDA_21CFR_PART11 = "FDA_21CFR_Part11"
    EMA_ANNEX11 = "EMA_Annex11"
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    GLP = "GLP"
    GMP = "GMP"
    ISO_13485 = "ISO_13485"
    ISO_27001 = "ISO_27001"

class EventSeverity(Enum):
    """Event severity levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"

@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    standard: ComplianceStandard
    description: str
    requirement: str
    validation_logic: str  # Python code or regex pattern
    severity: EventSeverity = EventSeverity.WARNING
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class CryptographyManager:
    """Cryptography manager for digital signatures and encryption"""
    
    def __init__(self, key_storage_path: str = "./crypto_keys"):
        self.key_storage = Path(key_storage_path)
        self.key_storage.mkdir(exist_ok=True, mode=0o700)
        
        # Load or generate keys
        self.private_key = self._load_or_generate_private_key()
        self.public_key = self.private_key.public_key()
        
        # Session keys for symmetric encryption
        self.session_keys = {}
        
    def _load_or_generate_private_key(self) -> rsa.RSAPrivateKey:
        """Load existing private key or generate new one"""
        key_file = self.key_storage / "private_key.pem"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                logger.info("Loaded existing private key")
                return private_key
            except Exception as e:
                logger.warning(f"Failed to load private key: {e}")
        
        # Generate new key
        logger.info("Generating new RSA private key (2048-bit)")
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Save key
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(key_file, 'wb') as f:
            f.write(pem)
        
        # Set proper permissions
        key_file.chmod(0o600)
        
        return private_key
    
    def generate_signature(self, data: bytes, 
                          signer_id: str = "SYSTEM") -> Tuple[bytes, str]:
        """Generate digital signature for data"""
        try:
            # Hash the data
            data_hash = hashlib.sha256(data).digest()
            
            # Sign the hash
            signature = self.private_key.sign(
                data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Create signature metadata
            signature_meta = {
                "signer": signer_id,
                "timestamp": datetime.utcnow().isoformat(),
                "algorithm": "RSA-PSS-SHA256",
                "key_fingerprint": self.get_key_fingerprint()
            }
            
            # Combine signature and metadata
            signature_package = {
                "signature": base64.b64encode(signature).decode('ascii'),
                "metadata": signature_meta
            }
            
            signature_bytes = json.dumps(signature_package).encode('utf-8')
            signature_hash = hashlib.sha256(signature_bytes).hexdigest()
            
            return signature_bytes, signature_hash
            
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature_bytes: bytes) -> bool:
        """Verify digital signature"""
        try:
            # Parse signature package
            signature_package = json.loads(signature_bytes.decode('utf-8'))
            signature = base64.b64decode(signature_package['signature'])
            
            # Hash the original data
            data_hash = hashlib.sha256(data).digest()
            
            # Verify signature
            self.public_key.verify(
                signature,
                data_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def encrypt_data(self, data: bytes, key_id: str = "default") -> bytes:
        """Encrypt data using AES-256-GCM"""
        try:
            # Generate or retrieve session key
            if key_id not in self.session_keys:
                self.session_keys[key_id] = {
                    'key': os.urandom(32),  # 256-bit key
                    'nonce': os.urandom(12)  # 96-bit nonce
                }
            
            session_key = self.session_keys[key_id]
            
            # Encrypt data
            cipher = Cipher(
                algorithms.AES(session_key['key']),
                modes.GCM(session_key['nonce']),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Return nonce + ciphertext + tag
            encrypted_data = session_key['nonce'] + encryptor.tag + ciphertext
            
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, key_id: str = "default") -> bytes:
        """Decrypt data using AES-256-GCM"""
        try:
            if key_id not in self.session_keys:
                raise ValueError(f"Session key {key_id} not found")
            
            session_key = self.session_keys[key_id]
            
            # Extract nonce, tag, and ciphertext
            nonce = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            # Decrypt
            cipher = Cipher(
                algorithms.AES(session_key['key']),
                modes.GCM(nonce, tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def get_key_fingerprint(self) -> str:
        """Get SHA-1 fingerprint of public key"""
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return hashlib.sha1(public_bytes).hexdigest()

class ComplianceLogger:
    """Main compliance logging system"""
    
    def __init__(self, db_url: str = "sqlite:///compliance.db",
                 crypto_manager: Optional[CryptographyManager] = None):
        
        # Database setup
        self.db_engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.db_engine)
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        
        # Cryptography
        self.crypto = crypto_manager or CryptographyManager()
        
        # Compliance rules
        self.rules = self._load_default_rules()
        
        # Event queue for async processing
        self.event_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_event_queue, daemon=True)
        self.processing_thread.start()
        
        # Audit trail validation scheduler
        self.validation_scheduler = threading.Thread(target=self._run_validation_scheduler, daemon=True)
        self.validation_scheduler.start()
        
        # Retention policy manager
        self.retention_manager = threading.Thread(target=self._run_retention_manager, daemon=True)
        self.retention_manager.start()
        
        logger.info("Compliance logging system initialized")
    
    def _load_default_rules(self) -> Dict[str, ComplianceRule]:
        """Load default compliance rules"""
        rules = {}
        
        # FDA 21 CFR Part 11 rules
        rules["FDA-001"] = ComplianceRule(
            rule_id="FDA-001",
            standard=ComplianceStandard.FDA_21CFR_PART11,
            description="Electronic Signature Requirement",
            requirement="All electronic records must be signed with a secure digital signature",
            validation_logic="hasattr(record, 'digital_signature') and record.digital_signature is not None",
            severity=EventSeverity.CRITICAL
        )
        
        rules["FDA-002"] = ComplianceRule(
            rule_id="FDA-002",
            standard=ComplianceStandard.FDA_21CFR_PART11,
            description="Audit Trail Integrity",
            requirement="All changes to electronic records must be captured in an audit trail",
            validation_logic="audit_trail_count > 0",
            severity=EventSeverity.CRITICAL
        )
        
        rules["FDA-003"] = ComplianceRule(
            rule_id="FDA-003",
            standard=ComplianceStandard.FDA_21CFR_PART11,
            description="Record Retention",
            requirement="Records must be retained for specified period",
            validation_logic="retention_days >= required_retention_days",
            severity=EventSeverity.WARNING
        )
        
        # HIPAA rules
        rules["HIPAA-001"] = ComplianceRule(
            rule_id="HIPAA-001",
            standard=ComplianceStandard.HIPAA,
            description="PHI Encryption",
            requirement="Protected Health Information must be encrypted at rest",
            validation_logic="is_encrypted == True",
            severity=EventSeverity.SECURITY
        )
        
        return rules
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def log_event(self, event_type: str, user_id: str, action: str,
                  entity_type: Optional[str] = None,
                  entity_id: Optional[str] = None,
                  before_state: Optional[Dict] = None,
                  after_state: Optional[Dict] = None,
                  severity: EventSeverity = EventSeverity.INFO,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  session_id: Optional[str] = None,
                  metadata: Optional[Dict] = None,
                  immediate: bool = False) -> str:
        """
        Log a compliance event with audit trail
        
        Args:
            event_type: Type of event (CREATE, UPDATE, DELETE, VIEW, SIGN, etc.)
            user_id: ID of user performing action
            action: Description of action performed
            entity_type: Type of entity affected
            entity_id: ID of entity affected
            before_state: State before change (for UPDATE/DELETE)
            after_state: State after change (for CREATE/UPDATE)
            severity: Event severity level
            ip_address: IP address of user
            user_agent: User agent string
            session_id: Session ID
            metadata: Additional metadata
            immediate: If True, log immediately (blocking)
        
        Returns:
            Event ID
        """
        
        event_data = {
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "before_state": before_state,
            "after_state": after_state,
            "severity": severity.value,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "session_id": session_id,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if immediate:
            return self._log_event_sync(event_data)
        else:
            # Queue for async processing
            self.event_queue.put(event_data)
            return "QUEUED"
    
    def _log_event_sync(self, event_data: Dict[str, Any]) -> str:
        """Synchronous event logging"""
        event_id = str(uuid.uuid4())
        
        with self.session_scope() as session:
            # Create audit trail record
            audit_record = AuditTrail(
                id=event_id,
                event_type=event_data["event_type"],
                user_id=event_data["user_id"],
                action=event_data["action"],
                entity_type=event_data["entity_type"],
                entity_id=event_data["entity_id"],
                before_state=event_data["before_state"],
                after_state=event_data["after_state"],
                ip_address=event_data["ip_address"],
                user_agent=event_data["user_agent"],
                session_id=event_data["session_id"],
                metadata=event_data["metadata"],
                timestamp=datetime.fromisoformat(event_data["timestamp"])
            )
            
            # Calculate checksum
            record_data = json.dumps(asdict(audit_record), sort_keys=True, default=str).encode('utf-8')
            audit_record.checksum = hashlib.sha256(record_data).hexdigest()
            
            # Generate digital signature
            try:
                signature, signature_hash = self.crypto.generate_signature(
                    record_data,
                    signer_id=event_data["user_id"]
                )
                audit_record.digital_signature = signature
            except Exception as e:
                logger.error(f"Failed to generate signature: {e}")
            
            # Save to database
            session.add(audit_record)
            
            # Apply compliance rules
            self._apply_compliance_rules(audit_record)
            
            logger.info(f"Logged compliance event: {event_type} by {user_id}")
        
        return event_id
    
    def _process_event_queue(self):
        """Process events from queue asynchronously"""
        while True:
            try:
                event_data = self.event_queue.get(timeout=1)
                self._log_event_sync(event_data)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
    
    def create_electronic_record(self, record_id: str, record_type: str,
                                content: Dict, created_by: str,
                                metadata: Optional[Dict] = None,
                                retention_days: int = 3650) -> str:
        """
        Create FDA-compliant electronic record
        
        Returns:
            Record ID
        """
        with self.session_scope() as session:
            # Calculate content hash
            content_json = json.dumps(content, sort_keys=True)
            content_hash = hashlib.sha256(content_json.encode('utf-8')).hexdigest()
            
            # Create record
            record = ElectronicRecord(
                record_id=record_id,
                record_type=record_type,
                content=content,
                content_hash=content_hash,
                created_by=created_by,
                created_at=datetime.utcnow(),
                status="DRAFT",
                metadata=metadata or {},
                retention_period_days=retention_days
            )
            
            # Generate digital signature
            record_data = json.dumps(asdict(record), sort_keys=True, default=str).encode('utf-8')
            signature, _ = self.crypto.generate_signature(record_data, created_by)
            record.digital_signature = signature
            
            session.add(record)
            
            # Log creation event
            self.log_event(
                event_type="CREATE",
                user_id=created_by,
                action=f"Created electronic record {record_id}",
                entity_type="ELECTRONIC_RECORD",
                entity_id=record_id,
                after_state=content,
                severity=EventSeverity.INFO,
                immediate=True
            )
            
            logger.info(f"Created electronic record: {record_id}")
        
        return record_id
    
    def update_electronic_record(self, record_id: str, new_content: Dict,
                                modified_by: str, change_reason: str) -> bool:
        """Update electronic record with audit trail"""
        with self.session_scope() as session:
            # Get current record
            record = session.query(ElectronicRecord).filter_by(record_id=record_id).first()
            if not record:
                logger.error(f"Record {record_id} not found")
                return False
            
            # Store previous state for audit
            previous_content = record.content.copy()
            previous_version = record.version
            
            # Update record
            record.content = new_content
            record.modified_by = modified_by
            record.modified_at = datetime.utcnow()
            record.version += 1
            
            # Update content hash
            content_json = json.dumps(new_content, sort_keys=True)
            record.content_hash = hashlib.sha256(content_json.encode('utf-8')).hexdigest()
            
            # Generate new signature
            record_data = json.dumps(asdict(record), sort_keys=True, default=str).encode('utf-8')
            signature, _ = self.crypto.generate_signature(record_data, modified_by)
            record.digital_signature = signature
            
            # Log update event
            self.log_event(
                event_type="UPDATE",
                user_id=modified_by,
                action=f"Updated electronic record {record_id}: {change_reason}",
                entity_type="ELECTRONIC_RECORD",
                entity_id=record_id,
                before_state=previous_content,
                after_state=new_content,
                severity=EventSeverity.INFO,
                immediate=True
            )
            
            logger.info(f"Updated electronic record: {record_id} v{record.version}")
        
        return True
    
    def sign_record(self, record_id: str, signer_id: str, signer_role: str,
                   signing_reason: str, valid_until: Optional[datetime] = None) -> bool:
        """Apply digital signature to record"""
        with self.session_scope() as session:
            # Get record
            record = session.query(ElectronicRecord).filter_by(record_id=record_id).first()
            if not record:
                logger.error(f"Record {record_id} not found")
                return False
            
            # Create signature record
            signature_data = {
                "record_id": record_id,
                "record_version": record.version,
                "content_hash": record.content_hash,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            signature_bytes = json.dumps(signature_data).encode('utf-8')
            encrypted_signature, signature_hash = self.crypto.generate_signature(
                signature_bytes,
                signer_id
            )
            
            signature_record = SignatureRecord(
                record_id=record_id,
                record_version=record.version,
                signer_id=signer_id,
                signer_role=signer_role,
                signature_data=encrypted_signature,
                signature_hash=signature_hash,
                signing_reason=signing_reason,
                valid_until=valid_until,
                certificate_thumbprint=self.crypto.get_key_fingerprint()
            )
            
            session.add(signature_record)
            
            # Update record status
            record.status = "SIGNED"
            record.approved_by = signer_id
            record.approved_at = datetime.utcnow()
            
            # Log signing event
            self.log_event(
                event_type="SIGN",
                user_id=signer_id,
                action=f"Digitally signed record {record_id}: {signing_reason}",
                entity_type="ELECTRONIC_RECORD",
                entity_id=record_id,
                severity=EventSeverity.SECURITY,
                immediate=True
            )
            
            logger.info(f"Record {record_id} signed by {signer_id}")
        
        return True
    
    def _apply_compliance_rules(self, audit_record: AuditTrail):
        """Apply compliance rules to audit record"""
        with self.session_scope() as session:
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    # Evaluate rule logic
                    # This is a simplified implementation
                    # In production, you'd use a rules engine
                    
                    if rule.standard == ComplianceStandard.FDA_21CFR_PART11:
                        if rule.rule_id == "FDA-001":
                            # Check for digital signature
                            if audit_record.digital_signature is None:
                                self._log_compliance_violation(
                                    rule_id, audit_record, "Missing digital signature"
                                )
                        
                        elif rule.rule_id == "FDA-002":
                            # Check audit trail integrity
                            if audit_record.checksum is None:
                                self._log_compliance_violation(
                                    rule_id, audit_record, "Missing checksum"
                                )
                
                except Exception as e:
                    logger.error(f"Error applying rule {rule_id}: {e}")
    
    def _log_compliance_violation(self, rule_id: str, audit_record: AuditTrail,
                                 violation_message: str):
        """Log compliance violation"""
        violation_event = AuditTrail(
            event_type="COMPLIANCE_VIOLATION",
            event_subtype="RULE_VIOLATION",
            user_id=audit_record.user_id,
            user_role=audit_record.user_role,
            action=f"Compliance violation: {violation_message}",
            entity_type=audit_record.entity_type,
            entity_id=audit_record.entity_id,
            metadata={
                "rule_id": rule_id,
                "violation_message": violation_message,
                "related_event_id": audit_record.id
            },
            compliance_standard="FDA_21CFR_Part11",
            severity="ERROR"
        )
        
        with self.session_scope() as session:
            session.add(violation_event)
    
    def validate_audit_trail(self, start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Validate audit trail integrity"""
        with self.session_scope() as session:
            query = session.query(AuditTrail)
            
            if start_date:
                query = query.filter(AuditTrail.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditTrail.timestamp <= end_date)
            
            records = query.all()
            
            validation_results = []
            for record in records:
                # Recalculate checksum
                record_data = json.dumps(asdict(record), sort_keys=True, default=str).encode('utf-8')
                calculated_checksum = hashlib.sha256(record_data).hexdigest()
                
                # Verify digital signature if present
                signature_valid = False
                if record.digital_signature:
                    signature_valid = self.crypto.verify_signature(
                        record_data,
                        record.digital_signature
                    )
                
                validation_results.append({
                    "event_id": record.id,
                    "timestamp": record.timestamp,
                    "event_type": record.event_type,
                    "user_id": record.user_id,
                    "checksum_valid": record.checksum == calculated_checksum,
                    "signature_valid": signature_valid,
                    "overall_valid": (record.checksum == calculated_checksum and 
                                    (signature_valid or record.digital_signature is None))
                })
            
            df = pd.DataFrame(validation_results)
            
            # Update validation status
            valid_count = df['overall_valid'].sum()
            total_count = len(df)
            
            logger.info(f"Audit trail validation: {valid_count}/{total_count} valid records")
            
            return df
    
    def _run_validation_scheduler(self):
        """Run periodic validation of audit trail"""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                logger.info("Running scheduled audit trail validation...")
                self.validate_audit_trail()
            except Exception as e:
                logger.error(f"Error in validation scheduler: {e}")
    
    def _run_retention_manager(self):
        """Manage record retention and deletion"""
        while True:
            try:
                time.sleep(86400)  # Run every day
                self._apply_retention_policy()
            except Exception as e:
                logger.error(f"Error in retention manager: {e}")
    
    def _apply_retention_policy(self):
        """Apply retention policies to electronic records"""
        with self.session_scope() as session:
            # Find records past retention period
            cutoff_date = datetime.utcnow() - timedelta(days=3650)  # 10 years default
            
            expired_records = session.query(ElectronicRecord).filter(
                ElectronicRecord.created_at < cutoff_date,
                ElectronicRecord.deletion_scheduled.is_(None)
            ).all()
            
            for record in expired_records:
                # Schedule for deletion
                record.deletion_scheduled = datetime.utcnow() + timedelta(days=30)
                
                # Log retention event
                self.log_event(
                    event_type="RETENTION",
                    user_id="SYSTEM",
                    action=f"Scheduled record {record.record_id} for deletion",
                    entity_type="ELECTRONIC_RECORD",
                    entity_id=record.record_id,
                    severity=EventSeverity.INFO,
                    immediate=True
                )
            
            # Actually delete records scheduled more than 30 days ago
            delete_cutoff = datetime.utcnow() - timedelta(days=30)
            records_to_delete = session.query(ElectronicRecord).filter(
                ElectronicRecord.deletion_scheduled < delete_cutoff
            ).all()
            
            for record in records_to_delete:
                # Archive before deletion (in production, move to cold storage)
                self._archive_record(record)
                session.delete(record)
                
                logger.info(f"Deleted expired record: {record.record_id}")
            
            session.commit()
    
    def _archive_record(self, record: ElectronicRecord):
        """Archive record before deletion"""
        # In production, this would move records to cold storage
        # or create an archive copy
        archive_data = {
            "record_id": record.record_id,
            "record_type": record.record_type,
            "content": record.content,
            "metadata": record.metadata,
            "deletion_timestamp": datetime.utcnow().isoformat()
        }
        
        archive_file = Path(f"./archives/{record.record_id}_{record.version}.json")
        archive_file.parent.mkdir(exist_ok=True)
        
        with open(archive_file, 'w') as f:
            json.dump(archive_data, f, indent=2)
    
    def generate_compliance_report(self, start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        with self.session_scope() as session:
            # Get audit trail statistics
            audit_stats = session.query(
                AuditTrail.event_type,
                sa.func.count(AuditTrail.id).label('count')
            ).filter(
                AuditTrail.timestamp >= start_date,
                AuditTrail.timestamp <= end_date
            ).group_by(AuditTrail.event_type).all()
            
            # Get user activity
            user_activity = session.query(
                AuditTrail.user_id,
                sa.func.count(AuditTrail.id).label('activity_count')
            ).filter(
                AuditTrail.timestamp >= start_date,
                AuditTrail.timestamp <= end_date
            ).group_by(AuditTrail.user_id).order_by(sa.desc('activity_count')).limit(10).all()
            
            # Get compliance violations
            violations = session.query(AuditTrail).filter(
                AuditTrail.event_type == "COMPLIANCE_VIOLATION",
                AuditTrail.timestamp >= start_date,
                AuditTrail.timestamp <= end_date
            ).all()
            
            # Calculate metrics
            validation_results = self.validate_audit_trail(start_date, end_date)
            validity_rate = validation_results['overall_valid'].mean() * 100
            
            report = {
                "report_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                    "generated": datetime.utcnow().isoformat()
                },
                "audit_trail_metrics": {
                    "total_events": sum(stat.count for stat in audit_stats),
                    "event_types": {stat.event_type: stat.count for stat in audit_stats},
                    "validity_rate_percent": round(validity_rate, 2)
                },
                "user_activity": [
                    {"user_id": user_id, "activity_count": count}
                    for user_id, count in user_activity
                ],
                "compliance_violations": {
                    "total": len(violations),
                    "by_rule": {},
                    "details": [
                        {
                            "timestamp": v.timestamp.isoformat(),
                            "user_id": v.user_id,
                            "violation": v.metadata.get("violation_message", "")
                        }
                        for v in violations
                    ]
                },
                "electronic_records": {
                    "total": session.query(ElectronicRecord).count(),
                    "by_status": dict(session.query(
                        ElectronicRecord.status,
                        sa.func.count(ElectronicRecord.id)
                    ).group_by(ElectronicRecord.status).all()),
                    "signed_records": session.query(ElectronicRecord).filter(
                        ElectronicRecord.status == "SIGNED"
                    ).count()
                },
                "recommendations": self._generate_recommendations(validation_results, violations)
            }
            
            return report
    
    def _generate_recommendations(self, validation_results: pd.DataFrame,
                                 violations: List[AuditTrail]) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Check audit trail validity
        validity_rate = validation_results['overall_valid'].mean()
        if validity_rate < 0.95:
            recommendations.append(
                f"Audit trail validity rate is {validity_rate:.1%}. "
                "Investigate and fix invalid records."
            )
        
        # Check for unsigned records
        unsigned_count = validation_results['signature_valid'].sum()
        if unsigned_count > 0:
            recommendations.append(
                f"Found {unsigned_count} records without valid signatures. "
                "Review and sign critical records."
            )
        
        # Check for compliance violations
        if violations:
            recommendations.append(
                f"Found {len(violations)} compliance violations. "
                "Review and address each violation."
            )
        
        return recommendations

# Factory function
def create_compliance_logger(config_file: str = "compliance_config.json") -> ComplianceLogger:
    """Create and configure compliance logging system"""
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        db_url = config.get("database_url", "sqlite:///compliance.db")
        crypto_path = config.get("crypto_path", "./crypto_keys")
        
        crypto_manager = CryptographyManager(crypto_path)
        logger = ComplianceLogger(db_url, crypto_manager)
        
        # Load custom rules
        if "rules" in config:
            for rule_config in config["rules"]:
                rule = ComplianceRule(**rule_config)
                logger.rules[rule.rule_id] = rule
        
        logger.info("Compliance logging system initialized with configuration")
        
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        logger = ComplianceLogger()
    
    return logger

# Example usage
def example_compliance_workflow():
    """Example compliance workflow"""
    
    # Create logger
    logger = create_compliance_logger()
    
    # Create electronic record
    record_content = {
        "experiment_id": "EXP_20240101_001",
        "protocol": "PCR_Setup_v1.0",
        "parameters": {
            "template_dna_ng": 100,
            "primer_concentration_nm": 500,
            "cycles": 35
        },
        "results": {
            "amplification_success": True,
            "ct_value": 23.5
        }
    }
    
    record_id = logger.create_electronic_record(
        record_id="EXP_20240101_001",
        record_type="EXPERIMENT_RESULT",
        content=record_content,
        created_by="researcher@company.com",
        retention_days=365 * 15  # 15 years retention
    )
    
    print(f"Created electronic record: {record_id}")
    
    # Update record
    updated_content = record_content.copy()
    updated_content["results"]["ct_value"] = 23.7  # Corrected value
    
    logger.update_electronic_record(
        record_id=record_id,
        new_content=updated_content,
        modified_by="supervisor@company.com",
        change_reason="Corrected CT value measurement"
    )
    
    print(f"Updated record: {record_id}")
    
    # Sign record
    logger.sign_record(
        record_id=record_id,
        signer_id="qa_manager@company.com",
        signer_role="QA_MANAGER",
        signing_reason="Verified experiment results",
        valid_until=datetime.utcnow() + timedelta(days=365 * 10)
    )
    
    print(f"Signed record: {record_id}")
    
    # Generate compliance report
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    report = logger.generate_compliance_report(start_date, end_date)
    
    print(f"\nCompliance Report for last 30 days:")
    print(f"Total events: {report['audit_trail_metrics']['total_events']}")
    print(f"Validity rate: {report['audit_trail_metrics']['validity_rate_percent']}%")
    print(f"Compliance violations: {report['compliance_violations']['total']}")
    
    # Save report
    report_file = f"compliance_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to {report_file}")

if __name__ == "__main__":
    example_compliance_workflow()