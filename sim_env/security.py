"""
User Management & Security System
Advanced security system with multi-factor authentication, role-based access control,
encryption, audit logging, and comprehensive user management.
"""

import hashlib
import hmac
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import json
import logging
import sqlite3
from pathlib import Path
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
import pickle
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class UserRole(Enum):
    """User roles with different permission levels"""
    GUEST = "guest"
    USER = "user"
    RESEARCHER = "researcher"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    SYSTEM = "system"

class PermissionLevel(Enum):
    """Permission levels for system access"""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMINISTER = 4

class SecurityEvent(Enum):
    """Security event types for audit logging"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    ROLE_CHANGE = "role_change"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SECURITY_ALERT = "security_alert"
    DATA_ACCESS = "data_access"
    SYSTEM_ACCESS = "system_access"

class MFAType(Enum):
    """Multi-factor authentication types"""
    TOTP = "totp"  # Time-based One-Time Password
    EMAIL = "email"
    SMS = "sms"
    BACKUP_CODES = "backup_codes"
    BIOMETRIC = "biometric"

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    mfa_enabled: bool = False
    mfa_type: Optional[MFAType] = None
    mfa_secret: Optional[str] = None
    profile_data: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserSession:
    """Active user session"""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str
    is_valid: bool = True
    last_activity: datetime = field(default_factory=datetime.now)

@dataclass
class Permission:
    """System permission definition"""
    permission_id: str
    name: str
    description: str
    resource: str
    level: PermissionLevel
    constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    policy_id: str
    name: str
    description: str
    rules: Dict[str, Any]
    is_active: bool = True

@dataclass
class AuditLog:
    """Security audit log entry"""
    log_id: str
    timestamp: datetime
    event_type: SecurityEvent
    user_id: Optional[str]
    ip_address: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"

class PasswordValidator:
    """Advanced password validation and strength assessment"""
    
    def __init__(self):
        self.min_length = 8
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_numbers = True
        self.require_special_chars = True
        self.common_passwords = self._load_common_passwords()
        
    def _load_common_passwords(self) -> Set[str]:
        """Load common passwords for validation"""
        common = {
            "password", "123456", "12345678", "1234", "qwerty", "12345", 
            "dragon", "baseball", "football", "letmein", "monkey", "abc123"
        }
        return common
        
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against security policies"""
        errors = []
        
        if len(password) < self.min_length:
            errors.append(f"Password must be at least {self.min_length} characters long")
            
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
            
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
            
        if self.require_numbers and not re.search(r'[0-9]', password):
            errors.append("Password must contain at least one number")
            
        if self.require_special_chars and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
            
        if password.lower() in self.common_passwords:
            errors.append("Password is too common")
            
        # Check for sequential characters
        if self._has_sequential_chars(password):
            errors.append("Password contains sequential characters")
            
        return len(errors) == 0, errors
        
    def _has_sequential_chars(self, password: str) -> bool:
        """Check for sequential characters (e.g., 123, abc)"""
        for i in range(len(password) - 2):
            if (ord(password[i+1]) == ord(password[i]) + 1 and 
                ord(password[i+2]) == ord(password[i]) + 2):
                return True
        return False
        
    def calculate_password_strength(self, password: str) -> float:
        """Calculate password strength score (0.0 to 1.0)"""
        score = 0.0
        
        # Length factor
        length_factor = min(len(password) / 20.0, 1.0)
        score += length_factor * 0.3
        
        # Character variety
        char_types = 0
        if re.search(r'[a-z]', password):
            char_types += 1
        if re.search(r'[A-Z]', password):
            char_types += 1
        if re.search(r'[0-9]', password):
            char_types += 1
        if re.search(r'[^a-zA-Z0-9]', password):
            char_types += 1
            
        variety_factor = char_types / 4.0
        score += variety_factor * 0.3
        
        # Entropy estimation
        entropy = self._calculate_entropy(password)
        entropy_factor = min(entropy / 4.0, 1.0)  # Normalize
        score += entropy_factor * 0.4
        
        return min(score, 1.0)
        
    def _calculate_entropy(self, password: str) -> float:
        """Calculate password entropy"""
        char_set_size = 0
        if re.search(r'[a-z]', password):
            char_set_size += 26
        if re.search(r'[A-Z]', password):
            char_set_size += 26
        if re.search(r'[0-9]', password):
            char_set_size += 10
        if re.search(r'[^a-zA-Z0-9]', password):
            char_set_size += 32  # Common special characters
            
        if char_set_size == 0:
            return 0.0
            
        entropy = len(password) * (char_set_size ** 0.5)
        return entropy

class EncryptionManager:
    """Advanced encryption and key management"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.master_key = master_key
        else:
            # Generate a new master key
            self.master_key = Fernet.generate_key()
            
        self.fernet = Fernet(self.master_key)
        self.key_derivation_salt = os.urandom(16)
        
    def derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
    def encrypt_data(self, data: Any) -> str:
        """Encrypt any data using Fernet symmetric encryption"""
        # Serialize data
        serialized_data = pickle.dumps(data)
        # Encrypt
        encrypted_data = self.fernet.encrypt(serialized_data)
        return base64.urlsafe_b64encode(encrypted_data).decode()
        
    def decrypt_data(self, encrypted_data: str) -> Any:
        """Decrypt data encrypted with encrypt_data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return pickle.loads(decrypted_bytes)
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
            
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash sensitive data with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
            
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return hashed.hex(), salt
        
    def generate_api_key(self, user_id: str) -> str:
        """Generate secure API key for user"""
        timestamp = str(int(time.time()))
        random_component = secrets.token_urlsafe(32)
        data = f"{user_id}:{timestamp}:{random_component}"
        
        # Create HMAC signature
        signature = hmac.new(
            self.master_key,
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        api_key = f"vss_{user_id}_{timestamp}_{random_component}_{signature[-16:]}"
        return api_key

class MFAManager:
    """Multi-factor authentication manager"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.backup_code_count = 10
        self.backup_code_length = 8
        
    def generate_totp_secret(self) -> str:
        """Generate TOTP secret for authenticator apps"""
        return base64.b32encode(os.urandom(20)).decode('utf-8')
        
    def verify_totp_code(self, secret: str, code: str, window: int = 1) -> bool:
        """Verify TOTP code (simplified implementation)"""
        # In production, use libraries like pyotp
        try:
            # This is a simplified verification - use proper TOTP in production
            current_time = int(time.time() // 30)
            expected_codes = []
            
            for i in range(-window, window + 1):
                time_counter = current_time + i
                # Simplified TOTP calculation (use proper implementation)
                simulated_code = str((hashlib.sha256(f"{secret}{time_counter}".encode()).hexdigest()[:6]))
                expected_codes.append(simulated_code)
                
            return code in expected_codes
        except:
            return False
            
    def generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA"""
        codes = []
        for _ in range(self.backup_code_count):
            code = ''.join(secrets.choice('23456789ABCDEFGHJKLMNPQRSTUVWXYZ') 
                          for _ in range(self.backup_code_length))
            codes.append(code)
        return codes
        
    def send_email_code(self, email: str, code: str):
        """Send MFA code via email (simulated)"""
        # In production, integrate with email service
        logging.info(f"MFA email sent to {email}: Code {code}")
        
    def send_sms_code(self, phone_number: str, code: str):
        """Send MFA code via SMS (simulated)"""
        # In production, integrate with SMS service
        logging.info(f"MFA SMS sent to {phone_number}: Code {code}")

class EmailService:
    """Email service for notifications and MFA"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        
    def send_verification_email(self, email: str, verification_token: str):
        """Send email verification message"""
        subject = "Verify Your frex Simulations Account"
        body = f"""
        Welcome to frex Simulations!
        
        Please verify your email address by clicking the following link:
        https://your-domain.com/verify-email?token={verification_token}
        
        If you didn't create an account, please ignore this email.
        
        Best regards,
        frex Simulations Team
        """
        
        self._send_email(email, subject, body)
        
    def send_password_reset_email(self, email: str, reset_token: str):
        """Send password reset email"""
        subject = "Reset Your frex Simulations Password"
        body = f"""
        You have requested to reset your password.
        
        Click the following link to reset your password:
        https://your-domain.com/reset-password?token={reset_token}
        
        This link will expire in 1 hour.
        
        If you didn't request this reset, please ignore this email.
        
        Best regards,
        frex Simulations Team
        """
        
        self._send_email(email, subject, body)
        
    def send_security_alert(self, email: str, alert_type: str, details: str):
        """Send security alert email"""
        subject = f"Security Alert: {alert_type}"
        body = f"""
        Security Alert
        
        Type: {alert_type}
        Details: {details}
        Time: {datetime.now()}
        
        If this wasn't you, please secure your account immediately.
        
        Best regards,
        frex Simulations Security Team
        """
        
        self._send_email(email, subject, body)
        
    def _send_email(self, to_email: str, subject: str, body: str):
        """Send email using SMTP"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
            logging.info(f"Email sent to {to_email}")
        except Exception as e:
            logging.error(f"Failed to send email to {to_email}: {e}")

class DatabaseManager:
    """Database management for user and security data"""
    
    def __init__(self, db_path: str = "security.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        try:
            # Users table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    is_verified BOOLEAN DEFAULT 0,
                    mfa_enabled BOOLEAN DEFAULT 0,
                    mfa_type TEXT,
                    mfa_secret TEXT,
                    profile_data TEXT,
                    preferences TEXT
                )
            ''')
            
            # Sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT NOT NULL,
                    is_valid BOOLEAN DEFAULT 1,
                    last_activity TIMESTAMP NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Permissions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS permissions (
                    permission_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    resource TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    constraints TEXT
                )
            ''')
            
            # Role permissions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS role_permissions (
                    role TEXT NOT NULL,
                    permission_id TEXT NOT NULL,
                    PRIMARY KEY (role, permission_id),
                    FOREIGN KEY (permission_id) REFERENCES permissions (permission_id)
                )
            ''')
            
            # Audit log table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    log_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    ip_address TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT,
                    severity TEXT DEFAULT 'info'
                )
            ''')
            
            # Security policies table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_policies (
                    policy_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    rules TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            conn.commit()
        finally:
            conn.close()
            
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
        
    def save_user(self, user: User):
        """Save user to database"""
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, username, email, password_hash, role, created_at, last_login, 
                 is_active, is_verified, mfa_enabled, mfa_type, mfa_secret, profile_data, preferences)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.user_id,
                user.username,
                user.email,
                user.password_hash,
                user.role.value,
                user.created_at.isoformat(),
                user.last_login.isoformat() if user.last_login else None,
                user.is_active,
                user.is_verified,
                user.mfa_enabled,
                user.mfa_type.value if user.mfa_type else None,
                user.mfa_secret,
                json.dumps(user.profile_data),
                json.dumps(user.preferences)
            ))
            conn.commit()
        finally:
            conn.close()
            
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        conn = self._get_connection()
        try:
            cursor = conn.execute('''
                SELECT * FROM users WHERE user_id = ?
            ''', (user_id,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()
            
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        conn = self._get_connection()
        try:
            cursor = conn.execute('''
                SELECT * FROM users WHERE username = ?
            ''', (username,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()
            
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        conn = self._get_connection()
        try:
            cursor = conn.execute('''
                SELECT * FROM users WHERE email = ?
            ''', (email,))
            row = cursor.fetchone()
            return self._row_to_user(row) if row else None
        finally:
            conn.close()
            
    def _row_to_user(self, row) -> User:
        """Convert database row to User object"""
        return User(
            user_id=row[0],
            username=row[1],
            email=row[2],
            password_hash=row[3],
            role=UserRole(row[4]),
            created_at=datetime.fromisoformat(row[5]),
            last_login=datetime.fromisoformat(row[6]) if row[6] else None,
            is_active=bool(row[7]),
            is_verified=bool(row[8]),
            mfa_enabled=bool(row[9]),
            mfa_type=MFAType(row[10]) if row[10] else None,
            mfa_secret=row[11],
            profile_data=json.loads(row[12]) if row[12] else {},
            preferences=json.loads(row[13]) if row[13] else {}
        )
        
    def save_session(self, session: UserSession):
        """Save session to database"""
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, user_id, created_at, expires_at, ip_address, user_agent, is_valid, last_activity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id,
                session.user_id,
                session.created_at.isoformat(),
                session.expires_at.isoformat(),
                session.ip_address,
                session.user_agent,
                session.is_valid,
                session.last_activity.isoformat()
            ))
            conn.commit()
        finally:
            conn.close()
            
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        conn = self._get_connection()
        try:
            cursor = conn.execute('''
                SELECT * FROM sessions WHERE session_id = ?
            ''', (session_id,))
            row = cursor.fetchone()
            return self._row_to_session(row) if row else None
        finally:
            conn.close()
            
    def _row_to_session(self, row) -> UserSession:
        """Convert database row to UserSession object"""
        return UserSession(
            session_id=row[0],
            user_id=row[1],
            created_at=datetime.fromisoformat(row[2]),
            expires_at=datetime.fromisoformat(row[3]),
            ip_address=row[4],
            user_agent=row[5],
            is_valid=bool(row[6]),
            last_activity=datetime.fromisoformat(row[7])
        )
        
    def log_audit_event(self, audit_log: AuditLog):
        """Save audit log entry"""
        conn = self._get_connection()
        try:
            conn.execute('''
                INSERT INTO audit_log 
                (log_id, timestamp, event_type, user_id, ip_address, description, details, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_log.log_id,
                audit_log.timestamp.isoformat(),
                audit_log.event_type.value,
                audit_log.user_id,
                audit_log.ip_address,
                audit_log.description,
                json.dumps(audit_log.details),
                audit_log.severity
            ))
            conn.commit()
        finally:
            conn.close()

class PermissionManager:
    """Role-based access control permission manager"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self._load_permissions()
        
    def _load_permissions(self):
        """Load default permissions into database"""
        default_permissions = [
            Permission(
                permission_id="simulation_read",
                name="Read Simulation",
                description="Read simulation data and results",
                resource="simulation",
                level=PermissionLevel.READ
            ),
            Permission(
                permission_id="simulation_write",
                name="Write Simulation",
                description="Create and modify simulations",
                resource="simulation",
                level=PermissionLevel.WRITE
            ),
            Permission(
                permission_id="simulation_execute",
                name="Execute Simulation",
                description="Run and control simulations",
                resource="simulation",
                level=PermissionLevel.EXECUTE
            ),
            Permission(
                permission_id="data_export",
                name="Export Data",
                description="Export simulation data",
                resource="data",
                level=PermissionLevel.WRITE
            ),
            Permission(
                permission_id="user_manage",
                name="Manage Users",
                description="Create and manage user accounts",
                resource="system",
                level=PermissionLevel.ADMINISTER
            ),
            Permission(
                permission_id="system_admin",
                name="System Administration",
                description="Full system administration access",
                resource="system",
                level=PermissionLevel.ADMINISTER
            )
        ]
        
        # Default role permissions
        role_permissions = {
            UserRole.GUEST: ["simulation_read"],
            UserRole.USER: ["simulation_read", "simulation_write", "simulation_execute", "data_export"],
            UserRole.RESEARCHER: ["simulation_read", "simulation_write", "simulation_execute", "data_export"],
            UserRole.ADMIN: ["simulation_read", "simulation_write", "simulation_execute", "data_export", "user_manage"],
            UserRole.SUPER_ADMIN: ["simulation_read", "simulation_write", "simulation_execute", "data_export", "user_manage", "system_admin"],
            UserRole.SYSTEM: ["system_admin"]
        }
        
        # Save permissions and role assignments
        conn = self.db._get_connection()
        try:
            # Clear existing data (for demo - in production, use migrations)
            conn.execute("DELETE FROM permissions")
            conn.execute("DELETE FROM role_permissions")
            
            # Insert permissions
            for perm in default_permissions:
                conn.execute('''
                    INSERT INTO permissions (permission_id, name, description, resource, level, constraints)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    perm.permission_id,
                    perm.name,
                    perm.description,
                    perm.resource,
                    perm.level.value,
                    json.dumps(perm.constraints)
                ))
                
            # Insert role permissions
            for role, perm_ids in role_permissions.items():
                for perm_id in perm_ids:
                    conn.execute('''
                        INSERT INTO role_permissions (role, permission_id)
                        VALUES (?, ?)
                    ''', (role.value, perm_id))
                    
            conn.commit()
        finally:
            conn.close()
            
    def user_has_permission(self, user: User, permission_id: str, resource: Optional[str] = None) -> bool:
        """Check if user has specific permission"""
        conn = self.db._get_connection()
        try:
            query = '''
                SELECT 1 FROM role_permissions rp
                JOIN permissions p ON rp.permission_id = p.permission_id
                WHERE rp.role = ? AND rp.permission_id = ?
            '''
            params = [user.role.value, permission_id]
            
            if resource:
                query += " AND p.resource = ?"
                params.append(resource)
                
            cursor = conn.execute(query, params)
            return cursor.fetchone() is not None
        finally:
            conn.close()
            
    def get_user_permissions(self, user: User) -> List[Permission]:
        """Get all permissions for a user"""
        conn = self.db._get_connection()
        try:
            cursor = conn.execute('''
                SELECT p.permission_id, p.name, p.description, p.resource, p.level, p.constraints
                FROM role_permissions rp
                JOIN permissions p ON rp.permission_id = p.permission_id
                WHERE rp.role = ?
            ''', (user.role.value,))
            
            permissions = []
            for row in cursor.fetchall():
                permissions.append(Permission(
                    permission_id=row[0],
                    name=row[1],
                    description=row[2],
                    resource=row[3],
                    level=PermissionLevel(row[4]),
                    constraints=json.loads(row[5]) if row[5] else {}
                ))
                
            return permissions
        finally:
            conn.close()

class SecurityMonitor:
    """Real-time security monitoring and threat detection"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db = database_manager
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.suspicious_activities: List[Dict[str, Any]] = []
        self.monitoring_enabled = True
        
        # Security thresholds
        self.max_failed_logins = 5
        self.failed_login_window = 900  # 15 minutes in seconds
        self.suspicious_ip_threshold = 10
        
    def monitor_login_attempt(self, username: str, ip_address: str, success: bool):
        """Monitor login attempts for suspicious patterns"""
        if not success:
            # Track failed attempts
            if username not in self.failed_login_attempts:
                self.failed_login_attempts[username] = []
                
            self.failed_login_attempts[username].append(datetime.now())
            
            # Clean old attempts
            cutoff_time = datetime.now() - timedelta(seconds=self.failed_login_window)
            self.failed_login_attempts[username] = [
                attempt for attempt in self.failed_login_attempts[username]
                if attempt > cutoff_time
            ]
            
            # Check for brute force
            if len(self.failed_login_attempts[username]) >= self.max_failed_logins:
                self._trigger_security_alert(
                    "brute_force_attempt",
                    f"Multiple failed login attempts for user {username} from {ip_address}",
                    "high",
                    {"username": username, "ip_address": ip_address, "attempts": len(self.failed_login_attempts[username])}
                )
                
    def monitor_user_activity(self, user_id: str, activity_type: str, details: Dict[str, Any]):
        """Monitor user activities for suspicious patterns"""
        # Check for unusual activity patterns
        if activity_type == "data_access" and details.get("sensitive", False):
            self._check_sensitive_data_access(user_id, details)
            
        elif activity_type == "permission_change":
            self._check_permission_escalation(user_id, details)
            
    def _check_sensitive_data_access(self, user_id: str, details: Dict[str, Any]):
        """Check for suspicious sensitive data access"""
        # Implement anomaly detection for data access patterns
        # This could include:
        # - Accessing large amounts of data quickly
        # - Accessing data outside normal hours
        # - Accessing data from unusual locations
        
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside normal hours (6 AM - 10 PM)
            self._trigger_security_alert(
                "after_hours_access",
                f"User {user_id} accessed sensitive data outside normal hours",
                "medium",
                {"user_id": user_id, "time": current_hour, "details": details}
            )
            
    def _check_permission_escalation(self, user_id: str, details: Dict[str, Any]):
        """Check for suspicious permission changes"""
        # Alert on rapid permission escalation
        self._trigger_security_alert(
            "permission_escalation",
            f"User {user_id} had permissions modified",
            "medium",
            {"user_id": user_id, "details": details}
        )
        
    def _trigger_security_alert(self, alert_type: str, description: str, severity: str, details: Dict[str, Any]):
        """Trigger security alert"""
        alert = {
            "timestamp": datetime.now(),
            "type": alert_type,
            "description": description,
            "severity": severity,
            "details": details
        }
        
        self.suspicious_activities.append(alert)
        
        # Log to audit system
        audit_log = AuditLog(
            log_id=f"alert_{int(time.time())}_{secrets.token_hex(4)}",
            timestamp=datetime.now(),
            event_type=SecurityEvent.SECURITY_ALERT,
            user_id=None,
            ip_address="system",
            description=description,
            details=details,
            severity=severity
        )
        
        self.db.log_audit_event(audit_log)
        logging.warning(f"SECURITY ALERT: {description}")

class UserManagementSecuritySystem:
    """
    Comprehensive User Management & Security System
    Provides authentication, authorization, audit logging, and security monitoring
    """
    
    def __init__(self, db_path: str = "security.db", master_key: Optional[bytes] = None):
        # Core components
        self.db = DatabaseManager(db_path)
        self.encryption = EncryptionManager(master_key)
        self.password_validator = PasswordValidator()
        self.mfa_manager = MFAManager(self.encryption)
        self.permission_manager = PermissionManager(self.db)
        self.security_monitor = SecurityMonitor(self.db)
        
        # Email service (configure with actual SMTP settings)
        self.email_service = None
        
        # Session configuration
        self.session_timeout = timedelta(hours=24)
        self.session_cleanup_interval = 3600  # 1 hour
        
        # Start session cleanup thread
        self.cleanup_thread = threading.Thread(target=self._session_cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logging.info("User Management & Security System initialized")
        
    def configure_email_service(self, smtp_server: str, smtp_port: int, username: str, password: str):
        """Configure email service for notifications"""
        self.email_service = EmailService(smtp_server, smtp_port, username, password)
        
    def register_user(self, username: str, email: str, password: str, 
                     role: UserRole = UserRole.USER, profile_data: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Register a new user account"""
        try:
            # Validate input
            if not username or not email or not password:
                return False, "Username, email, and password are required"
                
            # Check if username or email already exists
            if self.db.get_user_by_username(username):
                return False, "Username already exists"
                
            if self.db.get_user_by_email(email):
                return False, "Email already registered"
                
            # Validate password strength
            is_valid, errors = self.password_validator.validate_password(password)
            if not is_valid:
                return False, f"Password validation failed: {', '.join(errors)}"
                
            # Hash password
            password_hash = self._hash_password(password)
            
            # Create user
            user = User(
                user_id=f"user_{int(time.time())}_{secrets.token_hex(8)}",
                username=username,
                email=email,
                password_hash=password_hash,
                role=role,
                created_at=datetime.now(),
                profile_data=profile_data or {}
            )
            
            # Save user
            self.db.save_user(user)
            
            # Log event
            self.db.log_audit_event(AuditLog(
                log_id=f"register_{int(time.time())}",
                timestamp=datetime.now(),
                event_type=SecurityEvent.USER_CREATED,
                user_id=user.user_id,
                ip_address="system",
                description=f"User {username} registered with role {role.value}",
                details={"username": username, "role": role.value}
            ))
            
            # Send verification email if configured
            if self.email_service:
                verification_token = self._generate_verification_token(user.user_id)
                self.email_service.send_verification_email(email, verification_token)
                
            return True, "User registered successfully"
            
        except Exception as e:
            logging.error(f"User registration failed: {e}")
            return False, f"Registration failed: {str(e)}"
            
    def authenticate_user(self, username: str, password: str, ip_address: str, 
                         user_agent: str, mfa_code: Optional[str] = None) -> Tuple[bool, Optional[User], Optional[str], str]:
        """Authenticate user and create session"""
        try:
            # Get user
            user = self.db.get_user_by_username(username)
            if not user:
                self.security_monitor.monitor_login_attempt(username, ip_address, False)
                return False, None, None, "Invalid credentials"
                
            # Check if account is active
            if not user.is_active:
                return False, None, None, "Account is deactivated"
                
            # Verify password
            if not self._verify_password(password, user.password_hash):
                self.security_monitor.monitor_login_attempt(username, ip_address, False)
                return False, None, None, "Invalid credentials"
                
            # Check MFA if enabled
            if user.mfa_enabled and user.mfa_type:
                if not mfa_code:
                    return False, None, None, "MFA code required"
                    
                if not self._verify_mfa_code(user, mfa_code):
                    self.security_monitor.monitor_login_attempt(username, ip_address, False)
                    return False, None, None, "Invalid MFA code"
                    
            # Update last login
            user.last_login = datetime.now()
            self.db.save_user(user)
            
            # Create session
            session = self._create_session(user.user_id, ip_address, user_agent)
            
            # Log successful login
            self.db.log_audit_event(AuditLog(
                log_id=f"login_{int(time.time())}",
                timestamp=datetime.now(),
                event_type=SecurityEvent.LOGIN_SUCCESS,
                user_id=user.user_id,
                ip_address=ip_address,
                description=f"User {username} logged in successfully",
                details={"mfa_used": user.mfa_enabled}
            ))
            
            self.security_monitor.monitor_login_attempt(username, ip_address, True)
            
            return True, user, session.session_id, "Authentication successful"
            
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            return False, None, None, f"Authentication failed: {str(e)}"
            
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        
    def _verify_mfa_code(self, user: User, code: str) -> bool:
        """Verify MFA code based on user's MFA type"""
        if user.mfa_type == MFAType.TOTP and user.mfa_secret:
            return self.mfa_manager.verify_totp_code(user.mfa_secret, code)
        # Add other MFA type verifications here
        return False
        
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> UserSession:
        """Create new user session"""
        session = UserSession(
            session_id=f"session_{int(time.time())}_{secrets.token_hex(16)}",
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.save_session(session)
        
        self.db.log_audit_event(AuditLog(
            log_id=f"session_{int(time.time())}",
            timestamp=datetime.now(),
            event_type=SecurityEvent.SESSION_CREATED,
            user_id=user_id,
            ip_address=ip_address,
            description="New user session created",
            details={"session_id": session.session_id}
        ))
        
        return session
        
    def validate_session(self, session_id: str, ip_address: str) -> Tuple[bool, Optional[User]]:
        """Validate user session"""
        session = self.db.get_session(session_id)
        if not session or not session.is_valid:
            return False, None
            
        # Check expiration
        if datetime.now() > session.expires_at:
            session.is_valid = False
            self.db.save_session(session)
            
            self.db.log_audit_event(AuditLog(
                log_id=f"session_expired_{int(time.time())}",
                timestamp=datetime.now(),
                event_type=SecurityEvent.SESSION_EXPIRED,
                user_id=session.user_id,
                ip_address=ip_address,
                description="User session expired",
                details={"session_id": session_id}
            ))
            
            return False, None
            
        # Update last activity
        session.last_activity = datetime.now()
        self.db.save_session(session)
        
        # Get user
        user = self.db.get_user_by_id(session.user_id)
        return user is not None, user
        
    def logout_user(self, session_id: str, ip_address: str):
        """Logout user by invalidating session"""
        session = self.db.get_session(session_id)
        if session:
            session.is_valid = False
            self.db.save_session(session)
            
            self.db.log_audit_event(AuditLog(
                log_id=f"logout_{int(time.time())}",
                timestamp=datetime.now(),
                event_type=SecurityEvent.LOGOUT,
                user_id=session.user_id,
                ip_address=ip_address,
                description="User logged out",
                details={"session_id": session_id}
            ))
            
    def change_user_password(self, user_id: str, current_password: str, new_password: str) -> Tuple[bool, str]:
        """Change user password"""
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False, "User not found"
            
        # Verify current password
        if not self._verify_password(current_password, user.password_hash):
            return False, "Current password is incorrect"
            
        # Validate new password
        is_valid, errors = self.password_validator.validate_password(new_password)
        if not is_valid:
            return False, f"New password validation failed: {', '.join(errors)}"
            
        # Hash and save new password
        user.password_hash = self._hash_password(new_password)
        self.db.save_user(user)
        
        # Log event
        self.db.log_audit_event(AuditLog(
            log_id=f"password_change_{int(time.time())}",
            timestamp=datetime.now(),
            event_type=SecurityEvent.PASSWORD_CHANGE,
            user_id=user_id,
            ip_address="system",
            description="User changed password",
            details={}
        ))
        
        # Send security alert
        if self.email_service:
            self.email_service.send_security_alert(
                user.email,
                "Password Changed",
                "Your password was successfully changed."
            )
            
        return True, "Password changed successfully"
        
    def enable_mfa(self, user_id: str, mfa_type: MFAType) -> Tuple[bool, str, Optional[str]]:
        """Enable multi-factor authentication for user"""
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False, "User not found", None
            
        secret = None
        if mfa_type == MFAType.TOTP:
            secret = self.mfa_manager.generate_totp_secret()
            user.mfa_secret = secret
            
        user.mfa_enabled = True
        user.mfa_type = mfa_type
        self.db.save_user(user)
        
        # Log event
        self.db.log_audit_event(AuditLog(
            log_id=f"mfa_enable_{int(time.time())}",
            timestamp=datetime.now(),
            event_type=SecurityEvent.PERMISSION_GRANTED,
            user_id=user_id,
            ip_address="system",
            description=f"MFA enabled for user ({mfa_type.value})",
            details={"mfa_type": mfa_type.value}
        ))
        
        return True, "MFA enabled successfully", secret
        
    def check_permission(self, user: User, permission_id: str, resource: Optional[str] = None) -> bool:
        """Check if user has specific permission"""
        return self.permission_manager.user_has_permission(user, permission_id, resource)
        
    def get_user_permissions(self, user: User) -> List[Permission]:
        """Get all permissions for user"""
        return self.permission_manager.get_user_permissions(user)
        
    def generate_api_key(self, user_id: str) -> str:
        """Generate API key for user"""
        return self.encryption.generate_api_key(user_id)
        
    def _generate_verification_token(self, user_id: str) -> str:
        """Generate email verification token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.now() + timedelta(hours=24),
            'type': 'email_verification'
        }
        return jwt.encode(payload, self.encryption.master_key, algorithm='HS256')
        
    def _session_cleanup_loop(self):
        """Background thread for cleaning up expired sessions"""
        while True:
            try:
                self._cleanup_expired_sessions()
                time.sleep(self.session_cleanup_interval)
            except Exception as e:
                logging.error(f"Session cleanup error: {e}")
                time.sleep(60)  # Wait before retrying
                
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions from database"""
        conn = self.db._get_connection()
        try:
            cutoff_time = datetime.now().isoformat()
            conn.execute('''
                DELETE FROM sessions WHERE expires_at < ? OR is_valid = 0
            ''', (cutoff_time,))
            conn.commit()
            
            deleted_count = conn.total_changes
            if deleted_count > 0:
                logging.info(f"Cleaned up {deleted_count} expired sessions")
                
        finally:
            conn.close()
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        conn = self.db._get_connection()
        try:
            # Get user statistics
            user_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_users,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_users,
                    SUM(CASE WHEN is_verified = 1 THEN 1 ELSE 0 END) as verified_users,
                    SUM(CASE WHEN mfa_enabled = 1 THEN 1 ELSE 0 END) as mfa_users
                FROM users
            ''').fetchone()
            
            # Get session statistics
            session_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total_sessions,
                    SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) as active_sessions
                FROM sessions
            ''').fetchone()
            
            # Get recent security events
            recent_events = conn.execute('''
                SELECT event_type, severity, COUNT(*) as count
                FROM audit_log 
                WHERE timestamp > datetime('now', '-1 day')
                GROUP BY event_type, severity
            ''').fetchall()
            
            return {
                "user_statistics": {
                    "total_users": user_stats[0],
                    "active_users": user_stats[1],
                    "verified_users": user_stats[2],
                    "mfa_users": user_stats[3]
                },
                "session_statistics": {
                    "total_sessions": session_stats[0],
                    "active_sessions": session_stats[1]
                },
                "recent_security_events": [
                    {"event_type": row[0], "severity": row[1], "count": row[2]}
                    for row in recent_events
                ],
                "security_alerts": len(self.security_monitor.suspicious_activities),
                "system_health": "healthy"
            }
            
        finally:
            conn.close()

# Example usage and demonstration
def demo_security_system():
    """Demonstrate the user management and security system"""
    security_system = UserManagementSecuritySystem()
    
    print("User Management & Security System Demo")
    print("=" * 50)
    
    # Register a new user
    print("\n1. Registering new user...")
    success, message = security_system.register_user(
        username="researcher1",
        email="researcher@example.com",
        password="SecurePassword123!",
        role=UserRole.RESEARCHER,
        profile_data={"institution": "University", "department": "Physics"}
    )
    print(f"Registration: {message}")
    
    if success:
        # Authenticate user
        print("\n2. Authenticating user...")
        auth_success, user, session_id, auth_message = security_system.authenticate_user(
            username="researcher1",
            password="SecurePassword123!",
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        print(f"Authentication: {auth_message}")
        
        if auth_success and user:
            print(f"Authenticated as: {user.username} ({user.role.value})")
            print(f"Session ID: {session_id}")
            
            # Check permissions
            print("\n3. Checking permissions...")
            permissions = security_system.get_user_permissions(user)
            print(f"User has {len(permissions)} permissions:")
            for perm in permissions[:3]:  # Show first 3
                print(f"  - {perm.name} ({perm.resource})")
                
            # Validate session
            print("\n4. Validating session...")
            valid, validated_user = security_system.validate_session(session_id, "192.168.1.100")
            print(f"Session valid: {valid}")
            print(f"User matches: {validated_user.user_id == user.user_id}")
            
            # Generate API key
            print("\n5. Generating API key...")
            api_key = security_system.generate_api_key(user.user_id)
            print(f"API Key: {api_key[:20]}...")
            
            # Enable MFA
            print("\n6. Enabling MFA...")
            mfa_success, mfa_message, secret = security_system.enable_mfa(user.user_id, MFAType.TOTP)
            print(f"MFA enable: {mfa_message}")
            if secret:
                print(f"TOTP Secret: {secret}")
                
            # Change password
            print("\n7. Changing password...")
            change_success, change_message = security_system.change_user_password(
                user.user_id, "SecurePassword123!", "NewSecurePassword456!"
            )
            print(f"Password change: {change_message}")
            
            # Logout
            print("\n8. Logging out...")
            security_system.logout_user(session_id, "192.168.1.100")
            print("User logged out")
            
    # Get system status
    print("\n9. System status:")
    status = security_system.get_system_status()
    print(f"Total users: {status['user_statistics']['total_users']}")
    print(f"Active sessions: {status['session_statistics']['active_sessions']}")
    print(f"Security alerts: {status['security_alerts']}")
    
    return security_system

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    security_system = demo_security_system()