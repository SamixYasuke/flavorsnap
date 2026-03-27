"""
Security configuration and middleware for FlavorSnap API
"""
import os
import bleach
import re
from typing import Dict, Any, Optional
from flask import request, abort
from werkzeug.utils import secure_filename
import hashlib
import hmac
from datetime import datetime, timedelta

class SecurityConfig:
    """Security configuration class"""
    
    # Rate limiting configurations
    RATE_LIMITS = {
        'default': '100 per minute',
        'predict': '100 per minute',
        'health': '1000 per hour',
        'api_info': '60 per minute',
        'upload': '50 per minute'
    }
    
    # File upload security
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    ALLOWED_MIME_TYPES = {
        'image/jpeg',
        'image/png', 
        'image/gif',
        'image/webp'
    }
    
    # Input validation patterns
    PATTERNS = {
        'filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
        'api_key': re.compile(r'^[a-zA-Z0-9]{32,}$'),
        'request_id': re.compile(r'^[a-zA-Z0-9-_]{8,64}$')
    }
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }

class InputValidator:
    """Input validation and sanitization utilities"""
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not text:
            return ""
        
        # Clean HTML/Script tags
        cleaned = bleach.clean(text, tags=[], strip=True)
        
        # Limit length
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        
        return cleaned.strip()
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename format"""
        if not filename:
            return False
        
        # Check pattern
        if not SecurityConfig.PATTERNS['filename'].match(filename):
            return False
        
        # Check extension
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        return ext in SecurityConfig.ALLOWED_EXTENSIONS
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key:
            return False
        
        return bool(SecurityConfig.PATTERNS['api_key'].match(api_key))
    
    @staticmethod
    def validate_file_upload(file) -> tuple[bool, Optional[str]]:
        """Validate uploaded file"""
        if not file:
            return False, "No file provided"
        
        if file.filename == '':
            return False, "Empty filename"
        
        # Validate filename
        if not InputValidator.validate_filename(file.filename):
            return False, "Invalid filename"
        
        # Validate MIME type
        if file.content_type not in SecurityConfig.ALLOWED_MIME_TYPES:
            return False, f"Unsupported file type: {file.content_type}"
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > SecurityConfig.MAX_CONTENT_LENGTH:
            return False, f"File too large. Max size: {SecurityConfig.MAX_CONTENT_LENGTH} bytes"
        
        return True, None
    
    @staticmethod
    def secure_filename_custom(filename: str) -> str:
        """Custom secure filename generation"""
        # Use Werkzeug's secure_filename as base
        secure_name = secure_filename(filename)
        
        # Add timestamp to prevent collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(secure_name)
        
        return f"{name}_{timestamp}{ext}"

class APIKeyManager:
    """API key management utilities"""
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key"""
        import secrets
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def verify_api_key(api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash"""
        return hmac.compare_digest(
            hashlib.sha256(api_key.encode()).hexdigest(),
            hashed_key
        )

class RateLimitManager:
    """Rate limiting utilities"""
    
    @staticmethod
    def get_client_ip() -> str:
        """Get client IP address"""
        # Check for proxy headers
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        else:
            return request.remote_addr or 'unknown'
    
    @staticmethod
    def get_rate_limit_key(endpoint: str = None) -> str:
        """Generate rate limit key"""
        ip = RateLimitManager.get_client_ip()
        endpoint = endpoint or request.endpoint or 'unknown'
        return f"rate_limit:{endpoint}:{ip}"

class SecurityMiddleware:
    """Security middleware for Flask application"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security middleware"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """Before request security checks"""
        # Validate request size
        content_length = request.content_length or 0
        if content_length > SecurityConfig.MAX_CONTENT_LENGTH:
            abort(413, description="Request too large")
        
        # Log suspicious activity
        self._log_request()
    
    def after_request(self, response):
        """Add security headers to response"""
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response
    
    def _log_request(self):
        """Log request for security monitoring"""
        ip = RateLimitManager.get_client_ip()
        user_agent = request.headers.get('User-Agent', 'Unknown')
        endpoint = request.endpoint or 'unknown'
        
        # Log suspicious patterns
        suspicious_patterns = [
            '../', '<script', 'javascript:', 'data:',
            'vbscript:', 'onload=', 'onerror='
        ]
        
        request_data = str(request.data) if request.data else ""
        for pattern in suspicious_patterns:
            if pattern.lower() in request_data.lower():
                app.logger.warning(
                    f"Suspicious request from {ip}: {pattern} in {endpoint}"
                )
                break

# Security helper functions
def is_safe_url(url: str) -> bool:
    """Check if URL is safe for redirects"""
    if not url:
        return False
    
    # Prevent open redirects
    if url.startswith(('//', 'http://', 'https://')):
        return False
    
    return True

def validate_json_input(data: Dict[str, Any], required_fields: list = None) -> tuple[bool, Optional[str]]:
    """Validate JSON input"""
    if not isinstance(data, dict):
        return False, "Invalid JSON format"
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    return True, None

# Security monitoring
class SecurityMonitor:
    """Security monitoring and alerting"""
    
    def __init__(self, app=None):
        self.app = app
        self.suspicious_ips = {}
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize security monitor"""
        app.before_request(self.monitor_request)
    
    def monitor_request(self):
        """Monitor requests for suspicious activity"""
        ip = RateLimitManager.get_client_ip()
        current_time = datetime.now()
        
        # Track requests per IP
        if ip not in self.suspicious_ips:
            self.suspicious_ips[ip] = {'count': 0, 'first_seen': current_time}
        
        self.suspicious_ips[ip]['count'] += 1
        
        # Check for unusual patterns
        if self._is_suspicious(ip):
            app.logger.warning(f"Suspicious activity detected from {ip}")
    
    def _is_suspicious(self, ip: str) -> bool:
        """Determine if IP is showing suspicious behavior"""
        data = self.suspicious_ips[ip]
        
        # High request rate
        if data['count'] > 1000:  # More than 1000 requests
            return True
        
        # Check time-based patterns
        time_diff = datetime.now() - data['first_seen']
        if time_diff.total_seconds() < 60 and data['count'] > 100:  # 100 requests in 1 minute
            return True
        
        return False
