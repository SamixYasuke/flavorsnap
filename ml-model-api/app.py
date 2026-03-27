from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import os
import time
import logging
from functools import wraps
from datetime import datetime

# Import security modules
from security_config import (
    SecurityConfig, InputValidator, APIKeyManager, 
    RateLimitManager, SecurityMiddleware, SecurityMonitor,
    validate_json_input, is_safe_url
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    API_KEYS = os.environ.get('API_KEYS', '').split(',') if os.environ.get('API_KEYS') else []
    RATE_LIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    UPLOAD_FOLDER = 'uploads'
    ENV = os.environ.get('FLASK_ENV', 'development')

app.config.from_object(Config)

# Initialize security middleware
security_middleware = SecurityMiddleware(app)
security_monitor = SecurityMonitor(app)

# Initialize rate limiter
limiter = Limiter(
    app,
    key_func=RateLimitManager.get_client_ip,
    storage_uri=app.config['RATE_LIMIT_STORAGE_URL'],
    default_limits=[SecurityConfig.RATE_LIMITS['default']]
)

# CORS Configuration
CORS(app, 
     origins=['http://localhost:3000', 'https://yourdomain.com'],
     methods=['GET', 'POST', 'PUT', 'DELETE'],
     allow_headers=['Content-Type', 'Authorization', 'X-API-Key'],
     supports_credentials=True)

# API Key Authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Skip API key check in development
        if app.config.get('ENV') == 'development':
            return f(*args, **kwargs)
            
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            logger.warning(f"Missing API key from {RateLimitManager.get_client_ip()}")
            return jsonify({'error': 'API key required'}), 401
        
        # Validate API key format
        if not InputValidator.validate_api_key(api_key):
            logger.warning(f"Invalid API key format from {RateLimitManager.get_client_ip()}")
            return jsonify({'error': 'Invalid API key format'}), 401
        
        if api_key not in app.config['API_KEYS']:
            logger.warning(f"Unauthorized API key from {RateLimitManager.get_client_ip()}")
            return jsonify({'error': 'Invalid API key'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

# Rate Limiting for Specific Endpoints
@app.route('/predict', methods=['POST'])
@limiter.limit(SecurityConfig.RATE_LIMITS['predict'])
@require_api_key
def predict():
    """Food classification endpoint with comprehensive security measures"""
    start_time = time.time()
    
    try:
        # Validate input using security module
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        # Validate file upload
        is_valid, error_msg = InputValidator.validate_file_upload(file)
        if not is_valid:
            logger.warning(f"File validation failed: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Generate secure filename
        filename = InputValidator.secure_filename_custom(file.filename)
        
        # Mock prediction logic (replace with actual model)
        processing_time = time.time() - start_time
        
        logger.info(f"Prediction completed for {filename} in {processing_time:.3f}s")
        
        return jsonify({
            'prediction': 'Sample food',
            'confidence': 0.95,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'request_id': request.headers.get('X-Request-ID', 'unknown')
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint (exempt from rate limiting)
@app.route('/health', methods=['GET'])
@limiter.exempt
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'security': {
            'rate_limiting': True,
            'api_key_auth': True,
            'security_headers': True,
            'input_validation': True
        }
    })

# API info endpoint
@app.route('/api/info', methods=['GET'])
@limiter.limit(SecurityConfig.RATE_LIMITS['api_info'])
def api_info():
    return jsonify({
        'name': 'FlavorSnap API',
        'version': '2.0.0',
        'endpoints': {
            'predict': 'POST /predict - Food classification',
            'health': 'GET /health - Health check',
            'info': 'GET /api/info - API information'
        },
        'security': {
            'rate_limiting': SecurityConfig.RATE_LIMITS,
            'api_key_auth': 'Required for production',
            'cors_enabled': True,
            'security_headers': True,
            'input_validation': True,
            'file_upload_limits': {
                'max_size': f"{SecurityConfig.MAX_CONTENT_LENGTH} bytes",
                'allowed_types': list(SecurityConfig.ALLOWED_MIME_TYPES)
            }
        }
    })

# API key generation endpoint (admin only)
@app.route('/admin/api-key/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate_api_key():
    """Generate new API key (admin endpoint)"""
    if app.config.get('ENV') == 'production':
        # Add admin authentication here
        pass
    
    new_key = APIKeyManager.generate_api_key()
    logger.info("New API key generated")
    
    return jsonify({
        'api_key': new_key,
        'message': 'Store this key securely - it will not be shown again'
    })

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description),
        'retry_after': getattr(e, 'retry_after', 60),
        'limit': SecurityConfig.RATE_LIMITS['default']
    }), 429

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def request_too_large_handler(e):
    return jsonify({
        'error': 'Request too large',
        'max_size': f"{SecurityConfig.MAX_CONTENT_LENGTH} bytes"
    }), 413

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
