# FlavorSnap API Security Implementation

This document outlines the comprehensive security measures implemented for the FlavorSnap Flask API to address issue #85: API Rate Limiting and Security.

## 🔐 Security Features Implemented

### 1. Rate Limiting
- **Default Limit**: 100 requests per minute per IP
- **Predict Endpoint**: 100 requests per minute
- **Health Check**: 1000 requests per hour (exempt from default limit)
- **API Info**: 60 requests per minute
- **Admin Endpoints**: 10 requests per minute

### 2. Input Validation and Sanitization
- **File Upload Validation**:
  - Allowed extensions: `png`, `jpg`, `jpeg`, `gif`, `webp`
  - Allowed MIME types: `image/jpeg`, `image/png`, `image/gif`, `image/webp`
  - Maximum file size: 16MB
  - Secure filename generation with timestamps
  - Malicious filename detection

### 3. CORS Configuration
- **Development**: `http://localhost:3000`
- **Production**: `https://yourdomain.com`
- **Allowed Methods**: GET, POST, PUT, DELETE
- **Allowed Headers**: Content-Type, Authorization, X-API-Key
- **Credentials Support**: Enabled

### 4. API Key Authentication
- **Header**: `X-API-Key`
- **Format**: 32+ character alphanumeric string
- **Development**: Bypassed for testing
- **Production**: Required for all protected endpoints
- **Admin Endpoint**: `/admin/api-key/generate` for key generation

### 5. Security Headers
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`

## 📁 File Structure

```
ml-model-api/
├── app.py                 # Main Flask application with security
├── security_config.py     # Security configuration and utilities
├── test_security.py       # Comprehensive security test suite
├── test_rate_limiter.py   # Rate limiting tests
├── .env.example          # Environment configuration template
└── requirements.txt      # Updated dependencies
```

## 🚀 Setup and Configuration

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the environment template and configure:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```bash
# Security Configuration
SECRET_KEY=your-secret-key-here-change-in-production
API_KEYS=your-api-key-1,your-api-key-2
REDIS_URL=redis://localhost:6379

# CORS Settings
CORS_ORIGINS=https://yourdomain.com
```

### 3. Redis Setup (for rate limiting)
```bash
# Install Redis
sudo apt-get install redis-server  # Ubuntu/Debian
brew install redis                 # macOS

# Start Redis
redis-server
```

### 4. Run the Application
```bash
cd ml-model-api
python app.py
```

## 🔧 API Usage

### Authentication
Include the API key in the request header:
```bash
curl -X POST http://localhost:5000/predict \
  -H "X-API-Key: your-api-key-here" \
  -F "image=@food.jpg"
```

### Endpoints

#### Health Check (No Auth Required)
```bash
GET /health
```

#### Food Classification (API Key Required)
```bash
POST /predict
Headers: X-API-Key: <your-api-key>
Body: multipart/form-data with 'image' file
```

#### API Information
```bash
GET /api/info
```

#### Generate API Key (Admin)
```bash
POST /admin/api-key/generate
```

## 🧪 Testing

### Run Security Tests
```bash
# Comprehensive security test suite
python test_security.py

# Rate limiting tests
python test_rate_limiter.py
```

### Test Categories
1. **Rate Limiting Tests**
   - Endpoint-specific limits
   - Concurrent request handling
   - Header verification

2. **Authentication Tests**
   - Missing API key
   - Invalid API key
   - Valid API key

3. **Input Validation Tests**
   - File upload validation
   - Malicious filename detection
   - File type verification

4. **Security Headers Tests**
   - Header presence verification
   - Header value validation

5. **Error Handling Tests**
   - 404 handling
   - Large file upload
   - Malformed requests

## 🛡️ Security Monitoring

### Logging
- Security events logged with IP addresses
- Failed authentication attempts
- Suspicious activity detection
- Rate limit violations

### Monitoring Features
- IP-based request tracking
- Suspicious pattern detection
- High-frequency request alerts
- Request size monitoring

## 🔒 Production Deployment

### Security Checklist
- [ ] Set strong `SECRET_KEY`
- [ ] Configure production `API_KEYS`
- [ ] Set `FLASK_ENV=production`
- [ ] Configure Redis cluster for rate limiting
- [ ] Set up proper CORS origins
- [ ] Enable HTTPS (required for HSTS)
- [ ] Configure reverse proxy (nginx/Apache)
- [ ] Set up monitoring and alerting
- [ ] Regular security audits

### Environment Variables
```bash
export FLASK_ENV=production
export SECRET_KEY="your-strong-secret-key"
export API_KEYS="prod-key-1,prod-key-2"
export REDIS_URL="redis://your-redis-cluster:6379"
export CORS_ORIGINS="https://yourdomain.com"
```

## 📊 Rate Limiting Details

### Storage Backend
- **Development**: In-memory storage
- **Production**: Redis (recommended)

### Key Generation
- Rate limit keys: `rate_limit:{endpoint}:{ip}`
- IP detection with proxy support
- X-Forwarded-For header parsing

### Response Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## 🔄 API Key Management

### Generate New Keys
```bash
curl -X POST http://localhost:5000/admin/api-key/generate
```

### Key Format
- 32+ characters
- URL-safe base64 encoding
- Cryptographically secure random generation

### Key Storage
- Hash keys using SHA-256
- Store only hashed values
- Use HMAC for verification

## 🚨 Security Considerations

### File Upload Security
- MIME type verification
- File extension validation
- Size limitations
- Secure filename generation
- Content scanning (optional)

### Rate Limiting Bypasses
- IP rotation detection
- Proxy header parsing
- Distributed attack detection
- User-based limiting (future enhancement)

### Input Sanitization
- HTML tag removal
- Script injection prevention
- Path traversal protection
- Command injection prevention

## 📈 Performance Impact

### Rate Limiting Overhead
- Redis lookup: ~1ms
- Memory usage: ~1KB per active IP
- Minimal impact on response times

### Security Headers Overhead
- Header addition: <0.1ms
- No significant performance impact

### Input Validation Overhead
- File validation: ~5ms
- MIME type checking: ~2ms
- Overall impact: <10ms per request

## 🔄 Continuous Security

### Regular Updates
- Dependency updates
- Security patch application
- Configuration reviews
- Log analysis

### Monitoring Alerts
- High rate limit violations
- Authentication failures
- Suspicious IP patterns
- Large file upload attempts

## 📞 Support

For security-related issues:
1. Check the logs for error details
2. Run the security test suite
3. Verify environment configuration
4. Review rate limiting settings
5. Check API key validity

## 📝 License

This security implementation follows the same license as the main FlavorSnap project.
