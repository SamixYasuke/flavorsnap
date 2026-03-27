# FlavorSnap Configuration Management

This directory contains the environment-specific configuration management system for FlavorSnap.

## Structure

```
config/
├── environments/
│   ├── development.json    # Development environment configuration
│   ├── staging.json       # Staging environment configuration
│   └── production.json    # Production environment configuration
├── config_manager.py      # Python configuration manager
└── README.md             # This file
```

## Environment Files

Each environment file contains configuration for:

- **app**: Application settings (name, version, debug mode)
- **api**: API endpoints and timeouts
- **database**: Database connection settings
- **stellar**: Stellar blockchain configuration
- **ml**: Machine learning model settings
- **upload**: File upload configuration
- **security**: Security settings (JWT, rate limiting)
- **logging**: Logging configuration
- **features**: Feature flags

## Environment Variables

The system supports environment variable substitution using `${VAR_NAME}` syntax. For example:

```json
{
  "database": {
    "url": "${DATABASE_URL}"
  }
}
```

## Usage

### Python (Dashboard)

```python
from config.config_manager import get_config, is_feature_enabled

# Get configuration value
model_path = get_config('ml.model_path', 'models/best_model.pth')

# Check if feature is enabled
if is_feature_enabled('ml_classification'):
    # Load and use model
    pass
```

### TypeScript/JavaScript (Frontend)

```typescript
import { config, isFeatureEnabled } from '../lib/config';

// Get configuration value
const modelPath = config.ml.modelPath;

// Check if feature is enabled
if (isFeatureEnabled('mlClassification')) {
  // Enable ML features
}
```

## Environment Setup

### 1. Copy the template

```bash
cp .env.example .env.local  # For development
cp .env.example .env.staging  # For staging
cp .env.example .env.production  # For production
```

### 2. Set environment variables

Edit the copied file with your actual values:

```bash
# Required for all environments
NODE_ENV=development
JWT_SECRET=your-secure-jwt-secret

# Required for staging/production
DATABASE_URL=postgresql://user:pass@host:5432/db
ENCRYPTION_KEY=your-32-byte-hex-key

# Optional
STELLAR_NETWORK=testnet
SOROBAN_RPC_URL=https://soroban-testnet.stellar.org
CONTRACT_ID=your-contract-id
```

### 3. Generate encryption key (production only)

```python
from config.config_manager import secrets_manager
import base64

key = base64.urlsafe_b64encode(secrets_manager.cipher_suite.key).decode()
print(f"ENCRYPTION_KEY={key}")
```

## Secrets Management

The system includes a secrets manager for sensitive data:

### Encryption/Decryption

```python
from config.config_manager import encrypt_text, decrypt_text

# Encrypt sensitive data
encrypted = encrypt_text("sensitive-data")

# Decrypt when needed
decrypted = decrypt_text(encrypted)
```

### Secure Environment Variables

Store encrypted values in environment files:

```bash
# Encrypt the value first
ENCRYPTED_VALUE=enc:encrypted-data-here
```

The system will automatically decrypt when accessed via `get_secure_env_var()`.

## Feature Flags

Control features without code changes:

```json
{
  "features": {
    "ml_classification": true,
    "blockchain_integration": false,
    "analytics": true,
    "cache_enabled": false
  }
}
```

## Environment Differences

### Development
- Debug mode enabled
- Local database
- Testnet Stellar network
- Verbose logging
- Hot reload enabled

### Staging
- Debug mode disabled
- Staging database
- Testnet Stellar network
- Info-level logging
- Production-like security settings

### Production
- Debug mode disabled
- Production database
- Mainnet Stellar network
- Error-only logging
- Maximum security settings
- Performance optimizations

## Validation

The system validates required secrets on startup:

```python
from config.config_manager import secrets_manager

# Validate environment-specific secrets
validation = secrets_manager.validate_environment_secrets()
if not validation['valid']:
    for error in validation['errors']:
        print(f"Configuration error: {error}")
```

## Security Best Practices

1. **Never commit actual .env files** - Only commit .env.example
2. **Use strong secrets** - Minimum 32 characters for JWT_SECRET
3. **Rotate keys regularly** - Especially in production
4. **Use different secrets per environment**
5. **Limit access** - Only authorized personnel should know production secrets
6. **Audit access** - Monitor who accesses configuration

## Troubleshooting

### Common Issues

1. **Missing configuration file**
   - Check that the correct environment file exists
   - Verify NODE_ENV is set correctly

2. **Secret decryption failed**
   - Ensure ENCRYPTION_KEY is correct
   - Check that the encrypted value format is correct (enc:base64)

3. **Feature not working**
   - Verify the feature flag is enabled in the environment config
   - Check that required dependencies are available

### Debug Mode

Enable debug mode for detailed configuration logging:

```json
{
  "app": {
    "debug": true
  },
  "logging": {
    "level": "debug"
  }
}
```

## Migration from Old Configuration

If you're migrating from hardcoded configuration:

1. Identify all hardcoded values
2. Move them to the appropriate environment file
3. Use environment variables for sensitive data
4. Update code to use the configuration manager
5. Test in each environment
6. Remove old hardcoded values

This ensures consistency across environments and improves security.
