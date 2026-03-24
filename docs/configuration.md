# FlavorSnap Configuration Guide

This comprehensive guide covers all configuration options, environment variables, and settings for the FlavorSnap application.

## Table of Contents

- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Docker Configuration](#docker-configuration)
- [Security Considerations](#security-considerations)
- [Environment-Specific Settings](#environment-specific-settings)
- [Configuration Validation](#configuration-validation)
- [Troubleshooting](#troubleshooting)

## Environment Variables

### Application Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NODE_ENV` | Application environment (development/production/test) | `development` | No |
| `DEBUG` | Enable debug mode | `false` | No |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARN/ERROR) | `INFO` | No |

### Frontend Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:5000` | Yes |
| `NEXT_PUBLIC_MODEL_ENDPOINT` | Model prediction endpoint | `/predict` | No |
| `MAX_FILE_SIZE` | Maximum file upload size in bytes | `10485760` | No |
| `ALLOWED_FILE_TYPES` | Allowed file extensions | `jpg,jpeg,png,webp` | No |
| `MODEL_CONFIDENCE_THRESHOLD` | Minimum confidence for predictions | `0.6` | No |

### Backend Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FLASK_ENV` | Flask environment | `development` | No |
| `FLASK_DEBUG` | Enable Flask debug mode | `1` | No |
| `PYTHONPATH` | Python path | `/app` | No |
| `MODEL_BATCH_SIZE` | Model batch processing size | `32` | No |
| `MAX_IMAGE_SIZE` | Maximum image dimensions | `224` | No |

### Database Configuration

#### PostgreSQL
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_HOST` | PostgreSQL host | `localhost` | Yes |
| `POSTGRES_PORT` | PostgreSQL port | `5432` | No |
| `POSTGRES_DB` | Database name | `flavorsnap` | Yes |
| `POSTGRES_USER` | Database username | `flavorsnap` | Yes |
| `POSTGRES_PASSWORD` | Database password | - | Yes |
| `POSTGRES_SSL_MODE` | SSL connection mode | `prefer` | No |

#### Database Pool Settings
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DB_POOL_SIZE` | Connection pool size | `10` | No |
| `DB_MAX_OVERFLOW` | Maximum overflow connections | `20` | No |
| `DB_POOL_TIMEOUT` | Pool timeout in seconds | `30` | No |

#### Redis
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_HOST` | Redis host | `localhost` | Yes |
| `REDIS_PORT` | Redis port | `6379` | No |
| `REDIS_PASSWORD` | Redis password | - | No |
| `REDIS_DB` | Redis database number | `0` | No |

### Security Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `JWT_SECRET` | JWT signing secret | - | Yes |
| `SSL_CERT_PATH` | SSL certificate path | - | No |
| `SSL_KEY_PATH` | SSL private key path | - | No |

### Monitoring & Observability

#### Prometheus
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `PROMETHEUS_ENABLED` | Enable Prometheus metrics | `true` | No |
| `PROMETHEUS_RETENTION` | Data retention period | `200h` | No |
| `SCRAPE_INTERVAL` | Metrics scrape interval | `15s` | No |

#### Grafana
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `GRAFANA_ENABLED` | Enable Grafana dashboard | `true` | No |
| `GRAFANA_PASSWORD` | Grafana admin password | - | Yes |

### Performance & Scaling

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AUTOSCALING_ENABLED` | Enable auto-scaling | `true` | No |
| `MIN_REPLICAS` | Minimum replica count | `1` | No |
| `MAX_REPLICAS` | Maximum replica count | `10` | No |
| `TARGET_CPU_UTILIZATION` | Target CPU utilization (%) | `70` | No |
| `TARGET_MEMORY_UTILIZATION` | Target memory utilization (%) | `80` | No |
| `CACHING_ENABLED` | Enable caching | `true` | No |
| `CACHE_TTL` | Cache TTL | `3600s` | No |
| `CACHE_MAX_SIZE` | Maximum cache size | `100MB` | No |

### File Storage

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `UPLOAD_FOLDER` | Upload directory path | `/app/uploads` | No |
| `MAX_UPLOAD_SIZE` | Maximum upload size | `50MB` | No |
| `UPLOAD_CLEANUP_INTERVAL` | Cleanup interval | `24h` | No |
| `MODEL_PATH` | Model file path | `/app/model.pth` | No |
| `MODEL_BACKUP_INTERVAL` | Model backup interval | `7d` | No |

### Container Resources

#### Frontend
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `FRONTEND_REPLICAS` | Number of frontend replicas | `1` | No |
| `FRONTEND_MEMORY_REQUEST` | Memory request | `256Mi` | No |
| `FRONTEND_MEMORY_LIMIT` | Memory limit | `512Mi` | No |
| `FRONTEND_CPU_REQUEST` | CPU request | `250m` | No |
| `FRONTEND_CPU_LIMIT` | CPU limit | `500m` | No |

#### Backend
| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BACKEND_REPLICAS` | Number of backend replicas | `1` | No |
| `BACKEND_MEMORY_REQUEST` | Memory request | `1Gi` | No |
| `BACKEND_MEMORY_LIMIT` | Memory limit | `2Gi` | No |
| `BACKEND_CPU_REQUEST` | CPU request | `500m` | No |
| `BACKEND_CPU_LIMIT` | CPU limit | `1000m` | No |

### Feature Flags

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENABLE_ANALYTICS` | Enable analytics | `false` | No |
| `ENABLE_DARK_MODE` | Enable dark mode | `true` | No |
| `ENABLE_CLASSIFICATION_HISTORY` | Enable classification history | `true` | No |
| `ENABLE_SOCIAL_SHARING` | Enable social sharing | `true` | No |
| `ENABLE_ANALYTICS_DASHBOARD` | Enable analytics dashboard | `true` | No |
| `ENABLE_BATCH_PROCESSING` | Enable batch processing | `true` | No |
| `ENABLE_XAI_EXPLANATIONS` | Enable XAI explanations | `true` | No |
| `ENABLE_AB_TESTING` | Enable A/B testing | `false` | No |

### Health Check Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HEALTH_CHECK_INTERVAL` | Health check interval | `30s` | No |
| `HEALTH_CHECK_TIMEOUT` | Health check timeout | `10s` | No |
| `HEALTH_CHECK_RETRIES` | Health check retry count | `3` | No |
| `HEALTH_CHECK_START_PERIOD` | Health check start period | `30s` | No |

### Rate Limiting

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | API rate limit per minute | `60` | No |
| `RATE_LIMIT_BURST` | Rate limit burst size | `10` | No |
| `UPLOAD_RATE_LIMIT_PER_MINUTE` | Upload rate limit per minute | `10` | No |
| `UPLOAD_RATE_LIMIT_BURST` | Upload rate limit burst | `3` | No |

## Configuration Files

### YAML Configuration Structure

The application uses YAML configuration files to define settings for different environments:

- `config/default.yaml` - Default configuration
- `config/development.yaml` - Development-specific overrides
- `config/production.yaml` - Production-specific overrides

### Configuration Hierarchy

1. **Default Configuration** (`config/default.yaml`)
2. **Environment Configuration** (`config/{environment}.yaml`)
3. **Environment Variables** (`.env` file)
4. **Command-line Arguments** (highest priority)

### Configuration Sections

#### Application Settings
```yaml
app:
  name: "FlavorSnap"
  version: "1.0.0"
  environment: "${NODE_ENV:-development}"
  debug: "${DEBUG:-false}"
```

#### Docker Configuration
```yaml
docker:
  registry: "${DOCKER_REGISTRY:-}"
  image_tag: "${IMAGE_TAG:-latest}"
  build:
    no_cache: false
    parallel: true
    pull: true
```

#### Frontend Configuration
```yaml
frontend:
  container:
    name: "flavorsnap-frontend"
    port: 3000
    replicas: "${FRONTEND_REPLICAS:-1}"
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"
```

## Docker Configuration

### Environment Files

Copy `.env.example` to `.env` and customize for your environment:

```bash
cp .env.example .env
```

### Docker Compose

The application includes several Docker Compose files:

- `docker-compose.yml` - Development environment
- `docker-compose.prod.yml` - Production environment
- `docker-compose.test.yml` - Testing environment

### Container Images

Build images for different environments:

```bash
# Development
docker-compose build

# Production
docker-compose -f docker-compose.prod.yml build
```

## Security Considerations

### Required Security Settings

1. **JWT Secret**: Generate a strong, random JWT secret:
   ```bash
   openssl rand -base64 64
   ```

2. **Database Passwords**: Use strong, unique passwords for all database services.

3. **SSL/TLS**: In production, always use SSL/TLS certificates.

4. **Environment Variables**: Never commit secrets to version control.

### Security Best Practices

- Use read-only root filesystems in production
- Run containers as non-root users
- Implement proper network segmentation
- Regularly update dependencies
- Use secrets management for sensitive data

### Container Security

```yaml
security:
  containers:
    non_root: true
    read_only_rootfs: true
    drop_capabilities:
      - "ALL"
    add_capabilities:
      - "CHOWN"
      - "SETGID"
      - "SETUID"
```

## Environment-Specific Settings

### Development Environment

- Debug mode enabled
- Verbose logging
- Local database connections
- Hot reload enabled
- Reduced resource limits

### Production Environment

- Debug mode disabled
- Optimized logging
- Secure database connections
- SSL/TLS enabled
- Increased resource limits
- Security hardening

### Test Environment

- Isolated test database
- Mock external services
- Minimal resource usage
- Fast startup times

## Configuration Validation

### Validation Script

Use the provided validation script to check your configuration:

```bash
python scripts/validate_config.py
```

### Validation Checks

The script validates:

- Required environment variables
- Configuration file syntax
- Resource limit consistency
- Security configuration
- Database connectivity
- SSL certificate validity

### Automated Validation

Configure CI/CD pipeline to run validation:

```yaml
- name: Validate Configuration
  run: |
    python scripts/validate_config.py --environment production
```

## Troubleshooting

### Common Issues

#### Configuration Not Loading
- Check file permissions
- Verify YAML syntax
- Ensure environment variables are set
- Check configuration file paths

#### Database Connection Issues
- Verify database credentials
- Check network connectivity
- Confirm database is running
- Review SSL settings

#### Resource Limits
- Monitor container resource usage
- Adjust memory/CPU limits
- Check host system resources
- Review autoscaling settings

### Debug Configuration

Enable debug mode for detailed logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

### Configuration Debug Commands

```bash
# Check loaded configuration
python -c "from config.config_manager import get_config; print(get_config())"

# Validate specific environment
python scripts/validate_config.py --environment development

# Test database connection
python scripts/validate_config.py --check-database
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Configuration](https://kubernetes.io/docs/concepts/configuration/)
- [Environment Variables Best Practices](https://12factor.net/config)
- [Security Guidelines](https://owasp.org/)
