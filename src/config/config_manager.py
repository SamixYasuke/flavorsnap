"""
FlavorSnap Configuration Manager

This module handles loading and managing application configuration from:
1. Default YAML configuration file
2. Environment-specific YAML configuration
3. Environment variables
4. Command-line arguments

The configuration follows this priority order (highest to lowest):
1. Command-line arguments
2. Environment variables
3. Environment-specific YAML
4. Default YAML
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import argparse

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "flavorsnap"
    username: str = "flavorsnap"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    database: int = 0

@dataclass
class SecurityConfig:
    """Security configuration."""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration: str = "24h"
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

@dataclass
class ResourceConfig:
    """Resource configuration for containers."""
    memory_request: str = "256Mi"
    memory_limit: str = "512Mi"
    cpu_request: str = "250m"
    cpu_limit: str = "500m"

@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_retention: str = "200h"
    grafana_enabled: bool = True
    grafana_port: int = 3000
    grafana_password: Optional[str] = None

@dataclass
class AppConfig:
    """Main application configuration."""
    name: str = "FlavorSnap"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    
    # Database configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    
    # Security
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Resources
    frontend_resources: ResourceConfig = field(default_factory=ResourceConfig)
    backend_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(
        memory_request="1Gi", memory_limit="2Gi", cpu_request="500m", cpu_limit="1000m"
    ))
    
    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)
    
    # Paths
    upload_folder: str = "/app/uploads"
    model_path: str = "/app/model.pth"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 5000
    model_confidence_threshold: float = 0.6
    max_file_size: int = 10485760  # 10MB
    allowed_file_types: str = "jpg,jpeg,png,webp"

class ConfigurationManager:
    """Manages application configuration loading and access."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "config")
        self._config: Optional[AppConfig] = None
        self._raw_config: Optional[Dict[str, Any]] = None
    
    def load_config(self, environment: Optional[str] = None) -> AppConfig:
        """Load configuration from files and environment variables."""
        if self._config:
            return self._config
        
        environment = environment or os.getenv("NODE_ENV", "development")
        
        # Load YAML configurations
        self._raw_config = self._load_yaml_configs(environment)
        
        # Override with environment variables
        self._apply_env_overrides()
        
        # Create AppConfig object
        self._config = self._create_app_config()
        
        logger.info(f"Configuration loaded for environment: {environment}")
        return self._config
    
    def _load_yaml_configs(self, environment: str) -> Dict[str, Any]:
        """Load and merge YAML configuration files."""
        config = {}
        
        # Load default configuration
        default_config_path = self.config_dir / "default.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded default configuration from {default_config_path}")
        else:
            logger.warning(f"Default configuration file not found: {default_config_path}")
        
        # Load environment-specific configuration
        env_config_path = self.config_dir / f"{environment}.yaml"
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
            config = self._merge_configs(config, env_config)
            logger.debug(f"Loaded {environment} configuration from {env_config_path}")
        else:
            logger.warning(f"Environment configuration file not found: {env_config_path}")
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries recursively."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides to the configuration."""
        if not self._raw_config:
            return
        
        # Environment variable mappings
        env_mappings = {
            # Application
            "NODE_ENV": ["app", "environment"],
            "DEBUG": ["app", "debug"],
            "LOG_LEVEL": ["app", "log_level"],
            
            # Database
            "POSTGRES_HOST": ["database", "postgres", "host"],
            "POSTGRES_PORT": ["database", "postgres", "port"],
            "POSTGRES_DB": ["database", "postgres", "database"],
            "POSTGRES_USER": ["database", "postgres", "username"],
            "POSTGRES_PASSWORD": ["database", "postgres", "password"],
            "POSTGRES_SSL_MODE": ["database", "postgres", "ssl_mode"],
            "DB_POOL_SIZE": ["database", "postgres", "pool_size"],
            "DB_MAX_OVERFLOW": ["database", "postgres", "max_overflow"],
            "DB_POOL_TIMEOUT": ["database", "postgres", "pool_timeout"],
            
            # Redis
            "REDIS_HOST": ["database", "redis", "host"],
            "REDIS_PORT": ["database", "redis", "port"],
            "REDIS_PASSWORD": ["database", "redis", "password"],
            "REDIS_DB": ["database", "redis", "database"],
            
            # Security
            "JWT_SECRET": ["security", "jwt", "secret"],
            "JWT_ALGORITHM": ["security", "jwt", "algorithm"],
            "JWT_EXPIRATION": ["security", "jwt", "expiration"],
            "SSL_CERT_PATH": ["security", "ssl_cert_path"],
            "SSL_KEY_PATH": ["security", "ssl_key_path"],
            
            # Monitoring
            "PROMETHEUS_ENABLED": ["monitoring", "prometheus_enabled"],
            "PROMETHEUS_PORT": ["monitoring", "prometheus_port"],
            "PROMETHEUS_RETENTION": ["monitoring", "prometheus_retention"],
            "GRAFANA_ENABLED": ["monitoring", "grafana_enabled"],
            "GRAFANA_PORT": ["monitoring", "grafana_port"],
            "GRAFANA_PASSWORD": ["monitoring", "grafana_password"],
            
            # API
            "API_HOST": ["app", "api_host"],
            "API_PORT": ["app", "api_port"],
            "MODEL_CONFIDENCE_THRESHOLD": ["app", "model_confidence_threshold"],
            "MAX_FILE_SIZE": ["app", "max_file_size"],
            "ALLOWED_FILE_TYPES": ["app", "allowed_file_types"],
            
            # Paths
            "UPLOAD_FOLDER": ["app", "upload_folder"],
            "MODEL_PATH": ["app", "model_path"],
            
            # Feature flags
            "ENABLE_ANALYTICS": ["features", "analytics"],
            "ENABLE_DARK_MODE": ["features", "dark_mode"],
            "ENABLE_CLASSIFICATION_HISTORY": ["features", "classification_history"],
            "ENABLE_SOCIAL_SHARING": ["features", "social_sharing"],
            "ENABLE_ANALYTICS_DASHBOARD": ["features", "analytics_dashboard"],
            "ENABLE_BATCH_PROCESSING": ["features", "batch_processing"],
            "ENABLE_XAI_EXPLANATIONS": ["features", "xai_explanations"],
            "ENABLE_AB_TESTING": ["features", "ab_testing"],
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(self._raw_config, config_path, self._convert_env_value(value))
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """Set a nested value in the configuration dictionary."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _create_app_config(self) -> AppConfig:
        """Create AppConfig object from raw configuration."""
        if not self._raw_config:
            return AppConfig()
        
        app_config = self._raw_config.get("app", {})
        
        # Database configuration
        db_config = self._raw_config.get("database", {})
        postgres_config = db_config.get("postgres", {})
        redis_config = db_config.get("redis", {})
        
        database = DatabaseConfig(
            host=postgres_config.get("host", "localhost"),
            port=int(postgres_config.get("port", 5432)),
            database=postgres_config.get("database", "flavorsnap"),
            username=postgres_config.get("username", "flavorsnap"),
            password=postgres_config.get("password", ""),
            ssl_mode=postgres_config.get("ssl_mode", "prefer"),
            pool_size=int(postgres_config.get("pool_size", 10)),
            max_overflow=int(postgres_config.get("max_overflow", 20)),
            pool_timeout=int(postgres_config.get("pool_timeout", 30)),
        )
        
        redis = RedisConfig(
            host=redis_config.get("host", "localhost"),
            port=int(redis_config.get("port", 6379)),
            password=redis_config.get("password"),
            database=int(redis_config.get("database", 0)),
        )
        
        # Security configuration
        security_config = self._raw_config.get("security", {})
        jwt_config = security_config.get("jwt", {})
        
        security = SecurityConfig(
            jwt_secret=jwt_config.get("secret", ""),
            jwt_algorithm=jwt_config.get("algorithm", "HS256"),
            jwt_expiration=jwt_config.get("expiration", "24h"),
            ssl_enabled=security_config.get("ssl", {}).get("enabled", False),
            ssl_cert_path=security_config.get("ssl", {}).get("cert_path"),
            ssl_key_path=security_config.get("ssl", {}).get("key_path"),
        )
        
        # Resource configurations
        frontend_config = self._raw_config.get("frontend", {})
        frontend_resources_config = frontend_config.get("resources", {})
        
        frontend_resources = ResourceConfig(
            memory_request=frontend_resources_config.get("requests", {}).get("memory", "256Mi"),
            memory_limit=frontend_resources_config.get("limits", {}).get("memory", "512Mi"),
            cpu_request=frontend_resources_config.get("requests", {}).get("cpu", "250m"),
            cpu_limit=frontend_resources_config.get("limits", {}).get("cpu", "500m"),
        )
        
        backend_config = self._raw_config.get("backend", {})
        backend_resources_config = backend_config.get("resources", {})
        
        backend_resources = ResourceConfig(
            memory_request=backend_resources_config.get("requests", {}).get("memory", "1Gi"),
            memory_limit=backend_resources_config.get("limits", {}).get("memory", "2Gi"),
            cpu_request=backend_resources_config.get("requests", {}).get("cpu", "500m"),
            cpu_limit=backend_resources_config.get("limits", {}).get("cpu", "1000m"),
        )
        
        # Monitoring configuration
        monitoring_config = self._raw_config.get("monitoring", {})
        prometheus_config = monitoring_config.get("prometheus", {})
        grafana_config = monitoring_config.get("grafana", {})
        
        monitoring = MonitoringConfig(
            prometheus_enabled=prometheus_config.get("enabled", True),
            prometheus_port=int(prometheus_config.get("port", 9090)),
            prometheus_retention=prometheus_config.get("retention", "200h"),
            grafana_enabled=grafana_config.get("enabled", True),
            grafana_port=int(grafana_config.get("port", 3000)),
            grafana_password=grafana_config.get("admin_password"),
        )
        
        # Feature flags
        features_config = self._raw_config.get("features", {})
        features = {
            "analytics": features_config.get("analytics", False),
            "dark_mode": features_config.get("dark_mode", True),
            "classification_history": features_config.get("classification_history", True),
            "social_sharing": features_config.get("social_sharing", True),
            "analytics_dashboard": features_config.get("analytics_dashboard", True),
            "batch_processing": features_config.get("batch_processing", True),
            "xai_explanations": features_config.get("xai_explanations", True),
            "ab_testing": features_config.get("ab_testing", False),
        }
        
        return AppConfig(
            name=app_config.get("name", "FlavorSnap"),
            version=app_config.get("version", "1.0.0"),
            environment=app_config.get("environment", "development"),
            debug=app_config.get("debug", False),
            log_level=app_config.get("log_level", "INFO"),
            database=database,
            redis=redis,
            security=security,
            frontend_resources=frontend_resources,
            backend_resources=backend_resources,
            monitoring=monitoring,
            features=features,
            upload_folder=app_config.get("upload_folder", "/app/uploads"),
            model_path=app_config.get("model_path", "/app/model.pth"),
            api_host=app_config.get("api_host", "0.0.0.0"),
            api_port=int(app_config.get("api_port", 5000)),
            model_confidence_threshold=float(app_config.get("model_confidence_threshold", 0.6)),
            max_file_size=int(app_config.get("max_file_size", 10485760)),
            allowed_file_types=app_config.get("allowed_file_types", "jpg,jpeg,png,webp"),
        )
    
    def get_config(self) -> AppConfig:
        """Get the loaded configuration."""
        if not self._config:
            return self.load_config()
        return self._config
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get the raw configuration dictionary."""
        if not self._raw_config:
            self.load_config()
        return self._raw_config or {}
    
    def reload_config(self, environment: Optional[str] = None) -> AppConfig:
        """Reload configuration from files and environment."""
        self._config = None
        self._raw_config = None
        return self.load_config(environment)

# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def get_config() -> AppConfig:
    """Get the application configuration."""
    return get_config_manager().get_config()

def reload_config(environment: Optional[str] = None) -> AppConfig:
    """Reload the application configuration."""
    return get_config_manager().reload_config(environment)

def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments for configuration overrides."""
    parser = argparse.ArgumentParser(description="FlavorSnap Application")
    
    # Application arguments
    parser.add_argument("--environment", choices=["development", "production", "test"],
                       help="Application environment")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN", "ERROR"],
                       help="Logging level")
    
    # Database arguments
    parser.add_argument("--postgres-host", help="PostgreSQL host")
    parser.add_argument("--postgres-port", type=int, help="PostgreSQL port")
    parser.add_argument("--postgres-db", help="PostgreSQL database name")
    parser.add_argument("--postgres-user", help="PostgreSQL username")
    parser.add_argument("--postgres-password", help="PostgreSQL password")
    
    # Redis arguments
    parser.add_argument("--redis-host", help="Redis host")
    parser.add_argument("--redis-port", type=int, help="Redis port")
    parser.add_argument("--redis-password", help="Redis password")
    
    # API arguments
    parser.add_argument("--host", help="API host")
    parser.add_argument("--port", type=int, help="API port")
    
    return parser.parse_args()

def apply_cli_overrides(args: argparse.Namespace):
    """Apply command-line argument overrides to environment variables."""
    if args.environment:
        os.environ["NODE_ENV"] = args.environment
    if args.debug:
        os.environ["DEBUG"] = "true"
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
    
    # Database overrides
    if args.postgres_host:
        os.environ["POSTGRES_HOST"] = args.postgres_host
    if args.postgres_port:
        os.environ["POSTGRES_PORT"] = str(args.postgres_port)
    if args.postgres_db:
        os.environ["POSTGRES_DB"] = args.postgres_db
    if args.postgres_user:
        os.environ["POSTGRES_USER"] = args.postgres_user
    if args.postgres_password:
        os.environ["POSTGRES_PASSWORD"] = args.postgres_password
    
    # Redis overrides
    if args.redis_host:
        os.environ["REDIS_HOST"] = args.redis_host
    if args.redis_port:
        os.environ["REDIS_PORT"] = str(args.redis_port)
    if args.redis_password:
        os.environ["REDIS_PASSWORD"] = args.redis_password
    
    # API overrides
    if args.host:
        os.environ["API_HOST"] = args.host
    if args.port:
        os.environ["API_PORT"] = str(args.port)
