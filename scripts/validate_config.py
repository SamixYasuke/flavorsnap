#!/usr/bin/env python3
"""
FlavorSnap Configuration Validation Script

This script validates the application configuration by:
1. Checking required environment variables
2. Validating YAML configuration files
3. Checking database connectivity
4. Validating SSL certificates
5. Checking resource limits
6. Validating security settings

Usage:
    python scripts/validate_config.py [--environment ENV] [--check-database] [--check-ssl]
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import psycopg2
import redis
import ssl
import socket
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Configuration validator for FlavorSnap application."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.config_dir = Path("config")
        self.required_env_vars = self._get_required_env_vars()
        
    def _get_required_env_vars(self) -> Dict[str, str]:
        """Get required environment variables by environment."""
        common_vars = {
            "NODE_ENV": "Application environment",
            "JWT_SECRET": "JWT signing secret",
            "POSTGRES_HOST": "PostgreSQL host",
            "POSTGRES_DB": "PostgreSQL database name",
            "POSTGRES_USER": "PostgreSQL username",
            "POSTGRES_PASSWORD": "PostgreSQL password",
            "REDIS_HOST": "Redis host",
        }
        
        production_vars = {
            **common_vars,
            "SSL_CERT_PATH": "SSL certificate path",
            "SSL_KEY_PATH": "SSL private key path",
            "GRAFANA_PASSWORD": "Grafana admin password",
        }
        
        if self.environment == "production":
            return production_vars
        return common_vars
    
    def validate_environment_variables(self) -> bool:
        """Validate required environment variables."""
        logger.info("Validating environment variables...")
        
        missing_vars = []
        weak_vars = []
        
        for var, description in self.required_env_vars.items():
            value = os.getenv(var)
            if not value:
                missing_vars.append(f"{var}: {description}")
            elif var == "JWT_SECRET" and len(value) < 32:
                weak_vars.append(f"{var}: Secret too short (minimum 32 characters)")
            elif var.endswith("_PASSWORD") and len(value) < 8:
                weak_vars.append(f"{var}: Password too weak (minimum 8 characters)")
        
        if missing_vars:
            self.errors.extend([f"Missing environment variable: {var}" for var in missing_vars])
            
        if weak_vars:
            self.warnings.extend(weak_vars)
            
        return len(missing_vars) == 0
    
    def validate_yaml_files(self) -> bool:
        """Validate YAML configuration files."""
        logger.info("Validating YAML configuration files...")
        
        config_files = [
            self.config_dir / "default.yaml",
            self.config_dir / f"{self.environment}.yaml"
        ]
        
        valid = True
        
        for config_file in config_files:
            if not config_file.exists():
                self.errors.append(f"Configuration file not found: {config_file}")
                valid = False
                continue
                
            try:
                with open(config_file, 'r') as f:
                    yaml.safe_load(f)
                logger.info(f"✓ {config_file} is valid YAML")
            except yaml.YAMLError as e:
                self.errors.append(f"Invalid YAML in {config_file}: {e}")
                valid = False
                
        return valid
    
    def validate_database_connectivity(self) -> bool:
        """Validate database connectivity."""
        logger.info("Validating database connectivity...")
        
        # Check PostgreSQL
        postgres_valid = self._check_postgres()
        
        # Check Redis
        redis_valid = self._check_redis()
        
        return postgres_valid and redis_valid
    
    def _check_postgres(self) -> bool:
        """Check PostgreSQL connectivity."""
        try:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = int(os.getenv("POSTGRES_PORT", "5432"))
            database = os.getenv("POSTGRES_DB", "flavorsnap")
            user = os.getenv("POSTGRES_USER", "flavorsnap")
            password = os.getenv("POSTGRES_PASSWORD", "")
            
            conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password,
                connect_timeout=5
            )
            conn.close()
            logger.info("✓ PostgreSQL connection successful")
            return True
            
        except Exception as e:
            self.errors.append(f"PostgreSQL connection failed: {e}")
            return False
    
    def _check_redis(self) -> bool:
        """Check Redis connectivity."""
        try:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            password = os.getenv("REDIS_PASSWORD")
            db = int(os.getenv("REDIS_DB", "0"))
            
            r = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                socket_timeout=5
            )
            r.ping()
            logger.info("✓ Redis connection successful")
            return True
            
        except Exception as e:
            self.errors.append(f"Redis connection failed: {e}")
            return False
    
    def validate_ssl_certificates(self) -> bool:
        """Validate SSL certificates for production environment."""
        if self.environment != "production":
            logger.info("SSL validation skipped (not production environment)")
            return True
            
        logger.info("Validating SSL certificates...")
        
        cert_path = os.getenv("SSL_CERT_PATH")
        key_path = os.getenv("SSL_KEY_PATH")
        
        if not cert_path or not key_path:
            self.errors.append("SSL certificate paths not configured for production")
            return False
            
        if not Path(cert_path).exists():
            self.errors.append(f"SSL certificate not found: {cert_path}")
            return False
            
        if not Path(key_path).exists():
            self.errors.append(f"SSL private key not found: {key_path}")
            return False
            
        try:
            # Load and validate certificate
            context = ssl.create_default_context()
            context.load_cert_chain(cert_path, key_path)
            logger.info("✓ SSL certificates are valid")
            return True
            
        except Exception as e:
            self.errors.append(f"SSL certificate validation failed: {e}")
            return False
    
    def validate_resource_limits(self) -> bool:
        """Validate resource limit configurations."""
        logger.info("Validating resource limits...")
        
        config = self._load_config()
        if not config:
            return False
            
        warnings = []
        
        # Check frontend resources
        frontend = config.get("frontend", {})
        frontend_resources = frontend.get("resources", {})
        
        frontend_memory = frontend_resources.get("limits", {}).get("memory", "256Mi")
        frontend_cpu = frontend_resources.get("limits", {}).get("cpu", "250m")
        
        if frontend_memory == "256Mi" and self.environment == "production":
            warnings.append("Frontend memory limit may be too low for production")
            
        if frontend_cpu == "250m" and self.environment == "production":
            warnings.append("Frontend CPU limit may be too low for production")
        
        # Check backend resources
        backend = config.get("backend", {})
        backend_resources = backend.get("resources", {})
        
        backend_memory = backend_resources.get("limits", {}).get("memory", "1Gi")
        backend_cpu = backend_resources.get("limits", {}).get("cpu", "500m")
        
        if backend_memory == "1Gi" and self.environment == "production":
            warnings.append("Backend memory limit may be too low for production")
            
        if backend_cpu == "500m" and self.environment == "production":
            warnings.append("Backend CPU limit may be too low for production")
        
        self.warnings.extend(warnings)
        return True
    
    def validate_security_settings(self) -> bool:
        """Validate security configuration."""
        logger.info("Validating security settings...")
        
        config = self._load_config()
        if not config:
            return False
            
        security = config.get("security", {})
        warnings = []
        
        # Check container security
        containers = security.get("containers", {})
        
        if self.environment == "production":
            if not containers.get("non_root", False):
                warnings.append("Containers should run as non-root in production")
                
            if not containers.get("read_only_rootfs", False):
                warnings.append("Containers should have read-only root filesystem in production")
        
        # Check JWT configuration
        jwt_secret = os.getenv("JWT_SECRET")
        if jwt_secret and len(jwt_secret) < 32:
            warnings.append("JWT secret should be at least 32 characters long")
        
        # Check database SSL
        postgres_ssl = os.getenv("POSTGRES_SSL_MODE", "prefer")
        if self.environment == "production" and postgres_ssl != "require":
            warnings.append("PostgreSQL should require SSL in production")
        
        self.warnings.extend(warnings)
        return True
    
    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load and merge configuration files."""
        try:
            # Load default config
            default_config_path = self.config_dir / "default.yaml"
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load environment-specific config
            env_config_path = self.config_dir / f"{self.environment}.yaml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f)
                    config = self._merge_configs(config, env_config)
            
            return config
            
        except Exception as e:
            self.errors.append(f"Failed to load configuration: {e}")
            return None
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def validate_all(self, check_database: bool = False, check_ssl: bool = False) -> bool:
        """Run all validation checks."""
        logger.info(f"Starting configuration validation for environment: {self.environment}")
        
        all_valid = True
        
        # Basic validations
        all_valid &= self.validate_environment_variables()
        all_valid &= self.validate_yaml_files()
        all_valid &= self.validate_resource_limits()
        all_valid &= self.validate_security_settings()
        
        # Optional validations
        if check_database:
            all_valid &= self.validate_database_connectivity()
            
        if check_ssl:
            all_valid &= self.validate_ssl_certificates()
        
        return all_valid
    
    def print_results(self):
        """Print validation results."""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION RESULTS")
        print("="*60)
        
        if not self.errors and not self.warnings:
            print("✅ All validations passed!")
        else:
            if self.errors:
                print(f"\n❌ ERRORS ({len(self.errors)}):")
                for error in self.errors:
                    print(f"  • {error}")
                    
            if self.warnings:
                print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
                for warning in self.warnings:
                    print(f"  • {warning}")
        
        print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate FlavorSnap configuration"
    )
    parser.add_argument(
        "--environment",
        choices=["development", "production", "test"],
        default="development",
        help="Environment to validate"
    )
    parser.add_argument(
        "--check-database",
        action="store_true",
        help="Check database connectivity"
    )
    parser.add_argument(
        "--check-ssl",
        action="store_true",
        help="Check SSL certificates"
    )
    
    args = parser.parse_args()
    
    validator = ConfigValidator(environment=args.environment)
    valid = validator.validate_all(
        check_database=args.check_database,
        check_ssl=args.check_ssl
    )
    
    validator.print_results()
    
    # Exit with appropriate code
    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
