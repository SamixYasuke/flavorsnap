"""
FlavorSnap Configuration Module

This module provides configuration management for the FlavorSnap project.
It handles loading, validation, and environment-specific configuration
for all components of the system.

Usage:
    from flavorsnap import load_config, validate_config
    
    config = load_config('development')
    validate_config(config)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "default.yaml"
PROJECT_ROOT = Path(__file__).parent.parent

class ConfigError(Exception):
    """Configuration-related errors."""
    pass

def load_config(environment: str = "development", config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration for the specified environment.
    
    Args:
        environment: Environment name (development, staging, production)
        config_path: Optional custom config file path
        
    Returns:
        dict: Loaded configuration
        
    Raises:
        ConfigError: If configuration cannot be loaded
    """
    try:
        # Start with default configuration
        config = {}
        
        # Load default configuration
        if DEFAULT_CONFIG_FILE.exists():
            with open(DEFAULT_CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Default config file not found: {DEFAULT_CONFIG_FILE}")
        
        # Override with environment-specific configuration
        env_config_file = DEFAULT_CONFIG_DIR / f"{environment}.yaml"
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = yaml.safe_load(f) or {}
                config = _merge_configs(config, env_config)
        else:
            logger.warning(f"Environment config file not found: {env_config_file}")
        
        # Override with custom config if provided
        if config_path:
            custom_config_path = Path(config_path)
            if custom_config_path.exists():
                with open(custom_config_path, 'r') as f:
                    custom_config = yaml.safe_load(f) or {}
                    config = _merge_configs(config, custom_config)
            else:
                raise ConfigError(f"Custom config file not found: {config_path}")
        
        # Override with environment variables
        config = _override_with_env_vars(config)
        
        logger.info(f"Configuration loaded for environment: {environment}")
        return config
        
    except Exception as e:
        raise ConfigError(f"Failed to load configuration: {e}")

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the loaded configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ConfigError: If configuration is invalid
    """
    errors = []
    
    # Validate required sections
    required_sections = ['application', 'database', 'model']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate application configuration
    if 'application' in config:
        app_config = config['application']
        if 'name' not in app_config:
            errors.append("Missing application name")
        if 'version' not in app_config:
            errors.append("Missing application version")
    
    # Validate database configuration
    if 'database' in config:
        db_config = config['database']
        if 'url' not in db_config and 'host' not in db_config:
            errors.append("Database connection information missing")
    
    # Validate model configuration
    if 'model' in config:
        model_config = config['model']
        if 'path' not in model_config:
            errors.append("Model path not specified")
        else:
            model_path = PROJECT_ROOT / model_config['path']
            if not model_path.exists():
                errors.append(f"Model file not found: {model_path}")
    
    if errors:
        raise ConfigError(f"Configuration validation failed: {'; '.join(errors)}")
    
    logger.info("Configuration validation passed")
    return True

def get_environment_config() -> Dict[str, Any]:
    """
    Get configuration based on current environment.
    
    Returns:
        dict: Environment-specific configuration
    """
    environment = os.getenv('NODE_ENV', os.getenv('ENVIRONMENT', 'development'))
    return load_config(environment)

def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        dict: Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def _override_with_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration values with environment variables.
    
    Args:
        config: Base configuration
        
    Returns:
        dict: Configuration with environment variable overrides
    """
    # Environment variable mappings
    env_mappings = {
        'NODE_ENV': ['application', 'environment'],
        'DEBUG': ['application', 'debug'],
        'PORT': ['application', 'port'],
        'DATABASE_URL': ['database', 'url'],
        'DATABASE_HOST': ['database', 'host'],
        'DATABASE_PORT': ['database', 'port'],
        'DATABASE_NAME': ['database', 'name'],
        'DATABASE_USER': ['database', 'user'],
        'DATABASE_PASSWORD': ['database', 'password'],
        'MODEL_PATH': ['model', 'path'],
        'MODEL_CONFIDENCE_THRESHOLD': ['model', 'confidence_threshold'],
        'JWT_SECRET': ['security', 'jwt_secret'],
        'API_RATE_LIMIT': ['api', 'rate_limit'],
        'LOG_LEVEL': ['logging', 'level'],
    }
    
    result = config.copy()
    
    for env_var, config_path in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            # Navigate to the nested configuration path
            current = result
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the final value
            final_key = config_path[-1]
            
            # Type conversion
            if final_key in ['port', 'database_port']:
                try:
                    env_value = int(env_value)
                except ValueError:
                    continue
            elif final_key in ['debug']:
                env_value = env_value.lower() in ('true', '1', 'yes', 'on')
            elif final_key in ['confidence_threshold', 'rate_limit']:
                try:
                    env_value = float(env_value)
                except ValueError:
                    continue
            
            current[final_key] = env_value
    
    return result

def save_config(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        file_path: Output file path
        
    Raises:
        ConfigError: If configuration cannot be saved
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to: {file_path}")
        
    except Exception as e:
        raise ConfigError(f"Failed to save configuration: {e}")

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration template.
    
    Returns:
        dict: Default configuration
    """
    return {
        'application': {
            'name': 'FlavorSnap',
            'version': '1.0.0',
            'environment': 'development',
            'debug': True,
            'port': 3000
        },
        'database': {
            'host': 'localhost',
            'port': 5432,
            'name': 'flavorsnap',
            'user': 'flavorsnap',
            'password': 'password'
        },
        'model': {
            'path': 'model.pth',
            'confidence_threshold': 0.6,
            'input_size': [224, 224],
            'batch_size': 1
        },
        'api': {
            'rate_limit': 100,
            'timeout': 30,
            'max_file_size': 10485760  # 10MB
        },
        'security': {
            'jwt_secret': 'your-secret-key',
            'cors_origins': ['http://localhost:3000'],
            'allowed_file_types': ['jpg', 'jpeg', 'png', 'webp']
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'features': {
            'analytics': True,
            'social_sharing': True,
            'dark_mode': True,
            'internationalization': True
        }
    }
