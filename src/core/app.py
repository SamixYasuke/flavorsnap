"""
FlavorSnap Core Application

This module contains the main application initialization and configuration validation logic.
It handles:
- Application startup and shutdown
- Configuration validation
- Service initialization
- Error handling and logging setup
"""

import os
import sys
import logging
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_manager import get_config, AppConfig, parse_cli_args, apply_cli_overrides
from scripts.validate_config import ConfigValidator

logger = logging.getLogger(__name__)

class FlavorSnapApp:
    """Main FlavorSnap application class."""
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.validator: Optional[ConfigValidator] = None
        self._shutdown_handlers = []
        
    def initialize(self, validate_config: bool = True) -> bool:
        """Initialize the application."""
        try:
            # Parse command-line arguments
            args = parse_cli_args()
            apply_cli_overrides(args)
            
            # Load configuration
            self.config = get_config()
            self.validator = ConfigValidator(environment=self.config.environment)
            
            # Setup logging
            self._setup_logging()
            
            logger.info(f"Initializing FlavorSnap v{self.config.version} in {self.config.environment} mode")
            
            # Validate configuration if requested
            if validate_config:
                if not self._validate_configuration():
                    logger.error("Configuration validation failed")
                    return False
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            logger.info("Application initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize application: {e}")
            traceback.print_exc()
            return False
    
    def _setup_logging(self):
        """Setup application logging."""
        if not self.config:
            return
            
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'/app/logs/flavorsnap.log') if Path('/app/logs').exists() else logging.NullHandler()
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        
        logger.debug(f"Logging configured at level: {self.config.log_level}")
    
    def _validate_configuration(self) -> bool:
        """Validate application configuration."""
        if not self.validator:
            return False
            
        logger.info("Validating application configuration...")
        
        # Run basic validation
        valid = self.validator.validate_all()
        
        if not valid:
            logger.error("Configuration validation found errors:")
            for error in self.validator.errors:
                logger.error(f"  - {error}")
        
        if self.validator.warnings:
            logger.warning("Configuration validation found warnings:")
            for warning in self.validator.warnings:
                logger.warning(f"  - {warning}")
        
        return valid
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        logger.debug("Signal handlers configured")
    
    def validate_startup_requirements(self) -> bool:
        """Validate requirements for application startup."""
        if not self.config:
            logger.error("Configuration not loaded")
            return False
        
        logger.info("Validating startup requirements...")
        
        # Check required directories
        required_dirs = [
            Path(self.config.upload_folder),
            Path(self.config.model_path).parent,
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to create directory {dir_path}: {e}")
                    return False
        
        # Check model file
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        logger.info(f"Model file found: {model_path}")
        
        # Check database connectivity (optional in development)
        if self.config.environment == "production":
            try:
                if not self.validator.validate_database_connectivity():
                    logger.error("Database connectivity validation failed")
                    return False
            except Exception as e:
                logger.error(f"Database validation error: {e}")
                return False
        
        # Check SSL certificates in production
        if self.config.environment == "production" and self.config.security.ssl_enabled:
            try:
                if not self.validator.validate_ssl_certificates():
                    logger.error("SSL certificate validation failed")
                    return False
            except Exception as e:
                logger.error(f"SSL validation error: {e}")
                return False
        
        logger.info("Startup requirements validation completed successfully")
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get application health status."""
        if not self.config:
            return {"status": "error", "message": "Configuration not loaded"}
        
        health_info = {
            "status": "healthy",
            "version": self.config.version,
            "environment": self.config.environment,
            "timestamp": str(os.times()),
            "services": {}
        }
        
        # Check database connectivity
        try:
            if self.validator and self.validator.validate_database_connectivity():
                health_info["services"]["database"] = {"status": "healthy"}
            else:
                health_info["services"]["database"] = {"status": "unhealthy"}
                health_info["status"] = "degraded"
        except Exception as e:
            health_info["services"]["database"] = {"status": "error", "error": str(e)}
            health_info["status"] = "unhealthy"
        
        # Check model file
        model_path = Path(self.config.model_path)
        health_info["services"]["model"] = {
            "status": "healthy" if model_path.exists() else "error",
            "path": str(model_path),
            "exists": model_path.exists()
        }
        
        if not model_path.exists():
            health_info["status"] = "unhealthy"
        
        # Check upload directory
        upload_path = Path(self.config.upload_folder)
        health_info["services"]["storage"] = {
            "status": "healthy" if upload_path.exists() else "error",
            "path": str(upload_path),
            "exists": upload_path.exists(),
            "writable": upload_path.exists() and os.access(upload_path, os.W_OK)
        }
        
        if not upload_path.exists() or not os.access(upload_path, os.W_OK):
            health_info["status"] = "degraded"
        
        return health_info
    
    def add_shutdown_handler(self, handler):
        """Add a shutdown handler."""
        self._shutdown_handlers.append(handler)
    
    def shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Initiating application shutdown...")
        
        # Call shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")
        
        logger.info("Application shutdown completed")

# Global application instance
_app: Optional[FlavorSnapApp] = None

def get_app() -> FlavorSnapApp:
    """Get the global application instance."""
    global _app
    if _app is None:
        _app = FlavorSnapApp()
    return _app

def create_app(validate_config: bool = True) -> FlavorSnapApp:
    """Create and initialize the application."""
    app = get_app()
    
    if not app.initialize(validate_config=validate_config):
        raise RuntimeError("Failed to initialize application")
    
    return app

def main():
    """Main entry point for the application."""
    try:
        # Create and initialize application
        app = create_app(validate_config=True)
        
        # Validate startup requirements
        if not app.validate_startup_requirements():
            logger.error("Startup requirements validation failed")
            sys.exit(1)
        
        # Print startup information
        config = app.config
        print(f"""
🍲 FlavorSnap v{config.version} started successfully!

Environment: {config.environment}
Debug Mode: {config.debug}
API Server: http://{config.api_host}:{config.api_port}
Model: {config.model_path}
Uploads: {config.upload_folder}

Configuration loaded from:
- config/default.yaml
- config/{config.environment}.yaml
- Environment variables

Use Ctrl+C to stop the server
        """)
        
        # Keep the application running
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            app.shutdown()
            
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
