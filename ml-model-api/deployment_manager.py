"""
Model Deployment and Rollback System for FlavorSnap
Handles safe model deployment with automatic rollback capabilities
"""

import os
import shutil
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import torch
import logging

from model_registry import ModelRegistry, ModelMetadata


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    auto_rollback: bool = True
    rollback_threshold: float = 0.05  # 5% performance drop
    monitoring_window: int = 100  # Number of predictions to monitor
    health_check_interval: int = 60  # seconds
    backup_models: bool = True
    max_backup_count: int = 5


@dataclass
class DeploymentEvent:
    """Record of a deployment event"""
    timestamp: str
    event_type: str  # deploy, rollback, health_check
    model_version: str
    previous_version: Optional[str] = None
    success: bool = True
    message: str = ""
    metrics: Dict[str, Any] = None


class ModelDeploymentManager:
    """Manages model deployment with rollback capabilities"""
    
    def __init__(self, 
                 model_registry: ModelRegistry,
                 deployment_config: DeploymentConfig = None,
                 registry_path: str = "model_registry.db"):
        self.model_registry = model_registry
        self.config = deployment_config or DeploymentConfig()
        self.registry_path = registry_path
        self.deployment_dir = Path("deployments")
        self.backup_dir = Path("model_backups")
        
        # Setup directories
        self.deployment_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize deployment tracking
        self._init_deployment_tracking()
    
    def _setup_logging(self):
        """Setup logging for deployment events"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('deployment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ModelDeployment')
    
    def _init_deployment_tracking(self):
        """Initialize deployment tracking database"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployment_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    previous_version TEXT,
                    success BOOLEAN NOT NULL,
                    message TEXT,
                    metrics TEXT,
                    FOREIGN KEY (model_version) REFERENCES models(version),
                    FOREIGN KEY (previous_version) REFERENCES models(version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployment_health (
                    model_version TEXT PRIMARY KEY,
                    last_health_check TEXT,
                    health_score REAL,
                    error_count INTEGER DEFAULT 0,
                    total_requests INTEGER DEFAULT 0,
                    avg_response_time REAL,
                    last_error TEXT,
                    FOREIGN KEY (model_version) REFERENCES models(version)
                )
            """)
    
    def _record_deployment_event(self, event: DeploymentEvent):
        """Record a deployment event"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                INSERT INTO deployment_events 
                (timestamp, event_type, model_version, previous_version, 
                 success, message, metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp,
                event.event_type,
                event.model_version,
                event.previous_version,
                event.success,
                event.message,
                json.dumps(event.metrics) if event.metrics else None
            ))
    
    def _backup_current_model(self, current_version: str) -> bool:
        """Create backup of current model"""
        if not self.config.backup_models:
            return True
        
        try:
            current_model = self.model_registry.get_model(current_version)
            if not current_model:
                return False
            
            # Create backup directory
            backup_path = self.backup_dir / f"{current_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path.mkdir(exist_ok=True)
            
            # Copy model file
            shutil.copy2(current_model.model_path, backup_path / "model.pth")
            
            # Save metadata
            metadata_path = backup_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'version': current_model.version,
                    'created_at': current_model.created_at,
                    'created_by': current_model.created_by,
                    'description': current_model.description,
                    'accuracy': current_model.accuracy,
                    'loss': current_model.loss,
                    'backup_timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            self.logger.info(f"Model backup created: {backup_path}")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup model: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Remove old backups keeping only the most recent ones"""
        if not self.backup_dir.exists():
            return
        
        # Group backups by version
        version_backups = {}
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                version = backup_path.name.split('_')[0]
                if version not in version_backups:
                    version_backups[version] = []
                version_backups[version].append(backup_path)
        
        # Keep only the most recent backups for each version
        for version, backups in version_backups.items():
            if len(backups) > self.config.max_backup_count:
                # Sort by timestamp and remove oldest
                backups.sort(key=lambda x: x.name, reverse=True)
                for old_backup in backups[self.config.max_backup_count:]:
                    try:
                        shutil.rmtree(old_backup)
                        self.logger.info(f"Removed old backup: {old_backup}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove backup {old_backup}: {e}")
    
    def _validate_model(self, version: str) -> Tuple[bool, str]:
        """Validate model before deployment"""
        try:
            model_metadata = self.model_registry.get_model(version)
            if not model_metadata:
                return False, "Model not found in registry"
            
            # Check if model file exists
            if not os.path.exists(model_metadata.model_path):
                return False, "Model file not found"
            
            # Try to load model
            try:
                # Simple validation - try to load the model
                model_data = torch.load(model_metadata.model_path, map_location='cpu')
                self.logger.info(f"Model {version} validation passed")
                return True, "Model validation successful"
                
            except Exception as e:
                return False, f"Model loading failed: {str(e)}"
                
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def deploy_model(self, target_version: str, force: bool = False) -> bool:
        """Deploy a new model version with rollback capability"""
        
        # Validate target model
        is_valid, validation_msg = self._validate_model(target_version)
        if not is_valid:
            self.logger.error(f"Model validation failed: {validation_msg}")
            return False
        
        # Get current active model
        current_model = self.model_registry.get_active_model()
        current_version = current_model.version if current_model else None
        
        # Don't deploy if already active
        if current_version == target_version:
            self.logger.info(f"Model {target_version} is already active")
            return True
        
        # Backup current model
        if current_version and not self._backup_current_model(current_version):
            self.logger.warning("Failed to backup current model")
            if not force:
                return False
        
        # Perform deployment
        try:
            # Activate new model
            success = self.model_registry.activate_model(target_version)
            
            if success:
                # Record deployment event
                event = DeploymentEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="deploy",
                    model_version=target_version,
                    previous_version=current_version,
                    success=True,
                    message=f"Successfully deployed model {target_version}"
                )
                self._record_deployment_event(event)
                
                self.logger.info(f"Successfully deployed model {target_version}")
                return True
            else:
                # Rollback if deployment failed
                if current_version:
                    self.rollback_model(current_version, reason="Deployment activation failed")
                
                event = DeploymentEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="deploy",
                    model_version=target_version,
                    previous_version=current_version,
                    success=False,
                    message="Failed to activate model"
                )
                self._record_deployment_event(event)
                
                return False
                
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            
            # Attempt rollback
            if current_version:
                self.rollback_model(current_version, reason=f"Deployment error: {str(e)}")
            
            return False
    
    def rollback_model(self, target_version: str, reason: str = "") -> bool:
        """Rollback to a previous model version"""
        
        try:
            # Validate target model
            is_valid, validation_msg = self._validate_model(target_version)
            if not is_valid:
                self.logger.error(f"Rollback validation failed: {validation_msg}")
                return False
            
            # Get current model
            current_model = self.model_registry.get_active_model()
            current_version = current_model.version if current_model else None
            
            # Perform rollback
            success = self.model_registry.activate_model(target_version)
            
            if success:
                # Record rollback event
                event = DeploymentEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="rollback",
                    model_version=target_version,
                    previous_version=current_version,
                    success=True,
                    message=f"Rolled back to model {target_version}. Reason: {reason}"
                )
                self._record_deployment_event(event)
                
                self.logger.info(f"Successfully rolled back to model {target_version}")
                return True
            else:
                self.logger.error("Failed to perform rollback")
                return False
                
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def get_deployment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get deployment history"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM deployment_events 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'timestamp': row['timestamp'],
                    'event_type': row['event_type'],
                    'model_version': row['model_version'],
                    'previous_version': row['previous_version'],
                    'success': bool(row['success']),
                    'message': row['message'],
                    'metrics': json.loads(row['metrics']) if row['metrics'] else None
                })
            
            return history
    
    def get_available_rollback_versions(self) -> List[str]:
        """Get list of versions available for rollback"""
        models = self.model_registry.list_models()
        current_model = self.model_registry.get_active_model()
        current_version = current_model.version if current_model else None
        
        # Return all models except the current one
        available_versions = [m.version for m in models if m.version != current_version]
        
        # Also check backup directory
        backup_versions = []
        if self.backup_dir.exists():
            for backup_path in self.backup_dir.iterdir():
                if backup_path.is_dir():
                    version = backup_path.name.split('_')[0]
                    if version not in backup_versions and version not in available_versions:
                        backup_versions.append(version)
        
        return available_versions + backup_versions
    
    def health_check(self, model_version: str = None) -> Dict[str, Any]:
        """Perform health check on deployed model"""
        
        if not model_version:
            current_model = self.model_registry.get_active_model()
            model_version = current_model.version if current_model else None
        
        if not model_version:
            return {"healthy": False, "message": "No active model found"}
        
        try:
            # Validate model file exists and is loadable
            model_metadata = self.model_registry.get_model(model_version)
            if not model_metadata:
                return {"healthy": False, "message": "Model not found in registry"}
            
            # Try to load model
            model_data = torch.load(model_metadata.model_path, map_location='cpu')
            
            # Get current health metrics
            with sqlite3.connect(self.registry_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM deployment_health 
                    WHERE model_version = ?
                """, (model_version,))
                
                health_row = cursor.fetchone()
            
            health_score = 1.0  # Default healthy score
            error_count = 0
            total_requests = 0
            avg_response_time = 0.0
            last_error = None
            
            if health_row:
                health_score = health_row['health_score'] or 1.0
                error_count = health_row['error_count'] or 0
                total_requests = health_row['total_requests'] or 0
                avg_response_time = health_row['avg_response_time'] or 0.0
                last_error = health_row['last_error']
            
            # Determine health status
            healthy = health_score >= 0.8 and error_count < 10
            
            # Record health check
            event = DeploymentEvent(
                timestamp=datetime.now().isoformat(),
                event_type="health_check",
                model_version=model_version,
                success=healthy,
                message=f"Health check completed. Score: {health_score:.2f}",
                metrics={
                    "health_score": health_score,
                    "error_count": error_count,
                    "total_requests": total_requests,
                    "avg_response_time": avg_response_time
                }
            )
            self._record_deployment_event(event)
            
            return {
                "healthy": healthy,
                "model_version": model_version,
                "health_score": health_score,
                "error_count": error_count,
                "total_requests": total_requests,
                "avg_response_time": avg_response_time,
                "last_error": last_error,
                "message": "Model is healthy" if healthy else "Model health issues detected"
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "healthy": False,
                "model_version": model_version,
                "message": f"Health check failed: {str(e)}"
            }
    
    def update_health_metrics(self, 
                             model_version: str,
                             response_time: float,
                             success: bool,
                             error_message: str = None):
        """Update health metrics for a model"""
        
        with sqlite3.connect(self.registry_path) as conn:
            # Get current metrics
            cursor = conn.execute("""
                SELECT error_count, total_requests, avg_response_time
                FROM deployment_health
                WHERE model_version = ?
            """, (model_version,))
            
            row = cursor.fetchone()
            
            if row:
                error_count, total_requests, avg_response_time = row
                error_count = error_count or 0
                total_requests = total_requests or 0
                avg_response_time = avg_response_time or 0.0
            else:
                error_count = 0
                total_requests = 0
                avg_response_time = 0.0
            
            # Update metrics
            total_requests += 1
            if not success:
                error_count += 1
            
            # Calculate new average response time
            if total_requests == 1:
                avg_response_time = response_time
            else:
                avg_response_time = (avg_response_time * (total_requests - 1) + response_time) / total_requests
            
            # Calculate health score
            error_rate = error_count / total_requests if total_requests > 0 else 0
            health_score = max(0.0, 1.0 - error_rate - (response_time / 10.0))  # Penalize slow responses
            
            # Update database
            conn.execute("""
                INSERT OR REPLACE INTO deployment_health
                (model_version, last_health_check, health_score, error_count, 
                 total_requests, avg_response_time, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_version,
                datetime.now().isoformat(),
                health_score,
                error_count,
                total_requests,
                avg_response_time,
                error_message
            ))
        
        # Check if auto-rollback is needed
        if (self.config.auto_rollback and 
            health_score < (1.0 - self.config.rollback_threshold) and
            total_requests >= self.config.monitoring_window):
            
            self.logger.warning(f"Model {model_version} health degraded ({health_score:.2f}), considering rollback")
            
            # Get previous stable version
            stable_models = [m for m in self.model_registry.list_models() if m.is_stable]
            if stable_models:
                previous_stable = stable_models[0].version
                if previous_stable != model_version:
                    self.rollback_model(previous_stable, f"Auto-rollback due to health degradation: {health_score:.2f}")
                    return True
        
        return False
