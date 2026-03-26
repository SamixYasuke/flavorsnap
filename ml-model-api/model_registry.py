"""
Model Registry for FlavorSnap
Handles model versioning, metadata storage, and lifecycle management
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import torch.nn as nn


@dataclass
class ModelMetadata:
    """Metadata for a registered model version"""
    version: str
    model_path: str
    created_at: str
    created_by: str
    description: str
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    epochs_trained: Optional[int] = None
    dataset_version: Optional[str] = None
    model_hash: Optional[str] = None
    is_active: bool = False
    is_stable: bool = False
    tags: List[str] = None
    hyperparameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.hyperparameters is None:
            self.hyperparameters = {}


class ModelRegistry:
    """Central registry for managing model versions"""
    
    def __init__(self, registry_path: str = "model_registry.db"):
        self.registry_path = registry_path
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for model metadata"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    version TEXT PRIMARY KEY,
                    model_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    accuracy REAL,
                    loss REAL,
                    epochs_trained INTEGER,
                    dataset_version TEXT,
                    model_hash TEXT,
                    is_active BOOLEAN DEFAULT FALSE,
                    is_stable BOOLEAN DEFAULT FALSE,
                    tags TEXT,
                    hyperparameters TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_tests (
                    test_id TEXT PRIMARY KEY,
                    model_a_version TEXT NOT NULL,
                    model_b_version TEXT NOT NULL,
                    traffic_split REAL DEFAULT 0.5,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT DEFAULT 'active',
                    metrics_a TEXT,
                    metrics_b TEXT,
                    winner TEXT,
                    FOREIGN KEY (model_a_version) REFERENCES models(version),
                    FOREIGN KEY (model_b_version) REFERENCES models(version)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT,
                    model_version TEXT NOT NULL,
                    image_path TEXT,
                    prediction TEXT,
                    confidence REAL,
                    processing_time REAL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    FOREIGN KEY (test_id) REFERENCES ab_tests(test_id),
                    FOREIGN KEY (model_version) REFERENCES models(version)
                )
            """)
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model file"""
        hash_sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def register_model(self, 
                      version: str,
                      model_path: str,
                      created_by: str,
                      description: str,
                      accuracy: Optional[float] = None,
                      loss: Optional[float] = None,
                      epochs_trained: Optional[int] = None,
                      dataset_version: Optional[str] = None,
                      tags: List[str] = None,
                      hyperparameters: Dict[str, Any] = None) -> bool:
        """Register a new model version"""
        try:
            # Validate model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Calculate model hash
            model_hash = self._calculate_model_hash(model_path)
            
            # Create metadata
            metadata = ModelMetadata(
                version=version,
                model_path=model_path,
                created_at=datetime.now().isoformat(),
                created_by=created_by,
                description=description,
                accuracy=accuracy,
                loss=loss,
                epochs_trained=epochs_trained,
                dataset_version=dataset_version,
                model_hash=model_hash,
                tags=tags or [],
                hyperparameters=hyperparameters or {}
            )
            
            # Store in database
            with sqlite3.connect(self.registry_path) as conn:
                conn.execute("""
                    INSERT INTO models 
                    (version, model_path, created_at, created_by, description,
                     accuracy, loss, epochs_trained, dataset_version, model_hash,
                     is_active, is_stable, tags, hyperparameters)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.version,
                    metadata.model_path,
                    metadata.created_at,
                    metadata.created_by,
                    metadata.description,
                    metadata.accuracy,
                    metadata.loss,
                    metadata.epochs_trained,
                    metadata.dataset_version,
                    metadata.model_hash,
                    metadata.is_active,
                    metadata.is_stable,
                    json.dumps(metadata.tags),
                    json.dumps(metadata.hyperparameters)
                ))
            
            return True
            
        except Exception as e:
            print(f"Error registering model: {e}")
            return False
    
    def get_model(self, version: str) -> Optional[ModelMetadata]:
        """Get model metadata by version"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM models WHERE version = ?", (version,)
            )
            row = cursor.fetchone()
            
            if row:
                return ModelMetadata(
                    version=row['version'],
                    model_path=row['model_path'],
                    created_at=row['created_at'],
                    created_by=row['created_by'],
                    description=row['description'],
                    accuracy=row['accuracy'],
                    loss=row['loss'],
                    epochs_trained=row['epochs_trained'],
                    dataset_version=row['dataset_version'],
                    model_hash=row['model_hash'],
                    is_active=bool(row['is_active']),
                    is_stable=bool(row['is_stable']),
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    hyperparameters=json.loads(row['hyperparameters']) if row['hyperparameters'] else {}
                )
        return None
    
    def list_models(self, active_only: bool = False) -> List[ModelMetadata]:
        """List all registered models"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM models"
            if active_only:
                query += " WHERE is_active = TRUE"
            query += " ORDER BY created_at DESC"
            
            cursor = conn.execute(query)
            models = []
            for row in cursor.fetchall():
                models.append(ModelMetadata(
                    version=row['version'],
                    model_path=row['model_path'],
                    created_at=row['created_at'],
                    created_by=row['created_by'],
                    description=row['description'],
                    accuracy=row['accuracy'],
                    loss=row['loss'],
                    epochs_trained=row['epochs_trained'],
                    dataset_version=row['dataset_version'],
                    model_hash=row['model_hash'],
                    is_active=bool(row['is_active']),
                    is_stable=bool(row['is_stable']),
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    hyperparameters=json.loads(row['hyperparameters']) if row['hyperparameters'] else {}
                ))
            return models
    
    def activate_model(self, version: str) -> bool:
        """Activate a model version (deactivate all others)"""
        try:
            with sqlite3.connect(self.registry_path) as conn:
                # Deactivate all models
                conn.execute("UPDATE models SET is_active = FALSE")
                
                # Activate specified model
                conn.execute(
                    "UPDATE models SET is_active = TRUE WHERE version = ?",
                    (version,)
                )
                
                return conn.total_changes > 0
        except Exception as e:
            print(f"Error activating model: {e}")
            return False
    
    def get_active_model(self) -> Optional[ModelMetadata]:
        """Get the currently active model"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM models WHERE is_active = TRUE LIMIT 1"
            )
            row = cursor.fetchone()
            
            if row:
                return ModelMetadata(
                    version=row['version'],
                    model_path=row['model_path'],
                    created_at=row['created_at'],
                    created_by=row['created_by'],
                    description=row['description'],
                    accuracy=row['accuracy'],
                    loss=row['loss'],
                    epochs_trained=row['epochs_trained'],
                    dataset_version=row['dataset_version'],
                    model_hash=row['model_hash'],
                    is_active=bool(row['is_active']),
                    is_stable=bool(row['is_stable']),
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    hyperparameters=json.loads(row['hyperparameters']) if row['hyperparameters'] else {}
                )
        return None
    
    def delete_model(self, version: str) -> bool:
        """Delete a model version (only if not active)"""
        try:
            with sqlite3.connect(self.registry_path) as conn:
                # Check if model is active
                cursor = conn.execute(
                    "SELECT is_active FROM models WHERE version = ?",
                    (version,)
                )
                row = cursor.fetchone()
                
                if row and row[0]:
                    raise ValueError("Cannot delete active model")
                
                # Delete model
                conn.execute("DELETE FROM models WHERE version = ?", (version,))
                return conn.total_changes > 0
                
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
    
    def mark_stable(self, version: str, stable: bool = True) -> bool:
        """Mark a model version as stable/unstable"""
        try:
            with sqlite3.connect(self.registry_path) as conn:
                conn.execute(
                    "UPDATE models SET is_stable = ? WHERE version = ?",
                    (stable, version)
                )
                return conn.total_changes > 0
        except Exception as e:
            print(f"Error updating model stability: {e}")
            return False
