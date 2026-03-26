"""
A/B Testing Framework for FlavorSnap Models
Handles model comparison experiments and performance tracking
"""

import uuid
import json
import sqlite3
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from model_registry import ModelRegistry, ModelMetadata


@dataclass
class ABTestConfig:
    """Configuration for an A/B test"""
    test_id: str
    model_a_version: str
    model_b_version: str
    traffic_split: float = 0.5  # 0.0 = 100% model A, 1.0 = 100% model B
    start_time: str = None
    end_time: str = None
    status: str = "active"  # active, paused, completed
    description: str = ""
    min_sample_size: int = 100
    confidence_threshold: float = 0.95
    metrics_to_track: List[str] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now().isoformat()
        if self.metrics_to_track is None:
            self.metrics_to_track = ["accuracy", "confidence", "processing_time"]


@dataclass
class ModelMetrics:
    """Performance metrics for a model"""
    model_version: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    confidence_scores: List[float] = None
    processing_times: List[float] = None
    predictions_by_class: Dict[str, int] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.processing_times is None:
            self.processing_times = []
        if self.predictions_by_class is None:
            self.predictions_by_class = {}


class ABTestManager:
    """Manages A/B testing between model versions"""
    
    def __init__(self, model_registry: ModelRegistry, registry_path: str = "model_registry.db"):
        self.model_registry = model_registry
        self.registry_path = registry_path
        self.active_tests: Dict[str, ABTestConfig] = {}
        self._load_active_tests()
    
    def _load_active_tests(self):
        """Load active tests from database"""
        with sqlite3.connect(self.registry_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM ab_tests WHERE status = 'active'"
            )
            for row in cursor.fetchall():
                config = ABTestConfig(
                    test_id=row['test_id'],
                    model_a_version=row['model_a_version'],
                    model_b_version=row['model_b_version'],
                    traffic_split=row['traffic_split'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    status=row['status'],
                    min_sample_size=100,  # Default values
                    confidence_threshold=0.95,
                    metrics_to_track=["accuracy", "confidence", "processing_time"]
                )
                self.active_tests[config.test_id] = config
    
    def create_test(self, 
                   model_a_version: str,
                   model_b_version: str,
                   traffic_split: float = 0.5,
                   description: str = "",
                   min_sample_size: int = 100,
                   confidence_threshold: float = 0.95) -> str:
        """Create a new A/B test"""
        
        # Validate models exist
        model_a = self.model_registry.get_model(model_a_version)
        model_b = self.model_registry.get_model(model_b_version)
        
        if not model_a or not model_b:
            raise ValueError("Both models must be registered before creating A/B test")
        
        # Validate traffic split
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError("Traffic split must be between 0.0 and 1.0")
        
        test_id = str(uuid.uuid4())
        
        config = ABTestConfig(
            test_id=test_id,
            model_a_version=model_a_version,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            description=description,
            min_sample_size=min_sample_size,
            confidence_threshold=confidence_threshold
        )
        
        # Store in database
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                INSERT INTO ab_tests 
                (test_id, model_a_version, model_b_version, traffic_split, 
                 start_time, status, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id, model_a_version, model_b_version, traffic_split,
                config.start_time, config.status, description
            ))
        
        self.active_tests[test_id] = config
        return test_id
    
    def get_model_for_request(self, test_id: str = None) -> Tuple[str, str]:
        """
        Get which model to use for a request
        Returns: (model_version, test_id)
        """
        if test_id and test_id in self.active_tests:
            config = self.active_tests[test_id]
            # Use specific test
            if random.random() < config.traffic_split:
                return config.model_b_version, test_id
            else:
                return config.model_a_version, test_id
        else:
            # Use any active test
            for test_id, config in self.active_tests.items():
                if random.random() < config.traffic_split:
                    return config.model_b_version, test_id
                else:
                    return config.model_a_version, test_id
            
            # No active tests, use active model
            active_model = self.model_registry.get_active_model()
            if active_model:
                return active_model.version, None
            else:
                raise ValueError("No active models or tests available")
    
    def record_prediction(self,
                         test_id: str,
                         model_version: str,
                         image_path: str,
                         prediction: str,
                         confidence: float,
                         processing_time: float,
                         ground_truth: str = None,
                         user_id: str = None):
        """Record a prediction result"""
        
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                INSERT INTO predictions 
                (test_id, model_version, image_path, prediction, confidence,
                 processing_time, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_id, model_version, image_path, prediction, confidence,
                processing_time, datetime.now().isoformat(), user_id
            ))
    
    def get_test_metrics(self, test_id: str) -> Tuple[ModelMetrics, ModelMetrics]:
        """Get performance metrics for both models in a test"""
        
        config = self.active_tests.get(test_id)
        if not config:
            raise ValueError(f"Test {test_id} not found or not active")
        
        metrics_a = self._get_model_metrics(config.model_a_version, test_id)
        metrics_b = self._get_model_metrics(config.model_b_version, test_id)
        
        return metrics_a, metrics_b
    
    def _get_model_metrics(self, model_version: str, test_id: str) -> ModelMetrics:
        """Calculate metrics for a specific model in a test"""
        
        with sqlite3.connect(self.registry_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT prediction, confidence, processing_time
                FROM predictions
                WHERE test_id = ? AND model_version = ?
            """, (test_id, model_version))
            
            predictions = cursor.fetchall()
            
            metrics = ModelMetrics(model_version=model_version)
            metrics.total_predictions = len(predictions)
            
            if predictions:
                confidences = [p['confidence'] for p in predictions]
                processing_times = [p['processing_time'] for p in predictions]
                
                metrics.confidence_scores = confidences
                metrics.processing_times = processing_times
                metrics.avg_confidence = np.mean(confidences)
                metrics.avg_processing_time = np.mean(processing_times)
                
                # Count predictions by class
                for p in predictions:
                    class_name = p['prediction']
                    metrics.predictions_by_class[class_name] = \
                        metrics.predictions_by_class.get(class_name, 0) + 1
        
        return metrics
    
    def calculate_statistical_significance(self, test_id: str) -> Dict[str, Any]:
        """Calculate statistical significance for A/B test results"""
        
        metrics_a, metrics_b = self.get_test_metrics(test_id)
        
        if metrics_a.total_predictions < 100 or metrics_b.total_predictions < 100:
            return {
                "significant": False,
                "reason": "Insufficient sample size",
                "min_required": 100
            }
        
        # Simple t-test for accuracy difference
        # For simplicity, we'll use confidence scores as proxy for accuracy
        if len(metrics_a.confidence_scores) == 0 or len(metrics_b.confidence_scores) == 0:
            return {
                "significant": False,
                "reason": "No confidence data available"
            }
        
        from scipy import stats
        
        # Perform two-sample t-test
        t_stat, p_value = stats.ttest_ind(
            metrics_a.confidence_scores,
            metrics_b.confidence_scores
        )
        
        config = self.active_tests[test_id]
        is_significant = p_value < (1 - config.confidence_threshold)
        
        return {
            "significant": is_significant,
            "p_value": p_value,
            "confidence_threshold": config.confidence_threshold,
            "t_statistic": t_stat,
            "model_a_mean": np.mean(metrics_a.confidence_scores),
            "model_b_mean": np.mean(metrics_b.confidence_scores),
            "model_a_std": np.std(metrics_a.confidence_scores),
            "model_b_std": np.std(metrics_b.confidence_scores)
        }
    
    def end_test(self, test_id: str, winner: str = None) -> bool:
        """End an A/B test and optionally declare a winner"""
        
        if test_id not in self.active_tests:
            return False
        
        config = self.active_tests[test_id]
        
        # If no winner specified, determine based on metrics
        if winner is None:
            metrics_a, metrics_b = self.get_test_metrics(test_id)
            
            # Simple comparison based on average confidence
            if metrics_a.avg_confidence > metrics_b.avg_confidence:
                winner = config.model_a_version
            else:
                winner = config.model_b_version
        
        # Update database
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                UPDATE ab_tests 
                SET status = 'completed', end_time = ?, winner = ?
                WHERE test_id = ?
            """, (datetime.now().isoformat(), winner, test_id))
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        return True
    
    def list_tests(self, status: str = None) -> List[Dict[str, Any]]:
        """List all tests with optional status filter"""
        
        with sqlite3.connect(self.registry_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM ab_tests"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY start_time DESC"
            
            cursor = conn.execute(query, params)
            tests = []
            
            for row in cursor.fetchall():
                tests.append({
                    "test_id": row['test_id'],
                    "model_a_version": row['model_a_version'],
                    "model_b_version": row['model_b_version'],
                    "traffic_split": row['traffic_split'],
                    "start_time": row['start_time'],
                    "end_time": row['end_time'],
                    "status": row['status'],
                    "description": row.get('description', ''),
                    "winner": row.get('winner')
                })
            
            return tests
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a test"""
        
        config = self.active_tests.get(test_id)
        if not config:
            # Check completed tests
            with sqlite3.connect(self.registry_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM ab_tests WHERE test_id = ?",
                    (test_id,)
                )
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Test {test_id} not found")
                
                config = ABTestConfig(
                    test_id=row['test_id'],
                    model_a_version=row['model_a_version'],
                    model_b_version=row['model_b_version'],
                    traffic_split=row['traffic_split'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    status=row['status']
                )
        
        metrics_a, metrics_b = self.get_test_metrics(test_id)
        significance = self.calculate_statistical_significance(test_id)
        
        return {
            "test_config": asdict(config),
            "model_a_metrics": asdict(metrics_a),
            "model_b_metrics": asdict(metrics_b),
            "statistical_significance": significance
        }
