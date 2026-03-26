"""
Batch Image Processing Module for FlavorSnap
Handles multiple image uploads, queue management, and progress tracking
"""

import os
import uuid
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image
import torch
import queue
from pathlib import Path

from model_registry import ModelRegistry
from ab_testing import ABTestManager


@dataclass
class BatchJob:
    """Represents a batch processing job"""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    results: List[Dict] = None
    errors: List[Dict] = None
    progress_percentage: float = 0.0
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []


@dataclass
class BatchResult:
    """Result for a single image in batch"""
    filename: str
    label: str
    confidence: float
    all_predictions: List[Dict]
    processing_time: float
    model_version: str
    test_id: Optional[str] = None
    error: Optional[str] = None


class BatchProcessor:
    """Handles batch processing of multiple images"""
    
    def __init__(self, model_registry: ModelRegistry, ab_test_manager: ABTestManager):
        self.model_registry = model_registry
        self.ab_test_manager = ab_test_manager
        self.jobs: Dict[str, BatchJob] = {}
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.current_job_id = None
        self._lock = threading.Lock()
        
        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.processing_thread.start()
    
    def create_batch_job(self, files: List) -> str:
        """Create a new batch job"""
        job_id = str(uuid.uuid4())
        
        job = BatchJob(
            job_id=job_id,
            status='pending',
            created_at=datetime.now(),
            total_files=len(files)
        )
        
        with self._lock:
            self.jobs[job_id] = job
        
        # Add files to processing queue
        for file in files:
            self.processing_queue.put((job_id, file))
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a batch job"""
        with self._lock:
            return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[BatchJob]:
        """Get all batch jobs"""
        with self._lock:
            return list(self.jobs.values())
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job"""
        with self._lock:
            job = self.jobs.get(job_id)
            if job and job.status in ['pending', 'processing']:
                job.status = 'cancelled'
                return True
        return False
    
    def _process_queue(self):
        """Background thread to process queued images"""
        while True:
            try:
                if not self.processing_queue.empty():
                    job_id, file = self.processing_queue.get()
                    self._process_single_image(job_id, file)
                else:
                    time.sleep(0.1)  # Prevent busy waiting
            except Exception as e:
                print(f"Error in processing queue: {e}")
    
    def _process_single_image(self, job_id: str, file):
        """Process a single image in the batch"""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job or job.status == 'cancelled':
                return
            
            if job.status == 'pending':
                job.status = 'processing'
                job.started_at = datetime.now()
                self.current_job_id = job_id
        
        try:
            # Import here to avoid circular imports
            from app import get_model_for_prediction, class_names, transform
            
            # Get model for prediction
            model, used_version, test_id = get_model_for_prediction()
            
            # Process image
            start_time = time.time()
            image = Image.open(file.stream).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to label
            predicted_label = class_names[predicted_class.item()]
            confidence_score = confidence.item()
            processing_time = time.time() - start_time
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            all_predictions = [
                {
                    "label": class_names[idx.item()],
                    "confidence": prob.item()
                }
                for prob, idx in zip(top_probs[0], top_indices[0])
            ]
            
            # Create result
            result = BatchResult(
                filename=file.filename,
                label=predicted_label,
                confidence=confidence_score,
                all_predictions=all_predictions,
                processing_time=processing_time,
                model_version=used_version,
                test_id=test_id
            )
            
            # Update job with result
            with self._lock:
                job = self.jobs.get(job_id)
                if job:
                    job.results.append(asdict(result))
                    job.processed_files += 1
                    job.progress_percentage = (job.processed_files / job.total_files) * 100
                    
                    # Check if job is complete
                    if job.processed_files + job.failed_files >= job.total_files:
                        job.status = 'completed'
                        job.completed_at = datetime.now()
                        self.current_job_id = None
            
        except Exception as e:
            # Handle error for individual file
            with self._lock:
                job = self.jobs.get(job_id)
                if job:
                    job.errors.append({
                        'filename': file.filename,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
                    job.failed_files += 1
                    job.progress_percentage = (job.processed_files / job.total_files) * 100
                    
                    # Check if job is complete (with failures)
                    if job.processed_files + job.failed_files >= job.total_files:
                        job.status = 'completed' if job.processed_files > 0 else 'failed'
                        job.completed_at = datetime.now()
                        self.current_job_id = None
    
    def get_job_summary(self, job_id: str) -> Optional[Dict]:
        """Get summary statistics for a completed job"""
        job = self.get_job_status(job_id)
        if not job or job.status not in ['completed', 'failed']:
            return None
        
        # Calculate statistics
        labels = [r['label'] for r in job.results]
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        avg_confidence = sum(r['confidence'] for r in job.results) / len(job.results) if job.results else 0
        total_processing_time = sum(r['processing_time'] for r in job.results)
        
        return {
            'job_id': job_id,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'total_files': job.total_files,
            'processed_files': job.processed_files,
            'failed_files': job.failed_files,
            'success_rate': (job.processed_files / job.total_files) * 100 if job.total_files > 0 else 0,
            'label_distribution': label_counts,
            'average_confidence': avg_confidence,
            'total_processing_time': total_processing_time,
            'average_processing_time': total_processing_time / job.processed_files if job.processed_files > 0 else 0
        }
    
    def export_results(self, job_id: str, format: str = 'json') -> Optional[str]:
        """Export job results in specified format"""
        job = self.get_job_status(job_id)
        if not job or job.status not in ['completed', 'failed']:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = f"batch_results_{job_id}_{timestamp}.json"
            filepath = os.path.join('uploads', filename)
            
            export_data = {
                'job_info': asdict(job),
                'summary': self.get_job_summary(job_id),
                'results': job.results,
                'errors': job.errors
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return filepath
        
        elif format == 'csv':
            import csv
            filename = f"batch_results_{job_id}_{timestamp}.csv"
            filepath = os.path.join('uploads', filename)
            
            with open(filepath, 'w', newline='') as f:
                if job.results:
                    writer = csv.DictWriter(f, fieldnames=job.results[0].keys())
                    writer.writeheader()
                    writer.writerows(job.results)
            
            return filepath
        
        return None
