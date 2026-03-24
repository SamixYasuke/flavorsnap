"""
FlavorSnap Core Components

This module provides the core functionality for the FlavorSnap food classification system.
It includes the main classifier, model management, and image processing utilities.

Usage:
    from flavorsnap import FoodClassifier, ModelManager, ImageProcessor
    
    classifier = FoodClassifier()
    result = classifier.classify_image("path/to/image.jpg")
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages loading and caching of ML models.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = model_path or "model.pth"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = []
        
    def load_model(self, force_reload: bool = False) -> torch.nn.Module:
        """
        Load the trained model.
        
        Args:
            force_reload: Force reload even if already loaded
            
        Returns:
            torch.nn.Module: Loaded model
        """
        if self.model is not None and not force_reload:
            return self.model
        
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model state
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Load classes
            self._load_classes()
            
            logger.info(f"Model loaded successfully on {self.device}")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_classes(self):
        """Load the list of food classes."""
        try:
            classes_file = Path("food_classes.txt")
            if classes_file.exists():
                with open(classes_file, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines() if line.strip()]
            else:
                # Default classes
                self.classes = ["Akara", "Bread", "Egusi", "Moi Moi", "Rice and Stew", "Yam"]
            
            logger.info(f"Loaded {len(self.classes)} food classes")
            
        except Exception as e:
            logger.warning(f"Failed to load classes: {e}")
            self.classes = ["Unknown"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "device": str(self.device),
            "classes": self.classes,
            "num_classes": len(self.classes),
            "model_path": self.model_path,
            "input_size": [224, 224],  # ResNet18 standard input
            "architecture": "ResNet18"
        }

class ImageProcessor:
    """
    Handles image preprocessing for model inference.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image processor.
        
        Args:
            input_size: Target input size for the model
        """
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for model inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Load and validate image
            image = Image.open(image_path)
            
            # Convert RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            raise ValueError(f"Invalid image file: {e}")
    
    def validate_image(self, image_path: str) -> bool:
        """
        Validate if the image file is suitable for processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if image is valid
        """
        try:
            with Image.open(image_path) as img:
                # Check if it's a valid image
                img.verify()
            
            # Check file extension
            allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            if not any(image_path.lower().endswith(ext) for ext in allowed_extensions):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Get information about an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Image information
        """
        try:
            with Image.open(image_path) as img:
                return {
                    "format": img.format,
                    "mode": img.mode,
                    "size": img.size,
                    "file_size": os.path.getsize(image_path)
                }
        except Exception as e:
            return {"error": str(e)}

class FoodClassifier:
    """
    Main food classification interface.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.6):
        """
        Initialize the food classifier.
        
        Args:
            model_path: Path to the trained model file
            confidence_threshold: Minimum confidence for classification
        """
        self.model_manager = ModelManager(model_path)
        self.image_processor = ImageProcessor()
        self.confidence_threshold = confidence_threshold
        
        # Load model on initialization
        self.model_manager.load_model()
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a food image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Classification results
        """
        try:
            # Validate image
            if not self.image_processor.validate_image(image_path):
                raise ValueError("Invalid image file")
            
            # Preprocess image
            input_tensor = self.image_processor.preprocess_image(image_path)
            input_tensor = input_tensor.to(self.model_manager.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model_manager.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, min(5, len(self.model_manager.classes)))
            
            # Format results
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                class_name = self.model_manager.classes[idx.item()]
                confidence = prob.item() * 100
                
                predictions.append({
                    "label": class_name,
                    "confidence": round(confidence, 2),
                    "rank": i + 1
                })
            
            # Get best prediction
            best_prediction = predictions[0] if predictions else None
            
            # Check confidence threshold
            if best_prediction and best_prediction["confidence"] < self.confidence_threshold:
                best_prediction = {
                    "label": "Unknown",
                    "confidence": 0.0,
                    "rank": 1
                }
                predictions.insert(0, best_prediction)
            
            # Get image info
            image_info = self.image_processor.get_image_info(image_path)
            
            result = {
                "success": True,
                "prediction": best_prediction,
                "all_predictions": predictions,
                "image_info": image_info,
                "model_info": self.model_manager.get_model_info(),
                "threshold_used": self.confidence_threshold
            }
            
            logger.info(f"Classification completed for {image_path}: {best_prediction['label']}")
            return result
            
        except Exception as e:
            logger.error(f"Classification failed for {image_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "prediction": None,
                "all_predictions": []
            }
    
    def classify_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Classify multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            list: Classification results for each image
        """
        results = []
        
        for image_path in image_paths:
            result = self.classify_image(image_path)
            results.append(result)
        
        return results
    
    def get_supported_classes(self) -> List[str]:
        """
        Get the list of supported food classes.
        
        Returns:
            list: Supported class names
        """
        return self.model_manager.classes.copy()
    
    def update_confidence_threshold(self, threshold: float):
        """
        Update the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to: {threshold}")
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """
        Get information about the classifier.
        
        Returns:
            dict: Classifier information
        """
        return {
            "model_info": self.model_manager.get_model_info(),
            "confidence_threshold": self.confidence_threshold,
            "supported_classes": self.get_supported_classes(),
            "input_size": self.image_processor.input_size
        }
