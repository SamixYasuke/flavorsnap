from src.utils.memory_manager import MemoryManager
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import time
import io
import os
import gc

class ProgressClassifier:
    """
    Enhanced Image Classifier with progress tracking and memory management.
    """
    def __init__(self, model_path='models/best_model.pth'):
        self.model_path = model_path
        self.class_names = ['Akara', 'Bread', 'Egusi', 'Moi Moi', 'Rice and Stew', 'Yam']
        self.model = self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _load_model(self):
        # Initial Model Setup
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, len(self.class_names))
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def cleanup(self):
        """Standard cleanup for intermediate components."""
        MemoryManager.cleanup()

    def classify_with_progress(self, image_data, progress_callback=None):
        """
        Runs classification with simulated stages and strict memory management.
        """
        def update(p, msg):
            if progress_callback:
                progress_callback(p, msg)
            time.sleep(0.5)

        try:
            # 1. Image Loading (Skeleton Phase)
            update(5, "🖼️ Decoding image data...")
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            update(15, "✨ Normalizing pixels...")

            # 2. Pre-processing
            update(30, "📐 Resizing to 224x224...")
            img_tensor = self.transform(image).unsqueeze(0)
            update(45, "🧠 Preparing neural network tensors...")

            # 3. Model Inference (Main Process)
            start_time = time.time()
            update(60, "🔍 Deep-learning analysis in progress...")
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, pred = torch.max(outputs, 1)
                predicted_class = self.class_names[pred.item()]
            
            elapsed = time.time() - start_time
            update(85, "📊 Post-processing confidence scores...")

            # Explicitly delete result-heavy tensors
            del img_tensor
            del outputs

            # 4. Finalizing
            update(100, f"✅ Done! Found: {predicted_class}")
            
            return predicted_class, image

        except Exception as e:
            raise e
        finally:
            # Final broad cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
