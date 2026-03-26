"""
Image Enhancement Module for FlavorSnap

This module provides real-time image preprocessing capabilities including:
- Brightness adjustment
- Contrast adjustment  
- Rotation
- Cropping with aspect ratio presets
- Reset functionality
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import math
from typing import Tuple, Optional, Dict, Any


class ImageEnhancer:
    """Handles real-time image preprocessing and enhancement."""
    
    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.enhancement_params = {
            'brightness': 1.0,
            'contrast': 1.0,
            'rotation': 0,
            'crop_box': None,
            'aspect_ratio': None
        }
    
    def load_image(self, image: Image.Image) -> None:
        """Load the original image and reset all parameters."""
        self.original_image = image.convert('RGB')
        self.current_image = self.original_image.copy()
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Reset all enhancement parameters to default values."""
        self.enhancement_params = {
            'brightness': 1.0,
            'contrast': 1.0,
            'rotation': 0,
            'crop_box': None,
            'aspect_ratio': None
        }
        if self.original_image:
            self.current_image = self.original_image.copy()
    
    def apply_brightness(self, brightness: float) -> Image.Image:
        """Apply brightness adjustment (0.0 to 2.0, 1.0 = original)."""
        if not self.current_image:
            return None
        
        brightness = max(0.0, min(2.0, brightness))
        self.enhancement_params['brightness'] = brightness
        
        enhancer = ImageEnhance.Brightness(self.current_image)
        self.current_image = enhancer.enhance(brightness)
        return self.current_image
    
    def apply_contrast(self, contrast: float) -> Image.Image:
        """Apply contrast adjustment (0.0 to 2.0, 1.0 = original)."""
        if not self.current_image:
            return None
        
        contrast = max(0.0, min(2.0, contrast))
        self.enhancement_params['contrast'] = contrast
        
        enhancer = ImageEnhance.Contrast(self.current_image)
        self.current_image = enhancer.enhance(contrast)
        return self.current_image
    
    def apply_rotation(self, angle: float) -> Image.Image:
        """Apply rotation (angle in degrees)."""
        if not self.current_image:
            return None
        
        self.enhancement_params['rotation'] = angle
        
        # Rotate with expansion to avoid cropping
        rotated = self.current_image.rotate(angle, expand=True, fillcolor='white')
        self.current_image = rotated
        return self.current_image
    
    def apply_crop(self, crop_box: Tuple[int, int, int, int]) -> Image.Image:
        """Apply crop to the image (left, top, right, bottom)."""
        if not self.current_image:
            return None
        
        # Validate crop box
        width, height = self.current_image.size
        left, top, right, bottom = crop_box
        
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        
        crop_box = (left, top, right, bottom)
        self.enhancement_params['crop_box'] = crop_box
        
        cropped = self.current_image.crop(crop_box)
        self.current_image = cropped
        return self.current_image
    
    def apply_aspect_ratio_crop(self, aspect_ratio: str) -> Optional[Image.Image]:
        """Apply crop with specific aspect ratio."""
        if not self.current_image:
            return None
        
        width, height = self.current_image.size
        current_ratio = width / height
        
        # Define aspect ratios
        ratios = {
            '1:1': 1.0,
            '4:3': 4/3,
            '16:9': 16/9,
            '3:2': 3/2,
            '9:16': 9/16
        }
        
        target_ratio = ratios.get(aspect_ratio)
        if not target_ratio:
            return None
        
        self.enhancement_params['aspect_ratio'] = aspect_ratio
        
        # Calculate crop dimensions
        if current_ratio > target_ratio:
            # Image is wider, crop width
            new_width = int(height * target_ratio)
            left = (width - new_width) // 2
            crop_box = (left, 0, left + new_width, height)
        else:
            # Image is taller, crop height
            new_height = int(width / target_ratio)
            top = (height - new_height) // 2
            crop_box = (0, top, width, top + new_height)
        
        return self.apply_crop(crop_box)
    
    def get_processed_image(self) -> Image.Image:
        """Get the current processed image."""
        return self.current_image
    
    def get_enhancement_params(self) -> Dict[str, Any]:
        """Get current enhancement parameters."""
        return self.enhancement_params.copy()
    
    def apply_all_enhancements(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Apply all enhancements from parameters to an image."""
        self.load_image(image)
        
        # Apply brightness
        if 'brightness' in params:
            self.apply_brightness(params['brightness'])
        
        # Apply contrast
        if 'contrast' in params:
            self.apply_contrast(params['contrast'])
        
        # Apply rotation
        if 'rotation' in params:
            self.apply_rotation(params['rotation'])
        
        # Apply crop
        if 'crop_box' in params and params['crop_box']:
            self.apply_crop(params['crop_box'])
        elif 'aspect_ratio' in params and params['aspect_ratio']:
            self.apply_aspect_ratio_crop(params['aspect_ratio'])
        
        return self.current_image
    
    def get_image_info(self) -> Dict[str, Any]:
        """Get information about the current image."""
        if not self.current_image:
            return {}
        
        width, height = self.current_image.size
        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height,
            'size_bytes': len(self.current_image.tobytes()),
            'mode': self.current_image.mode
        }
    
    def auto_enhance(self) -> Image.Image:
        """Apply automatic enhancement to improve image quality."""
        if not self.current_image:
            return None
        
        # Convert to numpy array for OpenCV operations
        img_array = np.array(self.current_image)
        
        # Convert to LAB color space for better brightness/contrast adjustment
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_array = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(enhanced_array)
        
        # Apply mild sharpening
        from PIL import ImageFilter
        enhanced_image = enhanced_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        self.current_image = enhanced_image
        return self.current_image
