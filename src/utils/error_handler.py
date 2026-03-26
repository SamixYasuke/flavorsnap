"""
User-Friendly Error Handler for FlavorSnap

This module provides comprehensive error handling that converts technical exceptions
into user-friendly messages with recovery suggestions and appropriate actions.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from functools import wraps
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class UserFriendlyError(Exception):
    """Custom exception for user-friendly error messages."""
    
    def __init__(self, user_message: str, recovery_suggestion: str = None, 
                 technical_details: str = None, error_code: str = None):
        self.user_message = user_message
        self.recovery_suggestion = recovery_suggestion
        self.technical_details = technical_details
        self.error_code = error_code
        super().__init__(user_message)


class ErrorHandler:
    """Main error handling class for converting technical errors to user-friendly messages."""
    
    def __init__(self):
        self.error_mappings = {
            # Image-related errors
            UnidentifiedImageError: {
                'user_message': "🖼️ Unable to read this image file",
                'recovery_suggestion': "Please upload a valid image file (JPG, PNG, GIF, or BMP). "
                                    "If the file is corrupted, try opening it in an image editor first.",
                'error_code': 'IMG_001'
            },
            # PIL Image errors
            Image.DecompressionBombError: {
                'user_message': "🚫 This image is too large to process safely",
                'recovery_suggestion': "Please resize the image to under 10,000x10,000 pixels or "
                                    "compress it before uploading.",
                'error_code': 'IMG_002'
            },
            # File system errors
            FileNotFoundError: {
                'user_message': "📁 File not found",
                'recovery_suggestion': "Please check if the file path is correct and the file exists. "
                                    "Try re-uploading the image.",
                'error_code': 'FILE_001'
            },
            PermissionError: {
                'user_message': "🔒 Permission denied",
                'recovery_suggestion': "Please check file permissions and try again. "
                                    "Make sure the application has access to read and write files.",
                'error_code': 'FILE_002'
            },
            # Memory errors
            MemoryError: {
                'user_message': "💾 Not enough memory to process this image",
                'recovery_suggestion': "Try uploading a smaller image or close other applications "
                                    "to free up memory. Recommended size: under 5MB.",
                'error_code': 'MEM_001'
            },
            # PyTorch/CUDA errors
            torch.cuda.OutOfMemoryError: {
                'user_message': "🎮 GPU memory is full",
                'recovery_suggestion': "The GPU is busy with other tasks. Please try again in a moment "
                                    "or restart the application to clear GPU memory.",
                'error_code': 'GPU_001'
            },
            # Model-related errors
            RuntimeError: {
                'user_message': "⚡ Model processing error",
                'recovery_suggestion': "There was an issue with the AI model. Please try again. "
                                    "If the problem persists, restart the application.",
                'error_code': 'MODEL_001'
            },
            # Network/IO errors
            OSError: {
                'user_message': "💿 System error occurred",
                'recovery_suggestion': "A system error occurred. Please check your disk space and "
                                    "try again. Make sure the uploads folder is accessible.",
                'error_code': 'SYS_001'
            },
            # Value errors
            ValueError: {
                'user_message': "📊 Invalid data format",
                'recovery_suggestion': "The image data appears to be invalid. Please try uploading "
                                    "a different image or check if the file is corrupted.",
                'error_code': 'DATA_001'
            },
            # Type errors
            TypeError: {
                'user_message': "🔧 Invalid operation",
                'recovery_suggestion': "An unexpected operation was attempted. Please try again "
                                    "or restart the application.",
                'error_code': 'TYPE_001'
            }
        }
    
    def handle_error(self, error: Exception, context: str = None) -> UserFriendlyError:
        """
        Convert a technical exception into a user-friendly error.
        
        Args:
            error: The original exception
            context: Additional context about where the error occurred
            
        Returns:
            UserFriendlyError with user-friendly message and recovery suggestions
        """
        error_type = type(error)
        
        # Get error mapping or use default
        error_info = self.error_mappings.get(error_type, {
            'user_message': "❌ An unexpected error occurred",
            'recovery_suggestion': "Please try again. If the problem persists, contact support "
                                "with the error details below.",
            'error_code': 'UNKNOWN'
        })
        
        # Add context to recovery suggestion if provided
        if context:
            recovery_suggestion = f"{error_info['recovery_suggestion']} (Context: {context})"
        else:
            recovery_suggestion = error_info['recovery_suggestion']
        
        # Create user-friendly error
        user_error = UserFriendlyError(
            user_message=error_info['user_message'],
            recovery_suggestion=recovery_suggestion,
            technical_details=str(error),
            error_code=error_info['error_code']
        )
        
        # Log the technical error for debugging
        logger.error(f"User-friendly error created: {error_info['error_code']} - {str(error)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return user_error
    
    def format_error_message(self, user_error: UserFriendlyError) -> str:
        """
        Format a user-friendly error message for display.
        
        Args:
            user_error: UserFriendlyError instance
            
        Returns:
            Formatted error message string
        """
        message_parts = [
            f"**{user_error.user_message}**",
            "",
            "### 💡 What to do:",
            f"{user_error.recovery_suggestion}",
            ""
        ]
        
        # Add retry button suggestion
        message_parts.extend([
            "### 🔄 Try again:",
            "• Click the **Retry** button below",
            "• Upload a different image",
            "• Check the image file integrity"
        ])
        
        # Add support information for persistent issues
        message_parts.extend([
            "",
            "### 🆘 Still having trouble?",
            "• Contact our support team",
            f"• Reference error code: **{user_error.error_code}**",
            "• Include a screenshot of this message"
        ])
        
        return "\n".join(message_parts)
    
    def create_toast_message(self, user_error: UserFriendlyError) -> Dict[str, str]:
        """
        Create a toast notification message for quick errors.
        
        Args:
            user_error: UserFriendlyError instance
            
        Returns:
            Dictionary with toast message data
        """
        return {
            'title': user_error.user_message,
            'message': user_error.recovery_suggestion,
            'type': self._get_toast_type(user_error.error_code),
            'duration': 5000
        }
    
    def _get_toast_type(self, error_code: str) -> str:
        """Determine toast notification type based on error code."""
        if error_code.startswith('IMG_'):
            return 'warning'
        elif error_code.startswith('GPU_') or error_code.startswith('MEM_'):
            return 'error'
        elif error_code.startswith('FILE_'):
            return 'warning'
        else:
            return 'info'


# Global error handler instance
error_handler = ErrorHandler()


def handle_user_errors(context: str = None):
    """
    Decorator for automatically handling errors in user-facing functions.
    
    Args:
        context: Description of the function context for error messages
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except UserFriendlyError:
                # Re-raise user-friendly errors as-is
                raise
            except Exception as e:
                # Convert technical errors to user-friendly errors
                user_error = error_handler.handle_error(e, context)
                raise user_error
        return wrapper
    return decorator


def validate_image_file(image_data: bytes) -> Tuple[bool, Optional[UserFriendlyError]]:
    """
    Validate image file before processing.
    
    Args:
        image_data: Raw image data bytes
        
    Returns:
        Tuple of (is_valid, error_if_invalid)
    """
    try:
        # Check if data is empty
        if not image_data:
            return False, UserFriendlyError(
                user_message="📂 Empty file uploaded",
                recovery_suggestion="Please select a valid image file to upload.",
                error_code="IMG_003"
            )
        
        # Check file size (10MB limit)
        if len(image_data) > 10 * 1024 * 1024:
            return False, UserFriendlyError(
                user_message="📏 File too large",
                recovery_suggestion="Please upload an image smaller than 10MB. "
                                  "Try compressing the image or using a smaller file.",
                error_code="IMG_004"
            )
        
        # Try to open the image
        try:
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify without loading
        except UnidentifiedImageError:
            return False, UserFriendlyError(
                user_message="🖼️ Invalid image format",
                recovery_suggestion="Please upload a valid image file (JPG, PNG, GIF, or BMP). "
                                  "The file appears to be corrupted or not an image.",
                error_code="IMG_005"
            )
        except Exception as e:
            return False, error_handler.handle_error(e, "Image validation")
        
        return True, None
        
    except Exception as e:
        return False, error_handler.handle_error(e, "Image validation")


def safe_image_operation(operation_name: str):
    """
    Decorator for safely performing image operations with error handling.
    
    Args:
        operation_name: Description of the image operation
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except UserFriendlyError:
                raise
            except Exception as e:
                user_error = error_handler.handle_error(e, operation_name)
                raise user_error
        return wrapper
    return decorator


# Import io for image validation
import io
