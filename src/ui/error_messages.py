"""
Error Message Templates and UI Components for FlavorSnap

This module provides pre-defined error message templates and UI components
for displaying user-friendly error messages with recovery options.
"""

import panel as pn
from typing import Dict, Any, Optional
import param
from ..utils.error_handler import error_handler, UserFriendlyError


class ErrorMessageTemplates:
    """Collection of pre-defined error message templates."""
    
    # Image upload errors
    IMAGE_UPLOAD_FAILED = """
    **🖼️ Image Upload Failed**
    
    The image could not be processed. This usually happens when:
    - The file is not a valid image format
    - The image is corrupted
    - The file is too large
    
    **Try these solutions:**
    • Upload a JPG, PNG, or GIF file under 10MB
    • Open the image in an editor first to verify it's not corrupted
    • Try a different image file
    """
    
    IMAGE_TOO_LARGE = """
    **📏 Image Too Large**
    
    This image exceeds our size limits for optimal processing.
    
    **Quick fix:**
    • Resize the image to under 5MB
    • Use an online image compressor
    • Take a new photo with lower resolution
    """
    
    INVALID_FORMAT = """
    **🔧 Invalid File Format**
    
    This file type is not supported for image classification.
    
    **Supported formats:**
    • JPEG (.jpg, .jpeg)
    • PNG (.png)
    • GIF (.gif)
    • BMP (.bmp)
    
    Please convert your file to one of these formats and try again.
    """
    
    # Classification errors
    CLASSIFICATION_FAILED = """
    **🤖 AI Model Error**
    
    The AI model encountered an issue while analyzing your image.
    
    **What to do:**
    • Click the **Retry** button below
    • Upload a different image
    • Check if the image is clear and well-lit
    • Try preprocessing the image with the controls
    """
    
    MODEL_LOADING_ERROR = """
    **🎯 Model Loading Error**
    
    The AI model could not be loaded properly.
    
    **Solutions:**
    • Restart the application
    • Check your internet connection
    • Ensure you have enough memory available
    • Contact support if the problem persists
    """
    
    # System errors
    MEMORY_ERROR = """
    **💾 Memory Error**
    
    The system ran out of memory while processing your image.
    
    **Quick fixes:**
    • Upload a smaller image (under 2MB)
    • Close other applications
    • Restart the application
    • Try using a device with more RAM
    """
    
    GPU_ERROR = """
    **🎮 GPU Processing Error**
    
    The graphics processor is busy or unavailable.
    
    **Try these:**
    • Wait a moment and try again
    • Restart the application
    • The system will automatically switch to CPU if needed
    """
    
    # File system errors
    FILE_PERMISSION_ERROR = """
    **🔒 File Access Error**
    
    The application cannot access the necessary files.
    
    **Solutions:**
    • Restart the application with administrator privileges
    • Check if the uploads folder exists and is writable
    • Try running from a different location
    """
    
    # Network errors
    NETWORK_ERROR = """
    **🌐 Connection Error**
    
    Unable to connect to necessary services.
    
    **Check:**
    • Your internet connection
    • Firewall settings
    • Try again in a few moments
    """
    
    # Generic error
    GENERIC_ERROR = """
    **❌ Something Went Wrong**
    
    An unexpected error occurred.
    
    **Next steps:**
    • Try the operation again
    • Restart the application
    • If this continues, contact support with details
    """


class ErrorDisplayComponent:
    """UI component for displaying error messages with recovery options."""
    
    def __init__(self):
        self.error_pane = pn.pane.Markdown("", sizing_mode='stretch_width')
        self.retry_button = pn.widgets.Button(
            name='🔄 Retry',
            button_type='primary',
            visible=False
        )
        self.contact_support_button = pn.widgets.Button(
            name='🆘 Contact Support',
            button_type='default',
            visible=False
        )
        
        # Store current error and callback
        self.current_error = None
        self.retry_callback = None
        
        # Setup button callbacks
        self.retry_button.on_click(self._on_retry_click)
        self.contact_support_button.on_click(self._on_support_click)
    
    def display_error(self, error: UserFriendlyError, retry_callback=None):
        """
        Display an error message with recovery options.
        
        Args:
            error: UserFriendlyError instance
            retry_callback: Function to call when retry is clicked
        """
        self.current_error = error
        self.retry_callback = retry_callback
        
        # Format the error message
        formatted_message = error_handler.format_error_message(error)
        self.error_pane.object = formatted_message
        
        # Show appropriate buttons
        self.retry_button.visible = retry_callback is not None
        self.contact_support_button.visible = True
        
        # Add error styling
        self.error_pane.css_classes = ['error-message']
    
    def clear_error(self):
        """Clear the current error display."""
        self.error_pane.object = ""
        self.retry_button.visible = False
        self.contact_support_button.visible = False
        self.current_error = None
        self.retry_callback = None
    
    def _on_retry_click(self, event):
        """Handle retry button click."""
        if self.retry_callback:
            self.clear_error()
            self.retry_callback()
    
    def _on_support_click(self, event):
        """Handle contact support button click."""
        if self.current_error:
            support_message = f"""
            **Support Request Generated**
            
            Error Code: {self.current_error.error_code}
            Message: {self.current_error.user_message}
            Recovery: {self.current_error.recovery_suggestion}
            
            Please include this information when contacting support.
            """
            self.error_pane.object = support_message
    
    def get_layout(self):
        """Get the complete error display layout."""
        return pn.Column(
            self.error_pane,
            pn.Row(
                self.retry_button,
                self.contact_support_button,
                sizing_mode='fixed_width'
            ),
            visible=False,
            css_classes=['error-container']
        )


class ToastNotification:
    """Toast notification system for quick error messages."""
    
    def __init__(self):
        self.active_toasts = []
    
    def show_toast(self, message: str, toast_type: str = 'info', duration: int = 3000):
        """
        Show a toast notification.
        
        Args:
            message: Message to display
            toast_type: Type of toast (success, error, warning, info)
            duration: Duration in milliseconds
        """
        toast = pn.pane.HTML(
            self._create_toast_html(message, toast_type),
            sizing_mode='fixed_width',
            width=300,
            height=80
        )
        
        # Add to active toasts
        self.active_toasts.append(toast)
        
        # Auto-remove after duration
        if duration > 0:
            import time
            pn.state.add_periodic_callback(
                lambda: self._remove_toast(toast),
                count=1,
                period=duration
            )
        
        return toast
    
    def show_error_toast(self, user_error: UserFriendlyError):
        """Show an error toast from UserFriendlyError."""
        toast_data = error_handler.create_toast_message(user_error)
        return self.show_toast(
            f"{toast_data['title']}: {toast_data['message']}",
            toast_data['type'],
            toast_data['duration']
        )
    
    def _create_toast_html(self, message: str, toast_type: str) -> str:
        """Create HTML for toast notification."""
        css_classes = {
            'success': 'toast-success',
            'error': 'toast-error',
            'warning': 'toast-warning',
            'info': 'toast-info'
        }
        
        css_class = css_classes.get(toast_type, 'toast-info')
        
        return f"""
        <div class="toast-notification {css_class}">
            <div class="toast-content">
                <span class="toast-message">{message}</span>
            </div>
        </div>
        """
    
    def _remove_toast(self, toast):
        """Remove a toast notification."""
        if toast in self.active_toasts:
            self.active_toasts.remove(toast)


class ErrorStateManager:
    """Manages error state across the application."""
    
    def __init__(self):
        self.error_history = []
        self.current_error = None
        self.error_display = ErrorDisplayComponent()
        self.toast_system = ToastNotification()
    
    def handle_error(self, error: Exception, context: str = None, retry_callback=None):
        """
        Handle an error with appropriate UI updates.
        
        Args:
            error: Exception to handle
            context: Context where error occurred
            retry_callback: Optional retry function
        """
        # Convert to user-friendly error if needed
        if isinstance(error, UserFriendlyError):
            user_error = error
        else:
            user_error = error_handler.handle_error(error, context)
        
        # Store error
        self.current_error = user_error
        self.error_history.append(user_error)
        
        # Keep only last 10 errors
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        
        # Display error
        self.error_display.display_error(user_error, retry_callback)
        
        # Show toast for quick feedback
        self.toast_system.show_error_toast(user_error)
        
        return user_error
    
    def clear_error(self):
        """Clear current error state."""
        self.current_error = None
        self.error_display.clear_error()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        error_counts = {}
        for error in self.error_history:
            code = error.error_code
            error_counts[code] = error_counts.get(code, 0) + 1
        
        return {
            'current_error': self.current_error.error_code if self.current_error else None,
            'total_errors': len(self.error_history),
            'error_counts': error_counts,
            'recent_errors': [e.error_code for e in self.error_history[-5:]]
        }


# Global error state manager
error_state_manager = ErrorStateManager()


def create_error_banner() -> pn.Column:
    """Create a reusable error banner component."""
    return error_state_manager.error_display.get_layout()


def show_error_toast(error: Exception, context: str = None):
    """Quick function to show an error toast."""
    if isinstance(error, UserFriendlyError):
        error_state_manager.toast_system.show_error_toast(error)
    else:
        user_error = error_handler.handle_error(error, context)
        error_state_manager.toast_system.show_error_toast(user_error)


def handle_and_display_error(error: Exception, context: str = None, retry_callback=None):
    """Handle error and display it with full UI components."""
    return error_state_manager.handle_error(error, context, retry_callback)


# Panel extension for error styling
def setup_error_styles():
    """Setup custom CSS for error components."""
    error_css = """
    <style>
    .error-container {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .error-message {
        color: #c53030;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .error-message h3 {
        color: #e53e3e;
        margin-bottom: 8px;
    }
    
    .error-message ul {
        margin-left: 20px;
        line-height: 1.6;
    }
    
    .toast-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.3s ease-out;
    }
    
    .toast-success {
        background: #48bb78;
        color: white;
    }
    
    .toast-error {
        background: #f56565;
        color: white;
    }
    
    .toast-warning {
        background: #ed8936;
        color: white;
    }
    
    .toast-info {
        background: #4299e1;
        color: white;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    </style>
    """
    
    return error_css
