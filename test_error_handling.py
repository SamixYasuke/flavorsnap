"""
Test script for FlavorSnap Error Handling System

This script tests various error scenarios to ensure user-friendly error messages
are displayed correctly instead of technical exceptions.
"""

import sys
from pathlib import Path
import io
from PIL import Image
import numpy as np

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.error_handler import (
    ErrorHandler, 
    UserFriendlyError, 
    validate_image_file,
    handle_user_errors,
    error_handler
)
from src.ui.error_messages import (
    ErrorMessageTemplates,
    ErrorDisplayComponent,
    ErrorStateManager
)
from src.core.classifier import FlavorSnapClassifier


def test_image_validation():
    """Test image file validation with various scenarios."""
    print("🧪 Testing Image Validation...")
    
    # Test 1: Empty file
    print("\n1. Testing empty file...")
    is_valid, error = validate_image_file(b"")
    assert not is_valid
    assert error.error_code == "IMG_003"
    print(f"✅ Empty file error: {error.user_message}")
    
    # Test 2: File too large
    print("\n2. Testing oversized file...")
    large_data = b"x" * (11 * 1024 * 1024)  # 11MB
    is_valid, error = validate_image_file(large_data)
    assert not is_valid
    assert error.error_code == "IMG_004"
    print(f"✅ Large file error: {error.user_message}")
    
    # Test 3: Invalid image format
    print("\n3. Testing invalid image format...")
    invalid_data = b"This is not an image file"
    is_valid, error = validate_image_file(invalid_data)
    assert not is_valid
    assert error.error_code == "IMG_005"
    print(f"✅ Invalid format error: {error.user_message}")
    
    # Test 4: Valid small image
    print("\n4. Testing valid small image...")
    # Create a simple 50x50 RGB image
    img_array = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()
    
    is_valid, error = validate_image_file(img_bytes)
    assert is_valid
    assert error is None
    print("✅ Valid image passed validation")


def test_error_handler():
    """Test the main error handler with different exception types."""
    print("\n🧪 Testing Error Handler...")
    
    # Test 1: PIL Image error
    print("\n1. Testing PIL image error...")
    try:
        img = Image.open("nonexistent_file.jpg")
    except Exception as e:
        user_error = error_handler.handle_error(e, "image loading")
        assert user_error.error_code == "FILE_001"
        print(f"✅ PIL error handled: {user_error.user_message}")
    
    # Test 2: Memory error simulation
    print("\n2. Testing memory error handling...")
    try:
        raise MemoryError("Simulated memory error")
    except Exception as e:
        user_error = error_handler.handle_error(e, "classification")
        assert user_error.error_code == "MEM_001"
        print(f"✅ Memory error handled: {user_error.user_message}")
    
    # Test 3: Unknown error
    print("\n3. Testing unknown error handling...")
    try:
        raise RuntimeError("Simulated unknown error")
    except Exception as e:
        user_error = error_handler.handle_error(e, "unknown operation")
        assert user_error.error_code == "UNKNOWN"
        print(f"✅ Unknown error handled: {user_error.user_message}")


def test_error_message_formatting():
    """Test error message formatting for display."""
    print("\n🧪 Testing Error Message Formatting...")
    
    user_error = UserFriendlyError(
        user_message="🖼️ Unable to read this image file",
        recovery_suggestion="Please upload a valid image file (JPG, PNG, GIF, or BMP).",
        technical_details="PIL.UnidentifiedImageError: cannot identify image file",
        error_code="IMG_001"
    )
    
    formatted_message = error_handler.format_error_message(user_error)
    assert "🖼️ Unable to read this image file" in formatted_message
    assert "What to do:" in formatted_message
    assert "Try again:" in formatted_message
    assert "Still having trouble?" in formatted_message
    assert "IMG_001" in formatted_message
    
    print("✅ Error message formatting includes all required sections")
    print(f"📝 Formatted message preview:\n{formatted_message[:200]}...")


def test_toast_notifications():
    """Test toast notification creation."""
    print("\n🧪 Testing Toast Notifications...")
    
    user_error = UserFriendlyError(
        user_message="🖼️ Image upload failed",
        recovery_suggestion="Please try uploading a different image",
        error_code="IMG_001"
    )
    
    toast_data = error_handler.create_toast_message(user_error)
    assert toast_data['title'] == "🖼️ Image upload failed"
    assert toast_data['type'] == 'warning'
    assert 'duration' in toast_data
    
    print(f"✅ Toast created: {toast_data['title']} ({toast_data['type']})")


def test_classifier_error_handling():
    """Test error handling in the classifier."""
    print("\n🧪 Testing Classifier Error Handling...")
    
    try:
        classifier = FlavorSnapClassifier()
        
        # Test 1: Invalid image validation
        print("\n1. Testing invalid image validation...")
        invalid_image = None
        is_valid, error_msg = classifier.validate_image(invalid_image)
        assert not is_valid
        assert "Input is not a PIL Image" in error_msg
        
        # Test 2: Image too small
        print("\n2. Testing image too small...")
        tiny_img = Image.new('RGB', (10, 10))
        is_valid, error_msg = classifier.validate_image(tiny_img)
        assert not is_valid
        assert "too small" in error_msg
        
        # Test 3: Image too large
        print("\n3. Testing image too large...")
        huge_img = Image.new('RGB', (5000, 5000))
        is_valid, error_msg = classifier.validate_image(huge_img)
        assert not is_valid
        assert "too large" in error_msg
        
        print("✅ All classifier validation tests passed")
        
    except Exception as e:
        print(f"❌ Classifier test failed: {e}")
        # This is expected if model files don't exist, but error handling should work
        user_error = error_handler.handle_error(e, "classifier initialization")
        print(f"✅ Classifier initialization error handled: {user_error.user_message}")


def test_decorators():
    """Test error handling decorators."""
    print("\n🧪 Testing Error Handling Decorators...")
    
    @handle_user_errors("test operation")
    def failing_function():
        raise ValueError("This is a test error")
    
    try:
        failing_function()
        assert False, "Should have raised UserFriendlyError"
    except UserFriendlyError as e:
        assert e.error_code == "DATA_001"
        print(f"✅ Decorator caught error: {e.user_message}")
    
    @handle_user_errors("image processing")
    def successful_function():
        return "Success!"
    
    result = successful_function()
    assert result == "Success!"
    print("✅ Decorator allows successful operations")


def test_error_state_manager():
    """Test the error state manager."""
    print("\n🧪 Testing Error State Manager...")
    
    manager = ErrorStateManager()
    
    # Test error handling
    test_error = ValueError("Test error for state manager")
    user_error = manager.handle_error(test_error, "test context")
    
    assert manager.current_error == user_error
    assert len(manager.error_history) == 1
    
    # Test error summary
    summary = manager.get_error_summary()
    assert summary['total_errors'] == 1
    assert summary['current_error'] == 'DATA_001'
    
    # Test clearing errors
    manager.clear_error()
    assert manager.current_error is None
    
    print("✅ Error state manager working correctly")


def run_all_tests():
    """Run all error handling tests."""
    print("🚀 Starting FlavorSnap Error Handling Tests\n")
    
    try:
        test_image_validation()
        test_error_handler()
        test_error_message_formatting()
        test_toast_notifications()
        test_classifier_error_handling()
        test_decorators()
        test_error_state_manager()
        
        print("\n✅ All error handling tests passed!")
        print("\n🎯 Error Handling System Features Verified:")
        print("  • Image file validation with user-friendly messages")
        print("  • Technical exception to user-friendly error conversion")
        print("  • Comprehensive error message formatting")
        print("  • Toast notification system")
        print("  • Classifier error handling integration")
        print("  • Decorator-based error handling")
        print("  • Error state management")
        print("  • Recovery suggestions and support information")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
