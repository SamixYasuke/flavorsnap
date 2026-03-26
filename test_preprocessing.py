"""
Test script for preprocessing functionality
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all modules can be imported."""
    try:
        from src.core.image_enhancer import ImageEnhancer
        print("✅ ImageEnhancer imported successfully")
        
        from src.ui.preprocessing_controls import PreprocessingControls
        print("✅ PreprocessingControls imported successfully")
        
        from src.core.classifier import FlavorSnapClassifier
        print("✅ FlavorSnapClassifier imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_image_enhancer():
    """Test image enhancer functionality."""
    try:
        from src.core.image_enhancer import ImageEnhancer
        from PIL import Image
        import numpy as np
        
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        # Test enhancer
        enhancer = ImageEnhancer()
        enhancer.load_image(test_image)
        
        # Test brightness
        result = enhancer.apply_brightness(1.5)
        assert result is not None
        print("✅ Brightness adjustment works")
        
        # Test contrast
        result = enhancer.apply_contrast(1.2)
        assert result is not None
        print("✅ Contrast adjustment works")
        
        # Test rotation
        result = enhancer.apply_rotation(45)
        assert result is not None
        print("✅ Rotation works")
        
        # Test reset
        enhancer.reset_parameters()
        params = enhancer.get_enhancement_params()
        assert params['brightness'] == 1.0
        assert params['contrast'] == 1.0
        assert params['rotation'] == 0
        print("✅ Reset functionality works")
        
        return True
    except Exception as e:
        print(f"❌ Image enhancer test failed: {e}")
        return False

def test_classifier():
    """Test classifier functionality."""
    try:
        from src.core.classifier import FlavorSnapClassifier
        from PIL import Image
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='blue')
        
        # Test classifier initialization (will use untrained model)
        classifier = FlavorSnapClassifier()
        print("✅ Classifier initialized successfully")
        
        # Test image validation
        is_valid, error = classifier.validate_image(test_image)
        assert is_valid, f"Image validation failed: {error}"
        print("✅ Image validation works")
        
        # Test classification (will use untrained model)
        result = classifier.classify_image(test_image)
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'preprocessing_applied' in result
        print("✅ Classification works")
        
        # Test preprocessing recommendations
        recommendations = classifier.get_preprocessing_recommendations(test_image)
        assert 'recommendations' in recommendations
        assert 'analysis' in recommendations
        print("✅ Preprocessing recommendations work")
        
        # Test with preprocessing
        preprocessing_params = {
            'brightness': 1.2,
            'contrast': 1.1,
            'rotation': 0
        }
        result_with_preprocessing = classifier.classify_image(test_image, preprocessing_params)
        assert result_with_preprocessing['preprocessing_applied'] == True
        print("✅ Classification with preprocessing works")
        
        return True
    except Exception as e:
        print(f"❌ Classifier test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist."""
    required_files = [
        'src/core/image_enhancer.py',
        'src/ui/preprocessing_controls.py',
        'src/core/classifier.py',
        'static/css/controls.css',
        'static/js/preprocessing.js',
        'dashboard.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def main():
    """Run all tests."""
    print("🧪 Testing FlavorSnap Preprocessing Implementation")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Image Enhancer", test_image_enhancer),
        ("Classifier", test_classifier)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
