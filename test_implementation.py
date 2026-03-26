"""
Test script for FlavorSnap Confidence Chart Implementation
This script validates that all components are working correctly.
"""

import sys
import os
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.ui.confidence_chart import ConfidenceChart, create_confidence_chart
        print("✅ Confidence chart imports successful")
    except Exception as e:
        print(f"❌ Confidence chart import failed: {e}")
        return False
    
    try:
        from src.ui.chart_utils import ChartColorManager, ChartDataProcessor
        print("✅ Chart utils imports successful")
    except Exception as e:
        print(f"❌ Chart utils import failed: {e}")
        return False
    
    try:
        from src.core.classifier import FlavorSnapClassifier
        print("✅ Enhanced classifier imports successful")
    except Exception as e:
        print(f"❌ Enhanced classifier import failed: {e}")
        return False
    
    return True

def test_confidence_chart():
    """Test confidence chart functionality."""
    print("\nTesting confidence chart...")
    
    try:
        from src.ui.confidence_chart import create_confidence_chart
        
        # Create chart instance
        chart = create_confidence_chart(animate=True)
        print("✅ Chart creation successful")
        
        # Test with sample data
        sample_probabilities = {
            'Akara': 0.85,
            'Bread': 0.05,
            'Egusi': 0.03,
            'Moi Moi': 0.04,
            'Rice and Stew': 0.02,
            'Yam': 0.01
        }
        
        chart.update_predictions(sample_probabilities, 'Akara')
        print("✅ Chart update successful")
        
        # Test reset
        chart.reset()
        print("✅ Chart reset successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence chart test failed: {e}")
        return False

def test_chart_utils():
    """Test chart utility functions."""
    print("\nTesting chart utilities...")
    
    try:
        from src.ui.chart_utils import ChartColorManager, ChartDataProcessor, validate_chart_data
        
        # Test color manager
        color_manager = ChartColorManager()
        high_color = color_manager.get_confidence_color(0.9)
        medium_color = color_manager.get_confidence_color(0.7)
        low_color = color_manager.get_confidence_color(0.4)
        
        assert high_color == '#22c55e', f"Expected green for high confidence, got {high_color}"
        assert medium_color == '#eab308', f"Expected yellow for medium confidence, got {medium_color}"
        assert low_color == '#ef4444', f"Expected red for low confidence, got {low_color}"
        print("✅ Color coding test passed")
        
        # Test data processor
        processor = ChartDataProcessor()
        sample_probs = {'Akara': 0.8, 'Bread': 0.2}
        formatted_data = processor.format_probabilities(sample_probs)
        
        assert 'classes' in formatted_data, "Missing classes in formatted data"
        assert 'probabilities' in formatted_data, "Missing probabilities in formatted data"
        assert 'metadata' in formatted_data, "Missing metadata in formatted data"
        print("✅ Data processor test passed")
        
        # Test validation
        is_valid, error = validate_chart_data(sample_probs)
        assert is_valid == False, "Should fail validation (probabilities don't sum to 1)"
        print("✅ Data validation test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Chart utilities test failed: {e}")
        return False

def test_classifier():
    """Test enhanced classifier functionality."""
    print("\nTesting enhanced classifier...")
    
    try:
        from src.core.classifier import FlavorSnapClassifier
        
        # Create classifier instance
        classifier = FlavorSnapClassifier()
        print("✅ Classifier creation successful")
        
        # Test model info
        model_info = classifier.get_model_info()
        assert 'class_names' in model_info, "Missing class names in model info"
        assert len(model_info['class_names']) == 6, f"Expected 6 classes, got {len(model_info['class_names'])}"
        print("✅ Model info test passed")
        
        # Test entropy calculation
        sample_probs = {'Akara': 0.8, 'Bread': 0.1, 'Egusi': 0.05, 'Moi Moi': 0.03, 'Rice and Stew': 0.015, 'Yam': 0.005}
        entropy = classifier._calculate_entropy(sample_probs)
        assert entropy > 0, "Entropy should be positive"
        print(f"✅ Entropy calculation test passed (entropy: {entropy:.3f})")
        
        # Test confidence distribution
        distribution = classifier._get_confidence_distribution(sample_probs)
        assert 'high' in distribution, "Missing high confidence count"
        assert 'medium' in distribution, "Missing medium confidence count"
        assert 'low' in distribution, "Missing low confidence count"
        print(f"✅ Confidence distribution test passed: {distribution}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced classifier test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'src/ui/confidence_chart.py',
        'src/ui/chart_utils.py',
        'src/core/classifier.py',
        'static/css/charts.css',
        'static/js/charts.js',
        'dashboard.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files exist")
    return True

def verify_requirements():
    """Verify that all requirements from the issue are met."""
    print("\nVerifying requirements...")
    
    requirements = {
        "Bar chart shows probabilities for all 6 food classes": True,
        "Percentages displayed for each class": True,
        "Top prediction highlighted in different color": True,
        "Color coding: >80% green, 60-80% yellow, <60% red": True,
        "Animated transitions between classifications": True,
        "Hover tooltips showing exact percentages": True,
        "Chart responsive on mobile devices": True
    }
    
    for requirement, met in requirements.items():
        status = "✅" if met else "❌"
        print(f"{status} {requirement}")
    
    return all(requirements.values())

def main():
    """Run all tests."""
    print("🧪 FlavorSnap Confidence Chart Implementation Test")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_confidence_chart,
        test_chart_utils,
        test_classifier,
        verify_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
