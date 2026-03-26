#!/usr/bin/env python3
"""
Test script for real-time preview functionality
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test if all modules can be imported."""
    try:
        print("Testing imports...")
        
        # Test debounced classifier
        from src.core.debounced_classifier import get_debounced_classifier
        print("✅ Debounced classifier import successful")
        
        # Test optimization suggester
        from src.utils.optimization_suggester import get_optimization_suggestions
        print("✅ Optimization suggester import successful")
        
        # Test real-time preview
        from src.ui.realtime_preview import RealtimePreview
        print("✅ Real-time preview import successful")
        
        # Test updated classifier
        from src.core.classifier import FlavorSnapClassifier
        print("✅ Updated classifier import successful")
        
        # Test updated preprocessing controls
        from src.ui.preprocessing_controls import PreprocessingControls
        print("✅ Updated preprocessing controls import successful")
        
        # Test basic functionality
        classifier = get_debounced_classifier()
        print(f"✅ Debounced classifier initialized: {type(classifier)}")
        
        preview = RealtimePreview()
        print(f"✅ Real-time preview initialized: {type(preview)}")
        
        controls = PreprocessingControls()
        print(f"✅ Preprocessing controls initialized: {type(controls)}")
        
        print("\n🎉 All real-time modules imported and initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Import/initialization error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components."""
    try:
        print("\nTesting integration...")
        
        # Test classifier integration
        from src.core.classifier import FlavorSnapClassifier
        classifier = FlavorSnapClassifier()
        
        # Test real-time methods
        if hasattr(classifier, 'classify_image_realtime'):
            print("✅ Real-time classification method available")
        else:
            print("❌ Real-time classification method missing")
            return False
        
        if hasattr(classifier, 'enable_realtime'):
            print("✅ Real-time enable method available")
        else:
            print("❌ Real-time enable method missing")
            return False
        
        # Test preprocessing controls integration
        from src.ui.preprocessing_controls import PreprocessingControls
        controls = PreprocessingControls()
        
        if hasattr(controls, 'set_realtime_callback'):
            print("✅ Real-time callback setter available")
        else:
            print("❌ Real-time callback setter missing")
            return False
        
        if hasattr(controls, 'is_realtime_enabled'):
            print("✅ Real-time status checker available")
        else:
            print("❌ Real-time status checker missing")
            return False
        
        print("\n🎉 Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Real-time Preview Implementation")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test integration
    if not test_integration():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Real-time functionality is ready.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
