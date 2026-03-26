"""
Simple validation script for FlavorSnap Error Handling System
This script validates the error handling implementation without running the full application.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def validate_imports():
    """Validate that all error handling modules can be imported."""
    print("🔍 Validating Error Handling Imports...")
    
    try:
        from src.utils.error_handler import (
            ErrorHandler, 
            UserFriendlyError, 
            validate_image_file,
            handle_user_errors,
            error_handler
        )
        print("✅ utils.error_handler imports successful")
    except Exception as e:
        print(f"❌ utils.error_handler import failed: {e}")
        return False
    
    try:
        from src.ui.error_messages import (
            ErrorMessageTemplates,
            ErrorDisplayComponent,
            ErrorStateManager,
            handle_and_display_error,
            create_error_banner
        )
        print("✅ ui.error_messages imports successful")
    except Exception as e:
        print(f"❌ ui.error_messages import failed: {e}")
        return False
    
    try:
        from src.core.classifier import FlavorSnapClassifier
        print("✅ core.classifier imports successful")
    except Exception as e:
        print(f"❌ core.classifier import failed: {e}")
        return False
    
    return True

def validate_error_handler_functionality():
    """Validate basic error handler functionality."""
    print("\n🔍 Validating Error Handler Functionality...")
    
    try:
        from src.utils.error_handler import ErrorHandler, UserFriendlyError
        
        # Test error handler creation
        handler = ErrorHandler()
        print("✅ Error handler instance created")
        
        # Test user-friendly error creation
        user_error = UserFriendlyError(
            user_message="Test error message",
            recovery_suggestion="Test recovery suggestion",
            error_code="TEST_001"
        )
        print("✅ UserFriendlyError instance created")
        
        # Test error formatting
        formatted = handler.format_error_message(user_error)
        assert "Test error message" in formatted
        assert "What to do:" in formatted
        print("✅ Error message formatting works")
        
        # Test toast creation
        toast = handler.create_toast_message(user_error)
        assert toast['title'] == "Test error message"
        assert 'type' in toast
        print("✅ Toast notification creation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handler functionality test failed: {e}")
        return False

def validate_file_structure():
    """Validate that all required files are created."""
    print("\n🔍 Validating File Structure...")
    
    required_files = [
        'src/utils/error_handler.py',
        'src/ui/error_messages.py', 
        'static/css/error.css'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def validate_dashboard_integration():
    """Validate that dashboard.py has been properly modified."""
    print("\n🔍 Validating Dashboard Integration...")
    
    try:
        with open('dashboard.py', 'r') as f:
            content = f.read()
        
        # Check for required imports
        required_imports = [
            'from src.ui.error_messages import',
            'from src.utils.error_handler import',
            'UserFriendlyError'
        ]
        
        for import_stmt in required_imports:
            if import_stmt in content:
                print(f"✅ Found import: {import_stmt}")
            else:
                print(f"❌ Missing import: {import_stmt}")
                return False
        
        # Check for error handling functions
        required_functions = [
            '@handle_user_errors',
            'error_banner',
            'handle_and_display_error'
        ]
        
        for func in required_functions:
            if func in content:
                print(f"✅ Found error handling: {func}")
            else:
                print(f"❌ Missing error handling: {func}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard integration validation failed: {e}")
        return False

def validate_classifier_integration():
    """Validate that classifier.py has been properly modified."""
    print("\n🔍 Validating Classifier Integration...")
    
    try:
        with open('src/core/classifier.py', 'r') as f:
            content = f.read()
        
        # Check for required imports
        if 'from src.utils.error_handler import' in content:
            print("✅ Found error handler imports in classifier")
        else:
            print("❌ Missing error handler imports in classifier")
            return False
        
        # Check for decorators
        decorators = ['@handle_user_errors', '@safe_image_operation']
        for decorator in decorators:
            if decorator in content:
                print(f"✅ Found decorator: {decorator}")
            else:
                print(f"❌ Missing decorator: {decorator}")
                return False
        
        # Check for UserFriendlyError usage
        if 'UserFriendlyError(' in content:
            print("✅ Found UserFriendlyError usage")
        else:
            print("❌ Missing UserFriendlyError usage")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Classifier integration validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 FlavorSnap Error Handling Validation\n")
    
    tests = [
        ("File Structure", validate_file_structure),
        ("Imports", validate_imports), 
        ("Error Handler Functionality", validate_error_handler_functionality),
        ("Dashboard Integration", validate_dashboard_integration),
        ("Classifier Integration", validate_classifier_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 VALIDATION SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validations passed! Error handling system is ready.")
        print("\n✨ Features Implemented:")
        print("  • User-friendly error messages")
        print("  • Recovery suggestions")
        print("  • Toast notifications")
        print("  • Error display components")
        print("  • Comprehensive error handling decorators")
        print("  • Integration with dashboard and classifier")
        return True
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
