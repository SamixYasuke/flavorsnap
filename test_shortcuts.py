import sys
import os

# Import dashboard to test the instance
try:
    import dashboard
except Exception as e:
    print(f"Failed to load dashboard: {e}")
    sys.exit(1)

def test_shortcuts():
    try:
        # Initially history is not visible
        assert dashboard.ui.history_panel.visible == False
        
        # Test ctrl+h
        print("Testing ctrl+h")
        dashboard.keyboard_manager.keyboard_event.value = 'ctrl+h'
        assert dashboard.ui.history_panel.visible == True
        
        print("Testing ctrl+h again")
        dashboard.keyboard_manager.keyboard_event.value = 'ctrl+h'
        assert dashboard.ui.history_panel.visible == False
        
        # Test ctrl+d
        print("Testing ctrl+d")
        assert 'dark-theme' not in dashboard.app.css_classes
        dashboard.keyboard_manager.keyboard_event.value = 'ctrl+d'
        assert 'dark-theme' in dashboard.app.css_classes
        
        print("Testing ctrl+d again")
        dashboard.keyboard_manager.keyboard_event.value = 'ctrl+d'
        assert 'dark-theme' not in dashboard.app.css_classes
        
        # Test enter (Classify)
        print("Testing enter")
        dashboard.keyboard_manager.keyboard_event.value = 'enter'
        # With no image, it should show a warning
        assert dashboard.ui.output.object == "⚠️ Please upload an image first."
        
        # Test escape (Clear)
        print("Testing escape")
        dashboard.keyboard_manager.keyboard_event.value = 'escape'
        assert dashboard.ui.output.object == "Upload an image of food 🍲"
        
        # Test ctrl+s (Export)
        print("Testing ctrl+s")
        # Currently, if no image is visible, it shouldn't crash
        dashboard.keyboard_manager.keyboard_event.value = 'ctrl+s'
        
        print("✅ ALL TESTS PASSED.")
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    test_shortcuts()
