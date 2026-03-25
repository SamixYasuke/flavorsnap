"""
Test script for export functionality
"""

import sys
import os
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.export_manager import ExportManager

def test_export_functionality():
    """Test all export formats"""
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # Initialize export manager
    export_manager = ExportManager()
    
    print("Testing export functionality...")
    
    # Test single exports
    formats = ['csv', 'json', 'pdf', 'image']
    
    for fmt in formats:
        try:
            filepath = export_manager.export_single_result(
                image=test_image,
                predicted_class='Test Food',
                confidence=0.85,
                export_format=fmt,
                filename=f'test_{fmt}_export'
            )
            print(f"✅ {fmt.upper()} export successful: {filepath}")
        except Exception as e:
            print(f"❌ {fmt.upper()} export failed: {str(e)}")
    
    # Test batch export
    batch_results = [
        {
            'timestamp': '2024-01-01T10:00:00',
            'predicted_class': 'Bread',
            'confidence': 0.92,
            'image_data': test_image
        },
        {
            'timestamp': '2024-01-01T10:05:00',
            'predicted_class': 'Rice',
            'confidence': 0.78,
            'image_data': test_image
        }
    ]
    
    for fmt in ['csv', 'json', 'pdf']:
        try:
            filepath = export_manager.export_batch_results(
                results=batch_results,
                export_format=fmt,
                filename=f'test_batch_{fmt}_export'
            )
            print(f"✅ Batch {fmt.upper()} export successful: {filepath}")
        except Exception as e:
            print(f"❌ Batch {fmt.upper()} export failed: {str(e)}")
    
    # Test export history
    history = export_manager.get_export_history()
    print(f"📊 Export history contains {len(history)} entries")
    
    print("\nExport functionality test completed!")

if __name__ == "__main__":
    test_export_functionality()
