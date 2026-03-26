# 🎨 Real-time Image Preprocessing Controls

This feature adds real-time image preprocessing capabilities to FlavorSnap, allowing users to enhance images before classification for improved accuracy.

## ✨ Features

### 🎛️ Real-time Controls
- **Brightness Adjustment**: 0.0 to 2.0 range with real-time preview
- **Contrast Adjustment**: 0.0 to 2.0 range with real-time preview  
- **Rotation Control**: -180° to +180° rotation with preview
- **Aspect Ratio Presets**: 1:1, 4:3, 16:9, 3:2, 9:16, and Original
- **Crop Tool**: Custom crop with X, Y, width, and height controls
- **Auto Enhance**: Intelligent automatic enhancement using CLAHE and sharpening
- **Reset Button**: Instantly restore original image

### 🖼️ Dual Preview System
- **Original Image Preview**: Shows the uploaded image
- **Processed Image Preview**: Real-time preview of all applied enhancements

### 📊 Enhanced Classification
- Preprocessing parameters displayed in results
- Processed image saved to training data
- Confidence scores with preprocessing metadata

## 📁 File Structure

```
flavorsnap/
├── src/
│   ├── core/
│   │   ├── image_enhancer.py      # Core image enhancement logic
│   │   └── classifier.py          # Enhanced classifier with preprocessing
│   └── ui/
│       └── preprocessing_controls.py  # Panel UI components
├── static/
│   ├── css/
│   │   └── controls.css           # Styling for preprocessing controls
│   └── js/
│       └── preprocessing.js       # Real-time preview JavaScript
├── dashboard.py                   # Updated main dashboard
├── test_preprocessing.py          # Test suite
└── requirements_preprocessing.txt # Dependencies
```

## 🚀 Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_preprocessing.txt
   ```

2. **Run the Application**:
   ```bash
   python dashboard.py
   ```

3. **Access the Interface**:
   Open your browser to `http://localhost:5006`

## 🎯 Usage Guide

### Basic Workflow

1. **Upload an Image**: Click the file input to select a food image
2. **Preview Original**: The original image appears on the left
3. **Apply Enhancements**: Use the controls on the right to adjust:
   - Brightness and contrast sliders
   - Rotation for straightening
   - Aspect ratio presets for standard formats
   - Crop tool for custom framing
4. **Real-time Preview**: See changes instantly in the processed preview
5. **Classify**: Click "Classify" to analyze the enhanced image
6. **View Results**: See classification with preprocessing parameters

### Advanced Features

#### Auto Enhancement
- Click "✨ Auto Enhance" for intelligent image improvement
- Uses CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Applies subtle sharpening for better clarity

#### Crop Controls
- Enable crop with the checkbox
- Adjust X, Y position and width, height
- Real-time preview shows crop boundaries
- Perfect for focusing on food items

#### Aspect Ratio Presets
- **1:1**: Square format for social media
- **4:3**: Standard photo format
- **16:9**: Widescreen format
- **3:2**: Classic photo ratio
- **9:16**: Vertical format

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_preprocessing.py
```

The test suite covers:
- ✅ Module imports
- ✅ Image enhancement operations
- ✅ Classifier integration
- ✅ File structure validation
- ✅ Preprocessing recommendations

## 🔧 Technical Implementation

### Core Components

#### ImageEnhancer Class
```python
from src.core.image_enhancer import ImageEnhancer

enhancer = ImageEnhancer()
enhancer.load_image(image)
enhancer.apply_brightness(1.5)
enhancer.apply_contrast(1.2)
enhancer.apply_rotation(45)
processed_image = enhancer.get_processed_image()
```

#### PreprocessingControls UI
```python
from src.ui.preprocessing_controls import PreprocessingControls

controls = PreprocessingControls()
controls.load_image(image)
panel = controls.create_layout()
```

#### Enhanced Classifier
```python
from src.core.classifier import FlavorSnapClassifier

classifier = FlavorSnapClassifier()
result = classifier.classify_image(
    image, 
    preprocessing_params={'brightness': 1.2, 'contrast': 1.1}
)
```

### Preprocessing Parameters

```python
params = {
    'brightness': 1.0,      # 0.0 to 2.0 (1.0 = original)
    'contrast': 1.0,        # 0.0 to 2.0 (1.0 = original)
    'rotation': 0.0,        # -180 to 180 degrees
    'crop_box': None,       # (left, top, right, bottom) or None
    'aspect_ratio': None     # '1:1', '4:3', '16:9', etc. or None
}
```

## 🎨 Styling

The controls use modern CSS with:
- Gradient backgrounds
- Smooth transitions and hover effects
- Responsive design for mobile devices
- Custom slider styling
- Loading animations
- Success/error message styling

## 📱 Responsive Design

- **Desktop**: Side-by-side layout with full controls
- **Tablet**: Stacked layout with adjusted sizing
- **Mobile**: Single column layout with compact controls

## 🔍 API Integration

### Classification Endpoint
```python
# Enhanced classification with preprocessing
result = classifier.classify_image(
    image=image,
    preprocessing_params={
        'brightness': 1.3,
        'contrast': 1.1,
        'rotation': 5
    }
)
```

### Auto Enhancement
```python
# Get preprocessing recommendations
recommendations = classifier.get_preprocessing_recommendations(image)
params = recommendations['recommendations']
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Loading**: Check model path in `dashboard.py`
3. **Image Loading**: Verify image format is supported (JPEG, PNG, etc.)
4. **Performance**: Large images may take longer to process

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🚀 Future Enhancements

### Planned Features
- [ ] Advanced filters (blur, sharpen, noise reduction)
- [ ] Color balance controls
- [ ] Batch processing for multiple images
- [ ] Preset enhancement profiles
- [ ] Undo/redo functionality
- [ ] Keyboard shortcuts
- [ ] Touch gesture support

### Performance Optimizations
- [ ] GPU acceleration for image processing
- [ ] WebAssembly for client-side processing
- [ ] Image caching for faster previews
- [ ] Progressive loading for large images

## 📄 License

This feature is part of the FlavorSnap project and follows the same license terms.

## 🤝 Contributing

To contribute to this feature:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues or questions about this feature:
- Create an issue in the repository
- Check the troubleshooting section
- Review the test suite for usage examples
