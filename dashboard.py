import panel as pn
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os
import sys
from pathlib import Path
from datetime import datetime
import hashlib

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ui.preprocessing_controls import PreprocessingControls
from src.ui.confidence_chart import create_confidence_chart
from src.ui.error_messages import handle_and_display_error, create_error_banner, setup_error_styles
from src.core.image_enhancer import ImageEnhancer
from src.core.classifier import FlavorSnapClassifier
from src.utils.error_handler import handle_user_errors, validate_image_file, UserFriendlyError
from src.pwa.offline_manager import PWAManager

# Configure Panel extensions with custom CSS and JS
pn.extension('css', js_files={
    'charts': ['static/js/charts.js'],
    'pwa': ['static/js/pwa.js']
}, css_files={
    'charts': ['static/css/charts.css'],
    'error': ['static/css/error.css'],
    'pwa': ['static/css/pwa.css']
})

# Initialize PWA Manager
pwa_manager = PWAManager("offline_data.db")

# PWA JavaScript template for service worker registration
pwa_js_template = """
<script>
// Register Service Worker
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
                
                // Check for updates
                registration.addEventListener('updatefound', () => {
                    const newWorker = registration.installing;
                    newWorker.addEventListener('statechange', () => {
                        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                            // New content is available
                            if (confirm('New version available! Reload to update?')) {
                                window.location.reload();
                            }
                        }
                    });
                });
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// PWA Install Prompt
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    
    // Show install button or banner
    const installButton = document.getElementById('pwa-install-button');
    if (installButton) {
        installButton.style.display = 'block';
        installButton.addEventListener('click', () => {
            deferredPrompt.prompt();
            deferredPrompt.userChoice.then((choiceResult) => {
                if (choiceResult.outcome === 'accepted') {
                    console.log('User accepted the A2HS prompt');
                } else {
                    console.log('User dismissed the A2HS prompt');
                }
                deferredPrompt = null;
            });
        });
    }
});

// Online/Offline Status Detection
window.addEventListener('online', () => {
    console.log('App is online');
    document.body.classList.remove('offline');
    
    // Notify the server about online status
    fetch('/api/pwa/status', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({online: true})
    }).catch(console.error);
});

window.addEventListener('offline', () => {
    console.log('App is offline');
    document.body.classList.add('offline');
    
    // Notify the server about offline status
    fetch('/api/pwa/status', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({online: false})
    }).catch(console.error);
});

// Initialize PWA status
if (!navigator.onLine) {
    document.body.classList.add('offline');
}
</script>
"""

# Load model using the enhanced classifier
classifier = FlavorSnapClassifier()
model = classifier.model
class_names = classifier.class_names

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Save image to correct folder
def save_image(image_obj, predicted_class, image_name="uploaded_image.jpg"):
    save_dir = f"data/train/{predicted_class}"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, image_name)
    image_obj.save(image_path)
    
    # Log analytics event for PWA
    if pwa_manager:
        pwa_manager.offline_manager.log_analytics_event('classification', {
            'predicted_class': predicted_class,
            'image_name': image_name,
            'timestamp': datetime.now().isoformat()
        })

# PWA API endpoint handlers
def handle_pwa_status(request):
    """Handle PWA status updates from client."""
    try:
        data = request.json()
        is_online = data.get('online', True)
        
        if pwa_manager:
            pwa_manager.set_online_status(is_online)
        
        return {'status': 'success', 'online': is_online}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def handle_pwa_sync(request):
    """Handle PWA sync requests."""
    try:
        if not pwa_manager:
            return {'status': 'error', 'message': 'PWA manager not available'}
        
        # Get pending sync items
        pending_items = pwa_manager.offline_manager.get_sync_queue('pending')
        
        # Process sync items
        synced_count = 0
        for item in pending_items:
            success = pwa_manager._process_sync_item(item)
            if success:
                pwa_manager.offline_manager.mark_synced(item['id'], 'synced')
                synced_count += 1
        
        return {
            'status': 'success',
            'synced_items': synced_count,
            'pending_items': len(pending_items) - synced_count
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def handle_pwa_cache(request):
    """Handle PWA cache requests."""
    try:
        if not pwa_manager:
            return {'status': 'error', 'message': 'PWA manager not available'}
        
        # Get cache statistics
        stats = pwa_manager.get_status()
        
        return {
            'status': 'success',
            'cache_stats': stats
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# Panel UI
image_input = pn.widgets.FileInput(accept='image/*')
output = pn.pane.Markdown("Upload an image of food 🍲")
image_preview = pn.pane.Image(width=300, height=300, visible=False)
processed_preview = pn.pane.Image(width=300, height=300, visible=False)
spinner = pn.indicators.LoadingSpinner(value=False, width=50)

# Create error display components
error_banner = create_error_banner()
error_banner.visible = False

# Create confidence chart component
confidence_chart = create_confidence_chart(animate=True)
confidence_chart_component = confidence_chart.create_layout()

# Preprocessing controls
preprocessing_controls = PreprocessingControls()
preprocessing_panel = preprocessing_controls.create_layout()

# Global variables
original_image = None
processed_image = None

def on_image_update(image):
    """Handle image updates from preprocessing controls."""
    global processed_image
    processed_image = image
    if image:
        processed_preview.object = image
        processed_preview.visible = True

@handle_user_errors("image upload and processing")
def handle_image_upload():
    """Handle image upload and initialize preprocessing."""
    global original_image, processed_image
    
    if image_input.value is None:
        return
    
    # Validate image file first
    is_valid, validation_error = validate_image_file(image_input.value)
    if not is_valid:
        raise validation_error
    
    # Clear any previous errors
    error_banner.visible = False
    
    # Load the image
    original_image = Image.open(io.BytesIO(image_input.value)).convert('RGB')
    processed_image = original_image.copy()
    
    # Update previews
    image_preview.object = original_image
    image_preview.visible = True
    processed_preview.object = processed_image
    processed_preview.visible = True
    
    # Load image into preprocessing controls
    preprocessing_controls.load_image(original_image)
    preprocessing_controls.on_image_update = on_image_update
    
    output.object = "📸 Image loaded! Use preprocessing controls to enhance, then classify."

@handle_user_errors("image classification")
def classify(event=None):
    """Classify the uploaded image with preprocessing."""
    global original_image, processed_image
    
    if image_input.value is None:
        output.object = "⚠️ Please upload an image first."
        image_preview.visible = False
        processed_preview.visible = False
        return
    
    if processed_image is None:
        output.object = "⚠️ Please wait for image to load or apply preprocessing."
        return
    
    # Clear any previous errors
    error_banner.visible = False
    
    # Start spinner
    spinner.value = True
    output.object = "🔍 Classifying..."

    # Use processed image for classification
    image_to_classify = processed_image

    # Get preprocessing parameters
    preprocessing_params = preprocessing_controls.get_enhancement_params()
    
    # Check if we have cached results for this image (PWA feature)
    image_hash = hashlib.md5(image_input.value).hexdigest()
    cache_key = f"classification_{image_hash}_{hash(str(preprocessing_params))}"
    
    if pwa_manager and not pwa_manager.is_online:
        # Try to get cached result when offline
        cached_result = pwa_manager.get_cached_api_response(cache_key)
        if cached_result:
            result = cached_result
            output.object = "📱 Using cached classification result (offline mode)"
        else:
            output.object = "📱 No cached result available. Please connect to internet."
            spinner.value = False
            return
    else:
        # Perform classification when online
        result = classifier.classify_image(image_to_classify, preprocessing_params)
        
        # Cache the result for offline use
        if pwa_manager:
            pwa_manager.cache_api_response(cache_key, result, expires_in_hours=24)
    
    # Extract results
    predicted_class = result['predicted_class']
    confidence_score = result['confidence']
    all_probabilities = result['all_probabilities']
    
    # Update confidence chart with all probabilities
    confidence_chart.update_predictions(all_probabilities, predicted_class)

    # Save processed image
    save_image(image_to_classify, predicted_class)
    
    # Create enhanced result message
    confidence_percentage = confidence_score * 100
    entropy = result['metadata']['entropy']
    avg_confidence = result['metadata']['average_confidence']
    
    # Add offline status indicator
    offline_indicator = "📱 (Offline Mode)" if pwa_manager and not pwa_manager.is_online else "🌐 (Online)"
    
    result_message = f"""
✅ **Classification Result: {predicted_class}** {offline_indicator}

### 🎯 Confidence Scores
- **Top Prediction:** {predicted_class} ({confidence_percentage:.1f}%)
- **Model Uncertainty (Entropy):** {entropy:.3f}
- **Average Confidence:** {avg_confidence:.1f}%

### 📊 Preprocessing Parameters Applied:
- **Brightness**: {preprocessing_params['brightness']:.1f}
- **Contrast**: {preprocessing_params['contrast']:.1f}
- **Rotation**: {preprocessing_params['rotation']:.0f}°
- **Aspect Ratio**: {preprocessing_params.get('aspect_ratio', 'Original')}
- **Crop**: {preprocessing_params.get('crop_box', 'None')}

💾 Processed image saved to training data!

📈 **View the confidence chart below** to see probabilities for all food classes.
    """
    
    output.object = result_message
    spinner.value = False

# Setup image upload handler with error handling
def handle_image_upload_with_error_handling():
    """Wrapper function to handle image upload with error display."""
    try:
        handle_image_upload()
    except UserFriendlyError as e:
        handle_and_display_error(e, "image upload", handle_image_upload_with_error_handling)
    except Exception as e:
        handle_and_display_error(e, "image upload", handle_image_upload_with_error_handling)

image_input.param.watch(lambda event: handle_image_upload_with_error_handling(), 'value')

# Setup classification handler with error handling
def classify_with_error_handling(event):
    """Wrapper function to handle classification with error display."""
    try:
        classify(event)
    except UserFriendlyError as e:
        handle_and_display_error(e, "classification", classify_with_error_handling)
        spinner.value = False
    except Exception as e:
        handle_and_display_error(e, "classification", classify_with_error_handling)
        spinner.value = False
        confidence_chart.reset()

run_button = pn.widgets.Button(name='Classify', button_type='primary')
run_button.on_click(classify_with_error_handling)

# Create PWA status indicator
pwa_status = pn.pane.HTML("""
<div id="pwa-status" style="position: fixed; top: 10px; right: 10px; z-index: 1000;">
    <span id="connection-status" class="online-indicator">🌐 Online</span>
    <button id="pwa-install-button" style="display: none; margin-left: 10px;" class="pwa-install-btn">
        📱 Install App
    </button>
</div>
""")

# Create layout with preprocessing controls and PWA features
upload_section = pn.Column(
    "## 📤 Upload Image",
    image_input,
    pn.layout.Divider(),
)

preview_section = pn.Column(
    "## 🖼️ Image Preview",
    pn.Row(
        pn.Column("### Original", image_preview),
        pn.Column("### Processed", processed_preview),
    ),
)

controls_section = pn.Column(
    "## 🎨 Preprocessing Controls",
    preprocessing_panel,
)

classification_section = pn.Column(
    "## 🍽️ Classification",
    pn.Row(run_button, spinner),
    output,
)

confidence_section = pn.Column(
    "## 📈 Confidence Analysis",
    confidence_chart_component,
)

app = pn.Row(
    pn.Column(
        upload_section,
        preview_section,
        error_banner,  # Add error banner to the layout
        classification_section,
        confidence_section,
        sizing_mode='stretch_width',
        max_width=800,
    ),
    controls_section,
    sizing_mode='stretch_width',
)

app = pn.Row(
    pn.Column(
        pwa_status,
        upload_section,
        preview_section,
        classification_section,
        sizing_mode='stretch_width',
        max_width=600,
    ),
    controls_section,
    sizing_mode='stretch_width',
)

# Add PWA JavaScript to the app
app = pn.Column(
    pn.pane.HTML(pwa_js_template),
    app
)

app.servable()

# Cleanup PWA manager on exit
import atexit
atexit.register(lambda: pwa_manager.close() if pwa_manager else None)
