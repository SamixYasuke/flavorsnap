import panel as pn
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.ui.preprocessing_controls import PreprocessingControls
from src.core.image_enhancer import ImageEnhancer

pn.extension('css')

# Load model
model_path = 'models/best_model.pth'
class_names = ['Akara', 'Bread', 'Egusi', 'Moi Moi', 'Rice and Stew', 'Yam']
os.makedirs('models', exist_ok=True)
if os.path.exists(model_path):
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
else:
    model = None

# Save image function
def save_image(image_obj, predicted_class, image_name="uploaded_image.jpg"):
    save_dir = f"data/train/{predicted_class}"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, image_name)
    image_obj.save(image_path)
    return image_path

<<<<<<< HEAD
# Panel UI
image_input = pn.widgets.FileInput(accept='image/*')
output = pn.pane.Markdown("Upload an image of food 🍲")
image_preview = pn.pane.Image(width=300, height=300, visible=False)
processed_preview = pn.pane.Image(width=300, height=300, visible=False)
spinner = pn.indicators.LoadingSpinner(value=False, width=50)

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

def handle_image_upload():
    """Handle image upload and initialize preprocessing."""
    global original_image, processed_image
    
    if image_input.value is None:
        return
    
    try:
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
        
    except Exception as e:
        output.object = f"❌ Error loading image: {str(e)}"

def classify(event=None):
    global original_image, processed_image
    
    if image_input.value is None:
        output.object = "⚠️ Please upload an image first."
        image_preview.visible = False
        processed_preview.visible = False
        return
    
    if processed_image is None:
        output.object = "⚠️ Please wait for image to load or apply preprocessing."
        return
    
    try:
=======
# State variables
current_image = None
current_predicted_class = None

# Interface Setup
def classify(event=None):
    global current_image, current_predicted_class
    if ui.image_input.value is None and current_image is None:
        ui.output.object = "⚠️ Please upload an image first."
        ui.image_preview.visible = False
        return
        
    try:
        if ui.image_input.value is not None:
             image = Image.open(io.BytesIO(ui.image_input.value)).convert('RGB')
             current_image = image
        else:
             image = current_image

        # Update preview
        ui.image_preview.object = image
        ui.image_preview.visible = True

>>>>>>> fork/Real-time-image-preprocessing-controls
        # Start spinner
        ui.spinner.value = True
        ui.output.object = "🔍 Classifying..."

<<<<<<< HEAD
        # Use processed image for classification
        image_to_classify = processed_image

        # Transform and predict
        img_tensor = transform(image_to_classify).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

        # Save processed image
        save_image(image_to_classify, predicted_class)
        
        # Get preprocessing parameters
        params = preprocessing_controls.get_enhancement_params()
        
        # Create result message with preprocessing info
        result_message = f"""
✅ **Classification Result: {predicted_class}**

### 📊 Preprocessing Parameters Applied:
- **Brightness**: {params['brightness']:.1f}
- **Contrast**: {params['contrast']:.1f}
- **Rotation**: {params['rotation']:.0f}°
- **Aspect Ratio**: {params.get('aspect_ratio', 'Original')}
- **Crop**: {params.get('crop_box', 'None')}

💾 Processed image saved to training data!
        """
        
        output.object = result_message
        
=======
        if model is None:
            ui.output.object = "❌ Model weights not found. (Dummy run)"
            predicted_class = class_names[0]
        else:
            # Transform and predict
            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = torch.max(outputs, 1)
                predicted_class = class_names[pred.item()]

        current_predicted_class = predicted_class
        
        # Save image
        saved_path = save_image(image, predicted_class)
        ui.output.object = f"✅ Identified as **{predicted_class}**. Image saved!"
        
        # Add to history
        history_item = f"- Identified **{predicted_class}**"
        current_history = ui.history_panel[1].object
        if current_history == "No history yet.":
            ui.history_panel[1].object = history_item
        else:
            ui.history_panel[1].object = current_history + "\n" + history_item
            
>>>>>>> fork/Real-time-image-preprocessing-controls
    except Exception as e:
        ui.output.object = f"❌ Error: {str(e)}"
    finally:
        ui.spinner.value = False

def manual_export():
    if current_image and current_predicted_class:
        save_image(current_image, current_predicted_class, image_name="manual_export.jpg")
        ui.output.object = f"💾 Manually exported results for **{current_predicted_class}**"

<<<<<<< HEAD
# Setup image upload handler
image_input.param.watch(lambda event: handle_image_upload(), 'value')

# Create layout with preprocessing controls
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
=======
def handle_shortcut(combo):
    global current_image, current_predicted_class
    if combo == 'enter':
        classify()
    elif combo == 'escape':
        ui.clear_image()
        current_image = None
        current_predicted_class = None
    elif combo == 'ctrl+s':
        manual_export()
    elif combo == 'ctrl+h':
        ui.toggle_history()
    elif combo == 'ctrl+d':
        # Toggle dark mode
        if 'dark-theme' in app.css_classes:
            app.css_classes = [c for c in app.css_classes if c != 'dark-theme']
            try:
                pn.config.theme = 'default'
            except:
                pass
        else:
            app.css_classes = app.css_classes + ['dark-theme']
            try:
                pn.config.theme = 'dark'
            except:
                pass


ui = MainInterface(classify_fn=classify, save_image_fn=manual_export)
keyboard_manager = KeyboardManager(handle_shortcut)

# Header
header = pn.Row(
    pn.pane.Markdown("# 🍽️ FlavorSnap", styles={'margin-top': '0px', 'flex': '1'}),
    theme_toggle,
    sizing_mode='stretch_width',
    css_classes=['header']
)

# Dashboard Layout
app = pn.Column(
    shortcut_js,
    keyboard_manager.get_widget(),
    ui.get_layout(),
    css_classes=[]
)

# App Assembly
app = pn.Column(
    header,
    pn.layout.Divider(),
    dashboard_body,
    sizing_mode='stretch_width'
>>>>>>> fork/Real-time-image-preprocessing-controls
)

app = pn.Row(
    pn.Column(
        upload_section,
        preview_section,
        classification_section,
        sizing_mode='stretch_width',
        max_width=600,
    ),
    controls_section,
    sizing_mode='stretch_width',
)

app.servable()
