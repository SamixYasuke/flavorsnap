import panel as pn
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import os
import sys

# Add config directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from config_manager import get_config, is_feature_enabled, get_env_var

pn.extension()

# Load configuration
model_path = get_config('ml.model_path', 'models/best_model.pth')
classes_path = get_config('ml.classes_path', 'food_classes.txt')
confidence_threshold = get_config('ml.confidence_threshold', 0.7)
upload_dir = get_config('upload.upload_dir', 'uploads')
max_file_size = get_config('upload.max_file_size', 10485760)
allowed_types = get_config('upload.allowed_types', ['jpg', 'jpeg', 'png', 'gif'])

# Load class names
try:
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    class_names = ['Akara', 'Bread', 'Egusi', 'Moi Moi', 'Rice and Stew', 'Yam']

# Load model if feature is enabled
if is_feature_enabled('ml_classification'):
    try:
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Classification will not work.")
        model = None
else:
    print("ML classification feature is disabled")
    model = None

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Save image to correct folder
def save_image(image_obj, predicted_class, image_name="uploaded_image.jpg"):
    save_dir = os.path.join(upload_dir, predicted_class)
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, image_name)
    image_obj.save(image_path)
    return image_path

# Validate file type and size
def validate_file(file_data):
    if len(file_data) > max_file_size:
        return False, f"File size exceeds limit of {max_file_size // (1024*1024)}MB"
    
    # Check file type by examining the first few bytes
    image_formats = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF'
    }
    
    header = file_data[:8]
    for signature, format_name in image_formats.items():
        if header.startswith(signature):
            return True, f"Valid {format_name} file"
    
    return False, "Invalid file format. Only JPEG, PNG, and GIF are allowed"

# Panel UI
image_input = pn.widgets.FileInput(accept='image/*')
output = pn.pane.Markdown("Upload an image of food 🍲")
image_preview = pn.pane.Image(width=300, height=300, visible=False)
spinner = pn.indicators.LoadingSpinner(value=False, width=50)

def classify(event=None):
    if image_input.value is None:
        output.object = "⚠️ Please upload an image first."
        image_preview.visible = False
        return
    
    if not is_feature_enabled('ml_classification'):
        output.object = "❌ ML classification feature is disabled."
        image_preview.visible = False
        return
    
    if model is None:
        output.object = "❌ Model not available. Please check configuration."
        image_preview.visible = False
        return
    
    try:
        # Validate file
        is_valid, validation_message = validate_file(image_input.value)
        if not is_valid:
            output.object = f"❌ {validation_message}"
            image_preview.visible = False
            return
        
        image = Image.open(io.BytesIO(image_input.value)).convert('RGB')

        # Update preview
        image_preview.object = image
        image_preview.visible = True

        # Start spinner
        spinner.value = True
        output.object = "🔍 Classifying..."

        # Transform and predict
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, pred = torch.max(probabilities, 0)
            
            if confidence.item() < confidence_threshold:
                output.object = f"⚠️ Low confidence ({confidence.item():.2%}). Cannot classify with certainty."
                return
            
            predicted_class = class_names[pred.item()]

        # Save image
        saved_path = save_image(image, predicted_class)
        output.object = f"✅ Identified as **{predicted_class}** (confidence: {confidence.item():.2%}). Image saved to {saved_path}"
        
    except Exception as e:
        output.object = f"❌ Error: {str(e)}"
    finally:
        spinner.value = False

run_button = pn.widgets.Button(name='Classify', button_type='primary')
run_button.on_click(classify)

app = pn.Column(
    "# 🍽️ FlavorSnap",
    "Upload an image and click the button to classify your food!",
    image_input,
    run_button,
    spinner,
    image_preview,
    output,
)

app.servable()
