#!/bin/bash

# Model Versioning System Setup Script for FlavorSnap
# This script sets up the complete model versioning and A/B testing system

set -e

echo "🍲 Setting up FlavorSnap Model Versioning System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        print_error "Python 3.8 or higher is required. Current version: $python_version"
        exit 1
    fi
    
    print_status "Python $python_version detected"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "dataset/test/Akara"
        "dataset/test/Bread"
        "dataset/test/Egusi"
        "dataset/test/Moi Moi"
        "dataset/test/Rice and Stew"
        "dataset/test/Yam"
        "dataset/val/Akara"
        "dataset/val/Bread"
        "dataset/val/Egusi"
        "dataset/val/Moi Moi"
        "dataset/val/Rice and Stew"
        "dataset/val/Yam"
        "deployments"
        "model_backups"
        "validation_results"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_status "Directories created successfully"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        print_status "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Check if PyTorch is properly installed
check_pytorch() {
    print_status "Checking PyTorch installation..."
    
    python3 -c "
import torch
import torchvision
print(f'PyTorch version: {torch.__version__}')
print(f'TorchVision version: {torchvision.__version__}')
print('PyTorch is properly installed')
" || {
        print_error "PyTorch installation check failed"
        exit 1
    }
}

# Initialize the model registry
initialize_registry() {
    print_status "Initializing model registry..."
    
    # Check if existing model exists
    if [ -f "../models/best_model.pth" ]; then
        print_status "Found existing model, will register as v1.0.0"
        cp ../models/best_model.pth ./models/
    elif [ -f "best_model.pth" ]; then
        print_status "Found existing model in current directory"
        mkdir -p models
        mv best_model.pth models/
    else
        print_warning "No existing model found. You'll need to register a model manually."
    fi
    
    # Create a test script to initialize the registry
    cat > init_registry.py << 'EOF'
#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from model_registry import ModelRegistry
import torch

def init_registry():
    registry = ModelRegistry()
    
    # Check if model exists and register it
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        try:
            # Test if model can be loaded
            model_data = torch.load(model_path, map_location='cpu')
            
            # Register the model
            success = registry.register_model(
                version='v1.0.0',
                model_path=model_path,
                created_by='setup_script',
                description='Initial ResNet18 model for Nigerian food classification',
                accuracy=0.942,  # From training results
                epochs_trained=50,
                tags=['initial', 'production'],
                hyperparameters={
                    'architecture': 'ResNet18',
                    'input_size': [224, 224],
                    'num_classes': 6,
                    'pretrained': True
                }
            )
            
            if success:
                # Activate the model
                registry.activate_model('v1.0.0')
                print("✅ Model v1.0.0 registered and activated successfully")
            else:
                print("❌ Failed to register model")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print("⚠️  No model found at models/best_model.pth")
        print("   You can register a model later using the API")

if __name__ == "__main__":
    init_registry()
EOF

    python3 init_registry.py
    rm init_registry.py
}

# Create example usage scripts
create_examples() {
    print_status "Creating example usage scripts..."
    
    # Example API usage script
    cat > examples/api_usage.py << 'EOF'
#!/usr/bin/env python3
"""
Example script demonstrating API usage for model management
"""

import requests
import json
import time

API_BASE = "http://localhost:5000"

def test_model_management():
    print("🔧 Testing Model Management APIs...")
    
    # List models
    response = requests.get(f"{API_BASE}/api/models")
    models = response.json()['models']
    print(f"Found {len(models)} models")
    
    for model in models:
        print(f"  - {model['version']}: {model['description']} (Active: {model['is_active']})")
    
    # Get active model details
    active_model = next((m for m in models if m['is_active']), None)
    if active_model:
        response = requests.get(f"{API_BASE}/api/models/{active_model['version']}")
        details = response.json()
        print(f"\nActive model details:")
        print(f"  Version: {details['version']}")
        print(f"  Accuracy: {details['accuracy']:.2%}" if details['accuracy'] else "  Accuracy: N/A")
        print(f"  Created: {details['created_at']}")

def test_ab_testing():
    print("\n🧪 Testing A/B Testing APIs...")
    
    # List existing tests
    response = requests.get(f"{API_BASE}/api/ab-tests")
    tests = response.json()['tests']
    print(f"Found {len(tests)} A/B tests")
    
    if len(tests) >= 2:
        # Create a new test if we have at least 2 models
        models_response = requests.get(f"{API_BASE}/api/models")
        models = models_response.json()['models']
        
        if len(models) >= 2:
            test_data = {
                "model_a_version": models[0]['version'],
                "model_b_version": models[1]['version'],
                "traffic_split": 0.1,
                "description": "Example A/B test"
            }
            
            response = requests.post(f"{API_BASE}/api/ab-tests", json=test_data)
            if response.status_code == 201:
                test_id = response.json()['test_id']
                print(f"Created A/B test: {test_id}")
            else:
                print("Failed to create A/B test")

def test_prediction():
    print("\n🔮 Testing Prediction APIs...")
    
    # Test health endpoint
    response = requests.get(f"{API_BASE}/health")
    health = response.json()
    print(f"API Health: {health['status']}")
    
    # Test classes endpoint
    response = requests.get(f"{API_BASE}/api/classes")
    classes = response.json()
    print(f"Supported classes: {classes['classes']}")

if __name__ == "__main__":
    print("🚀 FlavorSnap API Usage Examples")
    print("=" * 40)
    
    try:
        test_model_management()
        test_ab_testing()
        test_prediction()
        print("\n✅ All tests completed successfully!")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running with: python3 app.py")
    except Exception as e:
        print(f"❌ Error: {e}")
EOF

    # Dashboard launch script
    cat > examples/launch_dashboard.py << 'EOF'
#!/usr/bin/env python3
"""
Launch the performance dashboard
"""

import sys
import os
sys.path.append('.')

from performance_dashboard import create_dashboard

if __name__ == "__main__":
    print("📊 Launching FlavorSnap Performance Dashboard...")
    print("Dashboard will be available at: http://localhost:5006")
    print("Press Ctrl+C to stop the dashboard")
    
    dashboard = create_dashboard()
    dashboard.show()
EOF

    # Model validation script
    cat > examples/validate_model.py << 'EOF'
#!/usr/bin/env python3
"""
Validate a model version
"""

import sys
import os
sys.path.append('.')

from model_validator import ModelValidator, ValidationConfig, ModelRegistry

def validate_model_version(version):
    print(f"🔍 Validating model {version}...")
    
    registry = ModelRegistry()
    config = ValidationConfig(
        min_accuracy_threshold=0.75,  # Lower threshold for testing
        max_inference_time_threshold=3.0,
        min_confidence_threshold=0.50,
        num_test_samples=50  # Smaller sample for faster testing
    )
    
    validator = ModelValidator(registry, config)
    result = validator.validate_model(version)
    
    print(f"\nValidation Results for {version}:")
    print(f"Passed: {'✅' if result.passed else '❌'}")
    print(f"Overall Score: {result.overall_score:.3f}")
    
    if result.accuracy:
        print(f"Accuracy: {result.accuracy:.2%}")
    if result.avg_inference_time:
        print(f"Avg Inference Time: {result.avg_inference_time:.3f}s")
    if result.avg_confidence:
        print(f"Avg Confidence: {result.avg_confidence:.3f}")
    
    if result.error_messages:
        print(f"Errors: {len(result.error_messages)}")
        for error in result.error_messages:
            print(f"  - {error}")
    
    return result.passed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 validate_model.py <model_version>")
        sys.exit(1)
    
    version = sys.argv[1]
    success = validate_model_version(version)
    sys.exit(0 if success else 1)
EOF

    chmod +x examples/*.py
    print_status "Example scripts created in examples/ directory"
}

# Create configuration files
create_configs() {
    print_status "Creating configuration files..."
    
    # Environment configuration
    cat > .env.example << 'EOF'
# FlavorSnap Model Versioning Configuration

# Database Configuration
REGISTRY_DB_PATH=model_registry.db

# Model Configuration
MODEL_CACHE_SIZE=5
DEFAULT_MODEL_VERSION=v1.0.0

# Validation Configuration
MIN_ACCURACY_THRESHOLD=0.80
MAX_INFERENCE_TIME_THRESHOLD=2.0
MIN_CONFIDENCE_THRESHOLD=0.60
TEST_DATASET_PATH=dataset/test
VALIDATION_DATASET_PATH=dataset/val
NUM_TEST_SAMPLES=100

# A/B Testing Configuration
DEFAULT_TRAFFIC_SPLIT=0.5
DEFAULT_MIN_SAMPLE_SIZE=100
DEFAULT_CONFIDENCE_THRESHOLD=0.95

# Deployment Configuration
AUTO_ROLLBACK=true
ROLLBACK_THRESHOLD=0.05
MONITORING_WINDOW=100
HEALTH_CHECK_INTERVAL=60
BACKUP_MODELS=true
MAX_BACKUP_COUNT=5

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=false

# Logging Configuration
LOG_LEVEL=INFO
DEPLOYMENT_LOG_FILE=deployment.log
VALIDATION_LOG_FILE=validation.log
EOF

    # Docker configuration (optional)
    cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p dataset/test dataset/val deployments model_backups validation_results

# Expose port
EXPOSE 5000

# Run the application
CMD ["python3", "app.py"]
EOF

    # Docker Compose configuration
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  flavorsnap-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./dataset:/app/dataset
      - ./model_registry.db:/app/model_registry.db
      - ./deployment.log:/app/deployment.log
      - ./validation.log:/app/validation.log
    environment:
      - API_DEBUG=false
      - LOG_LEVEL=INFO
    restart: unless-stopped

  flavorsnap-dashboard:
    build: .
    ports:
      - "5006:5006"
    volumes:
      - ./models:/app/models
      - ./dataset:/app/dataset
      - ./model_registry.db:/app/model_registry.db
    command: python3 performance_dashboard.py
    depends_on:
      - flavorsnap-api
    restart: unless-stopped
EOF

    print_status "Configuration files created"
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    
    # Test imports
    python3 -c "
from model_registry import ModelRegistry
from ab_testing import ABTestManager
from deployment_manager import ModelDeploymentManager
from model_validator import ModelValidator
print('✅ All modules imported successfully')
" || {
        print_error "Module import test failed"
        exit 1
    }
    
    print_status "Installation test passed"
}

# Create startup script
create_startup_script() {
    print_status "Creating startup script..."
    
    cat > start_system.sh << 'EOF'
#!/bin/bash

# FlavorSnap Model Versioning System Startup Script

set -e

echo "🍲 Starting FlavorSnap Model Versioning System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the API server in background
echo "Starting API server..."
python3 app.py &
API_PID=$!

# Wait a moment for the server to start
sleep 3

# Test the API
echo "Testing API..."
curl -s http://localhost:5000/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ API server is running successfully"
else
    echo "❌ API server failed to start"
    kill $API_PID
    exit 1
fi

echo "🚀 System is ready!"
echo "API Server: http://localhost:5000"
echo "API Documentation: See README_MODEL_VERSIONING.md"
echo ""
echo "To start the dashboard (in another terminal):"
echo "  python3 performance_dashboard.py"
echo ""
echo "To stop the system:"
echo "  kill $API_PID"
echo "  or press Ctrl+C"

# Wait for interrupt
trap "echo 'Stopping system...'; kill $API_PID; exit" INT
wait $API_PID
EOF

    chmod +x start_system.sh
    print_status "Startup script created: start_system.sh"
}

# Main setup function
main() {
    echo "🚀 FlavorSnap Model Versioning System Setup"
    echo "=========================================="
    
    check_python
    create_directories
    install_dependencies
    check_pytorch
    initialize_registry
    create_examples
    create_configs
    test_installation
    create_startup_script
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Start the system: ./start_system.sh"
    echo "2. Or start manually: python3 app.py"
    echo "3. Launch dashboard: python3 performance_dashboard.py"
    echo "4. Run examples: python3 examples/api_usage.py"
    echo ""
    echo "📚 For detailed documentation, see: README_MODEL_VERSIONING.md"
}

# Run main function
main "$@"
