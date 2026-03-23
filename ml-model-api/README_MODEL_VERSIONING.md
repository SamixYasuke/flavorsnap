# Model Versioning and A/B Testing System for FlavorSnap

This system provides comprehensive model versioning, A/B testing, performance monitoring, and automated deployment capabilities for the FlavorSnap food classification application.

## 🚀 Features

### Model Registry
- **Version Management**: Track multiple model versions with metadata
- **Model Validation**: Automated integrity and performance validation
- **Metadata Storage**: Store training metrics, hyperparameters, and model hashes
- **Active Model Management**: Easy activation and deactivation of model versions

### A/B Testing Framework
- **Traffic Splitting**: Configurable traffic distribution between model versions
- **Statistical Analysis**: Automated significance testing with configurable thresholds
- **Real-time Metrics**: Track accuracy, confidence, and processing time
- **Test Management**: Create, monitor, and conclude A/B tests

### Performance Dashboard
- **Interactive Visualizations**: Real-time charts and metrics comparison
- **Model Comparison**: Side-by-side performance analysis
- **Live Monitoring**: Real-time prediction metrics and health status
- **Historical Trends**: Track performance over time

### Deployment Management
- **Safe Deployments**: Automated rollback on performance degradation
- **Health Monitoring**: Continuous health checks with automatic alerts
- **Backup System**: Automatic model backups with version history
- **Deployment Tracking**: Complete audit trail of all deployment activities

### Automated Validation
- **Model Integrity**: File structure and architecture validation
- **Performance Testing**: Comprehensive accuracy and speed benchmarks
- **Regression Detection**: Automatic detection of performance regressions
- **Visual Validation**: Confusion matrices and performance visualizations

## 📁 Project Structure

```
ml-model-api/
├── app.py                    # Main Flask application
├── model_registry.py         # Model registry and version management
├── ab_testing.py            # A/B testing framework
├── deployment_manager.py    # Deployment and rollback system
├── model_validator.py       # Automated model validation
├── performance_dashboard.py # Performance monitoring dashboard
├── api_endpoints.py         # REST API endpoints
├── requirements.txt         # Python dependencies
├── model_registry.db       # SQLite database (created automatically)
├── deployment.log          # Deployment logs
├── validation.log          # Validation logs
├── deployments/            # Deployment artifacts
├── model_backups/          # Model backups
├── validation_results/     # Validation results and plots
└── dataset/
    ├── test/              # Test dataset for validation
    └── val/               # Validation dataset
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- 4GB+ RAM

### Setup

1. **Install Dependencies**
```bash
cd ml-model-api
pip install -r requirements.txt
```

2. **Prepare Dataset** (Optional, for validation)
```bash
# Create test dataset structure
mkdir -p dataset/test/{Akara,Bread,Egusi,"Moi Moi","Rice and Stew",Yam}
# Add test images to respective class folders
```

3. **Initialize the System**
```bash
python app.py
```

The system will automatically:
- Create the SQLite database
- Register the existing model as v1.0.0
- Start the API server on port 5000

## 📚 API Documentation

### Model Management

#### List Models
```http
GET /api/models
GET /api/models?active_only=true
```

#### Get Model Details
```http
GET /api/models/{version}
```

#### Register New Model
```http
POST /api/models/register
Content-Type: application/json

{
  "version": "v1.1.0",
  "model_path": "models/new_model.pth",
  "created_by": "data_scientist",
  "description": "Improved model with better accuracy",
  "accuracy": 0.95,
  "epochs_trained": 75,
  "tags": ["production", "high-accuracy"]
}
```

#### Activate Model
```http
POST /api/models/{version}/activate
```

#### Validate Model
```http
POST /api/models/{version}/validate
```

### A/B Testing

#### Create A/B Test
```http
POST /api/ab-tests
Content-Type: application/json

{
  "model_a_version": "v1.0.0",
  "model_b_version": "v1.1.0",
  "traffic_split": 0.3,
  "description": "Testing new model improvements",
  "min_sample_size": 200,
  "confidence_threshold": 0.95
}
```

#### List A/B Tests
```http
GET /api/ab-tests
GET /api/ab-tests?status=active
```

#### Get Test Results
```http
GET /api/ab-tests/{test_id}
```

#### End A/B Test
```http
POST /api/ab-tests/{test_id}/end
Content-Type: application/json

{
  "winner": "v1.1.0"
}
```

### Deployment Management

#### Deploy Model
```http
POST /api/models/{version}/deploy
Content-Type: application/json

{
  "force": false
}
```

#### Rollback Model
```http
POST /api/deployment/rollback
Content-Type: application/json

{
  "target_version": "v1.0.0",
  "reason": "Performance degradation detected"
}
```

#### Health Check
```http
GET /api/deployment/health
GET /api/deployment/health?model_version=v1.1.0
```

#### Deployment History
```http
GET /api/deployment/history?limit=50
```

### Prediction with Version Selection

#### Standard Prediction (uses active model or A/B test)
```http
POST /predict
Content-Type: multipart/form-data

image: [file]
```

#### Specific Model Version
```http
POST /predict
Content-Type: multipart/form-data

image: [file]
model_version: v1.1.0
```

#### A/B Test Participation
```http
POST /predict
Content-Type: multipart/form-data

image: [file]
test_id: abc-123-def
user_id: user456
```

## 🎯 Usage Examples

### 1. Register and Deploy a New Model

```python
import requests

# Register new model
response = requests.post('http://localhost:5000/api/models/register', json={
    'version': 'v1.2.0',
    'model_path': 'models/improved_model.pth',
    'created_by': 'ml_engineer',
    'description': 'Model with 2% accuracy improvement',
    'accuracy': 0.962,
    'epochs_trained': 100
})

# Validate the model
response = requests.post('http://localhost:5000/api/models/v1.2.0/validate')
validation_result = response.json()

if validation_result['passed']:
    # Deploy the model
    response = requests.post('http://localhost:5000/api/models/v1.2.0/deploy')
    print(f"Deployment: {response.json()}")
else:
    print(f"Validation failed: {validation_result['error_messages']}")
```

### 2. Create and Monitor A/B Test

```python
# Create A/B test
response = requests.post('http://localhost:5000/api/ab-tests', json={
    'model_a_version': 'v1.0.0',
    'model_b_version': 'v1.2.0',
    'traffic_split': 0.2,  # 20% traffic to new model
    'description': 'Testing v1.2.0 in production'
})

test_id = response.json()['test_id']

# Monitor test results
import time
while True:
    response = requests.get(f'http://localhost:5000/api/ab-tests/{test_id}')
    results = response.json()
    
    metrics_a = results['model_a_metrics']
    metrics_b = results['model_b_metrics']
    
    print(f"Model A Accuracy: {metrics_a['accuracy']:.3f}")
    print(f"Model B Accuracy: {metrics_b['accuracy']:.3f}")
    print(f"Significance: {results['statistical_significance']['significant']}")
    
    time.sleep(60)  # Check every minute
```

### 3. Launch Performance Dashboard

```python
# Start the dashboard (run in separate terminal)
python performance_dashboard.py

# Or access via API
response = requests.get('http://localhost:5000/api/models')
models = response.json()['models']
print(f"Total models: {len(models)}")
```

## 🔧 Configuration

### Validation Configuration
```python
validation_config = ValidationConfig(
    min_accuracy_threshold=0.80,      # Minimum accuracy to pass validation
    max_inference_time_threshold=2.0,  # Maximum inference time (seconds)
    min_confidence_threshold=0.60,     # Minimum confidence threshold
    test_dataset_path="dataset/test",  # Test dataset path
    num_test_samples=100,              # Number of samples to test
    enable_visual_validation=True,     # Generate confusion matrices
    check_performance_regression=True, # Check for performance regression
    baseline_comparison_version="v1.0.0"  # Baseline for regression checks
)
```

### Deployment Configuration
```python
deployment_config = DeploymentConfig(
    auto_rollback=True,           # Enable automatic rollback
    rollback_threshold=0.05,       # 5% performance drop triggers rollback
    monitoring_window=100,         # Number of predictions to monitor
    health_check_interval=60,      # Health check interval (seconds)
    backup_models=True,            # Create model backups
    max_backup_count=5            # Maximum number of backups to keep
)
```

## 📊 Monitoring and Alerts

### Health Metrics
- **Response Time**: Average inference time
- **Error Rate**: Percentage of failed predictions
- **Health Score**: Composite health metric (0-1)
- **Request Volume**: Number of predictions processed

### Performance Metrics
- **Accuracy**: Model prediction accuracy
- **Confidence**: Average prediction confidence
- **Processing Time**: Inference latency
- **Error Count**: Number of processing errors

### A/B Test Metrics
- **Statistical Significance**: P-value and confidence levels
- **Performance Difference**: Delta between model versions
- **Sample Size**: Number of predictions per model
- **Winner Determination**: Automated winner selection

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```bash
   # Check model file integrity
   python -c "import torch; print(torch.load('models/model.pth').keys())"
   ```

2. **Database Issues**
   ```bash
   # Recreate database
   rm model_registry.db
   python app.py
   ```

3. **Validation Fails**
   ```bash
   # Check test dataset
   ls -la dataset/test/
   # Ensure each class folder has images
   ```

4. **A/B Test Not Working**
   ```bash
   # Check active tests
   curl http://localhost:5000/api/ab-tests?status=active
   ```

### Logs
- **Deployment Logs**: `deployment.log`
- **Validation Logs**: `validation.log`
- **Flask Logs**: Console output

## 🔄 Integration with Existing Systems

### Frontend Integration
```javascript
// Make prediction with model version
const formData = new FormData();
formData.append('image', imageFile);
formData.append('model_version', 'v1.2.0');

const response = await fetch('/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Prediction: ${result.label} (Model: ${result.model_version})`);
```

### CI/CD Pipeline Integration
```bash
#!/bin/bash
# validate_and_deploy.sh

# 1. Validate new model
python -c "
from model_validator import ModelValidator, ModelRegistry
registry = ModelRegistry()
validator = ModelValidator(registry)
result = validator.validate_model('$MODEL_VERSION')
exit(0 if result.passed else 1)
"

# 2. Deploy if validation passes
if [ $? -eq 0 ]; then
  curl -X POST "http://localhost:5000/api/models/$MODEL_VERSION/deploy"
  echo "Model $MODEL_VERSION deployed successfully"
else
  echo "Model validation failed"
  exit 1
fi
```

## 📈 Best Practices

1. **Model Versioning**
   - Use semantic versioning (v1.0.0, v1.1.0, v2.0.0)
   - Include meaningful descriptions
   - Tag models with deployment stage

2. **A/B Testing**
   - Start with small traffic splits (5-10%)
   - Monitor for statistical significance
   - Set minimum sample sizes

3. **Deployment**
   - Always validate before deploying
   - Use automated rollback for safety
   - Monitor health metrics continuously

4. **Validation**
   - Use representative test datasets
   - Check for performance regressions
   - Generate visual validation reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
