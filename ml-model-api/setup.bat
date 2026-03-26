@echo off
REM Model Versioning System Setup Script for FlavorSnap (Windows)
REM This script sets up the complete model versioning and A/B testing system

echo 🍲 Setting up FlavorSnap Model Versioning System...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "dataset\test\Akara" mkdir "dataset\test\Akara"
if not exist "dataset\test\Bread" mkdir "dataset\test\Bread"
if not exist "dataset\test\Egusi" mkdir "dataset\test\Egusi"
if not exist "dataset\test\Moi Moi" mkdir "dataset\test\Moi Moi"
if not exist "dataset\test\Rice and Stew" mkdir "dataset\test\Rice and Stew"
if not exist "dataset\test\Yam" mkdir "dataset\test\Yam"

if not exist "dataset\val\Akara" mkdir "dataset\val\Akara"
if not exist "dataset\val\Bread" mkdir "dataset\val\Bread"
if not exist "dataset\val\Egusi" mkdir "dataset\val\Egusi"
if not exist "dataset\val\Moi Moi" mkdir "dataset\val\Moi Moi"
if not exist "dataset\val\Rice and Stew" mkdir "dataset\val\Rice and Stew"
if not exist "dataset\val\Yam" mkdir "dataset\val\Yam"

if not exist "deployments" mkdir "deployments"
if not exist "model_backups" mkdir "model_backups"
if not exist "validation_results" mkdir "validation_results"
if not exist "logs" mkdir "logs"
if not exist "examples" mkdir "examples"

echo ✅ Directories created successfully

REM Install Python dependencies
echo 📦 Installing Python dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed successfully
) else (
    echo ❌ requirements.txt not found
    pause
    exit /b 1
)

REM Check if PyTorch is properly installed
echo 🔍 Checking PyTorch installation...
python -c "import torch; import torchvision; print(f'PyTorch version: {torch.__version__}'); print('PyTorch is properly installed')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ PyTorch installation check failed
    pause
    exit /b 1
)

echo ✅ PyTorch is properly installed

REM Copy existing model if available
echo 📋 Checking for existing model...
if exist "..\models\best_model.pth" (
    echo 📁 Found existing model, copying to models directory...
    if not exist "models" mkdir "models"
    copy "..\models\best_model.pth" "models\best_model.pth"
    echo ✅ Model copied successfully
) else if exist "best_model.pth" (
    echo 📁 Found existing model in current directory...
    if not exist "models" mkdir "models"
    move "best_model.pth" "models\best_model.pth"
    echo ✅ Model moved successfully
) else (
    echo ⚠️  No existing model found. You'll need to register a model manually.
)

REM Create example scripts
echo 📝 Creating example usage scripts...

REM API usage example
echo import requests > examples\api_usage.py
echo import json >> examples\api_usage.py
echo import time >> examples\api_usage.py
echo. >> examples\api_usage.py
echo API_BASE = "http://localhost:5000" >> examples\api_usage.py
echo. >> examples\api_usage.py
echo def test_model_management(): >> examples\api_usage.py
echo     print("🔧 Testing Model Management APIs...") >> examples\api_usage.py
echo     response = requests.get(f"{API_BASE}/api/models") >> examples\api_usage.py
echo     models = response.json()['models'] >> examples\api_usage.py
echo     print(f"Found {len(models)} models") >> examples\api_usage.py
echo     for model in models: >> examples\api_usage.py
echo         print(f"  - {model['version']}: {model['description']} (Active: {model['is_active']})") >> examples\api_usage.py
echo. >> examples\api_usage.py
echo if __name__ == "__main__": >> examples\api_usage.py
echo     print("🚀 FlavorSnap API Usage Examples") >> examples\api_usage.py
echo     print("=" * 40) >> examples\api_usage.py
echo     try: >> examples\api_usage.py
echo         test_model_management() >> examples\api_usage.py
echo         print("\n✅ All tests completed successfully!") >> examples\api_usage.py
echo     except requests.exceptions.ConnectionError: >> examples\api_usage.py
echo         print("❌ Cannot connect to API. Make sure server is running with: python app.py") >> examples\api_usage.py
echo     except Exception as e: >> examples\api_usage.py
echo         print(f"❌ Error: {e}") >> examples\api_usage.py

REM Dashboard launch script
echo import sys >> examples\launch_dashboard.py
echo import os >> examples\launch_dashboard.py
echo sys.path.append('.') >> examples\launch_dashboard.py
echo. >> examples\launch_dashboard.py
echo from performance_dashboard import create_dashboard >> examples\launch_dashboard.py
echo. >> examples\launch_dashboard.py
echo if __name__ == "__main__": >> examples\launch_dashboard.py
echo     print("📊 Launching FlavorSnap Performance Dashboard...") >> examples\launch_dashboard.py
echo     print("Dashboard will be available at: http://localhost:5006") >> examples\launch_dashboard.py
echo     print("Press Ctrl+C to stop dashboard") >> examples\launch_dashboard.py
echo     dashboard = create_dashboard() >> examples\launch_dashboard.py
echo     dashboard.show() >> examples\launch_dashboard.py

echo ✅ Example scripts created in examples\ directory

REM Create environment configuration
echo ⚙️ Creating configuration files...
echo # FlavorSnap Model Versioning Configuration > .env.example
echo. >> .env.example
echo # Database Configuration >> .env.example
echo REGISTRY_DB_PATH=model_registry.db >> .env.example
echo. >> .env.example
echo # API Configuration >> .env.example
echo API_HOST=0.0.0.0 >> .env.example
echo API_PORT=5000 >> .env.example
echo API_DEBUG=false >> .env.example

echo ✅ Configuration files created

REM Test the installation
echo 🧪 Testing installation...
python -c "from model_registry import ModelRegistry; print('✅ ModelRegistry imported successfully')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Module import test failed
    pause
    exit /b 1
)

echo ✅ Installation test passed

REM Create startup script
echo 🚀 Creating startup script...
echo @echo off > start_system.bat
echo echo 🍲 Starting FlavorSnap Model Versioning System... >> start_system.bat
echo. >> start_system.bat
echo echo Starting API server... >> start_system.bat
echo python app.py >> start_system.bat
echo. >> start_system.bat
echo pause >> start_system.bat

echo ✅ Startup script created: start_system.bat

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Start system: start_system.bat
echo 2. Or start manually: python app.py
echo 3. Launch dashboard: python performance_dashboard.py
echo 4. Run examples: python examples\api_usage.py
echo.
echo 📚 For detailed documentation, see: README_MODEL_VERSIONING.md
echo.
echo Press any key to exit...
pause >nul
