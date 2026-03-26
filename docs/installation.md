# 🚀 FlavorSnap Installation Guide

This comprehensive guide will help you install and set up FlavorSnap on any platform. Follow the instructions that match your use case and technical expertise.

## 📋 Table of Contents

- [🎯 Prerequisites](#-prerequisites)
- [💻 Platform-Specific Setup](#-platform-specific-setup)
  - [Windows Setup](#windows-setup)
  - [macOS Setup](#macos-setup)
  - [Linux Setup](#linux-setup)
- [🔧 Installation Methods](#-installation-methods)
  - [Docker Installation (Recommended)](#docker-installation-recommended)
  - [Manual Installation](#manual-installation)
  - [Automated Installation Script](#automated-installation-script)
- [⚙️ Configuration](#️-configuration)
- [🧪 Verification](#-verification)
- [🎮 GPU Setup](#gpu-setup)
- [🛠️ Development Environment](#️-development-environment)
- [📊 Performance Requirements](#-performance-requirements)
- [🔍 Troubleshooting](#-troubleshooting)

## 🎯 Prerequisites

### Minimum System Requirements

- **OS**: Windows 10+, macOS 10.15+, or Ubuntu 18.04+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **Network**: Internet connection for downloads

### Required Software

#### Core Dependencies
- **Node.js** 18.0.0 or higher
- **Python** 3.8 or higher (3.9+ recommended)
- **Git** for version control
- **Docker** & **Docker Compose** (for containerized setup)

#### Optional but Recommended
- **CUDA** 11.0+ (for GPU acceleration)
- **VS Code** or similar code editor
- **PostgreSQL** client tools
- **Redis** client tools

### Hardware Requirements

#### CPU-based Setup
- **CPU**: 2+ cores, x86_64 architecture
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 5GB for application + 50MB for model

#### GPU-accelerated Setup
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: Version 11.0 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: Same as CPU-based

## 💻 Platform-Specific Setup

### Windows Setup

#### 1. Install Chocolatey (Package Manager)

```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

#### 2. Install Dependencies

```powershell
# Install Node.js
choco install nodejs --version=18.17.0

# Install Python
choco install python --version=3.9.7

# Install Git
choco install git

# Install Docker Desktop
choco install docker-desktop

# Restart PowerShell after installation
```

#### 3. Verify Installation

```powershell
node --version  # Should be v18.x.x
python --version  # Should be 3.9.x
git --version
docker --version
```

#### 4. Windows-Specific Configuration

```powershell
# Enable Windows Subsystem for Linux (optional but recommended)
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Restart your computer after running these commands
```

### macOS Setup

#### 1. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install Dependencies

```bash
# Install Node.js
brew install node@18

# Install Python
brew install python@3.9

# Install Git (usually pre-installed)
brew install git

# Install Docker Desktop
brew install --cask docker

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 3. Verify Installation

```bash
node --version  # Should be v18.x.x
python3 --version  # Should be 3.9.x
git --version
docker --version
```

#### 4. macOS-Specific Configuration

```bash
# Install Xcode Command Line Tools (required for some Python packages)
xcode-select --install

# Allow Docker Desktop to run (System Preferences > Security & Privacy)
```

### Linux Setup (Ubuntu/Debian)

#### 1. Update System Packages

```bash
sudo apt update && sudo apt upgrade -y
```

#### 2. Install Dependencies

```bash
# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Python 3.9
sudo apt install python3.9 python3.9-pip python3.9-venv -y

# Install Git
sudo apt install git -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and log back in to apply Docker group changes
```

#### 3. Verify Installation

```bash
node --version  # Should be v18.x.x
python3.9 --version  # Should be 3.9.x
git --version
docker --version
docker-compose --version
```

#### 4. Linux-Specific Configuration

```bash
# Install additional system dependencies
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y

# Create symbolic links for convenience
sudo ln -sf /usr/bin/python3.9 /usr/bin/python
sudo ln -sf /usr/bin/pip3 /usr/bin/pip
```

## 🔧 Installation Methods

### Docker Installation (Recommended)

#### Quick Start

```bash
# Clone the repository
git clone https://github.com/olaleyeolajide81-sketch/flavorsnap.git
cd flavorsnap

# Run the automated Docker setup
./scripts/docker_run.sh -e development -d

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000
```

#### Detailed Docker Setup

1. **Environment Configuration**

```bash
# Copy environment template
cp .env.example .env

# Edit the .env file with your preferences
nano .env
```

2. **Build Containers**

```bash
# Build development containers
./scripts/docker_build.sh -e development

# Or build production containers
./scripts/docker_build.sh -e production
```

3. **Run Services**

```bash
# Development environment
docker-compose up -d

# Production environment
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose ps
```

#### Docker Compose Files Available

- `docker-compose.yml` - Development environment
- `docker-compose.prod.yml` - Production environment
- `docker-compose.test.yml` - Testing environment

### Manual Installation

#### 1. Clone Repository

```bash
git clone https://github.com/olaleyeolajide81-sketch/flavorsnap.git
cd flavorsnap
```

#### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Copy environment configuration
cp .env.example .env.local

# Edit .env.local with your settings
nano .env.local
```

**Example .env.local:**
```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_MODEL_ENDPOINT=/predict

# File Upload Settings
MAX_FILE_SIZE=10485760  # 10MB
ALLOWED_FILE_TYPES=jpg,jpeg,png,webp

# Model Configuration
MODEL_CONFIDENCE_THRESHOLD=0.6
ENABLE_CLASSIFICATION_HISTORY=true

# Feature Flags
ENABLE_ANALYTICS=false
ENABLE_DARK_MODE=true

# Development
NODE_ENV=development
DEBUG=true
```

#### 3. Backend Setup

```bash
# Navigate to backend directory
cd ml-model-api

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional required packages
pip install flask flask-cors pillow numpy requests
```

#### 4. Model Setup

```bash
# Ensure model file exists in project root
ls -la model.pth

# If model file is missing, download it or train a new one
# See: Model Training section in README.md
```

#### 5. Start Services

```bash
# Start backend (in ml-model-api directory)
python app.py

# Start frontend (in new terminal, in frontend directory)
npm run dev
```

### Automated Installation Script

#### Run the Installer

```bash
# Make the script executable (Linux/macOS)
chmod +x scripts/install.py

# Run the installation
python scripts/install.py

# Or with specific options
python scripts/install.py --method docker --environment development
python scripts/install.py --method manual --with-gpu
```

#### Script Options

```bash
python scripts/install.py --help

# Options:
# --method: docker, manual, auto (default: auto)
# --environment: development, production, test (default: development)
# --with-gpu: Enable GPU support
# --skip-deps: Skip dependency installation
# --verbose: Verbose output
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file in project root:

```env
# Application Settings
NODE_ENV=development
DEBUG=true

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_DB=flavorsnap
POSTGRES_USER=flavorsnap
POSTGRES_PASSWORD=your_secure_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Security
JWT_SECRET=your_super_secret_jwt_key_here

# Model Configuration
MODEL_CONFIDENCE_THRESHOLD=0.6
MODEL_BATCH_SIZE=32

# File Upload Settings
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=jpg,jpeg,png,webp

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_MODEL_ENDPOINT=/predict

# Feature Flags
ENABLE_ANALYTICS=true
ENABLE_DARK_MODE=true
ENABLE_CLASSIFICATION_HISTORY=true

# Performance
CACHING_ENABLED=true
CACHE_TTL=3600
```

### YAML Configuration

The project uses `config.yaml` for detailed configuration. See the existing file for all available options.

## 🧪 Verification

### Health Check Script

```bash
# Run the environment validation
python scripts/check_environment.py

# Expected output:
# ✅ Node.js: v18.17.0
# ✅ Python: 3.9.7
# ✅ Docker: 24.0.0
# ✅ Model file: model.pth (44MB)
# ✅ Configuration: Valid
# ✅ All checks passed!
```

### Manual Verification

#### 1. Check Frontend

```bash
# Navigate to frontend
cd frontend

# Run development server
npm run dev

# Open http://localhost:3000
# Should see FlavorSnap landing page
```

#### 2. Check Backend

```bash
# Navigate to backend
cd ml-model-api

# Start Flask server
python app.py

# Test health endpoint
curl http://localhost:5000/health

# Expected response:
# {"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

#### 3. Test Classification

```bash
# Test with sample image
curl -X POST \
  http://localhost:5000/predict \
  -F 'image=@test-food.jpg'

# Expected response:
# {
#   "label": "Moi Moi",
#   "confidence": 85.7,
#   "all_predictions": [...],
#   "processing_time": 0.234
# }
```

## 🎮 GPU Setup

### NVIDIA GPU Setup

#### 1. Install CUDA

**Windows:**
```powershell
# Download and install CUDA Toolkit from NVIDIA
# https://developer.nvidia.com/cuda-downloads
```

**Linux:**
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### 2. Install PyTorch with CUDA

```bash
# Activate virtual environment
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 3. Configure for GPU

Update backend configuration:

```env
# In .env file
USE_GPU=true
CUDA_VISIBLE_DEVICES=0
MODEL_BATCH_SIZE=16  # Adjust based on GPU memory
```

### AMD GPU Setup (ROCm)

```bash
# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

# Verify ROCm availability
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

## 🛠️ Development Environment

### IDE Setup

#### VS Code Extensions

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.flake8",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next",
    "ms-azuretools.vscode-docker"
  ]
}
```

#### Development Tools

```bash
# Install frontend development tools
cd frontend
npm install -g @next/cli
npm install -g typescript

# Install backend development tools
cd ml-model-api
pip install black flake8 pytest pytest-cov

# Install Git hooks (optional)
pip install pre-commit
pre-commit install
```

### Database Setup (Optional)

#### PostgreSQL

```bash
# Using Docker
docker run -d \
  --name postgres-flavorsnap \
  -e POSTGRES_DB=flavorsnap \
  -e POSTGRES_USER=flavorsnap \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:13

# Or install locally
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib

# macOS:
brew install postgresql
brew services start postgresql

# Create database
sudo -u postgres createdb flavorsnap
sudo -u postgres createuser flavorsnap
sudo -u postgres psql -c "ALTER USER flavorsnap PASSWORD 'your_password';"
```

#### Redis

```bash
# Using Docker
docker run -d \
  --name redis-flavorsnap \
  -p 6379:6379 \
  redis:7-alpine

# Or install locally
# Ubuntu/Debian:
sudo apt install redis-server

# macOS:
brew install redis
brew services start redis
```

## 📊 Performance Requirements

### Resource Allocation

#### Development Environment
- **Frontend**: 256MB RAM, 250m CPU
- **Backend**: 1GB RAM, 500m CPU
- **Database**: 512MB RAM, 250m CPU
- **Total**: ~2GB RAM, 1 CPU core

#### Production Environment
- **Frontend**: 512MB RAM, 500m CPU (per instance)
- **Backend**: 2GB RAM, 1000m CPU (per instance)
- **Database**: 1GB RAM, 500m CPU
- **Monitoring**: 512MB RAM, 250m CPU
- **Total**: ~4GB RAM, 2 CPU cores (minimum)

### Performance Optimization

#### 1. Model Optimization

```bash
# Enable model quantization (reduces size, minor accuracy loss)
python -c "
import torch
model = torch.load('model.pth')
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
torch.save(quantized_model, 'model_quantized.pth')
"
```

#### 2. Caching Configuration

```env
# Enable Redis caching
REDIS_ENABLED=true
CACHE_TTL=3600
CACHE_MAX_SIZE=100MB
```

#### 3. Load Balancing

```yaml
# docker-compose.prod.yml
services:
  frontend:
    deploy:
      replicas: 3
  
  backend:
    deploy:
      replicas: 2
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Fails

**Problem**: `FileNotFoundError: model.pth not found`

**Solution**:
```bash
# Check if model exists
ls -la model.pth

# Download model if missing
wget https://example.com/flavorsnap-model.pth -O model.pth

# Verify model integrity
python -c "import torch; model = torch.load('model.pth'); print(f'Model keys: {list(model.keys())}')"
```

#### 2. Port Conflicts

**Problem**: `Port 3000/5000 already in use`

**Solution**:
```bash
# Find process using port
# Windows:
netstat -ano | findstr :3000
# macOS/Linux:
lsof -i :3000

# Kill process
# Windows:
taskkill /PID <PID> /F
# macOS/Linux:
kill -9 <PID>

# Or use different ports
PORT=3001 npm run dev
FLASK_PORT=5001 python app.py
```

#### 3. Python Environment Issues

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r ml-model-api/requirements.txt

# Check Python path
which python
python -c "import sys; print(sys.path)"
```

#### 4. Node.js Issues

**Problem**: `npm ERR! code ENOENT`

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Use different registry if needed
npm config set registry https://registry.npmjs.org/
```

#### 5. Docker Issues

**Problem**: `docker: command not found`

**Solution**:
```bash
# Check Docker installation
docker --version

# Start Docker service
# Linux:
sudo systemctl start docker
sudo systemctl enable docker

# Windows/macOS:
# Start Docker Desktop application

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and log back in
```

#### 6. Memory Issues

**Problem**: `OutOfMemoryError` or slow performance

**Solution**:
```bash
# Monitor memory usage
htop  # Linux
Task Manager  # Windows
Activity Monitor  # macOS

# Reduce batch size
export MODEL_BATCH_SIZE=8

# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

### Getting Help

#### 1. Check Logs

```bash
# Docker logs
docker-compose logs -f frontend
docker-compose logs -f backend

# Application logs
tail -f logs/app.log
tail -f logs/error.log
```

#### 2. Run Diagnostics

```bash
# Full environment check
python scripts/check_environment.py --verbose

# Configuration validation
python scripts/validate_config.py --environment development

# Health check
curl http://localhost:5000/health
```

#### 3. Community Support

- **GitHub Issues**: [Report bugs](https://github.com/olaleyeolajide81-sketch/flavorsnap/issues)
- **Telegram Group**: [Join community](https://t.me/+Tf3Ll4oRiGk5ZTM0)
- **Documentation**: [Full docs](https://flavorsnap-docs.vercel.app)

---

## 🎉 Installation Complete!

You've successfully installed FlavorSnap! Here's what to do next:

1. **Start the services** using your preferred method
2. **Visit the application** at http://localhost:3000
3. **Test with a food image** to verify everything works
4. **Explore the API documentation** for integration options
5. **Join our community** for support and updates

### Quick Verification Commands

```bash
# Check all services are running
docker-compose ps

# Test the API
curl http://localhost:5000/health

# Open the application
open http://localhost:3000  # macOS
start http://localhost:3000 # Windows
xdg-open http://localhost:3000 # Linux
```

Thank you for installing FlavorSnap! 🍲✨
