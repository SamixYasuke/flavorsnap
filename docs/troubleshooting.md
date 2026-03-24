# 🔧 FlavorSnap Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with FlavorSnap installation and operation.

## 📋 Table of Contents

- [🚨 Quick Diagnostics](#-quick-diagnostics)
- [🔍 Installation Issues](#-installation-issues)
- [⚙️ Configuration Problems](#️-configuration-problems)
- [🐳 Docker Issues](#-docker-issues)
- [🧠 Model & ML Problems](#-model--ml-problems)
- [🌐 Network & API Issues](#-network--api-issues)
- [🖥️ Frontend Problems](#️-frontend-problems)
- [💾 Database & Storage Issues](#-database--storage-issues)
- [🎮 GPU & Performance Issues](#-gpu--performance-issues)
- [🔒 Security & Permission Issues](#-security--permission-issues)
- [📱 Platform-Specific Issues](#-platform-specific-issues)
- [🛠️ Advanced Debugging](#️-advanced-debugging)
- [📞 Getting Help](#-getting-help)

## 🚨 Quick Diagnostics

### Run Health Check

```bash
# Full environment validation
python scripts/check_environment.py --all --verbose

# Quick health check
curl http://localhost:5000/health

# Check Docker services
docker-compose ps
```

### Common Error Patterns

| Error Type | Symptoms | Quick Fix |
|------------|----------|-----------|
| **Port Conflict** | "Address already in use" | Change ports or kill conflicting processes |
| **Missing Dependencies** | "ModuleNotFoundError" | Install missing packages |
| **Permission Denied** | "Access denied" errors | Fix file permissions |
| **Out of Memory** | Slow performance, crashes | Increase RAM or reduce batch size |
| **Network Issues** | Connection timeouts | Check firewall, DNS, proxy settings |

## 🔍 Installation Issues

### Problem: Python Not Found

**Symptoms:**
```
python: command not found
ModuleNotFoundError: No module named 'torch'
```

**Causes:**
- Python not installed
- Python not in PATH
- Wrong Python version

**Solutions:**

#### 1. Install Python
```bash
# Windows (using Chocolatey)
choco install python --version=3.9.7

# macOS (using Homebrew)
brew install python@3.9

# Linux (Ubuntu/Debian)
sudo apt install python3.9 python3.9-pip python3.9-venv
```

#### 2. Fix PATH
```bash
# Add to ~/.bashrc or ~/.zshrc (Linux/macOS)
export PATH="/usr/local/bin:$PATH"

# Windows (System Properties > Environment Variables)
# Add Python installation directory to PATH
```

#### 3. Verify Installation
```bash
python --version  # Should be 3.8+
pip --version
```

### Problem: Node.js Installation Failed

**Symptoms:**
```
node: command not found
npm: command not found
```

**Solutions:**

#### 1. Install Node.js
```bash
# Windows
choco install nodejs --version=18.17.0

# macOS
brew install node@18

# Linux
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### 2. Clear npm Cache
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Problem: Git Not Available

**Symptoms:**
```
git: command not found
fatal: not a git repository
```

**Solutions:**

#### 1. Install Git
```bash
# Windows
choco install git

# macOS
brew install git

# Linux
sudo apt install git
```

#### 2. Configure Git
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Problem: Insufficient System Resources

**Symptoms:**
```
OutOfMemoryError
Killed
System resources exhausted
```

**Solutions:**

#### 1. Check Resources
```bash
# Check RAM
free -h  # Linux
wmic computersystem get TotalPhysicalMemory  # Windows
sysctl hw.memsize  # macOS

# Check Disk Space
df -h  # Linux/macOS
dir  # Windows
```

#### 2. Optimize Usage
```bash
# Reduce model batch size
export MODEL_BATCH_SIZE=8

# Clear Python cache
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clear npm cache
npm cache clean --force
```

## ⚙️ Configuration Problems

### Problem: Environment Variables Not Working

**Symptoms:**
```
KeyError: 'NEXT_PUBLIC_API_URL'
Environment variable not found
```

**Solutions:**

#### 1. Create .env File
```bash
# Copy template
cp .env.example .env

# Edit with your values
nano .env
```

#### 2. Verify .env Structure
```env
# Correct format
NODE_ENV=development
NEXT_PUBLIC_API_URL=http://localhost:5000

# Incorrect format (no spaces around =)
NODE_ENV = development  # WRONG
```

#### 3. Restart Services
```bash
# Restart after changing .env
docker-compose down
docker-compose up -d
```

### Problem: YAML Configuration Invalid

**Symptoms:**
```
yaml.scanner.ScannerError
while scanning for the next token
```

**Solutions:**

#### 1. Validate YAML Syntax
```bash
# Using Python
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Using yamllint (if installed)
yamllint config.yaml
```

#### 2. Fix Common YAML Issues
```yaml
# Correct indentation
app:
  name: "FlavorSnap"
  version: "1.0.0"

# Incorrect indentation (mixed spaces/tabs)
app:
	name: "FlavorSnap"  # WRONG - use spaces only
```

#### 3. Use Online Validator
- [YAML Lint](https://yamllint.com/)
- [Online YAML Parser](https://codebeautify.org/yaml-validator)

## 🐳 Docker Issues

### Problem: Docker Command Not Found

**Symptoms:**
```
docker: command not found
Cannot connect to the Docker daemon
```

**Solutions:**

#### 1. Install Docker
```bash
# Linux
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Windows/macOS
# Download and install Docker Desktop
```

#### 2. Start Docker Service
```bash
# Linux
sudo systemctl start docker
sudo systemctl enable docker

# Windows/macOS
# Start Docker Desktop application
```

#### 3. Verify Installation
```bash
docker --version
docker run hello-world
```

### Problem: Docker Compose Issues

**Symptoms:**
```
docker-compose: command not found
unknown service: frontend
port already allocated
```

**Solutions:**

#### 1. Install Docker Compose
```bash
# Linux
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Windows/macOS
# Included with Docker Desktop
```

#### 2. Fix Port Conflicts
```bash
# Find process using port
lsof -i :3000  # Linux/macOS
netstat -ano | findstr :3000  # Windows

# Kill process
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows
```

#### 3. Rebuild Containers
```bash
# Clean rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Problem: Container Keeps Restarting

**Symptoms:**
```
Status: restarting
Exit code: 1
```

**Solutions:**

#### 1. Check Logs
```bash
docker-compose logs frontend
docker-compose logs backend
```

#### 2. Debug Container
```bash
# Run container interactively
docker-compose run --rm backend bash

# Check processes
ps aux
```

#### 3. Fix Common Issues
```bash
# Fix permissions
sudo chown -R $USER:$USER uploads/

# Fix volume mounts
# Check docker-compose.yml paths
```

## 🧠 Model & ML Problems

### Problem: Model Loading Failed

**Symptoms:**
```
FileNotFoundError: model.pth not found
RuntimeError: Error loading model
```

**Solutions:**

#### 1. Verify Model File
```bash
# Check if model exists
ls -la model.pth

# Check file size (should be ~44MB)
du -h model.pth

# Verify model integrity
python -c "import torch; model = torch.load('model.pth'); print(f'Model keys: {list(model.keys())}')"
```

#### 2. Download Model (if missing)
```bash
# Example download command (replace with actual URL)
wget https://example.com/flavorsnap-model.pth -O model.pth

# Or use curl
curl -L https://example.com/flavorsnap-model.pth -o model.pth
```

#### 3. Fix Model Path
```python
# In app.py, check model path
import os
model_path = os.path.join(os.getcwd(), 'model.pth')
print(f"Looking for model at: {model_path}")
print(f"Model exists: {os.path.exists(model_path)}")
```

### Problem: PyTorch Installation Issues

**Symptoms:**
```
ImportError: No module named 'torch'
CUDA out of memory
RuntimeError: CUDA device not found
```

**Solutions:**

#### 1. Install PyTorch
```bash
# CPU version
pip install torch torchvision

# CUDA version (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ROCm version (AMD GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

#### 2. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### 3. Fix Memory Issues
```python
# Reduce batch size in app.py
batch_size = 8  # Reduce from 32

# Enable memory optimization
torch.cuda.empty_cache()
```

### Problem: Model Prediction Errors

**Symptoms:**
```
ValueError: Expected 2D input
IndexError: list index out of range
```

**Solutions:**

#### 1. Check Input Format
```python
# Verify image preprocessing
from PIL import Image
import torchvision.transforms as transforms

# Test with sample image
image = Image.open('test-food.jpg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
tensor = transform(image)
print(f"Input shape: {tensor.shape}")  # Should be [3, 224, 224]
```

#### 2. Check Model Output
```python
# Test model prediction
import torch
model = torch.load('model.pth')
model.eval()

with torch.no_grad():
    output = model(torch.randn(1, 3, 224, 224))
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
```

## 🌐 Network & API Issues

### Problem: API Connection Failed

**Symptoms:**
```
Connection refused
ECONNREFUSED
Network error
```

**Solutions:**

#### 1. Check Backend Status
```bash
# Check if backend is running
curl http://localhost:5000/health

# Check process
ps aux | grep python
```

#### 2. Fix CORS Issues
```python
# In Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])
```

#### 3. Check Firewall
```bash
# Linux
sudo ufw status
sudo ufw allow 5000

# Windows
# Check Windows Defender Firewall settings
```

### Problem: Frontend Can't Reach Backend

**Symptoms:**
```
ERR_CONNECTION_REFUSED
NetworkError: Failed to fetch
```

**Solutions:**

#### 1. Verify API URL
```javascript
// In frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:5000

// Check in browser console
console.log(process.env.NEXT_PUBLIC_API_URL);
```

#### 2. Test API Directly
```bash
# Test with curl
curl -X POST http://localhost:5000/predict -F "image=@test-food.jpg"

# Test with browser
# Open http://localhost:5000/health in browser
```

#### 3. Check Proxy Settings
```javascript
// In next.config.js
module.exports = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/:path*',
      },
    ];
  },
};
```

## 🖥️ Frontend Problems

### Problem: Build Errors

**Symptoms:**
```
Module not found
Failed to compile
TypeScript errors
```

**Solutions:**

#### 1. Clear Cache
```bash
# Clear Next.js cache
rm -rf .next

# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

#### 2. Fix TypeScript Errors
```bash
# Check TypeScript errors
npx tsc --noEmit

# Fix common issues
# - Add type definitions
# - Fix import paths
# - Update tsconfig.json
```

#### 3. Check Environment Variables
```bash
# Verify .env.local
cat frontend/.env.local

# Restart development server
npm run dev
```

### Problem: CSS/Styling Issues

**Symptoms:**
```
Tailwind classes not working
CSS not loading
Styling broken
```

**Solutions:**

#### 1. Check Tailwind Setup
```bash
# Verify Tailwind config
cat tailwind.config.js

# Check CSS imports
# In styles/globals.css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

#### 2. Rebuild CSS
```bash
# Rebuild Tailwind
npm run build

# Check PostCSS config
cat postcss.config.js
```

#### 3. Debug in Browser
```javascript
// In browser console
// Check if Tailwind classes are applied
document.querySelectorAll('[class*="bg-"]');
```

## 💾 Database & Storage Issues

### Problem: Database Connection Failed

**Symptoms:**
```
Connection refused
Authentication failed
Database not found
```

**Solutions:**

#### 1. Check Database Status
```bash
# PostgreSQL
docker-compose ps postgres
docker-compose logs postgres

# Test connection
psql -h localhost -U flavorsnap -d flavorsnap
```

#### 2. Fix Connection String
```env
# In .env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=flavorsnap
POSTGRES_USER=flavorsnap
POSTGRES_PASSWORD=your_password
```

#### 3. Reset Database
```bash
# Reset with Docker
docker-compose down
docker volume rm flavorsnap_postgres_data
docker-compose up -d postgres
```

### Problem: File Upload Issues

**Symptoms:**
```
File too large
Invalid file type
Upload failed
```

**Solutions:**

#### 1. Check File Size Limits
```env
# In .env
MAX_FILE_SIZE=10485760  # 10MB

# In Flask app
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

#### 2. Fix File Permissions
```bash
# Create uploads directory
mkdir -p uploads
chmod 755 uploads

# Check ownership
ls -la uploads/
```

#### 3. Validate File Types
```python
# In Flask app
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

## 🎮 GPU & Performance Issues

### Problem: GPU Not Detected

**Symptoms:**
```
CUDA not available
GPU device not found
Falling back to CPU
```

**Solutions:**

#### 1. Check GPU Hardware
```bash
# NVIDIA
nvidia-smi

# AMD
rocm-smi
```

#### 2. Install Drivers
```bash
# NVIDIA CUDA
# Download from https://developer.nvidia.com/cuda-downloads

# AMD ROCm
# Download from https://rocm.docs.amd.com/en/latest/deploy/linux/index.html
```

#### 3. Verify PyTorch CUDA
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Problem: Performance Issues

**Symptoms:**
```
Slow inference
High memory usage
CPU at 100%
```

**Solutions:**

#### 1. Optimize Model
```python
# Use model quantization
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

#### 2. Enable Caching
```python
# Add Redis caching
import redis
r = redis.Redis(host='localhost', port=6379)

# Cache predictions
cache_key = f"prediction:{image_hash}"
cached_result = r.get(cache_key)
```

#### 3. Monitor Resources
```bash
# Monitor memory
htop  # Linux
Task Manager  # Windows
Activity Monitor  # macOS

# Monitor GPU
nvidia-smi -l 1  # NVIDIA
```

## 🔒 Security & Permission Issues

### Problem: Permission Denied

**Symptoms:**
```
Permission denied
Access denied
Operation not permitted
```

**Solutions:**

#### 1. Fix File Permissions
```bash
# Change ownership
sudo chown -R $USER:$USER /path/to/flavorsnap

# Fix permissions
chmod -R 755 uploads/
chmod +x scripts/*.py
```

#### 2. Run as Correct User
```bash
# Don't run as root
# Use regular user account

# For Docker, use USER directive
USER node  # In Dockerfile
```

#### 3. Check SELinux (Linux)
```bash
# Check SELinux status
sestatus

# Temporarily disable for testing
sudo setenforce 0

# Fix context
sudo chcon -R -t httpd_sys_content_rw_t uploads/
```

## 📱 Platform-Specific Issues

### Windows Issues

#### Problem: Path Separator Issues
**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'config\\model.pth'
```

**Solutions:**
```python
# Use pathlib for cross-platform compatibility
from pathlib import Path

model_path = Path('config') / 'model.pth'
```

#### Problem: Long Path Names
**Solutions:**
```powershell
# Enable long path support
# In Group Policy: Computer Configuration > Administrative Templates > System > Filesystem
# Enable "Enable Win32 long paths"
```

### macOS Issues

#### Problem: Gatekeeper Blocking Apps
**Symptoms:**
```
App can't be opened because Apple cannot check it for malicious software
```

**Solutions:**
```bash
# Allow app
sudo xattr -d com.apple.quarantine /path/to/app

# Or disable Gatekeeper (not recommended)
sudo spctl --master-disable
```

#### Problem: Python Version Conflicts
**Solutions:**
```bash
# Use pyenv for version management
brew install pyenv
pyenv install 3.9.7
pyenv global 3.9.7
```

### Linux Issues

#### Problem: Missing System Libraries
**Symptoms:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt install libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install mesa-libGL glib2
```

#### Problem: Port Privileges
**Solutions:**
```bash
# Use ports > 1024 for non-root users
# Or use authbind to bind to privileged ports
sudo apt install authbind
sudo touch /etc/authbind/byport/80
sudo chmod 755 /etc/authbind/byport/80
```

## 🛠️ Advanced Debugging

### Enable Debug Mode

```env
# In .env
DEBUG=true
LOG_LEVEL=debug
NODE_ENV=development
```

### Use Debug Tools

#### Python Debugging
```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or use ipdb
import ipdb; ipdb.set_trace()
```

#### JavaScript Debugging
```javascript
// Add console logs
console.log('Debug:', variable);

// Use debugger statement
debugger;
```

#### Docker Debugging
```bash
# Enter running container
docker-compose exec backend bash

# Inspect container
docker inspect flavorsnap_backend_1
```

### Performance Profiling

#### Python Profiling
```python
import cProfile
import pstats

# Profile function
profiler = cProfile.Profile()
profiler.enable()
# ... your code ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

#### Node.js Profiling
```bash
# Profile Next.js
npm run build
npm run start
# Open Chrome DevTools > Node.js profiling
```

### Log Analysis

#### Check Application Logs
```bash
# Docker logs
docker-compose logs -f --tail=100 backend

# Application logs
tail -f logs/app.log

# System logs
journalctl -u docker
```

#### Monitor Resources
```bash
# Real-time monitoring
htop                    # CPU/Memory
iotop                   # I/O
nethogs                 # Network
nvidia-smi -l 1         # GPU
```

## 📞 Getting Help

### Automated Help

```bash
# Run diagnostics
python scripts/check_environment.py --all --verbose

# Try auto-fix
python scripts/check_environment.py --fix

# Generate support bundle
tar -czf flavorsnap-support-bundle.tar.gz \
  logs/ \
  .env \
  config.yaml \
  docker-compose.yml
```

### Community Support

1. **GitHub Issues**
   - Search existing issues first
   - Provide detailed error messages
   - Include system information
   - Add steps to reproduce

2. **Telegram Community**
   - [Join our group](https://t.me/+Tf3Ll4oRiGk5ZTM0)
   - Ask questions respectfully
   - Share error logs (remove sensitive data)

3. **Documentation**
   - [Installation Guide](docs/installation.md)
   - [API Documentation](README.md#api-documentation)
   - [Configuration Guide](docs/configuration.md)

### Creating a Bug Report

Include the following information:

```markdown
## Bug Description
Brief description of the issue

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What should happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Node.js: [e.g., 18.17.0]
- Docker: [e.g., 24.0.0]
- Installation method: [Docker/Manual]

## Error Messages
```
Paste full error messages here
```

## Additional Context
Any other relevant information
```

### Emergency Recovery

If everything fails:

```bash
# Complete reset
docker-compose down -v
docker system prune -a
rm -rf node_modules .next venv
git clean -fdx

# Fresh installation
python scripts/install.py --method auto --verbose
```

---

## 🔧 Quick Reference

### Common Commands

```bash
# Health check
curl http://localhost:5000/health

# Restart services
docker-compose restart

# Check logs
docker-compose logs -f

# Clear cache
npm cache clean --force
pip cache purge

# Validate environment
python scripts/check_environment.py

# Rebuild everything
docker-compose down && docker-compose build --no-cache && docker-compose up -d
```

### Important Files

- `.env` - Environment variables
- `config.yaml` - Application configuration
- `logs/` - Application logs
- `uploads/` - User uploaded files
- `model.pth` - ML model file

### Default Ports

- Frontend: 3000
- Backend API: 5000
- PostgreSQL: 5432
- Redis: 6379
- Prometheus: 9090
- Grafana: 3001

---

**Remember**: Most issues are related to configuration, dependencies, or permissions. Start with the basic diagnostics and work through the solutions systematically. If you're still stuck, don't hesitate to ask for help! 🚀
