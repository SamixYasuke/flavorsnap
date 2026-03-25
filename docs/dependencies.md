# 📦 FlavorSnap Dependency Documentation

## Overview

FlavorSnap uses two requirement files to manage Python dependencies:

| File | Purpose | Install Command |
|------|---------|-----------------|
| `requirements.txt` | Core runtime dependencies | `pip install -r requirements.txt` |
| `requirements-dev.txt` | Development tools + core deps | `pip install -r requirements-dev.txt` |

> **Note:** `requirements-dev.txt` includes `requirements.txt` via `-r requirements.txt`, so you only need to install one file depending on your role.

---

## Dependency Categories

### 🧠 Machine Learning & Deep Learning

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | `>=2.0.0,<3.0.0` | PyTorch deep learning framework |
| `torchvision` | `>=0.15.0,<1.0.0` | Vision models, transforms, datasets |
| `numpy` | `>=1.24.0,<2.0.0` | Numerical computing |

**Used by:** `train_model.py`, `dashboard.py`, `ml-model-api/xai.py`, `ml-model-api/model_validator.py`

#### PyTorch Platform-Specific Installation

PyTorch installation varies by platform and GPU availability. The `requirements.txt` installs the **CPU-only** version by default. For GPU support:

**CUDA 11.8 (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1 (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**macOS (Apple Silicon / MPS):**
```bash
# Default pip install includes MPS support on macOS
pip install torch torchvision
```

**CPU Only (all platforms):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the latest installation matrix.

---

### 🖼️ Image Processing

| Package | Version | Purpose |
|---------|---------|---------|
| `Pillow` | `>=10.0.0,<11.0.0` | Image loading, resizing, format conversion |
| `opencv-python` | `>=4.8.0,<5.0.0` | Advanced image processing (XAI heatmaps) |

**Used by:** `dashboard.py`, `ml-model-api/xai.py`, `ml-model-api/social_sharing.py`

---

### 🌐 Web Framework & API

| Package | Version | Purpose |
|---------|---------|---------|
| `Flask` | `>=3.0.0,<4.0.0` | HTTP API server |
| `Flask-Cors` | `>=4.0.0,<5.0.0` | Cross-origin resource sharing |
| `Flask-Limiter` | `>=3.5.0,<4.0.0` | Rate limiting for API endpoints |
| `Werkzeug` | `>=3.0.0,<4.0.0` | WSGI utilities (file upload handling) |

**Used by:** `ml-model-api/app.py`, `ml-model-api/api_endpoints.py`, `ml-model-api/swagger_setup.py`

---

### 📊 Data Visualization & Dashboard

| Package | Version | Purpose |
|---------|---------|---------|
| `panel` | `>=1.3.0,<2.0.0` | Interactive dashboards (classification UI) |
| `plotly` | `>=5.18.0,<6.0.0` | Performance dashboard charts |
| `pandas` | `>=2.1.0,<3.0.0` | Data manipulation for analytics |
| `matplotlib` | `>=3.8.0,<4.0.0` | Confusion matrix plots |
| `seaborn` | `>=0.13.0,<1.0.0` | Statistical visualization (heatmaps) |

**Used by:** `dashboard.py`, `ml-model-api/performance_dashboard.py`, `ml-model-api/model_validator.py`

---

### 📐 ML Metrics & Evaluation

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | `>=1.3.0,<2.0.0` | Accuracy, precision, recall, F1, confusion matrix |
| `scipy` | `>=1.11.0,<2.0.0` | Statistical tests (A/B testing t-tests) |

**Used by:** `ml-model-api/model_validator.py`, `ml-model-api/ab_testing.py`

---

### 📡 Monitoring & Observability

| Package | Version | Purpose |
|---------|---------|---------|
| `prometheus-client` | `>=0.19.0,<1.0.0` | Metrics export for Prometheus/Grafana |
| `psutil` | `>=5.9.0,<6.0.0` | System resource monitoring (CPU, memory) |

**Used by:** `ml-model-api/monitoring.py`

---

### 🔧 Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| `PyYAML` | `>=6.0.0,<7.0.0` | OpenAPI spec parsing (Swagger docs) |
| `requests` | `>=2.31.0,<3.0.0` | HTTP client for test scripts |

**Used by:** `ml-model-api/swagger_setup.py`, `test_analytics.py`, `ml-model-api/test_rate_limiter.py`

---

## Development Dependencies

These are **only** needed for contributors working on the codebase:

### 🧪 Testing

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | `>=7.4.0,<9.0.0` | Test runner |
| `pytest-cov` | `>=4.1.0,<6.0.0` | Code coverage reporting |
| `pytest-flask` | `>=1.3.0,<2.0.0` | Flask test client fixtures |
| `pytest-mock` | `>=3.12.0,<4.0.0` | Mocking utilities |
| `pytest-xdist` | `>=3.5.0,<4.0.0` | Parallel test execution |

### 🧹 Code Quality

| Package | Version | Purpose |
|---------|---------|---------|
| `flake8` | `>=7.0.0,<8.0.0` | Linting (PEP 8 compliance) |
| `black` | `>=24.0.0,<25.0.0` | Code formatting |
| `isort` | `>=5.13.0,<6.0.0` | Import sorting |
| `mypy` | `>=1.8.0,<2.0.0` | Static type checking |

### 🔒 Security

| Package | Version | Purpose |
|---------|---------|---------|
| `bandit` | `>=1.7.0,<2.0.0` | Security vulnerability scanner |
| `safety` | `>=3.0.0,<4.0.0` | Known vulnerability checker |

### 📖 Documentation & Notebooks

| Package | Version | Purpose |
|---------|---------|---------|
| `Sphinx` | `>=7.2.0,<8.0.0` | Documentation generation |
| `sphinx-rtd-theme` | `>=2.0.0,<3.0.0` | Read the Docs theme |
| `jupyter` | `>=1.0.0` | Notebook support for model training |
| `ipykernel` | `>=6.29.0,<7.0.0` | Jupyter kernel |

---

## Versioning Strategy

FlavorSnap uses **compatible release ranges** (`>=X.Y.Z,<N.0.0`) for version pinning:

- **Floor pin** (`>=X.Y.Z`): Ensures minimum feature support and known bug fixes
- **Ceiling pin** (`<N.0.0`): Prevents breaking changes from major version bumps
- This approach allows **patch and minor** updates while blocking **major** updates

### Updating Dependencies

To update dependencies safely:

```bash
# Check for outdated packages
pip list --outdated

# Update a specific package
pip install --upgrade <package-name>

# Regenerate exact pins (for reproducible builds)
pip freeze > requirements.lock
```

---

## Troubleshooting

### Common Issues

#### `torch` installation fails on Windows
```bash
# Use the official PyTorch install command for your system:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### `opencv-python` conflicts with `opencv-contrib-python`
```bash
# Uninstall conflicting packages first:
pip uninstall opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python
```

#### `psutil` fails to build on Linux
```bash
# Install build dependencies:
sudo apt-get install python3-dev gcc
pip install psutil
```

#### `scipy` or `scikit-learn` build takes too long
```bash
# Install pre-built wheels:
pip install --only-binary :all: scipy scikit-learn
```

#### Memory errors when installing `torch`
PyTorch is a large package (~2GB). Ensure you have sufficient disk space and RAM.
```bash
# Install with no cache to save disk space:
pip install --no-cache-dir torch torchvision
```
