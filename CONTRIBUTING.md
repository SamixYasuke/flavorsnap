# 🤝 Contributing to FlavorSnap

Thank you for your interest in contributing to FlavorSnap! This guide will help you get your development environment set up and ready to contribute.

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Development Environment Setup](#-development-environment-setup)
- [Project Structure](#-project-structure)
- [Development Workflow](#-development-workflow)
- [Code Style & Quality](#-code-style--quality)
- [Testing](#-testing)
- [Pull Request Process](#-pull-request-process)
- [Contribution Areas](#-contribution-areas)

---

## ✅ Prerequisites

Before you begin, ensure you have the following installed:

| Tool | Version | Check Command |
|------|---------|---------------|
| Python | 3.8+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | 2.30+ | `git --version` |
| Rust *(optional, smart contracts)* | 1.70+ | `rustc --version` |

**System Requirements:**
- 4GB+ RAM (PyTorch model loading)
- ~3GB disk space (PyTorch is large)

---

## 🛠️ Development Environment Setup

### 1. Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/flavorsnap.git
cd flavorsnap
git remote add upstream https://github.com/SamixYasuke/flavorsnap.git
```

### 2. Python Backend Setup

Create a virtual environment and install **all** dev dependencies:

<details>
<summary><strong>🪟 Windows (PowerShell)</strong></summary>

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

</details>

<details>
<summary><strong>🪟 Windows (Command Prompt)</strong></summary>

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements-dev.txt
```

</details>

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

</details>

<details>
<summary><strong>🐧 Linux (Debian/Ubuntu)</strong></summary>

```bash
sudo apt-get install python3-venv python3-dev gcc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

</details>

> **Verify installation:**
> ```bash
> python -c "import torch; print(f'PyTorch {torch.__version__} — GPU: {torch.cuda.is_available()}')"
> ```

### 3. Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env.local
# Edit .env.local — set NEXT_PUBLIC_API_URL=http://localhost:5000
```

### 4. Start Development Servers

Open **two terminals**:

```bash
# Terminal 1: API Server (from project root, venv activated)
cd ml-model-api
python app.py
# → API running at http://localhost:5000

# Terminal 2: Frontend (from project root)
cd frontend
npm run dev
# → Frontend running at http://localhost:3000
```

### 5. Verify Everything Works

```bash
# Check API health
curl http://localhost:5000/health

# Open browser
# → http://localhost:3000
```

---

## 📁 Project Structure

```
flavorsnap/
├── requirements.txt          # Core Python dependencies (pinned)
├── requirements-dev.txt      # Dev dependencies (testing, linting)
├── frontend/                 # Next.js web app (TypeScript)
│   ├── pages/
│   ├── styles/
│   └── package.json
├── ml-model-api/             # Flask ML inference API (Python)
│   ├── app.py                # Main Flask application
│   ├── xai.py                # Explainable AI module
│   ├── model_registry.py     # Model versioning
│   ├── ab_testing.py         # A/B testing framework
│   └── monitoring.py         # Prometheus metrics
├── contracts/                # Soroban smart contracts (Rust)
├── dashboard.py              # Panel-based dashboard
├── train_model.py            # Model training script
├── model.pth                 # Trained PyTorch model
├── docs/
│   └── dependencies.md       # Dependency documentation
└── CONTRIBUTING.md           # ← You are here
```

---

## 🔄 Development Workflow

### Branching Strategy

1. **Create a feature branch** from `main`:

   ```bash
   git fetch upstream
   git checkout -b feature/your-feature-name upstream/main
   ```

2. **Branch naming conventions:**

   | Prefix | Example | Use Case |
   |--------|---------|----------|
   | `feature/` | `feature/image-cropping` | New features |
   | `fix/` | `fix/upload-cors-error` | Bug fixes |
   | `docs/` | `docs/api-examples` | Documentation |
   | `refactor/` | `refactor/model-loader` | Code improvements |
   | `test/` | `test/batch-processor` | Test additions |

3. **Keep your branch up to date:**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### Commit Messages

Follow [Conventional Commits](https://conventionalcommits.org/):

```
feat: add multi-image upload support
fix: resolve CORS error on /predict endpoint
docs: add API examples for batch processing
style: format model_registry.py with black
refactor: extract image transforms to utils
test: add unit tests for ABTestManager
chore: update torch to 2.2.0
```

---

## 🧹 Code Style & Quality

### Python (Backend)

- **Style:** PEP 8 compliant
- **Formatter:** [black](https://black.readthedocs.io/) (auto-format)
- **Imports:** Sorted by [isort](https://pycqa.github.io/isort/)
- **Linting:** [flake8](https://flake8.pycqa.org/)
- **Type Checking:** [mypy](https://mypy-lang.org/)

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Type check
mypy ml-model-api/ --ignore-missing-imports
```

### TypeScript (Frontend)

- **Style:** Strict mode TypeScript
- **Components:** Functional components with hooks
- **CSS:** TailwindCSS utility classes
- **Linting:** ESLint + Prettier

```bash
cd frontend
npm run lint
npm run format
```

### Rust (Smart Contracts)

- **Formatter:** `rustfmt`
- **Linting:** `clippy`

```bash
cargo fmt --check
cargo clippy -- -D warnings
```

---

## 🧪 Testing

### Python Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=ml-model-api --cov-report=html

# Run specific test file
pytest test_analytics.py -v
```

### Frontend Tests

```bash
cd frontend
npm run test              # Unit tests
npm run test:coverage     # With coverage
npm run test:e2e          # End-to-end tests
```

### Pre-commit Checks

Run all checks before committing:

```bash
# Python
black --check .
isort --check .
flake8 .
pytest

# Frontend
cd frontend && npm run lint && npm run test
```

---

## 📬 Pull Request Process

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows the project's style guidelines
- [ ] All existing tests pass (`pytest` and `npm run test`)
- [ ] New functionality has corresponding tests
- [ ] Documentation is updated (README, docstrings, etc.)
- [ ] Commit messages follow Conventional Commits
- [ ] No secrets or credentials committed
- [ ] Virtual environment files (`venv/`) are not included

### Submitting a PR

1. Push your branch to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request on GitHub against `main`

3. Fill in the PR template:
   - **What** does this PR do?
   - **Why** is this change needed?
   - **How** was this tested?
   - Screenshots (for UI changes)

4. Request a review from maintainers

5. Address review feedback with fixup commits

---

## 🏆 Contribution Areas

### 🎨 Frontend
- UI/UX improvements
- New React components
- Performance optimizations
- Mobile responsiveness
- Accessibility (a11y)
- Internationalization (i18n)

### 🧠 Machine Learning
- Model architecture improvements
- New food categories
- Training pipeline improvements
- Data augmentation strategies
- Model compression

### ⚙️ Backend API
- New API endpoints
- Rate limiting improvements
- Security hardening
- Database integration
- Caching layer

### 📝 Documentation
- API documentation
- Tutorials and guides
- Video walkthroughs
- Translation

### 🔗 Blockchain
- Soroban smart contract features
- Token incentive mechanisms
- Model governance

---

## 💬 Getting Help

- **Telegram Group:** [Join our community](https://t.me/+Tf3Ll4oRiGk5ZTM0)
- **GitHub Issues:** [Browse open issues](https://github.com/SamixYasuke/flavorsnap/issues)
- **Discussions:** Ask questions in GitHub Discussions

> Look for issues labeled `good first issue` or `help wanted` to find beginner-friendly tasks.

---

Thank you for contributing to FlavorSnap! 🍲✨
