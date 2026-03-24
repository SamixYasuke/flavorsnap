# 🔄 FlavorSnap Development Workflow

## 📋 Table of Contents

- [🚀 Overview](#-overview)
- [🛠️ Development Setup](#️-development-setup)
- [🌿 Git Workflow](#-git-workflow)
- [🏗️ Development Process](#️-development-process)
- [🧪 Testing Strategy](#-testing-strategy)
- [📦 Build & Deployment](#-build--deployment)
- [🔍 Code Review Process](#-code-review-process)
- [🐛 Debugging Guide](#-debugging-guide)
- [📊 Performance Monitoring](#-performance-monitoring)
- [🔄 Release Process](#-release-process)

## 🚀 Overview

This guide outlines the complete development workflow for FlavorSnap, from initial setup to production deployment. It's designed to help new contributors get started quickly and ensure consistency across the team.

### Development Principles

- **Iterative Development**: Small, frequent changes
- **Test-Driven**: Tests before implementation
- **Documentation First**: Document decisions and changes
- **Continuous Integration**: Automated validation
- **Code Quality**: Consistent style and best practices

## 🛠️ Development Setup

### Prerequisites

- **Node.js** 18+ and npm/yarn/pnpm
- **Python** 3.8+ and pip
- **Rust** 1.70+ (for contracts)
- **Docker** & Docker Compose
- **Git** for version control
- **VS Code** (recommended) with extensions

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/olaleyeolajide81-sketch/flavorsnap.git
cd flavorsnap

# Automated setup (recommended)
python scripts/install.py

# Or manual setup
npm run setup
```

### Environment Configuration

```bash
# Copy environment templates
cp .env.example .env
cp frontend/.env.example frontend/.env.local

# Configure essential variables
# Edit .env with your settings
python scripts/validate_config.py --environment development
```

### Development Tools Setup

#### VS Code Extensions

```json
{
  "recommendations": [
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "ms-python.python",
    "rust-lang.rust-analyzer",
    "ms-vscode.vscode-docker",
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint"
  ]
}
```

#### Git Hooks

```bash
# Install pre-commit hooks
npm run setup:hooks

# Manual hook installation
cp scripts/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

## 🌿 Git Workflow

### Branch Strategy

We use a simplified Git flow with main branches:

- **`main`**: Production-ready code
- **`develop`**: Integration branch for features
- **`feature/*`**: Individual feature development
- **`hotfix/*`**: Critical production fixes
- **`release/*`**: Release preparation

### Branch Naming Conventions

```bash
# Features
feature/food-classification-improvement
feature/user-authentication
feature/mobile-responsive-design

# Bug fixes
bugfix/image-upload-validation
bugfix/model-loading-error

# Hotfixes
hotfix/security-patch
hotfix/critical-bug-fix

# Releases
release/v1.1.0
release/v2.0.0-beta
```

### Commit Message Format

Follow [Conventional Commits](https://conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no functional changes)
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

#### Examples

```bash
feat(classification): add confidence threshold setting
fix(upload): resolve image validation error
docs(api): update endpoint documentation
style(frontend): format components with prettier
refactor(model): optimize image preprocessing
test(api): add integration tests for endpoints
chore(deps): update dependencies to latest versions
```

## 🏗️ Development Process

### 1. Feature Development Workflow

```bash
# Start new feature
git checkout develop
git pull origin develop
git checkout -b feature/amazing-feature

# Development setup
npm run dev:setup

# Make changes
# ... write code ...

# Run tests
npm run test
npm run lint
npm run build

# Commit changes
git add .
git commit -m "feat: add amazing feature"

# Push and create PR
git push origin feature/amazing-feature
```

### 2. Development Commands

#### Frontend Development

```bash
cd frontend

# Start development server
npm run dev

# Run tests
npm run test
npm run test:watch
npm run test:coverage

# Build for production
npm run build

# Lint and format
npm run lint
npm run lint:fix
npm run format
```

#### Backend Development

```bash
cd ml-model-api

# Start development server
python app.py

# Run tests
python -m pytest
python -m pytest --cov=app

# Lint code
flake8 .
black .
```

#### Smart Contract Development

```bash
cd contracts/model-governance

# Build contracts
cargo build

# Run tests
cargo test

# Deploy to testnet
soroban contract deploy ...
```

### 3. Local Development Environment

#### Docker Development

```bash
# Start all services
./scripts/docker_run.sh -e development -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild specific service
docker-compose up --build backend
```

#### Service URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Documentation**: http://localhost:5000/docs
- **Grafana**: http://localhost:3001
- **Prometheus**: http://localhost:9090

## 🧪 Testing Strategy

### Testing Pyramid

```
    ┌─────────────┐
    │   E2E       │  ← Few, slow, high value
    │   Tests     │
    └─────────────┘
  ┌─────────────────┐
  │  Integration   │  ← Medium number, medium speed
  │     Tests      │
  └─────────────────┘
┌─────────────────────┐
│     Unit Tests      │  ← Many, fast, focused
└─────────────────────┘
```

### Test Types

#### Unit Tests

- **Frontend**: Jest + React Testing Library
- **Backend**: pytest
- **Contracts**: cargo test

```bash
# Frontend unit tests
cd frontend
npm run test

# Backend unit tests
cd ml-model-api
python -m pytest tests/unit/

# Contract tests
cd contracts
cargo test
```

#### Integration Tests

```bash
# API integration tests
npm run test:integration

# Database integration tests
python -m pytest tests/integration/
```

#### End-to-End Tests

```bash
# E2E tests with Playwright
npm run test:e2e

# Visual regression tests
npm run test:visual
```

### Test Coverage Requirements

- **Frontend**: Minimum 80% coverage
- **Backend**: Minimum 85% coverage
- **Contracts**: Minimum 90% coverage

### Test Data Management

```bash
# Generate test data
python scripts/generate_test_data.py

# Clean test data
npm run test:clean

# Seed test database
npm run test:seed
```

## 📦 Build & Deployment

### Build Process

#### Frontend Build

```bash
cd frontend

# Development build
npm run build:dev

# Production build
npm run build:prod

# Analyze bundle size
npm run analyze
```

#### Backend Build

```bash
cd ml-model-api

# Create requirements
pip freeze > requirements.txt

# Build Docker image
docker build -t flavorsnap-backend .
```

### Deployment Environments

#### Development

```bash
# Deploy to development
./scripts/deploy.sh -e development

# or with Docker
docker-compose -f docker-compose.yml up -d
```

#### Staging

```bash
# Deploy to staging
./scripts/deploy.sh -e staging

# Run smoke tests
npm run test:smoke
```

#### Production

```bash
# Deploy to production
./scripts/deploy.sh -e production

# Health check
curl https://api.flavorsnap.com/health
```

### Environment Variables

#### Development

```env
NODE_ENV=development
DEBUG=true
NEXT_PUBLIC_API_URL=http://localhost:5000
MODEL_CONFIDENCE_THRESHOLD=0.6
```

#### Production

```env
NODE_ENV=production
DEBUG=false
NEXT_PUBLIC_API_URL=https://api.flavorsnap.com
MODEL_CONFIDENCE_THRESHOLD=0.7
DATABASE_URL=postgresql://...
JWT_SECRET=your-secret-key
```

## 🔍 Code Review Process

### Pull Request Guidelines

#### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] All tests pass
- [ ] Build succeeds
- [ ] No security vulnerabilities
- [ ] Performance impact considered

#### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated

## Screenshots (if applicable)
Add screenshots for UI changes

## Additional Notes
Any additional context or considerations
```

### Review Process

1. **Self-Review**: Review your own changes first
2. **Automated Checks**: CI/CD pipeline validation
3. **Peer Review**: At least one team member review
4. **Approval**: Required approvals before merge
5. **Merge**: Squash and merge to maintain clean history

### Review Guidelines

#### For Reviewers

- Check code quality and style
- Verify test coverage
- Assess performance impact
- Validate security considerations
- Ensure documentation completeness

#### For Authors

- Respond to all feedback
- Update code based on suggestions
- Provide context for complex changes
- Test thoroughly before resubmission

## 🐛 Debugging Guide

### Common Issues

#### Frontend Issues

```bash
# Clear cache
rm -rf .next node_modules
npm install

# Check dependencies
npm ls

# Debug build
npm run build:debug
```

#### Backend Issues

```bash
# Check Python environment
python --version
pip list

# Debug Flask app
FLASK_ENV=development python app.py

# Check model loading
python -c "import torch; print(torch.load('model.pth').keys())"
```

#### Docker Issues

```bash
# Check container logs
docker-compose logs backend

# Rebuild containers
docker-compose build --no-cache

# Check resource usage
docker stats
```

### Debugging Tools

#### Frontend

- **React DevTools**: Component inspection
- **Redux DevTools**: State management
- **Chrome DevTools**: Performance profiling
- **Lighthouse**: Accessibility and performance

#### Backend

- **Flask Debugger**: Interactive debugging
- **Python Debugger (pdb)**: Step-through debugging
- **Logging**: Application logs
- **Profiling**: Performance analysis

### Performance Debugging

```bash
# Frontend performance
npm run analyze
npm run lighthouse

# Backend performance
python -m cProfile app.py
python -m memory_profiler app.py

# Database performance
python scripts/db_profiler.py
```

## 📊 Performance Monitoring

### Metrics to Track

#### Frontend Metrics

- **Core Web Vitals**: LCP, FID, CLS
- **Bundle Size**: JavaScript and CSS sizes
- **Load Time**: Page load performance
- **Error Rate**: JavaScript errors

#### Backend Metrics

- **Response Time**: API endpoint performance
- **Throughput**: Requests per second
- **Error Rate**: HTTP error rates
- **Resource Usage**: CPU, memory, disk

#### ML Model Metrics

- **Inference Time**: Model prediction speed
- **Accuracy**: Classification accuracy
- **Memory Usage**: Model memory footprint
- **GPU Utilization**: GPU performance (if applicable)

### Monitoring Tools

#### Application Monitoring

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# View metrics
open http://localhost:3001  # Grafana
open http://localhost:9090  # Prometheus
```

#### Logging

```bash
# View application logs
docker-compose logs -f backend

# Filter logs
docker-compose logs backend | grep ERROR

# Log analysis
python scripts/analyze_logs.py
```

## 🔄 Release Process

### Version Management

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

#### Pre-Release

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number updated
- [ ] Security scan completed
- [ ] Performance testing completed

#### Release Steps

```bash
# Create release branch
git checkout -b release/v1.2.0

# Update version numbers
npm version patch  # or minor/major

# Build and test
npm run build
npm run test

# Tag release
git tag -a v1.2.0 -m "Release version 1.2.0"

# Push to main
git push origin main --tags
```

#### Post-Release

- [ ] Deploy to production
- [ ] Monitor for issues
- [ ] Update documentation
- [ ] Announce release
- [ ] Create next development iteration

### Automated Releases

```bash
# Automated release script
./scripts/release.sh --version 1.2.0 --environment production

# Rollback if needed
./scripts/rollback.sh --version 1.1.0
```

## 📚 Additional Resources

- [Project Structure](project_structure.md)
- [File Purposes](file_purposes.md)
- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)

---

## 🤝 Getting Help

- **Telegram Group**: [Join our community](https://t.me/+Tf3Ll4oRiGk5ZTM0)
- **GitHub Issues**: [Report bugs](https://github.com/olaleyeolajide81-sketch/flavorsnap/issues)
- **Documentation**: [Full docs](https://docs.flavorsnap.com)
- **Email**: dev@flavorsnap.com

---

*Last updated: March 2026*
*Version: 1.0.0*
