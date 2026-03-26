# 🏗️ FlavorSnap Project Structure Documentation

## 📋 Table of Contents

- [🌳 Overview](#-overview)
- [📁 Directory Structure](#-directory-structure)
- [🎯 Core Components](#-core-components)
- [🔧 Configuration Files](#-configuration-files)
- [📦 Package Management](#-package-management)
- [🐳 Container Structure](#-container-structure)
- [🔗 Dependencies](#-dependencies)
- [📊 Architecture Patterns](#-architecture-patterns)
- [🎨 Naming Conventions](#-naming-conventions)

## 🌳 Overview

FlavorSnap is a comprehensive AI-powered food classification application built with a modern microservices architecture. The project follows a modular structure that separates concerns across frontend, backend, machine learning, and blockchain components.

### Key Principles

- **Separation of Concerns**: Each component has a distinct responsibility
- **Scalability**: Modular design allows independent scaling
- **Maintainability**: Clear organization makes the codebase approachable
- **Testability**: Isolated components enable focused testing
- **Deployment Flexibility**: Containerized services for various environments

## 📁 Directory Structure

```
flavorsnap/
├── 📁 frontend/                    # Next.js web application
│   ├── 📁 pages/                   # React pages and API routes
│   │   ├── 📄 index.tsx           # Landing page
│   │   ├── 📄 classify.tsx        # Classification interface
│   │   └── 📁 api/                # Backend API endpoints
│   ├── 📁 components/             # Reusable React components
│   │   ├── 📁 ui/                 # UI component library
│   │   ├── 📁 forms/              # Form components
│   │   └── 📁 layout/             # Layout components
│   ├── 📁 hooks/                  # Custom React hooks
│   ├── 📁 lib/                    # Utility libraries
│   ├── 📁 public/                 # Static assets
│   │   ├── 📁 images/             # Hero images and icons
│   │   └── 📄 favicon.ico
│   ├── 📁 styles/                 # Global CSS and Tailwind
│   ├── 📁 __tests__/              # Test files
│   ├── 📄 package.json            # Frontend dependencies
│   ├── 📄 tsconfig.json           # TypeScript configuration
│   ├── 📄 next.config.ts          # Next.js configuration
│   └── 📄 tailwind.config.ts      # TailwindCSS configuration
├── 📁 ml-model-api/               # Flask ML inference API
│   ├── 📄 app.py                  # Main Flask application
│   ├── 📄 requirements.txt        # Python dependencies
│   ├── 📄 model_loader.py         # Model loading utilities
│   ├── 📄 analytics.py            # Analytics and metrics
│   ├── 📄 monitoring.py           # Performance monitoring
│   ├── 📄 api_endpoints.py        # API route definitions
│   ├── 📄 batch_processor.py      # Batch processing logic
│   ├── 📄 model_registry.py       # Model version management
│   ├── 📄 xai.py                  # Explainable AI features
│   └── 📁 logs/                   # Application logs
├── 📁 contracts/                  # Soroban smart contracts
│   ├── 📁 model-governance/       # Model governance contracts
│   ├── 📁 tokenized-incentive/    # Token incentive system
│   └── 📁 sensory-evaluation/     # Sensory evaluation contracts
├── 📁 flavorsnap-food-registry/   # Rust-based food registry
│   ├── 📁 src/                    # Rust source code
│   ├── 📄 Cargo.toml              # Rust dependencies
│   └── 📄 README.md               # Component documentation
├── 📁 dataset/                    # Training and validation data
│   ├── 📁 train/                  # Training images by class
│   ├── 📁 test/                   # Test images
│   ├── 📁 validation/             # Validation dataset
│   └── 📄 data_split.py           # Dataset utilities
├── 📁 models/                     # Trained model files
│   ├── 📄 model.pth               # Main PyTorch model (44MB)
│   ├── 📄 model_metadata.json    # Model information
│   └── 📁 checkpoints/            # Training checkpoints
├── 📁 uploads/                    # User uploaded images
│   ├── 📁 classified/            # Organized by prediction
│   └── 📁 temp/                   # Temporary uploads
├── 📁 docs/                       # Documentation
│   ├── 📄 project_structure.md    # This file
│   ├── 📄 development_workflow.md # Development guide
│   ├── 📄 file_purposes.md        # File responsibilities
│   ├── 📄 installation.md         # Setup instructions
│   ├── 📄 configuration.md        # Configuration guide
│   └── 📄 troubleshooting.md      # Common issues
├── 📁 scripts/                    # Utility scripts
│   ├── 📄 install.py              # Automated installation
│   ├── 📄 docker_run.sh           # Docker utilities
│   ├── 📄 check_environment.py     # Environment validation
│   └── 📄 analyze_structure.py     # Structure analysis
├── 📁 config/                     # Configuration files
│   ├── 📄 default.yaml            # Default settings
│   ├── 📄 development.yaml        # Development config
│   └── 📄 production.yaml         # Production config
├── 📁 kubernetes/                 # K8s deployment manifests
│   ├── 📄 deployment.yaml         # Main deployment
│   ├── 📄 service.yaml            # Service definitions
│   └── 📄 monitoring.yaml          # Monitoring stack
├── 📁 monitoring/                 # Monitoring configuration
│   ├── 📁 prometheus/             # Prometheus config
│   ├── 📁 grafana/               # Grafana dashboards
│   └── 📁 alerts/                # Alert rules
├── 📁 src/                       # Shared source code
│   ├── 📁 config/                 # Configuration utilities
│   └── 📁 core/                   # Core business logic
├── 📁 backend/                    # Backend services
│   ├── 📁 api/                   # API implementations
│   └── 📁 services/               # Business services
├── 📁 pages/                      # Additional documentation pages
├── 📁 api-testing/               # API testing utilities
├── 📁 dashboard/                  # Analytics dashboard
├── 📄 model.pth                   # Trained PyTorch model
├── 📄 food_classes.txt            # List of food categories
├── 📄 train_model.ipynb           # Model training notebook
├── 📄 train_model.py              # Model training script
├── 📄 dashboard.py                # Panel-based dashboard
├── 📄 Cargo.toml                  # Rust workspace configuration
├── 📄 config.yaml                 # Main configuration file
├── 📄 docker-compose.yml          # Development containers
├── 📄 docker-compose.prod.yml     # Production containers
├── 📄 docker-compose.test.yml     # Testing containers
├── 📄 Dockerfile                  # Production container
├── 📄 Dockerfile.dev              # Development container
├── 📄 Dockerfile.frontend.dev     # Frontend dev container
├── 📄 .env.example                # Environment template
├── 📄 .gitignore                  # Git ignore rules
├── 📄 .dockerignore               # Docker ignore rules
└── 📄 README.md                   # Project documentation
```

## 🎯 Core Components

### Frontend (`frontend/`)

**Purpose**: User interface and client-side application logic

**Key Files**:
- `pages/index.tsx` - Main landing page
- `pages/classify.tsx` - Food classification interface
- `components/` - Reusable UI components
- `hooks/` - Custom React hooks for state management
- `lib/` - Utility functions and helpers

**Technologies**: Next.js, React, TypeScript, TailwindCSS

### Machine Learning API (`ml-model-api/`)

**Purpose**: AI model inference and data processing

**Key Files**:
- `app.py` - Main Flask application entry point
- `model_loader.py` - Model loading and caching
- `api_endpoints.py` - RESTful API definitions
- `analytics.py` - Classification analytics
- `xai.py` - Explainable AI features

**Technologies**: Flask, PyTorch, Pillow, OpenCV

### Smart Contracts (`contracts/`)

**Purpose**: Blockchain-based model governance and incentives

**Key Directories**:
- `model-governance/` - Model version control and validation
- `tokenized-incentive/` - Reward system for contributions
- `sensory-evaluation/` - Community feedback mechanisms

**Technologies**: Soroban, Rust, Stellar

### Food Registry (`flavorsnap-food-registry/`)

**Purpose**: Decentralized food classification registry

**Key Files**:
- `src/` - Rust implementation
- `Cargo.toml` - Dependencies and configuration

**Technologies**: Rust, Soroban SDK

## 🔧 Configuration Files

### Root Configuration

| File | Purpose | Format |
|------|---------|--------|
| `config.yaml` | Main application configuration | YAML |
| `.env.example` | Environment variables template | Environment |
| `Cargo.toml` | Rust workspace configuration | TOML |
| `docker-compose.yml` | Development containers | YAML |

### Component Configuration

| Component | Config File | Purpose |
|-----------|-------------|---------|
| Frontend | `next.config.ts` | Next.js settings |
| Frontend | `tsconfig.json` | TypeScript configuration |
| Frontend | `tailwind.config.ts` | TailwindCSS setup |
| ML API | `requirements.txt` | Python dependencies |
| ML API | `model_registry.py` | Model version management |

## 📦 Package Management

### Frontend Dependencies

**Package Manager**: npm/yarn/pnpm
**Key Dependencies**:
- `next` - React framework
- `react` - UI library
- `typescript` - Type safety
- `tailwindcss` - Styling
- `axios` - HTTP client
- `lucide-react` - Icons

### Backend Dependencies

**Package Manager**: pip
**Key Dependencies**:
- `flask` - Web framework
- `torch` - ML framework
- `pillow` - Image processing
- `opencv-python` - Computer vision
- `numpy` - Numerical computing

### Rust Dependencies

**Package Manager**: Cargo
**Key Dependencies**:
- `soroban-sdk` - Blockchain development
- `serde` - Serialization
- `tokio` - Async runtime

## 🐳 Container Structure

### Multi-stage Build Process

1. **Base Stage**: System dependencies
2. **Build Stage**: Application compilation
3. **Runtime Stage**: Minimal runtime environment

### Container Types

| Container | Purpose | Configuration |
|-----------|---------|---------------|
| `frontend` | Next.js application | `Dockerfile.frontend.dev` |
| `backend` | Flask ML API | `Dockerfile.dev` |
| `production` | Optimized deployment | `Dockerfile` |

### Docker Compose Environments

- **Development**: `docker-compose.yml`
- **Production**: `docker-compose.prod.yml`
- **Testing**: `docker-compose.test.yml`

## 🔗 Dependencies

### External Services

- **Database**: PostgreSQL (configurable)
- **Cache**: Redis (optional)
- **Storage**: Local filesystem (configurable to cloud)
- **Monitoring**: Prometheus + Grafana

### API Dependencies

- **ML Model**: PyTorch ResNet18
- **Image Processing**: Pillow, OpenCV
- **Web Framework**: Flask, Next.js
- **Blockchain**: Stellar/Soroban

## 📊 Architecture Patterns

### Microservices Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Frontend  │    │   Backend   │    │   ML API    │
│   Next.js   │◄──►│   Services  │◄──►│   Flask     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                   ┌─────────────┐
                   │ Blockchain  │
                   │  Contracts  │
                   └─────────────┘
```

### Data Flow

1. **User Upload** → Frontend → ML API
2. **Classification** → ML Model → Database
3. **Results** → Frontend ← Backend Services
4. **Governance** → Blockchain Contracts

### Security Patterns

- **API Gateway**: Request routing and validation
- **Authentication**: JWT tokens (configurable)
- **Authorization**: Role-based access control
- **Data Encryption**: TLS for all communications

## 🎨 Naming Conventions

### File Naming

- **Components**: PascalCase (`FoodClassifier.tsx`)
- **Utilities**: camelCase (`imageProcessor.js`)
- **Configuration**: kebab-case (`development.yaml`)
- **Documentation**: kebab-case (`project-structure.md`)

### Directory Naming

- **Features**: kebab-case (`food-classification/`)
- **Components**: kebab-case (`ui-components/`)
- **Services**: kebab-case (`ml-services/`)

### Code Naming

- **Variables**: camelCase (`foodImage`)
- **Functions**: camelCase (`classifyFood`)
- **Classes**: PascalCase (`FoodClassifier`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_FILE_SIZE`)

### API Endpoints

- **REST**: kebab-case (`/food-classification`)
- **GraphQL**: camelCase (`classifyFood`)
- **WebSocket**: camelCase (`onClassificationResult`)

---

## 📚 Additional Resources

- [Development Workflow](development_workflow.md)
- [File Purposes](file_purposes.md)
- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)

---

*Last updated: March 2026*
*Version: 1.0.0*
