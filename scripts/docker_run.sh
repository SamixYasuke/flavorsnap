#!/bin/bash

# FlavorSnap Docker Run Script
# This script runs Docker containers for different environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Default values
ENVIRONMENT="development"
COMPOSE_FILE="docker-compose.yml"
DETACHED=false
REBUILD=false
SCALE_FRONTEND=1
SCALE_BACKEND=1
LOGS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--detached)
            DETACHED=true
            shift
            ;;
        -r|--rebuild)
            REBUILD=true
            shift
            ;;
        --scale-frontend)
            SCALE_FRONTEND="$2"
            shift 2
            ;;
        --scale-backend)
            SCALE_BACKEND="$2"
            shift 2
            ;;
        --logs)
            LOGS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --environment     Run environment (development|production|test) [default: development]"
            echo "  -d, --detached        Run in detached mode"
            echo "  -r, --rebuild         Rebuild images before starting"
            echo "  --scale-frontend      Number of frontend instances [default: 1]"
            echo "  --scale-backend       Number of backend instances [default: 1]"
            echo "  --logs               Show logs after starting"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
case $ENVIRONMENT in
    development|production|test)
        ;;
    *)
        print_error "Invalid environment: $ENVIRONMENT"
        print_error "Valid environments: development, production, test"
        exit 1
        ;;
esac

# Set compose file based on environment
case $ENVIRONMENT in
    development)
        COMPOSE_FILE="docker-compose.yml"
        ;;
    production)
        COMPOSE_FILE="docker-compose.prod.yml"
        ;;
    test)
        COMPOSE_FILE="docker-compose.test.yml"
        ;;
esac

# Check if compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
    print_error "Compose file not found: $COMPOSE_FILE"
    exit 1
fi

print_header "FlavorSnap Docker Runner"
print_status "Environment: $ENVIRONMENT"
print_status "Compose file: $COMPOSE_FILE"
print_status "Detached mode: $DETACHED"
print_status "Rebuild: $REBUILD"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p uploads logs test-uploads test-logs test-results

# Set environment variables
export COMPOSE_PROJECT_NAME="flavorsnap-${ENVIRONMENT}"
export COMPOSE_FILE="$COMPOSE_FILE"

# Stop existing containers if rebuilding
if [[ "$REBUILD" == "true" ]]; then
    print_status "Stopping existing containers..."
    docker-compose down --remove-orphans || true
    
    print_status "Building images..."
    ./scripts/docker_build.sh -e "$ENVIRONMENT"
fi

# Stop existing containers
print_status "Stopping existing containers..."
docker-compose down --remove-orphans || true

# Start containers
print_status "Starting containers..."

DETACHED_FLAG=""
if [[ "$DETACHED" == "true" ]]; then
    DETACHED_FLAG="-d"
fi

# Run with scaling
docker-compose up $DETACHED_FLAG \
    --scale frontend="$SCALE_FRONTEND" \
    --scale backend="$SCALE_BACKEND"

# Check container status
if [[ "$DETACHED" == "true" ]]; then
    print_status "Containers started in detached mode"
    sleep 5
    
    print_header "Container Status"
    docker-compose ps
    
    # Health check
    print_status "Performing health checks..."
    
    case $ENVIRONMENT in
        development|production)
            # Check frontend
            if curl -f http://localhost:3000 >/dev/null 2>&1; then
                print_status "✓ Frontend is healthy"
            else
                print_warning "⚠ Frontend health check failed"
            fi
            
            # Check backend
            if curl -f http://localhost:5000/health >/dev/null 2>&1; then
                print_status "✓ Backend API is healthy"
            else
                print_warning "⚠ Backend API health check failed"
            fi
            ;;
        test)
            print_status "Test environment running - check logs for test results"
            ;;
    esac
    
    # Show logs if requested
    if [[ "$LOGS" == "true" ]]; then
        print_header "Container Logs"
        docker-compose logs --tail=50
    fi
else
    print_status "Containers started. Press Ctrl+C to stop."
fi

# Show useful information
print_header "Useful Commands"
echo "Stop containers:          docker-compose down"
echo "View logs:               docker-compose logs -f"
echo "Scale services:          docker-compose up --scale frontend=3 --scale backend=2"
echo "Execute in container:    docker-compose exec frontend bash"
echo "View container status:   docker-compose ps"

case $ENVIRONMENT in
    development)
        echo ""
        echo "Development URLs:"
        echo "Frontend:               http://localhost:3000"
        echo "Backend API:            http://localhost:5000"
        echo "API Health:             http://localhost:5000/health"
        echo "API Documentation:      http://localhost:5000/docs"
        ;;
    production)
        echo ""
        echo "Production URLs:"
        echo "Application:            http://localhost"
        echo "Frontend:               http://localhost:3000"
        echo "Backend API:            http://localhost:5000"
        echo "Grafana Dashboard:      http://localhost:3001"
        echo "Prometheus:             http://localhost:9090"
        ;;
    test)
        echo ""
        echo "Test Commands:"
        echo "Run tests:               docker-compose run --rm integration-tests"
        echo "View test results:      ls -la test-results/"
        echo "Run specific test:      docker-compose run --rm backend-test python -m pytest tests/test_api.py"
        ;;
esac

print_status "Docker run script completed!"
