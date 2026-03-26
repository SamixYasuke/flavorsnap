#!/bin/bash

# FlavorSnap Docker Build Script
# This script builds Docker images for different environments

set -e

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

# Default values
ENVIRONMENT="development"
REGISTRY=""
TAG="latest"
PUSH=false
NO_CACHE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --environment   Build environment (development|production|test) [default: development]"
            echo "  -r, --registry       Docker registry prefix"
            echo "  -t, --tag           Image tag [default: latest]"
            echo "  --push             Push images to registry after build"
            echo "  --no-cache          Build without cache"
            echo "  -h, --help          Show this help message"
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

# Construct image names
if [[ -n "$REGISTRY" ]]; then
    REGISTRY="$REGISTRY/"
fi

FRONTEND_IMAGE="${REGISTRY}flavorsnap-frontend:${TAG}"
BACKEND_IMAGE="${REGISTRY}flavorsnap-backend:${TAG}"
FULL_IMAGE="${REGISTRY}flavorsnap:${TAG}"

# Build cache option
CACHE_OPTION=""
if [[ "$NO_CACHE" == "true" ]]; then
    CACHE_OPTION="--no-cache"
fi

print_status "Starting Docker build for $ENVIRONMENT environment..."
print_status "Frontend image: $FRONTEND_IMAGE"
print_status "Backend image: $BACKEND_IMAGE"

# Build based on environment
case $ENVIRONMENT in
    development)
        print_status "Building development images..."
        
        # Build frontend development image
        print_status "Building frontend development image..."
        docker build $CACHE_OPTION -f Dockerfile.frontend.dev -t "$FRONTEND_IMAGE" .
        
        # Build backend development image
        print_status "Building backend development image..."
        docker build $CACHE_OPTION -f Dockerfile.dev -t "$BACKEND_IMAGE" .
        ;;
        
    production)
        print_status "Building production image..."
        
        # Build multi-stage production image
        print_status "Building multi-stage production image..."
        docker build $CACHE_OPTION -f Dockerfile -t "$FULL_IMAGE" .
        
        # Also tag individual components
        docker tag "$FULL_IMAGE" "$FRONTEND_IMAGE"
        docker tag "$FULL_IMAGE" "$BACKEND_IMAGE"
        ;;
        
    test)
        print_status "Building test images..."
        
        # Build test image
        print_status "Building test image..."
        docker build $CACHE_OPTION -f Dockerfile.test -t "${REGISTRY}flavorsnap-test:${TAG}" .
        
        # Build E2E test image
        print_status "Building E2E test image..."
        docker build $CACHE_OPTION -f Dockerfile.e2e -t "${REGISTRY}flavorsnap-e2e:${TAG}" .
        
        # Build performance test image
        print_status "Building performance test image..."
        docker build $CACHE_OPTION -f Dockerfile.performance -t "${REGISTRY}flavorsnap-performance:${TAG}" .
        ;;
esac

print_status "Build completed successfully!"

# Show image sizes
print_status "Image sizes:"
docker images | grep flavorsnap | head -10

# Push to registry if requested
if [[ "$PUSH" == "true" ]]; then
    print_status "Pushing images to registry..."
    
    case $ENVIRONMENT in
        development|production)
            docker push "$FRONTEND_IMAGE"
            docker push "$BACKEND_IMAGE"
            if [[ "$ENVIRONMENT" == "production" ]]; then
                docker push "$FULL_IMAGE"
            fi
            ;;
        test)
            docker push "${REGISTRY}flavorsnap-test:${TAG}"
            docker push "${REGISTRY}flavorsnap-e2e:${TAG}"
            docker push "${REGISTRY}flavorsnap-performance:${TAG}"
            ;;
    esac
    
    print_status "Images pushed to registry successfully!"
fi

print_status "Docker build script completed!"
