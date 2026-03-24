# Multi-stage production Dockerfile for FlavorSnap
# Stage 1: Build frontend assets
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY frontend/package*.json ./
RUN npm ci --only=production

# Copy frontend source code
COPY frontend/ ./

# Build frontend application
RUN npm run build

# Stage 2: Build Python environment with ML model
FROM python:3.9-slim AS backend-builder

WORKDIR /app

# Install system dependencies for ML
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY ml-model-api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML model and application code
COPY model.pth ./
COPY ml-model-api/ ./

# Stage 3: Production runtime
FROM python:3.9-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=backend-builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend-runner /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=backend-builder /app/model.pth ./
COPY --from=backend-builder /app/ml-model-api/ ./ml-model-api/

# Copy frontend build
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next/
COPY --from=frontend-builder /app/frontend/public ./frontend/public/
COPY --from=frontend-builder /app/frontend/node_modules ./frontend/node_modules/
COPY --from=frontend-builder /app/frontend/package.json ./frontend/

# Configure nginx
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Create necessary directories
RUN mkdir -p /app/uploads /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 3000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health && curl -f http://localhost:3000 || exit 1

# Start script
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
