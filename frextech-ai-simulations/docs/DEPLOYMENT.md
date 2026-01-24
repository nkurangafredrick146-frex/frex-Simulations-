
Deployment Guide

This guide provides comprehensive instructions for deploying FrexTech AI Simulations in various environments, from local development to production cloud deployments.

Table of Contents

1. Deployment Overview
2. Local Deployment
3. Docker Deployment
4. Kubernetes Deployment
5. Cloud Provider Deployment
6. High Availability Setup
7. Monitoring & Observability
8. Security Hardening
9. Backup & Disaster Recovery
10. Maintenance & Operations

Deployment Overview

Deployment Options

Environment Use Case Complexity Cost Scalability
Local Development, Testing Low $ None
Docker Staging, Small Teams Medium $$ Limited
Kubernetes Production, Enterprise High $$$ Excellent
Managed Services Quick Start, No Ops Medium $$$$ Good

System Requirements

Minimum Production Requirements

```
API Servers: 2x (4 CPU, 8GB RAM)
Workers: 2x (8 CPU, 32GB RAM, 1x NVIDIA T4 GPU)
Database: PostgreSQL (4 CPU, 16GB RAM, 100GB SSD)
Cache: Redis (2 CPU, 4GB RAM)
Storage: 1TB SSD (or S3 equivalent)
Load Balancer: 1x
```

Recommended Production

```
API Servers: 3x (8 CPU, 16GB RAM) - Auto-scaling
Workers: 4x (16 CPU, 64GB RAM, 2x NVIDIA A100 GPU)
Database: PostgreSQL HA (8 CPU, 32GB RAM, 500GB SSD)
Cache: Redis Cluster (4 CPU, 8GB RAM)
Storage: 10TB SSD + S3 for backups
CDN: Cloudflare or similar
```

Local Deployment

Single Machine Deployment

```bash
# Clone repository
git clone https://github.com/frextech/frextech-ai-simulations.git
cd frextech-ai-simulations

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install production dependencies
pip install -e ".[api,gpu]"

# Setup environment
cp .env.production.example .env
# Edit .env with your production settings

# Initialize database
python scripts/deployment/init_database.py

# Download production models
python scripts/deployment/download_models.py --production

# Start services
# API Server
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4

# Worker (in separate terminal)
celery -A src.core.world_model.training.trainer worker --loglevel=info --concurrency=2

# Start monitoring (optional)
prometheus --config.file=configs/monitoring/prometheus.yml
grafana-server --config=configs/monitoring/grafana.ini
```

Systemd Service Configuration

```ini
# /etc/systemd/system/frextech-api.service
[Unit]
Description=FrexTech AI Simulations API
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=frextech
Group=frextech
WorkingDirectory=/opt/frextech
Environment="PATH=/opt/frextech/venv/bin"
EnvironmentFile=/opt/frextech/.env
ExecStart=/opt/frextech/venv/bin/uvicorn src.api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --proxy-headers \
    --forwarded-allow-ips="*"
Restart=always
RestartSec=5
LimitNOFILE=65536
LimitNPROC=65536

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/frextech-worker.service
[Unit]
Description=FrexTech AI Worker
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=frextech
Group=frextech
WorkingDirectory=/opt/frextech
Environment="PATH=/opt/frextech/venv/bin"
EnvironmentFile=/opt/frextech/.env
ExecStart=/opt/frextech/venv/bin/celery \
    -A src.core.world_model.training.trainer worker \
    --loglevel=info \
    --concurrency=4 \
    --hostname=worker@%h \
    --queues=generation,editing,rendering
Restart=always
RestartSec=5
LimitNOFILE=65536
LimitNPROC=65536

[Install]
WantedBy=multi-user.target
```

NGINX Configuration

```nginx
# /etc/nginx/sites-available/frextech
upstream frextech_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    keepalive 32;
}

server {
    listen 80;
    server_name api.frextech-sim.com;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self';" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://frextech_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        proxy_buffering off;
        proxy_request_buffering off;
    }
    
    location /health {
        access_log off;
        proxy_pass http://frextech_api;
    }
    
    location /metrics {
        access_log off;
        proxy_pass http://frextech_api;
    }
    
    location /docs {
        proxy_pass http://frextech_api;
    }
    
    location /redoc {
        proxy_pass http://frextech_api;
    }
}
```

Docker Deployment

Production Dockerfile

```dockerfile
# docker/Dockerfile.api
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash frextech

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-api.txt .

# Create virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY pyproject.toml .

# Set permissions
RUN chown -R frextech:frextech /app
USER frextech

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Command
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Docker Compose for Production

```yaml
# docker/docker-compose.production.yml
version: '3.8'

x-common-variables: &common-variables
  ENVIRONMENT: production
  LOG_LEVEL: INFO
  PYTHONPATH: /app/src
  TZ: UTC

x-database-variables: &database-variables
  DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
  REDIS_URL: redis://redis:6379/0

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: frextech-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-frextech}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-changeme}
      POSTGRES_DB: ${POSTGRES_DB:-frextech}
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --lc-collate=C --lc-ctype=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./configs/database/postgresql.conf:/etc/postgresql/postgresql.conf
    command: >
      postgres
      -c config_file=/etc/postgresql/postgresql.conf
      -c shared_preload_libraries=pg_stat_statements
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-frextech}"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - frextech-network

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: frextech-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - frextech-network

  # API Service
  api:
    build:
      context: ../
      dockerfile: docker/Dockerfile.api
    container_name: frextech-api
    restart: unless-stopped
    environment:
      <<: [*common-variables, *database-variables]
      API_HOST: 0.0.0.0
      API_PORT: 8000
      API_WORKERS: 4
      MODEL_CACHE_DIR: /app/models/cache
      STORAGE_TYPE: s3
      S3_BUCKET: ${S3_BUCKET}
      S3_REGION: ${S3_REGION}
    volumes:
      - model_cache:/app/models/cache
      - logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "8000:8000"
    networks:
      - frextech-network
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G

  # GPU Worker
  gpu-worker:
    build:
      context: ../
      dockerfile: docker/Dockerfile.training
    container_name: frextech-gpu-worker
    restart: unless-stopped
    environment:
      <<: [*common-variables, *database-variables]
      WORKER_TYPE: gpu
      WORKER_QUEUES: generation,rendering
      CUDA_VISIBLE_DEVICES: 0
      TORCH_CUDA_ARCH_LIST: "8.6;8.9;9.0"
    volumes:
      - model_cache:/app/models/cache
      - logs:/app/logs
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - frextech-network

  # CPU Worker
  cpu-worker:
    build:
      context: ../
      dockerfile: docker/Dockerfile.training
    container_name: frextech-cpu-worker
    restart: unless-stopped
    environment:
      <<: [*common-variables, *database-variables]
      WORKER_TYPE: cpu
      WORKER_QUEUES: editing,export
    volumes:
      - model_cache:/app/models/cache
      - logs:/app/logs
    depends_on:
      - redis
    networks:
      - frextech-network

  # Celery Beat (Scheduler)
  beat:
    image: frextech-api:latest
    container_name: frextech-beat
    restart: unless-stopped
    environment:
      <<: [*common-variables, *database-variables]
    command: celery -A src.core.world_model.training.trainer beat --loglevel=info
    volumes:
      - logs:/app/logs
    depends_on:
      - redis
    networks:
      - frextech-network

  # Flower (Celery Monitor)
  flower:
    image: frextech-api:latest
    container_name: frextech-flower
    restart: unless-stopped
    environment:
      <<: [*common-variables, *database-variables]
    command: celery -A src.core.world_model.training.trainer flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
    networks:
      - frextech-network

  # Prometheus (Metrics)
  prometheus:
    image: prom/prometheus:latest
    container_name: frextech-prometheus
    restart: unless-stopped
    volumes:
      - ./configs/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    networks:
      - frextech-network

  # Grafana (Dashboards)
  grafana:
    image: grafana/grafana:latest
    container_name: frextech-grafana
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_INSTALL_PLUGINS: "grafana-piechart-panel"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./configs/monitoring/dashboards:/etc/grafana/provisioning/dashboards
      - ./configs/monitoring/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - frextech-network

  # Traefik (Reverse Proxy)
  traefik:
    image: traefik:v3.0
    container_name: frextech-traefik
    restart: unless-stopped
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--providers.docker.exposedbydefault=false"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
      - "--certificatesresolvers.myresolver.acme.email=${SSL_EMAIL}"
      - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - "/var/run/docker.sock:/var/run/docker.sock:ro"
      - traefik_data:/letsencrypt
    networks:
      - frextech-network
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`${DOMAIN}`)"
      - "traefik.http.routers.api.entrypoints=websecure"
      - "traefik.http.routers.api.tls.certresolver=myresolver"
      - "traefik.http.services.api.loadbalancer.server.port=8000"

volumes:
  postgres_data:
  redis_data:
  model_cache:
  logs:
  prometheus_data:
  grafana_data:
  traefik_data:

networks:
  frextech-network:
    driver: bridge
```

Deployment Script

```bash
#!/bin/bash
# scripts/deployment/deploy_docker.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment
ENV_FILE=".env.production"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo -e "${RED}Error: $ENV_FILE not found${NC}"
    exit 1
fi

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker if GPU is required
    if [ "$USE_GPU" = "true" ]; then
        if ! docker run --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
            log_error "NVIDIA Docker is not properly installed"
            exit 1
        fi
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    # Build API image
    docker build -f docker/Dockerfile.api -t frextech-api:latest .
    
    # Build worker image
    docker build -f docker/Dockerfile.training -t frextech-worker:latest .
    
    # Build GPU worker image
    docker build -f docker/Dockerfile.gpu -t frextech-gpu-worker:latest .
}

create_volumes() {
    log_info "Creating Docker volumes..."
    
    # Create necessary volumes
    for volume in postgres_data redis_data model_cache logs prometheus_data grafana_data; do
        if ! docker volume inspect "$volume" &> /dev/null; then
            docker volume create "$volume"
            log_info "Created volume: $volume"
        else
            log_info "Volume exists: $volume"
        fi
    done
}

migrate_database() {
    log_info "Running database migrations..."
    
    # Wait for PostgreSQL to be ready
    until docker-compose -f docker/docker-compose.production.yml exec -T postgres pg_isready -U "$POSTGRES_USER"; do
        log_info "Waiting for PostgreSQL..."
        sleep 2
    done
    
    # Run migrations
    docker-compose -f docker/docker-compose.production.yml run --rm api \
        python scripts/deployment/migrate_database.py
}

deploy_services() {
    log_info "Deploying services..."
    
    # Start services
    docker-compose -f docker/docker-compose.production.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service status
    for service in api postgres redis; do
        if docker-compose -f docker/docker-compose.production.yml ps | grep "$service" | grep -q "Up"; then
            log_info "$service is up"
        else
            log_error "$service failed to start"
            docker-compose -f docker/docker-compose.production.yml logs "$service"
            exit 1
        fi
    done
}

download_models() {
    log_info "Downloading models..."
    
    # Download production models
    docker-compose -f docker/docker-compose.production.yml run --rm api \
        python scripts/deployment/download_models.py --production
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Import Grafana dashboards
    until curl -s "http://localhost:3000/api/health" &> /dev/null; do
        log_info "Waiting for Grafana..."
        sleep 5
    done
    
    # Setup datasource
    curl -X POST "http://admin:${GRAFANA_PASSWORD:-admin}@localhost:3000/api/datasources" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://prometheus:9090",
            "access": "proxy",
            "isDefault": true
        }'
}

health_check() {
    log_info "Performing health checks..."
    
    # API health
    if curl -s "http://localhost:8000/health" | grep -q "healthy"; then
        log_info "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Database health
    if docker-compose -f docker/docker-compose.production.yml exec -T postgres \
        pg_isready -U "$POSTGRES_USER"; then
        log_info "Database health check passed"
    else
        log_error "Database health check failed"
        return 1
    fi
    
    # Redis health
    if docker-compose -f docker/docker-compose.production.yml exec -T redis \
        redis-cli ping | grep -q "PONG"; then
        log_info "Redis health check passed"
    else
        log_error "Redis health check failed"
        return 1
    fi
    
    return 0
}

main() {
    log_info "Starting FrexTech AI Simulations deployment"
    
    # Check dependencies
    check_dependencies
    
    # Build images
    build_images
    
    # Create volumes
    create_volumes
    
    # Deploy services
    deploy_services
    
    # Migrate database
    migrate_database
    
    # Download models
    download_models
    
    # Setup monitoring
    setup_monitoring
    
    # Health check
    if health_check; then
        log_info "Deployment completed successfully!"
        log_info "API: https://${DOMAIN:-localhost}"
        log_info "Grafana: http://localhost:3000"
        log_info "Flower: http://localhost:5555"
    else
        log_error "Deployment failed health checks"
        exit 1
    fi
}

# Run main function
main "$@"
```

Kubernetes Deployment

Kubernetes Manifests

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: frextech
  labels:
    name: frextech
```

```yaml
# kubernetes/configmaps/config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: frextech-config
  namespace: frextech
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_WORKERS: "4"
  MODEL_CACHE_DIR: "/app/models/cache"
  STORAGE_TYPE: "s3"
  
  # Database will be set via secrets
  # S3 config will be set via secrets
```

```yaml
# kubernetes/secrets/database.yaml
apiVersion: v1
kind: Secret
metadata:
  name: database-secrets
  namespace: frextech
type: Opaque
stringData:
  POSTGRES_USER: "frextech"
  POSTGRES_PASSWORD: "changeme"
  POSTGRES_DB: "frextech"
  DATABASE_URL: "postgresql://frextech:changeme@postgres:5432/frextech"
```

```yaml
# kubernetes/secrets/s3.yaml
apiVersion: v1
kind: Secret
metadata:
  name: s3-secrets
  namespace: frextech
type: Opaque
stringData:
  S3_BUCKET: "frextech-production"
  S3_REGION: "us-east-1"
  AWS_ACCESS_KEY_ID: "your-access-key"
  AWS_SECRET_ACCESS_KEY: "your-secret-key"
```

```yaml
# kubernetes/secrets/api.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
  namespace: frextech
type: Opaque
stringData:
  SECRET_KEY: "your-secret-key-here"
  JWT_SECRET: "your-jwt-secret"
  API_KEY_SALT: "your-api-key-salt"
```

```yaml
# kubernetes/statefulsets/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: frextech
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_USER
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: POSTGRES_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: POSTGRES_PASSWORD
        - name: POSTGRES_DB
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: POSTGRES_DB
        volumeMounts:
        - name: postgres-data
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - "$(POSTGRES_USER)"
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - "$(POSTGRES_USER)"
          initialDelaySeconds: 5
          periodSeconds: 5
  volumeClaimTemplates:
  - metadata:
      name: postgres-data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: frextech
spec:
  ports:
  - port: 5432
    targetPort: 5432
  selector:
    app: postgres
  clusterIP: None
```

```yaml
# kubernetes/deployments/api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: frextech
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: api
        image: frextech-api:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: frextech-config
              key: ENVIRONMENT
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          value: "redis://redis:6379/0"
        - name: S3_BUCKET
          valueFrom:
            secretKeyRef:
              name: s3-secrets
              key: S3_BUCKET
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: SECRET_KEY
        volumeMounts:
        - name: model-cache
          mountPath: /app/models/cache
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: logs
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: api
  namespace: frextech
spec:
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: api
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: frextech
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```yaml
# kubernetes/deployments/gpu-worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-worker
  namespace: frextech
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gpu-worker
  template:
    metadata:
      labels:
        app: gpu-worker
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-a100
      containers:
      - name: gpu-worker
        image: frextech-gpu-worker:latest
        imagePullPolicy: Always
        env:
        - name: WORKER_TYPE
          value: "gpu"
        - name: WORKER_QUEUES
          value: "generation,rendering"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secrets
              key: DATABASE_URL
        volumeMounts:
        - name: model-cache
          mountPath: /app/models/cache
        - name: shm
          mountPath: /dev/shm
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
          limits:
            memory: "64Gi"
            cpu: "16"
            nvidia.com/gpu: 1
        securityContext:
          capabilities:
            add: ["IPC_LOCK"]
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 8Gi
```

```yaml
# kubernetes/ingress/api-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  namespace: frextech
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "*"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.frextech-sim.com
    secretName: frextech-tls
  rules:
  - host: api.frextech-sim.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api
            port:
              number: 80
```

Kubernetes Deployment Script

```bash
#!/bin/bash
# scripts/deployment/deploy_k8s.sh

set -e

# Configuration
NAMESPACE="frextech"
RELEASE_NAME="frextech"
CHART_DIR="./kubernetes/helm"
VALUES_FILE="./configs/environment/production.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check NVIDIA device plugin if GPU is required
    if kubectl get ds -n kube-system nvidia-device-plugin-daemonset &> /dev/null; then
        log_info "NVIDIA device plugin is installed"
    else
        log_warn "NVIDIA device plugin is not installed. GPU workers will not work."
    fi
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        kubectl create namespace "$NAMESPACE"
        log_info "Namespace created"
    else
        log_info "Namespace already exists"
    fi
}

setup_secrets() {
    log_info "Setting up Kubernetes secrets..."
    
    # Create secrets from environment files
    kubectl create secret generic database-secrets \
        --from-file=./kubernetes/secrets/database.yaml \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic s3-secrets \
        --from-file=./kubernetes/secrets/s3.yaml \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic api-secrets \
        --from-file=./kubernetes/secrets/api.yaml \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Create docker registry secret if using private registry
    if [ -n "$DOCKER_REGISTRY" ]; then
        kubectl create secret docker-registry regcred \
            --docker-server="$DOCKER_REGISTRY" \
            --docker-username="$DOCKER_USERNAME" \
            --docker-password="$DOCKER_PASSWORD" \
            --namespace="$NAMESPACE" \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
}

setup_storage() {
    log_info "Setting up storage..."
    
    # Apply storage classes
    kubectl apply -f ./kubernetes/storage/ -n "$NAMESPACE"
    
    # Create PVCs
    kubectl apply -f ./kubernetes/persistentvolumeclaims/ -n "$NAMESPACE"
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=bound pvc -n "$NAMESPACE" --timeout=300s
}

deploy_helm() {
    log_info "Deploying with Helm..."
    
    # Add helm repos if needed
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install dependencies
    helm dependency build "$CHART_DIR"
    
    # Install/upgrade release
    helm upgrade --install "$RELEASE_NAME" "$CHART_DIR" \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE" \
        --set image.tag="$IMAGE_TAG" \
        --atomic \
        --timeout 15m \
        --wait
    
    log_info "Helm deployment completed"
}

deploy_manifests() {
    log_info "Deploying Kubernetes manifests..."
    
    # Apply all manifests
    kubectl apply -f ./kubernetes/namespace.yaml
    kubectl apply -f ./kubernetes/configmaps/ -n "$NAMESPACE"
    kubectl apply -f ./kubernetes/secrets/ -n "$NAMESPACE"
    kubectl apply -f ./kubernetes/services/ -n "$NAMESPACE"
    kubectl apply -f ./kubernetes/deployments/ -n "$NAMESPACE"
    kubectl apply -f ./kubernetes/statefulsets/ -n "$NAMESPACE"
    kubectl apply -f ./kubernetes/ingress/ -n "$NAMESPACE"
    
    # Wait for deployments
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available deployment -n "$NAMESPACE" --timeout=300s
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Install Prometheus
    helm install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values ./configs/monitoring/prometheus-values.yaml
    
    # Install Grafana
    kubectl apply -f ./kubernetes/monitoring/grafana-dashboards.yaml -n monitoring
    
    # Setup service monitors
    kubectl apply -f ./kubernetes/monitoring/service-monitors.yaml -n "$NAMESPACE"
}

migrate_database() {
    log_info "Running database migrations..."
    
    # Create a job to run migrations
    kubectl create job --namespace "$NAMESPACE" \
        --from=cronjob/"$RELEASE_NAME"-migrate migrate-"$(date +%s)" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete job -n "$NAMESPACE" \
        --selector=job-name=migrate-"$(date +%s)" \
        --timeout=300s
}

health_check() {
    log_info "Performing health checks..."
    
    # Get ingress host
    INGRESS_HOST=$(kubectl get ingress -n "$NAMESPACE" api-ingress -o jsonpath='{.spec.rules[0].host}')
    
    # API health
    if curl -s "https://$INGRESS_HOST/health" | grep -q "healthy"; then
        log_info "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    # Check pods
    PODS_NOT_RUNNING=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running 2>/dev/null | wc -l)
    if [ "$PODS_NOT_RUNNING" -gt 0 ]; then
        log_error "Some pods are not running"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
        return 1
    fi
    
    return 0
}

main() {
    log_info "Starting Kubernetes deployment for FrexTech AI Simulations"
    
    # Check prerequisites
    check_prerequisites
    
    # Create namespace
    create_namespace
    
    # Setup secrets
    setup_secrets
    
    # Setup storage
    setup_storage
    
    # Deploy using manifests
    deploy_manifests
    
    # Or deploy using Helm
    # deploy_helm
    
    # Setup monitoring
    setup_monitoring
    
    # Migrate database
    migrate_database
    
    # Health check
    if health_check; then
        log_info "Kubernetes deployment completed successfully!"
        
        # Get endpoints
        INGRESS_HOST=$(kubectl get ingress -n "$NAMESPACE" api-ingress -o jsonpath='{.spec.rules[0].host}')
        GRAFANA_URL=$(kubectl get svc -n monitoring prometheus-grafana -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        
        log_info "API: https://$INGRESS_HOST"
        log_info "Grafana: http://$GRAFANA_URL:3000"
        log_info "Kibana: http://$GRAFANA_URL:5601"
    else
        log_error "Deployment failed health checks"
        exit 1
    fi
}

# Run main function
main "$@"
```

Cloud Provider Deployment

AWS Deployment

```yaml
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  backend "s3" {
    bucket = "frextech-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "frextech-vpc"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "frextech-cluster"
  role_arn = aws_iam_role.eks_cluster.arn
  version  = "1.27"
  
  vpc_config {
    subnet_ids = [
      aws_subnet.public_1.id,
      aws_subnet.public_2.id,
      aws_subnet.private_1.id,
      aws_subnet.private_2.id
    ]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]
}

# Node Group for CPU workloads
resource "aws_eks_node_group" "cpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "cpu-node-group"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = [aws_subnet.private_1.id, aws_subnet.private_2.id]
  
  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 3
  }
  
  instance_types = ["m6i.2xlarge"]
  capacity_type  = "ON_DEMAND"
  
  labels = {
    "node-type" = "cpu"
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ec2_container_registry_read
  ]
}

# Node Group for GPU workloads
resource "aws_eks_node_group" "gpu" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-node-group"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = [aws_subnet.private_1.id, aws_subnet.private_2.id]
  
  scaling_config {
    desired_size = 2
    max_size     = 4
    min_size     = 2
  }
  
  instance_types = ["g5.2xlarge"]
  capacity_type  = "ON_DEMAND"
  
  labels = {
    "node-type"     = "gpu"
    "accelerator"   = "nvidia-tesla-a10g"
  }
  
  taint {
    key    = "accelerator"
    value  = "gpu"
    effect = "NO_SCHEDULE"
  }
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.ec2_container_registry_read
  ]
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier           = "frextech-postgres"
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.m6g.2xlarge"
  allocated_storage    = 500
  storage_type         = "gp3"
  storage_encrypted    = true
  kms_key_id           = aws_kms_key.database.arn
  
  db_name              = "frextech"
  username             = var.db_username
  password             = var.db_password
  
  vpc_security_group_ids = [aws_security_group.database.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"
  
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "frextech-postgres-final"
  
  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  tags = {
    Name = "frextech-postgres"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "frextech-redis"
  engine              = "redis"
  node_type           = "cache.r6g.large"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379
  subnet_group_name   = aws_elasticache_subnet_group.main.name
  security_group_ids  = [aws_security_group.redis.id]
  
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-04:00"
  
  tags = {
    Name = "frextech-redis"
  }
}

# S3 Bucket for models and storage
resource "aws_s3_bucket" "models" {
  bucket = "frextech-models-${var.environment}"
  
  tags = {
    Name = "frextech-models"
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ALB for API
resource "aws_lb" "api" {
  name               = "frextech-api"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_1.id, aws_subnet.public_2.id]
  
  enable_deletion_protection = true
  
  tags = {
    Name = "frextech-api"
  }
}

resource "aws_lb_listener" "api_https" {
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.api.arn
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
```

GCP Deployment

```yaml
# gcp/deployment.yaml
# Google Kubernetes Engine Cluster
resource "google_container_cluster" "primary" {
  name     = "frextech-cluster"
  location = "us-central1"
  
  # We can't create a cluster with no node pool defined, but we want to use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.main.self_link
  subnetwork = google_compute_subnetwork.primary.self_link
  
  # Enable workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Enable vertical pod autoscaling
  vertical_pod_autoscaling {
    enabled = true
  }
  
  # Enable cluster autoscaling
  cluster_autoscaling {
    enabled = true
    
    resource_limits {
      resource_type = "cpu"
      minimum = 2
      maximum = 100
    }
    
    resource_limits {
      resource_type = "memory"
      minimum = 4
      maximum = 400
    }
    
    auto_provisioning_defaults {
      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform"
      ]
    }
  }
  
  release_channel {
    channel = "REGULAR"
  }
  
  maintenance_policy {
    recurring_window {
      start_time = "2023-01-01T02:00:00Z"
      end_time   = "2023-01-01T06:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }
}

# GPU Node Pool
resource "google_container_node_pool" "gpu_nodes" {
  name       = "gpu-node-pool"
  location   = "us-central1"
  cluster    = google_container_cluster.primary.name
  node_count = 2
  
  node_config {
    machine_type = "n1-standard-8"
    
    # GPU Configuration
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
    
    labels = {
      "accelerator" = "nvidia-tesla-t4"
    }
    
    taint {
      key    = "accelerator"
      value  = "gpu"
      effect = "NO_SCHEDULE"
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    # Enable workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
  
  autoscaling {
    min_node_count = 2
    max_node_count = 4
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Cloud SQL PostgreSQL
resource "google_sql_database_instance" "postgres" {
  name             = "frextech-postgres"
  database_version = "POSTGRES_15"
  region           = "us-central1"
  
  settings {
    tier = "db-custom-4-16384"
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.main.self_link
    }
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
    }
    
    maintenance_window {
      day  = 7
      hour = 4
    }
    
    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "on"
    }
  }
  
  deletion_protection = true
}

# Memorystore Redis
resource "google_redis_instance" "cache" {
  name           = "frextech-redis"
  tier           = "STANDARD_HA"
  memory_size_gb = 4
  
  region                  = "us-central1"
  location_id            = "us-central1-a"
  alternative_location_id = "us-central1-f"
  
  authorized_network = google_compute_network.main.self_link
  
  redis_version = "REDIS_7_0"
  display_name  = "FrexTech Redis Cache"
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 4
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
}

# Cloud Storage Bucket
resource "google_storage_bucket" "models" {
  name          = "frextech-models-${var.environment}"
  location      = "US"
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.bucket.self_link
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
}
```

Azure Deployment

```yaml
# azure/main.bicep
param location string = 'eastus'
param environment string = 'production'

resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: 'frextech-vnet'
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: 'aks-subnet'
        properties: {
          addressPrefix: '10.0.1.0/24'
        }
      }
      {
        name: 'postgres-subnet'
        properties: {
          addressPrefix: '10.0.2.0/24'
          delegations: [
            {
              name: 'postgresDelegation'
              properties: {
                serviceName: 'Microsoft.DBforPostgreSQL/flexibleServers'
              }
            }
          ]
        }
      }
    ]
  }
}

resource aks 'Microsoft.ContainerService/managedClusters@2023-05-02' = {
  name: 'frextech-aks'
  location: location
  sku: {
    name: 'Base'
    tier: 'Standard'
  }
  properties: {
    dnsPrefix: 'frextech'
    agentPoolProfiles: [
      {
        name: 'systempool'
        count: 3
        vmSize: 'Standard_D4s_v3'
        osType: 'Linux'
        mode: 'System'
      }
      {
        name: 'gpunodepool'
        count: 2
        vmSize: 'Standard_NC6s_v3'
        osType: 'Linux'
        mode: 'User'
        nodeLabels: {
          'accelerator': 'nvidia-tesla-v100'
        }
        taints: [
          'accelerator=gpu:NoSchedule'
        ]
      }
    ]
    networkProfile: {
      networkPlugin: 'azure'
      serviceCidr: '10.1.0.0/16'
      dnsServiceIP: '10.1.0.10'
    }
    oidcIssuerProfile: {
      enabled: true
    }
    workloadAutoScalerProfile: {
      verticalPodAutoscaler: {
        controlledValues: 'RequestsAndLimits'
        enabled: true
        updateMode: 'Auto'
      }
    }
  }
}

resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2023-06-01-preview' = {
  name: 'frextech-postgres'
  location: location
  sku: {
    name: 'Standard_D4s_v3'
    tier: 'GeneralPurpose'
  }
  properties: {
    administratorLogin: 'frextech'
    administratorLoginPassword: adminPassword
    version: '15'
    storage: {
      storageSizeGB: 512
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Enabled'
    }
    highAvailability: {
      mode: 'ZoneRedundant'
    }
    network: {
      delegatedSubnetResourceId: vnet::subnets[1].id
    }
  }
}

resource redis 'Microsoft.Cache/redis@2023-08-01' = {
  name: 'frextech-redis'
  location: location
  properties: {
    sku: {
      name: 'Premium'
      family: 'P'
      capacity: 1
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    redisConfiguration: {
      maxmemoryPolicy: 'allkeys-lru'
    }
    replicasPerMaster: 2
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'frextechstorage${uniqueString(resourceGroup().id)}'
  location: location
  sku: {
    name: 'Standard_RAGRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    networkAcls: {
      bypass: 'AzureServices'
      defaultAction: 'Deny'
      virtualNetworkRules: [
        {
          id: vnet.id
          action: 'Allow'
        }
      ]
      ipRules: []
    }
  }
}
```

High Availability Setup

Multi-Region Deployment

```yaml
# kubernetes/multiregion/values.yaml
global:
  region: us-east-1
  replicaCount: 3
  
api:
  replicaCount: 3
  autoscaling:
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
  
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - api
        topologyKey: topology.kubernetes.io/zone

database:
  postgresql:
    architecture: replication
    primary:
      persistence:
        size: 100Gi
      nodeSelector:
        topology.kubernetes.io/zone: us-east-1a
    readReplicas:
      - name: replica-1
        replicaCount: 2
        persistence:
          size: 100Gi
        nodeSelector:
          topology.kubernetes.io/zone: us-east-1b

redis:
  architecture: replication
  sentinel:
    enabled: true
  master:
    persistence:
      enabled: true
      size: 8Gi
  replica:
    replicaCount: 2
    persistence:
      enabled: true
      size: 8Gi

ingress:
  enabled: true
  className: alb
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/healthcheck-port: traffic-port
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: "15"
    alb.ingress.kubernetes.io/healthcheck-timeout-seconds: "5"
    alb.ingress.kubernetes.io/success-codes: "200"
    alb.ingress.kubernetes.io/healthy-threshold-count: "2"
    alb.ingress.kubernetes.io/unhealthy-threshold-count: "2"
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS":443}]'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-east-1:123456789012:certificate/abc123
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS13-1-2-2021-06
    alb.ingress.kubernetes.io/load-balancer-attributes: routing.http2.enabled=true
    alb.ingress.kubernetes.io/tags: Environment=production,Project=frextech
  
  hosts:
    - host: api.frextech-sim.com
      paths:
        - path: /
          pathType: Prefix

monitoring:
  prometheus:
    enabled: true
    retention: 30d
    alertmanager:
      enabled: true
    pushgateway:
      enabled: false
  
  grafana:
    enabled: true
    adminPassword: secret
    persistence:
      enabled: true
      size: 10Gi
    dashboardProviders:
      dashboardproviders.yaml:
        apiVersion: 1
        providers:
        - name: 'default'
          orgId: 1
          folder: ''
          type: file
          disableDeletion: false
          editable: true
          options:
            path: /var/lib/grafana/dashboards/default

backup:
  enabled: true
  schedule: "0 2 * * *"
  retention: 30
  s3:
    bucket: frextech-backups
    region: us-east-1
```

Service Mesh Configuration

```yaml
# kubernetes/service-mesh/istio.yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  namespace: istio-system
spec:
  profile: default
  components:
    egressGateways:
    - name: istio-egressgateway
      enabled: true
    ingressGateways:
    - name: istio-ingressgateway
      enabled: true
      k8s:
        serviceAnnotations:
          service.beta.kubernetes.io/aws-load-balancer-type: nlb
          service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 2000m
            memory: 1024Mi
        hpaSpec:
          minReplicas: 2
          maxReplicas: 5
          metrics:
          - type: Resource
            resource:
              name: cpu
              target:
                type: Utilization
                averageUtilization: 80
  meshConfig:
    enableTracing: true
    defaultConfig:
      holdApplicationUntilProxyStarts: true
      tracing:
        sampling: 100
        zipkin:
          address: zipkin.istio-system:9411
    accessLogFile: /dev/stdout
    accessLogEncoding: JSON
    enableAutoMtls: true
  values:
    global:
      proxy:
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 2000m
            memory: 1024Mi
      meshID: mesh1
      multiCluster:
        clusterName: us-east-1
      network: network1
    pilot:
      autoscaleEnabled: true
      autoscaleMin: 2
      autoscaleMax: 5
    gateways:
      istio-ingressgateway:
        type: LoadBalancer
        sds:
          enabled: true
---
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: frextech-gateway
  namespace: frextech
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "api.frextech-sim.com"
    tls:
      httpsRedirect: true
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: frextech-certificate
    hosts:
    - "api.frextech-sim.com"
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: frextech-api
  namespace: frextech
spec:
  hosts:
  - "api.frextech-sim.com"
  gateways:
  - frextech-gateway
  http:
  - match:
    - uri:
        prefix: /
    route:
    - destination:
        host: api
        port:
          number: 80
    timeout: 300s
    retries:
      attempts: 3
      retryOn: "gateway-error,connect-failure,refused-stream"
    corsPolicy:
      allowOrigins:
      - "*"
      allowMethods:
      - GET
      - POST
      - PUT
      - DELETE
      - OPTIONS
      allowHeaders:
      - "*"
    fault:
      abort:
        percentage:
          value: 0.1
        httpStatus: 500
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: api
  namespace: frextech
spec:
  host: api
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 30ms
      http:
        http2MaxRequests: 1000
        maxRequestsPerConnection: 10
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

Monitoring & Observability

Prometheus Configuration

```yaml
# configs/monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'frextech-production'
    environment: 'production'

rule_files:
  - 'alerts/*.yml'
  - 'rules/*.yml'

scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  # API Service
  - job_name: 'api'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
  
  # Worker metrics
  - job_name: 'workers'
    static_configs:
      - targets:
        - 'worker:8001'
        - 'gpu-worker:8001'
    metrics_path: '/metrics'
  
  # Node metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
  
  # Redis
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
  
  # PostgreSQL
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
  
  # GPU metrics
  - job_name: 'gpu'
    static_configs:
      - targets: ['dcgm-exporter:9400']
  
  # Blackbox exporter
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
        - https://api.frextech-sim.com/health
        - https://api.frextech-sim.com/docs
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - 'alertmanager:9093'
```

Alert Rules

```yaml
# configs/monitoring/alerts/api.yml
groups:
  - name: api
    rules:
      - alert: APIHighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100 > 5
        for: 2m
        labels:
          severity: warning
          team: api
        annotations:
          summary: "High error rate on API"
          description: "API error rate is {{ $value }}% (threshold: 5%)"
          runbook_url: "https://runbooks.frextech-sim.com/api-high-error-rate"
      
      - alert: APIHighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
          team: api
        annotations:
          summary: "High latency on API"
          description: "95th percentile latency is {{ $value }}s (threshold: 2s)"
      
      - alert: APIDown
        expr: up{job="api"} == 0
        for: 1m
        labels:
          severity: critical
          team: api
        annotations:
          summary: "API is down"
          description: "API instance {{ $labels.instance }} is down"
      
      - alert: APIHighMemoryUsage
        expr: (container_memory_working_set_bytes{container="api"} / container_spec_memory_limit_bytes{container="api"}) * 100 > 80
        for: 5m
        labels:
          severity: warning
          team: api
        annotations:
          summary: "High memory usage on API"
          description: "Memory usage is {{ $value }}% (threshold: 80%)"
  
  - name: database
    rules:
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
          team: database
        annotations:
          summary: "Database is down"
          description: "PostgreSQL instance {{ $labels.instance }} is down"
      
      - alert: DatabaseHighConnections
        expr: pg_stat_database_numbackends{datname="frextech"} > 100
        for: 5m
        labels:
          severity: warning
          team: database
        annotations:
          summary: "High database connections"
          description: "Database connections are {{ $value }} (threshold: 100)"
      
      - alert: DatabaseHighDiskUsage
        expr: (pg_database_size_bytes{datname="frextech"} / pg_database_size_limit_bytes{datname="frextech"}) * 100 > 80
        for: 10m
        labels:
          severity: warning
          team: database
        annotations:
          summary: "High database disk usage"
          description: "Database disk usage is {{ $value }}% (threshold: 80%)"
  
  - name: gpu
    rules:
      - alert: GPUOutOfMemory
        expr: DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100 > 90
        for: 2m
        labels:
          severity: critical
          team: gpu
        annotations:
          summary: "GPU out of memory"
          description: "GPU {{ $labels.gpu }} memory usage is {{ $value }}%"
      
      - alert: GPUHighTemperature
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 5m
        labels:
          severity: warning
          team: gpu
        annotations:
          summary: "GPU high temperature"
          description: "GPU {{ $labels.gpu }} temperature is {{ $value }}°C"
      
      - alert: GPUUtilizationHigh
        expr: DCGM_FI_DEV_GPU_UTIL > 95
        for: 10m
        labels:
          severity: warning
          team: gpu
        annotations:
          summary: "GPU high utilization"
          description: "GPU {{ $labels.gpu }} utilization is {{ $value }}%"
```

Grafana Dashboards

```json
{
  "dashboard": {
    "title": "FrexTech AI Simulations - Production",
    "tags": ["frextech", "production", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Requests",
        "type": "graph",
        "targets": [{
          "expr": "rate(http_requests_total[5m])",
          "legendFormat": "{{method}} {{status}} {{endpoint}}"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
          "legendFormat": "{{endpoint}}"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [{
          "expr": "DCGM_FI_DEV_GPU_UTIL",
          "legendFormat": "GPU {{gpu}}"
        }],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
      },
      {
        "title": "GPU Memory",
        "type": "graph",
        "targets": [{
          "expr": "DCGM_FI_DEV_FB_USED / 1024 / 1024 / 1024",
          "legendFormat": "GPU {{gpu}}"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12}
      },
      {
        "title": "Queue Length",
        "type": "graph",
        "targets": [{
          "expr": "celery_queue_length",
          "legendFormat": "{{queue}}"
        }],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12}
      },
      {
        "title": "System Resources",
        "type": "stat",
        "targets": [
          {"expr": "sum(container_memory_working_set_bytes{container!=\"\"}) / 1024 / 1024 / 1024", "legendFormat": "Memory Used"},
          {"expr": "sum(rate(container_cpu_usage_seconds_total[5m])) * 100", "legendFormat": "CPU Used"},
          {"expr": "sum(container_fs_usage_bytes{container!=\"\"}) / 1024 / 1024 / 1024", "legendFormat": "Disk Used"}
        ],
        "gridPos": {"h": 4, "w": 24, "x": 0, "y": 20}
      }
    ]
  }
}
```

Security Hardening

Security Checklist

```yaml
# security/checklist.yaml
network_security:
  - vpc_peering_disabled: true
  - default_security_group_restricted: true
  - flow_logs_enabled: true
  - network_acls_configured: true
  
encryption:
  - encryption_at_rest: true
  - encryption_in_transit: true
  - tls_1_2_minimum: true
  - kms_key_rotation: true
  
access_control:
  - mfa_enabled: true
  - root_account_restricted: true
  - iam_password_policy: true
  - least_privilege_principle: true
  
monitoring:
  - cloudtrail_enabled: true
  - config_enabled: true
  - guardduty_enabled: true
  - security_hub_enabled: true
  
container_security:
  - image_scanning: true
  - runtime_security: true
  - pod_security_policies: true
  - network_policies: true
  
compliance:
  - soc2: true
  - iso27001: true
  - gdpr: true
  - hipaa: false
```

Kubernetes Security Context

```yaml
# kubernetes/security/security-context.yaml
apiVersion: v1
kind: PodSecurityPolicy
metadata:
  name: restricted
  annotations:
    seccomp.security.alpha.kubernetes.io/allowedProfileNames: 'runtime/default'
    seccomp.security.alpha.kubernetes.io/defaultProfileName: 'runtime/default'
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  supplementalGroups:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
---
apiVersion: v1
kind: SecurityContextConstraints
metadata:
  name: restricted
allowPrivilegedContainer: false
allowedCapabilities: []
defaultAddCapabilities: []
fsGroup:
  type: MustRunAs
  ranges:
  - min: 1
    max: 65535
runAsUser:
  type: MustRunAsNonRoot
seLinuxContext:
  type: MustRunAs
supplementalGroups:
  type: MustRunAs
  ranges:
  - min: 1
    max: 65535
users: []
groups: []
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: frextech
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frextech
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: frextech
    ports:
    - protocol: TCP
      port: 5432
    - protocol: TCP
      port: 6379
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

Backup & Disaster Recovery

Backup Strategy

```yaml
# configs/backup/strategy.yaml
backup:
  schedule:
    database: "0 2 * * *"        # Daily at 2 AM
    redis: "0 3 * * *"           # Daily at 3 AM
    models: "0 4 * * *"          # Daily at 4 AM
    user_data: "*/30 * * * *"    # Every 30 minutes
  
  retention:
    daily: 7
    weekly: 4
    monthly: 12
    yearly: 3
  
  storage:
    local: false
    s3:
      bucket: frextech-backups
      region: us-east-1
      storage_class: STANDARD_IA
      lifecycle:
        transition_to_glacier: 30
        expiration: 365
  
  recovery:
    rto: "4 hours"      # Recovery Time Objective
    rpo: "15 minutes"   # Recovery Point Objective
    
    procedures:
      - name: "database_restore"
        steps:
          - "Stop API services"
          - "Restore PostgreSQL from latest backup"
          - "Run database migrations"
          - "Verify data integrity"
          - "Restart API services"
        
      - name: "full_restore"
        steps:
          - "Provision new infrastructure"
          - "Restore all backups"
          - "Update DNS records"
          - "Run health checks"
          - "Switch traffic to new environment"
```

Database Backup Script

```python
#!/usr/bin/env python3
# scripts/backup/database_backup.py

import os
import boto3
import subprocess
import datetime
import logging
from pathlib import Path

class DatabaseBackup:
    def __init__(self):
        self.setup_logging()
        self.setup_clients()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/var/log/frextech/backup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_clients(self):
        self.s3 = boto3.client('s3')
        self.ssm = boto3.client('ssm')
        
        # Get configuration from environment or SSM
        self.backup_bucket = os.getenv('BACKUP_BUCKET', 'frextech-backups')
        self.database_url = os.getenv('DATABASE_URL')
        
    def create_backup(self):
        """Create database backup."""
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = f'database_backup_{timestamp}.dump'
            backup_path = f'/tmp/{backup_file}'
            
            # Create backup using pg_dump
            self.logger.info(f'Creating database backup: {backup_file}')
            
            cmd = [
                'pg_dump',
                '-Fc',  # Custom format
                '-v',   # Verbose
                '-f', backup_path,
                self.database_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f'Backup failed: {result.stderr}')
            
            # Upload to S3
            self.upload_to_s3(backup_path, backup_file)
            
            # Cleanup local file
            os.remove(backup_path)
            
            self.logger.info('Database backup completed successfully')
            
            return backup_file
            
        except Exception as e:
            self.logger.error(f'Backup failed: {e}')
            raise
    
    def upload_to_s3(self, local_path, s3_key):
        """Upload backup to S3."""
        try:
            self.logger.info(f'Uploading to S3: {s3_key}')
            
            self.s3.upload_file(
                local_path,
                self.backup_bucket,
                f'database/{s3_key}',
                ExtraArgs={
                    'StorageClass': 'STANDARD_IA',
                    'Metadata': {
                        'backup-type': 'database',
                        'environment': 'production'
                    }
                }
            )
            
            # Add lifecycle rule
            self.add_lifecycle_rule(s3_key)
            
        except Exception as e:
            self.logger.error(f'S3 upload failed: {e}')
            raise
    
    def add_lifecycle_rule(self, s3_key):
        """Add lifecycle rule for backup."""
        # In production, use S3 lifecycle configuration
        pass
    
    def restore_backup(self, backup_file):
        """Restore database from backup."""
        try:
            self.logger.info(f'Restoring from backup: {backup_file}')
            
            # Download from S3
            local_path = f'/tmp/{backup_file}'
            self.download_from_s3(backup_file, local_path)
            
            # Restore database
            cmd = [
                'pg_restore',
                '-v',
                '--clean',
                '--if-exists',
                '--dbname', self.database_url,
                local_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f'Restore failed: {result.stderr}')
            
            # Cleanup
            os.remove(local_path)
            
            self.logger.info('Database restore completed successfully')
            
        except Exception as e:
            self.logger.error(f'Restore failed: {e}')
            raise
    
    def download_from_s3(self, s3_key, local_path):
        """Download backup from S3."""
        try:
            self.s3.download_file(
                self.backup_bucket,
                f'database/{s3_key}',
                local_path
            )
        except Exception as e:
            self.logger.error(f'S3 download failed: {e}')
            raise
    
    def list_backups(self):
        """List all available backups."""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.backup_bucket,
                Prefix='database/'
            )
            
            backups = []
            for obj in response.get('Contents', []):
                backups.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
            
            return backups
            
        except Exception as e:
            self.logger.error(f'Failed to list backups: {e}')
            raise
    
    def cleanup_old_backups(self, days_to_keep=30):
        """Cleanup old backups."""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            
            backups = self.list_backups()
            for backup in backups:
                if backup['last_modified'] < cutoff_date:
                    self.logger.info(f'Deleting old backup: {backup["key"]}')
                    self.s3.delete_object(
                        Bucket=self.backup_bucket,
                        Key=backup['key']
                    )
                    
        except Exception as e:
            self.logger.error(f'Cleanup failed: {e}')
            raise

def main():
    backup = DatabaseBackup()
    
    # Create backup
    backup_file = backup.create_backup()
    
    # List backups
    backups = backup.list_backups()
    print(f'Total backups: {len(backups)}')
    
    # Cleanup old backups
    backup.cleanup_old_backups(days_to_keep=30)

if __name__ == '__main__':
    main()
```

Maintenance & Operations

Daily Operations Checklist

```yaml
# operations/daily-checklist.yaml
daily_tasks:
  - name: "system_health_check"
    schedule: "08:00"
    actions:
      - "Check all service health endpoints"
      - "Review error logs from last 24 hours"
      - "Check disk usage on all nodes"
      - "Verify backup completion"
      - "Check certificate expiration dates"
  
  - name: "performance_review"
    schedule: "12:00"
    actions:
      - "Review API response times"
      - "Check queue lengths"
      - "Monitor GPU utilization"
      - "Review memory usage"
      - "Check database connection pool"
  
  - name: "security_review"
    schedule: "16:00"
    actions:
      - "Review security logs"
      - "Check for failed login attempts"
      - "Verify no unauthorized API keys"
      - "Review network traffic patterns"
      - "Check for security updates"
  
  - name: "capacity_planning"
    schedule: "weekly"
    actions:
      - "Review resource usage trends"
      - "Plan for upcoming capacity needs"
      - "Review auto-scaling metrics"
      - "Check storage growth"
      - "Update capacity forecasts"

alerts_to_monitor:
  critical:
    - "APIHighErrorRate"
    - "APIDown"
    - "DatabaseDown"
    - "GPUOutOfMemory"
  
  warning:
    - "APIHighLatency"
    - "DatabaseHighConnections"
    - "HighMemoryUsage"
    - "HighDiskUsage"

incident_response:
  steps:
    - "Acknowledge alert within 5 minutes"
    - "Assess impact and scope"
    - "Communicate with stakeholders"
    - "Execute runbook procedures"
    - "Document incident and resolution"
    - "Conduct post-mortem analysis"
```

Performance Tuning Guide

```yaml
# operations/performance-tuning.yaml
api_tuning:
  gunicorn_workers: "2 * cores + 1"
  worker_class: "uvicorn.workers.UvicornWorker"
  timeout: 300
  keepalive: 2
  max_requests: 1000
  max_requests_jitter: 50
  
  database_pool:
    pool_size: 20
    max_overflow: 40
    pool_recycle: 3600
    pool_pre_ping: true
  
  redis_pool:
    max_connections: 50
    socket_keepalive: true
    retry_on_timeout: true

gpu_tuning:
  mixed_precision: true
  cudnn_benchmark: true
  tf32_enabled: true
  
  memory:
    max_split_size_mb: 256
    pinned_memory: true
    allow_tf32: true
  
  batch_size:
    generation: 1
    rendering: 2
    training: 4

database_tuning:
  postgresql_conf:
    shared_buffers: "25% of RAM"
    effective_cache_size: "75% of RAM"
    maintenance_work_mem: "1GB"
    checkpoint_completion_target: 0.9
    wal_buffers: "16MB"
    default_statistics_target: 100
    random_page_cost: 1.1
    effective_io_concurrency: 200
    work_mem: "4MB"
    min_wal_size: "1GB"
    max_wal_size: "4GB"
    max_worker_processes: 8
    max_parallel_workers_per_gather: 4
    max_parallel_workers: 8
    max_parallel_maintenance_workers: 4

caching_strategy:
  model_cache:
    ttl: "24 hours"
    max_size: "50GB"
    strategy: "LRU"
  
  response_cache:
    ttl: "1 hour"
    max_size: "10GB"
    strategy: "LFU"
  
  user_cache:
    ttl: "30 minutes"
    max_size: "5GB"
    strategy: "TTL"
```

Rollback Procedures

```bash
#!/bin/bash
# scripts/deployment/rollback.sh

set -e

# Configuration
ENVIRONMENT=${1:-production}
ROLLBACK_VERSION=${2:-previous}
BACKUP_DIR="/backups/frextech"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

stop_services() {
    log "Stopping services..."
    
    # Stop API services
    if systemctl is-active --quiet frextech-api; then
        systemctl stop frextech-api
        log "API service stopped"
    fi
    
    # Stop workers
    if systemctl is-active --quiet frextech-worker; then
        systemctl stop frextech-worker
        log "Worker service stopped"
    fi
    
    # Stop GPU workers
    if systemctl is-active --quiet frextech-gpu-worker; then
        systemctl stop frextech-gpu-worker
        log "GPU worker service stopped"
    fi
}

restore_database() {
    log "Restoring database..."
    
    # Find latest backup
    BACKUP_FILE=$(ls -t "${BACKUP_DIR}/database/" | head -1)
    
    if [ -z "$BACKUP_FILE" ]; then
        error "No backup file found"
        return 1
    fi
    
    log "Using backup: $BACKUP_FILE"
    
    # Restore database
    pg_restore \
        --clean \
        --if-exists \
        --dbname="$DATABASE_URL" \
        "${BACKUP_DIR}/database/${BACKUP_FILE}"
    
    if [ $? -eq 0 ]; then
        log "Database restored successfully"
    else
        error "Database restore failed"
        return 1
    fi
}

restore_models() {
    log "Restoring models..."
    
    # Clear current models
    rm -rf /app/models/cache/*
    
    # Restore from backup
    cp -r "${BACKUP_DIR}/models/"* /app/models/cache/
    
    log "Models restored"
}

restore_configuration() {
    log "Restoring configuration..."
    
    # Restore environment file
    cp "${BACKUP_DIR}/config/.env" /app/.env
    
    # Restore application configuration
    cp -r "${BACKUP_DIR}/config/configs/"* /app/configs/
    
    log "Configuration restored"
}

start_services() {
    log "Starting services..."
    
    # Start services in order
    systemctl start frextech-api
    systemctl start frextech-worker
    systemctl start frextech-gpu-worker
    
    log "Services started"
}

health_check() {
    log "Performing health checks..."
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    if curl -s "http://localhost:8000/health" | grep -q "healthy"; then
        log "API health check passed"
    else
        error "API health check failed"
        return 1
    fi
    
    # Check worker health
    if systemctl is-active --quiet frextech-worker; then
        log "Worker health check passed"
    else
        error "Worker health check failed"
        return 1
    fi
    
    return 0
}

main() {
    log "Starting rollback procedure for $ENVIRONMENT"
    
    # Validate environment
    if [ "$ENVIRONMENT" != "production" ] && [ "$ENVIRONMENT" != "staging" ]; then
        error "Invalid environment: $ENVIRONMENT"
        exit 1
    fi
    
    # Confirm rollback
    read -p "Are you sure you want to rollback $ENVIRONMENT? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        log "Rollback cancelled"
        exit 0
    fi
    
    # Execute rollback steps
    stop_services
    restore_database
    restore_models
    restore_configuration
    start_services
    
    if health_check; then
        log "Rollback completed successfully!"
    else
        error "Rollback failed health checks"
        exit 1
    fi
}

# Load environment
if [ -f /app/.env ]; then
    set -a
    source /app/.env
    set +a
fi

# Run main function
main "$@"
```

---

This deployment guide is maintained by the FrexTech Operations Team.
Last Updated: January 1, 2024
Version: 3.0

For operational support:

· 24/7 On-call: +1-555-123-4567
· Slack: #production-support
· Email: operations@frextech-sim.com
· Status Page: https://status.frextech-sim.com