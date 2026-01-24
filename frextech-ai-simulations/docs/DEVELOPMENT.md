
Development Guide

This guide provides comprehensive instructions for setting up and working with the FrexTech AI Simulations development environment. Whether you're contributing to the codebase or using it for research, this document will help you get started.

Table of Contents

1. Development Environments
2. Local Setup
3. Containerized Development
4. Remote Development
5. IDE Configuration
6. Development Workflow
7. Testing and Debugging
8. Performance Optimization
9. Troubleshooting
10. Advanced Development

Development Environments

Environment Options

We support multiple development environments to suit different needs:

Environment Best For Setup Time Performance GPU Access
Local Daily development, debugging Medium Best Direct
Docker Consistent environment, CI/CD Fast Good Limited
Dev Container VS Code, cloud development Fast Good With GPU passthrough
Remote Server Heavy training, large models Fast Excellent Full

Hardware Requirements

Minimum Requirements

· CPU: 8 cores (Intel i7 or AMD Ryzen 7)
· RAM: 32 GB
· Storage: 100 GB SSD
· GPU: NVIDIA RTX 3060 (8 GB VRAM)
· OS: Ubuntu 22.04, Windows 11 WSL2, or macOS 13+

Recommended

· CPU: 16+ cores (Intel i9 or AMD Ryzen 9)
· RAM: 64+ GB
· Storage: 1 TB NVMe SSD
· GPU: NVIDIA RTX 4090 (24 GB VRAM) or A100 (40/80 GB)
· OS: Ubuntu 22.04 LTS

Cloud Development

· AWS: g5.2xlarge (1x A10G) or p4d.24xlarge (8x A100)
· GCP: a2-highgpu-1g (1x A100) or a2-ultragpu-1g (1x A100 80GB)
· Azure: NCas_T4_v3 (1x T4) or ND96amsr_A100_v4 (8x A100)

Local Setup

Step 1: System Dependencies

Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    cmake \
    ninja-build \
    libssl-dev \
    libffi-dev \
    libopenblas-dev \
    libopenmpi-dev \
    openmpi-bin \
    ocl-icd-opencl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# NVIDIA drivers (if using GPU)
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

macOS

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install \
    python@3.10 \
    cmake \
    ninja \
    git \
    curl \
    wget \
    openssl \
    libffi

# For M1/M2 Macs with GPU
brew install --cask mambaforge
conda init "$(basename "${SHELL}")"
```

Windows (WSL2)

```powershell
# Install WSL2
wsl --install -d Ubuntu-22.04

# In WSL2 Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip

# Install NVIDIA CUDA in WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-2
```

Step 2: Python Environment

```bash
# Clone repository
git clone https://github.com/frextech/frextech-ai-simulations.git
cd frextech-ai-simulations

# Create and activate virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install base dependencies
pip install -e ".[dev]"

# Install GPU dependencies (if available)
if [ -x "$(command -v nvidia-smi)" ]; then
    pip install -e ".[gpu]"
fi

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

Step 3: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

Example .env file:

```env
# Application Settings
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=DEBUG

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=True

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/frextech
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=50

# Storage
STORAGE_TYPE=local  # local, s3, gcs, azure
STORAGE_PATH=./storage
S3_BUCKET=frextech-development
S3_REGION=us-east-1

# Authentication
SECRET_KEY=development-secret-key-change-in-production
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60
API_KEY_SALT=development-salt

# ML/AI
MODEL_CACHE_DIR=./models/cache
MODEL_DOWNLOAD_TIMEOUT=300
TORCH_HOME=./models/torch

# GPU Settings
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"  # RTX 30/40 series

# Monitoring
SENTRY_DSN=
DATADOG_API_KEY=
PROMETHEUS_PORT=9090
```

Step 4: Database Setup

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql -c "CREATE USER frextech WITH PASSWORD 'frextech123';"
sudo -u postgres psql -c "CREATE DATABASE frextech OWNER frextech;"
sudo -u postgres psql -c "ALTER USER frextech WITH SUPERUSER;"

# Install Redis
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Run migrations
python scripts/setup/create_database.py
python scripts/setup/run_migrations.py
```

Step 5: Model Downloads

```bash
# Download base models
python scripts/setup/download_models.py --model world-model --version v1.0
python scripts/setup/download_models.py --model clip-vit-large --version openai

# Or download all models
python scripts/setup/download_models.py --all

# Verify installation
python scripts/setup/verify_installation.py
```

Containerized Development

Docker Development

```bash
# Build development image
docker build -f docker/Dockerfile.dev -t frextech-dev .

# Run with GPU support
docker run --gpus all \
  -it \
  -p 8000:8000 \
  -p 8888:8888 \
  -v $(pwd):/app \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  --env-file .env \
  --name frextech-dev \
  frextech-dev

# Run specific services
docker-compose -f docker/docker-compose.dev.yml up api
docker-compose -f docker/docker-compose.dev.yml up training-worker

# Attach to running container
docker exec -it frextech-dev /bin/bash
```

Docker Compose Development

```yaml
# docker/docker-compose.dev.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: frextech
      POSTGRES_PASSWORD: frextech123
      POSTGRES_DB: frextech
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U frextech"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ../:/app
      - model_cache:/models
      - data_volume:/data
    environment:
      - DATABASE_URL=postgresql://frextech:frextech123@postgres:5432/frextech
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    command: uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

  training-worker:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dev
    volumes:
      - ../:/app
      - model_cache:/models
      - data_volume:/data
    environment:
      - DATABASE_URL=postgresql://frextech:frextech123@postgres:5432/frextech
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: celery -A src.core.world_model.training.trainer worker --loglevel=info

  jupyter:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dev
    ports:
      - "8888:8888"
    volumes:
      - ../:/app
      - model_cache:/models
      - data_volume:/data
    environment:
      - JUPYTER_TOKEN=development
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

volumes:
  postgres_data:
  redis_data:
  model_cache:
  data_volume:
```

Kubernetes Development

```bash
# Install minikube for local Kubernetes
minikube start --cpus=4 --memory=8192 --driver=docker

# Enable GPU support
minikube addons enable gpu

# Deploy development stack
kubectl apply -f kubernetes/development/

# Port forward to access services
kubectl port-forward svc/api 8000:8000
kubectl port-forward svc/jupyter 8888:8888

# View logs
kubectl logs -f deployment/api
kubectl logs -f deployment/training-worker

# Scale services
kubectl scale deployment api --replicas=3
kubectl scale deployment training-worker --replicas=2
```

Remote Development

VS Code Remote Development

1. Install VS Code Extensions:
   · Remote - SSH
   · Remote - Containers
   · Python
   · Docker
   · Jupyter
2. Connect via SSH:
   ```bash
   # Generate SSH key
   ssh-keygen -t ed25519 -C "your-email@example.com"
   
   # Copy to remote server
   ssh-copy-id user@remote-server
   
   # Connect in VS Code
   # 1. Open Command Palette (Ctrl+Shift+P)
   # 2. "Remote-SSH: Connect to Host"
   # 3. Enter: user@remote-server
   ```
3. Using Dev Containers:
   ```json
   // .devcontainer/devcontainer.json
   {
     "name": "FrexTech Development",
     "image": "frextech-dev:latest",
     "runArgs": [
       "--gpus", "all",
       "--shm-size", "8g"
     ],
     "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
     "workspaceFolder": "/workspace",
     "customizations": {
       "vscode": {
         "extensions": [
           "ms-python.python",
           "ms-toolsai.jupyter",
           "ms-azuretools.vscode-docker",
           "GitHub.copilot"
         ],
         "settings": {
           "python.pythonPath": "/usr/local/bin/python",
           "python.linting.enabled": true,
           "python.linting.pylintEnabled": false,
           "python.linting.flake8Enabled": true,
           "python.formatting.provider": "black",
           "python.testing.pytestEnabled": true
         }
       }
     },
     "postCreateCommand": "pip install -e .[dev] && pre-commit install",
     "remoteUser": "vscode"
   }
   ```

Cloud Development Environments

GitHub Codespaces

```yaml
# .devcontainer/devcontainer.json for Codespaces
{
  "name": "FrexTech Codespace",
  "image": "mcr.microsoft.com/devcontainers/python:3.10",
  "features": {
    "ghcr.io/devcontainers/features/nvidia-cuda:1": {
      "installCudnn": true
    },
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  },
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python"]
    }
  },
  "forwardPorts": [8000, 8888],
  "postCreateCommand": "bash .devcontainer/setup.sh"
}
```

GitPod

```yaml
# .gitpod.yml
image:
  file: .gitpod.Dockerfile

tasks:
  - init: pip install -e .[dev]
    command: python src/api/server.py

ports:
  - port: 8000
    onOpen: open-preview
  - port: 8888
    onOpen: ignore

github:
  prebuilds:
    master: true
    branches: true
    pullRequests: true
    pullRequestsFromForks: true
    addCheck: true
    addComment: true
```

IDE Configuration

VS Code Configuration

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": [
    "--max-line-length=100",
    "--ignore=E203,W503"
  ],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "-v",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html"
  ],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/.coverage": true,
    "**/.mypy_cache": true,
    "**/.ruff_cache": true
  },
  "files.watcherExclude": {
    "**/venv": true,
    "**/models": true,
    "**/data": true,
    "**/.git": true
  },
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "jupyter.interactiveWindow.textEditor.executeSelection": true,
  "jupyter.kernels.excludePythonEnvironments": [
    "/usr/bin/python3",
    "/usr/local/bin/python3"
  ],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": "explicit"
    }
  },
  "[json]": {
    "editor.formatOnSave": true
  },
  "[yaml]": {
    "editor.formatOnSave": true
  },
  "terminal.integrated.env.linux": {
    "PYTHONPATH": "${workspaceFolder}/src:${env:PYTHONPATH}"
  }
}
```

VS Code Extensions

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "ms-toolsai.jupyter-keymap",
    "ms-toolsai.jupyter-renderers",
    "ms-azuretools.vscode-docker",
    "GitHub.copilot",
    "GitHub.copilot-chat",
    "eamodio.gitlens",
    "esbenp.prettier-vscode",
    "redhat.vscode-yaml",
    "streetsidesoftware.code-spell-checker",
    "charliermarsh.ruff",
    "ms-python.black-formatter",
    "ms-python.isort",
    "VisualStudioExptTeam.vscodeintellicode",
    "KevinRose.vsc-python-indent"
  ]
}
```

PyCharm Configuration

1. Project Interpreter:
   · Set to venv/bin/python
   · Install all dependencies from requirements.txt
2. Run/Debug Configurations:
   ```xml
   <!-- .idea/runConfigurations/API_Server.xml -->
   <component name="ProjectRunConfigurationManager">
     <configuration default="false" name="API Server" type="PythonConfigurationType">
       <module name="frextech-ai-simulations" />
       <option name="INTERPRETER_OPTIONS" value="" />
       <option name="PARENT_ENVS" value="true" />
       <envs>
         <env name="PYTHONPATH" value="$PROJECT_DIR$/src" />
         <env name="ENVIRONMENT" value="development" />
       </envs>
       <option name="SDK_HOME" value="$PROJECT_DIR$/venv/bin/python" />
       <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
       <option name="IS_MODULE_SDK" value="false" />
       <option name="ADD_CONTENT_ROOTS" value="true" />
       <option name="ADD_SOURCE_ROOTS" value="true" />
       <EXTENSION ID="PythonCoverageRunConfigurationExtension" runner="coverage.py" />
       <option name="SCRIPT_NAME" value="src/api/server.py" />
       <option name="PARAMETERS" value="" />
       <option name="SHOW_COMMAND_LINE" value="false" />
       <option name="EMULATE_TERMINAL" value="false" />
       <option name="MODULE_MODE" value="false" />
       <option name="REDIRECT_INPUT" value="false" />
       <option name="INPUT_FILE" value="" />
       <method v="2" />
     </configuration>
   </component>
   ```
3. Code Style:
   · Set line length to 100
   · Use Black formatter
   · Enable isort
   · Configure type checking with mypy

Jupyter Notebook Development

```python
# notebooks/development.ipynb
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "# Development notebook setup\n",
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import sys\n",
    "sys.path.append('../src')\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import torch\n",
    "\n",
    "# Set random seeds for reproducibility\n",
    "torch.manual_seed(42)\n",
    "np.random.seed(42)\n",
    "\n",
    "# Enable GPU if available\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")\n",
    "\n",
    "# Import project modules\n",
    "from core.world_model import WorldModel\n",
    "from utils.visualization import visualize_world"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "frextech-dev",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

Development Workflow

Daily Development Cycle

```bash
# Start development session
source venv/bin/activate
git pull origin develop

# Start development servers
make dev-services  # Starts API, workers, monitoring

# Make changes
git checkout -b feature/my-feature
# ... make code changes ...

# Test changes
pytest tests/unit/test_my_feature.py -xvs
pytest tests/integration/test_my_feature.py --tb=short

# Format and lint
black src/
isort src/
flake8 src/
mypy src/

# Commit changes
git add .
git commit -m "feat: implement my feature"

# Push and create PR
git push origin feature/my-feature
# Create PR on GitHub
```

Hot Reload Development

```python
# For API development with hot reload
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# For worker development
watchmedo auto-restart --directory=./src --pattern=*.py --recursive -- \
    celery -A src.core.world_model.training.trainer worker --loglevel=info

# For Jupyter with autoreload
%load_ext autoreload
%autoreload 2
```

Development Makefile

```makefile
# Makefile
.PHONY: help setup test lint format clean dev

help:
	@echo "Available commands:"
	@echo "  make setup      - Setup development environment"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make dev        - Start development servers"
	@echo "  make docker-dev - Start development in Docker"
	@echo "  make clean      - Clean temporary files"

setup:
	python -m venv venv
	. venv/bin/activate && pip install -e ".[dev]"
	pre-commit install
	cp .env.example .env
	@echo "Please edit .env with your configuration"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/performance/ -v

lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

dev:
	docker-compose -f docker/docker-compose.dev.yml up

docker-dev:
	docker build -f docker/Dockerfile.dev -t frextech-dev .
	docker run --gpus all -it -p 8000:8000 -v $(pwd):/app frextech-dev

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/ .coverage coverage.xml
	rm -rf .ruff_cache/ .hypothesis/
```

Debugging Workflow

```python
# Debug configuration for VS Code
# .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: API Server",
      "type": "python",
      "request": "launch",
      "program": "src/api/server.py",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "ENVIRONMENT": "development",
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: Debug Test",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/unit/test_world_model.py", "-v"],
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    },
    {
      "name": "Python: Interactive",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false,
      "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
      }
    }
  ]
}
```

Testing and Debugging

Debugging Techniques

1. Interactive Debugging with pdb

```python
import pdb
from core.world_model import WorldModel

def debug_function():
    model = WorldModel(config)
    
    # Set breakpoint
    pdb.set_trace()
    
    # Step through execution
    result = model.forward(input_tensor)
    
    # Common pdb commands:
    # n - next line
    # s - step into function
    # c - continue
    # l - list code
    # p variable - print variable
    # pp variable - pretty print
    
    return result
```

2. Remote Debugging

```python
# For remote debugging with debugpy
import debugpy

# Start debug server
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()

# Your code here
model = WorldModel(config)
result = model.forward(input_tensor)
```

3. Debugging in Jupyter

```python
# Enable debugger in Jupyter
%pdb on

# Or use ipdb
import ipdb
ipdb.set_trace()

# Magic commands
%debug  # Enter debugger after exception
%pdb  # Toggle automatic debugger on exception
```

Profiling and Optimization

```python
import cProfile
import pstats
import io
from pstats import SortKey

def profile_function():
    """Profile a function's performance."""
    pr = cProfile.Profile()
    pr.enable()
    
    # Code to profile
    result = expensive_function()
    
    pr.disable()
    
    # Print results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Top 20 functions
    print(s.getvalue())
    
    return result

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function that uses a lot of memory
    large_array = np.zeros((10000, 10000))
    return large_array.sum()
```

Logging Configuration

```python
# src/utils/logging_config.py
import logging
import logging.config
import json
from pathlib import Path

def setup_logging(
    default_level=logging.INFO,
    env_key="LOG_CFG",
    config_file="logging.json"
):
    """Setup logging configuration."""
    
    config_path = Path(config_file)
    if config_path.exists():
        with open(config_path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        # Default configuration
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/development.log")
            ]
        )
    
    # Set specific log levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Usage
logger = setup_logging()

# Structured logging
logger.info("Processing request", extra={
    "request_id": "req_123",
    "user_id": "user_456",
    "endpoint": "/generate"
})
```

Performance Optimization

GPU Optimization

```python
import torch

def optimize_gpu_usage():
    """Optimize GPU memory and performance."""
    
    # Enable TF32 for Ampere GPUs (RTX 30/40, A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cuDNN benchmark for fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
    torch.cuda.empty_cache()  # Clear unused memory
    
    # Use mixed precision training
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
    
    return scaler

def memory_efficient_training(model, data_loader):
    """Train with memory optimization techniques."""
    
    # Gradient checkpointing
    from torch.utils.checkpoint import checkpoint
    
    def forward_with_checkpoint(x):
        return checkpoint(model.forward, x, use_reentrant=False)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    for batch in data_loader:
        with torch.cuda.amp.autocast():
            outputs = forward_with_checkpoint(batch)
            loss = compute_loss(outputs)
        
        # Scale loss and backward
        scaler.scale(loss).backward()
        
        # Unscale gradients and optimize
        scaler.step(optimizer)
        scaler.update()
        
        # Clear gradients
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
```

Profiling Tools

```bash
# PyTorch profiler
python -m torch.utils.bottleneck scripts/training/train_world_model.py

# NVIDIA Nsight Systems
nsys profile -t cuda,nvtx,osrt --capture-range=cudaProfilerApi \
    --stop-on-range-end=true python scripts/training/train_world_model.py

# NVIDIA Nsight Compute (kernel-level)
ncu -k "kernel_name" --metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed" \
    python scripts/training/train_world_model.py

# Memory profiling with mprof
mprof run python scripts/training/train_world_model.py
mprof plot
```

Benchmarking

```python
# scripts/performance/benchmark_inference.py
import time
import torch
from core.world_model import WorldModel

def benchmark_inference():
    """Benchmark model inference performance."""
    
    model = WorldModel(config).cuda()
    model.eval()
    
    # Warmup
    for _ in range(10):
        input_tensor = torch.randn(1, 3, 256, 256).cuda()
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Benchmark
    batch_sizes = [1, 2, 4, 8, 16]
    results = {}
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, 256, 256).cuda()
        
        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                _ = model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        avg_time = (end_time - start_time) / 100
        throughput = batch_size / avg_time
        
        results[batch_size] = {
            "avg_inference_time": avg_time,
            "throughput_fps": throughput,
            "memory_allocated": torch.cuda.memory_allocated() / 1e9
        }
    
    return results
```

Troubleshooting

Common Issues and Solutions

1. CUDA Out of Memory

```python
# Solutions:
# 1. Reduce batch size
batch_size = 4  # instead of 16

# 2. Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(data_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Use model partitioning
from torch.nn.parallel import DistributedDataParallel
model = DistributedDataParallel(model, device_ids=[0, 1])

# 4. Use activation checkpointing
from torch.utils.checkpoint import checkpoint
def custom_forward(x):
    return checkpoint(model.forward, x)
```

2. Slow Training/Inference

```python
# Solutions:
# 1. Enable TF32 (Ampere GPUs)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 2. Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# 3. Optimize data loading
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

# 4. Use JIT compilation
model = torch.jit.script(model)
model = torch.jit.optimize_for_inference(model)
```

3. Dependency Conflicts

```bash
# Create clean environment
python -m venv fresh_venv
source fresh_venv/bin/activate

# Install with exact versions
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e . --no-deps  # Install without dependencies
pip install -r requirements.txt  # Install exact versions

# Or use conda for better dependency resolution
conda create -n frextech python=3.10
conda activate frextech
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

4. Debugging Memory Leaks

```python
import gc
import torch

def check_memory_leaks():
    """Check for memory leaks."""
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check memory usage
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Track tensor creation
    torch.cuda.memory._record_memory_history()
    
    # Run suspicious code
    # ...
    
    # Analyze memory snapshots
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    
    # Disable recording
    torch.cuda.memory._record_memory_history(enabled=None)
```

Diagnostic Commands

```bash
# System diagnostics
nvidia-smi  # GPU status
htop  # CPU/RAM usage
df -h  # Disk usage
docker stats  # Container resources

# Python diagnostics
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import torch; print(torch.__version__)"

# Network diagnostics
curl http://localhost:8000/health  # API health
curl http://localhost:5432  # PostgreSQL
redis-cli ping  # Redis

# Log analysis
tail -f logs/development.log
grep "ERROR" logs/development.log
journalctl -u frextech-api  # Systemd service logs
```

Advanced Development

Custom Model Development

```python
# src/core/world_model/custom_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomAttention(nn.Module):
    """Custom attention layer with efficiency optimizations."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Flash attention if available
        self.use_flash = hasattr(F, "scaled_dot_product_attention")
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash:
            # Use PyTorch 2.0's efficient attention
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            # Fallback to manual attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# Register custom layer
import src.core.world_model.architecture.custom_layers as custom_layers
setattr(custom_layers, "CustomAttention", CustomAttention)
```

Plugin System

```python
# src/utils/plugins.py
import importlib
from typing import Dict, Any, Type
from pathlib import Path

class PluginManager:
    """Manager for loading and using plugins."""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins = {}
        self.load_plugins()
    
    def load_plugins(self):
        """Load all plugins from the plugin directory."""
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            module_name = plugin_file.stem
            spec = importlib.util.spec_from_file_location(
                f"plugins.{module_name}",
                plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Register plugin
            if hasattr(module, "register"):
                plugin = module.register()
                self.plugins[plugin.name] = plugin
    
    def get_plugin(self, name: str):
        """Get a plugin by name."""
        return self.plugins.get(name)
    
    def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute a hook across all plugins."""
        results = []
        for plugin in self.plugins.values():
            if hasattr(plugin, hook_name):
                result = getattr(plugin, hook_name)(*args, **kwargs)
                results.append(result)
        return results

# Example plugin
# plugins/custom_gaussian.py
class CustomGaussianPlugin:
    name = "custom_gaussian"
    
    @staticmethod
    def register():
        return CustomGaussianPlugin()
    
    def pre_generation(self, prompt: str, config: Dict):
        """Hook called before generation."""
        if "mountain" in prompt.lower():
            config["terrain_scale"] = 2.0
        return config
    
    def post_generation(self, world, metadata: Dict):
        """Hook called after generation."""
        # Add custom metadata
        metadata["plugin_processed"] = True
        return world, metadata
```

Distributed Training

```python
# scripts/training/distributed_training.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def train(rank, world_size):
    """Training function for each process."""
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = WorldModel(config).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler
    dataset = create_dataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        
        for batch in dataloader:
            batch = batch.to(rank)
            
            # Forward pass
            output = ddp_model(batch)
            loss = compute_loss(output)
            
            # Backward pass
            loss.backward()
            
            # Synchronize gradients
            for param in ddp_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= world_size
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

Continuous Integration Setup

```yaml
# .github/workflows/development.yml
name: Development CI

on:
  push:
    branches: [develop, feature/*]
  pull_request:
    branches: [develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Lint with flake8
      run: |
        flake8 src/ tests/ --count --show-source --statistics
    
    - name: Type check with mypy
      run: |
        mypy src/ --ignore-missing-imports
    
    - name: Test with pytest
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build-docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        file: ./docker/Dockerfile.dev
        tags: frextech/dev:latest
        push: false
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

---

This guide is maintained by the FrexTech Development Team.
Last Updated: January 1, 2024
Version: 2.0

For additional help:

· Join our Discord community
· Check our documentation
· Email: development@frextech-sim.com
