---
name: containerization-research
description: Containerize research environments with Docker and Singularity for reproducible computation, HPC deployment, and research environment management.
tags:
  - docker
  - singularity
  - reproducibility
  - research-computing
  - hpc
version: "1.0.0"
authors:
  - "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - docker>=6.1
    - pandas>=2.0
    - numpy>=1.24
last_updated: "2026-03-17"
status: stable
---

# Containerization for Research

## When to Use This Skill

Use this skill when you need to:
- Freeze a research computational environment for reproducibility
- Deploy analyses to HPC clusters using Singularity/Apptainer
- Automate multi-container research workflows with Docker Compose
- Share an exact computational environment with collaborators
- Build reproducible research compendiums (data + code + environment)
- Run GPU-accelerated experiments in isolated containers
- Create research environment images for cloud computing (AWS, GCP)

**Trigger keywords**: Docker, Dockerfile, container, reproducible research, Singularity, Apptainer, Docker Compose, HPC container, research compendium, environment isolation, image, registry, Docker Hub, container orchestration, GPU container, NVIDIA Container Toolkit, conda-pack, conda-forge, reprozip, research environment, FAIR data.

## Background & Key Concepts

### Container vs. Virtual Machine

Containers share the host OS kernel (lightweight, fast startup) while VMs run a complete guest OS. For research:
- **Docker**: standard for development and cloud; root daemon (security concern on HPC)
- **Singularity/Apptainer**: designed for HPC, runs as user (no root daemon), can run Docker images

### Dockerfile Best Practices

1. Use specific base image tags (e.g., `python:3.11.5-slim`) not `latest`
2. Layer caching: put infrequently changing layers first (OS packages before pip installs)
3. Multi-stage builds: separate build dependencies from runtime
4. Non-root user: `RUN useradd -m researcher && USER researcher`
5. `.dockerignore`: exclude large data files and build artifacts

### Image Size Optimization

- Use `python:3.11-slim` or `python:3.11-alpine` instead of full `python:3.11`
- Combine `RUN` commands to reduce layers
- Delete package caches: `rm -rf /var/lib/apt/lists/*`
- Use `--no-install-recommends` with apt

### Reproducibility Guarantee

Pin all versions:
```
python==3.11.5
numpy==1.24.3
pandas==2.0.3
```
Or use lock files: `pip-compile` → `requirements.txt`, `conda-lock` → `conda-lock.yml`

## Environment Setup

```bash
# Install Docker (Linux)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install Docker Compose
docker compose version

# Python Docker SDK
pip install docker>=6.1

# Verify
docker --version && docker compose version
```

## Core Workflow

### Step 1: Dockerfile for Research Python Environment

```dockerfile
# File: Dockerfile
# Multi-stage research environment

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11.5-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python packages into /install
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11.5-slim AS runtime

LABEL maintainer="research@university.edu"
LABEL description="Reproducible research environment for quantitative analysis"
LABEL version="1.0.0"

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN useradd -m -u 1000 researcher
WORKDIR /workspace
RUN chown researcher:researcher /workspace

USER researcher

# Copy research code
COPY --chown=researcher:researcher src/ ./src/
COPY --chown=researcher:researcher scripts/ ./scripts/
COPY --chown=researcher:researcher config/ ./config/

# Expose Jupyter port
EXPOSE 8888

# Default command: start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--NotebookApp.token=''"]
```

```text
# File: requirements.txt
numpy==1.24.4
pandas==2.0.3
scipy==1.11.4
scikit-learn==1.3.2
statsmodels==0.14.1
matplotlib==3.7.3
seaborn==0.13.0
jupyter==1.0.0
jupyterlab==4.0.9
```

```text
# File: .dockerignore
.git
.github
__pycache__
*.pyc
*.pyo
*.egg-info
.pytest_cache
.coverage
htmlcov/
dist/
build/
data/*.csv
data/*.parquet
!data/sample_data.csv
*.log
.env
```

```bash
# Build the image
docker build -t research-env:1.0.0 .
docker build --platform linux/amd64 -t research-env:1.0.0 .  # for cross-platform

# Run interactively
docker run --rm -it \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/results:/workspace/results \
  -p 8888:8888 \
  research-env:1.0.0 bash

# Run a specific analysis script
docker run --rm \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/results:/workspace/results \
  research-env:1.0.0 \
  python src/analysis.py --config config/params.yaml

# Run Jupyter Lab
docker run --rm -d \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  --name research-jupyter \
  research-env:1.0.0
```

### Step 2: Docker Compose for Multi-Service Research Workflow

```yaml
# File: docker-compose.yml
version: "3.9"

services:
  # ── Main analysis service ──────────────────────────────────────────────────
  analysis:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    image: research-env:1.0.0
    container_name: research-analysis
    volumes:
      - ./data:/workspace/data:ro          # read-only data
      - ./results:/workspace/results       # write results
      - ./src:/workspace/src               # live code update (dev)
      - ./config:/workspace/config:ro
    environment:
      - PYTHONUNBUFFERED=1
      - DATA_DIR=/workspace/data
      - RESULTS_DIR=/workspace/results
    command: python src/main_analysis.py
    networks:
      - research-net
    depends_on:
      - database

  # ── PostgreSQL for intermediate results ───────────────────────────────────
  database:
    image: postgres:15-alpine
    container_name: research-db
    environment:
      POSTGRES_DB: research
      POSTGRES_USER: researcher
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - research-net
    secrets:
      - db_password

  # ── Jupyter Lab for interactive exploration ────────────────────────────────
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile
    image: research-env:1.0.0
    container_name: research-jupyter
    ports:
      - "8888:8888"
    volumes:
      - .:/workspace
    command: >
      jupyter lab
      --ip=0.0.0.0
      --port=8888
      --no-browser
      --NotebookApp.token=''
      --NotebookApp.password=''
    networks:
      - research-net

  # ── MLflow tracking server ─────────────────────────────────────────────────
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    container_name: research-mlflow
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    networks:
      - research-net

networks:
  research-net:
    driver: bridge

volumes:
  pgdata:
  mlflow-data:

secrets:
  db_password:
    file: ./secrets/db_password.txt
```

```bash
# Start all services
docker compose up -d

# Run analysis only
docker compose run --rm analysis

# View logs
docker compose logs -f analysis

# Stop and clean up
docker compose down -v
```

### Step 3: Singularity for HPC Deployment

```bash
# Singularity definition file: research.def
```

```singularity
# File: research.def
Bootstrap: docker
From: python:3.11.5-slim

%labels
    Author research@university.edu
    Version 1.0.0
    Description Research analysis environment

%environment
    export PATH=/opt/conda/bin:$PATH
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4

%post
    apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libopenblas-dev curl && \
        rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir \
        numpy==1.24.4 \
        pandas==2.0.3 \
        scipy==1.11.4 \
        scikit-learn==1.3.2 \
        statsmodels==0.14.1 \
        matplotlib==3.7.3

    # Create workspace directory
    mkdir -p /workspace /results /scratch

%runscript
    echo "Research environment v1.0.0"
    exec python "$@"

%test
    python -c "import numpy, pandas, scipy, sklearn; print('All imports OK')"
```

```bash
# Build Singularity image (requires root or fakeroot)
singularity build research.sif research.def

# On HPC: convert from Docker Hub
singularity pull research.sif docker://username/research-env:1.0.0

# Run on HPC (SLURM)
singularity exec research.sif python analysis.py

# Run with GPU support
singularity exec --nv research.sif python train_model.py

# Bind mount data
singularity exec \
  --bind /scratch/data:/data:ro \
  --bind /scratch/results:/results \
  research.sif python analysis.py
```

```bash
# SLURM job script: submit_job.sh
#!/bin/bash
#SBATCH --job-name=research_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=standard
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

module load singularity/3.9

singularity exec \
  --bind $SCRATCH/data:/data:ro \
  --bind $SCRATCH/results:/results \
  --bind $SLURM_SUBMIT_DIR/src:/src:ro \
  research.sif \
  python /src/analysis.py \
    --n-jobs ${SLURM_NTASKS} \
    --output /results/output_${SLURM_JOB_ID}
```

## Advanced Usage

### Python Docker SDK for Workflow Automation

```python
import docker
import os
import tarfile
import io
import json
from pathlib import Path

def run_analysis_container(
    image="research-env:1.0.0",
    script="analysis.py",
    data_dir="./data",
    results_dir="./results",
    env_vars=None,
):
    """Run an analysis in a Docker container.

    Args:
        image: Docker image name:tag
        script: Script to run inside container
        data_dir: Local data directory to mount (read-only)
        results_dir: Local results directory to mount (write)
        env_vars: Additional environment variables
    Returns:
        dict with exit_code, logs, and output_path
    """
    client = docker.from_env()
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    environment = {"PYTHONUNBUFFERED": "1"}
    if env_vars:
        environment.update(env_vars)

    volumes = {
        os.path.abspath(data_dir): {
            "bind": "/workspace/data", "mode": "ro"
        },
        os.path.abspath(results_dir): {
            "bind": "/workspace/results", "mode": "rw"
        },
    }

    container = client.containers.run(
        image=image,
        command=f"python /workspace/src/{script}",
        volumes=volumes,
        environment=environment,
        detach=True,
        remove=False,  # keep for log retrieval
    )

    # Wait for completion
    exit_code = container.wait()["StatusCode"]
    logs = container.logs(stdout=True, stderr=True).decode("utf-8")

    container.remove()

    return {
        "exit_code": exit_code,
        "logs": logs,
        "success": exit_code == 0,
        "results_dir": results_dir,
    }

# Demonstrate (without actually running Docker)
print("Docker container automation configured")
print("Usage: run_analysis_container('research-env:1.0.0', 'my_analysis.py')")
```

### Multi-Stage GPU Build

```dockerfile
# File: Dockerfile.gpu
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS gpu-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.0+cu121 \
        torchvision==0.16.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Runtime stage
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu-runtime

COPY --from=gpu-builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=gpu-builder /usr/local/bin/python* /usr/local/bin/

RUN useradd -m -u 1000 researcher
USER researcher
WORKDIR /workspace

CMD ["python3.11", "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()}')"]
```

### Environment Freeze Script

```python
import subprocess
import json
import platform
import sys
from datetime import datetime

def freeze_environment(output_file="environment_snapshot.json"):
    """Capture full environment state for reproducibility.

    Records Python version, installed packages, system info, and git state.
    """
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
    }

    # Installed packages
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        snapshot["pip_packages"] = json.loads(result.stdout)

    # Git commit hash
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True
    )
    if git_result.returncode == 0:
        snapshot["git_commit"] = git_result.stdout.strip()

    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True
    )
    snapshot["git_dirty"] = len(git_status.stdout.strip()) > 0

    with open(output_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Environment snapshot saved to {output_file}")
    print(f"  Python: {snapshot['python_version'].split()[0]}")
    print(f"  Packages: {len(snapshot.get('pip_packages', []))}")
    print(f"  Git commit: {snapshot.get('git_commit', 'unknown')[:8]}")
    return snapshot

snapshot = freeze_environment()
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `permission denied: /var/run/docker.sock` | User not in docker group | `sudo usermod -aG docker $USER && newgrp docker` |
| Container exits immediately | Script error or CMD syntax | Run with `bash` to debug: `docker run --rm -it image bash` |
| Large image size | Including data or build artifacts | Use `.dockerignore`; multi-stage build |
| Volume mount fails (Windows) | Path format | Use `//c/Users/...` or Docker Desktop volume GUI |
| Singularity on HPC: no root | Requires fakeroot or admin | Use `--fakeroot` flag or pull pre-built Docker image |
| GPU not visible in container | Missing NVIDIA runtime | Install `nvidia-container-toolkit`; run with `--gpus all` |

## External Resources

- [Docker documentation](https://docs.docker.com/)
- [Singularity/Apptainer user guide](https://apptainer.org/docs/)
- [Reproducible research with containers (The Turing Way)](https://book.the-turing-way.org/reproducible-research/renv/renv-containers)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Python SDK](https://docker-py.readthedocs.io/)

## Examples

### Example 1: Research Compendium with Make

```makefile
# File: Makefile
.PHONY: build run clean push

IMAGE := research-env
TAG := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

build:
	docker build -t $(IMAGE):$(TAG) -t $(IMAGE):latest .
	docker inspect $(IMAGE):$(TAG) | python -c \
	  "import json,sys; img=json.load(sys.stdin)[0]; print(f'Size: {img[\"Size\"]/1e6:.1f} MB')"

run:
	docker compose up -d
	@echo "Jupyter at http://localhost:8888"

analysis:
	docker compose run --rm analysis python src/main_analysis.py

clean:
	docker compose down -v
	docker image rm $(IMAGE):$(TAG) 2>/dev/null || true

push:
	docker push $(IMAGE):$(TAG)
	docker push $(IMAGE):latest
```

### Example 2: Verify Container Reproducibility

```python
import subprocess
import hashlib
import json

def verify_analysis_reproducibility(script, data_hash, n_runs=3):
    """Run a script multiple times and verify output hashes match.

    Args:
        script: path to analysis script
        data_hash: SHA256 hash of input data
        n_runs: number of repeated runs
    Returns:
        dict with consistency information
    """
    output_hashes = []
    for run in range(n_runs):
        # In practice: docker run --rm image python script
        # Here: simulate by running locally
        result = subprocess.run(
            ["python", script, "--seed", "42"],
            capture_output=True, text=True
        )
        output_hash = hashlib.sha256(result.stdout.encode()).hexdigest()
        output_hashes.append(output_hash)
        print(f"Run {run+1}: {output_hash[:16]}...")

    all_same = len(set(output_hashes)) == 1
    print(f"\nReproducibility: {'PASS' if all_same else 'FAIL'}")
    print(f"Input data hash: {data_hash[:16]}...")
    return {"reproducible": all_same, "hashes": output_hashes}
```
