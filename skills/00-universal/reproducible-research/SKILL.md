---
name: reproducible-research
description: >
  Use this Skill to make a research project fully reproducible: conda/renv
  environment locking, DVC data versioning, Dockerfile for analysis, and
  FAIR data principles.
tags:
  - universal
  - reproducibility
  - DVC
  - conda
  - Docker
  - FAIR
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - dvc>=3.0
    - pandas>=1.5
    - pyyaml>=6.0
  system:
    - conda
    - docker
    - git
last_updated: "2026-03-17"
status: stable
---

# Reproducible Research

> **TL;DR** — Make a research project fully reproducible end-to-end: lock Python
> environments with conda, version large data files with DVC, containerize the
> analysis with Docker, protect secrets with `.env` files, and satisfy FAIR
> data principles for long-term reusability.

---

## When to Use This Skill

Use this Skill when you need to:

- Share analysis code with collaborators or reviewers who must reproduce your results
- Submit to a journal that requires data and code availability
- Set up a continuous integration (CI) pipeline that re-runs your analysis
- Archive a project at the time of paper submission
- Comply with funder open-science mandates (NIH, Wellcome Trust, Horizon Europe)

| Task | Tool |
|---|---|
| Environment locking (Python) | `conda` + `environment.yml` |
| Environment locking (R) | `renv` + `renv.lock` |
| Large file / data versioning | DVC |
| Containerized execution | Docker |
| Secrets management | `.env` file (gitignored) |
| FAIR compliance | Metadata, DOI, open license |

---

## Background & Key Concepts

### The Reproducibility Crisis

A 2016 Nature survey found that > 70% of researchers failed to reproduce another
scientist's results. The leading causes are:

1. Missing or undocumented software dependencies
2. Unavailable or unversioned raw data
3. Undisclosed analysis deviations
4. Platform-specific code that fails on different OS/hardware

### FAIR Principles

| Letter | Meaning | Practical action |
|---|---|---|
| **F**indable | Persistent identifier, rich metadata | Assign a DOI via Zenodo |
| **A**ccessible | Open protocol, authentication if needed | Host on GitHub + Zenodo |
| **I**nteroperable | Standard formats, vocabularies | Use CSV/JSON/HDF5, Dublin Core metadata |
| **R**eusable | Clear license, provenance documented | Add LICENSE file, README with methods |

### Conda vs pip vs Docker

- **conda**: best for scientific Python stacks with compiled dependencies (NumPy, CUDA)
- **pip + venv**: lighter, good for pure-Python projects
- **Docker**: full OS-level isolation; best for exact reproducibility across machines

The recommended stack: conda for local dev, Docker for CI and archival.

---

## Environment Setup

```bash
# Install conda (if not already installed)
# https://docs.conda.io/en/latest/miniconda.html

# Install DVC
pip install "dvc[s3,gs,azure,ssh]>=3.0"

# Install Docker Desktop (see https://www.docker.com/products/docker-desktop)
docker --version   # verify

# Clone your project (or init a new one)
git init my-project
cd my-project
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

---

## Core Workflow

### Step 1 — Conda Environment Locking

Create a precise `environment.yml` with pinned versions, then generate a
fully-resolved lock file for bit-for-bit reproducibility.

```bash
# Create the project environment
conda create -n myproject python=3.11 -y
conda activate myproject

# Install packages
conda install -c conda-forge pandas=2.0.3 matplotlib=3.8.0 scikit-learn=1.3.2 -y
pip install dvc>=3.0 pyyaml>=6.0

# Export the human-readable spec
conda env export --no-builds > environment.yml

# Export a fully-pinned cross-platform lock file (requires conda-lock)
pip install conda-lock
conda-lock lock --file environment.yml --platform linux-64 --platform osx-arm64
# Produces: conda-lock.yml

# Reproduce the environment from the lock file on any machine
conda-lock install --name myproject conda-lock.yml
```

Example `environment.yml` structure:

```yaml
name: myproject
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas=2.0.3
  - matplotlib=3.8.0
  - scikit-learn=1.3.2
  - pip:
    - dvc>=3.0
    - pyyaml>=6.0
```

### Step 2 — DVC Data Versioning Pipeline

DVC tracks large files outside git, runs pipelines reproducibly, and stores
data in a remote storage backend (S3, GCS, SSH, or local).

```python
# dvc_setup.py — programmatic DVC configuration
import subprocess
import os
import yaml
from pathlib import Path


def init_dvc_project(project_root: str = ".") -> None:
    """
    Initialize a DVC project and configure a local remote.

    Args:
        project_root: Path to the project root directory.
    """
    root = Path(project_root)

    # Initialize DVC (idempotent)
    subprocess.run(["dvc", "init"], cwd=root, check=True)

    # Configure a local remote (change to S3/GCS URI in production)
    remote_path = root / ".dvc_remote"
    remote_path.mkdir(exist_ok=True)
    subprocess.run(
        ["dvc", "remote", "add", "-d", "local_remote", str(remote_path)],
        cwd=root, check=True,
    )
    print(f"DVC initialized with local remote at {remote_path}")


def track_data_file(file_path: str) -> None:
    """
    Add a data file to DVC tracking.

    The file is replaced by a small .dvc pointer file that is committed to git.

    Args:
        file_path: Relative path to the data file inside the project.
    """
    subprocess.run(["dvc", "add", file_path], check=True)
    dvc_file = file_path + ".dvc"
    subprocess.run(["git", "add", dvc_file, ".gitignore"], check=True)
    print(f"Tracked {file_path} with DVC. Commit {dvc_file} to git.")


def create_dvc_pipeline(stages: list[dict], output_path: str = "dvc.yaml") -> None:
    """
    Write a DVC pipeline YAML file from a list of stage definitions.

    Args:
        stages: List of dicts, each with keys:
                  name (str), cmd (str), deps (list), outs (list),
                  params (list, optional).
        output_path: Path to write dvc.yaml.

    Example stage:
        {
            "name": "preprocess",
            "cmd": "python src/preprocess.py",
            "deps": ["src/preprocess.py", "data/raw.csv"],
            "outs": ["data/processed.csv"],
            "params": ["params.yaml:preprocessing"],
        }
    """
    pipeline: dict = {"stages": {}}
    for stage in stages:
        entry: dict = {"cmd": stage["cmd"], "deps": stage.get("deps", []),
                       "outs": stage.get("outs", [])}
        if "params" in stage:
            entry["params"] = stage["params"]
        pipeline["stages"][stage["name"]] = entry

    with open(output_path, "w") as f:
        yaml.dump(pipeline, f, default_flow_style=False, sort_keys=False)
    print(f"DVC pipeline written to {output_path}")


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    stages = [
        {
            "name": "preprocess",
            "cmd": "python src/preprocess.py",
            "deps": ["src/preprocess.py", "data/raw.csv"],
            "outs": ["data/processed.csv"],
            "params": ["params.yaml:preprocessing"],
        },
        {
            "name": "train",
            "cmd": "python src/train.py",
            "deps": ["src/train.py", "data/processed.csv"],
            "outs": ["models/model.pkl"],
            "params": ["params.yaml:training"],
        },
        {
            "name": "evaluate",
            "cmd": "python src/evaluate.py",
            "deps": ["src/evaluate.py", "models/model.pkl", "data/processed.csv"],
            "outs": ["reports/metrics.json"],
        },
    ]
    create_dvc_pipeline(stages)
    # Then run: dvc repro   (executes only changed stages)
    # Or:       dvc dag     (visualize the pipeline DAG)
```

### Step 3 — Dockerfile for Reproducible Analysis

```dockerfile
# Dockerfile — place in project root
# Build: docker build -t myproject:1.0 .
# Run:   docker run --rm -v $(pwd)/data:/app/data myproject:1.0

FROM jupyter/scipy-notebook:python-3.11

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Back to the default non-root user
USER ${NB_UID}

# Copy and install Python dependencies
COPY --chown=${NB_UID}:${NB_GID} environment.yml /tmp/environment.yml
RUN conda env update --name base --file /tmp/environment.yml && \
    conda clean --all -f -y

# Install DVC
RUN pip install --no-cache-dir "dvc[s3]>=3.0"

# Copy project source code (not data — mount data volume at runtime)
WORKDIR /app
COPY --chown=${NB_UID}:${NB_GID} src/ /app/src/
COPY --chown=${NB_UID}:${NB_GID} params.yaml /app/params.yaml
COPY --chown=${NB_UID}:${NB_GID} dvc.yaml /app/dvc.yaml

# Default command: reproduce the DVC pipeline
CMD ["dvc", "repro"]
```

Build and run helpers:

```bash
# Build the image
docker build -t myproject:1.0 .

# Run the full analysis pipeline
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  myproject:1.0

# Interactive shell for debugging
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  myproject:1.0 /bin/bash
```

---

## Advanced Usage

### .env File Pattern for Secrets

Never commit API keys or passwords to git. Use a `.env` file that is listed
in `.gitignore`, and load it with `python-dotenv` or `os.getenv`.

```python
# src/config.py
import os
from pathlib import Path

# Load .env file if it exists (development convenience)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # In Docker, variables are injected via --env-file or -e flags

# export ZENODO_TOKEN="<paste-your-key>"
ZENODO_TOKEN: str = os.getenv("ZENODO_TOKEN", "")

# export S3_BUCKET="<your-bucket-name>"
S3_BUCKET: str = os.getenv("S3_BUCKET", "")

# export OPENAI_API_KEY="<paste-your-key>"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

def require_env(name: str) -> str:
    """Raise a clear error if a required environment variable is missing."""
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set.\n"
            f"Add it to your .env file:  {name}=<value>"
        )
    return val
```

`.env` file (gitignored):

```
ZENODO_TOKEN=<paste-your-key>
S3_BUCKET=my-research-bucket
OPENAI_API_KEY=<paste-your-key>
```

`.gitignore` additions:

```
.env
data/
models/
outputs/
*.pyc
__pycache__/
.dvc/tmp/
```

### R Project Reproducibility with renv

```bash
# In R console — one-time setup
# install.packages("renv")
# renv::init()          # creates renv.lock, .Rprofile, renv/

# After installing packages
# renv::snapshot()      # update renv.lock with current package state

# On a new machine or fresh clone:
# renv::restore()       # install exact package versions from renv.lock
```

### ORCID Integration for Author Disambiguation

Add ORCID identifiers to your README and metadata files:

```python
def generate_citation_cff(authors: list[dict], title: str, doi: str) -> str:
    """
    Generate a CITATION.cff file content for GitHub citation metadata.

    Args:
        authors: List of dicts with keys: name (str), orcid (str, optional).
        title:   Repository/paper title.
        doi:     DOI of the archived version.

    Returns:
        CITATION.cff file content as a string.
    """
    import yaml
    cff = {
        "cff-version": "1.2.0",
        "message": "If you use this software, please cite it using the metadata below.",
        "title": title,
        "doi": doi,
        "authors": [
            {"name": a["name"], **({"orcid": f"https://orcid.org/{a['orcid']}"} if "orcid" in a else {})}
            for a in authors
        ],
        "license": "MIT",
        "date-released": "2026-03-17",
    }
    return yaml.dump(cff, default_flow_style=False, allow_unicode=True)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `dvc repro` re-runs all stages | Parameter or dependency changed | Check `dvc status`; use `dvc params diff` |
| Docker build fails on ARM Mac | Base image architecture mismatch | Use `--platform linux/amd64` in `FROM` line |
| `conda env export` yields platform-specific packages | Build strings included | Use `--no-builds` flag |
| `.env` variables not loaded | `python-dotenv` not installed | `pip install python-dotenv` |
| DVC push fails | Remote not configured | Run `dvc remote add -d ...` and `dvc push` |
| `renv::restore()` fails | CRAN package removed | Pin from CRAN archive or use Posit Package Manager |

---

## External Resources

- DVC documentation: <https://dvc.org/doc>
- conda-lock: <https://github.com/conda/conda-lock>
- renv: <https://rstudio.github.io/renv/>
- FAIR principles: <https://www.go-fair.org/fair-principles/>
- ORCID: <https://orcid.org>
- Jupyter/scipy-notebook Docker image: <https://jupyter-docker-stacks.readthedocs.io>
- The Turing Way (reproducibility guide): <https://the-turing-way.netlify.app>

---

## Examples

### Example 1 — conda environment.yml + Lock File Workflow

```bash
# Step 1: Create and export the environment
conda create -n analysis python=3.11 pandas=2.0.3 matplotlib=3.8.0 -y
conda activate analysis
conda env export --no-builds > environment.yml

# Step 2: Generate a cross-platform lock file
pip install conda-lock
conda-lock lock --file environment.yml --platform linux-64

# Step 3: Commit to git
git add environment.yml conda-lock.yml
git commit -m "Pin environment: Python 3.11, pandas 2.0.3, matplotlib 3.8.0"

# Step 4: Reproduce on a new machine
conda-lock install --name analysis conda-lock.yml
```

### Example 2 — DVC Pipeline Reproduction

```bash
# Initialize DVC in the project
git init my-paper && cd my-paper
dvc init
git add . && git commit -m "Initialize DVC"

# Track raw data
cp ~/Downloads/raw_data.csv data/raw.csv
dvc add data/raw.csv
git add data/raw.csv.dvc data/.gitignore
git commit -m "Track raw data with DVC"

# Write dvc.yaml (see create_dvc_pipeline above), then:
dvc repro              # runs all stages
dvc dag                # visualize pipeline graph
dvc push               # push data to remote storage

# On collaborator's machine:
git clone <repo-url> && cd <repo>
dvc pull               # download tracked data
dvc repro              # reproduce all outputs
```

### Example 3 — Dockerfile Build and Run

```bash
# Build
docker build -t myproject:1.0 .

# Run the full pipeline
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  --env-file .env \
  myproject:1.0

# Push image to registry for archival
docker tag myproject:1.0 ghcr.io/myorg/myproject:1.0
docker push ghcr.io/myorg/myproject:1.0
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — conda locking, DVC pipeline, Dockerfile, .env pattern |
