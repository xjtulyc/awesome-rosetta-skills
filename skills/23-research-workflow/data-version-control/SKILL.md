---
name: data-version-control
description: Data version control with DVC covering pipeline tracking, remote storage, experiment comparison, and reproducible ML workflows for research.
tags:
  - dvc
  - data-version-control
  - mlops
  - reproducibility
  - experiment-tracking
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
    - dvc>=3.30
    - dvc-s3>=3.0
    - pandas>=2.0
    - numpy>=1.24
    - scikit-learn>=1.3
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: stable
---

# Data Version Control (DVC)

## When to Use This Skill

Use this skill when you need to:
- Version large datasets alongside code with Git-like semantics
- Define and run reproducible data processing pipelines
- Track ML experiment parameters, metrics, and artifacts
- Share and cache large data files on remote storage (S3, GCS, SSH, local)
- Compare experiment runs and identify best models
- Collaborate on data-intensive research with exact reproducibility
- Build CI/CD pipelines for model training and evaluation

**Trigger keywords**: DVC, data version control, dvc.yaml, dvc.lock, data pipeline, experiment tracking, ML reproducibility, model versioning, dataset versioning, remote storage, dvc run, dvc repro, dvc metrics, dvc plots, MLflow, wandb, artifacts tracking, Git-DVC integration, data registry.

## Background & Key Concepts

### DVC Architecture

DVC tracks data files by storing their MD5 hashes in `.dvc` files committed to Git, while the actual data is stored in a DVC cache (local or remote):

```
dataset.csv    →  dataset.csv.dvc (Git-tracked, stores hash)
                  .dvc/cache/ab/cdef... (actual data, Git-ignored)
```

Remote storage syncs the cache to cloud/server:

```bash
dvc push  # upload cache to remote
dvc pull  # download from remote to local cache
```

### Pipeline Definition (dvc.yaml)

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps: [data/raw/, src/preprocess.py]
    outs: [data/processed/]
    params: [params.yaml:preprocess]

  train:
    cmd: python src/train.py
    deps: [data/processed/, src/train.py]
    outs: [models/model.pkl]
    metrics: [metrics/train_metrics.json]
    params: [params.yaml:train]
```

### Experiment Tracking

DVC experiments are lightweight Git branches under the hood:

```bash
dvc exp run --set-param train.lr=0.01
dvc exp show  # compare experiments
dvc exp apply exp-abc123  # apply best experiment
```

### DAG Execution

DVC detects which stages are stale (dependencies changed) and re-runs only those:

```bash
dvc repro  # re-run full pipeline (skip up-to-date stages)
dvc dag    # visualize dependency graph
```

## Environment Setup

```bash
pip install dvc>=3.30 dvc-s3>=3.0 pandas>=2.0 numpy>=1.24 \
            scikit-learn>=1.3 matplotlib>=3.7

# Initialize DVC in a Git repository
git init my-research-project
cd my-research-project
dvc init

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-cache
# or local:
dvc remote add -d localremote /path/to/shared/storage

git add .dvc/config
git commit -m "Initialize DVC with remote storage"
```

## Core Workflow

### Step 1: Version a Dataset

```bash
# Add a large dataset to DVC tracking
dvc add data/raw/survey_2023.csv

# This creates:
# - data/raw/survey_2023.csv.dvc (commit this to Git)
# - .gitignore updated to ignore the actual CSV
git add data/raw/survey_2023.csv.dvc data/raw/.gitignore
git commit -m "Add survey 2023 dataset to DVC"

# Push data to remote
dvc push

# Later: reproduce from another machine
git clone https://github.com/org/my-research-project
cd my-research-project
dvc pull  # downloads data from remote
```

```python
# File: scripts/create_sample_dvc_structure.py
"""Create a sample DVC project structure."""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

# Create directory structure
for d in ["data/raw", "data/processed", "models", "metrics",
          "reports/figures", "src"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Create sample dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "feature_1": np.random.normal(5, 2, n),
    "feature_2": np.random.uniform(0, 10, n),
    "feature_3": np.random.binomial(1, 0.4, n),
    "feature_4": np.random.poisson(3, n),
    "target": (np.random.normal(5, 2, n) * 1.5 +
               np.random.uniform(0, 10, n) * 0.8 +
               np.random.binomial(1, 0.4, n) * 2 +
               np.random.normal(0, 1, n))
})
df.to_csv("data/raw/dataset.csv", index=False)
print(f"Created data/raw/dataset.csv: {len(df)} rows")

# Create params.yaml
params = {
    "preprocess": {
        "test_size": 0.2,
        "random_seed": 42,
        "scale_features": True,
    },
    "train": {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 5,
        "random_seed": 42,
    },
    "evaluate": {
        "metrics": ["rmse", "mae", "r2"],
    }
}
with open("params.yaml", "w") as f:
    import yaml
    yaml.dump(params, f, default_flow_style=False)
print("Created params.yaml")
```

### Step 2: Define a DVC Pipeline

```python
# File: src/preprocess.py
"""Data preprocessing stage."""

import numpy as np
import pandas as pd
import yaml
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    cfg = params["preprocess"]

    # Load raw data
    df = pd.read_csv("data/raw/dataset.csv")
    print(f"Loaded {len(df)} rows from data/raw/dataset.csv")

    # Feature engineering
    feature_cols = ["feature_1", "feature_2", "feature_3", "feature_4"]
    X = df[feature_cols].values
    y = df["target"].values

    # Scale if requested
    scaler = None
    if cfg.get("scale_features", True):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Applied StandardScaler")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_seed"]
    )

    # Save processed data
    Path("data/processed").mkdir(exist_ok=True)
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    if scaler:
        with open("data/processed/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    # Log split info
    split_info = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
    }
    with open("data/processed/split_info.json", "w") as f:
        json.dump(split_info, f)

    print(f"Train: {len(X_train)} | Test: {len(X_test)} samples")

if __name__ == "__main__":
    main()
```

```python
# File: src/train.py
"""Model training stage."""

import numpy as np
import pandas as pd
import yaml
import json
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from pathlib import Path

def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    cfg = params["train"]

    # Load processed data
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")

    # Select and configure model
    model_map = {
        "random_forest": RandomForestRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "ridge": Ridge,
    }
    ModelClass = model_map.get(cfg["model_type"], RandomForestRegressor)

    model_kwargs = {"random_state": cfg["random_seed"]}
    if cfg["model_type"] in ("random_forest", "gradient_boosting"):
        model_kwargs["n_estimators"] = cfg.get("n_estimators", 100)
        model_kwargs["max_depth"] = cfg.get("max_depth", 5)
    elif cfg["model_type"] == "ridge":
        model_kwargs["alpha"] = cfg.get("alpha", 1.0)

    model = ModelClass(**model_kwargs)
    model.fit(X_train, y_train)

    # Save model
    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Training metrics
    y_pred_train = model.predict(X_train)
    train_rmse = float(np.sqrt(np.mean((y_train - y_pred_train)**2)))

    metrics = {
        "train_rmse": train_rmse,
        "model_type": cfg["model_type"],
    }
    if hasattr(model, "feature_importances_"):
        metrics["top_feature_idx"] = int(model.feature_importances_.argmax())

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model trained: {cfg['model_type']}")
    print(f"Training RMSE: {train_rmse:.4f}")

if __name__ == "__main__":
    main()
```

```python
# File: src/evaluate.py
"""Model evaluation stage."""

import numpy as np
import json
import pickle
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # Load model and test data
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {"test_rmse": rmse, "test_mae": mae, "test_r2": r2}
    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R²:   {r2:.4f}")

    # Generate residual plot for DVC plots
    residuals = y_test - y_pred
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15)
    axes[0].axhline(0, color="red", ls="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")

    axes[1].scatter(y_test, y_pred, alpha=0.4, s=15)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                  "r--", lw=1)
    axes[1].set_xlabel("Actual"); axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Actual vs. Predicted (R²={r2:.3f})")

    plt.tight_layout()
    plt.savefig("reports/figures/evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    # DVC metrics format: list of dicts for plots
    metrics_history = [{"step": i, "actual": float(a), "predicted": float(p)}
                       for i, (a, p) in enumerate(zip(y_test[:50], y_pred[:50]))]
    with open("metrics/predictions.json", "w") as f:
        json.dump(metrics_history, f)

if __name__ == "__main__":
    main()
```

```yaml
# File: dvc.yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw/dataset.csv
      - src/preprocess.py
    outs:
      - data/processed/
    params:
      - params.yaml:
          - preprocess.test_size
          - preprocess.random_seed
          - preprocess.scale_features

  train:
    cmd: python src/train.py
    deps:
      - data/processed/X_train.npy
      - data/processed/y_train.npy
      - src/train.py
    outs:
      - models/model.pkl
    params:
      - params.yaml:
          - train.model_type
          - train.n_estimators
          - train.max_depth
          - train.random_seed
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/X_test.npy
      - data/processed/y_test.npy
      - src/evaluate.py
    plots:
      - metrics/predictions.json:
          x: step
          y: actual
      - reports/figures/evaluation.png
    metrics:
      - metrics/test_metrics.json:
          cache: false
```

```bash
# Run the full pipeline
dvc repro

# View metrics
dvc metrics show

# Visualize the DAG
dvc dag
```

### Step 3: Experiment Tracking and Comparison

```python
# File: scripts/run_experiments.py
"""Run multiple experiments varying hyperparameters."""

import subprocess
import json
import yaml
from pathlib import Path
import pandas as pd

def run_dvc_experiment(params_override, exp_name=None):
    """Run a DVC experiment with given parameter overrides.

    Args:
        params_override: dict of param_path → value (e.g., "train.n_estimators" → 200)
        exp_name: optional experiment name
    Returns:
        dict with command result
    """
    cmd = ["dvc", "exp", "run"]

    if exp_name:
        cmd.extend(["--name", exp_name])

    for param_path, value in params_override.items():
        cmd.extend(["--set-param", f"{param_path}={value}"])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }

def get_experiments_table():
    """Retrieve experiment comparison table from DVC."""
    result = subprocess.run(
        ["dvc", "exp", "show", "--csv"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        import io
        return pd.read_csv(io.StringIO(result.stdout))
    return pd.DataFrame()

# Define experiment grid
experiments = [
    {
        "name": "rf-100-depth5",
        "params": {
            "train.model_type": "random_forest",
            "train.n_estimators": 100,
            "train.max_depth": 5,
        }
    },
    {
        "name": "rf-200-depth8",
        "params": {
            "train.model_type": "random_forest",
            "train.n_estimators": 200,
            "train.max_depth": 8,
        }
    },
    {
        "name": "gb-100-depth4",
        "params": {
            "train.model_type": "gradient_boosting",
            "train.n_estimators": 100,
            "train.max_depth": 4,
        }
    },
]

print("=== DVC Experiment Grid ===")
for exp in experiments:
    print(f"\nExperiment: {exp['name']}")
    for k, v in exp['params'].items():
        print(f"  {k} = {v}")

# In a real workflow, uncomment:
# for exp in experiments:
#     result = run_dvc_experiment(exp["params"], exp_name=exp["name"])
#     print(f"  {'OK' if result['success'] else 'FAILED'}: {exp['name']}")

# Load results from metrics files directly (as DVC substitute)
metrics_files = list(Path("metrics").glob("*.json"))
results = []
for f in metrics_files:
    with open(f) as fp:
        data = json.load(fp)
    if isinstance(data, dict) and "test_rmse" in data:
        data["file"] = str(f)
        results.append(data)

if results:
    df_results = pd.DataFrame(results)
    print("\n=== Current Metrics ===")
    print(df_results.to_string(index=False))
```

## Advanced Usage

### DVC Data Registry Pattern

```bash
# Register shared datasets accessible to all team projects

# In a central "data-registry" repo
dvc add data/gold/benchmark_dataset_v2.parquet
git commit -am "Add benchmark dataset v2"
git tag -a "benchmark-v2.0" -m "Benchmark dataset version 2"
dvc push

# In a downstream project
dvc import git@github.com:org/data-registry.git \
    data/gold/benchmark_dataset_v2.parquet \
    -o data/benchmark.parquet

# Get updates when registry is updated
dvc update benchmark.parquet.dvc
```

### Parameterized Pipeline via Python

```python
# File: scripts/parametric_run.py
"""Generate DVC pipeline configurations programmatically."""

import yaml
from pathlib import Path
from itertools import product

def generate_dvc_yaml(model_types, n_estimators_list):
    """Generate dvc.yaml with multiple training configurations."""
    stages = {}

    for model, n_est in product(model_types, n_estimators_list):
        stage_name = f"train_{model}_{n_est}"
        stages[stage_name] = {
            "cmd": f"python src/train.py --model {model} --n-estimators {n_est}",
            "deps": ["data/processed/X_train.npy", "src/train.py"],
            "outs": [f"models/{model}_{n_est}/model.pkl"],
            "metrics": [{f"metrics/{model}_{n_est}_metrics.json": {"cache": False}}],
            "params": ["params.yaml:train.random_seed"],
        }

    dvc_config = {"stages": stages}
    with open("dvc_grid.yaml", "w") as f:
        yaml.dump(dvc_config, f, default_flow_style=False)
    print(f"Generated dvc_grid.yaml with {len(stages)} stages")
    return stages

stages = generate_dvc_yaml(
    model_types=["random_forest", "gradient_boosting"],
    n_estimators_list=[50, 100, 200]
)

# Run specific stage
# dvc repro -f dvc_grid.yaml dvc_grid:train_random_forest_100
```

### Integration with MLflow

```python
# File: src/train_with_mlflow.py
"""Train model with MLflow tracking alongside DVC."""

import numpy as np
import yaml
import pickle
import json
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

def train_with_tracking():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_test  = np.load("data/processed/y_test.npy")

    cfg = params["train"]
    model = RandomForestRegressor(
        n_estimators=cfg.get("n_estimators", 100),
        max_depth=cfg.get("max_depth", 5),
        random_state=cfg.get("random_seed", 42)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_r2": float(r2_score(y_test, y_pred)),
    }

    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name="dvc_experiment"):
            mlflow.log_params(cfg)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
        print(f"Logged to MLflow at {MLFLOW_TRACKING_URI}")
    except ImportError:
        print("MLflow not installed — logging to JSON only")
    except Exception as e:
        print(f"MLflow logging failed ({e}) — continuing with JSON")

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    Path("models").mkdir(exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")

if __name__ == "__main__":
    train_with_tracking()
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `dvc: command not found` | DVC not installed | `pip install dvc`; ensure venv is activated |
| `ERROR: Git is not initialized` | Running `dvc init` outside git repo | `git init` first, then `dvc init` |
| Remote push fails (S3) | Missing AWS credentials | Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env vars |
| Pipeline doesn't re-run | DVC thinks stage is up-to-date | `dvc repro --force` to override cache |
| Stage skipped despite code change | Script not in `deps` list | Add script file to `deps:` in dvc.yaml |
| `dvc.lock` conflict (git merge) | Parallel pipeline runs | Resolve manually; keep the run with better metrics |
| Large cache on disk | Many experiments cached | `dvc gc -w` to remove unused cached files |

## External Resources

- [DVC documentation](https://dvc.org/doc/)
- [DVC Iterative MLOps tutorial](https://iterative.ai/blog/)
- [MLEM model deployment](https://mlem.ai/) — deploy DVC-tracked models
- [CML (Continuous Machine Learning)](https://cml.dev/) — CI/CD for ML with DVC
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS*.

## Examples

### Example 1: Dataset Versioning with Automatic Changelog

```python
import hashlib
import json
import datetime
from pathlib import Path
import pandas as pd

def version_dataset(filepath, version_log="data/version_log.json"):
    """Compute hash and log dataset version.

    Args:
        filepath: path to dataset file
        version_log: path to JSON version log
    Returns:
        version dict
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    df = pd.read_csv(filepath) if filepath.suffix == ".csv" else None
    version = {
        "file": str(filepath),
        "sha256": file_hash,
        "size_bytes": filepath.stat().st_size,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_rows": len(df) if df is not None else None,
        "n_cols": df.shape[1] if df is not None else None,
    }

    log = []
    if Path(version_log).exists():
        with open(version_log) as f:
            log = json.load(f)

    # Only log if hash changed
    if not log or log[-1]["sha256"] != file_hash:
        log.append(version)
        Path(version_log).parent.mkdir(exist_ok=True)
        with open(version_log, "w") as f:
            json.dump(log, f, indent=2)
        print(f"New version logged: {file_hash[:16]}...")
    else:
        print(f"Dataset unchanged: {file_hash[:16]}...")

    return version

v = version_dataset("data/raw/dataset.csv")
print(f"Dataset: {v['n_rows']} rows × {v['n_cols']} cols")
print(f"SHA256: {v['sha256'][:32]}...")
```

### Example 2: Automated Report After Pipeline Run

```python
import json
import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_run_report(output_path="reports/run_report.md"):
    """Generate a markdown report from DVC pipeline outputs."""
    Path("reports").mkdir(exist_ok=True)

    # Load metrics
    train_metrics = {}
    test_metrics = {}
    if Path("metrics/train_metrics.json").exists():
        with open("metrics/train_metrics.json") as f:
            train_metrics = json.load(f)
    if Path("metrics/test_metrics.json").exists():
        with open("metrics/test_metrics.json") as f:
            test_metrics = json.load(f)

    report_lines = [
        f"# Pipeline Run Report",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Training Results",
        f"- Model: {train_metrics.get('model_type', 'Unknown')}",
        f"- Training RMSE: {train_metrics.get('train_rmse', 'N/A'):.4f}"
          if isinstance(train_metrics.get('train_rmse'), float) else "- Training RMSE: N/A",
        "",
        "## Test Results",
        f"- Test RMSE: {test_metrics.get('test_rmse', 'N/A'):.4f}"
          if isinstance(test_metrics.get('test_rmse'), float) else "- Test RMSE: N/A",
        f"- Test MAE:  {test_metrics.get('test_mae', 'N/A'):.4f}"
          if isinstance(test_metrics.get('test_mae'), float) else "- Test MAE: N/A",
        f"- Test R²:   {test_metrics.get('test_r2', 'N/A'):.4f}"
          if isinstance(test_metrics.get('test_r2'), float) else "- Test R²: N/A",
        "",
        "## Reproducibility",
        "All outputs tracked by DVC. Reproduce with: `dvc repro`",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to {output_path}")
    return output_path

report = generate_run_report()
```
