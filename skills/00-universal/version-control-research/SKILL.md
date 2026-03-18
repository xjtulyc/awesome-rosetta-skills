---
name: version-control-research
description: >
  Use this Skill for research version control: Git workflow for analysis code,
  DVC data pipelines, GitHub Actions for reproducibility CI, and .env secrets
  management.
tags:
  - universal
  - version-control
  - git
  - DVC
  - GitHub-Actions
  - reproducibility
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
    - pre-commit>=3.0
    - python-dotenv>=1.0
  system:
    - git
    - github-actions
last_updated: "2026-03-18"
status: stable
---

# Version Control for Research — Git, DVC, GitHub Actions, and Secrets

> **TL;DR** — Apply software-engineering best practices to research code:
> structured Git branching (main/dev/feature), semantic versioning for paper
> submissions, DVC experiment tracking, dvc.yaml pipelines, GitHub Actions CI
> for reproducibility checks, pre-commit hooks (black, flake8, nbstripout),
> and safe .env secrets management via python-dotenv.

---

## When to Use This Skill

Use this Skill whenever you need to:

- Set up a Git repository structure for a research analysis project from scratch
- Apply a branching strategy that separates stable code from active development
- Keep API keys, database passwords, and tokens out of version control
- Track large data files and ML model artifacts with DVC alongside Git commits
- Define a reproducible analysis pipeline with dvc.yaml and a params.yaml config
- Run DVC experiments with parameter sweeps and compare results
- Set up GitHub Actions to automatically re-run analyses and validate outputs on push
- Configure pre-commit hooks to enforce code style and strip Jupyter notebook outputs
- Tag Git commits at the point of paper submission with semantic version numbers

| Task | When to apply |
|---|---|
| Git branching setup | Start of every new research project |
| .gitignore + .env | Before the first commit |
| DVC pipeline | When analysis has ≥ 2 sequential steps or large data files |
| GitHub Actions CI | When collaboration or long-term reproducibility is required |
| pre-commit hooks | Any project with Python code or Jupyter notebooks |
| Semantic version tag | At paper submission, revision, and acceptance |

---

## Background & Key Concepts

### Git Branching for Research

Research projects benefit from a simplified Git flow:

| Branch | Purpose | Rules |
|---|---|---|
| `main` | Stable, reproducible code tied to paper versions | Only merge via PR; tag at submission |
| `dev` | Active analysis development | Direct commits allowed; merged to main when stable |
| `feature/<name>` | Isolated experiment or new analysis | Branch from dev; merge back when complete |

Naming convention for feature branches: `feature/exp-transformer-ablation`,
`feature/add-sensitivity-analysis`, `feature/fix-cohort-filtering`.

### Semantic Versioning for Analysis Code

Use [SemVer](https://semver.org/) to tag the state of the codebase at key
research milestones:

| Tag | Meaning | Trigger |
|---|---|---|
| `v1.0.0` | First paper submission | `git tag -a v1.0.0 -m "Submission to Nature Methods"` |
| `v1.1.0` | Major revision (new analyses added) | After reviewer-requested additions |
| `v1.0.1` | Minor fix (typo in script, updated figure) | Bug fix not affecting results |
| `v2.0.0` | Second paper using the same codebase | Incompatible structural changes |

### .env and Secrets Management

Never commit secrets to Git. Use a `.env` file that is `.gitignore`-listed and
load values at runtime with `python-dotenv`:

```
# .env  (never committed)
OPENAI_API_KEY=sk-...
DATABASE_PASSWORD=hunter2
QUALTRICS_API_TOKEN=abc123
```

At runtime:
```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
```

Provide a `.env.example` (committed) with placeholder values so collaborators
know which variables to set.

### DVC Experiment Tracking

DVC experiments extend Git commits with tracked parameter changes and metrics:

| Command | Effect |
|---|---|
| `dvc exp run` | Run pipeline; record params + metrics as an experiment |
| `dvc exp run --set-param train.n_estimators=300` | Override one parameter |
| `dvc exp show` | Table of all experiments with metrics and params |
| `dvc exp diff` | Compare two experiments |
| `dvc exp branch exp-best main` | Promote an experiment to a branch |

### pre-commit Hooks

`pre-commit` runs checks before each `git commit`. Relevant hooks for research:

| Hook | Package | What it does |
|---|---|---|
| `black` | `black` | Auto-format Python code to PEP 8 |
| `flake8` | `flake8` | Lint Python for style and logic errors |
| `nbstripout` | `nbstripout` | Remove cell outputs from Jupyter notebooks before commit |
| `check-yaml` | `pre-commit-hooks` | Validate YAML files |
| `end-of-file-fixer` | `pre-commit-hooks` | Ensure files end with a newline |

---

## Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install required packages
pip install "dvc>=3.0" "pre-commit>=3.0" "python-dotenv>=1.0"

# Optional DVC storage backends
pip install "dvc-s3" "dvc-gdrive"

# Verify
python -c "import dvc, pre_commit, dotenv; print('Setup OK')"
git --version
dvc version
pre-commit --version
```

One-time project initialization:

```bash
# Initialize Git and DVC in the project folder
git init research-project
cd research-project
dvc init
git add .dvc .dvcignore
git commit -m "chore: initialize Git and DVC"

# Install pre-commit hooks defined in .pre-commit-config.yaml
pre-commit install

# Create initial branch structure
git checkout -b dev
git checkout main
```

---

## Core Workflow

### Step 1 — .gitignore, .env Setup, and dotenv Loading

```python
import os
import pathlib
from dotenv import load_dotenv


# ── .gitignore template for research projects ─────────────────────────────────
GITIGNORE_CONTENT = """
# Data and outputs (tracked with DVC instead)
data/
outputs/
results/
models/
*.pkl
*.h5
*.parquet

# Secrets — NEVER commit these
.env
.env.local
*.pem
*_credentials.json

# Python artifacts
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
dist/
build/

# Jupyter notebooks (outputs stripped by nbstripout pre-commit hook)
.ipynb_checkpoints/

# OS and editor artifacts
.DS_Store
Thumbs.db
.idea/
.vscode/
*.swp

# DVC cache (managed by DVC itself)
.dvc/tmp/
""".lstrip()


# ── .env.example (committed; no real values) ──────────────────────────────────
ENV_EXAMPLE_CONTENT = """
# Copy this file to .env and fill in real values.
# export OPENAI_API_KEY="<paste-your-key>"
OPENAI_API_KEY=

# export ANTHROPIC_API_KEY="<paste-your-key>"
ANTHROPIC_API_KEY=

# export QUALTRICS_API_TOKEN="<paste-your-token>"
QUALTRICS_API_TOKEN=

# export QUALTRICS_DATA_CENTER="<your-datacenter-id>"
QUALTRICS_DATA_CENTER=

# export DATABASE_URL="<connection-string>"
DATABASE_URL=

# export DVC_REMOTE_URL="<storage-url>"
DVC_REMOTE_URL=
""".lstrip()


def write_gitignore(path: str = ".gitignore") -> None:
    """Write a research-appropriate .gitignore file."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(GITIGNORE_CONTENT)
    print(f"Written: {path}")


def write_env_example(path: str = ".env.example") -> None:
    """Write a .env.example template (safe to commit)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ENV_EXAMPLE_CONTENT)
    print(f"Written: {path}")


def load_env_safely(env_file: str = ".env") -> dict[str, str]:
    """
    Load environment variables from .env file using python-dotenv.

    The .env file must NOT be committed to Git. This function loads
    variables into os.environ and returns a sanitized summary.

    Args:
        env_file: Path to the .env file (default: '.env' in the working directory).

    Returns:
        Dictionary mapping variable names to presence status ('SET' or 'NOT SET').
    """
    env_path = pathlib.Path(env_file)
    if not env_path.exists():
        print(f"Warning: {env_file} not found. Using system environment variables.")
    else:
        load_dotenv(dotenv_path=env_path, override=False)

    # Check presence of expected keys without exposing values
    expected_keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "QUALTRICS_API_TOKEN",
        "DATABASE_URL",
        "DVC_REMOTE_URL",
    ]
    status = {
        k: "SET" if os.getenv(k, "") != "" else "NOT SET"
        for k in expected_keys
    }
    return status


if __name__ == "__main__":
    write_gitignore()
    write_env_example()
    env_status = load_env_safely()
    print("\nEnvironment variable status:")
    for k, v in env_status.items():
        print(f"  {k}: {v}")

    # Safe usage in analysis scripts:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("\nOpenAI key not set — AI-assisted features will be skipped.")
```

### Step 2 — dvc.yaml with Three Pipeline Stages and params.yaml

```python
import textwrap
from pathlib import Path


DVC_YAML = textwrap.dedent("""
stages:
  preprocess:
    desc: "Load raw data, validate, clean, and split into train/test"
    cmd: python scripts/preprocess.py
    deps:
      - data/raw/study_data.csv
      - scripts/preprocess.py
      - src/data_utils.py
    params:
      - params.yaml:
          - preprocess.missing_threshold
          - preprocess.test_size
          - preprocess.random_state
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
      - data/processed/feature_names.json

  train:
    desc: "Train the primary classification model"
    cmd: python scripts/train.py
    deps:
      - data/processed/train.csv
      - data/processed/feature_names.json
      - scripts/train.py
      - src/model_utils.py
    params:
      - params.yaml:
          - train.model_type
          - train.n_estimators
          - train.max_depth
          - train.min_samples_leaf
          - train.random_state
          - train.cv_folds
    outs:
      - models/model.pkl
      - models/preprocessor.pkl
    metrics:
      - results/train_metrics.json:
          cache: false

  evaluate:
    desc: "Evaluate model on held-out test set; generate SHAP plots"
    cmd: python scripts/evaluate.py
    deps:
      - models/model.pkl
      - models/preprocessor.pkl
      - data/processed/test.csv
      - scripts/evaluate.py
      - src/explainability.py
    params:
      - params.yaml:
          - evaluate.decision_threshold
          - evaluate.shap_n_samples
    outs:
      - results/predictions.csv
      - results/shap_summary.png
    metrics:
      - results/test_metrics.json:
          cache: false
    plots:
      - results/roc_curve.csv:
          cache: false
          x: fpr
          y: tpr
          x_label: "False Positive Rate"
          y_label: "True Positive Rate"
          title: "ROC Curve"
""").lstrip()


PARAMS_YAML = textwrap.dedent("""
preprocess:
  missing_threshold: 0.20      # Drop columns with > 20% missing values
  test_size: 0.20              # Fraction of data held out as test set
  random_state: 42

train:
  model_type: random_forest
  n_estimators: 200
  max_depth: 10
  min_samples_leaf: 5
  random_state: 42
  cv_folds: 5

evaluate:
  decision_threshold: 0.50
  shap_n_samples: 500          # Number of test samples for SHAP computation
""").lstrip()


PRE_COMMIT_CONFIG = textwrap.dedent("""
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
        language_version: python3
        args: ["--line-length=99"]

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=99", "--extend-ignore=E203,W503"]

  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
        files: \\.ipynb$
""").lstrip()


def write_pipeline_files(root: str = ".") -> None:
    """
    Write dvc.yaml, params.yaml, and .pre-commit-config.yaml to the project root.

    Args:
        root: Project root directory path.
    """
    p = Path(root)
    files = {
        "dvc.yaml": DVC_YAML,
        "params.yaml": PARAMS_YAML,
        ".pre-commit-config.yaml": PRE_COMMIT_CONFIG,
    }
    for filename, content in files.items():
        fpath = p / filename
        with open(fpath, "w", encoding="utf-8") as fh:
            fh.write(content)
        print(f"Written: {fpath}")

    print("\nNext steps:")
    print("  pre-commit install             # install hooks into .git/hooks")
    print("  dvc repro                      # run the full pipeline")
    print("  dvc exp run --set-param train.n_estimators=300  # experiment")
    print("  dvc exp show                   # compare experiments")


if __name__ == "__main__":
    write_pipeline_files()
```

### Step 3 — GitHub Actions CI yaml for Reproducibility Check

```python
import textwrap
from pathlib import Path


GITHUB_ACTIONS_CI = textwrap.dedent("""
name: Reproducibility CI

on:
  push:
    branches:
      - main
      - dev
  pull_request:
    branches:
      - main

jobs:
  reproduce-analysis:
    name: Reproduce analysis pipeline
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install dvc

      - name: Configure DVC remote (read-only token)
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.DVC_AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.DVC_AWS_SECRET_ACCESS_KEY }}
        run: |
          dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
          dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY

      - name: Pull DVC data
        run: dvc pull

      - name: Reproduce pipeline
        run: dvc repro

      - name: Validate test metrics
        run: |
          python - <<'PYEOF'
          import json, sys
          with open("results/test_metrics.json") as f:
              metrics = json.load(f)
          roc_auc = metrics.get("roc_auc", 0.0)
          threshold = 0.70
          print(f"Test ROC-AUC: {roc_auc:.4f} (threshold: {threshold})")
          if roc_auc < threshold:
              print(f"ERROR: ROC-AUC {roc_auc:.4f} is below threshold {threshold}")
              sys.exit(1)
          print("Metrics check PASSED.")
          PYEOF

      - name: Upload results as artifact
        uses: actions/upload-artifact@v4
        with:
          name: analysis-results
          path: results/
          retention-days: 30

  lint-and-format:
    name: Code quality checks
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install pre-commit
        run: pip install pre-commit

      - name: Run pre-commit hooks
        run: pre-commit run --all-files
""").lstrip()


def write_github_actions_ci(
    output_path: str = ".github/workflows/ci.yml",
) -> None:
    """
    Write the GitHub Actions CI workflow YAML file.

    Creates the .github/workflows/ directory if it does not exist.

    Args:
        output_path: Path for the workflow YAML file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(GITHUB_ACTIONS_CI)
    print(f"Written: {path}")
    print("\nAdd DVC secrets to your GitHub repository:")
    print("  Settings → Secrets and variables → Actions → New repository secret")
    print("  DVC_AWS_ACCESS_KEY_ID")
    print("  DVC_AWS_SECRET_ACCESS_KEY")


if __name__ == "__main__":
    write_github_actions_ci()
```

---

## Advanced Usage

### Git Tags for Paper Submissions

Use annotated tags to record the exact state of the codebase at key submission
milestones. Annotated tags store the tagger's name, email, date, and message.

```bash
# Tag at first submission
git tag -a v1.0.0 -m "Initial submission to Journal of Neuroscience (2026-03-18)"
git push origin v1.0.0

# Tag at revision submission
git tag -a v1.1.0 -m "Revision R1: added sensitivity analysis and Fig. S3"
git push origin v1.1.0

# List all tags
git tag -l --sort=-version:refname

# Check out the submission state (detached HEAD — read only)
git checkout v1.0.0

# Return to dev branch
git checkout dev
```

### DVC Experiment Workflow

```bash
# Run pipeline with default params
dvc exp run

# Run experiment with modified parameters
dvc exp run --set-param train.n_estimators=500 --set-param train.max_depth=15
dvc exp run --name exp-deeper-tree --set-param train.max_depth=20

# View all experiments in tabular format
dvc exp show

# Compare two experiments
dvc exp diff exp-deeper-tree

# Promote the best experiment to a named branch
dvc exp branch exp-deeper-tree feature/deeper-tree-model
git checkout feature/deeper-tree-model
```

### Protecting Secrets in GitHub Actions

Never echo or print secrets in CI steps. Use these safe patterns:

```python
import os
from dotenv import load_dotenv


def get_required_secret(key: str) -> str:
    """
    Retrieve a required secret from environment variables.

    Raises ValueError with a descriptive message if the variable is not set,
    instead of silently using an empty string.

    Args:
        key: Environment variable name.

    Returns:
        Secret value as string.

    Raises:
        ValueError: If the variable is not set or is empty.
    """
    load_dotenv()
    value = os.getenv(key, "")
    if not value:
        raise ValueError(
            f"Required secret '{key}' is not set. "
            f"Add it to your .env file (local) or GitHub Actions secrets (CI)."
        )
    return value


def get_optional_secret(key: str, default: str = "") -> str:
    """
    Retrieve an optional secret, returning default if not set.

    Use for non-critical API keys where the feature should be skipped
    gracefully if the key is absent.

    Args:
        key:     Environment variable name.
        default: Fallback value (default empty string).

    Returns:
        Secret value or default.
    """
    load_dotenv()
    return os.getenv(key, default)


# Safe usage pattern:
# export OPENAI_API_KEY="<paste-your-key>"
try:
    openai_key = get_required_secret("OPENAI_API_KEY")
    print("OpenAI key loaded successfully.")
except ValueError as exc:
    print(f"Skipping AI features: {exc}")
    openai_key = ""
```

### Directory Structure for a Reproducible Research Project

```
my-research-project/
├── .dvc/               # DVC internal files (committed)
├── .github/
│   └── workflows/
│       └── ci.yml      # GitHub Actions CI (this Skill)
├── data/
│   ├── raw/            # Original data tracked by DVC (not in Git)
│   └── processed/      # Pipeline outputs tracked by DVC
├── models/             # Trained models tracked by DVC
├── notebooks/          # Jupyter notebooks (outputs stripped by nbstripout)
├── results/            # Figures, metrics JSON (key outputs tracked by DVC)
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── data_utils.py
│   ├── model_utils.py
│   └── explainability.py
├── tests/              # Unit tests for analysis functions
├── .env                # Secrets — NOT committed (.gitignore-listed)
├── .env.example        # Template — committed; no real values
├── .gitignore
├── .pre-commit-config.yaml
├── dvc.yaml            # Pipeline definition
├── params.yaml         # Analysis parameters
├── requirements.txt    # Python dependencies with pinned versions
└── README.md
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `dvc repro` reruns unchanged stages | params.yaml modification detected | Only change params you intend to; use `--downstream` to limit scope |
| `pre-commit` fails on `black` | Formatting issues in committed code | Run `black .` manually; then re-stage and commit |
| `nbstripout` does not strip outputs | Hook not installed | Run `pre-commit install` in the repo root |
| GitHub Actions `dvc pull` fails | Missing DVC remote credentials | Add `DVC_AWS_ACCESS_KEY_ID` and `DVC_AWS_SECRET_ACCESS_KEY` as repo secrets |
| `git tag` push fails | No push access or tag already exists | Check `git remote -v`; delete conflicting tag with `git tag -d <tag>` |
| `.env` committed accidentally | `.gitignore` added after first commit | `git rm --cached .env && git commit -m "Remove .env from tracking"` |
| `flake8` line-length errors | Default 79-char limit too strict | Set `--max-line-length=99` in `.pre-commit-config.yaml` |
| DVC experiments not showing | Pipeline not run with `dvc exp run` | Run `dvc exp run` instead of `python script.py` directly |

---

## External Resources

- Git branching model: <https://nvie.com/posts/a-successful-git-branching-model/>
- Semantic versioning: <https://semver.org/>
- DVC experiment tracking: <https://dvc.org/doc/user-guide/experiment-management>
- GitHub Actions documentation: <https://docs.github.com/en/actions>
- pre-commit framework: <https://pre-commit.com/>
- nbstripout: <https://github.com/kynan/nbstripout>
- python-dotenv: <https://github.com/theskumar/python-dotenv>
- The Turing Way (reproducible research): <https://the-turing-way.netlify.app/>
- TIER Protocol 4.0 (reproducible project structure): <https://www.projecttier.org/tier-protocol/protocol-4-0/>

---

## Examples

### Example 1 — .gitignore, .env.example, and dotenv Loading

```python
# Run once at project setup to write the standard research .gitignore
# and .env.example, then check which secrets are loaded.
write_gitignore(".gitignore")
write_env_example(".env.example")

# Simulate loading from a populated .env file
env_status = load_env_safely(".env")
print("\nSecrets status:")
for k, v in env_status.items():
    marker = "[OK]" if v == "SET" else "[MISSING]"
    print(f"  {marker} {k}")
```

### Example 2 — Full dvc.yaml with 3 Stages and params.yaml

```python
# Write the complete pipeline files to the current directory
write_pipeline_files(".")

# After running this, execute in the terminal:
#   dvc repro                              -> run all three stages
#   dvc exp run --set-param train.max_depth=15  -> experiment
#   dvc exp show                           -> compare results table
#   git add dvc.yaml params.yaml .pre-commit-config.yaml
#   git commit -m "feat: add DVC pipeline and pre-commit hooks"
```

### Example 3 — GitHub Actions CI yaml for Full Reproducibility Check

```python
# Write the CI workflow to .github/workflows/ci.yml
write_github_actions_ci(".github/workflows/ci.yml")

# After writing, commit and push to trigger the workflow:
#   git add .github/workflows/ci.yml
#   git commit -m "ci: add reproducibility CI workflow"
#   git push origin main

# Tag the submission state:
#   git tag -a v1.0.0 -m "Submission to PLOS Computational Biology, 2026-03-18"
#   git push origin v1.0.0
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Git branching, .env management, DVC pipeline, GitHub Actions CI, pre-commit hooks, semantic versioning |
