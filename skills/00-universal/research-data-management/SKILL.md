---
name: research-data-management
description: >
  Use this Skill for research data lifecycle management: codebook generation,
  DVC versioning, anonymization (k-anonymity, pseudonymization), and README
  templates.
tags:
  - universal
  - data-management
  - DVC
  - anonymization
  - codebook
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
    - pandas>=1.5
    - dvc>=3.0
    - numpy>=1.23
    - python-dotenv>=1.0
last_updated: "2026-03-18"
status: stable
---

# Research Data Management — Codebook, DVC, and Anonymization

> **TL;DR** — Manage the full research data lifecycle: auto-generate codebooks
> from pandas DataFrames, apply k-anonymity checks and pseudonymization
> (SHA-256 hashing, date shifting), version data files with DVC, define
> reproducible pipelines in dvc.yaml, and produce FAIR-compliant README
> documentation.

---

## When to Use This Skill

Use this Skill whenever you need to:

- Document a new dataset with a machine-readable codebook (variable names, types,
  missing rates, value ranges)
- Assess and enforce k-anonymity before sharing data with collaborators
- Pseudonymize PII columns (IDs, names, emails, dates) before archival or transfer
- Version large data files with DVC so they are tracked in Git without being
  committed to the repository
- Define and run a reproducible data pipeline using dvc.yaml stages
- Write a FAIR-compliant README_data.md with provenance, licensing, and citation
  information
- Self-assess dataset compliance against FAIR principles (Findable, Accessible,
  Interoperable, Reusable)

| Task | When to apply |
|---|---|
| Codebook generation | After data collection; before sharing |
| k-anonymity check | Before external data sharing or publication |
| Pseudonymization | Compliance with GDPR, HIPAA, or IRB conditions |
| DVC data versioning | Whenever data files exceed 10 MB or must be reproduced |
| dvc.yaml pipeline | Multi-step analysis with cacheable intermediate outputs |
| README_data.md | Data deposition in Zenodo, OSF, Figshare, or institutional repos |

---

## Background & Key Concepts

### Data Lifecycle

```
Collection → Cleaning → Analysis → Archival
    ↑              ↑         ↑          ↑
  IRB/ethics   Codebook   DVC pipeline  README + FAIR
  consent      anonymize  versioning    assessment
```

Each stage has specific data management actions:

| Stage | Key actions |
|---|---|
| Collection | Define variable names, units, and coding schema before collecting |
| Cleaning | Validate ranges, flag outliers, impute or remove missing values |
| Analysis | Version code and data together; cache pipeline outputs |
| Archival | Generate codebook, pseudonymize, deposit in repository with README |

### Codebook

A codebook describes every variable in a dataset. Minimum fields:

| Field | Description |
|---|---|
| variable | Column name in the data file |
| label | Human-readable description |
| dtype | Python/pandas dtype |
| n_unique | Number of distinct non-null values |
| missing_pct | Percentage of missing values |
| min / max | Range for numeric variables |
| example_values | Up to 3 representative values |

### k-Anonymity

A dataset satisfies **k-anonymity** if every combination of quasi-identifier
(QI) values appears in at least k rows. Quasi-identifiers are attributes that
could be linked to external records (age, sex, ZIP code, diagnosis date).

Generalization strategies to achieve k-anonymity:
- **Age** → age group (e.g., 5-year bands: 20–24, 25–29)
- **ZIP code** → first 3 digits (region)
- **Exact date** → year-month or year only

### Pseudonymization

**Pseudonymization** replaces direct identifiers with artificial keys. The
mapping is stored separately and protected. Direct identifiers:

- Patient ID, social security number → SHA-256 hash (deterministic, one-way)
- Name, email → hash or delete
- Dates of birth / events → date shifting (add a per-individual random offset
  that preserves within-individual intervals)

SHA-256 is preferred over MD5 because it has no known practical collision
attacks and is acceptable under GDPR recital 26.

### DVC — Data Version Control

DVC tracks large data files outside Git using content-addressable storage.
Key commands:

| Command | Purpose |
|---|---|
| `dvc init` | Initialize DVC in the Git repository |
| `dvc add <file>` | Track a data file; creates `<file>.dvc` |
| `dvc remote add` | Register storage (local path, S3, GCS, Azure) |
| `dvc push` | Upload tracked files to remote storage |
| `dvc pull` | Download tracked files from remote storage |
| `dvc repro` | Re-run pipeline stages whose dependencies changed |
| `dvc exp run` | Run an experiment with parameter overrides |

### FAIR Principles

| Principle | Criteria |
|---|---|
| **F**indable | Persistent identifier (DOI), searchable metadata, rich metadata |
| **A**ccessible | Open access protocol (HTTP/S), metadata accessible even if data is not |
| **I**nteroperable | Standard formats (CSV, JSON, NetCDF), controlled vocabularies |
| **R**eusable | Clear license (CC-BY, CC0), provenance, community standards |

---

## Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install required packages
pip install "pandas>=1.5" "dvc>=3.0" "numpy>=1.23" "python-dotenv>=1.0"

# Optional: DVC remote backends
pip install "dvc-s3"    # for Amazon S3
pip install "dvc-gdrive"  # for Google Drive

# Verify
python -c "import pandas, dvc, numpy; print('Setup OK')"
dvc version
```

For S3 remote storage, set credentials:

```bash
# export AWS_ACCESS_KEY_ID="<paste-your-key>"
# export AWS_SECRET_ACCESS_KEY="<paste-your-secret>"
# export DVC_REMOTE_URL="s3://my-bucket/dvc-store"
python -c "import os; print(os.getenv('AWS_ACCESS_KEY_ID', 'NOT SET'))"
```

Initialize DVC in your project:

```bash
git init my-research-project
cd my-research-project
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

---

## Core Workflow

### Step 1 — Automated Codebook from pandas DataFrame

```python
import pandas as pd
import numpy as np
from pathlib import Path


def generate_codebook(
    df: pd.DataFrame,
    variable_labels: dict[str, str] | None = None,
    n_example_values: int = 3,
    output_path: str | None = "codebook.csv",
) -> pd.DataFrame:
    """
    Auto-generate a variable-level codebook from a pandas DataFrame.

    Args:
        df:               Input DataFrame (any size, any dtypes).
        variable_labels:  Optional mapping from column name to human-readable label.
        n_example_values: Number of example values to include per variable.
        output_path:      If provided, save the codebook CSV to this path.

    Returns:
        Codebook DataFrame with one row per variable.

    Example:
        >>> cb = generate_codebook(df, variable_labels={"age": "Age in years"})
        >>> print(cb.to_string(index=False))
    """
    if variable_labels is None:
        variable_labels = {}

    rows = []
    for col in df.columns:
        series = df[col]
        n_total = len(series)
        n_missing = int(series.isna().sum())
        missing_pct = round(100 * n_missing / n_total, 2) if n_total > 0 else 0.0
        n_unique = int(series.nunique(dropna=True))

        # Min / max for numeric; mode for categorical
        col_min = col_max = ""
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if len(valid) > 0:
                col_min = str(round(float(valid.min()), 4))
                col_max = str(round(float(valid.max()), 4))

        # Example values (up to n_example_values unique non-null)
        examples = series.dropna().unique()[:n_example_values].tolist()
        example_str = " | ".join(str(v) for v in examples)

        rows.append({
            "variable": col,
            "label": variable_labels.get(col, ""),
            "dtype": str(series.dtype),
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "n_unique": n_unique,
            "min": col_min,
            "max": col_max,
            "example_values": example_str,
        })

    codebook = pd.DataFrame(rows)

    if output_path:
        codebook.to_csv(output_path, index=False)
        print(f"Codebook saved to {output_path} ({len(codebook)} variables)")

    return codebook


def flag_high_missingness(
    codebook: pd.DataFrame,
    threshold_pct: float = 20.0,
) -> pd.DataFrame:
    """Return variables with missing rate above threshold for review."""
    flagged = codebook[codebook["missing_pct"] > threshold_pct].copy()
    if len(flagged) > 0:
        print(f"Variables with > {threshold_pct}% missing data:")
        print(flagged[["variable", "missing_pct"]].to_string(index=False))
    else:
        print(f"No variables exceed {threshold_pct}% missingness threshold.")
    return flagged


if __name__ == "__main__":
    # Simulate a research dataset
    rng = np.random.default_rng(42)
    n = 200
    sample_data = {
        "participant_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.integers(18, 80, n).astype(float),
        "sex": rng.choice(["M", "F"], n),
        "bmi": rng.normal(26.5, 4.5, n).round(1),
        "diagnosis": rng.choice(["AD", "MCI", "Control"], n),
        "apoe4_carrier": rng.choice([0, 1], n),
        "mmse_score": rng.integers(15, 30, n).astype(float),
        "collection_date": pd.date_range("2022-01-01", periods=n, freq="D"),
    }
    # Inject missing values
    sample_data["bmi"][rng.choice(n, 25, replace=False)] = np.nan
    sample_data["mmse_score"][rng.choice(n, 10, replace=False)] = np.nan

    df_raw = pd.DataFrame(sample_data)

    labels = {
        "participant_id": "Unique participant identifier",
        "age": "Age at baseline (years)",
        "sex": "Biological sex (M/F)",
        "bmi": "Body mass index (kg/m²)",
        "diagnosis": "Clinical diagnosis at enrollment",
        "apoe4_carrier": "APOE4 allele carrier status (0=No, 1=Yes)",
        "mmse_score": "Mini-Mental State Examination score (0-30)",
        "collection_date": "Date of biological sample collection",
    }

    cb = generate_codebook(df_raw, variable_labels=labels)
    flag_high_missingness(cb, threshold_pct=10.0)
    print("\nCodebook preview:")
    print(cb.to_string(index=False))
```

### Step 2 — Pseudonymization Pipeline (Hash IDs, Shift Dates, Generalize Age)

```python
import hashlib
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

# Salt for HMAC-style hashing (never store in code — use env var or secrets vault)
# export HASH_SALT="<paste-your-salt>"
HASH_SALT: str = os.getenv("HASH_SALT", "default_salt_change_in_production")


def hash_identifier(value: str, salt: str = HASH_SALT) -> str:
    """
    One-way SHA-256 hash of a string identifier with a salt.

    Args:
        value: Original identifier (e.g., participant ID, email).
        salt:  Per-study salt to prevent rainbow table attacks.

    Returns:
        16-character hex digest (truncated SHA-256).
    """
    salted = f"{salt}:{value}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()[:16]


def pseudonymize_pii_columns(
    df: pd.DataFrame,
    id_cols: list[str],
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Replace PII identifier columns with SHA-256 pseudonyms.

    Args:
        df:              Input DataFrame containing PII.
        id_cols:         List of columns to pseudonymize (e.g., ['participant_id', 'email']).
        drop_original:   If True, drop the original columns after pseudonymization.

    Returns:
        DataFrame with pseudonymized ID columns (suffixed with '_pseudo').
    """
    df = df.copy()
    for col in id_cols:
        if col not in df.columns:
            continue
        df[f"{col}_pseudo"] = df[col].astype(str).apply(hash_identifier)
        if drop_original:
            df.drop(columns=[col], inplace=True)
    return df


def shift_dates(
    df: pd.DataFrame,
    date_cols: list[str],
    id_col: str,
    shift_range_days: tuple[int, int] = (-365, 365),
    seed: int = 99,
) -> pd.DataFrame:
    """
    Apply a per-individual random date shift to preserve within-subject intervals.

    Each individual receives a single random offset (in days), applied
    consistently to all their date columns. This preserves the time elapsed
    between events while obscuring absolute dates.

    Args:
        df:               DataFrame with date columns (datetime or string YYYY-MM-DD).
        date_cols:        List of date column names to shift.
        id_col:           Column identifying the individual (used for consistent offset).
        shift_range_days: Tuple (min_days, max_days) for the uniform random offset.
        seed:             Random seed for reproducibility.

    Returns:
        DataFrame with shifted date columns.
    """
    df = df.copy()
    # Parse date columns
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    rng = np.random.default_rng(seed)
    unique_ids = df[id_col].unique()
    offset_map = {
        uid: int(rng.integers(shift_range_days[0], shift_range_days[1] + 1))
        for uid in unique_ids
    }

    for col in date_cols:
        df[col] = df.apply(
            lambda row: (row[col] + timedelta(days=offset_map[row[id_col]]))
            if pd.notna(row[col]) else pd.NaT,
            axis=1,
        )
    return df


def generalize_age(
    df: pd.DataFrame,
    age_col: str,
    bin_width: int = 5,
    output_col: str | None = None,
) -> pd.DataFrame:
    """
    Replace exact age values with age-group bins (quasi-identifier generalization).

    Args:
        df:         Input DataFrame.
        age_col:    Column name containing age in years (numeric).
        bin_width:  Width of age bins in years (default 5: 20-24, 25-29, ...).
        output_col: Output column name; defaults to '{age_col}_group'.

    Returns:
        DataFrame with added age-group column.
    """
    df = df.copy()
    out_col = output_col or f"{age_col}_group"
    min_age = int(df[age_col].min()) - (int(df[age_col].min()) % bin_width)
    max_age = int(df[age_col].max()) + bin_width
    bins = list(range(min_age, max_age + bin_width, bin_width))
    labels = [f"{b}-{b + bin_width - 1}" for b in bins[:-1]]
    df[out_col] = pd.cut(df[age_col], bins=bins, labels=labels, right=False)
    return df


def check_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: list[str],
    k: int = 5,
) -> tuple[bool, pd.DataFrame]:
    """
    Check whether a dataset satisfies k-anonymity for given quasi-identifiers.

    Args:
        df:                DataFrame to assess.
        quasi_identifiers: List of quasi-identifier column names.
        k:                 Minimum group size required for k-anonymity.

    Returns:
        Tuple of (passes_k_anonymity: bool, violation_groups: DataFrame).
    """
    qi_groups = df.groupby(quasi_identifiers, observed=True).size().reset_index(name="count")
    violations = qi_groups[qi_groups["count"] < k]
    passes = len(violations) == 0
    status = "PASS" if passes else f"FAIL — {len(violations)} groups have count < {k}"
    print(f"k-anonymity check (k={k}): {status}")
    if not passes:
        print("Violating groups (sample):")
        print(violations.head(10).to_string(index=False))
    return passes, violations


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    df_sensitive = pd.DataFrame({
        "participant_id": [f"PT{i:04d}" for i in range(n)],
        "name": [f"Patient_{i}" for i in range(n)],
        "email": [f"patient{i}@clinic.org" for i in range(n)],
        "age": rng.integers(20, 85, n),
        "sex": rng.choice(["M", "F"], n),
        "zip_code": rng.choice(["10001", "10002", "10003", "90210"], n),
        "enroll_date": pd.date_range("2020-01-01", periods=n, freq="3D"),
        "last_visit": pd.date_range("2023-01-01", periods=n, freq="5D"),
        "diagnosis": rng.choice(["AD", "MCI", "Control"], n),
    })

    # Step 1: pseudonymize IDs
    df_pseudo = pseudonymize_pii_columns(
        df_sensitive, id_cols=["participant_id", "name", "email"]
    )

    # Step 2: shift dates
    df_pseudo = shift_dates(
        df_pseudo,
        date_cols=["enroll_date", "last_visit"],
        id_col="participant_id_pseudo",
        shift_range_days=(-180, 180),
    )

    # Step 3: generalize age
    df_pseudo = generalize_age(df_pseudo, age_col="age", bin_width=5)
    df_pseudo = df_pseudo.drop(columns=["age"])  # drop exact age

    # Step 4: k-anonymity check
    qi_cols = ["age_group", "sex", "zip_code"]
    passes, violations = check_k_anonymity(df_pseudo, qi_cols, k=5)

    df_pseudo.to_csv("pseudonymized_data.csv", index=False)
    print(f"\nPseudonymized dataset shape: {df_pseudo.shape}")
    print(df_pseudo.head(5).to_string(index=False))
```

### Step 3 — DVC Pipeline YAML and dvc repro

```python
import os
import subprocess
import textwrap
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


DVC_YAML_CONTENT = textwrap.dedent("""
stages:
  preprocess:
    cmd: python scripts/preprocess.py
    deps:
      - data/raw/cohort_data.csv
      - scripts/preprocess.py
    params:
      - params.yaml:
          - preprocess.missing_threshold
          - preprocess.outlier_z_threshold
    outs:
      - data/processed/cohort_clean.csv

  train:
    cmd: python scripts/train.py
    deps:
      - data/processed/cohort_clean.csv
      - scripts/train.py
    params:
      - params.yaml:
          - train.model_type
          - train.n_estimators
          - train.max_depth
          - train.random_state
    outs:
      - models/model.pkl
    metrics:
      - results/metrics.json:
          cache: false

  evaluate:
    cmd: python scripts/evaluate.py
    deps:
      - models/model.pkl
      - data/processed/cohort_clean.csv
      - scripts/evaluate.py
    outs:
      - results/predictions.csv
    plots:
      - results/roc_curve.csv:
          cache: false
      - results/shap_summary.png:
          cache: false
""").strip()


PARAMS_YAML_CONTENT = textwrap.dedent("""
preprocess:
  missing_threshold: 0.20       # drop columns with > 20% missing
  outlier_z_threshold: 3.0      # z-score threshold for outlier flagging

train:
  model_type: random_forest
  n_estimators: 200
  max_depth: 10
  random_state: 42

evaluate:
  threshold: 0.5                # decision threshold for binary classification
""").strip()


def write_dvc_pipeline(
    project_root: str = ".",
    dvc_yaml_path: str = "dvc.yaml",
    params_yaml_path: str = "params.yaml",
) -> None:
    """
    Write dvc.yaml and params.yaml to the project root.

    Args:
        project_root:    Root directory of the DVC-initialized project.
        dvc_yaml_path:   Relative path for the DVC pipeline file.
        params_yaml_path: Relative path for the parameters file.
    """
    root = Path(project_root)
    dvc_path = root / dvc_yaml_path
    params_path = root / params_yaml_path

    with open(dvc_path, "w", encoding="utf-8") as fh:
        fh.write(DVC_YAML_CONTENT + "\n")
    print(f"Written: {dvc_path}")

    with open(params_path, "w", encoding="utf-8") as fh:
        fh.write(PARAMS_YAML_CONTENT + "\n")
    print(f"Written: {params_path}")


def run_dvc_command(args: list[str], cwd: str = ".") -> int:
    """
    Run a DVC CLI command as a subprocess.

    Args:
        args: DVC subcommand and arguments (e.g., ['dvc', 'repro']).
        cwd:  Working directory to run the command in.

    Returns:
        Return code (0 = success).
    """
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def setup_dvc_remote(
    remote_name: str = "myremote",
    remote_url: str | None = None,
) -> None:
    """
    Configure a DVC remote storage backend.

    Args:
        remote_name: Alias for the remote (e.g., 'myremote', 's3remote').
        remote_url:  Storage URL. Reads DVC_REMOTE_URL env var if not provided.
                     Examples:
                       Local:  /data/dvc-store
                       S3:     s3://my-bucket/dvc-store
                       GCS:    gs://my-bucket/dvc-store
    """
    url = remote_url or os.getenv("DVC_REMOTE_URL", "/tmp/dvc-store")
    run_dvc_command(["dvc", "remote", "add", "--default", remote_name, url])
    print(f"DVC remote '{remote_name}' configured at: {url}")


if __name__ == "__main__":
    # In practice, run from inside the Git+DVC-initialized project directory
    write_dvc_pipeline()
    print("\nTo run the pipeline:")
    print("  dvc repro")
    print("  dvc push")
    print("\nTo track a data file:")
    print("  dvc add data/raw/cohort_data.csv")
    print("  git add data/raw/cohort_data.csv.dvc .gitignore")
    print('  git commit -m "Track raw data with DVC"')
```

---

## Advanced Usage

### FAIR Assessment Checklist

```python
FAIR_CHECKLIST = {
    "Findable": [
        "Dataset has a globally unique persistent identifier (DOI via Zenodo/Figshare)",
        "Metadata is registered in a searchable resource (DataCite, Google Dataset Search)",
        "Metadata includes dataset title, author, creation date, keywords",
        "Identifier is included in the metadata and in the data file itself",
    ],
    "Accessible": [
        "Dataset is retrievable via standard open protocol (HTTPS)",
        "Protocol is free and open (no proprietary software required)",
        "Access conditions are clearly stated (open / restricted / embargoed)",
        "Metadata remains accessible even if dataset is unavailable",
    ],
    "Interoperable": [
        "Data use a formal, accessible, shared, and broadly applicable language (CSV/JSON/NetCDF)",
        "Data use FAIR-compliant vocabularies and ontologies (MeSH, SNOMED, OBI)",
        "Data include qualified references to other (meta)data",
    ],
    "Reusable": [
        "Data are released with a clear and accessible data usage license (CC-BY or CC0)",
        "Data are associated with detailed provenance (collection methods, instruments)",
        "Data meet domain-relevant community standards (MIAME, CONSORT, CDISC)",
        "README includes citation information and contact for reuse requests",
    ],
}


def assess_fair_compliance(checklist: dict[str, list[str]]) -> dict:
    """
    Interactively (or programmatically) assess FAIR compliance.

    For automated use, pass pre-filled responses instead of prompting.
    Here we print the checklist for manual self-assessment.
    """
    scores = {}
    for principle, criteria in checklist.items():
        print(f"\n{'='*60}")
        print(f"  {principle}")
        print(f"{'='*60}")
        for i, criterion in enumerate(criteria, 1):
            print(f"  {i}. {criterion}")
        scores[principle] = len(criteria)
    total = sum(scores.values())
    print(f"\nTotal checklist items: {total}")
    print("Review each item and record Y/N in your DMP or README.")
    return scores


# Run the FAIR assessment
assess_fair_compliance(FAIR_CHECKLIST)
```

### README_data.md Template

```python
README_DATA_TEMPLATE = """
# Dataset: {dataset_title}

## Description
{description}

## Variables
See `codebook.csv` for a full variable dictionary including type, range,
missing rates, and example values.

## Provenance
- **Source**: {source}
- **Collection period**: {collection_period}
- **Collection method**: {collection_method}
- **IRB / Ethics approval**: {irb_number}

## File structure
```
data/
  raw/           # Original unmodified data files (tracked with DVC)
  processed/     # Cleaned and pseudonymized versions
codebook.csv     # Auto-generated variable dictionary
README_data.md   # This file
dvc.yaml         # Pipeline definition
params.yaml      # Analysis parameters
```

## License
{license}

## Citation
If you use this dataset, please cite:
{citation}

## Contact
{contact_email}
"""


def write_readme_data(
    output_path: str = "README_data.md",
    **fields,
) -> None:
    """Write a populated README_data.md using the standard template."""
    content = README_DATA_TEMPLATE.format(**{
        "dataset_title": fields.get("dataset_title", "Research Dataset"),
        "description": fields.get("description", ""),
        "source": fields.get("source", ""),
        "collection_period": fields.get("collection_period", ""),
        "collection_method": fields.get("collection_method", ""),
        "irb_number": fields.get("irb_number", ""),
        "license": fields.get("license", "CC-BY 4.0"),
        "citation": fields.get("citation", "Author et al. (Year). Title. DOI."),
        "contact_email": fields.get("contact_email", ""),
    })
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"README_data.md written to {output_path}")
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `dvc repro` shows no changes | No dependency has changed | Force re-run with `dvc repro --force` |
| `dvc push` authentication error | Missing cloud credentials | Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env vars |
| k-anonymity violations after generalization | Rare combination of QIs | Increase bin width; suppress or merge rare groups |
| SHA-256 collision risk | Not a practical concern | SHA-256 has no known practical collisions; safe for pseudonymization |
| Date shift changes interval lengths | Shifting by different offsets per visit | Use per-individual (not per-row) offset — assign once per `id_col` |
| Codebook shows wrong dtype | Mixed-type column | Cast explicitly before calling `generate_codebook()` |
| DVC `.dvc` files not committed | `.dvc` files not in Git staging | `git add *.dvc && git commit -m "Track data files"` |

---

## External Resources

- DVC documentation: <https://dvc.org/doc>
- FAIR principles: <https://www.go-fair.org/fair-principles/>
- GDPR pseudonymization guidance: <https://edpb.europa.eu/>
- Zenodo data repository: <https://zenodo.org/>
- CESSDA Data Management Expert Guide: <https://dmeg.cessda.eu/>
- UK Data Archive data management: <https://ukdataservice.ac.uk/learning-hub/research-data-management/>
- DMP Tool (Data Management Plan): <https://dmptool.org/>

---

## Examples

### Example 1 — Automated Codebook for a Clinical Dataset

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(10)
n = 150
df_clinical = pd.DataFrame({
    "subject_id": [f"S{i:04d}" for i in range(n)],
    "age": rng.integers(18, 90, n).astype(float),
    "sex": rng.choice(["M", "F", "Other"], n, p=[0.48, 0.48, 0.04]),
    "systolic_bp": rng.normal(130, 20, n).round(1),
    "diastolic_bp": rng.normal(80, 12, n).round(1),
    "hba1c": rng.normal(7.2, 1.5, n).round(2),
    "smoking_status": rng.choice(["Never", "Former", "Current"], n),
    "event_date": pd.date_range("2021-01-01", periods=n, freq="2D"),
})
df_clinical.loc[rng.choice(n, 20, replace=False), "hba1c"] = np.nan

labels = {
    "subject_id": "Unique study subject identifier",
    "age": "Age at enrollment (years)",
    "sex": "Self-reported sex",
    "systolic_bp": "Systolic blood pressure (mmHg)",
    "diastolic_bp": "Diastolic blood pressure (mmHg)",
    "hba1c": "Glycated hemoglobin (%)",
    "smoking_status": "Tobacco smoking history",
    "event_date": "Date of index clinical event",
}

cb = generate_codebook(df_clinical, variable_labels=labels,
                        output_path="clinical_codebook.csv")
flag_high_missingness(cb, threshold_pct=10.0)
```

### Example 2 — Pseudonymization Pipeline for Cohort Data

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(55)
n = 80
df_cohort = pd.DataFrame({
    "participant_id": [f"PT{i:04d}" for i in range(n)],
    "email": [f"user{i}@university.edu" for i in range(n)],
    "age": rng.integers(20, 75, n),
    "sex": rng.choice(["M", "F"], n),
    "zip_code": rng.choice(["10001", "10002", "20001", "90210", "60601"], n),
    "enroll_date": pd.date_range("2019-03-01", periods=n, freq="7D"),
    "follow_up_date": pd.date_range("2021-03-01", periods=n, freq="7D"),
    "outcome": rng.choice([0, 1], n, p=[0.7, 0.3]),
})

# Pseudonymize identifiers
df_anon = pseudonymize_pii_columns(df_cohort, id_cols=["participant_id", "email"])

# Shift dates consistently per individual
df_anon = shift_dates(
    df_anon,
    date_cols=["enroll_date", "follow_up_date"],
    id_col="participant_id_pseudo",
)

# Generalize age
df_anon = generalize_age(df_anon, age_col="age", bin_width=10)
df_anon = df_anon.drop(columns=["age"])

# Check k-anonymity
passes, violations = check_k_anonymity(
    df_anon, quasi_identifiers=["age_group", "sex", "zip_code"], k=3
)
df_anon.to_csv("cohort_pseudonymized.csv", index=False)
print(f"\nFinal anonymized dataset: {df_anon.shape}")
```

### Example 3 — DVC Pipeline yaml + dvc repro Walkthrough

```python
from pathlib import Path

# Write pipeline definition files
write_dvc_pipeline(project_root=".", dvc_yaml_path="dvc.yaml",
                   params_yaml_path="params.yaml")

# Typical DVC workflow after writing the pipeline:
print("""
# Add a raw data file to DVC tracking
dvc add data/raw/cohort_data.csv

# Commit the .dvc tracking file and .gitignore
git add data/raw/cohort_data.csv.dvc data/.gitignore dvc.yaml params.yaml
git commit -m "Add data pipeline and track raw data"

# Configure remote and push data
dvc remote add -d myremote /data/shared/dvc-store
dvc push

# Run the full pipeline
dvc repro

# Check what changed
dvc status
dvc diff

# Run an experiment with a different parameter
dvc exp run --set-param train.n_estimators=300

# Show all experiments
dvc exp show
""")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — codebook generation, pseudonymization, k-anonymity, DVC pipeline, FAIR checklist, README template |
