---
name: open-data-management
description: >
  Use this Skill to deposit, document, and share research data: Zenodo/OSF API
  upload, FAIR assessment, DMP generation, data dictionary, and Dublin
  Core/DataCite metadata.
tags:
  - universal
  - open-data
  - FAIR
  - Zenodo
  - OSF
  - metadata
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
    - requests>=2.28
    - pandas>=1.5
    - pyyaml>=6.0
    - frictionless>=5.0
last_updated: "2026-03-17"
status: stable
---

# Open Data Management

> **TL;DR** — Deposit research data to Zenodo or OSF via API, generate rich
> Dublin Core / DataCite metadata, build a frictionless Data Package, write a
> machine-readable data dictionary, and produce a Data Management Plan (DMP)
> compliant with funder requirements.

---

## When to Use This Skill

Use this Skill whenever you need to:

- Deposit a dataset to a public repository and obtain a persistent DOI
- Satisfy journal data-availability statements or funder open-data mandates
- Write a machine-readable data dictionary and README for a dataset
- Create a frictionless Data Package (datapackage.json) for interoperability
- Draft a Data Management Plan (DMP) for grant proposals

| Task | Tool / standard |
|---|---|
| Data deposit | Zenodo REST API, OSF v2 API |
| Persistent identifier | DOI via DataCite (Zenodo auto-mints) |
| Metadata standard | DataCite Metadata Schema 4.4, Dublin Core |
| Data packaging | Frictionless Data Package |
| DMP writing | DMP Common Standard, DMPTool |
| FAIR assessment | F-UJI automated assessment |

---

## Background & Key Concepts

### FAIR Principles in Practice

| Principle | Concrete action |
|---|---|
| **Findable** | Assign a DOI; add keywords and author metadata |
| **Accessible** | Choose CC BY or CC0 license; use open protocols (HTTPS) |
| **Interoperable** | Use CSV/JSON/HDF5 formats; include a data dictionary |
| **Reusable** | Write a README with methods, provenance, and variable definitions |

### DataCite vs Dublin Core

- **DataCite Metadata Schema 4.4**: designed for research data and software.
  Required fields: identifier (DOI), creators, title, publisher, publicationYear,
  resourceType.
- **Dublin Core**: simpler 15-element standard. Good for general-purpose metadata
  interchange (OAI-PMH, institutional repositories).

### Zenodo vs OSF

| Feature | Zenodo | OSF |
|---|---|---|
| Primary use | Data/software archival | Project collaboration + preregistration |
| DOI | Auto-minted via DataCite | Via OSF or external |
| API | REST (sandbox + production) | REST v2 (osf.io/v2) |
| Storage limit | 50 GB per record (default) | 5 GB (free) |
| Versioning | Version DOI concept (new version = new record) | File versioning built-in |

---

## Environment Setup

```bash
conda create -n opendata python=3.11 -y
conda activate opendata
pip install requests pandas pyyaml "frictionless>=5.0"

# Verify frictionless
python -c "import frictionless; print(frictionless.__version__)"

# Register for a Zenodo API token at: https://zenodo.org/account/settings/applications/
# export ZENODO_TOKEN="<paste-your-key>"

# Register for an OSF Personal Access Token at: https://osf.io/settings/tokens/
# export OSF_TOKEN="<paste-your-key>"
```

---

## Core Workflow

### Step 1 — Zenodo API Deposit

```python
import os
import json
import requests
from pathlib import Path
from typing import Optional

# export ZENODO_TOKEN="<paste-your-key>"
ZENODO_TOKEN: str = os.getenv("ZENODO_TOKEN", "")
ZENODO_BASE = "https://zenodo.org/api"
ZENODO_SANDBOX = "https://sandbox.zenodo.org/api"  # use for testing


def zenodo_create_deposition(
    metadata: dict,
    use_sandbox: bool = True,
) -> dict:
    """
    Create a new Zenodo deposition (draft) and set its metadata.

    Args:
        metadata:    DataCite-compatible metadata dict (see example below).
        use_sandbox: Use the Zenodo sandbox for testing (default True).

    Returns:
        Deposition response dict including 'id', 'doi', 'links'.

    Example metadata:
        {
            "upload_type": "dataset",
            "title": "My Research Dataset v1.0",
            "creators": [{"name": "Smith, Jane", "affiliation": "MIT",
                          "orcid": "0000-0001-2345-6789"}],
            "description": "Processed data from the HF cohort study.",
            "access_right": "open",
            "license": "cc-by",
            "keywords": ["heart failure", "epidemiology"],
            "version": "1.0.0",
        }
    """
    base = ZENODO_SANDBOX if use_sandbox else ZENODO_BASE
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {ZENODO_TOKEN}"}

    # Create empty deposition
    resp = requests.post(f"{base}/deposit/depositions", json={}, headers=headers)
    resp.raise_for_status()
    dep = resp.json()
    dep_id = dep["id"]
    print(f"Created deposition ID: {dep_id}")

    # Set metadata
    resp2 = requests.put(
        f"{base}/deposit/depositions/{dep_id}",
        json={"metadata": metadata},
        headers=headers,
    )
    resp2.raise_for_status()
    print(f"Metadata set. Conceptual DOI: {dep['conceptdoi']}")
    return resp2.json()


def zenodo_upload_file(
    deposition_id: int,
    file_path: str,
    use_sandbox: bool = True,
) -> dict:
    """
    Upload a file to an existing Zenodo deposition.

    Args:
        deposition_id: Zenodo deposition integer ID.
        file_path:     Local path to the file to upload.
        use_sandbox:   Use Zenodo sandbox.

    Returns:
        File upload response dict.
    """
    base = ZENODO_SANDBOX if use_sandbox else ZENODO_BASE
    headers = {"Authorization": f"Bearer {ZENODO_TOKEN}"}
    bucket_url = f"{base}/deposit/depositions/{deposition_id}/files"

    file_path = Path(file_path)
    with open(file_path, "rb") as fh:
        resp = requests.post(
            bucket_url,
            data=fh,
            params={"filename": file_path.name},
            headers=headers,
        )
    resp.raise_for_status()
    print(f"Uploaded {file_path.name} ({resp.json()['filesize']} bytes)")
    return resp.json()


def zenodo_publish(deposition_id: int, use_sandbox: bool = True) -> dict:
    """
    Publish a Zenodo deposition and mint the DOI.

    WARNING: Publication is irreversible on production Zenodo.
    Always test on sandbox first.

    Args:
        deposition_id: Integer ID of the deposition to publish.
        use_sandbox:   Use Zenodo sandbox.

    Returns:
        Published deposition dict with 'doi' key.
    """
    base = ZENODO_SANDBOX if use_sandbox else ZENODO_BASE
    headers = {"Authorization": f"Bearer {ZENODO_TOKEN}"}
    resp = requests.post(
        f"{base}/deposit/depositions/{deposition_id}/actions/publish",
        headers=headers,
    )
    resp.raise_for_status()
    published = resp.json()
    print(f"Published! DOI: {published['doi']}")
    return published
```

### Step 2 — frictionless Data Package Generation

A Frictionless Data Package bundles your data files with machine-readable metadata
in a `datapackage.json` file, enabling automated validation and cross-tool ingestion.

```python
import pandas as pd
from frictionless import Package, Resource, Schema, Field
from pathlib import Path
import json


def infer_frictionless_type(dtype: str) -> str:
    """Map pandas dtype string to a Frictionless field type."""
    mapping = {
        "int64": "integer",
        "int32": "integer",
        "float64": "number",
        "float32": "number",
        "bool": "boolean",
        "object": "string",
        "datetime64[ns]": "datetime",
        "category": "string",
    }
    for k, v in mapping.items():
        if k in dtype:
            return v
    return "string"


def generate_data_package(
    data_files: dict[str, pd.DataFrame],
    package_name: str,
    title: str,
    description: str,
    version: str = "1.0.0",
    license_name: str = "CC-BY-4.0",
    output_dir: str = ".",
) -> dict:
    """
    Generate a frictionless datapackage.json for one or more datasets.

    Args:
        data_files:   Dict mapping filename stem -> DataFrame.
                      Files will be saved as CSV in output_dir.
        package_name: Machine-readable package identifier (kebab-case).
        title:        Human-readable package title.
        description:  Package description.
        version:      Semantic version string.
        license_name: SPDX license identifier.
        output_dir:   Directory to write CSV files and datapackage.json.

    Returns:
        datapackage.json content as a dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    resources = []
    for stem, df in data_files.items():
        csv_path = out / f"{stem}.csv"
        df.to_csv(csv_path, index=False)

        fields = []
        for col, dtype in df.dtypes.items():
            fields.append({
                "name": col,
                "type": infer_frictionless_type(str(dtype)),
                "description": f"Column: {col}",
            })

        resources.append({
            "name": stem,
            "path": f"{stem}.csv",
            "mediatype": "text/csv",
            "schema": {"fields": fields},
        })

    package = {
        "name": package_name,
        "title": title,
        "description": description,
        "version": version,
        "licenses": [{"name": license_name}],
        "resources": resources,
    }

    pkg_path = out / "datapackage.json"
    with open(pkg_path, "w") as f:
        json.dump(package, f, indent=2)
    print(f"Data package written to {pkg_path}")
    return package


def validate_data_package(package_path: str) -> bool:
    """
    Validate a frictionless Data Package using the frictionless library.

    Args:
        package_path: Path to the datapackage.json file.

    Returns:
        True if valid, False otherwise (errors printed to stdout).
    """
    from frictionless import validate
    report = validate(package_path)
    if report.valid:
        print("Data Package is VALID.")
    else:
        print("Data Package has ERRORS:")
        for error in report.flatten(["rowNumber", "fieldNumber", "message"]):
            print(f"  Row {error[0]}, Field {error[1]}: {error[2]}")
    return report.valid
```

### Step 3 — DMP Template Generation

```python
import os
from datetime import date
import yaml


def generate_dmp(
    project_title: str,
    principal_investigator: str,
    funder: str,
    grant_number: str,
    data_types: list[str],
    storage_plan: str,
    sharing_plan: str,
    preservation_period: str = "10 years",
    output_path: str = "data_management_plan.yaml",
) -> dict:
    """
    Generate a structured Data Management Plan (DMP) template.

    Covers sections required by major funders (NIH, NSF, UKRI, Horizon Europe).

    Args:
        project_title:         Full title of the research project.
        principal_investigator: Name of the PI.
        funder:                 Funding agency (e.g., 'NIH', 'NSF', 'Horizon Europe').
        grant_number:           Grant identifier.
        data_types:             List of data type descriptions
                                (e.g., ['clinical survey', 'MRI DICOM files']).
        storage_plan:           Where and how data will be stored during the project.
        sharing_plan:           How and when data will be shared (repository, license).
        preservation_period:    How long data will be retained after project end.
        output_path:            File to write the YAML DMP.

    Returns:
        DMP as a Python dictionary.
    """
    dmp = {
        "dmp": {
            "title": f"DMP for: {project_title}",
            "created": str(date.today()),
            "modified": str(date.today()),
            "contact": {
                "name": principal_investigator,
                "mbox": "pi@institution.edu",
                "contact_id": {"identifier": "https://orcid.org/0000-0000-0000-0000",
                               "type": "orcid"},
            },
            "project": [
                {
                    "title": project_title,
                    "funder_id": {"identifier": funder, "type": "fundref"},
                    "grant_id": {"identifier": grant_number, "type": "other"},
                }
            ],
            "dataset": [
                {
                    "type": dt,
                    "title": f"Dataset: {dt}",
                    "description": f"Data collected as part of {project_title}",
                    "data_quality_assurance": [
                        "Double data entry", "Range checks", "Audit trail"
                    ],
                    "distribution": [
                        {
                            "title": "Public repository deposit",
                            "access_url": "https://zenodo.org",
                            "available_until": (
                                str(date.today().year + 10) + "-12-31"
                            ),
                            "format": ["text/csv", "application/json"],
                            "license": [
                                {"license_ref": "https://creativecommons.org/licenses/by/4.0/",
                                 "start_date": str(date.today())}
                            ],
                        }
                    ],
                    "personal_data": "unknown",
                    "sensitive_data": "unknown",
                    "preservation_statement": preservation_period,
                }
                for dt in data_types
            ],
        }
    }

    with open(output_path, "w") as f:
        yaml.dump(dmp, f, default_flow_style=False, allow_unicode=True)
    print(f"DMP written to {output_path}")
    return dmp
```

---

## Advanced Usage

### Dublin Core Metadata Mapping

```python
def to_dublin_core(metadata: dict) -> dict:
    """
    Map a DataCite-style metadata dict to Dublin Core 15 elements.

    Args:
        metadata: Dict with keys: title, creators (list of name strings),
                  description, subject (list), date, identifier (DOI),
                  publisher, type, rights.

    Returns:
        Dublin Core dict with 'dc:' prefixed keys.
    """
    return {
        "dc:title": metadata.get("title", ""),
        "dc:creator": "; ".join(
            c.get("name", c) if isinstance(c, dict) else c
            for c in metadata.get("creators", [])
        ),
        "dc:subject": "; ".join(metadata.get("keywords", [])),
        "dc:description": metadata.get("description", ""),
        "dc:publisher": metadata.get("publisher", "Zenodo"),
        "dc:date": metadata.get("publication_date", str(date.today())),
        "dc:type": metadata.get("upload_type", "Dataset"),
        "dc:format": "text/csv",
        "dc:identifier": metadata.get("doi", ""),
        "dc:source": metadata.get("related_identifiers", [{}])[0].get("identifier", ""),
        "dc:rights": metadata.get("license", "CC-BY-4.0"),
    }
```

### Automated FAIR Assessment Checklist

```python
def assess_fair(dataset_dir: str) -> dict:
    """
    Run a simple automated FAIR self-assessment on a dataset directory.

    Checks for presence of: datapackage.json, README, LICENSE, DOI in README.

    Args:
        dataset_dir: Path to the dataset root directory.

    Returns:
        Dictionary of FAIR scores (0 or 1 per check) and overall percentage.
    """
    from pathlib import Path
    import re

    root = Path(dataset_dir)
    checks = {}

    # Findable
    checks["F1_has_doi"] = any(
        (root / f).exists() for f in ["README.md", "README.txt"]
    ) and bool(
        re.search(r"10\.\d{4,}/\S+",
                  (root / "README.md").read_text(errors="ignore") if (root / "README.md").exists() else "")
    )
    checks["F2_has_metadata"] = (root / "datapackage.json").exists()

    # Accessible
    checks["A1_open_access"] = (root / "LICENSE").exists() or (root / "LICENSE.txt").exists()

    # Interoperable
    checks["I1_standard_format"] = any(root.glob("*.csv")) or any(root.glob("*.json"))
    checks["I2_has_data_dictionary"] = any(
        f.name in ("codebook.csv", "data_dictionary.csv", "variables.csv")
        for f in root.iterdir() if f.is_file()
    )

    # Reusable
    checks["R1_has_readme"] = (root / "README.md").exists() or (root / "README.txt").exists()
    checks["R2_has_license"] = checks["A1_open_access"]

    score = sum(checks.values()) / len(checks) * 100
    checks["overall_percent"] = round(score, 1)
    return checks
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `401 Unauthorized` from Zenodo | Invalid or expired token | Regenerate token at zenodo.org/account/settings/applications |
| `400 Bad Request` on metadata | Missing required DataCite fields | Ensure `upload_type`, `title`, `creators`, `description` are present |
| Frictionless validation fails | Type mismatch (e.g., string in integer column) | Fix source data or change field type in datapackage.json |
| DOI not resolving | Deposition not published | Call `zenodo_publish()` after uploading files |
| OSF upload returns `403` | Project is private; token lacks write scope | Check OSF token permissions; verify project GUID |

---

## External Resources

- Zenodo REST API: <https://developers.zenodo.org>
- Zenodo sandbox: <https://sandbox.zenodo.org>
- OSF REST API: <https://developer.osf.io>
- Frictionless Data standards: <https://specs.frictionlessdata.io>
- DataCite Metadata Schema 4.4: <https://schema.datacite.org/meta/kernel-4>
- DMP Common Standard: <https://github.com/RDA-DMP-Common/RDA-DMP-Common-Standard>
- FAIR self-assessment tool: <https://www.f-uji.net>

---

## Examples

### Example 1 — Full Zenodo Sandbox Deposit

```python
import os
# export ZENODO_TOKEN="<paste-your-key>"

metadata = {
    "upload_type": "dataset",
    "title": "HF Cohort Study Dataset v1.0",
    "creators": [
        {"name": "Smith, Jane", "affiliation": "MIT",
         "orcid": "0000-0001-2345-6789"}
    ],
    "description": "Processed anonymized data from the heart failure cohort study.",
    "access_right": "open",
    "license": "cc-by",
    "keywords": ["heart failure", "cohort", "epidemiology"],
    "version": "1.0.0",
    "publication_date": "2026-03-17",
}

# Create deposition on sandbox
dep = zenodo_create_deposition(metadata, use_sandbox=True)
dep_id = dep["id"]

# Upload data files
zenodo_upload_file(dep_id, "data/processed.csv", use_sandbox=True)
zenodo_upload_file(dep_id, "datapackage.json",   use_sandbox=True)
zenodo_upload_file(dep_id, "README.md",          use_sandbox=True)

# Publish (get DOI)
published = zenodo_publish(dep_id, use_sandbox=True)
print(f"DOI: {published['doi']}")
```

### Example 2 — frictionless Data Package from DataFrames

```python
import pandas as pd
import numpy as np

# Create sample datasets
df_clinical = pd.DataFrame({
    "patient_id": range(1, 101),
    "age": np.random.randint(40, 85, 100),
    "ejection_fraction": np.random.uniform(20, 70, 100).round(1),
    "treatment_group": np.random.choice(["A", "B"], 100),
    "outcome_90d": np.random.choice([0, 1], 100, p=[0.8, 0.2]),
})

df_labs = pd.DataFrame({
    "patient_id": range(1, 101),
    "bnp_pgml": np.random.lognormal(5, 1, 100).round(1),
    "creatinine_mgdl": np.random.uniform(0.5, 3.0, 100).round(2),
})

pkg = generate_data_package(
    data_files={"clinical": df_clinical, "labs": df_labs},
    package_name="hf-cohort-v1",
    title="Heart Failure Cohort Study — Processed Data",
    description="Anonymized clinical and laboratory data from 100 HF patients.",
    output_dir="hf_dataset",
)

validate_data_package("hf_dataset/datapackage.json")
```

### Example 3 — DMP Template for NIH Grant

```python
dmp = generate_dmp(
    project_title="Cardiac Biomarkers in Heart Failure: A Prospective Cohort Study",
    principal_investigator="Jane Smith, MD PhD",
    funder="NIH",
    grant_number="R01-HL123456",
    data_types=[
        "Clinical survey data (REDCap CSV exports)",
        "Laboratory results (CSV)",
        "Echocardiography reports (PDF)",
    ],
    storage_plan=(
        "Data stored on institutional secure server with daily backups. "
        "REDCap hosted by university IT with HIPAA-compliant configuration."
    ),
    sharing_plan=(
        "De-identified dataset deposited to Zenodo under CC BY 4.0 within 12 months "
        "of primary publication. Code deposited to GitHub and Zenodo."
    ),
    preservation_period="10 years post project end",
    output_path="nih_r01_dmp.yaml",
)
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — Zenodo API, frictionless package, DMP generator |
