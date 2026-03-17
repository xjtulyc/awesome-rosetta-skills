---
name: bids-neuroimaging
description: >
  Organize, validate, and query neuroimaging datasets in BIDS 1.8 format using
  pybids, mne-bids, and datalad; covers EEG, fMRI, and derivatives conventions.
tags:
  - neuroscience
  - bids
  - neuroimaging
  - mne-bids
  - pybids
  - data-management
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
    - mne>=1.5.0
    - mne-bids>=0.13.0
    - pybids>=0.16.0
    - pandas>=2.0.0
    - numpy>=1.24.0
    - datalad>=0.19.0
  system:
    - nodejs>=18.0.0
last_updated: "2026-03-17"
status: "stable"
---

# BIDS Neuroimaging: Organizing and Querying Brain Data

> **One-line summary**: This Skill helps researchers convert raw neuroimaging data to BIDS 1.8
> format, validate the structure, query datasets with pybids, and share via OpenNeuro/datalad.

---

## When to Use This Skill

Use this Skill in the following scenarios:

- When you need to organize **EEG, MEG, or fMRI data** into a standards-compliant BIDS directory
- When your data has a **multi-subject, multi-session** structure requiring consistent naming
- When you need to **query a BIDS dataset** to filter by task, subject, run, or modality
- When you are using **OpenNeuro** to download or share public neuroimaging datasets
- When you need to produce **events.tsv files** and sidecar JSON metadata from experimental logs
- When preprocessing outputs must conform to **BIDS derivatives** conventions

**Trigger keywords**: BIDS, Brain Imaging Data Structure, pybids, BIDSLayout, mne-bids,
write_raw_bids, events.tsv, participants.tsv, dataset_description.json, datalad, OpenNeuro,
neuroimaging data organization, BIDS validator

---

## Background & Key Concepts

### BIDS Directory Hierarchy

The Brain Imaging Data Structure (BIDS; Gorgolewski et al., 2016) is a community standard for
organizing neuroimaging data. The top-level layout is:

```
<dataset_root>/
  dataset_description.json   # Dataset-level metadata (required)
  participants.tsv            # Demographic table
  participants.json           # Column descriptions
  sub-<label>/
    ses-<label>/              # Optional session folder
      eeg/                    # Modality folder (eeg, meg, fmri, anat, ...)
        sub-<label>_ses-<label>_task-<label>_run-<index>_eeg.edf
        sub-<label>_ses-<label>_task-<label>_run-<index>_eeg.json  # Sidecar
        sub-<label>_ses-<label>_task-<label>_run-<index>_channels.tsv
        sub-<label>_ses-<label>_task-<label>_run-<index>_events.tsv
  derivatives/
    pipeline-name/
      sub-<label>/...         # Preprocessed outputs follow same layout
```

BIDS entities are key-value pairs separated by underscores. The order is fixed:
`sub` → `ses` → `task` → `acq` → `run` → `suffix.extension`.

### Events and Timing

The events.tsv file records stimulus/response onsets with three mandatory columns:

| Column | Description |
|:-------|:------------|
| `onset` | Event start time in seconds from recording start |
| `duration` | Duration in seconds (0 for instantaneous) |
| `trial_type` | String label for the event category |

Additional columns (e.g., `response_time`, `stim_file`) are permitted.

### Comparison with Related Approaches

| Approach | Best for | Key assumption | Limitation |
|:---------|:---------|:---------------|:-----------|
| BIDS 1.8 | Multi-modal human neuroimaging | Consistent entities across files | Learning curve for new modalities |
| NWB 2.x | Invasive electrophysiology | Hierarchical HDF5 objects | Less adopted for non-invasive EEG |
| EEGLab STUDY | EEG-only pipelines | MATLAB ecosystem | Not interoperable with Python tools |
| LORIS | Clinical database | Longitudinal patient data | Infrastructure-heavy deployment |

---

## Environment Setup

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install Python dependencies
pip install mne>=1.5.0 mne-bids>=0.13.0 pybids>=0.16.0 pandas>=2.0.0 numpy>=1.24.0

# Install datalad for dataset management
pip install datalad>=0.19.0

# Install the BIDS validator (requires Node.js >= 18)
npm install -g bids-validator
```

### Verify Installation

```python
# Run this to confirm the environment is configured correctly
import mne
import mne_bids
import bids

print(f"mne version:       {mne.__version__}")
print(f"mne-bids version:  {mne_bids.__version__}")
print(f"pybids version:    {bids.__version__}")
# Expected: versions 1.5+, 0.13+, 0.16+ respectively
```

```bash
# Verify BIDS validator
bids-validator --version
# Expected: 1.x.x or higher
```

---

## Core Workflow

### Step 1: Create dataset_description.json and participants.tsv

Every BIDS dataset requires a root-level `dataset_description.json` and a `participants.tsv`.

```python
import json
import pandas as pd
from pathlib import Path

BIDS_ROOT = Path("my_bids_dataset")
BIDS_ROOT.mkdir(parents=True, exist_ok=True)

# --- dataset_description.json -------------------------------------------------
dataset_description = {
    "Name": "Auditory Oddball EEG Study",
    "BIDSVersion": "1.8.0",
    "DatasetType": "raw",
    "License": "CC0",
    "Authors": ["Jane Smith", "John Doe"],
    "Acknowledgements": "Data collected at ExampleLab.",
    "HowToAcknowledge": "Please cite Smith et al. (2024).",
    "ReferencesAndLinks": ["https://doi.org/10.xxxx/example"],
    "DatasetDOI": "",
}

with open(BIDS_ROOT / "dataset_description.json", "w") as f:
    json.dump(dataset_description, f, indent=2)
print("Written: dataset_description.json")

# --- participants.tsv ---------------------------------------------------------
participants = pd.DataFrame(
    {
        "participant_id": ["sub-01", "sub-02", "sub-03"],
        "age": [24, 31, 28],
        "sex": ["M", "F", "M"],
        "handedness": ["right", "right", "left"],
        "group": ["control", "control", "patient"],
    }
)
participants.to_csv(BIDS_ROOT / "participants.tsv", sep="\t", index=False)

# Companion sidecar describes each column
participants_json = {
    "age": {"Description": "Age at time of scan", "Units": "years"},
    "sex": {
        "Description": "Biological sex",
        "Levels": {"M": "male", "F": "female"},
    },
    "handedness": {
        "Description": "Dominant hand",
        "Levels": {"right": "right-handed", "left": "left-handed"},
    },
    "group": {
        "Description": "Study group",
        "Levels": {"control": "healthy control", "patient": "clinical patient"},
    },
}
with open(BIDS_ROOT / "participants.json", "w") as f:
    json.dump(participants_json, f, indent=2)
print("Written: participants.tsv and participants.json")
```

**Data requirements**:
- `participant_id`: must match `sub-<label>` directory names exactly
- `age`, `sex`: BIDS-recommended demographic columns
- All paths use forward slashes and lowercase entity keys

### Step 2: Convert EEG Data with mne-bids

`mne_bids.write_raw_bids` handles file conversion, channel coordinate writing, and sidecar JSON
generation from an MNE Raw object and a BIDSPath descriptor.

```python
import mne
import mne_bids
import numpy as np
import pandas as pd
from pathlib import Path

BIDS_ROOT = Path("my_bids_dataset")

# --- Simulate a raw EEG file (replace with mne.io.read_raw_edf / read_raw_bdf) ---
n_channels = 32
sfreq = 250.0
duration_s = 120.0
n_samples = int(sfreq * duration_s)

rng = np.random.default_rng(seed=42)
data = rng.normal(scale=5e-6, size=(n_channels, n_samples))  # ~5 µV noise

ch_names = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
    "FC1", "FC2", "CP1", "CP2",
    "AF3", "AF4", "FC5", "FC6", "CP5", "CP6",
    "EOGh", "EOGv",
]
ch_types = ["eeg"] * 30 + ["eog", "eog"]

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
info["line_freq"] = 50.0  # European power line

raw = mne.io.RawArray(data, info)
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="ignore")

# --- Add stimulus annotations (simulated events at 250, 750, 1250 ... ms) ---
event_onsets = np.arange(1.0, 100.0, 2.5)  # Every 2.5 s
event_durations = np.zeros(len(event_onsets))
# Alternate standard (80%) and deviant (20%)
event_descriptions = [
    "deviant" if i % 5 == 4 else "standard"
    for i in range(len(event_onsets))
]
annotations = mne.Annotations(
    onset=event_onsets,
    duration=event_durations,
    description=event_descriptions,
)
raw.set_annotations(annotations)

# --- Define BIDSPath ---
bids_path = mne_bids.BIDSPath(
    subject="01",
    session="01",
    task="oddball",
    run="1",
    datatype="eeg",
    root=BIDS_ROOT,
)

# --- Write to BIDS (creates sidecar JSON, channels.tsv, events.tsv) ---
mne_bids.write_raw_bids(
    raw,
    bids_path=bids_path,
    overwrite=True,
    verbose=False,
    format="EDF",          # Save as EDF; use "auto" to keep original format
    events=None,           # Derived from annotations automatically
    event_id={"standard": 1, "deviant": 2},
)
print(f"BIDS EEG written to: {bids_path.fpath}")

# Inspect generated sidecar
sidecar_path = bids_path.copy().update(suffix="eeg", extension=".json").fpath
import json
with open(sidecar_path) as f:
    sidecar = json.load(f)
print("EEG sidecar JSON keys:", list(sidecar.keys()))
```

**Parameter reference**:

| Parameter | Meaning | Recommended value | Notes |
|:----------|:--------|:------------------|:------|
| `format` | Output file format | `"EDF"` or `"auto"` | `"auto"` preserves original |
| `overwrite` | Overwrite existing files | `True` during development | Set to `False` for production |
| `event_id` | Annotation-to-code mapping | Dict matching annotation descriptions | Used to build events.tsv |
| `verbose` | Verbosity level | `False` or `"WARNING"` | MNE log level string or bool |

### Step 3: Query the BIDS Dataset with pybids BIDSLayout

`BIDSLayout` provides a Pythonic API to discover and filter files in a BIDS dataset.

```python
from bids import BIDSLayout
import pandas as pd
from pathlib import Path

BIDS_ROOT = Path("my_bids_dataset")

# --- Initialize layout (indexes all files on first call) ---------------------
layout = BIDSLayout(str(BIDS_ROOT), validate=False)

# --- High-level queries -------------------------------------------------------
subjects = layout.get_subjects()
tasks    = layout.get_tasks()
sessions = layout.get_sessions()
print(f"Subjects: {subjects}")
print(f"Tasks:    {tasks}")
print(f"Sessions: {sessions}")

# --- Filter files by entity --------------------------------------------------
# Get all EEG data files for sub-01
eeg_files = layout.get(
    subject="01",
    datatype="eeg",
    suffix="eeg",
    extension=[".edf", ".bdf", ".fif", ".set"],
)
for f in eeg_files:
    print(f"  {f.path}")

# --- Access the events.tsv for a specific file --------------------------------
events_files = layout.get(
    subject="01",
    task="oddball",
    suffix="events",
    extension=".tsv",
)
for ef in events_files:
    events_df = pd.read_csv(ef.path, sep="\t")
    print(f"\nevents.tsv — {ef.path}")
    print(events_df.head())

# --- Build a summary DataFrame of all EEG files ------------------------------
all_eeg = layout.get(datatype="eeg", suffix="eeg", extension=".edf")
summary_records = []
for f in all_eeg:
    summary_records.append(
        {
            "subject":  f.entities.get("subject"),
            "session":  f.entities.get("session"),
            "task":     f.entities.get("task"),
            "run":      f.entities.get("run"),
            "path":     f.path,
        }
    )
summary_df = pd.DataFrame(summary_records)
print("\nDataset summary:")
print(summary_df.to_string())
```

### Step 4: Validate with bids-validator

```bash
# Validate the entire dataset (requires Node.js bids-validator)
bids-validator my_bids_dataset --verbose

# Non-interactive mode (suitable for CI pipelines)
bids-validator my_bids_dataset --json 2>/dev/null | python -c "
import sys, json
report = json.load(sys.stdin)
errors = report.get('issues', {}).get('errors', [])
warnings = report.get('issues', {}).get('warnings', [])
print(f'Errors: {len(errors)}, Warnings: {len(warnings)}')
for e in errors[:5]:
    print(' ERROR:', e.get('key'), '-', e.get('reason'))
"
```

**Interpreting results**:
- **Errors**: must be fixed before the dataset can be published to OpenNeuro
- **Warnings**: recommended to fix but do not block submission
- **Common errors**: missing `IntendedFor` in fieldmap sidecar, wrong entity order in filename

---

## Advanced Usage

### Derivatives Folder Conventions

BIDS derivatives store preprocessed data under `derivatives/<pipeline-name>/`, maintaining the
same subject/session/datatype hierarchy and appending a `desc-<label>` entity.

```python
import mne
import mne_bids
from pathlib import Path

BIDS_ROOT = Path("my_bids_dataset")
DERIV_ROOT = BIDS_ROOT / "derivatives" / "mne-preprocessing"
DERIV_ROOT.mkdir(parents=True, exist_ok=True)

# Write derivatives dataset_description.json
import json
deriv_desc = {
    "Name": "MNE Preprocessing Pipeline",
    "BIDSVersion": "1.8.0",
    "DatasetType": "derivative",
    "GeneratedBy": [
        {
            "Name": "mne-bids",
            "Version": mne_bids.__version__,
            "CodeURL": "https://mne.tools/mne-bids/",
        }
    ],
    "SourceDatasets": [
        {"DOI": "doi:10.xxxx/example", "URL": "https://openneuro.org/datasets/ds001"}
    ],
}
with open(DERIV_ROOT / "dataset_description.json", "w") as f:
    json.dump(deriv_desc, f, indent=2)

# Assume `raw_clean` is a preprocessed MNE Raw object
# Save preprocessed EEG as derivatives
deriv_path = mne_bids.BIDSPath(
    subject="01",
    session="01",
    task="oddball",
    run="1",
    datatype="eeg",
    suffix="eeg",
    description="preprocessed",   # desc-preprocessed entity
    root=DERIV_ROOT,
    check=False,                   # Skip BIDS validation for derivatives
)

# Write epochs as derivative (FIF format preserves MNE metadata)
# epochs.save(str(deriv_path.update(suffix="epo", extension=".fif").fpath), overwrite=True)
print(f"Derivatives path: {deriv_path.fpath}")
```

### Download an OpenNeuro Dataset via datalad

```bash
# Install datalad and the git-annex backend
pip install datalad
# On Linux: sudo apt install git-annex
# On macOS: brew install git-annex

# List available datasets (browse openneuro.org for ds-numbers)
# Install (clone metadata only; no large files downloaded yet)
datalad install https://github.com/OpenNeuroDatasets/ds003490.git
cd ds003490

# Download only sub-01 data
datalad get sub-01/

# Download the entire dataset (may be large)
datalad get .
```

```python
import subprocess
from pathlib import Path

def datalad_get_subject(dataset_url: str, subject_id: str, local_dir: str) -> bool:
    """
    Clone an OpenNeuro dataset and download a single subject's files.

    Parameters
    ----------
    dataset_url : str
        GitHub URL of the OpenNeuro dataset, e.g.
        'https://github.com/OpenNeuroDatasets/ds003490.git'
    subject_id : str
        Subject label without 'sub-' prefix, e.g. '01'
    local_dir : str
        Local path to clone into.

    Returns
    -------
    bool : True if all commands succeeded.
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        result = subprocess.run(
            ["datalad", "install", dataset_url, str(local_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"datalad install failed: {result.stderr}")
            return False
        print(f"Installed dataset to {local_path}")

    subject_dir = local_path / f"sub-{subject_id}"
    result = subprocess.run(
        ["datalad", "get", str(subject_dir)],
        capture_output=True, text=True, cwd=str(local_path),
    )
    if result.returncode != 0:
        print(f"datalad get failed: {result.stderr}")
        return False

    print(f"Downloaded sub-{subject_id} to {subject_dir}")
    return True


# Example (replace with a real dataset URL)
# datalad_get_subject(
#     "https://github.com/OpenNeuroDatasets/ds003490.git",
#     subject_id="01",
#     local_dir="ds003490_local",
# )
```

### Building events.tsv from a Log File

```python
import pandas as pd
import numpy as np
from pathlib import Path

def log_to_events_tsv(
    log_csv: str,
    onset_col: str,
    trial_type_col: str,
    duration_col: str | None = None,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert a stimulus presentation log CSV to BIDS events.tsv format.

    Parameters
    ----------
    log_csv : str
        Path to the experiment log file.
    onset_col : str
        Column containing event onset times in seconds.
    trial_type_col : str
        Column containing event type labels.
    duration_col : str, optional
        Column for event durations. If None, duration is set to 0.
    extra_cols : list of str, optional
        Additional columns to include (e.g., 'response_time', 'accuracy').

    Returns
    -------
    pd.DataFrame with BIDS-compliant column order.
    """
    log_df = pd.read_csv(log_csv)

    events = pd.DataFrame()
    events["onset"] = log_df[onset_col].astype(float)
    events["duration"] = (
        log_df[duration_col].astype(float) if duration_col else 0.0
    )
    events["trial_type"] = log_df[trial_type_col].astype(str)

    if extra_cols:
        for col in extra_cols:
            if col in log_df.columns:
                events[col] = log_df[col]

    events = events.sort_values("onset").reset_index(drop=True)
    return events


# Demonstration with synthetic log data
np.random.seed(0)
n_trials = 40
synthetic_log = pd.DataFrame(
    {
        "onset_s": np.sort(np.random.uniform(2.0, 100.0, n_trials)),
        "condition": np.random.choice(["standard", "deviant"], n_trials, p=[0.8, 0.2]),
        "rt_s": np.where(
            np.random.rand(n_trials) < 0.85,
            np.random.normal(0.45, 0.12, n_trials),
            np.nan,
        ),
        "correct": np.random.choice([0, 1], n_trials, p=[0.15, 0.85]),
    }
)
synthetic_log.to_csv("/tmp/example_log.csv", index=False)

events_df = log_to_events_tsv(
    log_csv="/tmp/example_log.csv",
    onset_col="onset_s",
    trial_type_col="condition",
    extra_cols=["rt_s", "correct"],
)
print(events_df.head(8).to_string())

# Save to BIDS location
output_tsv = Path("my_bids_dataset/sub-01/ses-01/eeg") / \
    "sub-01_ses-01_task-oddball_run-1_events.tsv"
output_tsv.parent.mkdir(parents=True, exist_ok=True)
events_df.to_csv(output_tsv, sep="\t", index=False)
print(f"Saved: {output_tsv}")
```

---

## Troubleshooting

### Error: `BIDSValidationError: FILENAME_COLUMN_REQUIRED`

**Cause**: The `events.tsv` is missing the mandatory `onset`, `duration`, or `trial_type` column.

**Fix**:
```python
# Verify all three required columns exist
import pandas as pd
events = pd.read_csv("sub-01_task-oddball_events.tsv", sep="\t")
required = {"onset", "duration", "trial_type"}
missing = required - set(events.columns)
if missing:
    raise ValueError(f"Missing required events.tsv columns: {missing}")
```

### Error: `LayoutInitializationError` when building BIDSLayout

**Cause**: The `dataset_description.json` file is missing from the dataset root.

**Fix**:
```python
from pathlib import Path
import json

bids_root = Path("my_bids_dataset")
desc_path = bids_root / "dataset_description.json"
if not desc_path.exists():
    json.dump(
        {"Name": "My Dataset", "BIDSVersion": "1.8.0"},
        open(desc_path, "w"),
        indent=2,
    )
    print("Created minimal dataset_description.json")
```

### Issue: mne_bids.write_raw_bids raises ValueError about channel positions

**Cause**: The Raw object has no montage set, so electrode positions cannot be written.

**Fix**: Apply a standard montage before calling `write_raw_bids`:
```python
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="ignore")
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| mne | 1.5, 1.6, 1.7 | API changes in channel type handling between 1.5 and 1.6 |
| mne-bids | 0.13, 0.14 | `write_raw_bids` format parameter changed in 0.13 |
| pybids | 0.16, 0.17 | validate=True requires full BIDS layout; use False for partial datasets |
| datalad | 0.19, 0.20 | Requires git-annex system dependency; install separately |

---

## External Resources

### Official Documentation

- [BIDS Specification 1.8](https://bids-specification.readthedocs.io/en/stable/)
- [mne-bids documentation](https://mne.tools/mne-bids/stable/)
- [pybids documentation](https://bids-standard.github.io/pybids/)
- [datalad handbook](https://handbook.datalad.org/)
- [OpenNeuro](https://openneuro.org/)

### Key Papers

- Gorgolewski, K. J. et al. (2016). The brain imaging data structure, a format for organizing and
  describing outputs of neuroimaging experiments. *Scientific Data*, 3, 160044.
- Holdgraf, C. et al. (2019). iEEG-BIDS, extending the Brain Imaging Data Structure
  specification to human intracranial electrophysiology. *Scientific Data*, 6, 102.
- Appelhoff, S. et al. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS
  format and facilitating their analysis. *Journal of Open Source Software*, 4(44), 1896.

### Tutorials

- [BIDS Starter Kit](https://bids-standard.github.io/bids-starter-kit/)
- [mne-bids tutorial notebooks](https://mne.tools/mne-bids/stable/auto_examples/index.html)

### Data Sources

- [OpenNeuro](https://openneuro.org/): Public BIDS datasets (fMRI, EEG, MEG, iEEG)
- [EBRAINS](https://ebrains.eu/): European brain data platform with BIDS datasets

---

## Examples

### Example 1: Convert a Multi-Subject EEG Study to BIDS

**Scenario**: A lab has collected EDF files from 3 subjects across 2 sessions; convert to BIDS,
generate all sidecar metadata, and validate.

**Input data**: EDF files named `S01_ses1_oddball.edf`, `S01_ses2_oddball.edf`, etc.

```python
# =============================================
# End-to-end example: Multi-subject BIDS conversion
# Requirements: Python 3.10+; mne>=1.5.0, mne-bids>=0.13.0
# =============================================

import os
import mne
import mne_bids
import numpy as np
import json
from pathlib import Path

BIDS_ROOT = Path("example_bids_dataset")
BIDS_ROOT.mkdir(exist_ok=True)

# Write dataset_description.json
json.dump(
    {
        "Name": "Oddball Study",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "License": "CC0",
        "Authors": ["Example Researcher"],
    },
    open(BIDS_ROOT / "dataset_description.json", "w"),
    indent=2,
)

def make_synthetic_raw(seed: int) -> mne.io.BaseRaw:
    """Create a reproducible synthetic EEG Raw object."""
    rng = np.random.default_rng(seed)
    n_ch, sfreq, n_s = 19, 250.0, 30000
    ch_names = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz",
                 "C4","T8","P7","P3","Pz","P4","P8","O1","O2"]
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    info["line_freq"] = 50.0
    data = rng.normal(scale=5e-6, size=(n_ch, n_s))
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore")
    # Add synthetic events
    onsets = np.arange(1.0, 100.0, 2.0)
    descs = ["deviant" if i % 5 == 4 else "standard" for i in range(len(onsets))]
    raw.set_annotations(mne.Annotations(onsets, np.zeros(len(onsets)), descs))
    return raw

# Convert 2 subjects x 2 sessions
for sub_idx, sub in enumerate(["01", "02"], start=1):
    for ses_idx, ses in enumerate(["01", "02"], start=1):
        seed_val = sub_idx * 100 + ses_idx
        raw = make_synthetic_raw(seed=seed_val)

        bp = mne_bids.BIDSPath(
            subject=sub, session=ses, task="oddball", run="1",
            datatype="eeg", root=BIDS_ROOT,
        )
        mne_bids.write_raw_bids(
            raw, bids_path=bp, overwrite=True, verbose=False,
            format="EDF",
            event_id={"standard": 1, "deviant": 2},
        )
        print(f"  Written: sub-{sub} ses-{ses}")

# Query the resulting layout
from bids import BIDSLayout
layout = BIDSLayout(str(BIDS_ROOT), validate=False)
print(f"\nSubjects: {layout.get_subjects()}")
print(f"Tasks:    {layout.get_tasks()}")
print(f"EEG files: {len(layout.get(suffix='eeg', extension='.edf'))}")
# Expected:
# Subjects: ['01', '02']
# Tasks: ['oddball']
# EEG files: 4
```

**Interpreting these results**: The layout should index 4 EEG files (2 subjects × 2 sessions).
Run `bids-validator example_bids_dataset` to confirm zero errors.

---

### Example 2: Query an Existing OpenNeuro Dataset and Build an Analysis Table

**Scenario**: After downloading `ds003490` (a public BIDS EEG dataset), build a pandas table
linking every EEG file to its events.tsv and participant demographics.

```python
# =============================================
# End-to-end example 2: BIDSLayout query + analysis table
# Requirements: pybids>=0.16.0, pandas>=2.0.0
# =============================================

import pandas as pd
from bids import BIDSLayout
from pathlib import Path

# Replace with your local dataset path (after datalad install + get)
DATASET_PATH = os.environ.get("BIDS_DATASET_PATH", "example_bids_dataset")

layout = BIDSLayout(DATASET_PATH, validate=False)

# Load participants table
participants_tsv = Path(DATASET_PATH) / "participants.tsv"
if participants_tsv.exists():
    participants_df = pd.read_csv(participants_tsv, sep="\t")
else:
    participants_df = pd.DataFrame(columns=["participant_id"])

# Build file inventory
records = []
for eeg_file in layout.get(datatype="eeg", suffix="eeg"):
    sub = eeg_file.entities.get("subject", "")
    ses = eeg_file.entities.get("session", "")
    task = eeg_file.entities.get("task", "")
    run = eeg_file.entities.get("run", "")

    # Find matching events.tsv
    ev_files = layout.get(
        subject=sub, session=ses, task=task, run=run,
        suffix="events", extension=".tsv",
    )
    n_events = 0
    n_deviant = 0
    if ev_files:
        ev_df = pd.read_csv(ev_files[0].path, sep="\t")
        n_events = len(ev_df)
        if "trial_type" in ev_df.columns:
            n_deviant = (ev_df["trial_type"] == "deviant").sum()

    records.append(
        {
            "subject": f"sub-{sub}",
            "session": ses,
            "task": task,
            "run": run,
            "eeg_path": eeg_file.path,
            "n_events": n_events,
            "n_deviant": n_deviant,
        }
    )

inventory = pd.DataFrame(records)

# Merge with participants demographics
inventory = inventory.merge(
    participants_df.rename(columns={"participant_id": "subject"}),
    on="subject",
    how="left",
)

print("Analysis inventory:")
print(inventory.to_string())
print(f"\nTotal EEG files: {len(inventory)}")
print(f"Mean events per file: {inventory['n_events'].mean():.1f}")
# Expected output: a table merging file paths with demographics and event counts
```

**Interpreting these results**: Use the `inventory` DataFrame to loop over files for batch
preprocessing. The `n_deviant` column confirms event coding is consistent across subjects.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
