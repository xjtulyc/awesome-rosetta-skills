---
name: chembl-bioactivity
description: >
  Use this Skill to query ChEMBL for bioactivity data: target lookup, IC50/Ki
  retrieval, activity cliffs, SAR tables, and pChEMBL-normalized values.
tags:
  - chemistry
  - drug-discovery
  - chembl
  - bioactivity
  - sar
version: "1.0.0"
authors:
  - name: Rosetta Skills Contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - chembl-webresource-client>=0.10
    - pandas>=2.0
    - matplotlib>=3.7
    - numpy>=1.24
    - rdkit>=2023.3
    - seaborn>=0.12
last_updated: "2026-03-17"
status: "stable"
---

# ChEMBL Bioactivity Data Mining

> **One-line summary**: Query ChEMBL REST API to retrieve, filter, and analyze bioactivity data (IC50, Ki, pChEMBL) for structure-activity relationship (SAR) studies.

---

## When to Use This Skill

- When compiling IC50/Ki datasets for a specific protein target
- When benchmarking virtual screening models with experimental data
- When building QSAR models from ChEMBL bioactivity data
- When analyzing structure-activity relationships across a chemical series
- When identifying activity cliffs (similar structure, different potency)
- When standardizing assay data via pChEMBL normalized values

**Trigger keywords**: ChEMBL, bioactivity, IC50, Ki, pChEMBL, SAR, QSAR, drug target, assay data, compound activity

---

## Background & Key Concepts

### ChEMBL Database

ChEMBL is a manually curated database of bioactive molecules with drug-like properties maintained by EMBL-EBI. It contains:
- ~2.4M compounds
- ~20M bioactivity measurements
- ~15,000 targets
- Data extracted from primary literature

### pChEMBL Normalization

Raw activities (IC50, Ki, EC50) span many orders of magnitude and use different units. pChEMBL provides a normalized value:

$$
\text{pChEMBL} = -\log_{10}(\text{activity in molar})
$$

Higher pChEMBL = more potent. Typically pChEMBL ≥ 6 (≤ 1 μM) is considered "active" for drug discovery.

### Activity Cliffs

Activity cliffs are pairs of structurally similar compounds with large potency differences (typically ΔpChEMBL ≥ 2). They reveal key pharmacophoric features and are critical for SAR analysis.

---

## Environment Setup

### Install Dependencies

```bash
pip install chembl-webresource-client>=0.10 pandas>=2.0 matplotlib>=3.7 \
            numpy>=1.24 rdkit seaborn>=0.12
```

### Verify Installation

```python
from chembl_webresource_client.new_client import new_client
target_api = new_client.target
results = target_api.filter(target_synonym__icontains="EGFR").only(
    ["target_chembl_id", "pref_name"])[:3]
for r in results:
    print(r["target_chembl_id"], r["pref_name"])
# Expected: CHEMBL203 Epidermal growth factor receptor erbB1, ...
```

---

## Core Workflow

### Step 1: Target Lookup and Selection

```python
from chembl_webresource_client.new_client import new_client
import pandas as pd

def search_targets(keyword, organism="Homo sapiens", target_type="SINGLE PROTEIN"):
    """
    Search ChEMBL for protein targets matching a keyword.

    Parameters
    ----------
    keyword : str
        Gene name, protein name, or synonym
    organism : str
        Species filter (default: human)
    target_type : str
        'SINGLE PROTEIN', 'PROTEIN COMPLEX', etc.

    Returns
    -------
    pd.DataFrame
    """
    target_api = new_client.target
    results = target_api.filter(
        target_synonym__icontains=keyword,
        organism__icontains=organism,
        target_type=target_type
    ).only(["target_chembl_id", "pref_name", "organism", "target_type"])

    df = pd.DataFrame(list(results))
    return df

# Example: find EGFR-family kinases
targets_df = search_targets("EGFR")
print(f"Found {len(targets_df)} EGFR-related targets:")
print(targets_df[["target_chembl_id", "pref_name"]].to_string(index=False))

# Select a specific target
TARGET_ID = "CHEMBL203"  # EGFR
print(f"\nSelected target: {TARGET_ID}")
```

### Step 2: Retrieve Bioactivity Data

```python
from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np

def get_bioactivities(target_chembl_id, standard_types=("IC50", "Ki", "Kd"),
                      pchembl_min=5.0, max_records=5000):
    """
    Retrieve bioactivity data for a target.

    Returns
    -------
    pd.DataFrame with columns: molecule_chembl_id, smiles, standard_value,
                                standard_type, standard_units, pchembl_value, assay_chembl_id
    """
    activity_api = new_client.activity

    records = []
    for std_type in standard_types:
        results = activity_api.filter(
            target_chembl_id=target_chembl_id,
            standard_type=std_type,
            pchembl_value__gte=pchembl_min,  # filter: only active compounds
        ).only([
            "molecule_chembl_id", "canonical_smiles",
            "standard_value", "standard_type", "standard_units",
            "pchembl_value", "assay_chembl_id",
        ])[:max_records]
        records.extend(list(results))

    df = pd.DataFrame(records)
    if df.empty:
        print(f"No activities found for {target_chembl_id}")
        return df

    # Type conversion and cleaning
    df["pchembl_value"]  = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")

    # Drop missing SMILES and pChEMBL
    df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
    df = df.drop_duplicates(subset=["molecule_chembl_id", "standard_type"])

    print(f"Retrieved {len(df)} bioactivity records")
    print(df["standard_type"].value_counts().to_string())
    return df

activities = get_bioactivities("CHEMBL203", pchembl_min=6.0)
print(f"\npChEMBL distribution:")
print(activities["pchembl_value"].describe().round(3))
```

### Step 3: SAR Analysis and Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_activity_distribution(activities, title="EGFR IC50 Distribution"):
    """Plot pChEMBL distribution and scatter of IC50 values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of pChEMBL
    ic50_data = activities[activities["standard_type"] == "IC50"]["pchembl_value"].dropna()
    axes[0].hist(ic50_data, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(ic50_data.median(), color='red', linestyle='--',
                    label=f'Median pChEMBL={ic50_data.median():.2f}')
    axes[0].set_xlabel("pChEMBL (−log₁₀[IC50 M])")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title}\n(n={len(ic50_data)})")
    axes[0].legend()

    # Box plot by assay type
    if activities["standard_type"].nunique() > 1:
        activities.boxplot(column="pchembl_value", by="standard_type", ax=axes[1])
        axes[1].set_title("Activity by Assay Type")
        axes[1].set_xlabel("Standard Type")
        axes[1].set_ylabel("pChEMBL")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    plt.savefig("chembl_activity_distribution.png", dpi=150)
    plt.show()
    return ic50_data

ic50_vals = plot_activity_distribution(activities)
print(f"\nMost potent compound: {activities.nlargest(1, 'pchembl_value')['molecule_chembl_id'].values}")

# Compute RDKit molecular descriptors for a quick SAR table
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def compute_descriptors(smiles_series):
    results = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            results.append({
                "mw": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "tpsa": Descriptors.TPSA(mol),
                "n_rings": rdMolDescriptors.CalcNumRings(mol),
            })
        else:
            results.append({k: np.nan for k in ["mw","logp","hbd","hba","tpsa","n_rings"]})
    return pd.DataFrame(results)

sample = activities.head(200)
descriptors = compute_descriptors(sample["canonical_smiles"])
sar_df = pd.concat([
    sample[["molecule_chembl_id", "pchembl_value", "standard_type"]].reset_index(drop=True),
    descriptors
], axis=1)

print("\nSAR Table (first 5 rows):")
print(sar_df.head().to_string(index=False))

# Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = sar_df[["pchembl_value", "mw", "logp", "hbd", "hba", "tpsa"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Descriptor-Activity Correlation")
plt.tight_layout()
plt.savefig("chembl_sar_correlation.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Activity Cliff Detection

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from itertools import combinations
import pandas as pd

def compute_tanimoto_matrix(smiles_list, radius=2, n_bits=2048):
    """Compute pairwise Tanimoto similarity matrix from ECFP4 fingerprints."""
    from rdkit.DataStructs import TanimotoSimilarity
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))
        else:
            fps.append(None)

    n = len(fps)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if fps[i] and fps[j]:
                sim = TanimotoSimilarity(fps[i], fps[j])
                matrix[i,j] = matrix[j,i] = sim
    return matrix

def find_activity_cliffs(sar_df, sim_threshold=0.65, pchembl_diff=2.0):
    """
    Identify activity cliffs: similar structure (Tanimoto > threshold) +
    large potency difference (ΔpChEMBL > pchembl_diff).
    """
    df = sar_df.dropna(subset=["canonical_smiles", "pchembl_value"]).reset_index(drop=True)
    sim_matrix = compute_tanimoto_matrix(df["canonical_smiles"].tolist())

    cliffs = []
    for i, j in combinations(range(len(df)), 2):
        if sim_matrix[i,j] >= sim_threshold:
            delta = abs(df.loc[i, "pchembl_value"] - df.loc[j, "pchembl_value"])
            if delta >= pchembl_diff:
                cliffs.append({
                    "mol_a": df.loc[i, "molecule_chembl_id"],
                    "mol_b": df.loc[j, "molecule_chembl_id"],
                    "tanimoto": sim_matrix[i,j],
                    "pchembl_a": df.loc[i, "pchembl_value"],
                    "pchembl_b": df.loc[j, "pchembl_value"],
                    "delta_pchembl": delta,
                })
    return pd.DataFrame(cliffs).sort_values("delta_pchembl", ascending=False)

cliffs = find_activity_cliffs(sar_df.rename(columns={"canonical_smiles": "canonical_smiles"}
                             if "canonical_smiles" in sar_df.columns else {}))
print(f"\nActivity cliffs found: {len(cliffs)}")
if not cliffs.empty:
    print(cliffs.head(5).to_string(index=False))
```

---

## Troubleshooting

### Error: `ConnectionError` or timeout when querying ChEMBL

**Cause**: Network issues or ChEMBL API rate limiting.

**Fix**:
```python
import time
from chembl_webresource_client.new_client import new_client

# Reduce request size
activity_api = new_client.activity
# Use smaller page size
results = activity_api.filter(target_chembl_id="CHEMBL203")[:100]
time.sleep(1)  # polite delay
```

### Issue: Large query returns incomplete results

**Cause**: ChEMBL API paginates at 20 records by default.

**Fix**:
```python
results = list(activity_api.filter(target_chembl_id="CHEMBL203")[:5000])
# The [:5000] slice fetches all pages up to 5000 records
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| chembl-webresource-client | 0.10.8, 0.10.9 | Older versions use different API endpoints |
| rdkit | 2023.3, 2024.3 | None |

---

## External Resources

### Official Documentation

- [ChEMBL Web Services](https://chembl.gitbook.io/chembl-interface-documentation/web-services)
- [chembl_webresource_client GitHub](https://github.com/chembl/chembl_webresource_client)

### Key Papers

- Gaulton, A. et al. (2017). *The ChEMBL database in 2017*. Nucleic Acids Research.

---

## Examples

### Example 1: Build a QSAR Dataset for EGFR Inhibitors

```python
# =============================================
# End-to-end QSAR dataset from ChEMBL
# Requirements: chembl-webresource-client, rdkit, pandas
# =============================================
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pandas as pd, numpy as np

activity_api = new_client.activity
records = list(activity_api.filter(
    target_chembl_id="CHEMBL203",
    standard_type="IC50",
    pchembl_value__gte=5.0
).only(["molecule_chembl_id", "canonical_smiles", "pchembl_value"])[:1000])

df = pd.DataFrame(records).dropna()
df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
df = df.dropna()

def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    return list(fp)

fps = [featurize(s) for s in df["canonical_smiles"]]
valid = [fp is not None for fp in fps]
df = df[valid].reset_index(drop=True)
X = np.array([fp for fp, v in zip(fps, valid) if v])
y = df["pchembl_value"].values

print(f"QSAR dataset: {len(df)} compounds, {X.shape[1]} features")
print(f"pChEMBL range: {y.min():.2f} – {y.max():.2f}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
print(f"5-fold CV R² = {scores.mean():.3f} ± {scores.std():.3f}")
```

**Interpreting these results**: R² > 0.5 indicates a useful model. Use molecular fingerprints + gradient boosting or deep learning for improved predictive accuracy.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
