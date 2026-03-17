---
name: rdkit-cheminformatics
description: >
  Use this Skill for cheminformatics with RDKit: SMILES/InChI parsing, Morgan
  fingerprints, Tanimoto similarity, Murcko scaffold decomposition, substructure
  search, and chemical space visualization.
tags:
  - chemistry
  - cheminformatics
  - RDKit
  - molecular-fingerprints
  - drug-discovery
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
    - rdkit>=2023.3
    - pandas>=1.5
    - numpy>=1.23
    - matplotlib>=3.6
    - scikit-learn>=1.2
last_updated: "2026-03-17"
status: stable
---

# RDKit Cheminformatics

> **TL;DR** — Full cheminformatics pipeline with RDKit: parse SMILES/InChI/SDF,
> compute molecular properties (Lipinski), generate Morgan fingerprints, calculate
> Tanimoto similarity, decompose Murcko scaffolds, run SMARTS substructure searches,
> and visualize chemical space with PCA.

---

## When to Use

Use this Skill whenever you need to:

- Parse molecular structures from SMILES strings, InChI identifiers, or SDF files
- Calculate physicochemical properties: molecular weight, LogP, TPSA, HBD/HBA
- Assess Lipinski's Rule-of-Five compliance for drug-likeness filtering
- Generate Morgan (ECFP) fingerprints for similarity-based searches and clustering
- Compute pairwise or bulk Tanimoto similarity coefficients
- Decompose compound libraries into Murcko scaffolds for scaffold analysis
- Search compound collections with SMARTS substructure queries
- Visualize chemical space using PCA of fingerprint matrices
- Build structure-activity relationship (SAR) tables

| Task | Key RDKit Module |
|---|---|
| Mol parsing | `Chem.MolFromSmiles`, `Chem.MolFromInchi` |
| Properties | `Descriptors`, `rdMolDescriptors` |
| Fingerprints | `AllChem.GetMorganFingerprintAsBitVect` |
| Similarity | `DataStructs.TanimotoSimilarity` |
| Scaffolds | `MurckoDecomposition` |
| Substructure | `mol.HasSubstructMatch`, `mol.GetSubstructMatches` |
| Drawing | `Draw.MolToImage`, `Draw.MolsToGridImage` |

---

## Background & Key Concepts

### Molecular Representation

RDKit represents molecules as `Mol` objects containing atoms, bonds, ring systems,
and stereochemistry. The canonical entry points are:

- **SMILES** (Simplified Molecular Input Line Entry System): human-readable ASCII string,
  e.g. `CCO` for ethanol.
- **InChI** (IUPAC International Chemical Identifier): standardized, canonical; preferred
  for database exchange.
- **SDF/MOL** (Structure Data File): 2D/3D coordinates + properties; standard for compound
  libraries.

### Morgan Fingerprints (ECFP)

Morgan/circular fingerprints encode the chemical environment around each atom up to a
given radius. `radius=2` with `nBits=2048` corresponds to ECFP4, the industry standard
for virtual screening and ML models.

### Tanimoto Similarity

```
Tanimoto(A, B) = |A ∩ B| / |A ∪ B|
```

Values range from 0 (no common bits) to 1 (identical fingerprints). A threshold of
0.4 is commonly used for scaffold hopping; 0.85+ indicates very close analogs.

### Murcko Scaffolds

The Murcko scaffold (Bemis-Murcko decomposition) strips side chains to retain the ring
systems and their connecting linkers. It is the standard framework for scaffold frequency
analysis in medicinal chemistry.

---

## Environment Setup

```bash
# Create and activate a conda environment
conda create -n rdkit-env python=3.11 -y
conda activate rdkit-env

# Install RDKit and supporting packages
conda install -c conda-forge rdkit -y
pip install pandas numpy matplotlib scikit-learn

# Verify installation
python -c "from rdkit import Chem; print('RDKit version:', Chem.rdBase.rdkitVersion)"
```

Optional: Jupyter notebook support

```bash
pip install jupyter ipython
python -m ipykernel install --user --name rdkit-env
```

---

## Core Workflow

### Step 1 — Parse Molecules from Multiple Sources

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Draw, DataStructs
import pandas as pd
import numpy as np


def mol_from_smiles(smiles: str):
    """Parse a SMILES string; returns None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
    return mol


def mol_from_inchi(inchi: str):
    """Parse an InChI string to a RDKit Mol object."""
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        print(f"Invalid InChI: {inchi}")
    return mol


def mols_from_sdf(sdf_path: str) -> list:
    """
    Read all molecules from an SDF file.

    Args:
        sdf_path: Path to .sdf file.

    Returns:
        List of (mol, name) tuples; entries where mol is None are skipped.
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    molecules = []
    for mol in supplier:
        if mol is not None:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unnamed"
            molecules.append((mol, name))
    print(f"Loaded {len(molecules)} valid molecules from {sdf_path}")
    return molecules


# Quick demo
smiles_list = [
    "CC(=O)Oc1ccccc1C(=O)O",          # Aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",     # Pyrene
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",        # Caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",          # Ibuprofen
]

mols = [mol_from_smiles(s) for s in smiles_list]
mols = [m for m in mols if m is not None]
print(f"Parsed {len(mols)} molecules")
```

### Step 2 — Calculate Molecular Properties and Lipinski Filtering

```python
def calculate_properties(mol) -> dict:
    """
    Calculate key physicochemical descriptors for drug-likeness assessment.

    Returns:
        Dictionary with MW, LogP, TPSA, HBD, HBA, RotBonds, RingCount,
        and Lipinski Rule-of-Five compliance flag.
    """
    mw    = Descriptors.MolWt(mol)
    logp  = Descriptors.MolLogP(mol)
    tpsa  = rdMolDescriptors.CalcTPSA(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)   # H-bond donors
    hba   = rdMolDescriptors.CalcNumHBA(mol)   # H-bond acceptors
    rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol)
    arom  = rdMolDescriptors.CalcNumAromaticRings(mol)

    # Lipinski Rule-of-Five (oral bioavailability proxy)
    ro5 = (mw <= 500) and (logp <= 5) and (hbd <= 5) and (hba <= 10)

    # Veber rules (add-on for permeability)
    veber = (rot <= 10) and (tpsa <= 140)

    return {
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 2),
        "HBD": hbd,
        "HBA": hba,
        "RotBonds": rot,
        "Rings": rings,
        "AromaticRings": arom,
        "Lipinski_RO5": ro5,
        "Veber": veber,
    }


def batch_properties(smiles_df: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """
    Calculate properties for a DataFrame of SMILES strings.

    Args:
        smiles_df:  DataFrame containing SMILES strings.
        smiles_col: Column name with SMILES data.

    Returns:
        DataFrame with original columns plus all property columns.
    """
    records = []
    for _, row in smiles_df.iterrows():
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            records.append({k: None for k in [
                "MW", "LogP", "TPSA", "HBD", "HBA",
                "RotBonds", "Rings", "AromaticRings", "Lipinski_RO5", "Veber"
            ]})
        else:
            records.append(calculate_properties(mol))

    props_df = pd.DataFrame(records)
    return pd.concat([smiles_df.reset_index(drop=True), props_df], axis=1)


# Demonstration
names = ["Aspirin", "Testosterone", "Pyrene", "Caffeine", "Ibuprofen"]
demo_df = pd.DataFrame({"name": names, "smiles": smiles_list})
result_df = batch_properties(demo_df)
print(result_df[["name", "MW", "LogP", "TPSA", "HBD", "HBA", "Lipinski_RO5"]].to_string(index=False))
```

### Step 3 — Morgan Fingerprints and Tanimoto Similarity

```python
def get_morgan_fp(mol, radius: int = 2, n_bits: int = 2048):
    """
    Generate ECFP4-equivalent Morgan fingerprint as a bit vector.

    Args:
        mol:    RDKit Mol object.
        radius: Circular neighborhood radius (2 = ECFP4, 3 = ECFP6).
        n_bits: Fingerprint bit-vector length.

    Returns:
        RDKit ExplicitBitVect object.
    """
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def tanimoto_matrix(mols: list) -> np.ndarray:
    """
    Compute symmetric Tanimoto similarity matrix for a list of molecules.

    Args:
        mols: List of RDKit Mol objects.

    Returns:
        numpy array of shape (n, n) with pairwise Tanimoto values.
    """
    fps = [get_morgan_fp(m) for m in mols]
    n = len(fps)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    return matrix


def bulk_similarity_search(
    query_mol,
    library_mols: list,
    library_names: list,
    threshold: float = 0.4,
    radius: int = 2,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """
    Find all molecules in a library with Tanimoto >= threshold to the query.

    Args:
        query_mol:     Query RDKit Mol object.
        library_mols:  List of library RDKit Mol objects.
        library_names: Names corresponding to library_mols.
        threshold:     Minimum Tanimoto similarity to include.
        radius:        Morgan radius.
        n_bits:        Fingerprint bits.

    Returns:
        DataFrame sorted by similarity descending, columns: name, smiles, tanimoto.
    """
    query_fp = get_morgan_fp(query_mol, radius, n_bits)
    lib_fps  = [get_morgan_fp(m, radius, n_bits) for m in library_mols]

    sims = DataStructs.BulkTanimotoSimilarity(query_fp, lib_fps)

    hits = []
    for name, mol, sim in zip(library_names, library_mols, sims):
        if sim >= threshold:
            hits.append({
                "name": name,
                "smiles": Chem.MolToSmiles(mol),
                "tanimoto": round(sim, 4),
            })

    return pd.DataFrame(hits).sort_values("tanimoto", ascending=False).reset_index(drop=True)


# Similarity matrix example
sim_mat = tanimoto_matrix(mols)
print("Tanimoto similarity matrix:")
print(pd.DataFrame(sim_mat, index=names, columns=names).round(3).to_string())
```

---

## Advanced Usage

### Murcko Scaffold Decomposition

```python
def get_murcko_scaffold(mol, generic: bool = False) -> str:
    """
    Extract the Murcko scaffold SMILES from a molecule.

    Args:
        mol:     RDKit Mol object.
        generic: If True, return the generic scaffold (all atoms -> C, all bonds -> single).

    Returns:
        SMILES string of the scaffold, or '' if extraction fails.
    """
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        return Chem.MolToSmiles(scaffold_mol)
    except Exception as e:
        print(f"Scaffold error: {e}")
        return ""


def scaffold_frequency_analysis(
    smiles_list: list,
    names: list = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Count scaffold frequencies across a compound library.

    Args:
        smiles_list: List of SMILES strings.
        names:       Optional compound names (same length as smiles_list).
        top_n:       Number of top scaffolds to return.

    Returns:
        DataFrame: scaffold_smiles, count, fraction, example_compound.
    """
    from collections import defaultdict

    scaffold_to_compounds = defaultdict(list)
    names = names or [f"cpd_{i}" for i in range(len(smiles_list))]

    for smi, name in zip(smiles_list, names):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = get_murcko_scaffold(mol)
        scaffold_to_compounds[scaffold].append(name)

    records = []
    total = len(smiles_list)
    for scaffold_smi, compounds in scaffold_to_compounds.items():
        records.append({
            "scaffold_smiles": scaffold_smi,
            "count": len(compounds),
            "fraction": round(len(compounds) / total, 4),
            "example_compound": compounds[0],
        })

    df = pd.DataFrame(records).sort_values("count", ascending=False).head(top_n)
    return df.reset_index(drop=True)


# Example: scaffold analysis on a small library
scaffolds = scaffold_frequency_analysis(smiles_list, names)
print(scaffolds[["scaffold_smiles", "count", "fraction"]].to_string(index=False))
```

### SMARTS Substructure Search

```python
def substructure_search(
    library_mols: list,
    library_names: list,
    smarts_pattern: str,
) -> pd.DataFrame:
    """
    Filter a compound library by a SMARTS substructure pattern.

    Args:
        library_mols:    List of RDKit Mol objects.
        library_names:   Names for each molecule.
        smarts_pattern:  SMARTS string to match against.

    Returns:
        DataFrame of matching compounds with match atom indices.
    """
    query = Chem.MolFromSmarts(smarts_pattern)
    if query is None:
        raise ValueError(f"Invalid SMARTS pattern: {smarts_pattern}")

    hits = []
    for name, mol in zip(library_names, library_mols):
        if mol.HasSubstructMatch(query):
            matches = mol.GetSubstructMatches(query)
            hits.append({
                "name": name,
                "smiles": Chem.MolToSmiles(mol),
                "n_matches": len(matches),
                "match_atoms": str(matches[0]),
            })

    return pd.DataFrame(hits)


# Search for carboxylic acids
aromatic_ring_smarts = "c1ccccc1"          # benzene ring
carboxylic_acid_smarts = "C(=O)[OH]"       # carboxylic acid
sulfonamide_smarts = "S(=O)(=O)N"          # sulfonamide

print("Benzene ring hits:")
print(substructure_search(mols, names, aromatic_ring_smarts))
print("\nCarboxylic acid hits:")
print(substructure_search(mols, names, carboxylic_acid_smarts))
```

### Chemical Space Visualization with PCA

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def chemical_space_pca(
    mols: list,
    labels: list = None,
    color_by: list = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Project molecules into 2D chemical space using PCA of Morgan fingerprints.

    Args:
        mols:        List of RDKit Mol objects.
        labels:      Compound names for hover/annotation.
        color_by:    Numeric values for color-coding points (e.g., pIC50).
        output_path: Path to save the PNG plot.

    Returns:
        DataFrame with columns: name, PC1, PC2, (color_value).
    """
    # Build fingerprint matrix
    fps = [get_morgan_fp(m) for m in mols]
    fp_array = np.array([list(fp.ToBitString()) for fp in fps], dtype=float)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fp_array)

    labels = labels or [f"cpd_{i}" for i in range(len(mols))]
    df_pca = pd.DataFrame({
        "name": labels,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })

    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_kwargs = dict(s=60, alpha=0.8, edgecolors="k", linewidths=0.4)

    if color_by is not None:
        sc = ax.scatter(df_pca["PC1"], df_pca["PC2"], c=color_by,
                        cmap="RdYlGn", **scatter_kwargs)
        plt.colorbar(sc, ax=ax, label="Property value")
        df_pca["color_value"] = color_by
    else:
        ax.scatter(df_pca["PC1"], df_pca["PC2"], color="#4C72B0", **scatter_kwargs)

    for _, row in df_pca.iterrows():
        ax.annotate(row["name"], (row["PC1"], row["PC2"]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
    ax.set_title("Chemical Space — PCA of Morgan Fingerprints (ECFP4)")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved PCA plot to {output_path}")

    plt.show()
    return df_pca


# Visualize chemical space with LogP as color
logp_values = [Descriptors.MolLogP(m) for m in mols]
pca_df = chemical_space_pca(mols, labels=names, color_by=logp_values,
                             output_path="chemical_space.png")
print(pca_df)
```

---

## Examples

### Example 1 — Batch Property Calculation from SMILES CSV

```python
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem


def process_smiles_csv(
    input_csv: str,
    output_csv: str,
    smiles_col: str = "smiles",
    name_col: str = "name",
) -> pd.DataFrame:
    """
    Read a CSV of SMILES, compute properties, filter by Lipinski RO5, and save results.

    Expected CSV columns: name, smiles (+ any additional columns kept as-is).
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} compounds from {input_csv}")

    results = []
    failed = 0

    for _, row in df.iterrows():
        smi = str(row[smiles_col])
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            failed += 1
            continue

        props = {
            "name": row.get(name_col, "unknown"),
            "smiles": Chem.MolToSmiles(mol),    # canonical SMILES
            "MW": round(Descriptors.MolWt(mol), 2),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "Lipinski_pass": int(
                Descriptors.MolWt(mol) <= 500
                and Descriptors.MolLogP(mol) <= 5
                and rdMolDescriptors.CalcNumHBD(mol) <= 5
                and rdMolDescriptors.CalcNumHBA(mol) <= 10
            ),
            "scaffold": get_murcko_scaffold(mol),
        }
        results.append(props)

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"Processed {len(result_df)} valid molecules ({failed} failed)")
    print(f"Lipinski pass rate: {result_df['Lipinski_pass'].mean():.1%}")
    print(f"Saved to {output_csv}")
    return result_df


# --- Inline demo without a real CSV ---
sample_data = pd.DataFrame({
    "name": ["Aspirin", "Caffeine", "Ibuprofen", "Metformin"],
    "smiles": [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CN(C)C(=N)NC(=N)N",
    ],
})
sample_data.to_csv("/tmp/sample_compounds.csv", index=False)
result = process_smiles_csv(
    "/tmp/sample_compounds.csv",
    "/tmp/compound_properties.csv",
)
print(result[["name", "MW", "LogP", "TPSA", "Lipinski_pass"]].to_string(index=False))
```

### Example 2 — Similarity Search and Hierarchical Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def similarity_cluster(
    smiles_list: list,
    names: list,
    query_smiles: str = None,
    sim_threshold: float = 0.3,
    output_path: str = "dendrogram.png",
) -> dict:
    """
    Compute pairwise Tanimoto similarity, optionally filter by query, and
    produce a hierarchical clustering dendrogram.

    Args:
        smiles_list:    List of SMILES strings to cluster.
        names:          Compound names.
        query_smiles:   If given, first filter by similarity >= sim_threshold.
        sim_threshold:  Minimum similarity when using query filter.
        output_path:    Path to save the dendrogram image.

    Returns:
        Dictionary with 'similarity_matrix', 'cluster_linkage', 'filtered_names'.
    """
    # Parse
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [(m, n) for m, n in zip(mols, names) if m is not None]
    mols, names = zip(*valid)

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]

    # Optional: query filter
    if query_smiles:
        q_mol = Chem.MolFromSmiles(query_smiles)
        q_fp  = AllChem.GetMorganFingerprintAsBitVect(q_mol, 2, 2048)
        sims  = DataStructs.BulkTanimotoSimilarity(q_fp, fps)
        mask  = [s >= sim_threshold for s in sims]
        fps   = [fp for fp, m in zip(fps, mask) if m]
        names = [n  for n,  m in zip(names, mask) if m]
        print(f"After query filter (>= {sim_threshold}): {len(fps)} compounds")

    # Tanimoto distance matrix (1 - similarity)
    n = len(fps)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dist_matrix[i, j] = 1.0 - sim
            dist_matrix[j, i] = 1.0 - sim

    # Condensed distance for scipy
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    # Dendrogram
    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 5))
    dendrogram(Z, labels=list(names), ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_title("Hierarchical Clustering (Tanimoto Distance, Ward linkage)")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Dendrogram saved to {output_path}")
    plt.show()

    return {
        "similarity_matrix": 1.0 - dist_matrix,
        "cluster_linkage": Z,
        "filtered_names": list(names),
    }


# Example call
extended_smiles = smiles_list + [
    "O=C(O)c1ccccc1O",                  # Salicylic acid
    "CC(=O)Nc1ccc(O)cc1",               # Paracetamol
    "O=C(Oc1ccccc1)c1ccccc1",           # Phenyl benzoate
]
extended_names = names + ["Salicylic_acid", "Paracetamol", "Phenyl_benzoate"]

cluster_result = similarity_cluster(
    extended_smiles, extended_names,
    output_path="/tmp/dendrogram.png",
)
```

### Example 3 — Scaffold Frequency Analysis of a Drug Library

```python
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import Counter


def full_scaffold_analysis(
    smiles_list: list,
    names: list,
    top_n: int = 5,
    output_path: str = "scaffold_bar.png",
) -> pd.DataFrame:
    """
    Decompose a compound library into Murcko scaffolds, count frequencies,
    and plot a horizontal bar chart of the top scaffolds.

    Args:
        smiles_list:  List of SMILES strings.
        names:        Compound names (same length).
        top_n:        How many top scaffolds to visualize.
        output_path:  Path to save the bar chart.

    Returns:
        DataFrame with columns: scaffold, count, fraction, members.
    """
    scaffold_map = {}
    for smi, name in zip(smiles_list, names):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaf_mol)
        except Exception:
            scaf_smi = "__no_scaffold__"

        if scaf_smi not in scaffold_map:
            scaffold_map[scaf_smi] = []
        scaffold_map[scaf_smi].append(name)

    total = len(smiles_list)
    records = sorted(
        [{"scaffold": k, "count": len(v), "fraction": len(v)/total, "members": v}
         for k, v in scaffold_map.items()],
        key=lambda x: x["count"], reverse=True,
    )
    df = pd.DataFrame(records)

    # Plot
    top = df.head(top_n)
    short_labels = [s[:30] + "..." if len(s) > 30 else s for s in top["scaffold"]]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(range(len(top)), top["count"], color="#2196F3", edgecolor="k", lw=0.5)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(short_labels, fontsize=9)
    ax.set_xlabel("Compound count")
    ax.set_title(f"Top {top_n} Murcko Scaffolds (n={total} compounds)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Scaffold chart saved to {output_path}")
    plt.show()

    return df


scaffold_df = full_scaffold_analysis(
    extended_smiles, extended_names,
    top_n=5,
    output_path="/tmp/scaffold_bar.png",
)
print(scaffold_df[["scaffold", "count", "fraction"]].to_string(index=False))
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `MolFromSmiles` returns `None` | Invalid or unsanitizable SMILES | Check SMILES string; use `Chem.SanitizeMol` with error handling |
| `Kekulization failed` | Aromatic notation inconsistency | Call `Chem.Kekulize(mol, clearAromaticFlags=True)` |
| `ValueError: Bad SMARTS` | Malformed SMARTS pattern | Validate with `Chem.MolFromSmarts(pat)` and check for `None` |
| `AllChem.EmbedMolecule` returns -1 | 3D embedding failed | Add `AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)` |
| ImportError on `Draw` | Missing `Pillow` | `pip install Pillow` |
| Slow `BulkTanimotoSimilarity` for large libraries | Large library > 100k | Use `rdkit.DataStructs.BulkTanimotoSimilarity` vectorized; or use LSH |
| Scaffold is empty string for acyclic molecules | No ring systems | Expected behavior; treat '' as "no scaffold" |
| PCA explained variance too low | Fingerprint redundancy or tiny library | Increase `nBits`; use UMAP for high-dimensional spaces |

---

## External Resources

- RDKit documentation: <https://www.rdkit.org/docs/>
- RDKit GitHub: <https://github.com/rdkit/rdkit>
- RDKit cookbook: <https://www.rdkit.org/docs/Cookbook.html>
- Fingerprint comparison blog: <https://greglandrum.github.io/rdkit-blog/>
- Murcko scaffold paper: Bemis & Murcko (1996) *J. Med. Chem.* 39(15), 2887–2893
- ECFP paper: Rogers & Hahn (2010) *J. Chem. Inf. Model.* 50(5), 742–754
- Tanimoto similarity overview: <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3>
- ChEMBL compound database: <https://www.ebi.ac.uk/chembl/>
- PubChem: <https://pubchem.ncbi.nlm.nih.gov/>

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — SMILES parsing, properties, Morgan FP, Tanimoto, Murcko, PCA |
