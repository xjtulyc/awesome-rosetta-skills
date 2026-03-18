---
name: openbabel-conversion
description: >
  Use this Skill for chemical file format interconversion (SDF, SMILES, MOL2,
  PDB), 3D coordinate generation, conformer enumeration, and reaction SMARTS.
tags:
  - chemistry
  - cheminformatics
  - openbabel
  - rdkit
  - file-format
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
    - openbabel>=3.1
    - rdkit>=2023.3
    - pandas>=2.0
    - numpy>=1.24
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# OpenBabel Format Conversion & 3D Coordinate Generation

> **One-line summary**: Convert chemical file formats, generate 3D structures, enumerate conformers, and apply reaction SMARTS using OpenBabel and RDKit.

---

## When to Use This Skill

- When converting between SDF, SMILES, MOL2, PDB, XYZ, and other chemical formats
- When generating 3D coordinates from 2D SMILES strings (for docking or MD)
- When enumerating low-energy conformers for virtual screening
- When applying reaction SMARTS to transform molecules
- When standardizing molecular representations across databases
- When preparing ligand libraries for structure-based drug discovery

**Trigger keywords**: format conversion, SDF to SMILES, mol2 PDB, 3D coordinate generation, conformer generation, reaction SMARTS, OpenBabel, obabel

---

## Background & Key Concepts

### Chemical File Formats

| Format | Use case | Contains 3D? | Common tool |
|:-------|:---------|:-------------|:------------|
| SMILES | Compact line notation | No | Any toolkit |
| SDF/MOL | Structure + properties | Yes | OpenBabel, RDKit |
| MOL2 | Docking preparation | Yes | AMBER, Dock |
| PDB | Protein+ligand | Yes | PyMOL, VMD |
| XYZ | QM input | Yes | Gaussian, ORCA |
| InChI | Canonical identifier | No | IUPAC standard |

### 3D Coordinate Generation

Generating 3D coordinates from a 2D SMILES requires:
1. Ring perception and stereochemistry assignment
2. Distance geometry or systematic torsion search
3. Force-field minimization (MMFF94, UFF)

OpenBabel uses the OBBuilder class for this; RDKit uses the EmbedMolecule/ETKDG algorithm.

### Conformer Generation

For flexible molecules, multiple conformers span the accessible conformational space. The ETKDG (Experimental Torsion angle Knowledge Distance Geometry) method in RDKit produces pharmacophorically relevant conformers.

---

## Environment Setup

### Install Dependencies

```bash
# OpenBabel (recommended via conda for easiest install)
conda install -c conda-forge openbabel

# RDKit
conda install -c conda-forge rdkit

# Or pip
pip install openbabel-wheel rdkit pandas numpy matplotlib
```

### Verify Installation

```python
import openbabel.openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCO")
print(f"Ethanol atoms: {mol.GetNumAtoms()}")
obconv = ob.OBConversion()
print(f"OpenBabel version: {ob.OBReleaseVersion()}")
# Expected: Ethanol atoms: 3
```

---

## Core Workflow

### Step 1: Format Interconversion with OpenBabel

```python
import openbabel.openbabel as ob

def convert_format(input_str, from_fmt, to_fmt, gen3d=False):
    """
    Convert a chemical string from one format to another.

    Parameters
    ----------
    input_str : str
        Input chemical string (SMILES, InChI, etc.)
    from_fmt : str
        Input format ('smi', 'inchi', 'sdf', 'mol2', 'pdb', 'xyz')
    to_fmt : str
        Output format
    gen3d : bool
        If True, generate 3D coordinates before conversion

    Returns
    -------
    str
        Converted chemical string
    """
    obconv = ob.OBConversion()
    obconv.SetInAndOutFormats(from_fmt, to_fmt)

    mol = ob.OBMol()
    obconv.ReadString(mol, input_str)

    if mol.NumAtoms() == 0:
        raise ValueError(f"Failed to parse input as {from_fmt}")

    if gen3d:
        builder = ob.OBBuilder()
        builder.Build(mol)
        ff = ob.OBForceField.FindForceField("MMFF94")
        if ff:
            ff.Setup(mol)
            ff.ConjugateGradients(500)
            ff.GetCoordinates(mol)

    mol.AddHydrogens()
    return obconv.WriteString(mol)

# Examples
smiles_list = [
    "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
    "c1ccc(cc1)C(=O)O",         # Benzoic acid
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
]

for smi in smiles_list:
    inchi = convert_format(smi, "smi", "inchi")
    pdb_3d = convert_format(smi, "smi", "pdb", gen3d=True)
    print(f"SMILES: {smi[:30]}...")
    print(f"  InChI: {inchi.strip()[:50]}...")
    print(f"  PDB (first line): {pdb_3d.splitlines()[0]}")
    print()
```

### Step 2: 3D Coordinate Generation with RDKit ETKDG

```python
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
import numpy as np

def generate_3d_conformers(smiles, n_confs=50, seed=42):
    """
    Generate multiple 3D conformers via ETKDG, return lowest-energy one.

    Returns
    -------
    rdkit.Chem.Mol with 3D coordinates, conformer energies array
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # ETKDG v3 (best for drug-like molecules)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 0   # use all cores
    params.pruneRmsThresh = 0.5  # remove similar conformers

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(conf_ids) == 0:
        raise RuntimeError("ETKDG embedding failed — check SMILES validity")

    # MMFF94 optimization
    energies = []
    for cid in conf_ids:
        res = MMFFOptimizeMolecule(mol, confId=cid, maxIters=2000)
        if res == 0:  # converged
            ff = AllChem.MMFFGetMoleculeForceField(mol,
                    AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
            energies.append(ff.CalcEnergy() if ff else float('inf'))
        else:
            energies.append(float('inf'))

    energies = np.array(energies)
    best_conf_id = int(conf_ids[np.argmin(energies)])

    # Keep only lowest-energy conformer for output
    best_mol = Chem.RWMol(mol)
    best_mol = best_mol.GetMol()

    print(f"Generated {len(conf_ids)} conformers")
    print(f"Energy range: {energies.min():.2f} – {energies.max():.2f} kcal/mol")
    return best_mol, best_conf_id, energies

# Aspirin
mol_3d, cid, ens = generate_3d_conformers("CC(=O)Oc1ccccc1C(=O)O", n_confs=20)
print(f"Best conformer energy: {ens.min():.2f} kcal/mol")

# Save to SDF
from rdkit.Chem import SDWriter
with SDWriter("aspirin_confs.sdf") as w:
    for conf_id in range(mol_3d.GetNumConformers()):
        w.write(mol_3d, confId=conf_id)
print("Saved aspirin_confs.sdf")
```

### Step 3: Reaction SMARTS Application

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToImage
import pandas as pd

def apply_reaction(reactant_smiles, rxn_smarts):
    """
    Apply a reaction SMARTS to a reactant SMILES.

    Returns
    -------
    list of product SMILES strings
    """
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    if rxn is None:
        raise ValueError(f"Invalid reaction SMARTS: {rxn_smarts}")

    reactant = Chem.MolFromSmiles(reactant_smiles)
    if reactant is None:
        raise ValueError(f"Invalid SMILES: {reactant_smiles}")

    products = rxn.RunReactants((reactant,))
    product_smiles = []
    for prod_tuple in products:
        for prod in prod_tuple:
            try:
                Chem.SanitizeMol(prod)
                smi = Chem.MolToSmiles(prod)
                product_smiles.append(smi)
            except Exception:
                pass
    return list(set(product_smiles))  # deduplicate

# Example reactions
reactions = {
    "ester_hydrolysis":   "[C:1](=O)[O:2][C:3]>>[C:1](=O)[O:2].[C:3]",
    "amide_formation":    "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]",
    "aromatic_nitration": "c:1:c:c:c:c:c1>>[c:1]c(cc:c:c:c1)[N+](=O)[O-]",
}

test_molecules = {
    "ester_hydrolysis":   "CC(=O)OCC",    # ethyl acetate
    "amide_formation":    "CC(=O)O",       # acetic acid + NH3 (separate reactant)
}

for rxn_name, smarts in reactions.items():
    if rxn_name in test_molecules:
        reactant = test_molecules[rxn_name]
        try:
            products = apply_reaction(reactant, smarts)
            print(f"\n{rxn_name}:")
            print(f"  Reactant:  {reactant}")
            print(f"  Products:  {products[:3]}")
        except Exception as e:
            print(f"{rxn_name}: {e}")

# Batch processing
records = []
smiles_batch = [
    "CC(=O)Oc1ccccc1C(=O)O",    # aspirin
    "CC12CCC3C(C1CCC2O)",        # steroid skeleton
    "c1ccc(cc1)C(=O)O",          # benzoic acid
]
for smi in smiles_batch:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        records.append({
            "smiles": smi,
            "n_atoms": mol.GetNumAtoms(),
            "n_rings": mol.GetRingInfo().NumRings(),
            "inchi": Chem.MolToInchi(mol),
        })
df = pd.DataFrame(records)
print("\nBatch processing results:")
print(df[["smiles", "n_atoms", "n_rings"]].to_string(index=False))
```

---

## Advanced Usage

### Scaffold Decomposition and Murcko Scaffold

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd

def get_murcko_scaffold(smiles, generic=False):
    """
    Extract Murcko scaffold (framework) from a molecule.
    generic=True returns the carbon skeleton (bemis-murcko generic scaffold).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold)

drugs = {
    "Sildenafil":   "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
    "Ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
}

scaffold_df = pd.DataFrame([
    {"drug": name, "smiles": smi,
     "scaffold": get_murcko_scaffold(smi),
     "generic_scaffold": get_murcko_scaffold(smi, generic=True)}
    for name, smi in drugs.items()
])
print(scaffold_df[["drug", "scaffold"]].to_string(index=False))
```

---

## Troubleshooting

### Error: `Import "openbabel.openbabel" could not be resolved`

**Cause**: OpenBabel Python bindings not installed correctly.

**Fix**:
```bash
# Prefer conda installation
conda install -c conda-forge openbabel

# Verify
python -c "import openbabel.openbabel as ob; print(ob.OBReleaseVersion())"
```

### Issue: ETKDG returns 0 conformers

**Cause**: SMILES has unsupported features (e.g., unusual valence) or stereospecification issues.

**Fix**:
```python
from rdkit import Chem
mol = Chem.MolFromSmiles(smiles)
# Check for sanitization warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # suppress
mol, problems = Chem.SanitizeMol(mol, catchErrors=True), Chem.DetectChemistryProblems(mol)
print(problems)
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| openbabel | 3.1.1, 3.1.4  | Python bindings require manual build on some platforms |
| rdkit     | 2023.3, 2024.3 | ETKDG v3 added in 2020 |

---

## External Resources

### Official Documentation

- [OpenBabel Python bindings](https://openbabel.org/docs/dev/UseTheLibrary/Python.html)
- [RDKit Conformer Generation](https://www.rdkit.org/docs/GettingStartedInPython.html)

### Key Papers

- Ebejer, J.P. et al. (2012). *Freely Available Conformer Generation Methods*. J. Chem. Inf. Model.
- Riniker, S. & Landrum, G.A. (2015). *Better Informed Distance Geometry: Using What We Know*. J. Chem. Inf. Model.

---

## Examples

### Example 1: Convert a CSV of SMILES to SDF with 3D Coordinates

```python
# =============================================
# Batch SMILES → SDF with MMFF geometry
# =============================================
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, SDWriter
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

smiles_data = pd.DataFrame({
    "name":   ["Aspirin", "Caffeine", "Dopamine"],
    "smiles": ["CC(=O)Oc1ccccc1C(=O)O",
               "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
               "NCCc1ccc(O)c(O)c1"]
})

with SDWriter("library_3d.sdf") as writer:
    for _, row in smiles_data.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            print(f"WARN: could not parse {row['name']}")
            continue
        mol.SetProp("_Name", row["name"])
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        MMFFOptimizeMolecule(mol)
        writer.write(mol)

print("Saved library_3d.sdf")
```

**Interpreting these results**: The SDF file can be opened in PyMOL, Schrödinger, or AutoDock Vina for molecular docking.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
