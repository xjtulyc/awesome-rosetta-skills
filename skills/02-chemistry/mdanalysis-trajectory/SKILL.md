---
name: mdanalysis-trajectory
description: Analyze molecular dynamics trajectories with MDAnalysis — RMSD, RMSF, hydrogen bonds, contact maps, and protein-ligand distances.
tags:
  - molecular-dynamics
  - trajectory-analysis
  - mdanalysis
  - structural-biology
  - gromacs
version: "1.0.0"
authors:
  - name: "Rosetta Skills Contributors"
    github: "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - MDAnalysis>=2.6
  - matplotlib>=3.7
  - numpy>=1.24
  - pandas>=2.0
  - scipy>=1.11
last_updated: "2026-03-17"
status: stable
---

# MDAnalysis Trajectory Analysis

Parse and analyze molecular dynamics (MD) trajectories from GROMACS, AMBER, LAMMPS,
and NAMD simulations using MDAnalysis. This skill covers RMSD/RMSF computation,
hydrogen bond tracking, contact map generation, and protein-ligand distance analysis
with publication-ready matplotlib visualizations.

---

## When to Use This Skill

- You have an MD trajectory file (`.xtc`, `.trr`, `.dcd`, `.nc`, `.lammpstrj`) and
  want to extract quantitative structural metrics.
- You need to compute **RMSD** (global fold stability) or **RMSF** (per-residue
  flexibility) over a simulation.
- You want to enumerate **hydrogen bonds** between protein and ligand or between
  secondary-structure elements.
- You need a **contact map** or residue-residue distance matrix to identify
  persistent interactions.
- You are measuring **protein-ligand binding pocket distances** to assess whether a
  ligand stays bound throughout the simulation.
- You want to export per-frame structural data to a **pandas DataFrame** for
  downstream statistical analysis.

---

## Background & Key Concepts

### Universe and AtomGroup

MDAnalysis represents a simulation as a `Universe` object that couples a **topology**
file (connectivity, residue names, charges) with one or more **trajectory** files
(coordinate frames). Selections of atoms are returned as `AtomGroup` objects, which
support arithmetic, boolean masks, and iteration.

```python
import MDAnalysis as mda
u = mda.Universe("topology.tpr", "trajectory.xtc")
protein = u.select_atoms("protein")
```

### Trajectory Iteration

Iterating over `u.trajectory` moves the `Universe` to successive frames. All
`AtomGroup.positions` arrays are updated automatically — no manual frame loading is
required.

```python
for ts in u.trajectory:
    # ts.time is in ps; ts.frame is the 0-based frame index
    coords = protein.positions  # (N, 3) float32 array, angstroms
```

### RMSD vs RMSF

| Metric | Domain   | Measures                              |
|--------|----------|---------------------------------------|
| RMSD   | per-frame | global deviation from a reference    |
| RMSF   | per-atom  | time-averaged fluctuation amplitude  |

Both metrics require alignment (superposition) to remove overall rotation/translation
before computing distances.

### Hydrogen Bond Criteria

The default HBond criterion in MDAnalysis uses donor-acceptor distance ≤ 3.5 Å and
donor-H-acceptor angle ≥ 150°. These thresholds can be tuned for non-standard force
fields.

### Contact Maps

A **contact** is defined when Cα–Cα distance (or heavy-atom distance) drops below a
cutoff (typically 8 Å for Cα, 4.5 Å for heavy atoms). A contact map averaged over
trajectory frames reveals stable structural contacts.

---

## Environment Setup

### Installation

```bash
# Create a dedicated conda environment (recommended)
conda create -n mdanalysis-env python=3.11 -y
conda activate mdanalysis-env

# Install MDAnalysis and visualization stack
pip install "MDAnalysis>=2.6" "matplotlib>=3.7" "numpy>=1.24" "pandas>=2.0" "scipy>=1.11"

# Optional: install MDAnalysisTests for sample data
pip install MDAnalysisTests
```

### Verify Installation

```python
import MDAnalysis as mda
print(mda.__version__)   # e.g. 2.6.1

import MDAnalysis.tests
from MDAnalysisTests.datafiles import PSF, DCD
u = mda.Universe(PSF, DCD)
print(f"Atoms: {u.atoms.n_atoms}, Frames: {u.trajectory.n_frames}")
```

### Supported Formats

| Format | Topology | Trajectory |
|--------|----------|------------|
| GROMACS | `.tpr`, `.gro` | `.xtc`, `.trr` |
| AMBER   | `.prmtop`, `.parm7` | `.nc`, `.ncdf` |
| NAMD/CHARMM | `.psf` | `.dcd` |
| LAMMPS  | `.data` | `.lammpstrj` |
| PDB/mmCIF | `.pdb`, `.cif` | — |

---

## Core Workflow

### Step 1 — Load Universe and Inspect Topology

```python
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual files
TOPOLOGY  = "md_system.tpr"   # or .prmtop / .psf / .pdb
TRAJECTORY = "md_traj.xtc"    # or .dcd / .nc / .lammpstrj

u = mda.Universe(TOPOLOGY, TRAJECTORY)

print(f"Number of atoms   : {u.atoms.n_atoms}")
print(f"Number of residues: {u.residues.n_residues}")
print(f"Number of segments: {u.segments.n_segments}")
print(f"Number of frames  : {u.trajectory.n_frames}")
print(f"Time step (ps)    : {u.trajectory.dt:.3f}")
print(f"Total time (ns)   : {u.trajectory.totaltime / 1000:.2f}")

# Inspect segment / chain names
for seg in u.segments:
    print(f"  Segment {seg.segid}: {seg.atoms.n_atoms} atoms")

# Select key atom groups
protein  = u.select_atoms("protein")
backbone = u.select_atoms("backbone")
ligand   = u.select_atoms("resname LIG")   # adjust resname
water    = u.select_atoms("resname WAT SOL TIP3")

print(f"\nProtein atoms : {protein.n_atoms}")
print(f"Backbone atoms: {backbone.n_atoms}")
print(f"Ligand atoms  : {ligand.n_atoms}")
```

### Step 2 — RMSD Calculation and Visualization

```python
from MDAnalysis.analysis import rms, align

# --- Align trajectory to first frame (in-place) ---
aligner = align.AlignTraj(u, u, select="backbone", in_memory=False)
aligner.run()

# --- Compute RMSD for backbone and C-alpha ---
rmsd_analysis = rms.RMSD(
    u,
    select="backbone",
    groupselections=["backbone", "name CA"],
    ref_frame=0,
)
rmsd_analysis.run(verbose=True)

# Results array shape: (n_frames, 3+n_groups)
# Columns: frame, time(ps), backbone_rmsd, [group_rmsds...]
df_rmsd = pd.DataFrame(
    rmsd_analysis.results.rmsd[:, 1:],   # drop frame index column
    columns=["Time (ps)", "Backbone RMSD (Å)", "C-alpha RMSD (Å)"],
)
df_rmsd["Time (ns)"] = df_rmsd["Time (ps)"] / 1000.0

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_rmsd["Time (ns)"], df_rmsd["Backbone RMSD (Å)"], label="Backbone", lw=1.5)
ax.plot(df_rmsd["Time (ns)"], df_rmsd["C-alpha RMSD (Å)"], label="Cα", lw=1.5, ls="--")
ax.set_xlabel("Time (ns)", fontsize=12)
ax.set_ylabel("RMSD (Å)", fontsize=12)
ax.set_title("Backbone and Cα RMSD over Simulation", fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rmsd_plot.png", dpi=150)
plt.show()

# Save data
df_rmsd.to_csv("rmsd_data.csv", index=False)
print("RMSD data saved to rmsd_data.csv")
```

### Step 3 — RMSF Calculation (Per-Residue Flexibility)

```python
from MDAnalysis.analysis import rms

# Compute RMSF on C-alpha atoms
ca_atoms = u.select_atoms("name CA")

rmsf_analysis = rms.RMSF(ca_atoms)
rmsf_analysis.run(verbose=True)

rmsf_values = rmsf_analysis.results.rmsf   # shape (n_CA,)
residue_ids = ca_atoms.resids
residue_names = ca_atoms.resnames

df_rmsf = pd.DataFrame({
    "ResID"  : residue_ids,
    "ResName": residue_names,
    "RMSF_A" : rmsf_values,
})

# Identify highly flexible residues (top 10%)
threshold = np.percentile(rmsf_values, 90)
flexible = df_rmsf[df_rmsf["RMSF_A"] >= threshold]
print(f"Highly flexible residues (RMSF >= {threshold:.2f} Å):")
print(flexible.to_string(index=False))

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(df_rmsf["ResID"], df_rmsf["RMSF_A"], width=1.0, color="steelblue", alpha=0.8)
ax.axhline(threshold, color="red", ls="--", label=f"90th percentile ({threshold:.2f} Å)")
ax.set_xlabel("Residue ID", fontsize=12)
ax.set_ylabel("RMSF (Å)", fontsize=12)
ax.set_title("Per-Residue RMSF (Cα)", fontsize=13)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("rmsf_plot.png", dpi=150)
plt.show()

df_rmsf.to_csv("rmsf_data.csv", index=False)
```

### Step 4 — Hydrogen Bond Analysis

```python
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

# Protein-ligand hydrogen bonds
hbond_analysis = HydrogenBondAnalysis(
    universe=u,
    donors_sel="(protein) and (name N* O*)",
    hydrogens_sel="(protein) and (name H*)",
    acceptors_sel="(resname LIG) and (name N* O* S*)",
    d_a_cutoff=3.5,          # donor-acceptor distance cutoff (Å)
    d_h_a_angle_cutoff=150,  # minimum donor-H-acceptor angle (degrees)
    update_selections=False,
)
hbond_analysis.run(verbose=True)

# Convert results to DataFrame
hb_df = hbond_analysis.results.hbonds
columns = ["Frame", "Donor_idx", "H_idx", "Acceptor_idx", "Distance", "Angle"]
df_hb = pd.DataFrame(hb_df, columns=columns)

# Compute occupancy (fraction of frames with each H-bond)
n_frames = u.trajectory.n_frames
hb_occupancy = (
    df_hb.groupby(["Donor_idx", "Acceptor_idx"])
    .size()
    .reset_index(name="Count")
)
hb_occupancy["Occupancy"] = hb_occupancy["Count"] / n_frames

# Resolve atom names
def atom_label(idx):
    atom = u.atoms[int(idx)]
    return f"{atom.resname}{atom.resid}:{atom.name}"

hb_occupancy["Donor_label"]    = hb_occupancy["Donor_idx"].apply(atom_label)
hb_occupancy["Acceptor_label"] = hb_occupancy["Acceptor_idx"].apply(atom_label)
hb_occupancy_sorted = hb_occupancy.sort_values("Occupancy", ascending=False)

print("\nTop hydrogen bonds by occupancy:")
print(hb_occupancy_sorted[["Donor_label", "Acceptor_label", "Occupancy"]].head(10).to_string(index=False))

# --- Plot ---
top10 = hb_occupancy_sorted.head(10)
labels = top10["Donor_label"] + " → " + top10["Acceptor_label"]
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(labels, top10["Occupancy"], color="coral")
ax.set_xlabel("Occupancy (fraction of frames)", fontsize=12)
ax.set_title("Top 10 Protein-Ligand Hydrogen Bonds", fontsize=13)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("hbond_occupancy.png", dpi=150)
plt.show()

df_hb.to_csv("hbond_raw.csv", index=False)
hb_occupancy_sorted.to_csv("hbond_occupancy.csv", index=False)
```

### Step 5 — Protein-Ligand Distance Tracking

```python
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import calc_bonds

u = mda.Universe(TOPOLOGY, TRAJECTORY)
ligand  = u.select_atoms("resname LIG")
binding_residues = u.select_atoms("protein and resid 45 82 118 201")  # adjust resids

times, distances = [], []

for ts in u.trajectory:
    # Centroid of ligand heavy atoms
    lig_centroid = ligand.center_of_mass()
    # Centroid of binding pocket residues (Cα only)
    pocket_ca = binding_residues.select_atoms("name CA")
    pocket_centroid = pocket_ca.center_of_mass()
    dist = np.linalg.norm(lig_centroid - pocket_centroid)
    times.append(ts.time / 1000.0)   # ns
    distances.append(dist)

df_dist = pd.DataFrame({"Time (ns)": times, "Ligand-Pocket Distance (Å)": distances})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_dist["Time (ns)"], df_dist["Ligand-Pocket Distance (Å)"], lw=1, color="darkgreen")
ax.axhline(8.0, color="red", ls="--", label="8 Å binding threshold")
ax.fill_between(df_dist["Time (ns)"], df_dist["Ligand-Pocket Distance (Å)"], 8.0,
                where=df_dist["Ligand-Pocket Distance (Å)"] > 8.0,
                alpha=0.2, color="red", label="Unbound region")
ax.set_xlabel("Time (ns)", fontsize=12)
ax.set_ylabel("Distance (Å)", fontsize=12)
ax.set_title("Protein-Ligand Binding Pocket Distance", fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ligand_distance.png", dpi=150)
plt.show()

df_dist.to_csv("ligand_distance.csv", index=False)
```

---

## Advanced Usage

### Contact Map from Average Trajectory

```python
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array

u = mda.Universe(TOPOLOGY, TRAJECTORY)
ca = u.select_atoms("name CA")
n_res = ca.n_atoms
contact_cutoff = 8.0   # Å, Cα-Cα

# Accumulate contact frequency
contact_freq = np.zeros((n_res, n_res), dtype=np.float32)

for ts in u.trajectory:
    dist_matrix = distance_array(ca.positions, ca.positions, box=ts.dimensions)
    contact_freq += (dist_matrix < contact_cutoff).astype(np.float32)

contact_freq /= u.trajectory.n_frames  # normalize to occupancy [0, 1]

# Remove sequence-local contacts (|i-j| < 4) for clarity
for i in range(n_res):
    for j in range(max(0, i-3), min(n_res, i+4)):
        contact_freq[i, j] = 0.0

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(contact_freq, cmap="hot_r", origin="lower", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="Contact Frequency")
ax.set_xlabel("Residue Index", fontsize=12)
ax.set_ylabel("Residue Index", fontsize=12)
ax.set_title(f"Cα Contact Map (cutoff={contact_cutoff} Å)", fontsize=13)
plt.tight_layout()
plt.savefig("contact_map.png", dpi=150)
plt.show()

np.save("contact_map.npy", contact_freq)
print("Contact map saved.")
```

### Radius of Gyration

```python
import MDAnalysis as mda
import pandas as pd
import matplotlib.pyplot as plt

u = mda.Universe(TOPOLOGY, TRAJECTORY)
protein = u.select_atoms("protein")

rog_data = []
for ts in u.trajectory:
    rog_data.append({
        "Time (ns)": ts.time / 1000.0,
        "Rg (Å)": protein.radius_of_gyration(),
    })

df_rog = pd.DataFrame(rog_data)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_rog["Time (ns)"], df_rog["Rg (Å)"], lw=1.2, color="purple")
ax.set_xlabel("Time (ns)", fontsize=12)
ax.set_ylabel("Radius of Gyration (Å)", fontsize=12)
ax.set_title("Protein Radius of Gyration", fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("rg_plot.png", dpi=150)
plt.show()

df_rog.to_csv("radius_of_gyration.csv", index=False)
```

### Principal Component Analysis of Trajectory

```python
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.analysis import pca as mdapca

u = mda.Universe(TOPOLOGY, TRAJECTORY)

pc = mdapca.PCA(u, select="backbone", align=True, n_components=3)
pc.run(verbose=True)

# Project trajectory onto first 3 PCs
transformed = pc.transform(u.select_atoms("backbone"), n_components=3)
df_pca = pd.DataFrame(transformed, columns=["PC1", "PC2", "PC3"])
df_pca["Time (ns)"] = [ts.time / 1000.0 for ts in u.trajectory]

print(f"Variance explained: PC1={pc.results.variance_ratio[0]:.3f}, "
      f"PC2={pc.results.variance_ratio[1]:.3f}, "
      f"PC3={pc.results.variance_ratio[2]:.3f}")

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(df_pca["PC1"], df_pca["PC2"],
                c=df_pca["Time (ns)"], cmap="viridis", s=5, alpha=0.7)
plt.colorbar(sc, ax=ax, label="Time (ns)")
ax.set_xlabel(f"PC1 ({pc.results.variance_ratio[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pc.results.variance_ratio[1]*100:.1f}%)", fontsize=12)
ax.set_title("PCA of Backbone Trajectory", fontsize=13)
plt.tight_layout()
plt.savefig("pca_trajectory.png", dpi=150)
plt.show()

df_pca.to_csv("pca_projection.csv", index=False)
```

### Dihedral Angle (Ramachandran) Analysis

```python
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis.dihedrals import Ramachandran

u = mda.Universe(TOPOLOGY, TRAJECTORY)
protein = u.select_atoms("protein")

rama = Ramachandran(protein).run(verbose=True)

# rama.results.angles shape: (n_frames, n_residues, 2)  [phi, psi] in degrees
phi_psi = rama.results.angles.reshape(-1, 2)  # flatten frames × residues

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(phi_psi[:, 0], phi_psi[:, 1], s=0.5, alpha=0.1, color="navy")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xlabel("Phi (°)", fontsize=12)
ax.set_ylabel("Psi (°)", fontsize=12)
ax.set_title("Ramachandran Plot", fontsize=13)
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
plt.tight_layout()
plt.savefig("ramachandran.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### ImportError: No module named MDAnalysis

```bash
pip install "MDAnalysis>=2.6"
# On some systems you also need:
pip install "MDAnalysisTests>=2.6"
```

### MemoryError with Large Trajectories

Use `in_memory=False` in `AlignTraj` and process frames in chunks:

```python
import MDAnalysis as mda

u = mda.Universe(TOPOLOGY, TRAJECTORY)
chunk_size = 500   # frames per chunk

for start in range(0, u.trajectory.n_frames, chunk_size):
    stop = min(start + chunk_size, u.trajectory.n_frames)
    for ts in u.trajectory[start:stop]:
        pass   # process ts here
```

### NoDataError: Universe has no bonds

Many trajectory formats strip bond information. Provide a topology file that contains
bonds (`.tpr`, `.prmtop`, `.psf`) rather than a bare `.pdb`:

```python
u = mda.Universe("system.tpr", "traj.xtc")   # bonds present in .tpr
```

### Wrong PBC Wrapping (Molecules Split Across Box)

```python
from MDAnalysis.transformations import wrap, unwrap

workflow = [
    unwrap(u.select_atoms("protein")),
    wrap(u.select_atoms("all"), compound="residues"),
]
u.trajectory.add_transformations(*workflow)
```

### Slow Trajectory Iteration

Enable the built-in multi-core parallelism with `AnalysisBase`:

```python
rmsd_analysis.run(n_jobs=4, verbose=True)   # uses multiprocessing
```

---

## External Resources

- MDAnalysis Documentation: https://docs.mdanalysis.org
- MDAnalysis Tutorials: https://www.mdanalysis.org/MDAnalysisTutorial/
- GROMACS Manual: https://manual.gromacs.org
- AMBER Tutorials: https://ambermd.org/tutorials/
- MDAnalysis GitHub: https://github.com/MDAnalysis/mdanalysis
- UserGuide Selections: https://userguide.mdanalysis.org/stable/selections.html

---

## Examples

### Example 1 — Full RMSD/RMSF Pipeline with Sample Data

```python
"""
Full pipeline using MDAnalysis built-in test data.
No external files needed — runs out of the box.
"""
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysisTests.datafiles import PSF, DCD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load built-in test Universe (AdK protein, 98 frames)
u = mda.Universe(PSF, DCD)
print(f"System: {u.atoms.n_atoms} atoms, {u.trajectory.n_frames} frames")

# Step 1: Align trajectory to first frame
ref = mda.Universe(PSF, DCD)
aligner = align.AlignTraj(u, ref, select="backbone", in_memory=True)
aligner.run()

# Step 2: RMSD
rmsd_obj = rms.RMSD(u, select="backbone", ref_frame=0)
rmsd_obj.run()
rmsd_arr = rmsd_obj.results.rmsd   # (n_frames, 3)

# Step 3: RMSF
ca = u.select_atoms("name CA")
rmsf_obj = rms.RMSF(ca)
rmsf_obj.run()

# Step 4: Plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(rmsd_arr[:, 1] / 1000, rmsd_arr[:, 2], lw=1.5, color="steelblue")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("RMSD (Å)")
ax1.set_title("Backbone RMSD")
ax1.grid(alpha=0.3)

ax2.bar(ca.resids, rmsf_obj.results.rmsf, width=1, color="coral", alpha=0.8)
ax2.set_xlabel("Residue ID")
ax2.set_ylabel("RMSF (Å)")
ax2.set_title("Per-Residue Cα RMSF")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("adk_rmsd_rmsf.png", dpi=150)
plt.show()
print("Saved adk_rmsd_rmsf.png")
```

### Example 2 — Protein-Ligand Contact Frequency Matrix

```python
"""
Compute per-residue contact frequency with a ligand over a trajectory.
Requires your own topology + trajectory files.
"""
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import distance_array

TOPOLOGY   = "complex.tpr"
TRAJECTORY = "complex.xtc"
LIGAND_RESNAME = "LIG"
CONTACT_CUTOFF = 4.5   # Å heavy-atom cutoff

u = mda.Universe(TOPOLOGY, TRAJECTORY)
protein = u.select_atoms("protein")
ligand  = u.select_atoms(f"resname {LIGAND_RESNAME}")

if ligand.n_atoms == 0:
    raise ValueError(f"No atoms found with resname '{LIGAND_RESNAME}'. "
                     "Check your topology residue names.")

residues = protein.residues
n_res = len(residues)
contact_count = np.zeros(n_res, dtype=np.int32)

print(f"Protein residues: {n_res}, Ligand atoms: {ligand.n_atoms}")
print(f"Processing {u.trajectory.n_frames} frames...")

for ts in u.trajectory:
    lig_pos = ligand.positions
    for i, res in enumerate(residues):
        prot_heavy = res.atoms.select_atoms("not name H*")
        if prot_heavy.n_atoms == 0:
            continue
        dists = distance_array(prot_heavy.positions, lig_pos, box=ts.dimensions)
        if np.any(dists < CONTACT_CUTOFF):
            contact_count[i] += 1

contact_freq = contact_count / u.trajectory.n_frames

df_contacts = pd.DataFrame({
    "ResID"   : [r.resid for r in residues],
    "ResName" : [r.resname for r in residues],
    "Frequency": contact_freq,
})
df_contacts = df_contacts[df_contacts["Frequency"] > 0.05]  # filter < 5%

print("\nResidue-ligand contacts (frequency > 5%):")
print(df_contacts.sort_values("Frequency", ascending=False).to_string(index=False))

# Bar plot of contact frequencies
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(df_contacts["ResID"].astype(str), df_contacts["Frequency"],
       color="teal", alpha=0.85)
ax.set_xlabel("Residue ID", fontsize=12)
ax.set_ylabel("Contact Frequency", fontsize=12)
ax.set_title(f"Protein-Ligand ({LIGAND_RESNAME}) Contact Frequency (cutoff={CONTACT_CUTOFF} Å)", fontsize=13)
ax.axhline(0.5, color="red", ls="--", label="50% threshold")
ax.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("ligand_contact_frequency.png", dpi=150)
plt.show()

df_contacts.to_csv("ligand_contacts.csv", index=False)
print("Saved ligand_contacts.csv and ligand_contact_frequency.png")
```

### Example 3 — Hydrogen Bond Time Series for a Key Residue Pair

```python
"""
Track the hydrogen bond distance between two specific residue atoms over time.
Useful for monitoring a known catalytic or allosteric interaction.
"""
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MDAnalysis.lib.distances import calc_bonds

TOPOLOGY   = "system.tpr"
TRAJECTORY = "traj.xtc"

# Modify these to match your system
DONOR_SEL    = "resid 57 and name NE2"    # e.g. His57 NE2
ACCEPTOR_SEL = "resid 102 and name OD1"   # e.g. Asp102 OD1

u = mda.Universe(TOPOLOGY, TRAJECTORY)
donor    = u.select_atoms(DONOR_SEL)
acceptor = u.select_atoms(ACCEPTOR_SEL)

if donor.n_atoms != 1 or acceptor.n_atoms != 1:
    raise ValueError("Donor and acceptor selections must each return exactly 1 atom.")

times, dist_vals = [], []
for ts in u.trajectory:
    d = calc_bonds(donor.positions, acceptor.positions, box=ts.dimensions)[0]
    times.append(ts.time / 1000.0)
    dist_vals.append(float(d))

df_hbdist = pd.DataFrame({"Time (ns)": times, "D-A Distance (Å)": dist_vals})

# Fraction of frames with hydrogen bond (D-A < 3.5 Å)
hb_fraction = (df_hbdist["D-A Distance (Å)"] < 3.5).mean()
print(f"H-bond occupancy ({DONOR_SEL} -- {ACCEPTOR_SEL}): {hb_fraction:.3f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_hbdist["Time (ns)"], df_hbdist["D-A Distance (Å)"], lw=0.8, color="navy")
ax.axhline(3.5, color="red", ls="--", label="H-bond cutoff (3.5 Å)")
ax.fill_between(df_hbdist["Time (ns)"], df_hbdist["D-A Distance (Å)"], 3.5,
                where=df_hbdist["D-A Distance (Å)"] < 3.5,
                alpha=0.15, color="green", label=f"H-bond formed ({hb_fraction:.0%})")
ax.set_xlabel("Time (ns)", fontsize=12)
ax.set_ylabel("Donor-Acceptor Distance (Å)", fontsize=12)
ax.set_title(f"Hydrogen Bond: {DONOR_SEL} ··· {ACCEPTOR_SEL}", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("hbond_timeseries.png", dpi=150)
plt.show()

df_hbdist.to_csv("hbond_timeseries.csv", index=False)
print("Saved hbond_timeseries.csv and hbond_timeseries.png")
```
