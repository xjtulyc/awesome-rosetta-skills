---
name: lammps-md
description: >
  Use this Skill for classical molecular dynamics simulations with LAMMPS:
  force field selection (LJ, EAM, CHARMM), NVT/NPT ensembles, RDF/MSD
  post-processing, and OVITO visualization.
tags:
  - physics
  - molecular-dynamics
  - LAMMPS
  - force-field
  - simulation
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
    - ase>=3.22
    - numpy>=1.23
    - matplotlib>=3.6
    - ovito>=3.9
  system:
    - lammps
last_updated: "2026-03-17"
status: stable
---

# LAMMPS Molecular Dynamics

> **TL;DR** — Run classical molecular dynamics (MD) simulations with LAMMPS:
> choose a force field (LJ, EAM, CHARMM), select an ensemble (NVE/NVT/NPT),
> then post-process dump files to compute RDF, MSD, and diffusion coefficients.
> Visualize trajectories with OVITO.

---

## When to Use

Use this Skill when you need to:

- Simulate atomic or molecular systems at finite temperature using empirical potentials
- Compute structural properties (radial distribution function, coordination number)
- Extract dynamic properties (mean squared displacement, diffusion coefficient)
- Generate equilibrated configurations as starting points for ab initio or ML-FF calculations
- Visualize large MD trajectories with coloring by atom type, velocity, or potential energy

Do **not** use this Skill when:
- You need electronic structure accuracy → use DFT (VASP, Quantum ESPRESSO)
- Simulating systems with bond breaking/forming using ReaxFF → see separate ReaxFF skill
- You want coarse-grained polymer models → consider HOOMD-blue or GROMACS

---

## Background & Key Concepts

### Force Fields

| Force Field | System | Interaction Form |
|---|---|---|
| Lennard-Jones (LJ) | Noble gases, simple liquids | `4ε[(σ/r)¹²−(σ/r)⁶]` |
| EAM (Embedded Atom Method) | Metals (Cu, Ni, Fe, Al) | Pair + embedding energy |
| CHARMM / AMBER | Biomolecules, organic molecules | Bond + angle + dihedral + nonbond |
| Tersoff / AIREBO | Covalent materials (Si, C) | Three-body angular terms |

### Statistical Ensembles

- **NVE** (microcanonical): constant N, V, E — no thermostat, conserves total energy
- **NVT** (canonical): constant N, V, T — Nosé-Hoover thermostat controls temperature
- **NPT** (isothermal-isobaric): constant N, P, T — Parrinello-Rahman barostat

### LAMMPS Input Script Anatomy

A LAMMPS input consists of five logical sections:

1. **Initialization** — `units`, `atom_style`, `boundary`
2. **System definition** — `read_data` or `lattice` + `create_atoms`
3. **Force field** — `pair_style`, `pair_coeff`, `kspace_style`
4. **Settings** — `fix` for thermostat/barostat, `dump` for trajectory output
5. **Run** — `minimize`, `run`

---

## Environment Setup

```bash
# Install LAMMPS (Ubuntu/Debian via conda-forge)
conda create -n lammps-env python=3.11 -y
conda activate lammps-env
conda install -c conda-forge lammps -y

# Python post-processing tools
pip install ase numpy matplotlib ovito

# Verify LAMMPS
lammps -h | head -5

# Verify Python packages
python -c "import ase, ovito, numpy, matplotlib; print('All packages OK')"
```

For HPC clusters, load the module:

```bash
module load lammps/2023-Aug
module load python/3.11
pip install --user ase numpy matplotlib ovito
```

---

## Core Workflow

### Step 1 — Write the LAMMPS Input Script

Below is a complete NVT simulation of liquid argon using the Lennard-Jones potential.
Save as `argon_nvt.in`:

```bash
# ============================================================
# LAMMPS input: LJ argon NVT at 94 K
# Units: real (energy=kcal/mol, length=Angstrom, time=fs)
# ============================================================

# --- Initialization ---
units           real
atom_style      atomic
boundary        p p p

# --- System definition ---
# FCC lattice, lattice constant 5.26 Å → density ≈ 1.4 g/cm³
lattice         fcc 5.260
region          box block 0 6 0 6 0 6           # 6×6×6 unit cells = 864 atoms
create_box      1 box
create_atoms    1 box

# --- Masses ---
mass            1 39.948                         # Ar atomic mass (g/mol)

# --- Force field: LJ with ε=0.238 kcal/mol, σ=3.405 Å ---
pair_style      lj/cut 10.0                      # cutoff 10 Å
pair_coeff      1 1 0.238 3.405

# --- Settings ---
timestep        2.0                              # 2 fs
thermo          100
thermo_style    custom step temp press etotal ke pe

# --- Energy minimization (removes bad contacts) ---
minimize        1.0e-4 1.0e-6 1000 10000

# --- Velocity initialization at 94 K ---
velocity        all create 94.0 87654 dist gaussian

# --- NVT thermostat: Nosé-Hoover, T=94 K, damping=200 fs ---
fix             nvt all nvt temp 94.0 94.0 200.0

# --- Dump trajectory every 500 steps ---
dump            traj all atom 500 argon_nvt.dump
dump_modify     traj sort id

# --- Production run: 50000 steps = 100 ps ---
run             50000

# --- Write final configuration ---
write_data      argon_final.data
```

Run the simulation:

```bash
lammps -in argon_nvt.in -log argon_nvt.log
```

### Step 2 — Parse Dump File and Compute RDF

```python
"""
Compute radial distribution function (RDF) from a LAMMPS dump file.
Uses ASE for parsing; numpy for histogram computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read


def parse_lammps_dump(dump_file: str, frame_start: int = 50) -> list:
    """
    Read LAMMPS dump file using ASE.

    Args:
        dump_file:   Path to the LAMMPS atom dump file.
        frame_start: Index of the first frame to include (skip equilibration).

    Returns:
        List of ASE Atoms objects for the selected frames.
    """
    frames = read(dump_file, index=":", format="lammps-dump-text")
    return frames[frame_start:]


def compute_rdf(
    frames: list,
    r_max: float = 10.0,
    n_bins: int = 200,
    atom_type_1: int = 0,
    atom_type_2: int = 0,
) -> tuple:
    """
    Compute the radial distribution function g(r) by averaging over frames.

    The RDF is defined as:
        g(r) = (V / N²) * dn(r) / (4π r² Δr)
    where dn(r) is the number of pairs with separation in [r, r+Δr].

    Args:
        frames:       List of ASE Atoms objects.
        r_max:        Maximum distance in Angstroms.
        n_bins:       Number of histogram bins.
        atom_type_1:  Index of species 1 (0-indexed by ASE species list).
        atom_type_2:  Index of species 2.

    Returns:
        Tuple (r_centers, g_r) — bin centers (Å) and RDF values.
    """
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    dr = bin_edges[1] - bin_edges[0]
    rdf_accum = np.zeros(n_bins)

    for atoms in frames:
        n_atoms = len(atoms)
        cell = atoms.get_cell()
        volume = atoms.get_volume()
        positions = atoms.get_positions()

        # All pairwise distances with minimum image convention
        counts = np.zeros(n_bins)
        for i in range(n_atoms - 1):
            diff = positions[i + 1:] - positions[i]
            # Minimum image
            for k in range(3):
                diff[:, k] -= np.round(diff[:, k] / cell[k, k]) * cell[k, k]
            r = np.sqrt((diff ** 2).sum(axis=1))
            r = r[r < r_max]
            hist, _ = np.histogram(r, bins=bin_edges)
            counts += hist

        # Normalize
        r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        shell_vol = 4.0 * np.pi * r_centers ** 2 * dr
        rho = n_atoms / volume
        g_r = 2.0 * counts / (n_atoms * shell_vol * rho)
        rdf_accum += g_r

    return r_centers, rdf_accum / len(frames)


def plot_rdf(r: np.ndarray, g_r: np.ndarray, output_path: str = "rdf.png") -> None:
    """Plot and save the radial distribution function."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r, g_r, color="#2196F3", linewidth=1.5, label="Ar-Ar RDF")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_title("Radial Distribution Function — Liquid Argon (NVT, 94 K)")
    ax.legend()
    ax.set_xlim(0, r.max())
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"RDF plot saved to {output_path}")


if __name__ == "__main__":
    frames = parse_lammps_dump("argon_nvt.dump", frame_start=20)
    r, g_r = compute_rdf(frames, r_max=10.0, n_bins=200)
    plot_rdf(r, g_r, "argon_rdf.png")

    # First peak position gives the nearest-neighbor distance
    peak_idx = np.argmax(g_r[10:]) + 10
    print(f"First RDF peak at r = {r[peak_idx]:.2f} Å  (g(r) = {g_r[peak_idx]:.2f})")
```

### Step 3 — MSD and Diffusion Coefficient

```python
"""
Compute mean squared displacement (MSD) and the self-diffusion coefficient
from an NVT trajectory.

Einstein relation:
    D = lim_{t→∞} MSD(t) / (6t)   [3D system]
    D = lim_{t→∞} MSD(t) / (4t)   [2D system]
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read


def compute_msd(
    dump_file: str,
    n_origins: int = 10,
    frame_skip: int = 1,
    timestep_fs: float = 2.0,
    dump_every: int = 500,
) -> tuple:
    """
    Compute MSD using multiple time origins for better statistics.

    Args:
        dump_file:   Path to the LAMMPS dump file.
        n_origins:   Number of time origins to average over.
        frame_skip:  Stride for reading frames (1 = read all).
        timestep_fs: MD timestep in femtoseconds.
        dump_every:  Dump frequency in steps.

    Returns:
        Tuple (time_ps, msd_angstrom2) — time axis and MSD values.
    """
    frames = read(dump_file, index=f"::{frame_skip}", format="lammps-dump-text")
    n_frames = len(frames)
    n_atoms = len(frames[0])
    dt_ps = timestep_fs * dump_every * frame_skip / 1000.0  # fs → ps

    # Stack positions: shape (n_frames, n_atoms, 3)
    positions = np.array([f.get_positions() for f in frames])

    # Unwrap PBC using ASE cell info
    cell = frames[0].get_cell().diagonal()
    for t in range(1, n_frames):
        delta = positions[t] - positions[t - 1]
        delta -= np.round(delta / cell) * cell
        positions[t] = positions[t - 1] + delta

    max_lag = n_frames // 2
    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag, dtype=int)

    origin_stride = max(1, n_frames // n_origins)
    origins = range(0, n_frames - max_lag, origin_stride)

    for t0 in origins:
        for lag in range(1, max_lag):
            if t0 + lag >= n_frames:
                break
            disp = positions[t0 + lag] - positions[t0]
            msd[lag] += np.mean((disp ** 2).sum(axis=1))
            counts[lag] += 1

    valid = counts > 0
    msd[valid] /= counts[valid]
    time_ps = np.arange(max_lag) * dt_ps

    return time_ps[1:], msd[1:]


def fit_diffusion_coefficient(
    time_ps: np.ndarray,
    msd: np.ndarray,
    fit_start_frac: float = 0.2,
    fit_end_frac: float = 0.8,
) -> float:
    """
    Fit the linear regime of MSD to extract the self-diffusion coefficient.

    Args:
        time_ps:        Time array in picoseconds.
        msd:            MSD array in Å².
        fit_start_frac: Fraction of total time to start linear fit.
        fit_end_frac:   Fraction of total time to end linear fit.

    Returns:
        Diffusion coefficient D in cm²/s.
    """
    n = len(time_ps)
    i0 = int(n * fit_start_frac)
    i1 = int(n * fit_end_frac)

    slope, intercept = np.polyfit(time_ps[i0:i1], msd[i0:i1], 1)
    # slope has units Å²/ps; D = slope / 6 (3D)
    # Convert: 1 Å²/ps = 1e-20 m² / 1e-12 s = 1e-8 m²/s = 1e-4 cm²/s
    D_cm2_per_s = slope / 6.0 * 1e-4
    return D_cm2_per_s


def plot_msd(
    time_ps: np.ndarray,
    msd: np.ndarray,
    D: float,
    output_path: str = "msd.png",
) -> None:
    """Plot MSD curve and annotate with diffusion coefficient."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_ps, msd, color="#E91E63", linewidth=1.5, label="MSD")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("MSD (Å²)")
    ax.set_title(f"Mean Squared Displacement — D = {D:.3e} cm²/s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"MSD plot saved to {output_path}")


if __name__ == "__main__":
    time_ps, msd = compute_msd("argon_nvt.dump", n_origins=20)
    D = fit_diffusion_coefficient(time_ps, msd)
    print(f"Self-diffusion coefficient D = {D:.3e} cm²/s")
    print(f"  (Literature: Ar at 94 K ≈ 2.4×10⁻⁵ cm²/s)")
    plot_msd(time_ps, msd, D, "argon_msd.png")
```

---

## Advanced Usage

### EAM Potential for Metals (Copper)

```bash
# Requires: Cu_u3.eam from NIST or LAMMPS potentials directory
units           metal
atom_style      atomic
boundary        p p p

lattice         fcc 3.615              # Cu FCC lattice constant (Å)
region          box block 0 8 0 8 0 8
create_box      1 box
create_atoms    1 box

mass            1 63.546

pair_style      eam
pair_coeff      1 1 Cu_u3.eam

timestep        0.001                  # 1 fs (metal units: time in ps)
thermo          200

velocity        all create 300.0 12345

fix             npt all npt temp 300.0 300.0 0.1 iso 0.0 0.0 1.0
dump            traj all atom 1000 cu_npt.dump
run             100000
write_data      cu_final.data
```

### OVITO Python API for Visualization

```python
"""
Use OVITO's Python API to load a LAMMPS dump, color by kinetic energy,
and render a PNG snapshot.
"""

import ovito
from ovito.io import import_file
from ovito.modifiers import ColorCodingModifier, CommonNeighborAnalysisModifier
from ovito.vis import Viewport, RenderSettings
import ovito.vis


def render_trajectory_snapshot(
    dump_file: str,
    frame_index: int = -1,
    output_image: str = "snapshot.png",
    width: int = 800,
    height: int = 600,
) -> None:
    """
    Load a LAMMPS dump file in OVITO, apply CNA structure identification,
    color atoms by structure type, and render a PNG.

    Args:
        dump_file:    Path to the LAMMPS dump file.
        frame_index:  Frame to render (-1 = last frame).
        output_image: Output PNG file path.
        width:        Image width in pixels.
        height:       Image height in pixels.
    """
    # Import pipeline
    pipeline = import_file(dump_file, multiple_frames=True)

    # Common Neighbor Analysis to identify FCC/HCP/BCC/other structures
    cna = CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff)
    pipeline.modifiers.append(cna)

    # Color by structure type
    color_mod = ColorCodingModifier(
        property="Structure Type",
        gradient=ColorCodingModifier.Rainbow(),
    )
    pipeline.modifiers.append(color_mod)

    # Evaluate at the requested frame
    n_frames = pipeline.source.num_frames
    frame = n_frames - 1 if frame_index == -1 else frame_index
    data = pipeline.compute(frame)
    print(f"Frame {frame}: {data.particles.count} atoms")

    # Setup viewport and render
    vp = Viewport(type=Viewport.Type.Perspective)
    vp.zoom_all()
    rs = RenderSettings(size=(width, height), filename=output_image)
    vp.render_image(settings=rs, frame=frame, pipeline=pipeline)
    print(f"Snapshot saved to {output_image}")


if __name__ == "__main__":
    render_trajectory_snapshot(
        dump_file="argon_nvt.dump",
        frame_index=-1,
        output_image="argon_snapshot.png",
    )
```

### NPT Equation of State Calculation

```python
"""
Run LAMMPS across a grid of (T, P) conditions to build an equation of state.
Uses Python subprocess to launch LAMMPS jobs.
"""

import subprocess
import os
import numpy as np
import pandas as pd


LAMMPS_TEMPLATE = """
units           real
atom_style      atomic
boundary        p p p
read_data       argon_final.data

pair_style      lj/cut 10.0
pair_coeff      1 1 0.238 3.405

timestep        2.0
thermo          500
thermo_style    custom step temp press vol density etotal

fix             npt all npt temp {T} {T} 200.0 iso {P} {P} 2000.0
run             25000

variable        avg_density equal density
variable        avg_vol     equal vol
print           "RESULT density ${{avg_density}} vol ${{avg_vol}}"
"""


def run_eos_point(T: float, P: float, work_dir: str) -> dict:
    """Run a single LAMMPS NPT point and return density and volume."""
    os.makedirs(work_dir, exist_ok=True)
    script = os.path.join(work_dir, "run.in")
    log    = os.path.join(work_dir, "run.log")

    with open(script, "w") as f:
        f.write(LAMMPS_TEMPLATE.format(T=T, P=P))

    result = subprocess.run(
        ["lammps", "-in", script, "-log", log],
        capture_output=True, text=True
    )

    # Parse log for average density
    density = np.nan
    for line in open(log):
        if "RESULT density" in line:
            parts = line.split()
            density = float(parts[parts.index("density") + 1])

    return {"T_K": T, "P_atm": P, "density_g_cm3": density}


def compute_eos(
    temperatures: list = [80, 94, 110],
    pressures: list = [1, 100, 500],
) -> pd.DataFrame:
    """Compute EOS on a T-P grid."""
    records = []
    for T in temperatures:
        for P in pressures:
            wd = f"eos_T{T}_P{P}"
            rec = run_eos_point(T, P, wd)
            records.append(rec)
            print(f"T={T} K, P={P} atm → ρ={rec['density_g_cm3']:.4f} g/cm³")
    df = pd.DataFrame(records)
    df.to_csv("eos_argon.csv", index=False)
    return df
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Dangerous builds are not allowed` | Atoms outside simulation box after build | Add `boundary p p p` and check lattice constant |
| `Out of range atoms - cannot compute PPPM` | Atoms flew out of box | Reduce timestep; add `fix nve/limit` during equilibration |
| `Lost atoms: original N atoms now M` | Simulation exploded | Energy-minimize first; check force field parameters |
| `Communication cutoff is shorter than the ghost cutoff` | Cutoff > half box length | Increase box size or reduce cutoff |
| OVITO ImportError | Missing ovito package | `pip install ovito` or use ovito Pro |
| ASE cannot read dump | Non-standard dump columns | Add `dump_modify traj sort id` to LAMMPS script |
| Diffusion coefficient off by 6x | Wrong dimension factor | Use `/ 6` for 3D; `/ 4` for 2D |
| RDF integrates to wrong coordination number | PBC not applied | Ensure minimum image convention in displacement calculation |

---

## External Resources

- LAMMPS documentation: <https://docs.lammps.org>
- LAMMPS potentials library: <https://www.ctcms.nist.gov/potentials/>
- ASE I/O for LAMMPS: <https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#lammps-dump>
- OVITO documentation: <https://www.ovito.org/docs/>
- Frenkel & Smit, "Understanding Molecular Simulation", 2nd ed. (Academic Press)
- Allen & Tildesley, "Computer Simulation of Liquids", 2nd ed. (Oxford)

---

## Examples

### Example 1 — Full LJ Argon NVT Pipeline

```python
import subprocess
import os

# 1. Write LAMMPS input
lammps_input = """
units       real
atom_style  atomic
boundary    p p p
lattice     fcc 5.260
region      box block 0 5 0 5 0 5
create_box  1 box
create_atoms 1 box
mass        1 39.948
pair_style  lj/cut 10.0
pair_coeff  1 1 0.238 3.405
timestep    2.0
thermo      200
minimize    1.0e-4 1.0e-6 1000 10000
velocity    all create 94.0 42 dist gaussian
fix         nvt all nvt temp 94.0 94.0 200.0
dump        traj all atom 500 argon.dump
run         20000
write_data  argon_eq.data
"""
with open("run_argon.in", "w") as f:
    f.write(lammps_input)

# 2. Run LAMMPS
ret = subprocess.run(["lammps", "-in", "run_argon.in"], capture_output=True, text=True)
if ret.returncode != 0:
    print("LAMMPS error:", ret.stderr[:500])
else:
    print("LAMMPS simulation complete")

# 3. Post-process
if os.path.exists("argon.dump"):
    from ase.io import read
    import numpy as np
    frames = read("argon.dump", index="10:", format="lammps-dump-text")
    print(f"Loaded {len(frames)} production frames")

    r, g_r = compute_rdf(frames, r_max=10.0)
    peak = r[np.argmax(g_r[5:])+5]
    print(f"Nearest-neighbor distance: {peak:.2f} Å")
```

### Example 2 — EAM Copper Melting Point Estimation

```python
"""
Estimate the melting point of Cu by tracking potential energy vs temperature
in a series of NPT simulations.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulated PE data (replace with actual LAMMPS output from NPT runs)
temperatures = np.array([800, 900, 1000, 1100, 1200, 1300, 1357, 1400, 1500])
# Potential energy per atom (eV) — shows discontinuity at melting
pe_per_atom = np.array([-3.40, -3.35, -3.29, -3.23, -3.16, -3.08, -2.98, -2.71, -2.64])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(temperatures, pe_per_atom, "o-", color="#FF5722", linewidth=1.5)
ax.axvline(1357, color="gray", linestyle="--", linewidth=0.8, label="Exp. T_m = 1357 K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Potential Energy per Atom (eV)")
ax.set_title("Cu Melting: Potential Energy vs Temperature (EAM)")
ax.legend()
fig.tight_layout()
fig.savefig("cu_melting.png", dpi=150)
print("Melting point analysis saved to cu_melting.png")
```

### Example 3 — Coordination Number from RDF

```python
"""
Compute coordination number by integrating the first peak of g(r).
The coordination number gives the average number of nearest neighbors.
"""

import numpy as np


def coordination_number(
    r: np.ndarray,
    g_r: np.ndarray,
    r_min: float,
    r_max: float,
    rho: float,
) -> float:
    """
    Integrate g(r) from r_min to r_max to get coordination number.

        Z = 4π ρ ∫_{r_min}^{r_max} g(r) r² dr

    Args:
        r:     Radial distance array (Å).
        g_r:   RDF values.
        r_min: Integration lower bound (Å) — start of first peak.
        r_max: Integration upper bound (Å) — first minimum after peak.
        rho:   Number density of the fluid (atoms/Å³).

    Returns:
        Coordination number Z (dimensionless).
    """
    mask = (r >= r_min) & (r <= r_max)
    dr = r[1] - r[0]
    Z = 4.0 * np.pi * rho * np.trapz(g_r[mask] * r[mask] ** 2, r[mask])
    return Z


# Example: liquid argon at 94 K, density ≈ 0.021 atoms/Å³
rho_Ar = 0.021  # atoms/Å³
# Assuming r, g_r computed from compute_rdf() above
# Z = coordination_number(r, g_r, r_min=3.0, r_max=5.5, rho=rho_Ar)
# print(f"Coordination number: {Z:.1f}  (expected ~12 for liquid Ar)")
print("Coordination number formula ready — supply r, g_r from compute_rdf()")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — LJ argon NVT, EAM copper, RDF, MSD, OVITO visualization |
