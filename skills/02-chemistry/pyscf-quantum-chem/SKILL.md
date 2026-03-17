---
name: pyscf-quantum-chem
description: >
  Use this Skill for quantum chemistry calculations with PySCF: HF/DFT single-point
  energy, geometry optimization, MP2/CCSD(T) post-HF methods, basis set selection,
  and molecular orbital analysis.
tags:
  - chemistry
  - quantum-chemistry
  - PySCF
  - DFT
  - ab-initio
  - molecular-orbitals
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
    - pyscf>=2.3
    - numpy>=1.23
    - scipy>=1.9
    - matplotlib>=3.6
    - ase>=3.22
last_updated: "2026-03-17"
status: stable
---

# PySCF Quantum Chemistry

> **TL;DR** — Run ab initio and DFT calculations with PySCF: HF/B3LYP single-point
> energies, geometry optimization via ASE, MP2 and CCSD(T) correlation energies,
> Mulliken/Lowdin population analysis, and frontier MO visualization.

---

## When to Use

Use this Skill when you need to:

- Compute single-point energies at HF, DFT (B3LYP, PBE0, wB97X-D), MP2, or CCSD(T) level
- Optimize molecular geometries using DFT with ASE as the driver
- Analyze electronic structure: Mulliken/Lowdin charges, orbital energies
- Extract HOMO/LUMO energies and frontier molecular orbital gaps
- Benchmark small molecule energetics against high-accuracy CCSD(T) reference data
- Generate cube files for orbital visualization
- Study spin-unrestricted systems (radicals, triplet states)

| Method | Accuracy | Cost | When to Use |
|---|---|---|---|
| HF/STO-3G | Low | Minimal | Quick checks, large molecules |
| B3LYP/6-31G* | Medium | Low | Routine geometry, NMR, IR |
| PBE0/cc-pVDZ | Medium-high | Medium | More reliable thermochemistry |
| MP2/cc-pVTZ | High | High | Weak interactions, correlated |
| CCSD(T)/cc-pVTZ | Very high | Very high | Benchmark, small molecules only |

---

## Background & Key Concepts

### SCF Convergence

Hartree-Fock and DFT solve the Roothaan-Hall eigenvalue problem iteratively (SCF).
Key parameters:

- **conv_tol**: SCF energy convergence threshold (default 1e-9 Eh)
- **conv_tol_grad**: Gradient threshold (default ~3e-6)
- **DIIS**: Direct Inversion in the Iterative Subspace accelerates convergence

### Basis Sets

| Name | Quality | Notes |
|---|---|---|
| STO-3G | Minimal | 3 Gaussians per Slater; for demonstration only |
| 6-31G* | Split-valence + polarization | Common for organic geometry optimization |
| cc-pVDZ / cc-pVTZ | Correlation-consistent | Designed for correlated methods (MP2, CC) |
| aug-cc-pVTZ | Augmented | Diffuse functions; needed for anions, excited states |

### Post-HF Methods

- **MP2**: Second-order Moller-Plesset perturbation theory. Good for weak interactions.
  Scales as O(N^5).
- **CCSD**: Coupled cluster singles and doubles. Scales O(N^6); gold standard for
  thermochemistry within ~2 kJ/mol.
- **CCSD(T)**: CCSD + perturbative triples. Scales O(N^7); "gold standard" of quantum
  chemistry for small molecules.

---

## Environment Setup

```bash
# Create conda environment
conda create -n pyscf-env python=3.11 -y
conda activate pyscf-env

# Install PySCF and dependencies
pip install pyscf numpy scipy matplotlib ase

# Optional: GPU acceleration (requires CUDA)
# pip install pyscf-gpu

# Verify
python -c "import pyscf; print('PySCF version:', pyscf.__version__)"
python -c "from ase import Atoms; print('ASE OK')"
```

Set the number of OpenMP threads for multi-core runs:

```bash
export OMP_NUM_THREADS=4
export PYSCF_MAX_MEMORY=8000   # MB of RAM allocated to PySCF
```

---

## Core Workflow

### Step 1 — Build a Molecule Object

```python
from pyscf import gto, scf, dft, mp, cc
import numpy as np


def build_molecule(
    atom_spec: str,
    basis: str = "6-31g*",
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    symmetry: bool = False,
    verbose: int = 3,
) -> gto.Mole:
    """
    Build a PySCF Mole object from a Z-matrix or Cartesian atom specification.

    Args:
        atom_spec:  Atom coordinates as a multiline string, e.g.:
                    'O 0 0 0; H 0.96 0 0; H -0.24 0.93 0'
                    or as a list of (symbol, (x,y,z)) tuples.
        basis:      Basis set name. Supported: 'sto-3g', '6-31g*', 'cc-pvdz',
                    'cc-pvtz', 'aug-cc-pvtz'.
        charge:     Molecular charge (0 = neutral).
        spin:       Number of unpaired electrons (0 = singlet/closed-shell).
        unit:       Coordinate unit: 'Angstrom' or 'Bohr'.
        symmetry:   Use molecular symmetry to speed up calculation.
        verbose:    PySCF verbosity level (0=silent, 3=info, 5=debug).

    Returns:
        Built and initialized gto.Mole object.
    """
    mol = gto.Mole()
    mol.atom   = atom_spec
    mol.basis  = basis
    mol.charge = charge
    mol.spin   = spin
    mol.unit   = unit
    mol.symmetry = symmetry
    mol.verbose  = verbose
    mol.build()
    print(f"Molecule: {mol.nao} AOs, {mol.nelectron} electrons, "
          f"charge={charge}, spin={spin}, basis={basis}")
    return mol


# Water molecule at experimental geometry
water_xyz = """
O  0.000000  0.000000  0.119748
H  0.000000  0.757452 -0.478993
H  0.000000 -0.757452 -0.478993
"""

mol_water = build_molecule(water_xyz, basis="6-31g*")
```

### Step 2 — Hartree-Fock Single Point Energy

```python
def run_hf(
    mol: gto.Mole,
    unrestricted: bool = False,
    conv_tol: float = 1e-9,
    max_cycle: int = 100,
) -> dict:
    """
    Run a restricted (RHF) or unrestricted (UHF) Hartree-Fock calculation.

    Args:
        mol:          PySCF Mole object.
        unrestricted: If True, run UHF (for open-shell systems).
        conv_tol:     SCF convergence threshold in Hartree.
        max_cycle:    Maximum number of SCF iterations.

    Returns:
        Dictionary with energy, orbital energies, HOMO/LUMO gap, converged flag.
    """
    if unrestricted:
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)

    mf.conv_tol   = conv_tol
    mf.max_cycle  = max_cycle
    mf.kernel()

    if not mf.converged:
        print("WARNING: SCF did not converge!")

    n_occ = mol.nelectron // 2
    mo_energies = mf.mo_energy                  # Hartree
    homo_idx    = n_occ - 1
    lumo_idx    = n_occ
    homo_e      = mo_energies[homo_idx] * 27.2114  # eV
    lumo_e      = mo_energies[lumo_idx] * 27.2114  # eV

    return {
        "energy_hartree": mf.e_tot,
        "energy_kcal": mf.e_tot * 627.509,
        "homo_ev": homo_e,
        "lumo_ev": lumo_e,
        "gap_ev": lumo_e - homo_e,
        "converged": mf.converged,
        "mf_object": mf,
        "mo_energies_ev": mo_energies * 27.2114,
    }


hf_result = run_hf(mol_water)
print(f"HF/6-31G* energy: {hf_result['energy_hartree']:.6f} Eh")
print(f"HOMO: {hf_result['homo_ev']:.2f} eV, LUMO: {hf_result['lumo_ev']:.2f} eV")
print(f"HOMO-LUMO gap: {hf_result['gap_ev']:.2f} eV")
```

### Step 3 — DFT Calculation with Functional Selection

```python
def run_dft(
    mol: gto.Mole,
    functional: str = "b3lyp",
    grid_level: int = 3,
    conv_tol: float = 1e-9,
) -> dict:
    """
    Run a DFT single-point energy calculation.

    Args:
        mol:        PySCF Mole object.
        functional: XC functional name. Common choices:
                    'b3lyp', 'pbe', 'pbe0', 'wb97x-d', 'm06-2x', 'tpss'.
        grid_level: Numerical integration grid quality (1=coarse .. 9=fine).
                    Level 3 is sufficient for most cases.
        conv_tol:   SCF convergence threshold.

    Returns:
        Dictionary with DFT energy, orbital energies, HOMO/LUMO gap, dipole moment.
    """
    mf = dft.RKS(mol)
    mf.xc         = functional
    mf.grids.level = grid_level
    mf.conv_tol   = conv_tol
    mf.kernel()

    if not mf.converged:
        print(f"WARNING: DFT/{functional} SCF did not converge!")

    n_occ = mol.nelectron // 2
    mo_e  = mf.mo_energy
    homo_e = mo_e[n_occ - 1] * 27.2114
    lumo_e = mo_e[n_occ]     * 27.2114

    dipole = mf.dip_moment(unit="Debye")

    return {
        "functional": functional,
        "energy_hartree": mf.e_tot,
        "homo_ev": homo_e,
        "lumo_ev": lumo_e,
        "gap_ev": lumo_e - homo_e,
        "dipole_debye": np.linalg.norm(dipole),
        "converged": mf.converged,
        "mf_object": mf,
    }


# Compare functionals
for func in ["b3lyp", "pbe0"]:
    res = run_dft(mol_water, functional=func, grid_level=3)
    print(f"{func.upper():8s}/6-31G*: E={res['energy_hartree']:.6f} Eh, "
          f"gap={res['gap_ev']:.2f} eV, dipole={res['dipole_debye']:.2f} D")
```

---

## Advanced Usage

### Mulliken and Lowdin Population Analysis

```python
def population_analysis(mf, mol: gto.Mole) -> pd.DataFrame:
    """
    Compute Mulliken and Lowdin atomic charges from a converged SCF.

    Args:
        mf:  Converged PySCF SCF object (RHF, RKS, etc.).
        mol: Corresponding PySCF Mole object.

    Returns:
        DataFrame with columns: atom, symbol, mulliken_charge, lowdin_charge.
    """
    import pandas as pd

    # Mulliken
    mulliken = mf.mulliken_pop(verbose=0)
    mulliken_charges = mulliken[1]   # array of net charges per atom

    # Lowdin
    lowdin = mf.mulliken_pop_with_meta_lowdin_ao(verbose=0)
    lowdin_charges = lowdin[1]

    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    records = []
    for i, sym in enumerate(symbols):
        records.append({
            "atom_idx": i,
            "symbol": sym,
            "mulliken_charge": round(mulliken_charges[i], 4),
            "lowdin_charge": round(lowdin_charges[i], 4),
        })

    return pd.DataFrame(records)


dft_result = run_dft(mol_water, functional="b3lyp")
import pandas as pd
charges_df = population_analysis(dft_result["mf_object"], mol_water)
print(charges_df.to_string(index=False))
```

### MP2 Correlation Energy

```python
def run_mp2(
    mol: gto.Mole,
    basis: str = "cc-pvdz",
    frozen_core: bool = True,
) -> dict:
    """
    Run MP2 on top of HF reference. Reports HF, MP2 correlation, and total MP2 energy.

    Args:
        mol:         PySCF Mole object (should be built with the target basis).
        basis:       Basis set used (informational only; must match mol.basis).
        frozen_core: Freeze core orbitals (recommended for 1st and 2nd row).

    Returns:
        Dictionary with hf_energy, mp2_correlation, mp2_total (all in Hartree).
    """
    # HF reference
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError("HF reference did not converge")

    # MP2
    mp2_calc = mp.MP2(mf)
    if frozen_core:
        # Freeze orbitals below -10 eV (core)
        frozen = [i for i, e in enumerate(mf.mo_energy) if e < -10.0 / 27.2114]
        mp2_calc.frozen = frozen

    mp2_calc.kernel()
    e_corr = mp2_calc.e_corr
    e_tot  = mf.e_tot + e_corr

    print(f"MP2/{basis}:")
    print(f"  HF energy:          {mf.e_tot:>16.8f} Eh")
    print(f"  MP2 correlation:    {e_corr:>16.8f} Eh")
    print(f"  MP2 total:          {e_tot:>16.8f} Eh")

    return {
        "hf_energy":      mf.e_tot,
        "mp2_correlation": e_corr,
        "mp2_total":       e_tot,
        "mf_object":       mf,
        "mp2_object":      mp2_calc,
    }


mol_water_dz = build_molecule(water_xyz, basis="cc-pvdz", verbose=0)
mp2_res = run_mp2(mol_water_dz, basis="cc-pvdz")
```

### CCSD(T) Benchmark

```python
def run_ccsd_t(
    mol: gto.Mole,
    basis: str = "cc-pvtz",
    frozen_core: bool = True,
) -> dict:
    """
    Run CCSD and CCSD(T) on top of HF reference.
    NOTE: Scales as O(N^7) — only feasible for ~10 heavy atoms or fewer.

    Args:
        mol:         PySCF Mole object.
        basis:       Basis set (informational).
        frozen_core: Freeze core orbitals.

    Returns:
        Dictionary with HF, CCSD, and CCSD(T) total energies.
    """
    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError("HF did not converge")

    cc_calc = cc.CCSD(mf)
    if frozen_core:
        frozen = [i for i, e in enumerate(mf.mo_energy) if e < -10.0 / 27.2114]
        cc_calc.frozen = frozen

    cc_calc.kernel()
    e_ccsd = mf.e_tot + cc_calc.e_corr

    # Perturbative triples correction
    e_t = cc_calc.ccsd_t()
    e_ccsd_t = e_ccsd + e_t

    print(f"CCSD(T)/{basis}:")
    print(f"  HF:        {mf.e_tot:>18.10f} Eh")
    print(f"  CCSD:      {e_ccsd:>18.10f} Eh")
    print(f"  CCSD(T):   {e_ccsd_t:>18.10f} Eh")
    print(f"  (T) corr:  {e_t:>18.10f} Eh")

    return {
        "hf_energy":    mf.e_tot,
        "ccsd_energy":  e_ccsd,
        "ccsd_t_energy": e_ccsd_t,
        "t_correction": e_t,
    }


mol_water_tz = build_molecule(water_xyz, basis="cc-pvtz", verbose=0)
ccsd_t_res = run_ccsd_t(mol_water_tz, basis="cc-pvtz")
```

### Geometry Optimization with ASE

```python
from ase import Atoms
from ase.optimize import BFGS
from pyscf.geomopt.geometric_solver import optimize as pyscf_optimize


def optimize_geometry_ase(
    mol: gto.Mole,
    functional: str = "b3lyp",
    max_steps: int = 100,
    fmax: float = 0.05,
) -> dict:
    """
    Optimize molecular geometry using DFT forces via PySCF's geometric optimizer.

    Args:
        mol:        PySCF Mole object with initial geometry.
        functional: DFT functional for forces.
        max_steps:  Maximum optimization steps.
        fmax:       Convergence threshold for max force (eV/Ang).

    Returns:
        Dictionary with optimized_mol (new Mole), final_energy, and geometry.
    """
    mf = dft.RKS(mol)
    mf.xc = functional
    mf.grids.level = 3

    # Use PySCF's built-in geometric optimizer
    mol_opt = pyscf_optimize(mf, maxsteps=max_steps)

    mf_opt = dft.RKS(mol_opt)
    mf_opt.xc = functional
    mf_opt.kernel()

    # Extract optimized coordinates
    coords = mol_opt.atom_coords(unit="Angstrom")
    symbols = [mol_opt.atom_symbol(i) for i in range(mol_opt.natm)]

    geometry = pd.DataFrame({
        "atom": symbols,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })

    print(f"Optimization complete. Final energy: {mf_opt.e_tot:.6f} Eh")
    print(geometry.round(4).to_string(index=False))

    return {
        "optimized_mol": mol_opt,
        "final_energy":  mf_opt.e_tot,
        "geometry_df":   geometry,
    }


mol_opt_result = optimize_geometry_ase(mol_water, functional="b3lyp")
```

---

## Examples

### Example 1 — H2O HF/6-31G* Single Point + Mulliken Charges

```python
import pandas as pd
from pyscf import gto, scf

# Build molecule
mol = gto.Mole()
mol.atom = """
O  0.000  0.000  0.117
H  0.000  0.757 -0.471
H  0.000 -0.757 -0.471
"""
mol.basis   = "6-31g*"
mol.charge  = 0
mol.spin    = 0
mol.verbose = 4
mol.build()

# RHF
mf = scf.RHF(mol)
mf.conv_tol = 1e-10
mf.kernel()

print(f"\n--- Results ---")
print(f"Total energy:  {mf.e_tot:.8f} Eh  ({mf.e_tot * 627.509:.3f} kcal/mol)")
print(f"Converged:     {mf.converged}")

# Orbital energies in eV
mo_e = mf.mo_energy * 27.2114
n_occ = mol.nelectron // 2
print(f"HOMO ({n_occ}):   {mo_e[n_occ-1]:.3f} eV")
print(f"LUMO ({n_occ+1}): {mo_e[n_occ]:.3f} eV")
print(f"Gap:            {mo_e[n_occ] - mo_e[n_occ-1]:.3f} eV")

# Mulliken population
pop, chg = mf.mulliken_pop(verbose=0)
print("\nMulliken charges:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):3s}  {chg[i]:+.4f}")

# Dipole moment
dm = mf.dip_moment(unit="Debye", verbose=0)
print(f"\nDipole moment: {dm} Debye  |  |mu| = {np.linalg.norm(dm):.3f} D")
```

### Example 2 — B3LYP/cc-pVDZ Geometry Optimization with Energy Profile

```python
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, dft


def oh_stretch_potential(
    r_values: np.ndarray,
    basis: str = "cc-pvdz",
    functional: str = "b3lyp",
) -> np.ndarray:
    """
    Scan O-H bond length in water and compute DFT energy at each geometry.
    Illustrates a 1D potential energy surface (PES).

    Args:
        r_values:   Array of O-H distances in Angstrom.
        basis:      Basis set.
        functional: DFT functional.

    Returns:
        Array of DFT total energies in Hartree.
    """
    energies = []
    for r in r_values:
        mol = gto.Mole()
        # Water with one O-H bond stretched
        mol.atom = f"""
        O  0.0  0.0  0.0
        H  {r}  0.0  0.0
        H  -0.24 {r * 0.93 / 0.96}  0.0
        """
        mol.basis   = basis
        mol.charge  = 0
        mol.spin    = 0
        mol.verbose = 0
        mol.build()

        mf = dft.RKS(mol)
        mf.xc = functional
        mf.grids.level = 3
        e = mf.kernel()
        energies.append(e)
        print(f"  r={r:.2f} A: E={e:.6f} Eh")

    return np.array(energies)


r_vals = np.linspace(0.7, 2.0, 14)
energies_pes = oh_stretch_potential(r_vals, basis="cc-pvdz", functional="b3lyp")

# Plot PES
e_rel = (energies_pes - energies_pes.min()) * 627.509  # kcal/mol relative
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(r_vals, e_rel, "o-", color="#E74C3C", lw=2, ms=6)
ax.set_xlabel("O-H bond length (Angstrom)")
ax.set_ylabel("Relative energy (kcal/mol)")
ax.set_title("Water O-H Stretch PES — B3LYP/cc-pVDZ")
ax.axvline(r_vals[e_rel.argmin()], ls="--", color="gray", label="Minimum")
ax.legend()
fig.tight_layout()
fig.savefig("/tmp/pes_oh_stretch.png", dpi=150)
print(f"PES saved. Equilibrium r(O-H) ≈ {r_vals[e_rel.argmin()]:.2f} Angstrom")
```

### Example 3 — CCSD(T)/cc-pVTZ Thermochemistry Benchmark

```python
import numpy as np
from pyscf import gto, scf, cc


def atomization_energy_h2o(verbose: bool = True) -> dict:
    """
    Compute atomization energy of water: H2O -> 2H + O
    using CCSD(T)/cc-pVTZ and compare with experiment (917.8 kJ/mol).

    Returns:
        Dictionary with HF, CCSD, CCSD(T) atomization energies in kJ/mol.
    """
    hartree_to_kjmol = 2625.5

    def single_point_energy(atom_spec, basis, spin=0, charge=0):
        mol = gto.Mole()
        mol.atom   = atom_spec
        mol.basis  = basis
        mol.charge = charge
        mol.spin   = spin
        mol.verbose = 0
        mol.build()
        mf = scf.ROHF(mol) if spin > 0 else scf.RHF(mol)
        mf.kernel()
        cc_calc = cc.CCSD(mf)
        cc_calc.kernel()
        e_t = cc_calc.ccsd_t()
        return {
            "hf":     mf.e_tot,
            "ccsd":   mf.e_tot + cc_calc.e_corr,
            "ccsd_t": mf.e_tot + cc_calc.e_corr + e_t,
        }

    basis = "cc-pvtz"

    print("Computing H2O energy...")
    e_h2o = single_point_energy("""O 0 0 0.117; H 0 0.757 -0.471; H 0 -0.757 -0.471""",
                                 basis, spin=0)

    print("Computing H atom energy (doublet)...")
    e_h = single_point_energy("H 0 0 0", basis, spin=1)

    print("Computing O atom energy (triplet)...")
    e_o = single_point_energy("O 0 0 0", basis, spin=2)

    results = {}
    for method in ["hf", "ccsd", "ccsd_t"]:
        atomization = (2 * e_h[method] + e_o[method] - e_h2o[method]) * hartree_to_kjmol
        results[method] = atomization
        if verbose:
            print(f"{method.upper():8s}: atomization = {atomization:.1f} kJ/mol")

    print(f"\nExperimental reference: 917.8 kJ/mol")
    print(f"CCSD(T) error: {results['ccsd_t'] - 917.8:.1f} kJ/mol")

    return results


thermo = atomization_energy_h2o()
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `SCF not converged` | Difficult electronic structure | Increase `max_cycle`; use level-shifting (`mf.level_shift=0.2`) |
| `numpy.linalg.LinAlgError` | Near-linear-dependent basis | Use smaller basis or `mol.lindep_threshold=1e-8` |
| Memory error in CCSD | N^6 memory scaling | Reduce basis to cc-pVDZ; freeze more core orbitals |
| Wrong spin state | Incorrect `mol.spin` | Set `spin=2S` (not 2S+1); check `mol.nelectron % 2 == mol.spin % 2` |
| Geometry optimization diverges | Bad initial geometry | Pre-optimize with a lower level (HF/STO-3G) first |
| `KeyError` in functional | Unsupported XC name | Check PySCF functional list: `pyscf.dft.libxc.XC_CODES` |
| MP2 gives positive correlation | Wrong frozen orbital indices | Verify frozen list; correlation energy should always be negative |
| DFT grid error for anions | Insufficient diffuse functions | Use `aug-cc-pVDZ` or add diffuse functions via `mol.basis` dict |

---

## External Resources

- PySCF documentation: <https://pyscf.org/user_guide.html>
- PySCF GitHub: <https://github.com/pyscf/pyscf>
- Basis set exchange: <https://www.basissetexchange.org/>
- NIST CCCBDB (benchmark energies): <https://cccbdb.nist.gov/>
- Cramer, C.J. *Essentials of Computational Chemistry*, 2nd ed. (2004)
- Szabo & Ostlund, *Modern Quantum Chemistry* (1989)
- PySCF paper: Sun et al. (2020) *J. Chem. Phys.* 153, 024109

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — HF, DFT, MP2, CCSD(T), Mulliken charges, PES scan |
