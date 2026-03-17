---
name: ase-atomistic
description: >
  Atomic Simulation Environment (ASE) for atomistic simulations: structure
  building, geometry optimization, NEB, molecular dynamics, and trajectory analysis.
tags:
  - ase
  - atomistic-simulation
  - chemistry
  - molecular-dynamics
  - dft
  - computational-chemistry
version: "1.0.0"
authors:
  - name: "awesome-rosetta-skills contributors"
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - ase>=3.22.0
  - numpy>=1.24.0
  - scipy>=1.11.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# ASE — Atomic Simulation Environment

The Atomic Simulation Environment (ASE) is the de-facto Python toolkit for setting up,
running, and analyzing atomistic simulations. This skill covers structure building,
geometry optimization, transition state searches, molecular dynamics, and trajectory
analysis — with the EMT empirical potential as a dependency-free calculator.

---

## 1. Building Atomic Structures

ASE provides factory functions for common structure types.

```python
from ase import Atoms
from ase.build import bulk, surface, molecule, fcc111, add_adsorbate
from ase.visualize import view  # requires ASE GUI or nglview in Jupyter
import numpy as np


def build_structures_demo():
    """
    Demonstrate building bulk, surface, molecule, and nanoparticle structures.
    """
    # --- Bulk FCC copper ---
    cu_bulk = bulk("Cu", crystalstructure="fcc", a=3.615, cubic=True)
    print(f"Cu bulk: {len(cu_bulk)} atoms, cell = {cu_bulk.cell.diagonal().round(3)}")

    # --- FCC(111) surface slab ---
    def build_fcc_slab(element, miller, layers, vacuum=10.0, supercell=(2, 2)):
        """
        Build an FCC surface slab.

        Parameters
        ----------
        element : str
            Chemical symbol, e.g. 'Pt', 'Cu', 'Au'.
        miller : tuple
            Miller index, e.g. (1, 1, 1).
        layers : int
            Number of atomic layers.
        vacuum : float
            Vacuum thickness in Angstrom (added on both sides).
        supercell : tuple
            (nx, ny) supercell repetitions.

        Returns
        -------
        slab : ase.Atoms
            Surface slab with vacuum.
        """
        from ase.build import fcc111, fcc110, fcc100
        miller_map = {
            (1, 1, 1): fcc111,
            (1, 1, 0): fcc110,
            (1, 0, 0): fcc100,
        }
        builder = miller_map.get(miller, fcc111)
        slab = builder(element, size=(supercell[0], supercell[1], layers), vacuum=vacuum)
        return slab

    pt_slab = build_fcc_slab("Pt", (1, 1, 1), layers=4, vacuum=12.0, supercell=(3, 3))
    print(f"Pt(111) slab: {len(pt_slab)} atoms")
    print(f"  Cell: {pt_slab.cell.diagonal().round(3)} Å")
    print(f"  Species: {set(pt_slab.get_chemical_symbols())}")

    # --- Small molecules ---
    co = molecule("CO")
    h2o = molecule("H2O")
    print(f"\nCO bond length: {co.get_distance(0, 1):.4f} Å")
    print(f"H2O O-H bond:  {h2o.get_distance(0, 1):.4f} Å")
    print(f"H2O H-O-H angle: {h2o.get_angle(1, 0, 2):.2f}°")

    # --- Custom Atoms object ---
    h2 = Atoms(
        symbols="HH",
        positions=[[0, 0, 0], [0, 0, 0.74]],
        cell=[10, 10, 10],
        pbc=False,
    )
    print(f"\nH2 molecule: bond = {h2.get_distance(0, 1):.4f} Å")

    # --- Nanoparticle (FCC icosahedron) ---
    from ase.cluster import Icosahedron
    np_Au = Icosahedron("Au", noshells=3)
    print(f"\nAu nanoparticle: {len(np_Au)} atoms (Icosahedron, 3 shells)")

    return pt_slab, co, h2


if __name__ == "__main__":
    slab, co, h2 = build_structures_demo()
```

---

## 2. Reading and Writing Structure Files

```python
from ase.io import read, write
from ase.build import bulk, molecule
import os


def io_demo(output_dir="/tmp/ase_io_demo"):
    """
    Demonstrate reading and writing structure files in various formats.

    Supported formats: CIF, POSCAR/CONTCAR, XYZ, extXYZ, cube, cif, pdb, ...
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build a test structure
    cu = bulk("Cu", "fcc", a=3.615)
    h2o = molecule("H2O")

    # --- Write formats ---
    write(f"{output_dir}/cu_bulk.vasp", cu, format="vasp")          # POSCAR
    write(f"{output_dir}/cu_bulk.cif", cu, format="cif")            # CIF
    write(f"{output_dir}/h2o.xyz", h2o, format="xyz")               # XYZ
    write(f"{output_dir}/h2o_ext.xyz", h2o, format="extxyz")        # Extended XYZ

    print(f"Wrote files to {output_dir}/")
    for fname in os.listdir(output_dir):
        fpath = f"{output_dir}/{fname}"
        atoms = read(fpath)
        print(f"  {fname:25s}: {len(atoms):3d} atoms, formula={atoms.get_chemical_formula()}")

    # --- Read a trajectory (multiple frames) ---
    from ase.io.trajectory import Trajectory
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS

    traj_path = f"{output_dir}/test_traj.traj"
    atoms = h2o.copy()
    atoms.calc = EMT()

    with Trajectory(traj_path, "w", atoms) as traj:
        # Perturb and write a few frames
        rng = np.random.default_rng(0)
        for i in range(5):
            atoms.positions += 0.05 * rng.standard_normal(atoms.positions.shape)
            atoms.get_potential_energy()  # trigger calc
            traj.write(atoms)

    # Read back all frames
    frames = read(traj_path, index=":")
    print(f"\nTrajectory: {len(frames)} frames, {len(frames[0])} atoms per frame")
    energies = [f.get_potential_energy() for f in frames]
    print(f"Energies: {[f'{e:.4f}' for e in energies]}")

    return output_dir


import numpy as np
if __name__ == "__main__":
    io_demo()
```

---

## 3. Geometry Optimization

```python
from ase.calculators.emt import EMT
from ase.optimize import BFGS, LBFGS, BFGSLineSearch
from ase.build import molecule
from ase.constraints import FixAtoms
import numpy as np


def optimize_geometry(atoms, calculator=None, fmax=0.01, optimizer="BFGS",
                      max_steps=500, trajectory_file=None):
    """
    Optimize atomic positions to minimize forces.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to optimize.
    calculator : ase Calculator or None
        If None, uses EMT (works for metals).
    fmax : float
        Convergence criterion: max force component in eV/Å.
    optimizer : str
        'BFGS', 'LBFGS', or 'BFGSLineSearch'.
    max_steps : int
        Maximum optimization steps.
    trajectory_file : str or None
        Path to save trajectory; None = no save.

    Returns
    -------
    atoms : ase.Atoms (in-place modified)
    converged : bool
    """
    if calculator is None:
        calculator = EMT()

    atoms = atoms.copy()
    atoms.calc = calculator

    optimizer_map = {"BFGS": BFGS, "LBFGS": LBFGS, "BFGSLineSearch": BFGSLineSearch}
    Opt = optimizer_map.get(optimizer, BFGS)

    opt = Opt(atoms, logfile=None, trajectory=trajectory_file)
    converged = opt.run(fmax=fmax, steps=max_steps)

    e_final = atoms.get_potential_energy()
    f_max = np.abs(atoms.get_forces()).max()
    print(f"Geometry optimization ({optimizer}):")
    print(f"  Converged: {converged}")
    print(f"  Steps:     {opt.nsteps}")
    print(f"  Energy:    {e_final:.6f} eV")
    print(f"  Max force: {f_max:.6f} eV/Å")

    return atoms, converged


def optimize_molecule_demo():
    """Optimize H2O geometry and compute properties."""
    h2o = molecule("H2O")
    h2o.calc = EMT()

    print(f"Before optimization:")
    print(f"  Energy:    {h2o.get_potential_energy():.6f} eV")
    print(f"  Max force: {np.abs(h2o.get_forces()).max():.6f} eV/Å")

    h2o_opt, converged = optimize_geometry(h2o, fmax=0.001)

    print(f"\nAfter optimization:")
    print(f"  O-H bond 1: {h2o_opt.get_distance(0, 1):.4f} Å")
    print(f"  O-H bond 2: {h2o_opt.get_distance(0, 2):.4f} Å")
    print(f"  H-O-H angle: {h2o_opt.get_angle(1, 0, 2):.2f}°")
    return h2o_opt


def optimize_slab_with_constraints():
    """Optimize a surface slab with bottom layers fixed."""
    from ase.build import fcc111

    slab = fcc111("Pt", size=(2, 2, 4), vacuum=10.0)
    slab.calc = EMT()

    # Fix bottom 2 layers
    n_bottom = len(slab) // 2
    constraint = FixAtoms(indices=list(range(n_bottom)))
    slab.set_constraint(constraint)

    print(f"\nSlab optimization with FixAtoms constraint:")
    print(f"  {n_bottom} atoms fixed, {len(slab) - n_bottom} atoms free")

    slab_opt, _ = optimize_geometry(slab, fmax=0.05, optimizer="LBFGS")
    return slab_opt


if __name__ == "__main__":
    h2o_opt = optimize_molecule_demo()
    slab_opt = optimize_slab_with_constraints()
```

---

## 4. Nudged Elastic Band (NEB) for Transition States

```python
import numpy as np
from ase.calculators.emt import EMT
from ase.build import fcc111, add_adsorbate, molecule
from ase.optimize import BFGS
from ase.neb import NEB
from ase.constraints import FixAtoms
import matplotlib.pyplot as plt


def run_neb(initial, final, n_images=7, calculator_class=EMT, fmax=0.05):
    """
    Run a Nudged Elastic Band calculation to find the minimum energy path.

    Parameters
    ----------
    initial : ase.Atoms
        Initial state (relaxed).
    final : ase.Atoms
        Final state (relaxed).
    n_images : int
        Number of intermediate images (not counting endpoints).
    calculator_class : callable
        ASE calculator factory, e.g. EMT.
    fmax : float
        Force convergence threshold in eV/Å.

    Returns
    -------
    images : list of ase.Atoms
        All NEB images (initial + intermediate + final).
    neb : NEB object
    """
    # Build intermediate images by interpolation
    images = [initial.copy()]
    for _ in range(n_images):
        images.append(initial.copy())
    images.append(final.copy())

    # Assign calculator to each intermediate image
    for image in images[1:-1]:
        image.calc = calculator_class()

    # Linear interpolation
    neb = NEB(images, climb=True, k=0.1)
    neb.interpolate(method="idpp")  # IDPP is smoother than linear for molecules

    # Optimize
    opt = BFGS(neb, logfile=None)
    opt.run(fmax=fmax, steps=300)

    print(f"NEB optimization steps: {opt.nsteps}")
    return images, neb


def plot_neb_path(images):
    """Plot the NEB energy profile (MEP)."""
    energies = [img.get_potential_energy() for img in images]
    e_ref = energies[0]
    reaction_coord = np.linspace(0, 1, len(energies))

    activation_barrier = max(energies) - e_ref
    reaction_energy = energies[-1] - e_ref

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(reaction_coord, [e - e_ref for e in energies], "o-", color="steelblue",
            lw=2, ms=8, label="NEB images")
    ax.fill_between(reaction_coord, [e - e_ref for e in energies], alpha=0.15, color="steelblue")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(activation_barrier, color="red", ls=":", lw=1.5,
               label=f"Barrier = {activation_barrier:.3f} eV")
    ax.set_xlabel("Reaction Coordinate")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("NEB Minimum Energy Path")
    ax.legend()

    print(f"\nNEB results:")
    print(f"  Activation barrier: {activation_barrier:.4f} eV")
    print(f"  Reaction energy:    {reaction_energy:.4f} eV")

    plt.tight_layout()
    plt.show()
    return activation_barrier, reaction_energy


def neb_demo():
    """
    Demonstrate NEB for H2 dissociation on a flat potential (EMT demo).
    Uses a simple displacement as the reaction coordinate.
    """
    from ase import Atoms

    # Initial: H2 molecule (bound)
    initial = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    initial.calc = EMT()
    opt = BFGS(initial, logfile=None)
    opt.run(fmax=0.01)

    # Final: H atoms separated
    final = Atoms("H2", positions=[[0, 0, 0], [0, 0, 3.0]])
    final.calc = EMT()
    opt_f = BFGS(final, logfile=None)
    opt_f.run(fmax=0.01)

    print("H2 dissociation NEB:")
    print(f"  Initial energy: {initial.get_potential_energy():.4f} eV")
    print(f"  Final energy:   {final.get_potential_energy():.4f} eV")

    images, neb_obj = run_neb(initial, final, n_images=5, fmax=0.05)
    barrier, delta_e = plot_neb_path(images)
    return images


if __name__ == "__main__":
    neb_demo()
```

---

## 5. Molecular Dynamics

```python
import numpy as np
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
import matplotlib.pyplot as plt


def run_nvt_md(atoms, T=300, steps=1000, dt=1.0, friction=0.01,
               trajectory_file=None, log_interval=10):
    """
    Run NVT (constant temperature) molecular dynamics using Langevin thermostat.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure with calculator already set.
    T : float
        Target temperature in Kelvin.
    steps : int
        Total MD steps.
    dt : float
        Timestep in femtoseconds.
    friction : float
        Langevin friction coefficient (1/fs).
    trajectory_file : str or None
        Save trajectory to this file.
    log_interval : int
        Log energy/temperature every N steps.

    Returns
    -------
    log : dict
        Dictionary with 'time', 'temperature', 'potential_energy', 'kinetic_energy'.
    """
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    dyn = Langevin(
        atoms,
        timestep=dt * units.fs,
        temperature_K=T,
        friction=friction / units.fs,
        logfile=None,
    )

    log = {"time": [], "temperature": [], "potential_energy": [], "kinetic_energy": []}

    if trajectory_file:
        traj = Trajectory(trajectory_file, "w", atoms)
        dyn.attach(traj.write, interval=log_interval)

    def record():
        log["time"].append(dyn.get_time() / units.fs)
        log["temperature"].append(atoms.get_temperature())
        log["potential_energy"].append(atoms.get_potential_energy())
        log["kinetic_energy"].append(atoms.get_kinetic_energy())

    dyn.attach(record, interval=log_interval)
    dyn.run(steps)

    if trajectory_file:
        traj.close()

    print(f"\nNVT MD completed:")
    print(f"  Steps: {steps}, dt = {dt} fs, T_target = {T} K")
    print(f"  T_mean = {np.mean(log['temperature']):.1f} K")
    print(f"  T_std  = {np.std(log['temperature']):.2f} K")
    return log


def run_nve_md(atoms, steps=500, dt=1.0, trajectory_file=None):
    """
    Run NVE (microcanonical) MD using Velocity Verlet.
    Energy should be approximately conserved.
    """
    dyn = VelocityVerlet(atoms, timestep=dt * units.fs, logfile=None)

    log = {"time": [], "total_energy": [], "temperature": []}

    def record():
        log["time"].append(dyn.get_time() / units.fs)
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        log["total_energy"].append(e_pot + e_kin)
        log["temperature"].append(atoms.get_temperature())

    dyn.attach(record, interval=5)
    dyn.run(steps)

    e_total = np.array(log["total_energy"])
    drift = (e_total[-1] - e_total[0]) / abs(e_total[0]) * 100
    print(f"\nNVE MD: total energy drift = {drift:.4f}%")
    return log


def md_demo():
    """Run NVT MD on bulk Cu and plot thermodynamic observables."""
    cu = bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 2, 2)
    cu.calc = EMT()

    # Equilibrate at 500K
    MaxwellBoltzmannDistribution(cu, temperature_K=500)
    log = run_nvt_md(cu, T=500, steps=2000, dt=2.0, friction=0.02, log_interval=20)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(log["time"], log["temperature"], color="tomato", lw=0.8)
    axes[0].axhline(500, color="black", ls="--", lw=1, label="T_target = 500 K")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].legend()
    axes[0].set_title("NVT MD — Bulk Cu at 500 K")

    e_pot = np.array(log["potential_energy"])
    e_kin = np.array(log["kinetic_energy"])
    axes[1].plot(log["time"], e_pot, label="Potential", color="steelblue", lw=0.8)
    axes[1].plot(log["time"], e_kin, label="Kinetic", color="darkorange", lw=0.8)
    axes[1].plot(log["time"], e_pot + e_kin, label="Total", color="black", lw=1.2, ls="--")
    axes[1].set_xlabel("Time (fs)")
    axes[1].set_ylabel("Energy (eV)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return log


if __name__ == "__main__":
    md_demo()
```

---

## 6. Trajectory Analysis: RDF, MSD, RMSD

```python
import numpy as np
from ase.io import read
from ase.geometry import get_distances
import matplotlib.pyplot as plt


def compute_rdf(trajectory, species_pair=None, rmax=6.0, nbins=100):
    """
    Compute the radial distribution function g(r) from a trajectory.

    Parameters
    ----------
    trajectory : list of ase.Atoms
        MD frames.
    species_pair : tuple of str or None
        e.g. ('Cu', 'Cu'). If None, uses all pairs.
    rmax : float
        Maximum distance in Å.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    r_centers : ndarray
        Bin center distances.
    g_r : ndarray
        RDF values.
    """
    dr = rmax / nbins
    r_edges = np.linspace(0, rmax, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(nbins)

    n_frames = len(trajectory)
    for atoms in trajectory:
        symbols = np.array(atoms.get_chemical_symbols())
        N = len(atoms)

        if species_pair is not None:
            s1, s2 = species_pair
            idx1 = np.where(symbols == s1)[0]
            idx2 = np.where(symbols == s2)[0]
        else:
            idx1 = idx2 = np.arange(N)

        for i in idx1:
            for j in idx2:
                if i >= j:
                    continue
                r = atoms.get_distance(i, j, mic=True)
                if r < rmax:
                    bin_idx = int(r / dr)
                    if bin_idx < nbins:
                        hist[bin_idx] += 2  # count both i->j and j->i

    # Normalize to ideal gas
    n_frames_norm = n_frames
    cell_volume = np.mean([atoms.get_volume() for atoms in trajectory])
    N_avg = np.mean([len(a) for a in trajectory])
    rho = N_avg / cell_volume  # number density

    for k, (r_lo, r_hi) in enumerate(zip(r_edges[:-1], r_edges[1:])):
        shell_vol = 4 * np.pi / 3 * (r_hi**3 - r_lo**3)
        norm = rho * shell_vol * N_avg * n_frames_norm
        g_r_k = hist[k] / (norm + 1e-12)
        hist[k] = g_r_k

    return r_centers, hist


def compute_msd(trajectory, species=None):
    """
    Compute the Mean Squared Displacement as a function of time lag.

    Parameters
    ----------
    trajectory : list of ase.Atoms
        MD frames (equal time spacing assumed).
    species : str or None
        If given, compute MSD only for this element.

    Returns
    -------
    tau : ndarray
        Time lags (in frame units).
    msd : ndarray
        MSD in Å².
    """
    n_frames = len(trajectory)
    if species is not None:
        ref_atoms = trajectory[0]
        indices = [i for i, s in enumerate(ref_atoms.get_chemical_symbols()) if s == species]
    else:
        indices = list(range(len(trajectory[0])))

    positions = np.array([atoms.positions[indices] for atoms in trajectory])

    max_lag = n_frames // 2
    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    for lag in range(1, max_lag):
        disp = positions[lag:] - positions[:n_frames - lag]
        msd[lag] = np.mean(np.sum(disp**2, axis=-1))
        counts[lag] = n_frames - lag

    return np.arange(max_lag), msd


def compute_rmsd(trajectory, reference=None):
    """
    Compute RMSD of each frame relative to a reference structure.

    Returns rmsd array in Å.
    """
    if reference is None:
        reference = trajectory[0]
    ref_pos = reference.positions

    rmsd_values = []
    for atoms in trajectory:
        disp = atoms.positions - ref_pos
        rmsd = np.sqrt(np.mean(np.sum(disp**2, axis=-1)))
        rmsd_values.append(rmsd)

    return np.array(rmsd_values)


def trajectory_analysis_demo():
    """Generate an MD trajectory and perform full analysis."""
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    # Build system and run MD
    cu = bulk("Cu", "fcc", a=3.615, cubic=True) * (3, 3, 3)
    cu.calc = EMT()
    MaxwellBoltzmannDistribution(cu, temperature_K=800)

    dyn = Langevin(cu, timestep=2.0 * units.fs, temperature_K=800,
                   friction=0.01 / units.fs, logfile=None)

    trajectory = []

    def save_frame():
        trajectory.append(cu.copy())
        trajectory[-1].calc = None  # don't keep calculator in trajectory

    dyn.attach(save_frame, interval=5)
    dyn.run(2000)

    print(f"Collected {len(trajectory)} trajectory frames")

    # Compute RDF
    r, g_r = compute_rdf(trajectory[-50:], species_pair=("Cu", "Cu"), rmax=8.0, nbins=80)

    # Compute MSD
    tau, msd = compute_msd(trajectory, species="Cu")

    # Compute RMSD
    rmsd_vals = compute_rmsd(trajectory)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(r, g_r, color="steelblue", lw=1.5)
    axes[0].axhline(1, color="gray", ls="--", lw=0.8, label="Ideal gas limit")
    axes[0].set_xlabel("r (Å)")
    axes[0].set_ylabel("g(r)")
    axes[0].set_title("Radial Distribution Function — Cu")
    axes[0].set_xlim(0, 8)
    axes[0].legend()

    valid = tau[1:] > 0
    axes[1].plot(tau[1:][valid], msd[1:][valid], color="darkorange", lw=1.5)
    axes[1].set_xlabel("Time lag (frames × 10 fs)")
    axes[1].set_ylabel("MSD (Å²)")
    axes[1].set_title("Mean Squared Displacement")

    axes[2].plot(np.arange(len(rmsd_vals)) * 10, rmsd_vals, color="tomato", lw=1)
    axes[2].set_xlabel("Time (fs)")
    axes[2].set_ylabel("RMSD (Å)")
    axes[2].set_title("RMSD from Initial Structure")

    plt.tight_layout()
    plt.show()

    return trajectory, r, g_r, msd


if __name__ == "__main__":
    trajectory_analysis_demo()
```

---

## 7. Complete Example A — CO Adsorption on Pt(111)

Compute the adsorption energy of CO on a Pt(111) surface using EMT.

```python
import numpy as np
from ase.build import fcc111, molecule, add_adsorbate
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
import matplotlib.pyplot as plt


def co_adsorption_on_pt111(layers=4, vacuum=12.0, supercell=(3, 3)):
    """
    Calculate the adsorption energy of CO on Pt(111).

    E_ads = E(slab+CO) - E(slab) - E(CO)

    Positive E_ads means adsorption is favorable (exothermic).

    Adsorption sites tested: top, bridge, hollow-fcc, hollow-hcp
    """

    def build_clean_slab():
        slab = fcc111("Pt", size=(supercell[0], supercell[1], layers), vacuum=vacuum)
        # Fix bottom half of slab
        n_fix = len(slab) // 2
        slab.set_constraint(FixAtoms(indices=list(range(n_fix))))
        return slab

    def relax(atoms, fmax=0.05, label=""):
        atoms.calc = EMT()
        opt = LBFGS(atoms, logfile=None)
        opt.run(fmax=fmax, steps=300)
        e = atoms.get_potential_energy()
        print(f"  {label:25s}: E = {e:.4f} eV, steps = {opt.nsteps}")
        return atoms, e

    # 1. Relax clean slab
    print("Relaxing clean Pt(111) slab...")
    slab_clean, e_slab = relax(build_clean_slab(), label="Clean Pt(111)")

    # 2. Relax isolated CO molecule
    print("Relaxing CO molecule...")
    co = molecule("CO")
    co.cell = [15, 15, 15]
    co.center()
    co.pbc = True
    co_relaxed, e_co = relax(co, label="CO molecule")
    print(f"  CO bond length: {co_relaxed.get_distance(0, 1):.4f} Å")

    # 3. Test adsorption on top site
    adsorption_results = {}
    sites = {
        "top":     ("ontop",   (0, 0)),
        "bridge":  ("bridge",  (0, 0)),
        "fcc":     ("fcc",     (0, 0)),
        "hcp":     ("hcp",     (0, 0)),
    }

    print("\nCalculating adsorption energies on different sites:")
    for site_name, (site_type, offset) in sites.items():
        slab_co = build_clean_slab()
        slab_co.calc = EMT()

        co_mol = molecule("CO")
        try:
            add_adsorbate(slab_co, co_mol, height=1.8, position=site_type)
        except Exception:
            # Fallback: add CO manually on top of a surface atom
            surface_layer = [a for a in slab_co if a.tag == 1]
            if surface_layer:
                top_pos = slab_co[surface_layer[0].index].position[:2]
                add_adsorbate(slab_co, co_mol, height=1.8, position=top_pos)
            else:
                add_adsorbate(slab_co, co_mol, height=1.8)

        # Fix bottom layers
        n_fix = (len(slab_co) - len(co_mol)) // 2
        slab_co.set_constraint(FixAtoms(indices=list(range(n_fix))))

        slab_co_relaxed, e_slab_co = relax(slab_co, label=f"Pt(111)+CO/{site_name}")
        e_ads = e_slab + e_co - e_slab_co
        adsorption_results[site_name] = {
            "e_ads": e_ads,
            "e_total": e_slab_co,
        }

    print("\n" + "=" * 45)
    print("Adsorption Energies Summary:")
    print("=" * 45)
    for site, data in adsorption_results.items():
        print(f"  {site:8s}: E_ads = {data['e_ads']:+.4f} eV")

    best_site = max(adsorption_results, key=lambda s: adsorption_results[s]["e_ads"])
    print(f"\n  Most stable site: {best_site}")

    # Plot
    sites_list = list(adsorption_results.keys())
    e_ads_list = [adsorption_results[s]["e_ads"] for s in sites_list]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(sites_list, e_ads_list,
                  color=["green" if e > 0 else "tomato" for e in e_ads_list], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("E_ads (eV)")
    ax.set_title("CO Adsorption Energy on Pt(111) Sites")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()
    plt.show()

    return adsorption_results


if __name__ == "__main__":
    results = co_adsorption_on_pt111()
```

---

## 8. Complete Example B — Geometry Optimization and Vibrational Analysis

```python
import numpy as np
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
import matplotlib.pyplot as plt


def geometry_and_phonons(mol_name="H2O", outdir="/tmp/ase_vib"):
    """
    (b) Full workflow:
    1. Optimize geometry of a molecule
    2. Compute vibrational modes (numerical Hessian)
    3. Compute infrared intensities (if dipoles available)
    4. Plot vibrational spectrum

    Parameters
    ----------
    mol_name : str
        ASE molecule name, e.g. 'H2O', 'NH3', 'CH4'.
    outdir : str
        Directory to store vibrational calculation files.
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    # 1. Build and optimize
    atoms = molecule(mol_name)
    atoms.cell = [15, 15, 15]
    atoms.center()
    atoms.pbc = False
    atoms.calc = EMT()

    print(f"{'='*50}")
    print(f"Molecule: {mol_name}, N_atoms = {len(atoms)}")
    print(f"{'='*50}")

    # Pre-optimization energy
    e0 = atoms.get_potential_energy()
    f0_max = np.abs(atoms.get_forces()).max()
    print(f"Initial: E = {e0:.4f} eV, |F|_max = {f0_max:.4f} eV/Å")

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.001, steps=1000)

    e_opt = atoms.get_potential_energy()
    f_opt_max = np.abs(atoms.get_forces()).max()
    print(f"Optimized: E = {e_opt:.4f} eV, |F|_max = {f_opt_max:.6f} eV/Å")
    print(f"Relaxation: ΔE = {e_opt - e0:.4f} eV, steps = {opt.nsteps}")

    # Print bond lengths and angles
    print("\nGeometry:")
    symbols = atoms.get_chemical_symbols()
    if mol_name == "H2O":
        print(f"  O-H1: {atoms.get_distance(0, 1):.4f} Å")
        print(f"  O-H2: {atoms.get_distance(0, 2):.4f} Å")
        print(f"  H-O-H: {atoms.get_angle(1, 0, 2):.2f}°")
    elif mol_name == "NH3":
        for i in range(1, 4):
            print(f"  N-H{i}: {atoms.get_distance(0, i):.4f} Å")

    # 2. Vibrational analysis
    print(f"\nComputing vibrational modes (numerical Hessian)...")
    vib = Vibrations(atoms, name=f"{outdir}/{mol_name}_vib", delta=0.01)
    vib.run()
    vib.summary(log="-")  # print summary to stdout

    # Get vibrational energies
    vib_energies = vib.get_energies()
    real_modes = [e for e in vib_energies if e.real > 1e-3 and abs(e.imag) < 1e-3]
    imaginary_modes = [e for e in vib_energies if abs(e.imag) > 1.0]

    print(f"\nVibrational frequencies (real modes):")
    for i, e in enumerate(real_modes):
        freq_cm1 = e.real * 8065.5  # eV → cm^-1
        freq_thz = e.real * 241.8   # eV → THz
        print(f"  Mode {i+1}: {freq_cm1:.1f} cm⁻¹  ({freq_thz:.2f} THz)")

    if imaginary_modes:
        print(f"  WARNING: {len(imaginary_modes)} imaginary mode(s) found.")

    # Zero-point energy
    zpe = vib.get_zero_point_energy()
    print(f"\nZero-point energy: {zpe:.4f} eV")

    # 3. Plot vibrational spectrum (Gaussian broadening)
    freq_range = np.linspace(0, 5000, 2000)  # cm^-1
    spectrum = np.zeros_like(freq_range)
    sigma = 30.0  # broadening in cm^-1

    for e in real_modes:
        freq_cm1 = e.real * 8065.5
        spectrum += np.exp(-0.5 * ((freq_range - freq_cm1) / sigma) ** 2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Stick spectrum
    for e in real_modes:
        freq_cm1 = e.real * 8065.5
        axes[0].axvline(freq_cm1, ymin=0, ymax=1, color="steelblue", lw=2, alpha=0.8)
    axes[0].set_xlim(0, 5000)
    axes[0].set_ylabel("Intensity (arb.)")
    axes[0].set_title(f"{mol_name} Vibrational Frequencies (Stick Spectrum)")
    axes[0].set_xlabel("Wavenumber (cm⁻¹)")

    # Broadened spectrum
    axes[1].plot(freq_range, spectrum, color="darkorange", lw=1.5)
    axes[1].fill_between(freq_range, spectrum, alpha=0.3, color="darkorange")
    axes[1].set_xlim(0, 5000)
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].set_ylabel("Intensity (arb.)")
    axes[1].set_title(f"{mol_name} IR Spectrum (σ={sigma} cm⁻¹)")

    plt.tight_layout()
    plt.show()

    # Cleanup
    vib.clean()

    return atoms, vib_energies


if __name__ == "__main__":
    atoms_h2o, vib_h2o = geometry_and_phonons("H2O")
    atoms_nh3, vib_nh3 = geometry_and_phonons("NH3")
```

---

## Quick Reference

| Task                                | ASE API                                                    |
|-------------------------------------|------------------------------------------------------------|
| Build FCC bulk                      | `bulk("Cu", "fcc", a=3.615)`                               |
| Build surface slab                  | `fcc111("Pt", size=(3,3,4), vacuum=10.0)`                  |
| Build molecule                      | `molecule("H2O")`                                          |
| Add adsorbate                       | `add_adsorbate(slab, mol, height=1.8)`                     |
| Fix atoms (constraint)              | `FixAtoms(indices=[...])`                                  |
| Read structure file                 | `read("file.cif")`                                         |
| Write structure file                | `write("out.vasp", atoms, format="vasp")`                  |
| Set EMT calculator                  | `atoms.calc = EMT()`                                       |
| Get potential energy                | `atoms.get_potential_energy()`                             |
| Get forces                          | `atoms.get_forces()`                                       |
| Optimize geometry (BFGS)            | `BFGS(atoms); opt.run(fmax=0.01)`                          |
| Optimize geometry (LBFGS)           | `LBFGS(atoms); opt.run(fmax=0.05)`                         |
| NVT Langevin MD                     | `Langevin(atoms, dt*units.fs, T, friction/units.fs)`       |
| NVE Verlet MD                       | `VelocityVerlet(atoms, dt*units.fs)`                       |
| Maxwell-Boltzmann velocities        | `MaxwellBoltzmannDistribution(atoms, temperature_K=T)`     |
| NEB transition state                | `NEB(images, climb=True); BFGS(neb).run(fmax=0.05)`        |
| Vibrational analysis                | `Vibrations(atoms); vib.run(); vib.summary()`              |
| Get bond distance                   | `atoms.get_distance(i, j, mic=True)`                       |
| Get bond angle                      | `atoms.get_angle(i, j, k)`                                 |

### Calculator Hierarchy

| Calculator    | Use case                          | Availability     |
|---------------|-----------------------------------|------------------|
| EMT           | FCC metals (quick testing)        | Built into ASE   |
| GPAW          | DFT, PAW method                   | `pip install gpaw` |
| VASP          | DFT, plane waves                  | License required |
| CP2K          | DFT, mixed Gaussian/PW            | Free, open source|
| LAMMPS        | Classical MD, many potentials     | Free, open source|
| Psi4 / ORCA   | Quantum chemistry                 | Free/academic    |
