---
name: pymatgen-materials
description: >
  Use this Skill for materials science with pymatgen: crystal structure manipulation,
  Materials Project API queries (bandgap, DOS, formation energy), phase diagrams,
  and symmetry analysis.
tags:
  - chemistry
  - materials-science
  - pymatgen
  - crystal-structure
  - Materials-Project
  - DFT
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
    - pymatgen>=2023.9
    - mp-api>=0.39
    - numpy>=1.23
    - matplotlib>=3.6
last_updated: "2026-03-17"
status: stable
---

# Pymatgen Materials Science

> **TL;DR** — Manipulate crystal structures, query the Materials Project database for
> band gaps, DOS, and formation energies, construct phase diagrams, and perform
> space group symmetry analysis with pymatgen.

---

## When to Use

Use this Skill when you need to:

- Load and manipulate crystal structures from CIF, POSCAR (VASP), or XYZ files
- Query Materials Project for DFT-computed properties: band gap, DOS, formation energy,
  stability (e_above_hull)
- Construct binary or ternary phase diagrams and identify stable phases
- Perform symmetry analysis: detect space group, find equivalent sites, get conventional cell
- Build supercells, substitute atoms, or apply structure transformations
- Visualize electronic band structure and density of states (DOS)
- Calculate Pourbaix diagrams for electrochemical stability

| Task | Key pymatgen Class |
|---|---|
| Crystal structure | `Structure`, `Lattice` |
| File I/O | `CifParser`, `Poscar`, `XYZ` |
| API query | `MPRester` (mp-api) |
| Phase diagram | `PhaseDiagram`, `PDPlotter` |
| Symmetry | `SpacegroupAnalyzer` |
| Band structure | `BSPlotter` |
| Pourbaix | `PourbaixDiagram` |

---

## Background & Key Concepts

### The Structure Object

`Structure` is the central pymatgen object. It contains:
- `Lattice`: three lattice vectors defining the unit cell
- `Species`: list of element symbols or Species with oxidation states
- `frac_coords`: fractional coordinates of all sites

### Materials Project API

The Materials Project (<https://materialsproject.org>) provides DFT-computed properties
for ~150,000 inorganic compounds. Access requires a free API key. The new API uses
`mp-api` package with `MPRester`.

```bash
# Get your API key at https://materialsproject.org/api
export MP_API_KEY="<paste-your-key>"
```

### Phase Diagrams

A phase diagram maps thermodynamic stability in composition space. The `PhaseDiagram`
class constructs the convex hull of formation energies. Points on the hull are stable;
`e_above_hull` measures distance to the hull in eV/atom.

### Symmetry Analysis

`SpacegroupAnalyzer` uses spglib to detect the space group, find the conventional
cell, primitive cell, and symmetry operations. Essential for comparing structures and
setting up DFT calculations.

---

## Environment Setup

```bash
# Create and activate environment
conda create -n pymatgen-env python=3.11 -y
conda activate pymatgen-env

# Install packages
pip install pymatgen mp-api numpy matplotlib

# Set Materials Project API key
export MP_API_KEY="<paste-your-key>"

# Verify
python -c "import pymatgen; print('pymatgen:', pymatgen.__version__)"
python -c "from mp_api.client import MPRester; print('mp-api OK')"
```

---

## Core Workflow

### Step 1 — Build and Load Crystal Structures

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def structure_from_cif(cif_path: str) -> Structure:
    """
    Load a crystal structure from a CIF file.

    Args:
        cif_path: Path to the .cif file.

    Returns:
        pymatgen Structure object (first structure in file).
    """
    parser = CifParser(cif_path)
    structures = parser.get_structures(primitive=False)
    structure = structures[0]
    print(f"Loaded: {structure.formula}")
    print(f"  Space group: {SpacegroupAnalyzer(structure).get_space_group_symbol()}")
    print(f"  Lattice: a={structure.lattice.a:.3f}, b={structure.lattice.b:.3f}, "
          f"c={structure.lattice.c:.3f} Ang")
    print(f"  Sites: {len(structure)}")
    return structure


def build_bcc_iron() -> Structure:
    """Build a body-centred cubic Fe structure programmatically."""
    a = 2.87  # Angstrom (experimental)
    lattice = Lattice.cubic(a)
    structure = Structure(
        lattice,
        species=["Fe", "Fe"],
        coords=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],  # fractional coords
    )
    print(f"BCC Fe structure: {structure.formula}")
    return structure


def build_rocksalt(a: float, cation: str, anion: str) -> Structure:
    """
    Build a rocksalt (NaCl-type) crystal structure.

    Args:
        a:      Lattice parameter in Angstrom.
        cation: Cation element symbol (e.g., 'Na').
        anion:  Anion element symbol (e.g., 'Cl').

    Returns:
        pymatgen Structure (8-atom conventional cell).
    """
    lattice = Lattice.cubic(a)
    species = [cation] * 4 + [anion] * 4
    coords  = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],   # FCC cation sublattice
        [0.5, 0.0, 0.0], [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5], [0.5, 0.5, 0.5],   # FCC anion sublattice shifted
    ]
    return Structure(lattice, species, coords)


# Demo structures
fe_bcc = build_bcc_iron()
nacl   = build_rocksalt(5.64, "Na", "Cl")
print(f"\nNaCl structure: {nacl.formula}, {len(nacl)} sites")
```

### Step 2 — Symmetry Analysis

```python
def full_symmetry_analysis(structure: Structure) -> dict:
    """
    Perform complete symmetry analysis on a crystal structure.

    Args:
        structure: pymatgen Structure object.

    Returns:
        Dictionary with space group, Wyckoff positions, conventional cell.
    """
    analyzer = SpacegroupAnalyzer(structure, symprec=0.01)

    sg_symbol  = analyzer.get_space_group_symbol()
    sg_number  = analyzer.get_space_group_number()
    point_group = analyzer.get_point_group_symbol()
    crystal_sys = analyzer.get_crystal_system()
    lattice_typ = analyzer.get_lattice_type()

    conventional = analyzer.get_conventional_standard_structure()
    primitive     = analyzer.get_primitive_standard_structure()

    # Symmetry operations
    sym_ops = analyzer.get_symmetry_operations()

    # Wyckoff positions
    sym_dataset = analyzer.get_symmetry_dataset()
    wyckoff_letters = sym_dataset["wyckoffs"]

    print(f"Space group: {sg_symbol} (#{sg_number})")
    print(f"Point group: {point_group}")
    print(f"Crystal system: {crystal_sys}")
    print(f"Lattice type: {lattice_typ}")
    print(f"Symmetry operations: {len(sym_ops)}")
    print(f"Conventional cell: {len(conventional)} sites")
    print(f"Primitive cell:    {len(primitive)} sites")
    print(f"Wyckoff positions: {wyckoff_letters}")

    return {
        "space_group_symbol": sg_symbol,
        "space_group_number": sg_number,
        "point_group": point_group,
        "crystal_system": crystal_sys,
        "conventional_structure": conventional,
        "primitive_structure": primitive,
        "n_symmetry_ops": len(sym_ops),
        "wyckoff_letters": wyckoff_letters,
    }


sym_info = full_symmetry_analysis(nacl)
```

### Step 3 — Materials Project API Queries

```python
import os
from mp_api.client import MPRester


def query_material_properties(
    formula_or_mpid: str,
    api_key: str = None,
) -> list:
    """
    Query the Materials Project for key properties of a compound.

    Args:
        formula_or_mpid: Chemical formula (e.g., 'TiO2') or mp-id (e.g., 'mp-2657').
        api_key:         Materials Project API key. Falls back to MP_API_KEY env var.

    Returns:
        List of dicts with material_id, formula, bandgap, formation_energy,
        e_above_hull, and crystal_system for each matching entry.
    """
    api_key = api_key or os.getenv("MP_API_KEY")
    if not api_key:
        raise ValueError("Set MP_API_KEY environment variable:\n  export MP_API_KEY='<paste-your-key>'")

    results = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula_or_mpid,
            fields=[
                "material_id", "formula_pretty", "band_gap",
                "formation_energy_per_atom", "e_above_hull",
                "symmetry", "volume", "nsites",
            ],
        )

        for doc in docs:
            results.append({
                "material_id":            doc.material_id,
                "formula":                doc.formula_pretty,
                "band_gap_eV":            doc.band_gap,
                "formation_energy_eV_at": doc.formation_energy_per_atom,
                "e_above_hull_eV_at":     doc.e_above_hull,
                "space_group":            doc.symmetry.symbol if doc.symmetry else None,
                "volume_A3":              doc.volume,
                "nsites":                 doc.nsites,
                "is_stable":              doc.e_above_hull < 0.025,
            })

    results.sort(key=lambda x: x["e_above_hull_eV_at"])
    print(f"Found {len(results)} entries for '{formula_or_mpid}'")
    return results


def get_dos_and_bandgap(
    mp_id: str,
    api_key: str = None,
    output_plot: str = None,
) -> dict:
    """
    Retrieve the electronic DOS and plot it for a given Materials Project entry.

    Args:
        mp_id:       Materials Project ID (e.g., 'mp-2657' for TiO2 anatase).
        api_key:     API key; defaults to MP_API_KEY env var.
        output_plot: If given, save the DOS plot as PNG.

    Returns:
        Dictionary with band_gap, cbm, vbm, and DOS data object.
    """
    api_key = api_key or os.getenv("MP_API_KEY")

    with MPRester(api_key) as mpr:
        dos = mpr.get_dos_by_material_id(mp_id)
        summary = mpr.materials.summary.get_data_by_id(
            mp_id, fields=["band_gap", "cbm", "vbm", "is_gap_direct"]
        )

    if output_plot and dos is not None:
        from pymatgen.electronic_structure.plotter import DosPlotter
        plotter = DosPlotter()
        plotter.add_dos("Total DOS", dos.get_dos())
        ax = plotter.get_plot(xlim=(-5, 5))
        ax.set_title(f"DOS for {mp_id}")
        ax.figure.savefig(output_plot, dpi=150)
        print(f"DOS plot saved to {output_plot}")

    return {
        "mp_id": mp_id,
        "band_gap_eV": summary.band_gap,
        "is_gap_direct": summary.is_gap_direct,
        "dos_object": dos,
    }


# Example usage (requires API key):
# api_key = os.getenv("MP_API_KEY")
# tio2_data = query_material_properties("TiO2", api_key)
# for entry in tio2_data[:3]:
#     print(f"{entry['material_id']:12s} {entry['formula']:8s} "
#           f"Eg={entry['band_gap_eV']:.2f} eV  "
#           f"Ef={entry['formation_energy_eV_at']:.3f} eV/at")
```

---

## Advanced Usage

### Phase Diagram Construction

```python
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry
from pymatgen.core.composition import Composition


def build_phase_diagram_from_api(
    elements: list,
    api_key: str = None,
    output_path: str = None,
) -> dict:
    """
    Build a phase diagram for a set of elements using Materials Project data.

    Args:
        elements:    List of element symbols, e.g. ['Li', 'Fe', 'O'].
        api_key:     MP API key; defaults to MP_API_KEY env var.
        output_path: If given, save the diagram as PNG.

    Returns:
        Dictionary with PhaseDiagram object and list of stable entries.
    """
    api_key = api_key or os.getenv("MP_API_KEY")

    with MPRester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys(elements)

    print(f"Downloaded {len(entries)} entries for {'-'.join(elements)}")

    pd = PhaseDiagram(entries)

    stable = [e for e in pd.stable_entries]
    print(f"Stable phases ({len(stable)}):")
    for e in sorted(stable, key=lambda x: x.composition.reduced_formula):
        eabove = pd.get_e_above_hull(e)
        print(f"  {e.composition.reduced_formula:15s} Ef={e.energy_per_atom:.3f} eV/at  "
              f"e_hull={eabove:.3f} eV/at")

    if output_path:
        plotter = PDPlotter(pd, show_unstable=0.1)
        ax = plotter.get_plot()
        ax.set_title(f"{'-'.join(elements)} Phase Diagram")
        ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Phase diagram saved to {output_path}")

    return {"phase_diagram": pd, "stable_entries": stable}


# Example: Li-Fe-O phase diagram
# api_key = os.getenv("MP_API_KEY")
# pd_result = build_phase_diagram_from_api(["Li", "Fe", "O"], api_key,
#                                           output_path="/tmp/LiFeO_phase.png")


def build_manual_phase_diagram(
    compositions_energies: list,
    output_path: str = None,
) -> PhaseDiagram:
    """
    Build a phase diagram from manually supplied (composition, energy) pairs.
    Useful for data from your own DFT calculations.

    Args:
        compositions_energies: List of (formula_str, total_energy_eV) tuples.
        output_path:           Save path for the plot.

    Returns:
        PhaseDiagram object.
    """
    entries = []
    for formula, energy in compositions_energies:
        comp = Composition(formula)
        entries.append(PDEntry(comp, energy))

    pd = PhaseDiagram(entries)

    if output_path:
        plotter = PDPlotter(pd)
        ax = plotter.get_plot()
        ax.figure.savefig(output_path, dpi=150)

    return pd


# Example: Cu-Zn binary system (representative DFT energies)
cu_zn_data = [
    ("Cu",    -3.72 * 1),
    ("Zn",    -1.25 * 1),
    ("CuZn",  -4.80 * 2),
    ("Cu3Zn", -4.62 * 4),
    ("CuZn3", -4.09 * 4),
]
pd_cuzn = build_manual_phase_diagram(cu_zn_data, output_path="/tmp/CuZn_phase.png")
print(f"\nCu-Zn stable phases: "
      f"{[e.composition.reduced_formula for e in pd_cuzn.stable_entries]}")
```

### Structure Transformations

```python
from pymatgen.transformations.standard_transformations import (
    SupercellTransformation,
    SubstitutionTransformation,
)


def make_supercell(structure: Structure, scaling_matrix) -> Structure:
    """
    Create a supercell from a structure.

    Args:
        structure:      pymatgen Structure.
        scaling_matrix: 3x3 matrix or scalar. E.g., [[2,0,0],[0,2,0],[0,0,2]]
                        for a 2x2x2 supercell.

    Returns:
        Supercell Structure object.
    """
    trans = SupercellTransformation(scaling_matrix)
    supercell = trans.apply_transformation(structure)
    print(f"Supercell: {supercell.formula}, {len(supercell)} sites")
    return supercell


def substitute_atoms(
    structure: Structure,
    substitution_dict: dict,
) -> Structure:
    """
    Substitute one element for another across all sites.

    Args:
        structure:         pymatgen Structure.
        substitution_dict: Mapping from original to new element, e.g. {'Na': 'K'}.

    Returns:
        New Structure with substitutions applied.
    """
    trans = SubstitutionTransformation(substitution_dict)
    new_struct = trans.apply_transformation(structure)
    print(f"After substitution {substitution_dict}: {new_struct.formula}")
    return new_struct


# Make a 2x2x2 NaCl supercell
nacl_super = make_supercell(nacl, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

# Substitute Na -> K (KCl structure)
kcl = substitute_atoms(nacl, {"Na": "K"})
print(f"KCl lattice parameter: {kcl.lattice.a:.3f} Ang")
```

---

## Examples

### Example 1 — CIF Load + Full Symmetry Analysis

```python
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar


def analyze_structure_from_poscar_string(poscar_string: str) -> dict:
    """
    Parse a POSCAR-format string, run symmetry analysis, and return results.

    Args:
        poscar_string: VASP POSCAR format crystal structure as a multiline string.

    Returns:
        Dictionary with formula, space group, lattice parameters, and volume.
    """
    import tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".vasp", delete=False) as f:
        f.write(poscar_string)
        tmp_path = f.name

    structure = Poscar.from_file(tmp_path).structure
    os.unlink(tmp_path)

    analyzer = SpacegroupAnalyzer(structure, symprec=0.05)
    sg = analyzer.get_space_group_symbol()
    sg_num = analyzer.get_space_group_number()
    conventional = analyzer.get_conventional_standard_structure()
    lat = structure.lattice

    result = {
        "formula":       structure.formula,
        "space_group":   f"{sg} (#{sg_num})",
        "a": lat.a, "b": lat.b, "c": lat.c,
        "alpha": lat.alpha, "beta": lat.beta, "gamma": lat.gamma,
        "volume_A3":     lat.volume,
        "nsites":        len(structure),
        "density_g_cm3": structure.density,
        "conventional":  conventional,
    }

    print(f"Formula:     {result['formula']}")
    print(f"Space group: {result['space_group']}")
    print(f"a={lat.a:.3f} b={lat.b:.3f} c={lat.c:.3f} Ang")
    print(f"Volume:      {lat.volume:.2f} Ang^3")
    print(f"Density:     {structure.density:.3f} g/cm^3")

    return result


# Example: TiO2 rutile (POSCAR format)
rutile_poscar = """\
TiO2 rutile
1.0
4.5937  0.0000  0.0000
0.0000  4.5937  0.0000
0.0000  0.0000  2.9587
Ti O
2 4
Direct
0.00000 0.00000 0.00000
0.50000 0.50000 0.50000
0.30530 0.30530 0.00000
0.69470 0.69470 0.00000
0.80470 0.19530 0.50000
0.19530 0.80470 0.50000
"""

rutile_info = analyze_structure_from_poscar_string(rutile_poscar)
```

### Example 2 — Materials Project Band Gap Query and Visualization

```python
import os
import pandas as pd
import matplotlib.pyplot as plt


def compare_bandgaps_api(
    formulas: list,
    api_key: str = None,
    output_path: str = "/tmp/bandgap_comparison.png",
) -> pd.DataFrame:
    """
    Query Materials Project for the lowest-e_above_hull entry of each formula,
    collect band gaps, and produce a bar chart.

    Args:
        formulas:    List of chemical formulae, e.g. ['TiO2', 'ZnO', 'GaAs'].
        api_key:     MP API key; reads MP_API_KEY env var if not given.
        output_path: Path to save the bar chart.

    Returns:
        DataFrame with formula, mp_id, band_gap_eV, space_group, e_above_hull.
    """
    api_key = api_key or os.getenv("MP_API_KEY")

    records = []
    with MPRester(api_key) as mpr:
        for formula in formulas:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "band_gap",
                        "e_above_hull", "symmetry"],
                sort_fields=["e_above_hull"],
            )
            if docs:
                best = docs[0]   # lowest e_above_hull
                records.append({
                    "formula":       best.formula_pretty,
                    "mp_id":         best.material_id,
                    "band_gap_eV":   best.band_gap,
                    "e_above_hull":  best.e_above_hull,
                    "space_group":   best.symmetry.symbol if best.symmetry else "?",
                })

    df = pd.DataFrame(records).sort_values("band_gap_eV")

    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 4))
    colors = ["#2196F3" if g > 0 else "#F44336" for g in df["band_gap_eV"]]
    bars = ax.bar(df["formula"], df["band_gap_eV"], color=colors, edgecolor="k", lw=0.5)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Band gap (eV)")
    ax.set_title("Band Gaps from Materials Project (lowest-hull entry)")

    for bar, row in zip(bars, df.itertuples()):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Band gap chart saved to {output_path}")
    plt.show()

    print(df.to_string(index=False))
    return df


# Usage (requires MP_API_KEY):
# df_bg = compare_bandgaps_api(
#     ["TiO2", "ZnO", "GaN", "Si", "Ge", "GaAs"],
#     output_path="/tmp/bandgap_comparison.png",
# )
```

### Example 3 — Binary Phase Diagram from Materials Project

```python
import os
import matplotlib.pyplot as plt
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter


def plot_binary_phase_diagram(
    element_a: str,
    element_b: str,
    api_key: str = None,
    show_unstable: float = 0.05,
    output_path: str = None,
) -> dict:
    """
    Retrieve entries from Materials Project and plot a binary phase diagram.

    Args:
        element_a:     First element, e.g. 'Cu'.
        element_b:     Second element, e.g. 'Zn'.
        api_key:       MP API key.
        show_unstable: Show entries up to this eV/atom above hull as unstable.
        output_path:   Save path for the plot.

    Returns:
        Dictionary with phase_diagram, n_stable, n_total.
    """
    api_key = api_key or os.getenv("MP_API_KEY")

    with MPRester(api_key) as mpr:
        entries = mpr.get_entries_in_chemsys([element_a, element_b])

    pd_obj = PhaseDiagram(entries)
    stable = pd_obj.stable_entries
    print(f"{element_a}-{element_b} system: {len(entries)} total, "
          f"{len(stable)} stable phases")

    for e in sorted(stable, key=lambda x: x.composition.get_atomic_fraction(element_a)):
        print(f"  {e.composition.reduced_formula:10s}  "
              f"Ef = {pd_obj.get_form_energy_per_atom(e):.4f} eV/at")

    if output_path:
        plotter = PDPlotter(pd_obj, show_unstable=show_unstable, backend="matplotlib")
        ax = plotter.get_plot(label_stable=True, label_unstable=False)
        ax.set_title(f"{element_a}-{element_b} Phase Diagram (MP data)")
        ax.figure.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Phase diagram saved to {output_path}")

    return {
        "phase_diagram": pd_obj,
        "n_stable": len(stable),
        "n_total": len(entries),
    }


# Usage (requires MP_API_KEY):
# result = plot_binary_phase_diagram(
#     "Cu", "Zn",
#     output_path="/tmp/CuZn_pd.png",
# )
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `APIError: 401 Unauthorized` | Missing or invalid API key | Set `export MP_API_KEY="<paste-your-key>"` from materialsproject.org/api |
| `CifParserError` | Malformed or non-standard CIF | Try `primitive=True`; or repair with VESTA |
| Space group detected as P1 | Loose tolerance | Decrease `symprec` in `SpacegroupAnalyzer(struct, symprec=0.001)` |
| `CompositionError` | Unknown element symbol | Check formula string; use standard IUPAC symbols |
| Phase diagram has only terminal phases | No intermediate compounds in DB | Check with `mpr.get_entries_in_chemsys` and print all formulas |
| `e_above_hull` is None | Entry not on the computed hull | Use `pd_obj.get_e_above_hull(entry)` after building PhaseDiagram |
| `PDPlotter` error | Matplotlib backend issue | Use `backend="matplotlib"` explicitly |
| Supercell too large | Memory | Limit to 2x2x2 (~64 atoms) for interactive use |

---

## External Resources

- pymatgen documentation: <https://pymatgen.org/>
- pymatgen GitHub: <https://github.com/materialsproject/pymatgen>
- Materials Project: <https://materialsproject.org/>
- MP API docs: <https://api.materialsproject.org/>
- mp-api GitHub: <https://github.com/materialsproject/api>
- spglib (symmetry backend): <https://spglib.readthedocs.io/>
- Phase diagram tutorial: <https://matgenb.materialsvirtuallab.org/>
- Ong et al. (2013) pymatgen paper: *Comput. Mater. Sci.* 68, 314–319

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — CIF/POSCAR loading, MP API, phase diagrams, symmetry |
