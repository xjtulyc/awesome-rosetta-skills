---
name: openfoam-cfd
description: Run and post-process OpenFOAM CFD simulations using PyFoam, pyvista, and vtk for mesh setup, solver control, and visualization.
tags:
  - cfd
  - openfoam
  - simulation
  - fluid-dynamics
  - pyfoam
  - pyvista
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
  - PyFoam>=0.6.9
  - pyvista>=0.42
  - vtk>=9.2
  - matplotlib>=3.7
  - numpy>=1.24
  - pandas>=2.0
last_updated: "2026-03-17"
status: stable
---

# OpenFOAM CFD Simulation Skill

Automate the full OpenFOAM computational fluid dynamics (CFD) workflow — from case setup and mesh generation through solver execution to post-processing and visualization — using PyFoam, pyvista, and vtk.

---

## When to Use This Skill

Use this skill when you need to:

- Set up and run OpenFOAM cases programmatically (incompressible, multiphase, or turbulent flows).
- Generate or modify blockMesh / snappyHexMesh dictionaries via Python.
- Launch solvers (simpleFoam, interFoam, pisoFoam, etc.) and monitor convergence residuals in real time.
- Post-process OpenFOAM results (velocity, pressure, vorticity) without opening the ParaView GUI.
- Export simulation data to CSV, VTK, or image formats for reporting.
- Automate parameter sweeps or sensitivity studies over multiple OpenFOAM runs.

This skill is **not** suitable for:
- Structural or thermal FEA problems (see `finite-element-analysis` skill).
- Combustion simulations requiring detailed chemistry (see `cantera-combustion` skill).
- Real-time or hardware-in-the-loop control applications.

---

## Background & Key Concepts

### OpenFOAM Case Structure

Every OpenFOAM simulation lives in a *case directory* with a strict layout:

```
myCase/
  0/            # Initial and boundary conditions (one file per field)
  constant/     # Physical properties, mesh, turbulence models
  system/        # Solver control: controlDict, fvSchemes, fvSolution
```

The mesh is stored in `constant/polyMesh/` after running `blockMesh` or `snappyHexMesh`.

### Key Solvers

| Solver       | Flow type                              |
|--------------|----------------------------------------|
| simpleFoam   | Steady-state incompressible RANS       |
| pisoFoam     | Transient incompressible laminar/RANS  |
| interFoam    | Transient two-phase (VOF)              |
| rhoPimpleFoam| Transient compressible                 |
| buoyantPimpleFoam | Natural convection / heat transfer |

### Discretisation Vocabulary

- **fvSchemes**: Numerical schemes for divergence, gradient, Laplacian operators.
- **fvSolution**: Linear solver settings (GAMG, PCG, smoothSolver) and relaxation factors.
- **controlDict**: Start/end time, write interval, time step, solver selection.

### PyFoam Overview

PyFoam is a Python library that wraps OpenFOAM utilities:
- `BasicRunner` — launch any OpenFOAM utility/solver.
- `ParsedParameterFile` — read/write OpenFOAM dictionary files.
- `LogLineAnalyzer` / `FoamLogFile` — parse solver log for residuals.
- `SolutionDirectory` — introspect case structure.

### pyvista / vtk Overview

pyvista provides a high-level Python API over vtk. OpenFOAM writes results in the `VTK/` legacy format or as `*.foam` files readable by the OpenFOAM reader. pyvista can load these and perform slices, streamlines, isosurfaces, and rendering without a GUI.

---

## Environment Setup

### 1. Install OpenFOAM

OpenFOAM must be installed natively on Linux or via Docker/WSL2 on Windows/macOS.

```bash
# Ubuntu / Debian — OpenFOAM v2312
sudo add-apt-repository "deb http://dl.openfoam.com/ubuntu focal main"
sudo apt-get update
sudo apt-get install openfoam2312

# Source environment (add to ~/.bashrc for persistence)
source /usr/lib/openfoam/openfoam2312/etc/bashrc
```

### 2. Create and activate a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install "PyFoam>=0.6.9" "pyvista>=0.42" "vtk>=9.2" \
            "matplotlib>=3.7" "numpy>=1.24" "pandas>=2.0"
```

### 4. Verify installation

```python
import PyFoam
import pyvista as pv
import vtk
import numpy as np
import matplotlib
import pandas as pd

print("PyFoam version  :", PyFoam.__version__)
print("pyvista version :", pv.__version__)
print("vtk version     :", vtk.vtkVersion.GetVTKVersion())
```

### 5. Docker alternative

```bash
# Pull the official OpenFOAM + PyFoam image
docker pull openfoam/openfoam2312-paraview510

# Run interactively
docker run -it --rm \
  -v $(pwd)/cases:/cases \
  openfoam/openfoam2312-paraview510 bash
```

---

## Core Workflow

### Step 1 — Build the Case Directory Structure

```python
"""
create_case.py
Creates an OpenFOAM simpleFoam cavity case directory from scratch.
"""

import os
import textwrap
from pathlib import Path


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))
    print(f"  wrote {path}")


def create_cavity_case(case_dir: str = "cavity") -> Path:
    case = Path(case_dir)

    # ---------- 0/ ----------
    write_file(
        case / "0" / "U",
        """\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volVectorField;
            object      U;
        }
        dimensions      [0 1 -1 0 0 0 0];
        internalField   uniform (0 0 0);
        boundaryField
        {
            movingWall  { type fixedValue; value uniform (1 0 0); }
            fixedWalls  { type noSlip; }
            frontAndBack{ type empty; }
        }
        """,
    )

    write_file(
        case / "0" / "p",
        """\
        FoamFile
        {
            version     2.0;
            format      ascii;
            class       volScalarField;
            object      p;
        }
        dimensions      [0 2 -2 0 0 0 0];
        internalField   uniform 0;
        boundaryField
        {
            movingWall  { type zeroGradient; }
            fixedWalls  { type zeroGradient; }
            frontAndBack{ type empty; }
        }
        """,
    )

    # ---------- constant/ ----------
    write_file(
        case / "constant" / "transportProperties",
        """\
        FoamFile
        {
            version 2.0;
            format  ascii;
            class   dictionary;
            object  transportProperties;
        }
        nu  [0 2 -1 0 0 0 0]  0.01;
        """,
    )

    write_file(
        case / "constant" / "turbulenceProperties",
        """\
        FoamFile
        {
            version 2.0;
            format  ascii;
            class   dictionary;
            object  turbulenceProperties;
        }
        simulationType laminar;
        """,
    )

    # ---------- system/ ----------
    write_file(
        case / "system" / "blockMeshDict",
        """\
        FoamFile
        {
            version 2.0;
            format  ascii;
            class   dictionary;
            object  blockMeshDict;
        }
        scale 0.1;
        vertices
        (
            (0 0 0) (1 0 0) (1 1 0) (0 1 0)
            (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1)
        );
        blocks
        (
            hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)
        );
        boundary
        (
            movingWall  { type wall; faces ((3 7 6 2)); }
            fixedWalls  { type wall; faces ((0 4 7 3)(2 6 5 1)(1 5 4 0)); }
            frontAndBack{ type empty; faces ((0 3 2 1)(4 5 6 7)); }
        );
        """,
    )

    write_file(
        case / "system" / "controlDict",
        """\
        FoamFile
        {
            version 2.0;
            format  ascii;
            class   dictionary;
            object  controlDict;
        }
        application     icoFoam;
        startFrom       startTime;
        startTime       0;
        stopAt          endTime;
        endTime         0.5;
        deltaT          0.005;
        writeControl    timeStep;
        writeInterval   20;
        purgeWrite      0;
        writeFormat     ascii;
        writePrecision  6;
        runTimeModifiable true;
        """,
    )

    write_file(
        case / "system" / "fvSchemes",
        """\
        FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }
        ddtSchemes      { default Euler; }
        gradSchemes     { default Gauss linear; }
        divSchemes      { default none; div(phi,U) Gauss linearUpwind grad(U); }
        laplacianSchemes{ default Gauss linear corrected; }
        interpolationSchemes { default linear; }
        snGradSchemes   { default corrected; }
        """,
    )

    write_file(
        case / "system" / "fvSolution",
        """\
        FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
        solvers
        {
            p   { solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.05; }
            pFinal { $p; relTol 0; }
            U   { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0; }
        }
        PISO { nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
        """,
    )

    print(f"\nCase directory created at: {case.resolve()}")
    return case


if __name__ == "__main__":
    create_cavity_case("cavity")
```

### Step 2 — Generate the Mesh and Run the Solver

```python
"""
run_case.py
Runs blockMesh then icoFoam on the cavity case using PyFoam.
"""

import subprocess
import sys
from pathlib import Path

from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory


def run_openfoam_case(case_dir: str = "cavity") -> None:
    case = Path(case_dir).resolve()

    if not case.exists():
        print(f"Case directory not found: {case}")
        sys.exit(1)

    sol = SolutionDirectory(str(case))
    print(f"Case: {sol.name}")
    print(f"Times available before run: {sol.getTimes()}")

    # --- Step 2a: run blockMesh ---
    print("\n--- Running blockMesh ---")
    mesh_runner = BasicRunner(
        argv=["blockMesh", "-case", str(case)],
        silent=False,
        logname="blockMesh",
    )
    mesh_runner.start()
    if not mesh_runner.runOK():
        print("blockMesh FAILED — check blockMesh.logfile")
        sys.exit(1)

    # --- Step 2b: run icoFoam ---
    print("\n--- Running icoFoam ---")
    solver_runner = BasicRunner(
        argv=["icoFoam", "-case", str(case)],
        silent=False,
        logname="icoFoam",
    )
    solver_runner.start()
    if not solver_runner.runOK():
        print("icoFoam FAILED — check icoFoam.logfile")
        sys.exit(1)

    print("\nSolver finished successfully.")
    print(f"Times available after run: {sol.getTimes()}")


if __name__ == "__main__":
    run_openfoam_case("cavity")
```

### Step 3 — Parse Residuals and Post-Process with pyvista

```python
"""
postprocess.py
Reads icoFoam residuals, plots convergence, then loads VTK output
and renders a velocity magnitude slice.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv


# ------------------------------------------------------------------
# 3a: Parse solver log for residuals
# ------------------------------------------------------------------
def parse_residuals(log_path: str) -> pd.DataFrame:
    """Extract Time, Ux_residual, Uy_residual, p_residual from log."""
    pattern_time = re.compile(r"^Time = ([\d.eE+\-]+)")
    pattern_res  = re.compile(
        r"Solving for (\w+),\s+Initial residual = ([\d.eE+\-]+)"
    )

    records: list[dict] = []
    current_time: float | None = None

    with open(log_path) as fh:
        for line in fh:
            m_time = pattern_time.match(line)
            if m_time:
                current_time = float(m_time.group(1))
                records.append({"time": current_time})
            m_res = pattern_res.search(line)
            if m_res and records:
                field = m_res.group(1)
                residual = float(m_res.group(2))
                records[-1][field] = residual

    df = pd.DataFrame(records).set_index("time")
    return df


def plot_residuals(df: pd.DataFrame, out_png: str = "residuals.png") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in df.columns:
        ax.semilogy(df.index, df[col], label=col)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Initial residual")
    ax.set_title("icoFoam convergence residuals")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Residual plot saved to {out_png}")


# ------------------------------------------------------------------
# 3b: Load last time-step VTK and render velocity magnitude
# ------------------------------------------------------------------
def postprocess_vtk(case_dir: str = "cavity", out_png: str = "velocity.png") -> None:
    case = Path(case_dir)

    # Convert to VTK first (OpenFOAM utility)
    import subprocess
    subprocess.run(
        ["foamToVTK", "-case", str(case), "-latestTime"],
        check=True,
    )

    # Find the last VTK internal mesh file
    vtk_files = sorted((case / "VTK").glob("cavity_*.vtk"))
    if not vtk_files:
        print("No VTK files found — did foamToVTK run?")
        return

    mesh = pv.read(str(vtk_files[-1]))
    print(f"Loaded: {vtk_files[-1].name}")
    print(f"  Bounds : {mesh.bounds}")
    print(f"  N cells: {mesh.n_cells}")

    # Compute velocity magnitude
    if "U" in mesh.array_names:
        U = mesh["U"]                       # shape (N, 3)
        mesh["U_mag"] = np.linalg.norm(U, axis=1)

    # Slice at z = mid-plane
    mid_z = 0.5 * (mesh.bounds[4] + mesh.bounds[5])
    slc = mesh.slice(normal="z", origin=(0, 0, mid_z))

    # Off-screen rendering
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        slc,
        scalars="U_mag",
        cmap="turbo",
        scalar_bar_args={"title": "Velocity magnitude (m/s)"},
    )
    plotter.add_axes()
    plotter.view_xy()
    plotter.screenshot(out_png)
    print(f"Velocity magnitude plot saved to {out_png}")


if __name__ == "__main__":
    log_file = "cavity/icoFoam.logfile"
    if Path(log_file).exists():
        df = parse_residuals(log_file)
        print(df.head())
        plot_residuals(df)

    postprocess_vtk("cavity")
```

---

## Advanced Usage

### A — Parameter Sweep with PyFoam ParsedParameterFile

```python
"""
param_sweep.py
Runs icoFoam for multiple viscosity values and collects final p residuals.
"""

import shutil
from pathlib import Path

import pandas as pd
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile


def sweep_viscosity(
    base_case: str = "cavity",
    nu_values: list[float] | None = None,
) -> pd.DataFrame:
    if nu_values is None:
        nu_values = [0.005, 0.01, 0.02, 0.05]

    results = []

    for nu in nu_values:
        run_dir = Path(f"cavity_nu{nu:.4f}")
        shutil.copytree(base_case, str(run_dir), dirs_exist_ok=True)

        # Modify viscosity
        tp_path = run_dir / "constant" / "transportProperties"
        tp = ParsedParameterFile(str(tp_path))
        tp["nu"] = f"[0 2 -1 0 0 0 0] {nu}"
        tp.writeFile()

        # Run blockMesh + icoFoam
        for util in ["blockMesh", "icoFoam"]:
            runner = BasicRunner(
                argv=[util, "-case", str(run_dir)],
                silent=True,
                logname=util,
            )
            runner.start()

        # Collect last p residual from log
        log = run_dir / "icoFoam.logfile"
        import re
        last_p_res = None
        if log.exists():
            for line in log.read_text().splitlines():
                m = re.search(r"Solving for p,\s+Initial residual = ([\d.eE+\-]+)", line)
                if m:
                    last_p_res = float(m.group(1))

        re_num = 0.1 / nu  # U_wall * L / nu  (L=0.1, U=1)
        results.append({"nu": nu, "Re": re_num, "final_p_res": last_p_res})
        print(f"  nu={nu:.4f}  Re={re_num:.1f}  p_res={last_p_res}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = sweep_viscosity()
    print(df.to_string(index=False))
    df.to_csv("sweep_results.csv", index=False)
```

### B — snappyHexMesh Dictionary Generation

```python
"""
snappy_setup.py
Generates a minimal snappyHexMeshDict for an STL geometry.
"""

from pathlib import Path


SNAPPY_TEMPLATE = """\
FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    {stl_name}
    {{
        type triSurfaceMesh;
        name {region_name};
    }}
}}

castellatedMeshControls
{{
    maxLocalCells       1000000;
    maxGlobalCells      2000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces
    {{
        {region_name} {{ level (2 3); }}
    }}
    refinementRegions  {{}}
    locationInMesh ({loc_x} {loc_y} {loc_z});
    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch       3;
    tolerance          2.0;
    nSolveIter         30;
    nRelaxIter         5;
    nFeatureSnapIter   10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes        true;
    layers               {{}}
    expansionRatio       1.2;
    finalLayerThickness  0.3;
    minThickness         0.1;
    nGrow                0;
    featureAngle         60;
    nRelaxIter           3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals       3;
    nSmoothThickness     10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle   90;
    nBufferCellsNoExtrude 0;
    nLayerIter           50;
}}

meshQualityControls
{{
    maxNonOrtho          65;
    maxBoundarySkewness  20;
    maxInternalSkewness  4;
    maxConcave           80;
    minFlatness          0.5;
    minVol               1e-13;
    minTetQuality        1e-15;
    minArea              -1;
    minTwist             0.02;
    minDeterminant       0.001;
    minFaceWeight        0.02;
    minVolRatio          0.01;
    minTriangleTwist     -1;
    nSmoothScale         4;
    errorReduction       0.75;
}}

mergeTolerance 1e-6;
"""


def write_snappy_dict(
    case_dir: str,
    stl_name: str = "geometry.stl",
    region_name: str = "body",
    location_in_mesh: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> None:
    content = SNAPPY_TEMPLATE.format(
        stl_name=stl_name,
        region_name=region_name,
        loc_x=location_in_mesh[0],
        loc_y=location_in_mesh[1],
        loc_z=location_in_mesh[2],
    )
    out = Path(case_dir) / "system" / "snappyHexMeshDict"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content)
    print(f"snappyHexMeshDict written to {out}")


if __name__ == "__main__":
    write_snappy_dict("myCase", stl_name="wing.stl", region_name="wing")
```

### C — interFoam Two-Phase Flow Setup

```python
"""
interfoam_setup.py
Configures the alpha.water initial field and transportProperties
for an interFoam dam-break case.
"""

from pathlib import Path


def write_alpha_field(case_dir: str, water_height: float = 0.3) -> None:
    """Write alpha.water field with water column from 0 to water_height."""
    content = f"""\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alpha.water;
}}
dimensions      [0 0 0 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    atmosphere  {{ type inletOutlet; inletValue uniform 0; value uniform 0; }}
    walls       {{ type zeroGradient; }}
    defaultFaces{{ type empty; }}
}}
"""
    path = Path(case_dir) / "0" / "alpha.water"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"alpha.water written (water_height hint={water_height} — set via setFields)")


def write_interfoam_transport(
    case_dir: str,
    nu_water: float = 1e-6,
    nu_air: float = 1.48e-5,
    rho_water: float = 1000.0,
    rho_air: float = 1.0,
) -> None:
    content = f"""\
FoamFile
{{
    version 2.0;
    format  ascii;
    class   dictionary;
    object  transportProperties;
}}
phases (water air);
water
{{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] {nu_water};
    rho             [1 -3 0 0 0 0 0] {rho_water};
}}
air
{{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] {nu_air};
    rho             [1 -3 0 0 0 0 0] {rho_air};
}}
sigma   [1 0 -2 0 0 0 0] 0.07;
"""
    path = Path(case_dir) / "constant" / "transportProperties"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"transportProperties (interFoam) written to {path}")


if __name__ == "__main__":
    write_alpha_field("damBreak")
    write_interfoam_transport("damBreak")
```

---

## Troubleshooting

### T1 — "Cannot find patchField entry for …"

The boundary condition patch name in `0/` does not match the mesh patch names in `constant/polyMesh/boundary`. Run `checkMesh -case <caseDir>` and ensure every patch in the mesh has a corresponding entry in each initial field file.

### T2 — Divergence / NaN residuals

Common causes and fixes:

```bash
# Check mesh quality — look for maxNonOrtho > 70 or negative volumes
checkMesh -case cavity

# Reduce Courant number (deltaT in controlDict)
# For icoFoam: Co = U * deltaT / dx  should stay < 1
# Halve deltaT and double writeInterval to maintain same output frequency
```

```python
# Programmatically lower deltaT using PyFoam
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from pathlib import Path

cd = ParsedParameterFile(str(Path("cavity") / "system" / "controlDict"))
cd["deltaT"] = 0.0025
cd.writeFile()
print("deltaT updated to 0.0025")
```

### T3 — PyFoam "command not found"

```bash
# Ensure OpenFOAM environment is sourced BEFORE running Python
source /usr/lib/openfoam/openfoam2312/etc/bashrc
which icoFoam   # should resolve to an OpenFOAM binary
```

### T4 — pyvista cannot read VTK file

```python
import pyvista as pv

# Use the OpenFOAM reader directly (no need for foamToVTK)
reader = pv.OpenFOAMReader("cavity/cavity.foam")
reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()
print(mesh.keys())       # list available datasets
internal = mesh["internalMesh"]
print(internal.array_names)
```

### T5 — Out-of-memory on large meshes

```bash
# Run decomposePar for parallel execution (e.g., 8 cores)
decomposePar -case cavity
mpirun -np 8 icoFoam -parallel -case cavity > log.icoFoam 2>&1
reconstructPar -case cavity
```

---

## External Resources

- [OpenFOAM Official Documentation](https://www.openfoam.com/documentation/guides/latest/doc/)
- [OpenFOAM User Guide (ESI)](https://www.openfoam.com/documentation/user-guide)
- [PyFoam GitHub Repository](https://github.com/michaeltheoldman/PyFoam)
- [PyFoam Documentation](https://openfoamwiki.net/index.php/Contrib/PyFoam)
- [pyvista Documentation](https://docs.pyvista.org/)
- [OpenFOAM Wiki — Case Structures](https://openfoamwiki.net/index.php/Main_Page)
- [CFD Online Forums](https://www.cfd-online.com/Forums/openfoam/)
- [The OpenFOAM Technology Primer (Passalacqua et al.)](https://sourceforge.net/projects/openfoam-extend/files/OpenFOAM_Workshops/OFW9_2014_Zagreb/training/)

---

## Examples

### Example 1 — End-to-End Lid-Driven Cavity (Laminar)

```python
"""
example_cavity_full.py
Full pipeline: create case -> mesh -> solve -> plot residuals -> render velocity.
"""

import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.RunDictionary.SolutionDirectory import SolutionDirectory

CASE = "cavity_example"


def build_case() -> None:
    """Write all required OpenFOAM dictionary files."""
    base = Path(CASE)

    files = {
        "0/U": """\
FoamFile{version 2.0;format ascii;class volVectorField;object U;}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField{movingWall{type fixedValue;value uniform (1 0 0);}fixedWalls{type noSlip;}frontAndBack{type empty;}}
""",
        "0/p": """\
FoamFile{version 2.0;format ascii;class volScalarField;object p;}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField{movingWall{type zeroGradient;}fixedWalls{type zeroGradient;}frontAndBack{type empty;}}
""",
        "constant/transportProperties": "FoamFile{version 2.0;format ascii;class dictionary;object transportProperties;}\nnu [0 2 -1 0 0 0 0] 0.01;\n",
        "constant/turbulenceProperties": "FoamFile{version 2.0;format ascii;class dictionary;object turbulenceProperties;}\nsimulationType laminar;\n",
        "system/blockMeshDict": """\
FoamFile{version 2.0;format ascii;class dictionary;object blockMeshDict;}
scale 0.1;
vertices((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 0.1)(1 0 0.1)(1 1 0.1)(0 1 0.1));
blocks(hex(0 1 2 3 4 5 6 7)(20 20 1)simpleGrading(1 1 1));
boundary(movingWall{type wall;faces((3 7 6 2));}fixedWalls{type wall;faces((0 4 7 3)(2 6 5 1)(1 5 4 0));}frontAndBack{type empty;faces((0 3 2 1)(4 5 6 7));});
""",
        "system/controlDict": """\
FoamFile{version 2.0;format ascii;class dictionary;object controlDict;}
application icoFoam;startFrom startTime;startTime 0;stopAt endTime;endTime 0.5;
deltaT 0.005;writeControl timeStep;writeInterval 20;writeFormat ascii;
""",
        "system/fvSchemes": """\
FoamFile{version 2.0;format ascii;class dictionary;object fvSchemes;}
ddtSchemes{default Euler;}gradSchemes{default Gauss linear;}
divSchemes{default none;div(phi,U) Gauss linearUpwind grad(U);}
laplacianSchemes{default Gauss linear corrected;}
interpolationSchemes{default linear;}snGradSchemes{default corrected;}
""",
        "system/fvSolution": """\
FoamFile{version 2.0;format ascii;class dictionary;object fvSolution;}
solvers{p{solver PCG;preconditioner DIC;tolerance 1e-06;relTol 0.05;}
pFinal{$p;relTol 0;}
U{solver smoothSolver;smoother symGaussSeidel;tolerance 1e-05;relTol 0;}}
PISO{nCorrectors 2;nNonOrthogonalCorrectors 0;pRefCell 0;pRefValue 0;}
""",
    }

    for rel_path, content in files.items():
        p = base / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)


def run() -> None:
    for util in ["blockMesh", "icoFoam"]:
        r = BasicRunner(
            argv=[util, "-case", CASE], silent=False, logname=util
        )
        r.start()
        assert r.runOK(), f"{util} failed"


def plot_residuals() -> None:
    log = Path(CASE) / "icoFoam.logfile"
    records: list[dict] = []
    current: dict = {}
    for line in log.read_text().splitlines():
        m = re.match(r"^Time = ([\d.eE+\-]+)", line)
        if m:
            if current:
                records.append(current)
            current = {"time": float(m.group(1))}
        m2 = re.search(r"Solving for (\w+),\s+Initial residual = ([\d.eE+\-]+)", line)
        if m2:
            current[m2.group(1)] = float(m2.group(2))
    if current:
        records.append(current)

    df = pd.DataFrame(records).set_index("time")
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in df.columns:
        ax.semilogy(df.index, df[col], label=col)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual")
    ax.set_title("Cavity residuals")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.savefig("cavity_residuals.png", dpi=150)
    print("Saved cavity_residuals.png")


def render_velocity() -> None:
    subprocess.run(
        ["foamToVTK", "-case", CASE, "-latestTime"], check=True
    )
    vtk_files = sorted(Path(CASE).glob("VTK/*.vtk"))
    if not vtk_files:
        return
    mesh = pv.read(str(vtk_files[-1]))
    if "U" in mesh.array_names:
        mesh["U_mag"] = np.linalg.norm(mesh["U"], axis=1)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, scalars="U_mag", cmap="turbo")
    pl.view_xy()
    pl.screenshot("cavity_velocity.png")
    print("Saved cavity_velocity.png")


if __name__ == "__main__":
    build_case()
    run()
    plot_residuals()
    render_velocity()
```

### Example 2 — Residual Monitor with Live Plot during Solver Execution

```python
"""
example_live_monitor.py
Launches icoFoam in a subprocess and streams residuals to a matplotlib figure
updated every N time steps. Saves the final plot on completion.
"""

import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; swap for TkAgg for GUI
import matplotlib.pyplot as plt


def monitor_solver(
    case_dir: str = "cavity",
    solver: str = "icoFoam",
    update_every: int = 5,
    out_png: str = "live_residuals.png",
) -> None:
    cmd = [solver, "-case", case_dir]
    print(f"Launching: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    residuals: dict[str, list[float]] = defaultdict(list)
    times: list[float] = []
    current_time: float = 0.0
    step_count = 0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Initial residual")
    ax.set_title(f"{solver} — live residuals")
    ax.set_yscale("log")

    for line in proc.stdout:
        sys.stdout.write(line)

        m_t = re.match(r"^Time = ([\d.eE+\-]+)", line.strip())
        if m_t:
            current_time = float(m_t.group(1))
            times.append(current_time)
            step_count += 1

        m_r = re.search(
            r"Solving for (\w+),\s+Initial residual = ([\d.eE+\-]+)", line
        )
        if m_r:
            residuals[m_r.group(1)].append(float(m_r.group(2)))

        if step_count > 0 and step_count % update_every == 0:
            ax.cla()
            ax.set_yscale("log")
            ax.set_xlabel("Step")
            ax.set_ylabel("Initial residual")
            ax.set_title(f"{solver} — live residuals (t={current_time:.4f})")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            for field, vals in residuals.items():
                ax.plot(vals, label=field)
            ax.legend()
            fig.savefig(out_png, dpi=120)

    proc.wait()

    # Final save
    ax.cla()
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel("Initial residual")
    ax.set_title(f"{solver} — final residuals")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for field, vals in residuals.items():
        ax.plot(vals, label=field)
    ax.legend()
    fig.savefig(out_png, dpi=150)
    print(f"\nFinal residual plot saved to {out_png}")


if __name__ == "__main__":
    monitor_solver(case_dir="cavity")
```
