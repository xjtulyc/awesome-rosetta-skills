---
name: finite-element-analysis
description: >
  Use this Skill for FEA with FEniCSx or scikit-fem: mesh generation, boundary
  conditions, linear elasticity, heat conduction, and result visualization.
tags:
  - engineering
  - finite-element
  - fenics
  - pde
  - structural-analysis
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
    - scikit-fem>=9.0
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# Finite Element Analysis

> **One-line summary**: Solve PDEs numerically using the Finite Element Method with scikit-fem: structural mechanics, heat conduction, and eigenvalue problems on 2D/3D meshes.

---

## When to Use This Skill

- When solving structural mechanics problems (stress, displacement, strain)
- When computing temperature distributions in heat conduction problems
- When solving Poisson, Laplace, or Helmholtz equations on complex geometries
- When performing modal analysis (eigenfrequencies and mode shapes)
- When working with boundary conditions: Dirichlet, Neumann, Robin
- When post-processing FEA results (von Mises stress, heat flux)

**Trigger keywords**: finite element, FEA, FEM, FEniCS, scikit-fem, elasticity, heat conduction, PDE, mesh, stiffness matrix, boundary condition, Galerkin, stress analysis

---

## Background & Key Concepts

### Weak Formulation

For a PDE $\mathcal{L}u = f$ on domain $\Omega$, the weak form seeks $u \in V$:

$$
a(u, v) = L(v) \quad \forall v \in V
$$

where $a(\cdot,\cdot)$ is the bilinear form and $L(\cdot)$ is the linear functional.

### Linear Elasticity

Equilibrium: $\nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = \mathbf{0}$

Constitutive law (plane stress):

$$
\boldsymbol{\sigma} = \frac{E}{1-\nu^2}\begin{pmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{pmatrix} \boldsymbol{\varepsilon}
$$

### Heat Equation (Steady-State)

$$
-\nabla \cdot (k \nabla T) = q \quad \text{in } \Omega
$$

Boundary conditions: Dirichlet $T=T_0$ on $\Gamma_D$, Neumann $k\partial T/\partial n = h(T-T_\infty)$ on $\Gamma_N$.

---

## Environment Setup

### Install Dependencies

```bash
pip install scikit-fem>=9.0 numpy>=1.24 scipy>=1.11 matplotlib>=3.7
# Optional: meshio for mesh import/export
pip install meshio
```

### Verify Installation

```python
import skfem
import numpy as np
from skfem import MeshTri, Basis, ElementTriP1
from skfem.models.poisson import laplace, mass

mesh = MeshTri.init_sqsymmetric(3)  # 3×3 structured triangular mesh
print(f"scikit-fem version: {skfem.__version__}")
print(f"Mesh: {mesh.nvertices} nodes, {mesh.nelements} elements")
# Expected: ~32 nodes, 18 elements
```

---

## Core Workflow

### Step 1: Poisson Equation on Unit Square

```python
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm
from skfem import enforce, solve
from skfem.helpers import grad, dot

# ------------------------------------------------------------------ #
# Solve -∇²u = f  on [0,1]², u=0 on boundary
# Manufactured solution: u_exact = sin(πx)sin(πy), f = 2π²u_exact
# ------------------------------------------------------------------ #

# Create mesh
mesh = MeshTri.init_sqsymmetric(6)  # 6×6 regular triangular mesh
mesh = mesh.refined(3)               # 3 levels of uniform refinement

# Define function space (piecewise linear P1 elements)
basis = Basis(mesh, ElementTriP1())

@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm
def load(v, w):
    # f = 2π²sin(πx)sin(πy)
    x, y = w.x
    f = 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    return f * v

# Assemble global system
K = stiffness.assemble(basis)
F = load.assemble(basis)

# Dirichlet BC: u=0 on all boundary nodes
boundary_dofs = basis.get_dofs().all()
K, F = enforce(K, F, D=boundary_dofs)

# Solve
u = solve(K, F)

# ---- Post-processing -------------------------------------------- #
x_nodes, y_nodes = mesh.p[0], mesh.p[1]
u_exact = np.sin(np.pi * x_nodes) * np.sin(np.pi * y_nodes)
L2_error = np.sqrt(np.mean((u - u_exact)**2))
print(f"L² error: {L2_error:.2e}")  # Should be < 1e-3 for fine mesh

# Plot solution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# FEM solution
im0 = axes[0].tricontourf(x_nodes, y_nodes, mesh.t.T, u, levels=20, cmap='viridis')
plt.colorbar(im0, ax=axes[0])
axes[0].set_title("FEM Solution u")
axes[0].set_aspect('equal')

# Exact solution
im1 = axes[1].tricontourf(x_nodes, y_nodes, mesh.t.T, u_exact, levels=20, cmap='viridis')
plt.colorbar(im1, ax=axes[1])
axes[1].set_title("Exact Solution")
axes[1].set_aspect('equal')

# Error
error = np.abs(u - u_exact)
im2 = axes[2].tricontourf(x_nodes, y_nodes, mesh.t.T, error, levels=20, cmap='hot_r')
plt.colorbar(im2, ax=axes[2])
axes[2].set_title(f"|Error|  (L²={L2_error:.2e})")
axes[2].set_aspect('equal')

plt.suptitle("Poisson Equation FEM Solution on Unit Square")
plt.tight_layout()
plt.savefig("poisson_fem.png", dpi=150)
plt.show()
```

### Step 2: Steady-State Heat Conduction

```python
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm
from skfem import enforce, solve
from skfem.helpers import grad, dot

# ------------------------------------------------------------------ #
# Heat conduction: -k∇²T = q  on L-shaped domain
# T = 0 on left/bottom, q=1 W/m² internal, convection on top/right
# ------------------------------------------------------------------ #

# L-shaped domain mesh
mesh = MeshTri.init_lshaped()
mesh = mesh.refined(3)

basis = Basis(mesh, ElementTriP1())

# Material properties
k_cond = 50.0   # W/(m·K) — steel
h_conv = 25.0   # W/(m²·K) — convection coefficient
T_inf  = 20.0   # °C ambient
q_vol  = 1000.0 # W/m³ internal heat generation

@BilinearForm
def conduction(u, v, w):
    return k_cond * dot(grad(u), grad(v))

@BilinearForm
def convection_bc(u, v, w):
    return h_conv * u * v  # Robin BC contribution

@LinearForm
def heat_source(v, w):
    return q_vol * v

@LinearForm
def convection_load(v, w):
    return h_conv * T_inf * v  # Robin BC right-hand side

# Assemble volume integrals
K = conduction.assemble(basis)
F = heat_source.assemble(basis)

# Identify boundary facets for convection (top boundary y≈1)
# In L-shaped domain, apply convection on outer boundaries
facets_top    = mesh.facets_satisfying(lambda x: np.isclose(x[1], 1.0))
facets_right  = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))
conv_facets   = np.union1d(facets_top, facets_right)

if len(conv_facets) > 0:
    facet_basis = basis.boundary(facets=conv_facets)
    K += convection_bc.assemble(facet_basis)
    F += convection_load.assemble(facet_basis)

# Dirichlet BC: T=0 on left and bottom edges
dofs_left   = basis.get_dofs(mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.0))).all()
dofs_bottom = basis.get_dofs(mesh.facets_satisfying(lambda x: np.isclose(x[1], 0.0))).all()
dofs_fixed  = np.union1d(dofs_left, dofs_bottom)

K, F = enforce(K, F, D=dofs_fixed, overwrite=True)

# Solve for temperature
T = solve(K, F)

print(f"Temperature range: {T.min():.1f} – {T.max():.1f} °C")
print(f"Max temperature: {T.max():.1f} °C at node {T.argmax()}")

# Visualize
fig, ax = plt.subplots(figsize=(7, 6))
x_n, y_n = mesh.p[0], mesh.p[1]
im = ax.tricontourf(x_n, y_n, mesh.t.T, T, levels=30, cmap='hot')
plt.colorbar(im, ax=ax, label='Temperature (°C)')
ax.triplot(x_n, y_n, mesh.t.T, color='gray', linewidth=0.2, alpha=0.3)
ax.set_title("Steady-State Heat Conduction — L-shaped Domain")
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("heat_conduction.png", dpi=150)
plt.show()
```

### Step 3: Plane-Stress Linear Elasticity

```python
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, ElementTriP2, BilinearForm, LinearForm
from skfem import enforce, solve
from skfem.helpers import sym_grad, eye, transpose, dd, ddot, trace

# ------------------------------------------------------------------ #
# Plane-stress elasticity: thin plate with fixed left edge, uniaxial
# tension load on right edge. Material: structural steel.
# ------------------------------------------------------------------ #

# Rectangular plate mesh (width=2, height=1)
mesh = MeshTri.init_symmetric()
mesh = mesh.refined(3)
mesh = mesh.with_defaults()

E  = 200e9   # Young's modulus (Pa)
nu = 0.30    # Poisson's ratio
P  = 1e6     # Applied traction (Pa) on right edge

# Plane-stress stiffness tensor components
lam = E * nu / ((1 + nu) * (1 - 2*nu))  # Lame lambda
mu  = E / (2 * (1 + nu))                # Shear modulus

@BilinearForm
def elasticity(u, v, w):
    # Plane-stress: σ_ij = λδ_ij ε_kk + 2μ ε_ij
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    return (lam * trace(eps_u) * trace(eps_v)
            + 2 * mu * ddot(eps_u, eps_v))

basis = Basis(mesh, ElementTriP2())  # Quadratic elements for accuracy

K = elasticity.assemble(basis)
F = np.zeros(K.shape[0])

# Apply traction on right boundary (Neumann BC in x-direction)
facets_right = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))
if len(facets_right) > 0:
    facet_basis = basis.boundary(facets=facets_right)

    @LinearForm
    def traction_x(v, w):
        return P * v[0]  # traction in x-direction only

    F += traction_x.assemble(facet_basis)

# Dirichlet: fix left edge (u_x = u_y = 0)
dofs_left = basis.get_dofs(mesh.facets_satisfying(lambda x: np.isclose(x[0], 0.0)))
K, F = enforce(K, F, D=dofs_left.all(), overwrite=True)

# Solve
u_vec = solve(K, F)
uh = basis.interpolate(u_vec)

# Extract displacement components
u_x = u_vec[basis.nodal_dofs[0]]
u_y = u_vec[basis.nodal_dofs[1]]

print(f"Max displacement x: {u_x.max()*1e3:.4f} mm")
print(f"Max displacement y: {u_y.max()*1e3:.4f} mm")
# Analytical: u_x_max = P*L/(E) = 1e6*1/200e9 = 5e-6 m = 0.005 mm
u_analytical = P * 1.0 / E
print(f"Analytical u_x max: {u_analytical*1e3:.4f} mm")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x_n, y_n = mesh.p[0], mesh.p[1]

for ax, data, title, unit in zip(
    axes,
    [u_x * 1e6, u_y * 1e6],
    ["Displacement u_x", "Displacement u_y"],
    ["μm", "μm"]
):
    im = ax.tricontourf(x_n, y_n, mesh.t.T, data, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label=f"Displacement ({unit})")
    ax.set_title(title); ax.set_aspect('equal')
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

plt.suptitle("Plane-Stress Linear Elasticity — Uniaxial Tension")
plt.tight_layout()
plt.savefig("elasticity_fem.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Modal Analysis (Eigenvalue Problem)

```python
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, ElementTriP2, BilinearForm
from skfem import enforce
from skfem.helpers import sym_grad, trace, ddot
from scipy.sparse.linalg import eigsh

# Compute natural frequencies and mode shapes of a 2D plate
mesh = MeshTri.init_sqsymmetric(4)
mesh = mesh.refined(2)
basis = Basis(mesh, ElementTriP2())

E, nu, rho = 70e9, 0.33, 2700.0  # Aluminium

lam = E * nu / ((1 + nu) * (1 - 2*nu))
mu  = E / (2 * (1 + nu))

@BilinearForm
def stiffness(u, v, w):
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    return lam * trace(eps_u) * trace(eps_v) + 2*mu * ddot(eps_u, eps_v)

@BilinearForm
def mass_matrix(u, v, w):
    return rho * (u[0]*v[0] + u[1]*v[1])

K = stiffness.assemble(basis)
M = mass_matrix.assemble(basis)

# Fix all boundaries (clamped plate)
boundary_dofs = basis.get_dofs().all()
# Zero-out rows/cols for clamped DOFs
from skfem import enforce
K_free, _ = enforce(K.copy(), np.zeros(K.shape[0]), D=boundary_dofs, overwrite=True)

# Solve generalized eigenvalue problem K φ = ω² M φ
n_modes = 6
eigenvalues, eigenvectors = eigsh(K_free, M=M, k=n_modes, sigma=0, which='LM')
eigenvalues = np.abs(eigenvalues)  # Remove numerical noise
frequencies_hz = np.sqrt(eigenvalues) / (2 * np.pi)

print("Natural frequencies (clamped plate):")
for i, f in enumerate(frequencies_hz):
    print(f"  Mode {i+1}: {f:.2f} Hz")

# Plot first 4 mode shapes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
x_n, y_n = mesh.p[0], mesh.p[1]
for idx, ax in enumerate(axes.flat):
    # Extract x-displacement of mode shape
    mode_x = eigenvectors[basis.nodal_dofs[0], idx]
    im = ax.tricontourf(x_n, y_n, mesh.t.T, mode_x, levels=20, cmap='RdBu')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Mode {idx+1}: f={frequencies_hz[idx]:.1f} Hz")
    ax.set_aspect('equal')
plt.suptitle("Modal Analysis — Clamped Square Plate")
plt.tight_layout()
plt.savefig("modal_analysis.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Error: `solve` returns NaN or very large values

**Cause**: Singular stiffness matrix — missing or insufficient Dirichlet BCs (rigid body modes).

**Fix**:
```python
# Check condition number
from scipy.sparse.linalg import norm as spnorm
import numpy as np
# Ensure boundary conditions are applied before solving
boundary_dofs = basis.get_dofs().all()
print(f"Fixed DOFs: {len(boundary_dofs)}")  # Must be > 0
```

### Error: `enforce` changes matrix shape unexpectedly

**Fix**: Use `overwrite=True` to modify in-place, or capture the returned (K, F):
```python
K, F = enforce(K, F, D=dofs, overwrite=True)
```

### Poor convergence / large errors

**Cause**: Coarse mesh or low-order elements.

**Fix**:
```python
mesh = mesh.refined(4)   # More refinement levels
basis = Basis(mesh, ElementTriP2())  # Quadratic instead of P1
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| scikit-fem | 9.x | API stable; 9.0 renamed some helpers |
| scipy | 1.11, 1.12 | `eigsh` requires CSR sparse format |
| numpy | 1.24, 1.26 | No known issues |

---

## External Resources

### Official Documentation

- [scikit-fem documentation](https://scikit-fem.readthedocs.io)
- [FEniCSx (DOLFINx)](https://docs.fenicsproject.org)

### Key Textbooks

- Zienkiewicz, O.C. et al. (2005). *The Finite Element Method* (3 volumes). Elsevier.
- Langtangen, H.P. & Logg, A. (2016). *Solving PDEs in Python*. Springer Open.

---

## Examples

### Example 1: Convergence Study — Mesh Refinement

```python
import numpy as np
import matplotlib.pyplot as plt
from skfem import MeshTri, Basis, ElementTriP1, BilinearForm, LinearForm
from skfem import enforce, solve
from skfem.helpers import grad, dot

def solve_poisson_refined(n_refinements):
    """Return L² error for n levels of uniform mesh refinement."""
    mesh = MeshTri.init_sqsymmetric(2)
    for _ in range(n_refinements):
        mesh = mesh.refined()
    basis = Basis(mesh, ElementTriP1())

    @BilinearForm
    def a(u, v, w): return dot(grad(u), grad(v))

    @LinearForm
    def L(v, w):
        x, y = w.x
        return 2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y) * v

    K = a.assemble(basis)
    F = L.assemble(basis)
    K, F = enforce(K, F, D=basis.get_dofs().all())
    u = solve(K, F)

    x_n, y_n = mesh.p[0], mesh.p[1]
    u_ex = np.sin(np.pi * x_n) * np.sin(np.pi * y_n)
    h = 1.0 / (2**n_refinements * 2)
    L2_err = np.sqrt(np.mean((u - u_ex)**2))
    return h, L2_err, mesh.nvertices

refinements = range(1, 7)
results = [solve_poisson_refined(n) for n in refinements]
h_vals  = [r[0] for r in results]
err_vals = [r[1] for r in results]
nodes   = [r[2] for r in results]

# Compute convergence rate
rates = [np.log(err_vals[i]/err_vals[i-1]) / np.log(h_vals[i]/h_vals[i-1])
         for i in range(1, len(err_vals))]
print("Convergence rates:", [f"{r:.2f}" for r in rates])
# Expected: ~2.0 for P1 elements (quadratic convergence in L²)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(h_vals, err_vals, 'bo-', linewidth=2, markersize=8, label='FEM P1 error')
# Reference slope 2
h_ref = np.array([h_vals[0], h_vals[-1]])
ax.loglog(h_ref, 0.5 * h_ref**2, 'r--', label='O(h²) reference')
ax.set_xlabel("Mesh size h"); ax.set_ylabel("L² error")
ax.set_title("FEM Convergence Study — Poisson Equation (P1 Elements)")
ax.legend(); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("fem_convergence.png", dpi=150)
plt.show()
```

### Example 2: Thermal Stress Analysis

```python
import numpy as np
# Thermal stress: σ_thermal = -E*α*ΔT / (1-ν)  (plane stress, uniform ΔT)
# Combined mechanical + thermal loading

E  = 70e9     # Pa — aluminium
nu = 0.33
alpha_T = 23e-6  # /°C — thermal expansion coefficient
delta_T = 100.0  # °C — temperature change

# Thermal strain
eps_thermal = alpha_T * delta_T
print(f"Free thermal strain: {eps_thermal:.4f}")

# Thermal stress if constrained
sigma_thermal = -E * alpha_T * delta_T / (1 - nu)
print(f"Thermal stress (fully constrained): {sigma_thermal/1e6:.1f} MPa")
# Compare with yield strength (~270 MPa for Al 6061)
sigma_y = 270e6
print(f"Yield strength: {sigma_y/1e6:.0f} MPa")
print(f"Safety factor: {sigma_y / abs(sigma_thermal):.2f}")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
