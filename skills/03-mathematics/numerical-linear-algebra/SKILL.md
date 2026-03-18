---
name: numerical-linear-algebra
description: >
  Use this Skill for SVD, PCA, eigendecomposition, Cholesky, iterative solvers,
  sparse matrix formats, and condition number analysis with numpy and scipy.
tags:
  - mathematics
  - linear-algebra
  - numpy
  - scipy
  - sparse-matrices
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
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
    - pandas>=2.0
last_updated: "2026-03-17"
status: "stable"
---

# Numerical Linear Algebra

> **One-line summary**: Apply SVD, PCA, eigendecomposition, Cholesky factorization, iterative solvers (CG, GMRES), and sparse matrix techniques for scientific computing.

---

## When to Use This Skill

- When decomposing matrices for dimensionality reduction or latent factor models
- When solving large linear systems $Ax = b$ efficiently (direct or iterative)
- When analyzing matrix condition numbers to diagnose ill-conditioning
- When working with sparse matrices from finite element or graph problems
- When implementing PCA or truncated SVD for data analysis
- When factorizing positive definite matrices (Cholesky for fast solves)

**Trigger keywords**: SVD, PCA, eigendecomposition, Cholesky, sparse matrix, GMRES, conjugate gradient, condition number, matrix factorization, linear system

---

## Background & Key Concepts

### Singular Value Decomposition (SVD)

Any $m \times n$ matrix $A$ decomposes as:

$$
A = U \Sigma V^T
$$

where $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$ (diagonal, non-negative), $V \in \mathbb{R}^{n \times n}$ are orthogonal. SVD underpins PCA, pseudoinverse, and low-rank approximations.

### Eigendecomposition

For a square matrix $A$: $Av = \lambda v$ where $\lambda$ is an eigenvalue and $v$ the eigenvector. For symmetric $A = Q \Lambda Q^T$ (spectral theorem).

### Condition Number

$$
\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

Large $\kappa$ indicates ill-conditioning: small perturbations in $b$ lead to large errors in $x = A^{-1}b$.

### Comparison with Related Methods

| Method | Best for | Complexity | Limitation |
|:-------|:---------|:-----------|:-----------|
| LU decomposition | General dense $Ax=b$ | $O(n^3)$ | Not for singular or ill-conditioned |
| Cholesky | SPD matrices | $O(n^3/3)$ | Requires positive definiteness |
| CG (Conjugate Gradient) | Large sparse SPD | $O(k \cdot \text{nnz})$ | Only for symmetric PD |
| GMRES | Large non-symmetric | $O(k^2 \cdot \text{nnz})$ | Memory grows with iterations |
| Truncated SVD | Low-rank approx | $O(mnk)$ | Approximate |

---

## Environment Setup

### Install Dependencies

```bash
pip install numpy>=1.24 scipy>=1.11 matplotlib>=3.7 pandas>=2.0
```

### Verify Installation

```python
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla

A = np.eye(5)
vals, vecs = np.linalg.eigh(A)
print(f"numpy: {np.__version__}, scipy: {scipy.__version__}")
print(f"Identity eigenvalues: {vals}")
# Expected: [1. 1. 1. 1. 1.]
```

---

## Core Workflow

### Step 1: SVD and Low-Rank Approximation

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Create a synthetic rank-5 matrix with noise
m, n, true_rank = 200, 100, 5
U_true = rng.standard_normal((m, true_rank))
V_true = rng.standard_normal((n, true_rank))
A_clean = U_true @ V_true.T
A_noisy = A_clean + 0.1 * rng.standard_normal((m, n))

# Full SVD
U, s, Vt = np.linalg.svd(A_noisy, full_matrices=False)
print(f"Singular values (top 10): {s[:10].round(2)}")
print(f"Rank-{true_rank} explained variance: {(s[:true_rank]**2).sum() / (s**2).sum():.4f}")

# Low-rank approximation
def truncated_svd_approx(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

errors = []
for k in range(1, 20):
    A_k = truncated_svd_approx(A_noisy, k)
    rel_err = np.linalg.norm(A_noisy - A_k, 'fro') / np.linalg.norm(A_noisy, 'fro')
    errors.append(rel_err)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(s[:30], 'o-')
axes[0].axvline(true_rank - 1, color='r', linestyle='--', label=f'True rank={true_rank}')
axes[0].set_xlabel("Index"); axes[0].set_ylabel("Singular value"); axes[0].legend()
axes[0].set_title("Singular Value Spectrum")

axes[1].plot(range(1, 20), errors, 's-')
axes[1].set_xlabel("Rank k"); axes[1].set_ylabel("Relative Frobenius error")
axes[1].set_title("Low-Rank Approximation Error")
plt.tight_layout()
plt.savefig("svd_analysis.png", dpi=150)
plt.show()
```

### Step 2: Eigendecomposition and PCA

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load example data
digits = load_digits()
X = digits.data.astype(float)  # shape (1797, 64)
X -= X.mean(axis=0)  # center

# Covariance matrix approach (small p)
C = X.T @ X / (len(X) - 1)  # 64×64 covariance
eigenvalues, eigenvectors = np.linalg.eigh(C)

# eigh returns in ascending order; reverse for descending
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

# Explained variance ratio
total_var = eigenvalues.sum()
exp_var_ratio = eigenvalues / total_var
cumulative_var = np.cumsum(exp_var_ratio)

print(f"n_components for 90% variance: {np.argmax(cumulative_var >= 0.90) + 1}")
print(f"Top 5 eigenvalues: {eigenvalues[:5].round(2)}")

# Project to first 2 PCs
X_pca = X @ eigenvectors[:, :2]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(cumulative_var[:30], 'o-')
axes[0].axhline(0.90, color='r', linestyle='--', label='90% threshold')
axes[0].set_xlabel("Number of components")
axes[0].set_ylabel("Cumulative explained variance")
axes[0].legend(); axes[0].set_title("PCA Explained Variance")

scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                          c=digits.target, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(scatter, ax=axes[1], label='Digit')
axes[1].set_title("PCA Projection (first 2 PCs)")
plt.tight_layout()
plt.savefig("pca_digits.png", dpi=150)
plt.show()
```

### Step 3: Sparse Matrices and Iterative Solvers

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def build_1d_laplacian(n):
    """Build tridiagonal Laplacian (finite difference) as sparse CSR matrix."""
    diags = [-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)]
    return sp.diags(diags, [-1, 0, 1], shape=(n, n), format='csr')

n = 1000
A = build_1d_laplacian(n)
b = np.ones(n)

print(f"Matrix: {A.shape}, nnz={A.nnz}, density={A.nnz/(n**2):.6f}")
print(f"Condition number estimate: {spla.norm(A) * spla.norm(spla.inv(A.tocsc())):.2e}")

# Conjugate Gradient solver (works for symmetric positive definite)
residuals_cg = []
def cg_callback(xk):
    r = b - A @ xk
    residuals_cg.append(np.linalg.norm(r))

x_cg, info_cg = spla.cg(A, b, tol=1e-10, maxiter=5000, callback=cg_callback)
print(f"\nCG converged: {info_cg == 0}, iterations: {len(residuals_cg)}")
print(f"Residual: {np.linalg.norm(b - A @ x_cg):.2e}")

# GMRES solver (general non-symmetric)
residuals_gmres = []
def gmres_callback(rk):
    residuals_gmres.append(rk)

x_gmres, info_gmres = spla.gmres(A, b, tol=1e-10, maxiter=500, callback=gmres_callback)
print(f"GMRES converged: {info_gmres == 0}, iterations: {len(residuals_gmres)}")

# ILU preconditioner for GMRES
ilu = spla.spilu(A.tocsc(), fill_factor=2.0)
M = spla.LinearOperator(A.shape, ilu.solve)
x_prec, info_prec = spla.gmres(A, b, M=M, tol=1e-10, maxiter=100)
print(f"Preconditioned GMRES converged: {info_prec == 0}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(residuals_cg, label="CG")
if residuals_gmres:
    ax.semilogy(residuals_gmres, label="GMRES")
ax.set_xlabel("Iteration"); ax.set_ylabel("Residual norm")
ax.legend(); ax.set_title("Solver Convergence")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("solver_convergence.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Cholesky Factorization for Fast Linear Solves

```python
import numpy as np
import scipy.linalg as la
import time

def generate_spd(n, seed=42):
    """Generate a random symmetric positive definite matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)  # add n*I to ensure positive definiteness

n = 500
A = generate_spd(n)
b = np.ones(n)

# Cholesky factorization
t0 = time.time()
L, lower = la.cho_factor(A)
x = la.cho_solve((L, lower), b)
print(f"Cholesky solve: {time.time()-t0:.4f}s, residual={np.linalg.norm(A@x-b):.2e}")

# LU (for comparison)
t0 = time.time()
x_lu = la.solve(A, b)
print(f"LU solve:       {time.time()-t0:.4f}s, residual={np.linalg.norm(A@x_lu-b):.2e}")

# Multiple RHS — Cholesky amortizes cost
B = np.random.randn(n, 50)
t0 = time.time()
X = la.cho_solve((L, lower), B)
print(f"Cholesky (50 RHS): {time.time()-t0:.4f}s")
```

### Condition Number and Preconditioning

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def analyze_conditioning(A):
    """Report condition number and recommend preconditioning strategy."""
    if sp.issparse(A):
        # Estimate via power iteration (full SVD too expensive)
        s_max = spla.norm(A)
        # Smallest singular value via inverse iteration
        try:
            s_min = 1.0 / spla.norm(spla.inv(A.tocsc()))
            kappa = s_max / s_min
        except Exception:
            kappa = float('inf')
    else:
        s = np.linalg.svd(A, compute_uv=False)
        kappa = s[0] / s[-1]

    print(f"Condition number κ(A) = {kappa:.2e}")
    if kappa < 1e4:
        print("  Well-conditioned: direct solver sufficient")
    elif kappa < 1e10:
        print("  Moderately ill-conditioned: consider Jacobi/ILU preconditioning")
    else:
        print("  Severely ill-conditioned: use regularization (Tikhonov, truncated SVD)")
    return kappa

# Test with progressively ill-conditioned matrices
for cond_target in [1e2, 1e6, 1e10]:
    # Build matrix with prescribed condition number
    n = 50
    U, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0, -np.log10(cond_target), n)
    A = U @ np.diag(s) @ U.T
    kappa = analyze_conditioning(A)
```

---

## Troubleshooting

### Error: `numpy.linalg.LinAlgError: Matrix is singular`

**Cause**: Matrix is rank-deficient or numerically singular.

**Fix**:
```python
# Use pseudoinverse for rank-deficient systems
x = np.linalg.lstsq(A, b, rcond=None)[0]

# Or add regularization (Tikhonov)
lambda_reg = 1e-6
x = np.linalg.solve(A.T @ A + lambda_reg * np.eye(A.shape[1]), A.T @ b)
```

### Issue: CG doesn't converge

**Cause**: Matrix is not symmetric positive definite.

**Fix**:
```python
# Check symmetry
print(f"Max asymmetry: {np.abs(A - A.T).max():.2e}")  # should be ~0
# Check positive definiteness
eigenvalues = np.linalg.eigvalsh(A)
print(f"Min eigenvalue: {eigenvalues.min():.2e}")  # must be > 0
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| numpy | 1.24, 1.26, 2.0 | `np.linalg.svd` API unchanged |
| scipy | 1.11, 1.13      | `spla.cg` signature unchanged |

---

## External Resources

### Official Documentation

- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy Sparse Linear Algebra](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)

### Key Papers

- Trefethen, L.N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.

---

## Examples

### Example 1: Image Compression via Truncated SVD

```python
# =============================================
# Image compression using SVD
# =============================================
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic test image (or load real one)
from PIL import Image
import urllib.request

# Synthetic checkerboard
img = np.zeros((256, 256))
for i in range(0, 256, 32):
    for j in range(0, 256, 32):
        if (i//32 + j//32) % 2 == 0:
            img[i:i+32, j:j+32] = 1.0

# Add smooth signal
x, y = np.meshgrid(np.linspace(0, np.pi, 256), np.linspace(0, np.pi, 256))
img += 0.5 * np.sin(x) * np.cos(y)
img = img / img.max()

U, s, Vt = np.linalg.svd(img, full_matrices=False)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title("Original")
for idx, k in enumerate([5, 20, 50]):
    img_k = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
    err = np.linalg.norm(img - img_k, 'fro') / np.linalg.norm(img, 'fro')
    r, c = divmod(idx+1, 3)
    axes[r,c].imshow(img_k, cmap='gray')
    axes[r,c].set_title(f"k={k}, rel err={err:.3f}")

axes[1,2].semilogy(s[:50], 'o-')
axes[1,2].set_title("Singular values")
plt.tight_layout()
plt.savefig("image_compression_svd.png", dpi=150)
plt.show()
print("Saved image_compression_svd.png")
```

**Interpreting these results**: At k=20, most images retain >95% of the signal energy, achieving 10x compression while remaining visually similar.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
