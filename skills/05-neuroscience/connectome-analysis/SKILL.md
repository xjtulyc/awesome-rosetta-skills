---
name: connectome-analysis
description: >
  Use this Skill for functional connectome analysis: connectivity matrices,
  graph metrics (clustering, path length, modularity), rich-club, hub detection.
tags:
  - neuroscience
  - connectomics
  - functional-connectivity
  - bctpy
  - nilearn
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
    - bctpy>=0.5
    - nilearn>=0.10
    - networkx>=3.1
    - matplotlib>=3.7
    - numpy>=1.24
    - pandas>=2.0
    - scipy>=1.11
last_updated: "2026-03-17"
status: "stable"
---

# Connectome Analysis: Functional Brain Network Metrics

> **One-line summary**: Compute functional connectivity matrices, graph-theoretic brain network metrics (clustering, path length, modularity, rich-club), and hub detection using bctpy and nilearn.

---

## When to Use This Skill

- When computing functional connectivity matrices from ROI time series
- When characterizing brain network topology (small-world, scale-free)
- When detecting functional communities/modules in the connectome
- When identifying hub regions with high centrality
- When computing rich-club coefficient for network resilience
- When comparing connectivity patterns across groups (patients vs. controls)

**Trigger keywords**: functional connectivity, connectome, brain network, graph metrics, clustering coefficient, path length, modularity, rich-club, hub detection, bctpy, nilearn

---

## Background & Key Concepts

### Functional Connectivity

FC between ROIs $i$ and $j$ is typically:

$$
FC_{ij} = \text{Pearson}(x_i, x_j) = \frac{\text{cov}(x_i, x_j)}{\sigma_i \sigma_j}
$$

where $x_i$ is the BOLD time series of ROI $i$. The result is a symmetric $N \times N$ FC matrix.

### Graph-Theoretic Metrics

| Metric | Definition | Interpretation |
|:-------|:-----------|:---------------|
| Clustering coefficient | Fraction of triangles around a node | Local segregation |
| Characteristic path length | Average shortest path | Global integration |
| Global efficiency | Average inverse path length | Robustness to damage |
| Modularity Q | Strength of community structure | Functional specialization |
| Rich-club coefficient | Connectivity among high-degree hubs | Core-periphery structure |

### Small-World Property

A network has small-world topology if:

$$
\sigma = \frac{C / C_{rand}}{L / L_{rand}} > 1
$$

where $C$ is clustering coefficient, $L$ is path length, subscript "rand" refers to random graph.

---

## Environment Setup

### Install Dependencies

```bash
pip install bctpy>=0.5 nilearn>=0.10 networkx>=3.1 \
            matplotlib>=3.7 numpy>=1.24 pandas>=2.0 scipy>=1.11
```

### Verify Installation

```python
import bct
import nilearn
import networkx as nx
print(f"bctpy: {bct.__version__}")
print(f"nilearn: {nilearn.__version__}")
print(f"networkx: {nx.__version__}")
```

---

## Core Workflow

### Step 1: Compute Functional Connectivity Matrix

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from nilearn import plotting, datasets

def compute_fc_matrix(roi_timeseries, method="pearson", threshold_pct=None):
    """
    Compute functional connectivity matrix from ROI time series.

    Parameters
    ----------
    roi_timeseries : ndarray, shape (n_timepoints, n_rois)
    method : str
        'pearson', 'partial' (partial correlation via inverse covariance)
    threshold_pct : float or None
        If float, keep top X% of connections (proportional thresholding)

    Returns
    -------
    fc_matrix : ndarray, shape (n_rois, n_rois)
    """
    n_tp, n_rois = roi_timeseries.shape

    if method == "pearson":
        fc_matrix = np.corrcoef(roi_timeseries.T)
    elif method == "partial":
        from sklearn.covariance import LedoitWolf
        prec = LedoitWolf().fit(roi_timeseries).precision_
        # Convert to partial correlation
        D = np.diag(np.sqrt(np.diag(prec)))
        fc_matrix = -np.linalg.inv(D) @ prec @ np.linalg.inv(D)
        np.fill_diagonal(fc_matrix, 1.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    np.fill_diagonal(fc_matrix, 0.0)  # zero diagonal

    if threshold_pct is not None:
        k = int((n_rois * (n_rois - 1) / 2) * threshold_pct / 100)
        upper = fc_matrix[np.triu_indices_from(fc_matrix, k=1)]
        threshold = np.sort(upper)[-k]
        fc_matrix_thr = fc_matrix.copy()
        fc_matrix_thr[fc_matrix < threshold] = 0
        return fc_matrix_thr

    return fc_matrix

# Simulate ROI time series (n_timepoints × n_rois)
rng = np.random.default_rng(42)
n_tp = 200
n_rois = 50

# Community structure: 5 groups of 10 ROIs with high within-group correlation
labels = np.repeat(range(5), 10)
ts = rng.standard_normal((n_tp, n_rois))
for g in range(5):
    idx = np.where(labels == g)[0]
    shared = rng.standard_normal(n_tp)
    ts[:, idx] += 2 * shared[:, None]  # strong common signal within group

ts = (ts - ts.mean(axis=0)) / ts.std(axis=0)  # z-score

# Compute FC
fc = compute_fc_matrix(ts, method="pearson")
fc_thr = compute_fc_matrix(ts, method="pearson", threshold_pct=20)

print(f"FC matrix: {fc.shape}, range [{fc.min():.3f}, {fc.max():.3f}]")
print(f"Mean positive FC: {fc[fc > 0].mean():.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
im0 = axes[0].imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im0, ax=axes[0], label="Pearson r")
axes[0].set_title("Full FC Matrix")
axes[0].set_xlabel("ROI"); axes[0].set_ylabel("ROI")

im1 = axes[1].imshow(fc_thr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im1, ax=axes[1], label="Pearson r")
axes[1].set_title("Thresholded FC (top 20%)")
plt.tight_layout()
plt.savefig("fc_matrix.png", dpi=150)
plt.show()
```

### Step 2: Graph-Theoretic Metrics with BCT

```python
import bct
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Threshold and binarize for graph metrics
fc_pos = np.maximum(fc_thr, 0)  # keep positive connections
fc_bin = (fc_pos > 0).astype(float)  # binary adjacency matrix

print("Computing graph-theoretic metrics...")

# BCT metrics
C_nodal = bct.clustering_coef_bu(fc_bin)  # nodal clustering (binary undirected)
L, eff = bct.charpath(bct.distance_bin(fc_bin))  # path length, efficiency
Q_mod, ci = bct.modularity_und(fc_pos)  # modularity + community assignments

# Degree
degree = fc_bin.sum(axis=1)

# Betweenness centrality
bc = bct.betweenness_bin(fc_bin) / ((fc_bin.shape[0]-1)*(fc_bin.shape[0]-2))

print(f"\nNetwork metrics:")
print(f"  Mean clustering coefficient: {C_nodal.mean():.4f}")
print(f"  Characteristic path length:  {L:.4f}")
print(f"  Global efficiency:           {eff:.4f}")
print(f"  Modularity Q:                {Q_mod:.4f}")
print(f"  Communities detected:        {len(np.unique(ci))}")

# Small-world coefficient (compare to Erdős-Rényi random graph)
n = fc_bin.shape[0]
m = int(fc_bin.sum() / 2)
k_avg = degree.mean()

C_rand = k_avg / n  # approximate
L_rand = np.log(n) / np.log(k_avg)  # approximate

sigma = (C_nodal.mean() / C_rand) / (L / L_rand)
print(f"  Small-world σ:               {sigma:.3f} (>1 = small-world)")

# Plot nodal metrics
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics_data = [
    (degree, "Degree", "steelblue"),
    (C_nodal, "Clustering coefficient", "green"),
    (bc, "Betweenness centrality", "coral"),
]
for ax, (data, label, color) in zip(axes, metrics_data):
    ax.bar(range(n), sorted(data, reverse=True), color=color, alpha=0.7)
    ax.set_xlabel("ROI rank"); ax.set_ylabel(label)
    ax.set_title(label)

plt.tight_layout()
plt.savefig("brain_network_metrics.png", dpi=150)
plt.show()
```

### Step 3: Rich-Club Coefficient and Hub Detection

```python
import bct
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def compute_rich_club(fc_bin, n_rand=100):
    """
    Compute rich-club coefficient and normalize by random networks.

    Returns
    -------
    rc_coeff : ndarray — rich-club coefficients per degree
    rc_norm : ndarray — normalized rich-club (> 1 = rich-club present)
    """
    rc_coeff = bct.rich_club_bu(fc_bin)

    # Random graph normalization
    m = int(fc_bin.sum() / 2)
    n = fc_bin.shape[0]
    rc_rand_all = []
    for _ in range(n_rand):
        G_rand = nx.gnm_random_graph(n, m, seed=None)
        A_rand = nx.to_numpy_array(G_rand, dtype=float)
        rc_rand_all.append(bct.rich_club_bu(A_rand))

    rc_rand_mean = np.mean(rc_rand_all, axis=0)
    rc_norm = rc_coeff / (rc_rand_mean + 1e-10)

    return rc_coeff, rc_norm

rc, rc_norm = compute_rich_club(fc_bin, n_rand=20)

# Hub identification: nodes with high degree AND high betweenness
degree_z = (degree - degree.mean()) / degree.std()
bc_z = (bc - bc.mean()) / bc.std()
hub_score = (degree_z + bc_z) / 2

hub_threshold = 1.0  # z-score threshold
hubs = np.where(hub_score > hub_threshold)[0]
print(f"\nHub regions (degree + betweenness z > {hub_threshold}): {len(hubs)} / {n}")
print(f"Hub indices: {hubs}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Rich-club
degrees = np.arange(1, len(rc)+1)
axes[0].plot(degrees, rc, 'b-o', ms=4, label="Observed RC")
axes[0].plot(degrees, rc_norm, 'r-s', ms=4, label="Normalized RC")
axes[0].axhline(1, color='k', linestyle='--', linewidth=0.8)
axes[0].set_xlabel("Degree threshold k")
axes[0].set_ylabel("Rich-club coefficient")
axes[0].set_title("Rich-Club Coefficient")
axes[0].legend()

# Hub score scatter
axes[1].scatter(degree_z, bc_z, c=hub_score, cmap="YlOrRd", s=50, alpha=0.8)
axes[1].axvline(1, color='r', linestyle='--'); axes[1].axhline(1, color='r', linestyle='--')
for h in hubs:
    axes[1].annotate(f"ROI{h}", (degree_z[h], bc_z[h]), fontsize=7)
axes[1].set_xlabel("Degree z-score")
axes[1].set_ylabel("Betweenness z-score")
axes[1].set_title("Hub Detection")

plt.tight_layout()
plt.savefig("rich_club_hubs.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Group Comparison (Patients vs. Controls)

```python
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def group_fc_comparison(fc_controls, fc_patients, roi_labels=None, alpha=0.05):
    """
    Voxel-wise/ROI-wise two-sample t-test on FC matrices.

    Parameters
    ----------
    fc_controls : ndarray, (n_controls, n_rois, n_rois)
    fc_patients : ndarray, (n_patients, n_rois, n_rois)

    Returns
    -------
    t_matrix, p_matrix, sig_matrix (after FDR correction)
    """
    n_rois = fc_controls.shape[1]
    t_matrix = np.zeros((n_rois, n_rois))
    p_matrix = np.ones((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(i+1, n_rois):
            t, p = stats.ttest_ind(fc_controls[:, i, j], fc_patients[:, i, j])
            t_matrix[i, j] = t_matrix[j, i] = t
            p_matrix[i, j] = p_matrix[j, i] = p

    # FDR correction
    upper_idx = np.triu_indices_from(p_matrix, k=1)
    p_vals_flat = p_matrix[upper_idx]
    _, p_corrected, _, _ = multipletests(p_vals_flat, alpha=alpha, method="fdr_bh")
    sig_matrix = np.zeros_like(p_matrix)
    for k, (i, j) in enumerate(zip(*upper_idx)):
        sig_matrix[i, j] = sig_matrix[j, i] = p_corrected[k] < alpha

    n_sig = int(sig_matrix.sum() / 2)
    print(f"Significant connections (FDR-corrected): {n_sig}")
    return t_matrix, p_matrix, sig_matrix

# Simulate group data
rng = np.random.default_rng(42)
n_ctrl, n_pat = 20, 20
n_rois = 20

fc_ctrl = rng.normal(0, 0.3, (n_ctrl, n_rois, n_rois))
fc_pat  = rng.normal(0, 0.3, (n_pat, n_rois, n_rois))
# Simulate hypoconnectivity in patients (first 5 ROIs)
fc_pat[:, :5, 5:10] -= 0.3
fc_pat[:, 5:10, :5] -= 0.3

t_mat, p_mat, sig_mat = group_fc_comparison(fc_ctrl, fc_pat)
```

---

## Troubleshooting

### Error: `bct.charpath returns inf`

**Cause**: Disconnected graph (path doesn't exist between some nodes).

**Fix**:
```python
# Use only largest connected component
import networkx as nx
G = nx.from_numpy_array(fc_bin)
Gcc = G.subgraph(max(nx.connected_components(G), key=len))
fc_connected = nx.to_numpy_array(Gcc)
```

### Issue: Modularity detects only 1 community

**Cause**: Threshold too high (too sparse) or resolution too low.

**Fix**:
```python
# Use gamma parameter to adjust resolution
Q, ci = bct.modularity_und(fc_pos, gamma=1.5)  # finer communities
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| bctpy | 0.5.0, 0.5.2     | Some BCT functions use 1-indexed arrays |
| nilearn | 0.10, 0.11      | Atlas API stable |

---

## External Resources

### Official Documentation

- [Brain Connectivity Toolbox (BCT)](https://sites.google.com/site/bctnet/)
- [bctpy GitHub](https://github.com/aestrivex/bctpy)
- [nilearn connectivity](https://nilearn.github.io/stable/connectivity/index.html)

### Key Papers

- Rubinov, M. & Sporns, O. (2010). *Complex network measures of brain connectivity: Uses and interpretations*. NeuroImage.

---

## Examples

### Example 1: Resting-State Networks from Nilearn Atlases

```python
# =============================================
# Functional connectivity using Nilearn Power atlas
# =============================================
from nilearn import datasets, input_data, connectome
import numpy as np, matplotlib.pyplot as plt

# Download Power 264 ROI atlas
power = datasets.fetch_coords_power_2011()
coords = np.vstack((power.rois["x"], power.rois["y"], power.rois["z"])).T
print(f"Power atlas: {len(coords)} ROIs")

# Simulate connectivity matrix (replace with real fMRI data)
rng = np.random.default_rng(42)
n_tp = 300
ts = rng.standard_normal((n_tp, len(coords)))
# Add network structure
for net_start in range(0, len(coords), 20):
    net_end = min(net_start + 20, len(coords))
    shared = rng.standard_normal(n_tp)
    ts[:, net_start:net_end] += 1.5 * shared[:, None]

fc = np.corrcoef(ts.T)
np.fill_diagonal(fc, 0)

print(f"FC matrix: {fc.shape}")
print(f"Mean |FC|: {np.abs(fc[np.triu_indices_from(fc, k=1)]).mean():.4f}")
```

**Interpreting these results**: ROIs within the same functional network (default mode, somatomotor, frontoparietal, etc.) show high positive FC. Between-network FC is typically near zero or negative.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
