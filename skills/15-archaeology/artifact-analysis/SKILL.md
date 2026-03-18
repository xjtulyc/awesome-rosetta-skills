---
name: artifact-analysis
description: >
  Use this Skill for artifact analysis: typological classification, ceramic
  attribute coding, lithic reduction analysis, and correspondence analysis.
tags:
  - archaeology
  - artifact-analysis
  - typology
  - ceramics
  - lithics
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
    - pandas>=2.0
    - numpy>=1.24
    - scipy>=1.11
    - scikit-learn>=1.3
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# Artifact Analysis

> **One-line summary**: Code, classify, and statistically analyze archaeological artifacts: ceramic typology via correspondence analysis, lithic reduction sequences, and attribute cluster analysis.

---

## When to Use This Skill

- When coding ceramic sherds by fabric, form, decoration, and period
- When performing correspondence analysis on ceramic assemblages
- When analyzing lithic reduction sequences and debitage patterns
- When classifying artifacts using hierarchical or k-means clustering
- When building attribute databases for comparative analysis
- When computing diversity indices for assemblage comparison

**Trigger keywords**: artifact analysis, ceramic analysis, typology, lithic analysis, debitage, ceramic sherds, correspondence analysis, attribute coding, assemblage diversity, pottery classification, chaîne opératoire, flintknapping, cluster analysis

---

## Background & Key Concepts

### Correspondence Analysis (CA)

Exploratory multivariate technique for contingency tables. Decomposes the chi-squared distance between rows and columns of a frequency table into principal axes:

$$
\chi^2 = N \sum_{ij} \frac{(p_{ij} - p_{i+}p_{+j})^2}{p_{i+}p_{+j}}
$$

Biplot shows rows (types) and columns (attributes) in low-dimensional space, revealing typological patterns.

### Diversity Indices for Assemblages

- **Richness**: number of distinct types
- **Shannon H**: $H = -\sum_i p_i \ln p_i$
- **Simpson D**: $D = 1 - \sum_i p_i^2$
- **Evenness (Pielou J)**: $J = H / \ln(S)$

### Lithic Reduction Stages

Macroscopic attributes indicating reduction stage:
1. Primary flake (>50% cortex)
2. Secondary flake (1-50% cortex)
3. Interior flake (no cortex)
4. Blade, biface thinning flake, tool

---

## Environment Setup

### Install Dependencies

```bash
pip install pandas>=2.0 numpy>=1.24 scipy>=1.11 scikit-learn>=1.3 matplotlib>=3.7
# Optional: prince for correspondence analysis
pip install prince>=0.13
```

### Verify Installation

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Quick test with synthetic ceramic data
np.random.seed(42)
test_data = pd.DataFrame({
    'site':  ['Site A']*50 + ['Site B']*50,
    'ware':  np.random.choice(['Local','Imported','Coarseware'], 100),
})
ct = pd.crosstab(test_data['site'], test_data['ware'])
chi2, p, dof, expected = chi2_contingency(ct)
print(f"Chi-sq test: chi2={chi2:.3f}, p={p:.4f}")
print("Environment setup: OK")
```

---

## Core Workflow

### Step 1: Ceramic Attribute Coding and Database

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ------------------------------------------------------------------ #
# Simulate ceramic sherd database from multiple sites/contexts
# ------------------------------------------------------------------ #

np.random.seed(42)
n_sherds = 500

# Attribute coding (typical recording sheet variables)
data = {
    'sherd_id': [f"SH{i:04d}" for i in range(n_sherds)],
    'site': np.random.choice(['Tell A', 'Tell B', 'Tell C', 'Tell D'], n_sherds,
                              p=[0.35, 0.30, 0.20, 0.15]),
    'context': np.random.choice(['midden', 'floor', 'pit', 'fill'], n_sherds,
                                 p=[0.40, 0.25, 0.20, 0.15]),
    'ware': np.random.choice(['Local plain', 'Local painted', 'Imported fine',
                               'Coarseware', 'Pithos'], n_sherds,
                              p=[0.35, 0.25, 0.15, 0.20, 0.05]),
    'form': np.random.choice(['bowl', 'jar', 'jug', 'platter', 'flask', 'unknown'], n_sherds,
                              p=[0.30, 0.25, 0.20, 0.10, 0.05, 0.10]),
    'decoration': np.random.choice(['plain', 'geometric', 'floral', 'incised', 'none'], n_sherds,
                                    p=[0.30, 0.25, 0.20, 0.15, 0.10]),
    'firing': np.random.choice(['well-fired', 'underfired', 'overfired'], n_sherds,
                                p=[0.65, 0.20, 0.15]),
    'wall_thickness_mm': np.random.normal(8, 2, n_sherds).clip(3, 20),
    'weight_g': np.random.lognormal(3, 0.8, n_sherds),
    'period': np.random.choice(['Early Bronze', 'Middle Bronze', 'Late Bronze', 'Iron Age'], n_sherds,
                                p=[0.20, 0.30, 0.30, 0.20]),
    'has_soot': np.random.choice([True, False], n_sherds, p=[0.20, 0.80]),
}

df = pd.DataFrame(data)

print(f"Ceramic database: {len(df)} sherds from {df['site'].nunique()} sites")
print(f"\nWare distribution:")
print(df['ware'].value_counts())

# ---- Ware × Site cross-tabulation ------------------------------ #
ct_ware_site = pd.crosstab(df['site'], df['ware'])
chi2, p_chi2, dof, expected = chi2_contingency(ct_ware_site)
print(f"\nWare distribution varies by site? χ²={chi2:.2f}, df={dof}, p={p_chi2:.4f}")

# ---- Assemblage diversity by site ------------------------------ #
def shannon_diversity(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def evenness(counts):
    H = shannon_diversity(counts)
    S = (counts > 0).sum()
    return H / np.log(S) if S > 1 else 0

div_stats = []
for site_name, group in df.groupby('site'):
    ware_counts = group['ware'].value_counts()
    div_stats.append({
        'site': site_name,
        'n_sherds': len(group),
        'richness': len(ware_counts),
        'shannon_H': shannon_diversity(ware_counts),
        'evenness_J': evenness(ware_counts),
        'mean_thickness': group['wall_thickness_mm'].mean(),
    })

div_df = pd.DataFrame(div_stats)
print("\nAssemblage diversity by site:")
print(div_df.round(3).to_string(index=False))

# ---- Visualization --------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Stacked bar: ware by site
ct_pct = ct_ware_site.div(ct_ware_site.sum(axis=1), axis=0) * 100
ct_pct.plot(kind='bar', stacked=True, ax=axes[0][0], cmap='tab10', edgecolor='black', linewidth=0.5)
axes[0][0].set_title("Ware Composition by Site (%)"); axes[0][0].set_xlabel("Site")
axes[0][0].set_ylabel("Percentage"); axes[0][0].legend(fontsize=7, loc='upper right')
axes[0][0].tick_params(axis='x', rotation=20)

# Diversity indices
x = np.arange(len(div_df))
axes[0][1].bar(x - 0.2, div_df['shannon_H'], width=0.35, label="Shannon H",
               color='steelblue', edgecolor='black', linewidth=0.7)
axes[0][1].bar(x + 0.2, div_df['evenness_J'], width=0.35, label="Evenness J",
               color='coral', edgecolor='black', linewidth=0.7)
axes[0][1].set_xticks(x); axes[0][1].set_xticklabels(div_df['site'])
axes[0][1].set_title("Assemblage Diversity by Site"); axes[0][1].legend()
axes[0][1].grid(axis='y', alpha=0.3)

# Wall thickness by ware
df.boxplot(column='wall_thickness_mm', by='ware', ax=axes[1][0], grid=True)
axes[1][0].set_title("Wall Thickness by Ware"); axes[1][0].set_xlabel("Ware type")
axes[1][0].set_ylabel("Thickness (mm)"); plt.sca(axes[1][0]); plt.xticks(rotation=20, fontsize=8)

# Period distribution by site
ct_period = pd.crosstab(df['site'], df['period'])
ct_period.plot(kind='bar', ax=axes[1][1], cmap='viridis', edgecolor='black', linewidth=0.5)
axes[1][1].set_title("Chronological Distribution by Site")
axes[1][1].set_xlabel("Site"); axes[1][1].set_ylabel("Sherd count")
axes[1][1].tick_params(axis='x', rotation=20); axes[1][1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("ceramic_analysis.png", dpi=150)
plt.show()
```

### Step 2: Correspondence Analysis of Ceramic Types

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

# ------------------------------------------------------------------ #
# Correspondence Analysis (CA) on ceramics contingency table
# Reveals associations between sites/contexts and ware types
# ------------------------------------------------------------------ #

def correspondence_analysis(contingency_table):
    """
    Compute correspondence analysis.
    Returns: row_coords, col_coords, explained_variance (per axis)
    """
    N = contingency_table.values.astype(float)
    n = N.sum()
    P = N / n  # Relative frequencies

    row_masses = P.sum(axis=1)  # r
    col_masses = P.sum(axis=0)  # c

    # Expected frequencies
    E = np.outer(row_masses, col_masses)

    # Standardized residual matrix
    S = (P - E) / np.sqrt(E)

    # SVD
    U, d, Vt = svd(S, full_matrices=False)

    # Row coordinates (standard)
    row_coords = np.diag(1/np.sqrt(row_masses)) @ U @ np.diag(d)

    # Column coordinates (standard)
    col_coords = np.diag(1/np.sqrt(col_masses)) @ Vt.T @ np.diag(d)

    # Explained inertia
    inertia = d**2
    explained = inertia / inertia.sum() * 100

    return row_coords[:, :2], col_coords[:, :2], explained[:2]

# Use ware × context cross-tabulation
ct_ca = pd.crosstab(df['context'], df['ware'])
print("Contingency table (context × ware):")
print(ct_ca)

row_coords, col_coords, explained = correspondence_analysis(ct_ca)

print(f"\nCA explained variance: Axis 1 = {explained[0]:.1f}%, Axis 2 = {explained[1]:.1f}%")

# ---- Biplot ----------------------------------------------------- #
fig, ax = plt.subplots(figsize=(9, 7))

# Row points (contexts) — squares
ax.scatter(row_coords[:, 0], row_coords[:, 1], s=200, marker='s',
           c='#e74c3c', edgecolors='black', linewidths=0.8, zorder=5, label='Contexts')
for i, label in enumerate(ct_ca.index):
    ax.annotate(label, (row_coords[i, 0], row_coords[i, 1]),
                fontsize=11, fontweight='bold', color='#e74c3c',
                xytext=(5, 5), textcoords='offset points')

# Column points (ware types) — circles
ax.scatter(col_coords[:, 0], col_coords[:, 1], s=150, marker='o',
           c='#3498db', edgecolors='black', linewidths=0.8, zorder=5, label='Ware types')
for i, label in enumerate(ct_ca.columns):
    ax.annotate(label, (col_coords[i, 0], col_coords[i, 1]),
                fontsize=9, color='#3498db',
                xytext=(5, -10), textcoords='offset points')

ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel(f"CA Axis 1 ({explained[0]:.1f}%)")
ax.set_ylabel(f"CA Axis 2 ({explained[1]:.1f}%)")
ax.set_title("Correspondence Analysis: Ceramic Context × Ware Type")
ax.legend(loc='upper right'); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("correspondence_analysis.png", dpi=150)
plt.show()
```

### Step 3: Lithic Analysis — Reduction Sequence and Debitage

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal

# ------------------------------------------------------------------ #
# Lithic assemblage analysis: reduction sequence from debitage
# ------------------------------------------------------------------ #

np.random.seed(42)
n_lithics = 400

# Flake attributes (simulated)
lithics_data = {
    'artifact_id': [f"L{i:04d}" for i in range(n_lithics)],
    'site': np.random.choice(['Area 1', 'Area 2', 'Area 3'], n_lithics, p=[0.5, 0.3, 0.2]),
    'material': np.random.choice(['Flint', 'Obsidian', 'Chert', 'Quartzite'], n_lithics,
                                  p=[0.50, 0.20, 0.25, 0.05]),
    'type': np.random.choice(['Primary flake', 'Secondary flake', 'Interior flake',
                               'Blade', 'Core', 'Biface', 'Tool'], n_lithics,
                              p=[0.15, 0.25, 0.30, 0.10, 0.05, 0.05, 0.10]),
    'cortex_pct': np.random.choice([0, 10, 25, 50, 75, 100], n_lithics),
    'length_mm': np.random.lognormal(3.2, 0.5, n_lithics),
    'width_mm':  np.random.lognormal(2.9, 0.5, n_lithics),
    'thickness_mm': np.random.lognormal(1.8, 0.5, n_lithics),
    'platform': np.random.choice(['cortical', 'plain', 'faceted', 'punctiform', 'absent'], n_lithics,
                                   p=[0.15, 0.40, 0.25, 0.10, 0.10]),
    'flake_scars': np.random.randint(0, 15, n_lithics),
    'retouched': np.random.choice([True, False], n_lithics, p=[0.15, 0.85]),
}
lithics_df = pd.DataFrame(lithics_data)

# Compute reduction index (proxy: log weight ÷ dorsal scar count)
lithics_df['weight_g'] = (lithics_df['length_mm'] * lithics_df['width_mm'] * lithics_df['thickness_mm']) * 0.002
lithics_df['reduction_index'] = (lithics_df['flake_scars'] + 1) / (lithics_df['cortex_pct']/100 + 0.1)

print(f"Lithic assemblage: {len(lithics_df)} artifacts")
print("\nType distribution:")
print(lithics_df['type'].value_counts())

# ---- Reduction stage by cortex percentage ---------------------- #
cortex_by_type = lithics_df.groupby('type')['cortex_pct'].describe()
print("\nCortex % by artifact type:")
print(cortex_by_type[['mean','std','50%']].round(1))

# ---- Platform × flake type association ------------------------- #
ct_plat = pd.crosstab(lithics_df['type'], lithics_df['platform'])
from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(ct_plat)
print(f"\nPlatform × type: χ²={chi2:.2f}, p={p:.4f}")

# ---- Metric analysis by material -------------------------------- #
# Kruskal-Wallis test for length differences by material
materials = lithics_df['material'].unique()
groups = [lithics_df[lithics_df['material']==m]['length_mm'].values for m in materials]
h_stat, p_kw = kruskal(*groups)
print(f"\nLength by material (Kruskal-Wallis): H={h_stat:.2f}, p={p_kw:.4f}")

# ---- Visualization --------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Type distribution
type_counts = lithics_df['type'].value_counts()
colors_lit = ['#8B4513','#D2691E','#A0522D','#CD853F','#DEB887','#F4A460','#FFDEAD']
axes[0][0].barh(type_counts.index, type_counts.values, color=colors_lit[:len(type_counts)],
                edgecolor='black', linewidth=0.7)
axes[0][0].set_xlabel("Count"); axes[0][0].set_title("Lithic Type Distribution")
axes[0][0].grid(axis='x', alpha=0.3)

# Reduction sequence: cortex by type
lithics_df.boxplot(column='cortex_pct', by='type', ax=axes[0][1])
axes[0][1].set_title("Cortex % by Artifact Type (Reduction Stage)")
axes[0][1].set_xlabel("Artifact type"); axes[0][1].set_ylabel("Cortex %")
plt.sca(axes[0][1]); plt.xticks(rotation=30, fontsize=7)

# Length distribution by material
for mat in materials:
    subset = lithics_df[lithics_df['material']==mat]['length_mm']
    axes[1][0].hist(subset, bins=20, alpha=0.5, density=True, label=mat)
axes[1][0].set_xlabel("Length (mm)"); axes[1][0].set_ylabel("Density")
axes[1][0].set_title("Flake Length Distribution by Raw Material")
axes[1][0].legend(fontsize=8); axes[1][0].grid(True, alpha=0.3)

# Length × width scatter (flake shape)
axes[1][1].scatter(lithics_df['length_mm'], lithics_df['width_mm'],
                   c=['#3498db' if r else '#e74c3c' for r in lithics_df['retouched']],
                   s=15, alpha=0.5)
axes[1][1].set_xlabel("Length (mm)"); axes[1][1].set_ylabel("Width (mm)")
axes[1][1].set_title("Flake Dimensions (blue=unretouched, red=retouched)")
axes[1][1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lithic_analysis.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Hierarchical Cluster Analysis of Assemblages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Build site-level attribute matrix (ware proportions)
ct_site_ware = pd.crosstab(df['site'], df['ware'], normalize='index')

# Bray-Curtis dissimilarity (for compositional data)
try:
    from scipy.spatial.distance import braycurtis
    bc_matrix = np.zeros((len(ct_site_ware), len(ct_site_ware)))
    for i in range(len(ct_site_ware)):
        for j in range(len(ct_site_ware)):
            bc_matrix[i,j] = braycurtis(ct_site_ware.iloc[i], ct_site_ware.iloc[j])
except Exception:
    bc_matrix = pdist(ct_site_ware.values, metric='euclidean')

from scipy.spatial.distance import squareform
try:
    dist_condensed = squareform(bc_matrix)
except Exception:
    dist_condensed = pdist(ct_site_ware.values, metric='euclidean')

Z = linkage(dist_condensed, method='ward')

fig, ax = plt.subplots(figsize=(8, 5))
dendrogram(Z, labels=ct_site_ware.index.tolist(), ax=ax)
ax.set_title("Assemblage Cluster Analysis (Ward linkage, Bray-Curtis dissimilarity)")
ax.set_xlabel("Site"); ax.set_ylabel("Dissimilarity")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig("assemblage_cluster.png", dpi=150); plt.show()
```

---

## Troubleshooting

### CA axes flip sign unexpectedly

SVD sign is arbitrary. Multiply both row and column coordinates by -1 for a given axis if needed for interpretability — the relative positions remain the same.

### Chi-square test invalid with small expected frequencies

**Fix**: Merge rare categories before testing:
```python
# Merge rare wares (<10 counts) into "Other"
rare = ware_counts[ware_counts < 10].index
df['ware_merged'] = df['ware'].replace({w: 'Other' for w in rare})
```

### Outlier sherds distort CA

**Fix**: Remove outlier rows/columns with very low frequencies before CA:
```python
ct_filtered = ct_ca[ct_ca.sum(axis=1) >= 5]  # Min 5 sherds per context
ct_filtered = ct_filtered.loc[:, ct_filtered.sum(axis=0) >= 5]
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| pandas | 2.0, 2.1 | `crosstab` normalize parameter stable |
| scipy | 1.11, 1.12 | `chi2_contingency`, `kruskal` stable |
| scikit-learn | 1.3, 1.4 | For k-means and hierarchical clustering |

---

## External Resources

### Official Documentation

- [scipy hierarchical clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

### Key Papers / Books

- Rice, P.M. (1987). *Pottery Analysis: A Sourcebook*. University of Chicago Press.
- Andrefsky, W. (2005). *Lithics: Macroscopic Approaches to Analysis*. Cambridge University Press.
- Shennan, S. (1988). *Quantifying Archaeology*. Edinburgh University Press.

---

## Examples

### Example 1: Rim Diameter Reconstruction from Arc Measurement

```python
import numpy as np
import matplotlib.pyplot as plt

# Reconstruct rim diameter from arc chord length and height (tangent method)
def rim_diameter(chord_length_mm, arc_height_mm):
    """Compute diameter from chord and sagitta (standard ceramics method)."""
    r = (chord_length_mm**2 / (8 * arc_height_mm)) + arc_height_mm / 2
    return 2 * r

# Example measurements
examples = [(80, 8), (120, 12), (60, 5), (100, 15)]
for chord, sagitta in examples:
    diam = rim_diameter(chord, sagitta)
    print(f"Chord={chord}mm, Sagitta={sagitta}mm → Diameter={diam:.1f}mm")
```

### Example 2: Harris Matrix (Stratigraphic Sequence)

```python
import networkx as nx
import matplotlib.pyplot as plt

# Build Harris matrix from stratigraphic relationships
G = nx.DiGraph()
# (A, B) means A is above / later than B
relationships = [
    ('Topsoil', 'Layer 1'), ('Layer 1', 'Layer 2'), ('Layer 2', 'Pit 1'),
    ('Layer 2', 'Layer 3'), ('Pit 1', 'Layer 3'), ('Layer 3', 'Natural'),
]
G.add_edges_from(relationships)

pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw_networkx(G, pos, ax=ax, node_size=1500, node_color='lightyellow',
                 arrows=True, arrowsize=20, font_size=9, edge_color='gray')
ax.set_title("Harris Matrix — Stratigraphic Sequence"); ax.axis('off')
plt.tight_layout(); plt.savefig("harris_matrix.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
