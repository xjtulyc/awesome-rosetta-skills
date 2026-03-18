---
name: regional-economics
description: Regional economic analysis with shift-share, location quotients, input-output models, and spatial econometrics for research and policy.
tags:
  - regional-economics
  - shift-share
  - input-output
  - location-quotient
  - spatial-econometrics
version: "1.0.0"
authors:
  - "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - numpy>=1.24
    - pandas>=2.0
    - scipy>=1.11
    - statsmodels>=0.14
    - matplotlib>=3.7
    - geopandas>=0.14
last_updated: "2026-03-17"
status: stable
---

# Regional Economics Analysis

## When to Use This Skill

Use this skill when you need to:
- Analyze regional economic structure and competitive advantage
- Decompose employment/output changes into national, industry, and competitive effects
- Measure industrial specialization and diversification
- Build regional input-output tables and compute multipliers
- Estimate spatial econometric models for regional data
- Assess regional convergence and divergence patterns

**Trigger keywords**: shift-share analysis, location quotient, economic base theory, input-output model, regional multiplier, Krugman specialization, beta convergence, sigma convergence, spatial lag model, spatial error model, regional growth, industrial cluster, agglomeration economies, regional GDP, interregional trade.

## Background & Key Concepts

### Location Quotient

The location quotient (LQ) measures industry $i$'s relative concentration in region $r$ compared to the national average:

$$LQ_{ir} = \frac{e_{ir}/E_r}{e_{in}/E_n}$$

where $e_{ir}$ is employment in industry $i$, region $r$; $E_r$ is total regional employment; $e_{in}$ and $E_n$ are corresponding national totals. $LQ > 1$ implies specialization; $LQ > 1.25$ is often used as an export-base threshold.

### Shift-Share Analysis

The classic Dunn (1960) decomposition splits regional employment change $\Delta e_{ir}$ into three components:

$$\Delta e_{ir} = \underbrace{e_{ir} \cdot g_n}_{\text{National Share}} + \underbrace{e_{ir}(g_{in} - g_n)}_{\text{Industry Mix}} + \underbrace{e_{ir}(g_{ir} - g_{in})}_{\text{Competitive Effect}}$$

where $g_n$, $g_{in}$, $g_{ir}$ are national, national-industry, and regional-industry growth rates.

### Economic Base Multiplier

Basic sector employment $B$ drives total employment $T$ via multiplier $k$:

$$k = \frac{T}{B}, \quad B = \sum_i \max\!\left(e_{ir} - \frac{e_{in}}{E_n} E_r,\ 0\right)$$

### Input-Output Leontief Model

Given technical coefficient matrix $\mathbf{A}$ (where $a_{ij} = z_{ij}/X_j$):

$$\mathbf{X} = (\mathbf{I} - \mathbf{A})^{-1} \mathbf{f}$$

The Leontief inverse $\mathbf{L} = (\mathbf{I}-\mathbf{A})^{-1}$ gives total output multipliers. Output multiplier for sector $j$: $m_j = \sum_i L_{ij}$.

### Krugman Specialization Index

$$K_{rs} = \frac{1}{2}\sum_i \left|\frac{e_{ir}}{E_r} - \frac{e_{is}}{E_s}\right| \in [0,1]$$

$K=0$: identical structure; $K=1$: completely different.

### Beta-Convergence

Absolute $\beta$-convergence regression:

$$\frac{1}{T}\ln\frac{y_{r,T}}{y_{r,0}} = \alpha + \beta \ln y_{r,0} + \varepsilon_r$$

$\beta < 0$ indicates poorer regions grow faster (convergence). Implied convergence speed: $\lambda = -\ln(1+\beta)/T$.

## Environment Setup

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 statsmodels>=0.14 \
            matplotlib>=3.7 geopandas>=0.14
```

```python
import numpy as np
import pandas as pd
import scipy.linalg as la
import statsmodels.api as sm
import matplotlib.pyplot as plt
print("Regional economics environment ready")
```

## Core Workflow

### Step 1: Location Quotients and Shift-Share Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate regional employment data
# 8 regions, 6 industries, 2 time periods (t0, t1)
# -----------------------------------------------------------------
np.random.seed(42)
n_regions = 8
n_industries = 6
region_names = [f"Region_{r}" for r in "ABCDEFGH"]
industry_names = ["Manufacturing", "Finance", "Retail",
                  "Construction", "Healthcare", "Tech"]

# Base employment (t0)
emp_t0 = np.random.randint(500, 5000, (n_regions, n_industries)).astype(float)
# National industry growth rates (differential by industry)
nat_industry_growth = np.array([0.02, 0.08, 0.03, 0.04, 0.06, 0.12])
# Regional competitive advantages (slight variation around national)
comp_effect = np.random.uniform(-0.03, 0.05, (n_regions, n_industries))
# t1 employment
emp_t1 = emp_t0 * (1 + nat_industry_growth + comp_effect)

# -----------------------------------------------------------------
# Location Quotient
# -----------------------------------------------------------------
def location_quotient(emp_matrix):
    """Compute LQ for each region-industry cell.

    Args:
        emp_matrix: (n_regions, n_industries) array
    Returns:
        lq: same-shape array of location quotients
    """
    total_regional = emp_matrix.sum(axis=1, keepdims=True)   # E_r
    total_national = emp_matrix.sum()                          # E_n
    industry_national = emp_matrix.sum(axis=0, keepdims=True) # e_in

    share_regional = emp_matrix / total_regional               # e_ir/E_r
    share_national = industry_national / total_national        # e_in/E_n
    lq = share_regional / share_national
    return lq

lq = location_quotient(emp_t0)
lq_df = pd.DataFrame(lq, index=region_names, columns=industry_names)
print("Location Quotients (t0):")
print(lq_df.round(2))
print(f"\nSpecialized cells (LQ>1.25): {(lq > 1.25).sum()}")

# -----------------------------------------------------------------
# Classic Shift-Share (Dunn 1960)
# -----------------------------------------------------------------
def shift_share(emp_t0, emp_t1):
    """Dunn shift-share decomposition.

    Returns national_share, industry_mix, competitive_effect
    all shape (n_regions, n_industries).
    """
    E_n0 = emp_t0.sum()
    E_n1 = emp_t1.sum()
    g_n = (E_n1 - E_n0) / E_n0                              # national growth rate

    e_in0 = emp_t0.sum(axis=0)                               # national industry emp t0
    e_in1 = emp_t1.sum(axis=0)                               # national industry emp t1
    g_in = (e_in1 - e_in0) / e_in0                          # national industry growth rates

    g_ir = (emp_t1 - emp_t0) / emp_t0                       # regional-industry growth

    ns = emp_t0 * g_n                                        # national share effect
    im = emp_t0 * (g_in - g_n)                              # industry mix effect
    ce = emp_t0 * (g_ir - g_in)                             # competitive effect
    return ns, im, ce

ns, im, ce = shift_share(emp_t0, emp_t1)

# Aggregate by region
ss_df = pd.DataFrame({
    "National_Share": ns.sum(axis=1),
    "Industry_Mix": im.sum(axis=1),
    "Competitive": ce.sum(axis=1),
    "Total_Change": (emp_t1 - emp_t0).sum(axis=1)
}, index=region_names)
print("\nShift-Share Decomposition (aggregate by region):")
print(ss_df.round(0))

# -----------------------------------------------------------------
# Visualization: Stacked bar chart
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LQ heatmap
im_ax = axes[0].imshow(lq, cmap="RdYlGn", vmin=0.5, vmax=2.0, aspect="auto")
axes[0].set_xticks(range(n_industries)); axes[0].set_xticklabels(industry_names, rotation=30, ha="right")
axes[0].set_yticks(range(n_regions)); axes[0].set_yticklabels(region_names)
axes[0].set_title("Location Quotients")
plt.colorbar(im_ax, ax=axes[0], label="LQ")

# Shift-share stacked bar
x = np.arange(n_regions)
w = 0.6
axes[1].bar(x, ss_df["National_Share"], w, label="National Share", color="steelblue")
axes[1].bar(x, ss_df["Industry_Mix"], w, bottom=ss_df["National_Share"],
            label="Industry Mix", color="orange")
axes[1].bar(x, ss_df["Competitive"], w,
            bottom=ss_df["National_Share"] + ss_df["Industry_Mix"],
            label="Competitive", color="green")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(region_names, rotation=30, ha="right")
axes[1].set_title("Shift-Share Decomposition")
axes[1].set_ylabel("Employment Change")
axes[1].legend()

plt.tight_layout()
plt.savefig("shift_share_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: shift_share_analysis.png")
```

### Step 2: Input-Output Model and Multipliers

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Build a stylized 4-sector IO table
# Sectors: Agriculture, Manufacturing, Services, Construction
# -----------------------------------------------------------------
sectors = ["Agriculture", "Manufacturing", "Services", "Construction"]
n = len(sectors)

# Intermediate transactions matrix Z (4x4)
Z = np.array([
    [20,  80,  10,   5],   # Agriculture sells to ...
    [15, 120,  60,  30],   # Manufacturing sells to ...
    [ 5,  40, 150,  20],   # Services sells to ...
    [ 2,  15,  10,   8],   # Construction sells to ...
], dtype=float)

# Final demand vector f
f = np.array([100, 200, 300, 80], dtype=float)

# Gross output X = Z.sum(axis=1) + f
X = Z.sum(axis=1) + f
print("Gross Output X:", X)

# -----------------------------------------------------------------
# Technical coefficient matrix A
# -----------------------------------------------------------------
A = Z / X[np.newaxis, :]    # a_ij = z_ij / X_j
print("\nTechnical Coefficient Matrix A:")
print(pd.DataFrame(A, index=sectors, columns=sectors).round(3))

# -----------------------------------------------------------------
# Leontief Inverse L = (I - A)^{-1}
# -----------------------------------------------------------------
I = np.eye(n)
L = np.linalg.inv(I - A)
print("\nLeontief Inverse L:")
print(pd.DataFrame(L, index=sectors, columns=sectors).round(3))

# Output multipliers: column sums of L
output_mult = L.sum(axis=0)
print("\nOutput Multipliers (column sums of L):")
for s, m in zip(sectors, output_mult):
    print(f"  {s}: {m:.3f}")

# -----------------------------------------------------------------
# Employment multipliers (if we have employment coefficients)
# -----------------------------------------------------------------
# Suppose employment per unit output (jobs per $M output)
emp_coeff = np.array([0.05, 0.03, 0.04, 0.06])  # l vector
emp_mult = emp_coeff @ L                           # total employment requirements
print("\nEmployment Multipliers:")
for s, m in zip(sectors, emp_mult):
    print(f"  {s}: {m:.4f} jobs per $M final demand")

# -----------------------------------------------------------------
# Simulate a demand shock: +$50M final demand in Manufacturing
# -----------------------------------------------------------------
delta_f = np.array([0, 50, 0, 0], dtype=float)
delta_X = L @ delta_f
print(f"\nImpact of +$50M demand shock in Manufacturing:")
for s, dx in zip(sectors, delta_X):
    print(f"  {s}: +${dx:.1f}M output")

# -----------------------------------------------------------------
# Visualization: Leontief inverse heatmap
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im = axes[0].imshow(L, cmap="Blues", aspect="auto")
axes[0].set_xticks(range(n)); axes[0].set_xticklabels(sectors, rotation=30, ha="right")
axes[0].set_yticks(range(n)); axes[0].set_yticklabels(sectors)
axes[0].set_title("Leontief Inverse (I-A)⁻¹")
plt.colorbar(im, ax=axes[0])
# Annotate cells
for i in range(n):
    for j in range(n):
        axes[0].text(j, i, f"{L[i,j]:.2f}", ha="center", va="center",
                     color="white" if L[i,j] > 1.5 else "black", fontsize=8)

# Output and employment multipliers bar chart
x = np.arange(n)
w = 0.35
axes[1].bar(x - w/2, output_mult, w, label="Output Multiplier", color="steelblue")
axes[1].bar(x + w/2, emp_mult / emp_mult.max() * output_mult.max(), w,
            label="Emp. Multiplier (scaled)", color="orange")
axes[1].set_xticks(x); axes[1].set_xticklabels(sectors, rotation=20, ha="right")
axes[1].set_title("Output and Employment Multipliers")
axes[1].set_ylabel("Multiplier")
axes[1].legend()

plt.tight_layout()
plt.savefig("io_multipliers.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: io_multipliers.png")
```

### Step 3: Regional Convergence and Spatial Econometrics

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------------------------------------------
# Simulate panel data: 50 regions, 20 years
# -----------------------------------------------------------------
np.random.seed(123)
n_r = 50
T = 20

# Initial per-capita income (log scale, heterogeneous)
ln_y0 = np.random.uniform(9.0, 11.5, n_r)       # ln(initial income)
# True beta: -0.05 (convergence), alpha: 0.5
beta_true = -0.05
alpha_true = 0.5
g = alpha_true + beta_true * ln_y0 + np.random.normal(0, 0.02, n_r)  # annual growth rate

# -----------------------------------------------------------------
# Absolute beta-convergence OLS
# -----------------------------------------------------------------
X_reg = sm.add_constant(ln_y0)
model = sm.OLS(g, X_reg)
result = model.fit(cov_type="HC3")   # heteroskedasticity-robust SE
print("=== Absolute Beta-Convergence ===")
print(result.summary().tables[1])

beta_hat = result.params[1]
lam = -np.log(1 + beta_hat * T) / T if (1 + beta_hat * T) > 0 else np.nan
half_life = np.log(2) / lam if lam > 0 else np.inf
print(f"\nEstimated beta: {beta_hat:.4f}")
print(f"Convergence speed lambda: {lam:.4f}")
print(f"Half-life: {half_life:.1f} years")

# -----------------------------------------------------------------
# Sigma-convergence: dispersion over time
# -----------------------------------------------------------------
sigma = []
for t in range(T):
    ln_yt = ln_y0 + g * t   # simplified linear approximation
    sigma.append(ln_yt.std())
sigma = np.array(sigma)

# -----------------------------------------------------------------
# Krugman Specialization Index between all region pairs
# -----------------------------------------------------------------
n_regions_k = 8
n_industries_k = 6
emp_k = np.random.randint(500, 5000, (n_regions_k, n_industries_k)).astype(float)
shares = emp_k / emp_k.sum(axis=1, keepdims=True)   # industry shares per region

K_matrix = np.zeros((n_regions_k, n_regions_k))
for i in range(n_regions_k):
    for j in range(n_regions_k):
        K_matrix[i, j] = 0.5 * np.abs(shares[i] - shares[j]).sum()

print(f"\nKrugman Specialization Index (mean): {K_matrix[np.triu_indices(n_regions_k, k=1)].mean():.3f}")

# -----------------------------------------------------------------
# Economic base multiplier
# -----------------------------------------------------------------
# Using LQ > 1 rule to identify basic employment
LQ_k = location_quotient(emp_k)  # reuse function from Step 1

# Basic employment in each region
def economic_base_multiplier(emp, lq):
    """Compute economic base multiplier for each region."""
    total_emp = emp.sum(axis=1)
    basic_emp = np.where(lq > 1.0, emp - emp.sum(axis=0) / emp.sum() * emp.sum(axis=1, keepdims=True), 0)
    basic_emp = np.maximum(basic_emp, 0).sum(axis=1)
    multiplier = total_emp / np.where(basic_emp > 0, basic_emp, 1)
    return multiplier, basic_emp

mult_k, basic_k = economic_base_multiplier(emp_k, LQ_k)
for r, (m, b) in enumerate(zip(mult_k, basic_k)):
    print(f"Region {chr(65+r)}: Basic emp = {b:.0f}, Multiplier = {m:.2f}")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Beta-convergence scatter
axes[0].scatter(ln_y0, g, alpha=0.6, edgecolors="k", linewidths=0.4)
x_line = np.linspace(ln_y0.min(), ln_y0.max(), 100)
axes[0].plot(x_line, result.params[0] + result.params[1] * x_line, "r-", lw=2)
axes[0].set_xlabel("ln(Initial Income)")
axes[0].set_ylabel("Average Annual Growth Rate")
axes[0].set_title(f"β-Convergence (β={beta_hat:.3f})")

# Sigma-convergence
axes[1].plot(range(T), sigma, "o-", color="steelblue")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Std Dev of ln(Income)")
axes[1].set_title("σ-Convergence")

# Krugman specialization matrix
im2 = axes[2].imshow(K_matrix, cmap="YlOrRd", vmin=0, vmax=0.6)
axes[2].set_title("Krugman Specialization Index")
plt.colorbar(im2, ax=axes[2], label="K")

plt.tight_layout()
plt.savefig("regional_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: regional_convergence.png")


def location_quotient(emp_matrix):
    """Reusable LQ function."""
    total_regional = emp_matrix.sum(axis=1, keepdims=True)
    total_national = emp_matrix.sum()
    industry_national = emp_matrix.sum(axis=0, keepdims=True)
    share_regional = emp_matrix / total_regional
    share_national = industry_national / total_national
    return share_regional / share_national
```

## Advanced Usage

### Esteban-Ray Polarization Index

```python
import numpy as np

def esteban_ray_polarization(income, population, alpha=1.6):
    """Compute Esteban-Ray (1994) polarization index.

    Args:
        income: array of group mean incomes
        population: array of group population shares
        alpha: polarization sensitivity parameter (1 <= alpha <= 1.6)
    Returns:
        polarization index P
    """
    n = len(income)
    P = 0.0
    for i in range(n):
        for j in range(n):
            P += population[i]**(1 + alpha) * population[j] * abs(income[i] - income[j])
    return P

# Example: 5 income groups
income_groups = np.array([15000, 30000, 50000, 80000, 150000])
pop_shares = np.array([0.20, 0.25, 0.30, 0.15, 0.10])
P = esteban_ray_polarization(income_groups, pop_shares)
print(f"Esteban-Ray Polarization: {P:.2f}")
```

### Spatial Autocorrelation of Regional Income

```python
import numpy as np
from scipy.spatial.distance import cdist

def morans_i_regional(values, coords, k_neighbors=5):
    """Moran's I with k-nearest-neighbor spatial weights.

    Args:
        values: (n,) array of regional values
        coords: (n, 2) array of region centroids
        k_neighbors: number of neighbors
    Returns:
        morans_i, expected_i, z_score
    """
    n = len(values)
    dist_matrix = cdist(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)

    # Build binary KNN weight matrix
    W = np.zeros((n, n))
    for i in range(n):
        knn_idx = np.argsort(dist_matrix[i])[:k_neighbors]
        W[i, knn_idx] = 1
    W_sum = W.sum()

    # Moran's I
    z = values - values.mean()
    numer = n * (W * np.outer(z, z)).sum()
    denom = W_sum * (z**2).sum()
    I = numer / denom

    E_I = -1 / (n - 1)

    # Variance under normality assumption
    S1 = 0.5 * ((W + W.T)**2).sum()
    S2 = ((W.sum(axis=1) + W.sum(axis=0))**2).sum()
    n2 = n**2
    num_var = n * ((n2 - 3*n + 3)*S1 - n*S2 + 3*W_sum**2)
    den_var = (n-1)*(n-2)*(n-3)*W_sum**2
    kurtosis = (z**4).mean() / (z**2).mean()**2
    var_I = num_var / den_var - kurtosis * ((n2-n)*S1 - 2*n*S2 + 6*W_sum**2) / den_var

    z_score = (I - E_I) / np.sqrt(abs(var_I))
    return I, E_I, z_score

np.random.seed(42)
n_reg = 30
coords = np.random.uniform(0, 100, (n_reg, 2))
# Spatially autocorrelated income
income = 50000 + 10000 * np.sin(coords[:, 0] / 20) + np.random.normal(0, 2000, n_reg)

I_stat, E_I, z = morans_i_regional(income, coords)
print(f"Moran's I = {I_stat:.4f}, E[I] = {E_I:.4f}, Z = {z:.2f}")
```

### Gravity Model for Interregional Trade

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

def gravity_model(gdp_i, gdp_j, dist_ij, trade_ij):
    """Estimate gravity model: ln(T_ij) = a + b*ln(GDP_i) + c*ln(GDP_j) + d*ln(dist_ij).

    Args:
        gdp_i, gdp_j: (n,) arrays of exporter/importer GDP
        dist_ij: (n,) distance between regions
        trade_ij: (n,) bilateral trade flows
    Returns:
        OLS result
    """
    df = pd.DataFrame({
        "ln_trade": np.log(trade_ij),
        "ln_gdp_i": np.log(gdp_i),
        "ln_gdp_j": np.log(gdp_j),
        "ln_dist": np.log(dist_ij),
    }).dropna()
    X = sm.add_constant(df[["ln_gdp_i", "ln_gdp_j", "ln_dist"]])
    model = sm.OLS(df["ln_trade"], X)
    return model.fit(cov_type="HC3")

# Simulate 200 region pairs
np.random.seed(42)
n_pairs = 200
gdp_i = np.random.lognormal(10, 1.5, n_pairs)
gdp_j = np.random.lognormal(10, 1.5, n_pairs)
dist_ij = np.random.uniform(50, 2000, n_pairs)
# True gravity: T = GDP_i^0.8 * GDP_j^0.9 * dist^-1.2 * noise
trade_ij = (gdp_i**0.8 * gdp_j**0.9 * dist_ij**(-1.2) *
            np.exp(np.random.normal(0, 0.3, n_pairs)))

result = gravity_model(gdp_i, gdp_j, dist_ij, trade_ij)
print("Gravity Model Estimates:")
print(result.summary().tables[1])
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `np.linalg.inv` singular matrix | Near-zero or zero column in A | Check for all-zero industries; use `np.linalg.lstsq` or pseudoinverse |
| Negative basic employment | LQ < 1 or rounding | `np.maximum(basic_emp, 0)` clamp |
| Convergence beta insignificant | Too few regions or time span | Increase sample; use club-convergence tests |
| Division by zero in LQ | Region has zero employment | Filter out empty regions before computing |
| IO multiplier > 10 | Unrealistic A matrix | Check that column sums of A < 1 (no super-multiplier) |
| Moran's I close to -1/(n-1) | No spatial pattern | Expected; test for significance with permutation |

## External Resources

- Isard, W. (1960). *Methods of Regional Analysis*. MIT Press.
- Dunn, E. S. (1960). A statistical and analytical technique for regional analysis. *Papers in Regional Science*.
- Leontief, W. (1986). *Input-Output Economics*. Oxford University Press.
- Anselin, L. (1988). *Spatial Econometrics*. Kluwer Academic.
- Esteban, J., & Ray, D. (1994). On the measurement of polarization. *Econometrica*, 62(4), 819-851.
- [PySAL documentation](https://pysal.org/) — Python library for spatial analysis
- [FRED Regional Data](https://fred.stlouisfed.org/) — Federal Reserve regional economic data
- [BEA Input-Output Tables](https://www.bea.gov/industry/input-output-accounts-data)

## Examples

### Example 1: Full Shift-Share Report for a State

```python
import numpy as np
import pandas as pd

np.random.seed(2024)
n_industries = 10
industries = [
    "Agriculture", "Mining", "Utilities", "Construction",
    "Manufacturing", "Wholesale", "Retail", "Finance",
    "Professional Services", "Healthcare"
]

# State employment vs. national employment
state_t0 = np.array([1200, 800, 300, 2500, 8000, 1800, 5000, 3200, 4500, 6000], dtype=float)
state_t1 = state_t0 * np.array([0.98, 0.85, 1.02, 1.05, 0.92, 1.08, 1.03, 1.12, 1.18, 1.15])

nat_t0 = state_t0 * np.random.uniform(15, 25, n_industries)
nat_t1 = nat_t0 * np.array([0.96, 0.80, 1.03, 1.06, 0.90, 1.10, 1.04, 1.15, 1.20, 1.13])

# Compute components
E_n0, E_n1 = nat_t0.sum(), nat_t1.sum()
g_n = (E_n1 - E_n0) / E_n0
g_in = (nat_t1 - nat_t0) / nat_t0
g_ir = (state_t1 - state_t0) / state_t0

report = pd.DataFrame({
    "Industry": industries,
    "Emp_t0": state_t0,
    "Emp_t1": state_t1,
    "LQ": (state_t0 / state_t0.sum()) / (nat_t0 / nat_t0.sum()),
    "National_Share": state_t0 * g_n,
    "Industry_Mix": state_t0 * (g_in - g_n),
    "Competitive": state_t0 * (g_ir - g_in),
    "Total_Change": state_t1 - state_t0,
})
report["Growing_Basic"] = (report["LQ"] > 1.25) & (report["Competitive"] > 0)

pd.set_option("display.float_format", "{:.0f}".format)
print("=== State Shift-Share Report ===")
print(report.to_string(index=False))
print(f"\nTotal employment change: {(state_t1 - state_t0).sum():.0f}")
print(f"Competitive effect total: {report['Competitive'].sum():.0f}")
```

### Example 2: Input-Output Impact Assessment for Infrastructure Project

```python
import numpy as np
import pandas as pd

# 6-sector IO table for a metropolitan region
sectors = ["Agriculture", "Manufacturing", "Construction",
           "Trade", "Finance", "Services"]
n = len(sectors)

# Technical coefficient matrix (calibrated to reasonable values)
A = np.array([
    [0.05, 0.08, 0.02, 0.01, 0.00, 0.01],
    [0.10, 0.20, 0.15, 0.05, 0.02, 0.03],
    [0.02, 0.05, 0.10, 0.03, 0.01, 0.02],
    [0.08, 0.10, 0.08, 0.15, 0.05, 0.10],
    [0.03, 0.05, 0.04, 0.06, 0.10, 0.08],
    [0.12, 0.15, 0.10, 0.18, 0.20, 0.20],
])
# Verify sum < 1 (valid IO table)
assert all(A.sum(axis=0) < 1), "Column sums must be < 1"

L = np.linalg.inv(np.eye(n) - A)

# Infrastructure project: $500M construction spending
# Direct final demand shock
delta_f = np.zeros(n)
delta_f[2] = 500  # Construction sector

# Total output impact
delta_X = L @ delta_f
print("=== Infrastructure Project ($500M Construction) ===")
impact_df = pd.DataFrame({
    "Sector": sectors,
    "Direct_Demand": delta_f,
    "Total_Output_Impact": delta_X,
    "Multiplier": L[:, 2],  # Construction column
})
print(impact_df.to_string(index=False))
print(f"\nTotal output multiplier for Construction: {L[:, 2].sum():.3f}")
print(f"Total economic impact: ${delta_X.sum():.1f}M")
print(f"Indirect/induced effects: ${delta_X.sum() - 500:.1f}M ({(delta_X.sum()/500-1)*100:.1f}%)")
```
