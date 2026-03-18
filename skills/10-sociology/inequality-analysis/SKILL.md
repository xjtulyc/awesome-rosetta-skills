---
name: inequality-analysis
description: >
  Use this Skill for inequality analysis: Gini coefficient, Lorenz curves,
  Theil index decomposition, income mobility, and distributional regression.
tags:
  - sociology
  - inequality
  - gini
  - income-distribution
  - poverty
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
    - pandas>=2.0
    - scipy>=1.11
    - matplotlib>=3.7
    - statsmodels>=0.14
last_updated: "2026-03-17"
status: "stable"
---

# Inequality Analysis

> **One-line summary**: Quantify income and wealth inequality with Gini coefficients, Lorenz curves, Theil index decomposition, poverty measures, and distributional regression.

---

## When to Use This Skill

- When computing Gini coefficient and Lorenz curve from survey microdata
- When decomposing inequality within and between groups (Theil index)
- When measuring poverty (FGT indices, poverty gap, poverty depth)
- When analyzing income mobility (rank-rank regression, transition matrices)
- When running distributional regression or unconditional quantile regression
- When tracking inequality trends across years or countries

**Trigger keywords**: Gini coefficient, Lorenz curve, Theil index, income inequality, poverty rate, FGT index, income distribution, quantile regression, wealth inequality, income mobility, top income shares, Palma ratio

---

## Background & Key Concepts

### Gini Coefficient

$$
G = \frac{\sum_{i=1}^n \sum_{j=1}^n |y_i - y_j|}{2n^2 \bar{y}} = 1 - \frac{2}{n}\sum_{i=1}^n \frac{n-i+1}{n} \frac{y_i}{\bar{y}}
$$

where observations are sorted $y_1 \leq y_2 \leq \ldots \leq y_n$.

### Theil Index

$$
T = \frac{1}{n}\sum_{i=1}^n \frac{y_i}{\bar{y}} \ln\left(\frac{y_i}{\bar{y}}\right)
$$

Decomposition: $T = T_{\text{between}} + T_{\text{within}}$ where between-group component measures share explained by group differences.

### FGT Poverty Measures

$$
FGT_\alpha = \frac{1}{n}\sum_{i: y_i < z} \left(\frac{z - y_i}{z}\right)^\alpha
$$

- $\alpha=0$: headcount ratio; $\alpha=1$: poverty gap; $\alpha=2$: poverty severity

---

## Environment Setup

### Install Dependencies

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 matplotlib>=3.7 statsmodels>=0.14
```

### Verify Installation

```python
import numpy as np

# Quick Gini test
def gini(y):
    y_sorted = np.sort(y)
    n = len(y_sorted)
    idx = np.arange(1, n+1)
    return (2 * np.sum(idx * y_sorted) / (n * y_sorted.sum())) - (n+1)/n

# Perfectly equal: Gini = 0; perfectly unequal: Gini → 1
print(gini(np.ones(100)))     # Expected: 0.0
print(gini(np.arange(1, 101)))  # Expected: ~0.33
```

---

## Core Workflow

### Step 1: Gini Coefficient and Lorenz Curve

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# ------------------------------------------------------------------ #
# Compute Gini and Lorenz curve from income microdata
# ------------------------------------------------------------------ #

def gini_coefficient(y, weights=None):
    """
    Compute weighted Gini coefficient.

    Parameters
    ----------
    y : array — income values
    weights : array or None — survey weights (uniform if None)

    Returns
    -------
    float — Gini coefficient [0, 1]
    """
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    weights = np.asarray(weights, dtype=float)

    # Remove non-positive
    mask = y > 0
    y, weights = y[mask], weights[mask]

    # Sort by income
    order = np.argsort(y)
    y, weights = y[order], weights[order]

    cum_income = np.cumsum(y * weights)
    cum_weight = np.cumsum(weights)

    # Normalize
    total_income = cum_income[-1]
    total_weight = cum_weight[-1]

    # Lorenz ordinates
    L_income = cum_income / total_income   # Cumulative income share
    L_pop    = cum_weight / total_weight   # Cumulative population share

    # Gini = 1 - 2 * area under Lorenz curve (trapezoid rule)
    gini = 1 - 2 * np.trapz(L_income, L_pop)
    return gini, L_pop, L_income

def lorenz_curve(y, weights=None, n_points=100):
    """Return evenly-spaced Lorenz curve coordinates."""
    _, L_pop, L_income = gini_coefficient(y, weights)
    # Interpolate to uniform grid
    p_grid = np.linspace(0, 1, n_points)
    L_interp = np.interp(p_grid, L_pop, L_income)
    return p_grid, L_interp

# ---- Simulated income distributions for three countries --------- #
np.random.seed(42)

countries = {
    'Low inequality (Scandinavia-like)':  {'mu': 10.5, 'sigma': 0.45},
    'Medium inequality (US-like)':        {'mu': 10.3, 'sigma': 0.80},
    'High inequality (Brazil-like)':      {'mu': 8.5,  'sigma': 1.20},
}

n_hh = 5000
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

gini_results = {}
for country, params in countries.items():
    income = np.random.lognormal(params['mu'], params['sigma'], n_hh)
    weights = np.random.uniform(0.5, 1.5, n_hh)  # Sampling weights

    gini_val, L_pop, L_income = gini_coefficient(income, weights)
    gini_results[country] = gini_val

    # Lorenz curve
    p, L = lorenz_curve(income, weights)
    axes[0].plot(p * 100, L * 100, linewidth=2,
                 label=f"{country.split('(')[0].strip()} (G={gini_val:.3f})")

# Reference line (perfect equality)
axes[0].plot([0, 100], [0, 100], 'k--', linewidth=1, label='Perfect equality')
axes[0].fill_between([0, 100], [0, 100], [0, 0], alpha=0.05, color='gray')
axes[0].set_xlabel("Cumulative population share (%)")
axes[0].set_ylabel("Cumulative income share (%)")
axes[0].set_title("Lorenz Curves by Country")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

# Income density
for country, params in countries.items():
    income = np.random.lognormal(params['mu'], params['sigma'], n_hh)
    axes[1].hist(np.log(income), bins=60, alpha=0.5, density=True,
                 label=country.split('(')[0].strip())
axes[1].set_xlabel("log(Income)"); axes[1].set_ylabel("Density")
axes[1].set_title("log-Income Distributions")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lorenz_gini.png", dpi=150)
plt.show()

print("\nGini coefficients:")
for c, g in gini_results.items():
    print(f"  {c}: {g:.4f}")
```

### Step 2: Theil Index Decomposition and Top Income Shares

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Theil index: total, within-group, between-group decomposition
# ------------------------------------------------------------------ #

def theil_index(y, weights=None):
    """Mean log deviation (Theil L) and Theil T for array y."""
    y = np.asarray(y, dtype=float)
    y = y[y > 0]
    if weights is None:
        weights = np.ones_like(y)
    weights = weights[y > 0] if len(weights) > 0 else np.ones_like(y)
    weights = weights / weights.sum()
    mean_y = np.average(y, weights=weights)

    # Theil T = E[y/mu * ln(y/mu)]
    theil_T = np.sum(weights * (y/mean_y) * np.log(y/mean_y + 1e-12))
    # Theil L (MLD) = E[-ln(y/mu)]
    theil_L = -np.sum(weights * np.log(y/mean_y + 1e-12))
    return theil_T, theil_L

def theil_decomposition(y, groups, weights=None):
    """
    Decompose Theil T into within-group and between-group components.

    Returns
    -------
    dict with total, between, within, and share_between
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if weights is None:
        weights = np.ones_like(y)
    weights = np.asarray(weights, dtype=float)
    total_weight = weights.sum()
    total_mean = np.average(y, weights=weights)

    total_T, _ = theil_index(y, weights)

    # Within-group component
    within = 0
    between = 0
    for g in np.unique(groups):
        mask = groups == g
        y_g = y[mask]; w_g = weights[mask]
        if len(y_g) == 0 or y_g.sum() == 0:
            continue
        n_g_share = w_g.sum() / total_weight
        mean_g = np.average(y_g, weights=w_g)
        T_g, _ = theil_index(y_g, w_g)
        within  += (mean_g/total_mean) * n_g_share * T_g
        between += (mean_g/total_mean) * n_g_share * np.log(mean_g/total_mean + 1e-12)

    return {
        'total':         total_T,
        'within':        within,
        'between':       between,
        'share_between': between / (total_T + 1e-12) * 100,
    }

# ---- Simulated data: income by education level ------------------ #
np.random.seed(99)
n = 10000
education_group = np.random.choice(['<HS', 'HS', 'College', 'Graduate'],
                                    n, p=[0.15, 0.35, 0.30, 0.20])

group_means = {'<HS': 25000, 'HS': 42000, 'College': 72000, 'Graduate': 110000}
group_sds   = {'<HS': 0.5,   'HS': 0.5,   'College':  0.6,   'Graduate':  0.7}

income = np.array([
    np.random.lognormal(np.log(group_means[g]) - group_sds[g]**2/2, group_sds[g])
    for g in education_group
])

# Decompose
result = theil_decomposition(income, education_group)
print("=== Theil Decomposition by Education ===")
print(f"Total Theil T: {result['total']:.4f}")
print(f"Within-group:  {result['within']:.4f}")
print(f"Between-group: {result['between']:.4f}  ({result['share_between']:.1f}% of total)")

# ---- Top income shares ------------------------------------------ #
def income_shares(y, weights=None):
    """Return income shares by percentile (top 1%, 5%, 10%, 50%)."""
    y_sorted = np.sort(y)
    n = len(y_sorted)
    total = y_sorted.sum()
    shares = {}
    for pct in [1, 5, 10, 50]:
        threshold = np.percentile(y_sorted, 100 - pct)
        top_income = y_sorted[y_sorted >= threshold].sum()
        shares[f"Top {pct}%"] = top_income / total * 100
    return shares

shares = income_shares(income)
print("\n=== Top Income Shares ===")
for k, v in shares.items():
    print(f"  {k}: {v:.2f}%")

# Palma ratio: top 10% / bottom 40%
p90 = np.percentile(income, 90)
p40 = np.percentile(income, 40)
top10_share  = income[income >= p90].sum() / income.sum()
bottom40_share = income[income <= p40].sum() / income.sum()
palma = top10_share / bottom40_share
print(f"\nPalma ratio: {palma:.3f}")
```

### Step 3: Poverty Measures and Quantile Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ------------------------------------------------------------------ #
# FGT poverty measures and unconditional quantile regression
# ------------------------------------------------------------------ #

def fgt_poverty(y, poverty_line, weights=None, alpha_values=[0, 1, 2]):
    """
    Compute Foster-Greer-Thorbecke poverty measures.

    Parameters
    ----------
    y : array — income
    poverty_line : float — poverty threshold z
    alpha_values : list — [0]=headcount, [1]=gap, [2]=severity
    """
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(y)
    weights = np.asarray(weights)

    poor = y < poverty_line
    results = {}
    for alpha in alpha_values:
        gap = np.maximum(poverty_line - y, 0) / poverty_line
        fgt = np.average(gap**alpha, weights=weights)
        results[f"FGT_{alpha}"] = fgt
    return results

# Simulation: income data for a developing country
np.random.seed(42)
n = 20000
income_sim = np.random.lognormal(7.5, 1.1, n)  # Mean ~$1,500/month
poverty_line = 600  # $/month

poverty_measures = fgt_poverty(income_sim, poverty_line)
print("=== Poverty Measures ===")
print(f"Poverty line: ${poverty_line}/month")
print(f"FGT₀ (Headcount rate): {poverty_measures['FGT_0']*100:.2f}%")
print(f"FGT₁ (Poverty gap):    {poverty_measures['FGT_1']*100:.2f}%")
print(f"FGT₂ (Poverty severity):{poverty_measures['FGT_2']*100:.2f}%")

# ---- Quantile decomposition (between groups) ------------------- #
gender = np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52])
# Add gender income gap
income_with_gap = income_sim.copy()
income_with_gap[gender == 'Female'] *= 0.75  # 25% pay gap

df_qr = pd.DataFrame({
    'income': income_with_gap,
    'log_income': np.log(income_with_gap),
    'female': (gender == 'Female').astype(int),
})

# OLS and quantile regression
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
ols_res = smf.ols("log_income ~ female", data=df_qr).fit()
qr_coefs = []
for q in quantiles:
    qr = smf.quantreg("log_income ~ female", data=df_qr).fit(q=q)
    qr_coefs.append(qr.params['female'])

print(f"\nOLS gender log-income gap: {ols_res.params['female']:.4f} ({np.exp(ols_res.params['female'])*100-100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Quantile regression coefficients
axes[0].plot(quantiles, qr_coefs, 'bo-', linewidth=2, markersize=8, label='Quantile regression')
axes[0].axhline(ols_res.params['female'], color='red', linestyle='--',
                linewidth=1.5, label=f'OLS (={ols_res.params["female"]:.3f})')
axes[0].axhline(0, color='gray', linestyle='-', linewidth=0.5)
axes[0].set_xlabel("Quantile"); axes[0].set_ylabel("Gender gap in log income")
axes[0].set_title("Gender Income Gap Across Distribution\n(Quantile Regression)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Income distributions by gender
for g, color in [('Male','steelblue'), ('Female','coral')]:
    inc = df_qr[df_qr['female'] == (g=='Female')]['income']
    axes[1].hist(np.log(inc), bins=60, alpha=0.5, density=True, color=color, label=g)
axes[1].axvline(np.log(poverty_line), color='red', linewidth=2, linestyle='--', label=f'Poverty line (${poverty_line})')
axes[1].set_xlabel("log(Income)"); axes[1].set_ylabel("Density")
axes[1].set_title("Income Distribution by Gender")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("inequality_analysis.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Income Mobility (Rank-Rank Regression)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

np.random.seed(7)
n = 5000

# Simulate parent-child income (intergenerational persistence)
parent_income = np.random.lognormal(10.5, 0.8, n)
# IGE (intergenerational elasticity) ≈ 0.45 for US
child_income = np.exp(0.45 * np.log(parent_income) + np.random.normal(0, 0.7, n))

# Rank-rank regression (Chetty et al. approach)
parent_rank = pd.Series(parent_income).rank(pct=True) * 100
child_rank  = pd.Series(child_income).rank(pct=True) * 100

X = sm.add_constant(parent_rank)
rr_result = sm.OLS(child_rank, X).fit()
rank_slope = rr_result.params.iloc[1]
print(f"Rank-rank slope: {rank_slope:.4f}  (US benchmark: ~0.34)")

# Income transition matrix
n_quintiles = 5
parent_quintile = pd.qcut(parent_income, n_quintiles, labels=range(1,6))
child_quintile  = pd.qcut(child_income,  n_quintiles, labels=range(1,6))

transition = pd.crosstab(parent_quintile, child_quintile, normalize='index') * 100
print("\nIncome transition matrix (row=parent quintile, col=child quintile):")
print(transition.round(1))
print(f"Diagonal mean (intergenerational persistence): {np.diag(transition.values).mean():.1f}%")
```

---

## Troubleshooting

### Gini = 0 or 1 unexpectedly

**Cause**: Data contains zeros or negative values, or all values identical.

**Fix**:
```python
y = y[y > 0]  # Remove zeros/negatives before Gini computation
assert len(y) > 0, "All values non-positive"
```

### Theil index blows up with very small incomes

**Fix**: Add small epsilon before log:
```python
theil_T = np.sum(weights * (y/mean_y) * np.log(y/mean_y + 1e-12))
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| statsmodels | 0.14 | `quantreg` stable API |
| scipy | 1.11, 1.12 | No known issues |
| numpy | 1.24, 1.26 | No known issues |

---

## External Resources

### Official Documentation

- [World Bank PovcalNet methodology](https://documents.worldbank.org/en/publication/documents-reports/documentdetail/099235304052323819)
- [Luxembourg Income Study (LIS)](https://www.lisdatacenter.org)

### Key Papers

- Foster, J., Greer, J. & Thorbecke, E. (1984). *A class of decomposable poverty measures*. Econometrica.
- Chetty, R. et al. (2014). *Is the United States still a land of opportunity?* AER Papers & Proceedings.

---

## Examples

### Example 1: Gini Trend Over Time

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
years = range(1980, 2024, 2)
# Simulate rising inequality trend
gini_trend = [0.30 + 0.002*(y-1980) + np.random.normal(0, 0.01) for y in years]
gini_se    = [0.008 + 0.0001*(y-1980) for y in years]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(list(years), gini_trend, 'b-o', linewidth=2, markersize=6)
ax.fill_between(list(years),
                [g - 1.96*s for g,s in zip(gini_trend, gini_se)],
                [g + 1.96*s for g,s in zip(gini_trend, gini_se)],
                alpha=0.2, color='blue', label='95% CI')
ax.set_xlabel("Year"); ax.set_ylabel("Gini Coefficient")
ax.set_title("Income Inequality Trend (Gini Coefficient, 1980–2023)")
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig("gini_trend.png", dpi=150); plt.show()
```

### Example 2: Top 1% Income Share

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
years = np.arange(1980, 2024)
# Historical pattern: dipped 1990s, rose 2000s
top1_share = (8 + 0.15*(years-1980) - 0.02*(years-2000)**2 * (years>2000)
              + np.random.randn(len(years)) * 0.3)
top1_share = np.clip(top1_share, 8, 22)

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(years, 0, top1_share, alpha=0.3, color='crimson')
ax.plot(years, top1_share, 'crimson', linewidth=2)
ax.set_xlabel("Year"); ax.set_ylabel("Income share (%)")
ax.set_title("Top 1% Pre-Tax Income Share")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("top1_share.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
