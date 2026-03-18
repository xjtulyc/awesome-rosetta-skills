---
name: systematic-review-epi
description: >
  Use this Skill for systematic reviews: meta-regression, publication bias
  tests (Egger, funnel plot), GRADE evidence synthesis, and forest plots.
tags:
  - public-health
  - systematic-review
  - meta-analysis
  - evidence-synthesis
  - publication-bias
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
    - statsmodels>=0.14
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# Systematic Review and Evidence Synthesis

> **One-line summary**: Synthesize evidence from systematic reviews: compute pooled effects (OR, RR, RD, SMD), draw forest plots, test for heterogeneity (I², Q), detect publication bias (Egger test, funnel plot), and run meta-regression.

---

## When to Use This Skill

- When pooling effect sizes from multiple studies (meta-analysis)
- When computing I² heterogeneity and Cochran's Q test
- When drawing forest plots with random-effects pooled estimates
- When testing for publication bias using Egger's test or Begg's test
- When performing meta-regression on study-level covariates
- When rating quality of evidence using GRADE framework

**Trigger keywords**: systematic review, meta-analysis, forest plot, random effects, fixed effects, DerSimonian-Laird, heterogeneity, I-squared, Cochran Q, publication bias, Egger test, funnel plot, meta-regression, pooled effect, odds ratio, risk ratio, SMD, Cohen's d, GRADE

---

## Background & Key Concepts

### Random-Effects Model (DerSimonian-Laird)

$$
\hat{\theta}_{RE} = \frac{\sum_i w_i^* \hat{\theta}_i}{\sum_i w_i^*}, \quad w_i^* = \frac{1}{v_i + \hat{\tau}^2}
$$

where $\hat{\tau}^2$ is the between-study variance estimated from the Q statistic.

### Heterogeneity

Cochran's Q: $Q = \sum_i w_i (\hat{\theta}_i - \hat{\theta}_{FE})^2 \sim \chi^2(k-1)$

$$
I^2 = \max\left(0, \frac{Q - (k-1)}{Q}\right) \times 100\%
$$

$I^2 < 25\%$: low, 25-75%: moderate, >75%: high heterogeneity.

### Egger's Test for Publication Bias

Regress standard normal deviate ($\hat{\theta}/SE$) on precision ($1/SE$). Non-zero intercept indicates asymmetry.

---

## Environment Setup

### Install Dependencies

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 statsmodels>=0.14 matplotlib>=3.7
```

### Verify Installation

```python
import numpy as np
from scipy.stats import norm, chi2

# Quick meta-analysis test
log_ors = np.array([-0.1, 0.3, 0.5, 0.2])
se_ors  = np.array([0.15, 0.20, 0.25, 0.12])
weights = 1 / se_ors**2
pooled = np.sum(weights * log_ors) / np.sum(weights)
print(f"Fixed-effects OR: {np.exp(pooled):.3f}")
```

---

## Core Workflow

### Step 1: Data Extraction and Forest Plot

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chi2, norm

# ------------------------------------------------------------------ #
# Meta-analysis: effectiveness of hand hygiene on infection risk
# (Simulated systematic review data based on literature)
# ------------------------------------------------------------------ #

# Study data: name, log(OR), SE, year, setting
studies = [
    {'study': 'Aiello et al. 2008',  'log_or': -0.51, 'se': 0.14, 'year': 2008, 'setting': 'school'},
    {'study': 'Luby et al. 2005',    'log_or': -0.73, 'se': 0.18, 'year': 2005, 'setting': 'community'},
    {'study': 'Cairncross et al.',   'log_or': -0.45, 'se': 0.16, 'year': 2010, 'setting': 'community'},
    {'study': 'Jefferson et al.',    'log_or': -0.40, 'se': 0.20, 'year': 2009, 'setting': 'community'},
    {'study': 'Talaat et al. 2011',  'log_or': -0.35, 'se': 0.22, 'year': 2011, 'setting': 'school'},
    {'study': 'Grayson et al. 2009', 'log_or': -0.82, 'se': 0.25, 'year': 2009, 'setting': 'hospital'},
    {'study': 'Picheansanthian',     'log_or': -0.60, 'se': 0.30, 'year': 2004, 'setting': 'hospital'},
    {'study': 'Larson et al. 2012',  'log_or': -0.28, 'se': 0.17, 'year': 2012, 'setting': 'school'},
    {'study': 'Bowen et al. 2013',   'log_or': -0.55, 'se': 0.21, 'year': 2013, 'setting': 'community'},
    {'study': 'Hübner et al. 2013',  'log_or': -0.90, 'se': 0.35, 'year': 2013, 'setting': 'hospital'},
]

df = pd.DataFrame(studies)
k = len(df)

# ---- Fixed-effects pooled estimate ----------------------------- #
w_fe = 1 / df['se']**2
theta_fe = np.sum(w_fe * df['log_or']) / np.sum(w_fe)
se_fe = 1 / np.sqrt(np.sum(w_fe))

# ---- Cochran's Q and I² ------------------------------------ #
Q = np.sum(w_fe * (df['log_or'] - theta_fe)**2)
p_Q = 1 - chi2.cdf(Q, df=k-1)
I2 = max(0, (Q - (k-1)) / Q) * 100

# ---- DerSimonian-Laird random-effects estimate --------------- #
tau2 = max(0, (Q - (k-1)) / (np.sum(w_fe) - np.sum(w_fe**2)/np.sum(w_fe)))
w_re = 1 / (df['se']**2 + tau2)
theta_re = np.sum(w_re * df['log_or']) / np.sum(w_re)
se_re = 1 / np.sqrt(np.sum(w_re))

ci_lo_re = theta_re - 1.96 * se_re
ci_hi_re = theta_re + 1.96 * se_re

print(f"=== Meta-Analysis Results ===")
print(f"k = {k} studies")
print(f"\nHeterogeneity:")
print(f"  Q = {Q:.3f}  (df={k-1}, p={p_Q:.4f})")
print(f"  I² = {I2:.1f}%  ({'low' if I2<25 else 'moderate' if I2<75 else 'high'})")
print(f"  τ² = {tau2:.4f}  (between-study variance)")
print(f"\nFixed-effects: OR = {np.exp(theta_fe):.3f}  [{np.exp(theta_fe-1.96*se_fe):.3f}, {np.exp(theta_fe+1.96*se_fe):.3f}]")
print(f"Random-effects: OR = {np.exp(theta_re):.3f}  [{np.exp(ci_lo_re):.3f}, {np.exp(ci_hi_re):.3f}]")

# ---- Forest plot ---------------------------------------------- #
fig, ax = plt.subplots(figsize=(12, 8))

# Sort by log OR
df_sorted = df.sort_values('log_or').reset_index(drop=True)
w_re_sorted = 1 / (df_sorted['se']**2 + tau2)
marker_sizes = w_re_sorted / w_re_sorted.max() * 200 + 20

# Study estimates
for i, row in df_sorted.iterrows():
    ci_lo = row['log_or'] - 1.96 * row['se']
    ci_hi = row['log_or'] + 1.96 * row['se']
    ax.errorbar(np.exp(row['log_or']), i, xerr=[[np.exp(row['log_or'])-np.exp(ci_lo)],
                                                   [np.exp(ci_hi)-np.exp(row['log_or'])]],
                fmt='s', markersize=np.sqrt(marker_sizes.iloc[i]),
                color='steelblue', ecolor='black', capsize=3, linewidth=1.5)
    ax.text(-0.02, i, row['study'], ha='right', va='center', fontsize=8, transform=ax.get_yaxis_transform())
    or_text = f"OR={np.exp(row['log_or']):.2f} [{np.exp(ci_lo):.2f},{np.exp(ci_hi):.2f}]"
    ax.text(1.02, i, or_text, ha='left', va='center', fontsize=7, transform=ax.get_yaxis_transform())

# Pooled diamond
y_pool = k + 0.5
or_lo_re = np.exp(ci_lo_re)
or_hi_re = np.exp(ci_hi_re)
or_re    = np.exp(theta_re)
diamond_x = [or_lo_re, or_re, or_hi_re, or_re, or_lo_re]
diamond_y = [y_pool, y_pool + 0.4, y_pool, y_pool - 0.4, y_pool]
ax.fill(diamond_x, diamond_y, color='#e74c3c', alpha=0.9, zorder=5)
ax.text(-0.02, y_pool, "Pooled (RE)", ha='right', va='center', fontsize=9,
        fontweight='bold', transform=ax.get_yaxis_transform())
ax.text(1.02, y_pool, f"OR={or_re:.2f} [{or_lo_re:.2f},{or_hi_re:.2f}]",
        ha='left', va='center', fontsize=9, fontweight='bold', transform=ax.get_yaxis_transform())

ax.axvline(1, color='black', linewidth=1, linestyle='--', alpha=0.7)
ax.set_xscale('log')
ax.set_xlim(0.1, 3.0)
ax.set_yticks([]); ax.set_xlabel("Odds Ratio (log scale)")
ax.set_title(f"Forest Plot — Hand Hygiene vs. Infection Risk\n"
             f"(k={k}, I²={I2:.0f}%, RE-OR={or_re:.2f})")
ax.text(0.5, -0.08, "← Favours intervention | Favours control →",
        ha='center', transform=ax.transAxes, fontsize=9, style='italic')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("forest_plot.png", dpi=150)
plt.show()
```

### Step 2: Publication Bias Assessment

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ------------------------------------------------------------------ #
# Egger's test and funnel plot for publication bias
# ------------------------------------------------------------------ #

# Use the study data from Step 1
log_or = df['log_or'].values
se = df['se'].values
precision = 1 / se
standard_normal_deviate = log_or / se

# ---- Egger's regression test ----------------------------------- #
X = sm.add_constant(precision)
egger_model = sm.OLS(standard_normal_deviate, X).fit()
intercept = egger_model.params[0]
intercept_se = egger_model.bse[0]
t_stat = intercept / intercept_se
p_egger = egger_model.pvalues[0]

print("=== Egger's Publication Bias Test ===")
print(f"Intercept = {intercept:.4f}  SE = {intercept_se:.4f}")
print(f"t = {t_stat:.3f},  p = {p_egger:.4f}")
print(f"Interpretation: {'Evidence of asymmetry (p<0.10)' if p_egger < 0.10 else 'No significant asymmetry (p≥0.10)'}")

# ---- Begg's rank correlation test ------------------------------ #
rank_log_or = stats.rankdata(log_or)
rank_se = stats.rankdata(se)
tau, p_begg = stats.kendalltau(rank_log_or, rank_se)
print(f"\nBegg's test: τ = {tau:.4f}, p = {p_begg:.4f}")

# ---- Funnel plot ----------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Standard funnel plot (SE vs. OR)
ax = axes[0]
for i in range(len(df)):
    ax.scatter(np.exp(log_or[i]), se[i], s=60, c='steelblue',
               edgecolors='black', linewidths=0.5, zorder=5)

# Pooled OR line
ax.axvline(np.exp(theta_re), color='red', linewidth=2, linestyle='--', label=f"RE OR={np.exp(theta_re):.2f}")

# Pseudo-confidence cone (95% CI region)
se_range = np.linspace(0, se.max() * 1.1, 100)
ax.fill_betweenx(se_range,
                  np.exp(theta_re - 1.96 * se_range),
                  np.exp(theta_re + 1.96 * se_range),
                  alpha=0.1, color='gray', label='95% CI region')
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlabel("Odds Ratio (log scale)"); ax.set_ylabel("Standard Error")
ax.set_title(f"Funnel Plot\n(Egger p={p_egger:.3f})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Egger regression plot
ax2 = axes[1]
ax2.scatter(precision, standard_normal_deviate, s=60, c='steelblue',
            edgecolors='black', linewidths=0.5)
x_fit = np.linspace(precision.min(), precision.max(), 100)
y_fit = egger_model.params[0] + egger_model.params[1] * x_fit
ax2.plot(x_fit, y_fit, 'r-', linewidth=2, label=f"Egger fit (intercept={intercept:.2f})")
ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
ax2.set_xlabel("Precision (1/SE)"); ax2.set_ylabel("Standard Normal Deviate")
ax2.set_title("Egger's Regression Plot"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("publication_bias.png", dpi=150)
plt.show()
```

### Step 3: Meta-Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------------------------------------------ #
# Meta-regression: explain heterogeneity with study-level covariates
# ------------------------------------------------------------------ #

# Study-level covariate: setting (hospital=1, school/community=0)
df['hospital'] = (df['setting'] == 'hospital').astype(float)
df['log_sample_size'] = np.log(np.random.randint(100, 5000, len(df)))  # Simulated

# Random-effects meta-regression (WLS with RE weights)
w_re_array = 1 / (df['se']**2 + tau2)

X_meta = sm.add_constant(df[['hospital', 'year']])
y_meta = df['log_or']

# Weighted least squares (approximates random-effects meta-regression)
meta_reg = sm.WLS(y_meta, X_meta, weights=w_re_array).fit()
print("=== Meta-Regression Results ===")
print(meta_reg.summary())

# Plot: effect modifier by setting
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Setting comparison
for ax, (setting_val, label, color) in zip(
    [axes[0], axes[0]],
    [(0, 'Community/School', 'steelblue'), (1, 'Hospital', '#e74c3c')]
):
    mask = df['hospital'] == setting_val
    ors = np.exp(df.loc[mask, 'log_or'])
    ci_los = np.exp(df.loc[mask, 'log_or'] - 1.96*df.loc[mask, 'se'])
    ci_his = np.exp(df.loc[mask, 'log_or'] + 1.96*df.loc[mask, 'se'])
    for or_val, ci_lo, ci_hi in zip(ors, ci_los, ci_his):
        ax.errorbar(or_val, label, xerr=[[or_val-ci_lo],[ci_hi-or_val]],
                    fmt='o', color=color, alpha=0.5, capsize=3)

axes[0].axvline(1, color='black', linewidth=1, linestyle='--')
axes[0].set_xscale('log'); axes[0].set_xlabel("Odds Ratio")
axes[0].set_title("Effect by Setting"); axes[0].grid(True, alpha=0.3)

# Year trend
axes[1].scatter(df['year'], np.exp(df['log_or']), s=w_re_array/w_re_array.max()*200,
                c='steelblue', edgecolors='black', linewidths=0.5, alpha=0.8)
year_range = np.linspace(df['year'].min(), df['year'].max(), 50)
# Predicted OR from meta-regression (holding hospital=0.5)
X_pred = pd.DataFrame({'const': 1, 'hospital': 0.5, 'year': year_range})
log_or_pred = meta_reg.predict(X_pred)
axes[1].plot(year_range, np.exp(log_or_pred), 'r-', linewidth=2, label='Meta-regression trend')
axes[1].axhline(1, color='gray', linestyle='--', linewidth=1)
axes[1].set_xlabel("Publication year"); axes[1].set_ylabel("Odds Ratio")
axes[1].set_title("Meta-Regression: Year Trend"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("meta_regression.png", dpi=150)
plt.show()

# Residual heterogeneity after meta-regression
Q_residual = np.sum(w_re_array * meta_reg.resid**2)
p_Q_resid = 1 - chi2.cdf(Q_residual, df=k-2-1)
print(f"\nResidual Q (after meta-regression): {Q_residual:.3f}, p={p_Q_resid:.4f}")
```

---

## Advanced Usage

### GRADE Evidence Quality Assessment

```python
import pandas as pd

# ------------------------------------------------------------------ #
# GRADE framework: rate quality of evidence
# ------------------------------------------------------------------ #

# RCT starts as "High", observational as "Low"
grade_criteria = {
    'Starting quality': 'High (RCT)',
    'Risk of bias':     {'assessment': 'Moderate risk', 'downgrade': -1},
    'Inconsistency':    {'assessment': f'I²={I2:.0f}% (Moderate)', 'downgrade': -1 if I2 > 40 else 0},
    'Indirectness':     {'assessment': 'Direct evidence', 'downgrade': 0},
    'Imprecision':      {'assessment': 'Wide CI (upgrades rare)', 'downgrade': 0},
    'Publication bias': {'assessment': f"Egger p={p_egger:.3f}", 'downgrade': -1 if p_egger < 0.10 else 0},
}

quality_levels = {4: 'High', 3: 'Moderate', 2: 'Low', 1: 'Very Low'}
starting = 4  # High = RCT
total_downgrade = sum(v.get('downgrade', 0) for k,v in grade_criteria.items() if isinstance(v, dict))
final_quality = max(1, starting + total_downgrade)

print("=== GRADE Evidence Assessment ===")
print(f"{'Domain':<25} {'Assessment':<35} {'Downgrade'}")
print("-" * 70)
for domain, info in grade_criteria.items():
    if isinstance(info, dict):
        print(f"{domain:<25} {info['assessment']:<35} {info['downgrade']:+d}")
    else:
        print(f"{domain:<25} {info}")
print("-" * 70)
print(f"Total downgrade: {total_downgrade}")
print(f"Final quality:   {quality_levels[final_quality]} (score={final_quality})")
print(f"Recommendation:  {'↑↑ Strong' if final_quality >= 3 else '↑ Conditional'} recommendation")
```

---

## Troubleshooting

### Negative τ² in DerSimonian-Laird

This can happen with Q < k-1. Truncate at 0:
```python
tau2 = max(0, (Q - (k-1)) / (np.sum(w_fe) - np.sum(w_fe**2)/np.sum(w_fe)))
```

### Forest plot labels overflow figure

```python
fig, ax = plt.subplots(figsize=(16, 10))  # Wider figure
ax.set_xlim([0.05, 5.0])  # Extend x-axis to make room for labels
```

### Meta-regression with few studies

Meta-regression requires k >> number of predictors. Rule of thumb: ≥10 studies per predictor variable.

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| statsmodels | 0.14 | `WLS`, `OLS` stable API |
| scipy | 1.11, 1.12 | `kendalltau` stable |
| numpy | 1.24, 1.26 | No known issues |

---

## External Resources

### Official Documentation

- [GRADE Working Group](https://www.gradeworkinggroup.org)
- [Cochrane Handbook](https://training.cochrane.org/handbook)

### Key Papers

- DerSimonian, R. & Laird, N. (1986). *Meta-analysis in clinical trials*. Controlled Clinical Trials.
- Higgins, J.P.T. & Thompson, S.G. (2002). *Quantifying heterogeneity in a meta-analysis*. Statistics in Medicine.
- Egger, M. et al. (1997). *Bias in meta-analysis detected by a simple, graphical test*. BMJ.

---

## Examples

### Example 1: Dose-Response Meta-Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Dose-response: alcohol consumption and cardiovascular risk
doses = np.array([0, 1, 2, 3, 4, 5])  # drinks/day
rrs   = np.array([1.00, 0.85, 0.90, 1.05, 1.25, 1.60])
ci_lo = np.array([1.00, 0.78, 0.83, 0.97, 1.14, 1.40])
ci_hi = np.array([1.00, 0.93, 0.98, 1.14, 1.37, 1.83])

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(doses, ci_lo, ci_hi, alpha=0.2, color='steelblue')
ax.plot(doses, rrs, 'bo-', linewidth=2, markersize=8)
ax.axhline(1, color='black', linewidth=1, linestyle='--')
ax.set_xlabel("Alcohol consumption (drinks/day)")
ax.set_ylabel("Relative Risk (95% CI)")
ax.set_title("Dose-Response: Alcohol and Cardiovascular Risk")
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("dose_response.png", dpi=150); plt.show()
```

### Example 2: Sensitivity Analysis (Influence Analysis)

```python
import numpy as np
import matplotlib.pyplot as plt

# Leave-one-out sensitivity analysis
k = len(df)
lo_ors = []
for i in range(k):
    mask = np.ones(k, dtype=bool); mask[i] = False
    lo_or = df['log_or'].values[mask]
    lo_se = df['se'].values[mask]
    w = 1 / (lo_se**2 + tau2)
    pooled = np.sum(w * lo_or) / np.sum(w)
    lo_ors.append(np.exp(pooled))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(df['study'], lo_ors, color='steelblue', edgecolor='black', linewidth=0.6, alpha=0.8)
ax.axvline(np.exp(theta_re), color='red', linewidth=2, linestyle='--', label=f"All studies: OR={np.exp(theta_re):.2f}")
ax.set_xlabel("Pooled OR (leave-one-out)")
ax.set_title("Leave-One-Out Sensitivity Analysis")
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig("sensitivity_analysis.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
