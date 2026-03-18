---
name: meta-analysis-psych
description: >
  Use this Skill for psychology meta-analysis: Cohen's d, Hedges' g,
  random-effects pooling, forest plot, publication bias (Egger, p-curve,
  PET-PEESE), and sensitivity analysis.
tags:
  - psychology
  - meta-analysis
  - effect-sizes
  - forest-plot
  - publication-bias
  - p-curve
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
    - numpy>=1.23
    - scipy>=1.9
    - pandas>=1.5
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Meta-Analysis for Psychology

> **TL;DR** — Complete meta-analysis pipeline in pure Python/NumPy: Cohen's d
> and Hedges' g from raw statistics, DerSimonian-Laird random-effects pooling,
> I² heterogeneity, forest plot with summary diamond, funnel plot, Egger test,
> trim-and-fill, PET-PEESE regression correction, and p-curve analysis.

---

## When to Use

Use this Skill when you need to:

- Pool effect sizes from multiple independent studies on the same research question
- Quantify and visualize heterogeneity across studies (I², τ²)
- Detect and correct for publication bias (Egger test, trim-and-fill, PET-PEESE)
- Assess whether p-values are consistent with a true effect (p-curve analysis)
- Generate publication-quality forest and funnel plots
- Run sensitivity analyses (leave-one-out, cumulative meta-analysis)

---

## Background

### Effect Size Measures

**Cohen's d** (raw mean difference standardized by pooled SD):

```
d = (M1 - M2) / SD_pooled
```

**Hedges' g** (small-sample bias correction for d):

```
J = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
g = J × d
Var(g) = (n1 + n2) / (n1 × n2) + g² / (2 × (n1 + n2))
```

### Random-Effects Model (DerSimonian-Laird)

Under the random-effects model, true effect sizes θi vary:

```
θi = θ + ui + εi       where ui ~ N(0, τ²), εi ~ N(0, vi)
```

Between-study variance τ² is estimated by the DerSimonian-Laird method:

```
Q    = Σ wi(gi - g_FE)²        (Cochran's Q statistic)
τ²   = max(0, (Q - (k-1)) / (Σwi - Σwi²/Σwi))
wi*  = 1 / (vi + τ²)           (RE weights)
g_RE = Σ(wi* × gi) / Σwi*
I²   = (Q - (k-1)) / Q × 100%  (% of total variance due to heterogeneity)
```

### Publication Bias Methods

| Method | Description |
|---|---|
| Funnel plot | Asymmetry suggests missing small-n null studies |
| Egger test | Regress std normal deviate on precision; intercept ≠ 0 = bias |
| Trim-and-fill | Impute mirror studies to restore funnel symmetry |
| PET-PEESE | Regress g on SE (PET) or SE² (PEESE); intercept = bias-corrected effect |
| p-curve | Distribution of p-values below .05; right-skew = real effect |

---

## Environment Setup

```bash
conda create -n meta_env python=3.11 -y
conda activate meta_env

pip install numpy>=1.23 scipy>=1.9 pandas>=1.5 matplotlib>=3.6

# Verify
python -c "import numpy, scipy, pandas, matplotlib; print('All OK')"
```

---

## Core Workflow

### Step 1 — Effect Size Computation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from typing import Optional, Dict, List, Tuple

# ── Cohen's d from various input formats ─────────────────────────────────────

def cohens_d_from_means(
    m1: float, m2: float,
    sd1: float, sd2: float,
    n1: int, n2: int,
) -> Dict:
    """
    Compute Cohen's d and Hedges' g from group means and SDs.

    Args:
        m1, m2:     Group means.
        sd1, sd2:   Group standard deviations.
        n1, n2:     Sample sizes.

    Returns:
        Dict with d, g, var_g, se_g, 95% CI.
    """
    # Pooled SD
    sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / sd_pooled

    # Hedges' g correction
    df_total = n1 + n2 - 2
    J = 1 - 3 / (4 * df_total - 1)
    g = J * d

    # Variance of g
    var_g = (n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2))
    se_g = np.sqrt(var_g)

    return {
        "d": round(d, 4),
        "g": round(g, 4),
        "J": round(J, 4),
        "var_g": round(var_g, 6),
        "se_g": round(se_g, 4),
        "ci_lo": round(g - 1.96 * se_g, 4),
        "ci_hi": round(g + 1.96 * se_g, 4),
        "n1": n1, "n2": n2,
    }


def cohens_d_from_t(
    t: float, n1: int, n2: int,
) -> Dict:
    """
    Compute Cohen's d from an independent-samples t-statistic.

    Args:
        t:      t-statistic (signed).
        n1, n2: Group sample sizes.

    Returns:
        Dict with d, g, var_g, se_g, 95% CI.
    """
    d = t * np.sqrt((n1 + n2) / (n1 * n2))
    df_total = n1 + n2 - 2
    J = 1 - 3 / (4 * df_total - 1)
    g = J * d
    var_g = (n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2))
    se_g = np.sqrt(var_g)
    return {
        "d": round(d, 4), "g": round(g, 4), "J": round(J, 4),
        "var_g": round(var_g, 6), "se_g": round(se_g, 4),
        "ci_lo": round(g - 1.96 * se_g, 4), "ci_hi": round(g + 1.96 * se_g, 4),
        "n1": n1, "n2": n2,
    }


def cohens_d_from_F(
    F: float, n1: int, n2: int,
) -> Dict:
    """
    Compute Cohen's d from a one-df F-ratio (two-group comparison).

    Args:
        F:      F-statistic (F = t²).
        n1, n2: Group sample sizes.

    Returns:
        Dict with d, g, var_g, se_g, 95% CI.
    """
    t = np.sqrt(F)  # F = t² for one-df tests
    return cohens_d_from_t(t, n1, n2)
```

### Step 2 — Random-Effects Meta-Analysis

```python
def run_meta_analysis(
    df: pd.DataFrame,
    g_col: str = "g",
    var_col: str = "var_g",
    study_col: str = "study",
) -> Dict:
    """
    DerSimonian-Laird random-effects meta-analysis.

    Args:
        df:       DataFrame with one row per study.
        g_col:    Column of Hedges' g effect sizes.
        var_col:  Column of effect size variances.
        study_col: Column of study labels.

    Returns:
        Dict with pooled estimate, heterogeneity statistics, and weights.
    """
    g = df[g_col].values
    v = df[var_col].values
    k = len(g)

    # Fixed-effects weights and estimate
    w_FE = 1 / v
    g_FE = np.sum(w_FE * g) / np.sum(w_FE)

    # Cochran's Q
    Q = np.sum(w_FE * (g - g_FE) ** 2)

    # DerSimonian-Laird tau²
    c = np.sum(w_FE) - np.sum(w_FE ** 2) / np.sum(w_FE)
    tau2 = max(0.0, (Q - (k - 1)) / c)

    # I²
    I2 = max(0.0, (Q - (k - 1)) / Q * 100) if Q > 0 else 0.0

    # Random-effects weights and estimate
    w_RE = 1 / (v + tau2)
    g_RE = np.sum(w_RE * g) / np.sum(w_RE)
    var_RE = 1 / np.sum(w_RE)
    se_RE = np.sqrt(var_RE)
    z_RE = g_RE / se_RE
    p_RE = 2 * stats.norm.sf(abs(z_RE))
    ci_lo = g_RE - 1.96 * se_RE
    ci_hi = g_RE + 1.96 * se_RE

    result = {
        "k": k,
        "g_RE": round(g_RE, 4),
        "se_RE": round(se_RE, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "z": round(z_RE, 3),
        "p": round(p_RE, 4),
        "Q": round(Q, 3),
        "Q_df": k - 1,
        "Q_p": round(float(stats.chi2.sf(Q, df=k - 1)), 4),
        "I2": round(I2, 1),
        "tau2": round(tau2, 6),
        "tau": round(np.sqrt(tau2), 4),
        "w_RE": w_RE,
        "g_FE": round(g_FE, 4),
    }

    print(
        f"Random-effects meta-analysis (k={k} studies):\n"
        f"  g = {g_RE:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p = {p_RE:.4f}\n"
        f"  Q({k-1}) = {Q:.2f}, p = {result['Q_p']:.4f}\n"
        f"  I² = {I2:.1f}%, τ = {result['tau']:.4f}\n"
        f"  Heterogeneity: "
        + ("low" if I2 < 25 else "moderate" if I2 < 75 else "high")
    )
    return result
```

### Step 3 — Forest Plot

```python
def forest_plot(
    df: pd.DataFrame,
    meta_result: Dict,
    g_col: str = "g",
    ci_lo_col: str = "ci_lo",
    ci_hi_col: str = "ci_hi",
    study_col: str = "study",
    sort_by: str = "g",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a forest plot with study CIs and summary diamond.

    Args:
        df:           Study-level DataFrame (one row per study).
        meta_result:  Output from run_meta_analysis().
        g_col:        Column of effect sizes.
        ci_lo_col:    Column of lower 95% CIs.
        ci_hi_col:    Column of upper 95% CIs.
        study_col:    Column of study labels.
        sort_by:      Sort studies by 'g' (default) or 'year'.
        output_path:  Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    if sort_by == "g":
        df = df.sort_values(g_col).reset_index(drop=True)

    k = len(df)
    g_RE = meta_result["g_RE"]
    ci_lo_RE = meta_result["ci_lo"]
    ci_hi_RE = meta_result["ci_hi"]

    fig, ax = plt.subplots(figsize=(10, max(6, k * 0.45 + 2)))
    y_positions = list(range(k, 0, -1))

    # Study rows
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        g = row[g_col]
        lo = row[ci_lo_col]
        hi = row[ci_hi_col]
        ci_width = hi - lo
        # Marker size proportional to inverse variance (weight)
        w = 1 / (hi - lo) ** 2 if (hi - lo) > 0 else 1
        ms = min(max(4, w * 0.5), 16)

        ax.plot([lo, hi], [y, y], color="black", linewidth=1)
        ax.plot(g, y, "s", color="steelblue", markersize=ms, zorder=5)
        ax.text(-0.05 + min(df[ci_lo_col]) - 0.3, y,
                row[study_col], ha="right", va="center", fontsize=8)
        ax.text(max(df[ci_hi_col]) + 0.3, y,
                f"{g:.2f} [{lo:.2f}, {hi:.2f}]",
                ha="left", va="center", fontsize=7, color="dimgray")

    # Summary diamond
    y_summary = 0
    diamond_x = [ci_lo_RE, g_RE, ci_hi_RE, g_RE]
    diamond_y = [y_summary, y_summary + 0.4, y_summary, y_summary - 0.4]
    ax.fill(diamond_x, diamond_y, color="crimson", alpha=0.85, zorder=6)

    ax.text(max(df[ci_hi_col]) + 0.3, y_summary,
            f"{g_RE:.2f} [{ci_lo_RE:.2f}, {ci_hi_RE:.2f}]",
            ha="left", va="center", fontsize=8, fontweight="bold", color="crimson")

    # Reference line at 0
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    ax.set_yticks([])
    ax.set_xlabel("Hedges' g")
    ax.set_title(
        f"Forest Plot — {k} Studies\n"
        f"g = {g_RE:.3f}, I² = {meta_result['I2']:.1f}%, "
        f"τ = {meta_result['tau']:.3f}"
    )
    ax.set_xlim(min(df[ci_lo_col]) - 1.0, max(df[ci_hi_col]) + 1.5)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.show()
    return fig
```

---

## Advanced Usage

### Publication Bias: Egger Test, Trim-and-Fill, PET-PEESE

```python
def egger_test(
    df: pd.DataFrame,
    g_col: str = "g",
    var_col: str = "var_g",
) -> Dict:
    """
    Egger's test for funnel plot asymmetry (publication bias).

    Regresses standardized effect sizes on precision:
        z = g / se   (standard normal deviate)
        w = 1 / se   (precision)
        Regression: z = a + b × w
    Intercept a ≠ 0 indicates asymmetry.

    Args:
        df:      Study DataFrame.
        g_col:   Effect size column.
        var_col: Variance column.

    Returns:
        Dict with intercept, SE, t-stat, p-value.
    """
    g = df[g_col].values
    v = df[var_col].values
    se = np.sqrt(v)
    z = g / se   # standard normal deviate
    prec = 1 / se  # precision

    slope, intercept, r, p_val, se_intercept = stats.linregress(prec, z)
    t_stat = intercept / se_intercept
    df_resid = len(g) - 2
    p_intercept = 2 * stats.t.sf(abs(t_stat), df=df_resid)

    result = {
        "intercept": round(intercept, 4),
        "se_intercept": round(se_intercept, 4),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_intercept, 4),
        "significant": p_intercept < 0.05,
        "slope": round(slope, 4),
    }
    print(
        f"Egger test: intercept = {intercept:.3f} (SE = {se_intercept:.3f}), "
        f"t = {t_stat:.3f}, p = {p_intercept:.4f} "
        + ("← SIGNIFICANT bias" if p_intercept < 0.05 else "← non-significant")
    )
    return result


def pet_peese(
    df: pd.DataFrame,
    g_col: str = "g",
    var_col: str = "var_g",
    alpha_fat: float = 0.10,
) -> Dict:
    """
    PET-PEESE bias correction (Stanley & Doucouliagos, 2014).

    FAT-PET: g_i = β0 + β1 × SE_i + ε_i
        β0 = bias-corrected effect when SE = 0 (no bias)
    PEESE: g_i = β0 + β1 × SE_i² + ε_i
        Use if FAT-PET β0 is significant.

    Args:
        df:         Study DataFrame.
        g_col:      Effect size column.
        var_col:    Variance column.
        alpha_fat:  Significance threshold for FAT-PET intercept (default 0.10).

    Returns:
        Dict with PET estimate, PEESE estimate, and recommendations.
    """
    g = df[g_col].values
    v = df[var_col].values
    se = np.sqrt(v)
    w = 1 / v

    def wls_intercept(x_vals, y_vals, weights):
        """WLS regression — return intercept and p-value."""
        W = np.diag(weights)
        X = np.column_stack([np.ones(len(y_vals)), x_vals])
        XWX_inv = np.linalg.inv(X.T @ W @ X)
        coef = XWX_inv @ X.T @ W @ y_vals
        residuals = y_vals - X @ coef
        sigma2 = np.sum(weights * residuals**2) / (len(y_vals) - 2)
        var_coef = sigma2 * XWX_inv
        se_intercept = np.sqrt(var_coef[0, 0])
        t_stat = coef[0] / se_intercept
        p_val = 2 * stats.t.sf(abs(t_stat), df=len(y_vals) - 2)
        return coef[0], se_intercept, t_stat, p_val

    # PET (FAT: regress g on SE)
    pet_b0, pet_se, pet_t, pet_p = wls_intercept(se, g, w)

    # PEESE (regress g on SE²)
    peese_b0, peese_se, peese_t, peese_p = wls_intercept(se**2, g, w)

    # Decision rule
    if pet_p < alpha_fat:
        recommended = "PEESE"
        recommended_estimate = peese_b0
    else:
        recommended = "PET (effect non-significant)"
        recommended_estimate = pet_b0

    result = {
        "PET_estimate": round(pet_b0, 4),
        "PET_se": round(pet_se, 4),
        "PET_p": round(pet_p, 4),
        "PEESE_estimate": round(peese_b0, 4),
        "PEESE_se": round(peese_se, 4),
        "PEESE_p": round(peese_p, 4),
        "recommended_method": recommended,
        "corrected_estimate": round(recommended_estimate, 4),
    }
    print(
        f"PET-PEESE:\n"
        f"  PET:   β0 = {pet_b0:.4f} (SE = {pet_se:.4f}), p = {pet_p:.4f}\n"
        f"  PEESE: β0 = {peese_b0:.4f} (SE = {peese_se:.4f}), p = {peese_p:.4f}\n"
        f"  Recommended: {recommended} → corrected estimate = {recommended_estimate:.4f}"
    )
    return result


def p_curve_analysis(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    P-curve analysis for evidential value assessment.

    Under H0 (no true effect), p-values are uniform on [0, alpha].
    Under H1 (true effect), p-values are right-skewed (more very small p's).
    The p-curve plots conditional distribution of p-values < alpha.

    Args:
        p_values: Array of p-values from studies (all p < alpha are included).
        alpha:    Significance threshold.

    Returns:
        Dict with binomial test for right-skew and uniform comparison.
    """
    sig_p = p_values[p_values < alpha]
    k_sig = len(sig_p)

    if k_sig < 3:
        print("Warning: fewer than 3 significant p-values — p-curve unreliable.")
        return {"warning": "insufficient significant studies", "k_sig": k_sig}

    # Expected proportion below alpha/2 under uniform H0 distribution
    expected_prop = 0.5  # uniform: 50% below alpha/2 = 0.025
    observed_prop = np.mean(sig_p < alpha / 2)

    binom_result = stats.binom_test(
        int(observed_prop * k_sig), k_sig, p=expected_prop, alternative="greater"
    ) if hasattr(stats, "binom_test") else stats.binomtest(
        int(observed_prop * k_sig), k_sig, p=expected_prop, alternative="greater"
    ).pvalue

    result = {
        "k_sig": k_sig,
        "observed_prop_below_half_alpha": round(observed_prop, 4),
        "expected_prop_H0": expected_prop,
        "p_right_skew": round(float(binom_result), 4),
        "evidential_value": float(binom_result) < 0.05,
    }
    print(
        f"P-curve (k_sig={k_sig}):\n"
        f"  Prop p < {alpha/2:.3f}: {observed_prop:.3f} (expected under H0: {expected_prop})\n"
        f"  Right-skew test p = {float(binom_result):.4f} "
        + ("← evidential value present" if float(binom_result) < 0.05
           else "← insufficient evidential value")
    )
    return result
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| τ² is 0 (homogeneous) | Low heterogeneity; Q < k-1 | Report as-is; fixed-effects model appropriate |
| I² near 100% | Very heterogeneous studies | Report sub-group analyses; reconsider pooling |
| Egger test false positive with large k | Multiple comparisons | Use Egger only as exploratory; also check funnel |
| PET-PEESE gives impossible estimate | Small-study effect inflated d | Report with caution; use sensitivity analysis |
| p-curve requires `binom_test` | Older scipy version | Use `scipy.stats.binomtest` (scipy >= 1.7) |
| Forest plot labels overlap | Many studies | Reduce `fontsize` or increase figure height |
| Negative CI lower bound for g | Very small studies | Expected; report accurately |

---

## External Resources

- DerSimonian, R., & Laird, N. (1986). Meta-analysis in clinical trials.
  *Controlled Clinical Trials*, 7(3), 177–188.
- Hedges, L. V., & Olkin, I. (1985). *Statistical Methods for Meta-Analysis.*
- Egger, M., et al. (1997). Bias in meta-analysis detected by funnel plot.
  *BMJ*, 315, 629–634.
- Stanley, T. D., & Doucouliagos, H. (2014). Meta-regression approximations.
  *Research Synthesis Methods*, 5(1), 60–78.
- Simonsohn, U., Nelson, L. D., & Simmons, J. P. (2014). p-Curve.
  *Journal of Experimental Psychology: General*, 143(2), 534–547.
- metafor R package: <https://www.metafor-project.org/>

---

## Examples

### Example 1 — Full Meta-Analysis Pipeline with Forest Plot

```python
# Hypothetical studies on working-memory training and fluid intelligence
studies_data = [
    {"study": "Smith 2015",     "m1": 52.3, "sd1": 9.2, "n1": 30, "m2": 47.1, "sd2": 8.8, "n2": 30},
    {"study": "Jones 2016",     "m1": 61.2, "sd1": 11.0, "n1": 45, "m2": 57.8, "sd2": 10.5, "n2": 45},
    {"study": "Brown 2017",     "m1": 55.0, "sd1": 8.5,  "n1": 25, "m2": 49.3, "sd2": 9.0, "n2": 25},
    {"study": "Garcia 2018",    "m1": 70.1, "sd1": 13.2, "n1": 60, "m2": 66.4, "sd2": 12.8, "n2": 60},
    {"study": "Patel 2019",     "m1": 48.5, "sd1": 7.9,  "n1": 20, "m2": 44.2, "sd2": 8.1, "n2": 20},
    {"study": "Chen 2020",      "m1": 65.3, "sd1": 10.8, "n1": 35, "m2": 60.7, "sd2": 11.2, "n2": 35},
    {"study": "Wilson 2021",    "m1": 58.9, "sd1": 9.7,  "n1": 40, "m2": 53.2, "sd2": 9.3, "n2": 40},
    {"study": "Rodriguez 2022", "m1": 53.1, "sd1": 8.3,  "n1": 28, "m2": 48.7, "sd2": 8.0, "n2": 28},
]

# Compute effect sizes
rows = []
for s in studies_data:
    es = cohens_d_from_means(
        s["m1"], s["m2"], s["sd1"], s["sd2"], s["n1"], s["n2"]
    )
    rows.append({
        "study": s["study"],
        "n1": s["n1"], "n2": s["n2"],
        "g": es["g"], "var_g": es["var_g"],
        "ci_lo": es["ci_lo"], "ci_hi": es["ci_hi"],
    })

df_meta = pd.DataFrame(rows)
print("Effect sizes:")
print(df_meta[["study", "g", "ci_lo", "ci_hi"]].round(3))

# Random-effects meta-analysis
meta_res = run_meta_analysis(df_meta)

# Forest plot
forest_plot(df_meta, meta_res, output_path="forest_plot.png")

# Publication bias
egger_res = egger_test(df_meta)
pet_res = pet_peese(df_meta)
```

### Example 2 — Funnel Plot, Egger Test, and P-Curve

```python
# Funnel plot
fig, ax = plt.subplots(figsize=(7, 6))
se = np.sqrt(df_meta["var_g"].values)
ax.scatter(df_meta["g"].values, se, alpha=0.7, color="steelblue", s=60)
ax.axvline(meta_res["g_RE"], color="crimson", linestyle="--", linewidth=1.5, label="g_RE")
ax.axvline(0, color="gray", linestyle=":", linewidth=1)
ax.invert_yaxis()
ax.set_xlabel("Hedges' g")
ax.set_ylabel("Standard Error (SE)")
ax.set_title("Funnel Plot")
ax.legend()
fig.tight_layout()
plt.savefig("funnel_plot.png", dpi=150)
plt.show()

# Simulate p-values from effect sizes and sample sizes
p_vals_list = []
for _, row in df_meta.iterrows():
    t_stat = row["g"] / np.sqrt(row["var_g"])
    df_test = row["n1"] + row["n2"] - 2
    p_val = 2 * stats.t.sf(abs(t_stat), df=df_test)
    p_vals_list.append(p_val)

p_vals = np.array(p_vals_list)
print(f"\nP-values: {np.round(p_vals, 4)}")

# P-curve analysis
p_curve_res = p_curve_analysis(p_vals, alpha=0.05)

# P-curve visualization
sig_p = p_vals[p_vals < 0.05]
fig2, ax2 = plt.subplots(figsize=(8, 4))
bins = np.linspace(0, 0.05, 6)
ax2.hist(sig_p, bins=bins, color="steelblue", alpha=0.7, edgecolor="white", label="Observed")
ax2.axhline(len(sig_p) / 5, color="crimson", linestyle="--",
            linewidth=2, label="Expected under H0 (uniform)")
ax2.set_xlabel("p-value")
ax2.set_ylabel("Frequency")
ax2.set_title("P-Curve")
ax2.legend()
fig2.tight_layout()
plt.savefig("p_curve.png", dpi=150)
plt.show()
print("Meta-analysis complete.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Hedges' g, DL random-effects, forest/funnel plots, Egger, PET-PEESE, p-curve |
