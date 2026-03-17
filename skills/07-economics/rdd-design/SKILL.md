---
name: rdd-design
description: >
  Regression discontinuity design: sharp and fuzzy RDD, bandwidth selection, manipulation
  tests, covariate balance checks, and rdrobust-style Python/R implementation.
tags:
  - econometrics
  - causal-inference
  - rdd
  - quasi-experiment
  - python
  - policy-evaluation
version: "1.0.0"
authors:
  - name: "awesome-rosetta-skills contributors"
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - numpy>=1.24.0
  - pandas>=2.0.0
  - statsmodels>=0.14.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - rdrobust>=0.1.0
last_updated: "2026-03-17"
---

# Regression Discontinuity Design

Regression Discontinuity Design (RDD) exploits a known threshold in a continuous assignment
variable ("running variable") that determines treatment. Units just above and just below the
cutoff are locally comparable — providing near-experimental variation for causal identification.

---

## Conceptual Framework

### Sharp RDD

Treatment is a deterministic function of the running variable X:

D_i = 1(X_i ≥ c)

The treatment effect τ_SRD is identified as:

τ_SRD = lim_{x↓c} E[Y|X=x] − lim_{x↑c} E[Y|X=x]

Identification requires:
1. **Continuity**: E[Y(0)|X=x] and E[Y(1)|X=x] are continuous at c.
2. **No precise manipulation**: agents cannot perfectly sort just above the cutoff.

### Fuzzy RDD

When the probability of treatment changes discontinuously at c (but not from 0 to 1):

τ_FRD = jump in E[Y|X] at c / jump in E[D|X] at c

This is a local IV estimator; it recovers the LATE for compliers near the threshold.

### Bandwidth Selection

The MSE-optimal bandwidth (Calonico, Cattaneo, Titiunik 2014):

h_MSE = C_n · n^{-1/5}

In practice `rdrobust` computes data-driven bandwidths via IK (Imbens-Kalyanaraman) or
CCT (Calonico-Cattaneo-Titiunik) selectors.

---

## Full Implementation

```python
# rdd_design.py
"""
Regression Discontinuity Design toolkit.
Covers: sharp RDD, fuzzy RDD, manipulation test, covariate balance, bandwidth sensitivity.
Requires: numpy, pandas, statsmodels, scipy, matplotlib
Optional: rdrobust (pip install rdrobust)
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

try:
    import rdrobust as rdr
    HAS_RDROBUST = True
except ImportError:
    HAS_RDROBUST = False

COLORS = {"treat": "#d7191c", "control": "#2c7bb6", "neutral": "#636363",
          "ci": "#abd9e9"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. MANIPULATION TEST (McCrary / rddensity)
# ─────────────────────────────────────────────────────────────────────────────

def test_manipulation(running_var: np.ndarray, cutoff: float,
                      n_bins: int = 30) -> dict:
    """
    McCrary (2008) density continuity test for sorting / manipulation.

    H0: density of running variable is continuous at the cutoff.
    Rejects if agents can precisely manipulate their score to just exceed c.

    Method: estimate local linear density on each side of cutoff using
    a bin count approximation; compare slopes.

    Parameters
    ----------
    running_var : 1-D array of running variable values
    cutoff      : threshold value c
    n_bins      : number of histogram bins per side (default 30)

    Returns
    -------
    dict with t_statistic, p_value, interpretation, and density arrays
    """
    x = np.asarray(running_var)
    x_c = x - cutoff   # center at cutoff

    # separate sides
    x_left  = x_c[x_c < 0]
    x_right = x_c[x_c >= 0]

    # bin width: half the standard bandwidth rule
    bw = 2 * 1.06 * np.std(x_c) * len(x_c)**(-0.2)

    def _local_density(side, sign):
        """Fit local linear to binned density on one side."""
        mn, mx = (side.min(), 0) if sign < 0 else (0, side.max())
        bins   = np.linspace(mn, mx, n_bins + 1)
        counts, edges = np.histogram(side, bins=bins)
        mids   = (edges[:-1] + edges[1:]) / 2
        density = counts / (len(x_c) * (edges[1] - edges[0]))
        # local linear fit weighted by triangular kernel near 0
        kernel = np.maximum(0, 1 - np.abs(mids) / (mx - mn))
        w = sm.WLS(density, sm.add_constant(mids), weights=kernel).fit()
        return w.params[0], w.bse[0], mids, density  # intercept at 0

    int_left,  se_left,  m_l, d_l = _local_density(x_left,  -1)
    int_right, se_right, m_r, d_r = _local_density(x_right,  1)

    t_stat = (int_right - int_left) / np.sqrt(se_left**2 + se_right**2)
    p_val  = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "density_left_at_cutoff": int_left,
        "density_right_at_cutoff": int_right,
        "reject_H0": p_val < 0.05,
        "interpretation": (
            f"Potential manipulation detected (p={p_val:.4f})" if p_val < 0.05
            else f"No evidence of manipulation (p={p_val:.4f})"
        ),
        "_bins_left": (m_l, d_l),
        "_bins_right": (m_r, d_r),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. SHARP RDD
# ─────────────────────────────────────────────────────────────────────────────

def run_sharp_rdd(
    y: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    bandwidth: float = None,
    poly_order: int = 1,
    kernel: str = "triangular",
) -> dict:
    """
    Estimate sharp RDD treatment effect using local polynomial regression.

    Parameters
    ----------
    y         : outcome variable
    x         : running variable
    cutoff    : threshold c
    bandwidth : half-bandwidth h; if None, uses IK-style MSE-optimal selector
    poly_order: polynomial order for local regression (1 = local linear)
    kernel    : 'triangular', 'uniform', or 'epanechnikov'

    Returns
    -------
    dict with tau (LATE), se, t_stat, p_value, ci, bandwidth, and raw results
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_c = x - cutoff   # re-center

    if bandwidth is None:
        bandwidth = _ik_bandwidth(y, x_c)

    weights = _kernel_weights(x_c, bandwidth, kernel)
    mask    = weights > 0

    y_w   = y[mask]
    xc_w  = x_c[mask]
    wts   = weights[mask]
    treat = (xc_w >= 0).astype(float)

    # build polynomial interaction design matrix
    cols = {"const": np.ones(mask.sum()), "treat": treat}
    for p in range(1, poly_order + 1):
        cols[f"x{p}"]       = xc_w ** p
        cols[f"treat_x{p}"] = treat * xc_w ** p
    X_mat = np.column_stack(list(cols.values()))
    col_names = list(cols.keys())

    model  = sm.WLS(y_w, X_mat, weights=wts).fit(
        cov_type="HC3"
    )
    tau_idx = col_names.index("treat")
    tau = model.params[tau_idx]
    se  = model.bse[tau_idx]
    t   = model.tvalues[tau_idx]
    p   = model.pvalues[tau_idx]
    ci  = model.conf_int().iloc[tau_idx].tolist()

    return {
        "tau": tau,
        "se": se,
        "t_statistic": t,
        "p_value": p,
        "ci_95": ci,
        "bandwidth": bandwidth,
        "poly_order": poly_order,
        "kernel": kernel,
        "n_left":  int((xc_w < 0).sum()),
        "n_right": int((xc_w >= 0).sum()),
        "interpretation": (
            f"Treatment effect τ = {tau:.4f} (SE={se:.4f}, p={p:.4f}); "
            + ("Statistically significant at 5%" if p < 0.05
               else "Not statistically significant at 5%")
        ),
        "_model": model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FUZZY RDD
# ─────────────────────────────────────────────────────────────────────────────

def run_fuzzy_rdd(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cutoff: float,
    bandwidth: float = None,
    poly_order: int = 1,
    kernel: str = "triangular",
) -> dict:
    """
    Fuzzy RDD: local IV using cutoff as instrument for treatment.

    τ_FRDD = (jump in E[Y|X] at c) / (jump in E[D|X] at c)

    Parameters
    ----------
    y      : outcome
    x      : running variable
    z      : actual treatment take-up (binary or continuous compliance)
    cutoff : threshold
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    x_c = x - cutoff

    if bandwidth is None:
        bandwidth = _ik_bandwidth(y, x_c)

    # reduced form: effect of crossing on Y
    rf = run_sharp_rdd(y, x, cutoff, bandwidth, poly_order, kernel)
    # first stage: effect of crossing on treatment take-up
    fs = run_sharp_rdd(z, x, cutoff, bandwidth, poly_order, kernel)

    if abs(fs["tau"]) < 1e-10:
        raise ValueError("First stage is essentially zero — instrument is weak.")

    tau_fuzzy = rf["tau"] / fs["tau"]
    # delta method SE
    se_fuzzy = np.sqrt(
        (rf["se"] / fs["tau"])**2
        + (rf["tau"] * fs["se"] / fs["tau"]**2)**2
    )
    t_fuzzy = tau_fuzzy / se_fuzzy
    p_fuzzy = 2 * (1 - stats.norm.cdf(abs(t_fuzzy)))

    return {
        "tau_fuzzy":        tau_fuzzy,
        "se":               se_fuzzy,
        "t_statistic":      t_fuzzy,
        "p_value":          p_fuzzy,
        "ci_95":            [tau_fuzzy - 1.96 * se_fuzzy,
                             tau_fuzzy + 1.96 * se_fuzzy],
        "first_stage_tau":  fs["tau"],
        "first_stage_F":    fs["t_statistic"]**2,
        "reduced_form_tau": rf["tau"],
        "bandwidth":        bandwidth,
        "interpretation": (
            f"Fuzzy RDD LATE = {tau_fuzzy:.4f} (SE={se_fuzzy:.4f}, p={p_fuzzy:.4f})"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. RDD PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_rdd(
    y: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    bandwidth: float = None,
    n_bins: int = 40,
    title: str = "Regression Discontinuity",
    ylabel: str = "Outcome",
    xlabel: str = "Running Variable",
) -> plt.Figure:
    """
    Binned scatter plot with local linear fit on each side of the cutoff.
    Follows the rdplot style of Calonico, Cattaneo, Titiunik (2015).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_c = x - cutoff

    if bandwidth is None:
        bandwidth = _ik_bandwidth(y, x_c)

    fig, ax = plt.subplots(figsize=(10, 6))

    # binned means (IMSE-optimal number of bins per side ≈ n^{1/3}/2)
    for side, color, label in [(-1, COLORS["control"], "Control"),
                                 (1,  COLORS["treat"],   "Treatment")]:
        mask = (x_c * side >= 0)
        x_s  = x_c[mask]
        y_s  = y[mask]
        n_s  = max(5, int(len(x_s)**(1/3)) * 2)
        bins = np.linspace(x_s.min(), x_s.max(), n_s + 1)
        bin_idx = np.digitize(x_s, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_s - 1)
        mids = [(bins[i] + bins[i+1])/2 for i in range(n_s)]
        means = [y_s[bin_idx == i].mean() for i in range(n_s)
                 if (bin_idx == i).sum() > 0]
        mids_valid = [mids[i] for i in range(n_s)
                      if (bin_idx == i).sum() > 0]
        ax.scatter([m + cutoff for m in mids_valid], means, color=color,
                   s=30, alpha=0.8, label=f"{label} bins", zorder=3)

        # local linear fit within bandwidth
        bw_mask = np.abs(x_s) <= bandwidth
        if bw_mask.sum() > 3:
            fit_x = np.linspace(
                max(x_s.min(), -bandwidth) if side < 0 else 0,
                0 if side < 0 else min(x_s.max(), bandwidth),
                200
            )
            model_s = np.polyfit(x_s[bw_mask], y_s[bw_mask], 1)
            ax.plot(fit_x + cutoff, np.polyval(model_s, fit_x),
                    color=color, linewidth=2, zorder=4)

    ax.axvline(cutoff, color="black", linestyle="--", linewidth=1.5,
               label=f"Cutoff = {cutoff}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. COVARIATE BALANCE
# ─────────────────────────────────────────────────────────────────────────────

def covariate_balance_rdd(
    df: pd.DataFrame,
    running_var: str,
    cutoff: float,
    covariates: list,
    bandwidth: float = None,
) -> pd.DataFrame:
    """
    Test that predetermined covariates are balanced across the cutoff.
    Run sharp RDD for each covariate as the outcome — reject H0 indicates imbalance.

    Returns a DataFrame with tau, se, p_value, and pass/fail for each covariate.
    """
    x = df[running_var].values
    if bandwidth is None:
        bandwidth = _ik_bandwidth(df[covariates[0]].values, x - cutoff)

    rows = []
    for cov in covariates:
        res = run_sharp_rdd(
            df[cov].values, x, cutoff, bandwidth=bandwidth
        )
        rows.append({
            "covariate": cov,
            "tau":       res["tau"],
            "se":        res["se"],
            "p_value":   res["p_value"],
            "balanced":  res["p_value"] >= 0.1,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. BANDWIDTH SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

def rdd_sensitivity_bandwidth(
    y: np.ndarray,
    x: np.ndarray,
    cutoff: float,
    h_grid: np.ndarray = None,
    poly_order: int = 1,
) -> pd.DataFrame:
    """
    Estimate sharp RDD across a grid of bandwidths and report stability.

    A credible RDD should show τ stable across a reasonable bandwidth range.
    Large swings suggest model sensitivity or violation of continuity assumption.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_c = x - cutoff

    if h_grid is None:
        h_opt = _ik_bandwidth(y, x_c)
        h_grid = np.linspace(0.3 * h_opt, 2.5 * h_opt, 20)

    rows = []
    for h in h_grid:
        try:
            res = run_sharp_rdd(y, x, cutoff, bandwidth=h,
                                poly_order=poly_order)
            rows.append({
                "bandwidth": h,
                "tau":       res["tau"],
                "se":        res["se"],
                "ci_lower":  res["ci_95"][0],
                "ci_upper":  res["ci_95"][1],
                "p_value":   res["p_value"],
                "n_left":    res["n_left"],
                "n_right":   res["n_right"],
            })
        except Exception:
            pass

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _kernel_weights(x_c: np.ndarray, h: float, kernel: str) -> np.ndarray:
    u = np.abs(x_c) / h
    if kernel == "triangular":
        return np.maximum(0, 1 - u)
    elif kernel == "epanechnikov":
        return np.maximum(0, 0.75 * (1 - u**2))
    else:  # uniform
        return (u <= 1).astype(float)


def _ik_bandwidth(y: np.ndarray, x_c: np.ndarray) -> float:
    """
    Simplified Imbens-Kalyanaraman (2012) bandwidth selector.
    Uses a pilot bandwidth (Silverman) and regularises the optimal formula.
    """
    n = len(y)
    sigma_x = np.std(x_c)
    # pilot bandwidth
    h1 = 1.84 * sigma_x * n**(-1/5)

    left  = (x_c >= -h1) & (x_c < 0)
    right = (x_c >= 0)   & (x_c < h1)

    if left.sum() < 5 or right.sum() < 5:
        return sigma_x

    # curvature estimates via second-order polynomial fits
    def _curvature(mask):
        if mask.sum() < 4:
            return 0.0
        c = np.polyfit(x_c[mask], y[mask], 2)
        return 2 * abs(c[0])

    m2_l = _curvature(left)
    m2_r = _curvature(right)

    # regularisation constants
    r_l  = np.var(y[left])  / (h1**4)
    r_r  = np.var(y[right]) / (h1**4)

    ck   = 3.4375  # triangular kernel constant
    num  = 2 * ck * np.var(y)
    den  = n * ((m2_r - m2_l)**2 + (r_l + r_r))
    if den <= 0:
        return sigma_x
    h_opt = (num / den) ** (1/5)
    return max(h_opt, 0.1 * sigma_x)
```

---

## Example A — Education Policy at a Test Score Threshold

A school district assigns remedial tutoring to students scoring below 50 on a diagnostic
exam. We estimate the causal effect on year-end scores.

```python
# example_a_education_rdd.py
"""
Sharp RDD: effect of remedial tutoring assigned at test score < 50.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdd_design import (
    test_manipulation,
    run_sharp_rdd,
    plot_rdd,
    covariate_balance_rdd,
    rdd_sensitivity_bandwidth,
)

rng = np.random.default_rng(1234)
n   = 3000

# running variable: baseline test score (0–100)
score = rng.beta(5, 5, n) * 100
cutoff = 50.0

# treatment: tutoring if score < 50
treated = (score < cutoff).astype(float)

# covariates (pre-determined — should be balanced at cutoff)
age     = 10 + rng.normal(0, 0.5, n)
female  = rng.binomial(1, 0.5, n).astype(float)
ses     = rng.normal(0, 1, n)              # socioeconomic status index

# outcome: end-of-year test score
# true effect of tutoring: +6 points for compliers near cutoff
noise   = rng.normal(0, 8, n)
outcome = (
    40 + 0.5 * score
    - 0.3 * (score - cutoff)**2 * 0.005
    + 6 * treated
    - 2 * ses
    + noise
)

df = pd.DataFrame({
    "outcome": outcome, "score": score, "treated": treated,
    "age": age, "female": female, "ses": ses,
})

# ── 1. Manipulation test ──
manip = test_manipulation(df["score"].values, cutoff)
print("Manipulation test:", manip["interpretation"])

# ── 2. Main RDD estimate ──
result = run_sharp_rdd(df["outcome"].values, df["score"].values, cutoff)
print(f"\nSharp RDD τ = {result['tau']:.3f}  SE = {result['se']:.3f}  "
      f"p = {result['p_value']:.4f}")
print(f"95% CI: [{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}]")
print(f"Bandwidth: {result['bandwidth']:.2f}  N (left/right): "
      f"{result['n_left']}/{result['n_right']}")

# ── 3. Covariate balance ──
balance = covariate_balance_rdd(df, "score", cutoff, ["age", "female", "ses"])
print("\nCovariate balance:")
print(balance.to_string(index=False))

# ── 4. Bandwidth sensitivity ──
sensitivity = rdd_sensitivity_bandwidth(
    df["outcome"].values, df["score"].values, cutoff
)
print("\nBandwidth sensitivity (τ range):",
      f"[{sensitivity['tau'].min():.3f}, {sensitivity['tau'].max():.3f}]")

# ── 5. RDD plot ──
fig = plot_rdd(
    df["outcome"].values, df["score"].values, cutoff,
    title="Effect of Remedial Tutoring on End-of-Year Score",
    ylabel="End-of-year test score",
    xlabel="Baseline test score",
)

# overlay bandwidth sensitivity as inset
ax_inset = fig.add_axes([0.62, 0.15, 0.28, 0.30])
ax_inset.fill_between(sensitivity["bandwidth"], sensitivity["ci_lower"],
                       sensitivity["ci_upper"], alpha=0.3, color="#2c7bb6")
ax_inset.plot(sensitivity["bandwidth"], sensitivity["tau"],
              color="#2c7bb6", linewidth=2)
ax_inset.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax_inset.set_xlabel("Bandwidth", fontsize=8)
ax_inset.set_ylabel("τ", fontsize=8)
ax_inset.set_title("BW sensitivity", fontsize=8)

fig.savefig("rdd_education.png", dpi=150, bbox_inches="tight")
print("\nSaved rdd_education.png")
```

---

## Example B — Incumbency Advantage in Elections

Lee (2008) style: the running variable is the Democratic vote margin in election t;
the outcome is winning in election t+1.  The cutoff is 0 (bare majority).

```python
# example_b_incumbency_rdd.py
"""
Fuzzy RDD: incumbency advantage.
Running variable: vote share margin = DEM% - 50.
Outcome: probability of winning next election.
Treatment: actually serving as incumbent (compliance < 1 due to death, resignation, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdd_design import (
    test_manipulation,
    run_sharp_rdd,
    run_fuzzy_rdd,
    plot_rdd,
    rdd_sensitivity_bandwidth,
)

rng    = np.random.default_rng(99)
n      = 5000
cutoff = 0.0   # margin of 0 = bare majority

# running variable: Democratic vote margin (centred, in %)
margin = rng.normal(0, 15, n)
margin = np.clip(margin, -49, 49)

# fuzzy: incumbency take-up is 0.95 on winning side, 0 on losing side
# (small non-compliance: some winners vacate seat before next election)
above   = margin >= 0
prob_inc = np.where(above, 0.93, 0.0)
incumbent = rng.binomial(1, prob_inc, n).astype(float)

# true incumbency advantage: +15 pp in next election
noise    = rng.normal(0, 20, n)
win_next = (
    50 + 0.3 * margin + 15 * incumbent
    + noise
).clip(0, 100)

df = pd.DataFrame({
    "win_next": win_next, "margin": margin, "incumbent": incumbent,
})

# ── Manipulation test ──
manip = test_manipulation(df["margin"].values, cutoff, n_bins=40)
print("Manipulation test:", manip["interpretation"])

# ── Sharp RDD (treating instrument as treatment) ──
sharp = run_sharp_rdd(df["win_next"].values, df["margin"].values, cutoff)
print(f"\nSharp (reduced-form) τ = {sharp['tau']:.3f}  p = {sharp['p_value']:.4f}")

# ── Fuzzy RDD (LATE for compliers) ──
fuzzy = run_fuzzy_rdd(
    df["win_next"].values, df["margin"].values,
    df["incumbent"].values, cutoff
)
print(f"Fuzzy RDD LATE = {fuzzy['tau_fuzzy']:.3f}  SE = {fuzzy['se']:.3f}"
      f"  p = {fuzzy['p_value']:.4f}")
print(f"First-stage F = {fuzzy['first_stage_F']:.1f}  "
      f"(rule of thumb: >10 for strong instrument)")

# ── Bandwidth sensitivity plot ──
sens = rdd_sensitivity_bandwidth(df["win_next"].values, df["margin"].values, cutoff)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# RDD scatter
from rdd_design import plot_rdd as _plot_rdd
fig2 = plot_rdd(
    df["win_next"].values, df["margin"].values, cutoff,
    title="Incumbency Advantage (Sharp RDD)",
    ylabel="Vote share in next election (%)",
    xlabel="Vote margin in current election (%)",
)
fig2.savefig("rdd_incumbency.png", dpi=150, bbox_inches="tight")

# bandwidth sensitivity
axes[0].fill_between(sens["bandwidth"], sens["ci_lower"], sens["ci_upper"],
                     alpha=0.25, color="#2c7bb6", label="95% CI")
axes[0].plot(sens["bandwidth"], sens["tau"], "o-", color="#2c7bb6",
             linewidth=2, markersize=4, label="τ estimate")
axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[0].set_xlabel("Bandwidth (vote margin %)")
axes[0].set_ylabel("Estimated τ")
axes[0].set_title("Bandwidth sensitivity")
axes[0].legend()

# sample size by bandwidth
axes[1].plot(sens["bandwidth"], sens["n_left"],  "s--", color="#d7191c",
             label="N (control, left)")
axes[1].plot(sens["bandwidth"], sens["n_right"], "o--", color="#2c7bb6",
             label="N (treated, right)")
axes[1].set_xlabel("Bandwidth")
axes[1].set_ylabel("Observations in window")
axes[1].set_title("Sample size vs bandwidth")
axes[1].legend()

plt.tight_layout()
fig.savefig("rdd_sensitivity.png", dpi=150, bbox_inches="tight")
print("\nSaved rdd_incumbency.png and rdd_sensitivity.png")

# ── Summary ──
print(f"""
=== INCUMBENCY ADVANTAGE: SUMMARY ===
Sharp reduced-form τ  : {sharp['tau']:.2f} pp
Fuzzy LATE            : {fuzzy['tau_fuzzy']:.2f} pp
First-stage F         : {fuzzy['first_stage_F']:.1f}
Bandwidth used        : {sharp['bandwidth']:.2f} pp
Manipulation test     : {manip['interpretation']}
""")
```

---

## Validity Checks Checklist

| Check | Method | Pass criterion |
|---|---|---|
| No manipulation | McCrary density test | p ≥ 0.05 |
| Covariate balance | RDD on predetermined X | All p ≥ 0.10 |
| Bandwidth stability | τ across h ∈ [0.5h*, 2h*] | τ stable, does not cross 0 |
| Placebo cutoffs | Run RDD at c ± δ | No significant effects |
| Donut hole | Exclude observations near c | Estimate unchanged |
| Polynomial order | Compare p=1,2,3 | Consistent estimates |

---

## When RDD Is and Is Not Valid

**RDD is valid when:**
- The running variable is continuous at the threshold
- Agents cannot precisely manipulate their value just above/below c
- Only treatment status changes discontinuously at c (no other discontinuous policies)

**RDD is not valid when:**
- Bunching in the density at c (e.g., teachers round borderline exam scores)
- Multiple simultaneous treatment discontinuities at the same cutoff
- The bandwidth is so narrow that N is too small for reliable local estimates
- Extrapolation: the LATE applies only near the cutoff, not in the full population

---

## References

- Lee, D. S., & Lemieux, T. (2010). Regression discontinuity designs in economics.
  *Journal of Economic Literature*, 48(2), 281–355.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust nonparametric
  confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6).
- Imbens, G., & Kalyanaraman, K. (2012). Optimal bandwidth choice for the regression
  discontinuity estimator. *Review of Economic Studies*, 79(3), 933–959.
- McCrary, J. (2008). Manipulation of the running variable in the regression
  discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.
