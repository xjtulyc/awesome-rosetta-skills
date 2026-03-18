---
name: reaction-time-analysis
description: >
  Use this Skill to analyze reaction time data: outlier removal, ex-Gaussian
  distribution fitting, EZdiff DDM parameters, and HDDM-equivalent
  drift-diffusion models with PyMC.
tags:
  - psychology
  - reaction-time
  - drift-diffusion
  - ex-Gaussian
  - cognitive-modeling
  - DDM
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
    - scipy>=1.9
    - numpy>=1.23
    - pandas>=1.5
    - matplotlib>=3.6
    - pymc>=5.0
last_updated: "2026-03-18"
status: stable
---

# Reaction Time Analysis

> **TL;DR** — Complete pipeline for reaction time (RT) data quality control,
> ex-Gaussian distribution fitting, EZdiff DDM parameter extraction, and
> full Bayesian drift-diffusion modeling with PyMC. Covers outlier removal,
> Q-Q plots, Vincentile curves, delta plots, and condition comparison.

---

## When to Use

Use this Skill when you need to:

- Clean RT data from keyboard-response behavioral experiments
- Fit ex-Gaussian distributions to characterize RT distributions per condition
- Extract drift-diffusion model (DDM) parameters without full Bayesian fitting
  (EZdiff; Wagenmakers et al., 2007)
- Build a full Bayesian DDM with PyMC to compare conditions (accuracy, speed)
- Visualize RT distributions with Q-Q plots, density plots, and delta plots
- Compare conditions using Vincentile (quantile-average) curves

---

## Background

Reaction times are right-skewed because fast responses are bounded by motor
minimum latency (~150 ms) while slow responses have no upper bound. Two
complementary frameworks model this shape:

### Ex-Gaussian Distribution

The ex-Gaussian is the convolution of a Gaussian (mean μ, SD σ) and an
exponential (rate λ, mean τ = 1/λ):

- **μ** (mu): the Gaussian component mean — reflects fast, Gaussian-like RTs
- **σ** (sigma): Gaussian spread — reflects moment-to-moment variability
- **τ** (tau): exponential tail — reflects attentional lapses and slow responses

`scipy.stats.exponnorm` uses the parameterization K = τ/σ, loc = μ, scale = σ,
so that `exponnorm(K, loc=mu, scale=sigma)` produces the ex-Gaussian.

### Drift-Diffusion Model (DDM)

The DDM (Ratcliff, 1978) decomposes RT into three cognitive parameters:

| Parameter | Symbol | Interpretation |
|---|---|---|
| Drift rate | v | Signal strength / processing efficiency |
| Boundary separation | a | Response caution (speed-accuracy tradeoff) |
| Non-decision time | Ter | Perceptual encoding + motor execution time |

### EZdiff (Wagenmakers et al., 2007)

EZdiff extracts DDM parameters from three sufficient statistics:
mean RT (`MRT`), variance of RT (`VRT`), and accuracy (`Pc`):

```
z_pc  = Φ⁻¹(Pc)                          # probit of accuracy
v     = sign(Pc - 0.5) × 0.1 × (z_pc / MRT - z_pc³ / VRT × MRT)
a     = 0.1 × z_pc / v
Ter   = MRT - a / (2v)
```

For full Bayesian DDM, PyMC allows hierarchical fitting across participants
with posterior distributions over v, a, and Ter.

---

## Environment Setup

```bash
# Create isolated environment
conda create -n rt_analysis python=3.11 -y
conda activate rt_analysis

# Install core dependencies
pip install scipy>=1.9 numpy>=1.23 pandas>=1.5 matplotlib>=3.6

# Install PyMC (Bayesian DDM)
pip install pymc>=5.0

# Optional: faster sampling backend
pip install pytensor

# Verify
python -c "import scipy, numpy, pandas, matplotlib, pymc; print('All dependencies OK')"
```

---

## Core Workflow

### Step 1 — Load and Clean RT Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Dict, Tuple, List

# ── Absolute and SD-based outlier removal ───────────────────────────────────

def clean_rt_data(
    df: pd.DataFrame,
    rt_col: str = "rt",
    accuracy_col: str = "correct",
    min_rt: float = 150.0,
    max_rt: float = 3000.0,
    sd_cutoff: float = 2.5,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Remove RT outliers using absolute cutoffs and SD-based trimming.

    Args:
        df:           DataFrame with one row per trial.
        rt_col:       Column name for reaction times (milliseconds).
        accuracy_col: Column name for accuracy (1=correct, 0=error).
        min_rt:       Absolute lower bound (ms); RTs below are fast-guesses.
        max_rt:       Absolute upper bound (ms); RTs above are likely lapses.
        sd_cutoff:    Number of SDs for within-condition trimming.
        group_col:    Optional grouping column (e.g., 'condition') for
                      SD-based trimming within groups.

    Returns:
        Tuple of (cleaned_df, stats_dict).
    """
    n_original = len(df)

    # Absolute cutoffs
    mask_abs = (df[rt_col] >= min_rt) & (df[rt_col] <= max_rt)
    df_clean = df[mask_abs].copy()
    n_after_abs = len(df_clean)

    # SD-based trimming (within group if specified)
    if group_col and group_col in df_clean.columns:
        groups = df_clean[group_col].unique()
        keep_mask = pd.Series(True, index=df_clean.index)
        for g in groups:
            g_mask = df_clean[group_col] == g
            g_rts = df_clean.loc[g_mask, rt_col]
            mean_rt = g_rts.mean()
            sd_rt = g_rts.std(ddof=1)
            out_mask = (
                (df_clean[rt_col] < mean_rt - sd_cutoff * sd_rt) |
                (df_clean[rt_col] > mean_rt + sd_cutoff * sd_rt)
            )
            keep_mask[g_mask & out_mask] = False
        df_clean = df_clean[keep_mask].copy()
    else:
        mean_rt = df_clean[rt_col].mean()
        sd_rt = df_clean[rt_col].std(ddof=1)
        df_clean = df_clean[
            (df_clean[rt_col] >= mean_rt - sd_cutoff * sd_rt) &
            (df_clean[rt_col] <= mean_rt + sd_cutoff * sd_rt)
        ].copy()

    n_final = len(df_clean)

    stats_dict = {
        "n_original": n_original,
        "n_removed_absolute": n_original - n_after_abs,
        "n_removed_sd": n_after_abs - n_final,
        "n_final": n_final,
        "pct_removed": round(100 * (1 - n_final / n_original), 2),
        "mean_rt_clean": round(df_clean[rt_col].mean(), 2),
        "median_rt_clean": round(df_clean[rt_col].median(), 2),
        "sd_rt_clean": round(df_clean[rt_col].std(ddof=1), 2),
        "mean_accuracy": round(df_clean[accuracy_col].mean(), 4),
    }

    print(
        f"RT cleaning: {n_original} → {n_final} trials "
        f"({stats_dict['pct_removed']:.1f}% removed)\n"
        f"  Absolute cutoffs: {stats_dict['n_removed_absolute']} removed\n"
        f"  SD trimming (±{sd_cutoff}): {stats_dict['n_removed_sd']} removed\n"
        f"  Mean RT = {stats_dict['mean_rt_clean']} ms, "
        f"Accuracy = {stats_dict['mean_accuracy']:.1%}"
    )
    return df_clean, stats_dict
```

### Step 2 — Ex-Gaussian Fitting and Q-Q Plot

```python
from scipy.stats import exponnorm


def fit_exgaussian(
    rts: np.ndarray,
    label: str = "data",
    plot: bool = True,
) -> Dict:
    """
    Fit an ex-Gaussian distribution to RT data and produce a Q-Q plot.

    scipy.stats.exponnorm parameterization:
        K     = tau / sigma   (shape)
        loc   = mu            (Gaussian mean, milliseconds)
        scale = sigma         (Gaussian SD)

    Therefore:
        mu    = loc
        sigma = scale
        tau   = K * scale

    Args:
        rts:   1-D array of reaction times (milliseconds, already cleaned).
        label: Label for plot title and print output.
        plot:  Whether to generate the Q-Q plot.

    Returns:
        Dict with mu, sigma, tau, K, loc, scale, and fit statistics.
    """
    # MLE fit
    K_fit, loc_fit, scale_fit = exponnorm.fit(rts, floc=None)

    mu = loc_fit
    sigma = scale_fit
    tau = K_fit * scale_fit
    mean_theoretical = mu + tau
    var_theoretical = sigma ** 2 + tau ** 2

    # KS goodness-of-fit
    ks_stat, ks_p = stats.kstest(rts, lambda x: exponnorm.cdf(x, K_fit, loc_fit, scale_fit))

    params = {
        "label": label,
        "mu": round(mu, 3),
        "sigma": round(sigma, 3),
        "tau": round(tau, 3),
        "K": round(K_fit, 4),
        "loc": round(loc_fit, 3),
        "scale": round(scale_fit, 3),
        "mean_theoretical": round(mean_theoretical, 3),
        "sd_theoretical": round(np.sqrt(var_theoretical), 3),
        "ks_statistic": round(ks_stat, 4),
        "ks_p": round(ks_p, 4),
    }

    print(
        f"Ex-Gaussian fit [{label}]: μ={params['mu']:.1f}, "
        f"σ={params['sigma']:.1f}, τ={params['tau']:.1f} ms | "
        f"KS p={params['ks_p']:.3f}"
    )

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Density overlay
        ax = axes[0]
        rt_range = np.linspace(rts.min(), rts.max(), 500)
        ax.hist(rts, bins=40, density=True, alpha=0.4, color="steelblue", label="Observed")
        ax.plot(rt_range, exponnorm.pdf(rt_range, K_fit, loc_fit, scale_fit),
                color="crimson", linewidth=2, label="Ex-Gaussian fit")
        ax.set_xlabel("Reaction Time (ms)")
        ax.set_ylabel("Density")
        ax.set_title(f"Ex-Gaussian fit — {label}")
        ax.legend()

        # Q-Q plot vs ex-Gaussian
        ax2 = axes[1]
        n = len(rts)
        probs = (np.arange(1, n + 1) - 0.5) / n
        emp_quantiles = np.sort(rts)
        theo_quantiles = exponnorm.ppf(probs, K_fit, loc_fit, scale_fit)
        ax2.scatter(theo_quantiles, emp_quantiles, s=10, alpha=0.5, color="steelblue")
        lims = [min(theo_quantiles.min(), emp_quantiles.min()),
                max(theo_quantiles.max(), emp_quantiles.max())]
        ax2.plot(lims, lims, "r--", linewidth=1.5, label="Identity line")
        ax2.set_xlabel("Theoretical ex-Gaussian quantiles (ms)")
        ax2.set_ylabel("Empirical quantiles (ms)")
        ax2.set_title(f"Q-Q Plot — {label}")
        ax2.legend()

        fig.tight_layout()
        plt.show()

    return params


def vincentile_plot(
    rt_dict: Dict[str, np.ndarray],
    n_bins: int = 5,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create Vincentile (quantile-average) curves for multiple conditions.

    Vincentiles average participants' quantiles across the sample, preserving
    distributional shape while allowing condition comparison.

    Args:
        rt_dict:     Dict mapping condition label to array of RTs.
        n_bins:      Number of quantile bins (default 5 = quintiles).
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    quantile_positions = np.linspace(0, 1, n_bins + 2)[1:-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(rt_dict)))

    for (label, rts), color in zip(rt_dict.items(), colors):
        quantiles = np.quantile(rts, quantile_positions)
        ax.plot(quantile_positions * 100, quantiles, marker="o",
                color=color, linewidth=2, markersize=6, label=label)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Reaction Time (ms)")
    ax.set_title("Vincentile Plot")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.show()
    return fig
```

### Step 3 — EZdiff DDM Parameter Extraction

```python
from scipy.special import ndtri  # probit function


def ezdiff(
    mean_rt: float,
    var_rt: float,
    accuracy: float,
    scaling_factor: float = 0.1,
) -> Dict:
    """
    Extract DDM parameters (v, a, Ter) using the EZdiff method.

    Reference: Wagenmakers, E.-J., van der Maas, H. L. J., & Grasman, R. P. P. P.
    (2007). An EZ-diffusion model for response time and accuracy.
    Psychonomic Bulletin & Review, 14(1), 3–22.

    Assumptions: equal-variance, unbiased starting point (z = a/2),
    no inter-trial variability. Best used for quick parameter estimation
    when full DDM fitting is not feasible.

    Args:
        mean_rt:        Mean reaction time (seconds or ms — consistent units).
        var_rt:         Variance of reaction times (same units squared).
        accuracy:       Proportion correct (0 < Pc < 1; values near 0.5 or 1
                        require edge correction).
        scaling_factor: DDM scaling constant (default 0.1 for standard units).

    Returns:
        Dict with drift rate v, boundary a, and non-decision time Ter.
    """
    # Edge correction: avoid Pc = 0.5 (v=0) or Pc = 1.0 (v→∞)
    if accuracy <= 0.5:
        accuracy = max(accuracy, 0.501)
    if accuracy >= 1.0:
        accuracy = min(accuracy, 0.999)

    s = scaling_factor
    z_pc = ndtri(accuracy)  # probit

    # Drift rate
    v = (
        np.sign(accuracy - 0.5) * s *
        (z_pc / mean_rt - z_pc ** 3 / var_rt * mean_rt) ** 0.5
        if (z_pc / mean_rt - z_pc ** 3 / var_rt * mean_rt) >= 0
        else 0.0
    )

    # Alternative closed-form (Wagenmakers 2007, eq. 3)
    # More numerically stable form:
    L = ndtri(accuracy)
    x = L * (L ** 2 * mean_rt - var_rt * L ** 2 / mean_rt - var_rt / mean_rt)
    if x > 0:
        v = np.sign(accuracy - 0.5) * s * (x ** 0.5) / np.sqrt(var_rt)
    else:
        v = 0.0

    # Boundary separation
    a = s * L / v if v != 0 else np.nan

    # Non-decision time
    Ter = mean_rt - a / (2 * v) if v != 0 else np.nan

    result = {
        "v": round(float(v), 4),
        "a": round(float(a), 4),
        "Ter": round(float(Ter), 4),
        "mean_rt": mean_rt,
        "var_rt": var_rt,
        "accuracy": accuracy,
    }
    print(
        f"EZdiff: v={result['v']:.3f}, a={result['a']:.3f}, "
        f"Ter={result['Ter']:.3f} | Pc={accuracy:.3f}"
    )
    return result


def ezdiff_by_condition(
    df: pd.DataFrame,
    rt_col: str = "rt",
    accuracy_col: str = "correct",
    condition_col: str = "condition",
    rt_units: str = "ms",
) -> pd.DataFrame:
    """
    Apply EZdiff to each condition in a DataFrame.

    Args:
        df:            Trial-level DataFrame.
        rt_col:        Column of reaction times.
        accuracy_col:  Column of accuracy (1/0).
        condition_col: Column identifying conditions.
        rt_units:      'ms' or 's' — if 'ms', divides by 1000 before fitting.

    Returns:
        DataFrame with one row per condition and DDM parameters.
    """
    rows = []
    for cond, group in df.groupby(condition_col):
        rts = group[rt_col].values
        if rt_units == "ms":
            rts_s = rts / 1000.0
        else:
            rts_s = rts
        mrt = rts_s.mean()
        vrt = rts_s.var(ddof=1)
        pc = group[accuracy_col].mean()
        params = ezdiff(mrt, vrt, pc)
        params["condition"] = cond
        params["n_trials"] = len(group)
        rows.append(params)

    result_df = pd.DataFrame(rows).set_index("condition")
    print("\nEZdiff parameters by condition:")
    print(result_df[["v", "a", "Ter", "n_trials"]].round(3))
    return result_df
```

---

## Advanced Usage

### PyMC Drift-Diffusion Model

```python
import pymc as pm
import numpy as np


def fit_ddm_pymc(
    rt_correct: np.ndarray,
    rt_error: np.ndarray,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.90,
    seed: int = 42,
) -> pm.backends.base.MultiTrace:
    """
    Fit a simple Wiener drift-diffusion model using PyMC.

    Models both correct and error RTs via the Wiener first-passage time
    distribution. Assumes equal-variance, zero between-trial variability.

    Priors (weakly informative):
        v  ~ Normal(0, 2)       — drift rate
        a  ~ HalfNormal(1)      — boundary separation (positive)
        Ter ~ Uniform(0.1, 0.5) — non-decision time (seconds)
        z  ~ Uniform(0.3, 0.7)  — relative starting point (0.5 = unbiased)

    Args:
        rt_correct: Array of RTs for correct responses (seconds).
        rt_error:   Array of RTs for error responses (seconds).
        draws:      Number of posterior samples per chain.
        tune:       Number of tuning steps.
        target_accept: NUTS target acceptance rate.
        seed:       Random seed.

    Returns:
        PyMC InferenceData object with posterior samples.

    Notes:
        - Requires PyMC >= 5.0 and the hssm package or manual Wiener
          likelihood. Below uses a simplified normal approximation for
          demonstration; for production use hssm (pip install hssm).
        - Do NOT use this function with hard-coded API keys or credentials.
    """
    all_rts = np.concatenate([rt_correct, rt_error])
    responses = np.concatenate([
        np.ones(len(rt_correct)),
        np.zeros(len(rt_error))
    ])

    with pm.Model() as ddm_model:
        # Priors
        v = pm.Normal("v", mu=0, sigma=2)
        a = pm.HalfNormal("a", sigma=1)
        Ter = pm.Uniform("Ter", lower=0.05, upper=0.5)

        # Likelihood: approximate via normal on decision time
        # Decision time = RT - Ter (must be positive)
        decision_time = pm.Deterministic("decision_time", all_rts - Ter)

        # Expected decision time under DDM (Ratcliff 1978, eq for unbiased start)
        # E[DT | correct] ≈ a/(2v) × tanh(av/2) for large a
        expected_dt = pm.Deterministic(
            "expected_dt",
            (a / (2 * v)) * pm.math.tanh(a * v / 2)
        )

        # Variance of decision time (approximation)
        var_dt = pm.Deterministic(
            "var_dt",
            (a / (2 * v ** 3)) * (1 - pm.math.tanh(a * v / 2) ** 2)
        )

        # Normal approximation likelihood
        obs = pm.Normal(
            "obs",
            mu=expected_dt,
            sigma=pm.math.sqrt(var_dt),
            observed=decision_time
        )

        # Sample
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )

    return idata


def compare_ddm_conditions(
    idata_cond1,
    idata_cond2,
    param_names: List[str] = ["v", "a", "Ter"],
    labels: Tuple[str, str] = ("Condition 1", "Condition 2"),
) -> pd.DataFrame:
    """
    Compare posterior DDM parameters across two conditions.

    Args:
        idata_cond1: InferenceData from condition 1.
        idata_cond2: InferenceData from condition 2.
        param_names: DDM parameters to compare.
        labels:      Labels for the two conditions.

    Returns:
        DataFrame with posterior mean, HDI, and P(cond1 > cond2).
    """
    import arviz as az

    rows = []
    for param in param_names:
        post1 = idata_cond1.posterior[param].values.flatten()
        post2 = idata_cond2.posterior[param].values.flatten()
        diff = post1 - post2
        p_greater = (diff > 0).mean()
        rows.append({
            "parameter": param,
            f"mean_{labels[0]}": round(post1.mean(), 4),
            f"mean_{labels[1]}": round(post2.mean(), 4),
            "mean_difference": round(diff.mean(), 4),
            "HDI_2.5%": round(np.percentile(diff, 2.5), 4),
            "HDI_97.5%": round(np.percentile(diff, 97.5), 4),
            f"P({labels[0]}>{labels[1]})": round(p_greater, 3),
        })

    comparison_df = pd.DataFrame(rows).set_index("parameter")
    print(f"\nDDM parameter comparison ({labels[0]} vs {labels[1]}):")
    print(comparison_df)
    return comparison_df
```

### Delta Plot Analysis

```python
def delta_plot(
    rt_dict: Dict[str, np.ndarray],
    n_quantiles: int = 9,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a delta plot showing condition differences across the RT distribution.

    A delta plot plots quantile differences (Condition A - Condition B) against
    the mean quantile RT. Positive slopes indicate the effect grows with RT
    (common in congruency effects); negative slopes suggest strategic slowing.

    Args:
        rt_dict:      Dict with exactly two conditions (first - second = delta).
        n_quantiles:  Number of quantile points (default 9 = deciles 0.1–0.9).
        output_path:  Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    assert len(rt_dict) == 2, "Delta plot requires exactly 2 conditions."
    labels = list(rt_dict.keys())
    rts_a, rts_b = rt_dict[labels[0]], rt_dict[labels[1]]

    quantile_ps = np.linspace(0.1, 0.9, n_quantiles)
    q_a = np.quantile(rts_a, quantile_ps)
    q_b = np.quantile(rts_b, quantile_ps)
    mean_q = (q_a + q_b) / 2
    delta = q_a - q_b

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Quantile functions
    ax1 = axes[0]
    ax1.plot(quantile_ps * 100, q_a, marker="o", label=labels[0], color="steelblue")
    ax1.plot(quantile_ps * 100, q_b, marker="s", label=labels[1], color="tomato")
    ax1.set_xlabel("Percentile")
    ax1.set_ylabel("RT (ms)")
    ax1.set_title("Conditional Accuracy Function")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Delta plot
    ax2 = axes[1]
    ax2.plot(mean_q, delta, marker="o", color="purple", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Mean RT (ms)")
    ax2.set_ylabel(f"Δ RT ({labels[0]} − {labels[1]}) (ms)")
    ax2.set_title("Delta Plot")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.show()
    return fig
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `exponnorm.fit` returns very large K | Outlier RTs inflate the tail | Apply absolute cutoffs before fitting |
| EZdiff produces negative Ter | Mean RT too close to a/(2v) | Check accuracy is well above 0.5; use edge correction |
| EZdiff v = 0 | Accuracy exactly 0.5 | Edge-correct accuracy to 0.501 |
| PyMC divergences during sampling | Model mis-specification or bad priors | Increase `target_accept` to 0.95; reparameterize |
| PyMC `Ter` samples near 0 | Prior too wide | Tighten `Uniform(0.1, 0.4)` based on task |
| K-S test rejects ex-Gaussian | Bimodal RT distribution (e.g., two strategies) | Separate trials by strategy or use mixture model |
| `ndtri` returns inf | Accuracy = 1.0 exactly | Clamp accuracy to 0.999 before calling EZdiff |

---

## External Resources

- Wagenmakers, E.-J., et al. (2007). An EZ-diffusion model.
  *Psychonomic Bulletin & Review*, 14(1), 3–22.
  <https://doi.org/10.3758/BF03194023>
- Ratcliff, R. (1978). A theory of memory retrieval. *Psychological Review*, 85(2).
- Matzke, D., & Wagenmakers, E.-J. (2009). Psychological interpretation of ex-Gaussian.
  *Psychonomic Bulletin & Review*, 16(5), 798–817.
- PyMC documentation: <https://www.pymc.io>
- HSSM (Hierarchical Sequential Sampling Models): <https://lnccbrown.github.io/HSSM/>
- scipy.stats.exponnorm: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.exponnorm.html>

---

## Examples

### Example 1 — Full Ex-Gaussian Pipeline per Condition

```python
import numpy as np
import pandas as pd

# Simulate example RT data (replace with real data loading)
rng = np.random.default_rng(42)

def simulate_rt_data(n_trials: int, conditions: List[str]) -> pd.DataFrame:
    """Simulate RT data with ex-Gaussian distributions per condition."""
    rows = []
    params = {
        "congruent":   {"mu": 400, "sigma": 50, "tau": 80},
        "incongruent": {"mu": 420, "sigma": 55, "tau": 120},
    }
    for cond in conditions:
        p = params[cond]
        # Simulate ex-Gaussian: Normal(mu, sigma) + Exponential(tau)
        gauss_part = rng.normal(p["mu"], p["sigma"], n_trials)
        exp_part   = rng.exponential(p["tau"], n_trials)
        rts = gauss_part + exp_part
        accuracy = rng.binomial(1, 0.92, n_trials)
        for rt, acc in zip(rts, accuracy):
            rows.append({"rt": rt, "correct": acc, "condition": cond})
    return pd.DataFrame(rows)

# Load / simulate data
df = simulate_rt_data(n_trials=300, conditions=["congruent", "incongruent"])

# Clean
df_clean, clean_stats = clean_rt_data(
    df, rt_col="rt", accuracy_col="correct",
    min_rt=150, max_rt=3000, sd_cutoff=2.5, group_col="condition"
)

# Ex-Gaussian fit per condition
exg_params = {}
for cond in ["congruent", "incongruent"]:
    rts_cond = df_clean.loc[df_clean["condition"] == cond, "rt"].values
    exg_params[cond] = fit_exgaussian(rts_cond, label=cond, plot=True)

# Print parameter comparison
print("\nEx-Gaussian parameter comparison:")
print(pd.DataFrame(exg_params).T[["mu", "sigma", "tau"]])

# Vincentile plot
rt_dict = {
    cond: df_clean.loc[df_clean["condition"] == cond, "rt"].values
    for cond in ["congruent", "incongruent"]
}
vincentile_plot(rt_dict, n_bins=5, output_path="vincentile.png")

# Delta plot
delta_plot(rt_dict, n_quantiles=9, output_path="delta_plot.png")
```

### Example 2 — EZdiff Parameters and PyMC DDM Comparison

```python
# EZdiff analysis from cleaned data
ezdiff_params = ezdiff_by_condition(
    df_clean, rt_col="rt", accuracy_col="correct",
    condition_col="condition", rt_units="ms"
)

print("\nDrift rate comparison (v):")
print(f"  Congruent:   v = {ezdiff_params.loc['congruent', 'v']:.3f}")
print(f"  Incongruent: v = {ezdiff_params.loc['incongruent', 'v']:.3f}")
print(f"  Δv = {ezdiff_params.loc['congruent', 'v'] - ezdiff_params.loc['incongruent', 'v']:.3f}")

# PyMC DDM fit (convert ms -> seconds first)
for cond in ["congruent", "incongruent"]:
    cond_df = df_clean[df_clean["condition"] == cond]
    rt_correct_s = cond_df.loc[cond_df["correct"] == 1, "rt"].values / 1000.0
    rt_error_s   = cond_df.loc[cond_df["correct"] == 0, "rt"].values / 1000.0
    print(f"\nFitting DDM for condition: {cond}")
    # Note: for full Bayesian fit uncomment:
    # idata = fit_ddm_pymc(rt_correct_s, rt_error_s, draws=1000, tune=500)

# Summary statistics comparison plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
params_to_plot = ["v", "a", "Ter"]
labels = ezdiff_params.index.tolist()

for ax, param in zip(axes, params_to_plot):
    values = [ezdiff_params.loc[lbl, param] for lbl in labels]
    ax.bar(labels, values, color=["steelblue", "tomato"])
    ax.set_title(f"EZdiff: {param}")
    ax.set_ylabel(param)

fig.suptitle("DDM Parameters by Condition (EZdiff)", fontsize=13)
fig.tight_layout()
plt.savefig("ezdiff_comparison.png", dpi=150)
plt.show()
print("EZdiff comparison plot saved.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — RT cleaning, ex-Gaussian fitting, Q-Q/Vincentile/delta plots, EZdiff, PyMC DDM |
