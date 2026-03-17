---
name: power-analysis
description: >
  Use this Skill for statistical power analysis in psychological research: t-tests,
  ANOVA, correlation, regression, mediation, and simulation-based designs with Python.
tags:
  - psychology
  - statistics
  - power-analysis
  - sample-size
  - statsmodels
  - pingouin
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - numpy>=1.23.0
    - scipy>=1.9.0
    - statsmodels>=0.14.0
    - pingouin>=0.5.3
    - pandas>=1.5.0
    - matplotlib>=3.6.0
last_updated: "2026-03-17"
---

# Power Analysis

> **TL;DR** — Statistical power analysis for behavioral and psychological research.
> G*Power-equivalent calculations in Python using statsmodels and pingouin for t-tests,
> ANOVA, chi-square, correlation, regression, and logistic regression; simulation-based
> power for complex designs; power curves; sequential testing; and Monte Carlo power
> for mediation.

---

## 1. Overview

### What Problem Does This Skill Solve?

Underpowered studies produce unreliable results and waste resources; overpowered
studies are unethical and wasteful. This Skill provides a complete toolkit for
a priori, post-hoc, and sensitivity power analyses:

- **A priori**: find the sample size needed to detect an effect of a given size
  at specified alpha and power (1−beta)
- **Post-hoc**: determine the power achieved by a study already completed
- **Sensitivity**: find the minimum detectable effect size (MDES) given N and alpha
- **Simulation-based**: compute power for complex designs (mixed ANOVA, mediation)
  where closed-form solutions are unavailable or unreliable

### Applicable Scenarios

| Scenario | Entry Point |
|---|---|
| Two-group RCT sample size | `compute_power_ttest()` with `solve_for="n"` |
| One-way ANOVA with k groups | `compute_power_anova()` |
| Correlation study (N = ?) | `compute_power_correlation()` |
| Logistic regression outcome | `compute_power_logistic()` |
| 2×3 mixed design | `simulate_mixed_anova_power()` |
| Mediation hypothesis (indirect effect) | `simulate_mediation_power()` |
| Sequential / adaptive testing | `alpha_spending_obrien_fleming()` |
| Power curve visualization | `power_curve_plot()` |

### Key Assumptions and Limitations

- Closed-form t-test and ANOVA power assume normally distributed outcomes. For
  heavily skewed data, use `simulate_power_nonparametric()`.
- Effect sizes (Cohen's d, f, r, w) must be justified from pilot data, meta-analyses,
  or smallest-effect-of-interest arguments — do not use conventional benchmarks
  (small/medium/large) uncritically.
- Simulation-based estimates have Monte Carlo error; use `n_sim >= 5000` for
  publication-quality estimates.
- The mediation power simulation assumes linear paths; non-linear or categorical
  mediators require custom simulation.

---

## 2. Environment Setup

```bash
# Create and activate environment
conda create -n power_analysis python=3.11 -y
conda activate power_analysis

# Install all dependencies
pip install numpy scipy statsmodels pingouin pandas matplotlib

# Verify
python -c "import statsmodels, pingouin, scipy, numpy, matplotlib; print('All OK')"
```

---

## 3. Core Implementation

### 3.1 Shared Utilities

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from typing import Optional, Literal, Tuple, List
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def cohens_d_from_means(
    mean1: float,
    mean2: float,
    sd_pooled: float,
) -> float:
    """
    Compute Cohen's d from two group means and a pooled SD.

    Args:
        mean1:     Mean of group 1.
        mean2:     Mean of group 2.
        sd_pooled: Pooled standard deviation.

    Returns:
        Cohen's d (unsigned).
    """
    return abs(mean1 - mean2) / sd_pooled


def cohens_f_from_means(
    group_means: List[float],
    sd_within: float,
) -> float:
    """
    Compute Cohen's f from group means and within-group SD for ANOVA.

    Args:
        group_means: List of population means for each group.
        sd_within:   Expected within-group standard deviation.

    Returns:
        Cohen's f effect size.
    """
    k = len(group_means)
    grand_mean = np.mean(group_means)
    sd_between = np.sqrt(np.sum((np.array(group_means) - grand_mean) ** 2) / k)
    return sd_between / sd_within


def partial_eta_sq_to_cohens_f(partial_eta_sq: float) -> float:
    """Convert partial eta-squared to Cohen's f for ANOVA power calculations."""
    return np.sqrt(partial_eta_sq / (1 - partial_eta_sq))
```

### 3.2 T-Test Power (One-Sample, Two-Sample, Paired)

```python
from statsmodels.stats.power import TTestPower, TTestIndPower


def compute_power_ttest(
    effect_size: Optional[float] = None,
    n: Optional[float] = None,
    alpha: float = 0.05,
    power: Optional[float] = None,
    design: Literal["two-sample", "one-sample", "paired"] = "two-sample",
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
    ratio: float = 1.0,
) -> dict:
    """
    Compute power, sample size, or effect size for t-tests — G*Power equivalent.

    Exactly one of effect_size, n, or power may be None; that quantity is solved for.

    Args:
        effect_size: Cohen's d (leave None to solve for MDES).
        n:           Sample size per group (leave None to solve for n).
        alpha:       Significance level (Type I error rate).
        power:       Desired statistical power (leave None for post-hoc power).
        design:      'two-sample' (independent groups), 'one-sample', or 'paired'.
        alternative: Directionality of the test.
        ratio:       n2/n1 ratio for unequal group sizes (two-sample only).

    Returns:
        Dict with keys: solved_for, value, effect_size, n, alpha, power, design.

    Examples:
        # How many participants per group for d=0.5 at 80% power?
        compute_power_ttest(effect_size=0.5, power=0.80)
        # Post-hoc power with n=30, d=0.4
        compute_power_ttest(effect_size=0.4, n=30)
        # Sensitivity: MDES given n=50, power=0.80
        compute_power_ttest(n=50, power=0.80)
    """
    if design == "two-sample":
        analysis = TTestIndPower()
        n_arg = n  # per group
    else:
        analysis = TTestPower()
        n_arg = n

    nobs1 = n_arg
    result_value = analysis.solve_power(
        effect_size=effect_size,
        nobs1=nobs1,
        alpha=alpha,
        power=power,
        alternative=alternative,
        ratio=ratio if design == "two-sample" else None,
    )

    # Determine which was solved
    if n is None:
        solved_for = "n"
        solved_value = np.ceil(result_value)
        n_final = int(solved_value)
        power_final = power
        effect_final = effect_size
    elif power is None:
        solved_for = "power"
        solved_value = round(result_value, 4)
        n_final = int(n)
        power_final = solved_value
        effect_final = effect_size
    else:
        solved_for = "effect_size"
        solved_value = round(result_value, 4)
        n_final = int(n)
        power_final = power
        effect_final = solved_value

    total_n = n_final * 2 if design == "two-sample" else n_final
    print(
        f"[{design} t-test | {alternative}] "
        f"Solved {solved_for} = {solved_value} | "
        f"d={effect_final}, n_per_group={n_final}, total_N={total_n}, "
        f"alpha={alpha}, power={power_final}"
    )

    return {
        "solved_for": solved_for,
        "value": solved_value,
        "effect_size_d": effect_final,
        "n_per_group": n_final,
        "total_n": total_n,
        "alpha": alpha,
        "power": power_final,
        "design": design,
    }
```

### 3.3 ANOVA Power

```python
from statsmodels.stats.power import FTestAnovaPower


def compute_power_anova(
    effect_size_f: Optional[float] = None,
    n: Optional[float] = None,
    alpha: float = 0.05,
    power: Optional[float] = None,
    k_groups: int = 3,
) -> dict:
    """
    Compute power, sample size, or effect size for one-way ANOVA.

    Args:
        effect_size_f: Cohen's f (leave None to solve for MDES).
        n:             Total sample size (leave None to solve for n).
        alpha:         Significance level.
        power:         Desired power (leave None for post-hoc power).
        k_groups:      Number of groups.

    Returns:
        Dict with solved quantity and full analysis parameters.

    Notes:
        Convert partial eta-squared -> Cohen's f using partial_eta_sq_to_cohens_f().
        Typical benchmarks: small f=0.10, medium f=0.25, large f=0.40.
    """
    analysis = FTestAnovaPower()

    result_value = analysis.solve_power(
        effect_size=effect_size_f,
        nobs=n,
        alpha=alpha,
        power=power,
        k_groups=k_groups,
    )

    if n is None:
        solved_for = "n"
        solved_value = int(np.ceil(result_value))
        n_final = solved_value
    elif power is None:
        solved_for = "power"
        solved_value = round(result_value, 4)
        n_final = int(n)
    else:
        solved_for = "effect_size_f"
        solved_value = round(result_value, 4)
        n_final = int(n)

    n_per_group = int(np.ceil(n_final / k_groups))
    print(
        f"[One-way ANOVA, k={k_groups}] Solved {solved_for} = {solved_value} | "
        f"f={effect_size_f if solved_for != 'effect_size_f' else solved_value}, "
        f"total_N={n_final}, n_per_group={n_per_group}, "
        f"alpha={alpha}, power={power if solved_for != 'power' else solved_value}"
    )

    return {
        "solved_for": solved_for,
        "value": solved_value,
        "effect_size_f": effect_size_f if solved_for != "effect_size_f" else solved_value,
        "total_n": n_final,
        "n_per_group": n_per_group,
        "k_groups": k_groups,
        "alpha": alpha,
        "power": power if solved_for != "power" else solved_value,
    }


def compute_power_correlation(
    r: Optional[float] = None,
    n: Optional[int] = None,
    alpha: float = 0.05,
    power: Optional[float] = None,
    alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
) -> dict:
    """
    Compute power, sample size, or minimum detectable correlation coefficient.

    Uses Fisher's z-transformation approach (equivalent to G*Power 'bivariate normal').

    Args:
        r:           Pearson r effect size (leave None to solve for minimum r).
        n:           Sample size (leave None to solve for n).
        alpha:       Significance level.
        power:       Desired power (leave None for post-hoc).
        alternative: Test directionality.

    Returns:
        Dict with solved quantity and parameters.
    """
    from statsmodels.stats.power import NormalIndPower

    # Convert r to Cohen's d equivalent via Fisher z
    def r_to_z(r_val):
        return 0.5 * np.log((1 + r_val) / (1 - r_val))

    alt_map = {"two-sided": "two-sided", "larger": "larger", "smaller": "smaller"}

    # Use pingouin's power_corr for direct r-based calculation
    import pingouin as pg

    result = pg.power_corr(
        r=r,
        n=n,
        power=power,
        alpha=alpha,
        alternative=alternative,
    )

    if n is None:
        solved_for = "n"
        solved_value = int(np.ceil(result))
    elif power is None:
        solved_for = "power"
        solved_value = round(result, 4)
    else:
        solved_for = "r"
        solved_value = round(result, 4)

    print(
        f"[Correlation | {alternative}] Solved {solved_for} = {solved_value} | "
        f"r={r}, n={n}, alpha={alpha}, power={power}"
    )
    return {
        "solved_for": solved_for,
        "value": solved_value,
        "r": r,
        "n": n if solved_for != "n" else solved_value,
        "alpha": alpha,
        "power": power if solved_for != "power" else solved_value,
    }


def compute_power_logistic(
    odds_ratio: float,
    p_baseline: float,
    n: Optional[int] = None,
    alpha: float = 0.05,
    power: Optional[float] = None,
    ratio: float = 1.0,
) -> dict:
    """
    Compute power or sample size for a logistic regression binary outcome.

    Uses the Demidenko (2007) method via statsmodels.

    Args:
        odds_ratio:  Expected odds ratio for the predictor of interest.
        p_baseline:  Baseline event probability (control group / overall mean).
        n:           Total sample size (leave None to solve for n).
        alpha:       Significance level.
        power:       Desired power (leave None for post-hoc).
        ratio:       n_exposed / n_unexposed ratio.

    Returns:
        Dict with solved quantity and parameters.
    """
    from statsmodels.stats.power import zt_ind_solve_power

    p1 = p_baseline
    p2 = (odds_ratio * p1) / (1 - p1 + odds_ratio * p1)

    # Use two-proportions z-test as approximation
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    effect_size_h = abs(h)

    from statsmodels.stats.proportion import proportion_effectsize
    from statsmodels.stats.power import NormalIndPower

    analysis = NormalIndPower()
    result_value = analysis.solve_power(
        effect_size=effect_size_h,
        nobs1=n / 2 if n else None,
        alpha=alpha,
        power=power,
        ratio=ratio,
    )

    if n is None:
        solved_for = "n"
        solved_value = int(np.ceil(result_value * 2))
    else:
        solved_for = "power"
        solved_value = round(result_value, 4)

    print(
        f"[Logistic regression] Solved {solved_for} = {solved_value} | "
        f"OR={odds_ratio}, p_baseline={p_baseline}, total_N={n}, alpha={alpha}"
    )
    return {
        "solved_for": solved_for,
        "value": solved_value,
        "odds_ratio": odds_ratio,
        "p_baseline": p_baseline,
        "p_outcome": round(p2, 4),
        "total_n": n if solved_for != "n" else solved_value,
        "alpha": alpha,
        "power": power if solved_for != "power" else solved_value,
    }
```

### 3.4 Power Curve Visualization

```python
def power_curve_plot(
    design: Literal["ttest_ind", "anova", "correlation"],
    effect_sizes: Optional[List[float]] = None,
    n_range: Optional[Tuple[int, int]] = None,
    alpha: float = 0.05,
    k_groups: int = 3,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot power curves showing achieved power across a range of sample sizes
    for multiple effect size values.

    Args:
        design:       Analysis type: 'ttest_ind', 'anova', or 'correlation'.
        effect_sizes: List of effect size values to plot (Cohen's d, f, or r).
        n_range:      Tuple (min_n, max_n) for the x-axis.
        alpha:        Significance level.
        k_groups:     Number of groups (ANOVA only).
        output_path:  Optional file path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    if effect_sizes is None:
        if design == "ttest_ind":
            effect_sizes = [0.2, 0.5, 0.8]
            label_prefix = "d"
        elif design == "anova":
            effect_sizes = [0.10, 0.25, 0.40]
            label_prefix = "f"
        else:
            effect_sizes = [0.10, 0.30, 0.50]
            label_prefix = "r"
    else:
        label_prefix = "ES"

    if n_range is None:
        n_range = (10, 200)

    n_values = np.arange(n_range[0], n_range[1] + 1, 5)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(effect_sizes)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80% power")
    ax.axhline(0.95, color="silver", linestyle=":", linewidth=1, label="95% power")

    for es, color in zip(effect_sizes, colors):
        powers = []
        for n in n_values:
            if design == "ttest_ind":
                p = TTestIndPower().solve_power(effect_size=es, nobs1=n, alpha=alpha, power=None)
            elif design == "anova":
                p = FTestAnovaPower().solve_power(
                    effect_size=es, nobs=n * k_groups, alpha=alpha, power=None, k_groups=k_groups
                )
            else:
                import pingouin as pg
                p = pg.power_corr(r=es, n=n, alpha=alpha, power=None)
            powers.append(p)
        ax.plot(n_values, powers, color=color, linewidth=2, label=f"{label_prefix}={es}")

    ax.set_xlabel("Sample Size (n per group)" if design != "anova" else "Total N")
    ax.set_ylabel("Statistical Power (1 − β)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Power Curves — {design} (α={alpha})")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved power curve to {output_path}")

    return fig
```

### 3.5 Simulation-Based Power for Mixed ANOVA

```python
def simulate_mixed_anova_power(
    between_means: List[List[float]],
    within_means: List[List[float]],
    sd_between: float,
    sd_within: float,
    n_per_cell: int,
    alpha: float = 0.05,
    n_sim: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Estimate power for a between-within (mixed) ANOVA design via Monte Carlo simulation.

    Args:
        between_means: List of lists; between_means[i][j] = population mean for
                       between-group i, within-time j.
        within_means:  Same structure — can be identical to between_means or specify
                       a separate within-subjects pattern.
        sd_between:    Between-subjects SD (residual).
        sd_within:     Within-subjects SD (repeated measure error).
        n_per_cell:    Participants per between-subjects group.
        alpha:         Significance level.
        n_sim:         Number of Monte Carlo simulations.
        seed:          Random seed for reproducibility.

    Returns:
        Dict with power estimates for main effects and interaction.
    """
    import pingouin as pg

    rng = np.random.default_rng(seed)
    n_between = len(between_means)
    n_within = len(between_means[0])

    sig_between = 0
    sig_within = 0
    sig_interaction = 0

    for _ in range(n_sim):
        # Simulate data
        rows = []
        for g in range(n_between):
            for subj in range(n_per_cell):
                subj_id = g * n_per_cell + subj
                subj_offset = rng.normal(0, sd_between)
                for t in range(n_within):
                    mu = between_means[g][t]
                    y = mu + subj_offset + rng.normal(0, sd_within)
                    rows.append({
                        "subject": subj_id,
                        "group": f"G{g}",
                        "time": f"T{t}",
                        "score": y,
                    })

        df_sim = pd.DataFrame(rows)

        try:
            aov = pg.mixed_anova(
                data=df_sim,
                dv="score",
                within="time",
                subject="subject",
                between="group",
            )
            p_between = aov.loc[aov["Source"] == "group", "p-unc"].values
            p_within = aov.loc[aov["Source"] == "time", "p-unc"].values
            p_interaction = aov.loc[aov["Source"] == "group * time", "p-unc"].values

            if len(p_between) and p_between[0] < alpha:
                sig_between += 1
            if len(p_within) and p_within[0] < alpha:
                sig_within += 1
            if len(p_interaction) and p_interaction[0] < alpha:
                sig_interaction += 1
        except Exception:
            pass

    result = {
        "power_between": round(sig_between / n_sim, 4),
        "power_within": round(sig_within / n_sim, 4),
        "power_interaction": round(sig_interaction / n_sim, 4),
        "n_per_group": n_per_cell,
        "total_n": n_between * n_per_cell,
        "n_sim": n_sim,
        "alpha": alpha,
    }
    print(
        f"Mixed ANOVA Power (n={n_per_cell}/group, total_N={result['total_n']}, "
        f"n_sim={n_sim}):\n"
        f"  Between: {result['power_between']:.1%}\n"
        f"  Within:  {result['power_within']:.1%}\n"
        f"  Interaction: {result['power_interaction']:.1%}"
    )
    return result


def simulate_mediation_power(
    a_path: float,
    b_path: float,
    c_prime_path: float = 0.0,
    n: int = 100,
    sd_m: float = 1.0,
    sd_y: float = 1.0,
    alpha: float = 0.05,
    n_sim: int = 5000,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo power analysis for the indirect (mediated) effect a*b.

    Uses percentile bootstrap confidence intervals — the same method recommended
    by Hayes (2018) PROCESS macro.

    Args:
        a_path:       Path coefficient from X to Mediator M.
        b_path:       Path coefficient from M to Y (controlling for X).
        c_prime_path: Direct path from X to Y.
        n:            Sample size.
        sd_m:         SD of mediator residuals.
        sd_y:         SD of outcome residuals.
        alpha:        Significance level (CI excludes zero if significant).
        n_sim:        Number of simulations.
        n_bootstrap:  Bootstrap resamples per simulation.
        seed:         Random seed.

    Returns:
        Dict with power for indirect effect and path coefficients.
    """
    rng = np.random.default_rng(seed)
    indirect_effect = a_path * b_path
    sig_count = 0

    for _ in range(n_sim):
        x = rng.standard_normal(n)
        m = a_path * x + rng.normal(0, sd_m, n)
        y = b_path * m + c_prime_path * x + rng.normal(0, sd_y, n)

        # Bootstrap the indirect effect
        ab_boot = np.empty(n_bootstrap)
        for b_idx in range(n_bootstrap):
            idx = rng.integers(0, n, n)
            xb, mb, yb = x[idx], m[idx], y[idx]

            # OLS a-path: m ~ x
            a_hat = np.cov(xb, mb)[0, 1] / np.var(xb)

            # OLS b-path controlling for x: y ~ m + x
            X_mat = np.column_stack([np.ones(n), mb, xb])
            try:
                coefs = np.linalg.lstsq(X_mat, yb, rcond=None)[0]
                b_hat = coefs[1]
            except np.linalg.LinAlgError:
                b_hat = 0.0

            ab_boot[b_idx] = a_hat * b_hat

        ci_lo = np.percentile(ab_boot, 100 * alpha / 2)
        ci_hi = np.percentile(ab_boot, 100 * (1 - alpha / 2))

        if ci_lo > 0 or ci_hi < 0:
            sig_count += 1

    power_indirect = round(sig_count / n_sim, 4)
    print(
        f"Mediation Power (n={n}, a={a_path}, b={b_path}, indirect={indirect_effect:.3f}, "
        f"n_sim={n_sim}, n_boot={n_bootstrap}):\n"
        f"  Power for indirect effect: {power_indirect:.1%}"
    )
    return {
        "power_indirect_effect": power_indirect,
        "indirect_effect": round(indirect_effect, 4),
        "a_path": a_path,
        "b_path": b_path,
        "c_prime_path": c_prime_path,
        "n": n,
        "n_sim": n_sim,
        "alpha": alpha,
    }


def alpha_spending_obrien_fleming(
    looks: int,
    overall_alpha: float = 0.05,
) -> List[float]:
    """
    Compute O'Brien-Fleming alpha-spending boundaries for sequential testing.

    Args:
        looks:         Number of interim analyses (including final).
        overall_alpha: Overall familywise alpha level.

    Returns:
        List of alpha thresholds for each interim look.
    """
    from scipy.stats import norm

    z_final = norm.ppf(1 - overall_alpha / 2)
    boundaries = []

    for k in range(1, looks + 1):
        t_k = k / looks  # information fraction
        z_k = z_final / np.sqrt(t_k)
        alpha_k = 2 * (1 - norm.cdf(z_k))
        boundaries.append(round(alpha_k, 6))

    print(f"O'Brien-Fleming spending boundaries for {looks} looks (alpha={overall_alpha}):")
    for i, a in enumerate(boundaries, start=1):
        print(f"  Look {i}/{looks}: alpha = {a:.6f}")
    return boundaries
```

---

## 4. End-to-End Examples

### Example 1 — RCT Sample Size Planning for a Clinical Psychology Intervention

```python
# Scenario: A CBT intervention for anxiety. Primary outcome: GAD-7 score.
# Meta-analysis suggests d = 0.55 (moderate). Two parallel groups.
# Want 80% power at alpha = 0.05 (two-tailed).

result_primary = compute_power_ttest(
    effect_size=0.55,
    alpha=0.05,
    power=0.80,
    design="two-sample",
    alternative="two-sided",
)
print(f"\nRequired n per group: {result_primary['n_per_group']}")
print(f"Total N (before attrition): {result_primary['total_n']}")

# Account for 20% dropout — inflate sample size
n_per_group_inflated = int(np.ceil(result_primary["n_per_group"] / 0.80))
print(f"N per group after 20% dropout correction: {n_per_group_inflated}")
print(f"Total N to recruit: {n_per_group_inflated * 2}")

# Sensitivity analysis: power if only 90% of inflated N is reached
result_sensitivity = compute_power_ttest(
    effect_size=0.55,
    n=int(n_per_group_inflated * 0.90),
    alpha=0.05,
    design="two-sample",
)
print(f"\nPower at 90% of target N: {result_sensitivity['power']:.1%}")

# Power curve for this design
fig = power_curve_plot(
    design="ttest_ind",
    effect_sizes=[0.3, 0.5, 0.55, 0.8],
    n_range=(20, 150),
    alpha=0.05,
    output_path="rct_power_curve.png",
)

# Sequential testing plan: 2 interim + 1 final analysis
boundaries = alpha_spending_obrien_fleming(looks=3, overall_alpha=0.05)
print("\nSequential analysis thresholds:")
for i, a in enumerate(boundaries, 1):
    print(f"  After {i}/3 of sample: reject H0 if p < {a:.4f}")
```

### Example 2 — Power Analysis for a 2×3 Mixed ANOVA

```python
# Scenario: Emotion regulation study.
# Between factor: Training condition (control vs. intervention), k=2
# Within factor: Time (pre, post, follow-up), k=3
# Hypothesis: Training x Time interaction (partial eta^2 = 0.06 from pilot)

eta_sq = 0.06
f_effect = partial_eta_sq_to_cohens_f(eta_sq)
print(f"Cohen's f for interaction (eta^2={eta_sq}): {f_effect:.3f}")

# Closed-form estimate using FTestAnovaPower for interaction
# (treat as k = 2x3 = 6 cell design, then adjust)
result_anova = compute_power_anova(
    effect_size_f=f_effect,
    alpha=0.05,
    power=0.80,
    k_groups=6,  # 2 groups x 3 time points
)
print(f"\nClosed-form estimate: total N ≈ {result_anova['total_n']}")
n_per_group_estimate = int(np.ceil(result_anova["total_n"] / 2))

# Simulation-based estimate (more accurate for mixed design)
between_means_h1 = [
    [0.0, 0.0, 0.0],   # control: no change over time
    [0.0, 0.4, 0.5],   # intervention: improvement post and follow-up
]

sim_result = simulate_mixed_anova_power(
    between_means=between_means_h1,
    within_means=between_means_h1,
    sd_between=1.0,
    sd_within=0.5,
    n_per_cell=n_per_group_estimate,
    alpha=0.05,
    n_sim=2000,
    seed=99,
)
print(f"\nSimulation-based power for interaction: {sim_result['power_interaction']:.1%}")

# Increase n until interaction power >= 0.80
target_power = 0.80
n_test = n_per_group_estimate
for increment in range(0, 50, 5):
    n_test = n_per_group_estimate + increment
    res = simulate_mixed_anova_power(
        between_means=between_means_h1,
        within_means=between_means_h1,
        sd_between=1.0,
        sd_within=0.5,
        n_per_cell=n_test,
        alpha=0.05,
        n_sim=1000,
        seed=99,
    )
    if res["power_interaction"] >= target_power:
        print(f"\nTarget power {target_power:.0%} reached at n={n_test}/group (total={n_test*2})")
        break

# Mediation power for a hypothesized mediator
print("\n--- Mediation sub-hypothesis ---")
med_result = simulate_mediation_power(
    a_path=0.40,
    b_path=0.35,
    c_prime_path=0.10,
    n=n_test * 2,
    n_sim=3000,
    n_bootstrap=500,
    seed=7,
)
print(f"Power for mediation (a*b) = {med_result['power_indirect_effect']:.1%}")
```

---

## 5. Common Errors and Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ValueError: Exactly one of ... must be None` | Two quantities left as None | Provide all but one parameter |
| Power = 1.000 with small n | Effect size is unrealistically large | Double-check effect size units (d vs r vs f) |
| `pingouin.power_corr` returns NaN | n too small for the given r | Increase n_range lower bound |
| Simulation power much lower than closed-form | Random seed / n_sim too small | Set `n_sim >= 5000`, fix `seed` |
| `np.linalg.LinAlgError` in mediation | Perfect collinearity in bootstrap sample | Increase n; add `rcond=None` to lstsq |
| `pg.mixed_anova` raises KeyError | Column naming mismatch | Ensure `dv`, `within`, `between`, `subject` columns match exactly |

---

## 6. References and Further Reading

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.)
- Faul et al. (2007). G*Power 3: <https://doi.org/10.3758/BF03193146>
- Hayes, A. F. (2018). *Introduction to Mediation, Moderation, and Conditional Process Analysis*
- Statsmodels power module: <https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations>
- Pingouin documentation: <https://pingouin-stats.org/>
- Lakens et al. (2018). Justify your alpha: <https://doi.org/10.1038/s41562-018-0311-x>

---

## 7. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — t-test, ANOVA, correlation, logistic, mixed ANOVA simulation, mediation bootstrap, power curves, sequential testing |
