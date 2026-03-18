---
name: synthetic-control
description: >
  Use this Skill for synthetic control causal inference: donor pool selection,
  weight optimization via scipy, pre-period balance, post-period gap plot,
  placebo tests, and permutation p-values.
tags:
  - economics
  - synthetic-control
  - causal-inference
  - policy-evaluation
  - counterfactual
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
    - pandas>=1.5
    - numpy>=1.23
    - scipy>=1.9
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Synthetic Control Method

> **TL;DR** — Construct a data-driven counterfactual for a single treated unit using
> a weighted combination of untreated donor units. Estimate policy effects as the
> post-treatment gap between actual and synthetic outcomes, and assess significance
> via permutation placebo tests.

---

## When to Use

| Situation | Recommendation |
|---|---|
| Single treated unit (country, state, firm) | Primary use case for synthetic control |
| Comparative case studies | Strong alternative to simple DiD |
| Long pre-treatment time series (T_pre ≥ 10) | Required for reliable weight estimation |
| Small donor pool (5–40 donors) | Feasible; quality depends on fit |
| Staggered or multiple treated units | Use generalized synthetic control instead |
| Short pre-treatment window (< 5 periods) | Prefer DiD or regression discontinuity |

The synthetic control method (Abadie, Diamond, and Hainmueller 2010) is ideal when
you study a large-scale policy intervention affecting a single aggregate unit — a
country enacting a trade reform, a state adopting a minimum wage law, a firm
receiving a large government subsidy — and you have panel data on similar untreated
units over a long pre-treatment window.

**Key advantages over DiD:**
- No parallel trends assumption required in the same form
- Data-driven donor weighting makes counterfactual explicit and transparent
- Permutation-based inference does not rely on asymptotic theory
- Visual inspection of pre-period fit validates the control

---

## Background

### Abadie-Diamond-Hainmueller Framework

Let unit 1 be the treated unit and units 2, …, J+1 be the donor pool. For each
pre-treatment period t ≤ T_0, we observe outcome Y_{jt}. For the treated unit in
the post-treatment period t > T_0, we want to estimate the counterfactual Y_{1t}(0).

**Synthetic control estimator:**

    Ŷ_{1t}(0) = Σ_{j=2}^{J+1} w_j* × Y_{jt}

where weights W* = (w_2*, …, w_{J+1}*) are chosen to minimize:

    MSPE_pre = Σ_{t=1}^{T_0} (Y_{1t} − Σ_j w_j Y_{jt})²

subject to: w_j ≥ 0 for all j and Σ_j w_j = 1.

**Treatment effect estimate:**

    α̂_{1t} = Y_{1t} − Ŷ_{1t}(0)   for t > T_0

**Gap statistic:**

    Gap_t = Y_{1t} − Ŷ_{1t}(0)

A positive post-period gap indicates a positive treatment effect.

### Predictor Matching vs Outcome Matching

Two approaches exist for constructing W*:

1. **Outcome path matching** — minimize MSPE over all pre-treatment outcome periods.
   Simple and often sufficient for good fit.

2. **Predictor + outcome matching** — minimize a weighted distance over pre-treatment
   outcome averages AND covariate predictors (GDP per capita, population, etc.) using
   an outer optimization over predictor weights V.

This Skill implements outcome path matching (simpler, more transparent), which is the
most common choice in practice.

### Inference via Permutation

Because there is no asymptotic distribution to rely on with a single treated unit,
inference uses **in-space placebos**:

1. Apply the synthetic control method to each donor unit j as if it were treated.
2. Compute the post/pre RMSPE ratio for each placebo:
   RMSPE_ratio_j = RMSPE_{post,j} / RMSPE_{pre,j}
3. The permutation p-value = rank(RMSPE_ratio_treated) / (J + 1).
4. Restrict placebo set to donors with RMSPE_{pre} ≤ 2 × RMSPE_{pre, treated}
   to avoid donors with poor pre-period fit dominating the reference distribution.

A p-value ≤ 0.10 is considered significant in standard practice.

---

## Environment Setup

```bash
# Create a dedicated conda environment
conda create -n synth python=3.11 -y
conda activate synth

# Install dependencies
pip install pandas>=1.5 numpy>=1.23 scipy>=1.9 matplotlib>=3.6

# Verify installation
python -c "import pandas, numpy, scipy, matplotlib; print('All packages imported successfully')"
```

No proprietary data access is required. The examples below use synthetic data that
are generated within the script itself.

---

## Core Workflow

### Step 1 — Build the Panel and Donor Pool

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint

np.random.seed(42)


def generate_synthetic_panel(
    n_donors: int = 20,
    n_periods: int = 30,
    treatment_period: int = 20,
    true_effect: float = 3.0,
) -> pd.DataFrame:
    """
    Generate a panel dataset for synthetic control exercises.

    The treated unit (unit_id=0) follows a linear combination of donors in
    the pre-treatment period plus a post-treatment jump of `true_effect`.

    Args:
        n_donors:         Number of untreated donor units.
        n_periods:        Total number of time periods.
        treatment_period: First post-treatment period (0-indexed).
        true_effect:      True average treatment effect post-treatment.

    Returns:
        DataFrame with columns: unit_id, period, outcome, treated.
    """
    # Donor outcomes: AR(1) process with unit-specific means
    unit_means = np.random.uniform(5, 15, n_donors)
    donor_outcomes = np.zeros((n_donors, n_periods))
    for j in range(n_donors):
        donor_outcomes[j, 0] = unit_means[j] + np.random.normal(0, 0.5)
        for t in range(1, n_periods):
            donor_outcomes[j, t] = (
                0.7 * donor_outcomes[j, t - 1]
                + 0.3 * unit_means[j]
                + np.random.normal(0, 0.5)
            )

    # True synthetic weights: use first 4 donors with equal weight
    true_weights = np.zeros(n_donors)
    true_weights[:4] = 0.25

    # Treated unit: weighted combination of donors pre-treatment
    treated_outcomes = np.zeros(n_periods)
    for t in range(n_periods):
        treated_outcomes[t] = (
            true_weights @ donor_outcomes[:, t]
            + np.random.normal(0, 0.3)
            + (true_effect if t >= treatment_period else 0.0)
        )

    rows = []
    # Treated unit (unit_id=0)
    for t in range(n_periods):
        rows.append({"unit_id": 0, "period": t, "outcome": treated_outcomes[t], "treated": 1})
    # Donor units
    for j in range(n_donors):
        for t in range(n_periods):
            rows.append({"unit_id": j + 1, "period": t, "outcome": donor_outcomes[j, t], "treated": 0})

    return pd.DataFrame(rows)


def build_matrices(df: pd.DataFrame, treatment_period: int, treated_id: int = 0):
    """
    Extract treated outcome vector and donor outcome matrix.

    Args:
        df:               Panel DataFrame with columns unit_id, period, outcome.
        treatment_period: Index of first post-treatment period.
        treated_id:       unit_id of the treated unit.

    Returns:
        y_treated (T,), Y_donors (J, T), pre_mask (T,), post_mask (T,)
    """
    periods = sorted(df["period"].unique())
    T = len(periods)

    y_treated = np.array([
        df.loc[(df["unit_id"] == treated_id) & (df["period"] == t), "outcome"].values[0]
        for t in periods
    ])

    donor_ids = sorted(df.loc[df["unit_id"] != treated_id, "unit_id"].unique())
    Y_donors = np.array([
        [df.loc[(df["unit_id"] == j) & (df["period"] == t), "outcome"].values[0]
         for t in periods]
        for j in donor_ids
    ])  # shape: (J, T)

    pre_mask = np.array([t < treatment_period for t in periods])
    post_mask = ~pre_mask

    return y_treated, Y_donors, pre_mask, post_mask, donor_ids
```

### Step 2 — Optimize Synthetic Weights and Compute Gap

```python
def fit_synthetic_control(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    pre_mask: np.ndarray,
) -> np.ndarray:
    """
    Find donor weights W* by minimizing pre-period MSPE.

    Objective:  min_w  Σ_{t in pre} (Y_{treated,t} - w @ Y_donors[:,t])^2
    Subject to: w_j >= 0 for all j,  sum(w) = 1.

    Args:
        y_treated: Treated unit outcomes, shape (T,).
        Y_donors:  Donor outcomes, shape (J, T).
        pre_mask:  Boolean mask for pre-treatment periods, shape (T,).

    Returns:
        Optimal weight vector w*, shape (J,).
    """
    J = Y_donors.shape[0]
    y_pre = y_treated[pre_mask]           # (T_pre,)
    Y_pre = Y_donors[:, pre_mask]         # (J, T_pre)

    def mspe(w):
        synthetic_pre = w @ Y_pre         # (T_pre,)
        return np.mean((y_pre - synthetic_pre) ** 2)

    def mspe_grad(w):
        synthetic_pre = w @ Y_pre
        residuals = synthetic_pre - y_pre
        return 2 * (Y_pre @ residuals) / len(y_pre)

    # Constraints: sum(w) = 1, w >= 0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * J
    w0 = np.ones(J) / J

    result = minimize(
        mspe,
        w0,
        jac=mspe_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if not result.success:
        print(f"Warning: optimizer did not converge — {result.message}")

    w_star = result.x
    w_star = np.clip(w_star, 0, None)
    w_star /= w_star.sum()               # re-normalize for numerical safety
    return w_star


def compute_gap(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    w_star: np.ndarray,
) -> np.ndarray:
    """
    Compute the treatment gap: actual minus synthetic outcome.

    Args:
        y_treated: Treated outcomes, shape (T,).
        Y_donors:  Donor outcomes, shape (J, T).
        w_star:    Fitted weights, shape (J,).

    Returns:
        Gap series of shape (T,).
    """
    synthetic = w_star @ Y_donors        # (T,)
    return y_treated - synthetic


def plot_gap(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    w_star: np.ndarray,
    pre_mask: np.ndarray,
    donor_ids: list,
    output_path: str = None,
) -> None:
    """
    Plot actual vs synthetic outcome and the treatment gap.

    Args:
        y_treated:   Treated outcomes, shape (T,).
        Y_donors:    Donor outcomes, shape (J, T).
        w_star:      Fitted weights, shape (J,).
        pre_mask:    Boolean mask for pre-treatment periods.
        donor_ids:   List of donor unit IDs for the legend.
        output_path: If provided, save figure to this path.
    """
    T = len(y_treated)
    periods = np.arange(T)
    T_0 = np.sum(pre_mask)
    synthetic = w_star @ Y_donors

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel A: outcome paths
    ax = axes[0]
    ax.plot(periods, y_treated, color="#C0392B", linewidth=2, label="Treated unit")
    ax.plot(periods, synthetic, color="#2980B9", linewidth=2, linestyle="--", label="Synthetic control")
    for j in range(Y_donors.shape[0]):
        ax.plot(periods, Y_donors[j], color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(T_0 - 0.5, color="black", linestyle=":", linewidth=1.2, label="Treatment onset")
    ax.set_ylabel("Outcome")
    ax.set_title("Synthetic Control: Actual vs Synthetic")
    ax.legend()

    # Panel B: gap
    gap = y_treated - synthetic
    ax2 = axes[1]
    ax2.plot(periods, gap, color="#8E44AD", linewidth=2)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.axvline(T_0 - 0.5, color="black", linestyle=":", linewidth=1.2)
    ax2.fill_between(periods[~pre_mask], gap[~pre_mask], 0,
                     alpha=0.25, color="#8E44AD", label="Post-treatment gap")
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Gap (Actual − Synthetic)")
    ax2.set_title("Treatment Effect Gap")
    ax2.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved gap plot to {output_path}")
    plt.show()
```

### Step 3 — Permutation Placebo Tests

```python
def compute_rmspe_ratio(
    y_unit: np.ndarray,
    Y_donors_excl: np.ndarray,
    pre_mask: np.ndarray,
) -> tuple:
    """
    Fit synthetic control for a single unit and return pre/post RMSPE ratio.

    Args:
        y_unit:         Outcome for the focal unit, shape (T,).
        Y_donors_excl:  Donor pool excluding this unit, shape (J-1, T).
        pre_mask:       Boolean mask for pre-treatment periods.

    Returns:
        Tuple of (rmspe_pre, rmspe_post, rmspe_ratio, w_star).
    """
    w_star = fit_synthetic_control(y_unit, Y_donors_excl, pre_mask)
    gap = compute_gap(y_unit, Y_donors_excl, w_star)

    rmspe_pre = np.sqrt(np.mean(gap[pre_mask] ** 2))
    rmspe_post = np.sqrt(np.mean(gap[~pre_mask] ** 2))
    ratio = rmspe_post / rmspe_pre if rmspe_pre > 0 else np.inf
    return rmspe_pre, rmspe_post, ratio, w_star


def in_space_placebo_test(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    pre_mask: np.ndarray,
    donor_ids: list,
    max_pre_rmspe_multiplier: float = 2.0,
    output_path: str = None,
) -> dict:
    """
    Run in-space placebo tests and compute permutation p-value.

    For each donor unit, apply synthetic control as if it were treated.
    Compute RMSPE ratios for all units. The p-value is the fraction of
    donors (with acceptable pre-period fit) whose ratio exceeds the
    treated unit's ratio.

    Args:
        y_treated:                 Treated unit outcomes, shape (T,).
        Y_donors:                  Donor outcomes, shape (J, T).
        pre_mask:                  Boolean mask for pre-treatment periods.
        donor_ids:                 Donor unit IDs.
        max_pre_rmspe_multiplier:  Exclude placebos with pre-RMSPE >
                                   multiplier * treated_pre_rmspe.
        output_path:               If provided, save distribution plot.

    Returns:
        Dictionary with keys: treated_ratio, placebo_ratios, p_value, n_placebos.
    """
    J = Y_donors.shape[0]
    rmspe_pre_treated, rmspe_post_treated, ratio_treated, _ = compute_rmspe_ratio(
        y_treated, Y_donors, pre_mask
    )
    print(f"Treated unit — RMSPE_pre: {rmspe_pre_treated:.4f}, "
          f"RMSPE_post: {rmspe_post_treated:.4f}, ratio: {ratio_treated:.4f}")

    placebo_ratios = []
    placebo_pre_rmspes = []
    for j in range(J):
        # Treated = donor j; new donor pool = treated + remaining donors
        y_placebo = Y_donors[j]
        remaining = np.delete(np.arange(J), j)
        Y_pool = np.vstack([y_treated.reshape(1, -1), Y_donors[remaining]])

        rmspe_pre_j, _, ratio_j, _ = compute_rmspe_ratio(y_placebo, Y_pool, pre_mask)
        placebo_ratios.append(ratio_j)
        placebo_pre_rmspes.append(rmspe_pre_j)

    placebo_ratios = np.array(placebo_ratios)
    placebo_pre_rmspes = np.array(placebo_pre_rmspes)

    # Filter: only retain placebos with reasonable pre-period fit
    threshold = max_pre_rmspe_multiplier * rmspe_pre_treated
    keep = placebo_pre_rmspes <= threshold
    filtered_ratios = placebo_ratios[keep]
    n_placebos = len(filtered_ratios)

    # Permutation p-value
    p_value = np.mean(filtered_ratios >= ratio_treated)

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(filtered_ratios, bins=15, color="#BDC3C7", edgecolor="white", label="Placebo ratios")
    ax.axvline(ratio_treated, color="#C0392B", linewidth=2, label=f"Treated ratio = {ratio_treated:.2f}")
    ax.set_xlabel("Post/Pre RMSPE Ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"In-Space Placebo Distribution — p-value = {p_value:.3f}")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved placebo distribution to {output_path}")
    plt.show()

    return {
        "treated_ratio": ratio_treated,
        "placebo_ratios": filtered_ratios,
        "p_value": p_value,
        "n_placebos": n_placebos,
    }
```

---

## Advanced Usage

### Predictor-Augmented Weight Optimization

When you have pre-treatment predictors (X matrix), you can optimize an outer
weight matrix V over predictors and an inner weight matrix W over donors:

```python
def fit_predictor_augmented_sc(
    y_treated_pre: np.ndarray,
    Y_donors_pre: np.ndarray,
    X_treated: np.ndarray,
    X_donors: np.ndarray,
    n_v_restarts: int = 5,
) -> np.ndarray:
    """
    Fit predictor-augmented synthetic control (outer V + inner W optimization).

    Minimizes: ||V^{1/2}(X_treated - X_donors @ w)||^2 + ||y_pre - Y_donors_pre @ w||^2

    Args:
        y_treated_pre: Treated outcome in pre-period, shape (T_pre,).
        Y_donors_pre:  Donor outcomes in pre-period, shape (J, T_pre).
        X_treated:     Treated predictor vector, shape (K,).
        X_donors:      Donor predictors, shape (J, K).
        n_v_restarts:  Number of random restarts for outer optimization.

    Returns:
        Optimal weight vector w*, shape (J,).
    """
    J, K = X_donors.shape

    best_w = None
    best_obj = np.inf

    for _ in range(n_v_restarts):
        v0 = np.random.dirichlet(np.ones(K + 1))  # K predictors + outcome path

        def outer_objective(v_raw):
            v = np.exp(v_raw) / np.sum(np.exp(v_raw))  # softmax to ensure positive
            v_pred = v[:K]
            v_out = v[K]

            def inner_obj(w):
                pred_gap = X_treated - X_donors.T @ w   # (K,)
                out_gap = y_treated_pre - Y_donors_pre.T @ w  # (T_pre,)
                return v_pred @ (pred_gap ** 2) + v_out * np.mean(out_gap ** 2)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(0.0, 1.0)] * J
            res = minimize(inner_obj, np.ones(J) / J, method="SLSQP",
                           bounds=bounds, constraints=constraints)
            return res.fun

        res_outer = minimize(outer_objective, np.log(v0 + 1e-8),
                             method="Nelder-Mead",
                             options={"maxiter": 500, "xatol": 1e-6})

        if res_outer.fun < best_obj:
            best_obj = res_outer.fun
            v_best = np.exp(res_outer.x) / np.sum(np.exp(res_outer.x))
            v_pred_best = v_best[:K]
            v_out_best = v_best[K]

            def inner_obj_final(w):
                pred_gap = X_treated - X_donors.T @ w
                out_gap = y_treated_pre - Y_donors_pre.T @ w
                return v_pred_best @ (pred_gap ** 2) + v_out_best * np.mean(out_gap ** 2)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(0.0, 1.0)] * J
            res_inner = minimize(inner_obj_final, np.ones(J) / J, method="SLSQP",
                                 bounds=bounds, constraints=constraints)
            best_w = np.clip(res_inner.x, 0, None)
            best_w /= best_w.sum()

    return best_w


# In-time placebo: shift treatment date to an earlier period
def in_time_placebo(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    true_treatment_period: int,
    placebo_period: int,
) -> float:
    """
    Run an in-time placebo test by treating an earlier period as treatment.

    If the placebo gap is large, the identifying assumption may be violated.

    Args:
        y_treated:              Full outcome series for treated unit.
        Y_donors:               Full donor outcomes matrix.
        true_treatment_period:  Actual treatment start index.
        placebo_period:         Fake treatment start index (must be < true_treatment_period).

    Returns:
        Mean post-placebo gap (should be near zero for validity).
    """
    assert placebo_period < true_treatment_period, "Placebo period must precede real treatment."
    # Use only data strictly before the real treatment
    y_sub = y_treated[:true_treatment_period]
    Y_sub = Y_donors[:, :true_treatment_period]
    pre_mask_sub = np.arange(true_treatment_period) < placebo_period

    w_star = fit_synthetic_control(y_sub, Y_sub, pre_mask_sub)
    gap_sub = compute_gap(y_sub, Y_sub, w_star)
    post_gap = gap_sub[~pre_mask_sub]
    mean_placebo_gap = float(np.mean(post_gap))
    print(f"In-time placebo (period {placebo_period}): mean gap = {mean_placebo_gap:.4f}")
    return mean_placebo_gap
```

---

## Troubleshooting

| Error / Issue | Likely Cause | Resolution |
|---|---|---|
| Optimizer does not converge | Poorly scaled outcomes across donors | Standardize all outcome series before optimization |
| All weight placed on 1–2 donors | Highly collinear donor pool | Verify donor pool diversity; add more donors |
| Terrible pre-period fit (RMSPE_pre >> 0) | Treated unit unlike any donor | Reconsider donor pool; use generalized SC |
| p-value = 1.0 (no placebo exceeds treated) | Extremely large post-treatment effect | Check for data errors; verify treatment date |
| p-value = 0 (all placebos exceed treated) | Poor fit — pre-period RMSPE already large | Expand donor pool; check for unit-specific shocks |
| Negative weights from optimizer | Numerical drift in SLSQP | Re-clip and re-normalize; tighten bounds |
| `LinAlgError: SVD did not converge` | Collinear donor matrix | Drop duplicate or near-duplicate donors |
| `KeyError` on donor lookup | Mismatched unit_id encoding | Ensure integer unit IDs; reset index before build_matrices |

---

## External Resources

- Abadie, A., Diamond, A., Hainmueller, J. (2010). "Synthetic Control Methods for
  Comparative Case Studies." *JASA*, 105(490), 493–505.
  <https://doi.org/10.1198/jasa.2009.ap08746>
- Abadie, A., Diamond, A., Hainmueller, J. (2015). "Comparative Politics and the
  Synthetic Control Method." *AJPS*, 59(2), 495–510.
- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and
  Methodological Aspects." *Journal of Economic Literature*, 59(2), 391–425.
- Ben-Michael, E., Feller, A., Rothstein, J. (2021). "The Augmented Synthetic Control
  Method." *JASA*, 116(536), 1789–1803. (Augmented SCM with LASSO regularization)
- `pysyncon` Python package: <https://github.com/sdfordham/pysyncon>
- Stata `synth` package: <https://web.stanford.edu/~jhain/synthpage.html>

---

## Examples

### Example 1 — Full Pipeline on Synthetic Data

```python
# Full synthetic control pipeline
df = generate_synthetic_panel(n_donors=20, n_periods=30, treatment_period=20, true_effect=3.0)

T_0 = 20
y_treated, Y_donors, pre_mask, post_mask, donor_ids = build_matrices(df, T_0, treated_id=0)

# Step 1: optimize weights
w_star = fit_synthetic_control(y_treated, Y_donors, pre_mask)
top_donors = sorted(zip(donor_ids, w_star), key=lambda x: -x[1])[:5]
print("Top 5 donors by weight:")
for did, ww in top_donors:
    print(f"  Unit {did}: {ww:.4f}")

# Step 2: pre-period RMSPE
gap = compute_gap(y_treated, Y_donors, w_star)
rmspe_pre = np.sqrt(np.mean(gap[pre_mask] ** 2))
rmspe_post = np.sqrt(np.mean(gap[post_mask] ** 2))
print(f"\nRMSPE_pre  = {rmspe_pre:.4f}")
print(f"RMSPE_post = {rmspe_post:.4f}")
print(f"RMSPE_ratio = {rmspe_post / rmspe_pre:.4f}")

# Step 3: gap plot
plot_gap(y_treated, Y_donors, w_star, pre_mask, donor_ids, output_path="gap_plot.png")

# Step 4: permutation p-value
results = in_space_placebo_test(
    y_treated, Y_donors, pre_mask, donor_ids, output_path="placebo_dist.png"
)
print(f"\nPermutation p-value: {results['p_value']:.3f}  (n_placebos = {results['n_placebos']})")
```

### Example 2 — Sensitivity: Dropping Each Donor One at a Time

```python
def leave_one_out_sensitivity(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    pre_mask: np.ndarray,
    donor_ids: list,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Leave-one-out sensitivity: remove each donor and refit. Checks robustness.

    Args:
        y_treated:  Treated outcomes, shape (T,).
        Y_donors:   Donor outcomes, shape (J, T).
        pre_mask:   Pre-treatment mask.
        donor_ids:  List of donor unit IDs.
        output_path: Save figure if provided.

    Returns:
        DataFrame with columns: dropped_donor, rmspe_pre, rmspe_post, mean_post_gap.
    """
    T = len(y_treated)
    periods = np.arange(T)
    T_0 = np.sum(pre_mask)
    J = Y_donors.shape[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(periods, y_treated, color="#C0392B", linewidth=2.5, label="Treated unit", zorder=5)

    records = []
    for j in range(J):
        remaining = np.delete(np.arange(J), j)
        Y_loo = Y_donors[remaining]
        w_loo = fit_synthetic_control(y_treated, Y_loo, pre_mask)
        synthetic_loo = w_loo @ Y_loo
        gap_loo = y_treated - synthetic_loo

        rmspe_pre_j = np.sqrt(np.mean(gap_loo[pre_mask] ** 2))
        rmspe_post_j = np.sqrt(np.mean(gap_loo[~pre_mask] ** 2))

        ax.plot(periods, synthetic_loo, color="steelblue", linewidth=0.8, alpha=0.5)
        records.append({
            "dropped_donor": donor_ids[j],
            "rmspe_pre": rmspe_pre_j,
            "rmspe_post": rmspe_post_j,
            "mean_post_gap": float(np.mean(gap_loo[~pre_mask])),
        })

    # Full model synthetic
    w_full = fit_synthetic_control(y_treated, Y_donors, pre_mask)
    ax.plot(periods, w_full @ Y_donors, color="#2980B9", linewidth=2.5,
            linestyle="--", label="Full synthetic control", zorder=5)
    ax.axvline(T_0 - 0.5, color="black", linestyle=":", linewidth=1.2, label="Treatment onset")
    ax.set_xlabel("Period")
    ax.set_ylabel("Outcome")
    ax.set_title("Leave-One-Out Sensitivity")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved LOO sensitivity plot to {output_path}")
    plt.show()

    return pd.DataFrame(records)


# Run leave-one-out sensitivity
loo_df = leave_one_out_sensitivity(
    y_treated, Y_donors, pre_mask, donor_ids, output_path="loo_sensitivity.png"
)
print("\nLeave-one-out sensitivity summary:")
print(loo_df.describe().round(4))
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — weight optimization, gap plot, in-space placebo, LOO sensitivity |
