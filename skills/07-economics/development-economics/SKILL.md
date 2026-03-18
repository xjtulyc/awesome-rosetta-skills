---
name: development-economics
description: >
  Use this Skill for development economics: LSMS data loading, RCT ITT analysis,
  Lee sharp bounds, covariate balance table, propensity score matching, and
  poverty indices (FGT).
tags:
  - economics
  - development
  - RCT
  - propensity-score-matching
  - poverty
  - LSMS
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
    - statsmodels>=0.14
    - pandas>=1.5
    - numpy>=1.23
    - scipy>=1.9
    - scikit-learn>=1.2
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Development Economics Analysis

> **TL;DR** — Analyze RCT data with ITT regressions, test covariate balance with
> love plots, correct for selective attrition using Lee bounds, estimate treatment
> effects using propensity score matching, and compute FGT poverty indices from
> household survey data.

---

## When to Use

| Situation | Recommended Method |
|---|---|
| Randomized control trial — intention to treat | OLS with strata fixed effects |
| RCT with partial compliance — LATE | IV regression (treatment assignment as instrument) |
| Systematic attrition may bias estimates | Lee (2009) sharp bounds |
| Observational data with rich covariates | Propensity score matching (PSM) |
| Cross-country or household poverty measurement | FGT poverty indices |
| Validate randomization quality | Covariate balance table + love plot |

---

## Background

### Intent-to-Treat (ITT) in RCTs

The ITT estimator regresses the outcome on the randomly assigned treatment indicator
(not actual take-up), controlling for randomization strata:

    Y_i = α + β_ITT Z_i + γ_s strata_i + ε_i

where Z_i ∈ {0, 1} is random assignment. β_ITT is the average effect of being offered
treatment, regardless of compliance.

**Local Average Treatment Effect (LATE)** via IV is the ITT scaled by the compliance
rate. Z_i instruments for D_i (actual treatment take-up):
    LATE = ITT / (First Stage: E[D_i|Z_i=1] - E[D_i|Z_i=0])

### Covariate Balance

Balance test: At baseline, treated and control groups should have similar means for all
pre-treatment covariates. The standardized mean difference (SMD) is:
    SMD_k = (X̄_k^1 - X̄_k^0) / SD_pooled_k

Conventional threshold: |SMD| < 0.10 (Rosenbaum and Rubin 1985). A love plot displays
all SMDs graphically.

### Lee (2009) Sharp Bounds for Selective Attrition

If attrition is higher in treatment or control, the missing data may be non-random
(individuals who benefit more may be more likely to remain in the sample). Lee bounds
provide sharp bounds on the treatment effect by trimming the tails:

- Upper bound: Trim the bottom (1 - p_min/p_max) fraction from the high-attrition group.
- Lower bound: Trim the top (1 - p_min/p_max) fraction from the high-attrition group.

where p_0 and p_1 are retention rates in control and treatment groups respectively.

### Propensity Score Matching

PSM constructs a counterfactual by matching each treated unit to the most similar
control unit based on the propensity score e(X) = P(D=1|X).

Steps:
1. Estimate logistic propensity score.
2. Check common support (overlap).
3. Match using nearest-neighbor within a caliper (e.g., 0.25 × SD(log-odds)).
4. Assess balance after matching.
5. Estimate ATT = E[Y_i(1) - Y_i(0) | D_i = 1] on matched sample.

### FGT Poverty Indices

Foster-Greer-Thorbecke (1984) family of poverty measures:

    P_α = (1/n) Σ_{i: y_i < z} [(z - y_i) / z]^α

- α = 0: P_0 = headcount ratio (share below poverty line)
- α = 1: P_1 = poverty gap (average shortfall as share of poverty line)
- α = 2: P_2 = poverty severity (gives more weight to the poorest)

---

## Environment Setup

```bash
conda create -n devecono python=3.11 -y
conda activate devecono
pip install statsmodels>=0.14 pandas>=1.5 numpy>=1.23 scipy>=1.9 scikit-learn>=1.2 matplotlib>=3.6

python -c "import statsmodels, sklearn; print('OK')"
```

---

## Core Workflow

### Step 1 — ITT Regression and Covariate Balance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

np.random.seed(42)


def generate_rct_data(
    n: int = 1200,
    n_strata: int = 6,
    attrition_rate: float = 0.15,
    true_itt: float = 0.3,
) -> pd.DataFrame:
    """
    Simulate an RCT dataset with stratified randomization and selective attrition.

    Args:
        n:               Total sample size.
        n_strata:        Number of randomization strata.
        attrition_rate:  Overall attrition fraction (higher for treated group).
        true_itt:        True ITT effect size.

    Returns:
        DataFrame with columns: id, strata, Z (assignment), D (take-up),
                                 age, female, baseline_y, endline_y (NaN if attrited).
    """
    strata = np.random.randint(0, n_strata, n)
    # Stratified randomization: 50% treated in each stratum
    Z = np.zeros(n, dtype=int)
    for s in range(n_strata):
        idx = np.where(strata == s)[0]
        np.random.shuffle(idx)
        Z[idx[:len(idx) // 2]] = 1

    # Covariates
    age = np.random.normal(35, 8, n)
    female = np.random.binomial(1, 0.55, n)
    baseline_y = np.random.normal(1000, 200, n) + 50 * age * 0.01

    # Compliance: 80% of treated take up
    D = np.where(Z == 1, np.random.binomial(1, 0.8, n), 0)

    # Selective attrition: treated with lower baseline outcomes more likely to attrite
    attrit_prob_control = attrition_rate * 0.5
    attrit_prob_treat = attrition_rate * 1.5
    attrit = np.where(
        Z == 1,
        np.random.binomial(1, attrit_prob_treat, n),
        np.random.binomial(1, attrit_prob_control, n),
    )

    # Endline outcome
    endline_y = baseline_y + true_itt * 1000 * Z + np.random.normal(0, 150, n)
    endline_y = np.where(attrit == 1, np.nan, endline_y)

    return pd.DataFrame({
        "id": np.arange(n),
        "strata": strata,
        "Z": Z,
        "D": D,
        "age": age,
        "female": female,
        "baseline_y": baseline_y,
        "endline_y": endline_y,
        "attrited": attrit,
    })


def covariate_balance_table(
    df: pd.DataFrame,
    treatment_col: str = "Z",
    covariates: list = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Compute standardized mean differences and produce a love plot.

    Args:
        df:            RCT dataset.
        treatment_col: Binary treatment assignment column.
        covariates:    List of covariate names to check.
        output_path:   If provided, save love plot.

    Returns:
        DataFrame with columns: covariate, mean_treated, mean_control, smd, p_value.
    """
    if covariates is None:
        covariates = ["age", "female", "baseline_y"]

    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    records = []
    for cov in covariates:
        m1 = treated[cov].mean()
        m0 = control[cov].mean()
        s1 = treated[cov].std()
        s0 = control[cov].std()
        pooled_sd = np.sqrt((s1**2 + s0**2) / 2)
        smd = (m1 - m0) / (pooled_sd + 1e-10)

        t_stat, p_val = stats.ttest_ind(treated[cov].dropna(), control[cov].dropna())
        records.append({
            "covariate": cov,
            "mean_treated": round(m1, 4),
            "mean_control": round(m0, 4),
            "smd": round(smd, 4),
            "p_value": round(p_val, 4),
            "balanced": abs(smd) < 0.10,
        })

    bal_df = pd.DataFrame(records)
    print("Covariate Balance Table:")
    print(bal_df.to_string(index=False))
    n_unbalanced = (abs(bal_df["smd"]) >= 0.10).sum()
    print(f"\nUnbalanced covariates (|SMD| >= 0.10): {n_unbalanced} / {len(covariates)}")

    # Love plot
    fig, ax = plt.subplots(figsize=(8, max(4, len(covariates) * 0.5)))
    colors = ["#E74C3C" if not b else "#2ECC71" for b in bal_df["balanced"]]
    ax.scatter(bal_df["smd"], range(len(bal_df)), color=colors, s=80, zorder=3)
    ax.axvline(-0.10, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.10, color="gray", linestyle="--", linewidth=1, label="±0.10 threshold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(bal_df)))
    ax.set_yticklabels(bal_df["covariate"])
    ax.set_xlabel("Standardized Mean Difference (SMD)")
    ax.set_title("Love Plot — Covariate Balance")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved love plot to {output_path}")
    plt.show()

    return bal_df


def itt_regression(
    df: pd.DataFrame,
    outcome: str = "endline_y",
    treatment: str = "Z",
    strata_col: str = "strata",
    controls: list = None,
) -> dict:
    """
    ITT regression with strata fixed effects and optional covariate controls.

    Args:
        df:          RCT dataset (complete cases for outcome).
        outcome:     Endline outcome column.
        treatment:   Random assignment indicator.
        strata_col:  Randomization strata column (included as dummies).
        controls:    Additional baseline controls for efficiency.

    Returns:
        Dictionary with keys: itt_coef, se, pvalue, ci, late_coef, result.
    """
    dfc = df.dropna(subset=[outcome]).copy()

    # Strata dummies
    strata_dummies = pd.get_dummies(dfc[strata_col], prefix="strata", drop_first=True)
    X_cols = [treatment]
    if controls:
        X_cols += controls

    X = pd.concat([dfc[X_cols], strata_dummies], axis=1)
    X = sm.add_constant(X)

    model = sm.OLS(dfc[outcome], X.astype(float))
    result = model.fit(cov_type="HC3")

    itt = float(result.params[treatment])
    se = float(result.bse[treatment])
    pval = float(result.pvalues[treatment])
    ci_low = float(result.conf_int().loc[treatment, 0])
    ci_high = float(result.conf_int().loc[treatment, 1])

    print(f"\nITT: {itt:.4f}  SE: {se:.4f}  p: {pval:.4f}  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # LATE via IV (using D compliance and Z instrument)
    if "D" in dfc.columns:
        from statsmodels.sandbox.regression.gmm import IV2SLS
        X_iv = sm.add_constant(pd.concat([dfc[["D"] + (controls or [])], strata_dummies], axis=1).astype(float))
        Z_instr = sm.add_constant(pd.concat([dfc[[treatment] + (controls or [])], strata_dummies], axis=1).astype(float))
        iv_model = IV2SLS(dfc[outcome].values, X_iv.values, Z_instr.values)
        iv_result = iv_model.fit()
        late = float(iv_result.params[1])
        print(f"LATE (IV): {late:.4f}")
    else:
        late = None

    return {"itt_coef": itt, "se": se, "pvalue": pval,
            "ci": (ci_low, ci_high), "late_coef": late, "result": result}
```

### Step 2 — Lee Bounds for Selective Attrition

```python
def lee_bounds(
    df: pd.DataFrame,
    outcome: str = "endline_y",
    treatment: str = "Z",
) -> dict:
    """
    Lee (2009) sharp bounds for selective attrition.

    Computes lower and upper bounds on the ATT when attrition is non-random.
    Assumes: attrition in control is random; trims from appropriate tail in treatment.

    Args:
        df:        Dataset with outcome (NaN = attrited) and treatment assignment.
        outcome:   Outcome column (NaN means attrited/missing).
        treatment: Binary treatment assignment.

    Returns:
        Dictionary with keys: p0, p1, trim_fraction, lb_estimate,
                              ub_estimate, lb_se, ub_se.
    """
    treated = df[df[treatment] == 1].copy()
    control = df[df[treatment] == 0].copy()

    p1 = treated[outcome].notna().mean()
    p0 = control[outcome].notna().mean()

    print(f"Retention rate: Treated = {p1:.4f}, Control = {p0:.4f}")

    # Trim fraction from higher-retention group
    trim_fraction = (p1 - p0) / p1 if p1 > p0 else (p0 - p1) / p0

    # Lee bounds: trim from treated group (if p1 > p0)
    y_treated_obs = treated[outcome].dropna().sort_values()
    n_trim = int(np.ceil(trim_fraction * len(y_treated_obs)))

    y_lower = y_treated_obs.iloc[n_trim:].values   # trim from bottom (lower bound)
    y_upper = y_treated_obs.iloc[:len(y_treated_obs) - n_trim].values  # trim from top (upper bound)
    y_control = control[outcome].dropna().values

    lb_estimate = float(np.mean(y_lower) - np.mean(y_control))
    ub_estimate = float(np.mean(y_upper) - np.mean(y_control))

    # Bootstrap SE for bounds
    n_boot = 500
    lb_boots, ub_boots = [], []
    for _ in range(n_boot):
        y_tr_b = np.random.choice(y_treated_obs.values, size=len(y_treated_obs), replace=True)
        y_ctrl_b = np.random.choice(y_control, size=len(y_control), replace=True)
        y_tr_b_sorted = np.sort(y_tr_b)

        lb_boots.append(float(np.mean(y_tr_b_sorted[n_trim:]) - np.mean(y_ctrl_b)))
        ub_boots.append(float(np.mean(y_tr_b_sorted[:max(1, len(y_tr_b_sorted) - n_trim)]) - np.mean(y_ctrl_b)))

    lb_se = float(np.std(lb_boots))
    ub_se = float(np.std(ub_boots))

    print(f"\nLee Sharp Bounds (trim fraction = {trim_fraction:.4f}):")
    print(f"  Lower bound: {lb_estimate:.4f}  (SE = {lb_se:.4f})")
    print(f"  Upper bound: {ub_estimate:.4f}  (SE = {ub_se:.4f})")
    if lb_estimate > 0:
        print("  Both bounds positive — treatment effect significantly positive.")

    return {
        "p0": p0,
        "p1": p1,
        "trim_fraction": trim_fraction,
        "lb_estimate": lb_estimate,
        "ub_estimate": ub_estimate,
        "lb_se": lb_se,
        "ub_se": ub_se,
    }
```

### Step 3 — Propensity Score Matching and FGT Poverty Indices

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def propensity_score_matching(
    df: pd.DataFrame,
    treatment_col: str = "Z",
    outcome_col: str = "endline_y",
    covariate_cols: list = None,
    caliper: float = 0.25,
) -> dict:
    """
    Propensity score matching (nearest neighbor, 1-to-1, with caliper).

    Args:
        df:              Dataset (complete cases).
        treatment_col:   Binary treatment indicator.
        outcome_col:     Outcome variable.
        covariate_cols:  Matching covariates (pre-treatment).
        caliper:         Caliper width = caliper × SD(logit propensity score).

    Returns:
        Dictionary with keys: att, se, n_matched, balance_after.
    """
    dfc = df.dropna(subset=[outcome_col, treatment_col]).copy()
    if covariate_cols is None:
        covariate_cols = ["age", "female", "baseline_y"]

    X = dfc[covariate_cols].values
    T = dfc[treatment_col].values
    Y = dfc[outcome_col].values

    # Estimate propensity score
    lr = LogisticRegression(max_iter=500, random_state=0)
    lr.fit(X, T)
    ps = lr.predict_proba(X)[:, 1]
    logit_ps = np.log(ps / (1 - ps + 1e-10))

    caliper_width = caliper * np.std(logit_ps)
    dfc["ps"] = ps
    dfc["logit_ps"] = logit_ps

    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    # Nearest-neighbor matching with caliper
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(logit_ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))

    matched_pairs = []
    for i, (dist, ctrl_local_idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
        if dist <= caliper_width:
            t_idx = treated_idx[i]
            c_idx = control_idx[ctrl_local_idx]
            matched_pairs.append((t_idx, c_idx))

    n_matched = len(matched_pairs)
    print(f"Matched pairs: {n_matched} / {len(treated_idx)} treated units")

    if n_matched == 0:
        print("Warning: no matches within caliper. Widen caliper.")
        return {"att": np.nan, "se": np.nan, "n_matched": 0, "balance_after": None}

    y_treated_matched = Y[[p[0] for p in matched_pairs]]
    y_control_matched = Y[[p[1] for p in matched_pairs]]
    diffs = y_treated_matched - y_control_matched

    att = float(np.mean(diffs))
    se = float(np.std(diffs) / np.sqrt(n_matched))
    ci = (att - 1.96 * se, att + 1.96 * se)
    print(f"ATT: {att:.4f}  SE: {se:.4f}  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    return {"att": att, "se": se, "n_matched": n_matched, "ci": ci}


def fgt_poverty_indices(
    income: np.ndarray,
    poverty_line: float,
    alphas: list = None,
) -> dict:
    """
    Compute Foster-Greer-Thorbecke (1984) poverty measures.

    Args:
        income:       Array of household income or consumption per capita.
        poverty_line: Absolute poverty threshold (same units as income).
        alphas:       List of FGT alpha parameters. Default: [0, 1, 2].

    Returns:
        Dictionary with keys P0 (headcount), P1 (gap), P2 (severity),
                             poverty_line, n_poor, n_total.
    """
    if alphas is None:
        alphas = [0, 1, 2]
    n = len(income)
    poor = income < poverty_line

    result = {"poverty_line": poverty_line, "n_poor": int(poor.sum()), "n_total": n}
    for alpha in alphas:
        gaps = np.maximum(0, (poverty_line - income) / poverty_line)
        p_alpha = float(np.mean(gaps ** alpha))
        result[f"P{alpha}"] = round(p_alpha, 6)

    print(f"\nFGT Poverty Indices (poverty line = {poverty_line}):")
    print(f"  P0 (headcount ratio):  {result['P0']:.4f}  ({result['P0']*100:.2f}% below line)")
    print(f"  P1 (poverty gap):      {result.get('P1', 'N/A')}")
    print(f"  P2 (poverty severity): {result.get('P2', 'N/A')}")
    print(f"  Poor households: {result['n_poor']} / {n}")
    return result
```

---

## Advanced Usage

### Sensitivity Analysis: FGT Across Poverty Line Values

```python
def fgt_sensitivity(
    income: np.ndarray,
    poverty_lines: np.ndarray = None,
    alpha: int = 0,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Plot FGT P_alpha across a range of poverty lines for robustness.

    Args:
        income:        Household income/consumption array.
        poverty_lines: Array of threshold values. Default: 50th percentile ± 50%.
        alpha:         FGT parameter (0, 1, or 2).
        output_path:   If provided, save sensitivity plot.

    Returns:
        DataFrame with columns: poverty_line, p_alpha.
    """
    if poverty_lines is None:
        median_income = np.median(income)
        poverty_lines = np.linspace(0.5 * median_income, 1.5 * median_income, 50)

    records = []
    for z in poverty_lines:
        gaps = np.maximum(0, (z - income) / z)
        records.append({"poverty_line": z, "p_alpha": float(np.mean(gaps ** alpha))})

    df_sens = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_sens["poverty_line"], df_sens["p_alpha"], color="#8E44AD", linewidth=2)
    ax.set_xlabel("Poverty Line")
    ax.set_ylabel(f"P{alpha}")
    ax.set_title(f"FGT P{alpha} Sensitivity to Poverty Line Choice")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved FGT sensitivity plot to {output_path}")
    plt.show()

    return df_sens
```

---

## Troubleshooting

| Error / Issue | Cause | Resolution |
|---|---|---|
| ITT coefficient imprecise (wide CI) | Small sample or high variance outcome | Use baseline covariate controls to reduce residual variance |
| Lee bounds span zero | Moderate attrition; ambiguous effect direction | Improve study design to reduce attrition; report both bounds |
| PSM few matched pairs | Propensity scores outside common support | Widen caliper; trim extreme propensity scores; check overlap plot |
| PSM ATT biased after matching | Poor covariate balance after matching | Use genetic matching or reweighting (IPW) instead |
| IV2SLS `AttributeError` | Old statsmodels API change | Use `linearmodels.IV2SLS` instead |
| FGT P1/P2 undefined | All income above poverty line | Check poverty line choice; verify income units |

---

## External Resources

- Lee, D.S. (2009). "Training, Wages, and Sample Selection." *Review of Economic Studies*, 76(3), 1071–1102.
- Foster, J., Greer, J., Thorbecke, E. (1984). "A Class of Decomposable Poverty Measures." *Econometrica*, 52(3), 761–766.
- Duflo, E., Glennerster, R., Kremer, M. (2007). "Using Randomization in Development Economics Research."
  *Handbook of Development Economics*, Vol. 4.
- World Bank LSMS data: <https://microdata.worldbank.org/index.php/catalog/lsms>
- J-PAL methods guides: <https://www.povertyactionlab.org/research-resources/research-methods>

---

## Examples

### Example 1 — RCT ITT with Balance Check

```python
df = generate_rct_data(n=1500, n_strata=6, attrition_rate=0.15, true_itt=0.3)

# Balance check at baseline
bal = covariate_balance_table(df, treatment_col="Z",
                               covariates=["age", "female", "baseline_y"],
                               output_path="love_plot.png")

# ITT regression
itt_out = itt_regression(df, outcome="endline_y", treatment="Z",
                          strata_col="strata", controls=["age", "female", "baseline_y"])
print(f"True ITT = 0.3 × 1000 = 300.0  |  Estimated ITT = {itt_out['itt_coef']:.2f}")
```

### Example 2 — Lee Bounds + PSM + Poverty Indices

```python
df = generate_rct_data(n=2000, attrition_rate=0.20, true_itt=0.3)

# Lee bounds for attrition
bounds = lee_bounds(df, outcome="endline_y", treatment="Z")

# PSM on observed sample
psm_out = propensity_score_matching(
    df.dropna(subset=["endline_y"]),
    treatment_col="Z",
    outcome_col="endline_y",
    covariate_cols=["age", "female", "baseline_y"],
    caliper=0.25,
)

# Poverty indices on baseline income distribution
income = df["baseline_y"].values
poverty_line = np.percentile(income, 40)   # 40th percentile as poverty line
fgt = fgt_poverty_indices(income, poverty_line)
fgt_sensitivity(income, alpha=0, output_path="fgt_sensitivity.png")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — ITT, covariate balance, Lee bounds, PSM, FGT poverty |
