---
name: did-causal
description: >
  Use this Skill when the user needs to estimate causal treatment effects using
  difference-in-differences (DID) designs: two-way fixed effects (TWFE) regression,
  parallel trends pre-testing, Callaway-Sant'Anna staggered adoption estimator,
  and Goodman-Bacon decomposition. Covers both Python (linearmodels) and R (did package).
tags:
  - economics
  - causal-inference
  - difference-in-differences
  - panel-data
  - treatment-effects
  - econometrics
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
    - linearmodels>=4.28
    - pandas>=1.5.0
    - numpy>=1.23.0
    - matplotlib>=3.6.0
    - statsmodels>=0.14.0
  r:
    - did (Callaway-Sant'Anna)
    - bacondecomp (Goodman-Bacon)
    - fixest (fast TWFE with multiple FE)
last_updated: "2026-03-17"
---

# Difference-in-Differences Causal Inference

> **TL;DR** — Estimate causal treatment effects with DID designs: classic 2x2 DID,
> Two-Way Fixed Effects (TWFE), parallel trends testing, staggered adoption with
> Callaway-Sant'Anna, and Goodman-Bacon decomposition.

---

## 1. Overview

### What Problem Does This Skill Solve?

Difference-in-differences is the workhorse of applied microeconometrics. It identifies
causal effects by comparing treated and control units before and after treatment,
under the assumption that untreated units provide a valid counterfactual trend.

Recent methodological advances (2018–2023) show that the canonical TWFE estimator
is **biased under staggered adoption** when treatment effects are heterogeneous across
cohorts. This Skill implements the full modern DID toolkit:

| Method | When to Use |
|---|---|
| Classic 2x2 DID | Single treatment date, clear control group |
| TWFE with unit + time FE | Panel data, homogeneous effects assumed |
| Parallel trends pre-test | Always — validates identifying assumption |
| Callaway-Sant'Anna (C&S) | Staggered rollout, heterogeneous effects |
| Goodman-Bacon decomposition | Diagnose TWFE bias from staggered adoption |

### Key Identifying Assumption

**Parallel trends**: In the absence of treatment, the average outcomes of treated and
control units would have followed parallel paths over time. This is **untestable**
for the post-period but can be assessed using pre-treatment periods as placebo tests.

---

## 2. Environment Setup

```bash
# Python environment
conda create -n did python=3.11 -y
conda activate did
pip install linearmodels pandas numpy matplotlib statsmodels

# R packages (run in R console)
# install.packages(c("did", "bacondecomp", "fixest", "dplyr", "ggplot2"))
```

Verify Python setup:

```python
import linearmodels
import pandas as pd
import numpy as np
print(f"linearmodels version: {linearmodels.__version__}")
```

---

## 3. Core Implementation

### 3.1 Generate Synthetic Panel Data

```python
import numpy as np
import pandas as pd

np.random.seed(42)


def generate_did_panel(
    n_units: int = 200,
    n_periods: int = 10,
    treat_fraction: float = 0.5,
    treatment_period: int = 6,
    true_att: float = 2.0,
    parallel_trends_violation: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a synthetic balanced panel for DID exercises.

    Args:
        n_units:                   Number of units (firms / individuals / regions).
        n_periods:                 Number of time periods (e.g. years).
        treat_fraction:            Fraction of units assigned to treatment.
        treatment_period:          Period when treatment begins (1-indexed).
        true_att:                  True average treatment effect on the treated.
        parallel_trends_violation: Non-zero values introduce a pre-trend for testing.

    Returns:
        DataFrame with columns: unit_id, period, treated, post, y, x1, x2.
    """
    unit_ids = np.arange(1, n_units + 1)
    treated_units = np.random.choice(unit_ids, size=int(n_units * treat_fraction), replace=False)
    treated_set = set(treated_units)

    unit_fe = np.random.normal(0, 1, n_units)
    period_fe = np.linspace(0, 2, n_periods)

    rows = []
    for i, uid in enumerate(unit_ids):
        is_treated = uid in treated_set
        for t in range(1, n_periods + 1):
            post = int(t >= treatment_period)
            treatment_indicator = is_treated and post

            pre_trend = parallel_trends_violation * is_treated * (t - treatment_period) * (1 - post)
            y = (
                unit_fe[i]
                + period_fe[t - 1]
                + true_att * treatment_indicator
                + pre_trend
                + np.random.normal(0, 0.5)
            )

            rows.append({
                "unit_id": uid,
                "period": t,
                "treated": int(is_treated),
                "post": post,
                "did": int(is_treated and post),
                "y": y,
                "x1": np.random.normal(0, 1),
                "x2": np.random.binomial(1, 0.4),
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["unit_id", "period"])
    return df
```

### 3.2 Two-Way Fixed Effects (TWFE)

```python
from linearmodels import PanelOLS
import warnings


def run_twfe_did(
    df: pd.DataFrame,
    outcome: str = "y",
    treatment: str = "did",
    covariates: list = None,
    cluster_by: str = "unit_id",
) -> dict:
    """
    Estimate a TWFE DID model: y_it = alpha_i + lambda_t + beta*D_it + X_it*gamma + e_it.

    Args:
        df:          Panel DataFrame with MultiIndex (unit_id, period).
        outcome:     Name of the outcome variable column.
        treatment:   Name of the treatment indicator column (1 if treated and post).
        covariates:  List of additional control variable names.
        cluster_by:  Level of clustering for standard errors ('unit_id' recommended).

    Returns:
        Dictionary with keys: coef, se, tstat, pvalue, ci_low, ci_high, n_obs, r2.
    """
    covariates = covariates or []
    regressors = [treatment] + covariates
    formula = f"{outcome} ~ {' + '.join(regressors)} + EntityEffects + TimeEffects"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelOLS.from_formula(formula, data=df)
        result = model.fit(cov_type="clustered", cluster_entity=True)

    params = result.params
    tstat = result.tstats
    pval = result.pvalues
    ci = result.conf_int()

    return {
        "coef": float(params[treatment]),
        "se": float(result.std_errors[treatment]),
        "tstat": float(tstat[treatment]),
        "pvalue": float(pval[treatment]),
        "ci_low": float(ci.loc[treatment, "lower"]),
        "ci_high": float(ci.loc[treatment, "upper"]),
        "n_obs": result.nobs,
        "r2": result.rsquared,
        "summary": result.summary,
    }
```

### 3.3 Parallel Trends Pre-Test

```python
import matplotlib.pyplot as plt


def test_parallel_trends(
    df: pd.DataFrame,
    outcome: str = "y",
    treatment_period: int = 6,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Event-study plot for parallel trends pre-test.

    For each period t, estimate the interaction (treated_i x 1[period=t]) to
    check whether treated and control units diverge before treatment. All
    coefficients in pre-periods should be statistically indistinguishable from 0.

    Args:
        df:               Panel DataFrame with MultiIndex (unit_id, period).
        outcome:          Outcome variable name.
        treatment_period: The first treated period (base period will be t-1).
        output_path:      If given, save the event-study plot here.

    Returns:
        DataFrame of event-study coefficients: period, coef, ci_low, ci_high.
    """
    df_reset = df.reset_index()
    periods = sorted(df_reset["period"].unique())

    # Create period dummies interacted with treated
    base_period = treatment_period - 1
    event_dummies = []
    for t in periods:
        if t == base_period:
            continue
        col = f"event_{t}"
        df_reset[col] = (df_reset["treated"] == 1) & (df_reset["period"] == t)
        df_reset[col] = df_reset[col].astype(int)
        event_dummies.append(col)

    df_indexed = df_reset.set_index(["unit_id", "period"])
    formula = f"{outcome} ~ {' + '.join(event_dummies)} + EntityEffects + TimeEffects"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PanelOLS.from_formula(formula, data=df_indexed)
        result = model.fit(cov_type="clustered", cluster_entity=True)

    ci = result.conf_int()
    records = []
    for col in event_dummies:
        t = int(col.replace("event_", ""))
        records.append({
            "period": t,
            "coef": float(result.params[col]),
            "ci_low": float(ci.loc[col, "lower"]),
            "ci_high": float(ci.loc[col, "upper"]),
            "relative_period": t - treatment_period,
        })

    # Add base period at 0
    records.append({
        "period": base_period,
        "coef": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "relative_period": -1,
    })

    es_df = pd.DataFrame(records).sort_values("period").reset_index(drop=True)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(-0.5, color="red", linewidth=1, linestyle=":", label="Treatment onset")
    ax.errorbar(
        es_df["relative_period"],
        es_df["coef"],
        yerr=[es_df["coef"] - es_df["ci_low"], es_df["ci_high"] - es_df["coef"]],
        fmt="o",
        capsize=4,
        color="#4C72B0",
    )
    ax.set_xlabel("Periods relative to treatment")
    ax.set_ylabel(f"Coef on {outcome}")
    ax.set_title("Event Study — Parallel Trends Pre-Test")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved event-study plot to {output_path}")

    return es_df
```

### 3.4 Placebo Check

```python
def run_placebo_check(
    df: pd.DataFrame,
    outcome: str = "y",
    true_treatment_period: int = 6,
    n_placebo_periods: int = 3,
) -> pd.DataFrame:
    """
    Run placebo DID regressions using pre-treatment periods as fake treatment dates.

    All placebo estimates should be statistically insignificant (p > 0.05).
    A significant placebo suggests violations of parallel trends.

    Args:
        df:                    Panel DataFrame with MultiIndex (unit_id, period).
        outcome:               Outcome variable column name.
        true_treatment_period: Actual treatment start period.
        n_placebo_periods:     Number of fake treatment periods to test.

    Returns:
        DataFrame of placebo estimates: placebo_period, coef, pvalue, significant.
    """
    df_reset = df.reset_index()
    pre_periods = sorted(
        [p for p in df_reset["period"].unique() if p < true_treatment_period - 1]
    )
    placebo_periods = pre_periods[-n_placebo_periods:]

    results = []
    for fake_period in placebo_periods:
        df_placebo = df_reset[df_reset["period"] < true_treatment_period].copy()
        df_placebo["did_placebo"] = (
            (df_placebo["treated"] == 1) & (df_placebo["period"] >= fake_period)
        ).astype(int)
        df_indexed = df_placebo.set_index(["unit_id", "period"])

        formula = f"{outcome} ~ did_placebo + EntityEffects + TimeEffects"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PanelOLS.from_formula(formula, data=df_indexed)
            res = model.fit(cov_type="clustered", cluster_entity=True)

        results.append({
            "placebo_period": fake_period,
            "coef": float(res.params["did_placebo"]),
            "pvalue": float(res.pvalues["did_placebo"]),
            "significant": float(res.pvalues["did_placebo"]) < 0.05,
        })

    return pd.DataFrame(results)
```

### 3.5 Callaway-Sant'Anna in R (Staggered Adoption)

For staggered treatment rollout, TWFE is biased. Use the Callaway-Sant'Anna estimator
via the R `did` package.

```r
# Install once: install.packages(c("did", "dplyr", "ggplot2"))
library(did)
library(dplyr)
library(ggplot2)

# ── Data preparation ────────────────────────────────────────────────────
# Required columns:
#   - unit_id   : unit identifier (integer)
#   - period    : time period (integer)
#   - y         : outcome variable
#   - g          : first treatment period (0 = never treated)

df <- read.csv("panel_staggered.csv")

# Units that are never treated must have g = 0
df <- df %>%
  mutate(g = ifelse(is.na(first_treat_period), 0, first_treat_period))

# ── Callaway-Sant'Anna ATT(g,t) ─────────────────────────────────────────
cs_result <- att_gt(
  yname        = "y",
  tname        = "period",
  idname       = "unit_id",
  gname        = "g",
  xformla      = ~ x1 + x2,      # optional covariates
  data         = df,
  control_group = "nevertreated", # "notyettreated" is also valid
  est_method   = "reg",           # regression adjustment
  anticipation = 0,               # periods of anticipation
  clustervars  = "unit_id",
)
summary(cs_result)

# ── Aggregate to overall ATT ─────────────────────────────────────────────
att_overall <- aggte(cs_result, type = "simple")
summary(att_overall)

# ── Event-study aggregation ─────────────────────────────────────────────
att_event <- aggte(cs_result, type = "dynamic", min_e = -4, max_e = 4)
summary(att_event)

# ── Plot event study ────────────────────────────────────────────────────
ggdid(att_event) +
  labs(
    title = "Callaway-Sant'Anna Event Study",
    subtitle = paste0("Overall ATT = ", round(att_overall$overall.att, 3),
                      " (p = ", round(att_overall$overall.se, 3), ")")
  )
```

### 3.6 Goodman-Bacon Decomposition in R

```r
# Diagnose TWFE bias from heterogeneous staggered treatment effects
library(bacondecomp)

bacon_result <- bacon(
  formula  = y ~ treated_post,
  data     = df,
  id_var   = "unit_id",
  time_var = "period",
)

# Print the decomposition table
print(bacon_result)

# Visualize: scatter plot of 2x2 DID weights vs estimates
bacon_df <- as.data.frame(bacon_result)
ggplot(bacon_df, aes(x = weight, y = estimate, color = type)) +
  geom_point(size = 2, alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Goodman-Bacon Decomposition",
    x = "2x2 DID Weight",
    y = "2x2 DID Estimate",
    color = "Comparison type",
  ) +
  theme_minimal()
```

---

## 4. End-to-End Examples

### Example 1 — Classic DID (Python)

```python
# Step 1: generate data with a true ATT of 2.0
df = generate_did_panel(
    n_units=300,
    n_periods=12,
    treat_fraction=0.5,
    treatment_period=7,
    true_att=2.0,
)

# Step 2: pre-test parallel trends
es_df = test_parallel_trends(
    df,
    outcome="y",
    treatment_period=7,
    output_path="event_study.png",
)
pre_period_pvals = es_df[es_df["relative_period"] < 0]["ci_low"]
print("Pre-trend coefficients (should overlap zero):")
print(es_df[es_df["relative_period"] < 0][["relative_period", "coef", "ci_low", "ci_high"]])

# Step 3: TWFE estimate
twfe = run_twfe_did(df, outcome="y", treatment="did")
print(f"\nTWFE ATT estimate: {twfe['coef']:.3f} (95% CI: [{twfe['ci_low']:.3f}, {twfe['ci_high']:.3f}])")
print(f"True ATT: 2.000  |  p-value: {twfe['pvalue']:.4f}")

# Step 4: placebo check
placebo_df = run_placebo_check(df, outcome="y", true_treatment_period=7)
print("\nPlacebo checks (all p-values should be > 0.05):")
print(placebo_df.to_string(index=False))
```

### Example 2 — Staggered Adoption Warning

```python
# Generate staggered treatment panel
np.random.seed(1)
n_units = 400
periods = np.arange(1, 13)
treat_cohorts = {0: None, 4: "early", 7: "mid", 10: "late"}  # 0 = never treated

rows = []
for uid in range(1, n_units + 1):
    cohort_key = np.random.choice(list(treat_cohorts.keys()), p=[0.25, 0.25, 0.25, 0.25])
    first_treat = cohort_key if cohort_key > 0 else 999

    for t in periods:
        post = int(t >= first_treat)
        # Heterogeneous ATT: early cohort gets larger effect
        att = 3.0 if cohort_key == 4 else 1.5 if cohort_key == 7 else 0.8
        y = uid * 0.01 + t * 0.5 + att * post + np.random.normal(0, 0.5)
        rows.append({"unit_id": uid, "period": t, "y": y,
                     "treated_post": post, "g": cohort_key if cohort_key > 0 else 0})

df_stag = pd.DataFrame(rows).set_index(["unit_id", "period"])

# TWFE on staggered data -- will be biased!
twfe_stag = run_twfe_did(df_stag, outcome="y", treatment="treated_post")
print(f"TWFE on staggered data: {twfe_stag['coef']:.3f}  (biased -- use C&S instead!)")

# Correct approach: use Callaway-Sant'Anna in R (see Section 3.5 above)
print("Export to CSV and run the R code in Section 3.5 for unbiased estimates.")
df_stag.reset_index().to_csv("panel_staggered.csv", index=False)
```

---

## 5. Common Errors and Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Singleton groups found in the data` | Unit with only 1 observation | Drop singletons or use `drop_absorbed=True` in fixest |
| `ValueError: Panel is not balanced` | Missing unit-period observations | Impute or use unbalanced panel estimator |
| TWFE ATT far from truth on staggered data | Heterogeneous treatment effects | Switch to Callaway-Sant'Anna or stacked DID |
| Significant pre-trends in event study | Parallel trends violation | Check for confounders; consider synthetic control |
| R `att_gt()` returns `NA` for some cohorts | Too few never-treated units | Use `control_group="notyettreated"` |
| `CollinearityWarning` in linearmodels | Redundant fixed effects or perfect multicollinearity | Drop redundant columns; check for time-invariant regressors |

---

## 6. Methodological Notes

**Why TWFE fails under staggered adoption**: In a staggered design, early-treated units
act as controls for later-treated units. If treatment effects grow over time (dynamic
effects), already-treated units provide a bad counterfactual, creating negative weights
in the TWFE decomposition and biasing the aggregate estimate.

**Callaway-Sant'Anna solution**: Estimates ATT(g,t) — the average treatment effect for
cohort g at time t — using only clean comparisons (never-treated or not-yet-treated
units as controls). Aggregates to overall ATT or event-study coefficients without
imposing effect homogeneity.

**Rule of thumb**: Use TWFE when treatment timing is uniform. Use C&S whenever there
is staggered rollout and you suspect heterogeneous effects.

---

## 7. References and Further Reading

- Callaway, B. and Sant'Anna, P. (2021). "Difference-in-Differences with Multiple Time Periods."
  *Journal of Econometrics*, 225(2), 200–230. <https://doi.org/10.1016/j.jeconom.2020.12.001>
- Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing."
  *Journal of Econometrics*, 225(2), 254–277. <https://doi.org/10.1016/j.jeconom.2021.03.014>
- Roth, J., Sant'Anna, P., Bilinski, A., Poe, J. (2023). "What's Trending in Difference-in-Differences?"
  *Journal of Econometrics*, 235(2), 2218–2244.
- Baker, A.C. et al. (2022). "How Much Should We Trust Staggered Difference-in-Differences Estimates?"
  *Journal of Financial Economics*, 144(2), 370–395.
- `did` R package documentation: <https://bcallaway11.github.io/did/>
- `linearmodels` Python docs: <https://bashtage.github.io/linearmodels/>

---

## 8. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — TWFE, parallel trends test, placebo, C&S (R), Bacon decomp |
