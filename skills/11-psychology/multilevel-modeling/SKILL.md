---
name: multilevel-modeling
description: >
  Use this Skill for multilevel/mixed-effects models: random intercepts/slopes,
  ICC, model comparison, ESM data within-person centering, and pymer4 with lme4.
tags:
  - psychology
  - multilevel-modeling
  - mixed-effects
  - ICC
  - random-effects
  - ESM
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
    - pymer4>=0.8
    - statsmodels>=0.14
    - pandas>=1.5
    - numpy>=1.23
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Multilevel Modeling

> **TL;DR** — Fit multilevel/mixed-effects models with pymer4 (R's lme4 via Python).
> Covers random intercept and slope models, ICC computation, likelihood ratio
> model comparison, ESM within-person centering, lmerTest Satterthwaite df,
> emmeans contrasts, and three-level models for experience sampling data.

---

## When to Use

Use this Skill when:

- Data has a nested structure (observations within persons, students within
  schools, trials within participants)
- You need to partition variance into within-person and between-person components
- You are analyzing ESM (experience sampling method) or daily diary data with
  multiple repeated observations per person per day
- You want to test cross-level interactions (e.g., does a person-level trait
  moderate a time-varying predictor?)
- You need proper inference with small cluster sizes (Satterthwaite df via
  lmerTest)
- You want to compare models with different random effect structures

---

## Background

### Level Structure

| Level | Unit | Examples |
|---|---|---|
| Level 1 | Observations (lowest) | Trial, ESM beep, measurement occasion |
| Level 2 | Persons / clusters | Participant, classroom, patient |
| Level 3 | Contexts (optional) | Day, school, site |

### Random Intercept Model

```
Yij = γ00 + u0j + εij
```

Where `u0j ~ N(0, τ00)` is the between-person variance and `εij ~ N(0, σ²)` is
the within-person residual. The ICC (intraclass correlation) is:

```
ICC = τ00 / (τ00 + σ²)
```

ICC tells you what proportion of total variance is due to between-person
differences. ICC > 0.1 generally warrants multilevel modeling.

### Random Slope Model

```
Yij = (γ00 + u0j) + (γ10 + u1j) × Xij + εij
```

Allows each person to have their own regression slope for X. The
covariance between intercept and slope (`τ01`) indicates whether persons
with higher baselines tend to show stronger (or weaker) effects of X.

### Within-Person Centering (ESM)

For ESM, predictors often have both within-person and between-person
components. Person-mean centering separates them:

```
X_within_ij  = Xij − X̄j         (within-person deviation)
X_between_j  = X̄j − X̄..         (between-person deviation from grand mean)
```

This is called "contextual effects" modeling and prevents confounding
of within- and between-person effects.

---

## Environment Setup

```bash
# pymer4 requires R with lme4 installed
# Step 1: Install R (>= 4.0)
# Download from https://cran.r-project.org/

# Step 2: Install R packages
Rscript -e "install.packages(c('lme4', 'lmerTest', 'emmeans'), repos='https://cran.r-project.org')"

# Step 3: Create Python environment
conda create -n mlm_env python=3.11 -y
conda activate mlm_env

# Step 4: Install Python packages
pip install pymer4>=0.8 pandas>=1.5 numpy>=1.23 matplotlib>=3.6 statsmodels>=0.14

# Step 5: Verify
python -c "from pymer4.models import Lmer; print('pymer4 OK')"
```

---

## Core Workflow

### Step 1 — Unconditional Means Model and ICC

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymer4.models import Lmer
from typing import Optional, Dict, Tuple, List

# ── Generate synthetic ESM data for demonstrations ───────────────────────────

def simulate_esm_data(
    n_persons: int = 50,
    n_days: int = 7,
    n_beeps_per_day: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate experience sampling method (ESM) data.

    Structure: beep (level 1) nested in day (level 2) nested in person (level 3).
    Outcome: positive affect (PA, 1–7 scale).
    Predictor: current stress (1–5 scale).

    Args:
        n_persons:        Number of participants.
        n_days:           Days per participant.
        n_beeps_per_day:  Beeps per day.
        seed:             Random seed.

    Returns:
        DataFrame with beep-level observations.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Person-level random effects
    person_intercepts = rng.normal(4.0, 0.8, n_persons)
    person_slopes = rng.normal(-0.4, 0.2, n_persons)

    for p in range(n_persons):
        for d in range(n_days):
            for b in range(n_beeps_per_day):
                stress = rng.uniform(1, 5)
                pa = (
                    person_intercepts[p] +
                    person_slopes[p] * stress +
                    rng.normal(0, 0.5)
                )
                pa = np.clip(pa, 1, 7)
                rows.append({
                    "person_id": p + 1,
                    "day": d + 1,
                    "beep": b + 1,
                    "stress": round(stress, 2),
                    "positive_affect": round(pa, 2),
                    "beep_id": p * n_days * n_beeps_per_day + d * n_beeps_per_day + b + 1,
                })
    return pd.DataFrame(rows)


def compute_icc(
    df: pd.DataFrame,
    outcome: str,
    group: str,
) -> Dict:
    """
    Compute ICC via the unconditional means model.

    The unconditional means model (no predictors) partitions variance into
    between-group and within-group components.

    ICC = τ00 / (τ00 + σ²)

    Args:
        df:      DataFrame with observations.
        outcome: Dependent variable column name.
        group:   Grouping variable column name (e.g., 'person_id').

    Returns:
        Dict with ICC, tau00, sigma2, and model summary.
    """
    formula = f"{outcome} ~ 1 + (1 | {group})"
    model = Lmer(formula, data=df)
    model.fit(summarize=False)

    # Extract variance components
    vc = model.ranef_var
    tau00 = float(vc.loc[vc.index == group, "Var"].values[0])
    sigma2 = float(model.residual_variance)
    icc = tau00 / (tau00 + sigma2)

    result = {
        "ICC": round(icc, 4),
        "tau00": round(tau00, 4),
        "sigma2": round(sigma2, 4),
        "total_variance": round(tau00 + sigma2, 4),
        "group": group,
        "outcome": outcome,
        "n_groups": df[group].nunique(),
        "n_obs": len(df),
    }

    print(
        f"Unconditional means model: {outcome} ~ 1 + (1 | {group})\n"
        f"  ICC = {icc:.3f} ({icc:.1%} of variance between {group})\n"
        f"  τ00 = {tau00:.4f}, σ² = {sigma2:.4f}\n"
        f"  Interpretation: "
        + ("Multilevel modeling warranted (ICC > 0.10)." if icc > 0.10
           else "Low ICC — single-level model may suffice.")
    )
    return result
```

### Step 2 — Random Intercept and Slope Models

```python
def fit_random_slope_model(
    df: pd.DataFrame,
    outcome: str,
    fixed_predictors: List[str],
    random_group: str,
    random_slopes: Optional[List[str]] = None,
    REML: bool = True,
    print_summary: bool = True,
) -> Lmer:
    """
    Fit a random intercept (and optionally random slope) multilevel model.

    Args:
        df:                DataFrame with observations.
        outcome:           Dependent variable.
        fixed_predictors:  List of fixed-effect predictor names.
        random_group:      Grouping variable for random effects (e.g., 'person_id').
        random_slopes:     Predictors with random slopes. None = intercepts only.
        REML:              Use REML estimation (True for variance components,
                           False for fixed-effects comparison via LRT).
        print_summary:     Whether to print model summary.

    Returns:
        Fitted Lmer model object.

    Examples:
        # Random intercept only
        fit_random_slope_model(df, 'pa', ['stress'], 'person_id')
        # Random slope for stress
        fit_random_slope_model(df, 'pa', ['stress'], 'person_id', ['stress'])
    """
    fixed_part = " + ".join(fixed_predictors)

    if random_slopes:
        slopes_str = " + ".join(random_slopes)
        random_part = f"({slopes_str} | {random_group})"
    else:
        random_part = f"(1 | {random_group})"

    formula = f"{outcome} ~ {fixed_part} + {random_part}"
    print(f"Fitting: {formula}")

    model = Lmer(formula, data=df)
    model.fit(REML=REML, summarize=print_summary)
    return model


def model_comparison_lrt(
    model_null: Lmer,
    model_alt: Lmer,
) -> Dict:
    """
    Likelihood ratio test comparing nested multilevel models.

    Both models must be fitted with REML=False (ML estimation) for valid LRT.
    For random effects comparison, refit both with REML=True and use AIC/BIC.

    Args:
        model_null: More restricted (fewer parameters) model.
        model_alt:  More complex (more parameters) model.

    Returns:
        Dict with chi-square, df, p-value, AIC difference, BIC difference.
    """
    from scipy import stats

    # Log-likelihoods
    ll_null = model_null.logLike
    ll_alt = model_alt.logLike
    chi2 = -2 * (ll_null - ll_alt)
    df_diff = model_alt.coefs.shape[0] - model_null.coefs.shape[0]
    if df_diff <= 0:
        df_diff = 1

    p_value = stats.chi2.sf(chi2, df=df_diff)

    aic_null = model_null.AIC
    aic_alt = model_alt.AIC
    bic_null = model_null.BIC
    bic_alt = model_alt.BIC

    result = {
        "chi2": round(chi2, 3),
        "df": df_diff,
        "p_value": round(p_value, 4),
        "AIC_null": round(aic_null, 2),
        "AIC_alt": round(aic_alt, 2),
        "delta_AIC": round(aic_null - aic_alt, 2),
        "BIC_null": round(bic_null, 2),
        "BIC_alt": round(bic_alt, 2),
        "delta_BIC": round(bic_null - bic_alt, 2),
        "preferred": "alternative" if p_value < 0.05 else "null",
    }

    print(
        f"LRT: χ²({df_diff}) = {chi2:.3f}, p = {p_value:.4f}\n"
        f"  ΔAIC = {result['delta_AIC']:.2f}, ΔBIC = {result['delta_BIC']:.2f}\n"
        f"  Preferred model: {result['preferred']}"
    )
    return result
```

### Step 3 — ESM Within-Person Centering

```python
def within_person_center(
    df: pd.DataFrame,
    predictors: List[str],
    person_col: str = "person_id",
    grand_mean_center_between: bool = True,
) -> pd.DataFrame:
    """
    Apply within-person centering to ESM predictors.

    Creates three new columns per predictor:
        {var}_pm:      Person mean (Level-2 predictor)
        {var}_wpc:     Within-person centered (Level-1 predictor = item - pm)
        {var}_gmc:     Grand-mean-centered person mean (between-person effect)

    Args:
        df:                         Trial-level DataFrame.
        predictors:                 Column names to center.
        person_col:                 Participant identifier column.
        grand_mean_center_between:  Whether to grand-mean-center the Level-2 predictor.

    Returns:
        DataFrame with added centered columns.
    """
    df = df.copy()

    for var in predictors:
        # Person mean (Level-2 aggregate)
        person_means = df.groupby(person_col)[var].transform("mean")
        df[f"{var}_pm"] = person_means.round(4)

        # Within-person centering (Level-1 deviation)
        df[f"{var}_wpc"] = (df[var] - person_means).round(4)

        # Grand-mean centering of person mean (between-person effect)
        grand_mean = person_means.mean()
        df[f"{var}_gmc"] = (person_means - grand_mean).round(4)

    print(f"Within-person centering applied to: {predictors}")
    print(f"  New columns: {[f'{v}_wpc' for v in predictors]} (within)")
    print(f"  New columns: {[f'{v}_gmc' for v in predictors]} (between)")
    return df


def plot_esm_within_between(
    df: pd.DataFrame,
    outcome: str,
    predictor_wpc: str,
    predictor_gmc: str,
    person_col: str = "person_id",
    n_sample: int = 10,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot within-person and between-person associations for ESM data.

    Args:
        df:             Centered DataFrame from within_person_center().
        outcome:        Dependent variable column.
        predictor_wpc:  Within-person centered predictor column.
        predictor_gmc:  Grand-mean centered person-mean predictor column.
        person_col:     Participant identifier column.
        n_sample:       Number of persons to show in within-person panel.
        output_path:    Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(42)

    # Within-person panel: person-specific regression lines
    ax1 = axes[0]
    persons = df[person_col].unique()
    sampled = rng.choice(persons, min(n_sample, len(persons)), replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(sampled)))

    for person, color in zip(sampled, colors):
        pdata = df[df[person_col] == person]
        ax1.scatter(pdata[predictor_wpc], pdata[outcome],
                    alpha=0.3, s=15, color=color)
        if len(pdata) >= 3:
            m, b, *_ = np.polyfit(pdata[predictor_wpc], pdata[outcome], 1), None
            x_range = np.linspace(pdata[predictor_wpc].min(), pdata[predictor_wpc].max(), 50)
            coef = np.polyfit(pdata[predictor_wpc], pdata[outcome], 1)
            ax1.plot(x_range, np.polyval(coef, x_range), color=color, linewidth=1.5)

    ax1.set_xlabel(f"{predictor_wpc} (within-person deviation)")
    ax1.set_ylabel(outcome)
    ax1.set_title("Within-Person Association")
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # Between-person panel: person means
    ax2 = axes[1]
    person_means_df = df.groupby(person_col).agg(
        {predictor_gmc: "first", outcome: "mean"}
    ).reset_index()
    ax2.scatter(person_means_df[predictor_gmc], person_means_df[outcome],
                alpha=0.7, s=40, color="steelblue")
    coef2 = np.polyfit(person_means_df[predictor_gmc], person_means_df[outcome], 1)
    x2 = np.linspace(person_means_df[predictor_gmc].min(),
                     person_means_df[predictor_gmc].max(), 100)
    ax2.plot(x2, np.polyval(coef2, x2), color="crimson", linewidth=2)
    ax2.set_xlabel(f"{predictor_gmc} (grand-mean centered person mean)")
    ax2.set_ylabel(f"Mean {outcome}")
    ax2.set_title("Between-Person Association")

    fig.suptitle(f"Within vs. Between-Person Effects: {predictor_wpc} → {outcome}", fontsize=12)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.show()
    return fig
```

---

## Advanced Usage

### Three-Level Model and emmeans Contrasts

```python
def fit_three_level_model(
    df: pd.DataFrame,
    outcome: str,
    level1_predictors: List[str],
    level2_col: str = "day",
    level3_col: str = "person_id",
) -> Lmer:
    """
    Fit a three-level model: observations within days within persons.

    Formula: outcome ~ predictors + (1 | person_id/day)
    This specifies days nested within persons (cross-classified not supported
    in this formula notation — use (1|person_id) + (1|person_id:day) for that).

    Args:
        df:                  DataFrame with observations.
        outcome:             Dependent variable.
        level1_predictors:   Fixed-effect predictors.
        level2_col:          Level-2 grouping (e.g., 'day').
        level3_col:          Level-3 grouping (e.g., 'person_id').

    Returns:
        Fitted Lmer model.
    """
    fixed_part = " + ".join(level1_predictors)
    # Nested structure: days within persons
    formula = (
        f"{outcome} ~ {fixed_part} + "
        f"(1 | {level3_col}/{level2_col})"
    )
    print(f"Three-level model: {formula}")
    model = Lmer(formula, data=df)
    model.fit(REML=True, summarize=True)
    return model


def extract_random_effects_summary(model: Lmer) -> pd.DataFrame:
    """
    Extract and display random effects variance components.

    Args:
        model: Fitted Lmer model object.

    Returns:
        DataFrame with variance components and ICC estimates.
    """
    vc = model.ranef_var.copy()
    total_var = vc["Var"].sum() + model.residual_variance
    vc["ICC_component"] = (vc["Var"] / total_var).round(4)
    vc["SD"] = np.sqrt(vc["Var"]).round(4)

    print("Random effects variance components:")
    print(vc[["Var", "SD", "ICC_component"]].round(4))
    print(f"Residual variance (σ²): {model.residual_variance:.4f}")
    print(f"Total variance: {total_var:.4f}")
    return vc
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `pymer4` model fails to fit | R not found or rpy2 not installed | Check `R_HOME` env var; reinstall rpy2 |
| Singular fit warning | Random effects structure too complex | Simplify to random intercepts only |
| LRT gives negative chi-square | Models not nested | Ensure null model is subset of alternative |
| `ranef_var` KeyError | Version difference in pymer4/lme4 | Use `model.ranef_var` not `model.variance_components` |
| Very small ICC | Design has little clustering | Consider single-level analysis |
| Convergence warning | Many random effects, small N | Use `BOBYQA` optimizer or reduce random structure |
| emmeans not available | emmeans not installed in R | Run `Rscript -e "install.packages('emmeans')"` |

---

## External Resources

- Bliese, P. D. (2000). Within-group agreement, non-independence, and reliability.
  *Multilevel theory, research, and methods in organizations.*
- Hox, J. J. (2010). *Multilevel Analysis: Techniques and Applications* (2nd ed.)
- Bolger, N., & Laurenceau, J.-P. (2013). *Intensive Longitudinal Methods.*
- pymer4 documentation: <https://eshinjolly.com/pymer4/>
- lme4 documentation: <https://github.com/lme4/lme4>
- lmerTest: <https://cran.r-project.org/package=lmerTest>

---

## Examples

### Example 1 — pymer4 Random Slope Model with ICC

```python
# Simulate data
df = simulate_esm_data(n_persons=60, n_days=7, n_beeps_per_day=5, seed=0)

# Step 1: ICC from unconditional means model
icc_result = compute_icc(df, outcome="positive_affect", group="person_id")
print(f"\nICC = {icc_result['ICC']:.3f} — "
      f"{'multilevel warranted' if icc_result['ICC'] > 0.10 else 'single level OK'}")

# Step 2: Within-person centering of stress
df_centered = within_person_center(df, predictors=["stress"], person_col="person_id")

# Step 3: Null model (random intercept, no predictors)
m0 = fit_random_slope_model(
    df_centered, outcome="positive_affect",
    fixed_predictors=["1"],
    random_group="person_id", random_slopes=None, REML=False,
)

# Step 4: Random intercept with within+between stress
m1 = fit_random_slope_model(
    df_centered, outcome="positive_affect",
    fixed_predictors=["stress_wpc", "stress_gmc"],
    random_group="person_id", random_slopes=None, REML=False,
)

# Step 5: Random slope for within-person stress
m2 = fit_random_slope_model(
    df_centered, outcome="positive_affect",
    fixed_predictors=["stress_wpc", "stress_gmc"],
    random_group="person_id", random_slopes=["stress_wpc"], REML=False,
)

# Step 6: Model comparison
print("\nM0 vs M1 (adding stress predictors):")
lrt_01 = model_comparison_lrt(m0, m1)

print("\nM1 vs M2 (adding random slope):")
lrt_12 = model_comparison_lrt(m1, m2)

# Step 7: Visualization
plot_esm_within_between(
    df_centered, outcome="positive_affect",
    predictor_wpc="stress_wpc", predictor_gmc="stress_gmc",
    person_col="person_id", n_sample=12,
    output_path="within_between_stress.png",
)
```

### Example 2 — ESM Within-Person Centering and Three-Level Model

```python
# Center predictors
df = simulate_esm_data(n_persons=40, n_days=5, n_beeps_per_day=4, seed=7)
df_c = within_person_center(df, predictors=["stress"], person_col="person_id")

# Create a combined person-day ID for level-2
df_c["person_day"] = df_c["person_id"].astype(str) + "_" + df_c["day"].astype(str)

# Three-level model
m3 = fit_three_level_model(
    df_c,
    outcome="positive_affect",
    level1_predictors=["stress_wpc"],
    level2_col="day",
    level3_col="person_id",
)

# Variance component summary
vc_df = extract_random_effects_summary(m3)

# Report: print parameter table
print("\nFixed effects summary:")
print(m3.coefs[["Estimate", "SE", "T-stat", "P-val", "Sig"]].round(4))

# Bar chart of variance components
fig, ax = plt.subplots(figsize=(7, 4))
levels = vc_df.index.tolist() + ["Residual"]
variances = vc_df["Var"].tolist() + [m3.residual_variance]
colors = ["#4C72B0", "#DD8452", "#55A868"]
ax.bar(levels, variances, color=colors[:len(levels)])
ax.set_ylabel("Variance")
ax.set_title("Variance Components (Three-Level Model)")
fig.tight_layout()
plt.savefig("variance_components.png", dpi=150)
plt.show()
print("Three-level model complete.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — ICC, random slope models, ESM centering, LRT, three-level models |
