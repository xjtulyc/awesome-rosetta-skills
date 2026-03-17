---
name: survival-analysis
description: >
  Use this Skill for survival analysis: Kaplan-Meier estimator, log-rank test, Cox PH
  model (Schoenfeld residuals), AFT models, competing risks (Fine-Gray), and
  time-varying covariates with lifelines.
tags:
  - mathematics
  - survival-analysis
  - Kaplan-Meier
  - Cox-regression
  - competing-risks
  - lifelines
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
    - lifelines>=0.27
    - scikit-survival>=0.21
    - pandas>=1.5
    - matplotlib>=3.6
    - numpy>=1.23
last_updated: "2026-03-17"
status: stable
---

# Survival Analysis

> **TL;DR** — Estimate time-to-event models: Kaplan-Meier curves with log-rank test,
> Cox PH model with proportional-hazards diagnostics (Schoenfeld residuals),
> Weibull AFT, and competing risks (Aalen-Johansen / Fine-Gray) using lifelines.

---

## When to Use

Use this Skill when you need to:

- Estimate and plot Kaplan-Meier survival curves for one or more groups.
- Compare group survival with the log-rank test or Wilcoxon test.
- Fit a Cox proportional hazards model and interpret hazard ratios.
- Check the PH assumption with Schoenfeld residuals and log-log plots.
- Model time to failure with a parametric AFT model (Weibull, Log-Normal, Log-Logistic).
- Handle competing events with cause-specific cumulative incidence functions (CIF).
- Include time-varying covariates using long-format data.

| Task | lifelines class |
|---|---|
| Non-parametric survival | `KaplanMeierFitter` |
| Cumulative hazard | `NelsonAalenFitter` |
| Cox PH | `CoxPHFitter` |
| Weibull AFT | `WeibullAFTFitter` |
| Competing risks CIF | `AalenJohansenFitter` |
| Parametric hazard | `WeibullFitter`, `LogNormalFitter` |

---

## Background & Key Concepts

### Censoring

- **Right censoring** (most common): event not observed before study end.
- **Left censoring**: event known to have occurred before observation started.
- **Interval censoring**: event occurred in a known interval but exact time unknown.
- Standard lifelines functions handle right censoring; interval censoring requires
  the `interval_censoring_mode=True` argument in some fitters.

### Kaplan-Meier Estimator

Non-parametric maximum likelihood estimator of the survival function:

```
S(t) = prod_{t_i <= t} (1 - d_i / n_i)
```

where `d_i` = events at time `t_i`, `n_i` = at-risk count just before `t_i`.

### Cox Proportional Hazards Model

```
h(t | X) = h_0(t) * exp(beta^T X)
```

- `h_0(t)` is an unspecified baseline hazard (semi-parametric).
- `exp(beta_j)` is the hazard ratio for a one-unit increase in `X_j`.
- **PH assumption**: covariate effect is constant over time (test with Schoenfeld residuals).

### Competing Risks

When multiple event types can occur, the cause-specific CIF is estimated with
the Aalen-Johansen estimator. Regression is via the Fine-Gray subdistribution hazard
model, available through `lifelines` or `scikit-survival`.

---

## Environment Setup

```bash
conda create -n survival python=3.11 -y
conda activate survival
pip install lifelines>=0.27 scikit-survival>=0.21 \
            pandas>=1.5 matplotlib>=3.6 numpy>=1.23

python -c "import lifelines; print('lifelines', lifelines.__version__)"
```

---

## Core Workflow

### Step 1 — Kaplan-Meier Curves and Log-Rank Test

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

np.random.seed(42)

# ── Simulate two-arm clinical trial data ────────────────────────────────────────
def simulate_survival_data(
    n: int = 200,
    treatment_effect: float = 0.5,   # hazard ratio (< 1 = treatment reduces hazard)
    max_follow_up: float = 60.0,     # months
    censoring_rate: float = 0.3,
) -> pd.DataFrame:
    """
    Simulate exponential survival times for control and treatment arms.

    Args:
        n:                 Total number of subjects (split equally between arms).
        treatment_effect:  Hazard ratio for treatment vs control.
        max_follow_up:     Maximum follow-up time in months.
        censoring_rate:    Probability of random censoring.

    Returns:
        DataFrame with columns: subject_id, arm, duration, event_observed.
    """
    half = n // 2
    records = []

    for arm_label, hazard in [("Control", 0.05), ("Treatment", 0.05 * treatment_effect)]:
        n_arm = half
        true_times = np.random.exponential(1 / hazard, n_arm)
        censor_mask = np.random.rand(n_arm) < censoring_rate
        censor_times = np.random.uniform(1, max_follow_up, n_arm)

        observed_times = np.where(censor_mask, np.minimum(true_times, censor_times), true_times)
        observed_times = np.minimum(observed_times, max_follow_up)
        event_observed = (~censor_mask) & (true_times <= max_follow_up)

        for i in range(n_arm):
            records.append({
                "subject_id"    : len(records),
                "arm"           : arm_label,
                "duration"      : float(observed_times[i]),
                "event_observed": bool(event_observed[i]),
                "age"           : np.random.randint(30, 75),
                "stage"         : np.random.choice(["I", "II", "III"], p=[0.3, 0.4, 0.3]),
            })

    return pd.DataFrame(records)


df = simulate_survival_data(n=400, treatment_effect=0.55)
print(f"Dataset shape : {df.shape}")
print(df["arm"].value_counts())
print(f"Event rate    : {df['event_observed'].mean():.2%}")

# ── Kaplan-Meier fitter ─────────────────────────────────────────────────────────
kmf_ctrl = KaplanMeierFitter(label="Control")
kmf_trt  = KaplanMeierFitter(label="Treatment")

ctrl = df[df["arm"] == "Control"]
trt  = df[df["arm"] == "Treatment"]

kmf_ctrl.fit(ctrl["duration"], ctrl["event_observed"])
kmf_trt.fit(trt["duration"],  trt["event_observed"])

print(f"\nMedian survival (Control)  : {kmf_ctrl.median_survival_time_:.1f} months")
print(f"Median survival (Treatment): {kmf_trt.median_survival_time_:.1f} months")

# ── Log-rank test ───────────────────────────────────────────────────────────────
lr = logrank_test(
    ctrl["duration"], trt["duration"],
    ctrl["event_observed"], trt["event_observed"],
)
print(f"\nLog-rank test: chi2={lr.test_statistic:.4f}  p={lr.p_value:.6f}")
print(f"Reject H0 (equal survival): {lr.p_value < 0.05}")

# ── KM plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
kmf_ctrl.plot_survival_function(ax=ax, ci_show=True)
kmf_trt.plot_survival_function(ax=ax,  ci_show=True)
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival probability")
ax.set_title(f"Kaplan-Meier Curves\n(Log-rank p = {lr.p_value:.4f})")
ax.set_ylim(0, 1)
ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="Median (50%)")
ax.legend()
fig.tight_layout()
fig.savefig("km_curves.png", dpi=150)
print("Saved km_curves.png")

# ── Nelson-Aalen cumulative hazard ─────────────────────────────────────────────
naf = NelsonAalenFitter(label="Control")
naf.fit(ctrl["duration"], ctrl["event_observed"])
fig2, ax2 = plt.subplots(figsize=(7, 4))
naf.plot_cumulative_hazard(ax=ax2)
ax2.set_title("Nelson-Aalen Cumulative Hazard — Control Arm")
fig2.tight_layout()
fig2.savefig("nelson_aalen.png", dpi=150)
```

### Step 2 — Cox PH Model and Schoenfeld Residual Test

```python
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

# ── Prepare data: encode categorical variables ──────────────────────────────────
df_model = df.copy()
df_model["stage_II"]  = (df_model["stage"] == "II").astype(int)
df_model["stage_III"] = (df_model["stage"] == "III").astype(int)
df_model["arm_bin"]   = (df_model["arm"] == "Treatment").astype(int)

covariates = ["arm_bin", "age", "stage_II", "stage_III"]

# ── Fit Cox PH ───────────────────────────────────────────────────────────────────
cph = CoxPHFitter(penalizer=0.01)
cph.fit(
    df_model[covariates + ["duration", "event_observed"]],
    duration_col="duration",
    event_col="event_observed",
)
cph.print_summary(decimals=4)

# ── Hazard ratios with 95% CI ────────────────────────────────────────────────────
hr_df = pd.DataFrame({
    "HR"    : np.exp(cph.params_),
    "CI_low": np.exp(cph.confidence_intervals_["95% lower-bound"]),
    "CI_hi" : np.exp(cph.confidence_intervals_["95% upper-bound"]),
    "p"     : cph.summary["p"],
})
print("\nHazard Ratios:")
print(hr_df.round(4))

# ── Proportional hazard assumption test (Schoenfeld residuals) ───────────────────
ph_test = proportional_hazard_test(cph, df_model[covariates + ["duration", "event_observed"]],
                                   time_transform="rank")
print("\nSchoenfeld PH test:")
print(ph_test.summary[["test_statistic", "p", "conclusion"]].to_string())

# ── Forest plot ─────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 4))
cph.plot(ax=ax3)   # HR forest plot with CI
ax3.set_title("Cox PH Hazard Ratios (forest plot)")
ax3.axvline(0, color="grey", linestyle="--")
fig3.tight_layout()
fig3.savefig("cox_forest_plot.png", dpi=150)
print("Saved cox_forest_plot.png")

# ── Baseline survival prediction for specific profiles ──────────────────────────
profiles = pd.DataFrame({
    "arm_bin"  : [0, 1],
    "age"      : [55, 55],
    "stage_II" : [0, 0],
    "stage_III": [0, 0],
})

predicted_survival = cph.predict_survival_function(profiles)
fig4, ax4 = plt.subplots(figsize=(7, 4))
for col in predicted_survival.columns:
    label = "Control, age 55, stage I" if col == 0 else "Treatment, age 55, stage I"
    ax4.plot(predicted_survival.index, predicted_survival[col], label=label)
ax4.set_xlabel("Time (months)")
ax4.set_ylabel("Survival probability")
ax4.set_title("Cox PH Predicted Survival Functions")
ax4.legend()
fig4.tight_layout()
fig4.savefig("cox_predicted_survival.png", dpi=150)
```

### Step 3 — Competing Risks with Aalen-Johansen

```python
from lifelines import AalenJohansenFitter

# ── Simulate competing risks: cancer-specific death (event=1) vs other causes (event=2)
np.random.seed(7)
n_cr = 500
arm_cr = np.random.choice([0, 1], n_cr)

# Exponential times for each cause and censoring
lambda_cancer = 0.04 - 0.02 * arm_cr   # treatment halves cancer hazard
lambda_other  = np.full(n_cr, 0.02)
lambda_censor = np.full(n_cr, 0.03)

t_cancer = np.random.exponential(1 / lambda_cancer)
t_other  = np.random.exponential(1 / lambda_other)
t_censor = np.random.exponential(1 / lambda_censor)

observed_time = np.minimum.reduce([t_cancer, t_other, t_censor])
event_type = np.where(
    observed_time == t_cancer, 1,      # cancer death
    np.where(observed_time == t_other, 2, 0)   # other cause or censored
)

df_cr = pd.DataFrame({
    "duration"  : observed_time,
    "event_type": event_type,
    "arm"       : arm_cr,
})
print("\nCompeting risks event distribution:")
print(df_cr["event_type"].value_counts().rename({0: "Censored", 1: "Cancer", 2: "Other"}))

# ── Aalen-Johansen CIF estimator ─────────────────────────────────────────────────
# Cause 1: cancer-specific CIF by arm
fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))

for arm_val, arm_name, ax in zip([0, 1], ["Control", "Treatment"], axes5):
    mask = df_cr["arm"] == arm_val
    sub  = df_cr[mask]

    ajf = AalenJohansenFitter(calculate_variance=True)
    ajf.fit(sub["duration"], sub["event_type"], event_of_interest=1)

    ajf.plot_cumulative_density(ax=ax, label=f"{arm_name} (n={mask.sum()})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Incidence")
    ax.set_title(f"Cancer-specific CIF — {arm_name}")
    ax.set_ylim(0, 0.6)

fig5.suptitle("Competing Risks: Aalen-Johansen CIF", y=1.02)
fig5.tight_layout()
fig5.savefig("competing_risks_cif.png", dpi=150)
print("Saved competing_risks_cif.png")

# ── Gray's test for difference in CIF between arms ──────────────────────────────
from lifelines.statistics import aalen_johansen_multivariate_test

result_gray = aalen_johansen_multivariate_test(
    df_cr["duration"], df_cr["arm"], df_cr["event_type"], event_of_interest=1
)
print(f"\nGray's test for CIF equality: "
      f"stat={result_gray.test_statistic:.4f}  p={result_gray.p_value:.4f}")
```

---

## Advanced Usage

### Weibull AFT Model

```python
from lifelines import WeibullAFTFitter

waf = WeibullAFTFitter()
waf.fit(
    df_model[covariates + ["duration", "event_observed"]],
    duration_col="duration",
    event_col="event_observed",
)
waf.print_summary(decimals=4)

# AFT interpretation: exp(coef) is the time ratio (TR > 1 means longer survival)
print("\nTime Ratios (Weibull AFT):")
print(np.exp(waf.params_[waf.params_.index.get_level_values(0) == "lambda_"]).round(4))
```

### Time-Varying Covariates

```python
# Time-varying covariates require a long-format (start, stop) dataset.
# Example: a biomarker measured at each visit that changes over time.

# Convert wide-format to long-format with counting process notation
def to_counting_process(df_wide: pd.DataFrame,
                         id_col: str = "subject_id",
                         duration_col: str = "duration",
                         event_col: str = "event_observed",
                         baseline_covariate: str = "age",
                         time_covariate_col: str = "biomarker") -> pd.DataFrame:
    """
    Create a simple two-interval counting process data from wide format.
    In practice, visit data drives the interval boundaries.

    Returns:
        Long-format DataFrame with columns: id, start, stop, event, covariates.
    """
    rows = []
    for _, row in df_wide.iterrows():
        mid = row[duration_col] / 2
        # Interval 1: [0, mid) — pre-visit biomarker
        rows.append({
            id_col        : row[id_col],
            "start"       : 0.0,
            "stop"        : mid,
            event_col     : False,
            baseline_covariate: row[baseline_covariate],
            time_covariate_col: np.random.normal(5, 1),   # baseline value
        })
        # Interval 2: [mid, T] — post-visit biomarker
        rows.append({
            id_col        : row[id_col],
            "start"       : mid,
            "stop"        : row[duration_col],
            event_col     : row[event_col],
            baseline_covariate: row[baseline_covariate],
            time_covariate_col: np.random.normal(6, 1.2), # follow-up value
        })
    return pd.DataFrame(rows)


df_long = to_counting_process(df[["subject_id", "duration", "event_observed", "age"]])

cph_tv = CoxPHFitter()
cph_tv.fit(
    df_long[["start", "stop", "event_observed", "age", "biomarker"]],
    duration_col="stop",
    event_col="event_observed",
    entry_col="start",
)
cph_tv.print_summary(decimals=4)
```

---

## Troubleshooting

| Error / Symptom | Cause | Fix |
|---|---|---|
| `ConvergenceWarning: Newton-Raphson did not converge` | Near-singular information matrix | Add `penalizer=0.1` to `CoxPHFitter`; check for collinearity |
| Flat KM curve (S(t) = 1) | No events in group | Check event column encoding (True/False or 1/0); verify event is not all-censored |
| Schoenfeld test p < 0.05 for a covariate | PH assumption violated | Stratify on that covariate or include a time-interaction term |
| `AalenJohansenFitter` returns `NaN` CIF | All events are of a single type | Ensure at least one event of each competing type exists in the subset |
| `WeibullAFTFitter` log-likelihood = -inf | Negative duration values | Filter out `duration <= 0` rows |
| Log-rank test p-value incorrect | Tied event times not handled | Use Wilcoxon weight by passing `weightings='wilcoxon'` to `logrank_test` |

---

## External Resources

- lifelines documentation: <https://lifelines.readthedocs.io/>
- scikit-survival documentation: <https://scikit-survival.readthedocs.io/>
- Klein & Moeschberger, *Survival Analysis: Techniques for Censored and Truncated Data*
- Therneau & Grambsch, *Modeling Survival Data: Extending the Cox Model*
- Fine & Gray (1999), Competing risks model: <https://doi.org/10.2307/2291498>
- Aalen-Johansen estimator reference: <https://doi.org/10.1111/j.1467-9469.2008.00613.x>

---

## Examples

### Example 1 — KM Forest Plot by Stage

```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

stage_colors = {"I": "#2ecc71", "II": "#f39c12", "III": "#e74c3c"}
fig, ax = plt.subplots(figsize=(9, 5))

for stage_val, color in stage_colors.items():
    subset = df[df["stage"] == stage_val]
    kmf_s  = KaplanMeierFitter(label=f"Stage {stage_val} (n={len(subset)})")
    kmf_s.fit(subset["duration"], subset["event_observed"])
    kmf_s.plot_survival_function(ax=ax, ci_show=True, color=color)

ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival probability")
ax.set_title("Kaplan-Meier by Disease Stage")
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("km_by_stage.png", dpi=150)
print("Saved km_by_stage.png")

# Multi-group log-rank test
mlr = multivariate_logrank_test(df["duration"], df["stage"], df["event_observed"])
print(f"Multi-group log-rank: p = {mlr.p_value:.4f}")
```

### Example 2 — Sensitivity Check: Breslow vs Efron Tie Handling

```python
from lifelines import CoxPHFitter
import pandas as pd

# lifelines uses Breslow tie handling by default; switch to Efron for fewer ties
for tie_method in ["breslow", "efron"]:
    cph_tie = CoxPHFitter(baseline_estimation_method=tie_method)
    cph_tie.fit(
        df_model[["arm_bin", "age", "duration", "event_observed"]],
        duration_col="duration",
        event_col="event_observed",
    )
    print(f"{tie_method}: arm_bin HR = {np.exp(cph_tie.params_['arm_bin']):.4f}, "
          f"p = {cph_tie.summary.loc['arm_bin', 'p']:.4f}")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — KM, log-rank, Cox PH, AFT, competing risks, time-varying covariates |
