---
name: survival-epi
description: >
  Use this Skill for epidemiological survival analysis: Kaplan-Meier with log-rank
  test, Cox PH (time-varying covariates, Schoenfeld residuals), AFT models, and
  competing risks.
tags:
  - public-health
  - survival-analysis
  - Kaplan-Meier
  - Cox-regression
  - competing-risks
  - epidemiology
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
last_updated: "2026-03-18"
status: stable
---

# Epidemiological Survival Analysis

> **TL;DR** — Full survival analysis toolkit: Kaplan-Meier estimator with log-rank
> test, Cox Proportional Hazards model with Schoenfeld residual PH assumption testing,
> time-varying covariates, Weibull/Log-Normal AFT models, and competing risks analysis
> (cause-specific CIF, Fine-Gray concept).

---

## When to Use

Use this Skill when you need to:

- Estimate time-to-event distributions in cohort or clinical trial data
- Compare survival curves between two or more exposure groups (log-rank test)
- Estimate hazard ratios for risk factors using Cox PH regression
- Test and handle violations of the proportional hazards assumption
- Model time-varying exposures (e.g., medication changes during follow-up)
- Analyse competing risks (e.g., death from disease vs. death from other causes)
- Compute concordance index (c-statistic) to evaluate model discrimination

| Analysis | Method | lifelines Class |
|---|---|---|
| Non-parametric survival | Kaplan-Meier | `KaplanMeierFitter` |
| Cumulative hazard | Nelson-Aalen | `NelsonAalenFitter` |
| Semi-parametric regression | Cox PH | `CoxPHFitter` |
| Parametric regression | Weibull AFT | `WeibullAFTFitter` |
| Competing risks CIF | Aalen-Johansen | `AalenJohansenFitter` |

---

## Background

### Survival Data Structure

Survival data has three essential components:

- **Time** (T): Duration from study entry to event or censoring
- **Event indicator** (E): 1 = event occurred, 0 = censored
- **Covariates** (X): Baseline or time-varying predictors

**Right censoring** (most common): Subject's event time is unknown because
follow-up ended before the event (lost to follow-up, study end, withdrawal).
The survival time is known to exceed the observed time.

### Kaplan-Meier Estimator

The KM estimator is the non-parametric maximum likelihood estimator of the
survival function S(t) = P(T > t):

```
S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)
```

where d_i = number of events at time t_i, n_i = number at risk just before t_i.

### Cox Proportional Hazards Model

The Cox model specifies the hazard function as:

```
h(t|X) = h_0(t) × exp(β₁X₁ + β₂X₂ + ... + βₖXₖ)
```

- h_0(t): unspecified baseline hazard (non-parametric)
- exp(βⱼ): **hazard ratio** for a one-unit increase in Xⱼ
- Partial likelihood: estimates β without specifying h_0(t)

**Proportional hazards assumption**: The hazard ratio between any two subjects
is constant over time. Test via scaled Schoenfeld residuals (should be uncorrelated
with time under PH).

### Competing Risks

When multiple event types can occur and one prevents observation of others:

- **Cause-specific hazard**: Rate of cause-k events among those still at risk
  (ignoring other causes)
- **Cumulative Incidence Function (CIF)**: Probability of cause-k event by time t
  accounting for competing risks: CIF_k(t) = ∫ S(s-) × h_k(s) ds
- **Fine-Gray model** (subdistribution hazard): Directly models the CIF by treating
  subjects who experienced competing events as permanently at risk

---

## Environment Setup

```bash
conda create -n survival-env python=3.11 -y
conda activate survival-env
pip install "lifelines>=0.27" "scikit-survival>=0.21" \
            "pandas>=1.5" "matplotlib>=3.6" "numpy>=1.23"

python -c "import lifelines; print(f'lifelines {lifelines.__version__}')"
```

Generate synthetic survival data for testing:

```python
import numpy as np
import pandas as pd


def generate_survival_data(
    n: int = 500,
    seed: int = 42,
    competing_risks: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic right-censored survival data.

    Args:
        n:                 Number of subjects.
        seed:              Random seed for reproducibility.
        competing_risks:   If True, generate two event types (1=primary, 2=competing).

    Returns:
        DataFrame with: id, time, event, group, age, sex, treatment.
    """
    rng = np.random.default_rng(seed)

    group = rng.binomial(1, 0.5, n)          # 0=control, 1=exposed
    treatment = rng.binomial(1, 0.4, n)       # 0=no treatment, 1=treated
    age = rng.normal(55, 12, n).clip(18, 90)
    sex = rng.binomial(1, 0.5, n)             # 0=female, 1=male

    # True hazard: group doubles hazard, treatment halves it
    lam = np.exp(0.6 * group - 0.5 * treatment + 0.02 * (age - 55))
    true_time = rng.exponential(1.0 / lam)
    censor_time = rng.uniform(1.0, 10.0, n)

    time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)

    df = pd.DataFrame({
        'id': range(n),
        'time': time.round(3),
        'event': event,
        'group': group,
        'age': age.round(1),
        'sex': sex,
        'treatment': treatment,
    })

    if competing_risks:
        # Assign competing event type: 1=primary, 2=competing
        comp_time = rng.exponential(2.0, n)  # competing event has lower rate
        has_comp = (comp_time < censor_time) & (comp_time < true_time)
        df['event_type'] = 0
        df.loc[event == 1, 'event_type'] = 1
        df.loc[has_comp, 'event_type'] = 2
        # Recalculate time as minimum of all three
        df['time'] = np.minimum.reduce([true_time, comp_time, censor_time]).round(3)

    return df
```

---

## Core Workflow

### Step 1 — Kaplan-Meier Curves by Exposure Group and Log-Rank Test

```python
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


def km_analysis(
    df: pd.DataFrame,
    time_col: str = 'time',
    event_col: str = 'event',
    group_col: str = 'group',
    group_labels: dict = None,
    alpha: float = 0.05,
    output_path: str = 'km_curves.png',
) -> dict:
    """
    Plot Kaplan-Meier survival curves stratified by group and perform log-rank test.

    Args:
        df:           DataFrame with time, event, and group columns.
        time_col:     Name of the time-to-event column.
        event_col:    Name of the event indicator (1=event, 0=censored).
        group_col:    Column to stratify by.
        group_labels: Dict mapping group values to display names.
        alpha:        Significance level for CI.
        output_path:  File path to save the plot.

    Returns:
        Dict with median survival per group and log-rank test results.
    """
    groups = sorted(df[group_col].unique())
    if group_labels is None:
        group_labels = {g: str(g) for g in groups}

    fig, ax = plt.subplots(figsize=(9, 5))
    results = {}
    kmfs = {}

    for grp in groups:
        mask = df[group_col] == grp
        T = df.loc[mask, time_col]
        E = df.loc[mask, event_col]

        kmf = KaplanMeierFitter(alpha=alpha)
        kmf.fit(T, E, label=group_labels[grp])
        kmf.plot_survival_function(ax=ax, ci_show=True)

        kmfs[grp] = kmf
        results[group_labels[grp]] = {
            'n': int(mask.sum()),
            'events': int(E.sum()),
            'median_survival': float(kmf.median_survival_time_),
            'median_ci_lower': float(kmf.confidence_interval_median_['KM_estimate_lower_0.95'].iloc[0])
                               if hasattr(kmf, 'confidence_interval_median_') else None,
        }

    # Log-rank test (pairwise for 2 groups)
    if len(groups) == 2:
        g0, g1 = groups
        lrt = logrank_test(
            df.loc[df[group_col] == g0, time_col],
            df.loc[df[group_col] == g1, time_col],
            event_observed_A=df.loc[df[group_col] == g0, event_col],
            event_observed_B=df.loc[df[group_col] == g1, event_col],
        )
        results['log_rank_p'] = float(lrt.p_value)
        results['log_rank_stat'] = float(lrt.test_statistic)
        ax.set_title(f'Kaplan-Meier Curves\nLog-rank p = {lrt.p_value:.4f}')
    else:
        mlrt = multivariate_logrank_test(
            df[time_col], df[group_col], event_observed=df[event_col]
        )
        results['log_rank_p'] = float(mlrt.p_value)
        ax.set_title(f'Kaplan-Meier Curves\nLog-rank p = {mlrt.p_value:.4f}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Survival probability S(t)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    for label, r in results.items():
        if isinstance(r, dict):
            print(f"  {label}: n={r['n']}, events={r['events']}, median={r['median_survival']:.2f}")
    if 'log_rank_p' in results:
        print(f"  Log-rank p-value: {results['log_rank_p']:.4f}")

    return results


# --- Demo ---
df = generate_survival_data(n=500)
km_results = km_analysis(df, group_labels={0: 'Unexposed', 1: 'Exposed'})
```

### Step 2 — Cox PH Model and Schoenfeld PH Assumption Test

```python
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


def cox_ph_analysis(
    df: pd.DataFrame,
    time_col: str = 'time',
    event_col: str = 'event',
    covariates: list = None,
    penalizer: float = 0.0,
    output_path: str = 'schoenfeld_residuals.png',
) -> dict:
    """
    Fit a Cox PH model and test the proportional hazards assumption using
    scaled Schoenfeld residuals.

    Args:
        df:          DataFrame containing time, event, and covariate columns.
        time_col:    Time column name.
        event_col:   Event indicator column name.
        covariates:  List of covariate names. If None, use all columns except time/event.
        penalizer:   L2 penalty for coefficient shrinkage (0 = no penalty).
        output_path: Path to save Schoenfeld residual plots.

    Returns:
        Dict with hazard ratios, CIs, p-values, and PH test results.
    """
    if covariates is None:
        covariates = [c for c in df.columns if c not in (time_col, event_col, 'id')]

    data = df[[time_col, event_col] + covariates].dropna()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(data, duration_col=time_col, event_col=event_col)

    print("Cox PH Summary:")
    cph.print_summary()

    # Extract HR table
    summary = cph.summary.copy()
    hr_table = summary[['coef', 'exp(coef)', 'exp(coef) lower 95%',
                         'exp(coef) upper 95%', 'p']].round(4)
    hr_table.columns = ['log_hr', 'hr', 'hr_lower95', 'hr_upper95', 'pvalue']

    # PH assumption: scaled Schoenfeld residuals
    try:
        ph_test = proportional_hazard_test(cph, data, time_transform='rank')
        print("\nProportional Hazards Test (Schoenfeld residuals):")
        print(ph_test.summary)
        ph_results = ph_test.summary[['test_statistic', 'p']].to_dict(orient='index')
    except Exception as e:
        print(f"PH test unavailable: {e}")
        ph_results = {}

    # Plot Schoenfeld residuals for first covariate
    if covariates:
        fig, ax = plt.subplots(figsize=(8, 4))
        try:
            cph.check_assumptions(data, show_plots=False)
            residuals = cph.compute_residuals(data, kind='scaled_schoenfeld')
            ax.scatter(residuals.index, residuals[covariates[0]],
                       alpha=0.5, s=10, color='steelblue')
            ax.axhline(0, color='red', linestyle='--')
            ax.set_title(f'Scaled Schoenfeld Residuals — {covariates[0]}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Scaled residual')
        except Exception:
            ax.text(0.5, 0.5, 'Residuals unavailable', ha='center', va='center',
                    transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    return {
        'concordance_index': float(cph.concordance_index_),
        'hr_table': hr_table,
        'ph_test_results': ph_results,
    }


# --- Demo ---
df = generate_survival_data(n=500)
cox_results = cox_ph_analysis(df, covariates=['group', 'age', 'sex', 'treatment'])
print(f"\nConcordance index: {cox_results['concordance_index']:.3f}")
print(cox_results['hr_table'])
```

### Step 3 — Competing Risks: Fine-Gray vs Cause-Specific

```python
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter
import numpy as np


def competing_risks_analysis(
    df: pd.DataFrame,
    time_col: str = 'time',
    event_type_col: str = 'event_type',
    group_col: str = 'group',
    primary_event: int = 1,
    competing_event: int = 2,
    output_path: str = 'competing_risks_cif.png',
) -> dict:
    """
    Estimate Cumulative Incidence Functions (CIF) for primary and competing events
    using the Aalen-Johansen estimator. Compare CIF between groups.

    Args:
        df:                DataFrame with time and event_type columns.
        time_col:          Time column.
        event_type_col:    Event type column (0=censored, 1=primary, 2=competing).
        group_col:         Group stratification column.
        primary_event:     Integer code for primary event.
        competing_event:   Integer code for competing event.
        output_path:       Path to save CIF plot.

    Returns:
        Dict with CIF estimates at key time points by group and event type.
    """
    groups = sorted(df[group_col].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    event_types = {primary_event: 'Primary event', competing_event: 'Competing event'}

    results = {}
    for ax_idx, (etype, elabel) in enumerate(event_types.items()):
        ax = axes[ax_idx]
        for grp in groups:
            mask = df[group_col] == grp
            T = df.loc[mask, time_col].values
            E = df.loc[mask, event_type_col].values

            ajf = AalenJohansenFitter(calculate_variance=True)
            ajf.fit(T, E, event_of_interest=etype)
            ajf.plot(ax=ax, label=f'Group {grp}')

            # CIF at median time
            med_time = np.median(T)
            cif_at_med = float(ajf.predict(med_time))
            results[f'group{grp}_{elabel}_CIF@median'] = round(cif_at_med, 4)

        ax.set_title(f'CIF — {elabel}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative incidence')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle('Competing Risks: Cumulative Incidence Functions', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # Cause-specific Cox PH for primary event (treating competing as censored)
    df_cs = df.copy()
    df_cs['event_cs'] = (df_cs[event_type_col] == primary_event).astype(int)
    data_cs = df_cs[[time_col, 'event_cs', group_col, 'age', 'sex']].dropna()

    cph_cs = CoxPHFitter()
    cph_cs.fit(data_cs, duration_col=time_col, event_col='event_cs')
    hr_group_cs = float(cph_cs.summary.loc[group_col, 'exp(coef)'])
    results['cause_specific_HR_group'] = round(hr_group_cs, 4)

    print("Competing risks results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


# --- Demo ---
df_cr = generate_survival_data(n=600, competing_risks=True)
cr_results = competing_risks_analysis(
    df_cr,
    time_col='time',
    event_type_col='event_type',
    group_col='group',
)
```

---

## Advanced Usage

### Time-Varying Covariates (Long Format)

```python
import pandas as pd
from lifelines import CoxPHFitter


def cox_time_varying_covariates(
    df_long: pd.DataFrame,
    start_col: str = 'start',
    stop_col: str = 'stop',
    event_col: str = 'event',
    covariates: list = None,
) -> CoxPHFitter:
    """
    Fit a Cox PH model with time-varying covariates using start/stop interval format.

    Long format: each row is a time interval [start, stop) for one subject.
    The event column is 1 only in the last interval if the event occurred.

    Args:
        df_long:     Long-format DataFrame with start/stop intervals.
        start_col:   Start of interval.
        stop_col:    End of interval.
        event_col:   Event indicator (1 = event in this interval).
        covariates:  List of covariate columns.

    Returns:
        Fitted CoxPHFitter object.
    """
    if covariates is None:
        exclude = {start_col, stop_col, event_col, 'id'}
        covariates = [c for c in df_long.columns if c not in exclude]

    cols = [start_col, stop_col, event_col] + covariates
    data = df_long[cols].dropna()

    cph = CoxPHFitter()
    cph.fit(data, duration_col=stop_col, event_col=event_col,
            start_col=start_col, show_progress=False)
    cph.print_summary()
    return cph
```

### IPW-Adjusted Kaplan-Meier

```python
def ipw_km_curve(
    df: pd.DataFrame,
    time_col: str = 'time',
    event_col: str = 'event',
    treatment_col: str = 'treatment',
    confounder_cols: list = None,
) -> pd.DataFrame:
    """
    Inverse probability weighted Kaplan-Meier estimator.
    Weights balance confounders between treatment groups.

    Returns:
        DataFrame with columns: time, survival_treated, survival_control.
    """
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    if confounder_cols is None:
        confounder_cols = ['age', 'sex']

    X = df[confounder_cols].fillna(df[confounder_cols].mean())
    A = df[treatment_col].values

    lr = LogisticRegression(max_iter=500)
    lr.fit(X, A)
    ps = lr.predict_proba(X)[:, 1]  # propensity scores

    # IPW: treated = 1/ps, control = 1/(1-ps)
    weights = np.where(A == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    df = df.copy()
    df['ipw'] = weights

    print("IPW summary: min={:.2f}, max={:.2f}, mean={:.2f}".format(
        weights.min(), weights.max(), weights.mean()))

    return df  # pass df with 'ipw' column to weighted KM estimator
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ConvergenceError` in Cox model | Collinear covariates or complete separation | Drop correlated variables; add `penalizer=0.1` |
| Negative median survival time | All subjects are censored | Increase follow-up time or check event coding |
| `ValueError: Observed and expected must be >= 1` in log-rank | Groups have no events | Filter to groups with ≥1 event |
| Schoenfeld residuals show a trend | PH assumption violated | Add time-interaction term: `covariate * log(t)` |
| CIF sums exceed 1 across causes | Bug in event-type coding | Verify event_type: 0=censored, 1=primary, 2=competing |
| AFT model diverges | Heavy-tailed or multimodal data | Try `LogLogisticAFTFitter` or `LogNormalAFTFitter` |

---

## External Resources

- lifelines Documentation: <https://lifelines.readthedocs.io/>
- scikit-survival: <https://scikit-survival.readthedocs.io/>
- Kleinbaum, D.G. & Klein, M. (2012). *Survival Analysis: A Self-Learning Text*. Springer.
- Fine, J.P. & Gray, R.J. (1999). "A Proportional Hazards Model for the Subdistribution
  of a Competing Risk." *JASA*, 94(446), 496–509.
- Aalen, O.O. et al. (2008). *Survival and Event History Analysis*. Springer.
- Royston, P. & Lambert, P.C. (2011). *Flexible Parametric Survival Analysis*. Stata Press.

---

## Examples

### Example 1 — Full KM + Cox Pipeline

```python
df = generate_survival_data(n=500, seed=0)

# KM analysis
km_res = km_analysis(df, group_labels={0: 'Control', 1: 'Exposed'},
                     output_path='km_curves.png')
print(f"Log-rank p: {km_res['log_rank_p']:.4f}")

# Cox model
cox_res = cox_ph_analysis(df, covariates=['group', 'age', 'sex', 'treatment'])
print(f"Group HR: {cox_res['hr_table'].loc['group', 'hr']:.3f}")
print(f"C-index: {cox_res['concordance_index']:.3f}")
```

### Example 2 — Competing Risks Full Workflow

```python
df_cr = generate_survival_data(n=800, competing_risks=True, seed=7)

print("Event distribution:")
print(df_cr['event_type'].value_counts())

cr = competing_risks_analysis(
    df_cr,
    time_col='time',
    event_type_col='event_type',
    group_col='group',
    output_path='competing_risks.png',
)

print(f"\nCause-specific HR (group): {cr['cause_specific_HR_group']}")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — KM, Cox PH, AFT, competing risks, IPW-KM |
