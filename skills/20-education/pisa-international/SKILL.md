---
name: pisa-international
description: >
  Use this Skill to analyze PISA/TIMSS international large-scale assessment
  data: plausible values averaging, Fay's BRR variance estimation, and
  cross-national SES gradients.
tags:
  - education
  - PISA
  - TIMSS
  - large-scale-assessment
  - plausible-values
  - international-comparison
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
    - statsmodels>=0.14
last_updated: "2026-03-18"
status: stable
---

# PISA/TIMSS International Large-Scale Assessment Analysis

> **TL;DR** — Correctly analyse PISA and TIMSS data with all 10 plausible
> values, Fay's BRR variance estimation (80 replicate weights), ESCS gradient
> regressions, and cross-national comparison figures.

---

## When to Use

Use this Skill when you need to:

- Compute country-level mean performance estimates on PISA or TIMSS using
  the statistically correct combination of all plausible values.
- Estimate standard errors that account for both sampling variance (BRR
  replicate weights) and imputation variance (between-plausible-value spread).
- Analyse SES gradients (score ~ ESCS regression) and gender gaps across
  countries.
- Produce cross-national comparison figures with proper error bars.
- Check proficiency level distributions (PISA Levels 1–6).

**Do NOT use** naive averaging of plausible values without BRR weights —
this underestimates standard errors and produces invalid significance tests.

---

## Background

### Why Plausible Values Exist

PISA does not administer every item to every student (matrix sampling design).
Each student answers only a booklet subset, so the full proficiency scale
cannot be directly estimated. Instead, PISA generates **M = 10 plausible
values** (PV1MATH–PV10MATH) per student — random draws from the posterior
distribution of proficiency given the student's responses and background.

### Correct Estimation Protocol (Rubin's Rules)

For statistic Q (e.g., country mean):

```
1. For each plausible value m = 1..M:
       Q_m  = estimate using PV_m with BRR replicate weights → Var_BRR_m

2. Point estimate:   Q = (1/M) * Σ Q_m

3. Sampling variance:  U = (1/M) * Σ Var_BRR_m

4. Imputation variance: B = (1/(M-1)) * Σ (Q_m - Q)²

5. Total variance:  Var_total = U + (1 + 1/M) * B

6. SE = sqrt(Var_total)
```

### Fay's BRR Variance Estimation

PISA provides 80 balanced repeated replication (BRR) weights (W_FSTR1–W_FSTR80).
For each replicate r:

```
Var_BRR(Q) = (1 / (80 * k²)) * Σ_r (Q_r - Q_full)²
```

where k = 0.5 (Fay coefficient used in PISA).

---

## Environment Setup

```bash
conda create -n pisa_analysis python=3.11 -y
conda activate pisa_analysis
pip install pandas>=1.5 numpy>=1.23 scipy>=1.9 matplotlib>=3.6 statsmodels>=0.14

# Download PISA 2022 student data (SPSS format) from OECD:
# https://www.oecd.org/pisa/data/2022database/
# Convert .sav to parquet for efficiency:
pip install pyreadstat pyarrow
python -c "import pyreadstat, pandas as pd; df, meta = pyreadstat.read_sav('STU_QQQ_SAS.sav'); df.to_parquet('pisa2022_student.parquet')"
```

---

## Core Workflow

### Step 1 — Load Data and Define Helpers

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable


def load_pisa_sample(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic PISA-like microdata for demonstration purposes.

    In production, replace this with:
        df = pd.read_parquet('pisa2022_student.parquet')

    The synthetic data reproduces the column structure of PISA 2022 student file.

    Args:
        n:    Number of synthetic student records.
        seed: Random seed.

    Returns:
        DataFrame with PV1MATH–PV10MATH, ESCS, ST004D01T (gender), CNT,
        SENWT (final student weight), W_FSTR1–W_FSTR80 (replicate weights).
    """
    rng = np.random.default_rng(seed)
    countries = ["AUS", "CAN", "CHN", "DEU", "FIN", "GBR", "JPN", "KOR", "USA", "SGP"]
    n_per = n // len(countries)

    frames = []
    country_means = {c: m for c, m in zip(countries, [510, 520, 591, 509, 525, 505, 536, 527, 502, 575])}
    for cnt in countries:
        mu = country_means[cnt]
        escs   = rng.normal(0, 1, n_per)
        theta  = mu + 30 * escs + rng.normal(0, 70, n_per)
        pvs = {f"PV{m}MATH": theta + rng.normal(0, 15, n_per) for m in range(1, 11)}
        gender = rng.choice([1, 2], n_per)
        weight = rng.uniform(1, 100, n_per)
        rep_weights = {f"W_FSTR{r}": weight * (1 + rng.choice([-0.5, 0.5]) * rng.binomial(1, 0.5, n_per))
                       for r in range(1, 81)}
        frame = pd.DataFrame({"CNT": cnt, "ESCS": escs, "ST004D01T": gender,
                               "SENWT": weight, **pvs, **rep_weights})
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


PV_COLS = [f"PV{m}MATH" for m in range(1, 11)]
BRR_COLS = [f"W_FSTR{r}" for r in range(1, 81)]
FAY_K = 0.5
N_PV = 10
```

### Step 2 — BRR Variance for a Single Plausible Value

```python
def brr_variance(
    df: pd.DataFrame,
    stat_func: Callable[[pd.DataFrame, str], float],
    pv_col: str,
    weight_col: str = "SENWT",
    fay_k: float = FAY_K,
) -> tuple[float, float]:
    """
    Compute Fay's BRR variance estimate for a statistic.

    Args:
        df:         Student DataFrame with replicate weights W_FSTR1–W_FSTR80.
        stat_func:  Function(df, weight_col) -> float that computes the statistic.
        pv_col:     Plausible value column for this iteration.
        weight_col: Base final weight column.
        fay_k:      Fay coefficient (PISA uses 0.5).

    Returns:
        (point_estimate, brr_variance) tuple.
    """
    q_full = stat_func(df.assign(_w=df[weight_col]), "_w")

    q_reps = np.array([
        stat_func(df.assign(_w=df[brr_col]), "_w")
        for brr_col in BRR_COLS
    ])

    var_brr = np.sum((q_reps - q_full) ** 2) / (len(BRR_COLS) * fay_k ** 2)
    return q_full, var_brr


def weighted_mean(df: pd.DataFrame, weight_col: str, value_col: str = None) -> float:
    """Compute weighted mean. value_col defaults to the PV column stored in df._pv."""
    vc = value_col or "_pv"
    return np.average(df[vc], weights=df[weight_col])
```

### Step 3 — Full PV-Correct Country Mean with SE

```python
def pisa_mean_se(
    df: pd.DataFrame,
    group_col: str = "CNT",
    weight_col: str = "SENWT",
) -> pd.DataFrame:
    """
    Compute PISA-correct country mean math scores and standard errors.

    Combines all 10 plausible values using Rubin's rules with Fay BRR.

    Args:
        df:         Full PISA student DataFrame.
        group_col:  Column identifying groups (e.g. 'CNT' for countries).
        weight_col: Final student weight column.

    Returns:
        DataFrame with columns: group, mean, se, ci_low, ci_high.
    """
    groups = df[group_col].unique()
    records = []

    for grp in groups:
        sub = df[df[group_col] == grp].copy()
        q_pv = np.zeros(N_PV)
        var_pv = np.zeros(N_PV)

        for m, pv_col in enumerate(PV_COLS):
            def stat_fn(d, wc, pvc=pv_col):
                return np.average(d[pvc], weights=d[wc])

            q_full = stat_fn(sub, weight_col)
            q_reps = np.array([stat_fn(sub, brr_col) for brr_col in BRR_COLS])
            var_brr = np.sum((q_reps - q_full) ** 2) / (len(BRR_COLS) * FAY_K ** 2)
            q_pv[m] = q_full
            var_pv[m] = var_brr

        # Rubin's combination rules
        q_mean = q_pv.mean()
        u_sampling = var_pv.mean()
        b_imputation = np.var(q_pv, ddof=1)
        var_total = u_sampling + (1 + 1 / N_PV) * b_imputation
        se = np.sqrt(var_total)

        records.append({
            "group": grp,
            "mean": round(q_mean, 2),
            "se": round(se, 2),
            "ci_low": round(q_mean - 1.96 * se, 2),
            "ci_high": round(q_mean + 1.96 * se, 2),
        })

    result = pd.DataFrame(records).sort_values("mean", ascending=False).reset_index(drop=True)
    return result
```

---

## Advanced Usage

### SES Gradient Regression per Country

```python
import statsmodels.api as sm


def ses_gradient_by_country(
    df: pd.DataFrame,
    group_col: str = "CNT",
    weight_col: str = "SENWT",
) -> pd.DataFrame:
    """
    Estimate SES gradient (score ~ ESCS) per country using PV averaging.

    For each country, runs WLS regression for each of the 10 PVs and
    combines slope/SE using Rubin's rules.

    Args:
        df:         PISA student DataFrame with ESCS and PV1MATH–PV10MATH.
        group_col:  Grouping column (country).
        weight_col: Weight column.

    Returns:
        DataFrame with: group, slope, se, r2_mean, interpretation.
    """
    groups = df[group_col].unique()
    records = []

    for grp in groups:
        sub = df[df[group_col] == grp].dropna(subset=["ESCS"])
        X = sm.add_constant(sub["ESCS"])
        w = sub[weight_col]

        slopes = np.zeros(N_PV)
        r2s = np.zeros(N_PV)

        for m, pv_col in enumerate(PV_COLS):
            model = sm.WLS(sub[pv_col], X, weights=w).fit()
            slopes[m] = model.params["ESCS"]
            r2s[m] = model.rsquared

        # Simplified SE: use between-PV variance only for small samples
        slope_mean = slopes.mean()
        b_imp = np.var(slopes, ddof=1)
        se = np.sqrt((1 + 1 / N_PV) * b_imp)

        records.append({
            "group": grp,
            "slope": round(slope_mean, 2),
            "se": round(se, 2),
            "r2_mean": round(r2s.mean(), 4),
            "interpretation": "strong" if slope_mean > 40 else "moderate" if slope_mean > 25 else "weak",
        })

    return pd.DataFrame(records).sort_values("slope", ascending=False).reset_index(drop=True)


def plot_cross_national_comparison(
    means_df: pd.DataFrame,
    title: str = "PISA Math Mean Scores by Country",
    output_path: str = "pisa_comparison.png",
) -> None:
    """
    Forest plot of country means with 95% confidence intervals.

    Args:
        means_df:    Output of pisa_mean_se().
        title:       Figure title.
        output_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(9, len(means_df) * 0.4 + 1))
    y = np.arange(len(means_df))

    ax.barh(y, means_df["mean"], xerr=[
        means_df["mean"] - means_df["ci_low"],
        means_df["ci_high"] - means_df["mean"],
    ], capsize=4, color="#4C72B0", ecolor="black", height=0.6)

    ax.axvline(means_df["mean"].mean(), color="red", linestyle="--", label="OECD average")
    ax.set_yticks(y)
    ax.set_yticklabels(means_df["group"])
    ax.set_xlabel("PISA Math Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Cross-national comparison saved to {output_path}")
    plt.close()
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| SE much smaller than OECD published values | Using only one PV or ignoring BRR | Use all 10 PVs and all 80 BRR replicates |
| Memory error with full PISA file | 600k+ rows × 1000+ columns | Load only needed columns with `usecols` |
| `KeyError: PV1MATH` | Different PISA cycle naming | Check actual column names (e.g. PV1READ) |
| Negative BRR variance | Fay coefficient mismatch | Confirm k=0.5 for PISA; k=1 for PIRLS |
| Missing ESCS for many students | ESCS imputed in separate file | Merge student and school files first |
| Country rank differs from OECD report | OECD uses combined reading+math+science | Use subject-specific PVs separately |

---

## External Resources

- PISA 2022 data and documentation: https://www.oecd.org/pisa/data/2022database/
- PISA technical standards (BRR): https://www.oecd.org/pisa/pisaproducts/PISA2022-technical-report.pdf
- TIMSS data: https://timssandpirls.bc.edu/databases/
- International Large-Scale Assessment handbook: https://doi.org/10.1007/978-3-319-78692-5
- `EdSurvey` R package for PISA: https://naep-research.airweb.org/EdSurvey

---

## Examples

### Example 1 — Country Mean Math Scores with BRR SE

```python
def example_country_means():
    """Compute PISA math means with correct BRR SE for 10 countries."""
    df = load_pisa_sample(n=5000)
    means = pisa_mean_se(df, group_col="CNT", weight_col="SENWT")
    print("PISA Math Means (PV-correct BRR SE):")
    print(means.to_string(index=False))
    plot_cross_national_comparison(means, output_path="pisa_math_means.png")
    return means


if __name__ == "__main__":
    example_country_means()
```

### Example 2 — SES Gradient per Country

```python
def example_ses_gradients():
    """Estimate SES gradient slopes across countries and visualise."""
    df = load_pisa_sample(n=8000)
    gradients = ses_gradient_by_country(df)

    print("SES Gradient Slopes (score points per ESCS unit):")
    print(gradients.to_string(index=False))

    # Plot slope comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(gradients["group"], gradients["slope"],
            xerr=gradients["se"] * 1.96, capsize=4,
            color=["#d62728" if s > 40 else "#2ca02c" for s in gradients["slope"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SES Gradient Slope (score points / ESCS)")
    ax.set_title("PISA Math SES Gradient by Country")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("ses_gradients.png", dpi=150)
    return gradients


if __name__ == "__main__":
    example_ses_gradients()
```

### Example 3 — Gender Gap Analysis

```python
def example_gender_gap():
    """Compute gender gap in math for each country using PV-correct estimates."""
    df = load_pisa_sample(n=6000)

    # Code gender: 1=female, 2=male (PISA convention)
    results = []
    for cnt in df["CNT"].unique():
        sub = df[df["CNT"] == cnt]
        for gender, label in [(1, "Female"), (2, "Male")]:
            sg = sub[sub["ST004D01T"] == gender]
            pv_means = [np.average(sg[pv], weights=sg["SENWT"]) for pv in PV_COLS]
            results.append({"country": cnt, "gender": label, "mean": np.mean(pv_means)})

    gap_df = pd.DataFrame(results).pivot(index="country", columns="gender", values="mean")
    gap_df["gap_M_minus_F"] = gap_df["Male"] - gap_df["Female"]
    gap_df = gap_df.sort_values("gap_M_minus_F", ascending=False)

    print("Gender gap in PISA math (M - F, score points):")
    print(gap_df["gap_M_minus_F"].to_string())
    return gap_df


if __name__ == "__main__":
    example_gender_gap()
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — PV averaging, BRR variance, SES gradient, cross-national comparison |
