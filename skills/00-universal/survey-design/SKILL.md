---
name: survey-design
description: >
  Use this Skill to design, validate, and deploy research surveys: Likert scale
  construction, attention checks, pilot testing, ICC reliability, and
  Qualtrics/LimeSurvey export.
tags:
  - universal
  - survey-design
  - Likert
  - reliability
  - questionnaire
  - ICC
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
    - pingouin>=0.5
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Survey Design — Likert Scales, Reliability, and Quality Control

> **TL;DR** — Build psychometrically sound research surveys: construct and
> balance Likert scales, embed attention checks, run cognitive-interview pilot
> testing, compute ICC and Cronbach's alpha with pingouin, detect straightliners
> and speedy responders, and export clean data for Qualtrics or LimeSurvey.

---

## When to Use This Skill

Use this Skill whenever you need to:

- Design a new Likert-scale questionnaire or adapt an existing validated scale
- Choose between 5-point, 7-point, forced-choice, and visual analogue formats
- Write and embed instructed-response items (IRIs) and bogus attention checks
- Plan a pilot study (cognitive interviews + quantitative pilot) before main data collection
- Calculate intraclass correlation coefficients (ICC) for inter-rater or test-retest reliability
- Compute Cronbach's alpha and item-total correlations to assess internal consistency
- Filter low-quality responses (straight-lining, implausibly fast completion) before analysis
- Export survey data from Qualtrics CSV format and apply recoding schemas

| Task | When to apply |
|---|---|
| Scale construction | Designing a new multi-item measure |
| Reliability analysis | Validating a new instrument or adapted translation |
| Attention check filtering | Any online survey with monetary incentive |
| Pilot testing plan | Before main data collection begins |
| ICC calculation | Inter-rater agreement or test-retest studies |
| Qualtrics export processing | After data collection on Qualtrics platform |

---

## Background & Key Concepts

### Likert Scale Construction

A **Likert scale** measures attitudes or perceptions through a set of items
sharing a common response format. Key design choices:

| Choice | 5-point | 7-point |
|---|---|---|
| Reliability | Adequate for most purposes (α ≈ 0.70+) | Slightly higher discrimination |
| Respondent burden | Lower fatigue | Higher cognitive load |
| Use case | Short surveys, general populations | Nuanced attitude measurement |

**Fully labeled** response options (labels on every point) produce more
reliable data than **endpoint labeled** options (labels only at extremes),
particularly in non-Western samples.

**Forced choice** (even number of points, no neutral midpoint) removes
acquiescence bias but frustrates respondents who are genuinely neutral.
Include a neutral midpoint unless you have a specific theoretical reason to
omit it.

### Acquiescence Bias and Reverse Scoring

**Acquiescence bias**: the tendency to agree regardless of item content.
Mitigate by including 30–50 % **reverse-scored items** (negative wording).
Reverse-score during data cleaning: if the scale runs 1–5, reverse score =
(max + 1) − raw score = 6 − raw score.

### Attention Check Item Types

| Type | Description | Example |
|---|---|---|
| Instructed response item (IRI) | Explicit instruction to select a specific option | "To show you are paying attention, select 'Strongly agree'." |
| Bogus item | Asks about a fictional construct | "I often use a Markovian relaxation device when stressed." |
| Consistency check | Repeated item with minor rewording | High correlation expected between duplicates |

Exclude respondents who fail ≥ 1 IRI or select the bogus item. Report the
exclusion rate in the Methods section.

### Pilot Testing Workflow

**Phase 1 — Cognitive interviews (n = 5–10)**

- Recruit participants similar to your target population
- Use think-aloud protocol: ask participants to verbalize thoughts while answering
- Probe: "What does this question mean to you?" "How did you decide on that answer?"
- Revise items with comprehension or ambiguity issues

**Phase 2 — Quantitative pilot (n = 30–50)**

- Administer full survey including attention checks
- Calculate Cronbach's alpha (target ≥ 0.70 for exploratory, ≥ 0.80 for confirmatory)
- Remove items with item-total correlation < 0.30
- Check response distributions for floor/ceiling effects
- Estimate completion time for main survey planning

### ICC — Intraclass Correlation Coefficient

For inter-rater reliability or test-retest, use **ICC(2,1)** (two-way mixed,
absolute agreement, single measures):

| ICC range | Interpretation |
|---|---|
| < 0.50 | Poor |
| 0.50–0.74 | Moderate |
| 0.75–0.89 | Good |
| ≥ 0.90 | Excellent |

Cite Koo & Mae (2016) when reporting ICC values.

### Response Quality Indicators

- **Straight-lining**: all responses within a block are identical. Flag if
  standard deviation across block items = 0.
- **Speedy responding**: completion time < 1/3 of the median time across
  respondents signals insufficient attention.
- **Multivariate outliers**: Mahalanobis distance > χ²(df, p=0.001) threshold.

---

## Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install required packages
pip install "pandas>=1.5" "numpy>=1.23" "scipy>=1.9" "pingouin>=0.5" "matplotlib>=3.6"

# Verify
python -c "import pandas, numpy, scipy, pingouin, matplotlib; print('Setup OK')"
```

For Qualtrics API access (optional automated export):

```bash
# Register at https://api.qualtrics.com/ to obtain a Data Center ID and API token
# export QUALTRICS_API_TOKEN="<paste-your-key>"
# export QUALTRICS_DATA_CENTER="<your-datacenter-id>"
python -c "import os; print(os.getenv('QUALTRICS_API_TOKEN', 'NOT SET'))"
```

---

## Core Workflow

### Step 1 — ICC Calculation with pingouin

```python
import pandas as pd
import numpy as np
import pingouin as pg
from typing import Literal


def compute_icc(
    data: pd.DataFrame,
    raters_col: str,
    targets_col: str,
    ratings_col: str,
    icc_type: str = "ICC2",
    model_desc: str = "two-way mixed, absolute agreement",
) -> dict:
    """
    Compute intraclass correlation coefficient using pingouin.

    Args:
        data:        Long-format DataFrame with columns for rater, target, rating.
        raters_col:  Column name identifying the rater.
        targets_col: Column name identifying the rated subject/item.
        ratings_col: Column name containing the numeric rating.
        icc_type:    ICC type per Shrout & Fleiss (1979). Options:
                     'ICC1', 'ICC2', 'ICC3', 'ICC1k', 'ICC2k', 'ICC3k'.
        model_desc:  Descriptive label for the model.

    Returns:
        Dictionary with ICC estimate, 95 % CI, F-statistic, and p-value.

    Example:
        >>> icc_result = compute_icc(df_long, 'rater', 'participant', 'score')
        >>> print(icc_result)
    """
    icc_results = pg.intraclass_corr(
        data=data,
        raters=raters_col,
        targets=targets_col,
        ratings=ratings_col,
    )

    row = icc_results[icc_results["Type"] == icc_type].iloc[0]

    result = {
        "icc_type": icc_type,
        "model": model_desc,
        "icc": round(row["ICC"], 4),
        "ci_lower": round(row["CI95%"][0], 4),
        "ci_upper": round(row["CI95%"][1], 4),
        "f_value": round(row["F"], 4),
        "df1": int(row["df1"]),
        "df2": int(row["df2"]),
        "p_value": row["pval"],
        "interpretation": _interpret_icc(row["ICC"]),
    }
    return result


def _interpret_icc(icc_val: float) -> str:
    """Map ICC value to qualitative interpretation (Koo & Mae, 2016)."""
    if icc_val < 0.50:
        return "Poor"
    elif icc_val < 0.75:
        return "Moderate"
    elif icc_val < 0.90:
        return "Good"
    else:
        return "Excellent"


def simulate_icc_dataset(
    n_subjects: int = 30,
    n_raters: int = 2,
    true_icc: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a long-format rating dataset for ICC demonstration.

    Args:
        n_subjects: Number of rated subjects.
        n_raters:   Number of raters.
        true_icc:   Approximate ICC to target (controls signal-to-noise ratio).
        seed:       Random seed for reproducibility.

    Returns:
        Long-format DataFrame with columns: rater, participant, score.
    """
    rng = np.random.default_rng(seed)
    subject_effects = rng.normal(loc=50, scale=10 * np.sqrt(true_icc), size=n_subjects)
    rows = []
    for r in range(1, n_raters + 1):
        noise_sd = 10 * np.sqrt(1 - true_icc)
        for s_idx, subject_mean in enumerate(subject_effects):
            score = subject_mean + rng.normal(0, noise_sd)
            score = float(np.clip(round(score), 1, 100))
            rows.append({"rater": f"Rater_{r}", "participant": f"P{s_idx+1:03d}", "score": score})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_long = simulate_icc_dataset(n_subjects=30, n_raters=2, true_icc=0.85)
    result = compute_icc(df_long, "rater", "participant", "score")
    print("ICC Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    # Report line: ICC(2,1) = 0.87, 95% CI [0.75, 0.93], p < .001 (Good reliability)
    print(
        f"\nReport: ICC(2,1) = {result['icc']:.2f}, "
        f"95% CI [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}], "
        f"p = {result['p_value']:.3f} ({result['interpretation']} reliability)"
    )
```

### Step 2 — Straight-Lining and Attention Check Filtering Pipeline

```python
import pandas as pd
import numpy as np
from typing import Optional


def flag_straightliners(
    df: pd.DataFrame,
    item_cols: list[str],
    id_col: str = "response_id",
    sd_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Flag respondents who gave identical responses across all scale items.

    Args:
        df:           Wide-format response DataFrame.
        item_cols:    List of column names for scale items (all same Likert range).
        id_col:       Respondent identifier column.
        sd_threshold: Responses with SD ≤ this threshold are flagged.
                      Use 0.0 for exact straight-lining; raise to 0.3 for near-SL.

    Returns:
        DataFrame with added column 'is_straightliner' (bool).
    """
    df = df.copy()
    row_sds = df[item_cols].apply(pd.to_numeric, errors="coerce").std(axis=1)
    df["item_sd"] = row_sds
    df["is_straightliner"] = row_sds <= sd_threshold
    n_flagged = df["is_straightliner"].sum()
    print(f"Straight-liners flagged: {n_flagged}/{len(df)} "
          f"({100*n_flagged/len(df):.1f}%)")
    return df


def flag_speedy_responders(
    df: pd.DataFrame,
    duration_col: str,
    threshold_fraction: float = 1 / 3,
) -> pd.DataFrame:
    """
    Flag respondents whose survey duration is below a fraction of the median.

    Args:
        df:                 Response DataFrame with a duration column (seconds).
        duration_col:       Column name for total survey duration in seconds.
        threshold_fraction: Fraction of median; default 1/3 per standard practice.

    Returns:
        DataFrame with added column 'is_speedy' (bool).
    """
    df = df.copy()
    median_duration = df[duration_col].median()
    threshold = threshold_fraction * median_duration
    df["is_speedy"] = df[duration_col] < threshold
    n_flagged = df["is_speedy"].sum()
    print(f"Speedy responders flagged: {n_flagged}/{len(df)} "
          f"(threshold: {threshold:.0f}s; median: {median_duration:.0f}s)")
    return df


def apply_attention_check_filter(
    df: pd.DataFrame,
    iri_checks: list[dict],
    bogus_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Filter respondents who fail instructed-response items or select bogus items.

    Args:
        df:          Response DataFrame.
        iri_checks:  List of dicts, each with keys:
                     'col' (column name), 'expected' (correct response value).
        bogus_cols:  List of column names for bogus items; any non-null/non-zero
                     selection counts as a failure.

    Returns:
        DataFrame with added column 'failed_attention_check' (bool).
    """
    df = df.copy()
    fail_mask = pd.Series(False, index=df.index)

    for check in iri_checks:
        col, expected = check["col"], check["expected"]
        if col in df.columns:
            fail_mask |= df[col] != expected

    if bogus_cols:
        for col in bogus_cols:
            if col in df.columns:
                fail_mask |= df[col].notna() & (df[col] != 0) & (df[col] != "")

    df["failed_attention_check"] = fail_mask
    n_failed = fail_mask.sum()
    print(f"Attention check failures: {n_failed}/{len(df)} ({100*n_failed/len(df):.1f}%)")
    return df


def run_full_qc_pipeline(
    df: pd.DataFrame,
    item_cols: list[str],
    duration_col: str,
    iri_checks: list[dict],
    bogus_cols: Optional[list[str]] = None,
    reverse_score_cols: Optional[list[str]] = None,
    scale_max: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full data-quality pipeline: reverse scoring, straight-line detection,
    speed detection, and attention check filtering.

    Returns:
        Tuple of (cleaned DataFrame, excluded DataFrame) with exclusion reasons.
    """
    df = df.copy()

    # Reverse scoring
    if reverse_score_cols:
        for col in reverse_score_cols:
            if col in df.columns:
                df[col] = (scale_max + 1) - pd.to_numeric(df[col], errors="coerce")

    df = flag_straightliners(df, item_cols)
    df = flag_speedy_responders(df, duration_col)
    df = apply_attention_check_filter(df, iri_checks, bogus_cols)

    exclude_mask = (
        df["is_straightliner"] | df["is_speedy"] | df["failed_attention_check"]
    )
    excluded = df[exclude_mask].copy()
    cleaned = df[~exclude_mask].copy()

    print(f"\nQC Summary: {len(cleaned)} retained, {len(excluded)} excluded "
          f"({100*len(excluded)/len(df):.1f}% exclusion rate)")
    return cleaned, excluded


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n = 200
    item_names = [f"item_{i:02d}" for i in range(1, 11)]
    demo_data = {col: rng.integers(1, 6, size=n) for col in item_names}
    demo_data["response_id"] = [f"R{i:04d}" for i in range(n)]
    demo_data["duration_sec"] = rng.integers(60, 900, size=n)
    demo_data["iri_check"] = rng.choice([3, 1, 2, 4, 5], size=n, p=[0.85, 0.05, 0.04, 0.03, 0.03])
    demo_data["bogus_markov"] = rng.choice([None, 1, 2], size=n, p=[0.93, 0.04, 0.03])

    df_raw = pd.DataFrame(demo_data)
    # Inject 5 straight-liners
    for i in range(5):
        df_raw.loc[i, item_names] = 3

    cleaned, excluded = run_full_qc_pipeline(
        df=df_raw,
        item_cols=item_names,
        duration_col="duration_sec",
        iri_checks=[{"col": "iri_check", "expected": 3}],
        bogus_cols=["bogus_markov"],
        reverse_score_cols=["item_03", "item_07"],
        scale_max=5,
    )
    print(f"\nCleaned dataset shape: {cleaned.shape}")
```

### Step 3 — Cronbach's Alpha and Item-Total Correlations

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import pingouin as pg
import matplotlib.pyplot as plt


def compute_cronbach_alpha(
    df: pd.DataFrame,
    item_cols: list[str],
    ci_level: float = 0.95,
) -> dict:
    """
    Compute Cronbach's alpha with 95 % confidence interval using pingouin.

    Args:
        df:        DataFrame containing item responses (numeric).
        item_cols: List of item column names to include.
        ci_level:  Confidence interval level (default 0.95).

    Returns:
        Dictionary with alpha, CI, and interpretation.
    """
    items = df[item_cols].apply(pd.to_numeric, errors="coerce").dropna()
    result = pg.cronbach_alpha(data=items, ci=ci_level)
    alpha_val = result[0]
    ci = result[1]

    interpretation = (
        "Unacceptable" if alpha_val < 0.60 else
        "Questionable" if alpha_val < 0.70 else
        "Acceptable"   if alpha_val < 0.80 else
        "Good"         if alpha_val < 0.90 else
        "Excellent"
    )

    return {
        "cronbach_alpha": round(alpha_val, 4),
        "ci_lower": round(ci[0], 4),
        "ci_upper": round(ci[1], 4),
        "n_items": len(item_cols),
        "n_respondents": len(items),
        "interpretation": interpretation,
    }


def compute_item_total_correlations(
    df: pd.DataFrame,
    item_cols: list[str],
) -> pd.DataFrame:
    """
    Compute corrected item-total correlations (CITC) for all scale items.

    CITC = Pearson correlation between the item and the sum of all OTHER items.
    Items with CITC < 0.30 are flagged for removal.

    Args:
        df:        DataFrame with numeric item responses.
        item_cols: List of item column names.

    Returns:
        DataFrame with columns: item, citc, alpha_if_deleted, flag.
    """
    items_df = df[item_cols].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []

    for col in item_cols:
        rest_cols = [c for c in item_cols if c != col]
        rest_sum = items_df[rest_cols].sum(axis=1)
        r, p = pearsonr(items_df[col], rest_sum)

        # Alpha if deleted
        alpha_if_del = pg.cronbach_alpha(data=items_df[rest_cols])[0]

        rows.append({
            "item": col,
            "citc": round(r, 4),
            "alpha_if_deleted": round(alpha_if_del, 4),
            "flag": "REMOVE" if r < 0.30 else "",
        })

    return pd.DataFrame(rows).sort_values("citc", ascending=False)


def plot_item_total_correlations(
    citc_df: pd.DataFrame,
    output_path: str = "item_total_correlations.png",
) -> None:
    """Plot CITC bar chart with the 0.30 threshold line."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#D32F2F" if r < 0.30 else "#1976D2" for r in citc_df["citc"]]
    ax.barh(citc_df["item"], citc_df["citc"], color=colors)
    ax.axvline(0.30, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.30)")
    ax.set_xlabel("Corrected Item-Total Correlation")
    ax.set_title("Item-Total Correlations (red = below threshold)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Item-total correlation chart saved to {output_path}")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_resp, n_items = 150, 10
    item_cols = [f"item_{i:02d}" for i in range(1, n_items + 1)]
    # Create correlated item data (factor structure)
    latent = rng.normal(0, 1, size=n_resp)
    demo = {col: np.clip((latent + rng.normal(0, 0.8, n_resp) * 2 + 3).round(), 1, 5)
            for col in item_cols}
    df_items = pd.DataFrame(demo)

    alpha_result = compute_cronbach_alpha(df_items, item_cols)
    print("Cronbach's Alpha Results:")
    for k, v in alpha_result.items():
        print(f"  {k}: {v}")

    citc_df = compute_item_total_correlations(df_items, item_cols)
    print("\nItem-Total Correlations:")
    print(citc_df.to_string(index=False))

    items_to_remove = citc_df[citc_df["flag"] == "REMOVE"]["item"].tolist()
    if items_to_remove:
        print(f"\nRecommended for removal: {items_to_remove}")
    else:
        print("\nAll items meet the CITC ≥ 0.30 threshold.")

    plot_item_total_correlations(citc_df)
```

---

## Advanced Usage

### Qualtrics CSV Export Column Conventions

When downloading Qualtrics data as CSV, the default export has two header rows:
the first contains column labels (question text), the second contains encoded
question IDs (e.g., `Q1`, `Q1_1`). Use `skiprows=[1]` after loading the second
header row to parse cleanly:

```python
import pandas as pd

def load_qualtrics_csv(
    path: str,
    skip_preview_row: bool = True,
) -> pd.DataFrame:
    """
    Load a Qualtrics CSV export with proper header handling.

    The Qualtrics default export has three header rows:
    row 0: column labels (human-readable)
    row 1: question text (verbose)
    row 2+: data

    Args:
        path:              File path to the Qualtrics CSV export.
        skip_preview_row:  If True, skip row 1 (question text row).

    Returns:
        DataFrame with row-0 column labels and data rows only.
    """
    headers = pd.read_csv(path, nrows=0).columns.tolist()
    df = pd.read_csv(path, skiprows=[1, 2] if skip_preview_row else [1], header=0)
    df.columns = headers
    # Drop Qualtrics metadata columns
    meta_cols = ["StartDate", "EndDate", "Status", "IPAddress", "Progress",
                 "Duration (in seconds)", "Finished", "RecordedDate",
                 "ResponseId", "RecipientLastName", "RecipientFirstName",
                 "RecipientEmail", "ExternalReference", "LocationLatitude",
                 "LocationLongitude", "DistributionChannel", "UserLanguage"]
    data_cols = [c for c in df.columns if c not in meta_cols]
    return df[["ResponseId", "Duration (in seconds)"] + data_cols]
```

### Test-Retest Reliability

For scales administered twice (Time 1 and Time 2):

```python
import pingouin as pg
import pandas as pd


def test_retest_reliability(
    df_t1: pd.DataFrame,
    df_t2: pd.DataFrame,
    item_cols: list[str],
    id_col: str = "participant_id",
) -> dict:
    """
    Compute test-retest ICC(2,1) for composite scale scores.

    Args:
        df_t1:     Time-1 DataFrame with participant ID and item columns.
        df_t2:     Time-2 DataFrame with participant ID and item columns.
        item_cols: Scale item columns to sum into composite score.
        id_col:    Participant identifier column.

    Returns:
        Dictionary with ICC results and Pearson r.
    """
    scores_t1 = df_t1.set_index(id_col)[item_cols].sum(axis=1).rename("T1")
    scores_t2 = df_t2.set_index(id_col)[item_cols].sum(axis=1).rename("T2")
    merged = pd.concat([scores_t1, scores_t2], axis=1).dropna().reset_index()
    long_df = merged.melt(id_vars=id_col, value_vars=["T1", "T2"],
                          var_name="time", value_name="score")
    icc = pg.intraclass_corr(data=long_df, raters="time",
                              targets=id_col, ratings="score")
    icc2_row = icc[icc["Type"] == "ICC2"].iloc[0]
    r = merged[["T1", "T2"]].corr().iloc[0, 1]
    return {
        "icc2_1": round(icc2_row["ICC"], 4),
        "ci_lower": round(icc2_row["CI95%"][0], 4),
        "ci_upper": round(icc2_row["CI95%"][1], 4),
        "pearson_r": round(r, 4),
        "n_pairs": len(merged),
    }
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `pingouin.intraclass_corr` returns NaN | Missing values in ratings column | Drop rows with missing data before calling the function |
| Cronbach's alpha is negative | Reverse-coded items not reversed | Run reverse-scoring step before computing alpha |
| ICC lower bound is negative | Very small sample (n < 20) | Increase pilot sample; report wide CI transparently |
| High exclusion rate (> 30 %) | Overly strict attention checks | Review IRI wording; consider one IRI instead of two |
| Floor/ceiling effects on Likert items | Response scale too narrow or extreme population | Switch from 5-point to 7-point; check sample frame |
| Qualtrics CSV parse error | Extra header rows | Use `skiprows=[1, 2]` in `pd.read_csv()` |
| Item-total correlation < 0 | Item keyed in wrong direction | Reverse-score the item; recheck scale direction |

---

## External Resources

- Likert scale design guidelines: <https://www.surveymonkey.com/mp/likert-scale/>
- pingouin documentation (ICC): <https://pingouin-stats.org/build/html/generated/pingouin.intraclass_corr.html>
- Koo & Mae (2016) ICC guidelines: <https://doi.org/10.1016/j.joca.2015.09.020>
- Qualtrics API documentation: <https://api.qualtrics.com/>
- LimeSurvey documentation: <https://www.limesurvey.org/manual>
- COSMIN reporting guideline for reliability: <https://www.cosmin.nl/>
- Survey quality handbook (Groves et al.): <https://doi.org/10.1002/9781118121818>

---

## Examples

### Example 1 — ICC for Two Raters Rating Pain Scores

```python
import pandas as pd
import numpy as np

# Simulate two raters scoring pain intensity (0-100 NRS)
# for 40 patients assessed at clinic admission
rng = np.random.default_rng(0)
n = 40
true_scores = rng.normal(loc=45, scale=20, size=n)
rater1 = np.clip(true_scores + rng.normal(0, 5, n), 0, 100).round()
rater2 = np.clip(true_scores + rng.normal(0, 5, n), 0, 100).round()
patient_ids = [f"PT{i:03d}" for i in range(n)]

df_wide = pd.DataFrame({"patient": patient_ids, "Rater_A": rater1, "Rater_B": rater2})
df_long = df_wide.melt(id_vars="patient", var_name="rater", value_name="pain_score")

icc_res = compute_icc(df_long, raters_col="rater", targets_col="patient",
                       ratings_col="pain_score", icc_type="ICC2")
print("Pain score ICC(2,1):")
print(f"  ICC = {icc_res['icc']:.2f} [{icc_res['ci_lower']:.2f}, {icc_res['ci_upper']:.2f}]")
print(f"  Interpretation: {icc_res['interpretation']}")
```

### Example 2 — Full Quality Control Pipeline on Simulated Qualtrics Data

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(99)
n = 300
item_cols = [f"Q{i}" for i in range(1, 11)]
df_raw = pd.DataFrame({col: rng.integers(1, 6, n) for col in item_cols})
df_raw["response_id"] = [f"R{i:04d}" for i in range(n)]
df_raw["duration_sec"] = rng.integers(30, 1200, n)
df_raw["attention_iri"] = rng.choice([4, 1, 2, 3, 5], n, p=[0.88, 0.04, 0.03, 0.03, 0.02])
df_raw["bogus_relaxation"] = rng.choice([None, 1, 2, 3], n, p=[0.94, 0.02, 0.02, 0.02])
# Inject 8 straight-liners
df_raw.loc[:7, item_cols] = 2

cleaned_df, excluded_df = run_full_qc_pipeline(
    df=df_raw,
    item_cols=item_cols,
    duration_col="duration_sec",
    iri_checks=[{"col": "attention_iri", "expected": 4}],
    bogus_cols=["bogus_relaxation"],
    reverse_score_cols=["Q3", "Q7"],
    scale_max=5,
)

alpha = compute_cronbach_alpha(cleaned_df, item_cols)
print(f"\nPost-QC Cronbach's alpha: {alpha['cronbach_alpha']} "
      f"({alpha['interpretation']}), n = {alpha['n_respondents']}")

citc = compute_item_total_correlations(cleaned_df, item_cols)
print("\nTop 3 items by CITC:")
print(citc.head(3).to_string(index=False))
```

### Example 3 — Cronbach's Alpha with Item Deletion Recommendation

```python
import pandas as pd
import numpy as np

rng = np.random.default_rng(7)
n_resp = 180
item_cols = [f"item_{i:02d}" for i in range(1, 13)]
latent = rng.normal(3, 1, n_resp)
# Insert two weak items
data = {col: np.clip((latent + rng.normal(0, 0.6, n_resp)).round(), 1, 5)
        for col in item_cols[:10]}
data["item_11"] = rng.integers(1, 6, n_resp)  # random noise item
data["item_12"] = rng.integers(1, 6, n_resp)  # random noise item
df = pd.DataFrame(data)

alpha_full = compute_cronbach_alpha(df, item_cols)
print(f"Alpha (all 12 items): {alpha_full['cronbach_alpha']} — {alpha_full['interpretation']}")

citc_df = compute_item_total_correlations(df, item_cols)
weak = citc_df[citc_df["flag"] == "REMOVE"]["item"].tolist()
print(f"Items recommended for removal: {weak}")

if weak:
    final_items = [c for c in item_cols if c not in weak]
    alpha_final = compute_cronbach_alpha(df, final_items)
    print(f"Alpha after removing weak items ({len(final_items)} items): "
          f"{alpha_final['cronbach_alpha']} — {alpha_final['interpretation']}")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Likert design, ICC, Cronbach's alpha, CITC, QC pipeline, Qualtrics export |
