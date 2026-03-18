---
name: clinical-assessment
description: >
  Use this Skill to score and interpret clinical scales: PHQ-9, GAD-7, PCL-5,
  reliable change index (RCI), norm comparison, and longitudinal change
  visualization.
tags:
  - psychology
  - clinical-assessment
  - PHQ-9
  - PCL-5
  - reliable-change-index
  - questionnaire
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
    - matplotlib>=3.6
    - scipy>=1.9
last_updated: "2026-03-18"
status: stable
---

# Clinical Assessment Scoring and Interpretation

> **TL;DR** — Batch-score PHQ-9, GAD-7, PCL-5, BDI-II, and SWLS questionnaires.
> Compute the Reliable Change Index (RCI) for clinically significant change,
> handle missing items with prorated scoring, compare against published norms,
> and generate longitudinal trajectory visualizations.

---

## When to Use

Use this Skill when you need to:

- Score clinical questionnaires from raw item responses in a data frame
- Apply standard severity cutoffs (minimal, mild, moderate, severe)
- Determine whether pre-post change is statistically reliable (RCI)
- Classify individuals as recovered, improved, unchanged, or deteriorated
- Compute prorated scores when 1–2 items are missing
- Generate per-participant longitudinal plots showing clinical trajectories
- Compare scores to published normative samples (z-score lookup)

---

## Background

### Scoring Rules Summary

| Scale | Items | Range | Cutoffs |
|---|---|---|---|
| PHQ-9 | 9 items, 0–3 each | 0–27 | 0–4 minimal, 5–9 mild, 10–14 moderate, 15–27 severe |
| GAD-7 | 7 items, 0–3 each | 0–21 | 0–4 minimal, 5–9 mild, 10–14 moderate, 15–21 severe |
| PCL-5 | 20 items, 0–4 each | 0–80 | ≥ 33 provisional PTSD |
| BDI-II | 21 items, 0–3 each | 0–63 | 0–13 minimal, 14–19 mild, 20–28 moderate, 29–63 severe |
| SWLS | 5 items, 1–7 each | 5–35 | 5–9 extremely dissatisfied, ≥31 extremely satisfied |

### PCL-5 DSM-5 Cluster Subscores

| Cluster | Items (1-indexed) | Symptom Group |
|---|---|---|
| B (Intrusion) | 1–5 | Re-experiencing |
| C (Avoidance) | 6–7 | Avoidance |
| D (Neg cognition) | 8–14 | Negative alterations in cognition/mood |
| E (Hyperarousal) | 15–20 | Alterations in arousal/reactivity |

### Reliable Change Index (RCI)

Jacobson & Truax (1991):

```
SE_diff = SD_pre × √(2) × √(1 − r_tt)
RCI = (post_score − pre_score) / SE_diff
```

Where `r_tt` is the test-retest reliability of the scale. Reliable change:
`|RCI| ≥ 1.96` (two-tailed, α = .05).

**Clinical significance** requires BOTH reliable change AND movement from the
dysfunctional distribution (above cutoff) to the functional distribution
(below cutoff).

---

## Environment Setup

```bash
conda create -n clinical_env python=3.11 -y
conda activate clinical_env

pip install pandas>=1.5 numpy>=1.23 matplotlib>=3.6 scipy>=1.9

python -c "import pandas, numpy, matplotlib, scipy; print('All OK')"
```

---

## Core Workflow

### Step 1 — Questionnaire Scoring

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from typing import Optional, Dict, List, Tuple, Union

# ── Published reliability coefficients (test-retest r_tt) ──────────────────
SCALE_RELIABILITY = {
    "PHQ-9":  0.84,  # Kroenke et al., 2001
    "GAD-7":  0.83,  # Spitzer et al., 2006
    "PCL-5":  0.82,  # Blevins et al., 2015
    "BDI-II": 0.93,  # Beck et al., 1996
    "SWLS":   0.82,  # Diener et al., 1985
}

# Published SD for RCI computation (normative or clinical samples)
SCALE_NORM_SD = {
    "PHQ-9":  7.1,
    "GAD-7":  5.6,
    "PCL-5":  22.0,
    "BDI-II": 12.7,
    "SWLS":   6.4,
}

# Severity cutoffs: list of (max_score_inclusive, label) tuples
SEVERITY_CUTOFFS = {
    "PHQ-9":  [(4, "minimal"), (9, "mild"), (14, "moderate"), (27, "severe")],
    "GAD-7":  [(4, "minimal"), (9, "mild"), (14, "moderate"), (21, "severe")],
    "PCL-5":  [(32, "below_threshold"), (80, "provisional_PTSD")],
    "BDI-II": [(13, "minimal"), (19, "mild"), (28, "moderate"), (63, "severe")],
    "SWLS":   [(9, "extremely_dissatisfied"), (14, "dissatisfied"),
               (19, "slightly_below_avg"), (24, "average"),
               (29, "high"), (34, "very_high"), (35, "extremely_satisfied")],
}

# Clinical cutoffs for RCI clinical significance classification
CLINICAL_CUTOFF = {
    "PHQ-9":  10,   # moderate threshold
    "GAD-7":  10,
    "PCL-5":  33,
    "BDI-II": 20,
    "SWLS":   20,   # below average
}


def apply_severity_cutoff(score: float, scale: str) -> str:
    """Apply severity classification based on published cutoffs."""
    cutoffs = SEVERITY_CUTOFFS.get(scale, [])
    for max_val, label in cutoffs:
        if score <= max_val:
            return label
    return "unknown"


def score_phq9(
    df: pd.DataFrame,
    item_cols: Optional[List[str]] = None,
    id_col: str = "participant_id",
    allow_missing: int = 1,
) -> pd.DataFrame:
    """
    Score PHQ-9 depression questionnaire.

    Items are scored 0 (not at all) to 3 (nearly every day).
    Prorated scoring: if ≤ allow_missing items are missing, impute with
    the mean of valid items × 9.

    Args:
        df:           DataFrame with one row per participant.
        item_cols:    List of 9 PHQ-9 item column names (in order).
                      Defaults to ['phq1',...,'phq9'].
        id_col:       Participant ID column.
        allow_missing: Max missing items allowed for prorated scoring.

    Returns:
        DataFrame with PHQ-9 total, severity, and item-level flags.
    """
    if item_cols is None:
        item_cols = [f"phq{i}" for i in range(1, 10)]

    assert len(item_cols) == 9, "PHQ-9 requires exactly 9 item columns."

    result = df[[id_col]].copy() if id_col in df.columns else df.iloc[:, :1].copy()
    result.columns = [id_col]

    scores = []
    severities = []
    n_missing_list = []

    for _, row in df.iterrows():
        item_vals = row[item_cols]
        n_valid = item_vals.notna().sum()
        n_missing = 9 - n_valid

        if n_missing > allow_missing:
            total = np.nan
            severity = "missing"
        elif n_missing == 0:
            total = item_vals.sum()
            severity = apply_severity_cutoff(total, "PHQ-9")
        else:
            # Prorated: (sum of valid items / n_valid) × 9
            total = round((item_vals.sum() / n_valid) * 9)
            severity = apply_severity_cutoff(total, "PHQ-9")

        scores.append(total)
        severities.append(severity)
        n_missing_list.append(n_missing)

    result["PHQ9_total"] = scores
    result["PHQ9_severity"] = severities
    result["PHQ9_n_missing"] = n_missing_list
    result["PHQ9_item9_suicidal"] = df[item_cols[8]].values  # item 9 = suicidality

    return result


def score_gad7(
    df: pd.DataFrame,
    item_cols: Optional[List[str]] = None,
    id_col: str = "participant_id",
    allow_missing: int = 1,
) -> pd.DataFrame:
    """
    Score GAD-7 anxiety questionnaire (7 items, 0–3 each, range 0–21).

    Args:
        df:           DataFrame with one row per participant.
        item_cols:    List of 7 GAD-7 item column names.
        id_col:       Participant ID column.
        allow_missing: Max missing items for prorated scoring.

    Returns:
        DataFrame with GAD-7 total and severity classification.
    """
    if item_cols is None:
        item_cols = [f"gad{i}" for i in range(1, 8)]

    assert len(item_cols) == 7

    result = df[[id_col]].copy() if id_col in df.columns else df.iloc[:, :1].copy()
    result.columns = [id_col]

    scores, severities, n_missing_list = [], [], []

    for _, row in df.iterrows():
        vals = row[item_cols]
        n_valid = vals.notna().sum()
        n_missing = 7 - n_valid

        if n_missing > allow_missing:
            total, severity = np.nan, "missing"
        elif n_missing == 0:
            total = vals.sum()
            severity = apply_severity_cutoff(total, "GAD-7")
        else:
            total = round((vals.sum() / n_valid) * 7)
            severity = apply_severity_cutoff(total, "GAD-7")

        scores.append(total)
        severities.append(severity)
        n_missing_list.append(n_missing)

    result["GAD7_total"] = scores
    result["GAD7_severity"] = severities
    result["GAD7_n_missing"] = n_missing_list
    return result


def score_pcl5(
    df: pd.DataFrame,
    item_cols: Optional[List[str]] = None,
    id_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Score PCL-5 PTSD checklist (20 items, 0–4 each, range 0–80).
    Computes total score and DSM-5 cluster subscores (B/C/D/E).

    Args:
        df:        DataFrame with one row per participant.
        item_cols: List of 20 PCL-5 item column names.
        id_col:    Participant ID column.

    Returns:
        DataFrame with total, cluster subscores, and provisional PTSD flag.
    """
    if item_cols is None:
        item_cols = [f"pcl{i}" for i in range(1, 21)]

    assert len(item_cols) == 20

    result = df[[id_col]].copy() if id_col in df.columns else df.iloc[:, :1].copy()
    result.columns = [id_col]

    clusters = {
        "B_intrusion":       item_cols[0:5],
        "C_avoidance":       item_cols[5:7],
        "D_neg_cognition":   item_cols[7:14],
        "E_hyperarousal":    item_cols[14:20],
    }

    result["PCL5_total"] = df[item_cols].sum(axis=1)
    for cluster_name, cols in clusters.items():
        result[f"PCL5_{cluster_name}"] = df[cols].sum(axis=1)

    result["PCL5_provisional_PTSD"] = result["PCL5_total"] >= 33
    result["PCL5_severity"] = result["PCL5_total"].apply(
        lambda x: apply_severity_cutoff(x, "PCL-5")
    )
    return result
```

### Step 2 — Reliable Change Index

```python
def compute_rci(
    pre_scores: np.ndarray,
    post_scores: np.ndarray,
    scale: str,
    sd_pre: Optional[float] = None,
    r_tt: Optional[float] = None,
    clinical_cutoff: Optional[float] = None,
    dysfunctional_above_cutoff: bool = True,
) -> pd.DataFrame:
    """
    Compute Reliable Change Index (RCI) for pre-post score pairs.

    Classification (Jacobson & Truax, 1991):
        Recovered:    Reliable improvement AND crossed clinical cutoff
        Improved:     Reliable improvement only
        Unchanged:    No reliable change (|RCI| < 1.96)
        Deteriorated: Reliable worsening (RCI <= -1.96)

    Args:
        pre_scores:   Array of pre-treatment scores.
        post_scores:  Array of post-treatment scores.
        scale:        Scale name (e.g., 'PHQ-9') for default SD and r_tt lookup.
        sd_pre:       SD of pre-treatment scores (overrides norm SD if provided).
        r_tt:         Test-retest reliability (overrides default if provided).
        clinical_cutoff: Score threshold for functional vs dysfunctional range.
        dysfunctional_above_cutoff: True if high scores = clinical (PHQ-9, GAD-7);
                                    False if low scores = clinical (SWLS).

    Returns:
        DataFrame with RCI, classification, and pre/post severity.
    """
    sd = sd_pre if sd_pre is not None else SCALE_NORM_SD.get(scale, 10.0)
    rtt = r_tt if r_tt is not None else SCALE_RELIABILITY.get(scale, 0.85)
    cutoff = clinical_cutoff if clinical_cutoff is not None else CLINICAL_CUTOFF.get(scale, None)

    se_diff = sd * np.sqrt(2) * np.sqrt(1 - rtt)

    rci_values = (post_scores - pre_scores) / se_diff

    classifications = []
    for rci_val, pre, post in zip(rci_values, pre_scores, post_scores):
        if rci_val <= -1.96:  # reliable improvement (lower score = better)
            if cutoff is not None:
                if dysfunctional_above_cutoff:
                    crossed = pre >= cutoff and post < cutoff
                else:
                    crossed = pre < cutoff and post >= cutoff
                cls = "Recovered" if crossed else "Improved"
            else:
                cls = "Improved"
        elif rci_val >= 1.96:  # reliable worsening
            cls = "Deteriorated"
        else:
            cls = "Unchanged"
        classifications.append(cls)

    result_df = pd.DataFrame({
        "pre_score": pre_scores,
        "post_score": post_scores,
        "change": post_scores - pre_scores,
        "RCI": np.round(rci_values, 3),
        "classification": classifications,
        "pre_severity": [apply_severity_cutoff(s, scale) for s in pre_scores],
        "post_severity": [apply_severity_cutoff(s, scale) for s in post_scores],
    })

    # Summary statistics
    class_counts = result_df["classification"].value_counts()
    n = len(result_df)
    print(f"\nRCI Summary for {scale} (SE_diff = {se_diff:.2f}, r_tt = {rtt}):")
    for cls in ["Recovered", "Improved", "Unchanged", "Deteriorated"]:
        count = class_counts.get(cls, 0)
        print(f"  {cls:15s}: n = {count:3d} ({count/n:.1%})")

    return result_df
```

### Step 3 — Longitudinal Visualization

```python
def plot_longitudinal_trajectories(
    df_long: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    id_col: str,
    scale_name: str = "",
    highlight_clinical_cutoff: Optional[float] = None,
    output_path: Optional[str] = None,
    n_highlight: int = 5,
) -> plt.Figure:
    """
    Plot individual longitudinal trajectories with group mean overlay.

    Args:
        df_long:                  Long-format DataFrame (one row per person × time).
        outcome_col:              Score column.
        time_col:                 Time point column (numeric or ordered categorical).
        id_col:                   Participant ID column.
        scale_name:               Scale label for plot title and y-axis.
        highlight_clinical_cutoff: Draw a horizontal reference line at this score.
        output_path:              Optional path to save figure.
        n_highlight:              Number of individual trajectories to highlight.

    Returns:
        Matplotlib Figure.
    """
    time_points = sorted(df_long[time_col].unique())
    persons = df_long[id_col].unique()
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(10, 6))

    # All individual trajectories (light gray)
    for person in persons:
        pdata = df_long[df_long[id_col] == person].sort_values(time_col)
        ax.plot(pdata[time_col], pdata[outcome_col],
                color="lightgray", linewidth=0.8, alpha=0.5, zorder=1)

    # Highlighted individuals
    highlighted = rng.choice(persons, min(n_highlight, len(persons)), replace=False)
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(highlighted)))
    for person, color in zip(highlighted, colors):
        pdata = df_long[df_long[id_col] == person].sort_values(time_col)
        ax.plot(pdata[time_col], pdata[outcome_col],
                color=color, linewidth=2.0, alpha=0.9, zorder=3,
                label=f"{id_col}={person}")

    # Group mean trajectory
    group_mean = df_long.groupby(time_col)[outcome_col].agg(["mean", "sem"]).reset_index()
    ax.plot(group_mean[time_col], group_mean["mean"],
            color="black", linewidth=3, zorder=4, label="Group mean")
    ax.fill_between(
        group_mean[time_col],
        group_mean["mean"] - 1.96 * group_mean["sem"],
        group_mean["mean"] + 1.96 * group_mean["sem"],
        alpha=0.15, color="black", zorder=2,
    )

    # Clinical cutoff line
    if highlight_clinical_cutoff is not None:
        ax.axhline(highlight_clinical_cutoff, color="crimson", linestyle="--",
                   linewidth=1.5, label=f"Clinical cutoff ({highlight_clinical_cutoff})")

    ax.set_xlabel("Time Point")
    ax.set_ylabel(scale_name or outcome_col)
    ax.set_title(f"Longitudinal Trajectories — {scale_name}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.show()
    return fig
```

---

## Advanced Usage

### Batch Scoring with PHQ-9 + GAD-7 Combined

```python
def batch_score_all_scales(
    df: pd.DataFrame,
    id_col: str = "participant_id",
    phq9_items: Optional[List[str]] = None,
    gad7_items: Optional[List[str]] = None,
    pcl5_items: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Score PHQ-9, GAD-7, and PCL-5 in one call and merge results.

    Args:
        df:           Wide-format DataFrame.
        id_col:       Participant ID column.
        phq9_items:   PHQ-9 item columns (defaults to phq1–phq9).
        gad7_items:   GAD-7 item columns (defaults to gad1–gad7).
        pcl5_items:   PCL-5 item columns (defaults to pcl1–pcl20).

    Returns:
        DataFrame with all scale scores merged on id_col.
    """
    phq9_df = score_phq9(df, item_cols=phq9_items, id_col=id_col)
    gad7_df = score_gad7(df, item_cols=gad7_items, id_col=id_col)
    pcl5_df = score_pcl5(df, item_cols=pcl5_items, id_col=id_col)

    merged = phq9_df.merge(gad7_df, on=id_col).merge(pcl5_df, on=id_col)

    # Comorbidity flag
    merged["comorbid_dep_anx"] = (
        (merged["PHQ9_total"] >= 10) & (merged["GAD7_total"] >= 10)
    )

    print(f"\nBatch scoring complete: {len(merged)} participants")
    print(f"PHQ-9 ≥ 10 (moderate+): {(merged['PHQ9_total'] >= 10).sum()}")
    print(f"GAD-7 ≥ 10 (moderate+): {(merged['GAD7_total'] >= 10).sum()}")
    print(f"PCL-5 ≥ 33 (provisional PTSD): {merged['PCL5_provisional_PTSD'].sum()}")
    print(f"Comorbid depression+anxiety: {merged['comorbid_dep_anx'].sum()}")
    return merged
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| Negative RCI for "worsening" with PHQ-9 | Convention: lower = better | Check `rci_val >= 1.96` means deterioration |
| All classified as "Unchanged" | SD too large or wrong scale | Use clinical sample SD, not general population |
| Prorated score unexpectedly high | All non-missing items are high | Expected behavior; flag if > 2 items missing |
| PCL-5 cluster subscores don't sum to total | Rounding or missing items | Ensure no NaN; use `sum(axis=1, min_count=20)` |
| Longitudinal plot illegible | Too many participants | Reduce `n_highlight` or use mean + CI only |
| NaN in severity column | Score is NaN (too many missing) | Apply `dropna()` before severity lookup |

---

## External Resources

- Kroenke, K., Spitzer, R. L., & Williams, J. B. W. (2001). The PHQ-9.
  *Journal of General Internal Medicine*, 16(9), 606–613.
- Spitzer, R. L., et al. (2006). A brief measure for assessing GAD. *JAMA Internal Medicine.*
- Blevins, C. A., et al. (2015). PCL-5: Initial psychometric assessment.
  *Assessment*, 22(5), 477–482.
- Jacobson, N. S., & Truax, P. (1991). Clinical significance. *Journal of Consulting
  and Clinical Psychology*, 59(1), 12–19.
- Diener, E., et al. (1985). The Satisfaction with Life Scale. *Journal of Personality
  Assessment*, 49(1), 71–75.

---

## Examples

### Example 1 — Batch PHQ-9 + GAD-7 Scoring with Cutoff Classification

```python
import pandas as pd
import numpy as np

# Simulate participant data
rng = np.random.default_rng(0)
n = 80
data = {
    "participant_id": [f"P{i:03d}" for i in range(1, n + 1)],
}
# PHQ-9 items (0–3)
for i in range(1, 10):
    data[f"phq{i}"] = rng.integers(0, 4, n)
# GAD-7 items (0–3)
for i in range(1, 8):
    data[f"gad{i}"] = rng.integers(0, 4, n)
# PCL-5 items (0–4)
for i in range(1, 21):
    data[f"pcl{i}"] = rng.integers(0, 5, n)

# Introduce some missing values
data["phq3"][5] = np.nan
data["gad2"][12] = np.nan

df_raw = pd.DataFrame(data)

# Score all scales
df_scored = batch_score_all_scales(
    df_raw,
    id_col="participant_id",
    phq9_items=[f"phq{i}" for i in range(1, 10)],
    gad7_items=[f"gad{i}" for i in range(1, 8)],
    pcl5_items=[f"pcl{i}" for i in range(1, 21)],
)

print("\nSample of scored data:")
print(df_scored[["participant_id", "PHQ9_total", "PHQ9_severity",
                  "GAD7_total", "GAD7_severity", "PCL5_total"]].head(10))

# Severity distribution
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (col, title) in zip(axes, [
    ("PHQ9_severity", "PHQ-9 Severity"),
    ("GAD7_severity", "GAD-7 Severity"),
    ("PCL5_severity", "PCL-5 Severity"),
]):
    counts = df_scored[col].value_counts()
    ax.bar(counts.index, counts.values, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
plt.savefig("severity_distributions.png", dpi=150)
plt.show()
```

### Example 2 — RCI and Clinical Significance Classification

```python
# Simulate pre-post treatment data
rng = np.random.default_rng(1)
n_pts = 60
pre = rng.normal(16, 6, n_pts).clip(0, 27).round()  # moderate depression
post = (pre - rng.normal(5, 4, n_pts)).clip(0, 27).round()  # treatment effect

rci_df = compute_rci(
    pre_scores=pre,
    post_scores=post,
    scale="PHQ-9",
    clinical_cutoff=10,
    dysfunctional_above_cutoff=True,
)

print("\nRCI results sample:")
print(rci_df.head(10).to_string())

# Visualize classification
class_counts = rci_df["classification"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors_map = {"Recovered": "#2ecc71", "Improved": "#3498db",
              "Unchanged": "#f39c12", "Deteriorated": "#e74c3c"}
bars = axes[0].bar(class_counts.index,
                   class_counts.values,
                   color=[colors_map.get(c, "gray") for c in class_counts.index])
axes[0].set_title("Treatment Response Classification")
axes[0].set_ylabel("Number of Participants")
for bar, count in zip(bars, class_counts.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha="center", va="bottom", fontsize=10)

# Scatter pre vs post with RCI classification
scatter_colors = [colors_map.get(c, "gray") for c in rci_df["classification"]]
axes[1].scatter(rci_df["pre_score"], rci_df["post_score"],
                c=scatter_colors, alpha=0.7, s=40)
axes[1].plot([0, 27], [0, 27], "k--", linewidth=1, label="No change")
axes[1].axhline(10, color="crimson", linestyle=":", linewidth=1, label="Cutoff post")
axes[1].axvline(10, color="crimson", linestyle=":", linewidth=1, label="Cutoff pre")
axes[1].set_xlabel("Pre-treatment PHQ-9")
axes[1].set_ylabel("Post-treatment PHQ-9")
axes[1].set_title("RCI Scatter Plot")
legend_patches = [mpatches.Patch(color=c, label=l) for l, c in colors_map.items()]
axes[1].legend(handles=legend_patches, fontsize=8)

import matplotlib.patches as mpatches
fig.tight_layout()
plt.savefig("rci_analysis.png", dpi=150)
plt.show()
print("Clinical assessment analysis complete.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — PHQ-9/GAD-7/PCL-5 scoring, RCI, clinical significance, longitudinal plots |
