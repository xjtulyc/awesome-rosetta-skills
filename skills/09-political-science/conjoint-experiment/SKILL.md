---
name: conjoint-experiment
description: >
  Use this Skill for conjoint survey experiments: profile design, AMCE estimation via OLS,
  marginal means, interaction effects, and respondent-level heterogeneity.
tags:
  - political-science
  - conjoint-experiment
  - AMCE
  - survey-experiment
  - causal-inference
version: "1.0.0"
authors:
  - name: "awesome-rosetta-skills contributors"
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - pandas>=1.5
  - statsmodels>=0.14
  - numpy>=1.23
  - matplotlib>=3.6
last_updated: "2026-03-18"
status: "stable"
---

# Conjoint Survey Experiments

## When to Use

Use this skill when you need to:

- Design conjoint survey tasks (attributes, levels, profiles) with randomized full or partial
  factorial profile generation
- Estimate Average Marginal Component Effects (AMCEs) via linear probability model (OLS) with
  standard errors clustered by respondent
- Compute marginal means (unconditional on other attributes) as an alternative AMCE summary
- Estimate interaction AMCEs to test whether attribute effects vary by other profile attributes
- Analyze respondent-level heterogeneity in AMCEs (subgroup by party ID, ideology, demographics)
- Detect inattentive respondents through straightlining, response time, and consistency checks
- Power-analyze a conjoint study (tasks × respondents × attributes)

Conjoint experiments are used in political science to measure candidate evaluation, immigration
preferences, trade-off tolerance for policy packages, and partisan asymmetries.

## Background

**Conjoint design**: Respondents see multiple tasks. In each task, they compare two (or more)
profiles that vary across predefined attributes. Each attribute has a set of levels. Levels are
randomly assigned to profiles independently across attributes (full factorial) or according to a
restricted design (partial factorial, to avoid implausible combinations).

**AMCE (Average Marginal Component Effect)**: The AMCE for level `l` of attribute `A` is the
average difference in choice probability when `A = l` compared to a baseline level, averaging
over all possible randomizations of other attributes. Because profiles are randomized, the
estimator is simply OLS:

```
choice_ij = μ + Σ_A Σ_l β_{A,l} × 1[attribute_A = l] + ε_ij
```

where `i` indexes tasks and `j` respondents. Standard errors clustered by respondent account
for within-respondent correlation across tasks.

**Marginal means** (Leeper, Hobolt & Tilley 2020) measure the expected choice probability when
an attribute takes a given level, averaged over all other attributes — without conditioning on a
reference level. This avoids interpretational dependence on the baseline choice.

**Interaction AMCE**: When an attribute's effect varies by another attribute (or a respondent
characteristic), the interaction is estimated by including the product term in the OLS model.

**Subgroup AMCE**: Splitting the sample by party ID (or ideology) and re-estimating AMCE per
subgroup tests partisan asymmetry in attribute preferences (common in immigration and candidate
evaluation conjoint studies).

**Power**: For a binary forced-choice conjoint, the minimum detectable effect (MDE) for an AMCE
depends on:
- Number of tasks per respondent (T)
- Number of respondents (N)
- Number of levels per attribute (K) — more levels means each level appears less often

Rule of thumb (Bansak et al. 2021): N × T ≥ 5,000 tasks total for MDE ≈ 0.04.

## Environment Setup

```bash
pip install pandas>=1.5 statsmodels>=0.14 numpy>=1.23 matplotlib>=3.6
```

No external data downloads required. Conjoint datasets are typically generated during survey
programming (Qualtrics, Formr) and exported as CSV. Store file path as:

```bash
export CONJOINT_DATA_PATH="/data/conjoint_survey_results.csv"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Conjoint Profile Generator
# ---------------------------------------------------------------------------

def generate_conjoint_profiles(
    attributes: dict[str, list],
    n_respondents: int,
    n_tasks: int = 5,
    n_profiles: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a conjoint dataset with randomly assigned profile attributes.

    Parameters
    ----------
    attributes : dict
        {attribute_name: [level1, level2, ...]}
        Example: {'age': ['35', '45', '55'], 'gender': ['Man', 'Woman']}
    n_respondents : int
        Number of survey respondents.
    n_tasks : int
        Number of forced-choice tasks per respondent.
    n_profiles : int
        Number of profiles per task (typically 2 for forced-choice).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with one row per profile:
    respondent_id, task_id, profile_id, chosen (0/1), + one column per attribute.
    """
    rng = np.random.default_rng(seed)
    attr_names = list(attributes.keys())
    rows = []

    for resp_id in range(1, n_respondents + 1):
        for task_id in range(1, n_tasks + 1):
            # Randomly assign levels to each profile
            profile_levels = {}
            for attr, levels in attributes.items():
                profile_levels[attr] = rng.choice(levels, size=n_profiles, replace=True)

            # Simulate choice based on hidden utility (for testing)
            # In real data, this comes from respondent answers
            chosen_profile = rng.integers(0, n_profiles)

            for prof_id in range(n_profiles):
                row = {
                    "respondent_id": resp_id,
                    "task_id": task_id,
                    "profile_id": prof_id + 1,
                    "chosen": int(prof_id == chosen_profile),
                }
                for attr in attr_names:
                    row[attr] = profile_levels[attr][prof_id]
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. AMCE Estimation via OLS
# ---------------------------------------------------------------------------

def estimate_amce(
    df: pd.DataFrame,
    outcome: str = "chosen",
    attributes: list[str] | None = None,
    reference_levels: dict[str, str] | None = None,
    cluster_var: str = "respondent_id",
) -> pd.DataFrame:
    """
    Estimate Average Marginal Component Effects (AMCE) via linear probability model.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format conjoint dataset (one row per profile).
    outcome : str
        Binary choice indicator (0/1).
    attributes : list of str
        Attribute column names. Inferred from reference_levels keys if not provided.
    reference_levels : dict
        {attribute: reference_level} — the level dropped (baseline).
        If None, the first alphabetical level is used as baseline.
    cluster_var : str
        Variable for clustering standard errors (typically respondent_id).

    Returns
    -------
    pd.DataFrame with AMCE estimates: attribute, level, amce, se, ci_lower, ci_upper, p.
    """
    reference_levels = reference_levels or {}
    attributes = attributes or [c for c in df.columns if c not in
                                  [outcome, cluster_var, "task_id", "profile_id"]]

    # Encode dummies for each attribute
    dummy_frames = []
    for attr in attributes:
        ref = reference_levels.get(attr, sorted(df[attr].unique())[0])
        dummies = pd.get_dummies(df[attr], prefix=attr, drop_first=False)
        ref_col = f"{attr}_{ref}"
        if ref_col in dummies.columns:
            dummies = dummies.drop(columns=[ref_col])
        dummy_frames.append(dummies)

    X_df = pd.concat(dummy_frames, axis=1).astype(float)
    X = sm.add_constant(X_df.reset_index(drop=True))
    y = df[outcome].reset_index(drop=True).astype(float)
    groups = df[cluster_var].reset_index(drop=True)

    model = sm.OLS(y, X)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})

    rows = []
    for col in X_df.columns:
        parts = col.split("_", 1)
        attr = parts[0]
        level = parts[1] if len(parts) > 1 else col
        idx = col
        if idx not in result.params.index:
            continue
        coef = result.params[idx]
        se = result.bse[idx]
        rows.append({
            "attribute": attr,
            "level": level,
            "amce": round(coef, 5),
            "se": round(se, 5),
            "ci_lower": round(coef - 1.96 * se, 5),
            "ci_upper": round(coef + 1.96 * se, 5),
            "p": round(result.pvalues[idx], 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Marginal Means
# ---------------------------------------------------------------------------

def marginal_means(
    df: pd.DataFrame,
    outcome: str = "chosen",
    attributes: list[str] | None = None,
    weight_var: str | None = None,
) -> pd.DataFrame:
    """
    Compute marginal means: weighted average choice probability per attribute level.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
    attributes : list of str
    weight_var : str, optional
        Survey weight column.

    Returns
    -------
    pd.DataFrame with attribute, level, marginal_mean, se, n.
    """
    attributes = attributes or [c for c in df.columns if c not in
                                  [outcome, weight_var or "__none__", "respondent_id",
                                   "task_id", "profile_id"]]
    rows = []
    for attr in attributes:
        for level, grp in df.groupby(attr):
            y = grp[outcome].astype(float)
            if weight_var and weight_var in grp.columns:
                w = grp[weight_var].fillna(1.0)
                mm = np.average(y, weights=w)
            else:
                mm = y.mean()
            se = y.sem()
            rows.append({
                "attribute": attr, "level": str(level),
                "marginal_mean": round(mm, 5),
                "se": round(se, 5),
                "n": len(grp),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. AMCE Coefficient Plot
# ---------------------------------------------------------------------------

def plot_amce(
    amce_df: pd.DataFrame,
    title: str = "AMCE Estimates",
    figsize: tuple = (9, 7),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal forest plot of AMCE estimates with 95% CI.

    Parameters
    ----------
    amce_df : pd.DataFrame
        Output of estimate_amce().
    title : str
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib Figure.
    """
    df_plot = amce_df.copy()
    df_plot["label"] = df_plot["attribute"] + ": " + df_plot["level"]
    df_plot = df_plot.sort_values(["attribute", "amce"], ascending=[True, False])

    colors = plt.cm.tab10.colors
    attr_list = df_plot["attribute"].unique().tolist()
    attr_color = {attr: colors[i % 10] for i, attr in enumerate(attr_list)}

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(df_plot))
    row_colors = [attr_color[attr] for attr in df_plot["attribute"]]

    ax.errorbar(
        df_plot["amce"], list(y_pos),
        xerr=[df_plot["amce"] - df_plot["ci_lower"], df_plot["ci_upper"] - df_plot["amce"]],
        fmt="o", color="black", elinewidth=1.2, capsize=3, markersize=6, zorder=3
    )
    # Color-code by attribute
    ax.scatter(df_plot["amce"], list(y_pos), color=row_colors, s=50, zorder=4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_plot["label"].tolist(), fontsize=9)
    ax.set_xlabel("AMCE (change in Pr(chosen))")
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis="x", alpha=0.3)

    # Legend
    patches = [mpatches.Patch(color=attr_color[a], label=a) for a in attr_list]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 5. Subgroup AMCE Heterogeneity
# ---------------------------------------------------------------------------

def subgroup_amce(
    df: pd.DataFrame,
    subgroup_var: str,
    subgroup_values: list,
    outcome: str = "chosen",
    attributes: list[str] | None = None,
    reference_levels: dict[str, str] | None = None,
    cluster_var: str = "respondent_id",
) -> dict[str, pd.DataFrame]:
    """
    Estimate AMCE separately for each subgroup value.

    Parameters
    ----------
    df : pd.DataFrame
    subgroup_var : str
        Column for splitting (e.g., 'party_id').
    subgroup_values : list
        Values to include (e.g., ['Democrat', 'Republican']).

    Returns
    -------
    dict {subgroup_value: amce_df}
    """
    results = {}
    for val in subgroup_values:
        sub = df[df[subgroup_var] == val].copy()
        if len(sub) < 50:
            print(f"Warning: subgroup {val} has only {len(sub)} observations.")
            continue
        amce_df = estimate_amce(
            sub, outcome=outcome, attributes=attributes,
            reference_levels=reference_levels, cluster_var=cluster_var
        )
        results[val] = amce_df
    return results
```

## Advanced Usage

### Inattentive Respondent Detection

```python
import numpy as np
import pandas as pd


def detect_inattentive_respondents(
    df: pd.DataFrame,
    respondent_col: str = "respondent_id",
    chosen_col: str = "chosen",
    time_col: str | None = None,
    time_threshold_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Flag respondents for inattention via straightlining detection.

    Straightlining: always choosing the first (or second) profile across all tasks.
    Fast responders: completing tasks faster than a minimum plausible threshold.

    Parameters
    ----------
    df : pd.DataFrame
    respondent_col : str
    chosen_col : str
    time_col : str, optional
        Column for task completion time in seconds.
    time_threshold_seconds : float
        Minimum plausible time per task.

    Returns
    -------
    pd.DataFrame with respondent_id, n_tasks, always_chose_first, fast_responder, flag.
    """
    # Profile 1 corresponds to the first profile shown in each task
    # Assumes profile_id column exists
    results = []
    for resp_id, grp in df.groupby(respondent_col):
        tasks = grp.groupby("task_id")
        n_tasks = len(tasks)
        choices_first = 0
        for tid, task in tasks:
            first_profile = task["profile_id"].min()
            chosen_profile_id = task.loc[task[chosen_col] == 1, "profile_id"]
            if not chosen_profile_id.empty and chosen_profile_id.iloc[0] == first_profile:
                choices_first += 1

        always_first = (choices_first == n_tasks)
        fast_resp = False
        if time_col and time_col in grp.columns:
            avg_time = grp[time_col].mean()
            fast_resp = avg_time < time_threshold_seconds

        results.append({
            "respondent_id": resp_id,
            "n_tasks": n_tasks,
            "pct_chose_first": round(choices_first / n_tasks, 3),
            "always_chose_first": always_first,
            "fast_responder": fast_resp,
            "flag": always_first or fast_resp,
        })
    return pd.DataFrame(results)


def conjoint_power(
    n_respondents: int,
    n_tasks: int,
    effect_size: float = 0.05,
    alpha: float = 0.05,
) -> dict:
    """
    Approximate power for conjoint AMCE detection.

    Uses the normal approximation: power = Φ(|β| / SE - z_{α/2}).
    SE ≈ 1 / sqrt(N * T / K) where K = 2 profiles per task (binary choice).

    Parameters
    ----------
    n_respondents : int
    n_tasks : int
    effect_size : float
        AMCE to detect (e.g., 0.05 = 5 percentage points).
    alpha : float
        Significance level.

    Returns
    -------
    dict with power, total_tasks, se_approx, mde.
    """
    from scipy.stats import norm
    total_tasks = n_respondents * n_tasks
    se_approx = 1 / np.sqrt(total_tasks / 2)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = abs(effect_size) / se_approx - z_alpha
    power = norm.cdf(z_beta)
    mde = z_alpha * se_approx * 2
    return {
        "power": round(power, 4),
        "total_tasks": total_tasks,
        "se_approx": round(se_approx, 5),
        "mde_90pct": round(mde * 1.28 / z_alpha, 5),
    }
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| AMCE confidence intervals too wide | Too few tasks or respondents | Target ≥ 5,000 total profile observations; add tasks before respondents |
| Interaction AMCE is insignificant | Underpowered for interaction | Interactions require ~4× the observations; pre-specify if confirmatory |
| Marginal means sum to ≠ 0.5 | Unequal number of levels per attribute | This is expected; MM interpretation is probability, not contrast |
| Respondent clustering ignored | Standard errors too small | Always use `cov_type='cluster'` with groups = respondent_id |
| Profile imbalance | Non-random level assignment in survey software | Check that each level appears ~equally often using `df[attr].value_counts()` |

## External Resources

- Hainmueller, J., Hopkins, D. & Yamamoto, T. (2014). Causal inference in conjoint analysis.
  *Political Analysis*, 22(1), 1-30.
- Leeper, T.J., Hobolt, S.B. & Tilley, J. (2020). Measuring subgroup preferences in conjoint
  experiments. *Political Analysis*, 28(2), 207-221.
- Bansak, K. et al. (2021). Using conjoint experiments to analyze election outcomes.
  *Political Analysis*, 29(3), 380-395.
- Abramson, S.F., Kocak, K. & Magazinnik, A. (2022). What do we learn about voter preferences
  from conjoint experiments? *American Journal of Political Science*.
- cregg R package (reference implementation): https://github.com/leeper/cregg

## Examples

### Example 1: Conjoint Dataset Simulation + AMCE OLS

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define attributes and levels for a candidate evaluation conjoint
ATTRIBUTES = {
    "age":         ["35", "45", "55", "65"],
    "gender":      ["Man", "Woman"],
    "education":   ["High School", "College", "Graduate"],
    "experience":  ["None", "Local", "State", "Federal"],
    "party":       ["Democrat", "Republican", "Independent"],
    "issue_pos":   ["Progressive", "Moderate", "Conservative"],
}

# Generate dataset (1200 respondents, 5 tasks each)
df_conjoint = generate_conjoint_profiles(
    attributes=ATTRIBUTES,
    n_respondents=1200,
    n_tasks=5,
    n_profiles=2,
    seed=42,
)

# Inject a realistic preference signal: women, Democrats preferred
rng = np.random.default_rng(123)
pref_cols = ["gender", "party", "experience"]
# Re-draw chosen based on utility
for resp_id, resp_grp in df_conjoint.groupby("respondent_id"):
    for task_id, task_grp in resp_grp.groupby("task_id"):
        idx = task_grp.index
        if len(idx) < 2:
            continue
        # Simple utility: woman +0.15, Democrat +0.20, Federal exp +0.15
        utils = []
        for i in idx:
            u = (0.15 * (df_conjoint.loc[i, "gender"] == "Woman") +
                 0.20 * (df_conjoint.loc[i, "party"] == "Democrat") +
                 0.15 * (df_conjoint.loc[i, "experience"] == "Federal") +
                 rng.gumbel(0, 1))
            utils.append(u)
        chosen_idx = idx[np.argmax(utils)]
        df_conjoint.loc[idx, "chosen"] = 0
        df_conjoint.loc[chosen_idx, "chosen"] = 1

print(f"Dataset size: {len(df_conjoint)} rows, {df_conjoint['respondent_id'].nunique()} respondents")

# Estimate AMCE
reference = {
    "age": "35", "gender": "Man", "education": "High School",
    "experience": "None", "party": "Democrat", "issue_pos": "Progressive",
}
amce_results = estimate_amce(df_conjoint, reference_levels=reference,
                               attributes=list(ATTRIBUTES.keys()))
print("=== AMCE Estimates ===")
print(amce_results.to_string(index=False))
```

### Example 2: AMCE Coefficient Plot

```python
import matplotlib.pyplot as plt

# Using amce_results from Example 1
fig = plot_amce(
    amce_results,
    title="Candidate Evaluation Conjoint — AMCE",
    figsize=(10, 9),
    save_path="amce_coefficient_plot.png",
)
plt.show()

# Also compute marginal means
mm = marginal_means(df_conjoint, attributes=list(ATTRIBUTES.keys()))
print("\n=== Marginal Means (top entries) ===")
print(mm.sort_values("marginal_mean", ascending=False).head(10).to_string(index=False))
```

### Example 3: Subgroup Heterogeneity by Party ID

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add respondent-level party ID
n_resp = df_conjoint["respondent_id"].nunique()
resp_party = pd.DataFrame({
    "respondent_id": range(1, n_resp + 1),
    "resp_party": np.random.default_rng(77).choice(
        ["Democrat", "Republican", "Independent"],
        n_resp, p=[0.38, 0.36, 0.26]
    ),
})
df_c2 = df_conjoint.merge(resp_party, on="respondent_id")

subgroup_results = subgroup_amce(
    df_c2,
    subgroup_var="resp_party",
    subgroup_values=["Democrat", "Republican"],
    attributes=["gender", "party", "experience", "issue_pos"],
    reference_levels={"gender": "Man", "party": "Democrat",
                      "experience": "None", "issue_pos": "Progressive"},
)

# Compare AMCE on party attribute across subgroups
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
colors = {"Democrat": "#1f77b4", "Republican": "#d62728"}
for ax, (sub_val, amce_sub) in zip(axes, subgroup_results.items()):
    sub_plot = amce_sub[amce_sub["attribute"] == "party"].copy()
    y_pos = range(len(sub_plot))
    ax.errorbar(
        sub_plot["amce"], list(y_pos),
        xerr=[sub_plot["amce"] - sub_plot["ci_lower"],
              sub_plot["ci_upper"] - sub_plot["amce"]],
        fmt="o", color=colors[sub_val], capsize=4, markersize=8, elinewidth=1.5,
    )
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub_plot["level"].tolist())
    ax.set_xlabel("AMCE")
    ax.set_title(f"{sub_val} Respondents — Party Attribute AMCE")
    ax.grid(True, axis="x", alpha=0.3)

plt.suptitle("Partisan Heterogeneity in Candidate Party Preferences", fontsize=13)
plt.tight_layout()
plt.savefig("subgroup_amce_party.png", dpi=150)
plt.show()

# Power analysis
power_result = conjoint_power(n_respondents=1200, n_tasks=5, effect_size=0.05)
print("\n=== Power Analysis ===")
for k, v in power_result.items():
    print(f"  {k}: {v}")
```
