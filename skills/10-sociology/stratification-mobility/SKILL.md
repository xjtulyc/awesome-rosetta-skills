---
name: stratification-mobility
description: >
  Use this Skill for social stratification: intergenerational income mobility (Chetty rank-rank),
  occupational prestige (ISEI), EGP class schema, and transition matrices.
tags:
  - sociology
  - stratification
  - intergenerational-mobility
  - occupational-prestige
  - inequality
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
  - scipy>=1.9
last_updated: "2026-03-18"
status: "stable"
---

# Social Stratification and Intergenerational Mobility

## When to Use

Use this skill when you need to:

- Estimate intergenerational income elasticity (IGE) via log-log OLS regression
- Estimate the rank-rank slope (Chetty et al. 2014 approach) for more robust mobility estimates
- Compute upward mobility rates — probability of reaching the top quintile given bottom quintile
  origins
- Load and use the ISEI (International Socio-Economic Index of Occupational Status) to score
  occupations on a continuous prestige scale
- Apply the EGP (Erikson-Goldthorpe-Portocarero) class schema to categorize occupations into
  hierarchical service/intermediate/working class groups
- Build and analyze intergenerational occupational transition matrices
- Perform odds ratio analysis for social fluidity (relative mobility)
- Decompose income variance by education and occupation contributions (Shapley decomposition)

## Background

**Intergenerational Elasticity (IGE)**: Regress log child income on log parent income:

```
log(income_child) = α + β × log(income_parent) + ε
```

β is the IGE. High β (close to 1) = low mobility (child income closely tracks parent income).
The US IGE is approximately 0.45; Nordic countries are closer to 0.15-0.25.

**Rank-Rank Slope**: Convert income to percentile ranks within cohort, then regress:

```
rank_child = α + ρ × rank_parent + ε
```

ρ (rank-rank slope) is less sensitive to outliers than IGE and better suited to censored or
top-coded income data. Chetty et al. (2014) found ρ ≈ 0.341 for the United States.

**Upward Mobility**: P(child in Q5 | parent in Q1) — fraction of children born in the bottom
income quintile who reach the top quintile as adults. This measure (sometimes called "Chetty
mobility") varies enormously across geographic areas and demographic groups.

**ISEI (Hauser & Warren 1997)**: An occupation scoring system derived from the regression of
income and education on the Standard Occupational Classification. Scores range from ~16 (lowest
manual) to ~90 (physicians, judges). Assigned from 4-digit ISCO or national occupation codes.

**EGP Schema (Erikson, Goldthorpe & Portocarero 1979)**: Categorical class schema:
- I: Higher professionals, administrators (service class)
- II: Lower professionals, technicians (service class)
- IIIa: Routine non-manual (intermediate)
- IIIb: Personal services (intermediate)
- IVa/IVb/IVc: Self-employed (petty bourgeoisie)
- V/VI: Skilled manual workers
- VIIa/VIIb: Unskilled manual, agricultural workers

**Transition matrix and social fluidity**: The (origin × destination) mobility table shows
probabilities of ending in each class given each origin class. The odds ratio:

```
OR(a,b;c,d) = (n_ac × n_bd) / (n_ad × n_bc)
```

where a,b are origin classes and c,d are destination classes. OR = 1 means equal relative
odds (perfect fluidity). Deviations from 1 indicate barriers to mobility.

## Environment Setup

```bash
pip install pandas>=1.5 statsmodels>=0.14 numpy>=1.23 matplotlib>=3.6 scipy>=1.9
```

PSID (Panel Study of Income Dynamics) data: https://psidonline.isr.umich.edu/
Register for free access. Download cross-year individual/family files.

```bash
export PSID_PATH="/data/psid_crossyear.csv"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. IGE and Rank-Rank Slope
# ---------------------------------------------------------------------------

def intergenerational_elasticity(
    parent_income: pd.Series,
    child_income: pd.Series,
) -> dict:
    """
    Estimate intergenerational income elasticity (log-log OLS).

    Parameters
    ----------
    parent_income, child_income : pd.Series
        Income series (positive values; zeros will be dropped).

    Returns
    -------
    dict with ige (beta), r_squared, n, intercept.
    """
    mask = (parent_income > 0) & (child_income > 0)
    lp = np.log(parent_income[mask])
    lc = np.log(child_income[mask])
    slope, intercept, r, p, se = stats.linregress(lp, lc)
    return {
        "ige": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r ** 2, 4),
        "se": round(se, 4),
        "p_value": round(p, 4),
        "n": int(mask.sum()),
    }


def rank_rank_slope(
    parent_income: pd.Series,
    child_income: pd.Series,
    bootstrap_n: int = 500,
    seed: int = 42,
) -> dict:
    """
    Estimate rank-rank slope with bootstrap confidence interval (Chetty method).

    Parameters
    ----------
    parent_income, child_income : pd.Series
    bootstrap_n : int
        Number of bootstrap replications for CI.
    seed : int

    Returns
    -------
    dict with rr_slope, intercept, ci_lower, ci_upper, se, n.
    """
    mask = parent_income.notna() & child_income.notna()
    pr = parent_income[mask].rank(pct=True)
    cr = child_income[mask].rank(pct=True)

    slope, intercept, r, p, se = stats.linregress(pr, cr)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_slopes = []
    n = len(pr)
    pr_arr, cr_arr = pr.values, cr.values
    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, n)
        s, *_ = stats.linregress(pr_arr[idx], cr_arr[idx])
        boot_slopes.append(s)
    ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

    return {
        "rr_slope": round(slope, 5),
        "intercept": round(intercept, 5),
        "ci_lower": round(ci_lo, 5),
        "ci_upper": round(ci_hi, 5),
        "se": round(np.std(boot_slopes), 5),
        "n": int(mask.sum()),
    }


def upward_mobility_rate(
    parent_income: pd.Series,
    child_income: pd.Series,
    n_quintiles: int = 5,
    from_quintile: int = 1,
    to_quintile: int = 5,
) -> dict:
    """
    Compute upward mobility: P(child in top quintile | parent in bottom quintile).

    Returns
    -------
    dict with p_upward, n_origin, n_both.
    """
    mask = parent_income.notna() & child_income.notna()
    pq = pd.qcut(parent_income[mask], n_quintiles, labels=False) + 1
    cq = pd.qcut(child_income[mask], n_quintiles, labels=False) + 1
    origin_mask = pq == from_quintile
    n_origin = origin_mask.sum()
    n_both = ((pq == from_quintile) & (cq == to_quintile)).sum()
    return {
        "p_upward": round(n_both / n_origin, 5) if n_origin > 0 else np.nan,
        "n_origin": int(n_origin),
        "n_destination": int(n_both),
        "from_quintile": from_quintile,
        "to_quintile": to_quintile,
    }


# ---------------------------------------------------------------------------
# 2. ISEI Occupational Prestige
# ---------------------------------------------------------------------------

# Simplified ISEI lookup by ISCO-08 major group (real application uses 4-digit codes)
ISEI_ISCO_MAJOR = {
    1: 68,  # Managers
    2: 74,  # Professionals
    3: 56,  # Technicians and associate professionals
    4: 40,  # Clerical support workers
    5: 32,  # Services and sales workers
    6: 23,  # Skilled agricultural workers
    7: 34,  # Craft and related trades
    8: 31,  # Plant and machine operators
    9: 20,  # Elementary occupations
    0: 47,  # Armed forces
}


def assign_isei(occupation_codes: pd.Series, lookup: dict | None = None) -> pd.Series:
    """
    Assign ISEI scores from occupation codes.

    Parameters
    ----------
    occupation_codes : pd.Series
        ISCO major group codes (1-digit integers for simplified lookup).
    lookup : dict, optional
        Custom {occupation_code: isei_score} mapping.

    Returns
    -------
    pd.Series of ISEI scores.
    """
    lkp = lookup or ISEI_ISCO_MAJOR
    return occupation_codes.map(lkp)


# EGP schema: mapping from ISCO major group to EGP class (simplified)
EGP_ISCO_MAJOR = {
    1: "I",       # Managers → Higher service
    2: "I",       # Professionals → Higher service
    3: "II",      # Technicians → Lower service
    4: "IIIa",    # Clerical → Routine non-manual
    5: "IIIb",    # Service workers → Routine non-manual
    6: "IVc",     # Agricultural self-employed
    7: "V_VI",    # Craft → Skilled manual
    8: "V_VI",    # Operators → Skilled manual
    9: "VIIa",    # Elementary → Unskilled manual
}

EGP_ORDER = ["I", "II", "IIIa", "IIIb", "IVa", "IVb", "IVc", "V_VI", "VIIa", "VIIb"]


def assign_egp(occupation_codes: pd.Series, lookup: dict | None = None) -> pd.Series:
    """Assign EGP class labels from occupation codes."""
    lkp = lookup or EGP_ISCO_MAJOR
    return occupation_codes.map(lkp)


# ---------------------------------------------------------------------------
# 3. Intergenerational Transition Matrix
# ---------------------------------------------------------------------------

def mobility_transition_matrix(
    origin_class: pd.Series,
    destination_class: pd.Series,
    class_order: list[str] | None = None,
    normalize: str = "origin",
) -> pd.DataFrame:
    """
    Compute an intergenerational class transition matrix.

    Parameters
    ----------
    origin_class : pd.Series
        Parent's (or respondent's father's) class.
    destination_class : pd.Series
        Respondent's current class.
    class_order : list of str, optional
        Ordered list of class labels for the matrix.
    normalize : str
        'origin' (row percentages), 'destination' (column), or 'all'.

    Returns
    -------
    pd.DataFrame — transition proportions (percentages).
    """
    classes = class_order or sorted(set(origin_class.dropna()) | set(destination_class.dropna()))
    tab = pd.crosstab(
        origin_class, destination_class,
        values=np.ones(len(origin_class)), aggfunc="sum",
        normalize=normalize,
    ).reindex(index=classes, columns=classes, fill_value=0)
    return (tab * 100).round(1)


def compute_odds_ratio(
    table: pd.DataFrame,
    class_a: str,
    class_b: str,
    class_c: str,
    class_d: str,
) -> float:
    """
    Compute odds ratio for social fluidity from a mobility table.

    OR = (n_ac × n_bd) / (n_ad × n_bc)
    """
    n_ac = table.loc[class_a, class_c]
    n_bd = table.loc[class_b, class_d]
    n_ad = table.loc[class_a, class_d]
    n_bc = table.loc[class_b, class_c]
    if n_ad == 0 or n_bc == 0:
        return np.nan
    return (n_ac * n_bd) / (n_ad * n_bc)
```

## Advanced Usage

### Shapley Decomposition of Income Variance

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm


def shapley_r2_decomposition(
    outcome: pd.Series,
    predictors: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Shapley decomposition of R² across predictor groups.

    For each predictor group, compute the average marginal contribution to R²
    across all possible orderings (approximated by sequential addition and deletion).

    Parameters
    ----------
    outcome : pd.Series
    predictors : dict {group_name: pd.Series}

    Returns
    -------
    pd.DataFrame with group, shapley_r2, pct_contribution.
    """
    from itertools import combinations

    groups = list(predictors.keys())
    df_all = pd.concat([outcome] + list(predictors.values()), axis=1).dropna()
    y = df_all.iloc[:, 0]

    def r2_for_subset(group_subset):
        if not group_subset:
            return 0.0
        X = sm.add_constant(df_all[[g for g in group_subset]])
        model = sm.OLS(y, X).fit()
        return model.rsquared

    n = len(groups)
    shapley = {g: 0.0 for g in groups}
    weight = 1.0 / n

    for size in range(n):
        for subset in combinations(groups, size):
            base = r2_for_subset(list(subset))
            for g in groups:
                if g not in subset:
                    marginal = r2_for_subset(list(subset) + [g]) - base
                    shapley[g] += marginal / (n * len(list(combinations(
                        [x for x in groups if x != g], size))))

    total = sum(shapley.values())
    rows = [{"group": g, "shapley_r2": round(v, 5),
              "pct_contribution": round(v / total * 100, 2) if total > 0 else np.nan}
             for g, v in shapley.items()]
    return pd.DataFrame(rows).sort_values("shapley_r2", ascending=False)
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| IGE > 1 or < 0 | Outliers in income distribution | Winsorize at 1st/99th percentile before log transform |
| Rank-rank slope unstable | Small sample size | Bootstrap CI is wide with N < 500; increase sample or report CI |
| ISEI mapping fails | Non-standard occupation codes | Convert national codes to ISCO-08 first using crosswalk table |
| Transition matrix rows don't sum to 100 | Missing values in one class | Check for NaN after `assign_egp()`; assign "Unknown" category |
| Odds ratio is NaN | Zero cell in the table | Collapse rare classes; add 0.5 Laplace smoothing |

## External Resources

- Chetty, R. et al. (2014). Where is the land of opportunity? *QJE*, 129(4), 1553-1623.
- Erikson, R. & Goldthorpe, J.H. (1992). *The Constant Flux*. Oxford University Press.
- Hauser, R.M. & Warren, J.R. (1997). Socioeconomic indexes for occupations. *Sociological Methodology*, 27, 177-298.
- PSID Data Center: https://psidonline.isr.umich.edu/
- Opportunity Insights Data Library: https://opportunityinsights.org/data/

## Examples

### Example 1: Rank-Rank Regression with Bootstrap CI

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)
n = 8000
parent_income = np.exp(rng.normal(10.5, 0.9, n)).clip(5000, 5_000_000)
# True rho = 0.35
noise = rng.normal(0, 0.7, n)
child_income = np.exp(0.35 * np.log(parent_income) + 0.65 * 10.5 + noise).clip(5000, 5_000_000)

df_mob = pd.DataFrame({"parent_income": parent_income, "child_income": child_income})

ige_result = intergenerational_elasticity(df_mob["parent_income"], df_mob["child_income"])
rr_result = rank_rank_slope(df_mob["parent_income"], df_mob["child_income"])
upward = upward_mobility_rate(df_mob["parent_income"], df_mob["child_income"])

print("=== Intergenerational Mobility ===")
print(f"  IGE (log-log): {ige_result['ige']} (se={ige_result['se']})")
print(f"  Rank-rank slope: {rr_result['rr_slope']} "
      f"[{rr_result['ci_lower']}, {rr_result['ci_upper']}]")
print(f"  Upward mobility (Q1→Q5): {upward['p_upward']:.3%}")

# Rank-rank scatter
pr = df_mob["parent_income"].rank(pct=True)
cr = df_mob["child_income"].rank(pct=True)
pr_bins = pd.cut(pr, bins=20)
bin_means = pd.DataFrame({"pr": pr, "cr": cr, "bin": pr_bins}).groupby("bin").agg(
    pr_mean=("pr", "mean"), cr_mean=("cr", "mean")).reset_index()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(bin_means["pr_mean"], bin_means["cr_mean"], color="#2166ac", s=60, label="Binned means")
slope = rr_result["rr_slope"]
intercept = rr_result["intercept"]
x_line = np.linspace(0, 1, 100)
ax.plot(x_line, intercept + slope * x_line, "r-", linewidth=2,
        label=f"Rank-rank slope = {slope:.3f}")
ax.set_xlabel("Parent Income Percentile Rank")
ax.set_ylabel("Child Income Percentile Rank")
ax.set_title("Intergenerational Rank-Rank Mobility")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rank_rank_mobility.png", dpi=150)
plt.show()
```

### Example 2: Mobility Transition Matrix + Upward Mobility Rate

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(10)
n = 5000
isco_codes = rng.integers(1, 10, n)
df_mob2 = pd.DataFrame({
    "parent_isco": isco_codes,
    "child_isco": np.clip(isco_codes + rng.integers(-2, 3, n), 1, 9),
})
df_mob2["parent_egp"] = assign_egp(df_mob2["parent_isco"])
df_mob2["child_egp"] = assign_egp(df_mob2["child_isco"])
df_mob2["parent_isei"] = assign_isei(df_mob2["parent_isco"])
df_mob2["child_isei"] = assign_isei(df_mob2["child_isco"])

class_order = [c for c in EGP_ORDER if c in df_mob2["parent_egp"].unique()]
matrix = mobility_transition_matrix(df_mob2["parent_egp"], df_mob2["child_egp"],
                                     class_order=class_order)
print("=== Occupational Transition Matrix (row %) ===")
print(matrix.to_string())

# Heatmap
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(matrix.columns)))
ax.set_yticks(range(len(matrix.index)))
ax.set_xticklabels(matrix.columns.tolist(), rotation=45, ha="right")
ax.set_yticklabels(matrix.index.tolist())
ax.set_xlabel("Destination Class")
ax.set_ylabel("Origin Class")
ax.set_title("Intergenerational Occupational Transition Matrix (%)")
plt.colorbar(im, ax=ax, label="Percentage")
for i in range(len(matrix.index)):
    for j in range(len(matrix.columns)):
        ax.text(j, i, f"{matrix.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("mobility_transition_matrix.png", dpi=150)
plt.show()
```

### Example 3: ISEI Distribution by Birth Cohort

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(99)
cohorts = [1950, 1960, 1970, 1980, 1990]
n_per_cohort = 1000

df_isei = pd.concat([
    pd.DataFrame({
        "cohort": c,
        "isco": rng.integers(1, 10, n_per_cohort),
        "gender": rng.choice(["Male", "Female"], n_per_cohort),
    })
    for c in cohorts
], ignore_index=True)

# Later cohorts have higher professional occupations (educational upgrading)
df_isei["isco_adj"] = df_isei.apply(
    lambda r: max(1, min(9, int(r["isco"] - (r["cohort"] - 1950) / 20))), axis=1
)
df_isei["isei"] = assign_isei(df_isei["isco_adj"])

fig, ax = plt.subplots(figsize=(10, 5))
for cohort, grp in df_isei.groupby("cohort"):
    ax.hist(grp["isei"].dropna(), bins=15, alpha=0.5, label=str(cohort), density=True)
ax.set_xlabel("ISEI Score")
ax.set_ylabel("Density")
ax.set_title("ISEI Occupational Prestige Distribution by Birth Cohort")
ax.legend(title="Birth Cohort")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("isei_by_cohort.png", dpi=150)
plt.show()

cohort_means = df_isei.groupby("cohort")["isei"].agg(["mean", "std"])
print("=== Mean ISEI by Birth Cohort ===")
print(cohort_means.round(2).to_string())
```
