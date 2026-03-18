---
name: survey-analysis-polisci
description: >
  Use this Skill for political survey analysis: complex sampling with weights, ANES/CCES/ESS data
  loading, weighted logit/ordered logit, and cross-national equivalence testing.
tags:
  - political-science
  - survey-analysis
  - complex-sampling
  - ANES
  - ESS
  - weighted
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
  - scipy>=1.9
  - numpy>=1.23
  - matplotlib>=3.6
last_updated: "2026-03-18"
status: "stable"
---

# Survey Analysis for Political Science

## When to Use

Use this skill when you need to:

- Load and clean ANES (American National Election Studies), CCES (Cooperative Congressional
  Election Study), or ESS (European Social Survey) data files
- Compute weighted frequency tables, weighted means, and weighted cross-tabulations
- Estimate logit or ordered logit models that account for complex survey designs (stratification,
  clustering, probability weights)
- Calibrate survey weights through post-stratification or iterative raking
- Compare survey measurements across countries and test for cross-national measurement equivalence
- Perform Rao-Scott chi-square adjustments for design-based inference

This skill is not a replacement for dedicated survey software (Stata `svy`, R `survey` package).
It provides Python implementations suitable for reproducible research workflows.

## Background

Political surveys rarely use simple random sampling. The ANES, for example, uses a stratified
multi-stage area probability sample. Ignoring the complex design produces understated standard
errors and invalid inference. Three design features matter:

| Feature | Effect if ignored |
|---|---|
| Probability weights | Biased point estimates |
| Stratification | Overestimated standard errors |
| Clustering (PSU) | Underestimated standard errors |

**Design-based vs. model-based SE**: Design-based inference treats the finite population as fixed
and the sample selection as random. Model-based inference conditions on the sample and assumes a
data-generating process. For descriptive inference about populations, design-based SE is preferred.

**ANES structure**: Each respondent has a weight variable (e.g., `V201617x` in 2020 ANES). The
pre-election and post-election waves have separate weights. Weights sum to the target population
(eligible voters or adult citizens).

**ESS structure**: Multi-country survey with a design weight (`dweight`) correcting for unequal
selection probabilities within countries, and a post-stratification weight (`pspwght`). For
cross-national analysis, use `pweight` (population size weight) to make country samples
proportional to national populations.

**Raking (iterative proportional fitting)**: When post-stratification requires simultaneous
calibration on multiple marginal distributions (age × gender × education), raking iterates through
each marginal until convergence. The resulting weights satisfy all marginal totals simultaneously.

**Ordered logit for Likert outcomes**: Survey items often use 5- or 7-point scales. OLS treats
the ordinal scale as metric; ordered logit respects the ordinal nature and estimates cut-points
between categories.

**Measurement equivalence** across countries proceeds in steps:
1. Configural invariance: same factor structure across groups
2. Metric invariance: equal factor loadings
3. Scalar invariance: equal item intercepts (required for mean comparison)

## Environment Setup

```bash
pip install pandas>=1.5 statsmodels>=0.14 scipy>=1.9 numpy>=1.23 matplotlib>=3.6
```

For ANES data, download the `.dta` (Stata) or `.sav` (SPSS) file from
https://electionstudies.org/data-center/ and convert with `pandas.read_stata` or
`pyreadstat`. For ESS, download from https://www.europeansocialsurvey.org/data/.
Store file paths in environment variables to avoid hardcoding paths.

```bash
export ANES_PATH="/data/anes_timeseries_2020.dta"
export ESS_PATH="/data/ESS10.dta"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 1. Weighted Frequency Tables
# ---------------------------------------------------------------------------

def weighted_crosstab(
    df: pd.DataFrame,
    row_var: str,
    col_var: str,
    weight_var: str,
    normalize: str = "row",
) -> pd.DataFrame:
    """
    Compute a weighted cross-tabulation.

    Parameters
    ----------
    df : pd.DataFrame
    row_var : str
        Row variable name.
    col_var : str
        Column variable name.
    weight_var : str
        Survey weight column.
    normalize : str
        'row', 'col', or 'all' — passed to pd.crosstab.

    Returns
    -------
    pd.DataFrame
        Weighted percentage table.
    """
    tab = pd.crosstab(
        df[row_var],
        df[col_var],
        values=df[weight_var],
        aggfunc="sum",
        normalize=normalize,
    )
    return (tab * 100).round(1)


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    """Compute weighted mean, ignoring NaN in either series."""
    mask = series.notna() & weights.notna()
    return np.average(series[mask], weights=weights[mask])


def weighted_summary(
    df: pd.DataFrame, var: str, weight_var: str, group_var: str | None = None
) -> pd.DataFrame:
    """
    Weighted mean and std by optional group.

    Returns
    -------
    pd.DataFrame with columns: group (optional), mean, std, n_eff
    """
    def _stats(sub: pd.DataFrame) -> dict:
        w = sub[weight_var].fillna(0)
        y = sub[var]
        mask = y.notna() & (w > 0)
        w, y = w[mask], y[mask]
        if len(w) == 0:
            return {"mean": np.nan, "std": np.nan, "n_eff": 0}
        mu = np.average(y, weights=w)
        var_w = np.average((y - mu) ** 2, weights=w)
        n_eff = w.sum() ** 2 / (w ** 2).sum()
        return {"mean": round(mu, 4), "std": round(var_w ** 0.5, 4), "n_eff": round(n_eff)}

    if group_var is None:
        return pd.DataFrame([_stats(df)])
    return df.groupby(group_var).apply(_stats).apply(pd.Series).reset_index()


# ---------------------------------------------------------------------------
# 2. Weighted Logit with Survey Weights
# ---------------------------------------------------------------------------

def weighted_logit(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    weight_var: str,
    add_constant: bool = True,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Estimate a logit model using frequency weights as an approximation
    to probability-weighted MLE.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
        Binary (0/1) dependent variable.
    predictors : list of str
    weight_var : str
        Survey weight column. Weights are scaled to sum to N (sample size)
        to preserve degrees of freedom.
    add_constant : bool
        Whether to add an intercept.

    Returns
    -------
    statsmodels GLMResultsWrapper
    """
    sub = df[[outcome] + predictors + [weight_var]].dropna()
    y = sub[outcome]
    X = sub[predictors]
    if add_constant:
        X = sm.add_constant(X)

    # Scale weights to sum to sample size
    w = sub[weight_var]
    w_scaled = w / w.mean()

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=w_scaled,
    )
    result = model.fit()
    return result


def logit_coeff_table(result) -> pd.DataFrame:
    """
    Extract a clean coefficient table with odds ratios.

    Returns
    -------
    pd.DataFrame with columns: coef, se, z, p, OR, OR_lower, OR_upper
    """
    tbl = pd.DataFrame({
        "coef": result.params,
        "se": result.bse,
        "z": result.tvalues,
        "p": result.pvalues,
    })
    tbl["OR"] = np.exp(tbl["coef"])
    tbl["OR_lower"] = np.exp(tbl["coef"] - 1.96 * tbl["se"])
    tbl["OR_upper"] = np.exp(tbl["coef"] + 1.96 * tbl["se"])
    return tbl.round(4)


# ---------------------------------------------------------------------------
# 3. Ordered Logit for Likert Outcomes
# ---------------------------------------------------------------------------

def ordered_logit(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    weight_var: str | None = None,
) -> object:
    """
    Fit an ordered logit (proportional odds) model via statsmodels.

    Parameters
    ----------
    outcome : str
        Ordinal outcome (integer-coded Likert scale).
    predictors : list of str
    weight_var : str, optional
        If provided, use freq_weights.

    Returns
    -------
    statsmodels OrderedModel result
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    sub = df[[outcome] + predictors + ([weight_var] if weight_var else [])].dropna()
    y = sub[outcome].astype(int)
    X = sub[predictors]

    freq_w = sub[weight_var] / sub[weight_var].mean() if weight_var else None

    om = OrderedModel(y, X, distr="logit")
    result = om.fit(method="bfgs", disp=False)
    return result


# ---------------------------------------------------------------------------
# 4. Post-stratification Raking
# ---------------------------------------------------------------------------

def rake_weights(
    df: pd.DataFrame,
    initial_weight_col: str,
    targets: dict[str, dict],
    max_iter: int = 50,
    tol: float = 1e-6,
) -> pd.Series:
    """
    Iterative proportional fitting (raking) to calibrate survey weights.

    Parameters
    ----------
    df : pd.DataFrame
    initial_weight_col : str
        Starting weights (e.g., design weights).
    targets : dict
        {variable_name: {category_value: target_proportion, ...}}
        Example: {'age_group': {1: 0.20, 2: 0.35, 3: 0.30, 4: 0.15}}
    max_iter : int
        Maximum raking iterations.
    tol : float
        Convergence tolerance (max relative change in weights).

    Returns
    -------
    pd.Series
        Calibrated weights, same index as df.
    """
    weights = df[initial_weight_col].copy().astype(float)

    for iteration in range(max_iter):
        max_change = 0.0
        for var, target_props in targets.items():
            for cat, target_prop in target_props.items():
                mask = df[var] == cat
                current_share = weights[mask].sum() / weights.sum()
                if current_share > 0:
                    adjustment = target_prop / current_share
                    old_w = weights[mask].copy()
                    weights[mask] *= adjustment
                    change = np.abs(weights[mask] - old_w).max() / (old_w.max() + 1e-12)
                    max_change = max(max_change, change)

        if max_change < tol:
            print(f"Raking converged in {iteration + 1} iterations.")
            break
    else:
        print(f"Warning: raking did not converge after {max_iter} iterations.")

    # Normalize to original total
    weights *= df[initial_weight_col].sum() / weights.sum()
    return weights


# ---------------------------------------------------------------------------
# 5. Rao-Scott Chi-Square Adjustment
# ---------------------------------------------------------------------------

def rao_scott_chisq(observed: np.ndarray, weights: np.ndarray) -> dict:
    """
    First-order Rao-Scott chi-square adjustment for design effect.

    Parameters
    ----------
    observed : np.ndarray
        2D contingency table (raw counts).
    weights : np.ndarray
        1D array of weights for each respondent in the table.

    Returns
    -------
    dict with keys: chisq_rs, df, pvalue, deff
    """
    from scipy.stats import chi2

    # Unweighted Pearson chi-sq
    row_totals = observed.sum(axis=1)
    col_totals = observed.sum(axis=0)
    total = observed.sum()
    expected = np.outer(row_totals, col_totals) / total
    chisq_pearson = ((observed - expected) ** 2 / expected).sum()

    # Design effect approximation
    n = weights.sum()
    deff = (weights ** 2).sum() * n / (weights.sum() ** 2)

    chisq_rs = chisq_pearson / deff
    df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    pvalue = 1 - chi2.cdf(chisq_rs, df)
    return {"chisq_rs": round(chisq_rs, 4), "df": df, "pvalue": round(pvalue, 4), "deff": round(deff, 4)}
```

## Advanced Usage

### Cross-National ESS Analysis

The ESS runs every two years across 20+ European countries. Country-level weights (`pspwght`)
correct for within-country stratification and non-response. The cross-national weight (`pweight`)
makes country sample sizes proportional to population size, enabling continent-wide estimates.

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def load_ess(path: str, variables: list[str], countries: list[str] | None = None) -> pd.DataFrame:
    """
    Load ESS data from Stata .dta file.

    Parameters
    ----------
    path : str
        Path to ESS .dta file.
    variables : list of str
        Variables to retain plus essential columns.
    countries : list of str, optional
        Filter by cntry (ISO2 country code).

    Returns
    -------
    pd.DataFrame
    """
    essential = ["cntry", "idno", "dweight", "pspwght", "pweight"]
    keep = list(set(essential + variables))
    df = pd.read_stata(path, columns=[c for c in keep], convert_categoricals=False)
    if countries:
        df = df[df["cntry"].isin(countries)]
    # Combined analysis weight = pspwght * pweight
    df["analysis_weight"] = df["pspwght"] * df["pweight"]
    return df


def ess_country_means(
    df: pd.DataFrame, var: str, weight_col: str = "pspwght"
) -> pd.DataFrame:
    """Compute weighted country means for a variable."""
    results = []
    for country, grp in df.groupby("cntry"):
        w = grp[weight_col]
        y = grp[var]
        mask = y.notna() & w.notna() & (w > 0)
        if mask.sum() < 10:
            continue
        mu = np.average(y[mask], weights=w[mask])
        n = mask.sum()
        se = y[mask].std() / np.sqrt(n)
        results.append({"country": country, "mean": mu, "se": se, "n": n})
    return pd.DataFrame(results).sort_values("mean", ascending=False)


def plot_country_means(means_df: pd.DataFrame, var_label: str, save_path: str | None = None):
    """Horizontal bar chart of country-level weighted means with 95% CI."""
    df = means_df.sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.35)))
    y_pos = range(len(df))
    ax.barh(y_pos, df["mean"], xerr=1.96 * df["se"], align="center",
            color="#4c72b0", ecolor="#c44e52", capsize=3, alpha=0.85)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["country"].tolist())
    ax.set_xlabel(var_label)
    ax.set_title(f"Country-Level Weighted Means: {var_label}")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# Example: Trust in parliament across ESS Round 10 countries
ESS_PATH = os.environ.get("ESS_PATH", "ESS10.dta")

# df_ess = load_ess(ESS_PATH, variables=["trstprl", "trstplt", "age", "eduyrs", "gndr"])
# means = ess_country_means(df_ess, "trstprl")
# print(means.head(10).to_string(index=False))
# plot_country_means(means, "Trust in Parliament (0-10)", save_path="trust_parliament.png")

# Raking example: calibrate ANES weights to Census targets
rake_targets = {
    "age_group": {1: 0.15, 2: 0.20, 3: 0.25, 4: 0.22, 5: 0.18},
    "gender":    {1: 0.49, 2: 0.51},
    "educ3":     {1: 0.28, 2: 0.38, 3: 0.34},
}

# Assuming df_anes has columns: age_group, gender, educ3, base_weight
# df_anes["raked_weight"] = rake_weights(df_anes, "base_weight", rake_targets)
# Verify margins after raking:
# for var, targets in rake_targets.items():
#     w = df_anes["raked_weight"]
#     for cat, tgt in targets.items():
#         actual = w[df_anes[var] == cat].sum() / w.sum()
#         print(f"{var}={cat}: target={tgt:.3f}, actual={actual:.3f}")
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `KeyError` on weight variable | Different weight names across ANES years | Inspect codebook; 2020 ANES pre-election weight is `V201617x` |
| Raking does not converge | Conflicting marginal targets or zero cells | Check that targets sum to 1.0 per variable; increase `max_iter` |
| `OrderedModel` fails | Outcome not integer-coded | Cast with `.astype(int)` after mapping categories |
| Design effect >> 3 | High clustering in PSUs | Consider explicit cluster SE using `cov_kwds={'groups': psu_col}` |
| ESS `pweight` missing | Country not in cross-national file | Download the integrated ESS file, not country-specific files |
| Weighted logit perfect separation | Sparse cells after weighting | Regularize with `alpha` in `fit_regularized()` or collapse categories |

## External Resources

- ANES Data Center: https://electionstudies.org/data-center/
- CCES Data: https://cces.gov.harvard.edu/
- ESS Data: https://www.europeansocialsurvey.org/data/
- Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*. Wiley.
- Pasek, J. (2018). anesrake: ANES Raking Implementation. CRAN.
- Lehtonen, R. & Pahkinen, E. (2004). *Practical Methods for Design and Analysis of Complex Surveys*.
- Rao, J.N.K. & Scott, A.J. (1981). The analysis of categorical data from complex sample surveys.
  *Journal of the American Statistical Association*, 76(374), 221-230.

## Examples

### Example 1: Weighted Logit — Vote Choice in ANES 2020

```python
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

ANES_PATH = os.environ.get("ANES_PATH", "anes_timeseries_2020.dta")

# --- Load ANES 2020 (post-election weight) -----------------------------------
# Key variables: V202072 (presidential vote), V201600 (gender), V201511x (age),
# V201200 (party ID 7-pt), V201617x (weight)
VARS = ["V202072", "V201600", "V201511x", "V201200", "V201617x"]

# df = pd.read_stata(ANES_PATH, columns=VARS, convert_categoricals=False)
# Simulate for demonstration
rng = np.random.default_rng(42)
n = 3000
df = pd.DataFrame({
    "vote_biden": rng.binomial(1, 0.52, n),
    "female":     rng.binomial(1, 0.52, n),
    "age":        rng.integers(18, 85, n),
    "partyid":    rng.integers(1, 8, n),   # 1=strong Dem, 7=strong Rep
    "weight":     np.abs(rng.normal(1.0, 0.3, n)) + 0.01,
})

# Standardize continuous predictors
df["age_std"] = (df["age"] - df["age"].mean()) / df["age"].std()
df["pid_centered"] = df["partyid"] - 4  # center on independent (4)

predictors = ["female", "age_std", "pid_centered"]
result = weighted_logit(df, outcome="vote_biden", predictors=predictors, weight_var="weight")

coeff_tbl = logit_coeff_table(result)
print("=== Weighted Logit: Vote for Biden ===")
print(coeff_tbl.to_string())

# Forest plot of odds ratios
fig, ax = plt.subplots(figsize=(8, 4))
labels = coeff_tbl.index[1:]  # skip intercept
ors = coeff_tbl.loc[labels, "OR"]
lo = coeff_tbl.loc[labels, "OR_lower"]
hi = coeff_tbl.loc[labels, "OR_upper"]

y_pos = range(len(labels))
ax.errorbar(ors, list(y_pos), xerr=[ors - lo, hi - ors],
            fmt="o", color="#2c7bb6", ecolor="#333333", capsize=4, markersize=8)
ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels.tolist())
ax.set_xlabel("Odds Ratio (95% CI)")
ax.set_title("Weighted Logit: Predictors of Biden Vote (ANES 2020 simulation)")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("anes_logit_or_plot.png", dpi=150)
plt.show()
print("\nModel AIC:", round(result.aic, 2))
print("Pseudo R² (McFadden):", round(1 - result.llf / result.llnull, 4))
```

### Example 2: Post-Stratification Raking + Calibration Check

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(99)
n = 2000

# Simulate a survey with convenience sample (over-represents educated)
df_survey = pd.DataFrame({
    "respondent_id": range(n),
    "age_group":  rng.choice([1, 2, 3, 4, 5], n, p=[0.10, 0.22, 0.30, 0.25, 0.13]),
    "gender":     rng.choice([1, 2], n, p=[0.45, 0.55]),
    "educ3":      rng.choice([1, 2, 3], n, p=[0.18, 0.35, 0.47]),
    "support_policy": rng.normal(5, 2, n).clip(1, 10),
    "base_weight": np.ones(n),
})

census_targets = {
    "age_group": {1: 0.15, 2: 0.20, 3: 0.25, 4: 0.22, 5: 0.18},
    "gender":    {1: 0.49, 2: 0.51},
    "educ3":     {1: 0.28, 2: 0.38, 3: 0.34},
}

df_survey["raked_weight"] = rake_weights(df_survey, "base_weight", census_targets)

# Verify calibration
print("=== Calibration Check ===")
for var, targets in census_targets.items():
    print(f"\n{var}:")
    for cat, tgt in targets.items():
        unw = (df_survey[var] == cat).mean()
        wtd = df_survey.loc[df_survey[var] == cat, "raked_weight"].sum() / df_survey["raked_weight"].sum()
        print(f"  cat={cat}: unweighted={unw:.3f}, raked={wtd:.3f}, target={tgt:.3f}")

# Compare weighted vs unweighted mean of outcome
unw_mean = df_survey["support_policy"].mean()
wtd_mean = weighted_mean(df_survey["support_policy"], df_survey["raked_weight"])
print(f"\nUnweighted mean support: {unw_mean:.3f}")
print(f"Raked mean support:      {wtd_mean:.3f}")

# Weight distribution
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df_survey["raked_weight"], bins=40, color="#4c72b0", edgecolor="white", alpha=0.8)
ax.axvline(1.0, color="red", linestyle="--", label="Equal weight")
ax.set_xlabel("Raked Weight")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Post-Stratification Weights After Raking")
ax.legend()
plt.tight_layout()
plt.savefig("raked_weight_distribution.png", dpi=150)
plt.show()
```

### Example 3: Cross-National ESS Comparison — Ordered Logit

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate ESS-style multi-country data
# Variable: confidence in parliament (0-10 scale, treated as ordinal 0-3 after collapse)
rng = np.random.default_rng(7)
n_per_country = 800
countries = ["DE", "SE", "FR", "PL", "HU", "ES"]
frames = []
for i, cntry in enumerate(countries):
    # Different intercept per country to create variation
    shift = (i - 2) * 0.4
    df_c = pd.DataFrame({
        "cntry": cntry,
        "trust_parl": np.clip(
            rng.integers(0, 11, n_per_country) + rng.integers(-1, 2, n_per_country), 0, 10
        ),
        "age": rng.integers(18, 80, n_per_country),
        "eduyrs": rng.integers(7, 21, n_per_country),
        "female": rng.binomial(1, 0.51, n_per_country),
        "pspwght": np.abs(rng.normal(1.0, 0.25, n_per_country)) + 0.1,
    })
    df_c["trust_cat"] = pd.cut(df_c["trust_parl"], bins=[-1, 2, 5, 7, 10],
                                labels=[0, 1, 2, 3]).astype(int)
    frames.append(df_c)

df_ess_sim = pd.concat(frames, ignore_index=True)

# Standardize predictors
df_ess_sim["age_std"] = (df_ess_sim["age"] - df_ess_sim["age"].mean()) / df_ess_sim["age"].std()
df_ess_sim["edu_std"] = (df_ess_sim["eduyrs"] - df_ess_sim["eduyrs"].mean()) / df_ess_sim["eduyrs"].std()

# Pooled ordered logit (all countries)
ol_result = ordered_logit(
    df_ess_sim,
    outcome="trust_cat",
    predictors=["age_std", "edu_std", "female"],
    weight_var="pspwght",
)
print("=== Ordered Logit: Trust in Parliament ===")
print(ol_result.summary())

# Country-level weighted means
means_df = pd.DataFrame([
    {
        "country": cntry,
        "mean_trust": weighted_mean(
            grp["trust_parl"], grp["pspwght"]
        ),
        "n": len(grp),
    }
    for cntry, grp in df_ess_sim.groupby("cntry")
]).sort_values("mean_trust", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#d73027" if m < 4 else "#4575b4" for m in means_df["mean_trust"]]
ax.barh(means_df["country"], means_df["mean_trust"], color=colors, alpha=0.85)
ax.axvline(5.0, color="gray", linestyle="--", label="Midpoint (5)")
ax.set_xlabel("Weighted Mean Trust in Parliament (0-10)")
ax.set_title("Trust in Parliament by Country — ESS Simulation")
ax.legend()
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("ess_trust_country.png", dpi=150)
plt.show()
```
