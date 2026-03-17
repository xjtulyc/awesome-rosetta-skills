---
name: ols-regression
description: >
  Run OLS regressions with full diagnostics: heteroscedasticity tests, robust/clustered SEs,
  VIF, structural breaks, and publication-ready tables via statsmodels.
tags:
  - econometrics
  - regression
  - statistics
  - causal-inference
  - python
  - heteroscedasticity
version: "1.0.0"
authors:
  - name: "awesome-rosetta-skills contributors"
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - statsmodels>=0.14.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - patsy>=0.5.3
last_updated: "2026-03-17"
---

# OLS Regression with Full Diagnostics

Ordinary Least Squares is the workhorse of empirical economics. This skill covers the full
pipeline: model specification, assumption testing, robust inference, and presentation. It
follows best practices from Angrist & Pischke (2009) and Greene (2018).

---

## Core Concepts

### Why OLS?

Under the Gauss-Markov assumptions (linearity, random sampling, no perfect multicollinearity,
zero conditional mean, homoscedasticity), OLS is BLUE — Best Linear Unbiased Estimator. In
practice, homoscedasticity almost never holds for economic cross-sectional data, so **robust
standard errors** are the default. Consistency requires only that E[u|X] = 0.

### Key Assumptions and What Breaks Them

| Assumption | Violation | Consequence | Fix |
|---|---|---|---|
| E[u\|X] = 0 | Omitted variable, endogeneity | Biased, inconsistent β̂ | Controls, IV |
| No multicollinearity | High VIF | Inflated SE, unstable estimates | Drop/combine vars |
| Homoscedasticity | Heteroscedasticity | SE wrong, invalid inference | Robust SE |
| No serial correlation | Time series / clusters | SE wrong | Clustered SE, FGLS |
| Normality of u | Small samples | t/F tests invalid | Bootstrap |

---

## Full Implementation

```python
# ols_regression.py
"""
OLS regression with full econometric diagnostics.
Requires: statsmodels, pandas, numpy, scipy, matplotlib, patsy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.compat.python import lzip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from patsy import dmatrices
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {"primary": "#2c7bb6", "secondary": "#d7191c", "neutral": "#636363"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAIN OLS RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_ols_full(formula: str, df: pd.DataFrame, cov_type: str = "HC3") -> dict:
    """
    Run OLS regression with comprehensive diagnostics.

    Parameters
    ----------
    formula   : Patsy formula string, e.g. 'wage ~ educ + exper + C(female)'
    df        : pandas DataFrame
    cov_type  : covariance type for robust SE — 'nonrobust', 'HC0'–'HC3',
                or 'cluster' (needs cluster_var kwarg — use get_robust_se separately)

    Returns
    -------
    dict with keys:
        'result'         : statsmodels RegressionResultsWrapper (OLS, HC3 SE)
        'result_ols'     : plain OLS result (homoscedastic SE)
        'diagnostics'    : dict of test statistics
        'vif'            : DataFrame of VIF values
        'summary'        : formatted summary string
    """
    # fit plain OLS first (needed for some tests)
    result_ols = smf.ols(formula, data=df).fit()

    # fit with robust SE
    result = smf.ols(formula, data=df).fit(cov_type=cov_type)

    diagnostics = {}

    # ── heteroscedasticity ──
    het = check_heteroscedasticity(result_ols)
    diagnostics["heteroscedasticity"] = het

    # ── RESET test (functional form) ──
    try:
        reset = linear_reset(result_ols, power=3, use_f=True)
        diagnostics["reset_test"] = {
            "F_statistic": reset.fvalue,
            "p_value": reset.pvalue,
            "interpretation": "Reject H0 (misspecification)" if reset.pvalue < 0.05
                              else "Cannot reject H0 (adequate functional form)",
        }
    except Exception:
        diagnostics["reset_test"] = None

    # ── Durbin-Watson ──
    dw = durbin_watson(result_ols.resid)
    diagnostics["durbin_watson"] = {
        "statistic": dw,
        "interpretation": (
            "Positive autocorrelation" if dw < 1.5
            else "Negative autocorrelation" if dw > 2.5
            else "No strong autocorrelation"
        ),
    }

    # ── VIF ──
    vif = compute_vif(formula, df)
    diagnostics["multicollinearity"] = {
        "max_vif": vif["VIF"].max() if vif is not None else None,
        "high_vif_vars": (
            vif[vif["VIF"] > 10]["Variable"].tolist() if vif is not None else []
        ),
    }

    # ── normality of residuals ──
    _, norm_p = stats.shapiro(result_ols.resid) if len(result_ols.resid) <= 5000 \
                else stats.kstest(result_ols.resid, "norm",
                                  args=(result_ols.resid.mean(), result_ols.resid.std()))
    diagnostics["normality_shapiro"] = {
        "p_value": norm_p,
        "interpretation": "Residuals non-normal (p<0.05)" if norm_p < 0.05
                          else "Residuals approximately normal",
    }

    # ── model fit ──
    diagnostics["model_fit"] = {
        "n_obs": int(result.nobs),
        "r_squared": result.rsquared,
        "adj_r_squared": result.rsquared_adj,
        "AIC": result.aic,
        "BIC": result.bic,
        "F_statistic": result.fvalue,
        "F_p_value": result.f_pvalue,
    }

    return {
        "result": result,
        "result_ols": result_ols,
        "diagnostics": diagnostics,
        "vif": vif,
        "summary": _build_summary(result, diagnostics),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. HETEROSCEDASTICITY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def check_heteroscedasticity(result) -> dict:
    """
    Run Breusch-Pagan and White's tests for heteroscedasticity.

    Breusch-Pagan: tests whether residual variance is a linear function of regressors.
    White's test:  tests against general heteroscedasticity (includes cross-products).

    Returns a dict with test statistics, p-values, and plain-English interpretations.
    """
    out = {}

    # Breusch-Pagan
    try:
        bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(result.resid, result.model.exog)
        out["breusch_pagan"] = {
            "LM_statistic": bp_lm,
            "LM_p_value": bp_p,
            "F_statistic": bp_f,
            "F_p_value": bp_fp,
            "reject_H0": bp_p < 0.05,
            "interpretation": (
                "Evidence of heteroscedasticity (p<0.05)" if bp_p < 0.05
                else "No strong evidence of heteroscedasticity"
            ),
        }
    except Exception as e:
        out["breusch_pagan"] = {"error": str(e)}

    # White's test
    try:
        w_lm, w_p, w_f, w_fp = het_white(result.resid, result.model.exog)
        out["white"] = {
            "LM_statistic": w_lm,
            "LM_p_value": w_p,
            "F_statistic": w_f,
            "F_p_value": w_fp,
            "reject_H0": w_p < 0.05,
            "interpretation": (
                "Evidence of heteroscedasticity (p<0.05)" if w_p < 0.05
                else "No strong evidence of heteroscedasticity"
            ),
        }
    except Exception as e:
        out["white"] = {"error": str(e)}

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROBUST STANDARD ERRORS
# ─────────────────────────────────────────────────────────────────────────────

def get_robust_se(result, cov_type: str = "HC3", cluster_var=None,
                  df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Re-estimate with the requested covariance sandwich estimator.

    Parameters
    ----------
    result     : OLS result from statsmodels (plain, non-robust)
    cov_type   : 'HC0', 'HC1', 'HC2', 'HC3' (White), or 'cluster'
    cluster_var: column name for clustering (only when cov_type='cluster')
    df         : original DataFrame (needed for cluster variable)

    Returns
    -------
    DataFrame with columns: coef, se_ols, se_robust, t_robust, p_robust,
                            ci_lower, ci_upper, stars
    """
    if cov_type == "cluster":
        if cluster_var is None or df is None:
            raise ValueError("cluster_var and df required for clustered SE")
        groups = df[cluster_var]
        robust_result = result.model.fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
    else:
        robust_result = result.model.fit(cov_type=cov_type)

    table = pd.DataFrame({
        "coef":       robust_result.params,
        "se_ols":     result.bse,
        "se_robust":  robust_result.bse,
        "t_robust":   robust_result.tvalues,
        "p_robust":   robust_result.pvalues,
        "ci_lower":   robust_result.conf_int()[0],
        "ci_upper":   robust_result.conf_int()[1],
    })
    table["stars"] = table["p_robust"].apply(_stars)
    return table


# ─────────────────────────────────────────────────────────────────────────────
# 4. VIF (MULTICOLLINEARITY)
# ─────────────────────────────────────────────────────────────────────────────

def compute_vif(formula: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors for all continuous regressors.

    VIF = 1 / (1 - R²_j), where R²_j is from regressing X_j on all other regressors.
    Rule of thumb: VIF > 10 indicates serious multicollinearity.
    """
    try:
        y, X = dmatrices(formula, data=df, return_type="dataframe")
    except Exception:
        return None

    # drop Intercept
    X_no_intercept = X.drop(columns=["Intercept"], errors="ignore")
    if X_no_intercept.shape[1] < 2:
        return None

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_no_intercept.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X_no_intercept.values, i)
        for i in range(X_no_intercept.shape[1])
    ]
    vif_data["Severity"] = vif_data["VIF"].apply(
        lambda v: "None" if v < 5 else "Moderate" if v < 10 else "High"
    )
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHOW STRUCTURAL BREAK TEST
# ─────────────────────────────────────────────────────────────────────────────

def chow_test(formula: str, df: pd.DataFrame, break_var: str,
              break_value) -> dict:
    """
    Chow test for parameter stability across two sub-samples.

    H0: coefficients are the same in both groups.
    Statistic: F = [(RSS_pool - RSS_1 - RSS_2) / k] / [(RSS_1 + RSS_2) / (n - 2k)]

    Parameters
    ----------
    formula     : Patsy formula
    df          : full DataFrame
    break_var   : column name to split on
    break_value : threshold — group1 is df[break_var] < break_value
    """
    df1 = df[df[break_var] < break_value].copy()
    df2 = df[df[break_var] >= break_value].copy()

    res_pool = smf.ols(formula, data=df).fit()
    res_1    = smf.ols(formula, data=df1).fit()
    res_2    = smf.ols(formula, data=df2).fit()

    rss_pool = res_pool.ssr
    rss_1    = res_1.ssr
    rss_2    = res_2.ssr

    k  = len(res_pool.params)          # number of parameters
    n  = len(df)

    F = ((rss_pool - rss_1 - rss_2) / k) / ((rss_1 + rss_2) / (n - 2 * k))
    p = 1 - stats.f.cdf(F, k, n - 2 * k)

    return {
        "F_statistic": F,
        "p_value": p,
        "df_numerator": k,
        "df_denominator": n - 2 * k,
        "n_group1": len(df1),
        "n_group2": len(df2),
        "reject_H0": p < 0.05,
        "interpretation": (
            f"Structural break detected at {break_var}={break_value} (p={p:.4f})"
            if p < 0.05
            else f"No structural break at {break_var}={break_value} (p={p:.4f})"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. REGRESSION TABLE (stargazer-style in Python)
# ─────────────────────────────────────────────────────────────────────────────

def make_regression_table(
    results_list: list,
    model_names: list = None,
    title: str = "OLS Regression Results",
    dep_var_label: str = "Dependent variable",
    float_fmt: str = "{:.3f}",
) -> str:
    """
    Build a text regression table comparable to R's stargazer.

    Parameters
    ----------
    results_list : list of statsmodels result objects (use .fit(cov_type='HC3'))
    model_names  : list of column headers, e.g. ['(1)', '(2)', '(3)']
    title        : table title
    dep_var_label: row label for dependent variable

    Returns
    -------
    Formatted string suitable for printing or writing to a .txt file.
    """
    if model_names is None:
        model_names = [f"({i+1})" for i in range(len(results_list))]

    # collect all variable names across models
    all_vars = []
    for res in results_list:
        for v in res.params.index:
            if v not in all_vars:
                all_vars.append(v)

    col_w = 14
    sep = "=" * (20 + col_w * len(results_list))

    lines = [
        title,
        sep,
        f"{dep_var_label:<20}" + "".join(f"{m:>{col_w}}" for m in model_names),
        "-" * (20 + col_w * len(results_list)),
    ]

    for var in all_vars:
        coef_row = f"{var:<20}"
        se_row   = f"{'':20}"
        for res in results_list:
            if var in res.params:
                coef = res.params[var]
                se   = res.bse[var]
                p    = res.pvalues[var]
                star = _stars(p)
                coef_row += f"{float_fmt.format(coef)+star:>{col_w}}"
                se_row   += f"{'('+float_fmt.format(se)+')':>{col_w}}"
            else:
                coef_row += f"{'':>{col_w}}"
                se_row   += f"{'':>{col_w}}"
        lines.append(coef_row)
        lines.append(se_row)

    lines.append("-" * (20 + col_w * len(results_list)))

    # footer stats
    n_row   = f"{'Observations':<20}" + "".join(
        f"{int(res.nobs):>{col_w}}" for res in results_list
    )
    r2_row  = f"{'R²':<20}" + "".join(
        f"{float_fmt.format(res.rsquared):>{col_w}}" for res in results_list
    )
    ar2_row = f"{'Adj. R²':<20}" + "".join(
        f"{float_fmt.format(res.rsquared_adj):>{col_w}}" for res in results_list
    )
    lines += [n_row, r2_row, ar2_row, sep]
    lines.append("Note: * p<0.1  ** p<0.05  *** p<0.01  Robust (HC3) standard errors in parentheses")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 7. DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagnostics(result, title: str = "OLS Diagnostics") -> plt.Figure:
    """
    Four-panel diagnostic plot: residuals vs fitted, Q-Q, scale-location, leverage.
    """
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    fitted = result.fittedvalues
    resid  = result.resid
    std_resid = resid / resid.std()

    # panel 1 — residuals vs fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(fitted, resid, alpha=0.4, s=20, color=COLORS["primary"])
    ax1.axhline(0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Fitted values"); ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")

    # panel 2 — Q-Q plot
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax2.scatter(osm, osr, alpha=0.4, s=20, color=COLORS["primary"])
    ax2.plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=1)
    ax2.set_xlabel("Theoretical quantiles"); ax2.set_ylabel("Sample quantiles")
    ax2.set_title("Normal Q-Q")

    # panel 3 — scale-location (sqrt|std resid| vs fitted)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.4, s=20,
                color=COLORS["primary"])
    ax3.set_xlabel("Fitted values"); ax3.set_ylabel("√|Standardized residuals|")
    ax3.set_title("Scale-Location")

    # panel 4 — residuals vs leverage
    ax4 = fig.add_subplot(gs[1, 1])
    influence = result.get_influence()
    leverage  = influence.hat_matrix_diag
    ax4.scatter(leverage, std_resid, alpha=0.4, s=20, color=COLORS["primary"])
    ax4.axhline(0, color="red", linestyle="--", linewidth=1)
    ax4.set_xlabel("Leverage"); ax4.set_ylabel("Standardized residuals")
    ax4.set_title("Residuals vs Leverage")

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


def _build_summary(result, diagnostics: dict) -> str:
    lines = [
        "=" * 60,
        "OLS REGRESSION DIAGNOSTIC SUMMARY",
        "=" * 60,
        f"Observations   : {int(result.nobs)}",
        f"R²             : {result.rsquared:.4f}",
        f"Adj. R²        : {result.rsquared_adj:.4f}",
        f"AIC            : {result.aic:.2f}",
        "",
        "── Heteroscedasticity ──",
    ]
    bp = diagnostics.get("heteroscedasticity", {}).get("breusch_pagan", {})
    w  = diagnostics.get("heteroscedasticity", {}).get("white", {})
    if "LM_p_value" in bp:
        lines.append(f"  Breusch-Pagan p = {bp['LM_p_value']:.4f}  {bp['interpretation']}")
    if "LM_p_value" in w:
        lines.append(f"  White's test  p = {w['LM_p_value']:.4f}  {w['interpretation']}")

    reset = diagnostics.get("reset_test")
    if reset:
        lines += ["", "── RESET Test (functional form) ──",
                  f"  F = {reset['F_statistic']:.4f}  p = {reset['p_value']:.4f}  {reset['interpretation']}"]

    dw = diagnostics.get("durbin_watson")
    if dw:
        lines += ["", "── Durbin-Watson ──",
                  f"  DW = {dw['statistic']:.4f}  {dw['interpretation']}"]

    mc = diagnostics.get("multicollinearity")
    if mc and mc.get("max_vif"):
        lines += ["", "── Multicollinearity ──",
                  f"  Max VIF = {mc['max_vif']:.2f}"]
        if mc["high_vif_vars"]:
            lines.append(f"  High VIF variables: {', '.join(mc['high_vif_vars'])}")

    lines.append("=" * 60)
    return "\n".join(lines)
```

---

## Example A — Mincer Wage Regression

This replicates the classic Mincer earnings equation:

ln(wage) = β₀ + β₁ educ + β₂ exper + β₃ exper² + β₄ female + u

```python
# example_a_wage_regression.py
"""
Mincer wage regression with heteroscedasticity diagnostics.
Dataset: CPS-style synthetic wage data.
"""

import numpy as np
import pandas as pd
from ols_regression import run_ols_full, make_regression_table, plot_diagnostics

rng = np.random.default_rng(42)
n   = 2000

# simulate data
educ   = rng.integers(8, 21, n).astype(float)
exper  = np.clip(rng.normal(20, 10, n), 0, 45)
female = rng.binomial(1, 0.48, n).astype(float)
union  = rng.binomial(1, 0.15, n).astype(float)

# log-wage DGP: heteroscedastic — variance rises with education
sigma = 0.2 + 0.03 * educ
u     = rng.normal(0, sigma, n)
lnwage = (1.2 + 0.10 * educ + 0.04 * exper
          - 0.0006 * exper**2 - 0.22 * female + 0.12 * union + u)

df = pd.DataFrame({
    "lnwage": lnwage, "educ": educ, "exper": exper,
    "exper2": exper**2, "female": female, "union": union,
})

# ── Model 1: parsimonious ──
m1 = run_ols_full("lnwage ~ educ + exper + exper2 + female", df)
# ── Model 2: add union ──
m2 = run_ols_full("lnwage ~ educ + exper + exper2 + female + union", df)

# print diagnostic summary
print(m1["summary"])

# regression table
table = make_regression_table(
    [m1["result"], m2["result"]],
    model_names=["(1) Base", "(2) + Union"],
    dep_var_label="ln(wage)",
)
print(table)

# Marginal effect of education (partial derivative, evaluated at mean exper):
# ∂ ln(wage) / ∂ educ ≈ β_educ → wage rises by ~10% per extra year of schooling

# returns to education (percentage)
beta_educ = m2["result"].params["educ"]
print(f"\nReturn to education: {beta_educ*100:.1f}% per year of schooling")

# peak experience: β_exper / (2 * |β_exper2|)
b1 = m2["result"].params["exper"]
b2 = m2["result"].params["exper2"]
peak = -b1 / (2 * b2)
print(f"Peak experience (years): {peak:.1f}")

# save diagnostic plots
fig = plot_diagnostics(m2["result_ols"], title="Wage Regression Diagnostics")
fig.savefig("wage_diagnostics.png", dpi=150, bbox_inches="tight")
print("Saved wage_diagnostics.png")
```

---

## Example B — OLS vs Robust SE with Heteroscedastic Data

Demonstrates how plain OLS standard errors understate uncertainty when
variance depends on a regressor.

```python
# example_b_ols_vs_robust.py
"""
Compare OLS, HC1, HC3, and clustered SEs on heteroscedastic data.
Shows that t-statistics can be inflated up to 2× with plain OLS.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ols_regression import (
    run_ols_full,
    get_robust_se,
    check_heteroscedasticity,
)

rng = np.random.default_rng(0)
n   = 500

# DGP: Var(u|x) = (1 + 2x)²  — strong heteroscedasticity
x      = rng.uniform(0, 5, n)
u      = rng.normal(0, 1 + 2 * x, n)   # heteroscedastic errors
y      = 2 + 1.5 * x + u
state  = rng.integers(0, 20, n)         # 20 state clusters

df = pd.DataFrame({"y": y, "x": x, "state": state})

# plain OLS
res_plain = run_ols_full("y ~ x", df, cov_type="nonrobust")
het       = check_heteroscedasticity(res_plain["result_ols"])

print("Breusch-Pagan:", het["breusch_pagan"]["interpretation"])
print("White's test :", het["white"]["interpretation"])
print()

# compare SE across estimators
se_ols      = get_robust_se(res_plain["result_ols"], cov_type="nonrobust")
se_hc1      = get_robust_se(res_plain["result_ols"], cov_type="HC1")
se_hc3      = get_robust_se(res_plain["result_ols"], cov_type="HC3")
se_cluster  = get_robust_se(
    res_plain["result_ols"], cov_type="cluster",
    cluster_var="state", df=df
)

comparison = pd.DataFrame({
    "OLS SE":       se_ols["se_ols"],
    "HC1 SE":       se_hc1["se_robust"],
    "HC3 SE":       se_hc3["se_robust"],
    "Clustered SE": se_cluster["se_robust"],
}).loc[["x"]]

print("Standard error comparison for coefficient on x:")
print(comparison.to_string())

# visualise
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(x, u, alpha=0.3, s=15, color="#2c7bb6")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("x"); axes[0].set_ylabel("Residual u")
axes[0].set_title("Heteroscedastic residuals")

labels = ["OLS", "HC1", "HC3", "Clustered"]
values = [
    float(se_ols.loc["x", "se_ols"]),
    float(se_hc1.loc["x", "se_robust"]),
    float(se_hc3.loc["x", "se_robust"]),
    float(se_cluster.loc["x", "se_robust"]),
]
colors = ["#d7191c", "#2c7bb6", "#1a9641", "#ff7f00"]
bars = axes[1].bar(labels, values, color=colors, width=0.5)
axes[1].set_ylabel("Standard error of β̂_x")
axes[1].set_title("SE comparison: OLS vs robust estimators")
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig("se_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved se_comparison.png")
```

---

## Coefficient Interpretation Guide

| Specification | Interpretation of β |
|---|---|
| y = β₀ + βX + u | One unit ↑ X → β units ↑ y |
| ln(y) = β₀ + βX + u | One unit ↑ X → 100β% ↑ y |
| y = β₀ + β ln(X) + u | 1% ↑ X → β/100 units ↑ y |
| ln(y) = β₀ + β ln(X) + u | 1% ↑ X → β% ↑ y (elasticity) |
| y = β₀ + βX + γX² + u | ∂y/∂X = β + 2γX (non-linear) |
| y = β₀ + β D + u (D binary) | Being in group D → β units ↑ y |

### Omitted Variable Bias Formula

If the true model is y = β₀ + β₁X₁ + β₂X₂ + u but you omit X₂:

plim(β̂₁^short) = β₁ + β₂ · δ₁₂

where δ₁₂ = Cov(X₂, X₁) / Var(X₁) is the regression coefficient of X₂ on X₁.

**Direction of bias**: positive if β₂ and Corr(X₁,X₂) have the same sign.

---

## Checklist Before Reporting Results

- [ ] Report N, R², adjusted R², F-statistic
- [ ] Use HC3 or clustered SE as default (not plain OLS SE)
- [ ] Check VIF — flag any variable > 10
- [ ] Run RESET test for functional form misspecification
- [ ] Acknowledge endogeneity if present (consider IV)
- [ ] Show regression table with at least one robustness column
- [ ] Inspect residual plots for patterns
- [ ] Note economically meaningful effect sizes, not just p-values

---

## References

- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton UP.
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator. *Econometrica*, 48(4), 817–838.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305–325.
- Stock, J. H., & Watson, M. W. (2020). *Introduction to Econometrics* (4th ed.). Pearson.
