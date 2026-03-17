---
name: iv-2sls
description: >
  Instrumental variables and 2SLS: first-stage diagnostics, Wu-Hausman endogeneity test,
  Sargan overidentification, weak instrument tests, and linearmodels implementation.
tags:
  - econometrics
  - instrumental-variables
  - causal-inference
  - 2sls
  - python
  - endogeneity
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
  - linearmodels>=5.3.0
  - statsmodels>=0.14.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# Instrumental Variables and Two-Stage Least Squares (IV/2SLS)

IV estimation addresses endogeneity — the core challenge in causal inference when
the error term is correlated with the regressor of interest. A valid instrument Z must
satisfy: (1) **Relevance**: Cov(Z, X) ≠ 0 (testable), and (2) **Exclusion restriction**:
Cov(Z, u) = 0 (untestable, requires economic reasoning).

---

## Conceptual Framework

### The Endogeneity Problem

OLS: y = Xβ + u, but Cov(X, u) ≠ 0 ⟹ E[β̂_OLS] ≠ β.

Sources of endogeneity:
- Omitted variables (unmeasured ability in earnings equations)
- Simultaneous causality (price and quantity in supply/demand)
- Measurement error in X (attenuation bias toward zero)

### 2SLS Estimator

**First stage**: X̂ = Z π₁ + W π₂ + v  (W = exogenous controls)
**Second stage**: y = X̂ β + W γ + u

The 2SLS estimator: β̂_IV = (X̂'X)⁻¹ X̂'y

### Weak Instruments

The Stock-Yogo (2005) rule of thumb: first-stage F-statistic ≥ 10 for acceptable
size distortion. More precisely:
- F < 10: IV estimate may be nearly as biased as OLS (towards OLS)
- F ≥ 10: bias ≤ 10% of OLS bias (approximately)
- LIML is more robust than 2SLS with weak instruments

---

## Full Implementation

```python
# iv_2sls.py
"""
Instrumental variables / 2SLS estimation and diagnostics.
Primary backend: linearmodels.IV2SLS (Baum, Schaffer, Stillman style)
Fallback: manual 2SLS via statsmodels OLS.
Requires: linearmodels, statsmodels, pandas, numpy, scipy
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    from linearmodels.iv import IV2SLS, IVLIML
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    warnings.warn(
        "linearmodels not installed. Install via: pip install linearmodels\n"
        "Falling back to manual 2SLS via statsmodels."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAIN IV/2SLS ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

def run_iv2sls(
    y: pd.Series,
    endog: pd.DataFrame,
    instruments: pd.DataFrame,
    exog: pd.DataFrame = None,
    cov_type: str = "robust",
    estimator: str = "2sls",
) -> dict:
    """
    Estimate IV/2SLS or LIML and return comprehensive results.

    Parameters
    ----------
    y           : outcome variable (pd.Series)
    endog       : endogenous regressor(s) (pd.DataFrame, may have multiple columns)
    instruments : excluded instruments (pd.DataFrame)
    exog        : included exogenous regressors, e.g. controls (pd.DataFrame or None)
    cov_type    : 'robust' (HC), 'unadjusted' (OLS SE), or 'kernel' (HAC)
    estimator   : '2sls' or 'liml'

    Returns
    -------
    dict with keys: result, params, bse, pvalues, ci, diagnostics
    """
    # align all inputs on common index
    idx = y.index
    if exog is None:
        exog = pd.DataFrame({"const": np.ones(len(y))}, index=idx)
    elif "const" not in exog.columns and "Intercept" not in exog.columns:
        exog = exog.copy()
        exog.insert(0, "const", 1.0)

    if HAS_LINEARMODELS:
        result = _run_linearmodels(y, endog, instruments, exog, cov_type, estimator)
    else:
        result = _run_manual_2sls(y, endog, instruments, exog)

    # ── compile diagnostics ──
    diag = {}

    # first-stage (run manually for clean F-stat)
    fs = _first_stage_stats(y, endog, instruments, exog)
    diag["first_stage"] = fs

    # Wu-Hausman endogeneity test
    wh = hausman_test_iv(y, endog, instruments, exog)
    diag["hausman"] = wh

    # Sargan-Hansen overidentification (only if over-identified)
    n_endog = endog.shape[1] if hasattr(endog, "shape") else 1
    n_instr = instruments.shape[1] if hasattr(instruments, "shape") else 1
    if n_instr > n_endog:
        sargan = sargan_test(result, y, endog, instruments, exog)
        diag["sargan"] = sargan
    else:
        diag["sargan"] = {"note": "Exactly identified — Sargan test not applicable"}

    return {
        "result":    result,
        "estimator": estimator,
        "cov_type":  cov_type,
        "diagnostics": diag,
        "summary": _build_iv_summary(result, diag),
    }


def _run_linearmodels(y, endog, instruments, exog, cov_type, estimator):
    """Fit IV using linearmodels.IV2SLS / IVLIML."""
    Estimator = IV2SLS if estimator == "2sls" else IVLIML
    model  = Estimator(y, exog, endog, instruments)
    result = model.fit(cov_type=cov_type)
    return result


def _run_manual_2sls(y, endog, instruments, exog):
    """Manual 2SLS via two OLS passes (fallback)."""
    # stack included exog + instruments for first stage
    Z = pd.concat([exog, instruments], axis=1)
    X_hat_parts = {}
    for col in endog.columns:
        fs_model = sm.OLS(endog[col], Z).fit()
        X_hat_parts[col] = fs_model.fittedvalues

    X_hat = pd.DataFrame(X_hat_parts, index=y.index)
    X2 = pd.concat([exog, X_hat], axis=1)
    result = sm.OLS(y, X2).fit(cov_type="HC3")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIRST STAGE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def _first_stage_stats(y, endog, instruments, exog) -> dict:
    """
    First-stage F-statistic and partial R² for each endogenous variable.
    Uses the Sanderson-Windmeijer (2016) approach for multiple endogenous regressors.
    """
    Z = pd.concat([exog, instruments], axis=1)
    out = {}
    for col in endog.columns:
        fs = sm.OLS(endog[col], Z).fit()
        # partial F on excluded instruments only
        n_instr = instruments.shape[1]
        n_obs   = len(y)
        n_total = Z.shape[1]

        # F = [(R² - R²_restricted) / q] / [(1 - R²) / (n-k)]
        # restricted: regress endog on exog only
        fs_restricted = sm.OLS(endog[col], exog).fit()
        r2_full  = fs.rsquared
        r2_restr = fs_restricted.rsquared
        q = n_instr
        k = n_total
        F_partial = ((r2_full - r2_restr) / q) / ((1 - r2_full) / (n_obs - k))
        F_p       = 1 - stats.f.cdf(F_partial, q, n_obs - k)

        out[col] = {
            "F_statistic":         F_partial,
            "F_p_value":           F_p,
            "partial_R2":          r2_full - r2_restr,
            "first_stage_R2":      r2_full,
            "n_instruments":       n_instr,
            "weak_instrument":     F_partial < 10,
            "interpretation": (
                f"F = {F_partial:.2f} — "
                + ("STRONG instrument (F≥10)" if F_partial >= 10
                   else "WEAK instrument (F<10) — IV estimates unreliable")
            ),
        }
    return out


def weak_instrument_test(result, first_stage_f: float = None) -> dict:
    """
    Report Stock-Yogo critical values for weak instrument rejection.

    Stock & Yogo (2005) critical values for 5% Wald test size:
    b = 0.10 (10% maximal relative bias): F > 16.38 (1 instrument)
    b = 0.15:                             F > 8.96
    b = 0.20:                             F > 6.66
    b = 0.25:                             F > 5.53
    Rule of thumb (Staiger-Stock): F > 10
    """
    # try to extract F from linearmodels result
    if first_stage_f is None:
        try:
            first_stage_f = result.first_stage.diagnostics["f.stat"].iloc[0]
        except Exception:
            first_stage_f = None

    stock_yogo = {
        "10% max relative bias (F>16.38)": first_stage_f >= 16.38 if first_stage_f else None,
        "15% max relative bias (F>8.96)":  first_stage_f >= 8.96  if first_stage_f else None,
        "20% max relative bias (F>6.66)":  first_stage_f >= 6.66  if first_stage_f else None,
        "Rule of thumb F>10":              first_stage_f >= 10    if first_stage_f else None,
    }

    return {
        "first_stage_F": first_stage_f,
        "stock_yogo_tests": stock_yogo,
        "recommendation": (
            "Proceed with IV estimates" if (first_stage_f or 0) >= 10
            else "Consider LIML or Anderson-Rubin confidence sets (weak instrument)"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. WU-HAUSMAN ENDOGENEITY TEST
# ─────────────────────────────────────────────────────────────────────────────

def hausman_test_iv(y, endog, instruments, exog) -> dict:
    """
    Wu-Hausman test: H0 = OLS is consistent (X is exogenous).

    Method (Durbin-Wu-Hausman control function approach):
    1. Regress each endogenous variable on all instruments and exogenous controls.
    2. Save residuals v̂.
    3. Include v̂ in OLS of y on X and controls.
    4. Test H0: coefficient on v̂ = 0 via F-test.

    Under H0: OLS is consistent; reject ⟹ IV is preferred.
    """
    Z = pd.concat([exog, instruments], axis=1)
    residuals = {}
    for col in endog.columns:
        fs = sm.OLS(endog[col], Z).fit()
        residuals[col] = fs.resid.rename(f"resid_{col}")

    resid_df = pd.DataFrame(residuals)
    X_aug = pd.concat([exog, endog, resid_df], axis=1)

    aug_model = sm.OLS(y, X_aug).fit(cov_type="HC3")

    # F-test on residual coefficients
    resid_names = [f"resid_{col}" for col in endog.columns]
    try:
        f_test = aug_model.f_test([f"({name} = 0)" for name in resid_names])
        F_stat = float(f_test.fvalue)
        p_val  = float(f_test.pvalue)
    except Exception:
        # manual F-test
        n_restr = len(resid_names)
        n_obs   = len(y)
        n_params = X_aug.shape[1]
        # restricted model (without residuals)
        restr = sm.OLS(y, pd.concat([exog, endog], axis=1)).fit()
        F_stat = ((restr.ssr - aug_model.ssr) / n_restr) / (aug_model.ssr / (n_obs - n_params))
        p_val  = 1 - stats.f.cdf(F_stat, n_restr, n_obs - n_params)

    return {
        "F_statistic": F_stat,
        "p_value": p_val,
        "reject_H0": p_val < 0.05,
        "interpretation": (
            f"Endogeneity confirmed (p={p_val:.4f}) — use IV/2SLS" if p_val < 0.05
            else f"Cannot reject exogeneity (p={p_val:.4f}) — OLS may be consistent"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SARGAN-HANSEN OVERIDENTIFICATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def sargan_test(iv_result, y, endog, instruments, exog) -> dict:
    """
    Sargan-Hansen J-test for overidentifying restrictions.

    H0: all instruments are valid (uncorrelated with structural error).
    Rejection suggests at least one instrument is endogenous or mis-specified.
    Only valid in over-identified models (# instruments > # endogenous regressors).

    J = n · R² from regressing 2SLS residuals on all instruments and exogenous vars.
    J ~ χ²(m - k) under H0, where m = # instruments, k = # endogenous variables.
    """
    n_endog = endog.shape[1]
    n_instr = instruments.shape[1]
    if n_instr <= n_endog:
        return {"error": "Model is exactly identified — Sargan test requires over-identification"}

    # get 2SLS residuals
    try:
        resid = iv_result.resids
    except AttributeError:
        resid = iv_result.resid

    # regress residuals on all instruments + exog
    Z_all = pd.concat([exog, instruments], axis=1)
    aux   = sm.OLS(resid, Z_all).fit()

    n  = len(y)
    J  = n * aux.rsquared
    df = n_instr - n_endog
    p  = 1 - stats.chi2.cdf(J, df)

    return {
        "J_statistic": J,
        "df": df,
        "p_value": p,
        "reject_H0": p < 0.05,
        "interpretation": (
            f"Overidentifying restrictions rejected (p={p:.4f}) — suspect instrument validity"
            if p < 0.05
            else f"Cannot reject validity of instruments (p={p:.4f})"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPARISON TABLE: OLS vs IV
# ─────────────────────────────────────────────────────────────────────────────

def compare_ols_iv(
    y: pd.Series,
    endog: pd.DataFrame,
    instruments: pd.DataFrame,
    exog: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Side-by-side OLS vs 2SLS estimates for each regressor.

    OLS is biased if endogeneity exists; IV corrects this at the cost of larger SE.
    The direction of divergence between OLS and IV reveals the nature of the bias.
    """
    if exog is None:
        exog = pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
    elif "const" not in exog.columns:
        exog = exog.copy(); exog.insert(0, "const", 1.0)

    X_ols = pd.concat([exog, endog], axis=1)
    ols   = sm.OLS(y, X_ols).fit(cov_type="HC3")
    iv_out = run_iv2sls(y, endog, instruments, exog, cov_type="robust")

    try:
        iv_params = iv_out["result"].params
        iv_bse    = iv_out["result"].std_errors
        iv_pvals  = iv_out["result"].pvalues
    except AttributeError:
        iv_params = iv_out["result"].params
        iv_bse    = iv_out["result"].bse
        iv_pvals  = iv_out["result"].pvalues

    rows = []
    all_vars = list(dict.fromkeys(list(ols.params.index) + list(iv_params.index)))
    for v in all_vars:
        row = {"variable": v}
        if v in ols.params:
            row.update({
                "ols_coef": ols.params[v],
                "ols_se":   ols.bse[v],
                "ols_p":    ols.pvalues[v],
            })
        if v in iv_params:
            row.update({
                "iv_coef": iv_params[v],
                "iv_se":   iv_bse[v] if v in iv_bse.index else np.nan,
                "iv_p":    iv_pvals[v] if v in iv_pvals.index else np.nan,
            })
        rows.append(row)

    df = pd.DataFrame(rows).set_index("variable")
    df["bias_ols"] = df["ols_coef"] - df["iv_coef"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_iv_summary(result, diag: dict) -> str:
    lines = ["=" * 60, "IV/2SLS DIAGNOSTIC SUMMARY", "=" * 60]

    fs = diag.get("first_stage", {})
    for var, info in fs.items():
        lines += [
            f"First Stage — {var}:",
            f"  F = {info['F_statistic']:.2f}  (p={info['F_p_value']:.4f})",
            f"  Partial R² = {info['partial_R2']:.4f}",
            f"  {info['interpretation']}",
            "",
        ]

    wh = diag.get("hausman", {})
    if "F_statistic" in wh:
        lines += [
            "Wu-Hausman Test:",
            f"  F = {wh['F_statistic']:.4f}  p = {wh['p_value']:.4f}",
            f"  {wh['interpretation']}",
            "",
        ]

    sg = diag.get("sargan", {})
    if "J_statistic" in sg:
        lines += [
            "Sargan-Hansen J-Test:",
            f"  J = {sg['J_statistic']:.4f}  df = {sg['df']}  p = {sg['p_value']:.4f}",
            f"  {sg['interpretation']}",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)
```

---

## Example A — Returns to Education (Card 1995)

Card (1995) used geographic proximity to a college as an instrument for education.
Rationale: people who grew up near a college face lower costs of attending, so they
get more schooling — but proximity does not directly affect wages.

```python
# example_a_card_education.py
"""
Card (1995) style IV: proximity to college as instrument for education.
Simulated data replicating the key features of the NLS Young Men survey.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iv_2sls import run_iv2sls, compare_ols_iv, weak_instrument_test

rng = np.random.default_rng(42)
n   = 3010   # approximate NLS sample size

# ── latent ability (unobserved) ──
ability = rng.normal(0, 1, n)

# ── instrument: distance to nearest 4-year college (binary: close = 1) ──
near_college = rng.binomial(1, 0.40, n).astype(float)

# ── background controls ──
exp      = rng.uniform(1, 25, n)
exp2     = exp**2
black    = rng.binomial(1, 0.25, n).astype(float)
south    = rng.binomial(1, 0.35, n).astype(float)
urban    = rng.binomial(1, 0.60, n).astype(float)

# ── education: near_college raises schooling by ~0.8 years ──
educ_noise = rng.normal(0, 1.5, n)
educ = (
    12 + 0.8 * near_college
    + 2.0 * ability            # ability raises schooling
    - 0.3 * black
    + educ_noise
).clip(8, 20)

# ── log wage: true return to education = 0.10 ──
wage_noise = rng.normal(0, 0.25, n)
lnwage = (
    4.0 + 0.10 * educ
    + 0.04 * exp - 0.0007 * exp2
    + 0.4 * ability             # endogeneity: ability affects both educ and wage
    - 0.15 * black
    + 0.10 * urban
    + wage_noise
)

df = pd.DataFrame({
    "lnwage":      lnwage,
    "educ":        educ,
    "exp":         exp,
    "exp2":        exp2,
    "black":       black,
    "south":       south,
    "urban":       urban,
    "near_college": near_college,
})

y    = df["lnwage"]
endog = df[["educ"]]
instruments = df[["near_college"]]
exog = sm.add_constant(df[["exp", "exp2", "black", "south", "urban"]])

import statsmodels.api as sm

exog_df = pd.DataFrame(
    sm.add_constant(df[["exp", "exp2", "black", "south", "urban"]]),
    index=df.index, columns=["const","exp","exp2","black","south","urban"],
)

# ── run IV ──
iv_out = run_iv2sls(y, endog, instruments, exog_df, cov_type="robust")
print(iv_out["summary"])

# ── OLS vs IV comparison ──
compare = compare_ols_iv(y, endog, instruments, exog_df)
print("\nOLS vs IV Comparison:")
print(compare[["ols_coef","ols_se","iv_coef","iv_se","bias_ols"]].round(4))

# ── weak instrument check ──
fs_f = iv_out["diagnostics"]["first_stage"]["educ"]["F_statistic"]
wi   = weak_instrument_test(iv_out["result"], first_stage_f=fs_f)
print(f"\nFirst-stage F = {fs_f:.1f}")
print("Stock-Yogo tests:")
for k, v in wi["stock_yogo_tests"].items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print("Recommendation:", wi["recommendation"])

# ── visualise OLS vs IV ──
fig, ax = plt.subplots(figsize=(8, 5))
ols_coef = float(compare.loc["educ", "ols_coef"])
iv_coef  = float(compare.loc["educ", "iv_coef"])
ols_se   = float(compare.loc["educ", "ols_se"])
iv_se    = float(compare.loc["educ", "iv_se"])
true_val = 0.10

estimators = ["True value", "OLS", "IV (2SLS)"]
coefs = [true_val, ols_coef, iv_coef]
errors = [0, ols_se, iv_se]
colors = ["#636363", "#d7191c", "#2c7bb6"]

ax.barh(estimators, coefs, xerr=[1.96*e for e in errors],
        color=colors, height=0.4, capsize=5, alpha=0.8)
ax.axvline(true_val, color="#636363", linestyle="--", linewidth=1.5)
ax.set_xlabel("Return to schooling (log wage per year of education)")
ax.set_title("OLS vs IV Estimates of Return to Education\n(Card 1995 style)")
plt.tight_layout()
fig.savefig("iv_card.png", dpi=150, bbox_inches="tight")
print("\nSaved iv_card.png")
```

---

## Example B — Rainfall as IV for Conflict

Instruments rainfall (a supply shock to agricultural income) for conflict intensity,
following Miguel, Satyanath & Sergenti (2004). Lower rainfall → income shock → more conflict.

```python
# example_b_rainfall_conflict.py
"""
IV: rainfall as instrument for income shocks predicting conflict onset.
Panel data: country-year observations.
Exclusion restriction: rainfall affects conflict only through income.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from iv_2sls import run_iv2sls, _first_stage_stats, hausman_test_iv

rng = np.random.default_rng(77)
n_countries = 40
n_years     = 20
N           = n_countries * n_years

country_id = np.repeat(np.arange(n_countries), n_years)
year_id    = np.tile(np.arange(n_years), n_countries)

# ── rainfall: exogenous weather shock ──
rainfall = rng.normal(0, 1, N)   # standardised log deviation from mean

# ── income growth: driven partly by rainfall (agricultural channel) ──
country_fe = rng.normal(0, 0.5, n_countries)[country_id]  # fixed effect
income_growth = (
    0.0 + 0.4 * rainfall
    - 0.3 * country_fe    # structural heterogeneity
    + rng.normal(0, 0.8, N)
)

# ── conflict: income shocks worsen conflict; latent unobservable correlated with both ──
u_struct    = rng.normal(0, 0.5, N)
conflict    = (
    0.5 - 1.2 * income_growth   # true causal effect
    + 1.5 * country_fe          # omitted variable: country FE also affects conflict
    + u_struct
)
conflict_bin = (conflict > conflict.mean()).astype(float)

# ── population and ethnic fractionalization (controls) ──
log_pop  = rng.normal(15, 2, N)
eth_frac = rng.beta(2, 2, n_countries)[country_id]

df = pd.DataFrame({
    "conflict":      conflict,
    "conflict_bin":  conflict_bin,
    "income_growth": income_growth,
    "rainfall":      rainfall,
    "log_pop":       log_pop,
    "eth_frac":      eth_frac,
    "country":       country_id,
    "year":          year_id,
})

# add country fixed effects as dummies (partial out via demeaning)
# Demeaning (within transformation) is simpler for illustration
df["conflict_dm"]      = df["conflict"]      - df.groupby("country")["conflict"].transform("mean")
df["income_growth_dm"] = df["income_growth"] - df.groupby("country")["income_growth"].transform("mean")
df["rainfall_dm"]      = df["rainfall"]      - df.groupby("country")["rainfall"].transform("mean")
df["log_pop_dm"]       = df["log_pop"]       - df.groupby("country")["log_pop"].transform("mean")

y           = df["conflict_dm"]
endog_df    = df[["income_growth_dm"]]
instr_df    = df[["rainfall_dm"]]
exog_df     = pd.DataFrame({
    "const":   1.0,
    "log_pop": df["log_pop_dm"],
}, index=df.index)

# ── IV estimation ──
iv_out = run_iv2sls(y, endog_df, instr_df, exog_df, cov_type="robust")

fs = iv_out["diagnostics"]["first_stage"]["income_growth_dm"]
wh = iv_out["diagnostics"]["hausman"]

print("=== Rainfall IV for Conflict ===")
print(f"First-stage F = {fs['F_statistic']:.2f}  ({fs['interpretation']})")
print(f"Wu-Hausman:     {wh['interpretation']}")
print()

try:
    params = iv_out["result"].params
    se     = iv_out["result"].std_errors
    pvals  = iv_out["result"].pvalues
except AttributeError:
    params = iv_out["result"].params
    se     = iv_out["result"].bse
    pvals  = iv_out["result"].pvalues

print("IV coefficient on income growth:", f"{params['income_growth_dm']:.3f}")
print("SE:", f"{se['income_growth_dm']:.3f}")
print("p-value:", f"{pvals['income_growth_dm']:.4f}")
print("\nInterpretation: A 1 std-dev decline in income growth increases conflict")
print(f"by approximately {abs(params['income_growth_dm']):.2f} standard deviations.")
```

---

## IV Validity Checklist

| Criterion | How to Verify | What Failure Means |
|---|---|---|
| Relevance | First-stage F ≥ 10 (Stock-Yogo) | Weak instrument bias toward OLS |
| Exclusion restriction | Economic theory + domain expertise | IV is invalid — no statistical fix |
| Monotonicity (fuzzy IV) | Verify no defiers in the population | LATE interpretation breaks down |
| Homogeneity (if targeting ATE) | Assess treatment heterogeneity | IV recovers LATE, not ATE |
| No defiance | Context knowledge | Negative first-stage for some units |

---

## LIML vs 2SLS

LIML (Limited Information Maximum Likelihood) is preferred over 2SLS when instruments are
weak:
- LIML is **median-unbiased** even with weak instruments (2SLS is not).
- LIML has heavier tails than 2SLS — confidence intervals are wider but more honest.
- For strong instruments (F ≥ 10), 2SLS and LIML give nearly identical estimates.

```python
# switch to LIML by passing estimator='liml':
iv_liml = run_iv2sls(y, endog_df, instr_df, exog_df,
                     cov_type="robust", estimator="liml")
```

---

## References

- Card, D. (1995). Using geographic variation in college proximity to estimate the return
  to schooling. In *Aspects of Labour Market Behaviour* (pp. 201–222). U of Toronto Press.
- Angrist, J. D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton UP.
- Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression.
  In *Identification and Inference for Econometric Models* (pp. 80–108). Cambridge UP.
- Miguel, E., Satyanath, S., & Sergenti, E. (2004). Economic shocks and civil conflict.
  *Journal of Political Economy*, 112(4), 725–753.
- Baum, C. F., Schaffer, M. E., & Stillman, S. (2007). Enhanced routines for instrumental
  variables/generalized method of moments estimation and testing. *Stata Journal*, 7(4).
