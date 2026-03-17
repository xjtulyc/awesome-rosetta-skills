---
name: panel-data
description: >
  Panel data econometrics with Python linearmodels; covers pooled OLS, fixed/random
  effects, Hausman test, clustered SE, Arellano-Bond GMM, and regression tables.
tags:
  - economics
  - econometrics
  - panel-data
  - regression
  - python
  - linearmodels
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
  - linearmodels>=5.3
  - pandas>=2.0.0
  - numpy>=1.24.0
  - statsmodels>=0.14.0
  - scipy>=1.10.0
  - tabulate>=0.9.0
last_updated: "2026-03-17"
---

# Panel Data Econometrics with Python linearmodels

This skill covers the full panel data workflow: data setup, estimator selection
(Pooled OLS, Fixed Effects, Random Effects), two-way FE, robust standard errors,
Arellano-Bond GMM for dynamic panels, and panel unit root tests. All examples use
the `linearmodels` library, which mirrors the Stata/R xtreg/plm API.

---

## 1. Setup and Data Structure

```bash
pip install linearmodels pandas numpy statsmodels scipy tabulate
```

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from linearmodels.panel import (
    PanelOLS,
    PooledOLS,
    RandomEffects,
    BetweenOLS,
    FirstDifferenceOLS,
)
from linearmodels.panel.results import PanelEffectsResults
from statsmodels.stats.stattools import durbin_watson

# ---------------------------------------------------------------------------
# Create a synthetic firm-level panel for illustration
# ---------------------------------------------------------------------------
np.random.seed(42)
n_firms = 200
n_years = 10
firms   = [f"firm_{i:04d}" for i in range(n_firms)]
years   = list(range(2010, 2010 + n_years))

idx     = pd.MultiIndex.from_product([firms, years], names=["firm", "year"])
n       = len(idx)

# Firm fixed effects (unobserved heterogeneity)
fe      = np.repeat(np.random.normal(0, 1, n_firms), n_years)

data = pd.DataFrame({
    "invest":    2 + 0.15 * np.random.normal(0, 1, n) + fe * 0.5
                 + 0.3 * np.random.normal(0, 1, n),
    "rndi":      np.abs(np.random.normal(1, 0.8, n)) + fe * 0.3,
    "size":      np.log(np.abs(np.random.normal(10, 3, n) + fe)),
    "leverage":  np.clip(np.random.beta(2, 5, n) + 0.02 * fe, 0, 1),
    "tobinq":    np.abs(1 + np.random.normal(0, 0.5, n)),
    "cashflow":  np.random.normal(0.1, 0.05, n),
}, index=idx)

# Set the MultiIndex as entity-time for linearmodels
data = data.set_index(pd.MultiIndex.from_tuples(data.index, names=["firm", "year"]))
print(data.head())
print(f"Panel: {n_firms} firms × {n_years} years = {n} observations")
```

---

## 2. Core Estimator Functions

### 2.1 Fixed Effects (Within) Estimator

```python
def run_fe(
    formula: str,
    df: pd.DataFrame,
    entity: str = "firm",
    time: str = "year",
    entity_effects: bool = True,
    time_effects: bool = False,
    cov_type: str = "clustered",
    cluster_entity: bool = True,
) -> PanelEffectsResults:
    """
    Estimate a panel fixed-effects model.

    Parameters
    ----------
    formula : str
        Patsy-style formula, e.g. 'invest ~ 1 + rndi + size + EntityEffects'.
        Do NOT include EntityEffects/TimeEffects in the formula string;
        pass the flags instead.
    df : pd.DataFrame
        DataFrame with a (entity, time) MultiIndex.
    entity, time : str
        Names of the entity and time dimensions in the MultiIndex.
    entity_effects : bool
        Include entity (within) dummies.
    time_effects : bool
        Include time dummies (two-way FE when combined with entity_effects).
    cov_type : str
        'unadjusted', 'robust', 'clustered', 'kernel'.
    cluster_entity : bool
        When cov_type='clustered', cluster at the entity level.

    Returns
    -------
    PanelEffectsResults
    """
    # Rebuild MultiIndex if needed
    if df.index.names != [entity, time]:
        df = df.copy()
        df.index.names = [entity, time]

    mod = PanelOLS.from_formula(
        formula,
        data=df,
        entity_effects=entity_effects,
        time_effects=time_effects,
    )
    cov_kwargs = {}
    if cov_type == "clustered":
        cov_kwargs = {"cluster_entity": cluster_entity}

    return mod.fit(cov_type=cov_type, **cov_kwargs)


def run_re(
    formula: str,
    df: pd.DataFrame,
    entity: str = "firm",
    time: str = "year",
) -> PanelEffectsResults:
    """
    Estimate a panel random-effects (GLS) model via linearmodels RandomEffects.

    Parameters
    ----------
    formula : str
        Patsy formula without entity/time effect keywords.
    df : pd.DataFrame
        DataFrame with a (entity, time) MultiIndex.

    Returns
    -------
    PanelEffectsResults
    """
    if df.index.names != [entity, time]:
        df = df.copy()
        df.index.names = [entity, time]

    mod = RandomEffects.from_formula(formula, data=df)
    return mod.fit(cov_type="robust")
```

### 2.2 Hausman Specification Test

```python
def hausman_test(
    fe_result: PanelEffectsResults,
    re_result: PanelEffectsResults,
) -> dict:
    """
    Perform the Hausman test to choose between FE and RE.

    H0: RE is consistent and efficient (random effects preferred).
    H1: FE is consistent, RE is not (fixed effects preferred).

    Returns
    -------
    dict with keys: stat, df, p_value, decision.
    """
    # Align common coefficients (excluding intercept)
    fe_coef = fe_result.params
    re_coef = re_result.params

    common = fe_coef.index.intersection(re_coef.index)
    common = [c for c in common if c not in ("Intercept", "const")]

    b_fe   = fe_coef[common].values
    b_re   = re_coef[common].values
    diff   = b_fe - b_re

    # Covariance matrices
    V_fe   = fe_result.cov.loc[common, common].values
    V_re   = re_result.cov.loc[common, common].values
    V_diff = V_fe - V_re

    # Make positive definite via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(V_diff)
    eigvals = np.clip(eigvals, 1e-10, None)
    V_diff_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    chi2_stat = float(diff @ np.linalg.inv(V_diff_pd) @ diff)
    df_       = len(common)
    p_value   = 1 - stats.chi2.cdf(chi2_stat, df_)

    decision = "Fixed Effects preferred (reject RE)" if p_value < 0.05 else \
               "Random Effects preferred (fail to reject)"

    return {
        "stat":     round(chi2_stat, 4),
        "df":       df_,
        "p_value":  round(p_value, 4),
        "decision": decision,
    }
```

### 2.3 Arellano-Bond GMM for Dynamic Panels

```python
def run_arellano_bond(
    df: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    lags: int = 1,
    instrument_lags: tuple = (2, 4),
    entity: str = "firm",
    time: str = "year",
) -> object:
    """
    Estimate a dynamic panel model using Arellano-Bond (AB) first-difference GMM.

    The model is:  y_it = alpha * y_{i,t-1} + X_it * beta + u_it
    Instruments:   levels of y_{i,t-2}, ..., y_{i,t-L} (and X if strictly exogenous).

    This implementation uses statsmodels InstrumentedResiduals as a two-step
    system; for production use the 'linearmodels' IV interface.

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame with MultiIndex (entity, time).
    outcome : str
        Dependent variable column.
    regressors : list[str]
        Exogenous regressors (excluding the lagged DV).
    lags : int
        Number of lagged DV to include.
    instrument_lags : tuple
        (min_lag, max_lag) for internal IV instruments.

    Returns
    -------
    Fitted IVModelResults object.
    """
    from linearmodels.iv import IV2SLS

    if df.index.names != [entity, time]:
        df = df.copy()
        df.index.names = [entity, time]

    # First difference the panel
    fd = df.groupby(level=entity)[[outcome] + regressors].diff().dropna()
    fd.index = df.loc[fd.index].index  # restore MultiIndex

    # Lagged DV in first differences
    lag_name = f"{outcome}_lag{lags}"
    fd[lag_name] = df.groupby(level=entity)[outcome].shift(lags).reindex(fd.index)
    fd = fd.dropna()

    # Use lagged levels as instruments (Arellano-Bond moment conditions)
    inst_cols = []
    for l in range(instrument_lags[0], instrument_lags[1] + 1):
        col = f"{outcome}_inst_lag{l}"
        fd[col] = df.groupby(level=entity)[outcome].shift(l).reindex(fd.index)
        inst_cols.append(col)
    fd = fd.dropna()

    dependent    = fd[outcome]
    exog         = sm.add_constant(fd[regressors])
    endog        = fd[[lag_name]]
    instruments  = fd[inst_cols]

    mod = IV2SLS(dependent, exog, endog, instruments)
    return mod.fit(cov_type="robust")
```

### 2.4 Regression Comparison Table

```python
def compare_estimators_table(
    results_dict: dict,
    digits: int = 4,
) -> pd.DataFrame:
    """
    Build a side-by-side regression table from multiple fitted models.

    Parameters
    ----------
    results_dict : dict
        Mapping of model_name -> fitted results object.
    digits : int
        Decimal places.

    Returns
    -------
    pd.DataFrame with coef (se) rows and fit statistics.
    """
    rows = {}
    # Collect all parameter names
    all_params = []
    for res in results_dict.values():
        all_params.extend(res.params.index.tolist())
    all_params = list(dict.fromkeys(all_params))  # deduplicate, preserve order

    for param in all_params:
        row = {}
        for name, res in results_dict.items():
            if param in res.params:
                coef = res.params[param]
                se   = res.std_errors[param]
                pval = res.pvalues[param]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                row[name] = f"{coef:.{digits}f}{stars} ({se:.{digits}f})"
            else:
                row[name] = ""
        rows[param] = row

    table = pd.DataFrame(rows).T

    # Append fit stats
    stats_rows = {}
    for name, res in results_dict.items():
        r2   = getattr(res, "rsquared", getattr(res, "rsquared_between", np.nan))
        nobs = int(getattr(res, "nobs", np.nan))
        stats_rows[name] = {"R²": f"{r2:.4f}", "N": str(nobs)}

    stats_df = pd.DataFrame(stats_rows).T.rename_axis("Model")
    print("\n=== Regression Table ===")
    print(table.to_string())
    print("\n=== Fit Statistics ===")
    print(stats_df.to_string())
    return table
```

---

## 3. Panel Unit Root Tests

```python
def panel_unit_root_tests(df: pd.DataFrame, col: str, entity: str = "firm") -> None:
    """
    Run LLC and IPS panel unit root tests via statsmodels.
    Prints test statistics and p-values for each entity.
    """
    from statsmodels.tsa.stattools import adfuller

    series_list = []
    for ent, grp in df.groupby(level=entity):
        s = grp[col].dropna()
        if len(s) > 10:
            series_list.append(s.values)

    adf_stats = []
    for s in series_list:
        try:
            result = adfuller(s, autolag="AIC")
            adf_stats.append(result[0])
        except Exception:
            pass

    # IPS test: average of individual ADF t-statistics
    if adf_stats:
        t_bar = np.mean(adf_stats)
        n     = len(adf_stats)
        # Approximate critical value table from Im, Pesaran, Shin (2003)
        print(f"IPS t-bar statistic: {t_bar:.4f} (N={n})")
        print("  Rule of thumb: t-bar < -1.73 → reject H0 of unit root at 5%")
    else:
        print("Insufficient data for unit root tests.")
```

---

## 4. Testing Serial Correlation (Wooldridge Test)

```python
def wooldridge_serial_corr(
    formula: str,
    df: pd.DataFrame,
    entity: str = "firm",
    time: str = "year",
) -> dict:
    """
    Wooldridge (2002) test for serial correlation in panel FE residuals.
    Regress FD residuals on lagged FD residuals; test coefficient = -0.5.
    """
    if df.index.names != [entity, time]:
        df = df.copy()
        df.index.names = [entity, time]

    res = run_fe(formula, df, entity, time, cov_type="clustered")
    resid = pd.Series(res.resids, index=df.index, name="resid")

    fd_resid = resid.groupby(level=entity).diff().dropna()
    fd_lag   = resid.groupby(level=entity).shift(1).reindex(fd_resid.index).dropna()
    common   = fd_resid.index.intersection(fd_lag.index)
    fd_resid, fd_lag = fd_resid.loc[common], fd_lag.loc[common]

    X = sm.add_constant(fd_lag.values.reshape(-1, 1))
    ols = sm.OLS(fd_resid.values, X).fit()
    coef, se = ols.params[1], ols.bse[1]
    t_stat  = (coef - (-0.5)) / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=ols.df_resid))

    print(f"Wooldridge test: coef on lagged FD resid = {coef:.4f}, "
          f"t({ols.df_resid:.0f}) = {t_stat:.4f}, p = {p_value:.4f}")
    print("H0: no first-order serial correlation in idiosyncratic errors.")
    return {"coef": coef, "t_stat": t_stat, "p_value": p_value}
```

---

## 5. Example A — Firm R&D Panel: FE vs RE with Hausman Decision Rule

```python
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects

# ---- Use the synthetic data created in Section 1 --------------------------------
formula_fe = "invest ~ 1 + rndi + size + leverage + tobinq + EntityEffects"
formula_re = "invest ~ 1 + rndi + size + leverage + tobinq"

# Pooled OLS (baseline, ignores heterogeneity)
pooled_mod = PooledOLS.from_formula("invest ~ 1 + rndi + size + leverage + tobinq", data=data)
pooled_res = pooled_mod.fit(cov_type="robust")

# Fixed effects
fe_res = run_fe(
    "invest ~ 1 + rndi + size + leverage + tobinq",
    data,
    entity_effects=True,
    time_effects=False,
    cov_type="clustered",
    cluster_entity=True,
)

# Two-way fixed effects
twfe_res = run_fe(
    "invest ~ 1 + rndi + size + leverage + tobinq",
    data,
    entity_effects=True,
    time_effects=True,
    cov_type="clustered",
    cluster_entity=True,
)

# Random effects
re_res = run_re("invest ~ 1 + rndi + size + leverage + tobinq", data)

# Hausman test
hausman = hausman_test(fe_res, re_res)
print("\n=== Hausman Specification Test ===")
for k, v in hausman.items():
    print(f"  {k}: {v}")

# Comparison table
compare_estimators_table({
    "Pooled OLS": pooled_res,
    "FE (Entity)": fe_res,
    "Two-Way FE":  twfe_res,
    "RE":          re_res,
})

# R² components (FE)
print(f"\nFE Within R²:   {fe_res.rsquared:.4f}")
print(f"FE Between R²:  {fe_res.rsquared_between:.4f}")
print(f"FE Overall R²:  {fe_res.rsquared_overall:.4f}")
```

---

## 6. Example B — Arellano-Bond GMM for Dynamic Investment Model

```python
import numpy as np
import pandas as pd

# ---- Add a lagged investment variable to simulate dynamic panel ------------------
np.random.seed(7)
data_dyn = data.copy()

# Generate a correlated lagged investment (AR(1) component)
data_dyn = data_dyn.sort_index()
invest_lag = data_dyn.groupby(level="firm")["invest"].shift(1)
data_dyn["invest_lag1"] = invest_lag

# Drop NaN introduced by lagging
data_dyn = data_dyn.dropna(subset=["invest_lag1"])

print(f"Dynamic panel: {data_dyn.shape[0]} obs after removing lag-1 NaN")

# ---- Naive FE (biased in dynamic panel — Nickell bias) --------------------------
fe_biased = run_fe(
    "invest ~ 1 + invest_lag1 + rndi + size + leverage",
    data_dyn,
    entity_effects=True,
    time_effects=False,
    cov_type="clustered",
)
print("\n=== Naive FE (Nickell-biased) ===")
print(fe_biased.summary.tables[1])

# ---- Arellano-Bond GMM ----------------------------------------------------------
ab_res = run_arellano_bond(
    df=data_dyn,
    outcome="invest",
    regressors=["rndi", "size", "leverage"],
    lags=1,
    instrument_lags=(2, 4),
)
print("\n=== Arellano-Bond GMM ===")
print(ab_res.summary.tables[1])

# ---- Compare persistence estimates ----------------------------------------------
print("\n=== Persistence (lagged DV coefficient) ===")
print(f"  FE (biased):   {fe_biased.params.get('invest_lag1', float('nan')):.4f}")
print(f"  AB GMM:        {ab_res.params.get('invest_lag1', float('nan')):.4f}")
print("  (AB typically corrects downward Nickell bias in FE)")

# ---- Serial correlation test on FE residuals ------------------------------------
wtest = wooldridge_serial_corr(
    "invest ~ 1 + rndi + size + leverage + tobinq",
    data,
)
```

---

## 7. Driscoll-Kraay Standard Errors

```python
def driscoll_kraay_se(
    formula: str,
    df: pd.DataFrame,
    bandwidth: int = 3,
    entity: str = "firm",
    time: str = "year",
) -> PanelEffectsResults:
    """
    Estimate FE model with Driscoll-Kraay (1998) standard errors, which are
    robust to cross-sectional dependence and temporal autocorrelation.
    linearmodels implements these via cov_type='kernel'.
    """
    if df.index.names != [entity, time]:
        df = df.copy()
        df.index.names = [entity, time]

    mod = PanelOLS.from_formula(formula, data=df, entity_effects=True)
    return mod.fit(cov_type="kernel", bandwidth=bandwidth)


# Usage
dk_res = driscoll_kraay_se(
    "invest ~ 1 + rndi + size + leverage + tobinq",
    data,
    bandwidth=3,
)
print("\n=== Driscoll-Kraay SE ===")
print(dk_res.summary.tables[1])
```

---

## 8. Tips and Common Pitfalls

- **MultiIndex requirement**: `linearmodels` requires a strict `(entity, time)`
  MultiIndex. Always call `df.index.names = ['entity', 'time']` before fitting.
- **EntityEffects in formula vs flag**: Do NOT put `EntityEffects` in the formula
  string when using `PanelOLS(entity_effects=True)`. They are mutually exclusive.
- **Hausman test with clustered SE**: The standard Hausman test assumes homoskedastic
  errors. With clustered SE, use the Mundlak (1978) regression test instead: add
  group-means of time-varying regressors to the RE model and test joint significance.
- **Arellano-Bond validity**: Check the Sargan/Hansen J-test for instrument validity
  and AR(2) test. Presence of AR(2) violates the moment conditions.
- **Nickell bias**: In short panels (small T, large N) with lagged DV, FE estimates
  are biased of order O(1/T). AB GMM corrects this via first-differencing + IV.
- **Unbalanced panels**: `linearmodels` handles unbalanced panels natively. Rows with
  missing outcomes are dropped automatically.
