---
name: time-series-econometrics
description: >
  Use this Skill for multivariate time series: VAR, Granger causality, Johansen
  cointegration, VECM, impulse response functions, and forecast error variance
  decomposition.
tags:
  - economics
  - VAR
  - cointegration
  - Granger-causality
  - impulse-response
  - VECM
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
    - statsmodels>=0.14
    - arch>=6.0
    - numpy>=1.23
    - pandas>=1.5
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Multivariate Time Series Econometrics

> **TL;DR** — Fit VAR models, test Granger causality, test for cointegration with
> Johansen, estimate VECMs, plot impulse response functions with bootstrap confidence
> intervals, and decompose forecast error variance.

---

## When to Use

| Situation | Recommended Tool |
|---|---|
| Stationary multivariate system, reduced-form dynamics | VAR(p) |
| Does X help predict Y beyond Y's own lags? | Granger causality F-test |
| Non-stationary series that may share long-run trends | Johansen cointegration test |
| Cointegrated system with error correction | VECM |
| Dynamic response of one variable to a shock in another | Impulse Response Functions (IRF) |
| How much of Y's forecast variance is explained by X? | Forecast Error Variance Decomposition (FEVD) |

This Skill is the starting point for any empirical work involving multiple time series
in economics and finance — GDP, inflation, interest rates, exchange rates, commodity
prices, and asset prices.

---

## Background

### VAR(p) Model

A VAR(p) model for k-dimensional vector Y_t is:

    Y_t = c + A_1 Y_{t-1} + A_2 Y_{t-2} + ... + A_p Y_{t-p} + u_t

where u_t ~ N(0, Σ) is the innovation vector. Lag p is selected by information
criteria: AIC, BIC (Schwarz), or HQIC.

**VAR stability condition**: All eigenvalues of the companion matrix must lie strictly
inside the unit circle. If any eigenvalue >= 1, the system is non-stationary and
you should consider VECM.

### Granger Causality

Variable X Granger-causes Y if knowing past X improves forecasts of Y beyond past Y
alone. Tested via an F-test on the restriction that all lagged X coefficients in the
Y equation are jointly zero.

This is a predictive concept, not a structural causal one. Granger causality ≠
structural causality.

### Johansen Cointegration

Two or more I(1) series are cointegrated if a linear combination is I(0). Johansen
(1988) provides two likelihood-ratio tests:

- **Trace test**: H0 = at most r cointegrating vectors
- **Maximum eigenvalue test**: H0 = r vs H1 = r+1 cointegrating vectors

Critical values differ by deterministic specification (no constant, constant in CE,
trend in CE). Use `statsmodels.tsa.vector_ar.vecm.coint_johansen`.

### VECM

A VECM re-parameterizes the VAR for I(1) cointegrated variables:

    ΔY_t = c + Π Y_{t-1} + Γ_1 ΔY_{t-1} + ... + Γ_{p-1} ΔY_{t-p+1} + u_t

where Π = αβ' is the error correction matrix, β is the cointegrating vector,
and α is the speed-of-adjustment vector.

### Impulse Response Functions

The IRF measures the dynamic response of Y_{t+h} to a one-unit shock in u_{j,t}.
Cholesky decomposition of Σ identifies structural shocks via a lower-triangular
ordering (Cholesky ordering matters for interpretation).

### FEVD

FEVD at horizon h decomposes the forecast variance of variable i into contributions
from each structural shock j. FEVD_{ij}(h) ∈ [0, 1] and Σ_j FEVD_{ij}(h) = 1.

---

## Environment Setup

```bash
# Create environment
conda create -n tsecono python=3.11 -y
conda activate tsecono
pip install statsmodels>=0.14 arch>=6.0 numpy>=1.23 pandas>=1.5 matplotlib>=3.6

# Verify
python -c "import statsmodels; print('statsmodels', statsmodels.__version__)"
python -c "import arch; print('arch', arch.__version__)"
```

---

## Core Workflow

### Step 1 — Stationarity Tests and Data Preparation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
import warnings

np.random.seed(42)


def adf_summary(series: pd.Series, name: str = "", max_diff: int = 2) -> pd.DataFrame:
    """
    Run ADF test on a series and its differences until stationarity is achieved.

    Args:
        series:   Time series to test.
        name:     Label for display.
        max_diff: Maximum order of differencing to attempt.

    Returns:
        DataFrame with columns: variable, adf_stat, p_value, lags, conclusion.
    """
    records = []
    for d in range(max_diff + 1):
        s = series.diff(d).dropna() if d > 0 else series.dropna()
        label = f"Δ^{d} {name}" if d > 0 else name
        adf_result = adfuller(s, autolag="AIC")
        stat, pval, lags_used = adf_result[0], adf_result[1], adf_result[2]
        conclusion = "stationary (I(0))" if pval < 0.05 else "unit root (non-stationary)"
        records.append({
            "variable": label,
            "adf_stat": round(stat, 4),
            "p_value": round(pval, 4),
            "lags": lags_used,
            "conclusion": conclusion,
        })
        if pval < 0.05:
            break
    return pd.DataFrame(records)


def generate_var_data(
    n: int = 300,
    k: int = 3,
    p: int = 2,
    cointegrated: bool = False,
) -> pd.DataFrame:
    """
    Simulate a stationary VAR(p) or cointegrated I(1) system.

    Args:
        n:             Number of observations.
        k:             Number of variables.
        p:             Lag order of the true VAR.
        cointegrated:  If True, generate cointegrated I(1) system.

    Returns:
        DataFrame with k columns and n rows.
    """
    if not cointegrated:
        # Stationary VAR: companion matrix has spectral radius < 1
        A1 = np.array([[0.5, 0.1, 0.0],
                       [0.0, 0.4, 0.2],
                       [0.1, 0.0, 0.3]])
        A2 = np.array([[0.1, 0.0, 0.0],
                       [0.0, 0.1, 0.0],
                       [0.0, 0.0, 0.1]])
        Sigma = np.array([[1.0, 0.3, 0.1],
                          [0.3, 1.0, 0.2],
                          [0.1, 0.2, 1.0]])
        L = np.linalg.cholesky(Sigma)

        Y = np.zeros((n + 50, k))
        for t in range(2, n + 50):
            eps = L @ np.random.randn(k)
            Y[t] = A1 @ Y[t - 1] + A2 @ Y[t - 2] + eps
        Y = Y[50:]
        cols = [f"y{i+1}" for i in range(k)]
        return pd.DataFrame(Y, columns=cols)
    else:
        # I(1) cointegrated: three series share one common stochastic trend
        common_trend = np.cumsum(np.random.randn(n))
        Y = np.column_stack([
            common_trend + 0.5 * np.random.randn(n),
            2 * common_trend + np.random.randn(n),
            -0.5 * common_trend + 1.5 * np.random.randn(n),
        ])
        cols = ["x1", "x2", "x3"]
        return pd.DataFrame(Y, columns=cols)
```

### Step 2 — VAR Estimation and Granger Causality

```python
def fit_var_and_granger(
    df: pd.DataFrame,
    maxlags: int = 8,
    ic: str = "aic",
    verbose: bool = True,
) -> dict:
    """
    Fit a VAR(p) model, run stability check, and test pairwise Granger causality.

    Args:
        df:       DataFrame of stationary time series (k columns).
        maxlags:  Maximum lag order to consider.
        ic:       Information criterion for lag selection: 'aic', 'bic', or 'hqic'.
        verbose:  Print results to console.

    Returns:
        Dictionary with keys: model, results, lag_order, granger_table, stable.
    """
    model = VAR(df)

    # Lag selection
    lag_selection = model.select_order(maxlags=maxlags)
    if verbose:
        print(lag_selection.summary())

    # Fit at IC-optimal lag
    p_opt = getattr(lag_selection, ic)
    p_opt = max(1, p_opt)  # ensure at least 1 lag
    results = model.fit(p_opt)

    if verbose:
        print(results.summary())

    # Stability check
    eigenvalues = np.abs(results.roots)
    stable = bool(np.all(eigenvalues > 1))  # roots of char polynomial outside unit circle
    if verbose:
        print(f"\nVAR stability: {'STABLE' if stable else 'UNSTABLE'}")
        print(f"  Min |root| = {eigenvalues.min():.4f} (must be > 1)")

    # Granger causality: all pairwise tests
    variables = df.columns.tolist()
    granger_rows = []
    for caused in variables:
        for causing in variables:
            if caused == causing:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_test = results.test_causality(caused, causing, kind="f")
            granger_rows.append({
                "H0": f"{causing} does NOT Granger-cause {caused}",
                "F_stat": round(gc_test.test_statistic, 4),
                "p_value": round(gc_test.pvalue, 4),
                "reject_H0": gc_test.pvalue < 0.05,
            })

    granger_table = pd.DataFrame(granger_rows)
    if verbose:
        print("\nGranger Causality Tests:")
        print(granger_table.to_string(index=False))

    return {
        "model": model,
        "results": results,
        "lag_order": p_opt,
        "granger_table": granger_table,
        "stable": stable,
    }


def plot_irf_fevd(
    var_results,
    periods: int = 12,
    signif: float = 0.05,
    output_prefix: str = "var",
) -> None:
    """
    Plot impulse response functions and FEVD for a fitted VAR.

    Args:
        var_results: Fitted VAR results object (statsmodels VARResults).
        periods:     IRF horizon in periods.
        signif:      Significance level for bootstrap CI (default 0.05 = 95%).
        output_prefix: Prefix for saved figure filenames.
    """
    irf = var_results.irf(periods)
    k = var_results.neqs
    variables = var_results.names

    # IRF plot
    fig_irf = irf.plot(
        orth=True,  # Cholesky orthogonalization
        signif=signif,
        figsize=(12, 9),
    )
    fig_irf.suptitle(f"Orthogonalized IRFs (Cholesky ordering: {', '.join(variables)})")
    fig_irf.tight_layout()
    fig_irf.savefig(f"{output_prefix}_irf.png", dpi=150)
    print(f"Saved IRF plot to {output_prefix}_irf.png")

    # FEVD at horizons 1, 4, 8, 12
    fevd = var_results.fevd(periods)
    fig_fevd = fevd.plot(figsize=(12, 6))
    fig_fevd.suptitle("Forecast Error Variance Decomposition")
    fig_fevd.tight_layout()
    fig_fevd.savefig(f"{output_prefix}_fevd.png", dpi=150)
    print(f"Saved FEVD plot to {output_prefix}_fevd.png")

    # Print FEVD table at selected horizons
    print("\nFEVD at horizons 1 / 4 / 8 / 12:")
    for h in [1, 4, 8, 12]:
        if h <= periods:
            print(f"\n  Horizon {h}:")
            fevd_df = pd.DataFrame(fevd.decomp[h - 1],
                                   index=variables, columns=variables)
            print(fevd_df.round(3).to_string())
```

### Step 3 — Johansen Test and VECM Estimation

```python
def johansen_vecm_workflow(
    df_levels: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Run Johansen cointegration test and estimate VECM.

    Args:
        df_levels:  DataFrame of I(1) series in levels (not differenced).
        det_order:  Deterministic terms: -1 (none), 0 (constant), 1 (linear trend).
        k_ar_diff:  Lag order in the VECM (lags of differences).
        verbose:    Print results.

    Returns:
        Dictionary with keys: johansen_result, r_selected, vecm_results, beta, alpha.
    """
    # Step 1: Johansen test
    johansen = coint_johansen(df_levels.values, det_order, k_ar_diff)

    variables = df_levels.columns.tolist()
    k = len(variables)

    if verbose:
        print("=" * 60)
        print("Johansen Cointegration Test")
        print("=" * 60)
        trace_cv = johansen.cvt   # (k, 3): 90%, 95%, 99% critical values
        maxeig_cv = johansen.cvm

        print("\nTrace Test:")
        print(f"{'H0: r<=':>12}  {'Trace Stat':>12}  {'CV 5%':>10}  {'Reject?':>8}")
        for r in range(k):
            stat = johansen.lr1[r]
            cv5 = trace_cv[r, 1]
            reject = stat > cv5
            print(f"  r <= {r:>4}  {stat:>12.4f}  {cv5:>10.4f}  {'YES' if reject else 'NO':>8}")

        print("\nMax-Eigenvalue Test:")
        print(f"{'H0: r=':>12}  {'Max-Eig Stat':>12}  {'CV 5%':>10}  {'Reject?':>8}")
        for r in range(k):
            stat = johansen.lr2[r]
            cv5 = maxeig_cv[r, 1]
            reject = stat > cv5
            print(f"  r = {r:>5}  {stat:>12.4f}  {cv5:>10.4f}  {'YES' if reject else 'NO':>8}")

    # Select cointegrating rank r
    r_selected = 0
    for r in range(k):
        if johansen.lr1[r] > johansen.cvt[r, 1]:
            r_selected = r + 1

    if verbose:
        print(f"\nSelected cointegrating rank: r = {r_selected}")

    if r_selected == 0:
        print("No cointegration found — estimate VAR in differences.")
        return {"johansen_result": johansen, "r_selected": 0,
                "vecm_results": None, "beta": None, "alpha": None}

    # Step 2: VECM estimation
    vecm_model = VECM(df_levels, k_ar_diff=k_ar_diff, coint_rank=r_selected,
                      deterministic="co")  # constant in cointegrating equation
    vecm_result = vecm_model.fit()

    if verbose:
        print("\nVECM Results:")
        print(vecm_result.summary())

    beta = vecm_result.beta           # (k, r) cointegrating vectors
    alpha = vecm_result.alpha         # (k, r) speed of adjustment

    if verbose:
        print("\nCointegrating Vector(s) [normalized]:")
        beta_df = pd.DataFrame(beta, index=variables,
                               columns=[f"CE{i+1}" for i in range(r_selected)])
        print(beta_df.round(4))
        print("\nSpeed of Adjustment (α):")
        alpha_df = pd.DataFrame(alpha, index=variables,
                                columns=[f"CE{i+1}" for i in range(r_selected)])
        print(alpha_df.round(4))

    return {
        "johansen_result": johansen,
        "r_selected": r_selected,
        "vecm_results": vecm_result,
        "beta": beta,
        "alpha": alpha,
    }
```

---

## Advanced Usage

### Bootstrap Confidence Intervals for IRF

```python
def bootstrap_irf(
    var_results,
    periods: int = 12,
    n_boot: int = 500,
    signif: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Bootstrap percentile confidence intervals for Cholesky IRF.

    Uses residual bootstrap: resample VAR residuals with replacement,
    simulate new data, refit VAR, compute IRF.

    Args:
        var_results: Fitted VARResults object.
        periods:     IRF horizon.
        n_boot:      Number of bootstrap replications.
        signif:      Two-sided CI level (0.05 => 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        Dictionary with keys: irf_point, irf_lower, irf_upper.
            Each has shape (periods+1, k, k).
    """
    rng = np.random.default_rng(seed)
    k = var_results.neqs
    p = var_results.k_ar
    residuals = var_results.resid  # (T - p, k)
    T_eff = residuals.shape[0]

    # Centered residuals
    resid_centered = residuals - residuals.mean(axis=0)
    Y_orig = var_results.endog       # (T, k)
    coefs = var_results.coefs        # (p, k, k)
    intercept = var_results.intercept  # (k,)

    irf_boots = np.zeros((n_boot, periods + 1, k, k))

    for b in range(n_boot):
        # Resample residuals
        idx = rng.integers(0, T_eff, size=T_eff)
        boot_resid = resid_centered[idx]

        # Simulate new series
        Y_boot = np.zeros((p + T_eff, k))
        Y_boot[:p] = Y_orig[:p]
        for t in range(p, p + T_eff):
            Y_boot[t] = intercept.copy()
            for lag in range(p):
                Y_boot[t] += coefs[lag] @ Y_boot[t - lag - 1]
            Y_boot[t] += boot_resid[t - p]

        df_boot = pd.DataFrame(Y_boot[p:], columns=var_results.names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                var_boot = VAR(df_boot).fit(p)
                irf_boot = var_boot.irf(periods)
                irf_boots[b] = irf_boot.orth_irfs
            except Exception:
                irf_boots[b] = np.nan

    alpha_lo = signif / 2
    alpha_hi = 1 - signif / 2
    irf_lower = np.nanquantile(irf_boots, alpha_lo, axis=0)
    irf_upper = np.nanquantile(irf_boots, alpha_hi, axis=0)

    irf_point = var_results.irf(periods).orth_irfs

    return {"irf_point": irf_point, "irf_lower": irf_lower, "irf_upper": irf_upper}


def plot_bootstrap_irf(boot_dict: dict, variable_names: list, output_path: str = None) -> None:
    """
    Plot point-estimate IRF with bootstrap CI bands.

    Args:
        boot_dict:      Output of bootstrap_irf().
        variable_names: List of variable names.
        output_path:    If provided, save figure here.
    """
    k = len(variable_names)
    irf_point = boot_dict["irf_point"]
    irf_lower = boot_dict["irf_lower"]
    irf_upper = boot_dict["irf_upper"]
    H = irf_point.shape[0]
    horizons = np.arange(H)

    fig, axes = plt.subplots(k, k, figsize=(4 * k, 3 * k), sharex=True)
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            ax.plot(horizons, irf_point[:, i, j], color="#2980B9", linewidth=2)
            ax.fill_between(horizons, irf_lower[:, i, j], irf_upper[:, i, j],
                            alpha=0.2, color="#2980B9")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            if i == 0:
                ax.set_title(f"Shock: {variable_names[j]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Response: {variable_names[i]}", fontsize=9)
    fig.suptitle("Bootstrap IRF (95% CI)")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved bootstrap IRF to {output_path}")
    plt.show()
```

---

## Troubleshooting

| Error / Issue | Cause | Resolution |
|---|---|---|
| `MissingDataError` | NaN values in time series | Forward-fill or drop NaN rows before fitting |
| VAR eigenvalue >= 1 | Non-stationary system | First-difference data or use VECM |
| Johansen trace test selects r=k | Spurious cointegration | Check for structural breaks; verify I(1) order |
| VECM `alpha` near zero | No error correction | Re-examine cointegrating rank selection |
| Granger test F-stat = 0 | Lag order = 1 but variable not in equation | Increase maxlags; verify variable is in system |
| IRF does not decay to zero | VAR is near-non-stationary | Re-test lag selection; check for unit roots |
| `LinAlgError` in Cholesky | Σ not positive definite due to collinearity | Drop redundant variables or add jitter |
| Very wide bootstrap CI bands | Short sample or many variables | Reduce variables; increase sample size |

---

## External Resources

- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Sims, C.A. (1980). "Macroeconomics and Reality." *Econometrica*, 48(1), 1–48.
- Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." *JEDS*, 12(2–3), 231–254.
- `statsmodels` VAR documentation: <https://www.statsmodels.org/stable/vector_ar.html>
- `statsmodels` VECM documentation: <https://www.statsmodels.org/stable/vector_ar.html#vector-error-correction-models-vecm>

---

## Examples

### Example 1 — Stationary VAR: Full Pipeline

```python
# Generate stationary data
df_stat = generate_var_data(n=400, k=3, cointegrated=False)

# Check stationarity
for col in df_stat.columns:
    print(adf_summary(df_stat[col], name=col).to_string(index=False))

# Fit VAR and test Granger causality
var_out = fit_var_and_granger(df_stat, maxlags=8, ic="aic")

# IRF and FEVD plots
plot_irf_fevd(var_out["results"], periods=12, output_prefix="var_example")
```

### Example 2 — Cointegrated System: Johansen + VECM

```python
# Generate cointegrated I(1) data
df_coint = generate_var_data(n=400, k=3, cointegrated=True)

# Confirm unit roots in levels, stationarity in differences
for col in df_coint.columns:
    print(adf_summary(df_coint[col], name=col).to_string(index=False))

# Johansen test and VECM
vecm_out = johansen_vecm_workflow(df_coint, det_order=0, k_ar_diff=2)

# Confirm error-correction spreads are stationary
if vecm_out["beta"] is not None:
    beta = vecm_out["beta"]
    spread = df_coint.values @ beta[:, 0]
    adf_spread = adfuller(spread, autolag="AIC")
    print(f"\nSpread ADF p-value: {adf_spread[1]:.4f} (should be < 0.05)")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — VAR, Granger, Johansen, VECM, IRF, FEVD, bootstrap CI |
