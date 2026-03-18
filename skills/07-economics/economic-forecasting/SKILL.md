---
name: economic-forecasting
description: >
  Use this Skill for macroeconomic forecasting: ARIMA, VAR, ML ensemble (LightGBM),
  Diebold-Mariano test, fan chart generation, and real-time ALFRED data.
tags:
  - economics
  - forecasting
  - ARIMA
  - LightGBM
  - Diebold-Mariano
  - macroeconomics
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
    - lightgbm>=4.0
    - pandas>=1.5
    - numpy>=1.23
    - scipy>=1.9
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Macroeconomic Forecasting

> **TL;DR** — Build macroeconomic forecasts using ARIMA, VAR, and LightGBM with
> walk-forward time-series cross-validation. Compare models with the Diebold-Mariano
> test, visualize uncertainty with fan charts, and retrieve real-time ALFRED vintages
> from the FRED API.

---

## When to Use

| Situation | Recommended Approach |
|---|---|
| Single univariate series (GDP, CPI, unemployment) | ARIMA(p,d,q) |
| Multi-variable macroeconomic system | VAR multi-step forecast |
| High-frequency data with many predictors | LightGBM with lag features |
| Combining model outputs | Equal-weight or BMA combination |
| Testing if Model A beats Model B | Diebold-Mariano test |
| Real-time data vintages (avoid look-ahead bias) | ALFRED API |
| Visualizing forecast uncertainty | Bootstrap fan chart |

---

## Background

### ARIMA(p,d,q) Identification

Identify p, d, q via:
1. Unit root test (ADF/KPSS) to determine differencing order d.
2. ACF/PACF of differenced series for p (AR order) and q (MA order).
3. Grid search over (p, d, q) minimizing AIC or BIC.

Seasonal ARIMA: SARIMA(p,d,q)(P,D,Q,s) handles periodic seasonality (s=4 quarterly,
s=12 monthly).

### ML Forecasting with Time-Series Features

LightGBM/XGBoost forecasting of Y_{t+h} using:
- Lags: Y_{t-1}, Y_{t-2}, ..., Y_{t-L}
- Rolling statistics: mean, std, min, max over past 4/8/12 periods
- Calendar features: month, quarter, year (for seasonality)
- External predictors: other macro variables

**Walk-forward validation** — strictly avoids look-ahead bias:
- Train on [1, t]; predict t+h; expand window to [1, t+1]; repeat.

### Diebold-Mariano Test (1995)

Tests whether two forecasting models have equal predictive accuracy.
Let d_t = L(ê_{1t}) - L(ê_{2t}) be the loss differential.

    DM = d̄ / √(2π f̂_d(0) / T)  ~ N(0, 1) under H0: E[d_t] = 0

where f̂_d(0) is the spectral density of d_t at frequency zero (estimated with
Newey-West HAC variance). A negative DM statistic means Model 2 has lower loss.

Harvey-Leybourne-Newbold (1997) correction adjusts for small samples.

### Fan Chart

A fan chart plots the central forecast with probabilistic bands at different
confidence levels (e.g., 30%, 60%, 90% CI). Obtained by:
1. Bootstrap the forecast errors from historical walk-forward evaluation.
2. Add empirical quantiles of bootstrapped errors to the point forecast.

### Mincer-Zarnowitz (1969) Regression

Tests forecast unbiasedness:
    Y_{t+h} = α + β Ŷ_{t+h|t} + ε_t

Under H0 (unbiased, efficient): α = 0, β = 1 (joint Wald test).

---

## Environment Setup

```bash
conda create -n forecast python=3.11 -y
conda activate forecast
pip install statsmodels>=0.14 lightgbm>=4.0 pandas>=1.5 numpy>=1.23 scipy>=1.9 matplotlib>=3.6

# For FRED/ALFRED data access
pip install fredapi

# Set your FRED API key (never hardcode)
export FRED_API_KEY="<paste-your-key>"

python -c "import lightgbm as lgb; print('LightGBM', lgb.__version__)"
```

Register for a free FRED API key at: <https://fred.stlouisfed.org/docs/api/api_key.html>

---

## Core Workflow

### Step 1 — ARIMA vs LightGBM Forecast Comparison

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings

np.random.seed(42)


def simulate_arma_series(n: int = 200, ar: tuple = (0.6, -0.2), ma: tuple = (0.3,)) -> pd.Series:
    """
    Simulate a stationary ARMA series for forecasting experiments.

    Args:
        n:   Number of observations.
        ar:  AR coefficients (positive for phi_1, phi_2, ...).
        ma:  MA coefficients.

    Returns:
        pd.Series with DatetimeIndex at quarterly frequency.
    """
    from statsmodels.tsa.arima_process import ArmaProcess
    ar_poly = np.r_[1, -np.array(ar)]
    ma_poly = np.r_[1, np.array(ma)]
    arma = ArmaProcess(ar_poly, ma_poly)
    data = arma.generate_sample(nsample=n, scale=1.0)
    idx = pd.date_range(start="2000Q1", periods=n, freq="QS")
    return pd.Series(data, index=idx, name="y")


def make_lag_features(
    series: pd.Series,
    n_lags: int = 8,
    rolling_windows: list = None,
) -> pd.DataFrame:
    """
    Construct lag and rolling features for ML forecasting.

    Args:
        series:          Time series to featurize.
        n_lags:          Number of lag features.
        rolling_windows: Rolling statistic window sizes.

    Returns:
        DataFrame with lag and rolling features (NaN rows dropped).
    """
    if rolling_windows is None:
        rolling_windows = [4, 8]

    df = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)

    for w in rolling_windows:
        df[f"roll_mean_{w}"] = series.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = series.shift(1).rolling(w).std()

    df["month"] = series.index.month
    df["quarter"] = series.index.quarter
    df.dropna(inplace=True)
    return df


def walk_forward_forecast(
    series: pd.Series,
    h: int = 1,
    train_min: int = 40,
    method: str = "arima",
    arima_order: tuple = (2, 0, 1),
    n_lags: int = 8,
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) forecast evaluation.

    For each time step t from train_min to T-h:
      - Train on [0, t]
      - Forecast t+h
      - Record actual vs forecast

    Args:
        series:       Time series to forecast.
        h:            Forecast horizon (steps ahead).
        train_min:    Minimum training sample size.
        method:       'arima' or 'lightgbm'.
        arima_order:  (p, d, q) for ARIMA.
        n_lags:       Number of lag features for LightGBM.

    Returns:
        DataFrame with columns: period, actual, forecast, error.
    """
    T = len(series)
    records = []

    if method == "lightgbm":
        feat_df = make_lag_features(series, n_lags=n_lags)
        feat_cols = [c for c in feat_df.columns if c != "y"]

    for t in range(train_min, T - h):
        train_series = series.iloc[:t + 1]
        actual = series.iloc[t + h]

        if method == "arima":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = ARIMA(train_series, order=arima_order)
                    fitted = model.fit()
                    forecast = float(fitted.forecast(steps=h).iloc[-1])
                except Exception:
                    forecast = float(train_series.mean())

        elif method == "lightgbm":
            # Only use rows where all features are available and index <= t
            train_feat = feat_df.loc[feat_df.index <= train_series.index[-1]]
            if len(train_feat) < 10:
                forecast = float(train_series.mean())
            else:
                X_tr = train_feat[feat_cols].values
                y_tr = train_feat["y"].values
                lgb_train = lgb.Dataset(X_tr, label=y_tr)
                params = {"objective": "regression", "verbosity": -1,
                          "num_leaves": 15, "learning_rate": 0.05,
                          "n_estimators": 100}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bst = lgb.train(params, lgb_train, num_boost_round=100,
                                    callbacks=[lgb.log_evaluation(period=-1)])
                # Use last available row as features for h-step forecast
                last_feat = feat_df.loc[feat_df.index <= train_series.index[-1]].iloc[-1:][feat_cols]
                forecast = float(bst.predict(last_feat))
        else:
            raise ValueError(f"Unknown method: {method}")

        records.append({
            "period": series.index[t + h],
            "actual": float(actual),
            "forecast": forecast,
            "error": float(actual) - forecast,
        })

    return pd.DataFrame(records)


def compare_forecasts(
    series: pd.Series,
    h: int = 1,
    train_min: int = 40,
    output_path: str = None,
) -> dict:
    """
    Run walk-forward evaluation for ARIMA and LightGBM, plot and compare.

    Args:
        series:       Univariate time series.
        h:            Forecast horizon.
        train_min:    Minimum training sample.
        output_path:  If provided, save comparison plot.

    Returns:
        Dictionary with RMSE for each model and forecast DataFrames.
    """
    print("Running ARIMA walk-forward evaluation...")
    arima_df = walk_forward_forecast(series, h=h, train_min=train_min, method="arima")

    print("Running LightGBM walk-forward evaluation...")
    lgbm_df = walk_forward_forecast(series, h=h, train_min=train_min, method="lightgbm")

    rmse_arima = float(np.sqrt(np.mean(arima_df["error"] ** 2)))
    rmse_lgbm = float(np.sqrt(np.mean(lgbm_df["error"] ** 2)))

    print(f"\nRMSE (h={h}): ARIMA = {rmse_arima:.4f}  |  LightGBM = {rmse_lgbm:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, df, label, color in zip(axes, [arima_df, lgbm_df],
                                     ["ARIMA", "LightGBM"], ["#2980B9", "#E74C3C"]):
        ax.plot(df["period"], df["actual"], color="black", linewidth=1.5, label="Actual")
        ax.plot(df["period"], df["forecast"], color=color, linewidth=1.5,
                linestyle="--", label=f"{label} forecast")
        ax.set_ylabel("Value")
        ax.set_title(f"{label} Walk-Forward Forecast (RMSE = {np.sqrt(np.mean(df['error']**2)):.4f})")
        ax.legend()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved comparison plot to {output_path}")
    plt.show()

    return {"rmse_arima": rmse_arima, "rmse_lgbm": rmse_lgbm,
            "arima_df": arima_df, "lgbm_df": lgbm_df}
```

### Step 2 — Diebold-Mariano Test

```python
from scipy.stats import t as t_dist


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    loss: str = "mse",
    small_sample_correction: bool = True,
) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: E[d_t] = 0  (models have equal expected loss)
    H1: E[d_t] != 0  (one model has strictly lower expected loss)

    Negative DM statistic => Model 2 (errors2) has lower loss.

    Args:
        errors1:                   Forecast errors from Model 1, shape (T,).
        errors2:                   Forecast errors from Model 2, shape (T,).
        h:                         Forecast horizon (used for HAC lag truncation).
        loss:                      Loss function: 'mse' or 'mae'.
        small_sample_correction:   Apply Harvey-Leybourne-Newbold correction.

    Returns:
        Dictionary with keys: dm_stat, p_value, reject_h0, conclusion.
    """
    if loss == "mse":
        L1 = errors1 ** 2
        L2 = errors2 ** 2
    elif loss == "mae":
        L1 = np.abs(errors1)
        L2 = np.abs(errors2)
    else:
        raise ValueError(f"Unknown loss: {loss}. Choose 'mse' or 'mae'.")

    d = L1 - L2
    T = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance with truncation lag = h - 1
    gamma0 = np.mean((d - d_bar) ** 2)
    gamma_sum = gamma0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        gamma_sum += 2 * (1 - k / h) * gamma_k

    var_d_bar = gamma_sum / T
    dm_stat = d_bar / np.sqrt(var_d_bar + 1e-12)

    # HLN small-sample correction
    if small_sample_correction and h > 1:
        hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dm_stat_hln = dm_stat * hln_factor
        p_value = 2 * (1 - t_dist.cdf(abs(dm_stat_hln), df=T - 1))
        stat_used = dm_stat_hln
    else:
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
        stat_used = dm_stat

    reject = p_value < 0.05
    if d_bar < 0:
        conclusion = "Model 2 significantly better" if reject else "No significant difference"
    else:
        conclusion = "Model 1 significantly better" if reject else "No significant difference"

    print(f"\nDiebold-Mariano Test ({loss} loss, h={h}):")
    print(f"  DM statistic: {stat_used:.4f}")
    print(f"  p-value:      {p_value:.4f}")
    print(f"  Conclusion:   {conclusion}")

    return {"dm_stat": stat_used, "p_value": p_value,
            "reject_h0": reject, "conclusion": conclusion}
```

### Step 3 — Fan Chart with Bootstrap

```python
def fan_chart(
    point_forecast: np.ndarray,
    historical_errors: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    history_series: pd.Series = None,
    levels: list = None,
    n_boot: int = 2000,
    output_path: str = None,
) -> dict:
    """
    Generate a fan chart with bootstrap confidence bands.

    Args:
        point_forecast:    Central forecast values, shape (H,).
        historical_errors: In-sample or walk-forward errors for bootstrapping, shape (T,).
        forecast_dates:    DatetimeIndex for the forecast horizon.
        history_series:    Historical series to plot before the forecast.
        levels:            CI levels to display. Default: [0.30, 0.60, 0.90].
        n_boot:            Number of bootstrap draws.
        output_path:       If provided, save fan chart.

    Returns:
        Dictionary with CI bands: keys are CI levels, values are (lower, upper) arrays.
    """
    if levels is None:
        levels = [0.30, 0.60, 0.90]

    H = len(point_forecast)
    rng = np.random.default_rng(0)

    # Bootstrap: draw error sequences of length H
    T_err = len(historical_errors)
    boot_errors = rng.choice(historical_errors, size=(n_boot, H), replace=True)
    # Add cumulative errors for multi-step (random walk of errors)
    boot_paths = point_forecast[np.newaxis, :] + boot_errors

    ci_bands = {}
    for level in levels:
        alpha = (1 - level) / 2
        lo = np.quantile(boot_paths, alpha, axis=0)
        hi = np.quantile(boot_paths, 1 - alpha, axis=0)
        ci_bands[level] = (lo, hi)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    if history_series is not None:
        ax.plot(history_series.index[-40:], history_series.values[-40:],
                color="black", linewidth=2, label="Historical")

    colors_fill = ["#AED6F1", "#5DADE2", "#1B4F72"]
    for (level, (lo, hi)), color in zip(
        sorted(ci_bands.items(), reverse=True), colors_fill
    ):
        ax.fill_between(forecast_dates, lo, hi, alpha=0.4, color=color,
                        label=f"{int(level*100)}% CI")

    ax.plot(forecast_dates, point_forecast, color="#C0392B", linewidth=2,
            linestyle="--", label="Point forecast")

    if history_series is not None:
        ax.axvline(forecast_dates[0], color="gray", linestyle=":", linewidth=1.2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Fan Chart — Forecast Uncertainty")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved fan chart to {output_path}")
    plt.show()

    return ci_bands
```

---

## Advanced Usage

### FRED/ALFRED Real-Time Vintage Access

```python
import os


def fetch_fred_series(
    series_id: str,
    start: str = "2000-01-01",
    end: str = None,
) -> pd.Series:
    """
    Fetch a time series from FRED using the fredapi package.

    Requires FRED_API_KEY environment variable.
    Register at: https://fred.stlouisfed.org/docs/api/api_key.html

    Args:
        series_id: FRED series identifier (e.g., 'GDP', 'CPIAUCSL', 'UNRATE').
        start:     Start date as 'YYYY-MM-DD'.
        end:       End date as 'YYYY-MM-DD'. Defaults to today.

    Returns:
        pd.Series with DatetimeIndex.
    """
    try:
        import fredapi
    except ImportError:
        raise ImportError("Install fredapi: pip install fredapi")

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Run: export FRED_API_KEY='<paste-your-key>'"
        )

    fred = fredapi.Fred(api_key=api_key)
    series = fred.get_series(series_id, observation_start=start, observation_end=end)
    series.name = series_id
    print(f"Fetched FRED series '{series_id}': {len(series)} observations "
          f"from {series.index[0].date()} to {series.index[-1].date()}")
    return series


def mincer_zarnowitz_test(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> dict:
    """
    Mincer-Zarnowitz (1969) regression test for forecast unbiasedness.

    Regresses actual on forecast: Y = alpha + beta * Yhat + eps
    Joint test H0: alpha=0, beta=1.

    Args:
        actual:   Realized values, shape (T,).
        forecast: Forecast values, shape (T,).

    Returns:
        Dictionary with alpha, beta, joint_pvalue, unbiased.
    """
    import statsmodels.api as sm
    from scipy.stats import f as f_dist

    X = sm.add_constant(forecast)
    result = sm.OLS(actual, X).fit(cov_type="HC3")
    alpha_hat = float(result.params["const"])
    beta_hat = float(result.params[forecast.name if hasattr(forecast, 'name') else "x1"])

    # Joint Wald test: alpha=0, beta=1
    R = np.array([[1, 0], [0, 1]])
    r = np.array([0, 1])
    wald = result.wald_test(np.column_stack([R, -r.reshape(-1, 1)[:, :0]]),
                            use_f=True)
    # Simpler: F-test using statsmodels contrast
    from statsmodels.stats.contrast import ContrastResults
    hypotheses = "const = 0, x1 = 1"
    try:
        joint_test = result.f_test("const = 0, x1 = 1")
        joint_pval = float(joint_test.pvalue)
    except Exception:
        joint_pval = np.nan

    unbiased = joint_pval > 0.05 if not np.isnan(joint_pval) else None
    print(f"\nMincer-Zarnowitz Test:")
    print(f"  α̂ = {alpha_hat:.4f}  β̂ = {beta_hat:.4f}")
    print(f"  Joint p-value (H0: α=0, β=1): {joint_pval:.4f}")
    print(f"  Forecast {'unbiased' if unbiased else 'biased'} at 5%")

    return {"alpha": alpha_hat, "beta": beta_hat,
            "joint_pvalue": joint_pval, "unbiased": unbiased}
```

---

## Troubleshooting

| Error / Issue | Cause | Resolution |
|---|---|---|
| ARIMA `ConvergenceWarning` | Optimizer not converging | Increase `maxiter`; try different solver |
| LightGBM overfits short series | Too many leaves / high learning rate | Reduce `num_leaves`; increase `min_data_in_leaf` |
| DM test `ZeroDivisionError` | Identical forecasts | Check if models produce same predictions |
| DM rejects H0 but forecasts look similar | Small loss differences accumulate | Use MAE loss; check for outlier periods |
| Fan chart CI bands too wide | High forecast error variance | Use more training data; smooth historical errors |
| `FRED_API_KEY` not found | Environment variable not set | Run `export FRED_API_KEY="<paste-your-key>"` in shell |
| Walk-forward loop very slow | Many models, long series | Vectorize ARIMA with `pmdarima.auto_arima`; reduce `train_min` |

---

## External Resources

- Diebold, F.X., Mariano, R.S. (1995). "Comparing Predictive Accuracy." *JBES*, 13(3), 253–263.
- Harvey, D., Leybourne, S., Newbold, P. (1997). "Testing the Equality of Prediction Mean Squared
  Errors." *International Journal of Forecasting*, 13(2), 281–291.
- Mincer, J., Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts." *Economic Forecasts and
  Expectations*, NBER.
- FRED API documentation: <https://fred.stlouisfed.org/docs/api/fred/>
- `statsmodels` ARIMA: <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html>
- `lightgbm` documentation: <https://lightgbm.readthedocs.io/>

---

## Examples

### Example 1 — ARIMA vs LightGBM Walk-Forward with DM Test

```python
series = simulate_arma_series(n=200, ar=(0.6, -0.2), ma=(0.3,))

results = compare_forecasts(series, h=1, train_min=50, output_path="forecast_comparison.png")

# Diebold-Mariano test
arima_errors = results["arima_df"]["error"].values
lgbm_errors = results["lgbm_df"]["error"].values
dm_result = diebold_mariano_test(arima_errors, lgbm_errors, h=1, loss="mse")
```

### Example 2 — Fan Chart with Bootstrap Uncertainty Bands

```python
# Use ARIMA walk-forward errors as historical error distribution
arima_df = results["arima_df"]
historical_errs = arima_df["error"].values

# One-step-ahead point forecast using all available data
from statsmodels.tsa.arima.model import ARIMA
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    final_model = ARIMA(series, order=(2, 0, 1)).fit()
    h_forecast = 8
    fc = final_model.forecast(steps=h_forecast).values

fc_dates = pd.date_range(start=series.index[-1], periods=h_forecast + 1, freq="QS")[1:]
ci_bands = fan_chart(
    point_forecast=fc,
    historical_errors=historical_errs,
    forecast_dates=fc_dates,
    history_series=series,
    levels=[0.30, 0.60, 0.90],
    output_path="fan_chart.png",
)
print("Fan chart saved. 90% CI width at h=4:", round(float(ci_bands[0.90][1][3] - ci_bands[0.90][0][3]), 4))
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — ARIMA, LightGBM, DM test, fan chart, ALFRED/FRED access |
