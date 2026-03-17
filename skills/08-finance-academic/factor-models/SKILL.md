---
name: factor-models
description: >
  Asset pricing factor models: Fama-French 3/5-factor, Carhart 4-factor,
  q-factor; OLS regressions, rolling loadings, and GRS test for alpha.
tags:
  - factor-models
  - fama-french
  - asset-pricing
  - finance
  - statistics
  - pandas-datareader
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
  - numpy>=1.24.0
  - pandas>=2.0.0
  - pandas-datareader>=0.10.0
  - statsmodels>=0.14.0
  - scipy>=1.11.0
  - matplotlib>=3.7.0
  - yfinance>=0.2.0
last_updated: "2026-03-17"
---

# Factor Models

A comprehensive skill for estimating and interpreting asset pricing factor models
in academic and professional research. Covers downloading Fama-French factor data
from the Ken French Data Library, computing excess returns, OLS time-series
regressions, the Carhart (1997) 4-factor momentum model, the q-factor model,
rolling factor loadings, and the Gibbons-Ross-Shanken (1989) test for pricing
errors.

---

## Core Functions

### 1. Downloading Fama-French Factor Data

```python
import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Fama-French dataset names ─────────────────────────────────────────────────
_FF_DATASETS = {
    "3factor": "F-F_Research_Data_Factors",
    "5factor": "F-F_Research_Data_5_Factors_2x3",
    "momentum": "F-F_Momentum_Factor",
    "qfactor": "q5_factors_daily_2022",   # from AQR or q-factor website
}


def download_ff_factors(
    model: str = "5factor",
    start: str = "2000-01-01",
    end: str = "2024-12-31",
    freq: str = "monthly",
) -> pd.DataFrame:
    """
    Download Fama-French factor returns from the Ken French Data Library.

    Parameters
    ----------
    model : str
        One of '3factor', '5factor', 'momentum'. '4factor' downloads 3factor
        + momentum and merges them automatically.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    freq : str
        'monthly' or 'daily'. Ken French provides both.

    Returns
    -------
    pd.DataFrame
        DataFrame with factor columns in percent terms (divide by 100 for decimals).
        Always includes 'RF' (risk-free rate) and 'Mkt-RF' (market excess return).
    """
    if model == "4factor":
        ff3 = download_ff_factors("3factor", start, end, freq)
        mom = download_ff_factors("momentum", start, end, freq)
        combined = ff3.join(mom, how="inner")
        return combined

    dataset_key = _FF_DATASETS.get(model)
    if dataset_key is None:
        raise ValueError(f"Unsupported model: {model!r}. Choose from {list(_FF_DATASETS)}")

    # pandas_datareader fetches directly from Ken French's website
    factors_raw = web.DataReader(
        dataset_key,
        "famafrench",
        start=start,
        end=end,
    )
    # DataReader returns a dict; index 0 is monthly, index 1 is annual
    if freq == "monthly":
        df = factors_raw[0]
    elif freq == "daily":
        df = factors_raw[0]  # daily datasets only have one table
    else:
        raise ValueError(f"freq must be 'monthly' or 'daily', got {freq!r}")

    df.index = pd.to_datetime(df.index.astype(str), format="%Y-%m")
    df.index.name = "date"
    return df / 100.0  # convert from percent to decimal


def compute_excess_returns(
    prices: pd.DataFrame,
    rf: pd.Series,
    freq: str = "monthly",
) -> pd.DataFrame:
    """
    Compute simple excess returns for a portfolio or individual assets.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices. Index = DatetimeIndex, columns = ticker symbols.
    rf : pd.Series
        Risk-free rate series aligned to the same period frequency (decimal form).
    freq : str
        'monthly' or 'daily'. Used to determine the resample frequency.

    Returns
    -------
    pd.DataFrame
        Excess return columns: R_i - RF for each asset.
    """
    if freq == "monthly":
        # Resample to month-end and compute simple returns
        monthly_prices = prices.resample("ME").last()
        returns = monthly_prices.pct_change().dropna()
    elif freq == "daily":
        returns = prices.pct_change().dropna()
    else:
        raise ValueError(f"freq must be 'monthly' or 'daily', got {freq!r}")

    # Align rf index to returns index
    rf_aligned = rf.reindex(returns.index).ffill()
    excess = returns.subtract(rf_aligned, axis=0)
    return excess.dropna()
```

### 2. OLS Factor Regression

```python
def run_factor_regression(
    excess_ret: pd.Series,
    factors: pd.DataFrame,
    add_intercept: bool = True,
) -> dict:
    """
    Run an OLS time-series factor regression for a single asset or portfolio.

    Parameters
    ----------
    excess_ret : pd.Series
        Excess returns of the test asset (decimal).
    factors : pd.DataFrame
        Factor return data aligned to the same dates (decimal). Should NOT
        include the risk-free rate column.
    add_intercept : bool
        Whether to add a constant (alpha) to the regression.

    Returns
    -------
    dict with keys:
        alpha        : float  — annualised Jensen's alpha
        alpha_tstat  : float  — t-statistic for alpha
        alpha_pval   : float  — p-value for alpha
        betas        : dict   — factor loadings keyed by factor name
        r_squared    : float  — R² of the regression
        adj_r2       : float  — adjusted R²
        residuals    : pd.Series
        model        : RegressionResultsWrapper (full statsmodels result)
    """
    # Align on common dates
    aligned = pd.concat([excess_ret, factors], axis=1, join="inner").dropna()
    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    if add_intercept:
        X = sm.add_constant(X)

    model = sm.OLS(y, X).fit(cov_type="HC3")

    alpha_monthly = model.params.get("const", 0.0)
    alpha_annual = alpha_monthly * 12  # annualise for monthly data

    factor_names = [c for c in model.params.index if c != "const"]
    betas = {name: model.params[name] for name in factor_names}

    return {
        "alpha": alpha_annual,
        "alpha_monthly": alpha_monthly,
        "alpha_tstat": model.tvalues.get("const", np.nan),
        "alpha_pval": model.pvalues.get("const", np.nan),
        "betas": betas,
        "r_squared": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "residuals": model.resid,
        "model": model,
    }


def rolling_factor_loadings(
    excess_ret: pd.Series,
    factors: pd.DataFrame,
    window: int = 36,
) -> pd.DataFrame:
    """
    Estimate rolling OLS factor loadings using a fixed rolling window.

    Parameters
    ----------
    excess_ret : pd.Series
        Excess returns (decimal, monthly).
    factors : pd.DataFrame
        Factor returns aligned to same index.
    window : int
        Rolling window size in periods (default 36 months = 3 years).

    Returns
    -------
    pd.DataFrame
        Rolling coefficients. Columns: 'alpha', plus one per factor.
        First (window - 1) rows are NaN.
    """
    aligned = pd.concat([excess_ret, factors], axis=1, join="inner").dropna()
    y = aligned.iloc[:, 0]
    factor_cols = aligned.iloc[:, 1:]

    result_records = []
    for end in range(window, len(aligned) + 1):
        y_w = y.iloc[end - window : end]
        X_w = sm.add_constant(factor_cols.iloc[end - window : end])
        try:
            res = sm.OLS(y_w, X_w).fit()
            record = {"date": y.index[end - 1]}
            record["alpha"] = res.params.get("const", np.nan) * 12
            for col in factor_cols.columns:
                record[col] = res.params.get(col, np.nan)
        except Exception:
            record = {"date": y.index[end - 1], "alpha": np.nan}
            for col in factor_cols.columns:
                record[col] = np.nan
        result_records.append(record)

    df = pd.DataFrame(result_records).set_index("date")
    return df
```

### 3. GRS Test for Pricing Errors

```python
def grs_test(
    excess_rets: pd.DataFrame,
    factors: pd.DataFrame,
) -> dict:
    """
    Gibbons, Ross, and Shanken (1989) test for joint pricing errors (alphas = 0).

    The GRS statistic tests the null hypothesis H0: alpha_i = 0 for all i
    simultaneously under the assumption of multivariate normality.

    Parameters
    ----------
    excess_rets : pd.DataFrame
        Excess returns of N test assets (T × N).
    factors : pd.DataFrame
        Factor returns (T × K). Must be aligned to same dates as excess_rets.

    Returns
    -------
    dict with keys:
        grs_stat   : float  — GRS F-statistic
        p_value    : float  — p-value under F(N, T-N-K) distribution
        alphas     : np.ndarray  — estimated alphas (N,)
        t_stats    : np.ndarray  — individual alpha t-statistics
    """
    from scipy.stats import f as f_dist

    aligned = pd.concat([excess_rets, factors], axis=1, join="inner").dropna()
    N = excess_rets.shape[1]
    K = factors.shape[1]
    T = len(aligned)

    alpha_hat = np.zeros(N)
    residuals = np.zeros((T, N))
    t_stats = np.zeros(N)

    factor_X = sm.add_constant(aligned[factors.columns])

    for j, col in enumerate(excess_rets.columns):
        y = aligned[col]
        res = sm.OLS(y, factor_X).fit()
        alpha_hat[j] = res.params["const"]
        residuals[:, j] = res.resid
        t_stats[j] = res.tvalues["const"]

    # Residual covariance matrix
    Sigma_hat = (residuals.T @ residuals) / (T - K - 1)

    # Mean factor returns and factor covariance
    mu_f = aligned[factors.columns].mean().values  # (K,)
    Omega_f = aligned[factors.columns].cov().values  # (K, K)

    # Sharpe ratio adjustment term
    sh_squared = float(mu_f @ linalg.inv(Omega_f) @ mu_f)

    # GRS F-statistic
    grs_stat = (
        (T / N)
        * ((T - N - K) / (T - K - 1))
        * (alpha_hat @ linalg.inv(Sigma_hat) @ alpha_hat)
        / (1 + sh_squared)
    )

    df1, df2 = N, T - N - K
    p_value = 1.0 - f_dist.cdf(grs_stat, df1, df2)

    return {
        "grs_stat": grs_stat,
        "p_value": p_value,
        "alphas": alpha_hat,
        "t_stats": t_stats,
        "df1": df1,
        "df2": df2,
    }


def factor_exposure_attribution(
    excess_ret: pd.Series,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose a portfolio's expected return into factor contributions.

    Returns
    -------
    pd.DataFrame
        Columns: 'factor_mean_return', 'loading', 'contribution', 'pct_explained'.
    """
    result = run_factor_regression(excess_ret, factors)
    factor_means = factors.mean() * 12  # annualise

    rows = []
    for factor_name, beta in result["betas"].items():
        mean_ret = factor_means.get(factor_name, np.nan)
        contribution = beta * mean_ret
        rows.append({
            "factor": factor_name,
            "factor_mean_return": mean_ret,
            "loading": beta,
            "contribution": contribution,
        })

    df = pd.DataFrame(rows).set_index("factor")
    total_explained = df["contribution"].sum()
    df["pct_explained"] = df["contribution"] / total_explained * 100
    return df
```

---

## Example 1: Portfolio Alpha vs Fama-French 5-Factor Model

Estimate the risk-adjusted alpha (Jensen's alpha) of a custom equal-weight
portfolio against the Fama-French 5-factor model.

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Download factor data ──────────────────────────────────────────────────────
START = "2015-01-01"
END   = "2024-12-31"

ff5 = download_ff_factors(model="5factor", start=START, end=END, freq="monthly")
print("FF5 factors loaded:", ff5.columns.tolist())
print(ff5.tail(3))

# ── Download stock price data for a hypothetical portfolio ────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
raw_prices = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)["Close"]

# ── Compute portfolio excess returns ─────────────────────────────────────────
rf = ff5["RF"]
# Equal-weight portfolio returns
asset_excess = compute_excess_returns(raw_prices, rf, freq="monthly")
portfolio_excess = asset_excess.mean(axis=1)
portfolio_excess.name = "EW_Portfolio"

# ── Extract Fama-French 5 factors (exclude RF) ────────────────────────────────
factor_cols = [c for c in ff5.columns if c != "RF"]
factors_df = ff5[factor_cols]

# ── Run 5-factor regression ───────────────────────────────────────────────────
result = run_factor_regression(portfolio_excess, factors_df)

print("\n=== Fama-French 5-Factor Regression ===")
print(f"Alpha (annualised):   {result['alpha']:.4f}  ({result['alpha']*100:.2f}%)")
print(f"Alpha t-stat:         {result['alpha_tstat']:.3f}")
print(f"Alpha p-value:        {result['alpha_pval']:.4f}")
print(f"R-squared:            {result['r_squared']:.4f}")
print("\nFactor loadings:")
for factor, beta in result["betas"].items():
    print(f"  {factor:12s}: {beta:.4f}")

# ── Factor attribution ────────────────────────────────────────────────────────
attr = factor_exposure_attribution(portfolio_excess, factors_df)
print("\nReturn Attribution:")
print(attr.round(4))

# ── Plot rolling 3-year alpha ─────────────────────────────────────────────────
rolling = rolling_factor_loadings(portfolio_excess, factors_df, window=36)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(rolling.index, rolling["alpha"] * 100, label="Rolling Alpha (annualised %)")
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("Alpha (%)")
axes[0].set_title("Rolling 36-Month Alpha — EW Tech Portfolio vs FF5")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(rolling.index, rolling["Mkt-RF"], label="Market Beta", color="C1")
axes[1].set_ylabel("Beta")
axes[1].set_xlabel("Date")
axes[1].set_title("Rolling Market Beta")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("ff5_factor_regression.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Example 2: Compare Factor Loadings of Growth vs Value ETFs

Use the Carhart 4-factor model to compare factor exposures of a growth ETF
(IWF) and a value ETF (IWD), and run the GRS test to check if both alphas are
jointly zero.

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

START = "2010-01-01"
END   = "2024-12-31"

# ── Download factors ──────────────────────────────────────────────────────────
ff4 = download_ff_factors(model="4factor", start=START, end=END, freq="monthly")
factor_cols_4 = [c for c in ff4.columns if c != "RF"]
factors_4 = ff4[factor_cols_4]

# ── Download ETF prices ───────────────────────────────────────────────────────
etfs = ["IWF", "IWD"]   # iShares Russell 1000 Growth / Value
prices = yf.download(etfs, start=START, end=END, auto_adjust=True, progress=False)["Close"]

# ── Compute excess returns ────────────────────────────────────────────────────
rf = ff4["RF"]
excess_etfs = compute_excess_returns(prices, rf, freq="monthly")

# ── Individual regressions ────────────────────────────────────────────────────
results_summary = {}
for etf in etfs:
    r = run_factor_regression(excess_etfs[etf], factors_4)
    results_summary[etf] = r
    print(f"\n=== {etf} — Carhart 4-Factor Model ===")
    print(f"  Alpha (ann.):  {r['alpha']*100:.2f}%  t={r['alpha_tstat']:.2f}  p={r['alpha_pval']:.4f}")
    print(f"  R²:            {r['r_squared']:.4f}")
    for factor, beta in r["betas"].items():
        print(f"  {factor:12s}: {beta:.4f}")

# ── GRS test: are both alphas jointly zero? ───────────────────────────────────
grs = grs_test(excess_etfs[etfs], factors_4)
print(f"\n=== GRS Test ===")
print(f"  GRS F-statistic: {grs['grs_stat']:.4f}")
print(f"  p-value:         {grs['p_value']:.4f}")
print(f"  Individual alpha t-stats: IWF={grs['t_stats'][0]:.3f}, IWD={grs['t_stats'][1]:.3f}")
if grs["p_value"] < 0.05:
    print("  -> Reject H0: at least one alpha is statistically significant.")
else:
    print("  -> Fail to reject H0: alphas are jointly indistinguishable from zero.")

# ── Rolling loadings comparison ───────────────────────────────────────────────
roll_iwf = rolling_factor_loadings(excess_etfs["IWF"], factors_4, window=36)
roll_iwd = rolling_factor_loadings(excess_etfs["IWD"], factors_4, window=36)

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

factor_plot_pairs = [
    ("SMB", "Size (SMB) Exposure"),
    ("HML", "Value (HML) Exposure"),
    ("Mom", "Momentum (MOM) Exposure"),
]

for ax, (factor, title) in zip(axes, factor_plot_pairs):
    if factor in roll_iwf.columns:
        ax.plot(roll_iwf.index, roll_iwf[factor], label="IWF (Growth)", color="C0")
    if factor in roll_iwd.columns:
        ax.plot(roll_iwd.index, roll_iwd[factor], label="IWD (Value)", color="C3")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel("Beta")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Date")
plt.suptitle("Rolling 36-Month Factor Loadings: Growth vs Value ETFs (Carhart 4-Factor)", y=1.01)
plt.tight_layout()
plt.savefig("growth_vs_value_factor_loadings.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Bar chart: factor exposure comparison at full-sample ─────────────────────
bar_factors = list(results_summary["IWF"]["betas"].keys())
iwf_betas = [results_summary["IWF"]["betas"][f] for f in bar_factors]
iwd_betas = [results_summary["IWD"]["betas"][f] for f in bar_factors]

x = np.arange(len(bar_factors))
width = 0.35
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.bar(x - width / 2, iwf_betas, width, label="IWF (Growth)", color="steelblue")
ax2.bar(x + width / 2, iwd_betas, width, label="IWD (Value)", color="firebrick")
ax2.set_xticks(x)
ax2.set_xticklabels(bar_factors)
ax2.set_ylabel("Factor Loading (Beta)")
ax2.set_title("Full-Sample Factor Loadings: Growth vs Value (Carhart 4-Factor)")
ax2.legend()
ax2.axhline(0, color="black", linewidth=0.6)
ax2.grid(True, axis="y")
plt.tight_layout()
plt.savefig("factor_loadings_bar.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Notes and Best Practices

- **Newey-West standard errors**: For time-series regressions, heteroskedasticity
  and autocorrelation-consistent (HAC) standard errors improve inference. Pass
  `cov_type='HAC'` and `cov_kwds={'maxlags': 6}` to `sm.OLS.fit()`.
- **Frequency consistency**: Ensure prices, factors, and risk-free rates all use
  the same frequency. Do not mix daily prices with monthly factor data.
- **Stale RF alignment**: When resampling prices to monthly frequency, use
  month-end (`ME`) and align the RF series by index before subtracting.
- **GRS assumptions**: The GRS test assumes i.i.d. multivariate normal residuals
  and requires T > N + K. Verify this constraint before applying the test.
- **Look-ahead bias**: When running rolling regressions for backtesting, ensure
  you are using only data available at each estimation date.
- **Factor data license**: Fama-French factors from Kenneth French's website are
  freely available for academic research. Commercial use may require separate
  licensing agreements.

---

## Dependencies Installation

```bash
pip install numpy>=1.24.0 pandas>=2.0.0 pandas-datareader>=0.10.0 \
            statsmodels>=0.14.0 scipy>=1.11.0 matplotlib>=3.7.0 yfinance>=0.2.0
```
