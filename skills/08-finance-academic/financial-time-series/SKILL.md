---
name: financial-time-series
description: >
  Use this Skill for financial time series analysis: GARCH volatility modeling,
  cointegration, pairs trading, factor models, and risk decomposition.
tags:
  - finance
  - time-series
  - garch
  - volatility
  - quantitative-finance
version: "1.0.0"
authors:
  - name: Rosetta Skills Contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - numpy>=1.24
    - pandas>=2.0
    - scipy>=1.11
    - statsmodels>=0.14
    - arch>=6.2
    - matplotlib>=3.7
    - yfinance>=0.2
last_updated: "2026-03-17"
status: "stable"
---

# Financial Time Series Analysis

> **One-line summary**: Analyze financial time series with GARCH volatility models, cointegration tests, pairs trading signals, Fama-French factor regressions, and rolling risk decomposition.

---

## When to Use This Skill

- When modeling volatility clustering with GARCH/EGARCH models
- When testing and exploiting cointegration for pairs trading
- When running Fama-French 3/5 factor regressions for alpha/beta
- When computing rolling VaR, expected shortfall, and drawdowns
- When detecting structural breaks in financial time series
- When building momentum and mean-reversion trading signals

**Trigger keywords**: GARCH, volatility, cointegration, pairs trading, Fama-French, factor model, VaR, expected shortfall, momentum, mean reversion, financial time series, ARCH, unit root, ADF test, Kalman filter

---

## Background & Key Concepts

### GARCH(p,q) Volatility

$$
r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0,1)
$$

$$
\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2
$$

Stationarity: $\sum \alpha_i + \sum \beta_j < 1$.

### Cointegration (Engle-Granger)

Two I(1) series $X_t, Y_t$ are cointegrated if $\exists \beta: Y_t - \beta X_t = u_t$ where $u_t$ is I(0). The spread $u_t$ is mean-reverting and exploitable for pairs trading.

### Fama-French 3-Factor Model

$$
R_i - R_f = \alpha_i + \beta_{MKT}(R_m - R_f) + \beta_{SMB} \cdot SMB + \beta_{HML} \cdot HML + \epsilon_i
$$

---

## Environment Setup

### Install Dependencies

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 statsmodels>=0.14 \
            arch>=6.2 matplotlib>=3.7 yfinance>=0.2
```

### Verify Installation

```python
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import adfuller

# Quick test with synthetic data
np.random.seed(42)
returns = np.random.randn(500) * 0.01
am = arch_model(returns, vol='Garch', p=1, q=1)
res = am.fit(disp='off')
print(f"arch version OK; GARCH omega={res.params['omega']:.6f}")
```

---

## Core Workflow

### Step 1: GARCH Volatility Modeling

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf

# ------------------------------------------------------------------ #
# Download S&P 500 data and fit GARCH(1,1) volatility model
# ------------------------------------------------------------------ #
print("Downloading S&P 500 data...")
try:
    spy = yf.download("SPY", start="2010-01-01", end="2024-12-31", progress=False)
    prices = spy["Adj Close"].dropna()
except Exception:
    # Synthetic fallback
    np.random.seed(0)
    dates = pd.date_range("2010-01-01", periods=3780, freq='B')
    # Simulate GARCH(1,1) returns
    omega, alpha, beta = 5e-6, 0.09, 0.88
    n = len(dates)
    sigma2 = np.zeros(n); eps = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        eps[t] = np.sqrt(sigma2[t]) * np.random.randn()
    prices = pd.Series(100 * np.exp(np.cumsum(eps)), index=dates, name="SPY")

returns = 100 * prices.pct_change().dropna()

# ---- GARCH(1,1) fit -------------------------------------------- #
am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
res = am.fit(disp='off')
print(res.summary())

omega_hat = res.params['omega']
alpha_hat = res.params['alpha[1]']
beta_hat  = res.params['beta[1]']
persistence = alpha_hat + beta_hat
half_life = np.log(0.5) / np.log(persistence)

print(f"\nVolatility persistence: {persistence:.4f}")
print(f"Shock half-life: {half_life:.1f} trading days")
print(f"Long-run annualized vol: {np.sqrt(omega_hat/(1-persistence)*252):.2f}%")

# ---- Conditional volatility plot ------------------------------- #
cond_vol = res.conditional_volatility
annual_vol = cond_vol * np.sqrt(252)  # Annualize

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(returns.index, returns, 'k-', linewidth=0.4, alpha=0.7)
axes[0].set_title("Daily Returns (%)")
axes[0].set_ylabel("Return (%)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(returns.index, annual_vol, color='#e74c3c', linewidth=1)
axes[1].fill_between(returns.index, 0, annual_vol, alpha=0.2, color='#e74c3c')
axes[1].set_ylabel("Annualized Vol (%)")
axes[1].set_title("GARCH(1,1) Conditional Volatility")
axes[1].grid(True, alpha=0.3)

# Standardized residuals
std_resid = res.std_resid
axes[2].plot(returns.index, std_resid, 'b-', linewidth=0.4, alpha=0.7)
axes[2].axhline(0, color='gray', linewidth=0.5)
axes[2].axhline(3, color='red', linestyle='--', linewidth=1)
axes[2].axhline(-3, color='red', linestyle='--', linewidth=1)
axes[2].set_ylabel("Std. Residuals"); axes[2].set_xlabel("Date")
axes[2].set_title("Standardized Residuals (should be ~N(0,1))")
axes[2].grid(True, alpha=0.3)

plt.suptitle("GARCH(1,1) Volatility Model — SPY", y=1.01)
plt.tight_layout()
plt.savefig("garch_volatility.png", dpi=150)
plt.show()
```

### Step 2: Cointegration and Pairs Trading

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# ------------------------------------------------------------------ #
# Pairs trading: test cointegration and compute spread Z-score
# Using synthetic correlated price series
# ------------------------------------------------------------------ #
np.random.seed(7)
n = 1000
dates = pd.date_range("2020-01-01", periods=n, freq='B')

# Cointegrated pair: share common stochastic trend
trend = np.cumsum(np.random.randn(n) * 0.5)  # Common trend
spread_true = np.random.randn(n) * 2.0        # Stationary spread (mean=0)

price_X = 100 + trend + np.random.randn(n) * 0.3  # Stock X
price_Y = 50  + 0.6 * trend + spread_true + np.random.randn(n) * 0.3  # Stock Y

X = pd.Series(price_X, index=dates, name="Stock_X")
Y = pd.Series(price_Y, index=dates, name="Stock_Y")

# ---- Step 1: Unit root tests (Augmented Dickey-Fuller) --------- #
for name, series in [("Stock X", X), ("Stock Y", Y)]:
    adf_result = adfuller(series, autolag='AIC')
    print(f"ADF test — {name}: stat={adf_result[0]:.4f}, p={adf_result[1]:.4f} "
          f"({'I(1)' if adf_result[1] > 0.05 else 'stationary'})")

# ---- Step 2: Cointegration test (Engle-Granger) ---------------- #
score, pvalue, _ = coint(X, Y)
print(f"\nCointegration test: stat={score:.4f}, p={pvalue:.4f}")
print("Cointegrated!" if pvalue < 0.05 else "NOT cointegrated (p>0.05)")

# ---- Step 3: Estimate hedge ratio via OLS ---------------------- #
X_const = sm.add_constant(X)
ols_res = OLS(Y, X_const).fit()
beta_hedge = ols_res.params["Stock_X"]
alpha_const = ols_res.params["const"]
print(f"\nHedge ratio β = {beta_hedge:.4f}  (Y ≈ {alpha_const:.2f} + {beta_hedge:.4f}·X)")

# ---- Step 4: Spread and Z-score -------------------------------- #
spread = Y - beta_hedge * X - alpha_const

# Rolling z-score (entry/exit signals)
roll_mean = spread.rolling(window=60).mean()
roll_std  = spread.rolling(window=60).std()
z_score   = (spread - roll_mean) / roll_std

# Trading signals
entry_long  =  z_score < -2.0   # Buy spread (long Y, short X)
entry_short =  z_score > +2.0   # Short spread
exit_signal = (z_score.abs() < 0.5)

# ---- Plot ----------------------------------------------------- #
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes[0].plot(X.index, X, label="Stock X", linewidth=0.8)
axes[0].plot(Y.index, Y, label="Stock Y", linewidth=0.8)
axes[0].set_title("Cointegrated Price Pair"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(spread.index, spread, 'purple', linewidth=0.8)
axes[1].plot(roll_mean.index, roll_mean, 'gray', linestyle='--', linewidth=1, label='60d mean')
axes[1].set_title("Spread (Y - βX - α)"); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(z_score.index, z_score, 'b-', linewidth=0.7)
axes[2].axhline( 2, color='red',   linestyle='--', linewidth=1.5, label='Short (z>+2)')
axes[2].axhline(-2, color='green', linestyle='--', linewidth=1.5, label='Long (z<-2)')
axes[2].axhline( 0.5, color='gray', linestyle=':', linewidth=1)
axes[2].axhline(-0.5, color='gray', linestyle=':', linewidth=1)
axes[2].fill_between(z_score.index, -2, z_score, where=z_score<-2, alpha=0.3, color='green')
axes[2].fill_between(z_score.index, z_score, 2, where=z_score>2, alpha=0.3, color='red')
axes[2].set_title("Spread Z-Score — Trading Signals")
axes[2].set_xlabel("Date"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pairs_trading.png", dpi=150)
plt.show()

# Quick backtest stats
long_trades  = entry_long.sum()
short_trades = entry_short.sum()
print(f"\nLong entries: {long_trades}, Short entries: {short_trades}")
```

### Step 3: Factor Model Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ------------------------------------------------------------------ #
# Fama-French 3-Factor regression
# Uses synthetic factor data when FF data unavailable
# ------------------------------------------------------------------ #
np.random.seed(42)
n = 252 * 5  # 5 years of daily data
dates = pd.date_range("2019-01-01", periods=n, freq='B')

# Simulate FF3 factors (annualized / 252)
mkt_premium = np.random.randn(n) * 0.01 + 0.0003  # Market excess return
smb = np.random.randn(n) * 0.005                    # Small minus Big
hml = np.random.randn(n) * 0.004                    # High minus Low (Value)
rf  = np.full(n, 0.02/252)                           # Risk-free rate (2% annual)

# True betas for a hypothetical asset manager
true_alpha = 0.02/252   # Daily alpha
true_beta_mkt = 1.15
true_beta_smb = 0.3
true_beta_hml = -0.2

# Asset excess return
asset_excess = (true_alpha
                + true_beta_mkt * mkt_premium
                + true_beta_smb * smb
                + true_beta_hml * hml
                + np.random.randn(n) * 0.008)

factors = pd.DataFrame({
    "Mkt-RF": mkt_premium,
    "SMB": smb,
    "HML": hml,
    "Rf": rf,
}, index=dates)

y = pd.Series(asset_excess, index=dates, name="Asset excess return")

# ---- OLS regression ------------------------------------------- #
X = sm.add_constant(factors[["Mkt-RF", "SMB", "HML"]])
model = sm.OLS(y, X)
result = model.fit(cov_type='HAC', cov_kwds={'maxlags': 21})
print(result.summary())

# Extract factor exposures
alpha_daily = result.params["const"]
alpha_annual = alpha_daily * 252
t_alpha = result.tvalues["const"]
print(f"\nAnnualized alpha: {alpha_annual*100:.3f}%  (t={t_alpha:.2f})")
print(f"Market beta: {result.params['Mkt-RF']:.4f}  (true: {true_beta_mkt})")
print(f"SMB beta:    {result.params['SMB']:.4f}  (true: {true_beta_smb})")
print(f"HML beta:    {result.params['HML']:.4f}  (true: {true_beta_hml})")
print(f"R²: {result.rsquared:.4f}")

# ---- Rolling factor exposures (36-month window) ---------------- #
roll_window = 63  # ~3 months

def rolling_ols_beta(y_series, X_df, window):
    betas = {}
    for col in X_df.columns:
        betas[col] = [np.nan] * (window - 1)
        for end in range(window, len(y_series)+1):
            y_w = y_series.iloc[end-window:end]
            X_w = sm.add_constant(X_df.iloc[end-window:end])
            try:
                b = sm.OLS(y_w, X_w).fit().params[col]
            except Exception:
                b = np.nan
            betas[col].append(b)
    return pd.DataFrame(betas, index=y_series.index)

rolling_betas = rolling_ols_beta(y, factors[["Mkt-RF", "SMB", "HML"]], roll_window)

fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
colors = ['#e74c3c', '#3498db', '#2ecc71']
for ax, col, color, truth in zip(axes, ["Mkt-RF","SMB","HML"], colors, [true_beta_mkt, true_beta_smb, true_beta_hml]):
    ax.plot(rolling_betas.index, rolling_betas[col], color=color, linewidth=1.5)
    ax.axhline(truth, color='black', linestyle='--', linewidth=1, alpha=0.7, label=f'True β={truth}')
    ax.set_ylabel(f"β ({col})"); ax.legend(loc='upper right'); ax.grid(alpha=0.3)
axes[-1].set_xlabel("Date")
plt.suptitle("Rolling Fama-French Factor Exposures (63-day window)")
plt.tight_layout()
plt.savefig("factor_model.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### EGARCH and Fat-Tailed Distributions

```python
import numpy as np
from arch import arch_model

np.random.seed(0)
returns = np.random.randn(1000) * 0.01

# EGARCH(1,1) with Student's t innovations (handles fat tails)
am = arch_model(returns * 100, vol='EGARCH', p=1, q=1, dist='StudentsT')
res = am.fit(disp='off')
print(res.summary())

# Degrees of freedom (fat tails if nu < 10)
nu = res.params.get('nu', 'N/A')
print(f"\nStudent's t d.o.f.: {nu}")
print(f"EGARCH gamma (leverage): {res.params.get('gamma[1]', 'N/A'):.4f}")
```

---

## Troubleshooting

### Error: `ConvergenceWarning` in GARCH fit

**Cause**: Poor starting values or very short series.

**Fix**:
```python
am = arch_model(returns, vol='Garch', p=1, q=1)
res = am.fit(starting_values=np.array([0.01, 0.05, 0.90]), disp='off', options={'maxiter': 500})
```

### Error: `ValueError: NaN values` in ADF test

**Fix**: Drop NaN before testing:
```python
series = series.dropna()
result = adfuller(series, autolag='AIC')
```

### Spurious regression warning

Always difference I(1) series before OLS, or use proper cointegration:
```python
# Check order of integration first
from statsmodels.tsa.stattools import adfuller
p_level = adfuller(X)[1]
p_diff  = adfuller(X.diff().dropna())[1]
print(f"I(1) if p_level>0.05 and p_diff<0.05: {p_level:.3f}, {p_diff:.3f}")
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| arch | 6.2, 6.3 | `arch_model` API stable |
| statsmodels | 0.14 | `coint` requires statsmodels ≥ 0.11 |
| yfinance | 0.2.x | `Adj Close` column name may change |

---

## External Resources

### Official Documentation

- [arch library docs](https://arch.readthedocs.io)
- [statsmodels time series](https://www.statsmodels.org/stable/tsa.html)

### Key Papers

- Bollerslev, T. (1986). *Generalized autoregressive conditional heteroskedasticity*. Journal of Econometrics.
- Engle, R.F. & Granger, C.W.J. (1987). *Co-integration and error correction*. Econometrica.

---

## Examples

### Example 1: Value-at-Risk via Historical Simulation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
# Simulate portfolio returns (fat-tailed)
from scipy.stats import t as t_dist
returns = t_dist.rvs(df=5, size=1000) * 0.01

# Historical VaR
confidence = 0.99
var_99 = np.percentile(returns, (1-confidence)*100)
es_99  = returns[returns <= var_99].mean()

print(f"1-day VaR (99%): {var_99*100:.3f}%")
print(f"1-day ES  (99%): {es_99*100:.3f}%")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(returns*100, bins=50, color='steelblue', edgecolor='white', linewidth=0.5, alpha=0.7)
ax.axvline(var_99*100, color='red',    linewidth=2, linestyle='--', label=f'VaR(99%)={var_99*100:.2f}%')
ax.axvline(es_99*100,  color='orange', linewidth=2, linestyle='--', label=f'ES(99%)={es_99*100:.2f}%')
ax.set_xlabel("Daily Return (%)"); ax.set_title("Portfolio Return Distribution — VaR / ES")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("var_es.png", dpi=150); plt.show()
```

### Example 2: Hurst Exponent (Long Memory Detection)

```python
import numpy as np

def hurst_exponent(ts, max_lag=100):
    """Estimate Hurst exponent via R/S analysis."""
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        n_blocks = len(ts) // lag
        if n_blocks < 2:
            break
        rs_values = []
        for i in range(n_blocks):
            block = ts[i*lag:(i+1)*lag]
            mean_b = np.mean(block)
            dev = np.cumsum(block - mean_b)
            R = dev.max() - dev.min()
            S = np.std(block, ddof=1)
            if S > 0:
                rs_values.append(R/S)
        if rs_values:
            tau.append((lag, np.mean(rs_values)))

    lags_arr = np.array([x[0] for x in tau])
    rs_arr   = np.array([x[1] for x in tau])
    H, _ = np.polyfit(np.log(lags_arr), np.log(rs_arr), 1)
    return H

np.random.seed(1)
# Compare random walk (H≈0.5) vs. trending (H>0.5)
rw = np.cumsum(np.random.randn(1000))         # Random walk
trending = np.cumsum(np.random.randn(1000) + 0.02)  # With drift

print(f"Hurst(random walk):  H = {hurst_exponent(rw):.4f}  (expected ≈0.5)")
print(f"Hurst(with trend):   H = {hurst_exponent(trending):.4f}  (expected >0.5)")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
