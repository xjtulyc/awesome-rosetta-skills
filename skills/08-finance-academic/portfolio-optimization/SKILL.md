---
name: portfolio-optimization
description: >
  Use this Skill to build optimal portfolios: mean-variance frontier, minimum
  variance, maximum Sharpe, Black-Litterman views, and risk parity with
  PyPortfolioOpt.
tags:
  - finance
  - portfolio-optimization
  - mean-variance
  - Black-Litterman
  - risk-parity
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
    - PyPortfolioOpt>=1.5
    - cvxpy>=1.4
    - pandas>=1.5
    - numpy>=1.23
    - matplotlib>=3.6
    - scipy>=1.9
last_updated: "2026-03-18"
status: stable
---

# Portfolio Optimization

> **TL;DR** — Construct the mean-variance efficient frontier, find minimum-variance
> and maximum-Sharpe portfolios, incorporate investor views via Black-Litterman,
> and build risk parity portfolios using PyPortfolioOpt and CVXPY.

---

## When to Use

| Situation | Recommended Method |
|---|---|
| Allocate across risky assets, minimize variance at target return | Mean-variance optimization |
| Minimize variance without return target | Global Minimum Variance portfolio |
| Maximize risk-adjusted return (Sharpe ratio) | Tangency / Max-Sharpe portfolio |
| Incorporate subjective market views | Black-Litterman model |
| Equal risk contribution across assets | Risk Parity |
| Realistic portfolio with turnover constraints | Transaction cost-aware MVO |

---

## Background

### Mean-Variance Optimization (Markowitz 1952)

Given expected returns μ (k×1) and covariance matrix Σ (k×k), the efficient frontier
is traced by solving for each target return μ*:

    min  w' Σ w
    s.t. w' μ = μ*
         w' 1 = 1
         w ≥ 0  (long-only)

- **Global Minimum Variance (GMV)**: Drop the return constraint; minimize variance only.
- **Maximum Sharpe (Tangency)**: max (w' μ - r_f) / √(w' Σ w), equivalent to
  a convex problem after transformation.

### Shrinkage Estimators

Sample covariance matrix is ill-conditioned with many assets. Ledoit-Wolf (2004)
shrinkage estimator:
    Σ̂_LW = (1 - α) Σ_sample + α μ_LW I
reduces estimation error. PyPortfolioOpt wraps sklearn's LedoitWolf.

### Black-Litterman Model (1990/1992)

Combines the market equilibrium prior (from CAPM reverse optimization) with
investor views:

- **Prior**: Π = δ Σ w_mkt  (implied equilibrium returns)
- **Views**: P Q with uncertainty Ω (diagonal)
- **Posterior**: μ_BL = [(τΣ)^{-1} + P' Ω^{-1} P]^{-1} [(τΣ)^{-1} Π + P' Ω^{-1} Q]

where τ is a scalar (uncertainty in the prior, typically 0.05–0.25).

### Risk Parity / Equal Risk Contribution

The risk contribution of asset i is:
    RC_i = w_i × (∂σ_p / ∂w_i) = w_i × (Σ w)_i / σ_p

Equal Risk Contribution (ERC): solve w* such that RC_i = σ_p / k for all i.
This is a nonlinear optimization, solvable via CVXPY or scipy.

---

## Environment Setup

```bash
conda create -n portopt python=3.11 -y
conda activate portopt
pip install PyPortfolioOpt>=1.5 cvxpy>=1.4 pandas>=1.5 numpy>=1.23 matplotlib>=3.6 scipy>=1.9

python -c "from pypfopt import EfficientFrontier; print('PyPortfolioOpt OK')"
python -c "import cvxpy as cp; print('CVXPY', cp.__version__)"
```

---

## Core Workflow

### Step 1 — Efficient Frontier and Optimal Portfolios

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.plotting import plot_efficient_frontier
import warnings

np.random.seed(42)


def simulate_returns(
    n_assets: int = 10,
    n_periods: int = 252 * 5,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    Simulate daily asset returns with a factor structure.

    Args:
        n_assets:  Number of risky assets.
        n_periods: Number of daily return observations.
        annualize: If True, return annualized returns (252-day).

    Returns:
        DataFrame of daily returns (n_periods × n_assets).
    """
    # Factor loadings: all assets load on market factor + idiosyncratic
    beta = np.random.uniform(0.5, 1.5, n_assets)
    alpha = np.random.uniform(-0.0002, 0.0005, n_assets)

    market = np.random.normal(0.0004, 0.01, n_periods)
    idio = np.random.normal(0, 0.008, (n_periods, n_assets))

    returns_raw = alpha + np.outer(market, beta) + idio
    cols = [f"Asset{i+1}" for i in range(n_assets)]
    df = pd.DataFrame(returns_raw, columns=cols)
    df.index = pd.date_range("2020-01-01", periods=n_periods, freq="B")
    return df


def plot_efficient_frontier_custom(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.02,
    n_points: int = 100,
    output_path: str = None,
) -> dict:
    """
    Compute and plot the efficient frontier with key portfolio markers.

    Args:
        returns_df:     Daily returns DataFrame (T × k).
        risk_free_rate: Annual risk-free rate for Sharpe ratio.
        n_points:       Number of frontier points.
        output_path:    If provided, save the plot.

    Returns:
        Dictionary with keys: mu, S, min_vol_weights, max_sharpe_weights,
                              min_vol_perf, max_sharpe_perf.
    """
    mu = expected_returns.mean_historical_return(returns_df)
    S = risk_models.ledoit_wolf(returns_df)

    # Efficient frontier: vary target return
    ef_frontier_points = []
    min_ret = float(mu.min()) * 0.5
    max_ret = float(mu.max()) * 0.95

    for target_ret in np.linspace(min_ret, max_ret, n_points):
        try:
            ef = EfficientFrontier(mu, S)
            ef.efficient_return(target_ret, market_neutral=False)
            perf = ef.portfolio_performance(risk_free_rate=risk_free_rate / 252)
            ef_frontier_points.append({"sigma": perf[1], "mu": perf[0]})
        except Exception:
            pass

    ef_df = pd.DataFrame(ef_frontier_points)

    # GMV portfolio
    ef_gmv = EfficientFrontier(mu, S)
    ef_gmv.min_volatility()
    gmv_weights = ef_gmv.clean_weights()
    gmv_perf = ef_gmv.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate / 252)

    # Max Sharpe portfolio
    ef_ms = EfficientFrontier(mu, S)
    ef_ms.max_sharpe(risk_free_rate=risk_free_rate / 252)
    ms_weights = ef_ms.clean_weights()
    ms_perf = ef_ms.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate / 252)

    print(f"GMV Portfolio: return={gmv_perf[0]*252:.4f}  vol={gmv_perf[1]*np.sqrt(252):.4f}  Sharpe={gmv_perf[2]:.4f}")
    print(f"Max Sharpe:    return={ms_perf[0]*252:.4f}  vol={ms_perf[1]*np.sqrt(252):.4f}  Sharpe={ms_perf[2]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ef_df["sigma"] * np.sqrt(252), ef_df["mu"] * 252,
            color="#2980B9", linewidth=2, label="Efficient Frontier")
    ax.scatter(gmv_perf[1] * np.sqrt(252), gmv_perf[0] * 252,
               marker="*", s=200, color="#2ECC71", zorder=5, label="Min Variance")
    ax.scatter(ms_perf[1] * np.sqrt(252), ms_perf[0] * 252,
               marker="*", s=200, color="#E74C3C", zorder=5, label="Max Sharpe")

    # Individual assets
    for asset in returns_df.columns:
        asset_vol = float(returns_df[asset].std() * np.sqrt(252))
        asset_ret = float(mu[asset] * 252)
        ax.scatter(asset_vol, asset_ret, color="gray", alpha=0.6, s=40)
        ax.annotate(asset, (asset_vol, asset_ret), textcoords="offset points",
                    xytext=(3, 3), fontsize=7)

    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Mean-Variance Efficient Frontier")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved frontier plot to {output_path}")
    plt.show()

    return {"mu": mu, "S": S,
            "min_vol_weights": gmv_weights, "max_sharpe_weights": ms_weights,
            "min_vol_perf": gmv_perf, "max_sharpe_perf": ms_perf}
```

### Step 2 — Black-Litterman with Investor Views

```python
from pypfopt.black_litterman import BlackLittermanModel, market_implied_risk_aversion
from pypfopt.expected_returns import ema_historical_return


def black_litterman_optimization(
    returns_df: pd.DataFrame,
    market_caps: pd.Series = None,
    views_dict: dict = None,
    view_confidences: list = None,
    tau: float = 0.05,
    risk_free_rate: float = 0.02,
    output_path: str = None,
) -> dict:
    """
    Black-Litterman model: combine equilibrium with investor views.

    Args:
        returns_df:        Daily returns DataFrame.
        market_caps:       Market capitalizations for each asset (for prior weights).
                           If None, equal weights are used.
        views_dict:        Dictionary of views: {asset_name: expected_return}.
                           Example: {"Asset1": 0.10, "Asset3": 0.05}
        view_confidences:  Confidence level (0–1) for each view. Default: 0.5 each.
        tau:               Uncertainty in the prior (typically 0.01–0.25).
        risk_free_rate:    Annual risk-free rate.
        output_path:       If provided, save weight comparison bar chart.

    Returns:
        Dictionary with bl_returns, bl_weights, bl_performance.
    """
    n = len(returns_df.columns)
    S = risk_models.ledoit_wolf(returns_df)

    if market_caps is None:
        market_caps = pd.Series(np.ones(n), index=returns_df.columns)
    market_weights = market_caps / market_caps.sum()

    # Implied equilibrium returns (reverse CAPM)
    delta = market_implied_risk_aversion(returns_df.mean() * 252, market_weights, S)
    pi = BlackLittermanModel.market_implied_prior_returns(market_weights, delta, S)

    if views_dict is None:
        # Default: asset 1 outperforms by 3% vs asset 2
        assets = returns_df.columns.tolist()
        views_dict = {assets[0]: float(pi[assets[0]]) + 0.03,
                      assets[1]: float(pi[assets[1]]) - 0.02}

    if view_confidences is None:
        view_confidences = [0.5] * len(views_dict)

    # Build P matrix (absolute views)
    view_assets = list(views_dict.keys())
    P = np.zeros((len(view_assets), n))
    Q = np.zeros(len(view_assets))
    assets_list = returns_df.columns.tolist()
    for i, asset in enumerate(view_assets):
        P[i, assets_list.index(asset)] = 1.0
        Q[i] = views_dict[asset]

    # Omega: diagonal uncertainty matrix from confidences
    variances = np.diag(P @ S.values @ P.T)
    omega = np.diag(variances * (1 / np.array(view_confidences) - 1))

    bl = BlackLittermanModel(S, pi=pi, P=P, Q=Q, omega=omega, tau=tau)
    bl_returns = bl.bl_returns()

    ef_bl = EfficientFrontier(bl_returns, S)
    ef_bl.max_sharpe(risk_free_rate=risk_free_rate / 252)
    bl_weights = ef_bl.clean_weights()
    bl_perf = ef_bl.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate / 252)

    print(f"\nBlack-Litterman Max-Sharpe Portfolio:")
    for asset, w in bl_weights.items():
        if w > 0.01:
            print(f"  {asset}: {w:.4f}")
    print(f"Return: {bl_perf[0]*252:.4f}  Vol: {bl_perf[1]*np.sqrt(252):.4f}  Sharpe: {bl_perf[2]:.4f}")

    # Compare BL returns vs prior
    if output_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(n)
        width = 0.35
        ax.bar(x - width/2, pi.values * 252, width, label="Prior (Equilibrium)", color="#AED6F1")
        ax.bar(x + width/2, bl_returns.values * 252, width, label="BL Posterior", color="#2980B9")
        ax.set_xticks(x)
        ax.set_xticklabels(returns_df.columns, rotation=45)
        ax.set_ylabel("Expected Return (annual)")
        ax.set_title("Black-Litterman: Prior vs Posterior Expected Returns")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        print(f"Saved BL return comparison to {output_path}")

    return {"bl_returns": bl_returns, "bl_weights": bl_weights, "bl_performance": bl_perf}
```

### Step 3 — Risk Parity via CVXPY

```python
import cvxpy as cp
from scipy.optimize import minimize


def risk_parity_portfolio(
    S: pd.DataFrame,
    output_path: str = None,
) -> dict:
    """
    Equal Risk Contribution (risk parity) portfolio via scipy optimization.

    Each asset contributes equally to total portfolio volatility.

    Args:
        S:           Covariance matrix (pd.DataFrame or np.ndarray).
        output_path: If provided, save weight comparison plot.

    Returns:
        Dictionary with keys: weights, risk_contributions, portfolio_vol.
    """
    Sigma = S.values if isinstance(S, pd.DataFrame) else S
    n = Sigma.shape[0]
    asset_names = list(S.index) if isinstance(S, pd.DataFrame) else [f"A{i+1}" for i in range(n)]

    def risk_parity_objective(w):
        w = np.abs(w) / np.sum(np.abs(w))
        sigma_p = np.sqrt(w @ Sigma @ w)
        # Risk contributions
        marginal_rc = Sigma @ w
        rc = w * marginal_rc / sigma_p
        # Objective: sum of squared deviations from equal risk contribution
        target_rc = np.ones(n) / n
        return float(np.sum((rc / sigma_p - target_rc) ** 2))

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(risk_parity_objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 5000})

    w_rp = np.abs(result.x) / np.sum(np.abs(result.x))
    sigma_p = float(np.sqrt(w_rp @ Sigma @ w_rp))
    marginal_rc = Sigma @ w_rp
    rc = w_rp * marginal_rc / sigma_p

    print("\nRisk Parity Portfolio:")
    for name, wt, rci in zip(asset_names, w_rp, rc):
        print(f"  {name}: weight={wt:.4f}  RC={rci:.4f}  ({rci/sigma_p:.1%} of vol)")
    print(f"Portfolio volatility: {sigma_p:.4f}")

    # Compare: equal weight vs risk parity
    w_ew = np.ones(n) / n
    sigma_ew = float(np.sqrt(w_ew @ Sigma @ w_ew))

    if output_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(n)
        width = 0.35
        ax.bar(x - width/2, w_ew, width, label="Equal Weight", color="#AED6F1")
        ax.bar(x + width/2, w_rp, width, label="Risk Parity", color="#2ECC71")
        ax.set_xticks(x)
        ax.set_xticklabels(asset_names, rotation=45)
        ax.set_ylabel("Portfolio Weight")
        ax.set_title("Equal Weight vs Risk Parity Allocation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        print(f"Saved risk parity comparison to {output_path}")

    return {"weights": dict(zip(asset_names, w_rp)),
            "risk_contributions": dict(zip(asset_names, rc)),
            "portfolio_vol": sigma_p,
            "equal_weight_vol": sigma_ew}
```

---

## Advanced Usage

### Rolling Rebalancing Backtest

```python
def rolling_rebalance_backtest(
    returns_df: pd.DataFrame,
    method: str = "max_sharpe",
    lookback: int = 252,
    rebalance_freq: int = 63,
    risk_free_rate: float = 0.02,
    transaction_cost: float = 0.001,
) -> pd.DataFrame:
    """
    Walk-forward portfolio backtest with periodic rebalancing.

    Args:
        returns_df:       Daily returns (T × k).
        method:           'max_sharpe', 'min_vol', or 'equal_weight'.
        lookback:         Estimation window in trading days.
        rebalance_freq:   Rebalancing frequency in trading days.
        risk_free_rate:   Annual risk-free rate.
        transaction_cost: One-way cost per unit of weight change.

    Returns:
        DataFrame with columns: date, portfolio_return, cumulative_return, weights.
    """
    T, k = returns_df.shape
    assets = returns_df.columns.tolist()
    portfolio_returns = []
    current_weights = np.ones(k) / k
    rebalance_dates = list(range(lookback, T, rebalance_freq))

    for t in range(lookback, T):
        if t in rebalance_dates:
            hist = returns_df.iloc[t - lookback:t]
            try:
                mu = expected_returns.mean_historical_return(hist)
                S = risk_models.ledoit_wolf(hist)
                ef = EfficientFrontier(mu, S)

                if method == "max_sharpe":
                    ef.max_sharpe(risk_free_rate=risk_free_rate / 252)
                elif method == "min_vol":
                    ef.min_volatility()
                else:
                    new_weights = np.ones(k) / k
                    portfolio_returns.append({
                        "date": returns_df.index[t],
                        "portfolio_return": float(returns_df.iloc[t].values @ current_weights),
                    })
                    current_weights = new_weights
                    continue

                w_dict = ef.clean_weights()
                new_weights = np.array([w_dict.get(a, 0) for a in assets])
            except Exception:
                new_weights = current_weights.copy()

            # Transaction cost
            turnover = np.sum(np.abs(new_weights - current_weights))
            tc = turnover * transaction_cost
            current_weights = new_weights
        else:
            tc = 0.0

        daily_ret = float(returns_df.iloc[t].values @ current_weights) - tc
        portfolio_returns.append({"date": returns_df.index[t], "portfolio_return": daily_ret})

    port_df = pd.DataFrame(portfolio_returns)
    port_df["cumulative_return"] = (1 + port_df["portfolio_return"]).cumprod()
    return port_df
```

---

## Troubleshooting

| Error / Issue | Cause | Resolution |
|---|---|---|
| `OptimizationError: Optimization failed` | Expected return out of frontier | Reduce target return or use `efficient_risk()` |
| GMV weights all in 1 asset | Near-singular covariance matrix | Use Ledoit-Wolf shrinkage instead of sample covariance |
| Black-Litterman posterior = prior | Views matrix P rank-deficient | Check P dimensions match (n_views × n_assets) |
| Risk parity optimizer diverges | Very correlated assets | Add small ridge regularization: Σ += 1e-4 × I |
| Negative weights after `clean_weights()` | Rounding issue | Set `clean_weights(cutoff=1e-4)` |
| Covariance matrix not PSD | Short history or collinear assets | Use `risk_models.fix_nonpositive_semidefinite(S)` |

---

## External Resources

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77–91.
- Black, F., Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28–43.
- Ledoit, O., Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices."
  *Journal of Multivariate Analysis*, 88(2), 365–411.
- Maillard, S., Roncalli, T., Teïletche, J. (2010). "On the Properties of Equally Weighted Risk
  Contribution Portfolios." *Journal of Portfolio Management*, 36(4).
- `PyPortfolioOpt` docs: <https://pyportfolioopt.readthedocs.io/>
- `CVXPY` docs: <https://www.cvxpy.org/>

---

## Examples

### Example 1 — Efficient Frontier with Optimal Portfolios

```python
returns_df = simulate_returns(n_assets=10, n_periods=1260)

frontier_results = plot_efficient_frontier_custom(
    returns_df, risk_free_rate=0.02, output_path="efficient_frontier.png"
)
print("\nMax Sharpe weights (top 5):")
ms_w = frontier_results["max_sharpe_weights"]
for asset, w in sorted(ms_w.items(), key=lambda x: -x[1])[:5]:
    print(f"  {asset}: {w:.4f}")
```

### Example 2 — Black-Litterman and Risk Parity Comparison

```python
returns_df = simulate_returns(n_assets=8, n_periods=1260)
assets = returns_df.columns.tolist()

# Black-Litterman with 3 views
bl_result = black_litterman_optimization(
    returns_df,
    views_dict={assets[0]: 0.12, assets[1]: 0.05, assets[4]: 0.08},
    view_confidences=[0.8, 0.5, 0.6],
    output_path="bl_returns.png",
)

# Risk parity
S = risk_models.ledoit_wolf(returns_df)
rp_result = risk_parity_portfolio(S, output_path="risk_parity.png")

# Backtest comparison
bt_ms = rolling_rebalance_backtest(returns_df, method="max_sharpe")
bt_ew = rolling_rebalance_backtest(returns_df, method="equal_weight")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(bt_ms["date"], bt_ms["cumulative_return"], label="Max Sharpe")
ax.plot(bt_ew["date"], bt_ew["cumulative_return"], label="Equal Weight", linestyle="--")
ax.set_title("Portfolio Backtest: Max Sharpe vs Equal Weight")
ax.set_ylabel("Cumulative Return")
ax.legend()
plt.tight_layout()
plt.savefig("backtest.png", dpi=150)
plt.show()
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — MVO, GMV, Max Sharpe, BL, Risk Parity, backtest |
