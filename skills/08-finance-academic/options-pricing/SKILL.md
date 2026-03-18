---
name: options-pricing
description: >
  Use this Skill for option valuation: Black-Scholes pricing, Monte Carlo
  simulation with variance reduction, binomial tree, Greeks computation, and
  implied volatility surface.
tags:
  - finance
  - options
  - Black-Scholes
  - Monte-Carlo
  - Greeks
  - implied-volatility
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
    - numpy>=1.23
    - scipy>=1.9
    - matplotlib>=3.6
    - pandas>=1.5
last_updated: "2026-03-18"
status: stable
---

# Options Pricing

> **TL;DR** — Price European and American options using Black-Scholes analytics,
> Monte Carlo simulation with antithetic variates and control variates, and CRR
> binomial trees. Compute all Greeks, extract implied volatility with Brent's method,
> and visualize the implied volatility surface and smile.

---

## When to Use

| Situation | Recommended Method |
|---|---|
| European call/put, closed-form price | Black-Scholes formula |
| All five Greeks for a European option | Analytical BS Greeks |
| Confidence interval around MC price | Monte Carlo + standard error |
| Reduce MC variance | Antithetic variates / control variate |
| American option (early exercise) | Binomial CRR tree (backward induction) |
| Extract market-implied vol from option price | Brent root-finding on BS formula |
| Visualize vol smile/surface | Plot IV by strike and maturity |

---

## Background

### Black-Scholes-Merton Model (1973)

Assumes the underlying S_t follows GBM:
    dS_t = μ S_t dt + σ S_t dW_t

European call price:
    C = S N(d₁) - K e^{-rT} N(d₂)
    d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T

European put: P = K e^{-rT} N(-d₂) - S N(-d₁)

### Greeks

| Greek | Definition | Formula |
|---|---|---|
| Delta | ∂C/∂S | N(d₁) (call) |
| Gamma | ∂²C/∂S² | φ(d₁)/(S σ √T) |
| Vega | ∂C/∂σ | S φ(d₁) √T |
| Theta | ∂C/∂t | -(S φ(d₁) σ)/(2√T) - rK e^{-rT} N(d₂) |
| Rho | ∂C/∂r | K T e^{-rT} N(d₂) (call) |

### Monte Carlo Variance Reduction

**Antithetic variates**: For each path ε_t, also simulate -ε_t. Pair estimates and
average. Reduces variance by exploiting negative correlation between paired paths.

**Control variate**: Use BS analytical price as control:
    C_CV = C_MC + β (C_BS - E[C_BS])
where β = Cov(C_MC, C_control) / Var(C_control) ≈ 1 for European options.

Effective variance reduction factor can be 5–50x depending on the payoff.

### CRR Binomial Tree

Cox-Ross-Rubinstein (1979) recombining tree:
- Up factor: u = e^{σ√Δt}
- Down factor: d = 1/u
- Risk-neutral probability: p = (e^{rΔt} - d) / (u - d)

American option: at each node, take the maximum of intrinsic value and continuation
value (backward induction). European option: only backward induction without early
exercise check.

### Implied Volatility

Given observed market price C_mkt, find σ_IV such that:
    BS(S, K, T, r, σ_IV) = C_mkt

Solved numerically via scipy.optimize.brentq (Brent's method) over σ ∈ (0.001, 10).

---

## Environment Setup

```bash
conda create -n options python=3.11 -y
conda activate options
pip install numpy>=1.23 scipy>=1.9 matplotlib>=3.6 pandas>=1.5

python -c "import numpy, scipy, matplotlib; print('All packages OK')"
```

No external financial data API required. All examples use synthetic parameters.

---

## Core Workflow

### Step 1 — Black-Scholes Pricing and All Greeks

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import warnings


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes-Merton European option price.

    Args:
        S:           Current underlying price.
        K:           Strike price.
        T:           Time to expiry in years.
        r:           Continuously compounded risk-free rate.
        sigma:       Implied/historical volatility (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Option price.
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict:
    """
    Compute all five Black-Scholes Greeks analytically.

    Args:
        S:           Underlying price.
        K:           Strike price.
        T:           Time to expiry in years.
        r:           Risk-free rate (continuously compounded).
        sigma:       Volatility (annualized).
        option_type: 'call' or 'put'.

    Returns:
        Dictionary with keys: delta, gamma, vega, theta, rho.
        Theta is per calendar day (divided by 365).
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    phi_d1 = norm.pdf(d1)

    gamma = phi_d1 / (S * sigma * np.sqrt(T))
    vega = S * phi_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-(S * phi_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
        theta = (-(S * phi_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'")

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


def print_greeks_table(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
) -> pd.DataFrame:
    """
    Print a formatted table of BS prices and Greeks for call and put.

    Args:
        S, K, T, r, sigma: Standard Black-Scholes parameters.

    Returns:
        DataFrame with rows for call and put, columns for price and Greeks.
    """
    call_price = bs_price(S, K, T, r, sigma, "call")
    put_price = bs_price(S, K, T, r, sigma, "put")
    call_greeks = bs_greeks(S, K, T, r, sigma, "call")
    put_greeks = bs_greeks(S, K, T, r, sigma, "put")

    parity_check = call_price - put_price - (S - K * np.exp(-r * T))
    print(f"Put-Call Parity check: {parity_check:.2e}  (should be ~0)")

    rows = []
    for opt_type, price, greeks in [("call", call_price, call_greeks),
                                     ("put", put_price, put_greeks)]:
        rows.append({
            "type": opt_type,
            "price": round(price, 4),
            "delta": round(greeks["delta"], 4),
            "gamma": round(greeks["gamma"], 4),
            "vega": round(greeks["vega"], 4),
            "theta (per day)": round(greeks["theta"], 4),
            "rho": round(greeks["rho"], 4),
        })

    df = pd.DataFrame(rows)
    print(f"\nBS Option Parameters: S={S}, K={K}, T={T}y, r={r:.2%}, σ={sigma:.2%}")
    print(df.to_string(index=False))
    return df
```

### Step 2 — Monte Carlo with Antithetic and Control Variates

```python
def mc_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    n_paths: int = 100_000,
    n_steps: int = 1,
    use_antithetic: bool = True,
    use_control_variate: bool = True,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo European option pricing with variance reduction.

    Args:
        S:                   Initial underlying price.
        K:                   Strike price.
        T:                   Time to expiry (years).
        r:                   Risk-free rate.
        sigma:               Volatility.
        option_type:         'call' or 'put'.
        n_paths:             Number of GBM simulation paths.
        n_steps:             Time steps per path (1 = log-normal terminal price).
        use_antithetic:      Apply antithetic variates variance reduction.
        use_control_variate: Apply control variate (BS price).
        seed:                Random seed.

    Returns:
        Dictionary with keys: price, se, ci_lower, ci_upper, bs_price, variance_reduction.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if use_antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z_all = np.vstack([Z, -Z])
    else:
        Z_all = rng.standard_normal((n_paths, n_steps))

    # GBM simulation
    log_S = np.log(S) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_all, axis=1
    )
    S_T = np.exp(log_S[:, -1])

    # Payoff
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    # Discount
    payoffs_disc = np.exp(-r * T) * payoffs
    price_plain = float(np.mean(payoffs_disc))

    # Control variate: BS price as control
    if use_control_variate:
        bs_ref = bs_price(S, K, T, r, sigma, option_type)
        cov_matrix = np.cov(payoffs_disc, payoffs_disc)  # Placeholder; use S_T as control
        # Use S_T as a traded asset: known E[e^{-rT} S_T] = S (forward price)
        s_t_disc = np.exp(-r * T) * S_T
        beta = np.cov(payoffs_disc, s_t_disc)[0, 1] / np.var(s_t_disc)
        payoffs_cv = payoffs_disc - beta * (s_t_disc - S)
        price_cv = float(np.mean(payoffs_cv))
        se_cv = float(np.std(payoffs_cv) / np.sqrt(n_paths))
        var_reduction = float(np.var(payoffs_disc) / np.var(payoffs_cv))
        price_final = price_cv
        se_final = se_cv
    else:
        bs_ref = bs_price(S, K, T, r, sigma, option_type)
        var_reduction = 1.0
        price_final = price_plain
        se_final = float(np.std(payoffs_disc) / np.sqrt(n_paths))

    ci_lower = price_final - 1.96 * se_final
    ci_upper = price_final + 1.96 * se_final

    print(f"\nMC {option_type.upper()} Price: {price_final:.4f}  SE: {se_final:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  BS price: {bs_ref:.4f}  |  Variance reduction factor: {var_reduction:.2f}x")

    return {"price": price_final, "se": se_final,
            "ci_lower": ci_lower, "ci_upper": ci_upper,
            "bs_price": bs_ref, "variance_reduction": var_reduction}
```

### Step 3 — Implied Volatility Surface

```python
def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
) -> float:
    """
    Extract implied volatility from an observed market price using Brent's method.

    Args:
        market_price: Observed option price in the market.
        S:            Current underlying price.
        K:            Strike.
        T:            Time to expiry.
        r:            Risk-free rate.
        option_type:  'call' or 'put'.
        tol:          Solver tolerance.

    Returns:
        Implied volatility, or np.nan if no solution found.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == "call" else max(K * np.exp(-r * T) - S, 0)
    if market_price < intrinsic:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        iv = brentq(objective, 1e-4, 10.0, xtol=tol, full_output=False)
        return float(iv)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol_surface(
    S: float = 100.0,
    r: float = 0.03,
    strikes: list = None,
    maturities: list = None,
    sigma_atm: float = 0.20,
    skew: float = -0.02,
    term_slope: float = 0.01,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate a synthetic implied vol surface and plot as heatmap.

    Synthetic IV surface: σ(K, T) = σ_ATM + skew × moneyness + term_slope × T

    Args:
        S:           Current price.
        r:           Risk-free rate.
        strikes:     List of strikes. Default: S × [0.8, 0.85, ..., 1.2].
        maturities:  List of maturities in years. Default: [0.25, 0.5, 1.0, 1.5, 2.0].
        sigma_atm:   ATM implied vol level.
        skew:        Slope of vol smile with moneyness (ln(K/S)).
        term_slope:  Term structure slope per year.
        output_path: If provided, save heatmap.

    Returns:
        DataFrame of implied vols: rows = maturities, columns = strikes.
    """
    if strikes is None:
        strikes = [round(S * m, 1) for m in np.arange(0.80, 1.21, 0.05)]
    if maturities is None:
        maturities = [0.25, 0.5, 1.0, 1.5, 2.0]

    iv_matrix = {}
    for T in maturities:
        row = {}
        for K in strikes:
            moneyness = np.log(K / S)
            iv_synth = sigma_atm + skew * moneyness + term_slope * T
            iv_synth = max(0.05, iv_synth)
            # Generate market price and recover IV for self-consistency
            mkt_price = bs_price(S, K, T, r, iv_synth, "call")
            iv_rec = implied_vol(mkt_price, S, K, T, r, "call")
            row[K] = round(iv_rec if not np.isnan(iv_rec) else iv_synth, 4)
        iv_matrix[T] = row

    iv_df = pd.DataFrame(iv_matrix).T
    iv_df.index.name = "Maturity"

    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Heatmap
    im = axes[0].imshow(iv_df.values, aspect="auto", cmap="RdYlGn_r",
                        vmin=iv_df.values.min() * 0.95, vmax=iv_df.values.max() * 1.05)
    axes[0].set_xticks(range(len(strikes)))
    axes[0].set_xticklabels([str(k) for k in strikes], rotation=45, fontsize=8)
    axes[0].set_yticks(range(len(maturities)))
    axes[0].set_yticklabels([f"{m}y" for m in maturities])
    axes[0].set_xlabel("Strike")
    axes[0].set_ylabel("Maturity")
    axes[0].set_title("Implied Volatility Surface (Heatmap)")
    plt.colorbar(im, ax=axes[0])

    # Vol smile (cross-section at each maturity)
    for T in maturities:
        axes[1].plot(strikes, iv_df.loc[T].values, "o-", label=f"T={T}y", linewidth=1.5)
    axes[1].set_xlabel("Strike")
    axes[1].set_ylabel("Implied Volatility")
    axes[1].set_title("Implied Volatility Smile by Maturity")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved IV surface to {output_path}")
    plt.show()

    return iv_df
```

---

## Advanced Usage

### CRR Binomial Tree for American Options

```python
def binomial_tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int = 200,
    option_type: str = "call",
    american: bool = True,
) -> float:
    """
    CRR binomial tree pricing for European and American options.

    Args:
        S:           Underlying price.
        K:           Strike.
        T:           Time to expiry.
        r:           Risk-free rate.
        sigma:       Volatility.
        n_steps:     Number of time steps (higher = more accurate).
        option_type: 'call' or 'put'.
        american:    If True, allow early exercise (American option).

    Returns:
        Option price.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal stock prices
    j = np.arange(n_steps + 1)
    S_T = S * u ** (n_steps - 2 * j)

    # Terminal payoffs
    if option_type == "call":
        V = np.maximum(S_T - K, 0)
    else:
        V = np.maximum(K - S_T, 0)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            j_i = np.arange(i + 1)
            S_node = S * u ** (i - 2 * j_i)
            if option_type == "call":
                intrinsic = np.maximum(S_node - K, 0)
            else:
                intrinsic = np.maximum(K - S_node, 0)
            V = np.maximum(V, intrinsic)

    return float(V[0])


def early_exercise_premium(
    S: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.20,
) -> dict:
    """
    Compute early exercise premium: American price - European price.

    For calls on non-dividend paying stocks, this should be ~0 (no early exercise).
    For puts, the premium is positive.

    Args:
        S, K, T, r, sigma: Standard option parameters.

    Returns:
        Dictionary with european_call, american_call, european_put, american_put,
                         call_premium, put_premium.
    """
    eur_call = bs_price(S, K, T, r, sigma, "call")
    eur_put = bs_price(S, K, T, r, sigma, "put")
    amer_call = binomial_tree_price(S, K, T, r, sigma, n_steps=300, option_type="call", american=True)
    amer_put = binomial_tree_price(S, K, T, r, sigma, n_steps=300, option_type="put", american=True)

    call_prem = amer_call - eur_call
    put_prem = amer_put - eur_put

    print(f"\nEarly Exercise Premiums (S={S}, K={K}, T={T}, r={r:.1%}, σ={sigma:.1%}):")
    print(f"  Call: European={eur_call:.4f}  American={amer_call:.4f}  Premium={call_prem:.4f}")
    print(f"  Put:  European={eur_put:.4f}  American={amer_put:.4f}  Premium={put_prem:.4f}")

    return {"european_call": eur_call, "american_call": amer_call,
            "european_put": eur_put, "american_put": amer_put,
            "call_premium": call_prem, "put_premium": put_prem}
```

---

## Troubleshooting

| Error / Issue | Cause | Resolution |
|---|---|---|
| `brentq` raises `ValueError` | Market price below intrinsic or above S | Check arbitrage: intrinsic ≤ mkt_price ≤ S |
| IV = NaN for deep OTM options | Very low market price, solver loses bracket | Widen sigma bounds to (1e-5, 20.0) |
| MC price converges slowly | Too few paths | Use n_paths=500_000 or apply variance reduction |
| Binomial tree slow for large n_steps | O(n²) computation | Reduce to n_steps=200 for daily practice |
| BS Theta > 0 for deep ITM put | Theta can be positive for deep ITM puts | This is mathematically correct; not an error |
| Gamma → ∞ as T → 0 at-the-money | Near-expiry ATM option | Clip T at minimum 1e-6; expected behavior |

---

## External Resources

- Black, F., Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities."
  *Journal of Political Economy*, 81(3), 637–654.
- Merton, R.C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics*, 4(1), 141–183.
- Cox, J., Ross, S., Rubinstein, M. (1979). "Option Pricing: A Simplified Approach."
  *Journal of Financial Economics*, 7(3), 229–263.
- Hull, J.C. (2022). *Options, Futures, and Other Derivatives*, 11th Edition. Pearson.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.

---

## Examples

### Example 1 — BS Price + Greeks Parameter Table

```python
greeks_df = print_greeks_table(S=100, K=105, T=0.5, r=0.04, sigma=0.25)

# Vary spot price and plot Delta profile
spots = np.linspace(60, 140, 100)
deltas_call = [bs_greeks(s, 100, 0.5, 0.04, 0.25, "call")["delta"] for s in spots]
deltas_put = [bs_greeks(s, 100, 0.5, 0.04, 0.25, "put")["delta"] for s in spots]

plt.figure(figsize=(9, 4))
plt.plot(spots, deltas_call, label="Call Delta", color="#2980B9")
plt.plot(spots, deltas_put, label="Put Delta", color="#E74C3C")
plt.axvline(100, color="gray", linestyle=":", linewidth=1, label="ATM")
plt.xlabel("Spot Price")
plt.ylabel("Delta")
plt.title("Delta Profile vs Spot")
plt.legend()
plt.tight_layout()
plt.savefig("delta_profile.png", dpi=150)
plt.show()
```

### Example 2 — MC vs BS with Variance Reduction + IV Surface

```python
# Monte Carlo with antithetic + control variate
mc_result = mc_option_price(
    S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call",
    n_paths=100_000, use_antithetic=True, use_control_variate=True,
)
print(f"BS price: {mc_result['bs_price']:.4f}  |  MC price: {mc_result['price']:.4f}")

# American vs European put early exercise premium
premium = early_exercise_premium(S=100, K=110, T=1.0, r=0.05, sigma=0.25)

# Implied vol surface
iv_df = implied_vol_surface(
    S=100, r=0.03, sigma_atm=0.20, skew=-0.03, term_slope=0.01,
    output_path="iv_surface.png"
)
print("\nIV Surface (maturity × strike):")
print(iv_df.round(4))
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — BS pricing, Greeks, MC variance reduction, CRR tree, IV surface |
