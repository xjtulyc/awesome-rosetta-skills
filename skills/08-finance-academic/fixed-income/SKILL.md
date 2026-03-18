---
name: fixed-income
description: >
  Use this Skill for fixed income analytics: bond pricing, yield curves,
  duration/convexity, Nelson-Siegel fitting, and credit spread modeling.
tags:
  - finance
  - fixed-income
  - bonds
  - yield-curve
  - duration
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
    - scipy>=1.11
    - pandas>=2.0
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# Fixed Income Analytics

> **One-line summary**: Price bonds, bootstrap yield curves, compute duration and convexity, fit Nelson-Siegel models, and analyze credit spreads using pure Python with numpy and scipy.

---

## When to Use This Skill

- When pricing fixed-rate bonds, floaters, or zero-coupon bonds
- When computing modified duration, convexity, and DV01
- When bootstrapping spot rates from par yields or bond prices
- When fitting Nelson-Siegel or Svensson yield curve models
- When modeling credit spreads and CDS pricing
- When computing carry, roll-down, and term premium

**Trigger keywords**: bond pricing, yield to maturity, duration, convexity, DV01, PVBP, yield curve, Nelson-Siegel, Svensson, spot rate, forward rate, credit spread, CDS, OAS, zero curve, discount factor

---

## Background & Key Concepts

### Bond Pricing

A coupon bond with face value $F$, coupon $C$, yield $y$, and $T$ periods:

$$
P = \sum_{t=1}^{T} \frac{C}{(1+y)^t} + \frac{F}{(1+y)^T}
$$

### Modified Duration and Convexity

$$
\text{Duration} = \frac{1}{P}\sum_{t=1}^T \frac{t \cdot CF_t}{(1+y)^t}
$$

$$
\text{Convexity} = \frac{1}{P}\sum_{t=1}^T \frac{t(t+1) \cdot CF_t}{(1+y)^{t+2}}
$$

Price change: $\Delta P \approx -D^* \cdot P \cdot \Delta y + \frac{1}{2} C \cdot P \cdot (\Delta y)^2$

### Nelson-Siegel Model

$$
y(m) = \beta_0 + \beta_1 \frac{1-e^{-m/\lambda}}{m/\lambda} + \beta_2 \left(\frac{1-e^{-m/\lambda}}{m/\lambda} - e^{-m/\lambda}\right)
$$

where $m$ = maturity, $\beta_0$ = long-run level, $\beta_1$ = slope, $\beta_2$ = curvature.

---

## Environment Setup

### Install Dependencies

```bash
pip install numpy>=1.24 scipy>=1.11 pandas>=2.0 matplotlib>=3.7
```

### Verify Installation

```python
import numpy as np
from scipy.optimize import brentq

# Simple bond pricing test
def bond_price(face, coupon, y, n):
    """Annual coupon bond price given yield."""
    t = np.arange(1, n+1)
    pv = coupon / (1+y)**t
    return pv.sum() + face / (1+y)**n

price = bond_price(1000, 50, 0.05, 10)
print(f"10Y 5% coupon bond at y=5%: ${price:.4f}")
# Expected: 1000.00 (par bond)
```

---

## Core Workflow

### Step 1: Bond Pricing and Duration/Convexity

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ------------------------------------------------------------------ #
# Bond analytics: price, YTM, duration, convexity, DV01
# ------------------------------------------------------------------ #

class Bond:
    """Fixed-rate bond with annual coupon payments."""

    def __init__(self, face, coupon_rate, maturity, settlement_price=None, ytm=None):
        """
        Parameters
        ----------
        face : float — face (par) value
        coupon_rate : float — annual coupon rate (e.g., 0.05 for 5%)
        maturity : int — years to maturity
        settlement_price : float — dirty price (optional)
        ytm : float — yield to maturity (optional)
        """
        self.face = face
        self.coupon = face * coupon_rate
        self.T = maturity
        self.cashflows = np.array([self.coupon] * maturity)
        self.cashflows[-1] += face  # Final payment = coupon + face
        self.times = np.arange(1, maturity + 1, dtype=float)

        if ytm is not None:
            self.ytm = ytm
            self.price = self._price_from_ytm(ytm)
        elif settlement_price is not None:
            self.price = settlement_price
            self.ytm = self._ytm_from_price(settlement_price)
        else:
            raise ValueError("Provide either ytm or settlement_price")

    def _price_from_ytm(self, y):
        """Compute dirty price from YTM."""
        discount = (1 + y) ** (-self.times)
        return np.dot(self.cashflows, discount)

    def _ytm_from_price(self, P):
        """Find YTM via root-finding."""
        f = lambda y: self._price_from_ytm(y) - P
        return brentq(f, -0.5, 10.0)

    @property
    def modified_duration(self):
        """Modified duration (years)."""
        discount = (1 + self.ytm) ** (-self.times)
        pv_cf = self.cashflows * discount
        macaulay = np.dot(self.times, pv_cf) / self.price
        return macaulay / (1 + self.ytm)

    @property
    def convexity(self):
        """Convexity (years²)."""
        discount = (1 + self.ytm) ** (-self.times)
        pv_cf = self.cashflows * discount
        return np.dot(self.times * (self.times + 1), pv_cf) / (self.price * (1 + self.ytm)**2)

    @property
    def dv01(self):
        """Dollar value of 1 basis point (per face)."""
        return self.modified_duration * self.price * 0.0001

    def price_at_yield(self, y):
        return self._price_from_ytm(y)

    def summary(self):
        return {
            'Price': round(self.price, 4),
            'YTM': round(self.ytm * 100, 4),
            'Mod Duration': round(self.modified_duration, 4),
            'Convexity': round(self.convexity, 4),
            'DV01': round(self.dv01, 4),
        }

# ---- Example bonds --------------------------------------------- #
bonds = {
    '2Y  5% par bond':  Bond(1000, 0.05, 2,  ytm=0.05),
    '10Y 5% par bond':  Bond(1000, 0.05, 10, ytm=0.05),
    '30Y 5% par bond':  Bond(1000, 0.05, 30, ytm=0.05),
    '10Y 3% discount':  Bond(1000, 0.03, 10, ytm=0.05),
    '10Y 0% zero':      Bond(1000, 0.00, 10, ytm=0.05),
}

print(f"{'Bond':22s}  {'Price':>8s}  {'YTM%':>6s}  {'Dur':>6s}  {'Conv':>8s}  {'DV01':>7s}")
print("-" * 70)
for name, bond in bonds.items():
    s = bond.summary()
    print(f"{name:22s}  {s['Price']:8.2f}  {s['YTM']:6.2f}  {s['Mod Duration']:6.3f}  "
          f"{s['Convexity']:8.4f}  {s['DV01']:7.4f}")

# ---- Price-yield relationship ---------------------------------- #
bond_10y = bonds['10Y 5% par bond']
yields = np.linspace(0.01, 0.15, 200)
prices = [bond_10y.price_at_yield(y) for y in yields]

# Duration approximation
dy = np.array(yields) - bond_10y.ytm
p_linear = bond_10y.price * (1 - bond_10y.modified_duration * dy)
p_convex  = bond_10y.price * (1 - bond_10y.modified_duration * dy
                               + 0.5 * bond_10y.convexity * dy**2)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(yields * 100, prices, 'b-', linewidth=2.5, label='Exact price')
ax.plot(yields * 100, p_linear, 'r--', linewidth=1.5, label='Duration approx (linear)')
ax.plot(yields * 100, p_convex, 'g-.', linewidth=1.5, label='Duration + Convexity')
ax.axvline(bond_10y.ytm * 100, color='gray', linestyle=':', linewidth=1)
ax.axhline(bond_10y.price, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel("Yield (%)"); ax.set_ylabel("Price ($)")
ax.set_title("Price-Yield Relationship — 10Y 5% Bond")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bond_price_yield.png", dpi=150)
plt.show()
```

### Step 2: Yield Curve Bootstrapping

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ------------------------------------------------------------------ #
# Bootstrap zero/spot curve from par yields (Treasury benchmark)
# ------------------------------------------------------------------ #

# Par yield curve (observed market data)
maturities = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
par_yields  = np.array([5.20, 5.10, 4.90, 4.75, 4.55, 4.45, 4.30, 4.40, 4.35]) / 100

# ---- Bootstrap spot rates --------------------------------------- #
spot_rates = np.zeros(len(maturities))

for i, (T, y_par) in enumerate(zip(maturities, par_yields)):
    if T <= 1:
        # Short end: spot ≈ par (no intermediate coupons)
        spot_rates[i] = y_par
    else:
        # Price = 1 (par bond); coupon = y_par per period (semi-annual)
        # Sum of discounted intermediate coupons using already-bootstrapped spots
        n_periods = int(T * 2)  # Semi-annual
        coupon = y_par / 2       # Semi-annual coupon rate

        # Interpolate spots for intermediate maturities
        t_interp = np.arange(0.5, T, 0.5)  # Intermediate dates

        if i > 0:
            cs = CubicSpline(maturities[:i], spot_rates[:i])
            spots_interp = cs(t_interp)
        else:
            spots_interp = np.array([y_par])

        # Present value of intermediate coupons
        pv_coupons = sum(coupon / (1 + s/2)**(2*t)
                         for t, s in zip(t_interp, spots_interp))

        # Solve for final spot rate
        # 1 = pv_coupons + (1 + coupon) / (1 + s_T/2)^(2T)
        remaining = 1 - pv_coupons
        s_T = 2 * ((1 + coupon) / remaining) ** (1/(2*T)) - 2
        spot_rates[i] = s_T

# ---- Forward rates from spot rates ----------------------------- #
def forward_rate(s1, t1, s2, t2):
    """Continuously compounded forward rate between t1 and t2."""
    return (s2 * t2 - s1 * t1) / (t2 - t1)

forward_rates = []
t_fwd = []
for i in range(1, len(maturities)):
    f = forward_rate(spot_rates[i-1], maturities[i-1],
                     spot_rates[i], maturities[i])
    forward_rates.append(f)
    t_fwd.append((maturities[i-1] + maturities[i]) / 2)

# ---- Discount factors --------------------------------------- #
discount_factors = np.exp(-spot_rates * maturities)

df_result = pd.DataFrame({
    'Maturity':  maturities,
    'Par Yield': par_yields * 100,
    'Spot Rate': spot_rates * 100,
    'Discount':  discount_factors.round(6),
})
print(df_result.to_string(index=False))

# ---- Plot ---------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(maturities, par_yields * 100, 'bo-', label='Par yields', linewidth=1.5, markersize=7)
axes[0].plot(maturities, spot_rates * 100, 'rs-', label='Spot rates', linewidth=1.5, markersize=7)
axes[0].plot(t_fwd, np.array(forward_rates) * 100, 'g^--', label='Forward rates', linewidth=1.5, markersize=6)
axes[0].set_xlabel("Maturity (years)"); axes[0].set_ylabel("Rate (%)")
axes[0].set_title("Yield Curve: Par, Spot, and Forward"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(maturities, discount_factors, 'purple', linewidth=2.5)
axes[1].set_xlabel("Maturity (years)"); axes[1].set_ylabel("Discount Factor P(0,T)")
axes[1].set_title("Discount Factors"); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("yield_curve.png", dpi=150)
plt.show()
```

### Step 3: Nelson-Siegel Yield Curve Fitting

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------------------------------------------------ #
# Fit Nelson-Siegel model to observed par yields
# ------------------------------------------------------------------ #

def nelson_siegel(m, beta0, beta1, beta2, lam):
    """
    Nelson-Siegel yield curve.
    m   : maturity (years)
    beta0 : long-run level
    beta1 : slope
    beta2 : curvature
    lam  : decay factor
    """
    x = m / lam
    factor1 = (1 - np.exp(-x)) / x
    factor2 = factor1 - np.exp(-x)
    return beta0 + beta1 * factor1 + beta2 * factor2

def svensson(m, b0, b1, b2, b3, l1, l2):
    """Svensson extension with second hump."""
    x1 = m / l1; x2 = m / l2
    f1 = (1 - np.exp(-x1)) / x1
    f2_a = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return b0 + b1*f1 + b2*f2_a + b3*f3

# Observed yields (similar to US Treasury, 2023 inverted curve)
mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
yields_obs = np.array([5.30, 5.25, 5.10, 4.85, 4.65, 4.50, 4.40, 4.35, 4.50, 4.45]) / 100

# Fit Nelson-Siegel
def ns_loss(params):
    b0, b1, b2, lam = params
    if lam <= 0 or b0 <= 0:
        return 1e10
    y_fit = nelson_siegel(mats, b0, b1, b2, lam)
    return np.sum((y_fit - yields_obs)**2)

result = minimize(ns_loss, x0=[0.045, -0.01, -0.02, 2.0],
                  method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-9})
b0, b1, b2, lam = result.x
print(f"Nelson-Siegel parameters:")
print(f"  β₀ (level)    = {b0*100:.4f}%")
print(f"  β₁ (slope)    = {b1*100:.4f}%")
print(f"  β₂ (curvature)= {b2*100:.4f}%")
print(f"  λ  (decay)    = {lam:.4f} years")

# In-sample fit quality
y_ns = nelson_siegel(mats, b0, b1, b2, lam)
rmse = np.sqrt(np.mean((y_ns - yields_obs)**2)) * 10000
print(f"RMSE: {rmse:.2f} bps")

# Plot
m_dense = np.linspace(0.25, 30, 200)
y_dense = nelson_siegel(m_dense, b0, b1, b2, lam)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mats, yields_obs * 100, 'ko', markersize=8, label='Observed', zorder=5)
ax.plot(m_dense, y_dense * 100, 'r-', linewidth=2.5, label=f'Nelson-Siegel (RMSE={rmse:.1f}bps)')
ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Yield (%)")
ax.set_title("Nelson-Siegel Yield Curve Fitting")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("nelson_siegel.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Credit Spread and CDS Pricing

```python
import numpy as np

# ------------------------------------------------------------------ #
# CDS pricing: fair spread given hazard rate and recovery assumption
# ------------------------------------------------------------------ #

def cds_spread(lambda_h, R, T, dt=0.25):
    """
    Compute par CDS spread (annualized).

    Parameters
    ----------
    lambda_h : float — constant hazard rate (default intensity)
    R : float — recovery rate (e.g., 0.40)
    T : float — CDS maturity (years)
    dt : float — payment frequency (0.25 = quarterly)
    """
    times = np.arange(dt, T + dt, dt)
    # Survival probability: Q(t) = exp(-lambda_h * t)
    Q = np.exp(-lambda_h * times)
    # Risk-free discount: assume flat risk-free at 4%
    r = 0.04
    Z = np.exp(-r * times)

    # Premium leg: sum of Q(t_i) * Z(t_i) * dt
    pv_premium_leg = np.sum(Q * Z * dt)

    # Protection leg: (1-R) * integral of Z(t) * lambda_h * Q(t)
    # Approximation: sum over dt intervals
    pv_protection_leg = (1 - R) * lambda_h * np.sum(Z * Q * dt)

    return pv_protection_leg / pv_premium_leg

# Investment grade vs. high yield
for name, hazard, recovery in [
    ("AAA corporate", 0.0020, 0.40),
    ("BBB corporate", 0.0100, 0.40),
    ("BB  high-yield", 0.0300, 0.40),
    ("B   high-yield", 0.0600, 0.35),
    ("CCC distressed", 0.1500, 0.25),
]:
    spread_5y = cds_spread(hazard, recovery, T=5) * 10000  # bps
    print(f"{name:20s}: hazard={hazard*100:.2f}%/yr  CDS spread={spread_5y:.0f} bps")
```

---

## Troubleshooting

### YTM solver fails (`brentq` ValueError)

**Cause**: Price is outside the valid range (too high/low for bounds).

**Fix**:
```python
# Widen search bounds or check price
def safe_ytm(price, face, coupon, maturity):
    try:
        return brentq(lambda y: bond_price(face, coupon, y, maturity) - price,
                      -0.99, 50.0, xtol=1e-10)
    except ValueError as e:
        print(f"YTM search failed: {e}")
        return np.nan
```

### Nelson-Siegel fit doesn't converge

```python
# Try multiple starting values
best = np.inf
best_params = None
for b0_0 in [0.03, 0.05, 0.07]:
    for lam_0 in [1, 2, 5]:
        res = minimize(ns_loss, x0=[b0_0, -0.01, 0.0, lam_0], method='Nelder-Mead')
        if res.fun < best:
            best = res.fun; best_params = res.x
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| scipy | 1.11, 1.12 | `brentq` and `minimize` stable |
| numpy | 1.24, 1.26 | No known issues |

---

## External Resources

### Official Documentation

- [scipy.optimize.brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)

### Key Papers / Textbooks

- Fabozzi, F.J. (2012). *Bond Markets, Analysis and Strategies* (8th ed.). Pearson.
- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious modeling of yield curves*. Journal of Business.

---

## Examples

### Example 1: Immunization — Duration Matching

```python
import numpy as np
from scipy.optimize import minimize

# Match duration of asset portfolio to liability duration
liability_pv = 1_000_000   # $1M liability in 7 years
liability_dur = 7.0         # years (zero-coupon liability)

# Available bonds: 2Y and 10Y
bond_2y  = {'price': 960,  'duration': 1.90}
bond_10y = {'price': 900,  'duration': 7.50}

# Solve: w * dur_2 + (1-w) * dur_10 = liability_dur
# w * (dur_2 - dur_10) = liability_dur - dur_10
w = (liability_dur - bond_10y['duration']) / (bond_2y['duration'] - bond_10y['duration'])
print(f"Weight in 2Y bond: {w:.4f}  ({w*100:.1f}%)")
print(f"Weight in 10Y bond: {1-w:.4f}  ({(1-w)*100:.1f}%)")

pv_2y  = w * liability_pv
pv_10y = (1-w) * liability_pv
n_2y  = pv_2y  / bond_2y['price']
n_10y = pv_10y / bond_10y['price']
print(f"Units of 2Y bond to buy:  {n_2y:.0f}")
print(f"Units of 10Y bond to buy: {n_10y:.0f}")
```

### Example 2: Key Rate Duration

```python
import numpy as np

def key_rate_duration(bond, key_mats, bump=0.0001):
    """Compute key rate durations by bumping individual maturity points."""
    krds = {}
    for mat in key_mats:
        # Bump yield at that maturity ±1bp
        price_up   = bond.price_at_yield(bond.ytm + bump)  # simplified
        price_down = bond.price_at_yield(bond.ytm - bump)
        krd = -(price_up - price_down) / (2 * bump * bond.price)
        krds[f"{mat}Y"] = krd
    return krds

# For illustration with our Bond class
bond = Bond(1000, 0.05, 10, ytm=0.05)
# Single-factor KRD approximation (full KRD requires multi-factor spot curve)
print(f"Modified Duration (proxy for total KRD): {bond.modified_duration:.4f}")
print(f"Total DV01: ${bond.dv01:.4f} per $1000 face")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
