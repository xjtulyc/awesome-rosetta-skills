---
name: housing-market
description: Housing market analysis with hedonic pricing, spatial autocorrelation, affordability metrics, and repeat-sales index construction for urban research.
tags:
  - housing-economics
  - hedonic-regression
  - real-estate
  - spatial-analysis
  - affordability
version: "1.0.0"
authors:
  - "@xjtulyc"
license: MIT
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
    - scikit-learn>=1.3
    - matplotlib>=3.7
    - geopandas>=0.14
last_updated: "2026-03-17"
status: stable
---

# Housing Market Analysis

## When to Use This Skill

Use this skill when you need to:
- Estimate implicit prices of housing attributes (hedonic regression)
- Construct repeat-sales house price indices (Case-Shiller method)
- Measure housing affordability and price-to-income ratios
- Detect spatial clustering of housing prices (Moran's I, LISA)
- Forecast house prices using machine learning and spatial lag models
- Analyze rental yields and capitalization rates
- Study filtering theory and housing supply/demand dynamics

**Trigger keywords**: hedonic pricing, house price index, repeat sales, affordability ratio, housing bubble, capitalization rate, rental yield, spatial lag model, housing supply, zoning analysis, price-to-rent, mortgage payment, housing filtering, location gradient, CBD distance gradient, amenity valuation.

## Background & Key Concepts

### Hedonic Price Model

Houses are bundles of attributes. The hedonic regression decomposes price $P$ into contributions of individual characteristics:

$$\ln P_i = \alpha + \sum_k \beta_k X_{ik} + \varepsilon_i$$

where $X_{ik}$ includes structural attributes (sqft, bedrooms, age), location variables (distance to CBD, school quality), and neighborhood characteristics. The implicit price of attribute $k$ is $\partial P/\partial X_k = \beta_k \cdot P$.

### Repeat-Sales Index

For two sales of property $i$ at times $s$ and $t$:

$$\ln P_{it} - \ln P_{is} = \sum_{\tau=s+1}^{t} \delta_\tau D_\tau + \varepsilon_{it}$$

where $D_\tau$ are time dummies and $\delta_\tau$ estimates log price changes. The Case-Shiller weighted repeat-sales accounts for variance growing with holding period.

### Affordability Measures

- **Price-to-Income Ratio (PIR)**: $\text{PIR} = P / Y_{\text{median}}$
- **Housing Affordability Index (HAI)**: $\text{HAI} = \frac{\text{Qualifying Income}}{Y_{\text{median}}} \times 100$, where qualifying income = monthly payment × 12 / 0.28
- **Residual Income**: $I - \text{Housing Cost} - \text{Transport Cost}$ (location efficiency)

### Spatial Hedonic Model

$$\ln \mathbf{P} = \rho \mathbf{W} \ln \mathbf{P} + \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

The spatial autoregressive (SAR) model accounts for spatial spillovers: $\rho$ captures the degree to which nearby prices influence each other.

### Von Thünen / Bid-Rent Gradient

$$P(d) = P_0 \exp(-\gamma d)$$

Semilog regression of $\ln P$ on distance $d$ from CBD gives gradient $\gamma$ (price decay per km).

## Environment Setup

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 statsmodels>=0.14 \
            scikit-learn>=1.3 matplotlib>=3.7 geopandas>=0.14
```

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
print("Housing market analysis environment ready")
```

## Core Workflow

### Step 1: Hedonic Regression with Spatial Effects

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate hedonic dataset: 1000 transactions
# -----------------------------------------------------------------
np.random.seed(42)
n = 1000

# Location: x, y coordinates in a 20km x 20km city
x = np.random.uniform(0, 20, n)
y = np.random.uniform(0, 20, n)
# Distance to CBD at (10, 10)
cbd_dist = np.sqrt((x - 10)**2 + (y - 10)**2)

# Structural attributes
sqft = np.random.lognormal(7.0, 0.4, n)           # 330-2700 sqft
bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.35, 0.2, 0.1])
age = np.random.randint(0, 80, n).astype(float)
garage = np.random.binomial(1, 0.6, n)
school_rating = np.random.uniform(3, 10, n)

# True hedonic model (log-linear)
ln_P = (
    11.0                           # intercept ~ $60k baseline
    + 0.7 * np.log(sqft)          # size elasticity 0.7
    + 0.05 * bedrooms              # each bedroom +5%
    - 0.005 * age                  # depreciation
    + 0.10 * garage                # garage premium
    + 0.06 * school_rating         # school quality
    - 0.04 * cbd_dist              # CBD gradient: -4%/km
    + np.random.normal(0, 0.12, n) # unexplained
)
price = np.exp(ln_P)

df = pd.DataFrame({
    "price": price, "sqft": sqft, "bedrooms": bedrooms,
    "age": age, "garage": garage, "school_rating": school_rating,
    "cbd_dist": cbd_dist, "x": x, "y": y
})

# -----------------------------------------------------------------
# OLS hedonic regression
# -----------------------------------------------------------------
df["ln_price"] = np.log(df["price"])
df["ln_sqft"] = np.log(df["sqft"])
features = ["ln_sqft", "bedrooms", "age", "garage", "school_rating", "cbd_dist"]
X_reg = sm.add_constant(df[features])

ols = sm.OLS(df["ln_price"], X_reg).fit(cov_type="HC3")
print("=== OLS Hedonic Regression ===")
print(ols.summary().tables[1])
print(f"\nAdjusted R²: {ols.rsquared_adj:.4f}")

# Marginal implicit prices at median house
median_price = df["price"].median()
print(f"\nMedian house price: ${median_price:,.0f}")
print(f"Implicit price of 1 extra sqft: ${ols.params['ln_sqft'] * median_price / df['sqft'].median():,.0f}")
print(f"CBD gradient: {ols.params['cbd_dist']*100:.1f}% per km")

# -----------------------------------------------------------------
# Bid-rent gradient visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Price vs CBD distance
axes[0].scatter(df["cbd_dist"], df["ln_price"], alpha=0.3, s=5, c="steelblue")
d_line = np.linspace(0, 14, 100)
axes[0].plot(d_line, ols.params["const"] + ols.params["cbd_dist"] * d_line +
             ols.params["ln_sqft"] * np.log(df["sqft"].median()) +
             ols.params["bedrooms"] * 3 + ols.params["school_rating"] * 7,
             "r-", lw=2, label="Bid-rent gradient")
axes[0].set_xlabel("Distance from CBD (km)")
axes[0].set_ylabel("ln(Price)")
axes[0].set_title("Bid-Rent Gradient")
axes[0].legend()

# Price heatmap on city grid
from scipy.interpolate import griddata
xi = np.linspace(0, 20, 100)
yi = np.linspace(0, 20, 100)
Xi, Yi = np.meshgrid(xi, yi)
Pi = griddata((df["x"], df["y"]), df["price"], (Xi, Yi), method="linear")
im = axes[1].contourf(Xi, Yi, Pi / 1e6, levels=20, cmap="RdYlGn_r")
axes[1].plot(10, 10, "k*", ms=15, label="CBD")
axes[1].set_title("Price Surface ($ million)")
plt.colorbar(im, ax=axes[1])
axes[1].legend()

# Residual spatial pattern
residuals = ols.resid
sc = axes[2].scatter(df["x"], df["y"], c=residuals, cmap="RdBu_r",
                     vmin=-0.3, vmax=0.3, s=10)
axes[2].plot(10, 10, "k*", ms=15)
axes[2].set_title("OLS Residuals (Spatial Pattern?)")
plt.colorbar(sc, ax=axes[2], label="Residual")

plt.tight_layout()
plt.savefig("hedonic_regression.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: hedonic_regression.png")
```

### Step 2: Repeat-Sales House Price Index

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate repeat sales dataset
# 500 properties, each sold 2-3 times over 10 years (40 quarters)
# -----------------------------------------------------------------
np.random.seed(42)
n_properties = 500
n_quarters = 40

# True price index (simulates a housing boom-bust cycle)
true_index = np.zeros(n_quarters)
true_index[:20] = np.linspace(0, 0.8, 20)   # 80% appreciation over 5 years
true_index[20:30] = np.linspace(0.8, 0.5, 10)  # 30% crash
true_index[30:] = np.linspace(0.5, 0.7, 10)    # partial recovery

records = []
for prop_id in range(n_properties):
    # Two sales: first sale in 0-19, second in 10-39
    t1 = np.random.randint(0, 25)
    t2 = np.random.randint(t1 + 4, n_quarters)
    base_ln_price = np.random.normal(12.5, 0.5)  # property fixed effect

    # Prices at each sale time
    noise1 = np.random.normal(0, 0.08)
    noise2 = np.random.normal(0, 0.08)
    ln_p1 = base_ln_price + true_index[t1] + noise1
    ln_p2 = base_ln_price + true_index[t2] + noise2
    records.append({"prop_id": prop_id, "t": t1, "ln_price": ln_p1})
    records.append({"prop_id": prop_id, "t": t2, "ln_price": ln_p2})

sales_df = pd.DataFrame(records).sort_values(["prop_id", "t"])

# -----------------------------------------------------------------
# Construct repeat-sales pairs
# -----------------------------------------------------------------
pairs = []
for prop_id, grp in sales_df.groupby("prop_id"):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp) - 1):
        pairs.append({
            "prop_id": prop_id,
            "t_sell": int(grp.loc[i+1, "t"]),
            "t_buy": int(grp.loc[i, "t"]),
            "ln_price_diff": grp.loc[i+1, "ln_price"] - grp.loc[i, "ln_price"]
        })
pairs_df = pd.DataFrame(pairs)
print(f"Repeat-sale pairs: {len(pairs_df)}")

# -----------------------------------------------------------------
# Case-Shiller OLS repeat-sales regression
# Time dummies: omit period 0 as base
# -----------------------------------------------------------------
periods = list(range(1, n_quarters))  # periods 1..39

def build_repeat_sales_matrix(pairs_df, periods):
    """Build design matrix for repeat-sales regression.

    Each row: ln(P_sell) - ln(P_buy) = sum of time dummies between buy and sell.
    Returns X (design matrix) and y (price changes).
    """
    n = len(pairs_df)
    X = np.zeros((n, len(periods)))
    y = pairs_df["ln_price_diff"].values.copy()

    period_map = {p: i for i, p in enumerate(periods)}
    for row_idx, row in enumerate(pairs_df.itertuples()):
        for tau in range(row.t_buy + 1, row.t_sell + 1):
            if tau in period_map:
                X[row_idx, period_map[tau]] = 1
    return X, y

X_rs, y_rs = build_repeat_sales_matrix(pairs_df, periods)

# OLS (no constant — natural normalization: index[0] = 0)
result_rs = sm.OLS(y_rs, X_rs).fit()
index_ols = np.concatenate([[0.0], result_rs.params])  # prepend 0 for period 0

print(f"R² of repeat-sales regression: {result_rs.rsquared:.4f}")
print(f"Peak index level: {index_ols.max():.3f} at period {np.argmax(index_ols)}")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(true_index, "b--", label="True Index", lw=2)
axes[0].plot(index_ols, "r-", label="OLS Repeat-Sales", lw=2)
axes[0].set_xlabel("Quarter")
axes[0].set_ylabel("Log Price Change (base=0)")
axes[0].set_title("Repeat-Sales House Price Index")
axes[0].legend()
axes[0].axhline(0, color="gray", lw=0.8)

# Distribution of holding period lengths
holding = pairs_df["t_sell"] - pairs_df["t_buy"]
axes[1].hist(holding, bins=20, color="steelblue", edgecolor="black")
axes[1].set_xlabel("Holding Period (quarters)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Holding Periods")

plt.tight_layout()
plt.savefig("repeat_sales_index.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: repeat_sales_index.png")
```

### Step 3: Housing Affordability and Supply Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------------------------------------------
# Simulate 20-city panel: 2000-2023 (annual)
# -----------------------------------------------------------------
np.random.seed(42)
n_cities = 20
n_years = 24
years = np.arange(2000, 2024)
city_names = [f"City_{i:02d}" for i in range(n_cities)]

# City characteristics
median_income_2000 = np.random.uniform(40000, 80000, n_cities)
# Income growth (varying by city)
income_growth = np.random.uniform(0.01, 0.04, n_cities)
# House price growth (some cities have higher appreciation)
price_growth = income_growth + np.random.uniform(-0.01, 0.04, n_cities)
# Housing supply elasticity (Saiz-style: constrained vs unconstrained)
supply_elast = np.random.uniform(0.5, 4.0, n_cities)

# Generate panel
records = []
for c in range(n_cities):
    for t, yr in enumerate(years):
        income = median_income_2000[c] * (1 + income_growth[c]) ** t
        price = 200000 * (1 + price_growth[c]) ** t * (1 + np.random.normal(0, 0.02))
        rent = price * 0.055 / 12 * (1 + np.random.normal(0, 0.02))  # ~5.5% cap rate
        records.append({
            "city": city_names[c],
            "year": yr,
            "median_income": income,
            "median_price": price,
            "median_rent": rent,
            "supply_elasticity": supply_elast[c],
        })

panel = pd.DataFrame(records)

# -----------------------------------------------------------------
# Affordability measures
# -----------------------------------------------------------------
INTEREST_RATE = 0.06  # 6% 30-year fixed
LOAN_TO_VALUE = 0.80
N_MONTHS = 360

def monthly_payment(price, rate=INTEREST_RATE, ltv=LOAN_TO_VALUE, n=N_MONTHS):
    """Compute monthly mortgage payment (principal + interest)."""
    loan = price * ltv
    r = rate / 12
    return loan * r * (1 + r)**n / ((1 + r)**n - 1)

panel["monthly_pmt"] = panel["median_price"].apply(monthly_payment)
panel["annual_pmt"] = panel["monthly_pmt"] * 12
panel["PIR"] = panel["median_price"] / panel["median_income"]
panel["HAI"] = 100 * (panel["median_income"] / (panel["annual_pmt"] / 0.28))
panel["price_to_rent"] = panel["median_price"] / (panel["median_rent"] * 12)
panel["rent_burden"] = panel["median_rent"] / (panel["median_income"] / 12)

# Summary statistics
latest = panel[panel["year"] == 2023].copy()
print("=== 2023 Housing Affordability Summary ===")
print(latest[["city", "PIR", "HAI", "price_to_rent", "rent_burden"]].describe().round(2))

affordable = (latest["HAI"] >= 100).sum()
print(f"\nAffordable cities (HAI ≥ 100): {affordable}/{n_cities}")

# -----------------------------------------------------------------
# Supply elasticity and price growth relationship
# -----------------------------------------------------------------
price_growth_obs = latest.copy()
price_growth_obs["total_appreciation"] = (
    panel.groupby("city").apply(
        lambda g: (g.sort_values("year")["median_price"].iloc[-1] /
                   g.sort_values("year")["median_price"].iloc[0])
    ).values
)

corr, pval = stats.pearsonr(price_growth_obs["supply_elasticity"],
                             np.log(price_growth_obs["total_appreciation"]))
print(f"\nCorrelation(supply elasticity, ln price appreciation): r = {corr:.3f}, p = {pval:.3f}")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# PIR over time by city
pivot_pir = panel.pivot(index="year", columns="city", values="PIR")
for col in pivot_pir.columns:
    axes[0, 0].plot(pivot_pir.index, pivot_pir[col], alpha=0.4, lw=0.8, color="steelblue")
axes[0, 0].plot(pivot_pir.index, pivot_pir.mean(axis=1), "r-", lw=2, label="Mean PIR")
axes[0, 0].axhline(3.0, color="orange", ls="--", label="PIR=3 (affordable)")
axes[0, 0].set_title("Price-to-Income Ratio Over Time")
axes[0, 0].set_xlabel("Year"); axes[0, 0].set_ylabel("PIR")
axes[0, 0].legend()

# Housing Affordability Index
pivot_hai = panel.pivot(index="year", columns="city", values="HAI")
for col in pivot_hai.columns:
    axes[0, 1].plot(pivot_hai.index, pivot_hai[col], alpha=0.4, lw=0.8, color="green")
axes[0, 1].plot(pivot_hai.index, pivot_hai.mean(axis=1), "k-", lw=2, label="Mean HAI")
axes[0, 1].axhline(100, color="red", ls="--", label="HAI=100 (threshold)")
axes[0, 1].set_title("Housing Affordability Index")
axes[0, 1].set_xlabel("Year"); axes[0, 1].set_ylabel("HAI")
axes[0, 1].legend()

# Supply elasticity vs price growth
axes[1, 0].scatter(price_growth_obs["supply_elasticity"],
                   price_growth_obs["total_appreciation"],
                   c="steelblue", edgecolors="k", s=60)
axes[1, 0].set_xlabel("Housing Supply Elasticity")
axes[1, 0].set_ylabel("Total Price Appreciation (2000-2023)")
axes[1, 0].set_title(f"Supply Constraint & Price Growth\n(r={corr:.2f}, p={pval:.3f})")
# Fit line
m, b = np.polyfit(price_growth_obs["supply_elasticity"],
                  price_growth_obs["total_appreciation"], 1)
xf = np.linspace(price_growth_obs["supply_elasticity"].min(),
                 price_growth_obs["supply_elasticity"].max(), 100)
axes[1, 0].plot(xf, m*xf + b, "r--")

# 2023 affordability distribution
axes[1, 1].hist(latest["PIR"], bins=10, color="coral", edgecolor="black", alpha=0.7,
                label="PIR")
axes[1, 1].axvline(latest["PIR"].mean(), color="red", ls="--",
                   label=f"Mean PIR={latest['PIR'].mean():.1f}")
axes[1, 1].set_title("2023 PIR Distribution Across Cities")
axes[1, 1].set_xlabel("Price-to-Income Ratio")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("housing_affordability.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: housing_affordability.png")
```

## Advanced Usage

### Spatial Lag Model for House Prices

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import cdist

def spatial_weights_knn(coords, k=5):
    """Build row-standardized k-nearest neighbor spatial weight matrix.

    Args:
        coords: (n, 2) array of coordinates
        k: number of neighbors
    Returns:
        W: (n, n) row-standardized weight matrix
    """
    n = len(coords)
    dist = cdist(coords, coords)
    np.fill_diagonal(dist, np.inf)
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist[i])[:k]
        W[i, neighbors] = 1
    W = W / W.sum(axis=1, keepdims=True)  # row standardize
    return W

def spatial_lag_ols(y, X, W, max_iter=50, tol=1e-6):
    """Two-stage least squares estimator for spatial lag model.

    Model: y = rho * W*y + X*beta + epsilon
    Instruments: X, W*X for endogenous W*y term.

    Args:
        y: (n,) dependent variable
        X: (n, k) exogenous regressors (with constant)
        W: (n, n) spatial weight matrix
        max_iter, tol: convergence criteria
    Returns:
        dict with rho, beta, residuals, fitted
    """
    Wy = W @ y
    WX = W @ X

    # 2SLS: instrument Wy with WX
    instruments = np.hstack([X, WX])
    Z_hat = instruments @ np.linalg.pinv(instruments) @ np.column_stack([X, Wy])
    beta_2sls = np.linalg.lstsq(Z_hat, y, rcond=None)[0]

    rho = beta_2sls[-1]
    beta = beta_2sls[:-1]
    fitted = rho * Wy + X @ beta
    residuals = y - fitted
    return {"rho": rho, "beta": beta, "residuals": residuals, "fitted": fitted}

np.random.seed(42)
n_h = 400
coords_h = np.random.uniform(0, 20, (n_h, 2))
W_h = spatial_weights_knn(coords_h, k=8)

cbd_d = np.sqrt((coords_h[:, 0] - 10)**2 + (coords_h[:, 1] - 10)**2)
sqft_h = np.random.lognormal(7, 0.3, n_h)
# True DGP includes spatial lag
rho_true = 0.4
X_h = np.column_stack([np.ones(n_h), np.log(sqft_h), cbd_d])
eps_h = np.random.normal(0, 0.1, n_h)
ln_p_h = np.linalg.solve(np.eye(n_h) - rho_true * W_h,
                          X_h @ np.array([11, 0.7, -0.04]) + eps_h)

result_sl = spatial_lag_ols(ln_p_h, X_h, W_h)
print("=== Spatial Lag Model (2SLS) ===")
print(f"Spatial autoregressive parameter ρ = {result_sl['rho']:.4f} (true: {rho_true})")
print(f"Coefficients: {result_sl['beta'].round(4)}")
```

### Capitalization Rate and Yield Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_yields(purchase_price, annual_rent, operating_expenses_ratio=0.40,
                   appreciation_rate=0.03, hold_years=5, discount_rate=0.08):
    """Compute key real estate investment metrics.

    Args:
        purchase_price: acquisition price
        annual_rent: gross annual rent
        operating_expenses_ratio: fraction of rent for expenses (NOI/rent = 1-OER)
        appreciation_rate: annual property appreciation
        hold_years: investment horizon in years
        discount_rate: required return for NPV
    Returns:
        dict of metrics
    """
    noi = annual_rent * (1 - operating_expenses_ratio)
    cap_rate = noi / purchase_price

    gross_yield = annual_rent / purchase_price
    net_yield = noi / purchase_price

    # Cash flows assuming NOI grows at appreciation rate
    cash_flows = [-purchase_price]
    for t in range(1, hold_years + 1):
        noi_t = noi * (1 + appreciation_rate) ** (t - 1)
        if t < hold_years:
            cash_flows.append(noi_t)
        else:
            # Terminal sale: resale price based on terminal cap rate
            terminal_noi = noi * (1 + appreciation_rate) ** t
            resale_price = terminal_noi / cap_rate
            cash_flows.append(noi_t + resale_price)

    # NPV and IRR
    npv = sum(cf / (1 + discount_rate)**t for t, cf in enumerate(cash_flows))
    # IRR via Newton's method
    irr = discount_rate
    for _ in range(200):
        npv_irr = sum(cf / (1 + irr)**t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + irr)**(t+1) for t, cf in enumerate(cash_flows))
        if abs(dnpv) < 1e-10:
            break
        irr -= npv_irr / dnpv

    return {
        "cap_rate": cap_rate,
        "gross_yield": gross_yield,
        "net_yield": net_yield,
        "npv": npv,
        "irr": irr,
        "price_to_rent": purchase_price / annual_rent,
    }

metrics = compute_yields(500000, 24000, operating_expenses_ratio=0.35, discount_rate=0.07)
print("=== Investment Property Metrics ===")
for k, v in metrics.items():
    if k in ("cap_rate", "gross_yield", "net_yield", "irr"):
        print(f"  {k}: {v*100:.2f}%")
    elif k == "npv":
        print(f"  {k}: ${v:,.0f}")
    else:
        print(f"  {k}: {v:.1f}")
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Negative fitted prices in linear model | Outliers pulling fit | Use log-linear specification |
| Multicollinearity (sqft ↔ bedrooms) | Structural correlation | Check VIF; consider sqft per room instead |
| Spatial autocorrelation in residuals | Omitted spatial effects | Add spatial lag or spatial error term |
| Repeat-sales singular matrix | Too few observations per period | Aggregate to quarterly from monthly |
| IRR computation diverges | No sign change in cash flows | Check if investment is ever profitable |
| HAI > 200 (implausibly high) | Low interest rate scenario | Verify mortgage rate inputs |

## External Resources

- Rosen, S. (1974). Hedonic prices and implicit markets. *Journal of Political Economy*, 82(1).
- Case, K. E., & Shiller, R. J. (1989). The efficiency of the market for single-family homes. *AER*, 79(1).
- Saiz, A. (2010). The geographic determinants of housing supply. *QJE*, 125(3).
- Anselin, L. (1988). *Spatial Econometrics*. Kluwer Academic.
- [S&P Case-Shiller Index methodology](https://www.spglobal.com/spdji/en/indices/indicators/sp-corelogic-case-shiller-us-national-home-price-nsa-index/)
- [Zillow Research Data](https://www.zillow.com/research/data/) — ZHVI, ZRI

## Examples

### Example 1: Neighborhood Price Gradient by Amenity

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)
n = 800
x = np.random.uniform(0, 15, n)
y = np.random.uniform(0, 15, n)

# Two amenities: park at (3, 8) and school at (12, 5)
park_dist = np.sqrt((x - 3)**2 + (y - 8)**2)
school_dist = np.sqrt((x - 12)**2 + (y - 5)**2)
cbd_dist = np.sqrt((x - 7.5)**2 + (y - 7.5)**2)

sqft = np.random.lognormal(6.8, 0.35, n)

ln_p = (11.5
        + 0.65 * np.log(sqft)
        - 0.035 * cbd_dist
        - 0.045 * park_dist         # park premium: $45k per km closer
        - 0.020 * school_dist
        + np.random.normal(0, 0.10, n))

df = pd.DataFrame({"ln_p": ln_p, "ln_sqft": np.log(sqft),
                   "cbd_dist": cbd_dist, "park_dist": park_dist,
                   "school_dist": school_dist})

X = sm.add_constant(df[["ln_sqft", "cbd_dist", "park_dist", "school_dist"]])
res = sm.OLS(df["ln_p"], X).fit(cov_type="HC3")
med_p = np.exp(df["ln_p"].mean())

print("=== Amenity Valuation ===")
print(f"Park premium per 1km closer: ${abs(res.params['park_dist']) * med_p:,.0f}")
print(f"School proximity: ${abs(res.params['school_dist']) * med_p:,.0f} per km")
print(f"CBD gradient: {res.params['cbd_dist']*100:.2f}% per km")
```

### Example 2: Housing Market Bubble Detection (Price-to-Rent Ratio)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(2024)
quarters = pd.date_range("2000Q1", "2023Q4", freq="QE")
n_q = len(quarters)

# Simulate price and rent indices (rent grows more smoothly)
rent_idx = 100 * (1.025 ** (np.arange(n_q) / 4))  # 2.5%/yr
# Price: boom-bust cycle
price_growth = np.concatenate([
    np.linspace(0, 0.10, 24),   # Q1 2000 - Q4 2005 boom
    np.linspace(0.10, 0.02, 12),  # Q1 2006 - Q4 2008 peak/bust
    np.linspace(0.02, -0.05, 8),  # Q1 2009 - Q4 2010 trough
    np.linspace(-0.05, 0.05, 24), # recovery
    np.linspace(0.05, 0.12, n_q - 68),  # 2017-2023 new boom
])[:n_q]
price_idx = 100 * np.cumprod(1 + price_growth / 4)

ptr = price_idx / rent_idx * (rent_idx[0] / price_idx[0])  # normalized P/R ratio

# Hodrick-Prescott trend (approximate with polynomial)
from numpy.polynomial import polynomial as P
t = np.arange(n_q)
coeffs = P.polyfit(t, ptr, 3)
ptr_trend = P.polyval(t, coeffs)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(quarters, ptr, "steelblue", label="Price-to-Rent Ratio")
ax.plot(quarters, ptr_trend, "r--", label="Trend (cubic)")
ax.axhline(1.0, color="gray", ls=":", label="Baseline")
overvalued = ptr > ptr_trend * 1.10
ax.fill_between(quarters, ptr, ptr_trend,
                where=overvalued, alpha=0.3, color="red", label="Overvalued >10%")
ax.set_title("Housing Market Bubble Detection via Price-to-Rent Ratio")
ax.set_ylabel("Normalized P/R Ratio")
ax.legend()
plt.tight_layout()
plt.savefig("bubble_detection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Overvalued quarters: {overvalued.sum()}/{n_q}")
```
