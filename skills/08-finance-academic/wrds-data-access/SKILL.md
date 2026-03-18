---
name: wrds-data-access
description: >
  Use this Skill to access WRDS databases: Compustat fundamentals, CRSP
  returns, TAQ microstructure, and linking tables via wrds Python package.
tags:
  - finance
  - wrds
  - compustat
  - crsp
  - data-access
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
    - wrds>=3.1
    - pandas>=2.0
    - numpy>=1.24
    - sqlalchemy>=2.0
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# WRDS Data Access for Academic Finance Research

> **One-line summary**: Query Compustat, CRSP, and TAQ from WRDS using the Python API: financial fundamentals, stock returns, intraday microstructure, and CRSP-Compustat link table.

---

## When to Use This Skill

- When downloading Compustat balance sheet / income statement data
- When fetching CRSP monthly or daily stock returns and market cap
- When linking Compustat (gvkey) to CRSP (permno) via CCM link table
- When extracting TAQ intraday trades and quotes for microstructure research
- When building research-grade datasets (accounting ratios, anomalies)
- When accessing I/B/E/S earnings forecasts or Execucomp compensation

**Trigger keywords**: WRDS, Compustat, CRSP, CCM, TAQ, gvkey, permno, cusip, annual fundamentals, stock returns, book-to-market, earnings surprise, I/B/E/S, Execucomp, institutional holdings

---

## Background & Key Concepts

### WRDS Architecture

WRDS (Wharton Research Data Services) exposes PostgreSQL databases via:

1. **Python package** (`wrds`): Connects via SSH tunnel + PostgreSQL; requires institutional subscription
2. **SQL queries**: Standard SQL on schemas like `comp` (Compustat), `crsp`, `taq`

### Key Identifiers

| Identifier | Database | Description |
|:-----------|:---------|:-----------|
| `gvkey` | Compustat | Permanent company ID |
| `permno` | CRSP | Permanent security ID |
| `cusip` | Both | Committee on Uniform Securities |
| `ticker` | Multiple | Not permanent — changes |

### CRSP-Compustat Merge (CCM)

The `crsp.ccmxpf_lnkhist` table maps `gvkey` → `permno` with date-valid links. Always filter by `linktype IN ('LC','LU','LX','LD')` and `linkprim IN ('P','C')`.

---

## Environment Setup

### Install Dependencies

```bash
pip install wrds>=3.1 pandas>=2.0 numpy>=1.24 sqlalchemy>=2.0 matplotlib>=3.7
```

### WRDS Credentials

```bash
# Credentials are stored securely in ~/.pgpass after first login
# Set environment variables (do NOT hardcode)
export WRDS_USERNAME="<your-wrds-username>"
```

### Connect to WRDS

```python
import os
import wrds

# Credentials via environment variable
wrds_username = os.getenv("WRDS_USERNAME", "")
if not wrds_username:
    raise ValueError("Set WRDS_USERNAME environment variable")

db = wrds.Connection(wrds_username=wrds_username)
print("Connected to WRDS")

# List available libraries
libs = db.list_libraries()
print("Available libraries (first 10):", libs[:10])
```

---

## Core Workflow

### Step 1: Compustat Annual Fundamentals

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Download Compustat annual fundamentals for S&P 500 firms
# Key variables for accounting research
# ------------------------------------------------------------------ #

try:
    import wrds
    db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME", ""))

    # Query Compustat annual (comp.funda)
    query = """
        SELECT
            gvkey, datadate, fyear, conm, sich,
            at,         -- Total assets
            lt,         -- Total liabilities
            ceq,        -- Common equity (book value)
            ni,         -- Net income
            sale,       -- Net sales
            oibdp,      -- Operating income before depreciation
            dp,         -- Depreciation & amortization
            capx,       -- Capital expenditure
            che,        -- Cash and equivalents
            dltt,       -- Long-term debt
            dlc,        -- Short-term debt
            csho,       -- Common shares outstanding
            prcc_f,     -- Fiscal-year-end price
            sich        -- SIC code
        FROM comp.funda
        WHERE
            indfmt = 'INDL'       -- Industrial format
            AND datafmt = 'STD'    -- Standardized
            AND popsrc = 'D'       -- Domestic
            AND consol = 'C'       -- Consolidated
            AND fyear BETWEEN 2010 AND 2023
            AND at > 0
            AND sale > 0
        ORDER BY gvkey, datadate
        LIMIT 50000
    """
    comp = db.raw_sql(query, date_cols=['datadate'])
    print(f"Compustat: {comp.shape[0]:,} firm-years downloaded")

except Exception as e:
    print(f"WRDS connection failed ({e}), using synthetic data")
    # Synthetic Compustat-like data
    np.random.seed(42)
    n = 5000
    gvkeys = np.repeat(np.arange(1, 101), 50)[:n]
    fyears = np.tile(np.arange(2010, 2024), 500)[:n]

    comp = pd.DataFrame({
        'gvkey': gvkeys,
        'fyear': fyears,
        'at':    np.random.lognormal(7, 1.5, n),    # Total assets
        'ceq':   np.random.lognormal(6, 1.5, n),    # Book equity
        'ni':    np.random.normal(0.1, 0.3, n) * np.random.lognormal(6, 1.5, n)/10,
        'sale':  np.random.lognormal(6.5, 1.5, n),
        'csho':  np.random.lognormal(4, 1.5, n),
        'prcc_f': np.random.lognormal(3.5, 0.8, n),
        'dltt':  np.random.lognormal(5, 2, n),
        'capx':  np.abs(np.random.normal(0, 0.05, n)) * np.random.lognormal(6.5, 1.5, n),
    })

# ---- Compute key accounting ratios ----------------------------- #
comp = comp.copy()
comp['mve']  = comp['csho'] * comp['prcc_f']                    # Market cap
comp['bm']   = comp['ceq'] / comp['mve'].replace(0, np.nan)      # Book/Market
comp['roa']  = comp['ni'] / comp['at'].replace(0, np.nan)        # ROA
comp['roe']  = comp['ni'] / comp['ceq'].replace(0, np.nan)       # ROE
comp['lev']  = comp['dltt'] / comp['at'].replace(0, np.nan)      # Leverage
comp['asset_growth'] = comp.groupby('gvkey')['at'].pct_change()  # Asset growth

# Winsorize at 1st/99th percentile
def winsorize(series, low=0.01, high=0.99):
    lb = series.quantile(low)
    ub = series.quantile(high)
    return series.clip(lb, ub)

for col in ['bm', 'roa', 'roe', 'lev', 'asset_growth']:
    comp[col] = winsorize(comp[col])

print("\nAccounting ratios summary:")
print(comp[['bm', 'roa', 'roe', 'lev', 'asset_growth']].describe().round(4))

# ---- Portfolio B/M quintiles ----------------------------------- #
latest = comp[comp['fyear'] == comp['fyear'].max()].copy()
latest = latest[latest['mve'] > 0].dropna(subset=['bm'])
latest['bm_quintile'] = pd.qcut(latest['bm'], 5,
                                  labels=['Growth','Q2','Q3','Q4','Value'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROA by BM quintile
roa_by_bm = latest.groupby('bm_quintile')['roa'].mean()
axes[0].bar(roa_by_bm.index, roa_by_bm.values, color='steelblue', edgecolor='black', linewidth=0.7)
axes[0].set_xlabel("B/M Quintile"); axes[0].set_ylabel("Mean ROA")
axes[0].set_title("ROA by Book-to-Market Quintile")
axes[0].grid(axis='y', alpha=0.3)

# Leverage distribution
axes[1].hist(latest['lev'].dropna(), bins=40, color='coral', edgecolor='white', linewidth=0.5)
axes[1].axvline(latest['lev'].median(), color='navy', linewidth=2, linestyle='--',
                label=f"Median={latest['lev'].median():.2f}")
axes[1].set_xlabel("Leverage (LT Debt/Assets)"); axes[1].set_title("Leverage Distribution")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("compustat_analysis.png", dpi=150)
plt.show()
```

### Step 2: CRSP Monthly Returns

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Download CRSP monthly stock file and compute factor portfolio returns
# ------------------------------------------------------------------ #

try:
    import wrds
    import os
    db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME", ""))

    crsp_query = """
        SELECT
            permno, date, ret, retx, shrout, prc, exchcd, shrcd
        FROM crsp.msf
        WHERE
            date BETWEEN '2010-01-01' AND '2023-12-31'
            AND shrcd IN (10, 11)       -- Common shares, US incorporated
            AND exchcd IN (1, 2, 3)     -- NYSE, AMEX, NASDAQ
            AND ret IS NOT NULL
        ORDER BY permno, date
        LIMIT 200000
    """
    crsp = db.raw_sql(crsp_query, date_cols=['date'])
    print(f"CRSP monthly: {crsp.shape[0]:,} obs downloaded")

except Exception as e:
    print(f"Using synthetic CRSP data ({e})")
    np.random.seed(42)
    n = 50000
    permnos = np.repeat(np.arange(1, 1001), 60)[:n]
    dates = pd.bdate_range("2010-01-31", periods=60, freq='BM')
    dates = np.tile(dates[:60], 1000)[:n]

    crsp = pd.DataFrame({
        'permno': permnos,
        'date':   dates,
        'ret':    np.random.normal(0.01, 0.08, n),
        'prc':    np.abs(np.random.lognormal(3.5, 0.8, n)),
        'shrout': np.random.lognormal(8, 1.5, n) * 1000,
        'exchcd': np.random.choice([1, 2, 3], n),
    })

# ---- Market equity and value-weighted returns ----------------- #
crsp['me'] = crsp['prc'].abs() * crsp['shrout'] / 1000  # in thousands

# Value-weighted market return
crsp = crsp.sort_values(['permno', 'date'])
crsp['me_lag'] = crsp.groupby('permno')['me'].shift(1)

def vw_return(group):
    w = group['me_lag'].clip(lower=0)
    if w.sum() == 0:
        return np.nan
    return (group['ret'] * w).sum() / w.sum()

mkt_ret = crsp.groupby('date').apply(vw_return).rename('vw_mkt_ret')
ew_ret  = crsp.groupby('date')['ret'].mean().rename('ew_mkt_ret')

comparison = pd.concat([mkt_ret, ew_ret], axis=1).dropna()
cum_vw = (1 + comparison['vw_mkt_ret']).cumprod()
cum_ew = (1 + comparison['ew_mkt_ret']).cumprod()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(cum_vw.index, cum_vw, label='Value-Weighted', linewidth=1.5, color='navy')
axes[0].plot(cum_ew.index, cum_ew, label='Equal-Weighted', linewidth=1.5, color='coral')
axes[0].set_title("Cumulative Market Return"); axes[0].set_ylabel("Cumulative return (×)")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Cross-sectional return distribution
axes[1].hist(crsp['ret'].dropna(), bins=60, range=(-0.5, 0.5),
             color='steelblue', edgecolor='white', linewidth=0.3, alpha=0.8)
axes[1].axvline(crsp['ret'].mean(), color='red', linewidth=2, linestyle='--',
                label=f"Mean={crsp['ret'].mean()*100:.2f}%")
axes[1].set_xlabel("Monthly Return"); axes[1].set_title("Return Distribution (CRSP)")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("crsp_returns.png", dpi=150)
plt.show()
```

### Step 3: CRSP-Compustat Merge

```python
import pandas as pd
import numpy as np

# ------------------------------------------------------------------ #
# Merge CRSP and Compustat via CCM link table
# Follow standard academic convention (Fama-French style)
# ------------------------------------------------------------------ #

try:
    import wrds
    import os
    db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME", ""))

    # Download CCM link table
    ccm_query = """
        SELECT
            gvkey, lpermno AS permno,
            linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE
            linktype IN ('LC', 'LU', 'LX', 'LD')
            AND linkprim IN ('P', 'C')
    """
    ccm = db.raw_sql(ccm_query, date_cols=['linkdt', 'linkenddt'])
    print(f"CCM link table: {ccm.shape[0]:,} records")

    # Fill missing end dates
    ccm['linkenddt'] = ccm['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

    # Merge: given a date, find valid gvkey-permno link
    def get_permno(gvkey, date, ccm_df):
        """Return permno for a gvkey on a given date."""
        valid = ccm_df[
            (ccm_df['gvkey'] == gvkey) &
            (ccm_df['linkdt'] <= date) &
            (ccm_df['linkenddt'] >= date)
        ]
        if len(valid) == 0:
            return np.nan
        return valid['permno'].iloc[0]

    print("CCM merge function ready")
    print("Usage: permno = get_permno(gvkey='001234', date=pd.Timestamp('2020-06-30'), ccm_df=ccm)")

except Exception as e:
    print(f"WRDS not available ({e})")
    print("Demonstrating CCM merge logic with synthetic data...")

# ---- Standard merge protocol (Fama-French style) --------------- #
# 1. Use Compustat fiscal year ending in year t
# 2. Compute accounting variables in June of year t+1 (6-month lag)
# 3. Match to CRSP returns from July t+1 to June t+2

print("""
Standard CCM Merge Protocol:
1. Download comp.funda (fyear=t)
2. Set 'jdate' = fyear-end date + 6 months (June t+1)
3. Download crsp.msf for July t+1 onward
4. Merge via CCM: crsp.permno matched to comp.gvkey using jdate
5. Lag accounting variables by 6 months for look-ahead bias prevention

Example SQL (full merge):
    SELECT a.*, b.permno
    FROM comp.funda a
    LEFT JOIN crsp.ccmxpf_lnkhist b
        ON a.gvkey = b.gvkey
        AND b.linktype IN ('LC','LU')
        AND b.linkprim IN ('P','C')
        AND b.linkdt <= a.datadate
        AND (b.linkenddt >= a.datadate OR b.linkenddt IS NULL)
""")
```

---

## Advanced Usage

### TAQ Intraday Microstructure

```python
import pandas as pd
import numpy as np

# ------------------------------------------------------------------ #
# Effective spread and price impact from TAQ trades and quotes
# ------------------------------------------------------------------ #

try:
    import wrds, os
    db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME", ""))

    # TAQ daily trades for one stock (example: AAPL on 2023-01-03)
    taq_query = """
        SELECT time, price, size, tr_corr, tr_scond
        FROM taq.ct_20230103
        WHERE sym_root = 'AAPL'
            AND tr_corr = '00'   -- Regular trades only
            AND size > 0
        ORDER BY time
        LIMIT 10000
    """
    trades = db.raw_sql(taq_query)
    print(f"TAQ trades: {len(trades):,} records")
except Exception as e:
    print(f"TAQ not available ({e}) — using synthetic data")
    # Simulate intraday trades
    np.random.seed(42)
    n = 5000
    times = pd.date_range("2023-01-03 09:30", periods=n, freq='5s')
    mid_price = 130 + np.cumsum(np.random.randn(n) * 0.01)
    spread = 0.02  # $0.02 spread
    sign = np.random.choice([-1, 1], n)  # Buy/sell indicator
    trades = pd.DataFrame({
        'time':  times,
        'price': mid_price + sign * spread/2 + np.random.randn(n) * 0.001,
        'size':  np.random.randint(100, 2000, n),
        'mid':   mid_price,
        'sign':  sign,
    })

    # Effective spread = 2 * |price - mid| / mid
    trades['eff_spread'] = 2 * np.abs(trades['price'] - trades['mid']) / trades['mid']
    print(f"Mean effective spread: {trades['eff_spread'].mean()*10000:.2f} bps")

    # Amihud illiquidity: |return| / dollar volume
    trades['dollar_vol'] = trades['price'] * trades['size']
    trades['return_5min'] = trades['price'].pct_change(12)  # ~1min return
    trades['amihud'] = trades['return_5min'].abs() / (trades['dollar_vol'] / 1e6)
    print(f"Amihud illiquidity (per $M): {trades['amihud'].mean():.4f}")
```

---

## Troubleshooting

### Error: `ConnectionError` — WRDS authentication

**Fix**:
```bash
# Re-authenticate
python -c "import wrds; db = wrds.Connection()"
# Enter username and password when prompted — stored in ~/.pgpass
```

### Error: `psycopg2.OperationalError: SSL SYSCALL error`

**Cause**: VPN/firewall blocking WRDS connection (port 5432).

**Fix**: Use WRDS web query interface or connect via campus network/VPN.

### Memory error: large SQL query

**Fix**: Add row limits or filter by date range:
```python
# Download in annual chunks
for year in range(2010, 2024):
    chunk = db.raw_sql(f"""
        SELECT * FROM comp.funda WHERE fyear = {year} AND at > 0
    """)
    chunk.to_parquet(f"compustat_{year}.parquet")
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| wrds | 3.1.x | Requires institutional WRDS subscription |
| sqlalchemy | 2.0 | `wrds` 3.x requires SQLAlchemy 2.0+ |
| pandas | 2.0, 2.1 | `date_cols` parsing in `raw_sql` |

---

## External Resources

### Official Documentation

- [WRDS Python documentation](https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/python-from-your-computer/)
- [CRSP Data Descriptions](https://crsp.org/data-products/crsp-us-stock-databases/)

### Key Papers

- Fama, E.F. & French, K.R. (1993). *Common risk factors in returns on stocks and bonds*. Journal of Financial Economics.
- Davis, J.L. et al. (2000). *Characteristics, covariances, and average returns*. Journal of Finance.

---

## Examples

### Example 1: Momentum Factor Construction (UMD)

```python
import pandas as pd
import numpy as np

# ------------------------------------------------------------------ #
# Construct momentum factor: past 12-2 month return
# Following Jegadeesh & Titman (1993)
# ------------------------------------------------------------------ #
np.random.seed(42)
n_stocks = 500
n_months = 72  # 6 years
dates = pd.period_range("2018-01", periods=n_months, freq='M')

# Simulate monthly returns panel
ret_data = pd.DataFrame(
    np.random.normal(0.01, 0.08, (n_months, n_stocks)),
    index=dates,
    columns=[f"S{i:04d}" for i in range(n_stocks)]
)

def compute_momentum(ret_df, formation=12, skip=1):
    """Compute past (formation-skip) to (skip) month return."""
    cum_ret = (1 + ret_df).rolling(formation - skip).apply(np.prod, raw=True) - 1
    return cum_ret.shift(skip)  # Skip most recent month

mom = compute_momentum(ret_df, formation=12, skip=1)

# Sort into deciles each month, compute top-bottom spread
umd_monthly = []
for date in dates[12:]:
    m = mom.loc[date].dropna()
    if len(m) < 50:
        continue
    deciles = pd.qcut(m, 10, labels=False)
    winner = ret_df.loc[date][deciles == 9].mean()  # Top decile (Winners)
    loser  = ret_df.loc[date][deciles == 0].mean()  # Bottom decile (Losers)
    umd_monthly.append(winner - loser)

umd = pd.Series(umd_monthly)
print(f"UMD monthly mean: {umd.mean()*100:.3f}%  (annualized: {umd.mean()*1200:.2f}%)")
print(f"UMD monthly std:  {umd.std()*100:.3f}%")
print(f"UMD Sharpe ratio: {umd.mean()/umd.std()*np.sqrt(12):.3f}")
```

### Example 2: Earnings Surprise (SUE)

```python
import numpy as np
import pandas as pd

# Standardized Unexpected Earnings (SUE)
# SUE = (EPS_actual - EPS_expected) / std(forecast_error)

np.random.seed(0)
n = 2000
eps_actual   = np.random.normal(1.5, 0.8, n)
eps_expected = eps_actual + np.random.normal(0, 0.3, n)  # Analyst forecast
sue = (eps_actual - eps_expected) / (np.abs(eps_actual - eps_expected).rolling(8, min_periods=3).std()
       if hasattr(eps_actual, 'rolling') else np.std(np.abs(eps_actual - eps_expected)))

# Simple SUE
std_err = np.std(eps_actual - eps_expected)
sue_simple = (eps_actual - eps_expected) / (std_err + 1e-8)

# Portfolio returns by SUE quintile
next_ret = np.random.normal(0.01, 0.05, n) + sue_simple * 0.005  # Alpha correlated with SUE
quintiles = pd.qcut(sue_simple, 5, labels=['Q1 (neg)','Q2','Q3','Q4','Q5 (pos)'])
print("Mean next-month return by SUE quintile:")
for q in ['Q1 (neg)','Q2','Q3','Q4','Q5 (pos)']:
    mean_ret = next_ret[quintiles == q].mean() * 100
    print(f"  {q}: {mean_ret:.3f}%")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
