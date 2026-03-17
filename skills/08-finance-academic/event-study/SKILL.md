---
name: event-study
description: >
  Event study methodology in finance: estimate abnormal returns, CARs, BMP tests,
  Fama-French 3-factor normal returns, and long-run BHAR analysis using yfinance.
tags:
  - finance
  - event-study
  - abnormal-returns
  - yfinance
  - python
  - econometrics
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
  - yfinance>=0.2.28
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - statsmodels>=0.14.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# Event Study Methodology in Finance

Event studies measure the impact of corporate or macroeconomic events on asset prices
by comparing actual returns to a counterfactual (normal return model). This skill
covers the classic short-window CAR approach, non-parametric rank tests, the
Fama-French 3-factor model for normal returns, and long-run buy-and-hold abnormal
returns (BHAR).

---

## 1. Setup

```bash
pip install yfinance pandas numpy scipy statsmodels matplotlib
```

```python
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Estimation window and event window conventions
# ---------------------------------------------------------------------------
EST_START  = -250   # trading days before event date
EST_END    = -11    # trading days before event date (exclude pre-event window)
EVT_START  = -5     # trading days relative to event (inclusive)
EVT_END    = +5     # trading days relative to event (inclusive)
```

---

## 2. Data Download

```python
def download_returns(
    tickers: list[str],
    market_ticker: str = "^GSPC",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download adjusted closing prices and compute log returns.

    Parameters
    ----------
    tickers : list[str]
        Stock tickers to download.
    market_ticker : str
        Market index ticker (default: S&P 500).
    start, end : str
        Date range for download.

    Returns
    -------
    tuple of (stock_returns DataFrame, market_returns Series)
        Both use log returns: log(P_t / P_{t-1}).
    """
    all_tickers = list(set(tickers + [market_ticker]))
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    log_ret = np.log(prices / prices.shift(1)).dropna()

    market_ret   = log_ret[market_ticker].rename("market")
    stock_rets   = log_ret[[t for t in tickers if t in log_ret.columns]]

    return stock_rets, market_ret
```

---

## 3. Normal Return Estimation (Market Model OLS)

```python
def estimate_normal_returns(
    ret_series: pd.Series,
    market_ret: pd.Series,
    event_date: pd.Timestamp,
    est_window: tuple[int, int] = (EST_START, EST_END),
) -> dict:
    """
    Estimate normal return parameters using the OLS market model on the
    estimation window.

    Parameters
    ----------
    ret_series : pd.Series
        Daily log returns for one stock.
    market_ret : pd.Series
        Daily log returns for the market index.
    event_date : pd.Timestamp
        The event date (day 0).
    est_window : tuple
        (start_offset, end_offset) relative to event_date in trading days.

    Returns
    -------
    dict with keys: alpha, beta, sigma, r2, n_obs.
    """
    # Align stock and market returns
    combined = pd.concat([ret_series.rename("stock"), market_ret], axis=1).dropna()
    dates    = combined.index.tolist()

    # Find event_date position
    if event_date not in combined.index:
        # Use nearest available date
        pos = combined.index.searchsorted(event_date)
        pos = min(pos, len(dates) - 1)
    else:
        pos = dates.index(event_date)

    # Estimation window indices
    est_s = max(0, pos + est_window[0])
    est_e = max(0, pos + est_window[1])

    est_data = combined.iloc[est_s:est_e]
    if len(est_data) < 50:
        raise ValueError(f"Estimation window too short ({len(est_data)} obs). "
                         "Extend the sample or choose an earlier event date.")

    X = sm.add_constant(est_data["market"])
    ols = sm.OLS(est_data["stock"], X).fit()

    return {
        "alpha": ols.params["const"],
        "beta":  ols.params["market"],
        "sigma": ols.resid.std(),
        "r2":    ols.rsquared,
        "n_obs": len(est_data),
    }
```

---

## 4. Abnormal Returns and CAR

```python
def compute_car(
    ret_series: pd.Series,
    market_ret: pd.Series,
    event_date: pd.Timestamp,
    model_params: dict,
    event_window: tuple[int, int] = (EVT_START, EVT_END),
) -> pd.Series:
    """
    Compute abnormal returns (AR) and cumulative abnormal return (CAR)
    over the event window.

    Parameters
    ----------
    ret_series : pd.Series
        Stock daily log returns.
    market_ret : pd.Series
        Market daily log returns.
    event_date : pd.Timestamp
        Day 0.
    model_params : dict
        Output of estimate_normal_returns().
    event_window : tuple
        (start_offset, end_offset) relative to event_date.

    Returns
    -------
    pd.Series of cumulative abnormal returns indexed by relative day.
    """
    combined = pd.concat([ret_series.rename("stock"), market_ret], axis=1).dropna()
    dates    = combined.index.tolist()

    if event_date not in combined.index:
        pos = min(combined.index.searchsorted(event_date), len(dates) - 1)
    else:
        pos = dates.index(event_date)

    evt_s = max(0, pos + event_window[0])
    evt_e = min(len(dates), pos + event_window[1] + 1)
    evt_data = combined.iloc[evt_s:evt_e].copy()

    # Abnormal returns = actual - (alpha + beta * market)
    evt_data["expected"] = (model_params["alpha"]
                            + model_params["beta"] * evt_data["market"])
    evt_data["AR"]  = evt_data["stock"] - evt_data["expected"]
    evt_data["CAR"] = evt_data["AR"].cumsum()

    # Relative event day index
    rel_days = list(range(event_window[0], event_window[0] + len(evt_data)))
    evt_data.index = rel_days[:len(evt_data)]

    return evt_data["AR"], evt_data["CAR"]
```

---

## 5. Statistical Tests

### 5.1 Cross-Sectional t-Test and BMP Test

```python
def test_car_bmp(
    cars: np.ndarray,
    t_stats: np.ndarray,
) -> dict:
    """
    Test H0: mean CAR = 0 using:
      (1) Cross-sectional t-test (Fama, 1969)
      (2) Boehmer-Musumeci-Poulsen (BMP, 1991) standardized cross-sectional test

    Parameters
    ----------
    cars : np.ndarray
        Array of per-firm CARs over the event window.
    t_stats : np.ndarray
        Per-firm t-statistics (AR / sigma_i * sqrt(L)), where L is event window length.

    Returns
    -------
    dict with keys: mean_car, t_cs, p_cs, t_bmp, p_bmp.
    """
    n = len(cars)
    mean_car = np.mean(cars)
    std_car  = np.std(cars, ddof=1)

    # (1) Cross-sectional t-test
    t_cs = mean_car / (std_car / np.sqrt(n))
    p_cs = 2 * (1 - stats.t.cdf(abs(t_cs), df=n - 1))

    # (2) BMP test: based on standardized abnormal returns
    mean_t = np.mean(t_stats)
    std_t  = np.std(t_stats, ddof=1)
    t_bmp  = mean_t / (std_t / np.sqrt(n))
    p_bmp  = 2 * (1 - stats.t.cdf(abs(t_bmp), df=n - 1))

    return {
        "n":        n,
        "mean_car": round(mean_car * 100, 4),   # in percent
        "t_cs":     round(t_cs, 4),
        "p_cs":     round(p_cs, 4),
        "t_bmp":    round(t_bmp, 4),
        "p_bmp":    round(p_bmp, 4),
    }
```

### 5.2 Non-Parametric Rank Test (Corrado, 1989)

```python
def rank_test(ar_panel: pd.DataFrame) -> dict:
    """
    Corrado (1989) non-parametric rank test.

    Parameters
    ----------
    ar_panel : pd.DataFrame
        Rows = event-window days (relative index), columns = firms.
        Values are abnormal returns.

    Returns
    -------
    dict with z-statistic and p-value.
    """
    # Rank within each firm across the combined estimation + event period
    # Approximate: rank within the event window only
    ranks = ar_panel.rank(axis=0)   # rank across event days, per firm
    n, T  = ar_panel.shape

    # Expected rank under H0
    K_bar = (T + 1) / 2
    rank_excess = ranks - K_bar

    # Cross-sectional average of rank excess at each event day
    mean_rank_t = rank_excess.mean(axis=1)
    sigma_rank  = rank_excess.std(axis=1).mean()

    # Z at day 0 (relative day = 0)
    if 0 in mean_rank_t.index:
        z_stat = mean_rank_t.loc[0] / (sigma_rank / np.sqrt(n))
    else:
        z_stat = mean_rank_t.iloc[T // 2] / (sigma_rank / np.sqrt(n))

    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return {"z_stat": round(float(z_stat), 4), "p_value": round(p_val, 4)}
```

---

## 6. Fama-French 3-Factor Normal Returns

```python
def ff3_normal_returns(
    ret_series: pd.Series,
    ff3_factors: pd.DataFrame,
    event_date: pd.Timestamp,
    est_window: tuple[int, int] = (EST_START, EST_END),
) -> dict:
    """
    Estimate normal returns using the Fama-French 3-factor model.

    Parameters
    ----------
    ret_series : pd.Series
        Excess returns of the stock (ret - rf). Index = DatetimeIndex.
    ff3_factors : pd.DataFrame
        Columns: ['Mkt-RF', 'SMB', 'HML', 'RF']. Obtainable from
        Ken French's data library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
    event_date : pd.Timestamp
        Day 0.
    est_window : tuple
        Estimation window offsets.

    Returns
    -------
    dict with OLS params and sigma.
    """
    combined = pd.concat(
        [ret_series.rename("stock_excess"), ff3_factors[["Mkt-RF", "SMB", "HML"]]],
        axis=1,
    ).dropna()

    dates = combined.index.tolist()
    if event_date not in combined.index:
        pos = min(combined.index.searchsorted(event_date), len(dates) - 1)
    else:
        pos = dates.index(event_date)

    est_s = max(0, pos + est_window[0])
    est_e = max(0, pos + est_window[1])
    est   = combined.iloc[est_s:est_e]

    X   = sm.add_constant(est[["Mkt-RF", "SMB", "HML"]])
    ols = sm.OLS(est["stock_excess"], X).fit()

    return {
        "alpha":    ols.params["const"],
        "b_mkt":    ols.params["Mkt-RF"],
        "b_smb":    ols.params["SMB"],
        "b_hml":    ols.params["HML"],
        "sigma":    ols.resid.std(),
        "r2":       ols.rsquared,
    }
```

---

## 7. Long-Run BHAR

```python
def compute_bhar(
    ret_series: pd.Series,
    market_ret: pd.Series,
    event_date: pd.Timestamp,
    months: int = 36,
) -> float:
    """
    Compute the buy-and-hold abnormal return (BHAR) over a post-event window.

    BHAR = prod(1 + r_i,t) - prod(1 + r_m,t)  over `months` calendar months.

    Parameters
    ----------
    ret_series : pd.Series
        Stock daily log returns.
    market_ret : pd.Series
        Market daily log returns.
    event_date : pd.Timestamp
        Listing / event date.
    months : int
        Number of months for long-run window.

    Returns
    -------
    float: BHAR in decimal form.
    """
    combined = pd.concat([ret_series.rename("stock"), market_ret], axis=1).dropna()

    end_date = event_date + pd.DateOffset(months=months)
    window   = combined.loc[event_date:end_date]

    if len(window) < 20:
        return np.nan

    # Compounded gross returns (log to simple)
    bhr_stock  = np.exp(window["stock"].sum()) - 1
    bhr_market = np.exp(window["market"].sum()) - 1
    return bhr_stock - bhr_market
```

---

## 8. Event Study Plot

```python
def plot_event_study_car(
    car_means: pd.Series,
    ci_low: pd.Series,
    ci_high: pd.Series,
    event_window: tuple[int, int],
    title: str = "Cumulative Abnormal Returns",
) -> plt.Figure:
    """
    Plot mean CAR with 95% confidence interval across the event window.

    Parameters
    ----------
    car_means : pd.Series
        Mean CAR at each relative event day.
    ci_low, ci_high : pd.Series
        95% CI lower and upper bounds.
    event_window : tuple
        (start_day, end_day).
    title : str
        Chart title.

    Returns
    -------
    matplotlib Figure.
    """
    days = car_means.index

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, car_means * 100, color="steelblue", linewidth=2, label="Mean CAR")
    ax.fill_between(days, ci_low * 100, ci_high * 100,
                    alpha=0.25, color="steelblue", label="95% CI")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Event Day 0")
    ax.axhline(0, color="black", linewidth=0.7)

    ax.set_xlabel("Trading Days Relative to Event")
    ax.set_ylabel("CAR (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(range(event_window[0], event_window[1] + 1))
    plt.tight_layout()
    return fig
```

---

## 9. Example A — FDA Approval Announcement CAR [-1, +1]

```python
import pandas as pd
import numpy as np

# ---- Event list: (ticker, announcement_date) -----------------------------------
events = [
    ("MRNA",  "2020-12-18"),
    ("BNTX",  "2021-08-23"),
    ("PFE",   "2021-08-23"),
    ("NVAX",  "2022-07-13"),
]

START_DATE  = "2018-01-01"
END_DATE    = "2023-06-30"
EVENT_WIN   = (-1, 1)   # tight window for FDA news

tickers = list({t for t, _ in events})
stock_rets, market_ret = download_returns(tickers, market_ticker="^GSPC",
                                          start=START_DATE, end=END_DATE)

all_cars, all_tstats = [], []

for ticker, evt_date_str in events:
    if ticker not in stock_rets.columns:
        print(f"Skipping {ticker}: not in downloaded data.")
        continue

    evt_date = pd.Timestamp(evt_date_str)
    ret_s    = stock_rets[ticker]

    try:
        params = estimate_normal_returns(ret_s, market_ret, evt_date,
                                         est_window=(EST_START, EST_END))
        ar_series, car_series = compute_car(ret_s, market_ret, evt_date,
                                             params, event_window=EVENT_WIN)

        car_val = car_series.iloc[-1]
        # Standardised t-stat for BMP
        L      = len(EVENT_WIN[0] - EVENT_WIN[0]) + (EVENT_WIN[1] - EVENT_WIN[0]) + 1
        t_stat = car_val / (params["sigma"] * np.sqrt(L))

        all_cars.append(car_val)
        all_tstats.append(t_stat)

        print(f"{ticker} ({evt_date_str}): "
              f"CAR[-1,+1] = {car_val*100:.2f}%, "
              f"alpha={params['alpha']:.5f}, beta={params['beta']:.3f}")
    except Exception as exc:
        print(f"{ticker}: error — {exc}")

if all_cars:
    result = test_car_bmp(np.array(all_cars), np.array(all_tstats))
    print("\n=== CAR Test Results ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
```

---

## 10. Example B — Long-Run BHAR for Post-IPO Performance (36 Months)

```python
import pandas as pd
import numpy as np

# ---- Simulated IPO events (use real data from SDC Platinum or similar) ----------
ipo_events = [
    ("SNOW",  "2020-09-16"),
    ("COIN",  "2021-04-14"),
    ("RIVN",  "2021-11-10"),
    ("UBER",  "2019-05-10"),
    ("LYFT",  "2019-03-29"),
]

START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
MONTHS     = 36

tickers = list({t for t, _ in ipo_events})
stock_rets, market_ret = download_returns(tickers, market_ticker="^GSPC",
                                          start=START_DATE, end=END_DATE)

bhar_results = []

for ticker, ipo_date_str in ipo_events:
    if ticker not in stock_rets.columns:
        print(f"Skipping {ticker}: no data.")
        continue

    ipo_date = pd.Timestamp(ipo_date_str)
    bhar = compute_bhar(stock_rets[ticker], market_ret, ipo_date, months=MONTHS)

    bhar_results.append({"ticker": ticker, "ipo_date": ipo_date_str,
                          "bhar_36m": round(bhar * 100, 2) if not np.isnan(bhar) else np.nan})
    print(f"{ticker} ({ipo_date_str}): {MONTHS}-month BHAR = {bhar*100:.2f}%"
          if not np.isnan(bhar) else f"{ticker}: insufficient data")

bhar_df = pd.DataFrame(bhar_results)
print("\n=== Long-Run IPO BHAR Summary ===")
print(bhar_df.to_string(index=False))

mean_bhar = bhar_df["bhar_36m"].mean()
se_bhar   = bhar_df["bhar_36m"].std() / np.sqrt(len(bhar_df))
t_stat    = mean_bhar / se_bhar if se_bhar > 0 else np.nan
print(f"\nMean BHAR: {mean_bhar:.2f}%")
print(f"t-statistic: {t_stat:.3f}")
print("(Negative BHAR is typical for IPOs — long-run underperformance puzzle)")

# ---- Bar chart ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
colors = ["steelblue" if v >= 0 else "tomato" for v in bhar_df["bhar_36m"]]
ax.bar(bhar_df["ticker"], bhar_df["bhar_36m"], color=colors, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("36-Month BHAR (%)")
ax.set_title("Post-IPO Long-Run Abnormal Returns (36 months)", fontweight="bold")
plt.tight_layout()
plt.savefig("ipo_bhar_36m.png", dpi=150)
plt.show()
```

---

## 11. Tips and Common Pitfalls

- **Estimation window contamination**: Ensure the estimation window [-250, -11]
  does not overlap with other major announcements for the same firm. Use [-250, -31]
  for extra safety around quarterly earnings clusters.
- **Non-synchronous trading**: For thinly traded stocks, use weekly returns or the
  Scholes-Williams beta correction for the market model.
- **Event-date misalignment**: After-market announcements count as the next trading
  day (day +1). Always verify the announcement time (AMC vs BMO).
- **Cross-sectional dependence**: If events cluster in calendar time (e.g., industry
  shocks), individual AR are correlated. Use the calendar-time portfolio approach
  (Fama, 1998) as a robustness check.
- **BHAR vs CAAR**: BHAR compounds returns (multiplicative) while CAAR sums them
  (additive). BHAR better reflects investor wealth; CAAR is easier to test
  statistically. Use both and compare.
- **Benchmark choice for BHAR**: Common choices are the value-weighted market index,
  size-and-BM matched control firms, or the Fama-French 3-factor alpha.
