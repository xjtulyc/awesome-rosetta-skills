---
name: fred-macro
description: >
  Fetch and analyze FRED macroeconomic time series data using fredapi; covers GDP,
  unemployment, inflation, yield curves, recession shading, and HP filtering.
tags:
  - economics
  - macroeconomics
  - time-series
  - fred
  - visualization
  - python
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
  - fredapi>=0.5.1
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - statsmodels>=0.14.0
  - openpyxl>=3.1.0
last_updated: "2026-03-17"
---

# FRED Macroeconomic Data Analysis

This skill covers end-to-end macroeconomic data workflows using the Federal Reserve
Economic Data (FRED) API via the `fredapi` Python package. You will learn to
authenticate, fetch single and panel series, compute growth rates, apply the
Hodrick-Prescott filter, date business cycles, and produce publication-quality charts
with recession shading.

---

## 1. Authentication and Setup

Store your FRED API key as an environment variable. Never hard-code keys in source files.

```bash
# Obtain a free key at https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY="<paste-your-key>"
pip install fredapi pandas numpy matplotlib statsmodels openpyxl
```

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# ---------------------------------------------------------------------------
# Authenticate
# ---------------------------------------------------------------------------
FRED_API_KEY = os.environ["FRED_API_KEY"]
fred = Fred(api_key=FRED_API_KEY)

# Key FRED series IDs used throughout this skill
SERIES = {
    "gdp":          "GDPC1",      # Real GDP (quarterly, chained 2017 $)
    "unemployment": "UNRATE",     # Civilian unemployment rate (monthly)
    "cpi":          "CPIAUCSL",   # CPI All Urban Consumers (monthly)
    "pce":          "PCEPI",      # PCE Price Index (monthly)
    "ppi":          "PPIACO",     # PPI All Commodities (monthly)
    "fedfunds":     "FEDFUNDS",   # Effective federal funds rate (monthly)
    "t10y2y":       "T10Y2Y",     # 10Y-2Y Treasury yield spread (daily)
    "t10y3m":       "T10Y3M",     # 10Y-3M Treasury yield spread (daily)
    "recession":    "USREC",      # NBER recession indicator (monthly, 0/1)
    "indpro":       "INDPRO",     # Industrial production index (monthly)
    "m2":           "M2SL",       # M2 money supply (monthly)
    "realwage":     "AHETPI",     # Average hourly earnings (monthly)
}
```

---

## 2. Core Helper Functions

### 2.1 Fetch a Single Series

```python
def get_fred_series(
    series_id: str,
    start: str = "2000-01-01",
    end: str | None = None,
    frequency: str | None = None,
    units: str = "lin",
) -> pd.Series:
    """
    Fetch a FRED series as a pandas Series with a DatetimeIndex.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. 'GDPC1'.
    start : str
        ISO date string for the start of the sample.
    end : str or None
        ISO date string for the end of the sample. Defaults to today.
    frequency : str or None
        Aggregation frequency override: 'a', 'q', 'm', 'w', 'd'.
        Pass None to keep the native frequency.
    units : str
        Transformation: 'lin' (levels), 'chg', 'ch1', 'pch', 'pc1',
        'pca', 'cch', 'cca', 'log'.

    Returns
    -------
    pd.Series
        Named after series_id with a DatetimeIndex.
    """
    kwargs = {
        "observation_start": start,
        "units": units,
    }
    if end:
        kwargs["observation_end"] = end
    if frequency:
        kwargs["frequency"] = frequency

    series = fred.get_series(series_id, **kwargs)
    series.name = series_id
    series.index = pd.to_datetime(series.index)
    return series.dropna()


def get_recession_shading(start: str = "1950-01-01", end: str | None = None) -> pd.Series:
    """
    Return the NBER recession indicator (USREC) as a boolean Series.
    Value is 1 during recession months, 0 otherwise.
    """
    rec = get_fred_series("USREC", start=start, end=end)
    return rec.astype(bool)
```

### 2.2 Growth Rates

```python
def calculate_growth_rates(series: pd.Series, freq: str = "YoY") -> pd.Series:
    """
    Compute period-over-period growth rates.

    Parameters
    ----------
    series : pd.Series
        Level series with DatetimeIndex.
    freq : str
        'MoM'  — month-over-month % change
        'YoY'  — year-over-year % change
        'QoQ'  — quarter-over-quarter % change (annualised if series is quarterly)

    Returns
    -------
    pd.Series
        Percentage change series, same index as input.
    """
    freq = freq.upper()
    if freq == "MOM":
        return series.pct_change(1) * 100
    elif freq == "YOY":
        periods = 4 if series.index.inferred_freq in ("QS", "Q", "QE") else 12
        return series.pct_change(periods) * 100
    elif freq == "QOQ":
        return series.pct_change(1) * 100
    else:
        raise ValueError(f"Unknown freq '{freq}'. Choose MoM, YoY, or QoQ.")
```

### 2.3 Hodrick-Prescott Decomposition

```python
def hp_decompose(series: pd.Series, lamb: int | None = None) -> pd.DataFrame:
    """
    Apply the Hodrick-Prescott filter to separate trend from cycle.

    Parameters
    ----------
    series : pd.Series
        Macroeconomic level series.
    lamb : int or None
        Smoothing parameter. Defaults: 1600 for quarterly, 129600 for monthly,
        6.25 for annual (Ravn-Uhlig rule).

    Returns
    -------
    pd.DataFrame
        Columns: ['original', 'trend', 'cycle']
    """
    if lamb is None:
        freq = pd.infer_freq(series.index)
        if freq and freq.startswith("Q"):
            lamb = 1600
        elif freq and freq.startswith("M"):
            lamb = 129600
        else:
            lamb = 6.25  # annual

    cycle, trend = hpfilter(series.dropna(), lamb=lamb)
    return pd.DataFrame(
        {"original": series, "trend": trend, "cycle": cycle},
        index=series.index,
    )
```

### 2.4 Build a Macro Panel

```python
def build_macro_panel(
    series_dict: dict[str, str],
    start: str = "2000-01-01",
    end: str | None = None,
    frequency: str = "m",
) -> pd.DataFrame:
    """
    Fetch multiple FRED series and combine into a wide DataFrame.

    Parameters
    ----------
    series_dict : dict
        Mapping of column_name -> FRED series_id.
    start, end : str
        Sample bounds.
    frequency : str
        Target frequency for all series ('m', 'q', 'a').

    Returns
    -------
    pd.DataFrame
        Wide DataFrame; rows are dates, columns are variable names.
    """
    frames = {}
    for name, sid in series_dict.items():
        try:
            frames[name] = get_fred_series(sid, start=start, end=end, frequency=frequency)
        except Exception as exc:
            print(f"Warning: could not fetch {sid} ({exc})")
    return pd.DataFrame(frames)
```

### 2.5 Recession-Shaded Plot

```python
def plot_with_recession_shading(
    df: pd.DataFrame,
    series_col: str,
    title: str,
    ylabel: str = "",
    color: str = "steelblue",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot a time series with NBER recession shading.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `series_col` and optionally 'recession' column.
    series_col : str
        Column name to plot.
    title : str
        Chart title.
    ylabel : str
        Y-axis label.
    color : str
        Line color.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Draw series
    ax.plot(df.index, df[series_col], color=color, linewidth=1.8, label=series_col)

    # Recession shading
    if "recession" in df.columns:
        rec = df["recession"].fillna(0).astype(bool)
        in_rec = False
        rec_start = None
        for date, val in rec.items():
            if val and not in_rec:
                rec_start = date
                in_rec = True
            elif not val and in_rec:
                ax.axvspan(rec_start, date, color="lightgray", alpha=0.7, label="_nolegend_")
                in_rec = False
        if in_rec:
            ax.axvspan(rec_start, df.index[-1], color="lightgray", alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()
    ax.legend()
    plt.tight_layout()
    return fig
```

---

## 3. Business Cycle Dating

```python
def date_business_cycles(recession_series: pd.Series) -> pd.DataFrame:
    """
    Extract recession peak/trough dates from the USREC indicator.

    Returns a DataFrame with columns: ['peak', 'trough', 'duration_months'].
    """
    rec = recession_series.astype(int)
    starts, ends = [], []

    in_rec = False
    for date, val in rec.items():
        if val == 1 and not in_rec:
            starts.append(date)
            in_rec = True
        elif val == 0 and in_rec:
            ends.append(date)
            in_rec = False

    if in_rec:
        ends.append(rec.index[-1])

    records = []
    for s, e in zip(starts, ends):
        duration = len(rec.loc[s:e])
        records.append({"peak": s, "trough": e, "duration_months": duration})

    return pd.DataFrame(records)
```

---

## 4. Export Utilities

```python
def export_to_excel(df: pd.DataFrame, path: str, sheet_name: str = "FRED Data") -> None:
    """Save DataFrame to Excel with auto-fit column widths."""
    with pd.ExcelWriter(path, engine="openpyxl", datetime_format="YYYY-MM-DD") as writer:
        df.to_excel(writer, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            length = max(len(str(cell.value or "")) for cell in col_cells) + 2
            ws.column_dimensions[col_cells[0].column_letter].width = min(length, 30)
    print(f"Saved to {path}")


def export_to_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(path)
    print(f"Saved to {path}")
```

---

## 5. Example A — Yield Curve Inversion Tracker with Recession Overlay

This example plots the 10Y-2Y and 10Y-3M yield spreads, highlights inversion zones, and
overlays NBER recession shading to illustrate the leading indicator property.

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])

# ---- Fetch data ---------------------------------------------------------------
t10y2y  = get_fred_series("T10Y2Y",  start="1990-01-01", frequency="m")
t10y3m  = get_fred_series("T10Y3M",  start="1990-01-01", frequency="m")
usrec   = get_recession_shading(start="1990-01-01")

panel = pd.DataFrame({"t10y2y": t10y2y, "t10y3m": t10y3m, "recession": usrec}).dropna(subset=["t10y2y"])

# ---- Plot ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(panel.index, panel["t10y2y"], label="10Y-2Y Spread", color="navy",   linewidth=1.6)
ax.plot(panel.index, panel["t10y3m"], label="10Y-3M Spread", color="crimson", linewidth=1.3, linestyle="--")
ax.axhline(0, color="black", linewidth=0.8)

# Shade inversions (spread < 0)
ax.fill_between(panel.index, panel["t10y2y"], 0,
                where=panel["t10y2y"] < 0, alpha=0.25, color="navy", label="Inversion (10Y-2Y)")

# Recession shading
rec = panel["recession"].fillna(False).astype(bool)
in_rec, rec_start = False, None
for date, val in rec.items():
    if val and not in_rec:
        rec_start, in_rec = date, True
    elif not val and in_rec:
        ax.axvspan(rec_start, date, color="lightgray", alpha=0.6)
        in_rec = False
if in_rec:
    ax.axvspan(rec_start, panel.index[-1], color="lightgray", alpha=0.6, label="NBER Recession")

ax.set_title("U.S. Treasury Yield Curve Spreads vs NBER Recessions", fontsize=13, fontweight="bold")
ax.set_ylabel("Percentage Points")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(4))
fig.autofmt_xdate()
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig("yield_curve_inversion.png", dpi=150)
plt.show()

# ---- Business cycle table -------------------------------------------------------
cycles = date_business_cycles(usrec.reindex(panel.index).fillna(False))
print(cycles.to_string(index=False))
```

---

## 6. Example B — Inflation Decomposition Dashboard (CPI, PCE, PPI)

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from fredapi import Fred

fred = Fred(api_key=os.environ["FRED_API_KEY"])

START = "2010-01-01"

# ---- Fetch price series --------------------------------------------------------
series_map = {
    "CPI":  "CPIAUCSL",
    "PCE":  "PCEPI",
    "PPI":  "PPIACO",
    "Core CPI": "CPILFESL",  # CPI ex food and energy
}

panel = build_macro_panel(series_map, start=START, frequency="m")
usrec = get_recession_shading(start=START).reindex(panel.index).ffill().fillna(False)
panel["recession"] = usrec

# ---- Compute YoY and MoM growth rates ------------------------------------------
yoy, mom = {}, {}
for col in series_map:
    yoy[col] = calculate_growth_rates(panel[col], freq="YoY")
    mom[col] = calculate_growth_rates(panel[col], freq="MoM")

yoy_df = pd.DataFrame(yoy)
mom_df = pd.DataFrame(mom)

# ---- Dashboard layout -----------------------------------------------------------
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

colors = {"CPI": "steelblue", "PCE": "darkorange", "PPI": "seagreen", "Core CPI": "mediumpurple"}

# Panel 1: YoY inflation
ax1 = fig.add_subplot(gs[0, :])
for col, clr in colors.items():
    ax1.plot(yoy_df.index, yoy_df[col], label=col, color=clr, linewidth=1.5)
ax1.axhline(2, color="red", linestyle="--", linewidth=0.9, label="2% target")
ax1.set_title("Year-over-Year Inflation (%)", fontweight="bold")
ax1.set_ylabel("YoY %")
ax1.legend(ncol=4, fontsize=8)

# Panel 2: MoM CPI
ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(mom_df.index, mom_df["CPI"], color="steelblue", width=20, label="CPI MoM")
ax2.axhline(0, color="black", linewidth=0.6)
ax2.set_title("CPI Month-over-Month (%)", fontweight="bold")
ax2.set_ylabel("MoM %")

# Panel 3: HP trend of CPI
hp_cpi = hp_decompose(panel["CPI"])
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(hp_cpi.index, hp_cpi["original"], label="CPI Level",  color="steelblue", linewidth=1.2)
ax3.plot(hp_cpi.index, hp_cpi["trend"],    label="HP Trend",   color="red",       linewidth=1.8, linestyle="--")
ax3.set_title("CPI: Hodrick-Prescott Trend", fontweight="bold")
ax3.set_ylabel("Index (1982-84=100)")
ax3.legend()

plt.suptitle("U.S. Inflation Decomposition Dashboard", fontsize=15, fontweight="bold", y=1.01)
plt.savefig("inflation_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- Export -------------------------------------------------------------------
export_to_excel(yoy_df, "inflation_yoy.xlsx", sheet_name="YoY")
export_to_csv(mom_df, "inflation_mom.csv")
print("Done.")
```

---

## 7. Tips and Common Pitfalls

- **Frequency mismatch**: FRED series have different native frequencies. Use the
  `frequency` parameter in `get_fred_series` to aggregate to a common frequency before
  merging. Prefer `frequency="m"` for most macro panels.
- **Vintage data**: `fredapi` returns the most recent vintage by default. For real-time
  analysis use `get_series_all_releases` or the vintage endpoint.
- **Units transformation**: Passing `units="pc1"` returns the YoY percent change
  directly from FRED, avoiding manual calculation—useful as a sanity check.
- **Rate limits**: The FRED API allows 120 requests per minute. Add `time.sleep(0.5)`
  between large batch fetches.
- **Recession indicator alignment**: USREC is released with a lag. For the most
  up-to-date recession dating, cross-check with the NBER website.
- **HP filter criticism**: The HP filter has known endpoint and spurious-cycle issues.
  Use the Hamilton (2018) filter (`statsmodels.tsa.filters.filtertools`) as an
  alternative for robustness.

```python
# Quick sanity check: compare fredapi YoY vs manual calculation
cpi_pc1     = get_fred_series("CPIAUCSL", start="2015-01-01", units="pc1")
cpi_level   = get_fred_series("CPIAUCSL", start="2015-01-01", units="lin")
cpi_manual  = calculate_growth_rates(cpi_level, freq="YoY")

comparison = pd.DataFrame({"FRED_pc1": cpi_pc1, "manual_YoY": cpi_manual}).dropna()
max_diff = (comparison["FRED_pc1"] - comparison["manual_YoY"]).abs().max()
print(f"Max difference between FRED pc1 and manual YoY: {max_diff:.6f} pp")
```

---

## 8. Reference: Key FRED Series IDs

| Variable | Series ID | Frequency | Unit |
|---|---|---|---|
| Real GDP | GDPC1 | Quarterly | Bil. Chained 2017 $ |
| Unemployment Rate | UNRATE | Monthly | % |
| CPI All Items | CPIAUCSL | Monthly | Index 1982-84=100 |
| Core CPI | CPILFESL | Monthly | Index 1982-84=100 |
| PCE Price Index | PCEPI | Monthly | Index 2017=100 |
| PPI All Commodities | PPIACO | Monthly | Index 1982=100 |
| Fed Funds Rate | FEDFUNDS | Monthly | % |
| 10Y Treasury | DGS10 | Daily | % |
| 10Y-2Y Spread | T10Y2Y | Daily | % |
| 10Y-3M Spread | T10Y3M | Daily | % |
| NBER Recession | USREC | Monthly | 0/1 |
| Industrial Production | INDPRO | Monthly | Index 2017=100 |
| M2 Money Supply | M2SL | Monthly | Bil. $ |
