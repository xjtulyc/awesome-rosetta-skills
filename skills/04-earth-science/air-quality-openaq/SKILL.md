---
name: air-quality-openaq
description: >
  Use this Skill to access OpenAQ air quality data (PM2.5, O3, NO2), compute
  spatial interpolation, health exposure indices, and city comparisons.
tags:
  - earth-science
  - air-quality
  - openaq
  - environmental-health
  - spatial-interpolation
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
    - openaq>=2.0
    - pandas>=2.0
    - plotly>=5.15
    - pykrige>=1.7
    - matplotlib>=3.7
    - numpy>=1.24
    - geopandas>=0.14
    - requests>=2.31
last_updated: "2026-03-17"
status: "stable"
---

# Air Quality Analysis with OpenAQ

> **One-line summary**: Download and analyze PM2.5/PM10/O3/NO2 air quality time series from OpenAQ, compute kriging interpolation, health exposure indices, and city comparisons.

---

## When to Use This Skill

- When accessing ground-level air quality measurements globally
- When comparing air quality between cities or countries
- When computing AQI (Air Quality Index) from raw pollutant data
- When interpolating sparse station data to spatial grids (kriging)
- When estimating population exposure to PM2.5 exceedances
- When analyzing seasonal patterns and pollution episodes

**Trigger keywords**: OpenAQ, PM2.5, PM10, NO2, O3, air quality index, AQI, kriging interpolation, pollution, health exposure

---

## Background & Key Concepts

### OpenAQ Platform

OpenAQ aggregates real-time and historical air quality data from ~30,000 monitoring stations in 100+ countries. Data includes PM2.5, PM10, NO2, O3, SO2, CO, and BC.

### WHO Air Quality Guidelines (2021)

| Pollutant | Annual mean | 24h mean |
|:----------|:------------|:---------|
| PM2.5 | 5 µg/m³ | 15 µg/m³ |
| PM10 | 15 µg/m³ | 45 µg/m³ |
| NO2 | 10 µg/m³ | 25 µg/m³ |
| O3 | — | 100 µg/m³ (8h) |

### Air Quality Index (US EPA)

$$
\text{AQI} = \frac{I_{hi} - I_{lo}}{C_{hi} - C_{lo}} \times (C - C_{lo}) + I_{lo}
$$

AQI categories: Good (0-50), Moderate (51-100), Unhealthy for Sensitive Groups (101-150), Unhealthy (151-200), Very Unhealthy (201-300), Hazardous (301+).

### Ordinary Kriging

Spatial interpolation assuming stationary covariance:

$$
Z(x_0) = \sum_{i=1}^n \lambda_i Z(x_i), \quad \sum_i \lambda_i = 1
$$

Variogram model: $\gamma(h) = C_0 + C_1 (1 - e^{-h/a})$ (exponential).

---

## Environment Setup

### Install Dependencies

```bash
pip install openaq>=2.0 pandas>=2.0 plotly>=5.15 pykrige>=1.7 \
            matplotlib>=3.7 numpy>=1.24 geopandas>=0.14 requests>=2.31
```

### API Key Setup (OpenAQ v3)

```bash
# Register at https://openaq.org and get an API key
export OPENAQ_API_KEY="<paste-your-key>"
```

```python
import os
OPENAQ_KEY = os.getenv("OPENAQ_API_KEY", "")
if not OPENAQ_KEY:
    print("Warning: OPENAQ_API_KEY not set; using unauthenticated access (rate-limited)")
```

### Verify Installation

```python
import requests, pandas as pd
resp = requests.get("https://api.openaq.org/v3/parameters", timeout=10)
params = pd.DataFrame(resp.json()["results"])
print(f"OpenAQ parameters available: {len(params)}")
print(params[["name", "displayName", "units"]].head(6).to_string(index=False))
```

---

## Core Workflow

### Step 1: Query Locations and Download Data

```python
import requests
import pandas as pd
import numpy as np
import os

OPENAQ_KEY = os.getenv("OPENAQ_API_KEY", "")
HEADERS = {"X-API-Key": OPENAQ_KEY} if OPENAQ_KEY else {}

def get_locations(city, country="CN", parameter="pm25", limit=50):
    """
    Get monitoring station locations for a city/country.

    Returns
    -------
    pd.DataFrame with columns: id, name, latitude, longitude, lastUpdated
    """
    url = "https://api.openaq.org/v3/locations"
    params = {
        "city": city,
        "country": country,
        "parameters_name": parameter,
        "limit": limit,
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("results", [])
    if not data:
        print(f"No stations found for {city}, {country}")
        return pd.DataFrame()

    records = []
    for loc in data:
        coords = loc.get("coordinates", {})
        records.append({
            "id": loc["id"],
            "name": loc.get("name", "Unknown"),
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "lastUpdated": loc.get("lastUpdated"),
        })
    df = pd.DataFrame(records).dropna(subset=["latitude", "longitude"])
    print(f"Found {len(df)} stations in {city}")
    return df

def get_measurements(location_id, parameter="pm25",
                     date_from="2024-01-01", date_to="2024-12-31",
                     limit=1000):
    """
    Download hourly measurements for one station.

    Returns
    -------
    pd.DataFrame with datetime index and value column
    """
    url = f"https://api.openaq.org/v3/locations/{location_id}/measurements"
    params = {
        "parameters_name": parameter,
        "date_from": date_from,
        "date_to": date_to,
        "limit": limit,
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return pd.DataFrame()

    results = resp.json().get("results", [])
    records = [{"datetime": r["period"]["datetimeFrom"]["local"],
                "value": r["value"], "unit": r.get("parameter", {}).get("units", "")}
               for r in results]
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"] >= 0]
    return df

# Example (using simulated data if API unavailable)
# locations = get_locations("Beijing", country="CN", parameter="pm25")
# if not locations.empty:
#     station_id = locations.iloc[0]["id"]
#     pm25_data = get_measurements(station_id, parameter="pm25")

# Simulate realistic PM2.5 time series for demonstration
np.random.seed(42)
dates = pd.date_range("2024-01-01", "2024-12-31", freq="h")
# Seasonal pattern: higher in winter
seasonal = 20 + 30 * np.cos(2 * np.pi * (dates.dayofyear - 30) / 365)
diurnal = 5 * np.sin(2 * np.pi * dates.hour / 24)
noise = np.random.exponential(10, len(dates))
pm25_sim = pd.DataFrame({
    "value": np.maximum(0, seasonal + diurnal + noise)
}, index=dates)
print(f"Simulated PM2.5 data: {len(pm25_sim)} hourly records")
print(pm25_sim["value"].describe().round(2))
```

### Step 2: Time Series Analysis and AQI Computation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pm25_to_aqi(pm25):
    """
    Convert PM2.5 (µg/m³) to US EPA AQI.
    Breakpoints: https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html
    """
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    for C_lo, C_hi, I_lo, I_hi in breakpoints:
        if C_lo <= pm25 <= C_hi:
            return int((I_hi - I_lo) / (C_hi - C_lo) * (pm25 - C_lo) + I_lo)
    return 500  # beyond scale

AQI_CATEGORIES = {
    (0, 50): "Good",
    (51, 100): "Moderate",
    (101, 150): "Unhealthy (Sensitive)",
    (151, 200): "Unhealthy",
    (201, 300): "Very Unhealthy",
    (301, 500): "Hazardous",
}
AQI_COLORS = ["green", "yellow", "orange", "red", "purple", "maroon"]

def classify_aqi(aqi):
    for (lo, hi), cat in AQI_CATEGORIES.items():
        if lo <= aqi <= hi:
            return cat
    return "Hazardous"

# Compute AQI and daily means
pm25_sim["aqi"] = pm25_sim["value"].apply(pm25_to_aqi)
daily = pm25_sim.resample("D").mean()
daily["category"] = daily["aqi"].apply(lambda x: classify_aqi(x) if not np.isnan(x) else "Unknown")

print("Days by AQI category:")
print(daily["category"].value_counts())
print(f"\nAnnual mean PM2.5: {pm25_sim['value'].mean():.1f} µg/m³ "
      f"(WHO guideline: 5 µg/m³)")
exceedance_days = (daily["value"] > 15).sum()
print(f"Days exceeding WHO 24h limit (15 µg/m³): {exceedance_days}")

# Monthly box plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
monthly = pm25_sim.resample("ME")["value"]
axes[0].boxplot([g[1].values for g in monthly],
                labels=range(1, 13), patch_artist=True)
axes[0].axhline(15, color='r', linestyle='--', label="WHO 24h limit")
axes[0].axhline(5, color='orange', linestyle=':', label="WHO annual limit")
axes[0].set_xlabel("Month"); axes[0].set_ylabel("PM2.5 (µg/m³)")
axes[0].set_title("Monthly PM2.5 Distribution"); axes[0].legend()

axes[1].plot(daily.index, daily["value"], linewidth=0.5, alpha=0.7)
axes[1].axhline(15, color='r', linestyle='--')
axes[1].set_xlabel("Date"); axes[1].set_ylabel("Daily mean PM2.5 (µg/m³)")
axes[1].set_title("Daily Mean PM2.5 Time Series")

plt.tight_layout()
plt.savefig("air_quality_analysis.png", dpi=150)
plt.show()
```

### Step 3: Multi-City Comparison and Kriging

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate multiple city annual mean PM2.5
cities_data = {
    "Beijing":  {"lat": 39.9, "lon": 116.4, "pm25_annual": 35.5, "pop_M": 21.5},
    "Shanghai": {"lat": 31.2, "lon": 121.5, "pm25_annual": 28.1, "pop_M": 24.8},
    "Guangzhou": {"lat": 23.1, "lon": 113.3, "pm25_annual": 22.4, "pop_M": 15.3},
    "London":   {"lat": 51.5, "lon": -0.1,  "pm25_annual": 9.2,  "pop_M": 9.0},
    "Paris":    {"lat": 48.9, "lon": 2.3,   "pm25_annual": 11.0, "pop_M": 11.0},
    "New York": {"lat": 40.7, "lon": -74.0, "pm25_annual": 7.5,  "pop_M": 8.3},
    "Delhi":    {"lat": 28.6, "lon": 77.2,  "pm25_annual": 98.0, "pop_M": 31.0},
}

import pandas as pd
cities_df = pd.DataFrame(cities_data).T.reset_index()
cities_df.columns = ["city", "lat", "lon", "pm25_annual", "pop_M"]
cities_df["who_exceedance"] = cities_df["pm25_annual"] / 5.0  # WHO annual = 5 µg/m³
cities_df["exposed_millions"] = cities_df["pop_M"] * (cities_df["pm25_annual"] > 5)

print("City Air Quality Comparison:")
print(cities_df[["city", "pm25_annual", "who_exceedance"]].to_string(index=False))
print(f"\nTotal people exposed above WHO guideline: {cities_df['exposed_millions'].sum():.1f}M")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ["green" if x <= 5 else "orange" if x <= 15 else "red"
          for x in cities_df["pm25_annual"]]
axes[0].bar(cities_df["city"], cities_df["pm25_annual"], color=colors)
axes[0].axhline(5, color='darkgreen', linestyle='--', linewidth=2, label="WHO annual (5 µg/m³)")
axes[0].axhline(15, color='darkorange', linestyle='--', linewidth=2, label="WHO 24h (15 µg/m³)")
axes[0].set_ylabel("Annual mean PM2.5 (µg/m³)")
axes[0].set_title("City PM2.5 Levels vs. WHO Guidelines")
axes[0].legend(); axes[0].tick_params(axis='x', rotation=30)

# Population bubble map
scatter = axes[1].scatter(cities_df["lon"], cities_df["lat"],
                          s=cities_df["pop_M"] * 10,
                          c=cities_df["pm25_annual"],
                          cmap="RdYlGn_r", vmin=0, vmax=100, alpha=0.8)
plt.colorbar(scatter, ax=axes[1], label="PM2.5 (µg/m³)")
for _, row in cities_df.iterrows():
    axes[1].annotate(row["city"], (row["lon"], row["lat"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
axes[1].set_title("Global PM2.5 (bubble = population)")

plt.tight_layout()
plt.savefig("city_comparison.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Kriging Interpolation

```python
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt

# Simulate station observations in a region
rng = np.random.default_rng(42)
n_stations = 30
lons = rng.uniform(100, 122, n_stations)
lats = rng.uniform(20, 42, n_stations)
pm25_obs = 20 + 80 * np.exp(-0.01 * ((lons - 113)**2 + (lats - 35)**2)) + \
            rng.normal(0, 5, n_stations)

# Ordinary Kriging
OK = OrdinaryKriging(
    lons, lats, pm25_obs,
    variogram_model="exponential",
    enable_plotting=False,
    verbose=False,
)

# Interpolate to regular grid
grid_lons = np.linspace(100, 122, 50)
grid_lats = np.linspace(20, 42, 50)
z_kriged, ss = OK.execute("grid", grid_lons, grid_lats)

print(f"Kriging output range: {z_kriged.min():.1f} – {z_kriged.max():.1f} µg/m³")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im0 = axes[0].pcolormesh(grid_lons, grid_lats, z_kriged, cmap="RdYlGn_r",
                          vmin=0, vmax=100, shading="auto")
plt.colorbar(im0, ax=axes[0], label="PM2.5 (µg/m³)")
axes[0].scatter(lons, lats, c=pm25_obs, cmap="RdYlGn_r", vmin=0, vmax=100,
                edgecolors="k", s=60, linewidths=0.5, label="Stations")
axes[0].set_title("PM2.5 Kriging Interpolation"); axes[0].legend()

im1 = axes[1].pcolormesh(grid_lons, grid_lats, np.sqrt(ss), cmap="Blues", shading="auto")
plt.colorbar(im1, ax=axes[1], label="Kriging std (µg/m³)")
axes[1].set_title("Kriging Uncertainty (std)")

plt.tight_layout()
plt.savefig("kriging_pm25.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Error: HTTP 429 Too Many Requests

**Cause**: OpenAQ API rate limit exceeded.

**Fix**:
```python
import time
for station_id in station_ids:
    data = get_measurements(station_id)
    time.sleep(0.5)  # 0.5s between requests
```

### Issue: Missing data gaps in time series

**Fix**:
```python
# Resample to hourly, fill short gaps
pm25_hourly = pm25_data.resample("h")["value"].mean()
pm25_filled = pm25_hourly.interpolate(method="time", limit=6)  # fill up to 6h gaps
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| openaq | 2.0, 3.0        | v3 API requires API key for full access |
| pykrige | 1.7             | None |

---

## External Resources

### Official Documentation

- [OpenAQ API v3](https://docs.openaq.org)
- [pykrige documentation](https://geostat-framework.readthedocs.io/projects/pykrige/)

### Key Papers

- Martin, R.V. et al. (2019). *No one knows which city has the highest concentration of fine particulate matter*. Atmospheric Environment.

---

## Examples

### Example 1: Annual Trend Analysis

```python
# =============================================
# Multi-year PM2.5 trend with Mann-Kendall test
# =============================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats

# Simulate 10-year annual means (improving trend)
np.random.seed(42)
years = range(2014, 2024)
pm25_annual = np.array([65, 58, 55, 50, 48, 44, 40, 38, 35, 32]) + \
               np.random.normal(0, 2, 10)

# Mann-Kendall trend test
tau, p_value = stats.kendalltau(list(years), pm25_annual)
slope, intercept, r, p_lm, se = stats.linregress(list(years), pm25_annual)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(list(years), pm25_annual, color="steelblue", alpha=0.7, label="Annual PM2.5")
ax.plot(list(years), [slope*y + intercept for y in years], 'r--', linewidth=2,
        label=f"Trend: {slope:.1f} µg/m³/yr (p={p_lm:.3f})")
ax.axhline(5, color='darkgreen', linestyle=':', label="WHO annual guideline")
ax.set_xlabel("Year"); ax.set_ylabel("Annual mean PM2.5 (µg/m³)")
ax.set_title(f"10-Year PM2.5 Trend (Mann-Kendall τ={tau:.3f}, p={p_value:.3f})")
ax.legend(); ax.set_ylim(0)
plt.tight_layout()
plt.savefig("pm25_trend.png", dpi=150)
plt.show()
```

**Interpreting these results**: Negative slope and significant p-value (< 0.05) indicate a statistically significant improving trend. Compare with WHO guidelines to assess remaining health risk.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
