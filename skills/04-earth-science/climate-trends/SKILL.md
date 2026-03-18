---
name: climate-trends
description: >
  Use this Skill for climate trend analysis: Mann-Kendall test, Sen's slope,
  extreme indices (RX1day, R10mm), percentile thresholds, IPCC-style figures.
tags:
  - earth-science
  - climate
  - trend-analysis
  - xclim
  - extreme-events
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
    - xclim>=0.45
    - scipy>=1.11
    - matplotlib>=3.7
    - numpy>=1.24
    - xarray>=2023.6
    - pandas>=2.0
    - cartopy>=0.22
last_updated: "2026-03-17"
status: "stable"
---

# Climate Trend Analysis

> **One-line summary**: Compute climate trends (Mann-Kendall, Sen's slope), extreme precipitation/temperature indices, and produce IPCC AR6-style figures using xclim, scipy, and xarray.

---

## When to Use This Skill

- When testing for statistically significant trends in temperature or precipitation
- When computing ETCCDI climate extreme indices (RX1day, R10mm, TX90p, etc.)
- When creating IPCC AR6-style climate change figures
- When analyzing historical climate data (ERA5, CMIP6, station data)
- When quantifying changes in extreme events across decades
- When comparing model projections with observed trends

**Trigger keywords**: Mann-Kendall trend test, Sen's slope, climate extremes, ETCCDI, RX1day, R10mm, TX90p, precipitation trend, temperature trend, IPCC figure

---

## Background & Key Concepts

### Mann-Kendall Trend Test

Non-parametric test for monotonic trends in time series:

$$
S = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{sgn}(x_j - x_i)
$$

Under $H_0$ (no trend), $S \sim N(0, V(S))$. Reject $H_0$ if $|Z| > z_{\alpha/2}$.

### Sen's Slope Estimator

Robust, non-parametric trend magnitude:

$$
\hat{\beta} = \text{median}\left\{ \frac{x_j - x_i}{j - i} : i < j \right\}
$$

### ETCCDI Climate Extreme Indices

| Index | Definition | Unit |
|:------|:-----------|:-----|
| RX1day | Max 1-day precipitation | mm |
| RX5day | Max 5-day precipitation | mm |
| R10mm | Annual days with precip ≥ 10mm | days |
| R20mm | Annual days with precip ≥ 20mm | days |
| TX90p | Days with Tmax > 90th percentile | % |
| TN10p | Days with Tmin < 10th percentile | % |
| WSDI | Warm spell duration index | days |
| CDD | Consecutive dry days | days |

---

## Environment Setup

### Install Dependencies

```bash
pip install xclim>=0.45 scipy>=1.11 matplotlib>=3.7 numpy>=1.24 \
            xarray>=2023.6 pandas>=2.0 cartopy>=0.22
```

### Verify Installation

```python
import xclim
import xarray as xr
import scipy
import numpy as np

print(f"xclim: {xclim.__version__}")
print(f"xarray: {xr.__version__}")
print(f"scipy: {scipy.__version__}")
# Expected: xclim 0.45+, xarray 2023.6+
```

---

## Core Workflow

### Step 1: Load and Prepare Climate Data

```python
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

def create_synthetic_climate_dataset(n_years=60, seed=42):
    """
    Create a synthetic daily climate dataset (temperature + precipitation).
    Replace with: xr.open_dataset('era5_data.nc') or similar.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1965-01-01", periods=n_years*365, freq="D")
    years = dates.year
    doy = dates.day_of_year

    # Temperature with warming trend + seasonality
    trend_temp = 0.02 * (years - 1965)  # 0.02°C/year
    seasonal_temp = 15 * np.sin(2 * np.pi * (doy - 90) / 365)
    noise_temp = rng.normal(0, 2, len(dates))
    tas = 15 + trend_temp + seasonal_temp + noise_temp  # °C

    # Precipitation with slight increasing trend + seasonality
    trend_pr = 0.5 * (years - 1965) / n_years  # 0.5mm/day trend over all years
    seasonal_pr = 2 + 2 * np.cos(2 * np.pi * (doy - 200) / 365)
    pr = np.maximum(0, seasonal_pr + trend_pr + rng.exponential(3, len(dates)) - 3)

    ds = xr.Dataset({
        "tas": xr.DataArray(tas, dims=["time"],
                            attrs={"units": "degC", "long_name": "Near-surface air temperature"}),
        "pr":  xr.DataArray(pr, dims=["time"],
                            attrs={"units": "mm/day", "long_name": "Precipitation"}),
    }, coords={"time": dates})
    return ds

ds = create_synthetic_climate_dataset()
print(f"Dataset: {len(ds.time)} days ({ds.time.dt.year.min().item()}–{ds.time.dt.year.max().item()})")
print(f"Temperature range: {float(ds.tas.min()):.1f} – {float(ds.tas.max()):.1f} °C")
print(f"Max daily precip: {float(ds.pr.max()):.1f} mm/day")

# Annual aggregates
tas_annual = ds.tas.resample(time="YE").mean()
pr_annual = ds.pr.resample(time="YE").sum()  # annual total
years = tas_annual.time.dt.year.values

print(f"\nAnnual temp: {float(tas_annual.mean()):.2f} ± {float(tas_annual.std()):.2f} °C")
print(f"Annual precip: {float(pr_annual.mean()):.0f} ± {float(pr_annual.std()):.0f} mm/year")
```

### Step 2: Mann-Kendall Trend Test and Sen's Slope

```python
import numpy as np
import scipy.stats as stats
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

def mann_kendall_test(series):
    """
    Mann-Kendall trend test and Sen's slope estimator.

    Returns
    -------
    dict with: tau, p_value, sen_slope, confidence_interval
    """
    x = np.asarray(series)
    n = len(x)

    # Kendall's tau
    tau, p_value = kendalltau(np.arange(n), x)

    # Sen's slope
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    sen_slope = np.median(slopes)

    # 95% CI for Sen's slope
    C_alpha = 1.96 * np.sqrt(n*(n-1)*(2*n+5)/18)
    M1 = int((len(slopes) - C_alpha) / 2)
    M2 = int((len(slopes) + C_alpha) / 2)
    slopes_sorted = sorted(slopes)
    ci_lo = slopes_sorted[max(0, M1)]
    ci_hi = slopes_sorted[min(len(slopes)-1, M2)]

    return {
        "tau": tau,
        "p_value": p_value,
        "sen_slope": sen_slope,
        "ci_95": (ci_lo, ci_hi),
        "significant": p_value < 0.05,
    }

# Compute trends
tas_annual_np = float(tas_annual.mean()) + np.arange(len(tas_annual)) * 0.025 + \
                np.random.normal(0, 0.5, len(tas_annual))  # example annual values

result_temp = mann_kendall_test(tas_annual_np)
print("Temperature trend analysis:")
print(f"  Kendall τ = {result_temp['tau']:.4f}, p = {result_temp['p_value']:.4f}")
print(f"  Sen's slope = {result_temp['sen_slope']:.4f} °C/year (95% CI: {result_temp['ci_95'][0]:.4f} – {result_temp['ci_95'][1]:.4f})")
print(f"  Significant trend: {result_temp['significant']}")

years_arr = np.arange(1965, 1965 + len(tas_annual_np))
trend_line = result_temp['sen_slope'] * (years_arr - years_arr.mean()) + np.mean(tas_annual_np)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(years_arr, tas_annual_np, color="salmon", alpha=0.6, label="Annual mean T")
ax.plot(years_arr, trend_line, 'r-', linewidth=2.5,
        label=f"Sen's slope: {result_temp['sen_slope']*10:.2f} °C/decade")
ax.set_xlabel("Year"); ax.set_ylabel("Temperature (°C)")
ax.set_title(f"Annual Temperature Trend (τ={result_temp['tau']:.3f}, p={result_temp['p_value']:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig("temperature_trend.png", dpi=150)
plt.show()
```

### Step 3: ETCCDI Extreme Indices with xclim

```python
import xclim
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create dataset with xclim-compatible attributes
ds = create_synthetic_climate_dataset()

# Convert units for xclim (expects Kelvin for temperature, kg/m2/s for precip)
tas_K = ds.tas + 273.15
tas_K.attrs["units"] = "K"
pr_kgm2s = ds.pr / 86400  # mm/day → kg/m2/s
pr_kgm2s.attrs["units"] = "kg m-2 s-1"

print("Computing ETCCDI indices...")

# Maximum 1-day precipitation (RX1day)
rx1day = xclim.indices.max_1day_precipitation_amount(pr=pr_kgm2s, freq="YS")
print(f"RX1day range: {float(rx1day.min()):.1f} – {float(rx1day.max()):.1f} mm")

# Days with precipitation >= 10mm
r10mm = xclim.indices.days_over_precip_thresh(
    pr=pr_kgm2s,
    per=xr.DataArray([10/86400], dims=["quantile"],
                     attrs={"units": "kg m-2 s-1"}),
    freq="YS"
)
print(f"R10mm (days ≥ 10mm/day): {float(r10mm.mean()):.1f} days/year")

# Warm days (Tmax > 90th percentile)
# Compute 90th percentile climatology first
tasmax = tas_K + np.abs(np.random.normal(3, 1, len(tas_K)))  # simulate Tmax
tasmax_da = xr.DataArray(tasmax, dims=["time"],
                         coords={"time": ds.time},
                         attrs={"units": "K"})
t90p = xclim.indices.tg90p(tas=tas_K, t90=tas_K.quantile(0.90).values, freq="YS")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
time_years = rx1day.time.dt.year.values

for ax, data, label in zip(axes.flat,
    [rx1day*86400, r10mm, t90p, (tas_K - 273.15).resample(time="YE").mean()],
    ["RX1day (mm)", "R10mm (days)", "TX90p (%)", "Annual mean T (°C)"]):
    arr = data.values.flatten()
    ax.bar(time_years[:len(arr)], arr, alpha=0.7, color="steelblue")
    # Add trend line
    if len(arr) > 3:
        slope, intercept, *_ = np.polyfit(range(len(arr)), arr, 1), None, None
        slope = np.polyfit(range(len(arr)), arr, 1)[0]
        trend = np.polyfit(range(len(arr)), arr, 1)
        ax.plot(time_years[:len(arr)], np.polyval(trend, range(len(arr))),
                'r--', linewidth=1.5, label=f"Trend: {slope:.3f}/yr")
    ax.set_ylabel(label)
    ax.legend(fontsize=8)

plt.suptitle("ETCCDI Climate Extreme Indices (1965–2024)")
plt.tight_layout()
plt.savefig("etccdi_indices.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### IPCC AR6-Style Warming Stripe Figure

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def warming_stripes(annual_temps, start_year=1965, reference_period=(1981, 2010)):
    """
    Create IPCC AR6-style warming stripes.
    """
    temps = np.array(annual_temps)
    years = np.arange(start_year, start_year + len(temps))

    # Compute anomaly relative to reference period
    ref_mask = (years >= reference_period[0]) & (years <= reference_period[1])
    ref_mean = temps[ref_mask].mean()
    anomaly = temps - ref_mean

    # Colormap: blue (cool) → white (neutral) → red (warm)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "warming_stripes",
        ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#ffffff",
         "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"]
    )

    vmin, vmax = -np.abs(anomaly).max(), np.abs(anomaly).max()
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 3))
    for i, (year, anom) in enumerate(zip(years, anomaly)):
        ax.axvspan(year - 0.5, year + 0.5, color=cmap(norm(anom)))

    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.axis("off")
    ax.set_title(f"Temperature Anomaly ({start_year}–{start_year+len(temps)-1})",
                 fontsize=14, fontweight="bold", pad=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.04, pad=0.01, shrink=0.5)
    cbar.set_label("Temperature anomaly (°C)", fontsize=10)

    plt.tight_layout()
    plt.savefig("warming_stripes.png", dpi=200, bbox_inches="tight")
    plt.show()
    return anomaly

rng = np.random.default_rng(42)
n = 60
simulated_temps = (15 + 0.025 * np.arange(n) +
                   rng.normal(0, 0.5, n))
anomalies = warming_stripes(simulated_temps)
print(f"Max anomaly: {anomalies.max():.2f} °C")
```

---

## Troubleshooting

### Error: `xclim.core.utils.ValidationError: units`

**Cause**: Input DataArray units attribute missing or wrong.

**Fix**:
```python
# Always set units attribute before calling xclim
pr_kgm2s.attrs["units"] = "kg m-2 s-1"
tas_K.attrs["units"] = "K"
```

### Issue: Mann-Kendall returns p-value near 1.0 for obvious trend

**Cause**: Short time series (< 10 points) or too much variability.

**Fix**:
```python
# Use scipy's kendalltau with method='auto' for small samples
tau, p = stats.kendalltau(x, y, method='auto')
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| xclim | 0.45, 0.48      | Index API may change between major versions |
| xarray | 2023.6, 2024.3  | None |

---

## External Resources

### Official Documentation

- [xclim documentation](https://xclim.readthedocs.io/)
- [ETCCDI indices definitions](https://www.climdex.org/learn/indices/)
- [IPCC AR6 Technical Summary](https://www.ipcc.ch/report/ar6/wg1/)

### Key Papers

- Zhang, X. et al. (2011). *Indices for monitoring changes in extremes based on daily temperature and precipitation data*. WIREs Climate Change.

---

## Examples

### Example 1: Precipitation Frequency Analysis

```python
# =============================================
# Extreme precipitation frequency analysis
# =============================================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)
n_years = 60
years = np.arange(1965, 1965 + n_years)

# Simulate annual maximum 1-day precipitation (GEV distribution)
# In practice: use RX1day from real data
annual_max = stats.genextreme.rvs(c=0.1, loc=50, scale=15, size=n_years, random_state=42)
annual_max += 0.3 * (years - 1965) / n_years * annual_max  # slight trend

# GEV fit
shape, loc, scale = stats.genextreme.fit(annual_max)
print(f"GEV fit: shape={shape:.3f}, loc={loc:.1f}, scale={scale:.1f}")

# Return levels
return_periods = [2, 5, 10, 25, 50, 100]
exceedance_probs = [1/rp for rp in return_periods]
return_levels = stats.genextreme.isf(exceedance_probs, shape, loc, scale)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(years, annual_max, alpha=0.7, color="steelblue")
slope, intercept, *_ = stats.linregress(years, annual_max)
axes[0].plot(years, slope*years + intercept, 'r--', linewidth=2,
             label=f"Trend: {slope:.2f} mm/yr")
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Annual max 1-day precip (mm)")
axes[0].set_title("Annual Maximum Precipitation"); axes[0].legend()

axes[1].semilogx(return_periods, return_levels, 'bs-', linewidth=2)
axes[1].set_xlabel("Return period (years)")
axes[1].set_ylabel("Return level (mm)")
axes[1].set_title("Precipitation Frequency Curve (GEV)")
axes[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.savefig("precip_frequency.png", dpi=150)
plt.show()

for rp, rl in zip(return_periods, return_levels):
    print(f"  {rp:3d}-year return level: {rl:.1f} mm")
```

**Interpreting these results**: A 100-year return level of X mm means that, on average, this precipitation amount is exceeded once every 100 years. Positive trend in annual maxima suggests increasing extreme precipitation.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
