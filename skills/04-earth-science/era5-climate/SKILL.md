---
name: era5-climate
description: >
  Download and analyze ERA5 reanalysis climate data via the Copernicus CDS API,
  compute anomalies, trend analysis, and produce publication-quality climate maps.
tags:
  - climate
  - era5
  - reanalysis
  - xarray
  - cartopy
  - mann-kendall
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
  - cdsapi>=0.6.1
  - xarray>=2023.6.0
  - numpy>=1.24.0
  - scipy>=1.11.0
  - matplotlib>=3.7.0
  - cartopy>=0.22.0
  - pymannkendall>=1.4.3
  - netCDF4>=1.6.4
  - cfgrib>=0.9.10
  - pandas>=2.0.0
last_updated: "2026-03-17"
---

# ERA5 Climate Analysis Skill

This skill covers downloading ERA5 reanalysis data from the Copernicus Climate Data Store (CDS),
processing large NetCDF datasets with xarray, computing climate anomalies against a reference
baseline, detecting long-term trends via the Mann-Kendall test, and producing publication-quality
maps with Cartopy.

ERA5 is the fifth generation ECMWF atmospheric reanalysis of the global climate, providing hourly
estimates of atmospheric, land, and oceanic climate variables from 1940 to near-present at roughly
31 km horizontal resolution.

---

## Prerequisites

### CDS API Key Setup

The Copernicus Climate Data Store requires registration and an API key. After registering at
https://cds.climate.copernicus.eu, place your credentials in `~/.cdsapirc`:

```ini
# ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
verify: 1
```

Replace `YOUR_UID` and `YOUR_API_KEY` with your actual CDS user ID and API key from your CDS
account profile page. The file must be readable only by you (`chmod 600 ~/.cdsapirc`).

### Install Dependencies

```bash
pip install cdsapi xarray numpy scipy matplotlib cartopy pymannkendall netCDF4 cfgrib pandas
# On conda environments, cartopy is easier to install via conda:
# conda install -c conda-forge cartopy
```

---

## Core Functions

```python
"""
era5_climate.py
---------------
Core utilities for ERA5 reanalysis data download, anomaly computation,
trend analysis, and visualization.
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pymannkendall as mk

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 1. Data Download
# ---------------------------------------------------------------------------

def download_era5_monthly(
    variable: str,
    years: List[int],
    region: Optional[List[float]] = None,
    pressure_level: Optional[int] = None,
    output_dir: str = "era5_data",
) -> Path:
    """
    Download ERA5 monthly averaged reanalysis data from the Copernicus CDS.

    Parameters
    ----------
    variable : str
        ERA5 variable short name, e.g. '2m_temperature', 'total_precipitation',
        'mean_sea_level_pressure'.
    years : list of int
        Years to download, e.g. list(range(1979, 2024)).
    region : list of float, optional
        Bounding box [north, west, south, east] in degrees. None means global.
    pressure_level : int, optional
        Pressure level in hPa for pressure-level variables (e.g. 850, 500).
        Leave None for single-level variables.
    output_dir : str
        Directory in which to save the downloaded NetCDF file.

    Returns
    -------
    Path
        Path to the downloaded NetCDF file.

    Notes
    -----
    Large requests (many years, global domain) are split into 5-year chunks
    automatically to stay within CDS queue limits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    region_tag = "global" if region is None else f"{region[0]}N{region[1]}E"
    years_tag = f"{min(years)}-{max(years)}"
    filename = output_dir / f"era5_{variable}_{years_tag}_{region_tag}.nc"

    if filename.exists():
        print(f"[cache] {filename} already exists, skipping download.")
        return filename

    c = cdsapi.Client()

    dataset = "reanalysis-era5-single-levels-monthly-means"
    product_type = "monthly_averaged_reanalysis"

    request = {
        "product_type": product_type,
        "variable": variable,
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "format": "netcdf",
    }

    if region is not None:
        request["area"] = region  # [N, W, S, E]

    if pressure_level is not None:
        dataset = "reanalysis-era5-pressure-levels-monthly-means"
        request["pressure_level"] = str(pressure_level)

    print(f"Submitting CDS request for {variable} ({years_tag}) ...")
    c.retrieve(dataset, request, str(filename))
    print(f"Downloaded: {filename}")
    return filename


# ---------------------------------------------------------------------------
# 2. Anomaly Computation
# ---------------------------------------------------------------------------

def compute_anomaly(
    da: xr.DataArray,
    baseline_years: Tuple[int, int] = (1981, 2010),
) -> xr.DataArray:
    """
    Compute monthly climate anomalies relative to a climatological baseline.

    For each calendar month (Jan–Dec) the mean over ``baseline_years`` is
    subtracted from every time step in that month, yielding anomalies.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with a 'time' dimension encoded as ``np.datetime64``.
    baseline_years : tuple of int
        Inclusive start and end year of the reference period.

    Returns
    -------
    xr.DataArray
        Anomaly DataArray with the same shape as ``da``.

    Examples
    --------
    >>> ds = xr.open_dataset("era5_2m_temperature_1979-2023_global.nc")
    >>> t2m = ds["t2m"]
    >>> anom = compute_anomaly(t2m, baseline_years=(1991, 2020))
    """
    start, end = baseline_years
    baseline = da.sel(time=da.time.dt.year.isin(range(start, end + 1)))
    climatology = baseline.groupby("time.month").mean("time")
    anomaly = da.groupby("time.month") - climatology
    anomaly.attrs = da.attrs.copy()
    anomaly.attrs["long_name"] = da.attrs.get("long_name", da.name) + " anomaly"
    return anomaly


# ---------------------------------------------------------------------------
# 3. Trend Analysis
# ---------------------------------------------------------------------------

def calculate_trend(
    da: xr.DataArray,
    dim: str = "time",
) -> Dict[str, xr.DataArray]:
    """
    Apply the Mann-Kendall trend test and Theil-Sen slope estimation
    pixel-by-pixel to a spatial DataArray.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dimensions (time, lat, lon) or (time,).
    dim : str
        Name of the time dimension.

    Returns
    -------
    dict with keys:
        - 'slope'  : Theil-Sen slope per time step (same spatial dims as da)
        - 'p_value': Mann-Kendall p-value at each grid point
        - 'trend'  : string classification ('increasing', 'decreasing', 'no trend')

    Notes
    -----
    For large grids this can be slow; consider coarsening first or parallelising
    with dask.
    """

    def _mk_pixel(ts: np.ndarray) -> Tuple[float, float]:
        """Run MK test on a 1-D time series, returning (slope, p_value)."""
        valid = ts[~np.isnan(ts)]
        if len(valid) < 4:
            return np.nan, np.nan
        result = mk.original_test(valid)
        return result.slope, result.p

    spatial_dims = [d for d in da.dims if d != dim]

    if not spatial_dims:
        # 1-D time series
        result = mk.original_test(da.values[~np.isnan(da.values)])
        return {
            "slope": float(result.slope),
            "p_value": float(result.p),
            "trend": result.trend,
        }

    # Vectorised apply over spatial dimensions
    slope_vals = np.full(da.shape[1:], np.nan)
    pval_vals = np.full(da.shape[1:], np.nan)

    data_np = da.values  # (time, ...)
    flat = data_np.reshape(data_np.shape[0], -1)

    slopes_flat = np.full(flat.shape[1], np.nan)
    pvals_flat = np.full(flat.shape[1], np.nan)

    for i in range(flat.shape[1]):
        s, p = _mk_pixel(flat[:, i])
        slopes_flat[i] = s
        pvals_flat[i] = p

    slope_vals = slopes_flat.reshape(data_np.shape[1:])
    pval_vals = pvals_flat.reshape(data_np.shape[1:])

    coords = {d: da.coords[d] for d in spatial_dims}
    slope_da = xr.DataArray(slope_vals, dims=spatial_dims, coords=coords,
                            attrs={"long_name": "Theil-Sen slope", "units": f"{da.attrs.get('units', '')}/step"})
    pval_da = xr.DataArray(pval_vals, dims=spatial_dims, coords=coords,
                           attrs={"long_name": "Mann-Kendall p-value"})
    return {"slope": slope_da, "p_value": pval_da}


# ---------------------------------------------------------------------------
# 4. Area-Weighted Spatial Average
# ---------------------------------------------------------------------------

def area_weighted_mean(da: xr.DataArray, lat_dim: str = "latitude") -> xr.DataArray:
    """
    Compute the cosine-latitude area-weighted spatial mean.

    Parameters
    ----------
    da : xr.DataArray
        DataArray containing a latitude dimension.
    lat_dim : str
        Name of the latitude dimension.

    Returns
    -------
    xr.DataArray
        Time series of the area-weighted spatial mean.
    """
    weights = np.cos(np.deg2rad(da[lat_dim]))
    weights = weights / weights.sum()
    return (da * weights).sum([lat_dim, "longitude"])


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def plot_climate_map(
    da: xr.DataArray,
    projection: str = "Robinson",
    title: str = "",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    units: Optional[str] = None,
    stipple_mask: Optional[xr.DataArray] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a 2-D climate field on a global or regional map using Cartopy.

    Parameters
    ----------
    da : xr.DataArray
        2-D DataArray with latitude and longitude dimensions.
        Must be a single time step (already averaged/selected).
    projection : str
        Cartopy projection name: 'Robinson', 'PlateCarree', 'Mollweide',
        'LambertConformal', etc.
    title : str
        Map title.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Color scale limits. Defaults to symmetric about zero for anomaly data.
    units : str, optional
        Units label for the colorbar. Falls back to da.attrs['units'].
    stipple_mask : xr.DataArray, optional
        Boolean DataArray (True = significant). Significant grid cells are
        stippled with dots.
    output_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    matplotlib Figure
    """
    proj_map = {
        "Robinson": ccrs.Robinson(),
        "PlateCarree": ccrs.PlateCarree(),
        "Mollweide": ccrs.Mollweide(),
        "NorthPolarStereo": ccrs.NorthPolarStereo(),
        "SouthPolarStereo": ccrs.SouthPolarStereo(),
    }
    proj = proj_map.get(projection, ccrs.Robinson())

    fig, ax = plt.subplots(1, 1, figsize=(14, 7),
                           subplot_kw={"projection": proj})
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)

    lons = da["longitude"].values
    lats = da["latitude"].values
    data = da.values

    if vmin is None and vmax is None:
        absmax = np.nanpercentile(np.abs(data), 98)
        vmin, vmax = -absmax, absmax

    im = ax.pcolormesh(
        lons, lats, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto",
    )

    if stipple_mask is not None:
        sig_lons, sig_lats = np.meshgrid(lons, lats)
        mask = stipple_mask.values.astype(bool)
        ax.scatter(
            sig_lons[mask], sig_lats[mask],
            transform=ccrs.PlateCarree(),
            s=0.5, c="black", alpha=0.4, linewidths=0,
        )

    units_label = units or da.attrs.get("units", "")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        pad=0.04, shrink=0.7, aspect=40)
    cbar.set_label(units_label, fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig
```

---

## Example 1: Global Surface Temperature Anomaly Map (1979–2023)

This example downloads ERA5 2-metre temperature data, computes anomalies against the
1991–2020 reference period, and plots the mean anomaly for 2016 (a strong El Niño year).

```python
"""
example_temperature_anomaly.py
-------------------------------
Global surface temperature anomaly relative to 1991-2020 climatology.
Reproduces a figure similar to NOAA/WMO annual climate state reports.
"""

import xarray as xr
import matplotlib.pyplot as plt
from era5_climate import (
    download_era5_monthly,
    compute_anomaly,
    area_weighted_mean,
    plot_climate_map,
)

# ---- 1. Download data (skip if already cached) ----
nc_file = download_era5_monthly(
    variable="2m_temperature",
    years=list(range(1979, 2024)),
    region=None,           # global
    output_dir="era5_data",
)

# ---- 2. Load and unit-convert K -> °C ----
ds = xr.open_dataset(nc_file)
t2m = ds["t2m"] - 273.15
t2m.attrs["units"] = "°C"
t2m.attrs["long_name"] = "2-metre temperature"

# ---- 3. Compute monthly anomalies ----
anom = compute_anomaly(t2m, baseline_years=(1991, 2020))

# ---- 4. Annual mean anomaly time series (global) ----
gmt = area_weighted_mean(anom).resample(time="1Y").mean()

fig, ax = plt.subplots(figsize=(12, 4))
years_ts = gmt.time.dt.year.values
ax.bar(years_ts, gmt.values, color=["firebrick" if v > 0 else "steelblue" for v in gmt.values])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature anomaly (°C)")
ax.set_title("Global Mean Surface Temperature Anomaly 1979–2023\n(relative to 1991–2020)")
fig.tight_layout()
fig.savefig("gmt_anomaly_timeseries.png", dpi=150)
print("Saved: gmt_anomaly_timeseries.png")

# ---- 5. Spatial anomaly map for the year 2016 ----
anom_2016 = anom.sel(time=anom.time.dt.year == 2016).mean("time")

fig_map = plot_climate_map(
    anom_2016,
    projection="Robinson",
    title="ERA5 Surface Temperature Anomaly – Annual Mean 2016\n(relative to 1991–2020)",
    cmap="RdBu_r",
    units="°C",
    output_path="t2m_anomaly_2016.png",
)
plt.show()

# ---- 6. Quick summary statistics ----
print(f"\nWarmest year: {years_ts[gmt.values.argmax()]} "
      f"(+{gmt.values.max():.3f} °C)")
print(f"Coldest year: {years_ts[gmt.values.argmin()]} "
      f"({gmt.values.min():.3f} °C)")
print(f"Long-term trend: {(gmt.values[-1] - gmt.values[0]):.2f} °C over the period")
```

---

## Example 2: Precipitation Trend Analysis for Europe

This example downloads ERA5 total precipitation over Europe, computes grid-point
Mann-Kendall trends, and overlays significance stippling on the trend map.

```python
"""
example_precipitation_trend.py
--------------------------------
Detect statistically significant precipitation trends (1979-2023) over Europe
using the Mann-Kendall test and Theil-Sen slope estimation.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from era5_climate import (
    download_era5_monthly,
    calculate_trend,
    plot_climate_map,
)

# ---- 1. Download ERA5 total precipitation for Europe ----
# region = [north, west, south, east]
nc_file = download_era5_monthly(
    variable="total_precipitation",
    years=list(range(1979, 2024)),
    region=[72.0, -25.0, 25.0, 45.0],   # Europe
    output_dir="era5_data",
)

# ---- 2. Load and convert m/month -> mm/month ----
ds = xr.open_dataset(nc_file)
tp = ds["tp"] * 1000.0
tp.attrs["units"] = "mm/month"
tp.attrs["long_name"] = "Total precipitation"

# ---- 3. Compute annual totals (sum over months within each year) ----
tp_annual = tp.resample(time="1Y").sum("time")

# ---- 4. Grid-point trend analysis ----
print("Running Mann-Kendall test (this may take a few minutes for large grids) ...")
trend_results = calculate_trend(tp_annual, dim="time")
slope = trend_results["slope"]           # mm/year per year
pval  = trend_results["p_value"]

# Significance mask at 95% confidence
sig_mask = pval < 0.05

# ---- 5. Plot trend map with significance stippling ----
fig_trend = plot_climate_map(
    slope * 10,                          # scale to mm/decade
    projection="PlateCarree",
    title="ERA5 Annual Precipitation Trend 1979–2023 (Europe)\n"
          "Stippled: p < 0.05 (Mann-Kendall)",
    cmap="BrBG",
    units="mm / decade",
    stipple_mask=sig_mask,
    output_path="europe_precip_trend.png",
)
plt.show()

# ---- 6. Regional averages: Mediterranean vs Northern Europe ----
regions = {
    "Mediterranean": {"lat": slice(47, 30), "lon": slice(-10, 40)},
    "Northern Europe": {"lat": slice(72, 55), "lon": slice(-10, 40)},
}

print("\nRegional precipitation trends:")
print(f"{'Region':<20} {'Trend (mm/decade)':>18} {'MK p-value':>12}")
print("-" * 52)
for name, sel in regions.items():
    sub = tp_annual.sel(latitude=sel["lat"], longitude=sel["lon"])
    weights = np.cos(np.deg2rad(sub["latitude"]))
    ts = (sub * weights).sum(["latitude", "longitude"]) / weights.sum()
    res = calculate_trend(ts, dim="time")
    print(f"{name:<20} {res['slope']*10:>18.2f} {res['p_value']:>12.4f}")

# ---- 7. CMIP6 comparison stub ----
def compare_with_cmip6(era5_slope: xr.DataArray, cmip6_nc_path: str) -> None:
    """
    Overlay ERA5 trend with a CMIP6 multi-model mean trend for visual comparison.
    Load a pre-processed CMIP6 trend file and plot side-by-side.
    """
    cmip6_ds = xr.open_dataset(cmip6_nc_path)
    cmip6_slope = cmip6_ds["pr_trend"]   # adjust variable name as needed

    fig, axes = plt.subplots(1, 2, figsize=(18, 6),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    datasets = [(era5_slope * 10, "ERA5 (1979–2023)"),
                (cmip6_slope,    "CMIP6 MMM (historical)")]
    for ax, (data, label) in zip(axes, datasets):
        ax.set_extent([-25, 45, 25, 72])
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        im = ax.pcolormesh(data["longitude"], data["latitude"], data.values,
                           transform=ccrs.PlateCarree(), cmap="BrBG",
                           vmin=-30, vmax=30, shading="auto")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, orientation="horizontal",
                     pad=0.05, label="mm/decade")
    fig.suptitle("Precipitation Trend Comparison: ERA5 vs CMIP6", fontsize=14)
    fig.savefig("era5_vs_cmip6_precip_trend.png", dpi=150, bbox_inches="tight")
    plt.show()

# Uncomment the line below once you have a CMIP6 trend file available:
# compare_with_cmip6(slope, "cmip6_precip_trend_europe.nc")
```

---

## Extreme Event Detection

```python
"""
extreme_events.py
-----------------
Detect hot days and heavy precipitation events using percentile thresholds.
"""

import numpy as np
import xarray as xr


def detect_extremes(
    da: xr.DataArray,
    percentile: float = 95.0,
    baseline_years: tuple = (1981, 2010),
    extreme_type: str = "upper",
) -> xr.DataArray:
    """
    Flag time steps exceeding (or falling below) a percentile threshold
    computed from a reference period.

    Parameters
    ----------
    da : xr.DataArray
        Daily or monthly DataArray (time, lat, lon).
    percentile : float
        Threshold percentile (e.g. 95 for upper tail, 5 for lower tail).
    baseline_years : tuple
        Reference period for computing percentile thresholds.
    extreme_type : str
        'upper' (values above threshold) or 'lower' (values below threshold).

    Returns
    -------
    xr.DataArray
        Boolean DataArray; True where the condition is met.
    """
    start, end = baseline_years
    baseline = da.sel(time=da.time.dt.year.isin(range(start, end + 1)))
    threshold = baseline.quantile(percentile / 100.0, dim="time")

    if extreme_type == "upper":
        return da > threshold
    return da < threshold


def count_extreme_days_per_year(
    extreme_mask: xr.DataArray,
) -> xr.DataArray:
    """Count the number of extreme days per calendar year at each grid point."""
    return extreme_mask.resample(time="1Y").sum("time")


# Usage example
if __name__ == "__main__":
    ds = xr.open_dataset("era5_data/era5_2m_temperature_1979-2023_global.nc")
    t2m = ds["t2m"] - 273.15

    hot_days = detect_extremes(t2m, percentile=95, baseline_years=(1981, 2010))
    hot_days_per_year = count_extreme_days_per_year(hot_days)

    # Plot trend in hot-day frequency for 2000-2023
    hd_trend = hot_days_per_year.sel(
        time=hot_days_per_year.time.dt.year >= 2000
    ).mean(["latitude", "longitude"])
    print("Global mean hot days per year (2000-2023):")
    for yr, val in zip(hd_trend.time.dt.year.values, hd_trend.values):
        print(f"  {yr}: {val:.1f} days")
```

---

## Tips and Best Practices

- **Request size**: CDS has a 100,000-field limit per request. Split multi-year global requests
  into 5-year chunks if you hit errors.
- **Caching**: Always check for existing files before submitting a new CDS request. Downloads
  can take minutes to hours depending on queue length.
- **Chunking with Dask**: For very large grids, open datasets with `chunks={"time": 12}` to
  enable lazy, out-of-core processing.
- **Calendar handling**: ERA5 uses proleptic Gregorian calendar. Use `cftime_range` when
  selecting dates if xarray raises calendar errors.
- **Unit conventions**: ERA5 precipitation is in metres per time step. Always multiply by 1000
  to get mm. Temperature is in Kelvin; subtract 273.15 for Celsius.
- **Pressure-level data**: Use `reanalysis-era5-pressure-levels-monthly-means` dataset and
  supply `pressure_level` parameter. Common levels: 850, 700, 500, 250, 100 hPa.
- **Reproducibility**: Pin library versions and store the CDS request dict alongside the data.

---

## References

- Hersbach et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146, 1999–2049.
- Kendall, M.G. (1975). *Rank Correlation Methods*. Griffin, London.
- Sen, P.K. (1968). Estimates of the regression coefficient based on Kendall's tau.
  *JASA*, 63, 1379–1389.
- CDS documentation: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means
