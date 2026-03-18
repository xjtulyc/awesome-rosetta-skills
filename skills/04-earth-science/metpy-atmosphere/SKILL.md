---
name: metpy-atmosphere
description: >
  Atmospheric analysis with MetPy and Siphon: download NWP data from THREDDS/NOMADS,
  plot skew-T log-P diagrams, compute CAPE/CIN parcel metrics, and build synoptic composites.
tags:
  - meteorology
  - metpy
  - siphon
  - skew-t
  - cape-cin
  - synoptic
version: "1.0.0"
authors:
  - name: "Rosetta Skills Contributors"
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - metpy>=1.5
  - siphon>=0.9
  - xarray>=2023.6
  - matplotlib>=3.7
  - numpy>=1.24
  - pandas>=2.0
  - cartopy>=0.22
last_updated: "2026-03-17"
status: "stable"
---

# MetPy Atmospheric Analysis Skill

This skill covers the full workflow for operational and research-grade atmospheric science
in Python: programmatic data retrieval from UCAR THREDDS and NOAA NOMADS servers using
Siphon, thermodynamic parcel analysis (CAPE, CIN, LCL, LFC, EL) using MetPy, skew-T
log-P diagram generation, wind hodograph construction, and synoptic-scale composite
analysis over gridded model output.

MetPy is the de-facto standard Python library for atmospheric calculations. It handles
unit-aware arrays via Pint and provides a comprehensive set of thermodynamic, kinematic,
and interpolation routines. Siphon acts as the data-access layer that queries THREDDS
Data Server (TDS) catalogs and returns xarray datasets with CF-compliant metadata.

---

## When to Use This Skill

- You need to retrieve real-time or archived NWP model data (GFS, NAM, RAP) programmatically
  without manual download.
- You are producing upper-air soundings or comparing observed radiosondes with model profiles.
- You need convective instability metrics (CAPE, CIN, LIFTED index, K-index) for a case study
  or climatological analysis.
- You want to create publication-quality synoptic maps (500 hPa heights, SLP, moisture flux).
- You are building an automated workflow that runs nightly and ingests the latest model cycle.

---

## Background & Key Concepts

### Skew-T Log-P Diagram

A thermodynamic diagram where temperature is plotted on a skewed x-axis and pressure
decreases logarithmically upward. Dry adiabats, moist adiabats, and saturation mixing-ratio
lines help visualize parcel ascent and atmospheric stability.

### CAPE and CIN

- **CAPE** (Convective Available Potential Energy): the positive buoyancy energy a parcel
  gains ascending from the LFC to the EL. Units are J kg⁻¹. Values > 1000 J kg⁻¹ indicate
  significant severe-weather potential.
- **CIN** (Convective INhibition): the negative area below the LFC that must be overcome
  before free convection begins. Large CIN suppresses convection; small CIN allows it.

### THREDDS / NOMADS

The UCAR THREDDS Data Server hosts model grids and observational datasets accessible via
OPeNDAP, HTTP, and WMS. NOAA NOMADS is the National Operational Model Archive and
Distribution System providing real-time GFS, NAM, and RAP output. Siphon wraps both
services with a Pythonic catalog interface.

### Hodograph

A polar diagram of the wind vector at successive altitude levels. The shape of the hodograph
quantifies wind shear and storm-relative helicity (SRH), key ingredients for supercell
thunderstorm development.

---

## Environment Setup

### Install Dependencies

```bash
# Create and activate a conda environment (recommended for Cartopy)
conda create -n atmos python=3.11
conda activate atmos
conda install -c conda-forge metpy siphon xarray cartopy matplotlib numpy pandas
pip install metpy>=1.5 siphon>=0.9
```

```bash
# Alternatively, pure pip (may require system GEOS/PROJ for Cartopy)
pip install metpy>=1.5 siphon>=0.9 xarray>=2023.6 matplotlib>=3.7 \
            numpy>=1.24 pandas>=2.0 cartopy>=0.22
```

### Verify Installation

```python
import metpy
import siphon
import xarray as xr
import cartopy

print(f"MetPy    {metpy.__version__}")
print(f"Siphon   {siphon.__version__}")
print(f"xarray   {xr.__version__}")
print(f"Cartopy  {cartopy.__version__}")
```

### Optional: NOMADS API Access

NOMADS is public and requires no key. THREDDS catalogs at UCAR are also open. No
authentication is needed for the examples below.

---

## Core Workflow

### Step 1 — Retrieve Sounding Data from THREDDS

```python
"""
step1_siphon_sounding.py
------------------------
Retrieve the latest GFS 0-h analysis profile at a single point via the
UCAR THREDDS server using Siphon, then parse it into a tidy DataFrame.
"""

import os
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

warnings.filterwarnings("ignore")


def fetch_gfs_sounding(
    lat: float,
    lon: float,
    date: datetime | None = None,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch a vertical profile from the GFS 0.25-degree analysis via THREDDS/NCSS.

    Parameters
    ----------
    lat, lon : float
        Station latitude and longitude in decimal degrees.
    date : datetime, optional
        UTC time of the desired model run. Defaults to the most recent 00Z cycle.
    variables : list of str, optional
        List of NCSS variable names. Defaults to temperature, dewpoint, and wind.

    Returns
    -------
    pd.DataFrame
        Columns: pressure [hPa], temperature [degC], dewpoint [degC],
        u_wind [m/s], v_wind [m/s], height [m].
    """
    if date is None:
        now = datetime.now(tz=timezone.utc)
        # Roll back to the most recent 00Z or 12Z cycle
        hour = (now.hour // 12) * 12
        date = now.replace(hour=hour, minute=0, second=0, microsecond=0)

    if variables is None:
        variables = [
            "Temperature_isobaric",
            "Dewpoint_temperature_isobaric",
            "u-component_of_wind_isobaric",
            "v-component_of_wind_isobaric",
            "Geopotential_height_isobaric",
        ]

    catalog_url = (
        "https://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/"
        "Global_0p25deg/catalog.xml"
    )

    print(f"Connecting to THREDDS catalog: {catalog_url}")
    cat = TDSCatalog(catalog_url)
    # Select the latest dataset
    ds_name = sorted(cat.datasets.keys())[-1]
    print(f"Using dataset: {ds_name}")

    ncss = NCSS(cat.datasets[ds_name].access_urls["NetcdfSubset"])
    query = ncss.query()
    query.lonlat_point(lon, lat)
    query.variables(*variables)
    query.vertical_level_range(100, 1050)  # hPa
    query.accept("netcdf4")

    raw = ncss.get_data(query)

    # Parse pressure levels and profile arrays
    pressure = raw.variables["pressure"][:]  # hPa
    temp_k = raw.variables["Temperature_isobaric"][0, :, 0, 0]
    td_k = raw.variables["Dewpoint_temperature_isobaric"][0, :, 0, 0]
    u = raw.variables["u-component_of_wind_isobaric"][0, :, 0, 0]
    v = raw.variables["v-component_of_wind_isobaric"][0, :, 0, 0]
    z = raw.variables["Geopotential_height_isobaric"][0, :, 0, 0]

    df = pd.DataFrame({
        "pressure": pressure,
        "temperature": temp_k - 273.15,
        "dewpoint": td_k - 273.15,
        "u_wind": u,
        "v_wind": v,
        "height": z,
    }).sort_values("pressure", ascending=False).reset_index(drop=True)

    print(f"Retrieved {len(df)} pressure levels for ({lat:.2f}N, {lon:.2f}E)")
    return df


if __name__ == "__main__":
    # Example: profile over central Oklahoma (thunderstorm alley)
    sounding_df = fetch_gfs_sounding(lat=35.5, lon=-97.5)
    print(sounding_df.head(10).to_string(index=False))
    sounding_df.to_csv("gfs_sounding_okc.csv", index=False)
    print("Saved: gfs_sounding_okc.csv")
```

### Step 2 — Skew-T Log-P Diagram and Parcel Analysis

```python
"""
step2_skewt_parcel.py
---------------------
Build a skew-T log-P diagram from a sounding DataFrame and overlay
the parcel ascent path.  Compute CAPE, CIN, LCL, LFC, and EL.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units


def load_sounding(csv_path: str) -> tuple:
    """
    Load a sounding CSV (columns: pressure, temperature, dewpoint, u_wind, v_wind)
    and attach MetPy units.

    Returns
    -------
    tuple of pint Quantity arrays: pressure, temperature, dewpoint, u, v
    """
    df = pd.read_csv(csv_path)

    # Drop levels with missing temperature or dewpoint
    df = df.dropna(subset=["temperature", "dewpoint"]).copy()
    df = df.sort_values("pressure", ascending=False).reset_index(drop=True)

    pressure = df["pressure"].values * units.hPa
    temperature = df["temperature"].values * units.degC
    dewpoint = df["dewpoint"].values * units.degC
    u = df["u_wind"].values * units("m/s")
    v = df["v_wind"].values * units("m/s")

    return pressure, temperature, dewpoint, u, v


def plot_skewt(
    pressure, temperature, dewpoint, u, v,
    title: str = "Skew-T Log-P Diagram",
    output_path: str = "skewt.png",
) -> plt.Figure:
    """
    Produce a complete skew-T log-P diagram with parcel trace and indices.

    The figure includes:
    - Observed temperature and dewpoint profiles
    - Surface-based parcel ascent path
    - CAPE (green shading) and CIN (red shading) areas
    - Wind barbs on the right side
    - Text box with convective indices

    Parameters
    ----------
    pressure, temperature, dewpoint : pint Quantity arrays
        Mandatory profile data.
    u, v : pint Quantity arrays
        Wind components in m/s.
    title : str
        Plot title.
    output_path : str
        File path to save the PNG figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ----- Parcel analysis -----
    parcel_prof = mpcalc.parcel_profile(pressure, temperature[0], dewpoint[0])

    cape, cin = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_prof)
    lcl_p, lcl_t = mpcalc.lcl(pressure[0], temperature[0], dewpoint[0])
    lfc_p, lfc_t = mpcalc.lfc(pressure, temperature, dewpoint)
    el_p, el_t = mpcalc.el(pressure, temperature, dewpoint)

    # ----- Figure layout -----
    fig = plt.figure(figsize=(10, 12))
    skew = SkewT(fig, rotation=45)

    # Background lines
    skew.plot_dry_adiabats(alpha=0.25, colors="orange")
    skew.plot_moist_adiabats(alpha=0.25, colors="green")
    skew.plot_mixing_lines(alpha=0.25, colors="blue")

    # Observed profiles
    skew.plot(pressure, temperature, "r", linewidth=2, label="Temperature")
    skew.plot(pressure, dewpoint, "g", linewidth=2, label="Dewpoint")

    # Parcel trace
    skew.plot(pressure, parcel_prof, "k--", linewidth=1.5, label="Parcel")

    # CAPE / CIN shading
    skew.shade_cape(pressure, temperature, parcel_prof)
    skew.shade_cin(pressure, temperature, parcel_prof, dewpoint)

    # Key levels
    skew.ax.axhline(lcl_p.m, color="purple", linestyle=":", linewidth=1.2, label="LCL")
    if not np.isnan(lfc_p.m):
        skew.ax.axhline(lfc_p.m, color="darkorange", linestyle="-.", linewidth=1.2, label="LFC")
    if not np.isnan(el_p.m):
        skew.ax.axhline(el_p.m, color="cyan", linestyle="-.", linewidth=1.2, label="EL")

    # Wind barbs
    wind_slice = slice(None, None, 5)
    skew.plot_barbs(
        pressure[wind_slice], u[wind_slice], v[wind_slice],
        x_clip_radius=0.12, y_clip_radius=0.12,
    )

    # Indices text box
    indices_text = (
        f"CAPE: {cape.m:.0f} J kg⁻¹\n"
        f"CIN:  {cin.m:.0f} J kg⁻¹\n"
        f"LCL:  {lcl_p.m:.0f} hPa / {lcl_t.m:.1f} °C\n"
        f"LFC:  {lfc_p.m:.0f} hPa\n"
        f"EL:   {el_p.m:.0f} hPa"
    )
    skew.ax.text(
        0.02, 0.98, indices_text,
        transform=skew.ax.transAxes,
        fontsize=9, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    skew.ax.set_xlim(-40, 50)
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.set_title(title, fontsize=13)
    skew.ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"\nConvective indices:")
    print(f"  CAPE = {cape.m:.1f} J/kg")
    print(f"  CIN  = {cin.m:.1f} J/kg")
    print(f"  LCL  = {lcl_p.m:.1f} hPa, {lcl_t.m:.1f} °C")

    return fig


if __name__ == "__main__":
    p, t, td, u, v = load_sounding("gfs_sounding_okc.csv")
    fig = plot_skewt(
        p, t, td, u, v,
        title="GFS Analysis – Central Oklahoma\nSkew-T Log-P",
        output_path="skewt_okc.png",
    )
    plt.show()
```

### Step 3 — Wind Hodograph and Storm-Relative Helicity

```python
"""
step3_hodograph.py
------------------
Construct a wind hodograph and compute storm-relative helicity (SRH)
for the 0–1 km and 0–3 km layers.
"""

import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import Hodograph
from metpy.units import units
import pandas as pd


def plot_hodograph(
    pressure, u, v, height,
    title: str = "Wind Hodograph",
    output_path: str = "hodograph.png",
) -> plt.Figure:
    """
    Plot a wind hodograph colored by altitude layer and annotate SRH values.

    Parameters
    ----------
    pressure : pint Quantity (hPa)
    u, v : pint Quantity (m/s)
    height : pint Quantity (m AGL)
    title : str
    output_path : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Compute SRH for key layers
    srh_1km = mpcalc.storm_relative_helicity(
        height, u, v, depth=1000 * units.m
    )
    srh_3km = mpcalc.storm_relative_helicity(
        height, u, v, depth=3000 * units.m
    )

    # Bulk wind shear 0–6 km
    shear_u, shear_v = mpcalc.bulk_shear(pressure, u, v, depth=6000 * units.m)
    bws = np.sqrt(shear_u**2 + shear_v**2).to("kt")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    h = Hodograph(ax, component_range=80)
    h.add_grid(increment=10)

    # Color-coded by height layer
    intervals = np.array([0, 1000, 3000, 6000, 12000]) * units.m
    colors = ["purple", "red", "darkorange", "gold"]
    h.plot_colormapped(u, v, height, intervals=intervals, colors=colors)

    # Annotate wind speeds at key levels (surface, 850, 700, 500 hPa)
    key_levels = [0, 850, 700, 500]
    p_vals = pressure.m
    for target_p in key_levels:
        idx = np.argmin(np.abs(p_vals - target_p))
        ax.annotate(
            f"{target_p} hPa" if target_p > 0 else "Sfc",
            xy=(u[idx].m, v[idx].m),
            fontsize=7, color="black",
            xytext=(3, 3), textcoords="offset points",
        )

    # Annotations
    info = (
        f"0–1 km SRH: {srh_1km[0].m:.0f} m²/s²\n"
        f"0–3 km SRH: {srh_3km[0].m:.0f} m²/s²\n"
        f"0–6 km BWS: {bws.m:.1f} kt"
    )
    ax.text(
        0.02, 0.98, info,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return fig


if __name__ == "__main__":
    df = pd.read_csv("gfs_sounding_okc.csv").dropna()
    df = df.sort_values("pressure", ascending=False).reset_index(drop=True)

    p = df["pressure"].values * units.hPa
    u = df["u_wind"].values * units("m/s")
    v = df["v_wind"].values * units("m/s")
    z = df["height"].values * units.m

    fig = plot_hodograph(p, u, v, z,
                         title="GFS – OKC Hodograph",
                         output_path="hodograph_okc.png")
    plt.show()
```

### Step 4 — Synoptic Composite Map (500 hPa Heights + Vorticity)

```python
"""
step4_synoptic_map.py
---------------------
Download a GFS 500 hPa analysis grid from NOMADS and plot geopotential
height contours plus absolute vorticity with a Cartopy basemap.
"""

import os
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.units import units

warnings.filterwarnings("ignore")

# NOMADS real-time GFS 0.25-deg via OPeNDAP — no authentication required
# Adjust the cycle date/time as needed; this URL pattern works for recent runs.
NOMADS_BASE = (
    "https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date}/gfs_0p25_00z"
)


def fetch_500hpa_grid(date_str: str) -> xr.Dataset:
    """
    Open GFS 500 hPa geopotential height and wind via OPeNDAP from NOMADS.

    Parameters
    ----------
    date_str : str
        Date in YYYYMMDD format, e.g. '20240601'.

    Returns
    -------
    xr.Dataset
        Dataset with hgtprs (geopotential height) and ugrdprs/vgrdprs (winds).
    """
    url = NOMADS_BASE.format(date=date_str)
    print(f"Opening: {url}")
    ds = xr.open_dataset(url, engine="netcdf4")
    # Select 500 hPa level (lev coordinate in hPa) and first forecast hour
    ds_500 = ds.sel(lev=500, method="nearest").isel(time=0)
    return ds_500


def plot_500hpa_map(
    ds: xr.Dataset,
    extent: list[float] = [-130, -60, 20, 60],
    output_path: str = "500hpa_analysis.png",
) -> plt.Figure:
    """
    Plot 500 hPa geopotential height contours and absolute vorticity fill.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with hgtprs, ugrdprs, vgrdprs, lat, lon variables.
    extent : list of float
        Map extent [lon_min, lon_max, lat_min, lat_max].
    output_path : str
        Output file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    lat = ds["lat"].values
    lon = ds["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    hgt = ds["hgtprs"].values          # geopotential height in meters
    u = ds["ugrdprs"].values * units("m/s")
    v = ds["vgrdprs"].values * units("m/s")

    # Compute absolute vorticity
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    avor = mpcalc.absolute_vorticity(u, v, dx=dx, dy=dy,
                                      latitude=lat2d * units.degrees)
    avor_scaled = avor.m * 1e5  # units of 10^-5 s^-1

    # ---- Figure ----
    proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=35)
    fig, ax = plt.subplots(1, 1, figsize=(14, 9), subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.add_feature(cfeature.STATES, linewidth=0.3, alpha=0.5)

    # Vorticity fill
    vort_levels = np.arange(0, 50, 4)
    cf = ax.contourf(
        lon2d, lat2d, avor_scaled,
        levels=vort_levels, cmap="YlOrRd",
        transform=ccrs.PlateCarree(), extend="max", alpha=0.7,
    )
    plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04,
                 label="Absolute Vorticity (×10⁻⁵ s⁻¹)", shrink=0.7)

    # Height contours
    hgt_levels = np.arange(4800, 6000, 60)
    cs = ax.contour(
        lon2d, lat2d, hgt,
        levels=hgt_levels, colors="black", linewidths=1.0,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(cs, fmt="%d", fontsize=8, inline=True)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title("GFS 500 hPa Geopotential Height (dam) and Absolute Vorticity",
                 fontsize=13, pad=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return fig
```

---

## Advanced Usage

### Composite Soundings: CAPE Climatology Over a Season

```python
"""
advanced_cape_climatology.py
-----------------------------
Iterate over a list of dates, download GFS soundings, compute CAPE/CIN,
and build a monthly climatology DataFrame.
"""

import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import metpy.calc as mpcalc
from metpy.units import units

warnings.filterwarnings("ignore")


def compute_cape_cin_from_df(df: pd.DataFrame) -> dict:
    """
    Compute CAPE and CIN from a sounding DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: pressure [hPa], temperature [°C], dewpoint [°C].

    Returns
    -------
    dict with keys 'cape', 'cin', 'lcl_p', 'lifted_index'.
    """
    df = df.dropna(subset=["temperature", "dewpoint"])
    df = df.sort_values("pressure", ascending=False).reset_index(drop=True)

    p = df["pressure"].values * units.hPa
    t = df["temperature"].values * units.degC
    td = df["dewpoint"].values * units.degC

    parcel = mpcalc.parcel_profile(p, t[0], td[0])
    cape, cin = mpcalc.cape_cin(p, t, td, parcel)
    lcl_p, _ = mpcalc.lcl(p[0], t[0], td[0])
    li = mpcalc.lifted_index(p, t, parcel)

    return {
        "cape": float(cape.m),
        "cin": float(cin.m),
        "lcl_p": float(lcl_p.m),
        "lifted_index": float(li.m),
    }


def seasonal_cape_summary(records: list[dict]) -> pd.DataFrame:
    """
    Build a summary table from a list of per-sounding dicts.

    Each dict must have at least 'date', 'cape', 'cin', 'lifted_index'.
    """
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    monthly = (
        df.groupby("month")[["cape", "cin", "lifted_index"]]
        .agg(["mean", "std", "max"])
        .round(1)
    )
    return monthly


# --- Demonstration with synthetic data (replace with real sounding fetches) ---
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 180  # six months of daily soundings
    start = datetime(2023, 4, 1)
    records = []
    for i in range(n):
        # Synthetic CAPE: higher in summer, lower in spring/autumn
        day = start + timedelta(days=i)
        base_cape = 800 + 600 * np.sin(np.pi * i / 180)
        records.append({
            "date": day,
            "cape": max(0, base_cape + rng.normal(0, 200)),
            "cin": abs(rng.normal(80, 40)),
            "lifted_index": rng.normal(-3 - base_cape / 600, 1),
        })

    summary = seasonal_cape_summary(records)
    print("Monthly CAPE/CIN/LI summary (Apr – Sep 2023):")
    print(summary.to_string())
```

### Custom Parcel: Most-Unstable CAPE

```python
"""
advanced_mucape.py
------------------
Compute Most-Unstable CAPE (MUCAPE) by lifting the parcel from the
most buoyant 100-hPa mixed layer in the lowest 300 hPa.
"""

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units


def most_unstable_cape(pressure, temperature, dewpoint) -> tuple:
    """
    Find the most unstable parcel in the lowest 300 hPa and return its CAPE/CIN.

    Parameters
    ----------
    pressure, temperature, dewpoint : pint Quantity arrays
        Full sounding arrays sorted surface-first (decreasing altitude).

    Returns
    -------
    tuple: (mucape, mucin, parcel_pressure)
        All pint Quantities.
    """
    # Restrict search to lowest 300 hPa
    sfc_p = pressure[0]
    mask = pressure >= (sfc_p - 300 * units.hPa)

    p_layer = pressure[mask]
    t_layer = temperature[mask]
    td_layer = dewpoint[mask]

    best_cape = 0.0 * units("J/kg")
    best_cin = 0.0 * units("J/kg")
    best_idx = 0

    for i in range(len(p_layer)):
        try:
            parcel = mpcalc.parcel_profile(pressure, t_layer[i], td_layer[i])
            c, ci = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel)
            if c > best_cape:
                best_cape = c
                best_cin = ci
                best_idx = i
        except Exception:
            continue

    return best_cape, best_cin, p_layer[best_idx]


if __name__ == "__main__":
    # Example: load the previously saved sounding
    import pandas as pd

    df = pd.read_csv("gfs_sounding_okc.csv").dropna()
    df = df.sort_values("pressure", ascending=False).reset_index(drop=True)

    p = df["pressure"].values * units.hPa
    t = df["temperature"].values * units.degC
    td = df["dewpoint"].values * units.degC

    mucape, mucin, mu_p = most_unstable_cape(p, t, td)
    print(f"MUCAPE = {mucape.m:.1f} J/kg")
    print(f"MUCIN  = {mucin.m:.1f} J/kg")
    print(f"Most unstable parcel lifted from {mu_p.m:.0f} hPa")
```

---

## Troubleshooting

### Connection Errors to THREDDS / NOMADS

The UCAR THREDDS server and NOMADS occasionally time out under heavy load or during
maintenance windows. Strategies:

```python
import time
from siphon.catalog import TDSCatalog

def robust_catalog(url: str, retries: int = 3, wait: float = 5.0) -> TDSCatalog:
    """Retry a TDS catalog fetch up to `retries` times with a delay."""
    for attempt in range(1, retries + 1):
        try:
            return TDSCatalog(url)
        except Exception as exc:
            print(f"Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(wait)
    raise RuntimeError(f"Could not connect to {url} after {retries} attempts.")
```

### Units Mismatch Errors

MetPy raises `pint.errors.DimensionalityError` when incompatible units are combined.
Always attach units immediately after loading data:

```python
# Wrong — will raise error later:
temp_raw = df["temperature"].values        # plain numpy array

# Correct:
temp = df["temperature"].values * units.degC
```

### Parcel Analysis Fails Near the Surface

If `mpcalc.parcel_profile` raises or returns NaN arrays, check that the surface level
(lowest-pressure index) has valid temperature and dewpoint:

```python
print(f"Surface: p={p[0]:.1f}, T={t[0]:.1f}, Td={td[0]:.1f}")
assert not np.isnan(t[0].m), "Surface temperature is NaN"
assert not np.isnan(td[0].m), "Surface dewpoint is NaN"
assert td[0] <= t[0], "Dewpoint exceeds temperature at surface"
```

### Cartopy Installation Issues on Windows

Cartopy requires GEOS and PROJ. On Windows, use conda:

```bash
conda install -c conda-forge cartopy
```

### Large Grid Memory Usage

For global GFS grids (1440 × 721), load only the variables and region you need:

```python
ds = xr.open_dataset(url, engine="netcdf4")
ds_sub = ds.sel(lat=slice(20, 60), lon=slice(230, 300))
```

---

## External Resources

- MetPy documentation: https://unidata.github.io/MetPy/latest/
- MetPy example gallery: https://unidata.github.io/MetPy/latest/examples/index.html
- Siphon documentation: https://unidata.github.io/siphon/latest/
- UCAR THREDDS catalog browser: https://thredds.ucar.edu/thredds/catalog.html
- NOAA NOMADS server: https://nomads.ncep.noaa.gov/
- AMS Glossary of Meteorology: https://glossary.ametsoc.org/wiki/Main_Page
- Bluestein (1992) *Synoptic-Dynamic Meteorology in Midlatitudes*, Oxford University Press
- Doswell & Rasmussen (1994) The effect of hodograph smoothing on derived thermodynamic
  parameters. *Wea. Forecasting*, 9, 576–593.

---

## Examples

### Example 1: Full Sounding Workflow — OKC Tornado Outbreak Day

This example combines Steps 1–3 to reproduce a textbook severe-weather sounding from a
historical tornado outbreak day by querying archived GFS analyses.

```python
"""
example1_tornado_day_sounding.py
---------------------------------
Retrieve, process, and visualize a sounding profile for a tornado-outbreak
scenario: 27 April 2011 (Super Outbreak).  Demonstrates the full MetPy
sounding workflow from data retrieval to skew-T and hodograph figures.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units

warnings.filterwarnings("ignore")

# Synthetic high-CAPE sounding representative of 27 Apr 2011 OKC environment
# Replace with real sounding fetch via step1_siphon_sounding.py for live data.
PRESSURE_HPA = np.array([
    1003, 982, 964, 950, 925, 900, 875, 850, 825, 800, 775, 750,
    725, 700, 675, 650, 625, 600, 575, 550, 525, 500, 475, 450,
    425, 400, 375, 350, 325, 300, 275, 250, 225, 200, 175, 150, 125, 100,
])
TEMPERATURE_C = np.array([
    28.0, 25.5, 23.0, 21.8, 19.5, 17.2, 14.8, 12.5, 10.2, 7.8, 5.4, 3.2,
    1.0, -1.8, -4.5, -7.0, -9.8, -12.5, -15.0, -18.0, -21.5, -25.0, -28.5, -32.0,
    -36.0, -40.0, -44.5, -49.0, -53.5, -58.0, -61.0, -55.0, -51.0, -50.0, -53.0, -56.0, -60.0, -66.0,
])
DEWPOINT_C = np.array([
    22.0, 21.5, 21.0, 20.8, 19.0, 17.0, 14.5, 12.0, 9.0, 5.0, 1.5, -1.0,
    -3.5, -7.0, -11.0, -15.0, -19.5, -23.0, -27.0, -31.0, -35.5, -40.0, -44.0, -48.0,
    -52.0, -56.0, -60.0, -65.0, -68.0, -70.0, -72.0, -70.0, -68.0, -66.0, -66.0, -68.0, -70.0, -75.0,
])
U_WIND_MS = np.array([
    -4, -5, -6, -6, -7, -8, -9, -10, -11, -12, -13, -14,
    -14, -15, -16, -17, -18, -20, -22, -24, -26, -28, -30, -32,
    -33, -34, -36, -38, -40, -42, -43, -44, -45, -44, -42, -40, -38, -35,
], dtype=float)
V_WIND_MS = np.array([
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12,
    13, 14, 15, 15, 16, 16, 17, 17, 18, 18, 18, 17,
    16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3,
], dtype=float)
HEIGHT_M = np.array([
    87, 298, 505, 650, 902, 1160, 1424, 1695, 1973, 2260, 2555, 2859,
    3172, 3493, 3826, 4170, 4527, 4895, 5278, 5675, 6086, 6515, 6960, 7425,
    7910, 8415, 8945, 9500, 10082, 10696, 11343, 12028, 12756, 13534, 14368, 15272, 16267, 17476,
])

p = PRESSURE_HPA * units.hPa
t = TEMPERATURE_C * units.degC
td = DEWPOINT_C * units.degC
u = U_WIND_MS * units("m/s")
v = V_WIND_MS * units("m/s")
z = HEIGHT_M * units.m

# ---- Parcel analysis ----
parcel = mpcalc.parcel_profile(p, t[0], td[0])
cape, cin = mpcalc.cape_cin(p, t, td, parcel)
lcl_p, lcl_t = mpcalc.lcl(p[0], t[0], td[0])
lfc_p, _ = mpcalc.lfc(p, t, td)
el_p, _ = mpcalc.el(p, t, td)
srh_1 = mpcalc.storm_relative_helicity(z, u, v, depth=1000 * units.m)
srh_3 = mpcalc.storm_relative_helicity(z, u, v, depth=3000 * units.m)

print(f"CAPE={cape.m:.0f} J/kg | CIN={cin.m:.0f} J/kg")
print(f"LCL={lcl_p.m:.0f} hPa | LFC={lfc_p.m:.0f} hPa | EL={el_p.m:.0f} hPa")
print(f"0-1 km SRH={srh_1[0].m:.0f} m²/s² | 0-3 km SRH={srh_3[0].m:.0f} m²/s²")

# ---- Combined skew-T + hodograph figure ----
fig = plt.figure(figsize=(16, 10))
skew = SkewT(fig, rotation=45, rect=(0.05, 0.05, 0.55, 0.90))

skew.plot_dry_adiabats(alpha=0.25, colors="orange")
skew.plot_moist_adiabats(alpha=0.25, colors="green")
skew.plot_mixing_lines(alpha=0.25, colors="blue")
skew.plot(p, t, "r", linewidth=2, label="Temperature")
skew.plot(p, td, "g", linewidth=2, label="Dewpoint")
skew.plot(p, parcel, "k--", linewidth=1.5, label="Parcel")
skew.shade_cape(p, t, parcel)
skew.shade_cin(p, t, parcel, td)
skew.ax.axhline(lcl_p.m, color="purple", linestyle=":", linewidth=1.2)
skew.ax.axhline(lfc_p.m, color="darkorange", linestyle="-.", linewidth=1.2)
skew.ax.axhline(el_p.m, color="cyan", linestyle="-.", linewidth=1.2)
skew.plot_barbs(p[::4], u[::4], v[::4])
skew.ax.set_xlim(-40, 50)
skew.ax.set_ylim(1050, 100)
skew.ax.set_title("27 Apr 2011 Super Outbreak – OKC Sounding\nSkew-T Log-P", fontsize=12)
skew.ax.legend(loc="upper right", fontsize=8)

# Hodograph inset
ax_hod = fig.add_axes([0.63, 0.45, 0.33, 0.48])
h = Hodograph(ax_hod, component_range=60)
h.add_grid(increment=10)
intervals = np.array([0, 1000, 3000, 6000]) * units.m
colors = ["purple", "red", "darkorange"]
h.plot_colormapped(u, v, z, intervals=intervals, colors=colors)
ax_hod.set_title("Hodograph (m/s)", fontsize=10)

# Summary text
summary = (
    f"CAPE: {cape.m:.0f} J/kg\n"
    f"CIN:  {cin.m:.0f} J/kg\n"
    f"0-1 SRH: {srh_1[0].m:.0f} m²/s²\n"
    f"0-3 SRH: {srh_3[0].m:.0f} m²/s²"
)
fig.text(0.63, 0.30, summary, fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85))

fig.savefig("example1_tornado_sounding.png", dpi=150, bbox_inches="tight")
print("Saved: example1_tornado_sounding.png")
plt.show()
```

### Example 2: Multi-Station CAPE Comparison Along a Dryline

```python
"""
example2_dryline_cape_map.py
-----------------------------
Simulate a spatial survey of CAPE values across a dryline gradient and
plot them on a Cartopy map to illustrate spatial gradients in instability.
Replace the synthetic data section with real Siphon sounding fetches.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy.calc as mpcalc
from metpy.units import units

warnings.filterwarnings("ignore")

# Representative station locations along a north-south dryline transect
STATIONS = {
    "AMA": (35.2, -101.7),   # Amarillo, TX (dry side)
    "OKC": (35.4, -97.6),    # Oklahoma City (moist side)
    "LBB": (33.7, -101.8),   # Lubbock, TX (dry)
    "MKC": (39.1, -94.6),    # Kansas City (moist)
    "DDC": (37.8, -99.9),    # Dodge City, KS (transition zone)
    "SGF": (37.2, -93.4),    # Springfield, MO (moist)
    "SHV": (32.4, -93.8),    # Shreveport, LA (very moist)
    "OUN": (35.2, -97.5),    # Norman, OK (moist)
}

# Synthetic CAPE values representing a classic plains dryline scenario
# West of dryline (~101°W): low CAPE; east of dryline: high CAPE
CAPE_VALUES = {
    "AMA": 150,
    "OKC": 2800,
    "LBB": 200,
    "MKC": 2200,
    "DDC": 900,
    "SGF": 3100,
    "SHV": 3500,
    "OUN": 2950,
}
CIN_VALUES = {k: np.random.default_rng(i).integers(10, 120)
              for i, k in enumerate(STATIONS)}


def plot_dryline_cape(
    stations: dict,
    cape_values: dict,
    cin_values: dict,
    output_path: str = "example2_dryline_cape.png",
) -> plt.Figure:
    """
    Plot CAPE values at sounding stations on a plains regional map,
    with circle size proportional to CAPE and color indicating CIN.
    """
    proj = ccrs.LambertConformal(central_longitude=-98, central_latitude=37)
    fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={"projection": proj})
    ax.set_extent([-108, -88, 28, 46], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f0f0e8")
    ax.add_feature(cfeature.OCEAN, facecolor="#cce5ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor="gray")

    # Approximate dryline position
    dryline_lons = [-101.5, -101.0, -100.5, -100.8, -101.2]
    dryline_lats = [28, 32, 35, 38, 43]
    ax.plot(dryline_lons, dryline_lats,
            transform=ccrs.PlateCarree(),
            color="brown", linewidth=2.5, linestyle="--",
            label="Approximate Dryline", zorder=5)

    # Scatter stations
    lons = [v[1] for v in stations.values()]
    lats = [v[0] for v in stations.values()]
    capes = [cape_values[k] for k in stations]
    cins = [cin_values[k] for k in stations]

    sc = ax.scatter(
        lons, lats,
        s=[c / 15 for c in capes],
        c=cins, cmap="Reds_r", vmin=0, vmax=150,
        transform=ccrs.PlateCarree(),
        zorder=6, edgecolors="black", linewidths=0.7,
        label="Stations (size ∝ CAPE)",
    )
    plt.colorbar(sc, ax=ax, orientation="vertical", label="CIN (J/kg)", shrink=0.6)

    for name, (lat, lon) in stations.items():
        cape = cape_values[name]
        ax.annotate(
            f"{name}\n{cape} J/kg",
            xy=(lon, lat),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, transform=ccrs.PlateCarree(),
            zorder=7,
        )

    ax.set_title(
        "Synthetic Dryline CAPE Survey — Southern Great Plains\n"
        "Circle size ∝ CAPE; color = CIN (J/kg)",
        fontsize=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return fig


if __name__ == "__main__":
    fig = plot_dryline_cape(STATIONS, CAPE_VALUES, CIN_VALUES)
    plt.show()
```
