---
name: hydrology-pysheds
description: >
  Use this Skill for DEM-based hydrological analysis: watershed delineation,
  flow direction/accumulation, stream networks, and runoff estimation with pysheds.
tags:
  - earth-science
  - hydrology
  - pysheds
  - dem
  - watershed
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
    - pysheds>=0.3
    - rasterio>=1.3
    - geopandas>=0.14
    - matplotlib>=3.7
    - numpy>=1.24
    - scipy>=1.11
last_updated: "2026-03-17"
status: "stable"
---

# Hydrological Analysis with pysheds

> **One-line summary**: Derive watersheds, stream networks, and flow accumulation from Digital Elevation Models (DEMs) using pysheds, rasterio, and geopandas.

---

## When to Use This Skill

- When delineating watershed boundaries from a DEM for a pour point
- When extracting stream networks from topographic data
- When computing flow direction and accumulation grids
- When estimating runoff and drainage basin area
- When conditioning DEMs to remove pits and flat areas
- When analyzing upstream contributing area for flood modeling

**Trigger keywords**: watershed delineation, DEM, flow direction, flow accumulation, stream network, pysheds, hydrological analysis, basin area, pour point

---

## Background & Key Concepts

### DEM Processing Pipeline

$$
\text{Raw DEM} \xrightarrow{\text{pit-fill}} \text{Conditioned DEM} \xrightarrow{\text{flow direction}} \text{D8 grid} \xrightarrow{\text{accumulation}} \text{Flow acc.} \xrightarrow{\text{threshold}} \text{Stream network}
$$

### D8 Flow Direction

Each cell flows to one of 8 neighbors based on steepest descent. Flow direction encoding: N=64, NE=128, E=1, SE=2, S=4, SW=8, W=16, NW=32.

### Flow Accumulation

$A_{ij}$ = number of upstream cells draining through cell $(i,j)$. Cells with high accumulation are stream channels.

### Watershed Delineation

Starting from a pour point (outlet), trace upstream through the flow direction grid to find all contributing cells.

### SCS Curve Number Method

For runoff estimation:

$$
Q = \frac{(P - I_a)^2}{P - I_a + S}, \quad S = \frac{25400}{CN} - 254, \quad I_a = 0.2S
$$

where $P$ is precipitation depth (mm), $CN$ is the SCS curve number.

---

## Environment Setup

### Install Dependencies

```bash
pip install pysheds>=0.3 rasterio>=1.3 geopandas>=0.14 \
            matplotlib>=3.7 numpy>=1.24 scipy>=1.11
```

### Download a Test DEM

```bash
# Download SRTM DEM tile (90m resolution) for a test region
# Using elevation package or manual download from USGS
pip install elevation
eio clip -o dem_test.tif --bounds -105.5 39.5 -104.5 40.5  # Colorado Front Range
```

### Verify Installation

```python
from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt

print("pysheds installation OK")
# Create a synthetic DEM for testing
dem_arr = np.random.rand(50, 50) * 100
print(f"Test DEM shape: {dem_arr.shape}")
```

---

## Core Workflow

### Step 1: DEM Loading and Conditioning

```python
from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def create_synthetic_dem(nrows=200, ncols=200, seed=42):
    """
    Create a synthetic DEM with a valley and drainage divide.
    Replace with: grid = Grid.from_raster('your_dem.tif')
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2*np.pi, ncols)
    y = np.linspace(0, 2*np.pi, nrows)
    X, Y = np.meshgrid(x, y)

    # Base topography: slope + valley
    dem = 1000 - 200 * Y/Y.max() + 100 * np.sin(X) * np.cos(Y/2)
    # Add noise
    dem += 5 * rng.standard_normal((nrows, ncols))
    return dem

def load_and_condition_dem(dem_path=None):
    """
    Load a DEM and condition it for hydrological analysis.

    Parameters
    ----------
    dem_path : str or None
        Path to GeoTIFF DEM. If None, uses synthetic DEM.

    Returns
    -------
    grid : pysheds Grid
    dem : ndarray
    flooded_dem : ndarray (pit-filled)
    """
    if dem_path:
        grid = Grid.from_raster(dem_path)
        dem = grid.read_raster(dem_path)
    else:
        # Synthetic DEM wrapped in pysheds Grid
        dem_arr = create_synthetic_dem()
        # For real workflows, use Grid.from_raster(path)
        # Here we demonstrate with numpy arrays only
        print("Using synthetic DEM (replace with real DEM path)")
        return None, dem_arr, None

    # Pit filling (remove depressions that block flow)
    pit_filled = grid.fill_pits(dem)
    depressions_filled = grid.fill_depressions(pit_filled)
    # Resolve flats (areas with same elevation)
    inflated = grid.resolve_flats(depressions_filled)

    print(f"DEM shape: {dem.shape}")
    print(f"Elevation range: {dem.min():.1f} – {dem.max():.1f} m")
    print(f"Pit-filled cells: {(inflated != dem).sum():,}")

    # Visualize elevation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(dem, cmap='terrain')
    plt.colorbar(im0, ax=axes[0], label='Elevation (m)')
    axes[0].set_title("Original DEM")

    im1 = axes[1].imshow(inflated - dem, cmap='Blues')
    plt.colorbar(im1, ax=axes[1], label='Δ Elevation (m)')
    axes[1].set_title("Filled Depressions (difference)")

    plt.tight_layout()
    plt.savefig("dem_conditioning.png", dpi=150)
    plt.show()

    return grid, dem, inflated

# For demonstration with synthetic data
_, dem_raw, _ = load_and_condition_dem()
print(f"Synthetic DEM stats: mean={dem_raw.mean():.1f}, std={dem_raw.std():.1f}")
```

### Step 2: Flow Direction and Accumulation

```python
from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt

# With a real DEM file:
# grid = Grid.from_raster('dem.tif')
# dem = grid.read_raster('dem.tif')
# pit_filled = grid.fill_pits(dem)
# depressions = grid.fill_depressions(pit_filled)
# inflated = grid.resolve_flats(depressions)

# Compute flow direction (D8 algorithm)
# fdir = grid.flowdir(inflated)
# print(f"Flow direction computed: unique values={np.unique(fdir)}")

# Compute flow accumulation
# acc = grid.accumulation(fdir)
# print(f"Max accumulation: {acc.max():,} cells")

# Stream network (threshold accumulation)
# stream_threshold = int(acc.max() * 0.001)  # 0.1% of max
# streams = acc > stream_threshold
# print(f"Stream cells: {streams.sum():,}")

# For demonstration without DEM file, show the logic:
print("Flow direction and accumulation workflow:")
print("1. grid.flowdir(conditioned_dem) → D8 direction grid")
print("2. grid.accumulation(fdir) → upstream area in cells")
print("3. acc > threshold → binary stream network")
print("4. grid.snap_to_mask(streams, xy) → snap pour point to stream")

# Synthetic example showing accumulation concept
np.random.seed(42)
n = 50
acc_synthetic = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        acc_synthetic[i, j] = (i + 1) * (j + 1)  # simplified

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(np.log1p(acc_synthetic), cmap='Blues')
plt.colorbar(im, ax=ax, label='log(accumulation + 1)')
ax.set_title("Synthetic Flow Accumulation")
plt.tight_layout()
plt.savefig("flow_accumulation.png", dpi=150)
plt.show()
```

### Step 3: Watershed Delineation

```python
from pysheds.grid import Grid
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import matplotlib.pyplot as plt

# With a real DEM:
# grid = Grid.from_raster('dem.tif')
# ...
# fdir = grid.flowdir(inflated)
# acc = grid.accumulation(fdir)
#
# # Snap pour point to nearest stream
# x_outlet, y_outlet = -105.0, 40.0  # longitude, latitude
# x_snap, y_snap = grid.snap_to_mask(acc > threshold, (x_outlet, y_outlet))
#
# # Delineate watershed
# catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir,
#                        xytype='coordinate')
# catch_view = grid.view(catch)
#
# # Convert to polygon
# shapes = grid.polygonize(catch_view)
# for geom, val in shapes:
#     if val:
#         watershed_polygon = shape(geom)
#         break
#
# # Area calculation
# gdf = gpd.GeoDataFrame({'geometry': [watershed_polygon]}, crs=grid.crs)
# gdf_proj = gdf.to_crs('EPSG:32614')  # UTM zone 14N
# area_km2 = gdf_proj.geometry.area.values[0] / 1e6
# print(f"Watershed area: {area_km2:.1f} km²")

# Demonstration output
print("Watershed delineation workflow complete")
print("Key outputs:")
print("  - watershed polygon (GeoDataFrame)")
print("  - drainage area in km²")
print("  - stream network as LineString features")

# SCS Curve Number runoff calculation
def scs_runoff(P_mm, CN):
    """
    Estimate direct runoff depth using SCS Curve Number method.

    Parameters
    ----------
    P_mm : float or ndarray
        Total precipitation depth (mm)
    CN : float
        SCS Curve Number (0-100)

    Returns
    -------
    Q_mm : float or ndarray
        Direct runoff depth (mm)
    """
    S = (25400 / CN) - 254   # potential maximum retention
    Ia = 0.2 * S              # initial abstraction
    P = np.asarray(P_mm)
    Q = np.where(P > Ia, (P - Ia)**2 / (P - Ia + S), 0.0)
    return Q

# Example: 25mm storm on different land cover
storm_mm = 25.0
land_covers = {
    "Row crops (poor)":  86,
    "Row crops (good)":  75,
    "Meadow (good)":     58,
    "Forest (good)":     55,
    "Impervious":        98,
}

print("\nSCS Runoff Estimation (25mm storm):")
print(f"{'Land Cover':<25} {'CN':>4} {'Runoff (mm)':>12} {'Runoff/P (%)':>13}")
print("-" * 56)
for lc, cn in land_covers.items():
    Q = scs_runoff(storm_mm, cn)
    print(f"{lc:<25} {cn:>4} {Q:>12.1f} {100*Q/storm_mm:>13.1f}")
```

---

## Advanced Usage

### Automated Multi-Watershed Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def estimate_flood_frequency(area_km2, mean_annual_precip_mm, CN):
    """
    Simplified regional flood frequency using SCS method.
    Returns peak discharge estimates (m³/s) for return periods.
    """
    return_periods = [2, 5, 10, 25, 50, 100]
    # Regional rainfall frequency (simplified, use NOAA Atlas 14 for real values)
    rp_factors = {2: 1.0, 5: 1.4, 10: 1.7, 25: 2.1, 50: 2.4, 100: 2.8}

    base_storm = mean_annual_precip_mm / 12  # monthly average
    results = []
    for rp in return_periods:
        P = base_storm * rp_factors[rp]
        Q_mm = scs_runoff(P, CN)
        # Rational method approximation: Q = CIA/360 (m³/s)
        runoff_vol_m3 = Q_mm / 1000 * area_km2 * 1e6
        Tc_h = 0.0078 * (area_km2**0.5 / 0.3)**0.77  # time of concentration (h)
        Q_peak = runoff_vol_m3 / (Tc_h * 3600)
        results.append({"return_period": rp, "Q_peak_m3s": Q_peak})

    return pd.DataFrame(results)

flood_table = estimate_flood_frequency(area_km2=150, mean_annual_precip_mm=600, CN=72)
print(flood_table.round(2))

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(flood_table["return_period"], flood_table["Q_peak_m3s"], 'bs-', linewidth=2)
ax.set_xlabel("Return Period (years)")
ax.set_ylabel("Peak Discharge (m³/s)")
ax.set_title("Flood Frequency Curve")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("flood_frequency.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Error: `ValueError: No data in raster`

**Cause**: DEM has NoData values throughout or incorrect file path.

**Fix**:
```python
import rasterio
with rasterio.open("dem.tif") as src:
    print(f"Nodata value: {src.nodata}")
    print(f"Data range: {src.read(1).min()} – {src.read(1).max()}")
    print(f"CRS: {src.crs}")
```

### Issue: All cells flow in same direction (flat DEM)

**Cause**: DEM has no topographic relief (e.g., coastal plain or incorrect data).

**Fix**:
```python
# After fill_depressions, always resolve_flats before flowdir
inflated = grid.resolve_flats(depressions)
fdir = grid.flowdir(inflated)
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| pysheds | 0.3.5           | API changed in 0.3 (Grid class) |
| rasterio | 1.3, 1.4       | None |

---

## External Resources

### Official Documentation

- [pysheds documentation](https://mattbartos.com/pysheds/)
- [USGS SRTM DEM data](https://earthexplorer.usgs.gov/)

### Key Papers

- Bartos, M. et al. (2021). *pysheds: An open-source Python library for watershed delineation*. JOSS.

---

## Examples

### Example 1: Extract Stream Network from SRTM DEM

```python
# =============================================
# Stream network extraction workflow
# =============================================
# NOTE: Requires actual DEM file (download from USGS Earth Explorer)
# This shows the complete workflow assuming dem.tif exists

try:
    from pysheds.grid import Grid
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd

    # Step 1: Load
    grid = Grid.from_raster('dem.tif')
    dem = grid.read_raster('dem.tif')
    print(f"DEM loaded: {dem.shape}, range {dem.min():.0f}–{dem.max():.0f} m")

    # Step 2: Condition
    pit_filled = grid.fill_pits(dem)
    dep_filled = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(dep_filled)

    # Step 3: Flow analysis
    fdir = grid.flowdir(inflated)
    acc  = grid.accumulation(fdir)

    # Step 4: Extract streams at multiple thresholds
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, thresh_pct in zip(axes, [0.1, 0.5, 2.0]):
        thresh = int(acc.max() * thresh_pct / 100)
        streams = acc > thresh
        ax.imshow(np.log1p(acc), cmap='Blues', alpha=0.5)
        ax.imshow(np.ma.masked_where(~streams, streams), cmap='Reds', alpha=0.8)
        ax.set_title(f"Threshold: {thresh_pct}% ({thresh:,} cells)")
        ax.axis('off')

    plt.suptitle("Stream Networks at Different Thresholds")
    plt.tight_layout()
    plt.savefig("stream_networks.png", dpi=150)
    plt.show()

except FileNotFoundError:
    print("DEM file not found. Download from USGS Earth Explorer (https://earthexplorer.usgs.gov/)")
    print("Then run: eio clip -o dem.tif --bounds <lon_min> <lat_min> <lon_max> <lat_max>")
```

**Interpreting these results**: Lower thresholds yield denser stream networks; higher thresholds show only major rivers. Use Strahler stream order to classify streams.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
