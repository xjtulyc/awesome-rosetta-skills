---
name: rioxarray-remote-sensing
description: >
  Read, reproject, resample, and analyze multispectral GeoTIFF imagery with
  rioxarray and rasterio; compute NDVI/EVI, mask clouds, mosaic scenes, and
  export Cloud-Optimized GeoTIFFs.
tags:
  - remote-sensing
  - rioxarray
  - rasterio
  - ndvi
  - geotiff
  - cloud-optimized
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - rioxarray>=0.15.0
    - rasterio>=1.3.9
    - earthpy>=0.9.4
    - numpy>=1.24.0
    - matplotlib>=3.7.0
    - geopandas>=0.14.0
    - shapely>=2.0.0
last_updated: "2026-03-17"
status: "stable"
---

# rioxarray Remote Sensing Skill

> **One-line summary**: This Skill helps researchers read, reproject, resample,
> and analyse multispectral satellite imagery using rioxarray/rasterio, suited
> for NDVI/EVI vegetation analysis, cloud masking, mosaicking, and COG export.

---

## When to Use This Skill

Use this Skill in the following scenarios:

- When you need to **read and write GeoTIFF** files with full CRS/metadata preservation.
- When your raster data needs **reprojection or resampling** to match a target grid.
- When you need to compute **vegetation indices** (NDVI, EVI) from multispectral bands.
- When you are using **Landsat or Sentinel imagery** with QA bit-mask cloud masking.
- When you need to **mosaic multiple scenes** into a single seamless raster.
- When you need to produce **Cloud-Optimized GeoTIFFs (COGs)** for web delivery.
- When you need to **clip rasters to polygon boundaries** (watersheds, administrative units).

**Trigger keywords**: rioxarray, rasterio, NDVI, EVI, GeoTIFF, COG, CRS reprojection,
resampling, cloud masking, QA band, mosaicking, band math, clip to polygon.

---

## Background & Key Concepts

### Coordinate Reference Systems and Reprojection

A Coordinate Reference System (CRS) defines how 2-D map coordinates correspond to
locations on the Earth's surface. Satellite imagery is delivered in a variety of
projections — Landsat Collection 2 uses UTM zones, while global composites often use
WGS-84 geographic coordinates (EPSG:4326). Reprojection transforms pixel data from one
CRS to another by resampling values onto a new grid.

rioxarray wraps rasterio's GDAL-backed resampling. Available algorithms include:

| Algorithm | Use case |
|:----------|:---------|
| Nearest neighbour | Categorical data (land cover, QA bands) |
| Bilinear | Continuous data with moderate accuracy needs |
| Cubic | High-accuracy continuous data resampling |
| Lanczos | Downsampling with minimal aliasing |

### NDVI and EVI Formulae

The Normalized Difference Vegetation Index (NDVI) leverages the spectral contrast between
the Near-Infrared (NIR) and Red bands:

$$
\text{NDVI} = \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + \rho_{Red}}
$$

Values range from -1 to +1; healthy vegetation typically falls between 0.3 and 0.8.
Sparse vegetation, bare soil, and water produce values closer to 0 or negative.

The Enhanced Vegetation Index (EVI) reduces atmospheric and soil-background noise:

$$
\text{EVI} = G \cdot \frac{\rho_{NIR} - \rho_{Red}}{\rho_{NIR} + C_1 \cdot \rho_{Red} - C_2 \cdot \rho_{Blue} + L}
$$

Where $G = 2.5$, $C_1 = 6$, $C_2 = 7.5$, $L = 1$ (standard MODIS coefficients).

### Cloud Masking with QA Bits

Landsat Collection 2 Quality Assessment (QA_PIXEL) uses bit-packed integers.
Bit 3 = cloud shadow, Bit 4 = cloud. Reading a specific bit:

$$
\text{bit}_k = \left\lfloor \frac{QA\_PIXEL}{2^k} \right\rfloor \mod 2
$$

A pixel is considered cloud-contaminated when bit 3 OR bit 4 is set.

### Comparison with Related Methods

| Method | Best for | Key assumption | Limitation |
|:-------|:---------|:---------------|:-----------|
| rioxarray + rasterio | Python-native raster I/O | GDAL-supported formats | Python memory limits for very large mosaics |
| GDAL CLI (`gdalwarp`) | Batch reprojection, scripting | Shell environment | No Python integration |
| Google Earth Engine | Petabyte-scale cloud processing | Internet + GEE account | Proprietary platform |
| xarray + rasterio | NetCDF-heavy workflows | Rectangular grids | Less direct raster support |

---

## Environment Setup

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install Python dependencies
pip install rioxarray>=0.15.0 rasterio>=1.3.9 earthpy>=0.9.4 \
            numpy>=1.24.0 matplotlib>=3.7.0 geopandas>=0.14.0 shapely>=2.0.0

# On conda (recommended for GDAL/rasterio native libs):
# conda install -c conda-forge rioxarray rasterio earthpy geopandas shapely matplotlib
```

### Verify Installation

```python
import rioxarray
import rasterio
import earthpy
import numpy as np
import matplotlib
import geopandas

print(f"rioxarray  : {rioxarray.__version__}")
print(f"rasterio   : {rasterio.__version__}")
print(f"numpy      : {np.__version__}")
print(f"geopandas  : {geopandas.__version__}")
# Expected: rioxarray >= 0.15, rasterio >= 1.3.9
```

---

## Core Workflow

### Step 1: Reading a GeoTIFF and CRS Inspection

rioxarray's `open_rasterio` returns an `xarray.DataArray` with a `.rio` accessor that
exposes all CRS and spatial metadata.

```python
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# Synthetic example: create a small GeoTIFF with 4 bands simulating
# Landsat-8 Blue/Green/Red/NIR reflectance (SR, scale 0-10000)
# ------------------------------------------------------------------
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# Synthetic 4-band scene  (64 x 64 pixels, 30 m resolution, UTM zone 32N)
np.random.seed(42)
n = 64
synthetic = np.random.randint(500, 8000, (4, n, n), dtype=np.int16)
# Band 4 (NIR) elevated over vegetation-like pixels
synthetic[3, 10:40, 10:40] = np.random.randint(5000, 9000, (30, 30))
synthetic[2, 10:40, 10:40] = np.random.randint(500,  2000, (30, 30))

transform = from_bounds(500000, 5700000, 501920, 5701920, n, n)
crs = CRS.from_epsg(32632)

scene_path = Path("synthetic_landsat_b2345.tif")
with rasterio.open(
    scene_path, "w", driver="GTiff",
    height=n, width=n, count=4, dtype="int16",
    crs=crs, transform=transform,
) as dst:
    dst.write(synthetic)
    dst.update_tags(1, name="Blue")
    dst.update_tags(2, name="Green")
    dst.update_tags(3, name="Red")
    dst.update_tags(4, name="NIR")

print(f"Wrote synthetic scene: {scene_path}")

# ------------------------------------------------------------------
# Read with rioxarray
# ------------------------------------------------------------------
da = rxr.open_rasterio(scene_path, masked=True)
print(f"Shape         : {da.shape}")           # (bands, rows, cols)
print(f"CRS           : {da.rio.crs}")
print(f"Resolution    : {da.rio.resolution()}")
print(f"Bounds        : {da.rio.bounds()}")
print(f"NoData value  : {da.rio.nodata}")
# Expected shape: (4, 64, 64); CRS: EPSG:32632
```

**Data requirements**:
- `band`: integer band index starting at 1
- `y` / `x`: spatial coordinate dimensions in the native CRS units
- Recommended: floating-point or 16-bit integer reflectance values, scale factor noted

### Step 2: Reprojection and Resampling

```python
import rioxarray as rxr
from rasterio.enums import Resampling

da = rxr.open_rasterio("synthetic_landsat_b2345.tif", masked=True)

# --- Reproject to WGS-84 geographic (EPSG:4326) using bilinear resampling ---
da_wgs84 = da.rio.reproject("EPSG:4326", resampling=Resampling.bilinear)
print(f"Reprojected CRS      : {da_wgs84.rio.crs}")
print(f"Reprojected bounds   : {da_wgs84.rio.bounds()}")

# --- Reproject to a specific resolution (e.g., 0.0003 deg ~ 30 m) ---
da_resampled = da.rio.reproject(
    "EPSG:4326",
    resolution=0.0003,
    resampling=Resampling.cubic,
)
print(f"Custom-resolution shape: {da_resampled.shape}")

# --- Match another raster's grid exactly (reproject_match) ---
# Useful when aligning multi-sensor data
reference = rxr.open_rasterio("synthetic_landsat_b2345.tif", masked=True)
da_matched = da_wgs84.rio.reproject_match(reference)
print(f"Matched shape: {da_matched.shape}")
```

**Parameter reference**:

| Parameter | Meaning | Recommended value | Notes |
|:----------|:--------|:------------------|:------|
| `resampling` | Pixel value interpolation method | `bilinear` for continuous | Use `nearest` for categorical layers |
| `resolution` | Target pixel size in CRS units | Match target dataset | Degrees for geographic CRS, metres for projected |
| `nodata` | Fill value for areas outside original extent | `np.nan` | Set via `da.rio.write_nodata()` before reprojection |

### Step 3: NDVI and EVI Calculation

```python
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

da = rxr.open_rasterio("synthetic_landsat_b2345.tif", masked=True).astype(float)

# Landsat SR scale: divide by 10000 to get surface reflectance [0, 1]
blue = da.sel(band=1) / 10000.0
green = da.sel(band=2) / 10000.0
red   = da.sel(band=3) / 10000.0
nir   = da.sel(band=4) / 10000.0

# --- NDVI ---
ndvi = (nir - red) / (nir + red)
ndvi = ndvi.where(np.isfinite(ndvi))   # mask division-by-zero

# --- EVI (MODIS coefficients) ---
G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
evi_denom = nir + C1 * red - C2 * blue + L
evi = G * (nir - red) / evi_denom
evi = evi.clip(-1.0, 1.0)

print(f"NDVI  min/mean/max : {float(ndvi.min()):.3f} / "
      f"{float(ndvi.mean()):.3f} / {float(ndvi.max()):.3f}")
print(f"EVI   min/mean/max : {float(evi.min()):.3f} / "
      f"{float(evi.mean()):.3f} / {float(evi.max()):.3f}")

# --- Plot side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(ndvi.values, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
axes[0].set_title("NDVI")
axes[0].axis("off")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(evi.values, cmap="RdYlGn", vmin=-0.2, vmax=0.9)
axes[1].set_title("EVI")
axes[1].axis("off")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.suptitle("Vegetation Indices (synthetic Landsat-8 scene)")
plt.tight_layout()
plt.savefig("ndvi_evi_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ndvi_evi_comparison.png")

# --- Interpreting results ---
print("\nVegetation cover fractions:")
for label, lo, hi in [("Water/bare", -1.0, 0.1),
                       ("Sparse veg", 0.1, 0.3),
                       ("Moderate",   0.3, 0.6),
                       ("Dense veg",  0.6, 1.0)]:
    frac = float(((ndvi >= lo) & (ndvi < hi)).sum()) / ndvi.size
    print(f"  {label:<15}: {frac*100:5.1f} %")
```

**Interpreting results**:
- **NDVI > 0.6**: dense, healthy vegetation canopy
- **NDVI 0.2–0.6**: moderate/sparse vegetation
- **NDVI < 0.1**: bare soil, rock, urban areas, or water

---

## Advanced Usage

### Cloud Masking with QA Band (Bit Manipulation)

Landsat Collection 2 QA_PIXEL band encodes quality flags as bit fields.
Bits 3 (cloud shadow) and 4 (cloud) indicate unreliable pixels.

```python
import rioxarray as rxr
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

# --- Synthesize a QA_PIXEL band ---
np.random.seed(0)
n = 64
qa_data = np.zeros((1, n, n), dtype=np.uint16)
# Set cloud bit (bit 4, value 16) for random pixels
cloud_mask_idx = np.random.choice(n * n, size=200, replace=False)
rows, cols = np.unravel_index(cloud_mask_idx, (n, n))
qa_data[0, rows, cols] |= (1 << 4)   # bit 4 = cloud
# Set cloud shadow (bit 3, value 8) for a few pixels
qa_data[0, 5:10, 5:10] |= (1 << 3)   # bit 3 = cloud shadow

transform = from_bounds(500000, 5700000, 501920, 5701920, n, n)
qa_path = Path("synthetic_qa_pixel.tif")
with rasterio.open(
    qa_path, "w", driver="GTiff",
    height=n, width=n, count=1, dtype="uint16",
    crs=CRS.from_epsg(32632), transform=transform,
) as dst:
    dst.write(qa_data)

# --- Apply cloud mask ---
qa = rxr.open_rasterio(qa_path, masked=False).squeeze()   # 2-D array
qa_np = qa.values.astype(np.uint16)

cloud_bit        = 4   # bit index
cloud_shadow_bit = 3

cloud_flag        = (qa_np >> cloud_bit)        & 1
cloud_shadow_flag = (qa_np >> cloud_shadow_bit) & 1
bad_pixel_mask    = (cloud_flag | cloud_shadow_flag).astype(bool)

print(f"Flagged pixels: {bad_pixel_mask.sum()} / {bad_pixel_mask.size} "
      f"({bad_pixel_mask.mean()*100:.1f}%)")

# Apply mask to NDVI
da = rxr.open_rasterio("synthetic_landsat_b2345.tif", masked=True).astype(float)
ndvi = (da.sel(band=4) - da.sel(band=3)) / (da.sel(band=4) + da.sel(band=3))
ndvi_masked = ndvi.where(~bad_pixel_mask)

print(f"NDVI valid pixels after cloud masking: "
      f"{int(np.isfinite(ndvi_masked.values).sum())}")
```

### Clip Raster to Polygon Boundary

```python
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import box, mapping
import numpy as np

# --- Create a synthetic polygon (AOI) in UTM zone 32N ---
aoi = gpd.GeoDataFrame(
    geometry=[box(500300, 5700300, 501600, 5701600)],
    crs="EPSG:32632",
)

da = rxr.open_rasterio("synthetic_landsat_b2345.tif", masked=True)

# Ensure CRS matches before clipping
aoi = aoi.to_crs(da.rio.crs)

da_clipped = da.rio.clip(aoi.geometry.apply(mapping), aoi.crs, drop=True)
print(f"Original shape : {da.shape}")
print(f"Clipped shape  : {da_clipped.shape}")
print(f"Clipped bounds : {da_clipped.rio.bounds()}")
```

### Mosaicking Multiple Scenes

When a study area spans multiple raster tiles, merge them into a single seamless image.

```python
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import rasterio
import numpy as np
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path

# --- Synthesize two adjacent tiles ---
def make_tile(path, x_min, x_max, n=64):
    np.random.seed(int(x_min))
    data = np.random.randint(500, 8000, (4, n, n), dtype=np.int16)
    transform = from_bounds(x_min, 5700000, x_max, 5701920, n, n)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=n, width=n, count=4, dtype="int16",
        crs=CRS.from_epsg(32632), transform=transform,
    ) as dst:
        dst.write(data)

make_tile("tile_west.tif", 500000, 501920)
make_tile("tile_east.tif", 501920, 503840)

tile_w = rxr.open_rasterio("tile_west.tif", masked=True)
tile_e = rxr.open_rasterio("tile_east.tif", masked=True)

# merge_arrays handles overlapping regions with first-in priority by default
mosaic = merge_arrays([tile_w, tile_e])
print(f"West tile shape : {tile_w.shape}")
print(f"East tile shape : {tile_e.shape}")
print(f"Mosaic shape    : {mosaic.shape}")
print(f"Mosaic bounds   : {mosaic.rio.bounds()}")
```

### Export as Cloud-Optimized GeoTIFF (COG)

COGs store overviews and use tiled storage for efficient HTTP range-request access.

```python
import rioxarray as rxr
import numpy as np

da = rxr.open_rasterio("synthetic_landsat_b2345.tif", masked=True)

# Compute NDVI as float32 single-band output
nir = da.sel(band=4).astype("float32")
red = da.sel(band=3).astype("float32")
ndvi = (nir - red) / (nir + red + 1e-9)
ndvi = ndvi.expand_dims("band")  # add band dimension back

# Write COG using rasterio's GDAL driver options
ndvi.rio.to_raster(
    "ndvi_cog.tif",
    driver="GTiff",
    dtype="float32",
    compress="deflate",
    predictor=2,          # horizontal differencing (good for floating point with predictor=3)
    tiled=True,
    blockxsize=512,
    blockysize=512,
    copy_src_overwrite=True,
)
print("Exported Cloud-Optimized GeoTIFF: ndvi_cog.tif")

# Verify the file is valid
import rasterio
with rasterio.open("ndvi_cog.tif") as src:
    print(f"  Driver   : {src.driver}")
    print(f"  CRS      : {src.crs}")
    print(f"  Shape    : {src.width} x {src.height}")
    print(f"  Profile  : {src.profile}")
```

### Performance Optimization

```python
import rioxarray as rxr
import dask

# Open very large GeoTIFF lazily with dask chunks
# chunks={'x': 1024, 'y': 1024} keeps each chunk ~4 MB for float32
da_lazy = rxr.open_rasterio(
    "large_scene.tif",   # replace with actual large file
    masked=True,
    chunks={"band": 1, "x": 1024, "y": 1024},
    lock=False,           # allow parallel reads
)

print(f"Dask DataArray: {da_lazy}")
print(f"Chunk sizes   : {da_lazy.chunks}")

# Compute only the spatial extent you need
da_subset = da_lazy.isel(x=slice(0, 2048), y=slice(0, 2048)).compute()
print(f"Subset shape  : {da_subset.shape}")
```

---

## Troubleshooting

### Error: `CRSError: Invalid projection: EPSG:XXXX`

**Cause**: EPSG code not found in local PROJ database; proj.db may be outdated.

**Fix**:
```bash
# Update proj.db via conda-forge (preferred)
conda install -c conda-forge proj>=9.0
# Or reinstall rasterio with bundled PROJ
pip install --force-reinstall rasterio
```

### Error: `NoDataError` after clipping

**Cause**: The clip polygon does not overlap the raster extent, or CRS mismatch.

**Fix**:
```python
import rioxarray as rxr
import geopandas as gpd

da = rxr.open_rasterio("my_raster.tif")
gdf = gpd.read_file("my_polygon.shp")

# Always reproject the polygon to match the raster CRS
gdf_reproj = gdf.to_crs(da.rio.crs)
print(f"Raster CRS  : {da.rio.crs}")
print(f"Polygon CRS : {gdf_reproj.crs}")
print(f"Raster bounds  : {da.rio.bounds()}")
print(f"Polygon bounds : {gdf_reproj.total_bounds}")
# Inspect for overlap before calling .rio.clip()
```

### Issue: NDVI values outside [-1, 1]

**Cause**: Unscaled integer reflectance values (e.g., Landsat SR 0–10000 range)
passed directly into the NDVI formula.

**Fix**: Divide by the scale factor (10000 for Landsat C2 L2 SR) before computing
band math.

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| rioxarray | 0.15.x, 0.16.x | `reproject_match` signature changed in 0.14; use keyword args |
| rasterio | 1.3.x, 1.4.x | Windows wheels require GDAL ≥ 3.7; use conda on Windows |
| GDAL (system) | 3.7–3.9 | COG creation requires GDAL ≥ 3.1 |
| numpy | 1.24–2.0 | Boolean indexing behaviour stable across versions |

---

## External Resources

### Official Documentation

- [rioxarray documentation](https://corteva.github.io/rioxarray/stable/)
- [rasterio documentation](https://rasterio.readthedocs.io/)
- [GDAL documentation](https://gdal.org/drivers/raster/gtiff.html)
- [earthpy documentation](https://earthpy.readthedocs.io/)

### Key Papers

- Rouse et al. (1974). *Monitoring Vegetation Systems in the Great Plains with ERTS.* NASA.
- Huete et al. (2002). *Overview of the radiometric and biophysical performance of the MODIS vegetation indices.* Remote Sensing of Environment, 83, 195–213.
- Cloud Optimized GeoTIFF specification: https://cogeo.org/

### Tutorials

- [rioxarray quickstart](https://corteva.github.io/rioxarray/stable/getting_started/getting_started.html)
- [EarthPy remote sensing tutorials](https://earthpy.readthedocs.io/en/latest/gallery_vignettes/index.html)

### Data Sources

- [Landsat Collection 2 (USGS EarthExplorer)](https://earthexplorer.usgs.gov/): 30 m multispectral, 1972–present
- [Sentinel-2 (ESA Copernicus Open Access Hub)](https://scihub.copernicus.eu/): 10–60 m, 2015–present
- [NASA AppEEARS](https://appeears.earthdatacloud.nasa.gov/): MODIS, Landsat, Sentinel subsetting

---

## Examples

### Example 1: Full NDVI Processing Pipeline for a Landsat Scene

**Scenario**: Compute cloud-masked NDVI for a Landsat-8 Collection 2 Level-2 scene,
clip to an agricultural region of interest, and export as COG.

**Input data**: Landsat-8 Collection 2 Level-2 SR bands (B2–B5) + QA_PIXEL band.
The example below uses synthetic data to be fully runnable without downloading.

```python
# =============================================
# End-to-end example: NDVI pipeline
# Requirements: Python 3.10+; rioxarray, rasterio, numpy, matplotlib
# =============================================

import numpy as np
import rasterio
import rioxarray as rxr
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from rioxarray.merge import merge_arrays
import geopandas as gpd
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---- 1. Synthesize Landsat-like 4-band reflectance + QA ----
np.random.seed(123)
n = 128
transform = from_bounds(500000, 5700000, 503840, 5703840, n, n)
crs = CRS.from_epsg(32632)

reflectance = np.random.randint(200, 4000, (4, n, n), dtype=np.int16)
# Simulate a vegetated patch (high NIR, low Red)
reflectance[3, 30:80, 30:80] = np.random.randint(6000, 9500, (50, 50))
reflectance[2, 30:80, 30:80] = np.random.randint(300, 1200, (50, 50))

qa = np.zeros((1, n, n), dtype=np.uint16)
cloud_y, cloud_x = np.random.choice(n, 300), np.random.choice(n, 300)
qa[0, cloud_y, cloud_x] |= (1 << 4)   # cloud bit

for fname, data, dtype in [
    ("ls8_sr.tif", reflectance, "int16"),
    ("ls8_qa.tif", qa,          "uint16"),
]:
    with rasterio.open(fname, "w", driver="GTiff",
                       height=n, width=n,
                       count=data.shape[0], dtype=dtype,
                       crs=crs, transform=transform) as dst:
        dst.write(data)

# ---- 2. Load reflectance and QA ----
da = rxr.open_rasterio("ls8_sr.tif", masked=True).astype("float32")
qa_da = rxr.open_rasterio("ls8_qa.tif", masked=False).squeeze()

# ---- 3. Cloud mask ----
qa_np = qa_da.values.astype(np.uint16)
cloud_flag = (qa_np >> 4) & 1
cloud_shadow_flag = (qa_np >> 3) & 1
bad = (cloud_flag | cloud_shadow_flag).astype(bool)
print(f"Cloud/shadow pixels: {bad.sum()} ({bad.mean()*100:.1f}%)")

# ---- 4. NDVI ----
nir = da.sel(band=4) / 10000.0
red = da.sel(band=3) / 10000.0
ndvi = (nir - red) / (nir + red)
ndvi_clean = ndvi.where(~bad)

# ---- 5. Clip to AOI polygon ----
aoi = gpd.GeoDataFrame(geometry=[box(500500, 5700500, 503000, 5703000)], crs=crs)
ndvi_aoi = ndvi_clean.expand_dims("band").rio.clip(
    aoi.geometry.apply(mapping), aoi.crs, drop=True
).squeeze("band")

# ---- 6. Plot result ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(ndvi_clean.values, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
axes[0].set_title("NDVI (cloud-masked, full scene)")
axes[0].axis("off")
axes[1].imshow(ndvi_aoi.values, cmap="RdYlGn", vmin=-0.1, vmax=0.9)
axes[1].set_title("NDVI (clipped to AOI)")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("ndvi_pipeline_output.png", dpi=150)
plt.show()

# ---- 7. Export COG ----
ndvi_aoi.expand_dims("band").rio.to_raster(
    "ndvi_aoi_cog.tif",
    driver="GTiff", dtype="float32",
    compress="deflate", tiled=True, blockxsize=256, blockysize=256,
)
print(f"NDVI stats (AOI): min={float(ndvi_aoi.min()):.3f}, "
      f"mean={float(ndvi_aoi.mean()):.3f}, max={float(ndvi_aoi.max()):.3f}")
print("Exported: ndvi_aoi_cog.tif")
# Expected: NDVI mean ~0.4–0.6 in the vegetated patch area
```

**Interpreting these results**: The vegetated patch (rows 30–80, cols 30–80) should
show NDVI around 0.6–0.8. Cloud-masked pixels appear as NaN (white in the figure).
The COG is ready for web map tile services.

---

### Example 2: Multi-Scene Mosaic and EVI Map

**Scenario**: Combine two adjacent Landsat tiles and produce a gap-free EVI mosaic.

```python
# =============================================
# End-to-end example 2: mosaic + EVI
# =============================================

import numpy as np
import rasterio
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import matplotlib.pyplot as plt

def make_scene(path, x_min, x_max, seed=0):
    np.random.seed(seed)
    n = 64
    data = np.random.randint(200, 7000, (4, n, n), dtype=np.int16)
    # Sprinkle some vegetation
    data[3, 10:40, 10:40] = np.random.randint(5500, 9000, (30, 30))
    data[2, 10:40, 10:40] = np.random.randint(400, 1500, (30, 30))
    data[0, 10:40, 10:40] = np.random.randint(200, 700, (30, 30))  # Blue band
    t = from_bounds(x_min, 5700000, x_max, 5701920, n, n)
    with rasterio.open(path, "w", driver="GTiff",
                       height=n, width=n, count=4, dtype="int16",
                       crs=CRS.from_epsg(32632), transform=t) as dst:
        dst.write(data)

make_scene("scene_A.tif", 500000, 501920, seed=7)
make_scene("scene_B.tif", 501920, 503840, seed=13)

tiles = [rxr.open_rasterio(p, masked=True).astype("float32")
         for p in ["scene_A.tif", "scene_B.tif"]]

mosaic = merge_arrays(tiles)
print(f"Mosaic shape: {mosaic.shape}, bounds: {mosaic.rio.bounds()}")

# EVI calculation
blue = mosaic.sel(band=1) / 10000.0
red  = mosaic.sel(band=3) / 10000.0
nir  = mosaic.sel(band=4) / 10000.0
G, C1, C2, L = 2.5, 6.0, 7.5, 1.0
evi = G * (nir - red) / (nir + C1 * red - C2 * blue + L)
evi = evi.clip(-1.0, 1.0)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(evi.values, cmap="RdYlGn", vmin=-0.1, vmax=0.9, aspect="auto")
ax.set_title("EVI — Mosaicked Scene (2 tiles)")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="EVI")
plt.tight_layout()
plt.savefig("evi_mosaic.png", dpi=150)
plt.show()

print(f"EVI: min={float(evi.min()):.3f}, mean={float(evi.mean()):.3f}, "
      f"max={float(evi.max()):.3f}")
# Expected: vegetated pixels around 0.5-0.8; background ~0.1-0.3
```

**Interpreting these results**: The mosaic seamlessly joins both tiles.
EVI values in the simulated vegetation patches (top-left quadrant of each tile)
should exceed 0.4. EVI is preferred over NDVI in dense canopy conditions because
it saturates less in high-biomass areas.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
