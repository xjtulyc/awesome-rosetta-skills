---
name: xarray-netcdf
description: "Labeled multi-dimensional array analysis with xarray: NetCDF/HDF5 I/O, lazy Dask loading, rechunking, Zarr stores, and CF conventions."
tags:
  - xarray
  - netcdf
  - dask
  - zarr
  - climate-science
  - scientific-computing
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
  - xarray>=2023.6
  - dask>=2023.6
  - zarr>=2.15
  - netCDF4>=1.6
  - matplotlib>=3.7
  - numpy>=1.24
last_updated: "2026-03-17"
status: stable
---

# xarray + NetCDF — Labeled N-D Arrays, Lazy I/O & CF Conventions

xarray brings the power of pandas-style labeled indexing to N-dimensional arrays.
Combined with Dask for lazy out-of-core computation and Zarr for cloud-native storage,
it forms the backbone of modern climate, oceanography, and remote-sensing workflows.

---

## When to Use This Skill

Use this skill when you need to:

- Open, inspect, and process NetCDF4, HDF5, or GRIB files without loading all data to RAM.
- Work with coordinates (lat, lon, time, level) attached to array dimensions.
- Run climatological analyses: seasonal means, anomalies, rolling statistics.
- Rechunk large datasets for parallel processing with Dask workers.
- Convert legacy `.nc` files to chunked Zarr stores for fast cloud access.
- Attach CF-compliant metadata (units, standard_name, calendar) to output files.
- Merge, align, or interpolate multiple datasets with different grids.

Do **not** use this skill for purely tabular data (use pandas) or for small arrays
that fit comfortably in memory without any need for coordinate labels (use NumPy).

---

## Background & Key Concepts

### xarray Data Structures

| Class | Description |
|-------|-------------|
| `xr.DataArray` | Single N-D variable with named dims, coords, and attrs |
| `xr.Dataset` | Dict-like container of DataArrays sharing dims/coords |

A DataArray records *dimensions* (e.g., `"time"`, `"lat"`, `"lon"`),
*coordinates* (1-D arrays of tick values for each dim), and *attributes* (metadata dict).

### NetCDF & CF Conventions

NetCDF-4 is a hierarchical binary format built on HDF5.
The **CF (Climate and Forecast) Metadata Conventions** standardise:

- Coordinate names (`latitude`, `longitude`, `time`, `pressure`)
- `units` string format (`"degrees_east"`, `"K"`, `"days since 1900-01-01"`)
- `standard_name` vocabulary (e.g., `"air_temperature"`)
- Grid mapping attributes for projected grids

xarray decodes CF metadata automatically when opening NetCDF files.

### Lazy Loading with Dask

When `chunks={}` or `chunks="auto"` is passed to `xr.open_dataset`, each variable
becomes a `dask.array` — a graph of deferred operations. Nothing is loaded until
`.compute()` or `.load()` is called, or data is written to disk.

Choosing good chunk sizes:
- Target ~100 MB per chunk.
- Chunk along the dimensions you will iterate over (usually `time`).
- Avoid tiny chunks (overhead) and huge chunks (memory spikes).

### Zarr

Zarr stores each chunk as a separate compressed file (in a directory or cloud bucket).
This enables concurrent reads by many Dask workers, far outperforming NetCDF for
parallel workloads. The `xarray.Dataset.to_zarr()` method handles the conversion.

---

## Environment Setup

### Install dependencies

```bash
pip install "xarray>=2023.6" "dask[distributed]>=2023.6" "zarr>=2.15" \
            "netCDF4>=1.6" "matplotlib>=3.7" "numpy>=1.24" \
            bottleneck h5py scipy cftime
```

### Optional: start a local Dask cluster

```python
from dask.distributed import Client

client = Client(n_workers=4, threads_per_worker=2, memory_limit="4GB")
print(client.dashboard_link)   # open in browser for task graph visualization
```

### Environment variables

```bash
# Path to your local data archive
export NC_DATA_DIR="/data/netcdf"
# Optional: fsspec token for cloud access (GCS, S3)
export FSSPEC_S3_KEY="<paste-your-key>"
export FSSPEC_S3_SECRET="<paste-your-secret>"
```

```python
import os

data_dir = os.getenv("NC_DATA_DIR", "./data")
s3_key    = os.getenv("FSSPEC_S3_KEY", "")
s3_secret = os.getenv("FSSPEC_S3_SECRET", "")
```

### Verify installation

```python
import xarray as xr
import dask
import zarr
import netCDF4

print("xarray :", xr.__version__)
print("dask   :", dask.__version__)
print("zarr   :", zarr.__version__)
print("netCDF4:", netCDF4.__version__)
```

---

## Core Workflow

### Step 1 — Create, Inspect, and Index a DataArray

```python
import numpy as np
import xarray as xr
import pandas as pd

# ── Build a synthetic 3-D temperature dataset ────────────────────────────────
rng = np.random.default_rng(42)

times = pd.date_range("2020-01-01", periods=365, freq="D")
lats  = np.linspace(-90, 90,  73)    # 2.5° grid
lons  = np.linspace(  0, 357.5, 144) # 2.5° grid

# Shape: (time=365, lat=73, lon=144)
data = (
    280
    + 30 * np.cos(np.deg2rad(lats))[:, None]         # latitude gradient
    + 10 * np.sin(2 * np.pi * np.arange(365) / 365)[None, :, None]  # seasonal cycle
    + rng.standard_normal((73, 365, 144))              # noise
)
# Re-order to (time, lat, lon)
data = data.transpose(1, 0, 2)

temp = xr.DataArray(
    data,
    dims=["time", "lat", "lon"],
    coords={
        "time": times,
        "lat":  xr.Variable("lat",  lats,  attrs={"units": "degrees_north",
                                                    "standard_name": "latitude"}),
        "lon":  xr.Variable("lon",  lons,  attrs={"units": "degrees_east",
                                                    "standard_name": "longitude"}),
    },
    attrs={
        "long_name": "Air Temperature",
        "units": "K",
        "standard_name": "air_temperature",
    },
    name="tas",
)

print(temp)
print("\nDimensions:", dict(temp.dims))
print("Coordinates:", list(temp.coords))

# ── Label-based indexing ──────────────────────────────────────────────────────
# Select a single point by coordinate value
london = temp.sel(lat=51.5, lon=0.0, method="nearest")
print("\nLondon time series shape:", london.shape)

# Select a spatial slice
europe = temp.sel(lat=slice(35, 70), lon=slice(345, 360))
print("Europe slice shape:", europe.shape)

# Select by index position
first_week = temp.isel(time=slice(0, 7))
print("First week shape:", first_week.shape)

# Fancy: nearest-neighbor interpolation to irregular stations
station_lats = xr.DataArray([48.9, 51.5, 55.8], dims="station")
station_lons = xr.DataArray([2.3,  0.0, 37.6],  dims="station")
stations = temp.sel(lat=station_lats, lon=station_lons, method="nearest")
print("Station extraction shape:", stations.shape)
```

### Step 2 — NetCDF I/O and CF Metadata

```python
import xarray as xr
import numpy as np
import pandas as pd
import os

DATA_DIR = os.getenv("NC_DATA_DIR", "./data")
os.makedirs(DATA_DIR, exist_ok=True)
NC_PATH = os.path.join(DATA_DIR, "temperature.nc")

# ── Write a CF-compliant NetCDF file ─────────────────────────────────────────
times = pd.date_range("2000-01-01", periods=12, freq="ME")
lats  = np.linspace(-90, 90,  37)
lons  = np.linspace(  0, 355, 72)
data  = 280 + 20 * np.cos(np.deg2rad(lats))[:, None] * np.ones((37, 12, 72))
data  = data.transpose(1, 0, 2)

ds = xr.Dataset(
    {
        "tas": xr.DataArray(
            data.astype("float32"),
            dims=["time", "lat", "lon"],
            attrs={
                "long_name":     "Near-Surface Air Temperature",
                "standard_name": "air_temperature",
                "units":         "K",
                "cell_methods":  "time: mean",
            },
        )
    },
    coords={
        "time": xr.Variable("time", times,
                             attrs={"long_name": "time",
                                    "axis": "T"}),
        "lat": xr.Variable("lat", lats,
                            attrs={"long_name": "latitude",
                                   "standard_name": "latitude",
                                   "units": "degrees_north",
                                   "axis": "Y"}),
        "lon": xr.Variable("lon", lons,
                            attrs={"long_name": "longitude",
                                   "standard_name": "longitude",
                                   "units": "degrees_east",
                                   "axis": "X"}),
    },
    attrs={
        "Conventions":   "CF-1.10",
        "title":         "Synthetic monthly temperature",
        "institution":   "Demo",
        "source":        "Generated by xarray-netcdf SKILL",
        "history":       "Created 2026-03-17",
        "references":    "",
        "comment":       "Illustrative dataset only.",
    },
)

# Encode time and apply compression
encoding = {
    "tas":  {"zlib": True, "complevel": 4, "dtype": "float32",
             "_FillValue": 1e20},
    "time": {"dtype": "float64",
              "units": "days since 2000-01-01",
              "calendar": "proleptic_gregorian"},
}
ds.to_netcdf(NC_PATH, encoding=encoding)
print(f"Saved to {NC_PATH}")

# ── Re-open and inspect ───────────────────────────────────────────────────────
ds2 = xr.open_dataset(NC_PATH)
print(ds2)
print("\nVariable attrs:", ds2["tas"].attrs)
print("Time dtype:", ds2["time"].dtype)
ds2.close()

# ── Open multiple files with glob (MFDataset) ─────────────────────────────────
# xr.open_mfdataset concatenates files along a shared dimension automatically
# ds_all = xr.open_mfdataset(
#     os.path.join(DATA_DIR, "temperature_*.nc"),
#     combine="by_coords",
#     parallel=True,     # uses dask
# )
```

### Step 3 — Lazy Dask Operations and Rechunking

```python
import xarray as xr
import numpy as np
import pandas as pd
import dask
import os

DATA_DIR = os.getenv("NC_DATA_DIR", "./data")
NC_PATH  = os.path.join(DATA_DIR, "temperature.nc")

# ── Open with Dask chunking ───────────────────────────────────────────────────
ds = xr.open_dataset(
    NC_PATH,
    chunks={"time": 4, "lat": -1, "lon": -1},   # chunk over time
)
print(ds["tas"])                   # shows dask-backed array; no data read yet
print(ds["tas"].chunks)

# ── Compute seasonal mean (lazy until .compute()) ─────────────────────────────
seasonal_mean = ds["tas"].groupby("time.season").mean("time")
print("\nSeasonal mean (lazy):", seasonal_mean)
# Trigger computation
result = seasonal_mean.compute()
print("Seasonal mean (computed):", result)

# ── Rechunk for better parallelism ───────────────────────────────────────────
# Current: chunked over time → good for time slices
# New: chunked over space  → good for spatial operations
ds_rechunked = ds.chunk({"time": -1, "lat": 9, "lon": 18})
print("\nRechunked chunks:", ds_rechunked["tas"].chunks)

# ── Apply a custom ufunc lazily ───────────────────────────────────────────────
def compute_anomaly(x, clim):
    """Subtract climatology from each month."""
    return x - clim

climatology = ds["tas"].mean("time")   # spatial mean map (lazy)
anomaly = xr.apply_ufunc(
    compute_anomaly,
    ds["tas"],
    climatology,
    dask="parallelized",
    output_dtypes=[float],
)
print("\nAnomaly (lazy):", anomaly)
anomaly_computed = anomaly.compute()
print("Anomaly std:", float(anomaly_computed.std()))

# ── Rolling statistics ────────────────────────────────────────────────────────
rolling_mean = ds["tas"].rolling(time=3, center=True).mean()
print("\n3-month rolling mean shape:", rolling_mean.compute().shape)

# ── Resample to quarterly ─────────────────────────────────────────────────────
quarterly = ds["tas"].resample(time="QS").mean()
print("\nQuarterly shape:", quarterly.compute().shape)

ds.close()
```

---

## Advanced Usage

### Convert NetCDF to Zarr for Cloud Storage

```python
import xarray as xr
import zarr
import os
import numpy as np
import pandas as pd

DATA_DIR  = os.getenv("NC_DATA_DIR", "./data")
NC_PATH   = os.path.join(DATA_DIR, "temperature.nc")
ZARR_PATH = os.path.join(DATA_DIR, "temperature.zarr")

ds = xr.open_dataset(NC_PATH, chunks={"time": 4})

# Write to Zarr with explicit chunk encoding
zarr_encoding = {
    "tas": {
        "chunks":    (4, 37, 72),      # must match or divide dask chunks
        "compressor": zarr.Blosc(cname="lz4", clevel=5, shuffle=zarr.Blosc.SHUFFLE),
        "dtype":     "float32",
    }
}
ds.to_zarr(ZARR_PATH, encoding=zarr_encoding, mode="w")
print(f"Zarr store written to {ZARR_PATH}")

# Re-open from Zarr
ds_z = xr.open_zarr(ZARR_PATH)
print(ds_z)

# ── Inspect raw Zarr store ────────────────────────────────────────────────────
store = zarr.open(ZARR_PATH, mode="r")
print("\nZarr store keys:", list(store.keys()))
print("tas array info :", store["tas"].info)

ds.close(); ds_z.close()
```

### Interpolation and Grid Remapping

```python
import xarray as xr
import numpy as np
import pandas as pd

# Create two datasets on different grids
rng = np.random.default_rng(0)

lats_coarse = np.linspace(-90, 90, 19)    # 10° grid
lons_coarse = np.linspace(0, 350, 36)
times = pd.date_range("2020-01", periods=6, freq="ME")
data_coarse = rng.standard_normal((6, 19, 36)).astype("float32") + 280

ds_coarse = xr.Dataset(
    {"tas": (["time", "lat", "lon"], data_coarse)},
    coords={"time": times, "lat": lats_coarse, "lon": lons_coarse},
)

# Fine target grid
lats_fine = np.linspace(-90, 90, 73)    # 2.5° grid
lons_fine = np.linspace(0, 357.5, 144)

# Bi-linear interpolation
ds_fine = ds_coarse.interp(lat=lats_fine, lon=lons_fine, method="linear")
print("Coarse shape:", ds_coarse["tas"].shape)
print("Fine shape:  ", ds_fine["tas"].shape)

# Nearest-neighbor for categorical/mask data
ds_nn = ds_coarse.interp(lat=lats_fine, lon=lons_fine, method="nearest")

# ── Merge two datasets with different times ───────────────────────────────────
times2 = pd.date_range("2020-07", periods=6, freq="ME")
data2  = rng.standard_normal((6, 73, 144)).astype("float32") + 280
ds2    = xr.Dataset(
    {"tas": (["time", "lat", "lon"], data2)},
    coords={"time": times2, "lat": lats_fine, "lon": lons_fine},
)

ds_merged = xr.concat([ds_fine, ds2], dim="time")
print("Merged shape:", ds_merged["tas"].shape)
```

### Weighted Averaging (Latitude Weights)

Area-weighted global mean accounts for the fact that grid cells near the poles
are smaller than those at the equator.

```python
import xarray as xr
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
lats = np.linspace(-90, 90, 73)
lons = np.linspace(0, 357.5, 144)
times = pd.date_range("2020-01-01", periods=12, freq="ME")
data  = (280 + 20 * np.cos(np.deg2rad(lats))[:, None]
         + rng.standard_normal((73, 12, 144))).transpose(1, 0, 2)

da = xr.DataArray(data, dims=["time", "lat", "lon"],
                  coords={"time": times, "lat": lats, "lon": lons},
                  name="tas")

# Cosine-latitude weights
weights = np.cos(np.deg2rad(da.lat))
weights.name = "weights"

# Weighted mean over lat and lon
global_mean = da.weighted(weights).mean(dim=["lat", "lon"])
print("Global mean time series:", global_mean.values)
print("Annual mean:", float(global_mean.mean()))
```

### Working with CF Time Calendars

```python
import xarray as xr
import cftime
import numpy as np

# 360-day calendar (used in many climate models)
times_360 = xr.cftime_range(
    start="1850-01-01", periods=120, freq="MS", calendar="360_day"
)
data_360 = np.random.rand(120, 10, 20).astype("float32")

da_360 = xr.DataArray(
    data_360,
    dims=["time", "lat", "lon"],
    coords={
        "time": times_360,
        "lat": np.linspace(-90, 90, 10),
        "lon": np.linspace(0, 342, 20),
    },
)
print("360-day calendar time range:", da_360.time.values[[0, -1]])
print("Number of time steps:", len(da_360.time))

# Select a decade
decade = da_360.sel(time=slice("1900-01-01", "1909-12-30"))
print("Decade shape:", decade.shape)

# Convert to standard Gregorian calendar by resampling is not trivial;
# the recommended path is to convert units after writing back to NetCDF.
```

### Parallel Processing with Dask Distributed

```python
import xarray as xr
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
import os

DATA_DIR = os.getenv("NC_DATA_DIR", "./data")

def run_parallel_analysis():
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client  = Client(cluster)
    print("Dashboard:", client.dashboard_link)

    # Open a large (synthetic) dataset with dask
    times = pd.date_range("1950-01-01", periods=840, freq="ME")
    lats  = np.linspace(-90, 90, 73)
    lons  = np.linspace(0, 357.5, 144)
    rng   = np.random.default_rng(7)
    data  = rng.standard_normal((840, 73, 144)).astype("float32") + 280

    ds = xr.Dataset(
        {"tas": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    ).chunk({"time": 60, "lat": -1, "lon": -1})

    # Compute global mean in parallel
    weights = np.cos(np.deg2rad(ds["tas"].lat))
    global_mean = ds["tas"].weighted(weights).mean(["lat", "lon"])
    result = global_mean.compute()

    print("Global mean std (70 years):", float(result.std()))

    client.close(); cluster.close()
    return result

# result = run_parallel_analysis()
```

---

## Troubleshooting

### File opens but all variables show NaN

**Cause**: `_FillValue` is being decoded but the compression codec does not match
the version of `netCDF4` used to write the file.

**Fix**: Open with `mask_and_scale=False` to inspect raw values first.

```python
ds_raw = xr.open_dataset("file.nc", mask_and_scale=False)
print(ds_raw["tas"].values[:5])    # inspect without NaN substitution
```

Then re-encode with explicit `_FillValue`:

```python
encoding = {"tas": {"_FillValue": 1e20, "zlib": True}}
ds.to_netcdf("fixed.nc", encoding=encoding)
```

### `ValueError: conflicting sizes for dimension` on `open_mfdataset`

**Cause**: Files have slightly different grid sizes or coordinate values.

**Fix**: Use `join="override"` and verify the coordinate mismatch first.

```python
import xarray as xr

ds = xr.open_mfdataset(
    "data/*.nc",
    combine="by_coords",
    join="override",     # trust first file's coords
    compat="override",
)
```

### Dask workers run out of memory

**Cause**: Chunks are too large or too many tasks run in parallel.

**Fix**: Reduce chunk size and limit concurrency.

```python
from dask.distributed import Client

client = Client(n_workers=4, threads_per_worker=1, memory_limit="2GB")

# Rechunk to smaller pieces
ds = ds.chunk({"time": 12, "lat": 20, "lon": 36})
```

### `RuntimeError: NetCDF: HDF error` on parallel read

**Cause**: HDF5 (the backend for NetCDF4) is not built with parallel I/O support
by default; multiple workers cannot open the same file simultaneously.

**Fix**: Convert to Zarr first, then open with Dask.

```python
import xarray as xr

ds = xr.open_dataset("big_file.nc")
ds.to_zarr("big_file.zarr", mode="w")
ds_z = xr.open_zarr("big_file.zarr")   # Zarr supports concurrent reads
```

### Slow Zarr writes

**Cause**: Too many small chunks produce millions of tiny files.

**Fix**: Use at least 100 MB per chunk and pick a fast compressor.

```python
import zarr

enc = {
    "tas": {
        "chunks": (120, 73, 144),
        "compressor": zarr.Blosc(cname="lz4", clevel=3),
    }
}
ds.to_zarr("output.zarr", encoding=enc, mode="w")
```

---

## External Resources

- xarray documentation: https://docs.xarray.dev/en/stable/
- xarray tutorial: https://tutorial.xarray.dev/
- CF Conventions 1.11: https://cfconventions.org/cf-conventions/cf-conventions.html
- Dask documentation: https://docs.dask.org/en/stable/
- Zarr specification v2: https://zarr.readthedocs.io/en/stable/spec/v2.html
- netCDF4-python: https://unidata.github.io/netcdf4-python/
- Pangeo community (cloud-native geoscience): https://pangeo.io/
- xarray-tutorial notebooks: https://github.com/xarray-contrib/xarray-tutorial

---

## Examples

### Example 1 — ENSO Index from SST Data

Compute the Nino 3.4 sea-surface temperature anomaly index, a standard metric
for El Nino / La Nina monitoring.

```python
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Synthetic SST dataset ────────────────────────────────────────────────────
rng = np.random.default_rng(99)
times = pd.date_range("1980-01-01", periods=504, freq="ME")  # 42 years
lats  = np.linspace(-30, 30, 61)   # tropical band
lons  = np.linspace(120, 290, 171) # Pacific basin

# Add an El Nino-like signal with period ~4 years
enso_signal = 1.5 * np.sin(2 * np.pi * np.arange(504) / 48)
seasonal     = 0.5 * np.cos(2 * np.pi * np.arange(504) / 12)
base_sst     = 28.0 - 2.0 * np.abs(lats / 30)[:, None]  # (lat, 1)

# Broadcast: (time, lat, lon)
sst = (base_sst[None, :, :]
       + enso_signal[:, None, None] * np.cos(np.deg2rad(lats))[None, :, None]
       + seasonal[:, None, None]
       + 0.3 * rng.standard_normal((504, 61, 171))).astype("float32")

ds = xr.Dataset(
    {"sst": (["time", "lat", "lon"], sst,
              {"units": "degC", "long_name": "Sea Surface Temperature"})},
    coords={"time": times, "lat": lats, "lon": lons},
    attrs={"Conventions": "CF-1.10", "title": "Synthetic tropical SST"},
)

# ── Compute SST climatology (1980-2009 baseline) ─────────────────────────────
clim_period = ds.sel(time=slice("1980-01-01", "2009-12-31"))
climatology  = clim_period["sst"].groupby("time.month").mean("time")
print("Climatology shape:", climatology.shape)   # (12, lat, lon)

# ── Monthly anomaly ───────────────────────────────────────────────────────────
anomaly = ds["sst"].groupby("time.month") - climatology
print("Anomaly shape:", anomaly.shape)

# ── Nino 3.4 index: area mean in 5°S-5°N, 190°-240°E ─────────────────────────
nino34_box = anomaly.sel(lat=slice(-5, 5), lon=slice(190, 240))
weights    = np.cos(np.deg2rad(nino34_box.lat))
nino34     = nino34_box.weighted(weights).mean(["lat", "lon"])

# 5-month running mean (standard smoothing)
nino34_smooth = nino34.rolling(time=5, center=True, min_periods=3).mean()

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(nino34.time.values, nino34_smooth.values,
                where=nino34_smooth.values > 0.5,
                color="red", alpha=0.5, label="El Nino (>0.5°C)")
ax.fill_between(nino34.time.values, nino34_smooth.values,
                where=nino34_smooth.values < -0.5,
                color="blue", alpha=0.5, label="La Nina (<-0.5°C)")
ax.plot(nino34.time.values, nino34_smooth.values, "k", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Nino 3.4 SST anomaly (°C)")
ax.set_title("Nino 3.4 index — synthetic dataset")
ax.legend(); plt.tight_layout()
plt.savefig("enso_index.png", dpi=150)
print("Peak Nino 3.4 value:", float(nino34_smooth.max()))
```

### Example 2 — Vertical Profile Analysis with Pressure Levels

Work with 4-D (time, level, lat, lon) atmospheric data, compute geopotential
thickness, and save diagnostics to Zarr.

```python
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_DIR  = os.getenv("NC_DATA_DIR", "./data")
ZARR_OUT  = os.path.join(DATA_DIR, "thickness.zarr")
os.makedirs(DATA_DIR, exist_ok=True)

rng = np.random.default_rng(42)

# ── Create synthetic 4-D pressure-level dataset ───────────────────────────────
pressure_levels = np.array([1000, 925, 850, 700, 500, 300, 200, 100],
                            dtype="float32")  # hPa
times = pd.date_range("2020-01-01", periods=60, freq="D")
lats  = np.linspace(-90, 90, 37)
lons  = np.linspace(0, 355, 72)

# Temperature: lapse rate + random noise + seasonal
T_base   = 290 - 6.5e-3 * 8000 * (1 - pressure_levels / 1000)[:, None, None]
T_data   = (T_base[None, :, :, :]
            + 10 * np.cos(np.deg2rad(lats))[None, None, :, None]
            + rng.standard_normal((60, 8, 37, 72)).astype("float32") * 2)

# Geopotential height (synthetic, proportional to log-pressure)
Z_data = (-(8e3 * np.log(pressure_levels / 1013.25))[:, None, None]
           + rng.standard_normal((60, 8, 37, 72)).astype("float32") * 50).astype("float32")

ds = xr.Dataset(
    {
        "ta": (["time", "plev", "lat", "lon"], T_data.astype("float32"),
               {"units": "K", "standard_name": "air_temperature",
                "long_name": "Air Temperature"}),
        "zg": (["time", "plev", "lat", "lon"], Z_data,
               {"units": "m", "standard_name": "geopotential_height",
                "long_name": "Geopotential Height"}),
    },
    coords={
        "time": times,
        "plev": xr.Variable("plev", pressure_levels,
                             attrs={"units": "hPa", "axis": "Z",
                                    "positive": "down"}),
        "lat": lats, "lon": lons,
    },
    attrs={"Conventions": "CF-1.10"},
)

# ── 1000-500 hPa thickness ────────────────────────────────────────────────────
thickness = (ds["zg"].sel(plev=500)
             - ds["zg"].sel(plev=1000))
thickness.attrs = {"units": "m", "long_name": "1000-500 hPa Thickness"}

# ── Tropopause temperature (minimum T in column above 200 hPa) ────────────────
strat = ds["ta"].sel(plev=slice(None, 300))   # 300 hPa and above (lower pressure)
tropopause_T = strat.min(dim="plev")

# ── Zonal mean cross-section ──────────────────────────────────────────────────
zonal_mean_T = ds["ta"].mean("lon").mean("time")  # (plev, lat)

fig, ax = plt.subplots(figsize=(10, 6))
cf = ax.contourf(lats, pressure_levels, zonal_mean_T.values,
                 levels=20, cmap="RdBu_r")
ax.set_ylim(1000, 100)
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Pressure (hPa)")
ax.set_title("Zonal mean temperature (K)")
plt.colorbar(cf, ax=ax, label="K")
plt.tight_layout()
plt.savefig("zonal_mean_T.png", dpi=150)

# ── Save diagnostics to Zarr ─────────────────────────────────────────────────
diag_ds = xr.Dataset(
    {"thickness":    thickness,
     "tropopause_T": tropopause_T},
)
diag_ds.to_zarr(ZARR_OUT, mode="w")
print(f"Diagnostics saved to {ZARR_OUT}")

# Verify round-trip
check = xr.open_zarr(ZARR_OUT)
print("Zarr thickness range:",
      float(check["thickness"].min()), "to",
      float(check["thickness"].max()), "m")
check.close()
```
