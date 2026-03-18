---
name: ocean-data
description: Download and analyze oceanographic data from Copernicus Marine Service and Argo floats using copernicusmarine, gsw, and xarray.
tags:
  - oceanography
  - copernicus
  - argo
  - xarray
  - earth-science
version: "1.0.0"
authors:
  - name: "Rosetta Skills Contributors"
    github: "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - copernicusmarine>=1.0
  - gsw>=3.6
  - xarray>=2023.6
  - pandas>=2.0
  - matplotlib>=3.7
  - numpy>=1.24
  - cartopy>=0.22
last_updated: "2026-03-17"
status: stable
---

# Ocean Data Analysis with Copernicus Marine Service and Argo Floats

Retrieve, process, and visualize ocean physical and biogeochemical data using
the Copernicus Marine Service (CMEMS) Python client, Argo float profiles, and
the TEOS-10 Gibbs SeaWater (GSW) toolbox.

---

## When to Use This Skill

- You need sea surface temperature, salinity, or current fields for a specific
  region and time period from the Copernicus Marine Service catalog.
- You want to download individual Argo float profiles and compute derived
  quantities such as mixed layer depth (MLD) or potential density.
- You need to produce T-S diagrams, eddy kinetic energy (EKE) maps, or
  mixed-layer climatologies for research or operational purposes.
- You are building ocean-climate pipelines that combine model output with
  in-situ observations.

---

## Background & Key Concepts

### Copernicus Marine Service (CMEMS)
The Copernicus Marine Environment Monitoring Service provides free,
quality-controlled ocean datasets spanning physical, biogeochemical, and sea-ice
variables. Products include reanalysis, near-real-time, and forecast datasets.
The `copernicusmarine` Python package (v1+) replaced the legacy `motuclient`
and `ftplib` workflows.

### Argo Float Program
Argo is a global array of ~4000 autonomous profiling floats that measure
temperature and salinity from 2000 m to the surface every ~10 days. Data are
freely available via GDAC servers (Ifremer / BODC) and through the
`copernicusmarine` catalog.

### TEOS-10 / GSW Toolbox
The Thermodynamic Equation of Seawater 2010 (TEOS-10) defines Absolute
Salinity (SA) and Conservative Temperature (CT) as the preferred variables.
The `gsw` Python library converts Practical Salinity and in-situ temperature to
TEOS-10 variables and computes derived quantities such as potential density,
buoyancy frequency, and mixed layer depth.

### Mixed Layer Depth (MLD)
MLD is commonly estimated using a density threshold criterion: the depth at
which potential density exceeds the surface value by 0.03 kg/m³ (de Boyer
Montégut et al., 2004).

### Eddy Kinetic Energy (EKE)
EKE quantifies mesoscale variability: EKE = 0.5 * (u'^2 + v'^2), where u' and
v' are anomalies of eastward and northward geostrophic currents relative to a
long-term mean.

---

## Environment Setup

### Install dependencies

```bash
pip install "copernicusmarine>=1.0" "gsw>=3.6" "xarray>=2023.6" \
    "pandas>=2.0" "matplotlib>=3.7" "numpy>=1.24" "cartopy>=0.22"
```

### Authenticate with Copernicus Marine Service

Register for a free account at <https://marine.copernicus.eu/> then store your
credentials:

```bash
# Option 1 – interactive login (writes ~/.copernicusmarine/credentials)
copernicusmarine login

# Option 2 – environment variables (CI/CD friendly)
export COPERNICUSMARINE_USERNAME="<your-username>"
export COPERNICUSMARINE_PASSWORD=$(cat ~/.cmems_passwd)   # read from file, never hardcode
```

```python
import os
import copernicusmarine as cm

username = os.getenv("COPERNICUSMARINE_USERNAME", "")
password = os.getenv("COPERNICUSMARINE_PASSWORD", "")

# Verify login works
cm.login(username=username, password=password, overwrite_configuration_file=True)
```

---

## Core Workflow

### Step 1 – Browse the Catalog and Download SST Data

```python
import copernicusmarine as cm
import xarray as xr

# List available datasets matching a keyword
catalog = cm.describe(contains=["SST", "Mediterranean"])
for entry in catalog.products[:5]:
    print(entry.product_id, "-", entry.title)

# Download a subset of the CMEMS global SST L4 product
ds = cm.open_dataset(
    dataset_id="cmems_obs-sst_glo_phy_l4_my_0.25deg",
    variables=["analysed_sst", "analysis_error"],
    minimum_longitude=-20.0,
    maximum_longitude=40.0,
    minimum_latitude=25.0,
    maximum_latitude=50.0,
    start_datetime="2023-01-01T00:00:00",
    end_datetime="2023-03-31T23:59:59",
)
print(ds)
# Convert from Kelvin to Celsius
ds["sst_celsius"] = ds["analysed_sst"] - 273.15
ds["sst_celsius"].attrs["units"] = "degC"
```

### Step 2 – Download Argo Float Profiles and Compute MLD

```python
import gsw
import numpy as np
import xarray as xr
import copernicusmarine as cm

# Download Argo BGC profiles for the North Atlantic (float WMO 6902880)
argo = cm.open_dataset(
    dataset_id="cmems_obs-ins_glo_phy-temp-sal_nrt_argo_P1D-m",
    variables=["TEMP", "PSAL", "PRES", "LATITUDE", "LONGITUDE", "TIME"],
    minimum_longitude=-40.0,
    maximum_longitude=-20.0,
    minimum_latitude=40.0,
    maximum_latitude=60.0,
    start_datetime="2023-06-01T00:00:00",
    end_datetime="2023-08-31T23:59:59",
)

def compute_mld(temp, psal, pres, lat, threshold=0.03):
    """
    Compute mixed layer depth using a density threshold criterion.

    Parameters
    ----------
    temp  : array-like, in-situ temperature (ITS-90, °C)
    psal  : array-like, Practical Salinity (PSS-78)
    pres  : array-like, sea pressure (dbar)
    lat   : float, latitude (degrees N)
    threshold : float, density difference criterion (kg/m³), default 0.03

    Returns
    -------
    mld : float, mixed layer depth (m)
    """
    SA = gsw.SA_from_SP(psal, pres, 0.0, lat)
    CT = gsw.CT_from_t(SA, temp, pres)
    sigma0 = gsw.sigma0(SA, CT)          # potential density anomaly (kg/m³)

    # Reference density at shallowest valid level
    valid = np.isfinite(sigma0)
    if valid.sum() < 3:
        return np.nan
    ref_density = sigma0[valid][0]

    # Find first depth where density exceeds reference + threshold
    exceeds = np.where((sigma0 - ref_density) > threshold)[0]
    if len(exceeds) == 0:
        return float(pres[valid][-1])    # whole profile is mixed
    return float(pres[exceeds[0]])

# Apply to each profile
mld_values = []
for i in range(min(50, argo.dims.get("N_PROF", 0))):
    profile = argo.isel(N_PROF=i)
    mld = compute_mld(
        profile["TEMP"].values,
        profile["PSAL"].values,
        profile["PRES"].values,
        float(profile["LATITUDE"].values),
    )
    mld_values.append(mld)

print(f"Mean MLD (N=50 profiles): {np.nanmean(mld_values):.1f} dbar")
```

### Step 3 – Compute Eddy Kinetic Energy from Altimetry

```python
import copernicusmarine as cm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Download gridded surface current anomalies (AVISO altimetry)
alt = cm.open_dataset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
    variables=["ugosa", "vgosa"],        # geostrophic current anomalies
    minimum_longitude=-80.0,
    maximum_longitude=-40.0,
    minimum_latitude=25.0,
    maximum_latitude=55.0,
    start_datetime="2023-01-01T00:00:00",
    end_datetime="2023-12-31T23:59:59",
)

# Compute time-mean EKE (m²/s²)
eke = 0.5 * (alt["ugosa"] ** 2 + alt["vgosa"] ** 2)
eke_mean = eke.mean(dim="time")

# Plot
fig, ax = plt.subplots(
    subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 7)
)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--")

pcm = ax.pcolormesh(
    eke_mean.longitude,
    eke_mean.latitude,
    eke_mean.values * 1e4,               # convert to cm²/s²
    cmap="plasma",
    transform=ccrs.PlateCarree(),
    vmin=0,
    vmax=500,
)
plt.colorbar(pcm, ax=ax, label="EKE (cm²/s²)", shrink=0.8)
ax.set_title("Annual Mean Eddy Kinetic Energy – North Atlantic 2023")
plt.tight_layout()
plt.savefig("eke_north_atlantic_2023.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### T-S Diagram with Density Contours

```python
import gsw
import numpy as np
import matplotlib.pyplot as plt

def ts_diagram(temp_array, psal_array, pres_array, lat, color_by=None,
               title="T-S Diagram"):
    """
    Plot a Temperature-Salinity diagram with TEOS-10 potential density contours.

    Parameters
    ----------
    temp_array : 2-D array (N_PROF, N_LEVELS)
    psal_array : 2-D array (N_PROF, N_LEVELS)
    pres_array : 2-D array (N_PROF, N_LEVELS)
    lat        : float, representative latitude
    color_by   : 1-D array (N_PROF,) used to color-code profiles, optional
    title      : str
    """
    SA_all = gsw.SA_from_SP(psal_array, pres_array, 0.0, lat)
    CT_all = gsw.CT_from_t(SA_all, temp_array, pres_array)

    # Density contour grid
    sa_grid = np.linspace(SA_all[np.isfinite(SA_all)].min() - 0.5,
                          SA_all[np.isfinite(SA_all)].max() + 0.5, 100)
    ct_grid = np.linspace(CT_all[np.isfinite(CT_all)].min() - 1,
                          CT_all[np.isfinite(CT_all)].max() + 1, 100)
    SA_g, CT_g = np.meshgrid(sa_grid, ct_grid)
    sigma0_grid = gsw.sigma0(SA_g, CT_g)

    fig, ax = plt.subplots(figsize=(8, 7))
    cs = ax.contour(sa_grid, ct_grid, sigma0_grid,
                    levels=np.arange(22, 30, 0.5), colors="gray",
                    linewidths=0.6, linestyles="--")
    ax.clabel(cs, fmt="%.1f", fontsize=8)

    sc = ax.scatter(SA_all.ravel(), CT_all.ravel(),
                    c=color_by if color_by is not None else "steelblue",
                    s=1, alpha=0.4, cmap="viridis")
    if color_by is not None:
        plt.colorbar(sc, ax=ax, label="Depth (m)")

    ax.set_xlabel("Absolute Salinity SA (g/kg)")
    ax.set_ylabel("Conservative Temperature CT (°C)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("ts_diagram.png", dpi=150)
    plt.show()


# Example usage with synthetic data
rng = np.random.default_rng(42)
n_prof, n_lev = 100, 50
pres_syn = np.tile(np.linspace(5, 500, n_lev), (n_prof, 1))
psal_syn = 35.0 + rng.normal(0, 0.3, (n_prof, n_lev))
temp_syn = 15.0 - pres_syn / 50 + rng.normal(0, 0.5, (n_prof, n_lev))

ts_diagram(temp_syn, psal_syn, pres_syn, lat=45.0,
           color_by=np.repeat(np.arange(n_prof), n_lev),
           title="Synthetic T-S Diagram – North Atlantic")
```

### Seasonal SST Cycle from Reanalysis

```python
import copernicusmarine as cm
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

ds = cm.open_dataset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables=["thetao"],
    minimum_longitude=-10.0,
    maximum_longitude=36.0,
    minimum_latitude=30.0,
    maximum_latitude=46.0,
    minimum_depth=0.0,
    maximum_depth=1.0,
    start_datetime="2010-01-01T00:00:00",
    end_datetime="2020-12-31T23:59:59",
)

sst = ds["thetao"].isel(depth=0)

# Compute climatological monthly mean
sst_clim = sst.groupby("time.month").mean(dim="time")

# Spatial mean over the basin
sst_basin = sst_clim.mean(dim=["latitude", "longitude"])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(np.arange(1, 13), sst_basin.values, "o-", color="firebrick", lw=2)
ax.set_xticks(np.arange(1, 13))
ax.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"])
ax.set_xlabel("Month")
ax.set_ylabel("SST (°C)")
ax.set_title("Mediterranean Sea – 2010–2020 SST Climatological Cycle")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("mediterranean_sst_climatology.png", dpi=150)
plt.show()
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Missing credentials` error | `COPERNICUSMARINE_USERNAME` / `PASSWORD` not set | Run `copernicusmarine login` or export env vars |
| `TimeoutError` on large downloads | Requesting too large a spatial/temporal domain | Split into smaller chunks; use `cm.subset()` to download files locally |
| `KeyError: 'N_PROF'` | Argo dataset dimension name differs by product | Inspect `ds.dims` and adapt index variable |
| `gsw` returns NaN arrays | Pressure must be in dbar and non-negative; salinity must be > 0 | Filter out fill values (typically 99999) before calling gsw functions |
| Cartopy install fails on Windows | Binary wheel not available | Use `conda install -c conda-forge cartopy` instead of pip |
| `404 Not Found` for dataset_id | Product retired or renamed in CMEMS catalog | Run `cm.describe(contains=["keyword"])` to find current product ID |

---

## External Resources

- Copernicus Marine Service catalog: <https://data.marine.copernicus.eu/products>
- `copernicusmarine` Python package docs: <https://toolbox-docs.marine.copernicus.eu/>
- GSW-Python documentation: <https://teos-10.github.io/GSW-Python/>
- Argo data management: <https://www.argodatamgt.org/>
- de Boyer Montégut et al. (2004) MLD climatology: <https://doi.org/10.1029/2004JC002378>
- AVISO altimetry products: <https://www.aviso.altimetry.fr/>
- xarray documentation: <https://docs.xarray.dev/>

---

## Examples

### Example 1 – Black Sea Surface Salinity Trend (2015–2023)

```python
import copernicusmarine as cm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats

# Download monthly mean surface salinity
ds = cm.open_dataset(
    dataset_id="cmems_mod_blk_phy_my_2.5km_P1M-m",
    variables=["so"],
    minimum_depth=0.0,
    maximum_depth=1.0,
    start_datetime="2015-01-01T00:00:00",
    end_datetime="2023-12-31T23:59:59",
)

sal = ds["so"].isel(depth=0)
basin_mean = sal.mean(dim=["latitude", "longitude"])

# Convert time to decimal year for linear regression
times = basin_mean["time"].values
decimal_years = [
    float(str(t)[:4]) + float(str(t)[5:7]) / 12
    for t in basin_mean["time"].dt.strftime("%Y-%m").values
]

slope, intercept, r, p, se = stats.linregress(decimal_years, basin_mean.values)
trend_line = slope * np.array(decimal_years) + intercept

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(decimal_years, basin_mean.values, color="steelblue", lw=1.5,
        label="Monthly mean salinity")
ax.plot(decimal_years, trend_line, "r--", lw=2,
        label=f"Trend: {slope*10:.3f} PSU/decade (p={p:.3f})")
ax.set_xlabel("Year")
ax.set_ylabel("Surface Salinity (PSU)")
ax.set_title("Black Sea Basin-Mean Surface Salinity 2015–2023")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("black_sea_salinity_trend.png", dpi=150)
plt.show()

print(f"Trend: {slope*10:.4f} PSU/decade, R²={r**2:.3f}, p={p:.4f}")
```

### Example 2 – Argo Float Profile Visualization with MLD Overlay

```python
import gsw
import numpy as np
import matplotlib.pyplot as plt
import copernicusmarine as cm

# Download a single float's profiles
argo = cm.open_dataset(
    dataset_id="cmems_obs-ins_glo_phy-temp-sal_nrt_argo_P1D-m",
    variables=["TEMP", "PSAL", "PRES", "TIME", "LATITUDE", "LONGITUDE"],
    minimum_longitude=-35.0,
    maximum_longitude=-30.0,
    minimum_latitude=45.0,
    maximum_latitude=50.0,
    start_datetime="2023-07-01T00:00:00",
    end_datetime="2023-09-30T23:59:59",
)

# Select first valid profile
i_prof = 0
pres = argo["PRES"].isel(N_PROF=i_prof).values
temp = argo["TEMP"].isel(N_PROF=i_prof).values
psal = argo["PSAL"].isel(N_PROF=i_prof).values
lat  = float(argo["LATITUDE"].isel(N_PROF=i_prof).values)

# Mask fill values
valid = (pres > 0) & (psal > 0) & np.isfinite(temp)
pres, temp, psal = pres[valid], temp[valid], psal[valid]

# Compute derived quantities
SA = gsw.SA_from_SP(psal, pres, 0.0, lat)
CT = gsw.CT_from_t(SA, temp, pres)
sigma0 = gsw.sigma0(SA, CT)

# MLD (density threshold 0.03 kg/m³)
ref = sigma0[0]
mld_idx = np.where(sigma0 - ref > 0.03)[0]
mld = pres[mld_idx[0]] if len(mld_idx) > 0 else pres[-1]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
ax_t, ax_s, ax_d = axes

ax_t.plot(CT, pres, "b-", lw=1.5)
ax_t.axhline(mld, color="red", ls="--", label=f"MLD={mld:.0f} m")
ax_t.set_xlabel("Conservative Temp CT (°C)")
ax_t.set_ylabel("Pressure (dbar)")
ax_t.invert_yaxis()
ax_t.legend(fontsize=9)
ax_t.grid(True, linestyle="--", alpha=0.4)
ax_t.set_title("Temperature")

ax_s.plot(SA, pres, "g-", lw=1.5)
ax_s.axhline(mld, color="red", ls="--")
ax_s.set_xlabel("Absolute Salinity SA (g/kg)")
ax_s.grid(True, linestyle="--", alpha=0.4)
ax_s.set_title("Salinity")

ax_d.plot(sigma0, pres, "k-", lw=1.5)
ax_d.axhline(mld, color="red", ls="--")
ax_d.set_xlabel("Potential Density σ₀ (kg/m³)")
ax_d.grid(True, linestyle="--", alpha=0.4)
ax_d.set_title("Potential Density")

plt.suptitle(f"Argo Profile – Lat {lat:.2f}°N", fontsize=12)
plt.tight_layout()
plt.savefig("argo_profile_mld.png", dpi=150)
plt.show()
```
