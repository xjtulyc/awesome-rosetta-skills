---
name: soil-data
description: >
  Soil data analysis via SoilGrids 2.0 REST API and SSURGO: SOC stocks, texture
  classification, kriging interpolation, and soil profile visualization.
tags:
  - soil-science
  - agriculture
  - soilgrids
  - geospatial
  - carbon-stocks
  - geostatistics
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
  - requests>=2.31.0
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - scikit-learn>=1.3.0
  - pyarrow>=12.0.0
last_updated: "2026-03-17"
---

# soil-data: Soil Data Analysis

This skill covers retrieving, processing, and analysing soil data from global and
national databases. Topics include the SoilGrids 2.0 REST API, soil organic carbon
stock calculation, USDA texture classification, simple kriging interpolation, and
depth-profile visualisation.

## Installation

```bash
pip install requests numpy pandas scipy matplotlib scikit-learn pyarrow
# R packages (run inside R):
# install.packages(c("soilDB", "aqp", "sf"))
```

---

## 1. SoilGrids 2.0 REST API

SoilGrids 2.0 (ISRIC) provides global predictions at 250 m resolution for key soil
properties at six standard depth intervals (0-5, 5-15, 15-30, 30-60, 60-100,
100-200 cm).

Available properties: `bdod` (bulk density), `clay`, `silt`, `sand`, `phh2o` (pH),
`soc` (soil organic carbon), `cec` (cation exchange capacity), `nitrogen`, `ocd`
(organic carbon density), `ocs` (organic carbon stock).

### 1.1 Point Query

```python
import requests
import pandas as pd
import numpy as np

SOILGRIDS_BASE = "https://rest.isric.org/soilgrids/v2.0/properties/query"

DEPTH_LABELS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
DEPTH_CM = [5, 15, 30, 60, 100, 200]          # lower boundary
DEPTH_THICKNESS = [5, 10, 15, 30, 40, 100]    # thickness in cm

DEFAULT_PROPERTIES = ["bdod", "clay", "silt", "sand", "phh2o", "soc", "cec"]


def get_soilgrids_point(
    lat: float,
    lon: float,
    properties: list = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Query SoilGrids 2.0 for a single point and return a tidy DataFrame.

    Parameters
    ----------
    lat, lon : float  — WGS84 coordinates
    properties : list  — soil properties to retrieve (default: all seven)
    timeout : int  — HTTP timeout in seconds

    Returns
    -------
    pd.DataFrame with columns: property, depth, mean, uncertainty_5pct,
    uncertainty_95pct, unit
    """
    if properties is None:
        properties = DEFAULT_PROPERTIES

    params = {
        "lon": lon,
        "lat": lat,
        "property": properties,
        "depth": DEPTH_LABELS,
        "value": ["mean", "uncertainty"],
    }
    resp = requests.get(SOILGRIDS_BASE, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    records = []
    for layer in data["properties"]["layers"]:
        prop_name = layer["name"]
        unit = layer.get("unit_measure", {}).get("mapped_units", "")
        conv = layer.get("unit_measure", {}).get("conversion_factor", 1.0)
        for depth_data in layer["depths"]:
            depth_label = depth_data["label"]
            values = depth_data.get("values", {})
            records.append({
                "property": prop_name,
                "depth": depth_label,
                "mean": values.get("mean", np.nan) * (1 / conv if conv else 1),
                "Q0.05": values.get("Q0.05", np.nan) * (1 / conv if conv else 1),
                "Q0.95": values.get("Q0.95", np.nan) * (1 / conv if conv else 1),
                "unit": unit,
            })
    return pd.DataFrame(records)
```

### 1.2 Bounding-Box Grid Query

```python
def get_soilgrids_bbox(
    bbox: tuple,
    property: str = "soc",
    depth: str = "0-5cm",
    n_points: int = 25,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Sample a soil property on a regular grid within a bounding box.

    Parameters
    ----------
    bbox : (min_lon, min_lat, max_lon, max_lat)
    property : str  — SoilGrids property code
    depth : str  — depth label, e.g. '0-5cm'
    n_points : int  — approximate number of grid points (square root taken per axis)

    Returns
    -------
    pd.DataFrame with columns: lat, lon, property, depth, mean, unit
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    side = max(2, int(np.sqrt(n_points)))
    lons = np.linspace(min_lon, max_lon, side)
    lats = np.linspace(min_lat, max_lat, side)

    records = []
    total = side * side
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            try:
                df_pt = get_soilgrids_point(lat, lon, properties=[property], timeout=timeout)
                row = df_pt[df_pt["depth"] == depth]
                if not row.empty:
                    records.append({
                        "lat": lat,
                        "lon": lon,
                        "property": property,
                        "depth": depth,
                        "mean": float(row["mean"].iloc[0]),
                        "unit": row["unit"].iloc[0],
                    })
                    print(f"  [{i*side+j+1}/{total}] ({lat:.3f}, {lon:.3f}): "
                          f"{property} = {records[-1]['mean']:.2f}")
            except Exception as exc:
                print(f"  WARN ({lat:.3f}, {lon:.3f}): {exc}")

    return pd.DataFrame(records)
```

---

## 2. Derived Calculations

### 2.1 Soil Organic Carbon Stock

The SOC stock (kg C m⁻²) for a single depth interval is:

    SOC_stock = SOC_pct/100 × BD_g_cm3 × 1000 × thickness_cm × 0.01 × (1 - coarse_frac)

```python
def compute_soc_stock(
    bulk_density_gcm3: float,
    soc_pct: float,
    depth_cm: float,
    coarse_fragment_vol_pct: float = 0.0,
) -> float:
    """
    Compute soil organic carbon stock for one depth interval.

    Parameters
    ----------
    bulk_density_gcm3 : float  — fine-earth bulk density (g cm⁻³)
    soc_pct : float            — SOC concentration (g kg⁻¹ → divide by 10 for %)
                                 NOTE: SoilGrids 'soc' is in dg/kg; pass value/100
    depth_cm : float           — layer thickness (cm)
    coarse_fragment_vol_pct : float  — volumetric coarse fragment content (%)

    Returns
    -------
    float  — SOC stock in kg C m⁻²
    """
    cf_fraction = coarse_fragment_vol_pct / 100.0
    # Convert: soc_pct (g/100g) × BD (g/cm³) × depth (cm) → g/cm² → kg/m²
    stock_kg_m2 = (soc_pct / 100.0) * bulk_density_gcm3 * depth_cm * 100 * (1 - cf_fraction)
    # factor 100 converts cm-depth to m² accounting for unit normalisation
    # exact: (g_soc/g_soil) × (g_soil/cm³) × cm × (10000 cm²/m²) / (1000 g/kg) = kg/m²
    stock_kg_m2 = (soc_pct / 100.0) * bulk_density_gcm3 * depth_cm * 10.0 * (1 - cf_fraction)
    return stock_kg_m2


def compute_profile_soc_stock(df_profile: pd.DataFrame) -> float:
    """
    Sum SOC stocks across all depth intervals in a SoilGrids profile DataFrame.

    Expects columns: property, depth, mean.
    Uses matched bulk-density and SOC rows.

    Returns total SOC stock in kg C m⁻² for the full profile depth.
    """
    soc_rows = df_profile[df_profile["property"] == "soc"].set_index("depth")["mean"]
    bd_rows = df_profile[df_profile["property"] == "bdod"].set_index("depth")["mean"]

    depth_thickness_map = dict(zip(DEPTH_LABELS, DEPTH_THICKNESS))
    total = 0.0
    for depth_label in DEPTH_LABELS:
        if depth_label in soc_rows.index and depth_label in bd_rows.index:
            soc_val = soc_rows[depth_label]  # g/kg (SoilGrids mapped units)
            bd_val = bd_rows[depth_label]    # cg/cm³ → need /100 for g/cm³
            thickness = depth_thickness_map[depth_label]
            soc_pct = soc_val / 10.0        # g/kg → %
            bd_gcm3 = bd_val / 100.0        # cg/cm³ → g/cm³
            total += compute_soc_stock(bd_gcm3, soc_pct, thickness)
    return total
```

### 2.2 USDA Soil Texture Classification

```python
def classify_texture(sand_pct: float, clay_pct: float) -> str:
    """
    Classify soil texture using the USDA texture triangle.

    Parameters
    ----------
    sand_pct, clay_pct : float  — percentage by mass (silt = 100 - sand - clay)

    Returns
    -------
    str  — USDA texture class name
    """
    silt_pct = 100.0 - sand_pct - clay_pct
    if not (0 <= sand_pct <= 100 and 0 <= clay_pct <= 100 and 0 <= silt_pct <= 100):
        raise ValueError(f"Invalid fractions: sand={sand_pct}, clay={clay_pct}, silt={silt_pct}")

    # USDA classification rules (simplified, covers all 12 classes)
    if clay_pct >= 40:
        if sand_pct <= 45 and silt_pct <= 40:
            return "clay"
        elif sand_pct > 45:
            return "sandy clay"
        else:
            return "silty clay"
    elif clay_pct >= 27:
        if sand_pct > 20 and silt_pct < 28:
            return "sandy clay loam" if sand_pct >= 45 else "clay loam"
        elif silt_pct >= 28:
            return "silty clay loam"
        else:
            return "clay loam"
    elif clay_pct >= 7:
        if sand_pct >= 52 and clay_pct < 20:
            return "sandy loam"
        elif sand_pct < 52 and silt_pct >= 28:
            return "silt loam"
        elif clay_pct >= 7 and clay_pct < 27 and sand_pct < 52 and silt_pct < 28:
            return "loam"
        else:
            return "sandy loam"
    else:
        if sand_pct >= 85:
            return "sand"
        elif sand_pct >= 70:
            return "loamy sand"
        elif silt_pct >= 80:
            return "silt"
        else:
            return "sandy loam"
```

---

## 3. Visualisation

### 3.1 Soil Profile Depth Plot

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_soil_profile(
    depths: list,
    values: list,
    property_name: str,
    unit: str = "",
    title: str = None,
    ax=None,
    color: str = "#8B4513",
):
    """
    Horizontal bar chart showing a soil property vs. depth.

    Parameters
    ----------
    depths : list of str  — depth labels, e.g. DEPTH_LABELS
    values : list of float  — property values at each depth
    property_name : str
    unit : str
    title : str
    ax : matplotlib Axes or None
    color : str

    Returns
    -------
    fig, ax
    """
    thickness_map = dict(zip(DEPTH_LABELS, DEPTH_THICKNESS))
    y_starts = [0, 5, 15, 30, 60, 100]

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 7))
    else:
        fig = ax.figure

    for i, (label, val) in enumerate(zip(depths, values)):
        thick = thickness_map.get(label, 10)
        y_start = y_starts[i] if i < len(y_starts) else sum(DEPTH_THICKNESS[:i])
        bar = ax.barh(
            y=-y_start - thick / 2,
            width=val if not np.isnan(val) else 0,
            height=thick * 0.9,
            color=color,
            alpha=0.75,
            label=label,
        )
        ax.text(
            val * 0.02 if not np.isnan(val) else 0.02,
            -y_start - thick / 2,
            f"{val:.1f}" if not np.isnan(val) else "N/A",
            va="center",
            fontsize=8,
        )

    ax.set_yticks([-y - t / 2 for y, t in zip(y_starts, DEPTH_THICKNESS)])
    ax.set_yticklabels(depths, fontsize=9)
    ax.set_xlabel(f"{property_name} ({unit})" if unit else property_name)
    ax.set_title(title or f"Soil Profile — {property_name}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax
```

---

## 4. Kriging Interpolation (Ordinary Kriging via Variogram)

```python
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit


def _spherical_variogram(h, nugget, sill, range_):
    """Spherical variogram model γ(h)."""
    h = np.asarray(h, dtype=float)
    result = np.where(
        h <= range_,
        nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
        sill,
    )
    return result


def _fit_variogram(coords: np.ndarray, values: np.ndarray, n_lags: int = 15):
    """
    Fit a spherical variogram to experimental semi-variance.

    coords : (n, 2) array of (lon, lat)
    values : (n,) array
    Returns fitted parameters (nugget, sill, range).
    """
    dists = cdist(coords, coords)
    diffs_sq = (values[:, None] - values[None, :]) ** 2

    max_dist = np.percentile(dists[dists > 0], 50)
    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lag_centers = (lag_edges[:-1] + lag_edges[1:]) / 2

    semivariance = []
    for i in range(n_lags):
        mask = (dists > lag_edges[i]) & (dists <= lag_edges[i + 1])
        if mask.sum() > 0:
            semivariance.append(np.mean(diffs_sq[mask]) / 2)
        else:
            semivariance.append(np.nan)

    valid = ~np.isnan(semivariance)
    lc = lag_centers[valid]
    sv = np.array(semivariance)[valid]

    p0 = [0.0, float(np.var(values)), max_dist / 2]
    try:
        popt, _ = curve_fit(_spherical_variogram, lc, sv, p0=p0, maxfev=5000)
        return popt  # nugget, sill, range
    except RuntimeError:
        return p0


def krige_ordinary(
    known_coords: np.ndarray,
    known_values: np.ndarray,
    target_coords: np.ndarray,
) -> np.ndarray:
    """
    Simple ordinary kriging interpolation using a fitted spherical variogram.

    Parameters
    ----------
    known_coords : (n, 2)  — (lon, lat) of observation points
    known_values : (n,)    — observed soil property values
    target_coords : (m, 2) — grid points to interpolate

    Returns
    -------
    predicted : (m,)  — kriged predictions at target_coords
    """
    nugget, sill, range_ = _fit_variogram(known_coords, known_values)
    n = len(known_coords)

    # Build kriging system
    C = _spherical_variogram(cdist(known_coords, known_coords), nugget, sill, range_)
    C_aug = np.block([[C, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])

    predicted = np.zeros(len(target_coords))
    for i, tc in enumerate(target_coords):
        c0 = _spherical_variogram(
            cdist([tc], known_coords)[0], nugget, sill, range_
        )
        c0_aug = np.append(c0, 1.0)
        try:
            weights = np.linalg.solve(C_aug, c0_aug)
            predicted[i] = float(weights[:n] @ known_values)
        except np.linalg.LinAlgError:
            predicted[i] = np.mean(known_values)

    return predicted
```

---

## 5. SSURGO via soilDB (R)

```r
# Install once:
# install.packages(c("soilDB", "aqp", "sf", "ggplot2"))

library(soilDB)
library(aqp)
library(sf)

## Fetch SSURGO data for a set of map unit keys
fetch_ssurgo_profiles <- function(mukeys) {
  # Build SQL to get horizon data
  q <- sprintf(
    "SELECT cokey, hzname, hzdept_r, hzdepb_r, sandtotal_r,
            silttotal_r, claytotal_r, om_r, ph1to1h2o_r, dbthirdbar_r
     FROM chorizon
     WHERE cokey IN (
       SELECT cokey FROM component WHERE mukey IN (%s)
     )
     ORDER BY cokey, hzdept_r",
    paste(mukeys, collapse = ",")
  )
  hz <- SDA_query(q)
  return(hz)
}

## Compute SOC stock for SSURGO horizons
compute_ssurgo_soc <- function(hz_df) {
  hz_df$thickness_cm <- hz_df$hzdepb_r - hz_df$hzdept_r
  hz_df$soc_pct      <- hz_df$om_r / 1.724   # SOM to SOC conversion
  hz_df$bd_gcm3      <- hz_df$dbthirdbar_r
  hz_df$soc_stock    <- with(hz_df,
    (soc_pct / 100) * bd_gcm3 * thickness_cm * 10
  )  # kg C m⁻²
  total_stock <- tapply(hz_df$soc_stock, hz_df$cokey, sum, na.rm = TRUE)
  return(total_stock)
}

# Example usage:
# mukeys <- c("2494753", "2494754", "2494755")
# hz_data <- fetch_ssurgo_profiles(mukeys)
# soc_by_component <- compute_ssurgo_soc(hz_data)
# print(soc_by_component)
```

---

## 6. Examples

### Example A — Map SOC Stocks Across a Regional Grid

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define study area (e.g., part of the US Corn Belt)
BBOX = (-91.0, 41.0, -88.0, 43.0)   # (min_lon, min_lat, max_lon, max_lat)

# Step 1: Sample SOC at a coarse grid (25 points) via SoilGrids
df_grid = get_soilgrids_bbox(
    bbox=BBOX,
    property="soc",
    depth="0-5cm",
    n_points=25,
)
print(df_grid.head())

# Step 2: Krige to a finer grid
known_coords = df_grid[["lon", "lat"]].values
known_values = df_grid["mean"].values

# Build 10x10 target grid
lons = np.linspace(BBOX[0], BBOX[2], 10)
lats = np.linspace(BBOX[1], BBOX[3], 10)
lon_grid, lat_grid = np.meshgrid(lons, lats)
target_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

soc_predicted = krige_ordinary(known_coords, known_values, target_coords)
soc_map = soc_predicted.reshape(10, 10)

# Step 3: Visualise
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc = axes[0].scatter(df_grid["lon"], df_grid["lat"], c=df_grid["mean"],
                     cmap="YlOrBr", s=80, edgecolors="k")
plt.colorbar(sc, ax=axes[0], label="SOC (g/kg)")
axes[0].set_title("Observed SOC (0-5 cm)")
axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")

im = axes[1].imshow(
    soc_map, origin="lower", cmap="YlOrBr",
    extent=[BBOX[0], BBOX[2], BBOX[1], BBOX[3]], aspect="auto",
)
plt.colorbar(im, ax=axes[1], label="SOC (g/kg)")
axes[1].set_title("Kriged SOC (0-5 cm)")
axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("/tmp/soc_map.png", dpi=150)
plt.show()
print("Saved map to /tmp/soc_map.png")
```

### Example B — Compare Soil Texture Across Multiple Field Sites

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Field site coordinates
sites = [
    {"name": "Site A — Iowa",         "lat": 42.0308, "lon": -93.6319},
    {"name": "Site B — Kansas",       "lat": 38.9717, "lon": -95.2353},
    {"name": "Site C — Nebraska",     "lat": 41.2565, "lon": -95.9345},
    {"name": "Site D — Illinois",     "lat": 40.6331, "lon": -89.3985},
]

# Retrieve sand/silt/clay at 0-30 cm for each site
records = []
for site in sites:
    df = get_soilgrids_point(
        lat=site["lat"],
        lon=site["lon"],
        properties=["sand", "silt", "clay"],
    )
    for depth in ["0-5cm", "5-15cm", "15-30cm"]:
        row = {"site": site["name"], "depth": depth}
        for prop in ["sand", "silt", "clay"]:
            val = df[(df["property"] == prop) & (df["depth"] == depth)]["mean"]
            row[prop] = float(val.iloc[0]) / 10.0 if not val.empty else np.nan
        records.append(row)

df_tex = pd.DataFrame(records)

# Add texture classification
df_tex["texture_class"] = df_tex.apply(
    lambda r: classify_texture(r["sand"], r["clay"])
    if not (np.isnan(r["sand"]) or np.isnan(r["clay"])) else "unknown",
    axis=1,
)
print(df_tex.to_string(index=False))

# Ternary-style bar chart: stacked sand/silt/clay per site × depth
fig, ax = plt.subplots(figsize=(12, 5))
bar_width = 0.2
n_depths = 3
depth_labels = ["0-5cm", "5-15cm", "15-30cm"]
colours = {"sand": "#f4c542", "silt": "#a07850", "clay": "#c0392b"}
x = np.arange(len(sites))

for d_idx, depth in enumerate(depth_labels):
    sub = df_tex[df_tex["depth"] == depth].set_index("site")
    offset = (d_idx - 1) * bar_width
    bottom = np.zeros(len(sites))
    for frac in ["sand", "silt", "clay"]:
        vals = [sub.loc[s["name"], frac] if s["name"] in sub.index else 0
                for s in sites]
        ax.bar(x + offset, vals, bar_width * 0.9, bottom=bottom,
               color=colours[frac], label=frac if d_idx == 0 else "")
        bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([s["name"] for s in sites], rotation=15, ha="right")
ax.set_ylabel("Percentage (%)")
ax.set_title("Soil Texture by Site and Depth (SoilGrids 2.0)")
handles = [mpatches.Patch(color=v, label=k) for k, v in colours.items()]
ax.legend(handles=handles, loc="upper right")
plt.tight_layout()
plt.savefig("/tmp/texture_comparison.png", dpi=150)
plt.show()
print("Saved to /tmp/texture_comparison.png")
df_tex.to_csv("/tmp/soil_texture_sites.csv", index=False)
```

---

## 7. Tips and Gotchas

- **SoilGrids units**: Values returned by the API are in *mapped* units (e.g., bulk
  density in cg/cm³, SOC in dg/kg). Always divide by the `conversion_factor` in the
  response or check the `unit_measure` field before using values.
- **Rate limiting**: ISRIC does not publish a strict rate limit, but space requests 1-2 s
  apart for large grids to avoid HTTP 429 errors.
- **SSURGO coverage**: SSURGO covers the conterminous US only. Use SoilGrids or
  FAO/HWSD for global work.
- **Texture triangle edge cases**: Many simplified triangle implementations misclassify
  points near class boundaries. Validate against the official USDA NRCS chart.
- **Kriging assumptions**: Ordinary kriging assumes second-order stationarity. Always
  inspect the experimental variogram before trusting kriged maps.
- **SOC stock uncertainty**: Bulk density is often the largest source of error. Where
  available, use locally measured BD rather than SoilGrids estimates.

---

## 8. References

- Poggio et al. (2021). SoilGrids 2.0: producing soil information for the globe with
  quantified spatial uncertainty. *SOIL*, 7, 217-240.
- Soil Survey Staff (2022). *Keys to Soil Taxonomy*, 13th ed. USDA-NRCS.
- Beaudette, D. et al. (2023). soilDB: Soil Database Interface. R package.
  https://CRAN.R-project.org/package=soilDB
