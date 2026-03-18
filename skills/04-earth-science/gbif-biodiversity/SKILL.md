---
name: gbif-biodiversity
description: Query GBIF occurrence data, apply spatial thinning, and build species distribution models using pygbif, geopandas, and elapid.
tags:
  - biodiversity
  - gbif
  - species-distribution-modeling
  - geopandas
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
  - pygbif>=0.6
  - geopandas>=0.14
  - scikit-learn>=1.3
  - elapid>=0.7
  - matplotlib>=3.7
  - numpy>=1.24
  - pandas>=2.0
  - cartopy>=0.22
last_updated: "2026-03-17"
status: stable
---

# Biodiversity Analysis with GBIF, pygbif, and Species Distribution Modeling

Download occurrence records from the Global Biodiversity Information Facility
(GBIF), clean and spatially thin them, and fit species distribution models
(SDMs) including MaxEnt and Boosted Regression Trees (BRT) using `elapid` and
`scikit-learn`.

---

## When to Use This Skill

- You need georeferenced occurrence records for one or more species from the
  GBIF database.
- You want to model the potential distribution of a species based on
  environmental predictors (climate, land cover, elevation).
- You are computing species richness maps or diversity indices across a region.
- You need to prepare a bias-corrected, spatially thinned occurrence dataset
  for downstream SDM analysis.

---

## Background & Key Concepts

### Global Biodiversity Information Facility (GBIF)
GBIF is an international open-data infrastructure providing access to hundreds
of millions of biodiversity occurrence records from museums, citizen science
platforms (e.g., iNaturalist), and research institutions. The GBIF API supports
both synchronous (small) and asynchronous bulk downloads.

### pygbif
`pygbif` is the official Python client for the GBIF API. It wraps the
occurrences, species, maps, and downloads endpoints.

### Spatial Thinning
Occurrence records are often spatially biased (clustered near roads, cities,
research stations). Spatial thinning retains at most one record per grid cell
of a specified resolution to reduce sampling bias before SDM fitting.

### MaxEnt (Maximum Entropy Modeling)
MaxEnt is a presence-background SDM that models species distributions by
finding the maximum-entropy probability distribution constrained to match
observed feature statistics. The `elapid` package provides a scikit-learn
compatible MaxEnt implementation.

### Boosted Regression Trees (BRT)
BRT (Gradient Boosted Machines applied to SDMs) often outperforms MaxEnt when
background/pseudo-absence data are carefully sampled. `scikit-learn`'s
`GradientBoostingClassifier` or `HistGradientBoostingClassifier` is used here.

### Species Richness Maps
Stack individual species prediction rasters and sum presence probabilities
(thresholded at a chosen cutoff) to produce a richness map.

---

## Environment Setup

### Install dependencies

```bash
pip install "pygbif>=0.6" "geopandas>=0.14" "scikit-learn>=1.3" \
    "elapid>=0.7" "matplotlib>=3.7" "numpy>=1.24" "pandas>=2.0" \
    "cartopy>=0.22"
```

### GBIF credentials (required for bulk downloads)

```bash
export GBIF_USER="<paste-your-gbif-username>"
export GBIF_PWD="<paste-your-gbif-password>"
export GBIF_EMAIL="<paste-your-gbif-email>"
```

```python
import os

GBIF_USER  = os.getenv("GBIF_USER", "")
GBIF_PWD   = os.getenv("GBIF_PWD", "")
GBIF_EMAIL = os.getenv("GBIF_EMAIL", "")
```

---

## Core Workflow

### Step 1 – Search and Download Occurrence Records

```python
import os
import time
import pandas as pd
from pygbif import occurrences as occ
from pygbif import species as sp

GBIF_USER  = os.getenv("GBIF_USER", "")
GBIF_PWD   = os.getenv("GBIF_PWD", "")
GBIF_EMAIL = os.getenv("GBIF_EMAIL", "")

def get_species_key(scientific_name: str) -> int:
    """Look up the GBIF backbone taxon key for a scientific name."""
    result = sp.name_backbone(name=scientific_name, rank="species")
    if result.get("matchType") == "NONE":
        raise ValueError(f"Species not found in GBIF backbone: {scientific_name}")
    return result["usageKey"]


def download_occurrences_sync(
    species_name: str,
    country: str = None,
    year_range: tuple = (2000, 2023),
    limit: int = 5000,
) -> pd.DataFrame:
    """
    Synchronous download of up to `limit` occurrence records via the GBIF API.
    For >100k records use the async download endpoint instead.

    Parameters
    ----------
    species_name : str, e.g. 'Panthera pardus'
    country      : ISO 2-letter country code, optional
    year_range   : (start_year, end_year)
    limit        : maximum records to retrieve

    Returns
    -------
    pd.DataFrame with columns including decimalLatitude, decimalLongitude, year
    """
    taxon_key = get_species_key(species_name)
    kwargs = {
        "taxonKey": taxon_key,
        "hasCoordinate": True,
        "hasGeospatialIssue": False,
        "year": f"{year_range[0]},{year_range[1]}",
        "limit": min(limit, 300),
        "offset": 0,
    }
    if country:
        kwargs["country"] = country

    records = []
    while len(records) < limit:
        result = occ.search(**kwargs)
        batch = result.get("results", [])
        records.extend(batch)
        if result.get("endOfRecords", True) or len(batch) == 0:
            break
        kwargs["offset"] += len(batch)
        time.sleep(0.2)  # polite rate limiting

    df = pd.json_normalize(records)
    required = ["decimalLatitude", "decimalLongitude", "species", "year"]
    existing = [c for c in required if c in df.columns]
    df = df[existing].dropna(subset=["decimalLatitude", "decimalLongitude"])
    print(f"Retrieved {len(df)} records for {species_name}")
    return df


# Example: Leopard occurrences in sub-Saharan Africa
leopard_df = download_occurrences_sync("Panthera pardus", limit=2000)
leopard_df.to_csv("panthera_pardus_occurrences.csv", index=False)
```

### Step 2 – Spatial Thinning with GeoPandas

```python
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def spatial_thin(
    df: pd.DataFrame,
    lat_col: str = "decimalLatitude",
    lon_col: str = "decimalLongitude",
    resolution_deg: float = 0.5,
    random_state: int = 42,
) -> gpd.GeoDataFrame:
    """
    Keep at most one occurrence per grid cell defined by `resolution_deg`.

    Parameters
    ----------
    df             : DataFrame with occurrence records
    lat_col        : column name for latitude
    lon_col        : column name for longitude
    resolution_deg : grid cell size in decimal degrees
    random_state   : for reproducibility of random selection within cells

    Returns
    -------
    gpd.GeoDataFrame of thinned occurrences
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df["cell_lat"] = (df[lat_col] / resolution_deg).round().astype(int)
    df["cell_lon"] = (df[lon_col] / resolution_deg).round().astype(int)
    df["cell_id"] = df["cell_lat"].astype(str) + "_" + df["cell_lon"].astype(str)

    thinned = (
        df.groupby("cell_id")
          .apply(lambda g: g.sample(1, random_state=int(rng.integers(1e6))))
          .reset_index(drop=True)
    )

    geometry = [Point(row[lon_col], row[lat_col]) for _, row in thinned.iterrows()]
    gdf = gpd.GeoDataFrame(thinned, geometry=geometry, crs="EPSG:4326")
    print(f"Thinned from {len(df)} to {len(gdf)} records at {resolution_deg}° resolution")
    return gdf


leopard_df = pd.read_csv("panthera_pardus_occurrences.csv")
leopard_thin = spatial_thin(leopard_df, resolution_deg=0.5)
leopard_thin.to_file("leopard_thinned.gpkg", driver="GPKG")
```

### Step 3 – Fit a MaxEnt SDM with elapid

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import elapid
import rasterio
from rasterio.transform import from_bounds

# ------------------------------------------------------------------
# Prepare environment rasters (WorldClim BIO1, BIO12 as synthetic demo)
# In production: load real GeoTIFF rasters via rasterio
# ------------------------------------------------------------------
def make_synthetic_env_raster(output_path: str,
                               bbox=(-20, -40, 55, 40),
                               res: float = 0.5):
    """Write two synthetic climate bands to a GeoTIFF for demonstration."""
    west, south, east, north = bbox
    width  = int((east - west) / res)
    height = int((north - south) / res)
    transform = from_bounds(west, south, east, north, width, height)

    rng = np.random.default_rng(0)
    lats = np.linspace(north, south, height)[:, None] * np.ones((height, width))
    bio1  = 25 - 0.5 * np.abs(lats) + rng.normal(0, 2, (height, width))  # mean annual temp
    bio12 = 800 + 10 * lats + rng.normal(0, 50, (height, width))           # annual precip

    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=height, width=width, count=2, dtype="float32",
        crs="EPSG:4326", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(bio1.astype("float32"), 1)
        dst.write(bio12.astype("float32"), 2)
    print(f"Synthetic env raster written to {output_path}")

make_synthetic_env_raster("env_africa.tif")

# ------------------------------------------------------------------
# Sample presence and background points
# ------------------------------------------------------------------
presence_gdf = gpd.read_file("leopard_thinned.gpkg")

# Sample background points (10x presence count)
with rasterio.open("env_africa.tif") as src:
    bg_points = elapid.sample_raster(src, n=len(presence_gdf) * 10,
                                     ignore_mask=True)

bg_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(bg_points[:, 0], bg_points[:, 1]),
    crs="EPSG:4326"
)

# Annotate presence and background with environmental covariates
presence_annotated = elapid.annotate(presence_gdf, "env_africa.tif",
                                      labels=["bio1", "bio12"])
bg_annotated       = elapid.annotate(bg_gdf, "env_africa.tif",
                                      labels=["bio1", "bio12"])

# Fit MaxEnt model
model = elapid.MaxentModel(
    feature_types=["linear", "quadratic", "hinge"],
    regularization_multiplier=1.5,
    convergence_tolerance=1e-6,
    n_cpus=4,
)
model.fit(presence_annotated, bg_annotated)

# Apply model to raster and save prediction
pred_raster = elapid.apply_model_to_rasters(
    model, ["env_africa.tif"], output_path="leopard_suitability.tif",
    transform="cloglog",
)
print("Suitability raster written to leopard_suitability.tif")
```

---

## Advanced Usage

### Boosted Regression Trees (BRT) with scikit-learn

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import elapid
import rasterio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Assume presence_annotated and bg_annotated from previous step
env_cols = ["bio1", "bio12"]

X_pres = presence_annotated[env_cols].values
X_bg   = bg_annotated[env_cols].values

X = np.vstack([X_pres, X_bg])
y = np.array([1] * len(X_pres) + [0] * len(X_bg))

# Remove rows with NaN
mask = np.isfinite(X).all(axis=1)
X, y = X[mask], y[mask]

brt_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("brt", HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        random_state=42,
    ))
])

scores = cross_val_score(brt_pipeline, X, y, cv=5, scoring="roc_auc",
                          n_jobs=-1)
print(f"BRT 5-fold AUC: {scores.mean():.3f} ± {scores.std():.3f}")

brt_pipeline.fit(X, y)

# Predict on a grid read from raster
with rasterio.open("env_africa.tif") as src:
    band1 = src.read(1).ravel()
    band2 = src.read(2).ravel()
    meta  = src.meta.copy()
    h, w  = src.height, src.width

grid = np.column_stack([band1, band2])
valid_grid = np.isfinite(grid).all(axis=1)
pred_flat = np.full(len(grid), np.nan)
pred_flat[valid_grid] = brt_pipeline.predict_proba(grid[valid_grid])[:, 1]

pred_map = pred_flat.reshape(h, w)

meta.update(count=1, dtype="float32")
with rasterio.open("leopard_brt_suitability.tif", "w", **meta) as dst:
    dst.write(pred_map.astype("float32"), 1)
print("BRT suitability raster written.")
```

### Species Richness Map

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

raster_paths = [
    "leopard_suitability.tif",
    "leopard_brt_suitability.tif",
]
threshold = 0.5

richness_sum = None
meta = None

for path in raster_paths:
    with rasterio.open(path) as src:
        data = src.read(1)
        if meta is None:
            meta = src.meta.copy()
            lons = np.linspace(src.bounds.left, src.bounds.right, src.width)
            lats = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
        presence = (data >= threshold).astype(float)
        presence[data == src.nodata] = np.nan
        richness_sum = presence if richness_sum is None else richness_sum + presence

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()},
                        figsize=(10, 8))
ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.3)
ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--")

pcm = ax.pcolormesh(lons, lats, richness_sum,
                    cmap="YlOrRd", transform=ccrs.PlateCarree(),
                    vmin=0, vmax=len(raster_paths))
plt.colorbar(pcm, ax=ax, label="Modeled Richness (number of models)", shrink=0.8)
ax.set_title("Stacked Species Richness Map – Demo")
plt.tight_layout()
plt.savefig("species_richness_map.png", dpi=150)
plt.show()
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `pygbif` returns 0 results | Taxon key lookup failed or filters too strict | Check `sp.name_backbone()` output; relax `year` or `country` filters |
| GBIF download never completes | Async download stuck | Poll with `occ.download_meta(download_key)` until status is `SUCCEEDED` |
| `elapid.annotate` raises CRS mismatch | Raster and GDF have different CRS | Reproject GDF: `gdf = gdf.to_crs(raster_crs)` |
| MaxEnt AUC < 0.7 | Too few presence points or uninformative predictors | Add more bioclimatic variables; increase background sample size |
| `rasterio` cannot open GeoTIFF | GDAL not installed correctly | Use `conda install -c conda-forge rasterio` |
| `geopandas` spatial join slow on large datasets | Using default brute-force join | Ensure `shapely>=2.0` is installed (uses STRtree automatically) |

---

## External Resources

- GBIF occurrence API: <https://www.gbif.org/developer/occurrence>
- pygbif documentation: <https://pygbif.readthedocs.io/>
- elapid SDM library: <https://earth-chris.github.io/elapid/>
- WorldClim bioclimatic variables: <https://www.worldclim.org/data/bioclim.html>
- CHELSA climate data: <https://chelsa-climate.org/>
- geopandas documentation: <https://geopandas.org/en/stable/>
- Elith et al. (2006) comparative SDM evaluation: <https://doi.org/10.1111/j.1365-2699.2006.01437.x>

---

## Examples

### Example 1 – Range Map of a Migratory Bird

```python
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from pygbif import occurrences as occ
from pygbif import species as sp

def quick_range_map(scientific_name: str, year_min: int = 2018,
                    limit: int = 3000):
    """Download occurrences and plot a simple range map."""
    key = sp.name_backbone(name=scientific_name, rank="species")["usageKey"]
    res = occ.search(taxonKey=key, hasCoordinate=True,
                     year=f"{year_min},2023", limit=limit)
    records = res["results"]
    df = pd.json_normalize(records)[["decimalLatitude", "decimalLongitude"]].dropna()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(r.decimalLongitude, r.decimalLatitude) for r in df.itertuples()],
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()},
                            figsize=(14, 7))
    ax.add_feature(cfeature.LAND, facecolor="#d6d6d6")
    ax.add_feature(cfeature.OCEAN, facecolor="#c8e5f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.gridlines(linewidth=0.2, linestyle="--")
    ax.scatter(
        df["decimalLongitude"], df["decimalLatitude"],
        transform=ccrs.PlateCarree(),
        s=2, color="firebrick", alpha=0.5, label=f"n={len(gdf)}"
    )
    ax.set_title(f"GBIF Occurrences: {scientific_name} ({year_min}–2023)",
                 fontsize=13)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{scientific_name.replace(' ', '_')}_range_map.png", dpi=150)
    plt.show()

quick_range_map("Hirundo rustica")  # Barn swallow
```

### Example 2 – Spatial Thinning and AUC Comparison Across Resolutions

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import elapid
import rasterio
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

resolutions = [0.25, 0.5, 1.0]
leopard_df = pd.read_csv("panthera_pardus_occurrences.csv")

results = []
for res in resolutions:
    from shapely.geometry import Point
    thin = spatial_thin(leopard_df, resolution_deg=res)  # function from Step 2

    pres_ann = elapid.annotate(thin, "env_africa.tif", labels=["bio1", "bio12"])
    pres_ann = pres_ann.dropna(subset=["bio1", "bio12"])

    with rasterio.open("env_africa.tif") as src:
        bg_pts = elapid.sample_raster(src, n=len(pres_ann) * 10)
    bg_gdf  = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(bg_pts[:, 0], bg_pts[:, 1]),
        crs="EPSG:4326",
    )
    bg_ann = elapid.annotate(bg_gdf, "env_africa.tif", labels=["bio1", "bio12"])
    bg_ann = bg_ann.dropna(subset=["bio1", "bio12"])

    X = np.vstack([pres_ann[["bio1", "bio12"]].values,
                   bg_ann[["bio1", "bio12"]].values])
    y = np.array([1]*len(pres_ann) + [0]*len(bg_ann))

    model = elapid.MaxentModel(feature_types=["linear", "quadratic"],
                                regularization_multiplier=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Build temporary GDFs for elapid fit
        pres_mask = y_tr == 1
        bg_mask   = y_tr == 0
        tmp_pres = gpd.GeoDataFrame(
            {"bio1": X_tr[pres_mask, 0], "bio12": X_tr[pres_mask, 1]},
            geometry=gpd.points_from_xy(
                np.zeros(pres_mask.sum()), np.zeros(pres_mask.sum())
            ), crs="EPSG:4326"
        )
        tmp_bg = gpd.GeoDataFrame(
            {"bio1": X_tr[bg_mask, 0], "bio12": X_tr[bg_mask, 1]},
            geometry=gpd.points_from_xy(
                np.zeros(bg_mask.sum()), np.zeros(bg_mask.sum())
            ), crs="EPSG:4326"
        )
        model.fit(tmp_pres, tmp_bg)
        prob = model.predict(X_te)
        aucs.append(roc_auc_score(y_te, prob))

    results.append({"resolution_deg": res, "n_pres": len(pres_ann),
                    "mean_auc": np.mean(aucs), "std_auc": np.std(aucs)})
    print(f"Res={res}° | n={len(pres_ann)} | AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```
