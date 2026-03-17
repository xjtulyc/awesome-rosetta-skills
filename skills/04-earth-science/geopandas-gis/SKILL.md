---
name: geopandas-gis
description: >
  Vector GIS analysis with GeoPandas: spatial joins, overlays, buffer/dissolve,
  choropleth maps, and rasterio integration for raster-vector workflows.
tags:
  - geopandas
  - gis
  - spatial-analysis
  - cartography
  - rasterio
  - shapefile
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
  - geopandas>=0.14.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - shapely>=2.0.0
  - pyproj>=3.6.0
  - matplotlib>=3.7.0
  - rasterio>=1.3.8
  - rasterstats>=0.19.0
  - requests>=2.31.0
  - fiona>=1.9.4
last_updated: "2026-03-17"
---

# GeoPandas Vector GIS Skill

This skill covers the complete vector GIS workflow in Python using GeoPandas: reading and
writing spatial files, coordinate reference system (CRS) management, spatial joins, overlay
operations, geometry processing, spatial statistics, and choropleth mapping. It also covers
the integration point between vector data (GeoPandas) and raster data (rasterio/rasterstats).

GeoPandas extends pandas DataFrames with a geometry column powered by Shapely, making it
straightforward to apply familiar tabular operations alongside spatial predicates.

---

## Setup

```bash
# Recommended: install via conda for reliable binary dependencies
conda create -n gis python=3.11
conda activate gis
conda install -c conda-forge geopandas rasterio rasterstats matplotlib requests fiona

# Or with pip (may require system GDAL/GEOS on Linux/macOS):
pip install geopandas rasterio rasterstats matplotlib requests fiona
```

---

## Core Functions

```python
"""
geopandas_gis.py
----------------
Core GIS utilities built on GeoPandas, Shapely, and rasterio.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import box, Point, MultiPolygon
from pyproj import CRS

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# 1. Load / Save Spatial Data
# ---------------------------------------------------------------------------

def load_shapefile(
    path: Union[str, Path],
    crs_target: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """
    Read a vector file (shapefile, GeoJSON, GPKG, etc.) into a GeoDataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the file. Supports any format recognised by Fiona/GDAL
        (`.shp`, `.geojson`, `.gpkg`, `.gdb`, etc.).
    crs_target : str, optional
        EPSG code or PROJ string to reproject to, e.g. 'EPSG:3857'.
        If None, keeps the source CRS.
    bbox : tuple, optional
        Spatial filter (minx, miny, maxx, maxy) in the source CRS.

    Returns
    -------
    gpd.GeoDataFrame

    Examples
    --------
    >>> countries = load_shapefile("ne_10m_admin_0_countries.shp", crs_target="EPSG:4326")
    >>> europe    = load_shapefile("ne_10m_admin_0_countries.shp",
    ...                            crs_target="EPSG:3035",
    ...                            bbox=(-25, 34, 45, 72))
    """
    gdf = gpd.read_file(str(path), bbox=bbox)
    if crs_target is not None:
        gdf = gdf.to_crs(crs_target)
    return gdf


def save_geodataframe(
    gdf: gpd.GeoDataFrame,
    output_path: Union[str, Path],
    driver: str = "GeoJSON",
) -> None:
    """
    Write a GeoDataFrame to disk.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    output_path : str or Path
    driver : str
        Fiona driver name: 'GeoJSON', 'ESRI Shapefile', 'GPKG', etc.
    """
    gdf.to_file(str(output_path), driver=driver)
    print(f"Saved {len(gdf)} features to {output_path}")


# ---------------------------------------------------------------------------
# 2. Spatial Join
# ---------------------------------------------------------------------------

def spatial_join_nearest(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    max_distance: Optional[float] = None,
) -> gpd.GeoDataFrame:
    """
    Join each feature in gdf1 to its nearest neighbour in gdf2.

    Both GeoDataFrames must share the same (projected) CRS. Distances are
    reported in the units of that CRS (metres for metric projections).

    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        Left GeoDataFrame (e.g. census tracts to enrich).
    gdf2 : gpd.GeoDataFrame
        Right GeoDataFrame to join from (e.g. river segments).
    max_distance : float, optional
        If provided, only joins where the nearest feature is within this
        distance. Features beyond max_distance get NaN attribute values.

    Returns
    -------
    gpd.GeoDataFrame
        gdf1 augmented with columns from gdf2 and a 'distance' column.
    """
    if gdf1.crs != gdf2.crs:
        raise ValueError(
            f"CRS mismatch: gdf1={gdf1.crs}, gdf2={gdf2.crs}. "
            "Reproject to a common CRS before joining."
        )
    joined = gpd.sjoin_nearest(
        gdf1, gdf2,
        how="left",
        distance_col="distance",
        max_distance=max_distance,
    )
    return joined


def spatial_join_within(
    gdf_points: gpd.GeoDataFrame,
    gdf_polygons: gpd.GeoDataFrame,
    how: str = "left",
) -> gpd.GeoDataFrame:
    """
    Tag each point (or polygon centroid) with the polygon it falls within.

    Parameters
    ----------
    gdf_points : gpd.GeoDataFrame
        Point features to tag.
    gdf_polygons : gpd.GeoDataFrame
        Polygon features containing spatial attributes to transfer.
    how : str
        pandas-style join type: 'left', 'inner', 'right'.

    Returns
    -------
    gpd.GeoDataFrame
    """
    return gpd.sjoin(gdf_points, gdf_polygons, how=how, predicate="within")


# ---------------------------------------------------------------------------
# 3. Geometry Operations
# ---------------------------------------------------------------------------

def clip_to_bbox(
    gdf: gpd.GeoDataFrame,
    bbox: Tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to an axis-aligned bounding box.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    bbox : tuple
        (minx, miny, maxx, maxy) in the same CRS as gdf.

    Returns
    -------
    gpd.GeoDataFrame with geometries clipped (not just filtered).
    """
    clip_geom = box(*bbox)
    return gpd.clip(gdf, clip_geom)


def buffer_and_dissolve(
    gdf: gpd.GeoDataFrame,
    distance: float,
    dissolve_by: Optional[str] = None,
    cap_style: int = 1,
) -> gpd.GeoDataFrame:
    """
    Buffer features and optionally dissolve overlapping buffers.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must be in a projected (metric) CRS for meaningful distance buffers.
    distance : float
        Buffer distance in CRS units (metres for UTM/EPSG:3857).
    dissolve_by : str, optional
        Column to group/dissolve by. Pass None to dissolve all into one.
    cap_style : int
        Shapely cap style: 1=round, 2=flat, 3=square.

    Returns
    -------
    gpd.GeoDataFrame
    """
    buffered = gdf.copy()
    buffered["geometry"] = gdf.geometry.buffer(distance, cap_style=cap_style)

    if dissolve_by is not None:
        buffered = buffered.dissolve(by=dissolve_by, as_index=False)
    else:
        buffered = buffered.dissolve().reset_index(drop=True)

    return buffered


def overlay_intersection(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Return the geometric intersection of two GeoDataFrames (polygon-on-polygon).
    Attribute columns from both inputs are retained.
    """
    return gpd.overlay(gdf1, gdf2, how="intersection", keep_geom_type=True)


# ---------------------------------------------------------------------------
# 4. Spatial Statistics
# ---------------------------------------------------------------------------

def compute_spatial_stats(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute summary statistics (area, perimeter, centroid, value statistics)
    for each feature or group of features.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Must be in a projected CRS to get meaningful area/length values.
    value_col : str
        Numeric attribute column to summarise.
    group_col : str, optional
        Column to group by before aggregating.

    Returns
    -------
    pd.DataFrame with columns: count, area_km2, length_km, mean, std, min, max.
    """
    gdf = gdf.copy()
    gdf["_area_km2"] = gdf.geometry.area / 1e6          # m² -> km²
    gdf["_length_km"] = gdf.geometry.length / 1000      # m  -> km
    gdf["_cx"] = gdf.geometry.centroid.x
    gdf["_cy"] = gdf.geometry.centroid.y

    agg_cols = {
        "_area_km2": "sum",
        "_length_km": "sum",
        value_col: ["count", "mean", "std", "min", "max"],
    }

    if group_col:
        stats = gdf.groupby(group_col).agg(agg_cols)
        stats.columns = ["area_km2", "length_km", "count", "mean", "std", "min", "max"]
    else:
        stats = pd.DataFrame({
            "area_km2": [gdf["_area_km2"].sum()],
            "length_km": [gdf["_length_km"].sum()],
            "count": [len(gdf)],
            "mean": [gdf[value_col].mean()],
            "std": [gdf[value_col].std()],
            "min": [gdf[value_col].min()],
            "max": [gdf[value_col].max()],
        })

    return stats.reset_index()


# ---------------------------------------------------------------------------
# 5. Choropleth Mapping
# ---------------------------------------------------------------------------

def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    col: str,
    title: str = "",
    cmap: str = "YlOrRd",
    figsize: Tuple[int, int] = (12, 8),
    legend: bool = True,
    edgecolor: str = "white",
    linewidth: float = 0.4,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a choropleth map of a numeric column.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
    col : str
        Column to map.
    title : str
        Figure title.
    cmap : str
        Matplotlib colormap.
    figsize : tuple
    legend : bool
        Whether to include a colorbar legend.
    edgecolor : str
        Polygon border color.
    linewidth : float
        Polygon border width.
    output_path : str, optional
        If given, save the figure here.

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(
        column=col,
        ax=ax,
        cmap=cmap,
        legend=legend,
        legend_kwds={"label": col, "orientation": "horizontal", "pad": 0.04},
        edgecolor=edgecolor,
        linewidth=linewidth,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_axis_off()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig
```

---

## Example 1: Land-Use Area Statistics for Administrative Regions

Download Natural Earth administrative boundaries and a land-use GeoJSON, then compute
area statistics for each land-use class within each admin region.

```python
"""
example_landuse_stats.py
------------------------
Calculate land-use area statistics per administrative region using
spatial overlay and geopandas aggregation.
"""

import requests
import zipfile
import io
from pathlib import Path

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from geopandas_gis import (
    load_shapefile,
    overlay_intersection,
    compute_spatial_stats,
    plot_choropleth,
)

DATA_DIR = Path("gis_data")
DATA_DIR.mkdir(exist_ok=True)


def download_natural_earth(scale: str = "10m", category: str = "cultural",
                            name: str = "admin_1_states_provinces") -> Path:
    """Download a Natural Earth shapefile and return the .shp path."""
    url = (f"https://naciscdn.org/naturalearth/{scale}/{category}/"
           f"ne_{scale}_{name}.zip")
    out_zip = DATA_DIR / f"ne_{scale}_{name}.zip"
    if not out_zip.exists():
        print(f"Downloading {url} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out_zip.write_bytes(r.content)
    # Extract
    out_dir = DATA_DIR / f"ne_{scale}_{name}"
    if not out_dir.exists():
        with zipfile.ZipFile(out_zip) as zf:
            zf.extractall(out_dir)
    shp = next(out_dir.glob("*.shp"))
    return shp


# ---- 1. Load administrative boundaries (example: Germany federal states) ----
states_shp = download_natural_earth("10m", "cultural", "admin_1_states_provinces")
states = load_shapefile(states_shp, crs_target="EPSG:3035")
germany = states[states["admin"] == "Germany"].copy()
print(f"Loaded {len(germany)} German federal states")

# ---- 2. Create synthetic land-use data for demonstration ----
# In a real workflow, replace this with an actual land-use shapefile
# (e.g. CORINE Land Cover from https://land.copernicus.eu)
import numpy as np
from shapely.geometry import Polygon

rng = np.random.default_rng(42)
land_use_types = ["Forest", "Agriculture", "Urban", "Water", "Shrubland"]

# Generate random polygons inside Germany's bounding box
minx, miny, maxx, maxy = germany.total_bounds
synthetic_polygons = []
for _ in range(500):
    cx = rng.uniform(minx, maxx)
    cy = rng.uniform(miny, maxy)
    size = rng.uniform(5000, 50000)           # 5–50 km²
    poly = Polygon([
        (cx - size, cy - size), (cx + size, cy - size),
        (cx + size, cy + size), (cx - size, cy + size),
    ])
    lu_type = rng.choice(land_use_types)
    synthetic_polygons.append({"land_use": lu_type, "geometry": poly})

land_use = gpd.GeoDataFrame(synthetic_polygons, crs="EPSG:3035")
land_use = gpd.clip(land_use, germany.union_all())

# ---- 3. Overlay: intersect land-use with state boundaries ----
lu_by_state = overlay_intersection(land_use, germany[["NAME_1", "geometry"]])
lu_by_state["area_km2"] = lu_by_state.geometry.area / 1e6

# ---- 4. Summarise: total area by state and land-use type ----
summary = (
    lu_by_state.groupby(["NAME_1", "land_use"])["area_km2"]
    .sum()
    .unstack(fill_value=0)
    .round(1)
)
print("\nLand-use area (km²) by federal state:")
print(summary.to_string())

# ---- 5. Compute forest fraction per state ----
summary["forest_fraction"] = (
    summary.get("Forest", 0) / summary.sum(axis=1) * 100
)
germany = germany.merge(
    summary[["forest_fraction"]].reset_index(),
    left_on="NAME_1", right_on="NAME_1", how="left",
)

# ---- 6. Choropleth of forest fraction ----
fig = plot_choropleth(
    germany.to_crs("EPSG:4326"),
    col="forest_fraction",
    title="Estimated Forest Coverage by German Federal State (%)",
    cmap="Greens",
    output_path="germany_forest_fraction.png",
)
plt.show()

# ---- 7. Bar chart: top 5 states by urban area ----
if "Urban" in summary.columns:
    top_urban = summary["Urban"].sort_values(ascending=False).head(5)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    top_urban.plot.barh(ax=ax2, color="steelblue")
    ax2.set_xlabel("Urban area (km²)")
    ax2.set_title("Top 5 German States by Urban Area")
    fig2.tight_layout()
    fig2.savefig("germany_urban_top5.png", dpi=150)
    print("Saved: germany_urban_top5.png")
```

---

## Example 2: Census Tracts Within 5 km of a River Network

Find all census tracts whose centroid falls within a 5 km buffer of a river network,
then compute population statistics for the flood-risk zone.

```python
"""
example_river_buffer.py
-----------------------
Identify census tracts within 5 km of rivers and compute population at risk.
Uses synthetic data; swap in real shapefiles for production use.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from geopandas_gis import (
    buffer_and_dissolve,
    spatial_join_within,
    compute_spatial_stats,
    plot_choropleth,
)

rng = np.random.default_rng(0)

# ---- 1. Create synthetic river network (LineStrings) ----
# Replace with: gpd.read_file("rivers.shp").to_crs("EPSG:32632")
def make_river(start, n_segments=20, spread=5000):
    coords = [start]
    for _ in range(n_segments):
        last = coords[-1]
        coords.append((last[0] + rng.uniform(1000, 8000),
                       last[1] + rng.uniform(-spread, spread)))
    return LineString(coords)

rivers_data = [
    {"river_name": "Main River",  "geometry": make_river((400000, 5540000))},
    {"river_name": "Rhine",       "geometry": make_river((340000, 5600000), spread=8000)},
    {"river_name": "Tributary A", "geometry": make_river((380000, 5560000), n_segments=10)},
]
rivers = gpd.GeoDataFrame(rivers_data, crs="EPSG:32632")
print(f"River network: {len(rivers)} segments, "
      f"total length {rivers.geometry.length.sum()/1000:.1f} km")

# ---- 2. Create 5 km buffer around all rivers, dissolved ----
river_buffer = buffer_and_dissolve(rivers, distance=5000)
river_buffer["zone"] = "5km_buffer"
print(f"Buffer area: {river_buffer.geometry.area.sum()/1e6:.1f} km²")

# ---- 3. Create synthetic census tracts with population ----
minx, miny, maxx, maxy = rivers.total_bounds
n_tracts = 300
tract_data = []
for i in range(n_tracts):
    cx = rng.uniform(minx - 10000, maxx + 10000)
    cy = rng.uniform(miny - 10000, maxy + 10000)
    w  = rng.uniform(1500, 4000)
    from shapely.geometry import box as sbox
    poly = sbox(cx - w, cy - w, cx + w, cy + w)
    pop  = int(rng.lognormal(8, 0.8))   # log-normal population
    tract_data.append({"tract_id": i, "population": pop, "geometry": poly})

tracts = gpd.GeoDataFrame(tract_data, crs="EPSG:32632")
tracts["centroid_geom"] = tracts.geometry.centroid

# ---- 4. Find tracts whose centroid is inside the buffer ----
tract_centroids = tracts.copy()
tract_centroids["geometry"] = tracts["centroid_geom"]

in_buffer = spatial_join_within(tract_centroids, river_buffer[["zone", "geometry"]])
at_risk_ids = set(in_buffer["tract_id"].unique())
tracts["at_risk"] = tracts["tract_id"].isin(at_risk_ids)

print(f"\nTracts at risk (within 5 km): {tracts['at_risk'].sum()} / {len(tracts)}")
print(f"Population at risk: {tracts.loc[tracts['at_risk'], 'population'].sum():,}")
print(f"Total population:   {tracts['population'].sum():,}")
pct = tracts.loc[tracts['at_risk'], 'population'].sum() / tracts['population'].sum() * 100
print(f"Fraction at risk:   {pct:.1f}%")

# ---- 5. Map ----
fig, ax = plt.subplots(figsize=(12, 10))

# Background tracts
tracts.plot(ax=ax, column="at_risk",
            cmap="RdYlGn_r", alpha=0.6,
            edgecolor="grey", linewidth=0.3)

# River buffer overlay
river_buffer.plot(ax=ax, color="steelblue", alpha=0.25, label="5 km buffer")

# River lines
rivers.plot(ax=ax, color="blue", linewidth=1.5, label="Rivers")

# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="firebrick", alpha=0.7, label="At-risk tracts"),
    Patch(facecolor="green",     alpha=0.7, label="Safe tracts"),
    Patch(facecolor="steelblue", alpha=0.4, label="5 km river buffer"),
]
ax.legend(handles=legend_elements, loc="upper left")
ax.set_title("Census Tracts Within 5 km of River Network\n"
             f"Population at risk: {pct:.1f}%", fontsize=13)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("river_flood_risk.png", dpi=150, bbox_inches="tight")
print("\nSaved: river_flood_risk.png")
plt.show()

# ---- 6. Spatial statistics for at-risk vs safe tracts ----
stats_at_risk = compute_spatial_stats(
    tracts[tracts["at_risk"]].to_crs("EPSG:32632"),
    value_col="population",
)
stats_safe = compute_spatial_stats(
    tracts[~tracts["at_risk"]].to_crs("EPSG:32632"),
    value_col="population",
)

print("\nSpatial statistics — at-risk tracts:")
print(stats_at_risk.to_string(index=False))
print("\nSpatial statistics — safe tracts:")
print(stats_safe.to_string(index=False))
```

---

## Raster-Vector Integration with rasterio

```python
"""
raster_vector.py
----------------
Extract raster statistics (mean elevation, median NDVI, etc.) for each
polygon in a GeoDataFrame using rasterstats.
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterstats import zonal_stats
from pathlib import Path


def zonal_statistics(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    stats: list = None,
    all_touched: bool = False,
) -> gpd.GeoDataFrame:
    """
    Compute zonal statistics from a raster for each polygon.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Polygon GeoDataFrame. Will be reprojected to match the raster CRS.
    raster_path : str
        Path to a GeoTIFF or other GDAL-readable raster.
    stats : list, optional
        Statistics to compute. Defaults to ['mean', 'std', 'min', 'max', 'count'].
    all_touched : bool
        If True, include all cells touching a polygon (not just cell centres inside).

    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with new columns for each requested statistic.
    """
    if stats is None:
        stats = ["mean", "std", "min", "max", "count"]

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_epsg()

    gdf_proj = gdf.to_crs(f"EPSG:{raster_crs}")

    results = zonal_stats(
        gdf_proj,
        raster_path,
        stats=stats,
        all_touched=all_touched,
        geojson_out=False,
    )

    stats_df = gpd.GeoDataFrame(results, index=gdf.index)
    return gdf.join(stats_df)


def clip_raster_to_polygon(
    raster_path: str,
    gdf: gpd.GeoDataFrame,
    output_path: str,
) -> None:
    """
    Clip a raster to the union of all polygons in a GeoDataFrame.

    Parameters
    ----------
    raster_path : str
    gdf : gpd.GeoDataFrame
    output_path : str
        Path for the output clipped GeoTIFF.
    """
    with rasterio.open(raster_path) as src:
        gdf_proj = gdf.to_crs(src.crs)
        shapes = [geom.__geo_interface__ for geom in gdf_proj.geometry]
        out_image, out_transform = rio_mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"Clipped raster saved: {output_path}")
```

---

## CRS Quick Reference

| EPSG Code | Description                          | Use case                          |
|-----------|--------------------------------------|-----------------------------------|
| 4326      | WGS 84 geographic (lat/lon degrees)  | Default GPS, web display          |
| 3857      | Web Mercator (metres)                | Tile-based web maps               |
| 3035      | ETRS89 / LAEA Europe (metres)        | Area calculations in Europe       |
| 32632     | UTM zone 32N (metres)                | Local metric in Central Europe    |
| 4269      | NAD83 geographic                     | North America                     |
| 5070      | Albers Equal Area CONUS (metres)     | Area statistics for US lower 48   |

Always use a projected (metric) CRS before computing area, length, or buffer distance.

---

## Tips and Best Practices

- **Check CRS early**: call `gdf.crs` immediately after loading. A `None` CRS means no
  projection information was embedded; you must set it manually with `gdf.set_crs(...)`.
- **`sjoin` vs `sjoin_nearest`**: use `sjoin` (with predicates `within`, `intersects`,
  `contains`) when an exact spatial relationship is required; use `sjoin_nearest` when
  you need the closest feature regardless of whether it intersects.
- **Large files**: use `bbox` parameter in `gpd.read_file` to read only the spatial subset
  you need, avoiding loading millions of features into memory.
- **GeoPandas 1.0+**: `unary_union` is deprecated in favour of `union_all()`.
- **Shapely 2.0**: geometry operations are significantly faster. Ensure `shapely>=2.0`.
- **Avoid `apply` with geometry**: prefer vectorised methods (`gdf.geometry.buffer`,
  `gdf.geometry.area`) over row-wise `apply(lambda row: ...)` for performance.

---

## References

- GeoPandas documentation: https://geopandas.org
- Natural Earth data: https://www.naturalearthdata.com
- CORINE Land Cover: https://land.copernicus.eu/pan-european/corine-land-cover
- Shapely manual: https://shapely.readthedocs.io
- rasterio documentation: https://rasterio.readthedocs.io
