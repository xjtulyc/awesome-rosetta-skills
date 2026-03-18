---
name: historical-gis
description: >
  Use this Skill for historical GIS: georeferencing historical maps with GDAL,
  digitizing boundaries, temporal territory change overlays, and historical population interpolation.
tags:
  - history
  - GIS
  - historical-maps
  - georeferencing
  - spatial-humanities
  - territory
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
    - rasterio>=1.3
    - geopandas>=0.13
    - numpy>=1.23
    - matplotlib>=3.6
    - scipy>=1.9
    - shapely>=2.0
  system:
    - gdal
last_updated: "2026-03-18"
status: stable
---

# Historical GIS: Georeferencing and Spatial Humanities

> **TL;DR** — Georeference historical maps using GDAL, digitize territorial boundaries
> into GeoDataFrames, compare multi-period territories with Hausdorff distance, interpolate
> historical population from census points, and animate territorial change over centuries.

---

## When to Use

Use this Skill when you need to:

- Align a historical map scan to a modern coordinate reference system (CRS)
- Digitize boundaries of historical polities, dioceses, or trade zones
- Overlay territorial extents from multiple centuries in a single plot
- Interpolate population density from sparse historical census data to polygon areas
- Animate territorial change over time for scholarly visualization or publication

Do **not** use this Skill for:

- Modern satellite or aerial image registration (use QGIS auto-registration tools)
- Real-time geospatial databases (use PostGIS instead)
- Vector map editing at scale (use QGIS or ArcGIS Pro workflows)

---

## Background

Georeferencing assigns real-world coordinates to a raster image (map scan) using
**Ground Control Points (GCPs)** — identifiable features on the historical map matched
to known modern coordinates (church spires, river confluences, coast outlines).

| Concept | Explanation |
|---|---|
| GCP (Ground Control Point) | Pixel (col, row) → geographic (lon, lat) correspondence |
| Polynomial transformation | 1st order (affine, 6 params), 2nd order (quadratic, 12), 3rd order (20) |
| RMS error | Root mean square residual of GCP reprojection; <2 px target for 1st order |
| EPSG:4326 | WGS84 geographic CRS; EPSG:3857 = Web Mercator for display |
| Hausdorff distance | Max of directed distances between two boundary curves |
| Voronoi / Thiessen | Partition space so each polygon contains all points nearer to its site |

GDAL `gdal_translate` embeds GCPs into the raster metadata; `gdalwarp` then resamples
to produce a properly georeferenced GeoTIFF that can be loaded in any GIS tool.

---

## Environment Setup

```bash
# Install GDAL system library (Ubuntu/Debian)
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal

# Verify
gdalinfo --version

# Create Python environment
conda create -n hist-gis python=3.11 -y
conda activate hist-gis

# Install Python packages
pip install "rasterio>=1.3" "geopandas>=0.13" "numpy>=1.23" \
    "matplotlib>=3.6" "scipy>=1.9" "shapely>=2.0"

# Verify rasterio + GDAL linkage
python -c "import rasterio; print(rasterio.__gdal_version__)"
```

On macOS with Homebrew:

```bash
brew install gdal
pip install "rasterio>=1.3" "geopandas>=0.13" "numpy>=1.23" \
    "matplotlib>=3.6" "scipy>=1.9" "shapely>=2.0"
```

---

## Core Workflow

### Step 1 — GDAL Georeferencing with GCP File

```python
import subprocess
import os
from pathlib import Path


def write_gcp_file(gcps: list[dict], gcp_path: str) -> None:
    """
    Write a GCP text file compatible with GDAL gdal_translate -gcp flag.

    Each GCP dict must contain:
        pixel_x (float): column in the source image
        pixel_y (float): row in the source image
        geo_x  (float): target longitude (EPSG:4326)
        geo_y  (float): target latitude  (EPSG:4326)
        label  (str):   human-readable identifier

    Args:
        gcps:     List of GCP dicts.
        gcp_path: Output path for the generated shell-script gcp file.
    """
    lines = ["#!/usr/bin/env bash", "# Auto-generated GCPs for gdal_translate", ""]
    for g in gcps:
        lines.append(
            f"-gcp {g['pixel_x']} {g['pixel_y']} {g['geo_x']} {g['geo_y']}"
            f"  # {g.get('label', '')}"
        )
    Path(gcp_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"GCP file written to {gcp_path}")


def georeference_map(
    input_raster: str,
    output_raster: str,
    gcps: list[dict],
    order: int = 1,
    target_crs: str = "EPSG:4326",
    resampling: str = "bilinear",
) -> dict:
    """
    Georeference a historical map scan using GDAL GCPs + polynomial warp.

    Steps:
      1. gdal_translate: embed GCPs into intermediate GeoTIFF.
      2. gdalwarp:       polynomial transform and resample to target CRS.

    Args:
        input_raster:  Path to the raw scan (JPG, PNG, TIFF).
        output_raster: Path to write the georeferenced GeoTIFF.
        gcps:          List of GCP dicts (pixel_x, pixel_y, geo_x, geo_y, label).
        order:         Polynomial order (1=affine, 2=quadratic, 3=cubic).
        target_crs:    EPSG code string for the output CRS.
        resampling:    Resampling algorithm: "bilinear" | "cubic" | "near".

    Returns:
        Dict with intermediate_path, output_path, rms_estimate.
    """
    intermediate = input_raster.replace(".", "_gcps.")
    if not intermediate.endswith(".tif"):
        intermediate += ".tif"

    # Build -gcp arguments
    gcp_args = []
    for g in gcps:
        gcp_args.extend([
            "-gcp",
            str(g["pixel_x"]), str(g["pixel_y"]),
            str(g["geo_x"]), str(g["geo_y"]),
        ])

    # Step 1: embed GCPs
    cmd_translate = (
        ["gdal_translate", "-of", "GTiff"]
        + gcp_args
        + [input_raster, intermediate]
    )
    result = subprocess.run(cmd_translate, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdal_translate failed:\n{result.stderr}")

    # Step 2: warp to target CRS
    cmd_warp = [
        "gdalwarp",
        "-order", str(order),
        "-r", resampling,
        "-t_srs", target_crs,
        "-overwrite",
        intermediate, output_raster,
    ]
    result = subprocess.run(cmd_warp, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed:\n{result.stderr}")

    # Rough RMS estimate from residuals (manual if needed)
    print(f"Georeferenced raster written to {output_raster}")
    return {
        "intermediate_path": intermediate,
        "output_path": output_raster,
        "gcp_count": len(gcps),
        "order": order,
        "target_crs": target_crs,
    }
```

### Step 2 — Multi-Period Territory Overlay Plot

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
import numpy as np


def create_historical_territory(
    name: str,
    coords: list[tuple[float, float]],
    period: str,
    color: str,
) -> dict:
    """
    Construct a historical territory record from a polygon coordinate list.

    Args:
        name:   Territory name (e.g. "Holy Roman Empire").
        coords: List of (lon, lat) tuples defining the boundary.
        period: Century or date label (e.g. "1250 CE").
        color:  Matplotlib color string for this territory.

    Returns:
        Dict with keys: name, period, geometry, color.
    """
    poly = Polygon(coords)
    return {"name": name, "period": period, "geometry": poly, "color": color}


def plot_temporal_territories(
    territories: list[dict],
    title: str = "Historical Territory Change",
    output_path: str = None,
    alpha: float = 0.35,
) -> None:
    """
    Overlay historical territories from multiple periods on a single map.

    Args:
        territories: List of territory dicts from create_historical_territory().
        title:       Plot title string.
        output_path: If given, save figure here.
        alpha:       Polygon fill transparency.
    """
    gdf = gpd.GeoDataFrame(territories, crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(12, 8))
    periods = gdf["period"].unique()

    legend_patches = []
    for territory in territories:
        geom = territory["geometry"]
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=alpha, color=territory["color"], linewidth=1.5,
                edgecolor=territory["color"])

    # Build legend: one entry per territory name+period
    for _, row in gdf.iterrows():
        patch = mpatches.Patch(
            color=row["color"], alpha=0.7,
            label=f"{row['name']} ({row['period']})"
        )
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc="lower left", fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Territory map saved to {output_path}")
    plt.show()


def hausdorff_distance_territories(
    geom_a: Polygon,
    geom_b: Polygon,
) -> float:
    """
    Compute the Hausdorff distance between two historical boundary polygons.

    The Hausdorff distance is the maximum distance from any point on boundary A
    to the nearest point on boundary B (symmetric max). Values are in CRS units
    (degrees for EPSG:4326; project to a metric CRS for meaningful km values).

    Args:
        geom_a: Shapely Polygon for period A boundary.
        geom_b: Shapely Polygon for period B boundary.

    Returns:
        Hausdorff distance as float in CRS units.
    """
    return geom_a.hausdorff_distance(geom_b)
```

### Step 3 — Historical Population Interpolation Between Census Years

```python
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_bounds


def interpolate_historical_population(
    census_points: list[dict],
    bbox: tuple[float, float, float, float],
    output_raster: str,
    resolution: float = 0.1,
    method: str = "linear",
) -> np.ndarray:
    """
    Interpolate historical census point data to a population density raster.

    Args:
        census_points: List of dicts with keys: lon, lat, population, year.
        bbox:          Bounding box (min_lon, min_lat, max_lon, max_lat).
        output_raster: Path to write the output GeoTIFF.
        resolution:    Grid cell size in CRS units (degrees for WGS84).
        method:        Scipy interpolation method: "linear" | "cubic" | "nearest".

    Returns:
        Interpolated grid as 2D numpy array.
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    lon_vals = np.array([p["lon"] for p in census_points])
    lat_vals = np.array([p["lat"] for p in census_points])
    pop_vals = np.array([p["population"] for p in census_points], dtype=float)

    # Normalise population to density per sq degree
    grid_lon = np.arange(min_lon, max_lon, resolution)
    grid_lat = np.arange(min_lat, max_lat, resolution)
    grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

    grid_pop = griddata(
        points=np.column_stack([lon_vals, lat_vals]),
        values=pop_vals,
        xi=(grid_x, grid_y),
        method=method,
        fill_value=0.0,
    )
    grid_pop = np.nan_to_num(grid_pop, nan=0.0)

    # Write as single-band GeoTIFF
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat,
                            grid_pop.shape[1], grid_pop.shape[0])
    with rasterio.open(
        output_raster, "w",
        driver="GTiff",
        height=grid_pop.shape[0],
        width=grid_pop.shape[1],
        count=1,
        dtype=grid_pop.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(grid_pop, 1)

    print(f"Population raster saved to {output_raster}")
    return grid_pop
```

---

## Advanced Usage

### Territory Change Animation

```python
import matplotlib.animation as animation


def animate_territory_change(
    territory_frames: list[list[dict]],
    period_labels: list[str],
    output_gif: str,
    fps: int = 2,
) -> None:
    """
    Create an animated GIF showing territory change across periods.

    Args:
        territory_frames: List of territory lists, one per time step.
        period_labels:    Labels for each time step (same length as territory_frames).
        output_gif:       Path to write the output GIF.
        fps:              Frames per second.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    def update(frame_idx: int):
        ax.clear()
        territories = territory_frames[frame_idx]
        for t in territories:
            geom = t["geometry"]
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.4, color=t["color"], edgecolor=t["color"], linewidth=1.5)
        ax.set_title(f"Territory — {period_labels[frame_idx]}", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)

    ani = animation.FuncAnimation(
        fig, update, frames=len(territory_frames),
        interval=1000 // fps, repeat=True,
    )
    ani.save(output_gif, writer="pillow", fps=fps)
    print(f"Animation saved to {output_gif}")
    plt.close(fig)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `gdalwarp: command not found` | GDAL not on PATH | `export PATH=/usr/bin:$PATH`; verify with `which gdalwarp` |
| High RMS error (>5 px) after 1st order | Non-linear map projection or distortion | Use order=2 or add more GCPs near edges |
| `CRSError` in rasterio | CRS string not recognised | Use EPSG integer: `crs=4326`; verify PROJ data dir |
| Shapely `TopologicalError` on Polygon | Self-intersecting coordinates | Apply `.buffer(0)` to fix topology |
| Population grid shows NaN islands | Insufficient census points for interpolation area | Switch to `method="nearest"` or add synthetic boundary points |
| GDAL intermediate file already exists | Previous run crashed | Delete the `*_gcps.tif` file and retry |

---

## External Resources

- GDAL documentation: <https://gdal.org/programs/gdal_translate.html>
- GeoDataFrame and geopandas: <https://geopandas.org/en/stable/docs.html>
- Rasterio documentation: <https://rasterio.readthedocs.io/>
- Historical Boundaries Datasets (NHGIS, HGIS): <https://www.nhgis.org/>
- World Historical Gazetteer: <https://whgazetteer.org/>
- Digital Atlas of Roman and Medieval Civilizations: <https://darmc.harvard.edu/>

---

## Examples

### Example 1 — Georeference an 18th-Century Town Plan

```python
# Define GCPs: pixel coordinates on the scan matched to known modern GPS coordinates
gcps = [
    {"pixel_x": 245, "pixel_y": 180, "geo_x": 13.405, "geo_y": 52.520, "label": "Brandenburg Gate area"},
    {"pixel_x": 890, "pixel_y": 210, "geo_x": 13.450, "geo_y": 52.519, "label": "Alexanderplatz area"},
    {"pixel_x": 250, "pixel_y": 760, "geo_x": 13.406, "geo_y": 52.490, "label": "Tempelhof area"},
    {"pixel_x": 920, "pixel_y": 790, "geo_x": 13.452, "geo_y": 52.489, "label": "Treptow area"},
]

result = georeference_map(
    input_raster="/data/maps/berlin_1780.jpg",
    output_raster="/data/maps/berlin_1780_georef.tif",
    gcps=gcps,
    order=1,
    target_crs="EPSG:4326",
)
print(f"Georeferenced {result['gcp_count']} GCPs, order={result['order']}")
```

### Example 2 — Multi-Century Territory Overlay

```python
# Define approximate polygon coordinates for illustrative territories
territories = [
    create_historical_territory(
        name="Duchy of Bavaria",
        coords=[(10.0, 47.3), (13.5, 47.3), (13.5, 49.2), (10.0, 49.2), (10.0, 47.3)],
        period="1200 CE",
        color="#4C72B0",
    ),
    create_historical_territory(
        name="Kingdom of Bavaria",
        coords=[(9.5, 47.2), (13.8, 47.2), (13.8, 50.5), (9.5, 50.5), (9.5, 47.2)],
        period="1810 CE",
        color="#DD8452",
    ),
]

plot_temporal_territories(
    territories=territories,
    title="Territorial Expansion of Bavaria 1200–1810",
    output_path="/data/output/bavaria_territories.png",
)

dist = hausdorff_distance_territories(
    territories[0]["geometry"],
    territories[1]["geometry"],
)
print(f"Hausdorff distance between 1200 and 1810 boundaries: {dist:.3f} degrees")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — GDAL georeferencing, territory overlay, Hausdorff distance, population interpolation, animation |
