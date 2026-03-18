---
name: archaeological-gis
description: >
  Use this Skill for archaeological GIS: site catchment analysis, viewshed
  computation, kernel density estimation, and spatial statistics with GeoPandas.
tags:
  - archaeology
  - gis
  - spatial-analysis
  - geopandas
  - site-analysis
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
    - geopandas>=0.14
    - shapely>=2.0
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
    - rasterio>=1.3
last_updated: "2026-03-17"
status: "stable"
---

# Archaeological GIS Analysis

> **One-line summary**: Analyze archaeological site distributions with GeoPandas: kernel density estimation, nearest neighbor analysis, catchment areas, Thiessen polygons, and predictive site modeling.

---

## When to Use This Skill

- When mapping and analyzing spatial distributions of archaeological sites
- When computing site catchment analysis (resource accessibility areas)
- When detecting spatial clustering patterns (K-function, nearest neighbor)
- When building predictive site location models from environmental variables
- When creating Thiessen (Voronoi) polygons for territory analysis
- When overlaying sites with DEM, soil, and land cover data

**Trigger keywords**: archaeological GIS, site distribution, catchment analysis, kernel density, site prediction, Thiessen polygon, Voronoi, nearest neighbor analysis, K-function, predictive modeling, spatial archaeology, site location model, viewshed, survey data

---

## Background & Key Concepts

### Site Catchment Analysis

Resource territory of a site defined by walking time or buffer radius. Typical thresholds: 1-hour walk (~5 km for flat terrain), 2-hour walk (~10 km).

### Nearest Neighbor Analysis

Average nearest neighbor distance vs. expected random distance:

$$
R = \frac{\bar{d}_{observed}}{\bar{d}_{expected}} = \frac{\bar{d}_{obs}}{0.5/\sqrt{n/A}}
$$

$R < 1$: clustered; $R = 1$: random; $R > 1$: dispersed.

### Kernel Density Estimation (KDE)

$$
\hat{f}(x) = \frac{1}{nh^2} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)
$$

Smoothed density surface showing probability of site occurrence, suitable for predictive modeling.

---

## Environment Setup

### Install Dependencies

```bash
pip install geopandas>=0.14 shapely>=2.0 numpy>=1.24 scipy>=1.11 \
            matplotlib>=3.7 rasterio>=1.3
```

### Verify Installation

```python
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Create test GeoDataFrame
sites = gpd.GeoDataFrame(
    {'name': ['Site A', 'Site B', 'Site C']},
    geometry=[Point(0, 0), Point(1, 1), Point(2, 0.5)],
    crs='EPSG:4326',
)
print(f"GeoPandas: {gpd.__version__}")
print(f"Test GeoDataFrame: {len(sites)} sites")
```

---

## Core Workflow

### Step 1: Site Distribution Mapping and Nearest Neighbor Analysis

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.stats import norm
from shapely.geometry import Point, Polygon

# ------------------------------------------------------------------ #
# Simulate archaeological site survey data
# (In practice: load from shapefile, CSV, or geodatabase)
# ------------------------------------------------------------------ #

np.random.seed(42)
n_sites = 80
study_area_km2 = 2500  # km²

# Two clusters + random background (realistic site distribution)
# Cluster 1: river valley settlement cluster
c1_x = np.random.normal(120, 5, 30)
c1_y = np.random.normal(45, 3, 30)
# Cluster 2: upland sites
c2_x = np.random.normal(140, 4, 25)
c2_y = np.random.normal(55, 3, 25)
# Random background
rand_x = np.random.uniform(100, 160, 25)
rand_y = np.random.uniform(35, 65, 25)

x_all = np.concatenate([c1_x, c2_x, rand_x])
y_all = np.concatenate([c1_y, c2_y, rand_y])

# Site types and periods
site_type = np.array(
    ['settlement']*30 + ['ceremonial']*10 + ['artifact scatter']*15 +
    ['unknown']*15 + ['lithic workshop']*10
)[:n_sites]
period = np.random.choice(['Neolithic', 'Bronze Age', 'Iron Age'], n_sites, p=[0.3, 0.4, 0.3])
n_finds = np.random.randint(5, 500, n_sites)

# Create GeoDataFrame (UTM-like coordinates in km)
geometry = [Point(x, y) for x, y in zip(x_all[:n_sites], y_all[:n_sites])]
sites_gdf = gpd.GeoDataFrame({
    'site_id': [f"S{i:04d}" for i in range(n_sites)],
    'site_type': site_type,
    'period': period,
    'n_finds': n_finds,
    'x_km': x_all[:n_sites],
    'y_km': y_all[:n_sites],
}, geometry=geometry)

print(f"Archaeological survey: {len(sites_gdf)} sites")
print(sites_gdf['site_type'].value_counts())

# ---- Nearest Neighbor Analysis ---------------------------------- #
coords = np.array(list(zip(sites_gdf['x_km'], sites_gdf['y_km'])))
tree = KDTree(coords)
nn_dists, _ = tree.query(coords, k=2)  # k=2: skip self
nn_dists = nn_dists[:, 1]  # First neighbor (skip self)

mean_nn = nn_dists.mean()
expected_nn = 0.5 / np.sqrt(n_sites / study_area_km2)
R_statistic = mean_nn / expected_nn

# Z-test
sigma_nn = 0.26136 / np.sqrt(n_sites**2 / study_area_km2)
z_score = (mean_nn - expected_nn) / sigma_nn
p_value = 2 * norm.sf(abs(z_score))

print(f"\nNearest Neighbor Analysis:")
print(f"  Mean observed NN distance: {mean_nn:.3f} km")
print(f"  Expected (random):         {expected_nn:.3f} km")
print(f"  R statistic:               {R_statistic:.4f}  ({'clustered' if R_statistic < 1 else 'dispersed' if R_statistic > 1 else 'random'})")
print(f"  Z-score: {z_score:.3f},  p-value: {p_value:.4f}")

# ---- Map -------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Site distribution colored by type
type_colors = {'settlement':'#e74c3c', 'ceremonial':'#9b59b6',
               'artifact scatter':'#f39c12', 'unknown':'#95a5a6', 'lithic workshop':'#27ae60'}
for site_type_val, color in type_colors.items():
    mask = sites_gdf['site_type'] == site_type_val
    subset = sites_gdf[mask]
    axes[0].scatter(subset['x_km'], subset['y_km'],
                    c=color, s=subset['n_finds']/10 + 20, alpha=0.7,
                    edgecolors='black', linewidths=0.5, label=f"{site_type_val} (n={mask.sum()})")
axes[0].set_xlabel("Easting (km)"); axes[0].set_ylabel("Northing (km)")
axes[0].set_title("Archaeological Site Distribution")
axes[0].legend(fontsize=7, loc='upper left'); axes[0].grid(True, alpha=0.3)

# NN distance histogram
axes[1].hist(nn_dists, bins=20, color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.7)
axes[1].axvline(mean_nn, color='blue', linewidth=2, linestyle='-', label=f'Observed mean={mean_nn:.2f} km')
axes[1].axvline(expected_nn, color='red', linewidth=2, linestyle='--', label=f'Expected (random)={expected_nn:.2f} km')
axes[1].set_xlabel("Nearest Neighbor Distance (km)"); axes[1].set_ylabel("Frequency")
axes[1].set_title(f"NN Distance Distribution\nR={R_statistic:.3f}, p={p_value:.4f}")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("site_distribution.png", dpi=150)
plt.show()
```

### Step 2: Kernel Density Estimation and Hotspot Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------ #
# KDE site density surface for predictive mapping
# ------------------------------------------------------------------ #

x = sites_gdf['x_km'].values
y = sites_gdf['y_km'].values

# Grid for density estimation
grid_res = 0.5  # km
x_grid = np.arange(x.min()-2, x.max()+2, grid_res)
y_grid = np.arange(y.min()-2, y.max()+2, grid_res)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Kernel density estimation (Scott's rule bandwidth)
xy = np.vstack([x, y])
kde = gaussian_kde(xy, bw_method='scott')
density = kde(np.vstack([X_grid.ravel(), Y_grid.ravel()])).reshape(X_grid.shape)

# ---- Hotspot analysis: threshold at 75th percentile ----------- #
hotspot_threshold = np.percentile(density, 75)
hotspot_mask = density > hotspot_threshold

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# KDE density surface
cf = axes[0].contourf(X_grid, Y_grid, density, levels=20, cmap='hot_r', alpha=0.8)
plt.colorbar(cf, ax=axes[0], label='Site density')
axes[0].scatter(x, y, c='white', s=15, edgecolors='black', linewidths=0.5, zorder=5, alpha=0.7)
axes[0].set_xlabel("Easting (km)"); axes[0].set_ylabel("Northing (km)")
axes[0].set_title("Kernel Density Estimation — Archaeological Sites")
axes[0].grid(True, alpha=0.2)

# Hotspot zones
axes[1].contourf(X_grid, Y_grid, density, levels=20, cmap='YlOrRd', alpha=0.7)
axes[1].contour(X_grid, Y_grid, hotspot_mask.astype(float), levels=[0.5],
                colors='red', linewidths=2)
axes[1].scatter(x, y, c='black', s=15, zorder=5, alpha=0.5)
axes[1].set_xlabel("Easting (km)"); axes[1].set_ylabel("Northing (km)")
axes[1].set_title("Site Hotspot Zones\n(75th percentile density threshold)")
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("kde_hotspots.png", dpi=150)
plt.show()

# Count sites in hotspot zones
n_hotspot = np.sum(density[np.round(
    (y[:, None] - y_grid[0]) / grid_res).astype(int).clip(0, len(y_grid)-1),
    np.round((x[:, None] - x_grid[0]) / grid_res).astype(int).clip(0, len(x_grid)-1)
    [:, 0]] > hotspot_threshold)
print(f"\nSite density summary:")
print(f"  Grid resolution: {grid_res} km")
print(f"  Hotspot threshold: {hotspot_threshold:.4e} km⁻²")
print(f"  Hotspot area: {hotspot_mask.sum() * grid_res**2:.1f} km² ({hotspot_mask.sum()/hotspot_mask.size*100:.1f}% of study area)")
```

### Step 3: Site Catchment and Thiessen Polygon Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
import geopandas as gpd

# ------------------------------------------------------------------ #
# Site catchment buffers and Thiessen (Voronoi) territorial polygons
# ------------------------------------------------------------------ #

# ---- Site catchment buffers (1-hour and 2-hour walking radius) -- #
# Tobler's hiking function: ~5 km/hour on flat terrain
WALK_1H = 5.0  # km
WALK_2H = 10.0 # km

# Select settlement sites only
settlements = sites_gdf[sites_gdf['site_type'] == 'settlement'].copy()

# Create buffered catchment areas
settlements_1h = settlements.copy()
settlements_1h['geometry'] = settlements_1h.geometry.buffer(WALK_1H)
settlements_2h = settlements.copy()
settlements_2h['geometry'] = settlements_2h.geometry.buffer(WALK_2H)

# ---- Thiessen (Voronoi) polygons for territorial analysis ------- #
# Study area bounding polygon (padded)
x_min, x_max = x.min()-5, x.max()+5
y_min, y_max = y.min()-5, y.max()+5
study_area_polygon = Polygon([(x_min,y_min),(x_max,y_min),(x_max,y_max),(x_min,y_max)])

# Voronoi tessellation for all sites
points = np.array(list(zip(x, y)))

# Add bounding points to clip Voronoi
n_bound = 50
angle = np.linspace(0, 2*np.pi, n_bound, endpoint=False)
R_bound = 200
bound_pts = np.column_stack([R_bound*np.cos(angle) + points[:,0].mean(),
                              R_bound*np.sin(angle) + points[:,1].mean()])
all_points = np.vstack([points, bound_pts])

vor = Voronoi(all_points)

# Build Voronoi polygons clipped to study area
voronoi_polys = []
for i, pt in enumerate(points):
    region_idx = vor.point_region[i]
    region = vor.regions[region_idx]
    if -1 in region or len(region) == 0:
        voronoi_polys.append(None)
        continue
    poly = Polygon(vor.vertices[region])
    clipped = poly.intersection(study_area_polygon)
    voronoi_polys.append(clipped)

thiessen_gdf = gpd.GeoDataFrame({
    'site_id': sites_gdf['site_id'].values,
    'site_type': sites_gdf['site_type'].values,
    'period': sites_gdf['period'].values,
}, geometry=voronoi_polys)
thiessen_gdf = thiessen_gdf[thiessen_gdf.geometry.notna()]

# Compute territory areas
thiessen_gdf['area_km2'] = thiessen_gdf.geometry.area

print("\nThiessen polygon territory sizes:")
print(thiessen_gdf['area_km2'].describe().round(2))

# ---- Plot -------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Catchment buffers
settlements_1h.plot(ax=axes[0], color='blue', alpha=0.1, edgecolor='blue', linewidth=0.5)
settlements_2h.plot(ax=axes[0], color='green', alpha=0.05, edgecolor='green', linewidth=0.5)
sites_gdf.plot(ax=axes[0], column='site_type', cmap='tab10', markersize=15, legend=True,
               legend_kwds={'fontsize': 7, 'loc': 'upper left'})
axes[0].set_title("Site Catchment Analysis\n(Blue=1hr, Green=2hr walking radius)")
axes[0].set_xlabel("Easting (km)"); axes[0].set_ylabel("Northing (km)")
axes[0].grid(True, alpha=0.3)

# Thiessen polygons
thiessen_gdf.plot(ax=axes[1], column='area_km2', cmap='Blues', edgecolor='black',
                  linewidth=0.5, alpha=0.7, legend=True,
                  legend_kwds={'label': 'Territory (km²)'})
sites_gdf.plot(ax=axes[1], color='red', markersize=10, zorder=5)
axes[1].set_title("Thiessen Polygon Territories")
axes[1].set_xlabel("Easting (km)"); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("catchment_thiessen.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Predictive Site Location Modeling

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Predictive site model: classify site/non-site based on
# environmental variables (slope, water distance, soil type, elevation)
# ------------------------------------------------------------------ #

np.random.seed(42)
n_total = 300  # Sites + pseudo-absence points

# Environmental predictors (simulated)
elevation  = np.concatenate([np.random.normal(200, 50, n_sites),      # Sites prefer low elev.
                              np.random.normal(400, 100, n_total-n_sites)])  # Non-sites higher
water_dist = np.concatenate([np.random.exponential(2, n_sites),        # Sites near water
                              np.random.exponential(8, n_total-n_sites)])
slope      = np.concatenate([np.random.uniform(0, 5, n_sites),         # Sites on gentle slopes
                              np.random.uniform(0, 25, n_total-n_sites)])
soil_type  = np.concatenate([np.random.choice([1,2], n_sites, p=[0.7,0.3]),
                              np.random.choice([1,2], n_total-n_sites, p=[0.3,0.7])])

y_label = np.array([1]*n_sites + [0]*(n_total-n_sites))

X_pred = pd.DataFrame({
    'elevation':  elevation,
    'water_dist': water_dist,
    'slope':      slope,
    'soil_type':  soil_type,
})

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
cv_scores = cross_val_score(rf, X_pred, y_label, cv=5, scoring='roc_auc')
print(f"Cross-validated AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

rf.fit(X_pred, y_label)

# Feature importance
fi = pd.Series(rf.feature_importances_, index=X_pred.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(fi.round(4))

fig, ax = plt.subplots(figsize=(7, 4))
fi.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black', linewidth=0.7)
ax.set_ylabel("Importance"); ax.set_title("Predictive Site Model — Feature Importance")
ax.tick_params(axis='x', rotation=0); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig("predictive_model.png", dpi=150); plt.show()
```

---

## Troubleshooting

### Error: `CRSError: Input does not contain CRS`

**Fix**: Always set CRS when creating GeoDataFrame:
```python
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
# Convert to projected CRS for distance calculations:
gdf_projected = gdf.to_crs('EPSG:32633')  # UTM zone 33N
```

### Buffer distances are in degrees (not km)

**Cause**: Data is in geographic (lat/lon) CRS.

**Fix**: Project first:
```python
gdf_m = gdf.to_crs('EPSG:3857')  # Web Mercator (meters)
gdf_m['geometry'] = gdf_m.geometry.buffer(5000)  # 5 km buffer
```

### Voronoi regions extend to infinity

**Fix**: Add bounding box points before Voronoi computation (as shown above), then clip to study area.

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| geopandas | 0.14 | Requires shapely ≥ 2.0 for performance |
| shapely | 2.0, 2.1 | Major API change from 1.x to 2.x |
| rasterio | 1.3 | For DEM reading |

---

## External Resources

### Official Documentation

- [GeoPandas documentation](https://geopandas.org)
- [Shapely documentation](https://shapely.readthedocs.io)

### Key Papers / Books

- Wheatley, D. & Gillings, M. (2002). *Spatial Technology and Archaeology*. Taylor & Francis.
- Verhagen, P. (2007). *Case Studies in Archaeological Predictive Modelling*. Leiden University Press.

---

## Examples

### Example 1: Spatial Autocorrelation (Moran's I)

```python
import numpy as np
from scipy.spatial.distance import cdist

def morans_i_sites(values, coords, k=8):
    """Spatial autocorrelation of an attribute across sites."""
    n = len(values)
    y = values - values.mean()
    dist = cdist(coords, coords)
    W = np.zeros((n, n))
    for i in range(n):
        nn = np.argsort(dist[i])[1:k+1]
        W[i, nn] = 1
    W /= W.sum(axis=1, keepdims=True)
    I = n * np.sum(W * np.outer(y, y)) / (W.sum() * np.sum(y**2))
    return I

coords = np.array(list(zip(sites_gdf['x_km'], sites_gdf['y_km'])))
I_finds = morans_i_sites(sites_gdf['n_finds'].values, coords)
print(f"Moran's I — number of finds: {I_finds:.4f}")
print("(>0 = spatially clustered; <0 = dispersed; 0 = random)")
```

### Example 2: Chronological Phase Mapping

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sites by period
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
periods = ['Neolithic', 'Bronze Age', 'Iron Age']
colors_per = ['#2ecc71', '#f39c12', '#e74c3c']

for ax, period_name, color in zip(axes, periods, colors_per):
    subset = sites_gdf[sites_gdf['period'] == period_name]
    ax.scatter(subset['x_km'], subset['y_km'],
               c=color, s=50, edgecolors='black', linewidths=0.5, alpha=0.8)
    ax.set_title(f"{period_name} sites (n={len(subset)})")
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    ax.set_xlim(95, 165); ax.set_ylim(30, 70)
    ax.grid(True, alpha=0.3)
plt.suptitle("Chronological Site Distribution by Period"); plt.tight_layout()
plt.savefig("chronological_phases.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
