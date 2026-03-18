---
name: urban-remote-sensing
description: >
  Use this Skill for urban remote sensing: impervious surface mapping, NDVI
  urban heat island, building footprint extraction, and land use change detection.
tags:
  - urban-science
  - remote-sensing
  - satellite
  - urban-heat-island
  - land-use
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
    - numpy>=1.24
    - scipy>=1.11
    - scikit-learn>=1.3
    - matplotlib>=3.7
    - rasterio>=1.3
    - geopandas>=0.14
last_updated: "2026-03-17"
status: "stable"
---

# Urban Remote Sensing Analysis

> **One-line summary**: Analyze urban land cover from satellite imagery: impervious surface mapping, NDVI-based urban heat island, LULC classification, and spectral change detection using scikit-learn and rasterio.

---

## When to Use This Skill

- When mapping impervious surfaces and urban expansion from multispectral imagery
- When analyzing urban heat island effects using LST and NDVI correlation
- When classifying urban land use/land cover with random forest or SVM
- When detecting urban change between two time periods
- When computing urban spatial metrics (patch density, fragmentation, edge density)
- When analyzing building shadows and 3D urban form from satellite data

**Trigger keywords**: urban remote sensing, impervious surface, urban heat island, NDVI urban, LULC classification, land use change, satellite urban, building extraction, urban sprawl, rasterio, spectral unmixing, change detection, Landsat urban

---

## Background & Key Concepts

### Urban Spectral Signatures

| Surface type | Red | NIR | SWIR | NDVI | NDBI |
|:------------|:----|:----|:-----|:-----|:-----|
| Dense vegetation | Low | High | Low | 0.6–0.9 | −0.5 to −0.3 |
| Urban/impervious | Medium | Low | High | −0.1 to 0.1 | 0.1–0.4 |
| Water | Low | Very low | Low | −0.3 to 0 | −0.5 to −0.2 |
| Bare soil | Medium | Medium | High | 0.0–0.2 | −0.1 to 0.1 |

### Urban Heat Island (UHI)

Surface UHI = LST(urban) − LST(suburban/rural). Correlated with:
- Low NDVI (less vegetation cooling)
- High NDBI (more impervious surface)
- Building density and anthropogenic heat

LST estimated from Landsat Band 10 (TIRS):

$$
T_s = \frac{B_{10} / \varepsilon^{1/4}}{1 + (\lambda \cdot B_{10} / \rho) \cdot \ln(\varepsilon)}
$$

---

## Environment Setup

### Install Dependencies

```bash
pip install numpy>=1.24 scipy>=1.11 scikit-learn>=1.3 matplotlib>=3.7 \
            rasterio>=1.3 geopandas>=0.14
# For actual satellite data download:
pip install earthengine-api>=0.1.380 geemap>=0.29
```

### Verify Installation

```python
import numpy as np
import rasterio
from rasterio.transform import from_bounds
print(f"rasterio: {rasterio.__version__}")

# Create synthetic raster for testing
from rasterio.io import MemoryFile
import numpy as np
data = np.random.randint(0, 4000, (4, 100, 100), dtype=np.uint16)
print(f"Test raster: shape={data.shape}")
```

---

## Core Workflow

### Step 1: Urban Spectral Index Computation

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ------------------------------------------------------------------ #
# Compute urban spectral indices from multispectral imagery
# Simulates Sentinel-2 / Landsat 8 bands
# ------------------------------------------------------------------ #

np.random.seed(42)
SIZE = 200  # 200×200 pixel scene

# ---- Simulate a city with different land cover types ------------ #
# Create city layout: center=downtown, rings=suburban, edges=rural

y_grid, x_grid = np.mgrid[0:SIZE, 0:SIZE]
cx, cy = SIZE/2, SIZE/2
dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

# Land cover map based on distance zones
# 0=downtown, 1=residential, 2=industrial, 3=urban park, 4=suburban, 5=water, 6=agriculture
lc_map = np.zeros((SIZE, SIZE), dtype=int)
lc_map[dist_from_center < 30]  = 0  # Downtown core
lc_map[(dist_from_center >= 30) & (dist_from_center < 60)] = 1   # Residential
lc_map[(dist_from_center >= 60) & (dist_from_center < 80)] = 2   # Industrial
lc_map[(dist_from_center >= 80) & (dist_from_center < 100)] = 4  # Suburban
lc_map[dist_from_center >= 100] = 6  # Agriculture/rural

# Add parks (circles within residential zone)
for px, py in [(50, 80), (150, 70), (80, 140)]:
    mask = (np.sqrt((x_grid-px)**2 + (y_grid-py)**2) < 15)
    lc_map[mask & (dist_from_center < 100)] = 3

# Add water body
lc_map[170:190, 10:50] = 5
lc_map[100:130, 160:190] = 5

# ---- Spectral signatures per class (Sentinel-2 surface reflectance, ×1e4)
signatures = {
    0: {'B4': 1800, 'B8': 1500, 'B11': 2200, 'B10': 320},  # Downtown (high NDBI)
    1: {'B4': 1500, 'B8': 2800, 'B11': 1800, 'B10': 295},  # Residential
    2: {'B4': 2000, 'B8': 1200, 'B11': 2500, 'B10': 310},  # Industrial
    3: {'B4': 800,  'B8': 5500, 'B11': 900,  'B10': 275},  # Urban park
    4: {'B4': 1200, 'B8': 3500, 'B11': 1500, 'B10': 285},  # Suburban
    5: {'B4': 600,  'B8': 400,  'B11': 300,  'B10': 270},  # Water
    6: {'B4': 900,  'B8': 4500, 'B11': 1200, 'B10': 278},  # Agriculture
}
sig_names = ['Downtown', 'Residential', 'Industrial', 'Urban park', 'Suburban', 'Water', 'Agriculture']

# Generate band images with noise
def make_band(lc, sig_key, noise=150):
    band = np.zeros_like(lc, dtype=float)
    for cls, sig in signatures.items():
        band[lc == cls] = sig[sig_key]
    band += np.random.randn(*band.shape) * noise
    return np.clip(band, 0, 10000)

B4  = make_band(lc_map, 'B4')   # Red
B8  = make_band(lc_map, 'B8')   # NIR
B11 = make_band(lc_map, 'B11')  # SWIR
LST_sim = make_band(lc_map, 'B10')  # Simulated LST (K)

# ---- Spectral indices ------------------------------------------- #
NDVI = (B8 - B4) / (B8 + B4 + 1e-6)   # Vegetation
NDBI = (B11 - B8) / (B11 + B8 + 1e-6)  # Built-up
NDWI = (B4 - B11) / (B4 + B11 + 1e-6)  # Water (MNDWI variant)

# Impervious surface proxy: high NDBI, low NDVI
impervious_mask = (NDBI > 0.05) & (NDVI < 0.2)
print(f"Impervious surface fraction: {impervious_mask.mean()*100:.1f}%")

# ---- Visualize -------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Land cover map
lc_colors = ['#8B0000','#FFA500','#808080','#228B22','#90EE90','#0000FF','#90EE90']
cmap_lc = ListedColormap(lc_colors)
im0 = axes[0][0].imshow(lc_map, cmap=cmap_lc, vmin=0, vmax=6)
axes[0][0].set_title("Land Cover Map")
plt.colorbar(im0, ax=axes[0][0], ticks=range(7))
axes[0][0].set_xticks([]); axes[0][0].set_yticks([])

# NDVI
im1 = axes[0][1].imshow(NDVI, cmap='RdYlGn', vmin=-0.5, vmax=0.9)
plt.colorbar(im1, ax=axes[0][1], label='NDVI')
axes[0][1].set_title("NDVI (Vegetation)"); axes[0][1].set_xticks([]); axes[0][1].set_yticks([])

# NDBI
im2 = axes[0][2].imshow(NDBI, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
plt.colorbar(im2, ax=axes[0][2], label='NDBI')
axes[0][2].set_title("NDBI (Built-up Index)"); axes[0][2].set_xticks([]); axes[0][2].set_yticks([])

# LST
im3 = axes[1][0].imshow(LST_sim, cmap='hot', vmin=265, vmax=335)
plt.colorbar(im3, ax=axes[1][0], label='LST (K)')
axes[1][0].set_title("Land Surface Temperature"); axes[1][0].set_xticks([]); axes[1][0].set_yticks([])

# Impervious surface
im4 = axes[1][1].imshow(impervious_mask.astype(float), cmap='Reds', vmin=0, vmax=1)
plt.colorbar(im4, ax=axes[1][1], label='Impervious (1=yes)')
axes[1][1].set_title(f"Impervious Surface ({impervious_mask.mean()*100:.1f}%)")
axes[1][1].set_xticks([]); axes[1][1].set_yticks([])

# UHI: NDVI vs LST scatter
ndvi_flat = NDVI.flatten()
lst_flat  = LST_sim.flatten()
im5 = axes[1][2].scatter(ndvi_flat[::20], lst_flat[::20], c=lc_map.flatten()[::20],
                          cmap=cmap_lc, vmin=0, vmax=6, s=5, alpha=0.5)
from scipy.stats import pearsonr
r, p = pearsonr(ndvi_flat, lst_flat)
axes[1][2].set_xlabel("NDVI"); axes[1][2].set_ylabel("LST (K)")
axes[1][2].set_title(f"UHI: NDVI vs. LST\n(r={r:.3f}, p={p:.2e})")
axes[1][2].grid(True, alpha=0.3)

plt.suptitle("Urban Remote Sensing Analysis — Spectral Indices and UHI")
plt.tight_layout()
plt.savefig("urban_spectral_analysis.png", dpi=150)
plt.show()
```

### Step 2: Urban LULC Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------------ #
# Supervised LULC classification using spectral features
# ------------------------------------------------------------------ #

# Build feature matrix: band values + derived indices
features = np.column_stack([
    B4.flatten(), B8.flatten(), B11.flatten(),
    NDVI.flatten(), NDBI.flatten(), NDWI.flatten(),
    LST_sim.flatten(),
])
labels = lc_map.flatten()

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.25, random_state=42, stratify=labels
)

# Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Overall accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=sig_names, zero_division=0))

# Feature importance
fi = dict(zip(['B4','B8','B11','NDVI','NDBI','NDWI','LST'], rf.feature_importances_))
print("\nFeature importances:")
for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Classify full image
classified_map = rf.predict(features).reshape(SIZE, SIZE)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cmap_lc = ListedColormap(['#8B0000','#FFA500','#808080','#228B22','#90EE90','#0000FF','#90EE90'])
axes[0].imshow(lc_map, cmap=cmap_lc, vmin=0, vmax=6)
axes[0].set_title("True Land Cover"); axes[0].set_xticks([]); axes[0].set_yticks([])

axes[1].imshow(classified_map, cmap=cmap_lc, vmin=0, vmax=6)
axes[1].set_title(f"RF Classification (OA={acc*100:.1f}%)"); axes[1].set_xticks([]); axes[1].set_yticks([])

# Confusion heat map
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')
im = axes[2].imshow(cm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=axes[2])
axes[2].set_xticks(range(7)); axes[2].set_xticklabels(sig_names, rotation=40, ha='right', fontsize=7)
axes[2].set_yticks(range(7)); axes[2].set_yticklabels(sig_names, fontsize=7)
axes[2].set_title("Confusion Matrix (normalized)"); axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")

plt.tight_layout()
plt.savefig("lulc_classification.png", dpi=150)
plt.show()
```

### Step 3: Urban Heat Island Profile Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------ #
# Urban heat island profile: transect and ring-average analysis
# ------------------------------------------------------------------ #

# Convert LST to Celsius
LST_C = LST_sim - 273.15

# Ring-average analysis
y_grid, x_grid = np.mgrid[0:SIZE, 0:SIZE]
cx, cy = SIZE/2, SIZE/2
dist_pix = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

ring_radii = np.arange(0, dist_pix.max(), 5)
ring_lst  = []
ring_ndvi = []
ring_dist = []
for r0, r1 in zip(ring_radii[:-1], ring_radii[1:]):
    mask = (dist_pix >= r0) & (dist_pix < r1)
    if mask.sum() > 0:
        ring_lst.append(LST_C[mask].mean())
        ring_ndvi.append(NDVI[mask].mean())
        ring_dist.append((r0+r1)/2)

ring_dist  = np.array(ring_dist)
ring_lst   = np.array(ring_lst)
ring_ndvi  = np.array(ring_ndvi)

# Pixel scale: assume 10m resolution
dist_km = ring_dist * 10 / 1000

uhi_intensity = ring_lst[:3].mean() - ring_lst[-5:].mean()
print(f"Urban Heat Island intensity: {uhi_intensity:.2f}°C")

# NDVI–LST correlation
r_corr, p_corr = pearsonr(NDVI.flatten(), LST_C.flatten())
print(f"NDVI–LST Pearson r = {r_corr:.4f}, p = {p_corr:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Ring-average LST
axes[0].plot(dist_km, ring_lst, 'r-o', linewidth=2, markersize=5)
axes[0].set_xlabel("Distance from center (km)"); axes[0].set_ylabel("Mean LST (°C)")
axes[0].set_title("Urban Heat Island Profile\n(Distance from city center)")
axes[0].grid(True, alpha=0.3)
axes[0].annotate(f"UHI intensity={uhi_intensity:.1f}°C",
                  xy=(0, ring_lst[0]), xytext=(2, ring_lst[0]+1),
                  arrowprops=dict(arrowstyle='->', color='red'), color='red')

# Ring-average NDVI
axes[1].plot(dist_km, ring_ndvi, 'g-o', linewidth=2, markersize=5)
axes[1].set_xlabel("Distance from center (km)"); axes[1].set_ylabel("Mean NDVI")
axes[1].set_title("Vegetation Gradient\n(Distance from city center)")
axes[1].grid(True, alpha=0.3)

# LST vs. NDVI scatter
sc = axes[2].scatter(NDVI.flatten()[::5], LST_C.flatten()[::5],
                      c=lc_map.flatten()[::5],
                      cmap=ListedColormap(['#8B0000','#FFA500','#808080','#228B22','#90EE90','#0000FF','#90EE90']),
                      vmin=0, vmax=6, s=2, alpha=0.3)
# Regression line
z = np.polyfit(NDVI.flatten(), LST_C.flatten(), 1)
x_fit = np.linspace(NDVI.min(), NDVI.max(), 100)
axes[2].plot(x_fit, np.polyval(z, x_fit), 'k-', linewidth=2.5, label=f"y={z[0]:.1f}x+{z[1]:.1f}")
axes[2].set_xlabel("NDVI"); axes[2].set_ylabel("LST (°C)")
axes[2].set_title(f"UHI: NDVI–LST Relationship\nr={r_corr:.3f}")
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("uhi_analysis.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Urban Expansion Change Detection

```python
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Detect urban expansion between two time periods
# ------------------------------------------------------------------ #

# Simulate 5-year urban expansion (additional 15% urban growth)
np.random.seed(99)
ndbi_t1 = NDBI.copy()
ndbi_t2 = NDBI.copy()

# Expand urban fringe (add new development at suburban edge)
suburban_mask = (dist_from_center >= 75) & (dist_from_center < 95)
ndbi_t2[suburban_mask] += np.random.uniform(0.1, 0.3, suburban_mask.sum())
ndbi_t2 = np.clip(ndbi_t2, -1, 1)

# Change detection: NDBI difference
ndbi_change = ndbi_t2 - ndbi_t1
new_urban = (ndbi_change > 0.15) & (ndbi_t1 < 0.1)  # Was non-urban, now urban

new_urban_pct = new_urban.mean() * 100
print(f"New urban pixels detected: {new_urban_pct:.2f}% of scene")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(ndbi_t1, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[0].set_title("NDBI T1"); axes[0].axis('off')
axes[1].imshow(ndbi_t2, cmap='RdBu_r', vmin=-0.5, vmax=0.5); axes[1].set_title("NDBI T2"); axes[1].axis('off')
axes[2].imshow(new_urban.astype(float), cmap='hot', vmin=0, vmax=1)
axes[2].set_title(f"Urban Expansion ({new_urban_pct:.1f}% new)"); axes[2].axis('off')
plt.tight_layout(); plt.savefig("urban_expansion.png", dpi=150); plt.show()
```

---

## Troubleshooting

### Rasterio fails to read satellite data

```python
# Load actual Sentinel-2 / Landsat data:
import rasterio
with rasterio.open("landsat_band4.tif") as src:
    B4_real = src.read(1).astype(float)
    transform = src.transform
    crs = src.crs
    print(f"CRS: {crs}, shape: {B4_real.shape}")
```

### Classification accuracy too low (<80%)

**Fixes**:
1. Add more training samples per class
2. Include texture features (GLCM)
3. Use higher spectral resolution or more bands
4. Try SVM with RBF kernel for small sample sizes

### NDVI undefined (0/0 division)

```python
NDVI = np.where((B8 + B4) > 0, (B8 - B4) / (B8 + B4), 0)
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| rasterio | 1.3 | CRS handling changed in 1.3 |
| geopandas | 0.14 | Requires shapely 2.0 |
| scikit-learn | 1.3, 1.4 | RandomForest API stable |

---

## External Resources

### Official Documentation

- [rasterio documentation](https://rasterio.readthedocs.io)
- [Sentinel-2 band descriptions (ESA)](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial)

### Key Papers

- Weng, Q. (2012). *Remote sensing of impervious surfaces in the urban areas*. ISPRS Journal.
- Voogt, J.A. & Oke, T.R. (2003). *Thermal remote sensing of urban climates*. Remote Sensing of Environment.

---

## Examples

### Example 1: Urban Greenness Index by Neighborhood

```python
import numpy as np
import pandas as pd

# Compute per-zone NDVI statistics
zones_analysis = []
for cls_id, cls_name in enumerate(sig_names):
    mask = lc_map == cls_id
    if mask.sum() > 0:
        zones_analysis.append({
            'class': cls_name,
            'area_pix': mask.sum(),
            'mean_ndvi': NDVI[mask].mean(),
            'mean_lst_C': LST_C[mask].mean(),
            'pct_impervious': impervious_mask[mask].mean() * 100,
        })

df_zones = pd.DataFrame(zones_analysis)
print("Urban zone characteristics:")
print(df_zones.round(3).to_string(index=False))
```

### Example 2: Sky View Factor (Urban Canyon)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simplified sky view factor: fraction of sky visible from ground
# High buildings in dense downtown block sky (low SVF)
svf_approx = np.exp(-0.1 * (NDBI + 0.5) * 10)  # Simplified model
svf_approx = np.clip(svf_approx, 0, 1)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(svf_approx, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Sky View Factor (0=blocked, 1=open)')
ax.set_title("Sky View Factor (Urban Canyon Proxy)"); ax.axis('off')
plt.tight_layout(); plt.savefig("sky_view_factor.png", dpi=150); plt.show()
print(f"Mean SVF downtown: {svf_approx[lc_map==0].mean():.3f}")
print(f"Mean SVF rural:    {svf_approx[lc_map==6].mean():.3f}")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
