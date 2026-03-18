---
name: satellite-imagery
description: >
  Use this Skill for Google Earth Engine satellite analysis: NDVI time series,
  LULC classification, change detection, and GeoTIFF export via geemap and ee.
tags:
  - earth-science
  - remote-sensing
  - google-earth-engine
  - geemap
  - satellite
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
    - geemap>=0.29
    - earthengine-api>=0.1.380
    - matplotlib>=3.7
    - numpy>=1.24
    - pandas>=2.0
    - geopandas>=0.14
last_updated: "2026-03-17"
status: "stable"
---

# Satellite Imagery Analysis with Google Earth Engine

> **One-line summary**: Analyze satellite imagery at scale using Google Earth Engine Python API and geemap: NDVI, LULC classification, change detection, and GeoTIFF export.

---

## When to Use This Skill

- When computing vegetation indices (NDVI, EVI) from Landsat or Sentinel-2
- When classifying land use / land cover (LULC) with Random Forest
- When detecting land cover change between two time periods
- When creating composites and time series at regional/continental scale
- When exporting results to GeoTIFF for downstream GIS analysis
- When accessing MODIS, Sentinel-1/2, or Landsat data archives

**Trigger keywords**: Google Earth Engine, GEE, Sentinel-2, Landsat, NDVI, LULC, land cover classification, change detection, satellite imagery, geemap

---

## Background & Key Concepts

### Google Earth Engine (GEE)

GEE is a planetary-scale geospatial analysis platform maintained by Google. It provides access to petabytes of satellite imagery and geospatial datasets with cloud-based computation:
- No local data download needed for analysis
- Scales from local to global automatically
- Python API (earthengine-api) + interactive maps (geemap)

### NDVI (Normalized Difference Vegetation Index)

$$
\text{NDVI} = \frac{NIR - Red}{NIR + Red} \in [-1, 1]
$$

Values > 0.3 indicate vegetation; > 0.6 indicates dense forest.

### Random Forest Classification

Supervised LULC classification: each pixel is a feature vector (spectral bands + indices) classified into land cover categories (forest, urban, water, agriculture).

### Change Detection

Bitemporal comparison: compute band differences or spectral indices between two dates. Magnitude of change $= \sqrt{\sum_i (b_{i,t2} - b_{i,t1})^2}$.

---

## Environment Setup

### Install Dependencies

```bash
pip install geemap>=0.29 earthengine-api>=0.1.380 matplotlib>=3.7 \
            numpy>=1.24 pandas>=2.0 geopandas>=0.14
```

### GEE Authentication

```bash
# Authenticate once (opens browser)
earthengine authenticate

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="<path-to-service-account-key.json>"
```

```python
import ee
import geemap

# Initialize with default project
try:
    ee.Initialize()
    print("GEE initialized successfully")
except Exception as e:
    print(f"Authentication needed: {e}")
    ee.Authenticate()
    ee.Initialize()
```

### Verify Installation

```python
import ee
ee.Initialize()
# Test: count images in Sentinel-2 collection
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
print(f"Sentinel-2 collection size (sample): {s2.limit(5).size().getInfo()}")
# Expected: 5
```

---

## Core Workflow

### Step 1: Load and Filter Satellite Imagery

```python
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np

ee.Initialize()

# Define area of interest (GeoJSON or ee.Geometry)
# Example: rectangular region over Beijing
aoi = ee.Geometry.Rectangle([116.0, 39.7, 116.8, 40.2])

def get_sentinel2_composite(aoi, start_date, end_date, cloud_pct=20):
    """
    Create cloud-free Sentinel-2 median composite for a time period.

    Parameters
    ----------
    aoi : ee.Geometry
    start_date, end_date : str, format 'YYYY-MM-DD'
    cloud_pct : int
        Maximum cloud percentage filter

    Returns
    -------
    ee.Image — median composite with SR bands
    """
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(aoi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
                  .select(["B2", "B3", "B4", "B8", "B11", "B12"]))  # Blue,Green,Red,NIR,SWIR1,SWIR2

    n_images = collection.size().getInfo()
    print(f"Images in collection: {n_images}")

    composite = collection.median()
    return composite, collection

composite_2023, collection = get_sentinel2_composite(aoi, "2023-06-01", "2023-09-30")
print("Composite created")

# Interactive map
Map = geemap.Map(center=[39.95, 116.4], zoom=10)
vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.4}
Map.addLayer(composite_2023, vis_params, "Sentinel-2 RGB 2023")
Map.addLayer(aoi, {}, "AOI")
Map
```

### Step 2: Compute Vegetation Indices

```python
import ee
import numpy as np
import matplotlib.pyplot as plt

ee.Initialize()

def add_indices(image):
    """Add NDVI, EVI, NDWI, NDBI to an image."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {"NIR": image.select("B8"), "RED": image.select("B4"),
         "BLUE": image.select("B2")}
    ).rename("EVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")  # water
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")  # built-up
    return image.addBands([ndvi, evi, ndwi, ndbi])

aoi = ee.Geometry.Rectangle([116.0, 39.7, 116.8, 40.2])

# Monthly NDVI time series
def monthly_ndvi(year, month):
    start = f"{year}-{month:02d}-01"
    end = ee.Date(start).advance(1, "month").format("YYYY-MM-dd").getInfo()
    img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi)
           .filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .median())
    if img.bandNames().size().getInfo() == 0:
        return None
    ndvi = img.normalizedDifference(["B8", "B4"])
    mean_ndvi = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=30, maxPixels=1e8
    ).get("nd").getInfo()
    return mean_ndvi

print("Computing monthly NDVI for 2023...")
months = range(1, 13)
ndvi_values = []
for m in months:
    val = monthly_ndvi(2023, m)
    ndvi_values.append(val if val else np.nan)
    print(f"  Month {m:02d}: NDVI={val:.4f}" if val else f"  Month {m:02d}: no data")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(list(months), ndvi_values, 'go-', linewidth=2, markersize=8)
ax.set_xlabel("Month (2023)")
ax.set_ylabel("Mean NDVI")
ax.set_title("Monthly NDVI Time Series")
ax.set_xticks(list(months))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ndvi_time_series.png", dpi=150)
plt.show()
```

### Step 3: LULC Classification with Random Forest

```python
import ee
import geemap

ee.Initialize()

aoi = ee.Geometry.Rectangle([116.0, 39.7, 116.8, 40.2])

# Prepare composite with indices
def build_feature_image(aoi, year):
    """Build multi-band feature image for classification."""
    composite = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                 .filterBounds(aoi)
                 .filterDate(f"{year}-06-01", f"{year}-09-30")
                 .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                 .median()
                 .select(["B2","B3","B4","B8","B11","B12"]))

    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return composite.addBands([ndvi, ndwi, ndbi])

feature_image = build_feature_image(aoi, 2023)

# Training samples: ESRI 2020 land cover as labels
# (In practice, use field-collected training points)
esri_lc = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
lc_2020 = esri_lc.filterDate("2020-01-01", "2020-12-31").first().clip(aoi)

# Sample training data
training_samples = feature_image.addBands(lc_2020.rename("label")).stratifiedSample(
    numPoints=200,
    classBand="label",
    region=aoi,
    scale=30,
    seed=42,
)

# Train Random Forest
classifier = ee.Classifier.smileRandomForest(100).train(
    features=training_samples,
    classProperty="label",
    inputProperties=feature_image.bandNames(),
)

# Classify
classified = feature_image.classify(classifier)

# Accuracy assessment
test_samples = feature_image.addBands(lc_2020.rename("label")).stratifiedSample(
    numPoints=50, classBand="label", region=aoi, scale=30, seed=99
)
validated = test_samples.classify(classifier)
confusion_matrix = validated.errorMatrix("label", "classification")
print(f"Overall accuracy: {confusion_matrix.accuracy().getInfo():.4f}")
print(f"Kappa: {confusion_matrix.kappa().getInfo():.4f}")

# Export result to Google Drive
export_task = ee.batch.Export.image.toDrive(
    image=classified,
    description="LULC_Classification_2023",
    folder="GEE_exports",
    region=aoi,
    scale=30,
    crs="EPSG:4326",
)
export_task.start()
print(f"Export task started: {export_task.status()['state']}")
```

---

## Advanced Usage

### Change Detection (Bitemporal)

```python
import ee
import geemap
import matplotlib.pyplot as plt

ee.Initialize()

aoi = ee.Geometry.Rectangle([116.0, 39.7, 116.8, 40.2])

def get_composite(aoi, year):
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(f"{year}-06-01", f"{year}-09-30")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .median()
            .select(["B4", "B8", "B11"]))

comp_2018 = get_composite(aoi, 2018)
comp_2023 = get_composite(aoi, 2023)

# NDVI change
ndvi_2018 = comp_2018.normalizedDifference(["B8", "B4"])
ndvi_2023 = comp_2023.normalizedDifference(["B8", "B4"])
ndvi_change = ndvi_2023.subtract(ndvi_2018).rename("NDVI_change")

# Threshold for significant change
change_mask = ndvi_change.abs().gt(0.15)
change_type = ee.Image(0).where(ndvi_change.lt(-0.15), 1).where(ndvi_change.gt(0.15), 2)

print("Change detection complete")
print("Use geemap.Map() to visualize change_type layer interactively")
```

---

## Troubleshooting

### Error: `EEException: User memory limit exceeded`

**Cause**: Too large an area or too fine resolution.

**Fix**:
```python
# Increase scale (lower resolution)
result = image.reduceRegion(reducer=ee.Reducer.mean(),
                             geometry=aoi, scale=100, maxPixels=1e9)
# Or clip to smaller area
aoi_small = aoi.buffer(-10000)  # 10 km buffer inward
```

### Error: `EEException: Computation timed out`

**Fix**: Export to Drive/Cloud Storage instead of getInfo():
```python
task = ee.batch.Export.image.toDrive(image=result, ...)
task.start()
print(task.status())
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| earthengine-api | 0.1.380, 0.1.400 | API changes frequently; pin version |
| geemap | 0.29, 0.32 | Interactive maps require Jupyter |

---

## External Resources

### Official Documentation

- [Google Earth Engine Python quickstart](https://developers.google.com/earth-engine/guides/python_install)
- [geemap documentation](https://geemap.org)

### Key Papers

- Gorelick, N. et al. (2017). *Google Earth Engine: Planetary-scale geospatial analysis for everyone*. Remote Sensing of Environment.

---

## Examples

### Example 1: Urban Heat Island — Night-Time Light Analysis

```python
# =============================================
# VIIRS night-time lights analysis
# =============================================
import ee, geemap, pandas as pd
ee.Initialize()

# Monthly VIIRS average radiance for city comparison
def get_monthly_ntl(city_geometry, year, month):
    start = f"{year}-{month:02d}-01"
    end = ee.Date(start).advance(1, "month").format("YYYY-MM-dd").getInfo()
    ntl = (ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
           .filterDate(start, end)
           .first()
           .select("avg_rad"))
    mean_val = ntl.reduceRegion(ee.Reducer.mean(), city_geometry, 500, maxPixels=1e8)
    return mean_val.get("avg_rad").getInfo()

# Compare two cities
cities = {
    "Beijing": ee.Geometry.Rectangle([116.2, 39.8, 116.6, 40.1]),
    "Shanghai": ee.Geometry.Rectangle([121.3, 31.1, 121.7, 31.4]),
}

records = []
for month in range(1, 13):
    for city, geom in cities.items():
        val = get_monthly_ntl(geom, 2023, month)
        records.append({"month": month, "city": city, "radiance": val})

df = pd.DataFrame(records)
print(df.pivot(index="month", columns="city", values="radiance").round(2))
```

**Interpreting these results**: Higher radiance values indicate more intense light emissions. Seasonal patterns reflect economic activity cycles.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
