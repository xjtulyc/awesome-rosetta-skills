---
name: agri-remote-sensing
description: >
  Use this Skill for agricultural remote sensing: Sentinel-2 crop type mapping
  (Random Forest), NDVI phenology analysis, LAI estimation, and yield
  prediction from satellite composites.
tags:
  - agriculture
  - remote-sensing
  - Sentinel-2
  - NDVI
  - crop-mapping
  - yield-prediction
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
    - geemap>=0.28
    - ee>=0.1.370
    - rioxarray>=0.15
    - scikit-learn>=1.2
    - numpy>=1.23
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Agricultural Remote Sensing with Sentinel-2

> **TL;DR** — Map crop types with Sentinel-2 + Random Forest, extract NDVI
> phenology metrics (SOS/EOS/peak), estimate LAI, and predict yield from
> satellite composites using Google Earth Engine and Python.

---

## When to Use

Use this Skill when you need to:

- Map crop types across a study region using multi-temporal Sentinel-2
  spectral composites and machine-learning classification.
- Derive phenological metrics (start-of-season, end-of-season, peak NDVI)
  from dense NDVI time series to characterize crop calendars.
- Estimate leaf area index (LAI) from empirical regression models calibrated
  with Sentinel-2 bands.
- Predict crop yield at field or county level using phenology features derived
  from satellite imagery.
- Build cloud-free seasonal composites (biweekly or monthly median) at scale.

**Do NOT use** this Skill for individual-plant-level phenotyping (UAV/drone),
or when ground-truth training data are unavailable.

---

## Background

### Sentinel-2 Band Structure

| Band | Name | Centre wavelength | Resolution |
|---|---|---|---|
| B2 | Blue | 490 nm | 10 m |
| B3 | Green | 560 nm | 10 m |
| B4 | Red | 665 nm | 10 m |
| B8 | NIR (broad) | 842 nm | 10 m |
| B11 | SWIR 1 | 1610 nm | 20 m |
| B12 | SWIR 2 | 2190 nm | 20 m |
| QA60 | Cloud mask | — | 60 m |

### Vegetation Indices

```
NDVI = (B8 - B4) / (B8 + B4)           # General greenness / biomass
EVI  = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)  # Reduces soil/atmosphere noise
LSWI = (B8 - B11) / (B8 + B11)         # Leaf water content / crop moisture
```

### Phenological Metrics (Threshold Method)

- **SOS** (Start of Season): first day NDVI crosses 25% of seasonal amplitude (ascending)
- **EOS** (End of Season): last day NDVI crosses 25% of amplitude (descending)
- **Peak**: day of maximum NDVI
- **Length of Season (LOS)** = EOS - SOS (days)

---

## Environment Setup

```bash
# Create and activate conda environment
conda create -n agrisat python=3.11 -y
conda activate agrisat

# Install dependencies
pip install geemap>=0.28 earthengine-api>=0.1.370 rioxarray>=0.15 \
            scikit-learn>=1.2 numpy>=1.23 matplotlib>=3.6 pandas>=1.5

# Authenticate Google Earth Engine (one-time)
earthengine authenticate
python -c "import ee; ee.Initialize(); print('GEE initialized OK')"
```

---

## Core Workflow

### Step 1 — Build Cloud-Free Sentinel-2 NDVI Time Series (GEE)

```python
import ee
import geemap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ee.Initialize()


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds and cirrus in Sentinel-2 SR using the QA60 band.

    Bit 10 = opaque clouds, Bit 11 = cirrus clouds.

    Args:
        image: Sentinel-2 SR image from COPERNICUS/S2_SR_HARMONIZED collection.

    Returns:
        Cloud-masked image with pixel values scaled to [0, 1].
    """
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
        qa.bitwiseAnd(cirrus_bit_mask).eq(0)
    )
    return image.updateMask(mask).divide(10000)


def add_vegetation_indices(image: ee.Image) -> ee.Image:
    """
    Add NDVI, EVI, and LSWI bands to a Sentinel-2 SR image.

    Args:
        image: Cloud-masked, scaled Sentinel-2 SR image.

    Returns:
        Image with additional bands: NDVI, EVI, LSWI.
    """
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = image.expression(
        "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
        {"NIR": image.select("B8"), "RED": image.select("B4"), "BLUE": image.select("B2")},
    ).rename("EVI")
    lswi = image.normalizedDifference(["B8", "B11"]).rename("LSWI")
    return image.addBands([ndvi, evi, lswi])


def get_biweekly_ndvi_series(
    roi: ee.Geometry,
    start_date: str,
    end_date: str,
    cloud_pct: int = 30,
) -> pd.DataFrame:
    """
    Build a biweekly median NDVI composite time series for a region of interest.

    Args:
        roi:        ee.Geometry defining the study area.
        start_date: ISO date string, e.g. '2022-01-01'.
        end_date:   ISO date string, e.g. '2022-12-31'.
        cloud_pct:  Maximum cloud cover percentage for image filtering.

    Returns:
        DataFrame with columns: date, NDVI, EVI, LSWI (region means).
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2_clouds)
        .map(add_vegetation_indices)
    )

    # Biweekly composites
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    periods = pd.date_range(start, end, freq="14D")

    records = []
    for p in periods:
        p_end = p + pd.Timedelta(days=13)
        composite = collection.filterDate(
            p.strftime("%Y-%m-%d"), p_end.strftime("%Y-%m-%d")
        ).median()

        stats = composite.select(["NDVI", "EVI", "LSWI"]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=roi, scale=10, maxPixels=1e9
        ).getInfo()

        records.append({
            "date": p,
            "NDVI": stats.get("NDVI"),
            "EVI":  stats.get("EVI"),
            "LSWI": stats.get("LSWI"),
        })

    df = pd.DataFrame(records).dropna(subset=["NDVI"])
    return df
```

### Step 2 — Extract Phenological Metrics

```python
def extract_phenology_metrics(
    ndvi_series: pd.Series,
    dates: pd.DatetimeIndex,
    threshold_fraction: float = 0.25,
) -> dict:
    """
    Extract SOS, EOS, and peak phenological metrics from an NDVI time series.

    Uses the threshold method: SOS/EOS are defined as the dates when NDVI
    crosses (threshold_fraction * amplitude) above the seasonal minimum.

    Args:
        ndvi_series:        NDVI values (pd.Series or array-like).
        dates:              Corresponding dates (pd.DatetimeIndex).
        threshold_fraction: Fraction of amplitude for SOS/EOS detection.

    Returns:
        Dictionary with keys: SOS, EOS, peak_date, peak_NDVI, LOS (days).
    """
    ndvi = np.array(ndvi_series)
    ndvi_min = ndvi.min()
    ndvi_max = ndvi.max()
    amplitude = ndvi_max - ndvi_min
    threshold = ndvi_min + threshold_fraction * amplitude

    peak_idx = int(np.argmax(ndvi))
    peak_date = dates[peak_idx]
    peak_ndvi = float(ndvi[peak_idx])

    # SOS: ascending crossing before peak
    sos_date = dates[0]
    for i in range(peak_idx):
        if ndvi[i] >= threshold:
            sos_date = dates[i]
            break

    # EOS: descending crossing after peak
    eos_date = dates[-1]
    for i in range(peak_idx, len(ndvi)):
        if ndvi[i] <= threshold:
            eos_date = dates[i]
            break

    los_days = (eos_date - sos_date).days

    return {
        "SOS": sos_date,
        "EOS": eos_date,
        "peak_date": peak_date,
        "peak_NDVI": round(peak_ndvi, 4),
        "LOS_days": los_days,
        "amplitude": round(float(amplitude), 4),
    }
```

### Step 3 — Random Forest Crop Type Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def train_crop_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    n_estimators: int = 200,
    max_depth: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a Random Forest classifier for crop type mapping.

    Args:
        features:      2-D array of shape (n_samples, n_features) — stacked
                       spectral bands and vegetation indices from training pixels.
        labels:        1-D integer class labels (0-indexed).
        class_names:   List of class name strings matching label indices.
        n_estimators:  Number of trees in the forest.
        max_depth:     Maximum tree depth (None = unlimited).
        test_size:     Fraction of data held out for accuracy assessment.
        random_state:  Random seed for reproducibility.

    Returns:
        Dictionary with: model, OA (overall accuracy), F1 per class,
        confusion matrix, feature importances.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    oa = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 55)
    print("Random Forest Crop Classification — Accuracy Assessment")
    print("=" * 55)
    print(f"  Overall Accuracy: {oa:.4f} ({oa * 100:.2f}%)")
    print(f"  Macro F1:         {report['macro avg']['f1-score']:.4f}")
    print(f"\n  Per-class F1 scores:")
    for cls in class_names:
        f1 = report.get(cls, {}).get("f1-score", float("nan"))
        print(f"    {cls:20s}: {f1:.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": [f"band_{i}" for i in range(features.shape[1])],
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "model": clf,
        "OA": oa,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": importance_df,
    }
```

---

## Advanced Usage

### LAI Estimation from Sentinel-2 NDVI

```python
def estimate_lai_from_ndvi(
    ndvi: np.ndarray,
    method: str = "baret2007",
) -> np.ndarray:
    """
    Estimate Leaf Area Index (LAI) from NDVI using empirical equations.

    Two calibrated models are provided:
      - 'baret2007': LAI = -ln((0.57 - NDVI) / 0.57) / 0.5
        (Baret et al. 2007, valid for 0 < NDVI < 0.9)
      - 'linear':    LAI = 5.0 * NDVI - 0.5 (simplified linear fit)

    Args:
        ndvi:   Array of NDVI values (clipped to valid range).
        method: Estimation method ('baret2007' or 'linear').

    Returns:
        Array of estimated LAI values (m²/m²).
    """
    ndvi_clipped = np.clip(ndvi, 0.01, 0.89)

    if method == "baret2007":
        lai = -np.log((0.57 - ndvi_clipped) / 0.57) / 0.5
    elif method == "linear":
        lai = 5.0 * ndvi_clipped - 0.5
    else:
        raise ValueError(f"Unknown LAI method: {method}. Use 'baret2007' or 'linear'.")

    return np.clip(lai, 0, 10)


def yield_prediction_from_phenology(
    phenology_features: pd.DataFrame,
    observed_yields: pd.Series,
    feature_cols: list[str] = None,
) -> dict:
    """
    Predict crop yield from phenological features using OLS regression.

    Args:
        phenology_features: DataFrame with per-field phenology metrics as columns
                            (peak_NDVI, SOS_doy, EOS_doy, LOS_days, amplitude, etc.).
        observed_yields:    Series of observed yields (same index as features).
        feature_cols:       Columns to use as predictors (defaults to all numeric).

    Returns:
        Dictionary with: model, R2, RMSE, predictions, coefficients.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    if feature_cols is None:
        feature_cols = phenology_features.select_dtypes(include=np.number).columns.tolist()

    X = phenology_features[feature_cols].values
    y = observed_yields.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Yield prediction OLS — R²={r2:.4f}, RMSE={rmse:.2f} kg/ha")
    return {
        "model": model,
        "R2": r2,
        "RMSE": rmse,
        "predictions": y_pred,
        "coefficients": dict(zip(feature_cols, model.coef_)),
        "intercept": model.intercept_,
    }
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `ee.Initialize()` fails | Credentials not set up | Run `earthengine authenticate` in terminal |
| Empty NDVI time series | ROI too small or cloud pct too low | Increase `cloud_pct` or expand ROI |
| All NDVI values NaN | Image collection filtered to zero images | Loosen date/cloud filter |
| `EEException: Computation timed out` | ROI too large for 10 m computation | Use `scale=20` or export to GCS |
| Low classification accuracy | Insufficient training samples | Aim for ≥50 samples per class |
| SOS not found (threshold not crossed) | Very low NDVI season | Lower `threshold_fraction` to 0.15 |
| LAI values > 10 | NDVI near 1.0 in water/shadow pixels | Mask non-vegetation before LAI calc |

---

## External Resources

- Google Earth Engine Python API: https://developers.google.com/earth-engine/guides/python_install
- Sentinel-2 L2A Product Guide: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/product-types/level-2a
- geemap documentation: https://geemap.org/
- ESA WorldCover (10 m land cover): https://esa-worldcover.org/
- TIMESAT (phenology extraction): https://web.nateko.lu.se/timesat/timesat.asp
- Baret et al. 2007 LAI model: https://doi.org/10.1016/j.rse.2006.12.015

---

## Examples

### Example 1 — NDVI Time Series and Phenology for Iowa Corn Belt

```python
def example_iowa_corn_ndvi():
    """Build NDVI time series for a corn field in Iowa and extract phenology."""
    # Define a small ROI (1 km²) in central Iowa
    roi = ee.Geometry.Rectangle([-93.62, 41.98, -93.61, 41.99])

    print("Fetching biweekly NDVI composites for 2022 corn season ...")
    df = get_biweekly_ndvi_series(roi, "2022-04-01", "2022-11-30", cloud_pct=40)
    print(f"  Got {len(df)} composites with valid NDVI.")

    # Extract phenology
    metrics = extract_phenology_metrics(df["NDVI"], pd.DatetimeIndex(df["date"]))
    print("\nPhenology metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Plot NDVI time series with phenology markers
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["date"], df["NDVI"], "o-", color="#2ca02c", label="NDVI (biweekly)")
    ax.axvline(metrics["SOS"], color="blue", linestyle="--", label=f"SOS ({metrics['SOS'].date()})")
    ax.axvline(metrics["EOS"], color="red",  linestyle="--", label=f"EOS ({metrics['EOS'].date()})")
    ax.axvline(metrics["peak_date"], color="orange", linestyle=":", label=f"Peak NDVI={metrics['peak_NDVI']:.2f}")
    ax.set_ylabel("NDVI")
    ax.set_title("Iowa Corn NDVI Time Series — 2022")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("iowa_corn_ndvi.png", dpi=150)
    return df, metrics
```

### Example 2 — Crop Type Classification with Accuracy Assessment

```python
def example_crop_classification():
    """
    Simulate Random Forest crop classification from synthetic spectral data.
    In practice, replace synthetic data with real GEE-extracted pixel values.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    class_names = ["corn", "soybean", "wheat", "fallow"]
    n_samples_per_class = 200
    n_features = 16  # 8 biweekly NDVI + 4 EVI + 4 LSWI

    # Simulate separable spectral signatures
    means = {
        "corn":    rng.uniform(0.4, 0.8, n_features),
        "soybean": rng.uniform(0.3, 0.7, n_features),
        "wheat":   rng.uniform(0.2, 0.6, n_features),
        "fallow":  rng.uniform(0.1, 0.3, n_features),
    }

    features_list, labels_list = [], []
    for label, (cls, mean_vec) in enumerate(means.items()):
        feat = rng.normal(mean_vec, 0.05, (n_samples_per_class, n_features))
        features_list.append(feat)
        labels_list.extend([label] * n_samples_per_class)

    features = np.vstack(features_list)
    labels   = np.array(labels_list)

    result = train_crop_classifier(features, labels, class_names)
    print(f"\nOverall Accuracy: {result['OA']:.4f}")

    # Feature importance plot
    fi = result["feature_importances"].head(8)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(fi["feature"], fi["importance"], color="#1f77b4")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 8 Features — Crop Type RF Classifier")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    return result


if __name__ == "__main__":
    example_crop_classification()
```

### Example 3 — Yield Prediction from Phenology Features

```python
def example_yield_prediction():
    """Predict county-level corn yield from simulated phenology features."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 300  # number of fields

    peak_ndvi  = rng.uniform(0.55, 0.90, n)
    sos_doy    = rng.integers(100, 140, n).astype(float)
    los_days   = rng.integers(120, 180, n).astype(float)
    amplitude  = rng.uniform(0.30, 0.60, n)

    # Synthetic yield: positively correlated with peak NDVI and LOS
    yield_kgha = 6000 + 4000 * peak_ndvi + 10 * los_days - 8 * sos_doy + rng.normal(0, 300, n)

    feat_df = pd.DataFrame({
        "peak_NDVI": peak_ndvi,
        "SOS_doy":   sos_doy,
        "LOS_days":  los_days,
        "amplitude": amplitude,
    })

    result = yield_prediction_from_phenology(feat_df, pd.Series(yield_kgha))
    print(f"\nYield model coefficients: {result['coefficients']}")
    return result


if __name__ == "__main__":
    example_yield_prediction()
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Sentinel-2 composites, phenology, RF classification, LAI, yield prediction |
