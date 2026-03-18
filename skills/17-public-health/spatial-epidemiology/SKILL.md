---
name: spatial-epidemiology
description: >
  Use this Skill for spatial epidemiology: Moran's I spatial autocorrelation,
  LISA cluster detection, SaTScan-equivalent Knox test, and Bayesian disease mapping.
tags:
  - public-health
  - spatial-epidemiology
  - Moran-I
  - LISA
  - disease-mapping
  - cluster-detection
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
    - pysal>=2.3
    - libpysal>=4.7
    - geopandas>=0.13
    - numpy>=1.23
    - matplotlib>=3.6
    - scipy>=1.9
last_updated: "2026-03-18"
status: stable
---

# Spatial Epidemiology: Cluster Detection and Disease Mapping

> **TL;DR** — Spatial epidemiology toolkit: compute Moran's I global autocorrelation,
> LISA local cluster maps (HH/LL/HL/LH), standardized mortality ratios (SMR),
> Knox space-time clustering test, and choropleth disease maps.

---

## When to Use

Use this Skill when you need to:

- Test whether disease rates are spatially clustered or randomly distributed
- Identify specific local clusters (hot spots and cold spots) using LISA
- Calculate standardized mortality/morbidity ratios with indirect standardization
- Test for space-time clustering in point epidemic data (Knox test)
- Produce choropleth maps of disease burden by administrative unit
- Understand spatial weights matrices and their role in spatial statistics

| Analysis | Python Module | Key Function |
|---|---|---|
| Global Moran's I | `esda.Moran` | `Moran(y, w)` |
| Local Moran's I (LISA) | `esda.Moran_Local` | `Moran_Local(y, w)` |
| Spatial weights | `libpysal.weights` | `Queen.from_dataframe()` |
| Choropleth map | `geopandas` | `gdf.plot(column=...)` |
| Knox test | `scipy.spatial` | Custom implementation |

---

## Background

### Spatial Weights Matrix

The spatial weights matrix W defines neighbourhood relationships. Common types:

- **Queen contiguity**: two areas are neighbours if they share any boundary point
- **Rook contiguity**: neighbours share an edge (not just a point)
- **K-nearest neighbours**: the k closest centroids are neighbours
- **Distance band**: neighbours within a threshold distance d

Row-standardization converts W to W* where each row sums to 1, enabling
interpretation of WY as a spatial lag (weighted mean of neighbours).

### Moran's I Statistic

Global Moran's I measures overall spatial autocorrelation:

```
I = (N / S₀) × [Σᵢ Σⱼ wᵢⱼ(xᵢ - x̄)(xⱼ - x̄)] / Σᵢ(xᵢ - x̄)²
```

where S₀ = Σᵢ Σⱼ wᵢⱼ (sum of all weights).

- I ≈ E[I] = -1/(N-1): no spatial autocorrelation
- I > E[I]: positive autocorrelation (similar values cluster together)
- I < E[I]: negative autocorrelation (dissimilar values neighbour each other)

Significance is assessed via a permutation test (random reassignment of values
to areas, generating a null distribution).

### LISA (Local Indicators of Spatial Association)

Local Moran's I for area i:

```
Iᵢ = zᵢ × Σⱼ wᵢⱼ zⱼ
```

where zᵢ = (xᵢ - x̄) / std(x). LISA quadrant classification:

- **HH (High-High)**: hot spot — high value surrounded by high values
- **LL (Low-Low)**: cold spot — low value surrounded by low values
- **HL (High-Low)**: spatial outlier — high value surrounded by low values
- **LH (Low-High)**: spatial outlier — low value surrounded by high values

### Knox Test for Space-Time Clustering

The Knox test (Knox 1964) examines whether disease cases cluster in both space
and time simultaneously. It counts pairs of cases that are "close" in both
dimensions (within distance d AND within time t), then compares to a Poisson
null distribution under the hypothesis of random space-time independence.

---

## Environment Setup

```bash
conda create -n spatial-epi python=3.11 -y
conda activate spatial-epi
pip install "pysal>=2.3" "libpysal>=4.7" "esda>=2.4" \
            "geopandas>=0.13" "numpy>=1.23" "matplotlib>=3.6" "scipy>=1.9"

# Verify
python -c "import esda, libpysal, geopandas; print('Setup OK')"
```

---

## Core Workflow

### Step 1 — Moran's I and LISA Map for Disease Rates

```python
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from typing import Optional, Tuple


def simulate_disease_geodataframe(n_districts: int = 100) -> gpd.GeoDataFrame:
    """
    Create a synthetic grid GeoDataFrame with simulated disease rates.

    Returns a GeoDataFrame with: geometry (grid cells), district_id,
    observed_cases, expected_cases, smr, rate_per_100k.
    """
    from shapely.geometry import box
    import numpy as np

    rng = np.random.default_rng(42)
    side = int(np.ceil(np.sqrt(n_districts)))
    geometries, records = [], []

    # Add spatial structure: higher rates in upper-right quadrant
    for i in range(side):
        for j in range(side):
            if len(geometries) >= n_districts:
                break
            geom = box(j, i, j + 1, i + 1)
            # Spatially correlated rate: gradient + noise
            base_rate = 100 + 50 * (i + j) / (2 * side) + rng.normal(0, 20)
            base_rate = max(base_rate, 5)
            pop = int(rng.uniform(5000, 50000))
            expected = pop * 120 / 100_000  # population-expected at reference rate 120/100k
            observed = int(rng.poisson(expected * (base_rate / 120)))

            geometries.append(geom)
            records.append({
                'district_id': len(geometries),
                'population': pop,
                'observed_cases': observed,
                'expected_cases': round(expected, 2),
                'smr': round(observed / expected, 3) if expected > 0 else 0,
                'rate_per_100k': round(observed / pop * 100_000, 1),
            })

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs='EPSG:4326')
    return gdf


def compute_morans_i(
    gdf: gpd.GeoDataFrame,
    variable: str = 'smr',
    permutations: int = 999,
) -> dict:
    """
    Compute Global Moran's I for a spatial variable.

    Args:
        gdf:          GeoDataFrame with geometry and variable column.
        variable:     Name of the variable to test.
        permutations: Number of Monte Carlo permutations for significance test.

    Returns:
        Dict with I statistic, E[I], z-score, and p-value.
    """
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = 'r'  # row-standardize

    y = gdf[variable].values
    moran = Moran(y, w, permutations=permutations)

    result = {
        'I': round(float(moran.I), 4),
        'E_I': round(float(moran.EI), 4),
        'z_score': round(float(moran.z_norm), 4),
        'p_value_norm': round(float(moran.p_norm), 4),
        'p_value_sim': round(float(moran.p_sim), 4),
        'interpretation': 'positive autocorrelation' if moran.I > moran.EI else 'negative autocorrelation',
    }

    print(f"Global Moran's I for '{variable}':")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


def compute_lisa_clusters(
    gdf: gpd.GeoDataFrame,
    variable: str = 'smr',
    permutations: int = 999,
    significance: float = 0.05,
    output_path: str = 'lisa_cluster_map.png',
) -> gpd.GeoDataFrame:
    """
    Compute Local Moran's I (LISA) and produce a cluster map.

    Cluster types: HH (hot spot), LL (cold spot), HL/LH (outliers), NS (not significant).

    Args:
        gdf:          GeoDataFrame with spatial disease data.
        variable:     Variable to analyse.
        permutations: Permutations for pseudo p-values.
        significance: Alpha threshold for significance filter.
        output_path:  Path to save cluster map.

    Returns:
        GeoDataFrame with additional columns: local_I, p_sim, cluster_type.
    """
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = 'r'

    y = gdf[variable].values
    local_moran = Moran_Local(y, w, permutations=permutations)

    gdf = gdf.copy()
    gdf['local_I'] = local_moran.Is
    gdf['p_sim'] = local_moran.p_sim
    gdf['quadrant'] = local_moran.q  # 1=HH, 2=LH, 3=LL, 4=HL

    # Assign cluster type with significance filter
    quad_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    gdf['cluster_type'] = 'NS'
    sig_mask = gdf['p_sim'] < significance
    gdf.loc[sig_mask, 'cluster_type'] = gdf.loc[sig_mask, 'quadrant'].map(quad_map)

    # Plot
    color_map = {'HH': '#d7191c', 'LL': '#2c7bb6', 'HL': '#fdae61',
                 'LH': '#abd9e9', 'NS': '#d3d3d3'}
    gdf['color'] = gdf['cluster_type'].map(color_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: SMR choropleth
    gdf.plot(column=variable, cmap='RdYlGn_r', legend=True, ax=axes[0],
             legend_kwds={'label': variable, 'shrink': 0.7})
    axes[0].set_title(f'Choropleth: {variable}')
    axes[0].axis('off')

    # Right: LISA cluster map
    gdf.plot(color=gdf['color'], ax=axes[1], edgecolor='white', linewidth=0.3)
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    axes[1].legend(handles=patches, loc='lower right', fontsize=8)
    axes[1].set_title('LISA Cluster Map')
    axes[1].axis('off')

    fig.suptitle(f'Spatial Analysis of {variable}', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    cluster_counts = gdf['cluster_type'].value_counts()
    print(f"\nLISA cluster counts:\n{cluster_counts.to_string()}")

    return gdf


# --- Demo ---
gdf = simulate_disease_geodataframe(n_districts=100)
moran_result = compute_morans_i(gdf, variable='smr')
gdf_lisa = compute_lisa_clusters(gdf, variable='smr')
```

### Step 2 — SMR Calculation with Indirect Standardization

```python
import pandas as pd
import numpy as np


def indirect_standardization(
    observed_df: pd.DataFrame,
    reference_rates: pd.DataFrame,
    age_col: str = 'age_group',
    pop_col: str = 'population',
    cases_col: str = 'cases',
    ref_rate_col: str = 'reference_rate',
) -> pd.DataFrame:
    """
    Compute Standardized Mortality (or Morbidity) Ratios via indirect standardization.

    Indirect standardization applies reference (standard) age-specific rates to
    the study population to compute expected counts, then calculates:
    SMR = Observed / Expected

    Args:
        observed_df:     DataFrame with columns: district, age_group, population, cases.
        reference_rates: DataFrame with columns: age_group, reference_rate (per 100k).
        age_col:         Age group column name.
        pop_col:         Population column name.
        cases_col:       Observed cases column name.
        ref_rate_col:    Reference rate column name (per 100k population).

    Returns:
        DataFrame by district: observed, expected, smr, smr_lower95, smr_upper95.
    """
    from scipy import stats

    merged = observed_df.merge(reference_rates[[age_col, ref_rate_col]],
                                on=age_col, how='left')
    merged['expected'] = merged[pop_col] * merged[ref_rate_col] / 100_000

    summary = merged.groupby('district').agg(
        observed=(cases_col, 'sum'),
        expected=('expected', 'sum'),
    ).reset_index()

    summary['smr'] = summary['observed'] / summary['expected']

    # 95% CI using Poisson exact method (Byar's approximation)
    def poisson_ci(obs, exp, alpha=0.05):
        if obs == 0:
            lo = 0.0
        else:
            lo = stats.chi2.ppf(alpha / 2, 2 * obs) / (2 * exp)
        hi = stats.chi2.ppf(1 - alpha / 2, 2 * obs + 2) / (2 * exp)
        return lo, hi

    cis = summary['observed'].apply(
        lambda o: poisson_ci(o, summary.loc[summary['observed'] == o, 'expected'].iloc[0])
        if (summary.loc[summary['observed'] == o, 'expected'].iloc[0] > 0) else (0, np.inf)
    )

    lowers, uppers = [], []
    for i, row in summary.iterrows():
        lo, hi = poisson_ci(row['observed'], row['expected'])
        lowers.append(round(lo, 3))
        uppers.append(round(hi, 3))

    summary['smr_lower95'] = lowers
    summary['smr_upper95'] = uppers
    summary['smr'] = summary['smr'].round(3)

    print("SMR Summary (first 10 districts):")
    print(summary.head(10).to_string(index=False))
    return summary


# --- Demo ---
rng = np.random.default_rng(42)
age_groups = ['0-14', '15-44', '45-64', '65-74', '75+']

# Reference rates per 100k by age group
ref_rates = pd.DataFrame({
    'age_group': age_groups,
    'reference_rate': [5, 15, 80, 250, 600],
})

# Observed data for 20 districts
records = []
for dist in range(1, 21):
    for ag, ref_r in zip(age_groups, ref_rates['reference_rate']):
        pop = int(rng.uniform(1000, 8000))
        exp_cases = pop * ref_r / 100_000 * (1 + 0.3 * (dist > 10))
        cases = int(rng.poisson(exp_cases))
        records.append({'district': dist, 'age_group': ag, 'population': pop, 'cases': cases})

obs_df = pd.DataFrame(records)
smr_df = indirect_standardization(obs_df, ref_rates)
```

### Step 3 — Knox Space-Time Clustering Test

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt


def knox_test(
    cases_df: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    date_col: str = 'date',
    spatial_threshold_km: float = 2.0,
    temporal_threshold_days: int = 14,
    n_permutations: int = 999,
) -> dict:
    """
    Knox test for space-time clustering of disease cases.

    Counts pairs of cases that are both within spatial_threshold_km AND
    within temporal_threshold_days, then compares to a permutation null
    distribution (randomizing dates while keeping locations fixed).

    Args:
        cases_df:               DataFrame with lat, lon, date columns.
        lat_col, lon_col:       Coordinate column names.
        date_col:               Date column (datetime or date).
        spatial_threshold_km:   Spatial closeness threshold in km.
        temporal_threshold_days: Temporal closeness threshold in days.
        n_permutations:         Monte Carlo permutations.

    Returns:
        Dict with observed count, expected (mean), p-value, and Knox ratio.
    """
    df = cases_df[[lat_col, lon_col, date_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    n = len(df)

    if n < 10:
        raise ValueError(f"Too few cases ({n}) for Knox test; need >= 10.")

    # Haversine distance matrix (km)
    def haversine_matrix(lats, lons):
        R = 6371.0
        lat_r = np.radians(lats[:, None] - lats[None, :])
        lon_r = np.radians(lons[:, None] - lons[None, :])
        a = np.sin(lat_r / 2) ** 2 + np.cos(np.radians(lats[:, None])) * \
            np.cos(np.radians(lats[None, :])) * np.sin(lon_r / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    lats = df[lat_col].values
    lons = df[lon_col].values
    dates = df[date_col].values

    dist_km = haversine_matrix(lats, lons)
    np.fill_diagonal(dist_km, np.inf)

    dates_days = (dates.astype('datetime64[D]').astype(float))
    time_diff = np.abs(dates_days[:, None] - dates_days[None, :])
    np.fill_diagonal(time_diff, np.inf)

    # Indicator matrices (upper triangle only to avoid counting pairs twice)
    triu = np.triu_indices(n, k=1)
    space_close = dist_km[triu] <= spatial_threshold_km
    time_close = time_diff[triu] <= temporal_threshold_days

    # Observed Knox count
    observed = int(np.sum(space_close & time_close))

    # Permutation null distribution: shuffle dates
    rng = np.random.default_rng(0)
    null_counts = []
    for _ in range(n_permutations):
        perm_dates = rng.permutation(dates_days)
        perm_diff = np.abs(perm_dates[:, None] - perm_dates[None, :])
        perm_time_close = perm_diff[triu] <= temporal_threshold_days
        null_counts.append(int(np.sum(space_close & perm_time_close)))

    null_counts = np.array(null_counts)
    expected = float(null_counts.mean())
    p_value = float((null_counts >= observed).sum() + 1) / (n_permutations + 1)
    knox_ratio = observed / expected if expected > 0 else np.inf

    # Plot null distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_counts, bins=30, color='steelblue', alpha=0.7, density=True)
    ax.axvline(observed, color='red', linewidth=2, label=f'Observed = {observed}')
    ax.axvline(expected, color='orange', linewidth=1.5, linestyle='--',
               label=f'Expected = {expected:.1f}')
    ax.set_xlabel('Knox count (space-time pairs)')
    ax.set_ylabel('Density')
    ax.set_title(f'Knox Test — p = {p_value:.4f} | Knox ratio = {knox_ratio:.2f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig('knox_null_distribution.png', dpi=150)
    plt.close(fig)

    result = {
        'observed': observed,
        'expected_null_mean': round(expected, 2),
        'p_value': round(p_value, 4),
        'knox_ratio': round(knox_ratio, 3),
        'significant': p_value < 0.05,
        'n_cases': n,
        'spatial_threshold_km': spatial_threshold_km,
        'temporal_threshold_days': temporal_threshold_days,
    }

    print(f"Knox test: observed={observed}, expected={expected:.1f}, "
          f"ratio={knox_ratio:.2f}, p={p_value:.4f}")
    return result


# --- Demo ---
rng = np.random.default_rng(1)
n_cases = 200
# Simulate clustered outbreak in 30% of cases
cases = pd.DataFrame({
    'lat': np.concatenate([rng.uniform(40.5, 41.5, 140),
                           rng.uniform(40.8, 41.0, 60)]),
    'lon': np.concatenate([rng.uniform(-74.5, -73.5, 140),
                           rng.uniform(-74.1, -73.9, 60)]),
    'date': pd.to_datetime('2024-01-01') + pd.to_timedelta(
        np.concatenate([rng.integers(0, 90, 140),
                        rng.integers(10, 30, 60)]), unit='D')
})
knox_result = knox_test(cases, spatial_threshold_km=1.0, temporal_threshold_days=10)
```

---

## Advanced Usage

### Age-Standardized Rate Computation

```python
def age_standardized_rate(
    observed_df: pd.DataFrame,
    standard_population: pd.DataFrame,
    age_col: str = 'age_group',
    pop_col: str = 'population',
    cases_col: str = 'cases',
) -> float:
    """
    Compute age-standardized rate (direct standardization) per 100,000 population.
    Uses an external standard population (e.g., World Standard Population 2000).
    """
    merged = observed_df.merge(
        standard_population[[age_col, 'standard_pop']],
        on=age_col, how='left'
    )
    merged['age_specific_rate'] = merged[cases_col] / merged[pop_col]
    total_standard = merged['standard_pop'].sum()
    asr = (merged['age_specific_rate'] * merged['standard_pop']).sum() / total_standard * 100_000
    return round(asr, 2)
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `libpysal.weights` returns islands | Isolated polygons with no neighbours | Use `k=5` KNN weights instead of contiguity |
| Moran's I p-value = 0.001 | Insufficient permutations | Increase `permutations=9999` |
| LISA all NS | Uniform spatial distribution | Check that values actually vary; confirm row-standardization |
| Knox count = 0 | Thresholds too strict | Double spatial/temporal thresholds and re-run |
| Haversine gives wrong distance | Lat/lon swapped | Confirm column order: latitude first, longitude second |
| `CRS mismatch` in geopandas | Different projections | Use `gdf.to_crs('EPSG:4326')` before analysis |

---

## External Resources

- PySAL Project: <https://pysal.org/>
- ESDA (Exploratory Spatial Data Analysis): <https://esda.readthedocs.io/>
- Anselin, L. (1995). "Local Indicators of Spatial Association — LISA."
  *Geographical Analysis*, 27(2), 93–115.
- Knox, E.G. (1964). "The Detection of Space-Time Interactions." *Applied Statistics*, 13(1), 25–29.
- Elliott, P. et al. (2000). *Spatial Epidemiology: Methods and Applications*. Oxford University Press.
- GeoDa software (GUI spatial analysis): <https://geodacenter.github.io/>

---

## Examples

### Example 1 — Full Spatial Disease Rate Analysis

```python
gdf = simulate_disease_geodataframe(n_districts=144)
moran = compute_morans_i(gdf, variable='smr', permutations=999)
gdf_clusters = compute_lisa_clusters(gdf, variable='smr', significance=0.05,
                                     output_path='lisa_map.png')
hot_spots = gdf_clusters[gdf_clusters['cluster_type'] == 'HH']
print(f"\n{len(hot_spots)} significant hot spots detected (HH clusters)")
```

### Example 2 — SMR + Mapping

```python
smr_df = indirect_standardization(obs_df, ref_rates)
# Merge SMR back to GeoDataFrame for mapping
gdf_map = gdf.merge(smr_df[['district', 'smr']], left_on='district_id', right_on='district')

fig, ax = plt.subplots(figsize=(8, 8))
gdf_map.plot(column='smr', cmap='RdYlGn_r', legend=True, ax=ax,
             legend_kwds={'label': 'SMR', 'orientation': 'vertical'})
ax.set_title('Standardized Mortality Ratios by District')
ax.axis('off')
fig.tight_layout()
fig.savefig('smr_choropleth.png', dpi=150)
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Moran's I, LISA, SMR, Knox test, choropleth |
