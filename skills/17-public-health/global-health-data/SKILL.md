---
name: global-health-data
description: >
  Global health data access and analysis: WHO GHO API, IHME GBD, Our World in Data,
  DALYs, age-standardization, health inequality, and choropleth mapping.
tags:
  - public-health
  - epidemiology
  - who
  - global-health
  - health-inequality
  - geopandas
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
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.11.0
  - matplotlib>=3.7.0
  - geopandas>=0.14.0
  - shapely>=2.0.0
  - seaborn>=0.12.0
  - statsmodels>=0.14.0
last_updated: "2026-03-17"
---

# Global Health Data Access and Analysis

This skill provides tools for accessing and analyzing global health data from major
open sources: the WHO Global Health Observatory API, IHME Global Burden of Disease
study data, and Our World in Data health datasets. It covers computing burden-of-disease
metrics (DALYs, YLLs, YLDs), age-standardized rates, health inequality measures, and
publication-quality choropleth maps.

## Prerequisites

```bash
pip install requests pandas numpy scipy matplotlib geopandas shapely seaborn statsmodels
```

For GBD data downloads you may need an IHME account. Set API tokens via environment
variables (no hardcoded credentials):

```bash
export WHO_API_KEY="<paste-your-who-api-key>"       # currently optional for public endpoints
export IHME_API_TOKEN="<paste-your-ihme-token>"     # for authenticated GBD API endpoints
```

## Core Functions

### 1. WHO Global Health Observatory (GHO) API

```python
import os
import time
import requests
import pandas as pd
import numpy as np
from typing import Optional


# WHO GHO OData API v3 base URL (public, no API key required for most indicators)
WHO_GHO_BASE = "https://ghoapi.azureedge.net/api"


def get_who_indicator(
    indicator_code: str,
    countries: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
    dimension_filter: Optional[dict] = None,
    page_size: int = 1000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Retrieve data for a WHO GHO indicator via the OData REST API.

    Parameters
    ----------
    indicator_code : str
        WHO indicator code, e.g. 'MDG_0000000007' (under-5 mortality rate),
        'NCD_BMI_30A' (obesity prevalence), 'WHOSIS_000001' (life expectancy).
        Browse codes at: https://ghoapi.azureedge.net/api/Indicator
    countries : list[str] | None
        List of ISO 3166-1 alpha-3 country codes, e.g. ['NGA', 'KEN', 'ETH'].
        If None, retrieves all countries.
    years : list[int] | None
        List of years to filter, e.g. [2000, 2005, 2010, 2015, 2019].
        If None, retrieves all available years.
    dimension_filter : dict | None
        Extra OData filter key-value pairs, e.g. {'Dim1': 'MLE'} for males.
    page_size : int
        Number of records per API page request.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    pd.DataFrame
        Columns: indicator_code, country_code, country_name, year, value,
        low, high, sex, dim1, dim2, comments.
    """
    filters = []
    if countries:
        country_filter = " or ".join([f"SpatialDim eq '{c}'" for c in countries])
        filters.append(f"({country_filter})")
    if years:
        year_filter = " or ".join([f"TimeDim eq {y}" for y in years])
        filters.append(f"({year_filter})")
    if dimension_filter:
        for key, val in dimension_filter.items():
            filters.append(f"{key} eq '{val}'")

    odata_filter = " and ".join(filters) if filters else None

    records = []
    skip = 0
    session = requests.Session()
    headers = {}

    # Use API key if provided (currently optional for most WHO endpoints)
    api_key = os.environ.get("WHO_API_KEY")
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key

    while True:
        params = {"$top": page_size, "$skip": skip}
        if odata_filter:
            params["$filter"] = odata_filter

        url = f"{WHO_GHO_BASE}/{indicator_code}"
        try:
            resp = session.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
        except requests.HTTPError as e:
            print(f"HTTP error for indicator {indicator_code}: {e}")
            break
        except requests.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(5)
            continue

        data = resp.json()
        batch = data.get("value", [])
        if not batch:
            break

        for item in batch:
            records.append({
                "indicator_code": indicator_code,
                "country_code": item.get("SpatialDim", ""),
                "country_name": item.get("SpatialDimType", ""),
                "year": item.get("TimeDim"),
                "value": item.get("NumericValue"),
                "low": item.get("Low"),
                "high": item.get("High"),
                "sex": item.get("Dim1", ""),
                "dim2": item.get("Dim2", ""),
                "comments": item.get("Comments", ""),
            })

        skip += page_size
        if verbose:
            print(f"  Fetched {len(records)} records so far...")

        if len(batch) < page_size:
            break
        time.sleep(0.3)

    df = pd.DataFrame(records)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")

    if verbose:
        print(f"Total records: {len(df)} for indicator '{indicator_code}'")

    return df


def list_who_indicators(search_term: str = "") -> pd.DataFrame:
    """
    List available WHO GHO indicators, optionally filtered by search term.

    Returns DataFrame with columns: indicator_code, indicator_name.
    """
    url = f"{WHO_GHO_BASE}/Indicator"
    params = {}
    if search_term:
        params["$filter"] = f"contains(IndicatorName, '{search_term}')"

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    items = resp.json().get("value", [])
    return pd.DataFrame([{
        "indicator_code": it["IndicatorCode"],
        "indicator_name": it["IndicatorName"],
    } for it in items])
```

### 2. IHME Global Burden of Disease Data

```python
def download_gbd_data(
    cause: str,
    location: str | list[str],
    metric: str = "Rate",
    year: int | list[int] = 2019,
    measure: str = "DALYs",
    sex: str = "Both",
    age_group: str = "All Ages",
    fallback_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download IHME GBD data via API or fall back to a pre-downloaded CSV.

    The IHME GBD API requires registration at healthdata.org.
    For unauthenticated use, download CSVs from https://ghdx.healthdata.org/gbd-results
    and pass the path via `fallback_csv_path`.

    Parameters
    ----------
    cause : str
        GBD cause of death/disability name (e.g., 'Diabetes mellitus',
        'Lower respiratory infections', 'Ischemic heart disease').
    location : str | list[str]
        Country name(s) or GBD super-region name.
    metric : str
        'Rate' (per 100k), 'Number', or 'Percent'.
    year : int | list[int]
        Year(s) to retrieve.
    measure : str
        'DALYs', 'Deaths', 'YLLs', 'YLDs', 'Prevalence', 'Incidence'.
    sex : str
        'Both', 'Male', 'Female'.
    age_group : str
        GBD age group (e.g., 'All Ages', '<5 years', '70+ years').
    fallback_csv_path : str | None
        Path to a locally downloaded GBD CSV as fallback.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns: location, year, cause, measure,
        metric, sex, age_group, value, lower_ci, upper_ci.
    """
    ihme_token = os.environ.get("IHME_API_TOKEN")

    if ihme_token:
        headers = {"Authorization": f"Bearer {ihme_token}"}
        base_url = "https://api.healthdata.org/gbd/v1/results"

        locations = [location] if isinstance(location, str) else location
        years = [year] if isinstance(year, int) else year

        params = {
            "cause_name": cause,
            "location_name": ",".join(locations),
            "metric_name": metric,
            "year_id": ",".join(str(y) for y in years),
            "measure_name": measure,
            "sex_name": sex,
            "age_name": age_group,
            "format": "json",
        }
        try:
            resp = requests.get(base_url, params=params,
                                headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("results", [])
            if records:
                df = pd.DataFrame(records)
                df = df.rename(columns={
                    "location_name": "location",
                    "year_id": "year",
                    "cause_name": "cause",
                    "measure_name": "measure",
                    "metric_name": "metric",
                    "sex_name": "sex",
                    "age_name": "age_group",
                    "val": "value",
                    "lower": "lower_ci",
                    "upper": "upper_ci",
                })
                return df[["location", "year", "cause", "measure",
                           "metric", "sex", "age_group",
                           "value", "lower_ci", "upper_ci"]]
        except requests.RequestException as e:
            print(f"IHME API request failed: {e}. Falling back to CSV.")

    if fallback_csv_path:
        df = pd.read_csv(fallback_csv_path)
        # Standardize column names from IHME CSV export format
        col_map = {
            "location_name": "location",
            "year_id": "year",
            "cause_name": "cause",
            "measure_name": "measure",
            "metric_name": "metric",
            "sex_name": "sex",
            "age_name": "age_group",
            "val": "value",
            "lower": "lower_ci",
            "upper": "upper_ci",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Apply filters
        locations = [location] if isinstance(location, str) else location
        years = [year] if isinstance(year, int) else year

        if "location" in df.columns:
            df = df[df["location"].isin(locations)]
        if "year" in df.columns:
            df = df[df["year"].isin(years)]
        if "cause" in df.columns and cause:
            df = df[df["cause"].str.contains(cause, case=False, na=False)]
        if "measure" in df.columns:
            df = df[df["measure"] == measure]

        return df

    raise RuntimeError(
        "No IHME_API_TOKEN set and no fallback_csv_path provided. "
        "Download GBD data from https://ghdx.healthdata.org/gbd-results "
        "and pass the CSV path as fallback_csv_path."
    )
```

### 3. DALYs, YLLs, and YLDs

```python
def compute_dalys(
    yll_series: pd.Series,
    yld_series: pd.Series,
    index_cols: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute DALYs as the sum of YLLs and YLDs.

    DALYs (Disability-Adjusted Life Years) = YLLs + YLDs.
    YLLs (Years of Life Lost) = premature mortality burden.
    YLDs (Years Lived with Disability) = morbidity burden.

    Parameters
    ----------
    yll_series : pd.Series
        YLL values (same index as yld_series).
    yld_series : pd.Series
        YLD values.
    index_cols : pd.DataFrame | None
        Optional metadata DataFrame to attach (same index).

    Returns
    -------
    pd.DataFrame
        DataFrame with yll, yld, daly columns plus metadata if provided.
    """
    result = pd.DataFrame({
        "yll": yll_series.values,
        "yld": yld_series.values,
        "daly": yll_series.values + yld_series.values,
    })
    if index_cols is not None:
        result = pd.concat([index_cols.reset_index(drop=True), result], axis=1)
    return result
```

### 4. Age-Standardized Rates

```python
# WHO World Standard Population 2000-2025 age weights (18 age groups)
WHO_WORLD_STANDARD_POPULATION = pd.DataFrame({
    "age_group": [
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
        "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
        "60-64", "65-69", "70-74", "75-79", "80-84", "85+",
    ],
    "weight": [
        0.0886, 0.0869, 0.0860, 0.0847, 0.0822, 0.0793,
        0.0761, 0.0715, 0.0659, 0.0604, 0.0537, 0.0455,
        0.0372, 0.0296, 0.0221, 0.0152, 0.0091, 0.0063,
    ],
})


def age_standardize(
    crude_rates: pd.Series,
    age_weights: pd.Series,
    reference_pop: Optional[pd.Series] = None,
    per_unit: float = 100_000,
) -> dict:
    """
    Compute directly age-standardized rate using the WHO world standard population.

    Parameters
    ----------
    crude_rates : pd.Series
        Age-specific crude rates (cases per person, NOT per 100k).
        Index should match age_weights index.
    age_weights : pd.Series
        Proportional weight for each age group (must sum to 1.0).
        Use WHO_WORLD_STANDARD_POPULATION['weight'] as default.
    reference_pop : pd.Series | None
        If provided, use direct standardization with this reference population
        (absolute counts); age_weights is ignored.
    per_unit : float
        Multiplier for output (default 100,000 for rates per 100k).

    Returns
    -------
    dict
        Keys: age_standardized_rate, variance (approximate), ci_95_lower, ci_95_upper.
    """
    rates = np.asarray(crude_rates, dtype=float)
    weights = np.asarray(age_weights, dtype=float)

    if reference_pop is not None:
        ref = np.asarray(reference_pop, dtype=float)
        weights = ref / ref.sum()

    # Normalize weights just in case
    weights = weights / weights.sum()

    asr = np.sum(weights * rates) * per_unit

    # Approximate variance (Chiang 1961 method)
    variance = np.sum((weights**2) * rates * (1 - rates + 1e-12)) * per_unit**2

    se = np.sqrt(variance)
    ci_lower = max(0.0, asr - 1.96 * se)
    ci_upper = asr + 1.96 * se

    return {
        "age_standardized_rate": asr,
        "standard_error": se,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "variance": variance,
    }
```

### 5. Health Inequality Measures

```python
def compute_concentration_index(
    health_outcome: pd.Series,
    wealth_rank: pd.Series,
    weights: Optional[pd.Series] = None,
) -> dict:
    """
    Compute the concentration index (CI) for measuring socioeconomic health inequality.

    The concentration index ranges from -1 (all disease burden in the poorest) to
    +1 (all disease burden in the richest). CI = 0 means perfect equality.

    Parameters
    ----------
    health_outcome : pd.Series
        Health variable (e.g., prevalence, mortality rate per individual or group).
    wealth_rank : pd.Series
        Socioeconomic rank variable (higher = wealthier). Same index.
    weights : pd.Series | None
        Population weights for grouped data.

    Returns
    -------
    dict
        Keys: concentration_index, standard_error, t_statistic, p_value,
        erreygers_ci (normalized version for bounded outcomes).
    """
    df = pd.DataFrame({
        "y": health_outcome.values,
        "rank": wealth_rank.values,
    }).dropna()

    if weights is not None:
        df["w"] = weights.reindex(df.index).fillna(1.0).values
    else:
        df["w"] = 1.0

    total_w = df["w"].sum()
    df["w_norm"] = df["w"] / total_w

    # Fractional rank (Erreygers & van Ourti method)
    df = df.sort_values("rank")
    df["cum_w"] = df["w_norm"].cumsum()
    df["frac_rank"] = df["cum_w"].shift(1).fillna(0) + df["w_norm"] / 2

    y_mean = (df["y"] * df["w_norm"]).sum()

    # Covariance form of CI
    cov_xy = (df["w_norm"] * (df["y"] - y_mean) * (df["frac_rank"] - 0.5)).sum()
    ci = 2.0 * cov_xy / (y_mean + 1e-12)

    # Standard error via OLS regression (Kakwani et al.)
    from statsmodels.api import OLS, add_constant
    X_reg = add_constant(df["frac_rank"])
    y_reg = 2.0 * df["y"] / y_mean
    try:
        model = OLS(y_reg, X_reg, weights=df["w_norm"]).fit(
            cov_type="HC3"
        )
        se = model.bse["frac_rank"] / 2.0
        t_stat = ci / (se + 1e-12)
        p_val = float(2 * (1 - stats_module.t.cdf(abs(t_stat), df=len(df) - 2)))
    except Exception:
        se = np.nan
        t_stat = np.nan
        p_val = np.nan

    # Erreygers normalized CI (for bounded variables like prevalence 0-1)
    y_max = df["y"].max()
    y_min = df["y"].min()
    erreygers_ci = 4 * y_mean / (y_max - y_min + 1e-12) * ci if y_max > y_min else np.nan

    return {
        "concentration_index": ci,
        "standard_error": se,
        "t_statistic": t_stat,
        "p_value": p_val,
        "erreygers_ci": erreygers_ci,
        "mean_health": y_mean,
        "n": len(df),
    }


import scipy.stats as stats_module


def compute_inequality_gaps(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    reference_group: str,
) -> pd.DataFrame:
    """
    Compute absolute and relative health gaps relative to a reference group.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `value_col` (health metric) and `group_col` (socioeconomic group).
    value_col : str
        Column with health values (e.g., mortality rate).
    group_col : str
        Column with group labels (e.g., income quintile).
    reference_group : str
        Group to use as reference (usually the most advantaged).

    Returns
    -------
    pd.DataFrame
        Group-level table with absolute_gap and relative_gap columns.
    """
    group_means = df.groupby(group_col)[value_col].mean().reset_index()
    ref_value = group_means.loc[
        group_means[group_col] == reference_group, value_col
    ].values[0]

    group_means["absolute_gap"] = group_means[value_col] - ref_value
    group_means["relative_gap"] = group_means[value_col] / (ref_value + 1e-12)
    return group_means
```

### 6. Choropleth Map Visualization

```python
def plot_global_health_map(
    df: pd.DataFrame,
    value_col: str,
    year: int,
    title: str,
    country_col: str = "country_code",
    year_col: str = "year",
    cmap: str = "YlOrRd",
    legend_label: str = "Value",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    output_file: Optional[str] = None,
    missing_color: str = "#cccccc",
) -> None:
    """
    Plot a global choropleth map of a health indicator using geopandas.

    Parameters
    ----------
    df : pd.DataFrame
        Health data with country ISO-3 codes and values.
    value_col : str
        Column to map.
    year : int
        Year to filter.
    title : str
        Map title.
    country_col : str
        Column with ISO 3166-1 alpha-3 country codes.
    year_col : str
        Column with year values.
    cmap : str
        Matplotlib colormap.
    legend_label : str
        Label for the colorbar.
    vmin, vmax : float | None
        Color scale min/max. Auto-scaled if None.
    output_file : str | None
        Save figure to this path if provided.
    missing_color : str
        Fill color for countries with no data.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Load Natural Earth world shapefile (bundled with geopandas)
    world = gpd.read_file(
        gpd.datasets.get_path("naturalearth_lowres")
    )
    world = world.rename(columns={"iso_a3": "iso3"})

    # Filter data to the requested year
    df_year = df[df[year_col] == year][[country_col, value_col]].copy()
    df_year = df_year.drop_duplicates(subset=[country_col])

    merged = world.merge(
        df_year,
        left_on="iso3",
        right_on=country_col,
        how="left",
    )

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    # Plot countries with missing data
    missing = merged[merged[value_col].isna()]
    missing.plot(ax=ax, color=missing_color, edgecolor="white", linewidth=0.3)

    # Plot countries with data
    present = merged[~merged[value_col].isna()]
    present.plot(
        column=value_col,
        ax=ax,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.3,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={
            "label": legend_label,
            "orientation": "horizontal",
            "fraction": 0.03,
            "pad": 0.04,
            "shrink": 0.6,
        },
    )

    ax.set_title(f"{title}\n(Year: {year})", fontsize=16, fontweight="bold")
    ax.axis("off")

    # Coverage note
    n_covered = len(present)
    n_total = len(world)
    ax.text(
        0.01, 0.02,
        f"Countries with data: {n_covered}/{n_total}  |  "
        f"Missing: {n_total - n_covered}",
        transform=ax.transAxes,
        fontsize=9,
        color="grey",
    )

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Map saved to {output_file}")
    plt.show()
```

## Example 1: Under-5 Mortality Trends in Sub-Saharan Africa (2000–2019)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# WHO GHO indicator: MDG_0000000007 = Under-5 mortality rate (per 1000 live births)
# ------------------------------------------------------------------

# Sub-Saharan African country ISO-3 codes (representative sample)
SSA_COUNTRIES = [
    "NGA", "ETH", "COD", "TZA", "KEN", "UGA", "MOZ", "GHA",
    "ZMB", "ZWE", "SEN", "MLI", "NER", "BFA", "CMR", "AGO",
    "MWI", "RWA", "MDG", "SOM",
]

YEARS = list(range(2000, 2020))

print("Fetching under-5 mortality data from WHO GHO...")
u5mr_df = get_who_indicator(
    indicator_code="MDG_0000000007",
    countries=SSA_COUNTRIES,
    years=YEARS,
    verbose=True,
)

# Filter to both-sex estimates
u5mr_df = u5mr_df[u5mr_df["sex"].isin(["", "BTSX", None])].copy()
u5mr_df = u5mr_df.dropna(subset=["value", "year"])
u5mr_df["year"] = u5mr_df["year"].astype(int)

print(f"\nData shape: {u5mr_df.shape}")
print(u5mr_df.groupby("country_code")["value"].describe().round(1))

# ------------------------------------------------------------------
# Time series plot
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Individual country trends
pivot = u5mr_df.pivot_table(
    index="year", columns="country_code", values="value", aggfunc="mean"
)

for country in pivot.columns:
    axes[0].plot(pivot.index, pivot[country], alpha=0.4, linewidth=1.0, color="steelblue")

# Regional mean
regional_mean = u5mr_df.groupby("year")["value"].mean()
regional_median = u5mr_df.groupby("year")["value"].median()

axes[0].plot(regional_mean.index, regional_mean.values,
             color="navy", linewidth=3, label="Regional mean")
axes[0].plot(regional_median.index, regional_median.values,
             color="darkred", linewidth=2, linestyle="--", label="Regional median")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Deaths per 1,000 live births")
axes[0].set_title("Under-5 Mortality Rate\nSub-Saharan Africa (2000–2019)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Percent reduction per country
start_val = u5mr_df[u5mr_df["year"] == 2000].set_index("country_code")["value"]
end_val = u5mr_df[u5mr_df["year"] == 2019].set_index("country_code")["value"]
common_idx = start_val.index.intersection(end_val.index)
pct_change = ((end_val[common_idx] - start_val[common_idx]) / start_val[common_idx] * 100)
pct_change = pct_change.sort_values()

colors = ["#d73027" if x > -30 else "#4dac26" for x in pct_change.values]
axes[1].barh(range(len(pct_change)), pct_change.values, color=colors, alpha=0.8)
axes[1].set_yticks(range(len(pct_change)))
axes[1].set_yticklabels(pct_change.index, fontsize=8)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("% Change in U5MR (2000–2019)")
axes[1].set_title("Percent Reduction in Under-5 Mortality\n(Green = >30% reduction)")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("ssa_u5mr_trends.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# Choropleth: U5MR in 2019
# ------------------------------------------------------------------
plot_global_health_map(
    df=u5mr_df,
    value_col="value",
    year=2019,
    title="Under-5 Mortality Rate",
    country_col="country_code",
    cmap="YlOrRd",
    legend_label="Deaths per 1,000 live births",
    output_file="u5mr_2019_map.png",
)
```

## Example 2: DALYs from NCDs by WHO Region with Age-Standardization

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Simulate GBD-style NCD DALY data by WHO region
# (Replace with download_gbd_data() call when credentials are available)
# ------------------------------------------------------------------

who_regions = {
    "AFR": "African Region",
    "AMR": "Region of the Americas",
    "SEAR": "South-East Asia Region",
    "EUR": "European Region",
    "EMR": "Eastern Mediterranean Region",
    "WPR": "Western Pacific Region",
}

ncd_causes = [
    "Cardiovascular diseases",
    "Diabetes mellitus",
    "Chronic respiratory diseases",
    "Neoplasms",
]

age_groups = WHO_WORLD_STANDARD_POPULATION["age_group"].tolist()
age_weights = WHO_WORLD_STANDARD_POPULATION["weight"].values

np.random.seed(2024)
records = []

for region_code, region_name in who_regions.items():
    for cause in ncd_causes:
        for year in [2000, 2010, 2019]:
            for age_group in age_groups:
                base_yll = np.random.exponential(200)
                base_yld = np.random.exponential(150)
                records.append({
                    "region_code": region_code,
                    "region_name": region_name,
                    "cause": cause,
                    "year": year,
                    "age_group": age_group,
                    "yll": base_yll,
                    "yld": base_yld,
                    "population": np.random.randint(50_000, 5_000_000),
                })

gbd_df = pd.DataFrame(records)

# ------------------------------------------------------------------
# Compute DALYs
# ------------------------------------------------------------------
daly_df = compute_dalys(
    yll_series=gbd_df["yll"],
    yld_series=gbd_df["yld"],
    index_cols=gbd_df[["region_code", "region_name", "cause", "year",
                        "age_group", "population"]],
)

daly_df["daly_rate"] = daly_df["daly"] / daly_df["population"]

# ------------------------------------------------------------------
# Age-standardize DALYs by region, cause, and year
# ------------------------------------------------------------------
asr_records = []

for (region_code, cause, year), group in daly_df.groupby(
    ["region_code", "cause", "year"]
):
    group = group.set_index("age_group")
    # Align age groups
    age_order = WHO_WORLD_STANDARD_POPULATION["age_group"].tolist()
    crude_rates = group.reindex(age_order)["daly_rate"].fillna(0)
    weights = pd.Series(age_weights, index=age_order)

    asr_result = age_standardize(
        crude_rates=crude_rates,
        age_weights=weights,
        per_unit=100_000,
    )
    asr_records.append({
        "region_code": region_code,
        "region_name": who_regions[region_code],
        "cause": cause,
        "year": year,
        **asr_result,
    })

asr_df = pd.DataFrame(asr_records)

print("Age-standardized DALY rates by region and cause (2019):")
print(
    asr_df[asr_df["year"] == 2019]
    .pivot_table(index="region_name", columns="cause",
                 values="age_standardized_rate")
    .round(1)
    .to_string()
)

# ------------------------------------------------------------------
# Inequality analysis: concentration index across regions
# ------------------------------------------------------------------
# Use GDP per capita proxy as wealth rank
gdp_rank = {
    "AFR": 1, "SEAR": 2, "EMR": 3, "AMR": 4, "WPR": 5, "EUR": 6
}
asr_df_2019 = asr_df[asr_df["year"] == 2019].copy()
asr_df_2019["wealth_rank"] = asr_df_2019["region_code"].map(gdp_rank)

for cause in ncd_causes:
    sub = asr_df_2019[asr_df_2019["cause"] == cause]
    if len(sub) >= 3:
        ci_result = compute_concentration_index(
            health_outcome=sub["age_standardized_rate"],
            wealth_rank=sub["wealth_rank"],
        )
        direction = "pro-rich" if ci_result["concentration_index"] > 0 else "pro-poor"
        print(f"\n{cause}: CI={ci_result['concentration_index']:.3f} ({direction})")

# ------------------------------------------------------------------
# Visualization: grouped bar chart by WHO region for 2019
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

pivot_2019 = asr_df_2019.pivot_table(
    index="region_code",
    columns="cause",
    values="age_standardized_rate",
)
pivot_2019.plot(
    kind="bar",
    ax=axes[0],
    colormap="tab10",
    edgecolor="white",
    alpha=0.85,
)
axes[0].set_xlabel("WHO Region")
axes[0].set_ylabel("Age-Standardized DALY Rate (per 100k)")
axes[0].set_title("NCD Burden by WHO Region (2019)\nAge-Standardized DALY Rates")
axes[0].legend(title="Cause", bbox_to_anchor=(1.0, 1.0), fontsize=8)
axes[0].tick_params(axis="x", rotation=30)
axes[0].grid(axis="y", alpha=0.3)

# Trend lines: total NCD DALYs over time
trend = asr_df.groupby(["region_code", "year"])["age_standardized_rate"].sum().reset_index()
for region in trend["region_code"].unique():
    sub = trend[trend["region_code"] == region]
    axes[1].plot(
        sub["year"], sub["age_standardized_rate"],
        marker="o", linewidth=2, label=region,
    )
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Total Age-Standardized DALY Rate (per 100k)")
axes[1].set_title("Total NCD DALY Trend by WHO Region\n(2000–2019)")
axes[1].legend(title="Region", fontsize=9)
axes[1].grid(alpha=0.3)

plt.suptitle("Global NCD Burden Analysis — IHME GBD + Age-Standardization", fontsize=13)
plt.tight_layout()
plt.savefig("ncd_daly_burden.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Notes and Best Practices

- **WHO GHO API**: Most indicators are publicly accessible without authentication. Pass `WHO_API_KEY` env var for rate-limit exemptions. Browse indicator codes at `https://ghoapi.azureedge.net/api/Indicator`.
- **GBD data**: For large-scale GBD downloads, use the IHME GHD Results Tool at `https://ghdx.healthdata.org/gbd-results`. Save the CSV and pass via `fallback_csv_path`.
- **Our World in Data**: OWID health datasets can be fetched directly as CSVs, e.g., `pd.read_csv("https://ourworldindata.org/grapher/child-mortality.csv?tab=table")`.
- **Age standardization**: Always report the reference population used (WHO World Standard 2000-2025 vs. Segi world population). Results are not comparable across different standards.
- **Concentration index**: Interpret with caution for non-continuous wealth measures. Wagstaff's normalized CI is preferred for binary outcomes.
- **Mapping**: `geopandas.datasets.get_path('naturalearth_lowres')` is deprecated in newer geopandas. Use `geodatasets.get_path('naturalearth.land')` or download from Natural Earth directly for production code.
- **Causality**: Cross-country observational health data cannot establish causal effects. Use DAG-based confounder control or difference-in-differences designs for causal inference.
- **DALY weights**: Disability weights for YLD calculations should come from the GBD study directly; do not use outdated WHO 1990 weights.
