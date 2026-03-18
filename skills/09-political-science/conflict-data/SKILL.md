---
name: conflict-data
description: >
  Use this Skill for quantitative conflict research: ACLED and UCDP data download,
  spatio-temporal clustering (DBSCAN), conflict onset logit panel, and fatality estimation.
tags:
  - political-science
  - conflict
  - ACLED
  - UCDP
  - spatial-analysis
  - political-violence
version: "1.0.0"
authors:
  - name: "awesome-rosetta-skills contributors"
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - pandas>=1.5
  - geopandas>=0.13
  - requests>=2.28
  - sklearn>=1.2
  - numpy>=1.23
  - matplotlib>=3.6
last_updated: "2026-03-18"
status: "stable"
---

# Conflict Data Analysis

## When to Use

Use this skill when you need to:

- Download event-level political violence data from the ACLED API (Armed Conflict Location and
  Event Data Project) or UCDP GED (Uppsala Conflict Data Program — Georeferenced Event Dataset)
- Aggregate event data to monthly time series, country-year panels, or spatial grids
- Identify conflict hotspots using DBSCAN spatial clustering with geographic distance metrics
- Model conflict onset as a function of economic, demographic, and political risk factors in a
  panel logit framework with country fixed effects
- Analyze conflict diffusion — whether violence in neighboring countries/cells increases local risk
- Work with UCDP battle death uncertainty ranges (best/low/high estimates)

This skill does not cover causal inference for conflict (natural experiments, instrumental
variables) — see the `ir-gravity-model` skill for dyadic trade-conflict links.

## Background

**ACLED** provides near-real-time event data since 1997 (Africa) and 2016+ (global) covering
battles, explosions, protests, riots, violence against civilians (VAC), and strategic developments.
Each event has coordinates, date, actor names, fatalities (best estimate), and source notes.

**UCDP GED** covers organized violence (state-based, non-state, one-sided violence) globally
since 1989. Key advantage: fatality uncertainty ranges (low/best/high). Download from Zenodo
(https://zenodo.org/record/xxxxxxxxx) or UCDP API.

**Event type distributions** vary by region. In Sub-Saharan Africa, VAC is common; in the Middle
East, battles dominate. Normalizing counts by population or area produces comparable rates.

**DBSCAN hotspot detection**: Using geographic coordinates (lat/lon converted to km via Haversine
distance), DBSCAN identifies dense clusters without requiring a pre-specified number of clusters.
Parameters: `eps` (neighborhood radius in km), `min_samples` (minimum events for a core point).
Events labeled -1 are noise (isolated events).

**Conflict onset logit**: The dependent variable is a binary indicator: first year of conflict
after a period of peace. Key predictors (Fearon & Laitin 2003, Collier & Hoeffler 2004):
- Lag GDP per capita (proxy for state capacity and opportunity cost)
- Population (log): larger countries have higher absolute risk
- Oil/primary commodity exports (resource curse hypothesis)
- Past conflict indicator (recurrence is common)
- Ethnic or religious fractionalization

**Conflict diffusion**: Spatial lag of conflict in neighboring countries/grid cells tests whether
violence is contagious. Typically computed as a binary indicator of neighboring conflict or a
distance-weighted sum of neighbor conflict intensity.

## Environment Setup

```bash
pip install pandas>=1.5 geopandas>=0.13 requests>=2.28 scikit-learn>=1.2 numpy>=1.23 matplotlib>=3.6
```

Register for a free ACLED API key at https://developer.acleddata.com/. Store credentials in
environment variables — never hardcode them:

```bash
export ACLED_EMAIL="your_email@example.com"
export ACLED_KEY="your_api_key_here"
export UCDP_GED_PATH="/data/GEDEvent_v23_1.csv"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. ACLED API Download
# ---------------------------------------------------------------------------

ACLED_BASE_URL = "https://api.acleddata.com/acled/read"

ACLED_EVENT_TYPES = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
    "Protests",
    "Riots",
    "Strategic developments",
]


def download_acled(
    country: str,
    start_date: str,
    end_date: str,
    event_types: list[str] | None = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """
    Download events from the ACLED API.

    Reads credentials from environment variables ACLED_EMAIL and ACLED_KEY.
    No API keys are stored in code.

    Parameters
    ----------
    country : str
        Country name as recognized by ACLED (e.g., 'Nigeria', 'Ukraine').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    event_types : list of str, optional
        Filter by ACLED event type categories.
    page_size : int
        Records per API page (max 500).

    Returns
    -------
    pd.DataFrame with all events in the date range.
    """
    email = os.environ.get("ACLED_EMAIL")
    key = os.environ.get("ACLED_KEY")
    if not email or not key:
        raise EnvironmentError(
            "Set ACLED_EMAIL and ACLED_KEY environment variables before calling this function."
        )

    all_records = []
    page = 1
    while True:
        params = {
            "email": email,
            "key": key,
            "country": country,
            "event_date": f"{start_date}|{end_date}",
            "event_date_where": "BETWEEN",
            "limit": page_size,
            "page": page,
            "fields": "event_id_cnty|event_date|event_type|sub_event_type|"
                      "actor1|actor2|country|admin1|admin2|location|"
                      "latitude|longitude|fatalities|year",
        }
        if event_types:
            params["event_type"] = "|".join(event_types)
            params["event_type_where"] = "IN"

        resp = requests.get(ACLED_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            break
        all_records.extend(data)
        if len(data) < page_size:
            break
        page += 1

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df.sort_values("event_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Temporal Aggregation and Trend Plot
# ---------------------------------------------------------------------------

def monthly_conflict_trend(
    df: pd.DataFrame,
    event_type_col: str = "event_type",
    date_col: str = "event_date",
    fatality_col: str = "fatalities",
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate events and fatalities by month and event type.

    Returns
    -------
    pd.DataFrame with columns: year_month, event_type, n_events, fatalities.
    """
    df = df.copy()
    df["year_month"] = df[date_col].dt.to_period("M")
    agg = (
        df.groupby(["year_month", event_type_col])
        .agg(n_events=(event_type_col, "count"), fatalities=(fatality_col, "sum"))
        .reset_index()
    )
    agg["year_month_dt"] = agg["year_month"].dt.to_timestamp()

    if save_path:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for etype, grp in agg.groupby(event_type_col):
            grp = grp.sort_values("year_month_dt")
            axes[0].plot(grp["year_month_dt"], grp["n_events"], label=etype, linewidth=1.5)
            axes[1].plot(grp["year_month_dt"], grp["fatalities"], label=etype, linewidth=1.5)
        axes[0].set_ylabel("Events per Month")
        axes[0].set_title("Monthly Conflict Events by Type")
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_ylabel("Fatalities per Month")
        axes[1].set_title("Monthly Fatalities by Event Type")
        axes[1].legend(fontsize=7, ncol=2)
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.show()
    return agg


# ---------------------------------------------------------------------------
# 3. DBSCAN Conflict Hotspot Detection
# ---------------------------------------------------------------------------

def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Haversine distances (km) for an (N,2) array of [lat, lon].

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Latitude and longitude in degrees.

    Returns
    -------
    np.ndarray, shape (N, N)  — distance matrix in km.
    """
    R = 6371.0
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def detect_conflict_hotspots(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    eps_km: float = 50.0,
    min_samples: int = 5,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Identify conflict hotspot clusters using DBSCAN with Haversine distance.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level data with lat/lon columns.
    lat_col, lon_col : str
    eps_km : float
        Neighborhood radius in kilometers (default: 50 km).
    min_samples : int
        Minimum events to form a core point.
    save_path : str, optional
        Map save path.

    Returns
    -------
    pd.DataFrame with cluster_id column added; -1 = noise.
    """
    coords = df[[lat_col, lon_col]].dropna().values
    if len(coords) < min_samples:
        df["cluster_id"] = -1
        return df

    dist_matrix = haversine_matrix(coords)

    db = DBSCAN(eps=eps_km, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(dist_matrix)

    result = df.copy()
    valid_idx = df[[lat_col, lon_col]].dropna().index
    result.loc[valid_idx, "cluster_id"] = labels
    result["cluster_id"] = result["cluster_id"].fillna(-1).astype(int)

    n_clusters = (labels >= 0).sum()
    n_noise = (labels == -1).sum()
    print(f"DBSCAN: {len(set(labels)) - 1} clusters, {n_noise} noise events ({n_noise/len(labels):.1%})")

    if save_path:
        fig, ax = plt.subplots(figsize=(10, 8))
        noise_mask = result["cluster_id"] == -1
        ax.scatter(
            result.loc[noise_mask, lon_col],
            result.loc[noise_mask, lat_col],
            c="lightgray", s=5, alpha=0.5, label="Noise"
        )
        cluster_ids = sorted([c for c in result["cluster_id"].unique() if c >= 0])
        cmap = cm.get_cmap("tab20", max(len(cluster_ids), 1))
        for i, cid in enumerate(cluster_ids):
            mask = result["cluster_id"] == cid
            ax.scatter(
                result.loc[mask, lon_col],
                result.loc[mask, lat_col],
                c=[cmap(i)], s=20, alpha=0.8,
                label=f"Cluster {cid} (n={mask.sum()})"
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Conflict Hotspots (DBSCAN eps={eps_km}km, min={min_samples})")
        if len(cluster_ids) <= 15:
            ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.show()
    return result


# ---------------------------------------------------------------------------
# 4. Country-Year Panel Aggregation
# ---------------------------------------------------------------------------

def build_conflict_panel(
    df: pd.DataFrame,
    date_col: str = "event_date",
    country_col: str = "country",
    fatality_col: str = "fatalities",
) -> pd.DataFrame:
    """
    Aggregate event data to country-year panel.

    Returns
    -------
    pd.DataFrame with columns: country, year, n_events, fatalities, months_active.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    df["month"] = pd.to_datetime(df[date_col]).dt.month

    panel = df.groupby([country_col, "year"]).agg(
        n_events=(fatality_col, "count"),
        fatalities=(fatality_col, "sum"),
        months_active=("month", "nunique"),
    ).reset_index()
    panel = panel.rename(columns={country_col: "country"})
    return panel


# ---------------------------------------------------------------------------
# 5. Conflict Onset Logit
# ---------------------------------------------------------------------------

def conflict_onset_logit(
    panel: pd.DataFrame,
    conflict_col: str,
    predictors: list[str],
    country_col: str = "country",
    year_col: str = "year",
) -> object:
    """
    Logit model for conflict onset with lagged predictors and country FE via dummies.

    Parameters
    ----------
    panel : pd.DataFrame
        Country-year panel with conflict binary variable and predictors.
    conflict_col : str
        Binary conflict indicator.
    predictors : list of str
        Continuous/binary predictors (already lagged if desired).
    country_col : str
    year_col : str

    Returns
    -------
    statsmodels LogitResults
    """
    import statsmodels.api as sm

    # Create onset: first year of conflict after peace
    panel = panel.sort_values([country_col, year_col]).copy()
    panel["conflict_lag"] = panel.groupby(country_col)[conflict_col].shift(1)
    panel["onset"] = ((panel[conflict_col] == 1) & (panel["conflict_lag"] == 0)).astype(int)

    # Only include country-years at risk (no ongoing conflict)
    at_risk = panel[panel["conflict_lag"] == 0].copy()

    sub = at_risk[[country_col, year_col, "onset"] + predictors].dropna()

    # Country dummies (fixed effects via dummies; drop one for identification)
    dummies = pd.get_dummies(sub[country_col], drop_first=True, prefix="cty").astype(float)
    X = pd.concat([sub[predictors].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X)
    y = sub["onset"].reset_index(drop=True)

    model = sm.Logit(y, X)
    result = model.fit(method="bfgs", maxiter=200, disp=False)
    return result
```

## Advanced Usage

### UCDP GED Fatality Uncertainty Analysis

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_ucdp_ged(path: str, countries: list[str] | None = None,
                  years: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    Load UCDP Georeferenced Event Dataset (GED) CSV.

    Key columns:
    - id, year, date_start, date_end
    - country, region, dyad_name, conflict_name
    - latitude, longitude
    - deaths_a (side A), deaths_b (side B), deaths_civilians, deaths_unknown
    - best (best fatality estimate), low, high
    - type_of_violence: 1=state-based, 2=non-state, 3=one-sided

    Parameters
    ----------
    path : str
        Path to GED CSV file.
    countries : list of str, optional
    years : tuple, optional

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path, low_memory=False)
    df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    if countries:
        df = df[df["country"].isin(countries)]
    if years:
        df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]
    return df.reset_index(drop=True)


def ucdp_uncertainty_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annual fatality uncertainty summary: best/low/high by year.

    Returns
    -------
    pd.DataFrame with year, best_deaths, low_deaths, high_deaths.
    """
    agg = df.groupby("year").agg(
        best_deaths=("best", "sum"),
        low_deaths=("low", "sum"),
        high_deaths=("high", "sum"),
        n_events=("id", "count"),
    ).reset_index()
    return agg


def plot_ucdp_uncertainty(summary: pd.DataFrame, country: str | None = None,
                           save_path: str | None = None):
    """Plot annual fatalities with uncertainty band (low-high range)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(summary["year"], summary["low_deaths"], summary["high_deaths"],
                    alpha=0.3, color="#d73027", label="Low-High Range")
    ax.plot(summary["year"], summary["best_deaths"], color="#d73027",
            linewidth=2, label="Best Estimate")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fatalities")
    title = f"UCDP Battle Deaths{' — ' + country if country else ''}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| ACLED API returns 401 | Invalid credentials | Verify ACLED_EMAIL and ACLED_KEY env vars; key expires annually |
| DBSCAN produces one giant cluster | `eps` too large | Reduce `eps_km` to 20-30 km; check for geocoding errors (events at (0,0)) |
| Conflict onset logit does not converge | Separation by country FE | Use `method='nm'` or remove countries with zero/all-one outcomes |
| UCDP GED missing 2022-2023 | Recent years on Zenodo with delay | Check UCDP API at https://ucdpapi.pcr.uu.se/ for recent data |
| Memory error on GED | Full GED is ~300k rows | Load with `usecols` to retain only needed columns |
| Haversine matrix too slow | O(N²) for large N | Subsample or use `sklearn BallTree` with haversine metric |

## External Resources

- ACLED API Documentation: https://developer.acleddata.com/
- UCDP GED Codebook: https://ucdp.uu.se/downloads/
- Fearon, J. & Laitin, D. (2003). Ethnicity, insurgency, and civil war. *APSR*, 97(1), 75-90.
- Collier, P. & Hoeffler, A. (2004). Greed and grievance in civil war. *Oxford Econ. Papers*, 56(4), 563-595.
- Raleigh, C. et al. (2010). Introducing ACLED. *Journal of Peace Research*, 47(5), 651-660.
- Sundberg, R. & Melander, E. (2013). Introducing the UCDP Georeferenced Event Dataset. *Journal of Peace Research*, 50(4), 523-532.

## Examples

### Example 1: ACLED Monthly Trend Plot (Simulated)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n_events = 2000
dates = pd.date_range("2018-01-01", "2023-12-31", periods=n_events)
event_types_list = ["Battles", "Violence against civilians", "Protests",
                    "Explosions/Remote violence", "Riots"]
df_acled_sim = pd.DataFrame({
    "event_date": dates,
    "event_type": rng.choice(event_types_list, n_events, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    "fatalities": rng.negative_binomial(1, 0.4, n_events),
    "latitude": rng.uniform(4.0, 14.0, n_events),
    "longitude": rng.uniform(3.0, 15.0, n_events),
    "country": "Nigeria",
})

monthly = monthly_conflict_trend(df_acled_sim, save_path="monthly_conflict_trend.png")
print("Monthly aggregation (first 6 rows):")
print(monthly.head(6).to_string(index=False))
```

### Example 2: DBSCAN Hotspot Map

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n_events = 1500
# Two hotspot regions + scattered events
lats = np.concatenate([rng.normal(10.5, 0.3, 400), rng.normal(6.5, 0.4, 350),
                        rng.uniform(4, 14, 750)])
lons = np.concatenate([rng.normal(13.2, 0.3, 400), rng.normal(5.5, 0.5, 350),
                        rng.uniform(3, 15, 750)])
df_geo = pd.DataFrame({
    "latitude": lats,
    "longitude": lons,
    "event_type": rng.choice(["Battles", "VAC", "Protests"], n_events),
    "fatalities": rng.negative_binomial(2, 0.5, n_events),
})

df_clustered = detect_conflict_hotspots(
    df_geo, eps_km=40, min_samples=10, save_path="conflict_hotspots.png"
)
print("\nCluster distribution:")
print(df_clustered["cluster_id"].value_counts().head(10))
```

### Example 3: Conflict Onset Logit Panel

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
countries_panel = [f"Country_{i}" for i in range(1, 31)]
years_panel = list(range(2000, 2023))
records = []
for ctr in countries_panel:
    base_risk = rng.uniform(-2, 0.5)  # country-level baseline
    for yr in years_panel:
        gdppc = rng.lognormal(8, 1.2)
        pop_log = rng.uniform(14, 20)
        oil = rng.binomial(1, 0.2)
        conflict = int(rng.random() < 1 / (1 + np.exp(-(base_risk - 0.3 * np.log(gdppc) + 0.4 * oil))))
        records.append({
            "country": ctr, "year": yr,
            "conflict": conflict,
            "gdppc_log": np.log(gdppc),
            "pop_log": pop_log,
            "oil": oil,
        })

df_panel = pd.DataFrame(records)
result_logit = conflict_onset_logit(
    df_panel,
    conflict_col="conflict",
    predictors=["gdppc_log", "pop_log", "oil"],
)
print("=== Conflict Onset Logit (with Country FE) ===")
# Print only substantive predictors (not country dummies)
sub_params = result_logit.params[["const", "gdppc_log", "pop_log", "oil"]]
sub_pvals = result_logit.pvalues[["const", "gdppc_log", "pop_log", "oil"]]
print(pd.DataFrame({"coef": sub_params.round(4), "p": sub_pvals.round(4)}))
print(f"\nPseudo R² (McFadden): {1 - result_logit.llf / result_logit.llnull:.4f}")
```
