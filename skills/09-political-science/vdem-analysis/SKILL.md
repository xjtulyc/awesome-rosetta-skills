---
name: vdem-analysis
description: >
  Analyze V-Dem (Varieties of Democracy) cross-national panel data to measure democratic indices,
  detect backsliding, and run panel regressions linking institutional quality to economic outcomes.
tags:
  - political-science
  - democracy
  - panel-data
  - v-dem
  - comparative-politics
  - regression
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
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - linearmodels>=5.3.0
  - scipy>=1.10.0
  - statsmodels>=0.14.0
  - requests>=2.31.0
last_updated: "2026-03-17"
---

# V-Dem Analysis — Varieties of Democracy Dataset

## Overview

The **Varieties of Democracy (V-Dem)** project provides the most comprehensive cross-national
dataset on democratic institutions, covering over 200 countries from 1789 to the present. This
skill covers the full analytical pipeline: loading the large CSV dataset efficiently, computing
democratic indices, detecting backsliding episodes, visualizing trends, and running panel
regressions.

### Key Democratic Indices

| Column | Description |
|---|---|
| `v2x_polyarchy` | Electoral democracy index |
| `v2x_libdem` | Liberal democracy index |
| `v2x_partipdem` | Participatory democracy index |
| `v2x_delibdem` | Deliberative democracy index |
| `v2x_egaldem` | Egalitarian democracy index |

All indices are on a 0–1 scale where higher values indicate more democracy.

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn linearmodels scipy statsmodels requests
```

Download the V-Dem dataset from https://www.v-dem.net/data/the-v-dem-dataset/ (V-Dem Country-Year
Core dataset, CSV format, ~200 MB). Store the path as an environment variable or pass it directly.

---

## Core Functions

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

VDEM_INDICES = [
    "v2x_polyarchy",
    "v2x_libdem",
    "v2x_partipdem",
    "v2x_delibdem",
    "v2x_egaldem",
]

ALWAYS_KEEP = [
    "country_name",
    "country_text_id",
    "country_id",
    "year",
    "e_gdppc",          # GDP per capita (Maddison)
    "e_pop",            # Population
    "e_regionpol_6C",   # Region (6 categories)
]

REGION_LABELS = {
    1: "Eastern Europe & Central Asia",
    2: "Latin America & Caribbean",
    3: "Middle East & North Africa",
    4: "Sub-Saharan Africa",
    5: "Western Europe & North America",
    6: "Asia & Pacific",
}


def load_vdem(
    filepath: str,
    years: tuple[int, int] | None = None,
    countries: list[str] | None = None,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the V-Dem CSV efficiently using chunked reading and column pre-selection.

    Parameters
    ----------
    filepath : str
        Path to the V-Dem CSV file (e.g. ``V-Dem-CY-Full+Others-v13.csv``).
    years : tuple (start, end), optional
        Inclusive year range filter, e.g. ``(2000, 2023)``.
    countries : list of str, optional
        Filter by ``country_text_id`` (ISO 3-letter codes) or ``country_name``.
    extra_cols : list of str, optional
        Additional V-Dem columns to retain beyond the defaults.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe indexed by (country_name, year).
    """
    keep_cols = ALWAYS_KEEP + VDEM_INDICES + (extra_cols or [])

    # Read header to find available columns
    header = pd.read_csv(filepath, nrows=0)
    available = [c for c in keep_cols if c in header.columns]

    # Chunked read for large files
    chunks = []
    for chunk in pd.read_csv(filepath, usecols=available, chunksize=50_000, low_memory=False):
        if years:
            chunk = chunk[(chunk["year"] >= years[0]) & (chunk["year"] <= years[1])]
        if countries:
            mask = chunk["country_text_id"].isin(countries) | chunk["country_name"].isin(countries)
            chunk = chunk[mask]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df["region_label"] = df["e_regionpol_6C"].map(REGION_LABELS)
    df = df.sort_values(["country_name", "year"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Democratic Backsliding Detection
# ---------------------------------------------------------------------------


def compute_backsliding_episodes(
    df: pd.DataFrame,
    index_col: str = "v2x_libdem",
    threshold: float = 0.10,
    window: int = 5,
) -> pd.DataFrame:
    """
    Identify country-years where a democratic index declined by ``threshold``
    or more over the preceding ``window`` years.

    Parameters
    ----------
    df : pd.DataFrame
        V-Dem dataframe (output of ``load_vdem``).
    index_col : str
        Which democracy index to track.
    threshold : float
        Minimum absolute decline to flag as backsliding (default 0.10).
    window : int
        Number of years over which to compute the decline (default 5).

    Returns
    -------
    pd.DataFrame
        Rows flagged as backsliding episodes, sorted by severity.
    """
    df = df.sort_values(["country_name", "year"]).copy()
    df["index_lag"] = df.groupby("country_name")[index_col].shift(window)
    df["delta"] = df[index_col] - df["index_lag"]
    episodes = df[df["delta"] <= -threshold].copy()
    episodes["severity"] = episodes["delta"].abs()
    return (
        episodes[["country_name", "year", index_col, "index_lag", "delta", "severity"]]
        .sort_values("severity", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------------------------


def plot_democracy_trends(
    df: pd.DataFrame,
    countries: list[str],
    index_col: str = "v2x_libdem",
    title: str | None = None,
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot time-series democracy trends for a list of countries.

    Parameters
    ----------
    df : pd.DataFrame
        V-Dem dataframe.
    countries : list of str
        Country names to plot.
    index_col : str
        Democracy index column to plot.
    title : str, optional
        Plot title.
    figsize : tuple
        Matplotlib figure size.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    subset = df[df["country_name"].isin(countries)].copy()
    fig, ax = plt.subplots(figsize=figsize)

    palette = cm.get_cmap("tab10", len(countries))
    for i, country in enumerate(countries):
        cdata = subset[subset["country_name"] == country].sort_values("year")
        ax.plot(cdata["year"], cdata[index_col], label=country, color=palette(i), linewidth=2)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(index_col, fontsize=12)
    ax.set_title(title or f"Democracy Trends: {index_col}", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_regional_comparison(
    df: pd.DataFrame,
    year: int,
    index_col: str = "v2x_polyarchy",
    figsize: tuple = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Box plot of a democracy index by region for a given year.
    """
    subset = df[df["year"] == year].dropna(subset=[index_col, "region_label"])
    fig, ax = plt.subplots(figsize=figsize)
    order = (
        subset.groupby("region_label")[index_col]
        .median()
        .sort_values()
        .index.tolist()
    )
    sns.boxplot(
        data=subset,
        x=index_col,
        y="region_label",
        order=order,
        palette="coolwarm",
        ax=ax,
    )
    ax.set_title(f"Regional Distribution of {index_col} ({year})", fontsize=13)
    ax.set_xlabel(index_col)
    ax.set_ylabel("")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 4. Panel Regression
# ---------------------------------------------------------------------------


def run_panel_regression(
    df: pd.DataFrame,
    outcome: str = "v2x_libdem",
    treatment: str = "e_gdppc",
    controls: list[str] | None = None,
    model_type: str = "fe",
    log_transform_treatment: bool = True,
) -> dict:
    """
    Run a panel regression (Fixed Effects, Random Effects, or Pooled OLS).

    Parameters
    ----------
    df : pd.DataFrame
        V-Dem dataframe with country_name and year columns.
    outcome : str
        Dependent variable column name.
    treatment : str
        Main independent variable.
    controls : list of str, optional
        Additional control variable columns.
    model_type : str
        One of ``"fe"`` (fixed effects), ``"re"`` (random effects), ``"pooled"``.
    log_transform_treatment : bool
        If True, log-transform the treatment variable (useful for GDP per capita).

    Returns
    -------
    dict with keys ``"summary"``, ``"model"``, ``"rsquared"``.
    """
    controls = controls or []
    req_cols = [outcome, treatment] + controls + ["country_name", "year"]
    panel_df = df[req_cols].dropna().copy()

    if log_transform_treatment and panel_df[treatment].gt(0).all():
        panel_df[treatment] = np.log(panel_df[treatment])
        panel_df = panel_df.rename(columns={treatment: f"log_{treatment}"})
        treatment = f"log_{treatment}"

    panel_df = panel_df.set_index(["country_name", "year"])
    exog_vars = [treatment] + controls
    exog = panel_df[exog_vars]
    endog = panel_df[outcome]

    if model_type == "fe":
        model = PanelOLS(endog, exog, entity_effects=True, time_effects=False)
    elif model_type == "re":
        model = RandomEffects(endog, exog)
    else:
        model = PooledOLS(endog, exog)

    result = model.fit(cov_type="clustered", cluster_entity=True)
    return {
        "summary": result.summary,
        "model": result,
        "rsquared": result.rsquared,
        "params": result.params,
        "pvalues": result.pvalues,
    }


# ---------------------------------------------------------------------------
# 5. Index Correlation Analysis
# ---------------------------------------------------------------------------


def compare_indices_correlation(
    df: pd.DataFrame,
    indices: list[str] | None = None,
    year: int | None = None,
    method: str = "pearson",
    figsize: tuple = (8, 7),
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Compute and visualize pairwise correlation between democracy indices.

    Parameters
    ----------
    df : pd.DataFrame
        V-Dem dataframe.
    indices : list of str, optional
        Columns to correlate. Defaults to all five main V-Dem indices.
    year : int, optional
        Subset to a specific year. If None, uses all years.
    method : str
        Correlation method: ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    figsize : tuple
        Figure size for the heatmap.
    save_path : str, optional
        Save figure if provided.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    indices = indices or VDEM_INDICES
    subset = df if year is None else df[df["year"] == year]
    corr_matrix = subset[indices].corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    year_label = str(year) if year else "All Years"
    ax.set_title(f"V-Dem Index Correlations ({method.capitalize()}, {year_label})", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
    return corr_matrix
```

---

## Example A: Democratic Backsliding in Eastern Europe (2000–2023)

This example tracks liberal democracy scores for Hungary, Poland, and Turkey, detects backsliding
episodes, and produces a publication-ready trend chart.

```python
# ── Example A ──────────────────────────────────────────────────────────────
# Requires: V-Dem CSV path set via environment variable VDEM_CSV_PATH

VDEM_PATH = os.environ.get("VDEM_CSV_PATH", "V-Dem-CY-Full+Others-v13.csv")

EASTERN_EUROPE = [
    "Hungary",
    "Poland",
    "Turkey",
    "Czech Republic",
    "Romania",
    "Serbia",
    "Bulgaria",
]

# --- Step 1: Load data -------------------------------------------------------
df = load_vdem(
    filepath=VDEM_PATH,
    years=(2000, 2023),
    countries=EASTERN_EUROPE,
)
print(f"Loaded {len(df)} country-year rows for {df['country_name'].nunique()} countries.")

# --- Step 2: Detect backsliding ----------------------------------------------
episodes = compute_backsliding_episodes(
    df,
    index_col="v2x_libdem",
    threshold=0.10,
    window=5,
)
print("\nTop 10 backsliding episodes (liberal democracy, ≥0.10 decline over 5 years):")
print(episodes.head(10).to_string(index=False))

# --- Step 3: Plot trends -----------------------------------------------------
focus = ["Hungary", "Poland", "Turkey", "Romania"]
fig = plot_democracy_trends(
    df,
    countries=focus,
    index_col="v2x_libdem",
    title="Liberal Democracy Index — Eastern Europe (2000–2023)",
    save_path="eastern_europe_libdem.png",
)
plt.show()

# --- Step 4: Summary statistics by country -----------------------------------
summary = (
    df.groupby("country_name")["v2x_libdem"]
    .agg(["mean", "std", "min", "max"])
    .round(3)
    .sort_values("mean")
)
print("\nLiberal Democracy Summary (2000–2023):")
print(summary.to_string())

# --- Step 5: Regional box plot for 2022 -------------------------------------
df_all = load_vdem(filepath=VDEM_PATH, years=(2022, 2022))
fig2 = plot_regional_comparison(
    df_all,
    year=2022,
    index_col="v2x_polyarchy",
    save_path="regional_democracy_2022.png",
)
plt.show()

# --- Step 6: Annotate backsliding on trend chart ----------------------------
backsliders = episodes[episodes["country_name"].isin(focus)]
print("\nBacksliding episodes in focus countries:")
print(backsliders[["country_name", "year", "delta"]].to_string(index=False))
```

---

## Example B: Does GDP Predict Democracy? Cross-National Panel Regression

This example runs a country fixed-effects panel regression asking whether higher GDP per capita
(Maddison project) is associated with higher liberal democracy scores.

```python
# ── Example B ──────────────────────────────────────────────────────────────

# --- Load broad panel (1950–2020) -------------------------------------------
df_panel = load_vdem(
    filepath=VDEM_PATH,
    years=(1950, 2020),
    extra_cols=["e_pop"],
)

# Drop country-years with missing GDP or democracy score
df_panel = df_panel.dropna(subset=["v2x_libdem", "e_gdppc"])
print(f"Panel size: {len(df_panel)} obs, {df_panel['country_name'].nunique()} countries")

# --- Descriptive statistics --------------------------------------------------
desc = df_panel[["v2x_libdem", "v2x_polyarchy", "e_gdppc"]].describe().round(3)
print("\nDescriptive Statistics:")
print(desc.to_string())

# --- Fixed effects regression: GDP → Liberal Democracy ----------------------
result_fe = run_panel_regression(
    df_panel,
    outcome="v2x_libdem",
    treatment="e_gdppc",
    controls=[],
    model_type="fe",
    log_transform_treatment=True,
)
print("\n=== Fixed Effects: log(GDP per capita) → Liberal Democracy ===")
print(result_fe["summary"])
print(f"Within R²: {result_fe['rsquared']:.4f}")

# --- Pooled OLS for comparison -----------------------------------------------
result_ols = run_panel_regression(
    df_panel,
    outcome="v2x_libdem",
    treatment="e_gdppc",
    model_type="pooled",
    log_transform_treatment=True,
)
print("\n=== Pooled OLS: log(GDP per capita) → Liberal Democracy ===")
print(result_ols["summary"])

# --- Scatter: GDP vs Democracy (cross-sectional, latest year) ----------------
df_2020 = df_panel[df_panel["year"] == 2020].dropna(subset=["e_gdppc", "v2x_libdem"])
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    np.log(df_2020["e_gdppc"]),
    df_2020["v2x_libdem"],
    c=df_2020["e_regionpol_6C"],
    cmap="tab10",
    alpha=0.7,
    s=60,
    edgecolors="white",
    linewidth=0.4,
)

# Add regression line
x = np.log(df_2020["e_gdppc"])
y = df_2020["v2x_libdem"]
slope, intercept, r, p, se = stats.linregress(x.dropna(), y[x.notna()])
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, intercept + slope * x_line, "r--", linewidth=1.5, label=f"OLS (r={r:.2f})")

ax.set_xlabel("log(GDP per capita)", fontsize=12)
ax.set_ylabel("Liberal Democracy Index", fontsize=12)
ax.set_title("GDP per Capita vs. Liberal Democracy (2020)", fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gdp_democracy_scatter_2020.png", dpi=150)
plt.show()

# --- Compare all five V-Dem indices: correlation heatmap --------------------
corr = compare_indices_correlation(
    df_panel,
    method="spearman",
    save_path="vdem_index_correlations.png",
)
print("\nSpearman Correlations among V-Dem Indices:")
print(corr.round(3).to_string())
```

---

## Notes and Caveats

### Data Access

- The V-Dem dataset requires registration at https://www.v-dem.net/data/the-v-dem-dataset/
- The full dataset is ~200 MB; the `load_vdem` function handles memory efficiently via chunking.
- Column availability varies by version; always check `header.columns` for your version.

### Interpretation

- **Fixed effects** absorb all time-invariant country characteristics (geography, culture, history).
  The FE coefficient on log-GDP therefore captures *within-country* change, not cross-national
  variation.
- **Backsliding threshold of 0.10** over 5 years is a common rule-of-thumb in the literature
  (Lührmann & Lindberg 2019). Adjust based on your theoretical expectations.
- V-Dem scores carry **measurement uncertainty**; the full dataset includes confidence intervals
  for each estimate (columns ending in `_codelow`, `_codehigh`).

### Freedom House Integration

If you want to integrate Freedom House data (https://freedomhouse.org/reports/publication-archives):

```python
import pandas as pd

def load_freedom_house(csv_path: str, year_col: str = "Edition") -> pd.DataFrame:
    """Load and standardize Freedom House country scores."""
    fh = pd.read_csv(csv_path)
    # Typical columns: Country/Territory, Edition, PR, CL, Status, Total
    fh = fh.rename(columns={
        "Country/Territory": "country_name",
        year_col: "year",
        "Total": "fh_total",
        "Status": "fh_status",
    })
    fh["fh_total_norm"] = (100 - fh["fh_total"]) / 100  # Invert so higher = more free
    return fh[["country_name", "year", "fh_total", "fh_total_norm", "fh_status"]]


def merge_vdem_fh(vdem_df: pd.DataFrame, fh_df: pd.DataFrame) -> pd.DataFrame:
    """Merge V-Dem and Freedom House on country-year."""
    merged = vdem_df.merge(fh_df, on=["country_name", "year"], how="left")
    # Validate: correlation should be ~0.85+
    valid = merged.dropna(subset=["v2x_libdem", "fh_total_norm"])
    corr = valid["v2x_libdem"].corr(valid["fh_total_norm"])
    print(f"V-Dem libdem × FH normalized correlation: {corr:.3f}")
    return merged
```

### Common Issues

| Problem | Solution |
|---|---|
| `MemoryError` on large CSV | Use `chunksize` in `load_vdem` (already implemented) |
| `KeyError` on index column | Check V-Dem version; column names changed in v13 |
| `linearmodels` entity effects warning | Ensure panel is balanced or use `drop_absorbed=True` |
| Missing GDP for many countries | `e_gdppc` is sparse before 1950; try `e_migdppc` |

### References

- Coppedge et al. (2023). *V-Dem Codebook v13*. Varieties of Democracy Project.
- Lührmann, A., & Lindberg, S. I. (2019). A third wave of autocratization is here: What is new
  about it? *Democratization*, 26(7), 1095–1113.
- Pemstein et al. (2023). *The V-Dem Measurement Model*. V-Dem Working Paper 21.
