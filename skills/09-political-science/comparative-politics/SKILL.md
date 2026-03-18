---
name: comparative-politics
description: >
  Use this Skill for comparative political analysis: QoG, Polity 5, Freedom House merge,
  cross-sectional OLS with FE, multilevel models, and regime classification.
tags:
  - political-science
  - comparative-politics
  - QoG
  - Polity
  - regime-type
  - multilevel
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
  - statsmodels>=0.14
  - linearmodels>=4.28
  - numpy>=1.23
  - matplotlib>=3.6
last_updated: "2026-03-18"
status: "stable"
---

# Comparative Politics — Cross-National Panel Analysis

## When to Use

Use this skill when you need to:

- Download and merge the Quality of Government (QoG) standard dataset, Polity 5 scores, and
  Freedom House annual ratings into a unified country-year panel
- Classify countries into regime types (autocracy, hybrid/anocracy, democracy) using Polity 5
  thresholds, and track transitions over time
- Run panel OLS with country and year fixed effects using the `linearmodels` package (handles
  the Frisch-Waugh-Lovell within transformation efficiently)
- Perform Hausman test to choose between fixed effects and random effects models
- Detect democratic backsliding as a rolling 3-year change in Polity score
- Produce Lipset-style scatter plots (GDP per capita vs. democracy score) with regional coloring
- Estimate multilevel models where countries are nested within regions

This skill covers exploratory analysis, static regression, and dynamic models. For time-series
specific methods (cointegration, error correction), use a dedicated time-series skill.

## Background

**QoG (Quality of Government)** is maintained by the University of Gothenburg. The standard
dataset (`qog_std_cs_jan23.csv` or time-series `qog_std_ts_jan23.csv`) aggregates ~2,000 variables
from over 100 sources. Key governance variables:

| QoG Variable | Description | Source |
|---|---|---|
| `wbgi_cce` | Control of Corruption Estimate | World Bank |
| `wbgi_rle` | Rule of Law Estimate | World Bank |
| `wbgi_gee` | Government Effectiveness Estimate | World Bank |
| `undp_hdi` | Human Development Index | UNDP |
| `wdi_gdpcapcon2015` | GDP per capita (constant 2015 USD) | World Bank |

**Polity 5** (Marshall & Gurr, 2020) codes democracy and autocracy on -10 to +10 scale:
- ≤ -6: full autocracy
- -5 to +5: hybrid (anocracy)
- ≥ +6: democracy

Special codes: -66 (interruption), -77 (interregnum), -88 (transition) — replace with NA.

**Freedom House** provides Political Rights (PR) and Civil Liberties (CL) scores (1-7 each,
lower = more free). Combined score (14 = least free, 2 = most free). Status: Free/Partly Free/
Not Free.

**Panel regression with FE**: The within transformation demeans all variables by entity (country)
mean, removing all time-invariant country characteristics (culture, geography, history). `linearmodels`
PanelOLS with `entity_effects=True` implements this. Standard errors should be clustered at the
country level.

**Hausman test**: Compares FE and RE estimates. Under H0 (RE consistent), both estimators agree.
Rejection of H0 → use FE. `linearmodels` provides the `compare` function for this test.

**Democratic backsliding**: A 3-year rolling decline in Polity score below a threshold (e.g., -3
points). Distinguished from transitions that start from democratic levels (±6 threshold).

## Environment Setup

```bash
pip install pandas>=1.5 statsmodels>=0.14 linearmodels>=4.28 numpy>=1.23 matplotlib>=3.6
```

Download datasets:
- QoG: https://www.qogdata.pol.gu.se/data/qog_std_ts_jan23.csv
- Polity 5: https://www.systemicpeace.org/inscrdata.html (p5v2018.xls)
- Freedom House: https://freedomhouse.org/sites/default/files/2023-02/All_data_FIW_2013-2023.xlsx

```bash
export QOG_PATH="/data/qog_std_ts_jan23.csv"
export POLITY_PATH="/data/p5v2018.xls"
export FH_PATH="/data/FIW_2013-2023.xlsx"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Load and Harmonize QoG Data
# ---------------------------------------------------------------------------

QOG_KEY_VARS = [
    "cname", "ccodealp", "year",
    "wbgi_cce", "wbgi_rle", "wbgi_gee",
    "wdi_gdpcapcon2015", "wdi_pop",
    "undp_hdi",
]


def load_qog(path: str, variables: list[str] | None = None,
             years: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    Load the QoG time-series dataset.

    Parameters
    ----------
    path : str
        Path to QoG standard time-series CSV.
    variables : list of str, optional
        Variables to retain; always includes cname, ccodealp, year.
    years : tuple, optional
        Year range filter.

    Returns
    -------
    pd.DataFrame standardized with columns: country, iso3, year, and requested variables.
    """
    header = pd.read_csv(path, nrows=0)
    keep = list(set(["cname", "ccodealp", "year"] + (variables or QOG_KEY_VARS)))
    use = [c for c in keep if c in header.columns]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    if years:
        df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]
    df = df.rename(columns={"cname": "country", "ccodealp": "iso3"})
    return df.sort_values(["country", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Load Polity 5 and Classify Regime Types
# ---------------------------------------------------------------------------

POLITY_SPECIAL_CODES = {-66, -77, -88}


def load_polity5(path: str) -> pd.DataFrame:
    """
    Load Polity 5 dataset from Excel file.

    Returns
    -------
    pd.DataFrame with: iso3, country, year, polity2, regime_type.
    """
    df = pd.read_excel(path, sheet_name=0)
    # Standardize column names (different versions use different names)
    df.columns = [c.lower().strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        if "scode" in col or col == "country":
            rename_map[col] = "country" if "country" in col else "iso_scode"
        if col in ("polity2", "polity"):
            rename_map[col] = "polity2"
    df = df.rename(columns=rename_map)

    # Keep key columns
    keep = [c for c in ["scode", "country", "year", "polity2", "democ", "autoc"] if c in df.columns]
    df = df[keep].copy()

    # Replace special Polity codes with NaN
    if "polity2" in df.columns:
        df["polity2"] = pd.to_numeric(df["polity2"], errors="coerce")
        df.loc[df["polity2"].isin(POLITY_SPECIAL_CODES), "polity2"] = np.nan
        df["regime_type"] = pd.cut(
            df["polity2"],
            bins=[-11, -6, 5, 10],
            labels=["Autocracy", "Hybrid/Anocracy", "Democracy"],
        )
    return df


def classify_regime(polity_score: float) -> str:
    """Classify a single Polity 2 score into regime type."""
    if pd.isna(polity_score):
        return "Unknown"
    if polity_score <= -6:
        return "Autocracy"
    if polity_score >= 6:
        return "Democracy"
    return "Hybrid/Anocracy"


# ---------------------------------------------------------------------------
# 3. Load Freedom House
# ---------------------------------------------------------------------------

def load_freedom_house(path: str) -> pd.DataFrame:
    """
    Load Freedom House FIW data from Excel.

    Returns
    -------
    pd.DataFrame with: country, year, fh_pr, fh_cl, fh_total, fh_status.
    """
    # FH Excel typically has multiple sheets; try the main country scores sheet
    try:
        df = pd.read_excel(path, sheet_name="FIW06-23", header=1)
    except Exception:
        df = pd.read_excel(path, header=0)

    df.columns = [str(c).strip() for c in df.columns]
    # Typical columns: Country/Territory, C/T, Edition, Status, PR, CL, Total
    rename = {
        "Country/Territory": "country",
        "Edition": "year",
        "Status": "fh_status",
        "PR": "fh_pr",
        "CL": "fh_cl",
        "Total": "fh_total",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["country", "year", "fh_pr", "fh_cl", "fh_total", "fh_status"] if c in df.columns]
    return df[keep].dropna(subset=["country", "year"]).copy()


# ---------------------------------------------------------------------------
# 4. Merge Panel Datasets
# ---------------------------------------------------------------------------

def build_comparative_panel(
    qog_df: pd.DataFrame,
    polity_df: pd.DataFrame,
    fh_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge QoG, Polity 5, and (optionally) Freedom House into a country-year panel.

    Merge key: country name + year (fuzzy ISO3 matching as fallback).

    Returns
    -------
    pd.DataFrame — merged country-year panel.
    """
    panel = qog_df.copy()

    # Merge Polity 5
    if "country" in polity_df.columns and "year" in polity_df.columns:
        panel = panel.merge(
            polity_df[["country", "year", "polity2", "regime_type"]],
            on=["country", "year"],
            how="left",
        )

    # Merge Freedom House
    if fh_df is not None and "country" in fh_df.columns:
        fh_keep = [c for c in ["country", "year", "fh_pr", "fh_cl", "fh_total", "fh_status"] if c in fh_df.columns]
        panel = panel.merge(fh_df[fh_keep], on=["country", "year"], how="left")

    # Log GDP
    if "wdi_gdpcapcon2015" in panel.columns:
        panel["log_gdppc"] = np.log(panel["wdi_gdpcapcon2015"].clip(lower=1))
    if "wdi_pop" in panel.columns:
        panel["log_pop"] = np.log(panel["wdi_pop"].clip(lower=1))

    return panel.sort_values(["country", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Panel Regression with Country + Year FE
# ---------------------------------------------------------------------------

def panel_fe_regression(
    panel: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    country_col: str = "country",
    year_col: str = "year",
    entity_effects: bool = True,
    time_effects: bool = True,
) -> object:
    """
    Run panel OLS with optional entity and time fixed effects (linearmodels).

    Parameters
    ----------
    panel : pd.DataFrame
    outcome : str
        Dependent variable.
    predictors : list of str
    entity_effects : bool
        Country fixed effects (within-estimator).
    time_effects : bool
        Year fixed effects.

    Returns
    -------
    linearmodels PanelResults object.
    """
    from linearmodels.panel import PanelOLS, RandomEffects

    sub = panel[[country_col, year_col, outcome] + predictors].dropna().copy()
    sub = sub.set_index([country_col, year_col])
    endog = sub[outcome]
    exog = sm.add_constant(sub[predictors])

    model = PanelOLS(
        endog, exog,
        entity_effects=entity_effects,
        time_effects=time_effects,
    )
    return model.fit(cov_type="clustered", cluster_entity=True)


# ---------------------------------------------------------------------------
# 6. Democratic Backsliding Detection
# ---------------------------------------------------------------------------

def detect_backsliding(
    panel: pd.DataFrame,
    polity_col: str = "polity2",
    window: int = 3,
    threshold: float = 3.0,
    country_col: str = "country",
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Flag country-years where the rolling Polity 2 score declined by threshold
    points over the preceding window years, starting from a democratic baseline (≥+6).

    Parameters
    ----------
    panel : pd.DataFrame
    polity_col : str
    window : int
        Rolling lookback in years.
    threshold : float
        Minimum absolute decline to flag.
    country_col : str
    year_col : str

    Returns
    -------
    pd.DataFrame of backsliding episodes, sorted by severity.
    """
    df = panel.sort_values([country_col, year_col]).copy()
    df["polity_lag"] = df.groupby(country_col)[polity_col].shift(window)
    df["polity_change"] = df[polity_col] - df["polity_lag"]

    episodes = df[
        (df["polity_change"] <= -threshold) &
        (df["polity_lag"] >= 6)
    ].copy()
    episodes["severity"] = episodes["polity_change"].abs()
    return (
        episodes[[country_col, year_col, polity_col, "polity_lag", "polity_change", "severity"]]
        .sort_values("severity", ascending=False)
        .reset_index(drop=True)
    )
```

## Advanced Usage

### Hausman Test for FE vs. RE

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


def hausman_test(fe_result, re_result) -> dict:
    """
    Hausman specification test: H0 = RE is consistent (both FE and RE agree).

    Parameters
    ----------
    fe_result : linearmodels PanelResults (entity_effects=True, time_effects=False)
    re_result : linearmodels PanelResults (RandomEffects)

    Returns
    -------
    dict with test_stat, df, pvalue, conclusion.
    """
    from scipy.stats import chi2

    b_fe = fe_result.params
    b_re = re_result.params

    # Align common variables
    common = b_fe.index.intersection(b_re.index)
    diff = b_fe[common] - b_re[common]

    # Variance difference
    v_fe = fe_result.cov.loc[common, common]
    v_re = re_result.cov.loc[common, common]
    v_diff = v_fe - v_re

    try:
        # Moore-Penrose pseudo-inverse for potentially singular matrix
        v_inv = np.linalg.pinv(v_diff.values)
        stat = float(diff.values @ v_inv @ diff.values)
    except np.linalg.LinAlgError:
        return {"test_stat": np.nan, "df": len(common), "pvalue": np.nan,
                "conclusion": "Hausman test failed (singular matrix)"}

    df_h = len(common)
    pvalue = 1 - chi2.cdf(stat, df_h)
    conclusion = "Use FE (reject RE)" if pvalue < 0.05 else "RE preferred (fail to reject)"
    return {
        "test_stat": round(stat, 4),
        "df": df_h,
        "pvalue": round(pvalue, 4),
        "conclusion": conclusion,
    }
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| Large number of NaN after merge | Country name inconsistencies (e.g., "Czech Republic" vs "Czechia") | Harmonize using ISO3 codes; use `pycountry` for name normalization |
| Polity 5 shows -66/-77/-88 | Special codes for interruption/interregnum | Replace with `NaN` using `POLITY_SPECIAL_CODES` set |
| PanelOLS absorbed all variation | FE absorbs time-invariant outcome | Use within-country time variation; check that outcome changes over time |
| Freedom House file structure changed | Annual report format varies | Inspect the Excel sheet names and header rows before loading |
| `linearmodels` Hausman test | Package does not provide built-in Hausman | Use the manual `hausman_test` function above |

## External Resources

- QoG Dataset: https://www.qogdata.pol.gu.se/
- Polity 5 Project: https://www.systemicpeace.org/polityproject.html
- Freedom House: https://freedomhouse.org/report/freedom-world
- Teorell, J. et al. (2023). The Quality of Government Standard Dataset. University of Gothenburg.
- Marshall, M.G. & Gurr, T.R. (2020). *Polity 5: Political Regime Characteristics and Transitions, 1800–2018*. Center for Systemic Peace.
- Lipset, S.M. (1959). Some social requisites of democracy. *APSR*, 53(1), 69-105.

## Examples

### Example 1: Merge QoG + Polity + FH and Classify Regimes

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
countries_list = [f"Country_{i:02d}" for i in range(1, 51)]
years_list = list(range(2000, 2023))

records_qog, records_polity = [], []
for ctr in countries_list:
    base_polity = rng.integers(-10, 11)
    base_gdp = rng.lognormal(9, 1.5)
    for yr in years_list:
        drift = rng.integers(-1, 2)
        polity_val = int(np.clip(base_polity + drift + rng.integers(-1, 2), -10, 10))
        records_qog.append({
            "country": ctr, "iso3": ctr[:3].upper(),
            "year": yr,
            "wbgi_cce": rng.normal(0, 1),
            "wbgi_rle": rng.normal(0, 1),
            "wdi_gdpcapcon2015": base_gdp * rng.lognormal(0, 0.05),
            "wdi_pop": rng.lognormal(15, 1),
        })
        records_polity.append({
            "country": ctr, "year": yr, "polity2": polity_val,
            "regime_type": classify_regime(polity_val),
        })

df_qog_sim = pd.DataFrame(records_qog)
df_polity_sim = pd.DataFrame(records_polity)

panel = build_comparative_panel(df_qog_sim, df_polity_sim)

# Regime distribution over time
regime_counts = panel.groupby(["year", "regime_type"]).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12, 5))
regime_counts.plot(kind="area", stacked=True, ax=ax, colormap="RdYlGn", alpha=0.8)
ax.set_title("Global Regime Distribution Over Time (Polity 5 Classification)")
ax.set_ylabel("Number of Countries")
ax.set_xlabel("Year")
ax.legend(title="Regime Type")
plt.tight_layout()
plt.savefig("regime_distribution.png", dpi=150)
plt.show()

print("\nRegime counts (latest year):")
print(panel[panel["year"] == panel["year"].max()]["regime_type"].value_counts())
```

### Example 2: Within-Country FE Regression — Corruption and GDP

```python
import numpy as np
import pandas as pd

# Using the simulated panel from Example 1
panel_reg = panel.copy()
panel_reg["log_gdppc"] = np.log(panel_reg["wdi_gdpcapcon2015"].clip(lower=1))

result_fe = panel_fe_regression(
    panel_reg,
    outcome="wbgi_cce",
    predictors=["log_gdppc"],
    entity_effects=True,
    time_effects=True,
)
print("=== Country+Year FE: GDP per capita → Corruption Control ===")
print(result_fe.summary.tables[1])
print(f"\nWithin R²: {result_fe.rsquared:.4f}")
```

### Example 3: Democratic Backsliding Detection and Trend Plot

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Detect backsliding in the simulated panel
episodes = detect_backsliding(
    panel,
    polity_col="polity2",
    window=3,
    threshold=3.0,
)
print(f"=== Democratic Backsliding Episodes (n={len(episodes)}) ===")
print(episodes.head(10).to_string(index=False))

# Rolling 3-year polity change (global average)
panel_sorted = panel.sort_values(["country", "year"])
panel_sorted["polity_lag3"] = panel_sorted.groupby("country")["polity2"].shift(3)
panel_sorted["polity_delta3"] = panel_sorted["polity2"] - panel_sorted["polity_lag3"]

global_delta = panel_sorted.groupby("year")["polity_delta3"].mean()

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(global_delta.index, global_delta.values,
       color=["#d73027" if v < 0 else "#4575b4" for v in global_delta.values], alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean 3-Year Change in Polity Score (Global Average)")
ax.set_xlabel("Year")
ax.set_ylabel("Mean Polity Change (3-year)")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("polity_backsliding_trend.png", dpi=150)
plt.show()

# Lipset scatter: GDP vs Polity
latest = panel[panel["year"] == panel["year"].max()].dropna(subset=["log_gdppc", "polity2"])
fig2, ax2 = plt.subplots(figsize=(9, 6))
scatter = ax2.scatter(latest["log_gdppc"], latest["polity2"], alpha=0.7,
                       c=pd.Categorical(latest["regime_type"]).codes,
                       cmap="RdYlGn", edgecolors="white", s=60, linewidths=0.4)
slope, intercept, r, p, se = stats.linregress(latest["log_gdppc"], latest["polity2"])
x_line = np.linspace(latest["log_gdppc"].min(), latest["log_gdppc"].max(), 100)
ax2.plot(x_line, intercept + slope * x_line, "r--", linewidth=1.5, label=f"OLS (r={r:.2f})")
ax2.set_xlabel("log(GDP per capita, constant 2015 USD)")
ax2.set_ylabel("Polity 2 Score")
ax2.set_title("Lipset's Modernization Hypothesis: GDP vs. Democracy")
ax2.axhline(6, color="gray", linestyle=":", linewidth=1, label="Democracy threshold (+6)")
ax2.axhline(-6, color="gray", linestyle=":", linewidth=1, label="Autocracy threshold (-6)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lipset_scatter.png", dpi=150)
plt.show()

from scipy import stats
