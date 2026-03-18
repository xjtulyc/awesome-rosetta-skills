---
name: food-systems-data
description: >
  Use this Skill to analyze food systems data from FAOSTAT: production/trade/food
  balance sheets, food security indicators, dietary diversity, and trade flow
  Sankey diagrams.
tags:
  - agriculture
  - food-systems
  - FAOSTAT
  - food-security
  - dietary-diversity
  - trade-flows
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
    - pandas>=1.5
    - requests>=2.28
    - numpy>=1.23
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Food Systems Data Analysis with FAOSTAT

> **TL;DR** — Download production, trade, and food balance sheet data from
> the FAOSTAT REST API, compute food security indicators (caloric adequacy,
> import dependency), and visualize trade flows as Sankey diagrams.

---

## When to Use

Use this Skill when you need to:

- Retrieve country-level agricultural production statistics (area harvested,
  yield, production quantity) across crops, livestock, and fisheries.
- Analyse food balance sheets to understand caloric supply, utilisation,
  and food security status.
- Compute food import dependency ratios and price volatility indices.
- Build bilateral trade flow matrices and visualise major commodity corridors.
- Compare dietary energy adequacy across countries or over time.

**Do NOT use** this Skill for pixel-level land use data (use remote sensing
skills), or for individual-level dietary surveys (use household survey methods).

---

## Background

### FAOSTAT Dataset Codes

| Code | Dataset Name | Key Elements |
|---|---|---|
| QCL | Crop and livestock production | Area harvested, yield, production |
| TCL | Trade crops and livestock | Import/export quantity and value |
| FBS | Food Balance Sheets | Production, trade, losses, food supply |
| FS | Suite of Food Security Indicators | Prevalence of undernourishment, PoU |
| PP | Producer prices | Annual average price per commodity |

### Food Balance Sheet Structure

```
Food Supply = Production + Imports - Exports - Stock Changes - Non-food uses - Losses
Per capita food supply (kcal/person/day) = Food Supply (kcal) / Population / 365
```

### Food Security Indicators

- **Dietary Energy Adequacy (DEA)**: food supply / dietary energy requirement × 100 %
- **Food Import Dependency Ratio**: imports / (production + imports - exports) × 100 %
- **Share of dietary energy from cereals**: proxy for diet quality

---

## Environment Setup

```bash
conda create -n foodsys python=3.11 -y
conda activate foodsys
pip install pandas>=1.5 requests>=2.28 numpy>=1.23 matplotlib>=3.6

# Optional: plotly for interactive Sankey
pip install plotly>=5.18
```

```python
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("Dependencies loaded OK")
```

---

## Core Workflow

### Step 1 — FAOSTAT REST API Download

```python
FAOSTAT_BASE = "http://fenixservices.fao.org/faostat/api/v1/en/data"


def faostat_download(
    dataset: str,
    area_codes: list[str],
    element_codes: list[str],
    item_codes: list[str],
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    Download data from the FAOSTAT API.

    Args:
        dataset:       FAOSTAT dataset code (e.g. 'QCL', 'TCL', 'FBS', 'FS').
        area_codes:    List of ISO3 country codes or FAOSTAT area codes.
        element_codes: List of element codes (e.g. '5510'=production, '5312'=area).
        item_codes:    List of item codes (e.g. '15'=wheat, '56'=maize).
        year_start:    First year to retrieve.
        year_end:      Last year to retrieve.

    Returns:
        DataFrame with columns: Area, Element, Item, Year, Unit, Value.
    """
    params = {
        "area": ",".join(area_codes),
        "element": ",".join(element_codes),
        "item": ",".join(item_codes),
        "year": ",".join(str(y) for y in range(year_start, year_end + 1)),
        "output_type": "objects",
    }

    url = f"{FAOSTAT_BASE}/{dataset}/"
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    raw = resp.json()
    if "data" not in raw:
        raise ValueError(f"FAOSTAT returned no 'data' key. Response: {raw.get('message', '')}")

    df = pd.DataFrame(raw["data"])
    # Standardise column names
    rename_map = {
        "Area": "area", "Element": "element", "Item": "item",
        "Year": "year", "Unit": "unit", "Value": "value",
        "AreaCode": "area_code", "ItemCode": "item_code", "ElementCode": "element_code",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def faostat_list_areas() -> pd.DataFrame:
    """List all FAOSTAT area codes and names."""
    resp = requests.get("http://fenixservices.fao.org/faostat/api/v1/en/definitions/area/", timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json().get("data", []))
```

### Step 2 — Production Trend Analysis

```python
def plot_production_trend(
    countries: list[str],
    item_name: str = "Wheat",
    item_code: str = "15",
    year_start: int = 2000,
    year_end: int = 2022,
    output_path: str = "production_trend.png",
) -> pd.DataFrame:
    """
    Download and plot wheat (or other crop) production trends for selected countries.

    Args:
        countries:   List of FAOSTAT area codes (e.g. ['356'=India, '156'=China]).
        item_name:   Human-readable crop name for plot title.
        item_code:   FAOSTAT item code.
        year_start:  First year.
        year_end:    Last year.
        output_path: Output PNG path.

    Returns:
        Pivoted DataFrame: years as index, countries as columns, values in tonnes.
    """
    df = faostat_download(
        dataset="QCL",
        area_codes=countries,
        element_codes=["5510"],          # 5510 = Production
        item_codes=[item_code],
        year_start=year_start,
        year_end=year_end,
    )

    pivot = df.pivot_table(index="year", columns="area", values="value", aggfunc="sum")
    pivot /= 1e6  # Convert to million tonnes

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", markersize=3, label=col)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"{item_name} production (million tonnes)")
    ax.set_title(f"{item_name} Production Trends {year_start}–{year_end}")
    ax.legend(title="Country", bbox_to_anchor=(1.02, 1))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved production trend to {output_path}")
    plt.close()

    return pivot
```

### Step 3 — Food Balance Sheet Analysis

```python
def analyse_food_balance_sheet(
    country_code: str,
    year: int = 2020,
) -> pd.DataFrame:
    """
    Download and summarise a food balance sheet for a given country and year.

    Elements retrieved:
      - 5510: Production   - 5610: Import quantity
      - 5910: Export quantity  - 5154: Food supply (kcal/capita/day)
      - 5301: Food         - 5131: Feed
      - 5527: Losses

    Args:
        country_code: FAOSTAT area code for the country.
        year:         Reference year.

    Returns:
        DataFrame summarising the food balance sheet components per commodity.
    """
    elements = {
        "5510": "production_t",
        "5610": "imports_t",
        "5910": "exports_t",
        "5154": "food_supply_kcal_pc_d",
        "5527": "losses_t",
    }

    df = faostat_download(
        dataset="FBS",
        area_codes=[country_code],
        element_codes=list(elements.keys()),
        item_codes=["2901"],  # 2901 = Grand Total
        year_start=year,
        year_end=year,
    )

    pivot = df.pivot_table(index="item", columns="element_code", values="value", aggfunc="sum")
    pivot.rename(columns=elements, inplace=True)

    # Compute import dependency ratio
    if all(c in pivot.columns for c in ["production_t", "imports_t", "exports_t"]):
        pivot["import_dependency_pct"] = (
            pivot["imports_t"] / (pivot["production_t"] + pivot["imports_t"] - pivot["exports_t"]) * 100
        ).clip(0, 100)

    return pivot


def compute_caloric_sufficiency(
    country_code: str,
    kcal_requirement_per_day: float = 2100.0,
    year_start: int = 2000,
    year_end: int = 2020,
) -> pd.DataFrame:
    """
    Compute Dietary Energy Adequacy (DEA) over time for a country.

    DEA = (food supply kcal/cap/day) / kcal_requirement * 100

    Values above 100% indicate sufficient caloric supply at national level.

    Args:
        country_code:           FAOSTAT area code.
        kcal_requirement_per_day: Average dietary energy requirement per person.
        year_start:             First year.
        year_end:               Last year.

    Returns:
        DataFrame with columns: year, food_supply_kcal, DEA_pct.
    """
    df = faostat_download(
        dataset="FBS",
        area_codes=[country_code],
        element_codes=["5154"],          # Per capita food supply kcal/cap/day
        item_codes=["2901"],             # Grand Total
        year_start=year_start,
        year_end=year_end,
    )

    dea = df[["year", "value"]].copy()
    dea.rename(columns={"value": "food_supply_kcal"}, inplace=True)
    dea["DEA_pct"] = dea["food_supply_kcal"] / kcal_requirement_per_day * 100
    dea = dea.dropna().sort_values("year")

    print(f"Country {country_code} — DEA range {year_start}–{year_end}:")
    print(f"  Min: {dea['DEA_pct'].min():.1f}%  Max: {dea['DEA_pct'].max():.1f}%")
    return dea
```

---

## Advanced Usage

### Trade Flow Matrix and Top Corridors

```python
def build_trade_flow_matrix(
    item_code: str = "15",
    year: int = 2020,
    top_n: int = 15,
    output_path: str = "trade_flows.png",
) -> pd.DataFrame:
    """
    Build a bilateral trade matrix for a commodity and plot top export corridors.

    Args:
        item_code:   FAOSTAT item code (e.g. '15' = wheat).
        year:        Reference year.
        top_n:       Number of top corridors to display.
        output_path: Output path for bar chart.

    Returns:
        DataFrame with columns: reporter_area, partner_area, export_value_t.
    """
    # Download detailed trade matrix (TCL bilateral)
    resp = requests.get(
        f"{FAOSTAT_BASE}/TM/",
        params={
            "item": item_code,
            "element": "5910",       # Export quantity (tonnes)
            "year": year,
            "output_type": "objects",
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json().get("data", [])

    if not raw:
        print("No bilateral trade data returned — check item/year combination.")
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["Value"] = pd.to_numeric(df.get("Value", 0), errors="coerce").fillna(0)

    # Aggregate top export corridors
    df["corridor"] = df["Area"] + " → " + df.get("PartnerCountry", "?")
    top = df.groupby("corridor")["Value"].sum().nlargest(top_n).reset_index()
    top["Value_Mt"] = top["Value"] / 1e6

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["corridor"], top["Value_Mt"], color="#1f77b4")
    ax.set_xlabel("Export Volume (million tonnes)")
    ax.set_title(f"Top {top_n} Trade Corridors — Item {item_code}, {year}")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Trade flow chart saved to {output_path}")
    plt.close()

    return top
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `requests.exceptions.ConnectionError` | No internet or FAOSTAT server down | Check https://www.fao.org/faostat; retry |
| Empty `data` list in API response | Wrong area/item/element code combination | Use `faostat_list_areas()` to verify codes |
| `KeyError: 'element_code'` | API changed response column names | Print `df.columns` to inspect actual names |
| Very slow download | Large year range + many countries | Chunk requests by decade |
| Import dependency > 100% | Negative stock changes in balance | Clip to [0, 100] range |
| FAOSTAT returns -999 or flag 'M' | Missing data for that year/country | Filter `value > 0` before analysis |

---

## External Resources

- FAOSTAT API documentation: https://www.fao.org/faostat/en/#data
- FAOSTAT API bulk download guide: http://fenixservices.fao.org/faostat/api/v1/en/
- Food Balance Sheets methodology: https://www.fao.org/3/X9892E/X9892E05.htm
- Global Food Security Index (EIU): https://foodsecurityindex.eiu.com/
- USDA FoodData Central API: https://fdc.nal.usda.gov/api-guide.html

---

## Examples

### Example 1 — Global Wheat Production Trends (Top 5 Producers)

```python
def example_wheat_production_trends():
    """Download and plot wheat production for the top 5 global producers."""
    # FAOSTAT area codes: China=156, India=100, Russia=185, USA=231, France=68
    top5 = ["156", "100", "185", "231", "68"]

    pivot = plot_production_trend(
        countries=top5,
        item_name="Wheat",
        item_code="15",
        year_start=2000,
        year_end=2022,
        output_path="wheat_production_top5.png",
    )

    # Compute average growth rate
    for country in pivot.columns:
        series = pivot[country].dropna()
        if len(series) >= 2:
            cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / (len(series) - 1)) - 1
            print(f"  {country}: CAGR = {cagr * 100:.1f}% per year")

    return pivot


if __name__ == "__main__":
    example_wheat_production_trends()
```

### Example 2 — Food Balance Sheet and Caloric Sufficiency for Ethiopia

```python
def example_ethiopia_food_security():
    """Analyse Ethiopia food balance sheet and dietary energy adequacy trend."""
    country_code = "238"  # Ethiopia FAOSTAT code

    print("1. Food Balance Sheet for Ethiopia (2020):")
    fbs = analyse_food_balance_sheet(country_code, year=2020)
    print(fbs.to_string())

    print("\n2. Dietary Energy Adequacy trend (2000–2020):")
    dea_df = compute_caloric_sufficiency(
        country_code,
        kcal_requirement_per_day=2100.0,
        year_start=2000,
        year_end=2020,
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dea_df["year"], dea_df["DEA_pct"], "o-", color="#d62728")
    ax.axhline(100, color="black", linestyle="--", linewidth=0.8, label="100% adequacy threshold")
    ax.fill_between(dea_df["year"], 100, dea_df["DEA_pct"],
                    where=dea_df["DEA_pct"] < 100, alpha=0.2, color="red", label="Deficit")
    ax.set_xlabel("Year")
    ax.set_ylabel("DEA (%)")
    ax.set_title("Ethiopia — Dietary Energy Adequacy 2000–2020")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ethiopia_dea.png", dpi=150)
    print("DEA chart saved to ethiopia_dea.png")
    return dea_df


if __name__ == "__main__":
    example_ethiopia_food_security()
```

### Example 3 — Trade Flow Matrix for Global Wheat

```python
def example_wheat_trade_flows():
    """Visualise top global wheat export corridors."""
    top_corridors = build_trade_flow_matrix(
        item_code="15",   # Wheat
        year=2020,
        top_n=15,
        output_path="wheat_trade_corridors.png",
    )
    print("\nTop wheat trade corridors (million tonnes):")
    print(top_corridors.to_string(index=False))
    return top_corridors


if __name__ == "__main__":
    example_wheat_trade_flows()
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — FAOSTAT download, food balance sheet, DEA, trade flows |
