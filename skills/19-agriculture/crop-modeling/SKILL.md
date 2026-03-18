---
name: crop-modeling
description: >
  Use this Skill for crop growth simulation: WOFOST/AquaCrop via pcse,
  meteorological input from NASA POWER API, yield gap analysis, and climate
  scenario comparison.
tags:
  - agriculture
  - crop-modeling
  - WOFOST
  - AquaCrop
  - yield-gap
  - climate-scenarios
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
    - pcse>=5.5
    - pandas>=1.5
    - numpy>=1.23
    - matplotlib>=3.6
    - requests>=2.28
last_updated: "2026-03-18"
status: stable
---

# Crop Growth Modeling with WOFOST and AquaCrop

> **TL;DR** — Simulate crop growth using the PCSE/WOFOST framework, fetch
> meteorological inputs from the NASA POWER API, run yield gap analyses
> (potential / water-limited / actual), and compare scenarios under projected
> climate change.

---

## When to Use

Use this Skill when you need to:

- Simulate daily crop development and biomass accumulation for cereals,
  legumes, or root crops using the WOFOST or AquaCrop engine.
- Download site-specific weather data automatically from NASA POWER without
  manual data collection.
- Quantify the yield gap between potential (no stress), water-limited
  (attainable), and farmer-practice (actual) production levels.
- Evaluate the sensitivity of yield to temperature increases or altered
  precipitation under future climate scenarios.
- Provide evidence for agronomic interventions (irrigation, sowing date
  optimization, fertilizer scheduling).

**Do NOT use** this Skill for pixel-level remote sensing yield prediction
(see `agri-remote-sensing`), or for livestock / aquaculture modelling.

---

## Background

### PCSE Framework Architecture

The Python Crop Simulation Environment (PCSE) provides a modular implementation
of the WOFOST 8.1 model and the AquaCrop-OS model. WOFOST simulates:

| Component | Description |
|---|---|
| Phenology | Development stage (DVS) driven by temperature sum |
| Assimilation | Daily gross CO2 assimilation from radiation interception |
| Partitioning | Dry matter allocation to roots/stems/leaves/storage organs |
| Water balance | Soil water content, evapotranspiration (Penman-Monteith) |
| Nutrient | Simplified N/P/K balance (WOFOST 8.x) |

### Yield Gap Decomposition

The yield gap framework distinguishes three production levels:

```
Potential yield (Yp)       No water, nutrient, pest stress — radiation/temperature limited
       |
   Water gap
       |
Attainable yield (Yw)      Water-limited — irrigation feasibility boundary
       |
   Management gap
       |
Actual yield (Ya)          Farmer practice — observed in field/statistics
```

Yield gap = Yp - Ya (absolute) or (Yp - Ya)/Yp * 100 (relative %).

### NASA POWER API

NASA POWER (Prediction of Worldwide Energy Resources) provides daily meteorological
data at 0.5° × 0.5° resolution globally since 1984:

- `ALLSKY_SFC_SW_DWN` — all-sky surface shortwave downwelling irradiance (MJ/m²/day)
- `T2M` — temperature at 2 m (°C)
- `T2MDEW` — dew-point temperature at 2 m (°C) → relative humidity
- `WS10M` — wind speed at 10 m (m/s)
- `PRECTOTCORR` — bias-corrected precipitation (mm/day)

PCSE expects these fields in a `WeatherDataProvider` object with daily records.

---

## Environment Setup

```bash
# Create isolated environment
conda create -n cropmodel python=3.11 -y
conda activate cropmodel

# Core dependencies
pip install pcse>=5.5 pandas>=1.5 numpy>=1.23 matplotlib>=3.6 requests>=2.28

# Verify PCSE installation
python -c "import pcse; print('PCSE version:', pcse.__version__)"
```

Download WOFOST crop and soil parameter files:

```bash
# PCSE ships example data — copy to working directory
python -c "
import pcse, os, shutil
data_dir = os.path.join(os.path.dirname(pcse.__file__), 'db', 'pcse')
shutil.copytree(data_dir, './pcse_data', dirs_exist_ok=True)
print('Copied PCSE data to ./pcse_data')
"
```

---

## Core Workflow

### Step 1 — Download Weather Data from NASA POWER

```python
import requests
import pandas as pd
from datetime import date


def fetch_nasa_power(
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Download daily meteorological data from NASA POWER API.

    Args:
        lat:   Latitude in decimal degrees (positive = North).
        lon:   Longitude in decimal degrees (positive = East).
        start: First day of the requested period.
        end:   Last day of the requested period.

    Returns:
        DataFrame indexed by date with columns:
        IRRAD (MJ/m²/d), TMAX, TMIN, TEMP, VAP (hPa), WIND (m/s), RAIN (mm/d).
    """
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M,T2MDEW,T2M_MAX,T2M_MIN,WS10M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }

    resp = requests.get(base_url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    properties = data["properties"]["parameter"]
    dates = pd.date_range(start, end, freq="D")
    date_keys = [d.strftime("%Y%m%d") for d in dates]

    records = []
    for dk in date_keys:
        tmax = properties["T2M_MAX"].get(dk, float("nan"))
        tmin = properties["T2M_MIN"].get(dk, float("nan"))
        tdew = properties["T2MDEW"].get(dk, float("nan"))
        irrad_mj = properties["ALLSKY_SFC_SW_DWN"].get(dk, float("nan"))
        wind = properties["WS10M"].get(dk, float("nan"))
        rain = properties["PRECTOTCORR"].get(dk, 0.0)

        # Vapour pressure from dew-point temperature (Tetens formula)
        vap = 6.1078 * 10 ** (7.5 * tdew / (237.3 + tdew)) if not pd.isna(tdew) else float("nan")

        records.append({
            "DATE": pd.Timestamp(dk),
            "IRRAD": irrad_mj * 1e6,        # Convert MJ to J/m²/d for PCSE
            "TMAX": tmax,
            "TMIN": tmin,
            "TEMP": (tmax + tmin) / 2,
            "VAP": vap,
            "WIND": wind,
            "RAIN": rain,
        })

    df = pd.DataFrame(records).set_index("DATE")
    # Fill missing values with climatological defaults
    df.fillna(method="ffill", inplace=True)
    return df
```

### Step 2 — Build a PCSE Weather Provider from the DataFrame

```python
from pcse.base import WeatherDataContainer, WeatherDataProvider
from pcse.util import reference_ET


class NASAPOWERWeatherProvider(WeatherDataProvider):
    """
    Wrap a NASA POWER DataFrame as a PCSE WeatherDataProvider.

    Args:
        weather_df: DataFrame from fetch_nasa_power().
        lat:        Site latitude (decimal degrees).
        lon:        Site longitude (decimal degrees).
        elevation:  Site elevation in metres (default 0).
    """

    def __init__(self, weather_df: pd.DataFrame, lat: float, lon: float, elevation: float = 0.0):
        super().__init__()
        self.latitude = lat
        self.longitude = lon
        self.elevation = elevation
        self.description = [f"NASA POWER weather for ({lat:.2f}, {lon:.2f})"]
        self.angstA = 0.29
        self.angstB = 0.49

        for dt, row in weather_df.iterrows():
            et0 = reference_ET(
                day=dt.to_pydatetime().date(),
                lat=lat,
                elev=elevation,
                tmin=row["TMIN"],
                tmax=row["TMAX"],
                irrad=row["IRRAD"],
                vap=row["VAP"],
                wind=row["WIND"],
                angstA=self.angstA,
                angstB=self.angstB,
                etmodel="PM",
            )
            wdc = WeatherDataContainer(
                LAT=lat, LON=lon, ELEV=elevation,
                DAY=dt.to_pydatetime().date(),
                IRRAD=row["IRRAD"],
                TMIN=row["TMIN"],
                TMAX=row["TMAX"],
                TEMP=row["TEMP"],
                VAP=row["VAP"],
                WIND=row["WIND"],
                RAIN=row["RAIN"],
                E0=et0[0], ES0=et0[1], ET0=et0[2],
            )
            self._store_WeatherDataContainer(wdc, dt.to_pydatetime().date())
```

### Step 3 — Run WOFOST Potential Production for Winter Wheat

```python
import os
from datetime import date
import matplotlib.pyplot as plt
from pcse.fileinput import YAMLCropDataProvider, CABOFileReader
from pcse.util import WOFOST72SiteDataProvider
from pcse.models import Wofost72_PP


def run_wofost_wheat_potential(
    weather_provider: NASAPOWERWeatherProvider,
    sowing_date: date,
    harvest_date: date,
    crop_file: str = "WOFOST81_WHEAT.crop",
    soil_file: str = "ec3.soil",
    output_path: str = "wheat_wofost.png",
) -> pd.DataFrame:
    """
    Run WOFOST 7.2 potential production (no water/nutrient stress) for wheat.

    Args:
        weather_provider: NASAPOWERWeatherProvider instance.
        sowing_date:      Date of sowing.
        harvest_date:     Latest allowable harvest date.
        crop_file:        Path to WOFOST crop parameter file.
        soil_file:        Path to CABO soil parameter file.
        output_path:      Where to save the LAI + biomass plot.

    Returns:
        DataFrame with daily simulation output: day, LAI, TWSO (dry weight of
        storage organs), TAGP (above-ground biomass), DVS (development stage).
    """
    # Crop parameters
    cropd = YAMLCropDataProvider(fpath="./pcse_data")
    cropd.set_active_crop("wheat", "Winter_wheat_101")

    # Soil parameters
    soild = CABOFileReader(soil_file)

    # Site parameters
    sited = WOFOST72SiteDataProvider(WAV=100, CO2=360)

    # Agromanagement calendar
    agro = {
        "AgroManagement": [
            {
                sowing_date: {
                    "CropCalendar": {
                        "crop_name": "wheat",
                        "variety_name": "Winter_wheat_101",
                        "crop_start_date": sowing_date,
                        "crop_start_type": "sowing",
                        "crop_end_date": harvest_date,
                        "crop_end_type": "harvest",
                        "max_duration": 300,
                    },
                    "TimedEvents": None,
                    "StateEvents": None,
                }
            },
            {harvest_date: None},
        ]
    }

    # Instantiate model
    wofost = Wofost72_PP(
        parameterprovider=cropd,
        weatherdataprovider=weather_provider,
        agromanagement=agro,
    )
    wofost.run_till_terminate()

    output = wofost.get_output()
    df_out = pd.DataFrame(output)
    df_out["day"] = pd.to_datetime(df_out["day"])
    df_out = df_out.set_index("day")

    # Summary statistics
    print("=" * 50)
    print("WOFOST Potential Yield Summary")
    print("=" * 50)
    final = df_out.iloc[-1]
    print(f"  Final grain yield (TWSO): {final.get('TWSO', 0):.1f} kg/ha dry weight")
    print(f"  Days to anthesis:         {df_out['DVS'].ge(1.0).idxmax().date()}")
    print(f"  Days to maturity:         {df_out['DVS'].ge(2.0).idxmax().date() if (df_out['DVS'] >= 2.0).any() else 'not reached'}")
    print(f"  Peak LAI:                 {df_out['LAI'].max():.2f} m²/m²")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(df_out.index, df_out["LAI"], color="#2ca02c", label="LAI (m²/m²)")
    axes[0].set_ylabel("Leaf Area Index (m²/m²)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df_out.index, df_out.get("TWSO", 0), color="#1f77b4", label="Grain (TWSO)")
    axes[1].plot(df_out.index, df_out.get("TAGP", 0), color="#ff7f0e", linestyle="--", label="Total aboveground (TAGP)")
    axes[1].set_ylabel("Dry Weight (kg/ha)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle(f"WOFOST Potential Production — Winter Wheat\nSowing: {sowing_date}", fontsize=13)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()

    return df_out
```

---

## Advanced Usage

### Yield Gap Analysis Across Production Levels

```python
from pcse.models import Wofost72_WLP_FD  # water-limited production


def yield_gap_analysis(
    weather_provider: NASAPOWERWeatherProvider,
    sowing_date: date,
    harvest_date: date,
    actual_yield_kg_ha: float,
    location_label: str = "Site",
) -> dict:
    """
    Compute potential, water-limited, and actual yield for yield gap analysis.

    Args:
        weather_provider:     Prepared NASAPOWERWeatherProvider.
        sowing_date:          Crop sowing date.
        harvest_date:         Harvest date.
        actual_yield_kg_ha:   Observed farm yield (kg/ha dry weight).
        location_label:       Display label for the site.

    Returns:
        Dictionary with Yp, Yw, Ya, gap_total, gap_water, gap_management (all kg/ha).
    """
    agro_template = {
        "AgroManagement": [
            {
                sowing_date: {
                    "CropCalendar": {
                        "crop_name": "wheat",
                        "variety_name": "Winter_wheat_101",
                        "crop_start_date": sowing_date,
                        "crop_start_type": "sowing",
                        "crop_end_date": harvest_date,
                        "crop_end_type": "harvest",
                        "max_duration": 300,
                    },
                    "TimedEvents": None,
                    "StateEvents": None,
                }
            },
            {harvest_date: None},
        ]
    }

    cropd = YAMLCropDataProvider(fpath="./pcse_data")
    cropd.set_active_crop("wheat", "Winter_wheat_101")
    soild = CABOFileReader("ec3.soil")
    sited = WOFOST72SiteDataProvider(WAV=100, CO2=360)

    # Potential production (Yp)
    wofost_pp = Wofost72_PP(
        parameterprovider=cropd,
        weatherdataprovider=weather_provider,
        agromanagement=agro_template,
    )
    wofost_pp.run_till_terminate()
    yp = wofost_pp.get_variable("TWSO")

    # Water-limited production (Yw)
    wofost_wlp = Wofost72_WLP_FD(
        parameterprovider=cropd,
        weatherdataprovider=weather_provider,
        agromanagement=agro_template,
    )
    wofost_wlp.run_till_terminate()
    yw = wofost_wlp.get_variable("TWSO")

    ya = actual_yield_kg_ha
    gap_total = yp - ya
    gap_water = yp - yw
    gap_management = yw - ya
    exploitation_factor = ya / yp * 100 if yp > 0 else 0

    result = {
        "location": location_label,
        "Yp_kg_ha": round(yp, 1),
        "Yw_kg_ha": round(yw, 1),
        "Ya_kg_ha": round(ya, 1),
        "gap_total_kg_ha": round(gap_total, 1),
        "gap_water_kg_ha": round(gap_water, 1),
        "gap_management_kg_ha": round(gap_management, 1),
        "exploitation_pct": round(exploitation_factor, 1),
    }

    print(f"\nYield Gap Analysis — {location_label}")
    print(f"  Potential (Yp):         {yp:>8.1f} kg/ha")
    print(f"  Water-limited (Yw):     {yw:>8.1f} kg/ha  [water gap: {gap_water:.1f}]")
    print(f"  Actual (Ya):            {ya:>8.1f} kg/ha  [mgmt gap:  {gap_management:.1f}]")
    print(f"  Exploitation factor:    {exploitation_factor:.1f}%")

    return result


def plot_yield_gaps(results: list[dict], output_path: str = "yield_gap_bar.png") -> None:
    """Plot stacked bar chart of Yp / water gap / management gap for multiple sites."""
    import numpy as np
    labels = [r["location"] for r in results]
    ya = [r["Ya_kg_ha"] for r in results]
    mg = [r["gap_management_kg_ha"] for r in results]
    wg = [r["gap_water_kg_ha"] for r in results]

    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, ya, width, label="Actual yield (Ya)", color="#4CAF50")
    ax.bar(x, mg, width, bottom=ya, label="Management gap", color="#FF9800")
    ax.bar(x, wg, width, bottom=[a + m for a, m in zip(ya, mg)], label="Water gap", color="#2196F3")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Yield (kg/ha dry weight)")
    ax.set_title("Yield Gap Decomposition by Site")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Yield gap chart saved to {output_path}")
    plt.close()
```

### Climate Scenario: +2°C Temperature Shift

```python
def apply_temperature_scenario(
    weather_df: pd.DataFrame,
    delta_tmax: float = 2.0,
    delta_tmin: float = 2.0,
    delta_rain_fraction: float = 0.0,
) -> pd.DataFrame:
    """
    Apply a simple climate perturbation to a baseline weather DataFrame.

    Args:
        weather_df:          Baseline weather from fetch_nasa_power().
        delta_tmax:          Temperature increase for TMAX (°C).
        delta_tmin:          Temperature increase for TMIN (°C).
        delta_rain_fraction: Fractional change in precipitation (0.0 = no change,
                             -0.1 = -10%, +0.2 = +20%).

    Returns:
        Modified copy of weather_df with shifted climate variables.
    """
    df_scenario = weather_df.copy()
    df_scenario["TMAX"] = df_scenario["TMAX"] + delta_tmax
    df_scenario["TMIN"] = df_scenario["TMIN"] + delta_tmin
    df_scenario["TEMP"] = (df_scenario["TMAX"] + df_scenario["TMIN"]) / 2
    df_scenario["RAIN"] = df_scenario["RAIN"] * (1.0 + delta_rain_fraction)
    # Clip rain to non-negative
    df_scenario["RAIN"] = df_scenario["RAIN"].clip(lower=0.0)
    # Adjust vapour pressure with Clausius-Clapeyron (approx 7%/°C)
    mean_delta = (delta_tmax + delta_tmin) / 2
    df_scenario["VAP"] = df_scenario["VAP"] * (1 + 0.07 * mean_delta)
    return df_scenario
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `KeyError: ALLSKY_SFC_SW_DWN` in NASA response | API parameter name changed | Check https://power.larc.nasa.gov/api/pages/ for current names |
| `WeatherDataProvider` returns None for a day | Missing record in download | Use `fillna(method='ffill')` before wrapping |
| WOFOST terminates at DVS=2.0 too early | Sowing date too late for season | Adjust `crop_start_date` or reduce latitude |
| `AttributeError: 'NoneType' object has no attribute 'get'` | Crop file not loaded | Call `cropd.set_active_crop(crop, variety)` before model init |
| Negative TWSO at end of run | Soil parameter mismatch | Verify soil hydraulic properties in `.soil` file |
| NASA POWER returns -999.0 fill values | No satellite data for period | Filter rows where IRRAD < 0 and interpolate |
| `pcse.exceptions.WeatherDataProviderError` | Date out of data range | Ensure start/end dates are within 1984-present |

---

## External Resources

- PCSE documentation: https://pcse.readthedocs.io/
- WOFOST crop parameter database: https://github.com/ajwdewit/WOFOST_crop_parameters
- NASA POWER API: https://power.larc.nasa.gov/api/pages/
- GYGA (Global Yield Gap Atlas): https://www.yieldgap.org/
- AquaCrop-OS Python: https://aquacropos.github.io/
- FAO GAEZ (Agro-Ecological Zones) database: https://gaez.fao.org/

---

## Examples

### Example 1 — Complete WOFOST Run for Wheat in Kansas

```python
from datetime import date
import pandas as pd


def example_kansas_wheat():
    """End-to-end WOFOST potential production run for Kansas winter wheat."""
    # Kansas State (39.19°N, -96.58°E), elevation 320 m
    lat, lon, elev = 39.19, -96.58, 320.0
    season_start = date(2022, 10, 1)
    season_end   = date(2023, 8, 31)

    print("1. Downloading NASA POWER weather data ...")
    weather_df = fetch_nasa_power(lat, lon, season_start, season_end)
    print(f"   Downloaded {len(weather_df)} days of weather data.")

    print("2. Building PCSE weather provider ...")
    wp = NASAPOWERWeatherProvider(weather_df, lat, lon, elev)

    print("3. Running WOFOST potential production ...")
    df_sim = run_wofost_wheat_potential(
        weather_provider=wp,
        sowing_date=date(2022, 10, 15),
        harvest_date=date(2023, 7, 15),
        output_path="kansas_wheat_pp.png",
    )

    print(f"\n   Simulation length: {len(df_sim)} days")
    print(f"   Final grain yield: {df_sim['TWSO'].iloc[-1]:.1f} kg/ha")
    return df_sim


if __name__ == "__main__":
    df_result = example_kansas_wheat()
```

### Example 2 — Multi-Site Yield Gap Analysis

```python
def example_multi_site_yield_gap():
    """Compare yield gaps across three contrasting wheat-growing sites."""
    sites = [
        {"label": "Kansas (US)",      "lat": 39.19, "lon": -96.58, "elev": 320, "ya": 3800},
        {"label": "Punjab (India)",   "lat": 30.90, "lon": 75.85,  "elev": 234, "ya": 4200},
        {"label": "Hebei (China)",    "lat": 38.03, "lon": 114.47, "elev":  55, "ya": 6100},
    ]
    sowing = date(2022, 11, 1)
    harvest = date(2023, 6, 30)

    gap_results = []
    for site in sites:
        print(f"\nProcessing {site['label']} ...")
        wdf = fetch_nasa_power(site["lat"], site["lon"], date(2022, 10, 1), date(2023, 7, 31))
        wp  = NASAPOWERWeatherProvider(wdf, site["lat"], site["lon"], site["elev"])
        res = yield_gap_analysis(wp, sowing, harvest, site["ya"], site["label"])
        gap_results.append(res)

    plot_yield_gaps(gap_results, "multi_site_yield_gap.png")
    return pd.DataFrame(gap_results)


if __name__ == "__main__":
    df_gaps = example_multi_site_yield_gap()
    print(df_gaps.to_string(index=False))
```

### Example 3 — Climate Scenario Comparison

```python
def example_climate_scenario_comparison():
    """Compare baseline vs +2°C scenario yield for a single site."""
    lat, lon, elev = 39.19, -96.58, 320.0
    start, end = date(2022, 10, 1), date(2023, 8, 31)

    wdf_baseline = fetch_nasa_power(lat, lon, start, end)
    wdf_plus2    = apply_temperature_scenario(wdf_baseline, delta_tmax=2.0, delta_tmin=2.0)

    scenarios = {
        "Baseline": wdf_baseline,
        "+2°C":     wdf_plus2,
    }

    yields = {}
    for label, wdf in scenarios.items():
        wp = NASAPOWERWeatherProvider(wdf, lat, lon, elev)
        model = Wofost72_PP(
            parameterprovider=YAMLCropDataProvider(fpath="./pcse_data"),
            weatherdataprovider=wp,
            agromanagement={
                "AgroManagement": [
                    {date(2022, 10, 15): {
                        "CropCalendar": {
                            "crop_name": "wheat", "variety_name": "Winter_wheat_101",
                            "crop_start_date": date(2022, 10, 15),
                            "crop_start_type": "sowing",
                            "crop_end_date": date(2023, 7, 15),
                            "crop_end_type": "harvest",
                            "max_duration": 300,
                        },
                        "TimedEvents": None, "StateEvents": None,
                    }},
                    {date(2023, 7, 15): None},
                ]
            },
        )
        model.run_till_terminate()
        yields[label] = model.get_variable("TWSO")

    print("\nClimate scenario yield comparison:")
    for label, y in yields.items():
        print(f"  {label:12s}: {y:.1f} kg/ha")
    change = (yields["+2°C"] - yields["Baseline"]) / yields["Baseline"] * 100
    print(f"  Yield change: {change:+.1f}%")
    return yields
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — WOFOST PP/WLP, NASA POWER download, yield gap, climate scenario |
