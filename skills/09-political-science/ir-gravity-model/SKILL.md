---
name: ir-gravity-model
description: >
  Use this Skill for international relations quantitative research: gravity trade model (PPML),
  COW bilateral trade, RTA trade effects, and GDELT event data analysis.
tags:
  - political-science
  - international-relations
  - gravity-model
  - PPML
  - trade
  - GDELT
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
  - numpy>=1.23
  - matplotlib>=3.6
  - requests>=2.28
last_updated: "2026-03-18"
status: "stable"
---

# International Relations — Gravity Model and GDELT Analysis

## When to Use

Use this skill when you need to:

- Estimate the gravity equation of trade using PPML (Poisson Pseudo-Maximum Likelihood) to handle
  zero bilateral trade flows and heteroscedastic residuals
- Load and process COW (Correlates of War) bilateral trade data for directed dyad-year analysis
- Estimate the trade-creation effect of Regional Trade Agreements (RTAs) while controlling for
  multilateral resistance (importer and exporter fixed effects)
- Compare PPML vs. log-linear OLS and understand why OLS on log(trade) is biased with zeros
- Download and analyze GDELT bilateral event data (cooperation/conflict tone between country dyads)
- Integrate MID (Militarized Interstate Disputes) data for conflict-trade links

This skill assumes familiarity with fixed effects regression. For panel methods see the
`comparative-politics` skill; for conflict event data see the `conflict-data` skill.

## Background

**The Gravity Equation**: Bilateral trade X_ij is proportional to the economic mass of the
exporter (GDP_i) and importer (GDP_j), and inversely proportional to trade costs (captured by
bilateral distance and other barriers):

```
X_ij = exp(a_i + a_j + β_1 log(dist_ij) + β_2 contiguity + β_3 language + β_4 RTA + ε_ij)
```

Exporter FE (a_i) and importer FE (a_j) control for multilateral resistance (Anderson & van
Wincoop 2003). Without these, the estimates of RTA effects are biased.

**Why PPML?** Santos-Silva & Tenreyro (2006) showed:
1. log(X_ij) is undefined when X_ij = 0 (and many dyads have zero trade)
2. OLS on log(X_ij) is inconsistent when errors are heteroscedastic
3. Poisson GLM with log link (PPML) is consistent under weak regularity conditions

PPML minimizes the Poisson deviance: Σ[X_ij - exp(Xβ)]² / Var, which for Poisson equals the
count itself. The quasi-Poisson variant relaxes the mean=variance assumption.

**RTA effects**: Trade creation (intra-bloc trade increases) and trade diversion (imports shift
from non-members to members). To capture both:
- RTA_same = 1 if both i and j are in the same RTA
- RTA_one = 1 if only one partner is in the RTA (diversion measure)

**GDELT (Global Database of Events, Language, and Tone)**: Machine-coded news events using CAMEO
codes. Available from 1979 (GDELT 1.0) and 2015+ (GDELT 2.0, 15-minute updates). Key variables:
- Actor1CountryCode, Actor2CountryCode
- EventCode (CAMEO: 01=verbal cooperation, 19=use of force)
- GoldsteinScale: conflict-cooperation score (-10 to +10)
- NumMentions, NumArticles (media salience)

**Dyadic aggregation for IR**: Aggregate GDELT to monthly or annual directed dyad (i→j) events,
compute mean Goldstein scale per dyad-period, and merge with bilateral trade or MID data.

## Environment Setup

```bash
pip install pandas>=1.5 statsmodels>=0.14 numpy>=1.23 matplotlib>=3.6 requests>=2.28
```

COW Trade data: https://correlatesofwar.org/data-sets/bilateral-trade/
GDELT: https://www.gdeltproject.org/data.html (direct CSV or BigQuery)
CEPII GeoDist: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6

```bash
export COW_TRADE_PATH="/data/Dyadic_COW_4.0.csv"
export GEODIST_PATH="/data/dist_cepii.dta"
export GDELT_DIR="/data/gdelt/"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Load COW Bilateral Trade Data
# ---------------------------------------------------------------------------

COW_COLS = [
    "ccode1", "ccode2", "year",
    "flow1",   # Exports from ccode1 to ccode2 (millions current USD)
    "flow2",   # Exports from ccode2 to ccode1
    "smoothflow1", "smoothflow2",  # Smoothed flows
]


def load_cow_trade(
    path: str,
    years: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """
    Load COW bilateral trade data and create a directed dyad dataset.

    Parameters
    ----------
    path : str
        Path to Dyadic_COW_4.0.csv.
    years : tuple, optional
        Year range filter.

    Returns
    -------
    pd.DataFrame with directed dyad-year rows (two rows per undirected dyad-year):
    columns: importer, exporter, year, trade (millions USD).
    """
    df = pd.read_csv(path, usecols=[c for c in COW_COLS if c in pd.read_csv(path, nrows=0).columns],
                     low_memory=False)
    if years:
        df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]

    flow_col = "smoothflow1" if "smoothflow1" in df.columns else "flow1"

    # Create two directed rows per undirected dyad
    dir1 = df[["ccode1", "ccode2", "year", flow_col]].copy()
    dir1.columns = ["exporter", "importer", "year", "trade"]
    dir2 = df[["ccode2", "ccode1", "year", "smoothflow2" if "smoothflow2" in df.columns else "flow2"]].copy()
    dir2.columns = ["exporter", "importer", "year", "trade"]

    directed = pd.concat([dir1, dir2], ignore_index=True)
    directed["trade"] = pd.to_numeric(directed["trade"], errors="coerce").fillna(0).clip(lower=0)
    return directed.sort_values(["exporter", "importer", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Build Gravity Dataset
# ---------------------------------------------------------------------------

def build_gravity_dataset(
    trade_df: pd.DataFrame,
    geodist_df: pd.DataFrame,
    rta_df: pd.DataFrame | None = None,
    gdp_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge trade flows with bilateral controls from CEPII GeoDist.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Directed dyad-year trade (exporter, importer, year, trade).
    geodist_df : pd.DataFrame
        CEPII GeoDist (iso_o, iso_d, distw, contig, comlang_off, colony).
    rta_df : pd.DataFrame, optional
        RTA dataset with (iso1, iso2, year, rta_in_force).
    gdp_df : pd.DataFrame, optional
        GDP data (iso3, year, gdp_current_usd).

    Returns
    -------
    pd.DataFrame — merged gravity dataset ready for PPML.
    """
    df = trade_df.copy()

    # Merge bilateral controls
    if "iso_o" in geodist_df.columns and "iso_d" in geodist_df.columns:
        geo_keep = [c for c in ["iso_o", "iso_d", "distw", "dist", "contig",
                                  "comlang_off", "colony", "comcol"] if c in geodist_df.columns]
        df = df.merge(
            geodist_df[geo_keep].rename(columns={"iso_o": "exporter", "iso_d": "importer"}),
            on=["exporter", "importer"],
            how="left",
        )

    # Merge GDP
    if gdp_df is not None:
        for partner, side in [("exporter", "exp"), ("importer", "imp")]:
            df = df.merge(
                gdp_df.rename(columns={"iso3": partner, "gdp": f"gdp_{side}"}),
                on=[partner, "year"],
                how="left",
            )

    # Merge RTA
    if rta_df is not None:
        df = df.merge(rta_df, on=["exporter", "importer", "year"], how="left")
        df["rta"] = df.get("rta_in_force", 0).fillna(0)

    # Log distance
    if "distw" in df.columns:
        df["log_dist"] = np.log(df["distw"].replace(0, np.nan))
    if "gdp_exp" in df.columns:
        df["log_gdp_exp"] = np.log(df["gdp_exp"].clip(lower=1))
    if "gdp_imp" in df.columns:
        df["log_gdp_imp"] = np.log(df["gdp_imp"].clip(lower=1))

    return df


# ---------------------------------------------------------------------------
# 3. PPML Gravity Estimation
# ---------------------------------------------------------------------------

def ppml_gravity(
    df: pd.DataFrame,
    outcome: str = "trade",
    continuous_vars: list[str] | None = None,
    dummy_vars: list[str] | None = None,
    exporter_fe: bool = True,
    importer_fe: bool = True,
    year_fe: bool = True,
    exporter_col: str = "exporter",
    importer_col: str = "importer",
    year_col: str = "year",
) -> object:
    """
    Estimate a PPML gravity model via Poisson GLM.

    Parameters
    ----------
    df : pd.DataFrame
    outcome : str
        Trade flow variable (can be zero).
    continuous_vars : list of str
        Continuous log-scale bilateral variables (e.g., log_dist).
    dummy_vars : list of str
        Binary bilateral controls (contig, comlang_off, colony, rta).
    exporter_fe, importer_fe, year_fe : bool
        Whether to include fixed effects via dummies.

    Returns
    -------
    statsmodels GLMResults (Poisson family).
    """
    continuous_vars = continuous_vars or ["log_dist"]
    dummy_vars = dummy_vars or []

    # Drop rows with missing outcome or predictors
    sub = df[[outcome] + continuous_vars + dummy_vars +
              [exporter_col, importer_col, year_col]].dropna().copy()
    sub[outcome] = sub[outcome].clip(lower=0)

    # Build FE dummies
    fe_frames = []
    if exporter_fe:
        exp_dum = pd.get_dummies(sub[exporter_col], prefix="exp", drop_first=True)
        fe_frames.append(exp_dum.astype(float))
    if importer_fe:
        imp_dum = pd.get_dummies(sub[importer_col], prefix="imp", drop_first=True)
        fe_frames.append(imp_dum.astype(float))
    if year_fe:
        yr_dum = pd.get_dummies(sub[year_col], prefix="yr", drop_first=True)
        fe_frames.append(yr_dum.astype(float))

    X = sub[continuous_vars + dummy_vars].reset_index(drop=True).astype(float)
    if fe_frames:
        X = pd.concat([X] + [f.reset_index(drop=True) for f in fe_frames], axis=1)
    X = sm.add_constant(X)
    y = sub[outcome].reset_index(drop=True)

    model = sm.GLM(y, X, family=sm.families.Poisson())
    result = model.fit(maxiter=100)
    return result


def ppml_coeff_table(result, vars_of_interest: list[str]) -> pd.DataFrame:
    """
    Extract PPML coefficients with semi-elasticity interpretation.

    Returns
    -------
    pd.DataFrame with coef, se, z, p, semi_elasticity (exp(coef)-1 for dummies).
    """
    tbl = pd.DataFrame({
        "coef": result.params,
        "se": result.bse,
        "z": result.tvalues,
        "p": result.pvalues,
    })
    # Restrict to variables of interest
    tbl = tbl.loc[tbl.index.isin(vars_of_interest + ["const"])]
    tbl["semi_elas"] = np.exp(tbl["coef"]) - 1
    return tbl.round(4)


# ---------------------------------------------------------------------------
# 4. GDELT Event Data Download and Aggregation
# ---------------------------------------------------------------------------

GDELT_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year",
    "Actor1CountryCode", "Actor2CountryCode",
    "EventCode", "EventBaseCode", "EventRootCode",
    "GoldsteinScale", "NumMentions", "NumArticles", "NumSources",
    "AvgTone",
]


def download_gdelt_csv(date_str: str, gdelt_dir: str) -> pd.DataFrame | None:
    """
    Download a single GDELT 1.0 day file from the GDELT website.

    Parameters
    ----------
    date_str : str
        Date in 'YYYYMMDD' format (e.g., '20230101').
    gdelt_dir : str
        Local directory to cache files.

    Returns
    -------
    pd.DataFrame or None if download fails.
    """
    import os, requests, zipfile, io
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    local_csv = os.path.join(gdelt_dir, f"{date_str}.csv")

    if os.path.exists(local_csv):
        return pd.read_csv(local_csv, sep="\t", header=None,
                           names=GDELT_COLUMNS, usecols=range(len(GDELT_COLUMNS)),
                           low_memory=False)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        fname = z.namelist()[0]
        with z.open(fname) as f:
            df = pd.read_csv(f, sep="\t", header=None,
                             names=GDELT_COLUMNS, usecols=range(len(GDELT_COLUMNS)),
                             low_memory=False)
        df.to_csv(local_csv, index=False)
        return df
    except Exception as exc:
        print(f"Warning: could not download {url}: {exc}")
        return None


def gdelt_dyadic_aggregation(
    df: pd.DataFrame,
    country1: str,
    country2: str,
    freq: str = "M",
) -> pd.DataFrame:
    """
    Aggregate GDELT events for a specific bilateral dyad by time period.

    Parameters
    ----------
    df : pd.DataFrame
        GDELT event dataframe with Actor1CountryCode, Actor2CountryCode.
    country1, country2 : str
        ISO2 country codes.
    freq : str
        Aggregation frequency ('M' = monthly, 'Y' = annual).

    Returns
    -------
    pd.DataFrame with period, n_events, mean_goldstein, mean_tone.
    """
    mask = (
        (df["Actor1CountryCode"] == country1) & (df["Actor2CountryCode"] == country2)
    ) | (
        (df["Actor1CountryCode"] == country2) & (df["Actor2CountryCode"] == country1)
    )
    sub = df[mask].copy()
    sub["Day"] = pd.to_datetime(sub["Day"].astype(str), format="%Y%m%d", errors="coerce")
    sub = sub.dropna(subset=["Day"])
    sub["period"] = sub["Day"].dt.to_period(freq)
    sub["GoldsteinScale"] = pd.to_numeric(sub["GoldsteinScale"], errors="coerce")
    sub["AvgTone"] = pd.to_numeric(sub["AvgTone"], errors="coerce")

    agg = sub.groupby("period").agg(
        n_events=("GlobalEventID", "count"),
        mean_goldstein=("GoldsteinScale", "mean"),
        mean_tone=("AvgTone", "mean"),
        n_conflict=("GoldsteinScale", lambda x: (x < 0).sum()),
        n_coop=("GoldsteinScale", lambda x: (x > 0).sum()),
    ).reset_index()
    return agg
```

## Advanced Usage

### RTA Trade Effect and Comparison with OLS

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def compare_ppml_ols(df: pd.DataFrame, outcome: str = "trade",
                      continuous_vars: list[str] | None = None,
                      dummy_vars: list[str] | None = None) -> pd.DataFrame:
    """
    Fit both PPML and log-OLS and compare coefficients on shared variables.

    Returns
    -------
    pd.DataFrame comparing coefficients from PPML and OLS.
    """
    continuous_vars = continuous_vars or ["log_dist", "log_gdp_exp", "log_gdp_imp"]
    dummy_vars = dummy_vars or ["contig", "comlang_off", "rta"]

    # PPML
    ppml_res = ppml_gravity(df, outcome=outcome,
                             continuous_vars=continuous_vars, dummy_vars=dummy_vars,
                             exporter_fe=True, importer_fe=True, year_fe=False)

    # Log-OLS (drop zeros)
    sub_ols = df[df[outcome] > 0].copy()
    sub_ols["log_trade"] = np.log(sub_ols[outcome])
    avail = [v for v in continuous_vars + dummy_vars if v in sub_ols.columns]
    exp_dum = pd.get_dummies(sub_ols["exporter"], prefix="exp", drop_first=True).astype(float)
    imp_dum = pd.get_dummies(sub_ols["importer"], prefix="imp", drop_first=True).astype(float)
    X_ols = pd.concat([sub_ols[avail].reset_index(drop=True),
                        exp_dum.reset_index(drop=True),
                        imp_dum.reset_index(drop=True)], axis=1)
    X_ols = sm.add_constant(X_ols)
    ols_res = sm.OLS(sub_ols["log_trade"].reset_index(drop=True), X_ols).fit(
        cov_type="HC3"
    )

    vars_compare = ["const"] + continuous_vars + dummy_vars
    ppml_coefs = ppml_res.params.reindex(vars_compare)
    ols_coefs = ols_res.params.reindex(vars_compare)

    comparison = pd.DataFrame({
        "PPML_coef": ppml_coefs.round(4),
        "OLS_coef": ols_coefs.round(4),
        "Difference": (ppml_coefs - ols_coefs).round(4),
    }).dropna(how="all")
    return comparison
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| PPML does not converge | Many zeros or near-separation | Scale trade by 1000; start from OLS solution; reduce FE dimensions |
| RTA coefficient implausible | Endogeneity (countries sign RTAs selectively) | Use pair FE + year FE; control for lagged trade |
| Log-OLS on zero-trade dyads | log(0) undefined | Use PPML instead; or add a small constant (introduces bias) |
| GDELT download fails | Large files; rate limits | Use GDELT 2.0 API or Google BigQuery for efficient access |
| Bilateral distance missing | COW codes vs. ISO codes | Map COW codes to ISO3 using the COW country code crosswalk |
| Exporter/importer FE matrix singular | Perfect collinearity | Drop one category from each FE group; `linearmodels` handles this automatically |

## External Resources

- Santos-Silva, J.M.C. & Tenreyro, S. (2006). The log of gravity. *Review of Economics and Statistics*, 88(4), 641-658.
- Anderson, J. & van Wincoop, E. (2003). Gravity with gravitas. *American Economic Review*, 93(1), 170-192.
- COW Bilateral Trade v4.0: https://correlatesofwar.org/data-sets/bilateral-trade/
- CEPII GeoDist: https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6
- GDELT Project: https://www.gdeltproject.org/
- WTO RTA Database: https://rtais.wto.org/

## Examples

### Example 1: PPML Gravity Estimation + Coefficient Table

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n_dyads = 1500
exporters = [f"C{i:02d}" for i in range(1, 26)]
importers = [f"C{i:02d}" for i in range(1, 26)]
years_g = [2015, 2018, 2021]

records = []
for yr in years_g:
    for exp in exporters:
        for imp in importers:
            if exp == imp:
                continue
            dist = np.exp(rng.uniform(4, 9))
            contig = int(rng.random() < 0.05)
            lang = int(rng.random() < 0.15)
            rta = int(rng.random() < 0.20)
            gdp_e = np.exp(rng.uniform(22, 30))
            gdp_i = np.exp(rng.uniform(22, 30))
            # True gravity
            log_trade = (0.8 * np.log(gdp_e) + 0.7 * np.log(gdp_i)
                         - 1.1 * np.log(dist) + 0.5 * contig + 0.4 * lang
                         + 0.6 * rta + rng.normal(0, 1.5))
            trade = np.exp(log_trade) if rng.random() > 0.25 else 0
            records.append({
                "exporter": exp, "importer": imp, "year": yr,
                "trade": trade, "log_dist": np.log(dist),
                "contig": contig, "comlang_off": lang, "rta": rta,
                "log_gdp_exp": np.log(gdp_e), "log_gdp_imp": np.log(gdp_i),
            })

df_gravity = pd.DataFrame(records)
print(f"Zero-trade dyads: {(df_gravity['trade'] == 0).mean():.1%}")

ppml_result = ppml_gravity(
    df_gravity,
    outcome="trade",
    continuous_vars=["log_dist", "log_gdp_exp", "log_gdp_imp"],
    dummy_vars=["contig", "comlang_off", "rta"],
    exporter_fe=True, importer_fe=True, year_fe=True,
)

coeff_tbl = ppml_coeff_table(
    ppml_result,
    ["log_dist", "log_gdp_exp", "log_gdp_imp", "contig", "comlang_off", "rta"],
)
print("=== PPML Gravity Estimates ===")
print(coeff_tbl.to_string())
```

### Example 2: RTA Effect — PPML vs. OLS Comparison

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Using df_gravity from Example 1
comparison = compare_ppml_ols(
    df_gravity,
    outcome="trade",
    continuous_vars=["log_dist", "log_gdp_exp", "log_gdp_imp"],
    dummy_vars=["contig", "comlang_off", "rta"],
)
print("=== PPML vs OLS Coefficient Comparison ===")
print(comparison.to_string())

# Plot comparison
vars_plot = ["log_dist", "log_gdp_exp", "log_gdp_imp", "contig", "comlang_off", "rta"]
comp_plot = comparison.loc[comparison.index.isin(vars_plot)].dropna()
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(comp_plot))
ax.bar([xi - 0.2 for xi in x], comp_plot["PPML_coef"], width=0.35,
       label="PPML", color="#2166ac", alpha=0.85)
ax.bar([xi + 0.2 for xi in x], comp_plot["OLS_coef"], width=0.35,
       label="Log-OLS", color="#d73027", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(list(x))
ax.set_xticklabels(comp_plot.index.tolist(), rotation=20, ha="right")
ax.set_ylabel("Coefficient")
ax.set_title("PPML vs. Log-OLS Gravity Coefficients")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("ppml_vs_ols.png", dpi=150)
plt.show()
```

### Example 3: GDELT Bilateral Event Time Series

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
n_events = 5000
dates = pd.date_range("2015-01-01", "2022-12-31", periods=n_events)
cameo_goldstein = {
    "01": 3.0, "02": 4.0, "03": 5.0, "04": 7.0,
    "10": -2.0, "13": -5.0, "14": -7.0, "19": -10.0,
}
codes = rng.choice(list(cameo_goldstein.keys()), n_events)
goldstein = [cameo_goldstein[c] + rng.normal(0, 0.5) for c in codes]

df_gdelt_sim = pd.DataFrame({
    "GlobalEventID": range(n_events),
    "Day": dates.strftime("%Y%m%d"),
    "Actor1CountryCode": rng.choice(["US", "CN", "RU"], n_events, p=[0.4, 0.35, 0.25]),
    "Actor2CountryCode": rng.choice(["US", "CN", "RU"], n_events, p=[0.25, 0.4, 0.35]),
    "EventCode": codes,
    "GoldsteinScale": goldstein,
    "AvgTone": [g * 0.8 + rng.normal(0, 1) for g in goldstein],
    "NumMentions": rng.integers(1, 50, n_events),
    "NumArticles": rng.integers(1, 20, n_events),
    "NumSources": rng.integers(1, 10, n_events),
})
# Remove self-dyads
df_gdelt_sim = df_gdelt_sim[df_gdelt_sim["Actor1CountryCode"] != df_gdelt_sim["Actor2CountryCode"]]

us_cn = gdelt_dyadic_aggregation(df_gdelt_sim, "US", "CN", freq="M")
us_ru = gdelt_dyadic_aggregation(df_gdelt_sim, "US", "RU", freq="M")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for agg_df, label, color in [(us_cn, "US-China", "#1f77b4"), (us_ru, "US-Russia", "#d62728")]:
    agg_df["period_dt"] = agg_df["period"].dt.to_timestamp()
    axes[0].plot(agg_df["period_dt"], agg_df["mean_goldstein"],
                 label=label, color=color, linewidth=1.5)
    axes[1].bar(agg_df["period_dt"], agg_df["n_events"],
                label=label, color=color, alpha=0.5, width=20)

axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_ylabel("Mean Goldstein Scale")
axes[0].set_title("GDELT Bilateral Event Tone: US-China vs US-Russia")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].set_ylabel("Monthly Event Count")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gdelt_bilateral_events.png", dpi=150)
plt.show()
```
