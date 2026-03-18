---
name: electoral-data
description: >
  Use this Skill for electoral analysis: CLEA election data, effective number of parties,
  ecological inference, turnout modeling, and gerrymandering metrics.
tags:
  - political-science
  - elections
  - ecological-inference
  - gerrymandering
  - voting-behavior
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
  - numpy>=1.23
  - scipy>=1.9
  - matplotlib>=3.6
last_updated: "2026-03-18"
status: "stable"
---

# Electoral Data Analysis

## When to Use

Use this skill when you need to:

- Load and parse CLEA (Comparative Legislative Elections Archive) data for cross-national vote
  share analysis
- Compute the effective number of parties (Laakso-Taagepera ENP) for seats and votes
- Measure electoral disproportionality (Gallagher index, Loosemore-Hanby)
- Estimate ecological regression or King's EI for group-level voting patterns when individual
  data are unavailable
- Model voter turnout as a function of district competitiveness, demographic composition, and
  institutional factors
- Calculate gerrymandering metrics: efficiency gap, Polsby-Popper compactness, convex hull ratio,
  and mean-median difference

This skill covers national and sub-national election data from parliamentary and presidential
systems.

## Background

**Effective Number of Parties (ENP)**: The Laakso-Taagepera (1979) index compresses a multi-party
distribution into a single number:

```
ENP_votes = 1 / Σ(v_i²)
ENP_seats = 1 / Σ(s_i²)
```

where `v_i` and `s_i` are vote/seat shares as decimals. ENP = 1 in a single-party system; higher
values indicate more fragmentation. Values above 5 suggest highly fragmented parliaments.

**Gallagher Disproportionality Index**: Captures the deviation between vote and seat shares:

```
G = sqrt( 0.5 * Σ(v_i - s_i)² )
```

Values below 5 indicate proportional systems; above 10 indicate majoritarian distortion.

**Ecological Regression**: When we know group proportions per district (e.g., percent Black) and
district-level outcomes (e.g., percent voting Democratic), we can estimate group voting rates via
OLS with the constraint that estimated rates lie in [0,1]. King's EI (1997) relaxes the linearity
assumption and provides individual-level estimates.

**Voter Turnout Modeling**: Turnout in a district is a function of:
- *Competitiveness*: margin in the previous election (closer races → higher turnout)
- *Demographics*: median age, education share, income
- *Institutional*: voting ease laws, early voting, registration rules

**Efficiency Gap**: Wasted votes are those cast for losing candidates plus those beyond the
threshold needed to win. The efficiency gap measures systematic packing/cracking:

```
EG = (Wasted_R - Wasted_D) / Total_votes
```

Values above ±0.08 are considered legally significant gerrymandering thresholds in U.S. courts.

**Polsby-Popper Compactness**:

```
PP = 4π × Area / Perimeter²
```

A perfect circle scores 1.0. Gerrymandered districts tend toward elongated shapes with lower PP.

## Environment Setup

```bash
pip install pandas>=1.5 geopandas>=0.13 numpy>=1.23 scipy>=1.9 matplotlib>=3.6
```

CLEA data is available from https://electiondataarchive.org/. Download the CSV file
`CLEA_lc_20230424.csv`. Store paths as environment variables:

```bash
export CLEA_PATH="/data/CLEA_lc_20230424.csv"
export DISTRICT_SHP="/data/districts.shp"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. CLEA Data Loading
# ---------------------------------------------------------------------------

CLEA_COLS = [
    "ctr_n",   # Country name
    "ctr",     # Country code
    "yr",      # Year
    "mn",      # Month
    "mn_n",    # Election name
    "pty_n",   # Party name
    "pvs",     # Party vote share (percent)
    "pss",     # Party seat share (percent)
    "vot",     # Total votes cast
    "seat",    # Seats won
    "sts",     # Total seats
    "to1",     # Turnout (percent)
]


def load_clea(filepath: str, countries: list[str] | None = None,
              years: tuple[int, int] | None = None) -> pd.DataFrame:
    """
    Load CLEA lower chamber election data.

    Parameters
    ----------
    filepath : str
        Path to CLEA CSV file.
    countries : list of str, optional
        Filter by ctr_n (country name).
    years : tuple (start, end), optional
        Inclusive year filter.

    Returns
    -------
    pd.DataFrame with party-level records.
    """
    available_cols = pd.read_csv(filepath, nrows=0).columns.tolist()
    use_cols = [c for c in CLEA_COLS if c in available_cols]
    df = pd.read_csv(filepath, usecols=use_cols, low_memory=False)
    if countries:
        df = df[df["ctr_n"].isin(countries)]
    if years:
        df = df[(df["yr"] >= years[0]) & (df["yr"] <= years[1])]
    # Convert shares to decimals
    for col in ["pvs", "pss"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Effective Number of Parties and Disproportionality
# ---------------------------------------------------------------------------

def effective_number_of_parties(shares: pd.Series) -> float:
    """
    Laakso-Taagepera ENP = 1 / sum(p_i^2).

    Parameters
    ----------
    shares : pd.Series
        Party shares as proportions (sum to ~1.0). NaN ignored.

    Returns
    -------
    float
    """
    s = shares.dropna()
    s = s[s > 0]
    if s.sum() < 0.9:
        s = s / s.sum()
    return 1.0 / (s ** 2).sum()


def gallagher_index(vote_shares: pd.Series, seat_shares: pd.Series) -> float:
    """
    Gallagher least-squares disproportionality index.

    Parameters
    ----------
    vote_shares, seat_shares : pd.Series
        Party-level vote and seat shares (proportions). Must be aligned.

    Returns
    -------
    float
    """
    diff = (vote_shares - seat_shares).dropna()
    return np.sqrt(0.5 * (diff ** 2).sum())


def election_summary(df: pd.DataFrame, country: str, year: int) -> dict:
    """
    Compute ENP (votes), ENP (seats), and Gallagher index for one election.

    Parameters
    ----------
    df : pd.DataFrame
        CLEA-format dataframe.
    country : str
        Country name.
    year : int
        Election year.

    Returns
    -------
    dict with enp_votes, enp_seats, gallagher, n_parties, turnout.
    """
    sub = df[(df["ctr_n"] == country) & (df["yr"] == year)].copy()
    sub = sub.dropna(subset=["pvs"])
    if sub.empty:
        return {}
    enp_v = effective_number_of_parties(sub["pvs"])
    enp_s = effective_number_of_parties(sub["pss"]) if "pss" in sub else np.nan
    g = gallagher_index(sub["pvs"], sub["pss"]) if "pss" in sub else np.nan
    turnout = sub["to1"].iloc[0] if "to1" in sub.columns else np.nan
    return {
        "country": country,
        "year": year,
        "enp_votes": round(enp_v, 3),
        "enp_seats": round(enp_s, 3),
        "gallagher": round(g, 3),
        "n_parties": len(sub),
        "turnout": turnout,
    }


# ---------------------------------------------------------------------------
# 3. Ecological Regression
# ---------------------------------------------------------------------------

def ecological_regression(
    group_share: np.ndarray,
    outcome_share: np.ndarray,
    group_name: str = "Group",
    outcome_name: str = "Outcome",
    plot: bool = True,
    save_path: str | None = None,
) -> dict:
    """
    Estimate group voting rates via ecological regression (Goodman 1953).

    Model: outcome_i = b_G * group_share_i + b_OG * (1 - group_share_i) + e_i
    where b_G = voting rate of group, b_OG = voting rate of out-group.

    Parameters
    ----------
    group_share : np.ndarray
        Proportion of group in each geographic unit (districts).
    outcome_share : np.ndarray
        Proportion voting for candidate/party in each unit.
    group_name, outcome_name : str
        Labels for reporting.
    plot : bool
        Whether to produce a scatter plot with regression line.
    save_path : str, optional
        Save path for the plot.

    Returns
    -------
    dict: b_group (voting rate), b_outgroup, r_squared, slope, intercept
    """
    mask = ~(np.isnan(group_share) | np.isnan(outcome_share))
    x, y = group_share[mask], outcome_share[mask]

    slope, intercept, r, p, se = stats.linregress(x, y)

    # Goodman estimates: b_G = slope + intercept ... wait, correct formula:
    # E[Y] = intercept + slope * X
    # At X=1: b_group = intercept + slope
    # At X=0: b_outgroup = intercept
    b_group = intercept + slope
    b_outgroup = intercept

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, alpha=0.6, color="#2166ac", edgecolors="white", s=50)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "r-", linewidth=2,
                label=f"OLS (R²={r**2:.3f})")
        ax.set_xlabel(f"Proportion {group_name}")
        ax.set_ylabel(f"Proportion Voting {outcome_name}")
        ax.set_title(f"Ecological Regression: {group_name} → {outcome_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Annotation
        ax.annotate(
            f"Est. {group_name} vote rate: {b_group:.3f}\n"
            f"Est. non-{group_name} vote rate: {b_outgroup:.3f}",
            xy=(0.05, 0.85), xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"),
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()

    return {
        "b_group": round(b_group, 4),
        "b_outgroup": round(b_outgroup, 4),
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r ** 2, 4),
        "p_value": round(p, 4),
        "n_districts": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# 4. Gerrymandering Metrics
# ---------------------------------------------------------------------------

def efficiency_gap(
    district_results: pd.DataFrame,
    dem_votes_col: str,
    rep_votes_col: str,
) -> dict:
    """
    Compute the efficiency gap for a state legislative map.

    Wasted votes for the winner = votes above 50% threshold.
    Wasted votes for the loser = all votes cast.

    Parameters
    ----------
    district_results : pd.DataFrame
        One row per district with vote counts.
    dem_votes_col, rep_votes_col : str
        Column names for Democratic and Republican vote counts.

    Returns
    -------
    dict: efficiency_gap (positive favors R), wasted_D, wasted_R, total_votes
    """
    df = district_results.copy()
    df["total"] = df[dem_votes_col] + df[rep_votes_col]
    df["threshold"] = (df["total"] / 2).apply(np.ceil).astype(int)

    # Democratic wasted votes
    df["wasted_D"] = np.where(
        df[dem_votes_col] > df[rep_votes_col],
        df[dem_votes_col] - df["threshold"],   # winner: excess
        df[dem_votes_col]                       # loser: all votes
    )
    # Republican wasted votes
    df["wasted_R"] = np.where(
        df[rep_votes_col] >= df[dem_votes_col],
        df[rep_votes_col] - df["threshold"],
        df[rep_votes_col]
    )

    total_votes = df["total"].sum()
    wasted_D = df["wasted_D"].sum()
    wasted_R = df["wasted_R"].sum()
    eg = (wasted_R - wasted_D) / total_votes  # positive → favors R

    return {
        "efficiency_gap": round(eg, 5),
        "wasted_D": int(wasted_D),
        "wasted_R": int(wasted_R),
        "total_votes": int(total_votes),
        "favors": "Republican" if eg > 0 else "Democratic",
    }


def polsby_popper(area: float, perimeter: float) -> float:
    """
    Polsby-Popper compactness score: 4π × Area / Perimeter².
    Score of 1 = perfect circle; lower = less compact.
    """
    return 4 * np.pi * area / (perimeter ** 2)


def mean_median_difference(district_vote_shares: pd.Series) -> float:
    """
    Mean-median difference: mean D vote share minus median D vote share.
    Positive = skewed toward R (distribution is right-skewed).
    """
    v = district_vote_shares.dropna()
    return float(v.mean() - v.median())
```

## Advanced Usage

### Seat-Vote Elasticity

The seat-vote elasticity measures how many additional seats a party gains per additional percent
of the vote. In majoritarian systems, elasticity is high (small swings produce large seat changes).
In PR systems, elasticity approaches 1.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def seat_vote_elasticity(
    vote_shares: pd.Series,
    seat_shares: pd.Series,
    window: float = 0.05,
) -> float:
    """
    Estimate seat-vote elasticity using local log-log regression near the 50% mark.

    Parameters
    ----------
    vote_shares : pd.Series
        Party A vote share per election or district.
    seat_shares : pd.Series
        Party A seat share.
    window : float
        Restrict to observations with vote share in [0.5-window, 0.5+window].

    Returns
    -------
    float: elasticity (log seats / log votes)
    """
    mask = (vote_shares > 0.01) & (seat_shares > 0.01) & \
           (vote_shares.between(0.5 - window, 0.5 + window))
    v = np.log(vote_shares[mask])
    s = np.log(seat_shares[mask])
    if len(v) < 3:
        return np.nan
    slope, *_ = np.polyfit(v, s, 1)
    return round(slope, 3)


def enp_trend(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Compute ENP_votes time trend for a country using CLEA data.

    Parameters
    ----------
    df : pd.DataFrame
        CLEA-format data.
    country : str
        Country name.

    Returns
    -------
    pd.DataFrame with year, enp_votes, turnout.
    """
    elections = df[df["ctr_n"] == country].groupby("yr")
    rows = []
    for yr, grp in elections:
        pvs = grp["pvs"].dropna()
        if pvs.empty:
            continue
        enp = effective_number_of_parties(pvs)
        to = grp["to1"].iloc[0] if "to1" in grp.columns else np.nan
        rows.append({"year": yr, "enp_votes": enp, "turnout": to})
    return pd.DataFrame(rows).sort_values("year")
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| ENP > 15 | Micro-parties with tiny shares included | Filter `pvs < 0.005` before computing ENP |
| Efficiency gap undefined | Third-party votes in denominator | Keep only two-party vote for EG calculation |
| Ecological regression b outside [0,1] | Ecological fallacy / heterogeneity | Acknowledge as limitation; consider King's EI |
| CLEA `pvs` NaN for major parties | Coded differently for some elections | Check `pvs1`, `pvs2` columns in older CLEA versions |
| Polsby-Popper requires shapefile | Area/perimeter need GIS data | Use `geopandas`: `gdf["PP"] = gdf.apply(lambda r: polsby_popper(r.geometry.area, r.geometry.length), axis=1)` |

## External Resources

- CLEA (Comparative Legislative Elections Archive): https://electiondataarchive.org/
- MIT Election Data: https://electionlab.mit.edu/data
- Laakso, M. & Taagepera, R. (1979). "Effective" number of parties. *Comparative Political Studies*, 12(1), 3-27.
- King, G. (1997). *A Solution to the Ecological Inference Problem*. Princeton University Press.
- Stephanopoulos, N. & McGhee, E. (2015). Partisan gerrymandering and the efficiency gap. *U. Chi. L. Rev.*, 82, 831.
- Gallagher, M. (1991). Proportionality, disproportionality and electoral systems. *Electoral Studies*, 10(1), 33-51.

## Examples

### Example 1: ENP and Seat-Vote from CLEA-Format CSV

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLEA_PATH = os.environ.get("CLEA_PATH", "CLEA_lc_20230424.csv")

# Simulate CLEA-format data for demonstration
rng = np.random.default_rng(42)
countries_demo = ["Germany", "UK", "France", "Netherlands", "Poland"]
records = []
for ctr in countries_demo:
    for yr in range(2000, 2025, 4):
        n_parties = rng.integers(4, 9)
        raw_votes = rng.dirichlet(np.ones(n_parties) * 2)
        raw_seats = rng.dirichlet(np.ones(n_parties) * 3)  # seats more concentrated
        for i in range(n_parties):
            records.append({
                "ctr_n": ctr,
                "yr": yr,
                "pty_n": f"Party_{i+1}",
                "pvs": raw_votes[i],
                "pss": raw_seats[i],
                "to1": rng.uniform(55, 80),
            })
df_clea = pd.DataFrame(records)

# Compute election-level statistics
summaries = []
for (ctr, yr), grp in df_clea.groupby(["ctr_n", "yr"]):
    enp_v = effective_number_of_parties(grp["pvs"])
    enp_s = effective_number_of_parties(grp["pss"])
    gall = gallagher_index(grp["pvs"], grp["pss"])
    summaries.append({
        "country": ctr, "year": yr,
        "ENP_votes": round(enp_v, 2),
        "ENP_seats": round(enp_s, 2),
        "Gallagher": round(gall, 3),
        "turnout": round(grp["to1"].iloc[0], 1),
    })

df_summary = pd.DataFrame(summaries)
print("=== Election Summaries (CLEA simulation) ===")
print(df_summary.to_string(index=False))

# Plot ENP trend for each country
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ctr in countries_demo:
    sub = df_summary[df_summary["country"] == ctr].sort_values("year")
    axes[0].plot(sub["year"], sub["ENP_votes"], marker="o", label=ctr)
    axes[1].plot(sub["year"], sub["Gallagher"], marker="s", label=ctr)

axes[0].set_title("Effective Number of Parties (Votes)")
axes[0].set_ylabel("ENP")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
axes[1].set_title("Gallagher Disproportionality Index")
axes[1].set_ylabel("Gallagher Index")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("enp_gallagher_trends.png", dpi=150)
plt.show()
```

### Example 2: Efficiency Gap Calculation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(10)
n_districts = 100

# Simulate a gerrymandered map (Dems packed into fewer districts)
dem_base = 0.52  # statewide D vote share
# Packing: many districts where D wins by huge margins
# Cracking: many districts where D loses narrowly
dem_shares = np.concatenate([
    rng.normal(0.72, 0.05, 20),   # packed D districts
    rng.normal(0.46, 0.04, 80),   # cracked D districts (narrowly R)
]).clip(0.01, 0.99)

total_per_district = rng.integers(30_000, 80_000, n_districts)
dem_votes = (dem_shares * total_per_district).astype(int)
rep_votes = total_per_district - dem_votes

district_df = pd.DataFrame({
    "district": range(1, n_districts + 1),
    "dem_votes": dem_votes,
    "rep_votes": rep_votes,
    "dem_share": dem_shares,
})

eg_result = efficiency_gap(district_df, "dem_votes", "rep_votes")
print("=== Efficiency Gap Analysis ===")
for k, v in eg_result.items():
    print(f"  {k}: {v}")

# Seats won
dem_seats = (district_df["dem_votes"] > district_df["rep_votes"]).sum()
print(f"\n  D seats: {dem_seats} / {n_districts} ({dem_seats/n_districts:.1%})")
print(f"  D votes: {dem_shares.mean():.1%} statewide")

# Plot vote distribution
mm_diff = mean_median_difference(district_df["dem_share"])
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(district_df["dem_share"], bins=30, color="#1f77b4", edgecolor="white", alpha=0.8)
ax.axvline(district_df["dem_share"].mean(), color="blue", linestyle="--",
           label=f"Mean: {district_df['dem_share'].mean():.3f}")
ax.axvline(district_df["dem_share"].median(), color="orange", linestyle="--",
           label=f"Median: {district_df['dem_share'].median():.3f}")
ax.axvline(0.5, color="red", linestyle=":", linewidth=1.5, label="Majority threshold (0.5)")
ax.set_xlabel("Democratic Vote Share per District")
ax.set_ylabel("Number of Districts")
ax.set_title(f"District Vote Distribution — Mean-Median Diff: {mm_diff:+.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("efficiency_gap_histogram.png", dpi=150)
plt.show()
```

### Example 3: Ecological Regression for Group Voting Estimates

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(55)
n_counties = 120

# True group voting rates (unobservable)
true_black_vote_dem = 0.88
true_white_vote_dem = 0.42

# Observed: % Black in county, % voting Dem
black_share = rng.beta(2, 8, n_counties)
dem_share = (
    true_black_vote_dem * black_share
    + true_white_vote_dem * (1 - black_share)
    + rng.normal(0, 0.025, n_counties)
).clip(0.01, 0.99)

result = ecological_regression(
    group_share=black_share,
    outcome_share=dem_share,
    group_name="Black voters",
    outcome_name="Democratic",
    plot=True,
    save_path="ecological_regression.png",
)
print("=== Ecological Regression Results ===")
for k, v in result.items():
    print(f"  {k}: {v}")
print(f"\n  True Black Dem rate: {true_black_vote_dem}")
print(f"  True White Dem rate: {true_white_vote_dem}")
```
