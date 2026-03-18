---
name: research-impact
description: Research impact measurement with altmetrics, citation normalization, field-weighted indicators, and journal ranking analysis for evaluation studies.
tags:
  - research-impact
  - altmetrics
  - citation-normalization
  - journal-ranking
  - research-evaluation
version: "1.0.0"
authors:
  - "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - pandas>=2.0
    - numpy>=1.24
    - scipy>=1.11
    - statsmodels>=0.14
    - matplotlib>=3.7
    - requests>=2.31
last_updated: "2026-03-17"
status: stable
---

# Research Impact Measurement

## When to Use This Skill

Use this skill when you need to:
- Compute field-normalized citation indicators (MNCS, FWCI, PP-top10%)
- Calculate journal impact metrics (JIF, SJR, SNIP, h5-index)
- Measure altmetric attention (Altmetric score, PlumX)
- Evaluate institutional research performance (CWTS Leiden methodology)
- Assess open access impact and citation advantages
- Compare research profiles across departments or universities
- Compute co-citation and bibliographic coupling similarity

**Trigger keywords**: research impact, citation normalization, field-weighted citation impact, FWCI, MNCS, journal impact factor, Eigenfactor, h5-index, altmetrics, Altmetric score, open access advantage, institutional benchmarking, CWTS Leiden, CERN, REF, ERA, percentile-based indicators, PP-top10, research performance evaluation.

## Background & Key Concepts

### Field-Weighted Citation Impact (FWCI)

$$\text{FWCI}_p = \frac{c_p}{\langle c \rangle_{\text{field, year}}}$$

where $c_p$ is citation count of paper $p$ and $\langle c \rangle$ is the mean citation count of all papers in the same field and publication year. FWCI = 1.0 means world average; FWCI > 1.0 means above average.

### Mean Normalized Citation Score (MNCS)

$$\text{MNCS} = \frac{1}{|S|} \sum_{p \in S} \text{FWCI}_p$$

for a set $S$ of papers. Fractional counting assigns weight $1/k$ to multi-authored papers with $k$ authors.

### Percentage in Top 10% (PP-top10%)

$$\text{PP-top10\%} = \frac{|\{p \in S : \text{FWCI}_p \geq \text{P}_{90}(\text{field, year})\}|}{|S|} \times 100$$

### Journal Impact Factor (JIF)

$$\text{JIF}_{y} = \frac{\sum_{y-2}^{y-1} \text{citations to journal articles}}{\sum_{y-2}^{y-1} \text{citable items}}$$

### SCImago Journal Rank (SJR)

SJR is a prestige-weighted citation metric based on eigenvector centrality of the journal citation network:

$$\text{SJR}_j = \alpha \sum_k \frac{\text{cit}_{kj}}{A_k \cdot \text{cit}_k} \text{SJR}_k + (1 - \alpha) \frac{1}{N}$$

### Altmetric Attention Score

Weighted sum of online attention signals:
- News articles: +8 pts
- Blog posts: +5 pts
- Twitter/X mentions: +1 pt (capped at 16)
- Wikipedia: +3 pts
- Policy documents: +3 pts
- Mendeley saves: proportional

## Environment Setup

```bash
pip install pandas>=2.0 numpy>=1.24 scipy>=1.11 statsmodels>=0.14 \
            matplotlib>=3.7 requests>=2.31
```

```python
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
print("Research impact analysis environment ready")
```

## Core Workflow

### Step 1: Field-Normalized Citation Indicators

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate a publication dataset with field/year reference sets
# -----------------------------------------------------------------
np.random.seed(42)
n_papers = 500

fields = ["Computer Science", "Medicine", "Physics", "Chemistry", "Social Science"]
# Fields have different citation baselines
field_baselines = {
    "Computer Science": 12,
    "Medicine": 25,
    "Physics": 18,
    "Chemistry": 20,
    "Social Science": 6,
}
years = np.random.randint(2015, 2023, n_papers)
paper_fields = np.random.choice(fields, n_papers)

# Citations: field×year specific (lognormal with field baseline)
citations = np.array([
    int(np.random.lognormal(
        np.log(field_baselines[f]) + 0.1 * (yr - 2015), 1.0
    ))
    for f, yr in zip(paper_fields, years)
])

# Open access status (OA papers may have higher citations)
oa_status = np.random.binomial(1, 0.40, n_papers)
# OA advantage: ~25% citation boost
citations = (citations * (1 + 0.25 * oa_status)).astype(int)

df = pd.DataFrame({
    "paper_id": [f"P{i:05d}" for i in range(n_papers)],
    "field": paper_fields,
    "year": years,
    "citations": citations,
    "oa": oa_status,
    "n_authors": np.random.randint(1, 10, n_papers),
})

# -----------------------------------------------------------------
# Compute field-year baselines and FWCI
# -----------------------------------------------------------------
field_year_means = df.groupby(["field", "year"])["citations"].mean()
field_year_p90   = df.groupby(["field", "year"])["citations"].quantile(0.90)

def compute_fwci(row):
    """Compute Field-Weighted Citation Impact."""
    baseline = field_year_means.get((row["field"], row["year"]), 1.0)
    return row["citations"] / max(baseline, 0.1)

df["fwci"] = df.apply(compute_fwci, axis=1)

def is_top10(row):
    """Check if paper is in top 10% of its field-year."""
    p90 = field_year_p90.get((row["field"], row["year"]), 0)
    return int(row["citations"] >= p90)

df["top10"] = df.apply(is_top10, axis=1)

# -----------------------------------------------------------------
# Fractional counting MNCS for each field
# -----------------------------------------------------------------
df["frac_fwci"] = df["fwci"] / df["n_authors"]  # fractional

field_metrics = df.groupby("field").agg(
    n_papers=("paper_id", "count"),
    mncs=("fwci", "mean"),
    mncs_frac=("frac_fwci", "mean"),
    pp_top10=("top10", "mean"),
    total_cit=("citations", "sum"),
).reset_index()
field_metrics["pp_top10_pct"] = field_metrics["pp_top10"] * 100

print("=== Field-Normalized Research Impact ===")
print(field_metrics.round(3).to_string(index=False))

# -----------------------------------------------------------------
# Open Access citation advantage
# -----------------------------------------------------------------
from scipy.stats import mannwhitneyu

oa_fwci = df[df["oa"] == 1]["fwci"]
closed_fwci = df[df["oa"] == 0]["fwci"]
stat, p_val = mannwhitneyu(oa_fwci, closed_fwci, alternative="greater")

print(f"\n=== Open Access Citation Advantage ===")
print(f"OA MNCS:     {oa_fwci.mean():.3f}")
print(f"Non-OA MNCS: {closed_fwci.mean():.3f}")
print(f"OA advantage: {(oa_fwci.mean()/closed_fwci.mean() - 1)*100:.1f}%")
print(f"Mann-Whitney U test: p = {p_val:.4f} ({'significant' if p_val < 0.05 else 'not significant'})")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# FWCI distribution by field
for field in fields:
    mask = df["field"] == field
    axes[0].hist(df.loc[mask, "fwci"].clip(0, 10), bins=20, alpha=0.4,
                 label=field, density=True)
axes[0].axvline(1.0, color="red", ls="--", label="World average")
axes[0].set_xlabel("FWCI"); axes[0].set_ylabel("Density")
axes[0].set_title("FWCI Distribution by Field")
axes[0].legend(fontsize=7)

# PP-top10% bar chart
colors = ["steelblue" if mncs >= 1.0 else "orange"
          for mncs in field_metrics["mncs"]]
axes[1].bar(field_metrics["field"], field_metrics["pp_top10_pct"],
            color=colors, edgecolor="black")
axes[1].axhline(10, color="red", ls="--", label="World avg (10%)")
axes[1].set_xticklabels(field_metrics["field"], rotation=15, ha="right")
axes[1].set_ylabel("PP-top10% (%)")
axes[1].set_title("Percentage in Top 10% by Field")
axes[1].legend()

# OA vs. non-OA FWCI comparison
oa_labels = ["Open Access", "Closed Access"]
oa_data = [oa_fwci.clip(0, 10).values, closed_fwci.clip(0, 10).values]
bp = axes[2].boxplot(oa_data, labels=oa_labels, patch_artist=True,
                     showfliers=False)
bp["boxes"][0].set_facecolor("lightgreen")
bp["boxes"][1].set_facecolor("lightyellow")
axes[2].axhline(1.0, color="red", ls="--", label="FWCI=1 (world avg)")
axes[2].set_ylabel("FWCI")
axes[2].set_title("OA vs. Non-OA Citation Impact")
axes[2].legend()

plt.tight_layout()
plt.savefig("research_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: research_impact.png")
```

### Step 2: Journal Impact Metrics

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate a journal citation database
# 50 journals, 5 years
# -----------------------------------------------------------------
np.random.seed(42)
n_journals = 50
journal_names = [f"Journal_{i:02d}" for i in range(n_journals)]
years = [2020, 2021, 2022, 2023]

# Journal prestige tiers (affects citation rates)
prestige = np.random.uniform(0.1, 2.0, n_journals)  # higher = more prestigious
annual_articles = np.random.randint(50, 500, n_journals)

# Build 4-year citation matrix: citations_ij[t] = citations in year t of articles in year t
records = []
for j_idx, jname in enumerate(journal_names):
    for yr in years:
        # Articles published in year yr
        n_articles = int(annual_articles[j_idx] * (1 + 0.05 * (yr - 2020)))
        # Forward citations in following 2 years
        cit_yr1 = np.random.poisson(prestige[j_idx] * 8, n_articles).sum()
        cit_yr2 = np.random.poisson(prestige[j_idx] * 12, n_articles).sum()
        records.append({
            "journal": jname,
            "pub_year": yr,
            "n_articles": n_articles,
            "cit_yr1": cit_yr1,
            "cit_yr2": cit_yr2,
        })

jdf = pd.DataFrame(records)

# -----------------------------------------------------------------
# Compute Journal Impact Factor (JIF) for 2022
# JIF_2022 = (citations in 2022 to articles from 2020 and 2021) / (articles 2020+2021)
# -----------------------------------------------------------------
def compute_jif(jdf, impact_year=2022, window=2):
    """Compute Journal Impact Factor."""
    results = []
    for jname in jdf["journal"].unique():
        j = jdf[jdf["journal"] == jname]
        # Denominator: citable items in window years before impact_year
        window_df = j[j["pub_year"].between(impact_year - window, impact_year - 1)]
        n_citable = window_df["n_articles"].sum()

        # Numerator: approximate citations received in impact_year
        # Using cit_yr2 for 2-year lag
        citations_received = window_df["cit_yr2"].sum()

        if n_citable > 0:
            jif = citations_received / n_citable
        else:
            jif = 0

        results.append({"journal": jname, "JIF_2022": jif,
                         "n_citable_items": n_citable})
    return pd.DataFrame(results).sort_values("JIF_2022", ascending=False)

jif_df = compute_jif(jdf)
print("=== Journal Impact Factors (Top 10) ===")
print(jif_df.head(10).round(3).to_string(index=False))
print(f"\nMedian JIF: {jif_df['JIF_2022'].median():.2f}")
print(f"JIF distribution: {jif_df['JIF_2022'].describe().round(2).to_dict()}")

# -----------------------------------------------------------------
# Eigenfactor Score (simplified PageRank approach)
# -----------------------------------------------------------------
def eigenfactor_score(jif_df, n_iter=100):
    """Simplified Eigenfactor via iterative citation-weighted prestige.

    Note: True Eigenfactor uses full citation matrix; this is illustrative.
    """
    n = len(jif_df)
    jif_values = jif_df["JIF_2022"].values
    # Build stochastic transition matrix (random jumps + JIF-biased links)
    P = np.full((n, n), 1/n)  # base: uniform random surfer
    # Weight links by JIF
    col_sums = jif_values.sum()
    if col_sums > 0:
        P += jif_values[:, None] / col_sums * 0.5
    P = P / P.sum(axis=1, keepdims=True)

    # Power iteration
    v = np.ones(n) / n
    for _ in range(n_iter):
        v = v @ P
        v = v / v.sum()

    jif_df = jif_df.copy()
    jif_df["eigenfactor_score"] = v * 100  # scale to percentages
    return jif_df

jif_df = eigenfactor_score(jif_df)
print("\nTop 5 by Eigenfactor Score:")
print(jif_df.nlargest(5, "eigenfactor_score")[
    ["journal", "JIF_2022", "eigenfactor_score"]].round(3).to_string(index=False))

# -----------------------------------------------------------------
# Visualization: JIF distribution and ranking
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# JIF histogram
axes[0].hist(jif_df["JIF_2022"], bins=15, color="steelblue", edgecolor="black")
axes[0].axvline(jif_df["JIF_2022"].median(), color="red", ls="--",
                label=f"Median JIF={jif_df['JIF_2022'].median():.2f}")
axes[0].set_xlabel("Journal Impact Factor"); axes[0].set_ylabel("Frequency")
axes[0].set_title("JIF Distribution")
axes[0].legend()

# Top 15 journals ranked
top15 = jif_df.head(15)
axes[1].barh(top15["journal"], top15["JIF_2022"],
             color="steelblue", edgecolor="black")
axes[1].set_xlabel("JIF 2022"); axes[1].set_title("Top 15 Journals by JIF")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig("journal_impact.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: journal_impact.png")
```

### Step 3: Altmetrics and Online Attention Analysis

```python
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import os

def get_altmetric_score(doi, api_key=None):
    """Fetch Altmetric score for a paper DOI.

    Args:
        doi: DOI string (e.g., "10.1038/nature12373")
        api_key: Altmetric API key from environment variable
    Returns:
        dict with score and attention breakdown, or simulated data
    """
    if api_key is None:
        api_key = os.getenv("ALTMETRIC_API_KEY", "")

    url = f"https://api.altmetric.com/v1/doi/{doi}"
    params = {}
    if api_key:
        params["key"] = api_key

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                "doi": doi,
                "altmetric_score": data.get("score", 0),
                "news_mentions": data.get("cited_by_feeds_count", 0),
                "twitter_mentions": data.get("cited_by_tweeters_count", 0),
                "blog_mentions": data.get("cited_by_posts_count", 0),
                "policy_mentions": data.get("cited_by_policies_count", 0),
                "mendeley_readers": data.get("readers", {}).get("mendeley", 0),
            }
    except Exception:
        pass
    return None  # Caller handles fallback

# -----------------------------------------------------------------
# Simulate altmetric data for a set of papers
# -----------------------------------------------------------------
np.random.seed(42)
n_papers_am = 200

# Altmetric scores are extremely skewed (most = 0, few go viral)
altmetric_scores = np.random.exponential(2, n_papers_am)
altmetric_scores = np.where(np.random.random(n_papers_am) < 0.7, 0,
                             altmetric_scores * 5)

df_am = pd.DataFrame({
    "paper_id": [f"P{i:05d}" for i in range(n_papers_am)],
    "field": np.random.choice(["Medicine", "Physics", "CS", "Social Science"],
                               n_papers_am),
    "citations": np.random.negative_binomial(2, 0.1, n_papers_am),
    "altmetric_score": altmetric_scores,
    "twitter_mentions": (altmetric_scores * np.random.uniform(0.5, 2, n_papers_am)).astype(int),
    "news_mentions": (altmetric_scores * np.random.uniform(0, 0.3, n_papers_am)).astype(int),
    "mendeley_readers": np.random.randint(0, 500, n_papers_am),
    "oa": np.random.binomial(1, 0.4, n_papers_am),
})

# Correlation between altmetrics and citations
from scipy.stats import spearmanr
corr, p = spearmanr(df_am["altmetric_score"], df_am["citations"])
print(f"Spearman correlation (altmetric vs. citations): r={corr:.3f}, p={p:.4f}")

# Attention by field
field_attention = df_am.groupby("field").agg(
    mean_altmetric=("altmetric_score", "mean"),
    mean_twitter=("twitter_mentions", "mean"),
    mean_mendeley=("mendeley_readers", "mean"),
    n_viral=(  # viral = top 10% altmetric
        "altmetric_score",
        lambda x: (x > np.percentile(df_am["altmetric_score"], 90)).mean() * 100
    ),
).reset_index()
print("\n=== Altmetric Attention by Field ===")
print(field_attention.round(2).to_string(index=False))

# Altmetric weighting
WEIGHTS = {"news_mentions": 8, "twitter_mentions": 1,
           "mendeley_readers": 0.1}

def compute_altmetric_composite(row):
    """Simplified Altmetric-like composite score."""
    return sum(row.get(k, 0) * w for k, w in WEIGHTS.items())

df_am["composite_altmetric"] = df_am.apply(compute_altmetric_composite, axis=1)

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Altmetric vs. citations scatter
axes[0].scatter(np.log1p(df_am["citations"]),
                np.log1p(df_am["altmetric_score"]),
                alpha=0.4, s=15, color="steelblue")
axes[0].set_xlabel("ln(Citations + 1)")
axes[0].set_ylabel("ln(Altmetric Score + 1)")
axes[0].set_title(f"Altmetrics vs. Citations\n(Spearman r={corr:.2f})")

# Attention breakdown by field
x = np.arange(len(field_attention))
w = 0.25
axes[1].bar(x - w, field_attention["mean_altmetric"], w,
            label="Altmetric", color="steelblue")
axes[1].bar(x, field_attention["mean_twitter"], w,
            label="Twitter", color="orange")
axes[1].bar(x + w, field_attention["mean_mendeley"] / 10, w,
            label="Mendeley/10", color="green")
axes[1].set_xticks(x)
axes[1].set_xticklabels(field_attention["field"], rotation=15, ha="right")
axes[1].set_title("Attention Breakdown by Field")
axes[1].legend(fontsize=8)

# Altmetric score distribution
axes[2].hist(df_am["altmetric_score"][df_am["altmetric_score"] > 0],
             bins=30, log=True, color="coral", edgecolor="black")
axes[2].set_xlabel("Altmetric Score (> 0)")
axes[2].set_ylabel("Frequency (log)")
axes[2].set_title("Altmetric Score Distribution")

plt.tight_layout()
plt.savefig("altmetrics_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: altmetrics_analysis.png")
```

## Advanced Usage

### Institutional Research Profile Dashboard

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def institutional_profile(df, institution_col="institution", field_col="field",
                           fwci_col="fwci", pp10_col="top10"):
    """Compute full institutional research profile.

    Args:
        df: paper-level DataFrame
    Returns:
        profile DataFrame indexed by institution
    """
    profile = df.groupby(institution_col).agg(
        n_papers=(fwci_col, "count"),
        mncs=(fwci_col, "mean"),
        pp_top10=(pp10_col, lambda x: x.mean() * 100),
        n_fields=(field_col, "nunique"),
        citation_share=("citations", lambda x: x.sum()),
    ).reset_index()

    # Normalize citation share
    total_cit = profile["citation_share"].sum()
    profile["citation_share_pct"] = profile["citation_share"] / total_cit * 100

    # Compute Specialization Herfindahl
    def hhi(grp):
        shares = grp[field_col].value_counts(normalize=True)
        return (shares**2).sum()

    profile["specialization_hhi"] = [
        hhi(df[df[institution_col] == inst])
        for inst in profile[institution_col]
    ]
    return profile.sort_values("mncs", ascending=False)

# Simulate multi-institution dataset
np.random.seed(42)
n_paps = 400
institutions = ["Univ_A", "Univ_B", "Univ_C", "Institute_D", "Center_E"]
inst_prestige = {"Univ_A": 1.5, "Univ_B": 1.2, "Univ_C": 0.9,
                 "Institute_D": 1.8, "Center_E": 1.1}
df_inst = df.copy().head(n_paps)
df_inst["institution"] = np.random.choice(institutions, n_paps,
    p=np.array([0.30, 0.25, 0.20, 0.15, 0.10]))
# Adjust FWCI by institution prestige
df_inst["fwci"] = df_inst.apply(
    lambda r: r["fwci"] * inst_prestige.get(r["institution"], 1.0), axis=1)
df_inst["top10"] = (df_inst["fwci"] > df_inst["fwci"].quantile(0.90)).astype(int)
df_inst["citations"] = (df_inst["fwci"] * 15).astype(int)

profile = institutional_profile(df_inst)
print("=== Institutional Research Profile ===")
print(profile.round(3).to_string(index=False))
```

### Leiden Ranking-Style Indicator Computation

```python
import numpy as np
import pandas as pd

def leiden_indicators(df, fractional=True):
    """Compute CWTS Leiden Ranking-style indicators.

    Indicators:
        P: total publication count (or fractional count)
        MNCS: mean normalized citation score
        PP_top10: percentage in top 10%
        collab_int: international collaboration rate
    """
    if fractional:
        df = df.copy()
        df["frac_weight"] = 1 / df["n_authors"]
        P = df.groupby("institution")["frac_weight"].sum()
        MNCS = df.groupby("institution").apply(
            lambda g: (g["fwci"] * g["frac_weight"]).sum() / g["frac_weight"].sum()
        )
    else:
        P = df.groupby("institution")["paper_id"].count()
        MNCS = df.groupby("institution")["fwci"].mean()

    PP10 = df.groupby("institution")["top10"].mean() * 100

    leiden = pd.DataFrame({"P": P, "MNCS": MNCS, "PP_top10": PP10})
    leiden = leiden.sort_values("MNCS", ascending=False)
    return leiden

leiden = leiden_indicators(df_inst, fractional=True)
print("\n=== Leiden-Style Indicators (Fractional Counting) ===")
print(leiden.round(3))
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| FWCI = 0 for all papers | All baselines = 0 | Ensure field-year groups have ≥ 1 paper |
| JIF negative or zero | No citations in window | Check year range; expand window to 3 years |
| Altmetric API returns 404 | DOI not tracked or invalid | Verify DOI format; use bare DOI without URL prefix |
| Spearman correlation insignificant | Small sample or too many zero altmetric scores | Filter papers with altmetric > 0; increase sample size |
| PP-top10 always 0% | Percentile computed globally, not by field | Use field×year percentile thresholds |
| Eigenfactor power iteration diverges | Non-stochastic transition matrix | Ensure row sums = 1 before iteration |

## External Resources

- Waltman, L., & Eck, N. J. van (2012). A new methodology for constructing a publication-level classification system of science. *JASIST*, 63(12).
- Priem, J., et al. (2010). Altmetrics: A manifesto. [altmetrics.org](http://altmetrics.org/manifesto/)
- [CWTS Leiden Ranking methodology](https://www.leidenranking.com/information/indicators)
- [Altmetric API documentation](https://api.altmetric.com/)
- Garfield, E. (1955). Citation indexes for science. *Science*, 122(3159).
- [Scimago Journal Rank](https://www.scimagojr.com/)

## Examples

### Example 1: Department-Level Impact Report

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 300
departments = ["CS", "Physics", "Biology", "Chemistry", "Mathematics"]
dept = np.random.choice(departments, n)
year = np.random.randint(2018, 2024, n)
fwci = np.abs(np.random.normal(1.2, 0.8, n))
top10 = (fwci > 1.8).astype(int)

dept_df = pd.DataFrame({"dept": dept, "year": year, "fwci": fwci, "top10": top10})

report = dept_df.groupby(["dept", "year"]).agg(
    n_papers=("fwci", "count"),
    mncs=("fwci", "mean"),
    pp_top10=("top10", "mean"),
).reset_index()
report["pp_top10_pct"] = report["pp_top10"] * 100

print("=== Department Impact Report (2022) ===")
print(report[report["year"] == 2022].sort_values("mncs", ascending=False)
      .round(3).to_string(index=False))
```

### Example 2: Citation Decay Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def citation_decay_curve(publication_year, current_year=2023, field_halflife=10):
    """Model citation decay over time using exponential decay.

    Args:
        publication_year: year paper was published
        current_year: current year
        field_halflife: citation half-life in years (field-specific)
    Returns:
        citation_age_distribution (dict)
    """
    paper_age = current_year - publication_year
    ages = np.arange(0, paper_age + 1)
    # Citation probability at each age (gamma-like profile: rise then fall)
    peak_age = min(3, paper_age)
    weights = np.exp(-0.5 * ((ages - peak_age) / (field_halflife / 2.35))**2)
    weights = weights / weights.sum()
    return dict(zip(range(current_year - paper_age, current_year + 1), weights))

# Compare fields
field_halflives = {"Computer Science": 5, "Medicine": 10, "Mathematics": 20}
years = np.arange(0, 25)
fig, ax = plt.subplots(figsize=(8, 5))
for field, hl in field_halflives.items():
    weights = np.exp(-0.5 * ((years - 2) / (hl / 2.35))**2)
    weights = weights / weights.sum()
    ax.plot(years, weights, marker="o", ms=3, label=f"{field} (HL={hl}y)")
ax.set_xlabel("Citation Age (years)"); ax.set_ylabel("Relative Citation Probability")
ax.set_title("Citation Age Profiles by Field")
ax.legend()
plt.tight_layout()
plt.savefig("citation_decay.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved: citation_decay.png")
```
