---
name: patent-analysis
description: Patent landscape analysis with IPC classification, citation networks, technology emergence detection, and inventor collaboration mapping.
tags:
  - patent-analysis
  - intellectual-property
  - technology-forecasting
  - citation-network
  - ipc-classification
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
    - networkx>=3.2
    - scipy>=1.11
    - scikit-learn>=1.3
    - matplotlib>=3.7
    - requests>=2.31
last_updated: "2026-03-17"
status: stable
---

# Patent Landscape Analysis

## When to Use This Skill

Use this skill when you need to:
- Map technology landscapes using IPC/CPC patent classifications
- Analyze patent citation networks and identify pioneering inventions
- Measure technology emergence and diffusion using S-curve fitting
- Track inventor collaboration networks and knowledge transfer
- Compute patent quality indicators (forward citations, generality, originality)
- Identify white spaces and technology opportunities
- Benchmark R&D portfolios across assignees and countries

**Trigger keywords**: patent analysis, IPC classification, CPC, patent citations, forward citations, technology landscape, S-curve diffusion, inventor network, assignee analysis, patent portfolio, technology whitespace, knowledge spillover, patent value, patent family, PATSTAT, USPTO, EPO, patent claims, prior art, citation lag, technology cycle time.

## Background & Key Concepts

### Patent Quality Indicators (Trajtenberg 1990)

**Generality**: Measures how broadly a patent's technology is used across technology classes:

$$G_i = 1 - \sum_k s_{ik}^2$$

where $s_{ik}$ is the share of forward citations to patent $i$ from technology class $k$ (Herfindahl-type measure; higher = more general technology).

**Originality**: Same formula applied to backward citations (referenced patent classes):

$$O_i = 1 - \sum_k r_{ik}^2$$

### Technology Cycle Time (TCT)

Mean backward citation lag for a patent application year cohort:

$$\text{TCT} = \frac{1}{|B_i|} \sum_{j \in B_i} (\text{year}_i - \text{year}_j)$$

Shorter TCT = faster-moving technology field.

### S-Curve Diffusion (Bass Model)

Cumulative adoptions $N(t)$ follow a logistic-like curve:

$$\frac{dN}{dt} = (p + q \cdot \frac{N}{M})(M - N)$$

where $M$ is market potential, $p$ is coefficient of innovation, $q$ coefficient of imitation. Integrated: $N(t) = M \cdot \frac{1 - e^{-(p+q)t}}{1 + \frac{q}{p}e^{-(p+q)t}}$.

### Patent Value Distribution

Patent values follow a highly skewed distribution (Lanjouw & Schankerman 2004). The top 1% of patents account for roughly 25% of total value (approximated by forward citations). Value proxy:

$$v_i \approx \exp(\alpha \cdot \text{fwd\_cit}_i)$$

## Environment Setup

```bash
pip install pandas>=2.0 numpy>=1.24 networkx>=3.2 scipy>=1.11 \
            scikit-learn>=1.3 matplotlib>=3.7 requests>=2.31
```

```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
print("Patent analysis environment ready")
```

## Core Workflow

### Step 1: Patent Landscape Mapping with IPC Classification

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# -----------------------------------------------------------------
# Simulate a patent dataset: 2000 patents, 2000-2022
# IPC sections: A-H
# -----------------------------------------------------------------
np.random.seed(42)
n_patents = 2000

ipc_sections = list("ABCDEFGH")
ipc_classes = {
    "A": ["A01", "A23", "A61", "A63"],     # Human Necessities
    "B": ["B01", "B23", "B60", "B82"],     # Operations/Transport
    "C": ["C07", "C08", "C12", "C22"],     # Chemistry/Metallurgy
    "G": ["G01", "G06", "G09", "G16"],     # Physics/Computing
    "H": ["H01", "H04", "H05"],            # Electricity/Telecom
    "D": ["D01", "D06"],
    "E": ["E02", "E04"],
    "F": ["F01", "F16", "F28"],
}
all_classes = [cls for classes in ipc_classes.values() for cls in classes]

# Simulate tech-specific trends (AI patents surging in G06)
year_range = np.arange(2000, 2023)
class_trends = {}
for cls in all_classes:
    if cls == "G06":
        # AI/computing: strong growth
        class_trends[cls] = np.exp(0.15 * (year_range - 2000)) + np.random.normal(0, 2, len(year_range))
    elif cls in ["A61", "C12"]:
        # Biotech: moderate growth
        class_trends[cls] = 50 + 3 * (year_range - 2000) + np.random.normal(0, 5, len(year_range))
    else:
        class_trends[cls] = 20 + np.random.normal(0, 3, len(year_range))
    class_trends[cls] = np.maximum(class_trends[cls], 1)

# Assign patents to classes and years
years = np.random.choice(year_range, n_patents)
classes = []
for yr in years:
    t = int(yr - 2000)
    weights = np.array([class_trends[cls][t] for cls in all_classes])
    weights = weights / weights.sum()
    classes.append(np.random.choice(all_classes, p=weights))

# Additional patent attributes
forward_citations = np.random.negative_binomial(1, 0.15, n_patents)  # heavy tail
backward_citations = np.random.randint(2, 25, n_patents)
n_inventors = np.random.randint(1, 8, n_patents)
n_claims = np.random.randint(1, 30, n_patents)
country = np.random.choice(["US", "CN", "DE", "JP", "KR", "FR", "GB"],
                            n_patents, p=[0.35, 0.25, 0.10, 0.12, 0.08, 0.05, 0.05])

df = pd.DataFrame({
    "patent_id": [f"P{i:06d}" for i in range(n_patents)],
    "year": years,
    "ipc_class": classes,
    "ipc_section": [cls[0] for cls in classes],
    "forward_citations": forward_citations,
    "backward_citations": backward_citations,
    "n_inventors": n_inventors,
    "n_claims": n_claims,
    "country": country,
})

# -----------------------------------------------------------------
# Patent quality indicators
# -----------------------------------------------------------------
# Generality: diversity of citing classes (simulated)
def compute_generality(df):
    """Compute generality index for each patent."""
    gen = []
    for _, row in df.iterrows():
        # Simulate citing class distribution
        n_fwd = max(row["forward_citations"], 1)
        # Random class shares (herfindahl)
        shares = np.random.dirichlet(np.ones(5))
        gen.append(1 - (shares**2).sum())
    return np.array(gen)

df["generality"] = compute_generality(df)

# Originality
df["originality"] = df["backward_citations"].apply(
    lambda n: 1 - (np.random.dirichlet(np.ones(max(n, 1)))**2).sum()
)

# Technology Cycle Time (years)
df["tct"] = df["backward_citations"].apply(
    lambda n: np.random.uniform(3, 15) if n > 0 else np.nan
)

# -----------------------------------------------------------------
# Landscape summary
# -----------------------------------------------------------------
landscape = df.groupby("ipc_class").agg(
    n_patents=("patent_id", "count"),
    mean_fwd_cit=("forward_citations", "mean"),
    mean_generality=("generality", "mean"),
    mean_tct=("tct", "mean"),
).sort_values("n_patents", ascending=False)
print("=== Patent Landscape by IPC Class ===")
print(landscape.round(2))

# -----------------------------------------------------------------
# Technology trend visualization
# -----------------------------------------------------------------
top_classes = landscape.head(5).index.tolist()
annual_counts = df[df["ipc_class"].isin(top_classes)].groupby(
    ["year", "ipc_class"]).size().unstack(fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cls in top_classes:
    if cls in annual_counts.columns:
        axes[0].plot(annual_counts.index, annual_counts[cls], marker="o",
                     ms=3, label=cls)
axes[0].set_xlabel("Year"); axes[0].set_ylabel("Patent Filings")
axes[0].set_title("Technology Growth Trends (Top IPC Classes)")
axes[0].legend(fontsize=8)

# Country share bubble chart
country_ipc = df.groupby(["country", "ipc_section"]).size().reset_index(name="count")
pivot_ci = country_ipc.pivot(index="country", columns="ipc_section",
                              values="count").fillna(0)
im = axes[1].imshow(pivot_ci.values, cmap="YlOrRd", aspect="auto")
axes[1].set_xticks(range(len(pivot_ci.columns)))
axes[1].set_xticklabels(pivot_ci.columns)
axes[1].set_yticks(range(len(pivot_ci.index)))
axes[1].set_yticklabels(pivot_ci.index)
axes[1].set_title("Patent Counts by Country × IPC Section")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("patent_landscape.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: patent_landscape.png")
```

### Step 2: Patent Citation Network Analysis

```python
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Build a patent citation network (directed: patent → cited patent)
# -----------------------------------------------------------------
np.random.seed(42)
n_p = 300  # patents (nodes)
patent_ids = [f"P{i:06d}" for i in range(n_p)]
patent_years = np.random.randint(2000, 2023, n_p)

# Simulate citation edges (newer patents cite older ones)
citations = []
for i in range(n_p):
    # Each patent cites 3-8 older patents (preferential attachment)
    n_cites = np.random.randint(3, 9)
    older = [j for j in range(n_p) if patent_years[j] < patent_years[i]]
    if older:
        # Prefer highly cited (preferential attachment)
        cited = np.random.choice(older, size=min(n_cites, len(older)),
                                 replace=False)
        for c in cited:
            citations.append((patent_ids[i], patent_ids[c]))

G = nx.DiGraph()
for pid, yr in zip(patent_ids, patent_years):
    G.add_node(pid, year=yr)
for citing, cited in citations:
    G.add_edge(citing, cited)

print("=== Citation Network ===")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
print(f"Average in-degree (fwd citations): {sum(dict(G.in_degree()).values()) / n_p:.2f}")

# -----------------------------------------------------------------
# Network centrality metrics
# -----------------------------------------------------------------
in_degree = dict(G.in_degree())
pagerank = nx.pagerank(G, alpha=0.85)

# Top patents by forward citations (in-degree)
top_cited = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 most-cited patents:")
for pid, cit in top_cited:
    print(f"  {pid} (year {patent_years[patent_ids.index(pid)]}): {cit} citations, "
          f"PageRank={pagerank[pid]:.4f}")

# -----------------------------------------------------------------
# Technology Cycle Time via citation lags
# -----------------------------------------------------------------
citation_lags = []
for citing, cited in G.edges():
    i = patent_ids.index(citing)
    j = patent_ids.index(cited)
    lag = patent_years[i] - patent_years[j]
    if lag > 0:
        citation_lags.append(lag)

mean_tct = np.mean(citation_lags)
median_tct = np.median(citation_lags)
print(f"\nTechnology Cycle Time: mean={mean_tct:.1f}y, median={median_tct:.1f}y")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot largest weakly-connected component
wcc = max(nx.weakly_connected_components(G), key=len)
G_wcc = G.subgraph(wcc).copy()
pos = nx.spring_layout(G_wcc, seed=42, k=0.8)

node_sizes = [max(G_wcc.in_degree(n) * 30, 20) for n in G_wcc.nodes()]
node_colors = [patent_years[patent_ids.index(n)] if n in patent_ids else 2010
               for n in G_wcc.nodes()]

nx.draw_networkx_nodes(G_wcc, pos, ax=axes[0], node_size=node_sizes,
                       node_color=node_colors, cmap=plt.cm.viridis, alpha=0.8)
nx.draw_networkx_edges(G_wcc, pos, ax=axes[0], alpha=0.15, arrows=True,
                       arrowsize=5, edge_color="gray")
axes[0].set_title(f"Patent Citation Network (n={len(G_wcc)} patents)")
axes[0].axis("off")

# Citation lag distribution
axes[1].hist(citation_lags, bins=25, color="steelblue", edgecolor="black")
axes[1].axvline(mean_tct, color="red", ls="--", label=f"Mean TCT={mean_tct:.1f}y")
axes[1].set_xlabel("Citation Lag (years)"); axes[1].set_ylabel("Frequency")
axes[1].set_title("Technology Cycle Time Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig("patent_citation_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: patent_citation_network.png")
```

### Step 3: S-Curve Technology Emergence Detection

```python
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Fit Bass diffusion model to cumulative patent counts
# -----------------------------------------------------------------
def bass_model(t, M, p, q):
    """Bass model cumulative adoptions.

    Args:
        t: time array (starting from 0)
        M: market potential
        p: innovation coefficient
        q: imitation coefficient
    Returns:
        N(t): cumulative adoption
    """
    exp_term = np.exp(-(p + q) * t)
    return M * (1 - exp_term) / (1 + (q / p) * exp_term)

def fit_technology_scurve(annual_counts, technology_name):
    """Fit S-curve to annual patent filing data.

    Args:
        annual_counts: pd.Series with year as index, count as values
        technology_name: string label
    Returns:
        dict with fit parameters and forecast
    """
    years = annual_counts.index.values
    t = (years - years[0]).astype(float)
    cumulative = annual_counts.cumsum().values.astype(float)

    try:
        popt, pcov = opt.curve_fit(
            bass_model, t, cumulative,
            p0=[cumulative.max() * 3, 0.01, 0.3],
            bounds=([0, 0, 0], [1e6, 0.5, 1.0]),
            maxfev=5000
        )
        M_fit, p_fit, q_fit = popt
        perr = np.sqrt(np.diag(pcov))

        # Peak adoption rate time
        t_peak = np.log(q_fit / p_fit) / (p_fit + q_fit)
        year_peak = years[0] + t_peak

        # R²
        cum_pred = bass_model(t, *popt)
        ss_res = ((cumulative - cum_pred)**2).sum()
        ss_tot = ((cumulative - cumulative.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot

        return {
            "technology": technology_name,
            "M": M_fit, "p": p_fit, "q": q_fit,
            "year_peak": year_peak, "r2": r2,
            "popt": popt, "t0": years[0], "t": t,
            "cumulative": cumulative
        }
    except Exception as e:
        print(f"S-curve fit failed for {technology_name}: {e}")
        return None

# -----------------------------------------------------------------
# Simulate technology growth data for 3 technologies
# -----------------------------------------------------------------
np.random.seed(42)
years = np.arange(2000, 2023)
tech_data = {
    "AI/ML": np.maximum(0,
        5 * np.exp(0.2 * (years - 2000)) + np.random.normal(0, 10, len(years))),
    "Blockchain": np.maximum(0,
        np.where(years < 2014, 0,
                 2 * (years - 2014)**2 + np.random.normal(0, 5, len(years)))),
    "Fuel_Cell": np.maximum(0,
        80 / (1 + np.exp(-0.3 * (years - 2008))) + np.random.normal(0, 5, len(years))),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (tech, annual) in zip(axes, tech_data.items()):
    annual_series = pd.Series(annual.astype(int), index=years)
    fit = fit_technology_scurve(annual_series, tech)

    ax.bar(years, annual_series.cumsum(), alpha=0.4, color="steelblue",
           label="Observed cumulative")
    ax.bar(years, annual_series, alpha=0.7, color="lightblue",
           label="Annual filings")

    if fit:
        t_ext = np.linspace(0, len(years) + 10, 200)
        yrs_ext = fit["t0"] + t_ext
        ax.plot(yrs_ext, bass_model(t_ext, *fit["popt"]), "r-", lw=2,
                label=f"Bass fit (R²={fit['r2']:.2f})")
        if 2000 <= fit["year_peak"] <= 2040:
            ax.axvline(fit["year_peak"], color="orange", ls="--",
                       label=f"Peak: {fit['year_peak']:.0f}")
        print(f"\n{tech}: M={fit['M']:.0f}, p={fit['p']:.4f}, "
              f"q={fit['q']:.4f}, peak≈{fit['year_peak']:.0f}")

    ax.set_title(f"{tech} Technology S-Curve")
    ax.set_xlabel("Year"); ax.set_ylabel("Patent Count")
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("technology_scurves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: technology_scurves.png")
```

## Advanced Usage

### Inventor Collaboration Network

```python
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

def build_inventor_network(df, inventor_col="inventors", sep="; "):
    """Build weighted inventor co-invention network.

    Args:
        df: DataFrame with inventor column (semicolon-separated names)
    Returns:
        G: NetworkX weighted undirected graph
    """
    G = nx.Graph()
    edge_counter = Counter()

    for _, row in df.iterrows():
        inventors = [i.strip() for i in str(row[inventor_col]).split(sep)
                     if i.strip()]
        for inv in inventors:
            if inv not in G:
                G.add_node(inv, n_patents=0)
            G.nodes[inv]["n_patents"] += 1

        for i, j in combinations(inventors, 2):
            key = tuple(sorted([i, j]))
            edge_counter[key] += 1

    for (i, j), w in edge_counter.items():
        G.add_edge(i, j, weight=w)
    return G

# Simulate inventor data
np.random.seed(42)
n_inv_pool = 50
inventor_names = [f"Inventor_{i:02d}" for i in range(n_inv_pool)]
df_inv = df.copy()
df_inv["inventors"] = [
    "; ".join(np.random.choice(inventor_names,
                               size=np.random.randint(1, 5),
                               replace=False).tolist())
    for _ in range(len(df_inv))
]

G_inv = build_inventor_network(df_inv)
print("=== Inventor Network ===")
print(f"Inventors: {G_inv.number_of_nodes()}")
print(f"Collaborations: {G_inv.number_of_edges()}")

# Identify star inventors (high degree)
top_inv = sorted(G_inv.degree(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop inventors by collaboration degree:")
for inv, deg in top_inv:
    print(f"  {inv}: degree={deg}, patents={G_inv.nodes[inv]['n_patents']}")
```

### Technology Whitespace Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def technology_whitespace(df, class_col="ipc_class", year_col="year",
                           citation_col="forward_citations", cutoff_year=2015):
    """Identify technology whitespaces: classes with recent activity and high impact.

    Args:
        df: patent DataFrame
        cutoff_year: threshold for "recent" (only count patents after this year)
    Returns:
        DataFrame with opportunity scores
    """
    recent = df[df[year_col] >= cutoff_year]
    all_classes = df[class_col].unique()

    results = []
    for cls in all_classes:
        recent_cnt = len(recent[recent[class_col] == cls])
        all_cnt = len(df[df[class_col] == cls])
        mean_cit = df[df[class_col] == cls][citation_col].mean()
        growth_rate = (recent_cnt / max(all_cnt - recent_cnt, 1))

        results.append({
            "ipc_class": cls,
            "total_patents": all_cnt,
            "recent_patents": recent_cnt,
            "growth_rate": growth_rate,
            "mean_fwd_citations": mean_cit,
        })

    result_df = pd.DataFrame(results)
    scaler = StandardScaler()
    result_df["opportunity_score"] = scaler.fit_transform(
        result_df[["growth_rate", "mean_fwd_citations"]]
    ).mean(axis=1)

    return result_df.sort_values("opportunity_score", ascending=False)

whitespace = technology_whitespace(df, cutoff_year=2018)
print("=== Technology Whitespace Analysis (Top Opportunities) ===")
print(whitespace.head(10).round(3).to_string(index=False))
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Bass model fit diverges | Poor initial values or flat data | Use `p0=[cumulative[-1]*2, 0.005, 0.1]`; check for sufficient variation |
| Citation network is disconnected | Short time window | Extend year range; use weakly connected component |
| Generality = 1.0 for all | Only 1 citing class | Need at least 2 citing classes; filter patents with n_fwd_cit > 5 |
| Sparse IPC co-occurrence | Too granular classification | Roll up to 3-character IPC class |
| PageRank doesn't converge | Large disconnected graph | Use `max_iter=500, tol=1e-6`; prefilter to LCC |
| Name disambiguation in inventor network | Multiple name formats | Normalize: `str.lower().strip()`; use disambiguation tools |

## External Resources

- Trajtenberg, M. (1990). A penny for your quotes: Patent citations and the value of innovations. *RAND Journal of Economics*, 21(1).
- Hall, B. H., Jaffe, A. B., & Trajtenberg, M. (2001). *The NBER Patent Citations Data File*. NBER WP 8498.
- Bass, F. M. (1969). A new product growth model for consumer durables. *Management Science*, 15(5).
- [PATSTAT documentation](https://www.epo.org/en/searching-for-patents/business/patstat)
- [USPTO Open Data Portal](https://www.uspto.gov/ip-policy/economic-research/research-datasets)
- [Google Patents Public Data](https://console.cloud.google.com/marketplace/product/google_patents_public_data/)

## Examples

### Example 1: Assignee Portfolio Benchmarking

```python
import numpy as np
import pandas as pd

def assignee_benchmarking(df, assignee_col="assignee", citation_col="forward_citations"):
    """Compute portfolio metrics for each assignee."""
    return df.groupby(assignee_col).agg(
        n_patents=(citation_col, "count"),
        total_citations=(citation_col, "sum"),
        mean_citations=(citation_col, "mean"),
        citation_intensity=(citation_col, lambda x: (x > 10).mean()),
    ).sort_values("total_citations", ascending=False)

np.random.seed(42)
assignees = [f"Company_{i:02d}" for i in range(10)]
df_assign = df.copy()
df_assign["assignee"] = np.random.choice(assignees, len(df_assign),
                                          p=np.exp(-0.3 * np.arange(10)) /
                                            np.exp(-0.3 * np.arange(10)).sum())

portfolio = assignee_benchmarking(df_assign)
print("=== Assignee Portfolio Benchmarking ===")
print(portfolio.round(2))
```

### Example 2: Patent Family Analysis

```python
import pandas as pd
import numpy as np

def analyze_patent_families(df):
    """Compute family-level statistics from patent records.

    A patent family groups equivalent patents filed in multiple countries.
    Family size (number of countries) is a proxy for commercial value.
    """
    # Simulate family IDs (many patents share a family)
    np.random.seed(42)
    n_families = len(df) // 3
    df = df.copy()
    df["family_id"] = np.random.choice(range(n_families), len(df))

    family_stats = df.groupby("family_id").agg(
        n_family_members=("patent_id", "count"),
        n_countries=("country", "nunique"),
        max_fwd_citations=("forward_citations", "max"),
        earliest_year=("year", "min"),
    ).reset_index()

    family_stats["value_proxy"] = (
        np.log1p(family_stats["max_fwd_citations"]) *
        np.log1p(family_stats["n_countries"])
    )
    print(f"Patent families: {len(family_stats)}")
    print(f"Mean family size: {family_stats['n_family_members'].mean():.1f}")
    print(f"Multi-country families: {(family_stats['n_countries'] > 1).mean()*100:.1f}%")
    return family_stats

families = analyze_patent_families(df)
print("\nTop 5 highest-value patent families:")
print(families.nlargest(5, "value_proxy")[
    ["family_id", "n_family_members", "n_countries", "max_fwd_citations", "value_proxy"]
].round(2).to_string(index=False))
```
