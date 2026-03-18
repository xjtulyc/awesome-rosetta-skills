---
name: openalx-bibliometrics
description: Bibliometric analysis using the OpenAlex API covering co-authorship networks, citation analysis, h-index, and research trend mapping.
tags:
  - bibliometrics
  - openalx
  - citation-analysis
  - research-networks
  - scientometrics
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
    - requests>=2.31
    - pandas>=2.0
    - numpy>=1.24
    - networkx>=3.2
    - matplotlib>=3.7
    - scipy>=1.11
last_updated: "2026-03-17"
status: stable
---

# OpenAlex Bibliometrics

## When to Use This Skill

Use this skill when you need to:
- Retrieve publication metadata from OpenAlex API (open scholarly metadata)
- Compute h-index, g-index, i10-index for authors or institutions
- Build and analyze co-authorship networks
- Map citation networks and identify key papers
- Track research trends using publication counts and concept co-occurrence
- Compare institutional research output and collaboration patterns
- Perform systematic mapping studies without manual database downloads

**Trigger keywords**: OpenAlex, bibliometrics, h-index, citation count, co-authorship network, research output, institutional collaboration, VOSviewer, bibliographic coupling, co-citation analysis, journal impact, Altmetrics, systematic mapping, PRISMA, research front, scholarly network, scientometrics.

## Background & Key Concepts

### h-index (Hirsch 2005)

The h-index is the maximum $h$ such that the researcher has at least $h$ papers with $\geq h$ citations each:

$$h = \max\{h : |\{p : c_p \geq h\}| \geq h\}$$

### g-index (Egghe 2006)

The g-index is the largest $g$ such that the top $g$ papers together have at least $g^2$ citations:

$$g = \max\{g : \sum_{p=1}^{g} c_p \geq g^2\}$$

### Bibliographic Coupling

Two papers are bibliographically coupled if they share common references. Coupling strength between papers $A$ and $B$:

$$\text{BC}(A,B) = \frac{|R_A \cap R_B|}{|R_A \cup R_B|}$$

(Jaccard similarity of reference sets).

### Co-citation Strength

Two papers are co-cited when both appear in the reference list of a third paper. Co-citation strength:

$$\text{CC}(A,B) = \frac{|\text{papers citing both } A \text{ and } B|}{\sqrt{c_A \cdot c_B}}$$

(Salton's cosine normalization).

### Bradford's Law of Scattering

A small core of journals contributes a disproportionate fraction of articles on a subject. If journals are ranked by productivity:

$$\log r = a + b \log n$$

The Bradford multiplier $b$ describes how quickly productivity falls across zones.

## Environment Setup

```bash
pip install requests>=2.31 pandas>=2.0 numpy>=1.24 networkx>=3.2 \
            matplotlib>=3.7 scipy>=1.11
```

```python
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
print("Bibliometrics environment ready")
```

## Core Workflow

### Step 1: Retrieve Publications from OpenAlex API

```python
import requests
import pandas as pd
import numpy as np
import time
import os

OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")  # polite pool if provided

def openalex_works(query, max_results=200, email=OPENALEX_EMAIL):
    """Retrieve works from OpenAlex API with pagination.

    Args:
        query: search string
        max_results: maximum number of results to fetch
        email: optional email for polite pool (higher rate limit)
    Returns:
        list of work dicts
    """
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": min(max_results, 200),
        "cursor": "*",
        "select": ("id,title,publication_year,cited_by_count,"
                   "authorships,concepts,referenced_works,doi"),
    }
    if email:
        params["mailto"] = email

    all_works = []
    fetched = 0
    while fetched < max_results:
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"API call failed: {e}")
            break

        results = data.get("results", [])
        if not results:
            break
        all_works.extend(results)
        fetched += len(results)

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor or fetched >= max_results:
            break
        params["cursor"] = next_cursor
        time.sleep(0.1)  # be polite

    return all_works[:max_results]

def works_to_dataframe(works):
    """Flatten OpenAlex work objects to a tidy DataFrame."""
    rows = []
    for w in works:
        authors = [a.get("author", {}).get("display_name", "")
                   for a in w.get("authorships", [])]
        institutions = list({
            inst.get("institution", {}).get("display_name", "Unknown")
            for a in w.get("authorships", [])
            for inst in a.get("institutions", [])
        })
        top_concepts = sorted(w.get("concepts", []),
                               key=lambda c: c.get("score", 0), reverse=True)[:3]
        rows.append({
            "id": w.get("id", ""),
            "title": w.get("title", ""),
            "year": w.get("publication_year"),
            "citations": w.get("cited_by_count", 0),
            "doi": w.get("doi", ""),
            "n_authors": len(authors),
            "authors": "; ".join(authors[:5]),
            "institutions": "; ".join(institutions[:3]),
            "concepts": "; ".join(c["display_name"] for c in top_concepts),
            "n_references": len(w.get("referenced_works", [])),
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------
# Try API; fall back to synthetic data if offline
# -----------------------------------------------------------------
try:
    works = openalex_works("machine learning climate science", max_results=100)
    if len(works) < 5:
        raise ValueError("Too few results returned")
    df = works_to_dataframe(works)
    print(f"Fetched {len(df)} papers from OpenAlex")
    DATA_SOURCE = "API"
except Exception as e:
    print(f"API unavailable ({e}), using synthetic data")
    np.random.seed(42)
    n_papers = 100
    years = np.random.randint(2010, 2024, n_papers)
    # Simulate citation distribution (highly skewed)
    citations = np.random.zipf(1.8, n_papers) * 5
    n_authors = np.random.randint(1, 12, n_papers)
    df = pd.DataFrame({
        "id": [f"W{i:06d}" for i in range(n_papers)],
        "title": [f"Paper on ML and Climate Science #{i}" for i in range(n_papers)],
        "year": years,
        "citations": citations,
        "n_authors": n_authors,
        "authors": [f"Author_{i%30}; Author_{(i+1)%30}" for i in range(n_papers)],
        "institutions": [f"Univ_{i%10}" for i in range(n_papers)],
        "concepts": "Machine Learning; Climate Science",
        "n_references": np.random.randint(20, 80, n_papers),
    })
    DATA_SOURCE = "Synthetic"

print(f"\nData source: {DATA_SOURCE}")
print(df[["year", "citations", "n_authors"]].describe().round(1))
```

### Step 2: Bibliometric Indicators

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Compute h-index, g-index, i10-index
# -----------------------------------------------------------------
def h_index(citations):
    """Compute h-index from citation counts."""
    c = np.sort(citations)[::-1]
    h = 0
    for i, ci in enumerate(c):
        if ci >= i + 1:
            h = i + 1
        else:
            break
    return h

def g_index(citations):
    """Compute g-index from citation counts."""
    c = np.sort(citations)[::-1]
    cumsum = np.cumsum(c)
    g = 0
    for i in range(len(c)):
        if cumsum[i] >= (i + 1)**2:
            g = i + 1
    return g

def i10_index(citations):
    """Count papers with >= 10 citations."""
    return int((np.array(citations) >= 10).sum())

def compute_indices(citation_series):
    """Compute all standard bibliometric indices."""
    c = citation_series.values
    return {
        "n_papers": len(c),
        "total_citations": int(c.sum()),
        "mean_citations": float(c.mean()),
        "median_citations": float(np.median(c)),
        "h_index": h_index(c),
        "g_index": g_index(c),
        "i10_index": i10_index(c),
        "max_citations": int(c.max()),
    }

metrics = compute_indices(df["citations"])
print("=== Bibliometric Indices ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")

# -----------------------------------------------------------------
# Publication trend analysis
# -----------------------------------------------------------------
annual_counts = df.groupby("year").agg(
    n_papers=("id", "count"),
    total_cit=("citations", "sum"),
    mean_cit=("citations", "mean")
).reset_index()

# Fit linear trend to publication counts
from scipy.stats import linregress
slope, intercept, r, p, se = linregress(annual_counts["year"], annual_counts["n_papers"])
print(f"\nPublication growth trend: {slope:.2f} papers/year (p={p:.3f})")

# -----------------------------------------------------------------
# Bradford's Law: core journal analysis
# -----------------------------------------------------------------
# Simulate journal assignments
np.random.seed(42)
journal_pool = [f"Journal_{i:02d}" for i in range(20)]
journal_weights = np.exp(-0.3 * np.arange(20))  # Bradford-like decay
journal_weights /= journal_weights.sum()
df["journal"] = np.random.choice(journal_pool, len(df), p=journal_weights)

journal_counts = df.groupby("journal").size().reset_index(name="n_papers")
journal_counts = journal_counts.sort_values("n_papers", ascending=False)
journal_counts["rank"] = range(1, len(journal_counts) + 1)
journal_counts["cumsum"] = journal_counts["n_papers"].cumsum()
total_papers = len(df)

# Bradford zones
one_third = total_papers // 3
zone1 = journal_counts[journal_counts["cumsum"] <= one_third]
zone2 = journal_counts[(journal_counts["cumsum"] > one_third) &
                        (journal_counts["cumsum"] <= 2 * one_third)]
zone3 = journal_counts[journal_counts["cumsum"] > 2 * one_third]
print(f"\nBradford's Law: Core={len(zone1)}, Zone2={len(zone2)}, Zone3={len(zone3)} journals")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Citation distribution (log scale)
axes[0, 0].hist(df["citations"] + 1, bins=30, log=True, color="steelblue", edgecolor="k")
h_val = metrics["h_index"]
axes[0, 0].axvline(h_val, color="red", ls="--", label=f"h-index = {h_val}")
axes[0, 0].set_xlabel("Citations + 1 (log scale)")
axes[0, 0].set_ylabel("Frequency (log)")
axes[0, 0].set_title("Citation Distribution")
axes[0, 0].legend()

# Publication trend
axes[0, 1].bar(annual_counts["year"], annual_counts["n_papers"],
               color="steelblue", alpha=0.7)
trend_y = intercept + slope * annual_counts["year"]
axes[0, 1].plot(annual_counts["year"], trend_y, "r-", lw=2, label=f"Trend ({slope:+.2f}/yr)")
axes[0, 1].set_xlabel("Year"); axes[0, 1].set_ylabel("Publications")
axes[0, 1].set_title("Annual Publication Count")
axes[0, 1].legend()

# Hirsch plot (h-index visual)
sorted_cit = np.sort(df["citations"].values)[::-1]
x_papers = np.arange(1, len(sorted_cit) + 1)
axes[1, 0].bar(x_papers[:50], sorted_cit[:50], color="steelblue", alpha=0.8)
axes[1, 0].plot([0, 50], [0, 50], "r-", lw=1.5, label="y=x line")
axes[1, 0].axvline(h_val, color="orange", ls="--", label=f"h={h_val}")
axes[1, 0].set_xlabel("Paper Rank"); axes[1, 0].set_ylabel("Citations")
axes[1, 0].set_title("Hirsch Plot (top 50 papers)")
axes[1, 0].legend()

# Bradford's Law
axes[1, 1].plot(np.log(journal_counts["rank"]),
                journal_counts["n_papers"], "o-", color="steelblue", ms=4)
axes[1, 1].set_xlabel("ln(Journal Rank)")
axes[1, 1].set_ylabel("Papers")
axes[1, 1].set_title("Bradford's Law (Journal Productivity)")
# Zone boundaries
for zone_label, cutoff in [("Zone 1", len(zone1)), ("Zone 2", len(zone1)+len(zone2))]:
    axes[1, 1].axvline(np.log(cutoff + 1), color="red", ls="--", alpha=0.5,
                       label=zone_label)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("bibliometric_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: bibliometric_analysis.png")
```

### Step 3: Co-authorship Network Analysis

```python
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------------------------------------------
# Build co-authorship network from author pairs
# -----------------------------------------------------------------
def build_coauthorship_network(df, author_col="authors", sep="; "):
    """Build weighted co-authorship graph.

    Args:
        df: DataFrame with author column (semicolon-separated)
        author_col: column name
        sep: separator between authors
    Returns:
        G: NetworkX weighted graph
    """
    G = nx.Graph()
    edge_weights = Counter()

    for _, row in df.iterrows():
        authors = [a.strip() for a in str(row[author_col]).split(sep) if a.strip()]
        # Add nodes with paper count
        for a in authors:
            if G.has_node(a):
                G.nodes[a]["papers"] = G.nodes[a].get("papers", 0) + 1
            else:
                G.add_node(a, papers=1)
        # Add edges for all pairs
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                key = tuple(sorted([authors[i], authors[j]]))
                edge_weights[key] += 1

    for (a, b), w in edge_weights.items():
        G.add_edge(a, b, weight=w)

    return G

G = build_coauthorship_network(df)

# -----------------------------------------------------------------
# Network metrics
# -----------------------------------------------------------------
print("=== Co-authorship Network ===")
print(f"Nodes (authors): {G.number_of_nodes()}")
print(f"Edges (collaborations): {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
print(f"Density: {nx.density(G):.4f}")

components = list(nx.connected_components(G))
largest_cc = max(components, key=len)
print(f"Connected components: {len(components)}")
print(f"Largest component: {len(largest_cc)} authors "
      f"({len(largest_cc)/G.number_of_nodes()*100:.1f}%)")

# Centrality measures on the largest component
G_lcc = G.subgraph(largest_cc).copy()
degree_cent = nx.degree_centrality(G_lcc)
betw_cent = nx.betweenness_centrality(G_lcc, weight="weight")
eigv_cent = nx.eigenvector_centrality(G_lcc, weight="weight", max_iter=200)

# Top authors by each metric
def top_n(centrality_dict, n=5):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

print("\nTop 5 by degree centrality:")
for a, c in top_n(degree_cent): print(f"  {a}: {c:.3f}")
print("\nTop 5 by betweenness centrality:")
for a, c in top_n(betw_cent): print(f"  {a}: {c:.3f}")

# Community detection (Louvain via greedy modularity)
communities = nx.community.greedy_modularity_communities(G_lcc, weight="weight")
modularity = nx.community.modularity(G_lcc, communities, weight="weight")
print(f"\nLouvain communities: {len(communities)}, Modularity = {modularity:.4f}")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Co-authorship network
pos = nx.spring_layout(G_lcc, seed=42, k=1.5)
node_sizes = [G_lcc.nodes[n].get("papers", 1) * 50 for n in G_lcc.nodes()]
# Color by community
community_map = {}
for c_idx, comm in enumerate(communities):
    for author in comm:
        community_map[author] = c_idx
node_colors = [community_map.get(n, 0) for n in G_lcc.nodes()]

nx.draw_networkx_nodes(G_lcc, pos, ax=axes[0], node_size=node_sizes,
                       node_color=node_colors, cmap=plt.cm.Set1, alpha=0.8)
edges = G_lcc.edges(data=True)
weights = [e[2].get("weight", 1) for e in edges]
nx.draw_networkx_edges(G_lcc, pos, ax=axes[0], alpha=0.2,
                       width=[w * 0.5 for w in weights])
if len(G_lcc) < 30:
    nx.draw_networkx_labels(G_lcc, pos, ax=axes[0], font_size=6)
axes[0].set_title(f"Co-authorship Network\n({G_lcc.number_of_nodes()} nodes, "
                  f"{len(communities)} communities)")
axes[0].axis("off")

# Degree distribution
degrees = [d for _, d in G.degree()]
axes[1].hist(degrees, bins=20, color="steelblue", edgecolor="black", log=True)
axes[1].set_xlabel("Degree (collaborators)")
axes[1].set_ylabel("Frequency (log)")
axes[1].set_title("Degree Distribution of Co-authorship Network")

plt.tight_layout()
plt.savefig("coauthorship_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: coauthorship_network.png")
```

## Advanced Usage

### Concept Co-occurrence Analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

def concept_cooccurrence_matrix(df, concept_col="concepts", sep="; ", top_n=15):
    """Build concept co-occurrence matrix.

    Args:
        df: DataFrame with concept column
        top_n: number of most frequent concepts to include
    Returns:
        co_matrix (DataFrame), top_concepts (list)
    """
    # Count concept frequencies
    all_concepts = []
    concept_lists = []
    for _, row in df.iterrows():
        concepts = [c.strip() for c in str(row[concept_col]).split(sep) if c.strip()]
        concept_lists.append(concepts)
        all_concepts.extend(concepts)

    freq = Counter(all_concepts)
    top_concepts = [c for c, _ in freq.most_common(top_n)]

    # Build co-occurrence matrix
    co_matrix = pd.DataFrame(0, index=top_concepts, columns=top_concepts)
    for concepts in concept_lists:
        filtered = [c for c in concepts if c in top_concepts]
        for c1, c2 in combinations(filtered, 2):
            co_matrix.loc[c1, c2] += 1
            co_matrix.loc[c2, c1] += 1
        for c in filtered:
            co_matrix.loc[c, c] += 1  # self-count = frequency

    return co_matrix, top_concepts

co_matrix, top_concepts = concept_cooccurrence_matrix(df, top_n=10)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(co_matrix.values, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(top_concepts)))
ax.set_xticklabels(top_concepts, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(top_concepts)))
ax.set_yticklabels(top_concepts, fontsize=8)
ax.set_title("Concept Co-occurrence Matrix")
plt.colorbar(im, ax=ax, label="Co-occurrence count")
plt.tight_layout()
plt.savefig("concept_cooccurrence.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved: concept_cooccurrence.png")
```

### Institutional Collaboration Network

```python
import networkx as nx
import pandas as pd
from collections import Counter
from itertools import combinations

def institution_network(df, inst_col="institutions", sep="; "):
    """Build inter-institutional collaboration network."""
    G = nx.Graph()
    edge_w = Counter()

    for _, row in df.iterrows():
        insts = [i.strip() for i in str(row[inst_col]).split(sep) if i.strip()]
        insts = list(set(insts))  # unique per paper
        for inst in insts:
            if inst not in G:
                G.add_node(inst, papers=0)
            G.nodes[inst]["papers"] = G.nodes[inst].get("papers", 0) + 1
        for i, j in combinations(insts, 2):
            edge_w[tuple(sorted([i, j]))] += 1

    for (a, b), w in edge_w.items():
        G.add_edge(a, b, weight=w)

    return G

G_inst = institution_network(df)
print("\n=== Institutional Collaboration Network ===")
print(f"Institutions: {G_inst.number_of_nodes()}")
print(f"Collaborations: {G_inst.number_of_edges()}")

# Top institutions by paper count
top_insts = sorted(G_inst.nodes(data=True),
                   key=lambda x: x[1].get("papers", 0), reverse=True)[:5]
for inst, attrs in top_insts:
    print(f"  {inst}: {attrs.get('papers', 0)} papers, "
          f"degree={G_inst.degree(inst)}")
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| OpenAlex API 429 rate limit | Too many requests | Add `time.sleep(0.1)` between requests; use email in params |
| Empty API results | Query too specific | Broaden search terms; check OpenAlex concept IDs |
| h-index lower than expected | Incomplete citation counts | OpenAlex may lag Scopus/WoS; check `cited_by_count` |
| Network too large to visualize | Many authors | Filter to largest component; use degree threshold |
| Self-citations inflate h-index | All citations included | Filter out self-citations using author IDs |
| Duplicate authors (name variations) | Name disambiguation | Use OpenAlex author IDs, not display names |

## External Resources

- Hirsch, J. E. (2005). An index to quantify an individual's scientific research output. *PNAS*, 102(46).
- Egghe, L. (2006). Theory and practise of the g-index. *Scientometrics*, 69(1).
- [OpenAlex documentation](https://docs.openalex.org/)
- [PyAlex Python wrapper](https://github.com/J535D165/pyalex)
- [VOSviewer](https://www.vosviewer.com/) — visualization of bibliometric networks
- Bradford, S. C. (1934). Sources of information on specific subjects. *Engineering*, 137, 85-86.

## Examples

### Example 1: Compare Two Research Fields

```python
import numpy as np
import pandas as pd

def compare_fields(field_a_papers, field_b_papers, field_a_name, field_b_name):
    """Compare bibliometric profiles of two research fields."""
    metrics_a = compute_indices_simple(field_a_papers)
    metrics_b = compute_indices_simple(field_b_papers)

    comparison = pd.DataFrame({
        field_a_name: metrics_a,
        field_b_name: metrics_b
    })
    return comparison

def compute_indices_simple(citations):
    """Compute bibliometric indices from citation array."""
    c = np.sort(citations)[::-1]
    h = sum(1 for i, ci in enumerate(c) if ci >= i + 1)
    cumsum = np.cumsum(c)
    g = sum(1 for i in range(len(c)) if cumsum[i] >= (i + 1)**2)
    return {
        "n_papers": len(c),
        "total_citations": int(c.sum()),
        "mean_citations": float(c.mean()),
        "h_index": h,
        "g_index": g,
        "i10_index": int((c >= 10).sum()),
    }

np.random.seed(42)
# Field A: older, more established
citations_a = np.random.zipf(1.6, 150) * 8
# Field B: newer, fewer but higher-impact papers
citations_b = np.random.zipf(1.4, 80) * 12

comparison = compare_fields(citations_a, citations_b, "Field_A", "Field_B")
print(comparison)
```

### Example 2: Research Trend Forecasting

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fit ARIMA-like trend to annual publication counts
np.random.seed(42)
years = np.arange(2000, 2024)
# Simulate exponential growth in ML publications
pub_counts = (50 * np.exp(0.12 * (years - 2000))
              + np.random.normal(0, 10, len(years))).astype(int)

# Simple polynomial trend + forecast
t = years - 2000
coeff = np.polyfit(t, pub_counts, deg=2)
poly = np.poly1d(coeff)

forecast_years = np.arange(2024, 2028)
forecast_t = forecast_years - 2000
forecast = poly(forecast_t)

print("=== Research Trend Forecast ===")
print(f"Trend equation: {coeff[0]:.2f}t² + {coeff[1]:.2f}t + {coeff[2]:.2f}")
for yr, fc in zip(forecast_years, forecast):
    print(f"  {yr}: {int(fc)} papers (projected)")
```
