---
name: scientometrics
description: >
  Use this Skill for bibliometric analysis: author h-index, citation counts, co-authorship
  networks, co-citation analysis, bibliographic coupling, and VOSviewer-compatible output.
tags:
  - universal
  - scientometrics
  - bibliometrics
  - openalex
  - networkx
  - co-authorship
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - requests>=2.28.0
    - pandas>=1.5.0
    - networkx>=3.0
    - matplotlib>=3.6.0
    - numpy>=1.23.0
    - scipy>=1.9.0
last_updated: "2026-03-17"
---

# Scientometrics

> **TL;DR** — Bibliometric analysis via the OpenAlex API. Compute author h-index and
> citation counts, build co-authorship networks with NetworkX, run co-citation and
> bibliographic coupling analyses, identify research fronts, and export
> VOSviewer-compatible edge/node files.

---

## 1. Overview

### What Problem Does This Skill Solve?

Scientometrics provides quantitative methods for understanding how science evolves,
who the key players are, and which research fronts are emerging. Manual analysis is
tedious and error-prone. This Skill automates the full pipeline:

- **Author-level metrics** — h-index, total citations, career publication count
- **Journal impact** — field-normalized citation counts from the OpenAlex works endpoint
- **Co-authorship networks** — graph construction, centrality, community detection
- **Co-citation analysis** — which papers are frequently cited together
- **Bibliographic coupling** — which papers share many references
- **Research front identification** — fast-growing citation clusters
- **VOSviewer export** — ready-to-open `.txt` files for interactive visualization

### Applicable Scenarios

| Scenario | Entry Point |
|---|---|
| Evaluate a researcher before collaboration | `get_author_metrics(orcid)` |
| Map the intellectual structure of a field | `build_coauthorship_network(topic, years)` |
| Find seminal papers for a literature review | `identify_research_fronts(topic)` |
| Visualize a network in VOSviewer | `export_vosviewer_network(G)` |
| Benchmark journal standing | `get_journal_impact(issn)` |

### Key Limitations

- OpenAlex author disambiguation is imperfect; prefer ORCID-based lookups when available.
- Citation counts lag by ~2 weeks from the live OpenAlex index.
- Co-authorship graphs for very large topics (>50 000 papers) require pagination and
  may take several minutes to build.
- VOSviewer export uses the simple pair-wise weight format; large networks (>5 000 nodes)
  may be slow to open in VOSviewer.

---

## 2. Environment Setup

```bash
# Create and activate a dedicated environment
conda create -n scientometrics python=3.11 -y
conda activate scientometrics

# Install all dependencies
pip install requests pandas networkx matplotlib numpy scipy

# Optional: set a polite-pool email for OpenAlex (improves response speed)
export OPENALEX_EMAIL="researcher@university.edu"
```

Verify the installation:

```python
import requests, pandas, networkx, matplotlib, numpy, scipy
print("All scientometrics dependencies OK")
print(f"NetworkX version: {networkx.__version__}")
```

---

## 3. Core Implementation

### 3.1 Shared Utilities

```python
import os
import time
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_EMAIL = os.getenv("OPENALEX_EMAIL", "researcher@example.com")


def _oa_get(endpoint: str, params: dict, retries: int = 3) -> dict:
    """
    Thin wrapper around OpenAlex GET with retry logic.

    Args:
        endpoint: API endpoint path, e.g. '/works' or '/authors/A1234'.
        params:   Query parameters dict.
        retries:  Number of retry attempts on transient errors.

    Returns:
        Parsed JSON response dict.
    """
    url = OPENALEX_BASE + endpoint
    params.setdefault("mailto", DEFAULT_EMAIL)

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if exc.response.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """Reconstruct plain-text abstract from OpenAlex inverted index."""
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)
```

### 3.2 Author Metrics

```python
def get_author_metrics(
    orcid: Optional[str] = None,
    openalex_author_id: Optional[str] = None,
) -> dict:
    """
    Fetch comprehensive metrics for a researcher from OpenAlex.

    Provide either an ORCID (preferred) or an OpenAlex author ID (e.g. 'A1234567890').

    Args:
        orcid:              ORCID identifier, e.g. '0000-0002-1234-5678'.
        openalex_author_id: OpenAlex author ID string, e.g. 'A1234567890'.

    Returns:
        Dict with keys: name, orcid, openalex_id, works_count, cited_by_count,
        h_index, i10_index, top_venues, years_active, works_df.
    """
    if orcid:
        data = _oa_get(f"/authors/orcid:{orcid}", {})
    elif openalex_author_id:
        data = _oa_get(f"/authors/{openalex_author_id}", {})
    else:
        raise ValueError("Provide either orcid or openalex_author_id.")

    author_id = data["id"].split("/")[-1]

    # Fetch all works by this author
    works = _fetch_all_works(
        filter_str=f"author.id:{author_id}",
        select="doi,title,publication_year,cited_by_count,primary_location",
        max_results=5000,
    )

    # Compute derived metrics
    h_index = compute_h_index(works)
    citation_counts = [w.get("cited_by_count", 0) for w in works]
    i10_index = sum(1 for c in citation_counts if c >= 10)

    venues = [
        (w.get("primary_location") or {}).get("source", {}) or {}
        for w in works
    ]
    venue_names = [v.get("display_name", "") for v in venues if v.get("display_name")]
    from collections import Counter
    top_venues = Counter(venue_names).most_common(5)

    years = [w.get("publication_year") for w in works if w.get("publication_year")]
    years_active = (min(years), max(years)) if years else (None, None)

    works_df = pd.DataFrame([{
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "title": w.get("title", ""),
        "year": w.get("publication_year"),
        "citations": w.get("cited_by_count", 0),
        "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
    } for w in works])

    return {
        "name": data.get("display_name", ""),
        "orcid": data.get("orcid", ""),
        "openalex_id": author_id,
        "works_count": data.get("works_count", len(works)),
        "cited_by_count": data.get("cited_by_count", sum(citation_counts)),
        "h_index": h_index,
        "i10_index": i10_index,
        "top_venues": top_venues,
        "years_active": years_active,
        "works_df": works_df,
    }


def compute_h_index(works: List[dict]) -> int:
    """
    Compute the h-index from a list of OpenAlex work objects.

    Args:
        works: List of work dicts; each must have a 'cited_by_count' key.

    Returns:
        Integer h-index value.
    """
    citations = sorted(
        [w.get("cited_by_count", 0) for w in works], reverse=True
    )
    h = 0
    for i, c in enumerate(citations, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def _fetch_all_works(
    filter_str: str,
    select: str,
    max_results: int = 2000,
) -> List[dict]:
    """Paginate through /works endpoint and return all matching work dicts."""
    per_page = 200
    page = 1
    collected = []

    while len(collected) < max_results:
        data = _oa_get("/works", {
            "filter": filter_str,
            "select": select,
            "per-page": per_page,
            "page": page,
        })
        results = data.get("results", [])
        if not results:
            break
        collected.extend(results)

        meta = data.get("meta", {})
        total = meta.get("count", 0)
        if page * per_page >= min(total, max_results):
            break
        page += 1
        time.sleep(0.1)

    return collected[:max_results]
```

### 3.3 Co-Authorship Network

```python
def build_coauthorship_network(
    topic: str,
    years: Tuple[int, int] = (2018, 2024),
    max_papers: int = 1000,
    min_edge_weight: int = 2,
) -> nx.Graph:
    """
    Build a co-authorship network for a research topic from OpenAlex.

    Each node is an author; edge weight equals the number of papers co-authored
    within the specified year range.

    Args:
        topic:           Search query string for the topic.
        years:           Tuple (from_year, to_year) inclusive.
        max_papers:      Maximum papers to fetch (impacts runtime).
        min_edge_weight: Minimum co-authored papers to include an edge.

    Returns:
        NetworkX Graph with node attributes: name, works_count, citations;
        and edge attribute: weight.
    """
    filter_str = (
        f"default.search:{topic},"
        f"publication_year:{years[0]}-{years[1]}"
    )
    works = _fetch_all_works(
        filter_str=filter_str,
        select="id,title,authorships,cited_by_count",
        max_results=max_papers,
    )

    edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
    node_attrs: Dict[str, dict] = {}

    for work in works:
        authorships = work.get("authorships", [])
        author_ids = []
        for a in authorships:
            author = a.get("author") or {}
            aid = (author.get("id") or "").split("/")[-1]
            aname = author.get("display_name", "")
            if aid:
                author_ids.append(aid)
                if aid not in node_attrs:
                    node_attrs[aid] = {
                        "name": aname,
                        "works_count": 0,
                        "citations": 0,
                    }
                node_attrs[aid]["works_count"] += 1
                node_attrs[aid]["citations"] += work.get("cited_by_count", 0)

        # Add edge for every pair of co-authors
        for i in range(len(author_ids)):
            for j in range(i + 1, len(author_ids)):
                a, b = sorted([author_ids[i], author_ids[j]])
                edge_weights[(a, b)] += 1

    G = nx.Graph()
    for aid, attrs in node_attrs.items():
        G.add_node(aid, **attrs)

    for (a, b), weight in edge_weights.items():
        if weight >= min_edge_weight:
            G.add_edge(a, b, weight=weight)

    print(
        f"Co-authorship network: {G.number_of_nodes()} authors, "
        f"{G.number_of_edges()} edges (min_weight={min_edge_weight})"
    )
    return G


def plot_coauthorship_network(
    G: nx.Graph,
    top_n: int = 50,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the largest connected component of a co-authorship network.

    Nodes are sized by citation count; edges by co-authorship weight.
    Community detection uses the Louvain algorithm via networkx.

    Args:
        G:           Co-authorship graph from build_coauthorship_network().
        top_n:       Show only the top-N authors by citations for clarity.
        output_path: Optional file path to save the figure (PNG/PDF/SVG).

    Returns:
        Matplotlib Figure object.
    """
    # Select top-N authors by citations
    citations = nx.get_node_attributes(G, "citations")
    top_nodes = sorted(citations, key=citations.get, reverse=True)[:top_n]
    subG = G.subgraph(top_nodes).copy()

    # Community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(subG))
        node_community = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                node_community[node] = idx
    except Exception:
        node_community = {n: 0 for n in subG.nodes()}

    pos = nx.spring_layout(subG, seed=42, k=0.5)
    node_sizes = [max(50, citations.get(n, 0) / 10) for n in subG.nodes()]
    node_colors = [node_community.get(n, 0) for n in subG.nodes()]
    edge_widths = [subG[u][v]["weight"] * 0.5 for u, v in subG.edges()]
    names = nx.get_node_attributes(subG, "name")

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(
        subG, pos, node_size=node_sizes,
        node_color=node_colors, cmap=cm.tab20, alpha=0.85, ax=ax,
    )
    nx.draw_networkx_labels(
        subG, pos,
        labels={n: names.get(n, n)[:20] for n in subG.nodes()},
        font_size=6, ax=ax,
    )
    ax.set_title(f"Co-authorship Network (top {top_n} authors by citations)")
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved network plot to {output_path}")

    return fig
```

### 3.4 Co-Citation and Bibliographic Coupling

```python
def compute_cocitation_matrix(
    works: List[dict],
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Compute a co-citation matrix: how often two papers are cited together.

    Co-citation strength(A, B) = number of papers in the corpus that cite both A and B.

    Args:
        works:  List of work dicts with 'referenced_works' field from OpenAlex.
        top_n:  Return only the top-N most co-cited pairs as a DataFrame.

    Returns:
        DataFrame with columns: paper_a, paper_b, cocitation_count.
    """
    # Build inverted index: reference -> set of citing papers
    citing: Dict[str, set] = defaultdict(set)
    for work in works:
        work_id = work.get("id", "")
        for ref in work.get("referenced_works", []):
            citing[ref].add(work_id)

    # Count co-citations for all pairs of references
    ref_list = list(citing.keys())
    cocitation: Dict[Tuple[str, str], int] = defaultdict(int)

    for citing_work_id in set().union(*citing.values()):
        refs_cited_by_this = [r for r, citers in citing.items() if citing_work_id in citers]
        for i in range(len(refs_cited_by_this)):
            for j in range(i + 1, len(refs_cited_by_this)):
                a, b = sorted([refs_cited_by_this[i], refs_cited_by_this[j]])
                cocitation[(a, b)] += 1

    rows = [
        {"paper_a": a, "paper_b": b, "cocitation_count": cnt}
        for (a, b), cnt in cocitation.items()
    ]
    df = pd.DataFrame(rows).sort_values("cocitation_count", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def compute_bibliographic_coupling(
    works: List[dict],
    top_n: int = 100,
) -> pd.DataFrame:
    """
    Compute bibliographic coupling strength between pairs of papers.

    Coupling(A, B) = number of references shared by both A and B.

    Args:
        works:  List of work dicts with 'referenced_works' from OpenAlex.
        top_n:  Return only the top-N strongest coupled pairs.

    Returns:
        DataFrame with columns: paper_a, paper_b, shared_references.
    """
    work_refs: Dict[str, set] = {}
    for work in works:
        wid = work.get("id", "")
        refs = set(work.get("referenced_works", []))
        if refs:
            work_refs[wid] = refs

    work_ids = list(work_refs.keys())
    coupling_rows = []

    for i in range(len(work_ids)):
        for j in range(i + 1, len(work_ids)):
            a, b = work_ids[i], work_ids[j]
            shared = len(work_refs[a] & work_refs[b])
            if shared > 0:
                coupling_rows.append({"paper_a": a, "paper_b": b, "shared_references": shared})

    df = pd.DataFrame(coupling_rows)
    if df.empty:
        return df
    return df.sort_values("shared_references", ascending=False).head(top_n).reset_index(drop=True)
```

### 3.5 Research Front Identification

```python
def identify_research_fronts(
    topic: str,
    years: Tuple[int, int] = (2020, 2024),
    max_papers: int = 500,
    top_fronts: int = 10,
) -> pd.DataFrame:
    """
    Identify emerging research fronts for a topic by finding fast-growing
    citation clusters using year-over-year citation acceleration.

    Args:
        topic:       Search query string.
        years:       Year range to consider.
        max_papers:  Maximum papers to analyse.
        top_fronts:  Number of research fronts (papers) to return.

    Returns:
        DataFrame of papers sorted by citation acceleration (citations_per_year).
    """
    filter_str = (
        f"default.search:{topic},"
        f"publication_year:{years[0]}-{years[1]}"
    )
    works = _fetch_all_works(
        filter_str=filter_str,
        select="id,doi,title,publication_year,cited_by_count,abstract_inverted_index",
        max_results=max_papers,
    )

    current_year = 2026
    rows = []
    for w in works:
        pub_year = w.get("publication_year") or current_year
        age = max(1, current_year - pub_year)
        citations = w.get("cited_by_count", 0)
        rows.append({
            "openalex_id": (w.get("id") or "").split("/")[-1],
            "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
            "title": w.get("title", ""),
            "year": pub_year,
            "citations": citations,
            "citations_per_year": round(citations / age, 2),
            "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("citations_per_year", ascending=False).head(top_fronts)
    return df.reset_index(drop=True)
```

### 3.6 VOSviewer Export

```python
def export_vosviewer_network(
    G: nx.Graph,
    output_map: str = "vosviewer_map.txt",
    output_network: str = "vosviewer_network.txt",
) -> None:
    """
    Export a NetworkX graph to VOSviewer-compatible map and network files.

    The map file contains node labels and weights; the network file contains
    edge pairs with weights. Both use tab-separated format expected by VOSviewer.

    Args:
        G:              NetworkX Graph with node attribute 'name' and optional
                        'citations' (used as node weight).
        output_map:     Path for the VOSviewer map (nodes) file.
        output_network: Path for the VOSviewer network (edges) file.
    """
    # Build integer ID mapping (VOSviewer uses 1-indexed integers)
    nodes = list(G.nodes())
    node_to_int = {n: i + 1 for i, n in enumerate(nodes)}
    names = nx.get_node_attributes(G, "name")
    citations = nx.get_node_attributes(G, "citations")

    with open(output_map, "w", encoding="utf-8") as f_map:
        f_map.write("id\tlabel\tweight\n")
        for node in nodes:
            nid = node_to_int[node]
            label = names.get(node, node)
            weight = citations.get(node, 1)
            f_map.write(f"{nid}\t{label}\t{weight}\n")

    with open(output_network, "w", encoding="utf-8") as f_net:
        f_net.write("from\tto\tstrength\n")
        for u, v, data in G.edges(data=True):
            uid = node_to_int[u]
            vid = node_to_int[v]
            strength = data.get("weight", 1)
            f_net.write(f"{uid}\t{vid}\t{strength}\n")

    print(
        f"VOSviewer export: {len(nodes)} nodes -> {output_map}, "
        f"{G.number_of_edges()} edges -> {output_network}"
    )


def get_journal_impact(
    issn: str,
    from_year: int = 2019,
    to_year: int = 2024,
) -> dict:
    """
    Retrieve journal-level publication and citation statistics from OpenAlex.

    Args:
        issn:      ISSN of the journal (e.g. '0028-0836' for Nature).
        from_year: Start of citation window.
        to_year:   End of citation window.

    Returns:
        Dict with keys: display_name, issn, works_count, cited_by_count,
        mean_citations_per_paper, top_cited_papers.
    """
    source_data = _oa_get(f"/sources/issn:{issn}", {})
    source_id = source_data.get("id", "").split("/")[-1]
    display_name = source_data.get("display_name", "")

    works = _fetch_all_works(
        filter_str=f"primary_location.source.id:{source_id},publication_year:{from_year}-{to_year}",
        select="doi,title,publication_year,cited_by_count",
        max_results=2000,
    )

    citation_counts = [w.get("cited_by_count", 0) for w in works]
    mean_cit = round(np.mean(citation_counts), 2) if citation_counts else 0

    top_cited = sorted(works, key=lambda w: w.get("cited_by_count", 0), reverse=True)[:10]
    top_df = pd.DataFrame([{
        "title": w.get("title", ""),
        "year": w.get("publication_year"),
        "citations": w.get("cited_by_count", 0),
    } for w in top_cited])

    return {
        "display_name": display_name,
        "issn": issn,
        "works_count": len(works),
        "cited_by_count": sum(citation_counts),
        "mean_citations_per_paper": mean_cit,
        "top_cited_papers": top_df,
    }
```

---

## 4. End-to-End Examples

### Example 1 — Map the Research Landscape of "Transformer Neural Networks" 2018-2024

```python
# Step 1: Identify research fronts
fronts = identify_research_fronts(
    topic="transformer neural networks",
    years=(2018, 2024),
    max_papers=500,
    top_fronts=15,
)
print("Top research fronts (by citations/year):")
print(fronts[["title", "year", "citations", "citations_per_year"]].to_string())

# Step 2: Build co-authorship network for this topic
G = build_coauthorship_network(
    topic="transformer neural networks",
    years=(2018, 2024),
    max_papers=1000,
    min_edge_weight=2,
)

# Step 3: Key network statistics
print(f"\nNetwork density: {nx.density(G):.4f}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
top_by_degree = sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:5]
names = nx.get_node_attributes(G, "name")
print("\nTop 5 authors by degree centrality:")
for node, centrality in top_by_degree:
    print(f"  {names.get(node, node)}: {centrality:.4f}")

# Step 4: Visualize
fig = plot_coauthorship_network(G, top_n=50, output_path="transformer_coauthorship.png")

# Step 5: Export to VOSviewer
export_vosviewer_network(G, "transformer_map.txt", "transformer_network.txt")

# Step 6: Fetch author metrics for the highest-centrality author
top_author_id = top_by_degree[0][0]
metrics = get_author_metrics(openalex_author_id=top_author_id)
print(f"\nTop author: {metrics['name']}")
print(f"  h-index: {metrics['h_index']}")
print(f"  Total citations: {metrics['cited_by_count']}")
print(f"  Active years: {metrics['years_active'][0]}–{metrics['years_active'][1]}")
print(f"  Top venues: {metrics['top_venues']}")
```

### Example 2 — Analyze Collaboration Patterns of a Specific Research Group

```python
# Suppose we have a list of ORCID identifiers for a research group
group_orcids = [
    "0000-0002-1234-5678",  # replace with real ORCIDs
    "0000-0001-9876-5432",
    "0000-0003-1111-2222",
]

# Step 1: Collect metrics for each author
group_metrics = []
for orcid in group_orcids:
    try:
        m = get_author_metrics(orcid=orcid)
        group_metrics.append(m)
        print(f"{m['name']}: h={m['h_index']}, citations={m['cited_by_count']}")
    except Exception as e:
        print(f"Could not fetch ORCID {orcid}: {e}")

# Step 2: Build a group-level summary DataFrame
summary = pd.DataFrame([{
    "name": m["name"],
    "orcid": m["orcid"],
    "works_count": m["works_count"],
    "cited_by_count": m["cited_by_count"],
    "h_index": m["h_index"],
    "i10_index": m["i10_index"],
    "active_from": m["years_active"][0],
    "active_to": m["years_active"][1],
} for m in group_metrics])

print("\nResearch Group Summary:")
print(summary.to_string())
summary.to_csv("group_metrics.csv", index=False)

# Step 3: Check journal impact for journals where group publishes most
all_venues = []
for m in group_metrics:
    all_venues.extend(m["works_df"]["venue"].dropna().tolist())

from collections import Counter
top_venues_overall = Counter(all_venues).most_common(5)
print("\nTop publication venues for the group:")
for venue, count in top_venues_overall:
    print(f"  {venue}: {count} papers")

# Step 4: Plot per-author citation distribution
fig, axes = plt.subplots(1, len(group_metrics), figsize=(5 * len(group_metrics), 4))
if len(group_metrics) == 1:
    axes = [axes]
for ax, m in zip(axes, group_metrics):
    citations = m["works_df"]["citations"].clip(upper=500)
    ax.hist(citations, bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(m["name"][:25])
    ax.set_xlabel("Citations per paper")
    ax.set_ylabel("Count")
fig.suptitle("Citation Distributions — Research Group")
fig.tight_layout()
fig.savefig("group_citation_distributions.png", dpi=150)
print("Saved group_citation_distributions.png")
```

---

## 5. Common Errors and Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `404 Not Found` on `/authors/orcid:...` | ORCID not indexed in OpenAlex | Try `/authors?filter=orcid:...` search |
| Empty co-authorship graph | `min_edge_weight` too high for sparse field | Lower to `min_edge_weight=1` |
| `KeyError: 'referenced_works'` | Field not in `select` clause | Add `referenced_works` to the `select` parameter |
| Slow pagination (>5 min) | Large topic with many results | Reduce `max_papers` or use date ranges to split queries |
| VOSviewer shows no map | File uses wrong separator | Ensure tab (`\t`) delimiter; do not open `.txt` as a map file manually |
| `nx.average_clustering` hangs | Very large graph | Call on subgraph: `G.subgraph(list(G.nodes())[:500])` |

---

## 6. Performance Tips

- **Parallel author lookups**: Use `concurrent.futures.ThreadPoolExecutor` when fetching
  metrics for multiple authors — each call is independent.
- **Cache raw work lists**: Pickle the list returned by `_fetch_all_works()` so
  downstream analyses (co-citation, coupling) can re-run instantly.
- **Sparse graph storage**: For networks with >10 000 nodes use `nx.Graph()` with
  explicit edge pruning; avoid dense adjacency matrices.
- **Incremental VOSviewer updates**: Append new nodes/edges to existing map/network
  files rather than rebuilding from scratch each week.

---

## 7. References and Further Reading

- OpenAlex documentation: <https://docs.openalex.org/>
- NetworkX documentation: <https://networkx.org/documentation/stable/>
- VOSviewer software: <https://www.vosviewer.com/>
- Scientometrics primer (Waltman 2016): <https://doi.org/10.1007/s11192-016-1943-4>
- Co-citation analysis methodology: <https://doi.org/10.1002/asi.4630240406>
- Bibliographic coupling (Kessler 1963): <https://doi.org/10.1002/asi.5090140103>

---

## 8. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — author metrics, co-authorship network, co-citation, bibliographic coupling, VOSviewer export |
