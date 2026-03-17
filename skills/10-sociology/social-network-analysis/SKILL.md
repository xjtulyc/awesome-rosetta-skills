---
name: social-network-analysis
description: >
  Build, analyze, and visualize social networks with NetworkX: centrality, community detection,
  small-world metrics, bipartite networks, and Gephi export from edge lists or adjacency matrices.
tags:
  - sociology
  - network-analysis
  - networkx
  - community-detection
  - graph-theory
  - visualization
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
  - networkx>=3.1.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - python-louvain>=0.16
  - scipy>=1.10.0
  - requests>=2.31.0
last_updated: "2026-03-17"
---

# Social Network Analysis with NetworkX

## Overview

Social network analysis (SNA) treats actors (people, organizations, papers) as **nodes** and
their relationships as **edges**. This skill covers the full pipeline from raw edge-list data to
publication-ready network visualizations, covering:

- Building networks from edge lists and adjacency matrices
- Centrality measures (degree, betweenness, closeness, eigenvector, PageRank)
- Community detection (Louvain, Girvan-Newman)
- Small-world analysis
- Bipartite network projection
- Ego networks
- Export to Gephi (`.gexf`, `.graphml`)

---

## Setup

```bash
pip install networkx pandas numpy matplotlib python-louvain scipy requests
```

For Louvain community detection the package is installed as `python-louvain` but imported as
`community`. Verify with:

```python
import community as community_louvain
print(community_louvain.__version__)
```

---

## Core Functions

```python
import os
import json
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from typing import Any

# python-louvain
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False
    print("python-louvain not installed; Louvain community detection unavailable.")


# ---------------------------------------------------------------------------
# 1. Network Construction
# ---------------------------------------------------------------------------


def build_network_from_edgelist(
    df: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str | None = "weight",
    directed: bool = False,
    node_attr_df: pd.DataFrame | None = None,
    node_id_col: str = "node_id",
) -> nx.Graph | nx.DiGraph:
    """
    Build a NetworkX graph from a pandas edge-list DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Edge list with at least source and target columns.
    source_col, target_col : str
        Column names for edge endpoints.
    weight_col : str or None
        Column with edge weights. If None, all weights default to 1.
    directed : bool
        If True, return a DiGraph; otherwise an undirected Graph.
    node_attr_df : pd.DataFrame, optional
        Node attribute table with ``node_id_col`` as key.
    node_id_col : str
        Column in ``node_attr_df`` that matches node identifiers.

    Returns
    -------
    nx.Graph or nx.DiGraph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    for _, row in df.iterrows():
        w = float(row[weight_col]) if weight_col and weight_col in df.columns else 1.0
        G.add_edge(row[source_col], row[target_col], weight=w)

    if node_attr_df is not None:
        attr_dict = node_attr_df.set_index(node_id_col).to_dict(orient="index")
        nx.set_node_attributes(G, attr_dict)

    print(
        f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"{'directed' if directed else 'undirected'}"
    )
    return G


def build_network_from_adjacency(
    matrix: np.ndarray | pd.DataFrame,
    node_labels: list[str] | None = None,
    directed: bool = False,
    threshold: float = 0.0,
) -> nx.Graph:
    """
    Build a NetworkX graph from an adjacency (or weighted) matrix.

    Parameters
    ----------
    matrix : array-like or pd.DataFrame
        Square adjacency matrix. Use a DataFrame for automatic node labels.
    node_labels : list of str, optional
        Labels for rows/columns if ``matrix`` is a numpy array.
    directed : bool
        Build a DiGraph if True.
    threshold : float
        Only include edges where matrix value exceeds this threshold.

    Returns
    -------
    nx.Graph or nx.DiGraph
    """
    if isinstance(matrix, pd.DataFrame):
        node_labels = list(matrix.index)
        matrix = matrix.values
    elif node_labels is None:
        node_labels = [str(i) for i in range(len(matrix))]

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(node_labels)
    n = len(node_labels)
    for i in range(n):
        for j in range(i + 1 if not directed else 0, n):
            if directed and i == j:
                continue
            w = matrix[i, j]
            if w > threshold:
                G.add_edge(node_labels[i], node_labels[j], weight=float(w))
    return G


# ---------------------------------------------------------------------------
# 2. Centrality Measures
# ---------------------------------------------------------------------------


def compute_all_centralities(
    G: nx.Graph,
    weight: str = "weight",
    k_betweenness: int | None = None,
) -> pd.DataFrame:
    """
    Compute degree, betweenness, closeness, eigenvector, and PageRank centralities.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
    weight : str
        Edge attribute to use as weight (set to None for unweighted).
    k_betweenness : int or None
        Number of pivot nodes for approximate betweenness (faster for large graphs).

    Returns
    -------
    pd.DataFrame
        One row per node, columns for each centrality metric, sorted by degree.
    """
    undirected = G.to_undirected() if G.is_directed() else G

    degree_c = dict(G.degree(weight=weight))
    degree_c_norm = {n: v / (G.number_of_nodes() - 1) for n, v in degree_c.items()}

    bet_c = nx.betweenness_centrality(G, weight=weight, normalized=True, k=k_betweenness)
    close_c = nx.closeness_centrality(undirected)

    try:
        eig_c = nx.eigenvector_centrality_numpy(undirected, weight=weight)
    except nx.PowerIterationFailedConvergence:
        eig_c = {n: float("nan") for n in G.nodes()}

    pr = nx.pagerank(G if G.is_directed() else undirected, weight=weight)

    records = []
    for node in G.nodes():
        records.append({
            "node": node,
            "degree": degree_c.get(node, 0),
            "degree_centrality": degree_c_norm.get(node, 0.0),
            "betweenness": bet_c.get(node, 0.0),
            "closeness": close_c.get(node, 0.0),
            "eigenvector": eig_c.get(node, 0.0),
            "pagerank": pr.get(node, 0.0),
            **{k: v for k, v in G.nodes[node].items()},
        })

    df = pd.DataFrame(records).sort_values("degree", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Community Detection
# ---------------------------------------------------------------------------


def detect_communities_louvain(
    G: nx.Graph,
    weight: str = "weight",
    resolution: float = 1.0,
    random_state: int = 42,
) -> dict[Any, int]:
    """
    Detect communities using the Louvain algorithm (python-louvain).

    Parameters
    ----------
    G : nx.Graph
        Undirected graph (directed graphs are converted).
    weight : str
        Edge weight attribute.
    resolution : float
        Resolution parameter; higher values yield smaller communities.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping {node: community_id}.
    """
    if not HAS_LOUVAIN:
        raise ImportError("Install python-louvain: pip install python-louvain")
    undirected = G.to_undirected() if G.is_directed() else G
    partition = community_louvain.best_partition(
        undirected,
        weight=weight,
        resolution=resolution,
        random_state=random_state,
    )
    nx.set_node_attributes(undirected, partition, "community")
    n_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, undirected, weight=weight)
    print(f"Louvain: {n_communities} communities, modularity = {modularity:.4f}")
    return partition


def detect_communities_girvan_newman(
    G: nx.Graph,
    n_communities: int = 5,
) -> list[frozenset]:
    """
    Detect communities using the Girvan-Newman edge-betweenness algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph.
    n_communities : int
        Target number of communities to extract.

    Returns
    -------
    list of frozenset
        Each frozenset is a community of node identifiers.
    """
    comp = nx.community.girvan_newman(G)
    for communities in itertools.islice(comp, n_communities - 1):
        pass
    result = sorted(communities, key=len, reverse=True)
    print(f"Girvan-Newman: {len(result)} communities")
    return result


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------


def visualize_network(
    G: nx.Graph,
    node_color_attr: str | None = "community",
    node_size_attr: str | None = "degree",
    layout: str = "spring",
    title: str = "Network Graph",
    figsize: tuple = (14, 10),
    save_path: str | None = None,
    seed: int = 42,
) -> plt.Figure:
    """
    Visualize a NetworkX graph with attribute-driven node colors and sizes.

    Parameters
    ----------
    G : nx.Graph
        Graph to draw.
    node_color_attr : str or None
        Node attribute used for color mapping. Use None for uniform color.
    node_size_attr : str or None
        Node attribute used for size scaling. Use None for uniform size.
    layout : str
        Layout algorithm: ``"spring"``, ``"circular"``, ``"kamada_kawai"``, ``"random"``.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Save figure path.
    seed : int
        Random seed for layout reproducibility.

    Returns
    -------
    matplotlib.figure.Figure
    """
    layout_funcs = {
        "spring": lambda g: nx.spring_layout(g, seed=seed, k=1 / np.sqrt(g.number_of_nodes())),
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "random": lambda g: nx.random_layout(g, seed=seed),
    }
    pos = layout_funcs.get(layout, layout_funcs["spring"])(G)

    # Node colors
    if node_color_attr and nx.get_node_attributes(G, node_color_attr):
        raw_colors = [G.nodes[n].get(node_color_attr, 0) for n in G.nodes()]
        unique_vals = sorted(set(raw_colors))
        palette = cm.get_cmap("tab20", len(unique_vals))
        color_map = {v: palette(i) for i, v in enumerate(unique_vals)}
        node_colors = [color_map[c] for c in raw_colors]
    else:
        node_colors = ["steelblue"] * G.number_of_nodes()

    # Node sizes
    if node_size_attr and nx.get_node_attributes(G, node_size_attr):
        sizes_raw = np.array([G.nodes[n].get(node_size_attr, 1) for n in G.nodes()], dtype=float)
        sizes = 100 + 1500 * (sizes_raw - sizes_raw.min()) / (sizes_raw.ptp() + 1e-9)
    else:
        sizes = [200] * G.number_of_nodes()

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray", ax=ax, width=0.8)

    # Only label top-degree nodes to avoid clutter
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[: min(20, G.number_of_nodes())]
    labels = {n: str(n) for n, _ in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Gephi Export
# ---------------------------------------------------------------------------


def export_to_gephi(
    G: nx.Graph,
    output_prefix: str = "network",
    format: str = "gexf",
) -> None:
    """
    Export graph to Gephi-compatible formats (.gexf or .graphml).

    Parameters
    ----------
    G : nx.Graph
        Graph with optional node attributes.
    output_prefix : str
        Base filename (without extension).
    format : str
        ``"gexf"`` or ``"graphml"``.
    """
    path = f"{output_prefix}.{format}"
    if format == "gexf":
        nx.write_gexf(G, path)
    elif format == "graphml":
        nx.write_graphml(G, path)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'gexf' or 'graphml'.")
    print(f"Graph exported to {path}")


# ---------------------------------------------------------------------------
# 6. Small-World Analysis
# ---------------------------------------------------------------------------


def compute_small_world_metrics(
    G: nx.Graph,
    n_random: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compute clustering coefficient and average path length, compare to random graphs.

    Parameters
    ----------
    G : nx.Graph
        Input graph (should be connected for path length; uses largest component).
    n_random : int
        Number of random Erdos-Renyi graphs to average over.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: ``clustering``, ``avg_path_length``, ``sigma`` (small-world coefficient),
    ``omega``, ``random_clustering``, ``random_path_length``.
    """
    # Use largest connected component
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    n, m = lcc.number_of_nodes(), lcc.number_of_edges()
    p = m / (n * (n - 1) / 2)

    C = nx.average_clustering(lcc, weight="weight")
    L = nx.average_shortest_path_length(lcc)

    rng = np.random.default_rng(seed)
    C_rand_list, L_rand_list = [], []
    for s in rng.integers(0, 10_000, n_random):
        G_rand = nx.erdos_renyi_graph(n, p, seed=int(s))
        cc = nx.connected_components(G_rand)
        lcc_rand = G_rand.subgraph(max(cc, key=len)).copy()
        C_rand_list.append(nx.average_clustering(lcc_rand))
        L_rand_list.append(nx.average_shortest_path_length(lcc_rand))

    C_rand = np.mean(C_rand_list)
    L_rand = np.mean(L_rand_list)

    sigma = (C / C_rand) / (L / L_rand)  # sigma > 1 indicates small-world

    return {
        "n_nodes": n,
        "n_edges": m,
        "clustering": C,
        "avg_path_length": L,
        "random_clustering": C_rand,
        "random_path_length": L_rand,
        "sigma": sigma,
        "is_small_world": sigma > 1.0,
    }
```

---

## Example A: Co-Authorship Network from OpenAlex Data

This example builds a co-authorship network from OpenAlex API results, detects communities, and
exports a Gephi-ready file.

```python
# ── Example A ─────────────────────────────────────────────────────────────
import requests
import time


def fetch_openalex_works(topic: str, n_results: int = 200) -> list[dict]:
    """Fetch works from OpenAlex API for a given topic."""
    url = "https://api.openalex.org/works"
    works = []
    cursor = "*"
    per_page = min(n_results, 200)

    while len(works) < n_results:
        params = {
            "search": topic,
            "per-page": per_page,
            "cursor": cursor,
            "select": "id,title,authorships",
            "filter": "is_oa:true",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break
        works.extend(results)
        cursor = data.get("meta", {}).get("next_cursor", None)
        if not cursor:
            break
        time.sleep(0.1)  # Respect rate limits

    return works[:n_results]


def works_to_coauthorship_edgelist(works: list[dict]) -> pd.DataFrame:
    """Convert OpenAlex works list to co-authorship edge list."""
    edges = defaultdict(float)
    for work in works:
        authors = [
            a["author"]["display_name"]
            for a in work.get("authorships", [])
            if a.get("author") and a["author"].get("display_name")
        ]
        for a1, a2 in itertools.combinations(authors, 2):
            key = tuple(sorted([a1, a2]))
            edges[key] += 1.0

    records = [{"source": s, "target": t, "weight": w} for (s, t), w in edges.items()]
    return pd.DataFrame(records)


# --- Fetch co-authorship data ------------------------------------------------
TOPIC = "social network analysis"
print(f"Fetching OpenAlex works on: {TOPIC}")
works = fetch_openalex_works(TOPIC, n_results=300)
print(f"Retrieved {len(works)} works.")

edge_df = works_to_coauthorship_edgelist(works)
print(f"Edge list: {len(edge_df)} co-authorship pairs")

# --- Build network -----------------------------------------------------------
G_coauth = build_network_from_edgelist(
    edge_df,
    source_col="source",
    target_col="target",
    weight_col="weight",
    directed=False,
)

# Remove isolated nodes (authors who only appear in multi-author papers once)
G_coauth.remove_nodes_from(list(nx.isolates(G_coauth)))
print(f"After removing isolates: {G_coauth.number_of_nodes()} nodes")

# --- Centrality analysis -----------------------------------------------------
centrality_df = compute_all_centralities(G_coauth, k_betweenness=200)
print("\nTop 10 authors by betweenness centrality:")
print(centrality_df.nlargest(10, "betweenness")[["node", "degree", "betweenness", "pagerank"]])

# Set node attributes for visualization
for _, row in centrality_df.iterrows():
    if row["node"] in G_coauth.nodes:
        G_coauth.nodes[row["node"]]["degree"] = row["degree"]

# --- Community detection -----------------------------------------------------
partition = detect_communities_louvain(G_coauth, resolution=1.0)
nx.set_node_attributes(G_coauth, partition, "community")

# Community size distribution
from collections import Counter
comm_sizes = Counter(partition.values())
print(f"\nCommunity sizes (top 5): {comm_sizes.most_common(5)}")

# --- Visualize ---------------------------------------------------------------
fig = visualize_network(
    G_coauth,
    node_color_attr="community",
    node_size_attr="degree",
    layout="spring",
    title=f"Co-Authorship Network: '{TOPIC}' (OpenAlex)",
    save_path="coauthorship_network.png",
)
plt.show()

# --- Export to Gephi ---------------------------------------------------------
export_to_gephi(G_coauth, output_prefix="coauthorship_network", format="gexf")

# --- Small-world test --------------------------------------------------------
metrics = compute_small_world_metrics(G_coauth)
print(f"\nSmall-world metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
```

---

## Example B: Twitter/X Follower Ego Network Analysis

This example builds an ego network from a manually prepared follower list CSV and computes
structural properties: triadic closure, clustering, and centrality of the ego node.

```python
# ── Example B ─────────────────────────────────────────────────────────────
# Input: CSV with columns user_id, follower_id (followers of your ego node)
# Plus a second CSV: follower_follower_edges.csv — edges BETWEEN followers

import os

# --- Load data (replace paths with actual file locations) -------------------
EGO_ID = "ego_user_123"
FOLLOWERS_CSV = os.environ.get("FOLLOWERS_CSV", "followers.csv")
FF_EDGES_CSV = os.environ.get("FOLLOWER_FOLLOWER_CSV", "follower_follower_edges.csv")

followers_df = pd.read_csv(FOLLOWERS_CSV)          # columns: user_id, follower_id
ff_edges_df = pd.read_csv(FF_EDGES_CSV)            # columns: source, target

# Build ego network: add ego→follower edges + follower↔follower edges
ego_edges = pd.DataFrame({
    "source": EGO_ID,
    "target": followers_df["follower_id"],
    "weight": 1.0,
})
all_edges = pd.concat([ego_edges, ff_edges_df.assign(weight=1.0)], ignore_index=True)

G_ego = build_network_from_edgelist(
    all_edges,
    source_col="source",
    target_col="target",
    weight_col="weight",
    directed=True,
)

# --- Ego-specific metrics ----------------------------------------------------
# Alters = direct neighbors of ego
alters = list(G_ego.successors(EGO_ID)) + list(G_ego.predecessors(EGO_ID))
alters = list(set(alters))
G_alter = G_ego.subgraph(alters).copy()  # subgraph of alters only

print(f"Ego: {EGO_ID}")
print(f"Alters (direct neighbors): {len(alters)}")
print(f"Edges among alters: {G_alter.number_of_edges()}")

# Density of alter subgraph
n_alters = len(alters)
max_possible = n_alters * (n_alters - 1)
alter_density = G_alter.number_of_edges() / max_possible if max_possible > 0 else 0
print(f"Alter subgraph density: {alter_density:.4f}")

# Effective size (structural holes measure)
redundancy = sum(
    G_alter.degree(j) / n_alters
    for j in alters
    if G_alter.degree(j) > 0
)
effective_size = n_alters - redundancy
print(f"Effective network size (Burt): {effective_size:.2f}")

# --- Centrality of ego in its full network -----------------------------------
centrality_df = compute_all_centralities(G_ego, k_betweenness=300)
ego_row = centrality_df[centrality_df["node"] == EGO_ID]
print(f"\nEgo centrality profile:\n{ego_row.to_string(index=False)}")

# --- Community structure among alters ----------------------------------------
G_alter_undirected = G_alter.to_undirected()
if G_alter_undirected.number_of_edges() > 0:
    partition_alter = detect_communities_louvain(G_alter_undirected)
    nx.set_node_attributes(G_alter_undirected, partition_alter, "community")

    fig = visualize_network(
        G_alter_undirected,
        node_color_attr="community",
        node_size_attr=None,
        layout="spring",
        title=f"Ego Network Alters: {EGO_ID}",
        save_path="ego_network_alters.png",
    )
    plt.show()

# --- Triadic closure: fraction of open triads that are closed ----------------
transitivity = nx.transitivity(G_alter_undirected)
avg_clustering = nx.average_clustering(G_alter_undirected)
print(f"\nTriadic closure (transitivity): {transitivity:.4f}")
print(f"Average clustering coefficient: {avg_clustering:.4f}")

# --- Export ------------------------------------------------------------------
export_to_gephi(G_ego, output_prefix=f"ego_network_{EGO_ID}", format="graphml")
```

---

## Bipartite Networks

Bipartite networks connect two disjoint node sets (e.g., users and movies, authors and papers).
Use NetworkX's bipartite module to project onto one mode.

```python
from networkx.algorithms import bipartite

def build_bipartite_from_membership(
    membership_df: pd.DataFrame,
    actor_col: str = "actor",
    group_col: str = "group",
) -> tuple[nx.Graph, set, set]:
    """
    Build a bipartite graph from actor–group membership data.

    Returns the bipartite graph plus node sets for each layer.
    """
    B = nx.Graph()
    actors = set(membership_df[actor_col])
    groups = set(membership_df[group_col])
    B.add_nodes_from(actors, bipartite=0)
    B.add_nodes_from(groups, bipartite=1)
    B.add_edges_from(zip(membership_df[actor_col], membership_df[group_col]))
    return B, actors, groups


def project_bipartite(
    B: nx.Graph,
    nodes: set,
    weighted: bool = True,
) -> nx.Graph:
    """
    Project a bipartite graph onto one node set.

    Parameters
    ----------
    B : nx.Graph
        Bipartite graph.
    nodes : set
        The node set to project onto.
    weighted : bool
        If True, edge weight = number of shared neighbors in the other layer.

    Returns
    -------
    nx.Graph
        Projected unipartite graph.
    """
    if weighted:
        return bipartite.weighted_projected_graph(B, nodes)
    return bipartite.projected_graph(B, nodes)


# Usage example:
# membership_df = pd.DataFrame({"actor": ["A","A","B","C"], "group": ["G1","G2","G1","G2"]})
# B, actors, groups = build_bipartite_from_membership(membership_df)
# G_actors = project_bipartite(B, actors)
# print(nx.info(G_actors))
```

---

## Notes and Best Practices

### Performance on Large Graphs

| Graph Size | Recommended Approach |
|---|---|
| < 10 K nodes | All exact centralities safe |
| 10 K – 100 K nodes | Use `k_betweenness=500` approximation |
| > 100 K nodes | Use `nx.pagerank` only; consider graph-tool or igraph |

### Community Detection Comparison

- **Louvain**: Fast, scales to millions of nodes, non-deterministic (use `random_state`).
- **Girvan-Newman**: Slow (O(m² n)), but hierarchical; good for small networks (<1 K nodes).
- For directed networks, use `partition = community_louvain.best_partition(G.to_undirected())`.

### References

- Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford University Press.
- Blondel, V. D. et al. (2008). Fast unfolding of communities in large networks.
  *Journal of Statistical Mechanics*, P10008.
- Burt, R. S. (2004). Structural holes and good ideas. *American Journal of Sociology*, 110(2).
