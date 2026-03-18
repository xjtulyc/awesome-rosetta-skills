---
name: historical-network
description: >
  Use this Skill for historical network analysis: correspondence networks,
  prosopographic data, temporal community detection, betweenness over time, and Gephi GEXF export.
tags:
  - history
  - network-analysis
  - prosopography
  - temporal-networks
  - Gephi
  - digital-humanities
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
    - networkx>=3.1
    - pandas>=1.5
    - numpy>=1.23
    - matplotlib>=3.6
    - python-louvain>=0.16
last_updated: "2026-03-18"
status: stable
---

# Historical Network Analysis: Prosopography and Correspondence Networks

> **TL;DR** — Build correspondence networks from archival letter datasets, attach
> prosopographic node attributes, compute temporal betweenness centrality by decade,
> detect communities per period with Louvain, measure community persistence via Jaccard,
> project bipartite person-event networks, and export to Gephi GEXF for visualization.

---

## When to Use

Use this Skill when you need to:

- Analyze the structure of historical correspondence networks (epistolary, diplomatic, merchant)
- Study the rise and fall of influential intermediaries over decades or centuries
- Detect scholarly, religious, or political communities across time periods
- Visualize network evolution with node/edge attributes in Gephi
- Project person × event membership matrices into person-person co-attendance networks

Do **not** use this Skill for:

- Online social media graph analysis at millions-of-nodes scale (use GraphX or Snap.py)
- Phylogenetic tree reconstruction (use dedicated bioinformatics tools)
- Road or transport network optimization (use OR-Tools or igraph)

---

## Background

Historical networks differ from modern social networks in two key ways:

1. **Incompleteness**: Only a fraction of historical correspondence survives. Network
   metrics must be interpreted with an eye on archival gaps.
2. **Temporal dynamics**: Polities, alliances, and scholarly circles shift over decades.
   Static network metrics hide this change; time-windowed analysis is essential.

| Concept | Explanation |
|---|---|
| Prosopography | Systematic study of individuals in historical records; provides node attributes |
| Betweenness centrality | Fraction of shortest paths passing through a node; high = broker/gatekeeper |
| Louvain community detection | Modularity-maximizing algorithm; O(n log n) |
| Jaccard similarity | |A ∩ B| / |A ∪ B|; measures community persistence across periods |
| GEXF | Graph Exchange XML Format; native Gephi format supporting dynamic attributes |
| Bipartite projection | Person × event bipartite → person-person weighted unipartite graph |

---

## Environment Setup

```bash
# Create Python environment
conda create -n hist-net python=3.11 -y
conda activate hist-net

# Install dependencies
pip install "networkx>=3.1" "pandas>=1.5" "numpy>=1.23" \
    "matplotlib>=3.6" python-louvain

# Verify
python -c "import networkx as nx; print(nx.__version__)"
python -c "import community; print('louvain ok')"
```

---

## Core Workflow

### Step 1 — Letter Network from CSV with Temporal Betweenness

```python
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def build_correspondence_network(
    letters_df: pd.DataFrame,
    sender_col: str = "sender_id",
    recipient_col: str = "recipient_id",
    date_col: str = "year",
    weight_threshold: int = 1,
) -> nx.MultiDiGraph:
    """
    Construct a directed multigraph from a historical correspondence dataset.

    Each letter becomes an edge with year and doc_id attributes.
    Parallel edges are preserved (one edge per letter).

    Args:
        letters_df:       DataFrame of letters with sender, recipient, year, doc_id columns.
        sender_col:       Column name for sender identifier.
        recipient_col:    Column name for recipient identifier.
        date_col:         Column name for the year integer.
        weight_threshold: Minimum number of letters to include an edge in the graph.

    Returns:
        Directed multigraph with letter-level edges.
    """
    G = nx.MultiDiGraph()

    # Add all unique persons as nodes
    all_persons = pd.concat([
        letters_df[sender_col],
        letters_df[recipient_col],
    ]).unique()
    G.add_nodes_from(all_persons)

    # Add edges
    for _, row in letters_df.iterrows():
        G.add_edge(
            row[sender_col],
            row[recipient_col],
            year=int(row[date_col]),
            doc_id=str(row.get("doc_id", "")),
        )

    # Optionally collapse to weighted DiGraph
    if weight_threshold > 1:
        simple = nx.DiGraph()
        edge_weights = defaultdict(int)
        for u, v, data in G.edges(data=True):
            edge_weights[(u, v)] += 1
        for (u, v), w in edge_weights.items():
            if w >= weight_threshold:
                simple.add_edge(u, v, weight=w)
        return simple

    return G


def add_prosopographic_attributes(
    G: nx.Graph,
    prosop_df: pd.DataFrame,
    id_col: str = "person_id",
) -> nx.Graph:
    """
    Add prosopographic metadata as node attributes to a network.

    Args:
        G:          NetworkX graph with person IDs as nodes.
        prosop_df:  DataFrame with columns: person_id, name, title, birth_year,
                    death_year, gender, institution.
        id_col:     Column name for the person identifier.

    Returns:
        Graph with updated node attributes (modifies in place).
    """
    attr_cols = [c for c in prosop_df.columns if c != id_col]
    for _, row in prosop_df.iterrows():
        pid = row[id_col]
        if pid in G.nodes:
            for col in attr_cols:
                G.nodes[pid][col] = row[col]
    return G


def temporal_betweenness(
    letters_df: pd.DataFrame,
    decade_size: int = 10,
    sender_col: str = "sender_id",
    recipient_col: str = "recipient_id",
    date_col: str = "year",
) -> pd.DataFrame:
    """
    Compute betweenness centrality per decade window.

    For each decade, build an undirected weighted graph from letters
    in that window and compute normalized betweenness centrality.

    Args:
        letters_df:  DataFrame of letters.
        decade_size: Width of time window in years.
        sender_col:  Column for sender IDs.
        recipient_col: Column for recipient IDs.
        date_col:    Column for year.

    Returns:
        DataFrame with columns: person_id, decade_start, betweenness.
    """
    min_year = int(letters_df[date_col].min())
    max_year = int(letters_df[date_col].max())
    decades = range(min_year, max_year, decade_size)

    records = []
    for decade_start in decades:
        decade_end = decade_start + decade_size
        window = letters_df[
            (letters_df[date_col] >= decade_start) &
            (letters_df[date_col] < decade_end)
        ]
        if window.empty:
            continue

        # Build weighted undirected graph for this decade
        G_dec = nx.Graph()
        for _, row in window.iterrows():
            u, v = row[sender_col], row[recipient_col]
            if G_dec.has_edge(u, v):
                G_dec[u][v]["weight"] += 1
            else:
                G_dec.add_edge(u, v, weight=1)

        if G_dec.number_of_nodes() < 3:
            continue

        bc = nx.betweenness_centrality(G_dec, weight="weight", normalized=True)
        for person_id, score in bc.items():
            records.append({
                "person_id": person_id,
                "decade_start": decade_start,
                "betweenness": round(score, 6),
            })

    return pd.DataFrame(records)


def plot_betweenness_over_time(
    betweenness_df: pd.DataFrame,
    top_n: int = 5,
    prosop_df: pd.DataFrame = None,
    output_path: str = None,
) -> None:
    """
    Line plot of top-N persons by peak betweenness centrality over decades.

    Args:
        betweenness_df: DataFrame from temporal_betweenness().
        top_n:          Number of top persons to highlight.
        prosop_df:      Optional prosopographic DataFrame; if given, use 'name' column.
        output_path:    If given, save figure here.
    """
    peak_bc = (
        betweenness_df.groupby("person_id")["betweenness"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_persons = peak_bc.index.tolist()

    fig, ax = plt.subplots(figsize=(11, 5))
    for pid in top_persons:
        sub = betweenness_df[betweenness_df["person_id"] == pid].sort_values("decade_start")
        label = pid
        if prosop_df is not None and "name" in prosop_df.columns:
            name_row = prosop_df[prosop_df.iloc[:, 0] == pid]
            if not name_row.empty:
                label = name_row.iloc[0]["name"]
        ax.plot(sub["decade_start"], sub["betweenness"], marker="o", label=label)

    ax.set_xlabel("Decade start")
    ax.set_ylabel("Betweenness centrality (normalized)")
    ax.set_title(f"Top {top_n} Brokers by Betweenness Centrality Over Time")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Betweenness plot saved to {output_path}")
    plt.show()
```

### Step 2 — Decade-by-Decade Community Detection and Persistence

```python
import community as community_louvain


def detect_communities_per_period(
    letters_df: pd.DataFrame,
    decade_size: int = 10,
    sender_col: str = "sender_id",
    recipient_col: str = "recipient_id",
    date_col: str = "year",
    resolution: float = 1.0,
) -> dict:
    """
    Detect Louvain communities for each decade window.

    Args:
        letters_df:  DataFrame of letters.
        decade_size: Width of time window in years.
        sender_col:  Column for sender IDs.
        recipient_col: Column for recipient IDs.
        date_col:    Column for year integer.
        resolution:  Louvain resolution parameter (>1 = more smaller communities).

    Returns:
        Dict mapping decade_start (int) → community_partition (dict: node → community_id).
    """
    min_year = int(letters_df[date_col].min())
    max_year = int(letters_df[date_col].max())

    period_communities = {}
    for decade_start in range(min_year, max_year, decade_size):
        decade_end = decade_start + decade_size
        window = letters_df[
            (letters_df[date_col] >= decade_start) &
            (letters_df[date_col] < decade_end)
        ]
        if window.empty:
            continue

        G_dec = nx.Graph()
        for _, row in window.iterrows():
            u, v = row[sender_col], row[recipient_col]
            if G_dec.has_edge(u, v):
                G_dec[u][v]["weight"] += 1
            else:
                G_dec.add_edge(u, v, weight=1)

        if G_dec.number_of_nodes() < 4:
            continue

        partition = community_louvain.best_partition(
            G_dec, weight="weight", resolution=resolution, random_state=42
        )
        period_communities[decade_start] = partition

    return period_communities


def compute_community_jaccard_persistence(
    period_communities: dict,
) -> pd.DataFrame:
    """
    Measure the persistence of communities across consecutive periods using Jaccard similarity.

    For each pair of consecutive periods, compute the maximum Jaccard similarity
    between every community in period T and every community in period T+1.
    High Jaccard = stable community; low Jaccard = community dissolved.

    Args:
        period_communities: Dict from detect_communities_per_period().

    Returns:
        DataFrame with columns: period_a, period_b, community_a, community_b,
        jaccard, matched_count.
    """
    periods = sorted(period_communities.keys())
    records = []

    for i in range(len(periods) - 1):
        pa = periods[i]
        pb = periods[i + 1]

        partition_a = period_communities[pa]
        partition_b = period_communities[pb]

        # Group nodes by community
        def group_by_comm(part):
            groups = defaultdict(set)
            for node, comm in part.items():
                groups[comm].add(node)
            return groups

        groups_a = group_by_comm(partition_a)
        groups_b = group_by_comm(partition_b)

        for comm_a, nodes_a in groups_a.items():
            best_jacc = 0.0
            best_comm_b = None
            for comm_b, nodes_b in groups_b.items():
                intersection = len(nodes_a & nodes_b)
                union = len(nodes_a | nodes_b)
                jacc = intersection / union if union > 0 else 0.0
                if jacc > best_jacc:
                    best_jacc = jacc
                    best_comm_b = comm_b

            records.append({
                "period_a": pa,
                "period_b": pb,
                "community_a": comm_a,
                "community_b": best_comm_b,
                "jaccard": round(best_jacc, 4),
                "size_a": len(nodes_a),
            })

    return pd.DataFrame(records)
```

### Step 3 — GEXF Export for Gephi Visualization

```python
def build_annotated_network_for_export(
    letters_df: pd.DataFrame,
    prosop_df: pd.DataFrame = None,
    period_communities: dict = None,
    sender_col: str = "sender_id",
    recipient_col: str = "recipient_id",
    date_col: str = "year",
) -> nx.Graph:
    """
    Build a weighted undirected graph with full node/edge attributes for Gephi export.

    Node attributes: name, gender, institution, community_id (from last period).
    Edge attributes: weight (letter count), first_year, last_year.

    Args:
        letters_df:        DataFrame of letters.
        prosop_df:         Optional prosopographic DataFrame.
        period_communities: Optional dict from detect_communities_per_period().
        sender_col:        Column for sender IDs.
        recipient_col:     Column for recipient IDs.
        date_col:          Column for year.

    Returns:
        Annotated undirected NetworkX Graph ready for nx.write_gexf().
    """
    G = nx.Graph()

    # Build weighted edges
    edge_data = defaultdict(lambda: {"weight": 0, "years": []})
    for _, row in letters_df.iterrows():
        u, v = str(row[sender_col]), str(row[recipient_col])
        key = tuple(sorted([u, v]))
        edge_data[key]["weight"] += 1
        edge_data[key]["years"].append(int(row[date_col]))

    all_nodes = set()
    for (u, v), data in edge_data.items():
        all_nodes.update([u, v])
        G.add_edge(
            u, v,
            weight=data["weight"],
            first_year=min(data["years"]),
            last_year=max(data["years"]),
        )

    # Add prosopographic node attributes
    if prosop_df is not None:
        id_col = prosop_df.columns[0]
        for _, row in prosop_df.iterrows():
            pid = str(row[id_col])
            if pid in G.nodes:
                for col in prosop_df.columns:
                    if col != id_col:
                        val = row[col]
                        if pd.notna(val):
                            G.nodes[pid][col] = str(val)

    # Add community labels from the last period
    if period_communities:
        last_period = max(period_communities.keys())
        partition = period_communities[last_period]
        for node, comm_id in partition.items():
            node_str = str(node)
            if node_str in G.nodes:
                G.nodes[node_str]["community"] = int(comm_id)

    return G


def export_gexf(
    G: nx.Graph,
    output_path: str,
) -> None:
    """
    Export a NetworkX graph to GEXF format for Gephi visualization.

    GEXF supports node/edge attributes and can encode dynamic networks
    with time intervals. Open the output file directly in Gephi.

    Args:
        G:           Annotated NetworkX graph.
        output_path: Absolute path to write the .gexf file.
    """
    nx.write_gexf(G, output_path)
    print(f"GEXF file written to {output_path}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Open in Gephi: File → Open → {output_path}")
```

---

## Advanced Usage

### Bipartite Person-Event Projection

```python
def build_bipartite_person_event(
    attendance_df: pd.DataFrame,
    person_col: str = "person_id",
    event_col: str = "event_id",
) -> tuple[nx.Graph, nx.Graph]:
    """
    Build a bipartite person-event graph and project to person-person co-attendance.

    Edge weight in the projected graph = number of events co-attended.

    Args:
        attendance_df: DataFrame with person_id and event_id columns.
        person_col:    Column for person identifiers.
        event_col:     Column for event identifiers.

    Returns:
        Tuple of (bipartite_graph, person_person_projection).
    """
    from networkx.algorithms import bipartite

    B = nx.Graph()

    persons = attendance_df[person_col].unique()
    events = attendance_df[event_col].unique()

    B.add_nodes_from(persons, bipartite=0)
    B.add_nodes_from(events, bipartite=1)

    for _, row in attendance_df.iterrows():
        B.add_edge(row[person_col], row[event_col])

    # Project onto persons (bipartite=0 set)
    person_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}
    projected = bipartite.weighted_projected_graph(B, person_nodes)
    return B, projected
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: community` | python-louvain not installed | `pip install python-louvain` |
| Betweenness very slow on large graph | O(n*m) algorithm | Use `nx.betweenness_centrality(G, k=100)` for approximate BC |
| GEXF file won't open in Gephi | Non-string node IDs | Cast all node IDs to `str()` before `write_gexf()` |
| Louvain non-deterministic results | Random seed not set | Pass `random_state=42` to `best_partition()` |
| `NetworkXError: Graph has no edges` on thin decade | Very few letters in that window | Increase `decade_size` or skip windows with fewer than 5 edges |
| Community Jaccard all near 0 | Network turnover too high | Check for person ID inconsistencies in source data |

---

## External Resources

- NetworkX documentation: <https://networkx.org/documentation/stable/>
- python-louvain: <https://python-louvain.readthedocs.io/>
- Gephi graph visualization: <https://gephi.org/>
- GEXF format specification: <https://gexf.net/format/>
- Mapping the Republic of Letters (Stanford): <http://republicofletters.stanford.edu/>
- Early Modern Letters Online: <http://emlo.bodleian.ox.ac.uk/>

---

## Examples

### Example 1 — Republic of Letters Network Analysis

```python
import pandas as pd
import numpy as np

# Simulate a historical correspondence dataset (1600–1700)
np.random.seed(42)
n_letters = 500
persons = [f"P{i:03d}" for i in range(1, 51)]
years = np.random.randint(1600, 1700, n_letters)
senders = np.random.choice(persons, n_letters)
recipients = np.random.choice(persons, n_letters)

letters_df = pd.DataFrame({
    "sender_id": senders,
    "recipient_id": recipients,
    "year": years,
    "doc_id": [f"DOC_{i:04d}" for i in range(n_letters)],
})
# Remove self-loops
letters_df = letters_df[letters_df["sender_id"] != letters_df["recipient_id"]].reset_index(drop=True)

# Compute temporal betweenness
bc_df = temporal_betweenness(letters_df, decade_size=10)
print("Top brokers by peak betweenness:")
peak = bc_df.groupby("person_id")["betweenness"].max().sort_values(ascending=False).head(5)
print(peak)

# Plot betweenness over time
plot_betweenness_over_time(bc_df, top_n=5, output_path="/data/output/betweenness_timeline.png")
```

### Example 2 — Community Detection and GEXF Export

```python
# Detect communities per decade
communities = detect_communities_per_period(letters_df, decade_size=20)
print(f"Periods analyzed: {sorted(communities.keys())}")

# Community persistence
jaccard_df = compute_community_jaccard_persistence(communities)
stable = jaccard_df[jaccard_df["jaccard"] > 0.4]
print(f"\nStable community links (Jaccard > 0.4): {len(stable)}")
print(jaccard_df.sort_values("jaccard", ascending=False).head(10).to_string(index=False))

# Build full annotated graph and export for Gephi
G_full = build_annotated_network_for_export(
    letters_df, period_communities=communities
)
export_gexf(G_full, "/data/output/republic_of_letters.gexf")

print("\nOpen /data/output/republic_of_letters.gexf in Gephi.")
print("Use 'ForceAtlas 2' layout and color nodes by 'community' attribute.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — correspondence network, temporal betweenness, Louvain communities, Jaccard persistence, bipartite projection, GEXF export |
