---
name: network-analysis-math
description: >
  Use this Skill for graph construction, centrality measures (betweenness,
  PageRank), community detection (Louvain), and random graph models via networkx.
tags:
  - mathematics
  - network-analysis
  - graph-theory
  - networkx
  - community-detection
version: "1.0.0"
authors:
  - name: Rosetta Skills Contributors
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
    - python-igraph>=0.10
    - leidenalg>=0.10
    - matplotlib>=3.7
    - numpy>=1.24
    - pandas>=2.0
    - scipy>=1.11
last_updated: "2026-03-17"
status: "stable"
---

# Network Analysis — Graph Theory & Community Detection

> **One-line summary**: Construct and analyze complex networks with networkx and igraph: centrality, shortest paths, community detection (Louvain/Leiden), and random graph models.

---

## When to Use This Skill

- When analyzing social, biological, or infrastructure networks
- When computing centrality measures to identify important nodes
- When detecting communities or clusters in graphs
- When fitting random graph models (Erdős-Rényi, Barabási-Albert, Watts-Strogatz)
- When studying network robustness, percolation, or spreading processes
- When visualizing graph structure for publication figures

**Trigger keywords**: graph, network, centrality, betweenness, PageRank, community detection, Louvain, Leiden, Barabási-Albert, Watts-Strogatz, adjacency matrix

---

## Background & Key Concepts

### Centrality Measures

| Measure | Definition | Captures |
|:--------|:-----------|:---------|
| Degree | $k_i = \sum_j A_{ij}$ | Local connectivity |
| Betweenness | Fraction of shortest paths through node $i$ | Bridges/bottlenecks |
| Closeness | $1 / \text{avg. distance to others}$ | Proximity to all nodes |
| PageRank | Recursive importance from neighbors | Global influence |
| Katz | $\sum_{l=1}^{\infty} \alpha^l (A^l)_{ij}$ | Damped walk count |

### Modularity Optimization

Community detection maximizes modularity:

$$
Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)
$$

where $m$ is the number of edges, $k_i$ the degree of node $i$, and $\delta(c_i, c_j) = 1$ if nodes are in the same community.

### Random Graph Models

| Model | Parameters | Properties |
|:------|:-----------|:-----------|
| Erdős-Rényi G(n,p) | n nodes, edge prob p | Random, low clustering |
| Barabási-Albert | n nodes, attachment m | Scale-free, power-law degree |
| Watts-Strogatz | n, k, rewiring p | Small-world, high clustering |
| Stochastic Block Model | B (block matrix) | Community structure |

---

## Environment Setup

### Install Dependencies

```bash
pip install networkx>=3.1 python-igraph>=0.10 leidenalg>=0.10 \
            matplotlib>=3.7 numpy>=1.24 pandas>=2.0 scipy>=1.11
```

### Verify Installation

```python
import networkx as nx
import igraph as ig
import leidenalg

G = nx.karate_club_graph()
print(f"networkx {nx.__version__}: Karate club n={G.number_of_nodes()} e={G.number_of_edges()}")
print(f"igraph {ig.__version__}")
print(f"leidenalg {leidenalg.__version__}")
```

---

## Core Workflow

### Step 1: Graph Construction and Basic Analysis

```python
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Option 1: Build from edge list ---
edges = [
    (0, 1), (0, 2), (1, 2), (1, 3),
    (3, 4), (3, 5), (4, 5), (5, 6),
]
G = nx.Graph(edges)

# --- Option 2: From adjacency matrix ---
A = np.array([[0,1,1,0], [1,0,1,1], [1,1,0,0], [0,1,0,0]])
G2 = nx.from_numpy_array(A)

# --- Option 3: From pandas edge list ---
edge_df = pd.DataFrame({"source": [0,0,1,3,3], "target": [1,2,2,4,5], "weight": [1,2,3,1,2]})
G3 = nx.from_pandas_edgelist(edge_df, edge_attr="weight")

# Basic statistics
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Is connected: {nx.is_connected(G)}")
if nx.is_connected(G):
    print(f"Diameter: {nx.diameter(G)}")
    print(f"Avg shortest path: {nx.average_shortest_path_length(G):.4f}")
print(f"Avg clustering coefficient: {nx.average_clustering(G):.4f}")

# Degree distribution
degrees = dict(G.degree())
degree_vals = list(degrees.values())
print(f"\nDegree stats: mean={np.mean(degree_vals):.2f}, max={max(degree_vals)}")
```

### Step 2: Centrality Analysis

```python
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Use a real-world-like network
G = nx.barabasi_albert_graph(100, 3, seed=42)

# Compute all centrality measures
centralities = {
    "degree":      nx.degree_centrality(G),
    "betweenness": nx.betweenness_centrality(G, normalized=True),
    "closeness":   nx.closeness_centrality(G),
    "pagerank":    nx.pagerank(G, alpha=0.85),
    "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
}

# Assemble into DataFrame
cent_df = pd.DataFrame(centralities)
cent_df.index.name = "node"
cent_df.reset_index(inplace=True)

print("Top 10 nodes by PageRank:")
print(cent_df.nlargest(10, "pagerank")[["node", "degree", "betweenness", "pagerank"]].to_string(index=False))

# Correlation between centralities
print("\nCentrality correlations:")
print(cent_df[["degree", "betweenness", "closeness", "pagerank"]].corr().round(3))

# Visualize network with PageRank as node size
fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
node_size = [cent_df.loc[cent_df["node"]==n, "pagerank"].values[0] * 5000 + 50
             for n in G.nodes()]
nx.draw_networkx(G, pos, node_size=node_size, node_color="steelblue",
                 edge_color="gray", alpha=0.7, with_labels=False, ax=ax)
ax.set_title("Network with PageRank node sizing")
ax.axis("off")
plt.tight_layout()
plt.savefig("network_centrality.png", dpi=150)
plt.show()
```

### Step 3: Community Detection with Leiden Algorithm

```python
import igraph as ig
import leidenalg
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Convert networkx → igraph
G_nx = nx.karate_club_graph()
G_ig = ig.Graph.from_networkx(G_nx)

# Leiden community detection (resolution parameter controls granularity)
partition = leidenalg.find_partition(
    G_ig,
    leidenalg.ModularityVertexPartition,
    seed=42,
)

communities = list(partition)
n_communities = len(communities)
modularity = G_ig.modularity(partition.membership)

print(f"Communities found: {n_communities}")
print(f"Modularity Q = {modularity:.4f}")
print(f"Community sizes: {sorted([len(c) for c in communities], reverse=True)}")

# Map back to networkx for visualization
node_colors = {}
for comm_id, members in enumerate(communities):
    for node in members:
        node_colors[node] = comm_id

# Visualize
pos = nx.spring_layout(G_nx, seed=42)
colors = [node_colors[n] for n in G_nx.nodes()]
fig, ax = plt.subplots(figsize=(8, 6))
nx.draw_networkx(G_nx, pos, node_color=colors, cmap=plt.cm.tab20,
                 node_size=200, edge_color="gray", with_labels=True,
                 font_size=8, ax=ax)
ax.set_title(f"Karate Club: {n_communities} communities (Q={modularity:.3f})")
ax.axis("off")
plt.tight_layout()
plt.savefig("community_detection.png", dpi=150)
plt.show()

# Compare with ground truth (club membership)
ground_truth = [G_nx.nodes[n]["club"] for n in G_nx.nodes()]
from sklearn.metrics import adjusted_rand_score
pred_labels = [node_colors[n] for n in G_nx.nodes()]
ari = adjusted_rand_score(ground_truth, pred_labels)
print(f"\nAdjusted Rand Index vs. ground truth: {ari:.4f}")
```

---

## Advanced Usage

### Random Graph Model Fitting

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def fit_random_graph_model(G):
    """Fit Erdős-Rényi, BA, and WS models and compare degree distributions."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = 2 * m / (n * (n - 1))  # ER probability
    k_avg = 2 * m / n
    k = max(1, int(round(k_avg / 2)))  # BA attachment parameter

    G_er = nx.erdos_renyi_graph(n, p, seed=42)
    G_ba = nx.barabasi_albert_graph(n, k, seed=42)
    G_ws = nx.watts_strogatz_graph(n, min(n-1, 4*k), 0.1, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, G_model, name in zip(axes, [G_er, G_ba, G_ws],
                                  ["Erdős-Rényi", "Barabási-Albert", "Watts-Strogatz"]):
        orig_deg = [d for _, d in G.degree()]
        model_deg = [d for _, d in G_model.degree()]
        ax.hist(orig_deg, bins=20, alpha=0.6, density=True, label="Original")
        ax.hist(model_deg, bins=20, alpha=0.6, density=True, label=name)
        ax.set_xlabel("Degree"); ax.set_ylabel("Probability"); ax.legend()
        c = nx.average_clustering(G_model)
        l = nx.average_shortest_path_length(G_model) if nx.is_connected(G_model) else float('inf')
        ax.set_title(f"{name}\nC={c:.3f}, L={l:.2f}")

    plt.tight_layout()
    plt.savefig("random_graph_comparison.png", dpi=150)
    plt.show()

G = nx.karate_club_graph()
fit_random_graph_model(G)
```

### Network Robustness Analysis

```python
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def robustness_analysis(G, n_steps=50):
    """Simulate targeted (degree-based) vs. random node removal."""
    G_random = G.copy()
    G_targeted = G.copy()
    sizes_random, sizes_targeted = [1.0], [1.0]
    n_total = G.number_of_nodes()

    nodes_random = list(G_random.nodes())
    np.random.shuffle(nodes_random)

    for _ in range(n_steps):
        # Random removal
        if G_random.number_of_nodes() > 0:
            G_random.remove_node(nodes_random.pop())
            if G_random.number_of_nodes() > 0 and nx.is_connected(G_random):
                sizes_random.append(G_random.number_of_nodes() / n_total)
            else:
                comps = list(nx.connected_components(G_random))
                sizes_random.append(max(len(c) for c in comps) / n_total if comps else 0)

        # Targeted removal (highest degree node)
        if G_targeted.number_of_nodes() > 0:
            max_deg_node = max(G_targeted.degree(), key=lambda x: x[1])[0]
            G_targeted.remove_node(max_deg_node)
            comps = list(nx.connected_components(G_targeted))
            sizes_targeted.append(max(len(c) for c in comps) / n_total if comps else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.linspace(0, 1, len(sizes_random)), sizes_random, 'b-o', ms=3, label="Random removal")
    ax.plot(np.linspace(0, 1, len(sizes_targeted)), sizes_targeted, 'r-s', ms=3, label="Targeted removal")
    ax.set_xlabel("Fraction of nodes removed")
    ax.set_ylabel("Largest component fraction")
    ax.legend(); ax.set_title("Network Robustness Analysis")
    plt.tight_layout()
    plt.savefig("robustness_analysis.png", dpi=150)
    plt.show()

G = nx.barabasi_albert_graph(200, 3, seed=42)
robustness_analysis(G)
```

---

## Troubleshooting

### Error: `networkx.exception.NetworkXError: Graph is not connected`

**Cause**: Some metrics (diameter, avg path length) require connected graphs.

**Fix**:
```python
if not nx.is_connected(G):
    # Work with largest connected component
    Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    print(f"Using LCC: {Gcc.number_of_nodes()} / {G.number_of_nodes()} nodes")
```

### Issue: Leiden algorithm returns single large community

**Cause**: Resolution parameter too low.

**Fix**:
```python
# Increase resolution for more, smaller communities
partition = leidenalg.find_partition(
    G_ig, leidenalg.CPMVertexPartition,
    resolution_parameter=0.05   # increase from 0 to get more communities
)
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| networkx | 3.1, 3.2, 3.3   | API mostly stable |
| igraph   | 0.10, 0.11      | from_networkx requires igraph>=0.10 |
| leidenalg | 0.10           | requires python-igraph |

---

## External Resources

### Official Documentation

- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [igraph Python documentation](https://python.igraph.org/en/stable/)
- [Leidenalg documentation](https://leidenalg.readthedocs.io/)

### Key Papers

- Blondel, V.D. et al. (2008). *Fast unfolding of communities in large networks*. J. Statistical Mechanics.
- Traag, V.A. et al. (2019). *From Louvain to Leiden: guaranteeing well-connected communities*. Scientific Reports.

---

## Examples

### Example 1: Co-authorship Network Analysis

```python
# =============================================
# Synthetic co-authorship network analysis
# =============================================
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Simulate: 50 researchers, 5 research groups
n_researchers = 50
n_groups = 5
group_size = n_researchers // n_groups

G = nx.Graph()
G.add_nodes_from(range(n_researchers))

# Within-group: high connection probability
for g in range(n_groups):
    members = range(g*group_size, (g+1)*group_size)
    for i in members:
        for j in members:
            if i < j and rng.random() < 0.6:
                G.add_edge(i, j, weight=rng.integers(1, 5))

# Between groups: low connection probability
for i in range(n_researchers):
    for j in range(i+1, n_researchers):
        if not G.has_edge(i,j) and rng.random() < 0.03:
            G.add_edge(i, j, weight=1)

print(f"Network: n={G.number_of_nodes()}, m={G.number_of_edges()}")
print(f"Density: {nx.density(G):.4f}")
print(f"Avg clustering: {nx.average_clustering(G):.4f}")

# Community detection
import igraph as ig, leidenalg
G_ig = ig.Graph.from_networkx(G)
partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition, seed=42)
print(f"Communities detected: {len(partition)} (true: {n_groups})")
print(f"Modularity: {G_ig.modularity(partition.membership):.4f}")

# Hub identification
pagerank = nx.pagerank(G, alpha=0.85)
top5 = sorted(pagerank, key=pagerank.get, reverse=True)[:5]
print(f"\nTop 5 influential researchers (PageRank): {top5}")
```

**Interpreting these results**: Leiden detects the 5 research groups with high modularity. Top PageRank nodes are key collaborators bridging groups.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
