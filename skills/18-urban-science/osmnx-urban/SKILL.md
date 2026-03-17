---
name: osmnx-urban
description: >
  Urban street network analysis with OSMnx: network statistics, centrality,
  isochrone/walkability analysis, POI extraction, and multi-city comparisons.
tags:
  - osmnx
  - urban-science
  - street-network
  - walkability
  - networkx
  - openstreetmap
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
  - osmnx>=1.9.0
  - networkx>=3.1.0
  - numpy>=1.24.0
  - pandas>=2.0.0
  - geopandas>=0.14.0
  - matplotlib>=3.7.0
  - shapely>=2.0.0
  - scipy>=1.11.0
last_updated: "2026-03-17"
---

# OSMnx Urban Street Network Analysis Skill

This skill covers the complete workflow for downloading, analysing, and visualising urban
street networks using OSMnx — a Python package that retrieves OpenStreetMap data and models
it as directed NetworkX graphs. The skill includes network statistics, centrality measures,
isochrone generation for walkability assessment, point-of-interest (POI) extraction, routing,
and systematic multi-city comparisons.

Urban morphology shapes how people move, access services, and experience cities. Street network
analysis provides quantitative metrics that underpin urban planning, transport modelling, and
public health research.

---

## Setup

```bash
pip install osmnx networkx numpy pandas geopandas matplotlib shapely scipy
# On conda:
# conda install -c conda-forge osmnx
```

OSMnx fetches data from the Overpass API (OpenStreetMap). No API key is required, but
heavy use should respect the Overpass rate limits by caching graphs locally.

---

## Core Functions

```python
"""
osmnx_urban.py
--------------
Core utilities for urban street network analysis with OSMnx and NetworkX.
"""

from __future__ import annotations

import warnings
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=FutureWarning)

# Configure OSMnx caching (avoids redundant API calls)
ox.settings.use_cache = True
ox.settings.cache_folder = "osmnx_cache"
ox.settings.log_console = False


# ---------------------------------------------------------------------------
# 1. Download Street Network
# ---------------------------------------------------------------------------

def get_city_network(
    place_name: str,
    network_type: str = "walk",
    simplify: bool = True,
    retain_all: bool = False,
    custom_filter: Optional[str] = None,
) -> nx.MultiDiGraph:
    """
    Download an OSM street network for a named place or area.

    Parameters
    ----------
    place_name : str
        Geocodable place name, e.g. 'Manhattan, New York, USA' or
        'Amsterdam, Netherlands'. OSMnx uses Nominatim for geocoding.
    network_type : str
        Type of network edges to include:
          - 'walk'   : pedestrian-accessible streets and paths
          - 'drive'  : driveable roads (no pedestrian-only ways)
          - 'bike'   : cyclable edges
          - 'all'    : all OSM ways (including private)
          - 'all_public': all publicly accessible ways
    simplify : bool
        If True (default), simplify the graph topology by removing nodes of
        degree 2 (intermediate nodes on straight roads). Keeps only true
        intersections and dead-ends.
    retain_all : bool
        If True, retain all subgraphs including disconnected components.
        Default False (keeps only the largest strongly connected component).
    custom_filter : str, optional
        Custom OSM filter string, e.g.
        '["highway"~"motorway|trunk|primary"]'.

    Returns
    -------
    nx.MultiDiGraph
        Projected (UTM) street network graph. Node attributes include
        'x', 'y' (UTM coords) and 'lat', 'lon' (WGS84). Edge attributes
        include 'length' (metres), 'highway', 'oneway', 'maxspeed', etc.

    Examples
    --------
    >>> G = get_city_network("Barcelona, Spain", network_type="walk")
    >>> G_drive = get_city_network("Seoul, South Korea", network_type="drive")
    """
    if custom_filter:
        G = ox.graph_from_place(
            place_name, retain_all=retain_all,
            simplify=simplify, custom_filter=custom_filter,
        )
    else:
        G = ox.graph_from_place(
            place_name, network_type=network_type,
            retain_all=retain_all, simplify=simplify,
        )
    G = ox.project_graph(G)
    return G


def get_network_from_point(
    lat: float,
    lon: float,
    dist: int = 1000,
    network_type: str = "walk",
) -> nx.MultiDiGraph:
    """
    Download a network within ``dist`` metres of a lat/lon point.

    Useful when a named place is ambiguous or too large.
    """
    G = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type)
    return ox.project_graph(G)


def save_network(G: nx.MultiDiGraph, path: Union[str, Path]) -> None:
    """Save a graph to a GraphML file for later reuse."""
    ox.save_graphml(G, str(path))
    print(f"Saved network to {path}")


def load_network(path: Union[str, Path]) -> nx.MultiDiGraph:
    """Load a previously saved GraphML network."""
    return ox.load_graphml(str(path))


# ---------------------------------------------------------------------------
# 2. Network Statistics
# ---------------------------------------------------------------------------

def compute_network_stats(
    G: nx.MultiDiGraph,
    extended: bool = True,
) -> Dict:
    """
    Compute a comprehensive set of network statistics.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Projected OSMnx street network.
    extended : bool
        If True, compute additional circuity and intersection density metrics.
        These are slower for large networks.

    Returns
    -------
    dict with keys (subset):
        n           - number of nodes (intersections + dead-ends)
        m           - number of edges (street segments)
        k_avg       - average node degree
        edge_length_total   - total street length (m)
        edge_length_avg     - mean edge length (m)
        street_density      - km of street per km² of convex hull
        intersection_density- intersections per km²
        circuity_avg        - ratio actual/straight-line distance (1.0 = perfect grid)
        self_loop_proportion- fraction of edges that are self-loops
        clean_intersection_count - count of true intersections (degree >= 3)

    Examples
    --------
    >>> G = get_city_network("Vienna, Austria")
    >>> stats = compute_network_stats(G)
    >>> print(f"Avg circuity: {stats['circuity_avg']:.3f}")
    """
    basic = ox.basic_stats(G)
    stats = {
        "n": basic["n"],
        "m": basic["m"],
        "k_avg": basic["k_avg"],
        "edge_length_total_km": basic["edge_length_total"] / 1000,
        "edge_length_avg_m": basic["edge_length_avg"],
        "streets_per_node_avg": basic["streets_per_node_avg"],
        "intersection_count": basic.get("intersection_count", np.nan),
    }

    if extended:
        area_km2 = ox.project_graph(G)  # already projected
        # Compute convex hull area from node positions
        nodes_gdf, _ = ox.graph_to_gdfs(G)
        hull = nodes_gdf.unary_union.convex_hull
        area_km2 = hull.area / 1e6

        stats["area_km2"] = round(area_km2, 4)
        if area_km2 > 0:
            stats["street_density_km_per_km2"] = round(
                stats["edge_length_total_km"] / area_km2, 2
            )
            stats["node_density_per_km2"] = round(stats["n"] / area_km2, 1)

        # Circuity
        try:
            stats["circuity_avg"] = round(
                ox.circuity_avg(G), 4
            )
        except Exception:
            stats["circuity_avg"] = np.nan

        # Self-loops
        self_loops = sum(1 for u, v, _ in G.edges if u == v)
        stats["self_loop_proportion"] = round(self_loops / max(G.number_of_edges(), 1), 4)

    return stats


# ---------------------------------------------------------------------------
# 3. Centrality Analysis
# ---------------------------------------------------------------------------

def compute_centrality(
    G: nx.MultiDiGraph,
    measures: List[str] = None,
    weight: str = "length",
) -> pd.DataFrame:
    """
    Compute node-level centrality measures on the street network.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Projected network. For large cities, use a subgraph or coarsen first.
    measures : list of str, optional
        Centrality measures to compute. Defaults to
        ['betweenness', 'closeness', 'pagerank', 'degree'].
    weight : str
        Edge attribute to use as weight ('length' for distance-weighted).

    Returns
    -------
    pd.DataFrame
        One row per node (indexed by OSM node ID), columns for each measure.
        Merged with node lat/lon for spatial plotting.

    Notes
    -----
    Betweenness centrality is O(n * m) and can be slow for networks with
    > 10,000 nodes. Use ``nx.betweenness_centrality`` with ``k`` argument
    for approximation on large networks.
    """
    if measures is None:
        measures = ["betweenness", "closeness", "pagerank", "degree"]

    # Convert to undirected for symmetric centrality
    G_undirected = G.to_undirected()
    nodes_gdf, _ = ox.graph_to_gdfs(G)

    results = {"osmid": list(G.nodes())}

    if "degree" in measures:
        deg = dict(G.degree(weight=weight))
        results["degree"] = [deg.get(n, 0) for n in G.nodes()]

    if "betweenness" in measures:
        n_nodes = G.number_of_nodes()
        k_approx = min(n_nodes, 500) if n_nodes > 1000 else None
        bc = nx.betweenness_centrality(
            G_undirected, weight=weight, normalized=True, k=k_approx, seed=42
        )
        results["betweenness"] = [bc.get(n, 0) for n in G.nodes()]

    if "closeness" in measures:
        cc = nx.closeness_centrality(G_undirected, distance=weight)
        results["closeness"] = [cc.get(n, 0) for n in G.nodes()]

    if "pagerank" in measures:
        pr = nx.pagerank(G, weight=weight, alpha=0.85)
        results["pagerank"] = [pr.get(n, 0) for n in G.nodes()]

    df = pd.DataFrame(results).set_index("osmid")

    # Merge spatial coordinates
    df = df.join(nodes_gdf[["lat", "lon", "geometry"]])
    return df


# ---------------------------------------------------------------------------
# 4. Isochrone / Walkability Analysis
# ---------------------------------------------------------------------------

def get_isochrone(
    G: nx.MultiDiGraph,
    center_node: int,
    travel_time_min: Union[float, List[float]],
    speed_kph: float = 4.8,
) -> gpd.GeoDataFrame:
    """
    Generate isochrone polygon(s) — the area reachable within a given travel
    time by walking (or cycling / driving) from a centre node.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Projected network. Must have 'length' edge attributes (metres).
    center_node : int
        OSM node ID of the origin.
    travel_time_min : float or list of float
        Travel time threshold(s) in minutes. Multiple values produce
        concentric isochrone rings.
    speed_kph : float
        Travel speed in km/h. Default 4.8 km/h ≈ average walking speed.

    Returns
    -------
    gpd.GeoDataFrame
        One row per travel time threshold, with columns:
        'travel_time_min', 'geometry' (polygon in projected CRS).
    """
    if isinstance(travel_time_min, (int, float)):
        travel_time_min = [travel_time_min]

    travel_time_min = sorted(travel_time_min, reverse=True)
    speed_m_per_min = speed_kph * 1000 / 60
    records = []

    for t in travel_time_min:
        max_dist = t * speed_m_per_min
        subgraph = nx.ego_graph(G, center_node, radius=max_dist,
                                distance="length", undirected=True)
        node_points = [
            Point(data["x"], data["y"])
            for _, data in subgraph.nodes(data=True)
        ]
        if len(node_points) < 3:
            continue
        from shapely.ops import unary_union
        region = gpd.GeoSeries(node_points).buffer(max_dist * 0.05)
        iso_poly = unary_union(region).convex_hull

        records.append({
            "travel_time_min": t,
            "n_reachable_nodes": len(subgraph.nodes),
            "geometry": iso_poly,
        })

    crs = G.graph.get("crs", "EPSG:3857")
    return gpd.GeoDataFrame(records, crs=crs)


# ---------------------------------------------------------------------------
# 5. Nearest Node
# ---------------------------------------------------------------------------

def find_nearest_node(
    G: nx.MultiDiGraph,
    lat: float,
    lon: float,
) -> Tuple[int, float]:
    """
    Find the nearest graph node to a given WGS84 coordinate.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Projected network.
    lat : float
        Latitude (WGS84).
    lon : float
        Longitude (WGS84).

    Returns
    -------
    tuple : (node_id, distance_metres)
    """
    node_id, dist = ox.distance.nearest_nodes(G, X=lon, Y=lat, return_dist=True)
    return node_id, dist


# ---------------------------------------------------------------------------
# 6. Multi-City Comparison
# ---------------------------------------------------------------------------

def compare_cities_stats(
    places_list: List[str],
    network_type: str = "walk",
    cache_dir: str = "city_networks",
) -> pd.DataFrame:
    """
    Download networks and compute comparable statistics for multiple cities.

    Parameters
    ----------
    places_list : list of str
        List of geocodable place names.
    network_type : str
        Network type (consistent across all cities for fair comparison).
    cache_dir : str
        Directory to cache downloaded GraphML files.

    Returns
    -------
    pd.DataFrame
        One row per city, columns = network statistics.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    rows = []

    for place in places_list:
        safe_name = place.replace(", ", "_").replace(" ", "_")
        graphml_file = cache_path / f"{safe_name}_{network_type}.graphml"

        print(f"Processing: {place}")
        if graphml_file.exists():
            G = load_network(graphml_file)
        else:
            try:
                G = get_city_network(place, network_type=network_type)
                save_network(G, graphml_file)
            except Exception as e:
                print(f"  WARNING: failed to download {place}: {e}")
                continue

        stats = compute_network_stats(G, extended=True)
        stats["city"] = place
        rows.append(stats)

    df = pd.DataFrame(rows).set_index("city")
    return df
```

---

## Example 1: Walkability Comparison of 5 City Downtowns

Download walking networks for five downtown areas, compute network statistics and
circuity, and produce a comparative bar chart and spider plot.

```python
"""
example_city_walkability.py
---------------------------
Compare pedestrian street network characteristics for five world cities.
Uses OSMnx to download 1-km radius walking networks around each city centre.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import osmnx as ox
import networkx as nx
from osmnx_urban import (
    get_network_from_point,
    compute_network_stats,
    get_isochrone,
    find_nearest_node,
)

# City centres (lat, lon, label)
CITIES = [
    (40.7580, -73.9855, "Midtown Manhattan"),
    (48.8566,   2.3522, "Paris (1st arr.)"),
    (35.6762, 139.6503, "Shinjuku, Tokyo"),
    (-33.8688, 151.2093, "Sydney CBD"),
    (37.7749, -122.4194, "San Francisco"),
]
RADIUS_M = 1000          # 1-km radius around each centre
WALK_SPEED_KPH = 4.8     # km/h walking speed

stats_rows = []
isochrone_areas = {}

for lat, lon, label in CITIES:
    print(f"\nDownloading: {label} ...")
    G = get_network_from_point(lat, lon, dist=RADIUS_M, network_type="walk")
    stats = compute_network_stats(G, extended=True)
    stats["city"] = label
    stats_rows.append(stats)

    # 10-minute and 5-minute isochrones from city centre
    centre_node, _ = find_nearest_node(G, lat, lon)
    iso_gdf = get_isochrone(G, centre_node, travel_time_min=[5, 10],
                            speed_kph=WALK_SPEED_KPH)
    if not iso_gdf.empty:
        iso_10 = iso_gdf[iso_gdf["travel_time_min"] == 10]
        if not iso_10.empty:
            area_ha = iso_10.geometry.area.values[0] / 10000  # m² -> ha
            isochrone_areas[label] = area_ha
    else:
        isochrone_areas[label] = np.nan

df = pd.DataFrame(stats_rows).set_index("city")
df["walkable_area_10min_ha"] = pd.Series(isochrone_areas)

print("\n=== Network Statistics Summary ===")
display_cols = [
    "n", "m", "edge_length_avg_m", "circuity_avg",
    "street_density_km_per_km2", "node_density_per_km2",
    "walkable_area_10min_ha",
]
print(df[display_cols].round(2).to_string())

# ---- Visualisation 1: Bar chart comparison ----
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
metrics = [
    ("circuity_avg",              "Avg Circuity (lower = more grid-like)"),
    ("street_density_km_per_km2", "Street Density (km / km²)"),
    ("node_density_per_km2",      "Intersection Density (/ km²)"),
    ("edge_length_avg_m",         "Avg Block Length (m)"),
    ("walkable_area_10min_ha",    "10-min Walkable Area (ha)"),
    ("streets_per_node_avg",      "Streets per Intersection"),
]
colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

for ax, (col, label) in zip(axes.flat, metrics):
    vals = df[col].values
    bars = ax.bar(range(len(df)), vals, color=colors)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=20, ha="right", fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel(col.split("_")[0])
    # Annotate bar tops
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7)

fig.suptitle("Pedestrian Street Network Comparison — 1 km Radius from City Centre",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig("city_walkability_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved: city_walkability_comparison.png")
plt.show()

# ---- Visualisation 2: Network maps for all cities ----
fig2, axes2 = plt.subplots(1, 5, figsize=(25, 5))
for ax, (lat, lon, label) in zip(axes2, CITIES):
    G_plot = get_network_from_point(lat, lon, dist=RADIUS_M, network_type="walk")
    ox.plot_graph(
        G_plot, ax=ax,
        node_size=0, edge_color="#333333", edge_linewidth=0.5,
        bgcolor="white", show=False, close=False,
    )
    ax.set_title(label, fontsize=9)

fig2.suptitle("Walking Network Layouts — 1 km Radius", fontsize=12)
fig2.tight_layout()
fig2.savefig("city_network_maps.png", dpi=150, bbox_inches="tight")
print("Saved: city_network_maps.png")
plt.show()
```

---

## Example 2: Betweenness Centrality Map and Key Intersections

Download a city street network, compute betweenness centrality for all nodes, identify
the most central intersections, and produce a colour-scaled map.

```python
"""
example_betweenness_centrality.py
----------------------------------
Identify the most central intersections in a neighbourhood using
betweenness centrality and visualise the result as a colour-coded node map.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import osmnx as ox
import networkx as nx
import geopandas as gpd
from osmnx_urban import (
    get_city_network,
    compute_centrality,
    find_nearest_node,
)

# ---- 1. Download network ----
PLACE = "Williamsburg, Brooklyn, New York, USA"
print(f"Downloading network: {PLACE}")
G = get_city_network(PLACE, network_type="walk", simplify=True)
print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---- 2. Compute centrality measures ----
print("Computing centrality (betweenness approximation k=300) ...")
cent_df = compute_centrality(
    G, measures=["betweenness", "closeness", "pagerank", "degree"],
    weight="length",
)

# ---- 3. Identify top 10 most central intersections ----
# Filter to proper intersections (degree >= 3)
intersections = cent_df[cent_df["degree"] >= 3].copy()
top10 = intersections.nlargest(10, "betweenness")

print("\n=== Top 10 Most Central Intersections ===")
print(f"{'Rank':<5} {'Node ID':<12} {'Betweenness':>12} {'Lat':>10} {'Lon':>10}")
print("-" * 52)
for rank, (node_id, row) in enumerate(top10.iterrows(), 1):
    print(f"{rank:<5} {node_id:<12} {row['betweenness']:>12.6f} "
          f"{row['lat']:>10.5f} {row['lon']:>10.5f}")

# ---- 4. Colour-coded betweenness map ----
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
nodes_gdf = nodes_gdf.join(cent_df[["betweenness", "closeness", "pagerank"]])

# Normalise betweenness for colour mapping
bc_vals = nodes_gdf["betweenness"].fillna(0).values
bc_norm = (bc_vals - bc_vals.min()) / (bc_vals.max() - bc_vals.min() + 1e-12)

cmap = cm.plasma
node_colors = [cmap(v) for v in bc_norm]
node_sizes  = 2 + bc_norm * 30       # scale size by centrality

fig, ax = plt.subplots(figsize=(14, 12))
# Plot edges first (grey background)
edges_gdf.plot(ax=ax, color="#cccccc", linewidth=0.5, alpha=0.7)
# Plot nodes, coloured by betweenness
sc = ax.scatter(
    nodes_gdf.geometry.x, nodes_gdf.geometry.y,
    c=bc_vals, cmap="plasma", s=node_sizes,
    alpha=0.85, linewidths=0,
    norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=bc_vals.max()),
)

# Mark top-10 intersections
top_nodes = nodes_gdf.loc[top10.index]
ax.scatter(
    top_nodes.geometry.x, top_nodes.geometry.y,
    c="white", s=80, edgecolors="black", linewidths=1.5, zorder=5,
    label="Top-10 intersections",
)
for i, (nid, row) in enumerate(top10.iterrows(), 1):
    nrow = nodes_gdf.loc[nid]
    ax.annotate(
        str(i),
        xy=(nrow.geometry.x, nrow.geometry.y),
        xytext=(5, 5), textcoords="offset points",
        fontsize=7, color="white", fontweight="bold",
    )

cbar = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.01, shrink=0.7)
cbar.set_label("Betweenness Centrality", fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.set_title(f"Street Network Betweenness Centrality\n{PLACE}", fontsize=13)
ax.set_axis_off()
fig.tight_layout()
fig.savefig("betweenness_centrality_map.png", dpi=150, bbox_inches="tight")
print("\nSaved: betweenness_centrality_map.png")
plt.show()

# ---- 5. Closeness vs betweenness scatter ----
fig2, ax2 = plt.subplots(figsize=(8, 6))
sample = cent_df.sample(min(2000, len(cent_df)), random_state=42)
sc2 = ax2.scatter(
    sample["closeness"], sample["betweenness"],
    c=sample["pagerank"], cmap="viridis",
    s=8, alpha=0.6,
)
ax2.scatter(
    top10["closeness"], top10["betweenness"],
    c="red", s=60, zorder=5, label="Top-10 (betweenness)",
)
fig2.colorbar(sc2, ax=ax2, label="PageRank")
ax2.set_xlabel("Closeness Centrality")
ax2.set_ylabel("Betweenness Centrality")
ax2.set_title("Centrality Measures Comparison\n(coloured by PageRank)")
ax2.legend()
fig2.tight_layout()
fig2.savefig("centrality_scatter.png", dpi=150)
print("Saved: centrality_scatter.png")
plt.show()
```

---

## POI Extraction

```python
"""
poi_extraction.py
-----------------
Extract Points of Interest (POIs) from OpenStreetMap for a place,
compute nearest-POI distances for each street node, and assess
amenity accessibility.
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from osmnx_urban import get_city_network, find_nearest_node


def get_pois(
    place_name: str,
    amenity_types: list = None,
) -> gpd.GeoDataFrame:
    """
    Download POIs (amenity nodes/ways) for a place from OpenStreetMap.

    Parameters
    ----------
    place_name : str
        Nominatim geocodable place name.
    amenity_types : list of str, optional
        OSM amenity tags to retrieve, e.g. ['school', 'hospital', 'cafe'].
        If None, retrieves all amenity features.

    Returns
    -------
    gpd.GeoDataFrame with columns including 'name', 'amenity', 'geometry'.
    """
    tags = {"amenity": amenity_types if amenity_types else True}
    gdf = ox.features_from_place(place_name, tags=tags)
    # Keep only point geometries (centroids for polygon amenities)
    gdf["geometry"] = gdf.geometry.centroid
    return gdf[gdf.geometry.geom_type == "Point"].reset_index()


def nearest_poi_distance(
    G,
    pois: gpd.GeoDataFrame,
) -> pd.Series:
    """
    For each network node, compute the Euclidean distance to the nearest POI.

    Parameters
    ----------
    G : nx.MultiDiGraph
        Projected network.
    pois : gpd.GeoDataFrame
        Projected POI GeoDataFrame (same CRS as G).

    Returns
    -------
    pd.Series indexed by node ID, values = distance in metres.
    """
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    node_coords = np.column_stack([nodes_gdf.geometry.x, nodes_gdf.geometry.y])
    poi_proj = pois.to_crs(nodes_gdf.crs)
    poi_coords = np.column_stack([poi_proj.geometry.x, poi_proj.geometry.y])

    tree = cKDTree(poi_coords)
    distances, _ = tree.query(node_coords)

    return pd.Series(distances, index=nodes_gdf.index, name="dist_to_nearest_poi_m")


# Usage example
if __name__ == "__main__":
    PLACE = "Prenzlauer Berg, Berlin, Germany"
    G = get_city_network(PLACE, network_type="walk")
    pois = get_pois(PLACE, amenity_types=["cafe", "restaurant", "bakery"])
    print(f"Found {len(pois)} food/drink POIs in {PLACE}")

    # Reproject POIs to match network CRS
    pois_proj = pois.to_crs(G.graph["crs"])
    distances = nearest_poi_distance(G, pois_proj)

    print(f"\nDistance to nearest cafe/restaurant/bakery:")
    print(f"  Median: {distances.median():.0f} m")
    print(f"  Mean:   {distances.mean():.0f} m")
    print(f"  Max:    {distances.max():.0f} m")
    pct_within_200m = (distances < 200).mean() * 100
    print(f"  Nodes within 200m of food POI: {pct_within_200m:.1f}%")
```

---

## Routing Example

```python
"""
routing_example.py
------------------
Compute the shortest path between two points and plot it on the street network.
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from osmnx_urban import get_city_network, find_nearest_node


def shortest_path_map(
    G,
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    weight: str = "length",
    output_path: str = "route_map.png",
) -> dict:
    """
    Find and visualise the shortest path between two coordinates.
    """
    orig_node, orig_dist = find_nearest_node(G, origin_lat, origin_lon)
    dest_node, dest_dist = find_nearest_node(G, dest_lat, dest_lon)

    route = nx.shortest_path(G, orig_node, dest_node, weight=weight)
    route_length = sum(
        G[u][v][0].get("length", 0) for u, v in zip(route[:-1], route[1:])
    )

    fig, ax = ox.plot_graph_route(
        G, route,
        route_linewidth=4, route_color="firebrick",
        node_size=0, bgcolor="white", edge_color="#cccccc",
        edge_linewidth=0.5,
        show=False, close=False,
        figsize=(12, 10),
    )
    ax.set_title(f"Shortest Path — {route_length:.0f} m", fontsize=12)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Route saved: {output_path}")
    plt.show()

    return {
        "origin_node": orig_node,
        "dest_node": dest_node,
        "route_nodes": route,
        "route_length_m": route_length,
        "n_turns": len(route) - 1,
    }
```

---

## Tips and Best Practices

- **Caching**: Always enable `ox.settings.use_cache = True`. Re-downloading large city
  networks wastes time and is inconsiderate to the Overpass API.
- **Rate limiting**: For batch jobs over many cities, add a small sleep between requests
  (`ox.settings.overpass_rate_limit = True`) or use a local Overpass instance.
- **Graph projection**: OSMnx returns graphs in WGS84 by default. Call `ox.project_graph(G)`
  to reproject to the local UTM zone before computing distances, areas, or buffers.
- **Simplification**: `simplify=True` removes intermediate nodes (degree-2 nodes along
  straight roads). Use `simplify=False` only when you need every OSM node for attribute
  access or fine-grained routing.
- **Large networks**: For cities with >100,000 nodes, use `k` approximation in betweenness
  centrality (e.g. `k=500`) to keep runtime under a minute.
- **Network type**: Use `network_type="walk"` for pedestrian accessibility analysis.
  For cycling infrastructure studies, use `"bike"` to include cycle paths.
- **Reproducibility**: Save downloaded networks with `ox.save_graphml` and commit to DVC
  or store in object storage. OSM data changes daily.

---

## References

- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analysing, and
  visualising complex street networks. *Computers, Environment and Urban Systems*, 65, 126–139.
- Boeing, G. (2019). Urban spatial order: street network orientation, configuration, and
  entropy. *Applied Network Science*, 4, 67.
- OpenStreetMap contributors. OpenStreetMap. https://www.openstreetmap.org
- NetworkX documentation: https://networkx.org
- OSMnx documentation: https://osmnx.readthedocs.io
