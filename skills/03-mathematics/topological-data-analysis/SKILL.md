---
name: topological-data-analysis
description: Apply persistent homology, Betti numbers, and mapper graphs to extract topological features from complex datasets.
tags:
  - topology
  - persistent-homology
  - tda
  - feature-engineering
  - mathematics
  - machine-learning
version: "1.0.0"
authors:
  - name: "Rosetta Skills Contributors"
    github: "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - gudhi>=3.8
  - ripser>=0.6
  - scikit-tda>=1.1
  - matplotlib>=3.7
  - numpy>=1.24
  - pandas>=2.0
last_updated: "2026-03-17"
status: stable
---

# Topological Data Analysis (TDA)

Topological Data Analysis (TDA) provides mathematically rigorous methods for discovering the "shape" of data. Using tools like persistent homology, Betti numbers, persistence diagrams, and the Mapper algorithm, TDA uncovers multi-scale structural features invisible to classical statistics.

## When to Use This Skill

- You need shape-aware features that are robust to noise and deformation in point cloud data.
- Classical distance-based methods miss higher-order relational structure (loops, voids, clusters).
- You are working with time-series, biological sequences, images, or sensor networks where topology is meaningful.
- You want to detect connectivity, circular patterns, or cavities in high-dimensional datasets.
- You need features that are invariant to continuous deformations of the input space.
- You are engineering features for downstream machine learning pipelines and want topologically informed representations.

Typical domains: drug discovery (molecular topology), neuroscience (brain network loops), materials science (pore structures), computer vision (shape matching), and financial data (market regime topology).

## Background & Key Concepts

### Simplicial Complexes

A simplicial complex is a generalization of a graph that includes higher-dimensional "simplices": vertices (0-simplex), edges (1-simplex), triangles (2-simplex), tetrahedra (3-simplex), and so on. Given a point cloud, we build a nested family of simplicial complexes by varying a distance parameter epsilon.

### Vietoris-Rips Complex

For a point cloud X and threshold epsilon, the Vietoris-Rips complex VR(X, epsilon) contains a simplex for every finite subset of X with diameter at most epsilon. As epsilon grows from 0 to infinity, new topological features (connected components, loops, voids) are born and die.

### Persistent Homology

Persistent homology tracks topological features across a filtration of simplicial complexes. For each dimension k:
- **H_0** counts connected components (Betti number beta_0).
- **H_1** counts independent loops / 1-cycles (Betti number beta_1).
- **H_2** counts enclosed voids / 2-cycles (Betti number beta_2).

Each feature has a birth time b and a death time d. The persistence (d - b) measures how long the feature survives — long-lived features signal genuine topology; short-lived ones are noise.

### Persistence Diagrams and Barcodes

A persistence diagram is a scatter plot of (birth, death) pairs for each topological feature. A barcode is the equivalent interval representation. Features far from the diagonal are topologically significant.

### Betti Numbers

The k-th Betti number beta_k counts independent k-dimensional cycles in a topological space:
- beta_0 = number of connected components
- beta_1 = number of independent loops
- beta_2 = number of enclosed voids

### Mapper Algorithm

Mapper is a TDA algorithm that compresses high-dimensional point clouds into a graph (the Mapper graph) that captures topological shape. It uses a lens function (e.g., PCA projection, density), a cover of the range, partial clustering within preimages, and nerve construction to yield a navigable network.

### Wasserstein and Bottleneck Distance

These metrics measure distances between persistence diagrams. The bottleneck distance is the inf over matchings of the sup of point distances. The p-Wasserstein distance is the inf over matchings of the Lp norm of point distances.

## Environment Setup

```bash
# Create and activate a virtual environment
python -m venv tda-env
source tda-env/bin/activate  # On Windows: tda-env\Scripts\activate

# Install core TDA libraries
pip install gudhi>=3.8 ripser>=0.6 scikit-tda>=1.1

# Install supporting scientific stack
pip install numpy>=1.24 pandas>=2.0 matplotlib>=3.7 scikit-learn>=1.3 scipy>=1.11

# Optional: persim for diagram distances and vectorizations
pip install persim>=0.3

# Verify installation
python -c "import gudhi; print('gudhi', gudhi.__version__)"
python -c "import ripser; print('ripser OK')"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
```

```bash
# If gudhi installation fails on some platforms, try conda
conda install -c conda-forge gudhi
```

No API keys are required for TDA computations. All processing is local.

## Core Workflow

### Step 1: Generate or Load a Point Cloud

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons

# --- Synthetic example: two concentric circles ---
np.random.seed(42)
X_circles, _ = make_circles(n_samples=300, noise=0.05, factor=0.4)

# --- Synthetic example: noisy torus (parametric) ---
def sample_torus(n=500, R=3.0, r=1.0, noise=0.1):
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi   = np.random.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    pts = np.column_stack([x, y, z])
    pts += noise * np.random.randn(*pts.shape)
    return pts

X_torus = sample_torus(n=600)

# --- Loading from a CSV (real data) ---
# df = pd.read_csv("my_point_cloud.csv")
# X_real = df[["feature_1", "feature_2", "feature_3"]].to_numpy()

print(f"Circles shape: {X_circles.shape}")
print(f"Torus shape:   {X_torus.shape}")
```

### Step 2: Compute Persistent Homology with ripser

```python
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt

# --- Compute persistent homology up to dimension 2 ---
result_circles = ripser(X_circles, maxdim=1)
result_torus   = ripser(X_torus,   maxdim=2)

diagrams_circles = result_circles["dgms"]  # list: dgms[k] = array of (birth, death) pairs
diagrams_torus   = result_torus["dgms"]

# --- Print Betti numbers at a chosen threshold ---
def betti_at_threshold(dgms, threshold):
    betti = []
    for k, dgm in enumerate(dgms):
        if len(dgm) == 0:
            betti.append(0)
            continue
        # Count features alive at threshold (born <= threshold < death)
        alive = np.sum((dgm[:, 0] <= threshold) & (dgm[:, 1] > threshold))
        betti.append(int(alive))
    return betti

eps = 0.5
b_circles = betti_at_threshold(diagrams_circles, eps)
print(f"Betti numbers for circles at eps={eps}: {b_circles}")
# Expected: [1, 2] -> 1 component, 2 loops (inner + outer circle)

# --- Plot persistence diagrams ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plt.sca(axes[0])
plot_diagrams(diagrams_circles, show=False)
axes[0].set_title("Persistence Diagram: Two Circles")

plt.sca(axes[1])
plot_diagrams(diagrams_torus, show=False)
axes[1].set_title("Persistence Diagram: Torus")

plt.tight_layout()
plt.savefig("persistence_diagrams.png", dpi=150)
plt.show()
print("Saved persistence_diagrams.png")
```

### Step 3: Compute Persistent Homology with GUDHI

```python
import gudhi
import gudhi.representations
import numpy as np
import matplotlib.pyplot as plt

# --- Build a Rips complex from the circles dataset ---
rips_complex = gudhi.RipsComplex(
    points=X_circles,
    max_edge_length=2.0  # maximum edge length for the filtration
)

# Create a simplex tree up to dimension 2
simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

print(f"Number of simplices: {simplex_tree.num_simplices()}")
print(f"Number of vertices:  {simplex_tree.num_vertices()}")

# --- Compute persistence ---
simplex_tree.compute_persistence()

# Extract persistence pairs for each dimension
diag_gudhi = simplex_tree.persistence()  # list of (dim, (birth, death))

# Separate by dimension
def extract_by_dim(diag, dim):
    return np.array(
        [(b, d) for (k, (b, d)) in diag if k == dim and np.isfinite(d)],
        dtype=float
    )

h0 = extract_by_dim(diag_gudhi, 0)
h1 = extract_by_dim(diag_gudhi, 1)

print(f"H0 features: {len(h0)}")
print(f"H1 features: {len(h1)}")

# --- Plot barcode ---
gudhi.plot_persistence_barcode(diag_gudhi)
plt.title("Persistence Barcode: Two Circles (GUDHI)")
plt.savefig("barcode_circles.png", dpi=150)
plt.show()

# --- Plot persistence diagram ---
gudhi.plot_persistence_diagram(diag_gudhi)
plt.title("Persistence Diagram: Two Circles (GUDHI)")
plt.savefig("diagram_circles_gudhi.png", dpi=150)
plt.show()
```

### Step 4: Persistence Landscape and Vectorization

```python
import gudhi.representations as gr
import numpy as np

# GUDHI representations for machine learning pipelines

# Diagrams as list-of-arrays format expected by gudhi.representations
diag_list_h1 = [h1]  # wrap in list for transformer API

# --- Persistence Landscape ---
landscape = gr.Landscape(num_landscapes=5, resolution=100)
L = landscape.fit_transform(diag_list_h1)
print(f"Landscape shape: {L.shape}")  # (1, 5*100)

# --- Betti Curve ---
betti_curve = gr.BettiCurve(resolution=100)
B = betti_curve.fit_transform(diag_list_h1)
print(f"Betti curve shape: {B.shape}")  # (1, 100)

# --- Persistence Image ---
pers_image = gr.PersistenceImage(bandwidth=0.05, weight=lambda x: x[1], resolution=[20, 20])
PI = pers_image.fit_transform(diag_list_h1)
print(f"Persistence image shape: {PI.shape}")  # (1, 400)

# --- Silhouette ---
silhouette = gr.Silhouette(weight=lambda x: x[1]**2, resolution=100)
S = silhouette.fit_transform(diag_list_h1)
print(f"Silhouette shape: {S.shape}")  # (1, 100)

# Plot persistence image
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.imshow(PI[0].reshape(20, 20), origin="lower", cmap="hot_r")
plt.colorbar(label="Persistence density")
plt.title("Persistence Image (H1)")
plt.savefig("persistence_image.png", dpi=150)
plt.show()
```

## Advanced Usage

### Mapper Graph Construction

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import networkx as nx

# Attempt to use kmapper (part of scikit-tda / KeplerMapper)
try:
    import kmapper as km
    from kmapper import Cover

    np.random.seed(0)
    X, labels = make_circles(n_samples=500, noise=0.03, factor=0.5)

    # Initialize Mapper
    mapper = km.KeplerMapper(verbose=1)

    # Lens: 1D PCA projection
    lens = mapper.fit_transform(X, projection=PCA(n_components=1))

    # Build the graph
    graph = mapper.map(
        lens,
        X,
        cover=Cover(n_cubes=10, perc_overlap=0.5),
        clusterer=DBSCAN(eps=0.1, min_samples=5)
    )

    # Visualize
    mapper.visualize(
        graph,
        path_html="mapper_circles.html",
        title="Mapper: Two Circles",
        color_values=labels,
        color_function_name="Ground truth label"
    )
    print("Mapper graph saved to mapper_circles.html")

    # Convert to NetworkX for analysis
    G_mapper = nx.Graph()
    for node in graph["nodes"]:
        G_mapper.add_node(node, size=len(graph["nodes"][node]))
    for node, neighbors in graph["links"].items():
        for neighbor in neighbors:
            G_mapper.add_edge(node, neighbor)

    print(f"Mapper graph: {G_mapper.number_of_nodes()} nodes, {G_mapper.number_of_edges()} edges")
    print(f"Connected components: {nx.number_connected_components(G_mapper)}")

except ImportError:
    print("kmapper not installed. Install with: pip install kmapper")
```

### TDA-Based Feature Engineering for Classification

```python
import numpy as np
import pandas as pd
from ripser import ripser
from persim import PersStats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Generate multi-class dataset
np.random.seed(42)
X_raw, y = make_classification(n_samples=200, n_features=10, n_informative=5, n_classes=3, random_state=42)

def compute_tda_features(X_point_cloud, maxdim=1):
    """
    Compute a fixed-length TDA feature vector from a point cloud.
    Features: persistence statistics for each homology dimension.
    """
    result = ripser(X_point_cloud, maxdim=maxdim)
    dgms = result["dgms"]

    features = []
    for k, dgm in enumerate(dgms):
        # Filter out infinite deaths
        finite_mask = np.isfinite(dgm[:, 1])
        dgm_finite = dgm[finite_mask]

        if len(dgm_finite) == 0:
            # Pad with zeros if no features in this dimension
            features.extend([0.0] * 6)
            continue

        persistence = dgm_finite[:, 1] - dgm_finite[:, 0]
        features.extend([
            float(np.sum(persistence)),          # total persistence
            float(np.max(persistence)),           # max persistence
            float(np.mean(persistence)),          # mean persistence
            float(np.std(persistence)),           # std of persistence
            float(len(persistence)),              # number of features
            float(np.median(persistence)),        # median persistence
        ])
    return np.array(features)

# For each sample, treat its 10 features as a 1D time-series point cloud
# In real applications, each sample might be a separate point cloud
tda_features = np.array([
    compute_tda_features(X_raw[i].reshape(-1, 1))
    for i in range(len(X_raw))
])

print(f"TDA feature matrix shape: {tda_features.shape}")

# Build a classification pipeline
clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(n_estimators=100, random_state=42))
])

scores = cross_val_score(clf_pipeline, tda_features, y, cv=5, scoring="accuracy")
print(f"TDA-feature classification accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### Wasserstein Distance Between Diagrams

```python
from persim import wasserstein, bottleneck
from ripser import ripser
from sklearn.datasets import make_circles, make_moons
import numpy as np

np.random.seed(0)
X1, _ = make_circles(n_samples=200, noise=0.05, factor=0.4)
X2, _ = make_moons(n_samples=200, noise=0.05)
X3, _ = make_circles(n_samples=200, noise=0.05, factor=0.4)  # Same distribution

dgm1 = ripser(X1, maxdim=1)["dgms"][1]  # H1 diagrams
dgm2 = ripser(X2, maxdim=1)["dgms"][1]
dgm3 = ripser(X3, maxdim=1)["dgms"][1]

d_12_w = wasserstein(dgm1, dgm2)
d_13_w = wasserstein(dgm1, dgm3)
d_12_b = bottleneck(dgm1, dgm2)
d_13_b = bottleneck(dgm1, dgm3)

print(f"Wasserstein(circles, moons)   = {d_12_w:.4f}")
print(f"Wasserstein(circles, circles) = {d_13_w:.4f}")
print(f"Bottleneck(circles, moons)    = {d_12_b:.4f}")
print(f"Bottleneck(circles, circles)  = {d_13_b:.4f}")
# Expected: d_13 << d_12 (same topology should be closer)
```

### Cubical Homology for Image Data

```python
import gudhi
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters

# Load a grayscale image and compute its cubical persistent homology
image = data.coins()  # 303x384 grayscale

# Downsample to speed up computation
from skimage.transform import resize
image_small = resize(image, (60, 80), anti_aliasing=True)

# GUDHI cubical complex
cc = gudhi.CubicalComplex(
    dimensions=list(image_small.shape),
    top_dimensional_cells=image_small.flatten().tolist()
)
cc.compute_persistence()

diag_img = cc.persistence()
print(f"Number of persistence pairs: {len(diag_img)}")

# Plot
gudhi.plot_persistence_diagram(diag_img)
plt.title("Cubical Persistence: Coins Image")
plt.savefig("cubical_coins.png", dpi=150)
plt.show()
```

## Troubleshooting

**Problem: `ripser` is slow on large point clouds (N > 5000)**
```bash
# Use the fast C++ backend via scikit-tda / ripser++
pip install ripser
# For even larger clouds, consider subsampling with farthest point sampling
```

```python
def farthest_point_sample(X, n_samples):
    """Greedy farthest-point subsampling of a point cloud."""
    n = len(X)
    selected = [np.random.randint(n)]
    dists = np.full(n, np.inf)
    for _ in range(n_samples - 1):
        last = selected[-1]
        d = np.sum((X - X[last]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return X[selected]

X_sub = farthest_point_sample(X_torus, 300)
```

**Problem: GUDHI installation fails on Windows**
```bash
# Use conda-forge channel
conda install -c conda-forge gudhi
# Or use WSL/Linux for native builds
```

**Problem: Persistence diagram has too many noise points near the diagonal**
```python
# Filter low-persistence features before vectorization
def filter_diagram(dgm, min_persistence=0.05):
    persistence = dgm[:, 1] - dgm[:, 0]
    return dgm[persistence >= min_persistence]

dgm_filtered = filter_diagram(diagrams_circles[1], min_persistence=0.1)
print(f"Features before filter: {len(diagrams_circles[1])}")
print(f"Features after filter:  {len(dgm_filtered)}")
```

**Problem: `kmapper` graph is empty or disconnected**
```python
# Increase overlap percentage or reduce number of cubes
graph = mapper.map(
    lens, X,
    cover=Cover(n_cubes=8, perc_overlap=0.7),  # more overlap
    clusterer=DBSCAN(eps=0.2, min_samples=3)    # looser clustering
)
```

**Problem: Memory error when computing high-dimensional Rips complex**
```python
# Limit max dimension and max edge length
result = ripser(X, maxdim=1, thresh=1.5)  # thresh caps the filtration
```

**Problem: Inconsistent results between ripser and GUDHI**

Both libraries implement the standard persistence algorithm but may differ in tie-breaking for equal filtration values. This is expected. For reproducibility, fix `np.random.seed` and use consistent preprocessing.

## External Resources

- GUDHI Library documentation: https://gudhi.inria.fr/python/latest/
- Ripser Python bindings: https://ripser.scikit-tda.org/
- Scikit-TDA project: https://scikit-tda.org/
- Edelsbrunner & Harer, "Computational Topology" (2010) — foundational textbook
- Carlsson, "Topology and Data", Bulletin of the AMS (2009)
- KeplerMapper documentation: https://kepler-mapper.scikit-tda.org/
- Persim (persistence image/distance tools): https://persim.scikit-tda.org/
- TDA tutorial by Mathieu Carriere: https://github.com/MathieuCarriere/sklearn-tda

## Examples

### Example 1: Full Pipeline — TDA Feature Engineering on Real Tabular Data

```python
"""
End-to-end TDA pipeline on the Wisconsin Breast Cancer dataset.
We treat each sample's feature vector as a 1D signal point cloud,
extract topological features, and classify using a Random Forest.
"""

import numpy as np
import pandas as pd
from ripser import ripser
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# --- Load dataset ---
bc = load_breast_cancer()
X_raw, y = bc.data, bc.target
feature_names = bc.feature_names

print(f"Dataset shape: {X_raw.shape}")
print(f"Classes: {bc.target_names}")

# --- Normalize features before TDA ---
scaler_pre = StandardScaler()
X_scaled = scaler_pre.fit_transform(X_raw)

# --- Extract TDA features per sample ---
def tda_feature_vector(row_1d, n_points_per_feature=1):
    """
    Treat a 1D feature array as a scalar time series.
    Build a sliding-window point cloud and compute H0 and H1 persistence stats.
    """
    # Create 2D delay-embedding (Takens embedding)
    ts = row_1d
    n = len(ts) - 1
    pc = np.column_stack([ts[:-1], ts[1:]])  # 2D delay embedding

    result = ripser(pc, maxdim=1)
    dgms = result["dgms"]

    feats = []
    for k in range(2):
        dgm = dgms[k]
        fin = dgm[np.isfinite(dgm[:, 1])]
        if len(fin) == 0:
            feats.extend([0.0] * 5)
        else:
            p = fin[:, 1] - fin[:, 0]
            feats.extend([
                np.sum(p),
                np.max(p),
                np.mean(p),
                np.std(p) if len(p) > 1 else 0.0,
                float(len(p))
            ])
    return np.array(feats)

print("Computing TDA features (this may take ~30 seconds)...")
tda_feats = np.array([tda_feature_vector(X_scaled[i]) for i in range(len(X_scaled))])
print(f"TDA feature matrix: {tda_feats.shape}")

# --- Train / test split ---
X_tr, X_te, y_tr, y_te = train_test_split(tda_feats, y, test_size=0.25, random_state=42, stratify=y)

# --- Pipeline ---
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(n_estimators=200, random_state=42))
])

pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)

print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=bc.target_names))

cv_scores = cross_val_score(pipe, tda_feats, y, cv=5, scoring="f1_macro")
print(f"5-fold CV F1 (macro): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

### Example 2: Multi-Scale Topology Analysis of a 3D Point Cloud

```python
"""
Analyse the multi-scale topology of a sampled 3D surface.
We compute persistent homology, plot barcodes by dimension,
and report Betti numbers at multiple scales.
"""

import numpy as np
import matplotlib.pyplot as plt
import gudhi
from ripser import ripser

# --- Sample a genus-2 surface (two tori joined) ---
def sample_two_tori(n=800, R=3.0, r=1.0, sep=7.0, noise=0.08):
    """Sample two tori separated along the x-axis."""
    def single_torus(center_x, n_pts):
        theta = np.random.uniform(0, 2 * np.pi, n_pts)
        phi   = np.random.uniform(0, 2 * np.pi, n_pts)
        x = center_x + (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        return np.column_stack([x, y, z])

    pts1 = single_torus(-sep / 2, n // 2)
    pts2 = single_torus( sep / 2, n - n // 2)
    pts = np.vstack([pts1, pts2])
    pts += noise * np.random.randn(*pts.shape)
    return pts

np.random.seed(7)
X_two_tori = sample_two_tori(n=800)

# --- Subsample to speed up ---
idx = np.random.choice(len(X_two_tori), 300, replace=False)
X_sub = X_two_tori[idx]

# --- Compute persistent homology ---
result = ripser(X_sub, maxdim=2)
dgms = result["dgms"]

# --- Betti numbers at multiple scales ---
thresholds = np.linspace(0.2, 4.0, 20)
betti_curves = {0: [], 1: [], 2: []}

for eps in thresholds:
    for k in range(3):
        dgm = dgms[k]
        finite = dgm[np.isfinite(dgm[:, 1])]
        alive = int(np.sum((finite[:, 0] <= eps) & (finite[:, 1] > eps)))
        # Also count infinite features for H0 (they live forever)
        inf_feat = dgm[~np.isfinite(dgm[:, 1])]
        alive += int(np.sum(inf_feat[:, 0] <= eps))
        betti_curves[k].append(alive)

# --- Plot Betti number curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for k, color in zip(range(3), ["steelblue", "tomato", "forestgreen"]):
    ax.plot(thresholds, betti_curves[k], label=f"beta_{k}", color=color, linewidth=2)
ax.set_xlabel("Scale (epsilon)")
ax.set_ylabel("Betti number")
ax.set_title("Betti Numbers vs Scale: Two Tori")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Persistence barcode ---
ax2 = axes[1]
colors_dim = {0: "steelblue", 1: "tomato", 2: "forestgreen"}
y_offset = 0
for k in range(3):
    dgm = dgms[k]
    for (b, d) in dgm:
        d_plot = min(d, thresholds[-1]) if not np.isfinite(d) else d
        ax2.plot([b, d_plot], [y_offset, y_offset], color=colors_dim[k], linewidth=1.5, alpha=0.7)
        y_offset += 1
ax2.set_xlabel("Filtration value")
ax2.set_title("Persistence Barcode")
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors_dim[k], label=f"H{k}") for k in range(3)]
ax2.legend(handles=legend_elements)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("two_tori_topology.png", dpi=150)
plt.show()

# --- Summary ---
print("Topology Summary for Two Tori:")
for k in range(3):
    dgm = dgms[k]
    finite = dgm[np.isfinite(dgm[:, 1])]
    persistence = finite[:, 1] - finite[:, 0]
    significant = persistence[persistence > 0.3]
    print(f"  H{k}: {len(dgm)} total features, {len(significant)} significant (persistence > 0.3)")

# Expected for two tori:
#   H0: ~2 significant (2 connected components at coarse scale)
#   H1: ~4 significant (2 longitudinal + 2 meridional loops, one per torus)
#   H2: ~2 significant (one void per torus)
```
