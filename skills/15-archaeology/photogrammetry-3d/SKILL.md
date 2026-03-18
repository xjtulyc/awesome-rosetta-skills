---
name: photogrammetry-3d
description: >
  Use this Skill for archaeological photogrammetry: 3D point cloud processing,
  mesh analysis, cross-section profiles, and volume measurement with Open3D.
tags:
  - archaeology
  - photogrammetry
  - 3d-modeling
  - point-cloud
  - open3d
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
    - open3d>=0.18
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
    - pandas>=2.0
last_updated: "2026-03-17"
status: "stable"
---

# Archaeological Photogrammetry and 3D Analysis

> **One-line summary**: Process archaeological 3D models from photogrammetry: point cloud cleaning, surface normal estimation, cross-section extraction, volume computation, and color analysis with Open3D.

---

## When to Use This Skill

- When processing photogrammetric point clouds of archaeological sites or objects
- When computing surface areas and volumes from 3D meshes
- When extracting cross-sectional profiles from artifacts or excavations
- When performing outlier removal and surface smoothing
- When analyzing color distribution on ceramic or painted surfaces
- When computing roughness metrics for lithic use-wear analysis

**Trigger keywords**: photogrammetry, 3D model, point cloud, Open3D, mesh, SfM, structure from motion, cross-section, volume measurement, surface normal, artifact 3D, site survey, DEM, digital elevation model

---

## Background & Key Concepts

### Structure from Motion (SfM) Photogrammetry

SfM reconstructs 3D geometry from overlapping 2D photographs:
1. Feature detection and matching (SIFT/ORB)
2. Camera pose estimation
3. Bundle adjustment
4. Dense point cloud generation
5. Mesh reconstruction (Poisson or Delaunay)

Common software: Agisoft Metashape, COLMAP (free), RealityCapture.

### Point Cloud Processing Pipeline

1. Load PLY/LAS/XYZ file
2. Statistical outlier removal (SOR filter)
3. Normal estimation (PCA over k-NN)
4. Surface reconstruction (Ball-Pivoting or Poisson)
5. Mesh simplification and hole filling

### Roughness (Use-Wear Proxy)

$$
R_a = \frac{1}{N}\sum_{i=1}^N |z_i - \bar{z}|
$$

Higher roughness on lithic edges indicates use-wear; smooth facets = unused or polished.

---

## Environment Setup

### Install Dependencies

```bash
pip install open3d>=0.18 numpy>=1.24 scipy>=1.11 matplotlib>=3.7 pandas>=2.0
# For LAS/LAZ files:
pip install laspy>=2.5
```

### Verify Installation

```python
import open3d as o3d
import numpy as np
print(f"Open3D version: {o3d.__version__}")

# Create a simple point cloud for testing
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.random.randn(1000, 3))
print(f"Test point cloud: {len(pcd.points)} points")
```

---

## Core Workflow

### Step 1: Point Cloud Loading, Cleaning, and Normal Estimation

```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Simulate an archaeological artifact point cloud (ceramic vessel)
# In practice: pcd = o3d.io.read_point_cloud("artifact.ply")
# ------------------------------------------------------------------ #

def create_ceramic_vessel_pcd(n_points=5000, noise_level=0.002):
    """
    Simulate a pottery vessel point cloud (truncated ellipsoid + rim).
    """
    # Generate points on truncated ellipsoid surface
    phi   = np.random.uniform(0, np.pi * 0.75, n_points)  # Polar angle (truncated at top)
    theta = np.random.uniform(0, 2*np.pi, n_points)         # Azimuthal angle

    # Vessel shape (belly wider than neck)
    r_belly = 0.12  # meters
    r_neck  = 0.06
    h_total = 0.25

    # Parametric profile: wider in middle
    t = phi / (np.pi * 0.75)
    r_profile = r_belly * np.sin(np.pi * t) + r_neck * (1 - np.sin(np.pi * t))

    x = r_profile * np.sin(phi) * np.cos(theta)
    y = r_profile * np.sin(phi) * np.sin(theta)
    z = h_total * (1 - phi / (np.pi * 0.75))  # Height decreases with phi

    # Add photogrammetric noise
    xyz = np.column_stack([x, y, z])
    xyz += np.random.randn(*xyz.shape) * noise_level

    # Add color (terracotta-like: reddish-brown)
    base_color = np.array([0.78, 0.47, 0.33])
    colors = base_color + np.random.randn(n_points, 3) * 0.05
    colors = np.clip(colors, 0, 1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

pcd_raw = create_ceramic_vessel_pcd(n_points=5000, noise_level=0.003)
print(f"Raw point cloud: {len(pcd_raw.points)} points")

# ---- Statistical Outlier Removal (SOR filter) -------------------- #
pcd_cleaned, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
outlier_count = len(pcd_raw.points) - len(pcd_cleaned.points)
print(f"After SOR filtering: {len(pcd_cleaned.points)} points ({outlier_count} outliers removed)")

# ---- Estimate surface normals ----------------------------------- #
pcd_cleaned.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
)
pcd_cleaned.orient_normals_towards_camera_location(camera_location=[0, 0, 1])
print("Surface normals estimated and oriented")

# ---- Voxel downsampling for efficiency -------------------------- #
pcd_down = pcd_cleaned.voxel_down_sample(voxel_size=0.005)
print(f"After voxel downsampling: {len(pcd_down.points)} points")

# ---- 2D Projection for visualization (top view) ------------------ #
pts = np.asarray(pcd_down.points)
colors = np.asarray(pcd_down.colors) if pcd_down.has_colors() else np.ones((len(pts), 3)) * 0.7

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, (i, j, title) in zip(axes, [(0,2,'Top view (XZ)'), (0,1,'Side view (XY)'), (1,2,'Front view (YZ)')]):
    ax.scatter(pts[:,i], pts[:,j], c=colors, s=2, alpha=0.6)
    ax.set_xlabel(['X','X','Y'][axes.tolist().index(ax)] + ' (m)')
    ax.set_ylabel(['Z','Y','Z'][axes.tolist().index(ax)] + ' (m)')
    ax.set_title(title); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

plt.suptitle(f"Ceramic Vessel Point Cloud ({len(pts)} points after cleaning)")
plt.tight_layout()
plt.savefig("point_cloud_views.png", dpi=150)
plt.show()
```

### Step 2: Surface Reconstruction and Volume Computation

```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Reconstruct surface mesh and compute volume, surface area
# ------------------------------------------------------------------ #

# Use the cleaned+downsampled point cloud from Step 1
# (Regenerate if running independently)
pcd_down = create_ceramic_vessel_pcd(n_points=3000, noise_level=0.001)
pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
)
pcd_down.orient_normals_towards_camera_location([0, 0, 1])

# ---- Poisson surface reconstruction ----------------------------- #
print("Running Poisson surface reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd_down, depth=8, width=0, scale=1.1, linear_fit=False
)

# Trim low-density regions (artifacts of reconstruction)
densities_arr = np.asarray(densities)
density_threshold = np.percentile(densities_arr, 10)
vertices_to_remove = densities_arr < density_threshold
mesh.remove_vertices_by_mask(vertices_to_remove)
mesh.remove_degenerate_triangles()
mesh.remove_non_manifold_edges()

print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

# ---- Compute geometric properties ------------------------------ #
surface_area_m2 = mesh.get_surface_area()
print(f"Surface area: {surface_area_m2*1e4:.2f} cm²")

# Volume (if mesh is watertight)
if mesh.is_watertight():
    volume_m3 = mesh.get_volume()
    volume_ml = volume_m3 * 1e6
    print(f"Volume: {volume_ml:.0f} mL ({volume_ml/1000:.3f} L)")
else:
    print("Mesh not watertight — attempting to fill holes...")
    mesh.fill_holes()
    if mesh.is_watertight():
        volume_m3 = mesh.get_volume()
        print(f"Volume (after fill): {volume_m3*1e6:.0f} mL")
    else:
        print("Volume: mesh still not watertight; use alpha shape or convex hull approximation")
        hull, _ = pcd_down.compute_convex_hull()
        vol_approx = hull.get_volume()
        print(f"Convex hull volume (upper bound): {vol_approx*1e6:.0f} mL")

# ---- Extract height profile ------------------------------------- #
pts = np.asarray(pcd_down.points)
z_min, z_max = pts[:,2].min(), pts[:,2].max()
n_slices = 30
z_levels = np.linspace(z_min, z_max, n_slices)
radii = []

for z_lev in z_levels:
    mask = np.abs(pts[:,2] - z_lev) < 0.005
    if mask.sum() > 5:
        slice_pts = pts[mask, :2]  # XY coordinates
        radii.append(np.linalg.norm(slice_pts, axis=1).mean())
    else:
        radii.append(np.nan)

radii = np.array(radii)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Height profile (vessel cross-section)
axes[0].plot(radii * 100, z_levels * 100, 'b-', linewidth=2)
axes[0].plot(-radii * 100, z_levels * 100, 'b-', linewidth=2)  # Mirror
axes[0].fill_betweenx(z_levels*100, -radii*100, radii*100, alpha=0.2, color='brown')
axes[0].set_xlabel("Radius (cm)"); axes[0].set_ylabel("Height (cm)")
axes[0].set_title("Vessel Cross-Section Profile"); axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# Radius vs. height (rim and belly detection)
axes[1].plot(z_levels * 100, radii * 100, 'ro-', linewidth=1.5, markersize=5)
axes[1].set_xlabel("Height (cm)"); axes[1].set_ylabel("Radius (cm)")
axes[1].set_title("Radius Profile — Vessel Morphology")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("vessel_profile.png", dpi=150)
plt.show()

print(f"\nVessel metrics:")
print(f"  Height:            {(z_max-z_min)*100:.1f} cm")
print(f"  Maximum diameter:  {np.nanmax(radii)*200:.1f} cm")
print(f"  Rim diameter:      {radii[-1]*200:.1f} cm  (at top)")
print(f"  Base diameter:     {radii[0]*200:.1f} cm  (at bottom)")
```

### Step 3: Color Analysis and Surface Roughness

```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ------------------------------------------------------------------ #
# Analyze color distribution and surface roughness
# ------------------------------------------------------------------ #

# Regenerate point cloud (or load from file)
pcd = create_ceramic_vessel_pcd(n_points=5000)
pcd, _ = pcd.remove_statistical_outlier(20, 2.0)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))

pts    = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)
normals = np.asarray(pcd.normals)

# ---- Color analysis: RGB histogram by zone ---------------------- #
z_min, z_max = pts[:,2].min(), pts[:,2].max()
lower_third = pts[:,2] < z_min + (z_max-z_min)/3
upper_third = pts[:,2] > z_min + 2*(z_max-z_min)/3
middle = ~lower_third & ~upper_third

zones = {'Base zone': lower_third, 'Body zone': middle, 'Neck/rim': upper_third}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Color scatter (Hue visualization)
hue = colors[:,0]  # Red channel as proxy for hue
sc = axes[0][0].scatter(pts[:,0]*100, pts[:,1]*100, c=colors, s=4, alpha=0.7)
axes[0][0].set_xlabel("X (cm)"); axes[0][0].set_ylabel("Y (cm)")
axes[0][0].set_title("Color Distribution (Top View)"); axes[0][0].set_aspect('equal')

# RGB by zone
for ax, (zone_name, mask) in zip([axes[0][1], axes[1][0], axes[1][1]], zones.items()):
    if mask.sum() > 0:
        zone_colors = colors[mask]
        ax.hist(zone_colors[:,0], bins=30, alpha=0.5, color='red',   density=True, label='R')
        ax.hist(zone_colors[:,1], bins=30, alpha=0.5, color='green', density=True, label='G')
        ax.hist(zone_colors[:,2], bins=30, alpha=0.5, color='blue',  density=True, label='B')
        ax.set_title(f"RGB Distribution — {zone_name}\n(n={mask.sum()} points)")
        ax.set_xlabel("Intensity"); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("color_analysis.png", dpi=150)
plt.show()

# ---- Surface Roughness (Ra) ------------------------------------- #
# Roughness from deviation of normals from mean local normal
normal_z = normals[:,2]  # Z-component of surface normal (approx. surface curvature)
roughness_proxy = 1 - np.abs(normal_z)  # Low values = flat, High = curved/rough

print("\nSurface roughness (normal deviation proxy):")
print(f"  Mean roughness: {roughness_proxy.mean():.4f}")
print(f"  Std roughness:  {roughness_proxy.std():.4f}")
print(f"  Max roughness:  {roughness_proxy.max():.4f}")

# Roughness by height zone
print("\nRoughness by zone:")
for zone_name, mask in zones.items():
    if mask.sum() > 0:
        r = roughness_proxy[mask].mean()
        print(f"  {zone_name}: mean roughness = {r:.4f}")
```

---

## Advanced Usage

### ICP Registration (Aligning Fragments)

```python
import open3d as o3d
import numpy as np

# ------------------------------------------------------------------ #
# Align two artifact fragments using Iterative Closest Point (ICP)
# Useful for sherds from the same vessel
# ------------------------------------------------------------------ #

# Create two offset point clouds (simulating fragments)
pcd1 = create_ceramic_vessel_pcd(500)
# Simulate fragment 2: partial + offset + small rotation
pts2 = np.asarray(pcd1.points)[:250] + np.array([0.05, 0.02, 0.01])
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pts2)
pcd2.paint_uniform_color([0.8, 0.3, 0.3])

pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.02, 30))
pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.02, 30))

# ICP registration
threshold = 0.02
trans_init = np.eye(4)  # Initial transformation
reg = o3d.pipelines.registration.registration_icp(
    pcd2, pcd1, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
)
print(f"ICP registration fitness: {reg.fitness:.4f}")
print(f"ICP RMSE: {reg.inlier_rmse*1000:.2f} mm")
print(f"Translation: {reg.transformation[:3,3]*100} cm")
```

---

## Troubleshooting

### Open3D visualization doesn't open on headless server

```python
# Use offscreen rendering
import open3d as o3d
o3d.visualization.rendering.OffscreenRenderer  # Check availability

# OR: convert to numpy and plot with matplotlib (as in examples above)
pts = np.asarray(pcd.points)
import matplotlib.pyplot as plt
fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts[::10,0], pts[::10,1], pts[::10,2], s=1)
plt.savefig("pcd.png")
```

### Poisson reconstruction creates artifacts

**Fix**: Increase depth or add more points:
```python
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9, width=0, scale=1.1, linear_fit=True
)
# Remove low-density vertices (mesh boundary artifacts)
densities_arr = np.asarray(densities)
keep = densities_arr > np.percentile(densities_arr, 15)
mesh.remove_vertices_by_mask(~keep)
```

### Error: `AttributeError: module 'open3d' has no attribute 'geometry'`

```bash
pip install open3d --upgrade
# Make sure not to confuse with older open3D package names
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| open3d | 0.18 | API stable; 0.18 added better Poisson API |
| numpy | 1.24, 1.26 | Required by open3d |
| scipy | 1.11, 1.12 | For supplementary spatial analysis |

---

## External Resources

### Official Documentation

- [Open3D documentation](http://www.open3d.org/docs/release/)
- [COLMAP SfM (free, open-source)](https://colmap.github.io)

### Key Papers

- Schindler, G. & Dellaert, F. (2012). *4D cities*. 3DV Workshop.
- Opitz, R. & Limp, W.F. (2015). *Recent developments in high-density survey and measurement (HDSM) for archaeology*. Annual Review of Anthropology.

---

## Examples

### Example 1: Flat Surface Extraction for Profile Drawing

```python
import numpy as np
import matplotlib.pyplot as plt

# Extract a vertical cross-section from the vessel
pts = np.asarray(pcd.points)
# Slice at Y ≈ 0 (±2mm)
slice_mask = np.abs(pts[:,1]) < 0.002
slice_pts = pts[slice_mask]

fig, ax = plt.subplots(figsize=(5, 8))
ax.scatter(slice_pts[:,0]*100, slice_pts[:,2]*100, s=3, c='brown', alpha=0.7)
ax.set_xlabel("X (cm)"); ax.set_ylabel("Z/Height (cm)")
ax.set_title("Vessel Cross-Section Profile\n(Y ≈ 0 plane)")
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("cross_section_profile.png", dpi=150); plt.show()
```

### Example 2: Comparing Two Vessel Forms

```python
import numpy as np
import matplotlib.pyplot as plt

# Compare two vessel profile curves
def vessel_profile(belly_r, neck_r, height, n_pts=50):
    z = np.linspace(0, height, n_pts)
    t = z / height
    r = belly_r * np.sin(np.pi * t) + neck_r * (1 - np.sin(np.pi * t))
    return z, r

fig, ax = plt.subplots(figsize=(7, 7))
for (br, nr, h, label, color) in [
    (12, 5, 25, 'Wide-mouth jar', 'blue'),
    (8, 7, 20,  'Storage jar',    'red'),
]:
    z, r = vessel_profile(br, nr, h)
    ax.plot(r, z, color=color, linewidth=2.5, label=label)
    ax.plot(-r, z, color=color, linewidth=2.5)
    ax.fill_betweenx(z, -r, r, alpha=0.1, color=color)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel("Radius (cm)"); ax.set_ylabel("Height (cm)")
ax.set_title("Comparative Vessel Profile Analysis"); ax.legend()
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("vessel_comparison.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
