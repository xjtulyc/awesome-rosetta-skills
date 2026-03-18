---
name: mayavi-3d-viz
description: >
  Use this Skill for 3D scientific visualization with Mayavi: vector fields,
  isosurfaces, volume rendering, and animated 3D plots for physics data.
tags:
  - physics
  - visualization
  - mayavi
  - 3d
  - vtk
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
    - mayavi>=4.8
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# 3D Scientific Visualization with Mayavi

> **One-line summary**: Create publication-quality 3D scientific visualizations — vector fields, isosurfaces, streamlines, and volume rendering — using Mayavi and VTK.

---

## When to Use This Skill

- When visualizing 3D scalar fields (temperature, pressure, density)
- When plotting vector fields (electromagnetic, fluid velocity, magnetic)
- When rendering isosurfaces from volumetric data
- When creating streamlines and stream tubes for flow visualization
- When animating 3D physics simulations
- When exporting high-resolution 3D renders for publications

**Trigger keywords**: Mayavi, VTK, 3D visualization, isosurface, vector field, streamlines, volume rendering, mlab, 3D plot, scientific visualization, scalar field

---

## Background & Key Concepts

### Mayavi Architecture

Mayavi is built on VTK (Visualization Toolkit) and provides a Python API (`mlab`) for rapid 3D visualization:

- **mlab**: High-level API similar to matplotlib for 3D plots
- **Pipeline**: Source → Filter → Module chain (VTK-style)
- **Off-screen rendering**: Useful for batch/server rendering

### Key Visualization Types

| Data Type | Mayavi Module |
|:----------|:-------------|
| Scalar field (volume) | `mlab.contour3d`, `mlab.volume_slice` |
| Vector field | `mlab.quiver3d`, `mlab.flow` |
| Isosurface | `mlab.contour3d` |
| Streamlines | `mlab.flow` |
| Point cloud | `mlab.points3d` |

### Off-screen Rendering

For server/CI environments without a display:

```python
from mayavi import mlab
mlab.options.offscreen = True
```

---

## Environment Setup

### Install Dependencies

```bash
# Install via conda (recommended — VTK binary wheels)
conda install -c conda-forge mayavi

# Or via pip (requires VTK)
pip install mayavi>=4.8 numpy>=1.24 scipy>=1.11 matplotlib>=3.7
# On headless servers, also install:
pip install pyopengl
export DISPLAY=:0  # or use xvfb-run on Linux
```

### Verify Installation

```python
import numpy as np
from mayavi import mlab

# Test off-screen rendering
mlab.options.offscreen = True
fig = mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
x, y, z = np.mgrid[-2:2:20j, -2:2:20j, -2:2:20j]
s = np.exp(-(x**2 + y**2 + z**2))
mlab.contour3d(s, contours=5, opacity=0.5)
mlab.savefig("test_mayavi.png")
mlab.close()
print("Mayavi off-screen rendering: OK")
```

---

## Core Workflow

### Step 1: Scalar Field Visualization

```python
import numpy as np
from mayavi import mlab

mlab.options.offscreen = True  # Remove for interactive mode

# ------------------------------------------------------------------ #
# Visualize a 3D Gaussian scalar field with isosurfaces
# ------------------------------------------------------------------ #
N = 60
x, y, z = np.mgrid[-3:3:N*1j, -3:3:N*1j, -3:3:N*1j]

# Anisotropic Gaussian
sigma_x, sigma_y, sigma_z = 1.0, 1.5, 0.8
field = np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2) + z**2/(2*sigma_z**2)))

# ----- Figure 1: Isosurfaces ------------------------------------ #
fig1 = mlab.figure(bgcolor=(0.1, 0.1, 0.1), size=(800, 600))

# Multiple isosurfaces at different levels
for level, color, opacity in [
    (0.8, (1.0, 0.2, 0.2), 0.9),   # Inner (red)
    (0.5, (0.2, 0.8, 0.2), 0.5),   # Middle (green)
    (0.2, (0.2, 0.2, 1.0), 0.2),   # Outer (blue)
]:
    mlab.contour3d(x, y, z, field, contours=[level],
                   color=color, opacity=opacity)

mlab.colorbar(title="Field value", orientation='vertical')
mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
mlab.title("3D Gaussian — Isosurfaces", size=0.3)
mlab.view(azimuth=45, elevation=60, distance=12)
mlab.savefig("scalar_field_isosurfaces.png", magnification=2)
print("Saved: scalar_field_isosurfaces.png")

# ----- Figure 2: Volume slices ---------------------------------- #
fig2 = mlab.figure(bgcolor=(0.95, 0.95, 0.95), size=(800, 600))
mlab.pipeline.volume(mlab.pipeline.scalar_field(x, y, z, field))
mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
mlab.view(azimuth=30, elevation=70, distance=14)
mlab.savefig("scalar_field_volume.png", magnification=2)
mlab.close(all=True)
print("Saved: scalar_field_volume.png")
```

### Step 2: Vector Field Visualization

```python
import numpy as np
from mayavi import mlab

mlab.options.offscreen = True

# ------------------------------------------------------------------ #
# Visualize magnetic dipole field: B = (3(m·r̂)r̂ - m) / r³
# ------------------------------------------------------------------ #
N = 15
x, y, z = np.mgrid[-3:3:N*1j, -3:3:N*1j, -3:3:N*1j]

# Magnetic moment along z-axis
m = np.array([0, 0, 1])
eps = 0.3  # Avoid singularity at origin

r = np.sqrt(x**2 + y**2 + z**2) + eps
r_hat_x = x / r
r_hat_y = y / r
r_hat_z = z / r

# m · r̂
m_dot_r = m[2] * r_hat_z  # Only z-component of m is non-zero

# B = (3(m·r̂)r̂ - m) / r³
Bx = (3 * m_dot_r * r_hat_x - m[0]) / r**3
By = (3 * m_dot_r * r_hat_y - m[1]) / r**3
Bz = (3 * m_dot_r * r_hat_z - m[2]) / r**3

B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

fig = mlab.figure(bgcolor=(0.05, 0.05, 0.15), size=(900, 700))

# Quiver plot (arrows) — subsample to avoid clutter
step = 2
mlab.quiver3d(
    x[::step, ::step, ::step],
    y[::step, ::step, ::step],
    z[::step, ::step, ::step],
    Bx[::step, ::step, ::step],
    By[::step, ::step, ::step],
    Bz[::step, ::step, ::step],
    scalars=B_mag[::step, ::step, ::step],
    mode='arrow',
    scale_factor=0.3,
    colormap='jet',
)

# Add magnetic moment marker (dipole location)
mlab.points3d(0, 0, 0, scale_factor=0.3, color=(1, 1, 0))  # Yellow dot

# Field lines (streamlines seeded on circle in xy-plane)
seed_x = np.cos(np.linspace(0, 2*np.pi, 12)) * 1.0
seed_y = np.sin(np.linspace(0, 2*np.pi, 12)) * 1.0
seed_z = np.zeros(12)

field_src = mlab.pipeline.vector_field(x, y, z, Bx, By, Bz, scalars=B_mag)
streamlines = mlab.pipeline.streamline(
    field_src,
    seedtype='point',
    seed_resolution=0,
    integration_direction='both',
    linetype='tube',
    tube_radius=0.05,
)

mlab.colorbar(title="|B| (a.u.)", orientation='vertical')
mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
mlab.title("Magnetic Dipole Field", size=0.3)
mlab.view(azimuth=45, elevation=55, distance=14)
mlab.savefig("vector_field_dipole.png", magnification=2)
mlab.close(all=True)
print("Saved: vector_field_dipole.png")
```

### Step 3: 3D Streamlines for Fluid Flow

```python
import numpy as np
from mayavi import mlab

mlab.options.offscreen = True

# ------------------------------------------------------------------ #
# Potential flow: flow past a sphere using superposition
# Uniform flow U₀ + doublet (Stokes stream function)
# ------------------------------------------------------------------ #
N = 25
x, y, z = np.mgrid[-3:3:N*1j, -3:3:N*1j, -3:3:N*1j]

U0 = 1.0  # Free stream velocity
R  = 0.8  # Sphere radius

r = np.sqrt(x**2 + y**2 + z**2) + 1e-6

# Potential flow velocity components
# u = U0(1 - R³/r³ * (1 - 3x²/r²) * ...)
# Simplified for irrotational sphere flow:
factor = R**3 / r**5

u = U0 * (1 - factor * (r**2 - 3*x**2))
v = U0 * (3 * factor * x * y)
w = U0 * (3 * factor * x * z)

# Mask inside sphere
inside = r < R
u[inside] = 0; v[inside] = 0; w[inside] = 0

velocity_mag = np.sqrt(u**2 + v**2 + w**2)

fig = mlab.figure(bgcolor=(0.1, 0.15, 0.2), size=(1000, 700))

# Sphere surface
sphere_phi, sphere_theta = np.mgrid[0:np.pi:50j, 0:2*np.pi:50j]
sx = R * np.sin(sphere_phi) * np.cos(sphere_theta)
sy = R * np.sin(sphere_phi) * np.sin(sphere_theta)
sz = R * np.cos(sphere_phi)
mlab.mesh(sx, sy, sz, color=(0.7, 0.7, 0.9), opacity=0.8)

# Streamlines seeded on yz-plane upstream
src_y = np.linspace(-2, 2, 8)
src_z = np.linspace(-2, 2, 8)
SY, SZ = np.meshgrid(src_y, src_z)
SX = -2.5 * np.ones_like(SY)

field = mlab.pipeline.vector_field(x, y, z, u, v, w, scalars=velocity_mag)
stream = mlab.pipeline.streamline(
    field,
    seedtype='plane',
    integration_direction='forward',
    linetype='tube',
    tube_radius=0.03,
    colormap='cool',
)
stream.seed.widget.origin = np.array([-2.5, -2.0, -2.0])
stream.seed.widget.point1 = np.array([-2.5,  2.0, -2.0])
stream.seed.widget.point2 = np.array([-2.5, -2.0,  2.0])
stream.seed.widget.resolution = 7

mlab.colorbar(title="|u| (m/s)", orientation='vertical')
mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
mlab.title("Potential Flow Past a Sphere", size=0.25)
mlab.view(azimuth=20, elevation=70, distance=14)
mlab.savefig("streamlines_sphere.png", magnification=2)
mlab.close(all=True)
print("Saved: streamlines_sphere.png")
```

---

## Advanced Usage

### Animation — Rotating Scalar Field

```python
import numpy as np
from mayavi import mlab
import os

mlab.options.offscreen = True

# Create animated PNG sequence for a time-evolving field
N = 40
x, y, z = np.mgrid[-3:3:N*1j, -3:3:N*1j, -3:3:N*1j]

os.makedirs("frames", exist_ok=True)

for frame, t in enumerate(np.linspace(0, 2*np.pi, 36)):
    field = np.sin(x + t) * np.cos(y - t/2) * np.exp(-z**2/4)

    fig = mlab.figure(bgcolor=(0.0, 0.0, 0.0), size=(600, 600))
    mlab.contour3d(x, y, z, field, contours=4, colormap='RdBu',
                   opacity=0.6, vmin=-1, vmax=1)
    mlab.axes()
    mlab.view(azimuth=frame * 10, elevation=60, distance=15)
    mlab.savefig(f"frames/frame_{frame:03d}.png")
    mlab.close()

print("Frames saved. Combine with: ffmpeg -r 12 -i frames/frame_%03d.png animation.mp4")
```

---

## Troubleshooting

### Error: `No display name and no $DISPLAY environment variable`

**Cause**: Headless server, no X11 display.

**Fix**:
```bash
# Linux: use xvfb virtual display
pip install pyopengl
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1
# OR use EGL offscreen (no Xvfb needed)
export ETS_TOOLKIT=null
```

```python
from mayavi import mlab
mlab.options.offscreen = True  # Must set BEFORE any mlab calls
```

### Error: Import error on `tvtk` or `traits`

```bash
pip install traits traitsui pyface envisage
```

### Poor render quality / aliasing

```python
fig = mlab.figure(size=(1600, 1200))  # Higher resolution
mlab.savefig("output.png", magnification=3)  # Anti-aliased output
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| mayavi | 4.8.x | Requires VTK ≥ 9.0 |
| vtk | 9.0, 9.2 | Install via conda for prebuilt wheels |
| traits | 6.x | API stable |

---

## External Resources

### Official Documentation

- [Mayavi documentation](https://docs.enthought.com/mayavi/mayavi/)
- [VTK Examples Python](https://examples.vtk.org/site/Python/)

### Tutorials

- Ramachandran, P. & Varoquaux, G. (2011). *Mayavi: 3D Visualization of Scientific Data*. Computing in Science & Engineering, 13(2), 40–51.

---

## Examples

### Example 1: Electric Field of Two Point Charges

```python
import numpy as np
from mayavi import mlab

mlab.options.offscreen = True

N = 20
x, y, z = np.mgrid[-3:3:N*1j, -3:3:N*1j, -3:3:N*1j]

k_e = 1.0  # Normalized Coulomb constant

def point_charge_field(qx, qy, qz, q, x, y, z):
    """Electric field of a point charge at (qx,qy,qz)."""
    dx, dy, dz = x - qx, y - qy, z - qz
    r3 = (dx**2 + dy**2 + dz**2 + 0.1)**1.5
    return k_e * q * dx/r3, k_e * q * dy/r3, k_e * q * dz/r3

# Dipole: +1 at x=+1, -1 at x=-1
Ex1, Ey1, Ez1 = point_charge_field( 1, 0, 0, +1, x, y, z)
Ex2, Ey2, Ez2 = point_charge_field(-1, 0, 0, -1, x, y, z)

Ex = Ex1 + Ex2
Ey = Ey1 + Ey2
Ez = Ez1 + Ez2
E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

fig = mlab.figure(bgcolor=(0.05, 0.05, 0.1), size=(800, 600))
mlab.quiver3d(x[::2,::2,::2], y[::2,::2,::2], z[::2,::2,::2],
              Ex[::2,::2,::2], Ey[::2,::2,::2], Ez[::2,::2,::2],
              scalars=E_mag[::2,::2,::2], mode='arrow',
              scale_factor=0.4, colormap='hot')
mlab.points3d([ 1, -1], [0, 0], [0, 0],
              [1, -1], scale_factor=0.3, colormap='bwr', vmin=-1, vmax=1)
mlab.colorbar(title="|E| (a.u.)")
mlab.title("Electric Dipole Field")
mlab.view(azimuth=20, elevation=70, distance=12)
mlab.savefig("electric_dipole.png", magnification=2)
mlab.close(all=True)
```

### Example 2: Wavefunction Probability Density

```python
import numpy as np
from mayavi import mlab
from scipy.special import sph_harm
from scipy.special import genlaguerre, factorial

mlab.options.offscreen = True

N = 60
# Hydrogen-like wavefunction: ψ_{nlm} = R_{nl}(r) * Y_l^m(θ,φ)
x, y, z = np.mgrid[-10:10:N*1j, -10:10:N*1j, -10:10:N*1j]

r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
theta = np.arccos(z / r)
phi = np.arctan2(y, x)

# n=2, l=1, m=0  (2p_z orbital)
n, l, m_q = 2, 1, 0
a0 = 1.0  # Bohr radius (normalized)

# Radial part R_{21}: (1/sqrt(24)) * (r/a0) * exp(-r/(2a0))
R21 = (1/np.sqrt(24)) * (r/a0) * np.exp(-r/(2*a0))

# Spherical harmonic Y_1^0
Y10 = sph_harm(m_q, l, phi, theta).real

psi = R21 * Y10
prob_density = np.abs(psi)**2

fig = mlab.figure(bgcolor=(0.0, 0.0, 0.05), size=(800, 700))
# Isosurface at 50% of maximum
threshold = 0.5 * prob_density.max()
mlab.contour3d(x, y, z, prob_density, contours=[threshold],
               colormap='hot', opacity=0.7)
mlab.axes(xlabel='x/a₀', ylabel='y/a₀', zlabel='z/a₀')
mlab.title("H atom 2p_z orbital  |ψ|²", size=0.3)
mlab.view(azimuth=30, elevation=70, distance=30)
mlab.savefig("hydrogen_2pz.png", magnification=2)
mlab.close(all=True)
print("Saved: hydrogen_2pz.png")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
