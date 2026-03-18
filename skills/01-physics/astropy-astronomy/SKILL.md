---
name: astropy-astronomy
description: Astronomical data analysis with astropy and astroquery — FITS I/O, WCS transforms, catalog cross-matching, aperture photometry, and CMB power spectra.
tags:
  - astronomy
  - astrophysics
  - fits
  - wcs
  - photometry
  - cmb
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
  - astropy>=5.3
  - astroquery>=0.4.6
  - healpy>=1.16
  - matplotlib>=3.7
  - numpy>=1.24
last_updated: "2026-03-17"
status: stable
---

# Astropy Astronomy

A comprehensive skill for astronomical data analysis using the Python astronomy ecosystem.
Covers reading and writing FITS files, World Coordinate System (WCS) transforms, remote
catalog queries, aperture photometry on imaging data, and CMB angular power spectrum
analysis using HEALPix pixelisation.

## When to Use This Skill

Use this skill when you need to:

- Read, write, or inspect FITS images and tables produced by telescopes or pipelines
- Convert between pixel coordinates and sky coordinates (RA/Dec, Galactic, Ecliptic)
- Cross-match source lists against Vizier or SIMBAD to identify objects
- Perform aperture or PSF photometry on CCD images
- Analyse CMB or large-scale-structure maps stored as HEALPix ``.fits`` files
- Automate catalogue retrieval (Gaia, 2MASS, SDSS, NED) without manual downloads
- Build reproducible astronomy pipelines that run on any machine

This skill is **not** appropriate for:

- Real-time telescope control or instrument communication (use INDI / ASCOM instead)
- Spectral extraction from 2-D spectra (use ``specutils`` skill)
- N-body or hydrodynamical simulations (use ``yt`` or ``pynbody`` skill)

## Background & Key Concepts

### FITS — Flexible Image Transport System

FITS is the standard file format in astronomy. A FITS file contains one or more
**Header/Data Units (HDUs)**. Each HDU has a text header of 80-character keyword cards
and an optional data array or binary table. Common HDU types:

| HDU type | astropy class | Typical use |
|---|---|---|
| PrimaryHDU | `fits.PrimaryHDU` | 2-D image, metadata-only |
| ImageHDU | `fits.ImageHDU` | Additional image planes |
| BinTableHDU | `fits.BinTableHDU` | Source catalogs, spectra |
| CompImageHDU | `fits.CompImageHDU` | Tile-compressed image |

### WCS — World Coordinate System

The FITS WCS standard maps pixel indices ``(x, y)`` to sky coordinates
``(RA, Dec)`` via a projection (TAN, ZEA, CAR, …) plus a linear transform
encoded in the header keywords ``CRPIX``, ``CRVAL``, ``CD`` / ``CDELT``.
``astropy.wcs.WCS`` reads these keywords and provides ``pixel_to_world`` /
``world_to_pixel`` methods that return ``SkyCoord`` objects.

### SkyCoord and Angle

``astropy.coordinates.SkyCoord`` represents one or more positions on the sky in
any supported reference frame. Angle arithmetic, frame conversions, and
on-sky separations are all handled automatically. Key frames: ``ICRS`` (RA/Dec),
``Galactic`` (l, b), ``FK5``, ``AltAz`` (horizontal coordinates).

### Catalog Cross-Matching

``astroquery`` provides uniform Python interfaces to dozens of online services:

- **Vizier** — CDS catalogue service (Gaia, 2MASS, SDSS, …)
- **SIMBAD** — Object identification database
- **NED** — NASA/IPAC Extragalactic Database
- **ESASky / MAST / IRSA** — Archive portals

All queries return ``astropy.table.Table`` objects directly.

### HEALPix and healpy

HEALPix (Hierarchical Equal Area isoLatitude Pixelisation) divides the sphere
into ``12 * nside**2`` equal-area pixels. ``healpy`` wraps the C++ HEALPix
library and provides map I/O (``read_map``, ``write_map``), spherical harmonic
transforms (``map2alm``, ``alm2cl``), and visualisation (``mollview``, ``gnomview``).
The angular power spectrum ``C_l`` is the key observable for CMB cosmology.

### Aperture Photometry

``photutils`` (companion to astropy) implements circular and elliptical apertures,
sky annuli for local background estimation, and source detection (``DAOStarFinder``,
``IRAFStarFinder``). The result is a flux in counts per second that can be converted
to magnitudes using a photometric zero-point.

---

## Environment Setup

### Install dependencies

```bash
pip install "astropy>=5.3" "astroquery>=0.4.6" "healpy>=1.16" \
            "matplotlib>=3.7" "numpy>=1.24" photutils
```

On Apple Silicon / conda:

```bash
conda install -c conda-forge astropy astroquery healpy matplotlib numpy photutils
```

### Optional: download test FITS data

```bash
# Fetch a small SDSS r-band cutout for testing
python - <<'EOF'
from astroquery.skyview import SkyView
imgs = SkyView.get_images("M51", survey=["SDSSr"], pixels=512)
imgs[0].writeto("m51_r.fits", overwrite=True)
print("Saved m51_r.fits")
EOF
```

### Environment variables

No API keys are required for the public Vizier / SIMBAD / SkyView services.
If you use the ESO archive or proprietary data portals, store credentials as:

```bash
export ESO_USERNAME="<your-username>"
export ESO_PASSWORD=$(cat ~/.eso_passwd)   # read from file, never hardcode
```

Access them in Python with:

```python
import os
username = os.getenv("ESO_USERNAME", "")
password = os.getenv("ESO_PASSWORD", "")
```

---

## Core Workflow

### Step 1 — Read and inspect a FITS file

```python
from astropy.io import fits
import numpy as np

# Open without loading data into memory
with fits.open("m51_r.fits") as hdul:
    hdul.info()                          # print HDU summary
    header = hdul[0].header
    data   = hdul[0].data.astype(float)  # pixel array (counts)

print(f"Image shape : {data.shape}")
print(f"Instrument  : {header.get('INSTRUME', 'unknown')}")
print(f"Filter      : {header.get('FILTER',   'unknown')}")
print(f"Exposure    : {header.get('EXPTIME',  'unknown')} s")
print(f"Min / Max   : {data.min():.1f} / {data.max():.1f} counts")

# Mask NaN / Inf pixels that can corrupt photometry
data = np.where(np.isfinite(data), data, 0.0)
```

### Step 2 — WCS coordinate transforms

```python
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

with fits.open("m51_r.fits") as hdul:
    wcs = WCS(hdul[0].header)
    data = hdul[0].data.astype(float)

ny, nx = data.shape

# Pixel centre of the image -> sky coordinates
cx, cy = nx / 2.0, ny / 2.0
sky_centre = wcs.pixel_to_world(cx, cy)
print(f"Image centre: RA={sky_centre.ra.deg:.4f} deg, "
      f"Dec={sky_centre.dec.deg:.4f} deg")

# Convert to Galactic coordinates
gal = sky_centre.galactic
print(f"Galactic    : l={gal.l.deg:.4f} deg, b={gal.b.deg:.4f} deg")

# Sky -> pixel (useful for placing apertures on known objects)
m51_coord = SkyCoord(ra=202.4696 * u.deg, dec=47.1952 * u.deg)
pix = wcs.world_to_pixel(m51_coord)
print(f"M51 nucleus at pixel ({pix[0]:.1f}, {pix[1]:.1f})")

# Compute pixel scale
pixel_scale = wcs.proj_plane_pixel_scales()
print(f"Pixel scale : {pixel_scale[0].to(u.arcsec):.3f} / pixel")
```

### Step 3 — Catalog query with astroquery (Vizier + SIMBAD)

```python
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

# -- Gaia DR3 point sources within 5 arcmin of M51 --
coord = SkyCoord(ra=202.4696 * u.deg, dec=47.1952 * u.deg, frame="icrs")
radius = 5 * u.arcmin

v = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag", "BP-RP"],
           row_limit=500)
result = v.query_region(coord, radius=radius, catalog="I/355/gaiadr3")

if result:
    gaia = result[0]
    print(f"Found {len(gaia)} Gaia DR3 sources")
    print(gaia["Source", "RA_ICRS", "DE_ICRS", "Gmag"][:5])
else:
    print("No Gaia sources returned (check network)")

# -- SIMBAD identification of M51 --
simbad = Simbad()
simbad.add_votable_fields("distance", "flux(V)", "sptype")
result_s = simbad.query_object("M51")
if result_s:
    print("\nSIMBAD result for M51:")
    print(result_s["MAIN_ID", "RA", "DEC", "FLUX_V"])
```

### Step 4 — Aperture photometry

```python
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.detection import DAOStarFinder
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry)
from photutils.background import MMMBackground

with fits.open("m51_r.fits") as hdul:
    data   = hdul[0].data.astype(float)
    header = hdul[0].header
    wcs    = WCS(header)

# Estimate and subtract sky background
bkg_estimator = MMMBackground()
bkg = bkg_estimator(data)
data_sub = data - bkg
print(f"Background level: {bkg:.2f} counts")

# Detect sources with DAOStarFinder
from astropy.stats import sigma_clipped_stats
_, median, std = sigma_clipped_stats(data, sigma=3.0)
finder = DAOStarFinder(fwhm=3.5, threshold=5.0 * std)
sources = finder(data_sub)
print(f"Detected {len(sources)} sources")

# Perform circular aperture photometry (r=5 px, sky annulus 7-10 px)
positions = np.column_stack([sources["xcentroid"], sources["ycentroid"]])
aperture  = CircularAperture(positions, r=5.0)
annulus   = CircularAnnulus(positions, r_in=7.0, r_out=10.0)

phot_table = aperture_photometry(data_sub, [aperture, annulus])

# Local background correction
annulus_area   = annulus.area
aperture_area  = aperture.area
sky_per_pixel  = phot_table["aperture_sum_1"] / annulus_area
sky_in_aper    = sky_per_pixel * aperture_area
flux_corrected = phot_table["aperture_sum_0"] - sky_in_aper

# Instrumental magnitude (zero-point = 25.0 as example)
ZERO_POINT = 25.0
exptime    = header.get("EXPTIME", 1.0)
flux_rate  = flux_corrected / exptime
mag        = ZERO_POINT - 2.5 * np.log10(np.abs(flux_rate))

phot_table["flux_corrected"] = flux_corrected
phot_table["mag_inst"]       = mag
phot_table.sort("mag_inst")
print(phot_table["id", "xcenter", "ycenter", "flux_corrected", "mag_inst"][:10])
```

### Step 5 — CMB power spectrum with healpy

```python
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# ---- Simulate a Gaussian CMB temperature map ----
nside  = 512                          # HEALPix resolution (npix = 12*nside^2)
lmax   = 3 * nside - 1                # maximum multipole
npix   = hp.nside2npix(nside)

# Planck 2018-like flat LCDM power spectrum (toy model)
ells   = np.arange(lmax + 1)
Cl     = np.zeros(lmax + 1)
# Simple Sachs-Wolfe plateau + first acoustic peak shape
Cl[2:] = 1e-10 * (1.0 / (ells[2:] * (ells[2:] + 1))) * (
    1.0 + 6.0 * np.exp(-((ells[2:] - 200) / 40) ** 2)
)

# Generate a realisation
np.random.seed(42)
cmb_map = hp.synfast(Cl, nside=nside, lmax=lmax, verbose=False)
print(f"Map shape : {cmb_map.shape}, RMS = {cmb_map.std():.3e} K")

# Recover power spectrum from the map
alm      = hp.map2alm(cmb_map, lmax=lmax)
Cl_rec   = hp.alm2cl(alm)
ell_rec  = np.arange(len(Cl_rec))

# Visualise
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
hp.mollview(cmb_map, title="Simulated CMB (K)", unit="K",
            hold=False, fig=fig.number, sub=(1, 2, 1))
ax = axes[1]
ax.plot(ell_rec[2:], ell_rec[2:] * (ell_rec[2:] + 1) * Cl_rec[2:] / (2 * np.pi),
        lw=1.5, label=r"Recovered $C_\ell$")
ax.set_xlabel(r"Multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$  [K$^2$]")
ax.set_title("Angular Power Spectrum")
ax.legend()
fig.tight_layout()
plt.savefig("cmb_power_spectrum.png", dpi=150)
print("Saved cmb_power_spectrum.png")
```

---

## Advanced Usage

### Cross-matching two source catalogs by sky position

```python
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
import numpy as np

# Suppose cat_a and cat_b are astropy Tables with RA/Dec columns
# Here we mock them with random positions for illustration
rng = np.random.default_rng(0)
n_a, n_b = 200, 5000

ra_a  = rng.uniform(202.0, 203.0, n_a)
dec_a = rng.uniform(46.8,  47.5,  n_a)
ra_b  = rng.uniform(201.5, 203.5, n_b)
dec_b = rng.uniform(46.5,  47.8,  n_b)

coords_a = SkyCoord(ra=ra_a * u.deg, dec=dec_a * u.deg, frame="icrs")
coords_b = SkyCoord(ra=ra_b * u.deg, dec=dec_b * u.deg, frame="icrs")

# For each source in A, find the nearest in B
idx, sep2d, _ = match_coordinates_sky(coords_a, coords_b)

# Keep only matches within 1 arcsec
match_mask    = sep2d < 1.0 * u.arcsec
n_match       = match_mask.sum()
print(f"Matched {n_match} / {n_a} sources within 1 arcsec")

matched_a_idx = np.where(match_mask)[0]
matched_b_idx = idx[match_mask]
separations   = sep2d[match_mask].to(u.arcsec)
print(f"Median separation : {np.median(separations):.3f}")
```

### Building a WCS from scratch and creating a synthetic FITS image

```python
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ny, nx = 512, 512
wcs = WCS(naxis=2)
wcs.wcs.ctype  = ["RA---TAN", "DEC--TAN"]
wcs.wcs.crpix  = [nx / 2 + 0.5, ny / 2 + 0.5]   # reference pixel (1-based)
wcs.wcs.crval  = [202.4696, 47.1952]              # RA, Dec of reference (deg)
wcs.wcs.cdelt  = [-0.000277778, 0.000277778]      # 1 arcsec/pixel
wcs.wcs.cunit  = ["deg", "deg"]

# Create a synthetic Gaussian source
rng   = np.random.default_rng(1)
image = rng.normal(100.0, 10.0, (ny, nx))          # sky background + noise
yy, xx = np.mgrid[:ny, :nx]
# Place a point source at pixel (256, 256) with peak 5000 counts
sigma = 2.5
image += 5000.0 * np.exp(-((xx - 256) ** 2 + (yy - 256) ** 2) / (2 * sigma ** 2))

# Write to FITS
header = wcs.to_header()
header["EXPTIME"] = (300.0, "Exposure time [s]")
header["FILTER"]  = ("r",    "Filter name")
hdu = fits.PrimaryHDU(data=image.astype(np.float32), header=header)
hdu.writeto("synthetic_field.fits", overwrite=True)
print("Saved synthetic_field.fits")
```

### Querying the Gaia archive with ADQL via astroquery

```python
from astroquery.gaia import Gaia

# Download Gaia DR3 stars brighter than G=15 around M51
query = """
SELECT TOP 200
    source_id, ra, dec, parallax, pmra, pmdec,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
FROM gaiadr3.gaia_source
WHERE CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', 202.4696, 47.1952, 0.1)
) = 1
AND phot_g_mean_mag < 15
ORDER BY phot_g_mean_mag ASC
"""

job    = Gaia.launch_job(query)
result = job.get_results()
print(f"Retrieved {len(result)} Gaia DR3 stars")
print(result["source_id", "ra", "dec", "phot_g_mean_mag"][:5])
```

### Reading a multi-extension FITS table (binary table HDU)

```python
from astropy.io import fits
from astropy.table import Table
import numpy as np

# Example: Chandra source catalogue (or any BinTable FITS)
# hdul = fits.open("chandra_sources.fits")
# For demonstration, create a mock BinTable
n = 100
rng = np.random.default_rng(2)
col_ra   = fits.Column(name="RA",  format="D", array=rng.uniform(0, 360, n))
col_dec  = fits.Column(name="DEC", format="D", array=rng.uniform(-90, 90, n))
col_flux = fits.Column(name="FLUX", format="E",
                       array=rng.exponential(1e-13, n).astype(np.float32))
col_name = fits.Column(name="NAME", format="20A",
                       array=[f"CXO J{i:04d}" for i in range(n)])

hdu_table = fits.BinTableHDU.from_columns([col_ra, col_dec, col_flux, col_name])
hdu_table.name = "SOURCES"
primary   = fits.PrimaryHDU()
hdul      = fits.HDUList([primary, hdu_table])
hdul.writeto("mock_sources.fits", overwrite=True)

# Read back as an astropy Table (cleanest interface)
tbl = Table.read("mock_sources.fits", hdu="SOURCES")
print(tbl[:5])
print(f"Brightest source: {tbl[np.argmax(tbl['FLUX'])]['NAME']}")
```

---

## Troubleshooting

### `astroquery` times out or returns empty results

- Check your internet connection. Most services require outbound HTTP on port 80/443.
- Increase the timeout: `Vizier.TIMEOUT = 60` before the query.
- SIMBAD and Vizier occasionally have scheduled maintenance; retry after a few minutes.
- For offline use, cache results to disk:

```python
from astropy.table import Table
import os

CACHE = "gaia_cache.fits"
if os.path.exists(CACHE):
    gaia = Table.read(CACHE)
else:
    # ... run Vizier query ...
    gaia.write(CACHE, overwrite=True)
```

### `WCS` warnings about non-standard keywords

```python
import warnings
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)
```

### `photutils` source detection finds too many / too few sources

Adjust the detection threshold. A typical starting point:

```python
from astropy.stats import sigma_clipped_stats
_, _, std = sigma_clipped_stats(data, sigma=3.0)
finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)   # raise threshold to reduce
finder = DAOStarFinder(fwhm=3.0, threshold=3.0 * std)   # lower to detect fainter
```

Also ensure the PSF FWHM parameter matches the actual image seeing in pixels.

### `healpy` crashes on import (macOS)

On Apple Silicon, install via conda-forge rather than pip:

```bash
conda install -c conda-forge healpy
```

### FITS file has wrong byte order (big-endian on little-endian machine)

```python
data = hdul[0].data
if data.dtype.byteorder not in ("=", "|"):
    data = data.byteswap().newbyteorder()
```

---

## External Resources

- Astropy documentation: https://docs.astropy.org
- Astroquery documentation: https://astroquery.readthedocs.io
- Healpy / HEALPix: https://healpy.readthedocs.io
- Photutils: https://photutils.readthedocs.io
- FITS standard (NASA): https://fits.gsfc.nasa.gov/fits_standard.html
- Vizier catalogue service: https://vizier.cds.unistra.fr
- SIMBAD database: https://simbad.cds.unistra.fr
- Gaia archive ADQL: https://gea.esac.esa.int/archive/

---

## Examples

### Example 1 — Full photometry pipeline on a real SDSS image

```python
"""
End-to-end aperture photometry pipeline:
  1. Download SDSS r-band image of M51 via SkyView
  2. Read FITS, solve WCS
  3. Detect sources with DAOStarFinder
  4. Perform aperture photometry
  5. Cross-match detected sources against Gaia DR3
  6. Compute instrumental zero-point from Gaia G magnitudes
  7. Save a calibrated catalog to FITS
"""

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astroquery.skyview import SkyView
from astroquery.vizier import Vizier
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# --- 1. Download image ---
print("Downloading SDSS r-band image …")
imgs = SkyView.get_images("M51", survey=["SDSSr"], pixels=512)
hdul = imgs[0]
data   = hdul[0].data.astype(float)
header = hdul[0].header
wcs    = WCS(header)
exptime = float(header.get("EXPTIME", 53.91))    # SDSS typical 53.91 s
hdul.writeto("m51_sdssr.fits", overwrite=True)

# --- 2. Background subtraction ---
_, bkg_med, bkg_std = sigma_clipped_stats(data, sigma=3.0)
data_sub = data - bkg_med

# --- 3. Source detection ---
finder  = DAOStarFinder(fwhm=3.5, threshold=5.0 * bkg_std)
sources = finder(data_sub)
print(f"Detected {len(sources)} sources")

# --- 4. Aperture photometry ---
positions = np.column_stack([sources["xcentroid"], sources["ycentroid"]])
aper      = CircularAperture(positions, r=5.0)
ann       = CircularAnnulus(positions,  r_in=7.0, r_out=10.0)
phot      = aperture_photometry(data_sub, [aper, ann])

sky_bkg   = (phot["aperture_sum_1"] / ann.area) * aper.area
flux      = phot["aperture_sum_0"] - sky_bkg
flux_rate = flux / exptime

# --- 5. WCS -> sky coordinates for each source ---
sky_pos = wcs.pixel_to_world(sources["xcentroid"], sources["ycentroid"])
sources["ra"]  = sky_pos.ra.deg
sources["dec"] = sky_pos.dec.deg

# --- 6. Cross-match against Gaia DR3 ---
field_centre = wcs.pixel_to_world(data.shape[1] / 2, data.shape[0] / 2)
v = Vizier(columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag"], row_limit=2000)
gaia_result = v.query_region(field_centre, radius=5 * u.arcmin,
                              catalog="I/355/gaiadr3")
gaia = gaia_result[0] if gaia_result else None

zp = 25.0   # fallback zero-point
if gaia is not None:
    det_coords  = SkyCoord(ra=sources["ra"] * u.deg, dec=sources["dec"] * u.deg)
    gaia_coords = SkyCoord(ra=gaia["RA_ICRS"] * u.deg, dec=gaia["DE_ICRS"] * u.deg)
    idx, sep, _ = det_coords.match_to_catalog_sky(gaia_coords)
    good        = sep < 1.0 * u.arcsec
    if good.sum() > 5:
        inst_mag = -2.5 * np.log10(np.abs(flux_rate[good]))
        gaia_mag = gaia["Gmag"][idx[good]].data.astype(float)
        zp       = float(np.median(gaia_mag - inst_mag))
        print(f"Photometric zero-point (Gaia G): {zp:.3f} mag")

# --- 7. Build calibrated catalog ---
mag_cal = zp - 2.5 * np.log10(np.abs(flux_rate))
catalog = Table()
catalog["id"]    = sources["id"]
catalog["ra"]    = sources["ra"]
catalog["dec"]   = sources["dec"]
catalog["flux"]  = flux_rate
catalog["mag"]   = mag_cal
catalog.sort("mag")
catalog.write("m51_catalog.fits", overwrite=True)
print("Saved m51_catalog.fits with", len(catalog), "sources")
print(catalog["id", "ra", "dec", "mag"][:10])
```

### Example 2 — HEALPix CMB map analysis with masking and beam correction

```python
"""
CMB power spectrum analysis:
  1. Simulate a CMB temperature map with realistic Planck beam
  2. Apply a Galactic mask (|b| < 20 deg)
  3. Compute pseudo-Cl power spectrum
  4. Correct for the beam and pixel window functions
  5. Plot the final power spectrum
"""

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# --- Parameters ---
nside    = 256
lmax     = 2 * nside
fwhm_deg = 0.5               # Planck 143 GHz beam ~ 7 arcmin; use 30 arcmin here
fwhm_rad = np.radians(fwhm_deg)
npix     = hp.nside2npix(nside)
ells     = np.arange(lmax + 1)

# --- 1. Build input power spectrum (Planck 2018-like toy model) ---
Cl_in         = np.zeros(lmax + 1)
Cl_in[2:]     = 2.7e-9 / (ells[2:] * (ells[2:] + 1) / (2 * np.pi)) * (
    1 + 5 * np.exp(-((ells[2:] - 200) / 50) ** 2)
    + 2 * np.exp(-((ells[2:] - 540) / 40) ** 2)
)

# --- 2. Simulate map with beam ---
np.random.seed(7)
beam_window = hp.gauss_beam(fwhm_rad, lmax=lmax)
alm_true    = hp.synalm(Cl_in, lmax=lmax)
alm_beam    = hp.almxfl(alm_true, beam_window)
cmb_map     = hp.alm2map(alm_beam, nside=nside, verbose=False)

# --- 3. Build Galactic mask (cut |b| < 20 deg) ---
theta, phi = hp.pix2ang(nside, np.arange(npix))
b_rad      = np.pi / 2 - theta           # colatitude -> Galactic latitude proxy
mask       = np.abs(b_rad) > np.radians(20)
fsky       = mask.mean()
print(f"Sky fraction after masking: f_sky = {fsky:.3f}")

masked_map = cmb_map * mask

# --- 4. Pseudo-Cl estimation ---
alm_masked  = hp.map2alm(masked_map, lmax=lmax)
Cl_pseudo   = hp.alm2cl(alm_masked)

# --- 5. Beam and pixel window correction ---
pixel_win   = hp.pixwin(nside)[:lmax + 1]
correction  = (beam_window ** 2) * (pixel_win ** 2) * fsky
correction  = np.where(correction > 1e-30, correction, 1e-30)
Cl_corr     = Cl_pseudo / correction

# --- 6. Plot ---
fig, ax = plt.subplots(figsize=(10, 5))
norm = ells * (ells + 1) / (2 * np.pi)
ax.plot(ells[2:], norm[2:] * Cl_in[2:],    lw=2,  label="Input $C_\\ell$",    color="k")
ax.plot(ells[2:], norm[2:] * Cl_corr[2:],  lw=1.5, label="Recovered (beam-corrected)",
        color="tab:blue", alpha=0.8)
ax.set_xlabel(r"Multipole $\ell$")
ax.set_ylabel(r"$\ell(\ell+1)C_\ell / 2\pi$  [$\mu$K$^2$]")
ax.set_xlim(2, lmax)
ax.set_yscale("log")
ax.legend()
ax.set_title(f"CMB Power Spectrum (nside={nside}, mask f_sky={fsky:.2f})")
fig.tight_layout()
plt.savefig("cmb_analysis.png", dpi=150)
print("Saved cmb_analysis.png")
```
