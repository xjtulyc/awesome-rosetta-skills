---
name: nmr-analysis
description: Process and analyze NMR spectra with nmrglue — FID processing, peak picking, chemical shift referencing, and J-coupling extraction.
tags:
  - nmr
  - spectroscopy
  - nmrglue
  - peak-picking
  - chemical-shift
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
  - nmrglue>=0.9
  - matplotlib>=3.7
  - scipy>=1.11
  - numpy>=1.24
  - pandas>=2.0
last_updated: "2026-03-17"
status: stable
---

# NMR Spectrum Analysis

Process raw NMR free-induction decay (FID) data through apodization, zero-filling,
Fourier transformation, and phase correction to obtain interpretable 1D and 2D
spectra. Automate peak picking, chemical shift referencing, and J-coupling constant
extraction using nmrglue and scipy.

---

## When to Use This Skill

- You have raw NMR data in Bruker, Varian/Agilent, or JCAMP-DX format and need to
  convert it to a frequency-domain spectrum.
- You want to apply **apodization** (window functions) and **zero-filling** before
  Fourier transformation to improve resolution or sensitivity.
- You need automated **peak picking** on 1D or 2D NMR spectra and want results as a
  pandas DataFrame.
- You are performing **chemical shift referencing** against an internal standard
  (TMS, DSS, TSP) or a known solvent peak.
- You need to extract **J-coupling constants** from well-resolved multiplets using
  line shape fitting.
- You want to overlay, compare, or subtract spectra from different samples or
  experiments (titration, temperature series).

---

## Background & Key Concepts

### Free Induction Decay (FID)

The FID is the time-domain NMR signal recorded after a radiofrequency pulse. It is a
sum of exponentially decaying sinusoids; each frequency corresponds to a resonance.
The complex FID is Fourier-transformed to yield the frequency-domain spectrum.

### Apodization (Window Functions)

Before Fourier transformation, a window function is multiplied with the FID to:
- **Improve sensitivity** (exponential multiplication — line broadening).
- **Improve resolution** (Lorentz-to-Gauss transformation — line narrowing).
- **Reduce truncation artifacts** (cosine/sine bells for 2D data).

Common window functions:

| Function | Effect | Use Case |
|----------|--------|----------|
| Exponential (LB) | Sensitivity | 1D 13C, 15N |
| Gaussian (GM) | Resolution | 1D 1H |
| Cosine bell | Balanced | 2D indirect dimension |
| Sine bell | Resolution | 2D direct dimension |

### Zero-Filling

Appending zeros to the FID before Fourier transformation increases the number of
spectral points and therefore digital resolution. Zero-filling by a factor of 2 is
standard; zero-filling by 4–8 is used for high-resolution work.

### Phase Correction

The spectrum has real (absorption) and imaginary (dispersion) components. Phase
correction (zeroth-order P0 and first-order P1) ensures all peaks have pure
absorption lineshapes, which are symmetric and integrable.

### Chemical Shift Referencing

Chemical shifts in ppm are calculated relative to a reference compound:
- **1H / 13C**: TMS (tetramethylsilane) at 0.00 ppm
- **Aqueous solutions**: DSS or TSP at 0.00 ppm
- **31P**: H3PO4 at 0.00 ppm

### J-Coupling Constants

Spin-spin coupling splits resonances into multiplets. The coupling constant J (in Hz)
equals the frequency separation between adjacent lines of a first-order multiplet.
Accurate J values require fitting each line of the multiplet with a Lorentzian.

---

## Environment Setup

### Installation

```bash
# Conda environment (recommended)
conda create -n nmr-env python=3.11 -y
conda activate nmr-env

pip install "nmrglue>=0.9" "matplotlib>=3.7" "scipy>=1.11" "numpy>=1.24" "pandas>=2.0"
```

### Verify Installation

```python
import nmrglue as ng
print(ng.__version__)   # e.g. 0.9

import scipy, numpy, pandas, matplotlib
print("All dependencies imported successfully.")
```

### Data Formats Supported by nmrglue

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| Bruker | yes | yes | `fid`, `ser`, `1r`, `2rr` |
| Varian/Agilent | yes | yes | `fid` directory |
| NMRPipe | yes | yes | `.fid`, `.ft2` |
| JCAMP-DX | yes | no | `.jdx`, `.dx` |
| Sparky (UCSF) | yes | yes | `.ucsf` |
| NMRView | yes | yes | `.nv` |

---

## Core Workflow

### Step 1 — Load Raw FID and Inspect Parameters

```python
import nmrglue as ng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Bruker data directory ---
DATA_DIR = "/path/to/bruker/experiment"   # contains 'fid' and 'acqus' files

dic, data = ng.bruker.read(DATA_DIR)

# Print key acquisition parameters
acqus = dic["acqus"]
print(f"Spectrometer frequency (SF)  : {acqus['SFO1']:.4f} MHz")
print(f"Spectral width (SW)          : {acqus['SW']:.2f} ppm")
print(f"Number of points (TD)        : {acqus['TD']}")
print(f"Relaxation delay (D1)        : {acqus.get('D', [None]*2)[1]} s")
print(f"Number of scans (NS)         : {acqus['NS']}")
print(f"Solvent                      : {acqus.get('SOLVENT', 'N/A')}")
print(f"FID shape                    : {data.shape}")
print(f"FID dtype                    : {data.dtype}")

# Plot raw FID (real part)
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(data.real[:2048], lw=0.8, color="steelblue")
ax.set_xlabel("Point Index", fontsize=12)
ax.set_ylabel("Intensity (AU)", fontsize=12)
ax.set_title("Raw FID (first 2048 points)", fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("raw_fid.png", dpi=150)
plt.show()
```

### Step 2 — FID Processing (Apodization, Zero-Filling, FT, Phase Correction)

```python
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/path/to/bruker/experiment"
dic, data = ng.bruker.read(DATA_DIR)

# Remove digital filter group delay (Bruker-specific)
data = ng.bruker.remove_digital_filter(dic, data)

# ----- Apodization -----
# Lorentz-to-Gauss transformation: lb < 0 (line narrowing), gb > 0 (Gaussian)
data_apod = ng.proc_base.gm(data, g1=0.1, g2=0.1, g3=0.0)   # Gaussian multiply
# Alternative — exponential multiplication (sensitivity mode):
# data_apod = ng.proc_base.em(data, lb=0.3)   # lb in Hz

# ----- Zero-filling -----
# Double the number of points
n_td = data.shape[-1]
data_zf = ng.proc_base.zf_size(data_apod, n_td * 2)

# ----- Fourier Transform -----
data_ft = ng.proc_base.fft(data_zf)

# ----- Phase Correction (manual values) -----
# Determine P0, P1 empirically or from instrument file
P0 = 0.0    # zeroth-order phase (degrees)
P1 = 0.0    # first-order phase (degrees)
data_ph = ng.proc_base.ps(data_ft, p0=P0, p1=P1)

# ----- Reverse spectrum (if needed) -----
data_final = ng.proc_base.rev(data_ph)

print(f"Processed spectrum shape: {data_final.shape}")

# Build ppm axis using udic
udic = ng.bruker.guess_udic(dic, data_final)
uc = ng.fileiobase.uc_from_udic(udic)
ppm_axis = uc.ppm_scale()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ppm_axis, data_final.real, lw=0.8, color="navy")
ax.invert_xaxis()
ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("Processed 1H NMR Spectrum", fontsize=13)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("spectrum_1d.png", dpi=150)
plt.show()
```

### Step 3 — Automated Peak Picking

```python
import nmrglue as ng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Assume data_final and ppm_axis from Step 2 are available

spectrum = data_final.real

# Noise estimation (MAD of baseline region outside peaks)
baseline_mask = (ppm_axis < 0.5) | (ppm_axis > 11.0)  # adjust for your nucleus
if baseline_mask.sum() > 10:
    noise = np.median(np.abs(spectrum[baseline_mask]))
else:
    noise = np.std(spectrum[:100])  # fallback

threshold = 5.0 * noise   # 5-sigma threshold

# scipy peak finding
peaks_idx, properties = find_peaks(
    spectrum,
    height=threshold,
    distance=5,        # minimum separation in points
    prominence=noise,
)

peak_ppms = ppm_axis[peaks_idx]
peak_heights = spectrum[peaks_idx]

df_peaks = pd.DataFrame({
    "Chemical Shift (ppm)": peak_ppms,
    "Intensity": peak_heights,
    "Point Index": peaks_idx,
}).sort_values("Chemical Shift (ppm)", ascending=False).reset_index(drop=True)

print(f"Found {len(df_peaks)} peaks above threshold ({threshold:.2e}):")
print(df_peaks[["Chemical Shift (ppm)", "Intensity"]].to_string(index=True))

# Plot with peak annotations
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ppm_axis, spectrum, lw=0.7, color="navy", label="Spectrum")
ax.scatter(peak_ppms, peak_heights, color="red", s=20, zorder=5, label=f"Peaks (n={len(peaks_idx)})")
ax.axhline(threshold, color="gray", ls="--", lw=0.8, label="Threshold")
for ppm, height in zip(peak_ppms, peak_heights):
    ax.annotate(f"{ppm:.2f}", xy=(ppm, height), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7, color="red")
ax.invert_xaxis()
ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
ax.set_ylabel("Intensity", fontsize=12)
ax.set_title("1H NMR Spectrum with Peak Picks", fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("peak_picks.png", dpi=150)
plt.show()

df_peaks.to_csv("peak_list.csv", index=False)
print("Peak list saved to peak_list.csv")
```

### Step 4 — Chemical Shift Referencing

```python
import nmrglue as ng
import numpy as np
import pandas as pd

# Reference compound ppm value and expected ppm position
REF_COMPOUND = "TMS"
REF_EXPECTED_PPM = 0.00       # target ppm
REF_OBSERVED_PPM = 0.03       # observed from peak list (adjust from df_peaks above)

# Compute the required shift correction
ppm_correction = REF_EXPECTED_PPM - REF_OBSERVED_PPM
print(f"Referencing against {REF_COMPOUND}: shift correction = {ppm_correction:.4f} ppm")

# Apply correction to ppm axis
ppm_axis_ref = ppm_axis + ppm_correction

# Apply correction to peak list
df_peaks["Chemical Shift Referenced (ppm)"] = (
    df_peaks["Chemical Shift (ppm)"] + ppm_correction
)

print("\nReferenced peak list:")
print(df_peaks[["Chemical Shift Referenced (ppm)", "Intensity"]].to_string(index=True))

# Verify reference peak is now at 0.00 ppm
ref_peak_corrected = REF_OBSERVED_PPM + ppm_correction
print(f"\nReference peak after correction: {ref_peak_corrected:.4f} ppm (expected 0.000)")

df_peaks.to_csv("peak_list_referenced.csv", index=False)
```

### Step 5 — J-Coupling Extraction via Lorentzian Fitting

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def lorentzian(x, x0, gamma, amplitude, baseline):
    """Single Lorentzian peak."""
    return baseline + amplitude * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)

def multi_lorentzian(x, *params):
    """Sum of Lorentzians: params in groups of 3 (x0, gamma, amplitude) + baseline."""
    n = (len(params) - 1) // 3
    baseline = params[-1]
    y = np.full_like(x, baseline, dtype=float)
    for i in range(n):
        x0, gamma, amp = params[3*i], params[3*i+1], params[3*i+2]
        y += amp * (gamma / 2)**2 / ((x - x0)**2 + (gamma / 2)**2)
    return y

# Select a spectral region containing the multiplet of interest
# Adjust these ppm limits to isolate your multiplet
PPM_LOW  = 3.40   # lower ppm bound (right on inverted axis)
PPM_HIGH = 3.60   # upper ppm bound

mask = (ppm_axis_ref >= PPM_LOW) & (ppm_axis_ref <= PPM_HIGH)
x_region = ppm_axis_ref[mask]
y_region = spectrum[mask]

# Pre-pick sub-peaks in this region
sub_peaks_idx, _ = find_peaks(y_region, height=0.1 * y_region.max(), distance=2)
print(f"Sub-peaks found in [{PPM_LOW}, {PPM_HIGH}] ppm: {len(sub_peaks_idx)}")

# Build initial parameter guess
p0 = []
for idx in sub_peaks_idx:
    p0 += [x_region[idx], 0.003, y_region[idx]]   # x0, gamma (ppm), amplitude
p0.append(0.0)   # baseline

try:
    popt, pcov = curve_fit(multi_lorentzian, x_region, y_region, p0=p0,
                           maxfev=10000, method="trf")
    perr = np.sqrt(np.diag(pcov))
    n_peaks = (len(popt) - 1) // 3

    # Extract line positions and widths
    line_positions = [popt[3*i] for i in range(n_peaks)]
    line_widths_ppm = [abs(popt[3*i+1]) for i in range(n_peaks)]

    # J-coupling in Hz: separation in ppm × spectrometer frequency in MHz
    SF = 600.0   # MHz — replace with acqus['SFO1']
    line_positions_hz = np.array(line_positions) * SF

    print("\nLine positions and J-couplings:")
    for i in range(len(line_positions) - 1):
        j_hz = abs(line_positions_hz[i] - line_positions_hz[i+1])
        print(f"  J({i},{i+1}) = {j_hz:.2f} Hz  ({line_positions[i]:.4f} — {line_positions[i+1]:.4f} ppm)")

    # Plot fit
    x_dense = np.linspace(x_region.min(), x_region.max(), 500)
    y_fit = multi_lorentzian(x_dense, *popt)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_region, y_region, "k-", lw=1.5, label="Data")
    ax.plot(x_dense, y_fit, "r--", lw=1.5, label="Lorentzian fit")
    for pos in line_positions:
        ax.axvline(pos, color="gray", lw=0.5, ls=":")
    ax.invert_xaxis()
    ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title(f"Multiplet Fit: {PPM_LOW}–{PPM_HIGH} ppm", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("multiplet_fit.png", dpi=150)
    plt.show()

except RuntimeError as e:
    print(f"Fitting failed: {e}")
    print("Try adjusting PPM_LOW/PPM_HIGH or initial parameters.")
```

---

## Advanced Usage

### 2D HSQC Processing (Bruker)

```python
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR_2D = "/path/to/bruker/hsqc"   # contains 'ser' file

dic2d, data2d = ng.bruker.read(DATA_DIR_2D)
data2d = ng.bruker.remove_digital_filter(dic2d, data2d)

# Apodize in direct (F2, 1H) dimension
data2d = ng.proc_base.sp(data2d, off=0.5, end=0.95, pow=1, dim=1)  # sine bell

# Apodize in indirect (F1, 13C or 15N) dimension
data2d = ng.proc_base.sp(data2d, off=0.5, end=0.95, pow=1, dim=0)

# Zero-fill both dimensions
data2d = ng.proc_base.zf_size(data2d, data2d.shape[1] * 2, dim=1)  # F2
data2d = ng.proc_base.zf_size(data2d, data2d.shape[0] * 2, dim=0)  # F1

# Fourier Transform
data2d = ng.proc_base.fft(data2d, dim=1)
data2d = ng.proc_base.fft(data2d, dim=0)

# Phase correction (adjust P0/P1 values for your data)
data2d = ng.proc_base.ps(data2d, p0=0.0, p1=0.0, dim=1)
data2d = ng.proc_base.ps(data2d, p0=0.0, p1=0.0, dim=0)

data2d_real = data2d.real

# Build ppm axes
udic2d = ng.bruker.guess_udic(dic2d, data2d_real)
uc_f2 = ng.fileiobase.uc_from_udic(udic2d, dim=1)  # 1H
uc_f1 = ng.fileiobase.uc_from_udic(udic2d, dim=0)  # 13C or 15N
ppm_f2 = uc_f2.ppm_scale()
ppm_f1 = uc_f1.ppm_scale()

# Contour plot
noise_level = np.std(data2d_real[:10, :10])
contour_levels = noise_level * np.array([4, 8, 16, 32, 64, 128])

fig, ax = plt.subplots(figsize=(8, 8))
ax.contour(ppm_f2, ppm_f1, data2d_real,
           levels=contour_levels, colors=["navy"], linewidths=0.6)
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel("1H (ppm)", fontsize=12)
ax.set_ylabel("13C / 15N (ppm)", fontsize=12)
ax.set_title("2D HSQC Spectrum", fontsize=13)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("hsqc_2d.png", dpi=150)
plt.show()
```

### Integration and Relative Quantification

```python
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# Integrate spectral regions (ppm intervals)
regions = {
    "CH3 (TMS)"    : (0.0,  0.1),
    "Aliphatic CH3": (0.8,  1.0),
    "Aliphatic CH2": (1.2,  1.6),
    "CH alpha"     : (3.4,  3.7),
    "Aromatic"     : (7.0,  8.0),
}

integrals = {}
for label, (lo, hi) in regions.items():
    # ppm_axis_ref decreases, so lo and hi may need swapping depending on direction
    mask = (ppm_axis_ref >= lo) & (ppm_axis_ref <= hi)
    integrals[label] = trapezoid(spectrum[mask], ppm_axis_ref[mask])

# Normalize to reference region (e.g., TMS 9H)
ref_label  = "CH3 (TMS)"
ref_protons = 9
if integrals[ref_label] != 0:
    scale = ref_protons / abs(integrals[ref_label])
    normalized = {k: abs(v) * scale for k, v in integrals.items()}
else:
    normalized = integrals

df_integrals = pd.DataFrame({
    "Region"            : list(regions.keys()),
    "ppm Low"           : [v[0] for v in regions.values()],
    "ppm High"          : [v[1] for v in regions.values()],
    "Raw Integral"      : list(integrals.values()),
    "Normalized (Hcount)": list(normalized.values()),
})

print(df_integrals.to_string(index=False))
df_integrals.to_csv("integrations.csv", index=False)
```

### Spectral Comparison / Overlay

```python
import numpy as np
import matplotlib.pyplot as plt

# Suppose you have two spectra from Step 2, stored as arrays
# spectrum_A, spectrum_B and ppm_axis_A, ppm_axis_B

# Scale both spectra to unit max intensity for visual comparison
def normalize_spectrum(s):
    s_norm = s - s.min()
    return s_norm / s_norm.max()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ppm_axis_ref, normalize_spectrum(spectrum), lw=0.9,
        color="steelblue", label="Sample A", alpha=0.85)
# ax.plot(ppm_axis_ref_B, normalize_spectrum(spectrum_B), lw=0.9,
#         color="coral", label="Sample B", alpha=0.85)
ax.invert_xaxis()
ax.set_xlabel("Chemical Shift (ppm)", fontsize=12)
ax.set_ylabel("Normalized Intensity", fontsize=12)
ax.set_title("Spectral Overlay Comparison", fontsize=13)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("spectral_overlay.png", dpi=150)
plt.show()
```

### NMRPipe Format Export

```python
import nmrglue as ng
import numpy as np

# Convert processed spectrum to NMRPipe format for further analysis
udic_pipe = ng.bruker.guess_udic(dic, data_final)

# Write NMRPipe file
ng.pipe.write("spectrum.ft2", ng.pipe.create_dic(udic_pipe), data_final.real)
print("Exported spectrum.ft2 in NMRPipe format")
```

---

## Troubleshooting

### FileNotFoundError: acqus or fid not found

```bash
# Ensure you point to the numbered experiment directory, not the sample directory
ls /path/to/bruker/sample/1/   # should contain: fid, acqus, procs/
```

```python
# Correct path usage
dic, data = ng.bruker.read("/path/to/bruker/sample/1")   # experiment directory
```

### Spectrum Appears Inverted or Backwards

```python
# Reverse the spectrum if frequency axis runs backwards
data_final = ng.proc_base.rev(data_final)
# Or flip the ppm axis:
ppm_axis = ppm_axis[::-1]
```

### Phase Correction Looks Wrong

Use an automated first-order phase correction algorithm:

```python
# Automatic phase correction (entropy minimization)
data_ph, p0_auto, p1_auto = ng.proc_autophase.autops(
    data_ft, "acme", p0=0.0, p1=0.0
)
print(f"Auto P0={p0_auto:.2f}°, P1={p1_auto:.2f}°")
```

### Peak Picker Finds Too Many or Too Few Peaks

Adjust the noise threshold multiplier:

```python
# More conservative (fewer false positives)
threshold = 10.0 * noise

# More permissive (detects weak peaks)
threshold = 3.0 * noise

# Increase minimum inter-peak separation to avoid splitting broad peaks
peaks_idx, _ = find_peaks(spectrum, height=threshold, distance=15)
```

### Bruker Digital Filter Removal Fails

```python
# Fallback: skip digital filter removal and trim manually
data_trimmed = data[64:]   # discard first 64 points (typical for 500/600 MHz)
```

### 2D Spectrum Has Sinc Wiggles

Apply stronger apodization before FT:

```python
# Stronger cosine bell for indirect dimension
data2d = ng.proc_base.sp(data2d, off=0.45, end=0.98, pow=2, dim=0)
```

---

## External Resources

- nmrglue Documentation: https://nmrglue.readthedocs.io
- nmrglue GitHub: https://github.com/jjhelmus/nmrglue
- Bruker TopSpin Reference: https://www.bruker.com/en/products-and-solutions/mr/nmr-software/topspin.html
- NMRPipe Manual: https://www.ibbr.umd.edu/nmrpipe/
- CCPNMR Analysis: https://www.ccpn.ac.uk/ccpnmr-analysis/
- BMRB (Biological Magnetic Resonance Bank): https://bmrb.io
- NMRshiftDB2 Reference Database: https://nmrshiftdb.nmr.uni-koeln.de

---

## Examples

### Example 1 — Complete 1D 1H Processing Pipeline with Synthetic FID

```python
"""
End-to-end 1H NMR processing pipeline using a synthetic FID.
No external data file required — demonstrates the full workflow.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---- Generate a synthetic FID ----
SF       = 600.0          # spectrometer frequency in MHz
SW_HZ    = 12.0 * SF      # spectral width in Hz (12 ppm × 600 MHz = 7200 Hz)
TD       = 32768          # number of complex points
DT       = 1.0 / SW_HZ   # dwell time in seconds
T2_VALS  = [0.8, 0.3, 0.5, 0.6, 0.4]   # T2 relaxation times in s

# Chemical shifts (ppm) and their multiplicities
PEAKS = [
    (7.26, 1.0, 0.5),   # CHCl3 singlet
    (3.55, 2.0, 0.3),   # OCH2 singlet
    (2.10, 3.0, 0.8),   # COCH3 singlet
    (1.25, 6.0, 0.6),   # (CH3)2 doublet approximated as singlet
    (0.90, 3.0, 0.4),   # CH3 triplet center
]

t = np.arange(TD) * DT
fid = np.zeros(TD, dtype=complex)
np.random.seed(42)
noise_amplitude = 0.05

for ppm, amplitude, t2 in PEAKS:
    freq_hz = ppm * SF    # chemical shift in Hz
    fid += amplitude * np.exp(2j * np.pi * freq_hz * t) * np.exp(-t / t2)

fid += noise_amplitude * (np.random.randn(TD) + 1j * np.random.randn(TD))

# ---- Processing ----
# Exponential apodization (lb = 1 Hz)
lb = 1.0   # Hz
window = np.exp(-np.pi * lb * t)
fid_apod = fid * window

# Zero-fill to 2× TD
fid_zf = np.concatenate([fid_apod, np.zeros(TD, dtype=complex)])

# FFT and shift
spectrum_raw = np.fft.fft(fid_zf)
spectrum_raw = np.fft.fftshift(spectrum_raw)

# Build ppm axis
n_pts = len(spectrum_raw)
freq_axis_hz = np.linspace(-SW_HZ / 2, SW_HZ / 2, n_pts)
ppm_axis_synth = freq_axis_hz / SF

spectrum_proc = spectrum_raw.real

# ---- Peak Picking ----
noise_est = np.std(spectrum_proc[:200])
thr = 5.0 * noise_est
pk_idx, _ = find_peaks(spectrum_proc, height=thr, distance=20)
pk_ppms = ppm_axis_synth[pk_idx]
pk_ints = spectrum_proc[pk_idx]

df_peaks_synth = pd.DataFrame({
    "Chemical Shift (ppm)": pk_ppms,
    "Intensity": pk_ints,
}).sort_values("Chemical Shift (ppm)", ascending=False).reset_index(drop=True)

print("Picked peaks:")
print(df_peaks_synth.to_string(index=True))

# ---- Plot ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top: FID
axes[0].plot(t[:1024] * 1000, fid[:1024].real, lw=0.8, color="navy")
axes[0].set_xlabel("Time (ms)", fontsize=11)
axes[0].set_ylabel("Intensity", fontsize=11)
axes[0].set_title("Synthetic FID (first 1024 points)", fontsize=12)
axes[0].grid(alpha=0.3)

# Bottom: Spectrum
axes[1].plot(ppm_axis_synth, spectrum_proc, lw=0.7, color="darkgreen", label="Spectrum")
axes[1].scatter(pk_ppms, pk_ints, color="red", s=30, zorder=5,
                label=f"Peaks (n={len(pk_idx)})")
axes[1].invert_xaxis()
axes[1].set_xlabel("Chemical Shift (ppm)", fontsize=11)
axes[1].set_ylabel("Intensity", fontsize=11)
axes[1].set_title("Processed 1H NMR Spectrum (Synthetic)", fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("synthetic_nmr.png", dpi=150)
plt.show()
print("Saved synthetic_nmr.png")
```

### Example 2 — Bruker Data Full Pipeline with Export

```python
"""
Full pipeline for a real Bruker 1H experiment:
load → process → reference → peak pick → integrate → export.
Requires a Bruker experiment directory.
"""
import nmrglue as ng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

DATA_DIR = "/path/to/bruker/1"   # replace with your experiment directory

# 1. Load
dic, data = ng.bruker.read(DATA_DIR)
data = ng.bruker.remove_digital_filter(dic, data)

# 2. Apodize (Lorentz-to-Gauss)
data = ng.proc_base.gm(data, g1=0.2, g2=0.05)

# 3. Zero-fill to next power of 2
td = data.shape[-1]
next_pow2 = int(2 ** np.ceil(np.log2(td * 2)))
data = ng.proc_base.zf_size(data, next_pow2)

# 4. FFT
data = ng.proc_base.fft(data)

# 5. Phase (adjust p0/p1 for your spectrum)
data = ng.proc_base.ps(data, p0=0.0, p1=0.0)
data = ng.proc_base.rev(data)

spectrum = data.real

# 6. ppm axis
udic = ng.bruker.guess_udic(dic, data)
uc = ng.fileiobase.uc_from_udic(udic)
ppm = uc.ppm_scale()

# 7. Chemical shift referencing (TMS / solvent residual)
REF_PPM_OBSERVED = 7.26   # CDCl3 residual 1H — change to your reference
REF_PPM_TARGET   = 7.26
ppm_ref = ppm + (REF_PPM_TARGET - REF_PPM_OBSERVED)

# 8. Peak picking
noise = np.std(spectrum[ppm_ref > 11.0]) if (ppm_ref > 11.0).any() else np.std(spectrum[:100])
pk_idx, _ = find_peaks(spectrum, height=5*noise, distance=5)

df_pk = pd.DataFrame({
    "ppm": ppm_ref[pk_idx],
    "Intensity": spectrum[pk_idx],
}).sort_values("ppm", ascending=False)

print(f"Found {len(df_pk)} peaks")
print(df_pk.to_string(index=False))

# 9. Integration of user-defined regions
regions = {
    "Aromatic"  : (6.5, 8.5),
    "Vinyl"     : (4.5, 6.5),
    "Aliphatic" : (0.5, 3.5),
}
integrals = {}
for name, (lo, hi) in regions.items():
    mask = (ppm_ref >= lo) & (ppm_ref <= hi)
    integrals[name] = abs(trapezoid(spectrum[mask], ppm_ref[mask]))

df_int = pd.DataFrame(list(integrals.items()), columns=["Region", "Integral"])
print("\nIntegrals:")
print(df_int.to_string(index=False))

# 10. Export
df_pk.to_csv("bruker_peaks.csv", index=False)
df_int.to_csv("bruker_integrals.csv", index=False)
np.savetxt("spectrum_processed.txt",
           np.column_stack([ppm_ref, spectrum]),
           header="ppm\tintensity", delimiter="\t")

# 11. Publication-quality plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ppm_ref, spectrum, lw=0.7, color="black")
ax.scatter(df_pk["ppm"], df_pk["Intensity"], color="red", s=15, zorder=5)
for _, row in df_pk.iterrows():
    ax.annotate(f"{row['ppm']:.2f}", xy=(row["ppm"], row["Intensity"]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=7, color="darkred")
ax.invert_xaxis()
ax.set_xlim(ppm_ref.max() + 0.5, ppm_ref.min() - 0.5)
ax.set_xlabel("δ (ppm)", fontsize=13)
ax.set_ylabel("Intensity (AU)", fontsize=13)
ax.set_title(f"1H NMR Spectrum — {DATA_DIR}", fontsize=12)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("nmr_final.png", dpi=200)
plt.show()
print("Pipeline complete. Files saved: bruker_peaks.csv, bruker_integrals.csv, nmr_final.png")
```
