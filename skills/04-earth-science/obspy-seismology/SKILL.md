---
name: obspy-seismology
description: >
  Use this Skill for seismological analysis with ObsPy: waveform download,
  filtering, P/S phase picking, moment magnitude, and spectral analysis.
tags:
  - earth-science
  - seismology
  - obspy
  - waveform
  - earthquake
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
    - obspy>=1.4
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# Seismological Analysis with ObsPy

> **One-line summary**: Download, process, and analyze seismic waveforms with ObsPy: bandpass filtering, P/S phase picking, spectral analysis, focal mechanisms, and moment magnitude estimation.

---

## When to Use This Skill

- When downloading seismic waveforms from IRIS/FDSN data centers
- When filtering and deconvolving instrument response from seismograms
- When picking P and S wave arrivals automatically or manually
- When computing seismic spectra and corner frequencies
- When estimating moment magnitude from seismic records
- When computing traveltimes using 1D Earth models (iasp91, PREM)

**Trigger keywords**: ObsPy, seismology, seismogram, waveform, earthquake, P-wave, S-wave, FDSN, IRIS, bandpass filter, instrument response, moment magnitude, seismic, travel time

---

## Background & Key Concepts

### Seismic Wave Types

- **P-waves**: Compressional (primary) waves, fastest, arrive first
- **S-waves**: Shear waves, ~60% of P velocity, arrive second
- **Surface waves**: Rayleigh and Love waves, slowest, largest amplitude

### Instrument Response Removal

Raw data in counts → ground motion (m, m/s, or m/s²) by deconvolving the instrument response:

$$
U(\omega) = \frac{X(\omega)}{I(\omega)}
$$

where $X(\omega)$ is the raw spectrum and $I(\omega)$ is the instrument response (poles/zeros + sensitivity).

### Moment Magnitude

$$
M_w = \frac{2}{3}\log_{10}(M_0) - 10.7
$$

where $M_0$ is the seismic moment (N·m), estimated from the plateau of the displacement spectrum.

---

## Environment Setup

### Install Dependencies

```bash
pip install obspy>=1.4 numpy>=1.24 scipy>=1.11 matplotlib>=3.7
```

### Verify Installation

```python
import obspy
from obspy import read_events
print(f"ObsPy version: {obspy.__version__}")

# Test synthetic waveform generation
from obspy import Trace, Stream
import numpy as np
t = np.linspace(0, 10, 1000)
data = np.sin(2 * np.pi * 5 * t) * np.exp(-t/3)
tr = Trace(data=data)
tr.stats.sampling_rate = 100.0
tr.stats.network = "XX"; tr.stats.station = "TEST"
print(f"Test trace: {tr}")
# Expected: Trace XX.TEST.. | 10.00 s sample, sampling rate: 100.0 Hz
```

---

## Core Workflow

### Step 1: Download Waveforms and Remove Instrument Response

```python
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ #
# Download seismic data around a known earthquake
# Using IRIS/FDSN client — requires internet access
# ------------------------------------------------------------------ #

client = Client("IRIS")

# Example: 2011 Tohoku earthquake (M9.0)
event_time = UTCDateTime("2011-03-11T05:46:24")
origin_lat, origin_lon = 38.297, 142.373

# Download 3-component waveform from nearby station
network, station = "IU", "MAJO"  # Matsushiro, Japan
starttime = event_time - 60   # 1 min before
endtime   = event_time + 600  # 10 min after

print(f"Downloading waveforms for {network}.{station}...")
try:
    st = client.get_waveforms(
        network=network, station=station, location="00",
        channel="BH?",  # BHZ, BHN, BHE
        starttime=starttime, endtime=endtime,
    )
    print(f"Downloaded: {st}")

    # Download station inventory (for instrument response)
    inv = client.get_stations(
        network=network, station=station,
        starttime=starttime, endtime=endtime,
        level="response",
    )
    print(f"Inventory: {inv}")

except Exception as e:
    print(f"Download failed: {e}")
    print("Creating synthetic data for demonstration...")
    # Synthetic fallback: damped sinusoid with P and S arrivals
    from obspy import Trace, Stream, Inventory
    import numpy as np

    dt = 0.01; N = 66000
    t = np.arange(N) * dt
    # P-wave at t=100s, S-wave at t=180s
    p_wave = np.zeros(N); s_wave = np.zeros(N)
    p_idx = int(100/dt)
    s_idx = int(180/dt)
    p_wave[p_idx:p_idx+500] = np.sin(2*np.pi*1.5*t[:500]) * np.exp(-t[:500]/10)
    s_wave[s_idx:s_idx+800] = np.sin(2*np.pi*0.8*t[:800]) * np.exp(-t[:800]/15)
    noise = np.random.randn(N) * 0.02

    st = Stream()
    for ch, amp in [("BHZ", 1.0), ("BHN", 0.7), ("BHE", 0.5)]:
        tr = Trace(data=(p_wave + s_wave + noise) * amp)
        tr.stats.network = network; tr.stats.station = station
        tr.stats.channel = ch; tr.stats.sampling_rate = 100.0
        tr.stats.starttime = starttime
        st.append(tr)
    inv = None

# Make a copy before response removal
st_raw = st.copy()

# ---- Preprocessing ------------------------------------------- #
st.detrend('demean')
st.detrend('linear')
st.taper(max_percentage=0.05, type='cosine')

# Remove instrument response (if inventory available)
if inv is not None:
    st.remove_response(inventory=inv, output='VEL',  # velocity in m/s
                       pre_filt=(0.005, 0.01, 40, 45))
    y_unit = "Velocity (m/s)"
else:
    # Bandpass filter instead
    st.filter('bandpass', freqmin=0.01, freqmax=10.0)
    y_unit = "Counts (filtered)"

# ---- Plot seismograms --------------------------------------- #
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

t_axis = np.arange(len(st[0].data)) * st[0].stats.delta
t0 = float(event_time - starttime)

for ax, tr in zip(axes, st):
    ax.plot(t_axis, tr.data, 'k-', linewidth=0.6)
    ax.axvline(t0, color='red', linestyle='--', linewidth=1.5, label='Origin time')
    ax.set_ylabel(f"{tr.stats.channel}\n{y_unit}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)

axes[-1].set_xlabel("Time after start (s)")
plt.suptitle(f"Seismogram: {network}.{station}  |  2011 Tohoku M9.0")
plt.tight_layout()
plt.savefig("seismogram.png", dpi=150)
plt.show()
```

### Step 2: Phase Picking and Travel Time Calculation

```python
from obspy import UTCDateTime, Stream
from obspy.taup import TauPyModel
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Compute theoretical travel times using 1D Earth model (TauP)
# ------------------------------------------------------------------ #

model = TauPyModel(model="iasp91")  # Standard reference model

# Event parameters
source_depth_km = 25.0    # Focal depth
dist_deg = 15.0           # Epicentral distance (degrees)

# Compute arrivals
arrivals = model.get_travel_times(
    source_depth_in_km=source_depth_km,
    distance_in_degree=dist_deg,
    phase_list=["P", "pP", "PP", "S", "SS", "Rayleigh", "Love"],
)

print(f"Theoretical travel times (Δ={dist_deg}°, h={source_depth_km} km):")
print(f"{'Phase':12s} {'Time (s)':12s} {'Slowness':12s}")
print("-" * 40)
for arr in arrivals:
    print(f"{arr.name:12s} {arr.time:12.2f} {arr.ray_param_sec_degree:12.4f}")

# ---- Automatic onset detection using STA/LTA ------------------- #
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# Generate synthetic Z-component with P onset
np.random.seed(42)
sps = 100.0  # samples/s
N = int(300 * sps)
t_arr = np.arange(N) / sps
# P arrival at t=100s
data = np.random.randn(N) * 0.5
p_onset = int(100 * sps)
data[p_onset:p_onset+int(50*sps)] += (
    np.sin(2*np.pi*3*t_arr[:int(50*sps)]) * np.exp(-t_arr[:int(50*sps)]/8) * 10
)
# S arrival at t=180s
s_onset = int(180 * sps)
data[s_onset:s_onset+int(80*sps)] += (
    np.sin(2*np.pi*1.5*t_arr[:int(80*sps)]) * np.exp(-t_arr[:int(80*sps)]/15) * 15
)

# STA/LTA picker
cft = classic_sta_lta(data, int(0.5*sps), int(10*sps))  # STA=0.5s, LTA=10s
on_off = trigger_onset(cft, thres1=3.5, thres2=0.5)

print(f"\nSTA/LTA detected {len(on_off)} trigger(s):")
for on, off in on_off:
    print(f"  ON at t={on/sps:.2f}s, OFF at t={off/sps:.2f}s")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(t_arr, data, 'k-', linewidth=0.5)
for on, _ in on_off:
    axes[0].axvline(on/sps, color='red', linewidth=1.5, linestyle='--')
axes[0].set_ylabel("Amplitude"); axes[0].set_title("Seismogram + Phase Picks")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_arr, cft, 'b-', linewidth=1)
axes[1].axhline(3.5, color='red', linestyle='--', label='ON threshold')
axes[1].axhline(0.5, color='orange', linestyle='--', label='OFF threshold')
axes[1].set_ylabel("STA/LTA ratio"); axes[1].set_xlabel("Time (s)")
axes[1].set_title("Characteristic Function (STA/LTA)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("phase_picking.png", dpi=150)
plt.show()
```

### Step 3: Spectral Analysis and Magnitude Estimation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.optimize import curve_fit

# ------------------------------------------------------------------ #
# Compute displacement spectrum and estimate moment magnitude
# using Brune source model
# ------------------------------------------------------------------ #

# Simulated ground displacement time series (pre-deconvolved)
np.random.seed(0)
sps = 100.0
N = int(60 * sps)
t = np.arange(N) / sps

# Simulate a Brune pulse: u(t) ∝ t*exp(-t/t_r)
t_r = 2.0  # rise time (s)
signal = np.zeros(N)
onset = int(5 * sps)
pulse_t = t[:N-onset]
signal[onset:] = pulse_t * np.exp(-pulse_t / t_r)
signal += np.random.randn(N) * 1e-8  # noise floor

# Compute amplitude spectrum via FFT
freq = np.fft.rfftfreq(N, d=1/sps)
spec = np.abs(np.fft.rfft(signal)) / sps
# Convert to displacement spectrum (|U(f)|)

# ---- Fit Brune source model: Ω(f) = Ω0 / (1 + (f/fc)²) --------- #
freq_fit = freq[(freq > 0.05) & (freq < 20)]
spec_fit = spec[(freq > 0.05) & (freq < 20)]

def brune_model(f, omega0, fc):
    return omega0 / (1 + (f/fc)**2)

try:
    popt, _ = curve_fit(brune_model, freq_fit, spec_fit,
                         p0=[spec_fit.max(), 1.0], maxfev=5000)
    omega0_hat, fc_hat = popt
    print(f"Brune fit: Ω₀ = {omega0_hat:.4e}, fc = {fc_hat:.3f} Hz")

    # Moment magnitude (simplified: M0 = 4πρv³R*Ω0 / radiation_pattern)
    # Using normalization: M0 ≈ Ω0 (in normalized units for demo)
    rho = 2700.0   # kg/m³  crustal density
    v_p = 6000.0   # m/s    P-wave velocity
    R   = 100e3    # m      hypocentral distance
    F_p = 0.52     # radiation pattern (average)
    M0 = 4 * np.pi * rho * v_p**3 * R * omega0_hat / (2 * F_p)
    Mw = (2/3) * np.log10(M0) - 10.7
    print(f"Seismic moment M₀ = {M0:.3e} N·m")
    print(f"Moment magnitude Mw = {Mw:.2f}")
except Exception as e:
    print(f"Fit failed: {e}")
    omega0_hat, fc_hat = spec_fit.max(), 1.0

# ---- Plot spectrum and fit --------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(t, signal * 1e6, 'k-', linewidth=0.8)
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Displacement (μm)")
axes[0].set_title("Ground Displacement Seismogram"); axes[0].grid(True, alpha=0.3)

f_smooth = np.logspace(np.log10(0.05), np.log10(20), 200)
axes[1].loglog(freq_fit, spec_fit, 'b.', markersize=3, alpha=0.5, label='Data')
axes[1].loglog(f_smooth, brune_model(f_smooth, omega0_hat, fc_hat),
               'r-', linewidth=2.5, label=f'Brune fit (fc={fc_hat:.2f} Hz)')
axes[1].axvline(fc_hat, color='gray', linestyle='--', linewidth=1, alpha=0.7)
axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("Spectral amplitude")
axes[1].set_title("Displacement Spectrum — Brune Source Model")
axes[1].legend(); axes[1].grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("seismic_spectrum.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Receiver Function Computation

```python
from obspy import Stream, Trace
import numpy as np
from scipy.signal import correlate, fftconvolve

def compute_receiver_function(Z_data, R_data, dt, water_level=0.01, f_gauss=2.5):
    """
    Estimate P-to-S receiver function via iterative time-domain deconvolution.

    Parameters
    ----------
    Z_data : array — vertical component (reference)
    R_data : array — radial component (response)
    dt : float — sample interval (s)
    water_level : float — damping for spectral division
    f_gauss : float — Gaussian filter width (Hz)

    Returns
    -------
    rf : array — receiver function time series
    t  : array — time axis (s)
    """
    N = len(Z_data)
    freq = np.fft.rfftfreq(N, d=dt)

    # Gaussian filter to suppress high-frequency noise
    gauss = np.exp(-(freq/(2*f_gauss))**2)

    Z_f = np.fft.rfft(Z_data)
    R_f = np.fft.rfft(R_data)

    # Spectral division with water-level damping
    denom = np.abs(Z_f)**2 + water_level * np.max(np.abs(Z_f)**2)
    rf_f  = R_f * np.conj(Z_f) / denom * gauss

    rf = np.fft.irfft(rf_f, n=N)
    t  = np.arange(N) * dt - N*dt/2  # Center at 0
    return np.roll(rf, N//2), t


# Synthetic test
np.random.seed(42)
dt = 0.025
N = int(100/dt)
t_syn = np.arange(N) * dt - 50

# Z = source pulse; R = delayed converted phase
from scipy.signal import ricker
Z = ricker(N, 10) + np.random.randn(N) * 0.01
ps_delay = int(5 / dt)  # Ps conversion at 5s delay
R = np.roll(ricker(N, 10), ps_delay) * 0.4 + np.random.randn(N) * 0.01

rf, t_rf = compute_receiver_function(Z, R, dt)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
axes[0].plot(t_syn, Z, 'k-'); axes[0].set_title("Vertical (Z)"); axes[0].grid(alpha=0.3)
axes[1].plot(t_syn, R, 'b-'); axes[1].set_title("Radial (R)"); axes[1].grid(alpha=0.3)
# Show only ±20s of RF
mask = (t_rf > -5) & (t_rf < 20)
axes[2].plot(t_rf[mask], rf[mask], 'r-', linewidth=1.5)
axes[2].axvline(0, color='gray', linestyle='--'); axes[2].axhline(0, color='gray', linestyle='-', linewidth=0.5)
axes[2].set_title("Receiver Function"); axes[2].set_xlabel("Time (s)"); axes[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("receiver_function.png", dpi=150); plt.show()
```

---

## Troubleshooting

### Error: `No data available` from FDSN client

**Cause**: Station or channel not available for the requested time window.

**Fix**:
```python
# List available channels
from obspy.clients.fdsn import Client
client = Client("IRIS")
inv = client.get_stations(network="IU", station="MAJO", level="channel",
                          starttime=UTCDateTime("2011-01-01"),
                          endtime=UTCDateTime("2012-01-01"))
print(inv)  # Shows available channels and epochs
```

### Error: `NonLinearLSQError` in Brune fit

**Cause**: Poor initial guess for `curve_fit`.

**Fix**:
```python
p0 = [spec_fit[0], freq_fit[np.argmax(np.diff(np.log(spec_fit+1e-30)) < -0.1)]]
popt, _ = curve_fit(brune_model, freq_fit, spec_fit, p0=p0, maxfev=10000)
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| obspy | 1.4.x | Stable API; 1.4 added FDSN mass downloader |
| scipy | 1.11, 1.12 | `curve_fit` behavior stable |
| numpy | 1.24, 1.26 | No issues |

---

## External Resources

### Official Documentation

- [ObsPy documentation](https://docs.obspy.org)
- [IRIS DMC FDSN web services](https://service.iris.edu)

### Key Papers

- Beyreuther, M. et al. (2010). *ObsPy: A Python Toolbox for Seismology*. Seismological Research Letters, 81(3), 530–533.

---

## Examples

### Example 1: Earthquake Catalog Download

```python
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("USGS")
start = UTCDateTime("2023-01-01")
end   = UTCDateTime("2023-12-31")

# Download M≥6 events in 2023
catalog = client.get_events(
    starttime=start, endtime=end,
    minmagnitude=6.0,
    orderby="magnitude",
)
print(f"Found {len(catalog)} events (M≥6) in 2023:")
for event in catalog[:5]:
    mag  = event.magnitudes[0].mag
    orig = event.origins[0]
    print(f"  M{mag:.1f}  {orig.time.date}  "
          f"lat={orig.latitude:.2f}, lon={orig.longitude:.2f}, "
          f"depth={orig.depth/1000:.1f} km")
```

### Example 2: Waveform Filtering and Spectrogram

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

np.random.seed(1)
sps = 100.0
N = int(120 * sps)
t = np.arange(N) / sps

# Simulate: noise + P-wave (5 Hz) + S-wave (2 Hz)
data = np.random.randn(N) * 0.1
p_idx = int(30 * sps)
data[p_idx:p_idx+int(20*sps)] += np.sin(2*np.pi*5*(t[:int(20*sps)])) * np.exp(-t[:int(20*sps)]/5)
s_idx = int(60 * sps)
data[s_idx:s_idx+int(40*sps)] += np.sin(2*np.pi*2*(t[:int(40*sps)])) * np.exp(-t[:int(40*sps)]/8) * 2

f_spec, t_spec, Sxx = spectrogram(data, fs=sps, nperseg=256, noverlap=200)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
axes[0].plot(t, data, 'k-', linewidth=0.5)
axes[0].axvline(30, color='r', linestyle='--', label='P arrival'); axes[0].axvline(60, color='b', linestyle='--', label='S arrival')
axes[0].set_ylabel("Amplitude"); axes[0].set_title("Seismogram"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].pcolormesh(t_spec, f_spec[f_spec<20], 10*np.log10(Sxx[f_spec<20]+1e-20), shading='gouraud', cmap='viridis')
axes[1].set_ylabel("Frequency (Hz)"); axes[1].set_xlabel("Time (s)"); axes[1].set_title("Spectrogram")
plt.tight_layout(); plt.savefig("spectrogram_seismic.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
