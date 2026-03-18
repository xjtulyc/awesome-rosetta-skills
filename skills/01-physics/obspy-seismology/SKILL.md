---
name: obspy-seismology
description: Seismological data analysis with ObsPy — FDSN waveform download, response removal, phase picking, moment tensor inversion, and seismicity mapping.
tags:
  - seismology
  - geophysics
  - waveform
  - earthquake
  - fdsn
  - moment-tensor
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
  - obspy>=1.4.0
  - scipy>=1.11
  - matplotlib>=3.7
  - numpy>=1.24
  - pandas>=2.0
last_updated: "2026-03-17"
status: stable
---

# ObsPy Seismology

A comprehensive skill for seismological data acquisition, processing, and analysis
using the ObsPy framework. Covers FDSN waveform retrieval, instrument response
removal, P/S phase arrival picking, basic moment tensor inversion, and
seismicity visualisation.

## When to Use This Skill

Use this skill when you need to:

- Download seismogram waveforms from IRIS, GEOFON, ORFEUS, or any FDSN-compliant
  data centre
- Remove instrument response and convert raw counts to physical units (m/s, m/s², Pa)
- Filter, decimate, and taper seismic traces before analysis
- Automatically pick P and S phase arrivals with STA/LTA or kurtosis detectors
- Estimate earthquake source parameters (origin time, location, focal mechanism)
- Build seismicity catalogues and map earthquake distributions
- Compute and visualise spectrograms and particle motion diagrams

This skill is **not** appropriate for:

- Real-time continuous acquisition from hardware digitisers (use SeisComP or
  Earthworm instead)
- Full waveform tomography (use SPECFEM or SALVUS skill)
- Distributed seismic arrays with >100 stations in a single session

## Background & Key Concepts

### ObsPy Data Model

ObsPy organises seismic data in three nested containers:

| Class | Contains | Analogy |
|---|---|---|
| `Stream` | list of `Trace` objects | a multi-channel recording session |
| `Trace` | 1-D NumPy array + `Stats` | a single channel |
| `Stats` | dict-like metadata | network, station, channel, start time, sampling rate |

The **SEED channel code** (e.g., ``BHZ``) encodes band (B = broad-band),
instrument (H = high-gain seismometer), and orientation (Z = vertical).

### FDSN Web Services

The International Federation of Digital Seismograph Networks (FDSN) standardises
three web service endpoints:

- **dataselect** — returns MiniSEED waveforms
- **station** — returns StationXML inventory (instrument response)
- **event** — returns QuakeML earthquake catalogues

ObsPy's ``Client`` class wraps all three. Major nodes: ``"IRIS"``, ``"GEOFON"``,
``"ORFEUS"``, ``"ETH"``, ``"NCEDC"``.

### Instrument Response

Raw seismometer output is in digital counts. The **instrument response** (poles,
zeros, sensitivity) converts counts to ground motion. Removing it via spectral
division yields velocity [m/s], displacement [m], or acceleration [m/s²]
records. ObsPy reads response from StationXML and applies it with
``Trace.remove_response()``.

### Phase Picking

Seismic phase picking identifies the onset time of P (compressional) and S (shear)
waves. Classical approaches use the **STA/LTA** (short-term average / long-term
average) ratio: a sudden energy increase produces a ratio spike. The ``obspy.signal``
module provides ``classic_sta_lta`` and ``recursive_sta_lta``.

### Moment Tensor

A seismic **moment tensor** is a 3×3 symmetric matrix describing the equivalent
force system of an earthquake. The scalar seismic moment ``M_0`` and moment
magnitude ``M_w = (2/3)(log10(M_0) - 9.1)`` are derived from it. Full waveform
inversion (e.g., time-domain L2 misfit minimisation) fits synthetic seismograms
to observed data to recover the tensor.

### Seismicity Maps

Earthquake catalogues are typically distributed as QuakeML or CSV files with
origin time, latitude, longitude, depth, and magnitude. Matplotlib with a
Cartopy or Basemap projection renders these as geographic scatter plots, colour-
coded by depth and scaled by magnitude.

---

## Environment Setup

### Install dependencies

```bash
pip install "obspy>=1.4.0" "scipy>=1.11" "matplotlib>=3.7" \
            "numpy>=1.24" "pandas>=2.0"
```

On conda (recommended for ObsPy to resolve C-extension dependencies):

```bash
conda install -c conda-forge obspy scipy matplotlib numpy pandas
```

Verify the installation:

```bash
python -c "import obspy; print(obspy.__version__)"
```

### Optional: Cartopy for geographic maps

```bash
conda install -c conda-forge cartopy
```

### Environment variables

FDSN data centres are open for most uses. If you access restricted data
(embargoed networks), store credentials securely:

```bash
export FDSN_USER="<paste-your-username>"
export FDSN_PASSWORD="<paste-your-password>"
```

Access in Python:

```python
import os
from obspy.clients.fdsn import Client

user     = os.getenv("FDSN_USER", "")
password = os.getenv("FDSN_PASSWORD", "")

if user:
    client = Client("IRIS", user=user, password=password)
else:
    client = Client("IRIS")   # anonymous for open data
```

---

## Core Workflow

### Step 1 — Download waveforms via FDSN

```python
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Connect to IRIS FDSN data centre
client = Client("IRIS")

# Define a 5-minute window around the 2011 Tohoku earthquake (Mw 9.0)
origin_time = UTCDateTime("2011-03-11T05:46:24")
t_start     = origin_time - 60           # 1 min before origin
t_end       = origin_time + 300 - 60     # 4 min after

# Download broadband vertical (BHZ) from station IU.MAJO (Japan)
st = client.get_waveforms(
    network="IU", station="MAJO", location="00", channel="BHZ",
    starttime=t_start, endtime=t_end
)

print(st)                          # Stream summary
print(st[0].stats)                 # Trace metadata
print(f"Sampling rate: {st[0].stats.sampling_rate} Hz")
print(f"Duration     : {st[0].stats.npts / st[0].stats.sampling_rate:.1f} s")

# Save to MiniSEED for offline use
st.write("tohoku_IU_MAJO_BHZ.mseed", format="MSEED")
print("Saved tohoku_IU_MAJO_BHZ.mseed")
```

### Step 2 — Retrieve StationXML and remove instrument response

```python
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client

client = Client("IRIS")

# Read previously saved waveform
st = read("tohoku_IU_MAJO_BHZ.mseed")
tr = st[0]

origin_time = UTCDateTime("2011-03-11T05:46:24")

# Download StationXML inventory for the station and epoch
inv = client.get_stations(
    network="IU", station="MAJO", location="00", channel="BHZ",
    starttime=tr.stats.starttime, endtime=tr.stats.endtime,
    level="response"
)
inv.write("IU_MAJO_response.xml", format="STATIONXML")

# Pre-processing before response removal
st_proc = st.copy()
st_proc.detrend("linear")            # remove linear trend
st_proc.detrend("demean")            # subtract mean
st_proc.taper(max_percentage=0.05)   # 5% cosine taper at both ends

# Remove response: convert counts -> velocity [m/s]
pre_filt = (0.005, 0.01, 40.0, 45.0)   # four-corner bandpass [Hz]
st_proc.remove_response(
    inventory=inv,
    pre_filt=pre_filt,
    output="VEL",            # options: "DISP", "VEL", "ACC"
    water_level=60
)

print(f"Peak velocity: {abs(st_proc[0].data).max():.3e} m/s")

# Apply additional bandpass filter (0.01 – 1 Hz)
st_proc.filter("bandpass", freqmin=0.01, freqmax=1.0, corners=4, zerophase=True)
st_proc.write("tohoku_MAJO_vel.mseed", format="MSEED")
```

### Step 3 — Automatic P-phase picking with STA/LTA

```python
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import (classic_sta_lta, recursive_sta_lta,
                                  trigger_onset)

st = read("tohoku_MAJO_vel.mseed")
tr = st[0]

df    = tr.stats.sampling_rate        # Hz
npts  = tr.stats.npts
data  = tr.data

# STA/LTA parameters (tune to signal duration and noise)
sta_len = int(1.0  * df)   # 1 s short-term window
lta_len = int(60.0 * df)   # 60 s long-term window

# Compute the characteristic function
cft = recursive_sta_lta(data, sta_len, lta_len)

# Detect triggers: on-threshold=3.5, off-threshold=0.5
on_threshold  = 3.5
off_threshold = 0.5
triggers = trigger_onset(cft, on_threshold, off_threshold)

print(f"Number of triggers detected: {len(triggers)}")
for k, (ton, toff) in enumerate(triggers):
    t_on  = tr.stats.starttime + ton  / df
    t_off = tr.stats.starttime + toff / df
    print(f"  Trigger {k+1}: ON={t_on}  OFF={t_off}  "
          f"(duration {toff - ton:,.0f} samples)")

# Plot waveform + STA/LTA + trigger windows
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
times = tr.times()

ax1.plot(times, data, lw=0.8, color="k")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title(f"{tr.id}  |  {tr.stats.starttime.isoformat()}")

ax2.plot(times, cft, lw=0.8, color="tab:blue", label="STA/LTA")
ax2.axhline(on_threshold,  color="red",    ls="--", label=f"On  ({on_threshold})")
ax2.axhline(off_threshold, color="orange", ls="--", label=f"Off ({off_threshold})")
for ton, toff in triggers:
    ax2.axvspan(ton / df, toff / df, alpha=0.2, color="green")
ax2.set_ylabel("STA/LTA ratio")
ax2.set_xlabel("Time since trace start (s)")
ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout()
plt.savefig("stalta_picks.png", dpi=150)
print("Saved stalta_picks.png")
```

### Step 4 — Download an earthquake catalogue and compute b-value

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("IRIS")

# Fetch M ≥ 5 events in Japan 2020-2024
cat = client.get_events(
    starttime=UTCDateTime("2020-01-01"),
    endtime=UTCDateTime("2024-12-31"),
    minmagnitude=5.0,
    minlatitude=30, maxlatitude=46,
    minlongitude=130, maxlongitude=146,
    catalog="NEIC PDE"
)
print(f"Retrieved {len(cat)} events")

# Convert to DataFrame
records = []
for ev in cat:
    orig = ev.preferred_origin() or ev.origins[0]
    mag  = ev.preferred_magnitude() or ev.magnitudes[0]
    records.append({
        "time"     : orig.time.datetime,
        "lat"      : orig.latitude,
        "lon"      : orig.longitude,
        "depth_km" : orig.depth / 1e3,
        "mag"      : mag.mag,
        "mag_type" : mag.magnitude_type,
    })

df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
df.to_csv("japan_seismicity.csv", index=False)
print(df.head())

# Gutenberg-Richter b-value (maximum likelihood)
M_min = 5.0
M_arr = df["mag"].values
M_mean = M_arr[M_arr >= M_min].mean()
b_value = np.log10(np.e) / (M_mean - M_min)
a_value = np.log10(len(M_arr)) + b_value * M_min
print(f"\nGutenberg-Richter: a = {a_value:.2f},  b = {b_value:.2f}")

# Frequency-magnitude plot
M_bins = np.arange(M_min, 8.5, 0.1)
N_cum  = np.array([(M_arr >= m).sum() for m in M_bins])
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(M_bins, N_cum, "o", ms=4, label="Cumulative N(M≥m)")
ax.semilogy(M_bins, 10 ** (a_value - b_value * M_bins),
            "--", lw=2, label=f"G-R fit (b={b_value:.2f})")
ax.set_xlabel("Magnitude M")
ax.set_ylabel("Cumulative count N(M ≥ m)")
ax.set_title("Gutenberg-Richter relation — Japan 2020-2024")
ax.legend()
fig.tight_layout()
plt.savefig("gr_relation.png", dpi=150)
print("Saved gr_relation.png")
```

### Step 5 — Spectrogram analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import spectrogram

st = read("tohoku_MAJO_vel.mseed")
tr = st[0]
df = tr.stats.sampling_rate
data = tr.data

# Scipy spectrogram
freqs, times_sg, Sxx = spectrogram(
    data,
    fs=df,
    window="hann",
    nperseg=int(df * 20),       # 20-s window
    noverlap=int(df * 10),      # 50% overlap
    scaling="density"
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

t_axis = tr.times()
ax1.plot(t_axis, data, lw=0.6, color="k")
ax1.set_ylabel("Velocity (m/s)")
ax1.set_title(f"{tr.id}  spectrogram")

pcm = ax2.pcolormesh(
    times_sg, freqs,
    10 * np.log10(np.maximum(Sxx, 1e-40)),   # dB re 1 (m/s)^2/Hz
    shading="gouraud", cmap="inferno",
    vmin=-180, vmax=-100
)
ax2.set_ylim(0, min(df / 2, 2.0))   # up to 2 Hz or Nyquist
ax2.set_ylabel("Frequency (Hz)")
ax2.set_xlabel("Time since trace start (s)")
fig.colorbar(pcm, ax=ax2, label="PSD (dB)")
fig.tight_layout()
plt.savefig("spectrogram.png", dpi=150)
print("Saved spectrogram.png")
```

---

## Advanced Usage

### Multi-station P-arrival travel-time curve

```python
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel

client = Client("IRIS")
model  = TauPyModel(model="iasp91")

origin = UTCDateTime("2011-03-11T05:46:24")
ev_lat, ev_lon, ev_depth_km = 38.297, 142.373, 29.0

# Stations at varying epicentral distances
stations = [
    ("IU", "MAJO", "00", "BHZ"),
    ("IU", "HRV",  "00", "BHZ"),
    ("IU", "ANMO", "00", "BHZ"),
    ("IU", "TATO", "00", "BHZ"),
    ("IU", "POHA", "00", "BHZ"),
]

results = []
for net, sta, loc, cha in stations:
    try:
        inv = client.get_stations(network=net, station=sta, level="channel",
                                  starttime=origin)
        st_lat = inv[0][0].latitude
        st_lon = inv[0][0].longitude

        from obspy.geodetics import locations2degrees
        dist_deg = locations2degrees(ev_lat, ev_lon, st_lat, st_lon)

        arrivals = model.get_travel_times(
            source_depth_in_km=ev_depth_km,
            distance_in_degree=dist_deg,
            phase_list=["P", "S"]
        )
        p_time = next((a.time for a in arrivals if a.name == "P"), None)
        s_time = next((a.time for a in arrivals if a.name == "S"), None)
        results.append((f"{net}.{sta}", dist_deg, p_time, s_time))
        print(f"{net}.{sta:6s}  dist={dist_deg:6.2f} deg  P={p_time:.1f}s  S={s_time:.1f}s")
    except Exception as exc:
        print(f"  Skipped {net}.{sta}: {exc}")

# Plot travel-time curve
fig, ax = plt.subplots(figsize=(9, 6))
dists = np.linspace(0, 180, 500)
p_arr = [model.get_travel_times(ev_depth_km, d, ["P"])[0].time
         for d in dists if model.get_travel_times(ev_depth_km, d, ["P"])]

ax.plot(dists[:len(p_arr)], p_arr, lw=2, label="P (iasp91)")
for name, dist, p, s in results:
    ax.scatter(dist, p, zorder=5, s=60, label=name)
ax.set_xlabel("Epicentral distance (deg)")
ax.set_ylabel("Travel time (s)")
ax.set_title("P-wave travel-time curve — Tohoku 2011")
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig("travel_time_curve.png", dpi=150)
```

### Beachball focal mechanism plot

```python
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball

# Tohoku 2011 moment tensor (Harvard CMT) — (strike, dip, rake)
focal_mechanisms = [
    {"angles": [203, 10, 88],  "label": "Tohoku 2011 (Mw 9.0)", "color": "tab:red"},
    {"angles": [355, 85, 170], "label": "Strike-slip example",   "color": "tab:blue"},
    {"angles": [90,  45, -90], "label": "Normal fault",          "color": "tab:green"},
    {"angles": [90,  45,  90], "label": "Reverse fault",         "color": "tab:orange"},
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, fm in zip(axes, focal_mechanisms):
    beachball(fm["angles"], linewidth=2, facecolor=fm["color"], axes=ax)
    ax.set_title(fm["label"], fontsize=9)

fig.suptitle("Beachball focal mechanisms", fontsize=12)
fig.tight_layout()
plt.savefig("beachballs.png", dpi=150)
print("Saved beachballs.png")
```

### Velocity model and ray tracing with TauPy

```python
import numpy as np
import matplotlib.pyplot as plt
from obspy.taup import TauPyModel

model = TauPyModel(model="ak135")

# Compute multiple phase arrivals at 60 degrees for a 100 km deep source
arrivals = model.get_travel_times(
    source_depth_in_km=100,
    distance_in_degree=60,
    phase_list=["P", "pP", "PP", "S", "SS", "SKS", "ScS", "PKP"]
)

print(f"{'Phase':10s}  {'Time (s)':>10s}  {'Ray param':>10s}  {'Purist':>15s}")
print("-" * 50)
for arr in arrivals:
    print(f"{arr.name:10s}  {arr.time:10.2f}  {arr.ray_param:10.4f}  "
          f"{arr.purist_name:>15s}")

# Ray path plot
arrivals_plot = model.get_ray_paths(
    source_depth_in_km=100,
    distance_in_degree=60,
    phase_list=["P", "S", "ScS", "PKP"]
)
ax = arrivals_plot.plot_rays(plot_type="spherical", show=False,
                              legend=True, phase_list=["P", "S", "ScS", "PKP"])
ax.figure.savefig("ray_paths.png", dpi=150)
print("Saved ray_paths.png")
```

### Waveform cross-correlation for relative arrival times

```python
import numpy as np
from scipy.signal import correlate, correlation_lags
from obspy import read, Stream

def cross_correlate_picks(st_ref, st_cmp, freqmin=1.0, freqmax=10.0,
                           window_s=2.0, pick_sample=None):
    """Return sub-sample delay (seconds) between two aligned traces."""
    for tr in [st_ref[0], st_cmp[0]]:
        tr.detrend("linear")
        tr.taper(max_percentage=0.05)
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax,
                  corners=4, zerophase=True)

    df = st_ref[0].stats.sampling_rate
    if pick_sample is None:
        pick_sample = len(st_ref[0].data) // 2

    hw = int(window_s * df / 2)
    a  = st_ref[0].data[pick_sample - hw : pick_sample + hw]
    b  = st_cmp[0].data[pick_sample - hw : pick_sample + hw]

    cc   = correlate(a, b, mode="full")
    lags = correlation_lags(len(a), len(b), mode="full")
    lag_s = lags[np.argmax(cc)] / df
    cc_max = cc.max() / (np.std(a) * np.std(b) * len(a))
    return lag_s, cc_max

# Demo: shift a trace by 0.3 s and recover the delay
from obspy import Trace
import numpy as np

rng = np.random.default_rng(42)
df  = 100.0
t   = np.arange(0, 30, 1.0 / df)
sig = np.sin(2 * np.pi * 3.0 * t) * np.exp(-0.05 * t) + 0.1 * rng.standard_normal(len(t))

tr_ref = Trace(data=sig.copy())
tr_ref.stats.sampling_rate = df

true_delay = 0.3   # seconds
shift_samples = int(true_delay * df)
tr_cmp = Trace(data=np.roll(sig, shift_samples))
tr_cmp.stats.sampling_rate = df

st_ref = Stream([tr_ref])
st_cmp = Stream([tr_cmp])

delay, cc = cross_correlate_picks(st_ref, st_cmp, freqmin=1.0, freqmax=10.0)
print(f"True delay   : {true_delay:.3f} s")
print(f"Measured delay: {delay:.3f} s   (CC max = {cc:.4f})")
```

---

## Troubleshooting

### `FDSNNoDataException` — no data available

```python
# The time window or channel code may be wrong, or data was not archived.
# Verify by listing available channels first:
inv = client.get_stations(
    network="IU", station="MAJO", level="channel",
    starttime=UTCDateTime("2011-03-11"), endtime=UTCDateTime("2011-03-12")
)
for net in inv:
    for sta in net:
        for cha in sta:
            print(cha.code, cha.location_code, cha.start_date, cha.end_date)
```

### Instrument response removal produces unrealistic amplitudes

- Ensure the waveform and inventory cover the same time window.
- Use ``pre_filt`` to suppress low-frequency integration drift, for example
  ``pre_filt = (0.004, 0.008, 45.0, 50.0)`` for broadband data.
- Reduce ``water_level`` from 60 to 30 dB only if the signal-to-noise ratio
  is very high.

### STA/LTA picks on many false triggers (noise)

```python
# Increase the on-threshold, or apply a bandpass filter first:
st.filter("bandpass", freqmin=1.0, freqmax=20.0, corners=4, zerophase=True)
# then re-run STA/LTA
```

### `read()` raises `TypeError: Not a valid MiniSEED file`

Some archives deliver data in SAC, GSE2, or SEISAN format. ObsPy auto-detects
most formats:

```python
st = read("waveform.sac")    # SAC
st = read("waveform.gse")    # GSE2
```

For MSEED files with quality flags, add `headonly=False, check_compression=True`.

### Memory error when reading a long continuous stream

Process in chunks using ``starttime`` / ``endtime`` slices:

```python
chunk_size = 3600   # 1 hour per chunk
t0 = UTCDateTime("2020-01-01")
for i in range(24):
    t_start = t0 + i * chunk_size
    t_end   = t_start + chunk_size
    st_chunk = client.get_waveforms("IU", "ANMO", "00", "BHZ", t_start, t_end)
    # ... process st_chunk ...
```

---

## External Resources

- ObsPy documentation: https://docs.obspy.org
- ObsPy tutorial: https://docs.obspy.org/tutorial/
- IRIS FDSN web services: https://service.iris.edu
- GEOFON data centre: https://geofon.gfz-potsdam.de
- IRIS Wilber 3 waveform explorer: https://ds.iris.edu/wilber3/
- TauPy (travel time calculator): https://docs.obspy.org/packages/obspy.taup.html
- International Seismological Centre: https://www.isc.ac.uk
- USGS Earthquake Hazards Program: https://earthquake.usgs.gov

---

## Examples

### Example 1 — Complete waveform processing and multi-channel plot

```python
"""
Full pipeline:
  1. Download 3-component waveforms (BHE, BHN, BHZ) for the 2011 Tohoku earthquake
  2. Retrieve StationXML response
  3. Pre-process and remove response (velocity output)
  4. Bandpass filter 0.01 – 1 Hz
  5. Rotate ZNE -> ZRT (radial / transverse)
  6. Plot all components with phase arrival markers from TauPy
"""

import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees, gps2dist_azimuth

client = Client("IRIS")
model  = TauPyModel(model="ak135")

# --- Event parameters ---
ev_origin = UTCDateTime("2011-03-11T05:46:24")
ev_lat, ev_lon, ev_depth = 38.297, 142.373, 29.0

# --- Station ---
net, sta, loc = "IU", "MAJO", "00"
t_start = ev_origin + 200
t_end   = ev_origin + 900

# --- Download 3 components ---
print("Downloading waveforms …")
st = client.get_waveforms(net, sta, loc, "BH?", t_start, t_end)
print(st)

# --- StationXML ---
inv = client.get_stations(network=net, station=sta, location=loc,
                           channel="BH?", starttime=t_start, endtime=t_end,
                           level="response")
st_lat = inv[0][0].latitude
st_lon = inv[0][0].longitude

# --- Pre-processing ---
st.detrend("linear")
st.detrend("demean")
st.taper(max_percentage=0.05)

# --- Remove response ---
pre_filt = (0.005, 0.01, 40.0, 45.0)
st.remove_response(inventory=inv, pre_filt=pre_filt, output="VEL",
                   water_level=60)

# --- Bandpass filter ---
st.filter("bandpass", freqmin=0.01, freqmax=1.0, corners=4, zerophase=True)
st.sort(keys=["channel"])

# --- Rotate to ZRT ---
dist_deg = locations2degrees(ev_lat, ev_lon, st_lat, st_lon)
dist_m, az, baz = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
try:
    st.rotate(method="NE->RT", back_azimuth=baz)
    print(f"Rotated to ZRT (back-azimuth = {baz:.1f} deg)")
except Exception as exc:
    print(f"Rotation skipped: {exc}")

# --- TauPy phase arrivals ---
arrivals = model.get_travel_times(
    source_depth_in_km=ev_depth,
    distance_in_degree=dist_deg,
    phase_list=["P", "pP", "PP", "S", "SS", "SKS"]
)

# --- Multi-panel plot ---
n_comp = len(st)
fig, axes = plt.subplots(n_comp, 1, figsize=(14, 3 * n_comp), sharex=True)
if n_comp == 1:
    axes = [axes]

phase_colors = {"P": "red", "pP": "orangered", "PP": "tomato",
                "S": "blue", "SS": "cornflowerblue", "SKS": "navy"}

for ax, tr in zip(axes, st):
    times = tr.times(reftime=ev_origin)
    ax.plot(times, tr.data, lw=0.7, color="k")
    ax.set_ylabel(f"{tr.stats.channel}\n(m/s)", fontsize=9)

    for arr in arrivals:
        color = phase_colors.get(arr.name, "gray")
        ax.axvline(arr.time, color=color, lw=1.2, ls="--", alpha=0.8)
        ax.text(arr.time, ax.get_ylim()[1] * 0.85, arr.name,
                color=color, fontsize=7, rotation=90, va="top")

axes[-1].set_xlabel("Time after origin (s)")
fig.suptitle(f"{net}.{sta}  —  Tohoku 2011 (Mw 9.0)  |  "
             f"Dist={dist_deg:.1f}°  Az={az:.1f}°", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("tohoku_waveforms_ZRT.png", dpi=150)
print("Saved tohoku_waveforms_ZRT.png")
```

### Example 2 — Seismicity map with depth-coloured scatter plot

```python
"""
Seismicity map:
  1. Query FDSN event service for Japan 2023 (M ≥ 4)
  2. Build a pandas DataFrame
  3. Plot a geographic map coloured by focal depth
     (pure matplotlib, no Cartopy required)
  4. Annotate the largest event
  5. Add a magnitude-scaled marker size legend
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("IRIS")

print("Fetching earthquake catalogue …")
cat = client.get_events(
    starttime=UTCDateTime("2023-01-01"),
    endtime=UTCDateTime("2023-12-31"),
    minmagnitude=4.0,
    minlatitude=28, maxlatitude=46,
    minlongitude=128, maxlongitude=148,
)
print(f"  {len(cat)} events retrieved")

# Build DataFrame
records = []
for ev in cat:
    orig = ev.preferred_origin() or ev.origins[0]
    mag  = ev.preferred_magnitude() or ev.magnitudes[0]
    records.append({
        "lat"   : orig.latitude,
        "lon"   : orig.longitude,
        "depth" : orig.depth / 1e3,
        "mag"   : mag.mag,
        "time"  : str(orig.time),
    })

df = pd.DataFrame(records)
df.to_csv("japan_2023_catalog.csv", index=False)
print(df.describe())

# --- Map ---
fig, ax = plt.subplots(figsize=(9, 10))

# Colour by depth (0 – 200 km)
norm   = mcolors.Normalize(vmin=0, vmax=200)
cmap   = cm.plasma_r
colors = cmap(norm(df["depth"].values))

# Marker size proportional to magnitude
sizes  = 0.5 * 10 ** (0.7 * df["mag"].values)   # M4 -> ~8 pt, M7 -> ~200 pt

scatter = ax.scatter(
    df["lon"], df["lat"],
    s=sizes, c=df["depth"],
    cmap=cmap, norm=norm,
    alpha=0.7, linewidths=0.3, edgecolors="k"
)

# Largest event annotation
idx_max = df["mag"].idxmax()
ax.annotate(
    f"  Mw {df.loc[idx_max, 'mag']:.1f}\n  {df.loc[idx_max, 'time'][:10]}",
    xy=(df.loc[idx_max, "lon"], df.loc[idx_max, "lat"]),
    fontsize=8, color="white",
    arrowprops=dict(arrowstyle="->", color="white", lw=1.0),
    xytext=(df.loc[idx_max, "lon"] + 1.5, df.loc[idx_max, "lat"] + 0.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6)
)

# Colourbar and legend
cb = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Focal depth (km)", fontsize=10)

# Magnitude size legend
for mag_ref in [4.0, 5.0, 6.0, 7.0]:
    ax.scatter([], [], s=0.5 * 10 ** (0.7 * mag_ref), color="grey",
               alpha=0.7, label=f"M {mag_ref:.0f}")
ax.legend(title="Magnitude", loc="lower left", fontsize=8, title_fontsize=9)

ax.set_xlim(128, 148)
ax.set_ylim(28, 46)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("Japan Seismicity 2023  (M ≥ 4)")
ax.grid(True, lw=0.3, alpha=0.5)
fig.tight_layout()
plt.savefig("japan_seismicity_2023.png", dpi=150)
print("Saved japan_seismicity_2023.png")
```
