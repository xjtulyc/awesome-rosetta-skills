---
name: spike-sorting
description: "Automated spike sorting pipeline using SpikeInterface with Kilosort2/Mountainsort5, quality metrics, and Phy export for extracellular neural recordings."
tags:
  - neuroscience
  - electrophysiology
  - spike-sorting
  - spikeinterface
  - kilosort
  - neural-data
version: "1.0.0"
authors:
  - name: "Rosetta Skills Contributors"
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - spikeinterface>=0.100
  - matplotlib>=3.7
  - numpy>=1.24
  - pandas>=2.0
  - scipy>=1.11
  - h5py>=3.9
last_updated: "2026-03-17"
status: "stable"
---

# Spike Sorting with SpikeInterface

Automated pipeline for sorting spikes from extracellular multi-electrode recordings. Covers
multi-probe loading, signal preprocessing, running Kilosort2 or Mountainsort5, computing
unit quality metrics, and exporting curated results to Phy or CSV.

---

## When to Use This Skill

- You have raw binary, NWB, SpikeGLX (.bin/.meta), or Open Ephys recordings and need to
  identify individual neuron spike trains.
- You want a reproducible, multi-sorter comparison pipeline rather than running a single
  GUI-based sorter.
- You need automated quality control: ISI violations, SNR, presence ratio, amplitude cutoff.
- You want to export results for manual curation in Phy or downstream population analyses.
- You are processing multi-probe (Neuropixels) datasets with hundreds of channels.

---

## Background & Key Concepts

### Extracellular Electrophysiology

A multi-electrode array records local field potentials and action potentials (spikes) from
nearby neurons. Spike sorting is the computational step that assigns each detected spike
waveform to a putative single unit.

### SpikeInterface

SpikeInterface is a Python framework that provides a unified API over dozens of file formats
and sorters. Key objects:

- **RecordingExtractor** — wraps raw data, channel geometry, and probe info.
- **SortingExtractor** — wraps a set of unit spike trains.
- **WaveformExtractor / SortingAnalyzer** — computes templates and extensions (PCA, metrics).

### Preprocessing Steps

| Step | Purpose |
|------|---------|
| Bandpass filter (300–6000 Hz) | Remove LFP and high-frequency noise |
| Common Median Reference (CMR) | Cancel common-mode noise across channels |
| Whitening | Decorrelate channels for some sorters |
| Bad-channel removal | Prevent noisy channels from contaminating sorting |

### Quality Metrics

| Metric | Good threshold |
|--------|---------------|
| ISI violation ratio | < 0.05 |
| SNR (peak-to-peak / noise) | > 3 |
| Presence ratio | > 0.8 |
| Amplitude cutoff | < 0.1 |
| Firing rate (Hz) | > 0.1 |

### Sorters

- **Kilosort2** — GPU-accelerated template-matching sorter; best for Neuropixels data.
- **Mountainsort5** — CPU-based, reproducible; good for tetrodes and lower channel counts.
- **Tridesclous2** — Fast CPU sorter with built-in quality control.

---

## Environment Setup

### Install Dependencies

```bash
pip install "spikeinterface[full]>=0.100" matplotlib numpy pandas scipy h5py
```

### Kilosort2 (MATLAB-based, requires wrapper)

```bash
# Install the MATLAB Kilosort2 code separately, then point SpikeInterface to it
# Alternatively, use the Kilosort Python port:
pip install kilosort   # PyKilosort (Kilosort4 Python port)
```

### Mountainsort5 (pure Python, no MATLAB required)

```bash
pip install mountainsort5
```

### Verify Installation

```python
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

print("SpikeInterface version:", si.__version__)
print("Available sorters:", ss.available_sorters())
```

### Check GPU (for Kilosort2/4)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Core Workflow

### Step 1: Load and Inspect a Recording

```python
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from pathlib import Path

# ----- Load from SpikeGLX (Neuropixels) -----
spikeglx_folder = Path("/data/recordings/session01")
recording_raw = se.read_spikeglx(spikeglx_folder, stream_name="imec0.ap")

print(f"Num channels : {recording_raw.get_num_channels()}")
print(f"Sampling rate: {recording_raw.get_sampling_frequency()} Hz")
print(f"Duration     : {recording_raw.get_total_duration():.1f} s")
print(f"Dtype        : {recording_raw.get_dtype()}")

# ----- Load from Open Ephys -----
# recording_raw = se.read_openephys("/data/recordings/openephys_session")

# ----- Load from a raw binary file -----
# recording_raw = se.read_binary(
#     "recording.bin",
#     sampling_frequency=30000.0,
#     num_channels=384,
#     dtype="int16",
#     gain_to_uV=0.195,
#     offset_to_uV=0.0,
# )

# Inspect probe geometry
probe = recording_raw.get_probe()
print(probe)

# Quick snippet plot (first 0.1 s, first 10 channels)
snippet = recording_raw.get_traces(
    start_frame=0, end_frame=3000, channel_ids=recording_raw.channel_ids[:10]
)
fig, ax = plt.subplots(figsize=(12, 6))
offset = 0
for i, ch in enumerate(recording_raw.channel_ids[:10]):
    ax.plot(np.arange(snippet.shape[0]) / recording_raw.get_sampling_frequency(),
            snippet[:, i] + offset, lw=0.5)
    offset += 200
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (µV) + offset")
ax.set_title("Raw traces — first 10 channels")
plt.tight_layout()
plt.savefig("raw_traces.png", dpi=150)
plt.show()
```

### Step 2: Preprocess the Recording

```python
import spikeinterface.preprocessing as sp

# --- Bad channel detection ---
bad_channel_ids, channel_labels = sp.detect_bad_channels(recording_raw)
print(f"Bad channels detected: {bad_channel_ids}")
recording_good = recording_raw.remove_channels(bad_channel_ids)

# --- Bandpass filter: keep action potentials (300–6000 Hz) ---
recording_bp = sp.bandpass_filter(
    recording_good, freq_min=300.0, freq_max=6000.0
)

# --- Common Median Reference (CMR) across all channels ---
recording_cmr = sp.common_reference(
    recording_bp, reference="global", operator="median"
)

# --- Optional: whiten for sorters that expect it ---
# recording_final = sp.whiten(recording_cmr, dtype="float32")

recording_preprocessed = recording_cmr
print("Preprocessing done. Summary:")
print(recording_preprocessed)

# Save to binary for faster I/O during sorting
preprocessed_path = Path("/data/recordings/session01/preprocessed")
recording_saved = recording_preprocessed.save(
    folder=preprocessed_path,
    n_jobs=8,
    chunk_duration="1s",
    progress_bar=True,
)
```

### Step 3: Run Spike Sorting

```python
import spikeinterface.sorters as ss
from pathlib import Path

output_folder = Path("/data/recordings/session01/sorting_output")

# ---- Option A: Kilosort4 (PyKilosort, GPU) ----
sorting_kilosort = ss.run_sorter(
    "kilosort4",
    recording=recording_saved,
    output_folder=output_folder / "kilosort4",
    remove_existing_folder=True,
    # Kilosort4-specific params
    nblocks=5,           # number of drift blocks
    Th_universal=9,
    Th_learned=8,
)

# ---- Option B: Mountainsort5 (CPU, reproducible) ----
sorting_ms5 = ss.run_sorter(
    "mountainsort5",
    recording=recording_saved,
    output_folder=output_folder / "mountainsort5",
    remove_existing_folder=True,
    scheme="2",
    detect_threshold=5.5,
    detect_sign=-1,
    filter=False,         # already filtered
    whiten=True,
    snippet_T1=20,
    snippet_T2=40,
)

# ---- Option C: Tridesclous2 (CPU, fast) ----
sorting_tdc = ss.run_sorter(
    "tridesclous2",
    recording=recording_saved,
    output_folder=output_folder / "tridesclous2",
    remove_existing_folder=True,
)

print(f"Kilosort4   units: {sorting_kilosort.get_num_units()}")
print(f"Mountainsort5 units: {sorting_ms5.get_num_units()}")
print(f"Tridesclous2  units: {sorting_tdc.get_num_units()}")

# Work with one sorter going forward
sorting = sorting_kilosort
```

### Step 4: Compute Waveforms and Quality Metrics

```python
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

analyzer_folder = Path("/data/recordings/session01/analyzer_kilosort4")

# Build SortingAnalyzer — the modern replacement for WaveformExtractor
analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording_saved,
    folder=analyzer_folder,
    format="binary_folder",
    overwrite=True,
    sparse=True,                  # only store waveforms on best channels
    n_jobs=8,
)

# Compute extensions incrementally
analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms",     ms_before=1.0, ms_after=2.0, dtype="float32")
analyzer.compute("templates",     operators=["average", "std"])
analyzer.compute("noise_levels")
analyzer.compute("spike_amplitudes")
analyzer.compute("unit_locations", method="monopolar_triangulation")
analyzer.compute("template_similarity")
analyzer.compute("principal_components", n_components=5, mode="by_channel_local")
analyzer.compute("quality_metrics",
                 metric_names=["snr", "isi_violation", "presence_ratio",
                               "amplitude_cutoff", "firing_rate"])

# Extract quality metrics table
metrics_df = analyzer.get_extension("quality_metrics").get_data()
print(metrics_df.head())

# Apply thresholds to identify "good" units
good_units_mask = (
    (metrics_df["snr"] > 3.0) &
    (metrics_df["isi_violations_ratio"] < 0.05) &
    (metrics_df["presence_ratio"] > 0.8) &
    (metrics_df["amplitude_cutoff"] < 0.1) &
    (metrics_df["firing_rate"] > 0.1)
)
good_unit_ids = metrics_df.index[good_units_mask].tolist()
print(f"Total units  : {len(metrics_df)}")
print(f"Good units   : {len(good_unit_ids)}")

# Filter sorting to good units only
sorting_curated = sorting.select_units(good_unit_ids)
```

### Step 5: Visualize and Export

```python
import spikeinterface.widgets as sw
import pandas as pd

# ---- Raster plot of good units ----
fig, ax = plt.subplots(figsize=(14, 5))
for y_pos, uid in enumerate(good_unit_ids[:30]):
    spikes = sorting_curated.get_unit_spike_train(uid, segment_index=0)
    spike_times = spikes / recording_saved.get_sampling_frequency()
    ax.scatter(spike_times, np.full_like(spike_times, y_pos),
               s=1, c="black", alpha=0.6)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit index")
ax.set_title("Spike raster — curated good units")
plt.tight_layout()
plt.savefig("raster_good_units.png", dpi=150)
plt.show()

# ---- Template waveforms for a few units ----
templates_ext = analyzer.get_extension("templates")
templates_avg = templates_ext.get_templates(operator="average")  # (n_units, n_samples, n_channels)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, uid in zip(axes.flat, good_unit_ids[:10]):
    unit_idx = list(analyzer.unit_ids).index(uid)
    ch_best = int(np.argmax(np.ptp(templates_avg[unit_idx], axis=0)))
    ax.plot(templates_avg[unit_idx, :, ch_best], lw=1.5)
    ax.set_title(f"Unit {uid}")
    ax.set_xlabel("Sample")
plt.suptitle("Mean templates (best channel)", y=1.01)
plt.tight_layout()
plt.savefig("templates.png", dpi=150)
plt.show()

# ---- Export quality metrics to CSV ----
metrics_df["is_good"] = metrics_df.index.isin(good_unit_ids)
metrics_df.to_csv("unit_quality_metrics.csv")
print("Saved unit_quality_metrics.csv")

# ---- Export to Phy for manual curation ----
from spikeinterface.exporters import export_to_phy

phy_folder = Path("/data/recordings/session01/phy_kilosort4")
export_to_phy(
    analyzer,
    output_folder=phy_folder,
    compute_amplitudes=True,
    compute_pc_features=True,
    copy_binary=True,
    remove_if_exists=True,
    chunk_duration="1s",
    n_jobs=8,
)
print(f"Phy export ready at: {phy_folder}")
print("Run:  phy template-gui", phy_folder / "params.py")
```

---

## Advanced Usage

### Multi-Sorter Comparison

```python
from spikeinterface.comparison import compare_multiple_sorters
import spikeinterface.widgets as sw

# Run all three sorters on the same recording (see Step 3)
sorting_list = [sorting_kilosort, sorting_ms5, sorting_tdc]
sorter_names = ["kilosort4", "mountainsort5", "tridesclous2"]

comparison = compare_multiple_sorters(
    sorting_list=sorting_list,
    name_list=sorter_names,
    delta_time=0.4,       # ms window for matching spikes
    match_score=0.5,
)

# Agreement matrix — shows how many spikes each sorter pair share
print(comparison.agreement_scores)

# Units agreed upon by all three sorters (high confidence)
agreement_sorting = comparison.get_agreement_sorting(minimum_agreement_count=3)
print(f"Units agreed by all 3 sorters: {agreement_sorting.get_num_units()}")
```

### Drift Correction

```python
from spikeinterface.preprocessing import correct_motion

# Estimate and correct probe drift (important for long recordings)
recording_motion_corrected, motion_info = correct_motion(
    recording_preprocessed,
    preset="nonrigid_accurate",    # or "rigid_fast"
    output_motion_info=True,
    n_jobs=8,
)

# Visualize estimated drift
fig, ax = plt.subplots(figsize=(12, 4))
motion = motion_info["motion"]
time_axis = motion_info["temporal_bins_s"]
ax.plot(time_axis, motion.displacement[0][:, 0], lw=1)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Estimated drift (µm)")
ax.set_title("Probe drift over session")
plt.tight_layout()
plt.savefig("drift_estimate.png", dpi=150)
plt.show()
```

### Batch Processing Multiple Sessions

```python
from pathlib import Path
import pandas as pd

sessions = [
    "/data/recordings/session01",
    "/data/recordings/session02",
    "/data/recordings/session03",
]

all_metrics = []

for session_path in sessions:
    session = Path(session_path)
    print(f"\nProcessing {session.name} ...")

    # Load
    rec = se.read_spikeglx(session, stream_name="imec0.ap")

    # Preprocess
    rec = sp.bandpass_filter(rec, freq_min=300, freq_max=6000)
    rec = sp.common_reference(rec, reference="global", operator="median")
    bad_chs, _ = sp.detect_bad_channels(rec)
    rec = rec.remove_channels(bad_chs)

    # Sort
    sorting = ss.run_sorter(
        "mountainsort5",
        recording=rec,
        output_folder=session / "sorting_ms5",
        remove_existing_folder=True,
        filter=False,
    )

    # Metrics
    analyzer = si.create_sorting_analyzer(
        sorting, rec,
        folder=session / "analyzer",
        format="binary_folder",
        overwrite=True,
        n_jobs=4,
    )
    analyzer.compute("random_spikes")
    analyzer.compute("waveforms")
    analyzer.compute("templates")
    analyzer.compute("noise_levels")
    analyzer.compute("quality_metrics",
                     metric_names=["snr", "isi_violation",
                                   "presence_ratio", "firing_rate"])

    mdf = analyzer.get_extension("quality_metrics").get_data()
    mdf["session"] = session.name
    all_metrics.append(mdf)

summary = pd.concat(all_metrics)
summary.to_csv("all_sessions_metrics.csv")
print("Batch processing complete. Results in all_sessions_metrics.csv")
```

### ISI Violation Histogram

```python
def plot_isi_histogram(sorting, unit_id, fs, max_isi_ms=50.0, bin_ms=1.0):
    """Plot interspike interval histogram for a single unit."""
    spikes = sorting.get_unit_spike_train(unit_id, segment_index=0)
    isis_ms = np.diff(spikes) / fs * 1000.0

    bins = np.arange(0, max_isi_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(isis_ms, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(edges[:-1], counts, width=bin_ms * 0.9, color="steelblue", align="edge")
    ax.axvline(2.0, color="red", lw=1.5, linestyle="--", label="2 ms refractory")
    ax.set_xlabel("ISI (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"ISI histogram — Unit {unit_id}")
    ax.legend()
    plt.tight_layout()
    return fig

fs = recording_saved.get_sampling_frequency()
for uid in good_unit_ids[:3]:
    fig = plot_isi_histogram(sorting_curated, uid, fs)
    fig.savefig(f"isi_unit_{uid}.png", dpi=150)
    plt.close(fig)
```

---

## Troubleshooting

### Sorter Not Found

```bash
# Check which sorters are installed
python -c "import spikeinterface.sorters as ss; print(ss.installed_sorters())"

# Install a missing sorter
pip install mountainsort5   # for mountainsort5
pip install kilosort        # for kilosort4 (Python port)
```

### Out of Memory During Sorting

```python
# Reduce chunk size to lower RAM usage
recording_saved = recording_preprocessed.save(
    folder="/data/recordings/session01/preprocessed",
    chunk_duration="0.5s",   # smaller chunks
    n_jobs=4,                # fewer parallel jobs
)

# For Kilosort4, reduce batch size
sorting = ss.run_sorter(
    "kilosort4",
    recording=recording_saved,
    output_folder=output_folder,
    NT=65536,   # default is 65536 * 2; reduce if OOM
)
```

### Poor Sorting Quality (too few / too many units)

```python
# Adjust detection threshold (lower = more units, more noise)
sorting_ms5 = ss.run_sorter(
    "mountainsort5",
    recording=recording_saved,
    output_folder=output_folder / "mountainsort5_v2",
    detect_threshold=4.5,   # was 5.5; try 4.0–6.0
    detect_sign=-1,
    filter=False,
)

# Check noise levels to calibrate threshold
noise_levels = si.get_noise_levels(recording_saved, return_scaled=True)
print(f"Median noise: {np.median(noise_levels):.2f} µV")
print(f"5× noise    : {5 * np.median(noise_levels):.2f} µV  <- detection threshold")
```

### Bad Channel Detection Removes Too Many Channels

```python
# Inspect channel coherence before removal
recording_check = sp.detect_bad_channels(
    recording_raw,
    method="coherence+psd",
    dead_channel_threshold=0.4,
    noisy_channel_threshold=1.0,   # raise to keep noisier channels
)
```

### Phy Export Fails

```bash
# Ensure phy is installed
pip install phy

# Check that templates and PCA extensions are computed
python -c "
from pathlib import Path
import spikeinterface as si
analyzer = si.load_sorting_analyzer('path/to/analyzer')
print(analyzer.get_loaded_extension_names())
"
```

---

## External Resources

- SpikeInterface documentation: https://spikeinterface.readthedocs.io
- SpikeInterface tutorials (Jupyter): https://github.com/SpikeInterface/spikeinterface/tree/main/examples
- Kilosort4 (Python): https://github.com/MouseLand/Kilosort
- Mountainsort5: https://github.com/flatironinstitute/mountainsort5
- Phy curation GUI: https://github.com/cortex-lab/phy
- Quality metrics reference: https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html
- SpikeGLX recording system: https://billkarsh.github.io/SpikeGLX/
- Open Ephys: https://open-ephys.org/
- Neuropixels probe geometry: https://www.neuropixels.org/

---

## Examples

### Example 1: Full Pipeline on a Synthetic Recording

This example uses SpikeInterface's built-in MEArec synthetic generator so the pipeline
runs without any real data file.

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

# ------------------------------------------------------------------ #
#  1. Generate a synthetic recording (30 s, 32 channels, 10 units)   #
# ------------------------------------------------------------------ #
from spikeinterface.generation import generate_ground_truth_recording

recording_raw, sorting_true = generate_ground_truth_recording(
    durations=[30.0],
    sampling_frequency=30000.0,
    num_channels=32,
    num_units=10,
    seed=42,
)
print(f"Generated {recording_raw.get_num_channels()} channels, "
      f"{sorting_true.get_num_units()} ground-truth units")

# ------------------------------------------------------------------ #
#  2. Preprocess                                                       #
# ------------------------------------------------------------------ #
recording_bp  = sp.bandpass_filter(recording_raw, freq_min=300, freq_max=6000)
recording_cmr = sp.common_reference(recording_bp, reference="global", operator="median")

# ------------------------------------------------------------------ #
#  3. Sort (Mountainsort5 — no GPU required)                          #
# ------------------------------------------------------------------ #
sorting_result = ss.run_sorter(
    "mountainsort5",
    recording=recording_cmr,
    output_folder=Path("/tmp/ms5_synthetic"),
    remove_existing_folder=True,
    detect_threshold=5.0,
    filter=False,
)
print(f"Sorted units: {sorting_result.get_num_units()}")

# ------------------------------------------------------------------ #
#  4. Compare with ground truth                                        #
# ------------------------------------------------------------------ #
from spikeinterface.comparison import compare_sorter_to_ground_truth

comparison = compare_sorter_to_ground_truth(sorting_true, sorting_result, delta_time=0.4)
perf = comparison.get_performance(method="pooled_with_average")
print(perf[["accuracy", "precision", "recall"]])

# ------------------------------------------------------------------ #
#  5. Quality metrics                                                  #
# ------------------------------------------------------------------ #
analyzer = si.create_sorting_analyzer(
    sorting=sorting_result,
    recording=recording_cmr,
    folder=Path("/tmp/analyzer_synthetic"),
    format="binary_folder",
    overwrite=True,
)
analyzer.compute("random_spikes", max_spikes_per_unit=200)
analyzer.compute("waveforms")
analyzer.compute("templates")
analyzer.compute("noise_levels")
analyzer.compute("quality_metrics",
                 metric_names=["snr", "isi_violation", "presence_ratio"])

metrics = analyzer.get_extension("quality_metrics").get_data()
print(metrics)

# ------------------------------------------------------------------ #
#  6. Raster plot                                                      #
# ------------------------------------------------------------------ #
fs = recording_cmr.get_sampling_frequency()
fig, ax = plt.subplots(figsize=(12, 5))
for y, uid in enumerate(sorting_result.unit_ids):
    spikes = sorting_result.get_unit_spike_train(uid, segment_index=0) / fs
    ax.scatter(spikes, np.full_like(spikes, y), s=1.5, c="navy", alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Unit")
ax.set_title("Spike raster — Mountainsort5 on synthetic data")
plt.tight_layout()
plt.savefig("synthetic_raster.png", dpi=150)
plt.show()
```

### Example 2: Neuropixels Multi-Shank Recording with Drift Correction

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss
from spikeinterface.preprocessing import correct_motion

# ------------------------------------------------------------------ #
#  1. Load Neuropixels 2.0 multi-shank recording                      #
# ------------------------------------------------------------------ #
npx_folder = Path("/data/recordings/npx2_session")

# SpikeGLX saves one stream per shank: imec0.ap, imec1.ap, etc.
recording_shank0 = se.read_spikeglx(npx_folder, stream_name="imec0.ap")
recording_shank1 = se.read_spikeglx(npx_folder, stream_name="imec1.ap")

print(f"Shank 0: {recording_shank0.get_num_channels()} channels")
print(f"Shank 1: {recording_shank1.get_num_channels()} channels")

# Process each shank independently
results = {}
for shank_id, rec in [("shank0", recording_shank0), ("shank1", recording_shank1)]:
    print(f"\n--- Processing {shank_id} ---")

    # Preprocess
    bad_chs, _ = sp.detect_bad_channels(rec)
    rec_clean = rec.remove_channels(bad_chs)
    rec_bp    = sp.bandpass_filter(rec_clean, freq_min=300, freq_max=6000)
    rec_cmr   = sp.common_reference(rec_bp, reference="global", operator="median")

    # Drift correction (important for >1 h recordings)
    rec_corrected, motion_info = correct_motion(
        rec_cmr,
        preset="nonrigid_accurate",
        output_motion_info=True,
        n_jobs=8,
    )

    max_drift = float(np.ptp(motion_info["motion"].displacement[0][:, 0]))
    print(f"  Estimated max drift: {max_drift:.1f} µm")

    # Save preprocessed recording
    rec_saved = rec_corrected.save(
        folder=npx_folder / f"preprocessed_{shank_id}",
        n_jobs=8,
        chunk_duration="1s",
        overwrite=True,
    )

    # Sort with Kilosort4
    sorting = ss.run_sorter(
        "kilosort4",
        recording=rec_saved,
        output_folder=npx_folder / f"sorting_{shank_id}",
        remove_existing_folder=True,
        nblocks=5,
    )
    print(f"  Sorted units: {sorting.get_num_units()}")

    # Quality metrics
    analyzer = si.create_sorting_analyzer(
        sorting, rec_saved,
        folder=npx_folder / f"analyzer_{shank_id}",
        overwrite=True,
    )
    analyzer.compute("random_spikes", max_spikes_per_unit=300)
    analyzer.compute("waveforms")
    analyzer.compute("templates")
    analyzer.compute("noise_levels")
    analyzer.compute("quality_metrics",
                     metric_names=["snr", "isi_violation",
                                   "presence_ratio", "amplitude_cutoff"])

    mdf = analyzer.get_extension("quality_metrics").get_data()
    good = mdf[(mdf["snr"] > 3) & (mdf["isi_violations_ratio"] < 0.05)]
    print(f"  Good units: {len(good)}/{len(mdf)}")

    results[shank_id] = {"analyzer": analyzer, "metrics": mdf}

# ------------------------------------------------------------------ #
#  3. Combined unit location plot across shanks                        #
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, 2, figsize=(8, 12), sharey=True)

for ax, (shank_id, res) in zip(axes, results.items()):
    analyzer = res["analyzer"]
    analyzer.compute("unit_locations", method="monopolar_triangulation")
    locs = analyzer.get_extension("unit_locations").get_data()
    snr  = res["metrics"]["snr"].values

    sc = ax.scatter(locs[:, 0], locs[:, 1], c=snr, cmap="viridis",
                    vmin=0, vmax=10, s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="SNR")
    ax.set_title(shank_id)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("Depth (µm)")

plt.suptitle("Unit locations colored by SNR", y=1.01)
plt.tight_layout()
plt.savefig("unit_locations_multi_shank.png", dpi=150, bbox_inches="tight")
plt.show()
```
