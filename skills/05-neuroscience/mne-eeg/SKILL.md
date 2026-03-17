---
name: mne-eeg
description: >
  Full EEG/MEG analysis pipeline with MNE-Python: preprocessing, ICA artifact removal, ERP
  computation, time-frequency analysis, and resting-state power spectral density.
tags:
  - neuroscience
  - eeg
  - mne-python
  - erp
  - time-frequency
  - signal-processing
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
  - mne>=1.5.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
last_updated: "2026-03-17"
---

# EEG/MEG Analysis with MNE-Python

## Overview

MNE-Python is the standard open-source toolkit for analyzing M/EEG data. This skill provides a
complete pipeline from raw file loading through preprocessing, epoching, ERP computation,
time-frequency analysis, and basic source localization. All functions follow MNE idioms and
produce publication-ready figures.

### Supported File Formats

| Format | Extension | Notes |
|---|---|---|
| European Data Format | `.edf` | Common for clinical EEG |
| FIF (Neuromag) | `.fif` | Native MNE format |
| BrainVision | `.vhdr` | Requires `.eeg` + `.vmrk` |
| EEGLab | `.set` | Requires `.fdt` companion file |
| BioSemi | `.bdf` | High-density ActiveTwo systems |

---

## Setup

```bash
pip install mne numpy scipy matplotlib pandas scikit-learn
```

For optional source localization:

```bash
pip install mne[hdf5]       # HDF5 support
pip install nibabel          # NIfTI/FreeSurfer surfaces
```

---

## Core Functions

```python
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet, psd_array_welch

mne.set_log_level("WARNING")  # Reduce verbosity; use "INFO" for debugging


# ---------------------------------------------------------------------------
# 1. Raw Data Loading
# ---------------------------------------------------------------------------


def load_raw_eeg(
    filepath: str,
    preload: bool = True,
    montage_name: str = "standard_1020",
) -> mne.io.BaseRaw:
    """
    Load raw EEG data from EDF, FIF, BrainVision, or EEGLab formats.

    Parameters
    ----------
    filepath : str
        Path to the raw EEG file.
    preload : bool
        If True, load data into memory (required for most operations).
    montage_name : str
        Standard montage to apply if the file lacks electrode positions.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw object with standard montage applied.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".edf":
        raw = mne.io.read_raw_edf(filepath, preload=preload, verbose=False)
    elif ext == ".fif":
        raw = mne.io.read_raw_fif(filepath, preload=preload, verbose=False)
    elif ext in (".vhdr", ".vmrk"):
        raw = mne.io.read_raw_brainvision(filepath, preload=preload, verbose=False)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(filepath, preload=preload, verbose=False)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(filepath, preload=preload, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # Apply montage if no digitization points present
    if not raw.info.get("dig"):
        montage = mne.channels.make_standard_montage(montage_name)
        # Only set channels that exist in the montage
        available = set(montage.ch_names)
        raw.pick_channels([c for c in raw.ch_names if c in available], ordered=False)
        raw.set_montage(montage, on_missing="ignore")

    print(
        f"Loaded: {filepath}\n"
        f"  Channels: {len(raw.ch_names)}, "
        f"  Duration: {raw.times[-1]:.1f}s, "
        f"  Sfreq: {raw.info['sfreq']:.0f} Hz"
    )
    return raw


# ---------------------------------------------------------------------------
# 2. Preprocessing Pipeline
# ---------------------------------------------------------------------------


def preprocess_raw(
    raw: mne.io.BaseRaw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float | list[float] = 50.0,
    reference: str = "average",
    bad_channels: list[str] | None = None,
    resample_sfreq: float | None = None,
) -> mne.io.BaseRaw:
    """
    Apply standard EEG preprocessing: notch filter, bandpass, re-reference, interpolation.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input raw object (must be preloaded).
    l_freq : float
        High-pass cutoff (Hz). Use 0.1 for ERP, 1.0 for general use.
    h_freq : float
        Low-pass cutoff (Hz). Use 40 for ERPs, 100+ for oscillations.
    notch_freq : float or list of float
        Notch filter frequency (50 Hz Europe, 60 Hz North America).
    reference : str
        Re-referencing scheme: ``"average"``, ``"REST"``, or a channel name.
    bad_channels : list of str, optional
        Channels to mark as bad before interpolation.
    resample_sfreq : float, optional
        If provided, resample to this frequency after filtering.

    Returns
    -------
    mne.io.BaseRaw
        Preprocessed raw object.
    """
    raw = raw.copy()

    # Mark bad channels
    if bad_channels:
        raw.info["bads"] = bad_channels
        print(f"Marked bad: {bad_channels}")

    # Notch filter (power line noise)
    notch_freqs = [notch_freq] if isinstance(notch_freq, (int, float)) else notch_freq
    # Also remove harmonics
    all_notch = []
    for f in notch_freqs:
        harmonics = [f * k for k in range(1, int(raw.info["sfreq"] // (2 * f)) + 1)]
        all_notch.extend(harmonics)
    raw.notch_filter(freqs=all_notch, fir_window="hamming", verbose=False)

    # Bandpass filter
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_window="hamming", verbose=False)
    print(f"Filtered: {l_freq}–{h_freq} Hz, notch at {notch_freqs}")

    # Re-reference
    if reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose=False)
    elif reference == "REST":
        raw.set_eeg_reference("REST", verbose=False)
    else:
        raw.set_eeg_reference([reference], verbose=False)

    # Interpolate bad channels
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose=False)
        print("Interpolated bad channels.")

    # Optional resample
    if resample_sfreq and resample_sfreq != raw.info["sfreq"]:
        raw.resample(resample_sfreq, verbose=False)
        print(f"Resampled to {resample_sfreq} Hz")

    return raw


# ---------------------------------------------------------------------------
# 3. ICA Artifact Removal
# ---------------------------------------------------------------------------


def run_ica_artifact_removal(
    raw: mne.io.BaseRaw,
    n_components: int | float = 0.99,
    method: str = "fastica",
    eog_channels: list[str] | None = None,
    ecg_channel: str | None = None,
    random_state: int = 42,
) -> tuple[mne.io.BaseRaw, ICA]:
    """
    Run ICA and automatically identify eye/heart artifact components.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw (high-pass ≥ 1 Hz recommended for ICA stability).
    n_components : int or float
        Number of ICA components. Float (0–1) = variance explained.
    method : str
        ICA algorithm: ``"fastica"``, ``"infomax"``, ``"picard"``.
    eog_channels : list of str, optional
        EOG channel names for eye artifact detection. If None, MNE auto-detects.
    ecg_channel : str, optional
        ECG channel name for heart artifact detection.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple : (cleaned raw, fitted ICA object)
    """
    # ICA requires 1 Hz high-pass; apply temporarily if needed
    raw_for_ica = raw.copy()
    if raw.info.get("highpass", 0) < 1.0:
        raw_for_ica.filter(l_freq=1.0, h_freq=None, verbose=False)

    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw_for_ica, verbose=False)
    print(f"ICA fitted: {ica.n_components_} components")

    exclude_indices = []

    # Eye artifact detection
    eog_chs = eog_channels or [c for c in raw.ch_names if c.upper().startswith("EOG")]
    if eog_chs:
        eog_indices, eog_scores = ica.find_bads_eog(raw_for_ica, ch_name=eog_chs, verbose=False)
        exclude_indices.extend(eog_indices)
        print(f"EOG components found: {eog_indices}")
    else:
        # Fallback: use Fp1/Fp2 as proxy EOG
        fp_chs = [c for c in raw.ch_names if c in ("Fp1", "Fp2", "FP1", "FP2")]
        if fp_chs:
            eog_indices, _ = ica.find_bads_eog(raw_for_ica, ch_name=fp_chs[:1], verbose=False)
            exclude_indices.extend(eog_indices)

    # ECG artifact detection
    if ecg_channel and ecg_channel in raw.ch_names:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw_for_ica, ch_name=ecg_channel, verbose=False)
        exclude_indices.extend(ecg_indices)
        print(f"ECG components found: {ecg_indices}")

    ica.exclude = list(set(exclude_indices))
    print(f"Excluding {len(ica.exclude)} ICA components: {ica.exclude}")

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean, ica


# ---------------------------------------------------------------------------
# 4. Epoching
# ---------------------------------------------------------------------------


def epoch_events(
    raw: mne.io.BaseRaw,
    event_id: dict[str, int],
    tmin: float = -0.2,
    tmax: float = 0.8,
    baseline: tuple | None = (None, 0),
    reject: dict | None = None,
    picks: str = "eeg",
    stim_channel: str | None = None,
) -> mne.Epochs:
    """
    Extract epochs around events from a raw recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Cleaned raw object.
    event_id : dict
        Mapping of condition labels to trigger codes, e.g. ``{"standard": 1, "deviant": 2}``.
    tmin, tmax : float
        Epoch time window in seconds relative to event onset.
    baseline : tuple or None
        Baseline correction window, e.g. ``(None, 0)`` = pre-stimulus.
    reject : dict, optional
        Peak-to-peak rejection thresholds, e.g. ``{"eeg": 100e-6}``.
    picks : str
        Channel types to include.
    stim_channel : str, optional
        Stimulus channel name. If None, MNE auto-detects.

    Returns
    -------
    mne.Epochs
    """
    if reject is None:
        reject = {"eeg": 150e-6}  # 150 µV threshold

    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    if len(events) == 0 and stim_channel:
        events = mne.find_events(raw, stim_channel=stim_channel, verbose=False)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        picks=picks,
        preload=True,
        verbose=False,
    )
    print(f"Epochs: {len(epochs)} retained ({epochs.rejection_drop_percentage:.1f}% rejected)")
    return epochs


# ---------------------------------------------------------------------------
# 5. ERP Analysis
# ---------------------------------------------------------------------------


def compute_erp(
    epochs: mne.Epochs,
    conditions: list[str] | None = None,
    channels: list[str] | None = None,
) -> dict[str, mne.Evoked]:
    """
    Compute evoked (ERP) responses for each condition.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data.
    conditions : list of str, optional
        Subset of conditions to compute. Defaults to all keys in ``epochs.event_id``.
    channels : list of str, optional
        Channels to pick before averaging (e.g. ``["Cz", "Pz", "Fz"]``).

    Returns
    -------
    dict mapping condition name to mne.Evoked object.
    """
    conditions = conditions or list(epochs.event_id.keys())
    evokeds = {}
    for cond in conditions:
        if cond not in epochs.event_id:
            print(f"Warning: condition '{cond}' not found in epochs.")
            continue
        ep = epochs[cond]
        if channels:
            ep = ep.copy().pick_channels(channels)
        evoked = ep.average()
        evoked.comment = cond
        evokeds[cond] = evoked
        print(f"ERP '{cond}': {len(ep)} trials averaged.")
    return evokeds


def plot_erp_comparison(
    evokeds: dict[str, mne.Evoked],
    channel: str = "Cz",
    title: str | None = None,
    figsize: tuple = (10, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot ERP waveforms for multiple conditions on a single axis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set1(np.linspace(0, 1, len(evokeds)))

    for (cond, evoked), color in zip(evokeds.items(), colors):
        try:
            idx = evoked.ch_names.index(channel)
        except ValueError:
            print(f"Channel {channel} not found; skipping {cond}.")
            continue
        data = evoked.data[idx] * 1e6  # Convert V → µV
        times = evoked.times * 1e3      # Convert s → ms
        ax.plot(times, data, label=cond, color=color, linewidth=2)

    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.invert_yaxis()  # EEG convention: negative up
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Amplitude (µV)", fontsize=12)
    ax.set_title(title or f"ERP at {channel}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# 6. Time-Frequency Analysis
# ---------------------------------------------------------------------------


def compute_tfr_morlet(
    epochs: mne.Epochs,
    freqs: np.ndarray | None = None,
    n_cycles: np.ndarray | int = 7,
    picks: list[str] | None = None,
    return_itc: bool = True,
    average: bool = True,
    decim: int = 4,
) -> tuple:
    """
    Compute time-frequency representation using Morlet wavelets.

    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs.
    freqs : array-like, optional
        Frequencies to analyze (Hz). Defaults to 2–40 Hz in log space.
    n_cycles : array or int
        Number of wavelet cycles. Int = fixed; array = frequency-dependent.
    picks : list of str, optional
        Channels to include. Defaults to all EEG channels.
    return_itc : bool
        If True, also return inter-trial coherence (phase locking).
    average : bool
        If True, return average TFR; if False, return single-trial TFR.
    decim : int
        Temporal decimation factor to reduce memory.

    Returns
    -------
    tuple : (AverageTFR power, AverageTFR itc) if return_itc else (AverageTFR power,)
    """
    if freqs is None:
        freqs = np.logspace(np.log10(2), np.log10(40), 30)

    if isinstance(n_cycles, int):
        n_cycles_arr = np.full_like(freqs, n_cycles, dtype=float)
    else:
        n_cycles_arr = np.asarray(n_cycles, dtype=float)

    ep = epochs.copy()
    if picks:
        ep = ep.pick_channels(picks)

    power, itc = tfr_morlet(
        ep,
        freqs=freqs,
        n_cycles=n_cycles_arr,
        use_fft=True,
        return_itc=return_itc,
        decim=decim,
        average=average,
        verbose=False,
    )
    print(f"TFR computed: {len(freqs)} freqs × {power.data.shape[-1]} timepoints")
    return (power, itc) if return_itc else (power,)


# ---------------------------------------------------------------------------
# 7. Power Spectral Density
# ---------------------------------------------------------------------------


def compute_band_power(
    raw: mne.io.BaseRaw,
    picks: list[str] | None = None,
    fmin: float = 1.0,
    fmax: float = 50.0,
    method: str = "welch",
) -> pd.DataFrame:
    """
    Compute PSD and return band-averaged power (delta, theta, alpha, beta, gamma).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input raw (preloaded).
    picks : list of str, optional
        Channel names; defaults to all EEG channels.
    fmin, fmax : float
        Frequency range for PSD computation.
    method : str
        ``"welch"`` or ``"multitaper"``.

    Returns
    -------
    pd.DataFrame with columns: channel, delta, theta, alpha, beta, gamma (all in µV²/Hz).
    """
    BANDS = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    spectrum = raw.compute_psd(method=method, fmin=fmin, fmax=fmax, picks=picks or "eeg",
                               verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds_uv2 = psds * 1e12  # V²/Hz → µV²/Hz
    ch_names = spectrum.ch_names

    records = []
    for i, ch in enumerate(ch_names):
        row = {"channel": ch}
        for band, (lo, hi) in BANDS.items():
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            row[band] = np.trapz(psds_uv2[i, idx], freqs[idx])
        records.append(row)

    return pd.DataFrame(records)
```

---

## Example A: Auditory ERP (N100 / P300) from Oddball Paradigm

An auditory oddball paradigm presents frequent "standard" tones (80%) and rare "deviant" tones
(20%). This example preprocesses the raw data, extracts ERPs, and quantifies the N100 (auditory
response at ~100 ms) and P300 (cognitive response at ~300 ms on Pz).

```python
# ── Example A ─────────────────────────────────────────────────────────────
# Uses MNE's built-in sample dataset (auditory oddball data)

import mne
import numpy as np
import matplotlib.pyplot as plt

# --- Download MNE sample data (if needed) ------------------------------------
data_path = mne.datasets.sample.data_path()
raw_fif = str(data_path) + "/MEG/sample/sample_audvis_raw.fif"

# --- Load (use only EEG channels) -------------------------------------------
raw = mne.io.read_raw_fif(raw_fif, preload=True, verbose=False)
raw.pick_types(eeg=True, eog=True, stim=True)
print(raw.info)

# --- Preprocess --------------------------------------------------------------
raw_prep = preprocess_raw(
    raw,
    l_freq=0.5,
    h_freq=40.0,
    notch_freq=60.0,       # 60 Hz (North American power line)
    reference="average",
    bad_channels=["EEG 053"],  # Example bad channel
    resample_sfreq=None,
)

# --- ICA artifact removal ---------------------------------------------------
raw_clean, ica = run_ica_artifact_removal(
    raw_prep,
    n_components=20,
    method="fastica",
    eog_channels=["EOG 061"],
    random_state=0,
)

# --- Epoch around auditory events -------------------------------------------
# MNE sample dataset: event codes 1=LA, 2=RA, 3=LV, 4=RV, 5=smiley, 32=button
event_id = {"auditory/left": 1, "auditory/right": 2}
epochs = epoch_events(
    raw_clean,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.5,
    baseline=(-0.2, 0),
    reject={"eeg": 100e-6},
)

# --- Compute ERPs -----------------------------------------------------------
evokeds = compute_erp(epochs, conditions=["auditory/left", "auditory/right"])

# --- Plot ERPs at Cz ---------------------------------------------------------
fig = plot_erp_comparison(
    evokeds,
    channel="EEG 059",  # Approximately Cz in the sample data
    title="Auditory ERP: Left vs Right (Oddball Paradigm)",
    save_path="auditory_erp.png",
)
plt.show()

# --- Quantify N100 and P300 --------------------------------------------------
def peak_amplitude_latency(evoked: mne.Evoked, channel: str, tmin: float, tmax: float) -> dict:
    """Return peak amplitude (µV) and latency (ms) within a time window."""
    try:
        idx = evoked.ch_names.index(channel)
    except ValueError:
        return {"amplitude_uV": np.nan, "latency_ms": np.nan}
    t_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    data = evoked.data[idx, t_mask] * 1e6
    times = evoked.times[t_mask] * 1e3
    peak_idx = np.argmax(np.abs(data))
    return {"amplitude_uV": float(data[peak_idx]), "latency_ms": float(times[peak_idx])}


TARGET_CH = "EEG 059"
for cond, evoked in evokeds.items():
    n100 = peak_amplitude_latency(evoked, TARGET_CH, 0.070, 0.150)
    p300 = peak_amplitude_latency(evoked, TARGET_CH, 0.250, 0.450)
    print(f"\n{cond}")
    print(f"  N100: {n100['amplitude_uV']:.2f} µV @ {n100['latency_ms']:.0f} ms")
    print(f"  P300: {p300['amplitude_uV']:.2f} µV @ {p300['latency_ms']:.0f} ms")

# --- Difference wave (deviant minus standard) --------------------------------
# If you have a full oddball dataset with standard/deviant labels:
# diff_wave = mne.combine_evoked([evokeds["deviant"], evokeds["standard"]], weights=[1, -1])
# diff_wave.plot(picks=[TARGET_CH], titles={"eeg": "MMN / Difference Wave"})

# --- Topographic map at P300 peak (300–400 ms) -------------------------------
for cond, evoked in evokeds.items():
    fig_topo = evoked.plot_topomap(
        times=[0.1, 0.2, 0.3, 0.4],
        average=0.05,
        show=False,
    )
    fig_topo.suptitle(f"Topography: {cond}", fontsize=11)
    fig_topo.savefig(f"topo_{cond.replace('/', '_')}.png", dpi=120)
```

---

## Example B: Resting-State Alpha Power Between Eyes-Open and Eyes-Closed

Alpha band power (8–13 Hz) reliably increases during eyes-closed rest. This example compares
band power across conditions and produces a power spectral density plot.

```python
# ── Example B ─────────────────────────────────────────────────────────────
# Assumes two raw files: resting_eyes_open.edf and resting_eyes_closed.edf
# Replace paths with actual file locations

EYES_OPEN_FILE = os.environ.get("EYES_OPEN_EDF", "resting_eyes_open.edf")
EYES_CLOSED_FILE = os.environ.get("EYES_CLOSED_EDF", "resting_eyes_closed.edf")

OCCIPITAL_CHANNELS = ["O1", "Oz", "O2", "PO3", "PO4", "PO7", "PO8"]

results = {}
psds_dict = {}
spectra_dict = {}

for label, filepath in [("eyes_open", EYES_OPEN_FILE), ("eyes_closed", EYES_CLOSED_FILE)]:
    if not os.path.exists(filepath):
        print(f"Skipping {label}: file not found ({filepath})")
        continue

    raw = load_raw_eeg(filepath)
    raw = preprocess_raw(raw, l_freq=1.0, h_freq=50.0, notch_freq=50.0)
    raw_clean, _ = run_ica_artifact_removal(raw, n_components=0.99)

    # Keep only available occipital channels
    occ_available = [c for c in OCCIPITAL_CHANNELS if c in raw_clean.ch_names]
    if not occ_available:
        occ_available = None  # Fall back to all EEG

    band_df = compute_band_power(raw_clean, picks=occ_available)
    band_df["condition"] = label
    results[label] = band_df

    # Full PSD for plotting
    spectrum = raw_clean.compute_psd(method="welch", fmin=1.0, fmax=50.0,
                                     picks=occ_available or "eeg", verbose=False)
    spectra_dict[label] = spectrum

# --- Statistical comparison --------------------------------------------------
if len(results) == 2:
    from scipy.stats import ttest_rel, wilcoxon

    eo = results["eyes_open"]["alpha"].values
    ec = results["eyes_closed"]["alpha"].values

    if len(eo) == len(ec):
        t_stat, p_val = ttest_rel(ec, eo)
        print(f"\nAlpha power comparison (occipital channels):")
        print(f"  Eyes open:   {eo.mean():.2f} ± {eo.std():.2f} µV²/Hz")
        print(f"  Eyes closed: {ec.mean():.2f} ± {ec.std():.2f} µV²/Hz")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

# --- PSD plot comparison -----------------------------------------------------
if spectra_dict:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"eyes_open": "steelblue", "eyes_closed": "firebrick"}

    for label, spectrum in spectra_dict.items():
        psds, freqs = spectrum.get_data(return_freqs=True)
        mean_psd = psds.mean(axis=0) * 1e12  # µV²/Hz
        se_psd = psds.std(axis=0) / np.sqrt(len(psds)) * 1e12
        ax.semilogy(freqs, mean_psd, label=label, color=colors.get(label, "gray"), linewidth=2)
        ax.fill_between(
            freqs,
            mean_psd - se_psd,
            mean_psd + se_psd,
            alpha=0.2,
            color=colors.get(label, "gray"),
        )

    # Shade alpha band
    ax.axvspan(8, 13, alpha=0.12, color="gold", label="Alpha band (8–13 Hz)")
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density (µV²/Hz)", fontsize=12)
    ax.set_title("Resting-State PSD: Eyes Open vs Eyes Closed (Occipital)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plt.savefig("alpha_power_psd.png", dpi=150)
    plt.show()

# --- Time-frequency on resting-state (Eyes Closed) -------------------------
# Segment resting state into 2-second pseudo-epochs for TFR
if "eyes_closed" in results:
    raw_ec = load_raw_eeg(EYES_CLOSED_FILE)
    raw_ec = preprocess_raw(raw_ec, l_freq=1.0, h_freq=50.0)
    raw_clean_ec, _ = run_ica_artifact_removal(raw_ec, n_components=0.99)

    # Create fixed-length epochs (2 s, no overlap)
    epochs_rest = mne.make_fixed_length_epochs(
        raw_clean_ec, duration=2.0, preload=True, verbose=False
    )
    epochs_rest.pick_channels(
        [c for c in OCCIPITAL_CHANNELS if c in epochs_rest.ch_names] or epochs_rest.ch_names[:4]
    )

    freqs_tfr = np.arange(4, 30, 1)
    n_cycles_tfr = freqs_tfr / 2.0  # 0.5 cycles per Hz
    power, itc = compute_tfr_morlet(
        epochs_rest,
        freqs=freqs_tfr,
        n_cycles=n_cycles_tfr,
        return_itc=True,
        decim=4,
    )

    # Plot average TFR
    fig_tfr = power.plot(
        picks=[0],
        baseline=None,
        mode="logratio",
        title="Resting-State TFR (Eyes Closed)",
        show=False,
    )
    fig_tfr[0].savefig("resting_tfr_eyes_closed.png", dpi=120)
```

---

## Notes and Best Practices

### File Format Recommendations

- Store processed data in `.fif` format (raw, epochs, evoked) for lossless round-tripping.
- Use `epochs.save("sub-01_task-oddball-epo.fif", overwrite=True)` and reload with
  `mne.read_epochs("sub-01_task-oddball-epo.fif")`.

### ICA Stability

ICA is sensitive to the high-pass filter. Always high-pass at 1 Hz before fitting ICA, even
if your final analysis uses a lower cutoff (0.1 Hz for ERP). Apply ICA to the original
low-passed raw after fitting on the 1 Hz-filtered version.

### Rejection Thresholds

Typical peak-to-peak thresholds:

| System | Threshold |
|---|---|
| High-density (128+ ch) | 100 µV |
| Standard (32–64 ch) | 150 µV |
| Clinical/noisy | 200–250 µV |

### Source Localization (Minimum Norm)

```python
# Requires FreeSurfer reconstructed subject or fsaverage
subjects_dir = mne.datasets.sample.data_path() + "/subjects"
fwd = mne.read_forward_solution("sample-fwd.fif")
noise_cov = mne.compute_covariance(epochs, tmax=0)
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
stc = mne.minimum_norm.apply_inverse(evokeds["auditory/left"], inv, lambda2=1.0 / 9.0)
stc.plot(subjects_dir=subjects_dir, hemi="both", initial_time=0.1)
```

### References

- Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in
  Neuroscience*, 7, 267.
- Delorme, A., & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial
  EEG dynamics. *Journal of Neuroscience Methods*, 134(1), 9–21.
- Makeig, S., et al. (1996). Independent component analysis of electroencephalographic data.
  *Advances in Neural Information Processing Systems*, 8.
