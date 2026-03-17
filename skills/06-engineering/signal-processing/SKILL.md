---
name: signal-processing
description: >
  Digital signal processing with scipy.signal: filter design, spectral analysis,
  peak detection, PSD estimation, and adaptive filtering for engineering workflows.
tags:
  - signal-processing
  - scipy
  - dsp
  - filtering
  - spectral-analysis
  - engineering
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
  - numpy>=1.24.0
  - scipy>=1.11.0
  - matplotlib>=3.7.0
  - pandas>=2.0.0
last_updated: "2026-03-17"
---

# Signal Processing

A comprehensive skill for digital signal processing (DSP) using `scipy.signal`.
Covers FIR/IIR filter design, time-series filtering, spectral analysis, peak
detection, power spectral density estimation, cross-correlation, matched filters,
and adaptive filtering. Designed for real-world applications such as biomedical
signal processing and vibration/fault analysis.

---

## Core Functions

### 1. Filter Design

```python
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def design_bandpass_filter(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
    filter_type: str = "butter",
    rp: float = 1.0,
    rs: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Design a bandpass IIR or FIR filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Filter order (default 4).
    filter_type : str
        One of 'butter', 'cheby1', 'cheby2', 'ellip', 'firwin'.
    rp : float
        Maximum ripple in the passband (dB), used by cheby1/ellip.
    rs : float
        Minimum attenuation in the stopband (dB), used by cheby2/ellip.

    Returns
    -------
    b, a : array_like
        Numerator and denominator polynomials of the IIR filter.
        For FIR (firwin), a = [1.0].
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if filter_type == "butter":
        b, a = signal.butter(order, [low, high], btype="bandpass")
    elif filter_type == "cheby1":
        b, a = signal.cheby1(order, rp, [low, high], btype="bandpass")
    elif filter_type == "cheby2":
        b, a = signal.cheby2(order, rs, [low, high], btype="bandpass")
    elif filter_type == "ellip":
        b, a = signal.ellip(order, rp, rs, [low, high], btype="bandpass")
    elif filter_type == "firwin":
        # FIR bandpass via window method; order must be even for bandpass
        if order % 2 != 0:
            order += 1
        b = signal.firwin(order + 1, [low, high], pass_zero=False)
        a = np.array([1.0])
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type!r}")

    return b, a


def design_notch_filter(
    freq: float, fs: float, quality: float = 30.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Design a notch (band-stop) filter to remove a single frequency.

    Parameters
    ----------
    freq : float
        Frequency to attenuate in Hz.
    fs : float
        Sampling frequency in Hz.
    quality : float
        Quality factor Q = freq / bandwidth.

    Returns
    -------
    b, a : array_like
    """
    w0 = freq / (0.5 * fs)
    b, a = signal.iirnotch(w0, quality)
    return b, a


def design_lowpass_filter(
    cutoff: float, fs: float, order: int = 4, filter_type: str = "butter"
) -> tuple[np.ndarray, np.ndarray]:
    """Design a lowpass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if filter_type == "butter":
        b, a = signal.butter(order, normal_cutoff, btype="low")
    elif filter_type == "firwin":
        if order % 2 == 0:
            order += 1
        b = signal.firwin(order, normal_cutoff)
        a = np.array([1.0])
    else:
        raise ValueError(f"Unsupported filter_type: {filter_type!r}")
    return b, a
```

### 2. Applying Filters

```python
def apply_filter(
    data: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    method: str = "sosfilt",
) -> np.ndarray:
    """
    Apply a digital filter to a 1-D signal.

    Parameters
    ----------
    data : np.ndarray
        Input signal array (1-D).
    b : np.ndarray
        Numerator coefficients.
    a : np.ndarray
        Denominator coefficients.
    method : str
        'sosfilt' (recommended, numerically stable) or 'filtfilt' (zero-phase).

    Returns
    -------
    np.ndarray
        Filtered signal with the same length as `data`.
    """
    if method == "sosfilt":
        sos = signal.tf2sos(b, a)
        return signal.sosfiltfilt(sos, data)
    elif method == "filtfilt":
        return signal.filtfilt(b, a, data)
    elif method == "lfilter":
        return signal.lfilter(b, a, data)
    else:
        raise ValueError(f"Unknown method: {method!r}")


def apply_envelope_detection(
    data: np.ndarray, fs: float, lowpass_cutoff: float = 10.0
) -> np.ndarray:
    """
    Compute the amplitude envelope using the Hilbert transform followed by
    a lowpass filter.

    Parameters
    ----------
    data : np.ndarray
        Bandpass-filtered signal.
    fs : float
        Sampling frequency in Hz.
    lowpass_cutoff : float
        Cutoff for the smoothing lowpass filter (Hz).

    Returns
    -------
    np.ndarray
        Smoothed amplitude envelope.
    """
    analytic = signal.hilbert(data)
    envelope = np.abs(analytic)
    b, a = design_lowpass_filter(lowpass_cutoff, fs, order=4)
    return apply_filter(envelope, b, a)
```

### 3. Spectral Analysis

```python
def compute_spectrogram(
    data: np.ndarray,
    fs: float,
    window: str = "hann",
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform (STFT) spectrogram.

    Parameters
    ----------
    data : np.ndarray
        Input time-series signal.
    fs : float
        Sampling frequency in Hz.
    window : str
        Window function name (e.g. 'hann', 'hamming', 'blackman').
    nperseg : int
        Length of each STFT segment.
    noverlap : int or None
        Number of overlapping samples. Defaults to nperseg // 2.

    Returns
    -------
    f : np.ndarray
        Frequency bins (Hz).
    t : np.ndarray
        Time bins (seconds).
    Sxx : np.ndarray
        Power spectrogram in dB (shape: [freq_bins, time_bins]).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = signal.stft(data, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    Sxx = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return f, t, Sxx


def compute_psd_welch(
    data: np.ndarray,
    fs: float,
    nperseg: int = 512,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate power spectral density using Welch's method.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Length of each Welch segment.
    window : str
        Window function name.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins (Hz).
    psd : np.ndarray
        Power spectral density (V²/Hz).
    """
    freqs, psd = signal.welch(data, fs=fs, window=window, nperseg=nperseg)
    return freqs, psd


def plot_frequency_response(
    b: np.ndarray,
    a: np.ndarray,
    fs: float,
    title: str = "Filter Frequency Response",
) -> None:
    """Plot magnitude and phase response using freqz."""
    w, h = signal.freqz(b, a, worN=8000, fs=fs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title(title)
    ax1.grid(True)
    ax2.plot(w, np.angle(h, deg=True))
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
```

### 4. Peak Detection and Correlation

```python
def detect_peaks(
    data: np.ndarray,
    min_height: float | None = None,
    min_distance: int = 1,
    prominence: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Detect peaks in a 1-D signal using scipy.signal.find_peaks.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    min_height : float or None
        Minimum peak height.
    min_distance : int
        Minimum number of samples between peaks.
    prominence : float or None
        Minimum peak prominence.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks.
    properties : dict
        Peak properties (heights, prominences, etc.).
    """
    kwargs: dict = {"distance": min_distance}
    if min_height is not None:
        kwargs["height"] = min_height
    if prominence is not None:
        kwargs["prominence"] = prominence
    peaks, properties = signal.find_peaks(data, **kwargs)
    return peaks, properties


def compute_cross_correlation(
    x: np.ndarray, y: np.ndarray, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute full cross-correlation between two signals.

    Returns
    -------
    lags : np.ndarray
        Lag values in samples.
    corr : np.ndarray
        Cross-correlation coefficients.
    """
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    if normalize:
        corr = corr / (np.std(x) * np.std(y) * len(x))
    return lags, corr


def matched_filter(template: np.ndarray, noisy_signal: np.ndarray) -> np.ndarray:
    """
    Apply a matched filter (cross-correlate signal with time-reversed template).

    Returns
    -------
    np.ndarray
        Matched filter output (same length as noisy_signal).
    """
    h = template[::-1]
    return signal.convolve(noisy_signal, h, mode="same")
```

### 5. Adaptive Filtering (LMS)

```python
def lms_adaptive_filter(
    desired: np.ndarray,
    input_signal: np.ndarray,
    filter_order: int = 32,
    mu: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Least Mean Squares (LMS) adaptive filter.

    Parameters
    ----------
    desired : np.ndarray
        Desired (reference) signal.
    input_signal : np.ndarray
        Input (noisy) signal to filter.
    filter_order : int
        Number of filter taps.
    mu : float
        Step size (learning rate). Must be small for stability (< 1 / (filter_order * P_x)).

    Returns
    -------
    output : np.ndarray
        Filtered output signal.
    error : np.ndarray
        Error signal (desired - output).
    weights : np.ndarray
        Final filter weight vector.
    """
    n = len(input_signal)
    weights = np.zeros(filter_order)
    output = np.zeros(n)
    error = np.zeros(n)

    for i in range(filter_order, n):
        x = input_signal[i - filter_order : i][::-1]
        y = np.dot(weights, x)
        e = desired[i] - y
        weights += 2 * mu * e * x
        output[i] = y
        error[i] = e

    return output, error, weights
```

---

## Example 1: EMG Signal Processing Pipeline

This example simulates an electromyography (EMG) recording and runs a full
processing pipeline: bandpass filtering, power-line notch removal, and
amplitude envelope extraction.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ── Simulate a noisy EMG signal ───────────────────────────────────────────────
FS = 2000          # Hz
DURATION = 5.0     # seconds
T = np.arange(0, DURATION, 1.0 / FS)

# Simulated EMG: broadband noise modulated by a slow activation pattern
rng = np.random.default_rng(42)
activation = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * T))   # slow contraction
emg_raw = activation * rng.standard_normal(len(T))

# Add 50 Hz power-line interference
emg_raw += 0.3 * np.sin(2 * np.pi * 50 * T)
# Add 150 Hz harmonic
emg_raw += 0.1 * np.sin(2 * np.pi * 150 * T)
# Add low-frequency motion artifact
emg_raw += 0.4 * np.sin(2 * np.pi * 1.5 * T)

# ── Step 1: Bandpass 20–450 Hz ────────────────────────────────────────────────
b_bp, a_bp = design_bandpass_filter(20, 450, FS, order=4, filter_type="butter")
emg_bp = apply_filter(emg_raw, b_bp, a_bp)

# ── Step 2: Notch at 50 Hz and 150 Hz ────────────────────────────────────────
b_n1, a_n1 = design_notch_filter(50, FS, quality=30)
emg_notch = apply_filter(emg_bp, b_n1, a_n1)
b_n2, a_n2 = design_notch_filter(150, FS, quality=30)
emg_notch = apply_filter(emg_notch, b_n2, a_n2)

# ── Step 3: Amplitude envelope ────────────────────────────────────────────────
emg_envelope = apply_envelope_detection(emg_notch, FS, lowpass_cutoff=8.0)

# ── Step 4: Detect activation onset/offset peaks ─────────────────────────────
peaks, props = detect_peaks(emg_envelope, min_height=0.15, min_distance=int(FS * 0.5))

# ── Step 5: Compute PSD before and after filtering ───────────────────────────
freqs_raw, psd_raw = compute_psd_welch(emg_raw, FS, nperseg=512)
freqs_filt, psd_filt = compute_psd_welch(emg_notch, FS, nperseg=512)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)

axes[0].plot(T, emg_raw, linewidth=0.5)
axes[0].set_title("Raw EMG Signal")
axes[0].set_ylabel("Amplitude (mV)")

axes[1].plot(T, emg_notch, linewidth=0.5, color="C1")
axes[1].set_title("Filtered EMG (Bandpass + Notch)")
axes[1].set_ylabel("Amplitude (mV)")

axes[2].plot(T, emg_envelope, color="C2")
axes[2].plot(T[peaks], emg_envelope[peaks], "rx", markersize=8, label="Activations")
axes[2].set_title("EMG Amplitude Envelope")
axes[2].set_ylabel("Amplitude (mV)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

# PSD comparison
axes_psd = fig.add_axes([0.12, 0.02, 0.76, 0.18])
axes_psd.semilogy(freqs_raw, psd_raw, label="Raw PSD", alpha=0.7)
axes_psd.semilogy(freqs_filt, psd_filt, label="Filtered PSD", alpha=0.7)
axes_psd.set_xlabel("Frequency (Hz)")
axes_psd.set_ylabel("PSD (V²/Hz)")
axes_psd.set_title("Power Spectral Density Comparison")
axes_psd.legend()
axes_psd.grid(True, which="both")

plt.tight_layout()
plt.savefig("emg_processing_pipeline.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Detected {len(peaks)} activation events at samples: {peaks}")
print(f"Peak heights: {props['peak_heights'].round(3)}")
```

---

## Example 2: Vibration Analysis for Rotating Machinery

This example performs fault-frequency detection on a vibration signal from a
rotating machine. It uses FFT, spectrogram, and automated peak detection to
identify bearing defect frequencies.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq

# ── Machine parameters ────────────────────────────────────────────────────────
FS = 10_000      # Sampling rate (Hz)
DURATION = 2.0   # seconds
RPM = 1500
SHAFT_FREQ = RPM / 60           # 25 Hz
# Outer-race bearing defect frequency (BPFO) for a 6-ball bearing
BPFO = 3.585 * SHAFT_FREQ       # ≈ 89.6 Hz

T = np.arange(0, DURATION, 1.0 / FS)
rng = np.random.default_rng(0)

# Simulate clean vibration: shaft harmonics + bearing defect impacts
vib = np.zeros_like(T)
for k in range(1, 6):
    vib += (1.0 / k) * np.sin(2 * np.pi * k * SHAFT_FREQ * T)

# Add periodic impacts at BPFO (impulsive model)
impact_times = np.arange(0, DURATION, 1.0 / BPFO)
for t_imp in impact_times:
    idx = int(t_imp * FS)
    if idx < len(T):
        # Decaying exponential ring-down after impact
        t_local = T[idx : min(idx + 200, len(T))] - T[idx]
        vib[idx : min(idx + 200, len(T))] += 0.5 * np.exp(-500 * t_local) * np.sin(
            2 * np.pi * 2000 * t_local
        )

# Add broadband noise
vib += 0.15 * rng.standard_normal(len(T))

# ── FFT spectrum ──────────────────────────────────────────────────────────────
N = len(vib)
freqs_fft = rfftfreq(N, d=1.0 / FS)
spectrum = np.abs(rfft(vib)) * 2 / N

# ── Envelope spectrum (demodulation) ─────────────────────────────────────────
# Bandpass around resonance to isolate impacts, then envelope, then FFT
b_res, a_res = design_bandpass_filter(1500, 3000, FS, order=6)
vib_res = apply_filter(vib, b_res, a_res)
envelope = apply_envelope_detection(vib_res, FS, lowpass_cutoff=500)

env_spectrum = np.abs(rfft(envelope)) * 2 / N
env_freqs = rfftfreq(N, d=1.0 / FS)

# Detect fault frequency peaks in envelope spectrum
fault_peaks, fault_props = detect_peaks(
    env_spectrum,
    min_height=np.percentile(env_spectrum, 95),
    min_distance=int(FS / (2 * N)),
)

# ── Spectrogram ───────────────────────────────────────────────────────────────
f_sg, t_sg, Sxx = compute_spectrogram(vib, FS, window="hann", nperseg=512, noverlap=400)

# ── PSD ──────────────────────────────────────────────────────────────────────
freqs_w, psd_w = compute_psd_welch(vib, FS, nperseg=1024)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11))

# Time domain
axes[0].plot(T[:3000], vib[:3000], linewidth=0.5)
axes[0].set_title(f"Vibration Signal — Shaft: {SHAFT_FREQ} Hz, BPFO: {BPFO:.1f} Hz")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Acceleration (g)")

# Envelope spectrum
axes[1].plot(env_freqs[:int(N * 500 / FS)], env_spectrum[:int(N * 500 / FS)], linewidth=0.7)
axes[1].axvline(BPFO, color="r", linestyle="--", label=f"BPFO = {BPFO:.1f} Hz")
for k in range(1, 4):
    axes[1].axvline(k * BPFO, color="r", linestyle=":", alpha=0.5)
axes[1].set_title("Envelope Spectrum (Demodulated)")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].legend()

# Spectrogram
pcm = axes[2].pcolormesh(t_sg, f_sg[:200], Sxx[:200, :], shading="gouraud", cmap="inferno")
axes[2].set_title("Spectrogram (0–4 kHz)")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Frequency (Hz)")
plt.colorbar(pcm, ax=axes[2], label="Power (dB)")

plt.tight_layout()
plt.savefig("vibration_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# Report detected peaks near BPFO harmonics
tolerance = 5.0  # Hz
print(f"\nFault frequency detection (BPFO = {BPFO:.2f} Hz):")
for harmonic in range(1, 5):
    target = harmonic * BPFO
    nearby = [
        (env_freqs[p], env_spectrum[p])
        for p in fault_peaks
        if abs(env_freqs[p] - target) < tolerance
    ]
    status = "DETECTED" if nearby else "not found"
    print(f"  {harmonic}x BPFO ({target:.1f} Hz): {status}", end="")
    if nearby:
        freq, amp = nearby[0]
        print(f"  @ {freq:.2f} Hz, amplitude = {amp:.5f}")
    else:
        print()
```

---

## Notes and Best Practices

- **Filter stability**: Always use `sosfilt` / `sosfiltfilt` for high-order
  filters. Direct form II (`lfilter`) can suffer from numerical instability for
  orders above 6–8.
- **Zero-phase filtering**: `filtfilt` applies the filter twice (forward and
  backward), eliminating phase distortion but doubling the effective order.
  Avoid for online/real-time applications.
- **Nyquist constraint**: Cutoff frequencies must be strictly between 0 and
  `fs/2`. Always validate inputs before calling design functions.
- **FIR vs IIR**: FIR filters (firwin) are inherently stable and have linear
  phase, but require higher orders to achieve sharp transitions. IIR filters
  (butter, ellip) achieve steeper roll-off at lower order but introduce phase
  distortion.
- **LMS convergence**: The step size `mu` must satisfy `0 < mu < 1 / (M * Px)`
  where `M` is filter order and `Px` is the input signal power. Start small
  (e.g., `mu = 0.001`) and increase cautiously.
- **Window selection for STFT**: Hann window is a good default. Blackman-Harris
  offers lower sidelobes for detecting weak tones near strong ones.
- **Welch PSD segments**: Larger `nperseg` gives finer frequency resolution but
  fewer averages (higher variance). Typical values: `nperseg = fs` (1-second
  segments) for stationary signals.

---

## Dependencies Installation

```bash
pip install numpy>=1.24.0 scipy>=1.11.0 matplotlib>=3.7.0 pandas>=2.0.0
```
