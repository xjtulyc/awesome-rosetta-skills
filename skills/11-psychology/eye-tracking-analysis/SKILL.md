---
name: eye-tracking-analysis
description: >
  Use this Skill to analyze eye-tracking data: IVT fixation detection, AOI
  dwell time, scanpath similarity, heatmap generation, and pupillometry
  baseline correction.
tags:
  - psychology
  - eye-tracking
  - fixations
  - AOI
  - scanpath
  - pupillometry
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - pandas>=1.5
    - numpy>=1.23
    - scipy>=1.9
    - matplotlib>=3.6
    - sklearn>=1.2
last_updated: "2026-03-18"
status: stable
---

# Eye-Tracking Analysis

> **TL;DR** — Complete eye-tracking pipeline: blink detection and interpolation,
> IVT velocity-based fixation detection, AOI assignment with dwell-time metrics,
> Levenshtein scanpath similarity, Gaussian kernel density heatmaps, and
> event-locked pupillometry with baseline correction.

---

## When to Use

Use this Skill when you need to:

- Process raw gaze samples (x, y, timestamp, pupil diameter) from Tobii, SR Research,
  or SMI eye trackers
- Detect fixations using the I-VT (velocity threshold) algorithm
- Define AOIs as rectangles or polygons and extract per-AOI metrics
  (first fixation latency, dwell time, fixation count, proportion dwell)
- Compare scanpath sequences across participants or conditions using edit distance
- Generate fixation heatmaps for visual attention analysis
- Preprocess pupillometry data: blink removal, interpolation, and event-locked
  baseline correction

---

## Background

### Gaze Data Format

Raw gaze files typically have these columns:

| Column | Description |
|---|---|
| `timestamp_ms` | Sample time in milliseconds |
| `gaze_x` | Horizontal gaze coordinate (pixels from left) |
| `gaze_y` | Vertical gaze coordinate (pixels from top) |
| `pupil_diam` | Pupil diameter (mm or arbitrary units; 0 = blink) |

### IVT Fixation Detection

The I-VT algorithm classifies each sample as a fixation or saccade:

```
velocity_i = distance(sample_i, sample_{i-1}) / (t_i - t_{i-1})
```

Samples with velocity < threshold (~50 °/s or ~30 px/ms) are fixations.
Adjacent fixation samples are merged into fixation events.

### Pupillometry Preprocessing

1. Detect blinks (pupil_diam = 0 or NaN; ± 100 ms margin)
2. Linear interpolation for blinks < 150 ms
3. Baseline correction: subtract mean pupil diameter in a pre-stimulus window
4. Z-scoring across the trial or session

---

## Environment Setup

```bash
conda create -n eyetrack_env python=3.11 -y
conda activate eyetrack_env

pip install pandas>=1.5 numpy>=1.23 scipy>=1.9 matplotlib>=3.6 scikit-learn>=1.2

# Optional: shapely for polygon AOI
pip install shapely>=2.0

python -c "import pandas, numpy, scipy, matplotlib, sklearn; print('All OK')"
```

---

## Core Workflow

### Step 1 — Blink Detection and Interpolation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, ndimage
from typing import Optional, Dict, List, Tuple, Union

# Sample rate constant — adjust to your eye tracker
DEFAULT_SAMPLE_RATE_HZ = 1000  # 1000 Hz = 1 ms per sample


def detect_and_interpolate_blinks(
    df: pd.DataFrame,
    pupil_col: str = "pupil_diam",
    time_col: str = "timestamp_ms",
    max_blink_ms: float = 150.0,
    margin_ms: float = 50.0,
) -> pd.DataFrame:
    """
    Detect blinks (pupil_diam = 0 or NaN) and linearly interpolate short blinks.

    Blinks > max_blink_ms are left as NaN (genuine data loss).
    A margin is applied around each blink to remove partial blink artifacts.

    Args:
        df:           DataFrame sorted by timestamp with one row per gaze sample.
        pupil_col:    Pupil diameter column name.
        time_col:     Timestamp column (milliseconds).
        max_blink_ms: Maximum blink duration to interpolate (default 150 ms).
        margin_ms:    Samples within this margin of a blink are also marked invalid.

    Returns:
        DataFrame with 'pupil_clean' column (interpolated) and 'is_blink' bool.
    """
    df = df.copy().reset_index(drop=True)
    pupil = df[pupil_col].copy().astype(float)
    time = df[time_col].values.astype(float)

    # Mark blinks: pupil = 0 or NaN
    blink_mask = (pupil == 0) | pupil.isna()

    # Apply margin: also mark samples within margin_ms of blink
    blink_indices = np.where(blink_mask)[0]
    sample_dur = np.median(np.diff(time)) if len(time) > 1 else 1.0
    margin_samples = int(np.ceil(margin_ms / sample_dur))

    expanded_mask = blink_mask.copy()
    for idx in blink_indices:
        start = max(0, idx - margin_samples)
        end = min(len(pupil), idx + margin_samples + 1)
        expanded_mask.iloc[start:end] = True

    df["is_blink"] = expanded_mask
    pupil_clean = pupil.copy()
    pupil_clean[expanded_mask] = np.nan

    # Interpolate short blinks
    labeled, n_blinks = ndimage.label(expanded_mask.values)
    for blink_id in range(1, n_blinks + 1):
        blink_indices_seg = np.where(labeled == blink_id)[0]
        if len(blink_indices_seg) == 0:
            continue
        blink_duration_ms = (
            time[blink_indices_seg[-1]] - time[blink_indices_seg[0]]
        )
        if blink_duration_ms <= max_blink_ms:
            # Linear interpolation
            start_idx = max(0, blink_indices_seg[0] - 1)
            end_idx = min(len(pupil_clean) - 1, blink_indices_seg[-1] + 1)
            if not np.isnan(pupil_clean.iloc[start_idx]) and not np.isnan(pupil_clean.iloc[end_idx]):
                interp_vals = np.interp(
                    blink_indices_seg,
                    [start_idx, end_idx],
                    [pupil_clean.iloc[start_idx], pupil_clean.iloc[end_idx]],
                )
                pupil_clean.iloc[blink_indices_seg] = interp_vals

    df["pupil_clean"] = pupil_clean
    n_blinks_detected = int(n_blinks)
    pct_blink = expanded_mask.mean() * 100
    print(
        f"Blink detection: {n_blinks_detected} blinks, "
        f"{pct_blink:.1f}% of samples invalid"
    )
    return df
```

### Step 2 — IVT Fixation Detection

```python
def detect_fixations_ivt(
    df: pd.DataFrame,
    x_col: str = "gaze_x",
    y_col: str = "gaze_y",
    time_col: str = "timestamp_ms",
    velocity_threshold: float = 30.0,
    min_fixation_ms: float = 80.0,
    screen_width_px: int = 1920,
    screen_height_px: int = 1080,
    screen_width_cm: float = 53.0,
    viewing_distance_cm: float = 57.0,
) -> pd.DataFrame:
    """
    I-VT fixation detection from raw gaze samples.

    Velocity is computed in pixels/ms. An optional conversion to degrees/second
    is performed if screen geometry is provided.

    Fixation events are defined as contiguous runs of low-velocity samples
    exceeding min_fixation_ms.

    Args:
        df:                   Gaze DataFrame sorted by timestamp.
        x_col:                Horizontal gaze coordinate column.
        y_col:                Vertical gaze coordinate column.
        time_col:             Timestamp column (milliseconds).
        velocity_threshold:   Maximum velocity for fixation classification (px/ms).
        min_fixation_ms:      Minimum fixation duration (ms).
        screen_width_px:      Screen width in pixels.
        screen_height_px:     Screen height in pixels.
        screen_width_cm:      Physical screen width (cm) for degree conversion.
        viewing_distance_cm:  Viewing distance (cm) for degree conversion.

    Returns:
        DataFrame of fixation events with columns:
        fixation_id, onset_ms, offset_ms, duration_ms, x_mean, y_mean.
    """
    df = df.copy().reset_index(drop=True)
    x = df[x_col].values.astype(float)
    y = df[y_col].values.astype(float)
    t = df[time_col].values.astype(float)

    # Compute sample-to-sample velocity
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dt = np.diff(t, prepend=t[0] - 1.0)
    dt[dt == 0] = 1.0  # avoid division by zero

    velocity = np.sqrt(dx**2 + dy**2) / dt  # px/ms

    # Classify: fixation if velocity < threshold
    is_fixation = velocity < velocity_threshold

    # Merge into fixation events
    fixations = []
    labeled, n_events = ndimage.label(is_fixation)

    for evt_id in range(1, n_events + 1):
        indices = np.where(labeled == evt_id)[0]
        duration = t[indices[-1]] - t[indices[0]]
        if duration >= min_fixation_ms:
            fixations.append({
                "fixation_id": len(fixations) + 1,
                "onset_ms": t[indices[0]],
                "offset_ms": t[indices[-1]],
                "duration_ms": round(duration, 2),
                "x_mean": round(np.nanmean(x[indices]), 2),
                "y_mean": round(np.nanmean(y[indices]), 2),
                "n_samples": len(indices),
            })

    fix_df = pd.DataFrame(fixations)
    print(
        f"IVT fixation detection: {len(fix_df)} fixations detected\n"
        f"  Mean duration: {fix_df['duration_ms'].mean():.0f} ms\n"
        f"  Median duration: {fix_df['duration_ms'].median():.0f} ms"
    )
    return fix_df
```

### Step 3 — AOI Analysis

```python
def define_aoi_rectangles(
    aoi_specs: List[Dict],
) -> List[Dict]:
    """
    Define rectangular areas of interest.

    Args:
        aoi_specs: List of dicts with keys:
            name (str), x_min, x_max, y_min, y_max (floats in pixels).

    Returns:
        List of AOI dicts with added 'type' = 'rectangle'.

    Example:
        aoi_specs = [
            {"name": "face", "x_min": 400, "x_max": 800, "y_min": 100, "y_max": 600},
            {"name": "text", "x_min": 100, "x_max": 400, "y_min": 200, "y_max": 500},
        ]
    """
    for aoi in aoi_specs:
        aoi["type"] = "rectangle"
    return aoi_specs


def assign_fixation_to_aoi(
    fixation: Dict,
    aois: List[Dict],
) -> str:
    """
    Assign a fixation to an AOI by point-in-rectangle test.

    Args:
        fixation: Fixation dict with 'x_mean', 'y_mean'.
        aois:     List of AOI dicts.

    Returns:
        AOI name or 'outside' if not in any AOI.
    """
    fx, fy = fixation["x_mean"], fixation["y_mean"]
    for aoi in aois:
        if (aoi["x_min"] <= fx <= aoi["x_max"] and
                aoi["y_min"] <= fy <= aoi["y_max"]):
            return aoi["name"]
    return "outside"


def compute_aoi_metrics(
    fix_df: pd.DataFrame,
    aois: List[Dict],
    total_duration_ms: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute per-AOI dwell time, fixation count, and first fixation latency.

    Args:
        fix_df:           Fixation events DataFrame from detect_fixations_ivt().
        aois:             List of AOI dicts (from define_aoi_rectangles()).
        total_duration_ms: Total trial/stimulus duration for proportion dwell.

    Returns:
        DataFrame with one row per AOI and metrics columns.
    """
    fix_df = fix_df.copy()
    fix_df["aoi"] = fix_df.apply(
        lambda row: assign_fixation_to_aoi(row.to_dict(), aois), axis=1
    )

    if total_duration_ms is None:
        total_duration_ms = fix_df["duration_ms"].sum()

    rows = []
    for aoi in aois:
        aoi_name = aoi["name"]
        aoi_fixes = fix_df[fix_df["aoi"] == aoi_name]

        n_fix = len(aoi_fixes)
        dwell_ms = aoi_fixes["duration_ms"].sum()
        prop_dwell = dwell_ms / total_duration_ms if total_duration_ms > 0 else 0
        first_fix_latency = aoi_fixes["onset_ms"].min() if n_fix > 0 else np.nan
        mean_fix_dur = aoi_fixes["duration_ms"].mean() if n_fix > 0 else np.nan

        rows.append({
            "aoi": aoi_name,
            "n_fixations": n_fix,
            "total_dwell_ms": round(dwell_ms, 2),
            "proportion_dwell": round(prop_dwell, 4),
            "first_fixation_latency_ms": round(float(first_fix_latency), 2) if not np.isnan(first_fix_latency) else np.nan,
            "mean_fixation_duration_ms": round(float(mean_fix_dur), 2) if not np.isnan(mean_fix_dur) else np.nan,
        })

    result_df = pd.DataFrame(rows)
    print("\nAOI Metrics:")
    print(result_df.to_string(index=False))
    return result_df


def scanpath_similarity(
    seq_a: List[str],
    seq_b: List[str],
    normalize: bool = True,
) -> float:
    """
    Compute Levenshtein (edit) distance between two AOI scanpath sequences.

    Args:
        seq_a, seq_b: Lists of AOI labels (strings) in temporal order.
        normalize:    Divide by max(len(seq_a), len(seq_b)) for [0, 1] range.

    Returns:
        Edit distance (lower = more similar). Normalized: 0 = identical, 1 = maximally different.
    """
    m, n = len(seq_a), len(seq_b)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    dist = int(dp[m][n])
    if normalize and max(m, n) > 0:
        return round(dist / max(m, n), 4)
    return float(dist)
```

---

## Advanced Usage

### Heatmap Generation and Pupillometry

```python
def generate_fixation_heatmap(
    fix_df: pd.DataFrame,
    screen_width: int = 1920,
    screen_height: int = 1080,
    bandwidth: float = 50.0,
    weight_by_duration: bool = True,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a fixation density heatmap using Gaussian KDE.

    Args:
        fix_df:              Fixation events DataFrame.
        screen_width:        Screen width in pixels.
        screen_height:       Screen height in pixels.
        bandwidth:           Gaussian kernel bandwidth (pixels).
        weight_by_duration:  Weight fixation points by their duration.
        output_path:         Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    from scipy.stats import gaussian_kde

    x = fix_df["x_mean"].values
    y = fix_df["y_mean"].values
    weights = fix_df["duration_ms"].values if weight_by_duration else None

    # Fit KDE
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy, bw_method=bandwidth / np.std(xy))

    # Evaluate on grid
    grid_x = np.linspace(0, screen_width, 200)
    grid_y = np.linspace(0, screen_height, 150)
    XX, YY = np.meshgrid(grid_x, grid_y)
    grid_pts = np.vstack([XX.ravel(), YY.ravel()])
    Z = kde(grid_pts).reshape(XX.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        Z, origin="upper", extent=[0, screen_width, screen_height, 0],
        cmap="hot", alpha=0.85, aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="Fixation density")
    ax.scatter(x, y, s=5, c="cyan", alpha=0.3)
    ax.set_xlim(0, screen_width)
    ax.set_ylim(screen_height, 0)
    ax.set_xlabel("Screen X (px)")
    ax.set_ylabel("Screen Y (px)")
    ax.set_title("Fixation Heatmap")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    plt.show()
    return fig


def pupillometry_baseline_correct(
    df: pd.DataFrame,
    pupil_col: str = "pupil_clean",
    time_col: str = "timestamp_ms",
    event_onset_ms: float = 0.0,
    baseline_window: Tuple[float, float] = (-200.0, 0.0),
    analysis_window: Tuple[float, float] = (-200.0, 2000.0),
    zscore: bool = True,
) -> pd.DataFrame:
    """
    Event-locked pupillometry: baseline subtraction and optional z-scoring.

    Args:
        df:               Gaze DataFrame with cleaned pupil data.
        pupil_col:        Cleaned pupil diameter column.
        time_col:         Timestamp column (ms, absolute).
        event_onset_ms:   Absolute timestamp of the event of interest.
        baseline_window:  (start, end) relative to event onset for baseline (ms).
        analysis_window:  (start, end) relative to event onset to return (ms).
        zscore:           Whether to z-score the corrected signal.

    Returns:
        DataFrame with 'time_relative_ms', 'pupil_corrected', 'pupil_zscore' columns.
    """
    df = df.copy()
    df["time_relative_ms"] = df[time_col] - event_onset_ms

    # Baseline window mask
    baseline_mask = (
        (df["time_relative_ms"] >= baseline_window[0]) &
        (df["time_relative_ms"] <= baseline_window[1])
    )
    baseline_mean = df.loc[baseline_mask, pupil_col].mean()

    # Baseline subtraction
    df["pupil_corrected"] = df[pupil_col] - baseline_mean

    # Z-score
    if zscore:
        mu = df["pupil_corrected"].mean()
        sigma = df["pupil_corrected"].std()
        df["pupil_zscore"] = (df["pupil_corrected"] - mu) / sigma if sigma > 0 else 0

    # Trim to analysis window
    analysis_mask = (
        (df["time_relative_ms"] >= analysis_window[0]) &
        (df["time_relative_ms"] <= analysis_window[1])
    )
    result = df[analysis_mask][
        ["time_relative_ms", pupil_col, "pupil_corrected"] +
        (["pupil_zscore"] if zscore else [])
    ].copy()

    print(
        f"Pupillometry baseline correction:\n"
        f"  Baseline mean: {baseline_mean:.3f}\n"
        f"  Analysis window: {analysis_window} ms relative to event\n"
        f"  Samples in analysis window: {len(result)}"
    )
    return result
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| All samples classified as saccades | Velocity threshold too low | Increase `velocity_threshold` to 50–100 px/ms |
| Very few fixations detected | `min_fixation_ms` too large | Reduce to 80 ms; check sampling rate |
| IVT merges multiple fixations | Short saccades between fixations | Apply fixation merging if gap < 75 ms |
| Heatmap appears blank | Fixations outside screen bounds | Check coordinate origin (top-left vs center) |
| Pupil baseline window all NaN | Event onset timestamp mismatch | Verify `event_onset_ms` unit (ms vs seconds) |
| Levenshtein distance is 0 for different scanpaths | Same AOI sequence | Check AOI assignment; some fixations may be "outside" |
| KDE import error | scipy version | Use `from scipy.stats import gaussian_kde` |

---

## External Resources

- Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades.
  *ETRA 2000*, 71–78.
- Holmqvist, K., et al. (2011). *Eye Tracking: A Comprehensive Guide.*
- Mathôt, S., et al. (2018). Methods in cognitive pupillometry. *Behavior Research Methods.*
- PyGaze (open-source eye-tracking framework): <https://www.pygaze.org/>
- Tobii Pro SDK: <https://developer.tobiipro.com/>
- SR Research EyeLink: <https://www.sr-research.com/>

---

## Examples

### Example 1 — IVT Fixation Detection from Gaze Stream

```python
import numpy as np
import pandas as pd

# Simulate raw gaze data
rng = np.random.default_rng(10)
n_samples = 5000
time_ms = np.arange(0, n_samples)  # 1 kHz sampling

# Simulate fixations (clusters) and saccades (fast movements)
gaze_x = np.zeros(n_samples)
gaze_y = np.zeros(n_samples)

# Three fixation periods
fixation_regions = [
    (0, 800, 500, 400),
    (900, 1800, 1200, 600),
    (2000, 3000, 800, 300),
    (3100, 4000, 1500, 700),
    (4100, 4999, 960, 540),
]
for start, end, fx, fy in fixation_regions:
    gaze_x[start:end] = rng.normal(fx, 5, end - start)
    gaze_y[start:end] = rng.normal(fy, 5, end - start)

# Fill remaining with saccades (linear interpolation between fixation centers)
for i in range(len(fixation_regions) - 1):
    sac_start = fixation_regions[i][1]
    sac_end = fixation_regions[i + 1][0]
    fx_start = fixation_regions[i][2]
    fx_end = fixation_regions[i + 1][2]
    fy_start = fixation_regions[i][3]
    fy_end = fixation_regions[i + 1][3]
    n = sac_end - sac_start
    if n > 0:
        gaze_x[sac_start:sac_end] = np.linspace(fx_start, fx_end, n)
        gaze_y[sac_start:sac_end] = np.linspace(fy_start, fy_end, n)

# Simulate pupil with blinks
pupil = rng.normal(4.5, 0.2, n_samples)
pupil[1500:1600] = 0  # blink at 1.5–1.6 s
pupil[3500:3580] = 0  # blink at 3.5–3.58 s

df_gaze = pd.DataFrame({
    "timestamp_ms": time_ms, "gaze_x": gaze_x,
    "gaze_y": gaze_y, "pupil_diam": pupil,
})

# Step 1: Blink removal
df_clean = detect_and_interpolate_blinks(df_gaze, max_blink_ms=150, margin_ms=50)

# Step 2: Fixation detection
fix_df = detect_fixations_ivt(df_clean, velocity_threshold=30, min_fixation_ms=80)
print(fix_df)

# Step 3: AOI analysis
aois = define_aoi_rectangles([
    {"name": "left_region", "x_min": 0, "x_max": 960, "y_min": 0, "y_max": 1080},
    {"name": "right_region", "x_min": 960, "x_max": 1920, "y_min": 0, "y_max": 1080},
])
aoi_metrics = compute_aoi_metrics(fix_df, aois, total_duration_ms=5000)

# Step 4: Heatmap
generate_fixation_heatmap(fix_df, screen_width=1920, screen_height=1080,
                          bandwidth=60, output_path="heatmap.png")
```

### Example 2 — AOI Dwell Time Table and Pupillometry Baseline Correction

```python
# Scanpath comparison
seq1 = ["left_region", "right_region", "left_region", "right_region", "right_region"]
seq2 = ["left_region", "left_region", "right_region", "left_region", "right_region"]
seq3 = ["right_region", "right_region", "right_region", "left_region", "right_region"]

dist_12 = scanpath_similarity(seq1, seq2, normalize=True)
dist_13 = scanpath_similarity(seq1, seq3, normalize=True)
print(f"Scanpath distance seq1-seq2: {dist_12:.3f}")
print(f"Scanpath distance seq1-seq3: {dist_13:.3f}")

# Pupillometry: event-locked to stimulus onset at t=1000 ms
pupil_result = pupillometry_baseline_correct(
    df_clean,
    pupil_col="pupil_clean",
    time_col="timestamp_ms",
    event_onset_ms=1000.0,
    baseline_window=(-200, 0),
    analysis_window=(-200, 2000),
    zscore=True,
)

# Plot pupillometry
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(pupil_result["time_relative_ms"], pupil_result["pupil_zscore"],
        color="steelblue", linewidth=1.5)
ax.axvline(0, color="crimson", linestyle="--", linewidth=1, label="Event onset")
ax.axvspan(-200, 0, alpha=0.1, color="gray", label="Baseline window")
ax.set_xlabel("Time relative to event onset (ms)")
ax.set_ylabel("Pupil diameter (z-score)")
ax.set_title("Event-Locked Pupillometry")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig("pupillometry.png", dpi=150)
plt.show()
print("Eye-tracking analysis complete.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — blink interpolation, IVT fixation detection, AOI metrics, scanpath distance, heatmap, pupillometry |
