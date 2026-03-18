---
name: psychopy-neuro
description: >
  Use this Skill for PsychoPy experiment design: stimulus presentation, response
  collection, TTL synchronization, BIDS event files, and reaction time analysis.
tags:
  - neuroscience
  - psychopy
  - stimulus-presentation
  - bids
  - experimental-psychology
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
    - psychopy>=2023.2
    - pandas>=2.0
    - numpy>=1.24
    - matplotlib>=3.7
    - scipy>=1.11
last_updated: "2026-03-17"
status: "stable"
---

# PsychoPy Stimulus Presentation & Neuroimaging Synchronization

> **One-line summary**: Design and run neuroscience experiments with PsychoPy: visual/auditory stimuli, TTL triggers for EEG/fMRI, BIDS event file generation, and RT analysis.

---

## When to Use This Skill

- When designing visual or auditory cognitive paradigms (oddball, N-back, Stroop, etc.)
- When synchronizing stimulus delivery with EEG/fMRI acquisition via TTL pulses
- When generating BIDS-compatible events.tsv files for neuroimaging data
- When collecting keyboard/response-box reaction times with millisecond precision
- When deploying experiments online via Pavlovia
- When analyzing timing data from PsychoPy log files

**Trigger keywords**: PsychoPy, stimulus presentation, TTL trigger, EEG synchronization, BIDS events, reaction time, cognitive experiment, visual stimuli, auditory paradigm, Pavlovia

---

## Background & Key Concepts

### PsychoPy Architecture

PsychoPy offers two modes:
- **Builder**: GUI drag-and-drop experiment builder (generates Python code)
- **Coder**: Full Python scripting via `psychopy.visual`, `psychopy.core`, `psychopy.event`

For neuroscience experiments, **Coder** provides fine-grained timing control.

### Timing Precision

Stimulus timing in PsychoPy can be specified in:
- **Frames**: Most precise (e.g., 60 Hz monitor → 16.67 ms/frame)
- **Seconds**: Convenient but limited by monitor refresh

Critical timing: `win.flip()` synchronizes to vertical retrace (VBL).

### BIDS Events Format

BIDS (Brain Imaging Data Structure) `events.tsv` requires:

| Column | Description |
|:-------|:------------|
| `onset` | Stimulus onset time relative to first acquisition (s) |
| `duration` | Stimulus duration (s) |
| `trial_type` | Condition label |
| `response_time` | Reaction time (s), NaN if no response |
| `response` | Key pressed or NaN |

### TTL Synchronization

Parallel port pulse (8-bit trigger code) sent to EEG amplifier:
- Code 1-255 marks event type
- Duration typically 5-10 ms

---

## Environment Setup

### Install Dependencies

```bash
# PsychoPy is best installed as standalone or via pip in a fresh environment
pip install psychopy>=2023.2 pandas>=2.0 numpy>=1.24 matplotlib>=3.7 scipy>=1.11

# For parallel port triggers (EEG)
pip install pyparallel  # Linux
# pip install pywin32    # Windows (parallel port)
```

### Verify Installation

```python
import psychopy
from psychopy import core, visual, event
print(f"PsychoPy version: {psychopy.__version__}")
# Do NOT open a window in headless environments
```

---

## Core Workflow

### Step 1: Experiment Design and Trial List

```python
import numpy as np
import pandas as pd
from pathlib import Path

def create_oddball_trial_list(n_standards=120, n_oddballs=30, seed=42):
    """
    Create a randomized auditory oddball trial list.

    Standard tone: 1000 Hz (80% probability)
    Oddball tone: 2000 Hz (20% probability)

    Returns
    -------
    pd.DataFrame with columns: trial_num, trial_type, frequency_Hz, isi_s
    """
    rng = np.random.default_rng(seed)

    # Create trial types
    trials = (["standard"] * n_standards + ["oddball"] * n_oddballs)

    # Constraint: no more than 2 consecutive oddballs; at least 2 standards between oddballs
    while True:
        rng.shuffle(trials)
        # Check constraint: no two consecutive oddballs
        valid = True
        for i in range(len(trials) - 1):
            if trials[i] == "oddball" and trials[i+1] == "oddball":
                valid = False
                break
        if valid:
            break

    # Inter-stimulus intervals (jittered 0.8–1.2 s)
    isi_list = rng.uniform(0.8, 1.2, len(trials))

    trial_df = pd.DataFrame({
        "trial_num": range(1, len(trials) + 1),
        "trial_type": trials,
        "frequency_Hz": [1000 if t == "standard" else 2000 for t in trials],
        "isi_s": isi_list,
        "trigger_code": [10 if t == "standard" else 20 for t in trials],
    })
    trial_df["onset_s"] = trial_df["isi_s"].cumsum().shift(1).fillna(0)

    print(f"Trial list: {len(trial_df)} trials")
    print(f"  Standards: {(trial_df['trial_type']=='standard').sum()}")
    print(f"  Oddballs:  {(trial_df['trial_type']=='oddball').sum()}")
    print(f"  Total duration: {trial_df['onset_s'].max():.0f}s ({trial_df['onset_s'].max()/60:.1f} min)")

    # Save trial list
    trial_df.to_csv("oddball_trials.csv", index=False)
    return trial_df

trials = create_oddball_trial_list()
print("\nFirst 10 trials:")
print(trials.head(10).to_string(index=False))
```

### Step 2: PsychoPy Experiment Script

```python
"""
Auditory Oddball Experiment — PsychoPy Coder Mode
Run in a PsychoPy environment (not headless CI).
Replace 'SIMULATE = True' with 'SIMULATE = False' for real experiments.
"""
SIMULATE = True  # Set False for actual experiment

import numpy as np
import pandas as pd
from pathlib import Path

if not SIMULATE:
    from psychopy import core, visual, sound, event, prefs
    prefs.hardware["audioLib"] = ["ptb", "sounddevice"]

def run_oddball_experiment(trial_df, participant="P01", session=1, simulate=True):
    """
    Run auditory oddball experiment.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial list from create_oddball_trial_list()
    participant : str
    session : int
    simulate : bool
        If True, simulate timing without opening windows
    """
    if simulate:
        print("SIMULATION MODE: No window will open")
        results = []
        for _, trial in trial_df.iterrows():
            rt = np.random.exponential(0.35) + 0.15 if np.random.rand() < 0.90 else np.nan
            results.append({
                "trial_num": trial["trial_num"],
                "trial_type": trial["trial_type"],
                "frequency_Hz": trial["frequency_Hz"],
                "onset_s": trial["onset_s"],
                "response_time": rt,
                "response": "space" if not np.isnan(rt) else "none",
                "correct": not np.isnan(rt) if trial["trial_type"] == "oddball" else np.isnan(rt),
            })
        results_df = pd.DataFrame(results)
        save_results(results_df, participant, session)
        return results_df
    else:
        # Real experiment code
        from psychopy import core, visual, sound, event
        win = visual.Window([1280, 720], fullscr=True, units="norm")
        fixation = visual.TextStim(win, text="+", height=0.1)
        clock = core.Clock()

        results = []
        for _, trial in trial_df.iterrows():
            # Present fixation
            fixation.draw()
            win.flip()

            # Play tone (duration 100ms)
            tone = sound.Sound(trial["frequency_Hz"], secs=0.1, stereo=True)
            tone.play()
            onset_time = clock.getTime()

            # Send TTL trigger
            # parallel_port.setData(trial["trigger_code"])
            # core.wait(0.005)
            # parallel_port.setData(0)

            # Collect response
            event.clearEvents()
            response, rt = None, np.nan
            core.wait(trial["isi_s"] - 0.1)
            keys = event.getKeys(keyList=["space", "escape"], timeStamped=clock)
            if keys:
                key, key_time = keys[0]
                if key == "escape":
                    break
                response = key
                rt = key_time - onset_time

            results.append({
                "trial_num": trial["trial_num"],
                "trial_type": trial["trial_type"],
                "onset_s": onset_time,
                "response_time": rt,
                "response": response or "none",
            })

        win.close()
        results_df = pd.DataFrame(results)
        save_results(results_df, participant, session)
        return results_df

def save_results(results_df, participant, session):
    """Save results in BIDS-compatible events.tsv format."""
    output_dir = Path(f"sub-{participant}/ses-{session:02d}/beh")
    output_dir.mkdir(parents=True, exist_ok=True)

    # BIDS events file
    bids_events = results_df.rename(columns={
        "onset_s": "onset",
        "response_time": "response_time",
    })
    bids_events["duration"] = 0.1  # stimulus duration
    bids_events["trial_type"] = bids_events["trial_type"]
    bids_cols = ["onset", "duration", "trial_type", "response_time", "response"]
    bids_events[bids_cols].to_csv(
        output_dir / f"sub-{participant}_ses-{session:02d}_task-oddball_events.tsv",
        sep="\t", index=False, float_format="%.4f"
    )
    print(f"Saved BIDS events to {output_dir}")

results = run_oddball_experiment(trials, simulate=SIMULATE)
print(f"\nHit rate (oddball): {results[results['trial_type']=='oddball']['response_time'].notna().mean():.2%}")
print(f"FA rate (standard): {results[results['trial_type']=='standard']['response_time'].notna().mean():.2%}")
```

### Step 3: Reaction Time Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def analyze_reaction_times(results_df):
    """
    Comprehensive RT analysis for oddball paradigm.
    """
    # Filter valid responses (hits only)
    hits = results_df[
        (results_df["trial_type"] == "oddball") &
        (results_df["response_time"].notna()) &
        (results_df["response_time"] > 0.1) &
        (results_df["response_time"] < 1.0)
    ]["response_time"]

    print(f"Hit rate: {len(hits)} / {(results_df['trial_type']=='oddball').sum()}")
    print(f"\nRT descriptives (ms):")
    print(f"  Mean:   {hits.mean()*1000:.1f}")
    print(f"  Median: {hits.median()*1000:.1f}")
    print(f"  SD:     {hits.std()*1000:.1f}")
    print(f"  Skew:   {hits.skew():.3f}")

    # Outlier removal (cutoff method)
    rt_mean, rt_sd = hits.mean(), hits.std()
    hits_clean = hits[(hits > rt_mean - 2.5*rt_sd) & (hits < rt_mean + 2.5*rt_sd)]
    print(f"\nAfter ±2.5SD cutoff: {len(hits_clean)} trials retained")

    # Ex-Gaussian fit
    loc, scale, beta = stats.expon.fit(hits_clean)

    # Sequential RT effects (alertness)
    results_oddball = results_df[results_df["trial_type"] == "oddball"].copy()
    results_oddball["trial_position"] = np.arange(len(results_oddball))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # RT distribution
    axes[0].hist(hits_clean * 1000, bins=25, density=True,
                 color="steelblue", edgecolor="white", alpha=0.8)
    x = np.linspace(hits_clean.min(), hits_clean.max(), 100)
    axes[0].plot(x * 1000, stats.expon.pdf(x, loc, scale), 'r-', linewidth=2, label="Exp fit")
    axes[0].set_xlabel("RT (ms)"); axes[0].set_ylabel("Density")
    axes[0].set_title("Reaction Time Distribution"); axes[0].legend()

    # RT over trials
    valid_rts = results_oddball.dropna(subset=["response_time"])
    axes[1].scatter(valid_rts["trial_position"], valid_rts["response_time"] * 1000,
                    alpha=0.5, s=20)
    # Trend line
    if len(valid_rts) > 5:
        slope, intercept, *_ = stats.linregress(valid_rts["trial_position"],
                                                  valid_rts["response_time"] * 1000)
        axes[1].plot(valid_rts["trial_position"],
                     slope * valid_rts["trial_position"] + intercept, 'r-')
    axes[1].set_xlabel("Trial number"); axes[1].set_ylabel("RT (ms)")
    axes[1].set_title("RT over Experiment")

    # Cumulative accuracy
    results_oddball["hit"] = results_oddball["response_time"].notna()
    rolling_hr = results_oddball["hit"].rolling(10, min_periods=1).mean()
    axes[2].plot(rolling_hr.values, 'g-', linewidth=2)
    axes[2].set_xlabel("Trial number"); axes[2].set_ylabel("Hit rate")
    axes[2].set_title("Rolling Hit Rate (window=10)")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("rt_analysis.png", dpi=150)
    plt.show()

    return hits_clean

rt_clean = analyze_reaction_times(results)
```

---

## Advanced Usage

### Online Deployment Check

```python
# Pavlovia deployment checklist (not executable — reference guide)
checklist = {
    "1. Python → JavaScript":     "PsychoPy Builder auto-converts; avoid raw Python in Coder",
    "2. Stimuli files":            "Upload to Pavlovia GitLab repo; use relative paths",
    "3. Timing accuracy":          "Online timing ±10ms (vs. ±1ms local) — use frames not seconds",
    "4. Response keys":            "Browser keyboard events; test all target keys",
    "5. Data download":            "Pavlovia auto-saves CSV to project; download via GUI or API",
    "6. Ethical compliance":       "Add consent form as first component; store no PII",
}
for step, note in checklist.items():
    print(f"{step}: {note}")
```

---

## Troubleshooting

### Error: `psychopy.visual.Window fails to open` (in CI/headless)

**Cause**: No display available.

**Fix**:
```bash
# Linux: use Xvfb virtual display
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
python my_experiment.py
```

### Issue: Timing drift over long experiments

**Cause**: Audio/video scheduling accumulating delays.

**Fix**:
```python
from psychopy import core
# Use globalClock for all timing
global_clock = core.Clock()
# Reset at experiment start, use global_clock.getTime() for all onsets
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| psychopy | 2023.2, 2024.1  | Sound backend varies by OS; test ptb > sounddevice > pygame |

---

## External Resources

### Official Documentation

- [PsychoPy documentation](https://www.psychopy.org/api/index.html)
- [BIDS specification — events.tsv](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html)

### Key Papers

- Peirce, J.W. et al. (2019). *PsychoPy2: Experiments in behavior made easy*. Behavior Research Methods.

---

## Examples

### Example 1: Parse PsychoPy Log File to BIDS

```python
# =============================================
# Convert PsychoPy CSV output to BIDS events.tsv
# =============================================
import pandas as pd, numpy as np

# Simulate PsychoPy CSV output
psychopy_csv_data = {
    "trials.thisIndex": [0,1,2,3,4,5],
    "trial_type": ["standard","oddball","standard","standard","oddball","standard"],
    "stimulus_started": [0.0, 1.1, 2.2, 3.3, 4.4, 5.5],
    "key_resp.rt": [np.nan, 0.342, np.nan, np.nan, 0.418, np.nan],
    "key_resp.keys": [None, "space", None, None, "space", None],
}
df = pd.DataFrame(psychopy_csv_data)

# Convert to BIDS
bids = pd.DataFrame({
    "onset": df["stimulus_started"],
    "duration": 0.1,
    "trial_type": df["trial_type"],
    "response_time": df["key_resp.rt"].apply(lambda x: round(x, 4) if pd.notna(x) else "n/a"),
    "response": df["key_resp.keys"].apply(lambda x: x if x else "n/a"),
})

print(bids.to_string(index=False))
bids.to_csv("sub-P01_task-oddball_events.tsv", sep="\t", index=False)
print("\nSaved BIDS events.tsv")
```

**Interpreting these results**: The events.tsv file is directly compatible with MNE-Python, fMRIPrep, and other BIDS-aware analysis tools.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
