---
name: psychopy-experiment
description: >
  Use this Skill to design behavioral experiments with PsychoPy: stimulus
  presentation, response timing, BIDS events.tsv generation, and Pavlovia
  online deployment checklist.
tags:
  - psychology
  - PsychoPy
  - behavioral-experiment
  - stimulus-presentation
  - BIDS
  - Pavlovia
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
    - psychopy>=2023.1
    - pandas>=1.5
    - numpy>=1.23
last_updated: "2026-03-18"
status: stable
---

# PsychoPy Behavioral Experiment

> **TL;DR** — Build precise behavioral experiments with PsychoPy: window and
> monitor calibration, visual stimuli, keyboard response timing, jittered ISI,
> TTL triggers for EEG, trial-loop with TrialHandler, BIDS events.tsv output,
> and a Pavlovia compatibility checklist for online deployment.

---

## When to Use

Use this Skill when you need to:

- Implement a stimulus-response task (RT task, n-back, Stroop, flanker, etc.)
- Synchronize behavioral events with EEG, fMRI, or eye-tracker via TTL triggers
- Generate BIDS-compliant `events.tsv` files from experiment logs
- Deploy a PsychoPy task to Pavlovia for online data collection
- Randomize trial order and jitter ISI for neuroimaging compatibility
- Record precise keystroke timings with sub-millisecond accuracy

---

## Background

### Window and Monitor Setup

PsychoPy uses a `Monitor` object to map pixel distances to visual degrees,
ensuring stimuli are sized correctly across different screen setups. The
`visual.Window` class creates the display surface. Always specify:

- `units`: use `"deg"` for visual degrees or `"norm"` for normalized coords
- `colorSpace`: `"rgb"` (−1 to 1) or `"rgb255"` (0–255)
- `fullscr`: `True` for real experiments to minimize refresh timing jitter

### Timing Precision

PsychoPy achieves frame-accurate timing via `win.flip()`. Each call blocks
until the next screen refresh, so timing resolution equals one frame
(~16.67 ms at 60 Hz, ~8.33 ms at 120 Hz). For sub-frame precision, use
`core.Clock` objects, not Python's `time.time()`.

### BIDS Events Format

BIDS (Brain Imaging Data Structure) requires `events.tsv` with at minimum:

| onset | duration | trial_type |
|---|---|---|
| 0.0 | 0.5 | fixation |
| 0.5 | 1.0 | stimulus_congruent |

Onset and duration are in seconds relative to the start of the run.

---

## Environment Setup

```bash
# Install PsychoPy in a dedicated environment
conda create -n psychopy_env python=3.10 -y
conda activate psychopy_env

# Install PsychoPy (standalone installer preferred for lab PCs)
pip install psychopy>=2023.1

# Install additional dependencies
pip install pandas>=1.5 numpy>=1.23

# For EEG triggering via serial port
pip install pyserial

# Test installation
python -c "from psychopy import visual, core, event; print('PsychoPy OK')"

# For online deployment (Pavlovia via PsychoJS)
# Use PsychoPy Builder GUI → File → Export to HTML
# Then upload to Pavlovia via the built-in sync feature
```

---

## Core Workflow

### Step 1 — Window and Monitor Configuration

```python
from psychopy import visual, core, event, data, logging
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np
import os
from typing import Optional, Dict, List, Any, Tuple

# ── Monitor calibration ───────────────────────────────────────────────────────

def setup_monitor(
    monitor_name: str = "testMonitor",
    width_cm: float = 53.0,
    distance_cm: float = 57.0,
    resolution: Tuple[int, int] = (1920, 1080),
) -> "psychopy.monitors.Monitor":
    """
    Configure a PsychoPy Monitor object for visual degree calculations.

    Args:
        monitor_name:  Name saved in PsychoPy monitor center.
        width_cm:      Physical screen width in centimeters.
        distance_cm:   Viewing distance from eye to screen in centimeters.
        resolution:    Screen resolution (width, height) in pixels.

    Returns:
        Configured Monitor object.
    """
    from psychopy import monitors
    mon = monitors.Monitor(monitor_name)
    mon.setWidth(width_cm)
    mon.setDistance(distance_cm)
    mon.setSizePix(list(resolution))
    return mon


def create_window(
    monitor,
    fullscr: bool = False,
    size: Tuple[int, int] = (1024, 768),
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    units: str = "deg",
) -> visual.Window:
    """
    Create a PsychoPy Window.

    Args:
        monitor:  Monitor object from setup_monitor().
        fullscr:  True for full-screen (recommended for real experiments).
        size:     Window size in pixels (ignored if fullscr=True).
        color:    Background color in RGB (−1 to 1 range).
        units:    Default stimulus units ('deg', 'norm', 'pix', 'cm').

    Returns:
        PsychoPy Window object.

    Notes:
        Set fullscr=True in real experiments to reduce timing jitter.
        Logging is set to WARNING to suppress console noise.
    """
    logging.console.setLevel(logging.WARNING)
    win = visual.Window(
        size=list(size),
        monitor=monitor,
        fullscr=fullscr,
        color=list(color),
        colorSpace="rgb",
        units=units,
        allowGUI=False,
        waitBlanking=True,
    )
    actual_fps = win.getActualFrameRate(nIdentical=20, nMaxFrames=100)
    if actual_fps:
        print(f"Monitor refresh rate: {actual_fps:.2f} Hz")
    return win
```

### Step 2 — Stimuli and Fixation Cross

```python
def create_fixation(win: visual.Window, size: float = 0.5) -> visual.ShapeStim:
    """
    Create a fixation cross as two overlapping rectangles.

    Args:
        win:  PsychoPy Window.
        size: Arm length in window units (degrees by default).

    Returns:
        ShapeStim fixation cross.
    """
    fixation = visual.ShapeStim(
        win=win,
        vertices="cross",
        size=size,
        fillColor="white",
        lineColor="white",
        units=win.units,
    )
    return fixation


def create_text_stim(
    win: visual.Window,
    text: str = "",
    pos: Tuple[float, float] = (0, 0),
    height: float = 1.0,
    color: str = "white",
) -> visual.TextStim:
    """Create a TextStim for instructions or stimulus words."""
    return visual.TextStim(
        win=win, text=text, pos=list(pos),
        height=height, color=color, wrapWidth=20,
    )


def create_image_stim(
    win: visual.Window,
    image_path: str,
    pos: Tuple[float, float] = (0, 0),
    size: Tuple[float, float] = (5.0, 5.0),
) -> visual.ImageStim:
    """
    Create an ImageStim for image-based experiments.

    Args:
        win:        PsychoPy Window.
        image_path: Absolute path to image file (PNG, JPG, BMP).
        pos:        Position in window units.
        size:       (width, height) in window units.

    Returns:
        ImageStim object. Call .draw() then win.flip() to display.
    """
    return visual.ImageStim(win=win, image=image_path, pos=list(pos), size=list(size))
```

### Step 3 — Trial Loop with TrialHandler

```python
def run_rt_task(
    win: visual.Window,
    conditions_file: str,
    output_dir: str,
    participant_id: str,
    session: int = 1,
    isi_range: Tuple[float, float] = (0.8, 1.2),
    stimulus_duration: float = 1.0,
    serial_port: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a simple reaction-time task using a CSV conditions file.

    The conditions CSV must have at minimum:
        - stimulus: text or path to image
        - trial_type: label for BIDS events.tsv
        - correct_key: expected response key ('left', 'right', etc.)

    Args:
        win:              PsychoPy Window.
        conditions_file:  Path to CSV with trial conditions.
        output_dir:       Directory for output data files.
        participant_id:   Participant identifier string.
        session:          Session number.
        isi_range:        (min, max) inter-stimulus interval in seconds.
        stimulus_duration: Maximum stimulus display time (seconds).
        serial_port:      Serial port string for TTL triggers (e.g. 'COM3').
                          Leave None if no EEG trigger required.

    Returns:
        DataFrame with trial data including onsets, RTs, accuracy.
    """
    import serial
    import time

    # Setup serial port for EEG triggers
    ser = None
    if serial_port:
        try:
            ser = serial.Serial(serial_port, baudrate=9600, timeout=0.001)
            print(f"Serial port {serial_port} opened for TTL triggers.")
        except Exception as e:
            print(f"Warning: Could not open serial port: {e}")

    def send_trigger(code: int) -> None:
        """Send a 1-byte TTL trigger code via serial port."""
        if ser and ser.isOpen():
            ser.write(bytes([code]))
            core.wait(0.005)  # 5 ms pulse
            ser.write(bytes([0]))  # reset

    # Load conditions
    trials = data.TrialHandler(
        trialList=data.importConditions(conditions_file),
        nReps=1,
        method="random",
        originPath=conditions_file,
    )

    # Stimuli
    fixation = create_fixation(win)
    stim_text = create_text_stim(win, text="", height=1.5)
    kb = keyboard.Keyboard()

    # Data collection
    trial_data = []
    global_clock = core.Clock()
    global_clock.reset()

    # Instructions
    instr = create_text_stim(win, text="Press LEFT or RIGHT arrow key.\n\nPress SPACE to begin.")
    instr.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    # Trial loop
    rng = np.random.default_rng()
    for trial in trials:
        # Jittered ISI (uniform distribution)
        isi_duration = rng.uniform(isi_range[0], isi_range[1])
        fixation.draw()
        win.flip()
        core.wait(isi_duration)

        # Stimulus onset
        stim_text.setText(trial.get("stimulus", "???"))
        stim_text.draw()
        kb.clock.reset()
        kb.clearEvents()
        onset_time = global_clock.getTime()
        win.flip()
        send_trigger(trial.get("trigger_code", 1))

        # Response collection
        resp = None
        rt = None
        keys = kb.waitKeys(
            maxWait=stimulus_duration,
            keyList=["left", "right", "escape"],
            waitRelease=False,
        )

        if keys:
            key = keys[0]
            if key.name == "escape":
                print("Escape pressed — aborting task.")
                break
            resp = key.name
            rt = key.rt * 1000  # convert to ms

        # Accuracy
        correct_key = trial.get("correct_key", None)
        accuracy = int(resp == correct_key) if resp and correct_key else None

        trial_data.append({
            "participant_id": participant_id,
            "session": session,
            "trial_n": trials.thisN + 1,
            "trial_type": trial.get("trial_type", "unknown"),
            "stimulus": trial.get("stimulus", ""),
            "onset_s": round(onset_time, 4),
            "duration_s": round(stimulus_duration, 4),
            "rt_ms": round(rt, 2) if rt else None,
            "response": resp,
            "correct_key": correct_key,
            "accuracy": accuracy,
            "isi_s": round(isi_duration, 3),
        })

        # Blank screen between trials
        win.flip()

    if ser:
        ser.close()

    # Save raw data
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(trial_data)
    out_path = os.path.join(
        output_dir, f"sub-{participant_id}_ses-{session:02d}_task-rt_raw.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Data saved: {out_path}")
    return df
```

---

## Advanced Usage

### BIDS Events.tsv Generation

```python
def generate_bids_events(
    df: pd.DataFrame,
    output_path: str,
    run_start_time: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a BIDS-compliant events.tsv from PsychoPy trial data.

    BIDS events.tsv required columns:
        onset:      Event start time relative to run start (seconds).
        duration:   Event duration in seconds.
        trial_type: String label for the event type.

    Recommended optional columns:
        response_time, stim_file, HED tags, response, accuracy.

    Args:
        df:              DataFrame from run_rt_task().
        output_path:     Absolute path for the output .tsv file.
        run_start_time:  Time of first trigger / run start in global clock
                         seconds (subtract from all onsets).

    Returns:
        BIDS events DataFrame.
    """
    events = pd.DataFrame()
    events["onset"] = (df["onset_s"] - run_start_time).round(4)
    events["duration"] = df["duration_s"].round(4)
    events["trial_type"] = df["trial_type"]
    events["response_time"] = (df["rt_ms"] / 1000).round(4)
    events["response"] = df["response"]
    events["accuracy"] = df["accuracy"]
    events["stim_file"] = df.get("stimulus", "n/a")

    # Replace NaN with BIDS 'n/a'
    events = events.fillna("n/a")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    events.to_csv(output_path, sep="\t", index=False)
    print(f"BIDS events.tsv saved: {output_path}")
    print(events.head())
    return events
```

### Pavlovia Compatibility Checklist

```python
PAVLOVIA_CHECKLIST = """
Pavlovia / PsychoJS Online Deployment Checklist
================================================

PRE-EXPORT
----------
[ ] Experiment built in PsychoPy Builder (not pure Coder)
[ ] All stimuli use relative paths (e.g., 'stimuli/face01.png')
[ ] No absolute file paths or OS-specific separators
[ ] Conditions CSV is in the experiment root directory
[ ] No Python-only libraries (e.g., serial, pyserial, os.system)
[ ] Keyboard responses use keyList=['left','right','space']
[ ] All clocks reset explicitly before use

EXPORT
------
[ ] File → Export to HTML (creates 'html/' folder)
[ ] Check generated JS for syntax errors in browser console
[ ] Test locally by opening index.html in Chrome

PAVLOVIA
--------
[ ] Synchronize via PsychoPy Builder Pavlovia menu
[ ] Set experiment to 'PILOTING' mode for testing
[ ] Test on target device (desktop Chrome recommended)
[ ] Set to 'RUNNING' for live data collection
[ ] Download data as CSV from Pavlovia dashboard

KNOWN LIMITATIONS
-----------------
- No serial port triggers online (use jsPsych + Lab.js for EEG online)
- Image stimuli > 1 MB may cause loading lag on slow connections
- Timing precision is frame-rate dependent (use Chrome for best results)
- Math functions: use Math.random() not np.random in JS-compatible code
"""


def print_pavlovia_checklist():
    """Print the Pavlovia deployment checklist."""
    print(PAVLOVIA_CHECKLIST)
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `win.flip()` timing jitter | Background processes | Close all apps; use fullscr=True |
| `getActualFrameRate()` returns None | OpenGL initialization issue | Update graphics drivers |
| `TrialHandler` skips trials | Wrong column names in CSV | Match CSV headers exactly |
| Escape key not quitting | `event.clearEvents()` before check | Use `kb.waitKeys` with `'escape'` in keyList |
| Serial port not found | Wrong port name or driver | Check Device Manager (Windows) or `/dev/ttyUSB*` (Linux) |
| Pavlovia export fails | Python-only code in components | Move to Begin Experiment tab as JS-compatible |
| BIDS onset negative | Run start time wrong | Pass correct `run_start_time` from scanner trigger |
| Image not loading on Pavlovia | Absolute path used | Use relative path: `'stimuli/img.png'` |

---

## External Resources

- PsychoPy documentation: <https://www.psychopy.org/documentation.html>
- BIDS specification: <https://bids-specification.readthedocs.io>
- Pavlovia: <https://pavlovia.org>
- PsychoJS (PsychoPy JavaScript): <https://github.com/psychopy/psychojs>
- Peirce et al. (2019). PsychoPy2. *Behavior Research Methods*, 51, 195–203.
  <https://doi.org/10.3758/s13428-018-01193-y>
- BIDS events.tsv specification: <https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html>

---

## Examples

### Example 1 — Simple RT Task with Fixation, Stimulus, and Response Timing

```python
from psychopy import visual, core, event
from psychopy.hardware import keyboard
import pandas as pd
import numpy as np

def run_simple_rt_demo():
    """
    Minimal RT task: fixation → stimulus → response.
    Demonstrates core timing, response collection, and data saving.
    """
    # Setup
    mon = setup_monitor("testMonitor", width_cm=53, distance_cm=57)
    win = create_window(mon, fullscr=False, size=(1024, 768))
    kb = keyboard.Keyboard()
    fixation = create_fixation(win, size=0.4)
    stim = create_text_stim(win, text="", height=2.0)

    stimuli = ["LEFT", "RIGHT", "LEFT", "RIGHT", "RIGHT",
               "LEFT", "RIGHT", "LEFT", "RIGHT", "LEFT"]
    correct_keys = ["left", "right", "left", "right", "right",
                    "left", "right", "left", "right", "left"]

    results = []
    global_clock = core.Clock()

    # Instructions
    instr_stim = create_text_stim(win, text="Press LEFT or RIGHT.\nSPACE to start.")
    instr_stim.draw()
    win.flip()
    event.waitKeys(keyList=["space"])
    global_clock.reset()

    rng = np.random.default_rng(99)

    for i, (word, correct) in enumerate(zip(stimuli, correct_keys)):
        # Fixation (jittered 800–1200 ms)
        isi = rng.uniform(0.8, 1.2)
        fixation.draw()
        win.flip()
        core.wait(isi)

        # Stimulus
        stim.setText(word)
        stim.draw()
        kb.clearEvents()
        kb.clock.reset()
        onset = global_clock.getTime()
        win.flip()

        # Response window
        keys = kb.waitKeys(maxWait=1.5, keyList=["left", "right", "escape"])
        win.flip()  # blank

        if keys and keys[0].name == "escape":
            break

        rt = keys[0].rt * 1000 if keys else None
        resp = keys[0].name if keys else None
        acc = int(resp == correct) if resp else 0

        results.append({
            "trial": i + 1, "stimulus": word, "onset_s": round(onset, 4),
            "rt_ms": round(rt, 2) if rt else None,
            "response": resp, "correct_key": correct, "accuracy": acc,
        })
        print(f"Trial {i+1}: {word} → {resp} ({rt:.0f} ms) {'✓' if acc else '✗'}"
              if rt else f"Trial {i+1}: no response")

    win.close()
    core.quit()
    df = pd.DataFrame(results)
    print(f"\nMean RT: {df['rt_ms'].mean():.0f} ms, Accuracy: {df['accuracy'].mean():.0%}")
    return df

# Run demo (comment out in non-interactive environments)
# df_results = run_simple_rt_demo()
```

### Example 2 — BIDS Events.tsv Generation from PsychoPy Log

```python
import pandas as pd
import os

def psychopy_log_to_bids(
    raw_csv_path: str,
    output_bids_dir: str,
    participant_id: str,
    session: int,
    task_name: str,
    run: int = 1,
    tr_onset_col: str = "onset_s",
    scanner_trigger_time: float = 0.0,
) -> str:
    """
    Convert PsychoPy output CSV to BIDS events.tsv.

    Args:
        raw_csv_path:         Path to PsychoPy CSV output.
        output_bids_dir:      Root BIDS directory.
        participant_id:       Participant label (e.g., '01').
        session:              Session number.
        task_name:            BIDS task name (e.g., 'stroop').
        run:                  Run number.
        tr_onset_col:         Column in CSV with trial onset times.
        scanner_trigger_time: Time of first scanner TR in global clock (s).

    Returns:
        Path to generated events.tsv file.
    """
    df = pd.read_csv(raw_csv_path)

    # Build BIDS path
    bids_path = os.path.join(
        output_bids_dir,
        f"sub-{participant_id}",
        f"ses-{session:02d}",
        "func",
    )
    os.makedirs(bids_path, exist_ok=True)
    filename = (
        f"sub-{participant_id}_ses-{session:02d}_"
        f"task-{task_name}_run-{run:02d}_events.tsv"
    )
    output_path = os.path.join(bids_path, filename)

    events = pd.DataFrame({
        "onset":         (df[tr_onset_col] - scanner_trigger_time).round(4),
        "duration":      df.get("duration_s", pd.Series([1.0] * len(df))).round(4),
        "trial_type":    df.get("trial_type", "stimulus"),
        "response_time": (df.get("rt_ms", pd.Series([float("nan")] * len(df))) / 1000).round(4),
        "response":      df.get("response", "n/a"),
        "accuracy":      df.get("accuracy", "n/a"),
    })
    events = events.fillna("n/a")
    events.to_csv(output_path, sep="\t", index=False)

    print(f"BIDS events.tsv created: {output_path}")
    print(f"  Trials: {len(events)}")
    print(f"  Unique trial types: {events['trial_type'].unique().tolist()}")
    print(f"  Run duration: {events['onset'].max():.1f} s")
    return output_path


# Demonstrate with synthetic data
synthetic_data = pd.DataFrame({
    "onset_s": [0.0, 2.5, 5.1, 7.8, 10.2],
    "duration_s": [1.0, 1.0, 1.0, 1.0, 1.0],
    "trial_type": ["congruent", "incongruent", "congruent", "incongruent", "congruent"],
    "rt_ms": [512.3, 678.1, 489.7, 701.2, 523.4],
    "response": ["left", "right", "left", "right", "left"],
    "accuracy": [1, 1, 1, 0, 1],
})
synthetic_data.to_csv("/tmp/psychopy_output.csv", index=False)

output_tsv = psychopy_log_to_bids(
    raw_csv_path="/tmp/psychopy_output.csv",
    output_bids_dir="/tmp/bids_study",
    participant_id="01",
    session=1,
    task_name="stroop",
    run=1,
)
print_pavlovia_checklist()
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — window setup, TrialHandler, TTL triggers, BIDS export, Pavlovia checklist |
