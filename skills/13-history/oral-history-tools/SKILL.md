---
name: oral-history-tools
description: >
  Use this Skill to process oral history recordings: Whisper transcription with timestamps,
  pyannote speaker diarization, OHMS metadata XML, and speaker anonymization.
tags:
  - history
  - oral-history
  - Whisper
  - speaker-diarization
  - transcription
  - digital-humanities
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
    - openai-whisper>=20231117
    - pyannote.audio>=3.1
    - pandas>=1.5
    - numpy>=1.23
    - torch>=2.0
last_updated: "2026-03-18"
status: stable
---

# Oral History Processing: Transcription, Diarization, and OHMS Export

> **TL;DR** — Transcribe oral history recordings with OpenAI Whisper (word-level timestamps),
> assign speakers via pyannote.audio 3.x diarization, export OHMS-compatible XML cuepoints,
> and anonymize PII before archival deposit.

---

## When to Use

Use this Skill when you need to:

- Transcribe long-form oral history interviews (30 min – 4+ hours) with accurate timestamps
- Identify who is speaking at each moment in multi-speaker recordings
- Generate OHMS (Oral History Metadata Synchronizer) XML for AV archives
- Anonymize interviewee names and contact details before sharing transcripts
- Export transcripts as Markdown, SRT subtitles, or WebVTT for online publication

Do **not** use this Skill for:

- Real-time live captioning (use specialized streaming ASR services)
- Music or non-speech audio classification (use librosa or Essentia)
- Large-scale production pipelines requiring GPU clusters (use WhisperX on SLURM)

---

## Background

OpenAI Whisper is a transformer-based ASR model available in five sizes (tiny → large-v3).
The `word_timestamps=True` option produces per-word start/end times via dynamic time
warping alignment against the hidden states.

pyannote.audio 3.x provides an end-to-end speaker diarization pipeline (segmentation +
embedding clustering). It requires a HuggingFace token and acceptance of the model's
terms of use.

Combining Whisper and pyannote requires **timestamp alignment**: for each Whisper word
(start, end), find which diarization segment (speaker, seg_start, seg_end) the word
falls inside, and assign that speaker label.

OHMS (Oral History Metadata Synchronizer) is an open-source tool used by oral history
archives. Its XML schema stores interview metadata, keyword index, and time-coded cuepoints
that synchronize a transcript with an AV file.

| Component | Purpose |
|---|---|
| Whisper `large-v3` | Best accuracy for accented speech and historical vocabulary |
| `word_timestamps=True` | Per-word start/end times (±0.1 s accuracy) |
| pyannote Pipeline | End-to-end diarization; clusters speaker embeddings |
| OHMS cuepoint | Time offset + transcript segment + subject tags |
| SRT format | Standard subtitle format: index, timecode, text |

---

## Environment Setup

```bash
# Create Python environment
conda create -n oral-history python=3.11 -y
conda activate oral-history

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Whisper and pyannote
pip install openai-whisper "pyannote.audio>=3.1" "pandas>=1.5" "numpy>=1.23"

# Accept pyannote model license on HuggingFace (required once):
#   1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
#   2. Accept terms of use
#   3. Create a HuggingFace access token at https://huggingface.co/settings/tokens

# Test Whisper
python -c "import whisper; print(whisper.__version__)"

# Set your HuggingFace token as an environment variable (never hardcode it)
export HF_TOKEN="your_huggingface_token_here"
```

On Windows:

```bash
conda create -n oral-history python=3.11 -y
conda activate oral-history
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper "pyannote.audio>=3.1" "pandas>=1.5" "numpy>=1.23"
```

---

## Core Workflow

### Step 1 — Whisper Transcription with Word Timestamps

```python
import whisper
import numpy as np
from pathlib import Path


def transcribe_with_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = None,
    initial_prompt: str = None,
) -> dict:
    """
    Transcribe an audio file with Whisper, returning word-level timestamps.

    Model sizes and VRAM requirements:
        tiny   (~39M params, ~1 GB VRAM, fastest, lowest accuracy)
        base   (~74M params, ~1 GB VRAM)
        small  (~244M params, ~2 GB VRAM)
        medium (~769M params, ~5 GB VRAM)
        large-v3 (~1.55B params, ~10 GB VRAM, best accuracy)

    Args:
        audio_path:     Absolute path to the audio file (MP3, WAV, M4A, FLAC).
        model_size:     Whisper model to use (see above).
        language:       ISO 639-1 language code, e.g. "en", "de".
                        None = auto-detect.
        initial_prompt: Optional context string to improve accuracy on domain vocabulary
                        (e.g. names, technical terms) — max ~224 tokens.

    Returns:
        Dict with keys: text (full transcript), segments (list of segment dicts),
        words (flat list of word dicts with start/end/probability),
        language (detected language code).
    """
    model = whisper.load_model(model_size)

    transcribe_kwargs = {
        "word_timestamps": True,
        "verbose": False,
    }
    if language:
        transcribe_kwargs["language"] = language
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    result = model.transcribe(audio_path, **transcribe_kwargs)

    # Flatten word-level data from segments
    words = []
    for seg in result["segments"]:
        for word_data in seg.get("words", []):
            words.append({
                "word": word_data["word"].strip(),
                "start": round(float(word_data["start"]), 3),
                "end": round(float(word_data["end"]), 3),
                "probability": round(float(word_data["probability"]), 4),
            })

    return {
        "text": result["text"],
        "segments": result["segments"],
        "words": words,
        "language": result.get("language", "unknown"),
        "audio_path": audio_path,
    }
```

### Step 2 — Speaker Diarization Merge (Whisper + pyannote)

```python
import os
import pandas as pd
from pyannote.audio import Pipeline


def diarize_audio(
    audio_path: str,
    hf_token: str = None,
    min_speakers: int = None,
    max_speakers: int = None,
) -> list[dict]:
    """
    Run pyannote speaker diarization and return labelled segments.

    Reads the HuggingFace token from the environment variable HF_TOKEN
    if hf_token is not provided.

    Args:
        audio_path:   Absolute path to the audio file.
        hf_token:     HuggingFace access token. Falls back to os.getenv("HF_TOKEN").
        min_speakers: Minimum expected speakers (optional hint).
        max_speakers: Maximum expected speakers (optional hint).

    Returns:
        List of dicts: speaker (str), start (float), end (float).
    """
    token = hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token required. Set HF_TOKEN environment variable or pass hf_token."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    diarize_kwargs = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    diarization = pipeline(audio_path, **diarize_kwargs)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
        })

    return segments


def merge_whisper_diarization(
    whisper_words: list[dict],
    diarization_segments: list[dict],
) -> pd.DataFrame:
    """
    Assign a speaker label to each Whisper word using diarization timestamps.

    For each word, the speaker is the diarization segment whose interval
    [seg_start, seg_end] contains the word's midpoint. If no segment matches,
    the label is "UNKNOWN".

    Args:
        whisper_words:        Flat list of word dicts from transcribe_with_whisper().
        diarization_segments: List of segment dicts from diarize_audio().

    Returns:
        DataFrame with columns: word, start, end, probability, speaker.
    """
    records = []
    for w in whisper_words:
        midpoint = (w["start"] + w["end"]) / 2.0
        assigned = "UNKNOWN"
        for seg in diarization_segments:
            if seg["start"] <= midpoint <= seg["end"]:
                assigned = seg["speaker"]
                break
        records.append({
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "probability": w["probability"],
            "speaker": assigned,
        })

    return pd.DataFrame(records)


def aggregate_to_utterances(
    word_df: pd.DataFrame,
    silence_gap: float = 1.5,
) -> pd.DataFrame:
    """
    Aggregate word-level transcript into speaker utterances.

    A new utterance begins when the speaker changes or when the silence
    between consecutive words exceeds silence_gap seconds.

    Args:
        word_df:     DataFrame from merge_whisper_diarization().
        silence_gap: Minimum silence (seconds) to split utterances.

    Returns:
        DataFrame with columns: speaker, start, end, text.
    """
    if word_df.empty:
        return pd.DataFrame(columns=["speaker", "start", "end", "text"])

    utterances = []
    current_speaker = word_df.iloc[0]["speaker"]
    current_start = word_df.iloc[0]["start"]
    current_words = [word_df.iloc[0]["word"]]
    prev_end = word_df.iloc[0]["end"]

    for _, row in word_df.iloc[1:].iterrows():
        gap = row["start"] - prev_end
        speaker_changed = row["speaker"] != current_speaker
        long_silence = gap > silence_gap

        if speaker_changed or long_silence:
            utterances.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": prev_end,
                "text": " ".join(current_words).strip(),
            })
            current_speaker = row["speaker"]
            current_start = row["start"]
            current_words = [row["word"]]
        else:
            current_words.append(row["word"])

        prev_end = row["end"]

    utterances.append({
        "speaker": current_speaker,
        "start": current_start,
        "end": prev_end,
        "text": " ".join(current_words).strip(),
    })

    return pd.DataFrame(utterances)
```

### Step 3 — OHMS-Compatible XML Generation

```python
from lxml import etree
import re


def generate_ohms_xml(
    utterances_df: pd.DataFrame,
    interview_metadata: dict,
    output_path: str,
    cuepoint_interval: float = 60.0,
) -> str:
    """
    Generate an OHMS-compatible XML file from utterance data.

    The OHMS schema includes: interview metadata, keyword index,
    and time-coded cuepoints linking transcript segments to AV positions.

    Args:
        utterances_df:     DataFrame from aggregate_to_utterances().
        interview_metadata: Dict with keys: title, interviewee, interviewer,
                            date, collection, duration_seconds, av_filename.
        output_path:       Absolute path to write the XML file.
        cuepoint_interval: Create a cuepoint every N seconds (default 60).

    Returns:
        XML string.
    """
    root = etree.Element("ROOT")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

    # Record metadata
    record = etree.SubElement(root, "record")
    etree.SubElement(record, "date").text = interview_metadata.get("date", "")
    etree.SubElement(record, "title").text = interview_metadata.get("title", "")
    etree.SubElement(record, "interviewee").text = interview_metadata.get("interviewee", "")
    etree.SubElement(record, "interviewer").text = interview_metadata.get("interviewer", "")
    etree.SubElement(record, "collection").text = interview_metadata.get("collection", "")
    etree.SubElement(record, "mediafile").text = interview_metadata.get("av_filename", "")

    # Build cuepoints at regular intervals
    duration = interview_metadata.get("duration_seconds", 0.0)
    cuepoints_el = etree.SubElement(record, "cuepoints")

    cue_times = [t for t in
                 [i * cuepoint_interval for i in range(int(duration / cuepoint_interval) + 1)]
                 if t <= duration]

    for cue_time in cue_times:
        # Collect utterances in the cuepoint window
        window_start = cue_time
        window_end = cue_time + cuepoint_interval
        window_df = utterances_df[
            (utterances_df["start"] >= window_start) &
            (utterances_df["start"] < window_end)
        ]

        cuepoint = etree.SubElement(cuepoints_el, "cuepoint")
        etree.SubElement(cuepoint, "time").text = str(int(cue_time))

        # Format timestamp as HH:MM:SS
        hours = int(cue_time // 3600)
        mins = int((cue_time % 3600) // 60)
        secs = int(cue_time % 60)
        etree.SubElement(cuepoint, "timestamp").text = f"{hours:02d}:{mins:02d}:{secs:02d}"

        # Transcript snippet for this window
        snippet = " ".join(window_df["text"].tolist())
        etree.SubElement(cuepoint, "transcript").text = snippet[:500]  # OHMS limit
        etree.SubElement(cuepoint, "keywords").text = ""  # Manual entry in OHMS editor

    xml_str = etree.tostring(root, encoding="unicode", pretty_print=True,
                             xml_declaration=False)
    xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(xml_str)

    print(f"OHMS XML written to {output_path} ({len(cue_times)} cuepoints)")
    return xml_str


def anonymize_transcript(
    text: str,
    speaker_map: dict = None,
) -> str:
    """
    Anonymize a transcript by replacing real speaker names and PII.

    Removes phone numbers, email addresses, and optionally replaces
    named speakers with generic labels.

    Args:
        text:        Input transcript text.
        speaker_map: Optional dict mapping real names to anonymized labels,
                     e.g. {"John Smith": "[INTERVIEWEE_1]"}.

    Returns:
        Anonymized text string.
    """
    # Remove phone numbers (US and international formats)
    text = re.sub(
        r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "[PHONE_REDACTED]",
        text,
    )
    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[EMAIL_REDACTED]",
        text,
    )
    # Remove social security numbers (US format)
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", text)

    # Replace named speakers
    if speaker_map:
        for real_name, anon_label in speaker_map.items():
            text = text.replace(real_name, anon_label)

    return text
```

---

## Advanced Usage

### Export to SRT Subtitles

```python
def export_srt(
    utterances_df: pd.DataFrame,
    output_path: str,
    include_speaker: bool = True,
) -> None:
    """
    Export utterances to SRT subtitle format.

    Args:
        utterances_df: DataFrame from aggregate_to_utterances().
        output_path:   Absolute path to write the .srt file.
        include_speaker: Prefix each subtitle with the speaker label.
    """
    def seconds_to_srt(secs: float) -> str:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        ms = int((secs % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, row in utterances_df.iterrows():
        lines.append(str(i + 1))
        lines.append(f"{seconds_to_srt(row['start'])} --> {seconds_to_srt(row['end'])}")
        label = f"[{row['speaker']}] " if include_speaker else ""
        lines.append(f"{label}{row['text']}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"SRT file written to {output_path}")
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `RuntimeError: CUDA out of memory` | Model too large for GPU | Use `model_size="medium"` or add `--device cpu` |
| `pyannote.audio: 401 Unauthorized` | Invalid or missing HF token | Check `os.getenv("HF_TOKEN")`; accept model license on HuggingFace |
| Whisper hallucinates text on silence | Long silence in audio | Pre-process: trim silence with `ffmpeg -af silenceremove`  |
| Speaker diarization assigns all words to one speaker | Very poor microphone separation | Use Whisper-only mode; manually annotate speakers |
| `UNKNOWN` speaker for many words | Diarization gap between segments | Reduce `min_speakers`; check diarization `min_duration_on` param |
| OHMS XML not loading in OHMS editor | Encoding or schema mismatch | Validate XML; ensure UTF-8 BOM-free encoding |

---

## External Resources

- OpenAI Whisper: <https://github.com/openai/whisper>
- pyannote.audio 3.x: <https://github.com/pyannote/pyannote-audio>
- HuggingFace model cards: <https://huggingface.co/pyannote/speaker-diarization-3.1>
- OHMS (Oral History Metadata Synchronizer): <https://www.oralhistoryonline.org/>
- Oral History Association guidelines: <https://oralhistory.org/archives-principles-and-best-practices/>
- WhisperX (faster forced alignment): <https://github.com/m-bain/whisperX>

---

## Examples

### Example 1 — End-to-End Single Interview Pipeline

```python
import os

audio_file = "/data/interviews/interview_001.mp3"

# Step 1: Transcribe with Whisper
print("Transcribing with Whisper large-v3...")
whisper_result = transcribe_with_whisper(
    audio_path=audio_file,
    model_size="large-v3",
    language="en",
    initial_prompt="This is an oral history interview. Interviewee: Dr. Sarah Johnson.",
)
print(f"Detected language: {whisper_result['language']}")
print(f"Total words: {len(whisper_result['words'])}")

# Step 2: Diarize (requires HF_TOKEN env variable)
print("Running speaker diarization...")
diarization = diarize_audio(
    audio_path=audio_file,
    min_speakers=2,
    max_speakers=2,
)
print(f"Diarization segments: {len(diarization)}")

# Step 3: Merge and aggregate
word_df = merge_whisper_diarization(whisper_result["words"], diarization)
utterances = aggregate_to_utterances(word_df, silence_gap=1.5)
print(f"Total utterances: {len(utterances)}")

# Step 4: Anonymize
utterances["text"] = utterances["text"].apply(
    lambda t: anonymize_transcript(t, speaker_map={"Dr. Sarah Johnson": "[INTERVIEWEE_1]"})
)

# Step 5: Export
export_srt(utterances, "/data/output/interview_001.srt")
```

### Example 2 — OHMS XML Export for Archive Deposit

```python
# Generate OHMS XML for archival deposit
metadata = {
    "title": "Life History Interview with Former Textile Worker",
    "interviewee": "[INTERVIEWEE_1]",
    "interviewer": "Prof. A. Researcher",
    "date": "2025-06-15",
    "collection": "Northern England Labour History Project",
    "duration_seconds": 5400.0,
    "av_filename": "interview_001.mp3",
}

ohms_xml = generate_ohms_xml(
    utterances_df=utterances,
    interview_metadata=metadata,
    output_path="/data/output/interview_001_ohms.xml",
    cuepoint_interval=60.0,
)

print("OHMS XML preview (first 500 chars):")
print(ohms_xml[:500])
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Whisper word timestamps, pyannote diarization merge, OHMS XML, SRT export, PII anonymization |
