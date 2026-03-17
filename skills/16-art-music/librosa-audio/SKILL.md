---
name: librosa-audio
description: >
  Music information retrieval with librosa: tempo, chroma, MFCCs, spectral features,
  onset detection, harmonic-percussive separation, pitch, and k-NN similarity search.
tags:
  - audio
  - music
  - librosa
  - mfcc
  - signal-processing
  - similarity-search
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
  - librosa>=0.10.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
  - pandas>=2.0.0
  - soundfile>=0.12.0
  - pyarrow>=12.0.0
last_updated: "2026-03-17"
---

# librosa-audio: Music Information Retrieval

This skill covers end-to-end music information retrieval (MIR) using the `librosa`
library. You will learn how to load audio, extract a rich set of acoustic features,
detect musical events, and build a similarity-search index over a corpus of tracks.

## Installation

```bash
pip install librosa numpy scipy scikit-learn pandas soundfile pyarrow
# Optional: faster resampling backend
pip install soxr
# Optional: display waveforms / spectrograms in notebooks
pip install matplotlib
```

---

## 1. Loading Audio

`librosa.load` decodes any format supported by `soundfile` / `audioread` and resamples
to the target sample rate.

```python
import librosa
import numpy as np

def load_audio(path: str, sr: int = 22050, mono: bool = True):
    """
    Load an audio file and return the signal and sample rate.

    Parameters
    ----------
    path : str
        Path to the audio file (WAV, FLAC, MP3, OGG, …).
    sr : int
        Target sample rate in Hz.  Set to None to keep native rate.
    mono : bool
        Mix down to mono when True.

    Returns
    -------
    y : np.ndarray  shape (n_samples,)
    sr : int
    """
    y, sr_out = librosa.load(path, sr=sr, mono=mono)
    duration = len(y) / sr_out
    print(f"Loaded '{path}': {duration:.2f}s @ {sr_out} Hz")
    return y, sr_out
```

---

## 2. Core Feature Extraction

### 2.1 Tempo and Beat Tracking

```python
def extract_tempo_beats(y: np.ndarray, sr: int):
    """
    Estimate global tempo and beat frame positions.

    Returns
    -------
    tempo : float  — estimated BPM
    beat_times : np.ndarray  — beat positions in seconds
    """
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times
```

### 2.2 Chroma Features

Chroma (pitch-class profile) captures harmonic content independent of timbre.

```python
def extract_chroma(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Compute constant-Q chroma features.

    Returns
    -------
    chroma : np.ndarray  shape (12, T)
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    return chroma
```

### 2.3 MFCCs and Delta Coefficients

```python
def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    hop_length: int = 512,
) -> dict:
    """
    Compute MFCCs, delta, and delta-delta coefficients.

    Returns a dict with keys 'mfcc', 'delta', 'delta2', each shape (n_mfcc, T).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return {"mfcc": mfcc, "delta": delta, "delta2": delta2}
```

### 2.4 Spectral Features

```python
def extract_spectral_features(y: np.ndarray, sr: int, hop_length: int = 512) -> dict:
    """
    Compute spectral centroid, bandwidth, rolloff, and zero-crossing rate.

    All features returned as 1-D arrays (mean over time) plus raw frames.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

    return {
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "bandwidth_mean": float(np.mean(bandwidth)),
        "bandwidth_std": float(np.std(bandwidth)),
        "rolloff_mean": float(np.mean(rolloff)),
        "rolloff_std": float(np.std(rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        # raw frames kept for downstream use
        "_centroid": centroid,
        "_bandwidth": bandwidth,
        "_rolloff": rolloff,
        "_zcr": zcr,
    }
```

---

## 3. Advanced Features

### 3.1 Onset Detection

```python
def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect note onsets and return their times in seconds.
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times
```

### 3.2 Harmonic-Percussive Separation

```python
def separate_harmonic_percussive(y: np.ndarray):
    """
    Decompose a signal into harmonic and percussive components using median filtering.

    Returns
    -------
    y_harmonic : np.ndarray
    y_percussive : np.ndarray
    """
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_harmonic, y_percussive
```

### 3.3 Pitch Detection with YIN

```python
def estimate_pitch(y: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 2000.0):
    """
    Frame-wise fundamental frequency estimation using the YIN algorithm.

    Returns
    -------
    f0 : np.ndarray  — F0 in Hz per frame (NaN where unvoiced)
    """
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    return f0
```

### 3.4 Mel Spectrogram

```python
def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute a log-power mel spectrogram.

    Returns
    -------
    mel_db : np.ndarray  shape (n_mels, T)  — values in dB
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(S, ref=np.max)
    return mel_db
```

---

## 4. Compound Functions

### 4.1 Full Feature Extraction Pipeline

```python
import pandas as pd
import pathlib

def extract_audio_features(path: str, sr: int = 22050) -> dict:
    """
    Extract a flat feature vector from an audio file.

    Covers: tempo, chroma (mean per pitch class), MFCCs + deltas (mean + std),
    spectral centroid/bandwidth/rolloff/zcr, onset rate, and pitch statistics.

    Returns
    -------
    dict with scalar feature values and 'path' key.
    """
    y, sr = load_audio(path, sr=sr)

    features = {"path": path}

    # Tempo
    tempo, beat_times = extract_tempo_beats(y, sr)
    features["tempo_bpm"] = tempo
    features["n_beats"] = len(beat_times)

    # Chroma — 12 pitch classes
    chroma = extract_chroma(y, sr)
    for i, note in enumerate(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]):
        features[f"chroma_{note}"] = float(np.mean(chroma[i]))

    # MFCCs
    mfcc_dict = extract_mfcc(y, sr, n_mfcc=13)
    for k in ["mfcc", "delta", "delta2"]:
        arr = mfcc_dict[k]
        for i in range(arr.shape[0]):
            features[f"{k}_{i+1}_mean"] = float(np.mean(arr[i]))
            features[f"{k}_{i+1}_std"] = float(np.std(arr[i]))

    # Spectral
    spec = extract_spectral_features(y, sr)
    for k, v in spec.items():
        if not k.startswith("_"):
            features[k] = v

    # Onsets
    onsets = detect_onsets(y, sr)
    duration = len(y) / sr
    features["onset_rate_hz"] = len(onsets) / duration if duration > 0 else 0.0

    # Pitch (voiced frames only)
    f0 = estimate_pitch(y, sr)
    voiced = f0[~np.isnan(f0) & (f0 > 0)]
    features["pitch_mean_hz"] = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
    features["pitch_std_hz"] = float(np.std(voiced)) if len(voiced) > 0 else 0.0

    return features


def batch_extract_features(corpus_dir: str, sr: int = 22050, output_path: str = None) -> pd.DataFrame:
    """
    Extract features for every audio file in corpus_dir and return a DataFrame.

    Supports WAV, FLAC, MP3, OGG.  Saves to Parquet if output_path is given.
    """
    audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}
    paths = [
        p for p in pathlib.Path(corpus_dir).rglob("*")
        if p.suffix.lower() in audio_extensions
    ]
    print(f"Found {len(paths)} audio files in '{corpus_dir}'")

    records = []
    for p in paths:
        try:
            rec = extract_audio_features(str(p), sr=sr)
            records.append(rec)
        except Exception as exc:
            print(f"  WARN: skipping {p.name} — {exc}")

    df = pd.DataFrame(records)
    if output_path:
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"Saved features to '{output_path}'")
    return df
```

### 4.2 Similarity Matrix and k-NN Search

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

FEATURE_COLS_EXCLUDE = {"path", "tempo_bpm", "n_beats"}  # keep for reference

def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return numeric feature matrix with NaN filled to column mean."""
    numeric_cols = [c for c in df.select_dtypes("number").columns
                    if c not in FEATURE_COLS_EXCLUDE]
    X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
    return X


def compute_similarity_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a pairwise cosine similarity matrix over the audio feature space.

    Parameters
    ----------
    df : pd.DataFrame  — output of batch_extract_features

    Returns
    -------
    sim_matrix : np.ndarray  shape (n_tracks, n_tracks)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    X = _feature_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sim_matrix = cosine_similarity(X_scaled)
    return sim_matrix


def find_similar_tracks(
    query_path: str,
    corpus_dir: str,
    k: int = 5,
    sr: int = 22050,
) -> pd.DataFrame:
    """
    Find the k most similar tracks in corpus_dir to a query audio file.

    Steps
    -----
    1. Extract features for all tracks in corpus_dir.
    2. Extract features for the query file.
    3. Scale features and run k-NN search.

    Returns
    -------
    pd.DataFrame with columns ['path', 'distance', 'rank']
    """
    df_corpus = batch_extract_features(corpus_dir, sr=sr)
    query_feats = extract_audio_features(query_path, sr=sr)
    df_query = pd.DataFrame([query_feats])

    X_corpus = _feature_matrix(df_corpus)
    X_query = _feature_matrix(df_query)

    scaler = StandardScaler().fit(X_corpus)
    X_corpus_s = scaler.transform(X_corpus)
    X_query_s = scaler.transform(X_query)

    nn = NearestNeighbors(n_neighbors=min(k, len(df_corpus)), metric="euclidean")
    nn.fit(X_corpus_s)
    distances, indices = nn.kneighbors(X_query_s)

    results = df_corpus.iloc[indices[0]].copy()
    results["distance"] = distances[0]
    results["rank"] = range(1, len(results) + 1)
    return results[["path", "distance", "rank"]]
```

---

## 5. Examples

### Example A — Batch Feature Extraction and k-NN Music Similarity Search

```python
import os

# --- Configuration ---
CORPUS_DIR = "/data/music_corpus"          # directory with audio files
QUERY_FILE = "/data/query/my_song.wav"     # song to match against corpus
FEATURES_CACHE = "/tmp/corpus_features.parquet"
K_NEIGHBORS = 5

# Step 1: Extract and cache features (skip if cache exists)
if os.path.exists(FEATURES_CACHE):
    import pandas as pd
    df_corpus = pd.read_parquet(FEATURES_CACHE)
    print(f"Loaded cached features: {len(df_corpus)} tracks")
else:
    df_corpus = batch_extract_features(
        corpus_dir=CORPUS_DIR,
        output_path=FEATURES_CACHE,
    )

# Step 2: Compute full similarity matrix (optional — useful for playlist generation)
sim_matrix = compute_similarity_matrix(df_corpus)
print(f"Similarity matrix shape: {sim_matrix.shape}")

# Step 3: Find similar tracks
results = find_similar_tracks(
    query_path=QUERY_FILE,
    corpus_dir=CORPUS_DIR,
    k=K_NEIGHBORS,
)
print("\nTop similar tracks:")
print(results.to_string(index=False))
```

### Example B — Tempo and Key Estimation for a Music Collection

```python
import pathlib
import pandas as pd
import numpy as np
import librosa

def estimate_key(chroma_mean: np.ndarray) -> str:
    """
    Estimate musical key using the Krumhansl-Schmuckler key-finding algorithm
    applied to a 12-element chroma mean vector.
    """
    # Major and minor key profiles (Krumhansl 1990)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    best_corr, best_key = -np.inf, "C major"
    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
            corr = np.corrcoef(rotated, profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_key = f"{note_names[shift]} {mode}"
    return best_key


def analyze_collection(music_dir: str, sr: int = 22050) -> pd.DataFrame:
    """
    Estimate tempo and key for every track in music_dir.

    Returns a DataFrame with columns: path, tempo_bpm, key, duration_s.
    """
    audio_ext = {".wav", ".flac", ".mp3", ".ogg"}
    paths = [p for p in pathlib.Path(music_dir).rglob("*") if p.suffix.lower() in audio_ext]

    records = []
    for p in paths:
        try:
            y, sr_out = librosa.load(str(p), sr=sr, mono=True)
            duration = len(y) / sr_out
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr_out)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr_out)
            key = estimate_key(np.mean(chroma, axis=1))
            records.append({
                "path": str(p),
                "filename": p.name,
                "tempo_bpm": round(float(tempo), 1),
                "key": key,
                "duration_s": round(duration, 2),
            })
            print(f"  {p.name}: {tempo:.1f} BPM, {key}")
        except Exception as exc:
            print(f"  WARN {p.name}: {exc}")

    df = pd.DataFrame(records)

    # Summary statistics
    if not df.empty:
        print(f"\nCollection summary ({len(df)} tracks):")
        print(f"  Avg tempo : {df['tempo_bpm'].mean():.1f} BPM")
        print(f"  Most common key: {df['key'].value_counts().index[0]}")
        print(f"  Total duration : {df['duration_s'].sum()/60:.1f} minutes")

    return df


if __name__ == "__main__":
    MUSIC_DIR = "/data/my_collection"
    df_collection = analyze_collection(MUSIC_DIR)
    df_collection.to_csv("/tmp/collection_analysis.csv", index=False)
    print("Saved to /tmp/collection_analysis.csv")
```

---

## 6. Tips and Gotchas

- **Sample rate**: Always specify `sr` explicitly. Default `sr=22050` is fine for most
  MIR tasks; use `sr=44100` when you need full bandwidth (e.g., audio quality assessment).
- **MP3 support**: Requires `audioread` (bundled) or `ffmpeg` on PATH. FLAC/WAV work
  out-of-the-box via `soundfile`.
- **Memory**: For long files, use `librosa.load(..., offset=start, duration=length)` to
  process in chunks.
- **Normalization**: Always `StandardScaler` before k-NN — raw MFCC values dwarf chroma
  in magnitude.
- **Beat tracking**: Works best for music with a clear rhythmic pulse. Noisy or ambient
  recordings will yield unreliable BPM estimates.
- **YIN pitch detection**: `fmin` and `fmax` must bracket the expected fundamental.
  For singing voice use `fmin=80`, `fmax=600`.
- **HPSS margin**: Increase `margin` parameter in `librosa.effects.hpss` for cleaner
  separation at the cost of computation: `librosa.effects.hpss(y, margin=3.0)`.

---

## 7. References

- McFee et al. (2015). *librosa: Audio and Music Signal Analysis in Python*.
  SciPy Proceedings. https://librosa.org
- Krumhansl, C. L. (1990). *Cognitive Foundations of Musical Pitch*. Oxford UP.
- De Cheveigné & Kawahara (2002). YIN, a fundamental frequency estimator. JASA.
