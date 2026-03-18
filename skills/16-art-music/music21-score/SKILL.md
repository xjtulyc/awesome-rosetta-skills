---
name: music21-score
description: >
  Use this Skill for computational musicology with music21: score analysis,
  harmonic reduction, melodic contour, counterpoint checking, and corpus comparison.
tags:
  - musicology
  - music21
  - harmonic-analysis
  - corpus-analysis
  - MIDI
  - MusicXML
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
    - music21>=9.1
    - numpy>=1.23
    - matplotlib>=3.6
    - pandas>=1.5
last_updated: "2026-03-18"
status: stable
---

# Computational Musicology with music21

> **TL;DR** — Analyse musical scores programmatically: load MusicXML/MIDI/ABC/Kern,
> run Roman numeral harmonic analysis, compute melodic contour, detect counterpoint
> errors, and query the built-in Bach chorale corpus.

---

## When to Use

Use this Skill whenever you need to:

- Parse and analyse symbolic music files (MusicXML, MIDI, ABC notation, Humdrum Kern)
- Extract harmonic progressions and Roman numeral sequences from tonal music
- Compute melodic contour reduction and interval statistics
- Detect counterpoint rule violations (parallel fifths/octaves, voice crossing)
- Query the music21 built-in corpus (Bach chorales, classical works) for statistical analysis
- Compare melodic similarity across pieces using edit distance on interval sequences
- Build n-gram models of harmonic progressions for style analysis

| Task | music21 Entry Point |
|---|---|
| Load a score file | `converter.parse(filepath)` |
| Roman numeral analysis | `harmony.romanNumeralFromChord()` |
| Key detection | `score.analyze('key')` or `key.analyze('Krumhansl')` |
| Corpus queries | `corpus.search(composer='bach')` |
| Pitch class sets | `chord.Chord.pitchClasses` |

---

## Background

**music21** is an MIT-developed Python toolkit for computational musicology. Its object
model mirrors Western music notation: a `Score` contains one or more `Part` objects;
each `Part` is a sequence of `Measure` objects; each `Measure` contains `Note`,
`Chord`, `Rest`, and other `GeneralNote` subclasses.

### Core Object Hierarchy

```
Score
  └─ Part (e.g., Soprano, Alto, Tenor, Bass)
       └─ Measure (bar number, time signature, key signature)
            └─ Note / Chord / Rest
                  ├─ Note: pitch (Pitch object), duration (Duration), offset
                  └─ Chord: list of pitches, root(), inversion(), pitchClasses
```

### Harmonic Analysis Concepts

- **Roman numeral analysis**: Each chord is labelled relative to the local key
  (e.g., I, IV, V7, ii6). `harmony.romanNumeralFromChord(chord, key)` returns a
  `RomanNumeral` object with `.figure` (string label) and `.scaleDegree`.
- **Pitch class sets**: Ignore octave and register; a C major triad = {0, 4, 7}.
  Useful for atonal / post-tonal analysis.
- **Krumhansl-Schmuckler key-finding**: Correlates the distribution of pitch-class
  durations against major/minor key profiles to estimate the most likely key.

### Melodic Contour

Contour reduction (Morris 1987) simplifies a melody by retaining only local maxima
and minima. This enables comparison of melodic shape across transpositions and
rhythmic variations. Edit distance on the sequence of melodic intervals (in semitones)
gives a measure of melodic similarity.

### Counterpoint Rules (Two-Voice)

The standard prohibition rules in strict two-voice counterpoint:

| Rule | Definition |
|---|---|
| Parallel fifths | Two voices move in same direction, both intervals are P5 |
| Parallel octaves | Same motion, both intervals are P8/unison |
| Voice crossing | Lower voice pitch exceeds upper voice pitch |
| Hidden fifths | Outer voices approach P5 by similar motion |

---

## Environment Setup

```bash
# Create a dedicated conda environment
conda create -n music21-env python=3.11 -y
conda activate music21-env
pip install "music21>=9.1" "numpy>=1.23" "matplotlib>=3.6" "pandas>=1.5"

# Optional: install MuseScore for PDF/PNG rendering
# Ubuntu/Debian:
sudo apt-get install musescore3
# macOS:
brew install --cask musescore

# Configure music21 to find MuseScore (run once)
python -c "import music21; music21.environment.UserSettings()['musicxmlPath'] = '/usr/bin/musescore3'"
```

Verify installation:

```python
import music21
from music21 import corpus, converter, harmony, key
print(f"music21 version: {music21.__version__}")

# Quick smoke test: load a Bach chorale from the built-in corpus
bwv66 = corpus.parse('bach/bwv66.6')
print(f"Parts: {[p.partName for p in bwv66.parts]}")
print(f"Measures: {len(list(bwv66.parts[0].getElementsByClass('Measure')))}")
```

---

## Core Workflow

### Step 1 — Load a Score and Run Roman Numeral Analysis

```python
import pandas as pd
from music21 import corpus, harmony, roman, chord, key as m21key


def analyze_bach_chorale_harmony(bwv_id: str = 'bach/bwv66.6') -> pd.DataFrame:
    """
    Load a Bach chorale from the built-in corpus and perform Roman numeral
    harmonic analysis on every beat-level chord slice.

    Args:
        bwv_id: corpus path string, e.g. 'bach/bwv66.6'

    Returns:
        DataFrame with columns: measure, beat, chord_pitches, key, roman_numeral,
        scale_degree, chord_quality
    """
    score = corpus.parse(bwv_id)

    # Detect global key using Krumhansl-Schmuckler algorithm
    detected_key = score.analyze('key')
    print(f"Detected key: {detected_key} (confidence: {detected_key.correlationCoefficient:.3f})")

    # Reduce score to chordified version (one chord per beat position)
    chordified = score.chordify()

    records = []
    for measure in chordified.recurse().getElementsByClass('Measure'):
        m_num = measure.number
        for element in measure.notes:
            if not isinstance(element, chord.Chord):
                continue
            # Local key at this position
            local_key = element.getContextByClass('KeySignature')
            analysis_key = detected_key  # fall back to global key

            try:
                rn = roman.romanNumeralFromChord(element, analysis_key)
                figure = rn.figure
                scale_deg = rn.scaleDegree
                quality = element.commonName
            except Exception:
                figure = '?'
                scale_deg = -1
                quality = 'unknown'

            records.append({
                'measure': m_num,
                'beat': float(element.beat),
                'chord_pitches': ' '.join(str(p) for p in element.pitches),
                'key': str(detected_key),
                'roman_numeral': figure,
                'scale_degree': scale_deg,
                'chord_quality': quality,
            })

    df = pd.DataFrame(records)
    return df


def harmonic_progression_frequency(df: pd.DataFrame, n: int = 2) -> pd.Series:
    """
    Compute bigram (or n-gram) frequency of Roman numeral progressions.

    Args:
        df: Output of analyze_bach_chorale_harmony()
        n:  n-gram size (default 2 = bigrams like I->V)

    Returns:
        Series of progression counts, sorted descending.
    """
    rn_seq = df['roman_numeral'].tolist()
    ngrams = [
        ' -> '.join(rn_seq[i:i + n])
        for i in range(len(rn_seq) - n + 1)
    ]
    freq = pd.Series(ngrams).value_counts()
    return freq


# --- Run ---
df_harmony = analyze_bach_chorale_harmony('bach/bwv66.6')
print(df_harmony.head(10).to_string(index=False))

bigrams = harmonic_progression_frequency(df_harmony, n=2)
print("\nTop 10 harmonic bigrams:")
print(bigrams.head(10))
```

### Step 2 — Melodic Contour and Interval Histogram

```python
import numpy as np
import matplotlib.pyplot as plt
from music21 import corpus, note, interval


def extract_melodic_features(bwv_id: str = 'bach/bwv66.6',
                              part_index: int = 0) -> dict:
    """
    Extract melodic interval sequence, contour reduction, and interval histogram
    for a single part of a score.

    Args:
        bwv_id:     corpus path string
        part_index: which part to analyse (0=Soprano in Bach chorales)

    Returns:
        dict with keys: intervals_semitones, contour, interval_counts,
        mean_interval, std_interval
    """
    score = corpus.parse(bwv_id)
    part = score.parts[part_index]

    # Collect all Note objects (exclude rests and non-pitched)
    notes = [n for n in part.flat.notes if isinstance(n, note.Note)]

    # Compute melodic intervals in semitones
    semitones = []
    for i in range(len(notes) - 1):
        intv = interval.Interval(notes[i], notes[i + 1])
        semitones.append(intv.semitones)

    # Contour reduction: keep local extrema (local max and min)
    def contour_reduction(seq):
        if len(seq) < 3:
            return seq
        reduced = [seq[0]]
        for i in range(1, len(seq) - 1):
            is_local_max = seq[i] > seq[i - 1] and seq[i] > seq[i + 1]
            is_local_min = seq[i] < seq[i - 1] and seq[i] < seq[i + 1]
            if is_local_max or is_local_min:
                reduced.append(seq[i])
        reduced.append(seq[-1])
        return reduced

    midi_pitches = [n.pitch.midi for n in notes]
    contour = contour_reduction(midi_pitches)

    # Interval histogram
    interval_counts = {}
    for s in semitones:
        interval_counts[s] = interval_counts.get(s, 0) + 1

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pitch contour
    axes[0].plot(midi_pitches, linewidth=0.7, color='steelblue', label='Original')
    reduced_x = np.linspace(0, len(midi_pitches) - 1, len(contour))
    axes[0].plot(reduced_x, contour, 'ro-', linewidth=1.5, markersize=4, label='Contour reduction')
    axes[0].set_title(f'Melodic Contour — {bwv_id} Part {part_index}')
    axes[0].set_xlabel('Note index')
    axes[0].set_ylabel('MIDI pitch')
    axes[0].legend()

    # Interval histogram
    bins = sorted(interval_counts.keys())
    counts = [interval_counts[b] for b in bins]
    axes[1].bar(bins, counts, color='coral', edgecolor='black', linewidth=0.5)
    axes[1].set_title('Melodic Interval Histogram (semitones)')
    axes[1].set_xlabel('Interval (semitones)')
    axes[1].set_ylabel('Frequency')

    fig.tight_layout()
    fig.savefig('melodic_contour.png', dpi=150)
    plt.close(fig)

    return {
        'intervals_semitones': semitones,
        'contour': contour,
        'interval_counts': interval_counts,
        'mean_interval': float(np.mean(semitones)) if semitones else 0.0,
        'std_interval': float(np.std(semitones)) if semitones else 0.0,
    }


# --- Run ---
features = extract_melodic_features('bach/bwv66.6', part_index=0)
print(f"Mean interval: {features['mean_interval']:.2f} semitones")
print(f"Std interval:  {features['std_interval']:.2f} semitones")
print(f"Contour length after reduction: {len(features['contour'])} points")
```

### Step 3 — Parallel Fifths Detector in Two-Voice Counterpoint

```python
from music21 import corpus, chord, interval, note
from typing import List, Tuple


def detect_parallel_intervals(
    score_path: str = 'bach/bwv66.6',
    upper_part_idx: int = 0,
    lower_part_idx: int = 1,
    target_semitones: List[int] = None,
) -> List[dict]:
    """
    Detect parallel fifths and parallel octaves between two voices.

    A parallel fifth/octave occurs when two consecutive harmonic intervals
    of the same size are approached by both voices moving in the same direction.

    Args:
        score_path:       corpus path or file path
        upper_part_idx:   index of the upper voice part
        lower_part_idx:   index of the lower voice part
        target_semitones: list of interval sizes to flag (default: [7, 12] = P5, P8)

    Returns:
        List of dicts describing each violation: measure, beat, upper_notes,
        lower_notes, interval_size, violation_type
    """
    if target_semitones is None:
        target_semitones = [7, 12]  # Perfect fifth, perfect octave/unison

    # Load from corpus or file
    try:
        score = corpus.parse(score_path)
    except Exception:
        from music21 import converter
        score = converter.parse(score_path)

    upper = score.parts[upper_part_idx].flat.notes
    lower = score.parts[lower_part_idx].flat.notes

    # Align notes by offset
    upper_notes = [(n.offset, n) for n in upper if isinstance(n, note.Note)]
    lower_notes = [(n.offset, n) for n in lower if isinstance(n, note.Note)]

    # Build dict: offset -> pitch
    upper_dict = {off: n for off, n in upper_notes}
    lower_dict = {off: n for off, n in lower_notes}

    shared_offsets = sorted(set(upper_dict.keys()) & set(lower_dict.keys()))

    violations = []
    for i in range(len(shared_offsets) - 1):
        off1 = shared_offsets[i]
        off2 = shared_offsets[i + 1]

        u1, u2 = upper_dict[off1], upper_dict[off2]
        l1, l2 = lower_dict[off1], lower_dict[off2]

        # Harmonic intervals at each timepoint
        harm_int1 = abs(u1.pitch.midi - l1.pitch.midi) % 12
        harm_int2 = abs(u2.pitch.midi - l2.pitch.midi) % 12

        # Motion directions
        upper_motion = u2.pitch.midi - u1.pitch.midi
        lower_motion = l2.pitch.midi - l1.pitch.midi
        same_direction = (upper_motion > 0 and lower_motion > 0) or \
                         (upper_motion < 0 and lower_motion < 0)

        for target in target_semitones:
            # Check modulo 12 for compound intervals
            if harm_int1 == target % 12 and harm_int2 == target % 12 and same_direction:
                label = 'parallel fifths' if target == 7 else 'parallel octaves'
                # Get measure number
                m_num = u2.measureNumber if hasattr(u2, 'measureNumber') else '?'
                violations.append({
                    'measure': m_num,
                    'beat_offset': off2,
                    'upper_notes': f"{u1.nameWithOctave}->{u2.nameWithOctave}",
                    'lower_notes': f"{l1.nameWithOctave}->{l2.nameWithOctave}",
                    'interval_semitones': target,
                    'violation_type': label,
                })

    return violations


# --- Run ---
violations = detect_parallel_intervals('bach/bwv66.6', upper_part_idx=0, lower_part_idx=3)
if violations:
    print(f"Found {len(violations)} parallel interval violations:")
    for v in violations[:10]:
        print(f"  Measure {v['measure']}: {v['violation_type']} — "
              f"upper {v['upper_notes']} / lower {v['lower_notes']}")
else:
    print("No parallel fifths or octaves detected between selected voices.")
```

---

## Advanced Usage

### Corpus Search and Comparative Analysis

```python
from music21 import corpus
import pandas as pd


def compare_chorale_keys() -> pd.DataFrame:
    """
    Search all Bach chorales in the corpus and compare key distribution.
    Returns a DataFrame of detected keys and their frequencies.
    """
    results = []
    # Get all Bach chorale paths
    chorale_paths = corpus.getComposer('bach')

    for path in chorale_paths[:50]:  # limit for demo
        try:
            score = corpus.parse(path)
            detected_key = score.analyze('key')
            results.append({
                'path': str(path),
                'key': str(detected_key),
                'mode': detected_key.mode,
                'tonic': detected_key.tonic.name,
                'confidence': round(detected_key.correlationCoefficient, 3),
            })
        except Exception as e:
            continue

    df = pd.DataFrame(results)
    key_freq = df['key'].value_counts().head(20)
    print("Most common keys in Bach chorales:")
    print(key_freq)
    return df
```

### Melodic Similarity with Edit Distance

```python
def melodic_edit_distance(seq1: list, seq2: list) -> int:
    """
    Compute Levenshtein edit distance between two melodic interval sequences.
    Lower distance = more similar melodies.

    Args:
        seq1: list of semitone intervals for melody 1
        seq2: list of semitone intervals for melody 2

    Returns:
        Integer edit distance.
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[m][n]
```

### Voice Leading Statistics

```python
def voice_leading_stats(score_path: str) -> dict:
    """
    Summarize voice leading motion statistics for a four-voice score:
    fraction of oblique, contrary, similar, and parallel motion.
    """
    from music21 import corpus, note as m21note, converter

    try:
        score = corpus.parse(score_path)
    except Exception:
        score = converter.parse(score_path)

    parts = score.parts
    if len(parts) < 2:
        return {}

    motion_counts = {'contrary': 0, 'parallel': 0, 'similar': 0, 'oblique': 0}

    for part_a_idx in range(len(parts) - 1):
        for part_b_idx in range(part_a_idx + 1, len(parts)):
            notes_a = [n for n in parts[part_a_idx].flat.notes if isinstance(n, m21note.Note)]
            notes_b = [n for n in parts[part_b_idx].flat.notes if isinstance(n, m21note.Note)]
            min_len = min(len(notes_a), len(notes_b)) - 1

            for i in range(min_len):
                ma = notes_a[i + 1].pitch.midi - notes_a[i].pitch.midi
                mb = notes_b[i + 1].pitch.midi - notes_b[i].pitch.midi

                if ma == 0 or mb == 0:
                    motion_counts['oblique'] += 1
                elif ma > 0 and mb < 0 or ma < 0 and mb > 0:
                    motion_counts['contrary'] += 1
                elif ma == mb:
                    motion_counts['parallel'] += 1
                else:
                    motion_counts['similar'] += 1

    total = sum(motion_counts.values())
    return {k: round(v / total, 3) if total else 0.0 for k, v in motion_counts.items()}
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `SubConverterException: Cannot find a path to MuseScore` | MuseScore not installed or not on PATH | Install MuseScore or use `score.show('text')` for text output |
| `CorpusException: Could not find path 'bach/...'` | Incorrect corpus path | Use `corpus.search('bach')` to list available paths |
| `AttributeError: 'Rest' object has no attribute 'pitch'` | Iterating notes includes rests | Filter with `isinstance(n, note.Note)` |
| `romanNumeralFromChord()` returns wrong figure | Chord is ambiguous or atonal | Pass explicit key: `roman.romanNumeralFromChord(ch, key.Key('C'))` |
| `music21` slow on large corpus | Parsing XML is CPU-bound | Use `corpus.parse(..., forceSource=False)` to use the pickle cache |
| Parallel detector misses compound intervals | Modulo 12 reduction conflates intervals | Remove `% 12` and compare exact semitone distances |

---

## External Resources

- music21 Documentation: <https://web.mit.edu/music21/doc/>
- music21 User's Guide: <https://web.mit.edu/music21/doc/usersGuide/>
- Corpus Browser: <https://web.mit.edu/music21/doc/about/referenceCorpus.html>
- Cuthbert, M.S. & Ariza, C. (2010). "music21: A Toolkit for Computer-Aided Musicology."
  Proceedings of ISMIR 2010. <http://ismir2010.ismir.net/proceedings/ismir2010-108.pdf>
- Morris, R.D. (1987). *Composition with Pitch-Classes*. Yale University Press.
- Humdrum Toolkit: <https://www.humdrum.org/>
- jSymbolic feature extraction: <http://jmir.sourceforge.net/jSymbolic.html>

---

## Examples

### Example 1 — Full Bach Chorale Harmonic Analysis Pipeline

```python
# End-to-end: load chorale, detect key, extract chord progressions, plot bigram frequencies
import matplotlib.pyplot as plt

bwv = 'bach/bwv66.6'
df_harm = analyze_bach_chorale_harmony(bwv)
bigrams = harmonic_progression_frequency(df_harm, n=2)

fig, ax = plt.subplots(figsize=(10, 5))
bigrams.head(15).plot(kind='barh', ax=ax, color='steelblue')
ax.invert_yaxis()
ax.set_title(f'Top 15 Harmonic Bigrams — {bwv}')
ax.set_xlabel('Count')
fig.tight_layout()
fig.savefig('harmonic_bigrams.png', dpi=150)
print("Saved harmonic_bigrams.png")

# Most common: I->V, V->I, I->IV
```

### Example 2 — Counterpoint Checker Across Multiple Chorales

```python
import pandas as pd
from music21 import corpus

chorale_paths = corpus.getComposer('bach')[:20]
all_violations = []

for path in chorale_paths:
    try:
        viols = detect_parallel_intervals(str(path), 0, 3)
        for v in viols:
            v['chorale'] = str(path).split('/')[-1]
        all_violations.extend(viols)
    except Exception:
        continue

viol_df = pd.DataFrame(all_violations) if all_violations else pd.DataFrame()
if not viol_df.empty:
    summary = viol_df.groupby(['chorale', 'violation_type']).size().unstack(fill_value=0)
    print("Parallel interval violations per chorale:")
    print(summary.to_string())
else:
    print("No violations found in sample chorales.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — harmonic analysis, contour, counterpoint checker |
