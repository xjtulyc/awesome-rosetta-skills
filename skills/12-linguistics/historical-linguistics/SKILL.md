---
name: historical-linguistics
description: >
  Use this Skill for historical linguistics: sound correspondences, cognate
  detection, phylogenetic tree inference, and Swadesh list analysis in Python.
tags:
  - linguistics
  - historical-linguistics
  - phylogenetics
  - cognates
  - sound-change
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
    - pandas>=2.0
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
    - lingpy>=2.6
last_updated: "2026-03-17"
status: "stable"
---

# Historical Linguistics Analysis

> **One-line summary**: Detect cognates, model sound correspondences, infer phylogenetic language trees, and compute lexicostatistical distances from Swadesh list data using LingPy and scipy.

---

## When to Use This Skill

- When computing cognate sets from multilingual Swadesh list data
- When detecting regular sound correspondences between language pairs
- When inferring phylogenetic trees using Neighbor Joining or UPGMA
- When computing lexicostatistical distances (% shared cognates)
- When reconstructing Proto-language vocabulary via the comparative method
- When building distance matrices for historical subgrouping

**Trigger keywords**: historical linguistics, comparative method, cognates, sound correspondence, proto-language, phylogenetic tree, Swadesh list, lexicostatistics, glottochronology, language family, Neighbor Joining, LingPy, sound change, regular correspondence

---

## Background & Key Concepts

### The Comparative Method

Systematic comparison of related languages to:
1. Identify cognates (words inherited from a common ancestor)
2. Establish regular sound correspondences
3. Reconstruct Proto-language phonemes (via the correspondence sets)
4. Propose a subgrouping based on shared innovations

### Lexicostatistics / Glottochronology

Estimate time depth from percentage of shared cognates on Swadesh 100/200-word list.

Swadesh formula: $t = \frac{\ln C}{\ln r}$ where $C$ = proportion of cognates, $r$ = retention rate per millennium (~0.86 for basic vocabulary).

### Edit Distance for Cognate Detection

Normalized Levenshtein distance between IPA transcriptions is used as a proxy for cognate probability. Words with distance < threshold (typically 0.5–0.6 after alignment) are candidate cognates.

---

## Environment Setup

### Install Dependencies

```bash
pip install pandas>=2.0 numpy>=1.24 scipy>=1.11 matplotlib>=3.7 lingpy>=2.6
```

### Verify Installation

```python
try:
    import lingpy
    print(f"LingPy version: {lingpy.__version__}")
    from lingpy import Wordlist
except ImportError:
    print("LingPy not available; using manual implementations")

# Manual Levenshtein implementation
def levenshtein(a, b):
    """Compute Levenshtein edit distance between two strings/lists."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] if a[i-1] == b[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

print(f"levenshtein('cat','bat') = {levenshtein('cat','bat')}")  # Expected: 1
```

---

## Core Workflow

### Step 1: Swadesh List Analysis and Lexicostatistics

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# ------------------------------------------------------------------ #
# Swadesh 50-word list for Indo-European languages (simplified)
# Words represented as phonemic transcriptions
# ------------------------------------------------------------------ #

# Sample of Swadesh concepts (concept: {language: IPA form})
swadesh_data = {
    'one':    {'Latin':'uːnus', 'French':'œ̃', 'Spanish':'uno', 'Italian':'uno',
               'German':'aɪ̯ns', 'English':'wʌn', 'Dutch':'eːn'},
    'two':    {'Latin':'duːo', 'French':'dø', 'Spanish':'dos', 'Italian':'due',
               'German':'tsvaɪ̯', 'English':'tuː', 'Dutch':'tweː'},
    'water':  {'Latin':'akʷa', 'French':'oː', 'Spanish':'aɣwa', 'Italian':'akːwa',
               'German':'vasɐ', 'English':'wɔːtɐ', 'Dutch':'vatɛr'},
    'fire':   {'Latin':'ignis', 'French':'fø', 'Spanish':'fweɣo', 'Italian':'fwɔːko',
               'German':'fɔʏ̯ɐ', 'English':'faɪ̯ɐ', 'Dutch':'vuːr'},
    'mother': {'Latin':'maːter', 'French':'mɛːr', 'Spanish':'maðre', 'Italian':'madːre',
               'German':'mutɐ', 'English':'mʌðɐ', 'Dutch':'moːdɛr'},
    'father': {'Latin':'pater', 'French':'pɛːr', 'Spanish':'padre', 'Italian':'padre',
               'German':'fatɐ', 'English':'fɑːðɐ', 'Dutch':'vadɛr'},
    'eye':    {'Latin':'okulus', 'French':'œɪ̯', 'Spanish':'oxo', 'Italian':'okːjo',
               'German':'aʊ̯ɡə', 'English':'aɪ̯', 'Dutch':'oːɣ'},
    'new':    {'Latin':'nowus', 'French':'nɔf', 'Spanish':'nweβo', 'Italian':'nwɔvo',
               'German':'nɔʏ̯', 'English':'njuː', 'Dutch':'niuː'},
    'night':  {'Latin':'noks', 'French':'nɥɪ̯', 'Spanish':'notʃe', 'Italian':'notːe',
               'German':'naxt', 'English':'naɪ̯t', 'Dutch':'naxt'},
    'sun':    {'Latin':'soːl', 'French':'sɔlɛɪ̯', 'Spanish':'sol', 'Italian':'sole',
               'German':'zɔnə', 'English':'sʌn', 'Dutch':'zɔn'},
}

languages = ['Latin', 'French', 'Spanish', 'Italian', 'German', 'English', 'Dutch']
n_langs = len(languages)
n_concepts = len(swadesh_data)

# ---- Compute normalized edit distances -------------------------- #
def normalized_edit_distance(s1, s2):
    """Levenshtein distance normalized by max length."""
    m, n = len(s1), len(s2)
    if m == 0 and n == 0:
        return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] if s1[i-1] == s2[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n] / max(m, n)

# Pairwise distance matrix (averaged over all concepts)
dist_matrix = np.zeros((n_langs, n_langs))
cognate_matrix = np.zeros((n_langs, n_langs))  # 1 = likely cognate (dist < 0.5)

for i, lang1 in enumerate(languages):
    for j, lang2 in enumerate(languages):
        if i == j:
            continue
        dists = []
        for concept, words in swadesh_data.items():
            d = normalized_edit_distance(words[lang1], words[lang2])
            dists.append(d)
        dist_matrix[i, j] = np.mean(dists)
        cognate_matrix[i, j] = np.mean([d < 0.5 for d in dists])

print("Normalized phonological distance matrix:")
df_dist = pd.DataFrame(dist_matrix, index=languages, columns=languages)
print(df_dist.round(3))

print("\nEstimated % shared cognates (dist < 0.5):")
df_cog = pd.DataFrame(cognate_matrix * 100, index=languages, columns=languages)
print(df_cog.round(1))

# ---- Glottochronological time estimates ----------------------- #
r = 0.86  # Swadesh retention rate per millennium
print("\nEstimated divergence times (millennia BP):")
for i, lang1 in enumerate(languages):
    for j, lang2 in enumerate(languages):
        if j <= i:
            continue
        pct_cognates = cognate_matrix[i, j]
        if pct_cognates > 0:
            t = np.log(pct_cognates) / np.log(r)
            print(f"  {lang1}–{lang2}: {t:.1f} millennia  ({pct_cognates*100:.0f}% cognates)")
```

### Step 2: Phylogenetic Tree Inference

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from scipy.spatial.distance import squareform

# ------------------------------------------------------------------ #
# Infer language phylogeny using Neighbor Joining (via UPGMA here)
# ------------------------------------------------------------------ #

# Use the distance matrix computed above
condensed = squareform(dist_matrix)

# UPGMA linkage (approximates neighbor joining for well-nested data)
Z = linkage(condensed, method='average')

fig, ax = plt.subplots(figsize=(9, 6))
dendrogram(
    Z,
    labels=languages,
    orientation='right',
    leaf_font_size=12,
    color_threshold=0.4,
    ax=ax,
)
ax.set_xlabel("Phonological distance")
ax.set_title("Indo-European Language Phylogeny\n(UPGMA based on Swadesh list edit distances)")
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("language_phylogeny.png", dpi=150)
plt.show()

print("""
Expected subgroupings:
  Romance: French, Spanish, Italian (← Latin)
  Germanic: German, English, Dutch
  Outlier: Latin (ancestor)
""")

# ---- Bootstrap support ----------------------------------------- #
n_boot = 100
np.random.seed(42)
concepts = list(swadesh_data.keys())
n_concepts = len(concepts)

bootstrap_distances = []
for _ in range(n_boot):
    # Resample concepts with replacement
    sample_concepts = np.random.choice(concepts, n_concepts, replace=True)
    dist_boot = np.zeros((n_langs, n_langs))
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i == j:
                continue
            dists = [normalized_edit_distance(swadesh_data[c][lang1], swadesh_data[c][lang2])
                     for c in sample_concepts]
            dist_boot[i, j] = np.mean(dists)
    bootstrap_distances.append(dist_boot)

# Compute bootstrap support for Romance clade (French, Spanish, Italian)
romance = ['French', 'Spanish', 'Italian']
romance_idx = [languages.index(l) for l in romance]
non_romance_idx = [i for i in range(n_langs) if i not in romance_idx]

romance_support = 0
for d in bootstrap_distances:
    # Check if Romance languages cluster together (internal distances < Romance-Germanic distances)
    romance_internal = np.mean([d[i,j] for i in romance_idx for j in romance_idx if i!=j])
    romance_external = np.mean([d[i,j] for i in romance_idx for j in non_romance_idx])
    if romance_internal < romance_external:
        romance_support += 1

print(f"Bootstrap support for Romance clade: {romance_support}/{n_boot} ({romance_support:.0f}%)")
```

### Step 3: Sound Correspondence Detection

```python
import pandas as pd
import numpy as np
from collections import defaultdict

# ------------------------------------------------------------------ #
# Identify regular sound correspondences between two related languages
# (the basis of the comparative method)
# ------------------------------------------------------------------ #

# Known cognate pairs: Latin → Spanish (simplified)
latin_spanish_cognates = [
    # (Latin, Spanish, gloss)
    ('pater', 'padre', 'father'),
    ('mater', 'madre', 'mother'),
    ('uita', 'vida', 'life'),
    ('locus', 'lugar', 'place'),
    ('pluuia', 'lluvia', 'rain'),
    ('planta', 'planta', 'plant'),
    ('porta', 'puerta', 'door'),
    ('nox', 'noche', 'night'),
    ('luna', 'luna', 'moon'),
    ('terra', 'tierra', 'earth'),
    ('focus', 'fuego', 'fire/hearth'),
    ('filius', 'hijo', 'son'),
    ('filia', 'hija', 'daughter'),
    ('oculus', 'ojo', 'eye'),
    ('auris', 'oreja', 'ear'),
]

def align_sequences(s1, s2):
    """
    Simple global sequence alignment (Needleman-Wunsch, match=1, mismatch=-1, gap=-1).
    Returns aligned sequences.
    """
    m, n = len(s1), len(s2)
    # Score matrix
    score = np.zeros((m+1, n+1))
    score[0, :] = np.arange(n+1) * -1
    score[:, 0] = np.arange(m+1) * -1

    for i in range(1, m+1):
        for j in range(1, n+1):
            match = score[i-1,j-1] + (1 if s1[i-1]==s2[j-1] else -1)
            score[i,j] = max(match, score[i-1,j]-1, score[i,j-1]-1)

    # Traceback
    a1, a2 = [], []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i,j] == score[i-1,j-1] + (1 if s1[i-1]==s2[j-1] else -1):
            a1.append(s1[i-1]); a2.append(s2[j-1]); i-=1; j-=1
        elif i > 0 and score[i,j] == score[i-1,j]-1:
            a1.append(s1[i-1]); a2.append('-'); i-=1
        else:
            a1.append('-'); a2.append(s2[j-1]); j-=1

    return list(reversed(a1)), list(reversed(a2))

# Count correspondence pairs
correspondences = defaultdict(int)
for lat, spa, gloss in latin_spanish_cognates:
    a1, a2 = align_sequences(list(lat), list(spa))
    for c1, c2 in zip(a1, a2):
        correspondences[(c1, c2)] += 1

# Sort by frequency
corr_df = pd.DataFrame(
    [(c1, c2, count) for (c1,c2), count in sorted(correspondences.items(), key=lambda x: -x[1])],
    columns=['Latin', 'Spanish', 'Count']
)

print("Sound Correspondences: Latin → Spanish (top 20)")
print(corr_df[corr_df['Latin'] != corr_df['Spanish']].head(20).to_string(index=False))

# Highlight regular correspondences (appear ≥3 times)
regular = corr_df[(corr_df['Count'] >= 3) & (corr_df['Latin'] != corr_df['Spanish'])]
print(f"\nRegular correspondences (n≥3):")
for _, row in regular.iterrows():
    print(f"  Latin /{row['Latin']}/ → Spanish /{row['Spanish']}/ ({row['Count']}× attested)")
```

---

## Advanced Usage

### Automated Cognate Detection with LingPy

```python
# ------------------------------------------------------------------ #
# Using LingPy's SCA (Sound Class Alignment) for cognate detection
# ------------------------------------------------------------------ #

try:
    from lingpy import Wordlist, LexStat
    import lingpy.compare.partial as partial

    # Create a CLDF-compatible wordlist
    # (requires proper input format — see LingPy documentation)
    print("LingPy cognate detection:")
    print("  1. Load Wordlist: wl = Wordlist('my_data.tsv')")
    print("  2. Run LexStat: lex = LexStat(wl)")
    print("  3. Cluster cognates: lex.cluster(method='sca', threshold=0.45)")
    print("  4. Inspect cognate sets: lex.output('tsv', filename='results')")
except ImportError:
    print("LingPy not installed. Install with: pip install lingpy")
    print("Using manual implementations instead.")
```

---

## Troubleshooting

### LingPy import errors

```bash
pip install lingpy --upgrade
# On some systems:
pip install numpy scipy matplotlib pandas networkx
pip install lingpy
```

### Alignment produces all-gap output

**Cause**: Sequences are in different encoding (Unicode vs. ASCII).

**Fix**: Normalize to Unicode NFD and split by character:
```python
import unicodedata
def ipa_chars(word):
    """Split IPA string into characters (handles diacritics)."""
    return list(unicodedata.normalize('NFD', word))
```

### Dendrogram labels overlap

```python
fig, ax = plt.subplots(figsize=(12, 8))
dendrogram(Z, labels=languages, leaf_rotation=45, leaf_font_size=10, ax=ax)
plt.subplots_adjust(bottom=0.15)
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| lingpy | 2.6 | Requires Python 3.8+; networkx ≥ 3.0 |
| scipy | 1.11, 1.12 | `linkage` and `dendrogram` stable |
| numpy | 1.24, 1.26 | No known issues |

---

## External Resources

### Official Documentation

- [LingPy documentation](https://lingpy.org)
- [CLDF datasets](https://cldf.clld.org)

### Key Papers

- Swadesh, M. (1952). *Lexico-statistic dating of prehistoric ethnic contacts*. Proceedings of the American Philosophical Society.
- List, J.-M. et al. (2017). *LingPy: A Python library for historical linguistics*. Journal of Open Source Software.

---

## Examples

### Example 1: Proto-Language Reconstruction

```python
import pandas as pd

# Comparative reconstruction: identify the proto-phoneme from correspondences
# Latin /p/ → Spanish /p/ before vowels (preserved)
# Latin /p/ → Spanish /β̞/ between vowels (lenition)
# Latin /kt/ → Spanish /tʃ/ (palatalization)

reconstructions = [
    {'Latin': 'pater', 'Spanish': 'padre', 'Correspondence': 'L. p- / Sp. p- (onset, regular)'},
    {'Latin': 'caput', 'Spanish': 'cabo',  'Correspondence': 'L. -p- / Sp. -β̞- (intervocalic lenition)'},
    {'Latin': 'noctem','Spanish': 'noche', 'Correspondence': 'L. -ct- / Sp. -tʃ- (palatalization)'},
    {'Latin': 'filium','Spanish': 'hijo',  'Correspondence': 'L. f- / Sp. h- (Spanish F→h shift)'},
    {'Latin': 'factum','Spanish': 'hecho', 'Correspondence': 'L. -ct- → Sp. -tʃ-; L. f- → Sp. h-'},
]
df_rc = pd.DataFrame(reconstructions)
print("Sound Changes: Proto-Romance (Latin) → Spanish")
print(df_rc.to_string(index=False))
```

### Example 2: Language Distance Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(dist_matrix, cmap='YlOrRd', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='Mean phonological distance')
ax.set_xticks(range(n_langs)); ax.set_xticklabels(languages, rotation=45, ha='right')
ax.set_yticks(range(n_langs)); ax.set_yticklabels(languages)
ax.set_title("Pairwise Phonological Distance Matrix")
for i in range(n_langs):
    for j in range(n_langs):
        ax.text(j, i, f"{dist_matrix[i,j]:.2f}", ha='center', va='center', fontsize=8,
                color='white' if dist_matrix[i,j] > 0.6 else 'black')
plt.tight_layout(); plt.savefig("distance_heatmap.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
