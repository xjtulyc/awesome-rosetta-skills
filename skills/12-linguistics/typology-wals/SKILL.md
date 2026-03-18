---
name: typology-wals
description: >
  Use this Skill for linguistic typology with WALS data: cross-linguistic
  feature analysis, Greenbergian universals, and typological maps in Python.
tags:
  - linguistics
  - typology
  - wals
  - cross-linguistic
  - universals
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
    - geopandas>=0.14
last_updated: "2026-03-17"
status: "stable"
---

# Linguistic Typology with WALS Data

> **One-line summary**: Analyze cross-linguistic patterns using WALS (World Atlas of Language Structures): feature co-occurrence, typological implicational universals, and geographic distribution maps.

---

## When to Use This Skill

- When analyzing cross-linguistic typological features from WALS database
- When testing Greenbergian implicational universals statistically
- When mapping the geographic distribution of linguistic features
- When computing feature correlations across language families
- When studying word order typology (SOV, SVO, VSO, etc.)
- When building phylogenetic controls for typological sampling

**Trigger keywords**: WALS, typology, Greenberg universals, implicational universal, word order, SOV SVO, cross-linguistic, linguistic geography, language features, feature co-occurrence, typological database, language families

---

## Background & Key Concepts

### WALS (World Atlas of Language Structures)

WALS contains 192 typological features for ~2,600 languages, organized in chapters covering phonology, morphology, syntax, and lexicon. Available at [wals.info](https://wals.info) and as downloadable CSV.

### Implicational Universals

If a language has feature A, then it has feature B (Greenberg 1963). Statistical test: $P(B | A) \gg P(B | \neg A)$.

Quantified by:
- **Conditional probability**: $P(B|A)$
- **Yule's Q**: $Q = (ad - bc)/(ad + bc)$ where $a,b,c,d$ are 2×2 contingency cells
- **Fisher's exact test**: tests independence

### Word Order Typology

Greenberg's Universal 1: "In declarative sentences with nominal subject and object, the dominant order is almost always one in which the subject precedes the object." (SOV ≈ 41%, SVO ≈ 35% of languages).

---

## Environment Setup

### Install Dependencies

```bash
pip install pandas>=2.0 numpy>=1.24 scipy>=1.11 matplotlib>=3.7 geopandas>=0.14
# Download WALS data:
# wget https://github.com/cldf-datasets/wals/archive/refs/heads/main.zip
```

### Load WALS Data

```python
import pandas as pd
import numpy as np

# Download from: https://wals.info/download
# Or use CLDF format from: https://github.com/cldf-datasets/wals
# For this skill, we simulate the WALS structure

# Simulate WALS-style dataframe
# In practice: pd.read_csv("wals/cldf/values.csv") + language metadata

print("WALS data structure:")
print("  values.csv: Language_ID, Parameter_ID, Value, Comment")
print("  languages.csv: ID, Name, Family, Genus, Latitude, Longitude, ISO639P3code")
print("  parameters.csv: ID, Name, Description (= typological feature)")
```

---

## Core Workflow

### Step 1: Load and Explore WALS Features

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact

# ------------------------------------------------------------------ #
# Simulate WALS data (substitute with real WALS CSV)
# Key features used: word order, morphology, phonology
# ------------------------------------------------------------------ #

np.random.seed(42)
n_langs = 400

# Language families (approximate real distribution)
families = np.random.choice(
    ['Indo-European', 'Niger-Congo', 'Austronesian', 'Sino-Tibetan',
     'Afro-Asiatic', 'Nilo-Saharan', 'Dravidian', 'Uralic', 'Turkic', 'Other'],
    n_langs,
    p=[0.15, 0.20, 0.20, 0.10, 0.08, 0.05, 0.04, 0.03, 0.05, 0.10],
)

# Simulate WALS Feature 81A: Order of Subject, Object and Verb
# Real distribution: SOV 41%, SVO 35%, VSO 9%, VOS 2%, OVS 1%, free/other 12%
word_orders = np.random.choice(
    ['SOV', 'SVO', 'VSO', 'VOS', 'OVS', 'No dominant order'],
    n_langs,
    p=[0.41, 0.35, 0.09, 0.02, 0.01, 0.12],
)

# Feature 85A: Order of Adposition and NP (preposition vs. postposition)
# Correlated with word order (SOV → postpositional)
adposition = []
for wo in word_orders:
    if wo == 'SOV':
        adposition.append(np.random.choice(['Postpositions', 'Prepositions', 'Inpositions'], p=[0.75, 0.20, 0.05]))
    elif wo == 'SVO':
        adposition.append(np.random.choice(['Postpositions', 'Prepositions', 'Inpositions'], p=[0.25, 0.70, 0.05]))
    else:
        adposition.append(np.random.choice(['Postpositions', 'Prepositions', 'Inpositions'], p=[0.40, 0.50, 0.10]))

# Feature 13A: Tone (tonal vs. non-tonal)
tone = np.random.choice(['Simple tone system', 'Complex tone system', 'No tones'],
                         n_langs, p=[0.25, 0.20, 0.55])

# Geographic coordinates (rough approximation)
lat = np.random.uniform(-40, 65, n_langs)
lon = np.random.uniform(-150, 160, n_langs)

df_wals = pd.DataFrame({
    'language_id': [f"L{i:04d}" for i in range(n_langs)],
    'family': families,
    'word_order': word_orders,
    'adposition': adposition,
    'tone': tone,
    'latitude': lat,
    'longitude': lon,
})

print("Simulated WALS dataset:")
print(f"  {len(df_wals)} languages, {df_wals['family'].nunique()} families")

# ---- Feature frequencies --------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Word order
wo_counts = df_wals['word_order'].value_counts()
colors = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#f39c12','#95a5a6']
axes[0].bar(wo_counts.index, wo_counts.values, color=colors[:len(wo_counts)], edgecolor='black', linewidth=0.7)
axes[0].set_title("Word Order Distribution\n(WALS Feature 81A)")
axes[0].set_xlabel("Word Order"); axes[0].set_ylabel("Number of Languages")
axes[0].tick_params(axis='x', rotation=20); axes[0].grid(axis='y', alpha=0.3)

# Adposition
adp_counts = df_wals['adposition'].value_counts()
axes[1].pie(adp_counts.values, labels=adp_counts.index,
            autopct='%1.1f%%', startangle=90, colors=colors[:3])
axes[1].set_title("Adposition Type\n(WALS Feature 85A)")

# Tone
tone_counts = df_wals['tone'].value_counts()
axes[2].bar(tone_counts.index, tone_counts.values, color=['#3498db','#e74c3c','#2ecc71'],
            edgecolor='black', linewidth=0.7)
axes[2].set_title("Tone System\n(WALS Feature 13A)")
axes[2].set_xlabel("Tone"); axes[2].set_ylabel("Number of Languages")
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle("Typological Feature Distributions (Simulated WALS)")
plt.tight_layout()
plt.savefig("wals_features.png", dpi=150)
plt.show()
```

### Step 2: Testing Implicational Universals

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2_contingency

# ------------------------------------------------------------------ #
# Test Greenberg's Universal: SOV predicts Postpositions
# (and vice versa)
# ------------------------------------------------------------------ #

def test_implicational_universal(df, feature_a, value_a, feature_b, value_b):
    """
    Test: if language has feature A = value_a, does it tend to have feature_b = value_b?

    Returns
    -------
    dict with contingency table, conditional probabilities, Yule's Q, Fisher p-value
    """
    has_a = df[feature_a] == value_a
    has_b = df[feature_b] == value_b

    # 2×2 contingency table
    a = (has_a & has_b).sum()          # Both A and B
    b = (has_a & ~has_b).sum()         # A but not B
    c = (~has_a & has_b).sum()         # Not A but B
    d = (~has_a & ~has_b).sum()        # Neither

    table = np.array([[a, b], [c, d]])

    # Statistical test
    odds_ratio, p_fisher = fisher_exact(table, alternative='greater')
    chi2, p_chi2, _, _ = chi2_contingency(table)

    # Yule's Q
    Q = (a*d - b*c) / (a*d + b*c) if (a*d + b*c) > 0 else np.nan

    # Conditional probability
    P_b_given_a  = a / (a + b) if (a + b) > 0 else np.nan
    P_b_given_na = c / (c + d) if (c + d) > 0 else np.nan

    return {
        'A': f"{feature_a}={value_a}",
        'B': f"{feature_b}={value_b}",
        'n_AB': a, 'n_A_notB': b, 'n_notA_B': c, 'n_notA_notB': d,
        'P(B|A)': round(P_b_given_a, 4),
        'P(B|notA)': round(P_b_given_na, 4),
        "Yule's Q": round(Q, 4),
        'Fisher OR': round(odds_ratio, 4),
        'Fisher p': round(p_fisher, 6),
        'Significant': p_fisher < 0.05,
    }

# ---- Test key universals --------------------------------------- #
universals_to_test = [
    ('word_order', 'SOV', 'adposition', 'Postpositions'),
    ('word_order', 'SVO', 'adposition', 'Prepositions'),
    ('word_order', 'VSO', 'adposition', 'Prepositions'),
]

results = [test_implicational_universal(df_wals, *u) for u in universals_to_test]
df_results = pd.DataFrame(results)

print("=== Implicational Universal Tests ===\n")
for _, row in df_results.iterrows():
    sig = "✓ SIGNIFICANT" if row['Significant'] else "✗ not significant"
    print(f"Universal: If {row['A']} → {row['B']}")
    print(f"  Contingency: n(A∩B)={row['n_AB']}, n(A∩¬B)={row['n_A_notB']}, "
          f"n(¬A∩B)={row['n_notA_B']}, n(¬A∩¬B)={row['n_notA_notB']}")
    print(f"  P(B|A) = {row['P(B|A)']:.3f},  P(B|¬A) = {row['P(B|notA)']:.3f}")
    print(f"  Yule's Q = {row[\"Yule's Q\"]:.3f},  Fisher p = {row['Fisher p']:.4f}  {sig}\n")

# ---- Visualization: conditional probability comparison --------- #
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(df_results))
width = 0.35
bars1 = ax.bar(x - width/2, df_results["P(B|A)"],  width, label='P(B|A)',
               color='#3498db', edgecolor='black', linewidth=0.7)
bars2 = ax.bar(x + width/2, df_results["P(B|notA)"], width, label='P(B|¬A)',
               color='#e74c3c', edgecolor='black', linewidth=0.7)

labels = [f"{r['A']}\n→ {r['B'].split('=')[1]}" for _, r in df_results.iterrows()]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Conditional probability")
ax.set_title("Greenbergian Implicational Universals\n(Simulated WALS data)")
ax.legend(); ax.grid(axis='y', alpha=0.3)

for bar, row in zip(bars1, df_results.itertuples()):
    if row.Significant:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '**',
                ha='center', va='bottom', fontsize=14, color='green')
plt.tight_layout()
plt.savefig("implicational_universals.png", dpi=150)
plt.show()
```

### Step 3: Geographic Distribution Maps

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ------------------------------------------------------------------ #
# Map geographic distribution of word order types
# ------------------------------------------------------------------ #

# Try geopandas for proper world map; fallback to scatter plot
try:
    import geopandas as gpd
    from shapely.geometry import Point

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    fig, ax = plt.subplots(figsize=(15, 8))
    world.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5)

    order_colors = {
        'SOV': '#e74c3c', 'SVO': '#3498db', 'VSO': '#2ecc71',
        'VOS': '#9b59b6', 'OVS': '#f39c12', 'No dominant order': '#95a5a6',
    }

    for wo, color in order_colors.items():
        subset = df_wals[df_wals['word_order'] == wo]
        ax.scatter(subset['longitude'], subset['latitude'],
                   c=color, s=25, alpha=0.7, label=f"{wo} (n={len(subset)})",
                   edgecolors='black', linewidths=0.2, zorder=5)

    ax.set_xlim(-170, 175); ax.set_ylim(-55, 80)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Geographic Distribution of Word Order Types (Simulated WALS)")
    ax.legend(loc='lower left', fontsize=9, ncol=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig("wals_map_word_order.png", dpi=150)
    plt.show()

except Exception as e:
    print(f"geopandas unavailable ({e}); using scatter plot fallback")
    fig, ax = plt.subplots(figsize=(13, 7))

    order_colors = {
        'SOV': '#e74c3c', 'SVO': '#3498db', 'VSO': '#2ecc71',
        'VOS': '#9b59b6', 'OVS': '#f39c12', 'No dominant order': '#95a5a6',
    }
    for wo, color in order_colors.items():
        subset = df_wals[df_wals['word_order'] == wo]
        ax.scatter(subset['longitude'], subset['latitude'],
                   c=color, s=20, alpha=0.8,
                   label=f"{wo} (n={len(subset)})",
                   edgecolors='black', linewidths=0.2)

    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Word Order Geographic Distribution (Simulated WALS)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("wals_scatter_word_order.png", dpi=150)
    plt.show()

# ---- Cross-tabulation by family -------------------------------- #
cross_tab = pd.crosstab(df_wals['family'], df_wals['word_order'], normalize='index') * 100
print("\nWord order distribution by language family (%):")
print(cross_tab.round(1).to_string())
```

---

## Advanced Usage

### Phylogenetic Signal Test (Moran's I)

```python
import numpy as np
from scipy.spatial.distance import cdist

# ------------------------------------------------------------------ #
# Test whether typological feature clusters geographically
# (geographic signal = Moran's I)
# ------------------------------------------------------------------ #

def morans_i(values, coords, k_neighbors=10):
    """
    Compute Moran's I spatial autocorrelation.

    Parameters
    ----------
    values : binary array — feature presence (1/0)
    coords : (n,2) array — lat/lon
    k_neighbors : int — nearest neighbors for spatial weights
    """
    n = len(values)
    y = values.astype(float) - values.mean()

    # Build k-NN spatial weights matrix
    dist = cdist(coords, coords)
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist[i])[1:k_neighbors+1]
        W[i, neighbors] = 1

    # Normalize weights
    W = W / W.sum(axis=1, keepdims=True)

    # Moran's I
    num = n * np.sum(W * np.outer(y, y))
    denom = W.sum() * np.sum(y**2)
    return num / denom if denom > 0 else 0.0

coords = df_wals[['latitude', 'longitude']].values
is_sov = (df_wals['word_order'] == 'SOV').astype(int).values
is_tonal = (df_wals['tone'] != 'No tones').astype(int).values

I_sov   = morans_i(is_sov,   coords)
I_tonal = morans_i(is_tonal, coords)

print(f"Moran's I — SOV:    {I_sov:.4f}  (>0 = geographic clustering)")
print(f"Moran's I — Tonal:  {I_tonal:.4f}")
```

---

## Troubleshooting

### Loading real WALS CLDF data

```python
import pandas as pd

# After downloading WALS CLDF data:
values = pd.read_csv("wals/cldf/values.csv")
languages = pd.read_csv("wals/cldf/languages.csv")
parameters = pd.read_csv("wals/cldf/parameters.csv")

# Pivot: one row per language, columns = features
wals_wide = values.pivot_table(
    index='Language_ID', columns='Parameter_ID', values='Value', aggfunc='first'
)
# Join with language metadata
wals_full = languages.set_index('ID').join(wals_wide)
print(f"WALS full matrix: {wals_full.shape}")  # (~2600 languages × 192+ features)
```

### Sparse data / many NaN values

**Cause**: Most WALS features are only coded for a subset of languages.

**Fix**: Filter to languages with sufficient coverage:
```python
# Keep only languages with ≥20 features coded
coverage = (~wals_wide.isna()).sum(axis=1)
wals_filtered = wals_wide[coverage >= 20]
print(f"Languages with ≥20 features: {len(wals_filtered)}")
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| geopandas | 0.14 | Required for world map; fallback to scatter |
| pandas | 2.0, 2.1 | `pivot_table` for CLDF format |
| scipy | 1.11, 1.12 | `fisher_exact` stable |

---

## External Resources

### Official Documentation

- [WALS Online](https://wals.info)
- [WALS CLDF data (GitHub)](https://github.com/cldf-datasets/wals)

### Key Papers

- Greenberg, J.H. (1963). *Some universals of grammar with particular reference to the order of meaningful elements*. In Universals of Language, MIT Press.
- Dryer, M.S. & Haspelmath, M. (2013). *The World Atlas of Language Structures Online*. MPI EVA.

---

## Examples

### Example 1: Feature Co-occurrence Matrix

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

features = ['word_order', 'adposition', 'tone']
pivot = pd.get_dummies(df_wals[features])

corr = pivot.corr()
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=7)
ax.set_title("Feature Co-occurrence Correlation Matrix (WALS)")
plt.tight_layout(); plt.savefig("feature_correlation.png", dpi=150); plt.show()
```

### Example 2: Family-Level Feature Summary

```python
import pandas as pd
import numpy as np

family_summary = (df_wals.groupby('family')
    .agg(
        n_langs=('language_id','count'),
        pct_SOV=('word_order', lambda x: (x=='SOV').mean()*100),
        pct_SVO=('word_order', lambda x: (x=='SVO').mean()*100),
        pct_tonal=('tone', lambda x: (x!='No tones').mean()*100),
        pct_postpos=('adposition', lambda x: (x=='Postpositions').mean()*100),
    )
    .query("n_langs >= 10")
    .sort_values('pct_SOV', ascending=False)
)

print("Typological profile by language family (n≥10):")
print(family_summary.round(1).to_string())
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
