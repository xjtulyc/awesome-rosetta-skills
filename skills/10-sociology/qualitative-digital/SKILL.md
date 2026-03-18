---
name: qualitative-digital
description: >
  Use this Skill for digital qualitative methods: interview coding, thematic
  analysis, grounded theory, and NLP-assisted codebook development in Python.
tags:
  - sociology
  - qualitative
  - thematic-analysis
  - grounded-theory
  - text-analysis
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
    - scikit-learn>=1.3
    - sentence-transformers>=2.6
    - hdbscan>=0.8
    - matplotlib>=3.7
    - wordcloud>=1.9
last_updated: "2026-03-17"
status: "stable"
---

# Digital Qualitative Analysis

> **One-line summary**: Assist thematic analysis, grounded theory, and interview coding with NLP: sentence embeddings, topic modeling, co-occurrence networks, and inter-rater reliability.

---

## When to Use This Skill

- When organizing and coding interview transcripts or field notes
- When doing thematic analysis with NLP-assisted code suggestion
- When building grounded theory codebooks from text corpora
- When computing inter-rater reliability (Cohen's kappa, Krippendorff's alpha)
- When visualizing code co-occurrence networks and saturation curves
- When performing open, axial, and selective coding in Python

**Trigger keywords**: qualitative analysis, thematic analysis, grounded theory, interview coding, codebook, inter-rater reliability, Cohen kappa, Krippendorff alpha, NVivo alternative, ATLAS.ti alternative, discourse analysis, content analysis, sentence embeddings

---

## Background & Key Concepts

### Thematic Analysis (Braun & Clarke)

Six phases: (1) familiarization, (2) initial coding, (3) searching for themes, (4) reviewing themes, (5) defining themes, (6) writing up. NLP assists phases 2–4 by clustering similar text segments.

### Grounded Theory Coding

- **Open coding**: Break text into conceptual units, assign preliminary codes
- **Axial coding**: Find relationships between codes (cause-effect, context-condition)
- **Selective coding**: Identify core category integrating all codes

### Sentence Embeddings for Semantic Similarity

Sentence transformers map text to dense vectors. Cosine similarity identifies semantically similar passages even with different vocabulary — helpful for suggesting code assignments.

### Inter-Rater Reliability

Cohen's $\kappa$ for two raters:

$$
\kappa = \frac{P_o - P_e}{1 - P_e}
$$

where $P_o$ = observed agreement, $P_e$ = expected agreement by chance.
Krippendorff's $\alpha$ generalizes to multiple raters and ordinal/interval scales.

---

## Environment Setup

### Install Dependencies

```bash
pip install pandas>=2.0 numpy>=1.24 scikit-learn>=1.3 \
            sentence-transformers>=2.6 hdbscan>=0.8 \
            matplotlib>=3.7 wordcloud>=1.9
# Optional: umap-learn for dimensionality reduction
pip install umap-learn>=0.5
```

### Verify Installation

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "Hi there"])
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
print(f"Cosine similarity: {similarity:.4f}")
# Expected: ~0.8 (semantically similar)
```

---

## Core Workflow

### Step 1: Interview Corpus Processing and Initial Coding

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

# ------------------------------------------------------------------ #
# Simulate interview transcripts with open coding
# In practice: load from CSV/Word/PDF
# ------------------------------------------------------------------ #

transcripts = [
    {
        "id": "INT_01", "participant": "P01", "background": "working_class",
        "text": """
        I feel like the system is just not set up for people like us. You know, I work
        hard every day but the bills keep coming. My kids need things and I can't always
        provide. The landlord raised the rent again last month — third time in two years.
        I don't know what we're going to do. Sometimes I feel invisible, like nobody
        sees what families like mine are going through.
        """
    },
    {
        "id": "INT_02", "participant": "P02", "background": "middle_class",
        "text": """
        The cost of living is definitely a concern, but I think with proper planning
        you can manage. I own my home, so I'm somewhat insulated from rent increases.
        But I do worry about my kids' college costs and whether they'll have the same
        opportunities I had. The economy feels more uncertain than when I was young.
        I think community support networks are important.
        """
    },
    {
        "id": "INT_03", "participant": "P03", "background": "working_class",
        "text": """
        Healthcare is the biggest issue for me. I had to choose between buying medicine
        and buying food last winter. The ER is too expensive so you just try to manage.
        My workplace doesn't offer benefits. I feel like if you don't have money you're
        on your own. The government programs are too complicated to access — too much
        paperwork, too many hoops to jump through.
        """
    },
    {
        "id": "INT_04", "participant": "P04", "background": "upper_middle_class",
        "text": """
        I think the biggest challenge in society right now is the division — people
        are not talking to each other across class lines. I volunteer at the food bank
        and I see what people are going through. I feel fortunate but also guilty
        sometimes. I believe in collective responsibility. Strong communities make
        stronger individuals.
        """
    },
    {
        "id": "INT_05", "participant": "P05", "background": "working_class",
        "text": """
        Work is exhausting. Two jobs, still not enough. The gig economy is brutal —
        no stability, no benefits, no predictability. I feel like I'm running on a
        treadmill. Childcare costs eat everything. Nobody talks about how hard it is
        to be a single parent and work. The system wasn't built for us.
        """
    },
]

df = pd.DataFrame(transcripts)

# ---- Segment text into coded units (sentence-level) --------------- #
def segment_text(text, participant_id):
    """Split text into coded units (non-trivial sentences)."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    units = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if len(sent.split()) >= 5:  # Min 5 words
            units.append({
                'participant': participant_id,
                'segment_id': f"{participant_id}_S{i:02d}",
                'text': sent,
            })
    return units

coded_units = []
for _, row in df.iterrows():
    coded_units.extend(segment_text(row['text'], row['participant']))

df_units = pd.DataFrame(coded_units)
print(f"Total coded units: {len(df_units)}")
print(df_units[['segment_id','text']].head(8).to_string(index=False))

# ---- Manual codebook (open codes) -------------------------------- #
# In practice: researcher reads and assigns codes
# Here: keyword-based heuristic for demonstration
codebook = {
    'Economic stress':    ['rent', 'bills', 'money', 'cost', 'afford', 'income', 'pay'],
    'Systemic critique':  ['system', 'government', 'access', 'bureaucracy', 'program', 'built'],
    'Healthcare':         ['healthcare', 'medicine', 'hospital', 'ER', 'benefits', 'health'],
    'Work precarity':     ['work', 'job', 'gig', 'stability', 'hours', 'treadmill', 'jobs'],
    'Social visibility':  ['invisible', 'see', 'nobody', 'community', 'collective', 'division'],
    'Family/children':    ['kids', 'children', 'family', 'parent', 'childcare', 'child'],
    'Emotional state':    ['feel', 'worry', 'exhausting', 'guilty', 'fortunate', 'lucky'],
}

def assign_codes(text, codebook):
    """Assign codes based on keyword matching (approximation of manual coding)."""
    text_lower = text.lower()
    assigned = []
    for code, keywords in codebook.items():
        if any(kw in text_lower for kw in keywords):
            assigned.append(code)
    return assigned if assigned else ['Uncoded']

df_units['codes'] = df_units['text'].apply(lambda t: assign_codes(t, codebook))
df_units['n_codes'] = df_units['codes'].apply(len)

# Code frequency
code_counts = Counter(code for codes in df_units['codes'] for code in codes)
code_df = pd.DataFrame(code_counts.most_common(), columns=['Code','Count'])
print("\nCode frequencies:")
print(code_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(code_df['Code'], code_df['Count'], color='steelblue', edgecolor='black', linewidth=0.6)
ax.set_xlabel("Frequency"); ax.set_title("Code Frequency — Thematic Analysis")
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("code_frequency.png", dpi=150)
plt.show()
```

### Step 2: Semantic Clustering with Sentence Embeddings

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ------------------------------------------------------------------ #
# Use sentence embeddings to cluster coded units semantically
# ------------------------------------------------------------------ #

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_units['text'].tolist(), show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
except ImportError:
    print("sentence-transformers not available; using TF-IDF fallback")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    embeddings = vectorizer.fit_transform(df_units['text']).toarray()

# Normalize embeddings for cosine similarity
embeddings_norm = normalize(embeddings)

# ---- UMAP or PCA for 2D visualization -------------------------- #
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.3)
    embedding_2d = reducer.fit_transform(embeddings_norm)
except ImportError:
    # Fallback: PCA
    pca = PCA(n_components=2, random_state=42)
    embedding_2d = pca.fit_transform(embeddings_norm)
    print("Using PCA for 2D projection (install umap-learn for better results)")

# ---- HDBSCAN clustering ---------------------------------------- #
try:
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    labels = clusterer.fit_predict(embedding_2d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} semantic clusters")
except ImportError:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_norm)
    n_clusters = 4

df_units['cluster'] = labels

# Visualize
fig, ax = plt.subplots(figsize=(10, 7))
scatter_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters + 1))
for cluster_id in sorted(set(labels)):
    mask = labels == cluster_id
    label = f"Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    color = 'lightgray' if cluster_id == -1 else scatter_colors[cluster_id % len(scatter_colors)]
    ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
               c=[color], label=label, s=60, edgecolors='black', linewidths=0.3, alpha=0.8)

# Annotate with participant
for i, (x, y) in enumerate(embedding_2d):
    ax.annotate(df_units.iloc[i]['participant'], (x, y),
                fontsize=6, alpha=0.6, xytext=(2, 2), textcoords='offset points')

ax.set_title("Semantic Clustering of Coded Units\n(UMAP/PCA + HDBSCAN)")
ax.legend(loc='upper left', fontsize=8); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("semantic_clusters.png", dpi=150)
plt.show()

# Retrieve representative quotes per cluster
print("\nRepresentative quotes per cluster:")
for cid in sorted(set(labels)):
    if cid == -1:
        continue
    cluster_texts = df_units[df_units['cluster'] == cid]['text'].tolist()
    # Most central quote (closest to cluster centroid)
    cluster_emb = embeddings_norm[labels == cid]
    centroid = cluster_emb.mean(axis=0)
    dists = np.linalg.norm(cluster_emb - centroid, axis=1)
    rep_idx = dists.argmin()
    print(f"\nCluster {cid}: {cluster_texts[rep_idx][:120]}...")
```

### Step 3: Inter-Rater Reliability and Codebook Validation

```python
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------ #
# Compute inter-rater reliability for two coders
# ------------------------------------------------------------------ #

# Simulated coding by two researchers
np.random.seed(42)
n_segments = 40
codes_list = ['Economic stress', 'Systemic critique', 'Healthcare',
              'Work precarity', 'Social visibility', 'Family/children']

# Rater 1 (gold standard)
rater1 = np.random.choice(codes_list, n_segments)

# Rater 2 (agree ~75% of time, disagree on 25%)
rater2 = rater1.copy()
disagree_idx = np.random.choice(n_segments, int(n_segments * 0.25), replace=False)
rater2[disagree_idx] = np.random.choice(codes_list, len(disagree_idx))

# Cohen's kappa
kappa = cohen_kappa_score(rater1, rater2)
pct_agreement = np.mean(rater1 == rater2) * 100
print(f"Percentage agreement: {pct_agreement:.1f}%")
print(f"Cohen's κ: {kappa:.4f}")
if kappa > 0.80:
    strength = "Almost perfect"
elif kappa > 0.60:
    strength = "Substantial"
elif kappa > 0.40:
    strength = "Moderate"
else:
    strength = "Fair or less"
print(f"Interpretation: {strength} agreement")

# Confusion matrix
cm = confusion_matrix(rater1, rater2, labels=codes_list)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns_available = False
try:
    import seaborn as sns
    sns_available = True
except ImportError:
    pass

if sns_available:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=codes_list, yticklabels=codes_list, ax=axes[0])
    axes[0].set_xlabel("Rater 2"); axes[0].set_ylabel("Rater 1")
else:
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks(range(len(codes_list))); axes[0].set_xticklabels(codes_list, rotation=45, ha='right')
    axes[0].set_yticks(range(len(codes_list))); axes[0].set_yticklabels(codes_list)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i,j]), ha='center', va='center')
axes[0].set_title(f"Confusion Matrix\n(κ={kappa:.3f})")

# Code-level agreement
code_agreement = {}
for i, code in enumerate(codes_list):
    mask = (rater1 == code) | (rater2 == code)
    if mask.sum() > 0:
        agree = ((rater1 == code) & (rater2 == code)).sum()
        total = mask.sum()
        code_agreement[code] = agree / total * 100

codes_sorted = sorted(code_agreement, key=code_agreement.get)
axes[1].barh(codes_sorted, [code_agreement[c] for c in codes_sorted],
             color='steelblue', edgecolor='black', linewidth=0.6)
axes[1].axvline(75, color='red', linestyle='--', linewidth=1.5, label='75% threshold')
axes[1].set_xlabel("Code-level agreement (%)"); axes[1].set_title("Agreement by Code")
axes[1].legend(); axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("irr_analysis.png", dpi=150)
plt.show()

# Codes below threshold need discussion
problematic = [c for c, a in code_agreement.items() if a < 75]
if problematic:
    print(f"\nCodes requiring codebook clarification: {problematic}")
```

---

## Advanced Usage

### Saturation Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Theoretical saturation: track new codes discovered per interview
# ------------------------------------------------------------------ #

all_codes_by_interview = [
    ['Economic stress', 'Family/children'],
    ['Economic stress', 'Systemic critique', 'Healthcare'],
    ['Work precarity', 'Systemic critique', 'Emotional state'],
    ['Social visibility', 'Emotional state'],
    ['Economic stress', 'Work precarity', 'Family/children'],
    ['Healthcare', 'Systemic critique'],
    ['Social visibility'],
    ['Economic stress'],
]

cumulative_codes = set()
new_codes_per_interview = []
for codes in all_codes_by_interview:
    new = [c for c in codes if c not in cumulative_codes]
    cumulative_codes.update(codes)
    new_codes_per_interview.append(len(new))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, len(new_codes_per_interview)+1), new_codes_per_interview,
            color='steelblue', edgecolor='black', linewidth=0.7)
axes[0].set_xlabel("Interview number"); axes[0].set_ylabel("New codes discovered")
axes[0].set_title("Code Discovery Rate (Saturation Curve)")
axes[0].grid(axis='y', alpha=0.3)

cumulative_unique = np.cumsum([len(c) for c in [set()] + [set(cs) for cs in all_codes_by_interview[:-1]]])
cum_actual = []
seen = set()
for codes in all_codes_by_interview:
    seen.update(codes)
    cum_actual.append(len(seen))

axes[1].plot(range(1, len(cum_actual)+1), cum_actual, 'bo-', linewidth=2, markersize=8)
axes[1].set_xlabel("Interview number"); axes[1].set_ylabel("Unique codes")
axes[1].set_title("Cumulative Code Growth (Saturation)"); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig("saturation_curve.png", dpi=150); plt.show()
print(f"Theoretical saturation reached ~interview {next((i+1 for i,v in enumerate(new_codes_per_interview[3:],3) if v==0), 'not reached')}")
```

---

## Troubleshooting

### Error: `OSError: No model found` for SentenceTransformer

**Fix**: Download model explicitly:
```bash
python -m sentence_transformers download all-MiniLM-L6-v2
```
Or use offline-compatible TF-IDF as fallback.

### HDBSCAN produces all-noise labels (-1)

**Fix**: Reduce `min_cluster_size`:
```python
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
```

### Cohen's kappa is very low (<0.4)

This indicates poor inter-rater agreement. Steps:
1. Review codebook definitions — add inclusion/exclusion examples
2. Hold calibration session with shared coding examples
3. Re-code independently, then reconcile disagreements

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| sentence-transformers | 2.6, 2.7 | Model downloads require internet |
| hdbscan | 0.8.33 | Requires scikit-learn |
| scikit-learn | 1.3, 1.4 | No known issues |

---

## External Resources

### Official Documentation

- [sentence-transformers docs](https://www.sbert.net)
- [HDBSCAN documentation](https://hdbscan.readthedocs.io)

### Key Papers

- Braun, V. & Clarke, V. (2006). *Using thematic analysis in psychology*. Qualitative Research in Psychology.
- Glaser, B.G. & Strauss, A.L. (1967). *The Discovery of Grounded Theory*. Aldine.

---

## Examples

### Example 1: Co-occurrence Network of Codes

```python
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

# Build code co-occurrence network
G = nx.Graph()
for _, row in df_units.iterrows():
    codes = [c for c in row['codes'] if c != 'Uncoded']
    for c1, c2 in combinations(codes, 2):
        if G.has_edge(c1, c2):
            G[c1][c2]['weight'] += 1
        else:
            G.add_edge(c1, c2, weight=1)

if G.number_of_nodes() > 0:
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx(G, pos, ax=ax, node_size=1500, node_color='lightblue',
                     width=[w*2 for w in weights], edge_color='gray',
                     font_size=8, font_weight='bold')
    ax.set_title("Code Co-occurrence Network"); ax.axis('off')
    plt.tight_layout(); plt.savefig("code_network.png", dpi=150); plt.show()
```

### Example 2: Word Cloud per Theme

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

themes = {
    'Economic stress': 'rent bills money cost afford income pay landlord work struggling',
    'Systemic critique': 'system government program bureaucracy access paperwork hoops built',
    'Healthcare': 'healthcare medicine hospital benefits doctor insurance sick treatment',
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (theme, text) in zip(axes, themes.items()):
    wc = WordCloud(width=400, height=200, background_color='white',
                   colormap='Blues', max_words=30).generate(text)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(theme, fontsize=10); ax.axis('off')
plt.tight_layout(); plt.savefig("theme_wordclouds.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
