---
name: computational-pragmatics
description: >
  Use this Skill for computational pragmatics: speech act classification,
  implicature detection, discourse coherence, and politeness analysis with NLP.
tags:
  - linguistics
  - pragmatics
  - nlp
  - speech-acts
  - discourse
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
    - transformers>=4.38
    - torch>=2.1
    - scikit-learn>=1.3
    - pandas>=2.0
    - numpy>=1.24
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# Computational Pragmatics

> **One-line summary**: Analyze discourse beyond sentence meaning: classify speech acts (illocutionary force), detect implicature patterns, model politeness strategies, and analyze dialog coherence with transformer NLP.

---

## When to Use This Skill

- When classifying speech acts (assertion, question, directive, commissive)
- When detecting conversational implicature and indirect speech
- When analyzing politeness and face-threatening acts (Brown & Levinson)
- When modeling discourse coherence (RST, adjacency pairs)
- When building dialog state tracking systems
- When analyzing conversation structure in social media or interviews

**Trigger keywords**: pragmatics, speech acts, illocutionary force, implicature, politeness, face-threatening act, discourse coherence, dialog analysis, RST, adjacency pairs, conversation analysis, dialog act, indirect speech, hedging

---

## Background & Key Concepts

### Speech Act Theory (Austin/Searle)

Every utterance has:
- **Locutionary act**: literal meaning ("Can you pass the salt?")
- **Illocutionary act**: communicative intention (request)
- **Perlocutionary act**: effect on hearer (hearer passes salt)

Standard taxonomy: Assertives, Directives, Commissives, Expressives, Declarations.

### Gricean Maxims and Implicature

Cooperative principle with four maxims:
- **Quantity**: be informative (not over/under-specify)
- **Quality**: be truthful
- **Relation**: be relevant
- **Manner**: be clear and brief

Violation → conversational implicature (speaker communicates beyond the literal meaning).

### Politeness Theory (Brown & Levinson)

Face = public self-image:
- **Positive face**: desire for approval
- **Negative face**: desire for autonomy

Face-Threatening Acts (FTAs) mitigation strategies: indirect requests, hedges, apologies, softeners.

---

## Environment Setup

### Install Dependencies

```bash
pip install transformers>=4.38 torch>=2.1 scikit-learn>=1.3 \
            pandas>=2.0 numpy>=1.24 matplotlib>=3.7
```

### Verify Installation

```python
from transformers import pipeline

# Test zero-shot classification
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
result = classifier(
    "Could you please close the door?",
    candidate_labels=["directive", "assertion", "question", "expressive"]
)
print(f"Top prediction: {result['labels'][0]} (score={result['scores'][0]:.3f})")
```

---

## Core Workflow

### Step 1: Speech Act Classification

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------ #
# Classify dialog acts using zero-shot or fine-tuned transformer
# ------------------------------------------------------------------ #

# Sample dialog act dataset (5 categories)
da_examples = [
    # (utterance, label)
    ("The weather is nice today.", "assertion"),
    ("It's raining outside.", "assertion"),
    ("The meeting is at 3pm.", "assertion"),
    ("Can you help me with this?", "directive"),
    ("Please send me the report by Friday.", "directive"),
    ("Close the window.", "directive"),
    ("Could you possibly pass me the salt?", "directive"),  # Indirect directive
    ("What time does the train leave?", "question"),
    ("Where did you put the keys?", "question"),
    ("Did you finish the assignment?", "question"),
    ("I'll be there at 8.", "commissive"),
    ("I promise to call you back.", "commissive"),
    ("We will deliver by next week.", "commissive"),
    ("Thank you so much for your help!", "expressive"),
    ("I'm sorry for the inconvenience.", "expressive"),
    ("Congratulations on your promotion!", "expressive"),
    ("That's fantastic!", "expressive"),
]

df_da = pd.DataFrame(da_examples, columns=['utterance', 'label'])
labels_all = ['assertion', 'directive', 'question', 'commissive', 'expressive']

print(f"Dialog act examples: {len(df_da)} utterances, {df_da['label'].nunique()} categories")
print(df_da['label'].value_counts())

try:
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli",
                          device=-1)  # CPU

    # Classify each utterance
    predictions = []
    for _, row in df_da.iterrows():
        result = classifier(row['utterance'], candidate_labels=labels_all)
        pred = result['labels'][0]
        predictions.append(pred)
        print(f"  '{row['utterance'][:50]}': pred={pred}, true={row['label']}")

    df_da['predicted'] = predictions
    print("\nClassification Report:")
    print(classification_report(df_da['label'], df_da['predicted']))

except Exception as e:
    print(f"Transformer not available ({e}); using rule-based classifier")

    def rule_based_da(text):
        """Simple rule-based dialog act classifier."""
        text_lower = text.lower().strip()
        if text_lower.endswith('?') or text_lower.startswith(('what','where','when','who','how','did','do','is','are','can','could','would')):
            return 'question'
        elif any(w in text_lower for w in ["thank","sorry","congratulations","wonderful","great","fantastic"]):
            return 'expressive'
        elif any(w in text_lower for w in ["please","could you","would you","close","send","help","pass"]):
            return 'directive'
        elif any(w in text_lower for w in ["i'll","i will","i promise","i commit","we will","we'll"]):
            return 'commissive'
        else:
            return 'assertion'

    df_da['predicted'] = df_da['utterance'].apply(rule_based_da)
    print("Rule-based classification:")
    print(classification_report(df_da['label'], df_da['predicted']))

# ---- Visualize confusion matrix -------------------------------- #
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(df_da['label'], df_da['predicted'], labels=labels_all)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(labels_all))); ax.set_xticklabels(labels_all, rotation=30, ha='right')
ax.set_yticks(range(len(labels_all))); ax.set_yticklabels(labels_all)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Dialog Act Confusion Matrix")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig("dialog_act_cm.png", dpi=150)
plt.show()
```

### Step 2: Politeness and Hedging Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# ------------------------------------------------------------------ #
# Compute politeness features from text:
# - Hedges, boosters, face-threatening act mitigation
# - Based on Brown & Levinson (1987) and Danescu-Niculescu-Mizil et al. (2013)
# ------------------------------------------------------------------ #

# Define politeness markers
HEDGES    = r'\b(maybe|perhaps|possibly|I think|I believe|I suppose|sort of|kind of|fairly|quite|a bit|somewhat|it seems|apparently|probably|might|could|would)\b'
BOOSTERS  = r'\b(clearly|obviously|definitely|certainly|absolutely|exactly|of course|really|truly|in fact|undoubtedly)\b'
PLEASE    = r'\bplease\b'
APOLOGIZE = r'\b(sorry|apologize|excuse me|pardon|forgive)\b'
QUESTIONS = r'\?'
NEGATIONS = r'\b(not|no|never|nothing|nobody|nowhere|neither)\b'
DIRECT_CMD = r'^(close|open|send|bring|give|stop|start|make|do|go|come|take)\b'

def politeness_features(text):
    """Extract politeness features from an utterance."""
    text_l = text.lower()
    return {
        'n_hedges':    len(re.findall(HEDGES, text_l, re.IGNORECASE)),
        'n_boosters':  len(re.findall(BOOSTERS, text_l, re.IGNORECASE)),
        'has_please':  bool(re.search(PLEASE, text_l, re.IGNORECASE)),
        'has_apology': bool(re.search(APOLOGIZE, text_l, re.IGNORECASE)),
        'is_question': bool(re.search(QUESTIONS, text)),
        'is_direct_cmd': bool(re.search(DIRECT_CMD, text_l)),
        'length_words': len(text.split()),
    }

# Sample requests varying in politeness
requests = [
    ("Close the door.", "bare imperative"),
    ("Close the door, please.", "please-marked"),
    ("Could you close the door?", "modal question"),
    ("Would you mind closing the door?", "would you mind"),
    ("I was wondering if you could possibly close the door.", "hedged request"),
    ("I'm sorry to bother you, but would it be too much to ask you to close the door?", "highly mitigated"),
    ("The door needs to be closed.", "impersonalized"),
    ("It's a bit cold in here.", "off-record hint"),
]

df_req = pd.DataFrame(requests, columns=['utterance', 'politeness_strategy'])
feat_cols = list(politeness_features(requests[0][0]).keys())
features = df_req['utterance'].apply(politeness_features).apply(pd.Series)
df_req = pd.concat([df_req, features], axis=1)

# Compute politeness score: hedge+please+question−direct_cmd
df_req['politeness_score'] = (
    df_req['n_hedges'] * 2 +
    df_req['has_please'].astype(int) * 1.5 +
    df_req['has_apology'].astype(int) * 2 +
    df_req['is_question'].astype(int) * 1 -
    df_req['is_direct_cmd'].astype(int) * 2
)

print("Politeness Analysis of Request Formulations:")
print(df_req[['utterance','politeness_strategy','n_hedges','has_please','is_question',
              'is_direct_cmd','politeness_score']].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Politeness score bar chart
y_pos = range(len(df_req))
colors = ['#2ecc71' if s >= 2 else '#e67e22' if s >= 0 else '#e74c3c'
          for s in df_req['politeness_score']]
axes[0].barh(y_pos, df_req['politeness_score'], color=colors, edgecolor='black', linewidth=0.7)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels([f"{r['politeness_strategy']}" for _, r in df_req.iterrows()], fontsize=8)
axes[0].set_xlabel("Politeness Score"); axes[0].set_title("Request Politeness Ranking")
axes[0].axvline(0, color='gray', linewidth=1); axes[0].grid(axis='x', alpha=0.3)

# Feature heatmap
feat_matrix = df_req[['n_hedges','has_please','has_apology','is_question','is_direct_cmd']].values.astype(float)
im = axes[1].imshow(feat_matrix.T, cmap='RdYlGn', aspect='auto')
plt.colorbar(im, ax=axes[1])
axes[1].set_xticks(range(len(df_req)))
axes[1].set_xticklabels([f"R{i+1}" for i in range(len(df_req))], fontsize=8)
axes[1].set_yticks(range(5))
axes[1].set_yticklabels(['Hedges','Please','Apology','Question','Direct cmd'], fontsize=9)
axes[1].set_title("Politeness Feature Matrix")

plt.tight_layout()
plt.savefig("politeness_analysis.png", dpi=150)
plt.show()
```

### Step 3: Discourse Coherence and Adjacency Pairs

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

# ------------------------------------------------------------------ #
# Analyze dialog structure via adjacency pairs and turn-taking patterns
# ------------------------------------------------------------------ #

# Sample dialog (each entry: speaker, utterance, dialog act)
dialog = [
    ("A", "Hi, how are you doing?",                    "greeting"),
    ("B", "I'm fine, thanks. And you?",                 "greeting-response"),
    ("A", "Good. Did you finish the report?",           "question"),
    ("B", "Almost. I just need a bit more time.",       "assertion+commissive"),
    ("A", "When can I expect it?",                      "question"),
    ("B", "I'll have it ready by tomorrow morning.",    "commissive"),
    ("A", "Could you send it before 9am?",              "directive"),
    ("B", "Of course, no problem.",                     "commissive"),
    ("A", "Great. Also, could you add the Q4 data?",   "directive"),
    ("B", "I'll check if I have access to that.",       "commissive"),
    ("A", "If not, let me know.",                       "directive"),
    ("B", "Will do. Anything else?",                    "question"),
    ("A", "That's all for now. Thank you!",             "expressive"),
    ("B", "You're welcome. Talk later.",                "expressive"),
]

df_dialog = pd.DataFrame(dialog, columns=['speaker', 'utterance', 'dialog_act'])
df_dialog['turn'] = range(1, len(df_dialog) + 1)

# ---- Adjacency pair analysis ----------------------------------- #
# An adjacency pair = consecutive (first pair part, second pair part)
pair_types = {
    ('greeting', 'greeting-response'): 'Greeting pair',
    ('question', 'assertion'):         'Q-A pair',
    ('question', 'assertion+commissive'): 'Q-A pair',
    ('question', 'commissive'):        'Q-Commitment pair',
    ('directive', 'commissive'):       'Request-Accept pair',
    ('expressive', 'expressive'):      'Leave-taking pair',
}

pairs_found = []
for i in range(len(df_dialog) - 1):
    da1 = df_dialog.iloc[i]['dialog_act']
    da2 = df_dialog.iloc[i+1]['dialog_act']
    pair_key = (da1, da2)
    pair_name = pair_types.get(pair_key, f"{da1} → {da2}")
    pairs_found.append({
        'turn': df_dialog.iloc[i]['turn'],
        'speaker_1': df_dialog.iloc[i]['speaker'],
        'speaker_2': df_dialog.iloc[i+1]['speaker'],
        'first_part': da1,
        'second_part': da2,
        'pair_type': pair_name,
    })

df_pairs = pd.DataFrame(pairs_found)
print("Adjacency Pairs Found:")
print(df_pairs[['turn','speaker_1','first_part','speaker_2','second_part','pair_type']].to_string(index=False))

# ---- Dialog act sequence plot ---------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Turn-by-turn dialog act
da_colors = {
    'greeting': '#3498db', 'greeting-response': '#3498db',
    'question': '#e67e22', 'assertion': '#2ecc71',
    'commissive': '#9b59b6', 'directive': '#e74c3c',
    'expressive': '#1abc9c', 'assertion+commissive': '#8e44ad',
}
speaker_pos = {'A': 0.7, 'B': 0.3}

for _, row in df_dialog.iterrows():
    color = da_colors.get(row['dialog_act'], 'gray')
    y_pos = speaker_pos.get(row['speaker'], 0.5)
    axes[0].scatter(row['turn'], y_pos, c=color, s=200, zorder=5,
                    edgecolors='black', linewidths=0.8)
    axes[0].annotate(row['dialog_act'].split('+')[0][:4],
                      (row['turn'], y_pos),
                      fontsize=6, ha='center', va='bottom',
                      xytext=(0, 8), textcoords='offset points')

axes[0].set_yticks([0.3, 0.7]); axes[0].set_yticklabels(['Speaker B', 'Speaker A'])
axes[0].set_xlabel("Turn number"); axes[0].set_title("Dialog Act Sequence")
axes[0].grid(axis='x', alpha=0.3); axes[0].set_xlim(0, len(df_dialog)+1)

# Transition network
G = nx.DiGraph()
for _, row in df_pairs.iterrows():
    src = row['first_part']
    tgt = row['second_part']
    if G.has_edge(src, tgt):
        G[src][tgt]['weight'] += 1
    else:
        G.add_edge(src, tgt, weight=1)

pos = nx.spring_layout(G, seed=42)
weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
nx.draw_networkx(G, pos, ax=axes[1], node_size=1500, node_color='lightblue',
                 arrows=True, arrowsize=20, width=weights,
                 font_size=7, font_weight='bold', edge_color='gray')
axes[1].set_title("Dialog Act Transition Network"); axes[1].axis('off')

plt.tight_layout()
plt.savefig("discourse_analysis.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Implicature Detection

```python
import numpy as np

# ------------------------------------------------------------------ #
# Detect potential implicatures by checking Gricean maxim violations
# ------------------------------------------------------------------ #

def detect_implicature_signals(utterance):
    """
    Rule-based heuristics for identifying likely implicature contexts.
    Returns signals suggesting non-literal interpretation.
    """
    import re
    text = utterance.lower()
    signals = []

    # Quantity maxim: under-informativeness
    if re.search(r'\b(some|a few|a couple)\b', text):
        signals.append(('quantity', "Under-informative: 'some' implicates 'not all'"))

    # Manner: indirect phrasing
    if re.search(r'\b(it (would be|might be|could be))', text):
        signals.append(('manner', "Periphrastic phrasing signals indirectness"))

    # Relevance: apparent non-sequitur (harder to detect; needs context)
    # Hedges that weaken commitment
    if re.search(r'\b(or something|kind of|sort of|I guess)\b', text):
        signals.append(('quality', "Hedging suggests reduced epistemic commitment"))

    # Scalar: scalar implicature
    if re.search(r'\b(some|sometimes|occasionally|possible|might)\b', text):
        signals.append(('scalar', "Scalar item — may implicate negation of stronger term"))

    return signals

test_utterances = [
    "Some students passed the exam.",
    "The food was okay.",
    "John is or might be coming.",
    "I think it might possibly work.",
    "It would be nice if someone cleaned this up.",
]

for utt in test_utterances:
    signals = detect_implicature_signals(utt)
    print(f"\nUtterance: '{utt}'")
    if signals:
        for sig_type, sig_desc in signals:
            print(f"  [{sig_type.upper()}] {sig_desc}")
    else:
        print("  No strong implicature signals detected")
```

---

## Troubleshooting

### Transformer model download too slow

```python
# Use a smaller, faster model
classifier = pipeline("zero-shot-classification",
                      model="cross-encoder/nli-MiniLM2-L6-H768")
```

### Zero-shot accuracy is poor for domain-specific dialog acts

**Fix**: Fine-tune on labeled data:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
# Fine-tune on your labeled dialog act dataset
```

### Rule-based politeness features miss context

**Fix**: Add context window (previous utterance) and use embeddings for similarity:
```python
# Check if current utterance responds to a prior FTA
# E.g., "Sorry to bother you" responds to an implicit FTA of making a request
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| transformers | 4.38, 4.40 | `facebook/bart-large-mnli` for zero-shot |
| torch | 2.1, 2.2 | CPU inference works; GPU for large corpora |
| scikit-learn | 1.3, 1.4 | `classification_report` stable |

---

## External Resources

### Official Documentation

- [Hugging Face zero-shot classification](https://huggingface.co/tasks/zero-shot-classification)
- [SWDA Dialog Act Corpus](https://web.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz)

### Key Papers

- Austin, J.L. (1962). *How to Do Things with Words*. Oxford University Press.
- Searle, J.R. (1969). *Speech Acts*. Cambridge University Press.
- Brown, P. & Levinson, S.C. (1987). *Politeness*. Cambridge University Press.
- Grice, H.P. (1975). *Logic and conversation*. In Cole & Morgan (Eds.), Syntax and Semantics.

---

## Examples

### Example 1: Reddit Comment Politeness Scoring

```python
import pandas as pd

reddit_comments = [
    "Could someone please explain how this works?",
    "This is wrong. Learn to read.",
    "I might be mistaken, but perhaps the error is on line 23?",
    "RTFM lol.",
    "Would you mind sharing the error message? That would help a lot!",
]

for comment in reddit_comments:
    feats = politeness_features(comment)
    score = (feats['n_hedges']*2 + feats['has_please']*1.5 +
             feats['is_question']*1 - feats['is_direct_cmd']*2)
    print(f"Score={score:+.1f}  '{comment[:60]}'")
```

### Example 2: Topic-Control Analysis in Discourse

```python
import pandas as pd
import numpy as np

# Simplified topic shift detection using sentence embeddings similarity
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer('all-MiniLM-L6-v2')
    utterances = [u[1] for u in dialog]
    embeddings = model.encode(utterances)

    # Cosine similarity between consecutive utterances
    similarities = []
    for i in range(len(embeddings)-1):
        e1, e2 = embeddings[i], embeddings[i+1]
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        similarities.append(sim)

    # Low similarity = topic shift
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(range(1, len(similarities)+1), similarities, 'b-o', markersize=6)
    ax.axhline(0.7, color='red', linestyle='--', label='Topic shift threshold')
    ax.set_xlabel("Turn transition"); ax.set_ylabel("Cosine similarity")
    ax.set_title("Discourse Coherence — Consecutive Utterance Similarity")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("coherence.png", dpi=150); plt.show()
except ImportError:
    print("sentence-transformers not installed; skipping embedding-based coherence analysis")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
