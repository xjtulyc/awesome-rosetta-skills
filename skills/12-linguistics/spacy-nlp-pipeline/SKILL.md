---
name: spacy-nlp-pipeline
description: >
  Use this Skill to build NLP pipelines with spaCy: tokenization, POS tagging,
  dependency parsing, NER, custom rule components, and multi-language model
  selection.
tags:
  - linguistics
  - spaCy
  - NLP
  - named-entity-recognition
  - dependency-parsing
  - pipeline
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
    - spacy>=3.7
    - pandas>=1.5
    - matplotlib>=3.6
  system:
    - en_core_web_sm
last_updated: "2026-03-18"
status: stable
---

# spaCy NLP Pipeline

> **TL;DR** — Build production NLP pipelines with spaCy 3.7+: load multilingual
> models, add custom EntityRuler for domain-specific NER, extract dependency
> triples (subject-verb-object), visualize with displacy, and batch-process
> large document collections efficiently with `nlp.pipe()`.

---

## When to Use

Use this Skill when you need to:

- Tokenize, lemmatize, and POS-tag text with a validated statistical model
- Recognize named entities (persons, organizations, locations, custom labels)
- Add domain-specific entity rules without retraining the model
- Extract syntactic dependency trees and subject-verb-object triples
- Process a collection of documents efficiently (batch NLP inference)
- Support multiple languages in the same pipeline
- Visualize NER annotations and dependency parses in HTML or a notebook

---

## Background

### spaCy Pipeline Components

A spaCy `Language` object contains a sequential pipeline of components:

| Component | Purpose |
|---|---|
| `tokenizer` | Splits text into Token objects |
| `tagger` | Assigns part-of-speech tags (POS) |
| `morphologizer` | Assigns morphological features |
| `parser` | Builds dependency parse tree |
| `senter` | Sentence boundary detection |
| `ner` | Named entity recognition |
| `EntityRuler` | Rule-based NER (before or after statistical ner) |

### Language Model Selection

| Model | Language | Size | Use case |
|---|---|---|---|
| `en_core_web_sm` | English | 12 MB | Fast prototyping |
| `en_core_web_lg` | English | 685 MB | Production accuracy |
| `xx_ent_wiki_sm` | Multi | 15 MB | Cross-lingual NER |
| `zh_core_web_sm` | Chinese | 46 MB | Mandarin text |
| `de_core_news_sm` | German | 14 MB | German text |

### Token Attributes

| Attribute | Type | Description |
|---|---|---|
| `token.text` | str | Original word form |
| `token.lemma_` | str | Base lemma |
| `token.pos_` | str | Coarse POS (NOUN, VERB…) |
| `token.tag_` | str | Fine-grained POS tag |
| `token.dep_` | str | Dependency relation |
| `token.head` | Token | Syntactic head token |
| `token.ent_type_` | str | Entity type label |
| `token.is_stop` | bool | Is stop word |

---

## Environment Setup

```bash
# Create environment
conda create -n nlp_env python=3.11 -y
conda activate nlp_env

# Install spaCy and dependencies
pip install spacy>=3.7 pandas>=1.5 matplotlib>=3.6

# Download English model (small for development)
python -m spacy download en_core_web_sm

# For production accuracy
python -m spacy download en_core_web_lg

# For multilingual NER
python -m spacy download xx_ent_wiki_sm

# Chinese model
python -m spacy download zh_core_web_sm

# Verify
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy OK')"
```

---

## Core Workflow

### Step 1 — Load and Configure Pipeline

```python
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Iterator
from collections import Counter
import re


def load_pipeline(
    model_name: str = "en_core_web_sm",
    disable_components: Optional[List[str]] = None,
    add_sentencizer: bool = False,
) -> spacy.language.Language:
    """
    Load a spaCy model and optionally disable unused components for speed.

    Args:
        model_name:          spaCy model name (e.g., 'en_core_web_sm').
        disable_components:  List of component names to disable for speed
                             (e.g., ['parser'] if only NER is needed).
        add_sentencizer:     Add a fast sentencizer (rule-based) if parser is disabled.

    Returns:
        Configured spaCy Language object.
    """
    if disable_components:
        nlp = spacy.load(model_name, disable=disable_components)
        print(f"Loaded {model_name}, disabled: {disable_components}")
    else:
        nlp = spacy.load(model_name)
        print(f"Loaded {model_name}")

    if add_sentencizer and "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    print(f"Pipeline components: {nlp.pipe_names}")
    print(f"Vocab size: {len(nlp.vocab):,}")
    return nlp


def inspect_token(doc: spacy.tokens.Doc, n: int = 20) -> pd.DataFrame:
    """
    Display token-level linguistic annotations as a DataFrame.

    Args:
        doc: Processed spaCy Doc object.
        n:   Maximum number of tokens to display.

    Returns:
        DataFrame with text, lemma, POS, dependency, entity, and stop-word columns.
    """
    rows = []
    for token in list(doc)[:n]:
        rows.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "head": token.head.text,
            "ent_type": token.ent_type_ or "—",
            "is_stop": token.is_stop,
            "shape": token.shape_,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
```

### Step 2 — Custom EntityRuler for Domain-Specific NER

```python
def add_entity_ruler(
    nlp: spacy.language.Language,
    custom_patterns: List[Dict],
    ruler_position: str = "before",
    overwrite_ents: bool = True,
) -> spacy.language.Language:
    """
    Add an EntityRuler with custom patterns before or after the statistical NER.

    Pattern format:
        {"label": "DRUG", "pattern": "aspirin"}
        {"label": "GENE", "pattern": [{"LOWER": "tp53"}]}
        {"label": "ORG",  "pattern": [{"LOWER": "world"}, {"LOWER": "health"},
                                       {"LOWER": "organization"}]}

    Args:
        nlp:              spaCy pipeline.
        custom_patterns:  List of pattern dicts.
        ruler_position:   'before' (before NER, higher priority) or 'after'.
        overwrite_ents:   Whether to overwrite entities found by statistical NER.

    Returns:
        Updated pipeline.
    """
    ruler_name = "entity_ruler"
    if ruler_name in nlp.pipe_names:
        nlp.remove_pipe(ruler_name)

    config = {"overwrite_ents": overwrite_ents}
    if ruler_position == "before" and "ner" in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner", config=config)
    else:
        ruler = nlp.add_pipe("entity_ruler", last=True, config=config)

    ruler.add_patterns(custom_patterns)
    print(
        f"EntityRuler added with {len(custom_patterns)} patterns "
        f"({'before' if ruler_position == 'before' else 'after'} NER)"
    )
    return nlp


def run_ner_pipeline(
    nlp: spacy.language.Language,
    texts: List[str],
    show_visualization: bool = False,
    display_style: str = "ent",
) -> List[Dict]:
    """
    Run NER over a list of texts and return entity results.

    Args:
        nlp:                 Configured spaCy pipeline.
        texts:               List of input strings.
        show_visualization:  Render displacy HTML for the first text.
        display_style:       'ent' for entity colors, 'dep' for dependency tree.

    Returns:
        List of dicts: {text, entities: [{text, label, start, end}]}.
    """
    results = []
    for text in texts:
        doc = nlp(text)
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            }
            for ent in doc.ents
        ]
        results.append({"input_text": text, "entities": entities})

    if show_visualization and texts:
        first_doc = nlp(texts[0])
        html = displacy.render(first_doc, style=display_style, jupyter=False)
        print("displacy HTML generated (first text):")
        print(html[:500] + "...\n[truncated]")

    return results
```

### Step 3 — Dependency Parse and SVO Extraction

```python
def extract_svo_triples(
    doc: spacy.tokens.Doc,
) -> List[Dict]:
    """
    Extract Subject-Verb-Object triples from a parsed spaCy Doc.

    Algorithm:
    1. Find all tokens with dep_ == 'ROOT' (main verb).
    2. For each ROOT verb, find 'nsubj' (nominal subject) dependents.
    3. Find 'dobj' (direct object) or 'attr' dependents.
    4. Include compound and adjectival modifiers for richer triples.

    Args:
        doc: Parsed spaCy Doc (requires 'parser' component).

    Returns:
        List of {subject, verb, object, sentence} dicts.
    """
    triples = []

    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("ROOT", "relcl", "acl") and token.pos_ == "VERB":
                verb = token
                subjects = [
                    t for t in verb.lefts
                    if t.dep_ in ("nsubj", "nsubjpass", "csubj")
                ]
                objects = [
                    t for t in verb.rights
                    if t.dep_ in ("dobj", "attr", "nsubjpass", "pobj")
                ]

                for subj in subjects:
                    # Include compound modifiers
                    subj_span = " ".join(
                        [t.text for t in subj.subtree
                         if t.dep_ in ("compound", "amod", "det") or t == subj]
                    )
                    for obj in objects:
                        obj_span = " ".join(
                            [t.text for t in obj.subtree
                             if t.dep_ in ("compound", "amod", "det") or t == obj]
                        )
                        triples.append({
                            "subject": subj_span.strip(),
                            "verb": verb.lemma_,
                            "object": obj_span.strip(),
                            "sentence": sent.text.strip(),
                        })

    return triples


def analyze_dependency_structure(
    texts: List[str],
    nlp: spacy.language.Language,
) -> pd.DataFrame:
    """
    Analyze dependency structure across a list of texts.

    Returns DataFrame with sentence, SVO triples, and dependency counts.
    """
    all_triples = []
    dep_counts = Counter()

    for text in texts:
        doc = nlp(text)
        triples = extract_svo_triples(doc)
        all_triples.extend(triples)
        for token in doc:
            dep_counts[token.dep_] += 1

    print(f"Extracted {len(all_triples)} SVO triples from {len(texts)} texts")
    print("\nTop 10 dependency relations:")
    for dep, count in dep_counts.most_common(10):
        print(f"  {dep:15s}: {count}")

    return pd.DataFrame(all_triples) if all_triples else pd.DataFrame(
        columns=["subject", "verb", "object", "sentence"]
    )
```

---

## Advanced Usage

### Batch NER over Document Collection

```python
def batch_ner_collection(
    nlp: spacy.language.Language,
    texts: List[str],
    batch_size: int = 64,
    n_process: int = 1,
) -> pd.DataFrame:
    """
    Batch NER processing with nlp.pipe() for large document collections.

    `nlp.pipe()` is significantly faster than calling `nlp(text)` in a loop
    because it batches tokenization and vectorization.

    Args:
        nlp:        spaCy pipeline.
        texts:      List of document strings.
        batch_size: Batch size for nlp.pipe() (tune for memory vs speed).
        n_process:  Number of processes (1 for single-threaded; >1 for multicore).
                    Note: n_process > 1 requires 'fork' start method on Linux/Mac.

    Returns:
        DataFrame with columns: doc_id, entity_text, entity_label, start, end.
    """
    rows = []
    total = len(texts)
    print(f"Processing {total} documents (batch_size={batch_size})...")

    for doc_id, doc in enumerate(
        nlp.pipe(texts, batch_size=batch_size, n_process=n_process)
    ):
        for ent in doc.ents:
            rows.append({
                "doc_id": doc_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
        if (doc_id + 1) % 500 == 0:
            print(f"  Processed {doc_id + 1}/{total} documents")

    df = pd.DataFrame(rows)
    print(f"Total entities extracted: {len(df)}")
    print(f"Entity type distribution:\n{df['entity_label'].value_counts().head(10)}")
    return df


def entity_frequency_table(
    entity_df: pd.DataFrame,
    top_n: int = 20,
    entity_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate entity frequency table from batch NER output.

    Args:
        entity_df:    DataFrame from batch_ner_collection().
        top_n:        Number of top entities to return per type.
        entity_types: Filter to specific entity types (e.g., ['PERSON', 'ORG']).

    Returns:
        DataFrame with entity text, type, frequency, and document frequency.
    """
    if entity_types:
        entity_df = entity_df[entity_df["entity_label"].isin(entity_types)]

    freq = (
        entity_df.groupby(["entity_text", "entity_label"])
        .agg(
            frequency=("doc_id", "count"),
            doc_frequency=("doc_id", "nunique"),
        )
        .reset_index()
        .sort_values("frequency", ascending=False)
    )

    if entity_types:
        top_per_type = (
            freq.groupby("entity_label")
            .head(top_n)
            .reset_index(drop=True)
        )
        return top_per_type
    else:
        return freq.head(top_n)


@Language.component("custom_sentence_fixer")
def custom_sentence_fixer(doc):
    """
    Custom pipeline component that forces sentence starts after double newlines.
    Register with: nlp.add_pipe('custom_sentence_fixer', before='parser')
    """
    for token in doc[:-1]:
        if token.text == "\n\n":
            doc[token.i + 1].is_sent_start = True
    return doc
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| `OSError: Can't find model 'en_core_web_sm'` | Model not downloaded | Run `python -m spacy download en_core_web_sm` |
| Custom EntityRuler overridden by NER | Ruler position is 'after' | Set `ruler_position='before'` |
| SVO extraction empty | Parser disabled or short sentences | Ensure 'parser' is in `nlp.pipe_names` |
| `nlp.pipe()` OOM with large docs | Batch too large | Reduce `batch_size` to 16–32 |
| displacy shows no colors | Running in terminal | Use `jupyter=True` in Jupyter; save HTML otherwise |
| Non-ASCII characters tokenized incorrectly | Wrong model language | Use language-appropriate model |
| Slow processing on large corpus | Single-process | Set `n_process=4` on Linux (not Windows) |

---

## External Resources

- spaCy documentation: <https://spacy.io/usage>
- spaCy models: <https://spacy.io/models>
- EntityRuler patterns: <https://spacy.io/usage/rule-based-matching#entityruler>
- displacy visualizer: <https://spacy.io/usage/visualizers>
- Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding.
  <https://spacy.io/universe/project/spacy>

---

## Examples

### Example 1 — NER Pipeline with EntityRuler and displacy

```python
import spacy

# Load pipeline
nlp = load_pipeline("en_core_web_sm")

# Add domain-specific patterns (biomedical example)
medical_patterns = [
    {"label": "DRUG", "pattern": "aspirin"},
    {"label": "DRUG", "pattern": "ibuprofen"},
    {"label": "DRUG", "pattern": [{"LOWER": "selective"}, {"LOWER": "serotonin"},
                                   {"LOWER": "reuptake"}, {"LOWER": "inhibitor"}]},
    {"label": "CONDITION", "pattern": "major depressive disorder"},
    {"label": "CONDITION", "pattern": "PTSD"},
    {"label": "CONDITION", "pattern": "anxiety disorder"},
    {"label": "BIOMARKER", "pattern": "cortisol"},
    {"label": "BIOMARKER", "pattern": "BDNF"},
]

nlp = add_entity_ruler(nlp, medical_patterns, ruler_position="before")

# Test texts
texts = [
    "The patient was prescribed ibuprofen for pain and aspirin for cardiovascular prevention.",
    "Studies show BDNF levels are reduced in major depressive disorder.",
    "SSRIs are first-line treatment for anxiety disorder and PTSD symptoms.",
    "John Smith at WHO reported elevated cortisol in chronic stress patients.",
]

results = run_ner_pipeline(nlp, texts, show_visualization=True)

# Display results
for r in results:
    print(f"\nText: {r['input_text'][:80]}...")
    for ent in r["entities"]:
        print(f"  [{ent['label']:12s}] {ent['text']}")

# Token analysis
doc = nlp(texts[0])
inspect_token(doc, n=15)
```

### Example 2 — Dependency Parse SVO Extraction and Batch NER

```python
# SVO extraction
sentences = [
    "The scientists discovered a new protein that regulates cell division.",
    "Cognitive behavioral therapy reduces anxiety symptoms effectively.",
    "The pharmaceutical company developed a novel antidepressant drug.",
    "Patients who exercise regularly report lower depression scores.",
    "The researchers published their findings in Nature Neuroscience.",
]

svo_df = analyze_dependency_structure(sentences, nlp)
print("\nSVO Triples:")
print(svo_df.to_string(index=False))

# Batch NER over collection
news_corpus = [
    "Apple Inc. CEO Tim Cook announced new products in Cupertino, California.",
    "The United Nations held a summit in Geneva to discuss climate change.",
    "Dr. Jane Smith from Harvard University published research on Alzheimer's disease.",
    "Microsoft acquired OpenAI-backed startup for 1.5 billion dollars.",
    "Researchers at Johns Hopkins University found a cure for rare genetic disorder.",
    "The European Central Bank raised interest rates to combat inflation in France.",
    "Tesla founder Elon Musk tweeted about electric vehicles and SpaceX missions.",
    "The World Health Organization warned about rising cases of dengue fever.",
]

entity_df = batch_ner_collection(nlp, news_corpus, batch_size=4)

# Frequency table for persons and organizations
freq_table = entity_frequency_table(
    entity_df, top_n=10, entity_types=["PERSON", "ORG", "GPE"]
)
print("\nTop entities by type:")
print(freq_table.to_string(index=False))

# Visualize entity type distribution
fig, ax = plt.subplots(figsize=(9, 4))
entity_df["entity_label"].value_counts().plot(kind="bar", ax=ax, color="steelblue",
                                               edgecolor="white")
ax.set_title("Entity Type Distribution")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
plt.savefig("entity_distribution.png", dpi=150)
plt.show()
print("NLP pipeline analysis complete.")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — model loading, EntityRuler, SVO extraction, batch NER, displacy |
