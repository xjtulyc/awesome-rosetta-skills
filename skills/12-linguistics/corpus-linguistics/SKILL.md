---
name: corpus-linguistics
description: >
  Corpus linguistic analysis with NLTK and spaCy: tokenization, collocation, keyword analysis,
  KWIC concordance, distributional semantics with word2vec, and diachronic frequency trends.
tags:
  - linguistics
  - corpus-linguistics
  - nlp
  - spacy
  - nltk
  - word2vec
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
  - nltk>=3.8.0
  - spacy>=3.6.0
  - gensim>=4.3.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - scipy>=1.10.0
  - collections-extended>=2.0.0
last_updated: "2026-03-17"
---

# Corpus Linguistics with NLTK and spaCy

## Overview

Corpus linguistics studies language through large, principled collections of authentic texts.
This skill covers the complete analytical pipeline:

- Text preprocessing (tokenization, lemmatization, POS tagging, NER with spaCy)
- Frequency analysis (words, lemmas, bigrams, hapax legomena)
- Collocation analysis (MI, log-likelihood, t-score)
- KWIC (Key Word In Context) concordance
- Keyword analysis (log-likelihood comparison vs reference corpus)
- Distributional semantics (word2vec via gensim)
- N-gram language models
- Diachronic frequency trends

---

## Setup

```bash
pip install nltk spacy gensim pandas numpy matplotlib scipy

# Download spaCy models (English and German as examples)
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg   # Larger model for better NER/vectors

# NLTK data downloads (run once)
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
"
```

---

## Core Functions

```python
import re
import math
import string
import itertools
import collections
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import chi2_contingency

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.util import ngrams
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder

import spacy

import gensim
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser


# ---------------------------------------------------------------------------
# Globals / lazy loading
# ---------------------------------------------------------------------------

_SPACY_MODELS: dict[str, spacy.Language] = {}


def _get_spacy_model(lang: str = "en") -> spacy.Language:
    """Load and cache a spaCy model."""
    model_map = {
        "en": "en_core_web_sm",
        "en_lg": "en_core_web_lg",
        "de": "de_core_news_sm",
        "fr": "fr_core_news_sm",
        "es": "es_core_news_sm",
    }
    model_name = model_map.get(lang, lang)
    if model_name not in _SPACY_MODELS:
        _SPACY_MODELS[model_name] = spacy.load(model_name)
    return _SPACY_MODELS[model_name]


# ---------------------------------------------------------------------------
# 1. Corpus Preprocessing
# ---------------------------------------------------------------------------


def preprocess_corpus(
    texts: list[str],
    lang: str = "en",
    lowercase: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = False,
    lemmatize: bool = True,
    min_token_len: int = 2,
    batch_size: int = 256,
    n_process: int = 1,
) -> list[list[str]]:
    """
    Tokenize, lemmatize, and optionally clean a list of texts using spaCy.

    Parameters
    ----------
    texts : list of str
        Raw text documents.
    lang : str
        spaCy model key (``"en"``, ``"de"``, ``"fr"``, ``"es"``).
    lowercase : bool
        Lowercase all tokens.
    remove_punct : bool
        Remove punctuation tokens.
    remove_stopwords : bool
        Remove function words.
    lemmatize : bool
        Return lemmas instead of surface forms.
    min_token_len : int
        Minimum token character length.
    batch_size : int
        spaCy pipe batch size.
    n_process : int
        Number of processes for spaCy pipe (1 = single-threaded).

    Returns
    -------
    list of list of str
        Tokenized (and optionally lemmatized) documents.
    """
    nlp = _get_spacy_model(lang)
    # Disable unused pipes for speed
    disabled = ["ner", "parser"] if "ner" in nlp.pipe_names else []
    stop_set = set(nlp.Defaults.stop_words) if remove_stopwords else set()

    processed = []
    with nlp.select_pipes(disable=disabled):
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            tokens = []
            for tok in doc:
                if remove_punct and (tok.is_punct or tok.is_space):
                    continue
                form = tok.lemma_ if lemmatize else tok.text
                if lowercase:
                    form = form.lower()
                if remove_stopwords and form in stop_set:
                    continue
                if len(form) < min_token_len:
                    continue
                tokens.append(form)
            processed.append(tokens)
    return processed


def get_pos_tagged_corpus(
    texts: list[str],
    lang: str = "en",
) -> list[list[tuple[str, str]]]:
    """
    Return POS-tagged tokens as (token, POS_tag) pairs using spaCy Universal POS tags.
    """
    nlp = _get_spacy_model(lang)
    tagged_docs = []
    for doc in nlp.pipe(texts, batch_size=128):
        tagged_docs.append([(tok.text, tok.pos_) for tok in doc if not tok.is_space])
    return tagged_docs


def extract_named_entities(
    texts: list[str],
    lang: str = "en",
    entity_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract named entities across a corpus.

    Parameters
    ----------
    entity_types : list of str, optional
        Filter to specific types: ``["PERSON", "ORG", "GPE", "DATE"]`` etc.

    Returns
    -------
    pd.DataFrame with columns: doc_idx, text, label, start_char, end_char.
    """
    nlp = _get_spacy_model(lang)
    records = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        for ent in doc.ents:
            if entity_types and ent.label_ not in entity_types:
                continue
            records.append({
                "doc_idx": i,
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Frequency Analysis
# ---------------------------------------------------------------------------


def compute_frequency_profile(
    tokenized_corpus: list[list[str]],
    n: int = 2,
) -> dict:
    """
    Compute unigram and n-gram frequency distributions plus hapax legomena.

    Parameters
    ----------
    tokenized_corpus : list of list of str
        Output of ``preprocess_corpus``.
    n : int
        Maximum n-gram order to compute (1 = unigrams only, 2 = + bigrams, etc.).

    Returns
    -------
    dict with keys ``"unigrams"``, ``"bigrams"`` (if n≥2), ``"trigrams"`` (if n≥3),
    ``"hapax_legomena"``, ``"type_token_ratio"``, ``"total_tokens"``, ``"vocabulary_size"``.
    """
    all_tokens = list(itertools.chain.from_iterable(tokenized_corpus))
    total = len(all_tokens)
    freq = collections.Counter(all_tokens)
    hapax = [w for w, c in freq.items() if c == 1]
    ttr = len(freq) / total if total > 0 else 0.0

    result = {
        "unigrams": freq,
        "hapax_legomena": hapax,
        "type_token_ratio": ttr,
        "total_tokens": total,
        "vocabulary_size": len(freq),
    }

    if n >= 2:
        bigram_freq = collections.Counter(ngrams(all_tokens, 2))
        result["bigrams"] = bigram_freq

    if n >= 3:
        trigram_freq = collections.Counter(ngrams(all_tokens, 3))
        result["trigrams"] = trigram_freq

    print(
        f"Corpus: {total:,} tokens, {len(freq):,} types, "
        f"TTR={ttr:.4f}, hapax={len(hapax):,}"
    )
    return result


# ---------------------------------------------------------------------------
# 3. Collocation Analysis
# ---------------------------------------------------------------------------


def compute_collocations(
    corpus: list[list[str]],
    node_word: str | None = None,
    window: int = 5,
    stat: str = "pmi",
    top_n: int = 30,
    min_freq: int = 5,
) -> pd.DataFrame:
    """
    Compute collocations using pointwise mutual information, log-likelihood, or t-score.

    Parameters
    ----------
    corpus : list of list of str
        Tokenized corpus.
    node_word : str, optional
        Focus word. If None, compute the top collocating pairs in the whole corpus.
    window : int
        Context window size (tokens on each side).
    stat : str
        Association measure: ``"pmi"``, ``"ll"`` (log-likelihood), ``"tscore"``, ``"raw_freq"``.
    top_n : int
        Number of top collocates to return.
    min_freq : int
        Minimum joint frequency filter.

    Returns
    -------
    pd.DataFrame with columns: w1, w2, score, joint_freq.
    """
    flat = list(itertools.chain.from_iterable(corpus))
    finder = BigramCollocationFinder.from_words(flat, window_size=window)
    finder.apply_freq_filter(min_freq)

    measures = BigramAssocMeasures()
    stat_map = {
        "pmi": measures.pmi,
        "ll": measures.likelihood_ratio,
        "tscore": measures.student_t,
        "raw_freq": measures.raw_freq,
    }
    score_fn = stat_map.get(stat, measures.pmi)

    if node_word:
        scored = [
            (w1, w2, score)
            for (w1, w2), score in finder.score_ngrams(score_fn)
            if node_word in (w1, w2)
        ][:top_n]
    else:
        scored = [(w1, w2, score) for (w1, w2), score in finder.score_ngrams(score_fn)][:top_n]

    records = []
    for w1, w2, score in scored:
        joint = finder.ngram_fd.get((w1, w2), 0)
        records.append({"w1": w1, "w2": w2, "score": round(score, 4), "joint_freq": joint})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Keyword Analysis (Log-Likelihood)
# ---------------------------------------------------------------------------


def keyword_analysis(
    target_corpus: list[list[str]],
    reference_corpus: list[list[str]],
    top_n: int = 50,
    min_freq: int = 5,
    effect_size: bool = True,
) -> pd.DataFrame:
    """
    Identify statistically key words in a target corpus compared to a reference corpus.

    Uses the log-likelihood (G²) statistic (Dunning 1993), which is robust for frequency
    comparison across corpora of different sizes.

    Parameters
    ----------
    target_corpus : list of list of str
        Tokenized target corpus.
    reference_corpus : list of list of str
        Tokenized reference corpus.
    top_n : int
        Number of keywords to return (positive keywords only).
    min_freq : int
        Minimum frequency in target corpus.
    effect_size : bool
        If True, compute %DIFF and log ratio effect sizes.

    Returns
    -------
    pd.DataFrame sorted by G² descending.
    """
    target_flat = list(itertools.chain.from_iterable(target_corpus))
    ref_flat = list(itertools.chain.from_iterable(reference_corpus))

    target_freq = collections.Counter(target_flat)
    ref_freq = collections.Counter(ref_flat)
    N_t = len(target_flat)
    N_r = len(ref_flat)

    all_words = set(target_freq.keys()) | set(ref_freq.keys())
    records = []

    for word in all_words:
        a = target_freq.get(word, 0)
        b = ref_freq.get(word, 0)
        if a < min_freq:
            continue

        c = N_t - a
        d = N_r - b
        # G² = 2 * sum(observed * ln(observed / expected))
        contingency = np.array([[a, b], [c, d]], dtype=float)
        # Avoid log(0) by adding epsilon
        expected = np.outer(contingency.sum(axis=1), contingency.sum(axis=0)) / contingency.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            g2 = 2 * np.nansum(contingency * np.log((contingency + 1e-10) / (expected + 1e-10)))

        row = {
            "word": word,
            "target_freq": a,
            "ref_freq": b,
            "target_per_mil": a / N_t * 1_000_000,
            "ref_per_mil": b / N_r * 1_000_000,
            "G2": round(g2, 3),
            "p_approx": "< 0.001" if g2 > 10.83 else ("< 0.01" if g2 > 6.63 else "ns"),
            "keyness_direction": "positive" if (a / N_t) > (b / N_r) else "negative",
        }
        if effect_size:
            t_norm = a / N_t * 1_000_000
            r_norm = b / N_r * 1_000_000
            row["percent_diff"] = round((t_norm - r_norm) / (r_norm + 1e-9) * 100, 2)
            row["log_ratio"] = round(np.log2((t_norm + 1) / (r_norm + 1)), 4)
        records.append(row)

    df = pd.DataFrame(records)
    positive_keys = (
        df[df["keyness_direction"] == "positive"]
        .sort_values("G2", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return positive_keys


# ---------------------------------------------------------------------------
# 5. KWIC Concordance
# ---------------------------------------------------------------------------


def kwic(
    corpus: list[list[str]],
    keyword: str,
    window: int = 5,
    case_sensitive: bool = False,
    max_lines: int = 100,
    doc_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate a Key Word In Context (KWIC) concordance.

    Parameters
    ----------
    corpus : list of list of str
        Tokenized corpus.
    keyword : str
        Search keyword (exact token match after lowercasing if not case_sensitive).
    window : int
        Number of context tokens on each side.
    case_sensitive : bool
        If False, match regardless of case.
    max_lines : int
        Maximum concordance lines to return.
    doc_labels : list of str, optional
        Document identifiers for the ``doc_id`` column.

    Returns
    -------
    pd.DataFrame with columns: doc_id, position, left_context, node, right_context.
    """
    target = keyword if case_sensitive else keyword.lower()
    records = []

    for doc_idx, tokens in enumerate(corpus):
        label = doc_labels[doc_idx] if doc_labels else str(doc_idx)
        for pos, token in enumerate(tokens):
            t = token if case_sensitive else token.lower()
            if t == target:
                left = tokens[max(0, pos - window): pos]
                right = tokens[pos + 1: pos + 1 + window]
                records.append({
                    "doc_id": label,
                    "position": pos,
                    "left_context": " ".join(left),
                    "node": token,
                    "right_context": " ".join(right),
                })
                if len(records) >= max_lines:
                    break
        if len(records) >= max_lines:
            break

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Word2Vec Distributional Semantics
# ---------------------------------------------------------------------------


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = 200,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 10,
    detect_phrases: bool = True,
) -> Word2Vec:
    """
    Train a Word2Vec model on a tokenized corpus.

    Parameters
    ----------
    sentences : list of list of str
        Tokenized corpus (one document per inner list).
    vector_size : int
        Dimensionality of word vectors.
    window : int
        Context window size.
    min_count : int
        Ignore words with frequency below this threshold.
    workers : int
        Number of CPU threads.
    sg : int
        Training algorithm: 1 = Skip-gram, 0 = CBOW.
    epochs : int
        Training epochs.
    detect_phrases : bool
        If True, detect and join common bigram phrases (e.g. "New_York").

    Returns
    -------
    gensim.models.Word2Vec
    """
    corpus = sentences
    if detect_phrases:
        bigram_phrases = Phrases(sentences, min_count=5, threshold=10)
        bigram_phraser = Phraser(bigram_phrases)
        corpus = [bigram_phraser[sent] for sent in sentences]

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        seed=42,
    )
    print(
        f"Word2Vec trained: {len(model.wv)} vocabulary, "
        f"{vector_size}d vectors, {'Skip-gram' if sg else 'CBOW'}"
    )
    return model


def compute_cosine_similarity(
    model: Word2Vec,
    word_pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Compute cosine similarity for a list of word pairs."""
    records = []
    for w1, w2 in word_pairs:
        if w1 not in model.wv or w2 not in model.wv:
            sim = float("nan")
        else:
            sim = model.wv.similarity(w1, w2)
        records.append({"word1": w1, "word2": w2, "cosine_similarity": round(sim, 4)})
    return pd.DataFrame(records)
```

---

## Example A: Diachronic Lexical Change in a News Corpus

This example tracks how the normalized frequency of selected words changes over years in a
collection of news articles organized by publication year. It also computes keyword profiles
for individual decades.

```python
# ── Example A ─────────────────────────────────────────────────────────────
import os
import glob
import json

# --- Expected corpus structure -----------------------------------------------
# news_corpus/
#   1990/  article_001.txt  article_002.txt ...
#   2000/  ...
#   2010/  ...
#   2020/  ...

CORPUS_ROOT = os.environ.get("NEWS_CORPUS_ROOT", "news_corpus")
TARGET_WORDS = ["internet", "climate", "terrorism", "pandemic", "artificial", "algorithm"]


def load_corpus_by_year(root: str) -> dict[int, list[str]]:
    """Load text files organized in year subdirectories."""
    corpus_by_year: dict[int, list[str]] = {}
    for year_dir in sorted(Path(root).iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        texts = []
        for txt_file in year_dir.glob("*.txt"):
            texts.append(txt_file.read_text(encoding="utf-8", errors="ignore"))
        if texts:
            corpus_by_year[year] = texts
    return corpus_by_year


# --- Load and process -------------------------------------------------------
print("Loading corpus...")
corpus_by_year = load_corpus_by_year(CORPUS_ROOT)
years_available = sorted(corpus_by_year.keys())
print(f"Years available: {years_available}")

year_freqs: dict[int, dict] = {}
year_tokens: dict[int, list[list[str]]] = {}

for year in years_available:
    print(f"Processing {year}...")
    tokenized = preprocess_corpus(
        corpus_by_year[year],
        lang="en",
        lowercase=True,
        remove_punct=True,
        remove_stopwords=False,  # Keep stopwords for full frequency profile
        lemmatize=True,
    )
    year_tokens[year] = tokenized
    freq_profile = compute_frequency_profile(tokenized, n=1)
    year_freqs[year] = freq_profile

# --- Normalized frequency trends (per million words) ------------------------
records = []
for year in years_available:
    total = year_freqs[year]["total_tokens"]
    unigrams = year_freqs[year]["unigrams"]
    for word in TARGET_WORDS:
        freq = unigrams.get(word, 0)
        ppm = freq / total * 1_000_000 if total > 0 else 0.0
        records.append({"year": year, "word": word, "freq": freq, "per_million": ppm})

freq_df = pd.DataFrame(records)

# --- Plot frequency trends --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
palette = plt.cm.tab10(np.linspace(0, 1, len(TARGET_WORDS)))
for word, color in zip(TARGET_WORDS, palette):
    wdata = freq_df[freq_df["word"] == word].sort_values("year")
    ax.plot(wdata["year"], wdata["per_million"], marker="o", label=word, color=color, linewidth=2)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Frequency per Million Words", fontsize=12)
ax.set_title("Lexical Frequency Trends in News Corpus", fontsize=14)
ax.legend(loc="upper left", fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fig.tight_layout()
plt.savefig("diachronic_frequency_trends.png", dpi=150)
plt.show()

# --- Keyword analysis: 2020s vs 1990s ----------------------------------------
if 2020 in year_tokens and 1990 in year_tokens:
    target_texts = list(itertools.chain.from_iterable(
        year_tokens[y] for y in years_available if y >= 2020
    ))
    reference_texts = list(itertools.chain.from_iterable(
        year_tokens[y] for y in years_available if 1990 <= y < 2000
    ))

    # Wrap flat token lists as single-document corpora
    keywords_df = keyword_analysis(
        target_corpus=[target_texts],
        reference_corpus=[reference_texts],
        top_n=30,
        min_freq=10,
    )
    print("\nTop keywords (2020s vs 1990s reference):")
    print(keywords_df[["word", "target_per_mil", "ref_per_mil", "G2", "log_ratio"]].to_string())

# --- Collocation profile of "climate" over time ----------------------------
for year in [1990, 2000, 2010, 2020]:
    if year not in year_tokens:
        continue
    coll_df = compute_collocations(
        year_tokens[year],
        node_word="climate",
        window=4,
        stat="pmi",
        top_n=10,
        min_freq=3,
    )
    print(f"\nTop collocates of 'climate' in {year}:")
    print(coll_df[["w1", "w2", "score"]].to_string(index=False))
```

---

## Example B: Collocation Profile of Academic Vocabulary Across Disciplines

This example loads academic corpora for different disciplines, identifies the top collocates
of shared academic vocabulary (high-frequency multi-disciplinary terms), and builds word2vec
models per discipline to compare semantic neighbourhoods.

```python
# ── Example B ─────────────────────────────────────────────────────────────
# Input: one text file per discipline
# HUMANITIES_TXT, SCIENCES_TXT, SOCIAL_SCIENCES_TXT (paths via env vars)

DISCIPLINE_FILES = {
    "humanities": os.environ.get("HUMANITIES_TXT", "humanities_corpus.txt"),
    "sciences": os.environ.get("SCIENCES_TXT", "sciences_corpus.txt"),
    "social_sciences": os.environ.get("SOCIAL_SCIENCES_TXT", "social_sciences_corpus.txt"),
}

ACADEMIC_VOCAB = [
    "approach", "analysis", "framework", "evidence",
    "significant", "model", "theory", "context",
]

discipline_corpora: dict[str, list[list[str]]] = {}
discipline_models: dict[str, Word2Vec] = {}

for discipline, filepath in DISCIPLINE_FILES.items():
    if not os.path.exists(filepath):
        print(f"Skipping {discipline}: file not found.")
        continue
    print(f"\n--- {discipline.upper()} ---")
    raw_text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    # Split into ~sentence-length chunks for Word2Vec training
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", raw_text) if len(s.split()) > 5]
    tokenized = preprocess_corpus(
        sentences,
        lang="en",
        lowercase=True,
        remove_punct=True,
        remove_stopwords=False,
        lemmatize=True,
    )
    discipline_corpora[discipline] = tokenized

    # Frequency profile
    freq_profile = compute_frequency_profile(tokenized, n=2)
    print(f"Total tokens: {freq_profile['total_tokens']:,}")
    print(f"Vocabulary:   {freq_profile['vocabulary_size']:,}")

    # Train Word2Vec
    model = train_word2vec(
        tokenized,
        vector_size=150,
        window=5,
        min_count=5,
        sg=1,
        epochs=15,
        detect_phrases=True,
    )
    discipline_models[discipline] = model

# --- Collocation comparison: 'analysis' across disciplines ------------------
print("\n=== Collocation profiles: 'analysis' ===")
for discipline, tokenized in discipline_corpora.items():
    coll = compute_collocations(
        tokenized,
        node_word="analysis",
        window=3,
        stat="pmi",
        top_n=10,
        min_freq=5,
    )
    print(f"\n[{discipline}]")
    print(coll[["w1", "w2", "score"]].to_string(index=False))

# --- Semantic similarity: cosine similarity between ACADEMIC_VOCAB words ----
print("\n=== Semantic Similarity (sciences vs humanities) ===")
word_pairs = list(itertools.combinations(ACADEMIC_VOCAB, 2))

if "sciences" in discipline_models and "humanities" in discipline_models:
    sim_sci = compute_cosine_similarity(discipline_models["sciences"], word_pairs)
    sim_hum = compute_cosine_similarity(discipline_models["humanities"], word_pairs)
    comparison = sim_sci.merge(
        sim_hum,
        on=["word1", "word2"],
        suffixes=("_sci", "_hum"),
    )
    comparison["delta"] = (comparison["cosine_similarity_sci"] - comparison["cosine_similarity_hum"]).abs()
    print(comparison.sort_values("delta", ascending=False).head(15).to_string(index=False))

# --- Nearest neighbors of "theory" in each discipline ----------------------
TARGET_WORD = "theory"
print(f"\n=== Nearest Neighbours of '{TARGET_WORD}' ===")
for discipline, model in discipline_models.items():
    if TARGET_WORD in model.wv:
        neighbors = model.wv.most_similar(TARGET_WORD, topn=8)
        words = ", ".join(f"{w}({s:.2f})" for w, s in neighbors)
        print(f"[{discipline}]: {words}")

# --- KWIC concordance for 'framework' in social sciences --------------------
if "social_sciences" in discipline_corpora:
    kwic_df = kwic(
        discipline_corpora["social_sciences"],
        keyword="framework",
        window=4,
        max_lines=20,
    )
    print("\n=== KWIC: 'framework' in Social Sciences ===")
    for _, row in kwic_df.iterrows():
        print(f"... {row['left_context']} [{row['node']}] {row['right_context']} ...")

# --- Keyword analysis: Sciences vs Humanities --------------------------------
if "sciences" in discipline_corpora and "humanities" in discipline_corpora:
    kw_df = keyword_analysis(
        target_corpus=discipline_corpora["sciences"],
        reference_corpus=discipline_corpora["humanities"],
        top_n=25,
        min_freq=10,
    )
    print("\n=== Science Keywords (vs Humanities reference) ===")
    print(kw_df[["word", "target_per_mil", "ref_per_mil", "G2", "log_ratio"]].head(20).to_string())
```

---

## N-Gram Language Models

For probabilistic language modeling and text generation:

```python
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline


def train_ngram_lm(
    tokenized_corpus: list[list[str]],
    n: int = 3,
    model_type: str = "kneser_ney",
) -> tuple:
    """
    Train an n-gram language model.

    Parameters
    ----------
    tokenized_corpus : list of list of str
        Training corpus.
    n : int
        N-gram order (e.g. 3 = trigram).
    model_type : str
        ``"mle"``, ``"laplace"``, or ``"kneser_ney"``.

    Returns
    -------
    tuple: (fitted model, vocabulary)
    """
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_corpus)

    if model_type == "mle":
        lm = MLE(n)
    elif model_type == "laplace":
        lm = Laplace(n)
    else:
        lm = KneserNeyInterpolated(n)

    lm.fit(train_data, padded_sents)
    print(f"LM trained: {len(lm.vocab)} vocabulary, order={n}, type={model_type}")
    return lm, lm.vocab


def compute_perplexity(
    lm,
    test_corpus: list[list[str]],
    n: int,
) -> float:
    """
    Compute perplexity of the LM on a test corpus.

    Lower perplexity = better fit.
    """
    from nltk.lm.preprocessing import padded_everygram_pipeline

    test_data, _ = padded_everygram_pipeline(n, test_corpus)
    total_log_prob = 0.0
    total_count = 0
    for sent in test_corpus:
        for ngram in ngrams(["<s>"] * (n - 1) + sent + ["</s>"], n):
            lp = lm.logscore(ngram[-1], context=ngram[:-1])
            if not math.isinf(lp):
                total_log_prob += lp
                total_count += 1

    if total_count == 0:
        return float("inf")
    avg_log_prob = total_log_prob / total_count
    return 2 ** (-avg_log_prob)
```

---

## Notes and Best Practices

### spaCy Model Selection

| Use Case | Model |
|---|---|
| Speed-critical pipeline | `en_core_web_sm` |
| Better NER / accuracy | `en_core_web_lg` |
| Transformer-based | `en_core_web_trf` (requires `spacy-transformers`) |

### Log-Likelihood vs PMI

- **Log-likelihood (G²)**: Best for keyword analysis and low-frequency phenomena; not sensitive
  to corpus size imbalance.
- **PMI**: Tends to overestimate associations for rare words; use with `min_freq >= 5`.
- **T-score**: Conservative; best for identifying grammatically stable collocations.

### Word2Vec Practical Tips

- Preprocess consistently before training: lowercase, remove punctuation, optionally lemmatize.
- Use `sg=1` (Skip-gram) for rare words; `sg=0` (CBOW) is faster and works well for frequent words.
- `vector_size=200` is a good default; increase to 300 for large corpora (>50 M tokens).
- Save/load models with `model.save("model.bin")` / `Word2Vec.load("model.bin")`.

### References

- Biber, D., Conrad, S., & Reppen, R. (1998). *Corpus Linguistics: Investigating Language
  Structure and Use*. Cambridge University Press.
- Dunning, T. (1993). Accurate methods for the statistics of surprise and coincidence.
  *Computational Linguistics*, 19(1), 61–74.
- Mikolov, T. et al. (2013). Efficient estimation of word representations in vector space.
  *arXiv:1301.3781*.
- Rayson, P. (2008). From key words to key semantic domains. *International Journal of Corpus
  Linguistics*, 13(4), 519–549.
