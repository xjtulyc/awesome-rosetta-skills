---
name: topic-modeling-lit
description: >
  Scientific literature topic modeling with LDA and BERTopic: coherence optimization,
  dynamic trends, PyLDAvis visualization, and OpenAlex/PubMed abstract workflows.
tags:
  - nlp
  - topic-modeling
  - literature-review
  - bertopic
  - lda
  - scientometrics
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
  - gensim>=4.3.0
  - nltk>=3.8.0
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scikit-learn>=1.3.0
  - bertopic>=0.16.0
  - sentence-transformers>=2.2.0
  - umap-learn>=0.5.0
  - hdbscan>=0.8.0
  - pyLDAvis>=3.4.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# topic-modeling-lit: Scientific Literature Topic Modeling

This skill covers end-to-end topic modeling on scientific abstract corpora, from
preprocessing through LDA coherence selection, BERTopic clustering, dynamic topic
modeling, and publication-ready visualisations.

## Installation

```bash
pip install gensim nltk numpy pandas scikit-learn bertopic sentence-transformers \
            umap-learn hdbscan pyLDAvis matplotlib
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 1. Preprocessing Abstracts

```python
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update({
    "study", "result", "results", "show", "shows", "showed",
    "use", "used", "using", "based", "method", "methods",
    "data", "analysis", "approach", "propose", "proposed",
    "paper", "work", "also", "may", "however", "two", "one",
    "new", "present", "find", "found", "provide", "high", "large",
})
LEMMATIZER = WordNetLemmatizer()


def _clean_text(text: str) -> list:
    """Lowercase, remove punctuation/numbers, tokenise, stop-word filter, lemmatise."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2 and t not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


def preprocess_abstracts(
    texts: list,
    min_doc_len: int = 10,
    no_below: int = 5,
    no_above: float = 0.85,
    keep_n: int = 50000,
) -> tuple:
    """
    Tokenise, filter, and build a gensim dictionary and bag-of-words corpus.

    Parameters
    ----------
    texts : list of str
    min_doc_len : int  — discard docs with fewer tokens after cleaning
    no_below : int     — drop tokens appearing in fewer than N docs
    no_above : float   — drop tokens appearing in more than this fraction of docs
    keep_n : int       — vocabulary size cap

    Returns
    -------
    tokenised : list of list of str
    dictionary : gensim.corpora.Dictionary
    corpus : list of BoW vectors
    """
    tokenised = [_clean_text(t) for t in texts]
    tokenised = [t for t in tokenised if len(t) >= min_doc_len]
    print(f"Tokenised {len(tokenised)} documents (dropped {len(texts) - len(tokenised)} short)")

    dictionary = corpora.Dictionary(tokenised)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    print(f"Vocabulary size: {len(dictionary):,} tokens")

    corpus = [dictionary.doc2bow(doc) for doc in tokenised]
    return tokenised, dictionary, corpus
```

---

## 2. LDA with Coherence-Based Topic Selection

```python
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


def run_lda(
    docs: list,
    n_topics_range: range = range(5, 31, 5),
    passes: int = 15,
    random_state: int = 42,
) -> dict:
    """
    Train LDA models for each value in n_topics_range, score by C_v coherence,
    and return the best model along with all scores.

    Parameters
    ----------
    docs : list of str  — raw abstract texts
    n_topics_range : range  — topic counts to evaluate
    passes : int  — number of LDA training passes

    Returns
    -------
    dict with keys: best_model, best_n_topics, all_models, coherence_scores,
                    dictionary, corpus, tokenised
    """
    tokenised, dictionary, corpus = preprocess_abstracts(docs)

    models = {}
    coherence_scores = {}

    for k in n_topics_range:
        print(f"Training LDA k={k} …")
        lda = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=passes,
            alpha="auto",
            eta="auto",
            random_state=random_state,
        )
        cm = CoherenceModel(
            model=lda, texts=tokenised, dictionary=dictionary, coherence="c_v"
        )
        cv = cm.get_coherence()
        models[k] = lda
        coherence_scores[k] = cv
        print(f"  k={k}: C_v = {cv:.4f}")

    best_k = max(coherence_scores, key=coherence_scores.get)
    print(f"\nBest k={best_k}, C_v={coherence_scores[best_k]:.4f}")

    return {
        "best_model": models[best_k],
        "best_n_topics": best_k,
        "all_models": models,
        "coherence_scores": coherence_scores,
        "dictionary": dictionary,
        "corpus": corpus,
        "tokenised": tokenised,
    }


def compute_coherence_scores(
    dictionary, corpus, models: dict, texts: list
) -> pd.DataFrame:
    """
    Compute C_v and diversity scores for a dict of {k: LdaModel}.

    Diversity = fraction of unique words in top-10 words across all topics.
    """
    records = []
    for k, model in models.items():
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v")
        cv = cm.get_coherence()
        top_words = [
            word for _, words in model.show_topics(num_topics=k, num_words=10, formatted=False)
            for word, _ in words
        ]
        diversity = len(set(top_words)) / len(top_words) if top_words else 0.0
        records.append({"n_topics": k, "coherence_cv": cv, "diversity": diversity})
    return pd.DataFrame(records).sort_values("n_topics")
```

---

## 3. BERTopic

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN


def run_bertopic(
    docs: list,
    embedding_model: str = "all-MiniLM-L6-v2",
    n_components: int = 5,
    min_cluster_size: int = 15,
    min_samples: int = 5,
    nr_topics: str = "auto",
) -> tuple:
    """
    Run BERTopic on a list of abstracts.

    Parameters
    ----------
    docs : list of str
    embedding_model : str  — sentence-transformers model name
    n_components : int  — UMAP dimensionality
    min_cluster_size : int  — HDBSCAN minimum cluster size
    nr_topics : "auto" or int  — reduce to this many topics after initial fit

    Returns
    -------
    (BERTopic model, topics list, probabilities array)
    """
    embedder = SentenceTransformer(embedding_model)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"BERTopic: {n_topics} topics, {n_outliers} outliers ({100*n_outliers/len(docs):.1f}%)")
    return topic_model, topics, probs
```

---

## 4. Dynamic Topic Modeling

```python
def plot_topic_evolution(
    model,
    year_col: str,
    df: pd.DataFrame,
    docs: list,
    top_n_topics: int = 8,
    output_path: str = None,
):
    """
    Show how topic frequency changes over years using BERTopic's topics_over_time.

    Parameters
    ----------
    model : BERTopic  — fitted model
    year_col : str    — column in df with publication year (int)
    df : pd.DataFrame — one row per document, must align with docs list
    docs : list of str
    top_n_topics : int
    output_path : str  — save HTML figure if provided
    """
    timestamps = df[year_col].astype(int).tolist()
    topics_over_time = model.topics_over_time(
        docs, timestamps, nr_bins=len(df[year_col].unique())
    )

    fig = model.visualize_topics_over_time(
        topics_over_time, top_n_topics=top_n_topics
    )
    if output_path:
        fig.write_html(output_path)
        print(f"Saved temporal trend to '{output_path}'")
    return fig
```

---

## 5. Topic Export and Labeling

```python
def export_topic_summary(model, n_words: int = 10) -> pd.DataFrame:
    """
    Export a human-readable topic summary table.

    Works with both LdaModel (gensim) and BERTopic.

    Returns DataFrame with columns: topic_id, top_words, weight_sum (LDA)
    or count (BERTopic).
    """
    records = []

    # BERTopic
    if hasattr(model, "get_topic_info"):
        info = model.get_topic_info()
        for _, row in info.iterrows():
            topic_id = row["Topic"]
            if topic_id == -1:
                continue
            topic_words = model.get_topic(topic_id)
            words_str = ", ".join([w for w, _ in topic_words[:n_words]])
            records.append({
                "topic_id": topic_id,
                "top_words": words_str,
                "count": row.get("Count", np.nan),
            })
        return pd.DataFrame(records)

    # gensim LdaModel
    for topic_id in range(model.num_topics):
        word_probs = model.show_topic(topic_id, topn=n_words)
        words_str = ", ".join([w for w, _ in word_probs])
        weight_sum = sum(p for _, p in word_probs)
        records.append({
            "topic_id": topic_id,
            "top_words": words_str,
            "weight_sum": round(weight_sum, 4),
        })
    return pd.DataFrame(records)
```

---

## 6. PyLDAvis Integration

```python
def visualize_lda_pyldavis(lda_result: dict, output_path: str = "/tmp/lda_vis.html"):
    """
    Generate an interactive PyLDAvis visualisation and save to HTML.

    Parameters
    ----------
    lda_result : dict  — output of run_lda()
    output_path : str
    """
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis

    model = lda_result["best_model"]
    corpus = lda_result["corpus"]
    dictionary = lda_result["dictionary"]

    vis_data = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(vis_data, output_path)
    print(f"PyLDAvis saved to '{output_path}'")
    return vis_data
```

---

## 7. Examples

### Example A — LDA on 5000 Climate Science Abstracts with Coherence Selection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ---- Option 1: Fetch from OpenAlex (no API key required) ----
def fetch_openalex_abstracts(query: str, n: int = 200) -> pd.DataFrame:
    """
    Fetch paper metadata + reconstructed abstracts from OpenAlex.

    Parameters
    ----------
    query : str  — keyword search string
    n : int      — number of results (max 200 per call; paginate for more)
    """
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"abstract.search:{query}",
        "per-page": min(n, 200),
        "select": "title,publication_year,abstract_inverted_index",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    records = []
    for r in results:
        inv_idx = r.get("abstract_inverted_index") or {}
        if inv_idx:
            words_positions = [(w, pos) for w, positions in inv_idx.items()
                               for pos in positions]
            words_positions.sort(key=lambda x: x[1])
            abstract = " ".join(w for w, _ in words_positions)
        else:
            abstract = r.get("title", "")
        records.append({
            "title": r.get("title", ""),
            "year": r.get("publication_year"),
            "abstract": abstract,
        })
    return pd.DataFrame(records)


# Fetch 200 climate abstracts
df_climate = fetch_openalex_abstracts("climate change carbon", n=200)
df_climate = df_climate.dropna(subset=["abstract"])
df_climate = df_climate[df_climate["abstract"].str.len() > 100]
print(f"Working with {len(df_climate)} abstracts")

# Run LDA coherence sweep
lda_result = run_lda(
    docs=df_climate["abstract"].tolist(),
    n_topics_range=range(5, 21, 5),
    passes=10,
)

# Plot coherence curve
scores_df = compute_coherence_scores(
    lda_result["dictionary"],
    lda_result["corpus"],
    lda_result["all_models"],
    lda_result["tokenised"],
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(scores_df["n_topics"], scores_df["coherence_cv"], marker="o", color="steelblue")
ax.axvline(lda_result["best_n_topics"], linestyle="--", color="red",
           label=f"Best k={lda_result['best_n_topics']}")
ax.set_xlabel("Number of Topics")
ax.set_ylabel("C_v Coherence")
ax.set_title("LDA Coherence vs Number of Topics — Climate Abstracts")
ax.legend()
plt.tight_layout()
plt.savefig("/tmp/lda_coherence_climate.png", dpi=150)
plt.show()

# Print topic summary
summary = export_topic_summary(lda_result["best_model"], n_words=8)
print("\nTop LDA Topics:")
print(summary.to_string(index=False))

# Save PyLDAvis visualisation
visualize_lda_pyldavis(lda_result, "/tmp/lda_vis_climate.html")
```

### Example B — BERTopic on ML Paper Abstracts with 2015-2024 Temporal Trend

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch ML abstracts from OpenAlex across years
def fetch_ml_abstracts_by_year(start: int = 2015, end: int = 2024, per_year: int = 50) -> pd.DataFrame:
    """Retrieve ML abstracts for each year in [start, end]."""
    all_records = []
    for year in range(start, end + 1):
        url = "https://api.openalex.org/works"
        params = {
            "filter": f"abstract.search:deep learning neural network,publication_year:{year}",
            "per-page": per_year,
            "select": "title,publication_year,abstract_inverted_index",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for r in results:
                inv_idx = r.get("abstract_inverted_index") or {}
                if inv_idx:
                    words_positions = [(w, pos) for w, positions in inv_idx.items()
                                       for pos in positions]
                    words_positions.sort(key=lambda x: x[1])
                    abstract = " ".join(w for w, _ in words_positions)
                else:
                    abstract = r.get("title", "")
                all_records.append({
                    "title": r.get("title", ""),
                    "year": year,
                    "abstract": abstract,
                })
            print(f"  {year}: fetched {len(results)} papers")
        except Exception as exc:
            print(f"  {year}: WARN {exc}")
    return pd.DataFrame(all_records)


df_ml = fetch_ml_abstracts_by_year(2015, 2024, per_year=30)
df_ml = df_ml.dropna(subset=["abstract"])
df_ml = df_ml[df_ml["abstract"].str.len() > 80].reset_index(drop=True)
print(f"\nTotal ML abstracts: {len(df_ml)} across {df_ml['year'].nunique()} years")

docs_ml = df_ml["abstract"].tolist()

# Fit BERTopic
topic_model, topics, probs = run_bertopic(
    docs_ml,
    embedding_model="all-MiniLM-L6-v2",
    min_cluster_size=5,
    nr_topics=15,
)

# Export topic summary
summary_bt = export_topic_summary(topic_model, n_words=8)
print("\nBERTopic topics:")
print(summary_bt.head(10).to_string(index=False))

# Dynamic topic evolution
fig_dyn = plot_topic_evolution(
    model=topic_model,
    year_col="year",
    df=df_ml,
    docs=docs_ml,
    top_n_topics=8,
    output_path="/tmp/bertopic_ml_trends.html",
)

# Bar chart: topic distribution
df_ml["topic"] = topics
topic_counts = (
    df_ml[df_ml["topic"] >= 0]["topic"]
    .value_counts()
    .head(12)
    .reset_index()
)
topic_counts.columns = ["topic_id", "count"]
topic_counts = topic_counts.merge(summary_bt[["topic_id", "top_words"]], on="topic_id")

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(topic_counts)), topic_counts["count"], color="steelblue")
ax.set_yticks(range(len(topic_counts)))
ax.set_yticklabels(
    [f"T{row['topic_id']}: {row['top_words'][:40]}…"
     for _, row in topic_counts.iterrows()],
    fontsize=8,
)
ax.invert_yaxis()
ax.set_xlabel("Number of Documents")
ax.set_title("BERTopic — ML Paper Topics (2015-2024)")
plt.tight_layout()
plt.savefig("/tmp/bertopic_ml_distribution.png", dpi=150)
plt.show()
```

---

## 8. Tips and Gotchas

- **Coherence vs interpretability**: C_v coherence increases monotonically for large k
  on big corpora. Also use the diversity score and manual inspection to choose k.
- **BERTopic outliers**: Topic -1 is "noise". If outlier fraction exceeds 20%, reduce
  `min_cluster_size` or use `topic_model.reduce_outliers()` with a strategy.
- **GPU acceleration**: Pass `device="cuda"` to `SentenceTransformer` if a GPU is
  available; embeddings for 5000 docs drop from ~5 min to ~20 s.
- **Reproducibility**: UMAP uses random initialisation. Always set `random_state=42`
  in both UMAP and `BERTopic`.
- **OpenAlex rate limits**: The free tier allows ~10 requests/s. Add
  `time.sleep(0.15)` between requests in batch loops.
- **gensim LDA multicore**: Use `LdaMulticore` with `workers=4` for large corpora.

---

## 9. References

- Blei, Ng & Jordan (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993-1022.
- Grootendorst (2022). BERTopic: Neural topic modeling with class-based TF-IDF.
  arXiv:2203.05794.
- Röder, Both & Hinneburg (2015). Exploring the Space of Topic Coherence Measures.
  *WSDM 2015*.
- Priem et al. (2022). OpenAlex: A fully-open index of the world's research. arXiv:2205.01833.
