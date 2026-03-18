---
name: text-mining-science
description: Scientific text mining with topic modeling, NER for scientific entities, claim extraction, and trend detection in research literature corpora.
tags:
  - text-mining
  - topic-modeling
  - scientific-nlp
  - information-extraction
  - literature-analysis
version: "1.0.0"
authors:
  - "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - numpy>=1.24
    - pandas>=2.0
    - scikit-learn>=1.3
    - scipy>=1.11
    - matplotlib>=3.7
    - nltk>=3.8
    - gensim>=4.3
last_updated: "2026-03-17"
status: stable
---

# Scientific Text Mining

## When to Use This Skill

Use this skill when you need to:
- Extract topics from scientific abstracts using LDA or NMF
- Identify named entities (genes, chemicals, diseases, methods) in text
- Detect research trends and emerging topics over time
- Extract scientific claims and hypothesis statements
- Compute semantic similarity between papers or research areas
- Mine full-text papers for methodological information
- Build a search engine or recommendation system for papers

**Trigger keywords**: text mining, topic modeling, LDA, NMF, BERTopic, scientific NER, claim extraction, scientific literature, abstract analysis, trend detection, keyword extraction, TF-IDF, word embeddings, document similarity, information extraction, GENIA, BioNER, chemical NER, method extraction.

## Background & Key Concepts

### Latent Dirichlet Allocation (LDA)

LDA assumes each document $d$ is a mixture of topics, and each topic is a distribution over words:

$$P(w | d) = \sum_{k=1}^{K} P(w | z=k) \cdot P(z=k | d) = \sum_k \phi_{kw} \cdot \theta_{dk}$$

where $\theta_d \sim \text{Dir}(\alpha)$ and $\phi_k \sim \text{Dir}(\beta)$.

### Non-negative Matrix Factorization (NMF)

Factorize term-document matrix $\mathbf{V} \approx \mathbf{W}\mathbf{H}$ where all entries $\geq 0$. $\mathbf{W}$ gives topic-word loadings, $\mathbf{H}$ gives document-topic activations.

### TF-IDF Weighting

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \times \log\frac{N}{df(t)}$$

where $\text{tf}(t,d)$ is term frequency in document $d$, $N$ is total documents, $df(t)$ is number of documents containing term $t$.

### Coherence Score (Topic Quality)

$$C_V = \frac{1}{|T|} \sum_{t \in T} \text{score}(t)$$

where $\text{score}(t)$ measures pointwise mutual information of top-$n$ words in topic $t$.

### RAKE (Rapid Automatic Keyword Extraction)

Score phrases by word frequency and word degree in the co-occurrence graph:

$$S(w) = \frac{\text{degree}(w)}{\text{freq}(w)}$$

## Environment Setup

```bash
pip install numpy>=1.24 pandas>=2.0 scikit-learn>=1.3 scipy>=1.11 \
            matplotlib>=3.7 nltk>=3.8 gensim>=4.3
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
print("Text mining environment ready")
```

## Core Workflow

### Step 1: Topic Modeling with LDA and NMF

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import re

# -----------------------------------------------------------------
# Simulate a corpus of 300 scientific abstracts across 5 topics
# -----------------------------------------------------------------
np.random.seed(42)

topic_vocab = {
    "Machine Learning": [
        "neural network", "deep learning", "gradient descent", "training",
        "convolutional", "transformer", "attention mechanism", "overfitting",
        "regularization", "classification", "accuracy", "loss function",
        "backpropagation", "batch normalization", "dropout"
    ],
    "Climate Science": [
        "greenhouse gas", "carbon dioxide", "temperature anomaly", "precipitation",
        "sea level rise", "ocean acidification", "feedback mechanism", "forcing",
        "radiative", "atmosphere", "climate model", "warming", "emission",
        "renewable energy", "carbon sequestration"
    ],
    "Genomics": [
        "gene expression", "RNA sequencing", "genome", "mutation", "protein",
        "transcription factor", "epigenetics", "methylation", "chromosomal",
        "single nucleotide polymorphism", "CRISPR", "variant", "allele",
        "gene regulatory network", "pathway analysis"
    ],
    "Urban Planning": [
        "land use", "zoning regulation", "transit oriented development",
        "density", "walkability", "urban heat island", "infrastructure",
        "mixed use", "smart city", "mobility", "accessibility", "gentrification",
        "housing policy", "neighborhood", "transportation network"
    ],
    "Quantum Physics": [
        "quantum entanglement", "superposition", "wave function", "Hamiltonian",
        "decoherence", "qubit", "quantum gate", "Hilbert space", "operator",
        "measurement", "quantum computing", "interference", "photon",
        "Schrödinger equation", "eigenvalue"
    ],
}
topic_names = list(topic_vocab.keys())

def generate_abstract(topic, n_sentences=4):
    """Generate a synthetic abstract for a given topic."""
    words = topic_vocab[topic]
    sentences = []
    for _ in range(n_sentences):
        n_words = np.random.randint(3, 7)
        chosen = np.random.choice(words, n_words, replace=True)
        # Add generic scientific text
        fillers = ["we present", "this study investigates", "results show",
                   "we demonstrate", "our approach", "analysis reveals"]
        sentence = np.random.choice(fillers) + " " + " and ".join(chosen)
        sentences.append(sentence)
    return ". ".join(sentences) + "."

# Generate corpus with mixed topics
abstracts = []
doc_topics = []
for _ in range(300):
    # Allow some mixing
    primary_topic = np.random.choice(topic_names)
    text = generate_abstract(primary_topic)
    # 20% chance of adding secondary topic content
    if np.random.random() < 0.2:
        secondary = np.random.choice([t for t in topic_names if t != primary_topic])
        text += " " + generate_abstract(secondary, n_sentences=1)
    abstracts.append(text)
    doc_topics.append(primary_topic)

df_corpus = pd.DataFrame({"abstract": abstracts, "true_topic": doc_topics})

# -----------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------
def preprocess_text(text):
    """Basic text preprocessing: lowercase, remove special chars."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_corpus["clean_text"] = df_corpus["abstract"].apply(preprocess_text)

# -----------------------------------------------------------------
# TF-IDF + NMF Topic Modeling
# -----------------------------------------------------------------
n_topics = 5
n_top_words = 10

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, max_features=500,
                         stop_words="english")
X_tfidf = tfidf.fit_transform(df_corpus["clean_text"])
feature_names = tfidf.get_feature_names_out()

nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)
W_nmf = nmf.fit_transform(X_tfidf)   # document-topic matrix (n_docs, n_topics)
H_nmf = nmf.components_               # topic-word matrix (n_topics, n_vocab)

print("=== NMF Top Words per Topic ===")
for topic_idx in range(n_topics):
    top_words_idx = H_nmf[topic_idx].argsort()[-n_top_words:][::-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

# -----------------------------------------------------------------
# LDA Topic Modeling
# -----------------------------------------------------------------
tf_vec = CountVectorizer(max_df=0.95, min_df=2, max_features=500,
                          stop_words="english")
X_tf = tf_vec.fit_transform(df_corpus["clean_text"])
tf_features = tf_vec.get_feature_names_out()

lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                  learning_method="online", max_iter=50)
W_lda = lda.fit_transform(X_tf)      # document-topic matrix

print("\n=== LDA Top Words per Topic ===")
for topic_idx in range(n_topics):
    top_idx = lda.components_[topic_idx].argsort()[-n_top_words:][::-1]
    top_words = [tf_features[i] for i in top_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

# -----------------------------------------------------------------
# Assign dominant topic to each document
# -----------------------------------------------------------------
df_corpus["nmf_topic"] = W_nmf.argmax(axis=1)
df_corpus["lda_topic"] = W_lda.argmax(axis=1)

# Topic distribution
print("\nNMF Topic Distribution:")
print(df_corpus["nmf_topic"].value_counts().sort_index())

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Topic word bars (NMF)
for topic_idx in range(n_topics):
    top_n = 8
    top_idx = H_nmf[topic_idx].argsort()[-top_n:][::-1]
    words = [feature_names[i] for i in top_idx]
    scores = H_nmf[topic_idx][top_idx]
    axes[0].barh([f"T{topic_idx+1}: {w[:15]}" for w in words],
                  scores, alpha=0.6,
                  label=f"Topic {topic_idx+1}")
axes[0].set_title("NMF Topic-Word Loadings")
axes[0].set_xlabel("Weight")

# Document-topic heatmap (sample of 30 docs)
sample_idx = np.random.choice(len(df_corpus), 30, replace=False)
im = axes[1].imshow(W_nmf[sample_idx], cmap="YlOrRd", aspect="auto")
axes[1].set_xlabel("Topic"); axes[1].set_ylabel("Document (sample)")
axes[1].set_title("Document-Topic Matrix (NMF)")
axes[1].set_xticks(range(n_topics))
axes[1].set_xticklabels([f"T{i+1}" for i in range(n_topics)])
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("topic_modeling.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: topic_modeling.png")
```

### Step 2: Keyword Extraction and Trend Analysis

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter
import re

# -----------------------------------------------------------------
# Add years to corpus for trend analysis
# -----------------------------------------------------------------
years = np.random.randint(2010, 2024, len(df_corpus))
df_corpus["year"] = years

# -----------------------------------------------------------------
# RAKE-inspired keyword extraction
# -----------------------------------------------------------------
def rake_keywords(text, stopwords=None, min_chars=4, max_words=4):
    """Simple RAKE keyword extraction.

    Args:
        text: input string
        stopwords: set of stop words
        min_chars: minimum keyword character length
        max_words: max words in a keyphrase
    Returns:
        list of (score, keyphrase) tuples
    """
    if stopwords is None:
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "at",
                     "to", "of", "for", "we", "our", "this", "these", "is"}

    # Split on stop words and punctuation
    phrase_list = re.split(r"\b(?:" + "|".join(stopwords) + r")\b|[,\.;:()\[\]]",
                            text.lower())
    candidate_phrases = [p.strip() for p in phrase_list
                         if len(p.strip()) >= min_chars]
    candidate_phrases = [p for p in candidate_phrases
                         if 1 <= len(p.split()) <= max_words]

    # Word frequency and degree
    word_freq = Counter()
    word_degree = Counter()
    for phrase in candidate_phrases:
        words = phrase.split()
        for w in words:
            word_freq[w] += 1
            word_degree[w] += len(words)

    # Score phrases
    scored = []
    for phrase in candidate_phrases:
        words = phrase.split()
        score = sum(word_degree[w] / max(word_freq[w], 1) for w in words)
        scored.append((score, phrase))

    return sorted(scored, reverse=True)[:10]

# Extract keywords for a sample abstract
sample_text = df_corpus["abstract"].iloc[0]
print("=== RAKE Keywords from Sample Abstract ===")
print(f"Abstract: {sample_text[:200]}...")
for score, kw in rake_keywords(sample_text)[:5]:
    print(f"  '{kw}' (score={score:.2f})")

# -----------------------------------------------------------------
# TF-IDF keyword extraction over the full corpus
# -----------------------------------------------------------------
tfidf_kw = TfidfVectorizer(max_df=0.8, min_df=3, ngram_range=(1, 3),
                             max_features=1000, stop_words="english")
X_kw = tfidf_kw.fit_transform(df_corpus["clean_text"])
kw_names = tfidf_kw.get_feature_names_out()

# Top keywords by mean TF-IDF score
mean_tfidf = X_kw.mean(axis=0).A1
top_kw_idx = mean_tfidf.argsort()[-20:][::-1]
top_keywords = [(kw_names[i], mean_tfidf[i]) for i in top_kw_idx]
print("\n=== Top 20 Keywords (TF-IDF) ===")
for kw, score in top_keywords:
    print(f"  {kw}: {score:.4f}")

# -----------------------------------------------------------------
# Temporal keyword trend (annual frequency)
# -----------------------------------------------------------------
all_keywords = [kw for kw, _ in top_keywords[:8]]
trend_data = []
for yr in range(2010, 2024):
    yr_docs = df_corpus[df_corpus["year"] == yr]["clean_text"]
    if len(yr_docs) == 0:
        continue
    yr_text = " ".join(yr_docs)
    yr_counts = {kw: yr_text.count(kw) / max(len(yr_docs), 1)
                 for kw in all_keywords}
    yr_counts["year"] = yr
    trend_data.append(yr_counts)

trend_df = pd.DataFrame(trend_data).set_index("year")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top keywords bar chart
kw_labels, kw_scores = zip(*top_keywords)
axes[0].barh(kw_labels[::-1], kw_scores[::-1], color="steelblue", edgecolor="black")
axes[0].set_xlabel("Mean TF-IDF Score")
axes[0].set_title("Top Keywords by TF-IDF")

# Temporal trends for top keywords
for kw in all_keywords[:5]:
    if kw in trend_df.columns:
        axes[1].plot(trend_df.index, trend_df[kw], marker="o", ms=3, label=kw)
axes[1].set_xlabel("Year"); axes[1].set_ylabel("Avg. Frequency per Document")
axes[1].set_title("Keyword Temporal Trends")
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig("keyword_trends.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: keyword_trends.png")
```

### Step 3: Scientific NER and Claim Extraction

```python
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# -----------------------------------------------------------------
# Rule-based scientific NER (pattern matching approach)
# For production use: spaCy + SciBERT or SciSpacy
# -----------------------------------------------------------------
NER_PATTERNS = {
    "METHOD": [
        r"\b(random forest|neural network|LSTM|transformer|SVM|k-means|"
        r"linear regression|logistic regression|BERT|XGBoost|gradient boosting|"
        r"Monte Carlo|Bayesian inference|MCMC|principal component analysis|PCA)\b",
    ],
    "METRIC": [
        r"\b(accuracy|F1[- ]score|precision|recall|AUC|RMSE|MAE|R[²²]|"
        r"p[- ]value|confidence interval|statistical significance|"
        r"mean absolute error|root mean square)\b",
    ],
    "DATASET": [
        r"\b(ImageNet|MNIST|CIFAR|Wikipedia|PubMed|MEDLINE|arXiv|"
        r"MIMIC|UK Biobank|NHANES|CelebA|Penn Treebank|SQuAD)\b",
    ],
    "CLAIM_VERB": [
        r"\b(we show|we demonstrate|we find|we propose|our results|"
        r"results indicate|we report|our study|we observe|analysis shows)\b",
    ],
}

def scientific_ner(text, patterns=NER_PATTERNS):
    """Extract named entities from scientific text using regex patterns.

    Args:
        text: input string
        patterns: dict of {entity_type: [regex_patterns]}
    Returns:
        dict of {entity_type: [mentions]}
    """
    entities = {etype: [] for etype in patterns}
    text_lower = text.lower()

    for etype, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            entities[etype].extend([m.strip() for m in matches])

    # Deduplicate
    for etype in entities:
        entities[etype] = list(set(entities[etype]))
    return entities

def extract_claims(text):
    """Extract sentences that contain scientific claims.

    Args:
        text: abstract or paper section
    Returns:
        list of claim sentences
    """
    # Split into sentences
    sentences = re.split(r"[.!?]", text)
    claim_indicators = [
        r"\b(we show|we demonstrate|we find|we propose|we present|"
        r"our results|results indicate|results suggest|we observe|"
        r"analysis shows|our model|our approach|our method)\b"
    ]
    claims = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        for indicator in claim_indicators:
            if re.search(indicator, sent, re.IGNORECASE):
                claims.append(sent)
                break
    return claims

# -----------------------------------------------------------------
# Apply NER and claim extraction to corpus
# -----------------------------------------------------------------
all_entities = {etype: [] for etype in NER_PATTERNS}
all_claims = []

for text in df_corpus["abstract"].head(100):
    entities = scientific_ner(text)
    for etype, ents in entities.items():
        all_entities[etype].extend(ents)
    claims = extract_claims(text)
    all_claims.extend(claims)

# Entity frequency analysis
print("=== Scientific Entity Frequency (top 10) ===")
for etype, ents in all_entities.items():
    counter = Counter([e.lower() for e in ents])
    print(f"\n{etype}:")
    for ent, count in counter.most_common(5):
        print(f"  '{ent}': {count}")

print(f"\nExtracted {len(all_claims)} claim sentences from 100 abstracts")
print("Sample claims:")
for claim in all_claims[:3]:
    print(f"  - {claim[:100]}...")

# -----------------------------------------------------------------
# Visualization: Entity frequency heatmap
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
etype_list = list(NER_PATTERNS.keys())

for ax, etype in zip(axes.flatten(), etype_list):
    counter = Counter([e.lower() for e in all_entities[etype]])
    if counter:
        labels, counts = zip(*counter.most_common(8))
        ax.barh(list(labels)[::-1], list(counts)[::-1],
                color="steelblue", edgecolor="black")
        ax.set_title(f"{etype} Mentions")
        ax.set_xlabel("Count")
    else:
        ax.text(0.5, 0.5, f"No {etype} found", ha="center", va="center",
                transform=ax.transAxes)

plt.tight_layout()
plt.savefig("scientific_ner.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: scientific_ner.png")
```

## Advanced Usage

### BERTopic-Style Contextual Topic Modeling

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def bertopic_simplified(texts, n_topics=5, embedding_dim=50):
    """Simplified BERTopic pipeline without transformer models.

    Uses TF-IDF + PCA + K-Means as a lightweight substitute for
    BERT embeddings + UMAP + HDBSCAN.

    Args:
        texts: list of text strings
        n_topics: number of topics
        embedding_dim: PCA dimensions for pre-clustering
    Returns:
        topic assignments, topic keywords
    """
    # Step 1: TF-IDF document embeddings
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(texts).toarray()

    # Step 2: Dimensionality reduction (PCA as UMAP substitute)
    pca = PCA(n_components=min(embedding_dim, X.shape[1] - 1), random_state=42)
    X_reduced = pca.fit_transform(X)

    # Step 3: K-Means clustering (as HDBSCAN substitute)
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_reduced)

    # Step 4: Class-based TF-IDF for topic words (c-TF-IDF)
    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    for t in range(n_topics):
        doc_mask = labels == t
        if doc_mask.sum() == 0:
            continue
        # Mean TF-IDF in cluster
        cluster_tfidf = X[doc_mask].mean(axis=0)
        top_idx = cluster_tfidf.argsort()[-10:][::-1]
        topic_words[t] = [feature_names[i] for i in top_idx]

    return labels, topic_words

labels_bt, topic_words_bt = bertopic_simplified(
    df_corpus["clean_text"].tolist(), n_topics=5)

print("=== BERTopic-Simplified Topics ===")
for t, words in topic_words_bt.items():
    print(f"Topic {t}: {', '.join(words[:7])}")
```

### Topic Coherence Evaluation

```python
import numpy as np
from collections import defaultdict
from itertools import combinations

def compute_npmi_coherence(top_words_per_topic, corpus_texts, window=10):
    """Compute NPMI-based coherence for each topic.

    Args:
        top_words_per_topic: list of lists, each = top N words for a topic
        corpus_texts: list of preprocessed text strings
        window: co-occurrence window size (in words)
    Returns:
        list of per-topic NPMI coherence scores
    """
    # Build word co-occurrence counts
    word_count = defaultdict(int)
    pair_count = defaultdict(int)
    total_windows = 0

    for text in corpus_texts:
        words = text.split()
        for i, w in enumerate(words):
            word_count[w] += 1
            window_words = words[max(0, i-window):i+window+1]
            for w2 in window_words:
                if w2 != w:
                    pair_count[tuple(sorted([w, w2]))] += 1
        total_windows += max(1, len(words))

    # NPMI for word pairs
    def npmi(w1, w2, N):
        c1 = word_count.get(w1, 0) + 1
        c2 = word_count.get(w2, 0) + 1
        c12 = pair_count.get(tuple(sorted([w1, w2])), 0) + 1
        pmi = np.log(c12 * N / (c1 * c2 + 1e-10))
        npmi_val = pmi / max(-np.log(c12 / N + 1e-10), 1e-10)
        return npmi_val

    coherence_scores = []
    for topic_words in top_words_per_topic:
        N = max(total_windows, 1)
        pairs = list(combinations(topic_words[:10], 2))
        if not pairs:
            coherence_scores.append(0.0)
            continue
        pair_npmis = [npmi(w1, w2, N) for w1, w2 in pairs]
        coherence_scores.append(float(np.mean(pair_npmis)))

    return coherence_scores

# Get top words from NMF model (already fit above)
topic_words_nmf = []
for topic_idx in range(n_topics):
    top_idx = H_nmf[topic_idx].argsort()[-10:][::-1]
    topic_words_nmf.append([feature_names[i] for i in top_idx])

coh_scores = compute_npmi_coherence(topic_words_nmf,
                                     df_corpus["clean_text"].tolist())
print("\n=== NMF Topic Coherence (NPMI) ===")
for i, score in enumerate(coh_scores):
    print(f"Topic {i+1}: {score:.4f}")
print(f"Mean coherence: {np.mean(coh_scores):.4f}")
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| LDA gives incoherent topics | Too few documents or too many topics | Reduce n_topics; increase min_df in vectorizer |
| NMF topics overlap significantly | K too large or corpus not diverse enough | Decrease K; add orthogonality constraint |
| TF-IDF keywords are stopwords | `stop_words` not configured | Use `stop_words="english"` or custom stopword list |
| RAKE extracts long nonsense phrases | Aggressive splitting | Increase `min_chars`; add more stopwords |
| NER misses multi-word entities | Regex too strict | Add more patterns; consider spaCy + SciSpacy |
| Coherence score negative | Sparse co-occurrences | Increase corpus size or window size |

## External Resources

- Blei, D. M., et al. (2003). Latent Dirichlet Allocation. *JMLR*, 3, 993-1022.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by NMF. *Nature*.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with class-based TF-IDF. *arXiv*:2203.05794.
- [Gensim topic modeling documentation](https://radimrehurek.com/gensim/)
- [SciSpacy — NLP for biomedical text](https://allenai.github.io/scispacy/)
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly.

## Examples

### Example 1: Abstract Clustering for Literature Review

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

texts = df_corpus["clean_text"].tolist()
vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
X = vectorizer.fit_transform(texts).toarray()

# Find optimal number of clusters
silhouette_scores = []
k_range = range(3, 10)
for k in k_range:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    s = silhouette_score(X, labels, sample_size=200, random_state=42)
    silhouette_scores.append(s)

best_k = k_range[np.argmax(silhouette_scores)]
print(f"Optimal clusters: k={best_k} (silhouette={max(silhouette_scores):.3f})")

model = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
labels = model.fit_predict(X)
df_corpus["cluster"] = labels
print("\nCluster sizes:")
print(df_corpus["cluster"].value_counts().sort_index())
```

### Example 2: Research Front Detection via Temporal Topic Analysis

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def detect_emerging_topics(df, text_col="clean_text", year_col="year",
                             n_terms=20, window=3):
    """Detect emerging keywords by comparing recent vs. baseline TF-IDF.

    Args:
        df: DataFrame with text and year columns
        window: number of recent years to compare against baseline
    Returns:
        DataFrame with term emergence scores
    """
    max_yr = df[year_col].max()
    recent = df[df[year_col] >= max_yr - window + 1]
    baseline = df[df[year_col] < max_yr - window + 1]

    if len(recent) < 5 or len(baseline) < 5:
        return pd.DataFrame()

    vec = TfidfVectorizer(max_features=500, stop_words="english", min_df=2)
    vec.fit(df[text_col])
    X_recent = vec.transform(recent[text_col]).mean(axis=0).A1
    X_baseline = vec.transform(baseline[text_col]).mean(axis=0).A1

    terms = vec.get_feature_names_out()
    emergence = (X_recent - X_baseline) / (X_baseline + 1e-6)

    result = pd.DataFrame({
        "term": terms,
        "recent_tfidf": X_recent,
        "baseline_tfidf": X_baseline,
        "emergence_score": emergence,
    }).sort_values("emergence_score", ascending=False)

    return result.head(n_terms)

emerging = detect_emerging_topics(df_corpus, window=3)
print("=== Emerging Research Topics (last 3 years vs. baseline) ===")
print(emerging[["term", "emergence_score", "recent_tfidf"]].head(15).round(4)
      .to_string(index=False))
```
