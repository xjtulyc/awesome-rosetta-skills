---
name: text-as-data
description: >
  Quantitative text analysis for political science: LDA topic modeling, Wordfish
  scaling, sentiment with VADER, TF-IDF ideology classification, and NER for actors.
tags:
  - political-science
  - nlp
  - text-analysis
  - topic-modeling
  - python
  - ideology-scaling
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
  - spacy>=3.6.0
  - gensim>=4.3.0
  - vaderSentiment>=3.3.2
  - scikit-learn>=1.3.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - requests>=2.31.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# Quantitative Text Analysis for Political Science

This skill covers the principal methods for treating political texts as data:
preprocessing with spaCy, LDA topic modeling on legislative speeches and party
manifestos, Wordfish left-right scaling, VADER sentiment with custom political
dictionaries, TF-IDF ideology classification, NER for political actors, and access
to the MANIFESTO project corpus.

---

## 1. Setup

```bash
pip install spacy gensim vaderSentiment scikit-learn pandas numpy scipy requests matplotlib
python -m spacy download en_core_web_sm
# For German/EU manifesto corpora:
# python -m spacy download de_core_news_sm
```

```python
import os
import re
import numpy as np
import pandas as pd
import requests
import spacy
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
```

---

## 2. Text Preprocessing for Political Texts

```python
# Custom political stopwords (supplement spaCy's list)
POLITICAL_STOPWORDS = {
    "government", "parliament", "senator", "congressman", "bill",
    "act", "shall", "hereby", "whereas", "resolution", "amendment",
    "committee", "congress", "legislation", "member", "session",
    "house", "senate", "motion", "honorable", "distinguished",
}

def preprocess_political_text(
    text: str,
    min_token_len: int = 3,
    allowed_pos: set = ("NOUN", "VERB", "ADJ", "PROPN"),
    remove_political_stop: bool = True,
) -> list[str]:
    """
    Tokenize, lemmatize, and filter a political text using spaCy.

    Parameters
    ----------
    text : str
        Raw text (speech, manifesto paragraph, tweet).
    min_token_len : int
        Drop tokens shorter than this.
    allowed_pos : set
        Keep only these POS tags.
    remove_political_stop : bool
        Also remove domain-specific political stopwords.

    Returns
    -------
    list[str] of processed tokens.
    """
    # Normalize whitespace and remove legislative boilerplate
    text = re.sub(r"\[.*?\]", " ", text)          # remove bracketed notes
    text = re.sub(r"\b(MR|MS|MRS|DR|SEN|REP)\.", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    doc    = nlp(text[:100_000])  # spaCy default max 1M chars; clip for safety
    tokens = []

    for tok in doc:
        lemma = tok.lemma_.lower()
        if (
            not tok.is_stop
            and not tok.is_punct
            and not tok.like_num
            and tok.pos_ in allowed_pos
            and len(lemma) >= min_token_len
            and lemma.isalpha()
        ):
            if remove_political_stop and lemma in POLITICAL_STOPWORDS:
                continue
            tokens.append(lemma)

    return tokens
```

---

## 3. LDA Topic Modeling on Legislative Speeches

```python
def compute_lda_topics(
    texts: list[str],
    n_topics: int = 10,
    passes: int = 15,
    alpha: str | float = "auto",
    random_state: int = 42,
    min_df: int = 5,
    no_below: int = 5,
    no_above: float = 0.7,
) -> tuple:
    """
    Run Latent Dirichlet Allocation on a list of political texts.

    Parameters
    ----------
    texts : list[str]
        Raw text documents.
    n_topics : int
        Number of latent topics.
    passes : int
        Number of training passes over the corpus.
    alpha : str or float
        Document-topic prior: 'auto', 'symmetric', or float.
    random_state : int
        Seed for reproducibility.
    min_df : int
        Minimum document frequency for dictionary filtering.
    no_below : int
        Remove tokens appearing in fewer than this many docs.
    no_above : float
        Remove tokens appearing in more than this fraction of docs.

    Returns
    -------
    tuple: (lda_model, dictionary, corpus, token_lists)
    """
    token_lists = [preprocess_political_text(t) for t in texts]

    dictionary  = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus      = [dictionary.doc2bow(tok) for tok in token_lists]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=passes,
        alpha=alpha,
        random_state=random_state,
        per_word_topics=True,
    )

    print(f"LDA trained: {n_topics} topics, {len(dictionary)} vocab, {len(corpus)} docs")
    for i, topic in lda.print_topics(num_words=8):
        print(f"  Topic {i:02d}: {topic}")

    return lda, dictionary, corpus, token_lists


def get_document_topics(lda: LdaModel, corpus: list, n_topics: int) -> np.ndarray:
    """Return a (n_docs, n_topics) matrix of topic proportions."""
    theta = np.zeros((len(corpus), n_topics))
    for d, bow in enumerate(corpus):
        topic_dist = dict(lda.get_document_topics(bow, minimum_probability=0))
        for t, prob in topic_dist.items():
            theta[d, t] = prob
    return theta
```

---

## 4. Wordfish Scaling Model

```python
def run_wordfish(
    dtm: np.ndarray,
    n_iter: int = 200,
    tol: float = 1e-6,
    random_state: int = 0,
) -> dict:
    """
    Estimate the Wordfish (Slapin & Proksch, 2008) model for
    scaling texts on a latent ideological dimension.

    Model: E[y_ijt] = exp(alpha_i + psi_j + beta_j * omega_i)
    where omega_i is the document (party) position on the latent scale.

    Parameters
    ----------
    dtm : np.ndarray
        Document-term matrix (n_docs x n_terms), raw counts.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.

    Returns
    -------
    dict with keys:
        omega  — document positions (n_docs,)
        alpha  — document fixed effects (n_docs,)
        psi    — word fixed effects (n_terms,)
        beta   — word discrimination parameters (n_terms,)
        ll_history — log-likelihood trace
    """
    np.random.seed(random_state)
    n_docs, n_terms = dtm.shape
    Y = dtm.astype(float)

    # Initialise parameters
    omega  = np.random.normal(0, 1, n_docs)
    omega  = (omega - omega.mean()) / omega.std()     # identify: mean=0, sd=1
    alpha  = np.log(Y.sum(axis=1) / Y.sum() + 1e-9)
    psi    = np.log(Y.sum(axis=0) / Y.sum() + 1e-9)
    beta   = np.random.normal(0, 0.1, n_terms)

    ll_history = []

    for iteration in range(n_iter):
        # E[y] under current params
        mu = np.exp(
            alpha[:, None] + psi[None, :] + np.outer(omega, beta)
        )
        ll = (Y * np.log(mu + 1e-300) - mu).sum()
        ll_history.append(ll)

        # Update alpha (document fixed effects)
        alpha = np.log(Y.sum(axis=1)) - np.log(
            np.exp(psi[None, :] + np.outer(omega, beta)).sum(axis=1) + 1e-9
        )

        # Update psi (word fixed effects)
        psi = np.log(Y.sum(axis=0)) - np.log(
            np.exp(alpha[:, None] + np.outer(omega, beta)).sum(axis=0) + 1e-9
        )

        # Update beta (word discrimination) via gradient step
        for j in range(n_terms):
            mu_j = np.exp(alpha + psi[j] + beta[j] * omega)
            grad = (Y[:, j] - mu_j) @ omega
            hess = -(mu_j * omega**2).sum()
            if abs(hess) > 1e-12:
                beta[j] -= grad / hess

        # Update omega (document positions) via gradient step
        for i in range(n_docs):
            mu_i = np.exp(alpha[i] + psi + beta * omega[i])
            grad = ((Y[i] - mu_i) * beta).sum()
            hess = -(mu_i * beta**2).sum()
            if abs(hess) > 1e-12:
                omega[i] -= grad / hess

        # Re-identify: normalise omega
        omega = (omega - omega.mean()) / (omega.std() + 1e-9)

        if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            print(f"Wordfish converged at iteration {iteration}")
            break

    return {
        "omega":      omega,
        "alpha":      alpha,
        "psi":        psi,
        "beta":       beta,
        "ll_history": ll_history,
    }
```

---

## 5. VADER and Custom Political Dictionary Sentiment

```python
# Example political sentiment dictionary (extend with domain lexicon)
POLITICAL_SENTIMENT = {
    "positive": ["reform", "progress", "growth", "freedom", "prosperity",
                 "invest", "opportunity", "innovation", "secure", "protect"],
    "negative": ["crisis", "corruption", "decline", "threat", "failure",
                 "poverty", "violence", "exploitation", "injustice", "deficit"],
}

def apply_dictionary_sentiment(
    texts: list[str],
    dictionary: dict | None = None,
    use_vader: bool = True,
) -> pd.DataFrame:
    """
    Score texts using VADER and/or a custom political dictionary.

    Parameters
    ----------
    texts : list[str]
        Input documents.
    dictionary : dict or None
        {'positive': [words], 'negative': [words]}.
        If None, uses POLITICAL_SENTIMENT defined above.
    use_vader : bool
        Also compute VADER compound score.

    Returns
    -------
    pd.DataFrame with columns: vader_compound, dict_pos, dict_neg, dict_net.
    """
    if dictionary is None:
        dictionary = POLITICAL_SENTIMENT

    vader   = SentimentIntensityAnalyzer() if use_vader else None
    results = []

    pos_words = set(dictionary.get("positive", []))
    neg_words = set(dictionary.get("negative", []))

    for text in texts:
        tokens = preprocess_political_text(text, allowed_pos={"NOUN", "VERB", "ADJ", "PROPN", "ADV"})
        n      = max(len(tokens), 1)

        pos_count = sum(1 for t in tokens if t in pos_words)
        neg_count = sum(1 for t in tokens if t in neg_words)

        row = {
            "dict_pos": pos_count / n,
            "dict_neg": neg_count / n,
            "dict_net": (pos_count - neg_count) / n,
        }
        if use_vader:
            row["vader_compound"] = vader.polarity_scores(text[:512])["compound"]

        results.append(row)

    return pd.DataFrame(results)
```

---

## 6. Named Entity Recognition for Political Actors

```python
def extract_political_actors(
    text: str,
    target_labels: tuple = ("PERSON", "ORG", "GPE", "NORP"),
) -> pd.DataFrame:
    """
    Extract and count named entities relevant to political analysis.

    Parameters
    ----------
    text : str
        Political speech or news article.
    target_labels : tuple
        spaCy NER labels to extract.
        PERSON = politicians, ORG = parties/agencies, GPE = countries,
        NORP = nationalities/groups.

    Returns
    -------
    pd.DataFrame with columns: entity, label, count.
    """
    doc     = nlp(text[:100_000])
    counts  = {}

    for ent in doc.ents:
        if ent.label_ in target_labels:
            key = (ent.text.strip(), ent.label_)
            counts[key] = counts.get(key, 0) + 1

    rows = [{"entity": e, "label": l, "count": c} for (e, l), c in counts.items()]
    df   = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df
```

---

## 7. TF-IDF Ideology Classification

```python
def classify_ideology_tfidf(
    texts: list[str],
    labels: list[str],
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    cv: int = 5,
) -> dict:
    """
    Train a TF-IDF + Logistic Regression ideology classifier.

    Parameters
    ----------
    texts : list[str]
        Party manifesto paragraphs or speeches.
    labels : list[str]
        Ideological labels, e.g. 'left', 'center', 'right'.
    max_features : int
        Vocabulary size cap for TF-IDF.
    ngram_range : tuple
        Unigrams + bigrams by default.
    cv : int
        Cross-validation folds.

    Returns
    -------
    dict: accuracy scores and fitted model components.
    """
    le     = LabelEncoder()
    y      = le.fit_transform(labels)

    vec    = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )
    X      = vec.fit_transform(texts)

    clf    = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                 multi_class="multinomial")
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    clf.fit(X, y)

    # Top discriminating features per class
    feature_names = np.array(vec.get_feature_names_out())
    print(f"\nCross-val accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    for i, cls in enumerate(le.classes_):
        top_idx = np.argsort(clf.coef_[i])[-10:]
        print(f"  [{cls}] top features: {', '.join(feature_names[top_idx])}")

    return {
        "vectorizer":   vec,
        "classifier":   clf,
        "label_encoder": le,
        "cv_accuracy":  scores.mean(),
        "cv_std":       scores.std(),
    }
```

---

## 8. MANIFESTO Project API Access

```python
def fetch_manifesto_corpus(
    party_codes: list[int],
    election_dates: list[str],
    api_key_env: str = "MANIFESTO_API_KEY",
) -> pd.DataFrame:
    """
    Fetch annotated manifesto texts from the MANIFESTO Project API.
    Register at https://manifestoproject.wzb.eu/ to obtain a free API key.

    Parameters
    ----------
    party_codes : list[int]
        MANIFESTO party codes, e.g. [41320] for UK Labour.
    election_dates : list[str]
        Election dates in 'YYYYMM' format, e.g. ['201705', '201912'].
    api_key_env : str
        Name of environment variable holding the API key.

    Returns
    -------
    pd.DataFrame with columns: party, date, text, cmp_code (annotation).
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Set the {api_key_env} environment variable with your MANIFESTO API key."
        )

    BASE_URL = "https://manifesto-project.wzb.eu/api/v1"
    rows = []

    for party in party_codes:
        for date in election_dates:
            url    = f"{BASE_URL}/texts_and_annotations"
            params = {
                "api_key":    api_key,
                "keys[]":     f"{party}_{date}",
                "format":     "json",
            }
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get("items", []):
                    for annotation in item.get("annotations", {}).get("content", []):
                        rows.append({
                            "party":    party,
                            "date":     date,
                            "text":     annotation.get("text", ""),
                            "cmp_code": annotation.get("cmp_code", ""),
                        })
            except requests.RequestException as exc:
                print(f"Warning: Could not fetch {party}_{date}: {exc}")

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} annotated manifesto quasi-sentences.")
    return df
```

---

## 9. Partisan Topic Divergence

```python
def detect_partisan_divergence(
    topic_distributions_a: np.ndarray,
    topic_distributions_b: np.ndarray,
    topic_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Measure Jensen-Shannon divergence per topic between two partisan corpora.

    Parameters
    ----------
    topic_distributions_a : np.ndarray
        (n_docs_a, n_topics) topic proportion matrix for party A.
    topic_distributions_b : np.ndarray
        (n_docs_b, n_topics) topic proportion matrix for party B.
    topic_labels : list[str] or None
        Human-readable topic names.

    Returns
    -------
    pd.DataFrame sorted by divergence (highest first).
    """
    from scipy.special import rel_entr

    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        m = 0.5 * (p + q)
        return float(0.5 * rel_entr(p, m + 1e-12).sum()
                     + 0.5 * rel_entr(q, m + 1e-12).sum())

    mean_a = topic_distributions_a.mean(axis=0)
    mean_b = topic_distributions_b.mean(axis=0)
    n_topics = len(mean_a)

    rows = []
    for t in range(n_topics):
        label = topic_labels[t] if topic_labels else f"Topic_{t}"
        rows.append({
            "topic":    label,
            "mean_a":   round(mean_a[t], 4),
            "mean_b":   round(mean_b[t], 4),
            "diff_a_minus_b": round(mean_a[t] - mean_b[t], 4),
            "js_divergence":  round(js_divergence(
                np.array([mean_a[t], 1 - mean_a[t]]),
                np.array([mean_b[t], 1 - mean_b[t]]),
            ), 6),
        })

    return pd.DataFrame(rows).sort_values("js_divergence", ascending=False).reset_index(drop=True)
```

---

## 10. Example A — Scale UK Party Manifestos on Left-Right Dimension

```python
import numpy as np
import pandas as pd

# ---- Synthetic manifesto paragraphs (replace with MANIFESTO corpus) -------------
np.random.seed(0)

party_texts = {
    "Labour": [
        "We will invest in public services, raise wages, and tax corporations fairly.",
        "Our plan expands the NHS, funds affordable housing, and strengthens trade unions.",
        "Inequality must be tackled through redistribution and workers' rights protection.",
    ] * 20,
    "Conservative": [
        "Lower taxes and deregulation will unleash private sector growth and enterprise.",
        "We champion personal responsibility, free markets, and fiscal discipline.",
        "Strong borders, sound money, and traditional values underpin our programme.",
    ] * 20,
    "LibDem": [
        "We support proportional representation, civil liberties, and green investment.",
        "A fair economy requires open markets, strong rights, and ecological sustainability.",
        "Evidence-based policy and constitutional reform are central to our platform.",
    ] * 20,
}

all_texts  = []
doc_labels = []
for party, paras in party_texts.items():
    all_texts.extend(paras)
    doc_labels.extend([party] * len(paras))

# ---- Build document-term matrix using TF-IDF (non-negative, rounded to counts) ---
vec = TfidfVectorizer(max_features=300, ngram_range=(1, 1), sublinear_tf=False, min_df=2)
dtm_tfidf = vec.fit_transform(all_texts).toarray()
# Convert to pseudo-counts for Wordfish Poisson model
dtm_counts = np.round(dtm_tfidf * 100).astype(int)

# ---- Wordfish scaling -------------------------------------------------------
wf = run_wordfish(dtm_counts, n_topics=20, n_iter=100)

# ---- Aggregate position by party -------------------------------------------
positions = pd.DataFrame({"omega": wf["omega"], "party": doc_labels})
party_pos = positions.groupby("party")["omega"].agg(["mean", "std"]).sort_values("mean")
print("\n=== Wordfish Party Positions (left-right) ===")
print(party_pos.to_string())

# ---- Plot ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
for i, (party, row) in enumerate(party_pos.iterrows()):
    ax.errorbar(row["mean"], i, xerr=row["std"], fmt="o", capsize=4,
                label=party, markersize=8)
ax.set_yticks(range(len(party_pos)))
ax.set_yticklabels(party_pos.index)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Wordfish Score (negative=left, positive=right)")
ax.set_title("UK Party Manifesto Positions — Wordfish Scaling", fontweight="bold")
plt.tight_layout()
plt.savefig("wordfish_uk_parties.png", dpi=150)
plt.show()
```

---

## 11. Example B — Detect Partisan Topic Divergence in Congressional Speeches

```python
import numpy as np
import pandas as pd

# ---- Synthetic Congressional Record (replace with Gentzkow et al. corpus) --------
np.random.seed(42)

n_per_party = 150
dem_speeches = [
    "Healthcare reform protects working families from insurance company exploitation.",
    "We must expand Medicaid and lower drug prices for all Americans.",
    "Climate change threatens our future; we need clean energy investment now.",
    "Voting rights must be protected against voter suppression tactics.",
    "Gun violence is a public health crisis demanding sensible regulation.",
] * (n_per_party // 5)

rep_speeches = [
    "Lower taxes and regulatory relief unleash American economic freedom.",
    "We oppose government mandates on private business and personal liberty.",
    "Energy independence requires expanding domestic fossil fuel production.",
    "Border security protects American workers and upholds the rule of law.",
    "Second Amendment rights are non-negotiable constitutional freedoms.",
] * (n_per_party // 5)

all_speeches = dem_speeches + rep_speeches
party_labels = ["Democrat"] * len(dem_speeches) + ["Republican"] * len(rep_speeches)

# ---- LDA topic model --------------------------------------------------------
lda, dct, corpus, _ = compute_lda_topics(all_speeches, n_topics=5, passes=10)
theta = get_document_topics(lda, corpus, n_topics=5)

# ---- Split by party ---------------------------------------------------------
n_dem = len(dem_speeches)
theta_dem = theta[:n_dem]
theta_rep = theta[n_dem:]

topic_labels = [f"Topic_{i}" for i in range(5)]
divergence_df = detect_partisan_divergence(theta_dem, theta_rep, topic_labels)

print("\n=== Partisan Topic Divergence (Democrat vs Republican) ===")
print(divergence_df.to_string(index=False))

# ---- Sentiment trajectory over simulated years 2000-2020 --------------------
years      = list(range(2000, 2021))
sent_rows  = []
for yr in years:
    yr_texts = (dem_speeches[:10] + rep_speeches[:10])  # placeholder
    sent_df  = apply_dictionary_sentiment(yr_texts)
    sent_rows.append({
        "year":   yr,
        "dem_net": sent_df.iloc[:10]["dict_net"].mean(),
        "rep_net": sent_df.iloc[10:]["dict_net"].mean(),
    })

sent_yr = pd.DataFrame(sent_rows)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sent_yr["year"], sent_yr["dem_net"], label="Democrat", color="steelblue", linewidth=2)
ax.plot(sent_yr["year"], sent_yr["rep_net"], label="Republican", color="tomato",   linewidth=2)
ax.axhline(0, color="black", linewidth=0.7)
ax.set_xlabel("Year")
ax.set_ylabel("Net Sentiment (dictionary)")
ax.set_title("Partisan Sentiment in Congressional Speeches 2000-2020", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("partisan_sentiment.png", dpi=150)
plt.show()

# ---- Ideology classification using TF-IDF -----------------------------------
clf_result = classify_ideology_tfidf(
    all_speeches[:100],
    party_labels[:100],
    max_features=1000,
    cv=5,
)
print(f"\nIdeology classifier accuracy: {clf_result['cv_accuracy']:.3f}")
```

---

## 12. Tips and Common Pitfalls

- **Wordfish identification**: The model requires two normalisation constraints
  (mean=0, sd=1 for omega) to be identified. Some implementations also fix the
  positions of two anchor documents — use this when you have strong priors about
  the ideological extremes.
- **LDA coherence**: Use `gensim.models.CoherenceModel` with `coherence='c_v'`
  to select the number of topics. Avoid selecting K by held-out perplexity alone,
  which often favours too many topics.
- **VADER calibration**: VADER was trained on social media; its compound scores
  underestimate negativity in formal legislative language. Augment with the Lexicoder
  Sentiment Dictionary (Young & Soroka, 2012) for political texts.
- **MANIFESTO API rate limits**: The free tier allows 1000 requests/day. Cache
  responses to disk using `requests_cache` to avoid re-downloading.
- **TF-IDF vs bag-of-words for ideology**: TF-IDF with sublinear scaling
  (`sublinear_tf=True`) tends to outperform raw counts for short political texts.
  For longer corpora, consider fastText or transformer embeddings.
- **Cross-lingual analysis**: For multilingual manifesto corpora, use
  `spacy-transformers` with `xlm-roberta-base` or the `sentence-transformers`
  `paraphrase-multilingual-mpnet-base-v2` model.
