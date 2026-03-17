---
name: computational-sociology
description: >
  Computational social science methods: social media data collection, bot detection,
  network homophily, echo chamber analysis, and conjoint survey experiments.
tags:
  - sociology
  - social-networks
  - twitter
  - reddit
  - computational-social-science
  - survey-experiments
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
  - tweepy>=4.14.0
  - praw>=7.7.0
  - vaderSentiment>=3.3.1
  - networkx>=3.1
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.11.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - statsmodels>=0.14.0
last_updated: "2026-03-17"
---

# Computational Sociology

Computational social science methods for studying online behavior, social networks,
political polarization, and survey-based causal inference. This skill covers data
collection from major social media platforms, automated account detection, network
analysis metrics, and experimental designs for measuring social preferences.

## Prerequisites

```bash
pip install tweepy praw vaderSentiment networkx pandas numpy scipy \
            matplotlib seaborn statsmodels
```

Set the required environment variables before running any data collection:

```bash
export TWITTER_BEARER_TOKEN="<paste-your-bearer-token>"
export REDDIT_CLIENT_ID="<paste-your-client-id>"
export REDDIT_CLIENT_SECRET="<paste-your-client-secret>"
export REDDIT_USER_AGENT="ComputationalSociologyBot/1.0"
```

## Core Functions

### 1. Twitter/X Academic API v2 Data Collection

```python
import os
import time
import tweepy
import pandas as pd
from datetime import datetime, timezone


def collect_tweets(
    query: str,
    start_time: str,
    end_time: str,
    bearer_token_env: str = "TWITTER_BEARER_TOKEN",
    max_results_per_page: int = 100,
    max_total: int = 5000,
    include_fields: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collect tweets from Twitter/X Academic API v2 using tweepy paginator.

    Parameters
    ----------
    query : str
        Twitter search query string (supports operators like lang:en, -is:retweet).
    start_time : str
        ISO 8601 start datetime, e.g. "2023-01-01T00:00:00Z".
    end_time : str
        ISO 8601 end datetime, e.g. "2023-12-31T23:59:59Z".
    bearer_token_env : str
        Name of the environment variable holding the bearer token.
    max_results_per_page : int
        Results per API call (10-100 for basic, up to 500 for academic).
    max_total : int
        Maximum total tweets to retrieve.
    include_fields : list[str] | None
        Extra tweet fields to request.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, text, created_at, author_id,
        retweet_count, like_count, reply_count, lang, possibly_sensitive.
    """
    bearer_token = os.environ.get(bearer_token_env)
    if not bearer_token:
        raise EnvironmentError(
            f"Environment variable '{bearer_token_env}' is not set. "
            "Obtain a bearer token from developer.twitter.com and export it."
        )

    client = tweepy.Client(
        bearer_token=bearer_token,
        wait_on_rate_limit=True,
    )

    tweet_fields = [
        "id", "text", "created_at", "author_id", "lang",
        "public_metrics", "possibly_sensitive", "entities",
        "referenced_tweets",
    ]
    if include_fields:
        tweet_fields = list(set(tweet_fields + include_fields))

    user_fields = [
        "id", "name", "username", "created_at", "public_metrics",
        "description", "verified", "location",
    ]

    records = []
    paginator = tweepy.Paginator(
        client.search_all_tweets,
        query=query,
        start_time=start_time,
        end_time=end_time,
        tweet_fields=tweet_fields,
        user_fields=user_fields,
        expansions=["author_id"],
        max_results=max_results_per_page,
    )

    users_lookup: dict[str, dict] = {}

    for response in paginator:
        if response.includes and "users" in response.includes:
            for u in response.includes["users"]:
                users_lookup[str(u.id)] = {
                    "username": u.username,
                    "user_created_at": u.created_at,
                    "followers_count": u.public_metrics.get("followers_count", 0),
                    "following_count": u.public_metrics.get("following_count", 0),
                    "tweet_count": u.public_metrics.get("tweet_count", 0),
                    "verified": u.verified,
                }

        if response.data:
            for tweet in response.data:
                m = tweet.public_metrics or {}
                user_meta = users_lookup.get(str(tweet.author_id), {})
                records.append({
                    "id": str(tweet.id),
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "author_id": str(tweet.author_id),
                    "lang": tweet.lang,
                    "retweet_count": m.get("retweet_count", 0),
                    "like_count": m.get("like_count", 0),
                    "reply_count": m.get("reply_count", 0),
                    "possibly_sensitive": tweet.possibly_sensitive,
                    **user_meta,
                })

        if len(records) >= max_total:
            break
        time.sleep(0.5)

    df = pd.DataFrame(records).drop_duplicates(subset=["id"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df.head(max_total)


def collect_reddit_posts(
    subreddits: list[str],
    limit_per_sub: int = 500,
    sort: str = "new",
) -> pd.DataFrame:
    """
    Collect posts from Reddit using PRAW.

    Parameters
    ----------
    subreddits : list[str]
        List of subreddit names (without r/).
    limit_per_sub : int
        Maximum posts per subreddit.
    sort : str
        Sorting method: 'new', 'hot', 'top', 'rising'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, subreddit, title, selftext, score,
        num_comments, upvote_ratio, author, created_utc, url.
    """
    import praw

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "ComputationalSociologyBot/1.0"),
    )

    records = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        fetcher = {
            "new": subreddit.new,
            "hot": subreddit.hot,
            "top": subreddit.top,
            "rising": subreddit.rising,
        }.get(sort, subreddit.new)

        for post in fetcher(limit=limit_per_sub):
            records.append({
                "id": post.id,
                "subreddit": sub_name,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "num_comments": post.num_comments,
                "upvote_ratio": post.upvote_ratio,
                "author": str(post.author) if post.author else "[deleted]",
                "created_utc": datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                "url": post.url,
                "is_self": post.is_self,
            })

    return pd.DataFrame(records)
```

### 2. Text Preprocessing and VADER Sentiment

```python
import re
import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def preprocess_social_text(text: str, platform: str = "twitter") -> str:
    """
    Clean and normalize social media text for NLP tasks.
    """
    text = text.lower()
    if platform == "twitter":
        text = re.sub(r"http\S+|www\.\S+", " URL ", text)
        text = re.sub(r"@\w+", " USER ", text)
        text = re.sub(r"#(\w+)", r" \1 ", text)
        text = re.sub(r"rt\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score_vader_sentiment(texts: list[str]) -> pd.DataFrame:
    """
    Apply VADER sentiment analysis to a list of social media texts.

    Returns DataFrame with compound, positive, neutral, negative scores
    and a categorical sentiment label.
    """
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        scores = analyzer.polarity_scores(text)
        label = (
            "positive" if scores["compound"] >= 0.05
            else "negative" if scores["compound"] <= -0.05
            else "neutral"
        )
        results.append({**scores, "sentiment_label": label, "text": text})
    return pd.DataFrame(results)
```

### 3. Bot Detection

```python
import numpy as np


def detect_bots(
    user_df: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Heuristic bot detection based on account-level features.

    Parameters
    ----------
    user_df : pd.DataFrame
        Must contain: username, user_created_at, followers_count,
        following_count, tweet_count, verified (bool),
        and optionally: avg_daily_tweets, unique_hashtag_ratio,
        url_ratio, description_length.
    thresholds : dict | None
        Override default heuristic thresholds.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns: bot_score (0–1),
        bot_flag (bool), and individual heuristic flags.
    """
    defaults = {
        "min_account_age_days": 30,
        "max_follower_following_ratio": 0.05,  # very few followers vs following
        "max_daily_tweets": 72,               # more than 3/hr on average
        "min_content_diversity": 0.1,         # unique hashtag ratio
        "min_description_length": 10,
    }
    if thresholds:
        defaults.update(thresholds)

    df = user_df.copy()
    now = pd.Timestamp.now(tz="UTC")

    # Account age
    df["user_created_at"] = pd.to_datetime(df["user_created_at"], utc=True)
    df["account_age_days"] = (now - df["user_created_at"]).dt.days
    df["flag_new_account"] = df["account_age_days"] < defaults["min_account_age_days"]

    # Follower/following ratio
    df["follower_ratio"] = df["followers_count"] / (df["following_count"] + 1)
    df["flag_low_follower_ratio"] = df["follower_ratio"] < defaults["max_follower_following_ratio"]

    # Posting frequency
    if "avg_daily_tweets" not in df.columns:
        df["avg_daily_tweets"] = df["tweet_count"] / (df["account_age_days"] + 1)
    df["flag_high_frequency"] = df["avg_daily_tweets"] > defaults["max_daily_tweets"]

    # Content diversity
    if "unique_hashtag_ratio" in df.columns:
        df["flag_low_diversity"] = df["unique_hashtag_ratio"] < defaults["min_content_diversity"]
    else:
        df["flag_low_diversity"] = False

    # Missing description
    if "description_length" in df.columns:
        df["flag_no_description"] = df["description_length"] < defaults["min_description_length"]
    else:
        df["flag_no_description"] = False

    # Verified accounts are very unlikely to be bots
    flag_cols = [
        "flag_new_account", "flag_low_follower_ratio",
        "flag_high_frequency", "flag_low_diversity", "flag_no_description",
    ]
    df["bot_score"] = df[flag_cols].sum(axis=1) / len(flag_cols)
    df.loc[df["verified"] == True, "bot_score"] = 0.0
    df["bot_flag"] = df["bot_score"] >= 0.6

    return df
```

### 4. Network Homophily and Echo Chambers

```python
import networkx as nx
from scipy import stats


def measure_homophily(G: nx.Graph, attribute: str) -> dict:
    """
    Measure homophily in a social network using assortativity coefficient.

    Parameters
    ----------
    G : nx.Graph
        NetworkX graph with node attribute `attribute` set on each node.
    attribute : str
        Node attribute name (e.g., 'party', 'ideology_bin').

    Returns
    -------
    dict
        Dictionary with assortativity_coefficient (r), p_value (approximate
        via permutation), and a homophily_label string.
    """
    r = nx.attribute_assortativity_coefficient(G, attribute)

    # Permutation test: shuffle attributes, recompute
    attrs = [G.nodes[n][attribute] for n in G.nodes()]
    n_perms = 1000
    null_dist = []
    nodes = list(G.nodes())
    for _ in range(n_perms):
        shuffled = dict(zip(nodes, np.random.permutation(attrs)))
        H = G.copy()
        nx.set_node_attributes(H, shuffled, attribute)
        null_dist.append(nx.attribute_assortativity_coefficient(H, attribute))

    p_value = np.mean(np.array(null_dist) >= r)
    label = (
        "strong homophily" if r > 0.3
        else "moderate homophily" if r > 0.1
        else "no meaningful homophily" if r >= -0.1
        else "heterophily"
    )
    return {
        "assortativity_coefficient": r,
        "p_value": p_value,
        "homophily_label": label,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }


def echo_chamber_score(
    G: nx.Graph,
    opinion_col: str,
    n_walks: int = 1000,
    walk_length: int = 20,
    seed: int = 42,
) -> dict:
    """
    Estimate echo chamber strength via random-walk-based segregation score.

    A random walker starting from a node with opinion X is more likely to
    stay in opinion-X neighborhoods in a segregated (echo chamber) network.
    Score ranges from 0 (fully integrated) to 1 (fully segregated).

    Parameters
    ----------
    G : nx.Graph
        Graph with node attribute `opinion_col` set (binary or categorical).
    opinion_col : str
        Node attribute name containing opinion/ideology labels.
    n_walks : int
        Number of random walks to simulate.
    walk_length : int
        Steps per random walk.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        segregation_score, within_group_stay_rate, between_group_cross_rate.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    opinions = nx.get_node_attributes(G, opinion_col)
    unique_ops = list(set(opinions.values()))

    within_stays = 0
    total_steps = 0

    for _ in range(n_walks):
        start = rng.choice(nodes)
        current = start
        start_opinion = opinions.get(current)

        for _ in range(walk_length):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            nxt = rng.choice(neighbors)
            total_steps += 1
            if opinions.get(nxt) == start_opinion:
                within_stays += 1
            current = nxt

    if total_steps == 0:
        return {"segregation_score": 0.0, "within_group_stay_rate": 0.0}

    within_rate = within_stays / total_steps

    # Baseline: expected within-group rate under random mixing
    opinion_counts = pd.Series(list(opinions.values())).value_counts(normalize=True)
    baseline = (opinion_counts**2).sum()

    segregation_score = (within_rate - float(baseline)) / (1.0 - float(baseline) + 1e-9)
    segregation_score = float(np.clip(segregation_score, 0.0, 1.0))

    return {
        "segregation_score": segregation_score,
        "within_group_stay_rate": within_rate,
        "baseline_random_mixing": float(baseline),
        "n_walks": n_walks,
        "walk_length": walk_length,
    }
```

### 5. Conjoint Survey Experiment Analysis

```python
import statsmodels.formula.api as smf
from itertools import combinations


def analyze_conjoint_amce(
    df: pd.DataFrame,
    outcome: str,
    treatment_cols: list[str],
    respondent_id_col: str = "respondent_id",
    profile_id_col: str = "profile_id",
    weights_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute Average Marginal Component Effects (AMCEs) for a conjoint experiment.

    AMCEs measure the average effect of each attribute level (relative to a
    baseline) on the probability of choosing a profile, marginalizing over
    all other attribute combinations.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data: one row per profile shown to a respondent.
        Must contain `outcome` (0/1 choice), treatment columns, and IDs.
    outcome : str
        Binary outcome column (1 = chosen, 0 = not chosen).
    treatment_cols : list[str]
        List of conjoint attribute column names.
    respondent_id_col : str
        Column identifying unique respondents (used for clustering SEs).
    profile_id_col : str
        Column identifying profiles within a task.
    weights_col : str | None
        Optional survey weight column.

    Returns
    -------
    pd.DataFrame
        AMCE table with columns: attribute, level, amce, std_error,
        ci_lower, ci_upper, p_value, baseline_level.
    """
    results = []

    for col in treatment_cols:
        # Ensure string/categorical
        df[col] = df[col].astype(str)
        levels = sorted(df[col].unique())
        baseline = levels[0]

        # Dummy encode relative to baseline
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        baseline_col = f"{col}_{baseline}"
        dummy_cols = [c for c in dummies.columns if c != baseline_col]
        temp_df = pd.concat([df[[outcome, respondent_id_col]], dummies[dummy_cols]], axis=1)

        formula = f"{outcome} ~ " + " + ".join(dummy_cols)

        try:
            if weights_col and weights_col in df.columns:
                model = smf.wls(
                    formula, data=temp_df, weights=df[weights_col]
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": temp_df[respondent_id_col]},
                )
            else:
                model = smf.ols(formula, data=temp_df).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": temp_df[respondent_id_col]},
                )

            for level in levels[1:]:
                param_name = f"{col}_{level}"
                if param_name in model.params:
                    coef = model.params[param_name]
                    se = model.bse[param_name]
                    ci = model.conf_int().loc[param_name]
                    pval = model.pvalues[param_name]
                    results.append({
                        "attribute": col,
                        "level": level,
                        "baseline_level": baseline,
                        "amce": coef,
                        "std_error": se,
                        "ci_lower": ci[0],
                        "ci_upper": ci[1],
                        "p_value": pval,
                    })
        except Exception as e:
            print(f"Warning: could not fit model for attribute '{col}': {e}")

    return pd.DataFrame(results)


def compute_marginal_means(
    df: pd.DataFrame,
    outcome: str,
    attribute: str,
) -> pd.DataFrame:
    """
    Compute marginal means for a conjoint attribute.
    Marginal means are the average outcome for each level, averaging over
    all other attribute combinations.
    """
    df = df.copy()
    df[attribute] = df[attribute].astype(str)
    mm = (
        df.groupby(attribute)[outcome]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"mean": "marginal_mean", "sem": "std_error", "count": "n"})
    )
    mm["ci_lower"] = mm["marginal_mean"] - 1.96 * mm["std_error"]
    mm["ci_upper"] = mm["marginal_mean"] + 1.96 * mm["std_error"]
    return mm
```

## Example 1: Ideological Polarization in a Political Twitter Network

This example collects tweets about a political topic, infers ideology from
retweet networks, and measures homophily and echo chamber strength.

```python
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ------------------------------------------------------------------
# Step 1: Collect tweets
# ------------------------------------------------------------------
query = (
    "(#BidenBudget OR #TrumpTax) lang:en -is:retweet "
    "has:mentions -is:nullcast"
)

tweets_df = collect_tweets(
    query=query,
    start_time="2024-01-01T00:00:00Z",
    end_time="2024-03-01T00:00:00Z",
    bearer_token_env="TWITTER_BEARER_TOKEN",
    max_total=3000,
)
print(f"Collected {len(tweets_df)} tweets from {tweets_df['author_id'].nunique()} users")

# ------------------------------------------------------------------
# Step 2: Sentiment analysis
# ------------------------------------------------------------------
tweets_df["clean_text"] = tweets_df["text"].apply(
    lambda t: preprocess_social_text(t, platform="twitter")
)
sentiment_df = score_vader_sentiment(tweets_df["clean_text"].tolist())
tweets_df = pd.concat(
    [tweets_df.reset_index(drop=True), sentiment_df[["compound", "sentiment_label"]]], axis=1
)

# ------------------------------------------------------------------
# Step 3: Bot filtering
# ------------------------------------------------------------------
user_cols = [
    "author_id", "username", "user_created_at",
    "followers_count", "following_count", "tweet_count", "verified",
]
user_df = tweets_df[user_cols].drop_duplicates("author_id").copy()
user_df = detect_bots(user_df)
human_ids = set(user_df.loc[~user_df["bot_flag"], "author_id"])
tweets_clean = tweets_df[tweets_df["author_id"].isin(human_ids)].copy()
print(f"After bot removal: {len(tweets_clean)} tweets from {len(human_ids)} users")

# ------------------------------------------------------------------
# Step 4: Build retweet network and assign ideology from hashtags
# ------------------------------------------------------------------
G = nx.DiGraph()
ideology_map: dict[str, str] = {}

LEFT_HASHTAGS = {"bidenbullet", "democraticparty", "progressives", "vote blue"}
RIGHT_HASHTAGS = {"trumptax", "maga", "republicanparty", "freemarket"}

for _, row in tweets_clean.iterrows():
    text_lower = row["clean_text"]
    if any(h in text_lower for h in LEFT_HASHTAGS):
        ideology_map[row["author_id"]] = "left"
    elif any(h in text_lower for h in RIGHT_HASHTAGS):
        ideology_map[row["author_id"]] = "right"

# Add nodes with ideology attribute
for uid, ideology in ideology_map.items():
    G.add_node(uid, ideology=ideology)

# Simulate edges: co-mention connections (simplified)
for _, row in tweets_clean.iterrows():
    src = row["author_id"]
    if src in ideology_map and row["retweet_count"] > 0:
        G.add_node(src, ideology=ideology_map.get(src, "unknown"))

# For demonstration, create edges from users who share tweets
user_list = list(ideology_map.keys())
for i, u in enumerate(user_list[:200]):
    for v in user_list[i+1:min(i+5, len(user_list))]:
        if ideology_map.get(u) == ideology_map.get(v):
            G.add_edge(u, v)
        elif pd.Series([1]).sample().item() > 0.7:  # sparse cross-links
            G.add_edge(u, v)

G_undirected = G.to_undirected()

# ------------------------------------------------------------------
# Step 5: Measure homophily and echo chambers
# ------------------------------------------------------------------
homophily = measure_homophily(G_undirected, attribute="ideology")
echo = echo_chamber_score(G_undirected, opinion_col="ideology", n_walks=2000)

print("\n=== Ideological Network Analysis ===")
print(f"Assortativity (homophily r): {homophily['assortativity_coefficient']:.3f}")
print(f"Homophily label: {homophily['homophily_label']}")
print(f"Permutation p-value: {homophily['p_value']:.4f}")
print(f"Echo chamber segregation score: {echo['segregation_score']:.3f}")
print(f"Within-group stay rate: {echo['within_group_stay_rate']:.3f}")
print(f"Baseline (random mixing): {echo['baseline_random_mixing']:.3f}")

# ------------------------------------------------------------------
# Step 6: Visualize
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

color_map = {"left": "#1f77b4", "right": "#d62728", "unknown": "#aec7e8"}
node_colors = [color_map.get(G_undirected.nodes[n].get("ideology", "unknown"), "grey")
               for n in G_undirected.nodes()]

pos = nx.spring_layout(G_undirected, seed=42, k=0.3)
nx.draw_networkx(
    G_undirected, pos, ax=axes[0],
    node_color=node_colors, node_size=30,
    edge_color="grey", alpha=0.6, with_labels=False,
)
axes[0].set_title(
    f"Political Retweet Network\nr={homophily['assortativity_coefficient']:.2f}, "
    f"echo={echo['segregation_score']:.2f}"
)
axes[0].axis("off")

sent_counts = tweets_clean["sentiment_label"].value_counts()
axes[1].bar(sent_counts.index, sent_counts.values,
            color=["#2ca02c", "#d62728", "#7f7f7f"])
axes[1].set_title("Tweet Sentiment Distribution")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("political_network_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Example 2: Conjoint Analysis of Housing Preferences

This example analyzes a conjoint experiment measuring how race, income level,
and neighborhood density affect housing preference decisions.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2024)

# ------------------------------------------------------------------
# Simulate conjoint survey data
# (In practice, load from Qualtrics/SurveyMonkey export)
# ------------------------------------------------------------------
n_respondents = 800
n_tasks = 5
n_profiles_per_task = 2

races = ["White", "Black", "Hispanic", "Asian"]
incomes = ["Low (<$40k)", "Middle ($40k-$100k)", "High (>$100k)"]
densities = ["Suburban", "Urban High-Rise", "Rural"]
green_spaces = ["None", "Park within 1 mile", "Park on block"]

records = []
respondent_ids = range(1, n_respondents + 1)

for resp_id in respondent_ids:
    for task in range(1, n_tasks + 1):
        for profile in range(1, n_profiles_per_task + 1):
            race = np.random.choice(races)
            income = np.random.choice(incomes)
            density = np.random.choice(densities)
            green = np.random.choice(green_spaces)

            # Simulate choice with known AMCEs for testing
            latent = (
                0.0                               # intercept
                + (0.15 if race == "White" else 0.0)
                - (0.10 if race == "Black" else 0.0)
                + (0.20 if income == "High (>$100k)" else 0.0)
                - (0.08 if income == "Low (<$40k)" else 0.0)
                - (0.12 if density == "Urban High-Rise" else 0.0)
                + (0.10 if green == "Park on block" else 0.0)
                + np.random.logistic(0, 0.3)
            )
            records.append({
                "respondent_id": resp_id,
                "task_id": task,
                "profile_id": profile,
                "race_neighbor": race,
                "income_level": income,
                "density": density,
                "green_space": green,
                "chosen": 1 if latent > 0 else 0,
                "weight": np.random.uniform(0.8, 1.2),
            })

conjoint_df = pd.DataFrame(records)
print(f"Conjoint dataset: {len(conjoint_df)} rows, "
      f"{conjoint_df['respondent_id'].nunique()} respondents")

# ------------------------------------------------------------------
# Compute AMCEs
# ------------------------------------------------------------------
treatment_attributes = ["race_neighbor", "income_level", "density", "green_space"]

amce_results = analyze_conjoint_amce(
    df=conjoint_df,
    outcome="chosen",
    treatment_cols=treatment_attributes,
    respondent_id_col="respondent_id",
    weights_col="weight",
)

print("\n=== AMCE Results: Housing Preference Conjoint ===")
print(amce_results.to_string(index=False, float_format="{:.3f}".format))

# ------------------------------------------------------------------
# Compute marginal means for race attribute
# ------------------------------------------------------------------
mm_race = compute_marginal_means(conjoint_df, outcome="chosen", attribute="race_neighbor")
print("\n=== Marginal Means: Race of Neighbor ===")
print(mm_race.to_string(index=False, float_format="{:.3f}".format))

# ------------------------------------------------------------------
# Visualize AMCE plot (forest plot style)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, len(treatment_attributes), figsize=(16, 6), sharey=False)

attr_colors = {
    "race_neighbor": "#1f77b4",
    "income_level": "#ff7f0e",
    "density": "#2ca02c",
    "green_space": "#9467bd",
}

for ax, attr in zip(axes, treatment_attributes):
    sub = amce_results[amce_results["attribute"] == attr].copy()
    sub = sub.sort_values("amce")
    y_pos = range(len(sub))

    ax.barh(
        y_pos,
        sub["amce"],
        xerr=[sub["amce"] - sub["ci_lower"], sub["ci_upper"] - sub["amce"]],
        color=attr_colors.get(attr, "steelblue"),
        alpha=0.75,
        capsize=4,
        height=0.5,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sub["level"].tolist(), fontsize=8)
    ax.set_xlabel("AMCE (Δ Pr(chosen))", fontsize=9)
    ax.set_title(attr.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

plt.suptitle(
    "AMCEs: Housing Preference Conjoint Experiment\n"
    "(Baseline: Asian | Low income | Suburban | No green space)",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("housing_conjoint_amce.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Notes and Best Practices

- **Rate limits**: Twitter Academic API allows 500 requests/15 min. Use `wait_on_rate_limit=True` in tweepy.
- **PRAW rate limits**: Reddit allows ~60 requests/minute for read-only access. PRAW handles this automatically.
- **Bot detection**: Heuristic rules have ~70-80% accuracy. For production, consider Botometer API.
- **Homophily vs. selection**: Assortativity does not prove causal homophily; control for confounders (geography, topic).
- **Conjoint validity**: Ensure profiles are independently randomized; check for profile-order effects via task-position covariates.
- **Ethics**: Do not collect or publish individual-level data without IRB approval. Anonymize user IDs before sharing.
- **VADER limitations**: VADER works best on short English social media text; consider fine-tuned BERT models for other languages.
