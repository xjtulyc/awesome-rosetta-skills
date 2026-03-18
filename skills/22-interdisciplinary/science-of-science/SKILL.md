---
name: science-of-science
description: Science of science analysis covering disruption index, team size effects, sleeping beauty detection, and knowledge recombination with citation data.
tags:
  - science-of-science
  - disruption-index
  - knowledge-recombination
  - team-science
  - scientometrics
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
    - pandas>=2.0
    - numpy>=1.24
    - scipy>=1.11
    - networkx>=3.2
    - scikit-learn>=1.3
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: stable
---

# Science of Science Analysis

## When to Use This Skill

Use this skill when you need to:
- Compute disruption and consolidation indices for papers and patents
- Analyze team size effects on scientific output and impact
- Detect sleeping beauties (late-recognized influential papers)
- Measure knowledge recombination novelty and conventionality
- Study the Matthew effect and cumulative advantage in citations
- Map knowledge flow across disciplines using citation network analysis
- Quantify scientific collaboration patterns and cross-disciplinary research

**Trigger keywords**: science of science, disruption index, CD index, sleeping beauty, knowledge recombination, novelty conventionality, team size, Matthew effect, cumulative advantage, disciplinary distance, cross-disciplinary, big team small team, Wang and Barabasi, knowledge flows, citation network, research front, knowledge base, scientific discovery patterns.

## Background & Key Concepts

### Disruption Index (CD Index, Funk & Owen-Smith 2017)

Measures whether a paper pushes science forward (disruptive) or builds on existing work (consolidating):

$$\text{CD}_i = \frac{n_f - n_{fb}}{n_f + n_b + n_{fb}}$$

where for citing papers $c$ of paper $i$:
- $n_f$: papers that cite $i$ but not any of $i$'s references
- $n_{fb}$: papers that cite both $i$ and at least one of $i$'s references
- $n_b$: papers that cite $i$'s references but not $i$

CD = 1: fully disruptive; CD = -1: fully consolidating; CD = 0: neutral.

### Sleeping Beauty Detection (van Raan 2004)

A sleeping beauty is a paper that has a long dormancy period before receiving high citations. Define:
- $t_0$: year of publication
- $t_{max}$: year of peak citations
- $B = \frac{c_{max} - c_0}{t_{max} - t_0}$ (awakening intensity)
- $L = $ length of dormancy period ($c < $ threshold)

Sleeping beauty score: $B \cdot L$.

### Knowledge Recombination (Uzzi et al. 2013)

For a paper's reference list, compute the **conventionality** (journals that are typically cited together) and **novelty** (atypical combinations):

$$z\text{-score}_{ij} = \frac{\text{co-citation}_{ij} - \mu_{ij}}{\sigma_{ij}}$$

where $\mu_{ij}$ is the expected co-citation count under a random reference model (Monte Carlo permutations of reference lists).

### Matthew Effect

$$\frac{dc_i(t)}{dt} = \alpha c_i(t)^\beta + \gamma$$

preferential attachment: papers with more citations attract even more citations. Estimated via power regression on citation velocity.

## Environment Setup

```bash
pip install pandas>=2.0 numpy>=1.24 scipy>=1.11 networkx>=3.2 \
            scikit-learn>=1.3 matplotlib>=3.7
```

```python
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
print("Science of science environment ready")
```

## Core Workflow

### Step 1: Disruption Index Computation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate a citation graph for 300 papers
# Each paper has references; later papers cite earlier papers
# -----------------------------------------------------------------
np.random.seed(42)
n_papers = 300
paper_ids = list(range(n_papers))
years = np.sort(np.random.randint(2000, 2023, n_papers))

# Build reference lists: each paper cites 5-15 older papers
references = {}
for i in range(n_papers):
    older = [j for j in range(i) if years[j] >= years[i] - 10]
    if len(older) >= 3:
        n_refs = min(np.random.randint(5, 16), len(older))
        references[i] = list(np.random.choice(older, n_refs, replace=False))
    else:
        references[i] = older

# Who cites whom
cited_by = {i: [] for i in paper_ids}
for citer, refs in references.items():
    for cited in refs:
        cited_by[cited].append(citer)

# -----------------------------------------------------------------
# Compute CD index for each paper
# -----------------------------------------------------------------
def compute_cd_index(focal, references, cited_by):
    """Compute Funk & Owen-Smith CD disruption index.

    Args:
        focal: focal paper ID
        references: dict {paper: [references]}
        cited_by: dict {paper: [papers that cite it]}
    Returns:
        CD index in [-1, 1]
    """
    focal_refs = set(references.get(focal, []))
    citers_of_focal = set(cited_by.get(focal, []))

    if not citers_of_focal:
        return np.nan

    n_f = 0    # cite focal, not focal's refs
    n_fb = 0   # cite focal AND at least one focal ref
    n_b = 0    # cite focal's refs but not focal

    # For n_f and n_fb: iterate over papers that cite focal
    for c in citers_of_focal:
        c_refs = set(references.get(c, []))
        cites_focal_refs = bool(c_refs & focal_refs)
        if cites_focal_refs:
            n_fb += 1
        else:
            n_f += 1

    # For n_b: iterate over citers of focal's references who don't cite focal
    citers_of_refs = set()
    for ref in focal_refs:
        citers_of_refs.update(cited_by.get(ref, []))
    citers_of_refs -= citers_of_focal  # exclude those who also cite focal
    n_b = len(citers_of_refs)

    denom = n_f + n_fb + n_b
    if denom == 0:
        return np.nan
    return (n_f - n_fb) / denom

# Compute for all papers (subset for speed)
cd_scores = []
for pid in range(min(200, n_papers)):
    cd = compute_cd_index(pid, references, cited_by)
    cd_scores.append({"paper_id": pid, "year": years[pid],
                      "cd_index": cd,
                      "n_citations": len(cited_by[pid]),
                      "n_references": len(references[pid])})

cd_df = pd.DataFrame(cd_scores).dropna()
print("=== Disruption Index Summary ===")
print(cd_df["cd_index"].describe().round(3))
print(f"\nDisruptive papers (CD > 0.5): {(cd_df['cd_index'] > 0.5).sum()}")
print(f"Consolidating papers (CD < -0.5): {(cd_df['cd_index'] < -0.5).sum()}")

# Most disruptive papers
top_disruptive = cd_df.nlargest(5, "cd_index")[["paper_id", "year", "cd_index", "n_citations"]]
print("\nTop 5 Most Disruptive Papers:")
print(top_disruptive.round(3).to_string(index=False))

# CD index over time trend
annual_cd = cd_df.groupby("year")["cd_index"].agg(["mean", "std"]).reset_index()
print("\nCD Index Trend Over Time:")
print(annual_cd.round(3).to_string(index=False))

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# CD distribution
axes[0].hist(cd_df["cd_index"], bins=25, color="steelblue", edgecolor="black")
axes[0].axvline(0, color="red", ls="--", label="Neutral")
axes[0].axvline(cd_df["cd_index"].mean(), color="orange", ls="-",
                label=f"Mean={cd_df['cd_index'].mean():.2f}")
axes[0].set_xlabel("CD Index"); axes[0].set_ylabel("Frequency")
axes[0].set_title("Disruption Index Distribution")
axes[0].legend()

# CD vs. citations scatter
axes[1].scatter(cd_df["cd_index"], np.log1p(cd_df["n_citations"]),
                alpha=0.4, s=20, color="steelblue")
axes[1].set_xlabel("CD Index"); axes[1].set_ylabel("ln(Citations + 1)")
axes[1].set_title("Disruption vs. Citation Impact")

# Annual CD trend
axes[2].plot(annual_cd["year"], annual_cd["mean"], "o-", color="steelblue")
axes[2].fill_between(annual_cd["year"],
                     annual_cd["mean"] - annual_cd["std"],
                     annual_cd["mean"] + annual_cd["std"],
                     alpha=0.2, color="steelblue")
axes[2].axhline(0, color="red", ls="--")
axes[2].set_xlabel("Year"); axes[2].set_ylabel("Mean CD Index")
axes[2].set_title("Disruption Trend Over Time")

plt.tight_layout()
plt.savefig("disruption_index.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: disruption_index.png")
```

### Step 2: Team Size and Scientific Impact

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# -----------------------------------------------------------------
# Simulate paper dataset with team size metadata
# Based on Wu, Wang, and Evans (2019) findings
# -----------------------------------------------------------------
np.random.seed(42)
n_papers = 2000

team_sizes = np.random.choice(range(1, 20), n_papers,
                               p=np.exp(-0.15 * np.arange(19)) /
                                 np.exp(-0.15 * np.arange(19)).sum())

# Large teams tend to have higher average citations (incremental work)
# Small teams more likely to be disruptive but higher variance
base_cit = 10
citations = np.array([
    int(np.random.lognormal(
        np.log(base_cit) + 0.03 * ts,  # slight positive team effect on mean
        max(1.5 - 0.05 * ts, 0.5)      # smaller variance for larger teams
    ))
    for ts in team_sizes
])

# Disruption index (anticorrelated with team size)
disruption = np.array([
    np.random.normal(0.3 - 0.02 * ts, 0.3)
    for ts in team_sizes
])
disruption = np.clip(disruption, -1, 1)

# Novel work indicator (1 = novel combination)
novelty = np.where(np.random.random(n_papers) < np.exp(-0.1 * team_sizes) * 0.4, 1, 0)

df_team = pd.DataFrame({
    "team_size": team_sizes,
    "citations": citations,
    "disruption": disruption,
    "novel": novelty,
    "year": np.random.randint(2010, 2023, n_papers),
})

# -----------------------------------------------------------------
# Analysis: team size effects
# -----------------------------------------------------------------
# OLS: citations ~ log(team_size) + year
df_team["ln_team"] = np.log(df_team["team_size"])
df_team["ln_cit"] = np.log1p(df_team["citations"])
X = sm.add_constant(df_team[["ln_team", "year"]])
model = sm.OLS(df_team["ln_cit"], X).fit(cov_type="HC3")
print("=== Citations ~ Team Size (OLS) ===")
print(model.summary().tables[1])

# Spearman correlation: team size vs. disruption
r_d, p_d = spearmanr(df_team["team_size"], df_team["disruption"])
print(f"\nSpearman(team_size, disruption): r={r_d:.3f}, p={p_d:.4f}")

# Binned analysis: team size categories
df_team["team_cat"] = pd.cut(df_team["team_size"],
                               bins=[0, 3, 7, 12, 20],
                               labels=["Solo/Duo (1-3)", "Small (4-7)",
                                       "Medium (8-12)", "Large (13-20)"])
team_summary = df_team.groupby("team_cat", observed=True).agg(
    n_papers=("citations", "count"),
    mean_cit=("citations", "mean"),
    median_cit=("citations", "median"),
    mean_disruption=("disruption", "mean"),
    novel_rate=("novel", "mean"),
).reset_index()
print("\n=== Team Size Category Analysis ===")
print(team_summary.round(3).to_string(index=False))

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Citations by team size bin (boxplot)
grouped_cit = [df_team[df_team["team_cat"] == cat]["citations"].values
               for cat in df_team["team_cat"].cat.categories]
axes[0, 0].boxplot(grouped_cit, labels=df_team["team_cat"].cat.categories,
                    showfliers=False, patch_artist=True)
axes[0, 0].set_title("Citation Counts by Team Size")
axes[0, 0].set_ylabel("Citations")
axes[0, 0].set_xticklabels(df_team["team_cat"].cat.categories,
                             rotation=15, ha="right")

# Disruption by team size bin (violin plot alternative)
disruption_groups = [df_team[df_team["team_cat"] == cat]["disruption"].values
                     for cat in df_team["team_cat"].cat.categories]
axes[0, 1].boxplot(disruption_groups, labels=df_team["team_cat"].cat.categories,
                    showfliers=False, patch_artist=True)
axes[0, 1].axhline(0, color="red", ls="--")
axes[0, 1].set_title(f"Disruption Index by Team Size\n(r={r_d:.2f})")
axes[0, 1].set_ylabel("CD Index")
axes[0, 1].set_xticklabels(df_team["team_cat"].cat.categories,
                             rotation=15, ha="right")

# Scatter: team size vs. disruption
axes[1, 0].scatter(df_team["team_size"], df_team["disruption"],
                   alpha=0.2, s=10, c="steelblue")
# Running mean
ts_sorted = df_team.sort_values("team_size")
running_mean = ts_sorted.groupby("team_size")["disruption"].mean()
axes[1, 0].plot(running_mean.index, running_mean.values, "r-", lw=2)
axes[1, 0].set_xlabel("Team Size"); axes[1, 0].set_ylabel("CD Index")
axes[1, 0].set_title("Team Size vs. Disruption")

# Novelty rate by team size
axes[1, 1].bar(team_summary["team_cat"],
               team_summary["novel_rate"] * 100,
               color="green", edgecolor="black", alpha=0.7)
axes[1, 1].set_title("Novel Combination Rate by Team Size")
axes[1, 1].set_ylabel("Novel Papers (%)")
axes[1, 1].set_xticklabels(team_summary["team_cat"], rotation=15, ha="right")

plt.tight_layout()
plt.savefig("team_size_effects.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: team_size_effects.png")
```

### Step 3: Sleeping Beauty Detection

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate annual citation histories for 200 papers
# Most have normal decay; a few are sleeping beauties
# -----------------------------------------------------------------
np.random.seed(42)
n_papers_sb = 200
max_age = 25  # years since publication

def simulate_normal_citation_history(peak_age=3, peak_cit=50, decay=0.3, n_years=25):
    """Simulate a normal citation trajectory (rise then fall)."""
    ages = np.arange(n_years)
    cit = peak_cit * np.exp(-0.5 * ((ages - peak_age) / decay)**2)
    cit = np.maximum(cit + np.random.normal(0, 2, n_years), 0).astype(int)
    return cit

def simulate_sleeping_beauty(dormancy=15, awakening_year=18, n_years=25):
    """Simulate a sleeping beauty: dormant then sudden awakening."""
    cit = np.zeros(n_years, dtype=int)
    cit[:dormancy] = np.random.randint(0, 3, dormancy)  # near-zero before awakening
    wake_age = min(awakening_year, n_years - 1)
    for t in range(wake_age, n_years):
        cit[t] = int(5 * np.exp(0.4 * (t - wake_age)) + np.random.normal(0, 3))
    return np.maximum(cit, 0)

# Generate citation histories
cit_histories = []
for i in range(n_papers_sb):
    if i < 5:  # first 5 are sleeping beauties
        cit_histories.append(simulate_sleeping_beauty(
            dormancy=np.random.randint(10, 18),
            awakening_year=np.random.randint(15, 22)
        ))
    else:
        cit_histories.append(simulate_normal_citation_history(
            peak_age=np.random.randint(2, 7),
            peak_cit=np.random.randint(5, 100),
            decay=np.random.uniform(0.2, 0.8)
        ))

cit_matrix = np.array(cit_histories)

# -----------------------------------------------------------------
# Sleeping beauty score: B * L
# B = (c_max - c_t0) / (t_max - t0)
# L = dormancy length
# -----------------------------------------------------------------
def sleeping_beauty_score(cit_history, dormancy_threshold=5):
    """Compute sleeping beauty score for a citation trajectory.

    Args:
        cit_history: array of annual citations
        dormancy_threshold: max citations during sleeping period
    Returns:
        score, beauty_length, awakening_year
    """
    n = len(cit_history)
    c_max = cit_history.max()
    t_max = int(np.argmax(cit_history))
    c_0 = cit_history[0]

    # Find dormancy: longest initial period with citations <= threshold
    dormancy_end = 0
    for t in range(n):
        if cit_history[t] <= dormancy_threshold:
            dormancy_end = t
        else:
            if t > dormancy_end + 1:
                break

    beauty_length = dormancy_end
    if t_max <= beauty_length or (t_max - 0) == 0:
        return 0.0, 0, t_max

    B = (c_max - c_0) / max(t_max, 1)
    score = B * beauty_length
    return score, beauty_length, t_max

sb_scores = []
for i, hist in enumerate(cit_histories):
    score, length, awaken = sleeping_beauty_score(hist)
    total_cit = hist.sum()
    sb_scores.append({
        "paper_id": i,
        "is_sleeping_beauty": i < 5,
        "sb_score": score,
        "dormancy_length": length,
        "awakening_year": awaken,
        "total_citations": total_cit,
        "peak_citations": hist.max(),
    })

sb_df = pd.DataFrame(sb_scores)
print("=== Sleeping Beauty Detection ===")
print(f"Top 10 SB scores:")
print(sb_df.nlargest(10, "sb_score")[
    ["paper_id", "is_sleeping_beauty", "sb_score", "dormancy_length",
     "total_citations"]].round(2).to_string(index=False))

# How many true sleeping beauties in top 10?
top10_sb = sb_df.nlargest(10, "sb_score")["is_sleeping_beauty"].sum()
print(f"\nTrue sleeping beauties in top 10: {top10_sb}/5")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Citation histories
colors_sb = {True: "red", False: "steelblue"}
for i in range(n_papers_sb):
    is_sb = sb_df.loc[i, "is_sleeping_beauty"]
    alpha = 0.7 if is_sb else 0.05
    lw = 2 if is_sb else 0.5
    axes[0].plot(cit_histories[i], color=colors_sb[is_sb],
                 alpha=alpha, lw=lw)
axes[0].set_xlabel("Years Since Publication"); axes[0].set_ylabel("Annual Citations")
axes[0].set_title("Citation Trajectories\n(red = sleeping beauties)")

# SB score distribution
axes[1].scatter(sb_df["dormancy_length"],
                sb_df["peak_citations"],
                c=sb_df["sb_score"], cmap="hot", s=30, alpha=0.7)
sb_true = sb_df[sb_df["is_sleeping_beauty"]]
axes[1].scatter(sb_true["dormancy_length"],
                sb_true["peak_citations"],
                c="red", s=100, marker="*", zorder=5, label="True SBs")
axes[1].set_xlabel("Dormancy Length (years)")
axes[1].set_ylabel("Peak Citations")
axes[1].set_title("Sleeping Beauty Space")
axes[1].legend()

plt.tight_layout()
plt.savefig("sleeping_beauties.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: sleeping_beauties.png")
```

## Advanced Usage

### Knowledge Recombination Novelty (Uzzi et al.)

```python
import numpy as np
import pandas as pd
from scipy.stats import zscore

def compute_recombination_novelty(reference_lists, journal_assignments,
                                   n_permutations=100):
    """Compute median z-score of journal pair co-citations.

    Args:
        reference_lists: list of lists; each inner list = paper's references (paper IDs)
        journal_assignments: dict {paper_id: journal}
        n_permutations: Monte Carlo permutations
    Returns:
        per-paper novelty score (negative z-score = novel)
    """
    from collections import Counter

    # Observed co-citation counts by journal pair
    observed_pairs = Counter()
    for refs in reference_lists:
        journals = [journal_assignments.get(r, "Unknown") for r in refs]
        journals = [j for j in journals if j != "Unknown"]
        journals = list(set(journals))  # unique journals in reference list
        for i in range(len(journals)):
            for j in range(i + 1, len(journals)):
                key = tuple(sorted([journals[i], journals[j]]))
                observed_pairs[key] += 1

    # Null distribution via reference list permutations
    all_refs_flat = [r for refs in reference_lists for r in refs]
    null_pair_counts = []
    for _ in range(n_permutations):
        # Shuffle journal assignments
        shuffled = {pid: journal_assignments.get(pid, "Unknown")
                    for pid in np.random.permutation(list(journal_assignments.keys()))}
        perm_pairs = Counter()
        for refs in reference_lists:
            journals = [shuffled.get(r, "Unknown") for r in refs]
            journals = [j for j in journals if j != "Unknown"]
            journals = list(set(journals))
            for i in range(len(journals)):
                for j in range(i + 1, len(journals)):
                    key = tuple(sorted([journals[i], journals[j]]))
                    perm_pairs[key] += 1
        null_pair_counts.append(perm_pairs)

    # Z-scores for each journal pair
    all_pairs = set(observed_pairs.keys())
    z_scores = {}
    for pair in all_pairs:
        obs = observed_pairs[pair]
        null_vals = [null[pair] for null in null_pair_counts]
        mu = np.mean(null_vals)
        sigma = np.std(null_vals)
        z_scores[pair] = (obs - mu) / max(sigma, 0.1)

    # Paper-level median z-score
    paper_novelty = []
    for refs in reference_lists:
        journals = [journal_assignments.get(r, "Unknown") for r in refs]
        journals = [j for j in journals if j != "Unknown"]
        journals = list(set(journals))
        paper_z = []
        for i in range(len(journals)):
            for j in range(i + 1, len(journals)):
                key = tuple(sorted([journals[i], journals[j]]))
                if key in z_scores:
                    paper_z.append(z_scores[key])
        paper_novelty.append(np.median(paper_z) if paper_z else 0.0)

    return paper_novelty, z_scores

# Demonstrate
np.random.seed(42)
n_jrnl_pool = 20
journals = [f"J{i:02d}" for i in range(n_jrnl_pool)]
n_papers_r = 50
ref_lists = [list(np.random.choice(range(100), size=np.random.randint(5, 15),
                                    replace=False))
             for _ in range(n_papers_r)]
j_assign = {i: journals[i % n_jrnl_pool] for i in range(100)}

paper_novelty, pair_z = compute_recombination_novelty(ref_lists, j_assign,
                                                        n_permutations=20)
print(f"Mean paper novelty (median z): {np.mean(paper_novelty):.3f}")
print(f"Novel papers (z < -0.5): {sum(z < -0.5 for z in paper_novelty)}/{n_papers_r}")
```

### Matthew Effect Estimation

```python
import numpy as np
import pandas as pd
import scipy.optimize as opt

def estimate_matthew_effect(citation_histories, time_window=5):
    """Estimate preferential attachment exponent.

    Fit: delta_c ~ c(t)^beta using log-linear regression.
    """
    pairs = []
    for hist in citation_histories:
        for t in range(len(hist) - time_window):
            c_t = hist[t]
            delta_c = hist[t + time_window] - hist[t]
            if c_t > 0 and delta_c > 0:
                pairs.append((np.log(c_t), np.log(delta_c)))

    if len(pairs) < 10:
        return None

    ln_c, ln_delta = zip(*pairs)
    ln_c = np.array(ln_c); ln_delta = np.array(ln_delta)

    # OLS
    A = np.column_stack([np.ones_like(ln_c), ln_c])
    result = np.linalg.lstsq(A, ln_delta, rcond=None)
    alpha, beta = result[0]

    print(f"Matthew effect: ln(Δc) = {alpha:.3f} + {beta:.3f}·ln(c)")
    print(f"Beta > 1: super-linear preferential attachment = {beta > 1}")
    return {"alpha": alpha, "beta": beta, "n_pairs": len(pairs)}

# Use cit_histories from sleeping beauty step
matthew = estimate_matthew_effect(cit_histories[:50], time_window=3)
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| CD = NaN for many papers | No citing papers (zero forward citations) | Expected for recent papers; filter to age ≥ 5 years |
| All CD ≈ 0 | Very sparse citation graph | Need longer time window; check graph has sufficient edges |
| Sleeping beauty false positives | Threshold too low | Tune `dormancy_threshold` to 10% of field median citations |
| Recombination z-score undefined | Only 1 journal in references | Skip single-journal papers for pair analysis |
| Matthew effect beta > 2 | Outlier highly-cited papers | Winsorize citations at 99th percentile |
| Team size analysis confounded by field | Different publishing norms | Include field fixed effects in regression |

## External Resources

- Funk, R. J., & Owen-Smith, J. (2017). A dynamic network measure of technological change. *Management Science*, 63(3).
- Wu, L., Wang, D., & Evans, J. A. (2019). Large teams develop and small teams disrupt science and technology. *Nature*, 566, 378-382.
- Van Raan, A. F. (2004). Sleeping beauties in science. *Scientometrics*, 59(3), 467-472.
- Uzzi, B., et al. (2013). Atypical combinations and scientific impact. *Science*, 342(6157).
- Wang, D., & Barabási, A.-L. (2021). *The Science of Science*. Cambridge University Press.

## Examples

### Example 1: Disciplinary Diversity of References

```python
import numpy as np
import pandas as pd

def rao_stirling_diversity(field_shares):
    """Compute Rao-Stirling diversity: sum_ij d_ij * p_i * p_j.

    Args:
        field_shares: array of proportions (sum to 1)
    Returns:
        diversity index
    """
    n = len(field_shares)
    # Assume all fields equally distant (d_ij = 1 for i≠j)
    rs = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rs += 2 * field_shares[i] * field_shares[j]  # d_ij = 1
    return rs

np.random.seed(42)
n_pap = 200
n_fields = 8
ref_field_distributions = np.random.dirichlet(np.ones(n_fields) * 0.5, n_pap)
diversity = np.array([rao_stirling_diversity(shares)
                      for shares in ref_field_distributions])

print(f"Mean Rao-Stirling diversity: {diversity.mean():.3f}")
print(f"High diversity papers (RS > 0.8): {(diversity > 0.8).sum()}/{n_pap}")
```

### Example 2: Knowledge Flow Between Disciplines

```python
import numpy as np
import pandas as pd
import networkx as nx

def knowledge_flow_network(citation_df, source_field_col="source_field",
                             target_field_col="target_field"):
    """Build directed knowledge flow network between disciplines.

    Args:
        citation_df: DataFrame with citing and cited paper fields
    Returns:
        G: weighted directed NetworkX graph
    """
    G = nx.DiGraph()
    flow_counts = citation_df.groupby(
        [source_field_col, target_field_col]).size().reset_index(name="weight")

    for _, row in flow_counts.iterrows():
        src, tgt, w = row[source_field_col], row[target_field_col], row["weight"]
        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += w
            else:
                G.add_edge(src, tgt, weight=w)
    return G

fields = ["CS", "Physics", "Biology", "Chemistry", "Math", "Medicine"]
np.random.seed(42)
n_cit = 500
# Citations from papers in one field to papers in another
cit_df = pd.DataFrame({
    "source_field": np.random.choice(fields, n_cit,
                                      p=[0.30, 0.20, 0.20, 0.15, 0.10, 0.05]),
    "target_field": np.random.choice(fields, n_cit),
})
cit_df = cit_df[cit_df["source_field"] != cit_df["target_field"]]

G_flow = knowledge_flow_network(cit_df)
print(f"Knowledge flow network: {G_flow.number_of_nodes()} fields, "
      f"{G_flow.number_of_edges()} directed edges")

# Identify biggest knowledge importers (in-degree weighted)
in_strength = {n: sum(d["weight"] for _, _, d in G_flow.in_edges(n, data=True))
               for n in G_flow.nodes()}
print("\nKnowledge importers (in-strength):")
for f, s in sorted(in_strength.items(), key=lambda x: x[1], reverse=True):
    print(f"  {f}: {s}")
```
