---
name: mixed-methods-research
description: Mixed methods research design with qualitative-quantitative integration, triangulation, content analysis, and systematic comparison using NVivo-style coding.
tags:
  - mixed-methods
  - research-design
  - triangulation
  - content-analysis
  - systematic-comparison
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
    - scikit-learn>=1.3
    - statsmodels>=0.14
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: stable
---

# Mixed Methods Research Design and Analysis

## When to Use This Skill

Use this skill when you need to:
- Design and execute mixed methods research combining qualitative and quantitative data
- Apply content analysis to text corpora with systematic coding schemes
- Perform triangulation across multiple data sources and methods
- Integrate interview themes with survey statistics
- Conduct qualitative comparative analysis (QCA) for case studies
- Measure inter-rater reliability across coders
- Synthesize qualitative evidence with quantitative findings

**Trigger keywords**: mixed methods, convergent design, sequential explanatory, sequential exploratory, triangulation, content analysis, qualitative coding, constant comparative method, thematic analysis, NVivo, ATLAS.ti, qualitative comparative analysis, QCA, crisp-set QCA, fuzzy-set QCA, Boolean algebra, necessary conditions, sufficient conditions, MANOVA, ANOVA with coding, inter-rater reliability, saturation.

## Background & Key Concepts

### Mixed Methods Research Designs

1. **Convergent (parallel)**: Collect qual and quant data simultaneously, analyze separately, then merge interpretations
2. **Sequential explanatory**: Quant first → qual to explain quant findings
3. **Sequential exploratory**: Qual first → quant to test themes
4. **Embedded**: Quant primary, qual nested within (or vice versa)

### Triangulation

$$\text{Convergence} = \text{sign}(r_{\text{qual-quant}}) \times P(\text{same direction})$$

Strong triangulation: qual themes and quant results point in the same direction, increasing confidence in findings.

### Content Analysis (Krippendorff 2004)

For nominal coding of $n$ items by $k$ coders:

- **Cohen's kappa** (2 coders): $\kappa = \frac{P_o - P_e}{1 - P_e}$
- **Krippendorff's alpha** ($k$ coders, ordinal/interval): $\alpha = 1 - \frac{D_o}{D_e}$

where $D_o$ is observed disagreement, $D_e$ is expected disagreement.

### Qualitative Comparative Analysis (QCA)

Boolean truth table analysis. For crisp-set QCA (csQCA), each condition $C_i \in \{0,1\}$ and outcome $O \in \{0,1\}$.

**Necessity**: $C \Rightarrow O$: coverage $= \frac{\sum \min(C, O)}{\sum O}$, consistency $= \frac{\sum \min(C, O)}{\sum C}$

**Sufficiency**: $C \Rightarrow O$: coverage $= \frac{\sum \min(C, O)}{\sum O}$, consistency $= \frac{\sum \min(C, O)}{\sum C}$

### Integration Point Analysis

A joint display maps qualitative themes to quantitative outcomes, identifying where the two strands converge, diverge, or complement.

## Environment Setup

```bash
pip install pandas>=2.0 numpy>=1.24 scipy>=1.11 scikit-learn>=1.3 \
            statsmodels>=0.14 matplotlib>=3.7
```

```python
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
print("Mixed methods environment ready")
```

## Core Workflow

### Step 1: Systematic Content Analysis with Coding Scheme

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import re

# -----------------------------------------------------------------
# Simulate interview data: 50 interview segments
# Two coders independently assign codes
# -----------------------------------------------------------------
np.random.seed(42)

CODE_SCHEME = {
    "BARRIER": ["Barrier/Obstacle to implementation"],
    "ENABLER": ["Enabler/Facilitator of implementation"],
    "OUTCOME": ["Outcome/Result described"],
    "RECOMMENDATION": ["Policy/Practice recommendation"],
    "CONTEXT": ["Context/Background information"],
}
CODES = list(CODE_SCHEME.keys())

# Interview segments
themes_pool = [
    "lack of funding prevented the initiative from scaling",
    "management support was crucial for adoption",
    "the program led to improved outcomes for participants",
    "we recommend investing in training for frontline staff",
    "this occurred in a resource-constrained setting",
    "regulatory barriers slowed down implementation",
    "strong leadership facilitated rapid adoption",
    "participants reported significant benefits",
    "future policy should prioritize equity concerns",
    "the rural context posed unique challenges",
]

n_segments = 50
segments = np.random.choice(themes_pool, n_segments)

# True codes (ground truth based on segment content)
def assign_true_code(segment):
    """Assign code based on keyword matching."""
    if any(w in segment for w in ["barrier", "lack", "prevented", "slowed"]):
        return "BARRIER"
    elif any(w in segment for w in ["support", "leadership", "facilitated"]):
        return "ENABLER"
    elif any(w in segment for w in ["outcomes", "benefits", "improved", "reported"]):
        return "OUTCOME"
    elif any(w in segment for w in ["recommend", "policy", "should"]):
        return "RECOMMENDATION"
    else:
        return "CONTEXT"

true_codes = [assign_true_code(s) for s in segments]

# Coder 1: 90% agreement with truth
# Coder 2: 85% agreement with truth
def apply_coder_noise(true_codes, accuracy):
    noisy = []
    for code in true_codes:
        if np.random.random() < accuracy:
            noisy.append(code)
        else:
            noisy.append(np.random.choice([c for c in CODES if c != code]))
    return noisy

coder1 = apply_coder_noise(true_codes, 0.90)
coder2 = apply_coder_noise(true_codes, 0.85)

df_coding = pd.DataFrame({
    "segment_id": range(n_segments),
    "text": segments,
    "true_code": true_codes,
    "coder1": coder1,
    "coder2": coder2,
})

# -----------------------------------------------------------------
# Inter-rater reliability: Cohen's Kappa
# -----------------------------------------------------------------
def cohen_kappa(a, b):
    """Compute Cohen's kappa between two coders."""
    categories = sorted(set(a) | set(b))
    n = len(a)
    # Observed agreement
    P_o = sum(x == y for x, y in zip(a, b)) / n

    # Expected agreement
    freq_a = {c: a.count(c) / n for c in categories}
    freq_b = {c: b.count(c) / n for c in categories}
    P_e = sum(freq_a.get(c, 0) * freq_b.get(c, 0) for c in categories)

    if P_e == 1.0:
        return 1.0
    return (P_o - P_e) / (1 - P_e)

kappa = cohen_kappa(coder1, coder2)
print(f"Cohen's Kappa (Coder1 vs Coder2): κ = {kappa:.3f}")
if kappa >= 0.80: print("  → Excellent agreement")
elif kappa >= 0.61: print("  → Substantial agreement")
elif kappa >= 0.41: print("  → Moderate agreement")
else: print("  → Fair/poor agreement — recalibrate coders")

# -----------------------------------------------------------------
# Krippendorff's Alpha (nominal scale, 2 coders)
# -----------------------------------------------------------------
def krippendorff_alpha_nominal(codings_matrix):
    """Compute Krippendorff's alpha for nominal scale.

    Args:
        codings_matrix: (n_coders, n_items) array of codes (integers)
    Returns:
        alpha value
    """
    k, n = codings_matrix.shape
    # Convert to integer labels
    categories = sorted(set(codings_matrix.flatten()))
    cat_map = {c: i for i, c in enumerate(categories)}
    C = np.array([[cat_map[v] for v in row] for row in codings_matrix])

    # Observed disagreement (all pairs within same item)
    D_o_count = 0
    D_e_count = 0
    n_total_pairs = 0

    for unit in range(n):
        unit_codes = C[:, unit]
        for i in range(k):
            for j in range(i + 1, k):
                D_o_count += int(unit_codes[i] != unit_codes[j])
                n_total_pairs += 1

    # Expected disagreement (random pair from all code values)
    all_codes = C.flatten()
    n_vals = len(all_codes)
    for i in range(n_vals):
        for j in range(i + 1, n_vals):
            D_e_count += int(all_codes[i] != all_codes[j])

    D_o = D_o_count / max(n_total_pairs, 1)
    D_e = D_e_count / max(n_vals * (n_vals - 1) / 2, 1)

    if D_e == 0:
        return 1.0
    return 1 - D_o / D_e

cat_map = {c: i for i, c in enumerate(CODES)}
coding_matrix = np.array([
    [cat_map[c] for c in coder1],
    [cat_map[c] for c in coder2],
])
alpha = krippendorff_alpha_nominal(coding_matrix)
print(f"Krippendorff's Alpha: α = {alpha:.3f}")

# -----------------------------------------------------------------
# Code frequency analysis
# -----------------------------------------------------------------
# Use consensus code (coder1 where agree, majority where disagree)
def consensus_code(c1, c2):
    return c1 if c1 == c2 else np.random.choice([c1, c2])

df_coding["consensus"] = [consensus_code(c1, c2)
                           for c1, c2 in zip(coder1, coder2)]
code_freq = df_coding["consensus"].value_counts()
print("\n=== Code Frequency Distribution ===")
print(code_freq)

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Code frequency bar chart
code_freq.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_title("Code Frequency Distribution")
axes[0].set_xlabel("Code"); axes[0].set_ylabel("Count")
axes[0].set_xticklabels(code_freq.index, rotation=30, ha="right")

# Confusion matrix (coder1 vs. coder2)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(coder1, coder2, labels=CODES)
disp = ConfusionMatrixDisplay(cm, display_labels=CODES)
disp.plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"Coder Agreement Matrix\n(κ={kappa:.2f})")
plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

plt.tight_layout()
plt.savefig("content_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: content_analysis.png")
```

### Step 2: Qualitative Comparative Analysis (QCA)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# -----------------------------------------------------------------
# Simulate a crisp-set QCA dataset: 20 cases, 4 conditions, 1 outcome
# Research question: What combinations of conditions lead to successful reform?
# -----------------------------------------------------------------
np.random.seed(42)
n_cases = 20

conditions = {
    "leadership":    [1,1,1,1,0,0,0,0,1,1,0,0,1,1,0,1,0,0,1,0],  # strong leadership
    "resources":     [1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,0,1],  # adequate resources
    "stakeholders":  [1,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,1,0],  # stakeholder buy-in
    "policy_window": [1,0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,0,1,0,1],  # policy window open
}
# Outcome: reform succeeded
outcome = [1,1,0,1,0,0,0,0,1,1,0,0,1,1,0,1,0,0,1,0]

df_qca = pd.DataFrame(conditions)
df_qca["outcome"] = outcome
df_qca.index = [f"Case_{i+1:02d}" for i in range(n_cases)]

print("=== QCA Truth Table (first 10 cases) ===")
print(df_qca.head(10))

# -----------------------------------------------------------------
# Build truth table
# -----------------------------------------------------------------
def build_truth_table(df_qca, conditions, outcome_col="outcome"):
    """Aggregate cases into truth table rows.

    Returns truth table with consistency and coverage for each configuration.
    """
    cond_cols = list(conditions.keys())
    groups = df_qca.groupby(cond_cols)
    rows = []
    for config, group in groups:
        n = len(group)
        n_outcome = group[outcome_col].sum()
        consistency = n_outcome / n if n > 0 else 0
        rows.append({
            **dict(zip(cond_cols, config)),
            "n_cases": n,
            "n_outcome": n_outcome,
            "consistency": consistency,
            "outcome": int(consistency >= 0.75),  # threshold at 0.75
        })
    return pd.DataFrame(rows).sort_values("n_cases", ascending=False)

truth_table = build_truth_table(df_qca, conditions)
print("\n=== Truth Table ===")
print(truth_table.round(2).to_string(index=False))

# -----------------------------------------------------------------
# Necessity analysis: which single conditions are necessary?
# -----------------------------------------------------------------
def necessity_analysis(df_qca, conditions, outcome_col="outcome"):
    """Test each condition and its negation for necessity."""
    results = []
    for cond in conditions.keys():
        # Consistency of necessity: sum(min(C, O)) / sum(O)
        c = df_qca[cond].values
        o = df_qca[outcome_col].values
        consistency = np.minimum(c, o).sum() / max(o.sum(), 1)
        coverage = np.minimum(c, o).sum() / max(c.sum(), 1)
        results.append({"condition": cond, "type": "presence",
                        "necessity_consistency": consistency,
                        "necessity_coverage": coverage})

        # Negation
        c_neg = 1 - c
        cons_neg = np.minimum(c_neg, o).sum() / max(o.sum(), 1)
        cov_neg = np.minimum(c_neg, o).sum() / max(c_neg.sum(), 1)
        results.append({"condition": f"~{cond}", "type": "absence",
                        "necessity_consistency": cons_neg,
                        "necessity_coverage": cov_neg})

    return pd.DataFrame(results).sort_values("necessity_consistency", ascending=False)

nec_df = necessity_analysis(df_qca, conditions)
print("\n=== Necessity Analysis ===")
print(nec_df.round(3).to_string(index=False))
print("\nNecessary conditions (consistency > 0.9):")
necessary = nec_df[nec_df["necessity_consistency"] > 0.90]
print(necessary[["condition", "necessity_consistency", "necessity_coverage"]].to_string(index=False))

# -----------------------------------------------------------------
# Sufficiency: check two-condition combinations
# -----------------------------------------------------------------
def sufficiency_analysis(df_qca, conditions, outcome_col="outcome", min_cases=2):
    """Test all 2-condition combinations for sufficiency."""
    from itertools import combinations
    cond_names = list(conditions.keys())
    results = []

    for (c1, c2) in combinations(cond_names, 2):
        for val1, val2 in product([0, 1], repeat=2):
            mask = (df_qca[c1] == val1) & (df_qca[c2] == val2)
            n = mask.sum()
            if n < min_cases:
                continue
            n_out = df_qca.loc[mask, outcome_col].sum()
            consistency = n_out / n
            label = f"{'~' if val1==0 else ''}{c1} * {'~' if val2==0 else ''}{c2}"
            results.append({
                "configuration": label,
                "n_cases": n,
                "consistency": consistency,
                "coverage": n_out / max(df_qca[outcome_col].sum(), 1),
            })

    return pd.DataFrame(results).sort_values("consistency", ascending=False)

suf_df = sufficiency_analysis(df_qca, conditions)
print("\n=== Top Sufficient Configurations ===")
print(suf_df[suf_df["consistency"] >= 0.80].head(8).round(3).to_string(index=False))

# -----------------------------------------------------------------
# Visualization: necessity-sufficiency plot
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Necessity plot
axes[0].scatter(nec_df["necessity_coverage"], nec_df["necessity_consistency"],
                s=80, c="steelblue", edgecolors="k")
for _, row in nec_df.iterrows():
    axes[0].annotate(row["condition"],
                     (row["necessity_coverage"], row["necessity_consistency"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
axes[0].axhline(0.90, color="red", ls="--", label="Threshold (0.90)")
axes[0].axvline(0.75, color="blue", ls="--", alpha=0.5)
axes[0].set_xlabel("Coverage"); axes[0].set_ylabel("Consistency")
axes[0].set_title("Necessity Analysis")
axes[0].legend()

# Truth table grid
cond_cols = list(conditions.keys())
pivot = truth_table.set_index(cond_cols + ["n_cases"])[["consistency", "outcome"]]
im = axes[1].imshow(truth_table[cond_cols].values.T, cmap="RdYlGn",
                    aspect="auto", vmin=0, vmax=1)
axes[1].set_yticks(range(len(cond_cols)))
axes[1].set_yticklabels(cond_cols)
axes[1].set_xlabel("Truth Table Row")
axes[1].set_title("Truth Table (Green=Present)")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.savefig("qca_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: qca_analysis.png")
```

### Step 3: Triangulation and Integration of Qual+Quant Findings

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -----------------------------------------------------------------
# Simulate an explanatory sequential design:
# Phase 1 (Quant): survey data on program satisfaction
# Phase 2 (Qual): interviews coded for themes
# Integration: do interview themes explain survey variation?
# -----------------------------------------------------------------
np.random.seed(42)
n_participants = 80

# --- Quantitative phase ---
satisfaction_score = np.random.normal(65, 15, n_participants)
years_experience = np.random.uniform(1, 20, n_participants)
training_hours = np.random.randint(0, 50, n_participants)
# True DGP
satisfaction_score = (50
                       + 0.5 * years_experience
                       + 0.3 * training_hours
                       + np.random.normal(0, 8, n_participants))
satisfaction_score = np.clip(satisfaction_score, 0, 100)

# --- Qualitative phase (interview themes for 20 purposively sampled participants) ---
# Select participants: 10 high-satisfaction, 10 low-satisfaction
n_qual = 20
high_sat_idx = np.argsort(satisfaction_score)[-10:]
low_sat_idx = np.argsort(satisfaction_score)[:10]
qual_sample_idx = np.concatenate([high_sat_idx, low_sat_idx])
qual_sat_level = np.array(["High"] * 10 + ["Low"] * 10)

# Coded themes (frequency per interview)
np.random.seed(42)
themes = {
    "SUPPORT": np.concatenate([
        np.random.randint(3, 8, 10),   # high sat: more support mentions
        np.random.randint(0, 3, 10)    # low sat: few support mentions
    ]),
    "CHALLENGE": np.concatenate([
        np.random.randint(0, 3, 10),   # high sat: fewer challenges
        np.random.randint(3, 8, 10)    # low sat: more challenges
    ]),
    "EMPOWERMENT": np.concatenate([
        np.random.randint(2, 6, 10),
        np.random.randint(0, 2, 10)
    ]),
}

df_qual = pd.DataFrame(themes)
df_qual["sat_level"] = qual_sat_level
df_qual["satisfaction"] = satisfaction_score[qual_sample_idx]

# -----------------------------------------------------------------
# Triangulation analysis
# -----------------------------------------------------------------
# Quant: regression of satisfaction on experience and training
X_q = sm.add_constant(pd.DataFrame({
    "experience": years_experience,
    "training": training_hours,
}))
ols_result = sm.OLS(satisfaction_score, X_q).fit(cov_type="HC3")
print("=== Phase 1: Quantitative Results ===")
print(ols_result.summary().tables[1])
print(f"Adjusted R²: {ols_result.rsquared_adj:.3f}")

# Qual: theme frequency difference by satisfaction group
print("\n=== Phase 2: Qualitative Theme Frequencies ===")
theme_summary = df_qual.groupby("sat_level")[list(themes.keys())].mean()
print(theme_summary.round(2))

# Integration: correlate theme frequencies with satisfaction scores
print("\n=== Integration: Theme-Satisfaction Correlations ===")
for theme in themes.keys():
    r, p = pearsonr(df_qual[theme], df_qual["satisfaction"])
    direction = "positive" if r > 0 else "negative"
    sig = "**" if p < 0.05 else " (ns)"
    print(f"  {theme} ↔ Satisfaction: r = {r:.3f}, p = {p:.3f} {sig} ({direction})")

# Joint display: triangulation convergence
print("\n=== Triangulation: Convergence Assessment ===")
print("Quant finding: Training hours → higher satisfaction (β > 0)")
print("Qual finding: SUPPORT theme frequent in high-satisfaction group")
print("→ CONVERGENCE: Supportive training environment drives satisfaction")
print()
print("Quant finding: Experience → higher satisfaction (β > 0)")
print("Qual finding: EMPOWERMENT theme frequent in high-satisfaction group")
print("→ CONVERGENCE: Experience builds empowerment/confidence")

# -----------------------------------------------------------------
# Visualization: Joint display
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Quant: scatter plot
axes[0, 0].scatter(training_hours, satisfaction_score, alpha=0.5, s=20,
                   c="steelblue")
xf = np.linspace(0, 50, 100)
axes[0, 0].plot(xf, ols_result.params["const"] + ols_result.params["training"] * xf
                + ols_result.params["experience"] * years_experience.mean(),
                "r-", lw=2, label=f"β={ols_result.params['training']:.2f}")
axes[0, 0].set_xlabel("Training Hours"); axes[0, 0].set_ylabel("Satisfaction")
axes[0, 0].set_title("Quant: Training → Satisfaction")
axes[0, 0].legend()

# Qual: theme frequency by group
x_pos = np.arange(len(themes))
bar_w = 0.35
axes[0, 1].bar(x_pos - bar_w/2, theme_summary.loc["High"], bar_w,
               label="High Satisfaction", color="steelblue")
axes[0, 1].bar(x_pos + bar_w/2, theme_summary.loc["Low"], bar_w,
               label="Low Satisfaction", color="orange")
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(list(themes.keys()))
axes[0, 1].set_ylabel("Mean Mentions per Interview")
axes[0, 1].set_title("Qual: Theme Frequency by Satisfaction Group")
axes[0, 1].legend()

# Integration: Joint display scatter
for theme, color in zip(themes.keys(), ["steelblue", "orange", "green"]):
    r, p = pearsonr(df_qual[theme], df_qual["satisfaction"])
    axes[1, 0].scatter(df_qual[theme], df_qual["satisfaction"],
                       color=color, alpha=0.6, s=40, label=f"{theme} (r={r:.2f})")
axes[1, 0].set_xlabel("Theme Mentions")
axes[1, 0].set_ylabel("Satisfaction Score")
axes[1, 0].set_title("Integration: Qual Themes vs. Quant Score")
axes[1, 0].legend(fontsize=8)

# Convergence matrix
methods = ["Quant Regression", "Qual Themes", "QCA Analysis"]
findings = ["Training Effect", "Support Context", "Leadership Necessity"]
convergence = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
im = axes[1, 1].imshow(convergence, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
axes[1, 1].set_xticks(range(3)); axes[1, 1].set_xticklabels(findings, rotation=20, ha="right")
axes[1, 1].set_yticks(range(3)); axes[1, 1].set_yticklabels(methods)
axes[1, 1].set_title("Triangulation Convergence Matrix")
plt.colorbar(im, ax=axes[1, 1], label="Convergent (1) / Divergent (0)")

plt.tight_layout()
plt.savefig("mixed_methods_integration.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: mixed_methods_integration.png")
```

## Advanced Usage

### Thematic Saturation Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def saturation_curve(code_assignments, window=3):
    """Compute qualitative saturation curve: new codes per interview.

    Args:
        code_assignments: list of sets; each set = codes in one interview
        window: rolling window for smoothing
    Returns:
        cumulative_codes, new_per_interview, saturation_point
    """
    seen_codes = set()
    cumulative = []
    new_per_interview = []

    for codes in code_assignments:
        new_codes = set(codes) - seen_codes
        seen_codes.update(codes)
        new_per_interview.append(len(new_codes))
        cumulative.append(len(seen_codes))

    # Saturation: point where rolling mean of new codes < 1
    rolling_mean = np.convolve(new_per_interview,
                                np.ones(window) / window, mode="valid")
    saturation_point = None
    for i, v in enumerate(rolling_mean):
        if v < 1.0:
            saturation_point = i + window
            break

    return cumulative, new_per_interview, saturation_point

np.random.seed(42)
n_interviews = 30
# Simulate code assignments: more novel codes early, fewer later
code_pool = [f"Code_{i:03d}" for i in range(40)]
interviews = []
for i in range(n_interviews):
    # Probability of drawing from new codes decreases over time
    n_new = max(0, int(np.random.poisson(max(1, 5 - i * 0.2))))
    n_repeat = np.random.randint(2, 6)
    available_new = [c for c in code_pool[:min(i*2 + 5, 40)]]
    drawn = set(np.random.choice(available_new, min(n_repeat + n_new, len(available_new)),
                                  replace=False).tolist())
    interviews.append(drawn)

cumulative, new_per, sat_pt = saturation_curve(interviews)
print(f"Saturation point reached at interview: {sat_pt}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, n_interviews + 1), cumulative, "o-", color="steelblue")
if sat_pt:
    axes[0].axvline(sat_pt, color="red", ls="--",
                     label=f"Saturation at n={sat_pt}")
axes[0].set_xlabel("Interview #"); axes[0].set_ylabel("Cumulative Unique Codes")
axes[0].set_title("Saturation Curve")
axes[0].legend()

axes[1].bar(range(1, n_interviews + 1), new_per, color="steelblue", alpha=0.7)
axes[1].set_xlabel("Interview #"); axes[1].set_ylabel("New Codes Introduced")
axes[1].set_title("New Codes per Interview")
if sat_pt:
    axes[1].axvline(sat_pt, color="red", ls="--")
plt.tight_layout()
plt.savefig("saturation_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved: saturation_curve.png")
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Kappa < 0.4 | Coders have different code definitions | Hold calibration session; clarify codebook; recode |
| QCA has many logical contradictions | Cases with same conditions but different outcomes | Increase conditions or use fuzzy-set QCA |
| No necessary conditions found | All single conditions inconsistency < 0.9 | Try 2-condition intersections; lower threshold to 0.85 |
| Triangulation shows divergence | Conflicting qual/quant results | Report divergence as finding; re-interview for explanation |
| Saturation never reached | Too few interviews or codes too fine-grained | Increase n, or collapse related codes |
| QCA truth table has many empty rows | Too many conditions (complexity) | Reduce to theory-guided subset of conditions |

## External Resources

- Creswell, J. W., & Plano Clark, V. L. (2018). *Designing and Conducting Mixed Methods Research*. SAGE.
- Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology*. SAGE.
- Ragin, C. C. (2008). *Redesigning Social Inquiry: Fuzzy Sets and Beyond*. University of Chicago Press.
- Rihoux, B., & Ragin, C. C. (2009). *Configurational Comparative Methods*. SAGE.
- [COMPASSS — QCA resources](https://www.compasss.org/)

## Examples

### Example 1: Convergent Design Dashboard

```python
import pandas as pd
import numpy as np

def convergent_design_summary(quant_results, qual_themes, n_quant, n_qual):
    """Generate a summary table for a convergent mixed methods study.

    Args:
        quant_results: dict of {construct: effect_size}
        qual_themes: dict of {construct: theme_description}
        n_quant, n_qual: sample sizes
    Returns:
        joint display DataFrame
    """
    rows = []
    for construct in set(list(quant_results.keys()) + list(qual_themes.keys())):
        quant_effect = quant_results.get(construct, "Not measured")
        qual_desc = qual_themes.get(construct, "Not explored")
        convergent = (construct in quant_results and construct in qual_themes)
        rows.append({
            "Construct": construct,
            "Quant Finding": quant_effect,
            "Qual Theme": qual_desc,
            "Convergent": "Yes" if convergent else "No",
        })
    return pd.DataFrame(rows)

quant = {"support": "β=0.42, p<0.01", "training": "β=0.31, p<0.05"}
qual = {"support": "Participants emphasized peer support", "trust": "Trust in leadership"}
summary = convergent_design_summary(quant, qual, n_quant=150, n_qual=25)
print(summary.to_string(index=False))
```

### Example 2: Fuzzy-Set QCA (fsQCA) Simulation

```python
import numpy as np
import pandas as pd

def fsqca_analysis(df, conditions, outcome, threshold=0.8):
    """Compute fuzzy-set sufficiency for single conditions.

    Args:
        df: DataFrame with fuzzy-set membership scores (0-1)
        conditions: list of condition column names
        outcome: outcome column name
        threshold: minimum consistency for sufficiency
    Returns:
        DataFrame with sufficiency statistics
    """
    o = df[outcome].values
    results = []
    for cond in conditions:
        c = df[cond].values
        # Fuzzy subset: C → O: consistency = sum(min(C,O)) / sum(C)
        consistency = np.minimum(c, o).sum() / max(c.sum(), 1e-6)
        # Coverage
        coverage = np.minimum(c, o).sum() / max(o.sum(), 1e-6)
        results.append({"condition": cond, "consistency": consistency,
                        "coverage": coverage,
                        "sufficient": consistency >= threshold})
    return pd.DataFrame(results).sort_values("consistency", ascending=False)

np.random.seed(42)
n = 25
df_fs = pd.DataFrame({
    "leadership": np.random.beta(3, 2, n),
    "resources": np.random.beta(2, 2, n),
    "stakeholders": np.random.beta(2, 3, n),
    "outcome": np.random.beta(2, 2, n),
})

fsqca = fsqca_analysis(df_fs, ["leadership", "resources", "stakeholders"], "outcome")
print("=== Fuzzy-Set QCA: Single-Condition Sufficiency ===")
print(fsqca.round(3).to_string(index=False))
```
