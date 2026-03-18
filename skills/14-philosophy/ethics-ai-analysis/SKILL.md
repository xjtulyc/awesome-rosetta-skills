---
name: ethics-ai-analysis
description: >
  Use this Skill for AI ethics analysis: algorithmic fairness metrics,
  bias auditing, value alignment frameworks, and moral philosophy reasoning.
tags:
  - philosophy
  - ai-ethics
  - fairness
  - bias-auditing
  - alignment
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
    - scipy>=1.11
    - matplotlib>=3.7
    - fairlearn>=0.10
last_updated: "2026-03-17"
status: "stable"
---

# AI Ethics Analysis

> **One-line summary**: Audit ML models for algorithmic fairness violations using demographic parity, equalized odds, and calibration metrics; apply philosophical frameworks to AI value alignment problems.

---

## When to Use This Skill

- When measuring demographic parity, equalized odds, or calibration gaps in models
- When auditing hiring, credit, or criminal justice AI systems for disparate impact
- When analyzing trade-offs between competing fairness notions
- When applying utilitarian, deontological, or virtue ethics frameworks to AI decisions
- When designing fairness-aware reweighting or threshold-adjustment interventions
- When writing ethics impact assessments for AI systems

**Trigger keywords**: AI ethics, algorithmic fairness, bias audit, demographic parity, equalized odds, disparate impact, fairness metric, value alignment, moral philosophy, utilitarian, deontological, discrimination, protected attribute, AIF360, fairlearn

---

## Background & Key Concepts

### Fairness Definitions

| Metric | Definition | Formula |
|:-------|:-----------|:--------|
| Demographic Parity | Equal selection rates | $P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)$ |
| Equal Opportunity | Equal TPR across groups | $P(\hat{Y}=1|Y=1, A=0) = P(\hat{Y}=1|Y=1, A=1)$ |
| Equalized Odds | Equal TPR and FPR | Both TPR and FPR equal across groups |
| Calibration | Equal predicted probabilities match actual rates | $P(Y=1|\hat{p}=v, A=a) = v$ for all $a$ |

**Impossibility theorem** (Chouldechova, 2017): When base rates differ, no classifier can simultaneously satisfy calibration, equal FPR, and equal FPR. Choose the metric appropriate to the decision context.

### Disparate Impact (80% Rule)

EEOC 4/5ths rule: selection rate for any protected group should not be less than 80% of the rate for the highest-selected group.

$$
DI = \frac{P(\hat{Y}=1|A=\text{protected})}{P(\hat{Y}=1|A=\text{majority})}
$$

---

## Environment Setup

### Install Dependencies

```bash
pip install pandas>=2.0 numpy>=1.24 scikit-learn>=1.3 scipy>=1.11 \
            matplotlib>=3.7 fairlearn>=0.10
```

### Verify Installation

```python
import fairlearn
import numpy as np
print(f"fairlearn version: {fairlearn.__version__}")

# Quick test
from fairlearn.metrics import demographic_parity_difference
import pandas as pd

y_true = np.array([1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1])
groups = np.array([0, 0, 0, 1, 1, 1])
dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
print(f"Demographic parity difference: {dpd:.4f}")
```

---

## Core Workflow

### Step 1: Fairness Audit of a Binary Classifier

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from scipy import stats

# ------------------------------------------------------------------ #
# Simulate a credit scoring scenario with demographic disparities
# ------------------------------------------------------------------ #

np.random.seed(42)
n = 5000

# Demographic group (0 = majority, 1 = minority)
group = np.random.choice([0, 1], n, p=[0.70, 0.30])

# Features correlated with outcome AND group (creating proxy discrimination)
income  = np.where(group == 0, np.random.normal(55000, 15000, n), np.random.normal(40000, 12000, n))
credit  = np.where(group == 0, np.random.normal(700, 80, n), np.random.normal(650, 90, n))
history = np.random.randint(0, 10, n) + group * np.random.randint(-2, 1, n)

# True outcome (loan repayment)
log_odds = -3 + 0.00003*income + 0.004*credit + 0.2*history
prob = 1 / (1 + np.exp(-log_odds))
y = (np.random.uniform(size=n) < prob).astype(int)

X = pd.DataFrame({'income': income, 'credit_score': credit, 'credit_history': history})
df = pd.DataFrame({'group': group, 'y': y})
df = pd.concat([df, X], axis=1)

# ---- Train a biased classifier (includes group as implicit via proxies) #
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, df['group'], test_size=0.30, random_state=42
)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print(f"Overall AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Overall accuracy: {np.mean(y_pred == y_test)*100:.2f}%")

# ---- Compute fairness metrics ----------------------------------- #
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equalized_odds_difference,
        selection_rate,
        true_positive_rate,
        false_positive_rate,
        MetricFrame,
    )

    mf = MetricFrame(
        metrics={
            'accuracy':  lambda y_t, y_p: np.mean(y_t == y_p),
            'selection_rate': selection_rate,
            'TPR': true_positive_rate,
            'FPR': false_positive_rate,
            'AUC': roc_auc_score,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=g_test,
        y_pred_additional={'AUC': y_prob},
    )

    print("\nFairness Audit Results by Group:")
    print(mf.by_group.round(4))

    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=g_test)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=g_test)

    print(f"\nDemographic Parity Difference: {dpd:.4f}  (|≤0.05| = fair)")
    print(f"Equalized Odds Difference:      {eod:.4f}  (|≤0.05| = fair)")

    # 80% / 4-5ths rule (EEOC disparate impact)
    sr_majority  = selection_rate(y_test[g_test==0], y_pred[g_test==0])
    sr_minority  = selection_rate(y_test[g_test==1], y_pred[g_test==1])
    di_ratio = sr_minority / sr_majority
    print(f"\nDisparate Impact ratio: {di_ratio:.4f}  (>0.80 = passes 4/5ths rule)")
    print(f"{'PASSES' if di_ratio >= 0.80 else 'FAILS'} EEOC 4/5ths rule")

except ImportError:
    print("fairlearn not installed; using manual calculations")
    for g_val, g_name in [(0, 'Majority'), (1, 'Minority')]:
        mask = g_test.values == g_val
        acc  = np.mean(y_pred[mask] == y_test.values[mask])
        sel  = y_pred[mask].mean()
        auc  = roc_auc_score(y_test.values[mask], y_prob[mask])
        tp_mask = y_test.values[mask] == 1
        tpr  = y_pred[mask][tp_mask].mean() if tp_mask.sum() > 0 else 0
        fp_mask = y_test.values[mask] == 0
        fpr  = y_pred[mask][fp_mask].mean() if fp_mask.sum() > 0 else 0
        print(f"{g_name}: acc={acc:.3f}, sel={sel:.3f}, TPR={tpr:.3f}, FPR={fpr:.3f}, AUC={auc:.3f}")

# ---- Plot fairness metrics -------------------------------------- #
metrics = ['Accuracy', 'Selection rate', 'TPR', 'FPR']
groups_list = [0, 1]
group_names = ['Majority', 'Minority']

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, metric in zip(axes.flat, metrics):
    values = []
    for g in groups_list:
        mask = g_test.values == g
        if metric == 'Accuracy':
            v = np.mean(y_pred[mask] == y_test.values[mask])
        elif metric == 'Selection rate':
            v = y_pred[mask].mean()
        elif metric == 'TPR':
            tp_m = y_test.values[mask] == 1
            v = y_pred[mask][tp_m].mean() if tp_m.sum() > 0 else 0
        else:  # FPR
            fp_m = y_test.values[mask] == 0
            v = y_pred[mask][fp_m].mean() if fp_m.sum() > 0 else 0
        values.append(v)

    bars = ax.bar(group_names, values, color=['#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=0.7)
    ax.set_title(metric); ax.set_ylabel(metric); ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha='center', fontsize=10)

plt.suptitle("Fairness Audit: Credit Scoring Model by Demographic Group")
plt.tight_layout()
plt.savefig("fairness_audit.png", dpi=150)
plt.show()
```

### Step 2: Fairness Interventions (Threshold Adjustment)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ------------------------------------------------------------------ #
# Post-processing threshold adjustment for equalized odds
# (Hardt et al., 2016)
# ------------------------------------------------------------------ #

def find_threshold_equalized_odds(y_true, y_prob, sensitive, majority=0, minority=1):
    """
    Find group-specific classification thresholds to achieve equalized odds.
    Returns (threshold_majority, threshold_minority).
    """
    from sklearn.metrics import roc_curve

    # ROC curves by group
    fpr_maj, tpr_maj, thr_maj = roc_curve(
        y_true[sensitive==majority], y_prob[sensitive==majority])
    fpr_min, tpr_min, thr_min = roc_curve(
        y_true[sensitive==minority], y_prob[sensitive==minority])

    # Target operating point: equalize TPR at 0.70
    target_tpr = 0.70

    def find_threshold_at_tpr(fpr, tpr, thr, target):
        idx = np.argmin(np.abs(tpr - target))
        return thr[idx], fpr[idx], tpr[idx]

    t_maj, fpr_at_t_maj, tpr_at_t_maj = find_threshold_at_tpr(fpr_maj, tpr_maj, thr_maj, target_tpr)
    t_min, fpr_at_t_min, tpr_at_t_min = find_threshold_at_tpr(fpr_min, tpr_min, thr_min, target_tpr)

    return t_maj, t_min, (fpr_at_t_maj, tpr_at_t_maj), (fpr_at_t_min, tpr_at_t_min)

g_test_arr = g_test.values
t_maj, t_min, op_maj, op_min = find_threshold_equalized_odds(
    y_test.values, y_prob, g_test_arr)

# Apply group-specific thresholds
y_pred_adj = np.where(g_test_arr == 0,
                       (y_prob >= t_maj).astype(int),
                       (y_prob >= t_min).astype(int))

# Compare original vs. adjusted
for method, preds, name in [(y_pred, 'Original (uniform threshold)'),
                             (y_pred_adj, 'Adjusted (group thresholds)')]:
    print(f"\n{name}:")
    for g, gn in [(0, 'Majority'), (1, 'Minority')]:
        mask = g_test_arr == g
        tp_mask = y_test.values[mask] == 1
        tpr = preds[mask][tp_mask].mean() if tp_mask.sum() > 0 else 0
        sel = preds[mask].mean()
        acc = np.mean(preds[mask] == y_test.values[mask])
        print(f"  {gn}: sel={sel:.3f}, TPR={tpr:.3f}, acc={acc:.3f}")

# Plot ROC curves with operating points
fig, ax = plt.subplots(figsize=(7, 6))
from sklearn.metrics import roc_curve as roc_c
for g, gn, color in [(0,'Majority','#3498db'), (1,'Minority','#e74c3c')]:
    fpr, tpr, _ = roc_c(y_test.values[g_test_arr==g], y_prob[g_test_arr==g])
    ax.plot(fpr, tpr, color=color, linewidth=2, label=gn)
    if g == 0:
        ax.plot(op_maj[0], op_maj[1], 'o', color=color, markersize=10,
                label=f'{gn} threshold (TPR={op_maj[1]:.2f})')
    else:
        ax.plot(op_min[0], op_min[1], 's', color=color, markersize=10,
                label=f'{gn} threshold (TPR={op_min[1]:.2f})')
ax.plot([0,1],[0,1],'k--',alpha=0.5)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curves — Equalized Odds Adjustment")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("equalized_odds.png", dpi=150); plt.show()
```

### Step 3: Philosophical Framework Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Apply three moral frameworks to a concrete AI ethics scenario:
# Should an AI parole board system deny release to a high-risk individual?
# ------------------------------------------------------------------ #

print("""
=== AI Ethics Analysis: Automated Parole Decision System ===

SCENARIO: An AI system predicts recidivism risk. A defendant
is classified as "high risk" (predicted probability = 0.72).
The base rate for this demographic group is 0.35. The model
has a false positive rate of 0.28 for this group.

QUESTION: Should the parole board rely on this AI recommendation?
""")

frameworks = {
    "Utilitarian Analysis": {
        "core_principle": "Maximize aggregate welfare; minimize harm across all stakeholders.",
        "relevant_facts": [
            "If high-risk prediction correct (72% confidence): denying parole prevents ~0.72 recidivism events",
            "False positive rate 28%: 28% of actual non-recidivists in this group are incorrectly flagged",
            "Incarceration costs ~$35,000/year; victim costs of crime vary widely",
            "Evidence suggests incarceration has mixed effects on recidivism long-term",
        ],
        "analysis": """
        Expected utility calculation:
        - Let p=0.72 (predicted recid. prob), p_true≈0.35 (base rate)
        - Expected harm prevented = p_true × harm_of_crime
        - Expected harm imposed   = (1-p_true) × cost_of_unjust_incarceration

        At base rate 0.35: denying parole has 65% chance of being wrong.
        Utilitarian calculation depends on harm weights — not obvious that denial maximizes welfare.
        HIGH-RISK DESIGNATION IS NOT SUFFICIENT for utilitarian justification.
        """,
        "verdict": "CONTEXT-DEPENDENT (depends on harm magnitudes)",
    },
    "Deontological Analysis": {
        "core_principle": "Respect persons as ends; follow categorical duties regardless of outcomes.",
        "relevant_facts": [
            "Individual has right to liberty and due process",
            "Statistical prediction punishes based on group membership, not individual action",
            "Kant's categorical imperative: act only on maxims you could universalize",
            "False positives violate individuals' rights even if outcomes-good aggregate",
        ],
        "analysis": """
        Kantian objections to statistical risk prediction:
        1. Treats individuals as means (to aggregate crime reduction) not ends
        2. Statistical group-based predictions cannot be universalized without contradiction
        3. Due process requires individual-level evidence, not probabilistic grouping
        4. High FPR (28%) means systematic rights violations for the innocent

        The Rawlsian difference principle: acceptable only if worst-off benefit.
        Biased FPR (if higher for minorities) fails Rawlsian justice.
        """,
        "verdict": "REJECT AI-ONLY DECISION (rights violations are impermissible)",
    },
    "Virtue Ethics Analysis": {
        "core_principle": "What would a person of good character (prudent, just, compassionate) do?",
        "relevant_facts": [
            "A just judge weighs individual circumstances, not just actuarial scores",
            "Prudence requires understanding model limitations and error rates",
            "Compassion requires considering the human cost of false positives",
            "Epistemic humility: model ≠ ground truth about future behavior",
        ],
        "analysis": """
        Virtues applied:
        - Justice: mechanical application of risk score without deliberation lacks justice
        - Prudence: a prudent judge uses the score as one input, not the final word
        - Integrity: honest acknowledgment that 28% FPR means significant uncertainty
        - Compassion: humanizes the defendant beyond statistical categories

        The virtuous judge treats AI as a tool for reflection, not delegation.
        """,
        "verdict": "USE AS ADVISORY INPUT ONLY (not decisive)",
    },
}

for name, framework in frameworks.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Core principle: {framework['core_principle']}")
    print("\nKey facts:")
    for fact in framework['relevant_facts']:
        print(f"  • {fact}")
    print(f"\nAnalysis: {framework['analysis']}")
    print(f"\nVERDICT: {framework['verdict']}")

# ---- Summary visualization -------------------------------------- #
fig, ax = plt.subplots(figsize=(10, 4))
framework_names = list(frameworks.keys())
verdicts = [f['verdict'].split('(')[0].strip() for f in frameworks.values()]
colors = ['#f39c12', '#e74c3c', '#2ecc71']

y_pos = range(len(framework_names))
for pos, (name, verdict, color) in enumerate(zip(framework_names, verdicts, colors)):
    ax.barh(pos, 1, left=0, color=color, height=0.6, alpha=0.8, edgecolor='black', linewidth=0.7)
    ax.text(0.5, pos, verdict, ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels([n.replace(' Analysis','') for n in framework_names], fontsize=10)
ax.set_xlim(0, 1); ax.set_xticks([])
ax.set_title("Philosophical Framework Analysis:\nAI Parole Decision System")
ax.grid(False)
plt.tight_layout()
plt.savefig("ethics_frameworks.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Fairness-Accuracy Trade-Off Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Visualize accuracy vs. fairness trade-off curve
# ------------------------------------------------------------------ #

thresholds = np.linspace(0.01, 0.99, 99)
majority_mask = g_test_arr == 0
minority_mask = g_test_arr == 1

acc_vals, dpd_vals, eod_vals = [], [], []
for t in thresholds:
    y_t = (y_prob >= t).astype(int)
    acc = np.mean(y_t == y_test.values)
    sel_maj = y_t[majority_mask].mean()
    sel_min = y_t[minority_mask].mean()
    dp_diff = abs(sel_maj - sel_min)
    acc_vals.append(acc); dpd_vals.append(dp_diff)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(dpd_vals, acc_vals, c=thresholds, cmap='viridis', s=20, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Classification threshold')
ax.axvline(0.05, color='red', linestyle='--', linewidth=1.5, label='|DP difference| = 0.05')
ax.set_xlabel("Demographic Parity |Difference|")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy–Fairness Trade-off (Varying Threshold)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fairness_accuracy_tradeoff.png", dpi=150); plt.show()
```

---

## Troubleshooting

### Error: `fairlearn.metrics.demographic_parity_difference` signature changed

**Fix**: Check version — fairlearn 0.10 changed some function signatures:
```python
# fairlearn >= 0.10
from fairlearn.metrics import demographic_parity_difference
dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=group)
```

### Impossibility theorem confusion

When fairness metrics conflict, it's expected (not a code bug):
```python
# If base rates differ significantly, equalized odds AND calibration CANNOT both hold
print(f"Group 0 base rate: {y_test.values[g_test_arr==0].mean():.3f}")
print(f"Group 1 base rate: {y_test.values[g_test_arr==1].mean():.3f}")
# Large difference → fairness metrics will be in tension
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| fairlearn | 0.10 | API changed significantly from 0.7 to 0.10 |
| scikit-learn | 1.3, 1.4 | Required by fairlearn |
| scipy | 1.11, 1.12 | `stats.chi2_contingency` stable |

---

## External Resources

### Official Documentation

- [fairlearn documentation](https://fairlearn.org)
- [AI Fairness 360 (AIF360) by IBM](https://aif360.res.ibm.com)

### Key Papers

- Chouldechova, A. (2017). *Fair prediction with disparate impact*. Big Data.
- Hardt, M., Price, E. & Srebro, N. (2016). *Equality of opportunity in supervised learning*. NeurIPS.
- Dwork, C. et al. (2012). *Fairness through awareness*. ITCS.

---

## Examples

### Example 1: Counterfactual Fairness Check

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Counterfactual fairness: would the outcome change if group membership were flipped?
np.random.seed(42)
X_cf = X_test.copy()

y_pred_cf_0 = clf.predict_proba(X_cf[g_test_arr==1])[:,1]  # Minority features
y_pred_orig_1 = y_prob[g_test_arr==1]

# How much does predicted probability change on average?
cf_change = np.abs(y_pred_cf_0 - y_pred_orig_1)
print(f"Mean counterfactual prediction change (minority): {cf_change.mean():.4f}")
print(f"Counterfactually fair if ~0; actual: {cf_change.mean():.4f}")
```

### Example 2: SHAP Values for Bias Attribution

```python
import numpy as np

# Without SHAP: approximate feature importance for bias attribution
coefs = dict(zip(X_train.columns, clf.coef_[0]))
print("Logistic regression coefficients (proxy for feature importance):")
for feat, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat}: {coef:+.4f}")
print("\nNote: high coefficient for 'income' or 'credit_score' may encode")
print("indirect discrimination if these features are correlated with protected group.")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
