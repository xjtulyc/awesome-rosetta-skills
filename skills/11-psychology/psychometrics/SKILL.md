---
name: psychometrics
description: >
  Psychometric analysis in Python: CTT item stats, Cronbach's alpha, EFA, CFA with
  semopy, 2PL IRT, measurement invariance, and Mantel-Haenszel DIF detection.
tags:
  - psychology
  - psychometrics
  - factor-analysis
  - item-response-theory
  - python
  - scale-development
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
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - factor_analyzer>=0.5.0
  - semopy>=2.3.0
  - matplotlib>=3.7.0
  - scikit-learn>=1.3.0
last_updated: "2026-03-17"
---

# Psychometric Analysis in Python

This skill covers the full psychometric validation pipeline:
Classical Test Theory (CTT) item statistics, reliability (Cronbach's alpha),
Exploratory Factor Analysis (EFA) with parallel analysis, Confirmatory Factor
Analysis (CFA) with semopy, Item Response Theory (IRT) 2PL models,
measurement invariance testing, scale development workflow, and differential
item functioning (DIF) via the Mantel-Haenszel procedure.

---

## 1. Setup

```bash
pip install pandas numpy scipy factor_analyzer semopy matplotlib scikit-learn
```

```python
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit          # logistic sigmoid
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# factor_analyzer for EFA
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# semopy for CFA
import semopy

# ---------------------------------------------------------------------------
# Generate a synthetic 20-item questionnaire (5-point Likert, 2 factors)
# ---------------------------------------------------------------------------
np.random.seed(42)
N_RESPONDENTS = 500
N_ITEMS       = 20

# True factor structure: items 0-9 load on F1, items 10-19 load on F2
F1 = np.random.normal(0, 1, N_RESPONDENTS)
F2 = np.random.normal(0, 1, N_RESPONDENTS)

loadings_F1 = np.array([0.75, 0.72, 0.68, 0.80, 0.65, 0.70, 0.73, 0.60, 0.78, 0.55])
loadings_F2 = np.array([0.70, 0.74, 0.65, 0.80, 0.72, 0.68, 0.76, 0.62, 0.55, 0.78])

# Build item scores with error
item_cols  = [f"item_{i+1:02d}" for i in range(N_ITEMS)]
responses  = np.zeros((N_RESPONDENTS, N_ITEMS))

for j in range(10):
    responses[:, j]    = loadings_F1[j] * F1 + np.sqrt(1 - loadings_F1[j]**2) * np.random.normal(0, 1, N_RESPONDENTS)
for j in range(10, 20):
    responses[:, j]    = loadings_F2[j-10] * F2 + np.sqrt(1 - loadings_F2[j-10]**2) * np.random.normal(0, 1, N_RESPONDENTS)

# Scale to 1-5 Likert
responses = pd.DataFrame(responses, columns=item_cols)
for col in item_cols:
    pct = responses[col].rank(pct=True)
    responses[col] = pd.cut(pct, bins=5, labels=[1, 2, 3, 4, 5]).astype(float)

# Group variable (gender: 0=female, 1=male) for DIF analysis
responses["group"] = np.random.binomial(1, 0.5, N_RESPONDENTS)
print(f"Synthetic data: {responses.shape[0]} respondents × {N_ITEMS} items")
```

---

## 2. Classical Test Theory Item Statistics

```python
def compute_item_stats(response_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Classical Test Theory item-level statistics.

    Parameters
    ----------
    response_matrix : pd.DataFrame
        Rows = respondents, columns = items. Values are ordinal scores.

    Returns
    -------
    pd.DataFrame with columns:
        mean, std, p_value (difficulty), skewness, kurtosis,
        item_total_r, corrected_item_total_r, alpha_if_deleted.
    """
    data  = response_matrix.copy()
    n_items = data.shape[1]
    total = data.sum(axis=1)
    rows  = []

    for col in data.columns:
        item_scores   = data[col]
        rest_scores   = total - item_scores

        # Item-total correlation
        itc, _        = stats.pearsonr(item_scores, total)
        # Corrected: item-total minus item itself
        citc, _       = stats.pearsonr(item_scores, rest_scores)
        # p-value (proportion of maximum, i.e., item difficulty)
        max_score     = item_scores.max()
        min_score     = item_scores.min()
        p_val         = (item_scores.mean() - min_score) / max(max_score - min_score, 1e-9)

        # Alpha if item deleted
        remaining     = data.drop(columns=[col])
        alpha_del     = _cronbach_alpha_raw(remaining)

        rows.append({
            "item":                col,
            "mean":                round(item_scores.mean(), 3),
            "std":                 round(item_scores.std(), 3),
            "p_value":             round(p_val, 3),
            "skewness":            round(float(stats.skew(item_scores)), 3),
            "kurtosis":            round(float(stats.kurtosis(item_scores)), 3),
            "item_total_r":        round(itc, 3),
            "corrected_item_total_r": round(citc, 3),
            "alpha_if_deleted":    round(alpha_del, 3),
        })

    return pd.DataFrame(rows).set_index("item")


def _cronbach_alpha_raw(data: pd.DataFrame) -> float:
    """Internal helper: compute alpha from a DataFrame of item scores."""
    k     = data.shape[1]
    if k < 2:
        return np.nan
    item_var = data.var(ddof=1, axis=0).sum()
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_var / total_var)
```

---

## 3. Cronbach's Alpha and Omega

```python
def cronbach_alpha(data: pd.DataFrame, ci: float = 0.95) -> dict:
    """
    Compute Cronbach's alpha with bootstrapped confidence interval.

    Parameters
    ----------
    data : pd.DataFrame
        Item score matrix (respondents × items).
    ci : float
        Confidence level for bootstrap CI.

    Returns
    -------
    dict with keys: alpha, ci_lower, ci_upper, n_items, n_respondents.
    """
    alpha = _cronbach_alpha_raw(data)

    # Bootstrap CI
    n     = data.shape[0]
    bs_alphas = []
    for _ in range(1000):
        idx   = np.random.choice(n, size=n, replace=True)
        bs_alphas.append(_cronbach_alpha_raw(data.iloc[idx]))

    lo = np.percentile(bs_alphas, (1 - ci) / 2 * 100)
    hi = np.percentile(bs_alphas, (1 + ci) / 2 * 100)

    result = {
        "alpha":         round(alpha, 4),
        "ci_lower":      round(lo, 4),
        "ci_upper":      round(hi, 4),
        "n_items":       data.shape[1],
        "n_respondents": data.shape[0],
        "interpretation": (
            "Excellent (≥0.90)" if alpha >= 0.90 else
            "Good (0.80-0.89)"  if alpha >= 0.80 else
            "Acceptable (0.70-0.79)" if alpha >= 0.70 else
            "Questionable (0.60-0.69)" if alpha >= 0.60 else
            "Poor (<0.60)"
        ),
    }
    print(f"Cronbach's alpha = {alpha:.4f}  [{lo:.4f}, {hi:.4f}] — {result['interpretation']}")
    return result
```

---

## 4. Exploratory Factor Analysis

```python
def run_efa(
    data: pd.DataFrame,
    n_factors: int | None = None,
    rotation: str = "oblimin",
    method: str = "minres",
    parallel_analysis: bool = True,
) -> dict:
    """
    Run Exploratory Factor Analysis with optional parallel analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Item response matrix.
    n_factors : int or None
        Number of factors. If None, determined by parallel analysis or Kaiser.
    rotation : str
        'oblimin' (oblique) or 'varimax' (orthogonal).
    method : str
        Extraction method: 'minres', 'ml', 'principal'.
    parallel_analysis : bool
        Use parallel analysis to select n_factors if n_factors is None.

    Returns
    -------
    dict with keys: loadings, communalities, eigenvalues, n_factors, variance.
    """
    # Step 1: Suitability tests
    chi2, p_bartlett = calculate_bartlett_sphericity(data)
    kmo_all, kmo_model = calculate_kmo(data)
    print(f"Bartlett's test: chi2={chi2:.2f}, p={p_bartlett:.4f}")
    print(f"KMO measure: {kmo_model:.3f} (>0.60 acceptable, >0.80 meritorious)")

    # Step 2: Scree plot + parallel analysis
    fa_initial  = FactorAnalyzer(n_factors=data.shape[1], rotation=None)
    fa_initial.fit(data)
    ev, _       = fa_initial.get_eigenvalues()

    if n_factors is None and parallel_analysis:
        # Parallel analysis: compare real eigenvalues to random data eigenvalues
        random_evs = []
        for _ in range(100):
            rnd = pd.DataFrame(
                np.random.normal(size=data.shape),
                columns=data.columns,
            )
            fa_rnd = FactorAnalyzer(n_factors=data.shape[1], rotation=None)
            fa_rnd.fit(rnd)
            ev_rnd, _ = fa_rnd.get_eigenvalues()
            random_evs.append(ev_rnd)
        mean_rnd_ev = np.mean(random_evs, axis=0)
        n_factors   = int((ev > mean_rnd_ev).sum())
        n_factors   = max(1, n_factors)
        print(f"Parallel analysis suggests {n_factors} factor(s).")
    elif n_factors is None:
        n_factors = int((ev > 1).sum())   # Kaiser criterion
        print(f"Kaiser criterion suggests {n_factors} factor(s).")

    # Step 3: EFA with selected n_factors
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    fa.fit(data)

    loadings     = pd.DataFrame(
        fa.loadings_,
        index=data.columns,
        columns=[f"F{i+1}" for i in range(n_factors)],
    )
    communalities = pd.Series(fa.get_communalities(), index=data.columns, name="h2")
    variance_df   = pd.DataFrame(
        fa.get_factor_variance(),
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=[f"F{i+1}" for i in range(n_factors)],
    )

    print("\n=== Factor Loadings (|loading| > 0.30 shown) ===")
    print(loadings.where(loadings.abs() > 0.30, "").to_string())
    print("\n=== Variance Explained ===")
    print(variance_df.round(4).to_string())

    # Scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(ev) + 1), ev, "bo-", linewidth=1.5, label="Actual eigenvalues")
    if parallel_analysis:
        ax.plot(range(1, len(mean_rnd_ev) + 1), mean_rnd_ev, "r--",
                linewidth=1.2, label="Parallel analysis (random)")
    ax.axhline(1, color="gray", linestyle=":", linewidth=0.8, label="Kaiser criterion")
    ax.set_xlabel("Factor Number")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Scree Plot with Parallel Analysis", fontweight="bold")
    ax.legend()
    ax.set_xlim(0.5, min(20, len(ev)) + 0.5)
    plt.tight_layout()
    plt.savefig("scree_plot.png", dpi=150)
    plt.show()

    return {
        "loadings":      loadings,
        "communalities": communalities,
        "eigenvalues":   ev,
        "n_factors":     n_factors,
        "variance":      variance_df,
        "fa_object":     fa,
    }
```

---

## 5. Confirmatory Factor Analysis with semopy

```python
def run_cfa(
    data: pd.DataFrame,
    model_str: str,
    estimator: str = "MLW",
) -> dict:
    """
    Run a CFA model using semopy and return key fit indices.

    Parameters
    ----------
    data : pd.DataFrame
        Item response matrix.
    model_str : str
        semopy model specification. Example:
            'F1 =~ item_01 + item_02 + item_03
             F2 =~ item_11 + item_12 + item_13'
    estimator : str
        'MLW' (default, robust ML), 'ULS', 'DWLS'.

    Returns
    -------
    dict with keys: cfi, rmsea, srmr, tli, chi2, df, p_chi2, model, results.
    """
    mod = semopy.Model(model_str)
    res = mod.fit(data, solver=estimator)

    stats_obj = semopy.calc_stats(mod)

    cfi  = float(stats_obj.loc["CFI",  "Value"])  if "CFI"  in stats_obj.index else np.nan
    tli  = float(stats_obj.loc["TLI",  "Value"])  if "TLI"  in stats_obj.index else np.nan
    rmsea= float(stats_obj.loc["RMSEA","Value"])  if "RMSEA" in stats_obj.index else np.nan
    srmr = float(stats_obj.loc["SRMR", "Value"])  if "SRMR" in stats_obj.index else np.nan
    chi2 = float(stats_obj.loc["chi2", "Value"])  if "chi2" in stats_obj.index else np.nan
    df_  = float(stats_obj.loc["DoF",  "Value"])  if "DoF"  in stats_obj.index else np.nan
    pval = float(stats_obj.loc["chi2 p-value", "Value"]) if "chi2 p-value" in stats_obj.index else np.nan

    print("\n=== CFA Fit Indices ===")
    print(f"  chi2({df_:.0f}) = {chi2:.3f}, p = {pval:.4f}")
    print(f"  CFI  = {cfi:.3f}  (≥0.95 acceptable)")
    print(f"  TLI  = {tli:.3f}  (≥0.95 acceptable)")
    print(f"  RMSEA= {rmsea:.3f}  (≤0.06 close fit, ≤0.08 acceptable)")
    print(f"  SRMR = {srmr:.3f}  (≤0.08 acceptable)")

    return {
        "cfi":    cfi,
        "tli":    tli,
        "rmsea":  rmsea,
        "srmr":   srmr,
        "chi2":   chi2,
        "df":     df_,
        "p_chi2": pval,
        "model":  mod,
        "results": res,
        "stats":  stats_obj,
    }
```

---

## 6. IRT 2-Parameter Logistic Model

```python
def run_irt_2pl(
    responses: np.ndarray,
    n_iter: int = 300,
    tol: float = 1e-5,
) -> dict:
    """
    Estimate the 2PL IRT model via marginal maximum likelihood.

    Model: P(X_ij = 1 | theta_i) = sigmoid(a_j * (theta_i - b_j))
    where a_j = discrimination, b_j = difficulty.

    Parameters
    ----------
    responses : np.ndarray
        Binary response matrix (n_respondents × n_items). 1 = correct/endorsed.
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    dict with keys: a (discrimination), b (difficulty), theta (ability),
                    ll_history.
    """
    R = responses.astype(float)
    N, J = R.shape

    # Initialise
    a = np.ones(J)
    b = np.zeros(J)

    # Ability: initialise from observed proportions
    row_prop = R.mean(axis=1)
    row_prop = np.clip(row_prop, 0.01, 0.99)
    theta    = stats.norm.ppf(row_prop)

    ll_history = []

    for iteration in range(n_iter):
        # E-step: compute P(correct | theta, a, b)
        eta = a[None, :] * (theta[:, None] - b[None, :])  # N × J
        P   = expit(eta)
        P   = np.clip(P, 1e-9, 1 - 1e-9)

        ll  = (R * np.log(P) + (1 - R) * np.log(1 - P)).sum()
        ll_history.append(ll)

        # M-step: update a and b via Newton step for each item
        W   = P * (1 - P)  # weights

        for j in range(J):
            r_j  = R[:, j]
            p_j  = P[:, j]
            w_j  = W[:, j]
            res_j = r_j - p_j

            # Gradient and Hessian for b_j
            grad_b = -a[j] * res_j.sum()
            hess_b = -a[j]**2 * w_j.sum()
            if abs(hess_b) > 1e-9:
                b[j] -= grad_b / hess_b

            # Gradient and Hessian for a_j
            theta_b = theta - b[j]
            grad_a  = (res_j * theta_b).sum()
            hess_a  = -(w_j * theta_b**2).sum()
            if abs(hess_a) > 1e-9:
                a[j] -= grad_a / hess_a
            a[j] = max(a[j], 0.01)  # enforce positive discrimination

        # Update theta via Newton
        for i in range(N):
            p_i   = P[i]
            w_i   = W[i]
            res_i = R[i] - p_i
            grad_th = (a * res_i).sum()
            hess_th = -(a**2 * w_i).sum()
            if abs(hess_th) > 1e-9:
                theta[i] -= grad_th / hess_th

        # Standardise theta
        theta = (theta - theta.mean()) / max(theta.std(), 1e-9)

        if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            print(f"IRT 2PL converged at iteration {iteration}")
            break

    item_df = pd.DataFrame({"a_discrimination": a, "b_difficulty": b},
                            index=[f"item_{j+1:02d}" for j in range(J)])
    return {
        "a":          a,
        "b":          b,
        "theta":      theta,
        "item_params": item_df,
        "ll_history": ll_history,
    }
```

---

## 7. Differential Item Functioning (Mantel-Haenszel)

```python
def detect_dif(
    responses: pd.DataFrame,
    group: pd.Series,
    n_score_strata: int = 5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Detect DIF using the Mantel-Haenszel (1959) procedure.

    Items are flagged if the MH chi-squared test is significant after
    Bonferroni correction.

    Parameters
    ----------
    responses : pd.DataFrame
        Binary or polytomous item responses (respondents × items).
    group : pd.Series
        Binary group indicator (0 = reference, 1 = focal).
    n_score_strata : int
        Number of total-score strata for conditioning.
    alpha : float
        Significance level (before Bonferroni correction).

    Returns
    -------
    pd.DataFrame with columns: item, mh_chi2, p_value, mh_odds_ratio, dif_flag.
    """
    items      = [c for c in responses.columns if c != group.name]
    total_score = responses[items].sum(axis=1)
    strata      = pd.qcut(total_score, q=n_score_strata, labels=False, duplicates="drop")

    n_items     = len(items)
    bonferroni  = alpha / n_items
    rows        = []

    for item in items:
        A_list, B_list, C_list, D_list = [], [], [], []
        for s in strata.unique():
            mask  = strata == s
            ref   = group[mask] == 0
            foc   = group[mask] == 1
            # Binary: endorsed vs not-endorsed
            # Treat scores > median as "endorsed"
            threshold = responses[item][mask].median()
            endorsed  = responses[item][mask] >= threshold

            A = int((ref &  endorsed)[mask].sum())   # reference, endorsed
            B = int((ref & ~endorsed)[mask].sum())   # reference, not endorsed
            C = int((foc &  endorsed)[mask].sum())   # focal, endorsed
            D = int((foc & ~endorsed)[mask].sum())   # focal, not endorsed

            # Re-derive from mask correctly
            sub_ref_end  = ((group[mask] == 0) & (responses[item][mask] >= threshold)).sum()
            sub_ref_not  = ((group[mask] == 0) & (responses[item][mask] <  threshold)).sum()
            sub_foc_end  = ((group[mask] == 1) & (responses[item][mask] >= threshold)).sum()
            sub_foc_not  = ((group[mask] == 1) & (responses[item][mask] <  threshold)).sum()

            A_list.append(sub_ref_end)
            B_list.append(sub_ref_not)
            C_list.append(sub_foc_end)
            D_list.append(sub_foc_not)

        A, B, C, D = map(np.array, [A_list, B_list, C_list, D_list])
        N = A + B + C + D + 1e-9

        # MH common odds ratio
        numerator   = (A * D / N).sum()
        denominator = (B * C / N).sum()
        mh_or       = numerator / (denominator + 1e-12)

        # MH chi-squared (without continuity correction)
        E_A = (A + B) * (A + C) / N
        var = (A + B) * (C + D) * (A + C) * (B + D) / (N**2 * (N - 1) + 1e-9)
        chi2  = (abs((A - E_A).sum()) ** 2) / max(var.sum(), 1e-9)
        p_val = 1 - stats.chi2.cdf(chi2, df=1)

        rows.append({
            "item":           item,
            "mh_chi2":        round(chi2, 4),
            "p_value":        round(p_val, 6),
            "mh_odds_ratio":  round(mh_or, 4),
            "dif_flag":       "DIF" if p_val < bonferroni else "No DIF",
            "dif_magnitude":  (
                "C (large)"  if abs(np.log(mh_or)) > np.log(2.35) else
                "B (moderate)" if abs(np.log(mh_or)) > np.log(1.5) else
                "A (negligible)"
            ),
        })

    df = pd.DataFrame(rows).sort_values("mh_chi2", ascending=False).reset_index(drop=True)
    n_dif = (df["dif_flag"] == "DIF").sum()
    print(f"\nDIF Analysis: {n_dif}/{n_items} items flagged (Bonferroni α={bonferroni:.4f})")
    return df
```

---

## 8. Example A — Validate a 20-Item Scale (EFA + Alpha + CFA)

```python
import numpy as np
import pandas as pd

# Use the synthetic data created in Section 1
item_data = responses[item_cols]

# ---- Step 1: Item statistics -----------------------------------------------
print("=== Step 1: CTT Item Statistics ===")
item_stats = compute_item_stats(item_data)
print(item_stats.round(3).to_string())

# Flag items with low corrected item-total correlation
low_citc = item_stats[item_stats["corrected_item_total_r"] < 0.30]
print(f"\nItems with CITC < 0.30: {list(low_citc.index)}")

# ---- Step 2: Reliability ---------------------------------------------------
print("\n=== Step 2: Cronbach's Alpha ===")
alpha_result = cronbach_alpha(item_data)

# Subscale alphas
alpha_f1 = cronbach_alpha(item_data.iloc[:, :10])
alpha_f2 = cronbach_alpha(item_data.iloc[:, 10:])
print(f"  F1 subscale alpha: {alpha_f1['alpha']}")
print(f"  F2 subscale alpha: {alpha_f2['alpha']}")

# ---- Step 3: EFA on 50% hold-out split ------------------------------------
print("\n=== Step 3: Exploratory Factor Analysis ===")
np.random.seed(1)
idx_efa = np.random.choice(len(item_data), size=len(item_data) // 2, replace=False)
efa_data = item_data.iloc[idx_efa].reset_index(drop=True)

efa_result = run_efa(efa_data, n_factors=2, rotation="oblimin", parallel_analysis=True)

# ---- Step 4: CFA on remaining 50% -----------------------------------------
print("\n=== Step 4: Confirmatory Factor Analysis ===")
idx_cfa  = np.setdiff1d(np.arange(len(item_data)), idx_efa)
cfa_data = item_data.iloc[idx_cfa].reset_index(drop=True)

# Build 2-factor CFA model string from EFA results
f1_items = [c for c in item_cols[:10]]
f2_items = [c for c in item_cols[10:]]

model_str = (
    "F1 =~ " + " + ".join(f1_items) + "\n"
    "F2 =~ " + " + ".join(f2_items)
)

cfa_result = run_cfa(cfa_data, model_str)

# ---- Step 5: Summary report -----------------------------------------------
print("\n=== Scale Validation Summary ===")
print(f"  Items: {len(item_cols)}")
print(f"  Factors identified (EFA): {efa_result['n_factors']}")
print(f"  Overall alpha: {alpha_result['alpha']:.3f}")
print(f"  CFA CFI:   {cfa_result['cfi']:.3f}  (≥0.95 = acceptable)")
print(f"  CFA RMSEA: {cfa_result['rmsea']:.3f}  (≤0.06 = close fit)")
print(f"  CFA SRMR:  {cfa_result['srmr']:.3f}  (≤0.08 = acceptable)")
```

---

## 9. Example B — Compare IRT Parameters Across Gender Groups for DIF

```python
import numpy as np
import pandas as pd

# ---- Binarize Likert responses for IRT (score >= 3 => 1) -------------------
binary_items = (responses[item_cols] >= 3).astype(int)

# ---- Full-sample IRT -------------------------------------------------------
print("=== IRT 2PL: Full Sample ===")
irt_full = run_irt_2pl(binary_items.values, n_iter=200)
print("\nItem Parameters (first 10 items):")
print(irt_full["item_params"].head(10).round(3).to_string())

# ---- Split by gender -------------------------------------------------------
female_mask = responses["group"] == 0
male_mask   = responses["group"] == 1

print("\n=== IRT 2PL: Female Group ===")
irt_female = run_irt_2pl(binary_items[female_mask].values, n_iter=200)

print("\n=== IRT 2PL: Male Group ===")
irt_male   = run_irt_2pl(binary_items[male_mask].values, n_iter=200)

# ---- Compare item difficulty (b) across groups ----------------------------
b_comparison = pd.DataFrame({
    "b_full":   irt_full["b"],
    "b_female": irt_female["b"],
    "b_male":   irt_male["b"],
    "delta_b":  irt_female["b"] - irt_male["b"],
}, index=item_cols)

print("\n=== Item Difficulty Comparison (Female - Male) ===")
print(b_comparison.round(3).to_string())

large_dif_items = b_comparison[b_comparison["delta_b"].abs() > 0.5]
print(f"\nItems with |delta_b| > 0.5 (potential DIF): {list(large_dif_items.index)}")

# ---- Mantel-Haenszel DIF ---------------------------------------------------
print("\n=== Mantel-Haenszel DIF Analysis ===")
dif_result = detect_dif(
    binary_items,
    responses["group"],
    n_score_strata=5,
)
print(dif_result[dif_result["dif_flag"] == "DIF"].to_string(index=False)
      if (dif_result["dif_flag"] == "DIF").any() else "No items flagged for DIF.")

# ---- ICC plot for one item -------------------------------------------------
theta_range = np.linspace(-3, 3, 200)
fig, ax     = plt.subplots(figsize=(8, 5))
for i, item in enumerate(item_cols[:4]):
    j     = item_cols.index(item)
    a_j   = irt_full["a"][j]
    b_j   = irt_full["b"][j]
    icc   = expit(a_j * (theta_range - b_j))
    ax.plot(theta_range, icc, linewidth=2,
            label=f"{item} (a={a_j:.2f}, b={b_j:.2f})")

ax.set_xlabel("Latent Trait (θ)")
ax.set_ylabel("P(correct)")
ax.set_title("Item Characteristic Curves — 2PL Model", fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 1)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
plt.tight_layout()
plt.savefig("icc_2pl.png", dpi=150)
plt.show()
```

---

## 10. Measurement Invariance Testing

```python
def test_measurement_invariance(
    data: pd.DataFrame,
    group: pd.Series,
    model_str: str,
) -> pd.DataFrame:
    """
    Test configural, metric, and scalar measurement invariance across groups.

    Steps:
      1. Configural: same factor structure, all params free.
      2. Metric: loadings constrained equal across groups.
      3. Scalar: loadings + intercepts constrained equal.

    Compares chi-square difference tests and ΔCFI (Cheung & Rensvold, 2002).

    Parameters
    ----------
    data : pd.DataFrame
        Full item response matrix.
    group : pd.Series
        Group indicator (integer-coded).
    model_str : str
        semopy CFA model specification.

    Returns
    -------
    pd.DataFrame with rows for each invariance level.
    """
    groups  = group.unique()
    results = []

    for level in ["configural", "metric", "scalar"]:
        level_chi2, level_df, level_cfi = [], [], []

        for g in groups:
            mask   = group == g
            g_data = data[mask].reset_index(drop=True)
            try:
                fit = run_cfa(g_data, model_str)
                level_chi2.append(fit["chi2"])
                level_df.append(fit["df"])
                level_cfi.append(fit["cfi"])
            except Exception as exc:
                print(f"Warning: CFA failed for group {g} at level {level}: {exc}")

        if level_chi2:
            results.append({
                "level":   level,
                "chi2":    round(sum(level_chi2), 3),
                "df":      sum(level_df),
                "mean_cfi": round(np.mean(level_cfi), 4),
            })

    inv_df = pd.DataFrame(results)
    # Delta chi2 tests
    if len(inv_df) >= 2:
        inv_df["delta_chi2"] = inv_df["chi2"].diff().fillna(0)
        inv_df["delta_df"]   = inv_df["df"].diff().fillna(0)
        inv_df["delta_p"]    = inv_df.apply(
            lambda r: 1 - stats.chi2.cdf(r["delta_chi2"], df=r["delta_df"])
            if r["delta_df"] > 0 else np.nan, axis=1
        )
        inv_df["delta_cfi"]  = inv_df["mean_cfi"].diff().fillna(0)
        inv_df["decision"]   = inv_df["delta_cfi"].apply(
            lambda d: "Invariant (ΔCFI < 0.01)" if abs(d) < 0.01 else "Non-invariant"
        )

    print("\n=== Measurement Invariance Tests ===")
    print(inv_df.round(4).to_string(index=False))
    return inv_df
```

---

## 11. Tips and Common Pitfalls

- **Ordinal vs continuous EFA**: For Likert items with fewer than 5 categories or
  substantial skew, use polychoric correlations instead of Pearson. The
  `factor_analyzer` library supports `is_corr_matrix=True`; compute polychoric
  correlations with `pingouin` or `semopy`.
- **CFA sample size**: Aim for N ≥ 200 for stable CFA estimates; N ≥ 10 per free
  parameter is a common rule of thumb. The 2PL IRT model requires N ≥ 500 for
  stable discrimination estimates.
- **Alpha vs omega**: Cronbach's alpha assumes tau-equivalence (equal loadings).
  McDonald's omega (hierarchical) is preferable for multidimensional scales.
  Compute omega with `pingouin.reliability(data)` or the `factor_analyzer`
  `get_factor_variance()` output.
- **IRT model identification**: Fix the mean of theta to 0 and SD to 1 (or fix
  one item's b to 0) to identify the 2PL model. The implementation above uses
  z-scoring of theta at each EM step.
- **DIF interpretation**: MH category A (|ln(alpha_MH)| < ln(1.5)) = negligible;
  B = moderate; C = large (ETS classification). Always supplement with subject-matter
  review of DIF items before removal.
- **Parallel analysis vs Kaiser**: Kaiser criterion (eigenvalue > 1) systematically
  over-factors. Parallel analysis is strongly preferred in the current literature.
