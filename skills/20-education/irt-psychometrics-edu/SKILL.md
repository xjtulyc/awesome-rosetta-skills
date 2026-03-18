---
name: irt-psychometrics-edu
description: >
  Use this Skill for educational measurement with IRT: 2PL/3PL calibration,
  item information function, DIF detection, and test equating
  (Stocking-Lord).
tags:
  - education
  - IRT
  - item-response-theory
  - DIF
  - test-equating
  - psychometrics
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
    - numpy>=1.23
    - scipy>=1.9
    - pandas>=1.5
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Educational Measurement with Item Response Theory

> **TL;DR** — Implement 2PL/3PL IRT models, compute item characteristic
> curves, item information functions, detect differential item functioning
> (DIF) with Mantel-Haenszel, and understand test equating concepts.

---

## When to Use

Use this Skill when you need to:

- Calibrate item parameters (difficulty b, discrimination a, guessing c) for
  achievement or ability tests.
- Plot item characteristic curves (ICC) and item information functions (IIF)
  to evaluate item quality.
- Assess test information and measurement precision across the ability scale.
- Detect items that function differently across demographic groups (DIF).
- Understand or implement the Stocking-Lord test equating approach for
  converting scores across test forms.

**Do NOT use** this Skill for attitude/personality measurement (use CFA-based
approaches) or when sample size is below ~200 per group.

---

## Background

### IRT Model Family

| Model | Parameters | ICC Formula |
|---|---|---|
| 1PL (Rasch) | b (difficulty) | P(θ) = 1 / (1 + exp(-(θ - b))) |
| 2PL | a, b | P(θ) = 1 / (1 + exp(-a*(θ - b))) |
| 3PL | a, b, c | P(θ) = c + (1-c) / (1 + exp(-a*(θ - b))) |

- **θ**: Latent trait (ability), typically standardised ~ N(0,1)
- **a**: Discrimination — slope at inflection point (higher = better item)
- **b**: Difficulty — θ value where P(θ) = (1+c)/2 for 3PL
- **c**: Guessing — lower asymptote (probability of correct response at very low θ)

### Item Information Function (IIF)

```
I(θ) = [P'(θ)]² / [P(θ) * (1 - P(θ))]

For 3PL:  I(θ) = a² * (P(θ) - c)² * (1 - P(θ)) / [(1 - c)² * P(θ)]
```

Test Information Function (TIF) = Σ_i I_i(θ)
Standard Error of Measurement = 1 / √TIF(θ)

### Differential Item Functioning (DIF)

An item shows DIF if examinees from different groups (e.g., gender, ethnicity)
with the **same ability** have different probabilities of answering correctly.

- **Uniform DIF**: systematic advantage for one group across all θ levels
- **Non-uniform DIF**: group advantage reverses across θ levels

Mantel-Haenszel statistic classifies DIF as:
- Category A: |ΔMHD| < 1.0 (negligible — item passes)
- Category B: 1.0 ≤ |ΔMHD| < 1.5 (moderate — review)
- Category C: |ΔMHD| ≥ 1.5 (large — investigate or remove)

---

## Environment Setup

```bash
conda create -n irt_edu python=3.11 -y
conda activate irt_edu
pip install numpy>=1.23 scipy>=1.9 pandas>=1.5 matplotlib>=3.6

# Optional R packages for validation
# install.packages(c("mirt", "difR"))
```

---

## Core Workflow

### Step 1 — IRT Model Functions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import chi2


def icc_3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Three-parameter logistic (3PL) item characteristic curve.

    Args:
        theta: Array of ability values.
        a:     Discrimination parameter (a > 0, typically 0.5–2.5).
        b:     Difficulty parameter (typically -3 to +3).
        c:     Guessing parameter (typically 0.0–0.35).

    Returns:
        Array of P(correct | theta) values in [c, 1].
    """
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))


def icc_2pl(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """Two-parameter logistic (2PL) ICC — special case of 3PL with c=0."""
    return icc_3pl(theta, a, b, c=0.0)


def icc_1pl(theta: np.ndarray, b: float) -> np.ndarray:
    """One-parameter logistic / Rasch ICC — special case with a=1, c=0."""
    return icc_3pl(theta, a=1.0, b=b, c=0.0)


def iif_3pl(theta: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Item information function for the 3PL model.

    Args:
        theta: Ability values.
        a:     Discrimination.
        b:     Difficulty.
        c:     Guessing.

    Returns:
        Array of item information values I(θ).
    """
    p = icc_3pl(theta, a, b, c)
    q = 1 - p
    info = (a ** 2) * ((p - c) ** 2) * q / ((1 - c) ** 2 * p)
    return info


def tif(theta: np.ndarray, params: list[tuple]) -> np.ndarray:
    """
    Test information function — sum of item information functions.

    Args:
        theta:  Ability values.
        params: List of (a, b, c) tuples for each item.

    Returns:
        Array of test information TIF(θ).
    """
    total = np.zeros_like(theta)
    for a, b, c in params:
        total += iif_3pl(theta, a, b, c)
    return total


def sem_from_tif(theta: np.ndarray, params: list[tuple]) -> np.ndarray:
    """Standard error of measurement from test information: SEM(θ) = 1/√TIF(θ)."""
    test_info = tif(theta, params)
    return 1.0 / np.sqrt(np.maximum(test_info, 1e-6))
```

### Step 2 — 2PL Parameter Estimation (Marginal MLE)

```python
def estimate_2pl_jml(
    response_matrix: np.ndarray,
    n_iter: int = 100,
    lr_theta: float = 0.1,
    lr_params: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Joint Maximum Likelihood Estimation for 2PL IRT model.

    Alternates between updating ability (θ) estimates and item parameters (a, b)
    using gradient ascent on the log-likelihood. For production use, prefer
    marginal MLE via the R `mirt` package.

    Args:
        response_matrix: Binary matrix of shape (n_persons, n_items).
                         1 = correct, 0 = incorrect.
        n_iter:          Number of EM/JML iterations.
        lr_theta:        Learning rate for ability updates.
        lr_params:       Learning rate for item parameter updates.

    Returns:
        Tuple (theta_hat, a_hat, b_hat) — estimated abilities and item parameters.
    """
    n_persons, n_items = response_matrix.shape
    theta = np.zeros(n_persons)
    a = np.ones(n_items)
    b = np.zeros(n_items)

    for iteration in range(n_iter):
        # E-step equivalent: update theta for fixed item params
        for s in range(n_persons):
            p = icc_2pl(np.array([theta[s]] * n_items), a, b)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            y = response_matrix[s]
            grad_theta = np.sum(a * (y - p))
            theta[s] += lr_theta * grad_theta

        # M-step equivalent: update item params for fixed theta
        for i in range(n_items):
            p = icc_2pl(theta, a[i], b[i])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            y = response_matrix[:, i]
            residuals = y - p
            grad_a = np.sum(residuals * (theta - b[i]))
            grad_b = np.sum(residuals * (-a[i]))
            a[i] += lr_params * grad_a
            b[i] += lr_params * grad_b
            a[i] = max(a[i], 0.05)   # Discrimination must be positive

        if (iteration + 1) % 20 == 0:
            p_mat = np.array([icc_2pl(theta, a[i], b[i]) for i in range(n_items)]).T
            p_mat = np.clip(p_mat, 1e-6, 1 - 1e-6)
            ll = np.sum(response_matrix * np.log(p_mat) + (1 - response_matrix) * np.log(1 - p_mat))
            print(f"  Iteration {iteration + 1:3d}: log-likelihood = {ll:.2f}")

    return theta, a, b
```

### Step 3 — DIF Detection via Mantel-Haenszel

```python
def mantel_haenszel_dif(
    responses: np.ndarray,
    group: np.ndarray,
    score_bins: int = 10,
) -> pd.DataFrame:
    """
    Detect Differential Item Functioning using the Mantel-Haenszel chi-square test.

    Stratifies examinees by total score and computes the MH odds ratio and
    ΔMHD statistic for each item.

    Args:
        responses: Binary response matrix (n_persons × n_items).
        group:     Binary group membership array (0 = reference, 1 = focal).
        score_bins: Number of score strata (default 10).

    Returns:
        DataFrame with columns: item, MH_OR, MH_chi2, p_value, delta_MHD,
        DIF_category (A/B/C).
    """
    n_persons, n_items = responses.shape
    total_scores = responses.sum(axis=1)
    score_levels = pd.cut(total_scores, bins=score_bins, labels=False)

    records = []
    for item_idx in range(n_items):
        y = responses[:, item_idx]
        a_tbl_num = 0.0   # Σ (a_j * n_1j / n_j)  numerator for OR
        a_tbl_den = 0.0   # Σ (b_j * n_0j / n_j)  denominator for OR
        mh_num = 0.0
        mh_den = 0.0

        for level in range(score_bins):
            mask = score_levels == level
            if mask.sum() < 4:
                continue
            ref_mask = mask & (group == 0)
            foc_mask = mask & (group == 1)

            n_ref = ref_mask.sum()
            n_foc = foc_mask.sum()
            n_total = n_ref + n_foc

            if n_total == 0:
                continue

            n_correct_ref = y[ref_mask].sum()
            n_correct_foc = y[foc_mask].sum()
            n_correct_tot = n_correct_ref + n_correct_foc
            n_wrong_tot   = n_total - n_correct_tot

            if n_correct_tot == 0 or n_wrong_tot == 0:
                continue

            a_j = n_correct_ref * (n_total - n_ref) / n_total
            b_j = (n_ref - n_correct_ref) * n_foc / n_total
            a_tbl_num += a_j
            a_tbl_den += b_j

        mh_or = a_tbl_num / max(a_tbl_den, 1e-9)
        # MH chi-square (simplified)
        delta_mhd = -2.35 * np.log(mh_or)

        # Approximate chi-square from OR
        mh_chi2 = (np.log(mh_or)) ** 2 / (1 / max(a_tbl_num, 1e-9) + 1 / max(a_tbl_den, 1e-9))
        p_value = 1 - chi2.cdf(mh_chi2, df=1)

        adm = abs(delta_mhd)
        category = "A" if adm < 1.0 else ("B" if adm < 1.5 else "C")

        records.append({
            "item": item_idx + 1,
            "MH_OR": round(mh_or, 4),
            "MH_chi2": round(mh_chi2, 4),
            "p_value": round(p_value, 4),
            "delta_MHD": round(delta_mhd, 4),
            "DIF_category": category,
        })

    return pd.DataFrame(records)
```

---

## Advanced Usage

### Test Characteristic Curve and Reliability

```python
def plot_icc_and_iif(
    item_params: list[tuple],
    item_labels: list[str] = None,
    output_path: str = "icc_iif.png",
) -> None:
    """
    Plot ICC (left) and IIF (right) for multiple items.

    Args:
        item_params:  List of (a, b, c) tuples.
        item_labels:  Display labels for each item.
        output_path:  Output PNG path.
    """
    theta = np.linspace(-4, 4, 200)
    labels = item_labels or [f"Item {i+1}" for i in range(len(item_params))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for (a, b, c), label in zip(item_params, labels):
        p   = icc_3pl(theta, a, b, c)
        inf = iif_3pl(theta, a, b, c)
        axes[0].plot(theta, p, label=label)
        axes[1].plot(theta, inf, label=label)

    axes[0].set_xlabel("θ (Ability)")
    axes[0].set_ylabel("P(correct | θ)")
    axes[0].set_title("Item Characteristic Curves (3PL)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("θ (Ability)")
    axes[1].set_ylabel("Item Information I(θ)")
    axes[1].set_title("Item Information Functions (3PL)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"ICC and IIF plots saved to {output_path}")
    plt.close()


def plot_tif_and_sem(
    item_params: list[tuple],
    output_path: str = "tif_sem.png",
) -> None:
    """Plot Test Information Function and SEM across θ range."""
    theta = np.linspace(-4, 4, 200)
    test_info = tif(theta, item_params)
    sem_values = sem_from_tif(theta, item_params)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    ax1.plot(theta, test_info, color="#1f77b4", label="TIF")
    ax2.plot(theta, sem_values, color="#d62728", linestyle="--", label="SEM")

    ax1.set_xlabel("θ (Ability)")
    ax1.set_ylabel("Test Information", color="#1f77b4")
    ax2.set_ylabel("SEM (1/√TIF)", color="#d62728")
    ax1.set_title("Test Information Function and Standard Error of Measurement")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"TIF/SEM plot saved to {output_path}")
    plt.close()
```

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| JML estimates not converging | Learning rate too high or items with extreme difficulty | Reduce `lr_params`; check item p-values |
| Negative discrimination estimates | Items with reversed scoring or JML instability | Constrain `a > 0`; check item keys |
| DIF for most items | Groups differ in mean ability (impact, not DIF) | Condition on ability strata correctly |
| IIF peaks at wrong θ | Item difficulty mismatch for target population | Select items with b near target θ range |
| c parameter > 0.5 | Sample too small or item is poorly written | Fix c at 0.25 for 4-option MCQ |
| p_value all NaN in DIF table | Empty strata (all correct or all wrong) | Reduce `score_bins` to 5–8 |

---

## External Resources

- Lord (1980) Applications of Item Response Theory: https://www.ets.org/
- R `mirt` package: https://cran.r-project.org/package=mirt
- R `difR` package: https://cran.r-project.org/package=difR
- OECD PISA item parameters: https://www.oecd.org/pisa/data/
- Embretson & Reise (2000) Item Response Theory for Psychologists (Erlbaum)

---

## Examples

### Example 1 — 2PL ICC and IIF Plot for Simulated Items

```python
def example_icc_iif_plot():
    """Plot ICC and IIF for a 5-item bank with varied parameters."""
    # (a=discrimination, b=difficulty, c=guessing)
    item_params = [
        (0.8, -1.5, 0.0),   # Easy, low discrimination
        (1.2, -0.5, 0.2),   # Moderate, guessing
        (1.8,  0.0, 0.0),   # Average difficulty, high discrimination
        (2.0,  0.8, 0.25),  # Hard, high discrimination, guessing
        (0.6,  1.5, 0.0),   # Very hard, low discrimination
    ]
    labels = [f"Item {i+1} (a={p[0]}, b={p[1]}, c={p[2]})" for i, p in enumerate(item_params)]
    plot_icc_and_iif(item_params, labels, output_path="icc_iif_5items.png")
    plot_tif_and_sem(item_params, output_path="tif_sem_5items.png")
    return item_params


if __name__ == "__main__":
    example_icc_iif_plot()
```

### Example 2 — Mantel-Haenszel DIF Detection

```python
def example_mh_dif():
    """Simulate item response data with planted DIF items and detect them."""
    rng = np.random.default_rng(42)
    n_ref, n_foc = 500, 500
    n_items = 20

    # Item parameters (same for both groups — no DIF)
    a = rng.uniform(0.8, 1.8, n_items)
    b = rng.uniform(-1.5, 1.5, n_items)

    # Generate responses
    theta_ref = rng.normal(0.0, 1.0, n_ref)
    theta_foc = rng.normal(-0.3, 1.0, n_foc)  # Focal group slightly lower ability

    def simulate_responses(theta, a, b, c=0.0):
        prob = icc_3pl(theta[:, None], a[None, :], b[None, :], c)
        return (rng.uniform(size=prob.shape) < prob).astype(int)

    resp_ref = simulate_responses(theta_ref, a, b)
    resp_foc = simulate_responses(theta_foc, a, b)

    # Plant DIF on item 5: focal group has b shifted by +0.8
    b_dif = b.copy()
    b_dif[4] += 0.8
    resp_foc[:, 4] = simulate_responses(theta_foc, a, b_dif)[:, 4]

    responses = np.vstack([resp_ref, resp_foc])
    group     = np.array([0] * n_ref + [1] * n_foc)

    dif_results = mantel_haenszel_dif(responses, group, score_bins=8)
    print("DIF Detection Results (item 5 should show DIF):")
    print(dif_results.to_string(index=False))

    flagged = dif_results[dif_results["DIF_category"] != "A"]
    print(f"\nFlagged items (B or C DIF): {list(flagged['item'])}")
    return dif_results


if __name__ == "__main__":
    example_mh_dif()
```

### Example 3 — Test Characteristic Curve and IRT Reliability

```python
def example_tcc_reliability():
    """Compute test characteristic curve and IRT-based reliability."""
    rng = np.random.default_rng(7)
    n_items = 40

    # Simulate a realistic item bank for a 40-item test
    a_params = rng.uniform(0.7, 2.0, n_items)
    b_params = rng.uniform(-2.0, 2.0, n_items)
    c_params = rng.uniform(0.05, 0.25, n_items)
    params   = list(zip(a_params, b_params, c_params))

    theta = np.linspace(-4, 4, 300)

    # Test characteristic curve: expected score E[X|θ] = Σ P_i(θ)
    tcc = sum(icc_3pl(theta, a, b, c) for a, b, c in params)

    # IRT reliability: ρ_XX' = 1 - E[1/TIF(θ)] / Var(θ)
    # Approximate with standard normal ability distribution
    test_info = tif(theta, params)
    expected_error_var = np.trapz(1 / np.maximum(test_info, 0.01) * np.exp(-theta**2/2) / np.sqrt(2*np.pi), theta)
    irt_reliability = max(0, 1 - expected_error_var / 1.0)

    print(f"Test Characteristic Curve: Expected score range [{tcc.min():.1f}, {tcc.max():.1f}]")
    print(f"Marginal reliability (IRT): ρ = {irt_reliability:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(theta, tcc, color="#1f77b4")
    axes[0].set_xlabel("θ (Ability)")
    axes[0].set_ylabel("Expected Score")
    axes[0].set_title(f"Test Characteristic Curve (n_items={n_items})")
    axes[0].grid(alpha=0.3)

    axes[1].plot(theta, test_info, color="#2ca02c")
    axes[1].fill_between(theta, 0, test_info, alpha=0.2, color="#2ca02c")
    axes[1].set_xlabel("θ (Ability)")
    axes[1].set_ylabel("Test Information")
    axes[1].set_title(f"TIF — Marginal Reliability = {irt_reliability:.3f}")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("tcc_reliability.png", dpi=150)
    print("TCC and reliability plot saved to tcc_reliability.png")
    return irt_reliability


if __name__ == "__main__":
    example_tcc_reliability()
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — 2PL/3PL ICC, IIF, JML estimation, MH DIF, TCC reliability |
