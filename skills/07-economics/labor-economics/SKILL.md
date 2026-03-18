---
name: labor-economics
description: >
  Use this Skill for labor economics analysis: Mincer wage equation, Heckman
  selection, Oaxaca-Blinder decomposition, CPS data cleaning, and unconditional
  quantile regression.
tags:
  - economics
  - labor-economics
  - Mincer
  - Heckman-selection
  - Oaxaca-Blinder
  - wages
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
    - statsmodels>=0.14
    - pandas>=1.5
    - numpy>=1.23
    - scipy>=1.9
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Labor Economics Analysis

> **TL;DR** — Estimate wage equations, correct for sample selection bias with
> Heckman two-step, decompose wage gaps between groups with Oaxaca-Blinder,
> and analyze distributional wage effects with unconditional quantile regression.

---

## When to Use

| Situation | Recommended Method |
|---|---|
| Log wage regression with education and experience | Mincer wage equation |
| Wages observed only for employed workers | Heckman two-step selection correction |
| Wage gap between two groups (e.g., by gender, race) | Oaxaca-Blinder decomposition |
| Effect of a policy on wage distribution (not just mean) | Unconditional quantile regression |
| Union wage premium, minimum wage study | Event study or difference-in-means |

---

## Background

### Mincer Wage Equation

The canonical Mincer (1974) earnings equation:

    ln(wage_i) = β_0 + β_1 educ_i + β_2 exp_i + β_3 exp_i² + X_i γ + ε_i

Key coefficients:
- β_1: rate of return to one additional year of education (typically 6–12%)
- β_2: linear return to experience
- β_3: negative coefficient on exp² captures diminishing returns
- Peak experience: exp* = -β_2 / (2 β_3)

### Heckman Two-Step Selection Correction

**Problem**: Wages are only observed for workers who choose to work. If the decision
to work is correlated with unobserved wage determinants, OLS on the selected sample is
biased (selection bias).

**Heckman (1979) two-step procedure:**

Step 1 — Selection equation (Probit):
    P(work_i = 1 | Z_i) = Φ(Z_i δ)
    Z_i includes exclusion restrictions: variables that affect work participation
    but not wages (e.g., number of children under 6, non-labor income, spouse's income).

Step 2 — Wage equation with IMR:
    ln(wage_i) = X_i β + σ λ(Z_i δ̂) + η_i   for workers only

where λ(·) = φ(·)/Φ(·) is the Inverse Mills Ratio (IMR).
σ = cov(ε, u) / Var(u)^{1/2}. A significant σ (coefficient on IMR) indicates
selection bias in the naive OLS.

### Oaxaca-Blinder Decomposition

The Oaxaca (1973) and Blinder (1973) decomposition of the mean wage gap between
group A (e.g., men) and group B (e.g., women):

    Ȳ_A - Ȳ_B = [X̄_A - X̄_B]' β_A    +    X̄_B' [β_A - β_B]
               = "Explained" (endowments)  + "Unexplained" (coefficients/discrimination)

The unexplained component is often interpreted as an upper bound on discrimination,
though it also captures differences in unobserved characteristics.

### Unconditional Quantile Regression (Firpo-Fortin-Lemieux 2009)

Standard quantile regression (QR) estimates conditional quantile effects:
Q_τ(Y | X). But policy questions often concern the unconditional (marginal)
distribution of wages.

**RIF regression**: Replace Y with its recentered influence function (RIF):
    RIF(y; Q_τ) = Q_τ + (τ - 1{y ≤ Q_τ}) / f_Y(Q_τ)

Then regress RIF(Y; Q_τ) on X using OLS. Coefficients are interpretable as
marginal effects on the unconditional τ-th quantile of Y.

---

## Environment Setup

```bash
conda create -n labor python=3.11 -y
conda activate labor
pip install statsmodels>=0.14 pandas>=1.5 numpy>=1.23 scipy>=1.9 matplotlib>=3.6

python -c "import statsmodels; print('statsmodels', statsmodels.__version__)"
```

---

## Core Workflow

### Step 1 — Mincer Wage Equation and Heckman Correction

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm

np.random.seed(42)


def generate_labor_data(
    n: int = 2000,
    selection_bias: float = 0.5,
) -> pd.DataFrame:
    """
    Simulate a labor survey dataset with sample selection.

    Workers with higher unobserved productivity (ε) are both more likely to
    work and earn higher wages — generating selection bias.

    Args:
        n:               Number of individuals in the survey.
        selection_bias:  Correlation between selection and wage unobservables (0–1).

    Returns:
        DataFrame with columns: educ, exp, female, children_u6, nonlabor_inc,
                                 log_wage (NaN if not working), work.
    """
    educ = np.random.randint(8, 21, n).astype(float)
    exp = np.clip(np.random.normal(15, 8, n), 0, 45)
    female = np.random.binomial(1, 0.5, n)
    children_u6 = np.random.poisson(0.5, n)
    nonlabor_inc = np.abs(np.random.normal(1000, 500, n))

    # Correlated errors
    rho = selection_bias
    u_wage = np.random.randn(n)
    u_select = rho * u_wage + np.sqrt(1 - rho**2) * np.random.randn(n)

    # True log-wage equation
    log_wage_latent = (
        0.5
        + 0.08 * educ
        + 0.04 * exp
        - 0.0006 * exp**2
        - 0.15 * female
        + u_wage
    )

    # Selection equation (probit)
    index_select = (
        -1.0
        + 0.05 * educ
        + 0.02 * exp
        - 0.3 * female
        - 0.2 * children_u6
        - 0.0003 * nonlabor_inc
        + u_select
    )
    work = (index_select > 0).astype(int)

    log_wage = np.where(work == 1, log_wage_latent, np.nan)

    return pd.DataFrame({
        "educ": educ,
        "exp": exp,
        "exp2": exp**2,
        "female": female,
        "children_u6": children_u6,
        "nonlabor_inc": nonlabor_inc,
        "log_wage": log_wage,
        "work": work,
    })


def mincer_ols(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Estimate the Mincer wage equation via OLS (for workers only).

    Args:
        df: Labor dataset with log_wage, educ, exp, exp2, female columns.

    Returns:
        OLS results object.
    """
    workers = df.dropna(subset=["log_wage"]).copy()
    X = sm.add_constant(workers[["educ", "exp", "exp2", "female"]])
    model = sm.OLS(workers["log_wage"], X)
    result = model.fit(cov_type="HC3")
    print("Mincer OLS (workers only):")
    print(result.summary().tables[1])

    # Return to education
    educ_coef = result.params["educ"]
    print(f"\nReturn to education: {educ_coef:.4f} ({educ_coef*100:.2f}% per year)")
    # Peak experience
    b_exp = result.params["exp"]
    b_exp2 = result.params["exp2"]
    if b_exp2 < 0:
        peak_exp = -b_exp / (2 * b_exp2)
        print(f"Peak experience: {peak_exp:.1f} years")

    return result


def heckman_two_step(df: pd.DataFrame) -> dict:
    """
    Heckman two-step selection correction.

    Step 1: Probit selection equation (uses exclusion restrictions:
            children_u6, nonlabor_inc).
    Step 2: OLS wage equation with Inverse Mills Ratio.

    Args:
        df: Labor dataset.

    Returns:
        Dictionary with keys: selection_result, wage_result, imr_significant.
    """
    # Step 1: Probit on full sample
    X_sel = sm.add_constant(df[["educ", "exp", "exp2", "female",
                                  "children_u6", "nonlabor_inc"]])
    probit = sm.Probit(df["work"], X_sel)
    probit_result = probit.fit(disp=False)
    print("Step 1 — Selection Probit:")
    print(probit_result.summary().tables[1])

    # Compute Inverse Mills Ratio for all observations
    xb = probit_result.predict(X_sel, linear=True)
    df = df.copy()
    df["imr"] = norm.pdf(xb) / norm.cdf(xb)

    # Step 2: Wage OLS with IMR (workers only)
    workers = df.dropna(subset=["log_wage"]).copy()
    X_wage = sm.add_constant(workers[["educ", "exp", "exp2", "female", "imr"]])
    wage_model = sm.OLS(workers["log_wage"], X_wage)
    wage_result = wage_model.fit(cov_type="HC3")
    print("\nStep 2 — Wage Equation with IMR:")
    print(wage_result.summary().tables[1])

    imr_pval = float(wage_result.pvalues["imr"])
    imr_coef = float(wage_result.params["imr"])
    print(f"\nIMR coefficient: {imr_coef:.4f}  p-value: {imr_pval:.4f}")
    if imr_pval < 0.05:
        direction = "upward" if imr_coef > 0 else "downward"
        print(f"  Significant selection bias: OLS is biased {direction}.")
    else:
        print("  No significant selection bias detected.")

    return {
        "selection_result": probit_result,
        "wage_result": wage_result,
        "imr_significant": imr_pval < 0.05,
        "df_with_imr": df,
    }
```

### Step 2 — Oaxaca-Blinder Decomposition

```python
def oaxaca_blinder(
    df: pd.DataFrame,
    group_col: str = "female",
    group_A_value: int = 0,
    group_B_value: int = 1,
    outcome: str = "log_wage",
    covariates: list = None,
    output_path: str = None,
) -> dict:
    """
    Oaxaca-Blinder wage gap decomposition.

    Decomposes mean wage gap (Group A - Group B) into:
      - Explained: difference in endowments valued at Group A's coefficients
      - Unexplained: difference in coefficients applied to Group B's endowments

    Args:
        df:              Labor dataset (workers only for wage regressions).
        group_col:       Column identifying the two groups.
        group_A_value:   Value of group_col for Group A (reference, higher-wage).
        group_B_value:   Value of group_col for Group B.
        outcome:         Log wage column.
        covariates:      List of control variables (excluding group_col).
        output_path:     If provided, save decomposition bar chart.

    Returns:
        Dictionary with keys: total_gap, explained, unexplained, explained_share,
                              unexplained_share, coef_detail_A, coef_detail_B.
    """
    workers = df.dropna(subset=[outcome]).copy()

    if covariates is None:
        covariates = ["educ", "exp", "exp2"]

    df_A = workers[workers[group_col] == group_A_value]
    df_B = workers[workers[group_col] == group_B_value]

    # OLS for each group separately
    def ols_group(data, cols, dep):
        X = sm.add_constant(data[cols])
        return sm.OLS(data[dep], X).fit()

    res_A = ols_group(df_A, covariates, outcome)
    res_B = ols_group(df_B, covariates, outcome)

    X_A_mean = np.array([1.0] + [df_A[c].mean() for c in covariates])
    X_B_mean = np.array([1.0] + [df_B[c].mean() for c in covariates])
    beta_A = res_A.params.values
    beta_B = res_B.params.values

    mean_A = float(df_A[outcome].mean())
    mean_B = float(df_B[outcome].mean())
    total_gap = mean_A - mean_B

    # Oaxaca decomposition (reference = Group A coefficients)
    explained = float((X_A_mean - X_B_mean) @ beta_A)
    unexplained = float(X_B_mean @ (beta_A - beta_B))

    print(f"\nOaxaca-Blinder Decomposition")
    print(f"  Group A (value={group_A_value}) mean log wage: {mean_A:.4f}")
    print(f"  Group B (value={group_B_value}) mean log wage: {mean_B:.4f}")
    print(f"  Raw gap: {total_gap:.4f} ({np.exp(total_gap)-1:.1%} in levels)")
    print(f"  Explained (endowments): {explained:.4f} ({explained/total_gap:.1%} of gap)")
    print(f"  Unexplained (returns):  {unexplained:.4f} ({unexplained/total_gap:.1%} of gap)")

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Explained\n(Endowments)", "Unexplained\n(Returns)", "Total Gap"],
                  [explained, unexplained, total_gap],
                  color=["#3498DB", "#E74C3C", "#2ECC71"],
                  edgecolor="white", width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Log Wage Gap")
    ax.set_title("Oaxaca-Blinder Decomposition")
    for bar, val in zip(bars, [explained, unexplained, total_gap]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved decomposition chart to {output_path}")
    plt.show()

    return {
        "total_gap": total_gap,
        "explained": explained,
        "unexplained": unexplained,
        "explained_share": explained / total_gap if total_gap != 0 else np.nan,
        "unexplained_share": unexplained / total_gap if total_gap != 0 else np.nan,
    }
```

### Step 3 — Unconditional Quantile Regression (RIF)

```python
from scipy.stats import gaussian_kde


def rif_quantile_regression(
    df: pd.DataFrame,
    outcome: str = "log_wage",
    covariates: list = None,
    quantiles: list = None,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Recentered Influence Function (RIF) unconditional quantile regression.

    For each quantile τ, construct RIF(Y; Q_τ) and regress on X via OLS.
    Coefficients measure marginal effects on the unconditional τ-th quantile.

    Args:
        df:          Workers dataset with outcome variable.
        outcome:     Log wage column.
        covariates:  Control variables for the RIF regression.
        quantiles:   List of quantiles (0–1). Default: [0.1, 0.25, 0.5, 0.75, 0.9].
        output_path: If provided, save coefficient plot.

    Returns:
        DataFrame with rows = quantiles, columns = covariate coefficients.
    """
    workers = df.dropna(subset=[outcome]).copy()
    y = workers[outcome].values

    if covariates is None:
        covariates = ["educ", "exp", "exp2", "female"]
    if quantiles is None:
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    # KDE for density estimation at sample quantiles
    kde = gaussian_kde(y, bw_method="silverman")

    records = []
    for tau in quantiles:
        q_tau = np.quantile(y, tau)
        f_q_tau = float(kde.evaluate(q_tau))

        # RIF = Q_τ + (τ - 1{Y <= Q_τ}) / f(Q_τ)
        indicator = (y <= q_tau).astype(float)
        rif = q_tau + (tau - indicator) / (f_q_tau + 1e-10)

        X_rif = sm.add_constant(workers[covariates])
        rif_result = sm.OLS(rif, X_rif).fit(cov_type="HC3")

        row = {"quantile": tau}
        for var in covariates:
            row[f"{var}_coef"] = float(rif_result.params[var])
            row[f"{var}_se"] = float(rif_result.bse[var])
        records.append(row)

    rif_df = pd.DataFrame(records)

    # Plot education coefficient across quantiles
    fig, ax = plt.subplots(figsize=(9, 5))
    educ_coefs = [r["educ_coef"] for r in records]
    educ_ses = [r["educ_se"] for r in records]
    quantile_vals = [r["quantile"] for r in records]

    ax.plot(quantile_vals, educ_coefs, "o-", color="#2980B9", linewidth=2,
            label="Education RIF coefficient")
    ax.fill_between(quantile_vals,
                    [c - 1.96 * s for c, s in zip(educ_coefs, educ_ses)],
                    [c + 1.96 * s for c, s in zip(educ_coefs, educ_ses)],
                    alpha=0.2, color="#2980B9")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Quantile (τ)")
    ax.set_ylabel("Effect on Unconditional Quantile")
    ax.set_title("Unconditional Quantile Regression: Return to Education")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved RIF plot to {output_path}")
    plt.show()

    print("\nRIF Quantile Regression — Education Coefficients:")
    print(rif_df[["quantile", "educ_coef", "educ_se"]].round(4).to_string(index=False))

    return rif_df
```

---

## Advanced Usage

### Union Wage Premium with Selection Correction

```python
def union_wage_premium(
    df: pd.DataFrame,
    union_col: str = "union",
    wage_col: str = "log_wage",
    controls: list = None,
) -> dict:
    """
    Estimate union wage premium: naive OLS and selection-corrected version.

    Union membership is not random — workers may self-select into unions.
    Compare naive OLS estimate with Heckman-corrected estimate.

    Args:
        df:         Dataset with union indicator, log wages, and controls.
        union_col:  Binary union membership column.
        wage_col:   Log wage column.
        controls:   Control variables. Defaults to educ, exp, exp2, female.

    Returns:
        Dictionary with naive_premium, corrected_premium, selection_bias.
    """
    workers = df.dropna(subset=[wage_col]).copy()
    if controls is None:
        controls = ["educ", "exp", "exp2", "female"]

    # Naive OLS
    X_naive = sm.add_constant(workers[[union_col] + controls])
    ols_naive = sm.OLS(workers[wage_col], X_naive).fit(cov_type="HC3")
    naive_premium = float(ols_naive.params[union_col])
    print(f"Naive OLS union premium: {naive_premium:.4f} ({(np.exp(naive_premium)-1)*100:.2f}%)")

    # Heckman correction for union selection (if exclusion restriction available)
    # Here we use nonlabor_inc as a proxy exclusion restriction (for illustration)
    if "nonlabor_inc" in df.columns:
        X_sel = sm.add_constant(workers[controls + ["nonlabor_inc"]])
        probit_union = sm.Probit(workers[union_col], X_sel).fit(disp=False)
        xb = probit_union.predict(X_sel, linear=True)
        workers["imr_union"] = norm.pdf(xb) / norm.cdf(xb)

        X_corr = sm.add_constant(workers[[union_col] + controls + ["imr_union"]])
        ols_corr = sm.OLS(workers[wage_col], X_corr).fit(cov_type="HC3")
        corrected_premium = float(ols_corr.params[union_col])
        print(f"Selection-corrected union premium: {corrected_premium:.4f}")
    else:
        corrected_premium = None

    return {
        "naive_premium": naive_premium,
        "corrected_premium": corrected_premium,
        "selection_bias": (naive_premium - corrected_premium
                           if corrected_premium is not None else None),
    }
```

---

## Troubleshooting

| Error / Issue | Cause | Resolution |
|---|---|---|
| Heckman IMR coefficient not significant | Weak exclusion restriction | Find stronger exclusion restriction (nonlabor income, spouse's wage) |
| OB decomposition does not sum to total gap | Floating-point rounding | Check Σ(explained + unexplained) ≈ total_gap with tolerance 1e-6 |
| RIF regression coefs vary widely across quantiles | Insufficient bandwidth in KDE | Try `bw_method='scott'`; increase sample size |
| Perfect multicollinearity in Mincer model | exp and exp2 highly correlated | Use centered exp: `exp_c = exp - exp.mean()` |
| Probit does not converge | Near-perfect prediction | Check for perfect separation in selection equation |
| `statsmodels` Probit slow | Large sample | Reduce to 50k rows for initial exploration |

---

## External Resources

- Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
- Heckman, J. (1979). "Sample Selection Bias as a Specification Error."
  *Econometrica*, 47(1), 153–161.
- Oaxaca, R. (1973). "Male-Female Wage Differentials in Urban Labor Markets."
  *International Economic Review*, 14(3), 693–709.
- Firpo, S., Fortin, N., Lemieux, T. (2009). "Unconditional Quantile Regressions."
  *Econometrica*, 77(3), 953–973.
- `statsmodels` Heckman docs: <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html>

---

## Examples

### Example 1 — Mincer + Heckman on Simulated Data

```python
df = generate_labor_data(n=3000, selection_bias=0.6)
workers = df.dropna(subset=["log_wage"])
print(f"Labor force participation rate: {df['work'].mean():.3f}")

# Step 1: Naive Mincer OLS
ols_result = mincer_ols(df)

# Step 2: Heckman correction
heck = heckman_two_step(df)
print(f"\nOLS educ coef: {ols_result.params['educ']:.4f}")
print(f"Heckman educ coef: {heck['wage_result'].params['educ']:.4f}")
```

### Example 2 — Oaxaca-Blinder Gender Wage Gap

```python
df = generate_labor_data(n=5000, selection_bias=0.4)

ob = oaxaca_blinder(
    df.dropna(subset=["log_wage"]),
    group_col="female",
    group_A_value=0,
    group_B_value=1,
    outcome="log_wage",
    covariates=["educ", "exp", "exp2"],
    output_path="oaxaca_blinder.png",
)

# RIF unconditional quantile regression
rif_df = rif_quantile_regression(
    df.dropna(subset=["log_wage"]),
    outcome="log_wage",
    covariates=["educ", "exp", "exp2", "female"],
    quantiles=[0.10, 0.25, 0.50, 0.75, 0.90],
    output_path="rif_quantile.png",
)
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Mincer, Heckman two-step, Oaxaca-Blinder, RIF-QR |
