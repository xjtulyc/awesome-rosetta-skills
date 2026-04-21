---
name: bayesian-stats
description: "Use this Skill when the user needs to build Bayesian statistical models: prior selection, posterior sampling with MCMC (NUTS), convergence diagnostics, posterior predictive checks, LOO-CV model comparison, hierarchical models, and Bayesian A/B testing. Covers PyMC 5.x and ArviZ."
tags:
  - mathematics
  - bayesian-inference
  - mcmc
  - pymc
  - arviz
  - hierarchical-models
  - model-comparison
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - pymc>=5.0.0
    - arviz>=0.17.0
    - numpy>=1.23.0
    - pandas>=1.5.0
    - matplotlib>=3.6.0
    - scipy>=1.10.0
last_updated: "2026-03-17"
---

# Bayesian Statistical Inference

> **TL;DR** — Full Bayesian workflow with PyMC 5.x and ArviZ: prior selection, NUTS
> sampling, convergence diagnostics (R-hat, ESS), posterior predictive checks,
> LOO-CV model comparison, hierarchical models, and Bayesian A/B testing.

---

## 1. Environment Setup

```bash
conda create -n bayes python=3.11 -y
conda activate bayes
pip install pymc arviz numpy pandas matplotlib scipy

# Verify (PyMC uses pytensor backend)
python -c "import pymc as pm; print(pm.__version__)"
```

Quick sanity check — sample from a trivial model:

```python
import pymc as pm
import arviz as az

with pm.Model():
    mu = pm.Normal("mu", mu=0, sigma=1)
    idata = pm.sample(100, tune=100, chains=2, progressbar=False)

print(az.summary(idata))
```

---

## 2. Core Implementation

### 2.1 Prior Selection Helper

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional


def prior_predictive_check(
    model: pm.Model,
    n_samples: int = 500,
    observed_data: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
) -> az.InferenceData:
    """
    Draw from the prior predictive distribution and compare with observed data.

    Use this before fitting to verify priors produce plausible outcome ranges.
    If the prior predictive generates values far outside the observed range,
    consider tighter / more informative priors.

    Args:
        model:         A PyMC model object (not yet sampled).
        n_samples:     Number of prior predictive draws.
        observed_data: Optional observed data array for overlay comparison.
        output_path:   If given, save the figure to this path.

    Returns:
        ArviZ InferenceData with prior_predictive group populated.
    """
    with model:
        idata = pm.sample_prior_predictive(samples=n_samples, random_seed=42)

    obs_var = list(idata.prior_predictive.data_vars)[-1]
    ppc_draws = idata.prior_predictive[obs_var].values.flatten()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ppc_draws, bins=50, density=True, alpha=0.6, label="Prior predictive", color="#4C72B0")
    if observed_data is not None:
        ax.hist(observed_data, bins=30, density=True, alpha=0.5, label="Observed data", color="#DD8452")
    ax.set_xlabel("Outcome")
    ax.set_title("Prior Predictive Check")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved prior predictive check to {output_path}")

    return idata
```

### 2.2 Linear Regression with Weakly Informative Priors

```python
def build_linear_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
) -> pm.Model:
    """
    Build a Bayesian linear regression model with weakly informative priors.

    Model:
        alpha  ~ Normal(y_mean, y_sd * 2)
        beta_j ~ Normal(0, 1)          for each feature (standardized inputs recommended)
        sigma  ~ HalfNormal(y_sd)
        y_i   ~ Normal(alpha + X_i @ beta, sigma)

    Args:
        X:             Feature matrix, shape (n, p). Standardize before passing.
        y:             Target vector, shape (n,).
        feature_names: Names for the beta coefficients (for readable summaries).

    Returns:
        An unsampled PyMC Model.
    """
    n, p = X.shape
    feature_names = feature_names or [f"x{j}" for j in range(p)]

    y_mean = float(np.mean(y))
    y_sd = float(np.std(y))

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=y_mean, sigma=y_sd * 2)
        beta = pm.Normal("beta", mu=0, sigma=1, shape=p)
        sigma = pm.HalfNormal("sigma", sigma=y_sd)

        mu = alpha + pm.math.dot(X, beta)
        obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    return model
```

### 2.3 NUTS Sampling and Diagnostics

```python
def sample_and_diagnose(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    Sample posterior with NUTS and run convergence diagnostics.

    Convergence criteria:
      - R-hat (Gelman-Rubin): all parameters should have R-hat <= 1.01
      - Effective Sample Size (ESS): all parameters should have ESS >= 400
      - No divergences (divergences indicate posterior geometry problems)

    Args:
        model:         PyMC model.
        draws:         Number of posterior draws per chain.
        tune:          Number of tuning (warm-up) steps.
        chains:        Number of independent chains (>= 4 recommended).
        target_accept: NUTS target acceptance rate (raise to 0.95 for difficult posteriors).
        random_seed:   For reproducibility.

    Returns:
        ArviZ InferenceData with posterior, sample_stats, and posterior_predictive groups.
    """
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)

    # Convergence diagnostics
    summary = az.summary(idata, round_to=3)
    print("=== Convergence Diagnostics ===")

    rhat_max = summary["r_hat"].max()
    ess_min = summary[["ess_bulk", "ess_tail"]].min().min()
    n_divergences = int(idata.sample_stats["diverging"].sum())

    print(f"  Max R-hat:       {rhat_max:.4f}  (target: <= 1.01)")
    print(f"  Min ESS (bulk):  {int(summary['ess_bulk'].min())}  (target: >= 400)")
    print(f"  Min ESS (tail):  {int(summary['ess_tail'].min())}  (target: >= 400)")
    print(f"  Divergences:     {n_divergences}  (target: 0)")

    if rhat_max > 1.01:
        print("  WARNING: R-hat > 1.01 — chains may not have converged. Increase tune/draws.")
    if ess_min < 400:
        print("  WARNING: Low ESS — increase draws or check for funnel geometry.")
    if n_divergences > 0:
        print(f"  WARNING: {n_divergences} divergences — consider reparameterization or higher target_accept.")

    return idata


def plot_diagnostics(
    idata: az.InferenceData,
    var_names: Optional[list] = None,
    output_dir: str = ".",
) -> None:
    """
    Generate standard ArviZ diagnostic plots.

    Produces: trace plot, posterior plot, rank plot (better than trace for convergence).

    Args:
        idata:      ArviZ InferenceData from sample_and_diagnose().
        var_names:  Variables to plot; None = all scalar parameters.
        output_dir: Directory to save plots (PNG).
    """
    import os

    # Trace plot
    ax = az.plot_trace(idata, var_names=var_names, compact=True)
    plt.suptitle("Trace Plot", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trace_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Posterior plot
    az.plot_posterior(idata, var_names=var_names, round_to=3)
    plt.savefig(os.path.join(output_dir, "posterior_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Rank plot (recommended over trace for multi-chain convergence)
    az.plot_rank(idata, var_names=var_names)
    plt.savefig(os.path.join(output_dir, "rank_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved diagnostic plots to {output_dir}/")
```

### 2.4 LOO-CV Model Comparison

```python
def compare_models(
    idatas: dict,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare Bayesian models using Leave-One-Out Cross-Validation (LOO-CV).

    LOO-CV estimates expected log predictive density (ELPD). Higher ELPD is better.
    Models within 4 ELPD_diff units of each other are considered comparable.

    Args:
        idatas:      Dictionary mapping model names to ArviZ InferenceData objects.
                     Each must have log_likelihood computed (idata_kwargs={"log_likelihood": True}).
        output_path: If given, save the comparison plot to this path.

    Returns:
        DataFrame of LOO comparison results sorted by ELPD descending.
    """
    import pandas as pd

    loo_results = {}
    for name, idata in idatas.items():
        loo_results[name] = az.loo(idata, pointwise=True)

    comparison = az.compare(loo_results, ic="loo", method="stacking")

    print("=== LOO-CV Model Comparison ===")
    print(comparison[["elpd_loo", "se", "p_loo", "weight"]].to_string())

    if output_path:
        az.plot_compare(comparison)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved model comparison plot to {output_path}")

    return comparison
```

### 2.5 Hierarchical Model (Partial Pooling)

```python
def build_hierarchical_model(
    group_ids: np.ndarray,
    y: np.ndarray,
    X: Optional[np.ndarray] = None,
) -> pm.Model:
    """
    Build a hierarchical (multilevel) model with non-centered parameterization.

    Non-centered parameterization (NCP) avoids the funnel geometry problem in
    hierarchical models and greatly improves NUTS sampling efficiency.

    Model (NCP):
        mu_alpha   ~ Normal(0, 10)     # global intercept mean
        sigma_alpha ~ HalfNormal(1)    # between-group SD
        z_alpha_g  ~ Normal(0, 1)      # standardized group offsets
        alpha_g    = mu_alpha + sigma_alpha * z_alpha_g   # actual group intercepts
        sigma_obs  ~ HalfNormal(1)
        y_i ~ Normal(alpha_{group[i]}, sigma_obs)

    Args:
        group_ids: Integer array of group indices (0-indexed), shape (n,).
        y:         Observed outcomes, shape (n,).
        X:         Optional covariate matrix for group-level predictors, shape (n_groups, q).

    Returns:
        An unsampled PyMC Model.
    """
    n_groups = int(np.max(group_ids)) + 1

    with pm.Model() as hierarchical_model:
        # Hyperpriors
        mu_alpha = pm.Normal("mu_alpha", mu=float(np.mean(y)), sigma=float(np.std(y)) * 2)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)

        # Non-centered parameterization: z_alpha ~ N(0,1), alpha = mu + sigma * z
        z_alpha = pm.Normal("z_alpha", mu=0, sigma=1, shape=n_groups)
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * z_alpha)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=float(np.std(y)))

        mu_y = alpha[group_ids]
        obs = pm.Normal("y_obs", mu=mu_y, sigma=sigma_obs, observed=y)

    return hierarchical_model
```

### 2.6 Bayesian A/B Testing

```python
def bayesian_ab_test(
    n_control: int,
    n_conversions_control: int,
    n_treatment: int,
    n_conversions_treatment: int,
    draws: int = 10000,
    random_seed: int = 42,
) -> dict:
    """
    Bayesian A/B test for conversion rates using Beta-Binomial conjugacy.

    Directly computes P(treatment > control) without p-values or stopping rules.
    Uses an uninformative Beta(1,1) prior (uniform over [0,1]).

    Args:
        n_control:              Number of users in control group.
        n_conversions_control:  Number of conversions in control group.
        n_treatment:            Number of users in treatment group.
        n_conversions_treatment: Number of conversions in treatment group.
        draws:                  Number of posterior samples per group.
        random_seed:            For reproducibility.

    Returns:
        Dictionary with: p_control_mean, p_treatment_mean, prob_treatment_better,
        relative_uplift_mean, credible_interval_95.
    """
    rng = np.random.default_rng(random_seed)

    # Posterior: Beta(alpha + conversions, beta + non-conversions)
    alpha_prior, beta_prior = 1.0, 1.0

    p_control = rng.beta(
        alpha_prior + n_conversions_control,
        beta_prior + (n_control - n_conversions_control),
        size=draws,
    )
    p_treatment = rng.beta(
        alpha_prior + n_conversions_treatment,
        beta_prior + (n_treatment - n_conversions_treatment),
        size=draws,
    )

    prob_better = float(np.mean(p_treatment > p_control))
    relative_uplift = (p_treatment - p_control) / p_control
    ci_low, ci_high = np.percentile(relative_uplift, [2.5, 97.5])

    result = {
        "p_control_mean": float(np.mean(p_control)),
        "p_treatment_mean": float(np.mean(p_treatment)),
        "prob_treatment_better": prob_better,
        "relative_uplift_mean": float(np.mean(relative_uplift)),
        "credible_interval_95": (float(ci_low), float(ci_high)),
        "posterior_samples_control": p_control,
        "posterior_samples_treatment": p_treatment,
    }

    print(f"Control conversion rate:   {result['p_control_mean']:.4f}")
    print(f"Treatment conversion rate: {result['p_treatment_mean']:.4f}")
    print(f"P(treatment > control):    {prob_better:.4f}")
    print(f"Expected relative uplift:  {result['relative_uplift_mean']*100:.1f}%")
    print(f"95% credible interval:     [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

    return result
```

---

## 3. End-to-End Examples

### Example 1 — Bayesian Linear Regression

```python
import numpy as np

# Generate synthetic data: y = 1.5*x1 - 0.8*x2 + 3.0 + noise
np.random.seed(99)
n = 150
X = np.column_stack([np.random.normal(0, 1, n), np.random.normal(0, 1, n)])
true_alpha = 3.0
true_beta = np.array([1.5, -0.8])
y = true_alpha + X @ true_beta + np.random.normal(0, 0.5, n)

# Step 1: Build model
model = build_linear_model(X, y, feature_names=["x1", "x2"])

# Step 2: Prior predictive check
idata_prior = prior_predictive_check(model, n_samples=200, observed_data=y,
                                     output_path="prior_predictive.png")

# Step 3: Sample posterior
idata = sample_and_diagnose(model, draws=2000, tune=1000, chains=4)

# Step 4: Summarize
summary = az.summary(idata, var_names=["alpha", "beta", "sigma"], round_to=3)
print(summary)
# Expected: alpha ~ 3.0, beta[0] ~ 1.5, beta[1] ~ -0.8, sigma ~ 0.5

# Step 5: Diagnostic plots
plot_diagnostics(idata, var_names=["alpha", "beta", "sigma"], output_dir="./plots")
```

### Example 2 — Bayesian A/B Test

```python
# Website conversion experiment:
# Control: 2000 visitors, 120 conversions (6%)
# Treatment: 2100 visitors, 147 conversions (7%)

result = bayesian_ab_test(
    n_control=2000,
    n_conversions_control=120,
    n_treatment=2100,
    n_conversions_treatment=147,
)

# Visualize posterior distributions
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(result["posterior_samples_control"], bins=60, density=True,
        alpha=0.6, label="Control", color="#4C72B0")
ax.hist(result["posterior_samples_treatment"], bins=60, density=True,
        alpha=0.6, label="Treatment", color="#DD8452")
ax.set_xlabel("Conversion rate")
ax.set_title(f"Bayesian A/B Test  |  P(treatment > control) = {result['prob_treatment_better']:.3f}")
ax.legend()
fig.tight_layout()
plt.savefig("ab_test_posteriors.png", dpi=150)

# Decision rule: deploy if P(better) > 0.95 AND relative uplift CI lower bound > 0
if result["prob_treatment_better"] > 0.95 and result["credible_interval_95"][0] > 0:
    print("RECOMMENDATION: Deploy treatment variant.")
else:
    print("RECOMMENDATION: Collect more data or do not deploy.")
```

### Example 3 — Hierarchical Model

```python
# School exam scores: partial pooling across 30 schools
np.random.seed(7)
n_schools = 30
n_students_per_school = 20

school_means = np.random.normal(70, 8, n_schools)  # true school-level means
group_ids = np.repeat(np.arange(n_schools), n_students_per_school)
y_scores = school_means[group_ids] + np.random.normal(0, 5, n_schools * n_students_per_school)

# Build hierarchical model
hier_model = build_hierarchical_model(group_ids, y_scores)

# Sample
idata_hier = sample_and_diagnose(hier_model, draws=2000, tune=1000, chains=4)

# Compare partial pooling vs no-pooling via LOO
no_pool_model = build_linear_model(
    np.eye(n_schools)[group_ids], y_scores,
    feature_names=[f"school_{i}" for i in range(n_schools)]
)
idata_no_pool = sample_and_diagnose(no_pool_model, draws=2000, tune=1000, chains=4)

comparison = compare_models(
    {"hierarchical": idata_hier, "no_pooling": idata_no_pool},
    output_path="model_comparison.png",
)
```

---

## 4. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| R-hat > 1.01 | Chains not converged | Increase `tune` (e.g. 2000); check for multimodal posterior |
| Many divergences | Posterior funnel geometry | Use non-centered parameterization; increase `target_accept` to 0.95 |
| ESS < 400 | High autocorrelation | Increase `draws`; reparameterize; check for near-improper priors |
| `ValueError: observed RV` | `observed=` data has wrong shape | Ensure `y` is a 1D numpy array |
| Sampling extremely slow | High-dimensional model or bad priors | Standardize inputs; use more informative priors; check `pm.model_to_graphviz()` |
| `KeyError: log_likelihood` | LOO called without computing log likelihood | Add `idata_kwargs={"log_likelihood": True}` to `pm.sample()` |
| Prior predictive all near zero | Priors too tight / misspecified scale | Check data scale; use `pm.Normal("sigma", mu=y_sd, sigma=y_sd)` pattern |

---

## 5. Prior Selection Decision Tree

```
Is there strong domain knowledge?
  YES --> Use informative prior (e.g. Normal(literature_mean, literature_sd))
  NO  --> Use weakly informative prior:
            Continuous, unbounded  --> Normal(0, 1) on standardized scale
            Positive only          --> HalfNormal(1) or Exponential(1)
            Probability [0,1]      --> Beta(2, 2) or Beta(1, 1) uninformative
            Count data             --> Poisson or NegativeBinomial

Always run prior_predictive_check() to verify priors produce
plausible outcome values before fitting.
```

