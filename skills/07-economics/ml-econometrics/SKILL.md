---
name: ml-econometrics
description: >
  Use this Skill for machine learning in econometrics: Post-LASSO for high-dimensional
  controls, Double/Debiased ML (DML) for ATE/ATT, cross-fitting, and valid
  post-selection inference.
tags:
  - economics
  - machine-learning
  - LASSO
  - double-debiased-ML
  - high-dimensional
  - causal-inference
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
    - doubleml>=0.6
    - econml>=0.15
    - scikit-learn>=1.2
    - numpy>=1.23
    - pandas>=1.5
last_updated: "2026-03-18"
status: stable
---

# Machine Learning Econometrics

> **TL;DR** — When controls are high-dimensional (many covariates relative to sample
> size), use Post-LASSO for variable selection and Double/Debiased ML (DML) for valid
> causal inference on treatment effects. Use EconML CausalForest for heterogeneous
> treatment effects (CATE).

---

## When to Use

| Situation | Recommended Method |
|---|---|
| Many controls, single treatment, want ATE | Post-LASSO + OLS or DML-PLR |
| Need to partial out high-dimensional X from D and Y | Robinson (1988) / DML |
| Heterogeneous treatment effects across subgroups | CausalForest (EconML) |
| Program evaluation with rich covariate set | AIPW estimator |
| Valid inference after LASSO selection | Post-LASSO (Belloni et al. 2012) |

---

## Background

### Problem: Regularization Bias in LASSO

When estimating β_D in Y = β_D D + X β_X + ε with high-dimensional X, directly using
LASSO on the full model introduces regularization bias into β̂_D. This happens because
LASSO shrinks all coefficients, including the one on the treatment D.

**Naive LASSO estimate**: Biased, no valid standard errors.

### Post-LASSO (Belloni-Chernozhukov-Hansen 2014)

1. Run LASSO of Y on (D, X) to select a set S of controls.
2. Run OLS of Y on D and the selected subset X_S only.
3. Standard OLS inference is valid on the Post-LASSO estimator.

This two-step procedure recovers valid inference when the true model is approximately
sparse (few truly important controls).

### Frisch-Waugh-Lovell and Robinson's Estimator

By the FWL theorem, β_D in Y = β_D D + X β_X + ε equals the coefficient from
regressing M_X Y on M_X D, where M_X is the residual maker.

Robinson (1988) extended this to semiparametric partially linear models:
    Y = θ D + g(X) + ε
    D = m(X) + v

The estimator:
    θ̂ = [Σ (D_i - m̂(X_i)) (Y_i - ĝ(X_i))] / [Σ (D_i - m̂(X_i))²]

where ĝ and m̂ can be any ML estimators.

### Double/Debiased ML (Chernozhukov et al. 2018)

DML corrects for regularization bias via cross-fitting:

1. Split data into K folds.
2. For each fold k:
   - Fit ĝ_{-k}(X) = E[Y|X] and m̂_{-k}(X) = E[D|X] on all folds except k.
   - Compute residuals for fold k: ỹ_i = Y_i - ĝ_{-k}(X_i), d̃_i = D_i - m̂_{-k}(X_i).
3. Regress ỹ on d̃ using all pooled residuals.

Cross-fitting ensures the nuisance estimators have negligible bias on the held-out
sample, restoring √n-consistency and valid Gaussian inference.

### CATE with CausalForest

For heterogeneous effects τ(x) = E[Y(1) - Y(0) | X = x]:

- Fit a GRF (Generalized Random Forest) that solves a local version of the Robinson
  estimator at each point x.
- Valid confidence intervals via the infinitesimal jackknife.
- Feature importance identifies which covariates drive heterogeneity.

---

## Environment Setup

```bash
conda create -n mlecono python=3.11 -y
conda activate mlecono
pip install doubleml>=0.6 econml>=0.15 scikit-learn>=1.2 numpy>=1.23 pandas>=1.5 matplotlib>=3.6

# Verify
python -c "import doubleml; print('doubleml', doubleml.__version__)"
python -c "import econml; print('econml', econml.__version__)"
```

---

## Core Workflow

### Step 1 — Post-LASSO Variable Selection and OLS

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

np.random.seed(42)


def generate_highdim_data(
    n: int = 500,
    p_total: int = 200,
    p_relevant: int = 10,
    ate: float = 2.0,
) -> pd.DataFrame:
    """
    Generate high-dimensional dataset for ML econometrics.

    Y = ate * D + X_relevant @ beta + epsilon
    D = X_relevant @ gamma + v (endogenous through X)

    Args:
        n:           Number of observations.
        p_total:     Total number of control variables.
        p_relevant:  Number of truly relevant controls.
        ate:         True average treatment effect.

    Returns:
        DataFrame with columns: Y, D, X1, ..., Xp_total.
    """
    X = np.random.randn(n, p_total)
    X_rel = X[:, :p_relevant]

    # True coefficients
    beta = np.random.uniform(0.5, 2.0, p_relevant) * np.random.choice([-1, 1], p_relevant)
    gamma = np.random.uniform(0.3, 1.0, p_relevant) * np.random.choice([-1, 1], p_relevant)

    # Treatment (binary after probit)
    D_star = X_rel @ gamma + np.random.randn(n)
    D = (D_star > np.median(D_star)).astype(float)

    # Outcome
    Y = ate * D + X_rel @ beta + np.random.randn(n)

    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(p_total)])
    df.insert(0, "D", D)
    df.insert(0, "Y", Y)
    return df


def post_lasso_ols(
    df: pd.DataFrame,
    outcome: str = "Y",
    treatment: str = "D",
    covariates: list = None,
    cv_folds: int = 5,
) -> dict:
    """
    Post-LASSO: select controls via LassoCV, then run OLS on selected subset.

    This avoids regularization bias on the treatment coefficient β_D.

    Args:
        df:          DataFrame with outcome, treatment, and covariates.
        outcome:     Name of outcome variable.
        treatment:   Name of treatment variable.
        covariates:  List of control variable names. Defaults to all columns
                     except outcome and treatment.
        cv_folds:    Cross-validation folds for LassoCV penalty selection.

    Returns:
        Dictionary with keys: ate_estimate, se, tstat, pvalue, selected_controls, ols_result.
    """
    if covariates is None:
        covariates = [c for c in df.columns if c not in [outcome, treatment]]

    Y = df[outcome].values
    D = df[treatment].values
    X = df[covariates].values

    # Standardize X for LASSO
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stack D with X for selection (Belloni: select controls for Y equation)
    XD = np.hstack([D.reshape(-1, 1), X_scaled])

    # LassoCV: select lambda by cross-validation
    lasso_cv = LassoCV(cv=cv_folds, max_iter=5000, random_state=0, n_jobs=-1)
    lasso_cv.fit(XD, Y)
    alpha_opt = lasso_cv.alpha_

    # Fit Lasso at optimal alpha to identify selected controls
    lasso = Lasso(alpha=alpha_opt, max_iter=5000)
    lasso.fit(XD, Y)
    coef_x = lasso.coef_[1:]       # skip D coefficient (first column)
    selected_idx = np.where(np.abs(coef_x) > 1e-8)[0]
    selected_controls = [covariates[i] for i in selected_idx]

    n_selected = len(selected_controls)
    print(f"Post-LASSO: selected {n_selected} / {len(covariates)} controls at lambda={alpha_opt:.6f}")

    # OLS on treatment + selected controls
    X_selected = df[selected_controls].values if n_selected > 0 else np.zeros((len(Y), 0))
    regressors = np.hstack([D.reshape(-1, 1), X_selected])
    regressors_with_const = sm.add_constant(regressors)
    ols_result = sm.OLS(Y, regressors_with_const).fit(cov_type="HC3")

    # Treatment coefficient is the second element (after constant)
    ate_hat = float(ols_result.params[1])
    se_hat = float(ols_result.bse[1])
    tstat = float(ols_result.tvalues[1])
    pvalue = float(ols_result.pvalues[1])

    print(f"Post-LASSO ATE: {ate_hat:.4f}  SE: {se_hat:.4f}  t: {tstat:.3f}  p: {pvalue:.4f}")

    return {
        "ate_estimate": ate_hat,
        "se": se_hat,
        "tstat": tstat,
        "pvalue": pvalue,
        "selected_controls": selected_controls,
        "ols_result": ols_result,
    }
```

### Step 2 — Double/Debiased ML (PLR Model)

```python
import doubleml as dml
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV, LogisticRegressionCV


def fit_doubleml_plr(
    df: pd.DataFrame,
    outcome: str = "Y",
    treatment: str = "D",
    covariates: list = None,
    ml_type: str = "lasso",
    n_folds: int = 5,
    n_rep: int = 3,
    score: str = "partialling out",
) -> dict:
    """
    Fit a Partially Linear Regression (PLR) model using DoubleML.

    Args:
        df:          DataFrame with outcome, treatment, and controls.
        outcome:     Outcome variable name.
        treatment:   Treatment variable name (binary or continuous).
        covariates:  Control variable names. Defaults to all other columns.
        ml_type:     Nuisance learner: 'lasso' or 'gbm' (gradient boosting).
        n_folds:     Number of cross-fitting folds.
        n_rep:       Number of repeated cross-fitting splits.
        score:       Scoring method: 'partialling out' or 'IV-type'.

    Returns:
        Dictionary with keys: ate, se, pvalue, ci_lower, ci_upper, dml_obj.
    """
    if covariates is None:
        covariates = [c for c in df.columns if c not in [outcome, treatment]]

    # DoubleML data object
    data = dml.DoubleMLData(df, y_col=outcome, d_cols=treatment, x_cols=covariates)

    if ml_type == "lasso":
        ml_l = LassoCV(cv=5, max_iter=5000)       # E[Y|X] learner
        ml_m = LassoCV(cv=5, max_iter=5000)       # E[D|X] learner
    elif ml_type == "gbm":
        ml_l = GradientBoostingRegressor(n_estimators=200, max_depth=3)
        ml_m = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    else:
        raise ValueError(f"Unknown ml_type '{ml_type}'. Choose 'lasso' or 'gbm'.")

    # PLR model
    dml_plr = dml.DoubleMLPLR(
        obj_dml_data=data,
        ml_l=ml_l,
        ml_m=ml_m,
        n_folds=n_folds,
        n_rep=n_rep,
        score=score,
    )
    dml_plr.fit(store_predictions=True, store_models=True)

    summary = dml_plr.summary
    ate = float(summary["coef"].values[0])
    se = float(summary["std err"].values[0])
    pvalue = float(summary["P>|z|"].values[0])
    ci_lower = float(dml_plr.confint().iloc[0, 0])
    ci_upper = float(dml_plr.confint().iloc[0, 1])

    print(f"\nDML PLR ATE: {ate:.4f}  SE: {se:.4f}  p: {pvalue:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        "ate": ate,
        "se": se,
        "pvalue": pvalue,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "dml_obj": dml_plr,
    }
```

### Step 3 — CausalForest for Heterogeneous Treatment Effects

```python
from econml.grf import CausalForest
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
import warnings


def fit_causal_forest_cate(
    df: pd.DataFrame,
    outcome: str = "Y",
    treatment: str = "D",
    covariates: list = None,
    n_estimators: int = 500,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    output_path: str = None,
) -> dict:
    """
    Estimate CATE using EconML CausalForest (Generalized Random Forest).

    Args:
        df:               DataFrame with outcome, treatment, and controls.
        outcome:          Outcome variable name.
        treatment:        Binary treatment variable name.
        covariates:       Control variable names.
        n_estimators:     Number of trees in the causal forest.
        min_samples_leaf: Minimum samples per leaf for regularization.
        random_state:     Random seed.
        output_path:      If provided, save feature importance plot.

    Returns:
        Dictionary with keys: cate_estimates, feature_importance, cate_mean, cate_std,
                              ate_estimate, ate_se, model.
    """
    if covariates is None:
        covariates = [c for c in df.columns if c not in [outcome, treatment]]

    Y = df[outcome].values.reshape(-1, 1)
    T = df[treatment].values.reshape(-1, 1)
    X = df[covariates].values

    cf = CausalForest(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        verbose=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf.fit(X, T, Y)

    # CATE estimates with confidence intervals
    cate, cate_intervals = cf.predict(X, interval=True, alpha=0.05)
    cate = cate.flatten()
    cate_lower = cate_intervals[0].flatten()
    cate_upper = cate_intervals[1].flatten()

    # ATE
    ate_estimate = float(np.mean(cate))
    ate_se = float(np.std(cate) / np.sqrt(len(cate)))

    print(f"\nCausalForest ATE: {ate_estimate:.4f}  SE: {ate_se:.4f}")
    print(f"CATE range: [{cate.min():.4f}, {cate.max():.4f}]")
    print(f"CATE std:   {np.std(cate):.4f}")

    # Feature importance
    feat_imp = cf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": covariates,
        "importance": feat_imp,
    }).sort_values("importance", ascending=False)

    # Plot CATE distribution and feature importance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(cate, bins=30, color="#3498DB", edgecolor="white")
    axes[0].axvline(ate_estimate, color="#C0392B", linewidth=2, linestyle="--",
                    label=f"ATE = {ate_estimate:.3f}")
    axes[0].set_xlabel("CATE")
    axes[0].set_ylabel("Count")
    axes[0].set_title("CATE Distribution")
    axes[0].legend()

    n_top = min(15, len(covariates))
    top_feats = feat_imp_df.head(n_top)
    axes[1].barh(top_feats["feature"], top_feats["importance"], color="#2ECC71")
    axes[1].set_xlabel("Feature Importance")
    axes[1].set_title(f"Top {n_top} Features Driving CATE")
    axes[1].invert_yaxis()

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved CATE plot to {output_path}")
    plt.show()

    return {
        "cate_estimates": cate,
        "cate_lower": cate_lower,
        "cate_upper": cate_upper,
        "feature_importance": feat_imp_df,
        "ate_estimate": ate_estimate,
        "ate_se": ate_se,
        "model": cf,
    }
```

---

## Advanced Usage

### AIPW (Augmented Inverse Probability Weighting) Estimator

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold


def aipw_ate(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """
    AIPW (doubly-robust) ATE estimator with cross-fitting.

    Combines outcome model mu(x,d) and propensity score e(x)=P(D=1|X=x).
    Doubly robust: consistent if either model is correctly specified.

    Args:
        Y:        Outcome array, shape (n,).
        D:        Binary treatment array, shape (n,).
        X:        Covariate matrix, shape (n, p).
        n_folds:  Cross-fitting folds.
        seed:     Random seed.

    Returns:
        Dictionary with keys: ate, se, ci_lower, ci_upper.
    """
    n = len(Y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    mu1_hat = np.zeros(n)   # E[Y|X, D=1]
    mu0_hat = np.zeros(n)   # E[Y|X, D=0]
    e_hat = np.zeros(n)     # P(D=1|X)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        Y_tr, D_tr = Y[train_idx], D[train_idx]

        # Outcome models
        rf1 = RandomForestRegressor(n_estimators=200, random_state=seed)
        rf1.fit(X_tr[D_tr == 1], Y_tr[D_tr == 1])

        rf0 = RandomForestRegressor(n_estimators=200, random_state=seed)
        rf0.fit(X_tr[D_tr == 0], Y_tr[D_tr == 0])

        # Propensity score model
        rfc = RandomForestClassifier(n_estimators=200, random_state=seed)
        rfc.fit(X_tr, D_tr)

        mu1_hat[val_idx] = rf1.predict(X_val)
        mu0_hat[val_idx] = rf0.predict(X_val)
        e_hat[val_idx] = rfc.predict_proba(X_val)[:, 1]

    # Clip propensity scores to avoid extreme weights
    e_hat = np.clip(e_hat, 0.02, 0.98)

    # AIPW influence function
    psi = (
        mu1_hat - mu0_hat
        + D * (Y - mu1_hat) / e_hat
        - (1 - D) * (Y - mu0_hat) / (1 - e_hat)
    )
    ate = float(np.mean(psi))
    se = float(np.std(psi) / np.sqrt(n))

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - 1.96 * se,
        "ci_upper": ate + 1.96 * se,
    }
```

---

## Troubleshooting

| Error / Issue | Cause | Resolution |
|---|---|---|
| DML ATE far from truth | Nuisance model underfits | Switch to GBM or increase tree depth |
| `ConvergenceWarning` in LassoCV | Insufficient iterations | Add `max_iter=10000` |
| CausalForest slow on large n | Many trees, deep forest | Reduce `n_estimators`; increase `min_samples_leaf` |
| AIPW extremely wide CI | Near-zero propensity scores | Widen clip bounds (0.01, 0.99); trim extreme units |
| Post-LASSO selects all variables | Lambda too small | Use `alpha_multiplier` > 1 or `LassoIC` with BIC |
| Post-LASSO selects zero variables | Lambda too large | Check scaling; reduce `cv_folds`; inspect `lasso_cv.alpha_` |
| `KeyError` in DoubleML | Column name mismatch | Ensure `x_cols` list exactly matches DataFrame columns |

---

## External Resources

- Chernozhukov, V. et al. (2018). "Double/Debiased Machine Learning." *Econometrics Journal*, 21(1), C1–C68.
  <https://doi.org/10.1111/ectj.12097>
- Belloni, A., Chernozhukov, V., Hansen, C. (2014). "High-Dimensional Methods and Inference
  on Structural and Treatment Effects." *Journal of Economic Perspectives*, 28(2), 29–50.
- Athey, S., Tibshirani, J., Wager, S. (2019). "Generalized Random Forests." *Annals of Statistics*, 47(2).
- `DoubleML` package docs: <https://docs.doubleml.org/>
- `EconML` package docs: <https://econml.azurewebsites.net/>

---

## Examples

### Example 1 — Post-LASSO on High-Dimensional Data

```python
df = generate_highdim_data(n=600, p_total=200, p_relevant=10, ate=2.0)

result_pl = post_lasso_ols(df, outcome="Y", treatment="D")
print(f"True ATE = 2.0  |  Post-LASSO ATE = {result_pl['ate_estimate']:.4f}")
print(f"Selected {len(result_pl['selected_controls'])} controls")
```

### Example 2 — DML PLR and CausalForest Comparison

```python
df = generate_highdim_data(n=800, p_total=100, p_relevant=10, ate=2.0)
covariates = [c for c in df.columns if c.startswith("X")]

# DML
dml_result = fit_doubleml_plr(df, outcome="Y", treatment="D",
                               covariates=covariates, ml_type="lasso")

# CausalForest CATE
cf_result = fit_causal_forest_cate(df, outcome="Y", treatment="D",
                                    covariates=covariates, output_path="cate_plot.png")

print("\nSummary comparison:")
print(f"  True ATE   = 2.0000")
print(f"  DML ATE    = {dml_result['ate']:.4f}  95% CI: [{dml_result['ci_lower']:.4f}, {dml_result['ci_upper']:.4f}]")
print(f"  CF ATE     = {cf_result['ate_estimate']:.4f}  SE: {cf_result['ate_se']:.4f}")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Post-LASSO, DML PLR, CausalForest CATE, AIPW |
