---
name: ml-for-research
description: >
  Use this Skill to apply machine learning in research: scikit-learn pipelines,
  FLAML AutoML, SHAP explainability, nested cross-validation, and model cards.
tags:
  - universal
  - machine-learning
  - scikit-learn
  - SHAP
  - AutoML
  - explainability
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
    - scikit-learn>=1.2
    - flaml>=2.0
    - shap>=0.42
    - optuna>=3.0
    - pandas>=1.5
    - matplotlib>=3.6
last_updated: "2026-03-18"
status: stable
---

# Machine Learning for Research — Pipelines, AutoML, and Explainability

> **TL;DR** — Build reproducible ML workflows for research: scikit-learn
> ColumnTransformer pipelines, nested cross-validation, FLAML AutoML with
> baseline comparison, Optuna hyperparameter search, SHAP TreeExplainer
> summaries, calibration curves, and model cards for transparent reporting.

---

## When to Use This Skill

Use this Skill whenever you need to:

- Apply supervised ML to tabular research data (clinical, social science, omics)
- Construct a preprocessing pipeline handling mixed numeric and categorical features
- Evaluate models with nested cross-validation to obtain unbiased performance estimates
- Run AutoML (FLAML) with a time budget and compare it to a simple baseline
- Tune hyperparameters with Optuna TPE search
- Explain model predictions globally (SHAP summary) and locally (waterfall plots)
- Assess classifier calibration and apply probability calibration
- Document model properties in a structured model card for reproducible reporting

| Task | When to apply |
|---|---|
| ColumnTransformer pipeline | Any dataset with mixed feature types |
| Nested CV | Primary model evaluation for a research paper |
| FLAML AutoML | Quick strong baseline; feature selection exploration |
| SHAP TreeExplainer | Tree-based models (RF, XGBoost, LightGBM) |
| CalibratedClassifierCV | Any model used for probabilistic risk prediction |
| Model card | Methods section reporting; supplementary material |

---

## Background & Key Concepts

### scikit-learn Pipeline and ColumnTransformer

A `Pipeline` chains preprocessing steps and a final estimator so that
cross-validation splits are applied correctly — the scaler and imputer are fit
only on the training fold, preventing data leakage.

`ColumnTransformer` applies different transformers to numeric and categorical
columns in parallel, then concatenates the results:

- **Numeric**: `SimpleImputer(strategy='median')` → `StandardScaler()`
- **Categorical**: `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`

### Train / Validation / Test Split

| Split | Fraction | Purpose |
|---|---|---|
| Train | 60 % | Model fitting and cross-validation |
| Validation | 20 % | Hyperparameter selection and early stopping |
| Test | 20 % | Final unbiased performance estimate (touch once) |

Use `stratify=y` for classification to preserve class balance across splits.

### Nested Cross-Validation

Nested CV prevents optimistic bias when tuning and evaluating on the same data:

- **Outer loop**: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` — provides
  15 performance estimates that are averaged for the paper.
- **Inner loop**: `GridSearchCV` or `RandomizedSearchCV` — tunes hyperparameters
  on each outer training fold independently.

### FLAML AutoML

FLAML (Fast and Lightweight AutoML) explores model families and hyperparameters
within a user-defined `time_budget` (seconds). Key parameters:

| Parameter | Recommendation |
|---|---|
| `time_budget` | 60–300 s for initial exploration; 3600 s for final run |
| `task` | `'classification'` or `'regression'` |
| `metric` | `'roc_auc'`, `'accuracy'`, `'r2'`, `'rmse'` |
| `estimator_list` | `['lgbm', 'rf', 'xgboost', 'linear']` or `'auto'` |

### SHAP TreeExplainer

SHAP (SHapley Additive exPlanations) provides feature attributions that satisfy
desirable properties (efficiency, symmetry, dummy). `TreeExplainer` computes
exact SHAP values for tree-based models in polynomial time.

Key plots:
- **summary_plot**: global feature importance ranked by mean |SHAP|
- **waterfall_plot**: single-instance explanation showing additive contributions
- **dependence_plot**: interaction between one feature's value and its SHAP value

### Model Calibration

A well-calibrated model produces probabilities that match empirical frequencies:
predicted probability 0.7 should correspond to ~70 % event rate. Use
`CalibratedClassifierCV(method='isotonic')` for post-hoc calibration and
`CalibrationDisplay` for the reliability diagram.

---

## Environment Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install required packages
pip install "scikit-learn>=1.2" "flaml>=2.0" "shap>=0.42" \
            "optuna>=3.0" "pandas>=1.5" "matplotlib>=3.6"

# Verify installation
python -c "import sklearn, flaml, shap, optuna, pandas, matplotlib; print('Setup OK')"
```

Optional: suppress verbose Optuna and FLAML logs during development:

```bash
export OPTUNA_VERBOSITY=WARNING    # reduce Optuna console output
export FLAML_VERBOSITY=0           # 0 = silent, 1 = info, 2 = debug
```

---

## Core Workflow

### Step 1 — Nested CV Pipeline with SHAP Summary Plot

```python
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import make_classification
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def build_preprocessing_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for mixed numeric and categorical features.

    Args:
        numeric_cols:     List of numeric feature column names.
        categorical_cols: List of categorical feature column names.

    Returns:
        Fitted ColumnTransformer (fit inside a Pipeline to avoid leakage).
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ], remainder="drop")


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    outer_n_splits: int = 5,
    outer_n_repeats: int = 3,
    inner_cv_folds: int = 5,
    param_grid: dict | None = None,
    random_state: int = 42,
) -> dict:
    """
    Run nested cross-validation for a RandomForestClassifier.

    Args:
        X:                  Feature DataFrame.
        y:                  Binary target Series.
        numeric_cols:       Numeric feature column names.
        categorical_cols:   Categorical feature column names.
        outer_n_splits:     Number of outer CV folds.
        outer_n_repeats:    Number of outer CV repeats.
        inner_cv_folds:     Number of inner CV folds for hyperparameter tuning.
        param_grid:         Hyperparameter grid for GridSearchCV. Uses default if None.
        random_state:       Random seed.

    Returns:
        Dictionary with outer AUC scores list, mean AUC, and std AUC.
    """
    if param_grid is None:
        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_leaf": [1, 5],
        }

    preproc = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    pipeline = Pipeline([("preproc", preproc), ("clf", clf)])

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_n_splits, n_repeats=outer_n_repeats, random_state=random_state
    )
    inner_cv = RepeatedStratifiedKFold(n_splits=inner_cv_folds, n_repeats=1,
                                        random_state=random_state)

    outer_scores = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        gs = GridSearchCV(
            pipeline, param_grid, cv=inner_cv,
            scoring="roc_auc", n_jobs=-1, refit=True,
        )
        gs.fit(X_train, y_train)
        y_prob = gs.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        outer_scores.append(auc)

        if fold_idx % 5 == 0:
            print(f"  Outer fold {fold_idx+1}/{outer_n_splits*outer_n_repeats}: AUC = {auc:.4f}")

    mean_auc = float(np.mean(outer_scores))
    std_auc = float(np.std(outer_scores, ddof=1))
    print(f"\nNested CV result: AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    return {"outer_scores": outer_scores, "mean_auc": mean_auc, "std_auc": std_auc}


def compute_shap_values_and_plot(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    output_path: str = "shap_summary.png",
    max_display: int = 15,
) -> None:
    """
    Fit a RandomForest on X_train, compute SHAP values on X_test, and save summary plot.

    Args:
        X_train:          Training features (DataFrame).
        X_test:           Test features (DataFrame).
        y_train:          Training labels.
        numeric_cols:     Numeric feature column names.
        categorical_cols: Categorical feature column names.
        output_path:      Path to save the SHAP summary plot.
        max_display:      Maximum features to display in the summary plot.
    """
    preproc = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    pipeline = Pipeline([("preproc", preproc), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    # Transform test set for SHAP
    X_test_transformed = pipeline.named_steps["preproc"].transform(X_test)
    feature_names = (
        numeric_cols
        + list(pipeline.named_steps["preproc"]
               .named_transformers_["cat"]
               .named_steps["encoder"]
               .get_feature_names_out(categorical_cols))
    )

    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_transformed)

    # For binary classification, take class-1 SHAP values
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        sv,
        X_test_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {output_path}")


if __name__ == "__main__":
    # Generate synthetic research-like classification data
    X_raw, y_raw = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_redundant=4, random_state=0,
    )
    feature_names_num = [f"lab_{i}" for i in range(15)]
    feature_names_cat = [f"group_{i}" for i in range(5)]
    df = pd.DataFrame(X_raw, columns=feature_names_num + feature_names_cat)
    # Discretize last 5 features as fake categorical
    for col in feature_names_cat:
        df[col] = pd.cut(df[col], bins=3, labels=["low", "med", "high"])
    target = pd.Series(y_raw, name="outcome")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df, target, test_size=0.20, stratify=target, random_state=42
    )

    print("Running nested cross-validation...")
    results = run_nested_cv(
        X_train_full, y_train_full,
        numeric_cols=feature_names_num,
        categorical_cols=feature_names_cat,
    )
    print(f"\nFinal nested CV AUC: {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")

    print("\nComputing SHAP values...")
    compute_shap_values_and_plot(
        X_train_full, X_test, y_train_full,
        numeric_cols=feature_names_num,
        categorical_cols=feature_names_cat,
        output_path="shap_summary.png",
    )
```

### Step 2 — FLAML AutoML and Comparison with Baseline

```python
import time
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from flaml import AutoML


def run_flaml_automl(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    time_budget: int = 60,
    metric: str = "roc_auc",
    task: str = "classification",
    estimator_list: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """
    Run FLAML AutoML and compare with a logistic regression baseline.

    Args:
        X_train:        Training features.
        y_train:        Training labels.
        X_test:         Test features.
        y_test:         Test labels.
        time_budget:    Maximum seconds for FLAML search.
        metric:         Optimization metric (e.g., 'roc_auc', 'accuracy').
        task:           ML task type: 'classification' or 'regression'.
        estimator_list: Model families to consider; None = auto.
        seed:           Random seed.

    Returns:
        Dictionary with AutoML and baseline AUC, best model name, and elapsed time.
    """
    if estimator_list is None:
        estimator_list = ["lgbm", "rf", "xgboost", "lrl2"]

    # Baseline: majority class
    dummy = DummyClassifier(strategy="stratified", random_state=seed)
    dummy.fit(X_train, y_train)
    dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])

    # Logistic regression baseline
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    # FLAML AutoML
    automl = AutoML()
    settings = {
        "time_budget": time_budget,
        "metric": metric,
        "task": task,
        "estimator_list": estimator_list,
        "seed": seed,
        "verbose": 0,
    }

    t0 = time.time()
    automl.fit(X_train, y_train, **settings)
    elapsed = round(time.time() - t0, 1)

    automl_prob = automl.predict_proba(X_test)[:, 1]
    automl_auc = roc_auc_score(y_test, automl_prob)

    print(f"Baseline (Dummy)  AUC: {dummy_auc:.4f}")
    print(f"Baseline (LogReg) AUC: {lr_auc:.4f}")
    print(f"FLAML AutoML      AUC: {automl_auc:.4f} "
          f"[best model: {automl.best_estimator}, elapsed: {elapsed}s]")

    return {
        "dummy_auc": round(dummy_auc, 4),
        "logreg_auc": round(lr_auc, 4),
        "automl_auc": round(automl_auc, 4),
        "best_estimator": automl.best_estimator,
        "best_config": automl.best_config,
        "elapsed_sec": elapsed,
        "model": automl,
    }


if __name__ == "__main__":
    X, y = make_classification(n_samples=800, n_features=25, n_informative=12,
                                random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=1)
    result = run_flaml_automl(X_tr, y_tr, X_te, y_te, time_budget=60)
    print(f"\nAUC lift over LogReg: "
          f"{(result['automl_auc'] - result['logreg_auc'])*100:+.2f} pp")
```

### Step 3 — Model Card Generation

```python
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report,
)


MODEL_CARD_TEMPLATE = {
    "model_name": "",
    "version": "1.0.0",
    "date_trained": "",
    "intended_use": {
        "primary_use": "",
        "out_of_scope": [],
    },
    "training_data": {
        "source": "",
        "n_samples": 0,
        "n_features": 0,
        "target_prevalence": 0.0,
        "preprocessing": [],
    },
    "evaluation_data": {
        "source": "",
        "n_samples": 0,
    },
    "performance": {
        "roc_auc": 0.0,
        "average_precision": 0.0,
        "brier_score": 0.0,
        "calibrated": False,
    },
    "limitations": [],
    "ethical_considerations": [],
    "caveats_and_recommendations": [],
}


def generate_model_card(
    model_name: str,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    training_info: dict,
    limitations: list[str],
    ethical_considerations: list[str],
    output_path: str = "model_card.json",
) -> dict:
    """
    Populate and save a model card dictionary with performance metrics.

    Args:
        model_name:             Short name for the model.
        y_test:                 True binary labels on the held-out test set.
        y_prob:                 Predicted probabilities for the positive class.
        training_info:          Dict with keys: source, n_train, n_features,
                                prevalence, preprocessing.
        limitations:            List of known model limitations.
        ethical_considerations: List of ethical considerations.
        output_path:            File path to save the JSON model card.

    Returns:
        Populated model card dictionary.
    """
    import copy
    card = copy.deepcopy(MODEL_CARD_TEMPLATE)
    card["model_name"] = model_name
    card["version"] = "1.0.0"
    card["date_trained"] = datetime.date.today().isoformat()
    card["training_data"]["source"] = training_info.get("source", "")
    card["training_data"]["n_samples"] = training_info.get("n_train", 0)
    card["training_data"]["n_features"] = training_info.get("n_features", 0)
    card["training_data"]["target_prevalence"] = round(float(np.mean(y_test)), 4)
    card["training_data"]["preprocessing"] = training_info.get("preprocessing", [])
    card["evaluation_data"]["n_samples"] = len(y_test)
    card["performance"]["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)
    card["performance"]["average_precision"] = round(
        average_precision_score(y_test, y_prob), 4
    )
    card["performance"]["brier_score"] = round(brier_score_loss(y_test, y_prob), 4)
    card["limitations"] = limitations
    card["ethical_considerations"] = ethical_considerations

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    print(f"Model card saved to {output_path}")
    return card


def plot_calibration_curve(
    y_test: np.ndarray,
    y_prob_uncal: np.ndarray,
    y_prob_cal: np.ndarray,
    output_path: str = "calibration_curve.png",
) -> None:
    """Plot reliability diagram comparing uncalibrated and calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(7, 5))
    CalibrationDisplay.from_predictions(
        y_test, y_prob_uncal, n_bins=10, ax=ax, name="Uncalibrated"
    )
    CalibrationDisplay.from_predictions(
        y_test, y_prob_cal, n_bins=10, ax=ax, name="Calibrated (isotonic)"
    )
    ax.set_title("Calibration Curve (Reliability Diagram)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Calibration curve saved to {output_path}")


if __name__ == "__main__":
    X, y = make_classification(n_samples=600, n_features=20,
                                n_informative=10, random_state=5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=5)

    rf = RandomForestClassifier(n_estimators=100, random_state=5)
    rf.fit(X_tr, y_tr)
    y_prob_uncal = rf.predict_proba(X_te)[:, 1]

    cal_rf = CalibratedClassifierCV(rf, method="isotonic", cv=5)
    cal_rf.fit(X_tr, y_tr)
    y_prob_cal = cal_rf.predict_proba(X_te)[:, 1]

    plot_calibration_curve(y_te, y_prob_uncal, y_prob_cal)

    card = generate_model_card(
        model_name="RandomForest-AD-Classifier",
        y_test=y_te,
        y_prob=y_prob_cal,
        training_info={
            "source": "Synthetic research dataset",
            "n_train": len(X_tr),
            "n_features": X_tr.shape[1],
            "prevalence": float(y_tr.mean()),
            "preprocessing": ["StandardScaler", "SimpleImputer(median)"],
        },
        limitations=[
            "Trained on synthetic data; external validation required.",
            "Does not account for temporal drift in feature distributions.",
        ],
        ethical_considerations=[
            "Model should not be used for clinical decision-making without "
            "prospective validation.",
            "Performance has not been assessed in subgroups defined by race, "
            "sex, or socioeconomic status.",
        ],
    )
    print("\nModel Card:")
    import json
    print(json.dumps(card["performance"], indent=2))
```

---

## Advanced Usage

### Optuna Hyperparameter Search

```python
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
) -> float:
    """
    Optuna objective for RandomForest hyperparameter tuning.

    Suggests n_estimators and max_depth via the trial object, fits a
    RandomForest with cross-validation, and returns mean AUC.

    Args:
        trial:    Optuna trial object.
        X:        Feature matrix.
        y:        Binary labels.
        cv_folds: Number of stratified CV folds.

    Returns:
        Mean ROC-AUC across CV folds (to be maximized).
    """
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_float("max_features", 0.3, 1.0)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


X, y = make_classification(n_samples=500, n_features=20, random_state=0)
study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=50, show_progress_bar=False)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

### Permutation Feature Importance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=400, n_features=15, n_informative=8, random_state=3)
feature_names = [f"feature_{i:02d}" for i in range(X.shape[1])]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=3)

clf = RandomForestClassifier(n_estimators=100, random_state=3)
clf.fit(X_tr, y_tr)

perm = permutation_importance(clf, X_te, y_te, n_repeats=20,
                               scoring="roc_auc", random_state=3, n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)

print(perm_df.head(10).to_string(index=False))
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ValueError: could not convert string to float` | Categorical columns passed to scaler | Ensure `ColumnTransformer` routes categoricals to `OneHotEncoder` |
| SHAP `TypeError: unhashable type` | Non-numeric transformed features | Use `get_feature_names_out()` after transformation |
| FLAML timeout reached immediately | `time_budget` too small | Set `time_budget` ≥ 30 s; check system clock |
| Nested CV very slow | Too many grid combinations | Use `RandomizedSearchCV` instead of `GridSearchCV` |
| Calibration curve not flat | Overfit uncalibrated probabilities | Apply `CalibratedClassifierCV(method='isotonic', cv=5)` |
| Optuna `TrialPruned` errors | Intermediate value reporting missing | Remove pruning callbacks if not using early stopping |
| Memory error in SHAP | Dataset too large for TreeExplainer | Subsample `X_test` to 500–1000 rows for SHAP computation |

---

## External Resources

- scikit-learn Pipeline docs: <https://scikit-learn.org/stable/modules/compose.html>
- FLAML documentation: <https://microsoft.github.io/FLAML/>
- SHAP documentation: <https://shap.readthedocs.io/>
- Optuna documentation: <https://optuna.readthedocs.io/>
- Nested CV guide: <https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html>
- Model Cards for Model Reporting (Mitchell et al. 2019): <https://arxiv.org/abs/1810.03993>
- Calibration of classifiers: <https://scikit-learn.org/stable/modules/calibration.html>

---

## Examples

### Example 1 — Full Nested CV on a Clinical Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset (replace with real data path)
# df = pd.read_csv("clinical_data.csv")
# For demonstration, use synthetic data
from sklearn.datasets import make_classification
import numpy as np

X_raw, y_raw = make_classification(n_samples=600, n_features=18,
                                    n_informative=9, random_state=10)
num_cols = [f"biomarker_{i}" for i in range(13)]
cat_cols = [f"category_{i}" for i in range(5)]
df = pd.DataFrame(X_raw, columns=num_cols + cat_cols)
for col in cat_cols:
    df[col] = pd.cut(df[col], bins=3, labels=["A", "B", "C"])
y = pd.Series(y_raw, name="outcome")

X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.2,
                                            stratify=y, random_state=42)
nested_results = run_nested_cv(
    X_tr, y_tr, numeric_cols=num_cols, categorical_cols=cat_cols,
    outer_n_splits=5, outer_n_repeats=3,
)
print(f"Nested CV AUC: {nested_results['mean_auc']:.3f} ± {nested_results['std_auc']:.3f}")
```

### Example 2 — FLAML AutoML vs Logistic Regression Baseline

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=30,
                            n_informative=15, random_state=99)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=99)
result = run_flaml_automl(X_tr, y_tr, X_te, y_te, time_budget=60)
print(f"\nBest model: {result['best_estimator']}")
print(f"AUC improvement over LogReg: "
      f"{(result['automl_auc'] - result['logreg_auc'])*100:+.2f} pp")
```

### Example 3 — Model Card with Calibration

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

X, y = make_classification(n_samples=700, n_features=20, random_state=7)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=7)
base = RandomForestClassifier(n_estimators=100, random_state=7)
cal = CalibratedClassifierCV(base, method="isotonic", cv=5)
cal.fit(X_tr, y_tr)
y_prob = cal.predict_proba(X_te)[:, 1]

card = generate_model_card(
    model_name="CalibratedRF-v1",
    y_test=y_te,
    y_prob=y_prob,
    training_info={
        "source": "Demo dataset",
        "n_train": len(X_tr),
        "n_features": 20,
        "preprocessing": ["StandardScaler"],
    },
    limitations=["Synthetic data only; requires real-world validation."],
    ethical_considerations=["No subgroup fairness analysis performed."],
    output_path="model_card_demo.json",
)
import json
print(json.dumps(card["performance"], indent=2))
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — nested CV, FLAML AutoML, SHAP, calibration, Optuna, model cards |
