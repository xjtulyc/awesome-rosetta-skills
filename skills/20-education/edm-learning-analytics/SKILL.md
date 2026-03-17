---
name: edm-learning-analytics
description: >
  Educational data mining: BKT knowledge tracing, dropout survival analysis, IRT,
  sequence mining, and at-risk prediction from LMS logs.
tags:
  - education
  - learning-analytics
  - bayesian-knowledge-tracing
  - survival-analysis
  - sequence-mining
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
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scipy>=1.10.0
  - scikit-learn>=1.3.0
  - lifelines>=0.27.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# edm-learning-analytics: Educational Data Mining

This skill covers the full educational data mining (EDM) pipeline: ingesting LMS event
logs, modelling student knowledge with Bayesian Knowledge Tracing, predicting dropout
with survival analysis, mining activity sequences, and running Item Response Theory
assessments.

## Installation

```bash
pip install numpy pandas scipy scikit-learn lifelines matplotlib
# Optional: pyBKT for a production-grade BKT implementation
pip install pyBKT
```

---

## 1. Loading and Preparing LMS Logs

Moodle exports a CSV with columns such as `Time`, `User full name`, `Event name`,
`Component`.  Canvas exports a similar structure.

```python
import pandas as pd
import numpy as np

def load_lms_logs(
    path: str,
    time_col: str = "Time",
    user_col: str = "User full name",
    event_col: str = "Event name",
    component_col: str = "Component",
) -> pd.DataFrame:
    """
    Load a Moodle/Canvas CSV event log and normalise column names.

    Returns a DataFrame with columns: timestamp, user_id, event, component.
    """
    df = pd.read_csv(path, parse_dates=[time_col], infer_datetime_format=True)
    df = df.rename(columns={
        time_col: "timestamp",
        user_col: "user_id",
        event_col: "event",
        component_col: "component",
    })
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} events, {df['user_id'].nunique()} students")
    return df


def load_assessment_log(
    path: str,
    user_col: str = "user_id",
    skill_col: str = "skill",
    attempt_col: str = "attempt",
    correct_col: str = "correct",
) -> pd.DataFrame:
    """
    Load an assessment response log (e.g., from a quiz or intelligent tutoring system).

    Expected columns: user_id, skill, attempt (1-indexed), correct (0/1).
    """
    df = pd.read_csv(path)
    required = {user_col, skill_col, attempt_col, correct_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.rename(columns={
        user_col: "user_id", skill_col: "skill",
        attempt_col: "attempt", correct_col: "correct",
    })
    df["correct"] = df["correct"].astype(int)
    return df
```

---

## 2. Learning Curve Analysis

Power-law learning curves show how accuracy improves with practice.

```python
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def _power_law(n, a, b, c):
    """Error rate model: error(n) = a * n^(-b) + c  (n = opportunity count)."""
    return a * np.power(n, -b) + c


def compute_learning_curves(
    df: pd.DataFrame,
    skill: str,
    attempt_col: str = "attempt",
    correct_col: str = "correct",
) -> dict:
    """
    Fit a power-law learning curve for a given skill.

    Parameters
    ----------
    df : pd.DataFrame  — assessment log
    skill : str        — skill name (filtered from df['skill'])
    attempt_col : str
    correct_col : str

    Returns
    -------
    dict with keys: skill, params (a, b, c), r2, opportunity_error_rate
    """
    sub = df[df["skill"] == skill].copy()
    opp = (
        sub.groupby(attempt_col)[correct_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_students"})
    )
    opp["error_rate"] = 1.0 - opp["accuracy"]
    opp = opp[opp[attempt_col] >= 1].sort_values(attempt_col)

    n = opp[attempt_col].values.astype(float)
    err = opp["error_rate"].values

    try:
        popt, _ = curve_fit(
            _power_law, n, err,
            p0=[0.5, 0.5, 0.1],
            bounds=([0, 0.01, 0], [1.0, 3.0, 0.5]),
            maxfev=5000,
        )
        err_pred = _power_law(n, *popt)
        ss_res = np.sum((err - err_pred) ** 2)
        ss_tot = np.sum((err - np.mean(err)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    except RuntimeError:
        popt = [np.nan, np.nan, np.nan]
        r2 = np.nan

    return {
        "skill": skill,
        "params": dict(zip(["a", "b", "c"], popt)),
        "r2": r2,
        "opportunity_df": opp,
    }
```

---

## 3. Bayesian Knowledge Tracing (Manual BKT)

BKT tracks latent knowledge state with four parameters:
- `p_init`: prior probability of knowing the skill
- `p_transit`: probability of transitioning from unknown to known after each opportunity
- `p_slip`: P(wrong | knows)
- `p_guess`: P(correct | doesn't know)

```python
def run_bkt(
    df: pd.DataFrame,
    skill_col: str = "skill",
    correct_col: str = "correct",
    p_init: float = 0.1,
    p_transit: float = 0.1,
    p_slip: float = 0.1,
    p_guess: float = 0.2,
) -> pd.DataFrame:
    """
    Run per-student Bayesian Knowledge Tracing for each skill.

    Returns a DataFrame with columns:
    user_id, skill, attempt, correct, p_know (posterior knowledge estimate).
    """
    results = []
    for (user, skill), group in df.groupby(["user_id", skill_col]):
        group = group.sort_values("attempt").reset_index(drop=True)
        p_know = p_init
        for _, row in group.iterrows():
            correct = int(row[correct_col])

            # E-step: update P(know | observation)
            if correct == 1:
                p_obs_given_know = 1 - p_slip
                p_obs_given_not = p_guess
            else:
                p_obs_given_know = p_slip
                p_obs_given_not = 1 - p_guess

            numerator = p_obs_given_know * p_know
            denominator = numerator + p_obs_given_not * (1 - p_know)
            p_know_given_obs = numerator / denominator if denominator > 0 else p_know

            # M-step: forward prediction (transit)
            p_know_next = p_know_given_obs + (1 - p_know_given_obs) * p_transit

            results.append({
                "user_id": user,
                "skill": skill,
                "attempt": int(row["attempt"]),
                "correct": correct,
                "p_know": round(p_know_next, 4),
            })
            p_know = p_know_next

    return pd.DataFrame(results)


def plot_knowledge_state(
    bkt_df: pd.DataFrame,
    user_id,
    skill: str,
    ax=None,
):
    """
    Plot the knowledge state trajectory for a single student and skill.
    """
    sub = bkt_df[(bkt_df["user_id"] == user_id) & (bkt_df["skill"] == skill)]
    if sub.empty:
        raise ValueError(f"No data for user={user_id}, skill={skill}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(sub["attempt"], sub["p_know"], marker="o", linewidth=2, label="P(know)")
    correct_mask = sub["correct"] == 1
    ax.scatter(sub["attempt"][correct_mask], [0.05] * correct_mask.sum(),
               marker="^", color="green", zorder=5, label="Correct")
    ax.scatter(sub["attempt"][~correct_mask], [0.05] * (~correct_mask).sum(),
               marker="v", color="red", zorder=5, label="Incorrect")
    ax.axhline(0.95, linestyle="--", color="gray", label="Mastery threshold")
    ax.set_xlabel("Attempt Number")
    ax.set_ylabel("P(Knowledge)")
    ax.set_title(f"BKT — User: {user_id}, Skill: {skill}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    return ax
```

---

## 4. Dropout Prediction with Cox Proportional Hazards

```python
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler


def predict_dropout_cox(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    covariates: list,
    test_df: pd.DataFrame = None,
) -> dict:
    """
    Fit a Cox PH model and return hazard ratios and predicted survival functions.

    Parameters
    ----------
    df : pd.DataFrame  — one row per student; includes time_col, event_col, covariates
    time_col : str     — time-to-dropout (or censoring) in days / weeks
    event_col : str    — 1 = dropped out, 0 = censored (still enrolled)
    covariates : list  — feature column names
    test_df : optional — if provided, predictions are also returned for test set

    Returns
    -------
    dict with keys: model, summary, concordance_index, predictions (if test_df given)
    """
    train = df[[time_col, event_col] + covariates].dropna().copy()

    scaler = StandardScaler()
    train[covariates] = scaler.fit_transform(train[covariates])

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train, duration_col=time_col, event_col=event_col)

    result = {
        "model": cph,
        "summary": cph.summary,
        "concordance_index": cph.concordance_index_,
    }

    if test_df is not None:
        test = test_df[covariates].copy().fillna(test_df[covariates].mean())
        test[covariates] = scaler.transform(test[covariates])
        sf = cph.predict_survival_function(test)
        result["predictions"] = sf

    return result
```

---

## 5. Activity Sequence Mining (PrefixSpan)

```python
def mine_activity_sequences(
    sequences: list,
    min_support: float = 0.1,
    max_pattern_len: int = 4,
) -> list:
    """
    Mine frequent sequential patterns using a pure-Python PrefixSpan implementation.

    Parameters
    ----------
    sequences : list of list of str  — ordered activity sequences per student
    min_support : float  — minimum fraction of sequences that must contain the pattern
    max_pattern_len : int

    Returns
    -------
    list of (pattern, support_count) sorted by support descending
    """
    n = len(sequences)
    min_count = int(np.ceil(min_support * n))

    def _project(database, prefix_item):
        """Return projected database after prefix_item."""
        projected = []
        for seq in database:
            for i, item in enumerate(seq):
                if item == prefix_item:
                    projected.append(seq[i + 1:])
                    break
        return projected

    def _frequent_items(database):
        counts = {}
        for seq in database:
            for item in set(seq):
                counts[item] = counts.get(item, 0) + 1
        return {item: cnt for item, cnt in counts.items() if cnt >= min_count}

    results = []

    def _prefixspan(prefix, database):
        if len(prefix) >= max_pattern_len:
            return
        for item, count in _frequent_items(database).items():
            new_prefix = prefix + [item]
            results.append((new_prefix, count))
            projected = _project(database, item)
            _prefixspan(new_prefix, projected)

    _prefixspan([], sequences)
    results.sort(key=lambda x: -x[1])
    return results
```

---

## 6. Item Response Theory (1PL Rasch Model)

```python
from scipy.optimize import minimize


def fit_irt_1pl(response_matrix: np.ndarray) -> dict:
    """
    Fit a 1-parameter logistic (Rasch) IRT model via MLE.

    Parameters
    ----------
    response_matrix : np.ndarray  shape (n_students, n_items)
        Binary matrix: 1 = correct, 0 = incorrect, NaN = missing.

    Returns
    -------
    dict with keys: difficulty (n_items,), ability (n_students,), log_likelihood
    """
    n_students, n_items = response_matrix.shape

    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    def neg_log_likelihood(params):
        abilities = params[:n_students]
        difficulties = params[n_students:]
        theta = abilities[:, None] - difficulties[None, :]  # (n_students, n_items)
        p = logistic(theta)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        mask = ~np.isnan(response_matrix)
        ll = (
            response_matrix[mask] * np.log(p[mask])
            + (1 - response_matrix[mask]) * np.log(1 - p[mask])
        )
        return -ll.sum()

    x0 = np.zeros(n_students + n_items)
    res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", options={"maxiter": 500})

    abilities = res.x[:n_students]
    difficulties = res.x[n_students:]

    # Anchor: fix mean difficulty = 0
    shift = np.mean(difficulties)
    difficulties -= shift
    abilities -= shift

    return {
        "ability": abilities,
        "difficulty": difficulties,
        "log_likelihood": -res.fun,
    }
```

---

## 7. Examples

### Example A — Predict At-Risk Students from Early LMS Engagement

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Suppose we have a Moodle log exported as CSV
# df_logs = load_lms_logs("/data/moodle_course_2024.csv")

# Simulate a feature table (replace with real feature engineering from LMS logs)
np.random.seed(42)
n = 400
df_students = pd.DataFrame({
    "user_id": range(n),
    "week1_logins": np.random.poisson(5, n),
    "week1_forum_posts": np.random.poisson(1.5, n),
    "week1_quiz_score": np.random.beta(3, 2, n) * 100,
    "week1_video_minutes": np.random.exponential(30, n),
    "dropout_week": np.random.randint(2, 16, n),
    "dropped_out": np.random.binomial(1, 0.25, n),
})

covariates = ["week1_logins", "week1_forum_posts", "week1_quiz_score", "week1_video_minutes"]

# Cox PH model
cox_result = predict_dropout_cox(
    df_students,
    time_col="dropout_week",
    event_col="dropped_out",
    covariates=covariates,
)
print(f"Cox C-index: {cox_result['concordance_index']:.3f}")
print(cox_result["summary"][["coef", "exp(coef)", "p"]].to_string())

# Logistic regression (early binary warning)
X = df_students[covariates].fillna(0).values
y = df_students["dropped_out"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for train_idx, val_idx in cv.split(X_scaled, y):
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_scaled[train_idx], y[train_idx])
    proba = clf.predict_proba(X_scaled[val_idx])[:, 1]
    aucs.append(roc_auc_score(y[val_idx], proba))

print(f"\nLogistic Regression 5-fold AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# Survival plot for top-risk vs low-risk students
model = cox_result["model"]
df_test = df_students.head(20).copy()
df_test[covariates] = StandardScaler().fit_transform(df_test[covariates])
sf = model.predict_survival_function(df_test[covariates])

fig, ax = plt.subplots(figsize=(8, 5))
for col in sf.columns[:5]:
    sf[col].plot(ax=ax, alpha=0.6)
ax.set_title("Predicted Survival Functions (first 5 students)")
ax.set_xlabel("Week")
ax.set_ylabel("P(still enrolled)")
plt.tight_layout()
plt.savefig("/tmp/dropout_survival.png", dpi=150)
plt.show()
```

### Example B — Knowledge Tracing for a Mathematics Curriculum

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulate a mathematics quiz log
np.random.seed(0)
skills = ["addition", "subtraction", "multiplication", "division", "fractions"]
n_students = 50
records = []
for student in range(n_students):
    for skill in skills:
        p_know = 0.1
        for attempt in range(1, 11):
            correct = int(np.random.rand() < (p_know * 0.9 + (1 - p_know) * 0.2))
            records.append({
                "user_id": f"s{student:03d}",
                "skill": skill,
                "attempt": attempt,
                "correct": correct,
            })
            # Simulate learning
            p_know = p_know + (1 - p_know) * 0.15

df_quiz = pd.DataFrame(records)

# Run BKT with default parameters
bkt_results = run_bkt(df_quiz, skill_col="skill")
print(bkt_results.head(20).to_string(index=False))

# Plot knowledge state for one student and one skill
fig, axes = plt.subplots(1, len(skills), figsize=(18, 4), sharey=True)
for ax, skill in zip(axes, skills):
    plot_knowledge_state(bkt_results, user_id="s000", skill=skill, ax=ax)
plt.suptitle("BKT Knowledge State — Student s000", fontsize=13)
plt.tight_layout()
plt.savefig("/tmp/bkt_knowledge_state.png", dpi=150)
plt.show()

# Learning curve analysis for 'fractions'
lc = compute_learning_curves(df_quiz, skill="fractions")
print(f"\nFractions learning curve — R²: {lc['r2']:.3f}")
print(f"Parameters: {lc['params']}")

# IRT analysis
response_matrix = (
    df_quiz[df_quiz["skill"] == "fractions"]
    .pivot_table(index="user_id", columns="attempt", values="correct")
    .values.astype(float)
)
irt = fit_irt_1pl(response_matrix)
print(f"\nIRT item difficulties: {np.round(irt['difficulty'], 3)}")
print(f"IRT ability range: [{irt['ability'].min():.2f}, {irt['ability'].max():.2f}]")

# Sequence mining — find common patterns in activity sequences
activity_seqs = []
for sid, grp in df_quiz.groupby("user_id"):
    seq = grp.sort_values(["skill", "attempt"])["skill"].tolist()
    activity_seqs.append(seq)

patterns = mine_activity_sequences(activity_seqs, min_support=0.8, max_pattern_len=3)
print("\nFrequent learning sequences (support >= 80%):")
for pattern, count in patterns[:10]:
    print(f"  {' → '.join(pattern)}: {count} students")
```

---

## 8. Tips and Gotchas

- **BKT identifiability**: The four BKT parameters are not jointly identifiable from
  response data alone. Fix `p_slip` and `p_guess` from domain knowledge or use EM
  fitting with pyBKT for proper parameter estimation.
- **IRT convergence**: The 1PL MLE can diverge for students who answer everything
  correctly or incorrectly. Add a small regularisation penalty or use Bayesian priors.
- **Cox PH assumptions**: Test the proportional hazards assumption with
  `lifelines.statistics.proportional_hazard_test` before interpreting coefficients.
- **Sequence mining scalability**: The pure-Python PrefixSpan above is fine for
  hundreds of students but slow for tens of thousands. Use the `prefixspan` PyPI
  package for production.
- **Class imbalance in dropout**: Typical dropout rates are 10-30%. Use
  `class_weight='balanced'` in scikit-learn and report AUC-ROC, not accuracy.

---

## 9. References

- Corbett & Anderson (1994). Knowledge tracing. *User Modeling and User-Adapted Interaction*, 4.
- Baker & Yacef (2009). The State of EDM. *JEDM*, 1(1).
- Pei, J. et al. (2004). Mining Sequential Patterns by Pattern-Growth. *IEEE TKDE*.
- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and Attainment Tests*.
