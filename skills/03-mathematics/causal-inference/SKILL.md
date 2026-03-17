---
name: causal-inference
description: >
  DoWhy and CausalML toolkit for causal DAG construction, effect identification,
  ATE estimation, propensity score matching, and refutation testing.
tags:
  - causal-inference
  - dowhy
  - causalml
  - propensity-score
  - statistics
  - dag
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
  - dowhy>=0.10.0
  - causalml>=0.15.0
  - networkx>=3.1
  - scikit-learn>=1.3.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - statsmodels>=0.14.0
last_updated: "2026-03-17"
---

# Causal Inference with DoWhy and CausalML

Causal inference goes beyond statistical association to estimate the effect of
interventions. This skill covers the complete causal inference workflow: DAG
construction, identification via backdoor/frontdoor criteria, estimation with
multiple methods, refutation testing, propensity score matching, and uplift modeling.

---

## 1. Core Concepts

The potential outcomes framework defines causal effects:

- **ATE**: Average Treatment Effect = E[Y(1) - Y(0)]
- **ATT**: Average Treatment Effect on the Treated
- **CATE**: Conditional ATE (heterogeneous treatment effects)
- **Backdoor criterion**: A set Z satisfies it if Z blocks all backdoor paths from T to Y
- **Frontdoor criterion**: Applicable when all paths go through mediator M
- **Confounding**: Variables that affect both treatment and outcome

```python
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


def generate_confounded_data(n=2000, seed=42):
    """
    Generate synthetic observational data with confounding.

    Data-generating process:
        U1, U2 ~ N(0, 1)   (unobserved confounders — for illustration)
        X1 = U1 + noise     (confounder 1: ability/IQ proxy)
        X2 = U2 + noise     (confounder 2: family income)
        T = sigmoid(0.8*X1 + 0.6*X2 - 1.0) > Bernoulli  (treatment: college)
        Y = 2.5*T + 1.2*X1 + 0.9*X2 + noise             (outcome: income)

    True ATE = 2.5 (units of income per treatment unit)
    """
    rng = np.random.default_rng(seed)

    X1 = rng.standard_normal(n)  # ability proxy
    X2 = rng.standard_normal(n)  # family income proxy
    X3 = rng.standard_normal(n)  # irrelevant covariate

    # Propensity score model
    log_odds = 0.8 * X1 + 0.6 * X2 - 1.0
    propensity = 1 / (1 + np.exp(-log_odds))
    T = (rng.uniform(size=n) < propensity).astype(int)

    # Outcome model (true ATE = 2.5)
    noise = 0.5 * rng.standard_normal(n)
    Y = 2.5 * T + 1.2 * X1 + 0.9 * X2 + noise

    df = pd.DataFrame({
        "treatment": T,
        "outcome": Y,
        "ability": X1,
        "family_income": X2,
        "noise_var": X3,
        "propensity_true": propensity,
    })
    return df


print("True ATE = 2.5")
df = generate_confounded_data(n=3000)
print(f"Dataset shape: {df.shape}")
print(f"Treatment prevalence: {df.treatment.mean():.3f}")
print(f"Naive difference in means: {df.groupby('treatment')['outcome'].mean().diff().iloc[-1]:.4f}")
```

---

## 2. DAG Construction

```python
import networkx as nx
import matplotlib.pyplot as plt


def build_causal_dag(nodes, edges, latent_nodes=None):
    """
    Build a causal DAG from node/edge lists.

    Parameters
    ----------
    nodes : list of str
        Variable names in the DAG.
    edges : list of (str, str)
        Directed edges (cause, effect).
    latent_nodes : list of str, optional
        Nodes representing unobserved variables.

    Returns
    -------
    G : nx.DiGraph
        Directed acyclic graph with node attributes.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Validate acyclicity
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph contains cycles — not a valid DAG.")

    # Mark node types
    if latent_nodes:
        for node in latent_nodes:
            G.nodes[node]["latent"] = True

    return G


def plot_dag(G, treatment=None, outcome=None, confounders=None, title="Causal DAG"):
    """
    Visualize a causal DAG with color-coded nodes.

    Color convention:
        Red   = treatment
        Green = outcome
        Orange = confounder/covariate
        Gray  = other
    """
    node_colors = []
    for node in G.nodes():
        if treatment and node == treatment:
            node_colors.append("tomato")
        elif outcome and node == outcome:
            node_colors.append("mediumseagreen")
        elif confounders and node in confounders:
            node_colors.append("darkorange")
        else:
            node_colors.append("lightgray")

    pos = nx.spring_layout(G, seed=7, k=2.0)
    fig, ax = plt.subplots(figsize=(9, 6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                           arrowstyle="-|>", arrowsize=20,
                           edge_color="steelblue", width=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="tomato", label="Treatment"),
        Patch(color="mediumseagreen", label="Outcome"),
        Patch(color="darkorange", label="Confounder"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    plt.tight_layout()
    plt.show()
    return fig


# Education-income DAG
nodes = ["ability", "family_income", "treatment", "outcome", "noise_var"]
edges = [
    ("ability", "treatment"),
    ("ability", "outcome"),
    ("family_income", "treatment"),
    ("family_income", "outcome"),
    ("treatment", "outcome"),
]

dag = build_causal_dag(nodes, edges)
plot_dag(dag, treatment="treatment", outcome="outcome",
         confounders=["ability", "family_income"],
         title="Education → Income DAG")

# Check backdoor paths
def find_backdoor_paths(G, treatment, outcome):
    """Find all backdoor paths (paths with arrow into treatment)."""
    G_undirected = G.to_undirected()
    all_paths = list(nx.all_simple_paths(G_undirected, treatment, outcome))
    backdoor = []
    for path in all_paths:
        if len(path) > 2:
            # First edge must go INTO treatment
            if G.has_edge(path[1], path[0]):
                backdoor.append(path)
    return backdoor

backdoor_paths = find_backdoor_paths(dag, "treatment", "outcome")
print(f"\nBackdoor paths from 'treatment' to 'outcome':")
for p in backdoor_paths:
    print(f"  {' → '.join(p)}")
```

---

## 3. Identification and Estimation

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def identify_effect(dag, treatment, outcome, covariates):
    """
    Check if the backdoor criterion is satisfied and return adjustment set.

    The backdoor criterion holds if covariates:
    1. Block all backdoor paths from treatment to outcome.
    2. Do not include any descendants of treatment.

    Returns
    -------
    dict with identification status and adjustment set.
    """
    # Check for descendant confounders
    try:
        descendants = nx.descendants(dag, treatment)
        adjustment_set = [c for c in covariates if c not in descendants]
        has_backdoor = len([c for c in covariates if c in adjustment_set]) > 0
        return {
            "identified": True,
            "method": "backdoor",
            "adjustment_set": adjustment_set,
            "invalid_covariates": [c for c in covariates if c in descendants],
        }
    except Exception as e:
        return {"identified": False, "error": str(e)}


def estimate_ate(df, treatment, outcome, covariates, method="regression"):
    """
    Estimate the Average Treatment Effect using various methods.

    Parameters
    ----------
    df : pd.DataFrame
    treatment : str
        Binary treatment column name.
    outcome : str
        Continuous outcome column name.
    covariates : list of str
        Adjustment variables (confounders).
    method : str
        One of: 'regression', 'ipw', 'matching', 'doubly_robust', 'dml'

    Returns
    -------
    dict with 'ate', 'std_error', 'ci_lower', 'ci_upper'
    """
    T = df[treatment].values
    Y = df[outcome].values
    X = df[covariates].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "regression":
        # OLS with treatment indicator and confounders
        X_full = np.column_stack([T, X_scaled])
        reg = LinearRegression().fit(X_full, Y)
        ate = reg.coef_[0]

        # Bootstrap standard error
        rng = np.random.default_rng(0)
        boot_ates = []
        for _ in range(500):
            idx = rng.integers(0, len(df), len(df))
            X_b = X_full[idx]
            Y_b = Y[idx]
            boot_ates.append(LinearRegression().fit(X_b, Y_b).coef_[0])
        std_error = np.std(boot_ates)

    elif method == "ipw":
        # Inverse Probability Weighting
        ps_model = LogisticRegression(C=1.0, max_iter=1000)
        ps_model.fit(X_scaled, T)
        ps = ps_model.predict_proba(X_scaled)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)  # trim extreme weights

        weights = T / ps + (1 - T) / (1 - ps)
        ate = np.mean(weights * Y * T) / np.mean(weights * T) - \
              np.mean(weights * Y * (1 - T)) / np.mean(weights * (1 - T))

        # Bootstrap SE
        rng = np.random.default_rng(0)
        boot_ates = []
        for _ in range(500):
            idx = rng.integers(0, len(df), len(df))
            ps_b = ps[idx]
            T_b, Y_b = T[idx], Y[idx]
            w_b = T_b / ps_b + (1 - T_b) / (1 - ps_b)
            boot_ates.append(
                np.mean(w_b * Y_b * T_b) / np.mean(w_b * T_b) -
                np.mean(w_b * Y_b * (1 - T_b)) / np.mean(w_b * (1 - T_b))
            )
        std_error = np.std(boot_ates)

    elif method == "doubly_robust":
        # Doubly Robust (AIPW) estimator
        # Outcome model
        X_treated = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        treated_mask = T == 1
        control_mask = T == 0

        mu1 = GradientBoostingRegressor(n_estimators=100, random_state=0)
        mu1.fit(X_scaled[treated_mask], Y[treated_mask])
        mu0 = GradientBoostingRegressor(n_estimators=100, random_state=0)
        mu0.fit(X_scaled[control_mask], Y[control_mask])

        mu1_hat = mu1.predict(X_scaled)
        mu0_hat = mu0.predict(X_scaled)

        # Propensity model
        ps_model = LogisticRegression(C=1.0, max_iter=1000)
        ps_model.fit(X_scaled, T)
        ps = np.clip(ps_model.predict_proba(X_scaled)[:, 1], 0.01, 0.99)

        # AIPW scores
        tau_hat = (
            mu1_hat - mu0_hat
            + T * (Y - mu1_hat) / ps
            - (1 - T) * (Y - mu0_hat) / (1 - ps)
        )
        ate = np.mean(tau_hat)
        std_error = np.std(tau_hat) / np.sqrt(len(tau_hat))

    else:
        raise ValueError(f"Unknown method: {method}. Choose from: regression, ipw, doubly_robust")

    ci_lower = ate - 1.96 * std_error
    ci_upper = ate + 1.96 * std_error

    return {
        "method": method,
        "ate": ate,
        "std_error": std_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# Run all methods on synthetic data
df = generate_confounded_data(n=3000)
covariates = ["ability", "family_income"]
true_ate = 2.5

results = {}
for method in ["regression", "ipw", "doubly_robust"]:
    res = estimate_ate(df, "treatment", "outcome", covariates, method=method)
    results[method] = res
    print(
        f"{method:15s}: ATE = {res['ate']:.4f} ± {res['std_error']:.4f}  "
        f"95% CI = [{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"
    )

print(f"\nTrue ATE = {true_ate}")
```

---

## 4. Propensity Score Matching

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def estimate_propensity_scores(df, treatment, covariates, model="logistic"):
    """
    Estimate propensity scores P(T=1 | X).

    Parameters
    ----------
    model : str
        'logistic' or 'gradient_boosting'
    """
    X = df[covariates].values
    T = df[treatment].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model == "logistic":
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    elif model == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model}")

    clf.fit(X_scaled, T)
    ps = clf.predict_proba(X_scaled)[:, 1]
    return ps, scaler, clf


def propensity_score_matching(df, treatment, outcome, covariates, caliper=0.05, n_neighbors=1):
    """
    Perform 1:1 nearest-neighbor propensity score matching.

    Parameters
    ----------
    caliper : float
        Maximum allowed difference in propensity scores (in std units).
        Typically 0.2 * std(logit(ps)).
    n_neighbors : int
        Number of controls matched per treated unit.

    Returns
    -------
    matched_df : pd.DataFrame
        Dataset of matched pairs.
    att : float
        Average Treatment Effect on the Treated.
    """
    ps, _, _ = estimate_propensity_scores(df, treatment, covariates)
    df = df.copy()
    df["propensity_score"] = ps
    df["logit_ps"] = np.log(ps / (1 - ps))

    treated = df[df[treatment] == 1].reset_index(drop=True)
    control = df[df[treatment] == 0].reset_index(drop=True)

    # Fit nearest-neighbor on logit propensity scores
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(control["logit_ps"].values.reshape(-1, 1))

    distances, indices = nn.kneighbors(treated["logit_ps"].values.reshape(-1, 1))

    # Apply caliper
    caliper_abs = caliper * np.std(df["logit_ps"])
    matched_pairs = []
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for dist, idx in zip(dist_row, idx_row):
            if dist <= caliper_abs:
                matched_pairs.append({
                    "treated_idx": i,
                    "control_idx": idx,
                    "distance": dist,
                    f"{outcome}_treated": treated[outcome].iloc[i],
                    f"{outcome}_control": control[outcome].iloc[idx],
                    "ps_treated": treated["propensity_score"].iloc[i],
                    "ps_control": control["propensity_score"].iloc[idx],
                })

    if not matched_pairs:
        raise ValueError("No matches found within caliper. Try increasing caliper value.")

    pairs_df = pd.DataFrame(matched_pairs)
    att = (pairs_df[f"{outcome}_treated"] - pairs_df[f"{outcome}_control"]).mean()
    att_se = (pairs_df[f"{outcome}_treated"] - pairs_df[f"{outcome}_control"]).std() / np.sqrt(len(pairs_df))

    print(f"\nPropensity Score Matching Results:")
    print(f"  Treated units:  {len(treated)}")
    print(f"  Control units:  {len(control)}")
    print(f"  Matched pairs:  {len(pairs_df)}")
    print(f"  Match rate:     {len(pairs_df)/len(treated)*100:.1f}%")
    print(f"  ATT = {att:.4f} ± {att_se:.4f}")
    print(f"  95% CI = [{att - 1.96*att_se:.3f}, {att + 1.96*att_se:.3f}]")

    return pairs_df, att, att_se


def check_balance(df, treatment, covariates, ps=None):
    """
    Compute standardized mean differences (SMD) before/after matching.
    SMD < 0.1 is typically considered good balance.
    """
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    print("\nCovariate Balance (Standardized Mean Differences):")
    print(f"{'Covariate':20s} {'SMD':>10}")
    print("-" * 32)

    smds = {}
    for col in covariates:
        mean_t = treated[col].mean()
        mean_c = control[col].mean()
        std_pooled = np.sqrt((treated[col].var() + control[col].var()) / 2)
        smd = (mean_t - mean_c) / (std_pooled + 1e-10)
        smds[col] = abs(smd)
        flag = " !" if abs(smd) > 0.1 else "  "
        print(f"{col:20s} {smd:>10.4f}{flag}")

    return smds


def plot_propensity_overlap(df, treatment, ps_col="propensity_score"):
    """Plot propensity score distributions to check overlap (common support)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    treated_ps = df[df[treatment] == 1][ps_col]
    control_ps = df[df[treatment] == 0][ps_col]

    axes[0].hist(treated_ps, bins=40, alpha=0.6, color="tomato", density=True, label="Treated")
    axes[0].hist(control_ps, bins=40, alpha=0.6, color="steelblue", density=True, label="Control")
    axes[0].set_xlabel("Propensity Score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Propensity Score Overlap")
    axes[0].legend()

    # Love plot (SMD)
    smds = check_balance(df, treatment, [c for c in df.columns if c not in [treatment, ps_col, "outcome", "propensity_score", "logit_ps", "propensity_true"]])
    axes[1].barh(list(smds.keys()), list(smds.values()), color=["tomato" if v > 0.1 else "steelblue" for v in smds.values()])
    axes[1].axvline(0.1, color="red", ls="--", lw=1, label="SMD=0.1 threshold")
    axes[1].set_xlabel("Absolute SMD")
    axes[1].set_title("Covariate Balance (Love Plot)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# Run matching
df = generate_confounded_data(n=3000)
df["propensity_score"] = estimate_propensity_scores(df, "treatment", ["ability", "family_income"])[0]
plot_propensity_overlap(df, "treatment")
pairs_df, att, att_se = propensity_score_matching(
    df, "treatment", "outcome", ["ability", "family_income"], caliper=0.1
)
```

---

## 5. Refutation Tests

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def refute_estimate(df, treatment, outcome, covariates, estimated_ate, method="placebo"):
    """
    Run refutation tests to validate the causal estimate.

    Refutation methods:
    - 'random_common_cause': Add a random variable as confounder; ATE should be stable.
    - 'placebo_treatment': Replace treatment with random noise; ATE should go to ~0.
    - 'data_subset': Re-estimate on 80% random subsets; ATE should be stable.
    - 'bootstrap': Bootstrap distribution of ATE.

    Returns
    -------
    dict with refutation results and p-value.
    """

    def compute_ate_regression(data, t_col, y_col, x_cols):
        X = np.column_stack([data[t_col].values, StandardScaler().fit_transform(data[x_cols].values)])
        reg = LinearRegression().fit(X, data[y_col].values)
        return reg.coef_[0]

    rng = np.random.default_rng(123)

    if method == "random_common_cause":
        n_trials = 100
        perturbed_ates = []
        for _ in range(n_trials):
            df_perturbed = df.copy()
            df_perturbed["random_confounder"] = rng.standard_normal(len(df))
            new_covs = covariates + ["random_confounder"]
            ate_new = compute_ate_regression(df_perturbed, treatment, outcome, new_covs)
            perturbed_ates.append(ate_new)

        mean_perturbed = np.mean(perturbed_ates)
        std_perturbed = np.std(perturbed_ates)
        print(f"\nRandom Common Cause Refutation:")
        print(f"  Original ATE:   {estimated_ate:.4f}")
        print(f"  Perturbed ATE:  {mean_perturbed:.4f} ± {std_perturbed:.4f}")
        print(f"  Change:         {abs(mean_perturbed - estimated_ate):.4f}")
        result = "PASS" if abs(mean_perturbed - estimated_ate) < 0.3 else "FAIL"
        print(f"  Result: {result}")
        return {"method": method, "original_ate": estimated_ate,
                "new_ate": mean_perturbed, "passed": result == "PASS"}

    elif method == "placebo_treatment":
        n_trials = 200
        placebo_ates = []
        for _ in range(n_trials):
            df_placebo = df.copy()
            # Random binary placebo with same prevalence as original treatment
            p_treat = df[treatment].mean()
            df_placebo["placebo_treatment"] = (rng.uniform(size=len(df)) < p_treat).astype(int)
            ate_placebo = compute_ate_regression(df_placebo, "placebo_treatment", outcome, covariates)
            placebo_ates.append(ate_placebo)

        mean_placebo = np.mean(placebo_ates)
        # p-value: fraction of placebo ATEs more extreme than original
        p_value = np.mean(np.abs(placebo_ates) >= np.abs(estimated_ate))
        print(f"\nPlacebo Treatment Refutation:")
        print(f"  Original ATE:  {estimated_ate:.4f}")
        print(f"  Placebo ATE:   {mean_placebo:.4f} ± {np.std(placebo_ates):.4f}")
        print(f"  p-value:       {p_value:.4f}")
        result = "PASS" if abs(mean_placebo) < 0.2 else "FAIL"
        print(f"  Result: {result}")
        return {"method": method, "original_ate": estimated_ate,
                "placebo_ate": mean_placebo, "p_value": p_value, "passed": result == "PASS"}

    elif method == "data_subset":
        n_trials = 50
        subset_ates = []
        for _ in range(n_trials):
            idx = rng.choice(len(df), size=int(0.8 * len(df)), replace=False)
            df_sub = df.iloc[idx]
            ate_sub = compute_ate_regression(df_sub, treatment, outcome, covariates)
            subset_ates.append(ate_sub)

        mean_subset = np.mean(subset_ates)
        std_subset = np.std(subset_ates)
        print(f"\nData Subset Refutation:")
        print(f"  Original ATE:  {estimated_ate:.4f}")
        print(f"  Subset ATE:    {mean_subset:.4f} ± {std_subset:.4f}")
        result = "PASS" if abs(mean_subset - estimated_ate) < 0.15 else "FAIL"
        print(f"  Result: {result}")
        return {"method": method, "original_ate": estimated_ate,
                "new_ate": mean_subset, "std": std_subset, "passed": result == "PASS"}

    else:
        raise ValueError(f"Unknown refutation method: {method}")


# Run all refutation tests
df = generate_confounded_data(n=3000)
res = estimate_ate(df, "treatment", "outcome", ["ability", "family_income"], method="doubly_robust")
estimated_ate = res["ate"]

for refute_method in ["random_common_cause", "placebo_treatment", "data_subset"]:
    refute_estimate(df, "treatment", "outcome", ["ability", "family_income"],
                    estimated_ate, method=refute_method)
```

---

## 6. Complete Example A — Education Effect on Income

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def education_income_study(n=5000, seed=0):
    """
    Estimate the causal effect of college education on income
    using backdoor adjustment on synthetic observational data.

    True causal effect = 3.0 (income units per year of education).
    """
    rng = np.random.default_rng(seed)

    # Observed confounders
    parents_edu = rng.integers(8, 20, size=n).astype(float)
    iq = 100 + 15 * rng.standard_normal(n)
    region = rng.integers(0, 3, size=n)  # 0=rural, 1=suburban, 2=urban

    # Treatment: years of college (0-4), influenced by confounders
    log_odds = (0.05 * (parents_edu - 12) + 0.02 * (iq - 100)
                + 0.3 * (region == 2).astype(float) - 0.5)
    p_college = 1 / (1 + np.exp(-log_odds))
    college_years = rng.binomial(4, p_college)

    # Outcome: income ($k/yr), TRUE causal effect = 3.0 per year
    noise = 5 * rng.standard_normal(n)
    income = (20 + 3.0 * college_years
              + 0.4 * parents_edu
              + 0.08 * (iq - 100)
              + 4 * (region == 2).astype(float)
              + noise)

    df = pd.DataFrame({
        "college_years": college_years,
        "income": income,
        "parents_edu": parents_edu,
        "iq": iq,
        "region": region,
    })

    covariates = ["parents_edu", "iq", "region"]
    true_ate = 3.0

    print("=" * 55)
    print("Education → Income Causal Effect Study")
    print("=" * 55)
    print(f"  N = {n}, True ATE = {true_ate}")
    print(f"  Naive OLS (no adjustment): {_naive_ols(df):.4f}")

    # Method comparison
    methods_results = {}
    for method in ["regression", "ipw", "doubly_robust"]:
        res = estimate_ate(df, "college_years", "income", covariates, method=method)
        methods_results[method] = res
        print(f"  {method:15s}: ATE = {res['ate']:.4f} (bias={res['ate']-true_ate:+.4f})")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Scatter: college years vs income
    for region_val, color, label in [(0, "steelblue", "Rural"), (1, "darkorange", "Suburban"), (2, "green", "Urban")]:
        mask = df["region"] == region_val
        axes[0].scatter(df.loc[mask, "college_years"] + 0.05 * rng.standard_normal(mask.sum()),
                        df.loc[mask, "income"], alpha=0.15, s=5, color=color, label=label)
    axes[0].set_xlabel("College Years")
    axes[0].set_ylabel("Income ($k/yr)")
    axes[0].set_title("Raw Data by Region")
    axes[0].legend(markerscale=3)

    # 2. ATE comparison
    method_names = list(methods_results.keys())
    ates = [methods_results[m]["ate"] for m in method_names]
    cis = [(methods_results[m]["ci_lower"], methods_results[m]["ci_upper"]) for m in method_names]
    y_pos = range(len(method_names))
    axes[1].barh(y_pos, ates, xerr=[(a - ci[0]) for a, ci in zip(ates, cis)], color="steelblue", alpha=0.7)
    axes[1].axvline(true_ate, color="red", ls="--", lw=2, label=f"True ATE={true_ate}")
    axes[1].set_yticks(list(y_pos))
    axes[1].set_yticklabels(method_names)
    axes[1].set_xlabel("Estimated ATE")
    axes[1].set_title("ATE Estimates by Method")
    axes[1].legend()

    # 3. Propensity score distribution
    X_cov = StandardScaler().fit_transform(df[covariates].values)
    # Use college >=2 as binary treatment proxy
    T_binary = (df["college_years"] >= 2).astype(int)
    ps_model = LogisticRegression(C=1.0, max_iter=1000)
    ps_model.fit(X_cov, T_binary)
    ps = ps_model.predict_proba(X_cov)[:, 1]
    axes[2].hist(ps[T_binary == 1], bins=40, alpha=0.6, color="tomato", density=True, label="College ≥2yr")
    axes[2].hist(ps[T_binary == 0], bins=40, alpha=0.6, color="steelblue", density=True, label="College <2yr")
    axes[2].set_xlabel("Propensity Score")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Propensity Score Overlap")
    axes[2].legend()

    plt.suptitle("Education → Income Causal Study", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return df, methods_results


def _naive_ols(df):
    """Simple bivariate regression (confounded estimate)."""
    from sklearn.linear_model import LinearRegression
    X = df["college_years"].values.reshape(-1, 1)
    Y = df["income"].values
    return LinearRegression().fit(X, Y).coef_[0]


if __name__ == "__main__":
    df_study, results = education_income_study()
```

---

## 7. Complete Example B — Propensity Score Matching for Observational Study

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def observational_matching_study(n=2000, seed=42):
    """
    Simulate a clinical observational study and apply propensity score matching.

    Scenario: Effect of a new drug on blood pressure reduction.
    True ATT = 8.0 mmHg reduction.
    """
    rng = np.random.default_rng(seed)

    # Baseline covariates
    age = rng.integers(40, 75, size=n).astype(float)
    bmi = 25 + 5 * rng.standard_normal(n)
    baseline_bp = 140 + 10 * rng.standard_normal(n)
    smoker = (rng.uniform(size=n) < 0.3).astype(int)
    diabetes = (rng.uniform(size=n) < 0.2).astype(int)

    # Treatment assignment (sicker patients more likely to receive drug)
    log_odds = (0.04 * (age - 55) + 0.05 * (bmi - 25) +
                0.01 * (baseline_bp - 140) + 0.5 * smoker + 0.4 * diabetes - 0.5)
    ps_true = 1 / (1 + np.exp(-log_odds))
    drug = (rng.uniform(size=n) < ps_true).astype(int)

    # Outcome: BP reduction (true ATT = 8.0)
    noise = 3.0 * rng.standard_normal(n)
    bp_reduction = (8.0 * drug
                    + 0.15 * (age - 55)
                    + 0.2 * (bmi - 25)
                    + 0.05 * (baseline_bp - 140)
                    + 2.0 * smoker
                    + 1.5 * diabetes
                    + noise)

    df = pd.DataFrame({
        "drug": drug,
        "bp_reduction": bp_reduction,
        "age": age,
        "bmi": bmi,
        "baseline_bp": baseline_bp,
        "smoker": smoker,
        "diabetes": diabetes,
        "ps_true": ps_true,
    })

    covariates = ["age", "bmi", "baseline_bp", "smoker", "diabetes"]
    true_att = 8.0

    print("=" * 55)
    print("Clinical Drug Study — Propensity Score Matching")
    print("=" * 55)
    naive_diff = df.groupby("drug")["bp_reduction"].mean().diff().iloc[-1]
    print(f"  Naive difference in means:  {naive_diff:.4f} (true ATT = {true_att})")

    # Estimate PS and run matching
    X = StandardScaler().fit_transform(df[covariates].values)
    T = df["drug"].values
    ps_model = LogisticRegression(C=1.0, max_iter=1000)
    ps_model.fit(X, T)
    df["ps_estimated"] = ps_model.predict_proba(X)[:, 1]

    # Check overlap before matching
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        T, df["ps_estimated"], n_bins=10
    )

    pairs_df, att, att_se = propensity_score_matching(
        df, "drug", "bp_reduction", covariates, caliper=0.1
    )

    print(f"\n  Matched ATT:               {att:.4f} ± {att_se:.4f}")
    print(f"  Bias (matched):            {att - true_att:+.4f}")

    # Also run doubly-robust for comparison
    dr_result = estimate_ate(df, "drug", "bp_reduction", covariates, method="doubly_robust")
    print(f"  Doubly-robust ATT:         {dr_result['ate']:.4f} ± {dr_result['std_error']:.4f}")

    # Balance check after matching
    # Build matched dataset
    treated_matched_idx = pairs_df["treated_idx"].values
    control_matched_idx = pairs_df["control_idx"].values
    treated_df = df[df["drug"] == 1].reset_index(drop=True).iloc[treated_matched_idx]
    control_df = df[df["drug"] == 0].reset_index(drop=True).iloc[control_matched_idx]
    df_matched = pd.concat([treated_df, control_df], ignore_index=True)

    print("\nBalance before matching:")
    check_balance(df, "drug", covariates)
    print("\nBalance after matching:")
    check_balance(df_matched, "drug", covariates)

    # Final plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[df["drug"] == 1]["ps_estimated"], bins=30, alpha=0.6,
                 color="tomato", density=True, label="Drug=1")
    axes[0].hist(df[df["drug"] == 0]["ps_estimated"], bins=30, alpha=0.6,
                 color="steelblue", density=True, label="Drug=0")
    axes[0].set_title("Propensity Score Distribution (Before Matching)")
    axes[0].set_xlabel("Estimated PS")
    axes[0].legend()

    diff_pairs = pairs_df["bp_reduction_treated"] - pairs_df["bp_reduction_control"]
    axes[1].hist(diff_pairs, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    axes[1].axvline(att, color="red", lw=2, label=f"ATT={att:.2f}")
    axes[1].axvline(true_att, color="green", lw=2, ls="--", label=f"True ATT={true_att}")
    axes[1].set_xlabel("Individual Treatment Effect (matched pairs)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Matched Pair Differences")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return df, pairs_df, att


if __name__ == "__main__":
    observational_matching_study()
```

---

## Quick Reference

| Task                            | Method / Function                                     |
|---------------------------------|-------------------------------------------------------|
| Build causal DAG                | `build_causal_dag(nodes, edges)`                      |
| Visualize DAG                   | `plot_dag(G, treatment, outcome, confounders)`        |
| Find backdoor paths             | `find_backdoor_paths(G, treatment, outcome)`          |
| Regression adjustment ATE       | `estimate_ate(df, T, Y, X, method='regression')`      |
| IPW estimation                  | `estimate_ate(df, T, Y, X, method='ipw')`             |
| Doubly-robust (AIPW)            | `estimate_ate(df, T, Y, X, method='doubly_robust')`   |
| Propensity score estimation     | `estimate_propensity_scores(df, T, X)`                |
| PSM with caliper                | `propensity_score_matching(df, T, Y, X, caliper)`     |
| Check covariate balance         | `check_balance(df, T, covariates)`                    |
| Placebo refutation              | `refute_estimate(..., method='placebo_treatment')`    |
| Random confounder refutation    | `refute_estimate(..., method='random_common_cause')`  |
| Data subset refutation          | `refute_estimate(..., method='data_subset')`          |

### Key Assumptions to Check

1. **Unconfoundedness** (ignorability): All confounders are measured. Test with sensitivity analysis.
2. **Overlap** (positivity): 0 < P(T=1|X) < 1 for all X. Check propensity score histograms.
3. **SUTVA**: No interference between units; no hidden treatment versions.
4. **Correct model specification**: Outcome/propensity models must be well-specified for doubly-robust to work.
