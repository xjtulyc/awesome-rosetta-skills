---
name: learning-experiment
description: Educational psychology experiment design and analysis covering IRT, growth modeling, A/B testing, and learning curve estimation for research.
tags:
  - education-research
  - item-response-theory
  - learning-analytics
  - experimental-design
  - growth-modeling
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
    - numpy>=1.24
    - pandas>=2.0
    - scipy>=1.11
    - statsmodels>=0.14
    - scikit-learn>=1.3
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: stable
---

# Learning Experiment Design and Analysis

## When to Use This Skill

Use this skill when you need to:
- Design randomized controlled trials for educational interventions
- Analyze item response data with IRT models (1PL/2PL/3PL)
- Estimate student growth models and learning trajectories
- Conduct A/B testing for educational technology features
- Fit learning curves (power law, exponential decay of errors)
- Measure knowledge retention (forgetting curves, spacing effects)
- Compute test reliability (Cronbach's alpha, inter-rater reliability)

**Trigger keywords**: item response theory, Rasch model, 2PL, 3PL, student growth, learning curve, forgetting curve, spaced repetition, educational A/B test, mastery learning, knowledge tracing, Bayesian knowledge tracing, DIF analysis, NAEP, PISA, CAT, adaptive testing, effect size, pre-post design.

## Background & Key Concepts

### Item Response Theory (IRT)

The 3-parameter logistic model (3PL) gives the probability that student $\theta$ answers item $i$ correctly:

$$P(X_i=1|\theta) = c_i + (1-c_i) \cdot \frac{e^{a_i(\theta - b_i)}}{1 + e^{a_i(\theta - b_i)}}$$

- $a_i$: discrimination parameter
- $b_i$: difficulty parameter (item location)
- $c_i$: guessing parameter (lower asymptote)

The 2PL sets $c_i=0$; the 1PL (Rasch) additionally fixes $a_i=1$.

### Item Information Function

$$I_i(\theta) = a_i^2 \cdot \frac{(P_i(\theta) - c_i)^2}{(1-c_i)^2} \cdot \frac{(1-P_i(\theta))}{P_i(\theta)}$$

Total test information: $I(\theta) = \sum_i I_i(\theta)$.  Standard error: $SE(\theta) = 1/\sqrt{I(\theta)}$.

### Power Law of Learning (Newell & Rosenbloom 1981)

$$T_n = T_1 \cdot n^{-\alpha}$$

where $T_n$ is the time (or error rate) on trial $n$ and $\alpha$ is the learning rate. Log-linearized: $\ln T_n = \ln T_1 - \alpha \ln n$.

### Bayesian Knowledge Tracing (BKT)

Hidden Markov model with states $K \in \{0,1\}$ (unlearned/learned):

$$P(K_{n+1}=1 | K_n=0) = p_T \quad \text{(transit)}$$
$$P(X=1 | K=1) = 1-p_S,\quad P(X=1|K=0) = p_G$$

### Effect Size Measures

- Cohen's $d$: $d = (\bar{x}_1 - \bar{x}_2)/s_p$ (pooled SD)
- Glass's $\Delta$: $\Delta = (\bar{x}_1 - \bar{x}_2)/s_{\text{control}}$
- Hedges' $g$: small-sample corrected version of $d$

## Environment Setup

```bash
pip install numpy>=1.24 pandas>=2.0 scipy>=1.11 statsmodels>=0.14 \
            scikit-learn>=1.3 matplotlib>=3.7
```

```python
import numpy as np
import pandas as pd
import scipy.optimize as opt
import statsmodels.api as sm
import matplotlib.pyplot as plt
print("Learning experiment analysis environment ready")
```

## Core Workflow

### Step 1: IRT Model Estimation (2PL)

```python
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate a 2PL item response dataset
# 500 students, 20 items
# -----------------------------------------------------------------
np.random.seed(42)
n_students = 500
n_items = 20

# True parameters
theta_true = np.random.normal(0, 1, n_students)           # student abilities
a_true = np.random.uniform(0.5, 2.5, n_items)             # discriminations
b_true = np.random.linspace(-2.5, 2.5, n_items)           # difficulties

# 2PL response probability
def p2pl(theta, a, b):
    """2PL probability of correct response."""
    return 1.0 / (1.0 + np.exp(-a * (theta[:, None] - b[None, :])))

P_matrix = p2pl(theta_true, a_true, b_true)
responses = (np.random.uniform(size=P_matrix.shape) < P_matrix).astype(int)

print(f"Response matrix shape: {responses.shape}")
print(f"Mean item difficulty (p-values): {responses.mean(axis=0).round(2)}")
print(f"Overall correct rate: {responses.mean():.3f}")

# -----------------------------------------------------------------
# Marginal Maximum Likelihood (simplified via EM-like approach)
# For a production system use mirt, TAM, or irtpy packages
# -----------------------------------------------------------------
def neg_log_likelihood_2pl(params, responses):
    """Compute -2LL for 2PL model with fixed quadrature points for theta.

    Args:
        params: vector of length 2*n_items (a1..an, b1..bn)
        responses: (n_students, n_items) binary matrix
    Returns:
        negative log-likelihood
    """
    n_i = responses.shape[1]
    a = np.abs(params[:n_i]) + 0.2   # ensure a > 0
    b = params[n_i:]

    # Gaussian quadrature over theta
    n_quad = 15
    quad_pts, quad_wts = np.polynomial.hermite.hermgauss(n_quad)
    theta_q = quad_pts * np.sqrt(2)   # transform to N(0,1)
    wts = quad_wts / np.sqrt(np.pi)

    total_ll = 0.0
    for s in range(len(responses)):
        # Likelihood for student s at each quadrature point
        # P_iq = 2PL probability
        p_iq = 1.0 / (1.0 + np.exp(-a[None, :] * (theta_q[:, None] - b[None, :])))
        # Response likelihood at each quadrature point
        r = responses[s]
        log_lik_q = (r * np.log(np.clip(p_iq, 1e-9, 1 - 1e-9)) +
                     (1 - r) * np.log(np.clip(1 - p_iq, 1e-9, 1 - 1e-9))).sum(axis=1)
        lik_q = np.exp(log_lik_q)
        marginal_lik = (lik_q * wts).sum()
        total_ll += np.log(max(marginal_lik, 1e-300))
    return -total_ll

# Initialize with reasonable starting values
a_init = np.ones(n_items)
b_init = np.zeros(n_items)
params_init = np.concatenate([a_init, b_init])

print("\nFitting 2PL model (simplified MML)...")
# Use a subset for speed demonstration
subset_students = responses[:100]
result = opt.minimize(neg_log_likelihood_2pl, params_init,
                      args=(subset_students,),
                      method="L-BFGS-B",
                      options={"maxiter": 50, "ftol": 1e-4})

a_est = np.abs(result.x[:n_items]) + 0.2
b_est = result.x[n_items:]
print(f"Optimization converged: {result.success}")
print(f"Correlation(a_true, a_est): {np.corrcoef(a_true, a_est)[0,1]:.3f}")
print(f"Correlation(b_true, b_est): {np.corrcoef(b_true, b_est)[0,1]:.3f}")

# -----------------------------------------------------------------
# Item Characteristic Curves (ICC) visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

theta_plot = np.linspace(-4, 4, 200)

# Plot 4 ICCs
colors = ["steelblue", "orange", "green", "red"]
for idx, item_idx in enumerate([0, 5, 10, 15]):
    p_true = 1 / (1 + np.exp(-a_true[item_idx] * (theta_plot - b_true[item_idx])))
    axes[0].plot(theta_plot, p_true, color=colors[idx],
                 label=f"Item {item_idx+1} (b={b_true[item_idx]:.1f})")
axes[0].set_xlabel("θ (Ability)"); axes[0].set_ylabel("P(correct)")
axes[0].set_title("Item Characteristic Curves")
axes[0].legend(fontsize=8)
axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)

# Item Information Functions
for idx, item_idx in enumerate([0, 5, 10, 15]):
    a, b = a_true[item_idx], b_true[item_idx]
    p = 1 / (1 + np.exp(-a * (theta_plot - b)))
    info = a**2 * p * (1 - p)
    axes[1].plot(theta_plot, info, color=colors[idx],
                 label=f"Item {item_idx+1} (a={a:.2f})")
axes[1].set_xlabel("θ"); axes[1].set_ylabel("Information")
axes[1].set_title("Item Information Functions")
axes[1].legend(fontsize=8)

# Test information function (total)
P_all = 1 / (1 + np.exp(-a_true[:, None] * (theta_plot[None, :] - b_true[:, None])))
test_info = (a_true[:, None]**2 * P_all * (1 - P_all)).sum(axis=0)
se_theta = 1 / np.sqrt(np.maximum(test_info, 0.01))
axes[2].plot(theta_plot, test_info, "steelblue", lw=2, label="Test Information")
ax2r = axes[2].twinx()
ax2r.plot(theta_plot, se_theta, "r--", lw=1.5, label="SE(θ)")
ax2r.set_ylabel("Standard Error", color="red")
axes[2].set_xlabel("θ"); axes[2].set_ylabel("Information")
axes[2].set_title("Test Information & SE")
axes[2].legend(loc="upper left")

plt.tight_layout()
plt.savefig("irt_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: irt_analysis.png")
```

### Step 2: Learning Curves and Forgetting Models

```python
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate learning experiment: 30 students, 20 practice trials
# -----------------------------------------------------------------
np.random.seed(42)
n_students = 30
n_trials = 20
trials = np.arange(1, n_trials + 1, dtype=float)

# True power law parameters (student-specific)
T1_true = np.random.uniform(20, 40, n_students)   # initial response time (seconds)
alpha_true = np.random.uniform(0.2, 0.6, n_students)  # learning rate

# Observed response times with noise
times = np.zeros((n_students, n_trials))
for s in range(n_students):
    times[s] = T1_true[s] * trials**(-alpha_true[s]) * np.exp(
        np.random.normal(0, 0.05, n_trials))

# -----------------------------------------------------------------
# Power law of learning: T_n = T1 * n^(-alpha)
# -----------------------------------------------------------------
def power_law(n, T1, alpha):
    """Power law of practice."""
    return T1 * n**(-alpha)

# Fit to mean learning curve
mean_times = times.mean(axis=0)
popt, pcov = opt.curve_fit(power_law, trials, mean_times,
                            p0=[30, 0.3], bounds=([1, 0.01], [200, 2.0]))
T1_fit, alpha_fit = popt
print(f"Power law fit: T1 = {T1_fit:.2f}s, α = {alpha_fit:.3f}")
print(f"95% CI for α: [{alpha_fit - 1.96*np.sqrt(pcov[1,1]):.3f}, "
      f"{alpha_fit + 1.96*np.sqrt(pcov[1,1]):.3f}]")

# Exponential fit for comparison: T_n = T_inf + (T1 - T_inf)*exp(-k*(n-1))
def exponential_learning(n, T_inf, T1, k):
    """Exponential learning model."""
    return T_inf + (T1 - T_inf) * np.exp(-k * (n - 1))

popt_exp, _ = opt.curve_fit(exponential_learning, trials, mean_times,
                              p0=[5, 35, 0.1], bounds=([0, 1, 0], [50, 200, 5]))
print(f"Exponential fit: T_inf = {popt_exp[0]:.2f}s, T1 = {popt_exp[1]:.2f}s, "
      f"k = {popt_exp[2]:.4f}")

# AIC comparison
def aic(n_data, k_params, sse):
    """Compute AIC."""
    sigma2 = sse / n_data
    return n_data * np.log(sigma2) + 2 * k_params

sse_power = ((mean_times - power_law(trials, *popt))**2).sum()
sse_exp = ((mean_times - exponential_learning(trials, *popt_exp))**2).sum()
print(f"\nAIC (power law): {aic(n_trials, 2, sse_power):.2f}")
print(f"AIC (exponential): {aic(n_trials, 3, sse_exp):.2f}")

# -----------------------------------------------------------------
# Forgetting curve: Ebbinghaus retention function
# R(t) = exp(-t/S) where S is stability
# -----------------------------------------------------------------
def ebbinghaus_retention(t, S):
    """Ebbinghaus forgetting curve: R(t) = e^{-t/S}."""
    return np.exp(-t / S)

# Simulate retention test at 1, 2, 7, 14, 30 days post-learning
retention_days = np.array([1, 2, 7, 14, 30], dtype=float)
true_S = 12.0  # stability (days)
retention_obs = ebbinghaus_retention(retention_days, true_S) + np.random.normal(0, 0.03, 5)
retention_obs = np.clip(retention_obs, 0, 1)

S_fit, _ = opt.curve_fit(ebbinghaus_retention, retention_days, retention_obs,
                          p0=[10], bounds=([1], [100]))
print(f"\nEbbinghaus S (stability) = {S_fit[0]:.1f} days")
print(f"Predicted 50% retention at: {S_fit[0] * np.log(2):.1f} days")

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Individual learning curves + mean
for s in range(min(10, n_students)):
    axes[0].plot(trials, times[s], color="lightblue", alpha=0.5, lw=0.8)
axes[0].plot(trials, mean_times, "k-", lw=2, label="Mean")
axes[0].plot(trials, power_law(trials, *popt), "r--", lw=2,
             label=f"Power Law (α={alpha_fit:.2f})")
axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Response Time (s)")
axes[0].set_title("Learning Curves")
axes[0].legend()

# Log-log plot for power law verification
axes[1].plot(np.log(trials), np.log(mean_times), "ko", label="Data")
axes[1].plot(np.log(trials), np.log(power_law(trials, *popt)), "r-",
             label=f"Fit (slope={-alpha_fit:.2f})")
axes[1].set_xlabel("ln(Trial)"); axes[1].set_ylabel("ln(Time)")
axes[1].set_title("Power Law Verification (Log-Log)")
axes[1].legend()

# Forgetting curve
t_cont = np.linspace(0, 35, 300)
axes[2].scatter(retention_days, retention_obs, color="red", s=80, zorder=5,
                label="Observed")
axes[2].plot(t_cont, ebbinghaus_retention(t_cont, S_fit[0]), "steelblue", lw=2,
             label=f"Ebbinghaus (S={S_fit[0]:.1f}d)")
axes[2].set_xlabel("Days Since Learning"); axes[2].set_ylabel("Retention (fraction)")
axes[2].set_title("Forgetting Curve")
axes[2].set_ylim(0, 1.1)
axes[2].legend()

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: learning_curves.png")
```

### Step 3: A/B Testing Educational Interventions

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------------------------------------------
# Simulate pre-post RCT: 200 students randomized to treatment/control
# Treatment: intelligent tutoring system vs. conventional instruction
# Outcome: post-test score (0-100)
# -----------------------------------------------------------------
np.random.seed(42)
n = 200
n_t = n // 2  # treatment group size

# Pre-test scores (baseline, shared before randomization)
pre_test = np.random.normal(55, 15, n)
pre_test = np.clip(pre_test, 0, 100)

# Treatment assignment (completely randomized)
treat = np.concatenate([np.ones(n_t), np.zeros(n_t)])
np.random.shuffle(treat)

# Post-test: treatment effect = 8 points, with heterogeneous effects
te_individual = 8 + np.random.normal(0, 3, n)  # ATE = 8
post_test = (pre_test * 0.7
             + treat * te_individual
             + np.random.normal(0, 10, n))
post_test = np.clip(post_test, 0, 100)

df = pd.DataFrame({"student_id": range(n), "treat": treat,
                   "pre_test": pre_test, "post_test": post_test})
df["gain"] = df["post_test"] - df["pre_test"]

# -----------------------------------------------------------------
# Analysis 1: Simple t-test on gains
# -----------------------------------------------------------------
t_gains = df[df["treat"] == 1]["gain"]
c_gains = df[df["treat"] == 0]["gain"]

t_stat, p_val = stats.ttest_ind(t_gains, c_gains)
print("=== A/B Test: Treatment vs. Control ===")
print(f"Mean gain (treatment): {t_gains.mean():.2f}")
print(f"Mean gain (control):   {c_gains.mean():.2f}")
print(f"Difference: {t_gains.mean() - c_gains.mean():.2f}")
print(f"t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

# Effect sizes
pooled_sd = np.sqrt((t_gains.var() + c_gains.var()) / 2)
cohens_d = (t_gains.mean() - c_gains.mean()) / pooled_sd
# Hedges' g correction
j = 1 - 3 / (4 * (n - 2) - 1)
hedges_g = cohens_d * j
print(f"Cohen's d: {cohens_d:.3f}")
print(f"Hedges' g: {hedges_g:.3f}")

# -----------------------------------------------------------------
# Analysis 2: ANCOVA (controls for pre-test score)
# -----------------------------------------------------------------
X_ancova = sm.add_constant(df[["treat", "pre_test"]])
ancova = sm.OLS(df["post_test"], X_ancova).fit(cov_type="HC3")
print("\n=== ANCOVA (post ~ treat + pre) ===")
print(ancova.summary().tables[1])
print(f"\nANCOVA treatment effect: {ancova.params['treat']:.3f} "
      f"(SE={ancova.bse['treat']:.3f})")

# Power analysis (post-hoc)
from scipy.stats import norm
alpha_level = 0.05
observed_d = cohens_d
se_d = np.sqrt(2/n + observed_d**2 / (2*n))
power_obs = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha_level/2) - observed_d / se_d)
print(f"\nPost-hoc power: {power_obs:.3f}")

# -----------------------------------------------------------------
# Analysis 3: Subgroup analysis (high vs. low baseline)
# -----------------------------------------------------------------
df["baseline_group"] = pd.cut(df["pre_test"], bins=[0, 50, 100],
                               labels=["Low", "High"])
subgroup = df.groupby(["baseline_group", "treat"])["gain"].agg(["mean", "std", "count"])
print("\n=== Subgroup Analysis (by baseline) ===")
print(subgroup.round(2))

# -----------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Pre-post scatter by treatment
colors = {1: "steelblue", 0: "orange"}
labels = {1: "Treatment (n=100)", 0: "Control (n=100)"}
for t_val in [0, 1]:
    mask = df["treat"] == t_val
    axes[0].scatter(df.loc[mask, "pre_test"], df.loc[mask, "post_test"],
                    color=colors[t_val], alpha=0.5, s=20, label=labels[t_val])
axes[0].plot([0, 100], [0, 100], "k--", lw=0.8, label="No change")
axes[0].set_xlabel("Pre-test Score"); axes[0].set_ylabel("Post-test Score")
axes[0].set_title("Pre-Post Scores by Treatment")
axes[0].legend(fontsize=8)

# Distribution of gains
axes[1].hist(c_gains, bins=20, alpha=0.6, color="orange", label="Control", density=True)
axes[1].hist(t_gains, bins=20, alpha=0.6, color="steelblue", label="Treatment", density=True)
axes[1].axvline(c_gains.mean(), color="orange", ls="--", lw=2)
axes[1].axvline(t_gains.mean(), color="steelblue", ls="--", lw=2)
axes[1].set_xlabel("Score Gain"); axes[1].set_ylabel("Density")
axes[1].set_title(f"Distribution of Gains\n(d={cohens_d:.2f})")
axes[1].legend()

# ANCOVA fitted vs. actual
axes[2].scatter(ancova.fittedvalues, df["post_test"], alpha=0.4, s=15,
                c=df["treat"].map({0: "orange", 1: "steelblue"}))
axes[2].plot([20, 100], [20, 100], "k--")
axes[2].set_xlabel("ANCOVA Fitted"); axes[2].set_ylabel("Observed")
axes[2].set_title(f"ANCOVA Fit (R²={ancova.rsquared:.3f})")

plt.tight_layout()
plt.savefig("educational_ab_test.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved: educational_ab_test.png")
```

## Advanced Usage

### Bayesian Knowledge Tracing (BKT)

```python
import numpy as np
from scipy.optimize import minimize

def bkt_forward(responses, p_L0, p_T, p_S, p_G):
    """Forward pass of BKT (single skill, single student).

    Args:
        responses: binary list of responses (1=correct, 0=incorrect)
        p_L0: prior probability of knowing skill at trial 1
        p_T: probability of transitioning to learned state
        p_S: slip probability (knows but answers wrong)
        p_G: guess probability (doesn't know but answers correctly)
    Returns:
        p_Ln: posterior P(learned) at each trial
        log_likelihood
    """
    p_Ln = np.zeros(len(responses) + 1)
    p_Ln[0] = p_L0
    log_lik = 0.0

    for n, r in enumerate(responses):
        p_L = p_Ln[n]
        # Probability of correct response at trial n+1
        p_correct = p_L * (1 - p_S) + (1 - p_L) * p_G

        # Accumulate log-likelihood
        log_lik += np.log(p_correct + 1e-10) if r == 1 else np.log(1 - p_correct + 1e-10)

        # Bayesian update: P(L|response)
        if r == 1:
            p_L_given_r = p_L * (1 - p_S) / (p_correct + 1e-10)
        else:
            p_L_given_r = p_L * p_S / (1 - p_correct + 1e-10)

        # Transition
        p_Ln[n + 1] = p_L_given_r + (1 - p_L_given_r) * p_T

    return p_Ln[1:], log_lik

# Simulate a student mastering a skill
np.random.seed(42)
true_params = {"p_L0": 0.2, "p_T": 0.15, "p_S": 0.08, "p_G": 0.20}
n_trials = 25
p_L_true = np.zeros(n_trials + 1)
p_L_true[0] = true_params["p_L0"]
for i in range(n_trials):
    if np.random.random() > p_L_true[i]:
        p_L_true[i+1] = p_L_true[i] + (1 - p_L_true[i]) * true_params["p_T"]
    else:
        p_L_true[i+1] = p_L_true[i]

learned = p_L_true[1:] > 0.5
sim_responses = []
for is_learned in learned:
    if is_learned:
        sim_responses.append(1 if np.random.random() > true_params["p_S"] else 0)
    else:
        sim_responses.append(1 if np.random.random() < true_params["p_G"] else 0)

# Fit BKT via MLE
def neg_ll(params):
    p_L0, p_T, p_S, p_G = params
    if not all(0.01 < p < 0.99 for p in params):
        return 1e10
    _, ll = bkt_forward(sim_responses, p_L0, p_T, p_S, p_G)
    return -ll

result = minimize(neg_ll, x0=[0.3, 0.2, 0.1, 0.25], method="Nelder-Mead")
p_L0_f, p_T_f, p_S_f, p_G_f = result.x
print(f"BKT MLE: L0={p_L0_f:.3f}, T={p_T_f:.3f}, S={p_S_f:.3f}, G={p_G_f:.3f}")
p_mastery, _ = bkt_forward(sim_responses, p_L0_f, p_T_f, p_S_f, p_G_f)

print(f"Final mastery probability: {p_mastery[-1]:.3f}")
print(f"Trial when mastery > 0.95: "
      f"{np.argmax(p_mastery > 0.95) + 1 if (p_mastery > 0.95).any() else 'Not reached'}")
```

### Differential Item Functioning (DIF) Detection

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def mantel_haenszel_dif(responses, group, ability_strata=5):
    """Mantel-Haenszel DIF statistic for each item.

    Args:
        responses: (n_students, n_items) binary matrix
        group: (n_students,) binary array — 0=reference, 1=focal
        ability_strata: number of ability level strata
    Returns:
        DataFrame with item DIF statistics
    """
    n_items = responses.shape[1]
    total_score = responses.sum(axis=1)
    strata = pd.qcut(total_score, ability_strata, labels=False, duplicates="drop")

    results = []
    for item in range(n_items):
        A = B = C = D = 0  # MH 2x2 table cells aggregated across strata
        for s in range(ability_strata):
            mask_s = strata == s
            if mask_s.sum() < 4:
                continue
            ref_mask = mask_s & (group == 0)
            foc_mask = mask_s & (group == 1)
            n_s = mask_s.sum()

            a = (responses[foc_mask, item]).sum()
            b = (responses[ref_mask, item]).sum()
            c = foc_mask.sum() - a
            d = ref_mask.sum() - b

            A += a * d / n_s
            B += b * c / n_s
            C += (a + d) * (a + c) * (b + c) * (b + d) / (n_s**2 * (n_s - 1) + 1e-10)

        alpha_mh = A / (B + 1e-10)  # odds ratio
        # Delta MH (ETS scale)
        delta_mh = -2.35 * np.log(alpha_mh + 1e-10)

        results.append({"item": item, "MH_alpha": alpha_mh, "delta_MH": delta_mh})

    return pd.DataFrame(results)

np.random.seed(42)
n_s = 400
n_i = 15
theta = np.random.normal(0, 1, n_s)
group = np.random.binomial(1, 0.4, n_s)
# Focal group has lower mean ability
theta[group == 1] -= 0.5

a_dif = np.ones(n_i)
b_dif = np.linspace(-2, 2, n_i)
b_dif[5] += 0.8  # Item 6 has DIF against focal group

P_dif = 1 / (1 + np.exp(-a_dif[None, :] * (theta[:, None] - b_dif[None, :])))
resp_dif = (np.random.uniform(size=P_dif.shape) < P_dif).astype(int)

dif_results = mantel_haenszel_dif(resp_dif, group)
print("=== DIF Analysis (Mantel-Haenszel) ===")
print(dif_results.round(3).to_string())
flagged = dif_results[dif_results["delta_MH"].abs() > 1.0]
print(f"\nFlagged items (|ΔMH| > 1.0): {flagged['item'].tolist()}")
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| IRT estimation diverges | Extreme item p-values (< 0.05 or > 0.95) | Remove extreme items; increase sample |
| Power law fit R² < 0.80 | Plateau or ceiling effects | Check for ceiling; use exponential model |
| BKT mastery oscillates | Slip > 0.5 or guess > 0.5 | Constrain params: `p_S < 0.4, p_G < 0.4` |
| Negative Cohen's d | Score direction | Verify higher score = better; check treatment coding |
| ANCOVA violation | Non-parallel slopes | Test treatment × pre interaction; use moderated regression |
| Negative DIF delta | Focal group advantaged | Expected; classify DIF direction (A favors reference, B favors focal) |

## External Resources

- Lord, F. M. (1980). *Applications of Item Response Theory*. Educational Testing Service.
- Embretson, S. E., & Reise, S. P. (2000). *Item Response Theory for Psychologists*. Erlbaum.
- Anderson, J. R., et al. (1995). Cognitive tutors: Lessons learned. *JLS*, 4(2).
- [mirt R package documentation](https://cran.r-project.org/web/packages/mirt/)
- [EdPsych Measurement MOOC materials](https://www.coursera.org/learn/edpsych-measureassess)

## Examples

### Example 1: Test Reliability Analysis

```python
import numpy as np
import pandas as pd

def cronbach_alpha(responses):
    """Compute Cronbach's alpha reliability coefficient.

    Args:
        responses: (n_students, n_items) matrix of item scores
    Returns:
        alpha coefficient
    """
    n_items = responses.shape[1]
    item_variances = responses.var(axis=0, ddof=1)
    total_variance = responses.sum(axis=1).var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return alpha

def item_statistics(responses):
    """Compute classical test theory item statistics."""
    total = responses.sum(axis=1)
    stats = []
    for j in range(responses.shape[1]):
        p_val = responses[:, j].mean()
        corrected_total = total - responses[:, j]
        r_it = np.corrcoef(responses[:, j], corrected_total)[0, 1]
        stats.append({"item": j+1, "p_value": p_val, "r_item_total": r_it})
    return pd.DataFrame(stats)

np.random.seed(42)
n_s, n_i = 300, 25
theta_r = np.random.normal(0, 1, n_s)
b_r = np.random.normal(0, 1, n_i)
P_r = 1 / (1 + np.exp(-(theta_r[:, None] - b_r[None, :])))
resp_r = (np.random.uniform(size=P_r.shape) < P_r).astype(int)

alpha = cronbach_alpha(resp_r)
print(f"Cronbach's α = {alpha:.3f}")
if alpha >= 0.90: print("Excellent reliability")
elif alpha >= 0.80: print("Good reliability")
elif alpha >= 0.70: print("Acceptable reliability")
else: print("Poor reliability — revise items")

item_stats = item_statistics(resp_r)
print("\nItem Statistics (first 5):")
print(item_stats.head().round(3))
poor_items = item_stats[item_stats["r_item_total"] < 0.20]
print(f"\nPoor discriminating items (r < 0.20): {poor_items['item'].tolist()}")
```

### Example 2: Spaced Repetition Optimization

```python
import numpy as np
import matplotlib.pyplot as plt

def sm2_schedule(initial_easiness=2.5, n_reviews=10, min_ease=1.3):
    """Simulate SM-2 spaced repetition algorithm schedule.

    Args:
        initial_easiness: starting E-factor (2.5 by default)
        n_reviews: number of review sessions
        min_ease: minimum E-factor
    Returns:
        DataFrame with intervals and retention estimates
    """
    import pandas as pd

    records = []
    interval = 1
    easiness = initial_easiness
    stability = 1.0  # Anki-like

    for rev in range(1, n_reviews + 1):
        # Assume quality q = 4 (good recall)
        q = 4
        # Update easiness factor
        easiness = max(min_ease, easiness + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

        # Next interval
        if rev == 1:
            next_interval = 1
        elif rev == 2:
            next_interval = 6
        else:
            next_interval = round(interval * easiness)

        # Forgetting curve: R = e^{-interval/stability}
        stability_at_review = stability * easiness
        retention = np.exp(-interval / (stability_at_review * 10))

        records.append({"review": rev, "interval_days": interval,
                        "easiness": easiness, "retention": retention})
        stability = stability_at_review
        interval = next_interval

    return pd.DataFrame(records)

schedule = sm2_schedule()
print("SM-2 Spaced Repetition Schedule:")
print(schedule.to_string(index=False))
print(f"\nTotal study sessions to review 10 times: {schedule['interval_days'].sum()} days")
```
