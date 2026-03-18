---
name: health-economics-eval
description: >
  Use this Skill for health economic evaluation: cost-effectiveness analysis (ICER),
  Markov cohort models, QALY calculation, probabilistic sensitivity analysis, and
  cost-effectiveness planes.
tags:
  - public-health
  - health-economics
  - ICER
  - QALY
  - Markov-model
  - cost-effectiveness
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

# Health Economic Evaluation: Cost-Effectiveness Analysis

> **TL;DR** — Implement full health economic evaluations: Markov cohort state
> transition models with discounting, QALY calculation, PSA Monte Carlo simulation,
> cost-effectiveness planes, and CEAC curves at multiple WTP thresholds.

---

## When to Use

Use this Skill when you need to:

- Estimate the ICER (Incremental Cost-Effectiveness Ratio) for a new intervention
- Build a Markov cohort model comparing healthcare strategies over a time horizon
- Calculate QALYs (Quality-Adjusted Life Years) from utility weights and dwell times
- Conduct Probabilistic Sensitivity Analysis (PSA) with distributions on all parameters
- Plot cost-effectiveness planes and Cost-Effectiveness Acceptability Curves (CEAC)
- Evaluate interventions against WHO cost-per-DALY thresholds (1–3× GDP per capita)
- Support decision making using the net monetary benefit (NMB) framework

| Concept | Formula |
|---|---|
| ICER | ΔCost / ΔQALY |
| QALY | Utility × Time (years) |
| NMB | λ × ΔQALY − ΔCost |
| Discounted QALY | QALY_t × 1/(1+r)^t |
| Half-cycle correction | Add 0.5 cycle at start and end |

---

## Background

### Cost-Effectiveness Analysis Framework

Health economic evaluation compares interventions on two dimensions simultaneously:
incremental costs and incremental health outcomes. The **ICER** is the ratio:

```
ICER = (Cost_new − Cost_comparator) / (QALY_new − QALY_comparator)
```

An intervention is considered cost-effective if its ICER falls below the
willingness-to-pay (WTP) threshold λ. Common thresholds:

- UK (NICE): £20,000–£30,000 per QALY
- WHO: 1–3× GDP per capita per DALY averted
- USA: $50,000–$150,000 per QALY (no official threshold)

### Markov Cohort Model

A Markov model represents disease progression as transitions between mutually
exclusive health states over discrete time cycles (typically 1 year):

```
States: Healthy → Sick → Dead
Transition matrix P (3×3):
  P[i,j] = probability of moving from state i to state j in one cycle
  Row sums must equal 1
```

Each state has associated costs (annual cost of being in that state) and
utility weights (health-related quality of life, 0=death, 1=perfect health).

### Discounting

Future costs and QALYs are discounted to present value:

```
PV = Σₜ Value_t / (1 + r)^t
```

Standard discount rate: 3% per annum for both costs and outcomes (NICE, WHO).
Half-cycle correction shifts outcomes by half a cycle to account for events
occurring throughout the cycle rather than at the start.

### Probabilistic Sensitivity Analysis

PSA replaces point estimates with probability distributions to characterize
parameter uncertainty:

| Parameter Type | Distribution |
|---|---|
| Transition probabilities | Dirichlet or Beta |
| Utility weights | Beta (bounded 0–1) |
| Costs | Gamma (positive, right-skewed) |
| Log odds ratios | Normal |

Each PSA iteration draws from all distributions simultaneously and re-runs
the model, producing a cloud of (ΔCost, ΔQALY) points on the CE plane.

---

## Environment Setup

```bash
conda create -n health-econ python=3.11 -y
conda activate health-econ
pip install "numpy>=1.23" "scipy>=1.9" "pandas>=1.5" "matplotlib>=3.6"

python -c "import numpy, scipy, pandas, matplotlib; print('Setup OK')"
```

---

## Core Workflow

### Step 1 — Markov Cohort Model with State Transitions and Outcomes

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def run_markov_model(
    states: List[str],
    transition_matrix: np.ndarray,
    state_costs: np.ndarray,
    state_utilities: np.ndarray,
    initial_cohort: np.ndarray,
    n_cycles: int = 40,
    cycle_length_years: float = 1.0,
    discount_rate_cost: float = 0.03,
    discount_rate_qaly: float = 0.03,
    half_cycle_correction: bool = True,
) -> dict:
    """
    Run a Markov cohort model and compute total discounted costs and QALYs.

    Args:
        states:               List of health state names.
        transition_matrix:    (n_states × n_states) transition probability matrix.
                              Rows must sum to 1; absorbing states have P[i,i]=1.
        state_costs:          Annual cost per patient in each state (array of length n_states).
        state_utilities:      Utility weight for each state (array of length n_states).
        initial_cohort:       Proportion of cohort in each state at time 0 (sums to 1).
        n_cycles:             Number of model cycles (e.g., 40 years).
        cycle_length_years:   Duration of one cycle in years.
        discount_rate_cost:   Annual discount rate for costs.
        discount_rate_qaly:   Annual discount rate for QALYs.
        half_cycle_correction: Apply half-cycle correction to first and last cycles.

    Returns:
        Dict with: total_cost, total_qaly, state_trace (DataFrame),
        discounted_costs_per_cycle, discounted_qalys_per_cycle.
    """
    n_states = len(states)
    assert transition_matrix.shape == (n_states, n_states), "Transition matrix dimensions mismatch"
    assert np.allclose(transition_matrix.sum(axis=1), 1.0, atol=1e-6), "Rows must sum to 1"

    cohort = initial_cohort.copy()
    trace = [cohort.copy()]
    costs_per_cycle = []
    qalys_per_cycle = []

    for t in range(1, n_cycles + 1):
        cohort = cohort @ transition_matrix
        trace.append(cohort.copy())

        # Discount factor
        df_cost = 1.0 / (1 + discount_rate_cost) ** (t * cycle_length_years)
        df_qaly = 1.0 / (1 + discount_rate_qaly) ** (t * cycle_length_years)

        cycle_cost = np.dot(cohort, state_costs) * cycle_length_years * df_cost
        cycle_qaly = np.dot(cohort, state_utilities) * cycle_length_years * df_qaly

        costs_per_cycle.append(cycle_cost)
        qalys_per_cycle.append(cycle_qaly)

    costs_arr = np.array(costs_per_cycle)
    qalys_arr = np.array(qalys_per_cycle)

    if half_cycle_correction:
        # Half-cycle: first and last cycle get weight 0.5
        costs_arr[0] *= 0.5
        costs_arr[-1] *= 0.5
        qalys_arr[0] *= 0.5
        qalys_arr[-1] *= 0.5

    trace_df = pd.DataFrame(trace, columns=states)
    trace_df.index.name = 'cycle'

    return {
        'total_cost': float(costs_arr.sum()),
        'total_qaly': float(qalys_arr.sum()),
        'state_trace': trace_df,
        'discounted_costs_per_cycle': costs_arr,
        'discounted_qalys_per_cycle': qalys_arr,
    }


def compute_icer(
    strategy_a: dict,
    strategy_b: dict,
    strategy_a_name: str = 'Comparator',
    strategy_b_name: str = 'New intervention',
) -> dict:
    """
    Compute the ICER comparing a new intervention (B) to a comparator (A).

    Args:
        strategy_a: Output of run_markov_model() for comparator.
        strategy_b: Output of run_markov_model() for new intervention.

    Returns:
        Dict with: delta_cost, delta_qaly, icer.
    """
    delta_cost = strategy_b['total_cost'] - strategy_a['total_cost']
    delta_qaly = strategy_b['total_qaly'] - strategy_a['total_qaly']
    icer = delta_cost / delta_qaly if abs(delta_qaly) > 1e-9 else float('inf')

    result = {
        'strategy_a': strategy_a_name,
        'strategy_b': strategy_b_name,
        'cost_a': round(strategy_a['total_cost'], 2),
        'cost_b': round(strategy_b['total_cost'], 2),
        'qaly_a': round(strategy_a['total_qaly'], 4),
        'qaly_b': round(strategy_b['total_qaly'], 4),
        'delta_cost': round(delta_cost, 2),
        'delta_qaly': round(delta_qaly, 4),
        'icer': round(icer, 2) if icer != float('inf') else 'Dominant/Dominated',
    }

    print(f"ICER Analysis: {strategy_b_name} vs {strategy_a_name}")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


# --- Demo: Three-state model (Healthy / Sick / Dead) ---
states = ['Healthy', 'Sick', 'Dead']

# Comparator: standard of care
P_comparator = np.array([
    [0.75, 0.20, 0.05],  # Healthy -> Healthy, Sick, Dead
    [0.10, 0.70, 0.20],  # Sick -> Healthy, Sick, Dead
    [0.00, 0.00, 1.00],  # Dead (absorbing)
])

# New intervention: better transitions out of Sick state
P_intervention = np.array([
    [0.80, 0.15, 0.05],
    [0.20, 0.65, 0.15],
    [0.00, 0.00, 1.00],
])

state_costs = np.array([500, 8000, 0])         # Annual costs per state (£)
state_utilities = np.array([0.85, 0.50, 0.00]) # Utility weights
initial = np.array([1.0, 0.0, 0.0])            # All cohort starts healthy

res_comp = run_markov_model(states, P_comparator, state_costs, state_utilities,
                             initial, n_cycles=40)
res_intv = run_markov_model(states, P_intervention, state_costs * 1.0 + np.array([0, 2000, 0]),
                             state_utilities, initial, n_cycles=40)

icer_result = compute_icer(res_comp, res_intv)
```

### Step 2 — PSA Monte Carlo Simulation and CE Plane

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, gamma, dirichlet


def run_psa(
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Probabilistic Sensitivity Analysis via Monte Carlo simulation.

    Samples transition probabilities, utilities, and costs from their
    uncertainty distributions and re-runs the Markov model each iteration.

    Args:
        n_iterations: Number of PSA iterations.
        seed:         Random seed.

    Returns:
        Dict with arrays: delta_costs, delta_qalys, icers, nmb_at_thresholds.
    """
    rng = np.random.default_rng(seed)
    states = ['Healthy', 'Sick', 'Dead']
    initial = np.array([1.0, 0.0, 0.0])

    delta_costs = np.zeros(n_iterations)
    delta_qalys = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Sample transition probabilities using Dirichlet distributions
        # Comparator row for Healthy: alpha proportional to mean
        p_h_comp = rng.dirichlet([15, 4, 1])   # Healthy: mostly stays healthy
        p_s_comp = rng.dirichlet([2, 14, 4])    # Sick: mostly stays sick
        P_comp = np.array([p_h_comp, p_s_comp, [0, 0, 1]])

        p_h_intv = rng.dirichlet([16, 3, 1])    # Intervention: slightly better
        p_s_intv = rng.dirichlet([4, 13, 3])
        P_intv = np.array([p_h_intv, p_s_intv, [0, 0, 1]])

        # Sample utilities from Beta distributions
        u_healthy = rng.beta(34, 6)             # mean ~0.85
        u_sick = rng.beta(10, 10)               # mean ~0.50

        # Sample costs from Gamma distributions (mean, var)
        cost_healthy = rng.gamma(shape=25, scale=20)           # mean ~500
        cost_sick_comp = rng.gamma(shape=64, scale=125)        # mean ~8000
        cost_sick_intv = cost_sick_comp + rng.gamma(shape=4, scale=500)  # +2000 additional

        state_utilities = np.array([u_healthy, u_sick, 0.0])
        costs_comp = np.array([cost_healthy, cost_sick_comp, 0.0])
        costs_intv = np.array([cost_healthy, cost_sick_intv, 0.0])

        r_comp = run_markov_model(states, P_comp, costs_comp, state_utilities,
                                   initial, n_cycles=40, half_cycle_correction=True)
        r_intv = run_markov_model(states, P_intv, costs_intv, state_utilities,
                                   initial, n_cycles=40, half_cycle_correction=True)

        delta_costs[i] = r_intv['total_cost'] - r_comp['total_cost']
        delta_qalys[i] = r_intv['total_qaly'] - r_comp['total_qaly']

    icers = np.where(
        np.abs(delta_qalys) > 1e-6,
        delta_costs / delta_qalys,
        np.inf
    )

    # Plot CE plane
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['green' if dq > 0 and dc < 20000 * dq else 'red'
              for dc, dq in zip(delta_costs, delta_qalys)]
    ax.scatter(delta_qalys, delta_costs, c=colors, alpha=0.3, s=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    # WTP line (£20,000 per QALY)
    q_range = np.linspace(delta_qalys.min(), delta_qalys.max(), 100)
    ax.plot(q_range, 20000 * q_range, 'b--', linewidth=1.5, label='WTP = £20,000/QALY')
    ax.plot(q_range, 50000 * q_range, 'g--', linewidth=1.5, label='WTP = £50,000/QALY')
    ax.set_xlabel('Incremental QALYs')
    ax.set_ylabel('Incremental Cost (£)')
    ax.set_title(f'Cost-Effectiveness Plane (n={n_iterations} PSA iterations)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('ce_plane.png', dpi=150)
    plt.close(fig)

    return {
        'delta_costs': delta_costs,
        'delta_qalys': delta_qalys,
        'icers': icers,
        'mean_delta_cost': round(float(delta_costs.mean()), 2),
        'mean_delta_qaly': round(float(delta_qalys.mean()), 4),
        'mean_icer': round(float(np.median(icers[np.isfinite(icers)])), 2),
    }


# --- Demo ---
psa_results = run_psa(n_iterations=1000)
print(f"Mean ΔCost: {psa_results['mean_delta_cost']}")
print(f"Mean ΔQALY: {psa_results['mean_delta_qaly']}")
print(f"Median ICER: {psa_results['mean_icer']}")
```

### Step 3 — CEAC at Multiple WTP Thresholds

```python
import numpy as np
import matplotlib.pyplot as plt


def compute_ceac(
    delta_costs: np.ndarray,
    delta_qalys: np.ndarray,
    wtp_range: np.ndarray = None,
    output_path: str = 'ceac.png',
) -> pd.DataFrame:
    """
    Compute and plot the Cost-Effectiveness Acceptability Curve (CEAC).

    The CEAC shows the probability that the intervention is cost-effective
    at each willingness-to-pay (WTP) threshold λ.

    CEAC(λ) = P(NMB > 0) = P(λ × ΔQALY − ΔCost > 0)

    Args:
        delta_costs:  Array of incremental costs from PSA iterations.
        delta_qalys:  Array of incremental QALYs from PSA iterations.
        wtp_range:    Array of WTP thresholds to evaluate (£/QALY).
                      Default: 0 to 100,000 in steps of 1,000.
        output_path:  Path to save the CEAC plot.

    Returns:
        DataFrame with columns: wtp, prob_cost_effective.
    """
    if wtp_range is None:
        wtp_range = np.arange(0, 100_001, 1_000)

    probs = []
    for lam in wtp_range:
        nmb = lam * delta_qalys - delta_costs
        prob = float((nmb > 0).mean())
        probs.append(prob)

    ceac_df = pd.DataFrame({'wtp': wtp_range, 'prob_cost_effective': probs})

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wtp_range / 1000, probs, color='navy', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.axvline(20, color='red', linestyle='--', linewidth=1.2,
               label='£20k/QALY (NICE lower)')
    ax.axvline(30, color='orange', linestyle='--', linewidth=1.2,
               label='£30k/QALY (NICE upper)')
    ax.axvline(50, color='green', linestyle='--', linewidth=1.2,
               label='$50k/QALY')
    ax.set_xlabel('Willingness-to-Pay (£ thousands per QALY)')
    ax.set_ylabel('Probability cost-effective')
    ax.set_ylim(0, 1)
    ax.set_title('Cost-Effectiveness Acceptability Curve (CEAC)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # Report at key thresholds
    for threshold in [20_000, 30_000, 50_000, 100_000]:
        row = ceac_df[ceac_df['wtp'] == threshold]
        if not row.empty:
            print(f"P(cost-effective) at £{threshold:,}: "
                  f"{row['prob_cost_effective'].iloc[0]:.3f}")

    return ceac_df


# --- Demo ---
ceac_df = compute_ceac(
    psa_results['delta_costs'],
    psa_results['delta_qalys'],
    output_path='ceac_curve.png',
)
```

---

## Advanced Usage

### WHO Threshold Comparison

```python
def who_threshold_assessment(
    country_gdp_per_capita: float,
    icer: float,
    currency: str = 'USD',
) -> dict:
    """
    Assess cost-effectiveness against WHO thresholds (1× and 3× GDP per capita per DALY).

    Args:
        country_gdp_per_capita: GDP per capita in the given currency.
        icer:                   ICER per QALY/DALY of the intervention.
        currency:               Currency label for display.

    Returns:
        Dict with threshold values and acceptability verdict.
    """
    threshold_1x = country_gdp_per_capita
    threshold_3x = 3 * country_gdp_per_capita

    if icer <= threshold_1x:
        verdict = 'Highly cost-effective (< 1× GDP per capita)'
    elif icer <= threshold_3x:
        verdict = 'Cost-effective (1–3× GDP per capita)'
    else:
        verdict = 'Not cost-effective (> 3× GDP per capita)'

    result = {
        'icer': round(icer, 2),
        'gdp_per_capita': country_gdp_per_capita,
        'threshold_1x': threshold_1x,
        'threshold_3x': threshold_3x,
        'verdict': verdict,
        'currency': currency,
    }
    print(f"WHO threshold assessment: {verdict}")
    print(f"  ICER = {currency}{icer:,.0f} | 1× = {currency}{threshold_1x:,.0f} | 3× = {currency}{threshold_3x:,.0f}")
    return result
```

### Decision Tree Analysis

```python
def decision_tree(branches: list) -> dict:
    """
    Simple decision tree expected value calculation.

    Args:
        branches: List of dicts: {name, probability, cost, qaly}.
                  Probabilities in each strategy must sum to 1.

    Returns:
        Dict with expected_cost and expected_qaly.
    """
    total_prob = sum(b['probability'] for b in branches)
    if not np.isclose(total_prob, 1.0):
        raise ValueError(f"Branch probabilities must sum to 1 (got {total_prob})")

    exp_cost = sum(b['probability'] * b['cost'] for b in branches)
    exp_qaly = sum(b['probability'] * b['qaly'] for b in branches)
    return {'expected_cost': round(exp_cost, 2), 'expected_qaly': round(exp_qaly, 4)}
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `AssertionError: Rows must sum to 1` | Transition matrix rows don't sum to 1 | Normalize each row: `P[i] = P[i] / P[i].sum()` |
| ICER is negative | Intervention is dominant (lower cost, higher QALY) | Report as "Dominant" — negative ICER has no standard interpretation |
| PSA gives infinite ICER | delta_qaly ≈ 0 in some iterations | Use median ICER; filter `np.isfinite(icers)` before averaging |
| CEAC never reaches 1.0 | High parameter uncertainty | Wider distributions; more PSA iterations |
| Dead state not absorbing | Transition matrix allows escape from Dead | Set `P[dead, dead] = 1.0` and all other entries in that row to 0 |
| Utilities > 1 after Beta sampling | Beta parameters set incorrectly | Verify Beta alpha and beta: mean = α/(α+β), must be ≤ 1 |

---

## External Resources

- NICE Decision Support Unit Technical Support Documents: <https://www.sheffield.ac.uk/nice-dsu/tsds>
- Briggs, A., Claxton, K. & Sculpher, M. (2006). *Decision Modelling for Health Economic
  Evaluation*. Oxford University Press.
- Drummond, M.F. et al. (2015). *Methods for the Economic Evaluation of Health Care
  Programmes*. 4th ed. Oxford University Press.
- WHO Guide to cost-effectiveness analysis: <https://www.who.int/choice/publications/p_2003_generalised_cea.pdf>
- R `heemod` package for Markov models: <https://cran.r-project.org/web/packages/heemod/>
- OpenCB (open-source cost-effectiveness): <https://wellcome.org/grant-funding/resources-and-guidance>

---

## Examples

### Example 1 — Three-State Model + ICER Table

```python
states = ['Healthy', 'Sick', 'Dead']
initial = np.array([1.0, 0.0, 0.0])

P_comp = np.array([[0.75, 0.20, 0.05],
                   [0.10, 0.70, 0.20],
                   [0.00, 0.00, 1.00]])
P_intv = np.array([[0.80, 0.15, 0.05],
                   [0.20, 0.65, 0.15],
                   [0.00, 0.00, 1.00]])

costs_comp = np.array([500, 8000, 0])
costs_intv = np.array([500, 10000, 0])
utilities = np.array([0.85, 0.50, 0.0])

r_c = run_markov_model(states, P_comp, costs_comp, utilities, initial, n_cycles=40)
r_i = run_markov_model(states, P_intv, costs_intv, utilities, initial, n_cycles=40)
icer = compute_icer(r_c, r_i)
who = who_threshold_assessment(country_gdp_per_capita=40000, icer=icer['icer'], currency='USD')
```

### Example 2 — Full PSA + CEAC Workflow

```python
psa = run_psa(n_iterations=2000, seed=0)
print(f"Median ICER: £{psa['mean_icer']:,.0f}")

ceac = compute_ceac(psa['delta_costs'], psa['delta_qalys'],
                    wtp_range=np.arange(0, 100_001, 500))
p_at_30k = ceac.loc[ceac['wtp'] == 30_000, 'prob_cost_effective'].values[0]
print(f"P(cost-effective) at £30k/QALY: {p_at_30k:.3f}")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Markov model, PSA, CE plane, CEAC, WHO thresholds |
