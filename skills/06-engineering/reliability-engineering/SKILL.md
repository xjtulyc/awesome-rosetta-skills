---
name: reliability-engineering
description: >
  Use this Skill for reliability engineering: Weibull analysis, fault tree
  analysis, FMEA, reliability block diagrams, and accelerated life testing.
tags:
  - engineering
  - reliability
  - weibull
  - fmea
  - fault-tree
version: "1.0.0"
authors:
  - name: Rosetta Skills Contributors
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - numpy>=1.24
    - scipy>=1.11
    - matplotlib>=3.7
    - pandas>=2.0
    - reliability>=0.8
last_updated: "2026-03-17"
status: "stable"
---

# Reliability Engineering Analysis

> **One-line summary**: Quantify product reliability using Weibull analysis, fault trees, FMEA risk prioritization, and reliability block diagrams with the `reliability` library and scipy.

---

## When to Use This Skill

- When fitting Weibull distributions to failure time data
- When computing MTTF, B10/B50 life, and hazard rates
- When building fault tree diagrams and computing top-event probability
- When performing FMEA and calculating Risk Priority Numbers (RPN)
- When modeling series/parallel/k-of-n reliability block diagrams
- When designing accelerated life tests (Arrhenius or power-law stress)

**Trigger keywords**: Weibull, reliability, MTTF, B10 life, fault tree, FMEA, RPN, hazard function, accelerated life test, reliability block diagram, failure rate

---

## Background & Key Concepts

### Weibull Distribution

The 2-parameter Weibull PDF:

$$
f(t) = \frac{\beta}{\eta}\left(\frac{t}{\eta}\right)^{\beta-1} \exp\left[-\left(\frac{t}{\eta}\right)^\beta\right]
$$

- $\beta$: shape parameter — $\beta < 1$ decreasing failure rate (infant mortality), $\beta = 1$ constant (exponential), $\beta > 1$ increasing (wear-out)
- $\eta$: scale parameter (characteristic life at 63.2% failure)

Reliability function: $R(t) = \exp\left[-\left(t/\eta\right)^\beta\right]$

B-life: $B_p = \eta \left[-\ln(1-p/100)\right]^{1/\beta}$

### Fault Tree Analysis (FTA)

Top-down deductive analysis: top event probability computed by propagating basic event probabilities through AND/OR gates.

- AND gate: $P = \prod_i P_i$
- OR gate: $P = 1 - \prod_i (1-P_i)$

### Risk Priority Number (FMEA)

$$
\text{RPN} = \text{Severity} \times \text{Occurrence} \times \text{Detection}
$$

Each scored 1–10; RPNs ≥ 100 typically require corrective action.

---

## Environment Setup

### Install Dependencies

```bash
pip install numpy>=1.24 scipy>=1.11 matplotlib>=3.7 pandas>=2.0 reliability>=0.8
```

### Verify Installation

```python
import reliability
from reliability.Fitters import Fit_Weibull_2P
import numpy as np

# Synthetic failure data
np.random.seed(42)
failures = np.random.weibull(2.0, 20) * 1000  # beta=2, eta=1000 h
f = Fit_Weibull_2P(failures=failures, show_probability_plot=False, print_results=False)
print(f"Weibull fit: beta={f.beta:.3f}, eta={f.eta:.1f} h")
# Expected: beta≈2, eta≈1000
```

---

## Core Workflow

### Step 1: Weibull Parameter Estimation and B-life

```python
import numpy as np
import matplotlib.pyplot as plt
from reliability.Fitters import Fit_Weibull_2P
from reliability.Distributions import Weibull_Distribution
from reliability.Utils import colorprint

np.random.seed(42)

# ------------------------------------------------------------------ #
# Simulated field failure data (hours) with some right-censored units
# ------------------------------------------------------------------ #
failures = np.array([
    342, 487, 623, 741, 856, 980, 1105, 1230, 1380, 1520,
    1650, 1790, 1940, 2100, 2280, 2450, 2700, 3010, 3400, 4200
])
right_censored = np.array([500, 750, 1000, 1500, 2000, 3000, 5000])

# Fit 2-parameter Weibull (MLE with censored data)
f = Fit_Weibull_2P(
    failures=failures,
    right_censored=right_censored,
    show_probability_plot=False,
    print_results=True,
)

beta_hat = f.beta
eta_hat  = f.eta
print(f"\nEstimated parameters: beta={beta_hat:.4f}, eta={eta_hat:.2f} h")

# ----- B-life calculations ---------------------------------------- #
dist = Weibull_Distribution(alpha=eta_hat, beta=beta_hat)

def b_life(dist, p_pct):
    """Return the age by which p% of units have failed."""
    return dist.inverse_SF(1 - p_pct/100)

for p in [1, 5, 10, 50]:
    print(f"  B{p:2d} life = {b_life(dist, p):8.1f} h  "
          f"(R={100-p}% survive to this age)")

# MTTF and variance
mttf = dist.mean
sigma = dist.standard_deviation
print(f"\nMTTF = {mttf:.1f} h,  σ = {sigma:.1f} h")

# ----- Reliability and hazard plots -------------------------------- #
t_range = np.linspace(1, 8000, 500)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Reliability (survival)
R = dist.SF(xvals=t_range)
axes[0].plot(t_range, R, 'b-', linewidth=2)
axes[0].axhline(0.9, color='red', linestyle='--', alpha=0.7, label='R=0.9')
axes[0].set_xlabel("Time (h)"); axes[0].set_ylabel("Reliability R(t)")
axes[0].set_title("Survival Function"); axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Cumulative failure (CDF)
F = dist.CDF(xvals=t_range)
axes[1].plot(t_range, F * 100, 'r-', linewidth=2)
axes[1].set_xlabel("Time (h)"); axes[1].set_ylabel("Failures (%)")
axes[1].set_title("Cumulative Failure Probability"); axes[1].grid(True, alpha=0.3)

# Hazard rate
h = dist.HF(xvals=t_range)
axes[2].plot(t_range, h * 1000, 'g-', linewidth=2)
axes[2].set_xlabel("Time (h)"); axes[2].set_ylabel("Hazard rate (×10⁻³ /h)")
axes[2].set_title("Hazard Function"); axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Weibull Reliability  (β={beta_hat:.2f}, η={eta_hat:.0f} h)", y=1.02)
plt.tight_layout()
plt.savefig("weibull_reliability.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Step 2: Fault Tree Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ------------------------------------------------------------------ #
# Define the fault tree structure and compute top-event probability
# ------------------------------------------------------------------ #

# Basic event failure probabilities (per mission)
basic_events = {
    "BE1": {"name": "Pump A fails",        "prob": 0.02},
    "BE2": {"name": "Pump B fails",        "prob": 0.03},
    "BE3": {"name": "Control valve fails", "prob": 0.01},
    "BE4": {"name": "Sensor fails",        "prob": 0.05},
    "BE5": {"name": "Power supply fails",  "prob": 0.005},
}

def or_gate(*probs):
    """Probability that AT LEAST ONE event occurs."""
    return 1.0 - np.prod([1 - p for p in probs])

def and_gate(*probs):
    """Probability that ALL events occur simultaneously."""
    return np.prod(list(probs))

# Tree logic:
# G1 (OR): pump subsystem failure = BE1 OR BE2
# G2 (AND): control failure = BE3 AND (BE4 OR BE5)
# TOP (OR): G1 OR G2

p_be1 = basic_events["BE1"]["prob"]
p_be2 = basic_events["BE2"]["prob"]
p_be3 = basic_events["BE3"]["prob"]
p_be4 = basic_events["BE4"]["prob"]
p_be5 = basic_events["BE5"]["prob"]

# Intermediate gates
p_g1 = or_gate(p_be1, p_be2)          # Pump subsystem
p_g2_sub = or_gate(p_be4, p_be5)      # Sensor OR Power
p_g2 = and_gate(p_be3, p_g2_sub)      # Control subsystem

# Top event
p_top = or_gate(p_g1, p_g2)

print("=== Fault Tree Analysis Results ===")
print(f"G1  (Pump subsystem OR):   P = {p_g1:.6f}  ({p_g1*100:.3f}%)")
print(f"G2  (Control subsystem AND): P = {p_g2:.6f}  ({p_g2*100:.3f}%)")
print(f"TOP (System failure OR):    P = {p_top:.6f}  ({p_top*100:.3f}%)")

# ---- Importance measures (Birnbaum) ------------------------------ #
def birnbaum_importance(basic_event_key, p_top_original):
    """
    Compute Birnbaum structural importance:
    I_B = P(top | event=1) - P(top | event=0)
    """
    # Create modified event sets
    results = {}
    for state in [0, 1]:
        p = {k: v["prob"] for k, v in basic_events.items()}
        p[basic_event_key] = state
        _g1 = or_gate(p["BE1"], p["BE2"])
        _sub = or_gate(p["BE4"], p["BE5"])
        _g2 = and_gate(p["BE3"], _sub)
        results[state] = or_gate(_g1, _g2)
    return results[1] - results[0]

print("\n=== Birnbaum Importance Measures ===")
importances = {}
for key in basic_events:
    imp = birnbaum_importance(key, p_top)
    importances[key] = imp
    print(f"  {key} ({basic_events[key]['name']:25s}): I_B = {imp:.6f}")

# Bar chart of importances
fig, ax = plt.subplots(figsize=(8, 4))
labels = [f"{k}\n{basic_events[k]['name']}" for k in importances]
values = list(importances.values())
colors = ['#e74c3c' if v == max(values) else '#3498db' for v in values]
bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.8)
ax.set_ylabel("Birnbaum Importance")
ax.set_title("Fault Tree: Birnbaum Structural Importance")
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
            f"{val:.4f}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("fault_tree_importance.png", dpi=150)
plt.show()

print(f"\n[Top-event failure probability per mission: {p_top:.4%}]")
```

### Step 3: FMEA and RPN Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ------------------------------------------------------------------ #
# FMEA worksheet — Failure Mode and Effects Analysis
# ------------------------------------------------------------------ #
fmea_data = {
    "Component":    ["Pump Motor",   "Pump Motor",   "Bearing",      "Seal",
                     "Control PCB",  "Control PCB",  "Power Supply", "Sensor"],
    "Failure Mode": ["Winding short", "Overheating",  "Fatigue crack", "Leakage",
                     "ADC drift",    "Firmware hang", "Capacitor fail", "Bias offset"],
    "Effect":       ["No flow",      "Reduced life",  "Vibration",    "Contamination",
                     "Wrong reading","No control",   "Shutdown",      "False alarm"],
    "S":            [9, 7, 8, 6, 7, 9, 8, 5],   # Severity    (1-10)
    "O":            [3, 4, 2, 5, 3, 2, 4, 6],   # Occurrence  (1-10)
    "D":            [5, 6, 4, 3, 4, 3, 5, 4],   # Detection   (1-10)
}

df = pd.DataFrame(fmea_data)
df["RPN"] = df["S"] * df["O"] * df["D"]
df["Criticality"] = df["S"] * df["O"]  # SOD without detection

# Sort by RPN descending
df = df.sort_values("RPN", ascending=False).reset_index(drop=True)
df["Rank"] = range(1, len(df)+1)

# Threshold for corrective action
RPN_THRESHOLD = 100

print("=== FMEA Results (sorted by RPN) ===")
print(df[["Rank","Component","Failure Mode","S","O","D","RPN"]].to_string(index=False))
print(f"\nFailure modes requiring corrective action (RPN ≥ {RPN_THRESHOLD}):")
high_risk = df[df["RPN"] >= RPN_THRESHOLD]
for _, row in high_risk.iterrows():
    print(f"  [{row['Rank']}] {row['Component']} — {row['Failure Mode']}: RPN={row['RPN']}")

# ---- Pareto chart of RPNs --------------------------------------- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pareto
colors = ['#e74c3c' if r >= RPN_THRESHOLD else '#3498db' for r in df["RPN"]]
bars = ax1.bar(range(len(df)), df["RPN"], color=colors, edgecolor='black', linewidth=0.7)
cumulative = df["RPN"].cumsum() / df["RPN"].sum() * 100
ax1_twin = ax1.twinx()
ax1_twin.plot(range(len(df)), cumulative, 'ko-', markersize=5, linewidth=1.5)
ax1_twin.set_ylabel("Cumulative % of total RPN")
ax1_twin.axhline(80, color='gray', linestyle='--', alpha=0.5, label='80%')
ax1.axhline(RPN_THRESHOLD, color='red', linestyle='--', linewidth=1.5, label=f'Threshold={RPN_THRESHOLD}')
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels([f"{r}\n{m[:8]}" for r, m in zip(df["Component"], df["Failure Mode"])],
                     fontsize=7, rotation=30, ha='right')
ax1.set_ylabel("RPN"); ax1.set_title("FMEA Pareto Chart (RPN)")
ax1.legend(loc='upper right', fontsize=8)

# S-O criticality matrix bubble chart
sc = ax2.scatter(df["O"], df["S"], s=df["D"]*40, c=df["RPN"],
                  cmap='RdYlGn_r', vmin=0, vmax=400, edgecolors='black', linewidth=0.7)
plt.colorbar(sc, ax=ax2, label='RPN')
for _, row in df.iterrows():
    ax2.annotate(f"{row['Component'][:6]}", (row['O'], row['S']),
                  textcoords="offset points", xytext=(4, 4), fontsize=7)
ax2.set_xlabel("Occurrence (O)"); ax2.set_ylabel("Severity (S)")
ax2.set_title("FMEA Criticality Matrix\n(bubble size ∝ Detection)")
ax2.set_xlim(0, 11); ax2.set_ylim(0, 11)
ax2.grid(True, alpha=0.3)
# Risk zones
ax2.fill_between([7, 11], [7, 7], [11, 11], alpha=0.07, color='red')
ax2.text(8.5, 9.5, "High\nRisk", ha='center', color='red', fontsize=9)

plt.tight_layout()
plt.savefig("fmea_analysis.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Reliability Block Diagram (Series-Parallel System)

```python
import numpy as np
import matplotlib.pyplot as plt
from reliability.Distributions import Weibull_Distribution

# ------------------------------------------------------------------ #
# System: two subsystems in series; subsystem B has 2 parallel units
# Subsystem A: single component (must not fail)
# Subsystem B: 2-of-2 parallel redundancy (at least 1 must survive)
# ------------------------------------------------------------------ #

def R_subsystem_A(t):
    """Single Weibull component."""
    dist = Weibull_Distribution(alpha=2000, beta=2.0)
    return dist.SF(xvals=t)

def R_component_B(t):
    """Individual B component reliability."""
    dist = Weibull_Distribution(alpha=3000, beta=1.5)
    return dist.SF(xvals=t)

def R_subsystem_B_parallel(t):
    """1-of-2 parallel: R = 1 - (1-R_b)^2."""
    R_b = R_component_B(t)
    return 1 - (1 - R_b)**2

def R_system(t):
    """Series combination of A and B_parallel."""
    return R_subsystem_A(t) * R_subsystem_B_parallel(t)

t = np.linspace(1, 6000, 500)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, R_subsystem_A(t),        'b--',  label='Subsystem A (single)', linewidth=1.5)
ax.plot(t, R_component_B(t),        'g:',   label='Component B (single)', linewidth=1.5)
ax.plot(t, R_subsystem_B_parallel(t),'g--', label='Subsystem B (1-of-2 parallel)', linewidth=2)
ax.plot(t, R_system(t),             'r-',   label='SYSTEM (A series B_parallel)', linewidth=2.5)
ax.axhline(0.9, color='gray', linestyle='--', alpha=0.6, label='R=0.9 target')

# Find mission time for 90% system reliability
from scipy.optimize import brentq
t_mission = brentq(lambda x: R_system(np.array([x]))[0] - 0.90, 1, 5999)
ax.axvline(t_mission, color='orange', linestyle=':', linewidth=2)
ax.text(t_mission + 100, 0.5, f"T(R=0.9)={t_mission:.0f} h",
        color='orange', fontsize=10)

ax.set_xlabel("Time (h)"); ax.set_ylabel("Reliability R(t)")
ax.set_title("Reliability Block Diagram — Series-Parallel System")
ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rbd_system.png", dpi=150)
plt.show()
```

### Accelerated Life Testing (Arrhenius Model)

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Arrhenius acceleration model: MTTF(T) = A * exp(Ea / (k*T))
# Ea: activation energy (eV), k: Boltzmann constant
# ------------------------------------------------------------------ #

k_B = 8.617e-5  # Boltzmann constant (eV/K)

# ALT data: (stress temperature K, measured MTTF hours)
test_temps_K = np.array([398, 423, 448])  # 125°C, 150°C, 175°C
mttf_observed = np.array([5000, 1800, 700])

def arrhenius_mttf(T, A, Ea):
    return A * np.exp(Ea / (k_B * T))

popt, pcov = curve_fit(arrhenius_mttf, test_temps_K, mttf_observed, p0=[1e-5, 0.8])
A_hat, Ea_hat = popt
print(f"Arrhenius fit: A={A_hat:.4e}, Ea={Ea_hat:.4f} eV")

# Extrapolate to use condition (25°C = 298 K)
T_use = 298.0
mttf_use = arrhenius_mttf(T_use, A_hat, Ea_hat)
print(f"Extrapolated MTTF at {T_use-273:.0f}°C: {mttf_use:.0f} h  ({mttf_use/8760:.1f} years)")

# Acceleration factor vs. use condition
for T_test in test_temps_K:
    AF = mttf_use / arrhenius_mttf(T_test, A_hat, Ea_hat)
    print(f"  AF ({T_test-273:.0f}°C → 25°C) = {AF:.1f}x")

# Plot
T_range = np.linspace(280, 460, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(1000/T_range, arrhenius_mttf(T_range, A_hat, Ea_hat),
            'b-', linewidth=2, label='Arrhenius fit')
ax.semilogy(1000/test_temps_K, mttf_observed, 'rs', markersize=10,
            label='ALT data', zorder=5)
ax.semilogy(1000/T_use, mttf_use, 'g^', markersize=12, label=f'Use condition (25°C): {mttf_use:.0f} h')
ax.set_xlabel("1000/T  (1/K)"); ax.set_ylabel("MTTF (h)")
ax.set_title(f"Arrhenius Plot  (Ea = {Ea_hat:.3f} eV)")
ax.legend(); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("arrhenius_plot.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Error: `Fit_Weibull_2P` fails with small samples

**Cause**: MLE requires at least ~5 failures. With fewer, use median rank regression.

**Fix**:
```python
from reliability.Fitters import Fit_Weibull_2P
# Force rank regression for small samples
f = Fit_Weibull_2P(failures=failures, method='RRX', show_probability_plot=False)
```

### Error: `brentq` root not found for B-life

**Cause**: Target reliability outside the distribution range.

**Fix**:
```python
from scipy.optimize import brentq
# Always check that R(0) > target and R(large_t) < target
R_0 = dist.SF(xvals=np.array([1e-6]))[0]
print(f"R at t→0: {R_0:.6f}")  # Should be ~1.0
```

### Warning: Weibull fit with censored data

Use `right_censored` parameter — never mix censored and failure times in the `failures` array:

```python
f = Fit_Weibull_2P(
    failures=np.array([100, 200, 350]),
    right_censored=np.array([400, 500]),  # units still running
)
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| reliability | 0.8.x | API stable since 0.8; check changelog for 0.9 |
| scipy | 1.11, 1.12 | `curve_fit` p0 required for Arrhenius |
| numpy | 1.24, 1.26 | No known issues |

---

## External Resources

### Official Documentation

- [reliability library docs](https://reliability.readthedocs.io)
- [MIL-HDBK-217F: Reliability Prediction of Electronic Equipment](https://www.sre.org/pubs/Mil-Hdbk-217F.pdf)

### Key Papers

- Meeker, W.Q. & Escobar, L.A. (1998). *Statistical Methods for Reliability Data*. Wiley.
- ReVelle, J.B. (2004). *Manufacturing Handbook of Best Practices*. CRC Press.

---

## Examples

### Example 1: Fleet Replacement Interval Optimization

```python
import numpy as np
from scipy.optimize import minimize_scalar
from reliability.Distributions import Weibull_Distribution

# Cost model: minimize cost per unit time
# C_p = planned replacement cost, C_f = failure cost
C_p = 500   # USD — planned preventive replacement
C_f = 5000  # USD — unplanned corrective replacement (10× penalty)
beta, eta = 2.5, 8000  # Weibull parameters (hours)

dist = Weibull_Distribution(alpha=eta, beta=beta)

def cost_rate(tp):
    """Expected cost per unit time for preventive replacement interval tp."""
    if tp <= 0:
        return np.inf
    R_tp = dist.SF(xvals=np.array([tp]))[0]
    F_tp = 1 - R_tp
    # Expected cycle length (renewal theory)
    # MTTF conditional on surviving to tp, plus probability of failure * expected time to fail
    # Simplified: cost rate = (C_p*R_tp + C_f*F_tp) / expected_cycle_length
    # Expected cycle = integral_0^tp R(t)dt  (renewal reward theorem)
    t_arr = np.linspace(0, tp, 1000)
    mean_cycle = np.trapz(dist.SF(xvals=t_arr), t_arr)
    return (C_p * R_tp + C_f * F_tp) / mean_cycle

result = minimize_scalar(cost_rate, bounds=(100, 20000), method='bounded')
t_opt = result.x
print(f"Optimal replacement interval: {t_opt:.0f} h")
print(f"Cost rate at optimum: ${cost_rate(t_opt):.4f}/h")
print(f"Annual cost (8760 h/yr): ${cost_rate(t_opt)*8760:.0f}")

# Compare with run-to-failure
t_no_pm = minimize_scalar(lambda t: cost_rate(t), bounds=(100000, 200000), method='bounded').x
cr_rtf = C_f / dist.mean  # Run-to-failure cost rate
print(f"\nRun-to-failure cost rate: ${cr_rtf:.4f}/h  (annual: ${cr_rtf*8760:.0f})")
print(f"Savings from PM: {(cr_rtf - cost_rate(t_opt))/cr_rtf*100:.1f}%")
```

### Example 2: System Availability with Repair

```python
import numpy as np

# ------------------------------------------------------------------ #
# Steady-state availability: A = MTTF / (MTTF + MTTR)
# For repairable system with Weibull failures and exponential repair
# ------------------------------------------------------------------ #
from reliability.Distributions import Weibull_Distribution

beta, eta = 2.0, 5000   # Failure distribution
MTTR = 8.0              # Mean time to repair (hours)

dist = Weibull_Distribution(alpha=eta, beta=beta)
MTTF = dist.mean

A_steady = MTTF / (MTTF + MTTR)
unavailability = 1 - A_steady

print(f"MTTF    = {MTTF:.1f} h")
print(f"MTTR    = {MTTR:.1f} h")
print(f"Availability A = {A_steady:.6f}  ({A_steady*100:.4f}%)")
print(f"Unavailability  = {unavailability:.6f}  ({unavailability*1e6:.1f} PPM)")
print(f"Downtime/year   = {unavailability*8760:.1f} h/yr")

# Sensitivity: required MTTR to achieve 99.9% availability
target_A = 0.999
MTTR_required = MTTF * (1 - target_A) / target_A
print(f"\nTo achieve A={target_A}: MTTR ≤ {MTTR_required:.2f} h")
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
