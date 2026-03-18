---
name: cantera-combustion
description: >
  Use this Skill for combustion simulations with Cantera: mechanism loading,
  freely propagating flames, ignition delay, reactors, and sensitivity analysis.
tags:
  - engineering
  - combustion
  - cantera
  - chemical-kinetics
  - thermodynamics
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
    - cantera>=3.0
    - numpy>=1.24
    - matplotlib>=3.7
    - pandas>=2.0
    - scipy>=1.11
last_updated: "2026-03-17"
status: "stable"
---

# Cantera Combustion Simulation

> **One-line summary**: Simulate combustion processes with Cantera: load reaction mechanisms, compute adiabatic flame temperatures, ignition delays, laminar flame speeds, and reactor sensitivity analysis.

---

## When to Use This Skill

- When computing adiabatic flame temperature for fuel-air mixtures
- When calculating laminar burning velocity of a fuel
- When simulating ignition delay times for engine/safety applications
- When running perfectly stirred reactor (PSR) simulations
- When performing sensitivity analysis of reaction mechanisms
- When comparing GRI-3.0, YAML, and custom mechanisms

**Trigger keywords**: Cantera, combustion, flame speed, ignition delay, adiabatic flame temperature, reaction mechanism, GRI-3.0, PSR, chemical kinetics, equivalence ratio

---

## Background & Key Concepts

### Equivalence Ratio

$$
\phi = \frac{(F/A)_\text{actual}}{(F/A)_\text{stoichiometric}}
$$

$\phi < 1$: lean mixture; $\phi = 1$: stoichiometric; $\phi > 1$: rich mixture.

### Adiabatic Flame Temperature

Maximum temperature for complete combustion at constant enthalpy:

$$
h_\text{reactants}(T_0) = h_\text{products}(T_\text{ad})
$$

### Laminar Burning Velocity

The speed at which an unburnt gas mixture is consumed by a propagating flame. Key parameter for:
- Engine knock modeling
- Fire safety assessment
- Turbulent combustion modeling (closure)

### Reaction Mechanism

A set of elementary reactions with rate coefficients $k = A T^n e^{-E_a/RT}$ (modified Arrhenius).

---

## Environment Setup

### Install Dependencies

```bash
pip install cantera>=3.0 numpy>=1.24 matplotlib>=3.7 pandas>=2.0 scipy>=1.11
```

### Verify Installation

```python
import cantera as ct
print(f"Cantera version: {ct.__version__}")

# Test: load GRI-3.0 methane mechanism
gas = ct.Solution("gri30.yaml")
print(f"GRI-3.0: {gas.n_species} species, {gas.n_reactions} reactions")
# Expected: 53 species, 325 reactions
```

---

## Core Workflow

### Step 1: Adiabatic Flame Temperature

```python
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

def adiabatic_flame_temperature(fuel, oxidizer="air", phi_range=None, mechanism="gri30.yaml",
                                T_initial=300.0, P=101325.0):
    """
    Compute adiabatic flame temperature over a range of equivalence ratios.

    Parameters
    ----------
    fuel : str
        Fuel species name (e.g., "CH4", "H2", "C2H5OH")
    oxidizer : str
        "air" or specific composition string
    phi_range : array-like
        Equivalence ratios to sweep
    mechanism : str
        Cantera mechanism file (.yaml or .cti)

    Returns
    -------
    pd.DataFrame with phi, T_adiabatic, and selected species
    """
    if phi_range is None:
        phi_range = np.arange(0.5, 2.01, 0.05)

    gas = ct.Solution(mechanism)
    results = []

    for phi in phi_range:
        try:
            gas.set_equivalence_ratio(phi, fuel, oxidizer)
            gas.TP = T_initial, P

            # Equilibrate at constant enthalpy and pressure (adiabatic combustion)
            gas.equilibrate("HP")

            results.append({
                "phi": phi,
                "T_adiabatic_K": gas.T,
                "X_CO2": gas["CO2"].X[0],
                "X_CO":  gas["CO"].X[0],
                "X_H2O": gas["H2O"].X[0],
                "X_O2":  gas["O2"].X[0],
            })
        except ct.CanteraError as e:
            print(f"  phi={phi:.2f}: {e}")

    import pandas as pd
    df = pd.DataFrame(results)
    print(f"\nAFT for {fuel}/{oxidizer}:")
    print(f"  Max T_ad = {df['T_adiabatic_K'].max():.0f} K at φ = {df.loc[df['T_adiabatic_K'].idxmax(), 'phi']:.2f}")
    return df

# Compare methane and hydrogen
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for fuel, color, label in [("CH4", "b", "Methane (GRI-3.0)"),
                             ("H2",  "r", "Hydrogen (GRI-3.0)")]:
    aft_df = adiabatic_flame_temperature(fuel, phi_range=np.arange(0.5, 2.01, 0.05))

    axes[0].plot(aft_df["phi"], aft_df["T_adiabatic_K"], f"{color}-o",
                 ms=4, label=label)
    axes[1].plot(aft_df["phi"], aft_df["X_CO2"] * 100, f"{color}--", label=f"CO₂ ({label[:4]})")
    axes[1].plot(aft_df["phi"], aft_df["X_CO"] * 100, f"{color}:", label=f"CO ({label[:4]})")

axes[0].axvline(1.0, color='k', linestyle='--', linewidth=0.8, label="φ=1 (stoich)")
axes[0].set_xlabel("Equivalence ratio φ"); axes[0].set_ylabel("T_adiabatic (K)")
axes[0].set_title("Adiabatic Flame Temperature"); axes[0].legend()

axes[1].set_xlabel("Equivalence ratio φ"); axes[1].set_ylabel("Mole fraction (%)")
axes[1].set_title("Product Species Mole Fractions"); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("adiabatic_flame_temperature.png", dpi=150)
plt.show()
```

### Step 2: Ignition Delay Time

```python
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

def compute_ignition_delay(fuel, phi=1.0, T_range=None, P=20*101325,
                            mechanism="gri30.yaml"):
    """
    Compute ignition delay time via constant-volume homogeneous reactor.

    Parameters
    ----------
    T_range : array-like
        Initial temperatures (K) to sweep (Arrhenius plot)

    Returns
    -------
    dict: T_K → tau_ign_ms
    """
    if T_range is None:
        T_range = np.arange(1000, 1800, 50)

    gas = ct.Solution(mechanism)
    results = []

    for T0 in T_range:
        gas.set_equivalence_ratio(phi, fuel, "air")
        gas.TP = T0, P
        reactor = ct.IdealGasReactor(gas)
        net = ct.ReactorNet([reactor])

        # Integrate until OH peak (ignition criterion)
        t_end = 0.01  # 10 ms max
        dt = 1e-6
        t, OH_prev = 0.0, gas["OH"].X[0]
        t_ign = None

        while t < t_end:
            t = net.advance(t + dt)
            OH_cur = reactor.thermo["OH"].X[0]
            if OH_cur > 10 * OH_prev and OH_cur > 1e-4:
                t_ign = t
                break
            OH_prev = max(OH_cur, OH_prev)

        tau = (t_ign * 1000) if t_ign else np.nan
        results.append({"T_K": T0, "tau_ign_ms": tau})

    import pandas as pd
    return pd.DataFrame(results).dropna()

# Methane ignition delay
tau_df = compute_ignition_delay("CH4", phi=1.0,
                                 T_range=np.arange(1000, 1700, 50),
                                 P=20*101325)

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(1000 / tau_df["T_K"], tau_df["tau_ign_ms"], 'bs-', ms=6)
ax.set_xlabel("1000/T (K⁻¹)")
ax.set_ylabel("Ignition delay τ (ms)")
ax.set_title("Methane Ignition Delay Time (φ=1.0, P=20 atm)")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("ignition_delay.png", dpi=150)
plt.show()

print("Ignition delays:")
print(tau_df.to_string(index=False))
```

### Step 3: Laminar Flame Speed

```python
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

def laminar_flame_speed(fuel, phi_range=None, T0=300, P=101325, mechanism="gri30.yaml"):
    """
    Compute laminar burning velocity using 1D freely propagating flame.

    Returns
    -------
    pd.DataFrame: phi, Su_cm_s (laminar flame speed in cm/s)
    """
    if phi_range is None:
        phi_range = np.arange(0.7, 1.41, 0.1)

    results = []
    for phi in phi_range:
        gas = ct.Solution(mechanism)
        gas.set_equivalence_ratio(phi, fuel, "air")
        gas.TP = T0, P

        flame = ct.FreeFlame(gas, width=0.03)  # 3 cm domain
        flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        flame.set_max_jac_age(50, 50)
        flame.solve(loglevel=0, auto=True)

        Su = flame.velocity[0] * 100  # m/s → cm/s
        results.append({"phi": phi, "Su_cm_s": Su})
        print(f"  φ={phi:.1f}: Su={Su:.1f} cm/s")

    import pandas as pd
    return pd.DataFrame(results)

print("Computing methane laminar flame speeds...")
flame_df = laminar_flame_speed("CH4", phi_range=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(flame_df["phi"], flame_df["Su_cm_s"], 'ro-', ms=8, linewidth=2)
ax.set_xlabel("Equivalence ratio φ")
ax.set_ylabel("Laminar burning velocity Su (cm/s)")
ax.set_title("Methane/Air Laminar Flame Speed")
ax.set_ylim(0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("laminar_flame_speed.png", dpi=150)
plt.show()

peak_phi = flame_df.loc[flame_df["Su_cm_s"].idxmax(), "phi"]
peak_Su  = flame_df["Su_cm_s"].max()
print(f"\nPeak Su = {peak_Su:.1f} cm/s at φ = {peak_phi:.1f}")
```

---

## Advanced Usage

### Perfectly Stirred Reactor (PSR) and Sensitivity Analysis

```python
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

def psr_residence_time_sweep(fuel, phi=1.0, T0=800, P=101325,
                              tau_range=None, mechanism="gri30.yaml"):
    """
    Sweep residence times in a PSR to find extinction.
    """
    if tau_range is None:
        tau_range = np.logspace(-4, 0, 30)  # 0.1 ms to 1 s

    gas = ct.Solution(mechanism)
    results = []

    for tau in tau_range:
        gas.set_equivalence_ratio(phi, fuel, "air")
        gas.TP = T0, P

        reactor = ct.IdealGasReactor(gas)
        upstream = ct.Reservoir(gas)
        exhaust = ct.Reservoir(gas)

        intake_mfc = ct.MassFlowController(upstream, reactor)
        intake_mfc.mass_flow_rate = reactor.mass / tau

        ct.PressureController(reactor, exhaust, master=intake_mfc)
        net = ct.ReactorNet([reactor])

        try:
            net.advance_to_steady_state()
            results.append({"tau_s": tau, "T_K": reactor.T, "X_CO2": reactor.thermo["CO2"].X[0]})
        except Exception:
            break

    import pandas as pd
    df = pd.DataFrame(results)
    extinction_tau = df[df["T_K"] < T0 + 100]["tau_s"].min() if not df.empty else None
    print(f"PSR extinction residence time: {extinction_tau:.4f} s" if extinction_tau else "No extinction")
    return df

psr_df = psr_residence_time_sweep("CH4")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(psr_df["tau_s"] * 1000, psr_df["T_K"], 'b-o', ms=5)
ax.set_xlabel("Residence time τ (ms)")
ax.set_ylabel("Reactor temperature T (K)")
ax.set_title("PSR S-curve: CH4/Air, φ=1.0")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("psr_s_curve.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Error: `cantera._cantera.CanteraError: mechanism file not found`

**Fix**:
```python
import cantera as ct
# List available built-in mechanisms
import os
print([f for f in os.listdir(ct.get_data_path()) if f.endswith(".yaml")])
# Common: gri30.yaml, h2o2.yaml, air.yaml, nDodecane_Reitz.yaml
```

### Issue: Flame solver doesn't converge

**Fix**:
```python
# Use wider domain and coarser initial grid
flame = ct.FreeFlame(gas, width=0.05)
flame.set_refine_criteria(ratio=5, slope=0.2, curve=0.3)
flame.solve(loglevel=0, auto=True)
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| cantera | 3.0, 3.0.1      | YAML mechanism format preferred over CTI |

---

## External Resources

### Official Documentation

- [Cantera documentation](https://cantera.org/documentation/docs-2.6/sphinx/html/index.html)
- [GRI-3.0 mechanism](http://combustion.berkeley.edu/gri-mech/version30/files30/grimech30.dat)

### Key Papers

- Goodwin, D.G. et al. (2023). *Cantera: An object-oriented software toolkit for chemical kinetics, thermodynamics, and transport processes*. Zenodo.

---

## Examples

### Example 1: NOx Formation vs. Equivalence Ratio

```python
# =============================================
# NOx equilibrium concentration vs. phi
# =============================================
import cantera as ct, numpy as np, matplotlib.pyplot as plt

gas = ct.Solution("gri30.yaml")
phi_arr = np.linspace(0.6, 1.4, 30)
NOx_ppm = []

for phi in phi_arr:
    gas.set_equivalence_ratio(phi, "CH4", "air")
    gas.TP = 300, 101325
    gas.equilibrate("HP")
    NO  = gas["NO"].X[0]
    NO2 = gas["NO2"].X[0]
    NOx_ppm.append((NO + NO2) * 1e6)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(phi_arr, NOx_ppm, 'r-o', ms=5)
ax.set_xlabel("Equivalence ratio φ")
ax.set_ylabel("NOx (ppm)")
ax.set_title("Equilibrium NOx vs. Equivalence Ratio (CH4/air)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("nox_vs_phi.png", dpi=150)
plt.show()
print(f"Peak NOx: {max(NOx_ppm):.0f} ppm at φ = {phi_arr[np.argmax(NOx_ppm)]:.2f}")
```

**Interpreting these results**: NOx peaks near stoichiometric conditions due to the high adiabatic flame temperature. Lean or rich operation reduces NOx — the basis of lean-premix combustors.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
