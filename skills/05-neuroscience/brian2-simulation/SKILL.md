---
name: brian2-simulation
description: >
  Use this Skill to simulate spiking neural networks with Brian2: LIF/AdEx
  neurons, STDP, recurrent networks, raster plots, and mean-field analysis.
tags:
  - neuroscience
  - computational-neuroscience
  - brian2
  - spiking-networks
  - neural-simulation
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
    - brian2>=2.5
    - numpy>=1.24
    - matplotlib>=3.7
    - scipy>=1.11
    - pandas>=2.0
last_updated: "2026-03-17"
status: "stable"
---

# Brian2 Spiking Neural Network Simulation

> **One-line summary**: Simulate biologically plausible spiking neural networks with Brian2: LIF, AdEx, Hodgkin-Huxley neurons, STDP plasticity, and recurrent network dynamics.

---

## When to Use This Skill

- When simulating integrate-and-fire or conductance-based neuron models
- When studying spike-timing-dependent plasticity (STDP)
- When implementing recurrent excitatory-inhibitory networks
- When generating raster plots and firing rate analyses
- When fitting mean-field equations to network simulations
- When comparing neural coding hypotheses (rate vs. temporal)

**Trigger keywords**: Brian2, spiking neural network, integrate-and-fire, LIF, AdEx, Hodgkin-Huxley, STDP, synaptic plasticity, raster plot, neural simulation

---

## Background & Key Concepts

### Leaky Integrate-and-Fire (LIF) Model

$$
\tau_m \frac{dV}{dt} = -(V - V_\text{rest}) + R \cdot I(t)
$$

When $V \geq V_\text{thresh}$: fire a spike, reset $V = V_\text{reset}$.

Parameters: $\tau_m$ = membrane time constant, $R$ = membrane resistance, $V_\text{rest}$ = resting potential.

### Adaptive Exponential (AdEx) Model

Adds spike-triggered adaptation:

$$
C \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{(V-V_T)/\Delta_T} - w + I
$$
$$
\tau_w \frac{dw}{dt} = a(V - E_L) - w
$$

### Spike-Timing-Dependent Plasticity (STDP)

Synaptic weight changes based on spike timing:

$$
\Delta w = \begin{cases} A_+ e^{-|\Delta t|/\tau_+} & \text{if } t_\text{post} > t_\text{pre} \\ -A_- e^{-|\Delta t|/\tau_-} & \text{if } t_\text{post} < t_\text{pre} \end{cases}
$$

---

## Environment Setup

### Install Dependencies

```bash
pip install brian2>=2.5 numpy>=1.24 matplotlib>=3.7 scipy>=1.11 pandas>=2.0
```

### Verify Installation

```python
import brian2 as b2
print(f"Brian2 version: {b2.__version__}")
b2.start_scope()
G = b2.NeuronGroup(10, "dv/dt = -v/(10*ms) : 1", threshold="v>0.9", reset="v=0")
print("Brian2 functional")
```

---

## Core Workflow

### Step 1: Single Neuron Models

```python
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

b2.start_scope()

# --- LIF Neuron with step current ---
tau = 10 * b2.ms
Vr = -65 * b2.mV
Vt = -50 * b2.mV
Vreset = -70 * b2.mV
R = 10 * b2.Mohm

eqs_lif = """
    dV/dt = (-(V - Vr) + R*I) / tau : volt
    I : amp
"""

G_lif = b2.NeuronGroup(1, eqs_lif,
                        threshold="V >= Vt",
                        reset="V = Vreset",
                        method="exact",
                        namespace={"tau": tau, "Vr": Vr, "Vt": Vt, "R": R, "Vreset": Vreset})
G_lif.V = Vr

# Step current: 0→200 pA at t=100ms
M_lif = b2.StateMonitor(G_lif, ["V"], record=True)
S_lif = b2.SpikeMonitor(G_lif)

# First 100ms: no input; then step current
@b2.network_operation(dt=b2.defaultclock.dt)
def update_current():
    t = b2.defaultclock.t
    G_lif.I = 2.5 * b2.nA if t > 100 * b2.ms else 0 * b2.nA

net = b2.Network(G_lif, M_lif, S_lif, update_current)
net.run(500 * b2.ms)

print(f"LIF neuron fired {S_lif.num_spikes} times")
print(f"Firing rate: {S_lif.num_spikes / 0.4:.1f} Hz (during stimulus)")

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(M_lif.t / b2.ms, M_lif.V[0] / b2.mV, 'b-', linewidth=0.8)
ax.axvline(100, color='r', linestyle='--', label='Stimulus onset')
for t_sp in S_lif.t / b2.ms:
    ax.axvline(t_sp, color='k', alpha=0.5, linewidth=0.5)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("V (mV)")
ax.set_title(f"LIF Neuron — {S_lif.num_spikes} spikes")
ax.legend()
plt.tight_layout()
plt.savefig("lif_neuron.png", dpi=150)
plt.show()
```

### Step 2: Recurrent E-I Network

```python
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

b2.start_scope()
b2.defaultclock.dt = 0.1 * b2.ms

N_E = 400  # Excitatory neurons
N_I = 100  # Inhibitory neurons

# Neuron parameters
tau_E, tau_I = 20*b2.ms, 10*b2.ms
V_rest = -70*b2.mV
V_thresh = -50*b2.mV
V_reset = -60*b2.mV
tau_ref = 2*b2.ms

# LIF with refractory period
eqs = """
    dV/dt = (-(V - V_rest) + I_ext + I_syn) / tau : volt (unless refractory)
    I_ext : volt
    I_syn : volt
"""

# Create populations
P_E = b2.NeuronGroup(N_E, eqs,
    threshold="V >= V_thresh", reset="V = V_reset",
    refractory=tau_ref, method="euler",
    namespace={"V_rest": V_rest, "V_thresh": V_thresh,
               "V_reset": V_reset, "tau": tau_E})

P_I = b2.NeuronGroup(N_I, eqs,
    threshold="V >= V_thresh", reset="V = V_reset",
    refractory=tau_ref, method="euler",
    namespace={"V_rest": V_rest, "V_thresh": V_thresh,
               "V_reset": V_reset, "tau": tau_I})

P_E.V = V_rest + b2.rand(N_E) * 10 * b2.mV
P_I.V = V_rest + b2.rand(N_I) * 10 * b2.mV
P_E.I_ext = 5 * b2.mV  # tonic input

# Connections (random, sparse)
S_EE = b2.Synapses(P_E, P_E, on_pre="I_syn += 0.5*mV", delay=1*b2.ms)
S_EI = b2.Synapses(P_E, P_I, on_pre="I_syn += 1.0*mV", delay=1*b2.ms)
S_IE = b2.Synapses(P_I, P_E, on_pre="I_syn -= 2.0*mV", delay=1*b2.ms)

S_EE.connect(p=0.05, condition="i != j")
S_EI.connect(p=0.10)
S_IE.connect(p=0.25)

# Monitors
sp_E = b2.SpikeMonitor(P_E)
sp_I = b2.SpikeMonitor(P_I)
rate_E = b2.PopulationRateMonitor(P_E)

net = b2.Network(P_E, P_I, S_EE, S_EI, S_IE, sp_E, sp_I, rate_E)
net.run(500 * b2.ms)

# Raster plot + population rate
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Raster
axes[0].plot(sp_E.t/b2.ms, sp_E.i, 'k.', ms=1, alpha=0.5, label="Excitatory")
axes[0].plot(sp_I.t/b2.ms, sp_I.i + N_E, 'r.', ms=1, alpha=0.5, label="Inhibitory")
axes[0].axhline(N_E, color='r', linestyle='--', linewidth=0.5)
axes[0].set_ylabel("Neuron index")
axes[0].set_title(f"E-I Network: {N_E}E + {N_I}I neurons")
axes[0].legend(markerscale=5)

# Population rate
rate_smooth = b2.PopulationRateMonitor(P_E)
t = rate_E.t / b2.ms
rate_Hz = rate_E.smooth_rate("gaussian", width=20*b2.ms) / b2.Hz
axes[1].plot(t, rate_Hz, 'b-')
axes[1].set_xlabel("Time (ms)"); axes[1].set_ylabel("Rate (Hz)")
axes[1].set_title("Population Firing Rate (E)")

plt.tight_layout()
plt.savefig("ei_network.png", dpi=150)
plt.show()

print(f"Mean E rate: {rate_Hz[rate_Hz > 0].mean():.1f} Hz")
```

### Step 3: STDP Synaptic Plasticity

```python
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

b2.start_scope()

N = 100
tau_pre  = 20 * b2.ms
tau_post = 20 * b2.ms
A_plus  = 0.01
A_minus = 0.0105  # slightly asymmetric → LTD > LTP
w_max = 1.0
w_init = 0.5

stdp_eqs = """
    w : 1
    dA_pre/dt  = -A_pre  / tau_pre  : 1 (event-driven)
    dA_post/dt = -A_post / tau_post : 1 (event-driven)
"""

pre_on = """
    A_pre += {A_plus}
    w = clip(w + A_post * {A_minus}, 0, {w_max})
    v_post += w
""".format(A_plus=A_plus, A_minus=A_minus, w_max=w_max)

post_on = """
    A_post += {A_minus}
    w = clip(w + A_pre * {A_plus}, 0, {w_max})
""".format(A_minus=A_minus, A_plus=A_plus)

G_pre  = b2.PoissonGroup(N, rates=20*b2.Hz)
G_post = b2.NeuronGroup(N, "dv/dt = -v/(10*ms) : 1",
                        threshold="v > 1", reset="v = 0", method="euler")

syn = b2.Synapses(G_pre, G_post, model=stdp_eqs,
                  on_pre=pre_on, on_post=post_on)
syn.connect(j="i")  # one-to-one
syn.w = w_init

W_monitor = b2.StateMonitor(syn, "w", record=range(min(20, N)))
b2.run(5 * b2.second)

# Weight distribution before and after
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
final_weights = syn.w[:]
axes[0].hist(final_weights, bins=20, color="steelblue", edgecolor="white")
axes[0].axvline(w_init, color='r', linestyle='--', label=f"Initial w={w_init}")
axes[0].set_xlabel("Synaptic weight"); axes[0].set_ylabel("Count")
axes[0].set_title("Weight Distribution After STDP"); axes[0].legend()

# Weight trajectories for first 5 synapses
for i in range(min(5, len(W_monitor.t))):
    axes[1].plot(W_monitor.t/b2.second, W_monitor.w[i], lw=0.8, alpha=0.8)
axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("Weight")
axes[1].set_title("Weight Trajectories (5 synapses)")

plt.tight_layout()
plt.savefig("stdp_weights.png", dpi=150)
plt.show()

print(f"Initial mean weight: {w_init:.3f}")
print(f"Final mean weight:   {final_weights.mean():.3f}")
print(f"Final std weight:    {final_weights.std():.3f}")
```

---

## Advanced Usage

### Hodgkin-Huxley Conductance Model

```python
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

b2.start_scope()

hh_eqs = """
    dV/dt = (I_ext - gNa*m**3*h*(V-ENa) - gK*n**4*(V-EK) - gL*(V-EL)) / Cm : volt
    dm/dt = alpham*(1-m) - betam*m : 1
    dh/dt = alphah*(1-h) - betah*h : 1
    dn/dt = alphan*(1-n) - betan*n : 1
    alpham = (0.1/mV) * (-V - 40*mV) / (exp((-V - 40*mV)/(10*mV)) - 1)/ms : Hz
    betam  = 4 * exp((-V - 65*mV)/(18*mV))/ms : Hz
    alphah = 0.07 * exp((-V - 65*mV)/(20*mV))/ms : Hz
    betah  = 1 / (exp((-V - 35*mV)/(10*mV)) + 1)/ms : Hz
    alphan = (0.01/mV)*(-V - 55*mV) / (exp((-V-55*mV)/(10*mV)) - 1)/ms : Hz
    betan  = 0.125 * exp((-V - 65*mV)/(80*mV))/ms : Hz
    I_ext : amp/metre**2
"""
params = dict(gNa=120*b2.msiemens/b2.cm**2, gK=36*b2.msiemens/b2.cm**2,
              gL=0.3*b2.msiemens/b2.cm**2, ENa=50*b2.mV, EK=-77*b2.mV,
              EL=-54.4*b2.mV, Cm=1*b2.ufarad/b2.cm**2)

G_hh = b2.NeuronGroup(1, hh_eqs, method="exponential_euler", namespace=params)
G_hh.V = -65 * b2.mV
G_hh.m = 0.05; G_hh.h = 0.6; G_hh.n = 0.32
G_hh.I_ext = 10 * b2.uamp / b2.cm**2

M_hh = b2.StateMonitor(G_hh, ["V", "m", "h", "n"], record=True)
sp_hh = b2.SpikeMonitor(G_hh)
b2.run(100 * b2.ms)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(M_hh.t/b2.ms, M_hh.V[0]/b2.mV, 'k-', linewidth=1.2)
axes[0].set_ylabel("V (mV)"); axes[0].set_title("Hodgkin-Huxley Neuron")

for var, color in zip(["m", "h", "n"], ["b", "r", "g"]):
    axes[1].plot(M_hh.t/b2.ms, getattr(M_hh, var)[0], color=color, label=var, lw=1.2)
axes[1].set_xlabel("Time (ms)"); axes[1].set_ylabel("Gating variable")
axes[1].legend()
plt.tight_layout()
plt.savefig("hodgkin_huxley.png", dpi=150)
plt.show()
print(f"HH spikes: {sp_hh.num_spikes}")
```

---

## Troubleshooting

### Error: `BrianObjectException: Variable ... is not defined`

**Cause**: Namespace mismatch between equation string and Python scope.

**Fix**:
```python
# Pass all parameters explicitly in namespace dict
G = b2.NeuronGroup(N, eqs, namespace={"tau": tau, "Vr": Vr, "Vt": Vt})
```

### Issue: Simulation too slow for large networks

**Fix**:
```python
# Use C++ code generation (requires gcc)
b2.set_device("cpp_standalone", directory="output")
# ... define network ...
b2.run(...)
# Compile and run automatically
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| brian2 | 2.5, 2.6, 2.7   | Equation syntax stable since 2.4 |

---

## External Resources

### Official Documentation

- [Brian2 documentation](https://brian2.readthedocs.io/)
- [Brian2 tutorials](https://brian2.readthedocs.io/en/stable/resources/tutorials/index.html)

### Key Papers

- Stimberg, M. et al. (2019). *Brian 2, an intuitive and efficient neural simulator*. eLife.

---

## Examples

### Example 1: Gamma Oscillations in E-I Network

```python
# =============================================
# Gamma oscillation via E-I balance
# =============================================
import brian2 as b2
import numpy as np, matplotlib.pyplot as plt
from scipy.signal import welch

b2.start_scope()
b2.defaultclock.dt = 0.05 * b2.ms

N_E, N_I = 800, 200
eqs_simple = """
    dv/dt = (-v + I_net) / tau : 1
    I_net : 1
"""

P_E = b2.NeuronGroup(N_E, eqs_simple, threshold="v>1", reset="v=0",
                     method="euler", namespace={"tau": 10*b2.ms})
P_I = b2.NeuronGroup(N_I, eqs_simple, threshold="v>1", reset="v=0",
                     method="euler", namespace={"tau": 5*b2.ms})

P_E.v = "rand()"; P_I.v = "rand()"
P_E.I_net = 1.2; P_I.I_net = 1.0

S_EI = b2.Synapses(P_E, P_I, on_pre="I_net += 0.05", delay=1*b2.ms)
S_IE = b2.Synapses(P_I, P_E, on_pre="I_net -= 0.2", delay=2*b2.ms)
S_EI.connect(p=0.1); S_IE.connect(p=0.3)

rate_mon = b2.PopulationRateMonitor(P_E)
b2.run(500*b2.ms)

rate = rate_mon.smooth_rate("gaussian", width=5*b2.ms) / b2.Hz
t = rate_mon.t / b2.ms
f, Pxx = welch(rate[len(rate)//2:], fs=1000/0.1, nperseg=512)

peak_freq = f[np.argmax(Pxx)]
print(f"Peak oscillation frequency: {peak_freq:.1f} Hz")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(t, rate, 'b-', lw=0.8)
axes[0].set_xlabel("Time (ms)"); axes[0].set_ylabel("Rate (Hz)")
axes[0].set_title("Population Rate")

axes[1].semilogy(f[:100], Pxx[:100], 'r-')
axes[1].axvline(peak_freq, linestyle='--', label=f"Peak: {peak_freq:.0f} Hz")
axes[1].set_xlabel("Frequency (Hz)"); axes[1].set_ylabel("PSD")
axes[1].set_title("Power Spectrum"); axes[1].legend()

plt.tight_layout()
plt.savefig("gamma_oscillations.png", dpi=150)
plt.show()
```

**Interpreting these results**: Peak in the 30-80 Hz range indicates gamma-band synchrony, a hallmark of E-I balanced networks.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
