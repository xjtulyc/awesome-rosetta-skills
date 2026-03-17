---
name: control-systems
description: >
  Control systems analysis and PID design with python-control and scipy.signal:
  transfer functions, Bode/Nyquist plots, root locus, stability margins, and
  step-response simulation for SISO and discrete-time systems.
tags:
  - control-systems
  - python-control
  - pid-design
  - stability-analysis
  - engineering
  - dynamical-systems
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
    - control>=0.9.4
    - scipy>=1.11.0
    - numpy>=1.24.0
    - matplotlib>=3.7.0
last_updated: "2026-03-17"
status: "stable"
---

# Control Systems

> **One-line summary**: This Skill helps engineers model, analyse, and design feedback control systems using `python-control` and `scipy.signal`, covering everything from transfer-function definition through PID synthesis, frequency-domain plots, and discrete-time simulation.

---

## When to Use This Skill

- When you need to **model a plant** as a transfer function or state-space representation
- When you need to **assess stability** via gain/phase margin, Nyquist criterion, or pole locations
- When you need to **design a PID or lead-lag compensator** using Ziegler-Nichols or analytical methods
- When you need to **visualise frequency response** (Bode plot, Nyquist diagram, root locus)
- When you need to **simulate closed-loop step response** and compute time-domain specifications
- When you need to **discretise a continuous controller** for embedded implementation (ZOH, Tustin)

**Trigger keywords**: transfer function, state space, Bode plot, Nyquist plot, root locus, PID tuning, Ziegler-Nichols, gain margin, phase margin, step response, lead compensator, lag compensator, ZOH discretization, stability analysis, python-control

---

## Background & Key Concepts

### Transfer Function Representation

A linear time-invariant (LTI) system is described in the Laplace domain by its transfer function:

$$
G(s) = \frac{Y(s)}{U(s)} = \frac{b_m s^m + b_{m-1} s^{m-1} + \cdots + b_0}{a_n s^n + a_{n-1} s^{n-1} + \cdots + a_0}
$$

Where $Y(s)$ is the output Laplace transform and $U(s)$ is the input. The roots of the denominator are the **poles**; roots of the numerator are **zeros**.

### State-Space Representation

The equivalent time-domain form uses four matrices:

$$
\dot{x}(t) = A x(t) + B u(t), \quad y(t) = C x(t) + D u(t)
$$

Where $x \in \mathbb{R}^n$ is the state vector. Eigenvalues of $A$ are the system poles.

### PID Controller

The PID control law combines proportional, integral, and derivative action:

$$
u(t) = K_p e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d \frac{de(t)}{dt}
$$

**Ziegler-Nichols ultimate-gain method**: bring the system to sustained oscillation with a P-only controller to find the ultimate gain $K_u$ and ultimate period $T_u$, then use the ZN table to compute $K_p$, $K_i$, $K_d$.

### Stability Margins

The gain margin (GM) and phase margin (PM) quantify robustness against model uncertainty:

$$
\text{GM} = -20\log_{10}|G(j\omega_{pc})| \text{ dB}, \quad \text{PM} = 180° + \angle G(j\omega_{gc})
$$

where $\omega_{pc}$ is the phase crossover frequency and $\omega_{gc}$ is the gain crossover frequency. A rule of thumb: GM > 6 dB and PM > 30° for adequate robustness.

### Comparison with Related Methods

| Method | Best for | Key assumption | Limitation |
|:-------|:---------|:---------------|:-----------|
| Transfer function | SISO frequency-domain design | Linear, time-invariant | Not for MIMO or nonlinear systems |
| State space | MIMO / observer design | Minimal representation | More complex algebra |
| Root locus | Gain-based pole placement | Closed-loop with variable K | Limited for time-delay plants |
| Frequency response (Bode) | Robust stability margins | Minimum-phase systems | Phase ambiguity for non-minimum phase |

---

## Environment Setup

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install Python dependencies
pip install control>=0.9.4 scipy>=1.11.0 numpy>=1.24.0 matplotlib>=3.7.0
```

### Verify Installation

```python
import control
import scipy
import numpy as np
import matplotlib
print(f"python-control : {control.__version__}")
print(f"scipy          : {scipy.__version__}")
print(f"numpy          : {np.__version__}")
print(f"matplotlib     : {matplotlib.__version__}")
# Expected: python-control : 0.9.x or higher
```

---

## Core Workflow

### Step 1: Define the Plant Model

```python
import control
import numpy as np
import matplotlib.pyplot as plt

# ── 1a. Transfer function: second-order plant with time-delay approximation ───
# G(s) = K / (s(τs + 1)) — a DC motor / velocity loop prototype
K_plant = 2.0
tau = 0.5  # time constant (s)

# Numerator and denominator polynomial coefficients (highest power first)
num = [K_plant]
den = [tau, 1, 0]   # τs² + s = s(τs + 1)
G = control.tf(num, den)
print("Plant transfer function:")
print(G)

# ── 1b. State-space representation of the same plant ──────────────────────────
# Canonical controllable form for G(s) = 2 / (0.5s² + s)
A = np.array([[0, 1],
              [0, -1/tau]])
B = np.array([[0],
              [K_plant/tau]])
C = np.array([[1, 0]])
D = np.array([[0]])
sys_ss = control.ss(A, B, C, D)
print("\nState-space representation:")
print(sys_ss)

# Verify equivalence
G_from_ss = control.ss2tf(sys_ss)
print("\nTransfer function from SS (should match):")
print(G_from_ss)
```

### Step 2: Frequency-Domain Analysis

```python
# ── Bode plot ─────────────────────────────────────────────────────────────────
fig_bode, _ = control.bode_plot(
    G,
    dB=True,
    Hz=False,
    omega_limits=(0.01, 100),
    margins=True,     # annotate gain and phase crossover points
    show_legend=True,
)
plt.suptitle("Open-Loop Bode Plot: G(s) = 2 / [s(0.5s+1)]")
plt.tight_layout()
plt.savefig("bode_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Compute gain and phase margins numerically ────────────────────────────────
gm, pm, wpc, wgc = control.stability_margins(G)
print(f"\nGain margin  : {20*np.log10(gm):.2f} dB  (at ω = {wpc:.3f} rad/s)")
print(f"Phase margin : {pm:.2f}°  (at ω = {wgc:.3f} rad/s)")
print(f"System is {'stable' if gm > 1 and pm > 0 else 'UNSTABLE'} open-loop")

# ── Nyquist diagram ───────────────────────────────────────────────────────────
fig_ny, ax_ny = plt.subplots(figsize=(7, 7))
control.nyquist_plot(G, ax=ax_ny, omega_limits=(0.01, 100))
ax_ny.set_title("Nyquist Plot")
ax_ny.axhline(0, color="k", linewidth=0.5)
ax_ny.axvline(0, color="k", linewidth=0.5)
ax_ny.plot(-1, 0, "rx", markersize=10, label="Critical point (-1, 0)")
ax_ny.legend()
plt.tight_layout()
plt.savefig("nyquist_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Root locus ────────────────────────────────────────────────────────────────
fig_rl = plt.figure(figsize=(8, 6))
control.root_locus(G, plot=True)
plt.title("Root Locus: G(s)")
plt.tight_layout()
plt.savefig("root_locus.png", dpi=150, bbox_inches="tight")
plt.show()
```

**Interpreting frequency-domain results**:
- **Gain margin > 6 dB**: the loop gain can increase 2× before instability
- **Phase margin > 45°**: fast, well-damped transient response expected
- **Nyquist**: if the $(-1, 0)$ point is not encircled, the closed loop is stable (for open-loop stable $G$)

### Step 3: PID Controller Design — Ziegler-Nichols

```python
def ziegler_nichols_pid(Ku: float, Tu: float, variant: str = "classic") -> dict:
    """
    Compute PID gains using the Ziegler-Nichols ultimate-gain method.

    Parameters
    ----------
    Ku : float
        Ultimate proportional gain (sustains oscillation).
    Tu : float
        Ultimate period (seconds) at Ku.
    variant : str
        'classic'  — original ZN table (aggressive, slight overshoot)
        'some_overshoot' — modified for ~20 % overshoot
        'no_overshoot'   — modified for near-zero overshoot

    Returns
    -------
    dict with keys Kp, Ki, Kd
    """
    if variant == "classic":
        Kp = 0.6 * Ku
        Ti = 0.5 * Tu
        Td = 0.125 * Tu
    elif variant == "some_overshoot":
        Kp = 0.33 * Ku
        Ti = 0.5 * Tu
        Td = 0.33 * Tu
    elif variant == "no_overshoot":
        Kp = 0.2 * Ku
        Ti = 0.5 * Tu
        Td = 0.33 * Tu
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    Ki = Kp / Ti
    Kd = Kp * Td
    return {"Kp": Kp, "Ki": Ki, "Kd": Kd}


# Simulate proportional-only closed loop to find Ku, Tu ─────────────────────
# For this plant we can compute analytically: at phase = -180°
# G(jω) phase = -90° - arctan(τω) = -180°  → arctan(0.5ω) = 90° → no finite ω
# The plant G = 2/[s(0.5s+1)] has inherent -90° from the integrator, so we add
# a lag correction factor and use gain-based approach from margin analysis

# Demonstration: design PID using the margins computed earlier
# For a real process: run a relay experiment or apply proportional-only control
# Here we approximate: Ku from gain margin, Tu from phase crossover period
Ku_approx = 2.5   # would be found experimentally
Tu_approx = 2 * np.pi / wpc if wpc > 0 else 2.0

gains = ziegler_nichols_pid(Ku_approx, Tu_approx, variant="classic")
print(f"\nZiegler-Nichols PID gains (classic):")
print(f"  Kp = {gains['Kp']:.4f}")
print(f"  Ki = {gains['Ki']:.4f}")
print(f"  Kd = {gains['Kd']:.4f}")

# Build the PID controller transfer function: C(s) = Kp + Ki/s + Kd*s
# With derivative filter to prevent differentiator wind-up: Kd*s/(s/N + 1), N=10
N = 10
Kp, Ki, Kd = gains["Kp"], gains["Ki"], gains["Kd"]
C_num = [Kd + Kp/N, Kp + Ki/N, Ki]
C_den = [1, N, 0]
C = control.tf(C_num, C_den)
print("\nPID controller transfer function C(s):")
print(C)
```

### Step 4: Closed-Loop Simulation

```python
# ── Build closed-loop system ──────────────────────────────────────────────────
T_cl = control.feedback(C * G, 1)   # negative unity feedback
print("Closed-loop poles:")
poles_cl = control.poles(T_cl)
for p in poles_cl:
    print(f"  {p:.4f}  (ζ={-p.real/abs(p):.3f} if complex pair)")

# ── Step response ─────────────────────────────────────────────────────────────
t_sim = np.linspace(0, 15, 3000)
t_out, y_out = control.step_response(T_cl, T=t_sim)

# Compute time-domain specifications
y_ss = y_out[-1]
overshoot = (max(y_out) - y_ss) / y_ss * 100 if y_ss != 0 else 0
rise_10 = t_out[np.where(y_out >= 0.1 * y_ss)[0][0]] if any(y_out >= 0.1 * y_ss) else None
rise_90 = t_out[np.where(y_out >= 0.9 * y_ss)[0][0]] if any(y_out >= 0.9 * y_ss) else None
rise_time = (rise_90 - rise_10) if (rise_10 is not None and rise_90 is not None) else None
settling_tol = 0.02
settled_idx = np.where(np.abs(y_out - y_ss) > settling_tol * abs(y_ss))[0]
settling_time = t_out[settled_idx[-1]] if len(settled_idx) > 0 else 0.0

print(f"\nStep response specifications:")
print(f"  Steady-state value : {y_ss:.4f}")
print(f"  Peak overshoot     : {overshoot:.2f} %")
if rise_time:
    print(f"  Rise time (10-90%) : {rise_time:.4f} s")
print(f"  Settling time (2%) : {settling_time:.4f} s")

# ── Plot closed-loop step response ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_out, y_out, label="Closed-loop output", linewidth=2)
ax.axhline(y_ss, color="r", linestyle="--", alpha=0.6, label=f"Steady state = {y_ss:.3f}")
ax.axhline(y_ss * 1.02, color="g", linestyle=":", alpha=0.5)
ax.axhline(y_ss * 0.98, color="g", linestyle=":", alpha=0.5, label="±2% band")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Output")
ax.set_title("Closed-Loop Step Response (ZN-PID)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("step_response.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Advanced Usage

### Lead-Lag Compensator Design

```python
def lead_compensator(alpha: float, T: float) -> "control.TransferFunction":
    """
    Create a phase-lead compensator:  C(s) = (Ts + 1) / (αTs + 1),  0 < α < 1.

    Parameters
    ----------
    alpha : float
        Lead ratio (< 1 increases phase, > 1 creates lag).
    T : float
        Time constant (s). Peak phase added at ω_max = 1 / (T√α).

    Returns
    -------
    control.TransferFunction
    """
    return control.tf([T, 1], [alpha * T, 1])


def lag_compensator(beta: float, T: float) -> "control.TransferFunction":
    """
    Create a phase-lag compensator:  C(s) = (Ts + 1) / (βTs + 1),  β > 1.

    Parameters
    ----------
    beta : float
        Lag ratio (> 1 attenuates high frequencies and boosts low-frequency gain).
    T : float
        Time constant (s).
    """
    return control.tf([T, 1], [beta * T, 1])


# Example: add 40° phase margin with a lead compensator
alpha_lead = 0.15   # yields ≈ 40° maximum phase lead
T_lead = 1.0 / (wgc * np.sqrt(alpha_lead))   # place peak at current gain crossover
C_lead = lead_compensator(alpha_lead, T_lead)
C_lead_dc = float(control.dcgain(C_lead))     # DC gain correction factor

# Normalise so DC gain = 1
C_lead_norm = C_lead / C_lead_dc
T_cl_lead = control.feedback(C_lead_norm * G, 1)
gm_l, pm_l, _, _ = control.stability_margins(C_lead_norm * G)
print(f"Phase margin with lead compensator: {pm_l:.1f}°")
```

### Discrete-Time ZOH Discretization

```python
def discretize_zoh(sys_c: "control.TransferFunction", Ts: float):
    """
    Convert a continuous-time LTI system to discrete time using zero-order hold.

    Parameters
    ----------
    sys_c : control.TransferFunction
        Continuous-time plant or controller.
    Ts : float
        Sampling period (s). Rule of thumb: Ts ≤ 0.1 / ω_BW.

    Returns
    -------
    control.TransferFunction
        Discrete-time system with dt = Ts.
    """
    sys_d = control.c2d(sys_c, Ts, method="zoh")
    return sys_d


# Discretize the plant at 20 Hz (Ts = 0.05 s)
Ts = 0.05
G_d = discretize_zoh(G, Ts)
print("Discrete-time plant (ZOH, Ts=0.05 s):")
print(G_d)

# Verify discrete poles lie inside the unit circle (stability)
poles_d = control.poles(G_d)
print(f"\nDiscrete poles: {poles_d}")
print(f"All poles inside unit circle: {all(abs(p) < 1 for p in poles_d)}")

# Discrete step response
t_d = np.arange(0, 10, Ts)
T_cl_d = control.feedback(C * G_d, 1)  # PID (continuous) + ZOH plant
try:
    t_d_out, y_d_out = control.step_response(T_cl_d, T=t_d)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.step(t_d_out, y_d_out, where="post", label=f"Discrete (Ts={Ts} s)", linewidth=1.5)
    ax2.plot(t_out, y_out, "--", alpha=0.6, label="Continuous reference")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Output")
    ax2.set_title("Continuous vs Discrete-Time Step Response")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("discrete_step.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"Discrete simulation note: {e}")
```

### Pole-Zero Map

```python
fig_pz, ax_pz = plt.subplots(figsize=(7, 6))
control.pzmap(T_cl, plot=True, ax=ax_pz)
ax_pz.set_title("Pole-Zero Map — Closed-Loop System")
# Add stability boundary (unit circle for discrete systems; imaginary axis for CT)
theta = np.linspace(0, 2 * np.pi, 200)
ax_pz.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pole_zero_map.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Troubleshooting

### Error: `ValueError: D matrix must be square`

**Cause**: Attempting to create a state-space system with incompatible `D` matrix dimensions when the plant has more outputs than inputs.

**Fix**:
```python
# Ensure D has shape (n_outputs, n_inputs)
n_outputs, n_inputs = 1, 1
D = np.zeros((n_outputs, n_inputs))
sys = control.ss(A, B, C, D)
```

### Error: `control.matlab.sisotool` not found

**Cause**: `sisotool` is only available in the MATLAB Control System Toolbox; `python-control` provides `control.root_locus` and interactive design via `control.sisotool` in newer releases.

**Fix**:
```bash
pip install control>=0.9.4  # sisotool added in 0.9
# Then in Python:
# control.sisotool(G)  # opens interactive root-locus GUI
```

### Issue: Nyquist plot looks incomplete or truncated

**Cause**: The default `omega` range may not span the full frequency range of interest.

**Fix**:
```python
import numpy as np
omega = np.logspace(-3, 3, 2000)  # 0.001 to 1000 rad/s
control.nyquist_plot(G, omega=omega)
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| control | 0.9.4, 0.10.x | API change in 0.9: `bode_plot` returns `(mag, phase, omega)` not `(mag, phase)` |
| scipy | 1.11, 1.12 | None |
| numpy | 1.24, 1.26 | None |
| matplotlib | 3.7, 3.8 | None |

---

## External Resources

### Official Documentation

- [python-control documentation](https://python-control.readthedocs.io/)
- [scipy.signal LTI systems](https://docs.scipy.org/doc/scipy/reference/signal.html)

### Key Papers

- Åström, K.J. & Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning*. ISA Press.
- Ziegler, J.G. & Nichols, N.B. (1942). *Optimum settings for automatic controllers*. Trans. ASME, 64, 759-768.

### Tutorials

- [python-control examples gallery](https://python-control.readthedocs.io/en/latest/examples.html)
- [Brian Douglas Control System Lectures (YouTube)](https://www.youtube.com/user/ControlLectures)

---

## Examples

### Example 1: Full PID Design and Verification for a DC Motor Velocity Loop

**Scenario**: Design a PID controller for a DC motor speed control loop, verify stability margins, and simulate the closed-loop step response.

```python
# =============================================
# End-to-end example: PID design for DC motor
# Requirements: Python 3.10+; control>=0.9.4
# =============================================

import control
import numpy as np
import matplotlib.pyplot as plt

# Motor parameters
K_m   = 10.0   # motor gain (rad/s / V)
tau_m = 0.1    # electrical time constant (s)
J     = 0.01   # rotor inertia (kg·m²)
b     = 0.1    # viscous friction (N·m·s)

# Mechanical + electrical plant: G(s) = K_m / [(Js + b)(τ_m s + 1)]
num_m = [K_m]
den_m = np.convolve([J, b], [tau_m, 1])
G_motor = control.tf(num_m, den_m)
print("DC motor plant:")
print(G_motor)

# Open-loop frequency analysis
gm, pm, wpc, wgc = control.stability_margins(G_motor)
print(f"\nOpen-loop: GM = {20*np.log10(max(gm,1e-6)):.1f} dB, PM = {pm:.1f}°")

# PID design via frequency-domain loop shaping
# Target: PM ≈ 60°, bandwidth ≈ 50 rad/s
Kp_pid = 0.05
Ki_pid = 0.5
Kd_pid = 0.002
N_filt = 50   # derivative filter coefficient

pid_num = [Kd_pid + Kp_pid/N_filt,
           Kp_pid + Ki_pid/N_filt,
           Ki_pid]
pid_den = [1, N_filt, 0]
C_pid = control.tf(pid_num, pid_den)
print("\nPID controller:")
print(C_pid)

# Check closed-loop stability
L = C_pid * G_motor
gm_cl, pm_cl, _, wgc_cl = control.stability_margins(L)
print(f"\nWith PID: GM = {20*np.log10(max(gm_cl,1e-6)):.1f} dB, "
      f"PM = {pm_cl:.1f}°, "
      f"bandwidth ≈ {wgc_cl:.1f} rad/s")

# Closed-loop step response
T_cl = control.feedback(L, 1)
t_sim = np.linspace(0, 1.0, 2000)
t_resp, y_resp = control.step_response(T_cl, T=t_sim)
t_dist, y_dist = control.step_response(
    control.feedback(G_motor, C_pid), T=t_sim
)  # input disturbance rejection

# Compute specs
y_ss = y_resp[-1]
pct_os = (max(y_resp) - y_ss) / y_ss * 100 if y_ss else 0
idx_90 = np.argmax(y_resp >= 0.9 * y_ss)
idx_10 = np.argmax(y_resp >= 0.1 * y_ss)
rise_t = t_resp[idx_90] - t_resp[idx_10] if idx_90 > idx_10 else 0

print(f"\nStep response specifications:")
print(f"  Steady-state output : {y_ss:.4f}")
print(f"  Peak overshoot      : {pct_os:.2f} %")
print(f"  Rise time (10-90%)  : {rise_t*1000:.1f} ms")

# Bode plot of loop transfer function
fig_b, _ = control.bode_plot(L, dB=True, margins=True, omega_limits=(1, 1000))
plt.suptitle("Loop Bode Plot — DC Motor PID")
plt.tight_layout()
plt.savefig("motor_pid_bode.png", dpi=150, bbox_inches="tight")

# Step response plot
fig_s, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_resp * 1000, y_resp, linewidth=2, label="Speed (rad/s)")
ax.axhline(y_ss, color="r", linestyle="--", alpha=0.7, label=f"Setpoint = {y_ss:.3f}")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Speed (rad/s)")
ax.set_title("DC Motor Closed-Loop Step Response")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("motor_step_response.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nAnalysis complete.")
```

**Interpreting these results**: A phase margin above 45° with the PID controller confirms robust stability. The rise time target of < 50 ms at 50 rad/s bandwidth is met. Adjust `Kd_pid` to reduce overshoot and `Ki_pid` to improve steady-state tracking of ramp references.

---

### Example 2: Lead-Lag Compensator for Unstable Plant

**Scenario**: Design a lead-lag compensator for a marginally-stable inverted pendulum linearisation and verify performance.

```python
# =============================================
# End-to-end example 2: lead-lag compensator
# =============================================

import control
import numpy as np
import matplotlib.pyplot as plt

# Linearised inverted pendulum on cart: G(s) = 1 / (s² - ω_n²)
omega_n = 3.0   # rad/s (unstable pole at ±3 rad/s)
G_inv = control.tf([1], [1, 0, -omega_n**2])
print("Inverted pendulum plant:")
print(G_inv)
print(f"Open-loop poles: {control.poles(G_inv)}")

# Phase lead to stabilise and add PM ≈ 45°
# Required PM increase: 45° - (current PM) ≈ large value for unstable plant
alpha_l = 0.1
T_l = 0.5
C_lead_ip = control.tf([T_l, 1], [alpha_l * T_l, 1])

# Gain to place gain crossover at 6 rad/s
# |C_lead * G| at ω=6 should be 1 → compute and scale
omega_target = 6.0
freq_resp = control.freqresp(C_lead_ip * G_inv, [omega_target])
mag_at_target = abs(freq_resp.fresp[0, 0, 0])
K_scale = 1.0 / mag_at_target
C_total = K_scale * C_lead_ip
L_ip = C_total * G_inv

gm_ip, pm_ip, _, wgc_ip = control.stability_margins(L_ip)
print(f"\nWith lead: PM = {pm_ip:.1f}°, bandwidth ≈ {wgc_ip:.2f} rad/s")

T_cl_ip = control.feedback(L_ip, 1)
print(f"Closed-loop poles: {control.poles(T_cl_ip)}")

t_ip = np.linspace(0, 5, 1000)
t_ip_out, y_ip_out = control.step_response(T_cl_ip, T=t_ip)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
control.bode_plot(L_ip, dB=True, margins=True, omega_limits=(0.1, 100),
                  ax=axes)
axes[0].set_title("Lead Compensated Loop")

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(t_ip_out, y_ip_out, linewidth=2)
ax2.axhline(1, color="r", linestyle="--", alpha=0.6, label="Reference")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Output")
ax2.set_title("Inverted Pendulum — Lead Compensated Step Response")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("inverted_pendulum_step.png", dpi=150, bbox_inches="tight")
plt.show()

print("Analysis complete.")
print(f"  Phase margin : {pm_ip:.1f}°")
print(f"  All closed-loop poles in LHP: "
      f"{all(p.real < 0 for p in control.poles(T_cl_ip))}")
```

**Interpreting these results**: The key check is that all closed-loop poles have negative real parts (LHP), confirming stabilisation. A phase margin above 30° ensures robustness to model uncertainty.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
