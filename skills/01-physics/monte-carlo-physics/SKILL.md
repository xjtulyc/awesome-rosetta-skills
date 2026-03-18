---
name: monte-carlo-physics
description: >
  Use this Skill when simulating physical systems with Monte Carlo methods:
  statistical integration, MCMC sampling, Ising model, error estimation.
tags:
  - physics
  - monte-carlo
  - statistical-mechanics
  - numpy
  - numba
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
    - numba>=0.57
    - matplotlib>=3.7
    - pandas>=2.0
last_updated: "2026-03-17"
status: "stable"
---

# Monte Carlo Methods for Physics

> **One-line summary**: Apply Monte Carlo integration, MCMC sampling, and statistical mechanics simulations using numpy, scipy, and numba for physics research.

---

## When to Use This Skill

- When computing high-dimensional integrals analytically intractable (partition functions, path integrals)
- When sampling from complex probability distributions (Boltzmann, posterior distributions)
- When simulating statistical mechanical systems (Ising model, lattice gauge theory)
- When estimating uncertainties via bootstrap or error propagation
- When implementing Markov Chain Monte Carlo (Metropolis-Hastings, Gibbs sampling)
- When studying phase transitions and critical phenomena

**Trigger keywords**: Monte Carlo integration, MCMC, Metropolis-Hastings, Ising model, importance sampling, statistical mechanics, partition function, random sampling

---

## Background & Key Concepts

### Monte Carlo Integration

Monte Carlo integration estimates an integral by random sampling:

$$
\langle f \rangle = \int f(\mathbf{x}) p(\mathbf{x}) d\mathbf{x} \approx \frac{1}{N} \sum_{i=1}^{N} f(\mathbf{x}_i)
$$

where samples $\mathbf{x}_i \sim p(\mathbf{x})$. The statistical error scales as $1/\sqrt{N}$ regardless of dimensionality — Monte Carlo's key advantage over quadrature rules in high dimensions.

### Importance Sampling

For efficiency, sample from a proposal distribution $q(\mathbf{x})$ and reweight:

$$
\langle f \rangle_p = \frac{\langle f \cdot w \rangle_q}{\langle w \rangle_q}, \quad w(\mathbf{x}) = \frac{p(\mathbf{x})}{q(\mathbf{x})}
$$

Choose $q$ to be large where $|f \cdot p|$ is large.

### Markov Chain Monte Carlo

MCMC constructs a Markov chain whose stationary distribution is the target $p(\mathbf{x})$. The Metropolis-Hastings acceptance probability:

$$
A(x' | x) = \min\left(1, \frac{p(x') q(x | x')}{p(x) q(x' | x)}\right)
$$

### Ising Model

The 2D Ising model Hamiltonian:

$$
H = -J \sum_{\langle i,j \rangle} s_i s_j - h \sum_i s_i
$$

where $s_i \in \{-1, +1\}$ are spins, $J$ is the coupling constant, and $h$ is external field. The critical temperature is $T_c = 2J / k_B \ln(1 + \sqrt{2}) \approx 2.269 J/k_B$.

### Comparison with Related Methods

| Method | Best for | Key assumption | Limitation |
|:-------|:---------|:---------------|:-----------|
| Monte Carlo integration | High-dim integrals | Random samples available | Slow convergence $\propto 1/\sqrt{N}$ |
| MCMC | Complex posteriors | Ergodic Markov chain | Autocorrelation, burn-in needed |
| Ising Metropolis | Lattice spin systems | Local updates | Critical slowing down near $T_c$ |
| Importance sampling | Rare event estimation | Good proposal $q$ available | Exponential variance in high dim |

---

## Environment Setup

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install numpy>=1.24 scipy>=1.11 numba>=0.57 matplotlib>=3.7 pandas>=2.0
```

### Verify Installation

```python
import numpy as np
import scipy
import numba
import matplotlib

print(f"numpy:      {np.__version__}")
print(f"scipy:      {scipy.__version__}")
print(f"numba:      {numba.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
# Expected: numpy 1.24+, scipy 1.11+, numba 0.57+
```

---

## Core Workflow

### Step 1: Monte Carlo Integration

Estimate $\pi$ using the unit circle as a pedagogical example, then compute a physics integral.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- Example 1: Estimate π ---
N = 1_000_000
x, y = rng.uniform(-1, 1, size=(2, N))
inside = (x**2 + y**2) <= 1.0
pi_estimate = 4 * inside.mean()
pi_error = 4 * inside.std() / np.sqrt(N)
print(f"π estimate: {pi_estimate:.6f} ± {pi_error:.6f}  (true: {np.pi:.6f})")

# --- Example 2: 3D integral (partition function-style) ---
# ∫ exp(-x²-y²-z²) dx dy dz over [-3,3]³ (≈ (√π)³ ≈ 5.568)
N = 500_000
vol = 6**3
pts = rng.uniform(-3, 3, size=(N, 3))
f_vals = np.exp(-np.sum(pts**2, axis=1))
integral = vol * f_vals.mean()
integral_err = vol * f_vals.std() / np.sqrt(N)
print(f"3D Gaussian integral: {integral:.4f} ± {integral_err:.4f}  (exact: {np.pi**1.5:.4f})")
```

### Step 2: Metropolis-Hastings MCMC

```python
import numpy as np
import matplotlib.pyplot as plt

def log_target(x, mu=0.0, sigma=1.0):
    """Log of a Gaussian target — replace with your log-posterior."""
    return -0.5 * ((x - mu) / sigma)**2

def metropolis_hastings(log_target_fn, x0, n_steps, proposal_std=0.5, rng=None):
    """
    Metropolis-Hastings sampler.
    Returns chain of shape (n_steps,).
    """
    if rng is None:
        rng = np.random.default_rng()
    chain = np.empty(n_steps)
    x_cur = x0
    log_p_cur = log_target_fn(x_cur)
    n_accept = 0

    for i in range(n_steps):
        x_prop = x_cur + rng.normal(0, proposal_std)
        log_p_prop = log_target_fn(x_prop)
        log_alpha = log_p_prop - log_p_cur
        if np.log(rng.uniform()) < log_alpha:
            x_cur, log_p_cur = x_prop, log_p_prop
            n_accept += 1
        chain[i] = x_cur

    print(f"Acceptance rate: {n_accept / n_steps:.3f}  (target: 0.23–0.50)")
    return chain

rng = np.random.default_rng(42)
chain = metropolis_hastings(log_target, x0=0.0, n_steps=50_000, proposal_std=1.0, rng=rng)

# Diagnostics
burn_in = 5_000
samples = chain[burn_in:]
print(f"Sample mean:   {samples.mean():.4f}  (true: 0.0)")
print(f"Sample std:    {samples.std():.4f}  (true: 1.0)")

# Autocorrelation time (effective sample size)
from scipy.signal import correlate
def autocorr_time(chain):
    chain_centered = chain - chain.mean()
    acf = correlate(chain_centered, chain_centered, mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]
    tau = 1 + 2 * np.sum(acf[1:][acf[1:] > 0])
    return tau

tau = autocorr_time(samples)
ess = len(samples) / tau
print(f"Autocorr time: {tau:.1f}  ESS: {ess:.0f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(chain[:500], lw=0.5)
axes[0].axvline(burn_in, color='r', linestyle='--', label='burn-in end')
axes[0].set_title("MCMC Trace")
axes[0].set_xlabel("Step")

axes[1].hist(samples, bins=60, density=True, alpha=0.7, label="MCMC samples")
x_plot = np.linspace(-4, 4, 300)
axes[1].plot(x_plot, np.exp(-0.5*x_plot**2)/np.sqrt(2*np.pi), 'r-', label="True N(0,1)")
axes[1].legend()
axes[1].set_title("Sample Distribution")
plt.tight_layout()
plt.savefig("mcmc_diagnostics.png", dpi=150)
plt.show()
```

### Step 3: Ising Model Simulation

```python
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def ising_step(spins, J, T, rng_state):
    """
    Single Metropolis sweep of the 2D Ising lattice.
    Returns updated spins and delta energy.
    """
    N = spins.shape[0]
    for _ in range(N * N):
        i = int(np.random.rand() * N)
        j = int(np.random.rand() * N)
        # Sum of neighbors (periodic boundary)
        neighbors = (spins[(i+1) % N, j] + spins[(i-1) % N, j] +
                     spins[i, (j+1) % N] + spins[i, (j-1) % N])
        dE = 2.0 * J * spins[i, j] * neighbors
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1
    return spins

def simulate_ising(N=32, J=1.0, T=2.5, n_steps=2000, n_burn=500, seed=42):
    """
    Simulate 2D Ising model via Metropolis algorithm.
    Returns (magnetization_history, energy_history, final_spins).
    """
    np.random.seed(seed)
    spins = np.random.choice([-1, 1], size=(N, N)).astype(np.float64)

    mag_history, energy_history = [], []

    for step in range(n_steps):
        spins = ising_step(spins, J, T, 0)
        if step >= n_burn:
            # Magnetization per spin
            mag_history.append(np.abs(spins.mean()))
            # Energy per spin
            E = -J * (
                np.sum(spins * np.roll(spins, 1, axis=0)) +
                np.sum(spins * np.roll(spins, 1, axis=1))
            )
            energy_history.append(E / (N * N))

    return np.array(mag_history), np.array(energy_history), spins

# Simulate near critical temperature Tc ≈ 2.269
T_c = 2.0 / np.log(1 + np.sqrt(2))
mag, energy, final_spins = simulate_ising(N=32, T=T_c, n_steps=3000)

print(f"T/Tc = {T_c/T_c:.3f}")
print(f"Mean |m| = {mag.mean():.4f} ± {mag.std():.4f}")
print(f"Mean E/site = {energy.mean():.4f} ± {energy.std():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(final_spins, cmap='RdBu', vmin=-1, vmax=1)
axes[0].set_title(f"Ising lattice at T/Tc=1.00")
axes[0].axis('off')

axes[1].plot(mag)
axes[1].set_xlabel("MC sweep")
axes[1].set_ylabel("|m|")
axes[1].set_title("Magnetization")

axes[2].plot(energy)
axes[2].set_xlabel("MC sweep")
axes[2].set_ylabel("E/N²")
axes[2].set_title("Energy per site")

plt.tight_layout()
plt.savefig("ising_simulation.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Phase Transition Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

def measure_observables(N, J, temperatures, n_steps=3000, n_burn=1000):
    """Sweep temperatures and compute mean |m|, susceptibility, specific heat."""
    results = []
    for T in temperatures:
        mag, energy, _ = simulate_ising(N=N, J=J, T=T,
                                        n_steps=n_steps, n_burn=n_burn)
        chi = N**2 * (np.mean(mag**2) - np.mean(mag)**2) / T  # susceptibility
        Cv  = N**2 * (np.mean(energy**2) - np.mean(energy)**2) / T**2  # heat cap
        results.append({
            "T": T, "m": mag.mean(), "chi": chi, "Cv": Cv
        })
    return results

temperatures = np.linspace(1.5, 3.5, 20)
T_c = 2.0 / np.log(1 + np.sqrt(2))

results = measure_observables(N=16, J=1.0, temperatures=temperatures)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
T_vals = [r["T"] for r in results]
axes[0].plot(T_vals, [r["m"] for r in results], 'o-')
axes[0].axvline(T_c, color='r', linestyle='--', label=f'$T_c$={T_c:.3f}')
axes[0].set_xlabel("T"); axes[0].set_ylabel("|m|"); axes[0].legend()

axes[1].plot(T_vals, [r["chi"] for r in results], 's-')
axes[1].axvline(T_c, color='r', linestyle='--')
axes[1].set_xlabel("T"); axes[1].set_ylabel("χ (susceptibility)")

axes[2].plot(T_vals, [r["Cv"] for r in results], '^-')
axes[2].axvline(T_c, color='r', linestyle='--')
axes[2].set_xlabel("T"); axes[2].set_ylabel("$C_V$")

plt.tight_layout()
plt.savefig("ising_phase_transition.png", dpi=150)
plt.show()
```

### Importance Sampling for Rare Events

```python
import numpy as np
from scipy import stats

def importance_sampling(target_log_pdf, proposal_log_pdf, proposal_sampler,
                        f_func, n_samples=100_000, seed=42):
    """
    Estimate E_p[f(x)] using importance sampling from proposal q.
    """
    rng = np.random.default_rng(seed)
    samples = proposal_sampler(n_samples, rng)
    log_weights = target_log_pdf(samples) - proposal_log_pdf(samples)
    # Stabilize: subtract max to avoid overflow
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()

    estimate = np.sum(weights * f_func(samples))
    # Effective sample size
    ess = 1.0 / np.sum(weights**2)
    return estimate, ess

# Example: estimate P(X > 5) where X ~ N(0,1)
# Use N(6,1) as proposal
target   = lambda x: stats.norm(0, 1).logpdf(x)
proposal = lambda x: stats.norm(6, 1).logpdf(x)
sampler  = lambda n, rng: rng.normal(6, 1, n)
indicator = lambda x: (x > 5).astype(float)

est, ess = importance_sampling(target, proposal, sampler, indicator)
true_val = 1 - stats.norm.cdf(5)
print(f"P(X>5) estimate: {est:.2e}  (true: {true_val:.2e})")
print(f"Effective sample size: {ess:.0f}")
```

---

## Troubleshooting

### Error: Numba JIT compilation slow on first call

**Cause**: numba compiles to machine code on first invocation.

**Fix**: Add `@njit(cache=True)` so compilation is cached to disk:
```python
from numba import njit

@njit(cache=True)
def ising_step(spins, J, T, seed):
    ...
```

### Issue: MCMC chain doesn't mix (all samples the same)

**Cause**: Proposal standard deviation too small or target very multimodal.

**Fix**:
```python
# Tune proposal_std so acceptance rate ≈ 23–50%
for std in [0.1, 0.5, 1.0, 2.0]:
    chain = metropolis_hastings(log_target, x0=0.0, n_steps=5000, proposal_std=std)
    # acceptance rate is printed inside the function
```

### Issue: Ising simulation crashes with memory error

**Cause**: Lattice size too large for eager storage.

**Fix**: Use `N=64` maximum for in-memory simulations; store only summary statistics.

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| numpy   | 1.24, 1.26, 2.0 | None |
| numba   | 0.57, 0.59      | Older numba may not support `@njit(cache=True)` |
| scipy   | 1.11, 1.13      | None |

---

## External Resources

### Official Documentation

- [NumPy random generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [Numba @njit documentation](https://numba.readthedocs.io/en/stable/user/jit.html)
- [SciPy stats distributions](https://docs.scipy.org/doc/scipy/reference/stats.html)

### Key Papers

- Metropolis, N. et al. (1953). *Equation of State Calculations by Fast Computing Machines*. J. Chem. Phys.
- Hastings, W.K. (1970). *Monte Carlo Sampling Methods Using Markov Chains*. Biometrika.

### Tutorials

- [MCMC: The Markov Chain Monte Carlo Interactive Gallery](https://chi-feng.github.io/mcmc-demo/)

---

## Examples

### Example 1: Quantum Harmonic Oscillator Ground State Energy

```python
# =============================================
# Path integral Monte Carlo: harmonic oscillator
# =============================================
import numpy as np
import matplotlib.pyplot as plt

def path_integral_harmonic(N_beads=50, N_steps=100_000, beta=5.0, omega=1.0,
                            hbar=1.0, m=1.0, seed=42):
    """
    Estimate ground state energy of harmonic oscillator via PIMC.
    E_0 = hbar*omega/2 = 0.5 (in natural units).
    """
    rng = np.random.default_rng(seed)
    tau = beta / N_beads
    dtau = tau
    path = rng.normal(0, 1/np.sqrt(m*omega), N_beads)

    energies = []
    for step in range(N_steps):
        # Random bead update
        k = rng.integers(N_beads)
        delta = rng.normal(0, 0.3)
        x_new = path[k] + delta

        # Periodic boundary: bead k-1 and k+1
        k_m = (k - 1) % N_beads
        k_p = (k + 1) % N_beads

        # Action difference
        dS = (m / (2 * dtau)) * ((x_new - path[k_m])**2 - (path[k] - path[k_m])**2 +
                                  (path[k_p] - x_new)**2 - (path[k_p] - path[k])**2)
        dS += dtau * 0.5 * m * omega**2 * (x_new**2 - path[k]**2)

        if rng.uniform() < np.exp(-dS):
            path[k] = x_new

        if step > N_steps // 4:
            # Virial estimator
            E = 0.5 / beta + 0.5 * m * omega**2 * np.mean(path**2)
            energies.append(E)

    energies = np.array(energies)
    E_mean = energies.mean()
    E_err = energies.std() / np.sqrt(len(energies))
    return E_mean, E_err

E, dE = path_integral_harmonic()
print(f"E_0 (PIMC)  = {E:.4f} ± {dE:.4f}")
print(f"E_0 (exact) = 0.5000")
```

**Interpreting these results**: The PIMC estimate should converge to 0.5 ħω. Increase `N_beads` and `N_steps` to reduce statistical and systematic errors.

---

### Example 2: Bootstrap Error Estimation for Experimental Data

```python
# =============================================
# Bootstrap resampling for uncertainty quantification
# =============================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# Simulate experimental measurements with noise
N_exp = 200
true_mean, true_std = 3.14, 0.5
data = rng.normal(true_mean, true_std, N_exp)

def bootstrap_ci(data, statistic_fn, n_boot=5000, ci=95, seed=42):
    """
    Bootstrap confidence interval for any statistic.
    """
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    n = len(data)
    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = statistic_fn(sample)
    lo = np.percentile(boot_stats, (100 - ci) / 2)
    hi = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return boot_stats, lo, hi

# Estimate mean and its 95% CI
boot_means, lo, hi = bootstrap_ci(data, np.mean)
print(f"Sample mean:     {data.mean():.4f}")
print(f"Bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
print(f"Analytical 95% CI: [{data.mean() - 1.96*data.std()/np.sqrt(N_exp):.4f}, "
      f"{data.mean() + 1.96*data.std()/np.sqrt(N_exp):.4f}]")

# Estimate skewness and its CI
boot_skew, lo_sk, hi_sk = bootstrap_ci(data, stats.skew)
print(f"\nSkewness estimate: {stats.skew(data):.4f}")
print(f"Bootstrap 95% CI:  [{lo_sk:.4f}, {hi_sk:.4f}]")
```

**Interpreting these results**: Bootstrap CIs are distribution-free — use for any statistic (correlation, skewness, custom physics observable) where analytical formulas are unavailable.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
