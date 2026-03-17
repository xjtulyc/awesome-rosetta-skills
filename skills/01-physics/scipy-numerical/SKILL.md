---
name: scipy-numerical
description: >
  SciPy numerical toolkit for physics: ODE/PDE solving, FFT analysis, optimization,
  numerical integration, and sparse linear algebra with real-world examples.
tags:
  - scipy
  - numerical-methods
  - ode
  - fft
  - physics
  - optimization
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
  - scipy>=1.11.0
  - numpy>=1.24.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# SciPy Numerical Methods for Physics

SciPy provides a comprehensive suite of numerical algorithms essential for physics
simulation and data analysis. This skill covers ODE solving, PDE discretization,
numerical integration, FFT-based spectral analysis, nonlinear optimization, and
sparse linear algebra.

---

## 1. ODE Solving with `solve_ivp`

SciPy's `solve_ivp` supports multiple integration methods. Choose the right one:

| Method   | Best for                        | Notes                         |
|----------|---------------------------------|-------------------------------|
| RK45     | Non-stiff, smooth solutions     | Default, 4th/5th order        |
| RK23     | Non-stiff, low accuracy         | Cheaper per step              |
| DOP853   | Non-stiff, high accuracy        | 8th order Dormand-Prince      |
| Radau    | Stiff problems                  | Implicit, expensive but robust|
| BDF      | Very stiff (e.g., chemistry)    | Backward differentiation      |
| LSODA    | Automatic stiff detection       | Wraps ODEPACK                 |

### 1.1 Lorenz Attractor

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz_system(t, state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Lorenz attractor ODEs.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def simulate_lorenz(t_span=(0, 50), t_eval=None, initial_state=None, method="RK45"):
    """Simulate the Lorenz attractor and return the solution object."""
    if initial_state is None:
        initial_state = [1.0, 0.0, 0.0]
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 10000)

    sol = solve_ivp(
        lorenz_system,
        t_span,
        initial_state,
        method=method,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    return sol


def plot_lorenz_attractor(sol):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.4, alpha=0.8, color="steelblue")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Lorenz Attractor (RK45)")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    sol = simulate_lorenz()
    print(f"Integration status: {sol.status}, message: {sol.message}")
    print(f"Number of function evaluations: {sol.nfev}")
    fig = plot_lorenz_attractor(sol)
    plt.show()
```

### 1.2 Comparing RK45 vs Radau for a Damped Pendulum

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time


def damped_pendulum(t, state, omega0=2.0, zeta=0.1, A=0.5, omega_drive=1.9):
    """
    Driven, damped nonlinear pendulum.

    d^2 theta/dt^2 + 2*zeta*omega0 * dtheta/dt + omega0^2 * sin(theta)
        = A * cos(omega_drive * t)
    """
    theta, omega = state
    d_theta = omega
    d_omega = (
        -2 * zeta * omega0 * omega
        - omega0 ** 2 * np.sin(theta)
        + A * np.cos(omega_drive * t)
    )
    return [d_theta, d_omega]


def compare_pendulum_integrators():
    """Compare RK45 and Radau integrators for a damped pendulum problem."""
    t_span = (0, 60)
    t_eval = np.linspace(*t_span, 6000)
    y0 = [0.1, 0.0]

    results = {}
    for method in ["RK45", "RK23", "DOP853", "Radau"]:
        t0 = time.perf_counter()
        sol = solve_ivp(
            damped_pendulum,
            t_span,
            y0,
            method=method,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )
        elapsed = time.perf_counter() - t0
        results[method] = {
            "sol": sol,
            "time_s": elapsed,
            "nfev": sol.nfev,
            "nsteps": sol.t.size,
        }
        print(
            f"{method:8s}: wall={elapsed:.4f}s, nfev={sol.nfev:6d}, "
            f"success={sol.success}"
        )

    # Plot angle vs time for each method
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    colors = {"RK45": "steelblue", "RK23": "tomato", "DOP853": "green", "Radau": "purple"}
    for method, data in results.items():
        sol = data["sol"]
        axes[0].plot(sol.t, sol.y[0], label=method, color=colors[method], lw=1.2)
        axes[1].plot(sol.t, sol.y[1], color=colors[method], lw=1.2)

    axes[0].set_ylabel("Angle θ (rad)")
    axes[0].legend()
    axes[0].set_title("Driven Damped Pendulum — Method Comparison")
    axes[1].set_ylabel("Angular velocity ω (rad/s)")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":
    results = compare_pendulum_integrators()
```

---

## 2. PDE via Finite Difference (1D Diffusion)

Solving the 1D heat/diffusion equation using scipy.sparse for efficiency.

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def build_diffusion_matrix(N, dx, alpha, dt, scheme="crank-nicolson"):
    """
    Build the sparse matrices for the 1D diffusion equation.

    u_t = alpha * u_xx

    Schemes:
        'explicit'        : forward Euler (conditionally stable: dt <= dx^2/(2*alpha))
        'implicit'        : backward Euler (unconditionally stable)
        'crank-nicolson'  : CN (unconditionally stable, 2nd order in time)

    Returns (A, B) such that A @ u_new = B @ u_old.
    """
    r = alpha * dt / dx ** 2
    diagonals_inner = [-r * np.ones(N - 1), (1 + 2 * r) * np.ones(N), -r * np.ones(N - 1)]
    A = sp.diags(diagonals_inner, [-1, 0, 1], format="csc")

    if scheme == "crank-nicolson":
        r_half = 0.5 * alpha * dt / dx ** 2
        diag_A = [-r_half * np.ones(N - 1), (1 + 2 * r_half) * np.ones(N), -r_half * np.ones(N - 1)]
        diag_B = [r_half * np.ones(N - 1), (1 - 2 * r_half) * np.ones(N), r_half * np.ones(N - 1)]
        A = sp.diags(diag_A, [-1, 0, 1], format="csc")
        B = sp.diags(diag_B, [-1, 0, 1], format="csc")
    else:
        B = sp.eye(N, format="csc")

    return A, B


def solve_diffusion_1d(L=1.0, T=0.5, N=100, n_steps=500, alpha=0.01):
    """
    Solve 1D diffusion PDE on [0, L] over time [0, T].

    Initial condition: u(x, 0) = sin(pi * x / L)
    Boundary conditions: u(0, t) = u(L, t) = 0  (Dirichlet)
    Analytical solution: u(x, t) = exp(-alpha*(pi/L)^2*t) * sin(pi*x/L)
    """
    dx = L / (N + 1)
    dt = T / n_steps
    x = np.linspace(dx, L - dx, N)

    # Initial condition
    u = np.sin(np.pi * x / L)

    A, B = build_diffusion_matrix(N, dx, alpha, dt, scheme="crank-nicolson")
    lu = spla.splu(A)  # LU factorize once

    snapshots = {0: u.copy()}
    for step in range(n_steps):
        rhs = B @ u
        u = lu.solve(rhs)
        # Enforce boundary conditions
        u[0] = 0.0
        u[-1] = 0.0
        if (step + 1) % (n_steps // 5) == 0:
            t = (step + 1) * dt
            snapshots[t] = u.copy()

    # Analytical solution at final time
    t_final = T
    u_exact = np.exp(-alpha * (np.pi / L) ** 2 * t_final) * np.sin(np.pi * x / L)
    l2_error = np.sqrt(np.mean((u - u_exact) ** 2))
    print(f"L2 error at t={T}: {l2_error:.2e}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    for t_val, snap in sorted(snapshots.items()):
        ax.plot(x, snap, label=f"t={t_val:.2f}")
    ax.plot(x, u_exact, "k--", lw=2, label="Exact (final)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("1D Diffusion — Crank-Nicolson Scheme")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return x, u, u_exact


if __name__ == "__main__":
    solve_diffusion_1d()
```

---

## 3. Numerical Integration

```python
from scipy.integrate import quad, dblquad, nquad, romberg
import numpy as np


def demonstrate_numerical_integration():
    """Examples of numerical integration with scipy."""

    # --- 1D integration: Gaussian integral ---
    result, error = quad(lambda x: np.exp(-(x**2)), -np.inf, np.inf)
    print(f"Gaussian integral: {result:.8f}  (exact: {np.sqrt(np.pi):.8f}), error estimate: {error:.2e}")

    # --- Oscillatory integrand (specify limit for convergence) ---
    result_osc, err_osc = quad(
        lambda x: np.sin(100 * x) / x,
        0,
        np.inf,
        limit=200,
        epsabs=1e-10,
    )
    print(f"Integral of sin(100x)/x from 0 to inf: {result_osc:.6f} (exact: {np.pi/2:.6f})")

    # --- 2D integration ---
    def integrand_2d(y, x):
        return np.exp(-(x**2 + y**2))

    result_2d, err_2d = dblquad(
        integrand_2d,
        -np.inf, np.inf,      # x limits
        -np.inf, np.inf,      # y limits (can be functions of x)
    )
    print(f"2D Gaussian: {result_2d:.6f} (exact: {np.pi:.6f}), error: {err_2d:.2e}")

    # --- Romberg integration (requires 2^k + 1 evenly spaced points) ---
    result_rom = romberg(lambda x: np.sin(x) ** 4, 0, 2 * np.pi, tol=1e-12, show=False)
    print(f"Romberg integral of sin^4(x) on [0,2pi]: {result_rom:.8f} (exact: {3*np.pi/4:.8f})")

    # --- n-dimensional integration ---
    def banana(x):
        """Rosenbrock-like function on [0,1]^3."""
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + x[2]

    ranges = [(0, 1), (0, 1), (0, 1)]
    result_nd, err_nd = nquad(lambda z, y, x: banana([x, y, z]), ranges[::-1])
    print(f"3D integral of banana function: {result_nd:.6f}, error: {err_nd:.2e}")


if __name__ == "__main__":
    demonstrate_numerical_integration()
```

---

## 4. FFT Analysis — Harmonic Oscillator & Signal Processing

```python
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import welch, periodogram
import matplotlib.pyplot as plt


def generate_signal(fs=1000.0, duration=2.0, freqs=(50, 120, 300), noise_level=0.3):
    """
    Generate a synthetic multi-tone signal with noise.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    duration : float
        Signal duration in seconds.
    freqs : tuple
        Frequencies of sinusoidal components.
    noise_level : float
        Standard deviation of additive Gaussian noise.

    Returns
    -------
    t : ndarray
        Time array.
    signal : ndarray
        Signal values.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    rng = np.random.default_rng(42)
    signal += noise_level * rng.standard_normal(len(t))
    return t, signal


def fft_analysis(t, signal, fs=1000.0):
    """
    Perform FFT-based spectral analysis on a time-domain signal.

    Returns frequency array, one-sided power spectral density.
    """
    N = len(signal)
    # Real FFT (more efficient for real signals)
    yf = rfft(signal)
    xf = rfftfreq(N, d=1.0 / fs)

    # Power spectral density (one-sided, normalized)
    psd = (2.0 / N) * np.abs(yf) ** 2

    # Welch PSD for better noise averaging
    f_welch, psd_welch = welch(signal, fs=fs, nperseg=256, noverlap=128)

    return xf, psd, f_welch, psd_welch


def harmonic_oscillator_response(omega0=2 * np.pi * 50, zeta=0.05, fs=1000.0, duration=5.0):
    """
    Compute the frequency response of a harmonic oscillator and verify via FFT.

    H(omega) = omega0^2 / (omega0^2 - omega^2 + 2j*zeta*omega0*omega)
    """
    omega = 2 * np.pi * np.linspace(0, fs / 2, 5000)
    H = omega0 ** 2 / (omega0 ** 2 - omega ** 2 + 2j * zeta * omega0 * omega)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].semilogy(omega / (2 * np.pi), np.abs(H), color="steelblue")
    axes[0].axvline(omega0 / (2 * np.pi), color="red", ls="--", label=f"f0={omega0/(2*np.pi):.1f} Hz")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("|H(f)|")
    axes[0].set_title("Harmonic Oscillator Transfer Function")
    axes[0].legend()

    phase = np.angle(H, deg=True)
    axes[1].plot(omega / (2 * np.pi), phase, color="darkorange")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (°)")
    axes[1].axvline(omega0 / (2 * np.pi), color="red", ls="--")
    plt.tight_layout()
    plt.show()

    return omega / (2 * np.pi), np.abs(H), phase


def plot_fft_results(t, signal, xf, psd, f_welch, psd_welch, target_freqs=(50, 120, 300)):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(t[:500], signal[:500], lw=0.8, color="steelblue")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Time-Domain Signal (first 0.5 s)")

    axes[1].plot(xf, psd, lw=0.8, color="darkorange", label="FFT PSD")
    for f in target_freqs:
        axes[1].axvline(f, color="red", ls="--", alpha=0.6)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_xlim(0, 500)
    axes[1].set_title("Power Spectral Density (FFT)")
    axes[1].legend()

    axes[2].semilogy(f_welch, psd_welch, lw=1.2, color="green", label="Welch PSD")
    for f in target_freqs:
        axes[2].axvline(f, color="red", ls="--", alpha=0.6, label=f"f={f} Hz")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("PSD (log)")
    axes[2].set_xlim(0, 500)
    axes[2].set_title("Welch Power Spectral Density")
    axes[2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    fs = 1000.0
    t, signal = generate_signal(fs=fs)
    xf, psd, f_welch, psd_welch = fft_analysis(t, signal, fs=fs)
    plot_fft_results(t, signal, xf, psd, f_welch, psd_welch)
    harmonic_oscillator_response()
```

---

## 5. Optimization: Curve Fitting with Error Propagation

```python
import numpy as np
from scipy.optimize import curve_fit, minimize, minimize_scalar, least_squares
import matplotlib.pyplot as plt


def damped_cosine(t, A, tau, omega, phi):
    """Model: A * exp(-t/tau) * cos(omega*t + phi)."""
    return A * np.exp(-t / tau) * np.cos(omega * t + phi)


def fit_damped_signal(seed=0):
    """
    Generate noisy data from a damped cosine and recover parameters with
    uncertainty estimates from the covariance matrix.
    """
    rng = np.random.default_rng(seed)
    t_data = np.linspace(0, 10, 200)
    true_params = {"A": 3.5, "tau": 2.5, "omega": 4.0, "phi": 0.3}
    y_true = damped_cosine(t_data, **true_params)
    noise = 0.25 * rng.standard_normal(len(t_data))
    y_noisy = y_true + noise

    # Fit with bounds to avoid unphysical solutions
    p0 = [3.0, 2.0, 4.0, 0.0]
    bounds = ([0, 0.1, 0.1, -np.pi], [10, 20, 20, np.pi])

    popt, pcov = curve_fit(damped_cosine, t_data, y_noisy, p0=p0, bounds=bounds, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))  # 1-sigma uncertainties

    param_names = ["A", "tau", "omega", "phi"]
    print("\nFitted parameters (true value → fit ± 1σ):")
    for name, true, fit, err in zip(param_names, true_params.values(), popt, perr):
        print(f"  {name:5s}: {true:.3f} → {fit:.3f} ± {err:.3f}")

    # Residuals
    residuals = y_noisy - damped_cosine(t_data, *popt)
    chi_sq = np.sum((residuals / 0.25) ** 2)
    dof = len(t_data) - len(popt)
    print(f"\nReduced chi-squared: {chi_sq/dof:.3f} (ideal ~ 1)")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].scatter(t_data, y_noisy, s=8, alpha=0.6, label="Noisy data", color="steelblue")
    axes[0].plot(t_data, y_true, "k--", label="True signal", lw=1.5)
    axes[0].plot(t_data, damped_cosine(t_data, *popt), "r-", label="Fit", lw=2)
    axes[0].set_ylabel("Signal")
    axes[0].legend()
    axes[0].set_title("Damped Cosine Curve Fitting")

    axes[1].plot(t_data, residuals, color="darkorange", lw=0.8)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Fit Residuals")

    plt.tight_layout()
    plt.show()
    return popt, pcov


def scalar_optimization_demo():
    """Demonstrate minimize_scalar for finding potential energy minima."""
    def lennard_jones(r, epsilon=1.0, sigma=1.0):
        return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    result = minimize_scalar(lennard_jones, bounds=(0.9, 3.0), method="bounded")
    r_eq = result.x
    E_eq = result.fun
    print(f"\nLennard-Jones equilibrium: r* = {r_eq:.4f} σ, E* = {E_eq:.4f} ε")
    # Analytical: r* = 2^(1/6) * sigma
    print(f"Analytical r*: {2**(1/6):.4f} σ")

    r = np.linspace(0.9, 3.0, 300)
    plt.figure(figsize=(8, 4))
    plt.plot(r, lennard_jones(r), label="LJ potential")
    plt.scatter([r_eq], [E_eq], color="red", zorder=5, label=f"Min at r={r_eq:.3f}")
    plt.axhline(0, color="gray", ls="--", lw=0.8)
    plt.ylim(-1.5, 2.0)
    plt.xlabel("r / σ")
    plt.ylabel("U(r) / ε")
    plt.title("Lennard-Jones Potential Minimum")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    popt, pcov = fit_damped_signal()
    scalar_optimization_demo()
```

---

## 6. Sparse Linear Algebra — Eigenvalue Problem

```python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def particle_in_box_hamiltonian(N=500, L=1.0, hbar=1.0, m=1.0):
    """
    Construct the 1D particle-in-a-box Hamiltonian using finite differences.

    H = -hbar^2/(2m) * d^2/dx^2

    Discretized on N interior points with Dirichlet boundary conditions.

    Returns
    -------
    H : sparse matrix (CSC)
        Hamiltonian matrix.
    x : ndarray
        Grid points.
    """
    dx = L / (N + 1)
    x = np.linspace(dx, L - dx, N)

    diag_main = 2.0 * np.ones(N)
    diag_off = -1.0 * np.ones(N - 1)
    kinetic = sp.diags([diag_off, diag_main, diag_off], [-1, 0, 1], format="csc")
    H = (hbar ** 2 / (2 * m * dx ** 2)) * kinetic
    return H, x


def solve_particle_in_box(N=500, L=1.0, n_eigvals=6):
    """
    Solve for the lowest n_eigvals eigenstates of the particle in a box.

    Analytical energies: E_n = n^2 * pi^2 * hbar^2 / (2 m L^2)
    """
    H, x = particle_in_box_hamiltonian(N=N, L=L)

    # Use eigsh for symmetric matrices — much faster than full diagonalization
    eigenvalues, eigenvectors = spla.eigsh(H, k=n_eigvals, which="SM")

    # Sort by energy
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Normalize eigenvectors
    dx = L / (N + 1)
    for i in range(n_eigvals):
        norm = np.sqrt(np.trapz(eigenvectors[:, i] ** 2, dx=dx))
        eigenvectors[:, i] /= norm

    # Analytical energies (hbar=m=1)
    n_vals = np.arange(1, n_eigvals + 1)
    E_exact = n_vals ** 2 * np.pi ** 2 / (2 * L ** 2)

    print("\nParticle-in-a-box energy levels (hbar=m=1):")
    print(f"{'n':>4} {'Numerical':>14} {'Analytical':>14} {'Rel. Error':>12}")
    for i, (En, Ea) in enumerate(zip(eigenvalues, E_exact)):
        print(f"{i+1:>4} {En:>14.6f} {Ea:>14.6f} {abs(En-Ea)/Ea:>12.2e}")

    # Plot wavefunctions
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i in range(n_eigvals):
        psi = eigenvectors[:, i]
        # Fix sign convention
        if psi[N // 4] < 0:
            psi = -psi
        axes[0].plot(x, psi + eigenvalues[i], label=f"n={i+1}", lw=1.5)

    axes[0].set_xlabel("x / L")
    axes[0].set_ylabel("Energy + ψ(x)")
    axes[0].set_title("Particle-in-a-Box Wavefunctions")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].scatter(n_vals, eigenvalues, label="Numerical", zorder=5)
    axes[1].plot(n_vals, E_exact, "r--", label="Analytical", lw=1.5)
    axes[1].set_xlabel("Quantum number n")
    axes[1].set_ylabel("Energy")
    axes[1].set_title("Energy Levels Comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return eigenvalues, eigenvectors, x


if __name__ == "__main__":
    solve_particle_in_box(N=800, n_eigvals=8)
```

---

## 7. Complete Example A — Damped Pendulum: RK45 vs Radau

This comprehensive example wraps everything together: ODE solving with two stiff
and non-stiff methods, phase-space analysis, and Poincaré section extraction.

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def pendulum_ode(t, y, b=0.25, c=5.0):
    """
    Simple pendulum ODE:
        theta'' + b * theta' + c * sin(theta) = 0
    State: y = [theta, omega]
    """
    theta, omega = y
    dydt = [omega, -b * omega - c * np.sin(theta)]
    return dydt


def pendulum_event(t, y, **kwargs):
    """Event: detect zero-crossings of theta (Poincaré section)."""
    return y[0]


pendulum_event.terminal = False
pendulum_event.direction = 1


def run_pendulum_comparison(theta0=np.pi - 0.1, omega0=0.0, t_end=30.0):
    """
    Solve the pendulum ODE with RK45 and Radau, compare trajectories.
    """
    y0 = [theta0, omega0]
    t_eval = np.linspace(0, t_end, 3000)

    solutions = {}
    for method in ["RK45", "Radau"]:
        sol = solve_ivp(
            pendulum_ode,
            (0, t_end),
            y0,
            method=method,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
            events=pendulum_event,
        )
        solutions[method] = sol
        print(f"{method}: nfev={sol.nfev}, success={sol.success}, steps={sol.t.size}")

    # Compute energy (should be conserved for b=0)
    def energy(theta, omega, c=5.0):
        return 0.5 * omega ** 2 - c * np.cos(theta)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    colors = {"RK45": "steelblue", "Radau": "darkorange"}

    for method, sol in solutions.items():
        theta = sol.y[0]
        omega = sol.y[1]
        E = energy(theta, omega)

        axes[0, 0].plot(sol.t, theta, label=method, color=colors[method], lw=1)
        axes[0, 1].plot(theta, omega, color=colors[method], label=method, lw=0.8, alpha=0.8)
        axes[1, 0].plot(sol.t, E - E[0], color=colors[method], label=method, lw=1)

    # Difference between methods
    theta_rk45 = solutions["RK45"].y[0]
    theta_radau = solutions["Radau"].y[0]
    diff = np.abs(theta_rk45 - theta_radau)
    axes[1, 1].semilogy(solutions["RK45"].t, diff + 1e-16, color="purple", lw=1)
    axes[1, 1].set_title("RK45 vs Radau |Δθ|")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("|Δθ| (rad)")

    axes[0, 0].set_title("Pendulum Angle")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("θ (rad)")
    axes[0, 0].legend()

    axes[0, 1].set_title("Phase Portrait")
    axes[0, 1].set_xlabel("θ (rad)")
    axes[0, 1].set_ylabel("ω (rad/s)")
    axes[0, 1].legend()

    axes[1, 0].set_title("Energy Drift (b=0.25, damped)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("ΔE")
    axes[1, 0].legend()

    plt.suptitle("Damped Pendulum: RK45 vs Radau", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_pendulum_comparison()
```

---

## 8. Complete Example B — Frequency Analysis Pipeline

```python
import numpy as np
from scipy.fft import rfft, rfftfreq, irfft
from scipy.signal import butter, sosfilt, find_peaks
import matplotlib.pyplot as plt


def frequency_analysis_pipeline(fs=2048.0, duration=4.0):
    """
    Full frequency analysis pipeline:
    1. Generate composite signal
    2. Apply bandpass filter
    3. Find dominant peaks
    4. Reconstruct filtered signal via inverse FFT
    """
    rng = np.random.default_rng(123)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Composite: 3 tones + chirp + noise
    signal = (
        2.0 * np.sin(2 * np.pi * 60 * t)
        + 0.8 * np.sin(2 * np.pi * 180 * t)
        + 0.4 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * rng.standard_normal(len(t))
    )

    # FFT
    N = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(N, d=1.0 / fs)
    amplitude = (2.0 / N) * np.abs(yf)

    # Find spectral peaks
    peaks, props = find_peaks(amplitude, height=0.05, distance=20)
    print("Detected frequency peaks:")
    for p in peaks[:10]:
        print(f"  f = {xf[p]:.1f} Hz, amplitude = {amplitude[p]:.4f}")

    # Bandpass filter: 50–500 Hz using Butterworth
    sos = butter(6, [50, 500], btype="bandpass", fs=fs, output="sos")
    filtered = sosfilt(sos, signal)

    # Selective reconstruction via zeroing out-of-band FFT bins
    yf_filtered = yf.copy()
    mask = (xf < 50) | (xf > 500)
    yf_filtered[mask] = 0.0
    reconstructed = irfft(yf_filtered, n=N)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9))

    axes[0].plot(t[:1024], signal[:1024], lw=0.6, color="steelblue", label="Original")
    axes[0].plot(t[:1024], filtered[:1024], lw=1.0, color="red", alpha=0.8, label="Filtered")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].set_title("Time Domain")

    axes[1].plot(xf, amplitude, lw=0.8, color="darkorange")
    axes[1].scatter(xf[peaks], amplitude[peaks], color="red", zorder=5, s=40, label="Peaks")
    axes[1].set_xlim(0, 600)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].set_title("Amplitude Spectrum")

    axes[2].plot(t[:1024], filtered[:1024], lw=0.8, color="green", label="Butterworth")
    axes[2].plot(t[:1024], reconstructed[:1024], lw=1.0, ls="--", color="purple", label="IFFT reconstruction")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].set_title("Filtered Signal Comparison")

    plt.tight_layout()
    plt.show()

    return t, signal, filtered, xf, amplitude


if __name__ == "__main__":
    frequency_analysis_pipeline()
```

---

## Quick Reference

| Task                           | SciPy function                            |
|--------------------------------|-------------------------------------------|
| Solve IVP (non-stiff)          | `solve_ivp(..., method='RK45')`           |
| Solve IVP (stiff)              | `solve_ivp(..., method='Radau')`          |
| 1D quadrature                  | `quad(f, a, b)`                           |
| 2D quadrature                  | `dblquad(f, a, b, gfun, hfun)`            |
| FFT (real signal)              | `rfft(x)` + `rfftfreq(N, d=1/fs)`        |
| Welch PSD                      | `welch(x, fs=fs, nperseg=256)`            |
| Nonlinear optimization         | `minimize(f, x0, method='L-BFGS-B')`     |
| Curve fitting                  | `curve_fit(model, xdata, ydata, p0)`      |
| Sparse eigenvalues             | `eigsh(A, k=6, which='SM')`               |
| Sparse linear solve            | `spsolve(A, b)`                           |
| Sparse LU (repeated solves)    | `splu(A); lu.solve(b)`                    |

### Tips

- Always set `rtol` and `atol` explicitly in `solve_ivp`; defaults may be too loose.
- For stiff ODEs, try `Radau` first; `BDF` is faster for very large systems.
- Use `rfft`/`rfftfreq` instead of `fft`/`fftfreq` for real-valued signals — it's 2x faster.
- When doing repeated sparse solves with the same matrix, factorize once with `splu`.
- In `curve_fit`, always provide `bounds` to avoid unphysical parameter values.
- Use `scipy.sparse.linalg.eigsh` (not `eigs`) for real symmetric matrices.
