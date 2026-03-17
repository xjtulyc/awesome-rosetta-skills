---
name: sympy-symbolic
description: >
  SymPy symbolic computation for physics: algebra, calculus, linear algebra,
  ODE solving, Lagrangian mechanics, quantum physics, and code generation.
tags:
  - sympy
  - symbolic-computation
  - physics
  - mechanics
  - quantum
  - code-generation
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
  - sympy>=1.12
  - numpy>=1.24.0
  - matplotlib>=3.7.0
  - scipy>=1.11.0
last_updated: "2026-03-17"
---

# SymPy Symbolic Computation for Physics

SymPy is a pure-Python computer algebra system. This skill demonstrates how to use
SymPy for symbolic physics: manipulating equations, solving ODEs analytically,
deriving equations of motion via Lagrangian mechanics, performing quantum mechanics
calculations, and converting symbolic expressions to fast numerical code.

---

## 1. Symbolic Algebra Fundamentals

```python
import sympy as sp

# Define symbols with assumptions
x, y, z = sp.symbols("x y z", real=True)
t = sp.Symbol("t", positive=True)
n = sp.Symbol("n", integer=True, positive=True)
epsilon, omega, phi = sp.symbols("epsilon omega phi", positive=True)

# Basic algebra
expr = (x + y) ** 4
print("Expanded:", sp.expand(expr))

factored = sp.factor(x**4 - 1)
print("Factored:", factored)

# Simplification
trig_expr = sp.sin(x) ** 2 + sp.cos(x) ** 2
print("Trig simplified:", sp.trigsimp(trig_expr))

# Power series expansion
f = sp.exp(sp.I * x)
series = sp.series(f, x, 0, 6)
print("Euler series:", series)

# Solve algebraic equations
quadratic = sp.Eq(x**2 + 3*x + 2, 0)
solutions = sp.solve(quadratic, x)
print("Quadratic roots:", solutions)

# System of equations
eq1 = sp.Eq(2*x + y, 5)
eq2 = sp.Eq(x - y, 1)
system_sol = sp.solve([eq1, eq2], [x, y])
print("System solution:", system_sol)
```

---

## 2. Symbolic Calculus

```python
import sympy as sp

x, t, a, b = sp.symbols("x t a b", real=True)

# Differentiation
f = sp.sin(x**2) * sp.exp(-x)
df = sp.diff(f, x)
d2f = sp.diff(f, x, 2)
print("f(x)   =", f)
print("f'(x)  =", sp.simplify(df))
print("f''(x) =", sp.simplify(d2f))

# Partial derivatives
phi = sp.Function("phi")(x, t)
wave_eq = sp.diff(phi, t, 2) - sp.diff(phi, x, 2)
print("\nWave equation LHS:", wave_eq)

# Definite and indefinite integrals
I1 = sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo))
print("\nGaussian integral:", I1)  # sqrt(pi)

I2 = sp.integrate(sp.sin(x) * sp.cos(x), x)
print("Trig integral:", sp.simplify(I2))

# Integration with parameters
I3 = sp.integrate(sp.exp(-a * x**2), (x, 0, sp.oo))
print("Parameterized Gaussian:", sp.simplify(I3))

# Limits
lim1 = sp.limit(sp.sin(x) / x, x, 0)
print("\nlim_{x->0} sin(x)/x =", lim1)

lim2 = sp.limit((1 + 1/x)**x, x, sp.oo)
print("lim_{x->inf} (1+1/x)^x =", lim2)

# Fourier series (manual via integration)
def fourier_coefficient(f, L, n, var):
    """Compute the nth Fourier coefficient of f on [-L, L]."""
    a_n = (1 / L) * sp.integrate(f * sp.cos(n * sp.pi * var / L), (var, -L, L))
    b_n = (1 / L) * sp.integrate(f * sp.sin(n * sp.pi * var / L), (var, -L, L))
    return sp.simplify(a_n), sp.simplify(b_n)

n = sp.Symbol("n", integer=True, positive=True)
L = sp.pi
# Square wave f(x) = x on [-pi, pi]
a_n, b_n = fourier_coefficient(x, L, n, x)
print("\nFourier coefficients of f(x)=x on [-pi,pi]:")
print("  a_n =", a_n)
print("  b_n =", b_n)
```

---

## 3. Symbolic Linear Algebra

```python
import sympy as sp

# Symbolic matrix
a, b, c, d = sp.symbols("a b c d")
M = sp.Matrix([[a, b], [c, d]])
print("Matrix M:\n", M)
print("Determinant:", M.det())
print("Inverse:\n", sp.simplify(M.inv()))

# Eigenvalues and eigenvectors
eigenvals = M.eigenvals()
eigenvects = M.eigenvects()
print("\nEigenvalues:", eigenvals)
for eigenval, mult, vecs in eigenvects:
    print(f"  λ = {eigenval}, multiplicity = {mult}, eigenvector = {vecs[0].T}")

# Pauli matrices (quantum mechanics)
sx = sp.Matrix([[0, 1], [1, 0]])
sy = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sz = sp.Matrix([[1, 0], [0, -1]])

print("\nPauli matrix commutation relations:")
print("[sx, sy] = 2i*sz:", sp.simplify(sx * sy - sy * sx) == 2 * sp.I * sz)
print("[sy, sz] = 2i*sx:", sp.simplify(sy * sz - sz * sy) == 2 * sp.I * sx)
print("[sz, sx] = 2i*sy:", sp.simplify(sz * sx - sx * sz) == 2 * sp.I * sy)

# Rotation matrix and its properties
theta = sp.Symbol("theta", real=True)
R = sp.Matrix([
    [sp.cos(theta), -sp.sin(theta)],
    [sp.sin(theta),  sp.cos(theta)],
])
print("\nRotation matrix R @ R.T:")
print(sp.simplify(R * R.T))  # Should be identity
print("det(R):", sp.simplify(R.det()))

# Solve linear system symbolically
A = sp.Matrix([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b_vec = sp.Matrix([8, -11, -3])
solution = A.solve(b_vec)
print("\nLinear system solution:", solution.T)
```

---

## 4. Solving Differential Equations Symbolically

```python
import sympy as sp

t = sp.Symbol("t", positive=True)
x = sp.Function("x")

# --- Simple harmonic oscillator ---
omega = sp.Symbol("omega", positive=True)
ode_sho = sp.Eq(x(t).diff(t, 2) + omega**2 * x(t), 0)
sol_sho = sp.dsolve(ode_sho, x(t))
print("SHO solution:", sol_sho)

# Apply initial conditions x(0)=1, x'(0)=0
C1, C2 = sp.symbols("C1 C2")
ics = {x(0): 1, x(t).diff(t).subs(t, 0): 0}
sol_ics = sp.dsolve(ode_sho, x(t), ics=ics)
print("SHO with ICs:", sol_ics)

# --- Damped oscillator ---
gamma, omega0 = sp.symbols("gamma omega_0", positive=True)
ode_damped = sp.Eq(x(t).diff(t, 2) + 2*gamma*x(t).diff(t) + omega0**2*x(t), 0)
sol_damped = sp.dsolve(ode_damped, x(t))
print("\nDamped oscillator solution:", sp.simplify(sol_damped.rhs))

# --- Driven oscillator ---
F0, omega_d = sp.symbols("F_0 omega_d", positive=True)
ode_driven = sp.Eq(
    x(t).diff(t, 2) + 2*gamma*x(t).diff(t) + omega0**2*x(t),
    F0 * sp.cos(omega_d * t)
)
sol_driven = sp.dsolve(ode_driven, x(t))
print("\nDriven oscillator particular solution:")
print(sp.simplify(sol_driven.rhs))

# --- Heat equation (separation of variables) ---
# Verify that u(x, t) = exp(-alpha * k^2 * t) * sin(k * x) satisfies u_t = alpha * u_xx
x_var = sp.Symbol("x", real=True)
alpha, k = sp.symbols("alpha k", positive=True)
u = sp.exp(-alpha * k**2 * t) * sp.sin(k * x_var)
residual = sp.diff(u, t) - alpha * sp.diff(u, x_var, 2)
print("\nHeat equation residual (should be 0):", sp.simplify(residual))
```

---

## 5. Lagrangian Mechanics — Double Pendulum

Derive the equations of motion for a double pendulum using the Lagrangian formalism.

```python
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols, LagrangesMethod, Particle, Point, ReferenceFrame


def derive_double_pendulum():
    """
    Derive equations of motion for a planar double pendulum via Lagrangian mechanics.

    q1 = angle of rod 1 from vertical
    q2 = angle of rod 2 from vertical
    """
    t = sp.Symbol("t")
    m1, m2, l1, l2, g = sp.symbols("m_1 m_2 l_1 l_2 g", positive=True)

    # Generalized coordinates (functions of time)
    q1 = dynamicsymbols("q_1")
    q2 = dynamicsymbols("q_2")
    q1d = dynamicsymbols("q_1", 1)
    q2d = dynamicsymbols("q_2", 1)

    # Cartesian positions
    x1 = l1 * sp.sin(q1)
    y1 = -l1 * sp.cos(q1)
    x2 = x1 + l2 * sp.sin(q2)
    y2 = y1 - l2 * sp.cos(q2)

    # Velocities
    x1d = sp.diff(x1, t)
    y1d = sp.diff(y1, t)
    x2d = sp.diff(x2, t)
    y2d = sp.diff(y2, t)

    # Kinetic and potential energy
    T = sp.Rational(1, 2) * m1 * (x1d**2 + y1d**2) + \
        sp.Rational(1, 2) * m2 * (x2d**2 + y2d**2)
    V = m1 * g * y1 + m2 * g * y2

    L = T - V
    L = sp.simplify(L)
    print("Lagrangian L = T - V:")
    print(sp.simplify(L))

    # Euler-Lagrange equations
    def euler_lagrange(L, q, qd, t):
        """Compute d/dt(dL/dqd) - dL/dq = 0."""
        dL_dqd = sp.diff(L, qd)
        d_dt_dL_dqd = sp.diff(dL_dqd, t)
        dL_dq = sp.diff(L, q)
        return sp.simplify(d_dt_dL_dqd - dL_dq)

    eq1 = euler_lagrange(L, q1, q1d, t)
    eq2 = euler_lagrange(L, q2, q2d, t)

    print("\nEuler-Lagrange equation 1 (q1):")
    print(sp.simplify(eq1))
    print("\nEuler-Lagrange equation 2 (q2):")
    print(sp.simplify(eq2))

    # Linearize for small angles (sin(q) ≈ q, cos(q) ≈ 1)
    def linearize(expr, coords):
        result = expr
        for q in coords:
            result = result.subs(sp.sin(q), q).subs(sp.cos(q), 1)
        return sp.simplify(result)

    eq1_lin = linearize(eq1, [q1, q2])
    eq2_lin = linearize(eq2, [q1, q2])
    print("\nLinearized EOM (small angles):")
    print("EOM1:", eq1_lin)
    print("EOM2:", eq2_lin)

    return eq1, eq2, eq1_lin, eq2_lin


if __name__ == "__main__":
    eqs = derive_double_pendulum()
```

---

## 6. Quantum Harmonic Oscillator

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


def quantum_harmonic_oscillator():
    """
    Symbolic solution of the quantum harmonic oscillator.

    H psi_n = E_n psi_n
    E_n = hbar * omega * (n + 1/2)
    psi_n(x) = N_n * H_n(xi) * exp(-xi^2/2)  where xi = sqrt(m*omega/hbar)*x
    """
    x, xi = sp.symbols("x xi", real=True)
    n = sp.Symbol("n", nonnegative=True, integer=True)
    hbar, omega, m = sp.symbols("hbar omega m", positive=True)

    # Energy eigenvalues
    E_n = hbar * omega * (n + sp.Rational(1, 2))
    print("Energy eigenvalues:")
    for k in range(6):
        print(f"  E_{k} = {E_n.subs(n, k)} * hbar*omega")

    # Hermite polynomials (physicist's convention)
    def hermite(n_val, xi_sym):
        """Generate Hermite polynomial H_n(xi) via recurrence."""
        if n_val == 0:
            return sp.Integer(1)
        elif n_val == 1:
            return 2 * xi_sym
        else:
            H_prev2 = sp.Integer(1)
            H_prev1 = 2 * xi_sym
            for k in range(2, n_val + 1):
                H_curr = 2 * xi_sym * H_prev1 - 2 * (k - 1) * H_prev2
                H_prev2 = H_prev1
                H_prev1 = H_curr
            return sp.expand(H_curr)

    # Normalization constant N_n = 1/sqrt(2^n * n! * sqrt(pi))
    def norm_constant(n_val):
        return 1 / sp.sqrt(2**n_val * sp.factorial(n_val) * sp.sqrt(sp.pi))

    # Wavefunctions
    print("\nWavefunctions psi_n(xi):")
    wavefunctions = []
    for k in range(5):
        Hn = hermite(k, xi)
        Nn = norm_constant(k)
        psi = Nn * Hn * sp.exp(-xi**2 / 2)
        psi_simplified = sp.simplify(psi)
        wavefunctions.append(psi_simplified)
        print(f"  psi_{k}(xi) = {psi_simplified}")

    # Verify orthonormality of psi_0 and psi_1
    inner_00 = sp.integrate(wavefunctions[0]**2, (xi, -sp.oo, sp.oo))
    inner_01 = sp.integrate(wavefunctions[0] * wavefunctions[1], (xi, -sp.oo, sp.oo))
    inner_11 = sp.integrate(wavefunctions[1]**2, (xi, -sp.oo, sp.oo))
    print(f"\nOrthonormality check:")
    print(f"  <0|0> = {sp.simplify(inner_00)}")
    print(f"  <0|1> = {sp.simplify(inner_01)}")
    print(f"  <1|1> = {sp.simplify(inner_11)}")

    # Convert to numpy functions for plotting
    psi_funcs = [sp.lambdify(xi, psi, "numpy") for psi in wavefunctions]

    xi_vals = np.linspace(-4, 4, 500)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["steelblue", "darkorange", "green", "red", "purple"]
    for k, (psi_np, color) in enumerate(zip(psi_funcs, colors)):
        psi_vals = psi_np(xi_vals)
        ax.plot(xi_vals, psi_vals + k + 0.5, label=f"ψ_{k}(ξ)", color=color, lw=1.8)
        ax.axhline(k + 0.5, color=color, ls="--", alpha=0.3)

    # Harmonic potential
    V = xi_vals**2 / 2
    ax.plot(xi_vals, V / 5, "k-", lw=1.5, alpha=0.3, label="V(ξ)/5")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xlabel("ξ = x/x₀")
    ax.set_ylabel("ψₙ(ξ) + Eₙ/ℏω")
    ax.set_title("Quantum Harmonic Oscillator Wavefunctions")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    return wavefunctions


if __name__ == "__main__":
    quantum_harmonic_oscillator()
```

---

## 7. Code Generation: lambdify and CSE

```python
import sympy as sp
import numpy as np
import time


def code_generation_demo():
    """
    Demonstrate converting symbolic expressions to fast numerical functions.
    lambdify: converts SymPy expression to numpy/scipy callable.
    cse: common subexpression elimination for optimized computation.
    """
    x, y, z = sp.symbols("x y z", real=True)

    # Complex symbolic expression
    expr = (sp.sin(x) * sp.cos(y) + sp.exp(-x**2 - y**2)) / (1 + sp.sqrt(x**2 + y**2 + z**2))

    # lambdify with numpy backend
    f_np = sp.lambdify([x, y, z], expr, modules="numpy")

    # Test on arrays
    x_arr = np.random.randn(100_000)
    y_arr = np.random.randn(100_000)
    z_arr = np.random.randn(100_000)

    t0 = time.perf_counter()
    result = f_np(x_arr, y_arr, z_arr)
    print(f"lambdify evaluation on 100k points: {time.perf_counter()-t0:.4f}s")
    print(f"  Result sample: {result[:5]}")

    # CSE for expensive multi-output expressions
    expr_list = [
        sp.sin(x)**2 + sp.cos(x)*sp.cos(y),
        sp.sin(x)**2 * sp.exp(-y),
        sp.cos(x)**2 - sp.sin(x)*sp.sin(y),
    ]

    replacements, reduced_exprs = sp.cse(expr_list)
    print("\nCSE replacements:")
    for sym, val in replacements:
        print(f"  {sym} = {val}")
    print("Reduced expressions:")
    for e in reduced_exprs:
        print(f"  {e}")

    # Generate Python code from CSE output
    from sympy.printing.pycode import pycode
    print("\nGenerated Python code snippet:")
    for sym, val in replacements:
        print(f"  {sym} = {pycode(val)}")

    # LaTeX output for documentation
    omega, t_sym = sp.symbols("omega t", real=True)
    gamma_sym = sp.Symbol("gamma", positive=True)
    transfer_fn = omega**2 / (omega**2 - t_sym**2 + 2*sp.I*gamma_sym*t_sym)
    print("\nLaTeX representation of transfer function:")
    print(sp.latex(transfer_fn))

    # Numerical integration via lambdify + scipy.integrate
    from scipy.integrate import quad
    integrand_sym = sp.exp(-x**2) * sp.cos(x)
    integrand_np = sp.lambdify(x, integrand_sym, modules="numpy")
    result_int, err = quad(integrand_np, -np.inf, np.inf)
    # Analytical: sqrt(pi) * exp(-1/4)
    exact = np.sqrt(np.pi) * np.exp(-0.25)
    print(f"\nIntegral of exp(-x^2)*cos(x): {result_int:.8f} (exact: {exact:.8f})")


if __name__ == "__main__":
    code_generation_demo()
```

---

## 8. Maxwell Equations Symbolic Manipulation

```python
import sympy as sp
from sympy.vector import CoordSys3D, curl, divergence, gradient


def maxwell_equations_demo():
    """
    Demonstrate Maxwell equations using SymPy's vector calculus.

    In Gaussian units:
      div E = 4*pi*rho
      div B = 0
      curl E = -1/c * dB/dt
      curl B = 4*pi/c * J + 1/c * dE/dt
    """
    # Setup coordinate system
    N = CoordSys3D("N")
    x, y, z, t = sp.symbols("x y z t", real=True)
    epsilon0, mu0, c = sp.symbols("epsilon_0 mu_0 c", positive=True)
    omega_k, k_val = sp.symbols("omega k", positive=True)

    # Plane wave: E in x-direction, propagating in z-direction
    # E_x = E0 * cos(k*z - omega*t)
    E0 = sp.Symbol("E_0", real=True)
    E_x = E0 * sp.cos(k_val * z - omega_k * t)
    E_field = E_x * N.i + 0 * N.j + 0 * N.k

    # Corresponding B field from Faraday's law: B_y = E0/c * cos(k*z - omega*t)
    B_y = E0 / c * sp.cos(k_val * z - omega_k * t)
    B_field = 0 * N.i + B_y * N.j + 0 * N.k

    # Verify div E = 0 (vacuum)
    div_E = divergence(E_field)
    print("div(E) =", sp.simplify(div_E))

    # Verify div B = 0
    div_B = divergence(B_field)
    print("div(B) =", sp.simplify(div_B))

    # Verify Faraday's law: curl E = -1/c * dB/dt
    curl_E = curl(E_field)
    dB_dt = (E0 / c) * omega_k * sp.sin(k_val * z - omega_k * t) * N.j
    faraday_lhs = curl_E
    faraday_rhs = -1/c * dB_dt
    print("\nFaraday's law verification:")
    print("  curl(E) =", faraday_lhs)
    print("  -1/c * dB/dt =", sp.simplify(faraday_rhs))
    # Verify they're equal when k = omega/c
    substituted = sp.simplify(
        sp.nsimplify(faraday_lhs.dot(N.j) - faraday_rhs.dot(N.j)).subs(k_val, omega_k / c)
    )
    print(f"  Difference (k=omega/c): {substituted}")

    # Dispersion relation: from Ampere's law, k^2 = omega^2/c^2
    print("\nDispersion relation for electromagnetic waves in vacuum:")
    print("  k = omega/c  =>  v_phase = omega/k = c")

    return E_field, B_field


if __name__ == "__main__":
    maxwell_equations_demo()
```

---

## 9. Complete Example A — Double Pendulum: Symbolic → Numerical

```python
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def double_pendulum_symbolic_to_numerical():
    """
    (a) Derive double pendulum EOM symbolically
    (b) Lambdify to fast numerical functions
    (c) Simulate and plot
    """
    t = sp.Symbol("t")
    g_sym, l_sym, m_sym = sp.symbols("g l m", positive=True)

    q1 = sp.Function("q1")(t)
    q2 = sp.Function("q2")(t)
    dq1 = q1.diff(t)
    dq2 = q2.diff(t)

    # Equal masses and lengths for simplicity: m1=m2=m, l1=l2=l
    # Positions
    x1 = l_sym * sp.sin(q1)
    y1 = -l_sym * sp.cos(q1)
    x2 = x1 + l_sym * sp.sin(q2)
    y2 = y1 - l_sym * sp.cos(q2)

    # Kinetic energy
    T = sp.Rational(1, 2) * m_sym * (
        (x1.diff(t))**2 + (y1.diff(t))**2 +
        (x2.diff(t))**2 + (y2.diff(t))**2
    )
    T = sp.trigsimp(sp.expand(T))

    # Potential energy
    V = m_sym * g_sym * (y1 + y2)

    L = sp.expand(T - V)

    # Euler-Lagrange for q1
    dL_dq1 = L.diff(q1)
    dL_ddq1 = L.diff(dq1)
    EL1 = sp.expand(dL_ddq1.diff(t) - dL_dq1)

    # Euler-Lagrange for q2
    dL_dq2 = L.diff(q2)
    dL_ddq2 = L.diff(dq2)
    EL2 = sp.expand(dL_ddq2.diff(t) - dL_dq2)

    # Extract second derivatives
    ddq1, ddq2 = sp.symbols("ddq1 ddq2")
    EL1_sub = EL1.subs({q1.diff(t, 2): ddq1, q2.diff(t, 2): ddq2})
    EL2_sub = EL2.subs({q1.diff(t, 2): ddq1, q2.diff(t, 2): ddq2})

    sol = sp.solve([EL1_sub, EL2_sub], [ddq1, ddq2])
    ddq1_expr = sol[ddq1]
    ddq2_expr = sol[ddq2]

    # Substitute state variables: q1→Q1, dq1→W1, q2→Q2, dq2→W2
    Q1, W1, Q2, W2 = sp.symbols("Q1 W1 Q2 W2", real=True)
    subs_dict = {q1: Q1, dq1: W1, q2: Q2, dq2: W2}
    ddq1_num = ddq1_expr.subs(subs_dict)
    ddq2_num = ddq2_expr.subs(subs_dict)

    # Numerical parameters
    g_val, l_val, m_val = 9.81, 1.0, 1.0
    param_subs = {g_sym: g_val, l_sym: l_val, m_sym: m_val}
    ddq1_np = sp.lambdify([Q1, W1, Q2, W2], ddq1_num.subs(param_subs), modules="numpy")
    ddq2_np = sp.lambdify([Q1, W1, Q2, W2], ddq2_num.subs(param_subs), modules="numpy")

    def ode_system(t_val, state):
        q1v, w1v, q2v, w2v = state
        return [
            w1v,
            ddq1_np(q1v, w1v, q2v, w2v),
            w2v,
            ddq2_np(q1v, w1v, q2v, w2v),
        ]

    # Simulate
    y0 = [np.pi / 2, 0.0, np.pi / 4, 0.0]
    t_span = (0, 20)
    t_eval = np.linspace(*t_span, 5000)

    sol_num = solve_ivp(ode_system, t_span, y0, method="DOP853",
                        t_eval=t_eval, rtol=1e-9, atol=1e-11)

    # Convert to Cartesian for animation-ready output
    q1_t = sol_num.y[0]
    q2_t = sol_num.y[2]
    x1_t = l_val * np.sin(q1_t)
    y1_t = -l_val * np.cos(q1_t)
    x2_t = x1_t + l_val * np.sin(q2_t)
    y2_t = y1_t - l_val * np.cos(q2_t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sol_num.t, q1_t, label="q₁(t)", lw=1)
    axes[0].plot(sol_num.t, q2_t, label="q₂(t)", lw=1, ls="--")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].legend()
    axes[0].set_title("Double Pendulum Angles")

    axes[1].plot(x2_t, y2_t, lw=0.4, alpha=0.7, color="steelblue")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x₂ (m)")
    axes[1].set_ylabel("y₂ (m)")
    axes[1].set_title("Trajectory of Bob 2")

    plt.tight_layout()
    plt.show()

    return sol_num


if __name__ == "__main__":
    double_pendulum_symbolic_to_numerical()
```

---

## 10. Complete Example B — Symbolic ODE Solve then Plot

```python
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def symbolic_then_numerical_ode():
    """
    (b) Solve a 2D system of ODEs symbolically, then compare with numerical solution.

    System (predator-prey linearization about fixed point):
      dx/dt = a*x - b*x*y  → linearized: dx/dt = a*x - b*y
      dy/dt = -c*y + d*x*y → linearized: dy/dt = d*x - c*y
    """
    t = sp.Symbol("t", positive=True)
    x = sp.Function("x")
    y = sp.Function("y")
    a, b, c, d = sp.symbols("a b c d", positive=True)

    # Linearized Lotka-Volterra
    eq1 = sp.Eq(x(t).diff(t), a * x(t) - b * y(t))
    eq2 = sp.Eq(y(t).diff(t), d * x(t) - c * y(t))

    print("Linearized Lotka-Volterra system:")
    print("  ", eq1)
    print("  ", eq2)

    # Solve symbolically
    system = [eq1, eq2]
    sol = sp.dsolve(system, [x(t), y(t)])
    print("\nSymbolic solution:")
    for s in sol:
        print("  ", s)

    # Numerical values
    params = {a: 0.5, b: 0.1, c: 0.5, d: 0.1}
    x_sol_sym = sol[0].rhs.subs(params)
    y_sol_sym = sol[1].rhs.subs(params)

    # Apply ICs numerically: x(0)=10, y(0)=5
    C1, C2, C3, C4 = sp.symbols("C1 C2 C3 C4")
    x0_val, y0_val = 10.0, 5.0
    ic_eqs = [
        sp.Eq(x_sol_sym.subs(t, 0), x0_val),
        sp.Eq(y_sol_sym.subs(t, 0), y0_val),
    ]
    # Try to extract constants (may depend on sympy version)
    try:
        const_sol = sp.solve(ic_eqs, [C1, C2, C3, C4])
        x_particular = x_sol_sym.subs(const_sol)
        y_particular = y_sol_sym.subs(const_sol)
        x_np = sp.lambdify(t, x_particular, modules="numpy")
        y_np = sp.lambdify(t, y_particular, modules="numpy")
        t_arr = np.linspace(0, 20, 1000)
        x_sym_vals = x_np(t_arr)
        y_sym_vals = y_np(t_arr)
        has_symbolic = True
    except Exception as e:
        print(f"Symbolic IC application failed: {e}")
        has_symbolic = False

    # Numerical solution for comparison
    def lotka_volterra_linear(t_val, state):
        xv, yv = state
        a_n, b_n, c_n, d_n = 0.5, 0.1, 0.5, 0.1
        return [a_n * xv - b_n * yv, d_n * xv - c_n * yv]

    t_arr = np.linspace(0, 20, 1000)
    sol_num = solve_ivp(lotka_volterra_linear, (0, 20), [x0_val, y0_val],
                        t_eval=t_arr, method="RK45", rtol=1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if has_symbolic:
        axes[0].plot(t_arr, x_sym_vals.real, "r--", lw=1.5, label="x(t) symbolic")
        axes[0].plot(t_arr, y_sym_vals.real, "b--", lw=1.5, label="y(t) symbolic")

    axes[0].plot(sol_num.t, sol_num.y[0], "r-", lw=1, alpha=0.7, label="x(t) numerical")
    axes[0].plot(sol_num.t, sol_num.y[1], "b-", lw=1, alpha=0.7, label="y(t) numerical")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Population")
    axes[0].legend()
    axes[0].set_title("Linearized Lotka-Volterra")

    axes[1].plot(sol_num.y[0], sol_num.y[1], color="steelblue", lw=1.2)
    axes[1].scatter([x0_val], [y0_val], color="red", zorder=5, label="Initial state")
    axes[1].set_xlabel("x (prey)")
    axes[1].set_ylabel("y (predator)")
    axes[1].legend()
    axes[1].set_title("Phase Portrait")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    symbolic_then_numerical_ode()
```

---

## Quick Reference

| Task                              | SymPy API                                          |
|-----------------------------------|----------------------------------------------------|
| Define symbol with assumptions    | `sp.Symbol("x", real=True, positive=True)`         |
| Expand expression                 | `sp.expand(expr)`                                  |
| Factor polynomial                 | `sp.factor(expr)`                                  |
| Simplify (general)                | `sp.simplify(expr)`                                |
| Trig simplification               | `sp.trigsimp(expr)`                                |
| Differentiate                     | `sp.diff(expr, var, n)`                            |
| Integrate (indefinite)            | `sp.integrate(expr, var)`                          |
| Integrate (definite)              | `sp.integrate(expr, (var, a, b))`                  |
| Power series                      | `sp.series(expr, var, x0, n)`                      |
| Solve algebraic equation          | `sp.solve(eq, var)`                                |
| Solve ODE                         | `sp.dsolve(ode, func, ics={})`                     |
| Matrix operations                 | `sp.Matrix([[...]]); M.det(), M.inv(), M.eigenvects()` |
| Convert to numpy                  | `sp.lambdify([vars], expr, modules="numpy")`       |
| Common subexpression elimination  | `sp.cse([expr1, expr2])`                           |
| LaTeX output                      | `sp.latex(expr)`                                   |
| Pretty print                      | `sp.pprint(expr, use_unicode=True)`                |

### Performance Tips

- Use `sp.lambdify` with `modules="numpy"` for vectorized numerical evaluation.
- Apply `sp.cse` before lambdifying multi-output expressions to avoid redundant computation.
- Set assumptions on symbols (e.g., `real=True`, `positive=True`) — SymPy simplifies more aggressively.
- Use `sp.nsimplify` to convert floating-point results back to exact rational/algebraic form.
- For large matrix operations, consider `sp.MatrixSymbol` for abstract manipulation.
