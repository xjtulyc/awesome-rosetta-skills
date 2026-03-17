---
name: optimization-or
description: >
  Use this Skill for operations research and optimization: LP/QP/MILP with cvxpy and
  OR-Tools, convex relaxation, sensitivity analysis, and Gurobi/GLPK solver interface.
tags:
  - mathematics
  - optimization
  - OR-Tools
  - cvxpy
  - linear-programming
  - MILP
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
    - cvxpy>=1.4
    - ortools>=9.7
    - scipy>=1.9
    - numpy>=1.23
    - matplotlib>=3.6
last_updated: "2026-03-17"
status: stable
---

# Operations Research and Mathematical Optimization

> **TL;DR** — Formulate and solve LP, QP, and MILP problems using cvxpy and OR-Tools.
> Covers standard form, dual variables / shadow prices, sensitivity analysis, solver
> selection (GLPK, ECOS, SCS, Gurobi), OR-Tools CP-SAT, and portfolio optimization.

---

## When to Use

Use this Skill when you need to:

- Solve resource allocation, scheduling, or routing problems as linear programs (LP).
- Handle binary / integer decisions (facility location, knapsack, assignment) via MILP.
- Optimize a quadratic objective (portfolio variance, regression) via QP.
- Perform post-solve sensitivity analysis: shadow prices, reduced costs, allowable ranges.
- Switch between open-source solvers (GLPK, ECOS, SCS) and commercial ones (Gurobi).
- Model constraint satisfaction problems with OR-Tools CP-SAT.

| Problem class | Recommended tool |
|---|---|
| LP (continuous) | cvxpy + GLPK / ECOS |
| QP (quadratic objective) | cvxpy + OSQP / ECOS |
| MILP (integer variables) | cvxpy + GLPK_MI, OR-Tools CP-SAT |
| Large-scale LP/MILP | OR-Tools linear solver (pywraplp) |
| Commercial (fastest) | Gurobi via cvxpy |

---

## Background & Key Concepts

### Standard Form LP

A linear program in standard form:

```
minimize    c^T x
subject to  A_eq x  = b_eq
            A_ub x <= b_ub
            x >= 0
```

The **dual problem** attaches a multiplier (shadow price) to each constraint.
Shadow price = marginal value of relaxing that constraint by one unit.

### Mixed-Integer LP (MILP)

Replace some variables with `cp.Variable(integer=True)` or `cp.Variable(boolean=True)`.
Solved by branch-and-bound: the LP relaxation at each node is solved and branched on
fractional integer variables.

### Quadratic Programming (QP)

Markowitz portfolio optimization:

```
minimize    (1/2) w^T Sigma w   (portfolio variance)
subject to  mu^T w >= r_min     (return floor)
            1^T w  = 1          (fully invested)
            w >= 0              (long-only)
```

### Convex Relaxation

MILP is NP-hard in general. Convex (LP) relaxation of integer constraints provides
a lower bound and guides branch-and-bound. Tight relaxations lead to fast solves.

---

## Environment Setup

```bash
# Create a dedicated environment
conda create -n opt python=3.11 -y
conda activate opt

# Core packages
pip install cvxpy>=1.4 ortools>=9.7 scipy>=1.9 numpy>=1.23 matplotlib>=3.6

# Optional: Gurobi (requires a license — free academic license available)
# pip install gurobipy
# export GUROBI_LICENSE_FILE="/path/to/gurobi.lic"

# Verify installation
python -c "import cvxpy; import ortools; print('cvxpy', cvxpy.__version__)"
```

Solver availability check:

```python
import cvxpy as cp

# List all solvers cvxpy can find on this machine
print("Available solvers:", cp.installed_solvers())
# Expected output includes: ['CLARABEL', 'ECOS', 'ECOS_BB', 'GLPK', 'GLPK_MI',
#                            'OSQP', 'SCS', 'SCIPY']
# Gurobi will appear only if gurobipy is installed and licensed.
```

---

## Core Workflow

### Step 1 — Resource Allocation LP with cvxpy

Classic production planning: allocate limited resources across products to maximise profit.

```python
import cvxpy as cp
import numpy as np

# ── Problem data ────────────────────────────────────────────────────────────────
# 4 products, 3 resources
n_products = 4
n_resources = 3

# Profit per unit of each product
profit = np.array([25.0, 30.0, 15.0, 20.0])

# Resource consumption matrix A[i, j] = units of resource i per unit of product j
A = np.array([
    [1.0, 2.0, 1.0, 3.0],   # Labour (hours)
    [3.0, 1.0, 2.0, 1.0],   # Material (kg)
    [2.0, 2.0, 1.0, 2.0],   # Machine time (hrs)
])

# Available resource capacities
b = np.array([240.0, 300.0, 200.0])

# ── Decision variables ──────────────────────────────────────────────────────────
x = cp.Variable(n_products, name="production")

# ── Objective ───────────────────────────────────────────────────────────────────
objective = cp.Maximize(profit @ x)

# ── Constraints ─────────────────────────────────────────────────────────────────
constraints = [
    A @ x <= b,          # Resource capacity
    x >= 0,              # Non-negativity
]

# ── Solve ────────────────────────────────────────────────────────────────────────
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GLPK)

print(f"Status        : {problem.status}")
print(f"Optimal profit: {problem.value:.2f}")
print("Production plan:")
for j, xj in enumerate(x.value):
    print(f"  Product {j+1}: {xj:.3f} units")

# ── Dual variables (shadow prices) ──────────────────────────────────────────────
# Each shadow price = marginal profit gain per extra unit of that resource
shadow_prices = constraints[0].dual_value
resource_names = ["Labour", "Material", "Machine time"]
print("\nShadow prices (marginal value of relaxing each constraint):")
for name, lam in zip(resource_names, shadow_prices):
    print(f"  {name}: {lam:.4f}")
```

### Step 2 — Facility Location MILP with OR-Tools CP-SAT

Decide which warehouses to open and which customers to assign to minimise total cost.

```python
from ortools.sat.python import cp_model
import numpy as np

np.random.seed(0)

# ── Problem data ────────────────────────────────────────────────────────────────
n_facilities = 5   # candidate warehouse sites
n_customers  = 10  # demand points

# Fixed opening cost per facility
fixed_cost = np.array([100, 120, 80, 150, 90], dtype=int)

# Shipping cost[i][j] = cost to serve customer j from facility i
ship_cost = np.random.randint(5, 30, size=(n_facilities, n_customers))

# Demand (units) per customer
demand = np.random.randint(10, 50, size=n_customers)

# Capacity per facility
capacity = np.array([200, 180, 220, 160, 240], dtype=int)

# ── Scale to integers (CP-SAT requires integer coefficients) ────────────────────
SCALE = 1  # already integer; increase if using fractional costs

# ── Model ────────────────────────────────────────────────────────────────────────
model = cp_model.CpModel()

# y[i] = 1 if facility i is opened
y = [model.NewBoolVar(f"y_{i}") for i in range(n_facilities)]

# x[i][j] = fraction of customer j's demand served from facility i (scaled to int)
# Here we use full-integer assignment: x[i][j] in {0,1} (each customer assigned to 1 fac)
x = [[model.NewBoolVar(f"x_{i}_{j}") for j in range(n_customers)]
     for i in range(n_facilities)]

# ── Constraints ──────────────────────────────────────────────────────────────────
# 1. Each customer assigned to exactly one facility
for j in range(n_customers):
    model.Add(sum(x[i][j] for i in range(n_facilities)) == 1)

# 2. Customers can only be assigned to open facilities
for i in range(n_facilities):
    for j in range(n_customers):
        model.Add(x[i][j] <= y[i])

# 3. Capacity constraints
for i in range(n_facilities):
    model.Add(
        sum(x[i][j] * int(demand[j]) for j in range(n_customers)) <= int(capacity[i])
    )

# ── Objective: minimise fixed + shipping cost ─────────────────────────────────
total_cost = (
    sum(int(fixed_cost[i]) * y[i] for i in range(n_facilities))
    + sum(int(ship_cost[i][j]) * x[i][j]
          for i in range(n_facilities)
          for j in range(n_customers))
)
model.Minimize(total_cost)

# ── Solve ─────────────────────────────────────────────────────────────────────
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30.0
status = solver.Solve(model)

status_name = solver.StatusName(status)
print(f"Solver status : {status_name}")
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print(f"Total cost    : {solver.ObjectiveValue():.0f}")
    open_facs = [i for i in range(n_facilities) if solver.Value(y[i]) == 1]
    print(f"Open facilities: {open_facs}")
    for j in range(n_customers):
        assigned = next(i for i in range(n_facilities) if solver.Value(x[i][j]) == 1)
        print(f"  Customer {j:2d} -> Facility {assigned}")
```

### Step 3 — Portfolio QP (Markowitz) with cvxpy

```python
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Synthetic market data ────────────────────────────────────────────────────────
n_assets = 8
n_days   = 252  # one year of daily returns

# Random expected returns and covariance matrix
mu_true  = np.random.uniform(0.0005, 0.002, n_assets)   # daily expected return
# Build a random positive-definite covariance matrix
F = np.random.randn(n_assets, n_assets) * 0.01
Sigma = F @ F.T + np.diag(np.random.uniform(0.0001, 0.001, n_assets))

# ── Efficient frontier ──────────────────────────────────────────────────────────
w = cp.Variable(n_assets, name="weights")

r_targets = np.linspace(mu_true.min(), mu_true.max(), 40)
frontier_risk    = []
frontier_return  = []
frontier_weights = []

for r_target in r_targets:
    objective   = cp.Minimize(cp.quad_form(w, Sigma))
    constraints = [
        mu_true @ w >= r_target,   # return floor
        cp.sum(w)    == 1,          # fully invested
        w            >= 0,          # long-only
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        frontier_risk.append(float(cp.sqrt(cp.quad_form(w, Sigma)).value))
        frontier_return.append(float(mu_true @ w.value))
        frontier_weights.append(w.value.copy())

# ── Maximum Sharpe ratio portfolio ──────────────────────────────────────────────
rf = 0.0001  # daily risk-free rate
sharpe_ratios = [
    (ret - rf) / risk if risk > 0 else -np.inf
    for ret, risk in zip(frontier_return, frontier_risk)
]
best_idx = int(np.argmax(sharpe_ratios))
print(f"Max-Sharpe portfolio:")
print(f"  Daily return : {frontier_return[best_idx]*252:.4f} (annualised)")
print(f"  Daily vol    : {frontier_risk[best_idx]*np.sqrt(252):.4f} (annualised)")
print(f"  Sharpe ratio : {sharpe_ratios[best_idx]*np.sqrt(252):.4f} (annualised)")
print("  Weights:", np.round(frontier_weights[best_idx], 4))

# ── Plot efficient frontier ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    [r * np.sqrt(252) for r in frontier_risk],
    [r * 252 for r in frontier_return],
    "b-o", markersize=3, label="Efficient frontier",
)
ax.scatter(
    frontier_risk[best_idx] * np.sqrt(252),
    frontier_return[best_idx] * 252,
    color="red", zorder=5, s=80, label="Max Sharpe",
)
ax.set_xlabel("Annualised volatility")
ax.set_ylabel("Annualised expected return")
ax.set_title("Markowitz Efficient Frontier")
ax.legend()
fig.tight_layout()
fig.savefig("efficient_frontier.png", dpi=150)
print("Saved efficient_frontier.png")
```

---

## Advanced Usage

### Sensitivity Analysis with scipy.optimize.linprog

```python
from scipy.optimize import linprog
import numpy as np

# Minimise  c^T x
# s.t.      A_ub x <= b_ub
#           lb <= x <= ub

c      = np.array([-25.0, -30.0, -15.0, -20.0])   # negate for maximisation
A_ub   = np.array([
    [1, 2, 1, 3],
    [3, 1, 2, 1],
    [2, 2, 1, 2],
])
b_ub   = np.array([240.0, 300.0, 200.0])
bounds = [(0, None)] * 4

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(f"scipy linprog optimal value: {-result.fun:.2f}")
print(f"Solution: {result.x}")

# Sensitivity information (available with HiGHS solver)
if hasattr(result, "ineqlin"):
    print("\nShadow prices (HiGHS marginals):", result.ineqlin.marginals)
    print("Slack values:", result.ineqlin.residual)
```

### Solver Selection Guide and Gurobi Interface

```python
import cvxpy as cp
import numpy as np

# ── Build a simple LP ────────────────────────────────────────────────────────────
n = 100
np.random.seed(7)
c      = np.random.randn(n)
A      = np.random.randn(50, n)
b      = np.abs(A).sum(axis=1)

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= 0])

# ── Try different solvers ────────────────────────────────────────────────────────
solver_map = {
    "GLPK"     : cp.GLPK,
    "ECOS"     : cp.ECOS,
    "SCS"      : cp.SCS,
    "CLARABEL" : cp.CLARABEL,
    # "GUROBI" : cp.GUROBI,   # uncomment if Gurobi is installed and licensed
}

for name, solver in solver_map.items():
    try:
        prob.solve(solver=solver, warm_start=True)
        status = prob.status
        val    = prob.value if prob.value is not None else float("nan")
        print(f"  {name:12s}: status={status:8s}  value={val:.6f}")
    except cp.SolverError as e:
        print(f"  {name:12s}: UNAVAILABLE ({e})")
```

### OR-Tools pywraplp Linear Solver

```python
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver.CreateSolver("GLOP")  # "GLOP" = Google LP, "CBC" = MILP

# Variables
x1 = solver.NumVar(0.0, solver.infinity(), "x1")
x2 = solver.NumVar(0.0, solver.infinity(), "x2")

# Constraints
c1 = solver.Constraint(-solver.infinity(), 14.0, "c1")
c1.SetCoefficient(x1, 1.0)
c1.SetCoefficient(x2, 2.0)

c2 = solver.Constraint(-solver.infinity(), 14.0, "c2")
c2.SetCoefficient(x1, 3.0)
c2.SetCoefficient(x2, 1.0)

# Objective: maximise 3x1 + 5x2
obj = solver.Objective()
obj.SetCoefficient(x1, 3.0)
obj.SetCoefficient(x2, 5.0)
obj.SetMaximization()

status = solver.Solve()
if status == pywraplp.Solver.OPTIMAL:
    print(f"x1 = {x1.solution_value():.4f}")
    print(f"x2 = {x2.solution_value():.4f}")
    print(f"Objective = {solver.Objective().Value():.4f}")
    print(f"Shadow price c1 = {c1.dual_value():.4f}")
    print(f"Shadow price c2 = {c2.dual_value():.4f}")
```

---

## Troubleshooting

| Error / Symptom | Cause | Fix |
|---|---|---|
| `problem.status == "infeasible"` | Constraints are contradictory | Plot feasible region; check sign conventions; add slack variables to identify binding constraint |
| `problem.status == "unbounded"` | Objective can decrease infinitely | Add upper/lower bounds on variables |
| `SolverError: GLPK is not installed` | GLPK binary missing | `conda install -c conda-forge glpk` or use ECOS/CLARABEL |
| Slow MILP solve (branch-and-bound) | Large search tree | Tighten LP relaxation; add valid inequalities; use warm start; increase time limit |
| QP not converging with ECOS | Near-singular covariance matrix | Add regularisation: `Sigma += 1e-6 * np.eye(n)` |
| `cp.quad_form` gives non-PSD error | Covariance matrix not positive semi-definite | Use `np.linalg.eigvalsh` to check; add ridge regularisation |
| OR-Tools CP-SAT: status UNKNOWN | Time limit hit before optimal | Increase `solver.parameters.max_time_in_seconds`; use `FEASIBLE` solution |

---

## External Resources

- cvxpy documentation: <https://www.cvxpy.org/>
- OR-Tools documentation: <https://developers.google.com/optimization>
- Boyd & Vandenberghe, *Convex Optimization* (free PDF): <https://web.stanford.edu/~boyd/cvxbook/>
- Gurobi academic license: <https://www.gurobi.com/academia/>
- scipy.optimize.linprog: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html>
- CVXPY disciplined convex programming rules: <https://www.cvxpy.org/tutorial/dcp/index.html>

---

## Examples

### Example 1 — Diet Problem LP

```python
import cvxpy as cp
import numpy as np

# ── Nutritional requirements (daily minimums) ───────────────────────────────────
# Nutrients: calories, protein(g), fat(g), carbs(g), iron(mg)
min_nutrients = np.array([2000, 50, 20, 300, 8])
max_nutrients = np.array([2500, 200, 70, 500, 20])

# Foods: bread, milk, cheese, potato, fish, maize
n_foods = 6
costs = np.array([0.6, 0.35, 1.50, 0.25, 2.10, 0.15])  # cost per 100g serving

# Nutrient content per 100g serving [food x nutrient]
nutrient_content = np.array([
    [250, 4,   1,  50, 1.5],   # bread
    [61,  3,   3,   5, 0.1],   # milk
    [400, 25,  33,  0, 0.2],   # cheese
    [80,  2,   0,  18, 0.4],   # potato
    [180, 22,  8,   0, 1.2],   # fish
    [360, 9,   2,  75, 2.0],   # maize
])

# ── Decision: servings (100g units) of each food per day ────────────────────────
x = cp.Variable(n_foods, name="servings")

objective   = cp.Minimize(costs @ x)
constraints = [
    nutrient_content.T @ x >= min_nutrients,
    nutrient_content.T @ x <= max_nutrients,
    x >= 0,
    x <= 10,   # max 10 servings of any single food
]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS)

food_names = ["Bread", "Milk", "Cheese", "Potato", "Fish", "Maize"]
print(f"Optimal daily cost: ${prob.value:.2f}")
print("Optimal diet:")
for name, srv in zip(food_names, x.value):
    if srv > 0.01:
        print(f"  {name:8s}: {srv*100:.0f} g")

total_nutrients = nutrient_content.T @ x.value
nut_names = ["Calories", "Protein(g)", "Fat(g)", "Carbs(g)", "Iron(mg)"]
print("\nNutritional summary:")
for n, v, mn in zip(nut_names, total_nutrients, min_nutrients):
    print(f"  {n:12s}: {v:.1f}  (min: {mn})")
```

### Example 2 — Knapsack MILP with cvxpy

```python
import cvxpy as cp
import numpy as np

# ── 0-1 Knapsack problem ────────────────────────────────────────────────────────
np.random.seed(3)
n_items  = 20
values   = np.random.randint(10, 100, n_items).astype(float)
weights  = np.random.randint(5, 30,  n_items).astype(float)
capacity = 100.0

# ── Binary decision variables ────────────────────────────────────────────────────
x = cp.Variable(n_items, boolean=True)

objective   = cp.Maximize(values @ x)
constraints = [weights @ x <= capacity]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GLPK_MI)

print(f"Knapsack status : {prob.status}")
print(f"Total value     : {prob.value:.0f}")
print(f"Total weight    : {weights @ x.value:.0f} / {capacity:.0f}")
selected = np.where(x.value > 0.5)[0]
print(f"Selected items  : {selected.tolist()}")
print(f"Item values     : {values[selected].tolist()}")
print(f"Item weights    : {weights[selected].tolist()}")

# ── LP relaxation bound (upper bound) ───────────────────────────────────────────
x_rel = cp.Variable(n_items)
prob_rel = cp.Problem(cp.Maximize(values @ x_rel),
                      [weights @ x_rel <= capacity, x_rel >= 0, x_rel <= 1])
prob_rel.solve(solver=cp.ECOS)
print(f"\nLP relaxation upper bound: {prob_rel.value:.2f}")
print(f"Integrality gap          : {prob_rel.value - prob.value:.2f}")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — LP, MILP, QP, cvxpy, OR-Tools, sensitivity analysis |
