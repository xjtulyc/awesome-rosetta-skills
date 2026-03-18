---
name: pytorch-physics
description: "Physics-informed neural networks, neural ODEs, and data-driven force-field learning using PyTorch, DeepXDE, and torchdiffeq."
tags:
  - physics
  - pytorch
  - pinn
  - neural-ode
  - scientific-computing
  - deep-learning
version: "1.0.0"
authors:
  - name: "Rosetta Skills Contributors"
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - torch>=2.0
  - deepxde>=1.9
  - torchdiffeq>=0.2
  - numpy>=1.24
  - matplotlib>=3.7
last_updated: "2026-03-17"
status: stable
---

# PyTorch Physics — PINNs, Neural ODEs & Force-Field Learning

This skill covers the full workflow for applying deep learning to physical systems:
Physics-Informed Neural Networks (PINNs) that embed differential equations as loss terms,
Neural Ordinary Differential Equations (Neural ODEs) for continuous-time dynamics, and
data-driven force-field learning for molecular or mechanical simulations.

---

## When to Use This Skill

Use this skill when you need to:

- Solve forward or inverse problems governed by PDEs (heat, wave, Navier-Stokes, Schrödinger).
- Learn continuous-time dynamics from irregularly sampled trajectory data.
- Fit a differentiable force field from ab-initio or MD simulation data.
- Combine sparse observations with known physical priors to constrain model solutions.
- Benchmark neural surrogate models against finite-element or finite-difference baselines.

Do **not** use this skill for purely statistical forecasting where no physics is known, or
when a classical numerical solver is already fast enough and no generalization is required.

---

## Background & Key Concepts

### Physics-Informed Neural Networks (PINNs)

A PINN approximates the solution u(x, t) of a PDE with a neural network f_θ.
The training loss has three components:

```
L = L_data + λ_r * L_residual + λ_bc * L_boundary
```

- **L_data**: mean squared error on known measurements.
- **L_residual**: PDE residual evaluated at collocation points (via automatic differentiation).
- **L_boundary**: boundary and initial condition violations.

Because PyTorch tracks the full computation graph, ∂f/∂x and ∂²f/∂x² are computed
exactly with `torch.autograd.grad`, not finite differences.

### Neural ODEs

A Neural ODE replaces the discrete update rule of a ResNet with a continuous-time ODE:

```
dz/dt = f_θ(z, t)
```

The hidden state z(t) is integrated by a black-box ODE solver (e.g. Dormand-Prince RK45).
Gradients flow back through the solver via the adjoint method, keeping memory O(1).
`torchdiffeq` provides `odeint` and `odeint_adjoint` for this purpose.

### Data-Driven Force Fields

A force field maps atomic positions {r_i} to potential energy E and forces F = -∇E.
Training a neural network to predict E from a descriptor and back-propagating to get F
ensures energy conservation by construction. This is the foundation of NequIP, SchNet, etc.

### Key Autodiff Primitives in PyTorch

| Operation | Code |
|-----------|------|
| First derivative | `torch.autograd.grad(u, x, create_graph=True)` |
| Second derivative | nested `autograd.grad` calls |
| Jacobian | `torch.func.jacrev` (functorch) |
| Hessian | `torch.func.hessian` |

---

## Environment Setup

### Install dependencies

```bash
pip install torch>=2.0 deepxde>=1.9 torchdiffeq>=0.2 numpy>=1.24 matplotlib>=3.7
```

For GPU support (CUDA 12):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verify installation

```python
import torch
import deepxde as dde
import torchdiffeq

print("PyTorch:", torch.__version__)
print("DeepXDE:", dde.__version__)
print("CUDA available:", torch.cuda.is_available())
```

### Environment variables

```bash
# Optional: weights & biases logging
export WANDB_API_KEY="<paste-your-key>"
# Optional: custom data directory
export PHYSICS_DATA_DIR="/data/physics_datasets"
```

```python
import os

wandb_key = os.getenv("WANDB_API_KEY", "")
data_dir = os.getenv("PHYSICS_DATA_DIR", "./data")
```

### DeepXDE backend configuration

```bash
# DeepXDE supports multiple backends; set to pytorch
export DDE_BACKEND=pytorch
```

---

## Core Workflow

### Step 1 — Solve a 1-D Heat Equation with a PINN

The problem: ∂u/∂t = α ∂²u/∂x², x ∈ [0, 1], t ∈ [0, 1]
with u(x, 0) = sin(πx), u(0, t) = u(1, t) = 0.
Analytical solution: u(x, t) = exp(-α π² t) sin(πx).

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Hyper-parameters ────────────────────────────────────────────────────────
ALPHA = 0.01          # thermal diffusivity
N_COLLOC = 5000       # interior collocation points
N_BC = 200            # boundary / IC points
N_EPOCHS = 10000
LR = 1e-3
LAMBDA_R = 1.0        # residual weight
LAMBDA_BC = 10.0      # boundary weight
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Network ──────────────────────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self, layers=(2, 64, 64, 64, 1)):
        super().__init__()
        seq = []
        for i in range(len(layers) - 1):
            seq.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                seq.append(nn.Tanh())
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


def pde_residual(model, x, t, alpha=ALPHA):
    """Return PDE residual: u_t - alpha * u_xx."""
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    return u_t - alpha * u_xx


def sample_collocation(n):
    x = torch.rand(n, 1, device=DEVICE)
    t = torch.rand(n, 1, device=DEVICE)
    return x, t


def sample_boundary(n):
    """Sample IC (t=0) and Dirichlet BC (x=0, x=1)."""
    n3 = n // 3
    # Initial condition
    x_ic = torch.rand(n3, 1, device=DEVICE)
    t_ic = torch.zeros(n3, 1, device=DEVICE)
    u_ic = torch.sin(torch.pi * x_ic)
    # Left BC
    x_l = torch.zeros(n3, 1, device=DEVICE)
    t_l = torch.rand(n3, 1, device=DEVICE)
    u_l = torch.zeros(n3, 1, device=DEVICE)
    # Right BC
    x_r = torch.ones(n3, 1, device=DEVICE)
    t_r = torch.rand(n3, 1, device=DEVICE)
    u_r = torch.zeros(n3, 1, device=DEVICE)
    xb = torch.cat([x_ic, x_l, x_r])
    tb = torch.cat([t_ic, t_l, t_r])
    ub = torch.cat([u_ic, u_l, u_r])
    return xb, tb, ub


# ── Training loop ────────────────────────────────────────────────────────────
model = PINN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

losses = []
for epoch in range(N_EPOCHS):
    optimizer.zero_grad()
    # Residual loss
    xc, tc = sample_collocation(N_COLLOC)
    res = pde_residual(model, xc, tc)
    loss_r = LAMBDA_R * (res ** 2).mean()
    # Boundary loss
    xb, tb, ub = sample_boundary(N_BC)
    u_pred_b = model(xb, tb)
    loss_bc = LAMBDA_BC * ((u_pred_b - ub) ** 2).mean()
    loss = loss_r + loss_bc
    loss.backward()
    optimizer.step()
    scheduler.step()
    if epoch % 500 == 0:
        losses.append(loss.item())
        print(f"Epoch {epoch:5d} | loss={loss.item():.4e} "
              f"res={loss_r.item():.4e} bc={loss_bc.item():.4e}")

# ── Evaluation ───────────────────────────────────────────────────────────────
x_test = torch.linspace(0, 1, 100, device=DEVICE).unsqueeze(1)
t_test = torch.full_like(x_test, 0.5)
with torch.no_grad():
    u_pred = model(x_test, t_test).cpu().numpy().ravel()
u_exact = np.exp(-ALPHA * np.pi**2 * 0.5) * np.sin(np.pi * x_test.cpu().numpy().ravel())

plt.figure(figsize=(8, 4))
plt.plot(x_test.cpu().numpy(), u_exact, label="Exact", lw=2)
plt.plot(x_test.cpu().numpy(), u_pred, "--", label="PINN", lw=2)
plt.xlabel("x"); plt.ylabel("u(x, 0.5)")
plt.title("Heat equation — PINN vs exact at t=0.5")
plt.legend(); plt.tight_layout(); plt.savefig("heat_pinn.png", dpi=150)
print("L2 error:", np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact))
```

### Step 2 — Neural ODE for a Damped Oscillator

Fit a Neural ODE to noisy observations of a damped harmonic oscillator and then
integrate forward in time to forecast unseen states.

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Generate ground-truth data ───────────────────────────────────────────────
def true_dynamics(t, y):
    """Damped oscillator: y = [x, v], dy/dt = [v, -2ζω v - ω² x]."""
    omega, zeta = 2.0, 0.1
    x, v = y[..., 0:1], y[..., 1:2]
    dxdt = v
    dvdt = -2 * zeta * omega * v - omega**2 * x
    return torch.cat([dxdt, dvdt], dim=-1)

t_train = torch.linspace(0, 5, 60, device=DEVICE)
y0_true = torch.tensor([[1.0, 0.0]], device=DEVICE)

with torch.no_grad():
    y_true = odeint(true_dynamics, y0_true, t_train, method="rk4")  # (T, 1, 2)

noise = 0.05 * torch.randn_like(y_true)
y_obs = (y_true + noise).squeeze(1)  # (T, 2)

# ── Neural ODE model ─────────────────────────────────────────────────────────
class ODEFunc(nn.Module):
    def __init__(self, state_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),   nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, t, y):
        return self.net(y)


class NeuralODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = ODEFunc()

    def forward(self, y0, t_span):
        return odeint(self.func, y0, t_span, method="dopri5",
                      rtol=1e-4, atol=1e-6)


# ── Training ─────────────────────────────────────────────────────────────────
node = NeuralODE().to(DEVICE)
optimizer = torch.optim.Adam(node.parameters(), lr=5e-3)

y0_est = y_obs[0:1]  # use first observation as initial condition

for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = node(y0_est, t_train)          # (T, 1, 2)
    loss = ((y_pred.squeeze(1) - y_obs) ** 2).mean()
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | MSE={loss.item():.5f}")

# ── Forecasting ──────────────────────────────────────────────────────────────
t_fore = torch.linspace(0, 10, 120, device=DEVICE)
with torch.no_grad():
    y_fore = node(y0_est, t_fore).squeeze(1).cpu().numpy()

y_true_np = y_true.squeeze(1).cpu().numpy()
t_np = t_train.cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, label in enumerate(["position x", "velocity v"]):
    axes[i].scatter(t_np, y_obs[:, i].cpu().numpy(), s=10, label="observations")
    axes[i].plot(t_fore.cpu().numpy(), y_fore[:, i], label="Neural ODE forecast")
    axes[i].axvline(5, ls="--", color="gray", label="forecast boundary")
    axes[i].set_xlabel("time"); axes[i].set_ylabel(label); axes[i].legend()
plt.tight_layout(); plt.savefig("neural_ode_oscillator.png", dpi=150)
```

### Step 3 — Data-Driven Force Field with Energy Conservation

Train a simple equivariant-inspired force network where forces are derived from
a learned potential energy to ensure conservation.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Synthetic dataset: 2-body Lennard-Jones ──────────────────────────────────
def lj_potential(r, eps=1.0, sigma=1.0):
    """Lennard-Jones 6-12 potential energy."""
    sr6 = (sigma / r) ** 6
    return 4 * eps * (sr6**2 - sr6)


def generate_lj_data(n=2000):
    r = torch.FloatTensor(n, 1).uniform_(0.9, 3.0)
    E = lj_potential(r)
    F = -torch.autograd.functional.jacobian(
        lambda x: lj_potential(x).sum(), r
    ).squeeze()           # dE/dr; shape (n,)
    # pack as (n, 1) for convenient batching
    return r, E, F.unsqueeze(1)


r_data, E_data, F_data = generate_lj_data()
r_data = r_data.to(DEVICE)
E_data = E_data.to(DEVICE)
F_data = F_data.to(DEVICE)

# ── Potential network ─────────────────────────────────────────────────────────
class PotentialNet(nn.Module):
    """Predict scalar E(r); forces are F = -dE/dr computed at inference."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, r):
        return self.net(r)

    def energy_and_force(self, r):
        r = r.requires_grad_(True)
        E = self.net(r)
        F = -torch.autograd.grad(E.sum(), r, create_graph=True)[0]
        return E, F


net = PotentialNet().to(DEVICE)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

LAMBDA_F = 0.9    # weight for force matching
BATCH = 256

dataset = torch.utils.data.TensorDataset(r_data, E_data, F_data)
loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True)

for epoch in range(200):
    epoch_loss = 0.0
    for r_b, E_b, F_b in loader:
        optimizer.zero_grad()
        E_pred, F_pred = net.energy_and_force(r_b)
        loss_E = ((E_pred - E_b) ** 2).mean()
        loss_F = ((F_pred - F_b) ** 2).mean()
        loss = (1 - LAMBDA_F) * loss_E + LAMBDA_F * loss_F
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | loss={epoch_loss/len(loader):.5f}")

# ── Evaluation ────────────────────────────────────────────────────────────────
r_test = torch.linspace(0.95, 3.0, 200, device=DEVICE).unsqueeze(1)
with torch.no_grad():
    # Temporarily enable grad for force calculation
    pass

r_test.requires_grad_(True)
E_test, F_test = net.energy_and_force(r_test)
r_np = r_test.detach().cpu().numpy().ravel()
E_np = E_test.detach().cpu().numpy().ravel()
F_np = F_test.detach().cpu().numpy().ravel()
E_true = lj_potential(r_test.detach()).cpu().numpy().ravel()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(r_np, E_true, label="LJ exact"); ax1.plot(r_np, E_np, "--", label="NN")
ax1.set_ylim(-2, 5); ax1.set_xlabel("r"); ax1.set_ylabel("E"); ax1.legend()
ax2.plot(r_np, F_np, label="NN force")
ax2.set_xlabel("r"); ax2.set_ylabel("F = -dE/dr"); ax2.legend()
plt.tight_layout(); plt.savefig("force_field.png", dpi=150)
```

---

## Advanced Usage

### Inverse Problem — Identify PDE Coefficients

Given noisy measurements, identify the unknown parameter α in u_t = α u_xx.

```python
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InversePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
        # alpha is a learnable parameter
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], 1))

    def residual(self, x, t):
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self(x, t)
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        return u_t - self.alpha * u_xx


def train_inverse(alpha_true=0.02, n_data=200, n_colloc=3000, epochs=8000):
    model = InversePINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate synthetic noisy data from analytical solution
    import numpy as np
    x_d = torch.rand(n_data, 1, device=DEVICE)
    t_d = torch.rand(n_data, 1, device=DEVICE)
    u_d = (torch.exp(torch.tensor(-alpha_true) * np.pi**2 * t_d)
           * torch.sin(np.pi * x_d)
           + 0.01 * torch.randn(n_data, 1, device=DEVICE))

    for ep in range(epochs):
        optimizer.zero_grad()
        # Data loss
        u_pred = model(x_d, t_d)
        loss_data = ((u_pred - u_d) ** 2).mean()
        # PDE residual loss
        xc = torch.rand(n_colloc, 1, device=DEVICE)
        tc = torch.rand(n_colloc, 1, device=DEVICE)
        res = model.residual(xc, tc)
        loss_res = (res ** 2).mean()
        loss = loss_data + loss_res
        loss.backward()
        optimizer.step()
        if ep % 1000 == 0:
            print(f"Epoch {ep:5d} | loss={loss.item():.4e} "
                  f"alpha_est={model.alpha.item():.5f} (true={alpha_true})")

    return model


trained = train_inverse()
print("Final alpha estimate:", trained.alpha.item())
```

### Using DeepXDE for Burgers' Equation

DeepXDE provides a high-level API for common PDE problem types.

```python
import deepxde as dde
import numpy as np
import os

# Set backend
os.environ.setdefault("DDE_BACKEND", "pytorch")

# Burgers' equation: u_t + u * u_x = nu * u_xx
nu = 0.01 / np.pi

def burgers_pde(x, y):
    """x: (N,2) [x_coord, t]; y: (N,1) [u]"""
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - nu * dy_xx


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_b: on_b)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]),
                 lambda x, on_i: on_i)

data = dde.data.TimePDE(
    geomtime, burgers_pde, [bc, ic],
    num_domain=2500, num_boundary=100, num_initial=200,
)

net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=15000)
dde.saveplot(losshistory, train_state, issave=False, isplot=True)
```

### Hamiltonian Neural Networks (HNN)

Conserve energy exactly by parameterizing the Hamiltonian H(q, p).

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class HamiltonianNet(nn.Module):
    """Learn H(q, p) and derive canonical equations from it."""
    def __init__(self, dim=1, hidden=128):
        super().__init__()
        self.H = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.dim = dim

    def hamiltonian(self, qp):
        return self.H(qp)

    def forward(self, t, qp):
        """Returns dq/dt = dH/dp,  dp/dt = -dH/dq."""
        qp = qp.requires_grad_(True)
        H_val = self.H(qp).sum()
        grad = torch.autograd.grad(H_val, qp, create_graph=True)[0]
        dqdt =  grad[..., self.dim:]    # dH/dp
        dpdt = -grad[..., :self.dim]    # -dH/dq
        return torch.cat([dqdt, dpdt], dim=-1)


def train_hnn(n_traj=50, t_span=torch.linspace(0, 2, 40)):
    """Train on simple pendulum trajectories."""
    model = HamiltonianNet(dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_span = t_span.to(DEVICE)

    # Generate ground truth pendulum: H = 0.5 p^2 - cos(q)
    def pendulum(t, qp):
        q, p = qp[..., 0:1], qp[..., 1:2]
        return torch.cat([p, -torch.sin(q)], dim=-1)

    for step in range(3000):
        # Random initial conditions
        q0 = torch.FloatTensor(1, 1).uniform_(-2, 2).to(DEVICE)
        p0 = torch.FloatTensor(1, 1).uniform_(-2, 2).to(DEVICE)
        qp0 = torch.cat([q0, p0], dim=-1)
        with torch.no_grad():
            qp_true = odeint(pendulum, qp0, t_span, method="rk4")

        qp_pred = odeint(model, qp0, t_span, method="rk4")
        loss = ((qp_pred - qp_true) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 300 == 0:
            print(f"Step {step:4d} | loss={loss.item():.5f}")

    return model
```

---

## Troubleshooting

### Gradients are None or zero through the ODE solver

**Symptom**: `loss.backward()` raises a warning and gradients are zero.

**Cause**: Using `odeint` (non-adjoint) with too many time steps can exhaust memory
or detach the graph.

**Fix**: Switch to `odeint_adjoint` which uses the adjoint method and does not keep
the full trajectory in memory.

```python
from torchdiffeq import odeint_adjoint as odeint
# use exactly the same call signature as odeint
```

### PINN loss not converging (residual stays large)

**Symptom**: `loss_r` oscillates and does not decrease after several thousand epochs.

**Cause 1**: Learning rate too high — the residual term involves second derivatives
which amplify gradient magnitudes.

**Fix**: Lower LR to `1e-4` and add gradient clipping.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Cause 2**: Collocation points are not covering the domain evenly.

**Fix**: Use Latin-hypercube sampling instead of pure random.

```python
from scipy.stats.qmc import LatinHypercube

sampler = LatinHypercube(d=2)
pts = torch.tensor(sampler.random(n=N_COLLOC), dtype=torch.float32, device=DEVICE)
xc, tc = pts[:, 0:1], pts[:, 1:2]
```

### DeepXDE backend not loading PyTorch

**Symptom**: `ImportError` or wrong backend active.

**Fix**:

```bash
export DDE_BACKEND=pytorch
python -c "import deepxde; print(deepxde.backend.backend_name)"
```

### NaN loss with force-matching

**Symptom**: Loss becomes NaN after a few iterations when using `create_graph=True`.

**Cause**: Exploding gradients through the double backward pass.

**Fix**: Add gradient clipping and ensure input distances are bounded away from zero.

```python
r = torch.clamp(r, min=0.8)   # avoid singularity at r → 0
torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
```

---

## External Resources

- PyTorch Autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html
- DeepXDE documentation: https://deepxde.readthedocs.io/
- torchdiffeq paper (Chen et al. 2018): https://arxiv.org/abs/1806.07366
- Physics-Informed Neural Networks original paper (Raissi et al. 2019): https://doi.org/10.1016/j.jcp.2018.10.045
- Hamiltonian Neural Networks (Greydanus et al. 2019): https://arxiv.org/abs/1906.01563
- NequIP (Batzner et al. 2022): https://www.nature.com/articles/s41467-022-29939-5
- PyTorch Functorch (vmap, jacrev): https://pytorch.org/functorch/stable/

---

## Examples

### Example 1 — Full Pipeline: Navier-Stokes 2-D Lid-Driven Cavity

Solve the steady incompressible Navier-Stokes equations inside a unit square cavity
where the top wall moves at velocity U=1.  Re = 100.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RE = 100.0

class NSNet(nn.Module):
    """Predict (u, v, p) from (x, y)."""
    def __init__(self, hidden=128, depth=6):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)


def ns_residuals(model, xy):
    xy = xy.requires_grad_(True)
    uvp = model(xy)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]

    def grad1(f, x):
        return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]

    uvp_x = grad1(uvp, xy)[:, 0:1], grad1(uvp, xy)[:, 1:2], grad1(uvp, xy)[:, 2:3]

    # recompute per-component
    u_xy = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    v_xy = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0]
    p_xy = torch.autograd.grad(p, xy, torch.ones_like(p), create_graph=True)[0]

    ux, uy = u_xy[:, 0:1], u_xy[:, 1:2]
    vx, vy = v_xy[:, 0:1], v_xy[:, 1:2]
    px, py = p_xy[:, 0:1], p_xy[:, 1:2]

    uxx = torch.autograd.grad(ux, xy, torch.ones_like(ux), create_graph=True)[0][:, 0:1]
    uyy = torch.autograd.grad(uy, xy, torch.ones_like(uy), create_graph=True)[0][:, 1:2]
    vxx = torch.autograd.grad(vx, xy, torch.ones_like(vx), create_graph=True)[0][:, 0:1]
    vyy = torch.autograd.grad(vy, xy, torch.ones_like(vy), create_graph=True)[0][:, 1:2]

    # Continuity
    cont = ux + vy
    # Momentum x
    mom_x = u * ux + v * uy + px - (uxx + uyy) / RE
    # Momentum y
    mom_y = u * vx + v * vy + py - (vxx + vyy) / RE
    return cont, mom_x, mom_y


def sample_boundary_ns(n):
    """Return (xy, u_target, v_target) for all four walls."""
    n4 = n // 4
    # Bottom y=0: u=v=0
    bot = torch.cat([torch.rand(n4, 1), torch.zeros(n4, 1)], 1).to(DEVICE)
    # Top y=1: u=1, v=0
    top = torch.cat([torch.rand(n4, 1), torch.ones(n4, 1)], 1).to(DEVICE)
    # Left x=0: u=v=0
    left = torch.cat([torch.zeros(n4, 1), torch.rand(n4, 1)], 1).to(DEVICE)
    # Right x=1: u=v=0
    right = torch.cat([torch.ones(n4, 1), torch.rand(n4, 1)], 1).to(DEVICE)
    xy_bc = torch.cat([bot, top, left, right])
    u_bc = torch.zeros(4 * n4, 1, device=DEVICE)
    u_bc[n4:2*n4] = 1.0   # top wall
    v_bc = torch.zeros(4 * n4, 1, device=DEVICE)
    return xy_bc, u_bc, v_bc


model = NSNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    optimizer.zero_grad()
    # Interior residuals
    xy_int = torch.rand(2000, 2, device=DEVICE)
    cont, mom_x, mom_y = ns_residuals(model, xy_int)
    loss_pde = (cont**2 + mom_x**2 + mom_y**2).mean()
    # Boundary
    xy_bc, u_bc, v_bc = sample_boundary_ns(400)
    uvp_bc = model(xy_bc)
    loss_bc = ((uvp_bc[:, 0:1] - u_bc)**2 + (uvp_bc[:, 1:2] - v_bc)**2).mean()
    loss = loss_pde + 10 * loss_bc
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | total={loss.item():.4e}")

# Visualise u-velocity field
nx = 50
x_ = torch.linspace(0, 1, nx); y_ = torch.linspace(0, 1, nx)
X, Y = torch.meshgrid(x_, y_, indexing="ij")
xy_grid = torch.stack([X.ravel(), Y.ravel()], 1).to(DEVICE)
with torch.no_grad():
    uvp_grid = model(xy_grid).cpu().numpy()
U = uvp_grid[:, 0].reshape(nx, nx)
plt.figure(figsize=(6, 5))
plt.contourf(X.numpy(), Y.numpy(), U, levels=30, cmap="RdBu_r")
plt.colorbar(label="u velocity")
plt.title(f"Lid-driven cavity Re={RE} — PINN u-field")
plt.tight_layout(); plt.savefig("ns_cavity.png", dpi=150)
```

### Example 2 — Latent Neural ODE for Irregular Time Series

Encode a noisy trajectory with an RNN, decode the latent ODE, and reconstruct
the full trajectory including missing segments.

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 8
OBS_DIM = 2

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, t, z):
        return self.net(z)


class Encoder(nn.Module):
    """GRU encoder: maps a variable-length observation sequence to z0."""
    def __init__(self, obs_dim=OBS_DIM, hidden=32, latent_dim=LATENT_DIM):
        super().__init__()
        self.gru = nn.GRU(obs_dim + 1, hidden, batch_first=True)
        self.out = nn.Linear(hidden, latent_dim * 2)  # mean + log-var

    def forward(self, obs, times):
        # Concatenate time deltas to observations
        dt = torch.diff(times, prepend=times[:, :1], dim=1).unsqueeze(-1)
        inp = torch.cat([obs, dt], dim=-1)
        _, h = self.gru(inp)
        params = self.out(h.squeeze(0))
        mu, log_var = params.chunk(2, dim=-1)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, obs_dim=OBS_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


class LatentODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder  = Encoder()
        self.ode_func = LatentODEFunc()
        self.decoder  = Decoder()

    def forward(self, obs, t_obs, t_pred):
        mu, log_var = self.encoder(obs, t_obs)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z0 = mu + eps * std   # reparameterization

        # Integrate latent ODE from t=0 to all prediction times
        all_times = torch.cat([t_pred[0:1], t_pred]).unique(sorted=True)
        z_traj = odeint(self.ode_func, z0, all_times, method="dopri5")
        x_pred = self.decoder(z_traj)

        # KL divergence
        kl = -0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=-1).mean()
        return x_pred, kl


# Quick sanity-check forward pass
model = LatentODE().to(DEVICE)
T, B = 30, 4
obs_dummy = torch.randn(B, T, OBS_DIM, device=DEVICE)
t_obs_dummy = torch.linspace(0, 3, T, device=DEVICE).unsqueeze(0).expand(B, -1)
t_pred_dummy = torch.linspace(0, 5, 50, device=DEVICE)

x_pred, kl = model(obs_dummy, t_obs_dummy, t_pred_dummy)
print("x_pred shape:", x_pred.shape)   # (50, B, OBS_DIM)
print("KL:", kl.item())
```
