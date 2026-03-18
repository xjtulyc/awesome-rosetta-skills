---
name: agent-based-social
description: >
  Use this Skill for agent-based social modeling with Mesa: opinion dynamics,
  segregation, social contagion, and network diffusion simulations.
tags:
  - sociology
  - agent-based-model
  - mesa
  - simulation
  - social-dynamics
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
    - mesa>=2.3
    - numpy>=1.24
    - pandas>=2.0
    - networkx>=3.2
    - matplotlib>=3.7
last_updated: "2026-03-17"
status: "stable"
---

# Agent-Based Social Modeling with Mesa

> **One-line summary**: Build and analyze agent-based social models using Mesa: Schelling segregation, opinion dynamics, social contagion (SIR), and network diffusion with parameter sweeps.

---

## When to Use This Skill

- When simulating emergent social phenomena from individual-level rules
- When modeling opinion formation, polarization, or information spread
- When implementing Schelling segregation or residential sorting models
- When running SIR/SIS contagion models on social networks
- When conducting parameter sweeps to find critical thresholds
- When visualizing spatial or network dynamics of social agents

**Trigger keywords**: agent-based model, ABM, Mesa, Schelling segregation, opinion dynamics, social contagion, information diffusion, SIR model, emergence, simulation, complex systems, social simulation

---

## Background & Key Concepts

### Agent-Based Modeling

ABM simulates individual agents following local rules to study emergent macro-level behavior:
- **Agents**: Autonomous decision-makers with state and behavior
- **Environment**: Grid, network, or continuous space
- **Schedule**: Defines order of agent activation (random, sequential, staged)
- **Data collection**: Track population-level statistics over time

### Schelling Segregation

Agents move if fewer than $T$% of neighbors share their type. Even with low $T$ (30%), high global segregation emerges — demonstrating how individual preferences aggregate into unintended outcomes.

### Opinion Dynamics (Bounded Confidence)

Agents update opinions only if neighbors' opinions are within a "confidence bound" $\epsilon$:

$$
x_i(t+1) = x_i(t) + \mu \sum_{j: |x_j - x_i| < \epsilon} (x_j(t) - x_i(t))
$$

This leads to polarization into isolated opinion clusters.

---

## Environment Setup

### Install Dependencies

```bash
pip install mesa>=2.3 numpy>=1.24 pandas>=2.0 networkx>=3.2 matplotlib>=3.7
```

### Verify Installation

```python
import mesa
print(f"Mesa version: {mesa.__version__}")

# Quick test
from mesa import Agent, Model
from mesa.time import RandomActivation

class TestAgent(Agent):
    def step(self): pass

class TestModel(Model):
    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)
        for i in range(10):
            self.schedule.add(TestAgent(i, self))
    def step(self):
        self.schedule.step()

m = TestModel()
m.step()
print("Mesa OK — 10 agents stepped")
```

---

## Core Workflow

### Step 1: Schelling Segregation Model

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# ------------------------------------------------------------------ #
# Schelling (1971) segregation model
# ------------------------------------------------------------------ #

class SchellingAgent(Agent):
    """Residential agent with type (0 or 1) and similarity preference."""

    def __init__(self, unique_id, model, agent_type, homophily):
        super().__init__(unique_id, model)
        self.type = agent_type
        self.homophily = homophily  # Minimum fraction of same-type neighbors
        self.is_happy = False

    def step(self):
        """Check happiness; if unhappy, move to random empty cell."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        same_type = sum(1 for n in neighbors if n.type == self.type)
        total = len(neighbors)

        if total == 0:
            self.is_happy = True
            return

        self.is_happy = (same_type / total) >= self.homophily

        if not self.is_happy:
            # Move to random empty cell
            empty_cells = list(self.model.grid.empties)
            if empty_cells:
                new_pos = self.random.choice(empty_cells)
                self.model.grid.move_agent(self, new_pos)


class SchellingModel(Model):
    """Schelling segregation model on a grid."""

    def __init__(self, width=30, height=30, density=0.85, pct_type1=0.50, homophily=0.30):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus=True)
        self.happiness = 0.0

        # Place agents
        agent_id = 0
        for x in range(width):
            for y in range(height):
                if self.random.random() < density:
                    a_type = 0 if self.random.random() < pct_type1 else 1
                    agent = SchellingAgent(agent_id, self, a_type, homophily)
                    self.grid.place_agent(agent, (x, y))
                    self.schedule.add(agent)
                    agent_id += 1

        self.datacollector = DataCollector(
            model_reporters={
                "Happiness":     lambda m: self._happiness(),
                "Segregation":   lambda m: self._segregation_index(),
            }
        )

    def _happiness(self):
        agents = self.schedule.agents
        if not agents:
            return 0.0
        return sum(1 for a in agents if a.is_happy) / len(agents)

    def _segregation_index(self):
        """Dissimilarity index: proportion of one type that would need to move."""
        agents = self.schedule.agents
        type0 = [a for a in agents if a.type == 0]
        type1 = [a for a in agents if a.type == 1]
        N0, N1 = len(type0), len(type1)
        if N0 == 0 or N1 == 0:
            return 0.0
        D = 0.0
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                cell = self.grid.get_cell_list_contents([(x, y)])
                n0 = sum(1 for a in cell if a.type == 0)
                n1 = sum(1 for a in cell if a.type == 1)
                D += abs(n0/N0 - n1/N1)
        return D / 2

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# ---- Run simulation -------------------------------------------- #
model = SchellingModel(width=30, height=30, density=0.85, pct_type1=0.50, homophily=0.35)
for _ in range(50):
    model.step()

data = model.datacollector.get_model_vars_dataframe()

print(f"Final happiness:   {data['Happiness'].iloc[-1]:.3f}")
print(f"Final segregation: {data['Segregation'].iloc[-1]:.3f}")

# ---- Visualization --------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Final grid state
grid_array = np.zeros((model.grid.width, model.grid.height))
for agent in model.schedule.agents:
    x, y = agent.pos
    grid_array[x, y] = agent.type + 1  # 0=empty, 1=type0, 2=type1

axes[0].imshow(grid_array.T, cmap='RdBu', vmin=0, vmax=2, aspect='equal')
axes[0].set_title(f"Final Spatial Distribution\n(homophily={0.35})")
axes[0].axis('off')

axes[1].plot(data['Happiness'] * 100, 'g-', linewidth=2)
axes[1].set_xlabel("Time step"); axes[1].set_ylabel("Happiness (%)")
axes[1].set_title("Agent Happiness Over Time"); axes[1].grid(True, alpha=0.3)

axes[2].plot(data['Segregation'], 'r-', linewidth=2)
axes[2].set_xlabel("Time step"); axes[2].set_ylabel("Dissimilarity Index")
axes[2].set_title("Segregation Index Over Time"); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("schelling_segregation.png", dpi=150)
plt.show()
```

### Step 2: Opinion Dynamics (Bounded Confidence)

```python
import numpy as np
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

# ------------------------------------------------------------------ #
# Deffuant-Weisbuch bounded confidence model
# ------------------------------------------------------------------ #

class OpinionAgent(Agent):
    """Agent with a continuous opinion in [0, 1]."""

    def __init__(self, unique_id, model, initial_opinion):
        super().__init__(unique_id, model)
        self.opinion = initial_opinion
        self.new_opinion = initial_opinion

    def step(self):
        """Interact with a random neighbor within confidence bound."""
        # Select random agent to interact with
        partner = self.random.choice(self.model.schedule.agents)
        if partner is self:
            return
        diff = abs(self.opinion - partner.opinion)
        if diff < self.model.epsilon:
            mu = self.model.mu  # Convergence parameter
            self.new_opinion = self.opinion + mu * (partner.opinion - self.opinion)

    def advance(self):
        """Apply the update (simultaneous activation)."""
        self.opinion = np.clip(self.new_opinion, 0, 1)


class OpinionModel(Model):
    """Bounded confidence opinion dynamics."""

    def __init__(self, n_agents=500, epsilon=0.3, mu=0.5):
        super().__init__()
        self.epsilon = epsilon
        self.mu = mu
        self.schedule = SimultaneousActivation(self)

        for i in range(n_agents):
            op = self.random.uniform(0, 1)
            agent = OpinionAgent(i, self, op)
            self.schedule.add(agent)

        self.datacollector = DataCollector(
            agent_reporters={"Opinion": "opinion"},
            model_reporters={
                "Mean":    lambda m: np.mean([a.opinion for a in m.schedule.agents]),
                "Std":     lambda m: np.std([a.opinion for a in m.schedule.agents]),
                "N_clusters": lambda m: self._count_clusters(m),
            }
        )

    def _count_clusters(self, m, tol=0.05):
        """Count opinion clusters (groups within tol of each other)."""
        opinions = sorted([a.opinion for a in m.schedule.agents])
        if not opinions:
            return 0
        clusters, cluster_start = 1, opinions[0]
        for op in opinions[1:]:
            if op - cluster_start > tol:
                clusters += 1
                cluster_start = op
        return clusters

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


# ---- Compare two epsilon values -------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for col, eps in enumerate([0.15, 0.45]):
    model = OpinionModel(n_agents=300, epsilon=eps, mu=0.5)
    snapshots = {0: None, 50: None, 200: None}

    for step in range(201):
        model.step()
        if step in snapshots:
            snapshots[step] = [a.opinion for a in model.schedule.agents]

    # Opinion distribution at different times
    for t, opinions in snapshots.items():
        axes[0][col].hist(opinions, bins=40, range=(0, 1),
                          alpha=0.6, density=True, label=f"t={t}")
    axes[0][col].set_xlabel("Opinion"); axes[0][col].set_ylabel("Density")
    axes[0][col].set_title(f"Opinion Distribution (ε={eps})")
    axes[0][col].legend(fontsize=9); axes[0][col].grid(True, alpha=0.3)

    # Cluster count over time
    cluster_data = model.datacollector.get_model_vars_dataframe()['N_clusters']
    axes[1][col].plot(cluster_data, color='purple', linewidth=1.5)
    axes[1][col].set_xlabel("Time step"); axes[1][col].set_ylabel("Number of clusters")
    axes[1][col].set_title(f"Cluster Formation (ε={eps})"); axes[1][col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("opinion_dynamics.png", dpi=150)
plt.show()
```

### Step 3: SIR Contagion on Social Network

```python
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# ------------------------------------------------------------------ #
# SIR contagion model on a small-world network
# ------------------------------------------------------------------ #

class SIRAgent(Agent):
    """S→I→R agent on a contact network."""

    def __init__(self, unique_id, model, state='S'):
        super().__init__(unique_id, model)
        self.state = state
        self.days_infected = 0

    def step(self):
        if self.state == 'I':
            # Infect susceptible neighbors
            neighbors_ids = list(self.model.network.neighbors(self.unique_id))
            for nid in neighbors_ids:
                neighbor = self.model.agents_dict.get(nid)
                if neighbor and neighbor.state == 'S':
                    if self.random.random() < self.model.beta:
                        neighbor.state = 'I'

            # Recover
            self.days_infected += 1
            if self.days_infected >= self.model.gamma_inv:
                self.state = 'R'


class SIRModel(Model):
    """SIR contagion on Watts-Strogatz small-world network."""

    def __init__(self, n_agents=500, k=6, p_rewire=0.1,
                 beta=0.05, gamma_inv=7, seed_frac=0.01):
        super().__init__()
        self.beta     = beta       # Transmission probability per contact per day
        self.gamma_inv = gamma_inv  # Mean infectious period (days)

        # Create Watts-Strogatz network
        self.network = nx.watts_strogatz_graph(n_agents, k, p_rewire, seed=42)
        self.schedule = RandomActivation(self)
        self.agents_dict = {}

        for node in self.network.nodes():
            state = 'I' if self.random.random() < seed_frac else 'S'
            agent = SIRAgent(node, self, state)
            self.schedule.add(agent)
            self.agents_dict[node] = agent

        self.datacollector = DataCollector(
            model_reporters={
                "S": lambda m: sum(1 for a in m.schedule.agents if a.state=='S'),
                "I": lambda m: sum(1 for a in m.schedule.agents if a.state=='I'),
                "R": lambda m: sum(1 for a in m.schedule.agents if a.state=='R'),
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        # Stop when epidemic ends
        return sum(1 for a in self.schedule.agents if a.state=='I') > 0

    @property
    def R0(self):
        """Basic reproduction number: R₀ = β × <k> × γ_inv"""
        avg_degree = np.mean([d for _, d in self.network.degree()])
        return self.beta * avg_degree * self.gamma_inv


# ---- Simulation ------------------------------------------------- #
n_agents = 500
model = SIRModel(n_agents=n_agents, k=6, p_rewire=0.1,
                 beta=0.04, gamma_inv=7, seed_frac=0.02)

print(f"R₀ = {model.R0:.2f}  (epidemic if R₀>1)")

for day in range(100):
    active = model.step()
    if not active and day > 10:
        print(f"Epidemic ended at day {day}")
        break

data = model.datacollector.get_model_vars_dataframe()
final = data.iloc[-1]
print(f"\nFinal state: S={final['S']:.0f} ({final['S']/n_agents*100:.1f}%), "
      f"I={final['I']:.0f}, R={final['R']:.0f} ({final['R']/n_agents*100:.1f}%)")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['S']/n_agents*100, 'b-', label='Susceptible', linewidth=2)
ax.plot(data['I']/n_agents*100, 'r-', label='Infectious',  linewidth=2)
ax.plot(data['R']/n_agents*100, 'g-', label='Recovered',   linewidth=2)
ax.set_xlabel("Day"); ax.set_ylabel("Population fraction (%)")
ax.set_title(f"SIR Epidemic on Watts-Strogatz Network  (R₀={model.R0:.2f})")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sir_epidemic.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Parameter Sweep with BatchRunner

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Sweep homophily threshold in Schelling model
# ------------------------------------------------------------------ #

homophily_values = np.arange(0.1, 0.9, 0.1)
n_runs = 5
results = []

for homophily in homophily_values:
    for run in range(n_runs):
        model = SchellingModel(width=20, height=20, density=0.80,
                               pct_type1=0.50, homophily=float(homophily))
        for _ in range(30):
            model.step()
        data = model.datacollector.get_model_vars_dataframe()
        results.append({
            'homophily': homophily,
            'run': run,
            'segregation': data['Segregation'].iloc[-1],
            'happiness':   data['Happiness'].iloc[-1],
        })

df = pd.DataFrame(results)
summary = df.groupby('homophily')[['segregation','happiness']].agg(['mean','std'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, title in zip(axes, ['segregation','happiness'], ['Segregation','Happiness']):
    means = summary[col]['mean']
    stds  = summary[col]['std']
    ax.plot(homophily_values, means, 'bo-', linewidth=2, markersize=7)
    ax.fill_between(homophily_values, means-stds, means+stds, alpha=0.2, color='blue')
    ax.set_xlabel("Homophily threshold"); ax.set_ylabel(title)
    ax.set_title(f"{title} vs. Homophily (Schelling)"); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("schelling_sweep.png", dpi=150); plt.show()
```

---

## Troubleshooting

### Error: `AttributeError: 'NoneType' object has no attribute 'step'`

**Cause**: Agent scheduler not initialized.

**Fix**:
```python
class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)  # Must initialize before adding agents
```

### Error: `ImportError: cannot import name 'MultiGrid'`

**Cause**: Mesa 2.x moved some classes.

**Fix**:
```python
# Mesa >= 2.0
from mesa.space import MultiGrid        # Still available
from mesa.time import RandomActivation  # Still available
```

### Slow simulation with large grids

```python
# Use numpy vectorized operations instead of agent loops for large N
# Or reduce grid resolution: width=height=20 runs much faster than 100×100
```

### Version Compatibility

| Package | Tested versions | Notes |
|:--------|:----------------|:------|
| mesa | 2.3, 2.4 | API stable since 2.0; BatchRunner removed in 2.x — use loops |
| networkx | 3.2 | No known issues |

---

## External Resources

### Official Documentation

- [Mesa documentation](https://mesa.readthedocs.io)
- [Mesa examples repository](https://github.com/projectmesa/mesa-examples)

### Key Papers

- Schelling, T.C. (1971). *Dynamic models of segregation*. Journal of Mathematical Sociology.
- Deffuant, G. et al. (2000). *Mixing beliefs among interacting agents*. Advances in Complex Systems.

---

## Examples

### Example 1: Wealth Distribution (Sugarscape)

```python
import numpy as np
import matplotlib.pyplot as plt

# Simplified Sugarscape: agents accumulate wealth, consume, and die
np.random.seed(42)
n = 500
wealth = np.random.exponential(10, n)  # Initial wealth
metabolism = np.random.randint(1, 5, n)  # Daily consumption

for t in range(100):
    # Sugar grows (simplified as random income)
    income = np.random.poisson(3, n)
    wealth += income - metabolism
    # Dead agents replaced with wealth=random
    dead = wealth <= 0
    wealth[dead] = np.random.exponential(5, dead.sum())

# Compute Gini
w_sorted = np.sort(wealth)
n = len(w_sorted)
G = (2 * np.sum(np.arange(1,n+1) * w_sorted) / (n * w_sorted.sum())) - (n+1)/n
print(f"Sugarscape Gini coefficient: {G:.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(wealth, bins=50, color='gold', edgecolor='brown', linewidth=0.5)
ax.set_xlabel("Wealth"); ax.set_title(f"Sugarscape Wealth Distribution (Gini={G:.3f})")
ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig("sugarscape.png", dpi=150); plt.show()
```

### Example 2: Network Influence Cascade

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.barabasi_albert_graph(500, 3, seed=42)
# Simple threshold diffusion: adopt if >30% of neighbors adopted
adopted = set(list(G.nodes())[:5])  # Seed with 5 adopters
cascade_size = [len(adopted)]

for _ in range(50):
    new_adopters = set()
    for node in G.nodes():
        if node in adopted:
            continue
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
        frac_adopted = sum(1 for n in neighbors if n in adopted) / len(neighbors)
        if frac_adopted > 0.30:
            new_adopters.add(node)
    adopted.update(new_adopters)
    cascade_size.append(len(adopted))
    if not new_adopters:
        break

print(f"Final cascade size: {len(adopted)}/{G.number_of_nodes()} ({len(adopted)/G.number_of_nodes()*100:.1f}%)")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cascade_size, 'b-o', markersize=4, linewidth=1.5)
ax.set_xlabel("Time step"); ax.set_ylabel("Adopters"); ax.set_title("Information Cascade on BA Network")
ax.grid(alpha=0.3); plt.tight_layout(); plt.savefig("cascade.png", dpi=150); plt.show()
```

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
