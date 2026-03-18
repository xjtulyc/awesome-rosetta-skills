---
name: simpy-discrete-event
description: >
  Use this Skill for discrete-event simulation with SimPy: M/M/c queues,
  manufacturing lines, resource contention, and confidence intervals via replications.
tags:
  - engineering
  - simulation
  - simpy
  - queuing-theory
  - operations-research
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
    - simpy>=4.1
    - scipy>=1.11
    - matplotlib>=3.7
    - numpy>=1.24
    - pandas>=2.0
last_updated: "2026-03-17"
status: "stable"
---

# Discrete-Event Simulation with SimPy

> **One-line summary**: Build and analyze discrete-event simulations with SimPy: M/M/c queue models, manufacturing lines, resource scheduling, and statistical output analysis via replications.

---

## When to Use This Skill

- When modeling waiting lines and service systems (hospitals, call centers, factories)
- When analyzing manufacturing line throughput and bottlenecks
- When comparing alternative resource allocation policies
- When computing steady-state performance measures (utilization, waiting time)
- When performing output analysis (warm-up detection, confidence intervals)
- When simulating complex logistics or network systems

**Trigger keywords**: SimPy, discrete-event simulation, queuing, M/M/c, waiting time, manufacturing, throughput, resource contention, service system, replications

---

## Background & Key Concepts

### Queuing Theory (M/M/c)

For M/M/c queue (Poisson arrivals, exponential service, c servers):

$$
\rho = \frac{\lambda}{c\mu} \quad (\text{utilization per server})
$$

$$
P_0 = \left[\sum_{n=0}^{c-1} \frac{(\lambda/\mu)^n}{n!} + \frac{(\lambda/\mu)^c}{c!(1-\rho)}\right]^{-1}
$$

$$
W_q = \frac{P_0 (\lambda/\mu)^c \rho}{c \mu (1-\rho)^2 \lambda} \quad (\text{mean waiting time in queue})
$$

### SimPy Process Model

Every entity (customer, job) is a Python generator that uses `yield` to:
- `env.timeout(duration)` — wait for a duration
- `resource.request()` — seize a resource
- `resource.release(req)` — release a resource

### Warm-Up Period

Transient phase at simulation start can bias steady-state estimates. Detect using Welch's method or rule of thumb: discard first 10-20% of simulation time.

### Output Analysis

For $k$ replications, the $95\%$ CI for mean $\mu$ is:

$$
\bar{X} \pm t_{k-1, 0.025} \cdot \frac{S}{\sqrt{k}}
$$

---

## Environment Setup

### Install Dependencies

```bash
pip install simpy>=4.1 scipy>=1.11 matplotlib>=3.7 numpy>=1.24 pandas>=2.0
```

### Verify Installation

```python
import simpy
import numpy as np

env = simpy.Environment()
def test_process(env):
    yield env.timeout(5)
    print("SimPy process completed at t=5")

env.process(test_process(env))
env.run()
print(f"SimPy version: {simpy.__version__}")
```

---

## Core Workflow

### Step 1: Basic M/M/1 Queue Simulation

```python
import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def mm1_queue_simulation(arrival_rate, service_rate, sim_time=10000, seed=42):
    """
    Simulate M/M/1 queuing system.

    Parameters
    ----------
    arrival_rate : float
        λ (customers/time unit)
    service_rate : float
        μ (customers/time unit per server)
    sim_time : float
        Total simulation duration
    seed : int

    Returns
    -------
    dict: wait times, service times, utilization
    """
    rng = np.random.default_rng(seed)

    # Storage for results
    wait_times = []
    n_in_system = []
    departure_times = []
    arrival_times = []

    def customer(env, server, rng):
        arrival = env.now
        arrival_times.append(arrival)
        n_in_system.append(len(server.queue) + len(server.users))

        with server.request() as req:
            yield req
            wait_times.append(env.now - arrival)
            service_time = rng.exponential(1.0 / service_rate)
            yield env.timeout(service_time)
            departure_times.append(env.now)

    def arrivals(env, server, rng):
        while True:
            yield env.timeout(rng.exponential(1.0 / arrival_rate))
            env.process(customer(env, server, rng))

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    env.process(arrivals(env, server, rng))
    env.run(until=sim_time)

    rho = arrival_rate / service_rate
    W_q_theoretical = rho / (service_rate - arrival_rate)  # M/M/1 formula

    results = {
        "n_customers": len(wait_times),
        "mean_wait_sim": np.mean(wait_times),
        "mean_wait_theory": W_q_theoretical,
        "utilization_sim": np.mean([1 if w > 0 else 0 for w in wait_times]),
        "utilization_theory": rho,
        "wait_times": wait_times,
    }
    return results

# Test with ρ = 0.8 (λ=0.8, μ=1.0)
results = mm1_queue_simulation(arrival_rate=0.8, service_rate=1.0, sim_time=50000)

print(f"M/M/1 Queue (ρ = 0.8):")
print(f"  Customers served:  {results['n_customers']:,}")
print(f"  Mean wait (sim):   {results['mean_wait_sim']:.4f}")
print(f"  Mean wait (theory): {results['mean_wait_theory']:.4f}")
print(f"  Utilization (sim):  {results['utilization_sim']:.4f}")
print(f"  Utilization (theory): {results['utilization_theory']:.4f}")

# Wait time distribution
fig, ax = plt.subplots(figsize=(8, 4))
wait_arr = np.array(results["wait_times"])
ax.hist(wait_arr[wait_arr > 0], bins=50, density=True, alpha=0.7, label="Simulation")
x = np.linspace(0, np.percentile(wait_arr, 99), 100)
lam_fit = 1 / results["mean_wait_sim"]
ax.plot(x, lam_fit * np.exp(-lam_fit * x), 'r-', linewidth=2, label="Exp fit")
ax.set_xlabel("Waiting time"); ax.set_ylabel("Density")
ax.set_title("M/M/1 Waiting Time Distribution"); ax.legend()
plt.tight_layout()
plt.savefig("mm1_waiting_time.png", dpi=150)
plt.show()
```

### Step 2: Manufacturing Line Simulation

```python
import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ManufacturingLine:
    """
    Three-stage manufacturing line: Cutting → Machining → Assembly.
    Each stage has multiple machines.
    """

    def __init__(self, env, config, seed=42):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.machines = {
            "cutting":   simpy.Resource(env, capacity=config["n_cutting"]),
            "machining": simpy.Resource(env, capacity=config["n_machining"]),
            "assembly":  simpy.Resource(env, capacity=config["n_assembly"]),
        }
        self.service_times = config["service_times"]  # (mean, std) per stage
        self.completed = 0
        self.cycle_times = []
        self.queue_lengths = {k: [] for k in self.machines}

    def process_job(self, job_id):
        start = self.env.now

        for stage, resource in self.machines.items():
            mean, std = self.service_times[stage]
            service = max(0, self.rng.normal(mean, std))
            with resource.request() as req:
                yield req
                yield self.env.timeout(service)

        self.cycle_times.append(self.env.now - start)
        self.completed += 1

    def record_queues(self):
        while True:
            for stage, resource in self.machines.items():
                self.queue_lengths[stage].append(len(resource.queue))
            yield self.env.timeout(1)

def run_manufacturing_sim(config, arrival_rate, sim_time=2000, seed=42):
    env = simpy.Environment()
    line = ManufacturingLine(env, config, seed)
    rng = np.random.default_rng(seed)

    def job_arrivals():
        job_id = 0
        while True:
            yield env.timeout(rng.exponential(1.0 / arrival_rate))
            env.process(line.process_job(job_id))
            job_id += 1

    env.process(job_arrivals())
    env.process(line.record_queues())
    env.run(until=sim_time)

    # Discard warm-up (first 20%)
    warmup_idx = len(line.cycle_times) // 5
    steady_times = line.cycle_times[warmup_idx:]

    return {
        "throughput": line.completed / sim_time,
        "mean_cycle": np.mean(steady_times),
        "std_cycle":  np.std(steady_times),
        "utilization": {
            stage: 1 - (np.mean(line.queue_lengths[stage]) / max(resource.capacity, 1))
            for stage, resource in line.machines.items()
        },
    }

# Baseline configuration
config = {
    "n_cutting":   2,
    "n_machining": 3,
    "n_assembly":  2,
    "service_times": {
        "cutting":   (5, 1),   # mean 5 min, std 1 min
        "machining": (4, 0.8),
        "assembly":  (3, 0.6),
    }
}

results = run_manufacturing_sim(config, arrival_rate=0.3, sim_time=5000)
print(f"\nManufacturing Line Results:")
print(f"  Throughput: {results['throughput']:.4f} jobs/time")
print(f"  Mean cycle time: {results['mean_cycle']:.2f} ± {results['std_cycle']:.2f}")
print(f"  Machine utilization:")
for stage, util in results['utilization'].items():
    print(f"    {stage:12s}: {util:.1%}")
```

### Step 3: Replication Analysis and Confidence Intervals

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

def run_replications(sim_func, n_reps=20, **sim_kwargs):
    """
    Run multiple independent replications and compute CI.

    Parameters
    ----------
    sim_func : callable
        Function taking **sim_kwargs + seed parameter
    n_reps : int
        Number of replications

    Returns
    -------
    pd.DataFrame: per-replication results + CI summary
    """
    all_results = []
    for rep in range(n_reps):
        sim_kwargs["seed"] = rep * 1000  # different seed per replication
        r = sim_func(**sim_kwargs)
        r["replication"] = rep
        all_results.append(r)

    df = pd.DataFrame(all_results)

    # Confidence intervals for throughput
    mu = df["throughput"].mean()
    s  = df["throughput"].std()
    t_crit = stats.t.ppf(0.975, df=n_reps - 1)
    ci_half = t_crit * s / np.sqrt(n_reps)

    print(f"\nOutput Analysis ({n_reps} replications):")
    print(f"  Throughput: {mu:.4f} ± {ci_half:.4f} (95% CI)")
    print(f"  Mean cycle: {df['mean_cycle'].mean():.2f} ± "
          f"{stats.t.ppf(0.975, n_reps-1) * df['mean_cycle'].std() / np.sqrt(n_reps):.2f}")

    return df, (mu - ci_half, mu + ci_half)

reps_df, ci = run_replications(
    lambda seed: run_manufacturing_sim(config, arrival_rate=0.3, sim_time=3000, seed=seed),
    n_reps=15
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(reps_df["throughput"], 'bo-', ms=5)
axes[0].axhline(reps_df["throughput"].mean(), color='r', linestyle='--', label="Mean")
axes[0].fill_between(range(len(reps_df)), ci[0], ci[1], alpha=0.2, color='r', label="95% CI")
axes[0].set_xlabel("Replication"); axes[0].set_ylabel("Throughput")
axes[0].set_title("Throughput per Replication"); axes[0].legend()

axes[1].hist(reps_df["mean_cycle"], bins=10, color="steelblue", edgecolor="white")
axes[1].set_xlabel("Mean cycle time"); axes[1].set_ylabel("Count")
axes[1].set_title("Cycle Time Distribution (across reps)")
plt.tight_layout()
plt.savefig("replication_analysis.png", dpi=150)
plt.show()
```

---

## Advanced Usage

### Sensitivity Analysis: Resource Levels

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_machining_range = range(1, 6)
throughputs = []

for n_mach in n_machining_range:
    config_var = config.copy()
    config_var["n_machining"] = n_mach
    r = run_manufacturing_sim(config_var, arrival_rate=0.3, sim_time=3000, seed=42)
    throughputs.append(r["throughput"])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(n_machining_range), throughputs, 'gs-', ms=8, linewidth=2)
ax.set_xlabel("Number of machining stations")
ax.set_ylabel("Throughput (jobs/time)")
ax.set_title("Throughput vs. Machining Station Count")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sensitivity_analysis.png", dpi=150)
plt.show()
```

---

## Troubleshooting

### Error: `simpy.exceptions.StopSimulation`

**Cause**: Simulation ran out of events before `sim_time`.

**Fix**:
```python
# Ensure arrival process runs forever
def arrivals(env):
    while True:  # infinite loop required
        yield env.timeout(...)
        env.process(...)
```

### Issue: Results show high variance across replications

**Cause**: Simulation time too short; includes transient effects.

**Fix**:
```python
# Increase sim_time and warm-up fraction
sim_time = 20000
warmup_fraction = 0.2  # discard first 20%
```

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| simpy | 4.0, 4.1, 4.1.1 | API stable; process-based paradigm unchanged |

---

## External Resources

### Official Documentation

- [SimPy documentation](https://simpy.readthedocs.io/)
- [SimPy examples](https://simpy.readthedocs.io/en/latest/examples/index.html)

### Key Papers

- Kelton, W.D. et al. (2014). *Simulation with Arena*. McGraw-Hill.

---

## Examples

### Example 1: Hospital Emergency Department Simulation

```python
# =============================================
# Hospital ED: triage → consultation → discharge
# =============================================
import simpy, numpy as np, pandas as pd

def emergency_department(env, n_triage=2, n_docs=5, n_nurses=8,
                          arrival_rate=10, sim_time=480, seed=42):
    """Simulate 8-hour ED shift (sim_time=480 min)."""
    rng = np.random.default_rng(seed)
    triage = simpy.Resource(env, capacity=n_triage)
    doctors = simpy.Resource(env, capacity=n_docs)
    nurses = simpy.Resource(env, capacity=n_nurses)
    waits = []

    def patient(env, severity):
        arrival = env.now
        with triage.request() as t_req:
            yield t_req
            yield env.timeout(rng.uniform(3, 8))  # triage 3-8 min

        with doctors.request() as d_req:
            yield d_req
            treatment = rng.exponential(20 + severity * 15)  # high severity → longer
            yield env.timeout(treatment)

        with nurses.request() as n_req:
            yield n_req
            yield env.timeout(rng.uniform(5, 15))  # discharge 5-15 min

        waits.append(env.now - arrival)

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(60 / arrival_rate))
            severity = rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            env.process(patient(env, severity))

    env.process(arrivals())
    env.run(until=sim_time)
    return waits

env = simpy.Environment()
waits = emergency_department(env, sim_time=480)
print(f"Patients treated: {len(waits)}")
print(f"Mean LOS: {np.mean(waits):.1f} min")
print(f"90th percentile LOS: {np.percentile(waits, 90):.1f} min")
print(f"Patients >4h: {(np.array(waits) > 240).sum()}")
```

**Interpreting these results**: Mean LOS and 90th percentile LOS are key ED performance metrics. Patients exceeding 4 hours (240 min) represent quality-of-care concerns per NHS/CMS standards.

---

*Last updated: 2026-03-17 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
