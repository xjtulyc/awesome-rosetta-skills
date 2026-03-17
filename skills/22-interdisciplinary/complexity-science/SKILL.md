---
name: complexity-science
description: >
  Complex systems methods: power-law fitting, Hurst exponent, fractal dimension,
  Mesa agent-based modeling, sample entropy, percolation, and information theory.
tags:
  - complexity-science
  - power-law
  - agent-based-modeling
  - fractal
  - network-science
  - information-theory
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
  - powerlaw>=1.5.0
  - numpy>=1.24.0
  - scipy>=1.10.0
  - networkx>=3.1.0
  - mesa>=2.1.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
last_updated: "2026-03-17"
---

# complexity-science: Complex Systems Methods

This skill covers core quantitative methods for complex systems research: power-law
distribution testing, long-range correlations (Hurst exponent), fractal geometry,
agent-based modelling with Mesa, sample entropy, bond percolation, and
information-theoretic measures.

## Installation

```bash
pip install powerlaw numpy scipy networkx mesa pandas matplotlib
# Optional: antropy for sample/permutation entropy
pip install antropy
```

---

## 1. Power-Law Fitting and Testing

The `powerlaw` package fits power-law distributions using maximum-likelihood estimation
and performs likelihood-ratio tests against alternative heavy-tailed distributions.

```python
import powerlaw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_power_law(data: np.ndarray, xmin: float = None, discrete: bool = False) -> dict:
    """
    Fit a power-law distribution to empirical data using MLE.

    Parameters
    ----------
    data : np.ndarray  — positive values (e.g., degree sequence, word frequencies)
    xmin : float or None  — minimum value for fitting; estimated automatically if None
    discrete : bool  — True for integer data (e.g., degree counts)

    Returns
    -------
    dict with keys: alpha, xmin, sigma (standard error), n_tail, fit object
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]

    fit = powerlaw.Fit(data, xmin=xmin, discrete=discrete, verbose=False)

    return {
        "alpha": fit.alpha,
        "xmin": fit.xmin,
        "sigma": fit.sigma,
        "n_tail": int(np.sum(data >= fit.xmin)),
        "n_total": len(data),
        "fit": fit,
    }


def test_power_law_vs_alternatives(data: np.ndarray, discrete: bool = False) -> pd.DataFrame:
    """
    Test whether a power law is a better fit than lognormal, exponential,
    stretched exponential (Weibull), and truncated power law.

    Uses the Vuong likelihood-ratio test (p < 0.05 favours the first distribution
    if R > 0, the second if R < 0).

    Returns
    -------
    pd.DataFrame with columns: alternative, R (log-likelihood ratio), p_value, preferred
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    fit = powerlaw.Fit(data, discrete=discrete, verbose=False)

    alternatives = ["lognormal", "exponential", "stretched_exponential", "truncated_power_law"]
    records = []
    for alt in alternatives:
        try:
            R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
            preferred = "power_law" if (R > 0 and p < 0.05) else (alt if (R < 0 and p < 0.05) else "inconclusive")
            records.append({"alternative": alt, "R": round(R, 4), "p_value": round(p, 4), "preferred": preferred})
        except Exception as exc:
            records.append({"alternative": alt, "R": np.nan, "p_value": np.nan, "preferred": f"error: {exc}"})

    return pd.DataFrame(records)


def plot_power_law(fit_result: dict, title: str = "Power-Law Fit", ax=None):
    """
    Plot empirical CCDF with the fitted power-law (and lognormal comparison) overlay.
    """
    fit = fit_result["fit"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    fit.plot_ccdf(ax=ax, color="steelblue", linewidth=0.5, label="Empirical CCDF")
    fit.power_law.plot_ccdf(ax=ax, color="red", linestyle="--",
                            label=f"Power law α={fit.alpha:.2f}")
    fit.lognormal.plot_ccdf(ax=ax, color="green", linestyle=":",
                            label="Lognormal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("x"); ax.set_ylabel("P(X ≥ x)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return ax
```

---

## 2. Hurst Exponent via R/S Analysis

The Hurst exponent H characterises long-range correlations:
- H ≈ 0.5: random walk / white noise
- H > 0.5: persistent (trending) process
- H < 0.5: anti-persistent (mean-reverting) process

```python
def compute_hurst(ts: np.ndarray, min_window: int = 8) -> dict:
    """
    Estimate the Hurst exponent using rescaled range (R/S) analysis.

    Parameters
    ----------
    ts : np.ndarray  — 1-D time series
    min_window : int  — smallest window size (must be ≥ 4)

    Returns
    -------
    dict with keys: hurst, intercept, windows, rs_values, r_squared
    """
    ts = np.asarray(ts, dtype=float)
    n = len(ts)

    # Generate window sizes as powers of 2
    max_power = int(np.log2(n)) - 1
    windows = [2 ** p for p in range(int(np.log2(min_window)), max_power + 1)]

    rs_values = []
    for w in windows:
        rs_per_window = []
        for start in range(0, n - w + 1, w):
            sub = ts[start: start + w]
            mean_sub = np.mean(sub)
            deviation = np.cumsum(sub - mean_sub)
            R = np.max(deviation) - np.min(deviation)
            S = np.std(sub, ddof=1)
            if S > 0:
                rs_per_window.append(R / S)
        if rs_per_window:
            rs_values.append(np.mean(rs_per_window))
        else:
            rs_values.append(np.nan)

    windows_arr = np.array(windows, dtype=float)
    rs_arr = np.array(rs_values, dtype=float)
    valid = ~np.isnan(rs_arr) & (rs_arr > 0)

    log_w = np.log(windows_arr[valid])
    log_rs = np.log(rs_arr[valid])
    coeffs = np.polyfit(log_w, log_rs, 1)
    hurst = coeffs[0]

    rs_pred = np.polyval(coeffs, log_w)
    ss_res = np.sum((log_rs - rs_pred) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "hurst": round(hurst, 4),
        "intercept": coeffs[1],
        "windows": windows_arr[valid].tolist(),
        "rs_values": rs_arr[valid].tolist(),
        "r_squared": round(r2, 4),
    }
```

---

## 3. Fractal Dimension via Box-Counting

```python
def estimate_fractal_dimension(array_2d: np.ndarray, threshold: float = None) -> dict:
    """
    Estimate the fractal (Hausdorff) dimension of a binary 2-D pattern via
    box-counting.

    Parameters
    ----------
    array_2d : np.ndarray  — 2-D float or binary array
    threshold : float or None  — binarisation threshold (default: mean of array)

    Returns
    -------
    dict with keys: fractal_dimension, r_squared, box_sizes, counts
    """
    arr = np.asarray(array_2d, dtype=float)
    if threshold is None:
        threshold = np.mean(arr)
    binary = (arr >= threshold).astype(int)

    # Pad to next power of 2 for clean box counting
    max_dim = max(binary.shape)
    next_pow2 = 2 ** int(np.ceil(np.log2(max_dim)))
    padded = np.zeros((next_pow2, next_pow2), dtype=int)
    padded[: binary.shape[0], : binary.shape[1]] = binary

    box_sizes = []
    counts = []
    size = next_pow2
    while size >= 2:
        # Count non-empty boxes of given size
        n_boxes_per_side = next_pow2 // size
        reshaped = padded.reshape(n_boxes_per_side, size, n_boxes_per_side, size)
        box_sums = reshaped.sum(axis=(1, 3))
        count = np.sum(box_sums > 0)
        if count > 0:
            box_sizes.append(size)
            counts.append(count)
        size //= 2

    log_s = np.log(1.0 / np.array(box_sizes, dtype=float))
    log_c = np.log(np.array(counts, dtype=float))
    coeffs = np.polyfit(log_s, log_c, 1)
    fd = coeffs[0]

    c_pred = np.polyval(coeffs, log_s)
    ss_res = np.sum((log_c - c_pred) ** 2)
    ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "fractal_dimension": round(fd, 4),
        "r_squared": round(r2, 4),
        "box_sizes": box_sizes,
        "counts": counts,
    }
```

---

## 4. Agent-Based Modeling: Schelling Segregation

```python
import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector


class SchellingAgent(Agent):
    """A single household agent in the Schelling segregation model."""

    def __init__(self, unique_id: int, model: "SchellingModel", agent_type: int):
        super().__init__(unique_id, model)
        self.agent_type = agent_type   # 0 or 1 (two groups)
        self.is_happy = False

    def step(self):
        neighbours = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if not neighbours:
            self.is_happy = True
            return
        same_type = sum(1 for n in neighbours if n.agent_type == self.agent_type)
        self.is_happy = same_type / len(neighbours) >= self.model.tolerance
        if not self.is_happy:
            self.model.grid.move_to_empty(self)


class SchellingModel(Model):
    """Schelling (1971) segregation model."""

    def __init__(self, n: int = 20, density: float = 0.9, tolerance: float = 0.3, seed: int = 42):
        super().__init__()
        self.n = n
        self.density = density
        self.tolerance = tolerance
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(n, n, torus=True)
        self.running = True

        self.datacollector = DataCollector(
            model_reporters={
                "pct_happy": lambda m: sum(a.is_happy for a in m.schedule.agents) / m.schedule.get_agent_count(),
                "segregation_index": lambda m: m.compute_segregation(),
            }
        )

        agent_id = 0
        for cell in self.grid.coord_iter():
            _, x, y = cell
            if self.random.random() < density:
                agent_type = self.random.choice([0, 1])
                agent = SchellingAgent(agent_id, self, agent_type)
                self.grid.place_agent(agent, (x, y))
                self.schedule.add(agent)
                agent_id += 1

    def compute_segregation(self) -> float:
        """
        Moran's I-style segregation index:
        fraction of same-type neighbours averaged over all agents.
        """
        fractions = []
        for agent in self.schedule.agents:
            neighbours = self.grid.get_neighbors(agent.pos, moore=True, include_center=False)
            if neighbours:
                same = sum(1 for n in neighbours if n.agent_type == agent.agent_type)
                fractions.append(same / len(neighbours))
        return float(np.mean(fractions)) if fractions else 0.5

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        pct_happy = sum(a.is_happy for a in self.schedule.agents) / self.schedule.get_agent_count()
        if pct_happy >= 1.0:
            self.running = False


def run_schelling(
    n: int = 20,
    tolerance: float = 0.3,
    steps: int = 50,
    density: float = 0.9,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the Schelling segregation model and return step-level metrics.

    Returns
    -------
    pd.DataFrame with columns: step, pct_happy, segregation_index
    """
    model = SchellingModel(n=n, density=density, tolerance=tolerance, seed=seed)
    for _ in range(steps):
        if not model.running:
            break
        model.step()

    df = model.datacollector.get_model_vars_dataframe().reset_index()
    df.columns = ["step", "pct_happy", "segregation_index"]
    return df
```

---

## 5. Sample Entropy

```python
def compute_sample_entropy(ts: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Compute sample entropy (SampEn) of a 1-D time series.

    SampEn measures regularity: lower values indicate more self-similar series.

    Parameters
    ----------
    ts : np.ndarray  — 1-D time series
    m : int  — template length (embedding dimension)
    r : float or None  — tolerance (default: 0.2 × std(ts))

    Returns
    -------
    float  — sample entropy (nan if undefined)
    """
    ts = np.asarray(ts, dtype=float)
    N = len(ts)
    if r is None:
        r = 0.2 * np.std(ts, ddof=1)

    def _count_matches(template_len: int) -> int:
        count = 0
        for i in range(N - template_len):
            template = ts[i: i + template_len]
            for j in range(N - template_len):
                if i == j:
                    continue
                candidate = ts[j: j + template_len]
                if np.max(np.abs(template - candidate)) <= r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))
```

---

## 6. Bond Percolation Threshold

```python
import networkx as nx


def percolation_threshold_bond(
    G: nx.Graph,
    n_trials: int = 30,
    p_values: np.ndarray = None,
) -> dict:
    """
    Estimate the bond percolation threshold of a graph by measuring the relative
    size of the giant component as a function of edge occupation probability p.

    Parameters
    ----------
    G : nx.Graph
    n_trials : int  — Monte Carlo repetitions per p value
    p_values : np.ndarray or None  — occupation probabilities (default: 0 to 1 in 20 steps)

    Returns
    -------
    dict with keys: p_values, giant_fraction_mean, giant_fraction_std,
                    estimated_threshold
    """
    if p_values is None:
        p_values = np.linspace(0.0, 1.0, 21)

    N = G.number_of_nodes()
    edges = list(G.edges())

    giant_means = []
    giant_stds = []

    for p in p_values:
        fractions = []
        for _ in range(n_trials):
            kept = [e for e in edges if np.random.rand() < p]
            subgraph = nx.Graph()
            subgraph.add_nodes_from(G.nodes())
            subgraph.add_edges_from(kept)
            components = sorted(nx.connected_components(subgraph), key=len, reverse=True)
            if components:
                fractions.append(len(components[0]) / N)
            else:
                fractions.append(0.0)
        giant_means.append(np.mean(fractions))
        giant_stds.append(np.std(fractions))

    # Estimate threshold as the p where giant fraction crosses 0.5 × max
    half_max = max(giant_means) / 2.0
    threshold_idx = next(
        (i for i, gf in enumerate(giant_means) if gf >= half_max),
        len(p_values) - 1,
    )
    threshold = float(p_values[threshold_idx])

    return {
        "p_values": p_values,
        "giant_fraction_mean": np.array(giant_means),
        "giant_fraction_std": np.array(giant_stds),
        "estimated_threshold": threshold,
    }
```

---

## 7. Information-Theoretic Measures

```python
from scipy.stats import entropy as scipy_entropy
from scipy.special import entr


def compute_information_measures(x: np.ndarray, y: np.ndarray = None, bins: int = 20) -> dict:
    """
    Compute Shannon entropy, mutual information, and normalised mutual information
    for one or two continuous signals.

    Parameters
    ----------
    x : np.ndarray  — first signal
    y : np.ndarray or None  — second signal (required for MI)
    bins : int  — number of histogram bins for density estimation

    Returns
    -------
    dict with keys: entropy_x, entropy_y (if y given), mutual_info, nmi
    """
    x = np.asarray(x, dtype=float)
    px, _ = np.histogram(x, bins=bins, density=True)
    px = px / px.sum()
    px = px[px > 0]
    Hx = float(scipy_entropy(px, base=2))

    result = {"entropy_x": round(Hx, 4)}

    if y is not None:
        y = np.asarray(y, dtype=float)
        py, _ = np.histogram(y, bins=bins, density=True)
        py = py / py.sum()
        py = py[py > 0]
        Hy = float(scipy_entropy(py, base=2))
        result["entropy_y"] = round(Hy, 4)

        # Joint entropy
        pxy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
        pxy_flat = pxy.ravel()
        pxy_flat = pxy_flat / pxy_flat.sum()
        pxy_flat = pxy_flat[pxy_flat > 0]
        Hxy = float(scipy_entropy(pxy_flat, base=2))
        result["joint_entropy"] = round(Hxy, 4)

        MI = Hx + Hy - Hxy
        NMI = 2 * MI / (Hx + Hy) if (Hx + Hy) > 0 else 0.0
        result["mutual_info"] = round(MI, 4)
        result["nmi"] = round(NMI, 4)

    return result
```

---

## 8. Examples

### Example A — Test if a Citation Network Degree Distribution Follows a Power Law

```python
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Generate a synthetic scale-free citation network (Barabasi-Albert) ---
np.random.seed(42)
G = nx.barabasi_albert_graph(n=2000, m=3, seed=42)
degrees = np.array([d for _, d in G.degree()])

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Degree range: [{degrees.min()}, {degrees.max()}], mean={degrees.mean():.2f}")

# Fit power law
pl_result = fit_power_law(degrees, discrete=True)
print(f"\nPower-law fit:")
print(f"  alpha = {pl_result['alpha']:.4f}")
print(f"  xmin  = {pl_result['xmin']:.0f}")
print(f"  sigma = {pl_result['sigma']:.4f}")
print(f"  n_tail = {pl_result['n_tail']} / {pl_result['n_total']}")

# Compare against alternatives
comparison = test_power_law_vs_alternatives(degrees, discrete=True)
print("\nDistribution comparison (vs power law):")
print(comparison.to_string(index=False))

# Visualise
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_power_law(pl_result, title="Citation Network Degree Distribution", ax=axes[0])

# Percolation analysis on the same network
perc = percolation_threshold_bond(G, n_trials=20)
axes[1].plot(perc["p_values"], perc["giant_fraction_mean"], color="steelblue", marker="o")
axes[1].fill_between(
    perc["p_values"],
    perc["giant_fraction_mean"] - perc["giant_fraction_std"],
    perc["giant_fraction_mean"] + perc["giant_fraction_std"],
    alpha=0.3, color="steelblue",
)
axes[1].axvline(perc["estimated_threshold"], linestyle="--", color="red",
                label=f"p_c ≈ {perc['estimated_threshold']:.2f}")
axes[1].set_xlabel("Bond occupation probability p")
axes[1].set_ylabel("Relative size of giant component")
axes[1].set_title("Bond Percolation on BA Network")
axes[1].legend()

plt.tight_layout()
plt.savefig("/tmp/citation_network_analysis.png", dpi=150)
plt.show()
print(f"\nEstimated percolation threshold: p_c ≈ {perc['estimated_threshold']:.3f}")
```

### Example B — Schelling Segregation Simulation with Varying Tolerance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOLERANCES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N_GRID = 25
STEPS = 80

results = []
for tol in TOLERANCES:
    df_run = run_schelling(n=N_GRID, tolerance=tol, steps=STEPS, seed=0)
    df_run["tolerance"] = tol
    results.append(df_run)

df_all = pd.concat(results, ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: % happy over time for each tolerance
for tol, grp in df_all.groupby("tolerance"):
    axes[0].plot(grp["step"], grp["pct_happy"], label=f"τ={tol}")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Fraction of Happy Agents")
axes[0].set_title("Schelling Model: Happiness vs Time")
axes[0].legend(title="Tolerance", fontsize=8)
axes[0].set_ylim(0, 1.05)

# Plot 2: final segregation index vs tolerance
final_seg = (
    df_all.groupby("tolerance")
    .apply(lambda g: g.sort_values("step").iloc[-1]["segregation_index"])
    .reset_index()
)
final_seg.columns = ["tolerance", "final_segregation"]
axes[1].plot(final_seg["tolerance"], final_seg["final_segregation"], marker="o", color="tomato")
axes[1].set_xlabel("Tolerance Threshold τ")
axes[1].set_ylabel("Final Segregation Index")
axes[1].set_title("Schelling Model: Segregation vs Tolerance")
axes[1].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig("/tmp/schelling_analysis.png", dpi=150)
plt.show()

# Print summary table
summary = df_all.groupby("tolerance").agg(
    final_happy=("pct_happy", "last"),
    final_segregation=("segregation_index", "last"),
    steps_to_converge=("step", "max"),
).reset_index()
print("\nSchelling Model Summary:")
print(summary.to_string(index=False))

# Hurst exponent on the happiness time-series for τ=0.5
ts_happy = df_all[df_all["tolerance"] == 0.5]["pct_happy"].values
hurst_result = compute_hurst(ts_happy)
print(f"\nHurst exponent of happiness time-series (τ=0.5): H = {hurst_result['hurst']:.4f}")
print(f"R² of log-log fit: {hurst_result['r_squared']:.4f}")

# Sample entropy of the same series
se = compute_sample_entropy(ts_happy, m=2)
print(f"Sample entropy: {se:.4f}")

# Information measures comparing tolerant vs strict agents
ts_tol = df_all[df_all["tolerance"] == 0.7]["pct_happy"].values
ts_strict = df_all[df_all["tolerance"] == 0.2]["pct_happy"].values
min_len = min(len(ts_tol), len(ts_strict))
info = compute_information_measures(ts_tol[:min_len], ts_strict[:min_len])
print(f"\nInformation measures (tolerant vs strict happiness series):")
print(f"  H(tolerant) = {info['entropy_x']:.4f} bits")
print(f"  H(strict)   = {info['entropy_y']:.4f} bits")
print(f"  MI          = {info['mutual_info']:.4f} bits")
print(f"  NMI         = {info['nmi']:.4f}")
```

---

## 9. Tips and Gotchas

- **Power-law vs lognormal**: Real-world data rarely follows a pure power law. The
  Vuong test frequently returns "inconclusive". Report both the fit quality and the
  comparison result; never claim power law on visual inspection alone.
- **Hurst exponent bias**: Short series (N < 500) give biased H estimates. Use
  detrended fluctuation analysis (DFA) for non-stationary series; the `nolds` package
  provides `nolds.hurst_rs` and `nolds.dfa`.
- **Schelling grid size**: A 20×20 grid with density=0.9 has ~360 agents; results are
  stochastic. Average over ≥10 seeds before drawing conclusions about the tolerance
  threshold.
- **Sample entropy runtime**: The naive O(N²) implementation above is slow for N > 1000.
  Use `antropy.sample_entropy` (C extension) for large time series.
- **Mesa 2.x API**: `RandomActivation` and `SingleGrid` are in `mesa.time` and
  `mesa.space` respectively. Mesa 3.x reorganised these; pin `mesa>=2.1,<3` or
  update the imports accordingly.
- **Percolation on directed graphs**: `nx.connected_components` only works on undirected
  graphs. For directed networks use `nx.strongly_connected_components`.

---

## 10. References

- Clauset, Shalizi & Newman (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4).
- Hurst, H. E. (1951). Long-term storage of reservoirs. *Trans. Am. Soc. Civil Eng.*, 116.
- Schelling, T. C. (1971). Dynamic models of segregation. *J. Mathematical Sociology*, 1(2).
- Richman & Moorman (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am. J. Physiology*.
- Newman, M. E. J. (2010). *Networks: An Introduction*. Oxford UP.
