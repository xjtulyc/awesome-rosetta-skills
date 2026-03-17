---
name: qiskit-quantum
description: >
  Use this Skill for quantum computing experiments with Qiskit: qubit circuits,
  gates, measurement, VQE/QAOA variational algorithms, noise models, and
  statevector vs shot simulation.
tags:
  - physics
  - quantum-computing
  - Qiskit
  - variational-algorithms
  - quantum-simulation
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
    - qiskit>=0.45
    - qiskit-aer>=0.13
    - qiskit-algorithms>=0.3
    - numpy>=1.23
    - matplotlib>=3.6
last_updated: "2026-03-17"
status: stable
---

# Qiskit Quantum Computing

> **TL;DR** — Build and simulate quantum circuits with Qiskit. Use statevector
> or shot-based simulation, add realistic noise models, implement VQE for ground
> state energy estimation, and QAOA for combinatorial optimization (MaxCut).

---

## When to Use

Use this Skill when you need to:

- Design and test quantum gate circuits (H, CNOT, Ry, Rz, T, S, SWAP)
- Simulate quantum circuits on classical hardware (Aer statevector / QASM)
- Add realistic noise (depolarizing, bit-flip, thermal relaxation) for NISQ device modeling
- Run Variational Quantum Eigensolver (VQE) for quantum chemistry ground state energy
- Apply QAOA to combinatorial optimization problems (MaxCut, Max-2-SAT)
- Transpile circuits for a target backend with optimization

Do **not** use this Skill when:
- You need classical exact diagonalization of Hamiltonians → use SciPy eigsh
- You want to run on real IBM quantum hardware → use `qiskit-ibm-runtime`
- You need fault-tolerant error correction → use Stim for stabilizer circuits

---

## Background & Key Concepts

### Quantum Circuits

A quantum circuit acts on n qubits (initialized to |0⟩) by applying unitary gates,
then measuring in the computational basis.

| Gate | Matrix | Description |
|---|---|---|
| H (Hadamard) | `[[1,1],[1,-1]]/√2` | Creates superposition |
| CNOT | `[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]` | Entangles two qubits |
| Ry(θ) | `[[cos θ/2, -sin θ/2],[sin θ/2, cos θ/2]]` | Rotation around Y axis |
| Rz(φ) | `[[e^{-iφ/2},0],[0,e^{iφ/2}]]` | Rotation around Z axis |

### Variational Quantum Algorithms

VQE and QAOA are hybrid classical-quantum algorithms:
1. Parameterized quantum circuit (ansatz) prepares a trial state |ψ(θ)⟩
2. Expectation value ⟨ψ(θ)|H|ψ(θ)⟩ is estimated by repeated measurement
3. Classical optimizer updates θ to minimize ⟨H⟩

### Noise Models

NISQ devices suffer from:
- **Depolarizing noise**: random Pauli errors after each gate
- **Bit-flip**: X gate applied with probability p
- **Thermal relaxation**: T1 (amplitude damping) and T2 (dephasing) decay

---

## Environment Setup

```bash
# Create isolated environment
conda create -n qiskit-env python=3.11 -y
conda activate qiskit-env

# Install Qiskit and Aer simulator
pip install qiskit qiskit-aer qiskit-algorithms qiskit-nature numpy matplotlib

# Verify
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
python -c "from qiskit_aer import AerSimulator; print('Aer OK')"
```

---

## Core Workflow

### Step 1 — Bell State Circuit and Measurement

```python
"""
Build a Bell state circuit, simulate with statevector and shot-based simulators,
and plot the measurement histogram.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram


def create_bell_circuit() -> QuantumCircuit:
    """
    Create a 2-qubit Bell state circuit |Φ+⟩ = (|00⟩ + |11⟩)/√2.

    Circuit:
        q0: ──H──●──
        q1: ─────X──
    """
    qc = QuantumCircuit(2, 2)
    qc.h(0)       # Hadamard on qubit 0
    qc.cx(0, 1)   # CNOT: control=0, target=1
    qc.measure([0, 1], [0, 1])
    return qc


def simulate_statevector(qc_no_measure: QuantumCircuit) -> np.ndarray:
    """
    Get exact statevector amplitudes before measurement.

    Args:
        qc_no_measure: Circuit without measurement instructions.

    Returns:
        Complex statevector array of length 2^n.
    """
    sim = AerSimulator(method="statevector")
    qc_sv = qc_no_measure.copy()
    qc_sv.save_statevector()
    job = sim.run(transpile(qc_sv, sim))
    result = job.result()
    sv = np.array(result.get_statevector())
    return sv


def simulate_shots(qc: QuantumCircuit, n_shots: int = 4096) -> dict:
    """
    Run shot-based simulation and return measurement counts.

    Args:
        qc:      Circuit with measurement gates.
        n_shots: Number of repetitions.

    Returns:
        Dictionary of bitstring: count.
    """
    sim = AerSimulator(method="qasm")
    job = sim.run(transpile(qc, sim), shots=n_shots)
    return job.result().get_counts()


def plot_counts(counts: dict, title: str = "Bell State", output: str = "bell.png") -> None:
    """Plot measurement histogram."""
    fig, ax = plt.subplots(figsize=(5, 4))
    states = sorted(counts.keys())
    vals   = [counts[s] / sum(counts.values()) for s in states]
    ax.bar(states, vals, color="#4C72B0")
    ax.set_xlabel("Measurement outcome")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Histogram saved to {output}")


if __name__ == "__main__":
    # Statevector
    qc_bare = QuantumCircuit(2)
    qc_bare.h(0)
    qc_bare.cx(0, 1)
    sv = simulate_statevector(qc_bare)
    print("Statevector amplitudes:", sv)
    print(f"|00⟩ probability: {abs(sv[0])**2:.4f}")
    print(f"|11⟩ probability: {abs(sv[3])**2:.4f}")

    # Shot-based
    qc = create_bell_circuit()
    counts = simulate_shots(qc, n_shots=8192)
    print("Shot counts:", counts)
    plot_counts(counts, title="Bell State |Φ+⟩", output="bell_histogram.png")
```

### Step 2 — VQE for H₂ Ground State Energy

```python
"""
VQE: Variational Quantum Eigensolver for estimating the ground state energy
of a simple 2-qubit Hamiltonian representing H₂ at fixed bond length.

H = -1.0523 * II + 0.3979 * ZI - 0.3979 * IZ - 0.0112 * ZZ + 0.1809 * XX
(Jordan-Wigner encoding of H₂ minimal basis Hamiltonian)
"""

import numpy as np
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
import matplotlib.pyplot as plt


def build_h2_hamiltonian() -> SparsePauliOp:
    """
    Construct the H₂ Hamiltonian in the minimal (STO-3G) basis at R=0.735 Å.

    Returns:
        SparsePauliOp representing the qubit Hamiltonian (energy in Hartree).
    """
    # Pauli coefficients from Jordan-Wigner transformation
    # Strings are in reverse qubit order (Qiskit convention)
    H = SparsePauliOp.from_list([
        ("II", -1.0523732),
        ("IZ",  0.3979374),
        ("ZI", -0.3979374),
        ("ZZ", -0.0112801),
        ("XX",  0.1809312),
    ])
    return H


def run_vqe(
    hamiltonian: SparsePauliOp,
    reps: int = 2,
    max_iter: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run VQE with EfficientSU2 ansatz to find ground state energy.

    Args:
        hamiltonian: Target Hamiltonian as SparsePauliOp.
        reps:        Number of repetition layers in EfficientSU2.
        max_iter:    Maximum optimizer iterations.
        seed:        Random seed for reproducibility.

    Returns:
        Dictionary with keys: energy, optimal_params, num_evals.
    """
    n_qubits = hamiltonian.num_qubits

    # Ansatz: hardware-efficient EfficientSU2
    ansatz = EfficientSU2(n_qubits, reps=reps, entanglement="linear")
    print(f"Ansatz parameters: {ansatz.num_parameters}")

    # Estimator primitive (Aer)
    estimator = Estimator(approximation=True)

    # COBYLA optimizer (gradient-free)
    optimizer = COBYLA(maxiter=max_iter)

    np.random.seed(seed)
    initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)

    # Track convergence
    energies = []

    def callback(nfev, params, energy, stddev):
        energies.append(energy)
        if len(energies) % 50 == 0:
            print(f"  Iter {len(energies):4d}: E = {energy:.6f} Ha")

    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer,
              callback=callback, initial_point=initial_params)
    result = vqe.compute_minimum_eigenvalue(hamiltonian)

    print(f"\nVQE ground state energy: {result.eigenvalue.real:.6f} Hartree")
    print(f"  FCI (exact) reference: -1.137270 Hartree")
    print(f"  Error: {abs(result.eigenvalue.real - (-1.137270)):.2e} Ha")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(energies, color="#2196F3", linewidth=1.2)
    ax.axhline(-1.137270, color="red", linestyle="--", linewidth=0.8, label="FCI energy")
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("VQE Convergence for H₂")
    ax.legend()
    fig.tight_layout()
    fig.savefig("vqe_convergence.png", dpi=150)

    return {
        "energy": result.eigenvalue.real,
        "optimal_params": result.optimal_parameters,
        "num_evals": result.cost_function_evals,
        "convergence": energies,
    }


if __name__ == "__main__":
    H = build_h2_hamiltonian()
    result = run_vqe(H, reps=2, max_iter=300)
    print(f"Converged in {result['num_evals']} function evaluations")
```

### Step 3 — QAOA for MaxCut on a 4-Node Graph

```python
"""
QAOA: Quantum Approximate Optimization Algorithm for the MaxCut problem.

MaxCut: partition graph nodes into two sets S and V\S to maximize the number
of edges between them.

Cost Hamiltonian: H_C = Σ_{(i,j)∈E} (I - Z_i Z_j) / 2
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA


def build_maxcut_hamiltonian(graph: nx.Graph) -> SparsePauliOp:
    """
    Build the MaxCut cost Hamiltonian from a NetworkX graph.

    H_C = Σ_{(i,j)∈E} (I - Z_i Z_j) / 2

    Args:
        graph: NetworkX graph with integer node labels 0..n-1.

    Returns:
        SparsePauliOp for the MaxCut Hamiltonian.
    """
    n = graph.number_of_nodes()
    pauli_list = []
    for u, v in graph.edges():
        # Z_u Z_v term: Pauli string of length n with Z at positions u and v
        zz_str = ["I"] * n
        zz_str[u] = "Z"
        zz_str[v] = "Z"
        # Qiskit uses reversed qubit ordering
        pauli_list.append(("".join(reversed(zz_str)), -0.5))
        pauli_list.append(("I" * n, 0.5))

    return SparsePauliOp.from_list(pauli_list)


def run_qaoa_maxcut(
    graph: nx.Graph,
    p: int = 2,
    max_iter: int = 200,
) -> dict:
    """
    Run QAOA to solve MaxCut on the given graph.

    Args:
        graph: NetworkX graph.
        p:     QAOA depth (number of cost+mixer layers).
        max_iter: Maximum optimizer iterations.

    Returns:
        Dictionary with keys: best_cut, best_bitstring, counts, energy.
    """
    hamiltonian = build_maxcut_hamiltonian(graph)
    n = graph.number_of_nodes()

    sampler = Sampler()
    optimizer = COBYLA(maxiter=max_iter)

    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
    result = qaoa.compute_minimum_eigenvalue(hamiltonian)

    # Get best measurement outcome
    counts = result.best_measurement
    best_bitstring = result.best_measurement["bitstring"]
    partition = [int(b) for b in best_bitstring]

    # Count cut edges
    cut_value = sum(
        1 for u, v in graph.edges() if partition[u] != partition[v]
    )
    max_possible = graph.number_of_edges()

    print(f"QAOA (p={p}) MaxCut result:")
    print(f"  Best bitstring: {best_bitstring}")
    print(f"  Cut edges: {cut_value} / {max_possible}")
    print(f"  Approximation ratio: {cut_value / max_possible:.3f}")

    return {
        "best_bitstring": best_bitstring,
        "cut_value": cut_value,
        "energy": result.eigenvalue.real,
    }


def plot_maxcut_result(graph: nx.Graph, bitstring: str, output: str = "maxcut.png") -> None:
    """Visualize the MaxCut partition on the graph."""
    partition = [int(b) for b in bitstring]
    colors = ["#E91E63" if p == 0 else "#2196F3" for p in partition]

    fig, ax = plt.subplots(figsize=(5, 4))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, ax=ax, node_color=colors, with_labels=True,
            node_size=600, font_color="white", font_weight="bold")
    cut_edges = [(u, v) for u, v in graph.edges() if partition[u] != partition[v]]
    nx.draw_networkx_edges(graph, pos, edgelist=cut_edges, ax=ax,
                           edge_color="green", width=2.5, style="dashed")
    ax.set_title(f"MaxCut: {len(cut_edges)} edges cut\n(cut edges in dashed green)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"MaxCut visualization saved to {output}")


if __name__ == "__main__":
    # 4-node cycle graph (max cut = 4)
    G = nx.cycle_graph(4)
    G.add_edge(0, 2)  # 5-edge graph with max cut = 4
    result = run_qaoa_maxcut(G, p=2)
    plot_maxcut_result(G, result["best_bitstring"], "maxcut_result.png")
```

---

## Advanced Usage

### Noise Model Simulation

```python
"""
Add a realistic noise model to a circuit and compare noisy vs ideal results.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


def build_noise_model(
    p_depol_1q: float = 0.001,
    p_depol_2q: float = 0.01,
    T1: float = 50e3,   # ns
    T2: float = 70e3,   # ns
    gate_time_1q: float = 50,   # ns
    gate_time_2q: float = 300,  # ns
) -> NoiseModel:
    """
    Build a noise model with depolarizing + thermal relaxation errors.

    Args:
        p_depol_1q:   Depolarizing probability for single-qubit gates.
        p_depol_2q:   Depolarizing probability for two-qubit gates.
        T1:           Longitudinal relaxation time (ns).
        T2:           Transverse relaxation time (ns).
        gate_time_1q: Duration of single-qubit gates (ns).
        gate_time_2q: Duration of two-qubit gates (ns).

    Returns:
        Qiskit NoiseModel.
    """
    noise_model = NoiseModel()

    # Thermal relaxation (T1, T2 decoherence)
    error_1q = thermal_relaxation_error(T1, T2, gate_time_1q)
    error_2q = thermal_relaxation_error(T1, T2, gate_time_2q).expand(
                 thermal_relaxation_error(T1, T2, gate_time_2q))

    # Depolarizing noise
    dep_1q = depolarizing_error(p_depol_1q, 1)
    dep_2q = depolarizing_error(p_depol_2q, 2)

    # Combine: apply both thermal + depolarizing
    noise_model.add_all_qubit_quantum_error(error_1q.compose(dep_1q), ["h", "ry", "rz"])
    noise_model.add_all_qubit_quantum_error(error_2q.compose(dep_2q), ["cx"])

    return noise_model


def compare_ideal_vs_noisy(n_qubits: int = 4, n_shots: int = 8192) -> None:
    """Build a GHZ state circuit and compare ideal and noisy measurements."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    # Ideal simulation
    ideal_sim = AerSimulator()
    ideal_counts = ideal_sim.run(transpile(qc, ideal_sim), shots=n_shots).result().get_counts()

    # Noisy simulation
    noise_model = build_noise_model()
    noisy_sim = AerSimulator(noise_model=noise_model)
    noisy_counts = noisy_sim.run(transpile(qc, noisy_sim), shots=n_shots).result().get_counts()

    # Compare fidelity
    all_states = set(ideal_counts) | set(noisy_counts)
    ideal_probs = {s: ideal_counts.get(s, 0) / n_shots for s in all_states}
    noisy_probs = {s: noisy_counts.get(s, 0) / n_shots for s in all_states}
    fidelity = sum(np.sqrt(ideal_probs[s] * noisy_probs[s]) for s in all_states) ** 2
    print(f"GHZ state ({n_qubits} qubits) fidelity (noisy vs ideal): {fidelity:.4f}")
```

### Circuit Transpilation and Optimization

```python
"""
Transpile a circuit to a specific basis gate set with optimization.
"""

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit_aer import AerSimulator


def transpile_and_compare(qc: QuantumCircuit) -> None:
    """Show circuit depth before and after transpilation."""
    backend = AerSimulator()
    print(f"Original circuit depth: {qc.depth()}")
    print(f"Original gate count:    {qc.count_ops()}")

    for opt_level in [0, 1, 2, 3]:
        qc_t = transpile(qc, backend=backend, optimization_level=opt_level)
        print(f"Opt level {opt_level}: depth={qc_t.depth()}, "
              f"cx gates={qc_t.count_ops().get('cx', 0)}")
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ProviderError: No Backend found` | Old IBM provider API | Use `qiskit-ibm-runtime` for real hardware |
| `ValueError: The instruction save_statevector is not supported` | Wrong simulator method | Use `AerSimulator(method="statevector")` |
| VQE not converging | Poor initial parameters or ansatz | Try `SPSA` optimizer; increase `reps`; use parameter shift gradient |
| QAOA gives trivial bitstring | p too small | Increase p (circuit depth); try p=3 or p=4 |
| Slow VQE with Estimator | Too many shots per evaluation | Set `approximation=True` in `Estimator` |
| `DeprecationWarning: qiskit.opflow` | Opflow removed in Qiskit 1.0 | Use `qiskit.quantum_info.SparsePauliOp` |
| Measurement all zeros after noise | Noise too strong | Reduce `p_depol_2q`; check T1/T2 values |

---

## External Resources

- Qiskit documentation: <https://docs.quantum.ibm.com>
- Qiskit Textbook: <https://learning.quantum.ibm.com>
- Qiskit Aer noise models: <https://qiskit.github.io/qiskit-aer/apidocs/noise.html>
- VQE original paper: Peruzzo et al., Nature Communications 5, 4213 (2014)
- QAOA original paper: Farhi et al., arXiv:1411.4028 (2014)

---

## Examples

### Example 1 — Bell State Full Pipeline

```python
if __name__ == "__main__":
    print("=== Bell State Simulation ===")
    qc_bare = QuantumCircuit(2)
    qc_bare.h(0)
    qc_bare.cx(0, 1)
    sv = simulate_statevector(qc_bare)
    print(f"Statevector: {sv}")
    qc = create_bell_circuit()
    counts = simulate_shots(qc, n_shots=8192)
    print(f"Counts: {counts}")
    plot_counts(counts, "Bell State |Φ+⟩", "bell_result.png")
```

### Example 2 — GHZ State with Noise Comparison

```python
if __name__ == "__main__":
    print("=== GHZ State Noise Analysis ===")
    for n in [2, 3, 4, 5]:
        compare_ideal_vs_noisy(n_qubits=n, n_shots=4096)
```

### Example 3 — QAOA MaxCut Scaling

```python
if __name__ == "__main__":
    print("=== QAOA MaxCut ===")
    import networkx as nx

    graphs = {
        "cycle_4":  nx.cycle_graph(4),
        "petersen": nx.petersen_graph(),
        "cycle_6":  nx.cycle_graph(6),
    }

    for name, G in graphs.items():
        print(f"\nGraph: {name} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        result = run_qaoa_maxcut(G, p=2, max_iter=150)
        print(f"  Best cut: {result['cut_value']}")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — Bell state, VQE H₂, QAOA MaxCut, noise models, transpilation |
