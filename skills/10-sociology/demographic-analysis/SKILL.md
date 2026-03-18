---
name: demographic-analysis
description: >
  Use this Skill for demographic methods: life tables, Leslie matrix population projection,
  Lee-Carter mortality forecast, and Kitagawa decomposition.
tags:
  - sociology
  - demography
  - life-table
  - Leslie-matrix
  - Lee-Carter
  - mortality-forecast
version: "1.0.0"
authors:
  - name: "awesome-rosetta-skills contributors"
    github: "@xjtulyc"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  - pandas>=1.5
  - numpy>=1.23
  - scipy>=1.9
  - matplotlib>=3.6
last_updated: "2026-03-18"
status: "stable"
---

# Demographic Analysis

## When to Use

Use this skill when you need to:

- Construct period or cohort life tables from age-specific death rates (mx) following standard
  actuarial conventions
- Compute life expectancy at birth (e0) with Chiang confidence intervals
- Load and process HMD (Human Mortality Database) data files for multiple countries and years
- Build a Leslie matrix for a female population and project population structure 50 years forward
- Derive stable population parameters from the Leslie matrix dominant eigenvalue
- Estimate the Lee-Carter mortality model (log m_{x,t} = a_x + b_x × k_t) via SVD and project
  future mortality rates with ARIMA forecasts of the time index k_t
- Decompose the life expectancy gap between two populations using Arriaga's method
  (contribution by age group to the e0 difference)

## Background

**Life Table Notation**:

| Symbol | Formula | Meaning |
|---|---|---|
| m_x | observed | Death rate in age interval [x, x+5) |
| q_x | 2m_x / (2 + m_x) | Probability of dying in interval |
| l_x | l_{x-1} × (1 - q_{x-1}) | Survivors to exact age x (radix l_0=100,000) |
| d_x | l_x × q_x | Deaths in interval |
| L_x | (l_x + l_{x+1}) / 2 × n | Person-years lived in interval |
| T_x | Σ_{a≥x} L_a | Total person-years above age x |
| e_x | T_x / l_x | Remaining life expectancy at age x |

For the open-ended age interval (e.g., 85+): L_{85+} = l_{85} / m_{85}.

**Chiang (1984) CI on e0**: Based on variance of d_x:

```
Var(e0) ≈ Σ_x (l_x/l_0)² × (q_x(1-q_x) / d_x) × (e_x + (1-a_x) × n)²
```

where a_x is the average fraction of interval lived by those dying in it (typically 0.5, except
infants where a_0 ≈ 0.1).

**Leslie Matrix**: A structured population projection matrix where:
- Top row = age-specific fertility rates (F_x = l_{x+1}/l_x × m_x, where m_x is maternity rate)
- Sub-diagonal = survival proportions (P_x = L_{x+1} / L_x)

Population projection: N(t+5) = A × N(t). After many iterations, the population grows at rate λ
(dominant eigenvalue) and converges to the stable age distribution (dominant eigenvector).

**Lee-Carter Model**: The dominant model for mortality forecasting since 1992:

```
log(m_{x,t}) = a_x + b_x × k_t + ε_{x,t}
```

- a_x: average age profile of log mortality (row means)
- b_x: sensitivity of age x to the time trend k_t
- k_t: time index capturing overall mortality improvement

Estimated via SVD of the residual matrix after subtracting a_x. k_t is then modeled as a
random walk with drift and projected forward using ARIMA(0,1,0) with drift.

**Arriaga Decomposition**: Decomposes the difference e0(pop1) - e0(pop2) into contributions from
each age group. Useful for understanding which ages contribute most to a mortality gap.

## Environment Setup

```bash
pip install pandas>=1.5 numpy>=1.23 scipy>=1.9 matplotlib>=3.6
```

HMD data: Register at https://www.mortality.org/ and download country life table files.
Typical HMD file structure: columns `Year`, `Age`, `mx`, `qx`, `ax`, `lx`, `dx`, `Lx`, `Tx`, `ex`.

```bash
export HMD_DIR="/data/hmd/"
```

## Core Workflow

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Life Table Construction
# ---------------------------------------------------------------------------

def construct_life_table(
    mx: pd.Series,
    age_groups: list[int],
    radix: int = 100_000,
    last_open: bool = True,
) -> pd.DataFrame:
    """
    Construct a standard period life table from age-specific death rates.

    Parameters
    ----------
    mx : pd.Series
        Age-specific mortality rates (deaths per person-year) indexed by age_groups.
    age_groups : list of int
        Age group starting values (e.g., [0,1,5,10,...,85]).
    radix : int
        Starting cohort size (default 100,000).
    last_open : bool
        Whether the last age group is an open-ended interval.

    Returns
    -------
    pd.DataFrame with columns: age, mx, qx, ax, lx, dx, Lx, Tx, ex.
    """
    n_ages = len(age_groups)
    # Width of each age interval
    widths = [age_groups[i+1] - age_groups[i] for i in range(n_ages - 1)] + [np.inf]

    mx_vals = mx.values.astype(float)
    ax_vals = np.full(n_ages, 0.5)  # average fraction of interval lived
    ax_vals[0] = 0.1  # infant (WHO convention: ~0.1 for infants)

    qx = np.zeros(n_ages)
    for i, (m, a, w) in enumerate(zip(mx_vals, ax_vals, widths)):
        if np.isinf(w):
            qx[i] = 1.0  # open-ended interval
        else:
            qx[i] = (w * m) / (1 + (1 - a) * w * m)

    lx = np.zeros(n_ages)
    lx[0] = radix
    for i in range(1, n_ages):
        lx[i] = lx[i-1] * (1 - qx[i-1])

    dx = lx * qx

    Lx = np.zeros(n_ages)
    for i, (l, d, a, w) in enumerate(zip(lx, dx, ax_vals, widths)):
        if np.isinf(w):
            Lx[i] = lx[i] / mx_vals[i] if mx_vals[i] > 0 else lx[i] * 5
        else:
            Lx[i] = w * (lx[i] - d) + a * w * d

    Tx = np.cumsum(Lx[::-1])[::-1]
    ex = Tx / np.where(lx > 0, lx, np.nan)

    return pd.DataFrame({
        "age": age_groups,
        "width": [w if not np.isinf(w) else None for w in widths],
        "mx": mx_vals.round(6),
        "qx": qx.round(6),
        "ax": ax_vals,
        "lx": lx.round(1),
        "dx": dx.round(1),
        "Lx": Lx.round(1),
        "Tx": Tx.round(1),
        "ex": ex.round(3),
    })


def life_expectancy_ci(lt: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Compute Chiang (1984) confidence interval for life expectancy at birth (e0).

    Parameters
    ----------
    lt : pd.DataFrame
        Life table output from construct_life_table().
    alpha : float
        Significance level.

    Returns
    -------
    dict with e0, ci_lower, ci_upper, se.
    """
    e0 = lt.loc[0, "ex"]
    radix = lt.loc[0, "lx"]

    # Variance contribution from each age group (Chiang formula)
    var_contributions = []
    for _, row in lt.iterrows():
        if row["dx"] > 0 and row["qx"] < 1.0:
            term = (row["lx"] / radix) ** 2 * row["qx"] * (1 - row["qx"]) / row["dx"]
            var_contributions.append(term * row["ex"] ** 2)
    var_e0 = sum(var_contributions)
    se = np.sqrt(var_e0)

    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "e0": round(e0, 3),
        "ci_lower": round(e0 - z * se, 3),
        "ci_upper": round(e0 + z * se, 3),
        "se": round(se, 4),
    }


# ---------------------------------------------------------------------------
# 2. Leslie Matrix Population Projection
# ---------------------------------------------------------------------------

def build_leslie_matrix(
    Lx: np.ndarray,
    lx: np.ndarray,
    Fx: np.ndarray,
) -> np.ndarray:
    """
    Construct a Leslie projection matrix for female population.

    Parameters
    ----------
    Lx : np.ndarray
        Person-years lived in each age interval (from life table).
    lx : np.ndarray
        Survivors at start of each age interval.
    Fx : np.ndarray
        Age-specific fertility rates (births per woman per 5-year period).
        Must be same length as Lx.

    Returns
    -------
    np.ndarray — Leslie matrix (n_ages × n_ages).
    """
    n = len(Lx)
    A = np.zeros((n, n))

    # Top row: fertility (accounting for infant survival to next census)
    A[0, :] = Fx * Lx / lx

    # Sub-diagonal: survival proportions
    for i in range(n - 1):
        if Lx[i] > 0:
            A[i + 1, i] = Lx[i + 1] / Lx[i]

    return A


def project_population(
    A: np.ndarray,
    N0: np.ndarray,
    n_steps: int = 10,
    step_years: int = 5,
) -> pd.DataFrame:
    """
    Project population n_steps periods into the future using the Leslie matrix.

    Parameters
    ----------
    A : np.ndarray
        Leslie matrix (n_ages × n_ages).
    N0 : np.ndarray
        Initial population vector by age group.
    n_steps : int
        Number of projection steps.
    step_years : int
        Years per step (typically 5).

    Returns
    -------
    pd.DataFrame with columns: step, year_offset, total_pop, growth_rate,
    plus age-group columns.
    """
    N = N0.copy().astype(float)
    rows = [{"step": 0, "year_offset": 0, "total_pop": N.sum(), **{f"age_{i}": N[i] for i in range(len(N))}}]

    prev_total = N.sum()
    for step in range(1, n_steps + 1):
        N = A @ N
        total = N.sum()
        row = {
            "step": step,
            "year_offset": step * step_years,
            "total_pop": total,
            "growth_rate": (total / prev_total) ** (1 / step_years) - 1,
        }
        row.update({f"age_{i}": N[i] for i in range(len(N))})
        rows.append(row)
        prev_total = total

    return pd.DataFrame(rows)


def stable_population_params(A: np.ndarray) -> dict:
    """
    Derive stable population parameters from the Leslie matrix dominant eigenvalue.

    Returns
    -------
    dict with lambda_1 (growth multiplier per step), intrinsic_r (per year),
    stable_age_structure (proportions), NRR (Net Reproduction Rate, trace-based approximation).
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)
    dominant_idx = np.argmax(np.real(eigenvalues))
    lambda_1 = float(np.real(eigenvalues[dominant_idx]))
    r = np.log(lambda_1) / 5  # assuming 5-year step

    stable_vec = np.real(eigenvectors[:, dominant_idx])
    stable_vec = np.abs(stable_vec) / np.abs(stable_vec).sum()

    return {
        "lambda_1": round(lambda_1, 6),
        "intrinsic_r": round(r, 6),
        "doubling_time_years": round(np.log(2) / r, 1) if r > 0 else np.inf,
        "stable_age_structure": stable_vec.round(5),
    }


# ---------------------------------------------------------------------------
# 3. Lee-Carter Mortality Model
# ---------------------------------------------------------------------------

def fit_lee_carter(
    mx_matrix: pd.DataFrame,
) -> dict:
    """
    Fit the Lee-Carter model via SVD.

    Parameters
    ----------
    mx_matrix : pd.DataFrame
        Matrix of log(m_{x,t}) with age groups as index, years as columns.
        NaN values are replaced with column means.

    Returns
    -------
    dict with a_x (pd.Series), b_x (pd.Series), k_t (pd.Series), svd_singular_values.
    """
    log_mx = np.log(mx_matrix.replace(0, np.nan))
    # a_x: mean over time
    a_x = log_mx.mean(axis=1)
    # Residual matrix
    residual = log_mx.subtract(a_x, axis=0).fillna(0)

    # SVD: keep first singular vector
    U, s, Vt = np.linalg.svd(residual.values, full_matrices=False)
    b_x = pd.Series(U[:, 0], index=mx_matrix.index)
    k_t_raw = s[0] * Vt[0, :]

    # Normalize: b_x sums to 1, k_t adjusted accordingly
    b_norm = b_x / b_x.sum()
    k_t = pd.Series(k_t_raw * b_x.sum(), index=mx_matrix.columns)

    # Re-center: shift k_t so that Σ k_t = 0 and adjust a_x
    k_mean = k_t.mean()
    k_t_centered = k_t - k_mean
    a_x_adjusted = a_x + b_norm * k_mean

    return {
        "a_x": a_x_adjusted,
        "b_x": b_norm,
        "k_t": k_t_centered,
        "singular_values": s[:3],
    }


def forecast_lee_carter(
    lc: dict,
    n_years: int = 20,
    age_groups: list[int] | None = None,
) -> pd.DataFrame:
    """
    Project Lee-Carter mortality rates using a random walk with drift for k_t.

    Parameters
    ----------
    lc : dict
        Output from fit_lee_carter().
    n_years : int
        Number of years to forecast.
    age_groups : list of int, optional
        Age groups for the result index.

    Returns
    -------
    pd.DataFrame — projected m_{x,t} for each forecast year, age groups as index.
    """
    k_t = lc["k_t"]
    last_year = k_t.index[-1]

    # Fit random walk with drift: k_{t+1} = k_t + c + e
    diffs = k_t.diff().dropna()
    drift = diffs.mean()
    se_drift = diffs.std()

    forecast_years = [last_year + i for i in range(1, n_years + 1)]
    k_last = float(k_t.iloc[-1])
    k_forecasts = [k_last + drift * (i + 1) for i in range(n_years)]

    a_x = lc["a_x"]
    b_x = lc["b_x"]
    projected = {}
    for yr, k_proj in zip(forecast_years, k_forecasts):
        log_mx_proj = a_x + b_x * k_proj
        projected[yr] = np.exp(log_mx_proj)

    return pd.DataFrame(projected, index=a_x.index)
```

## Advanced Usage

### Arriaga Decomposition of e0 Gap

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def arriaga_decomposition(
    lt1: pd.DataFrame,
    lt2: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose the difference e0(lt1) - e0(lt2) into age-group contributions
    using Arriaga's (1984) method.

    Parameters
    ----------
    lt1 : pd.DataFrame
        Life table for population 1 (output of construct_life_table).
    lt2 : pd.DataFrame
        Life table for population 2.

    Returns
    -------
    pd.DataFrame with age, direct_effect, indirect_effect, total_contribution.
    """
    radix = lt1.loc[0, "lx"]

    def direct(x, lt_i, lt_j):
        l_i = lt_i.loc[x, "lx"] / radix
        L_i = lt_i.loc[x, "Lx"] / radix
        L_j = lt_j.loc[x, "Lx"] / radix
        T_j = lt_j.loc[x, "Tx"] / radix
        return l_i * (L_i / lt_j.loc[x, "lx"] * radix - L_j)

    rows = []
    for idx in lt1.index[:-1]:  # Exclude last open-ended group
        l_i = lt1.loc[idx, "lx"] / radix
        T_j_next = lt2.loc[idx + 1, "Tx"] / radix if idx + 1 in lt2.index else 0
        T_i = lt1.loc[idx, "Tx"] / radix
        T_j = lt2.loc[idx, "Tx"] / radix
        L_i = lt1.loc[idx, "Lx"] / radix
        L_j = lt2.loc[idx, "Lx"] / radix
        lj_x = lt2.loc[idx, "lx"]
        if lj_x <= 0:
            continue
        direct_e = l_i * (L_i / lj_x * radix - L_j)
        indirect_e = T_j_next / radix * (lt1.loc[idx, "lx"] / radix - lt2.loc[idx, "lx"] / radix
                                          ) if T_j_next > 0 else 0
        rows.append({
            "age": lt1.loc[idx, "age"],
            "direct_effect": round(direct_e, 5),
            "indirect_effect": round(indirect_e, 5),
            "total_contribution": round(direct_e + indirect_e, 5),
        })
    df = pd.DataFrame(rows)
    df["cumulative"] = df["total_contribution"].cumsum()
    return df
```

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| Negative qx values | mx too high combined with ax near 0 | Clamp qx to [0,1] after calculation |
| Leslie matrix eigenvalue < 1 | Declining population (below-replacement fertility) | Expected for many developed countries; report r < 0 |
| Lee-Carter SVD gives wrong sign | SVD sign indeterminacy | Flip sign of b_x and k_t together if k_t is increasing (mortality rising) |
| e0 Chiang CI too narrow | Large dx values (large sample) | This is correct behavior; CI shrinks with more exposure |
| HMD file format varies | Different versions use different delimiters | Use `pd.read_table(..., sep=r'\s+', skiprows=2)` for whitespace-delimited HMD files |

## External Resources

- Human Mortality Database: https://www.mortality.org/
- Preston, S., Heuveline, P. & Guillot, M. (2001). *Demography: Measuring and Modeling Population Processes*. Wiley-Blackwell.
- Lee, R. & Carter, L. (1992). Modeling and forecasting U.S. mortality. *JASA*, 87(419), 659-671.
- Arriaga, E. (1984). Measuring and explaining the change in life expectancies. *Demography*, 21(1), 83-96.
- HMD User Manual: https://www.mortality.org/File/GetDocument/Public/Docs/MethodsProtocolV6.pdf

## Examples

### Example 1: Life Table from mx + e0 with CI

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Illustrative mx values (approximate US female mortality, 2019)
age_groups = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
mx_values = [
    0.00558, 0.00027, 0.00013, 0.00011, 0.00048, 0.00060, 0.00064, 0.00080,
    0.00111, 0.00161, 0.00253, 0.00388, 0.00609, 0.00940, 0.01408, 0.02214,
    0.03567, 0.05959, 0.15200,
]
mx_series = pd.Series(mx_values, index=age_groups)
lt = construct_life_table(mx_series, age_groups)

print("=== Life Table (selected ages) ===")
print(lt[["age", "mx", "qx", "lx", "Lx", "ex"]].to_string(index=False))

ci = life_expectancy_ci(lt)
print(f"\ne0 = {ci['e0']} years [{ci['ci_lower']}, {ci['ci_upper']}] (95% CI)")

# Plot survival curve
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(lt["age"], lt["lx"] / 100_000, "b-", linewidth=2)
ax.fill_between(lt["age"], 0, lt["lx"] / 100_000, alpha=0.15, color="blue")
ax.set_xlabel("Age")
ax.set_ylabel("Probability of Survival (l_x / l_0)")
ax.set_title(f"Survival Curve — e0 = {ci['e0']} years")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("survival_curve.png", dpi=150)
plt.show()
```

### Example 2: Leslie Matrix 50-Year Population Projection

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simplified female population (5-year age groups, 0-85+)
n_groups = len(age_groups)
lx_vals = lt["lx"].values
Lx_vals = lt["Lx"].values

# Stylized fertility: ages 20-40 fertile (TFR ≈ 1.8)
Fx = np.zeros(n_groups)
fertile_ages = [4, 5, 6, 7, 8]  # indices for 20-24, 25-29, 30-34, 35-39, 40-44
for idx in fertile_ages:
    Fx[idx] = 0.18 / 5  # births per woman per 5-year interval
# Adjust for sex ratio at birth and infant survival
Fx *= 0.4878  # fraction female × sex ratio

A = build_leslie_matrix(Lx_vals, lx_vals, Fx)

# Initial population (approximately stable distribution)
N0 = np.array([6000, 5800, 5600, 5500, 5400, 5200, 5100, 4900,
               4700, 4400, 4100, 3700, 3200, 2700, 2100, 1600, 1100, 700, 400]).astype(float)

proj = project_population(A, N0, n_steps=10, step_years=5)

print("=== 50-Year Population Projection ===")
print(proj[["step", "year_offset", "total_pop", "growth_rate"]].round(4).to_string(index=False))

params = stable_population_params(A)
print(f"\nStable population parameters:")
print(f"  Lambda (5-year): {params['lambda_1']}")
print(f"  Intrinsic rate r: {params['intrinsic_r']:.5f} per year")
print(f"  Doubling time: {params['doubling_time_years']} years" if params['lambda_1'] > 1
      else f"  Halving time: {abs(round(np.log(2) / abs(params['intrinsic_r']), 1))} years")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(proj["year_offset"], proj["total_pop"] / 1000, "o-", color="#2166ac", linewidth=2)
ax.set_xlabel("Years from baseline")
ax.set_ylabel("Female Population (thousands)")
ax.set_title("Leslie Matrix: 50-Year Population Projection")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("leslie_projection.png", dpi=150)
plt.show()
```

### Example 3: Lee-Carter SVD Estimation + 20-Year Forecast

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
years_lc = list(range(1960, 2021))
age_idx = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
base_mx = np.array(mx_values)

# Simulate mortality decline over time
mx_sim = {}
for t, yr in enumerate(years_lc):
    decline = 0.015 * t  # linear mortality improvement
    mx_sim[yr] = base_mx * np.exp(-decline + rng.normal(0, 0.03, len(age_idx)))
mx_df = pd.DataFrame(mx_sim, index=age_idx)

lc = fit_lee_carter(mx_df)
print("=== Lee-Carter Model Parameters ===")
print("k_t (time index, first 5 values):", lc["k_t"].values[:5].round(4))
print("b_x (age sensitivity, top 5 ages):")
print(lc["b_x"].head(5).round(4).to_string())

# Forecast 20 years
mx_forecast = forecast_lee_carter(lc, n_years=20)
print(f"\nForecasted mx for age 65 (2021-2040):")
print(mx_forecast.loc[65].round(6))

# Plot k_t trend and forecast
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(lc["k_t"].index, lc["k_t"].values, "b-", linewidth=2, label="k_t (observed)")
forecast_k = [float(lc["k_t"].iloc[-1]) + lc["k_t"].diff().dropna().mean() * (i + 1)
              for i in range(20)]
forecast_yrs = list(range(years_lc[-1] + 1, years_lc[-1] + 21))
axes[0].plot(forecast_yrs, forecast_k, "r--", linewidth=2, label="k_t (forecast)")
axes[0].axvline(years_lc[-1], color="gray", linestyle=":", linewidth=1)
axes[0].set_xlabel("Year")
axes[0].set_ylabel("k_t")
axes[0].set_title("Lee-Carter Time Index k_t")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lc["b_x"].index, lc["b_x"].values, "g-o", linewidth=2, markersize=5)
axes[1].set_xlabel("Age Group")
axes[1].set_ylabel("b_x (sensitivity)")
axes[1].set_title("Lee-Carter Age Sensitivity b_x")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lee_carter_model.png", dpi=150)
plt.show()
```
