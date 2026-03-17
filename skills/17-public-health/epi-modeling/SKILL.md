---
name: epi-modeling
description: >
  Compartmental epidemic modeling (SIR/SEIR/SEIHR) with parameter fitting,
  Rt estimation, age-structured models, and sensitivity analysis.
tags:
  - epidemiology
  - seir
  - compartmental-model
  - rt-estimation
  - public-health
  - scipy
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
  - numpy>=1.24.0
  - scipy>=1.11.0
  - pandas>=2.0.0
  - matplotlib>=3.7.0
  - numba>=0.57.0
last_updated: "2026-03-17"
---

# Epidemic Compartmental Modeling Skill

This skill covers the mathematical and computational workflow for building, fitting, and
analysing compartmental epidemic models in Python. Starting from classic SIR and SEIR
formulations, it extends to age-structured models, hospitalisation compartments (SEIHR),
real-time reproduction number estimation, and global sensitivity analysis via Partial Rank
Correlation Coefficients (PRCC).

All numerical integration is done with `scipy.integrate.solve_ivp` using the Radau solver,
which handles stiff ODE systems arising from wide parameter ranges common in outbreak modeling.

---

## Setup

```bash
pip install numpy scipy pandas matplotlib numba
```

---

## Core Functions

```python
"""
epi_modeling.py
---------------
Compartmental epidemic model utilities: SIR, SEIR, SEIHR, Rt estimation,
parameter fitting, age-structured models, and sensitivity analysis.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, differential_evolution
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 1. SEIR Model
# ---------------------------------------------------------------------------

def build_seir_model(
    beta: float,
    sigma: float,
    gamma: float,
    N: int,
    mu: float = 0.0,
) -> Callable:
    """
    Construct and return a SEIR ODE right-hand-side function.

    Model compartments
    ------------------
    S : Susceptible
    E : Exposed (infected but not yet infectious)
    I : Infectious
    R : Recovered / Removed

    Equations
    ---------
        dS/dt = mu*(N - S) - beta*S*I/N
        dE/dt = beta*S*I/N - (sigma + mu)*E
        dI/dt = sigma*E - (gamma + mu)*I
        dR/dt = gamma*I - mu*R

    Parameters
    ----------
    beta : float
        Transmission rate (contacts per day × probability of transmission).
    sigma : float
        Incubation rate (1 / mean_incubation_period_days).
    gamma : float
        Recovery rate (1 / mean_infectious_period_days).
    N : int
        Total population size (assumed constant when mu=0).
    mu : float, optional
        Birth/death rate (1 / life_expectancy_days). Default 0 (no demography).

    Returns
    -------
    Callable
        ODE function compatible with ``scipy.integrate.solve_ivp``.
        Signature: f(t, y) -> [dS, dE, dI, dR]

    Examples
    --------
    >>> seir = build_seir_model(beta=0.3, sigma=1/5.2, gamma=1/10, N=1_000_000)
    >>> sol = solve_ivp(seir, [0, 180], [999990, 5, 5, 0],
    ...                 t_eval=np.arange(181), method="Radau")
    """
    def seir_ode(t: float, y: np.ndarray) -> List[float]:
        S, E, I, R = y
        force_of_infection = beta * S * I / N
        dS = mu * (N - S) - force_of_infection
        dE = force_of_infection - (sigma + mu) * E
        dI = sigma * E - (gamma + mu) * I
        dR = gamma * I - mu * R
        return [dS, dE, dI, dR]

    seir_ode.__doc__ = (f"SEIR ODE: beta={beta:.4f}, sigma={sigma:.4f}, "
                        f"gamma={gamma:.4f}, N={N}, R0={beta/gamma:.2f}")
    return seir_ode


def build_sir_model(beta: float, gamma: float, N: int) -> Callable:
    """
    Construct and return an SIR ODE function.

    dS/dt = -beta*S*I/N
    dI/dt =  beta*S*I/N - gamma*I
    dR/dt =  gamma*I
    """
    def sir_ode(t, y):
        S, I, R = y
        foi = beta * S * I / N
        return [-foi, foi - gamma * I, gamma * I]

    return sir_ode


def build_seihr_model(
    beta: float,
    sigma: float,
    gamma: float,
    gamma_h: float,
    hosp_rate: float,
    N: int,
) -> Callable:
    """
    SEIR extended with Hospitalisation (H) and fatality-adjusted Removed (R).

    Additional compartments
    -----------------------
    H : Hospitalised (a fraction of infectious individuals)
    R : Recovered (from community) + hospital discharges

    Parameters
    ----------
    gamma_h : float
        Hospital discharge rate (1 / mean_hospital_stay_days).
    hosp_rate : float
        Fraction of infectious individuals requiring hospitalisation (0–1).
    """
    def seihr_ode(t, y):
        S, E, I, H, R = y
        foi = beta * S * I / N
        dS = -foi
        dE = foi - sigma * E
        dI = sigma * E - gamma * I
        dH = hosp_rate * gamma * I - gamma_h * H
        dR = (1 - hosp_rate) * gamma * I + gamma_h * H
        return [dS, dE, dI, dH, dR]

    return seihr_ode


# ---------------------------------------------------------------------------
# 2. Parameter Fitting
# ---------------------------------------------------------------------------

def fit_model_to_data(
    incidence: np.ndarray,
    dates: Union[np.ndarray, pd.DatetimeIndex],
    model_func: Callable,
    N: int,
    param_bounds: Dict[str, Tuple[float, float]],
    compartment_idx: int = 2,
    initial_exposed: int = 10,
    method: str = "differential_evolution",
) -> Dict:
    """
    Fit a compartmental model to observed incidence data using least-squares
    optimisation.

    Parameters
    ----------
    incidence : np.ndarray
        Daily or weekly new case counts (length T).
    dates : array-like
        Date array corresponding to incidence (length T).
    model_func : Callable
        A function that accepts (beta, ...) and returns a scipy solve_ivp ODE
        callable. Must be consistent with ``param_bounds`` keys.
    N : int
        Population size.
    param_bounds : dict
        Dictionary of {parameter_name: (lower_bound, upper_bound)}.
        Keys determine order of optimisation variables.
    compartment_idx : int
        Index of the ODE state vector corresponding to the observed count
        (e.g. 2 for I in SEIR with states [S,E,I,R]).
    initial_exposed : int
        Assumed initial exposed individuals (E0).
    method : str
        Optimisation method: 'differential_evolution' (global, robust) or
        'Nelder-Mead' (fast, local).

    Returns
    -------
    dict with keys:
        - 'params'     : fitted parameter dict
        - 'solution'   : solve_ivp solution object
        - 'fitted_incidence': np.ndarray of modelled new cases
        - 'rmse'       : root mean squared error
        - 'R0'         : basic reproduction number (beta/gamma if present)
    """
    T = len(incidence)
    t_eval = np.arange(T)

    # Initial conditions: one infectious seed
    I0 = 1
    S0 = N - I0 - initial_exposed
    param_names = list(param_bounds.keys())
    bounds_list = [param_bounds[k] for k in param_names]

    def _residuals(params_vec):
        kwargs = dict(zip(param_names, params_vec))
        kwargs["N"] = N
        try:
            ode = model_func(**kwargs)
        except TypeError:
            return 1e12

        y0_len = ode.__code__.co_varcount - 2  # rough estimate; override below
        # Determine y0 from compartment count via a probe call
        n_states = 4 if "seir" in getattr(ode, "__doc__", "").lower() else 3
        if "seihr" in getattr(model_func, "__name__", "").lower():
            n_states = 5

        y0_map = {
            3: [S0,             I0, 0],
            4: [S0, initial_exposed, I0, 0],
            5: [S0, initial_exposed, I0, 0, 0],
        }
        y0 = y0_map.get(n_states, [S0, initial_exposed, I0, 0])

        try:
            sol = solve_ivp(
                ode, [0, T - 1], y0,
                t_eval=t_eval, method="Radau",
                max_step=1.0, dense_output=False,
            )
            if not sol.success:
                return 1e12
            # Convert cumulative to incidence via daily difference
            cumulative = sol.y[compartment_idx - 1] + sol.y[compartment_idx]
            modelled = np.maximum(np.diff(cumulative, prepend=cumulative[0]), 0)
            return float(np.sqrt(np.mean((modelled - incidence) ** 2)))
        except Exception:
            return 1e12

    if method == "differential_evolution":
        result = differential_evolution(
            _residuals, bounds_list,
            maxiter=1000, tol=1e-6, seed=42,
            workers=1, polish=True,
        )
    else:
        x0 = [(lo + hi) / 2 for lo, hi in bounds_list]
        result = minimize(_residuals, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-6})

    fitted_params = dict(zip(param_names, result.x))
    fitted_params["N"] = N
    ode_fitted = model_func(**fitted_params)

    n_states_f = 4
    y0_final = [S0, initial_exposed, I0, 0]
    sol_final = solve_ivp(
        ode_fitted, [0, T - 1], y0_final,
        t_eval=t_eval, method="Radau", max_step=1.0,
    )
    cumulative_f = sol_final.y[compartment_idx - 1] + sol_final.y[compartment_idx]
    fitted_inc = np.maximum(np.diff(cumulative_f, prepend=cumulative_f[0]), 0)

    R0 = None
    if "beta" in fitted_params and "gamma" in fitted_params:
        R0 = fitted_params["beta"] / fitted_params["gamma"]

    return {
        "params": {k: v for k, v in fitted_params.items() if k != "N"},
        "solution": sol_final,
        "fitted_incidence": fitted_inc,
        "rmse": result.fun,
        "R0": R0,
    }


# ---------------------------------------------------------------------------
# 3. Rt Estimation (sliding-window exponential growth)
# ---------------------------------------------------------------------------

def estimate_rt(
    incidence: np.ndarray,
    si_mean: float,
    si_sd: float,
    window: int = 7,
    quantiles: Tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """
    Estimate the time-varying reproduction number Rt using the Wallinga-Lipsitch
    exponential growth method with a discretised serial interval distribution.

    Parameters
    ----------
    incidence : np.ndarray
        Daily new case counts (length T). Should be smoothed (7-day rolling average).
    si_mean : float
        Mean serial interval in days.
    si_sd : float
        Standard deviation of serial interval in days.
    window : int
        Rolling window size in days for growth rate estimation.
    quantiles : tuple
        Lower and upper quantile for the uncertainty interval (default 95% CI).

    Returns
    -------
    pd.DataFrame with columns: ['Rt', 'Rt_lower', 'Rt_upper', 'r', 'valid']

    Notes
    -----
    The relationship Rt = 1 / M(-r) where M is the moment generating function of
    the serial interval distribution is used (Wallinga & Lipsitch 2007).
    For a gamma-distributed SI with mean mu and sd sigma:
        Rt = (1 + r * sigma^2 / mu)^(mu^2 / sigma^2)
    """
    from scipy.stats import gamma as gamma_dist

    T = len(incidence)
    # Fit gamma distribution to serial interval
    cv2 = (si_sd / si_mean) ** 2
    k = 1.0 / cv2          # shape
    theta = si_mean * cv2  # scale

    rt_vals   = np.full(T, np.nan)
    rt_lower  = np.full(T, np.nan)
    rt_upper  = np.full(T, np.nan)
    r_vals    = np.full(T, np.nan)

    inc_smooth = pd.Series(incidence).rolling(window, center=True,
                                               min_periods=3).mean().values

    for t in range(window, T - window):
        segment = inc_smooth[t - window:t + window + 1]
        if np.any(segment <= 0) or np.any(np.isnan(segment)):
            continue
        # Log-linear regression to estimate instantaneous growth rate r
        log_inc = np.log(segment + 1)
        time_idx = np.arange(len(segment)) - window
        slope, intercept = np.polyfit(time_idx, log_inc, 1)
        r = slope

        # Wallinga-Lipsitch Rt
        rt = (1 + r * theta) ** k
        # Approximate CI via bootstrap over segment
        boot_rt = []
        rng = np.random.default_rng(t)
        for _ in range(200):
            noise = rng.normal(0, 0.05 * np.abs(r) + 1e-6, len(segment))
            log_b = log_inc + noise
            b_slope, _ = np.polyfit(time_idx, log_b, 1)
            boot_rt.append((1 + b_slope * theta) ** k)
        rt_lower[t] = max(0, np.quantile(boot_rt, quantiles[0]))
        rt_upper[t] = np.quantile(boot_rt, quantiles[1])
        rt_vals[t]  = rt
        r_vals[t]   = r

    df = pd.DataFrame({
        "Rt": rt_vals,
        "Rt_lower": rt_lower,
        "Rt_upper": rt_upper,
        "r": r_vals,
        "valid": ~np.isnan(rt_vals),
    })
    return df


# ---------------------------------------------------------------------------
# 4. Epidemic Curve Plot
# ---------------------------------------------------------------------------

def plot_epidemic_curve(
    dates: Union[pd.DatetimeIndex, np.ndarray],
    cases: np.ndarray,
    model_output: Optional[np.ndarray] = None,
    rt_df: Optional[pd.DataFrame] = None,
    title: str = "Epidemic Curve",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot observed incidence bars with optional model fit line and Rt panel.
    """
    n_panels = 2 if rt_df is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5 * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.bar(dates, cases, color="steelblue", alpha=0.5, label="Observed cases", width=1)
    if model_output is not None:
        ax.plot(dates, model_output, color="firebrick", linewidth=2,
                label="Model fit")
    ax.set_ylabel("Daily new cases")
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    if rt_df is not None:
        ax2 = axes[1]
        valid = rt_df["valid"]
        ax2.plot(np.array(dates)[valid], rt_df.loc[valid, "Rt"],
                 color="darkorange", linewidth=2, label="Rt")
        ax2.fill_between(
            np.array(dates)[valid],
            rt_df.loc[valid, "Rt_lower"],
            rt_df.loc[valid, "Rt_upper"],
            color="darkorange", alpha=0.2, label="95% CI",
        )
        ax2.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Rt")
        ax2.set_ylim(0, max(4.0, rt_df["Rt_upper"].quantile(0.99) * 1.1))
        ax2.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# 5. Age-Structured SEIR
# ---------------------------------------------------------------------------

def age_structured_model(
    contact_matrix: np.ndarray,
    beta: float,
    sigma: float,
    gamma: float,
    N_by_age: np.ndarray,
) -> Callable:
    """
    Build an age-structured SEIR ODE system.

    Parameters
    ----------
    contact_matrix : np.ndarray
        (n_age x n_age) POLYMOD-style contact matrix. Entry C[i,j] is the
        average number of contacts individuals in age group i make with
        age group j per day.
    beta : float
        Per-contact transmission probability.
    sigma : float
        Incubation rate (scalar, shared across age groups).
    gamma : float
        Recovery rate (scalar, shared across age groups).
    N_by_age : np.ndarray
        Population size in each age group (length n_age).

    Returns
    -------
    Callable
        ODE function f(t, y) with y flattened as [S0,S1,...,E0,E1,...,I0,I1,...,R0,R1,...].
    """
    n = len(N_by_age)
    N_total = N_by_age.sum()

    def ode(t, y):
        y = np.reshape(y, (4, n))
        S, E, I, R = y

        # Force of infection for each age group
        foi = beta * (contact_matrix @ (I / N_by_age))

        dS = -foi * S
        dE =  foi * S - sigma * E
        dI =  sigma * E - gamma * I
        dR =  gamma * I
        return np.concatenate([dS, dE, dI, dR])

    return ode
```

---

## Example 1: Fit SEIR to COVID-19 Data and Estimate Rt

```python
"""
example_covid_seir_rt.py
-------------------------
Fit a SEIR model to a COVID-19-like incidence curve and estimate Rt
using the sliding-window exponential growth method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from epi_modeling import (
    build_seir_model,
    fit_model_to_data,
    estimate_rt,
    plot_epidemic_curve,
)

# ---- 1. Generate synthetic "observed" COVID-like incidence ----
# In production, replace this with real data from:
#   WHO: https://covid19.who.int/data
#   CDC: https://data.cdc.gov
rng = np.random.default_rng(7)
N_pop = 5_000_000

# True parameters (to be recovered by fitting)
BETA_TRUE  = 0.28
SIGMA_TRUE = 1 / 5.1    # mean incubation 5.1 days
GAMMA_TRUE = 1 / 10.0   # mean infectious period 10 days

from scipy.integrate import solve_ivp
true_ode = build_seir_model(BETA_TRUE, SIGMA_TRUE, GAMMA_TRUE, N_pop)
T_days = 120
sol_true = solve_ivp(
    true_ode, [0, T_days - 1],
    [N_pop - 20, 10, 10, 0],
    t_eval=np.arange(T_days), method="Radau",
)
S_true, E_true, I_true, R_true = sol_true.y
new_infections_true = np.maximum(np.diff(SIGMA_TRUE * E_true, prepend=0), 0)
observed = rng.poisson(np.clip(new_infections_true, 0, None)).astype(float)

start_date = pd.Timestamp("2023-01-01")
dates = pd.date_range(start_date, periods=T_days, freq="D")

# ---- 2. Fit SEIR model ----
print("Fitting SEIR model (this may take ~30-60 seconds) ...")
fit_result = fit_model_to_data(
    incidence=observed,
    dates=dates,
    model_func=build_seir_model,
    N=N_pop,
    param_bounds={
        "beta":  (0.05, 1.0),
        "sigma": (1/14, 1/2),
        "gamma": (1/21, 1/4),
    },
    compartment_idx=2,
    initial_exposed=10,
    method="differential_evolution",
)

fp = fit_result["params"]
print(f"\n--- Fitted Parameters ---")
print(f"  beta  : {fp['beta']:.4f}  (true: {BETA_TRUE:.4f})")
print(f"  sigma : {fp['sigma']:.4f}  (true: {SIGMA_TRUE:.4f})")
print(f"  gamma : {fp['gamma']:.4f}  (true: {GAMMA_TRUE:.4f})")
print(f"  R0    : {fit_result['R0']:.2f}  (true: {BETA_TRUE/GAMMA_TRUE:.2f})")
print(f"  RMSE  : {fit_result['rmse']:.1f} cases/day")

# ---- 3. Estimate Rt ----
rt_df = estimate_rt(
    incidence=observed,
    si_mean=7.5,    # mean serial interval (days)
    si_sd=3.4,
    window=7,
)
rt_df.index = range(len(rt_df))

# ---- 4. Plot epidemic curve + Rt ----
fig = plot_epidemic_curve(
    dates=dates,
    cases=observed,
    model_output=fit_result["fitted_incidence"],
    rt_df=rt_df,
    title="COVID-19-like Outbreak: SEIR Fit and Rt Estimation",
    output_path="seir_covid_fit.png",
)
plt.show()

# ---- 5. Peak incidence statistics ----
peak_day = np.argmax(fit_result["fitted_incidence"])
print(f"\nPeak incidence (model): {fit_result['fitted_incidence'][peak_day]:.0f} cases "
      f"on {dates[peak_day].strftime('%Y-%m-%d')} (day {peak_day})")

valid_rt = rt_df[rt_df["valid"]]
rt_above_1 = (valid_rt["Rt"] > 1).sum()
print(f"Days with Rt > 1: {rt_above_1} / {len(valid_rt)}")
```

---

## Example 2: Compare SIR vs SEIR vs SEIHR

```python
"""
example_model_comparison.py
----------------------------
Run SIR, SEIR, and SEIHR models with identical R0 and compare
epidemic dynamics: peak timing, peak magnitude, hospitalisation burden.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from epi_modeling import build_sir_model, build_seir_model, build_seihr_model

N_pop    = 1_000_000
R0       = 2.5
gamma    = 1 / 10.0      # recovery rate
beta     = R0 * gamma    # transmission rate
sigma    = 1 / 5.2       # incubation rate (SEIR / SEIHR only)
gamma_h  = 1 / 7.0       # hospital discharge rate (SEIHR only)
hosp_rate = 0.03         # 3% of infectious cases hospitalised (SEIHR only)

T_days = 200
t_span = [0, T_days - 1]
t_eval = np.arange(T_days)

# ---- SIR ----
sir_ode  = build_sir_model(beta, gamma, N_pop)
sol_sir  = solve_ivp(sir_ode,  t_span, [N_pop - 1, 1, 0],      t_eval=t_eval, method="Radau")

# ---- SEIR ----
seir_ode = build_seir_model(beta, sigma, gamma, N_pop)
sol_seir = solve_ivp(seir_ode, t_span, [N_pop - 11, 10, 1, 0], t_eval=t_eval, method="Radau")

# ---- SEIHR ----
seihr_ode = build_seihr_model(beta, sigma, gamma, gamma_h, hosp_rate, N_pop)
sol_seihr = solve_ivp(seihr_ode, t_span, [N_pop - 11, 10, 1, 0, 0], t_eval=t_eval, method="Radau")

# ---- Plot ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Infectious curves
ax = axes[0, 0]
ax.plot(t_eval, sol_sir.y[1]  / N_pop * 100, label="SIR  (I)",   color="steelblue")
ax.plot(t_eval, sol_seir.y[2] / N_pop * 100, label="SEIR (I)",   color="darkorange")
ax.plot(t_eval, sol_seihr.y[2]/ N_pop * 100, label="SEIHR (I)",  color="firebrick")
ax.set_xlabel("Day"); ax.set_ylabel("% of population")
ax.set_title("Infectious Prevalence")
ax.legend()

# Hospitalised (SEIHR only)
ax = axes[0, 1]
ax.plot(t_eval, sol_seihr.y[3] / N_pop * 100, color="purple", linewidth=2)
ax.set_xlabel("Day"); ax.set_ylabel("% of population")
ax.set_title("Hospitalised Prevalence (SEIHR)")
ax.fill_between(t_eval, 0, sol_seihr.y[3] / N_pop * 100, color="purple", alpha=0.2)

# Susceptible depletion
ax = axes[1, 0]
ax.plot(t_eval, sol_sir.y[0]  / N_pop * 100, label="SIR",   color="steelblue")
ax.plot(t_eval, sol_seir.y[0] / N_pop * 100, label="SEIR",  color="darkorange")
ax.plot(t_eval, sol_seihr.y[0]/ N_pop * 100, label="SEIHR", color="firebrick")
ax.set_xlabel("Day"); ax.set_ylabel("% of population")
ax.set_title("Susceptible Depletion")
ax.legend()

# Summary bar chart
ax = axes[1, 1]
models  = ["SIR", "SEIR", "SEIHR"]
peak_I  = [
    sol_sir.y[1].max() / N_pop * 100,
    sol_seir.y[2].max() / N_pop * 100,
    sol_seihr.y[2].max() / N_pop * 100,
]
peak_day = [
    sol_sir.y[1].argmax(),
    sol_seir.y[2].argmax(),
    sol_seihr.y[2].argmax(),
]
final_R = [
    sol_sir.y[2][-1] / N_pop * 100,
    sol_seir.y[3][-1] / N_pop * 100,
    sol_seihr.y[4][-1] / N_pop * 100,
]
x = np.arange(len(models))
width = 0.3
ax.bar(x - width, peak_I,  width, label="Peak I (%)", color="steelblue")
ax.bar(x,         final_R, width, label="Final R (%)", color="darkorange")
ax.bar(x + width, peak_day, width, label="Peak day",   color="firebrick")
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_title("Model Comparison Summary")
ax.legend()

fig.suptitle(f"Epidemic Model Comparison — R0 = {R0}", fontsize=14)
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: model_comparison.png")
plt.show()

# ---- Print table ----
print("\n{:<8} {:>12} {:>12} {:>12}".format("Model", "Peak I (%)", "Peak day", "Final R (%)"))
print("-" * 46)
for name, pi, pd_, fr in zip(models, peak_I, peak_day, final_R):
    print(f"{name:<8} {pi:>12.2f} {pd_:>12} {fr:>12.2f}")
```

---

## PRCC Sensitivity Analysis

```python
"""
sensitivity_analysis.py
-----------------------
Partial Rank Correlation Coefficient (PRCC) global sensitivity analysis
for epidemic model outputs (e.g., peak prevalence, final attack rate).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from scipy.integrate import solve_ivp
from epi_modeling import build_seir_model


def prcc_sensitivity(
    param_distributions: dict,
    model_func: callable,
    output_func: callable,
    N: int,
    n_samples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute PRCC indices for each parameter against a scalar model output.

    Parameters
    ----------
    param_distributions : dict
        {param_name: (low, high)} uniform distribution bounds.
    model_func : callable
        Function(params_dict, N) -> solve_ivp solution.
    output_func : callable
        Function(solution) -> scalar output of interest.
    N : int
        Population size.
    n_samples : int
        LHS sample size.
    seed : int

    Returns
    -------
    pd.DataFrame with columns: param, prcc, p_value.
    """
    rng = np.random.default_rng(seed)
    param_names = list(param_distributions.keys())
    k = len(param_names)

    # Latin Hypercube Sampling
    lhs = np.zeros((n_samples, k))
    for j, name in enumerate(param_names):
        lo, hi = param_distributions[name]
        perm = rng.permutation(n_samples)
        u = (perm + rng.uniform(size=n_samples)) / n_samples
        lhs[:, j] = lo + u * (hi - lo)

    # Evaluate model
    outputs = np.full(n_samples, np.nan)
    for i in range(n_samples):
        params = dict(zip(param_names, lhs[i]))
        try:
            sol = model_func(params, N)
            outputs[i] = output_func(sol)
        except Exception:
            pass

    valid = ~np.isnan(outputs)
    lhs_v = lhs[valid]
    out_v = outputs[valid]

    # Rank-transform
    ranked = np.column_stack([rankdata(lhs_v[:, j]) for j in range(k)] +
                             [rankdata(out_v)])

    # Partial correlations via residuals
    prcc_vals, pvals = [], []
    for j in range(k):
        other = [jj for jj in range(k) if jj != j]
        X = ranked[:, j]
        Y = ranked[:, k]
        Z = ranked[:, other]

        # Regress X on Z
        coef_x = np.linalg.lstsq(np.column_stack([np.ones(len(X)), Z]), X, rcond=None)[0]
        res_x = X - np.column_stack([np.ones(len(X)), Z]) @ coef_x

        # Regress Y on Z
        coef_y = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), Z]), Y, rcond=None)[0]
        res_y = Y - np.column_stack([np.ones(len(Y)), Z]) @ coef_y

        r, p = spearmanr(res_x, res_y)
        prcc_vals.append(r)
        pvals.append(p)

    df = pd.DataFrame({
        "param": param_names,
        "prcc": prcc_vals,
        "p_value": pvals,
    }).sort_values("prcc", key=abs, ascending=False)
    return df


if __name__ == "__main__":
    N = 1_000_000

    def run_seir(params, N):
        ode = build_seir_model(params["beta"], params["sigma"], params["gamma"], N)
        return solve_ivp(ode, [0, 200], [N - 11, 10, 1, 0],
                         t_eval=np.arange(201), method="Radau")

    def peak_prevalence(sol):
        return sol.y[2].max() / N * 100

    results = prcc_sensitivity(
        param_distributions={
            "beta":  (0.1, 0.6),
            "sigma": (1/14, 1/3),
            "gamma": (1/21, 1/3),
        },
        model_func=run_seir,
        output_func=peak_prevalence,
        N=N,
        n_samples=500,
    )

    print("\nPRCC Sensitivity Analysis — Peak Infectious Prevalence")
    print(results.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["firebrick" if v > 0 else "steelblue" for v in results["prcc"]]
    ax.barh(results["param"], results["prcc"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PRCC")
    ax.set_title("Sensitivity Analysis: Peak Infectious Prevalence")
    fig.tight_layout()
    fig.savefig("prcc_sensitivity.png", dpi=150)
    print("Saved: prcc_sensitivity.png")
    plt.show()
```

---

## Tips and Best Practices

- **Solver choice**: Use `method="Radau"` for stiff systems (large populations, wide R0 ranges).
  `LSODA` is also stiff-capable. `RK45` is fast but may fail for stiff problems.
- **Initial conditions**: Seed with a small number of exposed individuals (E0 > 0) to allow
  the latent period to seed infectious cases naturally; avoids discontinuities.
- **Identifiability**: SEIR with only I(t) observed is structurally identifiable only when
  `sigma` is fixed from literature. Fitting all three parameters simultaneously risks
  converging to incorrect minima.
- **Data smoothing**: Apply a 7-day rolling average to raw case counts before Rt estimation
  to suppress weekly reporting artefacts.
- **Serial interval vs generation time**: The serial interval (time between symptom onsets)
  is observable; the generation time (time between infections) is not. For Rt estimation,
  use the serial interval.
- **Age-structured models**: Use POLYMOD contact matrices (Mossong et al., 2008) or
  country-specific matrices from the `socialmixr` R package (exportable to CSV).

---

## References

- Kermack & McKendrick (1927). A contribution to the mathematical theory of epidemics.
  *Proc. R. Soc. A*, 115, 700–721.
- Wallinga & Lipsitch (2007). How generation intervals shape the relationship between
  growth rates and reproductive numbers. *Proc. R. Soc. B*, 274, 599–604.
- Cori et al. (2013). A new framework and software to estimate time-varying reproduction
  numbers during epidemics. *Am. J. Epidemiol.*, 178, 1505–1512.
- Marino et al. (2008). A methodology for performing global uncertainty and sensitivity
  analysis in systems biology. *J. Theor. Biol.*, 254, 178–196.
