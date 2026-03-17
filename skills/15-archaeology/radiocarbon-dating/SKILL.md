---
name: radiocarbon-dating
description: >
  Radiocarbon (14C) dating calibration and Bayesian age modeling using IntCal20/SHCal20/Marine20
  curves, sequence modeling with stratigraphic constraints, and uncertainty reporting.
tags:
  - archaeology
  - radiocarbon
  - bayesian
  - geochronology
  - calibration
  - dating
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
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - pandas>=2.0.0
  - requests>=2.28.0
last_updated: "2026-03-17"
---

# Radiocarbon Dating Analysis

Radiocarbon (14C) dating is the most widely used chronometric technique in archaeology,
providing calendar age estimates for organic materials up to ~50,000 years old. This skill
covers calibration against internationally agreed reference curves, Bayesian sequence
modeling, wiggle matching, and proper reporting conventions.

## Background

Raw radiocarbon measurements are reported as "conventional radiocarbon ages" in years BP
(Before Present, where Present = 1950 CE). Because atmospheric 14C concentration has varied
over time, a calibration step is required to convert these measurements into calendar ages.
The IntCal20, SHCal20, and Marine20 calibration curves (Reimer et al. 2020) provide the
relationship between radiocarbon years and calendar years.

### Key Concepts

- **BP (Before Present)**: Radiocarbon years before 1950 CE
- **cal BP**: Calibrated (calendar) years before 1950 CE
- **BCE/CE**: Calendar years; BCE = cal BP + 1950 (for dates before 1950)
- **Delta-14C (Δ14C)**: Per-mil deviation of 14C/12C ratio from a standard
- **F14C**: Fraction modern — ratio of sample 14C to modern standard 14C
- **Reservoir effect**: Marine/freshwater organisms appear older due to depleted 14C reservoirs
- **Marine reservoir correction (ΔR)**: Local deviation from global marine average

## Installation

```bash
pip install numpy scipy matplotlib pandas requests
# Optional: download calibration curves locally for offline use
python -c "import urllib.request; urllib.request.urlretrieve(
    'https://intcal.org/curves/intcal20.14c',
    'intcal20.14c'
)"
```

## Loading Calibration Curves

```python
import numpy as np
import pandas as pd
import requests
from io import StringIO
from pathlib import Path


def load_calibration_curve(curve_name: str = "intcal20") -> pd.DataFrame:
    """
    Load a calibration curve from file or download from intcal.org.

    Parameters
    ----------
    curve_name : str
        One of 'intcal20', 'shcal20', 'marine20'

    Returns
    -------
    pd.DataFrame
        Columns: cal_BP, c14_age, c14_error, delta14C, delta14C_error
    """
    curve_urls = {
        "intcal20": "https://intcal.org/curves/intcal20.14c",
        "shcal20":  "https://intcal.org/curves/shcal20.14c",
        "marine20": "https://intcal.org/curves/marine20.14c",
    }
    local_paths = {
        "intcal20": Path("intcal20.14c"),
        "shcal20":  Path("shcal20.14c"),
        "marine20": Path("marine20.14c"),
    }

    curve_name = curve_name.lower()
    if curve_name not in curve_urls:
        raise ValueError(f"Unknown curve: {curve_name}. Choose from {list(curve_urls)}")

    local = local_paths[curve_name]
    if local.exists():
        raw = local.read_text()
    else:
        print(f"Downloading {curve_name} from intcal.org ...")
        response = requests.get(curve_urls[curve_name], timeout=30)
        response.raise_for_status()
        raw = response.text
        local.write_text(raw)

    # Skip comment lines starting with '#'
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("#") and ln.strip()]
    text = "\n".join(lines)
    df = pd.read_csv(
        StringIO(text),
        header=None,
        names=["cal_BP", "c14_age", "c14_error", "delta14C", "delta14C_error"],
        sep=r"\s*,\s*",
        engine="python",
    )
    df = df.sort_values("cal_BP").reset_index(drop=True)
    return df


def _interpolate_curve(curve_df: pd.DataFrame, cal_years: np.ndarray):
    """Interpolate curve c14_age and c14_error at given cal BP values."""
    mu = np.interp(cal_years, curve_df["cal_BP"].values, curve_df["c14_age"].values)
    sigma = np.interp(cal_years, curve_df["cal_BP"].values, curve_df["c14_error"].values)
    return mu, sigma
```

## Single-Date Calibration

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm


def calibrate_date(
    c14_age: float,
    c14_error: float,
    curve_df: pd.DataFrame,
    cal_range: tuple[float, float] = (0, 55000),
    resolution: float = 1.0,
) -> pd.DataFrame:
    """
    Calibrate a single radiocarbon date against a calibration curve.

    Parameters
    ----------
    c14_age : float
        Conventional radiocarbon age in BP.
    c14_error : float
        1-sigma laboratory measurement error.
    curve_df : pd.DataFrame
        Calibration curve loaded by load_calibration_curve().
    cal_range : tuple
        (min_cal_BP, max_cal_BP) search window.
    resolution : float
        Step size in calendar years (default 1 year).

    Returns
    -------
    pd.DataFrame
        Columns: cal_BP, probability (unnormalized then normalized), cal_CE.
    """
    cal_years = np.arange(cal_range[0], cal_range[1] + resolution, resolution)
    curve_mu, curve_sigma = _interpolate_curve(curve_df, cal_years)

    # Combined variance: measurement error + calibration curve error
    combined_sigma = np.sqrt(c14_error**2 + curve_sigma**2)

    # Likelihood: N(c14_age | curve_mu, combined_sigma)
    prob = norm.pdf(c14_age, loc=curve_mu, scale=combined_sigma)
    prob /= prob.sum()  # normalize to sum to 1

    cal_CE = 1950 - cal_years  # negative = BCE

    return pd.DataFrame({
        "cal_BP": cal_years,
        "cal_CE": cal_CE,
        "probability": prob,
    })


def hpd_intervals(cal_dist: pd.DataFrame, confidence: float = 0.95) -> list[tuple]:
    """
    Compute Highest Posterior Density (HPD) intervals.

    Parameters
    ----------
    cal_dist : pd.DataFrame
        Output of calibrate_date().
    confidence : float
        Credible mass (e.g. 0.68 or 0.95).

    Returns
    -------
    list of (start_cal_BP, end_cal_BP, probability_mass) tuples
    """
    sorted_idx = np.argsort(cal_dist["probability"].values)[::-1]
    cum_prob = np.cumsum(cal_dist["probability"].values[sorted_idx])
    in_hpd = np.zeros(len(cal_dist), dtype=bool)
    in_hpd[sorted_idx[cum_prob <= confidence]] = True
    # Also include the first point that pushes past the threshold
    if cum_prob[np.sum(cum_prob <= confidence)] <= confidence + 1e-9:
        idx = np.sum(cum_prob <= confidence)
        if idx < len(sorted_idx):
            in_hpd[sorted_idx[idx]] = True

    cal_years = cal_dist["cal_BP"].values
    intervals = []
    in_block = False
    start = None
    block_prob = 0.0
    for i, (yr, flag, p) in enumerate(
        zip(cal_years, in_hpd, cal_dist["probability"].values)
    ):
        if flag and not in_block:
            in_block = True
            start = yr
            block_prob = 0.0
        if flag:
            block_prob += p
        if not flag and in_block:
            intervals.append((start, cal_years[i - 1], block_prob))
            in_block = False
    if in_block:
        intervals.append((start, cal_years[-1], block_prob))
    return intervals


def summary_statistics(cal_dist: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a calibrated date distribution.

    Returns dict with keys: mean_cal_BP, median_cal_BP, mode_cal_BP,
    sigma_cal_BP, hpd68, hpd95 (each as list of interval tuples).
    """
    probs = cal_dist["probability"].values
    years = cal_dist["cal_BP"].values

    mean_bp = np.sum(years * probs)
    variance = np.sum(probs * (years - mean_bp) ** 2)
    sigma = np.sqrt(variance)

    cdf = np.cumsum(probs)
    median_bp = years[np.searchsorted(cdf, 0.5)]
    mode_bp = years[np.argmax(probs)]

    return {
        "mean_cal_BP": float(mean_bp),
        "median_cal_BP": float(median_bp),
        "mode_cal_BP": float(mode_bp),
        "sigma_cal_BP": float(sigma),
        "mean_cal_CE": float(1950 - mean_bp),
        "hpd68": hpd_intervals(cal_dist, 0.68),
        "hpd95": hpd_intervals(cal_dist, 0.95),
    }


def plot_calibrated_date(
    cal_dist: pd.DataFrame,
    title: str = "Calibrated Radiocarbon Date",
    c14_age: float = None,
    c14_error: float = None,
    curve_df: pd.DataFrame = None,
    show_hpd: bool = True,
    output_path: str = None,
) -> None:
    """
    Plot a calibrated radiocarbon date with optional calibration curve overlay.

    Parameters
    ----------
    cal_dist : pd.DataFrame
        Output of calibrate_date().
    title : str
        Plot title.
    c14_age, c14_error : float, optional
        Original measurement for Gaussian overlay on 14C axis.
    curve_df : pd.DataFrame, optional
        Calibration curve to display.
    show_hpd : bool
        Shade 95% HPD intervals.
    output_path : str, optional
        Save figure to this path if provided.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                           hspace=0.05, wspace=0.05)

    ax_curve = fig.add_subplot(gs[0, 0])  # top: 14C curve
    ax_cal   = fig.add_subplot(gs[1, 0])  # bottom-left: calendar dist
    ax_c14   = fig.add_subplot(gs[1, 1])  # bottom-right: 14C Gaussian

    cal_years = cal_dist["cal_BP"].values
    probs = cal_dist["probability"].values

    # --- Calibration curve panel ---
    if curve_df is not None:
        mask = (curve_df["cal_BP"] >= cal_years.min()) & \
               (curve_df["cal_BP"] <= cal_years.max())
        sub = curve_df[mask]
        ax_curve.plot(sub["cal_BP"], sub["c14_age"], "k-", lw=1.5, label="IntCal20")
        ax_curve.fill_between(
            sub["cal_BP"],
            sub["c14_age"] - 2 * sub["c14_error"],
            sub["c14_age"] + 2 * sub["c14_error"],
            alpha=0.3, color="gray",
        )
        if c14_age is not None:
            ax_curve.axhline(c14_age, color="red", lw=1, ls="--")
            ax_curve.axhspan(c14_age - c14_error, c14_age + c14_error,
                             color="red", alpha=0.15)
    ax_curve.set_ylabel("14C Age (BP)")
    ax_curve.xaxis.set_visible(False)
    ax_curve.invert_xaxis()

    # --- Calendar probability panel ---
    ax_cal.fill_between(cal_years, probs, alpha=0.5, color="steelblue")
    ax_cal.plot(cal_years, probs, color="steelblue", lw=1)
    if show_hpd:
        intervals_95 = hpd_intervals(cal_dist, 0.95)
        for start, end, mass in intervals_95:
            mask = (cal_years >= start) & (cal_years <= end)
            ax_cal.fill_between(cal_years[mask], probs[mask],
                                 alpha=0.6, color="steelblue")
    ax_cal.set_xlabel("Calibrated Age (cal BP)")
    ax_cal.set_ylabel("Probability Density")
    ax_cal.invert_xaxis()

    # --- 14C Gaussian panel ---
    if c14_age is not None and c14_error is not None:
        y_range = np.linspace(c14_age - 4 * c14_error, c14_age + 4 * c14_error, 300)
        gauss = norm.pdf(y_range, c14_age, c14_error)
        ax_c14.plot(gauss, y_range, color="red", lw=1.5)
        ax_c14.fill_betweenx(y_range, gauss, alpha=0.3, color="red")
    ax_c14.yaxis.set_visible(False)
    ax_c14.set_xlabel("Probability")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
```

## Bayesian Sequence Modeling

```python
from scipy.optimize import minimize
from typing import Optional


def sequence_model(
    dates: list[dict],
    constraints: list[tuple[int, int]] = None,
    n_iterations: int = 50000,
    burn_in: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """
    Bayesian sequence model using MCMC for stratigraphically ordered dates.

    Each date dict must have keys:
        label: str
        c14_age: float (BP)
        c14_error: float
        curve_df: pd.DataFrame  (calibration curve)

    constraints: list of (i, j) meaning date[i] is OLDER than date[j]
                 (i.e., date[i].cal_BP > date[j].cal_BP)

    Returns list of dicts with posterior samples and statistics.

    Parameters
    ----------
    dates : list of dict
        Each entry: {label, c14_age, c14_error, curve_df}
    constraints : list of (int, int)
        Ordering constraints: (older_index, younger_index)
    n_iterations : int
        Total MCMC iterations.
    burn_in : int
        Discarded warm-up iterations.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = len(dates)

    # Initialize theta (cal BP values) from unconstrained calibrations
    theta = np.zeros(n)
    cal_dists = []
    for i, d in enumerate(dates):
        cd = calibrate_date(d["c14_age"], d["c14_error"], d["curve_df"])
        cal_dists.append(cd)
        theta[i] = float(summary_statistics(cd)["median_cal_BP"])

    if constraints is None:
        constraints = []

    def log_likelihood(th):
        ll = 0.0
        for i, d in enumerate(dates):
            cd = cal_dists[i]
            # Interpolate probability at this theta value
            prob = np.interp(th[i], cd["cal_BP"].values, cd["probability"].values,
                             left=0.0, right=0.0)
            if prob <= 0:
                return -np.inf
            ll += np.log(prob)
        return ll

    def satisfies_constraints(th):
        for (older, younger) in constraints:
            if th[older] <= th[younger]:  # older must have larger cal_BP
                return False
        return True

    # Metropolis-Hastings MCMC
    proposal_sigma = 50.0  # years
    current_ll = log_likelihood(theta)
    samples = np.zeros((n_iterations - burn_in, n))
    n_accepted = 0

    for it in range(n_iterations):
        # Propose new theta by perturbing one date at a time
        idx = rng.integers(0, n)
        theta_prop = theta.copy()
        theta_prop[idx] += rng.normal(0, proposal_sigma)

        if not satisfies_constraints(theta_prop):
            continue

        prop_ll = log_likelihood(theta_prop)
        log_alpha = prop_ll - current_ll

        if np.log(rng.uniform()) < log_alpha:
            theta = theta_prop
            current_ll = prop_ll
            n_accepted += 1

        if it >= burn_in:
            samples[it - burn_in] = theta

    acceptance_rate = n_accepted / n_iterations
    print(f"MCMC acceptance rate: {acceptance_rate:.3f}")

    results = []
    for i, d in enumerate(dates):
        samp = samples[:, i]
        results.append({
            "label": d["label"],
            "c14_age": d["c14_age"],
            "c14_error": d["c14_error"],
            "posterior_samples": samp,
            "mean_cal_BP": float(np.mean(samp)),
            "median_cal_BP": float(np.median(samp)),
            "std_cal_BP": float(np.std(samp)),
            "hpd95_lower": float(np.percentile(samp, 2.5)),
            "hpd95_upper": float(np.percentile(samp, 97.5)),
            "mean_cal_CE": float(1950 - np.mean(samp)),
        })
    return results


def wiggle_match(
    radiocarbon_ages: list[float],
    errors: list[float],
    time_gaps: list[float],
    curve_df: pd.DataFrame,
    search_range: tuple = (0, 10000),
) -> dict:
    """
    Wiggle matching: fit a floating tree-ring sequence to the calibration curve.

    Parameters
    ----------
    radiocarbon_ages : list of float
        14C ages (BP) for each ring, outermost first.
    errors : list of float
        Measurement errors for each ring.
    time_gaps : list of float
        Known gaps in years between rings (len = len(radiocarbon_ages) - 1).
    curve_df : pd.DataFrame
        Calibration curve.
    search_range : tuple
        (min_cal_BP, max_cal_BP) for the outermost ring.

    Returns
    -------
    dict with best_fit_cal_BP (outermost ring), chi2, and probability array.
    """
    n_rings = len(radiocarbon_ages)
    candidate_years = np.arange(search_range[0], search_range[1])
    chi2_values = np.zeros(len(candidate_years))

    for idx, outer_cal_BP in enumerate(candidate_years):
        chi2 = 0.0
        for ring_i in range(n_rings):
            gap = sum(time_gaps[:ring_i]) if ring_i > 0 else 0
            cal_bp_ring = outer_cal_BP + gap
            curve_mu, curve_sigma = _interpolate_curve(curve_df, np.array([cal_bp_ring]))
            combined_var = errors[ring_i]**2 + curve_sigma[0]**2
            chi2 += (radiocarbon_ages[ring_i] - curve_mu[0])**2 / combined_var
        chi2_values[idx] = chi2

    prob = np.exp(-0.5 * chi2_values)
    prob /= prob.sum()
    best_idx = np.argmin(chi2_values)

    return {
        "best_fit_cal_BP": float(candidate_years[best_idx]),
        "best_fit_cal_CE": float(1950 - candidate_years[best_idx]),
        "chi2_min": float(chi2_values[best_idx]),
        "candidate_years": candidate_years,
        "probability": prob,
    }
```

## Example 1: Calibrate a Single Radiocarbon Date with IntCal20

```python
# Example 1: Single date calibration
# A charcoal sample from a hearth, measured at 2850 ± 35 BP

def example_single_date():
    print("=== Example 1: Single Date Calibration ===\n")

    # Load calibration curve (will download on first run)
    curve = load_calibration_curve("intcal20")
    print(f"Loaded IntCal20: {len(curve)} data points")

    # Calibrate the date
    c14_age = 2850
    c14_error = 35
    cal_dist = calibrate_date(c14_age, c14_error, curve_df=curve)

    # Summary statistics
    stats = summary_statistics(cal_dist)
    print(f"\nRadiocarbon age: {c14_age} ± {c14_error} BP")
    print(f"Calibrated mean:   {stats['mean_cal_BP']:.0f} cal BP  "
          f"({stats['mean_cal_CE']:.0f} CE/BCE)")
    print(f"Calibrated median: {stats['median_cal_BP']:.0f} cal BP")
    print(f"Sigma (1σ):        ± {stats['sigma_cal_BP']:.0f} years")

    print("\n68.3% HPD intervals (cal BP):")
    for start, end, mass in stats["hpd68"]:
        ce_start = 1950 - end
        ce_end   = 1950 - start
        era_s = "BCE" if ce_start < 0 else "CE"
        era_e = "BCE" if ce_end < 0 else "CE"
        print(f"  {start:.0f} – {end:.0f} cal BP  "
              f"({abs(ce_start):.0f} {era_s} – {abs(ce_end):.0f} {era_e})  "
              f"[{mass*100:.1f}%]")

    print("\n95.4% HPD intervals (cal BP):")
    for start, end, mass in stats["hpd95"]:
        ce_start = 1950 - end
        ce_end   = 1950 - start
        era_s = "BCE" if ce_start < 0 else "CE"
        era_e = "BCE" if ce_end < 0 else "CE"
        print(f"  {start:.0f} – {end:.0f} cal BP  "
              f"({abs(ce_start):.0f} {era_s} – {abs(ce_end):.0f} {era_e})  "
              f"[{mass*100:.1f}%]")

    # Plot (comment out in headless environments)
    plot_calibrated_date(
        cal_dist,
        title=f"Charcoal from Hearth — {c14_age} ± {c14_error} BP",
        c14_age=c14_age,
        c14_error=c14_error,
        curve_df=curve,
        output_path="calibrated_hearth.png",
    )
    print("\nPlot saved to calibrated_hearth.png")


if __name__ == "__main__":
    example_single_date()
```

## Example 2: Bayesian Sequence Model for a Burial Site

```python
# Example 2: Bayesian sequence model — Bronze Age burial site
# Five radiocarbon dates from successive burial layers (top to bottom = younger to older)

def example_burial_sequence():
    print("=== Example 2: Bayesian Sequence Model — Bronze Age Burial Site ===\n")

    curve = load_calibration_curve("intcal20")

    # Dates from burial layers — outermost (youngest) to innermost (oldest)
    burial_dates = [
        {"label": "Layer 1 (youngest)", "c14_age": 2980, "c14_error": 40,  "curve_df": curve},
        {"label": "Layer 2",            "c14_age": 3060, "c14_error": 35,  "curve_df": curve},
        {"label": "Layer 3",            "c14_age": 3150, "c14_error": 45,  "curve_df": curve},
        {"label": "Layer 4",            "c14_age": 3280, "c14_error": 50,  "curve_df": curve},
        {"label": "Layer 5 (oldest)",   "c14_age": 3390, "c14_error": 40,  "curve_df": curve},
    ]

    # Stratigraphic constraints: each layer must be younger than the one below
    # Index 0 (youngest) < index 1 < index 2 < index 3 < index 4 (oldest)
    # constraint (i, j): date[i] is older (larger cal BP) than date[j]
    constraints = [
        (1, 0),  # Layer 2 older than Layer 1
        (2, 1),  # Layer 3 older than Layer 2
        (3, 2),  # Layer 4 older than Layer 3
        (4, 3),  # Layer 5 older than Layer 4
    ]

    print("Running MCMC sequence model (this may take 10–30 seconds)...")
    results = sequence_model(
        burial_dates,
        constraints=constraints,
        n_iterations=30000,
        burn_in=5000,
        seed=2024,
    )

    print("\nPosterior estimates (sequence-constrained):\n")
    print(f"{'Label':<25} {'Mean (cal BP)':>14} {'95% HPD (cal BP)':>22} {'Mean (CE/BCE)':>14}")
    print("-" * 80)
    for r in results:
        ce = r["mean_cal_CE"]
        era = "BCE" if ce < 0 else "CE"
        print(
            f"{r['label']:<25} "
            f"{r['mean_cal_BP']:>14.0f} "
            f"{r['hpd95_lower']:>10.0f} – {r['hpd95_upper']:<10.0f} "
            f"{abs(ce):>10.0f} {era}"
        )

    # Check that sequence is respected
    print("\nSequence consistency check:")
    for i in range(len(results) - 1):
        older = results[i + 1]["mean_cal_BP"]
        younger = results[i]["mean_cal_BP"]
        ok = "OK" if older > younger else "VIOLATION"
        print(f"  {results[i+1]['label']} > {results[i]['label']}: "
              f"{older:.0f} > {younger:.0f} [{ok}]")

    # Plot posterior distributions
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 10), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    for i, (r, ax) in enumerate(zip(results, axes)):
        samp = r["posterior_samples"]
        ax.hist(samp, bins=80, density=True, color=colors[i], alpha=0.7, edgecolor="none")
        ax.axvline(r["mean_cal_BP"], color="red", lw=1.5, ls="--", label="Mean")
        ax.axvspan(r["hpd95_lower"], r["hpd95_upper"], alpha=0.15, color="red")
        ax.set_ylabel(r["label"][:20], fontsize=8)
        ax.yaxis.set_ticklabels([])
    axes[-1].set_xlabel("Calibrated Age (cal BP)")
    axes[-1].invert_xaxis()
    fig.suptitle("Bayesian Sequence Model — Bronze Age Burial Site", fontweight="bold")
    plt.tight_layout()
    plt.savefig("burial_sequence.png", dpi=150, bbox_inches="tight")
    print("\nSequence plot saved to burial_sequence.png")


if __name__ == "__main__":
    example_burial_sequence()
```

## Reporting Conventions

When reporting calibrated dates in publications, follow OxCal/IntCal conventions:

- Always specify the calibration curve and software version
- Report both 68.3% (1σ) and 95.4% (2σ) HPD intervals
- State the original 14C measurement in BP with ± 1σ laboratory error
- For example: "2850 ± 35 BP (OxCal 4.4, IntCal20); 2σ: 3090–2870 cal BP (1060–920 BCE)"

## Delta-14C and F14C Conversions

```python
def c14_age_to_F14C(c14_age_BP: float) -> float:
    """Convert conventional radiocarbon age (BP) to Fraction Modern (F14C)."""
    # F14C = exp(-c14_age / 8033) using Libby half-life of 5568 years
    # but corrected to Godwin half-life of 5730 years: 8267 * ln(2) = 5730
    return np.exp(-c14_age_BP / 8267.0)


def F14C_to_c14_age(f14c: float) -> float:
    """Convert Fraction Modern back to conventional radiocarbon age (BP)."""
    if f14c <= 0:
        raise ValueError("F14C must be > 0")
    return -8267.0 * np.log(f14c)


def delta14C_to_F14C(delta14C_permil: float) -> float:
    """Convert Δ14C (per mil) to F14C."""
    return 1.0 + delta14C_permil / 1000.0


# Quick test
if __name__ == "__main__":
    age = 3000
    f = c14_age_to_F14C(age)
    recovered = F14C_to_c14_age(f)
    print(f"3000 BP → F14C={f:.4f} → {recovered:.1f} BP (round-trip check)")
```

## References

- Reimer, P.J., et al. (2020). The IntCal20 Northern Hemisphere Radiocarbon Age Calibration
  Curve (0–55 cal kBP). *Radiocarbon*, 62(4), 725–757.
- Hogg, A.G., et al. (2020). SHCal20 Southern Hemisphere Calibration, 0–55,000 Years cal BP.
  *Radiocarbon*, 62(4), 759–778.
- Bronk Ramsey, C. (2009). Bayesian Analysis of Radiocarbon Dates. *Radiocarbon*, 51(1), 337–360.
- Buck, C.E., Kenworthy, J.B., Litton, C.D., & Smith, A.F.M. (1991). Combining archaeological
  and radiocarbon information: a Bayesian approach to calibration. *Antiquity*, 65, 808–821.
