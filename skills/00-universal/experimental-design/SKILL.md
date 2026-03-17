---
name: experimental-design
description: >
  Design rigorous experiments: sample size calculation, randomization strategies,
  pre-registration, and AB test duration estimation.
tags:
  - experimental-design
  - sample-size
  - randomization
  - pre-registration
  - ab-testing
  - power-analysis
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
  python:
    - statsmodels>=0.14.0
    - numpy>=1.24.0
    - pandas>=2.0.0
    - scipy>=1.10.0
    - osfclient>=0.0.5
last_updated: "2026-03-17"
---

# Experimental Design

Plan experiments with correct statistical power, principled randomization, and
transparent pre-registration before data collection begins.

---

## Key Concepts

| Term | Definition |
|---|---|
| α (alpha) | Type I error rate — false positive probability (typically 0.05) |
| β (beta) | Type II error rate — false negative probability (typically 0.20) |
| Power (1−β) | Probability of detecting a true effect (typically 0.80 or 0.90) |
| MDES | Minimum Detectable Effect Size — smallest effect your design can detect |
| ICC | Intra-cluster correlation — needed for cluster-randomised trials |

---

## Setup

```bash
pip install statsmodels numpy pandas scipy osfclient
```

---

## Core Implementation

```python
"""
experimental_design.py
Sample size, randomization, and pre-registration utilities.
"""

import hashlib
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import (
    TTestIndPower,
    TTestPower,
    FTestAnovaPower,
    NormalIndPower,
)
from statsmodels.stats.proportion import proportion_effectsize


# ─────────────────────────────────────────────
# 1. Sample Size Calculation
# ─────────────────────────────────────────────

def calculate_sample_size(
    effect_size: Optional[float] = None,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = "two-sample-t",
    ratio: float = 1.0,
    k_groups: int = 2,
    p1: Optional[float] = None,
    p2: Optional[float] = None,
    two_tailed: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Calculate required sample size for common experimental designs.

    Parameters
    ----------
    effect_size : float, optional
        Standardised effect size (Cohen's d, f, h, w).
        Required unless p1 and p2 are provided.
    alpha : float
        Type I error rate.
    power : float
        Desired statistical power (1 - beta).
    test_type : str
        One of: 'two-sample-t', 'paired-t', 'one-sample-t',
                'anova', 'proportion-z', 'proportion-chi2'
    ratio : float
        n_group2 / n_group1 (for unequal allocation).
    k_groups : int
        Number of groups (for ANOVA only).
    p1, p2 : float, optional
        Event rates for proportion tests (auto-computes Cohen's h).
    two_tailed : bool
        Whether to use two-tailed test.

    Returns
    -------
    dict with n_per_group, total_n, effect_size, alpha, power, test_type
    """
    alternative = "two-sided" if two_tailed else "larger"

    if p1 is not None and p2 is not None:
        effect_size = abs(proportion_effectsize(p1, p2))
        test_type = "proportion-z"

    if effect_size is None:
        raise ValueError("Provide effect_size or both p1 and p2.")

    if test_type == "two-sample-t":
        analysis = TTestIndPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power,
                                  ratio=ratio, alternative=alternative)
    elif test_type == "paired-t":
        analysis = TTestPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power,
                                  alternative=alternative)
        ratio = 1.0
    elif test_type == "one-sample-t":
        analysis = TTestPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power,
                                  alternative=alternative)
        ratio = 1.0
    elif test_type == "anova":
        # effect_size here is Cohen's f
        analysis = FTestAnovaPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power,
                                  k_groups=k_groups)
    elif test_type in ("proportion-z", "proportion-chi2"):
        analysis = NormalIndPower()
        n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power,
                                  ratio=ratio, alternative=alternative)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")

    n_ceil = math.ceil(n)
    total_n = math.ceil(n_ceil * (1 + ratio)) if test_type not in ("paired-t", "one-sample-t") else n_ceil

    result = {
        "n_per_group": n_ceil,
        "total_n": total_n,
        "effect_size": round(effect_size, 4),
        "alpha": alpha,
        "power": power,
        "test_type": test_type,
        "ratio": ratio,
    }

    if verbose:
        print(f"Sample size calculation ({test_type})")
        print(f"  Effect size : {effect_size:.3f}")
        print(f"  Alpha       : {alpha}")
        print(f"  Power       : {power}")
        print(f"  N per group : {n_ceil}")
        print(f"  Total N     : {total_n}")
        if p1 is not None:
            print(f"  Rates       : p1={p1}, p2={p2}")
        print()

    return result


def mdes(
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = "two-sample-t",
) -> float:
    """
    Minimum Detectable Effect Size given a fixed sample size.
    """
    analysis = TTestIndPower() if test_type == "two-sample-t" else TTestPower()
    es = analysis.solve_power(nobs1=n_per_group, alpha=alpha, power=power,
                               ratio=1.0, alternative="two-sided")
    return round(es, 4)


def inflate_for_attrition(n: int, expected_attrition_rate: float = 0.15) -> int:
    """
    Inflate sample size to account for expected dropout / missing data.

    Parameters
    ----------
    n : int
        Required complete-case sample size.
    expected_attrition_rate : float
        Fraction of participants expected to drop out (0–1).
    """
    if not 0 <= expected_attrition_rate < 1:
        raise ValueError("attrition_rate must be in [0, 1).")
    return math.ceil(n / (1 - expected_attrition_rate))


# ─────────────────────────────────────────────
# 2. Randomization Strategies
# ─────────────────────────────────────────────

def simple_randomize(
    n: int,
    n_groups: int = 2,
    group_labels: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simple (unrestricted) randomization."""
    rng = np.random.default_rng(seed)
    if group_labels is None:
        group_labels = [f"Group_{i}" for i in range(n_groups)]
    assignments = rng.choice(group_labels, size=n)
    df = pd.DataFrame({"participant_id": range(1, n + 1), "assignment": assignments})
    return df


def block_randomize(
    n: int,
    block_size: int = 4,
    n_groups: int = 2,
    group_labels: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Permuted block randomization.

    Ensures balance within each block. Block size must be a multiple of n_groups.

    Parameters
    ----------
    n : int
        Total number of participants to randomize.
    block_size : int
        Size of each block (must be divisible by n_groups).
    n_groups : int
        Number of treatment arms.
    group_labels : list, optional
    seed : int, optional

    Returns
    -------
    DataFrame with participant_id and assignment columns.
    """
    if block_size % n_groups != 0:
        raise ValueError(f"block_size ({block_size}) must be divisible by n_groups ({n_groups}).")

    rng = np.random.default_rng(seed)
    if group_labels is None:
        group_labels = [f"Group_{i}" for i in range(n_groups)]

    per_arm = block_size // n_groups
    block_template = group_labels * per_arm

    assignments = []
    while len(assignments) < n:
        block = block_template.copy()
        rng.shuffle(block)
        assignments.extend(block)

    assignments = assignments[:n]
    df = pd.DataFrame({"participant_id": range(1, n + 1), "assignment": assignments})
    return df


def stratified_randomize(
    participants: pd.DataFrame,
    stratify_cols: List[str],
    block_size: int = 4,
    n_groups: int = 2,
    group_labels: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Stratified block randomization.

    Randomizes participants within strata defined by stratify_cols, using
    permuted blocks within each stratum to maintain balance.

    Parameters
    ----------
    participants : DataFrame
        Must contain an 'id' column and the stratify_cols.
    stratify_cols : list of str
        Columns to stratify on (e.g. ['site', 'sex', 'age_group']).
    block_size : int
    n_groups : int
    group_labels : list, optional
    seed : int, optional

    Returns
    -------
    DataFrame with original columns plus 'assignment'.
    """
    if group_labels is None:
        group_labels = [f"Group_{i}" for i in range(n_groups)]

    result_frames = []
    # Create stratum key
    df = participants.copy().reset_index(drop=True)
    df["_stratum"] = df[stratify_cols].astype(str).agg("|".join, axis=1)

    for stratum, group_df in df.groupby("_stratum"):
        n_stratum = len(group_df)
        assigned = block_randomize(
            n=n_stratum,
            block_size=block_size,
            n_groups=n_groups,
            group_labels=group_labels,
            seed=abs(hash(stratum)) % (2**31) if seed is None else seed + abs(hash(stratum)) % 1000,
        )
        group_df = group_df.reset_index(drop=True)
        group_df["assignment"] = assigned["assignment"].values
        result_frames.append(group_df)

    result = pd.concat(result_frames).drop(columns=["_stratum"]).sort_index()
    return result


def cluster_randomize(
    clusters: List[str],
    n_groups: int = 2,
    group_labels: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Cluster randomization — randomize entire clusters to arms.

    Parameters
    ----------
    clusters : list of str
        Cluster identifiers (e.g. school names, clinic IDs).
    n_groups : int
    group_labels : list, optional
    seed : int, optional
    """
    rng = np.random.default_rng(seed)
    if group_labels is None:
        group_labels = [f"Group_{i}" for i in range(n_groups)]

    shuffled_clusters = list(clusters)
    rng.shuffle(shuffled_clusters)

    assignments = []
    for i, cluster in enumerate(shuffled_clusters):
        assignments.append({"cluster": cluster, "assignment": group_labels[i % n_groups]})

    return pd.DataFrame(assignments)


# ─────────────────────────────────────────────
# 3. AB Test Duration Estimation
# ─────────────────────────────────────────────

def ab_test_duration(
    baseline_rate: float,
    minimum_detectable_effect: float,
    daily_traffic: int,
    allocation_fraction: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
    n_variants: int = 2,
) -> Dict[str, float]:
    """
    Estimate how many days an A/B test should run.

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion / success rate (0–1).
    minimum_detectable_effect : float
        Relative change to detect (e.g. 0.05 = 5% lift).
    daily_traffic : int
        Total eligible users per day.
    allocation_fraction : float
        Fraction of traffic included in the test.
    alpha : float
    power : float
    n_variants : int
        Number of arms including control.

    Returns
    -------
    dict with required_n, daily_n_in_test, days_needed
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    p2 = min(p2, 0.9999)

    size_result = calculate_sample_size(
        p1=p1, p2=p2, alpha=alpha / (n_variants - 1),  # Bonferroni for multiple arms
        power=power, verbose=False
    )
    required_n = size_result["n_per_group"] * n_variants

    daily_n = int(daily_traffic * allocation_fraction / n_variants) * n_variants
    days_needed = math.ceil(required_n / daily_n)

    return {
        "required_n_total": required_n,
        "daily_n_in_test": daily_n,
        "days_needed": days_needed,
        "baseline_rate": p1,
        "target_rate": round(p2, 4),
        "alpha_adjusted": round(alpha / (n_variants - 1), 4),
    }


# ─────────────────────────────────────────────
# 4. Pre-registration Helper
# ─────────────────────────────────────────────

def generate_preregistration_document(
    study_title: str,
    hypotheses: List[str],
    primary_outcome: str,
    sample_size_justification: str,
    analysis_plan: str,
    randomization_method: str,
    blinding: str,
    output_path: str = "preregistration.md",
) -> str:
    """
    Generate a structured pre-registration markdown document.
    Upload manually to OSF (https://osf.io) or via the OSF Python client.
    """
    lines = [
        f"# Pre-registration: {study_title}",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        "## Hypotheses",
        "",
    ]
    for i, h in enumerate(hypotheses, 1):
        lines.append(f"{i}. {h}")
    lines += [
        "",
        "## Primary Outcome",
        "",
        primary_outcome,
        "",
        "## Sample Size Justification",
        "",
        sample_size_justification,
        "",
        "## Randomization",
        "",
        randomization_method,
        "",
        "## Blinding",
        "",
        blinding,
        "",
        "## Analysis Plan",
        "",
        analysis_plan,
        "",
        "---",
        "_This document was auto-generated and should be reviewed before submission._",
    ]
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"Pre-registration document written to: {output_path}")
    return content
```

---

## Example 1 — Two-Arm RCT (Clinical Trial)

```python
import pandas as pd
from experimental_design import (
    calculate_sample_size,
    inflate_for_attrition,
    block_randomize,
    generate_preregistration_document,
)

# ── Step 1: Power calculation ──────────────────────────────────────────────
# Intervention expected to improve recovery rate from 40% to 55%
result = calculate_sample_size(p1=0.40, p2=0.55, alpha=0.05, power=0.80)
# n_per_group: 128, total_n: 256

# Account for expected 15% dropout
n_inflated = inflate_for_attrition(result["n_per_group"], expected_attrition_rate=0.15)
print(f"Enrol {n_inflated} per arm ({n_inflated * 2} total) to account for 15% attrition.")

# ── Step 2: Block randomization (block size 4, 2 arms) ─────────────────────
allocation = block_randomize(
    n=n_inflated * 2,
    block_size=4,
    n_groups=2,
    group_labels=["Control", "Intervention"],
    seed=2024,
)
print(allocation.head(8))
print(allocation["assignment"].value_counts())

# ── Step 3: Pre-registration document ─────────────────────────────────────
generate_preregistration_document(
    study_title="Effect of Intervention X on Recovery Rate",
    hypotheses=[
        "Intervention X will increase 30-day recovery rate compared to control.",
        "No difference in serious adverse events between arms.",
    ],
    primary_outcome="30-day binary recovery status (recovered / not recovered).",
    sample_size_justification=(
        f"N={n_inflated} per arm (inflated from {result['n_per_group']} "
        f"for 15% attrition). Based on 40% vs 55% recovery rates, "
        f"alpha=0.05, power=0.80."
    ),
    analysis_plan=(
        "Primary analysis: chi-square test. Secondary: logistic regression "
        "adjusting for age and baseline severity."
    ),
    randomization_method="Permuted block randomization, block size 4, centralised allocation.",
    blinding="Outcome assessors blinded; participants and clinicians not blinded.",
    output_path="rct_preregistration.md",
)
```

---

## Example 2 — Multi-Arm Web Experiment with Stratification

```python
import pandas as pd
from experimental_design import (
    calculate_sample_size,
    stratified_randomize,
    ab_test_duration,
)

# ── Sample size for 3-arm experiment (ANOVA approach) ─────────────────────
# Cohen's f = 0.25 (medium effect across 3 groups)
result = calculate_sample_size(
    effect_size=0.25,
    alpha=0.05,
    power=0.80,
    test_type="anova",
    k_groups=3,
)
print(result)

# ── Stratified randomization by country and user_type ─────────────────────
# Simulate a participant pool
rng_sim = pd.np.random.default_rng(99) if hasattr(pd, 'np') else __import__('numpy').random.default_rng(99)
import numpy as np
rng_sim = np.random.default_rng(99)

n_users = result["total_n"]
participants = pd.DataFrame({
    "id": range(1, n_users + 1),
    "country": np.random.choice(["US", "UK", "DE"], size=n_users, p=[0.5, 0.3, 0.2]),
    "user_type": np.random.choice(["free", "paid"], size=n_users, p=[0.7, 0.3]),
})

allocation = stratified_randomize(
    participants=participants,
    stratify_cols=["country", "user_type"],
    block_size=6,
    n_groups=3,
    group_labels=["Control", "Variant_A", "Variant_B"],
    seed=2024,
)
print(allocation.groupby(["country", "user_type", "assignment"]).size().unstack())

# ── AB test duration estimation ────────────────────────────────────────────
duration = ab_test_duration(
    baseline_rate=0.03,             # 3% baseline conversion
    minimum_detectable_effect=0.20, # detect 20% relative lift (3% → 3.6%)
    daily_traffic=5000,
    allocation_fraction=0.80,       # 80% of traffic in the experiment
    alpha=0.05,
    power=0.80,
    n_variants=3,
)
print(f"Run for at least {duration['days_needed']} days.")
```

---

## Blinding Protocols

| Trial Type | Who Is Blinded |
|---|---|
| Open-label | Nobody (appropriate when blinding is impractical) |
| Single-blind | Participants only |
| Double-blind | Participants + investigators |
| Triple-blind | Participants + investigators + outcome assessors |
| Cluster-blind | Outcome assessors blinded to cluster allocation |

---

## Pre-registration on OSF

```bash
# Install OSF client
pip install osfclient

# Authenticate (set token as env var — never hardcode)
export OSF_TOKEN=<paste-your-osf-token>

# Upload pre-registration document to your OSF project
osf -p <your-project-id> upload rct_preregistration.md osfstorage/prereg/rct_preregistration.md

# Verify upload
osf -p <your-project-id> ls
```

---

## Checklist Before Data Collection

- [ ] Primary outcome and analysis plan registered on OSF / ClinicalTrials.gov / AsPredicted
- [ ] Sample size calculation documented with effect size source
- [ ] Attrition inflation applied
- [ ] Randomization sequence sealed (allocation concealment)
- [ ] Blinding procedures in place
- [ ] Stopping rules defined (safety monitoring)
- [ ] IRB / ethics approval obtained
- [ ] Data management plan completed
