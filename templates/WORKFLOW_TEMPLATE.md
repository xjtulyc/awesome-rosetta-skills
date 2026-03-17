---
name: workflow-name
description: >
  Use this Skill when the user needs to execute the [WORKFLOW NAME] end-to-end workflow.
  Covers [step 1], [step 2], and [step 3] with [tool/method].
tags:
  - research-workflow
  - [discipline]
  - [method-type]
  - [tool-name]
version: "1.0.0"
authors:
  - name: [Your Name]
    github: "@your_github_username"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - package-name>=version
  r:
    - package-name
  system:
    - tool-name>=version
last_updated: "YYYY-MM-DD"
---

# [Workflow Name]

> **TL;DR** — [One sentence describing the complete workflow, key inputs, and outputs.]

---

## 1. When to Use This Workflow

Use this Skill when:

- [Trigger scenario 1 — be specific]
- [Trigger scenario 2]
- [Trigger scenario 3]
- [Trigger scenario 4]
- [Trigger scenario 5]

**Do NOT use this Skill when:**

- [Anti-pattern 1 — describe what a different skill handles]
- [Anti-pattern 2]

---

## 2. Workflow Overview

```
Input: [describe required input data/format]
  │
  ▼
Step 1: [Name]
  │   [Brief description of what happens]
  ▼
Step 2: [Name]
  │   [Brief description]
  ▼
Step 3: [Name]
  │   [Brief description]
  ▼
Output: [describe output format and location]
```

**Key decisions in this workflow:**

| Decision Point | Option A | Option B | When to choose |
|---|---|---|---|
| [Decision] | [Option A] | [Option B] | [Guidance] |

---

## 3. Environment Setup

```bash
# Create and activate environment
conda create -n [env-name] python=3.11 -y
conda activate [env-name]

# Install Python dependencies
pip install [package1] [package2] [package3]

# R dependencies (if applicable)
# Rscript -e 'install.packages(c("package1", "package2"))'

# Environment variables (never hard-code keys in scripts)
export API_KEY_NAME="<paste-your-key>"
```

Verify setup:

```python
import [package1]
import [package2]
print(f"[package1] version: {[package1].__version__}")
print("Environment OK")
```

---

## 4. Step 1: [Name — Data Collection / Input Preparation]

**Goal:** [What this step achieves]

**Inputs:** [What you need before this step]

**Outputs:** [What you produce in this step]

```python
import [required_packages]
from pathlib import Path
from typing import Optional
import pandas as pd


def step1_function(
    param1: str,
    param2: int,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    [Docstring describing what this function does.]

    Args:
        param1: [Description]
        param2: [Description]
        output_dir: [Description]

    Returns:
        [Description of return value]
    """
    # [Implementation here]
    pass
```

**Common issues in Step 1:**

- [Issue]: [Solution]
- [Issue]: [Solution]

---

## 5. Step 2: [Name — Core Processing / Analysis]

**Goal:** [What this step achieves]

```python
def step2_function(
    input_data: pd.DataFrame,
    config: dict = None,
) -> dict:
    """
    [Docstring]

    Args:
        input_data: Output from step1_function().
        config: Optional configuration overrides.

    Returns:
        Dictionary with keys: [key1 (description), key2 (description)]
    """
    config = config or {}
    results = {}

    # [Implementation]

    return results
```

---

## 6. Step 3: [Name — Output / Reporting]

**Goal:** [What this step achieves — visualization, export, report generation, etc.]

```python
import matplotlib.pyplot as plt


def step3_output(
    results: dict,
    output_prefix: str = "analysis",
    formats: list = None,
) -> None:
    """
    Generate output files from analysis results.

    Args:
        results: Output from step2_function().
        output_prefix: Filename prefix for output files.
        formats: Output formats, e.g. ["csv", "pdf", "html"].
    """
    formats = formats or ["csv", "pdf"]

    # [Implementation]
    pass
```

---

## 7. Complete Pipeline Function

```python
def run_complete_workflow(
    # Step 1 inputs
    param1: str,
    # Step 2 config
    config: dict = None,
    # Step 3 outputs
    output_dir: str = "results",
    output_formats: list = None,
    # Global options
    verbose: bool = True,
) -> dict:
    """
    Run the complete [workflow name] pipeline end-to-end.

    Args:
        param1:          [Description]
        config:          Optional configuration dict for Step 2.
        output_dir:      Directory for output files.
        output_formats:  List of output formats (default: ["csv", "pdf"]).
        verbose:         Print progress messages.

    Returns:
        Dictionary with keys: step1_output, step2_results, output_paths.
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Step 1: [Name]...")
    step1_result = step1_function(param1)

    if verbose:
        print(f"Step 2: [Name]...")
    step2_result = step2_function(step1_result, config)

    if verbose:
        print(f"Step 3: Generating outputs...")
    step3_output(step2_result, output_prefix=f"{output_dir}/results",
                 formats=output_formats or ["csv", "pdf"])

    return {
        "step1_output": step1_result,
        "step2_results": step2_result,
        "output_dir": output_dir,
    }
```

---

## 8. End-to-End Examples

### Example 1 — [Basic use case]

```python
# [Brief setup / data loading]

results = run_complete_workflow(
    param1="example_value",
    output_dir="example_1_output",
    verbose=True,
)

print(f"Results: {results['step2_results']}")
```

**Expected output:**

```
Step 1: [Name]...
Step 2: [Name]...
Step 3: Generating outputs...
Results: {key1: value, key2: value}
```

### Example 2 — [Advanced / edge case]

```python
# [More complex example with different configuration]

config = {
    "option1": True,
    "threshold": 0.05,
}

results = run_complete_workflow(
    param1="advanced_example",
    config=config,
    output_dir="example_2_output",
    output_formats=["csv", "pdf", "html"],
)
```

---

## 9. Common Errors and Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `[ErrorType]: [message]` | [Root cause] | [How to fix] |
| `ModuleNotFoundError: [name]` | Missing dependency | `pip install [package]` |
| `ValueError: [message]` | [Root cause] | [Fix] |
| Slow performance | [Cause] | [Optimization tip] |

---

## 10. Configuration Reference

All configuration keys for Step 2 `config` dict:

| Key | Type | Default | Description |
|---|---|---|---|
| `option1` | bool | `False` | [What it controls] |
| `threshold` | float | `0.05` | [What it controls] |
| `n_iterations` | int | `1000` | [What it controls] |

---

## 11. References and Further Reading

- [Primary documentation / paper]
- [Tutorial or guide]
- [Related Skill in this repo]

---

## 12. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | YYYY-MM-DD | Initial release |
