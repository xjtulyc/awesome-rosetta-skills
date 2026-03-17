---
# ============================================================
# SKILL.md Standard Template — awesome-rosetta-skills v1.0
# Usage: copy to skills/XX-discipline/skill-name/SKILL.md
#        then replace every [TODO] placeholder
# ============================================================

name: skill-name-in-kebab-case        # [TODO] replace with your skill name, matching directory name
description: >
  [TODO] Use this Skill when the user needs to perform X analysis.
  Covers Y method using Z tools. (50-150 chars, include trigger keywords.)
tags:
  - discipline-name          # [TODO] e.g. economics, physics
  - method-type              # [TODO] e.g. causal-inference, bayesian
  - main-tool                # [TODO] e.g. pymc, linearmodels
version: "1.0.0"
authors:
  - name: Your Name          # [TODO]
    github: "@your_github"   # [TODO]
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - package-name>=X.Y.Z    # [TODO] replace with real packages and versions
  # r:                       # [TODO] remove this block if no R deps
  #   - package-name
  # system:                  # [TODO] remove if no system deps
  #   - tool-name
last_updated: "YYYY-MM-DD"   # [TODO] today's date, format: 2026-03-17
status: "stable"             # stable | beta | experimental | awaiting-expert-review
---

# [TODO] Skill Name

> **One-line summary**: [TODO] This Skill helps researchers do X, suited for Y scenarios.

---

## When to Use This Skill

[TODO] Use this Skill in the following scenarios:

- When you need to analyze **[TODO scenario 1]**
- When your data has a **[TODO scenario 2]** structure
- When you need to test **[TODO hypothesis / method name]**
- When you are using **[TODO software/database]** to process **[TODO data type]**
- When you need to produce **[TODO output type]** (e.g. figures, tables, reports)

**Trigger keywords** (for agent search):
[TODO method1], [TODO method2], [TODO tool1], [TODO tool2]

---

## Background & Key Concepts

### [TODO Concept 1 Name]

[TODO] Explain the core principles of this method/tool in at least 100 words.

[TODO] Core formula in LaTeX:

$$
[TODO: core formula, e.g. \hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty]
$$

Where:
- $[TODO: symbol]$: [TODO: meaning]
- $[TODO: symbol]$: [TODO: meaning]

### [TODO Concept 2] (optional)

[TODO] Additional background knowledge.

### Comparison with Related Methods

| Method | Best for | Key assumption | Limitation |
|:-------|:---------|:---------------|:-----------|
| [TODO this method] | [TODO] | [TODO] | [TODO] |
| [TODO alternative 1] | [TODO] | [TODO] | [TODO] |
| [TODO alternative 2] | [TODO] | [TODO] | [TODO] |

---

## Environment Setup

### Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install Python dependencies
pip install package-name>=X.Y.Z another-package>=Y.Z   # [TODO replace]

# R dependencies (if needed)
# install.packages(c("package1", "package2"))           # [TODO replace]
```

### API Key Configuration (if required)

```bash
# [TODO] Steps to obtain an API key:
# 1. Visit https://[TODO provider website]
# 2. Register and request an API key
# 3. Set environment variable:

export TODO_API_KEY="your-api-key-here"   # Linux/macOS
# $env:TODO_API_KEY="your-api-key-here"   # Windows PowerShell
```

```python
# Read environment variable in Python (never hardcode!)
import os
api_key = os.getenv("TODO_API_KEY", "")
if not api_key:
    print("Warning: TODO_API_KEY not set; using anonymous access (rate-limited)")
```

### Verify Installation

```python
# Run this to confirm the environment is configured correctly
import package_name   # [TODO replace]
print(f"package_name version: {package_name.__version__}")
# Expected: package_name version: X.Y.Z or higher
```

---

## Core Workflow

### Step 1: Data Preparation

[TODO: Describe the required input data format]

```python
import numpy as np
import pandas as pd

# [TODO] Replace with real data loading code
# Example: synthetic data to demonstrate the format
np.random.seed(42)

df = pd.DataFrame({
    "id":      [TODO: column description],   # e.g. "entity ID, int or string"
    "time":    [TODO: column description],   # e.g. "time period, integer year"
    "outcome": [TODO: column description],   # e.g. "outcome variable, continuous"
    "feature": [TODO: column description],   # e.g. "control variable"
})

print(df.head())
print(df.info())
# Expected: [TODO describe expected shape and types]
```

**Data requirements**:
- `[TODO column]`: [TODO type and description]
- `[TODO column]`: [TODO type and description]
- Recommended sample size: [TODO e.g. "at least 30 observations per group"]

### Step 2: [TODO Analysis Execution]

[TODO: Describe the core analysis step]

```python
# [TODO] Core analysis code — must be genuinely executable, not pseudocode

try:
    result = [TODO: actual function call]
    print(result.summary)   # or equivalent output

except Exception as e:
    print(f"Analysis error: {e}")
    # [TODO: common causes of this error]
```

**Parameter reference**:

| Parameter | Meaning | Recommended value | Notes |
|:----------|:--------|:------------------|:------|
| `[TODO param1]` | [TODO] | [TODO] | [TODO] |
| `[TODO param2]` | [TODO] | [TODO] | [TODO] |

### Step 3: Results Interpretation & Output

[TODO: Describe how to interpret the output]

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
# [TODO: actual plotting code]

ax.set_xlabel("[TODO: x-axis label]")
ax.set_ylabel("[TODO: y-axis label]")
ax.set_title("[TODO: chart title]")
plt.tight_layout()
plt.savefig("output_figure.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Main estimate:  {result.params['[TODO]']:.4f}")
print(f"Standard error: {result.std_errors['[TODO]']:.4f}")
print(f"p-value:        {result.pvalues['[TODO]']:.4f}")
```

**Interpreting results**:
- **[TODO metric 1]**: [TODO how to interpret]
- **[TODO metric 2]**: [TODO how to interpret]

---

## Advanced Usage

### [TODO Advanced Scenario 1]

[TODO: When and why to use this advanced option]

```python
# [TODO] Advanced usage code
```

### Performance Optimization (if applicable)

```python
# [TODO] Parallelization, sparse matrices, memory optimization, etc.
```

---

## Troubleshooting

### Error: `[TODO common error message]`

**Cause**: [TODO: why this error occurs]

**Fix**:
```python
# [TODO: fix code or diagnostic steps]
```

### Issue: [TODO another common problem]

**Cause**: [TODO]

**Fix**: [TODO: description or code]

### Version Compatibility

| Package | Tested versions | Known issues |
|:--------|:----------------|:-------------|
| [TODO package] | [TODO version] | [TODO: none / or describe] |
| [TODO package] | [TODO version] | [TODO] |

---

## External Resources

### Official Documentation

- [TODO tool official docs](https://[TODO URL])
- [TODO another tool](https://[TODO URL])

### Key Papers

- [TODO Author et al. (YYYY). *Title*. Journal. DOI: xx.xxxx/xxxxxxx]
- [TODO Author et al. (YYYY). *Title*. Journal.]

### Tutorials

- [TODO tutorial title](https://[TODO URL])

### Data Sources (if applicable)

- [TODO database name](https://[TODO URL]): [TODO brief description]

---

## Examples

### Example 1: [TODO Scenario Name, e.g. "Estimating a Policy Intervention Effect"]

**Scenario**: [TODO concrete research question]

**Input data**: [TODO data format and source]

```python
# =============================================
# End-to-end example: [TODO scenario name]
# Requirements: Python 3.10+; see frontmatter for dependencies
# =============================================

import numpy as np
import pandas as pd
# [TODO: other imports]

# 1. Load data
np.random.seed(42)
# [TODO: construct or load example data]

# 2. Preprocessing
# [TODO: cleaning, transformation]

# 3. Core analysis
# [TODO: main computation]

# 4. Output results
print("Analysis complete. Key results:")
print(f"  [TODO metric]: [TODO example value]")
# Expected output: [TODO describe expected output]
```

**Interpreting these results**: [TODO]

---

### Example 2: [TODO Another Scenario]

**Scenario**: [TODO]

```python
# =============================================
# End-to-end example 2: [TODO scenario]
# =============================================

# [TODO: complete runnable code]
```

**Interpreting these results**: [TODO]

---

*Last updated: [TODO: YYYY-MM-DD] | Maintainer: [TODO: @github_username]*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
