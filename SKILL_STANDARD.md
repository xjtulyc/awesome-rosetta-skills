# Skill Quality Standard

**Version**: v1.0.0
**Last Updated**: 2026-03-17
**Maintainer**: awesome-rosetta-skills Core Team

This document defines the quality standards every `SKILL.md` file in this repository must meet.
All PRs must pass every check here before merging.

---

## Table of Contents

1. [Frontmatter Specification](#1-frontmatter-specification)
2. [Body Structure Specification](#2-body-structure-specification)
3. [Code Example Standards](#3-code-example-standards)
4. [Content Quality Standards](#4-content-quality-standards)
5. [Compliance Requirements](#5-compliance-requirements)
6. [Quality Checklist](#6-quality-checklist)
7. [Automated Validation Rules](#7-automated-validation-rules)

---

## 1. Frontmatter Specification

Every `SKILL.md` **must begin with a YAML frontmatter block** containing all required fields:

```yaml
---
# =============================================
# Required fields (missing any -> CI FAIL)
# =============================================

name: skill-name-in-kebab-case
# Rule: all lowercase, words separated by hyphens, matches directory name
# Example: did-causal, bayesian-stats, literature-search

description: >
  One-sentence description of the Skill's function and trigger conditions.
  Must include trigger keywords (e.g. "when the user needs to run X analysis").
  Length: 50-150 characters.
# Example:
#   Use this Skill when the user needs to perform Difference-in-Differences (DID)
#   causal inference, including parallel trends testing, Callaway-Sant'Anna
#   heterogeneous treatment effect estimation, and event study plots.

tags:
  - discipline-category   # e.g. economics, physics, universal
  - method-type           # e.g. causal-inference, bayesian, visualization
  - tool-name             # e.g. pymc, linearmodels, openalex
# At least 3 tags, no more than 8

version: "1.0.0"
# Follows Semantic Versioning (SemVer): Major.Minor.Patch
# First submission: always "1.0.0"

authors:
  - name: Contributor Name
    github: "@github_username"
# Multiple authors allowed; original authors are never removed on later edits

license: "MIT"
# Must be MIT or more permissive (Apache-2.0, CC0, BSD-2-Clause, etc.)

platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
# Must include at least claude-code

dependencies:
  python:                   # optional; omit block if no Python deps
    - package-name>=X.Y.Z   # explicit minimum version required
  r:                        # optional
    - package-name
  system:                   # optional
    - tool-name>=version

last_updated: "YYYY-MM-DD"
# ISO 8601 format; must be updated on every substantive edit

# =============================================
# Optional fields (if present, format must be valid)
# =============================================

status: "stable"
# Values: stable | beta | experimental | awaiting-expert-review | deprecated

related_skills:
  - ../other-skill/SKILL.md

external_refs:
  orchestra: "orchestra-ai-research/skills/xxx"
  kdense:    "kdense-scientific/xxx"

---
```

### Writing a good `description`

The `description` field is the primary signal an AI agent uses to decide whether to invoke this Skill. It must:

1. **Open with a trigger scenario**: `Use this Skill when the user needs to...`
2. **Include method keywords**: exact names of statistical/analytical methods
3. **Include tool keywords**: main library or API names
4. **Include discipline keywords**: the academic field

```yaml
# Good
description: >
  Use this Skill when the user needs to perform Difference-in-Differences (DID)
  causal inference. Covers parallel trends testing, Callaway-Sant'Anna heterogeneous
  treatment effect estimation, and event study visualization with linearmodels / R did.

# Bad (too vague — agent cannot decide when to trigger)
description: "A skill for econometrics analysis"
```

---

## 2. Body Structure Specification

Every `SKILL.md` body **must contain all of the following sections**, in order:

```markdown
# Skill Name (English)

## When to Use This Skill
(5-10 specific trigger scenarios as bullet list)
**Trigger keywords** (for agent search): keyword1, keyword2 ...

## Background & Key Concepts
(Domain knowledge to help the Agent understand context; minimum 200 words)
(Mathematical formulas in LaTeX: $inline$ or $$block$$)

## Environment Setup
(Full dependency installation commands)
(API key acquisition steps — never include real keys)
(Environment variable setup)

## Core Workflow

### Step 1: Data Preparation
(Code example + explanation)

### Step 2: Model Estimation / Analysis
(Code example + explanation)

### Step 3: Results Interpretation & Output
(Code example + explanation)

## Advanced Usage
(Advanced features, edge cases, performance optimization)

## Troubleshooting
(Known bugs, version compatibility issues, common error messages + fixes)

## External Resources
(Official docs, authoritative tutorials, key papers — all links must be real)

## Examples

### Example 1: [Concrete Scenario Name]
(End-to-end complete example from input to output)

### Example 2: [Another Scenario]
(Another complete example)
```

### Line count requirements

| Level | Lines | Notes |
|:------|:------|:------|
| CI minimum | ≥ 300 | Below this → CI fails |
| Recommended | 300–500 | Core workflow + 2 full examples |
| Complete Skill | 500+ | Advanced usage + edge cases + full troubleshooting |

---

## 3. Code Example Standards

### Quantity requirement

Every SKILL.md must contain **at least 2 code blocks** (CI check). Recommended distribution:

- Environment setup: 1 block
- Core workflow: 2–4 blocks (one per step)
- Advanced usage: 1–2 blocks
- Complete examples: 2 blocks

### Language annotation

All code blocks **must have a language tag**:

```
```python    <- Python code
```r         <- R code
```bash      <- Shell / command line
```yaml      <- Config files
```json      <- JSON data examples
```

### Code quality requirements

```python
# Good: complete runnable example with version comment and error handling

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS   # linearmodels>=4.28

np.random.seed(42)
n_firms, n_periods = 100, 8

df = pd.DataFrame({
    "firm_id": np.repeat(range(n_firms), n_periods),
    "year":    np.tile(range(2016, 2024), n_firms),
    "treated": np.repeat(np.random.binomial(1, 0.5, n_firms), n_periods),
    "post":    np.tile([0, 0, 0, 0, 1, 1, 1, 1], n_firms),
    "outcome": np.random.normal(0, 1, n_firms * n_periods),
})
df["did"] = df["treated"] * df["post"]
df = df.set_index(["firm_id", "year"])

model = PanelOLS(df["outcome"], df[["did"]],
                 entity_effects=True, time_effects=True, drop_absorbed=True)
result = model.fit(cov_type="clustered", cluster_entity=True)
print(result.summary)
```

### Sensitive information handling

```python
# Good: environment variable
import os
api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# Absolutely prohibited: hardcoded secret
# api_key = "sk-abc123xyz789..."   # CI detects and rejects this pattern
```

---

## 4. Content Quality Standards

### Professional accuracy

- Statistical method descriptions must be correct — **no substantive errors**
- Cite papers with author-year: `(Callaway & Sant'Anna, 2021)`
- Version numbers must be real releases (CI can verify against PyPI/CRAN)
- For contested methods: note "this implementation follows [paper/doc]"

### Timeliness

- API endpoints use the current latest version
- `last_updated` reflects the most recent substantive edit date
- Migration path for deprecated dependencies documented in Troubleshooting

### Differentiation from existing Skills

Before submitting:
1. `grep -r "your-core-method" skills/` to confirm no duplication
2. If related, cross-reference in `related_skills` — do not re-explain

---

## 5. Compliance Requirements

### License compliance

| Code source | Handling |
|:------------|:---------|
| Original code | MIT by default; no annotation needed |
| Official docs examples | Cite source URL |
| Stack Overflow / GitHub snippets | Cite URL + confirm CC BY-SA or MIT |
| Academic paper code | Cite paper reference + confirm license |

### Data compliance

- No real personal data in code
- Use `np.random.seed(42)` for synthetic data, or reference a public dataset
- Cite data source and copyright terms for public datasets used

---

## 6. Quality Checklist

Run through every item before submitting a PR:

### Format checks

```
[ ] Frontmatter has all required fields (name, description, tags, version,
    authors, license, platforms, dependencies, last_updated)
[ ] SKILL.md total line count >= 300
[ ] At least 2 code blocks, each with a language annotation
[ ] All required sections present (When to Use, Background, Setup,
    Workflow, Advanced, Troubleshooting, Resources, Examples)
[ ] Code block syntax correct (no obvious indentation errors)
```

### Content checks

```
[ ] description accurately describes trigger scenarios (method + tool + discipline)
[ ] All code examples genuinely executable (ran at least one complete example locally)
[ ] Dependency versions explicit and real
[ ] At least one error-handling example (try/except or equivalent)
[ ] No substantial overlap with existing Skills (searched)
[ ] Professional content accurate (or status: awaiting-expert-review added)
```

### Compliance checks

```
[ ] No hardcoded API keys (no sk-, token=, api_key= followed by literal strings)
[ ] Third-party code sources cited
[ ] license field is MIT or more permissive
[ ] No real personal data in examples
```

---

## 7. Automated Validation Rules

`scripts/validate_skill.py` enforces the following rules (see script comments for details):

| Rule ID | Description | Level |
|:--------|:------------|:------|
| F001 | Frontmatter exists and is valid YAML | ERROR |
| F002 | All required fields present | ERROR |
| F003 | `name` field is kebab-case | ERROR |
| F004 | `version` field follows SemVer | ERROR |
| F005 | `last_updated` field is ISO date | ERROR |
| F006 | `license` is an allowed value | ERROR |
| F007 | `tags` has at least 3 entries | WARNING |
| F008 | `authors` entries have `name` field | WARNING |
| F009 | `platforms` includes `claude-code` | WARNING |
| F010 | `description` is 30–300 characters | ERROR / WARNING |
| C001 | Total body line count ≥ 300 | ERROR |
| C002 | At least 2 code blocks | ERROR |
| C003 | All code blocks have language annotation | WARNING |
| C004 | Required section headings present | WARNING |
| C005 | File has a top-level heading | WARNING |
| S001 | No hardcoded API key patterns | ERROR |
| S002 | No `password=` or `secret=` literals | ERROR |

Any **ERROR** causes CI to block the PR merge. **WARNING** triggers a PR comment but does not block.

---

*This standard document is based on the project PRD v1.0 and will be updated quarterly with the repository's evolution.*
