# Contributing Guide

Thank you for your interest in contributing to **awesome-rosetta-skills**!

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Skill Writing Standards](#2-skill-writing-standards)
3. [PR Submission Workflow](#3-pr-submission-workflow)
4. [CI Checks Explained](#4-ci-checks-explained)
5. [Domain Expert Review](#5-domain-expert-review)
6. [Contributor Incentives](#6-contributor-incentives)
7. [Code of Conduct](#7-code-of-conduct)

---

## 1. Quick Start

```bash
# Clone and setup
git clone https://github.com/xjtulyc/awesome-rosetta-skills.git
cd awesome-rosetta-skills
git submodule update --init --recursive
pip install pyyaml

# Validate a Skill locally
python scripts/validate_skill.py skills/your-discipline/your-skill/SKILL.md

# Create a new Skill from template
mkdir -p skills/07-economics/my-new-skill
cp templates/SKILL_TEMPLATE.md skills/07-economics/my-new-skill/SKILL.md

# Submit
git checkout -b feat/07-economics-my-new-skill
git add skills/07-economics/my-new-skill/
git commit -m "feat(economics): add my-new-skill"
git push origin feat/07-economics-my-new-skill
```

---

## 2. Skill Writing Standards

See [SKILL_STANDARD.md](SKILL_STANDARD.md) for full details. Key points:

### File Naming

- Directory name: all lowercase + hyphens — `did-causal`, `bayesian-stats`
- Main file: always `SKILL.md` (uppercase)
- Script files: kebab-case — `run-analysis.py`, `setup-env.sh`

### Required Frontmatter

```yaml
---
name: skill-name
description: >
  Use this Skill when the user needs to perform X analysis.
  Covers Y method using Z tools. (50-150 chars)
tags:
  - discipline-category
  - method-type
  - tool-name
version: "1.0.0"
authors:
  - name: Your Name
    github: "@your_github"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
dependencies:
  python:
    - package-name>=1.0.0
last_updated: "YYYY-MM-DD"
---
```

### Required Body Sections

Every SKILL.md must contain these sections in order:

```
# Skill Name
## When to Use This Skill        (5-10 concrete trigger scenarios)
## Background & Key Concepts     (domain knowledge the Agent needs)
## Environment Setup             (install deps, configure API keys)
## Core Workflow                 (step-by-step with executable code)
## Advanced Usage                (edge cases, performance tuning)
## Troubleshooting               (known bugs, version compatibility)
## External Resources            (official docs, key papers)
## Examples                      (2-3 complete end-to-end examples)
```

### Code Quality Requirements

```python
# Good: real executable code, explicit versions, error handling
import pymc as pm   # pymc>=5.0
import arviz as az

with pm.Model() as model:
    mu = pm.Normal("mu", mu=0, sigma=1)
    obs = pm.Normal("obs", mu=mu, sigma=1, observed=data)
    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Bad: pseudocode — never do this
# trace = run_mcmc(model)
```

```python
# Good: include error handling
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
except requests.exceptions.Timeout:
    print("Request timed out — check your network connection")
except requests.exceptions.HTTPError as e:
    print(f"API error: {e.response.status_code}")
```

### Prohibited Practices

- **No hardcoded API keys** — use `os.getenv("API_KEY")`
- **No pseudocode** — all code must be genuinely executable
- **No copying upstream content** — reference via git submodule
- **No overlap with AI/ML or biomedicine** — covered by Orchestra/K-Dense
- **No Skills under 300 lines**

---

## 3. PR Submission Workflow

### Branch Naming

```
feat/XX-discipline-skill-name     # new Skill
fix/XX-discipline-skill-name      # bug fix
update/XX-discipline-skill-name   # version/content update
docs/topic-name                   # documentation
```

### Commit Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(economics): add did-causal skill with Callaway-Sant'Anna estimator
fix(mathematics): update PyMC to v5 API in bayesian-stats
update(universal): add Semantic Scholar API v2 endpoints
```

### Merge Requirements

| Skill type | CI | Reviewers | Domain expert |
|:-----------|:---|:----------|:--------------|
| P0 new Skill | All pass | 2 | Required |
| P1/P2 new Skill | All pass | 1 | Recommended |
| Fix/update existing | All pass | 1 | Optional |
| Docs only | All pass | 1 | Not needed |

---

## 4. CI Checks Explained

| Check | Pass condition |
|:------|:---------------|
| Frontmatter format | All required fields present and correctly typed |
| Minimum line count | ≥ 300 lines |
| Code block count | ≥ 2 code blocks |
| No hardcoded secrets | No API key pattern matches |
| External links (weekly) | HTTP 200 response |

Run locally before pushing:

```bash
python scripts/validate_skill.py path/to/SKILL.md --verbose
```

---

## 5. Domain Expert Review

P0 Skills must pass domain expert review before merging.

**Becoming a Domain Maintainer**: Contribute 5+ merged high-quality Skills in a discipline then open a maintainer application Issue. Maintainers can approve P1/P2 PRs directly and co-review P0 PRs.

If a Skill is awaiting expert review, add to frontmatter:

```yaml
status: "awaiting-expert-review"
```

---

## 6. Contributor Incentives

| Contribution | Points |
|:-------------|:-------|
| Merge a P0 Skill | 10 |
| Merge a P1/P2 Skill | 5 |
| Fix a bug | 2 |
| Update dependency version | 1 |
| Complete expert review | 3 |

Every Skill's `authors` field permanently records your contribution. Core Contributors appear in the README at 50+ points.

---

## 7. Code of Conduct

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

For concerns, contact maintainers via GitHub Issues or Discussions.
