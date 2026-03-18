---
name: preregistration-workflow
description: >
  Use this Skill to preregister a study on OSF or AsPredicted, generate CONSORT/STROBE/PRISMA
  compliance checklists, and track deviations between the registered protocol and final analysis.
tags:
  - universal
  - preregistration
  - open-science
  - OSF
  - CONSORT
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
    - requests>=2.28
    - pandas>=1.5
    - jinja2>=3.0
    - python-dotenv>=1.0
last_updated: "2026-03-18"
status: "stable"
---

# Preregistration Workflow

> **One-line summary**: Automate study preregistration on OSF/AsPredicted, generate compliance
> checklists (CONSORT, STROBE, PRISMA), and maintain a deviation log between plan and execution.

---

## When to Use This Skill

- When you need to **preregister a study** on OSF (Open Science Framework) or AsPredicted before data collection
- When you need to generate a **CONSORT checklist** for a randomized controlled trial (RCT)
- When you need a **STROBE checklist** for an observational study (cohort, case-control, cross-sectional)
- When you need a **PRISMA 2020 checklist** for a systematic review
- When you want to **log deviations** from the preregistered analysis plan
- When you are preparing a **Registered Report** (Stage 1 / Stage 2) for a journal

**Trigger keywords**: preregister, preregistration, OSF registration, AsPredicted, open science, registered report, CONSORT, STROBE, PRISMA, study protocol, HARKing, p-hacking prevention

---

## Background & Key Concepts

### Why Preregister?

Preregistration timestamps a study's hypotheses, design, and analysis plan *before* data collection.
It prevents two major threats to research validity:

- **HARKing** (Hypothesizing After Results are Known): presenting exploratory findings as confirmatory
- **p-hacking**: selectively reporting analyses that yield p < .05

Preregistered findings carry stronger evidential weight because the researcher could not have adjusted the
hypothesis to fit the data.

### Platform Comparison

| Platform | Best for | Output | API |
|---|---|---|---|
| **OSF** | All designs, flexible schema | Timestamped PDF + DOI | REST API |
| **AsPredicted** | Quick pre-registration (9 questions) | PDF | Manual only |
| **ClinicalTrials.gov** | Clinical trials (required by journals) | XML | REST API |
| **PROSPERO** | Systematic reviews | Web record | Limited |

### Deviation Tracking

Every deviation from the preregistered protocol must be documented with:
1. What was planned
2. What was done instead
3. Justification
4. Whether it was pre-specified as a contingency

---

## Environment Setup

### Install Dependencies

```bash
pip install requests>=2.28 pandas>=1.5 jinja2>=3.0 python-dotenv>=1.0 reportlab>=4.0
```

### OSF API Token

```bash
# 1. Go to https://osf.io/settings/tokens/
# 2. Create a personal access token with osf.full_write scope
# 3. Set environment variable:
export OSF_TOKEN="<paste-your-osf-token>"
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
OSF_TOKEN = os.getenv("OSF_TOKEN", "")
if not OSF_TOKEN:
    print("Warning: OSF_TOKEN not set. Set export OSF_TOKEN='<paste-your-token>'")

HEADERS = {
    "Authorization": f"Bearer {OSF_TOKEN}",
    "Content-Type": "application/json"
}
```

---

## Core Workflow

### Step 1: Create an OSF Project and Draft Registration

```python
import requests
import json
import os

OSF_API = "https://api.osf.io/v2"
OSF_TOKEN = os.getenv("OSF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {OSF_TOKEN}", "Content-Type": "application/json"}

def create_osf_project(title: str, description: str, public: bool = False) -> dict:
    """Create a new OSF project (node) via API."""
    payload = {
        "data": {
            "type": "nodes",
            "attributes": {
                "title": title,
                "description": description,
                "category": "project",
                "public": public,
            }
        }
    }
    resp = requests.post(f"{OSF_API}/nodes/", headers=HEADERS, json=payload)
    resp.raise_for_status()
    project = resp.json()["data"]
    print(f"Created project: {project['id']} — {project['attributes']['title']}")
    print(f"URL: https://osf.io/{project['id']}/")
    return project

def upload_protocol_file(node_id: str, file_path: str, filename: str) -> dict:
    """Upload a file (e.g., protocol PDF) to an OSF project."""
    upload_url = f"https://files.osf.io/v1/resources/{node_id}/providers/osfstorage/"
    with open(file_path, "rb") as fh:
        resp = requests.put(
            f"{upload_url}?name={filename}",
            headers={"Authorization": f"Bearer {OSF_TOKEN}"},
            data=fh
        )
    resp.raise_for_status()
    print(f"Uploaded {filename} → node {node_id}")
    return resp.json()

# Demo (requires valid OSF_TOKEN):
if OSF_TOKEN:
    project = create_osf_project(
        title="Study: Effect of X on Y — Preregistration",
        description="Preregistration for RCT testing the effect of intervention X on outcome Y.",
        public=False
    )
    PROJECT_ID = project["id"]
else:
    PROJECT_ID = "demo_node_id"
    print("Skipping OSF API call — set OSF_TOKEN to enable")
```

### Step 2: Generate AsPredicted-Style Protocol Document

```python
from jinja2 import Template
import datetime

ASPREDICTED_TEMPLATE = """
# AsPredicted Preregistration — {{ title }}

**Date:** {{ date }}
**Authors:** {{ authors }}
**Status:** DRAFT — not yet submitted

---

## 1. Data Collection
{{ data_collection }}

## 2. Hypothesis
{{ hypothesis }}

## 3. Dependent Variables (DVs)
{{ dependent_variables }}

## 4. Conditions / Manipulations (IVs)
{{ conditions }}

## 5. Analyses
{{ analyses }}

## 6. Outliers and Exclusions
{{ exclusions }}

## 7. Sample Size
{{ sample_size }}

## 8. Known to Any Authors
{{ known_data }}

## 9. Additional Comments
{{ additional }}

---
*Generated by awesome-rosetta-skills preregistration-workflow*
"""

def generate_aspredicted(
    title: str,
    authors: str,
    hypothesis: str,
    dv: str,
    iv: str,
    analyses: str,
    exclusions: str,
    n: str,
    data_collection: str = "No data has been collected for this study.",
    known_data: str = "No.",
    additional: str = ""
) -> str:
    """Render an AsPredicted-style preregistration document."""
    tmpl = Template(ASPREDICTED_TEMPLATE)
    return tmpl.render(
        title=title,
        date=datetime.date.today().isoformat(),
        authors=authors,
        data_collection=data_collection,
        hypothesis=hypothesis,
        dependent_variables=dv,
        conditions=iv,
        analyses=analyses,
        exclusions=exclusions,
        sample_size=n,
        known_data=known_data,
        additional=additional
    )

# Example preregistration
doc = generate_aspredicted(
    title="Mindfulness Training and Anxiety Reduction: RCT",
    authors="Jane Doe, John Smith",
    hypothesis=(
        "We predict that participants randomized to 8-week mindfulness-based stress reduction (MBSR) "
        "will show significantly lower GAD-7 scores at post-test (T2) relative to waitlist controls, "
        "with a medium effect size (d ≥ 0.50)."
    ),
    dv=(
        "Primary: GAD-7 total score at T2 (8 weeks). "
        "Secondary: PHQ-9 total score at T2; SWLS at T2."
    ),
    iv=(
        "Condition: MBSR group (n=50) vs. Waitlist control (n=50). "
        "Random assignment using computer-generated block randomization (block size=4)."
    ),
    analyses=(
        "Primary: ANCOVA on GAD-7 at T2, covariate = GAD-7 at T1 (baseline), "
        "with condition as between-subjects factor. Two-sided α=.05. "
        "Effect size: Cohen's d from adjusted means. "
        "Secondary analyses use same approach with PHQ-9 and SWLS. "
        "Mediation by mindfulness (FFMQ) using PROCESS macro (Hayes 2022), 5000 bootstrap samples."
    ),
    exclusions=(
        "Participants with >20% missing sessions excluded. "
        "Outliers: GAD-7 scores >3 SD from cell mean winsorized (not deleted). "
        "Intent-to-treat analysis as primary; per-protocol as sensitivity."
    ),
    n=(
        "N=100 (50 per arm). Based on: d=0.5, α=.05, power=.80, "
        "ANCOVA efficiency gain assumed 25% variance reduction by covariate. "
        "10% dropout buffer applied."
    )
)
print(doc[:500], "...[truncated]")

# Save to file
with open("preregistration_draft.md", "w", encoding="utf-8") as f:
    f.write(doc)
print("Saved: preregistration_draft.md")
```

### Step 3: Generate Compliance Checklists (CONSORT / STROBE / PRISMA)

```python
import pandas as pd

# --- CONSORT 2010 Checklist (abridged) ---
CONSORT_ITEMS = [
    ("1a", "Title", "Identification as RCT in title"),
    ("1b", "Abstract", "Structured summary of trial design, methods, results, conclusions"),
    ("2a", "Background", "Scientific background and explanation of rationale"),
    ("2b", "Objectives", "Specific objectives or hypotheses"),
    ("3a", "Trial design", "Description of trial design including allocation ratio"),
    ("3b", "Trial design", "Important changes to methods after trial commencement"),
    ("4a", "Participants", "Eligibility criteria for participants"),
    ("4b", "Setting", "Settings and locations where the data were collected"),
    ("5",  "Interventions", "Interventions for each group with details to allow replication"),
    ("6a", "Outcomes", "Pre-specified primary and secondary outcome measures"),
    ("7a", "Sample size", "How sample size was determined"),
    ("8a", "Randomization", "Method used to generate random allocation sequence"),
    ("9",  "Allocation concealment", "Mechanism used to implement allocation concealment"),
    ("10", "Implementation", "Who generated the sequence, enrolled, and assigned participants"),
    ("11a","Blinding", "If done, who was blinded after assignment"),
    ("12a","Statistical methods", "Methods for primary and secondary outcomes"),
    ("13a","Participant flow", "Numbers randomized to each group"),
    ("16", "Recruitment", "Dates defining the periods of recruitment and follow-up"),
    ("17a","Baseline data", "Baseline demographic and clinical characteristics"),
    ("18", "Numbers analyzed", "Number of participants in each group included in analysis"),
    ("19", "Outcomes", "Results for each outcome for each group, effect size and CI"),
    ("20", "Ancillary analyses", "Results of any subgroup or adjusted analyses"),
    ("21", "Harms", "All important harms or unintended effects in each group"),
    ("22", "Limitations", "Trial limitations, sources of potential bias"),
    ("23", "Generalisability", "Generalisability/external validity of trial findings"),
    ("24", "Interpretation", "Interpretation consistent with results"),
    ("25", "Registration", "Registration number and name of trial registry"),
    ("26", "Protocol", "Where the full trial protocol can be accessed"),
    ("27", "Funding", "Sources of funding and other support; role of funders"),
]

def generate_consort_checklist() -> pd.DataFrame:
    """Return CONSORT 2010 checklist as a DataFrame for self-assessment."""
    df = pd.DataFrame(CONSORT_ITEMS, columns=["Item", "Section", "Description"])
    df["Reported?"] = "[ ]"
    df["Page/Line"] = ""
    df["Notes"] = ""
    return df

consort = generate_consort_checklist()
print("CONSORT 2010 Checklist:")
print(consort[["Item", "Section", "Reported?"]].to_string(index=False))
consort.to_csv("consort_checklist.csv", index=False)
print(f"\nSaved consort_checklist.csv ({len(consort)} items)")

# --- STROBE Observational Checklist (brief) ---
STROBE_ITEMS = [
    ("1",  "Title/Abstract", "Indicate study design with commonly used term"),
    ("2",  "Background", "Explain scientific background and rationale"),
    ("3",  "Objectives", "State specific objectives, including pre-specified hypotheses"),
    ("4",  "Study design", "Present key elements of study design early"),
    ("5",  "Setting", "Describe setting, locations, and dates"),
    ("6",  "Participants", "Eligibility criteria, and methods of selection"),
    ("7",  "Variables", "Define all outcomes, exposures, predictors, confounders, effect modifiers"),
    ("8",  "Measurement", "Give sources and methods of assessment for each variable"),
    ("9",  "Bias", "Describe any efforts to address potential sources of bias"),
    ("10", "Study size", "Explain how study size was arrived at"),
    ("11", "Quantitative variables", "Explain how quantitative variables were handled"),
    ("12", "Statistical methods", "Describe all statistical methods including control of confounding"),
    ("16", "Main results", "Report unadjusted and adjusted estimates and precision (CIs)"),
    ("22", "Limitations", "Discuss limitations, taking into account sources of potential bias"),
]

strobe_df = pd.DataFrame(STROBE_ITEMS, columns=["Item", "Section", "Description"])
strobe_df["Reported?"] = "[ ]"
strobe_df.to_csv("strobe_checklist.csv", index=False)
print(f"Saved strobe_checklist.csv ({len(strobe_df)} items)")
```

---

## Advanced Usage

### Deviation Log

```python
import pandas as pd
import datetime

def create_deviation_log(preregistration_id: str) -> pd.DataFrame:
    """Initialize a structured deviation log for a preregistered study."""
    columns = [
        "deviation_id",
        "date_noted",
        "preregistered_plan",
        "actual_action",
        "reason",
        "impact_on_inference",
        "prespecified_contingency",
        "logged_by"
    ]
    return pd.DataFrame(columns=columns)

def log_deviation(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Add a deviation entry."""
    entry = {"date_noted": datetime.date.today().isoformat(), **kwargs}
    return pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

# Example usage
log = create_deviation_log("osf.io/abc12")

log = log_deviation(
    log,
    deviation_id="D001",
    preregistered_plan="Primary analysis: two-sided independent t-test",
    actual_action="Switched to Welch t-test due to unequal variances (Levene p=.02)",
    reason="Levene's test indicated heteroscedasticity; Welch is more robust",
    impact_on_inference="Minimal: df reduced from 98 to 94, conclusion unchanged",
    prespecified_contingency="Yes — protocol stated 'Welch if Levene p<.05'",
    logged_by="Jane Doe"
)

log = log_deviation(
    log,
    deviation_id="D002",
    preregistered_plan="Secondary analysis includes SWLS at T2",
    actual_action="SWLS data not collected at T2 due to survey software error",
    reason="Platform error deleted SWLS items for 12% of T2 responses; excluded as planned",
    impact_on_inference="Secondary outcome missing; primary unaffected",
    prespecified_contingency="No — unplanned deviation",
    logged_by="John Smith"
)

print("\nDeviation Log:")
print(log.to_string())
log.to_csv("deviation_log.csv", index=False)
```

### Registered Report Stage 1/Stage 2 Template

```python
REGISTERED_REPORT_OUTLINE = """
# Registered Report — Stage 1 Submission

## 1. Abstract (250 words max)

## 2. Introduction
- Theoretical background (~2 pages)
- Critical test: primary hypothesis with clear falsification criteria

## 3. Methods
### 3.1 Participants
- Target sample, eligibility, recruitment
- Power analysis with justification

### 3.2 Design
- Between/within subjects, counterbalancing

### 3.3 Stimuli and Procedure
- Sufficient detail for exact replication

### 3.4 Measures
- Primary DV with reliability estimate
- Secondary DVs

### 3.5 Analysis Plan
- Exact statistical test, software, alpha level
- Decision rule for H1 vs H0
- Planned exploratory analyses (clearly labeled)

### 3.6 Exclusion Criteria
- Data quality checks, outlier rules

## 4. Timeline

## References

---
# Stage 2 Addition (post-data-collection):

## 5. Results
- Follow Stage 1 analysis plan exactly
- Label any exploratory deviations with [EXPLORATORY] tag

## 6. Discussion

## 7. Deviation Log
[Attach deviation_log.csv]
"""
print(REGISTERED_REPORT_OUTLINE)
with open("registered_report_outline.md", "w") as f:
    f.write(REGISTERED_REPORT_OUTLINE)
print("Saved registered_report_outline.md")
```

---

## Troubleshooting

### Error: `401 Unauthorized` from OSF API

**Cause**: Invalid or expired OSF token.

**Fix**:
```bash
# Re-generate token at https://osf.io/settings/tokens/
# Ensure token has osf.full_write scope
export OSF_TOKEN="<your-new-token>"
```

### Error: `403 Forbidden` when creating registration

**Cause**: Registration endpoint requires project to have specific node settings.

**Fix**: Create the registration via the OSF web interface for initial setup; use API for file uploads.

### Checklist CSV appears truncated

**Cause**: Description column contains commas.

**Fix**:
```python
df.to_csv("file.csv", index=False, quoting=1)  # csv.QUOTE_ALL
```

### Version Compatibility

| Package | Tested versions | Notes |
|---|---|---|
| requests | 2.28–2.31 | stable API |
| jinja2 | 3.0–3.1 | Template syntax unchanged |
| pandas | 1.5–2.1 | DataFrame API stable |

---

## External Resources

### Official Documentation

- [OSF REST API v2](https://developer.osf.io/)
- [AsPredicted.org](https://aspredicted.org/)
- [ClinicalTrials.gov Protocol Registration](https://clinicaltrials.gov/ct2/manage-recs)
- [PROSPERO for Systematic Reviews](https://www.crd.york.ac.uk/prospero/)

### Reporting Guidelines

- [CONSORT 2010 Statement](https://www.consort-statement.org/)
- [STROBE Statement](https://www.strobe-statement.org/)
- [PRISMA 2020](http://www.prisma-statement.org/)
- [Equator Network](https://www.equator-network.org/) — comprehensive reporting guideline database

### Key Papers

- Nosek, B.A., et al. (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606.
- Chambers, C.D. (2013). Registered Reports: A new publishing initiative at Cortex. *Cortex*, 49(3), 609–610.

---

## Examples

### Example 1: RCT of Digital Intervention on Sleep Quality

**Scenario**: Randomized trial of a CBT-I app vs. sleep hygiene control, N=120.

```python
import pandas as pd
import datetime

# Generate preregistration document and checklists
prereg = generate_aspredicted(
    title="CBT-I App for Insomnia: RCT",
    authors="Research Team",
    hypothesis=(
        "Participants in the CBT-I app condition will show significantly lower "
        "ISI scores at 8-week follow-up vs. sleep hygiene control (d=0.6 expected)."
    ),
    dv="Insomnia Severity Index (ISI, 0-28) at 8 weeks (primary); PSQI, sleep diary (secondary)",
    iv="CBT-I app (n=60) vs. Sleep hygiene booklet control (n=60); random block allocation",
    analyses=(
        "ANCOVA: ISI at 8wk ~ condition + ISI baseline. Two-sided α=.05. "
        "Cohen's d from adjusted means. Multilevel model for weekly diary data."
    ),
    exclusions=(
        "ISI < 8 at screening (not clinically insomnic). "
        ">3 consecutive missed diary entries. Outliers: >3SD winsorized."
    ),
    n="120 (60/arm). Power: d=0.6, α=.05, 1−β=.80, ANCOVA R²_cov=.30 → N=86; +40% dropout buffer"
)

# Save and print summary
with open("sleep_rct_prereg.md", "w", encoding="utf-8") as f:
    f.write(prereg)

consort = generate_consort_checklist()
consort.to_csv("sleep_rct_consort.csv", index=False)

print("Preregistration materials saved:")
print("  - sleep_rct_prereg.md")
print("  - sleep_rct_consort.csv")
print(f"  - Registration date: {datetime.date.today()}")
```

### Example 2: Observational Cohort Study with STROBE

**Scenario**: Prospective cohort study of air pollution exposure and lung function.

```python
# Generate STROBE checklist and deviation log
strobe_items = [
    ("1", "Title", "Study type indicated in title: 'Prospective cohort'", "[ ]"),
    ("2", "Background", "Air pollution mechanism and prior evidence cited", "[ ]"),
    ("5", "Setting", "City, country, years 2020-2024, clinic network", "[ ]"),
    ("6", "Participants", "Adults 18-65, non-smokers, no prior COPD", "[ ]"),
    ("7", "Variables", "PM2.5 exposure (daily avg), FEV1, covariates (age, BMI, etc.)", "[ ]"),
    ("9", "Bias", "Selection: clinic vs population; measurement: monitor proximity", "[ ]"),
    ("12", "Statistics", "Linear mixed models with spatial autocorrelation correction", "[ ]"),
]

df = pd.DataFrame(strobe_items, columns=["Item", "Section", "Status_in_paper", "Reported"])
print("STROBE self-assessment:")
print(df.to_string(index=False))

# Initialize deviation log
log = create_deviation_log("osf.io/cohort123")
log = log_deviation(
    log,
    deviation_id="D001",
    preregistered_plan="Exclude participants with <180 days follow-up",
    actual_action="Threshold lowered to <90 days due to lower retention than expected",
    reason="Only 43% had 180+ days; 90-day threshold gives N=312 (vs N=156)",
    impact_on_inference="Sensitivity analysis with original threshold reported in supplement",
    prespecified_contingency="No",
    logged_by="Study team"
)
log.to_csv("cohort_deviation_log.csv", index=False)
print("\nDeviation log saved: cohort_deviation_log.csv")
```

---

*Last updated: 2026-03-18 | Maintainer: @xjtulyc*
*Issues: [GitHub Issues](https://github.com/xjtulyc/awesome-rosetta-skills/issues)*
