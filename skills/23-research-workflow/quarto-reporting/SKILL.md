---
name: quarto-reporting
description: Reproducible research reporting with Quarto covering parameterized reports, multi-format output, inline computation, and journal article templates.
tags:
  - quarto
  - reproducible-research
  - literate-programming
  - scientific-reporting
  - rmarkdown
version: "1.0.0"
authors:
  - "@xjtulyc"
license: MIT
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - jupyter>=1.0
    - nbformat>=5.9
    - pandas>=2.0
    - numpy>=1.24
    - matplotlib>=3.7
    - scipy>=1.11
last_updated: "2026-03-17"
status: stable
---

# Quarto Reproducible Research Reporting

## When to Use This Skill

Use this skill when you need to:
- Create reproducible research reports combining code, results, and narrative
- Generate multiple output formats (HTML, PDF, DOCX, reveal.js slides) from one source
- Write parameterized reports that can be re-run with different datasets
- Create journal-ready manuscripts with embedded analysis
- Build research websites and books with computed figures and tables
- Use Python or R (or both) within the same document
- Automate report generation as part of a data pipeline

**Trigger keywords**: Quarto, reproducible report, literate programming, parameterized report, RMarkdown, Jupyter, HTML report, PDF report, DOCX, reveal.js, manuscript, cross-references, figure captions, bibliography, BibTeX, research website, book, blog, academic writing, inline computation, code chunks.

## Background & Key Concepts

### Quarto Architecture

Quarto is a multi-language scientific publishing system:

1. `.qmd` source file → Quarto engine → Jupyter/knitr → Pandoc → Output
2. **Execution engines**: `jupyter` (Python/Julia/Observable) or `knitr` (R/Python via reticulate)
3. **Output formats**: `html`, `pdf` (via LaTeX), `docx`, `revealjs`, `beamer`, `typst`

### YAML Front Matter

```yaml
---
title: "Analysis Report"
author: "Researcher"
date: today
format:
  html:
    toc: true
    code-fold: true
  pdf:
    documentclass: article
    geometry: margin=1in
bibliography: references.bib
execute:
  echo: false
  warning: false
params:
  dataset: "data/results.csv"
---
```

### Cross-References

- Figures: `@fig-scatter` (label: `#| label: fig-scatter`)
- Tables: `@tbl-summary` (label: `#| label: tbl-summary`)
- Equations: `@eq-model` (label: `$$\text{...} \quad\quad(1)$$ {#eq-model}`)
- Sections: `@sec-methods`

### Parameterized Reports

Pass parameters at render time:

```bash
quarto render report.qmd -P dataset:data/2023.csv -P alpha:0.05
```

Access in code: `params$dataset` (R) or `params["dataset"]` (Python via YAML params).

## Environment Setup

```bash
# Install Quarto CLI
# macOS: brew install quarto
# Linux: wget https://github.com/quarto-dev/quarto-cli/releases/latest/download/quarto-linux-amd64.deb
# Windows: winget install Posit.Quarto

pip install jupyter>=1.0 nbformat>=5.9 pandas>=2.0 numpy>=1.24 \
            matplotlib>=3.7 scipy>=1.11

# Verify installation
quarto check
```

```bash
# Create a new Quarto document
quarto create-project my-report --type document
```

## Core Workflow

### Step 1: Basic Quarto Document Structure

The following shows the structure of a complete Quarto `.qmd` file. Save as `analysis_report.qmd`:

````markdown
---
title: "Regression Analysis Report"
subtitle: "Quantitative Research Summary"
author:
  - name: "Research Team"
    affiliation: "University Department"
date: today
abstract: |
  This report presents a regression analysis of the simulated dataset.
  Key findings include a statistically significant positive relationship
  between the predictor and outcome variables.
format:
  html:
    toc: true
    toc-depth: 3
    code-fold: true
    number-sections: true
    theme: cosmo
    fig-width: 8
    fig-height: 5
  pdf:
    documentclass: article
    geometry: "margin=1in"
    number-sections: true
execute:
  echo: true
  warning: false
  message: false
jupyter: python3
---

# Introduction {#sec-intro}

This report analyzes a synthetic dataset to demonstrate reproducible reporting
with Quarto. See @sec-methods for the analytical approach.

# Methods {#sec-methods}

## Data Generation

```{python}
#| label: setup
#| include: false
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from IPython.display import display
np.random.seed(42)
```

We simulated $n = 200$ observations from a linear model
$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$ where
$\varepsilon \sim \mathcal{N}(0, \sigma^2)$.

```{python}
#| label: data-generation
n = 200
x1 = np.random.normal(5, 2, n)
x2 = np.random.binomial(1, 0.5, n)
y  = 3.0 + 1.5 * x1 + 2.0 * x2 + np.random.normal(0, 3, n)
df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
print(f"Dataset: {len(df)} observations, {df.shape[1]} variables")
```

## Analytical Approach

We fit an OLS regression model and report heteroskedasticity-robust
standard errors (HC3).

# Results {#sec-results}

## Descriptive Statistics

```{python}
#| label: tbl-descriptive
#| tbl-cap: "Descriptive statistics for all variables."
desc = df.describe().round(2)
display(desc)
```

@tbl-descriptive presents the descriptive statistics.

## Regression Results

```{python}
#| label: regression-model
X = sm.add_constant(df[["x1", "x2"]])
model = sm.OLS(df["y"], X).fit(cov_type="HC3")
```

The regression results are shown in @tbl-regression.

```{python}
#| label: tbl-regression
#| tbl-cap: "OLS regression results with HC3 standard errors."
results_df = pd.DataFrame({
    "Coefficient": model.params,
    "Std. Error": model.bse,
    "t-stat": model.tvalues,
    "p-value": model.pvalues,
    "95% CI Lower": model.conf_int()[0],
    "95% CI Upper": model.conf_int()[1],
}).round(3)
display(results_df)
```

The model explains `{python} f"{model.rsquared_adj:.3f}"` of the variance
in the outcome ($\bar{R}^2$).

## Visualization

```{python}
#| label: fig-scatter
#| fig-cap: "Scatter plot of X1 vs. Y by group X2."
#| fig-width: 6
#| fig-height: 4
fig, ax = plt.subplots(figsize=(6, 4))
for x2_val, color, label in [(0, "steelblue", "X2=0"), (1, "orange", "X2=1")]:
    mask = df["x2"] == x2_val
    ax.scatter(df.loc[mask, "x1"], df.loc[mask, "y"],
               color=color, alpha=0.5, s=20, label=label)
# Fitted lines
xf = np.linspace(df["x1"].min(), df["x1"].max(), 100)
for x2_val, color in [(0, "steelblue"), (1, "orange")]:
    yf = model.params["const"] + model.params["x1"] * xf + model.params["x2"] * x2_val
    ax.plot(xf, yf, color=color, lw=2)
ax.set_xlabel("X1"); ax.set_ylabel("Y")
ax.set_title("Fitted Regression Lines by Group")
ax.legend()
plt.tight_layout()
plt.show()
```

@fig-scatter shows the relationship between X1 and Y, stratified by X2.

# Discussion {#sec-discussion}

The coefficient for X1 ($\hat{\beta}_1 = `{python} f"{model.params['x1']:.3f}"`$,
$p `{python} "< 0.001" if model.pvalues['x1'] < 0.001 else f"= {model.pvalues['x1']:.3f}"`$)
indicates a positive relationship.

# Conclusion {#sec-conclusion}

This analysis demonstrates Quarto's capabilities for reproducible reporting.
All figures and tables were generated from the same code that produced
the written results.

# References {.unnumbered}

::: {#refs}
:::
````

```bash
# Render to HTML
quarto render analysis_report.qmd --to html

# Render to PDF
quarto render analysis_report.qmd --to pdf

# Render all formats defined in YAML
quarto render analysis_report.qmd
```

### Step 2: Parameterized Report Generation

```python
# File: generate_reports.py
"""Automate parameterized Quarto report generation."""

import subprocess
import os
import json
from pathlib import Path


def render_parameterized_report(
    qmd_file,
    output_dir,
    params,
    output_format="html",
):
    """Render a Quarto report with specific parameter values.

    Args:
        qmd_file: Path to .qmd source file
        output_dir: Directory to save output
        params: dict of parameter name → value
        output_format: output format string

    Returns:
        Path to rendered output file, or None on failure
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build parameter flags
    param_flags = []
    for key, value in params.items():
        param_flags.extend(["-P", f"{key}:{value}"])

    # Output file name
    output_file = Path(output_dir) / f"report_{params.get('dataset', 'default')}.{output_format}"

    cmd = [
        "quarto", "render", qmd_file,
        "--to", output_format,
        "--output", str(output_file),
    ] + param_flags

    print(f"Rendering: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Success: {output_file}")
        return output_file
    else:
        print(f"Error rendering {qmd_file}:")
        print(result.stderr)
        return None


# Example: Generate reports for multiple datasets
datasets = ["2021", "2022", "2023"]
for year in datasets:
    render_parameterized_report(
        qmd_file="analysis_report.qmd",
        output_dir="reports/",
        params={"year": year, "alpha": "0.05"},
        output_format="html",
    )

# Example: Batch generation with different parameter combinations
from itertools import product

alphas = [0.01, 0.05]
methods = ["ols", "wls"]
for alpha, method in product(alphas, methods):
    report = render_parameterized_report(
        "sensitivity_analysis.qmd",
        "reports/sensitivity/",
        {"alpha": str(alpha), "method": method},
    )
```

### Step 3: Quarto Book and Website Setup

```bash
# Create a research book project
quarto create-project my-research-book --type book
```

```yaml
# File: _quarto.yml (book configuration)
project:
  type: book
  output-dir: _book

book:
  title: "Research Methods Handbook"
  author: "Research Group"
  date: today
  chapters:
    - index.qmd
    - chapters/01-introduction.qmd
    - chapters/02-methods.qmd
    - chapters/03-results.qmd
    - chapters/04-discussion.qmd
    - references.qmd

bibliography: references.bib
csl: apa.csl

format:
  html:
    theme: cosmo
    cover-image: cover.png
    toc: true
    toc-depth: 3
    number-sections: true
    code-fold: true
  pdf:
    documentclass: scrbook
    geometry: "margin=1in"
    number-sections: true
```

```yaml
# File: _quarto.yml (website configuration)
project:
  type: website
  output-dir: _site

website:
  title: "Research Lab"
  navbar:
    left:
      - href: index.qmd
        text: Home
      - text: Projects
        menu:
          - href: projects/project1.qmd
          - href: projects/project2.qmd
      - href: publications.qmd
        text: Publications
      - href: team.qmd
        text: Team

format:
  html:
    theme: [cosmo, custom.scss]
    code-fold: true
    toc: true
```

```python
# File: chapters/02-methods.qmd content generator
# This shows what a methods chapter might contain

methods_qmd = """---
title: "Statistical Methods"
---

# Statistical Methods {{#sec-stats-methods}}

## Power Analysis

```{{python}}
#| label: tbl-power
#| tbl-cap: "Minimum detectable effect sizes for varying sample sizes."
import numpy as np
import pandas as pd
from scipy.stats import norm

def power_analysis(n, alpha=0.05, power=0.80):
    \"\"\"Compute minimum detectable effect size given n, alpha, and power.\"\"\"
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta  = norm.ppf(power)
    return (z_alpha + z_beta) / np.sqrt(n)

sample_sizes = [20, 30, 50, 100, 200, 500]
mde_table = pd.DataFrame({
    "n": sample_sizes,
    "MDE (α=0.05, 80% power)": [power_analysis(n) for n in sample_sizes],
    "MDE (α=0.01, 80% power)": [power_analysis(n, alpha=0.01) for n in sample_sizes],
})
display(mde_table.round(3))
```
\"\"\"

print("Methods chapter content created")
print(methods_qmd[:200] + "...")
```

## Advanced Usage

### Custom Quarto Extensions and Templates

```python
# File: scripts/setup_journal_template.py
"""Set up a journal article Quarto template."""

import os

JOURNAL_TEMPLATE = """---
title: "Manuscript Title"
format:
  pdf:
    template: apa.tex
    keep-tex: false
    number-sections: false
    cite-method: biblatex
author:
  - name: First Author
    affiliations:
      - name: University Department
        city: City
        country: Country
    email: author@university.edu
    corresponding: true
  - name: Second Author
    affiliations:
      - name: Research Institute
abstract: |
  Background: ...
  Methods: ...
  Results: ...
  Conclusions: ...
keywords: [keyword1, keyword2, keyword3]
bibliography: references.bib
csl: nature.csl
---

# Introduction

# Methods

# Results

# Discussion

# Conclusion

# Acknowledgments

# References

::: {{#refs}}
:::
"""

def create_manuscript_template(output_path="manuscript.qmd"):
    with open(output_path, "w") as f:
        f.write(JOURNAL_TEMPLATE)
    print(f"Template created: {output_path}")
    return output_path

create_manuscript_template()
```

### Python + R Mixed Document

````markdown
---
title: "Multi-Language Analysis"
format: html
execute:
  warning: false
---

## Python Analysis

```{{python}}
import numpy as np
import pandas as pd

# Compute in Python
data = pd.DataFrame({"x": np.random.normal(0, 1, 100),
                     "y": np.random.normal(0, 1, 100)})
data.to_csv("shared_data.csv", index=False)
print("Data saved to CSV")
```

## R Visualization

```{{r}}
library(ggplot2)
data <- read.csv("shared_data.csv")
ggplot(data, aes(x=x, y=y)) +
  geom_point(alpha=0.5) +
  geom_smooth(method="lm") +
  theme_minimal() +
  labs(title="Correlation Plot (R ggplot2)")
```
````

### Automated Report Pipeline

```python
# File: pipeline.py
"""Pipeline to generate weekly research reports."""

import subprocess
import datetime
import os

def weekly_report_pipeline(data_path, output_dir="reports"):
    """Full pipeline: update data → render report → distribute.

    Args:
        data_path: path to latest data file
        output_dir: output directory for reports
    """
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.date.today().isoformat()
    output_file = f"{output_dir}/weekly_report_{today}.html"

    # Render parameterized report
    cmd = [
        "quarto", "render", "weekly_template.qmd",
        "--to", "html",
        "--output", output_file,
        "-P", f"data_path:{data_path}",
        "-P", f"report_date:{today}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quarto render failed:\n{result.stderr}")

    print(f"Report generated: {output_file}")
    return output_file

# Example (dry run — Quarto not actually invoked)
print("Pipeline configured for weekly report generation")
print(f"Next report: {datetime.date.today().isoformat()}")
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `quarto: command not found` | Quarto not installed or not in PATH | Install from quarto.org; add to PATH |
| PDF rendering fails | Missing LaTeX distribution | Install TinyTeX: `quarto install tool tinytex` |
| Python kernel not found | Wrong kernel name | Use `quarto check jupyter`; specify `jupyter: python3` in YAML |
| Figures missing in PDF | matplotlib `plt.show()` called wrong | Use `plt.savefig()` + display, or remove `plt.show()` |
| Cross-references show `?` | Label format incorrect | Ensure `#| label: fig-name` or `#| label: tbl-name` syntax |
| Params not accessible | Missing params block in YAML | Add `params:` section to YAML front matter |

## External Resources

- [Quarto documentation](https://quarto.org/docs/)
- [Quarto Gallery](https://quarto.org/docs/gallery/) — example reports, books, websites
- [Quarto journal templates](https://github.com/quarto-journals/)
- Xie, Y., Allaire, J. J., & Grolemund, G. (2019). *R Markdown: The Definitive Guide*. CRC Press.
- [Pandoc user guide](https://pandoc.org/MANUAL.html)

## Examples

### Example 1: Automated Results Section Generation

```python
# Generate a results section text from statistical output
import numpy as np
import pandas as pd
import statsmodels.api as sm

def format_p_value(p):
    """Format p-value for publication."""
    if p < 0.001: return "< .001"
    elif p < 0.01: return f"= {p:.3f}"
    else: return f"= {p:.3f}"

def regression_results_text(model, predictor):
    """Generate APA-style regression results text."""
    coeff = model.params[predictor]
    se    = model.bse[predictor]
    t     = model.tvalues[predictor]
    p     = model.pvalues[predictor]
    ci_lo, ci_hi = model.conf_int().loc[predictor]

    return (
        f"The predictor {predictor} was "
        f"{'significantly' if p < 0.05 else 'not significantly'} "
        f"associated with the outcome, "
        f"b = {coeff:.2f}, SE = {se:.2f}, "
        f"t({int(model.df_resid)}) = {t:.2f}, p {format_p_value(p)}, "
        f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]."
    )

np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = 2.5 * x + np.random.normal(0, 1, 100)
model = sm.OLS(y, sm.add_constant(pd.DataFrame({"x": x}))).fit()

print("Generated results text:")
print(regression_results_text(model, "x"))
```

### Example 2: Table Formatting for Publication

```python
import pandas as pd
import numpy as np

def format_regression_table(models_dict, variable_labels=None):
    """Create publication-ready regression comparison table.

    Args:
        models_dict: dict of {model_name: statsmodels result}
        variable_labels: dict of {variable: display_name}
    Returns:
        DataFrame suitable for display in Quarto
    """
    if variable_labels is None:
        variable_labels = {}

    all_vars = set()
    for model in models_dict.values():
        all_vars.update(model.params.index)
    all_vars = sorted(all_vars)

    rows = []
    for var in all_vars:
        row = {"Variable": variable_labels.get(var, var)}
        for model_name, model in models_dict.items():
            if var in model.params:
                coef = model.params[var]
                se = model.bse[var]
                p = model.pvalues[var]
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                row[model_name] = f"{coef:.3f}{stars}\n({se:.3f})"
            else:
                row[model_name] = "—"
        rows.append(row)

    df = pd.DataFrame(rows)
    # Add fit statistics
    for stat_name, stat_fn in [("R²", lambda m: f"{m.rsquared:.3f}"),
                                 ("Adj. R²", lambda m: f"{m.rsquared_adj:.3f}"),
                                 ("N", lambda m: str(int(m.nobs)))]:
        row = {"Variable": stat_name}
        for model_name, model in models_dict.items():
            row[model_name] = stat_fn(model)
        rows.append(row)

    return pd.DataFrame(rows)

# Demo
np.random.seed(42)
n = 200
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
y  = 1.5 * x1 + 0.8 * x2 + np.random.normal(0, 1, n)

import statsmodels.api as sm
m1 = sm.OLS(y, sm.add_constant(pd.DataFrame({"x1": x1}))).fit()
m2 = sm.OLS(y, sm.add_constant(pd.DataFrame({"x1": x1, "x2": x2}))).fit()

table = format_regression_table({"Model 1": m1, "Model 2": m2},
                                  {"const": "Intercept", "x1": "Predictor 1", "x2": "Predictor 2"})
print("Publication-ready regression table:")
print(table.to_string(index=False))
```
