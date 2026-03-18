# Changelog

<!-- markdownlint-configure-file {"MD024": {"siblings_only": true}} -->

All notable changes to this project are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning: [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [1.0.0] - 2026-03-18

### Added

**169 skills total** — all 169 SKILL.md files pass CI validation (`scripts/validate_skill.py`, 0 failures).

This release completes the full skill expansion plan across 24 domains:

#### Wave 3 — New Skills (this release)

##### 08 Finance Academic

- `financial-time-series` — GARCH, cointegration, pairs trading, Fama-French, VaR/ES
- `wrds-data-access` — WRDS Python API, Compustat/CRSP, CCM linking
- `fixed-income` — bond pricing, YTM, duration, Nelson-Siegel, CDS

##### 10 Sociology

- `inequality-analysis` — Gini, Theil, FGT poverty, UQR gender gap, Moran's I
- `agent-based-social` — Mesa: Schelling, Deffuant opinion dynamics, SIR epidemic
- `qualitative-digital` — NVivo-style coding, HDBSCAN semantic clustering, kappa IRR

##### 12 Linguistics

- `phonetics-praat` — parselmouth pitch/formant, VOT detection, vowel space
- `typology-wals` — Greenberg universals, Yule's Q, Moran's I typological signal
- `historical-linguistics` — Levenshtein cognate detection, UPGMA dendrogram, NW alignment
- `computational-pragmatics` — zero-shot dialog acts, politeness features, implicature

##### 14 Philosophy

- `ethics-ai-analysis` — fairlearn MetricFrame, Hardt threshold adjustment, fairness-accuracy tradeoff

##### 15 Archaeology

- `archaeological-gis` — KDTree R-statistic, HDBSCAN hotspot, Voronoi, RF predictive model
- `artifact-analysis` — diversity indices, correspondence analysis, Harris matrix
- `photogrammetry-3d` — Open3D point cloud, Poisson reconstruction, ICP registration

##### 17 Public Health

- `systematic-review-epi` — DerSimonian-Laird, forest plot, Egger's test, meta-regression

##### 18 Urban Science

- `urban-remote-sensing` — Sentinel-2 NDVI/NDBI/NDWI, RF LULC, UHI analysis
- `mobility-trajectories` — GPS stop detection, OD matrix, radius of gyration, entropy
- `regional-economics` — shift-share, LQ, Leontief IO, Krugman index, beta-convergence
- `housing-market` — hedonic regression, repeat-sales index, affordability, spatial lag model

##### 20 Education

- `learning-experiment` — IRT 2PL/3PL, power law of learning, Ebbinghaus forgetting, BKT, DIF

##### 21 Library Science

- `openalx-bibliometrics` — OpenAlex API, h-index, g-index, Bradford's Law, co-authorship network
- `knowledge-graph-sparql` — rdflib RDF, SPARQL, Wikidata, OWL reasoning, entity reconciliation
- `patent-analysis` — IPC landscape, citation network TCT, Bass S-curve, inventor network
- `research-impact` — FWCI, MNCS, PP-top10%, JIF, altmetrics, Leiden indicators

##### 22 Interdisciplinary

- `science-of-science` — CD disruption index, team size effects, sleeping beauty, knowledge flow
- `text-mining-science` — LDA/NMF topic modeling, RAKE keywords, scientific NER, trend detection
- `mixed-methods-research` — content analysis, kappa/alpha IRR, QCA/fsQCA, triangulation

##### 23 Research Workflow

- `python-package-dev` — pyproject.toml src layout, pytest, Sphinx autodoc, PyPI publishing
- `quarto-reporting` — parameterized reports, multi-format output, journal templates, automation
- `r-package-dev` — devtools/roxygen2/testthat, S3 classes, rpy2 bridge, CRAN check
- `containerization-research` — Dockerfile multi-stage, Docker Compose, Singularity HPC, Docker SDK
- `data-version-control` — DVC pipeline, experiment tracking, remote storage, MLflow integration

### Fixed

- Removed hardcoded credential patterns in 4 SKILL.md files (S001 rule compliance)

---

## [0.2.0] - 2026-03-17

### Added

43 Skills across 24 disciplines — all pass CI validation (`scripts/validate_skill.py`).

#### 00 Universal Research (6 skills)

- `literature-search` — cross-database search: OpenAlex, Semantic Scholar, arXiv, BibTeX
- `statistical-testing` — hypothesis testing: t-tests, ANOVA, non-parametric, effect sizes, FDR
- `experimental-design` — sample size, power analysis, randomization, pre-registration
- `data-visualization` — publication-quality figures: matplotlib/seaborn and ggplot2
- `scientometrics` — bibliometrics: co-authorship networks, h-index, research fronts
- `rebuttal-writing` — peer-review rebuttal: point-by-point format, tone calibration, LaTeX

#### 01 Physics (2)

- `scipy-numerical` — ODE/PDE solving, FFT, optimization, sparse linear algebra
- `sympy-symbolic` — symbolic calculus, mechanics, quantum physics, code generation

#### 02 Chemistry (1)

- `ase-atomistic` — ASE: structure building, geometry optimization, NEB, molecular dynamics

#### 03 Mathematics & Statistics (2)

- `bayesian-stats` — PyMC 5.x, NUTS, diagnostics, LOO-CV, hierarchical models
- `causal-inference` — DoWhy: DAGs, backdoor criterion, propensity score matching

#### 04 Earth & Environmental Science (2)

- `era5-climate` — ERA5 reanalysis: CDS API, xarray, climate anomalies, cartopy maps
- `geopandas-gis` — vector GIS: spatial joins, overlay, choropleth, raster integration

#### 05 Neuroscience (2)

- `mne-eeg` — EEG/MEG: preprocessing, ICA, ERP, time-frequency analysis
- `nilearn-fmri` — fMRI GLM, resting-state connectivity, MVPA decoding

#### 06 Engineering (1)

- `signal-processing` — DSP: filter design, spectrogram, Welch PSD, peak detection

#### 07 Economics (6)

- `ols-regression` — OLS: heteroscedasticity tests, robust SE, regression tables
- `did-causal` — DID: TWFE, parallel trends, Callaway-Sant'Anna, Goodman-Bacon
- `rdd-design` — RDD: rdrobust, bandwidth selection, McCrary test, RD plots
- `iv-2sls` — IV/2SLS: first-stage F-stat, Wu-Hausman, Sargan-Hansen overid test
- `fred-macro` — FRED API: GDP, unemployment, CPI, HP filter, recession shading
- `panel-data` — panel econometrics: FE/RE, Hausman, Arellano-Bond GMM, unit root

#### 08 Finance Academic (2)

- `factor-models` — Fama-French 3/5-factor, alpha, rolling loadings, GRS test
- `event-study` — abnormal returns, CAR/BHAR, BMP test, long-run performance

#### 09 Political Science (2)

- `vdem-analysis` — V-Dem democracy indices, panel regression, backsliding detection
- `text-as-data` — Wordfish scaling, LDA topics, VADER sentiment, ideology classification

#### 10 Sociology (2)

- `social-network-analysis` — NetworkX: centrality, community detection, Gephi export
- `computational-sociology` — social media APIs, bot detection, echo chambers

#### 11 Psychology (2)

- `power-analysis` — statistical power: t-test, ANOVA, regression, mediation simulation
- `psychometrics` — CTT, EFA/CFA, IRT 2PL, measurement invariance, DIF

#### 12 Linguistics (1)

- `corpus-linguistics` — frequency, MI/log-likelihood collocations, KWIC concordance

#### 13 History (1)

- `digital-archives` — Europeana, Chronicling America, Internet Archive APIs

#### 14 Philosophy (1)

- `sep-literature` — SEP scraping, PhilPapers API, concept genealogy tracing

#### 15 Archaeology (1)

- `radiocarbon-dating` — 14C calibration: IntCal20, Bayesian sequence modeling

#### 16 Art & Musicology (1)

- `librosa-audio` — MIR: tempo, chroma, MFCCs, onset detection, music similarity

#### 17 Public Health (2)

- `epi-modeling` — SEIR/SIR modeling, Rt estimation, parameter fitting
- `global-health-data` — WHO/IHME: DALYs, age-standardization, health inequality

#### 18 Urban Science (1)

- `osmnx-urban` — OSMnx: walkability, centrality, isochrones, city comparisons

#### 19 Agriculture (1)

- `soil-data` — SoilGrids API, SOC stocks, texture classification, profile visualization

#### 20 Education (1)

- `edm-learning-analytics` — BKT knowledge tracing, dropout prediction, learning curves

#### 21 Library Science (1)

- `topic-modeling-lit` — LDA + BERTopic on abstracts, coherence optimization, temporal trends

#### 22 Interdisciplinary (1)

- `complexity-science` — power laws, Hurst exponent, fractal dimension, ABM

#### 23 Research Workflow (1)

- `latex-workflow` — LaTeX packages, Makefile, bibliography, arXiv submission

#### Infrastructure

- `scripts/generate_index.py` — auto-regenerates README skill index from SKILL.md frontmatter
- `scripts/install.sh` — install skills to Claude Code / Codex / Gemini CLI / Cursor
- `scripts/check_compat.py` — validate platform declarations in frontmatter
- `templates/WORKFLOW_TEMPLATE.md` — workflow skill template with pipeline structure
- `.gitmodules` — git submodule references to Orchestra-Research and K-Dense-AI repos
- `README.md` — full discipline index with 43-skill table, badges, platform reference

---

## [0.1.0] - 2026-03-17

### Added — Repository Foundation

- Repository structure initialization
- `README.md`, `CONTRIBUTING.md`, `SKILL_STANDARD.md`
- `templates/SKILL_TEMPLATE.md`
- `scripts/validate_skill.py` — format validation (F/C/S rule series)
- `.github/workflows/validate-skills.yml` — CI auto-validation
- `.github/ISSUE_TEMPLATE/` — new-skill and skill-update templates
- `.github/PULL_REQUEST_TEMPLATE.md`
- 24 discipline directories (`skills/00-universal/` through `skills/23-research-workflow/`)
- `LICENSE` (MIT)
- `.gitignore` (excludes `research-skills-prd.md` and other private files)

---

[Unreleased]: https://github.com/xjtulyc/awesome-rosetta-skills/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xjtulyc/awesome-rosetta-skills/releases/tag/v0.1.0
