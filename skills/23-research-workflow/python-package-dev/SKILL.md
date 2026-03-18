---
name: python-package-dev
description: Python research package development with pyproject.toml, testing with pytest, documentation with Sphinx, and publishing to PyPI for academic software.
tags:
  - python-packaging
  - pyproject-toml
  - pytest
  - sphinx
  - scientific-software
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
    - build>=1.0
    - pytest>=7.4
    - pytest-cov>=4.1
    - sphinx>=7.2
    - numpy>=1.24
    - setuptools>=68
last_updated: "2026-03-17"
status: stable
---

# Python Research Package Development

## When to Use This Skill

Use this skill when you need to:
- Set up a modern Python package with `pyproject.toml` (PEP 517/518)
- Structure a research software repository with src layout
- Write unit and integration tests with pytest
- Generate API documentation with Sphinx and autodoc
- Publish a package to PyPI or a conda channel
- Set up continuous integration (GitHub Actions) for research software
- Create reproducible research packages with versioning

**Trigger keywords**: Python package, pyproject.toml, setuptools, build, wheel, pytest, coverage, Sphinx, autodoc, Read the Docs, PyPI, conda-forge, research software engineering, FAIR software, src layout, namespace package, entry points, semantic versioning.

## Background & Key Concepts

### Modern Python Packaging (PEP 517/518)

`pyproject.toml` is the single source of truth:
- `[build-system]`: specifies build backend (setuptools, flit, poetry)
- `[project]`: package metadata (name, version, dependencies)
- `[project.optional-dependencies]`: optional extras
- `[project.scripts]`: CLI entry points

### Semantic Versioning

`MAJOR.MINOR.PATCH` (e.g., `1.2.3`):
- MAJOR: breaking API changes
- MINOR: backward-compatible new features
- PATCH: backward-compatible bug fixes

Use `__version__ = "1.0.0"` in `src/mypackage/__init__.py`.

### Src Layout

Placing code in `src/mypackage/` prevents accidental imports from the project root and enforces installation before use:

```
mypackage/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── core.py
├── tests/
│   └── test_core.py
├── docs/
│   └── conf.py
└── pyproject.toml
```

### Test Coverage

Coverage measures the fraction of code lines executed during tests. Target ≥ 80% for research packages. Use `pytest-cov`:

```bash
pytest --cov=src/mypackage --cov-report=html
```

## Environment Setup

```bash
pip install build>=1.0 pytest>=7.4 pytest-cov>=4.1 sphinx>=7.2 \
            twine>=4.0 setuptools>=68
```

```bash
# Verify tools
python -m pytest --version
python -m build --version
sphinx-build --version
```

## Core Workflow

### Step 1: Package Structure and pyproject.toml

```bash
# Create package structure
mkdir -p my_research_package/src/my_research_package
mkdir -p my_research_package/tests
mkdir -p my_research_package/docs
cd my_research_package
```

```python
# File: src/my_research_package/__init__.py
"""My Research Package: Tools for quantitative analysis."""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@university.edu"

from .core import (
    DataProcessor,
    StatisticalAnalyzer,
    compute_effect_size,
)

__all__ = ["DataProcessor", "StatisticalAnalyzer", "compute_effect_size"]
```

```python
# File: src/my_research_package/core.py
"""Core analysis functions for my_research_package."""

import numpy as np
import pandas as pd
from typing import Optional, Union, List


class DataProcessor:
    """Preprocess and validate research datasets.

    Args:
        data: Input DataFrame or array-like
        missing_strategy: How to handle missing values.
            One of 'drop', 'mean', 'median', 'zero'.
    """

    VALID_STRATEGIES = ("drop", "mean", "median", "zero")

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        missing_strategy: str = "mean",
    ):
        if missing_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"missing_strategy must be one of {self.VALID_STRATEGIES}, "
                f"got '{missing_strategy}'"
            )
        self.missing_strategy = missing_strategy
        self._data = self._validate_input(data)

    def _validate_input(self, data):
        """Validate and convert input to DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame(data, columns=["value"])
            return pd.DataFrame(data)
        else:
            raise TypeError(f"Expected DataFrame or ndarray, got {type(data)}")

    def process(self) -> pd.DataFrame:
        """Apply missing value strategy and return cleaned data.

        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        df = self._data.copy()
        if self.missing_strategy == "drop":
            return df.dropna()
        elif self.missing_strategy == "mean":
            return df.fillna(df.mean(numeric_only=True))
        elif self.missing_strategy == "median":
            return df.fillna(df.median(numeric_only=True))
        elif self.missing_strategy == "zero":
            return df.fillna(0)

    @property
    def shape(self):
        """Return (rows, columns) of the dataset."""
        return self._data.shape

    @property
    def missing_count(self):
        """Return total number of missing values."""
        return int(self._data.isna().sum().sum())


class StatisticalAnalyzer:
    """Compute descriptive and inferential statistics.

    Args:
        data: Input array-like of values
    """

    def __init__(self, data: Union[np.ndarray, List[float], pd.Series]):
        self.data = np.asarray(data, dtype=float)

    def describe(self) -> dict:
        """Return descriptive statistics.

        Returns:
            dict: n, mean, std, min, q25, median, q75, max
        """
        d = self.data[~np.isnan(self.data)]
        if len(d) == 0:
            return {}
        return {
            "n": len(d),
            "mean": float(np.mean(d)),
            "std": float(np.std(d, ddof=1)),
            "min": float(np.min(d)),
            "q25": float(np.percentile(d, 25)),
            "median": float(np.median(d)),
            "q75": float(np.percentile(d, 75)),
            "max": float(np.max(d)),
        }

    def confidence_interval(self, alpha: float = 0.05) -> tuple:
        """Compute (1-alpha)% confidence interval for the mean.

        Args:
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            (lower, upper) confidence interval tuple
        """
        from scipy import stats
        d = self.data[~np.isnan(self.data)]
        n = len(d)
        if n < 2:
            raise ValueError("Need at least 2 observations for CI")
        se = stats.sem(d)
        h = se * stats.t.ppf(1 - alpha / 2, df=n - 1)
        mean = np.mean(d)
        return (float(mean - h), float(mean + h))


def compute_effect_size(
    group1: Union[np.ndarray, List[float]],
    group2: Union[np.ndarray, List[float]],
    method: str = "cohens_d",
) -> float:
    """Compute effect size between two groups.

    Args:
        group1: Observations from first group
        group2: Observations from second group
        method: Effect size measure. One of 'cohens_d', 'glass_delta', 'hedges_g'

    Returns:
        Effect size value (positive = group1 > group2)

    Raises:
        ValueError: If method is not recognized
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    mean_diff = np.mean(g1) - np.mean(g2)
    n1, n2 = len(g1), len(g2)

    if method == "cohens_d":
        pooled_var = ((n1 - 1) * np.var(g1, ddof=1) +
                      (n2 - 1) * np.var(g2, ddof=1)) / (n1 + n2 - 2)
        return float(mean_diff / np.sqrt(pooled_var))
    elif method == "glass_delta":
        return float(mean_diff / np.std(g2, ddof=1))
    elif method == "hedges_g":
        d = compute_effect_size(g1, g2, method="cohens_d")
        j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
        return float(d * j)
    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Choose from: cohens_d, glass_delta, hedges_g")
```

```toml
# File: pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "my-research-package"
version = "0.1.0"
description = "Tools for quantitative research analysis"
readme = "README.md"
license = {text = "MIT"}
authors = [
  {name = "Research Team", email = "research@university.edu"},
]
keywords = ["research", "statistics", "data analysis"]
classifiers = [
  "Development Status :: 3 - Alpha",
  "Intended Audience :: Science/Research",
  "License :: OSI Approved :: MIT License",
  "Programming Language :: Python :: 3",
  "Programming Language :: Python :: 3.9",
  "Programming Language :: Python :: 3.10",
  "Programming Language :: Python :: 3.11",
]
requires-python = ">=3.9"
dependencies = [
  "numpy>=1.24",
  "pandas>=2.0",
  "scipy>=1.11",
]

[project.optional-dependencies]
viz = ["matplotlib>=3.7"]
dev = [
  "pytest>=7.4",
  "pytest-cov>=4.1",
  "sphinx>=7.2",
  "sphinx-rtd-theme>=1.3",
  "twine>=4.0",
]

[project.urls]
Homepage = "https://github.com/example/my-research-package"
Documentation = "https://my-research-package.readthedocs.io"
Repository = "https://github.com/example/my-research-package"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/my_research_package --cov-report=term-missing"

[tool.coverage.run]
source = ["src/my_research_package"]

[tool.coverage.report]
exclude_lines = [
  "pragma: no cover",
  "if __name__ == .__main__.:",
]
```

### Step 2: Testing with Pytest

```python
# File: tests/test_core.py
"""Tests for my_research_package core module."""

import pytest
import numpy as np
import pandas as pd

# Adjust import path if running tests standalone
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from my_research_package.core import DataProcessor, StatisticalAnalyzer, compute_effect_size


class TestDataProcessor:
    """Tests for DataProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clean_data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        self.missing_data = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]})

    def test_basic_initialization(self):
        """Test DataProcessor initializes with valid input."""
        dp = DataProcessor(self.clean_data)
        assert dp.shape == (3, 2)
        assert dp.missing_count == 0

    def test_missing_count(self):
        """Test missing count is correctly reported."""
        dp = DataProcessor(self.missing_data)
        assert dp.missing_count == 2

    def test_mean_imputation(self):
        """Test mean imputation strategy."""
        dp = DataProcessor(self.missing_data, missing_strategy="mean")
        result = dp.process()
        assert result.isna().sum().sum() == 0
        assert result.loc[1, "a"] == pytest.approx(2.0)  # mean of [1, 3]

    def test_drop_strategy(self):
        """Test drop missing strategy removes rows with NaN."""
        dp = DataProcessor(self.missing_data, missing_strategy="drop")
        result = dp.process()
        assert len(result) == 1  # only row index 2 has no NaN

    def test_invalid_strategy_raises(self):
        """Test ValueError on invalid strategy."""
        with pytest.raises(ValueError, match="missing_strategy must be one of"):
            DataProcessor(self.clean_data, missing_strategy="invalid")

    def test_numpy_array_input(self):
        """Test DataProcessor accepts numpy arrays."""
        arr = np.array([1, 2, 3])
        dp = DataProcessor(arr)
        assert dp.shape == (3, 1)

    def test_invalid_input_type_raises(self):
        """Test TypeError on invalid input type."""
        with pytest.raises(TypeError):
            DataProcessor([1, 2, 3])  # plain list not accepted


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.data = np.random.normal(10, 2, 100)

    def test_describe_output_keys(self):
        """Test describe() returns expected keys."""
        sa = StatisticalAnalyzer(self.data)
        result = sa.describe()
        expected_keys = {"n", "mean", "std", "min", "q25", "median", "q75", "max"}
        assert set(result.keys()) == expected_keys

    def test_describe_mean_approx(self):
        """Test mean is approximately correct."""
        sa = StatisticalAnalyzer(self.data)
        result = sa.describe()
        assert result["mean"] == pytest.approx(np.mean(self.data))

    def test_confidence_interval_coverage(self):
        """Test CI width makes statistical sense."""
        sa = StatisticalAnalyzer(self.data)
        lo, hi = sa.confidence_interval(alpha=0.05)
        assert lo < hi
        assert lo < np.mean(self.data) < hi

    def test_ci_requires_minimum_data(self):
        """Test CI raises ValueError for insufficient data."""
        sa = StatisticalAnalyzer([5.0])
        with pytest.raises(ValueError, match="at least 2"):
            sa.confidence_interval()

    def test_nan_ignored_in_describe(self):
        """Test NaN values are excluded from computations."""
        data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        sa = StatisticalAnalyzer(data_with_nan)
        result = sa.describe()
        assert result["n"] == 4


class TestComputeEffectSize:
    """Tests for compute_effect_size function."""

    def setup_method(self):
        """Set up test groups with known effect size."""
        np.random.seed(42)
        self.g1 = np.random.normal(10, 2, 50)
        self.g2 = np.random.normal(8, 2, 50)  # true d ≈ 1.0

    def test_cohens_d_positive(self):
        """Test Cohen's d is positive when group1 > group2."""
        d = compute_effect_size(self.g1, self.g2, method="cohens_d")
        assert d > 0

    def test_cohens_d_symmetric(self):
        """Test that reversing groups negates the effect size."""
        d1 = compute_effect_size(self.g1, self.g2, method="cohens_d")
        d2 = compute_effect_size(self.g2, self.g1, method="cohens_d")
        assert d1 == pytest.approx(-d2, abs=1e-10)

    def test_hedges_g_smaller_than_d(self):
        """Test Hedges' g is smaller than Cohen's d in magnitude."""
        d = abs(compute_effect_size(self.g1, self.g2, method="cohens_d"))
        g = abs(compute_effect_size(self.g1, self.g2, method="hedges_g"))
        assert g < d

    def test_unknown_method_raises(self):
        """Test ValueError on unknown method."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_effect_size(self.g1, self.g2, method="unknown_es")

    @pytest.mark.parametrize("method", ["cohens_d", "glass_delta", "hedges_g"])
    def test_all_methods_return_float(self, method):
        """Test all methods return a float."""
        result = compute_effect_size(self.g1, self.g2, method=method)
        assert isinstance(result, float)


# Run tests with verbose output
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```

### Step 3: Documentation Setup with Sphinx

```bash
# Initialize Sphinx docs
cd docs
sphinx-quickstart --quiet --project "My Research Package" \
  --author "Research Team" --release "0.1.0" \
  --language en --ext-autodoc --ext-napoleon \
  --ext-viewcode --makefile --no-batchfile
```

```python
# File: docs/conf.py (after sphinx-quickstart, add these settings)
import os
import sys

# Add src to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

project = "My Research Package"
copyright = "2026, Research Team"
author = "Research Team"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",       # auto-generate API docs from docstrings
    "sphinx.ext.napoleon",      # support NumPy/Google docstring styles
    "sphinx.ext.viewcode",      # add source code links
    "sphinx.ext.intersphinx",   # link to external docs (numpy, pandas)
    "sphinx.ext.mathjax",       # render LaTeX equations
]

# Napoleon settings for NumPy docstring style
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "special-members": "__init__",
}

# Intersphinx links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "sphinx_rtd_theme"
```

```rst
.. File: docs/api.rst

API Reference
=============

Core Module
-----------

.. automodule:: my_research_package.core
   :members:
   :undoc-members:
   :show-inheritance:

DataProcessor
~~~~~~~~~~~~~

.. autoclass:: my_research_package.core.DataProcessor
   :members:
   :special-members: __init__

StatisticalAnalyzer
~~~~~~~~~~~~~~~~~~~

.. autoclass:: my_research_package.core.StatisticalAnalyzer
   :members:
   :special-members: __init__

Functions
~~~~~~~~~

.. autofunction:: my_research_package.core.compute_effect_size
```

```bash
# Build HTML documentation
cd docs
make html
# Documentation generated in docs/_build/html/index.html
```

## Advanced Usage

### GitHub Actions CI/CD Pipeline

```yaml
# File: .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test Python ${{ matrix.python-version }}
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install package in development mode
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests with coverage
      run: |
        pytest --cov=src/my_research_package \
               --cov-report=xml --cov-report=term-missing \
               -v

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: false

  publish:
    name: Publish to PyPI
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    - name: Build distribution
      run: |
        pip install build
        python -m build
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### Programmatic Version Bumping

```python
# File: scripts/bump_version.py
"""Script to bump version in pyproject.toml and __init__.py."""
import re
import sys
import os

def bump_version(version_type="patch"):
    """Bump semantic version.

    Args:
        version_type: 'major', 'minor', or 'patch'
    """
    # Read current version from __init__.py
    init_path = os.path.join("src", "my_research_package", "__init__.py")
    with open(init_path) as f:
        content = f.read()

    match = re.search(r'__version__ = "(\d+)\.(\d+)\.(\d+)"', content)
    if not match:
        raise ValueError("Could not find version string")

    major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))

    if version_type == "major":
        major += 1; minor = 0; patch = 0
    elif version_type == "minor":
        minor += 1; patch = 0
    elif version_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown version_type: {version_type}")

    new_version = f"{major}.{minor}.{patch}"
    new_content = content.replace(match.group(0), f'__version__ = "{new_version}"')

    with open(init_path, "w") as f:
        f.write(new_content)

    # Update pyproject.toml
    pyproject_path = "pyproject.toml"
    with open(pyproject_path) as f:
        proj_content = f.read()
    old_ver_str = f'version = "{major-1 if version_type=="major" else major}.'
    # Simple replacement (assumes version appears once)
    old_v = match.group(0).replace("__version__ = ", "version = ")
    proj_content = proj_content.replace(
        f'version = "{match.group(1)}.{match.group(2)}.{match.group(3)}"',
        f'version = "{new_version}"'
    )
    with open(pyproject_path, "w") as f:
        f.write(proj_content)

    print(f"Version bumped to {new_version}")
    return new_version

if __name__ == "__main__":
    vtype = sys.argv[1] if len(sys.argv) > 1 else "patch"
    bump_version(vtype)
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError` in tests | src not in PYTHONPATH | Install in editable mode: `pip install -e .` |
| Sphinx autodoc finds no members | Wrong sys.path in conf.py | Verify `sys.path.insert(0, "../src")` |
| pytest import error | Conflicting package installations | Use `python -m pytest` instead of plain `pytest` |
| Build fails: missing wheel | build backend misconfigured | Install `build`: `pip install build`; run `python -m build` |
| PyPI upload fails | Token not set | Create API token at pypi.org; store as environment variable |
| Coverage < 80% | Untested branches | Add tests for edge cases; use `--cov-branch` flag |

## External Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [pyproject.toml specification (PEP 518)](https://peps.python.org/pep-0518/)
- [pytest documentation](https://docs.pytest.org/)
- [Sphinx autodoc documentation](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Research Software Engineering best practices](https://software.ac.uk/guides)
- [JOSS — Journal of Open Source Software](https://joss.theoj.org/)

## Examples

### Example 1: CLI Entry Point for a Research Tool

```python
# File: src/my_research_package/cli.py
"""Command-line interface for my_research_package."""

import argparse
import sys
import numpy as np
from .core import StatisticalAnalyzer, compute_effect_size


def main():
    """Entry point for the research analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Compute statistics for research data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Describe subcommand
    desc_parser = subparsers.add_parser("describe", help="Compute descriptive statistics")
    desc_parser.add_argument("--values", nargs="+", type=float, required=True,
                              help="Space-separated values")

    # Effect size subcommand
    es_parser = subparsers.add_parser("effect-size", help="Compute effect size")
    es_parser.add_argument("--group1", nargs="+", type=float, required=True)
    es_parser.add_argument("--group2", nargs="+", type=float, required=True)
    es_parser.add_argument("--method", default="cohens_d",
                            choices=["cohens_d", "glass_delta", "hedges_g"])

    args = parser.parse_args()

    if args.command == "describe":
        sa = StatisticalAnalyzer(args.values)
        stats = sa.describe()
        for k, v in stats.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    elif args.command == "effect-size":
        es = compute_effect_size(args.group1, args.group2, method=args.method)
        print(f"{args.method} = {es:.4f}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

```toml
# Add to pyproject.toml [project.scripts]
[project.scripts]
research-analyze = "my_research_package.cli:main"
```

### Example 2: Automated Package Testing

```bash
# Run full test suite with coverage report
pytest tests/ -v --cov=src/my_research_package \
  --cov-report=html --cov-report=term-missing \
  --tb=short 2>&1 | head -50

# Build distribution packages
python -m build
ls dist/   # should show .tar.gz and .whl files

# Check distribution
twine check dist/*

# Install and verify
pip install dist/*.whl
python -c "import my_research_package; print(my_research_package.__version__)"
```
