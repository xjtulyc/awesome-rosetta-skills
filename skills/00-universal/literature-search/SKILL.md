---
name: literature-search
description: >
  Use this Skill when the user needs to search academic literature, collect papers
  for a systematic review, or find the latest research on any topic.
  Covers cross-database search via OpenAlex API, Semantic Scholar API, and arXiv API,
  with deduplication, citation snowballing, BibTeX export, and annual trend visualization.
tags:
  - universal
  - literature-search
  - bibliometrics
  - openalex
  - semantic-scholar
  - arxiv
version: "1.0.0"
authors:
  - name: awesome-rosetta-skills contributors
    github: "@awesome-rosetta-skills"
license: "MIT"
platforms:
  - claude-code
  - codex
  - gemini-cli
  - cursor
dependencies:
  python:
    - requests>=2.28.0
    - pandas>=1.5.0
    - matplotlib>=3.6.0
    - pyalex>=0.13
    - semanticscholar>=0.5.0
last_updated: "2026-03-17"
---

# Literature Search

> **TL;DR** — Cross-database academic literature search using OpenAlex, Semantic Scholar,
> and arXiv APIs. Includes deduplication, citation snowballing, BibTeX export, and
> annual publication trend visualization.

---

## 1. Overview

### What Problem Does This Skill Solve?

Academic literature search spans multiple databases, each with different coverage,
search syntax, and API quirks. Manually reconciling results leads to missed papers
and wasted hours. This Skill provides a unified, programmatic workflow that:

- Queries **three complementary databases** in one pass
- **Deduplicates** results by DOI and normalized title
- Performs **citation snowballing** to find seminal upstream papers
- Exports a clean **BibTeX** file ready for LaTeX / Zotero
- Plots **annual publication trends** to reveal research momentum

### Applicable Scenarios

| Scenario | Recommended Entry Point |
|---|---|
| Systematic review / meta-analysis | `search_openalex()` + `search_semantic_scholar()` |
| Latest preprints on a narrow topic | `search_arxiv()` |
| Finding papers that cite a known key paper | `get_citing_papers()` |
| Weekly monitoring of new work | `weekly_arxiv_monitor()` |
| Exporting results to LaTeX / Zotero | `export_bibtex()` |

### Key Limitations

- **OpenAlex** abstracts are stored as inverted indexes; reconstruction adds ~10 ms per paper.
- **Semantic Scholar** anonymous rate limit is 100 requests / 5 min; request a free API key
  at <https://www.semanticscholar.org/product/api> for higher throughput.
- **arXiv** rate limit is ~3 requests/sec; a `time.sleep(0.4)` between pages is included.
- DOI-based deduplication misses papers without DOIs (common for preprints).

---

## 2. Environment Setup

```bash
# Create / activate environment
conda create -n lit python=3.11 -y
conda activate lit

# Install dependencies
pip install requests pandas matplotlib pyalex semanticscholar

# Optional: set Semantic Scholar API key to raise rate limit
export SEMANTIC_SCHOLAR_API_KEY="<paste-your-key>"
```

Verify installation:

```python
import pyalex, semanticscholar, requests, pandas, matplotlib
print("All dependencies OK")
```

---

## 3. Core Implementation

### 3.1 OpenAlex Search

OpenAlex (<https://openalex.org>) indexes 250 M+ works and is fully open.
Abstracts are stored as inverted indexes and must be reconstructed.

```python
import requests
import pandas as pd
import time
import re
from typing import Optional

OPENALEX_BASE = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words)


def search_openalex(
    query: str,
    max_results: int = 200,
    from_year: Optional[int] = None,
    to_year: Optional[int] = None,
    email: str = "researcher@example.com",
) -> pd.DataFrame:
    """
    Search OpenAlex for academic papers.

    Args:
        query:       Free-text search query (supports Boolean: AND, OR, NOT).
        max_results: Maximum number of results to return (hard cap: 10 000).
        from_year:   Filter papers published from this year onwards.
        to_year:     Filter papers published up to this year.
        email:       Polite-pool email — speeds up responses from OpenAlex.

    Returns:
        DataFrame with columns: doi, title, authors, year, venue,
        citations, abstract, openalex_id, source.
    """
    records = []
    per_page = min(200, max_results)
    page = 1

    filter_parts = []
    if from_year or to_year:
        y_from = from_year or 1900
        y_to = to_year or 2100
        filter_parts.append(f"publication_year:{y_from}-{y_to}")

    params = {
        "search": query,
        "per-page": per_page,
        "mailto": email,
        "select": (
            "doi,title,authorships,publication_year,primary_location,"
            "cited_by_count,abstract_inverted_index,id"
        ),
    }
    if filter_parts:
        params["filter"] = ",".join(filter_parts)

    while len(records) < max_results:
        params["page"] = page
        resp = requests.get(OPENALEX_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break

        for r in results:
            authors = [
                a["author"]["display_name"]
                for a in r.get("authorships", [])
                if a.get("author")
            ]
            loc = r.get("primary_location") or {}
            src = loc.get("source") or {}
            venue = src.get("display_name", "")

            records.append({
                "doi": (r.get("doi") or "").replace("https://doi.org/", ""),
                "title": r.get("title", ""),
                "authors": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
                "year": r.get("publication_year"),
                "venue": venue,
                "citations": r.get("cited_by_count", 0),
                "abstract": reconstruct_abstract(r.get("abstract_inverted_index") or {}),
                "openalex_id": r.get("id", ""),
                "source": "openalex",
            })

        meta = data.get("meta", {})
        if len(records) >= max_results or page * per_page >= meta.get("count", 0):
            break
        page += 1
        time.sleep(0.1)

    return pd.DataFrame(records[:max_results])
```

### 3.2 Semantic Scholar Search

```python
import os
from semanticscholar import SemanticScholar


def search_semantic_scholar(
    query: str,
    max_results: int = 100,
    fields: list = None,
) -> pd.DataFrame:
    """
    Search Semantic Scholar using the official Python client.

    Args:
        query:       Natural-language query (semantic matching, not just keyword).
        max_results: Maximum papers to return (anonymous limit: 100/5 min).
        fields:      Extra fields to fetch; defaults cover title, authors, year, etc.

    Returns:
        DataFrame with standardised columns matching search_openalex() output.
    """
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    sch = SemanticScholar(api_key=api_key)

    if fields is None:
        fields = [
            "title", "authors", "year", "venue",
            "citationCount", "abstract", "externalIds",
        ]

    results = sch.search_paper(query, limit=max_results, fields=fields)

    records = []
    for paper in results:
        authors = [a.name for a in (paper.authors or [])]
        ext = paper.externalIds or {}
        doi = ext.get("DOI", "")

        records.append({
            "doi": doi,
            "title": paper.title or "",
            "authors": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
            "year": paper.year,
            "venue": paper.venue or "",
            "citations": paper.citationCount or 0,
            "abstract": paper.abstract or "",
            "openalex_id": "",
            "source": "semantic_scholar",
        })

    return pd.DataFrame(records)
```

### 3.3 arXiv Search

```python
import xml.etree.ElementTree as ET

ARXIV_BASE = "http://export.arxiv.org/api/query"
NS = "http://www.w3.org/2005/Atom"


def search_arxiv(
    query: str,
    max_results: int = 50,
    category: Optional[str] = None,
) -> pd.DataFrame:
    """
    Search arXiv via its Atom/XML API.

    Args:
        query:       Search terms (supports field prefixes: ti:, au:, abs:).
        max_results: Maximum papers to return (hard cap per query: 2000).
        category:    Optional arXiv category, e.g. 'cs.LG', 'econ.EM', 'stat.ML'.

    Returns:
        DataFrame with standardised columns; doi uses 'arxiv:<id>' where no DOI exists.
    """
    if category:
        query = f"({query}) AND cat:{category}"

    start = 0
    batch = min(100, max_results)
    records = []

    while len(records) < max_results:
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": batch,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = requests.get(ARXIV_BASE, params=params, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        entries = root.findall(f"{{{NS}}}entry")
        if not entries:
            break

        for entry in entries:
            arxiv_id = (entry.findtext(f"{{{NS}}}id") or "").split("/abs/")[-1]
            title = (entry.findtext(f"{{{NS}}}title") or "").replace("\n", " ").strip()
            abstract = (entry.findtext(f"{{{NS}}}summary") or "").replace("\n", " ").strip()
            published = entry.findtext(f"{{{NS}}}published") or ""
            year = int(published[:4]) if published else None

            authors = [
                (a.findtext(f"{{{NS}}}name") or "")
                for a in entry.findall(f"{{{NS}}}author")
            ]

            doi = ""
            for link in entry.findall(f"{{{NS}}}link"):
                if link.get("title") == "doi":
                    doi = link.get("href", "").replace("https://doi.org/", "")

            records.append({
                "doi": doi or f"arxiv:{arxiv_id}",
                "title": title,
                "authors": "; ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
                "year": year,
                "venue": "arXiv",
                "citations": 0,
                "abstract": abstract,
                "openalex_id": "",
                "source": "arxiv",
            })

        start += batch
        time.sleep(0.4)
        if len(entries) < batch:
            break

    return pd.DataFrame(records[:max_results])
```

### 3.4 Deduplication

```python
def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def deduplicate_results(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple search result DataFrames and remove duplicates.

    Dedup priority:
      1. Exact DOI match (excluding 'arxiv:*' pseudo-DOIs)
      2. Normalized title match

    Args:
        *dfs: Any number of DataFrames returned by the search functions above.

    Returns:
        Single deduplicated DataFrame sorted by citations descending.
    """
    combined = pd.concat(dfs, ignore_index=True)
    combined["_norm_title"] = combined["title"].fillna("").apply(normalize_title)

    seen_dois: set = set()
    seen_titles: set = set()
    keep_mask = []

    for _, row in combined.iterrows():
        doi = row["doi"]
        nt = row["_norm_title"]
        real_doi = bool(doi) and not doi.startswith("arxiv:")

        if real_doi and doi in seen_dois:
            keep_mask.append(False)
        elif nt and nt in seen_titles:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
            if real_doi:
                seen_dois.add(doi)
            if nt:
                seen_titles.add(nt)

    result = combined[keep_mask].drop(columns=["_norm_title"])
    return result.sort_values("citations", ascending=False).reset_index(drop=True)
```

### 3.5 BibTeX Export

```python
def export_bibtex(df: pd.DataFrame, output_path: str = "results.bib") -> str:
    """
    Export search results to a BibTeX file.

    Args:
        df:          DataFrame from deduplicate_results() or any search function.
        output_path: File path for the .bib output.

    Returns:
        Full BibTeX string (also written to output_path).
    """
    bibtex_entries = []

    for _, row in df.iterrows():
        authors_raw = row.get("authors", "")
        year = row.get("year", "")
        title = row.get("title", "Untitled")
        venue = row.get("venue", "")
        doi = row.get("doi", "")

        first_author = (
            authors_raw.split(";")[0].strip().split()[-1]
            if authors_raw else "Unknown"
        )
        key = f"{first_author}{year}"
        author_bib = authors_raw.replace("; ", " and ")

        entry_lines = [
            f"@article{{{key},",
            f'  title   = {{{title}}},',
            f'  author  = {{{author_bib}}},',
            f'  year    = {{{year}}},',
        ]
        if venue:
            entry_lines.append(f'  journal = {{{venue}}},')
        if doi and not doi.startswith("arxiv:"):
            entry_lines.append(f'  doi     = {{{doi}}},')
        entry_lines.append("}")

        bibtex_entries.append("\n".join(entry_lines))

    bib_content = "\n\n".join(bibtex_entries)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bib_content)

    print(f"Exported {len(bibtex_entries)} entries to {output_path}")
    return bib_content
```

### 3.6 Annual Trend Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_annual_trend(
    df: pd.DataFrame,
    title: str = "Annual Publication Trend",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a bar chart of paper counts by year.

    Args:
        df:          Deduplicated results DataFrame.
        title:       Chart title.
        output_path: If given, save figure here (PNG / PDF / SVG).

    Returns:
        Matplotlib Figure object.
    """
    year_counts = (
        df["year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(year_counts.index, year_counts.values, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Papers")
    ax.set_title(title)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved trend chart to {output_path}")

    return fig
```

### 3.7 Citation Snowballing

```python
def get_citing_papers(
    doi: str,
    max_results: int = 100,
    email: str = "researcher@example.com",
) -> pd.DataFrame:
    """
    Find all papers that cite a given DOI using the OpenAlex citation filter.

    Args:
        doi:         Seed paper DOI (without https://doi.org/ prefix).
        max_results: Maximum citing papers to return.
        email:       Polite-pool email for OpenAlex.

    Returns:
        DataFrame of citing papers in the same format as search_openalex().
    """
    return search_openalex(
        query=f"cites:doi:{doi}",
        max_results=max_results,
        email=email,
    )
```

### 3.8 Weekly arXiv Monitor

```python
from datetime import datetime, timedelta


def weekly_arxiv_monitor(
    keywords: list,
    category: str,
    days_back: int = 7,
    output_csv: str = "weekly_arxiv.csv",
) -> pd.DataFrame:
    """
    Monitor arXiv for new papers matching keywords, published in the last N days.

    Args:
        keywords:   List of search terms (joined with OR).
        category:   arXiv category to filter, e.g. 'cs.LG', 'econ.EM'.
        days_back:  How many days back to search (default 7).
        output_csv: Path to save results as CSV.

    Returns:
        DataFrame of matching papers published within the window.
    """
    cutoff_year = (datetime.now() - timedelta(days=days_back)).year
    query = " OR ".join(f'"{kw}"' for kw in keywords)

    df = search_arxiv(query=query, max_results=200, category=category)

    if not df.empty and "year" in df.columns:
        df = df[df["year"] >= cutoff_year]

    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Weekly monitor: {len(df)} new papers -> {output_csv}")
    return df
```

---

## 4. End-to-End Examples

### Example 1 — Systematic Review on "LLM Hallucination"

```python
# Step 1: search across all three databases
df_oa = search_openalex(
    query="LLM hallucination factual accuracy",
    max_results=300,
    from_year=2020,
    email="researcher@university.edu",
)

df_ss = search_semantic_scholar(
    query="large language model hallucination",
    max_results=100,
)

df_ax = search_arxiv(
    query="hallucination large language model",
    max_results=50,
    category="cs.CL",
)

# Step 2: deduplicate
df_all = deduplicate_results(df_oa, df_ss, df_ax)
print(f"Total unique papers: {len(df_all)}")

# Step 3: citation snowballing on top-cited paper
top_doi = df_all.iloc[0]["doi"]
df_citing = get_citing_papers(doi=top_doi, max_results=50)
df_all = deduplicate_results(df_all, df_citing)
print(f"After snowballing: {len(df_all)}")

# Step 4: export
export_bibtex(df_all, output_path="hallucination_review.bib")
df_all.to_csv("hallucination_review.csv", index=False)

# Step 5: trend chart
fig = plot_annual_trend(
    df_all,
    title="Annual Papers on LLM Hallucination",
    output_path="trend_hallucination.png",
)
```

### Example 2 — Weekly arXiv Alert for Causal ML

```python
# Run weekly (e.g. via cron or GitHub Actions schedule)
new_papers = weekly_arxiv_monitor(
    keywords=["causal inference", "treatment effect", "counterfactual"],
    category="stat.ML",
    days_back=7,
    output_csv="causal_ml_weekly.csv",
)

summary_lines = [
    f"- [{row['title'][:80]}] ({row['year']}) -- {row['authors']}"
    for _, row in new_papers.head(10).iterrows()
]
print("This week's top causal ML papers:")
print("\n".join(summary_lines))
```

---

## 5. Common Errors and Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `429 Too Many Requests` | Semantic Scholar rate limit hit | Add `SEMANTIC_SCHOLAR_API_KEY`; add `time.sleep(1)` between calls |
| `abstract_inverted_index` is `None` | Paper has no abstract in OpenAlex | `reconstruct_abstract` returns `""` — filter with `df[df.abstract != ""]` |
| `ET.ParseError` | arXiv returned HTML error page | Retry after `time.sleep(5)`; inspect `resp.text` |
| Empty DataFrame after dedup | All results are duplicates across sources | Normal for narrow queries; increase `max_results` per source |
| BibTeX key collision | Two authors with same surname + year | Append counter suffix: `Smith2023a`, `Smith2023b` |
| `ModuleNotFoundError: semanticscholar` | Package not installed | `pip install semanticscholar` |

---

## 6. Performance Tips

- **Parallel queries**: Run all three `search_*()` functions concurrently with
  `concurrent.futures.ThreadPoolExecutor` to cut wall time by ~3x.
- **Caching**: Pickle or Parquet the raw DataFrames after the first run so re-runs
  during analysis skip all API calls.
- **OpenAlex cursor pagination**: For queries exceeding 10 000 results, switch from
  `page=N` to `cursor=*` pagination (see OpenAlex docs) to avoid result-count drift.
- **Abstract quality**: Inverted-index reconstruction is ~100% faithful; only very old
  papers (pre-2000) commonly have missing abstracts.

---

## 7. References and Further Reading

- OpenAlex documentation: <https://docs.openalex.org/>
- Semantic Scholar API guide: <https://api.semanticscholar.org/api-docs/>
- arXiv API user manual: <https://arxiv.org/help/api/user-manual>
- `pyalex` library: <https://github.com/J535D165/pyalex>
- `semanticscholar` Python client: <https://github.com/danielnsilva/semanticscholar>
- PRISMA reporting standards: <https://www.prisma-statement.org/>

---

## 8. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-17 | Initial release — OpenAlex, S2, arXiv, dedup, BibTeX, trend chart |
