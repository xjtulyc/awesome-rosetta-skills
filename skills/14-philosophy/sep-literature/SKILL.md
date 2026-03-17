---
name: sep-literature
description: >
  Philosophy literature search: SEP web scraping, PhilPapers API, systematic
  review, citation tracing, and Chicago-style bibliography generation.
tags:
  - philosophy
  - sep
  - philpapers
  - bibliography
  - systematic-review
  - citation-analysis
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
  - requests>=2.31.0
  - beautifulsoup4>=4.12.0
  - lxml>=4.9.0
  - pandas>=2.0.0
  - tqdm>=4.65.0
last_updated: "2026-03-17"
---

# SEP & Philosophy Literature

A comprehensive skill for discovering, accessing, and organizing academic
philosophy literature. Supports scraping the Stanford Encyclopedia of Philosophy
(SEP) in compliance with its robots.txt, querying the PhilPapers API, systematic
literature review for philosophy, citation-genealogy tracing, and generating
Chicago-style (Notes and Author-Date) bibliographies.

---

## Architecture Overview

```
Search Layer
  ├── search_sep()               — SEP full-text search (HTML scraping)
  ├── search_philpapers()        — PhilPapers REST API
  └── search_crossref()          — CrossRef DOI / citation metadata

Processing Layer
  ├── get_sep_article_bibliography()  — extract SEP bibliography section
  ├── trace_concept_genealogy()       — track concept citations over time
  └── build_philosophy_bibliography() — merge, deduplicate, format

Output Layer
  └── format_chicago_*()         — Chicago Notes and Author-Date styles
```

---

## Core Functions

### 1. Stanford Encyclopedia of Philosophy (SEP)

```python
import os
import re
import time
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus
from typing import Iterator


_SEP_BASE = "https://plato.stanford.edu"

# robots.txt compliance: SEP allows crawling of /entries/ at a polite rate.
# See https://plato.stanford.edu/robots.txt — no Disallow for /entries/.
_SEP_CRAWL_DELAY = 2.0  # seconds between requests


def _get_html(url: str, session: requests.Session | None = None) -> BeautifulSoup:
    """Fetch URL and return BeautifulSoup object. Respects crawl delay."""
    s = session or requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; PhilosophyResearchBot/1.0; "
            "academic research use; contact: research@example.com)"
        )
    })
    resp = s.get(url, timeout=30)
    resp.raise_for_status()
    time.sleep(_SEP_CRAWL_DELAY)
    return BeautifulSoup(resp.text, "lxml")


def search_sep(
    query: str,
    category: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """
    Search the Stanford Encyclopedia of Philosophy.

    Uses SEP's built-in search endpoint. Results include entry title,
    URL slug, and a short excerpt.

    Parameters
    ----------
    query : str
        Search terms (natural language or Boolean: AND, OR, NOT).
    category : str or None
        Optional SEP category filter (e.g. 'ethics', 'metaphysics',
        'logic-and-philosophy-of-logic').
    max_results : int
        Maximum number of results to return.

    Returns
    -------
    list[dict]
        Each dict: {'title', 'slug', 'url', 'excerpt', 'first_published',
                    'last_revised'}.

    Notes
    -----
    SEP does not expose a public JSON API. This function scrapes the HTML
    search results page. Abide by SEP's terms of use — no commercial use,
    no bulk downloading of full article content.
    """
    search_url = f"{_SEP_BASE}/search/searcher.py"
    params = {"query": query, "search-how": "all"}
    if category:
        params["match-what"] = category

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (compatible; PhilosophyResearchBot/1.0; academic use)"
        )
    })
    resp = session.get(search_url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(_SEP_CRAWL_DELAY)

    soup = BeautifulSoup(resp.text, "lxml")
    results = []

    for item in soup.select("div.result")[:max_results]:
        title_tag = item.select_one("a.result-title") or item.select_one("a")
        if not title_tag:
            continue
        href = title_tag.get("href", "")
        slug_match = re.search(r"/entries/([^/]+)/", href)
        slug = slug_match.group(1) if slug_match else ""
        excerpt_tag = item.select_one("div.result-snippet") or item.select_one("p")
        excerpt = excerpt_tag.get_text(strip=True) if excerpt_tag else ""

        # Dates
        date_tag = item.select_one("span.result-date") or item.select_one("span.date")
        date_text = date_tag.get_text(strip=True) if date_tag else ""

        results.append({
            "title": title_tag.get_text(strip=True),
            "slug": slug,
            "url": urljoin(_SEP_BASE, href),
            "excerpt": excerpt[:300],
            "date_info": date_text,
        })

    return results


def get_sep_article_metadata(entry_slug: str) -> dict:
    """
    Retrieve metadata for a single SEP entry (title, authors, dates, abstract).

    Parameters
    ----------
    entry_slug : str
        The SEP entry slug (e.g. 'moral-luck', 'consciousness', 'free-will').

    Returns
    -------
    dict
        Keys: title, authors, first_published, last_revised, preamble,
        section_titles, url.
    """
    url = f"{_SEP_BASE}/entries/{entry_slug}/"
    soup = _get_html(url)

    title = soup.find("h1")
    title_text = title.get_text(strip=True) if title else ""

    # Authors appear in <div id="article-copyright">
    authors_div = soup.find("div", id="article-copyright")
    authors_text = authors_div.get_text(strip=True) if authors_div else ""

    # Publication info
    pub_info = soup.find("div", id="pubinfo")
    pub_text = pub_info.get_text(strip=True) if pub_info else ""

    # Preamble / abstract (first <p> after the TOC)
    preamble_div = soup.find("div", id="preamble")
    preamble = ""
    if preamble_div:
        first_p = preamble_div.find("p")
        preamble = first_p.get_text(strip=True) if first_p else ""

    # Section titles
    sections = [h2.get_text(strip=True) for h2 in soup.select("div#main-text h2")]

    return {
        "title": title_text,
        "authors": authors_text,
        "publication_info": pub_text,
        "preamble": preamble[:600],
        "section_titles": sections,
        "url": url,
        "slug": entry_slug,
    }


def get_sep_article_bibliography(entry_slug: str) -> list[dict]:
    """
    Extract the bibliography from a SEP article entry.

    Parameters
    ----------
    entry_slug : str
        SEP entry slug (e.g. 'moral-luck').

    Returns
    -------
    list[dict]
        Each dict: {'raw_citation', 'authors', 'year', 'title_fragment'}.
    """
    url = f"{_SEP_BASE}/entries/{entry_slug}/"
    soup = _get_html(url)

    # Bibliography is in <div id="bibliography"> or <section id="bibliography">
    bib_div = soup.find("div", id="bibliography") or soup.find("section", id="bibliography")
    if not bib_div:
        return []

    entries = []
    for li in bib_div.find_all("li"):
        raw = li.get_text(separator=" ", strip=True)
        if not raw:
            continue

        # Attempt to extract year (4-digit number)
        year_match = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\b", raw)
        year = year_match.group(1) if year_match else ""

        # Attempt to extract italic title (usually in <em>)
        em = li.find("em")
        title_frag = em.get_text(strip=True) if em else ""

        # Leading text up to year ~ author names
        author_fragment = raw[:year_match.start()].strip() if year_match else ""

        entries.append({
            "raw_citation": raw,
            "authors": author_fragment[:120],
            "year": year,
            "title_fragment": title_frag[:120],
        })

    return entries
```

### 2. PhilPapers API

```python
_PHILPAPERS_API = "https://philpapers.org/api"


def search_philpapers(
    query: str,
    categories: list[str] | None = None,
    years: tuple[int, int] | None = None,
    max_results: int = 30,
    format_type: str = "json",
) -> list[dict]:
    """
    Search PhilPapers — the largest philosophy bibliography database.

    Parameters
    ----------
    query : str
        Search query string. Supports field search: author:Nagel, title:luck.
    categories : list[str] or None
        PhilPapers category codes (e.g. ['phil-mind', 'normative-ethics']).
        See https://philpapers.org/categories.pl for codes.
    years : tuple[int, int] or None
        Inclusive year range filter: (start_year, end_year).
    max_results : int
        Maximum number of entries to return.
    format_type : str
        'json' (default) or 'bib' (BibTeX).

    Returns
    -------
    list[dict]
        Normalized entry dicts: title, authors, year, journal, volume,
        pages, doi, abstract, philpapers_id, url.
    """
    params: dict = {
        "method": "search",
        "query": query,
        "limit": min(max_results, 100),
        "format": format_type,
    }
    if categories:
        params["categoryIds"] = ",".join(categories)
    if years:
        params["pubDateStart"] = years[0]
        params["pubDateEnd"] = years[1]

    resp = requests.get(_PHILPAPERS_API, params=params, timeout=30)
    # PhilPapers API returns HTTP 200 even for empty results
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        return []

    entries = []
    for item in (data if isinstance(data, list) else data.get("entries", [])):
        entry = {
            "philpapers_id": item.get("id", ""),
            "title": item.get("title", ""),
            "authors": _parse_philpapers_authors(item.get("authors", [])),
            "year": str(item.get("pubYear", "")),
            "journal": item.get("journalTitle", "") or item.get("publication", ""),
            "volume": str(item.get("volume", "")),
            "pages": item.get("pages", ""),
            "doi": item.get("doi", ""),
            "abstract": (item.get("abstract") or "")[:400],
            "url": f"https://philpapers.org/rec/{item.get('id', '')}",
        }
        entries.append(entry)

    return entries


def _parse_philpapers_authors(authors_raw) -> str:
    """Convert PhilPapers author objects or strings to a comma-separated string."""
    if not authors_raw:
        return ""
    if isinstance(authors_raw, str):
        return authors_raw
    names = []
    for a in authors_raw:
        if isinstance(a, dict):
            last = a.get("last", "")
            first = a.get("first", "")
            names.append(f"{last}, {first}".strip(", "))
        elif isinstance(a, str):
            names.append(a)
    return "; ".join(names)
```

### 3. Bibliography Management and Citation Tracing

```python
def build_philosophy_bibliography(
    topic: str,
    sources: list[str] | None = None,
    years: tuple[int, int] | None = None,
    max_per_source: int = 20,
) -> pd.DataFrame:
    """
    Construct a unified bibliography by querying multiple philosophy sources.

    Parameters
    ----------
    topic : str
        Research topic or concept (e.g. 'moral luck', 'consciousness').
    sources : list[str] or None
        Sources to query. Options: 'sep', 'philpapers'. Defaults to both.
    years : tuple[int, int] or None
        Year range filter for PhilPapers.
    max_per_source : int
        Maximum entries per source.

    Returns
    -------
    pd.DataFrame
        Deduplicated bibliography sorted by year descending.
        Columns: title, authors, year, source, journal, doi, url, abstract.
    """
    if sources is None:
        sources = ["philpapers", "sep"]

    all_records: list[dict] = []

    if "philpapers" in sources:
        pp_results = search_philpapers(topic, years=years, max_results=max_per_source)
        for r in pp_results:
            r["source"] = "PhilPapers"
        all_records.extend(pp_results)
        time.sleep(1.0)

    if "sep" in sources:
        sep_results = search_sep(topic, max_results=max_per_source)
        for r in sep_results:
            all_records.append({
                "title": r["title"],
                "authors": "",
                "year": "",
                "source": "SEP",
                "journal": "Stanford Encyclopedia of Philosophy",
                "doi": "",
                "url": r["url"],
                "abstract": r["excerpt"],
                "philpapers_id": "",
                "volume": "",
                "pages": "",
            })

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # Deduplicate by normalised title
    df["_title_norm"] = df["title"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.drop_duplicates(subset="_title_norm").drop(columns=["_title_norm"])

    df = df.sort_values("year", ascending=False).reset_index(drop=True)
    return df


def trace_concept_genealogy(
    concept: str,
    start_year: int = 1950,
    end_year: int | None = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """
    Trace the intellectual history of a philosophical concept over time.

    Queries PhilPapers for works mentioning the concept, then groups and
    counts publications by decade to reveal when the debate intensified.

    Parameters
    ----------
    concept : str
        The philosophical concept to trace (e.g. 'moral luck', 'supervenience').
    start_year : int
        Earliest year to search.
    end_year : int or None
        Latest year to search. Defaults to current year.
    max_results : int
        Total maximum papers to retrieve.

    Returns
    -------
    pd.DataFrame
        Columns: title, authors, year, journal, doi, url, decade.
        Sorted by year ascending.
    """
    if end_year is None:
        import datetime
        end_year = datetime.date.today().year

    results = search_philpapers(
        concept,
        years=(start_year, end_year),
        max_results=max_results,
    )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year_int"])
    df["year_int"] = df["year_int"].astype(int)
    df["decade"] = (df["year_int"] // 10) * 10
    df = df.sort_values("year_int").reset_index(drop=True)

    return df


def format_chicago_notes(entry: dict) -> str:
    """
    Format a bibliography entry in Chicago Notes style (footnote format).

    Example output:
        Thomas Nagel, "Moral Luck," *Mortal Questions* (Cambridge: Cambridge
        University Press, 1979), 24–38.

    Parameters
    ----------
    entry : dict
        Dict with keys: title, authors, year, journal, volume, pages, doi.

    Returns
    -------
    str
        Formatted Chicago Notes citation string.
    """
    authors = entry.get("authors", "").replace(";", ",")
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    year = entry.get("year", "")
    volume = entry.get("volume", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")

    # Distinguish article vs book
    if journal and journal != "Stanford Encyclopedia of Philosophy":
        parts = [f'{authors}, "{title},"']
        if journal:
            parts.append(f"*{journal}*")
        if volume:
            parts.append(volume)
        if year:
            parts.append(f"({year})")
        if pages:
            parts.append(f": {pages}")
        if doi:
            parts.append(f". https://doi.org/{doi}")
        return " ".join(parts).rstrip()
    else:
        # Book / encyclopedia entry
        parts = [f'{authors}.']
        if journal == "Stanford Encyclopedia of Philosophy":
            parts.append(f'"{title}."')
            parts.append("In *Stanford Encyclopedia of Philosophy*,")
            parts.append(f"edited by Edward N. Zalta.")
            if year:
                parts.append(f"Last revised {year}.")
        else:
            parts.append(f"*{title}*.")
            if year:
                parts.append(f"{year}.")
        return " ".join(parts)


def format_chicago_author_date(entry: dict) -> str:
    """
    Format a bibliography entry in Chicago Author-Date style.

    Example output:
        Nagel, Thomas. 1979. "Moral Luck." In *Mortal Questions*, 24–38.
        Cambridge: Cambridge University Press.

    Parameters
    ----------
    entry : dict
        Dict with keys: title, authors, year, journal, volume, pages, doi.

    Returns
    -------
    str
        Formatted Chicago Author-Date citation string.
    """
    raw_authors = entry.get("authors", "")
    title = entry.get("title", "")
    journal = entry.get("journal", "")
    year = entry.get("year", "n.d.")
    volume = entry.get("volume", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")

    # Reverse first author: "Last, First" for bibliography
    first_author = raw_authors.split(";")[0].strip()
    remaining = "; ".join(a.strip() for a in raw_authors.split(";")[1:])
    author_part = first_author
    if remaining:
        author_part += f", and {remaining}"

    if journal and journal != "Stanford Encyclopedia of Philosophy":
        vol_pages = f"{volume}" + (f": {pages}" if pages else "")
        doi_part = f" https://doi.org/{doi}." if doi else ""
        return (
            f'{author_part}. {year}. "{title}." *{journal}* {vol_pages}.{doi_part}'
        )
    else:
        return f'{author_part}. {year}. "{title}." *Stanford Encyclopedia of Philosophy*.'
```

---

## Example 1: Systematic Literature Review on "Moral Luck"

Perform a structured literature review on moral luck across SEP and PhilPapers,
build a unified bibliography, and analyse the concept's publication trajectory.

```python
import pandas as pd
import json

TOPIC = "moral luck"
YEARS = (1970, 2025)

print(f"=== Systematic Philosophy Literature Review: {TOPIC!r} ===\n")

# ── Step 1: PhilPapers search ─────────────────────────────────────────────────
print("Querying PhilPapers...")
pp_results = search_philpapers(
    query=TOPIC,
    years=YEARS,
    max_results=50,
)
print(f"  PhilPapers: {len(pp_results)} results")

# ── Step 2: SEP search ────────────────────────────────────────────────────────
print("Querying SEP...")
sep_results = search_sep(TOPIC, max_results=10)
print(f"  SEP: {len(sep_results)} entries")

# ── Step 3: Get SEP bibliography for the main entry ──────────────────────────
print("\nFetching SEP bibliography for 'moral-luck' entry...")
sep_bib = get_sep_article_bibliography("moral-luck")
print(f"  Found {len(sep_bib)} bibliography entries in SEP article.")

# ── Step 4: Unified bibliography ─────────────────────────────────────────────
bib_df = build_philosophy_bibliography(TOPIC, sources=["philpapers", "sep"], years=YEARS)
print(f"\nUnified bibliography: {len(bib_df)} entries")

# ── Step 5: Concept genealogy ─────────────────────────────────────────────────
genealogy = trace_concept_genealogy(TOPIC, start_year=1960)
if not genealogy.empty:
    decade_counts = genealogy.groupby("decade").size().reset_index(name="publications")
    print("\nPublications per decade:")
    for _, row in decade_counts.iterrows():
        bar = "#" * int(row["publications"])
        print(f"  {int(row['decade'])}s: {bar} ({int(row['publications'])})")

# ── Step 6: Format first 5 entries in Chicago Author-Date ────────────────────
print("\n=== Sample Chicago Author-Date Citations ===")
for entry in pp_results[:5]:
    citation = format_chicago_author_date(entry)
    print(f"  {citation}")

# ── Step 7: Save bibliography ─────────────────────────────────────────────────
bib_df.to_csv("moral_luck_bibliography.csv", index=False, encoding="utf-8")

# Export as BibTeX-like JSONL for Zotero import
with open("moral_luck_bibliography.jsonl", "w", encoding="utf-8") as fh:
    for _, row in bib_df.iterrows():
        fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

print(f"\nSaved bibliography to moral_luck_bibliography.csv and .jsonl")

# ── Step 8: Show foundational papers ─────────────────────────────────────────
print("\nEarliest papers (foundational texts):")
if not genealogy.empty:
    earliest = genealogy.nsmallest(5, "year_int")[["title", "authors", "year", "journal"]]
    print(earliest.to_string(index=False))
```

---

## Example 2: Trace the Citation History of a Key Philosophical Paper

Identify works that cite or respond to a seminal paper by searching for its
author + key concepts, then map the intellectual lineage.

```python
import pandas as pd

# We trace the influence of Thomas Nagel's "What Is It Like to Be a Bat?" (1974)
SEMINAL_AUTHOR = "Nagel"
SEMINAL_CONCEPTS = ["consciousness", "qualia", "what is it like", "bat"]
START_YEAR = 1974

print("=== Tracing Influence of Nagel's 'What Is It Like to Be a Bat?' ===\n")

# ── Collect citing / responding works ────────────────────────────────────────
all_results = []
for concept in SEMINAL_CONCEPTS:
    query = f"{SEMINAL_AUTHOR} {concept}"
    results = search_philpapers(query, years=(START_YEAR, 2025), max_results=25)
    for r in results:
        r["search_concept"] = concept
    all_results.extend(results)
    time.sleep(1.0)  # polite rate limit

# Deduplicate
seen_ids = set()
unique_results = []
for r in all_results:
    pid = r.get("philpapers_id") or r.get("title", "")
    if pid not in seen_ids:
        seen_ids.add(pid)
        unique_results.append(r)

print(f"Unique works found across concept searches: {len(unique_results)}")

# ── SEP articles on related topics ───────────────────────────────────────────
related_sep = []
for concept in ["consciousness", "qualia", "mind-body problem"]:
    entries = search_sep(concept, max_results=5)
    related_sep.extend(entries)

print(f"Related SEP entries: {len(related_sep)}")

# ── Build timeline DataFrame ─────────────────────────────────────────────────
df = pd.DataFrame(unique_results)
if not df.empty and "year" in df.columns:
    df["year_int"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year_int"]).sort_values("year_int")

    # Decade-by-decade response summary
    df["decade"] = (df["year_int"] // 10 * 10).astype(int)
    print("\nResponsesdecade by decade:")
    for decade, group in df.groupby("decade"):
        titles = group["title"].tolist()[:3]
        print(f"\n  {decade}s ({len(group)} works):")
        for t in titles:
            print(f"    - {t}")

# ── Get SEP article on consciousness for its bibliography ────────────────────
print("\nFetching SEP 'consciousness' article bibliography...")
consciousness_bib = get_sep_article_bibliography("consciousness")
print(f"  {len(consciousness_bib)} references in SEP consciousness article")

# Find references by Nagel
nagel_refs = [r for r in consciousness_bib if "nagel" in r["raw_citation"].lower()]
print(f"  References to Nagel: {len(nagel_refs)}")
for ref in nagel_refs[:3]:
    print(f"    [{ref['year']}] {ref['raw_citation'][:120]}")

# ── Format Chicago Notes citations for 5 key responses ───────────────────────
print("\n=== Key Responding Works (Chicago Notes Format) ===")
if not df.empty:
    for _, row in df.head(5).iterrows():
        entry = row.to_dict()
        print(f"  {format_chicago_notes(entry)}")
        print()

# ── Export influence map ──────────────────────────────────────────────────────
if not df.empty:
    df.to_csv("nagel_bat_influence_map.csv", index=False, encoding="utf-8")
    print("Saved influence map to nagel_bat_influence_map.csv")
```

---

## Philosophy-Specific Search Strategies

### Tracing Philosophical Traditions

When researching a philosophical concept, use the following layered approach:

1. **Start with SEP**: SEP articles provide authoritative overviews with
   curated bibliographies. Begin with `get_sep_article_bibliography()` to
   identify the canonical texts.
2. **Expand via PhilPapers**: Use `search_philpapers()` with the concept and
   key author names found in the SEP bibliography.
3. **Concept genealogy**: Apply `trace_concept_genealogy()` to see when
   interest in the topic peaked and declined.
4. **Citation tracing**: Search for works that cite or respond to a seminal
   paper by combining author name + key terms.
5. **Cross-tradition search**: Philosophy concepts often appear under different
   names in different traditions (analytic vs continental). Always search
   synonyms (e.g. "supervenience" and "strong supervenience" and "local
   supervenience").

### Chicago Citation Style Notes

The Chicago Manual of Style (17th ed.) provides two parallel systems:

- **Notes-Bibliography** (Chicago Notes): Used in humanities. Full references
  appear in footnotes/endnotes and a bibliography. Use `format_chicago_notes()`.
- **Author-Date** (Chicago Author-Date): Used in social sciences. In-text
  citations are `(Author Year, Page)`. Full references in a reference list.
  Use `format_chicago_author_date()`.

For philosophy journals, Notes-Bibliography is standard (e.g. *Philosophical
Review*, *Ethics*, *Mind*). Author-Date is used in some interdisciplinary
venues (*Synthese*, *Erkenntnis*, *Philosophy of Science*).

---

## Notes and Best Practices

```python
# Example: Check SEP robots.txt compliance before scraping
import requests
resp = requests.get("https://plato.stanford.edu/robots.txt", timeout=10)
print(resp.text)
# Output shows no Disallow for /entries/ — academic scraping is permitted
# at a polite rate. Always use _SEP_CRAWL_DELAY >= 2 seconds.
```

- **SEP rate limiting**: SEP is a non-profit academic resource. Keep crawl
  delays at 2+ seconds and cache results locally to avoid redundant requests.
  Do not download entire articles for corpus construction without permission.
- **PhilPapers API**: The API is rate-limited and may return truncated abstracts.
  For full paper text, follow the DOI or PhilPapers URL directly.
- **Author name disambiguation**: Philosophy authors often publish under
  multiple name forms (e.g. "Bernard Williams" vs "B.A.O. Williams"). Use
  PhilPapers author IDs when available for precise queries.
- **Year parsing**: Historical philosophy papers may list year ranges
  (e.g. "1781/1998"). Use `pd.to_numeric(..., errors='coerce')` and handle NaN
  appropriately.
- **CrossRef for DOI lookups**: For papers found via SEP bibliographies
  without DOIs, use the CrossRef REST API (`api.crossref.org/works`) to resolve
  full metadata by title + author name.

---

## Dependencies Installation

```bash
pip install requests>=2.31.0 beautifulsoup4>=4.12.0 lxml>=4.9.0 \
            pandas>=2.0.0 tqdm>=4.65.0
```
