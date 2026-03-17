---
name: digital-archives
description: >
  Access historical digital archives via REST APIs: Europeana, Chronicling America,
  Internet Archive, HathiTrust, DPLA, and OAI-PMH metadata harvesting.
tags:
  - digital-archives
  - history
  - api
  - metadata
  - europeana
  - chronicling-america
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
  - pandas>=2.0.0
  - beautifulsoup4>=4.12.0
  - lxml>=4.9.0
  - tqdm>=4.65.0
  - internetarchive>=3.3.0
last_updated: "2026-03-17"
---

# Digital Archives

A comprehensive skill for accessing and harvesting historical materials from
major digital archive platforms. Supports structured metadata retrieval,
full-text search, bulk downloads, and cross-archive normalization for corpus
construction in historical research projects.

API keys are read exclusively from environment variables — never hardcoded.

---

## Supported Archives

| Archive | Access Method | API Key Required |
|---|---|---|
| Europeana | REST API v2/v3 | Yes (`EUROPEANA_API_KEY`) |
| Chronicling America | REST API (LoC) | No |
| Internet Archive | Python SDK + REST | No |
| HathiTrust | Data API (Bibliographic) | No (public) |
| DPLA | REST API | Yes (`DPLA_API_KEY`) |
| OAI-PMH providers | XML harvesting | No |

---

## Core Functions

### 1. Europeana API

```python
import os
import time
import json
import requests
import pandas as pd
from typing import Iterator
from urllib.parse import urlencode


_EUROPEANA_BASE = "https://api.europeana.eu/record/v2"


def search_europeana(
    query: str,
    rows: int = 20,
    start: int = 1,
    api_key_env: str = "EUROPEANA_API_KEY",
    media_type: str | None = None,
    reusability: str = "open",
    extra_params: dict | None = None,
) -> dict:
    """
    Search the Europeana REST API for cultural heritage objects.

    Parameters
    ----------
    query : str
        Full-text search query (supports Solr syntax: AND, OR, NOT, field:value).
    rows : int
        Number of results per page (max 100).
    start : int
        Offset for pagination (1-indexed).
    api_key_env : str
        Environment variable name holding the Europeana API key.
    media_type : str or None
        Filter by media type: 'IMAGE', 'TEXT', 'VIDEO', 'SOUND', '3D'.
    reusability : str
        'open', 'restricted', or 'permission'.
    extra_params : dict or None
        Additional query parameters passed verbatim.

    Returns
    -------
    dict
        Parsed JSON response from Europeana API.

    Environment
    -----------
    Set EUROPEANA_API_KEY in your shell before running:
        export EUROPEANA_API_KEY="<paste-your-key>"
    Register at https://apis.europeana.eu/
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {api_key_env!r} is not set. "
            "Register at https://apis.europeana.eu/ to get a free API key."
        )

    params: dict = {
        "wskey": api_key,
        "query": query,
        "rows": min(rows, 100),
        "start": start,
        "profile": "rich",
    }
    if media_type:
        params["qf"] = f"TYPE:{media_type}"
    if reusability:
        params["reusability"] = reusability
    if extra_params:
        params.update(extra_params)

    resp = requests.get(f"{_EUROPEANA_BASE}/search.json", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_europeana_record(
    record_id: str,
    api_key_env: str = "EUROPEANA_API_KEY",
) -> dict:
    """
    Retrieve a single Europeana record by its full ID (e.g. '/9200338/BibliographicResource_...').
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(f"Environment variable {api_key_env!r} is not set.")

    # record_id may start with '/' — strip and reconstruct URL
    clean_id = record_id.lstrip("/")
    url = f"{_EUROPEANA_BASE}/{clean_id}.json"
    resp = requests.get(url, params={"wskey": api_key}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def paginate_europeana(
    query: str,
    total: int = 200,
    page_size: int = 100,
    api_key_env: str = "EUROPEANA_API_KEY",
    **kwargs,
) -> Iterator[dict]:
    """
    Paginate through Europeana results, yielding individual item dicts.
    """
    fetched = 0
    start = 1
    while fetched < total:
        batch_size = min(page_size, total - fetched)
        response = search_europeana(
            query, rows=batch_size, start=start, api_key_env=api_key_env, **kwargs
        )
        items = response.get("items", [])
        if not items:
            break
        for item in items:
            yield item
        fetched += len(items)
        start += len(items)
        time.sleep(0.2)  # polite rate limiting
```

### 2. Chronicling America (Library of Congress)

```python
_CHRONICLING_BASE = "https://chroniclingamerica.loc.gov"


def search_chronicling_america(
    query: str,
    date_range: tuple[str, str] | None = None,
    state: str | None = None,
    rows: int = 20,
    page: int = 1,
) -> dict:
    """
    Search Chronicling America historical US newspaper archive (Library of Congress).

    Parameters
    ----------
    query : str
        Full-text search query (phrase search supported with quotes).
    date_range : tuple[str, str] or None
        Inclusive date range as ('YYYY-MM-DD', 'YYYY-MM-DD').
    state : str or None
        Two-letter US state abbreviation filter (e.g. 'NY', 'CA').
    rows : int
        Number of results per page (1–25).
    page : int
        Page number for pagination.

    Returns
    -------
    dict
        JSON response including 'items', 'totalItems', 'endIndex'.
    """
    params: dict = {
        "andtext": query,
        "rows": min(rows, 25),
        "page": page,
        "format": "json",
    }
    if date_range:
        params["date1"] = date_range[0].replace("-", "")[:8]
        params["date2"] = date_range[1].replace("-", "")[:8]
        params["dateFilterType"] = "range"
    if state:
        params["state"] = state

    resp = requests.get(
        f"{_CHRONICLING_BASE}/search/pages/results/",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_chronicling_page_ocr(page_url: str) -> str:
    """
    Download the OCR full text for a Chronicling America newspaper page.

    Parameters
    ----------
    page_url : str
        The page URL from a search result item (ends with /seq-N/ or similar).

    Returns
    -------
    str
        Raw OCR text content.
    """
    # Construct the OCR text endpoint
    ocr_url = page_url.rstrip("/") + "/ocr.txt"
    resp = requests.get(ocr_url, timeout=30)
    resp.raise_for_status()
    return resp.text


def build_chronicling_corpus(
    query: str,
    date_range: tuple[str, str],
    state: str | None = None,
    max_pages: int = 100,
    download_ocr: bool = True,
) -> pd.DataFrame:
    """
    Build a corpus DataFrame from Chronicling America search results.

    Returns
    -------
    pd.DataFrame
        Columns: title, date, state, city, page_url, ocr_text (if download_ocr=True).
    """
    records = []
    page = 1
    fetched = 0

    while fetched < max_pages:
        data = search_chronicling_america(
            query, date_range=date_range, state=state, rows=25, page=page
        )
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            record = {
                "title": item.get("title_normal", ""),
                "date": item.get("date", ""),
                "state": ", ".join(item.get("state", [])),
                "city": ", ".join(item.get("city", [])),
                "page_url": _CHRONICLING_BASE + item.get("id", ""),
                "ocr_text": "",
            }
            if download_ocr and item.get("id"):
                try:
                    record["ocr_text"] = get_chronicling_page_ocr(record["page_url"])
                    time.sleep(0.1)
                except Exception:
                    record["ocr_text"] = ""
            records.append(record)
            fetched += 1
            if fetched >= max_pages:
                break

        page += 1
        time.sleep(0.5)

    return pd.DataFrame(records)
```

### 3. Internet Archive

```python
def download_internet_archive_texts(
    query: str,
    max_items: int = 10,
    output_dir: str = "./ia_downloads",
) -> pd.DataFrame:
    """
    Search and download full-text items from the Internet Archive.

    Parameters
    ----------
    query : str
        Lucene query string (e.g. 'subject:"World War I" AND mediatype:texts').
    max_items : int
        Maximum number of items to download.
    output_dir : str
        Local directory for downloaded files.

    Returns
    -------
    pd.DataFrame
        Metadata for all found (and attempted) items.
    """
    import internetarchive as ia

    os.makedirs(output_dir, exist_ok=True)
    search_results = ia.search_items(query, fields=["identifier", "title", "date", "subject"])

    records = []
    for i, result in enumerate(search_results):
        if i >= max_items:
            break
        identifier = result.get("identifier", "")
        record = {
            "identifier": identifier,
            "title": result.get("title", ""),
            "date": result.get("date", ""),
            "subject": result.get("subject", ""),
            "download_path": "",
            "status": "pending",
        }
        try:
            item = ia.get_item(identifier)
            # Download only the first text file (.txt or .pdf)
            for f in item.files:
                name = f.get("name", "")
                if name.endswith((".txt", "_djvu.txt", ".pdf")):
                    dest = os.path.join(output_dir, identifier)
                    os.makedirs(dest, exist_ok=True)
                    ia.download(
                        identifier,
                        files=[name],
                        destdir=output_dir,
                        silent=True,
                        ignore_existing=True,
                    )
                    record["download_path"] = os.path.join(dest, name)
                    record["status"] = "downloaded"
                    break
            else:
                record["status"] = "no_text_file"
        except Exception as exc:
            record["status"] = f"error: {exc}"
        records.append(record)
        time.sleep(0.5)

    return pd.DataFrame(records)
```

### 4. OAI-PMH Metadata Harvesting

```python
from xml.etree import ElementTree as ET

_OAI_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
}


def harvest_metadata_oai_pmh(
    base_url: str,
    set_name: str | None = None,
    metadata_prefix: str = "oai_dc",
    from_date: str | None = None,
    until_date: str | None = None,
    max_records: int = 500,
) -> list[dict]:
    """
    Harvest metadata records from any OAI-PMH-compliant repository.

    Compatible repositories include: HathiTrust, British Library,
    Europeana OAI endpoint, Library of Congress, DPLA, and many others.

    Parameters
    ----------
    base_url : str
        OAI-PMH base URL (e.g. 'https://repox.europeana.eu/repox/OAIHandler').
    set_name : str or None
        OAI set identifier for selective harvesting.
    metadata_prefix : str
        Metadata format: 'oai_dc', 'marc21', 'mods', etc.
    from_date : str or None
        Start date for selective harvesting ('YYYY-MM-DD').
    until_date : str or None
        End date for selective harvesting ('YYYY-MM-DD').
    max_records : int
        Maximum number of records to harvest.

    Returns
    -------
    list[dict]
        Each dict contains Dublin Core fields: identifier, title, creator,
        subject, description, date, type, language, rights, source.
    """
    def oai_request(verb: str, extra: dict | None = None) -> ET.Element:
        params = {"verb": verb, "metadataPrefix": metadata_prefix}
        if set_name:
            params["set"] = set_name
        if from_date:
            params["from"] = from_date
        if until_date:
            params["until"] = until_date
        if extra:
            params.update(extra)
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        return ET.fromstring(resp.content)

    def parse_dc_record(record_elem: ET.Element) -> dict:
        metadata = record_elem.find(".//oai_dc:dc", _OAI_NS)
        if metadata is None:
            return {}
        dc_fields = ["title", "creator", "subject", "description", "date",
                     "type", "identifier", "language", "rights", "source", "publisher"]
        out: dict = {}
        for field in dc_fields:
            elems = metadata.findall(f"dc:{field}", _OAI_NS)
            values = [e.text for e in elems if e.text]
            out[field] = " | ".join(values) if values else ""
        # OAI identifier (header)
        header = record_elem.find("oai:header", _OAI_NS)
        if header is not None:
            oai_id = header.findtext("oai:identifier", default="", namespaces=_OAI_NS)
            out["oai_identifier"] = oai_id
        return out

    records: list[dict] = []
    resumption_token: str | None = None

    while len(records) < max_records:
        extra: dict = {}
        if resumption_token:
            # When using resumptionToken, only verb and token are allowed
            extra = {"resumptionToken": resumption_token}
            for key in ["metadataPrefix", "set", "from", "until"]:
                extra.pop(key, None)

        if resumption_token:
            params = {"verb": "ListRecords", "resumptionToken": resumption_token}
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        else:
            root = oai_request("ListRecords")

        list_records = root.find("oai:ListRecords", _OAI_NS)
        if list_records is None:
            break

        for record in list_records.findall("oai:record", _OAI_NS):
            parsed = parse_dc_record(record)
            if parsed:
                records.append(parsed)
            if len(records) >= max_records:
                break

        rt_elem = list_records.find("oai:resumptionToken", _OAI_NS)
        resumption_token = rt_elem.text if rt_elem is not None else None
        if not resumption_token:
            break
        time.sleep(1.0)

    return records


def normalize_archive_metadata(records_list: list[dict]) -> pd.DataFrame:
    """
    Normalize a heterogeneous list of metadata dicts into a uniform DataFrame.

    Applies consistent column naming, date parsing, and field standardization
    across records from different archive sources.

    Returns
    -------
    pd.DataFrame
        Normalized metadata with columns: oai_identifier, title, creator,
        subject, date_parsed, description, language, rights, source_archive.
    """
    df = pd.DataFrame(records_list)

    # Rename common alternative column names to standard names
    rename_map = {
        "creator": "creator",
        "author": "creator",
        "contributor": "creator",
        "subjects": "subject",
        "description": "description",
        "abstract": "description",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Attempt to parse dates
    if "date" in df.columns:
        df["date_parsed"] = pd.to_datetime(df["date"].str[:10], errors="coerce")
    else:
        df["date_parsed"] = pd.NaT

    # Fill missing columns with empty strings
    required_cols = ["oai_identifier", "title", "creator", "subject",
                     "description", "language", "rights"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df[required_cols + ["date_parsed"]].reset_index(drop=True)
```

---

## Example 1: Build a WWI Newspaper Corpus from Chronicling America

Construct a research corpus of US newspaper pages mentioning "World War" or
"Great War" published between 1914 and 1918, and save as CSV and JSONL.

```python
import pandas as pd
import json
import time

# ── Search parameters ─────────────────────────────────────────────────────────
QUERY = '"world war" OR "great war" soldiers trenches'
DATE_RANGE = ("19140101", "19181111")   # Armistice Day
MAX_PAGES = 200

print("Searching Chronicling America for WWI newspaper articles...")
corpus_df = build_chronicling_corpus(
    query=QUERY,
    date_range=DATE_RANGE,
    max_pages=MAX_PAGES,
    download_ocr=True,
)

print(f"\nCorpus size: {len(corpus_df)} pages")
print(f"Date range: {corpus_df['date'].min()} — {corpus_df['date'].max()}")
print(f"States covered: {corpus_df['state'].unique().tolist()}")

# ── Basic statistics ──────────────────────────────────────────────────────────
print("\nTop 10 newspapers by article count:")
print(corpus_df["title"].value_counts().head(10))

# ── Save as CSV ───────────────────────────────────────────────────────────────
corpus_df.to_csv("wwi_newspaper_corpus.csv", index=False, encoding="utf-8")
print("\nSaved to wwi_newspaper_corpus.csv")

# ── Save as JSONL for NLP pipelines ──────────────────────────────────────────
with open("wwi_newspaper_corpus.jsonl", "w", encoding="utf-8") as fh:
    for _, row in corpus_df.iterrows():
        fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
print("Saved to wwi_newspaper_corpus.jsonl")

# ── Show a sample article snippet ────────────────────────────────────────────
sample = corpus_df[corpus_df["ocr_text"].str.len() > 200].iloc[0]
print(f"\nSample article from: {sample['title']} ({sample['date']})")
print(f"URL: {sample['page_url']}")
print(f"Text preview:\n{sample['ocr_text'][:500]}...")
```

---

## Example 2: Search Europeana for Medieval Manuscripts and Download Metadata

Query Europeana for medieval manuscript images, harvest detailed metadata, and
save a structured catalogue for archival research.

```python
import os
import json
import pandas as pd

# ── Ensure API key is set ─────────────────────────────────────────────────────
# Run before executing: export EUROPEANA_API_KEY="your_key_here"
# Register free at: https://apis.europeana.eu/

QUERY = 'medieval manuscript illuminated'
MEDIA_TYPE = "IMAGE"
TOTAL_RESULTS = 150

print(f"Searching Europeana for: {QUERY!r}")
print(f"Media type: {MEDIA_TYPE}, target: {TOTAL_RESULTS} records\n")

# ── Paginate and collect ──────────────────────────────────────────────────────
raw_items = list(
    paginate_europeana(
        query=QUERY,
        total=TOTAL_RESULTS,
        page_size=100,
        media_type=MEDIA_TYPE,
        reusability="open",
    )
)
print(f"Retrieved {len(raw_items)} items from Europeana.")

# ── Extract relevant metadata fields ─────────────────────────────────────────
def extract_europeana_metadata(item: dict) -> dict:
    """Extract structured metadata from a single Europeana search result item."""
    def first(val):
        if isinstance(val, list):
            return val[0] if val else ""
        return val or ""

    return {
        "europeana_id": item.get("id", ""),
        "title": first(item.get("title")),
        "creator": first(item.get("dcCreator")),
        "description": first(item.get("dcDescriptionLangAware", {}).get("en", [])),
        "date": first(item.get("year")),
        "provider": first(item.get("dataProvider")),
        "country": first(item.get("country")),
        "language": first(item.get("language")),
        "type": item.get("type", ""),
        "rights": first(item.get("rights")),
        "thumbnail_url": item.get("edmPreview", [""])[0] if item.get("edmPreview") else "",
        "guid": item.get("guid", ""),
    }


metadata_records = [extract_europeana_metadata(item) for item in raw_items]
manuscripts_df = pd.DataFrame(metadata_records)

# ── Filter: retain records with titles and thumbnails ────────────────────────
manuscripts_df = manuscripts_df[
    (manuscripts_df["title"].str.len() > 0) &
    (manuscripts_df["thumbnail_url"].str.len() > 0)
].reset_index(drop=True)

print(f"\nFiltered to {len(manuscripts_df)} records with titles and thumbnails.")
print(f"Countries: {manuscripts_df['country'].value_counts().head(8).to_dict()}")
print(f"Providers: {manuscripts_df['provider'].value_counts().head(5).to_dict()}")

# ── Save full metadata catalogue ─────────────────────────────────────────────
manuscripts_df.to_csv("europeana_medieval_manuscripts.csv", index=False, encoding="utf-8")

with open("europeana_medieval_manuscripts.json", "w", encoding="utf-8") as fh:
    json.dump(manuscripts_df.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)

print("\nSaved catalogue to europeana_medieval_manuscripts.csv and .json")

# ── Show sample entries ───────────────────────────────────────────────────────
print("\nSample records:")
print(manuscripts_df[["title", "creator", "date", "country", "provider"]].head(5).to_string())

# ── Optionally download thumbnails ───────────────────────────────────────────
DOWNLOAD_THUMBNAILS = False  # set True to download
if DOWNLOAD_THUMBNAILS:
    import os
    thumb_dir = "europeana_thumbnails"
    os.makedirs(thumb_dir, exist_ok=True)
    for _, row in manuscripts_df.head(20).iterrows():
        if row["thumbnail_url"]:
            safe_id = row["europeana_id"].replace("/", "_").lstrip("_")
            path = os.path.join(thumb_dir, f"{safe_id}.jpg")
            try:
                resp = requests.get(row["thumbnail_url"], timeout=15)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                time.sleep(0.2)
            except Exception as exc:
                print(f"  Failed to download {safe_id}: {exc}")
    print(f"Downloaded thumbnails to {thumb_dir}/")
```

---

## Notes and Best Practices

- **Europeana API rate limits**: The free tier allows up to 5 requests per
  second. The `paginate_europeana` function includes a 0.2-second delay between
  requests. For large-scale harvesting (>10,000 records) register for a higher
  quota.
- **Chronicling America robots.txt**: The Library of Congress permits automated
  access for research. Include `User-Agent` headers identifying your project and
  respect the recommended 0.5-second delay between requests.
- **OAI-PMH resumption tokens**: Tokens expire (typically after 30–60 minutes).
  If harvesting millions of records, checkpoint your progress by saving the last
  `resumptionToken` to disk.
- **Character encoding**: Historical OCR text may contain unusual Unicode
  characters and encoding artifacts. Always open output files with
  `encoding="utf-8"` and use `errors="replace"` when reading legacy files.
- **Internet Archive bulk downloads**: The `internetarchive` Python library
  handles concurrent downloads. For very large corpora, use
  `ia download --itemlist identifiers.txt --formats=DjVuTXT` from the CLI.
- **API key security**: Never commit API keys to version control. Use
  environment variables or a secrets manager.

---

## Dependencies Installation

```bash
pip install requests>=2.31.0 pandas>=2.0.0 beautifulsoup4>=4.12.0 \
            lxml>=4.9.0 tqdm>=4.65.0 internetarchive>=3.3.0
```
