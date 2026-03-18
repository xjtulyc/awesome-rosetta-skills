---
name: tei-xml
description: >
  Use this Skill to encode historical documents in TEI XML P5: critical apparatus,
  named entity markup (persName/placeName), XPath analysis with lxml, and XSLT transformation.
tags:
  - history
  - TEI-XML
  - digital-edition
  - named-entity
  - XSLT
  - digital-humanities
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
    - lxml>=4.9
    - beautifulsoup4>=4.12
    - pandas>=1.5
last_updated: "2026-03-18"
status: stable
---

# TEI XML P5 Encoding for Historical Editions

> **TL;DR** — Encode, analyze, and transform TEI P5 XML documents: build teiHeader metadata,
> mark up named entities with VIAF authority links, model critical apparatus variant readings,
> run XPath queries with lxml, extract entity frequency tables, and transform to HTML or plain
> text via XSLT/Saxon-HE.

---

## When to Use

Use this Skill when you need to:

- Create a TEI P5 digital edition of a historical text with scholarly apparatus
- Mark up personal names, place names, and organization names with authority record links
- Represent multiple manuscript witnesses using `<app>`, `<lem>`, and `<rdg>` elements
- Run XPath queries to extract named entities, count occurrences, and build frequency tables
- Transform TEI XML to HTML reading text or plain text for downstream NLP

Do **not** use this Skill for:

- Lightweight bibliographic XML (use Dublin Core or MODS)
- Real-time XML database queries at scale (use eXist-db or BaseX)
- Automatic NER tagging without human review (TEI requires scholarly validation)

---

## Background

TEI (Text Encoding Initiative) P5 is the international standard XML vocabulary for
encoding humanities texts. Its hierarchical document model covers:

| Element | Purpose |
|---|---|
| `<teiHeader>` | Bibliographic and editorial metadata |
| `<text><body>` | The encoded text content |
| `<div>` | Textual divisions (chapter, section, act) |
| `<p>`, `<lg>`, `<l>` | Paragraph, line group, line |
| `<lb>`, `<pb>`, `<fw>` | Line break, page break, running header/footer |
| `<persName>`, `<placeName>`, `<orgName>` | Named entity markup with `@ref` to authority |
| `<abbr>`, `<expan>`, `<choice>` | Abbreviation and expansion pairs |
| `<app>`, `<lem>`, `<rdg>` | Critical apparatus: lemma + variant readings |
| `@wit` | Witness sigil on `<rdg>` elements |

The VIAF (Virtual International Authority File) and Wikidata provide stable URIs for
persons and places. Linking `@ref="https://viaf.org/viaf/22146956/"` disambiguates
historical figures across editions.

XPath 1.0/2.0 queries via lxml's `xpath()` method allow systematic extraction and
statistical analysis of encoded features without manual text parsing.

---

## Environment Setup

```bash
# Create Python environment
conda create -n tei-xml python=3.11 -y
conda activate tei-xml

# Install Python dependencies
pip install "lxml>=4.9" "beautifulsoup4>=4.12" "pandas>=1.5"

# Verify
python -c "import lxml; print(lxml.__version__)"

# Saxon-HE for XSLT 2.0/3.0 transforms (optional, Java required)
# Download from: https://www.saxonica.com/download/java.xml
# Then invoke via subprocess: java -jar saxon-he-*.jar -s:input.xml -xsl:style.xsl

# Verify Java for XSLT transforms
java -version
```

---

## Core Workflow

### Step 1 — Build a TEI P5 Document Structure

```python
from lxml import etree
from typing import Optional


TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"
TEI = f"{{{TEI_NS}}}"


def tei_element(tag: str, attrib: dict = None, text: str = None) -> etree._Element:
    """
    Create a TEI-namespaced element with optional attributes and text.

    Args:
        tag:    Local element name (without namespace prefix).
        attrib: Optional dict of attribute name → value.
        text:   Optional text content for the element.

    Returns:
        lxml Element node in the TEI namespace.
    """
    el = etree.Element(f"{TEI_NS}{tag}", nsmap={"tei": TEI_NS, None: TEI_NS})
    if attrib:
        for k, v in attrib.items():
            el.set(k, v)
    if text:
        el.text = text
    return el


def build_tei_header(
    title: str,
    author: str,
    editor: str,
    publisher: str,
    date: str,
    source_description: str,
    language: str = "la",
) -> etree._Element:
    """
    Construct a minimal TEI P5 teiHeader.

    Args:
        title:              Work title.
        author:             Original author's name.
        editor:             Digital edition editor name.
        publisher:          Publishing institution.
        date:               Publication year string.
        source_description: Description of the base manuscript/print.
        language:           ISO 639-1 language code.

    Returns:
        lxml Element for the complete teiHeader.
    """
    NS = TEI_NS

    def el(tag, **kwargs):
        return etree.SubElement  # convenience alias not used directly

    header = etree.Element(f"{{{NS}}}teiHeader")

    # fileDesc
    file_desc = etree.SubElement(header, f"{{{NS}}}fileDesc")

    title_stmt = etree.SubElement(file_desc, f"{{{NS}}}titleStmt")
    etree.SubElement(title_stmt, f"{{{NS}}}title").text = title
    author_el = etree.SubElement(title_stmt, f"{{{NS}}}author")
    author_el.text = author
    resp_stmt = etree.SubElement(title_stmt, f"{{{NS}}}respStmt")
    resp = etree.SubElement(resp_stmt, f"{{{NS}}}resp")
    resp.text = "Digital edition created by"
    etree.SubElement(resp_stmt, f"{{{NS}}}name").text = editor

    pub_stmt = etree.SubElement(file_desc, f"{{{NS}}}publicationStmt")
    etree.SubElement(pub_stmt, f"{{{NS}}}publisher").text = publisher
    etree.SubElement(pub_stmt, f"{{{NS}}}date").text = date
    availability = etree.SubElement(pub_stmt, f"{{{NS}}}availability",
                                    attrib={"status": "free"})
    etree.SubElement(availability, f"{{{NS}}}licence",
                     attrib={"target": "https://creativecommons.org/licenses/by/4.0/"}).text = "CC-BY 4.0"

    source_desc = etree.SubElement(file_desc, f"{{{NS}}}sourceDesc")
    etree.SubElement(source_desc, f"{{{NS}}}p").text = source_description

    # encodingDesc
    enc_desc = etree.SubElement(header, f"{{{NS}}}encodingDesc")
    project_desc = etree.SubElement(enc_desc, f"{{{NS}}}projectDesc")
    etree.SubElement(project_desc, f"{{{NS}}}p").text = (
        "Encoded following TEI P5 guidelines. Named entities linked to VIAF and Wikidata."
    )

    # profileDesc
    profile_desc = etree.SubElement(header, f"{{{NS}}}profileDesc")
    lang_usage = etree.SubElement(profile_desc, f"{{{NS}}}langUsage")
    etree.SubElement(lang_usage, f"{{{NS}}}language",
                     attrib={"ident": language}).text = f"Language: {language}"

    return header


def build_tei_document(
    header: etree._Element,
    paragraphs: list[str],
) -> etree._Element:
    """
    Assemble a complete TEI P5 document from a teiHeader and plain text paragraphs.

    Args:
        header:     teiHeader element from build_tei_header().
        paragraphs: List of paragraph text strings.

    Returns:
        Root <TEI> element with complete document tree.
    """
    NS = TEI_NS
    root = etree.Element(
        f"{{{NS}}}TEI",
        attrib={f"{{{XML_NS}}}lang": "la"},
        nsmap={None: NS},
    )
    root.append(header)

    text_el = etree.SubElement(root, f"{{{NS}}}text")
    body = etree.SubElement(text_el, f"{{{NS}}}body")
    div = etree.SubElement(body, f"{{{NS}}}div", attrib={"type": "chapter", "n": "1"})

    for i, para_text in enumerate(paragraphs):
        p_el = etree.SubElement(div, f"{{{NS}}}p", attrib={"n": str(i + 1)})
        p_el.text = para_text

    return root
```

### Step 2 — XPath: Extract Named Entities with VIAF Frequencies

```python
import pandas as pd
from collections import Counter


def extract_named_entities(
    tei_root: etree._Element,
    entity_types: list[str] = None,
) -> pd.DataFrame:
    """
    Extract all named entity elements from a TEI document and tabulate frequencies.

    Queries XPath for persName, placeName, and orgName elements.
    Groups by normalized text + @ref URI.

    Args:
        tei_root:     Root lxml Element of the TEI document.
        entity_types: List of entity element local names to query.
                      Defaults to ["persName", "placeName", "orgName"].

    Returns:
        DataFrame with columns: entity_type, text, ref_uri, count, sorted by count desc.
    """
    if entity_types is None:
        entity_types = ["persName", "placeName", "orgName"]

    ns = {"tei": TEI_NS}
    records = []

    for etype in entity_types:
        elements = tei_root.xpath(
            f"//tei:{etype}", namespaces=ns
        )
        for el in elements:
            text = "".join(el.itertext()).strip()
            ref = el.get("ref", "")
            if text:
                records.append({
                    "entity_type": etype,
                    "text": text,
                    "ref_uri": ref,
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    freq_df = (
        df.groupby(["entity_type", "text", "ref_uri"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return freq_df


def extract_critical_apparatus(
    tei_root: etree._Element,
) -> pd.DataFrame:
    """
    Extract all critical apparatus entries from a TEI document.

    Returns a table of lemma + variant readings per witness.

    Args:
        tei_root: Root lxml Element of the TEI document.

    Returns:
        DataFrame with columns: location, lemma_text, witness, reading_text.
    """
    ns = {"tei": TEI_NS}
    apps = tei_root.xpath("//tei:app", namespaces=ns)

    records = []
    for i, app_el in enumerate(apps):
        # Get lemma text
        lem_els = app_el.xpath("tei:lem", namespaces=ns)
        lemma_text = "".join(lem_els[0].itertext()).strip() if lem_els else ""

        # Get all variant readings
        rdg_els = app_el.xpath("tei:rdg", namespaces=ns)
        for rdg in rdg_els:
            wit = rdg.get("wit", "").replace("#", "").strip()
            rdg_text = "".join(rdg.itertext()).strip()
            records.append({
                "location": i + 1,
                "lemma_text": lemma_text,
                "witness": wit,
                "reading_text": rdg_text,
            })

    return pd.DataFrame(records)
```

### Step 3 — TEI to Plain Text and XSLT Transform

```python
import subprocess
from pathlib import Path


def tei_to_plain_text(tei_root: etree._Element) -> str:
    """
    Extract clean readable text from a TEI document, stripping all markup.

    Preserves paragraph breaks. Expands <abbr>/<expan> pairs to expanded form.
    Ignores teiHeader, apparatus elements (app/rdg), and metadata.

    Args:
        tei_root: Root lxml Element of the TEI document.

    Returns:
        Plain text string with paragraph separations.
    """
    ns = {"tei": TEI_NS}

    # Prefer expan over abbr in choice elements
    for choice in tei_root.xpath("//tei:choice", namespaces=ns):
        abbr_els = choice.xpath("tei:abbr", namespaces=ns)
        for a in abbr_els:
            a.getparent().remove(a)

    # Remove apparatus (rdg = variant readings, keep only lem)
    for rdg in tei_root.xpath("//tei:rdg", namespaces=ns):
        rdg.getparent().remove(rdg)

    # Remove header
    for header in tei_root.xpath("//tei:teiHeader", namespaces=ns):
        header.getparent().remove(header)

    # Extract paragraph texts
    paragraphs = tei_root.xpath("//tei:p | //tei:l", namespaces=ns)
    lines = []
    for p in paragraphs:
        text = "".join(p.itertext()).strip()
        if text:
            lines.append(text)

    return "\n\n".join(lines)


def transform_tei_xslt(
    input_xml_path: str,
    xsl_path: str,
    output_path: str,
    saxon_jar: str,
    extra_params: dict = None,
) -> None:
    """
    Apply an XSLT stylesheet to a TEI document using Saxon-HE (Java).

    Requires Java and the Saxon-HE JAR on disk. Download Saxon-HE from
    https://www.saxonica.com/download/java.xml

    Args:
        input_xml_path: Absolute path to the TEI XML file.
        xsl_path:       Absolute path to the XSLT 2.0/3.0 stylesheet.
        output_path:    Absolute path for the transformed output file.
        saxon_jar:      Absolute path to the saxon-he-*.jar file.
        extra_params:   Optional dict of XSLT parameter name → value pairs.
    """
    cmd = [
        "java", "-jar", saxon_jar,
        f"-s:{input_xml_path}",
        f"-xsl:{xsl_path}",
        f"-o:{output_path}",
    ]
    if extra_params:
        for k, v in extra_params.items():
            cmd.append(f"{k}={v}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Saxon-HE XSLT failed:\n{result.stderr}")
    print(f"XSLT transform complete. Output: {output_path}")
```

---

## Advanced Usage

### Adding Named Entity Markup with VIAF References

```python
def add_named_entity_markup(
    p_element: etree._Element,
    entity_spans: list[dict],
) -> None:
    """
    Insert <persName> or <placeName> markup into a paragraph element.

    This is a simplified in-place markup inserter. In production, use a
    standoff annotation approach or CATMA for complex overlapping markup.

    Args:
        p_element:    A <p> element whose text contains the entities.
        entity_spans: List of dicts with: text (str), entity_type (str),
                      viaf_ref (str, optional), wikidata_ref (str, optional).
    """
    NS = TEI_NS
    original_text = p_element.text or ""

    for span in entity_spans:
        entity_text = span["text"]
        entity_type = span.get("entity_type", "persName")
        ref = span.get("viaf_ref") or span.get("wikidata_ref", "")

        if entity_text not in original_text:
            continue

        # Split text at entity occurrence and wrap in element
        idx = original_text.find(entity_text)
        before = original_text[:idx]
        after = original_text[idx + len(entity_text):]

        p_element.text = before
        attrib = {}
        if ref:
            attrib["ref"] = ref
        entity_el = etree.SubElement(p_element, f"{{{NS}}}{entity_type}", attrib=attrib)
        entity_el.text = entity_text
        entity_el.tail = after
        original_text = after  # continue searching in remaining text
```

### Serialize TEI to File

```python
def serialize_tei(tei_root: etree._Element, output_path: str) -> None:
    """
    Serialize a TEI lxml tree to a well-formed XML file with XML declaration.

    Args:
        tei_root:    Root <TEI> element.
        output_path: Absolute path to write the XML file.
    """
    tree = etree.ElementTree(tei_root)
    tree.write(
        output_path,
        encoding="UTF-8",
        xml_declaration=True,
        pretty_print=True,
    )
    print(f"TEI document written to {output_path}")
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `XPathEvalError: Undefined namespace prefix` | Missing `ns` dict in `xpath()` call | Pass `namespaces={"tei": "http://www.tei-c.org/ns/1.0"}` |
| Empty XPath result on valid document | Default namespace not declared | Ensure `nsmap={None: TEI_NS}` on root element |
| Saxon-HE `java.lang.OutOfMemoryError` | Large XML document | Add `-Xmx2g` to Java command |
| `lxml.etree.XMLSyntaxError` on load | Malformed TEI XML | Validate against TEI All schema: `xmllint --schema tei_all.xsd file.xml` |
| Named entity `@ref` not preserved on serialize | Attribute lost in tree manipulation | Use `el.set("ref", value)` before appending to tree |
| Long `<app>` chain breaks text flow | `rdg` elements between text nodes | Use `itertext()` carefully; strip apparatus before plain text extraction |

---

## External Resources

- TEI P5 Guidelines: <https://tei-c.org/release/doc/tei-p5-doc/en/html/>
- VIAF Authority File: <https://viaf.org/>
- Wikidata: <https://www.wikidata.org/>
- Saxon-HE XSLT processor: <https://www.saxonica.com/download/java.xml>
- lxml XPath documentation: <https://lxml.de/xpathxslt.html>
- TEI Stylesheets (TEI to HTML/LaTeX): <https://github.com/TEIC/Stylesheets>
- CATMA collaborative annotation: <https://catma.de/>

---

## Examples

### Example 1 — Build and Analyze a TEI Edition

```python
# Build a minimal TEI P5 document with named entities and query it
header = build_tei_header(
    title="Epistola ad Leonem Papam",
    author="Erasmus Roterodamus",
    editor="Jane Scholar",
    publisher="Digital Humanities Lab",
    date="2026",
    source_description="Based on Basel 1519 first edition, Bibliothèque nationale de France Res-Z-2188",
    language="la",
)

paragraphs = [
    "Erasmus Roterodamus Leoni Pontifici Maximo salutem.",
    "Romam ipsam, urbem omnium gentium dominam, saepe miratus sum.",
]

tei_root = build_tei_document(header, paragraphs)

# Add entity markup to first paragraph
body_paras = tei_root.xpath("//tei:p", namespaces={"tei": TEI_NS})
if body_paras:
    add_named_entity_markup(body_paras[0], [
        {"text": "Erasmus Roterodamus", "entity_type": "persName",
         "viaf_ref": "https://viaf.org/viaf/51672454/"},
        {"text": "Leoni", "entity_type": "persName",
         "viaf_ref": "https://viaf.org/viaf/22146956/"},
    ])

# Extract named entities
freq_df = extract_named_entities(tei_root)
print("Named entity frequencies:")
print(freq_df.to_string(index=False))

# Serialize
serialize_tei(tei_root, "/data/output/erasmus_epistola.xml")
```

### Example 2 — Critical Apparatus Extraction

```python
# Parse an existing TEI critical edition and extract apparatus table
parser = etree.XMLParser(remove_blank_text=True)
tree = etree.parse("/data/editions/livy_ab_urbe_condita.xml", parser)
root = tree.getroot()

apparatus_df = extract_critical_apparatus(root)
print(f"Total apparatus entries: {len(apparatus_df)}")
print("\nFirst 10 variant readings:")
print(apparatus_df.head(10).to_string(index=False))

# Most divergent witnesses
witness_counts = apparatus_df["witness"].value_counts()
print("\nReadings per manuscript witness:")
print(witness_counts)

# Export to CSV for collation analysis
apparatus_df.to_csv("/data/output/livy_apparatus.csv", index=False)

# Extract plain text for NLP
plain = tei_to_plain_text(root)
print(f"\nPlain text length: {len(plain)} characters")
with open("/data/output/livy_plain.txt", "w", encoding="utf-8") as f:
    f.write(plain)
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — TEI P5 builder, XPath entity extraction, critical apparatus, XSLT transform, plain text pipeline |
