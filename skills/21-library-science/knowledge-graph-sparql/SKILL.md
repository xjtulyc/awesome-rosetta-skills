---
name: knowledge-graph-sparql
description: Knowledge graph construction, SPARQL querying, and entity linking for library and research data management using RDF and Wikidata.
tags:
  - knowledge-graph
  - sparql
  - rdf
  - wikidata
  - linked-data
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
    - rdflib>=6.3
    - requests>=2.31
    - pandas>=2.0
    - networkx>=3.2
    - matplotlib>=3.7
    - numpy>=1.24
last_updated: "2026-03-17"
status: stable
---

# Knowledge Graph and SPARQL Analysis

## When to Use This Skill

Use this skill when you need to:
- Build RDF knowledge graphs from structured or tabular data
- Query Wikidata, DBpedia, or local triple stores with SPARQL
- Perform entity linking and disambiguation using identifiers
- Construct ontologies and apply reasoning rules
- Visualize knowledge graph topology and relationship patterns
- Integrate heterogeneous data sources through linked data principles
- Answer complex multi-hop queries over scholarly metadata

**Trigger keywords**: SPARQL, RDF, knowledge graph, triple store, Wikidata, DBpedia, ontology, OWL, entity linking, linked data, URI, predicate, subject, object, triples, CONSTRUCT query, federated query, LOD, schema.org, FOAF, Dublin Core, library catalog, authority file.

## Background & Key Concepts

### RDF Data Model

The Resource Description Framework (RDF) represents information as triples:

$$\langle \text{subject},\ \text{predicate},\ \text{object} \rangle$$

For example: `<Person:Einstein> <property:birthPlace> <Place:Ulm>`. Nodes are either URIs (resources) or literals (strings, numbers). Collections of triples form a directed labeled graph.

### SPARQL Query Patterns

A basic SPARQL SELECT query:

```sparql
SELECT ?author ?title
WHERE {
  ?paper a schema:ScholarlyArticle ;
         schema:author ?author ;
         schema:name ?title .
  FILTER(?year > 2020)
}
LIMIT 100
```

Key query types: SELECT (tabular), CONSTRUCT (new graph), ASK (boolean), DESCRIBE (entity description).

### Graph Patterns

- **Optional patterns**: `OPTIONAL { ?s :p ?o }` — left outer join
- **Filters**: `FILTER(LANG(?label) = "en")`
- **Aggregates**: `GROUP BY`, `COUNT`, `SUM`, `AVG`
- **Property paths**: `?a :knows+ ?b` — transitive closure

### OWL Reasoning

OWL (Web Ontology Language) adds axioms: `owl:subClassOf`, `owl:equivalentClass`, `owl:inverseOf`. A reasoner can infer new triples: if `A subClassOf B` and `x type A`, then `x type B`.

### Entity Linking

Map text mentions to canonical URIs using:
1. String similarity matching
2. Context disambiguation (co-occurring entities)
3. Lookup in authority files (VIAF, GND, ORCID, ROR)

## Environment Setup

```bash
pip install rdflib>=6.3 requests>=2.31 pandas>=2.0 networkx>=3.2 \
            matplotlib>=3.7 numpy>=1.24
```

```python
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, FOAF, DCTERMS
import requests
import pandas as pd
print("Knowledge graph environment ready")
```

## Core Workflow

### Step 1: Build an RDF Knowledge Graph with rdflib

```python
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, FOAF, DCTERMS
import pandas as pd
import numpy as np

# -----------------------------------------------------------------
# Define namespaces
# -----------------------------------------------------------------
EX  = Namespace("http://example.org/research/")
SCHEMA = Namespace("https://schema.org/")
BIBO   = Namespace("http://purl.org/ontology/bibo/")

# -----------------------------------------------------------------
# Create a graph with sample scholarly data
# -----------------------------------------------------------------
g = Graph()
g.bind("ex",     EX)
g.bind("schema", SCHEMA)
g.bind("foaf",   FOAF)
g.bind("bibo",   BIBO)
g.bind("dcterms",DCTERMS)

# --- Authors ---
authors_data = [
    ("A001", "Alice Zhang",    "Tsinghua University", "0000-0001-2345-6789"),
    ("A002", "Bob Smith",      "MIT",                 "0000-0002-3456-7890"),
    ("A003", "Carol Johnson",  "Oxford University",   "0000-0003-4567-8901"),
    ("A004", "David Li",       "Tsinghua University", "0000-0004-5678-9012"),
]

for aid, name, affil, orcid in authors_data:
    author_uri = EX[f"author/{aid}"]
    inst_uri   = EX[f"institution/{affil.replace(' ', '_')}"]

    g.add((author_uri, RDF.type,    FOAF.Person))
    g.add((author_uri, FOAF.name,   Literal(name)))
    g.add((author_uri, SCHEMA.affiliation, inst_uri))
    g.add((author_uri, SCHEMA.identifier,
           Literal(f"https://orcid.org/{orcid}", datatype=XSD.anyURI)))
    g.add((inst_uri,   RDF.type,    SCHEMA.Organization))
    g.add((inst_uri,   SCHEMA.name, Literal(affil)))

# --- Papers ---
papers_data = [
    ("P001", "Deep Learning for Climate", 2022, ["A001", "A002"], ["C01", "C02"]),
    ("P002", "Transformer Models Survey", 2021, ["A002", "A003"], ["C02", "C03"]),
    ("P003", "Graph Neural Networks",     2023, ["A001", "A004"], ["C01", "C04"]),
    ("P004", "Federated Learning Privacy",2022, ["A003", "A004"], ["C03", "C05"]),
]
concepts_data = {
    "C01": "Deep Learning", "C02": "Natural Language Processing",
    "C03": "Privacy",       "C04": "Graph Theory",
    "C05": "Federated Learning",
}

for pid, title, year, author_ids, concept_ids in papers_data:
    paper_uri = EX[f"paper/{pid}"]
    g.add((paper_uri, RDF.type,           BIBO.AcademicArticle))
    g.add((paper_uri, DCTERMS.title,      Literal(title)))
    g.add((paper_uri, DCTERMS.date,
           Literal(str(year), datatype=XSD.gYear)))
    for aid in author_ids:
        g.add((paper_uri, DCTERMS.creator, EX[f"author/{aid}"]))
    for cid in concept_ids:
        concept_uri = EX[f"concept/{cid}"]
        g.add((paper_uri, SCHEMA.about, concept_uri))
        g.add((concept_uri, RDF.type,    SCHEMA.DefinedTerm))
        g.add((concept_uri, SCHEMA.name, Literal(concepts_data[cid])))

# --- Citation relationships ---
citations = [("P002", "P001"), ("P003", "P001"), ("P003", "P002"), ("P004", "P002")]
for citing, cited in citations:
    g.add((EX[f"paper/{citing}"], BIBO.cites, EX[f"paper/{cited}"]))

print(f"Graph has {len(g)} triples")
print(f"Unique subjects: {len(set(g.subjects()))}")

# Serialize to Turtle format
turtle_str = g.serialize(format="turtle")
print("\n--- First 1000 chars of Turtle serialization ---")
print(turtle_str[:1000])

# Save graph
g.serialize("research_graph.ttl", format="turtle")
print("\nGraph saved to research_graph.ttl")
```

### Step 2: SPARQL Queries Over the Knowledge Graph

```python
from rdflib import Graph
import pandas as pd

# Reload the graph
g = Graph()
g.parse("research_graph.ttl", format="turtle")

# -----------------------------------------------------------------
# Query 1: All papers with author names and years
# -----------------------------------------------------------------
q1 = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX bibo: <http://purl.org/ontology/bibo/>

SELECT ?title ?author_name ?year
WHERE {
  ?paper a bibo:AcademicArticle ;
         dcterms:title ?title ;
         dcterms:date  ?year ;
         dcterms:creator ?author .
  ?author foaf:name ?author_name .
}
ORDER BY ?year ?title
"""
results1 = g.query(q1)
df1 = pd.DataFrame(results1, columns=["title", "author_name", "year"])
print("=== Query 1: Papers and Authors ===")
print(df1.to_string(index=False))

# -----------------------------------------------------------------
# Query 2: Co-author pairs
# -----------------------------------------------------------------
q2 = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX bibo: <http://purl.org/ontology/bibo/>

SELECT DISTINCT ?author1 ?author2 (COUNT(?paper) AS ?collab_count)
WHERE {
  ?paper a bibo:AcademicArticle ;
         dcterms:creator ?a1 ;
         dcterms:creator ?a2 .
  ?a1 foaf:name ?author1 .
  ?a2 foaf:name ?author2 .
  FILTER(?a1 < ?a2)
}
GROUP BY ?author1 ?author2
ORDER BY DESC(?collab_count)
"""
results2 = g.query(q2)
df2 = pd.DataFrame(results2, columns=["author1", "author2", "collaborations"])
print("\n=== Query 2: Co-author Pairs ===")
print(df2.to_string(index=False))

# -----------------------------------------------------------------
# Query 3: Citation network (who cites whom via paper author)
# -----------------------------------------------------------------
q3 = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX bibo:   <http://purl.org/ontology/bibo/>
PREFIX foaf:   <http://xmlns.com/foaf/0.1/>

SELECT ?citing_paper_title ?cited_paper_title
WHERE {
  ?citing_paper bibo:cites ?cited_paper .
  ?citing_paper dcterms:title ?citing_paper_title .
  ?cited_paper  dcterms:title ?cited_paper_title .
}
"""
results3 = g.query(q3)
df3 = pd.DataFrame(results3, columns=["citing", "cited"])
print("\n=== Query 3: Citation Relationships ===")
print(df3.to_string(index=False))

# -----------------------------------------------------------------
# Query 4: Concept co-occurrence in papers
# -----------------------------------------------------------------
q4 = """
PREFIX schema:  <https://schema.org/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX bibo:    <http://purl.org/ontology/bibo/>

SELECT ?c1_name ?c2_name (COUNT(?paper) AS ?co_count)
WHERE {
  ?paper a bibo:AcademicArticle ;
         schema:about ?c1 ;
         schema:about ?c2 .
  ?c1 schema:name ?c1_name .
  ?c2 schema:name ?c2_name .
  FILTER(?c1 < ?c2)
}
GROUP BY ?c1_name ?c2_name
HAVING (?co_count >= 1)
ORDER BY DESC(?co_count)
"""
results4 = g.query(q4)
df4 = pd.DataFrame(results4, columns=["concept1", "concept2", "co_count"])
print("\n=== Query 4: Concept Co-occurrence ===")
print(df4.to_string(index=False))
```

### Step 3: Wikidata SPARQL Queries

```python
import requests
import pandas as pd
import time
import os

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

def wikidata_sparql(query, timeout=30):
    """Execute a SPARQL query against Wikidata.

    Args:
        query: SPARQL query string
        timeout: request timeout in seconds
    Returns:
        DataFrame with results, or synthetic fallback on error
    """
    headers = {
        "User-Agent": "ResearchBot/1.0 (academic research)",
        "Accept": "application/sparql-results+json",
    }
    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        data = response.json()
        vars_ = data["head"]["vars"]
        rows = []
        for binding in data["results"]["bindings"]:
            row = {v: binding.get(v, {}).get("value", "") for v in vars_}
            rows.append(row)
        return pd.DataFrame(rows), True
    except Exception as e:
        print(f"Wikidata query failed: {e}")
        return pd.DataFrame(), False

# Query: Top universities founded before 1600 with their location
query_universities = """
SELECT ?university ?universityLabel ?founded ?countryLabel
WHERE {
  ?university wdt:P31 wd:Q3918 .         # instance of: university
  ?university wdt:P571 ?founded .         # founded date
  ?university wdt:P17  ?country .         # country
  FILTER(YEAR(?founded) < 1600)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?founded
LIMIT 20
"""

df_univ, success = wikidata_sparql(query_universities)

if success and len(df_univ) > 0:
    print("=== Historical Universities (from Wikidata) ===")
    print(df_univ[["universityLabel", "founded", "countryLabel"]].to_string(index=False))
else:
    # Synthetic fallback
    print("=== Historical Universities (synthetic data) ===")
    synthetic = pd.DataFrame({
        "universityLabel": ["University of Bologna", "University of Oxford",
                            "University of Cambridge", "University of Salamanca",
                            "University of Paris"],
        "founded": ["1088", "1096", "1209", "1218", "1257"],
        "countryLabel": ["Italy", "UK", "UK", "Spain", "France"],
    })
    print(synthetic.to_string(index=False))

# Query: Nobel laureates in physics with their alma mater
query_nobel = """
SELECT ?person ?personLabel ?year ?universityLabel
WHERE {
  ?person wdt:P166 wd:Q38104 .            # award: Nobel Prize in Physics
  ?person p:P166 ?statement .
  ?statement pq:P585 ?year .
  OPTIONAL { ?person wdt:P69 ?university . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY DESC(?year)
LIMIT 20
"""

df_nobel, success = wikidata_sparql(query_nobel)
if success and len(df_nobel) > 0:
    print("\n=== Recent Nobel Physics Laureates ===")
    print(df_nobel[["personLabel", "year"]].head(10).to_string(index=False))
else:
    print("\n(Wikidata offline – skipping Nobel query)")

# -----------------------------------------------------------------
# Entity lookup by ORCID
# -----------------------------------------------------------------
def lookup_researcher_wikidata(orcid_id):
    """Find Wikidata entity for a researcher by ORCID.

    Args:
        orcid_id: ORCID string (e.g. "0000-0002-1825-0097")
    Returns:
        dict with Wikidata QID and labels, or empty dict on failure
    """
    query = f"""
    SELECT ?person ?personLabel ?employerLabel
    WHERE {{
      ?person wdt:P496 "{orcid_id}" .
      OPTIONAL {{ ?person wdt:P108 ?employer . }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}
    """
    df, ok = wikidata_sparql(query)
    if ok and len(df) > 0:
        return df.iloc[0].to_dict()
    return {}

# Example ORCID (Tim Berners-Lee: 0000-0003-1279-3709)
researcher = lookup_researcher_wikidata("0000-0003-1279-3709")
if researcher:
    print(f"\nWikidata entity found: {researcher}")
else:
    print("\nEntity lookup: API offline or ORCID not found in Wikidata")
```

## Advanced Usage

### Federated SPARQL Query

```sparql
-- Example: Link local data with Wikidata via federated query
-- (Run this in a SPARQL endpoint that supports SERVICE)

PREFIX owl:    <http://www.w3.org/2002/07/owl#>
PREFIX schema: <https://schema.org/>

SELECT ?localAuthor ?wd_label ?wd_birthdate
WHERE {
  # Local graph
  ?localAuthor a schema:Person ;
               owl:sameAs ?wikidataURI .

  # Federated to Wikidata
  SERVICE <https://query.wikidata.org/sparql> {
    ?wikidataURI wdt:P569 ?wd_birthdate .
    SERVICE wikibase:label {
      bd:serviceParam wikibase:language "en" .
      ?wikidataURI rdfs:label ?wd_label .
    }
  }
}
```

### OWL Reasoning with rdflib + OWL-RL

```python
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, OWL

def apply_rdfs_closure(g):
    """Apply basic RDFS inference rules (subClassOf, subPropertyOf).

    Args:
        g: rdflib Graph
    Returns:
        g: graph with inferred triples added
    """
    changed = True
    while changed:
        changed = False
        new_triples = set()

        # Rule: subClassOf transitivity
        for s, _, sc in g.triples((None, RDFS.subClassOf, None)):
            for _, _, sc2 in g.triples((sc, RDFS.subClassOf, None)):
                t = (s, RDFS.subClassOf, sc2)
                if t not in g:
                    new_triples.add(t)

        # Rule: type propagation via subClassOf
        for s, _, type_ in g.triples((None, RDF.type, None)):
            for _, _, superclass in g.triples((type_, RDFS.subClassOf, None)):
                t = (s, RDF.type, superclass)
                if t not in g:
                    new_triples.add(t)

        for t in new_triples:
            g.add(t)
            changed = True

    return g

# Example: Add subClassOf axioms and infer types
g_owl = Graph()
EX2 = Namespace("http://example.org/")
g_owl.bind("ex", EX2)

# Ontology axioms
g_owl.add((EX2.Professor, RDFS.subClassOf, EX2.AcademicStaff))
g_owl.add((EX2.AcademicStaff, RDFS.subClassOf, EX2.Person))
g_owl.add((EX2.alice, RDF.type, EX2.Professor))

print(f"Before inference: {len(g_owl)} triples")
g_owl = apply_rdfs_closure(g_owl)
print(f"After RDFS closure: {len(g_owl)} triples")

for s, p, o in g_owl.triples((EX2.alice, RDF.type, None)):
    print(f"  alice rdf:type {o.split('/')[-1]}")
```

### Graph Visualization with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph

def rdf_to_networkx(g, predicate_filter=None):
    """Convert rdflib Graph to NetworkX DiGraph for visualization.

    Args:
        g: rdflib.Graph
        predicate_filter: optional set of predicate URIs to include
    Returns:
        G: networkx DiGraph
    """
    G = nx.DiGraph()
    for s, p, o in g:
        s_label = str(s).split("/")[-1].split("#")[-1]
        p_label = str(p).split("/")[-1].split("#")[-1]
        o_label = str(o).split("/")[-1].split("#")[-1]

        if predicate_filter and str(p) not in predicate_filter:
            continue
        G.add_edge(s_label, o_label, predicate=p_label)
    return G

g2 = Graph()
g2.parse("research_graph.ttl", format="turtle")

from rdflib.namespace import DCTERMS
G_nx = rdf_to_networkx(g2, predicate_filter={
    str(DCTERMS.creator),
    "http://purl.org/ontology/bibo/cites",
})

print(f"Visualization graph: {G_nx.number_of_nodes()} nodes, "
      f"{G_nx.number_of_edges()} edges")

fig, ax = plt.subplots(figsize=(10, 7))
pos = nx.spring_layout(G_nx, seed=42)
edge_labels = {(u, v): d["predicate"] for u, v, d in G_nx.edges(data=True)}
nx.draw_networkx(G_nx, pos, ax=ax, node_color="lightblue",
                 node_size=1200, font_size=7, arrows=True)
nx.draw_networkx_edge_labels(G_nx, pos, edge_labels, font_size=6, ax=ax)
ax.set_title("Research Knowledge Graph")
ax.axis("off")
plt.tight_layout()
plt.savefig("knowledge_graph_viz.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure saved: knowledge_graph_viz.png")
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| SPARQL parse error | Missing prefix declaration | Add `PREFIX` declarations at top of query |
| rdflib parse failure | Wrong serialization format | Check file extension; specify `format="turtle"` or `"xml"` |
| Wikidata timeout | Complex query or busy endpoint | Add `LIMIT`; break into smaller queries; use `OPTIONAL` not required triples |
| Empty SPARQL results | Namespace mismatch | Verify namespace URIs match what was used when inserting data |
| rdflib reasoning too slow | Large graph | Use `owlrl` package for OWL 2 RL reasoning instead |
| Unicode in literals | Non-ASCII characters | rdflib handles UTF-8 natively; ensure Literal() wraps the string |

## External Resources

- [W3C RDF Primer](https://www.w3.org/TR/rdf-primer/)
- [SPARQL 1.1 Query Language](https://www.w3.org/TR/sparql11-query/)
- [Wikidata Query Service](https://query.wikidata.org/)
- [rdflib documentation](https://rdflib.readthedocs.io/)
- [OpenAlex linked data schema](https://docs.openalex.org/)
- Berners-Lee, T., Hendler, J., & Lassila, O. (2001). The semantic web. *Scientific American*.

## Examples

### Example 1: Import CSV Data into RDF Graph

```python
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD, DCTERMS, FOAF

def csv_to_rdf(df, base_uri="http://example.org/"):
    """Convert a publications CSV to RDF triples.

    Args:
        df: DataFrame with columns: id, title, year, authors, doi
        base_uri: base URI for entities
    Returns:
        rdflib Graph
    """
    EX = Namespace(base_uri)
    BIBO = Namespace("http://purl.org/ontology/bibo/")
    g = Graph()
    g.bind("ex", EX); g.bind("bibo", BIBO)
    g.bind("dcterms", DCTERMS); g.bind("foaf", FOAF)

    for _, row in df.iterrows():
        paper_uri = EX[f"paper/{row['id']}"]
        g.add((paper_uri, RDF.type,        BIBO.AcademicArticle))
        g.add((paper_uri, DCTERMS.title,   Literal(str(row["title"]))))
        if pd.notna(row.get("year")):
            g.add((paper_uri, DCTERMS.date,
                   Literal(str(int(row["year"])), datatype=XSD.gYear)))
        if pd.notna(row.get("doi")):
            g.add((paper_uri, BIBO.doi, Literal(str(row["doi"]))))

        for i, author_name in enumerate(str(row.get("authors", "")).split(";")):
            author_name = author_name.strip()
            if author_name:
                author_uri = EX[f"author/{author_name.replace(' ', '_')}"]
                g.add((author_uri, RDF.type,    FOAF.Person))
                g.add((author_uri, FOAF.name,   Literal(author_name)))
                g.add((paper_uri, DCTERMS.creator, author_uri))
    return g

import numpy as np
np.random.seed(42)
sample_df = pd.DataFrame({
    "id": [f"P{i:04d}" for i in range(10)],
    "title": [f"Research Article {i}" for i in range(10)],
    "year": np.random.randint(2018, 2024, 10),
    "authors": ["Alice; Bob", "Carol", "David; Alice", "Eve; Frank",
                "Grace", "Henry; Alice", "Iris", "Jack; Eve",
                "Kate; David", "Liam"],
    "doi": [f"10.1234/j.2023.{i:04d}" for i in range(10)],
})

g_imported = csv_to_rdf(sample_df)
print(f"Imported graph: {len(g_imported)} triples")
g_imported.serialize("imported.ttl", format="turtle")
print("Saved to imported.ttl")
```

### Example 2: SPARQL-Based Authority File Reconciliation

```python
from rdflib import Graph, Namespace, URIRef, Literal, OWL
from rdflib.namespace import RDF, FOAF
import pandas as pd

def reconcile_authors(g, external_authority_df, name_col="name", uri_col="wikidata_uri"):
    """Add owl:sameAs links from local author nodes to external authority URIs.

    Args:
        g: rdflib.Graph
        external_authority_df: DataFrame with name → external URI mapping
        name_col, uri_col: column names
    Returns:
        g: graph with sameAs triples added
    """
    name_to_uri = dict(zip(external_authority_df[name_col],
                           external_authority_df[uri_col]))
    added = 0
    for author, _, name in g.triples((None, FOAF.name, None)):
        name_str = str(name)
        if name_str in name_to_uri:
            ext_uri = URIRef(name_to_uri[name_str])
            g.add((author, OWL.sameAs, ext_uri))
            added += 1
    print(f"Added {added} owl:sameAs links")
    return g

# Simulate authority file
authority = pd.DataFrame({
    "name": ["Alice Zhang", "Bob Smith", "Carol Johnson"],
    "wikidata_uri": [
        "https://www.wikidata.org/entity/Q999991",
        "https://www.wikidata.org/entity/Q999992",
        "https://www.wikidata.org/entity/Q999993",
    ]
})

g3 = Graph()
g3.parse("research_graph.ttl", format="turtle")
g3 = reconcile_authors(g3, authority)
g3.serialize("reconciled.ttl", format="turtle")
print("Reconciled graph saved to reconciled.ttl")
```
