---
name: argument-mapping
description: >
  Use this Skill to extract and visualize argument structure: Toulmin reconstruction,
  support/attack graph, circularity detection, argument scheme classification, and NLP-based premise extraction.
tags:
  - philosophy
  - argument-mapping
  - Toulmin
  - argumentation
  - discourse-analysis
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
    - networkx>=3.1
    - pandas>=1.5
    - matplotlib>=3.6
    - nltk>=3.8
last_updated: "2026-03-18"
status: stable
---

# Argument Mapping: Toulmin Reconstruction and Argumentation Analysis

> **TL;DR** — Reconstruct arguments using the Toulmin model (claim, data, warrant, backing,
> rebuttal, qualifier), build directed support/attack graphs with NetworkX, detect
> circular reasoning and inconsistency, classify argument schemes, and visualize
> argument structure with colour-coded edges.

---

## When to Use

Use this Skill when you need to:

- Formalize and diagram an argument from a philosophical text, speech, or policy document
- Detect circular reasoning (cycles in the support graph) or inconsistency (contradictory claims)
- Classify an argument as from authority, analogy, cause, sign, or example
- Extract premise-conclusion structure from a passage using sentence similarity
- Produce publication-ready argument maps for scholarly articles or teaching materials

Do **not** use this Skill for:

- Formal logical validity checking (use formal-logic Skill with Z3)
- Automated summarization without argument-specific structure (use LLM summarization)
- Large-scale computational argumentation mining on corpora (use ArgMine or TARGER)

---

## Background

Argument mapping makes reasoning visible and evaluable. The Toulmin model (1958) breaks
every argument into six functional components:

| Component | Role | Example |
|---|---|---|
| Claim (C) | The conclusion being argued for | "We should ban single-use plastics." |
| Data (D) | Grounds / evidence supporting the claim | "Marine plastics kill 1M seabirds/year." |
| Warrant (W) | The inferential bridge from D to C | "Harms to wildlife justify bans." |
| Backing (B) | Support for the warrant itself | "Environmental ethics require harm reduction." |
| Rebuttal (R) | Exceptions or counter-considerations | "Unless economic disruption is too great." |
| Qualifier (Q) | Hedging the strength of the claim | "Presumably", "in most cases" |

An argument graph models propositions as nodes and support/attack relationships as edges.
Formal properties:

- **Circularity**: a directed cycle in the support sub-graph means a proposition
  ultimately supports itself — a logical fallacy.
- **Inconsistency**: both a proposition P and its negation ¬P appear as reachable from
  the same claim node via support edges.
- **Scheme classification**: argument schemes capture stereotyped patterns of reasoning
  (Walton 1996) with associated critical questions.

---

## Environment Setup

```bash
# Create Python environment
conda create -n arg-map python=3.11 -y
conda activate arg-map

# Install dependencies
pip install "networkx>=3.1" "pandas>=1.5" "matplotlib>=3.6" "nltk>=3.8"

# Download NLTK data for tokenization and similarity
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"

# Verify
python -c "import networkx as nx; print(nx.__version__)"
```

---

## Core Workflow

### Step 1 — Argument Graph from Structured Text with Circularity Check

```python
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional


SUPPORT_COLOR = "#2ca02c"   # green
ATTACK_COLOR = "#d62728"    # red
NODE_COLOR = "#aec7e8"      # light blue


def build_argument_graph(
    propositions: list[dict],
    relations: list[dict],
) -> nx.DiGraph:
    """
    Build a directed argument graph from proposition and relation lists.

    Args:
        propositions: List of dicts with keys:
                      id (str), text (str), type (str: claim/premise/rebuttal).
        relations:    List of dicts with keys:
                      source (str), target (str), relation_type (str: support/attack).

    Returns:
        Directed graph with node attribute 'text' and 'node_type',
        edge attribute 'relation_type'.
    """
    G = nx.DiGraph()

    for prop in propositions:
        G.add_node(
            prop["id"],
            text=prop.get("text", ""),
            node_type=prop.get("type", "premise"),
        )

    for rel in relations:
        G.add_edge(
            rel["source"],
            rel["target"],
            relation_type=rel.get("relation_type", "support"),
        )

    return G


def detect_circularity(G: nx.DiGraph) -> list[list]:
    """
    Detect circular reasoning in the argument graph.

    Restricts the search to support edges only (attack edges are legitimate
    counter-arguments, not circular reasoning per se).

    Args:
        G: Directed argument graph from build_argument_graph().

    Returns:
        List of cycles (each cycle is a list of node IDs).
        Empty list means no circularity detected.
    """
    support_subgraph = nx.DiGraph([
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relation_type") == "support"
    ])
    support_subgraph.add_nodes_from(G.nodes)

    cycles = list(nx.simple_cycles(support_subgraph))
    return cycles


def check_inconsistency(
    G: nx.DiGraph,
    negation_map: dict,
) -> list[tuple]:
    """
    Check for inconsistency: both a proposition and its negation appear
    as reachable premises supporting the same claim.

    Args:
        G:             Directed argument graph.
        negation_map:  Dict mapping proposition ID to the ID of its negation.
                       E.g. {"P1": "P1_neg"} means P1 and P1_neg are contradictory.

    Returns:
        List of (claim_node, prop_id, negation_id) triples where inconsistency found.
    """
    inconsistencies = []

    for claim_node in G.nodes:
        if G.nodes[claim_node].get("node_type") != "claim":
            continue

        # Find all nodes reachable via support edges from premises to this claim
        ancestors = nx.ancestors(G, claim_node)
        ancestors.add(claim_node)

        for prop_id, neg_id in negation_map.items():
            if prop_id in ancestors and neg_id in ancestors:
                inconsistencies.append((claim_node, prop_id, neg_id))

    return inconsistencies


def visualize_argument_graph(
    G: nx.DiGraph,
    title: str = "Argument Map",
    output_path: str = None,
    show_labels: bool = True,
) -> None:
    """
    Visualize an argument graph with green support edges and red attack edges.

    Node colour encodes type: claim (gold), premise (light blue), rebuttal (salmon).

    Args:
        G:           Directed argument graph.
        title:       Plot title.
        output_path: If given, save figure here.
        show_labels: Whether to display node text labels.
    """
    type_colors = {
        "claim": "#FFD700",     # gold
        "premise": "#AEC7E8",   # light blue
        "rebuttal": "#FFBB78",  # salmon
        "backing": "#C5B0D5",   # lavender
    }

    node_colors = [
        type_colors.get(G.nodes[n].get("node_type", "premise"), "#AEC7E8")
        for n in G.nodes
    ]

    edge_colors = [
        SUPPORT_COLOR if d.get("relation_type") == "support" else ATTACK_COLOR
        for _, _, d in G.edges(data=True)
    ]

    pos = nx.spring_layout(G, seed=42, k=2.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, arrows=True,
        arrowsize=20, width=2.0, ax=ax,
        connectionstyle="arc3,rad=0.1",
    )

    if show_labels:
        labels = {}
        for n in G.nodes:
            text = G.nodes[n].get("text", n)
            # Wrap long labels
            words = text.split()
            wrapped = "\n".join(
                " ".join(words[i:i+4]) for i in range(0, len(words), 4)
            )
            labels[n] = f"[{n}]\n{wrapped[:60]}"
        nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    # Legend
    legend_items = [
        mpatches.Patch(color=SUPPORT_COLOR, label="Support"),
        mpatches.Patch(color=ATTACK_COLOR, label="Attack"),
        mpatches.Patch(color="#FFD700", label="Claim"),
        mpatches.Patch(color="#AEC7E8", label="Premise"),
        mpatches.Patch(color="#FFBB78", label="Rebuttal"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Argument map saved to {output_path}")
    plt.show()
```

### Step 2 — Toulmin Reconstruction Template and Visualization

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToulminArgument:
    """
    Data class representing a fully articulated Toulmin argument structure.

    Attributes:
        claim:     The main conclusion (what is being argued).
        data:      The grounds/evidence (why the claim is made).
        warrant:   The inferential rule connecting data to claim.
        backing:   Support for the warrant.
        rebuttal:  Exceptions or counter-considerations.
        qualifier: Hedging phrase (e.g. "presumably", "certainly", "in most cases").
    """
    claim: str
    data: str
    warrant: str
    backing: str = ""
    rebuttal: str = ""
    qualifier: str = "presumably"

    def to_argument_graph(self) -> nx.DiGraph:
        """Convert the Toulmin structure to a NetworkX argument graph."""
        propositions = [
            {"id": "C", "text": self.claim, "type": "claim"},
            {"id": "D", "text": self.data, "type": "premise"},
            {"id": "W", "text": self.warrant, "type": "premise"},
        ]
        relations = [
            {"source": "D", "target": "C", "relation_type": "support"},
            {"source": "W", "target": "C", "relation_type": "support"},
        ]
        if self.backing:
            propositions.append({"id": "B", "text": self.backing, "type": "backing"})
            relations.append({"source": "B", "target": "W", "relation_type": "support"})
        if self.rebuttal:
            propositions.append({"id": "R", "text": self.rebuttal, "type": "rebuttal"})
            relations.append({"source": "R", "target": "C", "relation_type": "attack"})

        return build_argument_graph(propositions, relations)

    def summarize(self) -> str:
        """Return a formatted plain-text Toulmin summary."""
        lines = [
            f"CLAIM    : {self.qualifier.upper()}, {self.claim}",
            f"DATA     : {self.data}",
            f"WARRANT  : {self.warrant}",
        ]
        if self.backing:
            lines.append(f"BACKING  : {self.backing}")
        if self.rebuttal:
            lines.append(f"REBUTTAL : Unless {self.rebuttal}")
        return "\n".join(lines)


def classify_argument_scheme(
    argument_text: str,
    keywords: dict = None,
) -> str:
    """
    Classify an argument into one of Walton's standard argument schemes.

    Uses keyword heuristics to identify the dominant scheme. For production
    use, replace with a fine-tuned classifier.

    Schemes: authority, analogy, cause, sign, example, slippery-slope, ad-hominem.

    Args:
        argument_text: The raw argument text string.
        keywords:      Optional dict {scheme: [keywords]}. Uses built-in defaults if None.

    Returns:
        The most likely scheme name as a string.
    """
    if keywords is None:
        keywords = {
            "authority": ["expert", "authority", "professor", "studies show",
                          "according to", "research", "scientist"],
            "analogy": ["similarly", "just as", "like", "analogous", "comparable",
                        "in the same way", "resembles"],
            "cause": ["causes", "because", "therefore", "leads to", "results in",
                      "due to", "effect", "consequence"],
            "sign": ["indicates", "suggests", "is a sign", "symptom", "evidence",
                     "points to", "signal"],
            "example": ["for example", "for instance", "such as", "e.g.", "case",
                        "illustrates", "consider"],
            "slippery-slope": ["will lead to", "eventually", "first step", "next thing",
                               "inevitably", "chain of events"],
            "ad-hominem": ["bias", "interest", "corrupt", "liar", "untrustworthy",
                           "agenda", "motivated"],
        }

    text_lower = argument_text.lower()
    scheme_scores = {
        scheme: sum(1 for kw in kws if kw in text_lower)
        for scheme, kws in keywords.items()
    }

    best_scheme = max(scheme_scores, key=scheme_scores.get)
    if scheme_scores[best_scheme] == 0:
        return "unclassified"
    return best_scheme
```

### Step 3 — NLP-Based Support/Attack Relation Extraction

```python
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string


def extract_propositions(text: str) -> list[str]:
    """
    Segment a passage into candidate propositions (declarative sentences).

    Args:
        text: Input argument text.

    Returns:
        List of candidate proposition strings.
    """
    sentences = sent_tokenize(text)
    # Filter out non-declarative (questions, exclamations)
    declarative = [
        s.strip() for s in sentences
        if s.strip() and not s.strip().endswith("?") and not s.strip().endswith("!")
        and len(s.split()) >= 5
    ]
    return declarative


def sentence_similarity_jaccard(sent_a: str, sent_b: str) -> float:
    """
    Compute Jaccard similarity between two sentences as bags of content words.

    Args:
        sent_a: First sentence string.
        sent_b: Second sentence string.

    Returns:
        Jaccard similarity coefficient (0.0 to 1.0).
    """
    stop = set(stopwords.words("english")) | set(string.punctuation)

    def tokenize(s):
        return {
            w.lower() for w in nltk.word_tokenize(s)
            if w.lower() not in stop and w.isalpha()
        }

    a = tokenize(sent_a)
    b = tokenize(sent_b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def infer_support_attack_relations(
    propositions: list[str],
    support_cues: list[str] = None,
    attack_cues: list[str] = None,
    similarity_threshold: float = 0.15,
) -> list[dict]:
    """
    Infer support and attack relations between propositions using cue phrases
    and Jaccard content-word similarity.

    This is a heuristic approximation; manual review is always recommended
    for scholarly argument maps.

    Args:
        propositions:         List of sentence strings.
        support_cues:         Discourse markers indicating support.
        attack_cues:          Discourse markers indicating attack.
        similarity_threshold: Minimum Jaccard to propose a relation.

    Returns:
        List of relation dicts: source_idx, target_idx, relation_type, confidence.
    """
    if support_cues is None:
        support_cues = [
            "therefore", "thus", "hence", "because", "since",
            "this shows", "this means", "as a result", "consequently",
        ]
    if attack_cues is None:
        attack_cues = [
            "however", "but", "although", "nevertheless", "on the other hand",
            "despite", "yet", "in contrast", "this ignores",
        ]

    relations = []

    for i, sent_i in enumerate(propositions):
        sent_lower = sent_i.lower()
        for j, sent_j in enumerate(propositions):
            if i == j:
                continue

            jacc = sentence_similarity_jaccard(sent_i, sent_j)
            if jacc < similarity_threshold:
                continue

            # Classify by cue phrases
            has_support = any(cue in sent_lower for cue in support_cues)
            has_attack = any(cue in sent_lower for cue in attack_cues)

            if has_support and not has_attack:
                rel_type = "support"
            elif has_attack and not has_support:
                rel_type = "attack"
            else:
                # Default to support for thematically similar consecutive sentences
                if abs(i - j) == 1:
                    rel_type = "support"
                else:
                    continue

            relations.append({
                "source_idx": i,
                "target_idx": j,
                "relation_type": rel_type,
                "confidence": round(jacc, 3),
            })

    return relations
```

---

## Advanced Usage

### Argument Strength Evaluation

```python
def evaluate_argument_strength(
    G: nx.DiGraph,
    claim_node: str,
) -> dict:
    """
    Heuristically evaluate the strength of an argument network.

    Metrics:
        - support_count: Number of direct support edges to the claim
        - attack_count:  Number of direct attack edges to the claim
        - depth:         Maximum reasoning chain depth (longest support path to claim)
        - has_backing:   Whether any warrant has a backing node

    Args:
        G:           Directed argument graph.
        claim_node:  Node ID of the main claim.

    Returns:
        Dict with support_count, attack_count, depth, has_backing, verdict.
    """
    support_in = sum(
        1 for _, _, d in G.in_edges(claim_node, data=True)
        if d.get("relation_type") == "support"
    )
    attack_in = sum(
        1 for _, _, d in G.in_edges(claim_node, data=True)
        if d.get("relation_type") == "attack"
    )

    # Depth = longest path to claim from any leaf node via support edges
    support_G = nx.DiGraph([
        (u, v) for u, v, d in G.edges(data=True)
        if d.get("relation_type") == "support"
    ])
    support_G.add_nodes_from(G.nodes)

    try:
        paths = nx.single_target_shortest_path_length(support_G, claim_node)
        depth = max(paths.values()) if paths else 0
    except Exception:
        depth = 0

    has_backing = any(
        G.nodes[n].get("node_type") == "backing" for n in G.nodes
    )

    verdict = "strong" if support_in >= 2 and attack_in == 0 else \
              "contested" if attack_in > 0 else \
              "weak" if support_in <= 1 else "moderate"

    return {
        "support_count": support_in,
        "attack_count": attack_in,
        "depth": depth,
        "has_backing": has_backing,
        "verdict": verdict,
    }
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `LookupError: punkt` on sentence tokenization | NLTK data not downloaded | Run `nltk.download('punkt_tab')` (newer NLTK) |
| `nx.simple_cycles()` hangs on large graph | Dense graph with many cycles | Limit to support sub-graph; set `nx.simple_cycles(G, length_bound=5)` |
| All relations classified as "support" | Lack of cue phrases in text | Add domain-specific cue words to `attack_cues` list |
| Spring layout overlaps many nodes | Dense graph | Try `nx.kamada_kawai_layout()` or `nx.shell_layout()` |
| Argument scheme always "unclassified" | Academic/formal text without colloquial cues | Extend keyword dict with domain vocabulary |
| Circular detection finds false cycles | Attack edges form apparent cycles | Filter to support-only subgraph before cycle detection |

---

## External Resources

- Toulmin, S. (1958). *The Uses of Argument*. Cambridge University Press.
- Walton, D. (1996). *Argumentation Schemes for Presumptive Reasoning*. Lawrence Erlbaum.
- Argument Web (OVA+ online argument mapping): <https://arg.tech/ova/>
- Argument Interchange Format (AIF): <http://www.argumentinterchange.org/>
- NetworkX drawing documentation: <https://networkx.org/documentation/stable/reference/drawing.html>
- NLTK documentation: <https://www.nltk.org/>

---

## Examples

### Example 1 — Toulmin Reconstruction of an Environmental Argument

```python
# Reconstruct and visualize a Toulmin argument about plastic bans
arg = ToulminArgument(
    claim="Single-use plastics should be banned.",
    data="Marine plastic pollution kills over one million seabirds annually.",
    warrant="Practices that cause large-scale wildlife death ought to be prohibited.",
    backing="Environmental ethics requires minimizing unnecessary harm to other species.",
    rebuttal="the economic disruption to packaging industries is prohibitive",
    qualifier="presumably",
)

print(arg.summarize())

G = arg.to_argument_graph()
cycles = detect_circularity(G)
print(f"\nCircular reasoning detected: {len(cycles) > 0}")
print(f"Cycles: {cycles}")

strength = evaluate_argument_strength(G, claim_node="C")
print(f"\nArgument strength verdict: {strength['verdict']}")
print(f"  Support edges: {strength['support_count']}, Attack edges: {strength['attack_count']}")

visualize_argument_graph(
    G,
    title="Toulmin Map: Plastic Ban Argument",
    output_path="/data/output/plastic_ban_argument.png",
)
```

### Example 2 — NLP-Based Argument Extraction from a Policy Text

```python
policy_text = """
The government should invest in renewable energy because fossil fuels are depleting rapidly.
Since climate change threatens global stability, transitioning to renewables is essential.
However, the upfront costs of renewable infrastructure are extremely high.
Expert consensus from the IPCC confirms that renewable transition is technically feasible.
Therefore, a managed transition over twenty years represents the most viable approach.
"""

# Extract propositions and classify scheme
propositions = extract_propositions(policy_text)
print(f"Extracted {len(propositions)} propositions:")
for i, p in enumerate(propositions):
    print(f"  [{i}] {p}")

scheme = classify_argument_scheme(policy_text)
print(f"\nPrimary argument scheme: {scheme}")

# Infer relations between propositions
relations = infer_support_attack_relations(propositions, similarity_threshold=0.1)
print(f"\nInferred {len(relations)} relations:")
for r in relations:
    print(f"  [{r['source_idx']}] --{r['relation_type']}--> [{r['target_idx']}]  (conf={r['confidence']})")

# Build and visualize graph
prop_dicts = [{"id": str(i), "text": p, "type": "premise"} for i, p in enumerate(propositions)]
if prop_dicts:
    prop_dicts[0]["type"] = "claim"

rel_dicts = [
    {"source": str(r["source_idx"]), "target": str(r["target_idx"]),
     "relation_type": r["relation_type"]}
    for r in relations
]

G2 = build_argument_graph(prop_dicts, rel_dicts)
cycles2 = detect_circularity(G2)
print(f"\nCircular reasoning detected: {len(cycles2) > 0}")
visualize_argument_graph(G2, title="Policy Argument Map", output_path="/data/output/policy_argument.png")
```

---

## Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-03-18 | Initial release — Toulmin data class, argument graph, circularity detection, scheme classification, NLP relation extraction |
