#!/usr/bin/env python3
"""
generate_index.py — Scan all SKILL.md files and regenerate the README.md skill index table.

Usage:
    python scripts/generate_index.py                      # dry-run, print table
    python scripts/generate_index.py --update-readme      # overwrite README.md index section
    python scripts/generate_index.py --output index.md    # write to separate file
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: pyyaml not installed. Falling back to regex parsing.", file=sys.stderr)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
SKILLS_DIR = BASE_DIR / "skills"
README_PATH = BASE_DIR / "README.md"

# Markers in README.md that delimit the auto-generated index section
INDEX_START = "<!-- SKILLS_INDEX_START -->"
INDEX_END   = "<!-- SKILLS_INDEX_END -->"

DISCIPLINE_LABELS = {
    "00-universal":         ("🔬", "Universal Research"),
    "01-physics":           ("⚛️",  "Physics"),
    "02-chemistry":         ("🧪", "Chemistry"),
    "03-mathematics":       ("📐", "Mathematics & Statistics"),
    "04-earth-science":     ("🌍", "Earth & Environmental Science"),
    "05-neuroscience":      ("🧠", "Neuroscience"),
    "06-engineering":       ("⚙️",  "Engineering"),
    "07-economics":         ("📊", "Economics"),
    "08-finance-academic":  ("💹", "Finance (Academic)"),
    "09-political-science": ("🏛️",  "Political Science"),
    "10-sociology":         ("👥", "Sociology"),
    "11-psychology":        ("🧩", "Psychology"),
    "12-linguistics":       ("🗣️",  "Linguistics"),
    "13-history":           ("📜", "History"),
    "14-philosophy":        ("💡", "Philosophy"),
    "15-archaeology":       ("🏺", "Archaeology"),
    "16-art-music":         ("🎨", "Art & Music"),
    "17-public-health":     ("🏥", "Public Health & Epidemiology"),
    "18-urban-science":     ("🏙️",  "Urban Science & Planning"),
    "19-agriculture":       ("🌾", "Agriculture & Food Science"),
    "20-education":         ("📚", "Education"),
    "21-library-science":   ("📖", "Library Science & Bibliometrics"),
    "22-interdisciplinary": ("🔗", "Interdisciplinary Methods"),
    "23-research-workflow": ("📝", "Research Workflow"),
}

# ── Frontmatter parsing ──────────────────────────────────────────────────────

def parse_frontmatter(skill_path: Path) -> dict:
    """Parse YAML frontmatter from a SKILL.md file."""
    content = skill_path.read_text(encoding="utf-8")

    # Extract frontmatter block between --- delimiters
    fm_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not fm_match:
        return {}

    fm_text = fm_match.group(1)

    if HAS_YAML:
        try:
            return yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            pass

    # Fallback: simple key: value regex parsing
    fm = {}
    for line in fm_text.splitlines():
        m = re.match(r"^(\w[\w_-]*):\s*(.+)$", line)
        if m:
            key, val = m.group(1), m.group(2).strip().strip('"\'')
            fm[key] = val
    return fm


def get_skill_description(skill_path: Path) -> str:
    """Get the first non-empty line of the description field (cleaned)."""
    fm = parse_frontmatter(skill_path)
    desc = fm.get("description", "")
    if isinstance(desc, str):
        # Remove YAML block scalar indicators and collapse whitespace
        desc = re.sub(r"[>|]", "", desc)
        desc = " ".join(desc.split())
        return desc[:120] + ("…" if len(desc) > 120 else "")
    return ""


# ── Index generation ─────────────────────────────────────────────────────────

def collect_skills() -> dict:
    """Walk skills/ directory and collect all SKILL.md metadata."""
    skills_by_discipline: dict = {}

    for discipline_dir in sorted(SKILLS_DIR.iterdir()):
        if not discipline_dir.is_dir():
            continue
        dirname = discipline_dir.name
        if dirname not in DISCIPLINE_LABELS:
            continue

        skills = []
        for skill_dir in sorted(discipline_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            fm = parse_frontmatter(skill_md)
            name = fm.get("name") or skill_dir.name
            description = get_skill_description(skill_md)
            version = fm.get("version", "1.0.0")
            tags = fm.get("tags", [])
            if isinstance(tags, list):
                tags_str = ", ".join(f"`{t}`" for t in tags[:3])
            else:
                tags_str = ""

            rel_path = skill_md.relative_to(BASE_DIR).as_posix()
            skills.append({
                "name": name,
                "description": description,
                "version": version,
                "tags": tags_str,
                "path": rel_path,
                "skill_dir": skill_dir.name,
            })

        if skills:
            skills_by_discipline[dirname] = skills

    return skills_by_discipline


def generate_index_markdown(skills_by_discipline: dict) -> str:
    """Generate the full index markdown table."""
    lines = []
    lines.append(INDEX_START)
    lines.append("")

    total = sum(len(s) for s in skills_by_discipline.values())
    lines.append(f"> **{total} Skills** across {len(skills_by_discipline)} disciplines\n")

    for dirname, skills in skills_by_discipline.items():
        icon, label = DISCIPLINE_LABELS.get(dirname, ("📄", dirname))
        lines.append(f"### {icon} {label} (`{dirname}/`)\n")
        lines.append("| Skill | Description | Tags |")
        lines.append("|---|---|---|")
        for s in skills:
            name_link = f"[`{s['name']}`]({s['path']})"
            lines.append(f"| {name_link} | {s['description']} | {s['tags']} |")
        lines.append("")

    lines.append(INDEX_END)
    return "\n".join(lines)


def update_readme(index_md: str) -> None:
    """Replace the index section in README.md with fresh content."""
    readme = README_PATH.read_text(encoding="utf-8")

    # Replace content between markers
    pattern = re.escape(INDEX_START) + r".*?" + re.escape(INDEX_END)
    new_readme, count = re.subn(pattern, index_md, readme, flags=re.DOTALL)

    if count == 0:
        print("WARNING: Index markers not found in README.md.", file=sys.stderr)
        print(f"Add {INDEX_START} ... {INDEX_END} to README.md where you want the index.", file=sys.stderr)
        return

    README_PATH.write_text(new_readme, encoding="utf-8")
    print(f"README.md updated with {sum(len(s) for s in collect_skills().values())} skills.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate awesome-rosetta-skills index")
    parser.add_argument("--update-readme", action="store_true",
                        help="Write index back into README.md between marker comments")
    parser.add_argument("--output", metavar="FILE",
                        help="Write index to a separate file instead of stdout")
    args = parser.parse_args()

    skills_by_discipline = collect_skills()

    if not skills_by_discipline:
        print("No SKILL.md files found under skills/", file=sys.stderr)
        sys.exit(1)

    index_md = generate_index_markdown(skills_by_discipline)

    if args.update_readme:
        update_readme(index_md)
    elif args.output:
        Path(args.output).write_text(index_md, encoding="utf-8")
        print(f"Index written to {args.output}")
    else:
        print(index_md)

    # Summary
    total = sum(len(s) for s in skills_by_discipline.values())
    print(f"\nSummary: {total} skills across {len(skills_by_discipline)} disciplines",
          file=sys.stderr)
    for dirname, skills in skills_by_discipline.items():
        icon, label = DISCIPLINE_LABELS.get(dirname, ("📄", dirname))
        print(f"  {icon} {label}: {len(skills)}", file=sys.stderr)


if __name__ == "__main__":
    main()
