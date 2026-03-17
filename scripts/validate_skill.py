#!/usr/bin/env python3
"""
validate_skill.py — awesome-rosetta-skills Skill format validator

Usage:
    python scripts/validate_skill.py skills/07-economics/did-causal/SKILL.md
    python scripts/validate_skill.py skills/07-economics/did-causal/SKILL.md --verbose
    python scripts/validate_skill.py skills/  # batch-validate entire directory

Rule reference: SKILL_STANDARD.md §7
"""

import sys
import os
import re
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: pyyaml not installed; frontmatter validation will be skipped. "
          "Run: pip install pyyaml")


# ============================================================
# Rule definitions
# ============================================================

REQUIRED_FRONTMATTER_FIELDS = [
    "name",
    "description",
    "tags",
    "version",
    "authors",
    "license",
    "platforms",
    "last_updated",
]

ALLOWED_LICENSES = {
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
}

# Required section heading patterns (match English headings)
REQUIRED_SECTIONS = [
    r"When to Use|When To Use",
    r"Background|Key Concepts|Core Concept",
    r"Environment Setup|Installation|Setup",
    r"Core Workflow|Workflow",
    r"Troubleshooting|FAQ",
    r"External Resources|References|Resources",
    r"Examples|Example",
]

# Patterns that may indicate hardcoded secrets
SECRET_PATTERNS = [
    r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9_\-]{10,}["\']',
    r'token\s*=\s*["\'][a-zA-Z0-9_\-]{10,}["\']',
    r'password\s*=\s*["\'][^"\']+["\']',
    r'secret\s*=\s*["\'][a-zA-Z0-9_\-]{10,}["\']',
    r'sk-[a-zA-Z0-9]{20,}',
    r'Bearer\s+[a-zA-Z0-9_\-\.]{20,}',
]


# ============================================================
# Data structures
# ============================================================

@dataclass
class ValidationResult:
    rule_id: str
    level: str       # ERROR | WARNING | INFO
    message: str
    line_number: Optional[int] = None

    def __str__(self):
        loc = f":{self.line_number}" if self.line_number else ""
        return f"[{self.level}] {self.rule_id}{loc}: {self.message}"


@dataclass
class ValidationReport:
    file_path: str
    results: list = field(default_factory=list)

    def add(self, rule_id: str, level: str, message: str, line_number: int = None):
        self.results.append(ValidationResult(rule_id, level, message, line_number))

    @property
    def errors(self):
        return [r for r in self.results if r.level == "ERROR"]

    @property
    def warnings(self):
        return [r for r in self.results if r.level == "WARNING"]

    @property
    def passed(self):
        return len(self.errors) == 0

    def summary(self):
        status = "PASSED" if self.passed else "FAILED"
        icon = "✅" if self.passed else "❌"
        return (
            f"\n{'='*60}\n"
            f"{icon} {status} — {self.file_path}\n"
            f"  Errors:   {len(self.errors)}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"{'='*60}"
        )


# ============================================================
# Validation functions
# ============================================================

def parse_frontmatter(content: str):
    """
    Parse YAML frontmatter. Returns (frontmatter_dict, body_text, end_line).
    Returns (None, content, 0) if no frontmatter found.
    """
    if not content.startswith("---"):
        return None, content, 0

    lines = content.split("\n")
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return None, content, 0

    frontmatter_text = "\n".join(lines[1:end_idx])
    body_text = "\n".join(lines[end_idx + 1:])

    if HAS_YAML:
        try:
            fm = yaml.safe_load(frontmatter_text)
            return fm, body_text, end_idx + 1
        except yaml.YAMLError as e:
            return {"_parse_error": str(e)}, body_text, end_idx + 1
    else:
        # Lightweight fallback: just detect which top-level keys exist
        fm = {}
        for line in frontmatter_text.split("\n"):
            if ":" in line and not line.startswith(" "):
                key = line.split(":")[0].strip()
                fm[key] = True
        return fm, body_text, end_idx + 1


def check_frontmatter(fm, report: ValidationReport, fm_end_line: int):
    """F00x series: frontmatter format checks."""

    # F001: frontmatter exists
    if fm is None:
        report.add("F001", "ERROR",
                   "No YAML frontmatter found (file must start with ---)")
        return

    if isinstance(fm, dict) and "_parse_error" in fm:
        report.add("F001", "ERROR",
                   f"YAML frontmatter parse failure: {fm['_parse_error']}")
        return

    # F002: required fields
    for field_name in REQUIRED_FRONTMATTER_FIELDS:
        if field_name not in fm or fm[field_name] is None:
            report.add("F002", "ERROR", f"Missing required field: {field_name}")

    if "name" not in fm:
        return  # dependent checks require name

    # F003: name is kebab-case
    name = fm.get("name", "")
    if isinstance(name, str):
        if not re.match(r'^[a-z0-9]+(-[a-z0-9]+)*$', name):
            report.add("F003", "ERROR",
                f"name field '{name}' is not valid kebab-case "
                "(only lowercase letters, digits, and hyphens allowed)")
    else:
        report.add("F003", "ERROR", "name field must be a string")

    # F004: version follows SemVer
    version = fm.get("version", "")
    if isinstance(version, str):
        if not re.match(r'^\d+\.\d+\.\d+', version):
            report.add("F004", "ERROR",
                f"version field '{version}' does not follow SemVer (expected X.Y.Z)")
    else:
        report.add("F004", "WARNING",
                   'version should be a quoted string, e.g. "1.0.0"')

    # F005: last_updated is ISO date
    last_updated = fm.get("last_updated", "")
    if isinstance(last_updated, str):
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', last_updated):
            report.add("F005", "ERROR",
                f"last_updated field '{last_updated}' must be ISO format YYYY-MM-DD")
    else:
        report.add("F005", "WARNING",
                   'last_updated should be a quoted string, e.g. "2026-03-17"')

    # F006: license is allowed value
    license_val = fm.get("license", "")
    if isinstance(license_val, str) and license_val not in ALLOWED_LICENSES:
        report.add("F006", "ERROR",
            f"license '{license_val}' is not in the allowed list: "
            f"{', '.join(sorted(ALLOWED_LICENSES))}")

    # F007: tags has at least 3 entries
    tags = fm.get("tags", [])
    if isinstance(tags, list) and len(tags) < 3:
        report.add("F007", "WARNING",
                   f"tags has only {len(tags)} entries; at least 3 recommended")

    # F008: authors have name field
    authors = fm.get("authors", [])
    if isinstance(authors, list):
        for i, author in enumerate(authors):
            if not isinstance(author, dict) or "name" not in author:
                report.add("F008", "WARNING",
                           f"authors[{i}] should be a dict with 'name' and 'github' keys")

    # F009: platforms includes claude-code
    platforms = fm.get("platforms", [])
    if isinstance(platforms, list) and "claude-code" not in platforms:
        report.add("F009", "WARNING",
                   "platforms list should include 'claude-code'")

    # F010: description length
    desc = fm.get("description", "")
    if isinstance(desc, str):
        desc_len = len(desc.strip())
        if desc_len < 30:
            report.add("F010", "ERROR",
                f"description is too short ({desc_len} chars); minimum 50 recommended")
        elif desc_len > 300:
            report.add("F010", "WARNING",
                f"description is long ({desc_len} chars); 50-150 chars recommended")


def check_content(body: str, report: ValidationReport, fm_end_line: int):
    """C00x series: content checks."""
    lines = body.split("\n")
    total_lines = len(lines)

    # C001: minimum line count
    if total_lines < 300:
        report.add("C001", "ERROR",
            f"Body line count too low ({total_lines} lines; ≥ 300 required)")
    else:
        report.add("C001", "INFO", f"Line count OK ({total_lines} lines)")

    # C002: minimum 2 code blocks
    code_open_tags = re.findall(r'^```\w*', body, re.MULTILINE)
    num_blocks = len(code_open_tags) // 2
    if len(code_open_tags) < 4:
        report.add("C002", "ERROR",
            f"Too few code blocks (detected ~{num_blocks}; ≥ 2 required)")
    else:
        report.add("C002", "INFO", f"Code block count OK (~{num_blocks} blocks)")

    # C003: code blocks have language annotation
    unlabeled = re.findall(r'^```\s*$', body, re.MULTILINE)
    if unlabeled:
        report.add("C003", "WARNING",
            f"{len(unlabeled)} code block(s) have no language annotation "
            "(add e.g. ```python, ```bash, ```r)")

    # C004: required section headings
    for section_pattern in REQUIRED_SECTIONS:
        if not re.search(section_pattern, body, re.IGNORECASE):
            report.add("C004", "WARNING",
                f"Missing recommended section: '{section_pattern}' "
                "(see SKILL_STANDARD.md §2.1)")

    # C005: top-level heading
    if not re.search(r'^# .+', body, re.MULTILINE):
        report.add("C005", "WARNING", "Missing top-level heading (# Skill Name)")


def check_security(content: str, report: ValidationReport):
    """S00x series: security checks (hardcoded secret detection)."""
    lines = content.split("\n")

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Skip comment lines
        if stripped.startswith("#"):
            continue

        for pattern in SECRET_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                # Check whether this looks like an example / placeholder
                safe_indicators = [
                    r'your[_-]?api[_-]?key',
                    r'your[_-]?token',
                    r'your-api',
                    r'example',
                    r'placeholder',
                    r'os\.getenv',
                    r'environ',
                    r'\[TODO',
                    r'your-key',
                    r'<YOUR_',
                    r'\*{4,}',
                ]
                is_safe = any(
                    re.search(sp, line, re.IGNORECASE)
                    for sp in safe_indicators
                )
                if not is_safe:
                    report.add("S001", "ERROR",
                        f"Possible hardcoded secret on line {i} "
                        f"(matched pattern: {pattern})",
                        line_number=i)
                break


# ============================================================
# Main validation entry points
# ============================================================

def validate_file(file_path: str, verbose: bool = False) -> ValidationReport:
    """Validate a single SKILL.md file."""
    report = ValidationReport(file_path=file_path)

    path = Path(file_path)
    if not path.exists():
        report.add("FILE", "ERROR", f"File not found: {file_path}")
        return report

    if path.name != "SKILL.md":
        report.add("FILE", "WARNING",
                   f"File should be named SKILL.md (got {path.name})")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        report.add("FILE", "ERROR",
                   "File encoding error — please save as UTF-8")
        return report

    fm, body, fm_end_line = parse_frontmatter(content)
    check_frontmatter(fm, report, fm_end_line)
    check_content(body, report, fm_end_line)
    check_security(content, report)

    return report


def validate_directory(dir_path: str, verbose: bool = False) -> list:
    """Batch-validate all SKILL.md files under a directory."""
    reports = []
    skill_files = list(Path(dir_path).rglob("SKILL.md"))

    if not skill_files:
        print(f"No SKILL.md files found under {dir_path}")
        return reports

    print(f"Found {len(skill_files)} SKILL.md file(s) — validating...\n")

    for skill_file in sorted(skill_files):
        report = validate_file(str(skill_file), verbose)
        reports.append(report)

        status = "✅" if report.passed else "❌"
        print(f"  {status} {skill_file}")
        if not report.passed or verbose:
            for result in report.results:
                if result.level in ("ERROR", "WARNING") or verbose:
                    print(f"      {result}")

    return reports


def print_report(report: ValidationReport, verbose: bool = False):
    """Print a single-file validation report."""
    print(report.summary())

    if report.errors:
        print("\nErrors (block CI merge):")
        for r in report.errors:
            print(f"  {r}")

    if report.warnings:
        print("\nWarnings (do not block CI):")
        for r in report.warnings:
            print(f"  {r}")

    if verbose:
        info_results = [r for r in report.results if r.level == "INFO"]
        if info_results:
            print("\nPassed checks:")
            for r in info_results:
                print(f"  {r}")


def export_json_report(reports: list, output_path: str):
    """Export validation results as JSON (for CI system consumption)."""
    data = []
    for report in reports:
        data.append({
            "file": report.file_path,
            "passed": report.passed,
            "errors": [{"rule": r.rule_id, "message": r.message, "line": r.line_number}
                       for r in report.errors],
            "warnings": [{"rule": r.rule_id, "message": r.message}
                         for r in report.warnings],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nJSON report exported: {output_path}")


# ============================================================
# CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate SKILL.md format against awesome-rosetta-skills standards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_skill.py skills/07-economics/did-causal/SKILL.md
  python scripts/validate_skill.py skills/07-economics/did-causal/SKILL.md --verbose
  python scripts/validate_skill.py skills/ --verbose
  python scripts/validate_skill.py skills/ --json-output report.json
        """
    )
    parser.add_argument("path", help="Path to a SKILL.md file or a skills/ directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output including passing checks")
    parser.add_argument("--json-output", metavar="FILE",
                        help="Export results to a JSON file (for CI systems)")
    parser.add_argument("--fail-on-warning", action="store_true",
                        help="Treat warnings as failures (stricter mode)")

    args = parser.parse_args()
    target = Path(args.path)

    if target.is_dir():
        reports = validate_directory(str(target), args.verbose)

        total = len(reports)
        passed = sum(1 for r in reports if r.passed)
        failed = total - passed

        print(f"\n{'='*60}")
        print(f"Batch validation complete: {total} file(s), "
              f"{passed} passed, {failed} failed")
        print(f"{'='*60}")

        if args.json_output:
            export_json_report(reports, args.json_output)

        has_errors = any(not r.passed for r in reports)
        has_warnings = any(r.warnings for r in reports)
        if has_errors:
            sys.exit(1)
        elif args.fail_on_warning and has_warnings:
            sys.exit(2)
        else:
            sys.exit(0)

    elif target.is_file():
        report = validate_file(str(target), args.verbose)
        print_report(report, args.verbose)

        if args.json_output:
            export_json_report([report], args.json_output)

        if not report.passed:
            sys.exit(1)
        elif args.fail_on_warning and report.warnings:
            sys.exit(2)
        else:
            sys.exit(0)

    else:
        print(f"Error: path does not exist — {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
