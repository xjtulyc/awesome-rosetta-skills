#!/usr/bin/env python3
"""
check_compat.py — Check SKILL.md compatibility across agent platforms.

Verifies that each SKILL.md:
  1. Declares all supported platforms in its frontmatter
  2. Does not use platform-specific syntax that would break on other agents
  3. Has valid platform names

Usage:
    python scripts/check_compat.py skills/
    python scripts/check_compat.py skills/07-economics/did-causal/SKILL.md
"""

import re
import sys
import json
from pathlib import Path
from typing import Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

SUPPORTED_PLATFORMS = {"claude-code", "codex", "gemini-cli", "cursor", "vscode-copilot"}

# Platform-specific syntax patterns to warn about
PLATFORM_SPECIFIC_PATTERNS = [
    # Claude Code specific
    (r"/\w+\s", "claude-code",
     "Slash command syntax (/command) is Claude Code specific"),
    # Cursor specific
    (r"\.mdc\b", "cursor",
     "MDC file references are Cursor-specific"),
    # VS Code specific
    (r"\.github/copilot", "vscode-copilot",
     "GitHub Copilot path is VS Code specific"),
]


def parse_frontmatter(content: str) -> dict:
    fm_match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not fm_match:
        return {}
    fm_text = fm_match.group(1)
    if HAS_YAML:
        try:
            return yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            return {}
    fm = {}
    for line in fm_text.splitlines():
        m = re.match(r"^(\w[\w_-]*):\s*(.+)$", line)
        if m:
            fm[m.group(1)] = m.group(2).strip().strip("\"'")
    return fm


def check_file(file_path: Path, verbose: bool = False) -> dict:
    content = file_path.read_text(encoding="utf-8")
    fm = parse_frontmatter(content)

    issues = []

    # Check platforms field
    platforms = fm.get("platforms", [])
    if not platforms:
        issues.append({
            "level": "WARNING",
            "message": "No 'platforms' field declared in frontmatter",
        })
    else:
        invalid = [p for p in platforms if p not in SUPPORTED_PLATFORMS]
        for p in invalid:
            issues.append({
                "level": "WARNING",
                "message": f"Unknown platform: '{p}'. Valid: {sorted(SUPPORTED_PLATFORMS)}",
            })

        missing = SUPPORTED_PLATFORMS - set(platforms) - {"vscode-copilot"}
        for p in missing:
            issues.append({
                "level": "INFO",
                "message": f"Platform not declared: '{p}'",
            })

    # Check for platform-specific syntax in body
    body = content[content.find("---", 3) + 3:]  # after frontmatter
    for pattern, platform, message in PLATFORM_SPECIFIC_PATTERNS:
        matches = re.findall(pattern, body, re.IGNORECASE)
        if matches:
            declared_platforms = fm.get("platforms", [])
            if platform not in declared_platforms:
                issues.append({
                    "level": "WARNING",
                    "message": f"{message} (found {len(matches)} occurrence(s)). "
                               f"Add '{platform}' to platforms or use cross-platform syntax.",
                })

    passed = not any(i["level"] == "ERROR" for i in issues)
    return {
        "file": str(file_path),
        "passed": passed,
        "issues": issues,
        "platforms": platforms,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check SKILL.md platform compatibility")
    parser.add_argument("target", help="SKILL.md file or directory to check")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json-output", metavar="FILE",
                        help="Write JSON report to file")
    args = parser.parse_args()

    target = Path(args.target)
    files = []

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = list(target.rglob("SKILL.md"))
    else:
        print(f"Error: {target} is not a file or directory", file=sys.stderr)
        sys.exit(1)

    reports = []
    for f in sorted(files):
        report = check_file(f, verbose=args.verbose)
        reports.append(report)

        if args.verbose:
            status = "✅" if report["passed"] else "❌"
            print(f"  {status} {f}")
            for issue in report["issues"]:
                if issue["level"] != "INFO" or args.verbose:
                    print(f"      [{issue['level']}] {issue['message']}")

    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as out:
            json.dump(reports, out, indent=2)

    passed = sum(1 for r in reports if r["passed"])
    failed = len(reports) - passed
    print(f"\nCompatibility check: {len(reports)} files, {passed} passed, {failed} with issues")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
