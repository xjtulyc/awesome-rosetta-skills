#!/usr/bin/env bash
# install.sh — Install awesome-rosetta-skills to your AI agent's skills directory.
#
# Usage:
#   bash scripts/install.sh                          # install all, auto-detect agent
#   bash scripts/install.sh --agent claude-code      # install for a specific agent
#   bash scripts/install.sh --category economics     # install only one discipline
#   bash scripts/install.sh --list                   # list available disciplines
#   bash scripts/install.sh --dry-run                # preview what would be installed
#
# Supported agents: claude-code, codex, gemini-cli, cursor
#
# MIT License — awesome-rosetta-skills contributors

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SKILLS_DIR="$REPO_ROOT/skills"

AGENT=""
CATEGORY="all"
DRY_RUN=false
LIST_ONLY=false

# ── Agent destination paths ────────────────────────────────────────────────────

CLAUDE_CODE_SKILLS_DIR="${CLAUDE_CODE_SKILLS_DIR:-$HOME/.claude/skills}"
CODEX_SKILLS_DIR="${CODEX_SKILLS_DIR:-$HOME/.codex/skills}"
GEMINI_SKILLS_DIR="${GEMINI_SKILLS_DIR:-$HOME/.gemini/skills}"
CURSOR_RULES_DIR="${CURSOR_RULES_DIR:-./.cursor/rules}"

# ── Helpers ────────────────────────────────────────────────────────────────────

log()  { echo "[install] $*"; }
warn() { echo "[install] WARNING: $*" >&2; }
err()  { echo "[install] ERROR: $*" >&2; exit 1; }

detect_agent() {
    if command -v claude &>/dev/null; then
        echo "claude-code"
    elif command -v codex &>/dev/null; then
        echo "codex"
    elif command -v gemini &>/dev/null; then
        echo "gemini-cli"
    else
        echo ""
    fi
}

list_disciplines() {
    echo "Available disciplines (category filter values):"
    echo ""
    for d in "$SKILLS_DIR"/*/; do
        name=$(basename "$d")
        count=$(find "$d" -name "SKILL.md" 2>/dev/null | wc -l | tr -d ' ')
        # Extract short name for --category flag
        short="${name#??-}"
        printf "  %-30s  %s skills\n" "$short" "$count"
    done
}

install_skill() {
    local skill_md="$1"
    local dest_dir="$2"
    local skill_name
    skill_name=$(basename "$(dirname "$skill_md")")

    local target_dir="$dest_dir/$skill_name"

    if "$DRY_RUN"; then
        log "[DRY RUN] Would copy: $skill_md -> $target_dir/SKILL.md"
        return
    fi

    mkdir -p "$target_dir"
    cp "$skill_md" "$target_dir/SKILL.md"
    log "Installed: $skill_name -> $target_dir/"
}

install_all_to() {
    local dest_dir="$1"
    local category="$2"

    mkdir -p "$dest_dir"
    local count=0

    for discipline_dir in "$SKILLS_DIR"/*/; do
        local discipline
        discipline=$(basename "$discipline_dir")
        local short="${discipline#??-}"

        # Category filter
        if [[ "$category" != "all" && "$short" != "$category" && "$discipline" != *"$category"* ]]; then
            continue
        fi

        for skill_dir in "$discipline_dir"*/; do
            local skill_md="$skill_dir/SKILL.md"
            if [[ -f "$skill_md" ]]; then
                install_skill "$skill_md" "$dest_dir"
                ((count++)) || true
            fi
        done
    done

    log "Installed $count skill(s) to $dest_dir"
}

# ── Argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --agent)
            AGENT="$2"; shift 2 ;;
        --category)
            CATEGORY="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --list)
            LIST_ONLY=true; shift ;;
        --help|-h)
            sed -n '2,25p' "$0"   # Print usage comment at top of file
            exit 0 ;;
        *)
            warn "Unknown option: $1"; shift ;;
    esac
done

# ── List only ─────────────────────────────────────────────────────────────────

if "$LIST_ONLY"; then
    list_disciplines
    exit 0
fi

# ── Auto-detect agent if not specified ────────────────────────────────────────

if [[ -z "$AGENT" ]]; then
    AGENT=$(detect_agent)
    if [[ -z "$AGENT" ]]; then
        warn "Could not auto-detect agent. Please specify --agent <name>."
        warn "Supported: claude-code, codex, gemini-cli, cursor"
        warn "Defaulting to claude-code."
        AGENT="claude-code"
    else
        log "Auto-detected agent: $AGENT"
    fi
fi

# ── Install ────────────────────────────────────────────────────────────────────

case "$AGENT" in
    claude-code|claude)
        log "Installing to Claude Code: $CLAUDE_CODE_SKILLS_DIR"
        install_all_to "$CLAUDE_CODE_SKILLS_DIR" "$CATEGORY"
        ;;
    codex|openai-codex)
        log "Installing to OpenAI Codex: $CODEX_SKILLS_DIR"
        install_all_to "$CODEX_SKILLS_DIR" "$CATEGORY"
        ;;
    gemini-cli|gemini)
        log "Installing to Gemini CLI: $GEMINI_SKILLS_DIR"
        install_all_to "$GEMINI_SKILLS_DIR" "$CATEGORY"
        ;;
    cursor)
        log "Installing to Cursor rules: $CURSOR_RULES_DIR"
        install_all_to "$CURSOR_RULES_DIR" "$CATEGORY"
        ;;
    all)
        log "Installing to all supported agents..."
        install_all_to "$CLAUDE_CODE_SKILLS_DIR" "$CATEGORY"
        install_all_to "$CODEX_SKILLS_DIR" "$CATEGORY"
        install_all_to "$GEMINI_SKILLS_DIR" "$CATEGORY"
        ;;
    *)
        err "Unknown agent: $AGENT. Supported: claude-code, codex, gemini-cli, cursor, all"
        ;;
esac

# ── Post-install message ──────────────────────────────────────────────────────

if ! "$DRY_RUN"; then
    echo ""
    echo "Installation complete!"
    echo ""
    echo "Next steps:"
    case "$AGENT" in
        claude-code|claude)
            echo "  Restart Claude Code, then try: 'search for papers on X using OpenAlex'"
            ;;
        codex)
            echo "  Reload Codex, then type: '@skills list' to see installed skills"
            ;;
        gemini-cli)
            echo "  Reload Gemini CLI to pick up new skills."
            ;;
        cursor)
            echo "  Reload Cursor. Skills appear as .cursor/rules/*.md"
            ;;
    esac
    echo ""
    echo "For more info: https://github.com/xjtulyc/awesome-rosetta-skills"
fi
