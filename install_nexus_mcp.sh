#!/usr/bin/env bash
# install_nexus_mcp.sh — Register the NEXUS MCP server with Claude Code
#
# Usage:
#   bash install_nexus_mcp.sh
#
# What it does:
#   1. Installs nexus-memory[mcp] with pip
#   2. Locates the Python executable that owns the install
#   3. Patches ~/.claude.json to register the nexus MCP server
#
# Requirements: Python 3.9+, pip, Claude Code (creates ~/.claude.json if missing)

set -euo pipefail

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo "[nexus] $*"; }
success() { echo "[nexus] ✓ $*"; }
error()   { echo "[nexus] ✗ $*" >&2; exit 1; }

# ── 1. Install package ────────────────────────────────────────────────────────

info "Installing nexus-memory[mcp]..."
pip install "nexus-memory[mcp]" --quiet || error "pip install failed. Is pip available?"
success "nexus-memory[mcp] installed"

# ── 2. Locate the Python that owns the install ────────────────────────────────

PYTHON=$(python3 -c "import sys; print(sys.executable)")

# Verify nexus actually imports from this Python
if ! "$PYTHON" -c "import nexus" 2>/dev/null; then
    # pip may have installed into a different Python — search common locations
    for candidate in \
        "$(pip show nexus-memory 2>/dev/null | awk '/^Location:/{print $2}')/../../../bin/python3" \
        "$HOME/.local/bin/python3" \
        "/usr/local/bin/python3" \
        "/usr/bin/python3"; do
        candidate=$(realpath "$candidate" 2>/dev/null || true)
        if [[ -x "$candidate" ]] && "$candidate" -c "import nexus" 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
fi

"$PYTHON" -c "import nexus" 2>/dev/null \
    || error "nexus package not importable from $PYTHON. Try: pip3 install 'nexus-memory[mcp]' and re-run."
"$PYTHON" -c "import mcp" 2>/dev/null \
    || error "mcp package not importable from $PYTHON. Try: pip3 install 'nexus-memory[mcp]' and re-run."

success "Using Python: $PYTHON"

# ── 3. Prompt for LLM config ──────────────────────────────────────────────────

echo ""
echo "Which LLM should NEXUS use for memory consolidation?"
echo "  1) mistral (local Ollama — default, no API key needed)"
echo "  2) claude-haiku-4-5-20251001 (Anthropic API key required)"
echo "  3) gpt-4o-mini (OpenAI API key required)"
echo "  4) Enter custom model name"
echo ""
read -rp "Choice [1]: " MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-1}"

case "$MODEL_CHOICE" in
    1) LLM_MODEL="mistral";                  LLM_API_KEY="" ;;
    2) LLM_MODEL="claude-haiku-4-5-20251001"; read -rsp "Anthropic API key: " LLM_API_KEY; echo ;;
    3) LLM_MODEL="gpt-4o-mini";              read -rsp "OpenAI API key: " LLM_API_KEY; echo ;;
    4) read -rp "Model name: " LLM_MODEL;    read -rsp "API key (leave blank for Ollama): " LLM_API_KEY; echo ;;
    *) LLM_MODEL="mistral";                  LLM_API_KEY="" ;;
esac

read -rp "Memory storage path [~/.nexus/global]: " STORAGE_PATH
STORAGE_PATH="${STORAGE_PATH:-~/.nexus/global}"

# ── 4. Patch ~/.claude.json ───────────────────────────────────────────────────

CLAUDE_JSON="$HOME/.claude.json"

info "Registering nexus MCP server in $CLAUDE_JSON..."

"$PYTHON" - <<PYEOF
import json, os, sys

claude_json = os.path.expanduser("~/.claude.json")

# Load existing config or start fresh
if os.path.exists(claude_json):
    with open(claude_json) as f:
        config = json.load(f)
else:
    config = {}

# Ensure mcpServers key exists
if "mcpServers" not in config:
    config["mcpServers"] = {}

# Write nexus entry
config["mcpServers"]["nexus"] = {
    "command": "$PYTHON",
    "args": ["-m", "nexus.integrations.mcp_server"],
    "env": {
        "PYTHONPATH": "",
        "NEXUS_STORAGE_PATH": "$STORAGE_PATH",
        "NEXUS_LLM_MODEL": "$LLM_MODEL",
        "NEXUS_LLM_API_KEY": "$LLM_API_KEY",
    },
}

with open(claude_json, "w") as f:
    json.dump(config, f, indent=2)

print(f"[nexus] ✓ Written to {claude_json}")
PYEOF

# ── 5. Smoke test ─────────────────────────────────────────────────────────────

info "Verifying server starts..."
if NEXUS_STORAGE_PATH=/tmp/nexus_install_test NEXUS_LLM_MODEL="$LLM_MODEL" \
    "$PYTHON" -c "
import sys, os
os.environ['NEXUS_STORAGE_PATH'] = '/tmp/nexus_install_test'
os.environ['NEXUS_LLM_MODEL'] = '$LLM_MODEL'
os.environ['NEXUS_LLM_API_KEY'] = '$LLM_API_KEY'
from nexus.integrations.mcp_server import build_nexus_config, mcp_server
cfg = build_nexus_config()
assert cfg.llm_model == '$LLM_MODEL'
print('ok')
" 2>/dev/null | grep -q ok; then
    success "Server verified"
else
    echo "[nexus] ⚠ Could not verify server start — check your LLM config after launch"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " NEXUS MCP server registered successfully!"
echo " Restart Claude Code to activate it."
echo " Then run /mcp to confirm it appears."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
