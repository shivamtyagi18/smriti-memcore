#!/usr/bin/env bash
# install_nexus_mcp.sh — Register the NEXUS MCP server with Claude Code
#
# Usage:
#   bash install_nexus_mcp.sh
#
# What it does:
#   1. Creates a dedicated venv at ~/.nexus/venv
#   2. Installs nexus-memory[mcp] into it
#   3. Patches ~/.claude.json to register the nexus MCP server
#   4. Patches ~/.claude/settings.json to add recall/encode hooks
#   5. Patches ~/.claude/CLAUDE.md with NEXUS memory instructions
#
# Requirements: Python 3.9+, Claude Code

set -euo pipefail

VENV_DIR="$HOME/.nexus/venv"

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo "[nexus] $*"; }
success() { echo "[nexus] ✓ $*"; }
error()   { echo "[nexus] ✗ $*" >&2; exit 1; }

# ── 1. Create dedicated venv ──────────────────────────────────────────────────

PY=$(command -v python3 || command -v python || true)
[[ -z "$PY" ]] && error "Python 3.9+ not found. Install it first."

PY_VERSION=$("$PY" -c "import sys; print(sys.version_info.minor)")
[[ "$PY_VERSION" -lt 9 ]] && error "Python 3.9+ required (found 3.$PY_VERSION)."

if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
    info "Creating venv at $VENV_DIR..."
    "$PY" -m venv "$VENV_DIR"
    success "Venv created"
else
    info "Using existing venv at $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python3"

# ── 2. Install package into venv ──────────────────────────────────────────────

info "Installing nexus-memory[mcp]..."
"$PYTHON" -m pip install "nexus-memory[mcp]" --quiet --upgrade
# Ensure mcp is installed even if the PyPI release pre-dates the extra
"$PYTHON" -c "import mcp" 2>/dev/null || "$PYTHON" -m pip install "mcp>=1.0.0" --quiet
success "nexus-memory[mcp] installed"

# Verify imports
"$PYTHON" -c "import nexus" 2>/dev/null \
    || error "nexus not importable after install — check pip output above."
"$PYTHON" -c "import mcp" 2>/dev/null \
    || error "mcp not importable after install — check pip output above."

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
    1) LLM_MODEL="mistral";                   LLM_API_KEY="" ;;
    2) LLM_MODEL="claude-haiku-4-5-20251001"; read -rsp "Anthropic API key: " LLM_API_KEY; echo ;;
    3) LLM_MODEL="gpt-4o-mini";               read -rsp "OpenAI API key: " LLM_API_KEY; echo ;;
    4) read -rp "Model name: " LLM_MODEL;     read -rsp "API key (leave blank for Ollama): " LLM_API_KEY; echo ;;
    *) LLM_MODEL="mistral";                   LLM_API_KEY="" ;;
esac

read -rp "Memory storage path [~/.nexus/global]: " STORAGE_PATH
STORAGE_PATH="${STORAGE_PATH:-~/.nexus/global}"

# ── 4. Patch ~/.claude.json ───────────────────────────────────────────────────

info "Registering nexus MCP server in ~/.claude.json..."

"$PYTHON" - <<PYEOF
import json, os

claude_json = os.path.expanduser("~/.claude.json")

if os.path.exists(claude_json):
    with open(claude_json) as f:
        config = json.load(f)
else:
    config = {}

if "mcpServers" not in config:
    config["mcpServers"] = {}

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
if "$PYTHON" -c "
import os
os.environ['NEXUS_STORAGE_PATH'] = '/tmp/nexus_install_test'
os.environ['NEXUS_LLM_MODEL'] = '$LLM_MODEL'
os.environ['NEXUS_LLM_API_KEY'] = '$LLM_API_KEY'
from nexus.integrations.mcp_server import build_nexus_config
cfg = build_nexus_config()
assert cfg.llm_model == '$LLM_MODEL'
print('ok')
" 2>/dev/null | grep -q ok; then
    success "Server verified"
else
    echo "[nexus] ⚠ Could not verify server — check your LLM config after launch"
fi

# ── 6. Patch ~/.claude/settings.json with hooks ───────────────────────────────

info "Configuring Claude Code hooks in ~/.claude/settings.json..."

"$PYTHON" - <<PYEOF
import json, os

settings_path = os.path.expanduser("~/.claude/settings.json")
os.makedirs(os.path.dirname(settings_path), exist_ok=True)

if os.path.exists(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)
else:
    settings = {}

if "hooks" not in settings:
    settings["hooks"] = {}

# UserPromptSubmit — inject nexus_recall reminder into Claude's context
nexus_prompt_hook = {
    "hooks": [{
        "type": "command",
        "command": "echo '{\"hookSpecificOutput\": {\"hookEventName\": \"UserPromptSubmit\", \"additionalContext\": \"NEXUS MEMORY: Call nexus_recall with keywords from the user message to retrieve relevant memories before responding.\"}}'",
        "statusMessage": "Recalling NEXUS memories..."
    }]
}

if "UserPromptSubmit" not in settings["hooks"]:
    settings["hooks"]["UserPromptSubmit"] = []

# Only add if not already present
existing_cmds = [h["hooks"][0]["command"] for h in settings["hooks"]["UserPromptSubmit"] if h.get("hooks")]
if not any("nexus_recall" in cmd for cmd in existing_cmds):
    settings["hooks"]["UserPromptSubmit"].append(nexus_prompt_hook)

# Stop — remind Claude to encode takeaways
nexus_stop_hook = {
    "hooks": [{
        "type": "command",
        "command": "echo '{\"systemMessage\": \"NEXUS: Remember to call nexus_encode to store key facts, decisions, and context from this session before ending.\"}'",
        "statusMessage": "Saving NEXUS memory..."
    }]
}

if "Stop" not in settings["hooks"]:
    settings["hooks"]["Stop"] = []

existing_stop_cmds = [h["hooks"][0]["command"] for h in settings["hooks"]["Stop"] if h.get("hooks")]
if not any("nexus_encode" in cmd for cmd in existing_stop_cmds):
    settings["hooks"]["Stop"].append(nexus_stop_hook)

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)

print(f"[nexus] ✓ Hooks written to {settings_path}")
PYEOF

# ── 7. Patch ~/.claude/CLAUDE.md with NEXUS memory instructions ───────────────

info "Adding NEXUS memory instructions to ~/.claude/CLAUDE.md..."

CLAUDE_MD="$HOME/.claude/CLAUDE.md"
NEXUS_SECTION="
## NEXUS Memory

Use the nexus MCP tools to maintain memory across sessions:
- At the start of each session, call \`nexus_recall\` with relevant keywords to retrieve prior context
- Throughout the session, call \`nexus_encode\` when you learn important facts about the user, their preferences, decisions made, or project context
- At the end of each session, call \`nexus_encode\` to store any key takeaways"

if [[ ! -f "$CLAUDE_MD" ]]; then
    echo "# Global Claude Instructions" > "$CLAUDE_MD"
fi

if ! grep -q "NEXUS Memory" "$CLAUDE_MD" 2>/dev/null; then
    echo "$NEXUS_SECTION" >> "$CLAUDE_MD"
    success "NEXUS instructions added to $CLAUDE_MD"
else
    info "NEXUS instructions already present in $CLAUDE_MD — skipping"
fi

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " NEXUS MCP server registered successfully!"
echo " Restart Claude Code to activate it."
echo " Then run /mcp to confirm it appears."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
