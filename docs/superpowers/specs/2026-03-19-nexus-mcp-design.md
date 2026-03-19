# NEXUS MCP Server — Design Spec

**Date:** 2026-03-19
**Status:** Approved
**Author:** Shivam Tyagi

---

## Overview

Wrap the NEXUS memory system as a Claude Code MCP server so Claude can encode memories, recall context, and introspect memory state interactively during conversations. The server runs as a stdio subprocess managed by Claude Code and exposes the full NEXUS API surface as 10 MCP tools.

---

## Architecture

### New Files

| File | Purpose |
|------|---------|
| `nexus/integrations/mcp_server.py` | MCP server — main deliverable |

### Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Add `[mcp]` optional dependency group |
| `nexus/integrations/__init__.py` | Export server entry point |

### Transport

**stdio** — Claude Code spawns `python -m nexus.integrations.mcp_server` as a child process and communicates over stdin/stdout using the MCP protocol. Claude Code manages the process lifecycle.

### Startup Sequence

```
Claude Code spawns subprocess
  → mcp_server.py reads env vars
  → expands ~ in NEXUS_STORAGE_PATH
  → initializes NEXUS(NexusConfig(...))
  → registers 10 tools with @mcp.tool() decorators
  → mcp.run(transport="stdio")  ← blocks on stdin
```

### Claude Code Registration (`~/.claude/settings.json`)

```json
{
  "mcpServers": {
    "nexus": {
      "command": "python",
      "args": ["-m", "nexus.integrations.mcp_server"],
      "env": {
        "NEXUS_STORAGE_PATH": "~/.nexus/global",
        "NEXUS_LLM_PROVIDER": "ollama",
        "NEXUS_LLM_MODEL": "mistral"
      }
    }
  }
}
```

---

## Configuration

All configuration via environment variables:

| Env Var | Default | Purpose |
|---------|---------|---------|
| `NEXUS_STORAGE_PATH` | `~/.nexus/global` | Where NEXUS persists data. Shared across projects by default; override per-project if isolation needed. |
| `NEXUS_LLM_PROVIDER` | `ollama` | LLM backend for consolidation/reflection (`ollama`, `anthropic`, `openai`, `gemini`) |
| `NEXUS_LLM_MODEL` | `mistral` | Model name within the chosen provider |
| `NEXUS_LLM_API_KEY` | _(none)_ | API key if using a cloud provider |

---

## MCP Tools

### Core Memory

**`nexus_encode`**
- Input: `content: str`, `source?: str` (default `"user_stated"`), `modality?: str` (default `"text"`)
- Output: `{ "memory_id": str }`
- Encodes content into NEXUS episode buffer + semantic palace. Calls `nexus.save()` after encoding.

**`nexus_recall`**
- Input: `query: str`, `top_k?: int` (default `10`)
- Output: List of memory dicts: `{ id, content, strength, room_id, reflection_level, source, last_accessed }`
- Queries semantic palace + episode buffer. Every retrieval strengthens the memory (testing effect).

**`nexus_get_context`**
- Input: _(none)_
- Output: `{ "context": str }` — formatted working memory string ready to prepend to a prompt
- Returns the current capacity-bounded working memory (7±2 slots) as a formatted block.

### Confidence & Gaps

**`nexus_how_well_do_i_know`**
- Input: `topic: str`
- Output: `{ coverage, freshness, strength, depth, overall, decision }`
- Returns a 5-dimensional confidence score and a decision (`recall_confidently`, `recall_but_verify`, `admit_gap_and_ask`).

**`nexus_knowledge_gaps`**
- Input: _(none)_
- Output: List of topic strings NEXUS has registered as unknown
- Surfaces failed retrievals that were logged by MetaMemory.

### Memory Management

**`nexus_pin`**
- Input: `memory_id: str`
- Output: `{ "status": "pinned", "memory_id": str }` or `{ "error": str }`
- Marks a memory as PINNED — exempt from decay and forgetting.

**`nexus_forget`**
- Input: `memory_id: str`
- Output: `{ "status": "forgotten", "memory_id": str }` or `{ "error": str }`
- Gracefully forgets a memory (leaves tombstone, does not hard-delete).

**`nexus_consolidate`**
- Input: `depth?: str` (`"light"` or `"full"`, default `"light"`)
- Output: `{ "depth": str, "processed": int, "summary": str }`
- Triggers a consolidation cycle. `light` = chunking + conflict detection. `full` = all 8 processes.

### Introspection

**`nexus_stats`**
- Input: _(none)_
- Output: `{ total_memories, active, archived, pinned, decaying, palace_rooms, vector_count, episode_count }`
- Returns a snapshot of NEXUS system state.

**`nexus_get_suggestions`**
- Input: _(none)_
- Output: List of suggestion strings from the ambient monitor
- Proactive insights NEXUS has surfaced from background processing.

---

## Data Flow

### Per-Tool Call

```
Claude calls nexus_recall(query="billing auth flow")
  → mcp_server deserializes args
  → calls nexus.recall(query, top_k)
  → converts Memory dataclasses → plain dicts
    (enums → .value strings, datetimes → ISO strings)
  → returns JSON-serializable result via MCP response
```

### Serialization

NEXUS `Memory` dataclasses contain `datetime`, `Enum`, and nested dataclass fields. The server serializes all outputs to plain dicts before returning:
- Enums → `.value` (string)
- `datetime` → `.isoformat()`
- Nested dataclasses → recursive dict conversion
- `float` fields → passed through (JSON-safe)

---

## Error Handling

Three tiers:

| Tier | Scenario | Behavior |
|------|----------|----------|
| Startup failure | Bad storage path, missing `mcp` dep | Log to stderr, exit code 1. Claude Code surfaces "MCP server failed to start". |
| Tool call failure | NEXUS internal error, LLM timeout | Caught per-tool, returns `{"error": "<message>"}`. Server stays alive. |
| Shutdown | Claude Code kills subprocess | `atexit` handler calls `nexus.save()`. No memory lost. |

---

## Testing

**File:** `tests/test_mcp_server.py`

**Strategy:** Import tool functions directly — no live MCP transport needed. Tests instantiate NEXUS with a temp `storage_path` and call tool functions as plain Python functions.

| Test | Verifies |
|------|---------|
| `test_encode_returns_memory_id` | Returns a string ID |
| `test_recall_returns_serializable` | Output is JSON-serializable (no datetimes/enums) |
| `test_recall_empty_store` | Graceful empty result, not exception |
| `test_consolidate_light` | Completes without error, returns summary dict |
| `test_how_well_do_i_know` | Returns all 5 confidence fields |
| `test_pin_and_forget` | Memory status changes correctly |
| `test_stats_structure` | All expected keys present |
| `test_error_handling` | Bad `memory_id` to `nexus_pin` returns `{"error": ...}` |
| `test_env_var_storage_path` | `NEXUS_STORAGE_PATH` env var is respected |

All tests use `NexusConfig(llm_model=None)` to skip LLM-dependent paths — no API calls in tests.

---

## Intended Usage Pattern

```
Session starts
  → Claude calls nexus_recall(project_name) to load relevant context

During work
  → Claude calls nexus_encode(content) when learning something worth remembering

Before answering a complex question
  → Claude calls nexus_how_well_do_i_know(topic) to check confidence

End of session
  → Claude calls nexus_consolidate("light") + nexus_get_suggestions()

Periodically (when palace is large)
  → Claude calls nexus_consolidate("full") + nexus_stats()
```

---

## Dependencies

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0.0"]
```

Install: `pip install nexus-memory[mcp]`