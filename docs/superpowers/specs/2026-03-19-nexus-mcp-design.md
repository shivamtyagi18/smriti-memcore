
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
| `nexus/integrations/__init__.py` | Export server entry point behind try/import guard (same pattern as LangChain integration) |

### Transport

**stdio** — Claude Code spawns `python -m nexus.integrations.mcp_server` as a child process and communicates over stdin/stdout using the MCP protocol. Claude Code manages the process lifecycle.

### Startup Sequence

```
Claude Code spawns subprocess
  → mcp_server.py reads env vars
  → calls os.path.expanduser() on NEXUS_STORAGE_PATH (must be explicit — ~ is not auto-expanded in subprocess env)
  → infers LLM API key from model name prefix (see Configuration section)
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
        "NEXUS_LLM_MODEL": "mistral",
        "NEXUS_LLM_API_KEY": ""
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
| `NEXUS_STORAGE_PATH` | `~/.nexus/global` | Where NEXUS persists data. Must be expanded with `os.path.expanduser()` at startup. |
| `NEXUS_LLM_MODEL` | `mistral` | Model name — provider is inferred from prefix (see below). |
| `NEXUS_LLM_API_KEY` | `""` | API key for cloud providers. Leave empty for Ollama. |

### LLM Provider Inference

`LLMInterface` routes by model name prefix (see `llm_interface.py:61-68`). There is no `llm_provider` field in `NexusConfig`. The server infers which API key field to set based on the model name:

| Model prefix | Provider | NexusConfig field set |
|---|---|---|
| `claude` | Anthropic | `anthropic_api_key=NEXUS_LLM_API_KEY` |
| `gpt-` | OpenAI | `openai_api_key=NEXUS_LLM_API_KEY` |
| `gemini` | Google | `gemini_api_key=NEXUS_LLM_API_KEY` |
| anything else | Ollama (local) | no key needed |

Example: `NEXUS_LLM_MODEL=claude-sonnet-4-6` + `NEXUS_LLM_API_KEY=sk-ant-...` → sets `anthropic_api_key`.

---

## MCP Tools

### Core Memory

**`nexus_encode`**
- Input: `content: str`, `source?: str` (default `"direct"`), `modality?: str` (default `"text"`)
- Output: `{ "memory_id": str }` if encoded, `{ "memory_id": null, "status": "discarded" }` if the Attention Gate rejected the content as low-salience
- Source defaults to `"direct"` (matching `MemorySource.DIRECT`, the actual API default). Use `"user_stated"` only when encoding explicit user statements — it sets `confidence=1.0` vs `0.8` for other sources.
- The tool wrapper calls `nexus.save()` after `nexus.encode()` returns (`encode()` itself does not persist — save is the wrapper's responsibility).

**`nexus_recall`**
- Input: `query: str`, `top_k?: int` (default `10`)
- Output: List of memory dicts: `{ id, content, strength, room_id, reflection_level, source, last_accessed }`
- Every retrieval strengthens the memory (testing effect). Enums serialized as `.value` strings; datetimes as ISO strings.

**`nexus_get_context`**
- Input: _(none)_
- Output: `{ "context": str }` — formatted working memory string ready to prepend to a prompt
- Returns the current capacity-bounded working memory (7±2 slots) as a formatted block.

### Confidence & Gaps

**`nexus_how_well_do_i_know`**
- Input: `topic: str`
- Output: `{ coverage, freshness, strength, depth, overall, decision }`
- Internally makes **two calls**: `meta_memory.confidence_map(topic)` → `ConfidenceLevel` (provides `coverage`, `freshness`, `strength`, `depth`, `overall`), then `meta_memory.should_recall_or_ask(topic)` → `DecisionType` (provides `decision`). The `decision` field is **not** on `ConfidenceLevel` — it is a separate call. Both must be made and merged into the output dict.

**`nexus_knowledge_gaps`**
- Input: _(none)_
- Output: List of gap dicts: `{ "topic": str, "context": str, "discovered_at": str (ISO), "resolved": bool }`
- This is the raw shape returned by `meta_memory.knowledge_gaps()` → `List[Dict]`. The `discovered_at` field is already an ISO string (set at registration time).

### Memory Management

**`nexus_pin`**
- Input: `memory_id: str`
- Output (success): `{ "status": "pinned", "memory_id": str }` — constructed by the tool wrapper after calling `nexus.pin()` (which returns `None`) and then verifying via `palace.get_memory(memory_id).status == MemoryStatus.PINNED`
- Output (failure): `{ "error": "Memory not found: <id>" }` when `palace.get_memory()` returns `None`

**`nexus_forget`**
- Input: `memory_id: str`
- Output (success): `{ "status": "archived", "memory_id": str }` — `nexus.forget()` sets `memory.status = MemoryStatus.ARCHIVED` (not a tombstone — that path exists in models but is not triggered by this call)
- Output (failure): `{ "error": "Memory not found: <id>" }` when `palace.get_memory()` returns `None`

**`nexus_consolidate`**
- Input: `depth?: str` (`"light"` or `"full"`, default `"light"`)
- Output: `{ "depth": str, "processed": int, "summary": str }`
- `ConsolidationDepth.DEFER` is intentionally excluded — it means "let the scheduler decide later" and is not meaningful as an explicit MCP call. Passing an invalid string returns `{ "error": "depth must be 'light' or 'full'" }`.

### Introspection

**`nexus_stats`**
- Input: _(none)_
- Output: The raw nested dict returned by `nexus.stats()`. Shape (from `core.py:315-332`):
  ```json
  {
    "palace": { ... },
    "working_memory": { ... },
    "retrieval": { ... },
    "consolidation": { ... },
    "meta_memory": { ... },
    "episode_buffer": { "total_episodes": int, "unconsolidated": int },
    "vector_store": { "total_vectors": int },
    "metrics": { ... }
  }
  ```
  The full dict is passed through as-is — it is already built from primitives (no dataclasses or enums).

**`nexus_get_suggestions`**
- Input: _(none)_
- Output: List of memory dicts using the same serialization as `nexus_recall`: `{ id, content, strength, room_id, reflection_level, source, last_accessed }`
- `nexus.get_suggestions()` returns `List[Memory]` — the tool serializes each `Memory` using the same `serialize_memory()` helper as `nexus_recall`.

---

## Data Flow

### Per-Tool Call

```
Claude calls nexus_recall(query="billing auth flow")
  → mcp_server deserializes args
  → calls nexus.recall(query, top_k)
  → converts Memory dataclasses → plain dicts via serialize_memory() helper
  → returns JSON-serializable result via MCP response
```

### Serialization Helper

A single `serialize_memory(memory: Memory) -> dict` helper handles all `Memory` → dict conversion, used by both `nexus_recall` and `nexus_get_suggestions`:
- Enums → `.value` (string)
- `datetime` → `.isoformat()`
- `float` fields → passed through (JSON-safe)
- Nested dataclasses (e.g. `SalienceScore`) → recursive dict conversion

---

## Error Handling

Three tiers:

| Tier | Scenario | Behavior |
|------|----------|----------|
| Startup failure | Bad storage path, missing `mcp` dep | Log to stderr, exit code 1. Claude Code surfaces "MCP server failed to start". |
| Tool call failure | NEXUS internal error, LLM timeout, bad memory ID | Caught per-tool, returns `{"error": "<message>"}`. Server stays alive. |
| Shutdown | Claude Code kills subprocess | `atexit` handler calls `nexus.save()`. No memory lost. |

---

## Testing

**File:** `tests/test_mcp_server.py`

**Strategy:** Import tool functions directly — no live MCP transport needed. Tests instantiate NEXUS with a temp `storage_path` and call tool functions as plain Python functions. Use `NexusConfig(llm_model="none")` to avoid Ollama dependency in tests — `LLMInterface` will be initialized but never called since tests don't trigger consolidation paths that require LLM.

| Test | Verifies |
|------|---------|
| `test_encode_returns_memory_id` | Returns a string ID when content is salient |
| `test_encode_discarded` | Returns `{"memory_id": null, "status": "discarded"}` for empty/whitespace content |
| `test_recall_returns_serializable` | Output is JSON-serializable (no datetimes/enums) |
| `test_recall_empty_store` | Returns empty list, not an exception |
| `test_consolidate_light` | Completes without error, returns summary dict |
| `test_how_well_do_i_know_fields` | Returns all 6 fields: coverage, freshness, strength, depth, overall, decision |
| `test_how_well_do_i_know_decision_values` | `decision` is one of the 3 valid `DecisionType` values |
| `test_knowledge_gaps_shape` | Each gap dict has `topic`, `context`, `discovered_at`, `resolved` keys |
| `test_pin_success` | Memory status is PINNED after call; returns `{"status": "pinned", ...}` |
| `test_forget_sets_archived` | Memory status is ARCHIVED (not deleted) after call |
| `test_pin_not_found` | Returns `{"error": ...}` for unknown memory_id |
| `test_stats_top_level_keys` | All 8 top-level keys present in stats output |
| `test_get_suggestions_serializable` | Each suggestion is a JSON-serializable dict |
| `test_error_handling` | Bad `memory_id` to `nexus_pin` returns `{"error": ...}` not exception |
| `test_env_var_storage_path` | `NEXUS_STORAGE_PATH` env var + `expanduser()` is respected |
| `test_llm_model_anthropic_routing` | `NEXUS_LLM_MODEL=claude-*` sets `anthropic_api_key` in config |
| `test_consolidate_invalid_depth` | `depth="defer"` returns `{"error": ...}` |

---

## Dependencies

```toml
[project.optional-dependencies]
mcp = ["mcp>=1.0.0"]
```

Install: `pip install nexus-memory[mcp]`

---

## Intended Usage Pattern

```
Session starts
  → Claude calls nexus_recall(project_name) to load relevant context

During work
  → Claude calls nexus_encode(content) when learning something worth remembering
  → nexus_encode may return {"memory_id": null, "status": "discarded"} — this is normal

Before answering a complex question
  → Claude calls nexus_how_well_do_i_know(topic) to check confidence + get decision

End of session
  → Claude calls nexus_consolidate("light") + nexus_get_suggestions()

Periodically (when palace is large)
  → Claude calls nexus_consolidate("full") + nexus_stats()
```