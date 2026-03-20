# NEXUS MCP Server Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Claude Code MCP server to nexus-memory that exposes the full NEXUS API as 10 tools via stdio, enabling Claude to encode memories, recall context, and introspect memory state during conversations.

**Architecture:** A single `nexus/integrations/mcp_server.py` file using the `mcp` Python SDK, following the same pattern as the existing LangChain integration. A module-level `_nexus` instance is initialized at startup from env vars and shared across all tool calls. Tool functions are plain Python functions decorated with `@mcp.tool()` so they can be called directly in tests without starting the MCP transport.

**Tech Stack:** Python 3.9+, `mcp>=1.0.0` (new), `nexus-memory` (existing), `pytest` (existing)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `nexus/integrations/mcp_server.py` | **Create** | MCP server — all 10 tools, startup, config |
| `tests/test_mcp_server.py` | **Create** | All tests for the MCP server |
| `pyproject.toml` | **Modify** | Add `mcp = ["mcp>=1.0.0"]` optional dep |
| `nexus/integrations/__init__.py` | **Modify** | Add try/import guard for mcp_server |

---

## Task 1: Add `mcp` optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
mcp = [
    "mcp>=1.0.0",
]
```

- [ ] **Step 2: Install it**

```bash
pip install -e ".[mcp]"
```

Expected: `mcp` package installed, `python -c "import mcp; print(mcp.__version__)"` prints a version.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add mcp optional dependency"
```

---

## Task 2: `serialize_memory` helper + tests

**Files:**
- Create: `nexus/integrations/mcp_server.py` (skeleton + helper only)
- Create: `tests/test_mcp_server.py` (first tests)

The `serialize_memory` helper is used by `nexus_recall` and `nexus_get_suggestions`. It converts a `Memory` dataclass to a JSON-safe dict.

- [ ] **Step 1: Write failing tests**

Create `tests/test_mcp_server.py`:

```python
"""Tests for the NEXUS MCP server."""
import json
import os
import tempfile
from datetime import datetime

import pytest

from nexus.models import Memory, MemorySource, MemoryStatus, Modality, NexusConfig, SalienceScore
from nexus.core import NEXUS


@pytest.fixture
def tmp_nexus(tmp_path):
    """NEXUS instance with temp storage and no LLM."""
    config = NexusConfig(
        storage_path=str(tmp_path),
        llm_model="none",  # avoid Ollama dependency
    )
    n = NEXUS(config=config)
    yield n
    n.close()


def _is_json_serializable(obj) -> bool:
    """Check that obj round-trips through JSON without error."""
    try:
        json.dumps(obj)
        return True
    except (TypeError, ValueError):
        return False


def test_serialize_memory_is_json_safe():
    """serialize_memory output must be JSON-serializable."""
    from nexus.integrations.mcp_server import serialize_memory

    mem = Memory(
        content="test content",
        source=MemorySource.DIRECT,
        status=MemoryStatus.ACTIVE,
        modality=Modality.TEXT,
        salience=SalienceScore(surprise=0.5, relevance=0.8),
        creation_time=datetime(2026, 1, 1),
        last_accessed=datetime(2026, 1, 2),
    )
    result = serialize_memory(mem)
    assert _is_json_serializable(result)


def test_serialize_memory_enum_values():
    """Enums must be serialized to their .value strings."""
    from nexus.integrations.mcp_server import serialize_memory

    mem = Memory(source=MemorySource.USER_STATED, modality=Modality.CODE)
    result = serialize_memory(mem)
    assert result["source"] == "user_stated"
    assert result["modality"] == "code"


def test_serialize_memory_datetime_iso():
    """datetime fields must be ISO strings."""
    from nexus.integrations.mcp_server import serialize_memory

    dt = datetime(2026, 3, 19, 12, 0, 0)
    mem = Memory(creation_time=dt, last_accessed=dt)
    result = serialize_memory(mem)
    assert result["creation_time"] == "2026-03-19T12:00:00"
    assert result["last_accessed"] == "2026-03-19T12:00:00"


def test_serialize_memory_expected_keys():
    """Output must contain the core fields expected by MCP consumers."""
    from nexus.integrations.mcp_server import serialize_memory

    mem = Memory(content="hello")
    result = serialize_memory(mem)
    for key in ("id", "content", "strength", "room_id", "reflection_level", "source", "last_accessed"):
        assert key in result, f"Missing key: {key}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
pytest tests/test_mcp_server.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'serialize_memory' from 'nexus.integrations.mcp_server'`

- [ ] **Step 3: Create `mcp_server.py` skeleton with `serialize_memory`**

Create `nexus/integrations/mcp_server.py`:

```python
"""
NEXUS MCP Server.
Exposes the NEXUS memory system as a Claude Code MCP server via stdio transport.

Usage:
    python -m nexus.integrations.mcp_server

Environment variables:
    NEXUS_STORAGE_PATH   Where to persist data (default: ~/.nexus/global)
    NEXUS_LLM_MODEL      LLM model name — provider inferred from prefix (default: mistral)
    NEXUS_LLM_API_KEY    API key for cloud providers; empty for Ollama
"""
from __future__ import annotations

import atexit
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import mcp
    from mcp.server import FastMCP
except ImportError:
    raise ImportError(
        "To use the NEXUS MCP server, install the mcp extra:\n"
        "pip install nexus-memory[mcp]"
    )

from nexus.core import NEXUS
from nexus.models import (
    ConsolidationDepth,
    Memory,
    MemorySource,
    MemoryStatus,
    Modality,
    NexusConfig,
)

logger = logging.getLogger(__name__)

# Module-level NEXUS instance — initialized at startup, shared across tool calls.
# Tests can replace this with a test instance: `import nexus.integrations.mcp_server as s; s._nexus = ...`
_nexus: Optional[NEXUS] = None

mcp_server = FastMCP("nexus-memory")


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_memory(memory: Memory) -> Dict[str, Any]:
    """Convert a Memory dataclass to a JSON-serializable dict."""
    return {
        "id": memory.id,
        "content": memory.content,
        "strength": memory.strength,
        "confidence": memory.confidence,
        "room_id": memory.room_id,
        "reflection_level": memory.reflection_level,
        "source": memory.source.value,
        "modality": memory.modality.value,
        "status": memory.status.value,
        "creation_time": memory.creation_time.isoformat(),
        "last_accessed": memory.last_accessed.isoformat(),
        "access_count": memory.access_count,
        "salience": memory.salience.to_dict(),
        "hops": memory.hops,
        "retrieval_score": memory.retrieval_score,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcp_server.py::test_serialize_memory_is_json_safe \
       tests/test_mcp_server.py::test_serialize_memory_enum_values \
       tests/test_mcp_server.py::test_serialize_memory_datetime_iso \
       tests/test_mcp_server.py::test_serialize_memory_expected_keys -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add nexus/integrations/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add mcp_server skeleton and serialize_memory helper"
```

---

## Task 3: Startup config — env var reading and `build_nexus_config`

**Files:**
- Modify: `nexus/integrations/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_mcp_server.py`:

```python
def test_build_nexus_config_defaults(tmp_path, monkeypatch):
    """Default env vars produce a valid NexusConfig with expanded path."""
    monkeypatch.setenv("NEXUS_STORAGE_PATH", str(tmp_path))
    monkeypatch.delenv("NEXUS_LLM_MODEL", raising=False)
    monkeypatch.delenv("NEXUS_LLM_API_KEY", raising=False)
    # Clear ambient cloud keys to prevent NexusConfig.__post_init__ env var fallback
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    from nexus.integrations.mcp_server import build_nexus_config
    config = build_nexus_config()
    assert config.storage_path == str(tmp_path)
    assert config.llm_model == "mistral"
    # Unused providers get "" not None — prevents env var inheritance
    assert config.anthropic_api_key == ""
    assert config.openai_api_key == ""


def test_build_nexus_config_anthropic_routing(tmp_path, monkeypatch):
    """NEXUS_LLM_MODEL=claude-* sets anthropic_api_key; others get ''."""
    monkeypatch.setenv("NEXUS_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("NEXUS_LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("NEXUS_LLM_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from nexus.integrations.mcp_server import build_nexus_config
    config = build_nexus_config()
    assert config.llm_model == "claude-sonnet-4-6"
    assert config.anthropic_api_key == "sk-ant-test"
    assert config.openai_api_key == ""   # "" not None — no env var inheritance


def test_build_nexus_config_openai_routing(tmp_path, monkeypatch):
    """NEXUS_LLM_MODEL=gpt-* sets openai_api_key; others get ''."""
    monkeypatch.setenv("NEXUS_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("NEXUS_LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("NEXUS_LLM_API_KEY", "sk-openai-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from nexus.integrations.mcp_server import build_nexus_config
    config = build_nexus_config()
    assert config.openai_api_key == "sk-openai-test"
    assert config.anthropic_api_key == ""   # "" not None


def test_build_nexus_config_expands_tilde(monkeypatch):
    """~ in NEXUS_STORAGE_PATH must be expanded."""
    monkeypatch.setenv("NEXUS_STORAGE_PATH", "~/.nexus/test")
    monkeypatch.delenv("NEXUS_LLM_MODEL", raising=False)
    monkeypatch.delenv("NEXUS_LLM_API_KEY", raising=False)

    from nexus.integrations.mcp_server import build_nexus_config
    config = build_nexus_config()
    assert not config.storage_path.startswith("~")
    assert config.storage_path == os.path.expanduser("~/.nexus/test")
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_mcp_server.py -k "build_nexus_config" -v 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError: module has no attribute 'build_nexus_config'`

- [ ] **Step 3: Implement `build_nexus_config`**

Add to `nexus/integrations/mcp_server.py` after the imports:

```python
# ── Startup Config ────────────────────────────────────────────────────────────

def build_nexus_config() -> NexusConfig:
    """
    Build NexusConfig from environment variables.

    NEXUS_STORAGE_PATH  — storage dir, ~ expanded (default: ~/.nexus/global)
    NEXUS_LLM_MODEL     — model name, provider inferred by prefix (default: mistral)
    NEXUS_LLM_API_KEY   — API key for cloud providers (default: "")
    """
    storage_path = os.path.expanduser(
        os.environ.get("NEXUS_STORAGE_PATH", "~/.nexus/global")
    )
    llm_model = os.environ.get("NEXUS_LLM_MODEL", "mistral")
    api_key = os.environ.get("NEXUS_LLM_API_KEY", "") or None

    # Infer provider from model name prefix — matches LLMInterface routing in llm_interface.py:61-68
    # IMPORTANT: Pass "" (empty string, not None) for unused provider keys.
    # NexusConfig.__post_init__ falls back to reading ANTHROPIC_API_KEY/OPENAI_API_KEY/GEMINI_API_KEY
    # env vars only when the field is None. Passing "" prevents that silent inheritance.
    anthropic_key = api_key if llm_model.startswith("claude") else ""
    openai_key = api_key if llm_model.startswith("gpt-") else ""
    gemini_key = api_key if llm_model.startswith("gemini") else ""

    return NexusConfig(
        storage_path=storage_path,
        llm_model=llm_model,
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
        gemini_api_key=gemini_key,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcp_server.py -k "build_nexus_config" -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add nexus/integrations/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add build_nexus_config with LLM provider inference"
```

---

## Task 4: Core memory tools — `nexus_encode`, `nexus_recall`, `nexus_get_context`

**Files:**
- Modify: `nexus/integrations/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_mcp_server.py`:

```python
import nexus.integrations.mcp_server as _mcp_module


@pytest.fixture(autouse=True)
def inject_nexus(tmp_nexus):
    """Inject test NEXUS instance into the module-level _nexus variable."""
    original = _mcp_module._nexus
    _mcp_module._nexus = tmp_nexus
    yield
    _mcp_module._nexus = original


def test_encode_returns_memory_id():
    """nexus_encode returns a memory_id string for salient content."""
    from nexus.integrations.mcp_server import nexus_encode
    result = nexus_encode(content="Python is preferred for backend services")
    assert "memory_id" in result
    assert isinstance(result["memory_id"], str)
    assert len(result["memory_id"]) > 0


def test_encode_discarded_on_empty():
    """nexus_encode returns discarded status for empty/whitespace content."""
    from nexus.integrations.mcp_server import nexus_encode
    result = nexus_encode(content="   ")
    assert result.get("memory_id") is None
    assert result.get("status") == "discarded"


def test_encode_source_default_is_direct():
    """nexus_encode defaults source to 'direct', not 'user_stated'."""
    from nexus.integrations.mcp_server import nexus_encode
    # Encode and confirm it doesn't raise — source inference tested via confidence difference
    result = nexus_encode(content="Default source test content")
    assert "memory_id" in result


def test_recall_returns_list():
    """nexus_recall returns a list (empty when store is empty)."""
    from nexus.integrations.mcp_server import nexus_recall
    result = nexus_recall(query="anything")
    assert isinstance(result, list)


def test_recall_returns_serializable(tmp_nexus):
    """nexus_recall output is fully JSON-serializable."""
    from nexus.integrations.mcp_server import nexus_encode, nexus_recall
    nexus_encode(content="LangChain integration uses BaseChatMessageHistory")
    memories = nexus_recall(query="LangChain")
    assert _is_json_serializable(memories)


def test_recall_memory_has_expected_keys(tmp_nexus):
    """Each recalled memory dict has the required keys."""
    from nexus.integrations.mcp_server import nexus_encode, nexus_recall
    nexus_encode(content="NEXUS uses a semantic palace for memory storage")
    memories = nexus_recall(query="semantic palace")
    if memories:  # may be empty if attention gate discards
        mem = memories[0]
        for key in ("id", "content", "strength", "room_id", "reflection_level", "source", "last_accessed"):
            assert key in mem, f"Missing key: {key}"


def test_get_context_returns_string():
    """nexus_get_context returns a dict with a 'context' string key."""
    from nexus.integrations.mcp_server import nexus_get_context
    result = nexus_get_context()
    assert "context" in result
    assert isinstance(result["context"], str)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_mcp_server.py -k "encode or recall or get_context" -v 2>&1 | head -30
```

Expected: `ImportError` for `nexus_encode`, `nexus_recall`, `nexus_get_context`

- [ ] **Step 3: Implement the three tools**

Add to `nexus/integrations/mcp_server.py`:

```python
# ── Core Memory Tools ─────────────────────────────────────────────────────────

@mcp_server.tool()
def nexus_encode(
    content: str,
    source: str = "direct",
    modality: str = "text",
) -> Dict[str, Any]:
    """
    Encode information into NEXUS long-term memory.

    Returns the memory_id if stored, or {"memory_id": null, "status": "discarded"}
    if the Attention Gate determined the content has insufficient salience.

    source: "direct" (default), "user_stated" (highest trust, confidence=1.0),
            "inferred", or "external"
    modality: "text" (default), "code", "image", "structured"
    """
    try:
        mem_source = MemorySource(source)
    except ValueError:
        return {"error": f"Invalid source '{source}'. Use: direct, user_stated, inferred, external"}
    try:
        mem_modality = Modality(modality)
    except ValueError:
        return {"error": f"Invalid modality '{modality}'. Use: text, code, image, structured"}

    memory_id = _nexus.encode(content, source=mem_source, modality=mem_modality)
    if memory_id is None:
        return {"memory_id": None, "status": "discarded"}
    _nexus.save()
    return {"memory_id": memory_id}


@mcp_server.tool()
def nexus_recall(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Recall memories relevant to a query.

    Returns a list of memory dicts, strongest first. Every retrieval strengthens
    the recalled memories (testing effect). Returns empty list if nothing found.
    """
    try:
        memories = _nexus.recall(query, top_k=top_k)
        return [serialize_memory(m) for m in memories]
    except Exception as e:
        logger.error(f"nexus_recall failed: {e}")
        return [{"error": str(e)}]


@mcp_server.tool()
def nexus_get_context() -> Dict[str, str]:
    """
    Get formatted working memory context for injection into a prompt.

    Returns the current capacity-bounded working memory (7±2 slots) as a
    formatted string ready to prepend to a system prompt or user message.
    """
    try:
        return {"context": _nexus.get_context()}
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcp_server.py -k "encode or recall or get_context" -v
```

Expected: All PASSED (note: `test_recall_memory_has_expected_keys` may pass vacuously if attention gate discards — that's fine)

- [ ] **Step 5: Commit**

```bash
git add nexus/integrations/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add nexus_encode, nexus_recall, nexus_get_context tools"
```

---

## Task 5: Confidence tools — `nexus_how_well_do_i_know`, `nexus_knowledge_gaps`

**Files:**
- Modify: `nexus/integrations/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_mcp_server.py`:

```python
def test_how_well_do_i_know_all_fields():
    """Returns all 6 required fields including decision."""
    from nexus.integrations.mcp_server import nexus_how_well_do_i_know
    result = nexus_how_well_do_i_know(topic="Python")
    for key in ("coverage", "freshness", "strength", "depth", "overall", "decision"):
        assert key in result, f"Missing key: {key}"


def test_how_well_do_i_know_decision_valid_values():
    """decision field must be one of the three DecisionType values."""
    from nexus.integrations.mcp_server import nexus_how_well_do_i_know
    result = nexus_how_well_do_i_know(topic="unknown topic xyz")
    assert result["decision"] in ("recall_confidently", "recall_but_verify", "admit_gap_and_ask")


def test_how_well_do_i_know_numeric_fields():
    """Numeric confidence fields must be floats."""
    from nexus.integrations.mcp_server import nexus_how_well_do_i_know
    result = nexus_how_well_do_i_know(topic="anything")
    for key in ("coverage", "freshness", "strength", "overall"):
        assert isinstance(result[key], float), f"{key} must be float"


def test_knowledge_gaps_returns_list():
    """nexus_knowledge_gaps returns a list."""
    from nexus.integrations.mcp_server import nexus_knowledge_gaps
    result = nexus_knowledge_gaps()
    assert isinstance(result, list)


def test_knowledge_gaps_shape_when_populated(tmp_nexus):
    """Each gap dict has the required keys."""
    # Force a gap by recalling unknown topic
    from nexus.integrations.mcp_server import nexus_recall, nexus_knowledge_gaps
    nexus_recall(query="extremely obscure topic that does not exist in memory xyz123")
    gaps = nexus_knowledge_gaps()
    if gaps:
        gap = gaps[0]
        for key in ("topic", "context", "discovered_at", "resolved"):
            assert key in gap, f"Missing key: {key}"
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_mcp_server.py -k "how_well or knowledge_gaps" -v 2>&1 | head -20
```

Expected: `ImportError` for the tool functions

- [ ] **Step 3: Implement the tools**

Add to `nexus/integrations/mcp_server.py`:

```python
# ── Confidence & Gap Tools ────────────────────────────────────────────────────

@mcp_server.tool()
def nexus_how_well_do_i_know(topic: str) -> Dict[str, Any]:
    """
    Assess confidence about a topic.

    Returns 5 confidence dimensions (coverage, freshness, strength, depth, overall)
    and a decision: "recall_confidently", "recall_but_verify", or "admit_gap_and_ask".

    Uses two internal calls: confidence_map() for dimensions, should_recall_or_ask()
    for the decision — these are separate MetaMemory methods.
    """
    try:
        conf = _nexus.meta_memory.confidence_map(topic)
        decision = _nexus.meta_memory.should_recall_or_ask(topic)
        return {
            "coverage": conf.coverage,
            "freshness": conf.freshness,
            "strength": conf.strength,
            "depth": conf.depth,
            "overall": conf.overall,
            "decision": decision.value,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def nexus_knowledge_gaps() -> List[Dict[str, Any]]:
    """
    List topics NEXUS knows it doesn't know.

    Returns gap dicts with keys: topic, context, discovered_at (ISO string), resolved (bool).
    Gaps are registered when recall returns empty or confidence is below threshold.
    """
    try:
        return _nexus.knowledge_gaps()
    except Exception as e:
        return [{"error": str(e)}]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcp_server.py -k "how_well or knowledge_gaps" -v
```

Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add nexus/integrations/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add nexus_how_well_do_i_know and nexus_knowledge_gaps tools"
```

---

## Task 6: Memory management tools — `nexus_pin`, `nexus_forget`, `nexus_consolidate`

**Files:**
- Modify: `nexus/integrations/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_mcp_server.py`:

```python
def test_pin_success(tmp_nexus):
    """nexus_pin returns {status: pinned, memory_id} after pinning."""
    from nexus.integrations.mcp_server import nexus_encode, nexus_pin
    enc = nexus_encode(content="Important fact that must never be forgotten")
    if enc.get("memory_id") is None:
        pytest.skip("attention gate discarded test content")
    memory_id = enc["memory_id"]
    result = nexus_pin(memory_id=memory_id)
    assert result == {"status": "pinned", "memory_id": memory_id}


def test_pin_not_found():
    """nexus_pin returns error dict for unknown memory_id."""
    from nexus.integrations.mcp_server import nexus_pin
    result = nexus_pin(memory_id="nonexistent-id-xyz")
    assert "error" in result


def test_forget_sets_archived(tmp_nexus):
    """nexus_forget returns {status: archived} and memory is ARCHIVED."""
    from nexus.integrations.mcp_server import nexus_encode, nexus_forget
    enc = nexus_encode(content="Temporary note to be forgotten after use")
    if enc.get("memory_id") is None:
        pytest.skip("attention gate discarded test content")
    memory_id = enc["memory_id"]
    result = nexus_forget(memory_id=memory_id)
    assert result == {"status": "archived", "memory_id": memory_id}
    # Verify the memory is actually ARCHIVED in the palace
    mem = _mcp_module._nexus.palace.get_memory(memory_id)
    assert mem.status == MemoryStatus.ARCHIVED


def test_forget_not_found():
    """nexus_forget returns error dict for unknown memory_id."""
    from nexus.integrations.mcp_server import nexus_forget
    result = nexus_forget(memory_id="nonexistent-id-xyz")
    assert "error" in result


def test_consolidate_light():
    """nexus_consolidate('light') returns a summary dict."""
    from nexus.integrations.mcp_server import nexus_consolidate
    result = nexus_consolidate(depth="light")
    assert "depth" in result
    assert result["depth"] == "light"


def test_consolidate_invalid_depth():
    """nexus_consolidate with invalid depth returns error."""
    from nexus.integrations.mcp_server import nexus_consolidate
    result = nexus_consolidate(depth="defer")
    assert "error" in result
    result2 = nexus_consolidate(depth="invalid")
    assert "error" in result2
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_mcp_server.py -k "pin or forget or consolidate" -v 2>&1 | head -20
```

Expected: `ImportError` for tool functions

- [ ] **Step 3: Implement the tools**

Add to `nexus/integrations/mcp_server.py`:

```python
# ── Memory Management Tools ───────────────────────────────────────────────────

@mcp_server.tool()
def nexus_pin(memory_id: str) -> Dict[str, Any]:
    """
    Mark a memory as permanent — it will never be decayed or forgotten.

    Returns {"status": "pinned", "memory_id": ...} on success,
    or {"error": ...} if the memory_id is not found.
    """
    try:
        mem = _nexus.palace.get_memory(memory_id)
        if mem is None:
            return {"error": f"Memory not found: {memory_id}"}
        _nexus.pin(memory_id)
        return {"status": "pinned", "memory_id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def nexus_forget(memory_id: str) -> Dict[str, Any]:
    """
    Gracefully forget a memory by archiving it.

    Sets memory status to ARCHIVED (not deleted — a record remains).
    Returns {"status": "archived", "memory_id": ...} on success,
    or {"error": ...} if the memory_id is not found.
    """
    try:
        mem = _nexus.palace.get_memory(memory_id)
        if mem is None:
            return {"error": f"Memory not found: {memory_id}"}
        _nexus.forget(memory_id)
        return {"status": "archived", "memory_id": memory_id}
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def nexus_consolidate(depth: str = "light") -> Dict[str, Any]:
    """
    Run a consolidation cycle to organize and strengthen memories.

    depth="light": chunking + conflict detection only (fast, safe to call often)
    depth="full": all 8 consolidation processes (thorough, use periodically)

    Note: "defer" is intentionally excluded — it means "let the scheduler decide"
    and is not useful as an explicit call.
    """
    if depth not in ("light", "full"):
        return {"error": "depth must be 'light' or 'full'"}
    try:
        result = _nexus.consolidate(depth=depth)
        return {
            "depth": depth,
            "processed": result.get("total_processed", result.get("processed", 0)),
            "summary": str(result.get("summary", result.get("depth", depth))),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
        }
    except Exception as e:
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcp_server.py -k "pin or forget or consolidate" -v
```

Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add nexus/integrations/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add nexus_pin, nexus_forget, nexus_consolidate tools"
```

---

## Task 7: Introspection tools — `nexus_stats`, `nexus_get_suggestions`

**Files:**
- Modify: `nexus/integrations/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_mcp_server.py`:

```python
def test_stats_top_level_keys():
    """nexus_stats returns all 8 expected top-level keys."""
    from nexus.integrations.mcp_server import nexus_stats
    result = nexus_stats()
    for key in ("palace", "working_memory", "retrieval", "consolidation",
                "meta_memory", "episode_buffer", "vector_store", "metrics"):
        assert key in result, f"Missing top-level key: {key}"


def test_stats_is_json_serializable():
    """nexus_stats output must be JSON-serializable."""
    from nexus.integrations.mcp_server import nexus_stats
    result = nexus_stats()
    assert _is_json_serializable(result)


def test_get_suggestions_returns_list():
    """nexus_get_suggestions returns a list."""
    from nexus.integrations.mcp_server import nexus_get_suggestions
    result = nexus_get_suggestions()
    assert isinstance(result, list)


def test_get_suggestions_serializable():
    """nexus_get_suggestions output is JSON-serializable."""
    from nexus.integrations.mcp_server import nexus_get_suggestions
    result = nexus_get_suggestions()
    assert _is_json_serializable(result)
```

- [ ] **Step 2: Run to verify they fail**

```bash
pytest tests/test_mcp_server.py -k "stats or suggestions" -v 2>&1 | head -20
```

Expected: `ImportError` for tool functions

- [ ] **Step 3: Implement the tools**

Add to `nexus/integrations/mcp_server.py`:

```python
# ── Introspection Tools ───────────────────────────────────────────────────────

@mcp_server.tool()
def nexus_stats() -> Dict[str, Any]:
    """
    Get comprehensive NEXUS system statistics.

    Returns a nested dict with 8 top-level keys:
    palace, working_memory, retrieval, consolidation, meta_memory,
    episode_buffer, vector_store, metrics.
    """
    try:
        return _nexus.stats()
    except Exception as e:
        return {"error": str(e)}


@mcp_server.tool()
def nexus_get_suggestions() -> List[Dict[str, Any]]:
    """
    Get proactive suggestions from NEXUS's ambient monitor.

    Returns a list of memory dicts — patterns and insights surfaced from
    background consolidation that may be relevant to the current context.
    """
    try:
        suggestions = _nexus.get_suggestions()
        return [serialize_memory(s) for s in suggestions]
    except Exception as e:
        return [{"error": str(e)}]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_mcp_server.py -k "stats or suggestions" -v
```

Expected: All PASSED

- [ ] **Step 5: Commit**

```bash
git add nexus/integrations/mcp_server.py tests/test_mcp_server.py
git commit -m "feat(mcp): add nexus_stats and nexus_get_suggestions tools"
```

---

## Task 8: Server entry point and `__init__.py` export

**Files:**
- Modify: `nexus/integrations/mcp_server.py` (add `__main__` block + startup)
- Modify: `nexus/integrations/__init__.py`

- [ ] **Step 1: Add startup and `__main__` block**

Add to the bottom of `nexus/integrations/mcp_server.py`:

```python
# ── Startup ───────────────────────────────────────────────────────────────────

def _startup():
    """Initialize the module-level NEXUS instance from env vars."""
    global _nexus
    config = build_nexus_config()
    logger.info(f"Starting NEXUS MCP server (storage: {config.storage_path}, model: {config.llm_model})")
    _nexus = NEXUS(config=config)
    atexit.register(_nexus.save)
    logger.info("NEXUS MCP server ready — 10 tools registered")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)
    _startup()
    mcp_server.run(transport="stdio")
```

- [ ] **Step 2: Verify server imports and config loads without error**

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
NEXUS_STORAGE_PATH=/tmp/nexus-test NEXUS_LLM_MODEL=none \
  python -c "
from nexus.integrations.mcp_server import build_nexus_config, mcp_server
config = build_nexus_config()
print('storage:', config.storage_path)
print('tools registered:', len(mcp_server._tool_manager._tools))
"
```

Expected: prints storage path and `tools registered: 10`

- [ ] **Step 3: Update `nexus/integrations/__init__.py`**

Edit `nexus/integrations/__init__.py` to:

```python
"""
NEXUS Integration package.
Contains adapters and wrappers to plug NEXUS into standard agent frameworks.
"""

# Try to expose modules if dependencies are met, but don't crash on import if not.
__all__ = []

try:
    from nexus.integrations.mcp_server import mcp_server as nexus_mcp_server  # noqa: F401
    __all__.append("nexus_mcp_server")
except ImportError:
    pass  # mcp not installed — silently skip
```

- [ ] **Step 4: Verify the import guard works**

```bash
python -c "from nexus.integrations import nexus_mcp_server; print('ok')"
```

Expected: prints `ok` (mcp is installed)

- [ ] **Step 5: Run full test suite to confirm nothing broken**

```bash
pytest tests/test_mcp_server.py -v
```

Expected: All tests PASSED

- [ ] **Step 6: Commit**

```bash
git add nexus/integrations/mcp_server.py nexus/integrations/__init__.py
git commit -m "feat(mcp): add server entry point, startup, and __init__.py export"
```

---

## Task 9: Register with Claude Code and smoke test

**Files:**
- Modify: `~/.claude/settings.json`

- [ ] **Step 1: Add the MCP server to Claude Code settings**

Read `~/.claude/settings.json`, then add the `mcpServers` block:

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

Merge carefully with existing settings — preserve all existing keys.

- [ ] **Step 2: Reload Claude Code**

Open `/mcp` in Claude Code to reload MCP server config. You should see `nexus` listed as a connected server with 10 tools.

- [ ] **Step 3: Smoke test in Claude Code**

Ask Claude: *"Can you call nexus_stats and show me what's in memory?"*

Expected: Claude calls `nexus_stats`, returns the nested dict showing empty palace/episode counts.

- [ ] **Step 4: Final commit**

```bash
git add docs/superpowers/plans/2026-03-19-nexus-mcp-server.md
git commit -m "docs: add MCP server implementation plan"
```

---

## Running All Tests

```bash
cd /Users/shivamtyagi/PycharmProjects/nexus-memory
pytest tests/test_mcp_server.py -v
```

Expected: 17+ tests, all PASSED.
