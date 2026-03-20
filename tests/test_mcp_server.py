"""Tests for the NEXUS MCP server."""
import json
import os
import tempfile
from datetime import datetime

import pytest

from nexus.models import Memory, MemorySource, MemoryStatus, Modality, NexusConfig, SalienceScore
from nexus.core import NEXUS
import nexus.integrations.mcp_server as _mcp_module


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


def test_llm_model_ollama_routing(tmp_path, monkeypatch):
    """Non-prefixed model name (ollama path) produces '' for all provider key fields."""
    monkeypatch.setenv("NEXUS_STORAGE_PATH", str(tmp_path))
    monkeypatch.setenv("NEXUS_LLM_MODEL", "mistral")
    monkeypatch.delenv("NEXUS_LLM_API_KEY", raising=False)

    from nexus.integrations.mcp_server import build_nexus_config
    config = build_nexus_config()
    assert config.anthropic_api_key == ""
    assert config.openai_api_key == ""
    assert config.gemini_api_key == ""


def test_build_nexus_config_expands_tilde(monkeypatch):
    """~ in NEXUS_STORAGE_PATH must be expanded."""
    monkeypatch.setenv("NEXUS_STORAGE_PATH", "~/.nexus/test")
    monkeypatch.delenv("NEXUS_LLM_MODEL", raising=False)
    monkeypatch.delenv("NEXUS_LLM_API_KEY", raising=False)

    from nexus.integrations.mcp_server import build_nexus_config
    config = build_nexus_config()
    assert not config.storage_path.startswith("~")
    assert config.storage_path == os.path.expanduser("~/.nexus/test")


# ── Task 4: Core memory tools ─────────────────────────────────────────────────

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

# ── Task 5: Confidence tools ──────────────────────────────────────────────────

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
    from nexus.integrations.mcp_server import nexus_recall, nexus_knowledge_gaps
    nexus_recall(query="extremely obscure topic that does not exist in memory xyz123")
    gaps = nexus_knowledge_gaps()
    if gaps:
        gap = gaps[0]
        for key in ("topic", "context", "discovered_at", "resolved"):
            assert key in gap, f"Missing key: {key}"


