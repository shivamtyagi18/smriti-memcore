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
