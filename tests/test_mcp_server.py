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
