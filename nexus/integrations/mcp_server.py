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
    from mcp.server.fastmcp import FastMCP
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
    api_key = os.environ.get("NEXUS_LLM_API_KEY", "")

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


# Module-level NEXUS instance — initialized at startup, shared across tool calls.
# Tests replace this: `import nexus.integrations.mcp_server as s; s._nexus = test_instance`
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
