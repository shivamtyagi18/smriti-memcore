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


