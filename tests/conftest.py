"""
Shared test fixtures for the SMRITI test suite.
Provides mock LLM, temp-directory stores, and pre-built objects.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smriti.models import (
    SmritiConfig, Memory, Episode, SalienceScore, MemorySource, Modality,
)
from smriti.vector_store import VectorStore
from smriti.episode_buffer import EpisodeBuffer
from smriti.palace import SemanticPalace
from smriti.working_memory import WorkingMemory
from smriti.llm_interface import LLMInterface, LLMResponse
from smriti.metrics import SmritiMetrics


# ── Temp Directories ────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temp directory."""
    return str(tmp_path)


# ── Vector Store ────────────────────────────────────────

@pytest.fixture
def vector_store(tmp_dir):
    """VectorStore backed by a temp directory."""
    return VectorStore(
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        storage_path=os.path.join(tmp_dir, "vectors"),
    )


# ── Episode Buffer ──────────────────────────────────────

@pytest.fixture
def episode_buffer(tmp_dir, vector_store):
    """EpisodeBuffer backed by a temp directory."""
    eb = EpisodeBuffer(
        storage_path=os.path.join(tmp_dir, "episodes"),
        vector_store=vector_store,
    )
    yield eb
    eb.close()


# ── Mock LLM ────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """LLMInterface with mocked generate methods."""
    llm = LLMInterface(default_model="test-model")

    # Mock generate to return a simple response
    def mock_generate(prompt, **kwargs):
        return LLMResponse(text="mock response", model="test-model")

    def mock_generate_json(prompt, **kwargs):
        return {
            "surprise": 0.5, "relevance": 0.6,
            "emotional": 0.3, "novelty": 0.4, "utility": 0.7,
        }

    def mock_score_salience(content, context=""):
        return {
            "surprise": 0.5, "relevance": 0.6,
            "emotional": 0.3, "novelty": 0.4, "utility": 0.7,
        }

    def mock_chunk_memories(contents):
        return {"summary": " | ".join(contents)}

    def mock_generate_reflection(contents, level=1):
        return f"Reflection L{level}: pattern across {len(contents)} items"

    def mock_detect_contradiction(a, b):
        return {"contradicts": False, "confidence": 0.9, "explanation": "no conflict"}

    llm.generate = mock_generate
    llm.generate_json = mock_generate_json
    llm.score_salience = mock_score_salience
    llm.chunk_memories = mock_chunk_memories
    llm.generate_reflection = mock_generate_reflection
    llm.detect_contradiction = mock_detect_contradiction

    return llm


# ── Palace ──────────────────────────────────────────────

@pytest.fixture
def palace(tmp_dir, vector_store):
    """SemanticPalace backed by a temp directory."""
    return SemanticPalace(
        vector_store=vector_store,
        storage_path=os.path.join(tmp_dir, "palace"),
    )


# ── Working Memory ──────────────────────────────────────

@pytest.fixture
def working_memory():
    """WorkingMemory with default settings."""
    return WorkingMemory(max_slots=7, active_chunks=4)


# ── Metrics ─────────────────────────────────────────────

@pytest.fixture
def metrics():
    """Fresh SmritiMetrics instance."""
    return SmritiMetrics()


# ── Helper Factories ────────────────────────────────────

@pytest.fixture
def make_memory():
    """Factory for creating Memory objects with sensible defaults."""
    def _make(content="test memory", **kwargs):
        defaults = {
            "salience": SalienceScore(
                surprise=0.5, relevance=0.6,
                emotional=0.3, novelty=0.4, utility=0.7,
            ),
            "source": MemorySource.DIRECT,
        }
        defaults.update(kwargs)
        return Memory(content=content, **defaults)
    return _make


@pytest.fixture
def make_episode():
    """Factory for creating Episode objects with sensible defaults."""
    def _make(content="test episode", **kwargs):
        defaults = {
            "salience": SalienceScore(
                surprise=0.5, relevance=0.6,
                emotional=0.3, novelty=0.4, utility=0.7,
            ),
        }
        defaults.update(kwargs)
        return Episode(content=content, **defaults)
    return _make


# ── Config ──────────────────────────────────────────────

@pytest.fixture
def config(tmp_dir):
    """SmritiConfig with temp storage path."""
    return SmritiConfig(storage_path=os.path.join(tmp_dir, "smriti_db"))
