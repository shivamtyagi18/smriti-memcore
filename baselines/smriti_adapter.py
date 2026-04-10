"""
SMRITI adapter for the benchmark harness.
Wraps the SMRITI core into the BaseMemorySystem interface.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, List, Optional

from baselines.base import BaseMemorySystem, MemoryResponse
from smriti_memcore.core import SMRITI
from smriti_memcore.models import MemorySource, SmritiConfig
from smriti_memcore.llm_interface import LLMInterface


class SmritiAdapter(BaseMemorySystem):
    """Wraps SMRITI v2 into the BaseMemorySystem benchmark interface."""

    def __init__(self, llm: LLMInterface, config: Optional[SmritiConfig] = None):
        super().__init__("SMRITI_v2", llm)
        self.config = config or SmritiConfig()
        self.smriti = SMRITI(self.config)
        # Override SMRITI's internal LLM with the benchmark's LLM
        # so queries and consolidation use the same model (e.g. gpt-4o-mini)
        self.smriti.llm = llm
        self.smriti.attention_gate.llm = llm
        self.smriti.consolidation_engine.llm = llm

    def ingest(self, message: str, role: str = "user", metadata: Optional[Dict] = None):
        source = MemorySource.USER_STATED if role == "user" else MemorySource.DIRECT
        # Use fast heuristic scoring for benchmark speed
        # Temporarily disable auto-consolidation during batch ingest
        original_trigger = self.smriti.config.episode_buffer_trigger
        self.smriti.config.episode_buffer_trigger = 99999  # Prevent mid-ingest consolidation
        self.smriti.encode(message, source=source, use_llm=False)
        self.smriti.config.episode_buffer_trigger = original_trigger
        self._ingest_count += 1

    def query(self, question: str, context: str = "") -> MemoryResponse:
        def _do_query(q, ctx):
            # Recall memories
            memories = self.smriti.recall(q, context=ctx, top_k=5)

            # Build prompt with retrieved memories and confidence
            confidence = self.smriti.how_well_do_i_know(q)
            memory_texts = [m.content for m in memories]
            memory_str = "\n".join(f"- {t}" for t in memory_texts) if memory_texts else "No relevant memories."

            confidence_note = ""
            if confidence.overall < 0.3:
                confidence_note = "\nNote: I have limited knowledge on this topic."

            prompt = f"""Based on the following memories, answer the question.{confidence_note}

Memories:
{memory_str}

Question: {question}
Answer:"""

            response = self.smriti.llm.generate(prompt)
            return MemoryResponse(
                answer=response.text.strip(),
                memories_used=memory_texts,
                confidence=confidence.overall,
                tokens_used=response.tokens_used,
            )

        return self._timed_query(_do_query, question, context)

    def reset(self):
        import shutil
        self._ingest_count = 0
        self._query_count = 0
        self._total_latency_ms = 0.0
        # Re-initialize SMRITI
        if os.path.exists(self.config.storage_path):
            shutil.rmtree(self.config.storage_path, ignore_errors=True)
        self.smriti = SMRITI(self.config)
        # Re-inject benchmark LLM
        self.smriti.llm = self.llm
        self.smriti.attention_gate.llm = self.llm
        self.smriti.consolidation_engine.llm = self.llm

    def run_consolidation(self):
        """Run FULL consolidation on the SMRITI memory system."""
        from smriti_memcore.models import ConsolidationDepth
        self.smriti.consolidation_engine.consolidate(depth=ConsolidationDepth.FULL)

    def get_stats(self) -> Dict:
        base = super().get_stats()
        base.update(self.smriti.stats())
        return base
