# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2026-04-05

### Fixed
- **MCP Server**: Fixed `nexus_consolidate` response key mismatch — previously always returned `processed: 0` and a bare depth string instead of actual process count and status. Now correctly reports successful process count, process names, and handles the deferred case explicitly

### Changed
- **README**: Updated benchmark tables with validated v1.0.0 results across 4 models (gpt-4o-mini, Mistral 7B, CodeLlama 7B, Llama 3.2 3B). Added local model comparison section and "What's New in v1.0.0" section

## [1.0.0] - 2026-04-05

### Fixed
- **Consolidation Scheduling**: Added high-salience trigger — any unconsolidated episode with composite salience ≥ 0.55 now immediately triggers FULL consolidation, instead of waiting for 50+ episodes to accumulate
- **Singleton Episode Leak**: Episodes that don't cluster with others during chunking are now properly marked as consolidated, preventing them from blocking the buffer indefinitely
- **Contradiction Detection**: Updated LLM prompt to explicitly distinguish between genuine contradictions and mere similarity/redundancy — prevents Mistral from incorrectly superseding agreeing memories
- **Reflection on Single Episodes**: Highly salient single episodes (≥ 0.7) can now generate reflections on their own, instead of requiring groups of 3+
- **Reflection Buffer Cleanup**: Episodes processed by reflection are now marked consolidated so they don't re-trigger consolidation cycles

### Changed
- **Heuristic Salience Scoring**: Overhauled `score_fast` to produce differentiated scores based on content type — personal facts, knowledge updates, instructions, and code now score significantly higher than generic content. User-stated facts score 0.65–0.73 (full encoding) vs trivial messages at 0.24 (summary)
- **Chunking Minimum Removed**: `_process_chunking` no longer requires 3+ episodes to run — works with any number of episodes

### Benchmarked
- Validated across 4 models (gpt-4o-mini, Mistral 7B, CodeLlama 7B, Llama 3.2 3B) on LoCoMo dataset with consolidation enabled
- Unconsolidated episode leak eliminated (11 → 0 across all models)
- Best local model: CodeLlama 7B (F1=0.317, Exact Match=0.200)

## [0.1.14] - 2026-03-27

### Fixed
- **Claude Code Hooks**: Fixed JSON schema for hooks in `install_nexus_mcp.sh` — now uses the correct two-level nesting with matcher groups so `nexus_recall`, `nexus_encode`, and `nexus_get_context` hooks trigger properly
- **Install Script**: Added required `"matcher"` field to all hook definitions

### Changed
- **README**: Promoted MCP-based Claude Code setup to a top-level "Quick Start" section for better discoverability

## [0.1.13] - 2026-03-19

### Added
- **MCP Server** (`nexus/integrations/mcp_server.py`): Exposes NEXUS as a Claude Code MCP server via stdio transport
  - 10 tools: `nexus_encode`, `nexus_recall`, `nexus_get_context`, `nexus_how_well_do_i_know`, `nexus_knowledge_gaps`, `nexus_pin`, `nexus_forget`, `nexus_consolidate`, `nexus_stats`, `nexus_get_suggestions`
  - LLM provider auto-detected from model name prefix (`claude-*` → Anthropic, `gpt-*` → OpenAI, `gemini*` → Gemini, else Ollama)
  - Configured via environment variables: `NEXUS_STORAGE_PATH`, `NEXUS_LLM_MODEL`, `NEXUS_LLM_API_KEY`
- **Install script** (`install_nexus_mcp.sh`): One-command setup that installs `nexus-memory[mcp]` in a dedicated venv, sets up git hooks, adds a `SessionStart` hook for Claude Code, patches `~/.claude.json` safely, and validates Ollama models
- **31 MCP server tests** covering all tools, routing logic, error handling, and edge cases
- LongMemEval benchmark integration and updated benchmark results in README

### Changed
- **LangChain Integration**: `NexusLangChainHistory.messages` now injects both **System 2** (abstract knowledge from the Semantic Palace) and **System 1** (raw episodic events from the Episode Buffer) into the LLM context, achieving true Dual-Process memory recall.
- **LLM Interface**: `generate_json()` now accepts and forwards a `max_tokens` parameter (default `4096`) for finer control over JSON generation responses.

### Fixed
- **Episode Buffer**: `search_semantic` now falls back to SQLite for consolidated episodes by using `self.get()` instead of `self._episodes.get()`, which previously missed any episode that had been consolidated out of memory. Also over-fetches candidates and truncates to correctly respect `top_k`.

## [0.1.1] - 2025-03-03

### Added
- **FAISS Backend**: Optional FAISS accelerated vector search — auto-detected, falls back to NumPy
- New `backend` parameter for `VectorStore`: `"auto"` (default), `"faiss"`, `"numpy"`
- Vector benchmark script (`benchmarks/vector_benchmark.py`)
- §5.4 Vector Backend Performance Analysis in research paper
- 5 new backend tests (159 total)

### Changed
- `pip install nexus-memory[faiss]` now installs FAISS support

## [0.1.0] - 2025-03-03

### Added
- **Core Architecture**: NEXUS orchestrator with encode/recall/consolidate lifecycle
- **Semantic Palace**: Graph-based memory clustering with typed edges, room auto-creation, and multi-hop associative retrieval
- **Working Memory**: Capacity-bounded (7±2 slots) priority queue with deduplication, active/peripheral context split
- **Attention Gate**: Dual scoring (heuristic + LLM) with 5-dimension salience filter
- **Episode Buffer**: SQLite-backed temporal log with lazy-loading for unconsolidated episodes
- **Consolidation Engine**: Async background processes — chunking, spaced repetition, skill extraction, conflict resolution, reflection generation
- **Retrieval Engine**: Multi-factor scoring (recency × relevance × strength × salience) with spreading activation
- **Meta-Memory**: Confidence mapping, knowledge gap detection, failed-retrieval tracking
- **Vector Store**: Sentence-transformer embeddings with add/search/remove, persistence (`.npy` + `.json`)
- **LLM Interface**: Multi-provider support — Ollama (default), OpenAI, Anthropic, Google Gemini — with retry + exponential backoff
- **Metrics & Observability**: Thread-safe counters, gauges, histograms with JSON snapshot and Prometheus text export
- **Test Suite**: 154 tests across 13 files covering all 12 modules
- **Production Hardening**: Thread safety (locks on all shared state), atomic saves, crash recovery (atexit hooks), input validation, bounded data structures, idempotent close

### Security
- Prompt injection guardrails (`<content>` tag wrapping)
- API key environment variable fallbacks
- Content length limits
- Input validation on all config parameters
