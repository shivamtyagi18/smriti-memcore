# NEXUS Memory

**A neuro-inspired long-term memory architecture for AI agents.**

NEXUS combines a capacity-bounded Working Memory, a graph-based Semantic Palace, and asynchronous background consolidation to give LLM agents persistent, scalable memory — without blocking real-time interactions.

> 📄 **Paper:** *NEXUS: A Scalable, Neuro-Inspired Architecture for Long-Term Event Memory in LLM Agents* — Shivam Tyagi, 2025 — [DOI: 10.13140/RG.2.2.25477.82407](https://doi.org/10.13140/RG.2.2.25477.82407)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```text
                           ┌─────────────────────────────────┐
                           │    Asynchronous Consolidation   │
                           │      (8 Background Processes)   │
                           │  • Chunking      • Cross-Ref.   │
                           │  • Conflict Res. • Skill Ext.   │
                           │  • Forgetting    • Spaced Rep.  │
                           │  • Reflection    • Defragment.  │
                           └────────────────┬────────────────┘
                                            │ background
  ┌──────────┐   ┌──────────┐   ┌───────────▼─────────┐   ┌──────────┐
  │  Input   │──▶│ Attention │──▶│   Episode Buffer    │──▶│ Semantic │
  │  Text    │   │   Gate    │   │  (append-only log)  │   │  Palace  │
  └──────────┘   │ (salience │   └─────────────────────┘   │  Graph   │
                 │  filter)  │                              │ G=(V,E)  │
                 └──────────┘                              └────┬─────┘
                                                                │
  ┌──────────┐   ┌──────────┐   ┌───────────────────┐           │
  │  Query   │──▶│ Retrieval│──▶│  Working Memory   │◀──────────┘
  │          │   │  Engine  │   │   (7 ± 2 slots)   │
  └──────────┘   │ Q(v) =   │   └───────────────────┘
                 │ β₁cos +  │
                 │ β₂decay+ │   ┌───────────────────┐
                 │ β₃freq + │──▶│    Meta-Memory    │
                 │ β₄sal    │   │ (confidence map)  │
                 └──────────┘   └───────────────────┘
```

**Core idea:** Inspired by human Dual-Process Theory (Daniel Kahneman's *Thinking, Fast and Slow*), NEXUS decouples memory operations into two pathways:
- **System 1 (Fast & Heuristic):** Real-time ingestion. Routes interactions to the short-term Episode Buffer in milliseconds without blocking the agent.
- **System 2 (Slow & Analytical):** Background consolidation. Uses LLM reasoning to chunk, organize, and abstract semantic knowledge asynchronously while the agent is idle.
---

## Installation

```bash
pip install nexus-memory
```

With optional **FAISS** accelerated vector search:

```bash
pip install nexus-memory[faiss]
```

Or install from source:

```bash
git clone https://github.com/shivamtyagi18/nexus-memory.git
cd nexus-memory
pip install -e .
```

### Prerequisites

NEXUS uses an LLM for reasoning tasks (consolidation, reflection, skill extraction). By default it connects to a local [Ollama](https://ollama.ai) instance:

```bash
ollama pull mistral
```

Alternatively, you can use **OpenAI**, **Anthropic**, or **Google Gemini** — see [Using Cloud LLM Providers](#using-cloud-llm-providers) below.

---

## Using Cloud LLM Providers

NEXUS is **provider-agnostic**. Just change the `llm_model` and pass your API key:

```python
from nexus import NEXUS, NexusConfig

# ── OpenAI ──────────────────────────────────────────────
config = NexusConfig(
    llm_model="gpt-4o",
    openai_api_key="sk-...",
)

# ── Anthropic ───────────────────────────────────────────
config = NexusConfig(
    llm_model="claude-3-5-sonnet-20241022",
    anthropic_api_key="sk-ant-...",
)

# ── Google Gemini ───────────────────────────────────────
config = NexusConfig(
    llm_model="gemini-1.5-flash",
    gemini_api_key="AIza...",
)

# ── Local Ollama (default) ──────────────────────────────
config = NexusConfig(
    llm_model="mistral",  # or llama3, codellama, phi3, etc.
)

memory = NEXUS(config=config)
```

Routing is automatic based on the model name prefix: `gpt-*` → OpenAI, `claude*` → Anthropic, `gemini*` → Gemini, everything else → Ollama.

---

## Quick Start

```python
from nexus import NEXUS, NexusConfig

# Initialize
config = NexusConfig(
    storage_path="./my_agent_memory",
    llm_model="mistral",
)
memory = NEXUS(config=config)

# Encode information
memory.encode("User prefers Python for backend development.")
memory.encode("User is allergic to shellfish.", context="medical")

# Recall by natural-language query
results = memory.recall("What language does the user prefer?")
for mem in results:
    print(f"  [{mem.strength:.2f}] {mem.content}")

# Check what you know (and don't know)
confidence = memory.how_well_do_i_know("programming languages")
print(f"Confidence: {confidence.overall:.0%}")

# Run background consolidation
memory.consolidate()

# Persist to disk
memory.save()
```

### Framework Integrations
NEXUS can be used natively inside standard agent frameworks. 

#### LangChain
Use `NexusLangChainMemory` to replace `ConversationBufferMemory`. This gives your agent the cost-savings of a capacity-bounded Working Memory while asynchronously archiving the conversation into the Semantic Palace.

```python
from langchain.chains import ConversationChain
from nexus.integrations.langchain_memory import NexusLangChainMemory
from nexus import NEXUS

# 1. Initialize NEXUS
nexus_engine = NEXUS(storage_path="./langchain_nexus_db")

# 2. Wrap it for LangChain
nexus_memory = NexusLangChainMemory(nexus_client=nexus_engine, top_k=3)

# 3. Plug it into standard chains
conversation = ConversationChain(
    llm=my_llm,
    memory=nexus_memory,
)

conversation.predict(input="I prefer using PyTorch.")
```

See [`examples/langchain_agent.py`](examples/langchain_agent.py) or [`examples/quickstart.py`](examples/quickstart.py) for complete working code.

#### Claude Code (MCP Server)

Give Claude Code persistent long-term memory across every session using the built-in MCP server.

**Prerequisites:**

- Python 3.9+
- [Claude Code](https://claude.ai/code) installed
- An LLM for consolidation (pick one):
  - **Local (free):** [Ollama](https://ollama.ai) + `ollama pull mistral`
  - **Cloud:** Anthropic, OpenAI, or Google API key

**Setup (2 steps):**

**Step 1 — Run the install script:**

```bash
bash <(curl -s https://raw.githubusercontent.com/shivamtyagi18/nexus-memory/main/install_nexus_mcp.sh)
```

The script will:
- Create a dedicated venv at `~/.nexus/venv`
- Install `nexus-memory` into it
- Prompt for your LLM choice and API key
- Prompt for memory storage path (default: `~/.nexus/global`)
- Register the MCP server in `~/.claude.json`

**Step 2 — Restart Claude Code**

**Verify:** Run `/mcp` inside Claude Code — `nexus` should appear as connected.

**Available tools (10):**

| Tool | Description |
|---|---|
| `nexus_encode` | Store information in long-term memory |
| `nexus_recall` | Retrieve memories by natural-language query |
| `nexus_get_context` | Inject working memory into the current prompt |
| `nexus_how_well_do_i_know` | Confidence check on a topic |
| `nexus_knowledge_gaps` | List topics NEXUS knows it doesn't know |
| `nexus_pin` | Mark a memory as permanent (never decayed) |
| `nexus_forget` | Archive a memory |
| `nexus_consolidate` | Run a consolidation cycle |
| `nexus_stats` | System-wide statistics |
| `nexus_get_suggestions` | Proactive insights from background consolidation |

**LLM options** — set during install or via environment variables:

| Model | Provider | Requires |
|---|---|---|
| `mistral` (default) | Local Ollama | `ollama pull mistral` |
| `claude-*` | Anthropic | `NEXUS_LLM_API_KEY` |
| `gpt-*` | OpenAI | `NEXUS_LLM_API_KEY` |
| `gemini*` | Google | `NEXUS_LLM_API_KEY` |

---

## Key API

| Method | Description |
|---|---|
| `encode(content, context, source)` | Ingest new information through the Attention Gate |
| `recall(query, top_k)` | Retrieve relevant memories via graph traversal |
| `how_well_do_i_know(topic)` | Meta-memory confidence check |
| `consolidate(depth)` | Run background consolidation (`"full"`, `"light"`, `"defer"`) |
| `save()` | Persist all state to disk |
| `pin(memory_id)` | Mark a memory as permanent |
| `forget(memory_id)` | Gracefully forget a memory (leaves a tombstone) |
| `stats()` | System-wide statistics |

---

## Configuration

All parameters are optional and have sensible defaults:

```python
from nexus import NexusConfig

config = NexusConfig(
    # Working Memory
    working_memory_slots=7,          # Miller's Law: 7 ± 2

    # Retrieval scoring weights
    recency_weight=0.2,
    relevance_weight=0.4,
    strength_weight=0.2,
    salience_weight=0.2,

    # Forgetting
    decay_rate=0.99,                 # per-day temporal decay
    strength_hard_threshold=0.05,    # below this → forget

    # Palace graph
    room_merge_threshold=0.85,       # similarity to auto-merge rooms

    # LLM provider (pick one)
    llm_model="mistral",                     # Ollama (default)
    # llm_model="gpt-4o",                    # OpenAI
    # llm_model="claude-3-5-sonnet-20241022",# Anthropic
    # llm_model="gemini-1.5-flash",          # Google
    ollama_base_url="http://localhost:11434",

    # Storage
    storage_path="./nexus_data",
)
```

---

## Benchmarks

NEXUS was benchmarked against four baseline architectures on the [LoCoMo](https://github.com/snap-research/locomo) long-sequence conversational dataset (419 dialog turns):

| System | F1 Score | Latency (p95) | Ingestion Time |
|---|---|---|---|
| FullContext | 0.040 | 9.07s | 0.0s |
| MemGPT-style | 0.025 | 10.16s | ~15 min |
| Mem0-style | 0.024 | 8.39s | ~45 min |
| NaiveRAG | 0.012 | 8.07s | 9.4s |
| **NEXUS v2** | 0.010 | **7.62s** | **32.1s** |

### LongMemEval (Long-Term Interactive Memory)

NEXUS integrates an evaluation harness for the [LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark to rigorously test an LLM assistant's ability to maintain context over 50+ chat sessions.

In our exact-match recall tests over isolated needle-in-a-haystack multi-session transcripts, NEXUS achieves highly competitive factual precision with a fraction of the computational latency by utilizing its Dual-Process semantic routing:

| System Configuration | Exact Match Accuracy | Average Query Latency |
|---|---|---|
| **Baseline (Full Context)** | 100.0% | 11.98s |
| **NEXUS Dual-Process** | **80.0%** | **0.98s** |

*NEXUS restricts the LLM context envelope to only the most relevant episodic graph nodes, resulting in a **>12× latency reduction** compared to standard context-stuffing.*

To run the full 500-question benchmark with GPT-4o-mini as the evaluator:

```bash
# 1. Download the cleaned JSON datasets to data/longmemeval/
# 2. Run the baseline LLM (Standard ConversationBufferMemory):
python benchmarks/longmem_eval.py --baseline

# 3. Run the optimized NEXUS Dual-Process Evaluator:
python benchmarks/longmem_eval.py
```

### Vector Search Backend

NEXUS supports two vector search backends. FAISS is auto-detected when installed:

| Backend | 1K vectors | 10K vectors | 100K vectors | Memory (100K) |
|---|---|---|---|---|
| NumPy | 22 µs | 179 µs | 2.75 ms | 146.5 MB |
| **FAISS** | 28 µs | 200 µs | **2.24 ms** | **979 B** |

At scale, FAISS is **1.2× faster** with **150,000× less memory**.

To reproduce:

```bash
pip install -e ".[benchmarks]"
python benchmarks/run_benchmark.py --systems nexus naiverag fullcontext --dataset locomo
python benchmarks/vector_benchmark.py   # NumPy vs FAISS comparison
```

---

## Project Structure

```
nexus-memory/
├── nexus/                 # Core library
│   ├── __init__.py
│   ├── core.py            # NEXUS orchestrator
│   ├── models.py          # Data models & NexusConfig
│   ├── palace.py          # Semantic Palace graph
│   ├── episode_buffer.py  # Append-only temporal log
│   ├── working_memory.py  # Capacity-bounded priority queue
│   ├── attention_gate.py  # Salience filter
│   ├── retrieval.py       # Multi-factor retrieval engine
│   ├── consolidation.py   # Async background processes
│   ├── meta_memory.py     # Confidence mapping
│   ├── vector_store.py    # Vector persistence
│   ├── llm_interface.py   # Multi-provider LLM connector (Ollama/OpenAI/Anthropic/Gemini)
│   ├── metrics.py         # Observability: counters, gauges, histograms, Prometheus export
│   └── integrations/      # Framework adapters
│       ├── langchain_memory.py  # LangChain BaseMemory component
│       └── mcp_server.py        # Claude Code MCP server (10 tools)
├── install_nexus_mcp.sh   # One-command Claude Code setup
├── tests/                 # 190 tests across 14 files
├── baselines/             # Baseline implementations for comparison
├── benchmarks/            # Benchmark harness & scripts
├── examples/              # Usage examples
├── paper/                 # IEEE research paper (LaTeX + Markdown)
│   └── figures/           # Benchmark charts and UI diagrams
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Citation

If you use NEXUS in your research, please cite:

```bibtex
@article{tyagi2025nexus,
  title={NEXUS: A Scalable, Neuro-Inspired Architecture for Long-Term Event Memory in LLM Agents},
  author={Tyagi, Shivam},
  year={2025},
  doi={10.13140/RG.2.2.25477.82407}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
