# Hybrid Search: SQLite FTS5 + Reciprocal Rank Fusion

**Date:** 2026-05-01
**Status:** Approved for implementation
**Scope:** `smriti_memcore` — retrieval layer only

---

## Problem

Smriti's retrieval scores memories on semantic similarity (cosine) plus recency, strength, and salience. This works well for conceptual queries but fails on exact-term queries — ticket IDs, version numbers, function names, proper nouns — where the embedding similarity to the query may be low even though the memory contains the exact term.

The fix is hybrid search: run keyword (BM25) and vector searches in parallel, merge their candidate pools before scoring, then apply the existing multi-factor scoring pipeline unchanged.

---

## Approach: Parallel Pools + RRF

Vector search and FTS5 each independently produce a ranked candidate list. Reciprocal Rank Fusion (RRF) merges them into a single ordered pool. The existing `_score_memory()` pipeline runs on this merged pool.

RRF formula: `score(d) = Σ 1 / (k + rank(d))` across all lists, `k = 60` (standard).

RRF is parameter-free — no α/β weights to tune. A memory appearing only in the FTS list still contributes `1/(60+1)` to its RRF score and enters the merged pool, guaranteeing exact-match memories reach the scoring stage.

---

## Components

### New: `smriti_memcore/fts_index.py` — `FTSIndex`

Self-contained SQLite FTS5 wrapper. Single responsibility: keyword candidate generation.

**Public interface:**

```python
class FTSIndex:
    def __init__(self, storage_path: str): ...
    def add(self, memory_id: str, content: str) -> None: ...
    def remove(self, memory_id: str) -> None: ...
    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]: ...
    def rebuild(self, memories: List[Memory]) -> None: ...
    def close(self) -> None: ...
```

**Storage:** `{storage_path}/fts.db` — SQLite file, same directory as `palace.json`.

**Schema:**
```sql
CREATE VIRTUAL TABLE IF NOT EXISTS memories USING fts5(
    memory_id UNINDEXED,
    content,
    tokenize = 'porter ascii'
);
```

`porter ascii` provides Porter stemming — "consolidating" matches "consolidate" — while preserving exact tokens like "YEP-293".

**Startup:** On `__init__`, if `fts.db` is missing or row count differs from active memory count, call `rebuild()`. Rebuild is a DELETE-all + bulk INSERT, fast at current scale (<10ms for 74 memories).

**WAL mode:** `PRAGMA journal_mode=WAL` on connection open — reads do not block writes.

**`fts.db` is expendable:** it is a derived index, not source of truth. Deleting it triggers a clean rebuild on next startup.

---

### Modified: `smriti_memcore/retrieval.py` — `RetrievalEngine`

`FTSIndex` injected at construction via `__init__`. `retrieve()` gains a pre-scoring merge step between existing steps 1 and 2.

**New method: `_rrf_merge`**

```python
def _rrf_merge(
    self,
    vector_candidates: List[Memory],
    fts_results: List[Tuple[str, float]],
    pool_size: int,
    k: int = 60,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for rank, memory in enumerate(vector_candidates):
        scores[memory.id] += 1.0 / (k + rank + 1)
    for rank, (memory_id, _) in enumerate(fts_results):
        scores[memory_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda id: scores[id], reverse=True)[:pool_size]
```

**Updated `retrieve()` flow:**

```
Step 1a  palace.search(query, top_k=top_k*3)       → vector_candidates: List[Memory]
Step 1b  fts_index.search(query, top_k=top_k*3)    → fts_results: List[(id, score)]
Step 1c  _rrf_merge(vector_candidates, fts_results) → merged_ids: List[str]
Step 1d  fetch Memory objects for FTS-only ids      → combined_pool: List[Memory]
Steps 2–5 unchanged: _score_memory(), testing effect, effort bonus, WorkingMemory
```

Candidate pool size is `top_k * 3` for each source (up from `top_k * 2` for vector alone), giving the merged pool enough depth.

---

### Modified: `smriti_memcore/core.py` — `SMRITI`

- `__init__()`: instantiate `FTSIndex(config.storage_path)`, pass to `RetrievalEngine`
- `encode()`: call `fts_index.add(memory.id, memory.content)` after palace write
- `forget()`: call `fts_index.remove(memory_id)` after palace archive
- `close()` / `_atexit_save()`: call `fts_index.close()`

---

## Data Flow

### Encode

```
smriti.encode(text)
  → AttentionGate check
  → EpisodeBuffer.add()
  → palace.add_memory(memory)          [existing]
  → vector_store.add(memory.id, text)  [existing]
  → fts_index.add(memory.id, text)     [new, ~0.1ms]
```

### Recall

```
smriti.recall(query)
  → RetrievalEngine.retrieve(query)
      1a. vector_candidates = palace.search(query, top_k*3)
      1b. fts_results       = fts_index.search(query, top_k*3)
      1c. merged_ids        = _rrf_merge(vector_candidates, fts_results)
      1d. combined_pool     = fetch Memory objects for merged_ids
      2.  score each: composite = cosine + decay + strength + salience  [unchanged]
      3.  sort, select top_k
      4.  testing effect reinforcement                                   [unchanged]
      5.  effort bonus                                                   [unchanged]
      6.  admit to WorkingMemory                                         [unchanged]
```

### Forget

```
smriti.forget(memory_id)
  → palace.archive(memory_id)   [existing]
  → fts_index.remove(memory_id) [new]
```

---

## Error Handling

FTS failures are non-fatal. If `fts_index.search()` raises any exception, `retrieve()` logs a warning and falls back to vector-only candidates — the existing pipeline runs unchanged.

```python
try:
    fts_results = self.fts_index.search(query, top_k=pool_size)
except Exception:
    logger.warning("FTS search failed — falling back to vector-only retrieval")
    fts_results = []
```

Same pattern for `fts_index.add()` in `encode()`: log and continue, never propagate to caller. A corrupt or missing `fts.db` is self-healing on next startup.

---

## Testing

Three tests cover the meaningful behaviour:

**1. Exact-term recall**
Encode a memory containing a low-semantic-similarity token (e.g. `"YEP-293"`). Query `"YEP-293"`. Assert the memory appears in results. Optionally: assert it would _not_ appear in vector-only top-k by temporarily disabling FTS, confirming the test is meaningful.

**2. RRF merge correctness** (unit test `_rrf_merge` directly)
- A memory present only in the FTS list appears in merged pool.
- A memory present in both lists scores higher than one present in only one.
- Verify with synthetic ranked lists, no I/O needed.

**3. FTS rebuild idempotency**
Call `rebuild()` twice with the same memory list. Assert row count equals input length both times. Assert `search()` returns identical results after each rebuild.

All tests use SQLite `":memory:"` path — no disk I/O, no test cleanup needed.

---

## Out of Scope

- Indexing memory metadata, room IDs, or salience fields — `content` only
- Exposing FTS5 operators (`NEAR`, `*`, `""`) to callers
- Async FTS writes — synchronous is sufficient at this scale
- Any changes to `palace.py`, `vector_store.py`, `models.py`, `consolidation.py`, or `mcp_server.py`

---

## Files Changed

| File | Change |
|---|---|
| `smriti_memcore/fts_index.py` | New |
| `smriti_memcore/retrieval.py` | Add `fts_index` param, `_rrf_merge()`, update `retrieve()` |
| `smriti_memcore/core.py` | Instantiate `FTSIndex`, wire into encode/forget/close |
| `tests/test_fts_index.py` | New — 3 tests above |
