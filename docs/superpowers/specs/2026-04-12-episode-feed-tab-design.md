# Episode Feed Tab — Design Spec

**Date:** 2026-04-12
**Status:** Approved

## Problem

The Smriti Memory Browser UI currently shows only palace memories (`palace.json` — 31 long-term memories). Raw episodic memories stored in `episodes.db` (49 rows) are invisible, making it impossible to see the full encoding history or debug why certain memories were or weren't promoted to the palace.

## Goal

Add a `📼 Episode Feed` tab to the UI that shows the raw episode buffer as a reverse-chronological table, with consolidation status clearly visible.

## Scope

- New `/api/episodes` server endpoint
- New `_read_episodes()` data function
- New `📼 Episode Feed` tab in the browser UI
- No changes to existing tabs or endpoints

Out of scope: search/filter, trajectory grouping, reflections display.

## Data Layer

### Storage location
```
<SMRITI_STORAGE_PATH>/episodes/episodes.db
```

### New function: `_read_episodes(storage_path: str) -> list[dict]`

Reads `episodes.db` using stdlib `sqlite3`. Returns a list of dicts ordered by `timestamp DESC`:

```python
{
    "id":          str,   # UUID
    "content":     str,   # raw episode text
    "timestamp":   str,   # ISO8601 string
    "source":      str,   # "user_stated" | "inferred" | "direct" | "external"
    "salience":    float, # composite from salience_json
    "consolidated": bool, # True if promoted to palace
}
```

`salience_json` is parsed with `json.loads()` and the `composite` key extracted. Defaults to `0.0` if missing or malformed. Returns empty list if `episodes.db` does not exist.

### New endpoint: `GET /api/episodes`

Returns `json.dumps(episodes_list)` with `Content-Type: application/json`. Handled in `_Handler.do_GET` alongside the existing `/api/graph` route.

## UI Layer

### Tab bar change

Add a 4th tab after Statistics:

```html
<div class="tab" id="tab-episodes" onclick="showTab('episodes')">📼 Episode Feed</div>
```

Update `showTab()` to include `'episodes'` in the tab/view list.

### View container

```html
<div id="view-episodes" style="display:none; flex:1; overflow:auto; padding:24px;">
  <table id="episodes-table">...</table>
</div>
```

### Table columns

| Column | Field | Notes |
|--------|-------|-------|
| Timestamp | `timestamp` | Formatted as `YYYY-MM-DD HH:MM`, newest first |
| Content | `content` | Truncated at 120 chars; full text in `title` tooltip |
| Source | `source` | Plain text |
| Salience | `salience` | 3 decimal places |
| Status | `consolidated` | Green `✓ consolidated` or amber `⏳ pending` badge |

### Lazy loading

`buildEpisodes()` fetches `/api/episodes` only when the tab is first activated. Subsequent clicks reuse cached data unless the user hits `↻ Refresh` (which clears the cache and re-fetches).

### Styling

Reuses existing `.tbl-wrap`, `table`, `th`, `td`, `.status-ok` CSS classes for visual consistency with the Memory Table tab.

## Error handling

- `episodes.db` missing → `_read_episodes` returns `[]` → UI shows "No episodes found."
- `salience_json` malformed → defaults to `0.0`, does not crash
- Fetch error in JS → logs to console, shows inline error message in the tab

## Changes summary

| File | Change |
|------|--------|
| `smriti_memcore/ui/server.py` | Add `_read_episodes()`, add `/api/episodes` route, add tab HTML, add JS `buildEpisodes()` and `showTab` update |

## Non-goals

- Trajectory grouping (future)
- Reflections display (future)
- Search/filter on episodes (future)
- Linking episodes to their corresponding palace memory (future)
