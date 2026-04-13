#!/usr/bin/env python3
"""
palace_to_obsidian.py — Export smriti Semantic Palace → Obsidian vault.

Maps palace architecture to Obsidian:
  Room      → Palace/<topic-slug>.md  (one note per room)
  Memory    → Section inside room note
  Room edge → [[wikilink]] between room notes
  Salience  → YAML frontmatter field

Usage:
    python3 scripts/palace_to_obsidian.py \
        --palace ~/.smriti/global/palace/palace.json \
        --vault  ~/Flex/Obsidian-vault/flex-vault/Palace

Re-run after each consolidation to keep vault in sync.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def slugify(text: str, max_len: int = 40) -> str:
    """Turn a room topic into a filename-safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"-+", "-", text)
    return text[:max_len].rstrip("-")


def short_topic(topic: str, max_words: int = 6) -> str:
    """First N words of a topic — used as the note title."""
    words = topic.split()
    return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")


def format_ts(ts: str) -> str:
    """ISO timestamp → readable date."""
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d")
    except Exception:
        return ts[:10] if ts else "unknown"


def status_icon(status: str) -> str:
    return {"active": "🟢", "pinned": "📌", "archived": "🗄️"}.get(status, "⚪")


# ── Core export ───────────────────────────────────────────────────────────────

def build_room_slug_map(rooms: dict, memories: dict) -> dict[str, str]:
    """
    Build {room_id → slug} mapping.
    Slugs are derived from room topic; duplicates get a numeric suffix.
    """
    slug_map: dict[str, str] = {}
    seen: dict[str, int] = {}

    for room_id, room in rooms.items():
        topic = room.get("topic") or room.get("centroid_topic") or room_id[:8]
        slug = slugify(topic)
        if not slug:
            slug = room_id[:8]
        if slug in seen:
            seen[slug] += 1
            slug = f"{slug}-{seen[slug]}"
        else:
            seen[slug] = 0
        slug_map[room_id] = slug

    return slug_map


def render_room_note(
    room_id: str,
    room: dict,
    memories: list[dict],
    slug_map: dict[str, str],
) -> str:
    """Render a single room as a markdown note."""
    topic = room.get("topic") or room.get("centroid_topic") or room_id[:8]
    slug = slug_map[room_id]
    links: dict = room.get("links", {})

    # ── Frontmatter ──
    avg_strength = (
        sum(m.get("strength", 0) for m in memories) / len(memories) if memories else 0
    )
    avg_salience = (
        sum(m.get("salience", {}).get("composite", 0) for m in memories) / len(memories)
        if memories else 0
    )
    connected_rooms = [slug_map[rid] for rid in links if rid in slug_map]

    frontmatter_lines = [
        "---",
        f"smriti_room_id: {room_id}",
        f"memory_count: {len(memories)}",
        f"avg_strength: {avg_strength:.2f}",
        f"avg_salience: {avg_salience:.3f}",
        f"connected_rooms: [{', '.join(connected_rooms)}]",
        "type: palace-room",
        "---",
    ]

    # ── Title + connections ──
    lines = frontmatter_lines + [
        "",
        f"# {short_topic(topic)}",
        "",
    ]

    if connected_rooms:
        links_str = " · ".join(f"[[{r}]]" for r in connected_rooms)
        lines += [f"**Connected rooms:** {links_str}", ""]

    lines += [
        f"> 🏛️ Room `{room_id[:8]}` · {len(memories)} {'memory' if len(memories) == 1 else 'memories'} · avg strength {avg_strength:.2f}",
        "",
        "---",
        "",
    ]

    # ── Memories ──
    # Sort strongest first
    sorted_mems = sorted(memories, key=lambda m: m.get("strength", 0), reverse=True)

    for m in sorted_mems:
        content = m.get("content", "")
        strength = m.get("strength", 0)
        salience = m.get("salience", {})
        sal_composite = salience.get("composite", 0) if isinstance(salience, dict) else 0
        source = m.get("source", "")
        status = m.get("status", "active")
        created = format_ts(m.get("creation_time", ""))
        mem_id = m.get("id", "")[:8]

        # Wrap long content
        wrapped = textwrap.fill(content, width=90)

        lines += [
            f"## {status_icon(status)} `{mem_id}`",
            "",
            wrapped,
            "",
            f"*strength {strength:.2f} · salience {sal_composite:.3f} · source: {source} · {created}*",
            "",
            "---",
            "",
        ]

    return "\n".join(lines)


def render_index(rooms: dict, slug_map: dict[str, str], memories: dict) -> str:
    """Render Palace/_index.md — overview of all rooms."""
    lines = [
        "---",
        "type: palace-index",
        f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"total_rooms: {len(rooms)}",
        f"total_memories: {len(memories)}",
        "---",
        "",
        "# 🏛️ Semantic Palace — Index",
        "",
        f"**{len(rooms)} rooms · {len(memories)} memories**",
        "",
        "| Room | Memories | Avg Strength | Connections |",
        "|---|---|---|---|",
    ]

    for room_id, room in sorted(rooms.items(), key=lambda x: -len(
        [m for m in memories.values() if m.get("room_id") == x[0]]
    )):
        slug = slug_map[room_id]
        room_mems = [m for m in memories.values() if m.get("room_id") == room_id]
        avg_str = (
            sum(m.get("strength", 0) for m in room_mems) / len(room_mems)
            if room_mems else 0
        )
        connected = [slug_map[rid] for rid in room.get("links", {}) if rid in slug_map]
        conn_str = " ".join(f"[[{r}]]" for r in connected) if connected else "—"
        lines.append(
            f"| [[{slug}]] | {len(room_mems)} | {avg_str:.2f} | {conn_str} |"
        )

    lines += ["", "---", "", "*Auto-generated by `scripts/palace_to_obsidian.py`*"]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def export(palace_path: str, vault_path: str) -> None:
    palace_file = Path(palace_path).expanduser()
    vault_dir = Path(vault_path).expanduser()

    print(f"Reading palace: {palace_file}")
    with open(palace_file) as f:
        palace = json.load(f)

    rooms: dict = palace.get("rooms", {})
    memories: dict = palace.get("memories", {})

    print(f"  {len(rooms)} rooms, {len(memories)} memories")

    slug_map = build_room_slug_map(rooms, memories)

    vault_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to: {vault_dir}")

    # Write room notes
    for room_id, room in rooms.items():
        slug = slug_map[room_id]
        room_mems = [m for m in memories.values() if m.get("room_id") == room_id]
        content = render_room_note(room_id, room, room_mems, slug_map)
        out_file = vault_dir / f"{slug}.md"
        out_file.write_text(content, encoding="utf-8")
        print(f"  ✓ {slug}.md ({len(room_mems)} mems)")

    # Write index
    index_content = render_index(rooms, slug_map, memories)
    (vault_dir / "_index.md").write_text(index_content, encoding="utf-8")
    print(f"  ✓ _index.md")

    print(f"\nDone — {len(rooms) + 1} files written to {vault_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export smriti palace → Obsidian vault")
    parser.add_argument(
        "--palace",
        default="~/.smriti/global/palace/palace.json",
        help="Path to palace.json",
    )
    parser.add_argument(
        "--vault",
        default="~/Flex/Obsidian-vault/flex-vault/Palace",
        help="Output directory inside Obsidian vault",
    )
    args = parser.parse_args()
    export(args.palace, args.vault)


if __name__ == "__main__":
    main()
