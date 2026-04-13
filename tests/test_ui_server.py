"""Tests for smriti_memcore.ui.server — episode feed data layer."""
import json
import sqlite3
import pytest
from pathlib import Path
from smriti_memcore.ui.server import _read_episodes


def _make_db(tmp_path, rows):
    """Create a minimal episodes.db with given rows."""
    db = tmp_path / "episodes" / "episodes.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            content TEXT,
            timestamp TEXT,
            salience_json TEXT,
            source TEXT,
            trajectory_id TEXT,
            trajectory_step INTEGER,
            reflections_json TEXT,
            consolidated INTEGER DEFAULT 0,
            metadata_json TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return str(tmp_path)


def test_read_episodes_returns_list_of_dicts(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "content one", "2026-04-12T10:00:00", '{"composite": 0.5}',
         "user_stated", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert isinstance(result, list)
    assert len(result) == 1
    ep = result[0]
    assert ep["id"] == "id-1"
    assert ep["content"] == "content one"
    assert ep["timestamp"] == "2026-04-12T10:00:00"
    assert ep["source"] == "user_stated"
    assert ep["salience"] == pytest.approx(0.5)
    assert ep["consolidated"] is True  # must be Python bool, not int 1


def test_read_episodes_consolidated_cast_to_bool(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-0", "pending ep", "2026-04-12T09:00:00", '{"composite":0.1}',
         "direct", None, 0, None, 0, None),
        ("id-1", "done ep",   "2026-04-12T10:00:00", '{"composite":0.9}',
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    by_id = {ep["id"]: ep for ep in result}
    assert by_id["id-0"]["consolidated"] is False
    assert by_id["id-1"]["consolidated"] is True


def test_read_episodes_null_salience_json(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "no sal", "2026-04-12T10:00:00", None,
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert result[0]["salience"] == pytest.approx(0.0)


def test_read_episodes_malformed_salience_json(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "bad sal", "2026-04-12T10:00:00", "not-json",
         "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert result[0]["salience"] == pytest.approx(0.0)


def test_read_episodes_ordered_newest_first(tmp_path):
    storage = _make_db(tmp_path, [
        ("id-1", "old", "2026-04-10T10:00:00", '{"composite":0.3}', "direct", None, 0, None, 1, None),
        ("id-2", "new", "2026-04-12T10:00:00", '{"composite":0.7}', "direct", None, 0, None, 1, None),
        ("id-3", "mid", "2026-04-11T10:00:00", '{"composite":0.5}', "direct", None, 0, None, 1, None),
    ])
    result = _read_episodes(storage)
    assert [ep["id"] for ep in result] == ["id-2", "id-3", "id-1"]


def test_read_episodes_missing_db_returns_empty(tmp_path):
    result = _read_episodes(str(tmp_path))  # no episodes/ subdir
    assert result == []
