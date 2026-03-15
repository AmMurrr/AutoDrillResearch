from __future__ import annotations

import sqlite3
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = WORKSPACE_ROOT / "data" / "reference_paths.sqlite3"
REFERENCE_DIR = WORKSPACE_ROOT / "data" / "reference"
AUDIO_EXTENSIONS = {".wav", ".mp3"}


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                label TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def list_reference_paths() -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, path, label, created_at FROM reference_paths ORDER BY created_at DESC, id DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def add_reference_path(path: str, label: str = "") -> bool:
    clean_path = path.strip()
    if not clean_path:
        return False

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO reference_paths(path, label) VALUES (?, ?)",
            (clean_path, label.strip()),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_reference_path(reference_id: int) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM reference_paths WHERE id = ?", (reference_id,))
        conn.commit()
        return cursor.rowcount > 0


def scan_reference_dir() -> int:
    init_db()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    inserted = 0
    for file_path in sorted(REFERENCE_DIR.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(WORKSPACE_ROOT).as_posix()
        if add_reference_path(relative_path):
            inserted += 1

    return inserted