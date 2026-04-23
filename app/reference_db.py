from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from app.logging_config import get_logger

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = WORKSPACE_ROOT / "data" / "reference_paths.sqlite3"
REFERENCE_DIR = WORKSPACE_ROOT / "data" / "reference"
AUDIO_EXTENSIONS = {".wav", ".mp3"}
logger = get_logger(__name__)


def _normalize_word(word: str) -> str:
    return re.sub(r"\s+", " ", word.strip().lower())


def _guess_word_from_path(path: str) -> str:
    stem = Path(path).stem.strip().lower()

    # Формат сканируемых файлов: pronunciation_en_<word>[ <suffix>],
    # где suffix может быть вида "(1)", "(2)" и т.п.
    prefixed = re.match(r"^pronunciation_en_(.+)$", stem)
    if prefixed:
        tail = prefixed.group(1).strip()
        # Берем первое слово до пробела: именно оно является целевым словом.
        head = tail.split(maxsplit=1)[0] if tail else ""
        if head:
            return _normalize_word(head)

    parts = [p for p in re.split(r"[^a-zA-Z0-9а-яА-ЯёЁ]+", stem) if p]
    if not parts:
        return "unknown"

    stop_words = {
        "pronunciation",
        "reference",
        "sample",
        "audio",
        "upload",
        "mic",
        "en",
        "ru",
        "wav",
        "mp3",
    }
    filtered = [p for p in parts if p not in stop_words]
    return _normalize_word(filtered[-1] if filtered else parts[-1])


def _create_reference_paths_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reference_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            path TEXT NOT NULL,
            label TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(word, path)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reference_paths_word ON reference_paths(word)")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _has_unique_word_path(conn: sqlite3.Connection) -> bool:
    for row in conn.execute("PRAGMA index_list(reference_paths)").fetchall():
        is_unique = int(row[2]) == 1
        if not is_unique:
            continue

        index_name = str(row[1])
        index_cols = [
            str(col_row[2])
            for col_row in conn.execute(f"PRAGMA index_info('{index_name}')").fetchall()
        ]
        if index_cols == ["word", "path"]:
            return True
    return False


def _migrate_reference_paths_table(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE reference_paths RENAME TO reference_paths_legacy")
    _create_reference_paths_table(conn)

    legacy_columns = _get_table_columns(conn, "reference_paths_legacy")
    if "path" not in legacy_columns:
        conn.execute("DROP TABLE reference_paths_legacy")
        return

    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM reference_paths_legacy").fetchall()
    for row in rows:
        path_value = str(row["path"] or "").strip()
        if not path_value:
            continue

        label_value = str(row["label"] or "").strip() if "label" in legacy_columns else ""
        raw_word = str(row["word"] or "").strip() if "word" in legacy_columns else ""
        word_value = _normalize_word(raw_word) or _normalize_word(label_value) or _guess_word_from_path(path_value)
        created_at_value = str(row["created_at"] or "").strip() if "created_at" in legacy_columns else ""

        if created_at_value:
            conn.execute(
                """
                INSERT OR IGNORE INTO reference_paths(word, path, label, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (word_value, path_value, label_value, created_at_value),
            )
        else:
            conn.execute(
                """
                INSERT OR IGNORE INTO reference_paths(word, path, label)
                VALUES (?, ?, ?)
                """,
                (word_value, path_value, label_value),
            )

    conn.execute("DROP TABLE reference_paths_legacy")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing reference DB at %s", DB_PATH)

    with sqlite3.connect(DB_PATH) as conn:
        if not _table_exists(conn, "reference_paths"):
            logger.info("Creating reference_paths table")
            _create_reference_paths_table(conn)
        else:
            columns = _get_table_columns(conn, "reference_paths")
            if "word" not in columns or not _has_unique_word_path(conn):
                logger.warning("Migrating legacy reference_paths schema")
                _migrate_reference_paths_table(conn)
            else:
                _create_reference_paths_table(conn)
        conn.commit()


def list_reference_paths(word: str | None = None, limit: int | None = None) -> list[dict]:
    init_db()

    where_parts: list[str] = []
    params: list[object] = []

    clean_word = _normalize_word(word or "")
    if clean_word:
        where_parts.append("word = ?")
        params.append(clean_word)

    query = "SELECT id, word, path, label, created_at FROM reference_paths"
    if where_parts:
        query += " WHERE " + " AND ".join(where_parts)
    query += " ORDER BY created_at DESC, id DESC"

    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        result = [dict(row) for row in rows]
        logger.info("Loaded %s reference paths (word=%s, limit=%s)", len(result), clean_word or None, limit)
        return result


def list_reference_words() -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT word, COUNT(*) AS reference_count
            FROM reference_paths
            GROUP BY word
            ORDER BY word ASC
            """
        ).fetchall()
        result = [dict(row) for row in rows]
        logger.debug("Loaded %s words from reference DB", len(result))
        return result


def add_reference_path(word: str, path: str, label: str = "") -> bool:
    clean_word = _normalize_word(word)
    clean_path = path.strip()
    if not clean_word or not clean_path:
        logger.warning("Rejected empty reference insert (word=%s, path=%s)", word, path)
        return False

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO reference_paths(word, path, label) VALUES (?, ?, ?)",
            (clean_word, clean_path, label.strip()),
        )
        conn.commit()
        inserted = cursor.rowcount > 0
        if inserted:
            logger.info("Added reference path: word=%s path=%s", clean_word, clean_path)
        else:
            logger.warning("Reference path already exists: word=%s path=%s", clean_word, clean_path)
        return inserted


def delete_reference_path(reference_id: int) -> bool:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("DELETE FROM reference_paths WHERE id = ?", (reference_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted reference path id=%s", reference_id)
        else:
            logger.warning("Reference path id=%s not found for deletion", reference_id)
        return deleted


def scan_reference_dir() -> int:
    init_db()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Scanning reference dir: %s", REFERENCE_DIR)

    inserted = 0
    for file_path in sorted(REFERENCE_DIR.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(WORKSPACE_ROOT).as_posix()
        guessed_word = _guess_word_from_path(relative_path)
        if add_reference_path(guessed_word, relative_path):
            inserted += 1

    logger.info("Finished scanning reference dir, inserted=%s", inserted)

    return inserted