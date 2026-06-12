"""SQLite logging for ``/analyze`` predictions.

Each successful call logged by the FastAPI handler becomes one row in the
``predictions`` table at :data:`DB_PATH`; the schema is created on first
access. :func:`init_schema` raises on failure (called at startup), while
:func:`log_prediction` swallows errors so logging can never break a response.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Co-located with the dataset under data/ (already gitignored).
DB_PATH: Path = Path(__file__).resolve().parent / "data" / "predictions.db"

_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    input_text TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    flag_low_confidence INTEGER NOT NULL,
    flag_model_agreement INTEGER NOT NULL,
    flag_multipolarity INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);
"""

_INSERT_SQL: str = (
    "INSERT INTO predictions ("
    "created_at, input_text, label, confidence, "
    "flag_low_confidence, flag_model_agreement, flag_multipolarity, "
    "latency_ms"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)


def init_schema(db_path: Path | None = None) -> None:
    """Create the ``predictions`` table if absent (idempotent).

    ``db_path`` is read at call time, not bound as a default, so tests can
    redirect by patching :data:`DB_PATH` on the module.
    """
    path = db_path if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA_SQL)


def log_prediction(
    result: dict[str, Any],
    latency_ms: float,
    db_path: Path | None = None,
) -> None:
    """Persist one ``/analyze`` result, best-effort.

    Args:
        result: The dict returned by :func:`pipeline.analyze`.
        latency_ms: Time spent inside ``analyze()``, in milliseconds.
        db_path: SQLite file override (mainly for tests); when ``None``,
            :data:`DB_PATH` is read at call time.

    On any sqlite error this warns to stderr and returns normally.
    """
    path = db_path if db_path is not None else DB_PATH
    try:
        with sqlite3.connect(path) as conn:
            conn.execute(
                _INSERT_SQL,
                (
                    datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    result["text"],
                    result["label"],
                    float(result["confidence"]),
                    int(result["flags"]["low_confidence"]),
                    int(result["flags"]["model_agreement"]),
                    int(result["flags"]["multipolarity"]),
                    float(latency_ms),
                ),
            )
    except sqlite3.Error as exc:
        print(f"[db] WARN: failed to log prediction: {exc}", file=sys.stderr)
