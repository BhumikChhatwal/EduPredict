"""SQLite persistence for prediction history, using the standard library only.

Why a database at all: "recent prediction activity" has to survive a restart to be
truthful. Storing the submissions in a process-level list would make the dashboard's
activity table a fiction after the first reload.

Why no ORM: the schema is one table and four queries. An ORM would add a dependency and
hide the SQL without removing any of it.

Privacy: the service stores what the caller submitted (attributes the user chose to
enter) plus the estimate. Anonymous submissions carry `student_id = NULL`. Nothing here
identifies a real person - the dataset has no identifiers to begin with.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app.core import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id    TEXT,
    subject       TEXT NOT NULL,
    feature_set   TEXT NOT NULL,
    model_version TEXT NOT NULL,
    family        TEXT NOT NULL,
    predicted_g3  REAL NOT NULL,
    risk_band     TEXT NOT NULL,
    payload_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_student
    ON predictions (student_id, created_at DESC);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """One connection per operation, committed on success and closed either way.

    Endpoints are synchronous and therefore run in a worker thread, so a connection
    shared across threads would need `check_same_thread=False` and its own locking. A
    short-lived connection avoids the question entirely.
    """
    connection = sqlite3.connect(config.db_path())
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    """Create the schema if it is absent. Safe to call on every startup."""
    path = config.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect() as connection:
        connection.executescript(SCHEMA)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "student_id": row["student_id"],
        "subject": row["subject"],
        "feature_set": row["feature_set"],
        "model_version": row["model_version"],
        "family": row["family"],
        "predicted_g3": float(row["predicted_g3"]),
        "risk_band": row["risk_band"],
        "attributes": json.loads(row["payload_json"]),
        "created_at": row["created_at"],
    }


def save_prediction(
    *,
    subject: str,
    feature_set: str,
    model_version: str,
    family: str,
    predicted_g3: float,
    risk_band: str,
    attributes: dict[str, Any],
    student_id: str | None = None,
) -> dict[str, Any]:
    """Insert one prediction and return the stored row."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO predictions (student_id, subject, feature_set, model_version,
                                     family, predicted_g3, risk_band, payload_json,
                                     created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                subject,
                feature_set,
                model_version,
                family,
                float(predicted_g3),
                risk_band,
                json.dumps(attributes, sort_keys=True),
                created_at,
            ),
        )
        prediction_id = int(cursor.lastrowid)

    return {
        "id": prediction_id,
        "student_id": student_id,
        "subject": subject,
        "feature_set": feature_set,
        "model_version": model_version,
        "family": family,
        "predicted_g3": float(predicted_g3),
        "risk_band": risk_band,
        "attributes": attributes,
        "created_at": created_at,
    }


def recent_predictions(limit: int = 10) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def predictions_for_student(student_id: str) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM predictions WHERE student_id = ? ORDER BY id DESC",
            (student_id,),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def count_predictions() -> int:
    with connect() as connection:
        row = connection.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()
    return int(row["n"])
