"""Dataset access for the API, loaded and validated once per process.

The CSV files are the single source of truth for the historical records the dashboard
describes. They are loaded through `ml.data.load`, which means the API can only ever
serve rows that passed the same schema and domain checks the models were trained
against - the service cannot invent a student.

`lru_cache` is the whole caching strategy. At most 1,044 rows are aggregated per
request; anything more elaborate would be theatre.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from ml import cbse, config, risk
from ml.data import load


@lru_cache(maxsize=1)
def dataset() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Both validated subject frames plus their load reports."""
    return load.load_all()


def frames() -> dict[str, pd.DataFrame]:
    return dataset()[0]


def reports() -> list[dict[str, Any]]:
    return dataset()[1]


def frame(subject: str) -> pd.DataFrame:
    """One subject frame, or KeyError naming the valid subjects."""
    available = frames()
    if subject not in available:
        raise KeyError(subject)
    return available[subject]


def require_subject(subject: str) -> str:
    """Validate a subject name and return it unchanged."""
    if subject not in config.SUBJECTS:
        raise KeyError(subject)
    return subject


def band_of(grade: float) -> str:
    """Band label for a native 0-20 grade, via the documented CBSE band policy."""
    return risk.band_label(cbse.to_marks(grade))


def find_student(student_id: str) -> tuple[str, pd.Series]:
    """Locate one record by its generated ID (`MAT-0001`), across both subjects.

    The ID prefix is the subject, so the lookup is exact rather than a scan of both
    frames: `MAT-` rows only exist in the maths file.
    """
    key = student_id.strip().upper()
    subject = key.split("-", 1)[0].lower()
    if subject not in config.SUBJECTS:
        raise KeyError(student_id)

    table = frame(subject)
    matches = table[table[load.ID_COLUMN] == key]
    if matches.empty:
        raise KeyError(student_id)
    return subject, matches.iloc[0]
