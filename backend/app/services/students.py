"""Dataset browsing: paginated records and single-record detail.

Records are the public UCI rows. IDs (`MAT-0001`) are generated row handles, not
anonymised identities; the dataset contains no names or identifiers to begin with.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from app.core import dataset
from ml import config
from ml.data import load


def _row(frame: pd.DataFrame, subject: str, position: int) -> dict[str, Any]:
    record = frame.iloc[position]
    return {
        "student_id": str(record[load.ID_COLUMN]),
        "subject": subject,
        "age": int(record["age"]),
        "sex": str(record["sex"]),
        "studytime": int(record["studytime"]),
        "failures": int(record["failures"]),
        "absences": int(record["absences"]),
        "G1": int(record["G1"]),
        "G2": int(record["G2"]),
        "G3": int(record["G3"]),
        "band": dataset.band_of(record["G3"]),
    }


def list_students(
    *,
    subject: str | None = None,
    search: str | None = None,
    band: str | None = None,
    sort: str = "student_id",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """Filter, sort and paginate the records across one or both subjects."""
    subjects = [subject] if subject else list(config.SUBJECTS)

    rows: list[dict[str, Any]] = []
    for name in subjects:
        frame = dataset.frame(name)
        for position in range(len(frame)):
            rows.append(_row(frame, name, position))

    if search:
        needle = search.strip().upper()
        rows = [row for row in rows if needle in row["student_id"]]
    if band:
        rows = [row for row in rows if row["band"] == band]

    sorters = {
        "student_id": lambda row: row["student_id"],
        "grade_asc": lambda row: (row["G3"], row["student_id"]),
        "grade_desc": lambda row: (-row["G3"], row["student_id"]),
        "absences_desc": lambda row: (-row["absences"], row["student_id"]),
    }
    if sort not in sorters:
        raise KeyError(sort)
    rows.sort(key=sorters[sort])

    total = len(rows)
    pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    window = rows[start:start + page_size]

    return {
        "items": window,
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": pages,
    }


def get_student(student_id: str) -> dict[str, Any]:
    """Full attribute record for one student, plus the band derived from the actual grade."""
    subject, record = dataset.find_student(student_id)

    attributes = {
        column: (
            int(record[column])
            if column in config.NUMERIC_COLUMNS
            else str(record[column])
        )
        for column in config.EXPECTED_COLUMNS
    }

    return {
        "student_id": str(record[load.ID_COLUMN]),
        "subject": subject,
        "attributes": attributes,
        "band": dataset.band_of(record[config.TARGET]),
        "note": (
            "Record from the public UCI dataset. The ID is a generated row index, not an "
            "anonymised real identifier."
        ),
    }
