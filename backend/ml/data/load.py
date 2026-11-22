"""Load and validate the raw UCI Student Performance CSVs.

The dataset is a fixed public file, so any deviation from it is a bug or the wrong
file, never something to patch over at runtime. This module therefore fails loudly:
`DatasetSchemaError` is raised with the offending column and values instead of
coercing, dropping or imputing.

Nothing is cleaned here beyond type coercion and the addition of a row-derived ID.
Cleaning decisions (for example what to do with `G3 = 0`) belong to the next stage and
must be visible there, not hidden in a loader.

Run the report with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.data.load
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ml import config

# Generated from the row position, because the dataset carries no student identifier.
# It is an internal handle for joins and API calls, not an anonymised real identity.
ID_COLUMN = "student_id"


class DatasetSchemaError(ValueError):
    """A CSV does not match the documented schema or value domains.

    A named exception (rather than bare ValueError) lets callers and tests distinguish
    "the data is not what we expect" from ordinary programming errors.
    """


def raw_path(subject: str) -> Path:
    """Return the CSV path for a subject, rejecting unknown subject names."""
    if subject not in config.SUBJECT_FILES:
        raise KeyError(
            f"Unknown subject {subject!r}; expected one of {config.SUBJECTS}."
        )
    return config.RAW_DIR / config.SUBJECT_FILES[subject]


def _check_columns(frame: pd.DataFrame, subject: str) -> None:
    expected = set(config.EXPECTED_COLUMNS)
    found = set(frame.columns)

    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing or unexpected:
        raise DatasetSchemaError(
            f"{subject}: column mismatch. missing={missing} unexpected={unexpected}"
        )

    # Order is not semantically important, but the published file has a fixed one, and
    # a reordered file is a signal that this is not the file we validated against.
    if list(frame.columns) != list(config.EXPECTED_COLUMNS):
        raise DatasetSchemaError(
            f"{subject}: columns are correct but in an unexpected order."
        )


def _check_row_count(frame: pd.DataFrame, subject: str) -> None:
    expected = config.EXPECTED_ROW_COUNTS[subject]
    if len(frame) != expected:
        raise DatasetSchemaError(
            f"{subject}: expected {expected} rows, found {len(frame)}."
        )


def _check_nulls(frame: pd.DataFrame, subject: str) -> None:
    null_counts = frame.isna().sum()
    offenders = null_counts[null_counts > 0]
    if not offenders.empty:
        detail = ", ".join(f"{col}={count}" for col, count in offenders.items())
        raise DatasetSchemaError(
            f"{subject}: unexpected null values ({detail}). The published dataset has "
            "none, so this indicates a damaged or different file."
        )


def _coerce_numeric(frame: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Convert numeric columns to int64, failing on anything non-numeric or fractional.

    `pd.to_numeric(errors="coerce")` is used only to *detect* bad cells; the resulting
    NaN is treated as an error rather than kept, so a typo can never silently become a
    missing value.
    """
    out = frame.copy()
    for column in config.NUMERIC_COLUMNS:
        original = out[column]
        numeric = pd.to_numeric(original, errors="coerce")

        bad = numeric.isna()
        if bad.any():
            samples = original[bad].head(5).tolist()
            raise DatasetSchemaError(
                f"{subject}: column {column!r} has {int(bad.sum())} non-numeric "
                f"value(s), e.g. {samples}."
            )

        fractional = numeric != numeric.round()
        if fractional.any():
            samples = numeric[fractional].head(5).tolist()
            raise DatasetSchemaError(
                f"{subject}: column {column!r} has {int(fractional.sum())} fractional "
                f"value(s), e.g. {samples}; the dataset is integer-valued."
            )

        out[column] = numeric.astype("int64")

    return out


def _check_dtypes(frame: pd.DataFrame, subject: str) -> None:
    for column in config.NUMERIC_COLUMNS:
        if str(frame[column].dtype) != "int64":
            raise DatasetSchemaError(
                f"{subject}: column {column!r} is {frame[column].dtype}, expected int64."
            )
    for column in config.CATEGORICAL_COLUMNS:
        if str(frame[column].dtype) != "object":
            raise DatasetSchemaError(
                f"{subject}: column {column!r} is {frame[column].dtype}, expected "
                "object/string."
            )


def _check_ranges(frame: pd.DataFrame, subject: str) -> None:
    for column, (low, high) in config.NUMERIC_RANGES.items():
        series = frame[column]
        out_of_range = (series < low) | (series > high)
        if out_of_range.any():
            samples = series[out_of_range].head(5).tolist()
            raise DatasetSchemaError(
                f"{subject}: column {column!r} has {int(out_of_range.sum())} value(s) "
                f"outside the documented range [{low}, {high}], e.g. {samples}."
            )


def _check_categories(frame: pd.DataFrame, subject: str) -> None:
    for column, allowed in config.CATEGORY_LEVELS.items():
        seen = set(frame[column].unique())
        unexpected = sorted(seen - set(allowed))
        if unexpected:
            raise DatasetSchemaError(
                f"{subject}: column {column!r} contains undocumented level(s) "
                f"{unexpected}; allowed are {list(allowed)}."
            )


def _grade_band_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count target values per product risk band (cut-offs are policy, see config)."""
    bands = pd.cut(
        frame[config.TARGET],
        bins=config.RISK_BAND_EDGES,
        labels=config.RISK_BAND_LABELS,
        right=False,
    )
    counts = bands.value_counts(sort=False)
    return {label: int(counts.get(label, 0)) for label in config.RISK_BAND_LABELS}


def validate_subject(frame: pd.DataFrame, subject: str) -> dict[str, Any]:
    """Run every schema and domain check, then return a factual report.

    Raises:
        DatasetSchemaError: on the first structural problem found.
    """
    _check_columns(frame, subject)
    _check_row_count(frame, subject)
    _check_nulls(frame, subject)
    _check_dtypes(frame, subject)
    _check_ranges(frame, subject)
    _check_categories(frame, subject)

    zero_rows = int((frame[config.TARGET] == 0).sum())

    return {
        "subject": subject,
        "source": config.SUBJECT_FILES[subject],
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "grade_zero_rows": zero_rows,
        "grade_zero_share": round(zero_rows / len(frame), 4),
        "target_zero_share_of_total": None,  # filled in by load_all()
        "grade_means": {
            column: round(float(frame[column].mean()), 3)
            for column in config.GRADE_COLUMNS
        },
        "grade_bands": _grade_band_counts(frame),
    }


def load_subject(subject: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read, validate and ID one subject. Returns the frame and its report.

    The returned frame is the raw file plus `student_id` as the first column; no rows
    are dropped and no values are altered.
    """
    path = raw_path(subject)
    if not path.exists():
        raise DatasetSchemaError(
            f"Missing dataset file: {path}. Download the UCI Student Performance "
            "archive and place both CSVs in backend/ml/data/raw/."
        )

    frame = pd.read_csv(path, sep=config.CSV_DELIMITER)
    frame = _coerce_numeric(frame, subject)
    report = validate_subject(frame, subject)

    ids = [f"{subject.upper()}-{position:04d}" for position in range(1, len(frame) + 1)]
    if len(set(ids)) != len(ids):
        raise DatasetSchemaError(f"{subject}: generated student IDs are not unique.")

    frame.insert(0, ID_COLUMN, pd.Series(ids, index=frame.index, dtype="object"))
    report["id_example"] = ids[0]

    return frame, report


def load_all() -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Load both subjects. Never concatenates them (see docs/architecture.md §3)."""
    frames: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, Any]] = []

    for subject in config.SUBJECTS:
        frame, report = load_subject(subject)
        frames[subject] = frame
        reports.append(report)

    total_zero = sum(report["grade_zero_rows"] for report in reports)
    total_rows = sum(report["rows"] for report in reports)
    for report in reports:
        report["target_zero_share_of_total"] = round(total_zero / total_rows, 4)

    return frames, reports


def format_report(reports: list[dict[str, Any]]) -> str:
    """Render the load report as plain text for the terminal."""
    lines = ["Dataset load report", "=" * 72]

    for report in reports:
        bands = "  ".join(
            f"{label}={count}" for label, count in report["grade_bands"].items()
        )
        means = "  ".join(
            f"{column}={value}" for column, value in report["grade_means"].items()
        )
        lines += [
            "",
            f"{report['subject']}  ({report['source']})",
            f"  rows={report['rows']}  columns={report['columns']}  "
            f"id example={report['id_example']}",
            f"  duplicate rows={report['duplicate_rows']}",
            f"  G3 = 0: {report['grade_zero_rows']} rows "
            f"({report['grade_zero_share']:.1%} of this subject, "
            f"{report['target_zero_share_of_total']:.1%} of both files)",
            f"  grade means: {means}",
            f"  G3 bands:    {bands}",
        ]

    lines += [
        "",
        "All schema and domain checks passed.",
        "G3 = 0 rows are kept, counted, and documented as a known artefact.",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        _, reports = load_all()
    except DatasetSchemaError as error:
        print(f"Dataset validation failed.\n  {error}", file=sys.stderr)
        return 1

    print(format_report(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
