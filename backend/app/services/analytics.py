"""Aggregations over the validated dataset and the saved ML records.

Everything here is computed from the CSVs on disk and from the artifacts produced by
the training pipeline. No value is typed in by hand: if a number appears on the
dashboard, it can be traced to a file and re-derived.

The two subjects are never pooled into one grade distribution or one mean. They share
schools and probably students, and were measured on different courses, so a combined
figure would describe nothing. Where a total is useful (headcount, at-risk count) it is
labelled as a sum of two separate populations.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from app.core import dataset, db
from ml import config, risk
from ml.training import compare

# Attributes offered for the "mean grade by attribute" view. Numeric grades are excluded
# on purpose: grouping by G1 or G2 would restate the with_grades model, not describe the
# population.
BREAKDOWN_ATTRIBUTES = (
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
)

CORRELATION_COLUMNS = (
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
)


def _round(value: float | None, places: int = 3) -> float | None:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return round(float(value), places)


def _band_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Rows per risk band, keyed in the documented band order."""
    bands = pd.cut(
        frame[config.TARGET],
        bins=config.RISK_BAND_EDGES,
        labels=config.RISK_BAND_LABELS,
        right=False,
    )
    counts = bands.value_counts()
    # Empty bands must still appear: a band with zero students is a finding, and a
    # missing key would make the chart silently drop a category.
    return {label: int(counts.get(label, 0)) for label in config.RISK_BAND_LABELS}


def subject_stats(subject: str) -> dict[str, Any]:
    """Descriptive statistics for one subject's final grades."""
    frame = dataset.frame(subject)
    grades = frame[config.TARGET]

    return {
        "subject": subject,
        "rows": int(len(frame)),
        "mean": _round(grades.mean()),
        "median": _round(grades.median()),
        "std": _round(grades.std()),
        "min": int(grades.min()),
        "max": int(grades.max()),
        "pass_rate": _round(float((grades >= 10).mean())),
        "at_risk": int((grades < 10).sum()),
        "bands": _band_counts(frame),
        "mean_absences": _round(frame["absences"].mean()),
        "mean_studytime": _round(frame["studytime"].mean()),
        "failures_rate": _round(float((frame["failures"] > 0).mean())),
    }


def summary() -> dict[str, Any]:
    """Headline figures for the dashboard's metric cards."""
    stats = {subject: subject_stats(subject) for subject in config.SUBJECTS}
    frames = dataset.frames()

    return {
        "students": {
            "total": int(sum(len(table) for table in frames.values())),
            **{subject: stats[subject]["rows"] for subject in config.SUBJECTS},
        },
        "grades": {
            subject: {
                "mean": stats[subject]["mean"],
                "median": stats[subject]["median"],
                "pass_rate": stats[subject]["pass_rate"],
            }
            for subject in config.SUBJECTS
        },
        "bands": {
            subject: stats[subject]["bands"] for subject in config.SUBJECTS
        },
        "at_risk": {
            **{subject: stats[subject]["at_risk"] for subject in config.SUBJECTS},
            "total": int(sum(stats[subject]["at_risk"] for subject in config.SUBJECTS)),
        },
        "predictions_logged": db.count_predictions(),
        "recent_predictions": db.recent_predictions(limit=6),
        "risk_bands": risk.band_ranges(),
        "dataset": {
            "source": "UCI ML Repository, dataset 320 - Student Performance",
            "subjects": list(config.SUBJECTS),
        },
    }


def distribution(subject: str | None = None) -> dict[str, Any]:
    """Counts per grade point (0-20) for one subject, or both side by side.

    Integers are used as bins rather than ranges because the scale itself is discrete:
    a histogram of 0-20 with five-point bins would hide the pass threshold, which is the
    only boundary that matters here.
    """
    subjects = [subject] if subject else list(config.SUBJECTS)
    series = []
    for name in subjects:
        counts = (
            dataset.frame(name)[config.TARGET]
            .value_counts()
            .reindex(range(0, 21), fill_value=0)
            .sort_index()
        )
        series.append(
            {
                "subject": name,
                "n": int(counts.sum()),
                "counts": [
                    {"grade": int(grade), "count": int(count)}
                    for grade, count in counts.items()
                ],
            }
        )

    return {
        "subjects": series,
        "pass_mark": 10,
    }


def correlation(subject: str) -> dict[str, Any]:
    """Pearson correlation matrix over numeric attributes, including all three grades."""
    frame = dataset.frame(subject)
    matrix = frame[list(CORRELATION_COLUMNS)].corr(numeric_only=True)

    labels = list(CORRELATION_COLUMNS)
    values = [
        [_round(value, 3) for value in matrix.loc[row].tolist()] for row in labels
    ]

    return {
        "subject": subject,
        "n": int(len(frame)),
        "labels": labels,
        "matrix": values,
        "caveat": (
            "Pearson correlation measures linear association only. It is not causation, "
            "and the grades are strongly related by construction (same course, three "
            "periods)."
        ),
    }


def attribute_breakdown(subject: str, attribute: str) -> dict[str, Any]:
    """Mean final grade per level of one attribute, beside the group size.

    The size is always returned: a mean drawn from four students is displayed
    differently in the interface, and hiding the count is how small-sample noise becomes
    a confident-looking claim.
    """
    if attribute not in BREAKDOWN_ATTRIBUTES:
        raise KeyError(attribute)

    frame = dataset.frame(subject)
    grouped = frame.groupby(attribute, observed=True)[config.TARGET]
    aggregate = grouped.agg(["count", "mean"])

    rows = [
        {
            "value": int(value) if isinstance(value, (np.integer, int)) else str(value),
            "n": int(aggregate.loc[value, "count"]),
            "mean": _round(aggregate.loc[value, "mean"], 2),
        }
        for value in aggregate.index
    ]
    rows.sort(key=lambda row: row["value"])

    overall = float(frame[config.TARGET].mean())
    return {
        "subject": subject,
        "attribute": attribute,
        "overall_mean": _round(overall, 2),
        "n": int(len(frame)),
        "groups": rows,
        "caveat": (
            "Group means are descriptive, not adjusted for anything else. Groups differ "
            "in size and in every other attribute too."
        ),
    }


def feature_importance(subject: str, feature_set: str, limit: int = 14) -> dict[str, Any]:
    """Permutation importance recorded by the serializer, largest effect first."""
    metadata = _metadata(subject, feature_set)
    rows = metadata["feature_importance"][:limit]

    return {
        "subject": subject,
        "feature_set": feature_set,
        "n": metadata["rows_test"],
        "rows": [
            {
                "feature": row["feature"],
                "importance": _round(row["importance"], 3),
                "std": _round(row["std"], 3),
            }
            for row in rows
        ],
        "basis": (
            "Permutation importance on the held-out rows: the increase in RMSE when one "
            "feature is shuffled. Descriptive, not causal, and unstable when features "
            "are correlated."
        ),
    }


def model_comparison() -> dict[str, Any]:
    """Every candidate's CV score per subject and feature set, beside the held-out score."""
    candidates: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []

    for subject in config.SUBJECTS:
        for feature_set in ("early", "with_grades"):
            record = compare.load_results(subject, feature_set)
            for result in record["results"]:
                candidates.append(
                    {
                        "subject": subject,
                        "feature_set": feature_set,
                        "name": result["name"],
                        "baseline": result["baseline"],
                        "selected": result["name"] == record["selected"],
                        "cv_rmse": _round(result["rmse_mean"]),
                        "cv_rmse_std": _round(result["rmse_std"]),
                        "cv_mae": _round(result["mae_mean"]),
                        "cv_r2": _round(result["r2_mean"]),
                    }
                )

            metadata = _metadata(subject, feature_set)
            chosen.append(
                {
                    "subject": subject,
                    "feature_set": feature_set,
                    "family": metadata["family"],
                    "chosen": metadata["chosen"],
                    "params": metadata["params"],
                    "cv_rmse": _round(metadata["metrics"]["cv_rmse"]),
                    "cv_mae": _round(metadata["metrics"]["cv_mae"]),
                    "baseline_cv_rmse": _round(metadata["metrics"]["baseline_cv_rmse"]),
                    "test_rmse": _round(metadata["metrics"]["test_rmse"]),
                    "test_mae": _round(metadata["metrics"]["test_mae"]),
                    "test_r2": _round(metadata["metrics"]["test_r2"]),
                    "within_2_marks": _round(metadata["metrics"]["within_2_marks"]),
                    "rows_train": metadata["rows_train"],
                    "rows_test": metadata["rows_test"],
                }
            )

    return {
        "candidates": candidates,
        "chosen": chosen,
        "selection_rule": "Lowest 5-fold CV RMSE, tie-broken by MAE then fold spread.",
        "caveat": (
            "CV scores are on training rows only. Held-out scores come from a single "
            "80/20 split and were opened once, so they are not re-estimates."
        ),
    }


def subject_comparison() -> dict[str, Any]:
    """Population-level statistics for both subjects, side by side."""
    return {
        "subjects": [subject_stats(subject) for subject in config.SUBJECTS],
        "caveat": (
            "The two files share schools and likely students but cover different "
            "courses; they are separate populations and are never pooled."
        ),
    }


@lru_cache(maxsize=8)
def _metadata(subject: str, feature_set: str) -> dict[str, Any]:
    path = config.MODELS_DIR / f"{subject}__{feature_set}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
