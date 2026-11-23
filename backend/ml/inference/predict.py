"""Prediction entry point: raw student attributes in, an explained estimate out.

This module is what the API calls. It deliberately owns the whole serving path -
validation, feature engineering, encoding, prediction, clipping, banding and
explanation - so the API layer never touches NumPy, pandas or scikit-learn.

What one prediction returns, and what each part means:

* **grade** - the model's point estimate, clipped to the documented 0-20 scale.
* **interval** - grade +/- the model's measured RMSE on the held-out rows. It is a
  *context band*, not a confidence interval: it uses one average error figure for every
  student, while the Phase 7 band table shows the real error is much larger for students
  far from the middle. The response says so in `interval_basis`.
* **band** - the product's risk label, with the distance to the nearest boundary so the
  interface can flag estimates that a normal-sized error would move across a cut-off.
* **contributions** - an ablation explanation. Each feature is replaced, one at a time,
  with its training reference value (median or mode) and the model is re-run. The
  difference is that feature's contribution to *this* estimate. Positive means the
  feature raised the estimate. It is a marginal effect around one reference point, not
  a causal claim and not a SHAP value; correlated features share credit arbitrarily.

The model is always the one recorded in the metadata: `early` by default, because that
is the variant usable before any period grades exist.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from ml import config, risk
from ml.preprocessing import features

DEFAULT_FEATURE_SET = features.FEATURE_SET_EARLY
DEFAULT_SUBJECT = "mat"


class ModelNotFoundError(FileNotFoundError):
    """No model artifact for the requested subject and feature set."""


class AttributeValidationError(ValueError):
    """The submitted attributes are missing, unknown or outside the documented domain."""


def artifact_path(subject: str, feature_set: str) -> Path:
    """Path of the joblib pipeline for one model."""
    return config.MODELS_DIR / f"{subject}__{feature_set}.joblib"


def metadata_path(subject: str, feature_set: str) -> Path:
    """Path of the JSON metadata for one model."""
    return config.MODELS_DIR / f"{subject}__{feature_set}.json"


def model_paths(subject: str, feature_set: str) -> tuple[Path, Path]:
    """Both artifact paths, in the order the serializer writes them."""
    return artifact_path(subject, feature_set), metadata_path(subject, feature_set)


def available_models() -> list[tuple[str, str]]:
    """Subject/feature-set pairs that have both artifacts on disk."""
    pairs = []
    for subject in config.SUBJECTS:
        for feature_set in features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES:
            artifact, metadata = model_paths(subject, feature_set)
            if artifact.exists() and metadata.exists():
                pairs.append((subject, feature_set))
    return pairs


@lru_cache(maxsize=8)
def load(subject: str, feature_set: str) -> tuple[Any, dict[str, Any]]:
    """Load one pipeline and its metadata.

    Cached because the API loads each model once: unpickling a 500-tree forest per
    request would dominate the response time.
    """
    artifact, metadata = model_paths(subject, feature_set)
    if not artifact.exists() or not metadata.exists():
        raise ModelNotFoundError(
            f"No serialized model for {subject}/{feature_set}. "
            "Run `python -m ml.inference.serialize`."
        )
    pipeline = joblib.load(artifact)
    return pipeline, json.loads(metadata.read_text(encoding="utf-8"))


def clear_cache() -> None:
    """Drop loaded artifacts (used by tests and after retraining)."""
    load.cache_clear()


def _coerce_numeric(column: str, value: Any) -> float:
    """Validate one numeric attribute against the documented domain."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AttributeValidationError(
            f"{column!r} must be numeric, got {value!r}."
        ) from error

    if not np.isfinite(number):
        raise AttributeValidationError(f"{column!r} must be finite, got {value!r}.")

    low, high = config.NUMERIC_RANGES[column]
    if not low <= number <= high:
        raise AttributeValidationError(
            f"{column!r} must be between {low} and {high}, got {number:g}."
        )
    return number


def raw_inputs(feature_set: str) -> tuple[str, ...]:
    """The attributes a caller must supply for a feature set.

    These are the published columns only. The derived columns (`parent_edu_mean`,
    `grade_mean`, ...) are computed by `features.build_features` from these inputs, so
    asking for them would be asking the caller to reimplement the pipeline.
    """
    if feature_set == features.FEATURE_SET_EARLY:
        return tuple(config.EARLY_FEATURES)
    if feature_set == features.FEATURE_SET_WITH_GRADES:
        return tuple(config.WITH_GRADES_FEATURES)
    raise AttributeValidationError(
        f"Unknown feature set {feature_set!r}; expected 'early' or 'with_grades'."
    )


def validate_attributes(
    attributes: Mapping[str, Any], feature_set: str
) -> pd.DataFrame:
    """Turn a raw attribute mapping into a validated one-row feature source frame.

    Strict on purpose: an unknown or missing column would otherwise be silently dropped
    or defaulted, and the model would answer a question nobody asked. Ranges and
    category levels come from `ml.config`, the same source the loader validates the CSV
    against, so the API cannot accept a row the training data could never contain.
    """
    required = list(raw_inputs(feature_set))
    missing = [column for column in required if column not in attributes]
    unexpected = sorted(set(attributes) - set(required))
    if missing or unexpected:
        raise AttributeValidationError(
            f"Attributes do not match the {feature_set!r} feature set: "
            f"missing={missing} unexpected={unexpected}."
        )

    row: dict[str, Any] = {}
    for column in required:
        value = attributes[column]
        if column in config.CATEGORY_LEVELS:
            allowed = config.CATEGORY_LEVELS[column]
            if value not in allowed:
                raise AttributeValidationError(
                    f"{column!r} must be one of {list(allowed)}, got {value!r}."
                )
            row[column] = value
        else:
            row[column] = _coerce_numeric(column, value)

    return pd.DataFrame([row], columns=required)


def _predict_frame(pipeline: Any, frame: pd.DataFrame, feature_set: str) -> float:
    """Predict one row and clip to the grade scale."""
    matrix = features.feature_matrix(frame, feature_set)
    raw = float(np.asarray(pipeline.predict(matrix), dtype=np.float64)[0])
    low, high = config.NUMERIC_RANGES[config.TARGET]
    return float(np.clip(raw, low, high))


def explain(
    pipeline: Any,
    metadata: dict[str, Any],
    frame: pd.DataFrame,
    feature_set: str,
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Ablation contributions for one row, largest absolute effect first.

    For each feature: replace its value with the training reference, re-predict, and
    record `full - ablated`. The estimates are recomputed from the same fitted pipeline,
    so nothing is sampled and the numbers are exactly reproducible.
    """
    matrix = features.feature_matrix(frame, feature_set)
    reference = metadata["reference_values"]
    full = _predict_frame(pipeline, frame, feature_set)

    contributions = []
    for column in matrix.columns:
        ablated = matrix.copy()
        ablated[column] = reference[column]
        ablated_grade = float(
            np.clip(
                np.asarray(pipeline.predict(ablated), dtype=np.float64)[0],
                *config.NUMERIC_RANGES[config.TARGET],
            )
        )
        contributions.append(
            {
                "feature": str(column),
                "value": matrix[column].iloc[0].item()
                if hasattr(matrix[column].iloc[0], "item")
                else matrix[column].iloc[0],
                "reference": reference[column],
                "delta": float(full - ablated_grade),
            }
        )

    contributions.sort(key=lambda row: abs(row["delta"]), reverse=True)
    kept = contributions[:limit]
    for row in kept:
        row["delta"] = round(row["delta"], 3)
    return kept


LIMITATIONS = [
    "This is a statistical estimate from a small public dataset (395 maths, 649 "
    "Portuguese records), not a guaranteed outcome.",
    "The dataset is cross-sectional, from two Portuguese secondary schools in 2005-2006; "
    "it does not establish how a student at another school would perform.",
    "Predictions use a single held-out error figure for every student. The measured "
    "error is much larger for students at the extremes of the grade scale.",
    "Risk bands are product policy on the predicted grade, not a validated "
    "institutional standard; a shift of about one RMSE can change the band.",
    "Contributions are correlational marginal effects around one reference point. They "
    "are not causal, and they do not tell you what would happen if an attribute changed.",
    "Attendance is never predicted: absences is an input attribute, not a target.",
]


def predict(
    subject: str,
    attributes: Mapping[str, Any],
    feature_set: str = DEFAULT_FEATURE_SET,
) -> dict[str, Any]:
    """Estimate one student's final grade and explain the estimate."""
    pipeline, metadata = load(subject, feature_set)
    frame = validate_attributes(attributes, feature_set)

    grade = _predict_frame(pipeline, frame, feature_set)
    metrics = metadata["metrics"]
    lower, upper = config.NUMERIC_RANGES[config.TARGET]
    rmse = float(metrics["test_rmse"])

    contributions = explain(pipeline, metadata, frame, feature_set)
    band = risk.band_label(grade)

    return {
        "subject": subject,
        "feature_set": feature_set,
        "family": metadata["family"],
        "model_version": metadata["model_version"],
        "prediction": {
            "grade": round(grade, 2),
            "band": band,
            "interval": [round(max(lower, grade - rmse), 2),
                         round(min(upper, grade + rmse), 2)],
            "interval_basis": (
                f"estimate +/- the model's held-out RMSE ({rmse:.2f} grade points); "
                "one average figure for every student, not a confidence interval"
            ),
            "distance_to_band_edge": (
                None
                if risk.distance_to_edge(grade) is None
                else round(float(risk.distance_to_edge(grade)), 2)
            ),
        },
        "context": {
            "test_rmse": rmse,
            "test_mae": float(metrics["test_mae"]),
            "test_r2": float(metrics["test_r2"]),
            "cv_rmse": float(metrics["cv_rmse"]),
            "baseline_cv_rmse": float(metrics["baseline_cv_rmse"]),
            "within_2_marks": float(metrics["within_2_marks"]),
            "rows_train": int(metadata["rows_train"]),
            "rows_test": int(metadata["rows_test"]),
            "dataset_sha256": metadata["dataset_sha256"],
        },
        "contributions": contributions,
        "contribution_basis": (
            "Ablation: each feature is replaced by its training reference value (median "
            "or mode) and the estimate is recomputed. Positive means the value raised "
            "the estimate. Correlated features share credit, so read direction, not "
            "precise magnitudes."
        ),
        "limitations": LIMITATIONS,
    }
