"""Derived features for the student performance models.

Every feature here is computed row by row from that row's own attributes, so nothing
can leak across rows or from the target. Derived values are *added*; the raw columns
stay in the frame so the encoding stage can still choose between them once importance
has been measured rather than assumed.

Design rules:

* No feature may use `G3`. `grade_mean` and `grade_trend` use `G1`/`G2` only, and they
  exist only in the `with_grades` variant.
* A feature earns its place only if the reason is statable without looking at the
  target. `parent_edu_gap` is included on that basis and may still turn out to carry
  no signal; that is measured later, not assumed now.
* Derived columns are float or int64 and are range-checked like the raw data.

Run the summary with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.preprocessing.features
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from ml import config
from ml.data import load

FEATURE_SET_EARLY = "early"
FEATURE_SET_WITH_GRADES = "with_grades"

# Each entry: name -> one-line justification. Kept next to the code so the "why" cannot
# drift away from the implementation.
DERIVED_RATIONALE = {
    "parent_edu_mean": (
        "Average parental education level: a single household-education signal instead "
        "of two highly correlated columns (Medu and Fedu correlate strongly)."
    ),
    "parent_edu_max": (
        "Highest parental education: the 'best available' household resource, which "
        "literature on educational attainment finds more predictive than the average."
    ),
    "parent_edu_gap": (
        "Absolute difference between the parents' education levels: whether the "
        "household is educationally even or uneven. Direction is discarded because the "
        "dataset gives no reason to treat mother and father asymmetrically."
    ),
    "alc_total": (
        "Weekday plus weekend alcohol use as one interpretable total, instead of two "
        "columns that move together. Never presented as a lever in the interface."
    ),
    "support_total": (
        "Count of the three support sources present (school support, family support, "
        "paid classes), 0-3. Assumes equal weight, an assumption recorded as a caveat: "
        "school support often flags a struggling student, so it can point the other way."
    ),
    "failures_any": (
        "Binary flag for any past class failure. The data dictionary already buckets "
        "failures at 3 or more into 4, which shows the effect is threshold-like rather "
        "than linear in the count."
    ),
    "grade_mean": (
        "Mean of the two period grades: a variance-reduced level signal that trees "
        "cannot construct themselves. `with_grades` only."
    ),
    "grade_trend": (
        "Second period minus first: direction of change, which neither G1 nor G2 "
        "carries on its own. `with_grades` only."
    ),
}

EARLY_DERIVED = (
    "parent_edu_mean",
    "parent_edu_max",
    "parent_edu_gap",
    "alc_total",
    "support_total",
    "failures_any",
)

GRADE_DERIVED = ("grade_mean", "grade_trend")

# Documented bounds for the derived columns, checked by validate_derived().
DERIVED_RANGES: dict[str, tuple[float, float]] = {
    "parent_edu_mean": (0.0, 4.0),
    "parent_edu_max": (0.0, 4.0),
    "parent_edu_gap": (0.0, 4.0),
    "alc_total": (2.0, 10.0),
    "support_total": (0.0, 3.0),
    "failures_any": (0.0, 1.0),
    "grade_mean": (0.0, 20.0),
    "grade_trend": (-20.0, 20.0),
}


def feature_columns(feature_set: str) -> tuple[str, ...]:
    """Return the model inputs for a variant, in a stable order.

    `early` is the product default: no period grades. `with_grades` adds `G1`, `G2` and
    the two grade-derived columns. `G3` is never a feature.
    """
    if feature_set == FEATURE_SET_EARLY:
        return tuple(config.EARLY_FEATURES) + EARLY_DERIVED

    if feature_set == FEATURE_SET_WITH_GRADES:
        return tuple(config.WITH_GRADES_FEATURES) + EARLY_DERIVED + GRADE_DERIVED

    raise ValueError(
        f"Unknown feature set {feature_set!r}; expected "
        f"{FEATURE_SET_EARLY!r} or {FEATURE_SET_WITH_GRADES!r}."
    )


def build_features(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Add the derived columns for `feature_set` and return a new frame.

    Raw columns are left untouched so later stages can compare raw against derived.
    """
    # Called first so an unknown variant fails before any work is done.
    feature_columns(feature_set)

    out = frame.copy()

    out["parent_edu_mean"] = (out["Medu"] + out["Fedu"]) / 2.0
    out["parent_edu_max"] = out[["Medu", "Fedu"]].max(axis=1)
    out["parent_edu_gap"] = (out["Medu"] - out["Fedu"]).abs()

    out["alc_total"] = out["Dalc"] + out["Walc"]
    out["support_total"] = (
        (out[["schoolsup", "famsup", "paid"]] == "yes").sum(axis=1).astype("int64")
    )
    out["failures_any"] = (out["failures"] > 0).astype("int64")

    if feature_set == FEATURE_SET_WITH_GRADES:
        out["grade_mean"] = (out["G1"] + out["G2"]) / 2.0
        out["grade_trend"] = out["G2"] - out["G1"]

    return out


def feature_matrix(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Return the engineered frame restricted to the model inputs, in a fixed order.

    This is the single entry point for producing X. Selecting the columns here means the
    encoder and the trainer always receive exactly the documented features, with no
    accidental extras (such as `G3` or the generated ID) slipping into the matrix.
    """
    built = build_features(frame, feature_set)
    columns = list(feature_columns(feature_set))

    missing = [column for column in columns if column not in built.columns]
    if missing:
        raise load.DatasetSchemaError(
            f"Cannot build feature matrix for {feature_set!r}: missing {missing}."
        )

    return built.loc[:, columns].copy()


def validate_derived(
    frame: pd.DataFrame, feature_set: str, subject: str = "unknown"
) -> dict[str, Any]:
    """Check the derived columns exist, are null-free and stay in documented bounds.

    Returns observed min/max per column so the numbers are visible, not just asserted.
    """
    columns = [
        column for column in feature_columns(feature_set) if column in DERIVED_RANGES
    ]

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise load.DatasetSchemaError(
            f"{subject}: derived columns missing after build_features: {missing}. "
            "Build with build_features() before validating."
        )

    summary: dict[str, Any] = {}
    for column in columns:
        series = frame[column]
        if series.isna().any():
            raise load.DatasetSchemaError(
                f"{subject}: derived column {column!r} contains nulls."
            )

        low, high = DERIVED_RANGES[column]
        outside = (series < low) | (series > high)
        if outside.any():
            raise load.DatasetSchemaError(
                f"{subject}: derived column {column!r} has {int(outside.sum())} value(s)"
                f" outside [{low}, {high}], e.g. {series[outside].head(5).tolist()}."
            )

        summary[column] = (float(series.min()), float(series.max()))

    return summary


def summarize(frames: dict[str, pd.DataFrame]) -> str:
    """Plain-text summary of derived ranges and the correlations used to justify them."""
    lines = ["Derived feature summary", "=" * 74]

    for subject, frame in frames.items():
        lines += ["", f"{subject}  rows={len(frame)}  columns={frame.shape[1]}"]
        for feature_set in (FEATURE_SET_EARLY, FEATURE_SET_WITH_GRADES):
            built = build_features(frame, feature_set)
            ranges = validate_derived(built, feature_set, subject)
            shape = "  ".join(
                f"{column}=[{low:g},{high:g}]" for column, (low, high) in ranges.items()
            )
            lines.append(
                f"  {feature_set:<11} {len(feature_columns(feature_set))} features  "
                f"{shape}"
            )

        built = build_features(frame, FEATURE_SET_EARLY)
        dalc_walc = built[["Dalc", "Walc"]].corr().iloc[0, 1]
        medu_fedu = built[["Medu", "Fedu"]].corr().iloc[0, 1]
        alc_with_total = built["alc_total"].corr(built["Dalc"])
        lines += [
            f"  corr(Dalc, Walc)    = {dalc_walc:+.3f}  (why alc_total exists)",
            f"  corr(Medu, Fedu)    = {medu_fedu:+.3f}  (why parent_edu_* exist)",
            f"  corr(alc_total,Dalc)= {alc_with_total:+.3f}  (near 1.0 is expected)",
        ]

    lines += [
        "",
        "All derived features stayed inside their documented bounds.",
        "Feature sets are defined in ml/preprocessing/features.py; G3 is never a feature.",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        frames, _ = load.load_all()
    except load.DatasetSchemaError as error:
        print(f"Dataset validation failed.\n  {error}", file=sys.stderr)
        return 1

    print(summarize(frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
