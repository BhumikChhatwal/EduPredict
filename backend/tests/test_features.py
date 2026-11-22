"""Unit tests for the derived features.

Each feature is checked on a hand-computed frame, so a formula regression fails here
rather than quietly changing model behaviour later.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.data import load
from ml.preprocessing import features


def sample_frame() -> pd.DataFrame:
    """Two rows with known, deliberately asymmetric parent and support values."""
    return pd.DataFrame(
        {
            "Medu": [4, 1],
            "Fedu": [2, 3],
            "Dalc": [1, 3],
            "Walc": [2, 5],
            "schoolsup": ["yes", "no"],
            "famsup": ["no", "yes"],
            "paid": ["no", "no"],
            "failures": [0, 2],
            "G1": [10, 8],
            "G2": [12, 6],
        }
    )


def test_parent_education_features() -> None:
    result = features.build_features(sample_frame(), features.FEATURE_SET_EARLY)

    assert result["parent_edu_mean"].tolist() == [3.0, 2.0]
    assert result["parent_edu_max"].tolist() == [4, 3]
    assert result["parent_edu_gap"].tolist() == [2, 2]


def test_alc_total_sums_weekday_and_weekend() -> None:
    result = features.build_features(sample_frame(), features.FEATURE_SET_EARLY)

    assert result["alc_total"].tolist() == [3, 8]


def test_support_total_counts_yes_answers() -> None:
    result = features.build_features(sample_frame(), features.FEATURE_SET_EARLY)

    assert result["support_total"].tolist() == [1, 1]
    assert result["support_total"].between(0, 3).all()


def test_failures_any_is_a_threshold_flag() -> None:
    result = features.build_features(sample_frame(), features.FEATURE_SET_EARLY)

    assert result["failures_any"].tolist() == [0, 1]
    assert str(result["failures_any"].dtype) == "int64"


def test_grade_features_only_exist_in_with_grades() -> None:
    early = features.build_features(sample_frame(), features.FEATURE_SET_EARLY)
    with_grades = features.build_features(
        sample_frame(), features.FEATURE_SET_WITH_GRADES
    )

    assert "grade_mean" not in early.columns
    assert "grade_trend" not in early.columns
    assert with_grades["grade_mean"].tolist() == [11.0, 7.0]
    assert with_grades["grade_trend"].tolist() == [2, -2]


def test_features_ignore_the_target() -> None:
    frame = sample_frame()
    frame["G3"] = [9, 19]
    changed = frame.copy()
    changed["G3"] = [20, 0]

    before = features.build_features(frame, features.FEATURE_SET_WITH_GRADES)
    after = features.build_features(changed, features.FEATURE_SET_WITH_GRADES)

    pd.testing.assert_frame_equal(
        before.drop(columns=["G3"]), after.drop(columns=["G3"])
    )


def test_feature_columns_never_include_the_target() -> None:
    for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES):
        columns = features.feature_columns(feature_set)
        assert "G3" not in columns
        assert len(columns) == len(set(columns))  # no duplicates

    assert "G1" not in features.feature_columns(features.FEATURE_SET_EARLY)
    assert "G1" in features.feature_columns(features.FEATURE_SET_WITH_GRADES)


def test_build_does_not_mutate_the_input_or_drop_rows() -> None:
    frame = sample_frame()
    original = frame.copy()

    result = features.build_features(frame, features.FEATURE_SET_EARLY)

    pd.testing.assert_frame_equal(frame, original)
    assert len(result) == len(frame)
    assert result.index.equals(frame.index)


def test_unknown_feature_set_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown feature set"):
        features.build_features(sample_frame(), "everything")

    with pytest.raises(ValueError, match="Unknown feature set"):
        features.feature_columns("everything")


def test_validate_derived_accepts_the_sample() -> None:
    built = features.build_features(sample_frame(), features.FEATURE_SET_WITH_GRADES)
    summary = features.validate_derived(
        built, features.FEATURE_SET_WITH_GRADES, "test"
    )

    assert set(summary) == set(features.DERIVED_RANGES)
    assert summary["support_total"] == (1.0, 1.0)


def test_validate_derived_catches_out_of_range_values() -> None:
    built = features.build_features(sample_frame(), features.FEATURE_SET_EARLY)
    built.loc[0, "support_total"] = 7

    with pytest.raises(load.DatasetSchemaError, match="outside"):
        features.validate_derived(built, features.FEATURE_SET_EARLY, "test")


def test_validate_derived_catches_missing_columns() -> None:
    with pytest.raises(load.DatasetSchemaError, match="missing"):
        features.validate_derived(
            sample_frame(), features.FEATURE_SET_EARLY, "test"
        )
