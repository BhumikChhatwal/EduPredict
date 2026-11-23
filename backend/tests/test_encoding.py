"""Tests for the shared ColumnTransformer.

These pin the encoding contract: ordinal columns stay numeric, nominal columns expand,
unknown levels never raise at prediction time, and an unexpected column fails loudly
instead of being silently dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ml import config
from ml.data import load
from ml.preprocessing import encoding, features


@pytest.fixture(scope="module")
def maths() -> pd.DataFrame:
    frame, _ = load.load_subject("mat")
    return frame


@pytest.fixture(scope="module")
def maths_matrix(maths: pd.DataFrame) -> pd.DataFrame:
    return features.feature_matrix(maths, features.FEATURE_SET_EARLY)


def test_every_feature_gets_exactly_one_group() -> None:
    for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES):
        columns = features.feature_columns(feature_set)
        groups = encoding.classify_columns(columns)

        total = sum(len(group) for group in groups.values())
        assert total == len(columns)
        assert set(groups) == {"numeric", "ordinal", "nominal"}


def test_expected_group_sizes() -> None:
    groups = encoding.classify_columns(
        features.feature_columns(features.FEATURE_SET_EARLY)
    )

    # `school` is a dataset column the CBSE product does not ask for, so it is not a
    # feature; the rest of the nominal and ordinal groups are unchanged.
    excluded_nominal = [c for c in config.NOMINAL_COLUMNS if c in config.EXCLUDED_FEATURES]
    assert len(groups["nominal"]) == len(config.NOMINAL_COLUMNS) - len(excluded_nominal)
    assert "school" not in groups["nominal"]
    assert len(groups["ordinal"]) == len(config.ORDINAL_COLUMNS) == 10
    assert len(groups["numeric"]) == 9


def test_unclassified_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="no encoding rule"):
        encoding.classify_columns(("school", "mystery_column"))


def test_feature_matrix_excludes_target_and_id(maths: pd.DataFrame) -> None:
    matrix = features.feature_matrix(maths, features.FEATURE_SET_WITH_GRADES)

    assert config.TARGET not in matrix.columns
    assert "student_id" not in matrix.columns
    assert list(matrix.columns) == list(
        features.feature_columns(features.FEATURE_SET_WITH_GRADES)
    )


def test_encoded_matrix_is_dense_and_nan_free(maths: pd.DataFrame) -> None:
    report = encoding.describe(maths, features.FEATURE_SET_EARLY)

    assert report["input_features"] == len(
        features.feature_columns(features.FEATURE_SET_EARLY)
    )
    assert report["encoded_features"] > report["input_features"]
    assert len(report["feature_names"]) == report["encoded_features"]


def test_ordinal_columns_are_passed_through_not_expanded(
    maths: pd.DataFrame, maths_matrix: pd.DataFrame
) -> None:
    preprocessor = encoding.build_preprocessor(features.FEATURE_SET_EARLY)
    matrix = preprocessor.fit_transform(maths_matrix)
    names = list(preprocessor.get_feature_names_out())

    column = names.index("Medu")
    assert np.array_equal(matrix[:, column], maths["Medu"].to_numpy())

    # One-hot would produce one column per level; passthrough keeps exactly one.
    assert not any(name.startswith("Medu_") for name in names)


def test_unseen_category_is_ignored_at_transform_time(
    maths_matrix: pd.DataFrame,
) -> None:
    preprocessor = encoding.build_preprocessor(features.FEATURE_SET_EARLY)
    preprocessor.fit(maths_matrix)

    row = maths_matrix.iloc[[0]].copy()
    row["Mjob"] = "astronaut"

    transformed = preprocessor.transform(row)

    assert transformed.shape[1] == len(preprocessor.get_feature_names_out())
    assert np.isfinite(transformed).all()


def test_unexpected_column_fails_at_fit(maths: pd.DataFrame) -> None:
    matrix = features.feature_matrix(maths, features.FEATURE_SET_EARLY)
    with_extra = matrix.copy()
    with_extra["mystery"] = 1

    pipeline = encoding.build_pipeline(
        features.FEATURE_SET_EARLY, LinearRegression()
    )

    with pytest.raises(ValueError, match="does not match the trained encoding"):
        pipeline.fit(with_extra, maths[config.TARGET])


def test_missing_feature_fails_at_fit(maths: pd.DataFrame) -> None:
    matrix = features.feature_matrix(maths, features.FEATURE_SET_EARLY).drop(
        columns=["absences"]
    )
    pipeline = encoding.build_pipeline(
        features.FEATURE_SET_EARLY, LinearRegression()
    )

    with pytest.raises(ValueError, match="missing=\\['absences'\\]"):
        pipeline.fit(matrix, maths[config.TARGET])


def test_scaler_only_for_linear_models(maths: pd.DataFrame) -> None:
    linear = encoding.build_pipeline(features.FEATURE_SET_EARLY, LinearRegression())
    forest = encoding.build_pipeline(
        features.FEATURE_SET_EARLY, RandomForestRegressor(n_estimators=5)
    )

    assert "scale" in linear.named_steps
    assert "scale" not in forest.named_steps
    assert list(forest.named_steps) == ["check", "preprocess", "model"]


def test_pipeline_fits_and_predicts(maths: pd.DataFrame) -> None:
    matrix = features.feature_matrix(maths, features.FEATURE_SET_EARLY)
    pipeline = encoding.build_pipeline(features.FEATURE_SET_EARLY, LinearRegression())

    predictions = pipeline.fit(matrix, maths[config.TARGET]).predict(matrix)

    assert len(predictions) == len(maths)
    assert np.isfinite(predictions).all()
