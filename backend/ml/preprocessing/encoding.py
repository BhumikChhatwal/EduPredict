"""Shared preprocessing transformer for both feature sets and both subjects.

One `ColumnTransformer` definition is used everywhere, so a column cannot be encoded
one way during training and another way at prediction time. The transformer is fitted
inside a scikit-learn `Pipeline`, which means it is fitted on training rows only and
travels with the model in the saved artifact.

Encoding scheme (from docs/architecture.md §4.2):

* Nominal attributes (17) - one-hot encoded with `handle_unknown="ignore"`, so an
  unseen level at prediction time becomes all-zeros instead of raising.
* Ordinal attributes (10) - passed through as ordered integers. One-hot would discard
  the ordering that `studytime` or `Medu` actually carries.
* Numeric and derived attributes (9) - passed through; scaling is added only for
  linear models, because it changes nothing for trees.

`ColumnTransformer` has no `remainder="error"` option, so unexpected columns are caught
by a guard step (`_require_feature_columns`) instead of by the remainder. The guard is a
module-level function so the fitted pipeline still pickles cleanly for the API.

Run the summary with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.preprocessing.encoding
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml import config
from ml.data import load
from ml.preprocessing import features

# Estimators whose coefficients are only comparable, and whose regularisation only
# behaves, on comparable scales. Everything else here is a tree, which is invariant to
# any monotone rescaling, so scaling it would add cost without information.
SCALED_MODELS = ("LinearRegression", "Ridge", "Lasso", "ElasticNet")


def classify_columns(columns: tuple[str, ...]) -> dict[str, list[str]]:
    """Split feature columns into numeric / ordinal / nominal, or fail loudly.

    Priority is nominal, then ordinal, then numeric: the ten ordinal attributes are also
    numeric, so they must not be claimed twice. Raising on an unclassified column keeps
    this module and `features.py` in sync - add a feature without deciding how to encode
    it and the tests fail immediately.
    """
    numeric_kinds = set(config.NUMERIC_COLUMNS) | set(features.DERIVED_RANGES)
    ordinal = [column for column in columns if column in config.ORDINAL_COLUMNS]
    nominal = [column for column in columns if column in config.NOMINAL_COLUMNS]

    already_grouped = set(ordinal) | set(nominal)
    numeric = [
        column
        for column in columns
        if column in numeric_kinds and column not in already_grouped
    ]

    unclassified = sorted(set(columns) - set(ordinal) - set(nominal) - set(numeric))
    if unclassified:
        raise ValueError(
            f"Feature columns with no encoding rule: {unclassified}. Add them to "
            "config.ORDINAL_COLUMNS, config.NOMINAL_COLUMNS or features.DERIVED_RANGES."
        )

    grouped = ordinal + nominal + numeric
    if len(grouped) != len(columns):
        raise ValueError("A column was classified into more than one group.")

    return {"numeric": numeric, "ordinal": ordinal, "nominal": nominal}


def _require_feature_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Guard step: the input must carry exactly the model's features, nothing else.

    Without this, `ColumnTransformer` would silently drop extra columns - a feature-set
    mismatch at serving time would then produce confident nonsense instead of an error.
    """
    expected = set(columns)
    found = set(frame.columns)
    if found != expected:
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        raise ValueError(
            f"Feature matrix does not match the trained encoding: "
            f"missing={missing} unexpected={unexpected}."
        )
    return frame


def build_preprocessor(feature_set: str) -> ColumnTransformer:
    """Return the transformer for a feature set, fitted on demand."""
    columns = features.feature_columns(feature_set)
    groups = classify_columns(columns)

    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", groups["numeric"]),
            ("ordinal", "passthrough", groups["ordinal"]),
            (
                "nominal",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
                groups["nominal"],
            ),
        ],
        # Safe because the guard step has already rejected unexpected columns.
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(feature_set: str, model: Any) -> Pipeline:
    """Assemble guard + preprocessor (+ optional scaler) + model.

    The result takes an engineered feature matrix (see `features.feature_matrix`) and
    can be fitted, predicted and pickled as one object.
    """
    columns = features.feature_columns(feature_set)

    steps: list[tuple[str, Any]] = [
        (
            "check",
            # kw_args keeps the fitted object picklable, unlike a closure.
            FunctionTransformer(
                _require_feature_columns,
                kw_args={"columns": columns},
                validate=False,
                feature_names_out="one-to-one",
            ),
        ),
        ("preprocess", build_preprocessor(feature_set)),
    ]

    if type(model).__name__ in SCALED_MODELS:
        steps.append(("scale", StandardScaler()))

    steps.append(("model", model))

    return Pipeline(steps=steps)


def describe(frame: pd.DataFrame, feature_set: str) -> dict[str, Any]:
    """Fit the transformer on an engineered matrix and report the resulting shape.

    Reporting the encoded width makes the cost of one-hot encoding visible: 17 nominal
    columns expand into many more.
    """
    matrix_input = features.feature_matrix(frame, feature_set)
    groups = classify_columns(features.feature_columns(feature_set))

    preprocessor = build_preprocessor(feature_set)
    matrix = preprocessor.fit_transform(matrix_input)
    names = preprocessor.get_feature_names_out()

    if np.isnan(matrix).any():
        raise load.DatasetSchemaError(f"{feature_set}: encoded matrix contains NaN.")

    encoded = int(matrix.shape[1])
    return {
        "input_features": len(groups["numeric"]) + len(groups["ordinal"]) + len(groups["nominal"]),
        "encoded_features": encoded,
        "numeric": len(groups["numeric"]),
        "ordinal": len(groups["ordinal"]),
        "nominal_raw": len(groups["nominal"]),
        "nominal_expanded": encoded - len(groups["numeric"]) - len(groups["ordinal"]),
        "feature_names": list(names),
    }


def summarize(frames: dict[str, pd.DataFrame]) -> str:
    lines = ["Encoding summary", "=" * 74]

    for subject, frame in frames.items():
        lines += ["", f"{subject}  rows={len(frame)}"]
        for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES):
            report = describe(frame, feature_set)
            lines.append(
                f"  {feature_set:<11} in={report['input_features']:>2}  "
                f"out={report['encoded_features']:>2}  "
                f"(numeric={report['numeric']}, ordinal={report['ordinal']}, "
                f"nominal {report['nominal_raw']}->{report['nominal_expanded']})"
            )

    example = describe(frames[config.SUBJECTS[0]], features.FEATURE_SET_EARLY)
    lines += [
        "",
        f"Encoded feature names ({len(example['feature_names'])}):",
        "  " + ", ".join(example["feature_names"]),
        "",
        "Nominal columns are one-hot encoded with handle_unknown='ignore'.",
        "Ordinal and numeric columns pass through unchanged; the matrix is NaN-free.",
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
