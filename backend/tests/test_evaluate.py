"""Tests for the final test-set evaluation.

The properties worth locking down: bands follow the documented cut-offs, clipping is
visible, the band table keeps the sign of the bias, and the reported metrics really are
the metrics of the predictions the record stores (so a number in the README can be
re-derived from the JSON).
"""

from __future__ import annotations

import numpy as np
import pytest

from ml import config
from ml.evaluation import evaluate as evaluation
from ml.preprocessing import features, split
from ml.training import compare


@pytest.fixture(scope="module")
def record() -> dict:
    return evaluation.evaluate("mat", features.FEATURE_SET_EARLY)


def test_risk_band_index_uses_documented_cut_offs() -> None:
    # Band edges are CBSE marks (33 / 51 / 71), so the native grades are their fifths.
    grades = np.array([0.0, 6.59, 6.6, 10.19, 10.2, 14.19, 14.2, 20.0])
    assert evaluation.risk_band_index(grades).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert len(config.RISK_BAND_LABELS) == 4
    assert list(config.RISK_BAND_EDGES) == [0, 33, 51, 71, 101]


def test_clipping_only_touches_out_of_range_predictions() -> None:
    predicted = np.array([-1.5, 0.0, 12.3, 20.0, 23.4])
    clipped, changed = evaluation.clip_predictions(predicted)

    assert changed == 2
    assert clipped.tolist() == [0.0, 0.0, 12.3, 20.0, 20.0]


def test_band_table_reports_the_sign_of_the_bias() -> None:
    # Native 5 and 15 are 25 and 75 CBSE marks: critical and strong respectively.
    actual = np.array([5.0, 5.0, 15.0, 15.0])
    predicted = np.array([6.0, 6.0, 14.0, 14.0])
    table = evaluation.band_table(actual, predicted)

    assert table["critical"]["n"] == 2
    assert table["critical"]["mean_residual"] == pytest.approx(1.0)  # over-predicted
    assert table["strong"]["mean_residual"] == pytest.approx(-1.0)  # under-predicted
    assert table["critical"]["mae"] == pytest.approx(1.0)
    assert table["critical"]["rmse"] == pytest.approx(1.0)


def test_band_table_keeps_an_empty_band_visible() -> None:
    table = evaluation.band_table(np.array([12.0, 13.0]), np.array([12.0, 12.0]))

    assert table["critical"]["n"] == 0
    assert table["critical"]["mae"] is None
    assert set(table) == set(config.RISK_BAND_LABELS)


def test_record_covers_exactly_the_saved_test_rows(record: dict) -> None:
    payload = split.load_split("mat")
    predicted_rows = {row["row_index"] for row in record["predictions"]}

    assert record["rows_test"] == len(payload["test_indices"])
    assert predicted_rows == set(payload["test_indices"])
    assert not predicted_rows & set(payload["train_indices"])
    assert record["rows_test"] + record["rows_train"] == payload["rows"]


def test_record_metrics_reproduce_from_the_stored_predictions(record: dict) -> None:
    """The headline numbers must be the metrics of the rows the JSON actually stores."""
    actual = np.array([row["actual"] for row in record["predictions"]])
    predicted = np.array([row["predicted"] for row in record["predictions"]])
    recomputed = compare.regression_metrics(actual, predicted)

    for key in ("rmse", "mae", "r2"):
        assert record["metrics"][key] == pytest.approx(recomputed[key], abs=1e-9)

    residuals = predicted - actual
    assert record["metrics"]["mean_residual"] == pytest.approx(residuals.mean(), abs=1e-12)
    assert record["metrics"]["within_2_marks"] == pytest.approx(
        float((np.abs(residuals) <= 2.0).mean()), abs=1e-12
    )


def test_record_carries_the_chosen_configuration_and_cv_baseline(record: dict) -> None:
    assert record["family"] in ("LinearRegression", "DecisionTree", "RandomForest",
                                "GradientBoosting")
    assert record["chosen"] in {"tuned", "untuned"}
    assert record["cv_rmse"] > 0.0
    assert record["rows_test"] > 0
    assert record["prediction_range"][0] >= 0.0
    assert record["prediction_range"][1] <= 20.0


def test_figures_are_written(record: dict, tmp_path) -> None:
    scatter = evaluation.plot_predictions(record, tmp_path / "scatter.png")
    residuals = evaluation.plot_residuals(record, tmp_path / "residuals.png")

    for path in (scatter, residuals):
        assert path.exists()
        assert path.stat().st_size > 5_000  # a blank canvas would be far smaller
