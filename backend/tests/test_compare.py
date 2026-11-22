"""Tests for the cross-validation comparison stage.

The checks that matter here are the ones a wrong implementation would still pass a
"numbers came out" test on: fold membership, that folds are shared, that the test rows
are never used, and that selection is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from ml import config
from ml.data import load
from ml.preprocessing import features, split
from ml.training import compare, models

FEATURE_SET = features.FEATURE_SET_EARLY


@pytest.fixture(scope="module")
def training_set() -> tuple[object, np.ndarray, dict]:
    return compare.training_arrays("mat", FEATURE_SET)


def test_fold_assignment_covers_every_row_exactly_once() -> None:
    folds = compare.fold_assignment(100, folds=5)
    seen = np.concatenate(folds)

    assert len(folds) == 5
    assert len(seen) == 100
    assert len(np.unique(seen)) == 100
    assert [len(fold) for fold in folds] == [20] * 5


def test_fold_assignment_is_reproducible_and_seed_dependent() -> None:
    first = compare.fold_assignment(60, folds=3, seed=7)
    second = compare.fold_assignment(60, folds=3, seed=7)
    other = compare.fold_assignment(60, folds=3, seed=8)

    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_fold_assignment_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="Cannot make"):
        compare.fold_assignment(4, folds=5)


def test_metrics_match_hand_computation() -> None:
    y_true = np.array([10.0, 12.0, 14.0])
    y_pred = np.array([11.0, 12.0, 12.0])
    metrics = compare.regression_metrics(y_true, y_pred)

    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["mae"] == pytest.approx(1.0)


def test_training_arrays_use_the_saved_split_only(training_set) -> None:
    matrix, target, payload = training_set

    assert len(matrix) == len(payload["train_indices"])
    assert len(target) == len(payload["train_indices"])
    assert len(matrix) == payload["rows"] - len(payload["test_indices"])
    assert config.TARGET not in matrix.columns
    assert list(matrix.columns) == list(features.feature_columns(FEATURE_SET))


def test_evaluate_candidate_never_fits_on_held_out_rows(training_set) -> None:
    matrix, target, _ = training_set
    folds = compare.fold_assignment(len(matrix), folds=3)
    result = compare.evaluate_candidate(
        "LinearRegression", LinearRegression, FEATURE_SET, matrix, target, folds
    )

    assert [row["n_test"] for row in result["folds"]] == [len(fold) for fold in folds]
    for row, fold in zip(result["folds"], folds):
        assert row["n_train"] == len(matrix) - len(fold)
        assert row["n_train"] + row["n_test"] == len(matrix)
    assert set(result["folds"][0]) >= {"fold", "n_train", "n_test", "rmse", "mae", "r2"}


def test_deterministic_signal_is_learned_and_beats_baseline(training_set) -> None:
    """A target that is an exact linear function of a feature must be fitted near-exactly.

    This separates a real fit from a plausible-looking number: if the scaling or the
    column selection were misaligned with the target, the residual would not vanish.
    """
    matrix, _, _ = training_set
    designed = 10.0 + 0.5 * matrix["absences"].to_numpy(dtype=np.float64)
    folds = compare.fold_assignment(len(matrix), folds=3)

    linear = compare.evaluate_candidate(
        "LinearRegression", LinearRegression, FEATURE_SET, matrix, designed, folds
    )
    baseline = compare.evaluate_candidate(
        models.BASELINE, DummyRegressor, FEATURE_SET, matrix, designed, folds
    )

    assert linear["rmse_mean"] < 1e-6
    assert baseline["rmse_mean"] > 0.0


def test_selection_prefers_lower_rmse_then_mae_then_spread() -> None:
    def result(name: str, rmse: float, mae: float, std: float, baseline=False):
        return {
            "name": name,
            "baseline": baseline,
            "rmse_mean": rmse,
            "mae_mean": mae,
            "rmse_std": std,
        }

    only_mae_differs = [
        result("a", 2.0, 1.4, 0.2),
        result("b", 2.0, 1.3, 0.5),
    ]
    only_spread_differs = [
        result("a", 2.0, 1.3, 0.5),
        result("b", 2.0, 1.3, 0.1),
    ]

    assert compare.select(only_mae_differs)["name"] == "b"
    assert compare.select(only_spread_differs)["name"] == "b"


def test_selection_ignores_the_baseline_even_if_it_wins() -> None:
    results = [
        {"name": models.BASELINE, "baseline": True, "rmse_mean": 1.0,
         "mae_mean": 0.8, "rmse_std": 0.0},
        {"name": "Model", "baseline": False, "rmse_mean": 2.0,
         "mae_mean": 1.6, "rmse_std": 0.3},
    ]

    assert compare.select(results)["name"] == "Model"


def test_comparison_record_is_complete_and_ignores_test_rows() -> None:
    payload = compare.compare("mat", FEATURE_SET, folds=3)

    assert payload["selected"] in models.candidates()
    assert payload["rows_used"] == payload["rows_total"] - payload["rows_held_out"]
    assert "DummyMean" in [result["name"] for result in payload["results"]]
    assert payload["selected_metrics"]["rmse"] < payload["baseline_rmse"]
    assert len(payload["fold_sizes"]) == 3
    assert payload["dataset_sha256"] == split.dataset_sha256("mat")
