"""Tests for the tuning stage.

These check the properties that make a tuned number comparable and reproducible: the
search runs on the saved folds, the grids address real pipeline parameters, a known
optimum is actually found, and a comparison against a differently-partitioned run is
refused rather than silently mixed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from ml import config
from ml.preprocessing import features
from ml.training import compare, models, tune

FEATURE_SET = features.FEATURE_SET_EARLY


@pytest.fixture(scope="module")
def training_set() -> tuple[object, np.ndarray]:
    matrix, target, _ = compare.training_arrays("mat", FEATURE_SET)
    return matrix, target


def test_search_splits_are_exact_complements() -> None:
    folds = compare.fold_assignment(50, folds=5)
    pairs = tune.search_splits(folds, 50)

    assert len(pairs) == 5
    for (train, test), fold in zip(pairs, folds):
        assert np.array_equal(test, fold)
        assert len(train) + len(test) == 50
        assert not set(train) & set(test)
        assert len(np.union1d(train, test)) == 50


@pytest.mark.parametrize("family", list(models.candidates()))
def test_every_family_has_a_small_grid_of_real_pipeline_parameters(family: str) -> None:
    grid = tune.grid_for(family)
    params = models.candidates()[family]().get_params()

    assert grid, f"{family} has an empty grid"
    assert tune.combo_count(grid) <= 24, "grids are meant to be small and explicit"
    for key, values in grid.items():
        assert key.startswith(f"{tune.MODEL_STEP}__")
        assert key.removeprefix(f"{tune.MODEL_STEP}__") in params, f"{key} is not a real parameter"
        assert values, f"{key} has no values to try"


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="No tuning grid"):
        tune.grid_for("SupportVectorMachine")


def test_search_finds_an_exact_linear_solution(training_set) -> None:
    """A target that is an exact linear function must be solved, not approximated.

    This is the check that the grid, the fold wiring and the estimator are all aimed at
    the same object: an intercept/feature misalignment would leave a visible residual.
    """
    matrix, _ = training_set
    designed = 10.0 + 0.5 * matrix["absences"].to_numpy(dtype=np.float64)
    folds = compare.fold_assignment(len(matrix), folds=3)

    search = tune.run_search(
        "LinearRegression",
        FEATURE_SET,
        matrix,
        designed,
        folds,
        grid={"model__fit_intercept": [True]},
    )

    assert search["best_params"] == {"fit_intercept": True}
    assert search["best_rmse_mean"] < 1e-6
    assert search["combos"] == 1


def test_search_score_agrees_with_plain_cross_validation(training_set) -> None:
    """The grid search and `compare.evaluate_candidate` must score the same folds alike.

    If they disagreed, the untuned and tuned rows of the report would not be comparable.
    """
    matrix, target = training_set
    folds = compare.fold_assignment(len(matrix), folds=3)
    grid = {"model__fit_intercept": [True, False]}

    search = tune.run_search(
        "LinearRegression", FEATURE_SET, matrix, target, folds, grid=grid
    )
    plain = compare.evaluate_candidate(
        "LinearRegression",
        tune.configured_builder("LinearRegression", search["best_params"]),
        FEATURE_SET,
        matrix,
        target,
        folds,
    )

    assert search["best_rmse_mean"] == pytest.approx(plain["rmse_mean"], abs=1e-9)
    assert plain["mae_mean"] > 0.0


def test_search_is_reproducible(training_set) -> None:
    matrix, target = training_set
    folds = compare.fold_assignment(len(matrix), folds=3)
    grid = {"model__fit_intercept": [True, False]}

    first = tune.run_search("LinearRegression", FEATURE_SET, matrix, target, folds, grid=grid)
    second = tune.run_search("LinearRegression", FEATURE_SET, matrix, target, folds, grid=grid)

    assert first["best_params"] == second["best_params"]
    assert first["best_rmse_mean"] == second["best_rmse_mean"]
    assert [row["params"] for row in first["combinations"]] == [
        row["params"] for row in second["combinations"]
    ]


def test_search_record_ranks_every_configuration(training_set) -> None:
    """Full-record shape, checked on a deliberately tiny forest grid for speed."""
    matrix, target = training_set
    folds = compare.fold_assignment(len(matrix), folds=3)
    grid = {
        "model__n_estimators": [50],
        "model__max_depth": [None, 12],
        "model__min_samples_leaf": [1, 2],
    }

    search = tune.run_search(
        "RandomForest", FEATURE_SET, matrix, target, folds, grid=grid
    )

    assert len(search["combinations"]) == search["combos"] == math.prod(
        len(values) for values in search["grid"].values()
    )
    scores = [row["rmse_mean"] for row in search["combinations"]]
    assert scores == sorted(scores)
    assert search["combinations"][0]["rmse_mean"] == pytest.approx(search["best_rmse_mean"])
    assert set(search["best_params"]) <= {
        key.removeprefix(f"{tune.MODEL_STEP}__") for key in search["grid"]
    }


def test_tuning_refuses_fold_counts_that_cannot_be_compared() -> None:
    """A 3-fold search cannot be compared with a 5-fold record, so it must not run."""
    with pytest.raises(ValueError, match="folds"):
        tune.tune("mat", FEATURE_SET, folds=3)


def test_tuning_record_keeps_untuned_numbers_and_a_decision() -> None:
    payload = tune.tune("mat", FEATURE_SET)

    assert payload["chosen"] in {"tuned", "untuned"}
    # The family is the one that won the comparison, not one chosen by hand.
    assert payload["family"] == payload["untuned"]["model"]
    assert payload["chosen_params"] == (
        payload["best_params"] if payload["chosen"] == "tuned" else {}
    )
    # The reported number is the cross-validated score, never the training fit.
    assert payload["tuned_metrics"]["rmse"] >= payload["best_train_rmse"]
    assert payload["folds"] == config.CV_FOLDS
    assert payload["dataset_sha256"] == compare.split.dataset_sha256("mat")
