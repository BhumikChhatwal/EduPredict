"""Hyperparameter tuning of the family that won the comparison.

Rules this module enforces, because they are what make a tuned number trustworthy:

* **Only the selected family is searched.** The family comes from the saved comparison
  record, not from a command-line argument, so it is never tuned towards a family that
  lost and then reported as "the model".
* **The same folds.** `GridSearchCV` is given the exact fold partitions the untuned
  comparison used, as explicit `(train, test)` index pairs. Score differences are then
  a property of the configuration, not of a different random partition.
* **Small, written-down grids.** Each axis is listed with the values actually tried, so
  a low score can be traced to a specific setting. A large random or exhaustive search
  on 316 training rows finds noise: with five folds, differences below roughly one
  fold's standard error are not resolvable, and searching hundreds of configurations
  guarantees one will look good by luck.
* **Tuning has to earn its place.** If the best configuration does not beat the untuned
  family on the same folds, the record says `chosen: "untuned"` and the untuned
  configuration is what later stages serialise.

What the parameters do (the "why" for every grid below):

`RandomForest`

* `n_estimators` - how many trees are averaged. Averaging 300 noisy trees lowers the
  variance of the prediction; going to 500 usually buys a little more stability and
  costs time. It does not make the forest overfit more, because each tree still sees a
  bootstrap sample and a random feature subset.
* `max_depth` - the deepest chain of splits a tree may make. Depth controls the finest
  interaction the tree can express; unlimited depth lets a tree isolate individual rows
  and memorise their grade. Limiting it is the main brake on that.
* `min_samples_leaf` - the smallest number of rows allowed in a leaf. A leaf holding one
  or two students is a memorised exception, not a pattern; raising this smooths the
  prediction and is the second brake on variance.
* `max_features` is left at its default (`"sqrt"`): that is the setting responsible for
  making the trees differ from each other, which is why averaging them helps at all.
  `min_samples_split` is left out deliberately - it controls the same thing as
  `min_samples_leaf` from the other side, and including both would double the grid
  without adding a distinct hypothesis.

`LinearRegression`

* `fit_intercept` - whether the model is allowed to shift its line away from the origin.
  With standardised features and grades far from zero, an intercept is essentially
  always needed; the grid still checks the claim rather than asserting it. Beyond that
  there is nothing to tune in ordinary least squares: it has a closed-form solution, so
  the coefficients are determined, not searched. That is why the tuned and untuned
  numbers for this family are expected to be identical.

The other two families get small grids only so this module cannot fail if one of them
ever wins a comparison; their grids are not the result of a search.

Run it with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.training.compare    # once, to pick the family
    ./.venv/Scripts/python.exe -m ml.training.tune
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from ml import config
from ml.preprocessing import encoding, features
from ml.training import compare, models

TUNED_SUFFIX = "__tuned"

# Pipeline step name the estimated model sits in; GridSearchCV addresses parameters as
# "<step>__<parameter>".
MODEL_STEP = "model"

TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "RandomForest": {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "LinearRegression": {
        "model__fit_intercept": [True, False],
    },
    "DecisionTree": {
        "model__max_depth": [None, 8, 12],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    },
}


def grid_for(family: str) -> dict[str, list[Any]]:
    """Return the documented grid for a family, or fail with the families that exist."""
    if family not in TUNING_GRIDS:
        raise ValueError(
            f"No tuning grid for {family!r}. Known families: {sorted(TUNING_GRIDS)}."
        )
    return TUNING_GRIDS[family]


def combo_count(grid: dict[str, list[Any]]) -> int:
    """How many configurations a grid contains."""
    return math.prod(len(values) for values in grid.values())


def search_splits(folds: list[np.ndarray], n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Turn held-out fold positions into explicit `(train, test)` pairs for sklearn.

    Passing index pairs instead of a splitter object is what pins the search to the
    comparison's folds: `GridSearchCV` never generates a partition of its own.
    """
    positions = np.arange(n_rows)
    pairs = []
    for test_positions in folds:
        train_positions = np.setdiff1d(positions, test_positions, assume_unique=False)
        missing = len(positions) - len(train_positions) - len(test_positions)
        if missing != 0:
            raise ValueError("Fold positions do not cover every row exactly once.")
        pairs.append((train_positions, test_positions))
    return pairs


def configured_builder(family: str, params: dict[str, Any]) -> Callable[[], Any]:
    """Return a constructor producing the family with `params` applied.

    `params` uses bare estimator parameter names (`n_estimators`), not pipeline names.
    """
    def build() -> Any:
        estimator = models.candidates()[family]()
        if params:
            estimator.set_params(**params)
        # The search (or the outer CV loop) parallelises; one worker per tree would only
        # oversubscribe the CPU.
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=1)
        return estimator

    return build


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip the pipeline step prefix so the saved config reads like an estimator."""
    prefix = f"{MODEL_STEP}__"
    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in params.items()
    }


def run_search(
    family: str,
    feature_set: str,
    matrix: pd.DataFrame,
    target: np.ndarray,
    folds: list[np.ndarray],
    *,
    grid: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """Grid-search a family on the given folds and return the search record."""
    param_grid = grid if grid is not None else grid_for(family)
    builder = configured_builder(family, {})
    pipeline = encoding.build_pipeline(feature_set, builder())

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        # Negative because sklearn maximises; the sign is undone below.
        scoring="neg_root_mean_squared_error",
        cv=search_splits(folds, len(matrix)),
        refit=True,
        # Deliberately serial: the data is a few hundred rows, so process start-up on
        # this platform would cost more than the fits themselves.
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(matrix, target)

    results = search.cv_results_
    combinations = []
    for index, params in enumerate(results["params"]):
        combinations.append(
            {
                "params": _clean_params(params),
                "rmse_mean": float(-results["mean_test_score"][index]),
                "rmse_std": float(results["std_test_score"][index]),
                # Reported to make the overfit gap visible rather than inferred.
                "train_rmse": float(-results["mean_train_score"][index]),
            }
        )
    combinations.sort(key=lambda row: (row["rmse_mean"], row["rmse_std"]))

    best_index = int(search.best_index_)
    return {
        "family": family,
        "grid": param_grid,
        "combos": combo_count(param_grid),
        "best_params": _clean_params(search.best_params_),
        "best_pipeline_params": dict(search.best_params_),
        "best_rmse_mean": float(-results["mean_test_score"][best_index]),
        "best_rmse_std": float(results["std_test_score"][best_index]),
        "best_train_rmse": float(-results["mean_train_score"][best_index]),
        "combinations": combinations,
    }


def tune(subject: str, feature_set: str, *, folds: int = config.CV_FOLDS) -> dict[str, Any]:
    """Tune the family named in the saved comparison and return the tuning record."""
    untuned = compare.load_results(subject, feature_set)
    family = untuned["selected"]
    matrix, target, payload = compare.training_arrays(subject, feature_set)
    assignments = compare.fold_assignment(len(matrix), folds=folds)

    # Comparability guards: a tuned score may only be compared with the untuned one if
    # both came from the same rows, the same number of folds and the same partitions.
    if untuned["folds"] != folds:
        raise ValueError(
            f"{subject}/{feature_set}: saved comparison used {untuned['folds']} folds, "
            f"tuning requested {folds}; rerun `ml.training.compare` with the same value."
        )
    if untuned["fold_sizes"] != [int(len(fold)) for fold in assignments]:
        raise ValueError(
            f"{subject}/{feature_set}: fold sizes differ from the saved comparison; the "
            "training rows changed. Rerun `ml.training.compare` before tuning."
        )
    if untuned["dataset_sha256"] != payload["dataset_sha256"]:
        raise ValueError(
            f"{subject}: the dataset on disk differs from the one the comparison used."
        )

    search = run_search(family, feature_set, matrix, target, assignments)

    tuned_result = compare.evaluate_candidate(
        f"{family} (tuned)",
        configured_builder(family, search["best_params"]),
        feature_set,
        matrix,
        target,
        assignments,
    )

    untuned_rmse = float(untuned["selected_metrics"]["rmse"])
    tuned_rmse = tuned_result["rmse_mean"]
    decimals = compare.SELECTION_DECIMALS
    # Ties keep the untuned configuration: an equal score does not justify extra
    # hyperparameters in the saved artifact.
    use_tuned = round(tuned_rmse, decimals) < round(untuned_rmse, decimals)

    return {
        "subject": subject,
        "feature_set": feature_set,
        "family": family,
        "rows_used": int(len(matrix)),
        "folds": folds,
        "fold_seed": config.SEED,
        "fold_sizes": [int(len(fold)) for fold in assignments],
        "dataset_sha256": payload["dataset_sha256"],
        "grid": search["grid"],
        "combos": search["combos"],
        "search": search["combinations"],
        "best_params": search["best_params"],
        "best_train_rmse": search["best_train_rmse"],
        "tuned_metrics": {
            "rmse": tuned_rmse,
            "rmse_std": tuned_result["rmse_std"],
            "mae": tuned_result["mae_mean"],
            "r2": tuned_result["r2_mean"],
        },
        "untuned": {
            "model": untuned["selected"],
            "metrics": untuned["selected_metrics"],
        },
        "baseline_rmse": untuned["baseline_rmse"],
        "chosen": "tuned" if use_tuned else "untuned",
        "chosen_params": search["best_params"] if use_tuned else {},
        "improvement_over_untuned": untuned_rmse - tuned_rmse,
        "improvement_over_baseline": float(untuned["baseline_rmse"]) - tuned_rmse,
    }


def tuned_path(subject: str, feature_set: str) -> Path:
    return compare.CV_DIR / f"{subject}__{feature_set}{TUNED_SUFFIX}.json"


def save_record(payload: dict[str, Any]) -> Path:
    compare.CV_DIR.mkdir(parents=True, exist_ok=True)
    path = tuned_path(payload["subject"], payload["feature_set"])
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def format_report(payloads: list[dict[str, Any]]) -> str:
    lines = [
        "Hyperparameter tuning - selected family only, same folds as the comparison",
        "=" * 74,
    ]

    for payload in payloads:
        untuned = payload["untuned"]["metrics"]
        tuned = payload["tuned_metrics"]
        lines += [
            "",
            f"{payload['subject']}  {payload['feature_set']}  "
            f"family={payload['family']}  combos={payload['combos']}  "
            f"rows={payload['rows_used']}",
            f"  {'RMSE':>8} {'+-std':>7} {'MAE':>7} {'R2':>7}",
            f"  untuned {untuned['rmse']:>7.3f} {untuned['rmse_std']:>7.3f} "
            f"{untuned['mae']:>7.3f} {untuned['r2']:>7.3f}",
            f"  tuned   {tuned['rmse']:>7.3f} {tuned['rmse_std']:>7.3f} "
            f"{tuned['mae']:>7.3f} {tuned['r2']:>7.3f}",
            f"  best params: {payload['best_params']}",
            f"  train RMSE at best config: {payload['best_train_rmse']:.3f} "
            f"(gap to CV: {payload['best_train_rmse'] - tuned['rmse']:+.3f})",
            f"  chosen: {payload['chosen']}"
            + ("" if payload["chosen"] == "tuned" else " - tuning did not improve on the defaults"),
        ]

        top = payload["search"][:3]
        if len(payload["search"]) > len(top):
            lines.append(f"  best {len(top)} of {payload['combos']} configurations:")
        for row in top:
            lines.append(
                f"    {row['rmse_mean']:>6.3f} +-{row['rmse_std']:.3f}  {row['params']}"
            )

    lines += [
        "",
        "Tuning uses the comparison's folds, so the untuned, tuned and baseline rows are "
        "directly comparable.",
        f"Records written to {compare.CV_DIR.relative_to(config.ML_DIR.parent)}/.",
    ]
    return "\n".join(lines)


def main() -> int:
    payloads: list[dict[str, Any]] = []
    try:
        for subject in config.SUBJECTS:
            for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES):
                print(f"Tuning {subject}/{feature_set} ...", flush=True)
                payload = tune(subject, feature_set)
                save_record(payload)
                payloads.append(payload)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Tuning failed.\n  {error}", file=sys.stderr)
        return 1

    print()
    print(format_report(payloads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
