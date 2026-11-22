"""Cross-validated comparison of the candidates, per subject and per feature set.

What this stage does, and what it deliberately does not:

* It reads the **saved split** and models on the training rows only. The test rows are
  not touched, so the numbers here are honest estimates of generalisation and the test
  set stays unopened until the final evaluation.
* Every candidate is scored on the **same folds**. Fold membership is generated once per
  comparison and passed to each candidate, so a difference in score is a difference
  between models rather than between two random partitions.
* RMSE is the selection criterion because the target is a grade in points: an RMSE of
  2.5 is readable as "typically about two and a half marks out". MAE is reported because
  it is less sensitive to the few extreme errors, and the fold-to-fold RMSE spread is
  reported because a model with the best mean but a wild spread is not a safe pick.
* Ties are broken deterministically: RMSE to four decimals, then MAE, then RMSE spread.
  Without a total order the "selected" model could change between runs on identical data.

Run it with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.training.compare
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from ml import config
from ml.data import load
from ml.preprocessing import encoding, features, split
from ml.training import models

CV_DIR = config.ARTIFACTS_DIR / "cv"

# Decimal places used when comparing selection keys. Float noise below this level is not
# a real difference between two models.
SELECTION_DECIMALS = 4


def fold_assignment(
    n_rows: int,
    *,
    folds: int = config.CV_FOLDS,
    seed: int = config.SEED,
) -> list[np.ndarray]:
    """Return the held-out row positions for each fold, generated exactly once.

    The same list is handed to every candidate in a comparison, which is what makes the
    scores comparable. Positions are into the training frame, not into the full dataset.
    """
    if n_rows < folds * 2:
        raise ValueError(
            f"Cannot make {folds} folds from {n_rows} rows; each fold needs train and "
            "test rows."
        )

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [
        np.sort(test_positions)
        for _, test_positions in splitter.split(np.arange(n_rows))
    ]


def training_arrays(
    subject: str, feature_set: str
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Load the subject, apply the saved split, and return training features and target.

    Raises if the saved split does not cover the file on disk, so a stale split cannot be
    silently combined with a changed dataset.
    """
    frame, _ = load.load_subject(subject)
    payload = split.load_split(subject)

    train_positions = np.asarray(payload["train_indices"], dtype=int)
    if train_positions.max() >= len(frame):
        raise ValueError(
            f"{subject}: saved train index {int(train_positions.max())} is outside the "
            f"{len(frame)}-row dataset; regenerate the split."
        )

    train_frame = frame.iloc[train_positions].reset_index(drop=True)
    matrix = features.feature_matrix(train_frame, feature_set)
    target = train_frame[config.TARGET].to_numpy(dtype=np.float64)

    if not np.isfinite(target).all():
        raise ValueError(f"{subject}: target contains non-finite values.")

    return matrix, target, payload


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and R² for one set of held-out predictions."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate_candidate(
    name: str,
    build: Callable[[], Any],
    feature_set: str,
    matrix: pd.DataFrame,
    target: np.ndarray,
    folds: list[np.ndarray],
) -> dict[str, Any]:
    """Fit `build()` once per fold and score it on the rows held out of that fold."""
    positions = np.arange(len(matrix))
    fold_rows: list[dict[str, Any]] = []

    for number, test_positions in enumerate(folds, start=1):
        train_positions = np.setdiff1d(positions, test_positions, assume_unique=False)

        # A fresh pipeline per fold: preprocessor and model are both fitted on this
        # fold's training rows only, so no encoding statistic leaks from held-out rows.
        pipeline = encoding.build_pipeline(feature_set, build())
        pipeline.fit(matrix.iloc[train_positions], target[train_positions])
        predicted = pipeline.predict(matrix.iloc[test_positions])

        fold_rows.append(
            {
                "fold": number,
                "n_train": int(len(train_positions)),
                "n_test": int(len(test_positions)),
                **regression_metrics(target[test_positions], predicted),
            }
        )

    rmse = np.array([row["rmse"] for row in fold_rows], dtype=np.float64)
    mae = np.array([row["mae"] for row in fold_rows], dtype=np.float64)
    r2 = np.array([row["r2"] for row in fold_rows], dtype=np.float64)

    return {
        "name": name,
        "baseline": name == models.BASELINE,
        "folds": fold_rows,
        # ddof=0: this is the spread of the five observed folds, not an estimate of a
        # population variance, so the sample correction would be the wrong question.
        "rmse_mean": float(rmse.mean()),
        "rmse_std": float(rmse.std()),
        "mae_mean": float(mae.mean()),
        "r2_mean": float(r2.mean()),
    }


def selection_key(result: dict[str, Any]) -> tuple[float, float, float, str]:
    """Total order over candidates: best RMSE, then MAE, then stability, then name.

    The trailing name makes the order total even if every score is identical, so the
    winner never depends on dict iteration luck.
    """
    return (
        round(result["rmse_mean"], SELECTION_DECIMALS),
        round(result["mae_mean"], SELECTION_DECIMALS),
        round(result["rmse_std"], SELECTION_DECIMALS),
        result["name"],
    )


def select(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best real candidate. The baseline is a floor, never a choice."""
    real = [result for result in results if not result["baseline"]]
    if not real:
        raise ValueError("No candidates to select from.")
    return min(real, key=selection_key)


def compare(
    subject: str,
    feature_set: str,
    *,
    folds: int = config.CV_FOLDS,
) -> dict[str, Any]:
    """Run the full comparison for one subject and feature set and return the record."""
    matrix, target, payload = training_arrays(subject, feature_set)
    assignments = fold_assignment(len(matrix), folds=folds)

    baseline_result = evaluate_candidate(
        models.BASELINE, models.baseline, feature_set, matrix, target, assignments
    )
    results = [baseline_result]
    for name, build in models.candidates().items():
        results.append(
            evaluate_candidate(name, build, feature_set, matrix, target, assignments)
        )

    selected = select(results)
    if selected["rmse_mean"] >= baseline_result["rmse_mean"]:
        # Not a style preference: every candidate failing to beat "always predict the
        # mean" means the pipeline or the split is wrong, and continuing would produce
        # a confident model that has learned nothing.
        raise RuntimeError(
            f"{subject}/{feature_set}: no candidate beats the mean baseline "
            f"(best RMSE {selected['rmse_mean']:.3f} vs "
            f"{baseline_result['rmse_mean']:.3f}). Check the split and the encoder."
        )

    return {
        "subject": subject,
        "feature_set": feature_set,
        "rows_total": int(payload["rows"]),
        "rows_used": int(len(matrix)),
        "rows_held_out": int(len(payload["test_indices"])),
        "folds": folds,
        "fold_seed": config.SEED,
        "fold_sizes": [int(len(fold)) for fold in assignments],
        "target_range": [float(target.min()), float(target.max())],
        "dataset_sha256": payload["dataset_sha256"],
        "results": results,
        "selected": selected["name"],
        "selected_metrics": {
            "rmse": selected["rmse_mean"],
            "rmse_std": selected["rmse_std"],
            "mae": selected["mae_mean"],
            "r2": selected["r2_mean"],
        },
        "baseline_rmse": baseline_result["rmse_mean"],
        "rmse_improvement_over_baseline": (
            baseline_result["rmse_mean"] - selected["rmse_mean"]
        ),
        "selection_key": ["rmse_mean", "mae_mean", "rmse_std", "name"],
    }


def save_results(payload: dict[str, Any]) -> Path:
    """Write the comparison record beside the other artifacts."""
    CV_DIR.mkdir(parents=True, exist_ok=True)
    path = CV_DIR / f"{payload['subject']}__{payload['feature_set']}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def format_report(payloads: list[dict[str, Any]]) -> str:
    lines = [
        "Model comparison - 5-fold CV on training rows only",
        "=" * 74,
        "RMSE is in grade points. The test set is not used at this stage.",
    ]

    for payload in payloads:
        lines += [
            "",
            f"{payload['subject']}  {payload['feature_set']}  "
            f"rows={payload['rows_used']}  folds={payload['folds']}  "
            f"(test rows reserved: {payload['rows_held_out']})",
            f"  {'model':<20} {'RMSE':>7} {'+-std':>6} {'MAE':>7} {'R2':>7}",
        ]

        for result in payload["results"]:
            label = result["name"] + ("*" if result["baseline"] else "")
            lines.append(
                f"  {label:<20} {result['rmse_mean']:>7.3f} "
                f"{result['rmse_std']:>6.3f} {result['mae_mean']:>7.3f} "
                f"{result['r2_mean']:>7.3f}"
            )

        metrics = payload["selected_metrics"]
        gain = payload["rmse_improvement_over_baseline"]
        share = 100.0 * gain / payload["baseline_rmse"]
        lines.append(
            f"  selected: {payload['selected']}  "
            f"(RMSE {metrics['rmse']:.3f}, {gain:.3f} better than baseline, "
            f"{share:.1f}%)"
        )

    lines += [
        "",
        "* = mean baseline, a floor rather than a candidate. "
        "Selection key: RMSE, then MAE, then RMSE spread.",
        f"Records written to {CV_DIR.relative_to(config.ML_DIR.parent)}/.",
    ]
    return "\n".join(lines)


def main() -> int:
    payloads: list[dict[str, Any]] = []
    try:
        for subject in config.SUBJECTS:
            for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES):
                print(f"Comparing {subject}/{feature_set} ...", flush=True)
                payload = compare(subject, feature_set)
                save_results(payload)
                payloads.append(payload)
    except (load.DatasetSchemaError, ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Comparison failed.\n  {error}", file=sys.stderr)
        return 1

    print()
    print(format_report(payloads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
