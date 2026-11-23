"""Serialise the chosen configuration per subject and feature set.

What is written, and why each part exists:

* **`<subject>__<feature_set>.joblib`** - the *whole fitted pipeline*, not just the
  estimator. That includes the column guard, the `ColumnTransformer` (with the one-hot
  vocabulary learned on training rows) and, for linear models, the scaler. Saving the
  estimator alone is the classic serving bug: the same attributes are then encoded with
  a different vocabulary than the one the model was fitted on, and predictions are
  quietly wrong. Joblib is used rather than pickle because the payload contains NumPy
  arrays, which joblib stores more efficiently.
* **`<subject>__<feature_set>.json`** - everything needed to describe and operate the
  artifact without unpickling it: the chosen family and hyperparameters, the dataset
  hash, the split protocol, CV and held-out metrics, the risk-band policy, the training
  reference values used for ablation explanations, and permutation importance.

Two facts worth stating plainly:

* The test rows are already spent (Phase 7). The metrics copied here are *the Phase 7
  numbers*, recorded for provenance - they are not a fresh estimate. Re-splitting would
  be the only way to get a new one.
* Permutation importance is computed on the test rows because the training rows would
  reward memorisation. It is descriptive, unstable when features are correlated, and
  it measures how much the model's error grows when a column is shuffled - not what
  would happen if a student changed that attribute.

Run it with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.inference.serialize
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance

from ml import config
from ml.data import load
from ml.evaluation import evaluate as evaluation
from ml.inference import predict as inference
from ml.preprocessing import encoding, features, split
from ml.training import compare, tune

# Bumped when the artifact contract changes (a new field that consumers rely on, or a
# different meaning for an existing one). The API reports it with every prediction.
MODEL_VERSION = "2.0.0"
METADATA_SCHEMA = 1

PERMUTATION_REPEATS = 5


def reference_values(matrix: pd.DataFrame) -> dict[str, Any]:
    """Per-feature reference values: median for numbers, mode for categories.

    These are what the ablation explanation substitutes, so they are computed from the
    training rows and shipped inside the metadata - the serving process must not need
    the dataset to explain a prediction.
    """
    references: dict[str, Any] = {}
    for column in matrix.columns:
        series = matrix[column]
        if pd.api.types.is_numeric_dtype(series):
            references[column] = float(series.median())
        else:
            modes = series.mode(dropna=True)
            references[column] = str(modes.iloc[0]) if not modes.empty else str(series.iloc[0])
    return references


def feature_importance(
    pipeline: Any,
    matrix_test: pd.DataFrame,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    """Permutation importance on the held-out rows, in grade points of added RMSE.

    Shuffling one column at a time and re-scoring gives an interpretable quantity:
    "how much worse does the error get when this column's information is destroyed?"
    A near-zero value does not prove the feature is irrelevant, only that the model
    does not rely on it on these rows.
    """
    result = permutation_importance(
        pipeline,
        matrix_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=PERMUTATION_REPEATS,
        random_state=config.SEED,
        n_jobs=1,
    )

    rows = [
        {
            "feature": str(name),
            "importance": float(result.importances_mean[index]),
            "std": float(result.importances_std[index]),
        }
        for index, name in enumerate(matrix_test.columns)
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return rows


def serialize(subject: str, feature_set: str) -> dict[str, Any]:
    """Fit the Phase 6/7 configuration on all training rows and write both artifacts."""
    tuned_record = tune.load_record(subject, feature_set)
    evaluation_record = json.loads(
        (evaluation.EVAL_DIR / f"{subject}__{feature_set}.json").read_text(encoding="utf-8")
    )

    family = tuned_record["family"]
    params = tuned_record["chosen_params"]

    matrix_train, y_train, payload = compare.training_arrays(subject, feature_set)

    frame, _ = load.load_subject(subject)
    test_positions = np.asarray(payload["test_indices"], dtype=int)
    test_frame = frame.iloc[test_positions].reset_index(drop=True)
    matrix_test = features.feature_matrix(test_frame, feature_set)
    y_test = test_frame[config.TARGET].to_numpy(dtype=np.float64)

    pipeline = encoding.build_pipeline(feature_set, tune.configured_builder(family, params)())
    pipeline.fit(matrix_train, y_train)

    # A sanity check against the Phase 7 record: the artifact must reproduce the
    # predictions the evaluation reported, or something changed under us.
    predicted, _ = evaluation.clip_predictions(
        np.asarray(pipeline.predict(matrix_test), dtype=np.float64)
    )
    reported = np.array([row["predicted"] for row in evaluation_record["predictions"]])
    if len(predicted) != len(reported):
        raise RuntimeError(f"{subject}/{feature_set}: test row count changed since Phase 7.")
    largest_gap = float(np.max(np.abs(predicted - reported)))
    if largest_gap > 1e-6:
        raise RuntimeError(
            f"{subject}/{feature_set}: refitted pipeline disagrees with the Phase 7 "
            f"record by {largest_gap:.6f} grade points; regenerate the evaluation first."
        )

    metadata: dict[str, Any] = {
        "metadata_schema": METADATA_SCHEMA,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "subject": subject,
        "feature_set": feature_set,
        "family": family,
        "params": params,
        "chosen": tuned_record["chosen"],
        "target": config.TARGET,
        "target_range": [float(config.NUMERIC_RANGES[config.TARGET][0]),
                         float(config.NUMERIC_RANGES[config.TARGET][1])],
        "rows_train": int(len(matrix_train)),
        "rows_test": int(len(y_test)),
        "dataset_sha256": payload["dataset_sha256"],
        "split": {
            "test_size": config.TEST_SIZE,
            "seed": config.SEED,
            "stratification_edges": list(config.STRATIFICATION_EDGES),
            "folds": config.CV_FOLDS,
        },
        "features": {
            "count": int(matrix_train.shape[1]),
            "encoded_count": int(len(pipeline[:-1].get_feature_names_out())),
            "names": [str(column) for column in matrix_train.columns],
        },
        "metrics": {
            "cv_rmse": evaluation_record["cv_rmse"],
            "cv_mae": evaluation_record["cv_mae"],
            "baseline_cv_rmse": evaluation_record["baseline_cv_rmse"],
            "test_rmse": evaluation_record["metrics"]["rmse"],
            "test_mae": evaluation_record["metrics"]["mae"],
            "test_r2": evaluation_record["metrics"]["r2"],
            "test_mean_residual": evaluation_record["metrics"]["mean_residual"],
            "within_2_marks": evaluation_record["metrics"]["within_2_marks"],
            "worst_error": evaluation_record["metrics"]["mae_absolute_max"],
            "clipped_predictions": evaluation_record["clipped_predictions"],
        },
        "bands": evaluation_record["bands"],
        "risk_bands": {
            "edges": list(config.RISK_BAND_EDGES),
            "labels": list(config.RISK_BAND_LABELS),
            "policy": "Product policy on the predicted grade, not learned from the data.",
        },
        "feature_importance": feature_importance(pipeline, matrix_test, y_test),
        "reference_values": reference_values(matrix_train),
        "artifact": inference.artifact_path(subject, feature_set).name,
    }

    artifact_path, metadata_path = inference.model_paths(subject, feature_set)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # Reload once, so a broken artifact is caught here rather than by the API.
    reloaded = joblib.load(artifact_path)
    if not np.allclose(
        np.asarray(reloaded.predict(matrix_test), dtype=np.float64), pipeline.predict(matrix_test)
    ):
        raise RuntimeError(f"{subject}/{feature_set}: reloaded artifact predicts differently.")

    metadata["artifact_bytes"] = artifact_path.stat().st_size
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def format_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "Serialized models",
        "=" * 74,
        f"Model version {MODEL_VERSION}; written to "
        f"{config.MODELS_DIR.relative_to(config.ML_DIR.parent)}/",
    ]

    for record in records:
        metrics = record["metrics"]
        lines += [
            "",
            f"{record['subject']}  {record['feature_set']}  {record['family']}"
            f"  ({record['chosen']})",
            f"  params {record['params'] or 'family defaults'}",
            f"  features {record['features']['count']} -> "
            f"{record['features']['encoded_count']} encoded",
            f"  test RMSE {metrics['test_rmse']:.3f}  MAE {metrics['test_mae']:.3f}  "
            f"R2 {metrics['test_r2']:.3f}  (CV RMSE {metrics['cv_rmse']:.3f})",
            f"  artifact {record['artifact']}  "
            f"({record.get('artifact_bytes', 0) / 1024:.0f} KiB)",
        ]

        top = record["feature_importance"][:5]
        lines.append("  top permutation importance (added RMSE when shuffled):")
        for row in top:
            lines.append(f"    {row['feature']:<20} {row['importance']:+.3f} +-{row['std']:.3f}")

    lines += [
        "",
        "The metadata JSON holds the metrics, the band policy, the reference values used "
        "for explanations and the importance table.",
        "Artifacts are build outputs: regenerate them with this command, do not edit them.",
    ]
    return "\n".join(lines)


def main() -> int:
    records: list[dict[str, Any]] = []
    try:
        for subject in config.SUBJECTS:
            for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES):
                print(f"Serializing {subject}/{feature_set} ...", flush=True)
                records.append(serialize(subject, feature_set))
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Serialization failed.\n  {error}", file=sys.stderr)
        return 1

    print()
    print(format_report(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
