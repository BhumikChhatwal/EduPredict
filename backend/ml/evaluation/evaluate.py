"""Final evaluation on the held-out test rows.

This module is the one place the test set is used. Everything upstream - the comparison
and the tuning - deliberately saw training rows only, so this is the first and only
honest estimate of how the chosen configuration behaves on unseen students.

What is measured, and what each number means:

* **RMSE** - root mean squared error, in grade points. It is the typical magnitude of
  the error with large errors weighted more heavily (because they are squared), so a
  few bad misses move it more than MAE does. Saying "RMSE 2.6" means "typically about
  two and a half marks out", with the caveat that it is dominated by the worst cases.
* **MAE** - mean absolute error, in grade points: the average miss, treating every error
  by its size. Less sensitive than RMSE to the handful of worst predictions.
* **R²** - the share of the variance in the held-out grades explained relative to always
  predicting the training mean. 0.0 is no better than that mean, negative is worse than
  it (which is possible on a small test set). It is not "accuracy", and it is inflated
  by `with_grades`, where G1/G2 are near-copies of the target.
* **Mean residual** - mean of (predicted − actual). A non-zero value is *bias*: the model
  systematically over- or under-predicts, which RMSE and MAE both hide.
* **Within 2 marks** - share of predictions inside ±2 points of the actual grade. The
  number a teacher would actually care about.

Predictions are clipped to the documented 0–20 scale, because the product never serves an
estimate of 21.3. The record notes how many rows were clipped, so the effect is visible
rather than silent.

Per-band figures split the test rows by their **actual** grade band. Small bands on a
small test set (the `strong` band has only 5 and 9 rows) give unstable numbers, so the
band table reports the row count beside every statistic.

Run it with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.evaluation.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

# Headless backend: this runs as a script, not inside a notebook.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml import config  # noqa: E402
from ml.data import load  # noqa: E402
from ml.preprocessing import encoding, features, split  # noqa: E402
from ml.training import compare, tune  # noqa: E402

EVAL_DIR = config.ARTIFACTS_DIR / "evaluation"
FIGURES_DIR = config.REPORTS_DIR / "figures"

# Palette matching the frontend tokens: navy text, one blue accent, neutral rules.
INK = "#1e293b"
MUTED = "#64748b"
RULE = "#e2e8f0"
ACCENT = "#2563eb"

TARGET_MIN, TARGET_MAX = config.NUMERIC_RANGES[config.TARGET]
WITHIN_MARKS = 2.0


def risk_band_index(grades: np.ndarray) -> np.ndarray:
    """Map grades onto the four risk bands (0-3) using the documented cut-offs."""
    edges = np.asarray(config.RISK_BAND_EDGES[1:-1], dtype=float)
    return np.digitize(np.asarray(grades, dtype=float), edges, right=False)


def clip_predictions(predicted: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip to the grade scale and report how many rows were affected."""
    clipped = np.clip(predicted, TARGET_MIN, TARGET_MAX)
    changed = int(np.sum(clipped != predicted))
    return clipped, changed


def band_table(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, Any]]:
    """Per-band error statistics, keyed by band label.

    Bands are assigned from the *actual* grade, so the table answers "how wrong are we
    for students who really did land in this band?", not "what did we predict?".
    """
    bands = risk_band_index(y_true)
    table: dict[str, dict[str, Any]] = {}

    for index, label in enumerate(config.RISK_BAND_LABELS):
        mask = bands == index
        count = int(mask.sum())
        if count == 0:
            # Kept as an explicit empty entry: a missing band is information too.
            table[label] = {"n": 0, "mae": None, "rmse": None, "mean_residual": None,
                            "mean_actual": None, "mean_predicted": None}
            continue

        actual = y_true[mask]
        predicted = y_pred[mask]
        residual = predicted - actual
        table[label] = {
            "n": count,
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            # Positive means over-prediction.
            "mean_residual": float(np.mean(residual)),
            "mean_actual": float(np.mean(actual)),
            "mean_predicted": float(np.mean(predicted)),
        }

    return table


def evaluate(subject: str, feature_set: str) -> dict[str, Any]:
    """Fit the chosen configuration on training rows and score it on the test rows once."""
    tuned = tune.load_record(subject, feature_set)
    family = tuned["family"]
    params = tuned["chosen_params"]

    matrix_train, y_train, payload = compare.training_arrays(subject, feature_set)
    frame, _ = load.load_subject(subject)
    test_positions = np.asarray(payload["test_indices"], dtype=int)

    if set(payload["train_indices"]) & set(test_positions.tolist()):
        raise ValueError(f"{subject}: saved split shares rows between train and test.")

    test_frame = frame.iloc[test_positions].reset_index(drop=True)
    matrix_test = features.feature_matrix(test_frame, feature_set)
    y_test = test_frame[config.TARGET].to_numpy(dtype=np.float64)

    builder = tune.configured_builder(family, params)
    pipeline = encoding.build_pipeline(feature_set, builder())
    pipeline.fit(matrix_train, y_train)

    raw_predictions = pipeline.predict(matrix_test)
    predicted, clipped = clip_predictions(np.asarray(raw_predictions, dtype=np.float64))

    metrics = compare.regression_metrics(y_test, predicted)
    absolute = np.abs(predicted - y_test)
    residual = predicted - y_test

    rows = [
        {
            "row_index": int(test_positions[position]),
            "band": config.RISK_BAND_LABELS[int(risk_band_index(y_test)[position])],
            "actual": float(y_test[position]),
            "predicted": float(predicted[position]),
            "residual": float(residual[position]),
        }
        for position in range(len(y_test))
    ]

    return {
        "subject": subject,
        "feature_set": feature_set,
        "family": family,
        "params": params or "family defaults",
        "chosen": tuned["chosen"],
        "rows_train": int(len(matrix_train)),
        "rows_test": int(len(y_test)),
        "dataset_sha256": payload["dataset_sha256"],
        "metrics": {
            **metrics,
            "mean_residual": float(residual.mean()),
            "mae_absolute_max": float(absolute.max()),
            "within_2_marks": float((absolute <= WITHIN_MARKS).mean()),
        },
        "clipped_predictions": clipped,
        "prediction_range": [float(predicted.min()), float(predicted.max())],
        "bands": band_table(y_test, predicted),
        "cv_rmse": float(tuned["tuned_metrics"]["rmse"]),
        "cv_mae": float(tuned["tuned_metrics"]["mae"]),
        "untuned_cv_rmse": float(tuned["untuned"]["metrics"]["rmse"]),
        "untuned_cv_mae": float(tuned["untuned"]["metrics"]["mae"]),
        "baseline_cv_rmse": float(tuned["baseline_rmse"]),
        "predictions": rows,
    }


def _style_axes(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_facecolor("white")


def _band_rules(ax: plt.Axes, axis: str = "x") -> None:
    """Draw the risk-band boundaries as faint rules, so band edges are visible."""
    for edge in config.RISK_BAND_EDGES[1:-1]:
        if axis == "x":
            ax.axvline(edge, color=RULE, linewidth=1.0, zorder=0)
        else:
            ax.axhline(edge, color=RULE, linewidth=1.0, zorder=0)


def plot_predictions(record: dict[str, Any], path: Path) -> Path:
    """Predicted against actual, with the perfect-prediction diagonal."""
    actual = np.array([row["actual"] for row in record["predictions"]])
    predicted = np.array([row["predicted"] for row in record["predictions"]])
    metrics = record["metrics"]

    figure, ax = plt.subplots(figsize=(5.4, 4.8), dpi=170)
    _band_rules(ax, "x")
    _band_rules(ax, "y")
    ax.plot([0, 20], [0, 20], linestyle="--", linewidth=1.1, color=MUTED,
            label="perfect prediction", zorder=1)
    ax.scatter(actual, predicted, s=24, color=ACCENT, alpha=0.7,
               edgecolors="white", linewidths=0.4, zorder=2)

    ax.set_xlim(TARGET_MIN - 0.5, TARGET_MAX + 0.5)
    ax.set_ylim(TARGET_MIN - 0.5, TARGET_MAX + 0.5)
    ax.set_xlabel("Actual final grade (G3)", fontsize=9, color=INK)
    ax.set_ylabel("Predicted final grade", fontsize=9, color=INK)
    ax.set_title(
        f"{record['subject']}  ·  {record['feature_set']}  ·  {record['family']}",
        fontsize=10, color=INK, loc="left",
    )
    ax.text(
        0.03, 0.97,
        f"RMSE {metrics['rmse']:.2f}\nMAE {metrics['mae']:.2f}\n"
        f"R\u00b2 {metrics['r2']:.2f}\nn = {record['rows_test']}",
        transform=ax.transAxes, va="top", fontsize=8.5, color=INK,
        bbox={"facecolor": "white", "edgecolor": RULE, "boxstyle": "round,pad=0.35"},
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axes(ax)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_residuals(record: dict[str, Any], path: Path) -> Path:
    """Residual against the actual grade: shows bias and where the errors concentrate."""
    actual = np.array([row["actual"] for row in record["predictions"]])
    residual = np.array([row["residual"] for row in record["predictions"]])

    figure, ax = plt.subplots(figsize=(5.4, 4.6), dpi=170)
    _band_rules(ax, "x")
    ax.axhline(0.0, color=MUTED, linewidth=1.1, zorder=1)
    ax.scatter(actual, residual, s=24, color=ACCENT, alpha=0.7,
               edgecolors="white", linewidths=0.4, zorder=2)

    # Mean residual within each band, which is where systematic bias would show up.
    bands = risk_band_index(actual)
    for index in range(len(config.RISK_BAND_LABELS)):
        mask = bands == index
        if mask.sum() < 2:
            continue
        ax.plot(
            [actual[mask].min(), actual[mask].max()],
            [residual[mask].mean(), residual[mask].mean()],
            color=INK, linewidth=1.6, solid_capstyle="round", zorder=3,
        )

    ax.set_xlim(TARGET_MIN - 0.5, TARGET_MAX + 0.5)
    ax.set_xlabel("Actual final grade (G3)", fontsize=9, color=INK)
    ax.set_ylabel("Residual (predicted \u2212 actual)", fontsize=9, color=INK)
    ax.set_title(
        f"Residuals  ·  {record['subject']}  {record['feature_set']}",
        fontsize=10, color=INK, loc="left",
    )
    ax.text(
        0.98, 0.03,
        "bars: band mean residual",
        transform=ax.transAxes, ha="right", fontsize=8, color=MUTED,
    )
    _style_axes(ax)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def save_record(payload: dict[str, Any]) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{payload['subject']}__{payload['feature_set']}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def save_figures(payload: dict[str, Any]) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{payload['subject']}_{payload['feature_set']}"
    return [
        plot_predictions(payload, FIGURES_DIR / f"{stem}_predicted_vs_actual.png"),
        plot_residuals(payload, FIGURES_DIR / f"{stem}_residuals.png"),
    ]


def format_report(payloads: list[dict[str, Any]]) -> str:
    lines = [
        "Final evaluation - held-out test rows, used once",
        "=" * 74,
        "Grades are 0-20. RMSE/MAE are in grade points; R2 is against the training mean.",
    ]

    for payload in payloads:
        metrics = payload["metrics"]
        gap = metrics["rmse"] - payload["cv_rmse"]
        lines += [
            "",
            f"{payload['subject']}  {payload['feature_set']}  "
            f"{payload['family']}  n_test={payload['rows_test']}",
            f"  RMSE {metrics['rmse']:.3f}   MAE {metrics['mae']:.3f}   "
            f"R2 {metrics['r2']:.3f}   bias {metrics['mean_residual']:+.3f}",
            f"  within 2 marks: {metrics['within_2_marks']:.1%}   "
            f"worst miss: {metrics['mae_absolute_max']:.1f}   "
            f"clipped: {payload['clipped_predictions']}",
            f"  5-fold CV RMSE was {payload['cv_rmse']:.3f} "
            f"(test {'worse' if gap > 0 else 'better'} by {abs(gap):.3f})",
            "  band             n   MAE   RMSE    bias   actual   predicted",
        ]

        for label, stats in payload["bands"].items():
            if stats["n"] == 0:
                lines.append(f"  {label:<14} {0:>4}     -      -       -        -           -")
                continue
            lines.append(
                f"  {label:<14} {stats['n']:>4} "
                f"{stats['mae']:>5.2f} {stats['rmse']:>6.2f} "
                f"{stats['mean_residual']:>+7.2f} {stats['mean_actual']:>8.2f} "
                f"{stats['mean_predicted']:>11.2f}"
            )

    lines += [
        "",
        "bias = mean(predicted - actual); positive means over-prediction.",
        "Small bands (n below ~10) are indicative only; their MAE is one or two students.",
        f"Records: {EVAL_DIR.relative_to(config.ML_DIR.parent)}/   "
        f"Figures: {FIGURES_DIR.relative_to(config.ML_DIR.parent.parent)}/",
    ]
    return "\n".join(lines)


def main() -> int:
    payloads: list[dict[str, Any]] = []
    try:
        for subject in config.SUBJECTS:
            for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES):
                print(f"Evaluating {subject}/{feature_set} ...", flush=True)
                payload = evaluate(subject, feature_set)
                save_record(payload)
                save_figures(payload)
                payloads.append(payload)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"Evaluation failed.\n  {error}", file=sys.stderr)
        return 1

    print()
    print(format_report(payloads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
