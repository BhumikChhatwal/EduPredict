"""Stratified train/test split with saved indices.

Why the split is written down rather than recomputed everywhere:

* The test set is opened **once**. If every script re-derives it, someone eventually
  splits differently and the reported metric silently becomes a validation score.
* Saved indices plus a dataset hash make a result auditable: the JSON says exactly which
  rows produced which number, and against which file.

Stratification is on coarse grade bands, not on the raw 0-20 grade. There are 395 maths
rows, so several individual grade values have only a handful of rows; stratifying on
those would make the split fragile, while the bands (below pass, marginal, comfortable,
strong) are the distinctions that matter here.

Run the split and save the indices with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.preprocessing.split
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import config
from ml.data import load

SPLITS_DIR = config.ARTIFACTS_DIR / "splits"


def dataset_sha256(subject: str) -> str:
    """Hash the source CSV so saved indices are pinned to an exact file version."""
    digest = hashlib.sha256()
    with load.raw_path(subject).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grade_bands(frame: pd.DataFrame) -> pd.Series:
    """Map each row's target onto its risk band index (0-3)."""
    return pd.cut(
        frame[config.TARGET],
        bins=config.STRATIFICATION_EDGES,
        labels=False,
        right=False,
    ).astype("int64")


def stratified_split(
    frame: pd.DataFrame,
    *,
    seed: int = config.SEED,
    test_size: float = config.TEST_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted train and test index arrays.

    Sorted so the saved JSON is stable and a diff of two runs is meaningful.
    """
    bands = grade_bands(frame)
    smallest_band = int(bands.value_counts().min())

    # Stratified splitting needs at least two rows per stratum to place one on each
    # side; a single-row band cannot be split without breaking the guarantee.
    if smallest_band < 2:
        raise ValueError(
            f"Cannot stratify: a grade band has only {smallest_band} row(s). "
            f"Bands are defined by {config.STRATIFICATION_EDGES}."
        )

    positions = np.arange(len(frame))
    train, test = train_test_split(
        positions,
        test_size=test_size,
        random_state=seed,
        stratify=bands.to_numpy(),
    )

    return np.sort(train), np.sort(test)


def split_payload(
    subject: str,
    frame: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int = config.SEED,
    test_size: float = config.TEST_SIZE,
) -> dict[str, Any]:
    """Build the JSON payload: indices plus the facts needed to trust them."""
    bands = grade_bands(frame)
    band_summary = {}
    for position, label in enumerate(config.RISK_BAND_LABELS):
        mask = bands == position
        band_summary[label] = {
            "total": int(mask.sum()),
            "train": int(mask.iloc[train].sum()),
            "test": int(mask.iloc[test].sum()),
        }

    zero = frame[config.TARGET] == 0
    return {
        "subject": subject,
        "source_file": config.SUBJECT_FILES[subject],
        "dataset_sha256": dataset_sha256(subject),
        "rows": int(len(frame)),
        "seed": seed,
        "test_size": test_size,
        "stratification_edges": list(config.STRATIFICATION_EDGES),
        "band_labels": list(config.RISK_BAND_LABELS),
        "bands": band_summary,
        "grade_zero_rows": {"train": int(zero.iloc[train].sum()), "test": int(zero.iloc[test].sum())},
        "train_indices": train.tolist(),
        "test_indices": test.tolist(),
    }


def save_split(subject: str, payload: dict[str, Any]) -> Path:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{subject}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_split(subject: str) -> dict[str, Any]:
    """Read a saved split and check it against the data currently on disk."""
    path = SPLITS_DIR / f"{subject}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved split at {path}. Run `python -m ml.preprocessing.split` first."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_rows = config.EXPECTED_ROW_COUNTS[subject]
    if payload["rows"] != expected_rows:
        raise ValueError(
            f"{subject}: saved split covers {payload['rows']} rows but the dataset has "
            f"{expected_rows}; regenerate the split."
        )

    train = payload["train_indices"]
    test = payload["test_indices"]
    if len(set(train) & set(test)):
        raise ValueError(f"{subject}: saved split has rows in both train and test.")
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError(f"{subject}: saved split has duplicate indices.")

    return payload


def build_split(subject: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load, validate, split, save and return the two frames plus the payload."""
    frame, _ = load.load_subject(subject)
    train_idx, test_idx = stratified_split(frame)
    payload = split_payload(subject, frame, train_idx, test_idx)
    save_split(subject, payload)

    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    return train, test, payload


def format_summary(payloads: list[dict[str, Any]]) -> str:
    lines = ["Stratified split", "=" * 74]

    for payload in payloads:
        train_rows = len(payload["train_indices"])
        test_rows = len(payload["test_indices"])
        test_share = test_rows / payload["rows"]
        lines += [
            "",
            f"{payload['subject']}  rows={payload['rows']}  "
            f"train={train_rows}  test={test_rows}  ({test_share:.1%} test, "
            f"seed={payload['seed']})",
            f"  dataset sha256: {payload['dataset_sha256'][:16]}...",
            f"  G3 = 0 rows: train={payload['grade_zero_rows']['train']}  "
            f"test={payload['grade_zero_rows']['test']}",
            "  band             total  train  test  test share",
        ]
        for label, counts in payload["bands"].items():
            share = counts["test"] / counts["total"] if counts["total"] else 0.0
            lines.append(
                f"  {label:<15} {counts['total']:>5} {counts['train']:>6} "
                f"{counts['test']:>5}  {share:>9.1%}"
            )

    lines += [
        "",
        f"Indices written to {SPLITS_DIR.relative_to(config.ML_DIR.parent)}/",
        "Each band's test share tracks the overall test size; the test set is unchanged "
        "until the final evaluation.",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        payloads = [build_split(subject)[2] for subject in config.SUBJECTS]
    except (load.DatasetSchemaError, ValueError) as error:
        print(f"Split failed.\n  {error}", file=sys.stderr)
        return 1

    print(format_summary(payloads))
    return 0


if __name__ == "__main__":
    sys.exit(main())
