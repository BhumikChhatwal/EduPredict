"""Risk bands: the documented mapping from predicted CBSE marks to a priority label.

The bands are product policy, not something the model learns. They live here, in one
small module, so the evaluation, the inference layer and the API cannot disagree about
where `watch` ends and `on_track` begins.

Units are **CBSE marks out of 100**, matching what the interface shows. The lowest band
starts at CBSE's 33% pass mark, so "critical" means "the estimate is a failing one".
The model itself predicts the dataset's native 0-20 grade; callers convert first
(`ml.cbse.to_marks`).

A band is a coarse summary of a continuous estimate. A prediction of 50.9 and one of
51.0 are in different bands but are, in practical terms, the same estimate - which is
why every consumer of this module is expected to show the model's error beside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml import config


def band_index(marks: float) -> int:
    """Return the band index (0-3) for one predicted mark out of 100.

    Uses the documented `RISK_BAND_EDGES` with left-closed intervals: 33.0 is `watch`,
    not `critical`.
    """
    edges = np.asarray(config.RISK_BAND_EDGES[1:-1], dtype=float)
    return int(np.digitize(float(marks), edges, right=False))


def band_label(marks: float) -> str:
    """Return the band label for one predicted mark out of 100."""
    return config.RISK_BAND_LABELS[band_index(marks)]


def band_ranges() -> list[dict[str, Any]]:
    """The four bands as label + inclusive lower / exclusive upper bound, in marks.

    Returned to the API so the interface can print the exact rule instead of
    paraphrasing it.
    """
    edges = config.RISK_BAND_EDGES
    return [
        {
            "label": label,
            "min": float(edges[index]),
            "max": float(edges[index + 1]),
            "max_inclusive": index == len(config.RISK_BAND_LABELS) - 1,
        }
        for index, label in enumerate(config.RISK_BAND_LABELS)
    ]


def distance_to_edge(marks: float) -> float | None:
    """How far the estimate sits from the nearest band boundary, in marks.

    Returns None when the mark is outside the documented scale (which the pipeline
    clips away, so this should not happen in practice).
    """
    marks = float(marks)
    edges = [float(edge) for edge in config.RISK_BAND_EDGES]
    if not edges[0] <= marks <= edges[-1]:
        return None
    return min(abs(marks - edge) for edge in edges[1:-1])
