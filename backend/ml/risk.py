"""Risk bands: the documented mapping from a predicted grade to a priority label.

The bands are product policy, not something the model learns. They live here, in one
small module, so the evaluation, the inference layer and the API cannot disagree about
where `watch` ends and `on_track` begins.

A band is a coarse summary of a continuous estimate. A prediction of 13.9 and one of
14.0 are in different bands but are, in practical terms, the same estimate - which is
why every consumer of this module is expected to show the model's error beside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml import config


def band_index(grade: float) -> int:
    """Return the band index (0-3) for one predicted grade.

    Uses the documented `RISK_BAND_EDGES` with left-closed intervals: 10.0 is `watch`,
    not `critical`.
    """
    edges = np.asarray(config.RISK_BAND_EDGES[1:-1], dtype=float)
    return int(np.digitize(float(grade), edges, right=False))


def band_label(grade: float) -> str:
    """Return the band label for one predicted grade."""
    return config.RISK_BAND_LABELS[band_index(grade)]


def band_ranges() -> list[dict[str, Any]]:
    """The four bands as label + inclusive lower / exclusive upper bound.

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


def distance_to_edge(grade: float) -> float | None:
    """How far the estimate sits from the nearest band boundary, in grade points.

    Returns None when the grade is outside the documented scale (which the pipeline
    clips away, so this should not happen in practice).
    """
    grade = float(grade)
    edges = [float(edge) for edge in config.RISK_BAND_EDGES]
    if not edges[0] <= grade <= edges[-1]:
        return None
    return min(abs(grade - edge) for edge in edges[1:-1])
