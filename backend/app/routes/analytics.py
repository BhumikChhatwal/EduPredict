"""Analytics endpoints: dataset aggregates and model records.

Each response carries the caveat that applies to it, so the interface can print the
qualification beside the number rather than in a separate page nobody reads.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from app.services import analytics

router = APIRouter(tags=["analytics"])

SubjectQuery = Literal["mat", "por"]
FeatureSetQuery = Literal["early", "with_grades"]


@router.get("/dashboard/summary")
def summary() -> dict[str, Any]:
    return analytics.summary()


@router.get("/analytics/options")
def options() -> dict[str, Any]:
    """The filter values the interface may offer, from the single source of truth."""
    from ml import config, risk

    return {
        "subjects": list(config.SUBJECTS),
        "subject_labels": dict(config.SUBJECT_LABELS),
        "feature_sets": ["early", "with_grades"],
        "risk_bands": risk.band_ranges(),
        "breakdown_attributes": list(analytics.BREAKDOWN_ATTRIBUTES),
    }


@router.get("/analytics/distribution")
def distribution(subject: SubjectQuery | None = None) -> dict[str, Any]:
    return analytics.distribution(subject)


@router.get("/analytics/correlation")
def correlation(subject: SubjectQuery = "mat") -> dict[str, Any]:
    return analytics.correlation(subject)


@router.get("/analytics/feature-importance")
def feature_importance(
    subject: SubjectQuery = "mat",
    feature_set: FeatureSetQuery = "early",
    limit: int = Query(default=14, ge=4, le=40),
) -> dict[str, Any]:
    return analytics.feature_importance(subject, feature_set, limit=limit)


@router.get("/analytics/model-comparison")
def model_comparison() -> dict[str, Any]:
    return analytics.model_comparison()


@router.get("/analytics/subject-comparison")
def subject_comparison() -> dict[str, Any]:
    return analytics.subject_comparison()


@router.get("/analytics/attributes")
def attributes(
    subject: SubjectQuery = "mat",
    attribute: str = Query(default="studytime", max_length=32),
) -> dict[str, Any]:
    try:
        return analytics.attribute_breakdown(subject, attribute)
    except KeyError as error:
        raise HTTPException(
            status_code=404, detail=f"Unknown breakdown attribute: {error.args[0]}"
        ) from None
