"""Prediction orchestration: inference plus persistence, in that order.

The estimate is only stored once inference has succeeded, so a rejected request never
leaves a row behind claiming a model answered it.
"""

from __future__ import annotations

from typing import Any

from app.core import db
from app.schemas.prediction import PredictionRequest
from ml.inference import predict as inference


def create(request: PredictionRequest) -> dict[str, Any]:
    """Estimate a grade, explain it, persist it and return the full response."""
    attributes = {
        key: value
        for key, value in request.attributes.model_dump().items()
        if value is not None
    }

    result = inference.predict(request.subject, attributes, request.feature_set)
    estimate = result["prediction"]

    stored = db.save_prediction(
        subject=result["subject"],
        feature_set=result["feature_set"],
        model_version=result["model_version"],
        family=result["family"],
        predicted_g3=estimate["grade"],
        risk_band=estimate["band"],
        attributes=attributes,
        student_id=request.student_id,
    )

    return {
        **result,
        "id": stored["id"],
        "created_at": stored["created_at"],
        "student_id": stored["student_id"],
    }
