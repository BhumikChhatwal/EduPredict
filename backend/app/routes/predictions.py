"""Prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    StoredPrediction,
)
from app.services import predictions as service
from ml.inference import predict as inference

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    """Estimate a final grade, explain it, and record it.

    A rejected attribute set returns 422 with the offending column named. A request that
    passes validation is always persisted, which is what makes the dashboard's activity
    list truthful.
    """
    try:
        return service.create(request)
    except inference.AttributeValidationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/predictions/recent", response_model=list[StoredPrediction])
def recent(limit: int = Query(default=10, ge=1, le=50)) -> list[dict]:
    """The most recent stored predictions, newest first."""
    return db.recent_predictions(limit=limit)
