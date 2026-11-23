"""Model metadata: what answered the request, and how well it measured.

Served from the serializer's JSON, verbatim where possible. The interface's Model page
is therefore a rendering of a real artifact, not a hand-written description that drifts.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.core import config
from ml import risk
from ml.inference import predict as inference

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/info")
def model_info(request: Request) -> dict[str, Any]:
    """Every loaded model's metadata, the defaults, and the band policy."""
    store = request.app.state.models

    models = []
    for metadata in store.all_metadata():
        models.append(
            {
                "subject": metadata["subject"],
                "feature_set": metadata["feature_set"],
                "family": metadata["family"],
                "params": metadata["params"],
                "chosen": metadata["chosen"],
                "model_version": metadata["model_version"],
                "metadata_schema": metadata["metadata_schema"],
                "target": metadata["target"],
                "target_range": metadata["target_range"],
                "trained_at": metadata["trained_at"],
                "runtime": metadata["runtime"],
                "rows_train": metadata["rows_train"],
                "rows_test": metadata["rows_test"],
                "dataset_sha256": metadata["dataset_sha256"],
                "split": metadata["split"],
                "features": {
                    "count": metadata["features"]["count"],
                    "encoded_count": metadata["features"]["encoded_count"],
                    "names": metadata["features"]["names"],
                },
                "metrics": metadata["metrics"],
                "bands": metadata["bands"],
                "feature_importance": metadata["feature_importance"][:10],
            }
        )

    return {
        "defaults": {
            "subject": config.DEFAULT_SUBJECT,
            "feature_set": config.DEFAULT_FEATURE_SET,
        },
        "models": models,
        "risk_bands": risk.band_ranges(),
        "risk_band_policy": (
            "Bands are fixed cut-offs on the predicted grade, chosen for this product. "
            "They are not learned and not an institutional standard."
        ),
        "feature_sets": {
            "early": (
                "No period grades. The product default, and the only variant usable "
                "before any assessment has happened."
            ),
            "with_grades": (
                "Adds G1, G2 and two grade-derived columns. More accurate, but largely "
                "restates two earlier grades."
            ),
        },
        "limitations": inference.LIMITATIONS,
    }


@router.get("/artifact/{subject}/{feature_set}")
def model_artifact(subject: str, feature_set: str, request: Request) -> dict[str, Any]:
    """The raw metadata JSON exactly as the serializer wrote it.

    Kept separate from `/info` so an auditor can diff the artifact against the training
    records without the API reshaping anything.
    """
    try:
        metadata = request.app.state.models.metadata(subject, feature_set)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"No model artifact for {subject}/{feature_set}."
        ) from None
    return json.loads(json.dumps(metadata))
