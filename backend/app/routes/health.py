"""Health and readiness."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.core import config
from app.schemas.prediction import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, object]:
    """Liveness plus the model state the frontend shows in its footer.

    `model_loaded` is read from the store the lifespan created, not from the presence of
    a file on disk: this answers "can this process serve a prediction right now".
    """
    store = getattr(request.app.state, "models", None)
    loaded = bool(store and store.ready())

    return {
        "status": "ok",
        "app": config.APP_NAME,
        "version": config.APP_VERSION,
        "model_loaded": loaded,
        "model_version": store.version() if loaded else None,
        "models": len(store) if loaded else 0,
    }
