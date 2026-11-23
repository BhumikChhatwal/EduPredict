"""EduPredict API entrypoint.

The service is a consumer of two things produced elsewhere: the serialized ML artifacts
and the validated dataset on disk. It never trains, and it never edits the data. That
separation is why a model can be retrained and swapped without touching request
handling, and why every response can name the exact model version and dataset hash that
produced it.

Startup order matters: models are loaded first, because an API that answers without its
model is worse than one that refuses to start. The SQLite schema is created next; it is
idempotent.

Run it with:

    cd backend
    ./.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import config, db
from app.core.model_store import ModelStore
from app.routes import analytics, health, model, predictions, students

logger = logging.getLogger("edupredict")

DESCRIPTION = (
    "Academic performance analytics and grade estimation, built on a small public "
    "dataset. Every prediction is a statistical estimate with a measured error - not a "
    "guaranteed outcome and not a judgement about a student."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and prepare storage once, before the first request."""
    store = ModelStore().load()
    db.init_db()

    app.state.models = store
    logger.info(
        "Loaded %d models (version %s); database at %s",
        len(store),
        store.version(),
        config.db_path(),
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=config.APP_NAME,
        version=config.APP_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.FRONTEND_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Everything is mounted under /api so the frontend has one prefix to proxy and one
    # prefix to allow through CORS.
    for router in (health.router, model.router, predictions.router,
                   analytics.router, students.router):
        app.include_router(router, prefix="/api")

    return app


app = create_app()
