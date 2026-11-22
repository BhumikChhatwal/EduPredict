"""EduPredict API entrypoint.

Phase 1 scope: the application object, health check and CORS wiring only.
Prediction, analytics, records and model-metadata routes are added in
milestones 9 and 10, and each is mounted from its own router module.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

APP_NAME = "EduPredict API"
APP_VERSION = "0.1.0"


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=(
            "Academic performance analytics and grade estimation. "
            "Estimates are statistical, not guarantees."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", tags=["system"])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "app": APP_NAME,
            "version": APP_VERSION,
            "model_loaded": False,
            "model_version": None,
        }

    return app


app = create_app()
