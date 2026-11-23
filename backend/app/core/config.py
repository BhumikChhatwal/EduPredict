"""Application settings: names, paths, defaults, origins.

Kept separate from `ml/config.py` because the two answer different questions. That one
defines what the *dataset and pipeline* are; this one defines what the *service* is -
its version string, where its database lives, which model it serves by default.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ml import config as ml_config

APP_NAME = "EduPredict API"
APP_VERSION = "1.0.0"

# `backend/`, resolved from this file, so the service behaves the same regardless of the
# working directory uvicorn was started from.
BACKEND_DIR = Path(__file__).resolve().parents[2]

DEFAULT_SUBJECT = "mat"
DEFAULT_FEATURE_SET = "early"

# The dataset is fixed at 395 + 649 rows, so serving it from memory is honest: there is
# no scale story to tell, and a second copy in SQLite would be a second source of truth.
SUBJECTS = ml_config.SUBJECTS

FRONTEND_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

# Overridable so the test suite never writes to the development database.
DB_ENV_VAR = "EDUPREDICT_DB"


def db_path() -> Path:
    """SQLite file used for prediction history."""
    override = os.environ.get(DB_ENV_VAR)
    if override:
        return Path(override)
    if os.environ.get("VERCEL"):
        # A serverless filesystem is read-only apart from the temporary directory, and
        # that directory belongs to one instance. History therefore survives while an
        # instance stays warm and is not durable storage; the deployment notes say so.
        return Path(tempfile.gettempdir()) / "edupredict.db"
    return BACKEND_DIR / "edupredict.db"
