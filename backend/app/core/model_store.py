"""Model artifacts loaded once, at startup.

Why a class here: unlike the pipeline functions, the store has genuine state that must
outlive a request - the unpickled pipelines - and a lifecycle (load at startup, fail
loudly if absent, answer from memory thereafter). That is exactly what justifies an
object. Nothing else in the service needs one.

A missing artifact is a startup failure, not a degraded mode. An API that answers
predictions with an arbitrary or stale model is worse than one that refuses to start.
"""

from __future__ import annotations

from typing import Any

from ml import config
from ml.inference import predict as inference


class ModelStore:
    """Holds one fitted pipeline per (subject, feature set), plus its metadata."""

    def __init__(self) -> None:
        self._pipelines: dict[tuple[str, str], Any] = {}
        self._metadata: dict[tuple[str, str], dict[str, Any]] = {}

    def load(self) -> "ModelStore":
        pairs = inference.available_models()
        if not pairs:
            raise RuntimeError(
                "No serialized models found. Run `python -m ml.inference.serialize` "
                f"before starting the API (looked in {config.MODELS_DIR})."
            )

        missing = [
            (subject, feature_set)
            for subject in config.SUBJECTS
            for feature_set in ("early", "with_grades")
            if (subject, feature_set) not in pairs
        ]
        if missing:
            raise RuntimeError(
                f"Incomplete model set: missing artifacts for {missing}. Rerun the "
                "serializer so every subject and feature set is available."
            )

        for subject, feature_set in pairs:
            pipeline, metadata = inference.load(subject, feature_set)
            self._pipelines[(subject, feature_set)] = pipeline
            self._metadata[(subject, feature_set)] = metadata
        return self

    def __len__(self) -> int:
        return len(self._pipelines)

    def ready(self) -> bool:
        return bool(self._pipelines)

    def metadata(self, subject: str, feature_set: str) -> dict[str, Any]:
        key = (subject, feature_set)
        if key not in self._metadata:
            raise KeyError(f"No model loaded for {subject}/{feature_set}.")
        return self._metadata[key]

    def all_metadata(self) -> list[dict[str, Any]]:
        return [
            self._metadata[key]
            for key in sorted(self._metadata)
        ]

    def version(self) -> str | None:
        if not self._metadata:
            return None
        return next(iter(self._metadata.values()))["model_version"]
