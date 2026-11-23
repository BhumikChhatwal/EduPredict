"""Pydantic contracts for prediction requests and responses.

These models are the API's public specification: the generated OpenAPI document, the
422 messages and the frontend's expectations all come from here. Bounds and category
levels mirror `ml.config`, which is the same source the dataset was validated against,
so a request the API accepts is always a row the training data could contain.

`G1` and `G2` are optional because the default `early` variant must not receive them.
The inference layer then enforces the feature-set contract exactly: sending grades with
`early` is rejected as unexpected, and omitting them with `with_grades` is rejected as
missing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentAttributes(BaseModel):
    """One student's attributes, on the documented domains."""

    model_config = ConfigDict(extra="forbid")

    school: Literal["GP", "MS"]
    sex: Literal["F", "M"]
    age: int = Field(ge=15, le=22)
    address: Literal["R", "U"]
    famsize: Literal["GT3", "LE3"]
    Pstatus: Literal["A", "T"]
    Medu: int = Field(ge=0, le=4)
    Fedu: int = Field(ge=0, le=4)
    Mjob: Literal["at_home", "health", "other", "services", "teacher"]
    Fjob: Literal["at_home", "health", "other", "services", "teacher"]
    reason: Literal["course", "home", "other", "reputation"]
    guardian: Literal["father", "mother", "other"]
    traveltime: int = Field(ge=1, le=4)
    studytime: int = Field(ge=1, le=4)
    failures: int = Field(ge=0, le=4)
    schoolsup: Literal["no", "yes"]
    famsup: Literal["no", "yes"]
    paid: Literal["no", "yes"]
    activities: Literal["no", "yes"]
    nursery: Literal["no", "yes"]
    higher: Literal["no", "yes"]
    internet: Literal["no", "yes"]
    romantic: Literal["no", "yes"]
    famrel: int = Field(ge=1, le=5)
    freetime: int = Field(ge=1, le=5)
    goout: int = Field(ge=1, le=5)
    Dalc: int = Field(ge=1, le=5)
    Walc: int = Field(ge=1, le=5)
    health: int = Field(ge=1, le=5)
    absences: int = Field(ge=0, le=93)

    G1: int | None = Field(default=None, ge=0, le=20)
    G2: int | None = Field(default=None, ge=0, le=20)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: Literal["mat", "por"] = "mat"
    feature_set: Literal["early", "with_grades"] = "early"
    # Optional link to a historical record, so the prediction shows in that student's
    # history. Anonymous submissions omit it.
    student_id: str | None = Field(default=None, max_length=16)
    attributes: StudentAttributes


class PredictionEstimate(BaseModel):
    grade: float
    band: str
    interval: list[float]
    interval_basis: str
    distance_to_band_edge: float | None


class PredictionContext(BaseModel):
    test_rmse: float
    test_mae: float
    test_r2: float
    cv_rmse: float
    baseline_cv_rmse: float
    within_2_marks: float
    rows_train: int
    rows_test: int
    dataset_sha256: str


class Contribution(BaseModel):
    feature: str
    value: float | str
    reference: float | str
    delta: float


class PredictionResponse(BaseModel):
    id: int
    created_at: str
    student_id: str | None
    subject: str
    feature_set: str
    family: str
    model_version: str
    prediction: PredictionEstimate
    context: PredictionContext
    contributions: list[Contribution]
    contribution_basis: str
    limitations: list[str]


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    model_loaded: bool
    model_version: str | None
    models: int


class StudentSummary(BaseModel):
    student_id: str
    subject: str
    age: int
    sex: str
    studytime: int
    failures: int
    absences: int
    G1: int
    G2: int
    G3: int
    band: str


class StudentPage(BaseModel):
    items: list[StudentSummary]
    page: int
    page_size: int
    total: int
    pages: int


class StoredPrediction(BaseModel):
    id: int
    student_id: str | None
    subject: str
    feature_set: str
    model_version: str
    family: str
    predicted_g3: float
    risk_band: str
    attributes: dict[str, Any]
    created_at: str
