"""Student record endpoints."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.prediction import StoredPrediction, StudentPage
from app.services import students as service

router = APIRouter(prefix="/students", tags=["students"])


@router.get("", response_model=StudentPage)
def list_students(
    subject: Literal["mat", "por"] | None = None,
    search: str | None = Query(default=None, max_length=16),
    band: Literal["critical", "watch", "on_track", "strong"] | None = None,
    sort: Literal["student_id", "grade_asc", "grade_desc", "absences_desc"] = "student_id",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=5, le=100),
) -> dict[str, Any]:
    """Paginated records, described by their actual final grade and band."""
    return service.list_students(
        subject=subject,
        search=search,
        band=band,
        sort=sort,
        page=page,
        page_size=page_size,
    )


@router.get("/{student_id}")
def get_student(student_id: str) -> dict[str, Any]:
    try:
        return service.get_student(student_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"No record with ID {student_id!r}."
        ) from None


@router.get("/{student_id}/predictions", response_model=list[StoredPrediction])
def student_predictions(student_id: str) -> list[dict]:
    """Predictions stored against this student ID, newest first."""
    key = student_id.strip().upper()
    if not db.predictions_for_student(key):
        # Distinguishing "no history" from "no such student" matters: both are empty
        # lists otherwise, and the interface would claim the record does not exist.
        try:
            service.get_student(key)
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"No record with ID {student_id!r}."
            ) from None
    return db.predictions_for_student(key)
