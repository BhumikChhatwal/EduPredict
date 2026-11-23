"""CBSE assessment facts, in one place.

The product is built for CBSE secondary schools (classes IX-X): marks out of 100 for
each subject, a pass mark of 33, and the board's absolute scholastic grades. Those are
external facts about the school system, not properties of the dataset, so they live here
rather than inside the model configuration - the model still learns the dataset's native
0-20 grade and nothing else.

Why the conversion exists: the only training data is the Portuguese 0-20 course grade.
Multiplying by five puts an estimate on the CBSE marks scale, which is what a teacher in
India reads, but it does not make the estimate a CBSE measurement. Every surface that
shows marks says where they came from.

CBSE references (secondary curriculum and grading circulars):
* Compulsory subjects at classes IX-X: Language 1, Language 2, Mathematics (Basic or
  Standard), Science, Social Science; 80 marks external + 20 internal = 100.
* A student must secure 33% to pass a subject.
* School-level (class IX) grade bands are absolute: A1 91-100, A2 81-90, B1 71-80,
  B2 61-70, C1 51-60, C2 41-50, D 33-40, E below 33. The class X board result uses
  relative grading by rank within each subject; that cannot be reproduced for one
  student, so the absolute school-level table is used and labelled as such.
"""

from __future__ import annotations

from typing import Any

# One native grade point (0-20 dataset scale) is five CBSE marks (0-100 scale).
MARKS_PER_POINT = 5.0
MARKS_MAX = 100.0
PASS_MARKS = 33.0

# Absolute scholastic grades for classes IX-X, highest band first.
GRADE_LETTERS: tuple[tuple[float, str], ...] = (
    (91.0, "A1"),
    (81.0, "A2"),
    (71.0, "B1"),
    (61.0, "B2"),
    (51.0, "C1"),
    (41.0, "C2"),
    (33.0, "D"),
    (0.0, "E"),
)


def to_marks(grade: float, digits: int = 1) -> float:
    """Convert a native 0-20 dataset grade to CBSE marks out of 100."""
    return round(float(grade) * MARKS_PER_POINT, digits)


def to_native(marks: float) -> float:
    """Convert CBSE marks back to the dataset's 0-20 scale (used by plots)."""
    return float(marks) / MARKS_PER_POINT


def letter(marks: float) -> str:
    """CBSE absolute scholastic grade for a mark out of 100."""
    marks = float(marks)
    for floor, label in GRADE_LETTERS:
        if marks >= floor:
            return label
    return "E"


def passed(marks: float) -> bool:
    """Whether a mark clears CBSE's 33% pass requirement."""
    return float(marks) >= PASS_MARKS


def letter_ranges() -> list[dict[str, Any]]:
    """The grade table as label + mark band, for the interface and the docs."""
    ranges = []
    for index, (floor, label) in enumerate(GRADE_LETTERS):
        ceiling = GRADE_LETTERS[index - 1][0] if index > 0 else MARKS_MAX + 1
        ranges.append({"label": label, "min": floor, "below": ceiling})
    return ranges
