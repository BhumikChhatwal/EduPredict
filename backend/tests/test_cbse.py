"""Tests for the CBSE assessment layer.

These lock the external facts the product leans on: the marks conversion, the board's
pass mark, the absolute scholastic grade table, and the fact that the table is ordered
high to low so a mark never falls through to the wrong letter.
"""

from __future__ import annotations

import pytest

from ml import cbse


def test_marks_are_the_native_grade_times_five() -> None:
    assert cbse.to_marks(0) == 0
    assert cbse.to_marks(10.4) == pytest.approx(52.0)
    assert cbse.to_marks(20) == 100
    assert cbse.to_native(75) == pytest.approx(15.0)


def test_pass_mark_is_cbse_thirty_three() -> None:
    assert cbse.passed(33)
    assert not cbse.passed(32.9)
    assert cbse.PASS_MARKS == 33


@pytest.mark.parametrize(
    ("marks", "expected"),
    [
        (100, "A1"),
        (91, "A1"),
        (90.9, "A2"),
        (81, "A2"),
        (80, "B1"),
        (71, "B1"),
        (70, "B2"),
        (61, "B2"),
        (60, "C1"),
        (51, "C1"),
        (50, "C2"),
        (41, "C2"),
        (40, "D"),
        (33, "D"),
        (32.9, "E"),
        (0, "E"),
    ],
)
def test_grade_letters_follow_the_absolute_table(marks: float, expected: str) -> None:
    assert cbse.letter(marks) == expected


def test_letter_ranges_are_contiguous_and_ordered_high_to_low() -> None:
    ranges = cbse.letter_ranges()

    assert [row["label"] for row in ranges] == [
        "A1", "A2", "B1", "B2", "C1", "C2", "D", "E",
    ]
    assert ranges[0]["min"] == 91
    assert ranges[-1]["min"] == 0
    for higher, lower in zip(ranges, ranges[1:]):
        assert lower["below"] == higher["min"]
