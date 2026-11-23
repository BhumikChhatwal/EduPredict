"""Tests for serialization and the serving-side prediction path.

The properties that matter: validation rejects rows the training data could not
contain, a prediction stays on the grade scale, the band follows the documented
cut-offs, contributions are signed consistently and reproducible, and the serialized
artifact loads and predicts identically to the in-memory pipeline.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml import config, risk
from ml.inference import predict as inference
from ml.preprocessing import features

EARLY_EXAMPLE = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 3,
    "Fedu": 2,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 2,
    "health": 3,
    "absences": 4,
}


@pytest.fixture(scope="module")
def early_result() -> dict:
    return inference.predict("mat", EARLY_EXAMPLE)


def test_every_chosen_model_is_serialized_with_metadata() -> None:
    pairs = inference.available_models()
    assert set(pairs) == {
        (subject, feature_set)
        for subject in config.SUBJECTS
        for feature_set in (features.FEATURE_SET_EARLY, features.FEATURE_SET_WITH_GRADES)
    }

    for subject, feature_set in pairs:
        _, metadata = inference.load(subject, feature_set)
        assert metadata["subject"] == subject
        assert metadata["feature_set"] == feature_set
        assert metadata["dataset_sha256"]
        assert metadata["metrics"]["cv_rmse"] > 0
        assert metadata["features"]["count"] == len(features.feature_columns(feature_set))


def test_metadata_matches_the_phase_seven_record() -> None:
    _, metadata = inference.load("mat", features.FEATURE_SET_EARLY)
    record = json.loads(
        (config.ARTIFACTS_DIR / "evaluation" / "mat__early.json").read_text(encoding="utf-8")
    )

    assert metadata["metrics"]["test_rmse"] == pytest.approx(record["metrics"]["rmse"])
    assert metadata["metrics"]["test_mae"] == pytest.approx(record["metrics"]["mae"])
    assert metadata["dataset_sha256"] == record["dataset_sha256"]


def test_prediction_is_a_grade_on_the_documented_scale(early_result: dict) -> None:
    grade = early_result["prediction"]["grade"]
    low, high = config.NUMERIC_RANGES[config.TARGET]

    assert low <= grade <= high
    assert early_result["prediction"]["band"] == risk.band_label(grade)
    lower, upper = early_result["prediction"]["interval"]
    assert lower <= grade <= upper
    assert early_result["feature_set"] == features.FEATURE_SET_EARLY


def test_predictions_are_deterministic(early_result: dict) -> None:
    again = inference.predict("mat", EARLY_EXAMPLE)
    assert again["prediction"]["grade"] == early_result["prediction"]["grade"]
    assert again["contributions"] == early_result["contributions"]


def test_contributions_are_signed_and_reproducible(early_result: dict) -> None:
    contributions = early_result["contributions"]
    assert 0 < len(contributions) <= 12
    assert contributions == sorted(
        contributions, key=lambda row: abs(row["delta"]), reverse=True
    )

    # Removing the largest positive contributor must lower the estimate: that is the
    # definition of the sign the API documents.
    largest = contributions[0]
    _, metadata = inference.load("mat", features.FEATURE_SET_EARLY)
    pipeline = inference.load("mat", features.FEATURE_SET_EARLY)[0]
    frame = inference.validate_attributes(EARLY_EXAMPLE, features.FEATURE_SET_EARLY)
    matrix = features.feature_matrix(frame, features.FEATURE_SET_EARLY)
    ablated = matrix.copy()
    ablated[largest["feature"]] = metadata["reference_values"][largest["feature"]]
    ablated_grade = float(pipeline.predict(ablated)[0])

    assert ablated_grade == pytest.approx(
        early_result["prediction"]["grade"] - largest["delta"], abs=0.01
    )


def test_with_grades_variant_requires_period_grades() -> None:
    attributes = dict(EARLY_EXAMPLE, G1=12, G2=13)
    result = inference.predict("mat", attributes, features.FEATURE_SET_WITH_GRADES)

    assert result["feature_set"] == features.FEATURE_SET_WITH_GRADES
    assert 0 <= result["prediction"]["grade"] <= 20


def test_missing_grades_fail_loudly() -> None:
    with pytest.raises(inference.AttributeValidationError) as error:
        inference.predict("mat", EARLY_EXAMPLE, features.FEATURE_SET_WITH_GRADES)

    assert "G1" in str(error.value)


def test_out_of_domain_values_are_rejected() -> None:
    with pytest.raises(inference.AttributeValidationError):
        inference.predict("mat", dict(EARLY_EXAMPLE, age=40))

    with pytest.raises(inference.AttributeValidationError):
        inference.predict("mat", dict(EARLY_EXAMPLE, school="XX"))

    with pytest.raises(inference.AttributeValidationError):
        inference.predict("mat", dict(EARLY_EXAMPLE, studytime="lots"))


def test_unexpected_attributes_are_rejected_not_ignored() -> None:
    with pytest.raises(inference.AttributeValidationError) as error:
        inference.predict("mat", dict(EARLY_EXAMPLE, G3=18))

    assert "G3" in str(error.value)


def test_risk_band_boundaries_are_left_closed() -> None:
    assert risk.band_label(9.99) == "critical"
    assert risk.band_label(10.0) == "watch"
    assert risk.band_label(13.99) == "watch"
    assert risk.band_label(14.0) == "on_track"
    assert risk.band_label(17.0) == "strong"

    assert risk.distance_to_edge(13.9) == pytest.approx(0.1)
    ranges = risk.band_ranges()
    assert ranges[0]["min"] == 0.0
    assert ranges[-1]["max"] == 21.0 and ranges[-1]["max_inclusive"] is True


def test_a_missing_artifact_raises_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(inference.config, "MODELS_DIR", tmp_path)
    inference.clear_cache()

    with pytest.raises(inference.ModelNotFoundError):
        inference.load("mat", features.FEATURE_SET_EARLY)

    inference.clear_cache()
