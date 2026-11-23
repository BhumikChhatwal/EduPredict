"""API contract tests.

These run the real application with the real serialized artifacts. The point is not to
re-test the model - `test_inference.py` does that - but to lock the HTTP contract: which
inputs are accepted, what a rejected one returns, that the numbers in a response are the
numbers in the artifact, and that a prediction is actually persisted where the dashboard
will look for it.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

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
def client(tmp_path_factory):
    # Set before the app answers its first request: the path is read per connection.
    os.environ["EDUPREDICT_DB"] = str(tmp_path_factory.mktemp("db") / "test.db")

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["models"] == 4
    assert body["model_version"] == "1.0.0"


def test_model_info_lists_all_four_models_with_their_metrics(client: TestClient) -> None:
    body = client.get("/api/model/info").json()

    assert body["defaults"] == {"subject": "mat", "feature_set": "early"}
    assert len(body["models"]) == 4
    assert {model["feature_set"] for model in body["models"]} == {"early", "with_grades"}
    assert {model["subject"] for model in body["models"]} == {"mat", "por"}

    for model in body["models"]:
        assert model["metrics"]["cv_rmse"] > 0
        assert model["metrics"]["test_rmse"] > 0
        assert model["rows_train"] + model["rows_test"] > 300
        assert len(model["feature_importance"]) > 0

    assert len(body["risk_bands"]) == 4
    assert body["limitations"]


def test_model_info_exposes_the_target_and_its_scale(client: TestClient) -> None:
    """The Model page states what is predicted; the route must keep reshaping these through.

    `/info` rebuilds each model's metadata rather than returning it verbatim, so a field
    can disappear without any artifact changing. The interface reads `target` and
    `target_range` to caption the provenance line, and an empty caption shipped once
    because the route dropped them.
    """
    body = client.get("/api/model/info").json()

    assert len(body["models"]) == 4
    for model in body["models"]:
        assert model["target"] == "G3"
        assert model["target_range"] == [0.0, 20.0]
        assert model["metadata_schema"] >= 1


def test_raw_artifact_endpoint_matches_the_serializer(client: TestClient) -> None:
    response = client.get("/api/model/artifact/mat/early")

    assert response.status_code == 200
    assert response.json()["dataset_sha256"]
    assert client.get("/api/model/artifact/xxx/early").status_code == 404


def test_prediction_returns_an_estimate_band_and_explanations(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={"subject": "mat", "feature_set": "early", "attributes": EARLY_EXAMPLE},
    )

    assert response.status_code == 200
    body = response.json()

    assert 0 <= body["prediction"]["grade"] <= 20
    assert body["prediction"]["band"] in {"critical", "watch", "on_track", "strong"}
    assert body["prediction"]["interval"][0] <= body["prediction"]["grade"]
    assert len(body["contributions"]) > 0
    assert body["context"]["test_rmse"] == pytest.approx(4.086223125303643)
    assert body["limitations"]
    assert isinstance(body["id"], int)


def test_prediction_is_persisted_and_listed(client: TestClient) -> None:
    created = client.post(
        "/api/predict",
        json={
            "subject": "mat",
            "feature_set": "early",
            "student_id": "MAT-0001",
            "attributes": EARLY_EXAMPLE,
        },
    ).json()

    recent = client.get("/api/predictions/recent?limit=5").json()
    assert any(row["id"] == created["id"] for row in recent)

    history = client.get("/api/students/MAT-0001/predictions").json()
    assert [row["id"] for row in history][0] == created["id"]
    assert history[0]["risk_band"] == created["prediction"]["band"]

    summary = client.get("/api/dashboard/summary").json()
    assert summary["predictions_logged"] >= 2


def test_grades_sent_with_the_early_variant_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={
            "subject": "mat",
            "feature_set": "early",
            "attributes": dict(EARLY_EXAMPLE, G1=12, G2=13),
        },
    )

    assert response.status_code == 422
    assert "G1" in response.json()["detail"]


def test_with_grades_requires_both_period_grades(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={
            "subject": "mat",
            "feature_set": "with_grades",
            "attributes": EARLY_EXAMPLE,
        },
    )

    assert response.status_code == 422
    assert "G1" in response.json()["detail"]


def test_with_grades_prediction_is_served(client: TestClient) -> None:
    response = client.post(
        "/api/predict",
        json={
            "subject": "por",
            "feature_set": "with_grades",
            "attributes": dict(EARLY_EXAMPLE, G1=14, G2=15),
        },
    )

    assert response.status_code == 200
    assert response.json()["feature_set"] == "with_grades"
    assert response.json()["family"] == "LinearRegression"


def test_out_of_domain_and_unknown_fields_are_rejected(client: TestClient) -> None:
    bad_age = client.post(
        "/api/predict",
        json={"attributes": dict(EARLY_EXAMPLE, age=40)},
    )
    extra = client.post(
        "/api/predict",
        json={"attributes": dict(EARLY_EXAMPLE, G3=18)},
    )

    assert bad_age.status_code == 422
    assert extra.status_code == 422


def test_unknown_subject_is_a_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/predict", json={"subject": "che", "attributes": EARLY_EXAMPLE}
    )

    assert response.status_code == 422


def test_dashboard_summary_reports_both_populations(client: TestClient) -> None:
    body = client.get("/api/dashboard/summary").json()

    assert body["students"] == {"total": 1044, "mat": 395, "por": 649}
    assert body["at_risk"]["total"] == body["at_risk"]["mat"] + body["at_risk"]["por"]
    assert sum(body["bands"]["mat"].values()) == 395
    assert sum(body["bands"]["por"].values()) == 649
    assert len(body["risk_bands"]) == 4


def test_distribution_has_every_grade_slot(client: TestClient) -> None:
    body = client.get("/api/analytics/distribution").json()

    assert [series["subject"] for series in body["subjects"]] == ["mat", "por"]
    for series in body["subjects"]:
        assert len(series["counts"]) == 21
        assert sum(row["count"] for row in series["counts"]) == series["n"]


def test_correlation_matrix_is_square_and_bounded(client: TestClient) -> None:
    body = client.get("/api/analytics/correlation?subject=por").json()

    size = len(body["labels"])
    assert all(len(row) == size for row in body["matrix"])
    values = [value for row in body["matrix"] for value in row if value is not None]
    assert all(-1.0 <= value <= 1.0 for value in values)


def test_feature_importance_is_ordered(client: TestClient) -> None:
    body = client.get(
        "/api/analytics/feature-importance?subject=mat&feature_set=with_grades"
    ).json()

    importance = [row["importance"] for row in body["rows"]]
    assert importance == sorted(importance, reverse=True)
    assert body["rows"][0]["feature"] in {"G1", "G2", "grade_mean"}


def test_model_comparison_covers_every_candidate(client: TestClient) -> None:
    body = client.get("/api/analytics/model-comparison").json()

    assert len(body["chosen"]) == 4
    # 4 combinations x (4 candidates + 1 mean baseline)
    assert len(body["candidates"]) == 20
    assert sum(1 for row in body["candidates"] if row["baseline"]) == 4
    assert all(row["cv_rmse"] > 0 for row in body["candidates"])


def test_subject_comparison_reports_real_counts(client: TestClient) -> None:
    body = client.get("/api/analytics/subject-comparison").json()
    by_subject = {row["subject"]: row for row in body["subjects"]}

    assert by_subject["mat"]["rows"] == 395
    assert by_subject["por"]["rows"] == 649
    assert 0.0 <= by_subject["mat"]["pass_rate"] <= 1.0


def test_attribute_breakdown_keeps_group_sizes(client: TestClient) -> None:
    body = client.get("/api/analytics/attributes?subject=mat&attribute=studytime").json()

    assert body["attribute"] == "studytime"
    assert sum(group["n"] for group in body["groups"]) == 395
    assert [group["value"] for group in body["groups"]] == sorted(
        group["value"] for group in body["groups"]
    )
    assert client.get("/api/analytics/attributes?attribute=nope").status_code == 404


def test_analytics_options_come_from_config(client: TestClient) -> None:
    body = client.get("/api/analytics/options").json()

    assert body["subjects"] == ["mat", "por"]
    assert "studytime" in body["breakdown_attributes"]
    assert len(body["risk_bands"]) == 4


def test_students_are_paginated_and_filterable(client: TestClient) -> None:
    page = client.get("/api/students?page=2&page_size=10").json()

    assert page["page"] == 2
    assert page["page_size"] == 10
    assert page["total"] == 1044
    assert page["pages"] == 105
    assert len(page["items"]) == 10

    filtered = client.get("/api/students?subject=por&band=critical").json()
    assert filtered["total"] > 0
    assert all(row["band"] == "critical" for row in filtered["items"])

    searched = client.get("/api/students?search=MAT-0001").json()
    assert searched["total"] == 1
    assert searched["items"][0]["student_id"] == "MAT-0001"


def test_student_detail_and_missing_record(client: TestClient) -> None:
    body = client.get("/api/students/POR-0142").json()

    assert body["student_id"] == "POR-0142"
    assert body["subject"] == "por"
    assert set(body["attributes"]) >= {"school", "age", "absences", "G3"}
    assert body["band"] in {"critical", "watch", "on_track", "strong"}

    assert client.get("/api/students/MAT-9999").status_code == 404
    assert client.get("/api/students/NOPE-0001").status_code == 404
    assert client.get("/api/students/MAT-0001/predictions").status_code == 200
