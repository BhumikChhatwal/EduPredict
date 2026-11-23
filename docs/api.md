# API reference

Base path: `/api`. Interactive schema (generated from the Pydantic models, therefore
authoritative): `http://127.0.0.1:8000/docs`. This document is a summary with the
behaviour that OpenAPI cannot express.

Errors are always `{"detail": "..."}`. 422 carries FastAPI's field-error list; 404 is
an unknown student or artifact; 503 means the model store is not ready. Validation
bounds mirror the dataset's real domains, so an accepted request is always a row the
training data could contain.

## System

### `GET /health`

```json
{
  "status": "ok", "app": "EduPredict API", "version": "1.0.0",
  "model_loaded": true, "model_version": "1.0.0", "models": 4
}
```

`model_loaded` is read from the process's model store, not from a file's existence: it
answers "can this process serve a prediction right now".

## Model

### `GET /model/info`

Defaults, the four models' metadata, the CBSE grade scale, the band policy and the
limitations list. Each model carries `family`, `params`, `chosen` (`tuned` /
`untuned`), `model_version`, `metadata_schema`, `target`, `target_range`,
`target_range_marks`, `trained_at`, `runtime`, `rows_train`, `rows_test`,
`dataset_sha256`, `split`, `features` (counts and names), `metrics` (in CBSE marks),
`bands` (held-out error per actual band) and the top ten `feature_importance` rows.

`grade_scale` states the conversion (`marks = native grade × 5`), the pass mark (33)
and the absolute A1–E letter table. `subject_labels` maps `mat` to Mathematics and
`por` to Language.

### `GET /model/artifact/{subject}/{feature_set}`

The metadata JSON exactly as the serializer wrote it, unshaped. Intended for auditing
the served artifact against the training records.

`subject` is `mat` or `por`; `feature_set` is `early` or `with_grades`.

## Prediction

### `POST /predict`

Request:

```json
{
  "subject": "mat",
  "feature_set": "early",
  "student_id": "MAT-0001",
  "attributes": {
    "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 4, "Fedu": 4, "Mjob": "at_home", "Fjob": "teacher",
    "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
    "failures": 0, "schoolsup": "yes", "famsup": "no", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "no",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 1,
    "Walc": 1, "health": 3, "absences": 6
  }
}
```

`subject` defaults to `mat`, `feature_set` to `early`. `student_id` is optional and only
links the estimate to a record's history. `G1` and `G2` must be omitted for `early`
and are required for `with_grades`; both are enforced by the inference layer, not just
by Pydantic, because sending grades with `early` would silently change the model's
meaning.

There is **no `school` field**: the CBSE product dropped it, the models were retrained
without it (v2.0.0), and a request containing `school` is rejected with 422 rather than
ignored.

Response (abridged):

```json
{
  "id": 1, "created_at": "2026-11-23T10:00:00+00:00", "student_id": "MAT-0001",
  "subject": "mat", "feature_set": "early",
  "family": "RandomForest", "model_version": "1.0.0",
  "prediction": {
    "grade": 10.96,
    "marks": 54.8,
    "grade_letter": "C1",
    "band": "on_track",
    "interval": [34.4, 75.2],
    "interval_basis": "estimate +/- the model's held-out RMSE (20.4 CBSE marks); one average figure for every student, not a confidence interval",
    "distance_to_band_edge": 3.8
  },
  "context": {
    "test_rmse": 20.42, "test_mae": 15.7, "test_r2": 0.197, "cv_rmse": 19.27,
    "baseline_cv_rmse": 22.95, "within_10_marks": 0.443,
    "rows_train": 316, "rows_test": 79, "dataset_sha256": "..."
  },
  "contributions": [
    {"feature": "absences", "value": 6, "reference": 5, "delta": -0.21}
  ],
  "contribution_basis": "Each feature replaced by its training median or mode...",
  "limitations": ["..."]
}
```

The estimate is stored only after inference succeeds, so a rejected request never
leaves a row claiming a model answered it.

Units: `grade` is the dataset's native 0–20 value, kept so the artifact's own scale
stays visible. `marks` is `grade × 5`, `grade_letter` is CBSE's absolute school-level
grade, and `interval`, `distance_to_band_edge`, `contributions[].delta` and every
`context` error figure are in CBSE marks. `interval` is the estimate ± one held-out
RMSE — the same width for every student, a documented simplification (the measured
error is larger at the extremes).

### `GET /predictions/recent?limit=10`

Most recent stored predictions, newest first. `limit` is 1–50.

## Analytics

Dataset-derived; every figure is computed from the validated in-memory DataFrames.

| Endpoint | Query | Returns |
|---|---|---|
| `GET /dashboard/summary` | — | student counts, mean/median G3 per subject, actual band counts, at-risk total, predictions logged, recent predictions |
| `GET /analytics/options` | — | subjects, feature sets, breakdown attributes, band labels (the frontend's filter source) |
| `GET /analytics/distribution` | `subject?` | per-grade counts for each subject |
| `GET /analytics/correlation` | `subject` (default `mat`) | numeric labels plus the correlation matrix |
| `GET /analytics/feature-importance` | `subject`, `feature_set`, `limit` (1–20) | permutation importance rows from the artifact metadata, with the held-out row count |
| `GET /analytics/model-comparison` | — | every candidate's CV mean, std, MAE, R² with `selected` and `baseline` flags, the chosen models with test metrics, and the selection rule |
| `GET /analytics/subject-comparison` | — | rows, mean, median, pass rate, at-risk count, mean absences per subject, plus the pooling caveat |
| `GET /analytics/attributes` | `subject`, `attribute` | mean G3 and n per value of one attribute |

Bands in `dashboard/summary` describe the **actual** final grade of historical
records. The frontend labels them "Actual" so they are never confused with predicted
bands.

## Students

### `GET /students`

Query: `subject` (`mat`/`por`), `band`, `search` (student ID prefix/substring),
`sort` (`student_id`, `grade_asc`, `grade_desc`, `absences_desc`), `page` (≥1),
`page_size` (5–100, default 20).

Returns `{items, page, page_size, total, pages}`; total is 1044 unfiltered.

### `GET /students/{id}`

One record's full attributes with `G1`, `G2` and `G3` in CBSE marks, its actual band,
and the note that the ID is a generated row index rather than a pseudonym. The raw
`school` column is not exposed.

### `GET /students/{id}/predictions`

Predictions stored against that ID, newest first. Usually empty: submissions are
anonymous unless an ID was supplied.

## Persistence

SQLite, table `predictions(id, student_id, subject, feature_set, model_version, family,
predicted_g3, risk_band, payload_json, created_at)`. The path is `EDUPREDICT_DB` or
`backend/edupredict.db`; tests always override it so the development history is never
touched. Historical student records are **not** copied into SQLite — they are served
from the validated dataset, so there is one source of truth.
