# EduPredict

An intelligent student performance prediction and analytics system.

EduPredict analyses historical academic records to estimate a student's expected
final grade, identify students who may need additional academic support, and explain
which attributes influenced each estimate.

> Predictions are statistical estimates produced from a small public dataset. They
> are not guaranteed outcomes and not judgements about any student.
>
> The product is intended for Indian secondary schools. The only training data is
> public and Portuguese (two schools, 2005–2006), so the estimates are an early-warning
> aid there, not a validated local prediction; see the domain gap in
> [docs/model_card.md](docs/model_card.md#5-intended-use-and-non-use).

## Status

Version 1.0.0. The pipeline, the API and the five-view dashboard are built and tested
end to end; the numbers below come from the saved artifacts.

| Phase | Milestone | State |
|---|---|---|
| 1 | Environment and skeleton | Complete |
| 2 | Dataset loading | Complete |
| 3 | Cleaning and validation | Complete (validated; no cleaning was required) |
| 4 | Exploratory analysis | Complete — `ml/evaluation/eda.py`; 5 figures in `reports/figures/`, summary in `ml/artifacts/evaluation/eda.json` |
| 5 | Features and preprocessing | Complete — shared `ColumnTransformer`, two feature sets |
| 6 | Model training and selection | Complete — 20 candidates compared on identical folds, winner tuned |
| 7 | Evaluation on held-out rows | Complete — metrics, per-band error and 8 figures in `ml/artifacts/evaluation/` |
| 8 | Serialization and inference | Complete — 4 joblib pipelines + JSON metadata; refit reproduces Phase 7 predictions |
| 9–11 | API, persistence, tests | Complete — 93 tests, SQLite prediction history |
| 12–14 | Frontend | Complete — overview, predict, records, analytics, model |
| 15–16 | Documentation, cleanup | Complete — model card, API reference, architecture |

## Results

Held-out rows (79 maths, 130 Portuguese) scored once on a fixed, stratified 80/20
split; CV is 5-fold on the training rows. Full detail, including per-band bias, is in
[docs/model_card.md](docs/model_card.md).

| Model | CV RMSE | Mean baseline | Test RMSE | Test MAE | Test R² | Within 2 marks |
|---|---|---|---|---|---|---|
| Maths / `early` | 3.856 | 4.590 | 4.086 | 3.143 | 0.196 | 44.3% |
| Maths / `with_grades` | 1.591 | 4.590 | 1.638 | 1.129 | 0.871 | 88.6% |
| Portuguese / `early` | 2.553 | 3.167 | 3.249 | 2.146 | 0.095 | 63.8% |
| Portuguese / `with_grades` | 1.299 | 3.167 | 1.371 | 0.947 | 0.839 | 92.3% |

The `early` models are the honest product: no period grades, and correspondingly
modest accuracy that still beats the mean baseline. The `with_grades` models are
accurate because `G1` and `G2` are the same course's earlier assessments; they are
reported beside the early models, never instead of them.

### What the data shows

From `python -m ml.evaluation.eda`:

- Maths: mean G3 10.41, sd 4.58, pass rate 67.1%, 130 of 395 records in the `critical`
  band. Portuguese: mean 11.91, sd 3.23, pass rate 84.6%, 100 of 649 `critical`.
- `G2` correlates 0.905 (maths) and 0.919 (Portuguese) with `G3`; without it, the
  remaining attributes are weak. `failures` is the strongest negative signal
  (−0.36 / −0.39), and mean G3 falls monotonically with it in maths (11.25 → 8.12 →
  6.24 → 5.69 for 0–3 failures).
- 38 maths records and 15 Portuguese records have `G3 = 0`. They are kept and counted:
  a zero here can mean withdrawal or non-assessment rather than a measured score.

Figures: `reports/figures/eda_grade_distribution.png`,
`eda_correlation_mat.png`, `eda_correlation_por.png`, `eda_risk_bands.png`,
`eda_subject_comparison.png`.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the system architecture, the ML
methodology, risk-band definitions, design system and the milestone plan;
[docs/model_card.md](docs/model_card.md) for measured performance, per-band error and
limitations; and [docs/api.md](docs/api.md) for the endpoint reference.

```
backend/    FastAPI service (app/) and offline ML pipeline (ml/)
frontend/   React + Vite dashboard
docs/       Architecture, methodology, model card, API reference
reports/    Figures (exploratory and evaluation) referenced by this README
```

## Technology

Python, NumPy, pandas, scikit-learn, Matplotlib, Seaborn, joblib, FastAPI, Pydantic,
SQLite, React, Vite, Recharts.

## Setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

The API is then available at `http://127.0.0.1:8000`, with interactive documentation
at `http://127.0.0.1:8000/docs`. `GET /api/health` should return HTTP 200.

The service refuses to start without model artifacts, and the `.joblib` files are
intentionally not committed. On a fresh clone, reproduce them once:

```bash
python -m ml.training.split
python -m ml.training.compare
python -m ml.training.tune
python -m ml.evaluation.evaluate
python -m ml.evaluation.eda        # descriptive figures and eda.json
python -m ml.inference.serialize
```

The dataset CSVs are committed, so this needs no network access. `serialize` refits
the chosen pipelines, verifies the refit reproduces the evaluation's predictions
exactly, and writes the artifacts plus their metadata.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard is served at `http://127.0.0.1:5173` and proxies `/api` to the backend
on port 8000.

## Dataset

UCI Machine Learning Repository, dataset 320 — *Student Performance*: 395 mathematics
and 649 Portuguese language records from two Portuguese secondary schools, with
grades on a 0–20 scale.

P. Cortez and A. Silva, "Using data mining to predict secondary school student
performance", Proceedings of the 5th Annual Future Business Technology Conference,
2008. <https://archive.ics.uci.edu/dataset/320/student+performance>

The dataset contains no student identifiers. Any IDs used in this project are
generated row indices, not anonymized real identities.

## Known limitations

The short version: the training data is Portuguese, the product is aimed at Indian
schools, and that gap is unmeasured — no Indian records were used to fit or validate
anything. On top of that, the data is small and dated (2005–2006); the early models
regress toward the middle of the scale, over-estimating students who go on to fail and
under-estimating top students; the reported error is an average, not a per-student
figure; the target is the Portuguese 0–20 grade, not marks out of 100; the risk bands
are product policy, not an institutional standard; attendance is never predicted
(`absences` is an input); and no result is causal.

The full, measured list — including per-band error and the `G3 = 0` ambiguity — is in
[docs/model_card.md](docs/model_card.md#6-limitations). Dataset-level constraints are
in [docs/architecture.md](docs/architecture.md#3-dataset).

## Licence

MIT — see [LICENSE](LICENSE).
