# EduPredict

An intelligent student performance prediction and analytics system.

EduPredict analyses historical academic records to estimate a student's expected
final grade, identify students who may need additional academic support, and explain
which attributes influenced each estimate.

> Predictions are statistical estimates produced from a small public dataset. They
> are not guaranteed outcomes and not judgements about any student.
>
> The product is built around CBSE assessment: subjects are reported as CBSE subjects,
> results are shown as marks out of 100 with the board's grade letter, and the pass
> mark is 33. The only training data is public and Portuguese (two schools, 2005–2006),
> so the estimates are an early-warning aid for a CBSE classroom, not a validated local
> prediction; see the domain gap in
> [docs/model_card.md](docs/model_card.md#5-intended-use-and-non-use).

## Status

Version 2.0.0. The pipeline, the API and the five-view dashboard are built and tested
end to end; the numbers below come from the saved artifacts. Version 2 drops the
`school` input (a CBSE deployment has no GP/MS distinction) and reports results on the
CBSE 0–100 marks scale.

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
| 9–11 | API, persistence, tests | Complete — 115 tests, SQLite prediction history |
| 12–14 | Frontend | Complete — overview, predict, records, analytics, model |
| 15–16 | Documentation, cleanup | Complete — model card, API reference, architecture |

## Results

Held-out rows (79 Mathematics, 130 Language) scored once on a fixed, stratified 80/20
split; CV is 5-fold on the training rows. Errors are in CBSE marks out of 100. Full
detail, including per-band bias, is in [docs/model_card.md](docs/model_card.md).

| Model | CV RMSE | Mean baseline | Test RMSE | Test MAE | Test R² | Within 10 marks |
|---|---|---|---|---|---|---|
| Mathematics / `early` | 19.27 | 22.95 | 20.42 | 15.70 | 0.197 | 44.3% |
| Mathematics / `with_grades` | 7.95 | 22.95 | 7.99 | 5.48 | 0.877 | 88.6% |
| Language / `early` | 12.91 | 15.84 | 16.20 | 10.66 | 0.100 | 64.6% |
| Language / `with_grades` | 6.51 | 15.84 | 6.87 | 4.67 | 0.838 | 93.1% |

The `early` models are the honest product: no period grades, and correspondingly
modest accuracy that still beats the mean baseline. The `with_grades` models are
accurate because `G1` and `G2` are the same course's earlier assessments; they are
reported beside the early models, never instead of them.

### What the data shows

From `python -m ml.evaluation.eda`:

- Mathematics: mean 52.1 marks, sd 22.9, pass rate 84.6% at the CBSE 33 mark, 61 of 395
  records `critical`. Language: mean 59.5, sd 16.2, pass rate 96.9%, 20 of 649
  `critical`.
- `G2` correlates 0.905 (Mathematics) and 0.919 (Language) with `G3`; without it, the
  remaining attributes are weak. `failures` is the strongest negative signal
  (−0.36 / −0.39), and mean marks fall monotonically with it in Mathematics
  (56.3 → 40.6 → 31.2 → 28.5 for 0–3 failures).
- 38 Mathematics and 15 Language records have `G3 = 0` (0 marks). They are kept and
  counted: a zero here can mean withdrawal or non-assessment rather than a measured
  score.

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

Two product decisions sit on top of the raw data:

- **CBSE presentation.** The recorded 0–20 grades are multiplied by five to give marks
  out of 100, and the letter beside an estimate is CBSE's absolute school-level grade
  (A1–E), with 33 as the pass mark. The model is fitted and evaluated on the native
  scale; the conversion is display only, and `ml/cbse.py` is the single place it lives.
- **No school input.** The dataset's two schools are a Portuguese distinction with no
  CBSE equivalent, so the feature was removed from the models (v2.0.0) and retrained
  rather than filled with a default. The raw column stays in the file; the API rejects
  a `school` key.

## Known limitations

The short version: the training data is Portuguese, the product is built for CBSE
schools, and that gap is unmeasured — no CBSE records were used to fit or validate
anything. On top of that, the data is small and dated (2005–2006); the early models
regress toward the middle of the scale, over-estimating students who go on to fail and
under-estimating top students; the reported error is an average, not a per-student
figure; marks out of 100 are the dataset's 0–20 grades multiplied by five, not a
board-calibrated measurement; the risk bands are product policy, not a board standard;
attendance is never predicted
(`absences` is an input); and no result is causal.

The full, measured list — including per-band error and the `G3 = 0` ambiguity — is in
[docs/model_card.md](docs/model_card.md#6-limitations). Dataset-level constraints are
in [docs/architecture.md](docs/architecture.md#3-dataset).

## Licence

MIT — see [LICENSE](LICENSE).
