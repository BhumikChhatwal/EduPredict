# EduPredict

An intelligent student performance prediction and analytics system.

EduPredict analyses historical academic records to estimate a student's expected
final grade, identify students who may need additional academic support, and explain
which attributes influenced each estimate.

> Predictions are statistical estimates produced from a small public dataset. They
> are not guaranteed outcomes and not judgements about any student.

## Status

Early development. The offline ML pipeline is being built milestone by milestone; the
API, analytics endpoints and dashboard follow it.

| Phase | Milestone | State |
|---|---|---|
| 1 | Environment and skeleton | Complete |
| 2 | Dataset loading | Complete |
| 3 | Cleaning and validation | Complete (validated; no cleaning was required) |
| 4 | Exploratory analysis | Not started |
| 5 | Features and preprocessing | Complete |
| 6 | Model training and selection | In progress — candidates compared, tuning next |
| 7–8 | Evaluation and serialization | Not started |
| 9–11 | API, persistence, tests | Not started |
| 12–14 | Frontend | Not started |
| 15–16 | Documentation, cleanup | Not started |

## Architecture

See [docs/architecture.md](docs/architecture.md) for the system architecture, the ML
methodology, risk-band definitions, design system and the full milestone plan.

```
backend/    FastAPI service (app/) and offline ML pipeline (ml/)
frontend/   React + Vite dashboard
notebooks/  Exploratory analysis
docs/       Architecture, methodology, model card, API reference
reports/    Figures referenced by this README
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

Not yet documented in full; the model card and limitations section are written in
milestone 15, from measured results. The dataset-level limitations that constrain the
design are recorded in [docs/architecture.md](docs/architecture.md#3-dataset).

## Licence

MIT — see [LICENSE](LICENSE).
