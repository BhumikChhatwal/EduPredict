# EduPredict — Architecture and Delivery Plan

Status: approved. This document is the reference for every phase of the build.
Where the implementation deviates from this plan, this file is updated in the same
commit as the code change, with the reason recorded.

---

## 1. What this system is

EduPredict is an intelligent academic analytics system. It analyses historical
student academic records, estimates a student's expected final grade, identifies
students who may need additional academic support, and explains which attributes
influenced each estimate.

It is a **regression system over a small, public, cross-sectional education
dataset**, presented through a professional analytics interface. Three facts about
that dataset drive the design:

- It is small (395 and 649 rows). Evaluation discipline matters more than model
  sophistication.
- The strongest predictors (`G1`, `G2`) are themselves earlier grades. Including
  them makes the model accurate and nearly tautological; excluding them makes it
  useful and much weaker. The project reports **both** feature sets.
- A prediction of `12.4/20` with a measured error of roughly 1.5 grade points is
  an estimate with real uncertainty. The product communicates that in the
  interface, not only in a disclaimer.

Predictions are estimates. They are not guaranteed outcomes and not judgements
about any student.

---

## 2. System architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  React SPA (Vite, JS, Recharts)                                  │
│  Dashboard · Predict · Analytics · Students · Model              │
│  Talks only to /api/* — no ML logic, no dataset access           │
└───────────────────────────────┬──────────────────────────────────┘
                                │ JSON over HTTP
┌───────────────────────────────▼──────────────────────────────────┐
│  FastAPI service                                                 │
│  routes/   → thin HTTP layer (validation, status codes)          │
│  services/ → analytics queries, prediction orchestration,        │
│              record persistence, explanation generation          │
│  schemas/  → Pydantic contracts (the API's public spec)          │
│  core/     → settings, paths, logging, model-artifact lifecycle  │
└───────────────┬───────────────────────────────┬──────────────────┘
                │ loads at startup              │ reads/writes
┌───────────────▼──────────────┐   ┌────────────▼──────────────────┐
│  ML artifacts (joblib)       │   │  SQLite (edupredict.db)       │
│  pipeline + JSON metadata    │   │  student_records, predictions │
│  produced by training scripts│   │  seeded from the dataset      │
└───────────────▲──────────────┘   └───────────────────────────────┘
                │ writes
┌───────────────┴──────────────────────────────────────────────────┐
│  Offline ML pipeline (CLI, not part of the running API)          │
│  data → clean → features → split → train → evaluate → serialize  │
│  outputs: metrics.json, model.joblib, model_metadata.json, plots │
└──────────────────────────────────────────────────────────────────┘
```

The key rule: **training never happens inside the API process.** The API consumes
versioned artifacts. That allows retraining, metric diffing and model swaps without
touching request-handling code, and lets the API state exactly which model version
answered a request.

### Repository layout

```
edupredict/
├── backend/
│   ├── app/
│   │   ├── main.py                  # app factory, lifespan, router mounting, CORS
│   │   ├── core/config.py           # paths, model version source, settings
│   │   ├── core/db.py               # sqlite3 connection, schema init, helpers
│   │   ├── core/model_store.py      # loads artifact once, predict + metadata
│   │   ├── routes/                  # health, predictions, analytics, students, model
│   │   ├── services/                # analytics, prediction, explanation, records
│   │   └── schemas/                 # Pydantic contracts per domain
│   ├── ml/
│   │   ├── config.py                # SEED, paths, column groups, split ratios
│   │   ├── data/load.py             # read raw CSVs, validate schema, assign IDs
│   │   ├── data/raw/                # the two UCI CSVs (public, committed)
│   │   ├── preprocessing/           # clean.py, features.py, pipeline.py
│   │   ├── training/                # split.py, train.py, tune.py
│   │   ├── evaluation/              # metrics.py, plots.py
│   │   ├── artifacts/               # model.joblib, metrics.json, metadata
│   │   └── reports/                 # generated figures + comparison tables
│   ├── scripts/seed_db.py           # anonymized records → SQLite
│   ├── tests/                       # pytest: pipeline, API contract, banding
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/              # MetricCard, ChartFrame, DataTable, ...
│   │   ├── pages/                   # Dashboard, Predict, Analytics, Students, Model
│   │   ├── services/api.js          # single fetch wrapper, typed error surface
│   │   ├── hooks/                   # useApi, usePrediction
│   │   └── styles/                  # tokens.css, global.css, *.module.css
│   └── package.json
├── notebooks/                       # 01_exploratory_analysis.ipynb
├── reports/                         # figures referenced by the README
├── docs/                            # architecture, ML methodology, model card, API
├── .gitignore
└── README.md
```

Deviations from the initial sketch, and why:

1. `backend/ml/config.py` is the single source of truth for the random seed, column
   groups, paths and split parameters. Column lists duplicated across five files is
   the most common way projects of this shape rot.
2. `docs/model_card.md` exists so the "estimates, not judgements" claim is
   auditable rather than decorative.

---

## 3. Dataset

Source: UCI Machine Learning Repository, dataset 320 — *Student Performance*.
Citation: P. Cortez and A. Silva, "Using data mining to predict secondary school
student performance", Proceedings of the 5th Annual Future Business Technology
Conference, 2008.

| Property | Value |
|---|---|
| Files | `student-mat.csv` (395 rows), `student-por.csv` (649 rows) |
| Columns | 33 per file: 30 attributes + `G1`, `G2` (period grades) + `G3` (target) |
| Grade scale | 0–20 (Portuguese secondary scale); pass mark 10 |
| Missing values | None reported by UCI |
| Attribute types | 17 nominal/binary categorical, 13 integer |
| Collection | Questionnaires and school reports, two Portuguese secondary schools |

Attribute groups:

- Demographic: `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`
- Family / social: `Medu`, `Fedu` (0–4 ordinal), `Mjob`, `Fjob`, `guardian`,
  `famsup`, `famrel` (1–5)
- Academic behaviour: `reason`, `traveltime`, `studytime`, `failures`, `schoolsup`,
  `paid`, `activities`, `nursery`, `higher`, `internet`, `absences`
- Lifestyle: `romantic`, `freetime`, `goout`, `Dalc`, `Walc`, `health`
- Grades: `G1`, `G2`, `G3`

### Three facts that shape the ML design

**G1/G2 dominate.** UCI states that `G3` correlates strongly with `G2` and `G1`
because they are the same course's earlier periods. A model with `G1`+`G2` reports
a high R² and tells a teacher nothing new. A model without them is a genuine early
warning tool and is far weaker. Both are built and reported.

**The two files are not independent samples.** Same schools, plausibly overlapping
students, two subjects, no student ID to join on. Concatenating them into 1044 rows
would place the same student on both sides of a split and inflate the score.
Therefore: **two separate per-subject models** (`mat`, `por`), compared side by side.
This also gives the subject comparison view a legitimate basis.

**`G3 = 0` is ambiguous.** Some `student-mat` records have `G3 = 0`, reflecting
withdrawal or non-assessment rather than a genuine zero score. The count is measured
from the data in Phase 2, not asserted. The records are kept (dropping them silently
is a modelling decision disguised as cleaning), counted, and documented as a
systematic bias: the model learns that some students receive 0 without learning why.

### Documented limitations

- Cross-sectional, single country, two schools, 2005–2006 era. No claim of
  generalisation to other institutions, curricula or countries.
- No attendance prediction: `absences` is an input, not a target, and it is a
  reported absence count, not an attendance rate.
- No temporal modelling: an estimate of a period-end grade from current attributes,
  not a trajectory forecast.
- Grades are discrete and capped at 20, so residuals near the ceiling are
  structurally compressed.
- Test sets of roughly 79 (`mat`) and 130 (`por`) rows: a single split carries wide
  uncertainty, so CV mean ± std is the primary reported figure.
- Correlational only. Feature importance is not causation, and the lifestyle
  attributes (`Dalc`, `Walc`, `goout`) are sensitive and easily misread, so the
  interface never presents them as levers.

---

## 4. Machine learning design

### 4.1 Task definition

Regression on `G3` (0–20), per subject, with two feature sets:

| Variant | Features | Purpose | Honest framing |
|---|---|---|---|
| `with_grades` | all attributes + `G1`, `G2` | "Given two periods of results, where will this student land?" | High accuracy, low actionable value |
| `early` | all attributes, **no** `G1`, `G2` | "At intake, who needs attention?" | Lower accuracy, real decision value |

`early` is the product default; `with_grades` metrics are shown next to it for
contrast.

### 4.2 Pipeline stages

1. **Load and validate** — read CSV, assert the exact expected 33 columns and
   dtypes, coerce, fail loudly on schema drift.
2. **Clean** — per-attribute range validation, duplicate-row report, `G3 = 0` count
   reported. No invented imputation story: the dataset has no NaNs, and a guard
   raises if unexpected nulls appear.
3. **Feature engineering** — each feature justified and unit-tested:
   - `parent_edu_mean`, `parent_edu_max`, `parent_edu_gap` from `Medu`/`Fedu`.
   - `alc_total = Dalc + Walc` (reduces collinearity, one interpretable load).
   - `support_total = schoolsup + famsup + paid` (0–3).
   - `failures_any` — binarised past-failure flag; the threshold effect matters more
     than the count.
   - `with_grades` only: `grade_mean = (G1+G2)/2`, `grade_trend = G2 - G1`.
4. **Encoding** — one shared `ColumnTransformer`:
   - nominal → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
   - ordinal (`Medu`, `Fedu`, `studytime`, `traveltime`, `famrel`, `freetime`,
     `goout`, `Dalc`, `Walc`, `health`) → passed through as ordered integers
   - numeric → passed through; `StandardScaler` lives inside the linear model's own
     pipeline, because scaling is meaningless for trees.
5. **Split** — 80/20 holdout, `random_state=42`, stratified on binned `G3`
   (0–9 / 10–13 / 14–16 / 17–20). Split indices saved.
6. **Cross-validation** — `KFold(5, shuffle=True, random_state=42)` on the training
   portion only; identical folds for every candidate; test set opened once.
7. **Training** — four candidates, see 4.3.
8. **Evaluation** — MAE, MSE, RMSE, R², plus a `DummyRegressor(mean)` baseline.
9. **Serialization** — the full fitted `Pipeline` via joblib, plus
   `model_metadata.json` (family, hyperparameters, feature set, subject, row count,
   dataset SHA-256, scikit-learn version, training timestamp, CV and test metrics,
   semantic version). The API serves this verbatim for the Model page.

### 4.3 Candidate models

| Model | Why included | Expected behaviour (to be measured, not assumed) |
|---|---|---|
| `LinearRegression` | Interpretable floor. If a tree barely beats it, the complexity is unjustified. | Underfits non-linear interactions; sensitive to ordinal-as-linear assumptions |
| `DecisionTreeRegressor` | Maximally interpretable; demonstrates the bias–variance trade-off concretely. | High variance on 300–500 training rows |
| `RandomForestRegressor` | Variance reduction by bagging; robust to irrelevant features and input scale. | Usually strongest here; importance estimates biased toward high-cardinality features, so permutation importance is used |
| `GradientBoostingRegressor` | Sequential residual fitting, strong on small tabular data. | Competitive, needs care with `n_estimators`/`learning_rate` on 300 rows |

Deliberately excluded: neural networks (dataset too small), XGBoost/LightGBM (added
dependency for marginal gain at this sample size), stacking (complexity without a
defensible payoff).

**Tuning** — `GridSearchCV` (5-fold, same seed) on the selected family only, with
small grids. Tuning all four families on 300 rows mostly tunes noise.

**Selection** — lowest CV RMSE, tie-broken by MAE and CV std. RMSE because it is in
grade points and penalises the large errors that matter near the pass threshold. R²
is reported for interpretability, not used for selection.

**No fabricated numbers.** Every metric in the README is copied from a `metrics.json`
produced by a real run. Mediocre numbers are reported as-is, with reasoning.

### 4.4 Risk banding

Derived from the predicted grade on the 0–20 scale (pass mark 10):

| Band | Predicted G3 | Meaning |
|---|---|---|
| `critical` | < 10 | Below the pass threshold |
| `watch` | 10 – 13 | Passing but marginal |
| `on_track` | 14 – 16 | Comfortable |
| `strong` | ≥ 17 | High |

The interface states, in place, that these thresholds are **product policy**, not
learned from data and not validated against an institutional standard, and that a
prediction within roughly one RMSE of a boundary can change band. The API returns the
band together with the model's measured RMSE so the frontend can flag predictions
near a cut-off.

### 4.5 Explanations

- **Global** — `sklearn.inspection.permutation_importance` on the held-out set,
  aggregated per feature into `metrics.json`. Documented caveat: unstable with
  correlated features.
- **Per prediction** — ablation contributions: replace one feature with the
  training-set reference (median for numeric, mode for categorical), re-predict, and
  attribute the difference. Positive means the value pushed the estimate up. This is
  an approximation of a marginal effect, labelled as such in the API response and the
  interface — not a SHAP value.

SHAP is listed under future improvements rather than added now: it is a real
dependency with subtle interpretation, and the ablation method can be explained
end to end.

### 4.6 Reproducibility contract

Single `SEED = 42` in `backend/ml/config.py`, propagated to every split, CV fold and
estimator. Dataset hash recorded in model metadata. scikit-learn version pinned.
Training twice yields an identical `metrics.json` except for the timestamp.

---

## 5. Backend design

FastAPI. No ORM — stdlib `sqlite3` with parameterized queries and a thin repository
module. No authentication: nothing personal is stored beyond what is already public,
and login is not a feature of this product.

**Startup (lifespan):** resolve artifact path, load the pipeline once into a
`ModelStore`, load metadata, initialise the SQLite schema if absent. Fail fast with a
clear message if artifacts are missing — an API that silently starts without a model
is worse than one that refuses to start.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness, model-loaded flag, model version |
| GET | `/api/model/info` | Model, dataset, metrics, training date, version, feature set |
| POST | `/api/predict` | Subject + attributes (+ variant) → grade, band, contributions, RMSE context, limitations |
| GET | `/api/dashboard/summary` | Student count, mean/median G3, band distribution, recent predictions |
| GET | `/api/analytics/distribution` | Histogram bins of G3 per subject |
| GET | `/api/analytics/correlation` | Correlation matrix over numeric attributes |
| GET | `/api/analytics/feature-importance` | From `metrics.json` |
| GET | `/api/analytics/model-comparison` | All candidates' CV metrics |
| GET | `/api/analytics/subject-comparison` | `mat` vs `por` summary statistics |
| GET | `/api/analytics/attributes` | Grouped comparisons (mean G3 by attribute) |
| GET | `/api/students` | Paginated, filterable anonymized records |
| GET | `/api/students/{id}` | One record |
| GET | `/api/students/{id}/predictions` | That record's prediction history |

**Schemas:** separate Pydantic models for requests (`Field(ge=..., le=...)` bounds
matching real dataset domains, `Literal` for categoricals), responses, and domain
entities. The generated OpenAPI document is then an accurate API spec; `docs/api.md`
summarises it rather than duplicating it.

**Errors:** consistent `{"detail": "..."}` shape; 422 for validation, 503 when the
model store is unavailable, 404 for unknown student IDs; no traceback ever reaches a
client.

**Analytics caching:** dataset-derived endpoints aggregate at most 649 rows, so
`functools.lru_cache` on (subject, metric) is sufficient. No Redis, no pretence of
scale.

**Persistence:** `predictions(id, student_id NULL, subject, feature_set,
model_version, family, predicted_g3, risk_band, payload_json, created_at)`.
Anonymous submissions store `student_id = NULL`. Prediction history survives restarts,
which is what makes "recent prediction activity" truthful rather than decorative.
Historical student records are **not** copied into SQLite; they are served from the
validated in-memory DataFrames (see §9.2).

### Anonymization

The dataset contains **no student identifier**. IDs (`MAT-0001`, `POR-0142`) are
generated row indices and are documented as such — not pseudonymized real
identities. The interface shows no names, and the README states plainly that
"anonymized" here means "no identifier exists to begin with", not "identifiers were
removed". Claims are exactly as strong as the truth.

---

## 6. Frontend design

React + Vite, JavaScript, React Router, Recharts. No component library, no Tailwind,
no animation library — plain CSS with design tokens, so the interface reads as
engineered rather than generated.

### Design tokens

| Token | Value | Role |
|---|---|---|
| `--bg` | `#F5F6F8` | Page background |
| `--surface` | `#FFFFFF` | Cards, tables |
| `--border` | `#E2E5EA` | 1px hairlines |
| `--text` | `#101B2D` | Dark navy primary |
| `--text-muted` | `#5C6A7D` | Labels, secondary |
| `--accent` | `#1F5AA6` | Single restrained blue |
| `--positive` / `--warning` / `--critical` | `#1C6B4F` / `#8A6212` / `#9E2F2A` | Semantic states only |
| Radius | `6px` | Cards, inputs |
| Shadow | `0 1px 2px rgba(16,27,45,.06)` | Single elevation level |
| Spacing | 4px scale, 8px layout rhythm | |
| Type | Inter or system stack, 13–14px body, 12px uppercase labels at 0.04em, tabular numerals for metrics | |

Rules: borders instead of shadows for separation; one accent colour; no gradient
backgrounds; charts use the accent for the primary series and neutral grey for
context; every chart sits in a `ChartFrame` with title, subtitle stating sample size,
and consistent axis labelling; numbers right-aligned in tables.

### Pages

1. **Dashboard** — four metric cards (students, mean G3, at-risk count, predictions
   logged), G3 distribution histogram, risk-band horizontal bars (not a donut),
   recent predictions table. Every figure captions its n.
2. **Predict** — grouped form (academic history / family and support / lifestyle) with
   domain-constrained inputs, subject selector, and variant toggle (`early` default).
   Result panel: predicted grade with range indicator, band badge, signed
   contribution bars, and a fixed limitations block quoting the model's RMSE and the
   threshold caveat.
3. **Analytics** — distribution by subject, correlation heatmap, feature-importance
   bars, attribute comparisons, model-comparison table with CV mean ± std and test
   metrics. Recharts has no heatmap primitive, so the correlation matrix is a small
   custom SVG grid component.
4. **Students** — dense paginated table (ID, subject, age, studytime, failures,
   absences, G1, G2, G3); row click opens detail with full attributes and prediction
   history.
5. **Model Information** — everything from `/api/model/info`: family,
   hyperparameters, dataset and hash, split protocol, CV and test metrics, training
   timestamp, version, and the model card in readable form.

`services/api.js` is the only module that knows about HTTP; `hooks/useApi` owns
loading/error/empty states so no page invents its own. The Vite dev server proxies
`/api` to `http://127.0.0.1:8000`, so CORS is a production concern only.

---

## 7. Milestones

Each milestone is independently runnable, independently testable, and ends in
exactly one commit. Nothing is committed before it works.

| # | Milestone | Deliverable | Definition of done | Commit |
|---|---|---|---|---|
| 0 | Planning | `docs/architecture.md`, `.gitignore`, README skeleton | Repo initialises; venv, `node_modules`, DB and artifacts ignored | `chore: initialize project structure` |
| 1 | Environment | `backend/requirements.txt`, venv, Vite scaffold | `pip install -r` succeeds; API answers `/api/health`; `npm run dev` serves the shell | `chore: scaffold backend and frontend toolchains` |
| 2 | Data acquisition | raw CSVs, `ml/data/load.py`, anonymized IDs | Loads both files, validates 33 columns, prints shape and dtypes | `data: add student performance dataset loader` |
| 3 | Cleaning | `preprocessing/clean.py`, validation report | Range violations and `G3 = 0` counts printed and asserted; tests cover bad input | `feat: add data validation and cleaning pipeline` |
| 4 | EDA | notebook, `ml/config.py`, committed figures | Distribution, correlation, band counts, subject comparison; findings in prose | `feat: add exploratory data analysis` |
| 5 | Features | `features.py`, `pipeline.py` | Engineered features unit-tested; ColumnTransformer fits and round-trips | `feat: implement feature engineering and preprocessing` |
| 6 | Baselines | `split.py`, `train.py` (linear + tree), metrics | Fixed split reproducible; mean-baseline reported | `ml: add train-test split and baseline models` |
| 7 | Ensembles | RF + GB in the comparison | All four compared on identical CV folds; comparison table generated | `feat: train and compare ensemble regressors` |
| 8 | Evaluation | tuning, artifacts, plots | `metrics.json` and metadata written; test set touched once; artifacts reload and predict identically | `ml: add cross-validation, tuning and model serialization` |
| 9 | API core | FastAPI app, health, model info, predict | `/docs` accurate; predict returns grade, band, contributions; artifact loaded once | `feat: build prediction API with model artifact loading` |
| 10 | API analytics | analytics and students routes, SQLite, seed script | Dashboard endpoints return real aggregates; predictions persist across restart | `feat: add analytics, records and persistence endpoints` |
| 11 | API tests | pytest suite | Pipeline invariants, banding boundaries, API contract, missing-artifact failure | `test: add backend and ML pipeline tests` |
| 12 | Frontend shell | tokens, layout, routing, `api.js`, `useApi` | All five routes render; empty and error states handled | `feat: build dashboard shell and design system` |
| 13 | Dashboard + Analytics | charts wired to real endpoints | Figures render from live data with correct n and labels | `feat: implement analytics dashboard and charts` |
| 14 | Predict + Students | form, result panel, records, detail view | End-to-end prediction from the UI; contributions rendered; history visible | `feat: integrate prediction and student records UI` |
| 15 | Docs | README, model card, API doc, diagram, screenshots | Every claim traceable to code or `metrics.json`; limitations complete | `docs: document architecture, methodology and limitations` |
| 16 | Cleanup | dead code, dependency audit, final verification | Fresh clone → install → seed → train → run | `chore: final cleanup and release preparation` |

---

## 8. Working agreements

- One logical change per commit, conventional commit prefixes
  (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `data:`, `ml:`).
- The privacy and hygiene rules: no secrets or `.env` in Git, no private student data,
  no fabricated metrics, no simulated ML output in the interface.
- `G1`/`G2`-included results are never presented without the `early` results beside
  them, and no accuracy claim appears in documentation unless it is copied from a
  real `metrics.json`.

---

## 9. Implementation notes and deviations

Where the built system differs from the plan above, the difference is recorded here
rather than silently absorbed. Everything else is implemented as written.

### 9.1 EDA is a script, not a notebook

§7 milestone 4 planned `notebooks/01_exploratory_analysis.ipynb`. It is implemented as
`backend/ml/evaluation/eda.py` instead: a headless Matplotlib/Seaborn script that reads
the validated DataFrames, writes five figures to `reports/figures/`, and writes
`ml/artifacts/evaluation/eda.json` with the numbers the README quotes.

Reason: the figures have to be regenerable by a command in the same way the training
artifacts are, and a notebook diff is not reviewable. The script is the deliverable; the
notebook directory is no longer part of the layout.

### 9.2 Historical records are served from memory, not seeded into SQLite

§2 and §5 describe a `student_records` table seeded by `scripts/seed_db.py`. There is
no such table and no seed script. `app/core/dataset.py` validates both CSVs once at
startup (through the same loader the pipeline uses) and serves records from those
DataFrames; SQLite holds prediction history only.

Reason: the dataset is 1,044 rows and never changes at runtime. Copying it into a
second store would create two sources of truth for a page that claims to show "the
validated dataset", with nothing to gain. Prediction history is genuinely new state, so
it is the only thing persisted.

Consequence: `GET /students` filters and pages in memory. At this size that is a
non-issue; at a size where it is, the table becomes worth adding.

### 9.3 Styling is three stylesheets, not CSS modules

§2 planned `styles/tokens.css`, `global.css` and `*.module.css`. The implementation uses
`tokens.css`, `global.css` and one `app.css`. Classes are plain, prefixed by component
(`frame-`, `result-`, `band-`), and the design tokens are unchanged from §6.

Reason: the app has roughly a dozen presentational components and one theme. A single
readable stylesheet is easier to keep consistent than a dozen module files, and there is
no naming collision to prevent.

### 9.4 Milestones shipped in grouped commits

§7 assigns one commit per milestone. Milestones 9–11 (API, analytics and persistence,
tests) and 12–14 (frontend) were each developed and verified as a group before
committing, because the tests for the API and the pages that consume it cannot be
validated in isolation from each other. The commit prefixes and the "nothing is
committed before it works" rule still hold.

### 9.5 Artifact metadata is the single source for the Model page

`GET /api/model/info` returns the metadata the serializer wrote, with `target` and
`target_range` added. The interface renders that response; it does not hard-code model
names, parameters, metrics or training dates. Retraining changes the page.

`.joblib` files are intentionally untracked (`.gitignore`); the JSON metadata and
evaluation records are committed because they are small and are quoted in the docs. A
fresh clone therefore runs the pipeline once before starting the API, which fails fast
with a clear message if artifacts are missing.

### 9.6 All endpoints live under `/api`

§5 lists paths without the prefix. `main.py` mounts every router with `prefix="/api"`,
so `/api/health`, `/api/predict`, `/api/students` and the rest match the frontend's
proxy and the documentation.
