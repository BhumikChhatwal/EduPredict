# Model card — EduPredict 1.0.0

Every number in this card was read from `backend/ml/artifacts/models/*.json`, which the
serializer wrote from the same fitted pipelines the API serves. Nothing here is
estimated, rounded up, or copied from a tutorial.

- Artifact version: `1.0.0`, metadata schema `1`
- Trained: 2026-11-22 (UTC)
- Runtime recorded with the artifacts: scikit-learn 1.9.1, NumPy 2.5.5, pandas 2.3.3,
  Python 3.12.3
- Dataset: UCI 320 *Student Performance*, `student-mat.csv` (395 rows) and
  `student-por.csv` (649 rows), SHA-256 recorded in each artifact.

---

## 1. What the model does

Predicts `G3`, the final course grade on a 0–20 scale, for one student in one subject.
Four pipelines are served:

| Subject | Variant | Family | Parameters | Selection |
|---|---|---|---|---|
| Mathematics | `early` | RandomForest | `n_estimators=500`, `max_depth=12`, `min_samples_leaf=4` | tuned |
| Mathematics | `with_grades` | RandomForest | `n_estimators=500`, `max_depth=None`, `min_samples_leaf=4` | tuned |
| Portuguese | `early` | RandomForest | `n_estimators=500`, `max_depth=12`, `min_samples_leaf=4` | tuned |
| Portuguese | `with_grades` | LinearRegression | library defaults | untuned |

`early` is the product default: it never sees `G1` or `G2`, so it can answer before any
assessment has happened. `with_grades` adds `G1`, `G2` and two derived columns
(`grade_mean`, `grade_trend`); it is accurate precisely because it restates two earlier
grades, so it is always reported next to `early` and never on its own.

Two subjects, two models: the two CSV files are separate populations from the same two
schools, plausibly including the same students. They are never pooled and never
concatenated.

## 2. Measured performance

Held-out test rows: 79 (`mat`), 130 (`por`), scored once on a fixed, stratified 80/20
split (seed 42). CV figures are 5-fold on the training rows only.

| Model | CV RMSE | Mean-baseline CV RMSE | Test RMSE | Test MAE | Test R² | Within 2 marks | Worst error |
|---|---|---|---|---|---|---|---|
| `mat` / `early` | 3.856 | 4.590 | 4.086 | 3.143 | 0.196 | 44.3% | 11.20 |
| `mat` / `with_grades` | 1.591 | 4.590 | 1.638 | 1.129 | 0.871 | 88.6% | 5.98 |
| `por` / `early` | 2.553 | 3.167 | 3.249 | 2.146 | 0.095 | 63.8% | 14.65 |
| `por` / `with_grades` | 1.299 | 3.167 | 1.371 | 0.947 | 0.839 | 92.3% | 8.50 |

Reading these honestly:

- The `early` models beat the mean baseline, but narrowly on R². `mat/early` explains
  about a fifth of the variance in the held-out rows; `por/early` about a tenth. They
  are weak early-warning signals, not a ranking of students.
- The `with_grades` models are accurate because two of their inputs are the same
  course's earlier assessments. That is useful for "where will this land" questions
  and nearly useless as a lever.
- Test RMSE is above CV RMSE for `mat/early` (4.09 vs 3.86). With 79 test rows that
  gap is within sampling noise, but the held-out figure is the one to trust because it
  was not used for selection.
- No prediction needed clipping to the 0–20 range in any of the four models.

### 2.1 Error by predicted band

Held-out rows grouped by the band the model assigned; bias is predicted minus actual.

| Model | Critical (n / MAE / bias) | Watch | On track | Strong |
|---|---|---|---|---|
| `mat` / `early` | 26 / 4.43 / +4.36 | 33 / 1.55 / −0.79 | 15 / 3.28 / −3.28 | 5 / 6.56 / −6.56 |
| `mat` / `with_grades` | 26 / 1.36 / +0.43 | 33 / 1.27 / −0.89 | 15 / 0.63 / −0.14 | 5 / 0.49 / −0.29 |
| `por` / `early` | 20 / 5.19 / +4.87 | 71 / 1.40 / +0.68 | 30 / 1.44 / −1.30 | 9 / 3.59 / −3.59 |
| `por` / `with_grades` | 20 / 1.55 / +0.88 | 71 / 0.83 / −0.06 | 30 / 0.85 / −0.14 | 9 / 0.85 / +0.09 |

The early models regress toward the middle: a student who goes on to fail is estimated
about 4.4–4.9 marks too high, and a top student 3.6–6.6 marks too low. The extreme
bands are small samples (5 and 9 rows in `strong`), so their error figures are
themselves noisy.

## 3. What drives the estimates

Global permutation importance on the held-out rows (`n_repeats=5`), as the increase in
RMSE when one feature is shuffled:

| Model | Top features |
|---|---|
| `mat` / `early` | `absences` +0.327, `failures` +0.246, `failures_any` +0.149, `sex` +0.086, `goout` +0.062 |
| `mat` / `with_grades` | `G2` +2.168, `grade_mean` +1.043, `absences` +0.742, `age` +0.108 |
| `por` / `early` | `higher` +0.075, `failures` +0.059, `failures_any` +0.051, `studytime` +0.031 |
| `por` / `with_grades` | `G2` +0.784, `grade_mean` +0.642, `G1` +0.429, `grade_trend` +0.222 |

Permutation importance measures dependence, not causation, and it splits credit between
correlated features: `failures` and `failures_any` are near-duplicates, and
`grade_mean` is a function of `G1` and `G2`.

Per-prediction explanations are a different method: each feature is replaced by its
training median (numeric) or mode (categorical) and the model is re-run, so the bar is
the change in that one student's estimate attributable to that attribute. These are
local, correlational marginal effects around one reference point. They are not SHAP
values and not causal claims.

## 4. Risk bands

Fixed cut-offs on the predicted grade, defined in `backend/ml/risk.py`:

| Band | Range |
|---|---|
| `critical` | under 10 |
| `watch` | 10 to under 14 |
| `on_track` | 14 to under 17 |
| `strong` | 17 and above |

These are product policy chosen for triage, not learned thresholds and not an
institutional standard. A shift of about one RMSE crosses a boundary, which is why the
interface shows the distance to the nearest edge next to the band.

## 5. Intended use and non-use

Intended: exploring how intake attributes relate to final grades; flagging students
who may benefit from support; demonstrating an honest, reproducible regression
workflow with reported error.

Not intended: decisions about a real student's placement, admission, discipline or
funding; comparing schools, teachers or curricula; any use on data from a population
other than the one measured.

## 6. Limitations

1. **Small, dated, local data.** 395 and 649 rows, two Portuguese secondary schools,
   2005–2006. There is no evidence that these relationships hold elsewhere or now.
2. **One split.** The held-out figures come from a single 20% sample. CV mean and fold
   spread are the better stability estimate; both are shown in the interface.
3. **Heteroscedastic error.** The measured error is far larger at the extremes of the
   scale than in the middle, and the single RMSE shown with every prediction is an
   average, not that student's error.
4. **Regress-to-the-mean bias** in the early models, quantified in §2.1.
5. **`G3 = 0` ambiguity.** Some records carry a zero that reflects withdrawal or
   non-assessment rather than a genuine score. They are kept and counted rather than
   silently dropped, and the model learns "some students receive 0" without learning
   why.
6. **No attendance prediction.** `absences` is an input count of reported absences, not
   an attendance rate and not a target.
7. **Correlational only.** No claim that changing an attribute changes the grade, and
   the sensitive lifestyle attributes (`Dalc`, `Walc`, `goout`, `romantic`) are never
   presented as levers.
8. **Cross-sectional.** No temporal modelling; an estimate of a period-end grade, not a
   trajectory.

## 7. Reproducing it

```bash
cd backend
python -m ml.training.split          # fixed, stratified 80/20 split, saved indices
python -m ml.training.compare         # candidates on identical CV folds
python -m ml.training.tune            # winning family only, same folds
python -m ml.evaluation.evaluate      # one held-out pass, figures, metrics
python -m ml.inference.serialize      # refit on train, verify identical predictions, write artifacts
```

The serializer raises if the refit does not reproduce the evaluation's predictions
exactly; that guard is what lets this card cite evaluation numbers for the served
artifacts.
