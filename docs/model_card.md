# Model card — EduPredict 2.0.0 (CBSE)

Every number in this card was read from `backend/ml/artifacts/models/*.json`, which the
serializer wrote from the same fitted pipelines the API serves. Nothing here is
estimated, rounded up, or copied from a tutorial.

- Artifact version: `2.0.0`, metadata schema `1`
- Trained: 2026-11-23 (UTC), after dropping the `school` input and moving the reported
  scale to CBSE marks
- Runtime recorded with the artifacts: scikit-learn 1.9.1, NumPy 2.5.5, pandas 2.3.3,
  Python 3.12.3
- Dataset: UCI 320 *Student Performance*, `student-mat.csv` (395 rows) and
  `student-por.csv` (649 rows), SHA-256 recorded in each artifact.

---

## 1. What the model does

Predicts a final subject result for one student, reported as **CBSE marks out of 100**
with the board's scholastic grade letter. Four pipelines are served:

| Subject | Variant | Family | Parameters | Selection |
|---|---|---|---|---|
| Mathematics | `early` | RandomForest | `n_estimators=500`, `max_depth=12`, `min_samples_leaf=4` | tuned |
| Mathematics | `with_grades` | RandomForest | `n_estimators=500`, `max_depth=None`, `min_samples_leaf=4` | tuned |
| Language | `early` | RandomForest | `n_estimators=500`, `max_depth=12`, `min_samples_leaf=4` | tuned |
| Language | `with_grades` | LinearRegression | library defaults | untuned |

**Subjects.** `Mathematics` is the dataset's maths cohort. `Language` is the
Portuguese-language cohort; it is not presented as English or Hindi, because neither
would be true of the records behind it.

**No school input.** The dataset distinguishes two schools (`GP`, `MS`). A CBSE
deployment has no such distinction, so the feature was removed and every model was
retrained without it — not filled with a hidden default. The API rejects a `school` key
rather than ignoring it.

`early` is the product default: it never sees `G1` or `G2`, so it can answer before any
assessment has happened. `with_grades` adds `G1`, `G2` and two derived columns
(`grade_mean`, `grade_trend`); it is accurate precisely because it restates two earlier
grades, so it is always reported next to `early` and never on its own.

**Scale.** The artifact predicts the dataset's native 0-20 course grade. CBSE marks are
that value × 5 (`ml/cbse.py`), which places the estimate on a scale a teacher in India
reads. The conversion is arithmetic, not evidence: see §5.

## 2. Measured performance

Held-out test rows: 79 (Mathematics), 130 (Language), scored once on a fixed,
stratified 80/20 split (seed 42). CV figures are 5-fold on the training rows only.
Errors are in CBSE marks (the artifact stores native points; ×5 here).

| Model | CV RMSE | Mean baseline | Test RMSE | Test MAE | Test R² | Within 10 marks | Worst error |
|---|---|---|---|---|---|---|---|
| Mathematics / `early` | 19.27 | 22.95 | 20.42 | 15.70 | 0.197 | 44.3% | 56.0 |
| Mathematics / `with_grades` | 7.95 | 22.95 | 7.99 | 5.48 | 0.877 | 88.6% | 29.9 |
| Language / `early` | 12.91 | 15.84 | 16.20 | 10.66 | 0.100 | 64.6% | 72.5 |
| Language / `with_grades` | 6.51 | 15.84 | 6.87 | 4.67 | 0.838 | 93.1% | 43.3 |

Reading these honestly:

- The `early` models beat the mean baseline, but narrowly on R². `Mathematics / early`
  explains about a fifth of the held-out variance; `Language / early` about a tenth.
  They are weak early-warning signals, not a ranking of students.
- A typical `Mathematics / early` miss is about 20 marks out of 100. That is wider than
  a CBSE grade band, so the letter is not a safe call on its own.
- The `with_grades` models are accurate because two of their inputs are earlier
  assessments of the same course. Useful for "where will this land", nearly useless as
  a lever.
- Test RMSE is above CV RMSE for both early models; with 79 and 130 test rows that gap
  is within sampling noise, and the held-out figure is the one to trust because it was
  not used for selection.
- No prediction needed clipping to the 0–20 range in any model.
- Removing `school` moved the scores by roughly 0.01–0.03 native points: the feature
  carried little signal, which is why the CBSE product can drop it without loss.

### 2.1 Error by actual band

Held-out rows grouped by the band the **actual** result falls in; bias is predicted
minus actual, in marks.

| Model | Critical | Watch | On track | Strong |
|---|---|---|---|---|
| Mathematics / `early` | n=13, MAE 33.5, bias +33.5 | n=22, 10.8, +5.5 | n=28, 7.2, −5.8 | n=16, 23.0, −23.0 |
| Mathematics / `with_grades` | n=13, 7.0, +6.7 | n=22, 6.8, −3.5 | n=28, 5.1, −3.9 | n=16, 3.3, −0.6 |
| Language / `early` | n=5, 58.5, +58.5 | n=36, 13.1, +9.7 | n=66, 4.8, +0.3 | n=23, 13.4, −13.4 |
| Language / `with_grades` | n=5, 18.2, +18.2 | n=36, 5.0, +1.2 | n=66, 3.8, −1.3 | n=23, 3.8, −0.4 |

The early models regress toward the middle: a student whose result is a failing mark is
estimated tens of marks too high, and a top student similarly too low. The `critical`
band holds only 5 Language rows, so its error figure is itself extremely noisy.

## 3. What drives the estimates

Global permutation importance on the held-out rows (`n_repeats=5`), as the increase in
RMSE when one feature is shuffled, in marks:

| Model | Top features |
|---|---|
| Mathematics / `early` | `absences` +1.66, `failures` +1.24, `failures_any` +0.74, `sex` +0.44 |
| Mathematics / `with_grades` | `G2` +10.95, `grade_mean` +5.33, `absences` +3.84 |
| Language / `early` | `higher` +0.37, `failures` +0.30, `failures_any` +0.25, `studytime` +0.25 |
| Language / `with_grades` | `G2` +3.96, `grade_mean` +3.26, `G1` +2.20, `grade_trend` +1.10 |

Permutation importance measures dependence, not causation, and it splits credit between
correlated features: `failures` and `failures_any` are near-duplicates, and
`grade_mean` is a function of `G1` and `G2`.

Per-prediction explanations are a different method: each feature is replaced by its
training median (numeric) or mode (categorical) and the model is re-run, so the bar is
the change in that one student's estimate attributable to that attribute, reported in
marks. These are local, correlational marginal effects around one reference point. They
are not SHAP values and not causal claims.

## 4. CBSE bands and grade letters

Risk bands are fixed cut-offs on **predicted CBSE marks**, defined in
`backend/ml/risk.py`. The lowest begins at the board's 33% pass mark:

| Band | Marks |
|---|---|
| `critical` | under 33 |
| `watch` | 33 to under 51 |
| `on_track` | 51 to under 71 |
| `strong` | 71 and above |

These are product triage levels, not learned thresholds and not a board standard.

The letter shown beside the estimate is CBSE's **absolute school-level** table:

| Marks | Grade |
|---|---|
| 91–100 | A1 |
| 81–90 | A2 |
| 71–80 | B1 |
| 61–70 | B2 |
| 51–60 | C1 |
| 41–50 | C2 |
| 33–40 | D |
| below 33 | E |

CBSE uses relative grading (rank within each subject) for the class X board result;
that cannot be computed for one student from one estimate, so the absolute table is
used and labelled as the school-level convention in the interface.

A shift of about one RMSE crosses at least one band, which is why the interface shows
the distance to the nearest edge next to the band.

## 5. Intended use and non-use

Intended audience: **CBSE secondary schools in India** (classes IX–XII). Every number
in this card was measured on Portuguese records; no Indian or CBSE data was used to
fit, tune or validate any pipeline.

The gaps that matter when it is used there:

- **Population.** Different curriculum, assessment pattern, language of instruction,
  class sizes and support structures. Relationships learned in two Portuguese schools
  need not transfer.
- **Assessment structure.** CBSE subject marks combine an 80-mark external paper with
  20 marks of internal assessment, and grades for the class X board result are
  relative. This model predicts a single end-of-course grade; it does not know either
  component.
- **Scale.** Multiplying by five puts an estimate on the 0-100 scale; it does not
  create CBSE-calibrated accuracy. The error figures in §2 are as true on the marks
  scale as on the native one, and no more.

Intended: exploring how intake attributes relate to final results; flagging students
who may benefit from support; demonstrating an honest, reproducible regression workflow
with reported error.

Not intended: decisions about a real student's placement, admission, discipline or
funding; comparing schools, teachers or boards; any use on data from a population other
than the one measured.

## 6. Limitations

1. **Small, dated, local data with an unmeasured domain shift.** 395 and 649 rows, two
   Portuguese secondary schools, 2005–2006, used for a CBSE product. Nothing here is
   validated for Indian students.
2. **One split.** The held-out figures come from a single 20% sample; CV mean and fold
   spread are the better stability estimate.
3. **Heteroscedastic error.** Error is far larger at the extremes of the scale, and the
   single RMSE shown with every estimate is an average, not that student's error.
4. **Regress-to-the-mean bias** in the early models, quantified in §2.1.
5. **`G3 = 0` ambiguity.** Some records carry a zero that reflects withdrawal or
   non-assessment rather than a genuine score. They are kept and counted.
6. **No attendance prediction.** `absences` is an input count of reported absences, not
   an attendance rate and not a target.
7. **Correlational only.** No claim that changing an attribute changes the result, and
   the sensitive lifestyle attributes (`Dalc`, `Walc`, `goout`, `romantic`) are never
   presented as levers.
8. **School information is discarded.** The model does not use it, so it cannot account
   for differences between schools — a deliberate product choice, not an oversight.
9. **Cross-sectional.** No temporal modelling; an estimate of an end-of-course result,
   not a trajectory.

## 7. Reproducing it

```bash
cd backend
python -m ml.training.split          # fixed, stratified 80/20 split, saved indices
python -m ml.training.compare        # candidates on identical CV folds
python -m ml.training.tune           # winning family only, same folds
python -m ml.evaluation.evaluate     # one held-out pass, figures, metrics
python -m ml.evaluation.eda          # descriptive figures
python -m ml.inference.serialize     # refit on train, verify identical predictions, write artifacts
```

The serializer raises if the refit does not reproduce the evaluation's predictions
exactly; that guard is what lets this card cite evaluation numbers for the served
artifacts.
