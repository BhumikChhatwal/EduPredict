"""Single source of truth for the ML pipeline.

Only constants live here: paths, column groups, the documented value domains of the
dataset, and split parameters. Keeping them in one module means the loader, the
training code and the API cannot disagree about what "valid" means or which columns
are features.

Value domains below come from the UCI Student Performance data dictionary
(Cortez & Silva, 2008), not from scanning the CSVs; the loader then checks the files
against them so a swapped or corrupted file is caught instead of silently modelled.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------

# One seed reused for every stochastic step so a run can be reproduced exactly.
SEED = 42

# --------------------------------------------------------------------------------------
# Paths
#
# Resolved from this file rather than the current directory, so scripts behave the same
# whether they are run as `python -m ml.training.train` from `backend/` or from a test.
# --------------------------------------------------------------------------------------

ML_DIR = Path(__file__).resolve().parent
RAW_DIR = ML_DIR / "data" / "raw"
ARTIFACTS_DIR = ML_DIR / "artifacts"
# Serialized pipelines (joblib) and their JSON metadata. Build outputs, not source:
# regenerate with `python -m ml.inference.serialize`.
MODELS_DIR = ARTIFACTS_DIR / "models"
# Figures live at the repository root, where the README and docs/architecture.md point.
REPORTS_DIR = ML_DIR.parent.parent / "reports"

# --------------------------------------------------------------------------------------
# Dataset files
#
# Two subjects, two models. They share schools and very likely students, and carry no
# student identifier, so they are never concatenated (see docs/architecture.md §3).
# --------------------------------------------------------------------------------------

CSV_DELIMITER = ";"

SUBJECT_FILES = {
    "mat": "student-mat.csv",
    "por": "student-por.csv",
}

SUBJECTS = tuple(SUBJECT_FILES)

# Rows the published dataset contains. A mismatch means the file is not the one this
# project was designed and evaluated on.
EXPECTED_ROW_COUNTS = {
    "mat": 395,
    "por": 649,
}

TARGET = "G3"

GRADE_COLUMNS = ("G1", "G2", "G3")

# Canonical column order, exactly as published.
EXPECTED_COLUMNS = (
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
)

# --------------------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------------------

DEMOGRAPHIC_COLUMNS = ("school", "sex", "age", "address", "famsize", "Pstatus")

FAMILY_COLUMNS = (
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "guardian",
    "famsup",
    "famrel",
)

ACADEMIC_BEHAVIOUR_COLUMNS = (
    "reason",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "absences",
)

LIFESTYLE_COLUMNS = ("romantic", "freetime", "goout", "Dalc", "Walc", "health")

CATEGORICAL_COLUMNS = (
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
)

# Ordinal attributes: ordered integers, so they are passed through as numbers instead of
# one-hot expanded. One-hot would throw away the order, which is real information here.
ORDINAL_COLUMNS = (
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
)

# Nominal attributes: no order, so they are one-hot encoded.
NOMINAL_COLUMNS = tuple(
    column for column in CATEGORICAL_COLUMNS if column not in ORDINAL_COLUMNS
)

NUMERIC_COLUMNS = (
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
    "G2",
    "G3",
)

# --------------------------------------------------------------------------------------
# Documented value domains (inclusive bounds)
# --------------------------------------------------------------------------------------

NUMERIC_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    # Data dictionary: n if 1 <= n < 3, else 4.
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    # Upper bound is not fixed by the dictionary; 93 is the documented ceiling.
    "absences": (0, 93),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}

CATEGORY_LEVELS = {
    "school": ("GP", "MS"),
    "sex": ("F", "M"),
    "address": ("R", "U"),
    "famsize": ("GT3", "LE3"),
    "Pstatus": ("A", "T"),
    "Mjob": ("at_home", "health", "other", "services", "teacher"),
    "Fjob": ("at_home", "health", "other", "services", "teacher"),
    "reason": ("course", "home", "other", "reputation"),
    "guardian": ("father", "mother", "other"),
    "schoolsup": ("no", "yes"),
    "famsup": ("no", "yes"),
    "paid": ("no", "yes"),
    "activities": ("no", "yes"),
    "nursery": ("no", "yes"),
    "higher": ("no", "yes"),
    "internet": ("no", "yes"),
    "romantic": ("no", "yes"),
}

# --------------------------------------------------------------------------------------
# Feature sets
#
# The `early` set is the product default: at intake there are no period grades yet.
# `with_grades` is reported beside it because G1/G2 predict G3 almost tautologically.
# --------------------------------------------------------------------------------------

EARLY_FEATURES = tuple(
    column
    for column in EXPECTED_COLUMNS
    if column not in GRADE_COLUMNS
)

WITH_GRADES_FEATURES = tuple(
    column for column in EXPECTED_COLUMNS if column != TARGET
)

FEATURE_SETS = {
    "early": EARLY_FEATURES,
    "with_grades": WITH_GRADES_FEATURES,
}

# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------

TEST_SIZE = 0.2
CV_FOLDS = 5

# Stratification is on coarse grade bands so every split sees weak, middle and strong
# students in roughly the published proportions.
STRATIFICATION_EDGES = (0, 10, 14, 17, 21)

# --------------------------------------------------------------------------------------
# Risk bands
#
# Product policy, not something the model learns: fixed cut-offs on the predicted grade,
# so the bands are explainable and identical for every subject. The API returns the
# prediction, its error estimate and this band, so the UI can show how close a case sits
# to a boundary rather than implying the band is a fact.
# --------------------------------------------------------------------------------------

RISK_BAND_EDGES = (0, 10, 14, 17, 21)
RISK_BAND_LABELS = ("critical", "watch", "on_track", "strong")
