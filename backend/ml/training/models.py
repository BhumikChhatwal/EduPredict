"""Candidate estimators for the G3 regression, plus the mean baseline.

Every candidate here is a *constructor*, not an instance. Cross-validation fits a fresh
model per fold, so no fold can inherit fitted state from a fold it has already seen -
the most common way a cross-validation score is quietly too good.

Why exactly these four candidates (docs/architecture.md §4.3):

* `LinearRegression` - the reference for how much of the relationship is genuinely
  linear in the encoded features. If a tree cannot beat it by a useful margin, the extra
  complexity is not buying anything.
* `DecisionTree` - captures thresholds and interactions directly (a single absence
  threshold, an interaction between failures and study time), but a single deep tree is
  high variance: it fits the training rows' noise.
* `RandomForest` - averages many trees to cut that variance while keeping the
  non-linearities. Strong default on small tabular data, and its importance scores feed
  the explanation stage.
* `GradientBoosting` - shallow trees fitted sequentially to the residual, which often
  wins on small tabular problems; its weakness is that it can overfit a small training
  set if the number of trees is not controlled.

The `DummyRegressor(strategy="mean")` baseline is not a candidate: it predicts the
training mean for every row. It is the floor. A model that cannot beat it has learned
nothing, and reporting R² against that floor is what makes the other numbers meaningful.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from ml import config

BASELINE = "DummyMean"

# One-line justifications, kept beside the definitions so they cannot drift.
CANDIDATE_RATIONALE = {
    "LinearRegression": "Linear reference; scaled inputs, so coefficients are comparable.",
    "DecisionTree": "Thresholds and interactions, but a single high-variance tree.",
    "RandomForest": "Averages many trees: keeps interactions, cuts variance.",
    "GradientBoosting": "Sequential shallow trees on residuals; often best on small tables.",
}

# Forest size is fixed rather than tuned: it is a cost/accuracy knob, not a
# hypothesis, and 300 trees already averages away most of the sampling noise on
# a few hundred rows. Tuning happens on the selected family only.
FOREST_TREES = 300


def baseline() -> DummyRegressor:
    """The mean predictor every candidate must beat."""
    return DummyRegressor(strategy="mean")


def candidates() -> dict[str, Callable[[], Any]]:
    """Name -> constructor, in a stable order (dicts preserve insertion order)."""
    return {
        "LinearRegression": LinearRegression,
        "DecisionTree": lambda: DecisionTreeRegressor(random_state=config.SEED),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=FOREST_TREES,
            random_state=config.SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=config.SEED),
    }
