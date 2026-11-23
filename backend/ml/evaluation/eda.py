"""Exploratory analysis of the validated dataset.

This is the descriptive counterpart to the modelling pipeline: it answers "what does
the data look like" before any estimator is involved, and writes the figures the README
and architecture document refer to. Nothing here touches the test set or trains a model.

What is measured, and why each figure earns its place:

* **Distribution of `G3`** - the target is a discrete 0-20 integer with a ceiling. A
  histogram, not a density plot, because a smooth curve over 21 possible values invents
  resolution the data does not have.
* **Correlation matrix** - linear association between numeric attributes. It is shown
  per subject and never pooled, and the `G1`/`G2`/`G3` block is strong by construction:
  they are three periods of the same course.
* **Actual risk-band shares** - how much of each subject falls in each product band,
  which is what makes an early-warning tool plausible or not.
* **Subject comparison** - mean grade per period with the spread beside it. The two
  files are separate populations; the figure exists to show they are similar in shape
  and different in level, not to rank schools.

Run it with:

    cd backend
    ./.venv/Scripts/python.exe -m ml.evaluation.eda
"""

from __future__ import annotations

import json
import sys
from typing import Any

import matplotlib

# Headless backend: this runs as a script, not inside a notebook.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ml import config  # noqa: E402
from ml.data import load  # noqa: E402

FIGURES_DIR = config.REPORTS_DIR / "figures"
SUMMARY_PATH = config.ARTIFACTS_DIR / "evaluation" / "eda.json"

# Palette matching the frontend tokens: navy ink, one blue accent, neutral rules.
INK = "#1e293b"
MUTED = "#64748b"
RULE = "#e2e8f0"
ACCENT = "#2563eb"

SUBJECT_TITLES = {"mat": "Mathematics", "por": "Portuguese"}


def _style() -> None:
    """Apply one restrained style to every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 10,
            "axes.edgecolor": RULE,
            "axes.labelcolor": MUTED,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
        }
    )


def summarise(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Descriptive statistics per subject, with no modelling and no pooling."""
    summary: dict[str, Any] = {}

    for subject, frame in frames.items():
        target = frame[config.TARGET]
        bands = pd.cut(
            target,
            bins=config.RISK_BAND_EDGES,
            labels=config.RISK_BAND_LABELS,
            right=False,
        ).value_counts(sort=False)

        summary[subject] = {
            "rows": int(len(frame)),
            "g3_mean": round(float(target.mean()), 3),
            "g3_median": round(float(target.median()), 3),
            "g3_std": round(float(target.std()), 3),
            "g3_min": int(target.min()),
            "g3_max": int(target.max()),
            "grade_zero_rows": int((target == 0).sum()),
            "pass_rate": round(float((target >= 10).mean()), 4),
            "bands": {label: int(bands.get(label, 0)) for label in config.RISK_BAND_LABELS},
            "correlation_with_g3": {
                column: round(float(frame[column].corr(target)), 3)
                for column in (
                    "G2",
                    "G1",
                    "absences",
                    "failures",
                    "studytime",
                    "goout",
                    "Medu",
                    "age",
                )
            },
            "mean_g3_by_failures": {
                str(int(value)): round(float(group[config.TARGET].mean()), 2)
                for value, group in frame.groupby("failures", observed=True)
            },
            "mean_g3_by_studytime": {
                str(int(value)): round(float(group[config.TARGET].mean()), 2)
                for value, group in frame.groupby("studytime", observed=True)
            },
        }

    return summary


def plot_distribution(frames: dict[str, pd.DataFrame]) -> str:
    """Target histogram per subject, on the same axes for direct comparison."""
    fig, axes = plt.subplots(1, len(frames), figsize=(9.5, 3.4), sharey=True)

    for axis, (subject, frame) in zip(axes, frames.items(), strict=True):
        counts = frame[config.TARGET].value_counts().sort_index()
        axis.bar(
            counts.index,
            counts.values,
            width=0.8,
            color=ACCENT,
            edgecolor="none",
        )
        axis.axvline(10, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        axis.set_title(f"{SUBJECT_TITLES[subject]} (n={len(frame)})")
        axis.set_xlabel("Final grade G3")
        axis.set_xticks(range(0, 21, 2))
        axis.grid(axis="y", color=RULE, linewidth=0.8)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Students")
    fig.text(
        0.01,
        0.01,
        "Dashed line marks the pass mark (10). Grades are integers on a 0-20 scale.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    path = FIGURES_DIR / "eda_grade_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_correlation(frames: dict[str, pd.DataFrame]) -> list[str]:
    """Correlation heatmaps of the numeric attributes, one panel per subject."""
    paths: list[str] = []

    for subject, frame in frames.items():
        numeric = frame[list(config.NUMERIC_COLUMNS)]
        correlation = numeric.corr()

        fig, axis = plt.subplots(figsize=(8.0, 6.6))
        sns.heatmap(
            correlation,
            ax=axis,
            cmap=sns.color_palette("Blues", as_cmap=True),
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"shrink": 0.7, "label": "Pearson r"},
            annot=False,
        )
        axis.set_title(f"{SUBJECT_TITLES[subject]} - correlation of numeric attributes")
        axis.tick_params(axis="x", rotation=55, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        fig.text(
            0.01,
            0.01,
            "G1, G2 and G3 are three periods of the same course, so their association is "
            "strong by construction.",
            fontsize=8,
            color=MUTED,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))

        path = FIGURES_DIR / f"eda_correlation_{subject}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

    return paths


def plot_risk_bands(summary: dict[str, Any]) -> str:
    """Share of each subject in each product risk band, horizontal by design."""
    labels = list(config.RISK_BAND_LABELS)
    positions = range(len(labels))
    height = 0.36
    offsets = {"mat": -height / 2, "por": height / 2}

    fig, axis = plt.subplots(figsize=(8.0, 3.6))
    for subject, values in summary.items():
        total = values["rows"]
        shares = [100 * values["bands"][label] / total for label in labels]
        axis.barh(
            [position + offsets[subject] for position in positions],
            shares,
            height=height,
            color=ACCENT if subject == "mat" else MUTED,
            label=f"{SUBJECT_TITLES[subject]} (n={total})",
        )

    axis.set_yticks(list(positions), [label.replace("_", " ") for label in labels])
    axis.invert_yaxis()
    axis.set_xlabel("Share of records (%)")
    axis.set_title("Actual final grade by risk band")
    axis.grid(axis="x", color=RULE, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="lower right")
    fig.text(
        0.01,
        0.01,
        "Bands are fixed product cut-offs on the observed grade, not a validated standard.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    path = FIGURES_DIR / "eda_risk_bands.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_subject_comparison(frames: dict[str, pd.DataFrame]) -> str:
    """Mean grade per period, with one standard deviation beside it."""
    periods = list(config.GRADE_COLUMNS)
    positions = range(len(periods))
    width = 0.36

    fig, axis = plt.subplots(figsize=(7.4, 3.6))
    for index, (subject, frame) in enumerate(frames.items()):
        means = [float(frame[column].mean()) for column in periods]
        deviations = [float(frame[column].std()) for column in periods]
        offset = (index - 0.5) * width
        axis.bar(
            [position + offset for position in positions],
            means,
            width=width,
            yerr=deviations,
            capsize=3,
            error_kw={"ecolor": MUTED, "elinewidth": 1},
            color=ACCENT if subject == "mat" else MUTED,
            label=f"{SUBJECT_TITLES[subject]} (n={len(frame)})",
        )

    axis.set_xticks(list(positions), periods)
    axis.set_ylabel("Mean grade")
    axis.set_ylim(0, 20)
    axis.set_title("Mean grade by period, with one standard deviation")
    axis.grid(axis="y", color=RULE, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right")
    fig.text(
        0.01,
        0.01,
        "The two subjects are separate populations and are never pooled.",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    path = FIGURES_DIR / "eda_subject_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def format_findings(summary: dict[str, Any]) -> str:
    """Plain-text findings, printed so the numbers are visible without the images."""
    lines = ["Exploratory analysis", "=" * 72]

    for subject, values in summary.items():
        bands = "  ".join(
            f"{label}={count}" for label, count in values["bands"].items()
        )
        correlations = "  ".join(
            f"{column}={value}"
            for column, value in values["correlation_with_g3"].items()
        )
        lines += [
            "",
            f"{SUBJECT_TITLES[subject]}  (n={values['rows']})",
            f"  G3 mean={values['g3_mean']:.2f}  median={values['g3_median']:.1f}  "
            f"sd={values['g3_std']:.2f}  range={values['g3_min']}-{values['g3_max']}",
            f"  pass rate (G3 >= 10) = {values['pass_rate']:.1%}  "
            f"G3 = 0 rows = {values['grade_zero_rows']}",
            f"  bands: {bands}",
            f"  correlation with G3: {correlations}",
            "  mean G3 by failures: "
            + "  ".join(f"{key}={value}" for key, value in values["mean_g3_by_failures"].items()),
        ]

    lines += [
        "",
        "G1 and G2 correlate strongly with G3 in both subjects; they are earlier periods",
        "of the same course, which is why the product default excludes them.",
        "G3 = 0 rows are kept and counted: they reflect withdrawal or non-assessment, not",
        "a measured zero, and dropping them would silently change the target.",
    ]
    return "\n".join(lines)


def add_eda_figures(frames: dict[str, pd.DataFrame], summary: dict[str, Any]) -> list[str]:
    """Write every figure and return their paths, so a caller can log them."""
    _style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    paths = [plot_distribution(frames)]
    paths += plot_correlation(frames)
    paths.append(plot_risk_bands(summary))
    paths.append(plot_subject_comparison(frames))
    return paths


def main() -> int:
    frames, _ = load.load_all()
    summary = summarise(frames)

    paths = add_eda_figures(frames, summary)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(format_findings(summary))
    print("")
    print(f"Wrote {len(paths)} figures to {FIGURES_DIR}")
    print(f"Wrote summary to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
