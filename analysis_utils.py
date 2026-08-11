"""Shared helpers and visual constants for the interiority analysis notebooks.

All bootstrap resampling is performed at the book level.  The functions and
constants here are intentionally small and notebook-agnostic so analysis cells
can stay focused on data summaries and figures.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns


BOOTSTRAP_REPETITIONS = 5_000
BOOTSTRAP_SEED = 20260731

TIME_PERIOD_ORDER = ["<1840", "1840-1899", "1900+"]
PUBLICATION_PERIODS = ["Pre-1840", "1840–1899", "1900+"]

BROAD_GROUP_ORDER = [
    "domestic_only",
    "domestic_and_genre",
    "major_genre_non_domestic",
    "other_non_domestic_non_major_genre",
    "missing",
]
BROAD_GROUP_LABELS = {
    "domestic_only": "Domestic only",
    "domestic_and_genre": "Domestic–genre hybrid",
    "major_genre_non_domestic": "Non-domestic major genre",
    "other_non_domestic_non_major_genre": "Other non-domestic",
    "missing": "Missing broad group",
}
BROAD_GROUP_COLORS = {
    "domestic_only": "#D96C1F",
    "domestic_and_genre": "#C98A45",
    "major_genre_non_domestic": "#4F78A1",
    "other_non_domestic_non_major_genre": "#6F9292",
    "missing": "#D9DDE1",
}
BROAD_GROUP_MARKERS = {
    "domestic_only": "o",
    "domestic_and_genre": "D",
    "major_genre_non_domestic": "o",
    "other_non_domestic_non_major_genre": "^",
    "missing": "o",
}

AUTHOR_GENDER_COLORS = {
    "female": "#9C6F91",
    "male": "#5B8E67",
    "missing": "#B8BDC3",
}

GENRE_COLUMNS = [
    "is_domestic",
    "is_detective_mystery",
    "is_science_fiction",
    "is_adventure",
    "is_gothic",
    "is_historical",
    "is_canonical_author",
    "is_english_fiction_19c",
]
GENRE_LABELS = {
    "is_domestic": "Domestic",
    "is_detective_mystery": "Detective / mystery",
    "is_science_fiction": "Science fiction",
    "is_adventure": "Adventure",
    "is_gothic": "Gothic",
    "is_historical": "Historical",
    "is_canonical_author": "Canonical author",
    "is_english_fiction_19c": "English fiction, 19th c.",
}
SUBSTANTIVE_GENRES = [
    "is_detective_mystery",
    "is_science_fiction",
    "is_adventure",
    "is_gothic",
    "is_historical",
]

DOMESTIC_ORANGE = BROAD_GROUP_COLORS["domestic_only"]
DOMESTIC_HYBRID_ORANGE = BROAD_GROUP_COLORS["domestic_and_genre"]
NONDOMESTIC_BLUE = BROAD_GROUP_COLORS["major_genre_non_domestic"]
NONDOMESTIC_DARK = "#4F5963"
RESIDUAL_GRAY = "#AEB4BA"
LIGHT_GRAY = BROAD_GROUP_COLORS["missing"]

PERIOD_COLORS = {
    "Pre-1840": "#6F7F90",
    "1840–1899": "#374A5E",
    "1900+": "#8FA3B5",
}
PERIOD_MARKERS = {
    "Pre-1840": "o",
    "1840–1899": "s",
    "1900+": "^",
}


def configure_analysis_style() -> None:
    """Apply the shared display and plotting defaults used by the notebooks."""
    pd.set_option("display.max_columns", 100)
    pd.set_option("display.max_rows", 100)
    sns.set_theme(style="whitegrid", context="notebook")


def author_gender_value(values: pd.Series) -> str:
    """Return an author's modal recorded gender, or ``missing``."""
    observed = values.dropna()
    if observed.empty:
        return "missing"
    return observed.mode().iloc[0]


def _cell_rng(*key_parts):
    key = "|".join(str(part) for part in key_parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "little")
    return np.random.default_rng((BOOTSTRAP_SEED + offset) % (2**32))


def sample_status(n: int) -> str:
    if n >= 20:
        return "standard_n_ge_20"
    if n >= 10:
        return "limited_n_10_19"
    if n > 0:
        return "mean_only_n_lt_10"
    return "no_data"


def bootstrap_mean_summary(values, *seed_key):
    """Mean and fixed-seed percentile bootstrap CI for book-level values."""
    scores = (
        pd.to_numeric(pd.Series(values), errors="coerce")
        .dropna()
        .to_numpy(float)
    )
    n = len(scores)
    status = sample_status(n)
    if n == 0:
        return {
            "n_books": 0,
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "sample_status": status,
            "ci_displayed": False,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }

    rng = _cell_rng("mean", *seed_key)
    bootstrap_indices = rng.integers(
        0,
        n,
        size=(BOOTSTRAP_REPETITIONS, n),
    )
    bootstrap_means = scores[bootstrap_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "n_books": n,
        "estimate": scores.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sample_status": status,
        "ci_displayed": n >= 10,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def bootstrap_difference_summary(genre_values, domestic_values, *seed_key):
    """Independently bootstrap a book-level genre-minus-Domestic mean gap."""
    genre_scores = (
        pd.to_numeric(pd.Series(genre_values), errors="coerce")
        .dropna()
        .to_numpy(float)
    )
    domestic_scores = (
        pd.to_numeric(pd.Series(domestic_values), errors="coerce")
        .dropna()
        .to_numpy(float)
    )
    n_genre = len(genre_scores)
    n_domestic = len(domestic_scores)
    comparison_n = min(n_genre, n_domestic)
    status = sample_status(comparison_n)
    if n_genre == 0 or n_domestic == 0:
        return {
            "n_genre": n_genre,
            "n_domestic": n_domestic,
            "genre_mean": genre_scores.mean() if n_genre else np.nan,
            "domestic_mean": domestic_scores.mean() if n_domestic else np.nan,
            "difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "sample_status": "no_data",
            "ci_displayed": False,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        }

    rng_genre = _cell_rng("difference_genre", *seed_key)
    rng_domestic = _cell_rng("difference_domestic", *seed_key)
    genre_indices = rng_genre.integers(
        0,
        n_genre,
        size=(BOOTSTRAP_REPETITIONS, n_genre),
    )
    domestic_indices = rng_domestic.integers(
        0,
        n_domestic,
        size=(BOOTSTRAP_REPETITIONS, n_domestic),
    )
    bootstrap_differences = (
        genre_scores[genre_indices].mean(axis=1)
        - domestic_scores[domestic_indices].mean(axis=1)
    )
    ci_low, ci_high = np.quantile(bootstrap_differences, [0.025, 0.975])
    return {
        "n_genre": n_genre,
        "n_domestic": n_domestic,
        "genre_mean": genre_scores.mean(),
        "domestic_mean": domestic_scores.mean(),
        "difference": genre_scores.mean() - domestic_scores.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sample_status": status,
        "ci_displayed": comparison_n >= 10,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def rolling_book_bootstrap(
    data,
    group_column,
    groups,
    year_column,
    score_column,
    *,
    window_years=20,
    evaluation_step=5,
    minimum_books=20,
    seed_key="rolling",
):
    """Book-level rolling means and fixed-seed percentile bootstrap CIs."""
    working = data[[group_column, year_column, score_column]].copy()
    working[year_column] = pd.to_numeric(working[year_column], errors="coerce")
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna(subset=[group_column, year_column, score_column])

    first_year = int(
        np.ceil(working[year_column].min() / evaluation_step) * evaluation_step
    )
    last_year = int(
        np.floor(working[year_column].max() / evaluation_step) * evaluation_step
    )
    evaluation_years = range(
        first_year,
        last_year + evaluation_step,
        evaluation_step,
    )
    lower_width = window_years // 2
    upper_width = window_years - lower_width

    rows = []
    for group in groups:
        group_data = working.loc[working[group_column].eq(group)]
        for evaluation_year in evaluation_years:
            window_start = evaluation_year - lower_width
            window_end_exclusive = evaluation_year + upper_width
            scores = group_data.loc[
                group_data[year_column].ge(window_start)
                & group_data[year_column].lt(window_end_exclusive),
                score_column,
            ]
            n_books = int(scores.notna().sum())
            if n_books >= minimum_books:
                summary = bootstrap_mean_summary(
                    scores,
                    seed_key,
                    group,
                    evaluation_year,
                    window_years,
                )
                estimate = summary["estimate"]
                ci_low = summary["ci_low"]
                ci_high = summary["ci_high"]
                ci_displayed = True
            else:
                estimate = np.nan
                ci_low = np.nan
                ci_high = np.nan
                ci_displayed = False

            rows.append(
                {
                    group_column: group,
                    "evaluation_year": evaluation_year,
                    "window_start": window_start,
                    "window_end_exclusive": window_end_exclusive,
                    "window_years": window_years,
                    "evaluation_step_years": evaluation_step,
                    "minimum_books": minimum_books,
                    "n_books": n_books,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_displayed": ci_displayed,
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )

    return pd.DataFrame(rows)


def save_figure(fig, figure_dir, stem):
    """Save a figure as a 300-dpi PNG and a vector PDF."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return png_path, pdf_path


def draw_period_estimate(ax, x_value, row, color, marker, domestic=False):
    """Draw a period estimate with the notebook's sample-size encoding."""
    linewidth = 2.4 if domestic else 1.5
    if row["sample_status"] == "standard_n_ge_20":
        if row["ci_displayed"]:
            ax.vlines(
                x_value,
                row["ci_low"],
                row["ci_high"],
                color=color,
                linewidth=linewidth,
            )
        ax.plot(
            x_value,
            row["estimate"],
            marker=marker,
            markersize=7.5,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.4,
            linestyle="none",
            zorder=4,
        )
    elif row["sample_status"] == "limited_n_10_19":
        ax.vlines(
            x_value,
            row["ci_low"],
            row["ci_high"],
            color=color,
            linewidth=linewidth,
        )
        ax.plot(
            x_value,
            row["estimate"],
            marker=marker,
            markersize=7.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.7,
            linestyle="none",
            zorder=4,
        )
    elif row["sample_status"] == "mean_only_n_lt_10":
        ax.plot(
            x_value,
            row["estimate"],
            marker="x",
            markersize=8,
            color=color,
            markeredgewidth=1.8,
            linestyle="none",
            zorder=4,
        )
