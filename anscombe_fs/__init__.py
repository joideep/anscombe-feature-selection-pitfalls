"""How common filter feature-selection scores behave on Anscombe's quartet."""

from pathlib import Path

import pandas as pd

from .measures import (
    SCORES,
    chi_square,
    distance_correlation,
    fisher_exact_p,
    kendall,
    leave_one_out_range,
    mutual_information,
    mutual_information_spread,
    pearson,
    score_all,
    spearman,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "anscombe.csv"


def load_anscombe() -> pd.DataFrame:
    """Anscombe's quartet in long format with columns ``dataset, x, y``."""
    return pd.read_csv(DATA_PATH)


def score_table(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per dataset (I-IV), one column per measure."""
    df = load_anscombe() if df is None else df
    rows = {name: score_all(g["x"], g["y"]) for name, g in df.groupby("dataset")}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("dataset")


__all__ = [
    "SCORES",
    "chi_square",
    "distance_correlation",
    "fisher_exact_p",
    "kendall",
    "leave_one_out_range",
    "load_anscombe",
    "mutual_information",
    "mutual_information_spread",
    "pearson",
    "score_all",
    "score_table",
    "spearman",
]
