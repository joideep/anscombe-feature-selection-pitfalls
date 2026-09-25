"""Reproduce every table and figure in the README.

Usage (from the repository root):
    python scripts/run_all.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from anscombe_fs import (  # noqa: E402
    SCORES,
    leave_one_out_range,
    load_anscombe,
    mutual_information_spread,
    score_table,
)
from anscombe_fs.plots import quartet_grid, score_heatmap  # noqa: E402


def robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("dataset"):
        for measure, fn in SCORES.items():
            lo, hi, undefined = leave_one_out_range(g["x"], g["y"], fn)
            rows.append({"dataset": name, "measure": measure, "loo_min": lo, "loo_max": hi,
                         "loo_spread": hi - lo, "loo_undefined": undefined})
    return pd.DataFrame(rows)


def mi_stability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in df.groupby("dataset"):
        mean, std = mutual_information_spread(g["x"], g["y"], seeds=50)
        rows.append({"dataset": name, "mi_mean": mean, "mi_std": std})
    return pd.DataFrame(rows).set_index("dataset")


def main() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)

    df = load_anscombe()
    scores = score_table(df)
    robustness = robustness_table(df)
    mi = mi_stability_table(df)

    scores.round(4).to_csv(ROOT / "results" / "scores.csv")
    robustness.round(4).to_csv(ROOT / "results" / "leave_one_out.csv", index=False)
    mi.round(4).to_csv(ROOT / "results" / "mutual_information_stability.csv")

    quartet_grid(df, scores).savefig(ROOT / "figures" / "quartet_scores.png", dpi=150)
    score_heatmap(scores, list(SCORES)).savefig(ROOT / "figures" / "score_heatmap.png", dpi=150)

    pd.set_option("display.width", 160)
    print(scores.round(3), "\n")
    print(mi.round(3), "\n")
    print(robustness.pivot(index="measure", columns="dataset", values="loo_spread").round(3))


if __name__ == "__main__":
    main()
