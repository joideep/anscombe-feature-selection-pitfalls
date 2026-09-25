"""Filter-style feature-relevance measures applied to a single (x, y) pair.

Every function takes two 1-D array-likes and returns a float. They are the
scores a filter feature-selection method would use to decide whether x is
worth keeping as a predictor of y.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


def pearson(x, y) -> float:
    """Pearson's r: strength of the *linear* relationship."""
    return float(stats.pearsonr(x, y)[0])


def spearman(x, y) -> float:
    """Spearman's rho: Pearson's r computed on ranks (monotonic relationship)."""
    return float(stats.spearmanr(x, y)[0])


def kendall(x, y) -> float:
    """Kendall's tau-b: concordant vs discordant pairs, corrected for ties."""
    return float(stats.kendalltau(x, y, variant="b")[0])


def mutual_information(x, y, random_state: int = 0, n_neighbors: int = 3) -> float:
    """Mutual information (nats) via the Kraskov k-NN estimator.

    The estimator adds a tiny amount of noise to break ties, so the result
    depends on ``random_state``. With only 11 points it is noticeably unstable;
    see :func:`mutual_information_spread`.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    return float(
        mutual_info_regression(x, y, n_neighbors=n_neighbors, random_state=random_state)[0]
    )


def mutual_information_spread(x, y, seeds: int = 50) -> tuple[float, float]:
    """Mean and standard deviation of MI across ``seeds`` random states."""
    values = [mutual_information(x, y, random_state=s) for s in range(seeds)]
    return float(np.mean(values)), float(np.std(values))


def median_split(values) -> np.ndarray:
    """Binarise a variable into 'high' (> median) and 'low' (<= median)."""
    values = np.asarray(values, dtype=float)
    return np.where(values > np.median(values), "high", "low")


def _contingency(x, y) -> np.ndarray:
    xb, yb = median_split(x), median_split(y)
    levels = ["low", "high"]
    return np.array([[np.sum((xb == a) & (yb == b)) for b in levels] for a in levels])


def chi_square(x, y) -> tuple[float, float]:
    """Chi-square test of independence after a median split of x and y.

    Returns ``(statistic, p_value)``. SciPy applies Yates' continuity
    correction to 2x2 tables. With 11 observations most expected counts are
    below 5, so the chi-square approximation is unreliable here; compare with
    :func:`fisher_exact_p`.
    """
    table = _contingency(x, y)
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return float("nan"), float("nan")
    statistic, p_value, _, _ = stats.chi2_contingency(table)
    return float(statistic), float(p_value)


def fisher_exact_p(x, y) -> float:
    """Fisher's exact test p-value on the same median-split 2x2 table."""
    return float(stats.fisher_exact(_contingency(x, y))[1])


def distance_correlation(x, y) -> float:
    """Szekely's distance correlation, in [0, 1]; 0 iff x and y are independent.

    Unlike Pearson it responds to non-linear dependence as well as linear.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def centred(v):
        d = np.abs(v[:, None] - v[None, :])
        return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()

    a, b = centred(x), centred(y)
    dcov2 = (a * b).mean()
    dvar = np.sqrt((a * a).mean() * (b * b).mean())
    return float(np.sqrt(max(dcov2, 0.0) / dvar)) if dvar > 0 else 0.0


SCORES = {
    "Pearson r": pearson,
    "Spearman rho": spearman,
    "Kendall tau-b": kendall,
    "Mutual information": mutual_information,
    "Distance correlation": distance_correlation,
}


def score_all(x, y) -> dict[str, float]:
    """Every measure for one (x, y) pair, as a flat dict."""
    row = {name: fn(x, y) for name, fn in SCORES.items()}
    row["Chi-square"], row["Chi-square p"] = chi_square(x, y)
    row["Fisher exact p"] = fisher_exact_p(x, y)
    return row


def leave_one_out_range(x, y, fn) -> tuple[float, float, int]:
    """Min and max of ``fn`` when each single observation is removed in turn.

    Also returns how many of those removals leave the score undefined (x
    becomes constant). A wide range, or any undefined case, means the score
    hinges on one point, which is exactly what Anscombe's datasets III and IV
    are built to expose.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        xs, ys = x[keep], y[keep]
        values.append(fn(xs, ys) if np.ptp(xs) > 0 else float("nan"))
    undefined = int(np.isnan(values).sum())
    return float(np.nanmin(values)), float(np.nanmax(values)), undefined
