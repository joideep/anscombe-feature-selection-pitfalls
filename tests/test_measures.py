import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anscombe_fs import (  # noqa: E402
    distance_correlation,
    leave_one_out_range,
    load_anscombe,
    pearson,
    score_table,
)


@pytest.fixture(scope="module")
def scores():
    return score_table()


def test_dataset_shape():
    df = load_anscombe()
    assert df.shape == (44, 3)
    assert sorted(df["dataset"].unique()) == ["I", "II", "III", "IV"]


def test_pearson_is_identical_across_quartet(scores):
    assert np.allclose(scores["Pearson r"], 0.816, atol=1e-3)


def test_rank_measures_disagree(scores):
    # Spearman separates the datasets that Pearson cannot.
    assert scores.loc["III", "Spearman rho"] > 0.99
    assert scores.loc["IV", "Spearman rho"] == pytest.approx(0.5)


def test_chi_square_matches_original_notebook(scores):
    # Values from the original Colab notebook (median split, Yates-corrected).
    assert scores.loc["I", "Chi-square"] == pytest.approx(2.2275, abs=1e-4)
    assert scores.loc["III", "Chi-square"] == pytest.approx(7.3364, abs=1e-4)


def test_distance_correlation_bounds():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert distance_correlation(x, 3 * x + 1) == pytest.approx(1.0)
    assert 0 <= distance_correlation(x, rng.normal(size=200)) < 0.2


def test_dataset_iv_pearson_hinges_on_one_point():
    g = load_anscombe().query("dataset == 'IV'")
    _, _, undefined = leave_one_out_range(g["x"], g["y"], pearson)
    assert undefined == 1
