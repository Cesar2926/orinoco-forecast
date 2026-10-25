"""Metric values checked against hand-computed cases."""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from orinoco_forecast.evaluation.metrics import evaluate_cv, kge, mae, mase, nse, rmse

Y = [1.0, 2.0, 3.0, 4.0]


def test_perfect_forecast() -> None:
    assert mae(Y, Y) == 0.0
    assert rmse(Y, Y) == 0.0
    assert nse(Y, Y) == 1.0
    assert kge(Y, Y) == pytest.approx(1.0)


def test_mae_rmse_known_errors() -> None:
    # errors 1, 0, 1, 2
    pred = [2.0, 2.0, 2.0, 2.0]
    assert mae(Y, pred) == pytest.approx(1.0)
    assert rmse(Y, pred) == pytest.approx(math.sqrt(6 / 4))


def test_nse_mean_forecast_is_zero() -> None:
    assert nse(Y, [2.5] * 4) == pytest.approx(0.0)
    assert nse(Y, [2.0, 2.0, 2.0, 2.0]) == pytest.approx(1 - 6 / 5)


def test_mase_scales_by_in_sample_naive() -> None:
    train = [1.0, 3.0, 2.0, 5.0]  # lag-1 abs diffs 2, 1, 3 -> 2; lag-2: 1, 2 -> 1.5
    pred = [2.0, 2.0, 2.0, 2.0]  # MAE 1
    assert mase(Y, pred, train) == pytest.approx(0.5)
    assert mase(Y, pred, train, season=2) == pytest.approx(1 / 1.5)


def test_mase_requires_history_longer_than_season() -> None:
    with pytest.raises(ValueError):
        mase(Y, Y, [1.0, 2.0], season=2)


def test_kge_bias_and_variability() -> None:
    # pred = y + 1: r = 1, alpha = 1, beta = 3.5 / 2.5 = 1.4
    assert kge(Y, [v + 1 for v in Y]) == pytest.approx(0.6)
    # pred = 2y: r = 1, alpha = 2, beta = 2
    assert kge(Y, [2 * v for v in Y]) == pytest.approx(1 - math.sqrt(2))


def _cv_frame() -> tuple[pl.DataFrame, pl.DataFrame]:
    start = date(2020, 1, 1)
    train = pl.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": [start + timedelta(days=i) for i in range(4)],
            "y": [1.0, 3.0, 2.0, 5.0],
        }
    )
    rows = []
    for w, cutoff in enumerate([date(2020, 1, 4), date(2020, 1, 6)]):
        for h in (1, 2):
            y = 10.0 + w + h
            rows.append(("a", cutoff + timedelta(days=h), cutoff, y, y, y + 2.0))
    cv = pl.DataFrame(
        rows, schema=["unique_id", "ds", "cutoff", "y", "good", "bad"], orient="row"
    )
    return cv, train


def test_evaluate_cv_per_horizon_and_all() -> None:
    cv, train = _cv_frame()
    out = evaluate_cv(cv, ["good", "bad"], train, season=1)
    assert out.height == 2 * 3  # 2 models x (h=1, h=2, all)
    assert set(out["horizon"]) == {"1", "2", "all"}
    good = out.filter(pl.col("model") == "good")
    assert good["mae"].to_list() == [0.0, 0.0, 0.0]
    bad_all = out.filter((pl.col("model") == "bad") & (pl.col("horizon") == "all"))
    assert bad_all["mae"].item() == pytest.approx(2.0)
    assert bad_all["mase"].item() == pytest.approx(1.0)  # naive scale = 2
    assert bad_all["n"].item() == 4
