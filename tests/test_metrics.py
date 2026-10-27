"""Pruebas de las métricas de evaluación (casos calculados a mano)."""

import math
from datetime import date, timedelta
from math import sqrt

import numpy as np
import polars as pl
import pytest

from orinoco_forecast.evaluation.metrics import (
    evaluate_cv,
    kge,
    mae,
    mape,
    mase,
    mase_horizon,
    nse,
    relative_to_baseline,
    rmse,
    seasonal_scale,
    smape,
)

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


def test_mase_horizon_scales_by_in_sample_naive() -> None:
    train = [1.0, 3.0, 2.0, 5.0]  # lag-1 abs diffs 2, 1, 3 -> 2; lag-2: 1, 2 -> 1.5
    pred = [2.0, 2.0, 2.0, 2.0]  # MAE 1
    assert mase_horizon(Y, pred, train) == pytest.approx(0.5)
    assert mase_horizon(Y, pred, train, season=2) == pytest.approx(1 / 1.5)


def test_mase_horizon_requires_history_longer_than_season() -> None:
    with pytest.raises(ValueError):
        mase_horizon(Y, Y, [1.0, 2.0], season=2)


def test_nse_is_nan_with_one_point_or_constant_observations() -> None:
    assert math.isnan(nse([1.0], [1.0]))
    assert math.isnan(nse([2.0, 2.0], [2.0, 2.0]))


def test_kge_is_nan_with_one_point_or_constant_observations() -> None:
    assert math.isnan(kge([1.0], [1.0]))
    assert math.isnan(kge([2.0, 2.0], [2.0, 2.0]))


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
    out = evaluate_cv(cv, ["good", "bad"], train)
    assert out.height == 2 * 3  # 2 models x (h=1, h=2, all)
    assert set(out["horizon"]) == {"1", "2", "all"}
    good = out.filter(pl.col("model") == "good")
    assert good["mae"].to_list() == [0.0, 0.0, 0.0]
    bad = out.filter(pl.col("model") == "bad")
    mase_by_h = dict(zip(bad["horizon"], bad["mase"], strict=True))
    # MAE 2 at every step; lag-1 scale 2, lag-2 scale (1 + 2) / 2 = 1.5.
    assert mase_by_h["1"] == pytest.approx(1.0)
    assert mase_by_h["2"] == pytest.approx(2 / 1.5)
    assert mase_by_h["all"] == pytest.approx((1.0 + 2 / 1.5) / 2)
    bad_all = bad.filter(pl.col("horizon") == "all")
    assert bad_all["mae"].item() == pytest.approx(2.0)
    assert bad_all["n"].item() == 4


def test_evaluate_cv_naive_on_random_walk_has_mase_near_one() -> None:
    rng = np.random.default_rng(0)
    n, horizon = 5000, 14
    days = [date(2000, 1, 1) + timedelta(days=i) for i in range(n)]
    y = np.cumsum(rng.normal(size=n))
    train = pl.DataFrame({"unique_id": "rw", "ds": days, "y": y})
    rows = [
        ("rw", days[c + h], days[c], y[c + h], y[c])
        for c in range(2000, n - horizon, 15)
        for h in range(1, horizon + 1)
    ]
    cv = pl.DataFrame(
        rows, schema=["unique_id", "ds", "cutoff", "y", "Naive"], orient="row"
    )
    out = evaluate_cv(cv, ["Naive"], train)
    per_h = out.filter(pl.col("horizon") != "all")["mase"]
    assert per_h.len() == horizon
    assert per_h.to_list() == pytest.approx([1.0] * horizon, abs=0.15)
    all_mase = out.filter(pl.col("horizon") == "all")["mase"].item()
    assert all_mase == pytest.approx(per_h.mean())


def _metrics(
    model: str, maes: dict[tuple[str, str], float], n: int = 4
) -> pl.DataFrame:
    rows = [
        {"unique_id": uid, "model": model, "horizon": h, "n": n, "mae": v}
        for (uid, h), v in maes.items()
    ]
    return pl.DataFrame(rows).with_columns(
        *(pl.lit(0.0).alias(m) for m in ("rmse", "mase", "nse", "kge"))
    )


def test_relative_to_baseline_divides_by_naive_mae() -> None:
    keys = {("a", "1"): 1.0, ("a", "all"): 2.0, ("b", "all"): 3.0}
    model = _metrics("M", {k: v / 2 for k, v in keys.items()})
    baseline = pl.concat(
        [_metrics("Naive", keys), _metrics("SeasonalNaive", {k: 9.0 for k in keys})]
    )
    out = relative_to_baseline(model, baseline).sort("unique_id", "horizon")
    assert out.height == 3
    assert out["mae_naive"].to_list() == [1.0, 2.0, 3.0]
    assert out["rel_mae"].to_list() == pytest.approx([0.5, 0.5, 0.5])
    seasonal = relative_to_baseline(model, baseline, baseline_model="SeasonalNaive")
    assert seasonal["mae_naive"].to_list() == [9.0, 9.0, 9.0]


def test_relative_to_baseline_requires_baseline_rows() -> None:
    model = _metrics("M", {("a", "all"): 1.0, ("b", "all"): 1.0})
    with pytest.raises(ValueError, match="Naive"):
        relative_to_baseline(model, _metrics("SeasonalNaive", {("a", "all"): 1.0}))
    with pytest.raises(ValueError, match="b"):
        relative_to_baseline(model, _metrics("Naive", {("a", "all"): 1.0}))


def test_relative_to_baseline_rejects_different_windows() -> None:
    model = _metrics("M", {("a", "all"): 1.0}, n=4)
    with pytest.raises(ValueError, match="ventanas"):
        relative_to_baseline(model, _metrics("Naive", {("a", "all"): 1.0}, n=6))


def _history(start: date = date(2020, 1, 1), days: int = 800) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ds": [start + timedelta(days=index) for index in range(days)],
            "y": [float(index) for index in range(days)],
        },
        schema={"ds": pl.Date, "y": pl.Float64},
    )


def test_mae() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 3.0, 5.0]) == 1.0


def test_rmse() -> None:
    assert rmse([1.0, 2.0, 3.0], [1.0, 3.0, 5.0]) == pytest.approx(sqrt(5 / 3))


def test_smape_uses_the_two_in_the_numerator() -> None:
    # |1-1| y |2-0|: el segundo término satura en 2.
    assert smape([1.0, 2.0], [1.0, 0.0]) == pytest.approx(1.0)
    # Un cero contra un cero no aporta error.
    assert smape([0.0, 1.0], [0.0, 1.0]) == 0.0


def test_mape() -> None:
    assert mape([2.0, 4.0], [1.0, 3.0]) == pytest.approx(0.375)


def test_mape_rejects_values_near_zero() -> None:
    with pytest.raises(ValueError, match="cerca de cero"):
        mape([1e-12, 2.0], [1.0, 2.0])


def test_seasonal_scale_uses_training_history() -> None:
    history = _history()
    assert seasonal_scale(history, seasonality=365) == pytest.approx(365.0)


def test_mase_divides_by_the_seasonal_scale() -> None:
    history = _history()
    assert mase([800.0], [799.0], history, seasonality=365) == pytest.approx(1 / 365)


def test_mase_rejects_a_zero_scale() -> None:
    history = pl.DataFrame(
        {
            "ds": [date(2020, 1, 1) + timedelta(days=index) for index in range(10)],
            "y": [5.0] * 10,
        },
        schema={"ds": pl.Date, "y": pl.Float64},
    )
    with pytest.raises(ValueError, match="escala estacional"):
        seasonal_scale(history, seasonality=2)


def test_metrics_reject_null_and_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="nulos"):
        mae([1.0, None], [1.0, 2.0])
    with pytest.raises(ValueError, match="longitud"):
        mae([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="vacía"):
        mae([], [])


def test_metrics_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finitos"):
        mae([1.0, float("nan")], [1.0, 2.0])


def test_seasonal_scale_rejects_duplicate_dates() -> None:
    history = pl.concat([_history(days=10), _history(days=10).tail(1)])
    with pytest.raises(ValueError, match="duplicad"):
        seasonal_scale(history, seasonality=2)
