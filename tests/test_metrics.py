"""Pruebas de las métricas de evaluación."""

from datetime import date, timedelta
from math import sqrt

import polars as pl
import pytest

from orinoco_forecast.evaluation.metrics import (
    mae,
    mape,
    mase,
    rmse,
    seasonal_scale,
    smape,
)


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
