"""Pruebas del adaptador ARIMA."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from orinoco_forecast.models.base import PREDICTION_COLUMNS
from orinoco_forecast.models.statistical.arima import ArimaForecaster


def _series(
    start: date = date(2019, 1, 1),
    days: int = 400,
    *,
    unique_id: str = "ciudad_bolivar",
    offset: float = 0.0,
) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    index = np.arange(days, dtype=float)
    values = (
        offset
        + 50.0
        + 0.01 * index
        + 3.0 * np.sin(2 * np.pi * index / 365)
        + rng.normal(scale=0.2, size=days)
    )
    return pl.DataFrame(
        {
            "unique_id": [unique_id] * days,
            "ds": [start + timedelta(days=int(step)) for step in index],
            "y": values,
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def test_arima_forecast_schema_and_horizon() -> None:
    history = _series()
    cutoff = date(2020, 1, 5)
    predictions = (
        ArimaForecaster(order=(1, 0, 0)).fit(history, cutoff=cutoff).predict(14)
    )

    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions.height == 14
    assert predictions["model"].unique().to_list() == ["arima"]
    assert predictions["unique_id"].unique().to_list() == ["ciudad_bolivar"]
    assert predictions["cutoff"].unique().to_list() == [cutoff]
    assert predictions["horizon_step"].to_list() == list(range(1, 15))
    assert predictions["ds"].to_list() == [
        cutoff + timedelta(days=step) for step in range(1, 15)
    ]
    assert predictions["y_pred"].is_finite().all()


def test_arima_fit_ignores_values_after_cutoff() -> None:
    history = _series(days=500)
    cutoff = date(2020, 1, 20)
    tampered = history.with_columns(
        pl.when(pl.col("ds") > cutoff)
        .then(pl.lit(1e9))
        .otherwise(pl.col("y"))
        .alias("y")
    )

    assert_frame_equal(
        ArimaForecaster(order=(1, 0, 0)).fit(history, cutoff=cutoff).predict(14),
        ArimaForecaster(order=(1, 0, 0)).fit(tampered, cutoff=cutoff).predict(14),
    )


def test_arima_tolerates_null_targets_without_imputation() -> None:
    history = _series(days=300).with_columns(
        pl.when(pl.col("ds") == date(2019, 3, 1))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    predictions = (
        ArimaForecaster(order=(1, 0, 1))
        .fit(history, cutoff=date(2019, 7, 1))
        .predict(14)
    )

    assert predictions["y_pred"].is_finite().all()


def test_arima_stacks_both_stations() -> None:
    history = pl.concat(
        [
            _series(days=200),
            _series(days=200, unique_id="palua", offset=10.0),
        ]
    )
    predictions = (
        ArimaForecaster(order=(1, 0, 0))
        .fit(history, cutoff=date(2019, 5, 1))
        .predict(3)
    )

    assert predictions.height == 6
    assert predictions["unique_id"].unique().sort().to_list() == [
        "ciudad_bolivar",
        "palua",
    ]


def test_arima_auto_order_forecasts() -> None:
    history = _series(days=120)
    predictions = (
        ArimaForecaster(auto_order=True, max_p=2, max_q=2, max_d=1)
        .fit(history, cutoff=date(2019, 3, 1))
        .predict(3)
    )

    assert predictions.height == 3
    assert predictions["y_pred"].is_finite().all()


def test_arima_auto_order_rejects_nulls() -> None:
    history = _series(days=60).with_columns(
        pl.when(pl.col("ds") == date(2019, 1, 10))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    model = ArimaForecaster(auto_order=True)
    with pytest.raises(ValueError, match="AutoARIMA no admite nulos"):
        model.fit(history, cutoff=date(2019, 2, 1))


def test_arima_rejects_station_without_observed_levels() -> None:
    history = _series(days=30).with_columns(pl.lit(None, dtype=pl.Float64).alias("y"))
    with pytest.raises(ValueError, match="niveles"):
        ArimaForecaster().fit(history, cutoff=date(2019, 1, 20))


def test_arima_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        ArimaForecaster().predict(14)


def test_arima_rejects_invalid_order() -> None:
    with pytest.raises(ValueError, match="orden"):
        ArimaForecaster(order=(1, 2))


def test_arima_rejects_negative_order_values() -> None:
    with pytest.raises(ValueError, match="orden"):
        ArimaForecaster(seasonal_order=(0, 0, 0, -1))
