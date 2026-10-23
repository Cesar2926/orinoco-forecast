"""Pruebas de los baselines Naive y SeasonalNaive."""

from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from orinoco_forecast.models.base import PREDICTION_COLUMNS
from orinoco_forecast.models.statistical.baselines import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def _series(
    start: date = date(2020, 1, 1),
    days: int = 800,
    *,
    unique_id: str = "ciudad_bolivar",
    offset: float = 0.0,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": [unique_id] * days,
            "ds": [start + timedelta(days=index) for index in range(days)],
            "y": [offset + float(index) for index in range(days)],
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def test_naive_repeats_the_last_observed_level() -> None:
    history = _series()
    cutoff = date(2020, 12, 30)
    predictions = NaiveForecaster().fit(history, cutoff=cutoff).predict(14)

    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions.height == 14
    assert predictions["model"].unique().to_list() == ["naive"]
    assert predictions["unique_id"].unique().to_list() == ["ciudad_bolivar"]
    assert predictions["cutoff"].unique().to_list() == [cutoff]
    assert predictions["horizon_step"].to_list() == list(range(1, 15))
    assert predictions["ds"].to_list() == [
        cutoff + timedelta(days=step) for step in range(1, 15)
    ]
    assert predictions["y_pred"].unique().to_list() == [364.0]


def test_naive_skips_trailing_nulls() -> None:
    history = _series(days=10).with_columns(
        pl.when(pl.col("ds") >= date(2020, 1, 8))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    predictions = NaiveForecaster().fit(history, cutoff=date(2020, 1, 10)).predict(2)

    assert predictions["y_pred"].to_list() == [6.0, 6.0]


def test_naive_fit_ignores_values_after_cutoff() -> None:
    cutoff = date(2020, 12, 30)
    history = _series()
    tampered = history.with_columns(
        pl.when(pl.col("ds") > cutoff)
        .then(pl.lit(1e9))
        .otherwise(pl.col("y"))
        .alias("y")
    )

    assert_frame_equal(
        NaiveForecaster().fit(history, cutoff=cutoff).predict(14),
        NaiveForecaster().fit(tampered, cutoff=cutoff).predict(14),
    )


def test_seasonal_naive_repeats_the_value_365_days_earlier() -> None:
    history = _series()
    cutoff = date(2021, 12, 31)
    predictions = (
        SeasonalNaiveForecaster(seasonality=365).fit(history, cutoff=cutoff).predict(14)
    )

    expected = [
        history.filter(
            pl.col("ds") == cutoff + timedelta(days=step) - timedelta(days=365)
        )["y"].item()
        for step in range(1, 15)
    ]
    assert predictions["y_pred"].to_list() == expected
    assert predictions["model"].unique().to_list() == ["seasonal_naive"]


def test_seasonal_naive_rejects_missing_seasonal_values() -> None:
    history = _series(start=date(2021, 1, 1), days=400)
    with pytest.raises(ValueError, match="no encuentra"):
        SeasonalNaiveForecaster(seasonality=365).fit(
            history, cutoff=date(2021, 12, 20)
        ).predict(14)


def test_baselines_stack_both_stations() -> None:
    history = pl.concat(
        [
            _series(unique_id="ciudad_bolivar"),
            _series(unique_id="palua", offset=100.0),
        ]
    )
    predictions = NaiveForecaster().fit(history, cutoff=date(2020, 12, 30)).predict(3)

    assert predictions.height == 6
    assert predictions["unique_id"].unique().sort().to_list() == [
        "ciudad_bolivar",
        "palua",
    ]
    palua = predictions.filter(pl.col("unique_id") == "palua")
    assert palua["y_pred"].unique().to_list() == [464.0]


def test_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        NaiveForecaster().predict(14)


def test_predict_rejects_non_positive_horizon() -> None:
    history = _series(days=10)
    model = NaiveForecaster().fit(history, cutoff=date(2020, 1, 10))
    with pytest.raises(ValueError, match="horizon"):
        model.predict(0)


def test_fit_rejects_duplicate_dates() -> None:
    history = pl.concat([_series(days=10), _series(days=10).tail(1)])
    with pytest.raises(ValueError, match="duplicad"):
        NaiveForecaster().fit(history, cutoff=date(2020, 1, 5))


def test_fit_rejects_empty_history_at_cutoff() -> None:
    history = _series(days=10)
    with pytest.raises(ValueError, match="corte"):
        NaiveForecaster().fit(history, cutoff=date(2019, 12, 31))


def test_naive_rejects_stations_without_observed_levels() -> None:
    history = _series(days=10).with_columns(pl.lit(None, dtype=pl.Float64).alias("y"))
    model = NaiveForecaster().fit(history, cutoff=date(2020, 1, 5))
    with pytest.raises(ValueError, match="niveles"):
        model.predict(2)
