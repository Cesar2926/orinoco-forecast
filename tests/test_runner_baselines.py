"""Naive and SeasonalNaive runners: contract, values and cutoff discipline."""

from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.statistical.baselines import (
    NaiveForecaster,
    NaiveRunner,
    SeasonalNaiveRunner,
)

CUTOFF = date(2020, 2, 20)


def _series(
    start: date = date(2020, 1, 1),
    days: int = 60,
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


def _windows(frame: pl.DataFrame, *, horizon: int = 3) -> pl.DataFrame:
    return build_windows(
        frame,
        cutoff=CUTOFF,
        input_length=3,
        horizon=horizon,
    )


def test_naive_runner_returns_one_forecast_per_step_and_station() -> None:
    frame = _series()
    runner = NaiveRunner().fit(
        _windows(frame), frame, cutoff=CUTOFF, horizon=3
    )

    result = runner.predict(frame, cutoff=CUTOFF, horizon=3)

    assert result.runner == "naive"
    assert result.horizon == 3
    assert result.frame.height == 3
    assert result.frame["model"].unique().to_list() == ["naive"]
    assert result.frame["horizon_step"].to_list() == [1, 2, 3]
    assert result.frame["ds"].to_list() == [
        CUTOFF + timedelta(days=step) for step in range(1, 4)
    ]
    assert result.frame["y_pred"].unique().to_list() == [
        float((CUTOFF - date(2020, 1, 1)).days)
    ]


def test_naive_runner_stacks_both_stations() -> None:
    frame = pl.concat(
        [
            _series(),
            _series(unique_id="palua", offset=100.0),
        ]
    )
    runner = NaiveRunner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=3)

    result = runner.predict(frame, cutoff=CUTOFF, horizon=3)

    assert result.frame.height == 6
    assert result.frame["unique_id"].unique().sort().to_list() == [
        "ciudad_bolivar",
        "palua",
    ]
    palua = result.frame.filter(pl.col("unique_id") == "palua")
    assert palua["y_pred"].unique().to_list() == [
        100.0 + (CUTOFF - date(2020, 1, 1)).days
    ]


def test_naive_runner_ignores_values_after_the_cutoff() -> None:
    frame = _series()
    tampered = frame.with_columns(
        pl.when(pl.col("ds") > CUTOFF)
        .then(pl.lit(1e9))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    base = NaiveRunner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=3)
    other = NaiveRunner().fit(_windows(tampered), tampered, cutoff=CUTOFF, horizon=3)

    assert_frame_equal(
        base.predict(frame, cutoff=CUTOFF, horizon=3).frame,
        other.predict(tampered, cutoff=CUTOFF, horizon=3).frame,
    )


def test_seasonal_naive_runner_repeats_the_value_365_days_earlier() -> None:
    start = date(2020, 1, 1)
    frame = _series(start=start, days=800)
    cutoff = date(2021, 12, 31)
    windows = build_windows(frame, cutoff=cutoff, input_length=3, horizon=3)

    runner = SeasonalNaiveRunner(seasonality=365).fit(
        windows, frame, cutoff=cutoff, horizon=3
    )
    result = runner.predict(frame, cutoff=cutoff, horizon=3)

    expected = [
        frame.filter(
            pl.col("ds") == cutoff + timedelta(days=step) - timedelta(days=365)
        )["y"].item()
        for step in range(1, 4)
    ]
    assert result.frame["model"].unique().to_list() == ["seasonal_naive"]
    assert result.frame["y_pred"].to_list() == expected


def test_runner_predict_before_fit_fails() -> None:
    frame = _series()
    with pytest.raises(RuntimeError, match="fit"):
        NaiveRunner().predict(frame, cutoff=CUTOFF, horizon=3)


def test_runner_rejects_invalid_horizons() -> None:
    frame = _series()
    for invalid in (0, 15):
        with pytest.raises(ValueError, match="horizonte"):
            NaiveRunner().fit(
                _windows(frame), frame, cutoff=CUTOFF, horizon=invalid
            )


def test_runner_rejects_empty_training_windows() -> None:
    frame = _series()
    empty = _windows(frame).head(0)
    with pytest.raises(ValueError, match="ventanas"):
        NaiveRunner().fit(empty, frame, cutoff=CUTOFF, horizon=3)


def test_runner_rejects_windows_without_the_requested_targets() -> None:
    frame = _series()
    windows = _windows(frame).drop("target_3")
    with pytest.raises(ValueError, match="target_3"):
        NaiveRunner().fit(windows, frame, cutoff=CUTOFF, horizon=3)


def test_naive_runner_uses_the_injected_engine() -> None:
    engine = NaiveForecaster()
    runner = NaiveRunner(engine)
    assert runner._engine is engine
