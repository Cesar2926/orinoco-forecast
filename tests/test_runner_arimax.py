"""ArimaxRunner: wraps the tested ARIMAX adapter in the runner contract."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.statistical.arima import ArimaForecaster, ArimaxRunner

CUTOFF = date(2020, 1, 5)


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


def _windows(frame: pl.DataFrame, *, horizon: int = 3) -> pl.DataFrame:
    return build_windows(
        frame,
        cutoff=CUTOFF,
        input_length=3,
        horizon=horizon,
    )


def _runner() -> ArimaxRunner:
    return ArimaxRunner(ArimaForecaster(order=(1, 0, 0)))


def test_arimax_runner_returns_the_runner_name_and_horizon() -> None:
    frame = _series()
    runner = _runner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=3)

    result = runner.predict(frame, cutoff=CUTOFF, horizon=3)

    assert result.runner == "arimax"
    assert result.frame["model"].unique().to_list() == ["arimax"]
    assert result.frame.height == 3
    assert result.frame["horizon_step"].to_list() == [1, 2, 3]
    assert result.frame["y_pred"].is_finite().all()


def test_arimax_runner_ignores_values_after_the_cutoff() -> None:
    frame = _series()
    tampered = frame.with_columns(
        pl.when(pl.col("ds") > CUTOFF)
        .then(pl.lit(1e9))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    base = _runner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=3)
    other = _runner().fit(_windows(tampered), tampered, cutoff=CUTOFF, horizon=3)

    assert_frame_equal(
        base.predict(frame, cutoff=CUTOFF, horizon=3).frame,
        other.predict(tampered, cutoff=CUTOFF, horizon=3).frame,
    )


def test_arimax_runner_stacks_both_stations() -> None:
    frame = pl.concat(
        [
            _series(),
            _series(unique_id="palua", offset=10.0),
        ]
    )
    runner = _runner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=3)

    result = runner.predict(frame, cutoff=CUTOFF, horizon=3)

    assert result.frame.height == 6
    assert result.frame["unique_id"].unique().sort().to_list() == [
        "ciudad_bolivar",
        "palua",
    ]


def test_arimax_runner_predict_before_fit_fails() -> None:
    frame = _series()
    with pytest.raises(RuntimeError, match="fit"):
        _runner().predict(frame, cutoff=CUTOFF, horizon=3)


def test_arimax_runner_rejects_invalid_horizons() -> None:
    frame = _series()
    with pytest.raises(ValueError, match="horizonte"):
        _runner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=0)


def test_arimax_runner_rejects_empty_windows() -> None:
    frame = _series()
    with pytest.raises(ValueError, match="ventanas"):
        _runner().fit(_windows(frame).head(0), frame, cutoff=CUTOFF, horizon=3)


def test_arimax_runner_builds_the_configured_engine_by_default() -> None:
    runner = ArimaxRunner()
    assert isinstance(runner._engine, ArimaForecaster)
    assert runner._engine.exog is True
