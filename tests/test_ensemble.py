"""Forecast averaging and the replay runner used to score the ensemble."""

from datetime import date, timedelta

import polars as pl
import pytest

from orinoco_forecast.ensemble.averaging import ReplayRunner, average_forecasts

CUTOFF = date(2022, 12, 17)


def _rows(runner: str, model: str, value: float, cutoff: date = CUTOFF) -> list[dict]:
    return [
        {
            "runner": runner,
            "model": model,
            "unique_id": "palua",
            "cutoff": cutoff,
            "ds": cutoff + timedelta(days=step),
            "horizon_step": step,
            "y_pred": value,
        }
        for step in (1, 2)
    ]


def test_seeds_are_averaged_before_runners() -> None:
    # TCN seeds average to 2.0; a plain row mean would give (1 + 3 + 6) / 3.
    frame = pl.DataFrame(
        _rows("tcn", "TCN_s1", 1.0) + _rows("tcn", "TCN_s2", 3.0) + _rows("x", "X", 6.0)
    )
    out = average_forecasts(frame)
    assert out["y_pred"].to_list() == [4.0, 4.0]
    assert set(out["model"]) == {"ENSEMBLE"}
    assert out.columns == [
        "model",
        "unique_id",
        "cutoff",
        "ds",
        "horizon_step",
        "y_pred",
    ]


def test_missing_member_is_rejected() -> None:
    other = CUTOFF - timedelta(days=30)
    frame = pl.DataFrame(
        _rows("a", "A", 1.0) + _rows("b", "B", 2.0) + _rows("a", "A", 1.0, other)
    )
    with pytest.raises(ValueError, match="Faltan miembros"):
        average_forecasts(frame)


def test_replay_runner_returns_the_cutoff_rows() -> None:
    averaged = average_forecasts(pl.DataFrame(_rows("a", "A", 1.0)))
    runner = ReplayRunner("ensemble", averaged)
    runner.fit(pl.DataFrame(), pl.DataFrame(), cutoff=CUTOFF, horizon=2)
    result = runner.predict(pl.DataFrame(), cutoff=CUTOFF, horizon=1)
    assert result.runner == "ensemble"
    assert result.frame["horizon_step"].to_list() == [1]
    with pytest.raises(ValueError, match="no hay predicciones"):
        runner.fit(pl.DataFrame(), pl.DataFrame(), cutoff=date(2020, 1, 1))
