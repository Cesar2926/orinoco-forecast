"""Execution layer: windows, both runners, validation slice and statistics."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.evaluation.execution import (
    ExecutionResult,
    build_windows_for,
    default_cutoff,
    execute,
    load_history,
    run_runners,
)
from orinoco_forecast.models.statistical.baselines import (
    NaiveRunner,
    SeasonalNaiveRunner,
)

CUTOFF = date(2022, 6, 1)
HORIZON = 3
STATIONS = ("ayacucho", "caicara", "ciudad_bolivar", "palua")


def _frame(days: int = 800) -> pl.DataFrame:
    start = CUTOFF - timedelta(days=days - 30)
    dates = pl.date_range(start, CUTOFF + timedelta(days=30), "1d", eager=True)
    index = pl.int_range(pl.len()).cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": station, "ds": dates}).with_columns(
            y=50.0 + offset + 0.01 * index,
            rain_ciudad_bolivar=1.0 + index % 7,
            rain_palua=2.0 + index % 5,
            oni=0.1,
        )
        for offset, station in enumerate(STATIONS)
    )


def _windows() -> pl.DataFrame:
    frame = _frame()
    return build_windows_for(frame, cutoff=CUTOFF, horizon=HORIZON)


def test_default_cutoff_matches_the_protocol() -> None:
    cfg = load_backtest_config()
    assert default_cutoff(cfg, 14) == date(2022, 12, 17)


def test_build_windows_uses_rain_and_oni_features() -> None:
    windows = _windows()
    assert "rain_ciudad_bolivar_lag_0" in windows.columns
    assert "rain_palua_lag_0" in windows.columns
    assert "oni_lag_0" in windows.columns
    assert "target_1" in windows.columns
    assert windows["unique_id"].unique().sort().to_list() == [
        "ciudad_bolivar",
        "palua",
    ]


def test_run_runners_returns_targets_and_statistics() -> None:
    result = run_runners(
        NaiveRunner(),
        SeasonalNaiveRunner(seasonality=365),
        _frame(),
        cutoff=CUTOFF,
        horizon=HORIZON,
    )

    assert isinstance(result, ExecutionResult)
    assert result.cutoff == CUTOFF
    assert result.horizon == HORIZON
    assert result.prediction.frame.height == 2 * HORIZON
    assert set(result.prediction.frame["unique_id"]) == {
        "ciudad_bolivar",
        "palua",
    }
    assert result.prediction.runner == "naive"
    assert result.baseline.runner == "seasonal_naive"
    assert set(result.statistics.frame["model"]) == {"naive", "seasonal_naive"}
    assert set(result.statistics.frame["against"]) == {"seasonal_naive"}


def test_run_runners_rejects_an_incomplete_validation_slice() -> None:
    frame = _frame()
    last = frame.filter(pl.col("unique_id") == "palua")["ds"].max()
    with pytest.raises(ValueError, match="segmento de validación"):
        run_runners(
            NaiveRunner(),
            SeasonalNaiveRunner(seasonality=365),
            frame,
            cutoff=last - timedelta(days=1),
            horizon=HORIZON,
        )


def test_execute_runs_the_catalog_without_writing() -> None:
    result = execute(
        "naive",
        "seasonal_naive",
        frame=_frame(),
        cutoff=CUTOFF,
        horizon=HORIZON,
        write=False,
    )
    assert result.prediction.runner == "naive"
    assert result.statistics.runner == "naive"


def test_execute_rejects_structure_only_runners_before_loading_data() -> None:
    with pytest.raises(ValueError, match="estructura"):
        execute("nsmt", frame=_frame(), cutoff=CUTOFF, horizon=HORIZON, write=False)


def test_load_history_reports_a_missing_parquet(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Parquet"):
        load_history(tmp_path / "missing.parquet")


def test_clean_history_limits_period_and_fills_short_gaps() -> None:
    from datetime import date, timedelta

    from orinoco_forecast.evaluation.execution import clean_history

    days = [date(2008, 12, 30) + timedelta(days=i) for i in range(6)]
    frame = pl.DataFrame(
        {
            "unique_id": ["palua"] * 6,
            "ds": days,
            "y": [1.0, 2.0, 3.0, None, 5.0, 6.0],
        }
    )
    out = clean_history(frame)
    assert out["ds"].min() == date(2009, 1, 1)
    assert out["y"].null_count() == 0
    assert out["y"].to_list() == [3.0, 4.0, 5.0, 6.0]


def test_clean_history_drops_rows_with_unavailable_lags() -> None:
    from datetime import date, timedelta

    from orinoco_forecast.evaluation.execution import clean_history

    days = [date(2009, 1, 1) + timedelta(days=i) for i in range(4)]
    frame = pl.DataFrame(
        {
            "unique_id": ["palua"] * 4,
            "ds": days,
            "y": [1.0, 2.0, 3.0, 4.0],
            "oni_lag_30": [None, None, 0.1, 0.1],
        }
    )
    assert clean_history(frame)["ds"].min() == days[2]
