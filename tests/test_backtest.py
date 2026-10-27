"""Walk-forward protocol, multi-fold backtest and cross-fold comparison."""

import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.evaluation.backtest import build_splits

STATIONS = ("ayacucho", "caicara", "ciudad_bolivar", "palua")
START = date(2019, 1, 1)
END = date(2025, 3, 31)


def _protocol_frame(
    *,
    start: date = START,
    end: date = END,
    stations: tuple[str, ...] = STATIONS,
) -> pl.DataFrame:
    dates = pl.date_range(start, end, "1d", eager=True)
    index = pl.int_range(pl.len()).cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": station, "ds": dates}).with_columns(
            y=50.0 + offset + 0.01 * index,
        )
        for offset, station in enumerate(STATIONS)
        if station in stations
    )


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "test_start": "2023-01-01",
        "n_windows": 3,
        "step_size": 30,
    }
    config.update(overrides)
    return config


def test_build_splits_matches_the_protocol() -> None:
    splits = build_splits(_protocol_frame(), _config(), horizon=14)

    assert [split.cutoff for split in splits] == [
        date(2022, 10, 18),
        date(2022, 11, 17),
        date(2022, 12, 17),
    ]
    assert splits[-1].validation_start == date(2022, 12, 18)
    assert splits[-1].validation_end == date(2022, 12, 31)


def test_build_splits_rejects_a_missing_protocol_key() -> None:
    config = _config()
    del config["n_windows"]

    with pytest.raises(ValueError, match="n_windows"):
        build_splits(_protocol_frame(), config, horizon=14)


def test_build_splits_rejects_data_that_ends_before_the_default_cutoff() -> None:
    frame = _protocol_frame(end=date(2022, 12, 1))

    with pytest.raises(ValueError, match="no coincide"):
        build_splits(frame, _config(), horizon=14)


def test_build_splits_rejects_an_incomplete_target_coverage() -> None:
    frame = _protocol_frame().filter(
        ~(
            (pl.col("unique_id") == "palua")
            & (pl.col("ds") >= date(2022, 11, 17))
            & (pl.col("ds") <= date(2022, 11, 20))
        )
    )

    with pytest.raises(ValueError, match="palua"):
        build_splits(frame, _config(), horizon=14)


def test_backtest_module_imports_first_without_a_cycle() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}

    completed = subprocess.run(
        [sys.executable, "-c", "import orinoco_forecast.evaluation.backtest"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_build_splits_keeps_the_step_clock_when_the_horizon_changes() -> None:
    splits = build_splits(_protocol_frame(), _config(), horizon=7)

    assert splits[-1].cutoff == date(2022, 12, 24)
    assert splits[-1].validation_end == date(2022, 12, 31)
    assert (splits[-1].cutoff - splits[-2].cutoff).days == 30


def test_station_dates_are_read_from_a_single_series() -> None:
    splits = build_splits(_protocol_frame(), _config(), horizon=14)
    first = splits[0]

    assert first.validation_start == first.cutoff + timedelta(days=1)
