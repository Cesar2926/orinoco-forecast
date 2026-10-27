"""Statistics from two runner results: per-horizon metrics and rel_mae."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.evaluation.metrics import StatisticsResult, compute_statistics
from orinoco_forecast.models.base import PredictionResult

CUTOFF = date(2022, 12, 17)
HORIZON = 3
OBSERVED = [10.0, 20.0, 30.0]
SCHEMA = {
    "model": pl.String,
    "unique_id": pl.String,
    "cutoff": pl.Date,
    "ds": pl.Date,
    "horizon_step": pl.Int32,
    "y_pred": pl.Float64,
}


def _result(
    models: dict[str, list[float]],
    *,
    runner: str,
    horizon: int = HORIZON,
) -> PredictionResult:
    rows = [
        {
            "model": model,
            "unique_id": "ciudad_bolivar",
            "cutoff": CUTOFF,
            "ds": CUTOFF + timedelta(days=step),
            "horizon_step": step,
            "y_pred": float(value),
        }
        for model, values in models.items()
        for step, value in enumerate(values, start=1)
    ]
    return PredictionResult.from_frame(
        pl.DataFrame(rows, schema=SCHEMA), runner=runner, horizon=horizon
    )


def _observed() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["ciudad_bolivar"] * HORIZON,
            "ds": [CUTOFF + timedelta(days=step) for step in range(1, HORIZON + 1)],
            "y": OBSERVED,
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def _history() -> pl.DataFrame:
    start = CUTOFF - timedelta(days=9)
    return pl.DataFrame(
        {
            "unique_id": ["ciudad_bolivar"] * 10,
            "ds": [start + timedelta(days=index) for index in range(10)],
            "y": [float(index + 1) for index in range(10)],
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def test_statistics_score_both_runners_and_rel_mae() -> None:
    first = _result({"arimax": [11.0, 21.0, 31.0]}, runner="arimax")
    second = _result({"naive": [12.0, 22.0, 32.0]}, runner="naive")

    stats = compute_statistics(
        first, second, observed=_observed(), history=_history()
    )

    assert isinstance(stats, StatisticsResult)
    assert stats.runner == "arimax"
    assert stats.against == "naive"
    assert stats.frame.columns == [
        "unique_id",
        "model",
        "horizon",
        "n",
        "mae",
        "rmse",
        "mase",
        "nse",
        "kge",
        "rel_mae",
        "against",
    ]
    rows = {
        (row["model"], row["horizon"]): row
        for row in stats.frame.iter_rows(named=True)
    }
    assert rows[("arimax", "1")]["mae"] == pytest.approx(1.0)
    assert rows[("arimax", "1")]["rel_mae"] == pytest.approx(0.5)
    assert rows[("arimax", "all")]["mae"] == pytest.approx(1.0)
    assert rows[("arimax", "all")]["rel_mae"] == pytest.approx(0.5)
    assert rows[("arimax", "all")]["n"] == 3
    assert rows[("naive", "all")]["rel_mae"] == pytest.approx(1.0)
    assert rows[("naive", "all")]["mae"] == pytest.approx(2.0)
    assert set(stats.frame["against"]) == {"naive"}


def test_statistics_average_the_baseline_across_seeds() -> None:
    first = _result(
        {"tcn_s1": [11.0, 21.0, 31.0], "tcn_s2": [13.0, 23.0, 33.0]},
        runner="tcn",
    )
    second = _result({"naive": [12.0, 22.0, 32.0]}, runner="naive")

    stats = compute_statistics(
        first, second, observed=_observed(), history=_history()
    )

    rows = {
        (row["model"], row["horizon"]): row
        for row in stats.frame.iter_rows(named=True)
    }
    assert rows[("naive", "all")]["rel_mae"] == pytest.approx(1.0)
    assert rows[("tcn_s1", "all")]["mae"] == pytest.approx(1.0)
    assert rows[("tcn_s2", "all")]["mae"] == pytest.approx(3.0)


def test_statistics_reject_different_horizons() -> None:
    first = _result({"arimax": [11.0, 21.0, 31.0]}, runner="arimax")
    second = _result({"naive": [12.0, 22.0]}, runner="naive", horizon=2)
    with pytest.raises(ValueError, match="horizontes"):
        compute_statistics(first, second, observed=_observed(), history=_history())


def test_statistics_reject_missing_observations() -> None:
    first = _result({"arimax": [11.0, 21.0, 31.0]}, runner="arimax")
    second = _result({"naive": [12.0, 22.0, 32.0]}, runner="naive")
    empty = _observed().head(0)
    with pytest.raises(ValueError, match="observaciones"):
        compute_statistics(first, second, observed=empty, history=_history())


def test_statistics_write_uses_the_runner_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _result({"arimax": [11.0, 21.0, 31.0]}, runner="arimax")
    second = _result({"naive": [12.0, 22.0, 32.0]}, runner="naive")
    stats = compute_statistics(
        first, second, observed=_observed(), history=_history()
    )
    monkeypatch.setattr(
        "orinoco_forecast.evaluation.metrics.predictions_dir", lambda: tmp_path
    )

    written = stats.write()

    assert written == tmp_path / "arimax" / "statistics.parquet"
    assert written.is_file()
