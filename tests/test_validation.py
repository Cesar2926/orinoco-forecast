"""Validation script: recompute statistics from exported predictions."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.evaluation.validation import read_predictions, validate_predictions
from orinoco_forecast.models.base import PredictionResult

CUTOFF = date(2022, 6, 1)
HORIZON = 3
STATIONS = ("ciudad_bolivar", "palua")
SCHEMA = {
    "model": pl.String,
    "unique_id": pl.String,
    "cutoff": pl.Date,
    "ds": pl.Date,
    "horizon_step": pl.Int32,
    "y_pred": pl.Float64,
}


def _frame() -> pl.DataFrame:
    start = CUTOFF - timedelta(days=20)
    dates = pl.date_range(start, CUTOFF + timedelta(days=10), "1d", eager=True)
    return pl.concat(
        pl.DataFrame({"unique_id": station, "ds": dates}).with_columns(
            y=10.0 + offset + pl.int_range(pl.len()).cast(pl.Float64)
        )
        for offset, station in enumerate(STATIONS)
    )


def _result(model: str, *, bias: float, horizon: int = HORIZON) -> PredictionResult:
    frame = _frame()
    observed = frame.filter(
        pl.col("ds").is_between(
            CUTOFF + timedelta(days=1), CUTOFF + timedelta(days=horizon)
        )
    )
    rows = [
        {
            "model": model,
            "unique_id": row["unique_id"],
            "cutoff": CUTOFF,
            "ds": row["ds"],
            "horizon_step": (row["ds"] - CUTOFF).days,
            "y_pred": row["y"] + bias,
        }
        for row in observed.iter_rows(named=True)
    ]
    return PredictionResult.from_frame(
        pl.DataFrame(rows, schema=SCHEMA), runner=model, horizon=horizon
    )


def _write(root: Path, model: str, *, bias: float, horizon: int = HORIZON) -> None:
    (root / model).mkdir(parents=True, exist_ok=True)
    _result(model, bias=bias, horizon=horizon).write(root / model / "predictions.parquet")


def test_validate_predictions_reads_exports_and_writes_statistics(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "arimax", bias=1.0)
    _write(tmp_path, "naive", bias=2.0)

    statistics = validate_predictions(
        "arimax",
        "naive",
        root=tmp_path,
        frame=_frame(),
        cutoff=CUTOFF,
    )

    assert statistics.runner == "arimax"
    assert statistics.against == "naive"
    assert (tmp_path / "arimax" / "statistics.parquet").is_file()
    rows = {
        (row["model"], row["horizon"]): row
        for row in statistics.frame.iter_rows(named=True)
    }
    assert rows[("arimax", "all")]["mae"] == pytest.approx(1.0)
    assert rows[("naive", "all")]["mae"] == pytest.approx(2.0)
    assert rows[("arimax", "all")]["rel_mae"] == pytest.approx(0.5)


def test_validate_predictions_requires_both_files(tmp_path: Path) -> None:
    _write(tmp_path, "arimax", bias=1.0)
    with pytest.raises(FileNotFoundError, match="predicciones"):
        validate_predictions(
            "arimax",
            "naive",
            root=tmp_path,
            frame=_frame(),
            cutoff=CUTOFF,
        )


def test_validate_predictions_rejects_a_different_cutoff(tmp_path: Path) -> None:
    _write(tmp_path, "arimax", bias=1.0)
    _write(tmp_path, "naive", bias=2.0)
    with pytest.raises(ValueError, match="corte"):
        validate_predictions(
            "arimax",
            "naive",
            root=tmp_path,
            frame=_frame(),
            cutoff=CUTOFF - timedelta(days=1),
        )


def test_validate_predictions_rejects_incomplete_observations(tmp_path: Path) -> None:
    _write(tmp_path, "arimax", bias=1.0)
    _write(tmp_path, "naive", bias=2.0)
    frame = _frame().filter(pl.col("ds") <= CUTOFF + timedelta(days=1))
    with pytest.raises(ValueError, match="segmento de validación"):
        validate_predictions(
            "arimax",
            "naive",
            root=tmp_path,
            frame=frame,
            cutoff=CUTOFF,
        )


def test_validate_predictions_rejects_different_horizons(tmp_path: Path) -> None:
    _write(tmp_path, "arimax", bias=1.0, horizon=HORIZON)
    _write(tmp_path, "naive", bias=2.0, horizon=HORIZON - 1)
    with pytest.raises(ValueError, match="horizontes"):
        validate_predictions(
            "arimax",
            "naive",
            root=tmp_path,
            frame=_frame(),
            cutoff=CUTOFF,
        )


def test_read_predictions_infers_the_horizon(tmp_path: Path) -> None:
    _write(tmp_path, "arimax", bias=1.0)
    result = read_predictions(tmp_path / "arimax" / "predictions.parquet", "arimax")
    assert result.horizon == HORIZON
    assert result.frame.height == 2 * HORIZON
