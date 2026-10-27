"""Runner contract: horizon bounds, prediction result validation and export."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    MAX_HORIZON,
    PredictionResult,
    validate_horizon,
)

CUTOFF = date(2022, 12, 17)
STATIONS = ("ciudad_bolivar", "palua")
SCHEMA = {
    "model": pl.String,
    "unique_id": pl.String,
    "cutoff": pl.Date,
    "ds": pl.Date,
    "horizon_step": pl.Int32,
    "y_pred": pl.Float64,
}


def _frame(
    horizon: int = DEFAULT_HORIZON,
    *,
    stations: tuple[str, ...] = STATIONS,
    model: str = "naive",
) -> pl.DataFrame:
    rows = [
        {
            "model": model,
            "unique_id": station,
            "cutoff": CUTOFF,
            "ds": CUTOFF + timedelta(days=step),
            "horizon_step": step,
            "y_pred": 10.0 + step,
        }
        for station in stations
        for step in range(1, horizon + 1)
    ]
    return pl.DataFrame(rows, schema=SCHEMA)


def test_horizon_defaults_and_bounds() -> None:
    assert DEFAULT_HORIZON == MAX_HORIZON == 14
    assert validate_horizon(1) == 1
    assert validate_horizon(DEFAULT_HORIZON) == DEFAULT_HORIZON
    for invalid in (0, -1, MAX_HORIZON + 1, 14.0, True):
        with pytest.raises(ValueError, match="horizonte"):
            validate_horizon(invalid)  # type: ignore[arg-type]


def test_prediction_result_accepts_a_full_grid() -> None:
    result = PredictionResult.from_frame(_frame(), runner="naive", horizon=14)

    assert result.runner == "naive"
    assert result.horizon == 14
    assert result.frame.columns == list(SCHEMA)
    assert result.frame.height == len(STATIONS) * 14
    assert result.frame["cutoff"].unique().to_list() == [CUTOFF]


def test_prediction_result_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="columnas"):
        PredictionResult.from_frame(
            _frame().drop("y_pred"), runner="naive", horizon=14
        )


def test_prediction_result_rejects_unknown_horizon_steps() -> None:
    frame = _frame(horizon=13)
    with pytest.raises(ValueError, match="pasos"):
        PredictionResult.from_frame(frame, runner="naive", horizon=14)


def test_prediction_result_rejects_dates_that_do_not_match_the_step() -> None:
    frame = _frame().with_columns(
        pl.when(pl.col("horizon_step") == 1)
        .then(pl.lit(CUTOFF + timedelta(days=2)))
        .otherwise(pl.col("ds"))
        .alias("ds")
    )
    with pytest.raises(ValueError, match="cutoff"):
        PredictionResult.from_frame(frame, runner="naive", horizon=14)


def test_prediction_result_rejects_duplicate_rows() -> None:
    frame = pl.concat([_frame(), _frame().head(1)])
    with pytest.raises(ValueError, match="duplicad"):
        PredictionResult.from_frame(frame, runner="naive", horizon=14)


def test_prediction_result_rejects_null_or_non_finite_values() -> None:
    nulls = _frame().with_columns(
        pl.when(pl.col("horizon_step") == 3)
        .then(None)
        .otherwise(pl.col("y_pred"))
        .alias("y_pred")
    )
    with pytest.raises(ValueError, match="nulos"):
        PredictionResult.from_frame(nulls, runner="naive", horizon=14)

    infinite = _frame().with_columns(
        pl.when(pl.col("horizon_step") == 3)
        .then(float("inf"))
        .otherwise(pl.col("y_pred"))
        .alias("y_pred")
    )
    with pytest.raises(ValueError, match="finit"):
        PredictionResult.from_frame(infinite, runner="naive", horizon=14)


def test_prediction_result_requires_one_cutoff() -> None:
    other = _frame().with_columns(
        pl.lit(CUTOFF + timedelta(days=1)).alias("cutoff")
    )
    frame = pl.concat([_frame(), other])
    with pytest.raises(ValueError, match="Un solo corte"):
        PredictionResult.from_frame(frame, runner="naive", horizon=14)


def test_write_uses_the_runner_folder_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "orinoco_forecast.models.base.predictions_dir", lambda: tmp_path
    )
    result = PredictionResult.from_frame(_frame(), runner="naive", horizon=14)

    written = result.write()

    assert written == tmp_path / "naive" / "predictions.parquet"
    assert written.is_file()
    assert pl.read_parquet(written).height == result.frame.height


def test_write_accepts_a_directory_or_an_exact_file(tmp_path: Path) -> None:
    result = PredictionResult.from_frame(_frame(), runner="naive", horizon=14)

    directory = tmp_path / "custom"
    assert result.write(directory) == directory / "predictions.parquet"
    assert (directory / "predictions.parquet").is_file()

    exact = tmp_path / "other.parquet"
    assert result.write(exact) == exact
    assert exact.is_file()


def test_write_rejects_an_unsafe_runner_name() -> None:
    with pytest.raises(ValueError, match="runner"):
        PredictionResult.from_frame(_frame(), runner="naive/../x", horizon=14)
