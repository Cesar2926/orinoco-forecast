"""Tests for validate_processed_dataset: structural errors raise, quality warns."""

import warnings
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.data.loading import DataQualityWarning
from orinoco_forecast.data.validation import (
    PROCESSED_SCHEMA,
    validate_processed_dataset,
)

CONFIG = {"targets": ["ciudad_bolivar", "palua"]}
START = date(2000, 1, 1)


def _frame(days: int = 5) -> pl.DataFrame:
    rows = []
    for uid in ("ciudad_bolivar", "palua"):
        for i in range(days):
            rows.append(
                {
                    "unique_id": uid,
                    "ds": START + timedelta(days=i),
                    "y": 5.0 + i,
                    "rain_ayacucho": 1.0,
                    "rain_caicara": 1.0,
                    "rain_ciudad_bolivar": 1.0,
                    "rain_palua": 1.0,
                    "oni": 0.1,
                }
            )
    return pl.DataFrame(rows, schema=PROCESSED_SCHEMA)


def _write(tmp_path: Path, frame: pl.DataFrame) -> Path:
    path = tmp_path / "processed.parquet"
    frame.write_parquet(path)
    return path


def _validate(path: Path) -> pl.DataFrame:
    return validate_processed_dataset(path, CONFIG)


def test_valid_frame_passes_unchanged(tmp_path: Path) -> None:
    frame = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        out = _validate(_write(tmp_path, frame))
    assert out.equals(frame)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _validate(tmp_path / "absent.parquet")


def test_missing_column_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Faltan columnas"):
        _validate(_write(tmp_path, _frame().drop("oni")))


def test_wrong_type_raises(tmp_path: Path) -> None:
    frame = _frame().with_columns(pl.col("oni").cast(pl.Float32))
    with pytest.raises(ValueError, match="Esquema incorrecto"):
        _validate(_write(tmp_path, frame))


def test_missing_target_raises(tmp_path: Path) -> None:
    frame = _frame().filter(pl.col("unique_id") != "palua")
    with pytest.raises(ValueError, match="Faltan targets"):
        _validate(_write(tmp_path, frame))


def test_duplicates_warn(tmp_path: Path) -> None:
    frame = _frame()
    frame = pl.concat([frame, frame.head(1)]).sort("unique_id", "ds")
    with pytest.warns(DataQualityWarning, match="duplicadas"):
        _validate(_write(tmp_path, frame))


def test_unsorted_warns(tmp_path: Path) -> None:
    frame = _frame().reverse()
    with pytest.warns(DataQualityWarning, match="no está ordenado"):
        _validate(_write(tmp_path, frame))


def test_missing_days_warn(tmp_path: Path) -> None:
    frame = _frame().filter(pl.col("ds") != START + timedelta(days=2))
    with pytest.warns(DataQualityWarning, match="días ausentes"):
        _validate(_write(tmp_path, frame))


def test_null_target_warns(tmp_path: Path) -> None:
    frame = _frame().with_columns(
        pl.when(pl.col("ds") == START).then(None).otherwise(pl.col("y")).alias("y")
    )
    with pytest.warns(DataQualityWarning, match="valores nulos en y"):
        _validate(_write(tmp_path, frame))


def test_null_covariate_warns(tmp_path: Path) -> None:
    frame = _frame().with_columns(
        pl.when(pl.col("ds") == START).then(None).otherwise(pl.col("oni")).alias("oni")
    )
    with pytest.warns(DataQualityWarning, match="oni: .* nulos en la covariable"):
        _validate(_write(tmp_path, frame))
