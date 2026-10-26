"""Persistencia Parquet del dataset largo con features de niveles."""

from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.data.datasets import (
    STATIONS,
    ParquetDataset,
    TimeSeriesDataset,
)
from orinoco_forecast.data.exogenous import load_nasa_power, load_oni
from orinoco_forecast.data.loading import load_config
from orinoco_forecast.features.lags import RiverFeatureConfig, add_river_features

RAIN_LAGS = (14, 21, 28)
ONI_LAGS = (0, 30, 60)


def _wide(days: int = 35) -> pl.DataFrame:
    """Tabla ancha de estaciones con un nivel distinto cada día."""
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(days)]
    return pl.DataFrame(
        {
            "ds": dates,
            **{
                station: [float(100 * offset + step) for step in range(days)]
                for offset, station in enumerate(STATIONS)
            },
        }
    )


def _patch_wide(monkeypatch, dataset: TimeSeriesDataset, frame: pl.DataFrame) -> None:
    """Evita leer el Excel real: ``extract_xlsx`` devuelve el frame de prueba."""
    monkeypatch.setattr(dataset, "extract_xlsx", lambda *args, **kwargs: frame)


def test_write_parquet_writes_long_frame_with_lag_features(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "levels.parquet"
    dataset = TimeSeriesDataset(output)
    wide = _wide()
    _patch_wide(monkeypatch, dataset, wide)

    written = dataset.write_parquet()

    assert written == output
    frame = pl.read_parquet(output)
    assert frame.columns[:3] == ["unique_id", "ds", "y"]
    assert frame.height == wide.height * len(STATIONS)
    assert frame["unique_id"].n_unique() == len(STATIONS)
    assert frame.select(pl.struct("unique_id", "ds").is_duplicated().sum()).item() == 0
    assert {
        "ayacucho_lag_7",
        "ayacucho_lag_14",
        "caicara_lag_4",
        "ciudad_bolivar_lag_1",
        "d_ayacucho_lag_3",
        "d_caicara_lag_3",
        "d_ciudad_bolivar_lag_1",
    } <= set(frame.columns)
    assert "sin_doy" not in frame.columns
    assert "cos_doy" not in frame.columns
    assert frame.equals(frame.sort("unique_id", "ds"))
    # Un feature en t solo lee la fuente en t - lag.
    day = date(2020, 1, 31)
    value = frame.filter(
        (pl.col("unique_id") == "ciudad_bolivar") & (pl.col("ds") == day)
    ).item(0, "ayacucho_lag_14")
    source = wide.filter(pl.col("ds") == day - timedelta(days=14))["ayacucho"].item()
    assert value == pytest.approx(source)


def test_write_parquet_adds_rain_and_oni_from_external(tmp_path, monkeypatch) -> None:
    output = tmp_path / "levels.parquet"
    dataset = TimeSeriesDataset(output)
    _patch_wide(monkeypatch, dataset, _wide())

    dataset.write_parquet()

    frame = pl.read_parquet(output)
    rain_columns = [f"rain_{station}" for station in STATIONS]
    assert set(rain_columns) | {"oni"} <= set(frame.columns)
    assert {
        f"{column}_lag_{lag}" for column in rain_columns for lag in RAIN_LAGS
    } <= set(frame.columns)
    assert {f"oni_lag_{lag}" for lag in ONI_LAGS} <= set(frame.columns)

    settings = load_config()["exogenous"]
    nasa = load_nasa_power(
        settings["nasa_power"]["files"], settings["nasa_power"]["variable"]
    )
    oni = load_oni(
        settings["noaa_oni"]["path"],
        settings["noaa_oni"]["publication_lag_months"],
    )
    day = date(2020, 1, 31)
    row = frame.filter(
        (pl.col("unique_id") == "ciudad_bolivar") & (pl.col("ds") == day)
    ).row(0, named=True)
    expected_rain = nasa.filter(pl.col("ds") == day)
    expected_oni = oni.filter(pl.col("ds") == day)["oni"].item()
    for station in STATIONS:
        assert row[f"rain_{station}"] == pytest.approx(
            expected_rain[f"rain_{station}"].item()
        )
        lagged = nasa.filter(pl.col("ds") == day - timedelta(days=14))
        assert row[f"rain_{station}_lag_14"] == pytest.approx(
            lagged[f"rain_{station}"].item()
        )
    assert row["oni"] == pytest.approx(expected_oni)
    assert row["oni_lag_30"] == pytest.approx(
        oni.filter(pl.col("ds") == day - timedelta(days=30))["oni"].item()
    )


def test_write_parquet_can_skip_external_sources(tmp_path, monkeypatch) -> None:
    output = tmp_path / "levels.parquet"
    dataset = TimeSeriesDataset(output)
    _patch_wide(monkeypatch, dataset, _wide())

    dataset.write_parquet(include_exogenous=False)

    columns = set(pl.read_parquet(output).columns)
    assert "oni" not in columns
    assert not any(column.startswith("rain_") for column in columns)
    assert "ayacucho_lag_14" in columns


def test_write_parquet_accepts_output_override(tmp_path, monkeypatch) -> None:
    dataset = TimeSeriesDataset(tmp_path / "default.parquet")
    override = tmp_path / "override.parquet"
    wide = _wide()
    _patch_wide(monkeypatch, dataset, wide)

    written = dataset.write_parquet(override)

    assert written == override
    assert dataset.output_path == override
    assert pl.read_parquet(override).height == wide.height * len(STATIONS)


def test_write_parquet_accepts_feature_config(tmp_path, monkeypatch) -> None:
    output = tmp_path / "levels.parquet"
    dataset = TimeSeriesDataset(output)
    _patch_wide(monkeypatch, dataset, _wide())
    config = RiverFeatureConfig(
        station_lags={"ayacucho": (1,)},
        diff_lags={},
        rain_lags=(),
        oni_lags=(0,),
        calendar_terms=0,
    )

    dataset.write_parquet(feature_config=config)

    columns = set(pl.read_parquet(output).columns)
    assert "ayacucho_lag_1" in columns
    assert "ayacucho_lag_14" not in columns
    assert "sin_doy" not in columns


def test_base_can_disable_external_features() -> None:
    levels = _wide().unpivot(index="ds", variable_name="unique_id", value_name="y")
    config = RiverFeatureConfig(rain_lags=(14,), oni_lags=(0,))

    output = add_river_features(
        levels,
        config,
        include_rain=False,
        include_oni=False,
    )

    assert "ayacucho_lag_14" in output.columns
    assert "oni_lag_0" not in output.columns


def _interface_frame(days: int = 6) -> pl.DataFrame:
    """Tabla larga mínima con covariables crudas y dos rezagos."""
    levels = (
        _wide(days)
        .unpivot(index="ds", variable_name="unique_id", value_name="y")
        .sort("unique_id", "ds")
    )
    return levels.with_columns(
        pl.lit(0.5).alias("rain_ayacucho"),
        pl.lit(0.1).alias("oni"),
    ).with_columns(
        pl.col("y").shift(1).over("unique_id").alias("ayacucho_lag_1"),
        pl.col("oni").shift(1).alias("oni_lag_1"),
    )


def _write_interface_frame(tmp_path: Path) -> tuple[Path, pl.DataFrame]:
    path = tmp_path / "dataset.parquet"
    frame = _interface_frame()
    frame.write_parquet(path)
    return path, frame


def test_parquet_interface_reads_and_filters(tmp_path) -> None:
    path, frame = _write_interface_frame(tmp_path)
    reader = ParquetDataset(path)

    assert reader.path == path
    assert reader.columns() == tuple(frame.columns)
    assert reader.schema() == dict(frame.schema)
    assert reader.n_rows() == frame.height
    assert reader.unique_ids() == sorted(STATIONS)
    assert reader.date_range() == (frame["ds"].min(), frame["ds"].max())
    assert reader.read().equals(frame)
    assert reader.read(columns=("unique_id", "ds", "y")).columns == [
        "unique_id",
        "ds",
        "y",
    ]

    palua = reader.read(unique_ids="palua")
    assert palua["unique_id"].unique().to_list() == ["palua"]
    assert palua.height == frame.height // len(STATIONS)

    window = reader.read(
        unique_ids=["palua", "caicara"],
        start=date(2020, 1, 2),
        end=date(2020, 1, 4),
    )
    assert window["unique_id"].unique().sort().to_list() == ["caicara", "palua"]
    assert window["ds"].min() == date(2020, 1, 2)
    assert window["ds"].max() == date(2020, 1, 4)
    assert window.height == 2 * 3

    lazy = reader.lazy(columns=("unique_id", "y"), unique_ids="ayacucho")
    assert lazy.collect().columns == ["unique_id", "y"]


def test_parquet_interface_introspects_columns(tmp_path) -> None:
    path, frame = _write_interface_frame(tmp_path)
    reader = ParquetDataset(path)

    assert set(reader.feature_columns()) == set(frame.columns) - {
        "unique_id",
        "ds",
        "y",
    }
    assert reader.lag_columns() == ("ayacucho_lag_1", "oni_lag_1")
    assert set(reader.exogenous_columns()) == {"rain_ayacucho", "oni"}
    assert reader.null_counts(("y", "ayacucho_lag_1")) == {
        "y": 0,
        "ayacucho_lag_1": len(STATIONS),
    }

    summary = reader.summary()
    assert summary["rows"] == frame.height
    assert summary["columns"] == frame.width
    assert summary["unique_ids"] == sorted(STATIONS)
    assert (summary["start"], summary["end"]) == reader.date_range()


def test_parquet_interface_rejects_invalid_arguments(tmp_path) -> None:
    path, _ = _write_interface_frame(tmp_path)
    reader = ParquetDataset(path)

    with pytest.raises(ValueError, match="columnas inexistentes"):
        reader.read(columns=("y", "no_existe"))
    with pytest.raises(ValueError, match="columns no puede"):
        reader.read(columns=())
    with pytest.raises(ValueError, match="unique_ids inexistentes"):
        reader.read(unique_ids="no_existe")
    with pytest.raises(ValueError, match="start debe ser datetime.date"):
        reader.read(start=datetime(2020, 1, 2))
    with pytest.raises(ValueError, match="posterior a end"):
        reader.read(start=date(2020, 1, 5), end=date(2020, 1, 2))


def test_parquet_interface_requires_an_existing_dataset(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="write_parquet"):
        ParquetDataset(tmp_path / "missing.parquet")


def test_parquet_interface_validates_required_schema(tmp_path) -> None:
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"unique_id": ["a"], "ds": ["2020-01-01"], "y": [1.0]}).write_parquet(
        path
    )

    with pytest.raises(ValueError, match="esquema inválido"):
        ParquetDataset(path)

    path = tmp_path / "missing_y.parquet"
    pl.DataFrame({"unique_id": ["a"], "ds": [date(2020, 1, 1)]}).write_parquet(path)

    with pytest.raises(ValueError, match="faltan columnas obligatorias"):
        ParquetDataset(path)


def test_dataset_interface_exposes_written_parquet(tmp_path, monkeypatch) -> None:
    output = tmp_path / "levels.parquet"
    dataset = TimeSeriesDataset(output)
    _patch_wide(monkeypatch, dataset, _wide())

    dataset.write_parquet()

    reader = dataset.interface()
    assert isinstance(reader, ParquetDataset)
    assert reader.path == output
    assert reader.unique_ids() == sorted(STATIONS)
