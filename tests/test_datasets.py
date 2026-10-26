"""Persistencia Parquet del dataset largo con features de niveles."""

from datetime import date, timedelta

import polars as pl
import pytest

from orinoco_forecast.data.datasets import STATIONS, TimeSeriesDataset
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
