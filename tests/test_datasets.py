"""Persistencia Parquet del dataset ancho y features de niveles."""

from datetime import date, timedelta

import polars as pl

from orinoco_forecast.data.datasets import STATIONS, TimeSeriesDataset
from orinoco_forecast.features.lags import RiverFeatureConfig, add_river_features


def _levels() -> pl.DataFrame:
    days = [date(2020, 1, 1) + timedelta(days=index) for index in range(35)]
    return pl.DataFrame(
        {
            "ds": days,
            **{station: [float(index)] * 35 for index, station in enumerate(STATIONS)},
        }
    )


def test_write_parquet_uses_configured_output(tmp_path) -> None:
    output = tmp_path / "levels.parquet"
    dataset = TimeSeriesDataset(output)

    written = dataset.write_parquet(_levels())

    assert written == output
    assert pl.read_parquet(output).equals(_levels())


def test_write_parquet_accepts_output_override(tmp_path) -> None:
    dataset = TimeSeriesDataset(tmp_path / "default.parquet")
    override = tmp_path / "override.parquet"

    written = dataset.write_parquet(_levels(), override)

    assert written == override
    assert dataset.output_path == override
    assert pl.read_parquet(override).equals(_levels())


def test_base_can_disable_external_features() -> None:
    levels = _levels().unpivot(
        index="ds", variable_name="unique_id", value_name="y"
    )
    config = RiverFeatureConfig(rain_lags=(14,), oni_lags=(0,))

    output = add_river_features(
        levels,
        config,
        include_rain=False,
        include_oni=False,
    )

    assert "ayacucho_lag_14" in output.columns
    assert "oni_lag_0" not in output.columns
