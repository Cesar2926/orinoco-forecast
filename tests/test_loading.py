"""Loader tests against the raw workbook DATOS RIO ORINOCO.xlsx (sheet Entrenar)."""

from datetime import date

import polars as pl
import pytest

from orinoco_forecast.data.loading import (
    DataQualityWarning,
    load_config,
    load_source,
    validate_daily_dates,
)

STATIONS = ["ayacucho", "caicara", "ciudad_bolivar", "palua"]
SCHEMA = {"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64}
N_DAYS = 5934  # 2009-01-01..2025-03-31.


def test_config_declares_targets_and_quality() -> None:
    config = load_config()
    assert config["targets"] == ["ciudad_bolivar", "palua"]
    assert set(config["quality"]) == {
        "flat_min_days",
        "jump_max_m_per_day",
        "source_conflict_m",
    }


def test_config_has_single_primary_source_on_new_workbook() -> None:
    sources = load_config()["sources"]
    assert [s["role"] for s in sources] == ["primary"]
    assert sources[0]["path"] == "data/raw/DATOS RIO ORINOCO.xlsx"
    assert sources[0]["sheet"] == "Entrenar"


def test_primary_long_format(primary: pl.DataFrame) -> None:
    assert dict(primary.schema) == SCHEMA
    assert primary.height == 4 * N_DAYS
    assert primary.equals(primary.sort("unique_id", "ds"))
    spans = primary.group_by("unique_id").agg(
        pl.len(), pl.col("ds").min().alias("start"), pl.col("ds").max().alias("end")
    )
    assert sorted(spans["unique_id"].to_list()) == STATIONS
    assert spans["len"].to_list() == [N_DAYS] * 4
    assert set(spans["start"]) == {date(2009, 1, 1)}
    assert set(spans["end"]) == {date(2025, 3, 31)}


def test_primary_nulls_reported(primary_loaded) -> None:
    nulls = (
        primary_loaded.frame.group_by("unique_id")
        .agg(pl.col("y").null_count())
        .sort("unique_id")
    )
    assert nulls["y"].to_list() == [35, 35, 35, 38]
    assert any("palua" in m and "38" in m for m in primary_loaded.messages)


def test_primary_rows_after_2025_02_24_are_empty(primary: pl.DataFrame) -> None:
    tail = primary.filter(pl.col("ds") > date(2025, 2, 24))
    assert tail.height == 4 * 35
    assert tail["y"].null_count() == tail.height


def test_primary_nulls_in_2009_2024_are_palua_feb29(primary: pl.DataFrame) -> None:
    period = primary.filter(pl.col("ds") <= date(2024, 12, 31))
    nulls = period.filter(pl.col("y").is_null())
    assert nulls["unique_id"].unique().to_list() == ["palua"]
    assert nulls["ds"].to_list() == [
        date(2012, 2, 29),
        date(2016, 2, 29),
        date(2020, 2, 29),
    ]


def test_primary_keeps_feb29(primary: pl.DataFrame) -> None:
    feb29 = primary.filter(pl.col("ds") == date(2024, 2, 29))
    assert feb29.height == 4
    assert feb29["y"].null_count() == 0


def test_load_source_rejects_duplicate_and_unknown() -> None:
    config = {"sources": [{"name": "copy", "role": "duplicate"}]}
    with pytest.raises(ValueError, match="duplicada"):
        load_source("copy", config)
    with pytest.raises(KeyError):
        load_source("does_not_exist")


@pytest.mark.parametrize(
    ("days", "fragment"),
    [
        ([1, 2, 2, 3], "duplicadas"),
        ([1, 3, 2, 4], "orden"),
        ([1, 2, 5, 6], "huecos"),
    ],
)
def test_validate_daily_dates_warns(days: list[int], fragment: str) -> None:
    dates = pl.Series("ds", [date(2000, 1, d) for d in days])
    with pytest.warns(DataQualityWarning, match=fragment):
        validate_daily_dates(dates, "sintetica")
