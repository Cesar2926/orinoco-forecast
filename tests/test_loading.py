"""Loader tests against the audited raw workbooks in data/raw/."""

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


def test_config_declares_targets_and_quality() -> None:
    config = load_config()
    assert config["targets"] == ["ciudad_bolivar", "palua"]
    assert set(config["quality"]) == {
        "flat_min_days",
        "jump_max_m_per_day",
        "source_conflict_m",
    }


def test_primary_long_format(primary: pl.DataFrame) -> None:
    assert dict(primary.schema) == SCHEMA
    assert primary.height == 4 * 18683
    assert primary.equals(primary.sort("unique_id", "ds"))
    spans = primary.group_by("unique_id").agg(
        pl.len(), pl.col("ds").min().alias("start"), pl.col("ds").max().alias("end")
    )
    assert sorted(spans["unique_id"].to_list()) == STATIONS
    assert spans["len"].to_list() == [18683] * 4
    assert set(spans["start"]) == {date(1974, 1, 1)}
    assert set(spans["end"]) == {date(2025, 2, 24)}


def test_primary_nulls_only_in_palua(primary_loaded) -> None:
    nulls = (
        primary_loaded.frame.group_by("unique_id")
        .agg(pl.col("y").null_count())
        .sort("unique_id")
    )
    assert nulls["y"].to_list() == [0, 0, 0, 17]
    assert any("palua" in m and "17" in m for m in primary_loaded.messages)


def test_primary_keeps_feb29(primary: pl.DataFrame) -> None:
    feb29 = primary.filter(pl.col("ds") == date(2000, 2, 29))
    assert feb29.height == 4


def test_historical_real_calendar(historical: pl.DataFrame) -> None:
    assert dict(historical.schema) == SCHEMA
    assert historical["unique_id"].unique().to_list() == ["ciudad_bolivar"]
    assert historical["ds"].min() == date(1930, 1, 1)
    assert historical["ds"].max() == date(2025, 12, 31)
    expected = pl.date_range(date(1930, 1, 1), date(2025, 12, 31), "1d", eager=True)
    assert historical["ds"].to_list() == expected.to_list()


def test_historical_feb29_inserted_as_null(historical_loaded) -> None:
    feb29 = historical_loaded.frame.filter(
        (pl.col("ds").dt.month() == 2) & (pl.col("ds").dt.day() == 29)
    )
    assert feb29.height == 24
    assert feb29["y"].null_count() == 24
    assert sum("29 de febrero" in m for m in historical_loaded.messages) == 1


def test_historical_gap_1933_kept_as_nulls(historical: pl.DataFrame) -> None:
    gap = historical.filter(
        pl.col("ds").is_between(date(1933, 10, 11), date(1934, 3, 20))
    )
    assert gap.height == 161
    assert gap["y"].null_count() == 161
    # 179 calendar nulls plus 24 inserted Feb 29 rows.
    assert historical["y"].null_count() == 179 + 24


def test_historical_excludes_2026(historical: pl.DataFrame) -> None:
    assert historical["ds"].dt.year().max() == 2025


def test_load_source_rejects_duplicate_and_unknown() -> None:
    with pytest.raises(ValueError, match="duplicada"):
        load_source("orinoco_stations_csv_export")
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
