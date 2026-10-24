"""Pruebas de las features de rezago aguas arriba y calendario."""

from datetime import date, timedelta

import polars as pl
import pytest

from orinoco_forecast.features.lags import (
    RIVER_ORDER,
    RiverFeatureConfig,
    add_river_features,
    calendar_columns,
    calendar_value,
    future_exog_matrix,
    load_feature_config,
    parse_feature_column,
    raw_predictors,
    select_exog_columns,
    upstream_of,
)

START = date(2020, 1, 1)


def _levels(
    days: int = 90,
    *,
    river_order: tuple[str, ...] = RIVER_ORDER,
    with_oni: bool = True,
) -> pl.DataFrame:
    rows = []
    for offset, station in enumerate(river_order):
        for step in range(days):
            row = {
                "unique_id": station,
                "ds": START + timedelta(days=step),
                "y": float(10 * (offset + 1) + step),
                "rain_ayacucho": float(step),
                "rain_caicara": float(step + 1),
                "rain_ciudad_bolivar": float(step + 2),
                "rain_palua": float(step + 3),
            }
            if with_oni:
                row["oni"] = float(step) / 10.0
            rows.append(row)
    return pl.DataFrame(rows, schema_overrides={"ds": pl.Date})


def _value(frame: pl.DataFrame, station: str, day: date, column: str) -> float:
    rows = frame.filter((pl.col("unique_id") == station) & (pl.col("ds") == day))
    assert rows.height == 1
    return rows.item(0, column)


def test_river_order_upstream_mapping() -> None:
    assert upstream_of("ayacucho") == ()
    assert upstream_of("caicara") == ("ayacucho",)
    assert upstream_of("ciudad_bolivar") == ("ayacucho", "caicara")
    assert upstream_of("palua") == ("ayacucho", "caicara", "ciudad_bolivar")
    with pytest.raises(ValueError, match="river_order"):
        upstream_of("unknown")


def test_features_are_lagged_by_calendar_days() -> None:
    featured = add_river_features(_levels(days=60))
    day = START + timedelta(days=30)

    assert _value(featured, "ciudad_bolivar", day, "ayacucho_lag_14") == (
        pytest.approx(26.0)
    )
    assert _value(featured, "ciudad_bolivar", day, "caicara_lag_21") == (
        pytest.approx(29.0)
    )
    assert _value(featured, "palua", day, "ciudad_bolivar_lag_14") == (
        pytest.approx(46.0)
    )
    assert _value(featured, "palua", day, "ayacucho_lag_14") == (pytest.approx(26.0))
    assert _value(
        featured, "ciudad_bolivar", day, "rain_ciudad_bolivar_lag_14"
    ) == pytest.approx(18.0)
    assert _value(featured, "ciudad_bolivar", day, "oni_lag_30") == (pytest.approx(0.0))
    # The downstream column exists in the table, but selection must not use it.
    assert _value(featured, "ciudad_bolivar", day, "palua_lag_14") == (
        pytest.approx(56.0)
    )


def test_calendar_features_are_deterministic() -> None:
    featured = add_river_features(
        _levels(days=10), RiverFeatureConfig(calendar_terms=2)
    )
    day = START

    assert _value(featured, "ciudad_bolivar", day, "sin_doy") == pytest.approx(
        calendar_value("sin_doy", day)
    )
    assert _value(featured, "ciudad_bolivar", day, "cos_doy") == pytest.approx(
        calendar_value("cos_doy", day)
    )
    assert _value(featured, "ciudad_bolivar", day, "sin_doy_2") == pytest.approx(
        calendar_value("sin_doy_2", day)
    )
    assert _value(featured, "ciudad_bolivar", day, "cos_doy_2") == pytest.approx(
        calendar_value("cos_doy_2", day)
    )
    assert calendar_columns(0) == ()
    assert calendar_columns(2) == (
        "sin_doy",
        "cos_doy",
        "sin_doy_2",
        "cos_doy_2",
    )
    assert calendar_value("sin_doy", date(2020, 1, 1)) == pytest.approx(
        0.017201575418260506
    )
    assert calendar_value("cos_doy", date(2020, 1, 1)) == pytest.approx(
        0.9998520429168804
    )


def test_select_exog_columns_uses_only_upstream_and_safe_lags() -> None:
    featured = add_river_features(_levels(days=60))
    columns = featured.columns

    city = select_exog_columns(columns, station="ciudad_bolivar")
    assert "ayacucho_lag_14" in city
    assert "caicara_lag_14" in city
    assert "ciudad_bolivar_lag_14" not in city
    assert "palua_lag_14" not in city
    assert "rain_ciudad_bolivar_lag_14" in city
    assert "oni_lag_0" not in city
    assert "oni_lag_30" in city
    assert "sin_doy" in city

    palua = select_exog_columns(columns, station="palua")
    assert "ciudad_bolivar_lag_14" in palua
    assert "palua_lag_14" not in palua
    assert "rain_palua_lag_14" in palua

    safe = select_exog_columns(columns, station="ciudad_bolivar", min_exog_lag=21)
    assert "ayacucho_lag_14" not in safe
    assert "ayacucho_lag_21" in safe
    assert "oni_lag_30" in safe


def test_select_exog_columns_can_disable_groups() -> None:
    featured = add_river_features(_levels(days=60))
    selected = select_exog_columns(
        featured.columns,
        station="palua",
        use_rain=False,
        use_oni=False,
        use_calendar=False,
    )

    assert "ciudad_bolivar_lag_14" in selected
    assert "ayacucho_lag_14" in selected
    assert "rain_palua_lag_14" not in selected
    assert "oni_lag_30" not in selected
    assert "sin_doy" not in selected


def test_select_exog_columns_rejects_unknown_station() -> None:
    with pytest.raises(ValueError, match="river_order"):
        select_exog_columns(["ayacucho_lag_14"], station="unknown")


def test_parse_feature_column() -> None:
    assert parse_feature_column("ayacucho_lag_14") == ("ayacucho", 14)
    assert parse_feature_column("rain_ciudad_bolivar_lag_21") == (
        "rain_ciudad_bolivar",
        21,
    )
    assert parse_feature_column("oni_lag_0") == ("oni", 0)
    assert parse_feature_column("sin_doy") is None
    assert parse_feature_column("ayacucho") is None


def test_load_feature_config_defaults_and_overrides() -> None:
    default = load_feature_config({})
    assert default.river_order == RIVER_ORDER
    assert default.min_exog_lag == 14

    custom = load_feature_config(
        {
            "features": {
                "river_order": ["a", "b", "c"],
                "min_exog_lag": 7,
                "upstream_lags": [7, 14],
                "target_lags": [1, 2],
                "rain_lags": [7],
                "oni_lags": [0, 30],
                "calendar_terms": 2,
            }
        }
    )
    assert custom.river_order == ("a", "b", "c")
    assert custom.min_exog_lag == 7
    assert custom.upstream_lags == (7, 14)
    assert custom.target_lags == (1, 2)
    assert custom.rain_lags == (7,)
    assert custom.oni_lags == (0, 30)
    assert custom.calendar_terms == 2


def test_feature_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="river_order"):
        RiverFeatureConfig(river_order=("a", "a"))
    with pytest.raises(ValueError, match="upstream_lags"):
        RiverFeatureConfig(upstream_lags=(0,))
    with pytest.raises(ValueError, match="oni_lags"):
        RiverFeatureConfig(oni_lags=(-1,))
    with pytest.raises(ValueError, match="min_exog_lag"):
        RiverFeatureConfig(min_exog_lag=-1)
    with pytest.raises(ValueError, match="calendar_terms"):
        RiverFeatureConfig(calendar_terms=-1)
    with pytest.raises(ValueError, match="features"):
        load_feature_config({"features": ["not", "a", "dict"]})


def test_add_river_features_preserves_columns_and_sorts() -> None:
    frame = _levels(days=30).with_columns(pl.lit("ok").alias("quality"))
    featured = add_river_features(frame)

    for column in frame.columns:
        assert column in featured.columns
    assert "quality" in featured.columns
    assert featured.sort("unique_id", "ds").equals(featured)


def test_add_river_features_rejects_missing_station() -> None:
    frame = _levels(
        days=30,
        river_order=("ayacucho", "caicara", "ciudad_bolivar"),
    )
    with pytest.raises(ValueError, match="river_order"):
        add_river_features(frame)


def test_add_river_features_rejects_duplicate_dates() -> None:
    frame = pl.concat([_levels(days=30), _levels(days=30).head(1)])
    with pytest.raises(ValueError, match="duplicadas"):
        add_river_features(frame)


def test_add_river_features_requires_oni_when_configured() -> None:
    with pytest.raises(ValueError, match="oni"):
        add_river_features(_levels(days=30, with_oni=False))


def test_add_river_features_can_skip_calendar_and_lags() -> None:
    frame = _levels(days=30)
    config = RiverFeatureConfig(
        upstream_lags=(),
        target_lags=(),
        rain_lags=(),
        oni_lags=(),
        calendar_terms=0,
    )
    featured = add_river_features(frame, config)

    assert featured.columns == frame.columns


def test_features_do_not_read_future_values() -> None:
    frame = _levels(days=60)
    day = START + timedelta(days=30)
    tampered = frame.with_columns(
        pl.when(pl.col("ds") > day).then(1e9).otherwise(pl.col("y")).alias("y")
    )

    original = add_river_features(frame)
    changed = add_river_features(tampered)
    for column in ("ayacucho_lag_14", "caicara_lag_21", "ciudad_bolivar_lag_1"):
        assert _value(original, "palua", day, column) == _value(
            changed, "palua", day, column
        )


def test_raw_predictors_aligns_levels_rain_and_oni() -> None:
    raw = raw_predictors(_levels(days=10))

    assert set(raw.columns) == {
        "ds",
        "ayacucho",
        "caicara",
        "ciudad_bolivar",
        "palua",
        "rain_ayacucho",
        "rain_caicara",
        "rain_ciudad_bolivar",
        "rain_palua",
        "oni",
    }
    assert raw.height == 10
    assert raw["ds"][0] == START
    assert raw["ayacucho"][0] == pytest.approx(10.0)
    assert raw["oni"][0] == pytest.approx(0.0)


def test_future_exog_matrix_uses_observed_values_only() -> None:
    days = 60
    raw = pl.DataFrame(
        {
            "ds": [START + timedelta(days=step) for step in range(days)],
            "ayacucho": [float(step) for step in range(days)],
            "rain_ciudad_bolivar": [float(step) * 10 for step in range(days)],
            "oni": [float(step) * 100 for step in range(days)],
        },
        schema_overrides={"ds": pl.Date},
    )
    cutoff = START + timedelta(days=39)
    matrix = future_exog_matrix(
        raw,
        ["ayacucho_lag_14", "rain_ciudad_bolivar_lag_14", "oni_lag_30"],
        station="ciudad_bolivar",
        cutoff=cutoff,
        horizon=3,
    )

    assert matrix.tolist() == [
        [26.0, 260.0, 1000.0],
        [27.0, 270.0, 1100.0],
        [28.0, 280.0, 1200.0],
    ]


def test_future_exog_matrix_rejects_unsafe_lag() -> None:
    raw = pl.DataFrame(
        {
            "ds": [START + timedelta(days=step) for step in range(60)],
            "ayacucho": [float(step) for step in range(60)],
        },
        schema_overrides={"ds": pl.Date},
    )
    with pytest.raises(ValueError, match="menor"):
        future_exog_matrix(
            raw,
            ["ayacucho_lag_1"],
            station="ciudad_bolivar",
            cutoff=START + timedelta(days=39),
            horizon=14,
        )


def test_future_exog_matrix_rejects_missing_source() -> None:
    raw = pl.DataFrame(
        {
            "ds": [START + timedelta(days=step) for step in range(60)],
            "ayacucho": [float(step) for step in range(60)],
        },
        schema_overrides={"ds": pl.Date},
    )
    with pytest.raises(ValueError, match="finito"):
        future_exog_matrix(
            raw,
            ["palua_lag_14"],
            station="ciudad_bolivar",
            cutoff=START + timedelta(days=39),
            horizon=1,
        )


def test_future_exog_matrix_includes_calendar() -> None:
    raw = pl.DataFrame(
        {
            "ds": [START + timedelta(days=step) for step in range(10)],
            "ayacucho": [float(step) for step in range(10)],
        },
        schema_overrides={"ds": pl.Date},
    )
    matrix = future_exog_matrix(
        raw,
        ["sin_doy", "cos_doy"],
        station="ciudad_bolivar",
        cutoff=START + timedelta(days=4),
        horizon=1,
    )
    day = START + timedelta(days=5)

    assert matrix[0, 0] == pytest.approx(calendar_value("sin_doy", day))
    assert matrix[0, 1] == pytest.approx(calendar_value("cos_doy", day))


def test_add_river_features_rejects_already_featured_table() -> None:
    featured = add_river_features(_levels(days=30))
    with pytest.raises(ValueError, match="ya contiene"):
        add_river_features(featured)
