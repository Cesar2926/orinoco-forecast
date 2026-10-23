"""Pruebas de las ventanas supervisadas (paso 3 del plan de entrenamiento)."""

from datetime import date, timedelta
from math import cos, pi, sin

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from orinoco_forecast.data.loading import DataQualityWarning
from orinoco_forecast.data.windows import build_windows, prediction_window


def _series(
    start: date = date(2022, 1, 1),
    days: int = 40,
    *,
    unique_id: str = "ciudad_bolivar",
    offset: float = 0.0,
) -> pl.DataFrame:
    """Serie diaria sintética con ``y = offset + índice del día``."""
    return pl.DataFrame(
        {
            "unique_id": [unique_id] * days,
            "ds": [start + timedelta(days=index) for index in range(days)],
            "y": [offset + float(index) for index in range(days)],
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def test_build_windows_returns_expected_columns_and_values() -> None:
    frame = _series()
    windows = build_windows(
        frame,
        cutoff=date(2022, 1, 11),
        input_length=3,
        horizon=2,
    )

    assert windows.columns == [
        "unique_id",
        "origin",
        "y_lag_0",
        "y_lag_1",
        "y_lag_2",
        "day_of_year",
        "day_of_week",
        "month",
        "sin_1",
        "cos_1",
        "sin_2",
        "cos_2",
        "sin_3",
        "cos_3",
        "target_1",
        "target_2",
    ]
    # Orígenes con entrada completa (índice >= 2) y objetivo dentro del corte
    # (índice <= 8): 2022-01-03 .. 2022-01-09.
    assert windows.height == 7

    row = windows.filter(pl.col("origin") == date(2022, 1, 6)).row(0, named=True)
    assert row["y_lag_0"] == 5.0
    assert row["y_lag_1"] == 4.0
    assert row["y_lag_2"] == 3.0
    assert row["target_1"] == 6.0
    assert row["target_2"] == 7.0
    assert row["day_of_year"] == 6
    assert row["month"] == 1
    # 2022-01-06 fue jueves; ISO usa 1 = lunes .. 7 = domingo.
    assert row["day_of_week"] == 4
    assert row["sin_1"] == pytest.approx(sin(2 * pi * 6 / 365))
    assert row["cos_1"] == pytest.approx(cos(2 * pi * 6 / 365))
    assert row["sin_2"] == pytest.approx(sin(2 * pi * 2 * 6 / 365))


def test_build_windows_ignores_values_after_cutoff() -> None:
    frame = _series(days=60)
    cutoff = date(2022, 1, 20)
    arguments = {
        "cutoff": cutoff,
        "input_length": 5,
        "horizon": 14,
    }
    baseline = build_windows(frame, **arguments)

    tampered = frame.with_columns(
        pl.when(pl.col("ds") > cutoff)
        .then(pl.lit(1e9))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    assert_frame_equal(baseline, build_windows(tampered, **arguments))


def test_build_windows_drops_examples_with_null_targets() -> None:
    frame = _series(days=20).with_columns(
        pl.when(pl.col("ds") == date(2022, 1, 10))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    with pytest.warns(DataQualityWarning, match="ventanas descartadas"):
        windows = build_windows(
            frame,
            cutoff=date(2022, 1, 16),
            input_length=2,
            horizon=3,
        )

    origins = set(windows["origin"].to_list())
    # El nulo en el índice 9 contamina los objetivos de los orígenes 6, 7 y 8.
    assert date(2022, 1, 7) not in origins
    assert date(2022, 1, 8) not in origins
    assert date(2022, 1, 9) not in origins
    assert date(2022, 1, 6) in origins
    assert date(2022, 1, 12) in origins


def test_build_windows_drops_examples_with_null_inputs() -> None:
    frame = _series(days=20).with_columns(
        pl.when(pl.col("ds") == date(2022, 1, 6))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    with pytest.warns(DataQualityWarning, match="ventanas descartadas"):
        windows = build_windows(
            frame,
            cutoff=date(2022, 1, 13),
            input_length=3,
            horizon=2,
        )

    origins = set(windows["origin"].to_list())
    # El nulo en el índice 5 contamina la entrada de 5, 6 y 7 y el objetivo de 3 y 4.
    for index in (3, 4, 5, 6, 7):
        assert date(2022, 1, 1) + timedelta(days=index) not in origins
    assert date(2022, 1, 3) in origins
    assert date(2022, 1, 9) in origins


def test_build_windows_lags_exogenous_columns() -> None:
    frame = _series(days=20).with_columns(
        (pl.col("y") * 10).alias("rain"),
        pl.lit(0.5).alias("oni"),
    )
    windows = build_windows(
        frame,
        cutoff=date(2022, 1, 10),
        input_length=2,
        horizon=1,
        feature_columns=("y", "rain", "oni"),
    )

    row = windows.filter(pl.col("origin") == date(2022, 1, 6)).row(0, named=True)
    assert row["y_lag_0"] == 5.0
    assert row["rain_lag_0"] == 50.0
    assert row["rain_lag_1"] == 40.0
    assert row["oni_lag_0"] == 0.5
    assert row["target_1"] == 6.0


def test_prediction_window_is_the_latest_observed_context() -> None:
    frame = _series()
    window = prediction_window(
        frame,
        origin=date(2022, 1, 6),
        input_length=3,
    )

    assert window.height == 1
    assert "target_1" not in window.columns
    row = window.row(0, named=True)
    assert row["origin"] == date(2022, 1, 6)
    assert row["y_lag_0"] == 5.0
    assert row["y_lag_1"] == 4.0
    assert row["y_lag_2"] == 3.0
    assert row["day_of_year"] == 6


def test_prediction_window_does_not_use_future_values() -> None:
    frame = _series()
    origin = date(2022, 1, 10)
    tampered = frame.with_columns(
        pl.when(pl.col("ds") > origin)
        .then(pl.lit(1e9))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    assert_frame_equal(
        prediction_window(frame, origin=origin, input_length=4),
        prediction_window(tampered, origin=origin, input_length=4),
    )


def test_prediction_window_reports_missing_coverage() -> None:
    frame = _series(days=10).with_columns(
        pl.when(pl.col("ds") == date(2022, 1, 5))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )

    with pytest.warns(DataQualityWarning, match="rezagos nulos"):
        window = prediction_window(
            frame,
            origin=date(2022, 1, 6),
            input_length=2,
        )

    assert window["y_lag_0"].item() == 5.0
    assert window["y_lag_1"].item() is None


def test_build_windows_stacks_both_stations() -> None:
    frame = pl.concat(
        [
            _series(unique_id="ciudad_bolivar"),
            _series(unique_id="palua", offset=100.0),
        ]
    )
    windows = build_windows(
        frame,
        cutoff=date(2022, 1, 11),
        input_length=2,
        horizon=2,
    )

    assert windows.height == 16
    assert windows["unique_id"].unique().sort().to_list() == [
        "ciudad_bolivar",
        "palua",
    ]
    palua = windows.filter(pl.col("unique_id") == "palua")
    assert palua["y_lag_0"].min() == 101.0


def test_build_windows_rejects_duplicate_dates() -> None:
    frame = pl.concat([_series(), _series().tail(1)])
    with pytest.raises(ValueError, match="duplicad"):
        build_windows(frame, cutoff=date(2022, 1, 11), input_length=3, horizon=2)


def test_build_windows_rejects_unknown_station() -> None:
    with pytest.raises(ValueError, match="No hay filas"):
        build_windows(
            _series(),
            cutoff=date(2022, 1, 11),
            input_length=3,
            horizon=2,
            unique_id="palua",
        )


def test_build_windows_rejects_insufficient_history() -> None:
    with pytest.raises(ValueError, match="no hay ventanas"):
        build_windows(
            _series(days=10),
            cutoff=date(2022, 1, 4),
            input_length=5,
            horizon=2,
        )


def test_build_windows_rejects_non_numeric_features() -> None:
    frame = _series().with_columns(pl.lit("texto").alias("rain"))
    with pytest.raises(ValueError, match="numérica"):
        build_windows(
            frame,
            cutoff=date(2022, 1, 11),
            input_length=3,
            horizon=2,
            feature_columns=("y", "rain"),
        )
