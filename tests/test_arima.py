"""Pruebas del adaptador ARIMA."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from orinoco_forecast.data.loading import DataQualityWarning
from orinoco_forecast.features.lags import RiverFeatureConfig, add_river_features
from orinoco_forecast.models.base import PREDICTION_COLUMNS
from orinoco_forecast.models.statistical.arima import ArimaForecaster


def _series(
    start: date = date(2019, 1, 1),
    days: int = 400,
    *,
    unique_id: str = "ciudad_bolivar",
    offset: float = 0.0,
) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    index = np.arange(days, dtype=float)
    values = (
        offset
        + 50.0
        + 0.01 * index
        + 3.0 * np.sin(2 * np.pi * index / 365)
        + rng.normal(scale=0.2, size=days)
    )
    return pl.DataFrame(
        {
            "unique_id": [unique_id] * days,
            "ds": [start + timedelta(days=int(step)) for step in index],
            "y": values,
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def _network_history(days: int = 500) -> pl.DataFrame:
    start = date(2019, 1, 1)
    rng = np.random.default_rng(7)
    index = np.arange(days, dtype=float)
    levels = {
        "ayacucho": 40.0 + 0.01 * index + 2.0 * np.sin(2 * np.pi * index / 365),
        "caicara": (30.0 + 0.008 * index + 1.5 * np.sin(2 * np.pi * index / 365 + 0.3)),
        "ciudad_bolivar": (
            20.0 + 0.005 * index + 1.0 * np.sin(2 * np.pi * index / 365 + 0.6)
        ),
        "palua": 10.0 + 0.003 * index + 0.8 * np.sin(2 * np.pi * index / 365 + 0.9),
    }
    rows = []
    for station, values in levels.items():
        for step, value in enumerate(values):
            rows.append(
                {
                    "unique_id": station,
                    "ds": start + timedelta(days=step),
                    "y": float(value + rng.normal(scale=0.05)),
                    "rain_ayacucho": float(step % 7),
                    "rain_caicara": float((step + 1) % 5),
                    "rain_ciudad_bolivar": float((step + 2) % 3),
                    "rain_palua": float((step + 3) % 4),
                    "oni": float(0.1 * np.sin(step / 30.0)),
                }
            )
    frame = pl.DataFrame(rows, schema_overrides={"ds": pl.Date})
    return add_river_features(
        frame,
        RiverFeatureConfig(
            upstream_lags=(14, 21),
            target_lags=(1, 2, 14, 21),
            rain_lags=(14, 21),
            oni_lags=(0, 30),
            calendar_terms=1,
        ),
    )


def _base_columns(history: pl.DataFrame) -> pl.DataFrame:
    raw_rain = [
        column
        for column in history.columns
        if column.startswith("rain_") and "_lag_" not in column
    ]
    return history.select("unique_id", "ds", "y", *raw_rain, "oni")


def test_arima_forecast_schema_and_horizon() -> None:
    history = _series()
    cutoff = date(2020, 1, 5)
    predictions = (
        ArimaForecaster(order=(1, 0, 0)).fit(history, cutoff=cutoff).predict(14)
    )

    assert predictions.columns == list(PREDICTION_COLUMNS)
    assert predictions.height == 14
    assert predictions["model"].unique().to_list() == ["arima"]
    assert predictions["unique_id"].unique().to_list() == ["ciudad_bolivar"]
    assert predictions["cutoff"].unique().to_list() == [cutoff]
    assert predictions["horizon_step"].to_list() == list(range(1, 15))
    assert predictions["ds"].to_list() == [
        cutoff + timedelta(days=step) for step in range(1, 15)
    ]
    assert predictions["y_pred"].is_finite().all()


def test_arima_fit_ignores_values_after_cutoff() -> None:
    history = _series(days=500)
    cutoff = date(2020, 1, 20)
    tampered = history.with_columns(
        pl.when(pl.col("ds") > cutoff)
        .then(pl.lit(1e9))
        .otherwise(pl.col("y"))
        .alias("y")
    )

    assert_frame_equal(
        ArimaForecaster(order=(1, 0, 0)).fit(history, cutoff=cutoff).predict(14),
        ArimaForecaster(order=(1, 0, 0)).fit(tampered, cutoff=cutoff).predict(14),
    )


def test_arima_tolerates_null_targets_without_imputation() -> None:
    history = _series(days=300).with_columns(
        pl.when(pl.col("ds") == date(2019, 3, 1))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    predictions = (
        ArimaForecaster(order=(1, 0, 1))
        .fit(history, cutoff=date(2019, 7, 1))
        .predict(14)
    )

    assert predictions["y_pred"].is_finite().all()


def test_arima_stacks_both_stations() -> None:
    history = pl.concat(
        [
            _series(days=200),
            _series(days=200, unique_id="palua", offset=10.0),
        ]
    )
    predictions = (
        ArimaForecaster(order=(1, 0, 0))
        .fit(history, cutoff=date(2019, 5, 1))
        .predict(3)
    )

    assert predictions.height == 6
    assert predictions["unique_id"].unique().sort().to_list() == [
        "ciudad_bolivar",
        "palua",
    ]


def test_arima_auto_order_forecasts() -> None:
    history = _series(days=120)
    predictions = (
        ArimaForecaster(auto_order=True, max_p=2, max_q=2, max_d=1)
        .fit(history, cutoff=date(2019, 3, 1))
        .predict(3)
    )

    assert predictions.height == 3
    assert predictions["y_pred"].is_finite().all()


def test_arima_auto_order_rejects_nulls() -> None:
    history = _series(days=60).with_columns(
        pl.when(pl.col("ds") == date(2019, 1, 10))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    model = ArimaForecaster(auto_order=True)
    with pytest.raises(ValueError, match="AutoARIMA no admite nulos"):
        model.fit(history, cutoff=date(2019, 2, 1))


def test_arima_rejects_station_without_observed_levels() -> None:
    history = _series(days=30).with_columns(pl.lit(None, dtype=pl.Float64).alias("y"))
    with pytest.raises(ValueError, match="niveles"):
        ArimaForecaster().fit(history, cutoff=date(2019, 1, 20))


def test_arima_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        ArimaForecaster().predict(14)


def test_arima_rejects_invalid_order() -> None:
    with pytest.raises(ValueError, match="orden"):
        ArimaForecaster(order=(1, 2))


def test_arima_rejects_negative_order_values() -> None:
    with pytest.raises(ValueError, match="orden"):
        ArimaForecaster(seasonal_order=(0, 0, 0, -1))


def test_arimax_uses_upstream_features_and_forecasts() -> None:
    history = _network_history()
    cutoff = date(2020, 3, 1)

    model = ArimaForecaster(order=(1, 0, 1), exog=True, min_exog_lag=14)
    predictions = model.fit(history, cutoff=cutoff).predict(14)

    assert predictions.height == 56
    assert predictions["y_pred"].is_finite().all()
    selected = model.selected_exog_columns
    assert "ayacucho_lag_14" in selected["ciudad_bolivar"]
    assert "caicara_lag_21" in selected["ciudad_bolivar"]
    assert "ciudad_bolivar_lag_14" in selected["ciudad_bolivar"]
    assert "ciudad_bolivar_lag_1" not in selected["ciudad_bolivar"]
    assert "palua_lag_14" not in selected["ciudad_bolivar"]
    assert "ciudad_bolivar_lag_14" in selected["palua"]
    assert "palua_lag_14" in selected["palua"]
    assert "palua_lag_1" not in selected["palua"]
    assert "rain_palua_lag_14" in selected["palua"]
    assert "oni_lag_30" in selected["palua"]
    assert "oni_lag_0" not in selected["palua"]
    assert "sin_doy" in selected["palua"]


def test_arimax_min_exog_lag_filters_columns() -> None:
    history = _network_history()
    model = ArimaForecaster(order=(1, 0, 1), exog=True, min_exog_lag=21).fit(
        history, cutoff=date(2020, 3, 1)
    )

    selected = model.selected_exog_columns["ciudad_bolivar"]
    assert "ayacucho_lag_14" not in selected
    assert "ayacucho_lag_21" in selected
    assert "ciudad_bolivar_lag_14" not in selected
    assert "ciudad_bolivar_lag_21" in selected
    assert "oni_lag_30" in selected
    assert "oni_lag_0" not in selected


def test_arimax_ignores_values_after_cutoff() -> None:
    history = _network_history()
    cutoff = date(2020, 3, 1)
    tampered = history.with_columns(
        pl.when(pl.col("ds") > cutoff).then(1e9).otherwise(pl.col("y")).alias("y")
    )

    original = (
        ArimaForecaster(order=(1, 0, 1), exog=True)
        .fit(history, cutoff=cutoff)
        .predict(14)
    )
    changed = (
        ArimaForecaster(order=(1, 0, 1), exog=True)
        .fit(tampered, cutoff=cutoff)
        .predict(14)
    )

    assert_frame_equal(original, changed)


def test_arimax_requires_feature_columns() -> None:
    history = _base_columns(_network_history())
    with pytest.raises(ValueError, match="columnas exógenas"):
        ArimaForecaster(exog=True).fit(history, cutoff=date(2020, 3, 1))


def test_arimax_without_exog_ignores_feature_columns() -> None:
    history = _network_history()
    base = _base_columns(history)
    cutoff = date(2020, 3, 1)

    with_features = (
        ArimaForecaster(order=(1, 0, 1)).fit(history, cutoff=cutoff).predict(7)
    )
    without_features = (
        ArimaForecaster(order=(1, 0, 1)).fit(base, cutoff=cutoff).predict(7)
    )

    assert_frame_equal(with_features, without_features)


def test_arimax_explicit_exog_columns() -> None:
    history = _network_history()
    model = ArimaForecaster(
        order=(1, 0, 1),
        exog_columns={
            "ciudad_bolivar": ["ayacucho_lag_14", "sin_doy"],
            "palua": ["ciudad_bolivar_lag_14"],
        },
    )
    predictions = model.fit(history, cutoff=date(2020, 3, 1)).predict(7)

    assert predictions["y_pred"].is_finite().all()
    assert model.selected_exog_columns["ciudad_bolivar"] == (
        "ayacucho_lag_14",
        "sin_doy",
    )
    assert model.selected_exog_columns["palua"] == ("ciudad_bolivar_lag_14",)
    assert model.selected_exog_columns["ayacucho"] == ()


def test_arimax_explicit_missing_column_fails() -> None:
    history = _network_history()
    model = ArimaForecaster(
        exog_columns={"ciudad_bolivar": ["does_not_exist"]},
    )
    with pytest.raises(ValueError, match="inexistentes"):
        model.fit(history, cutoff=date(2020, 3, 1))


def test_arimax_rejects_unsafe_explicit_lag_at_predict() -> None:
    history = _network_history()
    model = ArimaForecaster(
        order=(1, 0, 1),
        exog_columns={"ciudad_bolivar": ["ayacucho_lag_1"]},
    ).fit(history, cutoff=date(2020, 3, 1))

    with pytest.raises(ValueError, match="menor"):
        model.predict(14)


def test_arimax_tolerates_null_target_with_exog() -> None:
    history = _network_history().with_columns(
        pl.when((pl.col("unique_id") == "palua") & (pl.col("ds") == date(2020, 2, 10)))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    with pytest.warns(DataQualityWarning):
        predictions = (
            ArimaForecaster(order=(1, 0, 1), exog=True)
            .fit(history, cutoff=date(2020, 3, 1))
            .predict(14)
        )

    assert predictions["y_pred"].is_finite().all()


def test_arimax_auto_order_with_exog() -> None:
    history = _network_history(days=150)
    predictions = (
        ArimaForecaster(
            auto_order=True,
            max_p=2,
            max_q=2,
            max_d=1,
            exog=True,
        )
        .fit(history, cutoff=date(2019, 5, 1))
        .predict(3)
    )

    assert predictions["y_pred"].is_finite().all()


def test_arimax_rejects_station_outside_river_order() -> None:
    history = _network_history().with_columns(
        pl.when(pl.col("unique_id") == "palua")
        .then(pl.lit("tributary"))
        .otherwise(pl.col("unique_id"))
        .alias("unique_id")
    )
    with pytest.raises(ValueError, match="river_order"):
        ArimaForecaster(exog=True).fit(history, cutoff=date(2020, 3, 1))


def test_arimax_rejects_missing_upstream_station() -> None:
    history = _network_history().filter(pl.col("unique_id") != "ayacucho")
    with pytest.raises(ValueError, match="aguas arriba"):
        ArimaForecaster(exog=True).fit(history, cutoff=date(2020, 3, 1))


def test_arimax_selected_exog_columns_start_empty() -> None:
    assert ArimaForecaster(exog=True).selected_exog_columns == {}
