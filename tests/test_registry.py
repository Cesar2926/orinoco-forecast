"""Pruebas del registro de modelos."""

import pytest

from orinoco_forecast.models.registry import build_forecaster, load_model_config
from orinoco_forecast.models.statistical.arima import ArimaForecaster
from orinoco_forecast.models.statistical.baselines import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def test_registry_builds_baselines() -> None:
    assert isinstance(build_forecaster("naive"), NaiveForecaster)

    seasonal = build_forecaster("seasonal_naive", seasonality=365)
    assert isinstance(seasonal, SeasonalNaiveForecaster)
    assert seasonal.seasonality == 365


def test_registry_reads_arima_config() -> None:
    model = build_forecaster("arima")

    assert isinstance(model, ArimaForecaster)
    assert model.order == (2, 1, 2)
    assert model.seasonal_order == (0, 0, 0, 0)
    assert model.auto_order is False
    assert model.exog is True
    assert model.min_exog_lag == 14
    assert model.river_order == (
        "ayacucho",
        "caicara",
        "ciudad_bolivar",
        "palua",
    )
    assert model.use_upstream is True
    assert model.use_rain is True
    assert model.use_oni is True
    assert model.use_calendar is True


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="no registrado"):
        build_forecaster("does_not_exist")


def test_load_model_config_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Configuración"):
        load_model_config("does_not_exist")
