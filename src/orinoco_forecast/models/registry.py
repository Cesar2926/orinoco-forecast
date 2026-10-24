"""Registro de pronosticadores configurados por nombre."""

from __future__ import annotations

from typing import Any

import yaml

from orinoco_forecast.data.loading import load_config, repo_root
from orinoco_forecast.features.lags import load_feature_config
from orinoco_forecast.models.base import Forecaster
from orinoco_forecast.models.statistical.arima import ArimaForecaster
from orinoco_forecast.models.statistical.baselines import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
)

AVAILABLE_MODELS = ("arima", "naive", "seasonal_naive")


def load_model_config(name: str) -> dict[str, Any]:
    """Lee ``configs/models/<name>.yaml`` y devuelve su contenido."""
    path = repo_root() / "configs" / "models" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Configuración de modelo no encontrada: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_forecaster(name: str, *, seasonality: int = 365) -> Forecaster:
    """Construye un modelo registrado con la configuración del repositorio."""
    if name == "naive":
        return NaiveForecaster()
    if name == "seasonal_naive":
        return SeasonalNaiveForecaster(seasonality=seasonality)
    if name == "arima":
        config = load_model_config(name)
        features = load_feature_config(load_config())
        return ArimaForecaster(
            order=config["order"],
            seasonal_order=config["seasonal_order"],
            auto_order=bool(config.get("auto_order", False)),
            exog=bool(config.get("exog", False)),
            min_exog_lag=int(config.get("min_exog_lag", features.min_exog_lag)),
            river_order=features.river_order,
            use_upstream=bool(config.get("use_upstream", True)),
            use_own_lags=bool(config.get("use_own_lags", True)),
            use_rain=bool(config.get("use_rain", True)),
            use_oni=bool(config.get("use_oni", True)),
            use_calendar=bool(config.get("use_calendar", True)),
        )
    raise ValueError(
        f"Modelo no registrado: {name!r}. Disponibles: {list(AVAILABLE_MODELS)}."
    )
