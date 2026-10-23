"""Registro de pronosticadores configurados por nombre."""

from __future__ import annotations

from typing import Any

import yaml

from orinoco_forecast.data.loading import repo_root
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
        return ArimaForecaster(
            order=config["order"],
            seasonal_order=config["seasonal_order"],
            auto_order=bool(config.get("auto_order", False)),
        )
    raise ValueError(
        f"Modelo no registrado: {name!r}. Disponibles: {list(AVAILABLE_MODELS)}."
    )
