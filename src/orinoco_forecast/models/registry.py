"""Registros de modelos por nombre (CLI / YAML).

- ``AVAILABLE_MODELS`` / ``build_forecaster``: construyen un ``Forecaster`` que
  se ajusta y predice un corte a la vez.
- ``RUNNERS`` / ``get_runner``: instancian un ``ForecastRunner`` que produce la
  validación cruzada completa al estilo Nixtla; sus fábricas importan de forma
  diferida para cargar solo la librería necesaria.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import yaml

from orinoco_forecast.data.loading import load_config, repo_root
from orinoco_forecast.features.lags import load_feature_config
from orinoco_forecast.models.base import Forecaster, ForecastRunner, Runner
from orinoco_forecast.models.statistical.arima import ArimaForecaster
from orinoco_forecast.models.statistical.baselines import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
)

AVAILABLE_MODELS = ("arima", "naive", "seasonal_naive")
RUNNER_ALIASES = {
    "arima": "arimax",
    "lstm": "nsmt",
    "timefm3": "timesfm3",
    "timesfm": "timesfm3",
}


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


def _naive() -> Runner:
    from orinoco_forecast.models.statistical.baselines import NaiveRunner

    return NaiveRunner()


def _seasonal_naive() -> Runner:
    from orinoco_forecast.evaluation.backtest import load_backtest_config
    from orinoco_forecast.models.statistical.baselines import SeasonalNaiveRunner

    seasonality = int(load_backtest_config()["seasonality"])
    return SeasonalNaiveRunner(seasonality=seasonality)


def _arimax() -> Runner:
    from orinoco_forecast.models.statistical.arima import ArimaxRunner

    return ArimaxRunner()


def _booster() -> Runner:
    from orinoco_forecast.models.trees.tree import BoosterRunner

    return BoosterRunner()


def _nsmt() -> Runner:
    from orinoco_forecast.models.neural.lstm import NsmtRunner

    return NsmtRunner()


def _transformer() -> Runner:
    from orinoco_forecast.models.neural.transformer import TransformerRunner

    return TransformerRunner()


def _timesfm3() -> Runner:
    from orinoco_forecast.models.pretrained.timesfm import TimesFM3Runner

    return TimesFM3Runner()


def _tcn() -> Runner:
    from orinoco_forecast.models.neural.cnn import TCNRunner

    return TCNRunner()


RUNNERS: dict[str, Callable[[], Runner]] = {
    "arimax": _arimax,
    "booster": _booster,
    "naive": _naive,
    "nsmt": _nsmt,
    "seasonal_naive": _seasonal_naive,
    "tcn": _tcn,
    "timesfm3": _timesfm3,
    "transformer": _transformer,
}


def available_runners() -> list[str]:
    """Canonical runner keys exposed by the CLI and the documentation."""
    return sorted(RUNNERS)


def _legacy_runner(name: str) -> ForecastRunner | None:
    """Old CV runners kept for tuning/dashboard compatibility."""
    if name == "baselines":
        from orinoco_forecast.models.statistical.baselines import BaselinesRunner

        return BaselinesRunner()
    if name == "xgboost":
        from orinoco_forecast.models.trees.tree import XGBoostRunner

        return XGBoostRunner()
    return None


def get_runner(name: str) -> Runner | ForecastRunner:
    """Instantiate a runner by canonical name or alias; KeyError if unknown."""
    canonical = RUNNER_ALIASES.get(name, name)
    if canonical in RUNNERS:
        return RUNNERS[canonical]()
    legacy = _legacy_runner(name)
    if legacy is not None:
        return legacy
    raise KeyError(f"Modelo no registrado: {name!r}. Opciones: {available_runners()}.")
