"""Registro único de runners por nombre.

``RUNNERS`` / ``get_runner`` instancian un ``Runner`` que ajusta, predice
1..14 días y devuelve un ``PredictionResult``. Las fábricas importan de forma
diferida para cargar solo la librería necesaria.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import yaml

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.models.base import Runner

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


def get_runner(name: str) -> Runner:
    """Instantiate a runner by canonical name or alias; KeyError if unknown."""
    canonical = RUNNER_ALIASES.get(name, name)
    if canonical in RUNNERS:
        return RUNNERS[canonical]()
    raise KeyError(f"Modelo no registrado: {name!r}. Opciones: {available_runners()}.")
