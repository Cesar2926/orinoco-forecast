"""Map model names (CLI / YAML) to runner factories.

Factories import lazily so a CLI call only loads the library it needs.
"""

from collections.abc import Callable

from orinoco_forecast.models.base import ForecastRunner


def _baselines() -> ForecastRunner:
    from orinoco_forecast.models.statistical.baselines import BaselinesRunner

    return BaselinesRunner()


def _xgboost() -> ForecastRunner:
    from orinoco_forecast.models.trees.tree import XGBoostRunner

    return XGBoostRunner()


RUNNERS: dict[str, Callable[[], ForecastRunner]] = {
    "baselines": _baselines,
    "xgboost": _xgboost,
}


def get_runner(name: str) -> ForecastRunner:
    """Instancia el runner registrado como ``name``; KeyError si no existe."""
    if name not in RUNNERS:
        raise KeyError(f"Modelo no registrado: {name!r}. Opciones: {sorted(RUNNERS)}")
    return RUNNERS[name]()
