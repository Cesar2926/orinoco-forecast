"""Model-neutral forecasting interface shared by every model family."""

from typing import Any, Protocol

import polars as pl


class ForecastRunner(Protocol):
    """Cualquier modelo que produzca validación cruzada con el protocolo común.

    ``cross_validation`` recibe la tabla del modelo y ``configs/backtest.yaml``
    (``horizon``, ``step_size``, ``n_windows``, ``refit``) y devuelve
    ``unique_id, ds, cutoff, y`` más una columna por nombre en ``model_columns``
    (polars o pandas; ``run_cv`` normaliza el resultado).
    """

    name: str
    model_columns: list[str]

    def cross_validation(self, frame: pl.DataFrame, cfg: dict[str, Any]) -> Any: ...
