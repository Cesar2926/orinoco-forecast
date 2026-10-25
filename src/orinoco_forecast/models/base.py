"""Interfaz neutral de pronosticadores y utilidades del protocolo de cortes.

Todos los modelos reciben un histórico en formato largo (``unique_id``, ``ds``,
``y`` y covariables opcionales) y un ``cutoff``. El ajuste usa solo fechas
menores o iguales al corte y el pronóstico cubre ``cutoff + 1 .. cutoff + H``.

``Forecaster`` ajusta y predice un corte a la vez; ``ForecastRunner`` produce la
validación cruzada completa con el protocolo común de ``configs/backtest.yaml``.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, Self

import polars as pl

FORECAST_INDEX_COLUMNS = (
    "model",
    "unique_id",
    "cutoff",
    "ds",
    "horizon_step",
)
PREDICTION_COLUMNS = (*FORECAST_INDEX_COLUMNS, "y_pred")
PREDICTION_SCHEMA = {
    "model": pl.String,
    "unique_id": pl.String,
    "cutoff": pl.Date,
    "ds": pl.Date,
    "horizon_step": pl.Int32,
    "y_pred": pl.Float64,
}


class Forecaster(Protocol):
    """Contrato mínimo de un modelo de pronóstico multi-paso."""

    name: str

    def fit(self, history: pl.DataFrame, *, cutoff: date) -> Self:
        """Ajusta el modelo usando solo observaciones hasta ``cutoff``."""
        ...

    def predict(self, horizon: int) -> pl.DataFrame:
        """Devuelve exactamente ``horizon`` filas por estación."""
        ...


def prepare_history(history: pl.DataFrame, *, cutoff: date) -> pl.DataFrame:
    """Valida el histórico long y lo recorta al corte.

    Conserva las columnas extra para los modelos que usan covariables. Los
    nulos en ``y`` se conservan: cada modelo decide cómo tratarlos sin imputar
    fuera del período de entrenamiento.
    """
    if type(cutoff) is not date:
        raise ValueError("cutoff debe ser datetime.date, sin hora.")
    required = {"unique_id", "ds", "y"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if history.schema["unique_id"] != pl.String:
        raise ValueError(
            f"unique_id debe ser pl.String, no {history.schema['unique_id']}."
        )
    if history.schema["ds"] != pl.Date:
        raise ValueError(f"ds debe ser pl.Date, no {history.schema['ds']}.")
    if not history.schema["y"].is_numeric():
        raise ValueError(f"y debe ser numérica, no {history.schema['y']}.")
    if history["unique_id"].null_count():
        raise ValueError("unique_id no puede contener nulos.")
    if history["ds"].null_count():
        raise ValueError("ds no puede contener nulos.")
    duplicates = history.select(
        pl.struct(["unique_id", "ds"]).is_duplicated().sum()
    ).item()
    if duplicates:
        raise ValueError(
            f"{duplicates} filas duplicadas por unique_id y ds; el histórico "
            "no se puede usar sin ambigüedad."
        )

    trimmed = history.filter(pl.col("ds") <= cutoff)
    if trimmed.is_empty():
        raise ValueError(f"No hay observaciones hasta el corte {cutoff}.")
    return trimmed.sort("unique_id", "ds")


def forecast_index(
    history: pl.DataFrame,
    *,
    model: str,
    cutoff: date,
    horizon: int,
) -> pl.DataFrame:
    """Construye la identidad de cada predicción: estación, corte y fecha.

    Una fila por estación y paso ``1 .. horizon``, con ``ds = cutoff + paso``.
    """
    if type(cutoff) is not date:
        raise ValueError("cutoff debe ser datetime.date, sin hora.")
    if horizon < 1:
        raise ValueError("horizon debe ser positivo.")
    if history.is_empty():
        raise ValueError("El histórico no puede estar vacío.")

    steps = pl.DataFrame(
        {"horizon_step": pl.int_range(1, horizon + 1, eager=True, dtype=pl.Int32)}
    )
    return (
        history["unique_id"]
        .unique()
        .sort()
        .to_frame()
        .join(steps, how="cross")
        .with_columns(
            pl.lit(model).alias("model"),
            pl.lit(cutoff).alias("cutoff"),
            (pl.lit(cutoff) + pl.duration(days=pl.col("horizon_step"))).alias("ds"),
        )
        .select(FORECAST_INDEX_COLUMNS)
    )


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
