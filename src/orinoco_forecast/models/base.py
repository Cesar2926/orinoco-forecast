"""Interfaz neutral de pronosticadores y utilidades del protocolo de cortes.

Todos los modelos reciben un histórico en formato largo (``unique_id``, ``ds``,
``y`` y covariables opcionales) y un ``cutoff``. El ajuste usa solo fechas
menores o iguales al corte y el pronóstico cubre ``cutoff + 1 .. cutoff + H``.

``Forecaster`` ajusta y predice un corte a la vez; ``ForecastRunner`` produce la
validación cruzada completa con el protocolo común de ``configs/backtest.yaml``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, Self

import polars as pl

from orinoco_forecast.utils.io import (
    predictions_dir,
    resolve_output_path,
    write_parquet,
)

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
DEFAULT_HORIZON = 14
MAX_HORIZON = 14
RUNNER_NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")


def validate_horizon(horizon: int) -> int:
    """Validate a forecast horizon in ``1..MAX_HORIZON`` (default 14)."""
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(
            f"El horizonte debe ser un entero entre 1 y {MAX_HORIZON}: {horizon!r}."
        )
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValueError(f"El horizonte debe estar entre 1 y {MAX_HORIZON}: {horizon}.")
    return horizon


def validate_windows(windows: pl.DataFrame, *, horizon: int) -> pl.DataFrame:
    """Validate the supervised windows passed to ``Runner.fit``."""
    if not isinstance(windows, pl.DataFrame):
        raise ValueError("windows debe ser un DataFrame de Polars.")
    if windows.is_empty():
        raise ValueError(
            "windows no puede estar vacío: no hay ventanas de entrenamiento."
        )
    required = {
        "unique_id",
        "origin",
        *(f"target_{step}" for step in range(1, horizon + 1)),
    }
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise ValueError(f"Faltan columnas en windows: {missing}.")
    return windows


def validate_runner_name(runner: str) -> str:
    if not isinstance(runner, str) or not RUNNER_NAME_PATTERN.fullmatch(runner):
        raise ValueError(f"runner inválido: {runner!r}; usa minúsculas, números y '_'.")
    return runner


def _validated_predictions(
    frame: pl.DataFrame, *, horizon: int, label: str = "predicciones"
) -> pl.DataFrame:
    """Validate and normalize a prediction frame to ``PREDICTION_COLUMNS``."""
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Faltan columnas de {label}: {missing}.")
    selected = frame.select(*PREDICTION_COLUMNS)
    if selected.is_empty():
        raise ValueError(f"Las {label} no pueden estar vacías.")
    if selected.schema["cutoff"] != pl.Date:
        raise ValueError(f"cutoff debe ser pl.Date, no {selected.schema['cutoff']}.")
    if selected.schema["ds"] != pl.Date:
        raise ValueError(f"ds debe ser pl.Date, no {selected.schema['ds']}.")
    if not selected.schema["horizon_step"].is_integer():
        raise ValueError(
            f"horizon_step debe ser entero, no {selected.schema['horizon_step']}."
        )
    if not selected.schema["y_pred"].is_numeric():
        raise ValueError(f"y_pred debe ser numérica, no {selected.schema['y_pred']}.")
    if selected["model"].null_count() or selected["unique_id"].null_count():
        raise ValueError("model y unique_id no pueden contener nulos.")
    if selected["cutoff"].unique().len() != 1:
        raise ValueError("Un solo corte por archivo de predicciones.")
    steps = sorted(set(selected["horizon_step"].to_list()))
    if steps != list(range(1, horizon + 1)):
        raise ValueError(
            f"Los pasos del horizonte deben ser 1..{horizon}; se recibieron {steps}."
        )
    mismatched = selected.filter(
        (pl.col("ds") - pl.col("cutoff")).dt.total_days() != pl.col("horizon_step")
    )
    if mismatched.height:
        raise ValueError(
            f"{mismatched.height} filas tienen ds distinto de cutoff + horizon_step."
        )
    duplicates = selected.select(
        pl.struct(["model", "unique_id", "horizon_step"]).is_duplicated().sum()
    ).item()
    if duplicates:
        raise ValueError(
            f"{duplicates} predicciones duplicadas por modelo, estación y paso."
        )
    if selected["y_pred"].null_count():
        raise ValueError("y_pred no puede contener valores nulos.")
    values = selected["y_pred"].cast(pl.Float64)
    if not values.is_finite().all():
        raise ValueError("y_pred debe contener solo valores finitos.")
    return selected.with_columns(y_pred=values).sort(
        "model", "unique_id", "horizon_step"
    )


class Writable(Protocol):
    """Result object that can export itself to Parquet."""

    def write(self, path: str | Path | None = None) -> Path:
        """Write the result and return the written path."""
        ...


@dataclass(frozen=True)
class PredictionResult:
    """Runner output: one row per model column, station and horizon step."""

    frame: pl.DataFrame
    runner: str
    horizon: int

    def __post_init__(self) -> None:
        validate_horizon(self.horizon)
        validate_runner_name(self.runner)
        object.__setattr__(
            self,
            "frame",
            _validated_predictions(self.frame, horizon=self.horizon),
        )

    @classmethod
    def from_frame(
        cls, frame: pl.DataFrame, *, runner: str, horizon: int = DEFAULT_HORIZON
    ) -> Self:
        """Validate ``frame`` and build a result tied to ``runner``."""
        return cls(frame=frame, runner=runner, horizon=horizon)

    def write(self, path: str | Path | None = None) -> Path:
        """Write ``data/predictions/<runner>/predictions.parquet`` by default."""
        default = predictions_dir() / self.runner / "predictions.parquet"
        return write_parquet(self.frame, resolve_output_path(path, default))


class Runner(Protocol):
    """Model executor: fits, predicts and returns an exportable result.

    ``frame`` must contain only observations with ``ds <= cutoff`` and
    ``windows`` is the ``build_windows`` table for the same cutoff and horizon.
    Statistical models may use the contiguous ``frame`` because their
    formulation cannot be fitted from overlapping 28-day windows.
    """

    name: str
    implemented: bool

    def fit(
        self,
        windows: pl.DataFrame,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> Self:
        """Fit using the windows and the history up to ``cutoff``."""
        ...

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> PredictionResult:
        """Predict ``cutoff + 1 .. cutoff + horizon`` for every station."""
        ...


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
