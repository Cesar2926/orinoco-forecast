"""Simple forecast averaging over saved runner predictions."""

from datetime import date
from typing import Self

import polars as pl

from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    PREDICTION_COLUMNS,
    PredictionResult,
)

# Fixed before looking at the test block: every validated runner, equal weight.
ENSEMBLE_MEMBERS = ("arimax", "booster", "tcn", "cnn_lstm", "transformer", "timesfm3")
ENSEMBLE_MODEL = "ENSEMBLE"
KEYS = ("unique_id", "cutoff", "ds", "horizon_step")


def average_forecasts(
    predictions: pl.DataFrame, *, model: str = ENSEMBLE_MODEL
) -> pl.DataFrame:
    """Media simple entre runners, con las semillas de cada runner promediadas antes.

    ``predictions`` trae ``runner`` además de ``PREDICTION_COLUMNS``. Cada fila de
    salida exige a todos los runners: un ensamble con miembros faltantes no sería
    el mismo modelo en todos los cortes, así que se rechaza.
    """
    runners = predictions["runner"].n_unique()
    per_runner = predictions.group_by("runner", *KEYS).agg(pl.col("y_pred").mean())
    averaged = per_runner.group_by(*KEYS).agg(
        pl.col("y_pred").mean(), members=pl.col("runner").n_unique()
    )
    incomplete = averaged.filter(pl.col("members") != runners)
    if not incomplete.is_empty():
        cutoffs = sorted(set(incomplete["cutoff"]))
        raise ValueError(f"Faltan miembros del ensamble en los cortes {cutoffs}.")
    return (
        averaged.with_columns(model=pl.lit(model))
        .select(*PREDICTION_COLUMNS)
        .sort(*KEYS)
    )


class ReplayRunner:
    """Runner que devuelve predicciones ya calculadas para cada corte.

    Permite puntuar el ensamble con el mismo pipeline (``run_backtest``) que los
    demás modelos sin reentrenar a sus miembros.
    """

    implemented = True

    def __init__(self, name: str, predictions: pl.DataFrame) -> None:
        self.name = name
        self.predictions = predictions.select(*PREDICTION_COLUMNS)

    def _rows(self, cutoff: date, horizon: int) -> pl.DataFrame:
        rows = self.predictions.filter(
            (pl.col("cutoff") == cutoff) & (pl.col("horizon_step") <= horizon)
        )
        if rows.is_empty():
            raise ValueError(
                f"{self.name}: no hay predicciones para el corte {cutoff}."
            )
        return rows

    def fit(
        self,
        windows: pl.DataFrame,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> Self:
        self._rows(cutoff, horizon)
        return self

    def predict(
        self, frame: pl.DataFrame, *, cutoff: date, horizon: int = DEFAULT_HORIZON
    ) -> PredictionResult:
        return PredictionResult.from_frame(
            self._rows(cutoff, horizon), runner=self.name, horizon=horizon
        )
