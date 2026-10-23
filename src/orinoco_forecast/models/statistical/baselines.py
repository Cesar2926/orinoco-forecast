"""Baselines de persistencia (Naive) y persistencia estacional (SeasonalNaive)."""

from __future__ import annotations

from datetime import date
from typing import Self

import polars as pl

from orinoco_forecast.models.base import (
    PREDICTION_COLUMNS,
    forecast_index,
    prepare_history,
)


class NaiveForecaster:
    """Repite el último nivel observado antes del corte."""

    name = "naive"

    def __init__(self) -> None:
        self._history: pl.DataFrame | None = None
        self._cutoff: date | None = None

    def fit(self, history: pl.DataFrame, *, cutoff: date) -> Self:
        self._history = prepare_history(history, cutoff=cutoff)
        self._cutoff = cutoff
        return self

    def predict(self, horizon: int) -> pl.DataFrame:
        if self._history is None or self._cutoff is None:
            raise RuntimeError("El modelo no está ajustado; llama a fit primero.")

        index = forecast_index(
            self._history,
            model=self.name,
            cutoff=self._cutoff,
            horizon=horizon,
        )
        levels = (
            self._history.filter(pl.col("y").is_not_null())
            .group_by("unique_id")
            .agg(pl.col("y").sort_by("ds").last().alias("y_pred"))
        )
        predictions = index.join(levels, on="unique_id", how="left")
        missing = (
            predictions.filter(pl.col("y_pred").is_null())["unique_id"].unique().sort()
        )
        if missing.len():
            raise ValueError(f"No hay niveles observados para: {missing.to_list()}.")
        return predictions.select(PREDICTION_COLUMNS)


class SeasonalNaiveForecaster:
    """Repite el nivel observado ``seasonality`` días antes de cada objetivo."""

    name = "seasonal_naive"

    def __init__(self, *, seasonality: int = 365) -> None:
        if seasonality < 1:
            raise ValueError("seasonality debe ser positiva.")
        self.seasonality = seasonality
        self._history: pl.DataFrame | None = None
        self._cutoff: date | None = None

    def fit(self, history: pl.DataFrame, *, cutoff: date) -> Self:
        self._history = prepare_history(history, cutoff=cutoff)
        self._cutoff = cutoff
        return self

    def predict(self, horizon: int) -> pl.DataFrame:
        if self._history is None or self._cutoff is None:
            raise RuntimeError("El modelo no está ajustado; llama a fit primero.")

        index = forecast_index(
            self._history,
            model=self.name,
            cutoff=self._cutoff,
            horizon=horizon,
        )
        lookup = self._history.select(
            "unique_id",
            (pl.col("ds") + pl.duration(days=self.seasonality)).alias("ds"),
            pl.col("y").alias("y_pred"),
        )
        predictions = index.join(lookup, on=["unique_id", "ds"], how="left")
        missing = predictions.filter(pl.col("y_pred").is_null())
        if missing.height:
            raise ValueError(
                f"SeasonalNaive no encuentra el valor de hace {self.seasonality} "
                f"días para {missing.height} predicciones."
            )
        return predictions.select(PREDICTION_COLUMNS)
