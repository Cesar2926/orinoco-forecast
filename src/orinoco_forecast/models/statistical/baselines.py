"""Baselines de persistencia (Naive) y persistencia estacional (SeasonalNaive).

``NaiveForecaster`` y ``SeasonalNaiveForecaster`` son los motores estadísticos;
``NaiveRunner`` y ``SeasonalNaiveRunner`` los exponen con el contrato de
ejecución que devuelve un ``PredictionResult``.
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Self

import polars as pl

from orinoco_forecast.data.loading import DataQualityWarning
from orinoco_forecast.models.base import (
    PREDICTION_COLUMNS,
    PredictionResult,
    forecast_index,
    prepare_history,
    validate_horizon,
    validate_windows,
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
        seasonal_dates = (
            index.select(
                "unique_id",
                (pl.col("ds") - pl.duration(days=self.seasonality)).alias(
                    "seasonal_ds"
                ),
            )
            .unique()
            .sort("unique_id", "seasonal_ds")
        )
        available = (
            self._history.filter(pl.col("y").is_not_null())
            .select("unique_id", "ds", "y")
            .sort("unique_id", "ds")
        )
        lookup = seasonal_dates.join_asof(
            available,
            left_on="seasonal_ds",
            right_on="ds",
            by="unique_id",
            strategy="backward",
            check_sortedness=False,
        )
        shifted = lookup.filter(pl.col("ds") != pl.col("seasonal_ds"))
        if shifted.height:
            warnings.warn(
                f"SeasonalNaive: {shifted.height} valores de hace "
                f"{self.seasonality} días no están disponibles; se usó el "
                "último nivel anterior.",
                DataQualityWarning,
                stacklevel=2,
            )
        predictions = (
            index.with_columns(
                (pl.col("ds") - pl.duration(days=self.seasonality)).alias("seasonal_ds")
            )
            .join(
                lookup.select(
                    "unique_id",
                    "seasonal_ds",
                    pl.col("y").alias("y_pred"),
                ),
                on=["unique_id", "seasonal_ds"],
                how="left",
            )
            .drop("seasonal_ds")
        )
        missing = predictions.filter(pl.col("y_pred").is_null())
        if missing.height:
            raise ValueError(
                f"SeasonalNaive no encuentra ningún nivel anterior para "
                f"{missing.height} predicciones."
            )
        return predictions.select(PREDICTION_COLUMNS)


class NaiveRunner:
    """Persistencia como runner: ajusta, predice y devuelve un resultado exportable.

    ``windows`` se valida porque es la entrada común del contrato, pero Naive
    solo necesita el histórico contiguo hasta ``cutoff``.
    """

    name = "naive"
    implemented = True

    def __init__(self, engine: NaiveForecaster | None = None) -> None:
        self._engine = engine or NaiveForecaster()
        self._fitted = False

    def fit(
        self,
        windows: pl.DataFrame,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = 14,
    ) -> Self:
        validate_horizon(horizon)
        validate_windows(windows, horizon=horizon)
        self._engine.fit(frame, cutoff=cutoff)
        self._fitted = True
        return self

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = 14,
    ) -> PredictionResult:
        validate_horizon(horizon)
        if not self._fitted:
            raise RuntimeError("El runner no está ajustado; llama a fit primero.")
        predictions = self._engine.predict(horizon).with_columns(
            pl.lit(self.name).alias("model")
        )
        return PredictionResult.from_frame(
            predictions, runner=self.name, horizon=horizon
        )


class SeasonalNaiveRunner:
    """Persistencia estacional como runner sobre el histórico contiguo."""

    name = "seasonal_naive"
    implemented = True

    def __init__(
        self,
        engine: SeasonalNaiveForecaster | None = None,
        *,
        seasonality: int = 365,
    ) -> None:
        self._engine = engine or SeasonalNaiveForecaster(seasonality=seasonality)
        self._fitted = False

    def fit(
        self,
        windows: pl.DataFrame,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = 14,
    ) -> Self:
        validate_horizon(horizon)
        validate_windows(windows, horizon=horizon)
        self._engine.fit(frame, cutoff=cutoff)
        self._fitted = True
        return self

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = 14,
    ) -> PredictionResult:
        validate_horizon(horizon)
        if not self._fitted:
            raise RuntimeError("El runner no está ajustado; llama a fit primero.")
        predictions = self._engine.predict(horizon).with_columns(
            pl.lit(self.name).alias("model")
        )
        return PredictionResult.from_frame(
            predictions, runner=self.name, horizon=horizon
        )
