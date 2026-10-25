"""Baselines de persistencia (Naive) y persistencia estacional (SeasonalNaive).

``NaiveForecaster`` y ``SeasonalNaiveForecaster`` implementan ``Forecaster``
(un corte a la vez); ``BaselinesRunner`` implementa ``ForecastRunner`` con la
validación cruzada de StatsForecast.
"""

from __future__ import annotations

import warnings
from datetime import date
from typing import Any, Self

import polars as pl

from orinoco_forecast.data.loading import DataQualityWarning
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


class BaselinesRunner:
    """Persistencia y "lo mismo que el año pasado"; solo usan ``y``."""

    name = "baselines"
    model_columns = ["Naive", "SeasonalNaive"]

    def cross_validation(
        self, frame: pl.DataFrame, cfg: dict[str, Any]
    ) -> pl.DataFrame:
        # Imported here so the per-cutoff forecasters do not load StatsForecast.
        from statsforecast import StatsForecast
        from statsforecast.models import Naive, SeasonalNaive

        models = [Naive(), SeasonalNaive(season_length=cfg["seasonality"])]
        # With polars input StatsForecast requires a polars offset ("1d"), not "D".
        sf = StatsForecast(models=models, freq="1d")
        return sf.cross_validation(
            h=cfg["horizon"],
            df=frame.select("unique_id", "ds", "y"),
            n_windows=cfg["n_windows"],
            step_size=cfg["step_size"],
            refit=cfg["refit"],
        )
