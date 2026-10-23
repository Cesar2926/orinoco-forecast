"""Adaptador ARIMA sobre statsmodels, con AutoARIMA opcional vía pmdarima."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, Self

import numpy as np
import polars as pl
from statsmodels.tsa.arima.model import ARIMA

from orinoco_forecast.models.base import (
    PREDICTION_COLUMNS,
    forecast_index,
    prepare_history,
)


def _validate_order(
    values: Sequence[int],
    *,
    length: int,
    label: str,
) -> tuple[int, ...]:
    try:
        order = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} debe contener enteros: {values!r}.") from error
    if len(order) != length:
        raise ValueError(f"{label} debe tener {length} valores, no {len(order)}.")
    if any(value < 0 for value in order):
        raise ValueError(f"{label} no puede contener valores negativos: {order}.")
    return order


class ArimaForecaster:
    """ARIMA univariado por estación.

    Con ``auto_order=False`` usa ``statsmodels`` y admite nulos en ``y`` como
    observaciones faltantes. Con ``auto_order=True`` usa ``pmdarima``, que no
    admite nulos y busca el orden con AICc.
    """

    name = "arima"

    def __init__(
        self,
        *,
        order: Sequence[int] = (2, 1, 2),
        seasonal_order: Sequence[int] = (0, 0, 0, 0),
        auto_order: bool = False,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
    ) -> None:
        self.order = _validate_order(order, length=3, label="El orden ARIMA")
        self.seasonal_order = _validate_order(
            seasonal_order, length=4, label="El orden estacional"
        )
        for value, label in ((max_p, "max_p"), (max_q, "max_q"), (max_d, "max_d")):
            if value < 0:
                raise ValueError(f"{label} no puede ser negativo.")
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self._history: pl.DataFrame | None = None
        self._cutoff: date | None = None
        self._fits: dict[str, Any] = {}

    def fit(self, history: pl.DataFrame, *, cutoff: date) -> Self:
        self._history = prepare_history(history, cutoff=cutoff)
        self._cutoff = cutoff
        self._fits = {}
        for key, station in self._history.partition_by(
            "unique_id", as_dict=True
        ).items():
            unique_id = key[0] if isinstance(key, tuple) else key
            values = station["y"].to_numpy().astype(float)
            if np.isnan(values).all():
                raise ValueError(
                    f"{unique_id}: no hay niveles observados hasta el corte {cutoff}."
                )
            if self.auto_order and np.isnan(values).any():
                raise ValueError(
                    f"{unique_id}: AutoARIMA no admite nulos en y; usa "
                    "auto_order=false o trata los nulos."
                )
            self._fits[unique_id] = self._fit_station(values)
        return self

    def _fit_station(self, values: np.ndarray) -> Any:
        if self.auto_order:
            # Importación diferida: pmdarima es costoso y solo se usa aquí.
            from pmdarima import auto_arima

            return auto_arima(
                values,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                ic="aicc",
            )
        return ARIMA(
            values,
            order=self.order,
            seasonal_order=self.seasonal_order,
        ).fit()

    def predict(self, horizon: int) -> pl.DataFrame:
        if self._history is None or self._cutoff is None:
            raise RuntimeError("El modelo no está ajustado; llama a fit primero.")

        index = forecast_index(
            self._history,
            model=self.name,
            cutoff=self._cutoff,
            horizon=horizon,
        )
        steps = pl.int_range(1, horizon + 1, eager=True, dtype=pl.Int32)
        rows = []
        for unique_id, fit in self._fits.items():
            if self.auto_order:
                mean = np.asarray(fit.predict(n_periods=horizon), dtype=float)
            else:
                mean = np.asarray(
                    fit.get_forecast(steps=horizon).predicted_mean, dtype=float
                )
            if mean.shape != (horizon,):
                raise RuntimeError(
                    f"{unique_id}: ARIMA devolvió {mean.shape} predicciones, "
                    f"se esperaban {horizon}."
                )
            if not np.isfinite(mean).all():
                raise ValueError(f"{unique_id}: ARIMA produjo predicciones no finitas.")
            rows.append(
                pl.DataFrame(
                    {
                        "unique_id": pl.Series([unique_id] * horizon, dtype=pl.String),
                        "horizon_step": steps,
                        "y_pred": pl.Series(mean, dtype=pl.Float64),
                    }
                )
            )
        predictions = index.join(
            pl.concat(rows), on=["unique_id", "horizon_step"], how="left"
        )
        if predictions["y_pred"].null_count():
            raise RuntimeError("Faltan predicciones para algunas estaciones.")
        return predictions.select(PREDICTION_COLUMNS)
