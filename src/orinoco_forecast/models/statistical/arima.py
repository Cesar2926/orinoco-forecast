"""Adaptador ARIMA/ARIMAX sobre statsmodels, con AutoARIMA opcional vía pmdarima."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Self

import numpy as np
import polars as pl
from statsmodels.tsa.arima.model import ARIMA

from orinoco_forecast.data.loading import DataQualityWarning
from orinoco_forecast.features.lags import (
    RIVER_ORDER,
    future_exog_matrix,
    raw_predictors,
    select_exog_columns,
    upstream_of,
)
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


def _validate_river_order(values: Sequence[str]) -> tuple[str, ...]:
    order = tuple(str(value).strip() for value in values)
    if not order:
        raise ValueError("river_order no puede estar vacío.")
    if any(not station for station in order):
        raise ValueError("river_order no puede contener estaciones vacías.")
    if len(set(order)) != len(order):
        raise ValueError(f"river_order no puede contener duplicados: {order}.")
    return order


class ArimaForecaster:
    """ARIMA univariado o ARIMAX por estación.

    Con ``auto_order=False`` usa ``statsmodels`` y admite nulos en ``y`` como
    observaciones faltantes. Con ``auto_order=True`` usa ``pmdarima``, que no
    admite nulos y busca el orden con AICc.

    Con ``exog=True`` el modelo consume las features rezagadas construidas por
    :mod:`orinoco_forecast.features.lags`: niveles de estaciones aguas arriba,
    rezagos propios seguros, lluvia propia, ONI y calendario. Las exógenas
    futuras se obtienen solo de valores observados en el corte, y cualquier
    columna con un rezago menor que el paso pronosticado se rechaza para evitar
    fuga de información.
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
        exog: bool = False,
        exog_columns: Mapping[str, Sequence[str]] | None = None,
        river_order: Sequence[str] = RIVER_ORDER,
        min_exog_lag: int = 14,
        use_upstream: bool = True,
        use_own_lags: bool = True,
        use_rain: bool = True,
        use_oni: bool = True,
        use_calendar: bool = True,
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

        minimum = int(min_exog_lag)
        if minimum < 0:
            raise ValueError("min_exog_lag no puede ser negativo.")
        self.min_exog_lag = minimum
        self.river_order = _validate_river_order(river_order)
        self.use_upstream = bool(use_upstream)
        self.use_own_lags = bool(use_own_lags)
        self.use_rain = bool(use_rain)
        self.use_oni = bool(use_oni)
        self.use_calendar = bool(use_calendar)
        self.exog_columns: dict[str, tuple[str, ...]] | None = None
        if exog_columns is not None:
            explicit: dict[str, tuple[str, ...]] = {}
            for station, columns in exog_columns.items():
                if isinstance(columns, str):
                    raise ValueError(
                        f"exog_columns[{station!r}] debe ser una secuencia de "
                        "columnas, no un string."
                    )
                explicit[station] = tuple(columns)
            self.exog_columns = explicit
        self.exog = bool(exog or exog_columns is not None)

        self._history: pl.DataFrame | None = None
        self._cutoff: date | None = None
        self._fits: dict[str, Any] = {}
        self._selected_exog: dict[str, tuple[str, ...]] = {}
        self._raw: pl.DataFrame | None = None

    @property
    def selected_exog_columns(self) -> dict[str, tuple[str, ...]]:
        """Columnas exógenas efectivamente usadas por estación tras ``fit``."""
        return dict(self._selected_exog)

    def fit(self, history: pl.DataFrame, *, cutoff: date) -> Self:
        self._history = prepare_history(history, cutoff=cutoff)
        self._cutoff = cutoff
        self._fits = {}
        self._selected_exog = {}
        self._raw = raw_predictors(self._history) if self.exog else None
        self._validate_upstream_stations()
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
            columns = self._columns_for(unique_id, station.columns)
            exog, start = self._exog_matrix(unique_id, station, columns)
            if start:
                values = values[start:]
            if exog is not None and values.size != exog.shape[0]:
                raise RuntimeError(
                    f"{unique_id}: endog y exog tienen filas distintas "
                    f"({values.size} y {exog.shape[0]})."
                )
            self._selected_exog[unique_id] = tuple(columns)
            self._fits[unique_id] = self._fit_station(values, exog=exog)
        return self

    def _validate_upstream_stations(self) -> None:
        if not self.exog or self.exog_columns is not None or not self.use_upstream:
            return
        history = self._history
        if history is None:
            raise RuntimeError("El modelo no está ajustado; llama a fit primero.")
        available = set(history["unique_id"].unique().to_list())
        for station in sorted(available):
            if station not in self.river_order:
                raise ValueError(
                    f"{station!r} no está en river_order {self.river_order}; "
                    "agrega la estación al orden del río o desactiva use_upstream."
                )
            for source in upstream_of(station, self.river_order):
                if source not in available:
                    raise ValueError(
                        f"{station}: falta la estación aguas arriba {source!r} en "
                        "el histórico; incluye todas las estaciones del río antes "
                        "de ajustar ARIMAX."
                    )

    def _columns_for(self, unique_id: str, columns: Sequence[str]) -> list[str]:
        if self.exog_columns is not None:
            selected = list(self.exog_columns.get(unique_id, ()))
            missing = [column for column in selected if column not in columns]
            if missing:
                raise ValueError(
                    f"{unique_id}: columnas exógenas inexistentes: {missing}."
                )
            return selected
        if not self.exog:
            return []
        selected = list(
            select_exog_columns(
                columns,
                station=unique_id,
                river_order=self.river_order,
                min_exog_lag=self.min_exog_lag,
                use_upstream=self.use_upstream,
                use_own_lags=self.use_own_lags,
                use_rain=self.use_rain,
                use_oni=self.use_oni,
                use_calendar=self.use_calendar,
            )
        )
        if not selected:
            raise ValueError(
                f"{unique_id}: no hay columnas exógenas utilizables. Ejecuta "
                "scripts/build_exogenous_dataset.py con BUILD_RIVER_FEATURES=true "
                "o revisa las opciones use_*."
            )
        return selected

    def _exog_matrix(
        self,
        unique_id: str,
        station: pl.DataFrame,
        columns: Sequence[str],
    ) -> tuple[np.ndarray | None, int]:
        if not columns:
            return None, 0
        frame = station.select(columns)
        non_numeric = [
            column for column in columns if not frame.schema[column].is_numeric()
        ]
        if non_numeric:
            raise ValueError(
                f"{unique_id}: columnas exógenas no numéricas: {non_numeric}."
            )
        filled = frame.with_columns(
            *(pl.col(column).forward_fill() for column in columns)
        )
        before = frame.select(pl.all().null_count()).row(0)
        after = filled.select(pl.all().null_count()).row(0)
        n_filled = sum(
            int(left) - int(right) for left, right in zip(before, after, strict=True)
        )
        if n_filled:
            warnings.warn(
                f"{unique_id}: se rellenaron {n_filled} valores nulos de exog "
                "con el último valor observado dentro del entrenamiento, sin "
                "usar datos futuros.",
                DataQualityWarning,
                stacklevel=2,
            )
        exog = filled.to_numpy().astype(float)
        if exog.ndim != 2 or exog.shape[1] != len(columns):
            raise RuntimeError(
                f"{unique_id}: la matriz exógena tiene forma {exog.shape}; "
                f"se esperaban ({station.height}, {len(columns)})."
            )
        finite = np.isfinite(exog).all(axis=1)
        if not finite.any():
            raise ValueError(
                f"{unique_id}: todas las filas tienen exógenas nulas o no finitas."
            )
        invalid = np.flatnonzero(~finite)
        start = int(invalid[-1]) + 1 if invalid.size else 0
        trimmed = exog[start:]
        if trimmed.shape[0] < 2:
            raise ValueError(
                f"{unique_id}: quedan {trimmed.shape[0]} filas con exógenas "
                "completas; se necesitan al menos 2."
            )
        return trimmed, start

    def _fit_station(
        self,
        values: np.ndarray,
        *,
        exog: np.ndarray | None = None,
    ) -> Any:
        if self.auto_order:
            # Importación diferida: pmdarima es costoso y solo se usa aquí.
            from pmdarima import auto_arima

            kwargs: dict[str, Any] = {
                "seasonal": False,
                "stepwise": True,
                "suppress_warnings": True,
                "error_action": "ignore",
                "max_p": self.max_p,
                "max_q": self.max_q,
                "max_d": self.max_d,
                "ic": "aicc",
            }
            if exog is not None:
                kwargs["X"] = exog
            return auto_arima(values, **kwargs)
        return ARIMA(
            values,
            exog=exog,
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
            columns = self._selected_exog.get(unique_id, ())
            future = None
            if columns:
                if self._raw is None:
                    raise RuntimeError("El modelo no está ajustado; llama a fit.")
                future = future_exog_matrix(
                    self._raw,
                    columns,
                    station=unique_id,
                    cutoff=self._cutoff,
                    horizon=horizon,
                )
            if self.auto_order:
                if future is None:
                    mean = np.asarray(fit.predict(n_periods=horizon), dtype=float)
                else:
                    mean = np.asarray(
                        fit.predict(n_periods=horizon, X=future), dtype=float
                    )
            elif future is None:
                mean = np.asarray(
                    fit.get_forecast(steps=horizon).predicted_mean, dtype=float
                )
            else:
                mean = np.asarray(
                    fit.get_forecast(steps=horizon, exog=future).predicted_mean,
                    dtype=float,
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
