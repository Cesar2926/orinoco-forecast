"""Ventanas supervisadas de longitud fija sin cruzar el corte cronológico.

Cada fila representa un origen ``t``: la entrada son los ``input_length`` días
hasta ``t`` (aplanados como columnas ``<variable>_lag_<k>``, donde ``k = 0`` es
``t``) y la salida son los ``horizon`` objetivos ``t+1 .. t+horizon``.

Las características solo usan fechas menores o iguales al origen y los
objetivos nunca superan el corte. Las ventanas con nulos en la entrada o en los
objetivos se descartan y se reportan con ``DataQualityWarning``; no se imputan.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from datetime import date, timedelta

import polars as pl

from orinoco_forecast.data.loading import DataQualityWarning

LAG_SEPARATOR = "_lag_"
CALENDAR_COLUMNS = ("day_of_year", "day_of_week", "month")
FOURIER_TERMS = 3
SEASONALITY = 365


def _validate_frame(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    input_length: int,
    fourier_terms: int,
    seasonality: float,
) -> tuple[str, ...]:
    """Valida la entrada común y devuelve las columnas de características."""
    required = {"unique_id", "ds", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if frame.schema["ds"] != pl.Date:
        raise ValueError(f"ds debe ser pl.Date, no {frame.schema['ds']}.")
    if not frame.schema["y"].is_numeric():
        raise ValueError(f"y debe ser numérica, no {frame.schema['y']}.")
    if frame["unique_id"].null_count():
        raise ValueError("unique_id no puede contener nulos.")
    if frame["ds"].null_count():
        raise ValueError("ds no puede contener nulos.")
    duplicates = frame.select(
        pl.struct(["unique_id", "ds"]).is_duplicated().sum()
    ).item()
    if duplicates:
        raise ValueError(
            f"{duplicates} filas duplicadas por unique_id y ds; las ventanas "
            "no se pueden construir sin ambigüedad."
        )

    if input_length < 1:
        raise ValueError("input_length debe ser positivo.")
    if fourier_terms < 0:
        raise ValueError("fourier_terms no puede ser negativo.")
    if seasonality <= 0:
        raise ValueError("seasonality debe ser positiva.")

    columns = tuple(feature_columns)
    if not columns:
        raise ValueError("feature_columns no puede estar vacío.")
    if len(set(columns)) != len(columns):
        raise ValueError("feature_columns contiene columnas repetidas.")
    reserved = {"unique_id", "ds"}
    for column in columns:
        if column in reserved:
            raise ValueError(f"{column!r} no puede usarse como característica.")
        if column not in frame.columns:
            raise ValueError(f"Columna de característica inexistente: {column!r}.")
        if not frame.schema[column].is_numeric():
            raise ValueError(
                f"{column!r} debe ser numérica, no {frame.schema[column]}."
            )
    return columns


def _station_ids(frame: pl.DataFrame, unique_id: str | None) -> list[str]:
    """Resuelve las estaciones a procesar y falla ante una desconocida."""
    available = frame["unique_id"].unique().sort().to_list()
    if not available:
        raise ValueError("El frame no contiene series.")
    if unique_id is None:
        return available
    if unique_id not in available:
        raise ValueError(f"No hay filas para unique_id={unique_id!r}.")
    return [unique_id]


def _station_frame(
    frame: pl.DataFrame,
    unique_id: str,
    columns: Sequence[str],
) -> pl.DataFrame:
    """Serie diaria de una estación, ordenada y limitada a las columnas usadas."""
    station = (
        frame.filter(pl.col("unique_id") == unique_id).select("ds", *columns).sort("ds")
    )
    if station.is_empty():
        raise ValueError(f"No hay filas para unique_id={unique_id!r}.")
    return station


def _daily_calendar(station: pl.DataFrame) -> pl.DataFrame:
    """Reindexa la serie a un calendario diario completo.

    Los días ausentes quedan con nulos; así un hueco no se convierte en un
    desplazamiento implícito de los rezagos.
    """
    start, end = station["ds"].min(), station["ds"].max()
    calendar = pl.DataFrame({"ds": pl.date_range(start, end, "1d", eager=True)})
    return calendar.join(station, on="ds", how="left").sort("ds")


def _feature_expressions(
    feature_columns: Sequence[str],
    input_length: int,
) -> list[pl.Expr]:
    """Rezagos de cada variable: ``k = 0`` es el origen y ``k`` días atrás."""
    return [
        pl.col(column).shift(lag).alias(f"{column}{LAG_SEPARATOR}{lag}")
        for column in feature_columns
        for lag in range(input_length)
    ]


def _feature_names(feature_columns: Sequence[str], input_length: int) -> list[str]:
    return [
        f"{column}{LAG_SEPARATOR}{lag}"
        for column in feature_columns
        for lag in range(input_length)
    ]


def _fourier_columns(fourier_terms: int, seasonality: float) -> list[str]:
    return [
        f"{name}_{term}"
        for term in range(1, fourier_terms + 1)
        for name in ("sin", "cos")
    ]


def _add_calendar(
    frame: pl.DataFrame,
    *,
    fourier_terms: int,
    seasonality: float,
) -> pl.DataFrame:
    """Añade origen, calendario y términos de Fourier del ciclo anual."""
    enriched = frame.with_columns(
        pl.col("ds").alias("origin"),
        pl.col("ds").dt.ordinal_day().alias("day_of_year"),
        pl.col("ds").dt.weekday().alias("day_of_week"),
        pl.col("ds").dt.month().alias("month"),
    )
    for term in range(1, fourier_terms + 1):
        angle = 2 * math.pi * term * pl.col("day_of_year") / seasonality
        enriched = enriched.with_columns(
            angle.sin().alias(f"sin_{term}"),
            angle.cos().alias(f"cos_{term}"),
        )
    return enriched


def _complete_mask(columns: Sequence[str]) -> pl.Expr:
    return pl.all_horizontal([pl.col(column).is_not_null() for column in columns])


def _station_windows(
    station: pl.DataFrame,
    *,
    unique_id: str,
    cutoff: date,
    input_length: int,
    horizon: int,
    feature_columns: Sequence[str],
    fourier_terms: int,
    seasonality: float,
) -> pl.DataFrame:
    observed = station.filter(pl.col("ds") <= cutoff)
    if observed.is_empty():
        raise ValueError(f"{unique_id}: no hay observaciones hasta el corte {cutoff}.")
    start = observed["ds"].min()
    earliest_origin = start + timedelta(days=input_length - 1)
    latest_origin = cutoff - timedelta(days=horizon)
    if earliest_origin > latest_origin:
        raise ValueError(
            f"{unique_id}: no hay ventanas completas antes del corte {cutoff}."
        )

    feature_names = _feature_names(feature_columns, input_length)
    target_names = [f"target_{step}" for step in range(1, horizon + 1)]
    windows = _daily_calendar(observed).with_columns(
        *_feature_expressions(feature_columns, input_length),
        *[
            pl.col("y").shift(-step).alias(f"target_{step}")
            for step in range(1, horizon + 1)
        ],
    )
    windows = _add_calendar(
        windows, fourier_terms=fourier_terms, seasonality=seasonality
    )
    eligible = windows.filter(
        (pl.col("ds") >= earliest_origin) & (pl.col("ds") <= latest_origin)
    )
    if eligible.is_empty():
        raise ValueError(
            f"{unique_id}: no hay ventanas completas antes del corte {cutoff}."
        )

    value_columns = [*feature_names, *target_names]
    complete = eligible.filter(_complete_mask(value_columns))
    dropped = eligible.height - complete.height
    if dropped:
        warnings.warn(
            f"{unique_id}: {dropped} ventanas descartadas por valores nulos "
            "en la entrada o en los objetivos.",
            DataQualityWarning,
            stacklevel=3,
        )

    return complete.select(
        pl.lit(unique_id).alias("unique_id"),
        "origin",
        *feature_names,
        *CALENDAR_COLUMNS,
        *_fourier_columns(fourier_terms, seasonality),
        *target_names,
    )


def build_windows(
    frame: pl.DataFrame,
    *,
    cutoff: date,
    input_length: int,
    horizon: int,
    feature_columns: Sequence[str] = ("y",),
    unique_id: str | None = None,
    fourier_terms: int = FOURIER_TERMS,
    seasonality: float = SEASONALITY,
) -> pl.DataFrame:
    """Construye las ventanas de entrenamiento anteriores a ``cutoff``.

    Una fila por ``(unique_id, origin)``. La entrada son los rezagos de
    ``feature_columns`` hasta ``origin`` y la salida son ``target_1 ..
    target_horizon``. Solo se conservan ventanas cuya entrada y objetivos estén
    completos y cuyos objetivos no superen ``cutoff``. Las columnas de
    covariables pueden incluir lluvia, ONI o niveles de otras estaciones ya
    alineados por fecha; ``y`` es siempre el objetivo. El calendario y los
    términos de Fourier se calculan sobre el origen.
    """
    if type(cutoff) is not date:
        raise ValueError("cutoff debe ser datetime.date, sin hora.")
    if horizon < 1:
        raise ValueError("horizon debe ser positivo.")

    columns = _validate_frame(
        frame,
        feature_columns=feature_columns,
        input_length=input_length,
        fourier_terms=fourier_terms,
        seasonality=seasonality,
    )
    stations = [
        _station_windows(
            _station_frame(frame, station, columns),
            unique_id=station,
            cutoff=cutoff,
            input_length=input_length,
            horizon=horizon,
            feature_columns=columns,
            fourier_terms=fourier_terms,
            seasonality=seasonality,
        )
        for station in _station_ids(frame, unique_id)
    ]
    return pl.concat(stations).sort("unique_id", "origin")


def prediction_window(
    frame: pl.DataFrame,
    *,
    origin: date,
    input_length: int,
    feature_columns: Sequence[str] = ("y",),
    unique_id: str | None = None,
    fourier_terms: int = FOURIER_TERMS,
    seasonality: float = SEASONALITY,
) -> pl.DataFrame:
    """Devuelve la entrada disponible en ``origin`` para pronosticar.

    Una fila por estación, sin columnas ``target_*``. Solo se usan fechas
    menores o iguales a ``origin``. Si alguna covariable no tiene cobertura
    hasta el origen, su rezago queda nulo y el modelo debe tratarlo.
    """
    if type(origin) is not date:
        raise ValueError("origin debe ser datetime.date, sin hora.")

    columns = _validate_frame(
        frame,
        feature_columns=feature_columns,
        input_length=input_length,
        fourier_terms=fourier_terms,
        seasonality=seasonality,
    )
    feature_names = _feature_names(columns, input_length)
    rows = []
    for station_id in _station_ids(frame, unique_id):
        station = _station_frame(frame, station_id, columns)
        if origin < station["ds"].min() or origin > station["ds"].max():
            raise ValueError(
                f"{station_id}: el origen {origin} está fuera del rango observado."
            )
        observed = station.filter(pl.col("ds") <= origin)
        row = (
            _daily_calendar(observed)
            .with_columns(*_feature_expressions(columns, input_length))
            .pipe(
                _add_calendar,
                fourier_terms=fourier_terms,
                seasonality=seasonality,
            )
            .filter(pl.col("ds") == origin)
        )
        if row.is_empty():
            raise ValueError(f"{station_id}: no se pudo construir el origen {origin}.")
        missing = [name for name in feature_names if row[name].null_count()]
        if missing:
            warnings.warn(
                f"{station_id}: el origen {origin} tiene rezagos nulos en {missing}.",
                DataQualityWarning,
                stacklevel=2,
            )
        rows.append(
            row.select(
                pl.lit(station_id).alias("unique_id"),
                "origin",
                *feature_names,
                *CALENDAR_COLUMNS,
                *_fourier_columns(fourier_terms, seasonality),
            )
        )
    return pl.concat(rows).sort("unique_id")
