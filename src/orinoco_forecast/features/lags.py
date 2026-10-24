"""Leakage-safe lag and calendar features for the Orinoco river network.

The long table keeps one row per station and date. A feature dated ``t`` is
built only from a source value dated ``t - lag``; it never reads ``t`` or a
later date. The river order is directional: a station may only consume levels
from stations that are upstream of it.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl

from orinoco_forecast.data.loading import DataQualityWarning

RIVER_ORDER = ("ayacucho", "caicara", "ciudad_bolivar", "palua")
MAX_EXOG_FALLBACK_DAYS = 7
FEATURE_PATTERN = re.compile(r"^(?P<source>.+)_lag_(?P<lag>\d+)$")
CALENDAR_PATTERN = re.compile(r"^(?P<base>sin|cos)_doy(?:_(?P<harmonic>\d+))?$")


def _station_tuple(values: Iterable[str]) -> tuple[str, ...]:
    stations = tuple(str(value).strip() for value in values)
    if not stations:
        raise ValueError("river_order no puede estar vacío.")
    if any(not station for station in stations):
        raise ValueError("river_order no puede contener estaciones vacías.")
    if len(set(stations)) != len(stations):
        raise ValueError(f"river_order no puede contener duplicados: {stations}.")
    return stations


def _lag_tuple(
    values: Iterable[int],
    label: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    try:
        lags = tuple(sorted({int(value) for value in values}))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} debe contener enteros: {values!r}.") from error
    minimum = 0 if allow_zero else 1
    if any(lag < minimum for lag in lags):
        raise ValueError(
            f"{label} no puede contener valores menores que {minimum}: {lags}."
        )
    return lags


@dataclass(frozen=True)
class RiverFeatureConfig:
    """Configuration for the river lag features written to the long table."""

    river_order: tuple[str, ...] = RIVER_ORDER
    min_exog_lag: int = 14
    upstream_lags: tuple[int, ...] = (14, 21, 28)
    target_lags: tuple[int, ...] = (1, 2, 3, 7, 14, 21, 28)
    rain_lags: tuple[int, ...] = (14, 21, 28)
    oni_lags: tuple[int, ...] = (0, 30, 60)
    calendar_terms: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "river_order", _station_tuple(self.river_order))
        object.__setattr__(
            self, "upstream_lags", _lag_tuple(self.upstream_lags, "upstream_lags")
        )
        object.__setattr__(
            self, "target_lags", _lag_tuple(self.target_lags, "target_lags")
        )
        object.__setattr__(self, "rain_lags", _lag_tuple(self.rain_lags, "rain_lags"))
        object.__setattr__(
            self,
            "oni_lags",
            _lag_tuple(self.oni_lags, "oni_lags", allow_zero=True),
        )
        minimum = int(self.min_exog_lag)
        if minimum < 0:
            raise ValueError("min_exog_lag no puede ser negativo.")
        object.__setattr__(self, "min_exog_lag", minimum)
        terms = int(self.calendar_terms)
        if terms < 0:
            raise ValueError("calendar_terms no puede ser negativo.")
        object.__setattr__(self, "calendar_terms", terms)


def load_feature_config(config: dict[str, Any]) -> RiverFeatureConfig:
    """Build :class:`RiverFeatureConfig` from the ``features`` YAML section."""
    settings = config.get("features")
    if settings is None:
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("features debe ser un diccionario.")
    return RiverFeatureConfig(
        river_order=tuple(settings.get("river_order", RIVER_ORDER)),
        min_exog_lag=int(settings.get("min_exog_lag", 14)),
        upstream_lags=tuple(settings.get("upstream_lags", (14, 21, 28))),
        target_lags=tuple(settings.get("target_lags", (1, 2, 3, 7, 14, 21, 28))),
        rain_lags=tuple(settings.get("rain_lags", (14, 21, 28))),
        oni_lags=tuple(settings.get("oni_lags", (0, 30, 60))),
        calendar_terms=int(settings.get("calendar_terms", 1)),
    )


def upstream_of(
    station: str,
    river_order: Sequence[str] = RIVER_ORDER,
) -> tuple[str, ...]:
    """Return the stations upstream of ``station`` in river order."""
    order = _station_tuple(river_order)
    try:
        index = order.index(station)
    except ValueError as error:
        raise ValueError(
            f"La estación {station!r} no está en river_order: {order}."
        ) from error
    return order[:index]


def calendar_columns(terms: int) -> tuple[str, ...]:
    """Return deterministic annual Fourier columns for ``terms`` harmonics."""
    if terms < 0:
        raise ValueError("terms no puede ser negativo.")
    columns: list[str] = []
    for harmonic in range(1, terms + 1):
        suffix = "" if harmonic == 1 else f"_{harmonic}"
        columns.extend((f"sin_doy{suffix}", f"cos_doy{suffix}"))
    return tuple(columns)


def calendar_value(column: str, day: date) -> float:
    """Return the deterministic value of a calendar column on ``day``."""
    match = CALENDAR_PATTERN.fullmatch(column)
    if match is None:
        raise ValueError(f"Columna de calendario desconocida: {column!r}.")
    harmonic = int(match["harmonic"] or 1)
    angle = 2.0 * math.pi * harmonic * day.timetuple().tm_yday / 365.25
    return math.sin(angle) if match["base"] == "sin" else math.cos(angle)


def parse_feature_column(column: str) -> tuple[str, int] | None:
    """Split ``<source>_lag_<n>`` into source and lag; ``None`` otherwise."""
    match = FEATURE_PATTERN.fullmatch(column)
    if match is None:
        return None
    return match["source"], int(match["lag"])


def is_calendar_column(column: str) -> bool:
    """Return whether ``column`` is one of the deterministic calendar terms."""
    return CALENDAR_PATTERN.fullmatch(column) is not None


def _matching_lag_columns(
    columns: Iterable[str],
    source: str,
    min_lag: int,
) -> list[str]:
    matches: list[tuple[int, str]] = []
    for column in columns:
        parsed = parse_feature_column(column)
        if parsed is None:
            continue
        name, lag = parsed
        if name == source and lag >= min_lag:
            matches.append((lag, column))
    return [column for _, column in sorted(matches)]


def _calendar_sort_key(column: str) -> tuple[int, int]:
    match = CALENDAR_PATTERN.fullmatch(column)
    if match is None:
        raise ValueError(f"Columna de calendario desconocida: {column!r}.")
    return (int(match["harmonic"] or 1), 0 if match["base"] == "sin" else 1)


def select_exog_columns(
    columns: Iterable[str],
    *,
    station: str,
    river_order: Sequence[str] = RIVER_ORDER,
    min_exog_lag: int = 14,
    use_upstream: bool = True,
    use_own_lags: bool = True,
    use_rain: bool = True,
    use_oni: bool = True,
    use_calendar: bool = True,
) -> tuple[str, ...]:
    """Select the leakage-safe exog columns a station may consume.

    Upstream levels, own safe lags, own rain and ONI keep only lags
    ``>= min_exog_lag``. Calendar terms are deterministic and need no lag.
    Own lags below the cutoff horizon stay out of ARIMAX: the AR terms carry
    short-term inertia and those lags would not be known for every step.
    """
    if min_exog_lag < 0:
        raise ValueError("min_exog_lag no puede ser negativo.")
    available = set(columns)
    selected: list[str] = []
    if use_upstream:
        for source in upstream_of(station, river_order):
            selected.extend(_matching_lag_columns(available, source, min_exog_lag))
    if use_own_lags:
        selected.extend(_matching_lag_columns(available, station, min_exog_lag))
    if use_rain:
        selected.extend(
            _matching_lag_columns(available, f"rain_{station}", min_exog_lag)
        )
    if use_oni:
        selected.extend(_matching_lag_columns(available, "oni", min_exog_lag))
    if use_calendar:
        calendar = [column for column in available if is_calendar_column(column)]
        selected.extend(sorted(calendar, key=_calendar_sort_key))
    return tuple(selected)


def _validate_levels(levels: pl.DataFrame) -> None:
    required = {"unique_id", "ds", "y"}
    missing = required.difference(levels.columns)
    if missing:
        raise ValueError(f"Faltan columnas de niveles: {sorted(missing)}")
    if levels.schema["unique_id"] != pl.String:
        raise ValueError(
            f"unique_id debe ser pl.String, no {levels.schema['unique_id']}."
        )
    if levels.schema["ds"] != pl.Date:
        raise ValueError(f"ds debe ser pl.Date, no {levels.schema['ds']}.")
    if not levels.schema["y"].is_numeric():
        raise ValueError(f"y debe ser numérica, no {levels.schema['y']}.")
    duplicates = levels.select(
        pl.struct(["unique_id", "ds"]).is_duplicated().sum()
    ).item()
    if duplicates:
        raise ValueError(
            f"{duplicates} filas duplicadas por unique_id y ds; no se pueden "
            "construir rezagos sin ambigüedad."
        )


def _external_frame(levels: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    return (
        levels.select("ds", *columns)
        .group_by("ds")
        .agg(*(pl.col(column).drop_nulls().first().alias(column) for column in columns))
        .sort("ds")
    )


def _lag_frame(
    source: pl.DataFrame,
    column: str,
    lags: Sequence[int],
    prefix: str,
) -> pl.DataFrame:
    if not lags:
        raise ValueError("lags no puede estar vacío.")
    frames = [
        source.select(
            (pl.col("ds") + pl.duration(days=lag)).alias("ds"),
            pl.col(column).cast(pl.Float64).alias(f"{prefix}{lag}"),
        )
        for lag in lags
    ]
    result = frames[0]
    for frame in frames[1:]:
        result = result.join(frame, on="ds", how="full", coalesce=True)
    return result.sort("ds")


def _calendar_frame(dates: pl.Series, terms: int) -> pl.DataFrame:
    expressions = []
    day_of_year = pl.col("ds").dt.ordinal_day().cast(pl.Float64)
    for harmonic in range(1, terms + 1):
        suffix = "" if harmonic == 1 else f"_{harmonic}"
        angle = pl.lit(2.0 * math.pi * harmonic) * day_of_year / pl.lit(365.25)
        expressions.extend(
            (
                angle.sin().alias(f"sin_doy{suffix}"),
                angle.cos().alias(f"cos_doy{suffix}"),
            )
        )
    return pl.DataFrame({"ds": dates}).select("ds", *expressions)


def add_river_features(
    levels: pl.DataFrame,
    config: RiverFeatureConfig | None = None,
) -> pl.DataFrame:
    """Append leakage-safe station, rain, ONI and calendar features.

    Station levels are written as ``<station>_lag_<n>``, rain as
    ``rain_<station>_lag_<n>``, ONI as ``oni_lag_<n>`` and the annual cycle as
    ``sin_doy``/``cos_doy`` plus optional harmonics. The input frame is not
    modified; the returned frame preserves its columns and sorts by
    ``unique_id, ds``.
    """
    config = config or RiverFeatureConfig()
    _validate_levels(levels)
    present = set(levels["unique_id"].unique().to_list())
    missing = [station for station in config.river_order if station not in present]
    if missing:
        raise ValueError(
            f"Faltan estaciones de river_order en el dataset: {missing}. "
            "Incluye todas las estaciones del río antes de construir features."
        )

    wide = (
        levels.select("ds", "unique_id", "y")
        .pivot(index="ds", on="unique_id", values="y", aggregate_function="first")
        .sort("ds")
    )
    station_lags = tuple(sorted(set(config.target_lags) | set(config.upstream_lags)))
    feature_frames: list[pl.DataFrame] = []
    for station in config.river_order:
        if station_lags:
            feature_frames.append(
                _lag_frame(
                    wide.select("ds", station),
                    station,
                    station_lags,
                    f"{station}_lag_",
                )
            )
    for station in config.river_order:
        column = f"rain_{station}"
        if config.rain_lags and column in levels.columns:
            feature_frames.append(
                _lag_frame(
                    _external_frame(levels, [column]),
                    column,
                    config.rain_lags,
                    f"{column}_lag_",
                )
            )
    if config.oni_lags:
        if "oni" not in levels.columns:
            raise ValueError(
                "La configuración pide rezagos de ONI, pero falta la columna 'oni'."
            )
        feature_frames.append(
            _lag_frame(
                _external_frame(levels, ["oni"]),
                "oni",
                config.oni_lags,
                "oni_lag_",
            )
        )
    if config.calendar_terms:
        feature_frames.append(_calendar_frame(wide["ds"], config.calendar_terms))
    if not feature_frames:
        return levels.sort("unique_id", "ds")

    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.join(frame, on="ds", how="full", coalesce=True)
    features = features.sort("ds")
    overlaps = (set(features.columns) - {"ds"}).intersection(levels.columns)
    if overlaps:
        raise ValueError(
            f"El dataset ya contiene columnas de features: {sorted(overlaps)}."
        )
    return levels.join(features, on="ds", how="left").sort("unique_id", "ds")


def raw_predictors(history: pl.DataFrame) -> pl.DataFrame:
    """Return one row per date with raw station levels, rain and ONI."""
    _validate_levels(history)
    wide = (
        history.select("ds", "unique_id", "y")
        .pivot(index="ds", on="unique_id", values="y", aggregate_function="first")
        .sort("ds")
    )
    external = [
        column
        for column in history.columns
        if (column.startswith("rain_") and "_lag_" not in column) or column == "oni"
    ]
    if not external:
        return wide
    values = _external_frame(history, external)
    return wide.join(values, on="ds", how="left").sort("ds")


def _lookup_raw_value(
    by_date: dict[date, dict[str, Any]],
    source: str,
    source_date: date,
    *,
    station: str,
    column: str,
) -> float:
    """Read a raw source at ``source_date`` or the last previous finite value.

    A missing source date is a data-quality event, not a reason to leak future
    data or to impute the whole table. The fallback is local to the forecast
    exog and always reported.
    """
    for offset in range(MAX_EXOG_FALLBACK_DAYS + 1):
        candidate = source_date - timedelta(days=offset)
        row = by_date.get(candidate)
        if row is None:
            continue
        value = row.get(source)
        if value is None or not math.isfinite(float(value)):
            continue
        if offset:
            warnings.warn(
                f"{station}: {source!r} no tiene valor en {source_date}; se usa "
                f"el último valor observado en {candidate} para construir la "
                f"exógena {column!r}.",
                DataQualityWarning,
                stacklevel=3,
            )
        return float(value)
    raise ValueError(
        f"{station}: falta un valor finito de {source!r} en {source_date} "
        f"(y en los {MAX_EXOG_FALLBACK_DAYS} días anteriores) para construir "
        f"la exógena {column!r}."
    )


def future_exog_matrix(
    raw: pl.DataFrame,
    columns: Sequence[str],
    *,
    station: str,
    cutoff: date,
    horizon: int,
) -> np.ndarray:
    """Build future exog rows using only values observed at or before cutoff.

    For a forecast step ``h`` and a column with lag ``L``, the value is the raw
    source at ``cutoff + h - L``. ``L >= h`` is required, so that date is never
    later than ``cutoff``.
    """
    if type(cutoff) is not date:
        raise ValueError("cutoff debe ser datetime.date, sin hora.")
    if horizon < 1:
        raise ValueError("horizon debe ser positivo.")
    if "ds" not in raw.columns or raw.schema["ds"] != pl.Date:
        raise ValueError("raw debe contener una columna ds de tipo pl.Date.")
    if not columns:
        return np.empty((horizon, 0), dtype=float)
    by_date = {row["ds"]: row for row in raw.iter_rows(named=True)}
    rows: list[list[float]] = []
    for step in range(1, horizon + 1):
        target_date = cutoff + timedelta(days=step)
        values: list[float] = []
        for column in columns:
            if is_calendar_column(column):
                values.append(calendar_value(column, target_date))
                continue
            parsed = parse_feature_column(column)
            if parsed is None:
                raise ValueError(f"{station}: columna exógena desconocida {column!r}.")
            source, lag = parsed
            if lag < step:
                raise ValueError(
                    f"{station}: la columna {column!r} usa un rezago {lag} menor "
                    f"que el paso {step}; no se puede conocer en el corte."
                )
            source_date = target_date - timedelta(days=lag)
            values.append(
                _lookup_raw_value(
                    by_date,
                    source,
                    source_date,
                    station=station,
                    column=column,
                )
            )
        rows.append(values)
    return np.asarray(rows, dtype=float)
