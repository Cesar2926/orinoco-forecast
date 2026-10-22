"""Carga y alineación diaria de las covariables externas del proyecto."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import polars as pl

from orinoco_forecast.data.loading import DataQualityWarning, repo_root

ONI_SEASON_END_MONTH = {
    "DJF": 2,
    "JFM": 3,
    "FMA": 4,
    "MAM": 5,
    "AMJ": 6,
    "MJJ": 7,
    "JJA": 8,
    "JAS": 9,
    "ASO": 10,
    "SON": 11,
    "OND": 12,
    "NDJ": 1,
}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def _warn_nulls(frame: pl.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        count = frame[column].null_count()
        if count:
            warnings.warn(
                f"{label}: {count} valores nulos en {column}.",
                DataQualityWarning,
                stacklevel=3,
            )


def load_nasa_power(
    files: dict[str, str | Path],
    variable: str = "PRECTOTCORR",
) -> pl.DataFrame:
    """Lee los CSV de NASA POWER y devuelve ``ds`` más lluvia por estación."""
    result: pl.DataFrame | None = None
    for station, source in files.items():
        path = _path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Fuente NASA POWER no encontrada: {path}")
        raw = pl.read_csv(
            path,
            skip_rows=9,
            schema_overrides={"YEAR": pl.Int32, "DOY": pl.Int16, variable: pl.Float64},
        )
        expected = {"YEAR", "DOY", variable}
        missing = expected.difference(raw.columns)
        if missing:
            raise ValueError(f"{path}: faltan columnas NASA POWER: {sorted(missing)}")
        station_frame = raw.select(
            pl.date(
                pl.col("YEAR"),
                pl.lit(1),
                pl.lit(1),
            )
            .dt.offset_by((pl.col("DOY") - 1).cast(pl.String) + "d")
            .alias("ds"),
            pl.when(pl.col(variable) == -999)
            .then(None)
            .otherwise(pl.col(variable))
            .alias(f"rain_{station}"),
        ).sort("ds")
        _warn_nulls(station_frame, [f"rain_{station}"], f"NASA POWER/{station}")
        result = (
            station_frame
            if result is None
            else result.join(station_frame, on="ds", how="full", coalesce=True)
        )
    if result is None:
        raise ValueError("No se proporcionaron archivos NASA POWER.")
    return result.sort("ds")


def load_oni(
    path: str | Path,
    publication_lag_months: int = 3,
) -> pl.DataFrame:
    """Lee ONI, lo fecha por el fin de cada estación y aplica el desfase configurado.

    NOAA publica ONI como promedios móviles de tres meses. Se usa el mes central
    representado por la estación y se desplaza antes de hacer el ``forward fill``,
    para no exponer información que aún no estaría publicada.
    """
    source = _path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Fuente NOAA ONI no encontrada: {source}")
    rows: list[tuple[int, str, float]] = []
    pattern = re.compile(
        r"^\s*(?P<season>[A-Z]{3})\s+(?P<year>\d{4})\s+"
        r"(?P<total>-?\d+(?:\.\d+)?)\s+(?P<anom>-?\d+(?:\.\d+)?)\s*$"
    )
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                rows.append(
                    (
                        int(match["year"]),
                        match["season"],
                        float(match["anom"]),
                    )
                )
    if not rows:
        raise ValueError(f"{source}: no se encontraron filas ONI.")
    monthly = pl.DataFrame(
        rows, schema=["year", "season", "oni"], orient="row"
    ).with_columns(
        pl.col("season")
        .replace_strict(ONI_SEASON_END_MONTH, return_dtype=pl.Int8)
        .alias("month")
    )
    monthly = (
        monthly.with_columns(
            pl.when(pl.col("season") == "NDJ")
            .then(pl.col("year") + 1)
            .otherwise(pl.col("year"))
            .alias("date_year")
        )
        .select(
            pl.date("date_year", "month", pl.lit(1))
            .dt.offset_by(f"{publication_lag_months}mo")
            .alias("ds"),
            "oni",
        )
        .sort("ds")
    )
    if monthly.select(pl.col("ds").is_duplicated()).to_series().any():
        warnings.warn(
            "NOAA ONI: hay fechas mensuales duplicadas.",
            DataQualityWarning,
            stacklevel=2,
        )
    start, end = monthly["ds"].min(), monthly["ds"].max()
    daily = pl.DataFrame({"ds": pl.date_range(start, end, "1d", eager=True)}).join(
        monthly, on="ds", how="left"
    )
    return daily.with_columns(pl.col("oni").forward_fill()).sort("ds")


def add_exogenous(
    levels: pl.DataFrame,
    nasa: pl.DataFrame,
    oni: pl.DataFrame,
) -> pl.DataFrame:
    """Añade lluvia y ONI a una tabla larga de niveles sin modificar ``y``."""
    required = {"unique_id", "ds", "y"}
    missing = required.difference(levels.columns)
    if missing:
        raise ValueError(f"Faltan columnas de niveles: {sorted(missing)}")
    rain_columns = [column for column in nasa.columns if column != "ds"]
    rain = nasa.unpivot(
        index="ds", on=rain_columns, variable_name="rain_source", value_name="rain"
    )
    rain = rain.with_columns(
        pl.col("rain_source").str.replace("^rain_", "").alias("station")
    )
    stations = rain["station"].unique().to_list()
    rain = rain.pivot(index="ds", on="station", values="rain").rename(
        {station: f"rain_{station}" for station in stations}
    )
    aligned = (
        levels.join(rain, on="ds", how="left")
        .join(oni, on="ds", how="left")
        .sort("unique_id", "ds")
    )
    exogenous = [*rain_columns, "oni"]
    _warn_nulls(aligned, exogenous, "Covariables externas alineadas")
    return aligned


def load_exogenous_config(config: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carga NASA POWER y ONI desde la sección ``exogenous`` de la configuración."""
    settings = config["exogenous"]
    nasa_settings = settings["nasa_power"]
    nasa = load_nasa_power(nasa_settings["files"], nasa_settings["variable"])
    oni_settings = settings["noaa_oni"]
    oni = load_oni(oni_settings["path"], oni_settings.get("publication_lag_months", 3))
    return nasa, oni
