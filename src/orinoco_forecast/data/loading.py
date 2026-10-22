"""Read-only loaders for the sources registered in ``configs/data.yaml``.

Every loader returns a long frame ``unique_id`` (str), ``ds`` (Date), ``y``
(Float64) sorted by ``unique_id, ds``. Anomalies are reported with
``DataQualityWarning`` and never corrected.
"""

import re
import warnings
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

import polars as pl
import yaml

LONG_SCHEMA = {"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64}


class DataQualityWarning(UserWarning):
    """Anomalía detectada en una fuente cruda (se reporta, no se corrige)."""


@cache
def repo_root() -> Path:
    """Devuelve la raíz del repositorio (directorio que contiene pyproject.toml)."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("No se encontró pyproject.toml sobre el paquete.")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee ``configs/data.yaml`` (o ``path``) y devuelve el diccionario."""
    path = Path(path) if path else repo_root() / "configs" / "data.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_source(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Devuelve la entrada de ``sources`` con ese ``name``; KeyError si no existe."""
    config = config or load_config()
    for source in config["sources"]:
        if source["name"] == name:
            return source
    raise KeyError(f"Fuente no registrada en data.yaml: {name!r}")


def _source_by_role(role: str, config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or load_config()
    matches = [s for s in config["sources"] if s.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"Se esperaba una fuente con role={role!r}: {len(matches)}")
    return matches[0]


def load_raw_excel(path: str | Path, sheet_name: str) -> pl.DataFrame:
    """Lee una hoja como texto, sin inferencia; la primera fila es el encabezado.

    Las columnas sin nombre se llaman ``column_<i>``. El casteo queda a cargo del
    llamador, para que los dtypes sean explícitos.
    """
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"Fuente declarada no encontrada: {path}")
    with warnings.catch_warnings():
        # Print header/footer metadata of the workbook, unrelated to the data.
        warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        raw = pl.read_excel(
            path,
            sheet_name=sheet_name,
            engine="openpyxl",
            has_header=False,
            infer_schema_length=0,
            drop_empty_rows=False,
            drop_empty_cols=False,
        )
    header = [v if v is not None else f"column_{i}" for i, v in enumerate(raw.row(0))]
    return raw.slice(1).rename(dict(zip(raw.columns, header, strict=True)))


def validate_daily_dates(dates: pl.Series, label: str) -> None:
    """Emite warnings si ``dates`` tiene duplicados, desorden o huecos diarios."""
    if (n_dup := dates.is_duplicated().sum()) > 0:
        warnings.warn(
            f"{label}: {n_dup} filas con fechas duplicadas.",
            DataQualityWarning,
            stacklevel=2,
        )
    if not dates.is_sorted():
        warnings.warn(
            f"{label}: fechas fuera de orden cronológico.",
            DataQualityWarning,
            stacklevel=2,
        )
    unique = dates.drop_nulls().unique()
    if unique.len() > 0:
        expected = (unique.max() - unique.min()).days + 1
        if (n_missing := expected - unique.len()) > 0:
            warnings.warn(
                f"{label}: {n_missing} días ausentes (huecos en la frecuencia diaria).",
                DataQualityWarning,
                stacklevel=2,
            )


def _warn_nulls(long: pl.DataFrame, label: str) -> None:
    counts = long.group_by("unique_id").agg(pl.col("y").null_count()).sort("unique_id")
    for uid, n_null in counts.iter_rows():
        if n_null:
            warnings.warn(
                f"{label}/{uid}: {n_null} valores nulos.",
                DataQualityWarning,
                stacklevel=3,
            )


def load_primary(config: dict[str, Any] | None = None) -> pl.DataFrame:
    """Carga la fuente ``role: primary`` (una columna por estación) en formato largo."""
    return _load_wide_stations(_source_by_role("primary", config))


def load_historical(config: dict[str, Any] | None = None) -> pl.DataFrame:
    """Carga la fuente ``role: historical`` (años en columnas) en formato largo.

    Usa el calendario real: el libro no tiene 29 de febrero, así que esas filas se
    insertan con ``y`` nulo y se avisa una sola vez.
    """
    return _load_wide_years(_source_by_role("historical", config))


def load_source(name: str, config: dict[str, Any] | None = None) -> pl.DataFrame:
    """Carga una fuente por ``name`` según su ``role``/``layout`` en data.yaml."""
    source = get_source(name, config)
    role = source.get("role")
    if role == "duplicate":
        raise ValueError(f"{name!r} es una fuente duplicada; no se carga.")
    if source.get("layout") == "wide_years":
        return _load_wide_years(source)
    if role == "primary":
        return _load_wide_stations(source)
    raise ValueError(f"Fuente {name!r} sin loader para role={role!r}.")


def _load_wide_stations(source: dict[str, Any]) -> pl.DataFrame:
    raw = load_raw_excel(source["path"], source["sheet"])
    stations = [source["target_column"], *source.get("covariates", [])]
    wide = raw.select(
        pl.col(source["date_column"])
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=True)
        .dt.date()
        .alias("ds"),
        *(pl.col(s).str.strip_chars().cast(pl.Float64, strict=True) for s in stations),
    ).filter(~pl.all_horizontal(pl.all().is_null()))
    validate_daily_dates(wide["ds"], source["name"])
    long = (
        wide.unpivot(index="ds", on=stations, variable_name="unique_id", value_name="y")
        .select(LONG_SCHEMA.keys())
        .cast(LONG_SCHEMA)
        .sort("unique_id", "ds")
    )
    _warn_nulls(long, source["name"])
    return long


def _load_wide_years(source: dict[str, Any]) -> pl.DataFrame:
    name, parts = source["name"], source["date_parts"]
    first, last = parts["first_year"], parts["last_year"]
    pattern = re.compile(parts["year_column_pattern"])
    raw = load_raw_excel(source["path"], source["sheet"])
    years = [c for c in raw.columns if pattern.match(c) and first <= int(c) <= last]
    if len(years) != last - first + 1:
        raise ValueError(f"{name}: columnas de año incompletas ({len(years)}).")

    # Calendar rows carry a day of month; summary rows (PROMEDIO, ...) do not.
    calendar = (
        raw.with_columns(
            pl.col(parts["day_column"])
            .str.strip_chars()
            .cast(pl.Int8, strict=False)
            .alias("day"),
            pl.col(parts["month_column"])
            .str.strip_chars()
            .forward_fill()
            .replace_strict(parts["month_mapping"], return_dtype=pl.Int8)
            .alias("month"),
        )
        .filter(pl.col("day").is_not_null())
        .select("month", "day", *years)
    )
    if calendar.select("month", "day").is_duplicated().any():
        warnings.warn(
            f"{name}: días del calendario duplicados.", DataQualityWarning, stacklevel=3
        )
    observed = calendar.unpivot(
        index=["month", "day"], on=years, variable_name="year", value_name="y"
    ).select(
        pl.date(pl.col("year").cast(pl.Int32), "month", "day").alias("ds"),
        pl.col("y").str.strip_chars().cast(pl.Float64, strict=True),
    )
    full = pl.date_range(date(first, 1, 1), date(last, 12, 31), "1d", eager=True)
    n_feb29 = full.filter((full.dt.month() == 2) & (full.dt.day() == 29)).len()
    has_feb29 = calendar.filter((pl.col("month") == 2) & (pl.col("day") == 29)).height
    if n_feb29 and not has_feb29:
        warnings.warn(
            f"{name}: la fuente no contiene 29 de febrero; "
            f"se insertan {n_feb29} filas con y nulo.",
            DataQualityWarning,
            stacklevel=3,
        )
    long = (
        full.alias("ds")
        .to_frame()
        .join(observed, on="ds", how="left")
        .select(pl.lit(source["unique_id"]).alias("unique_id"), "ds", "y")
        .cast(LONG_SCHEMA)
        .sort("unique_id", "ds")
    )
    _warn_nulls(long, name)
    return long
