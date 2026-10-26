"""Read-only loaders for the sources registered in ``configs/data.yaml``.

Every loader returns a long frame ``unique_id`` (str), ``ds`` (Date), ``y``
(Float64) sorted by ``unique_id, ds``. Anomalies are reported with
``DataQualityWarning`` and never corrected. The level workbook is parsed by
``TimeSeriesDataset.extract_levels`` (``data/datasets.py``).
"""

import warnings
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


def load_primary(config: dict[str, Any] | None = None) -> pl.DataFrame:
    """Carga la fuente ``role: primary`` (una columna por estación) en formato largo."""
    return _load_levels(_source_by_role("primary", config))


def load_source(name: str, config: dict[str, Any] | None = None) -> pl.DataFrame:
    """Carga una fuente por ``name`` según su ``role`` en data.yaml."""
    source = get_source(name, config)
    role = source.get("role")
    if role == "duplicate":
        raise ValueError(f"{name!r} es una fuente duplicada; no se carga.")
    if role == "primary":
        return _load_levels(source)
    raise ValueError(f"Fuente {name!r} sin loader para role={role!r}.")


def _load_levels(source: dict[str, Any]) -> pl.DataFrame:
    # Lazy import: data.datasets imports this module.
    from orinoco_forecast.data.datasets import TimeSeriesDataset

    long = TimeSeriesDataset().extract_levels(source["path"], source["sheet"])
    return long.select(LONG_SCHEMA.keys()).cast(LONG_SCHEMA).sort("unique_id", "ds")
