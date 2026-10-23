"""Validación del parquet procesado antes de entrenar modelos."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import polars as pl

from orinoco_forecast.data.loading import DataQualityWarning, load_config, repo_root

PROCESSED_SCHEMA = {
    "unique_id": pl.String,
    "ds": pl.Date,
    "y": pl.Float64,
    "rain_ayacucho": pl.Float64,
    "rain_caicara": pl.Float64,
    "rain_ciudad_bolivar": pl.Float64,
    "rain_palua": pl.Float64,
    "oni": pl.Float64,
}


def validate_processed_dataset(
    path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Valida el parquet procesado y devuelve el frame sin modificarlo.

    Las anomalías de calidad se reportan como ``DataQualityWarning``. Los
    errores estructurales, como columnas o tipos incorrectos, detienen la
    validación porque el dataset no puede entrar al entrenamiento.
    """
    config = config or load_config()
    parquet_path = Path(path) if path else repo_root() / "data" / "processed" / (
        "orinoco_exogenous.parquet"
    )
    if not parquet_path.is_absolute():
        parquet_path = repo_root() / parquet_path
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet procesado no encontrado: {parquet_path}")

    frame = pl.read_parquet(parquet_path)
    missing = set(PROCESSED_SCHEMA).difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    actual_schema = {column: frame.schema[column] for column in PROCESSED_SCHEMA}
    if actual_schema != PROCESSED_SCHEMA:
        raise ValueError(
            f"Esquema incorrecto. Esperado: {PROCESSED_SCHEMA}; "
            f"recibido: {actual_schema}"
        )

    targets = set(config["targets"])
    found_targets = set(frame["unique_id"].unique().to_list())
    if not targets.issubset(found_targets):
        raise ValueError(
            f"Faltan targets configurados: {sorted(targets - found_targets)}"
        )

    duplicate_count = frame.select(
        pl.struct(["unique_id", "ds"]).is_duplicated().sum()
    ).item()
    if duplicate_count:
        warnings.warn(
            f"El parquet contiene {duplicate_count} filas duplicadas por "
            "unique_id y ds.",
            DataQualityWarning,
            stacklevel=2,
        )

    if frame.sort("unique_id", "ds").equals(frame) is False:
        warnings.warn(
            "El parquet no está ordenado por unique_id y ds.",
            DataQualityWarning,
            stacklevel=2,
        )

    for unique_id, station in frame.partition_by("unique_id", as_dict=True).items():
        label = unique_id[0] if isinstance(unique_id, tuple) else unique_id
        dates = station["ds"]
        unique_dates = dates.unique()
        expected_days = (unique_dates.max() - unique_dates.min()).days + 1
        missing_days = expected_days - unique_dates.len()
        if missing_days:
            warnings.warn(
                f"{label}: {missing_days} días ausentes en el rango temporal.",
                DataQualityWarning,
                stacklevel=2,
            )
        null_target = station["y"].null_count()
        if null_target:
            warnings.warn(
                f"{label}: {null_target} valores nulos en y; se conservarán "
                "y no se usarán como objetivos de entrenamiento.",
                DataQualityWarning,
                stacklevel=2,
            )

    for column in PROCESSED_SCHEMA:
        if column not in {"unique_id", "ds", "y"}:
            nulls = frame[column].null_count()
            if nulls:
                warnings.warn(
                    f"{column}: {nulls} valores nulos en la covariable.",
                    DataQualityWarning,
                    stacklevel=2,
                )
    return frame
