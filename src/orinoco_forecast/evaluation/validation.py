"""Recompute statistics from exported predictions without retraining."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.evaluation.execution import (
    history_frame,
    load_history,
    observed_frame,
)
from orinoco_forecast.evaluation.metrics import StatisticsResult, compute_statistics
from orinoco_forecast.models.base import PredictionResult
from orinoco_forecast.utils.io import predictions_dir


def read_predictions(path: str | Path, runner: str) -> PredictionResult:
    """Read a ``predictions.parquet`` file and infer its horizon."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de predicciones: {source}. "
            "Ejecuta `scripts/run_model.py` primero."
        )
    frame = pl.read_parquet(source)
    if frame.is_empty():
        raise ValueError(f"El archivo de predicciones está vacío: {source}.")
    missing = [column for column in ("horizon_step", "cutoff") if column not in frame]
    if missing:
        raise ValueError(f"Faltan columnas de predicción en {source}: {missing}.")
    horizon = int(frame["horizon_step"].max())
    return PredictionResult.from_frame(frame, runner=runner, horizon=horizon)


def _validation_slice(
    frame: pl.DataFrame, cutoff: date, horizon: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    history = history_frame(frame, cutoff)
    observed = observed_frame(frame, cutoff, horizon)
    steps = set((observed["ds"] - cutoff).dt.total_days().to_list())
    missing = sorted(set(range(1, horizon + 1)).difference(steps))
    if missing:
        raise ValueError(
            f"El segmento de validación no está completo; faltan los pasos {missing}."
        )
    return history, observed


def validate_predictions(
    model: str,
    against: str = "naive",
    *,
    root: str | Path | None = None,
    frame: pl.DataFrame | None = None,
    cutoff: date | None = None,
) -> StatisticsResult:
    """Score two exported prediction files against the validation segment."""
    base = Path(root) if root is not None else predictions_dir()
    first = read_predictions(base / model / "predictions.parquet", model)
    second = read_predictions(base / against / "predictions.parquet", against)
    if first.horizon != second.horizon:
        raise ValueError(
            f"Los archivos tienen horizontes distintos: {first.horizon} y "
            f"{second.horizon}."
        )
    file_cutoff = first.frame["cutoff"].unique().to_list()
    other_cutoff = second.frame["cutoff"].unique().to_list()
    if len(file_cutoff) != 1 or len(other_cutoff) != 1:
        raise ValueError("Cada archivo debe contener un solo corte.")
    if file_cutoff[0] != other_cutoff[0]:
        raise ValueError(
            f"Los cortes no coinciden: {file_cutoff[0]} y {other_cutoff[0]}."
        )
    selected = file_cutoff[0]
    if cutoff is not None and cutoff != selected:
        raise ValueError(
            f"El corte pedido ({cutoff}) no coincide con el archivo ({selected})."
        )
    data = load_history() if frame is None else frame
    history, observed = _validation_slice(data, selected, first.horizon)
    cfg = load_backtest_config()
    statistics = compute_statistics(
        first,
        second,
        observed=observed,
        history=history,
        seasonality=int(cfg["seasonality"]),
    )
    statistics.write(base / model / "statistics.parquet")
    return statistics
