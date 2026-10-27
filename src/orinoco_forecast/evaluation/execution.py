"""Execute one runner against a comparison runner on a single cutoff.

The runner only fits and predicts. This module owns data loading, window
construction, the validation slice and the statistics call, so the CLI and the
tests share the same path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from orinoco_forecast.data.datasets import TimeSeriesDataset
from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.evaluation.metrics import StatisticsResult, compute_statistics
from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    PredictionResult,
    Runner,
    validate_horizon,
)
from orinoco_forecast.models.registry import get_runner

TARGET_STATIONS = ("ciudad_bolivar", "palua")
UPSTREAM_STATIONS = ("ayacucho", "caicara")
INPUT_LENGTH = 28
FEATURE_COLUMNS = ("y", "rain_ciudad_bolivar", "rain_palua", "oni")
REQUIRED_COLUMNS = ("unique_id", "ds", "y", *FEATURE_COLUMNS)


@dataclass(frozen=True)
class ExecutionResult:
    """One runner, its comparison runner and their validation statistics."""

    prediction: PredictionResult
    baseline: PredictionResult
    statistics: StatisticsResult
    windows: pl.DataFrame
    cutoff: date
    horizon: int

    def write(self) -> tuple[Path, Path, Path]:
        """Write the model prediction, the baseline prediction and the statistics."""
        return (
            self.prediction.write(),
            self.baseline.write(),
            self.statistics.write(),
        )


def load_history(path: str | Path | None = None) -> pl.DataFrame:
    """Load the long Parquet dataset for the target and upstream stations."""
    dataset = TimeSeriesDataset(path) if path else TimeSeriesDataset()
    try:
        reader = dataset.interface()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"No se encontró el Parquet del dataset ({dataset.output_path}). "
            "Genéralo con `uv run python scripts/run_model.py <model> "
            "--write-dataset` o con `TimeSeriesDataset().write_parquet()`."
        ) from error
    frame = reader.read(unique_ids=[*UPSTREAM_STATIONS, *TARGET_STATIONS])
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"El dataset no contiene las columnas requeridas: {missing}. "
            "Regenéralo con `TimeSeriesDataset().write_parquet()` (incluye "
            "lluvia y ONI por defecto)."
        )
    return frame.sort("unique_id", "ds")


def default_cutoff(cfg: dict[str, Any], horizon: int) -> date:
    """Last cutoff whose validation ends the day before ``test_start``."""
    test_start = date.fromisoformat(str(cfg["test_start"]))
    return test_start - timedelta(days=horizon + 1)


def resolve_cutoff(
    frame: pl.DataFrame,
    cfg: dict[str, Any],
    cutoff: date | None,
    horizon: int,
) -> date:
    """Validate an explicit cutoff or use the protocol default."""
    if cutoff is None:
        cutoff = default_cutoff(cfg, horizon)
    if type(cutoff) is not date:
        raise ValueError(f"cutoff debe ser datetime.date, no {type(cutoff).__name__}.")
    targets = frame.filter(pl.col("unique_id").is_in(TARGET_STATIONS))
    if targets.is_empty():
        raise ValueError("El dataset no contiene las estaciones objetivo.")
    if targets["ds"].max() < cutoff:
        raise ValueError(
            f"El corte {cutoff} es posterior a la última fecha del dataset "
            f"({targets['ds'].max()})."
        )
    return cutoff


def build_windows_for(
    frame: pl.DataFrame,
    *,
    cutoff: date,
    horizon: int,
    input_length: int = INPUT_LENGTH,
) -> pl.DataFrame:
    """Supervised windows for the target stations with rain and ONI features."""
    targets = frame.filter(pl.col("unique_id").is_in(TARGET_STATIONS))
    return build_windows(
        targets,
        cutoff=cutoff,
        input_length=input_length,
        horizon=horizon,
        feature_columns=FEATURE_COLUMNS,
    )


def history_frame(frame: pl.DataFrame, cutoff: date) -> pl.DataFrame:
    """Observations known at the cutoff, including upstream stations for ARIMAX."""
    return frame.filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")


def observed_frame(frame: pl.DataFrame, cutoff: date, horizon: int) -> pl.DataFrame:
    """Observed target values for ``cutoff + 1 .. cutoff + horizon``."""
    return frame.filter(
        pl.col("unique_id").is_in(TARGET_STATIONS)
        & (pl.col("ds") > cutoff)
        & (pl.col("ds") <= cutoff + timedelta(days=horizon))
    ).select("unique_id", "ds", "y")


def _target_prediction(result: PredictionResult) -> PredictionResult:
    """Keep only the two forecast targets, never the upstream covariates."""
    frame = result.frame.filter(pl.col("unique_id").is_in(TARGET_STATIONS))
    return PredictionResult.from_frame(
        frame, runner=result.runner, horizon=result.horizon
    )


def require_implemented(runner: Any, label: str) -> Runner:
    """Reject structure-only runners before any data is loaded or fitted."""
    if not getattr(runner, "implemented", True):
        raise ValueError(
            f"El runner {runner.name!r} ({label}) solo tiene estructura; "
            "todavía no se puede ejecutar."
        )
    if not hasattr(runner, "fit") or not hasattr(runner, "predict"):
        raise ValueError(
            f"El runner {getattr(runner, 'name', label)!r} no implementa el "
            "contrato fit/predict."
        )
    return runner


def run_runners(
    runner: Runner,
    against: Runner,
    frame: pl.DataFrame,
    *,
    cutoff: date,
    horizon: int = DEFAULT_HORIZON,
    input_length: int = INPUT_LENGTH,
    seasonality: int = 365,
) -> ExecutionResult:
    """Fit and predict both runners, then score the observed validation slice."""
    validate_horizon(horizon)
    require_implemented(runner, "modelo")
    require_implemented(against, "comparación")
    history = history_frame(frame, cutoff)
    observed = observed_frame(frame, cutoff, horizon)
    steps = set((observed["ds"] - cutoff).dt.total_days().to_list())
    missing = sorted(set(range(1, horizon + 1)).difference(steps))
    if missing:
        raise ValueError(
            f"El segmento de validación no está completo; faltan los pasos {missing}. "
            "Elige un corte con observaciones hasta cutoff + horizonte."
        )
    windows = build_windows_for(
        frame, cutoff=cutoff, horizon=horizon, input_length=input_length
    )
    runner.fit(windows, history, cutoff=cutoff, horizon=horizon)
    prediction = _target_prediction(
        runner.predict(history, cutoff=cutoff, horizon=horizon)
    )
    against.fit(windows, history, cutoff=cutoff, horizon=horizon)
    baseline = _target_prediction(
        against.predict(history, cutoff=cutoff, horizon=horizon)
    )
    statistics = compute_statistics(
        prediction,
        baseline,
        observed=observed,
        history=history,
        seasonality=seasonality,
    )
    return ExecutionResult(
        prediction=prediction,
        baseline=baseline,
        statistics=statistics,
        windows=windows,
        cutoff=cutoff,
        horizon=horizon,
    )


def execute(
    model: str,
    against: str = "naive",
    *,
    frame: pl.DataFrame | None = None,
    cutoff: date | None = None,
    horizon: int | None = None,
    input_length: int | None = None,
    write: bool = True,
) -> ExecutionResult:
    """Run ``model`` against ``against`` with the shared protocol."""
    # Local import: ``evaluation.backtest`` imports this module to run the
    # walk-forward sweep, so a module-level import would create a cycle.
    from orinoco_forecast.evaluation.backtest import load_backtest_config

    cfg = load_backtest_config()
    horizon = int(cfg["horizon"]) if horizon is None else validate_horizon(horizon)
    input_length = (
        int(cfg.get("input_length", INPUT_LENGTH))
        if input_length is None
        else input_length
    )
    runner = require_implemented(get_runner(model), "modelo")
    baseline = require_implemented(get_runner(against), "comparación")
    data = load_history() if frame is None else frame
    cutoff = resolve_cutoff(data, cfg, cutoff, horizon)
    result = run_runners(
        runner,
        baseline,
        data,
        cutoff=cutoff,
        horizon=horizon,
        input_length=input_length,
        seasonality=int(cfg["seasonality"]),
    )
    if write:
        result.write()
    return result
