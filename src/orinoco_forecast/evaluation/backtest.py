"""Shared execution protocol (``configs/backtest.yaml``) and walk-forward sweeps.

The protocol values used by the runner execution path live here: horizon, input
length, test start, seasonality and the walk-forward folds. ``run_backtest``
reuses ``run_runners`` for every fold, so the model and its comparison runner
are always fitted and scored on exactly the same cutoff, windows and validation
segment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.splits import WalkForwardSplit, walk_forward_splits
from orinoco_forecast.evaluation.execution import (
    INPUT_LENGTH,
    TARGET_STATIONS,
    default_cutoff,
    observed_frame,
    require_implemented,
    run_runners,
)
from orinoco_forecast.evaluation.metrics import STATISTICS_COLUMNS
from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    Runner,
    validate_horizon,
    validate_runner_name,
)
from orinoco_forecast.models.registry import get_runner

DEFAULT_STATION = "ciudad_bolivar"
PROTOCOL_KEYS = ("test_start", "n_windows", "step_size")
BACKTEST_PREDICTION_COLUMNS = (
    "fold",
    "model",
    "unique_id",
    "cutoff",
    "ds",
    "horizon_step",
    "y_pred",
    "y_true",
)
BACKTEST_STATISTICS_COLUMNS = ("fold", "cutoff", *STATISTICS_COLUMNS)


def load_backtest_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs/backtest.yaml`` (or ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "backtest.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _protocol_int(cfg: dict[str, Any], key: str) -> int:
    """Read a positive integer protocol value, naming the missing key."""
    if key not in cfg:
        raise ValueError(
            f"El protocolo no define {key!r}; agrégalo a configs/backtest.yaml."
        )
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key!r} debe ser un entero, no {value!r}.")
    if value <= 0:
        raise ValueError(f"{key!r} debe ser positivo, no {value}.")
    return value


def _test_start(cfg: dict[str, Any]) -> date:
    """Read ``test_start`` as a date, naming the missing key."""
    if "test_start" not in cfg:
        raise ValueError(
            "El protocolo no define 'test_start'; agrégalo a configs/backtest.yaml."
        )
    try:
        return date.fromisoformat(str(cfg["test_start"]))
    except ValueError as error:
        raise ValueError(
            f"test_start debe ser YYYY-MM-DD, no {cfg['test_start']!r}."
        ) from error


def _validation_days(split: WalkForwardSplit) -> set[date]:
    """Every day the fold scores, both ends included."""
    total = (split.validation_end - split.validation_start).days
    return {
        split.validation_start + timedelta(days=index) for index in range(total + 1)
    }


def _check_coverage(frame: pl.DataFrame, splits: tuple[WalkForwardSplit, ...]) -> None:
    """Fail before the sweep when a target station misses a validation day."""
    for station in TARGET_STATIONS:
        available = set(frame.filter(pl.col("unique_id") == station)["ds"].to_list())
        if not available:
            raise ValueError(f"El dataset no contiene la estación {station!r}.")
        for split in splits:
            missing = sorted(_validation_days(split) - available)
            if missing:
                raise ValueError(
                    f"{station}: faltan observaciones en la validación del corte "
                    f"{split.cutoff} ({missing[0]}..{missing[-1]})."
                )


def build_splits(
    frame: pl.DataFrame,
    cfg: dict[str, Any],
    *,
    horizon: int,
    station: str = DEFAULT_STATION,
) -> tuple[WalkForwardSplit, ...]:
    """Build the configured walk-forward folds without touching the test block.

    ``walk_forward_splits`` needs a single chronological series, so the dates
    come from ``station``; every target station is then checked for complete
    validation coverage. The last cutoff must match the protocol default used
    by the single-cutoff path, because a data gap would otherwise move the
    whole sweep to a different boundary without warning.
    """
    required = {"unique_id", "ds"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}.")
    dates = frame.filter(pl.col("unique_id") == station)["ds"].sort().unique().to_list()
    if not dates:
        raise ValueError(f"No hay fechas para la estación {station!r}.")

    splits = walk_forward_splits(
        dates,
        test_start=_test_start(cfg),
        horizon=horizon,
        step_size=_protocol_int(cfg, "step_size"),
        n_windows=_protocol_int(cfg, "n_windows"),
    )
    expected = default_cutoff(cfg, horizon)
    if splits[-1].cutoff != expected:
        raise ValueError(
            f"El último corte walk-forward ({splits[-1].cutoff}) no coincide con "
            f"el corte por defecto del protocolo ({expected}); revisa la "
            "cobertura de datos o configs/backtest.yaml."
        )
    folds = tuple(splits)
    _check_coverage(frame, folds)
    return folds


@dataclass(frozen=True)
class BacktestResult:
    """Predictions and per-fold statistics for one walk-forward sweep.

    ``predictions`` carries ``fold, model, unique_id, cutoff, ds, horizon_step,
    y_pred, y_true`` and ``statistics`` is the fold-tagged output of
    ``compute_statistics`` for the same folds.
    """

    predictions: pl.DataFrame
    statistics: pl.DataFrame
    splits: tuple[WalkForwardSplit, ...]
    model: str
    against: str
    horizon: int

    def __post_init__(self) -> None:
        validate_runner_name(self.model)
        validate_runner_name(self.against)
        if self.model == self.against:
            raise ValueError(
                "El modelo y el runner de comparación deben ser distintos."
            )
        validate_horizon(self.horizon)
        if not self.splits:
            raise ValueError("El backtest necesita al menos un pliegue.")
        self._validate_predictions()
        self._validate_statistics()

    def _expected_folds(self) -> list[int]:
        return list(range(1, len(self.splits) + 1))

    def _expected_cutoffs(self) -> dict[int, date]:
        return {index: split.cutoff for index, split in enumerate(self.splits, start=1)}

    def _cutoffs_by_fold(self, frame: pl.DataFrame, label: str) -> dict[int, date]:
        duplicated = frame.group_by("fold").agg(
            pl.col("cutoff").n_unique().alias("cutoffs")
        )
        if (duplicated["cutoffs"] > 1).any():
            raise ValueError(f"Cada pliegue debe usar un solo corte en {label}.")
        return dict(frame.select("fold", "cutoff").unique().iter_rows())

    def _validate_predictions(self) -> None:
        missing = [
            column
            for column in BACKTEST_PREDICTION_COLUMNS
            if column not in self.predictions.columns
        ]
        if missing:
            raise ValueError(f"Faltan columnas de predicciones: {missing}.")
        frame = self.predictions.select(*BACKTEST_PREDICTION_COLUMNS)
        if frame.is_empty():
            raise ValueError("Las predicciones del backtest no pueden estar vacías.")
        if sorted(frame["fold"].unique().to_list()) != self._expected_folds():
            raise ValueError(
                "Los fold de las predicciones no coinciden con los pliegues: "
                f"{sorted(frame['fold'].unique().to_list())}."
            )
        if self._cutoffs_by_fold(frame, "las predicciones") != self._expected_cutoffs():
            raise ValueError("Cada fold de las predicciones debe usar su cutoff.")
        mismatched = frame.filter(
            (pl.col("ds") - pl.col("cutoff")).dt.total_days() != pl.col("horizon_step")
        )
        if mismatched.height:
            raise ValueError(
                f"{mismatched.height} predicciones no cumplen ds = cutoff + paso."
            )
        for index, split in enumerate(self.splits, start=1):
            fold = frame.filter(pl.col("fold") == index)
            if fold["ds"].max() > split.validation_end:
                raise ValueError(
                    f"El pliegue {index} predice más allá de su validación "
                    f"({split.validation_end})."
                )
        object.__setattr__(
            self,
            "predictions",
            frame.sort("fold", "model", "unique_id", "horizon_step"),
        )

    def _validate_statistics(self) -> None:
        missing = [
            column
            for column in BACKTEST_STATISTICS_COLUMNS
            if column not in self.statistics.columns
        ]
        if missing:
            raise ValueError(f"Faltan columnas de estadísticas por pliegue: {missing}.")
        frame = self.statistics.select(*BACKTEST_STATISTICS_COLUMNS)
        if frame.is_empty():
            raise ValueError(
                "Las estadísticas por pliegue del backtest no pueden estar vacías."
            )
        folds = sorted(frame["fold"].unique().to_list())
        if folds != self._expected_folds():
            raise ValueError(
                f"Los fold de las estadísticas no coinciden con los pliegues: {folds}."
            )
        if self._cutoffs_by_fold(frame, "las estadísticas") != self._expected_cutoffs():
            raise ValueError("Cada fold de las estadísticas debe usar su cutoff.")
        models = set(frame["model"])
        if self.model not in models or self.against not in models:
            raise ValueError(
                "Las estadísticas deben incluir el modelo y su runner de comparación."
            )
        order = (
            pl.when(pl.col("horizon") == "all")
            .then(999)
            .otherwise(pl.col("horizon").cast(pl.Int32))
        )
        object.__setattr__(
            self,
            "statistics",
            frame.with_columns(_order=order)
            .sort("fold", "unique_id", "model", "_order")
            .drop("_order"),
        )


def run_backtest(
    model: str,
    against: str,
    frame: pl.DataFrame,
    splits: Sequence[WalkForwardSplit],
    *,
    horizon: int = DEFAULT_HORIZON,
    input_length: int = INPUT_LENGTH,
    seasonality: int = 365,
    runner_factory: Callable[[str], Runner] = get_runner,
) -> BacktestResult:
    """Fit and score ``model`` and ``against`` on every fold.

    Each fold calls ``run_runners`` with fresh runner instances, so both share
    the same cutoff, history, windows and validation segment. Nothing after the
    last fold's ``validation_end`` is read.
    """
    validate_horizon(horizon)
    validate_runner_name(model)
    validate_runner_name(against)
    if model == against:
        raise ValueError("El modelo y el runner de comparación deben ser distintos.")
    folds = tuple(splits)
    if not folds:
        raise ValueError("El backtest necesita al menos un pliegue.")
    if any(
        left.cutoff >= right.cutoff
        for left, right in zip(folds, folds[1:], strict=False)
    ):
        raise ValueError(
            "Los cortes de los pliegues deben ser estrictamente crecientes."
        )
    require_implemented(runner_factory(model), "modelo")
    require_implemented(runner_factory(against), "comparación")

    predictions: list[pl.DataFrame] = []
    statistics: list[pl.DataFrame] = []
    for index, split in enumerate(folds, start=1):
        result = run_runners(
            runner_factory(model),
            runner_factory(against),
            frame,
            cutoff=split.cutoff,
            horizon=horizon,
            input_length=input_length,
            seasonality=seasonality,
        )
        truth = observed_frame(frame, split.cutoff, horizon).select(
            "unique_id", "ds", pl.col("y").cast(pl.Float64).alias("y_true")
        )
        predictions.append(
            result.prediction.frame.join(truth, on=["unique_id", "ds"], how="left")
            .with_columns(fold=pl.lit(index, dtype=pl.Int32))
            .select(*BACKTEST_PREDICTION_COLUMNS)
        )
        statistics.append(
            result.statistics.frame.with_columns(
                fold=pl.lit(index, dtype=pl.Int32),
                cutoff=pl.lit(split.cutoff, dtype=pl.Date),
            ).select(*BACKTEST_STATISTICS_COLUMNS)
        )
    return BacktestResult(
        predictions=pl.concat(predictions),
        statistics=pl.concat(statistics),
        splits=folds,
        model=model,
        against=against,
        horizon=horizon,
    )
