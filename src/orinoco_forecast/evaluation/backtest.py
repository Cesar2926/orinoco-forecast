"""Shared execution protocol (``configs/backtest.yaml``) and walk-forward sweeps.

The protocol values used by the runner execution path live here: horizon, input
length, test start, seasonality and the walk-forward folds. ``run_backtest``
reuses ``run_runners`` for every fold, so the model and its comparison runner
are always fitted and scored on exactly the same cutoff, windows and validation
segment.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.splits import WalkForwardSplit, walk_forward_splits
from orinoco_forecast.evaluation.execution import TARGET_STATIONS, default_cutoff

DEFAULT_STATION = "ciudad_bolivar"
PROTOCOL_KEYS = ("test_start", "n_windows", "step_size")


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
