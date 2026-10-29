"""Zero-shot search of TimesFM 3 on the tuning windows before validation.

Usage: ``uv run --extra timesfm python scripts/tune_timesfm3.py``.

Same cutoffs and objective as ``scripts/tune_cnn_lstm.py`` (12 tuning cutoffs
every 23 days before the first validation cutoff; mean over the two target
stations of ``rel_mae`` = model MAE / Naive MAE). TimesFM is not trained, so the
space is a finite grid, ``context_len`` x ``use_covariates`` = 6 configurations,
evaluated exhaustively instead of sampling 20 Optuna trials: the space is
exhausted, not under-budgeted. Trials go to ``reports/tuning/timesfm3_trials.csv``
in the same columns as the other searches.
"""

import itertools
import time
from datetime import date
from typing import Any

import polars as pl
from tune_cnn_lstm import score, tuning_cutoffs

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.splits import walk_forward_splits
from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.evaluation.execution import (
    TARGET_STATIONS,
    load_history,
    observed_frame,
)
from orinoco_forecast.features.dataset import assert_no_imputation_near_cutoffs
from orinoco_forecast.models.pretrained.timesfm import (
    TimesFM3Runner,
    load_timesfm_config,
)
from orinoco_forecast.models.registry import get_runner

GRID = {"context_len": [512, 1024, 2048], "use_covariates": [False, True]}


def grid() -> list[dict[str, Any]]:
    """Every ``GRID`` combination, in a fixed order."""
    keys = list(GRID)
    return [dict(zip(keys, v, strict=True)) for v in itertools.product(*GRID.values())]


def evaluate(
    params: dict[str, Any], frame: pl.DataFrame, cutoffs: list[date], horizon: int
) -> tuple[float, dict[str, float], float]:
    """Predict every cutoff with ``params`` and score against Naive."""
    # Both runners ignore the supervised windows; validate_windows needs the shape.
    windows = pl.DataFrame(
        {"unique_id": ["x"], "origin": [cutoffs[0]]}
        | {f"target_{s}": [0.0] for s in range(1, horizon + 1)}
    )
    model = TimesFM3Runner(params).fit(
        windows, frame, cutoff=cutoffs[0], horizon=horizon
    )
    # NaiveRunner.predict ignores ``cutoff`` and repeats the level at its fit
    # cutoff, so it is refitted at every cutoff (one fit for all would score
    # every window against the first cutoff's persistence).
    naive = [
        get_runner("naive")
        .fit(windows, frame, cutoff=c, horizon=horizon)
        .predict(frame, cutoff=c, horizon=horizon)
        .frame
        for c in cutoffs
    ]
    results = {
        "model": pl.concat(
            model.predict(frame, cutoff=c, horizon=horizon).frame for c in cutoffs
        ),
        "naive": pl.concat(naive),
    }
    results = {
        k: v.filter(pl.col("unique_id").is_in(TARGET_STATIONS))
        for k, v in results.items()
    }
    observed = pl.concat(observed_frame(frame, c, horizon) for c in cutoffs)
    return score(results["model"], results["naive"], observed)


def main() -> None:
    cfg = load_backtest_config()
    horizon = int(cfg["horizon"])
    frame = load_history()
    dates = frame.filter(pl.col("unique_id") == TARGET_STATIONS[0])["ds"].to_list()
    first_validation = walk_forward_splits(
        dates,
        test_start=date.fromisoformat(str(cfg["test_start"])),
        horizon=horizon,
        step_size=int(cfg["step_size"]),
        n_windows=int(cfg["n_windows"]),
    )[0].cutoff
    cutoffs = tuning_cutoffs(first_validation, horizon)
    assert_no_imputation_near_cutoffs(frame, cutoffs, horizon)
    print(f"Cortes de ajuste: {cutoffs[0]} .. {cutoffs[-1]} ({len(cutoffs)})")
    base = load_timesfm_config()
    rows = []
    for number, combo in enumerate(grid()):
        start = time.perf_counter()
        value, stations, mae = evaluate({**base, **combo}, frame, cutoffs, horizon)
        row: dict[str, Any] = {
            "number": number,
            "state": "COMPLETE",
            "value": value,
            "duration_s": time.perf_counter() - start,
            "mae_mean": mae,
        }
        row |= {f"rel_mae_{k}": v for k, v in stations.items()}
        row |= {f"params_{k}": v for k, v in combo.items()}
        rows.append(row)
        print(f"#{number} {combo}: rel_mae medio = {value:.4f}", flush=True)
    table = pl.DataFrame(rows)
    out = repo_root() / "reports" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "timesfm3_trials.csv")
    best = table.sort("value").row(0, named=True)
    print(f"Mejor #{best['number']}: rel_mae medio = {best['value']:.4f}")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_FULL_CONDENSED"):
        print(table.sort("value"))


if __name__ == "__main__":
    main()
