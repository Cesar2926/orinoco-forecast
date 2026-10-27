"""Validate a runner over many cutoffs (rolling origin) against a baseline.

Usage: ``uv run python scripts/validate_cv.py <model> [--against naive]
[--windows 24] [--step 30]``.

Reuses ``run_runners`` (one fit per cutoff, no leakage) on the cleaned history
and averages the per-cutoff statistics. Reports days 7 and 14 and all steps.
The final test block (``test_start`` onward) is never evaluated.
"""

import argparse
from datetime import date

import polars as pl

from orinoco_forecast.data.splits import walk_forward_splits
from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.evaluation.execution import (
    TARGET_STATIONS,
    load_history,
    run_runners,
)
from orinoco_forecast.models.registry import available_runners, get_runner

REPORT = ("7", "14", "all")


def validate_cv(
    model: str, against: str, *, windows: int, step: int, horizon: int
) -> pl.DataFrame:
    """Mean MAE/RMSE per station, model and report horizon over all cutoffs."""
    cfg = load_backtest_config()
    frame = load_history()
    dates = frame.filter(pl.col("unique_id") == TARGET_STATIONS[0])["ds"].to_list()
    splits = walk_forward_splits(
        dates,
        test_start=date.fromisoformat(str(cfg["test_start"])),
        horizon=horizon,
        step_size=step,
        n_windows=windows,
    )
    stats = []
    for i, split in enumerate(splits, 1):
        print(f"corte {i}/{len(splits)}: {split.cutoff}", flush=True)
        result = run_runners(
            get_runner(model),
            get_runner(against),
            frame,
            cutoff=split.cutoff,
            horizon=horizon,
            seasonality=int(cfg["seasonality"]),
        )
        stats.append(result.statistics.frame.with_columns(cutoff=pl.lit(split.cutoff)))
    table = (
        pl.concat(stats)
        .with_columns(pl.col("horizon").cast(pl.String))
        .filter(pl.col("horizon").is_in(REPORT))
        .group_by("unique_id", "model", "horizon")
        .agg(pl.col("mae").mean(), pl.col("rmse").mean(), pl.col("nse").mean())
    )
    base = table.filter(pl.col("model") == against).select(
        "unique_id", "horizon", base_mae="mae"
    )
    return (
        table.join(base, on=["unique_id", "horizon"])
        .with_columns(rel_mae=pl.col("mae") / pl.col("base_mae"))
        .drop("base_mae")
        .sort("unique_id", "horizon", "model")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=available_runners())
    parser.add_argument("--against", default="naive", choices=available_runners())
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()
    table = validate_cv(
        args.model,
        args.against,
        windows=args.windows,
        step=args.step,
        horizon=args.horizon,
    )
    with pl.Config(
        tbl_rows=-1,
        tbl_hide_dataframe_shape=True,
        tbl_formatting="ASCII_FULL_CONDENSED",
        float_precision=4,
    ):
        print(table)


if __name__ == "__main__":
    main()
