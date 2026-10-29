"""Score the average of the validated runners on the walk-forward folds.

Usage: ``uv run python scripts/make_ensemble.py [--source reports/predictions]``.

Reads each member's ``backtest/naive/fold_predictions.parquet`` from ``--source``
(the versioned results by default), averages them with
``ensemble.average_forecasts`` and scores the average against Naive with the same
``run_backtest`` / ``run_runners`` pipeline as any model. Nothing is retrained.
Writes ``data/predictions/ensemble/`` (single cutoff and ``backtest/naive/``).
"""

import argparse
from pathlib import Path

import polars as pl

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.ensemble.averaging import (
    ENSEMBLE_MEMBERS,
    ReplayRunner,
    average_forecasts,
)
from orinoco_forecast.evaluation.backtest import (
    build_splits,
    load_backtest_config,
    run_backtest,
    summarize_comparison,
)
from orinoco_forecast.evaluation.execution import (
    load_history,
    resolve_cutoff,
    run_runners,
)
from orinoco_forecast.models.registry import get_runner

NAME = "ensemble"
FOLDS = Path("backtest") / "naive" / "fold_predictions.parquet"


def load_member_folds(source: Path, members: tuple[str, ...]) -> pl.DataFrame:
    """Fold predictions of every member, tagged with its runner name."""
    missing = [m for m in members if not (source / m / FOLDS).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Faltan los 24 cortes de {missing} en {source}; "
            "ejecuta run_model.py <modelo> --folds."
        )
    return pl.concat(
        pl.read_parquet(source / m / FOLDS).with_columns(runner=pl.lit(m))
        for m in members
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        type=Path,
        default=repo_root() / "reports" / "predictions",
        help="Carpeta con los resultados de cada miembro (por defecto reports/).",
    )
    args = parser.parse_args(argv)

    averaged = average_forecasts(load_member_folds(args.source, ENSEMBLE_MEMBERS))
    cfg = load_backtest_config()
    horizon = int(cfg["horizon"])
    frame = load_history()

    def factory(name: str):
        return ReplayRunner(NAME, averaged) if name == NAME else get_runner(name)

    backtest = run_backtest(
        NAME,
        "naive",
        frame,
        build_splits(frame, cfg, horizon=horizon),
        horizon=horizon,
        seasonality=int(cfg["seasonality"]),
        runner_factory=factory,
    )
    backtest.write()
    cutoff = resolve_cutoff(frame, cfg, None, horizon)
    single = run_runners(
        factory(NAME),
        get_runner("naive"),
        frame,
        cutoff=cutoff,
        horizon=horizon,
        seasonality=int(cfg["seasonality"]),
    )
    single.write()

    summary = summarize_comparison(backtest.statistics).filter(
        (pl.col("horizon") == "all") & (pl.col("metric") == "mae")
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(
            f"Ensamble de {', '.join(ENSEMBLE_MEMBERS)}: {len(backtest.splits)} cortes"
        )
        print(
            summary.select(
                "unique_id",
                "mean_model",
                "std_model",
                "mean_baseline",
                "win_rate",
                "sign_test_p",
            )
        )


if __name__ == "__main__":
    main()
