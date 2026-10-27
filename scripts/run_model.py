"""Run one model and compare it with a baseline runner.

Usage: ``uv run python scripts/run_model.py <model> [--against <runner>]
[--cutoff YYYY-MM-DD | --folds] [--horizon N] [--write-dataset]``.

The runner only fits and predicts. Predictions and statistics are written to
``data/predictions/<runner>/`` and a separate script can revalidate them. The
default comparison runner is ``naive`` and the default cutoff is the last one
whose validation ends before ``test_start``. ``--folds`` runs every configured
walk-forward fold and writes the comparison under
``data/predictions/<runner>/backtest/<against>/``.
"""

import argparse
from datetime import date

import polars as pl

from orinoco_forecast.data.datasets import TimeSeriesDataset
from orinoco_forecast.evaluation.backtest import (
    BacktestResult,
    execute_backtest,
    summarize_comparison,
)
from orinoco_forecast.evaluation.execution import ExecutionResult, execute
from orinoco_forecast.models.registry import available_runners, get_runner


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"fecha inválida {value!r}; usa YYYY-MM-DD."
        ) from error


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"entero inválido: {value!r}.") from error
    if not 1 <= number <= 14:
        raise argparse.ArgumentTypeError("el horizonte debe estar entre 1 y 14.")
    return number


def _table_config() -> pl.Config:
    return pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
        tbl_width_chars=200,
        tbl_formatting="ASCII_FULL_CONDENSED",
        float_precision=4,
    )


def _report_single(result: ExecutionResult) -> None:
    with _table_config():
        print(
            f"{result.prediction.runner} vs {result.baseline.runner}: corte "
            f"{result.cutoff}, horizonte {result.horizon}, "
            f"{result.prediction.frame.height} predicciones."
        )
        print(result.statistics.frame)


def _report_backtest(result: BacktestResult) -> None:
    comparison = summarize_comparison(result.statistics).filter(
        pl.col("horizon") == "all"
    )
    with _table_config():
        print(
            f"{result.model} vs {result.against}: {len(result.splits)} pliegues "
            f"walk-forward, horizonte {result.horizon}."
        )
        print(
            f"cortes: {result.splits[0].cutoff} .. {result.splits[-1].cutoff} "
            f"(validación hasta {result.splits[-1].validation_end})."
        )
        print(comparison)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=available_runners())
    parser.add_argument(
        "--against",
        default="naive",
        choices=available_runners(),
        help="Runner de comparación (baseline). Por defecto: naive.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cutoff",
        type=_date,
        help="Último día de entrenamiento (YYYY-MM-DD).",
    )
    selection.add_argument(
        "--folds",
        action="store_true",
        help="Ejecuta todos los pliegues walk-forward de configs/backtest.yaml.",
    )
    parser.add_argument(
        "--horizon",
        type=_positive_int,
        default=14,
        help="Días a pronosticar, 1..14 (por defecto 14).",
    )
    parser.add_argument(
        "--input-length",
        type=int,
        default=None,
        help="Días de contexto de cada ventana (por defecto configs/backtest.yaml).",
    )
    parser.add_argument(
        "--write-dataset",
        action="store_true",
        help="Regenera el Parquet del dataset antes de ejecutar.",
    )
    args = parser.parse_args()

    if args.model == args.against:
        parser.error("El modelo y el runner de comparación deben ser distintos.")
    for name in (args.model, args.against):
        runner = get_runner(name)
        if not getattr(runner, "implemented", True):
            parser.error(
                f"El runner {name!r} solo tiene estructura; "
                "todavía no se puede ejecutar."
            )

    try:
        if args.write_dataset:
            TimeSeriesDataset().write_parquet()
        if args.folds:
            backtest = execute_backtest(
                args.model,
                args.against,
                horizon=args.horizon,
                input_length=args.input_length,
            )
        else:
            single = execute(
                args.model,
                args.against,
                cutoff=args.cutoff,
                horizon=args.horizon,
                input_length=args.input_length,
            )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if args.folds:
        _report_backtest(backtest)
    else:
        _report_single(single)


if __name__ == "__main__":
    main()
