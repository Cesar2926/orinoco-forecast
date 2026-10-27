"""Run one model and compare it with a baseline runner on a single cutoff.

Usage: ``uv run python scripts/run_model.py <model> [--against <runner>]
[--cutoff YYYY-MM-DD] [--horizon N] [--write-dataset]``.

The runner only fits and predicts. Predictions and statistics are written to
``data/predictions/<runner>/`` and a separate script can revalidate them. The
default comparison runner is ``naive`` and the default cutoff is the last one
whose 14-day validation ends before ``test_start``.
"""

import argparse
from datetime import date

import polars as pl

from orinoco_forecast.data.datasets import TimeSeriesDataset
from orinoco_forecast.evaluation.execution import execute
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=available_runners())
    parser.add_argument(
        "--against",
        default="naive",
        choices=available_runners(),
        help="Runner de comparación (baseline). Por defecto: naive.",
    )
    parser.add_argument(
        "--cutoff",
        type=_date,
        help="Último día de entrenamiento (YYYY-MM-DD).",
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
        default=28,
        help="Días de contexto de cada ventana supervisada (por defecto 28).",
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
        result = execute(
            args.model,
            args.against,
            cutoff=args.cutoff,
            horizon=args.horizon,
            input_length=args.input_length,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
        tbl_width_chars=200,
        tbl_formatting="ASCII_FULL_CONDENSED",
        float_precision=4,
    ):
        print(
            f"{args.model} vs {args.against}: corte {result.cutoff}, "
            f"horizonte {result.horizon}, "
            f"{result.prediction.frame.height} predicciones."
        )
        print(result.statistics.frame)


if __name__ == "__main__":
    main()
