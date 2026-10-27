"""Recompute statistics from exported predictions against the validation segment.

Usage: ``uv run python scripts/validate_predictions.py <model> [--against naive]
[--cutoff YYYY-MM-DD] [--predictions-dir PATH]``.

This script never trains a model: it reads
``data/predictions/<runner>/predictions.parquet`` for both runners, joins the
observed values for ``cutoff + 1 .. cutoff + horizon`` and writes
``data/predictions/<model>/statistics.parquet``.
"""

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from orinoco_forecast.evaluation.validation import validate_predictions
from orinoco_forecast.models.registry import available_runners


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"fecha inválida {value!r}; usa YYYY-MM-DD."
        ) from error


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
        help="Corte esperado; si se omite se lee del archivo de predicciones.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        help="Carpeta alternativa de predicciones (por defecto data/predictions).",
    )
    args = parser.parse_args()

    if args.model == args.against:
        parser.error("El modelo y el runner de comparación deben ser distintos.")

    try:
        statistics = validate_predictions(
            args.model,
            args.against,
            root=args.predictions_dir,
            cutoff=args.cutoff,
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
        print(f"{args.model} vs {args.against}: estadísticas de validación")
        print(statistics.frame)


if __name__ == "__main__":
    main()
