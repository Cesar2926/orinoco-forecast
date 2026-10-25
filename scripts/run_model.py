"""Run the shared cross-validation for one registered model.

Usage: ``uv run python scripts/run_model.py baselines [--stage A|B] [--final-test]``.
Writes data/predictions/<model>/cv.parquet and metrics.parquet.
"""

import argparse

import polars as pl

from orinoco_forecast.evaluation.backtest import load_backtest_config, run_cv
from orinoco_forecast.features.dataset import build_model_frame
from orinoco_forecast.models.registry import RUNNERS, get_runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(RUNNERS))
    parser.add_argument("--stage", choices=["A", "B"], default="A")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evalúa el bloque de test final (usar una sola vez).",
    )
    args = parser.parse_args()
    if args.final_test:
        print(
            "ADVERTENCIA: se evaluará el bloque de test final. Úselo una sola vez, "
            "con el modelo ya elegido; mirarlo para decidir es fuga de datos."
        )

    cfg = load_backtest_config()
    runner = get_runner(args.model)
    frame = build_model_frame(args.stage)
    cv, metrics = run_cv(args.model, runner, frame, cfg, final_test=args.final_test)

    summary = metrics.filter(pl.col("horizon") == "all").drop("horizon")
    # ASCII borders: Windows consoles default to cp1252 and cannot encode box chars.
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
        tbl_formatting="ASCII_FULL_CONDENSED",
        float_precision=4,
    ):
        print(
            f"{args.model} (etapa {args.stage}): {cv['cutoff'].n_unique()} cortes, "
            f"{cv.height} predicciones. Métricas h=1..{cfg['horizon']}:"
        )
        print(summary)


if __name__ == "__main__":
    main()
