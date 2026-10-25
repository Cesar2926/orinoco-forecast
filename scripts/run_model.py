"""Run the shared cross-validation for one registered model.

Usage: ``uv run python scripts/run_model.py baselines [--stage A|B] [--final-test]``.
Writes data/predictions/<model>/cv.parquet and metrics.parquet, plus
feature_importance.parquet for runners that expose ``feature_importance``.
Prints ``rel_mae`` (MAE / Naive MAE) when the baselines were run first.
"""

import argparse

import polars as pl

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.evaluation.backtest import load_backtest_config, run_cv
from orinoco_forecast.evaluation.metrics import relative_to_baseline
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

    root = repo_root() / "data" / "predictions"
    out = root / args.model
    importance = None
    # Fit on the development rows only; never on the final test run.
    if hasattr(runner, "feature_importance") and not args.final_test:
        importance = runner.feature_importance(frame, cfg)
        importance.write_parquet(out / "feature_importance.parquet")

    note = None
    if args.model != "baselines":
        sub = "final_test" if args.final_test else ""
        base_file = root / "baselines" / sub / "metrics.parquet"
        if base_file.exists():
            metrics = relative_to_baseline(metrics, pl.read_parquet(base_file))
        else:
            note = (
                "Nota: sin data/predictions/baselines; ejecute primero "
                "`run_model.py baselines` para ver rel_mae (MAE / MAE de Naive)."
            )
    summary = metrics.filter(pl.col("horizon") == "all").drop("horizon")
    # ASCII borders: Windows consoles default to cp1252 and cannot encode box chars.
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
        tbl_width_chars=200,
        tbl_formatting="ASCII_FULL_CONDENSED",
        float_precision=4,
    ):
        print(
            f"{args.model} (etapa {args.stage}): {cv['cutoff'].n_unique()} cortes, "
            f"{cv.height} predicciones. Métricas h=1..{cfg['horizon']}:"
        )
        print(summary)
        if note:
            print(note)
        if importance is not None:
            print("Importancia por ganancia (top 10):")
            print(importance.head(10))


if __name__ == "__main__":
    main()
