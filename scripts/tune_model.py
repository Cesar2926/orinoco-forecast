"""Run or resume an Optuna study for one model on the tuning windows.

Usage: ``uv run python scripts/tune_model.py xgboost|tcn [--trials 20] [--stage A]``.
Tuning windows end at the first validation cutoff (``tuning`` in
configs/backtest.yaml). The study lives in ``artifacts/optuna/<model>.db``
(resumable); trials are written to ``reports/tuning/<model>_trials.csv``.
The default 20 trials is the same budget for every model: with unequal budgets
the comparison would measure tuning effort, not the model.
"""

import argparse
import json

import polars as pl

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.evaluation.tuning import SPACES, run_study, trials_table
from orinoco_forecast.features.dataset import build_model_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(SPACES))
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Pruebas completas (mismo presupuesto para todos los modelos).",
    )
    parser.add_argument("--stage", choices=["A", "B"], default="A")
    args = parser.parse_args()

    root = repo_root()
    db_dir = root / "artifacts" / "optuna"
    db_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(db_dir / f'{args.model}.db').as_posix()}"
    study = run_study(
        args.model,
        build_model_frame(args.stage),
        load_backtest_config(),
        args.trials,
        storage,
        study_name=f"{args.model}_stage{args.stage}",
    )

    table = trials_table(study)
    out_dir = root / "reports" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_dir / f"{args.model}_trials.csv")

    best = study.best_trial
    counts = table.group_by("state").len().sort("state").iter_rows()
    print(f"{args.model} (etapa {args.stage}): {dict(counts)}")
    print(f"Mejor prueba #{best.number}: rel_mae medio = {best.value:.4f}")
    print("Parámetros:", json.dumps(best.params, indent=2))
    print("rel_mae por estación:", best.user_attrs["rel_mae"])
    print(f"MAE medio: {best.user_attrs['mae_mean']:.4f}")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_FULL_CONDENSED"):
        print(table.select("number", "state", "value", "duration_s").sort("value"))


if __name__ == "__main__":
    main()
