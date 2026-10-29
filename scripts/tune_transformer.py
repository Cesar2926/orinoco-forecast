"""Run or resume the Optuna search of the Transformer on windows before validation.

Usage: ``uv run python scripts/tune_transformer.py [--trials 20]``.

Same protocol, budget and objective as ``scripts/tune_cnn_lstm.py`` (12 tuning
cutoffs every 23 days before the first validation cutoff, one fit at the first
tuning cutoff, mean ``rel_mae`` against Naive over the two target stations and
the seeds of ``configs/models/transformer.yaml``); only the search space
changes. The study lives in ``artifacts/optuna/transformer_2009.db``
(resumable); trials go to ``reports/tuning/transformer_trials.csv``.
"""

import argparse
import json
from typing import Any

import optuna
import polars as pl
from tune_cnn_lstm import run_study, trials_table

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.models.neural.transformer import (
    TransformerRunner,
    load_transformer_config,
)


def suggest(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    """Search space; epochs and patience stay fixed (30 / 5) as in the CNN-LSTM.

    Every ``d_model`` choice is a multiple of every ``nhead`` choice, so no trial
    is pruned for an invalid head split.
    """
    return {
        **base,
        "input_size": trial.suggest_categorical("input_size", [28, 56, 112]),
        "d_model": trial.suggest_categorical("d_model", [16, 32, 64]),
        "nhead": trial.suggest_categorical("nhead", [2, 4]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "max_epochs": 30,
        "early_stop_patience": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    root = repo_root()
    db_dir = root / "artifacts" / "optuna"
    db_dir.mkdir(parents=True, exist_ok=True)
    study = run_study(
        args.trials,
        f"sqlite:///{(db_dir / 'transformer_2009.db').as_posix()}",
        suggest_fn=suggest,
        base=load_transformer_config(),
        runner_cls=TransformerRunner,
        study_name="transformer_2009",
    )
    table = trials_table(study)
    out = root / "reports" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "transformer_trials.csv")
    best = study.best_trial
    states = dict(table.group_by("state").len().sort("state").iter_rows())
    print(f"transformer: {states}")
    print(f"Mejor prueba #{best.number}: rel_mae medio = {best.value:.4f}")
    print("Parámetros:", json.dumps(best.params, indent=2))
    print("rel_mae por estación:", best.user_attrs["rel_mae"])
    print(f"MAE medio: {best.user_attrs['mae_mean']:.4f}")
    with pl.Config(tbl_rows=-1, tbl_formatting="ASCII_FULL_CONDENSED"):
        print(table.select("number", "state", "value", "duration_s").sort("value"))


if __name__ == "__main__":
    main()
