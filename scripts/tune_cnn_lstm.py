"""Run or resume the Optuna search of the CNN-LSTM on windows before validation.

Usage: ``uv run python scripts/tune_cnn_lstm.py [--trials 20]``.

Protocol (same as the XGBoost and TCN searches, so the budget is equal):
12 tuning cutoffs every 23 days that end one horizon before the first validation
cutoff (the 24 validation folds and the final test are never seen); the model is
fitted once with ``ds <= first tuning cutoff`` (no refit per cutoff) and predicts
every cutoff. Objective: mean over the two target stations of ``rel_mae`` = model
MAE / Naive MAE on the same windows, all horizon steps, averaged over the seeds
of ``configs/models/cnn_lstm.yaml``. Pruned or failed trials do not count towards
the 20 COMPLETE trials. The study lives in ``artifacts/optuna/cnn_lstm_2009.db``
(resumable); trials go to ``reports/tuning/cnn_lstm_trials.csv``.
"""

import argparse
import json
from datetime import date, timedelta
from typing import Any

import optuna
import polars as pl

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.splits import walk_forward_splits
from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.evaluation.execution import (
    TARGET_STATIONS,
    load_history,
    observed_frame,
)
from orinoco_forecast.features.dataset import assert_no_imputation_near_cutoffs
from orinoco_forecast.models.neural.cnn_lstm import (
    CnnLstmRunner,
    load_cnn_lstm_config,
)
from orinoco_forecast.models.registry import get_runner

SAMPLER_SEED = 42
TUNING_WINDOWS = 12
TUNING_STEP = 23


def tuning_cutoffs(first_validation_cutoff: date, horizon: int) -> list[date]:
    """12 cutoffs every 23 days; the last observes up to the first validation cutoff."""
    last = first_validation_cutoff - timedelta(days=horizon)
    return [
        last - timedelta(days=TUNING_STEP * i) for i in reversed(range(TUNING_WINDOWS))
    ]


def suggest(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    """Search space; epochs and patience stay fixed (30 / 5).

    Trimmed after trial 17: a 128-unit, 2-layer LSTM on a 112-day window ran over
    30 min on CPU (one 128-unit layer took 6.5 min), so that combination is pruned.
    """
    params = {
        **base,
        "input_size": trial.suggest_categorical("input_size", [28, 56, 112]),
        "conv_channels": trial.suggest_categorical("conv_channels", [16, 32, 64]),
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
        "conv_layers": trial.suggest_int("conv_layers", 1, 3),
        "lstm_hidden_size": trial.suggest_categorical(
            "lstm_hidden_size", [32, 64, 128]
        ),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
        "max_epochs": 30,
        "early_stop_patience": 5,
    }
    if params["lstm_hidden_size"] == 128 and params["lstm_layers"] == 2:
        raise optuna.TrialPruned(
            "LSTM de 128 unidades y 2 capas: demasiado lento en CPU."
        )
    return params


def score(
    predictions: pl.DataFrame, naive: pl.DataFrame, observed: pl.DataFrame
) -> tuple[float, dict[str, float], float]:
    """``(objective, rel_mae per station, mean MAE)``; seeds are averaged."""

    def mae(frame: pl.DataFrame) -> pl.DataFrame:
        return (
            frame.join(observed, on=["unique_id", "ds"])
            .group_by("model", "unique_id")
            .agg((pl.col("y_pred") - pl.col("y")).abs().mean().alias("mae"))
        )

    base = mae(naive).select("unique_id", base_mae="mae")
    rel = mae(predictions).join(base, on="unique_id")
    per_station = (
        rel.with_columns(rel_mae=pl.col("mae") / pl.col("base_mae"))
        .group_by("unique_id")
        .agg(pl.col("rel_mae").mean())
        .sort("unique_id")
    )
    stations = dict(per_station.iter_rows())
    return sum(stations.values()) / len(stations), stations, float(rel["mae"].mean())


def evaluate(
    params: dict[str, Any],
    frame: pl.DataFrame,
    cutoffs: list[date],
    horizon: int,
    runner_cls: type[CnnLstmRunner] = CnnLstmRunner,
) -> tuple[float, dict[str, float], float]:
    """Fit once at ``cutoffs[0]``, predict every cutoff, score against Naive."""
    # Both runners ignore the supervised windows; validate_windows needs the shape.
    windows = pl.DataFrame(
        {"unique_id": ["x"], "origin": [cutoffs[0]]}
        | {f"target_{s}": [0.0] for s in range(1, horizon + 1)}
    )
    results = {}
    for name, runner in (
        ("model", runner_cls(params)),
        ("naive", get_runner("naive")),
    ):
        runner.fit(windows, frame, cutoff=cutoffs[0], horizon=horizon)
        results[name] = pl.concat(
            runner.predict(frame, cutoff=c, horizon=horizon).frame for c in cutoffs
        ).filter(pl.col("unique_id").is_in(TARGET_STATIONS))
    observed = pl.concat(observed_frame(frame, c, horizon) for c in cutoffs)
    return score(results["model"], results["naive"], observed)


def run_study(
    n_trials: int,
    storage: str,
    *,
    suggest_fn=suggest,
    base: dict[str, Any] | None = None,
    runner_cls: type[CnnLstmRunner] = CnnLstmRunner,
    study_name: str = "cnn_lstm_2009",
) -> optuna.Study:
    """Create or resume the study until it has ``n_trials`` COMPLETE trials.

    The keyword arguments let other runners built on ``CnnLstmRunner`` (the
    Transformer) reuse the same cutoffs, budget and objective.
    """
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
    print(
        f"Cortes de ajuste: {cutoffs[0]} .. {cutoffs[-1]} ({len(cutoffs)})", flush=True
    )
    base = load_cnn_lstm_config() if base is None else base

    def objective(trial: optuna.Trial) -> float:
        try:
            value, stations, mae = evaluate(
                suggest_fn(trial, base), frame, cutoffs, horizon, runner_cls
            )
        except ValueError as error:  # e.g. not enough windows for this input_size
            raise optuna.TrialPruned(str(error)) from error
        trial.set_user_attr("rel_mae", stations)
        trial.set_user_attr("mae_mean", mae)
        return value

    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        direction="minimize",
        load_if_exists=True,
    )
    # Offset the seed by the stored trials: re-seeding a resumed study replays draws.
    study.sampler = optuna.samplers.TPESampler(seed=SAMPLER_SEED + len(study.trials))
    complete = optuna.trial.TrialState.COMPLETE
    done = len(study.get_trials(deepcopy=False, states=(complete,)))
    if done < n_trials:
        study.optimize(
            objective,
            n_trials=10 * n_trials,  # cap attempts so a pruning-only space ends
            callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=(complete,))],
        )
    return study


def trials_table(study: optuna.Study) -> pl.DataFrame:
    """One row per trial: number, state, value, duration, params and rel_mae."""
    rows = []
    for t in study.trials:
        row: dict[str, Any] = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "duration_s": t.duration.total_seconds() if t.duration else None,
            "mae_mean": t.user_attrs.get("mae_mean"),
        }
        row |= {f"rel_mae_{k}": v for k, v in t.user_attrs.get("rel_mae", {}).items()}
        row |= {f"params_{k}": v for k, v in t.params.items()}
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    root = repo_root()
    db_dir = root / "artifacts" / "optuna"
    db_dir.mkdir(parents=True, exist_ok=True)
    study = run_study(
        args.trials, f"sqlite:///{(db_dir / 'cnn_lstm_2009.db').as_posix()}"
    )
    table = trials_table(study)
    out = root / "reports" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    table.write_csv(out / "cnn_lstm_trials.csv")
    best = study.best_trial
    print(f"cnn_lstm: {dict(table.group_by('state').len().sort('state').iter_rows())}")
    print(f"Mejor prueba #{best.number}: rel_mae medio = {best.value:.4f}")
    print("Parámetros:", json.dumps(best.params, indent=2))
    print("rel_mae por estación:", best.user_attrs["rel_mae"])
    print(f"MAE medio: {best.user_attrs['mae_mean']:.4f}")


if __name__ == "__main__":
    main()
