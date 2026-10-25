"""Optuna hyperparameter search on tuning windows placed before validation.

- Windows: ``tuning`` block of ``configs/backtest.yaml`` (``tuning_frame`` /
  ``tuning_config``); they end at the first validation cutoff, so the 24
  validation windows that compare models and the final test are never seen.
- Objective: mean over target stations of ``rel_mae`` (model MAE / Naive MAE on
  the same tuning windows, all horizons). Lower is better; < 1 beats persistence.
- Budget: the same number of COMPLETE trials for every model (pruned trials are
  invalid configurations and do not count), so the comparison measures the
  model, not the tuning effort.
- Trials are not written to ``data/predictions``; studies persist in SQLite so
  an interrupted run resumes.
"""

from collections.abc import Callable
from typing import Any

import optuna
import polars as pl

from orinoco_forecast.evaluation.backtest import (
    predict_cv,
    tuning_config,
    tuning_frame,
)
from orinoco_forecast.evaluation.metrics import evaluate_cv, relative_to_baseline
from orinoco_forecast.models.base import ForecastRunner

Suggest = Callable[[optuna.Trial, dict[str, Any]], dict[str, Any]]

SAMPLER_SEED = 42
COVARIATE_LAGS = {
    "14-21-28-56": [14, 21, 28, 56],
    "14-28-56-90": [14, 28, 56, 90],
    "14-21-28-56-90": [14, 21, 28, 56, 90],
}
DILATIONS = {"1-16": [1, 2, 4, 8, 16], "1-32": [1, 2, 4, 8, 16, 32]}
# One seed while tuning bounds the cost; final runs use every seed in cnn.yaml.
TUNING_SEEDS = [1]


def suggest_xgboost(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    """Espacio de XGBoost; rezagos del target, medias móviles y diferencias fijos."""
    xgb = {
        **base["xgb_params"],
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
    }
    lags = trial.suggest_categorical("covariate_lags", list(COVARIATE_LAGS))
    return {**base, "xgb_params": xgb, "covariate_lags": COVARIATE_LAGS[lags]}


def suggest_tcn(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    """Espacio del TCN (una semilla). Poda si el campo receptivo > ``input_size``."""
    from orinoco_forecast.models.neural.cnn import receptive_field

    params = {
        **base,
        "target_transform": trial.suggest_categorical(
            "target_transform", ["level", "difference"]
        ),
        "input_size": trial.suggest_categorical("input_size", [28, 56, 112]),
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3]),
        "dilations": DILATIONS[trial.suggest_categorical("dilations", list(DILATIONS))],
        "encoder_hidden_size": trial.suggest_categorical(
            "encoder_hidden_size", [32, 64, 128]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        # {500, 1000}, not {1000, 2000}: one CPU trial at 1000 steps with the
        # largest net took ~9 min, so 2000 steps projected 20 trials over 2 h.
        "max_steps": trial.suggest_categorical("max_steps", [500, 1000]),
        "seeds": TUNING_SEEDS,
    }
    field = receptive_field(params["kernel_size"], params["dilations"])
    if field > params["input_size"]:
        raise optuna.TrialPruned(
            f"Campo receptivo {field} > input_size {params['input_size']}."
        )
    return params


SPACES: dict[str, Suggest] = {"xgboost": suggest_xgboost, "tcn": suggest_tcn}


def _base_params(model: str) -> dict[str, Any]:
    if model == "xgboost":
        from orinoco_forecast.models.trees.tree import load_tree_config

        return load_tree_config()
    from orinoco_forecast.models.neural.cnn import load_cnn_config

    return load_cnn_config()


def _runner(model: str, params: dict[str, Any]) -> ForecastRunner:
    if model == "xgboost":
        from orinoco_forecast.models.trees.tree import XGBoostRunner

        return XGBoostRunner(params)
    from orinoco_forecast.models.neural.cnn import TCNRunner

    return TCNRunner(params)


def score_trial(
    metrics: pl.DataFrame, naive: pl.DataFrame
) -> tuple[float, dict[str, float], float]:
    """``(objetivo, rel_mae por estación, MAE medio)`` sobre ``horizon == "all"``.

    El objetivo es la media de ``rel_mae`` entre estaciones (y columnas del
    modelo, si hay varias); ambos marcos vienen de ``evaluate_cv`` en las
    mismas ventanas.
    """
    rel = relative_to_baseline(metrics.filter(pl.col("horizon") == "all"), naive)
    per_station = (
        rel.group_by("unique_id").agg(pl.col("rel_mae").mean()).sort("unique_id")
    )
    stations = dict(per_station.iter_rows())
    value = sum(stations.values()) / len(stations)
    return value, stations, float(rel["mae"].mean())


def run_study(
    model: str,
    frame: pl.DataFrame,
    cfg: dict[str, Any],
    n_trials: int,
    storage: str,
    study_name: str | None = None,
    suggest: Suggest | None = None,
) -> optuna.Study:
    """Crea o reanuda el estudio hasta tener ``n_trials`` pruebas completas.

    Naive se evalúa una sola vez en las ventanas de ajuste. Cada prueba guarda
    ``rel_mae`` por estación y ``mae_mean`` como ``user_attrs``.
    """
    from orinoco_forecast.models.statistical.baselines import BaselinesRunner

    data, tcfg = tuning_frame(frame, cfg), tuning_config(cfg)
    naive_cv = predict_cv("baselines", BaselinesRunner(), data, tcfg)
    naive = evaluate_cv(naive_cv, ["Naive"], data)
    base, suggest = _base_params(model), suggest or SPACES[model]

    def objective(trial: optuna.Trial) -> float:
        runner = _runner(model, suggest(trial, base))
        cv = predict_cv(model, runner, data, tcfg)
        value, stations, mae = score_trial(
            evaluate_cv(cv, runner.model_columns, data), naive
        )
        trial.set_user_attr("rel_mae", stations)
        trial.set_user_attr("mae_mean", mae)
        return value

    study = optuna.create_study(
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=SAMPLER_SEED),
        study_name=study_name or model,
        direction="minimize",
        load_if_exists=True,
    )
    complete = optuna.trial.TrialState.COMPLETE
    done = len(study.get_trials(deepcopy=False, states=(complete,)))
    if done < n_trials:
        # Cap total attempts so a space that only prunes cannot loop forever.
        study.optimize(
            objective,
            n_trials=10 * n_trials,
            callbacks=[optuna.study.MaxTrialsCallback(n_trials, states=(complete,))],
        )
    return study


def trials_table(study: optuna.Study) -> pl.DataFrame:
    """Una fila por prueba: número, estado, valor, duración, parámetros y rel_mae."""
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
