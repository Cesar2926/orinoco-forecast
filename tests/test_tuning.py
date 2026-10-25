"""Tuning harness: windows before validation, search spaces, objective and resume."""

from datetime import date, timedelta
from pathlib import Path

import optuna
import polars as pl
import pytest

from orinoco_forecast.evaluation.backtest import (
    cutoffs,
    development_frame,
    load_backtest_config,
    predict_cv,
    tuning_config,
    tuning_frame,
)
from orinoco_forecast.evaluation.tuning import (
    run_study,
    score_trial,
    suggest_tcn,
    suggest_xgboost,
    trials_table,
)
from orinoco_forecast.features.dataset import assert_no_imputation_near_cutoffs
from orinoco_forecast.models.neural.cnn import load_cnn_config
from orinoco_forecast.models.trees.tree import load_tree_config

EPOCH = date(1970, 1, 1)


def _synthetic(start: date, end: date) -> pl.DataFrame:
    days = pl.date_range(start, end, "1d", eager=True)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=(t / 30).sin() * 3 + 10 + k,
            ayacucho=(t / 30 + 1).sin() + k,
            sin_1=(t / 58.1).sin(),
            cos_1=(t / 58.1).cos(),
            imputed=False,
        )
        for k, uid in enumerate(["a", "b"])
    )


def test_tuning_windows_end_at_first_validation_cutoff() -> None:
    cfg = load_backtest_config()
    frame = _synthetic(date(2015, 1, 1), date(2023, 6, 30))
    first_val = cutoffs(development_frame(frame, cfg), cfg)[0]

    data, tcfg = tuning_frame(frame, cfg), tuning_config(cfg)
    assert data["ds"].max() == first_val
    assert tcfg["n_windows"] == cfg["tuning"]["n_windows"]
    assert tcfg["step_size"] == cfg["tuning"]["step_size"]
    assert (tcfg["horizon"], tcfg["refit"]) == (cfg["horizon"], cfg["refit"])
    tune_cuts = cutoffs(data, tcfg)
    assert len(tune_cuts) == cfg["tuning"]["n_windows"]
    # Every tuning target date is on or before the first validation cutoff.
    assert tune_cuts[-1] + timedelta(days=tcfg["horizon"]) == first_val


def test_tuning_cutoffs_pass_leakage_guard_on_real_frame(
    model_frame_a: pl.DataFrame,
) -> None:
    cfg = load_backtest_config()
    data, tcfg = tuning_frame(model_frame_a, cfg), tuning_config(cfg)
    assert_no_imputation_near_cutoffs(data, cutoffs(data, tcfg), tcfg["horizon"])


def test_predict_cv_runs_leakage_guard_on_tuning_cutoffs() -> None:
    cfg = load_backtest_config()
    frame = _synthetic(date(2015, 1, 1), date(2022, 12, 31))
    data, tcfg = tuning_frame(frame, cfg), tuning_config(cfg)
    near = cutoffs(data, tcfg)[3]
    dirty = data.with_columns(imputed=pl.col("ds") == near)

    class Never:
        name = "never"
        model_columns = ["X"]

        def cross_validation(self, frame, cfg):  # pragma: no cover
            raise AssertionError("el guard debe fallar antes de entrenar")

    with pytest.raises(ValueError, match="imputadas"):
        predict_cv("never", Never(), dirty, tcfg)


def test_suggest_xgboost_maps_covariate_lags_and_keeps_target_features() -> None:
    base = load_tree_config()
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.7,
            "colsample_bytree": 0.9,
            "min_child_weight": 2.0,
            "reg_lambda": 3.0,
            "covariate_lags": "14-28-56-90",
        }
    )
    params = suggest_xgboost(trial, base)
    assert params["covariate_lags"] == [14, 28, 56, 90]
    assert params["xgb_params"]["max_depth"] == 4
    assert params["xgb_params"]["tree_method"] == base["xgb_params"]["tree_method"]
    assert params["lags"] == base["lags"]
    assert params["target_differences"] == base["target_differences"]


def _tcn_trial(**overrides: object) -> optuna.trial.FixedTrial:
    values = {
        "target_transform": "difference",
        "input_size": 56,
        "kernel_size": 2,
        "dilations": "1-16",
        "encoder_hidden_size": 32,
        "learning_rate": 1e-3,
        "max_steps": 1000,
    }
    return optuna.trial.FixedTrial({**values, **overrides})


def test_suggest_tcn_uses_one_seed_and_maps_dilations() -> None:
    params = suggest_tcn(_tcn_trial(), load_cnn_config())
    assert params["seeds"] == [1]
    assert params["dilations"] == [1, 2, 4, 8, 16]
    assert params["target_transform"] == "difference"


def test_suggest_tcn_prunes_receptive_field_longer_than_input() -> None:
    trial = _tcn_trial(kernel_size=3, dilations="1-32", input_size=56)
    with pytest.raises(optuna.TrialPruned):
        suggest_tcn(trial, load_cnn_config())


def _metrics(model: str, maes: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": list(maes),
            "model": [model] * len(maes),
            "horizon": ["all"] * len(maes),
            "n": [10] * len(maes),
            "mae": list(maes.values()),
        }
    )


def test_score_trial_is_mean_rel_mae_over_stations() -> None:
    naive = _metrics("Naive", {"a": 2.0, "b": 4.0})
    model = _metrics("XGBoost", {"a": 1.0, "b": 3.0})
    value, per_station, mean_mae = score_trial(model, naive)
    assert per_station == {"a": 0.5, "b": 0.75}
    assert value == pytest.approx(0.625)
    assert mean_mae == pytest.approx(2.0)


def _tiny_xgboost(trial: optuna.Trial, base: dict) -> dict:
    xgb = {**base["xgb_params"], "n_estimators": 5, "n_jobs": 1}
    xgb["max_depth"] = trial.suggest_int("max_depth", 2, 3)
    return {**base, "xgb_params": xgb}


def test_run_study_records_attrs_and_resumes(tmp_path: Path) -> None:
    cfg = load_backtest_config()
    frame = _synthetic(date(2015, 1, 1), date(2022, 12, 31))
    storage = f"sqlite:///{(tmp_path / 'x.db').as_posix()}"
    study = run_study("xgboost", frame, cfg, 2, storage, suggest=_tiny_xgboost)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(done) == 2
    attrs = done[0].user_attrs
    assert set(attrs["rel_mae"]) == {"a", "b"}
    assert attrs["mae_mean"] > 0
    # Resume: the same storage continues until 3 complete trials, not 3 more.
    study = run_study("xgboost", frame, cfg, 3, storage, suggest=_tiny_xgboost)
    assert len(study.trials) == 3
    table = trials_table(study)
    assert {"number", "value", "state", "rel_mae_a", "rel_mae_b"} <= set(table.columns)
    assert "params_max_depth" in table.columns
