"""Optuna search for the CNN-LSTM: tuning cutoffs, space, scoring, objective."""

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import optuna
import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tune_cnn_lstm.py"
spec = importlib.util.spec_from_file_location("tune_cnn_lstm", SCRIPT)
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)

EPOCH = date(1970, 1, 1)
BASE = {
    "target_transform": "difference",
    "val_size": 28,
    "max_epochs": 2,
    "early_stop_patience": 2,
    "seeds": [1],
    "drop_covariates": ["rain_", "oni"],
}


def test_tuning_cutoffs_end_one_horizon_before_the_first_validation_cutoff() -> None:
    cutoffs = tune.tuning_cutoffs(date(2021, 1, 26), horizon=14)
    assert len(cutoffs) == 12
    assert cutoffs[0] == date(2020, 5, 4)
    assert cutoffs[-1] == date(2021, 1, 12)
    assert {b - a for a, b in zip(cutoffs, cutoffs[1:], strict=False)} == {
        timedelta(days=23)
    }


def test_search_space_stays_inside_the_declared_bounds() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(25):
        trial = study.ask()
        try:
            params = tune.suggest(trial, BASE)
        except optuna.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        assert params["input_size"] in (28, 56, 112)
        assert params["conv_channels"] in (16, 32, 64)
        assert params["kernel_size"] in (2, 3, 5)
        assert params["conv_layers"] in (1, 2, 3)
        assert params["lstm_hidden_size"] in (32, 64, 128)
        assert params["lstm_layers"] in (1, 2)
        assert 0.0 <= params["dropout"] <= 0.3
        assert 3e-4 <= params["learning_rate"] <= 1e-2
        assert params["batch_size"] in (128, 256)
        assert params["max_epochs"] == 30
        assert params["early_stop_patience"] == 5
        study.tell(trial, 0.0)


def _predictions(model: str, error: float) -> pl.DataFrame:
    rows = [
        (model, uid, date(2020, 1, 1), date(2020, 1, 1 + s), s, 10.0 + error)
        for uid in ("ciudad_bolivar", "palua")
        for s in (1, 2)
    ]
    return pl.DataFrame(
        rows,
        schema=["model", "unique_id", "cutoff", "ds", "horizon_step", "y_pred"],
        orient="row",
    )


def test_score_is_one_for_naive_errors_and_zero_for_perfect_forecasts() -> None:
    observed = _predictions("x", 0.0).select("unique_id", "ds", y_pred="y_pred")
    observed = observed.rename({"y_pred": "y"})
    naive = _predictions("naive", 2.0)
    same, per_station, _ = tune.score(_predictions("CNNLSTM_s1", 2.0), naive, observed)
    assert same == 1.0 and set(per_station) == {"ciudad_bolivar", "palua"}
    perfect, _, mae = tune.score(_predictions("CNNLSTM_s1", 0.0), naive, observed)
    assert perfect == 0.0 and mae == 0.0


def _frame() -> pl.DataFrame:
    days = pl.date_range(date(2021, 1, 1), date(2022, 6, 30), "1d", eager=True)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=(t / 30).sin() * 3 + 10 + k,
            ayacucho=(t / 30 + 1).sin() + k,
            rain_x=(t / 7).cos(),
            imputed=False,
        )
        for k, uid in enumerate(["ciudad_bolivar", "palua"])
    )


def test_evaluate_returns_rel_mae_per_station_on_synthetic_data() -> None:
    params = {
        **BASE,
        "input_size": 8,
        "conv_channels": 4,
        "kernel_size": 3,
        "conv_layers": 1,
        "lstm_hidden_size": 8,
        "lstm_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "batch_size": 64,
    }
    cutoffs = [date(2022, 5, 1), date(2022, 5, 24)]
    value, per_station, mae = tune.evaluate(params, _frame(), cutoffs, horizon=3)
    assert value > 0 and mae > 0
    assert set(per_station) == {"ciudad_bolivar", "palua"}


def test_heaviest_lstm_combination_is_pruned() -> None:
    fixed = {
        "input_size": 112,
        "conv_channels": 16,
        "kernel_size": 3,
        "conv_layers": 1,
        "lstm_hidden_size": 128,
        "lstm_layers": 2,
        "dropout": 0.1,
        "learning_rate": 0.0004,
        "batch_size": 256,
    }
    with pytest.raises(optuna.TrialPruned):
        tune.suggest(optuna.trial.FixedTrial(fixed), BASE)
    tune.suggest(optuna.trial.FixedTrial({**fixed, "lstm_layers": 1}), BASE)
