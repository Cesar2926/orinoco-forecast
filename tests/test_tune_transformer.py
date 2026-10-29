"""Optuna search for the Transformer: space and objective on the shared protocol."""

import importlib.util
import sys
from datetime import date
from pathlib import Path

import optuna
import polars as pl
import pytest

from orinoco_forecast.models.neural.transformer import (
    TemporalTransformer,
    TransformerRunner,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
EPOCH = date(1970, 1, 1)
BASE = {
    "target_transform": "difference",
    "val_size": 28,
    "max_epochs": 2,
    "early_stop_patience": 2,
    "seeds": [1],
    "drop_covariates": ["rain_", "oni"],
}


@pytest.fixture(scope="module")
def tune():
    # The script imports tune_cnn_lstm as a sibling module, as when run directly.
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(SCRIPTS))
        spec = importlib.util.spec_from_file_location(
            "tune_transformer", SCRIPTS / "tune_transformer.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def test_every_suggested_configuration_builds_a_network(tune) -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(10):
        params = tune.suggest(study.ask(), BASE)
        net = TemporalTransformer(
            n_features=2,
            horizon=14,
            input_size=params["input_size"],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_layers=params["num_layers"],
            dim_feedforward=params["dim_feedforward"],
        )
        assert sum(p.numel() for p in net.parameters()) > 0
        assert params["max_epochs"] == 30


def test_objective_scores_the_transformer_against_naive(tune) -> None:
    days = pl.date_range(date(2021, 1, 1), date(2022, 12, 31), "1d", eager=True)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    frame = pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=(t / 30).sin() * 3 + 10 + k, ayacucho=(t / 30 + 1).sin() + k
        )
        for k, uid in enumerate(["ciudad_bolivar", "palua"])
    )
    params = {
        **BASE,
        "input_size": 8,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "batch_size": 64,
    }
    cutoffs = [date(2022, 11, 1), date(2022, 11, 24)]
    evaluate = sys.modules["tune_cnn_lstm"].evaluate  # loaded by the script
    value, stations, mae = evaluate(params, frame, cutoffs, 3, TransformerRunner)
    assert value > 0 and mae > 0
    assert set(stations) == {"ciudad_bolivar", "palua"}
