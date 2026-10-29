"""TransformerRunner: single-cutoff fit/predict of the temporal Transformer."""

from datetime import date

import polars as pl
import pytest
import torch

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.neural.transformer import (
    TemporalTransformer,
    TransformerRunner,
    load_transformer_config,
)
from orinoco_forecast.models.registry import get_runner

EPOCH = date(1970, 1, 1)
CUTOFF = date(2022, 12, 1)
HORIZON = 3


def _synthetic(start: date = date(2021, 1, 1), end: date = date(2022, 12, 31)):
    days = pl.date_range(start, end, "1d", eager=True)
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


def _tiny_params(**overrides) -> dict:
    return {
        **load_transformer_config(),
        "input_size": 8,
        "d_model": 8,
        "nhead": 2,
        "num_layers": 1,
        "dim_feedforward": 16,
        "batch_size": 64,
        "max_epochs": 2,
        "early_stop_patience": 2,
        "val_size": 28,
        "seeds": [1],
        **overrides,
    }


def _fit(frame: pl.DataFrame, **overrides) -> TransformerRunner:
    windows = build_windows(frame, cutoff=CUTOFF, input_length=3, horizon=HORIZON)
    return TransformerRunner(_tiny_params(**overrides)).fit(
        windows, frame, cutoff=CUTOFF, horizon=HORIZON
    )


def test_returns_one_prediction_per_seed_station_and_step() -> None:
    frame = _synthetic()
    result = _fit(frame, seeds=[1, 2]).predict(frame, cutoff=CUTOFF, horizon=HORIZON)

    assert result.runner == "transformer"
    assert result.frame.height == 2 * 2 * HORIZON
    assert sorted(result.frame["model"].unique().to_list()) == [
        "TRANSFORMER_s1",
        "TRANSFORMER_s2",
    ]
    assert result.frame["y_pred"].is_finite().all()


def test_ignores_values_after_the_cutoff() -> None:
    frame = _synthetic()
    future = pl.col("ds") > CUTOFF
    tampered = frame.with_columns(
        y=pl.when(future).then(1e6).otherwise(pl.col("y")),
        ayacucho=pl.when(future).then(-1e6).otherwise(pl.col("ayacucho")),
    )

    left = _fit(frame).predict(frame, cutoff=CUTOFF, horizon=HORIZON).frame
    right = _fit(tampered).predict(tampered, cutoff=CUTOFF, horizon=HORIZON).frame

    assert left["y_pred"].to_list() == pytest.approx(right["y_pred"].to_list())


def test_module_outputs_one_value_per_horizon_step() -> None:
    net = TemporalTransformer(
        n_features=3,
        horizon=14,
        input_size=20,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
    )
    assert net(torch.randn(5, 20, 3)).shape == (5, 14)


def test_rejects_d_model_not_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="múltiplo"):
        TemporalTransformer(
            n_features=2,
            horizon=3,
            input_size=8,
            d_model=10,
            nhead=4,
            num_layers=1,
            dim_feedforward=8,
        )


def test_runner_is_registered_and_uses_its_config() -> None:
    runner = get_runner("transformer")
    assert runner.name == "transformer"
    assert runner.implemented is True
    assert runner.params["d_model"] % runner.params["nhead"] == 0
