"""CnnLstmRunner: single-cutoff fit/predict of the hybrid CNN-LSTM."""

from datetime import date

import polars as pl
import pytest
import torch

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.neural.cnn_lstm import (
    CnnLstm,
    CnnLstmRunner,
    load_cnn_lstm_config,
    reconstruct_levels,
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
        **load_cnn_lstm_config(),
        "input_size": 8,
        "conv_channels": 4,
        "kernel_size": 3,
        "conv_layers": 2,
        "lstm_hidden_size": 8,
        "lstm_layers": 1,
        "batch_size": 64,
        "max_epochs": 2,
        "early_stop_patience": 2,
        "val_size": 28,
        "seeds": [1],
        **overrides,
    }


def _windows(frame: pl.DataFrame) -> pl.DataFrame:
    return build_windows(frame, cutoff=CUTOFF, input_length=3, horizon=HORIZON)


def _fit(frame: pl.DataFrame, **overrides) -> CnnLstmRunner:
    return CnnLstmRunner(_tiny_params(**overrides)).fit(
        _windows(frame), frame, cutoff=CUTOFF, horizon=HORIZON
    )


def test_returns_one_prediction_per_seed_station_and_step() -> None:
    frame = _synthetic()
    result = _fit(frame, seeds=[1, 2]).predict(frame, cutoff=CUTOFF, horizon=HORIZON)

    assert result.runner == "cnn_lstm"
    assert result.frame.height == 2 * 2 * HORIZON
    assert sorted(result.frame["model"].unique().to_list()) == [
        "CNNLSTM_s1",
        "CNNLSTM_s2",
    ]
    assert set(result.frame["horizon_step"]) == {1, 2, 3}
    assert result.frame["y_pred"].is_finite().all()
    assert result.frame["y_pred"].min() > 0


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


def test_reconstruct_levels_zero_change_is_persistence() -> None:
    changes = torch.zeros(2, 4)
    levels = reconstruct_levels(changes, torch.tensor([10.0, 20.0]))
    assert levels.tolist() == [[10.0] * 4, [20.0] * 4]
    steps = reconstruct_levels(torch.ones(1, 3), torch.tensor([5.0]))
    assert steps.tolist() == [[6.0, 7.0, 8.0]]


def test_difference_transform_returns_levels_near_the_last_observation() -> None:
    frame = _synthetic()
    runner = _fit(frame, target_transform="difference")
    result = runner.predict(frame, cutoff=CUTOFF, horizon=HORIZON)

    assert result.frame["y_pred"].is_finite().all()
    assert result.frame["y_pred"].min() > 5


def test_level_transform_is_supported() -> None:
    frame = _synthetic()
    result = _fit(frame, target_transform="level").predict(
        frame, cutoff=CUTOFF, horizon=HORIZON
    )
    assert result.frame["y_pred"].is_finite().all()


def test_module_outputs_one_value_per_horizon_step() -> None:
    net = CnnLstm(
        n_features=3,
        horizon=14,
        conv_channels=4,
        kernel_size=3,
        conv_layers=2,
        lstm_hidden_size=8,
        lstm_layers=1,
    )
    assert net(torch.randn(5, 20, 3)).shape == (5, 14)


def test_convolution_is_causal() -> None:
    net = CnnLstm(
        n_features=3,
        horizon=2,
        conv_channels=4,
        kernel_size=3,
        conv_layers=2,
        lstm_hidden_size=8,
        lstm_layers=1,
    ).eval()
    x = torch.randn(2, 12, 3)
    changed = x.clone()
    changed[:, -1, :] += 10.0
    before = net.conv(x.transpose(1, 2))
    after = net.conv(changed.transpose(1, 2))
    assert torch.allclose(before[..., :-1], after[..., :-1])
    assert not torch.allclose(before[..., -1], after[..., -1])


def test_predict_before_fit_fails() -> None:
    frame = _synthetic()
    with pytest.raises(RuntimeError, match="fit"):
        CnnLstmRunner(_tiny_params()).predict(frame, cutoff=CUTOFF, horizon=HORIZON)


def test_rejects_invalid_horizon_and_validation_size() -> None:
    frame = _synthetic()
    with pytest.raises(ValueError, match="horizonte"):
        CnnLstmRunner(_tiny_params()).fit(
            _windows(frame), frame, cutoff=CUTOFF, horizon=0
        )
    with pytest.raises(ValueError, match="val_size"):
        _fit(frame, val_size=HORIZON - 1)


def test_drop_covariates_hides_prefixed_columns() -> None:
    frame = _synthetic()
    assert "rain_x" in _fit(frame, drop_covariates=[]).features
    hidden = _fit(frame, drop_covariates=["rain_"]).features
    assert "rain_x" not in hidden
    assert "ayacucho" in hidden


def test_runner_is_registered() -> None:
    runner = get_runner("cnn_lstm")
    assert runner.name == "cnn_lstm"
    assert runner.implemented is True
