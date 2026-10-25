"""TCN runner: receptive field guard, covariate split, registry and a CV smoke test."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.evaluation.backtest import load_backtest_config, run_cv
from orinoco_forecast.models.neural.cnn import (
    TCNRunner,
    load_cnn_config,
    prepare_frame,
    receptive_field,
    reconstruct_levels,
    split_covariates,
)
from orinoco_forecast.models.registry import get_runner

EPOCH = date(1970, 1, 1)


def _synthetic(start: date, end: date) -> pl.DataFrame:
    days = pl.date_range(start, end, "1d", eager=True)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=(t / 30).sin() * 3 + 10 + k,
            ayacucho=(t / 30 + 1).sin() + k,
            rain_x=(t / 7).cos(),
            sin_1=(t / 58.1).sin(),
            cos_1=(t / 58.1).cos(),
            imputed=False,
        )
        for k, uid in enumerate(["a", "b"])
    )


def test_receptive_field_formula() -> None:
    assert receptive_field(2, [1, 2, 4, 8, 16]) == 32
    assert receptive_field(3, [1, 2, 4, 8, 16]) == 63


def test_runner_rejects_input_shorter_than_receptive_field() -> None:
    params = {**load_cnn_config(), "input_size": 28}
    cfg = {**load_backtest_config(), "n_windows": 2}
    with pytest.raises(ValueError, match="campo receptivo"):
        TCNRunner(params).cross_validation(
            _synthetic(date(2020, 1, 1), date(2020, 12, 31)), cfg
        )


def test_split_covariates_future_fourier_and_historical_rest() -> None:
    futr, hist = split_covariates(_synthetic(date(2020, 1, 1), date(2020, 1, 31)))
    assert futr == ["sin_1", "cos_1"]
    assert hist == ["ayacucho", "rain_x"]


def test_model_columns_follow_seeds() -> None:
    runner = TCNRunner({**load_cnn_config(), "seeds": [1, 2, 3]})
    assert runner.model_columns == ["TCN_s1", "TCN_s2", "TCN_s3"]


def test_registry_returns_tcn_runner() -> None:
    runner = get_runner("tcn")
    assert isinstance(runner, TCNRunner)
    assert runner.name == "tcn"


def _tiny_params() -> dict:
    return {
        **load_cnn_config(),
        "input_size": 8,
        "dilations": [1, 2, 4],
        "encoder_hidden_size": 8,
        "decoder_hidden_size": 8,
        "max_steps": 5,
        "val_check_steps": 5,
        "val_size": 28,
        "seeds": [1],
    }


def test_runner_cross_validation_smoke(tmp_path: Path) -> None:
    cfg = {**load_backtest_config(), "n_windows": 2}
    frame = _synthetic(date(2021, 1, 1), date(2022, 12, 31))
    runner = TCNRunner(_tiny_params())
    cv, metrics = run_cv("tcn", runner, frame, cfg, output_root=tmp_path)
    assert cv.columns == ["unique_id", "ds", "cutoff", "y", "TCN_s1"]
    assert cv["cutoff"].n_unique() == 2
    assert cv.height == 2 * 2 * cfg["horizon"]
    assert cv["TCN_s1"].null_count() == 0
    assert set(metrics["model"]) == {"TCN_s1"}


def test_historical_covariates_after_last_cutoff_do_not_change_forecasts() -> None:
    cfg = {**load_backtest_config(), "n_windows": 2}
    frame = _synthetic(date(2021, 1, 1), date(2022, 12, 31))
    last_cutoff = frame["ds"].max() - timedelta(days=cfg["horizon"])
    future = pl.col("ds") > last_cutoff
    shocked = frame.with_columns(
        ayacucho=pl.when(future).then(1e6).otherwise(pl.col("ayacucho")),
        rain_x=pl.when(future).then(-1e6).otherwise(pl.col("rain_x")),
    )
    base = TCNRunner(_tiny_params()).cross_validation(frame, cfg)
    other = TCNRunner(_tiny_params()).cross_validation(shocked, cfg)
    assert base["TCN_s1"].to_list() == pytest.approx(other["TCN_s1"].to_list())


def test_reconstruct_levels_adds_cumulative_differences_to_cutoff_level() -> None:
    cutoff = date(2021, 1, 10)
    days = [cutoff + timedelta(days=k) for k in range(1, 4)]
    diffs = pl.DataFrame(
        {
            "unique_id": ["a"] * 3,
            "ds": days,
            "cutoff": [cutoff] * 3,
            "y": [0.0] * 3,
            "TCN_s1": [0.5, -0.2, 0.1],
        }
    )
    # Level at the cutoff is 10; later observed levels differ on purpose.
    levels = pl.DataFrame(
        {
            "unique_id": ["a"] * 4,
            "ds": [cutoff, *days],
            "y": [10.0, 99.0, 98.0, 97.0],
        }
    )
    out = reconstruct_levels(diffs, levels, ["TCN_s1"])
    assert out["TCN_s1"].to_list() == pytest.approx([10.5, 10.3, 10.4])
    assert out["y"].to_list() == [99.0, 98.0, 97.0]


def test_level_y_is_historical_only_with_difference_transform() -> None:
    frame = _synthetic(date(2020, 1, 1), date(2020, 1, 31))
    data, futr, hist = prepare_frame(frame, "level")
    assert "level_y" not in hist and "level_y" not in data.columns
    data, futr, hist = prepare_frame(frame, "difference")
    assert hist[-1] == "level_y" and futr == ["sin_1", "cos_1"]
    first = frame.filter(pl.col("unique_id") == "a").sort("ds")
    a = data.filter(pl.col("unique_id") == "a")
    assert a.height == first.height - 1
    assert a["y"][0] == pytest.approx(first["y"][1] - first["y"][0])
    assert a["level_y"][0] == pytest.approx(first["y"][1])


def test_difference_cross_validation_returns_levels() -> None:
    cfg = {**load_backtest_config(), "n_windows": 2}
    frame = _synthetic(date(2021, 1, 1), date(2022, 12, 31))
    params = {**_tiny_params(), "target_transform": "difference"}
    cv = TCNRunner(params).cross_validation(frame, cfg)
    assert cv.columns == ["unique_id", "ds", "cutoff", "y", "TCN_s1"]
    truth = cv.join(frame, on=["unique_id", "ds"], suffix="_obs")
    assert truth["y"].to_list() == pytest.approx(truth["y_obs"].to_list())
    assert cv["TCN_s1"].null_count() == 0
    # Reconstructed levels stay near the series range (8..14), not near 0.
    assert cv["TCN_s1"].min() > 5
