"""TCN helpers: receptive field, covariate split, seeds and difference target."""

from datetime import date

import polars as pl
import pytest

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.neural.cnn import (
    TCNRunner,
    load_cnn_config,
    prepare_frame,
    receptive_field,
    split_covariates,
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
    frame = _synthetic()
    windows = build_windows(frame, cutoff=CUTOFF, input_length=3, horizon=HORIZON)
    runner = TCNRunner({**load_cnn_config(), "input_size": 28})
    with pytest.raises(ValueError, match="campo receptivo"):
        runner.fit(windows, frame, cutoff=CUTOFF, horizon=HORIZON)


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
    assert runner.implemented is True


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
