"""TCNRunner: single-cutoff fit/predict while the legacy CV stays available."""

from datetime import date

import polars as pl
import pytest

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.neural.cnn import TCNRunner, load_cnn_config

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
        for k, uid in enumerate(["ciudad_bolivar", "palua"])
    )


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


def _windows(frame: pl.DataFrame) -> pl.DataFrame:
    return build_windows(
        frame,
        cutoff=CUTOFF,
        input_length=3,
        horizon=HORIZON,
    )


def test_tcn_returns_one_prediction_per_seed_and_station() -> None:
    frame = _synthetic()
    runner = TCNRunner(_tiny_params()).fit(
        _windows(frame), frame, cutoff=CUTOFF, horizon=HORIZON
    )

    result = runner.predict(frame, cutoff=CUTOFF, horizon=HORIZON)

    assert result.runner == "tcn"
    assert result.frame.height == 6
    assert result.frame["model"].unique().to_list() == ["TCN_s1"]
    assert set(result.frame["horizon_step"]) == {1, 2, 3}
    assert result.frame["y_pred"].is_finite().all()
    assert result.frame["y_pred"].min() > 0


def test_tcn_ignores_values_after_the_cutoff() -> None:
    frame = _synthetic()
    future = pl.col("ds") > CUTOFF
    tampered = frame.with_columns(
        y=pl.when(future).then(1e6).otherwise(pl.col("y")),
        ayacucho=pl.when(future).then(-1e6).otherwise(pl.col("ayacucho")),
    )
    base = TCNRunner(_tiny_params()).fit(
        _windows(frame), frame, cutoff=CUTOFF, horizon=HORIZON
    )
    other = TCNRunner(_tiny_params()).fit(
        _windows(tampered), tampered, cutoff=CUTOFF, horizon=HORIZON
    )

    left = base.predict(frame, cutoff=CUTOFF, horizon=HORIZON).frame
    right = other.predict(tampered, cutoff=CUTOFF, horizon=HORIZON).frame
    assert left["y_pred"].to_list() == pytest.approx(right["y_pred"].to_list())


def test_tcn_predict_before_fit_fails() -> None:
    frame = _synthetic()
    with pytest.raises(RuntimeError, match="fit"):
        TCNRunner(_tiny_params()).predict(frame, cutoff=CUTOFF, horizon=HORIZON)


def test_tcn_rejects_invalid_horizons() -> None:
    frame = _synthetic()
    with pytest.raises(ValueError, match="horizonte"):
        TCNRunner(_tiny_params()).fit(_windows(frame), frame, cutoff=CUTOFF, horizon=0)


def test_tcn_difference_transform_returns_levels() -> None:
    frame = _synthetic()
    params = {**_tiny_params(), "target_transform": "difference"}
    runner = TCNRunner(params).fit(
        _windows(frame), frame, cutoff=CUTOFF, horizon=HORIZON
    )

    result = runner.predict(frame, cutoff=CUTOFF, horizon=HORIZON)

    assert result.frame["y_pred"].is_finite().all()
    assert result.frame["y_pred"].min() > 5
