"""TimesFM3Runner: context, covariates and output, with a fake forecaster."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
import pytest

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.pretrained import timesfm
from orinoco_forecast.models.pretrained.timesfm import (
    TimesFM3Runner,
    load_timesfm_config,
)
from orinoco_forecast.models.registry import get_runner

EPOCH = date(1970, 1, 1)
CUTOFF = date(2022, 12, 1)
HORIZON = 3


@dataclass
class _Output:
    forecast: np.ndarray


class FakeForecaster:
    """Records its inputs; forecasts the last context value (persistence)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict_batch(self, contexts, horizon, past_only_covariates=None, **kwargs):
        self.calls.append(
            {"contexts": contexts, "past": past_only_covariates, "kwargs": kwargs}
        )
        # TimesFM returns whole output patches; the runner must trim to horizon.
        return (_Output(np.full(horizon + 5, c[-1])) for c in contexts)


def _synthetic():
    days = pl.date_range(date(2021, 1, 1), date(2022, 12, 31), "1d", eager=True)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=(t / 30).sin() * 3 + 10 + k,
            ayacucho_lag_7=(t / 30 + 1).sin() + k,
            rain_x=(t / 7).cos(),
            imputed=False,
        )
        for k, uid in enumerate(["ciudad_bolivar", "palua"])
    )


def _run(frame, **overrides):
    fake = FakeForecaster()
    params = {**load_timesfm_config(), "context_len": 30, **overrides}
    windows = build_windows(frame, cutoff=CUTOFF, input_length=3, horizon=HORIZON)
    runner = TimesFM3Runner(params, forecaster=fake).fit(
        windows, frame, cutoff=CUTOFF, horizon=HORIZON
    )
    return runner.predict(frame, cutoff=CUTOFF, horizon=HORIZON), fake


def test_one_trimmed_forecast_per_station_and_step() -> None:
    frame = _synthetic()
    result, _ = _run(frame)

    assert result.runner == "timesfm3"
    assert result.frame.height == 2 * HORIZON
    assert set(result.frame["model"]) == {"TIMESFM3"}
    last = frame.filter(pl.col("ds") == CUTOFF).sort("unique_id")["y"].to_list()
    by_station = result.frame.group_by("unique_id").agg(pl.col("y_pred").first())
    assert sorted(by_station["y_pred"].to_list()) == pytest.approx(sorted(last))


def test_context_has_context_len_days_up_to_the_cutoff() -> None:
    frame = _synthetic()
    _, fake = _run(frame, context_len=30)
    contexts = fake.calls[0]["contexts"]
    expected = (
        frame.filter(
            (pl.col("unique_id") == "ciudad_bolivar") & (pl.col("ds") <= CUTOFF)
        )
        .tail(30)["y"]
        .to_numpy()
    )
    assert len(contexts) == 2
    assert contexts[0] == pytest.approx(expected)
    assert fake.calls[0]["past"] is None


def test_ignores_values_after_the_cutoff() -> None:
    frame = _synthetic()
    tampered = frame.with_columns(
        y=pl.when(pl.col("ds") > CUTOFF).then(1e6).otherwise(pl.col("y"))
    )
    left, _ = _run(frame, use_covariates=True)
    right, _ = _run(tampered, use_covariates=True)
    assert left.frame["y_pred"].to_list() == right.frame["y_pred"].to_list()


def test_covariates_are_past_only_and_respect_drop_covariates() -> None:
    frame = _synthetic()
    _, fake = _run(frame, use_covariates=True, drop_covariates=["rain_"])
    past = fake.calls[0]["past"]
    assert len(past) == 2
    assert past[0].shape == (1, 30)  # only ayacucho_lag_7 survives the drop


def test_weights_are_loaded_once_per_checkpoint(monkeypatch) -> None:
    loads = []

    class Stub:
        @staticmethod
        def from_pretrained(checkpoint, device=None):
            loads.append(checkpoint)
            return FakeForecaster()

    monkeypatch.setattr(timesfm, "_FORECASTERS", {})
    monkeypatch.setitem(
        __import__("sys").modules,
        "timesfm",
        type("M", (), {"TimesFM3Forecaster": Stub}),
    )
    first = timesfm.load_forecaster("ckpt", None)
    second = timesfm.load_forecaster("ckpt", None)
    assert first is second
    assert loads == ["ckpt"]


def test_predict_before_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        TimesFM3Runner(load_timesfm_config()).predict(
            _synthetic(), cutoff=CUTOFF, horizon=HORIZON
        )


def test_runner_is_registered() -> None:
    runner = get_runner("timesfm")
    assert runner.name == "timesfm3"
    assert runner.implemented is True
