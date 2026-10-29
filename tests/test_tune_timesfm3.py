"""Zero-shot search of TimesFM 3: grid and objective, with a fake forecaster."""

import importlib.util
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from orinoco_forecast.models.pretrained import timesfm

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
EPOCH = date(1970, 1, 1)


@pytest.fixture(scope="module")
def tune():
    # The script imports tune_cnn_lstm as a sibling module, as when run directly.
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(SCRIPTS))
        spec = importlib.util.spec_from_file_location(
            "tune_timesfm3", SCRIPTS / "tune_timesfm3.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def test_grid_is_the_six_zero_shot_configurations(tune) -> None:
    combos = tune.grid()
    assert len(combos) == 6
    assert {c["context_len"] for c in combos} == {512, 1024, 2048}
    assert {c["use_covariates"] for c in combos} == {False, True}


@dataclass
class _Output:
    forecast: np.ndarray


class _Persistence:
    def predict_batch(self, contexts, horizon, **kwargs):
        return (_Output(np.full(horizon, c[-1])) for c in contexts)


def test_objective_scores_against_naive(tune, monkeypatch) -> None:
    # Persistence equals Naive, so every station's rel_mae must be exactly 1.
    monkeypatch.setattr(timesfm, "load_forecaster", lambda *a: _Persistence())
    days = pl.date_range(date(2021, 1, 1), date(2022, 12, 31), "1d", eager=True)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    frame = pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=(t / 30).sin() * 3 + 10 + k
        )
        for k, uid in enumerate(["ciudad_bolivar", "palua"])
    )
    params = {**timesfm.load_timesfm_config(), "context_len": 64}
    cutoffs = [date(2022, 11, 1), date(2022, 11, 24)]
    value, stations, _ = tune.evaluate(params, frame, cutoffs, 3)
    assert value == pytest.approx(1.0)
    assert set(stations) == {"ciudad_bolivar", "palua"}
