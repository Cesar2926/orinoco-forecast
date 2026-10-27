"""BoosterRunner: XGBoost/mlforecast adapted to the single-cutoff contract."""

from datetime import date

import polars as pl
import pytest

from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.models.trees.tree import BoosterRunner, load_tree_config

EPOCH = date(1970, 1, 1)
CUTOFF = date(2022, 12, 1)
HORIZON = 3


def _synthetic(start: date = date(2019, 1, 1), end: date = date(2022, 12, 31)):
    days = pl.date_range(start, end, "1d", eager=True)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=(t / 30).sin() * 3 + 10 + k,
            ayacucho=t + 100_000 * k,
            caicara=t + 200_000 * k,
            sin_1=(t / 58.1).sin(),
            cos_1=(t / 58.1).cos(),
            imputed=False,
        )
        for k, uid in enumerate(["ciudad_bolivar", "palua"])
    )


def _tiny_runner() -> BoosterRunner:
    params = load_tree_config()
    xgb = {**params["xgb_params"], "n_estimators": 5, "n_jobs": 1}
    return BoosterRunner({**params, "xgb_params": xgb})


def _windows(frame: pl.DataFrame) -> pl.DataFrame:
    return build_windows(
        frame,
        cutoff=CUTOFF,
        input_length=3,
        horizon=HORIZON,
    )


def test_booster_returns_a_long_prediction_frame() -> None:
    frame = _synthetic()
    runner = _tiny_runner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=HORIZON)

    result = runner.predict(frame, cutoff=CUTOFF, horizon=HORIZON)

    assert result.runner == "booster"
    assert result.frame.height == 6  # 2 stations x 3 steps
    assert result.frame["model"].unique().to_list() == ["booster"]
    assert result.frame["horizon_step"].to_list() == [1, 2, 3, 1, 2, 3]
    assert result.frame["y_pred"].is_finite().all()


def test_booster_ignores_values_after_the_cutoff() -> None:
    frame = _synthetic()
    tampered = frame.with_columns(
        pl.when(pl.col("ds") > CUTOFF).then(1e9).otherwise(pl.col("y")).alias("y")
    )
    base = _tiny_runner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=HORIZON)
    other = _tiny_runner().fit(
        _windows(tampered), tampered, cutoff=CUTOFF, horizon=HORIZON
    )

    left = base.predict(frame, cutoff=CUTOFF, horizon=HORIZON).frame
    right = other.predict(tampered, cutoff=CUTOFF, horizon=HORIZON).frame
    assert left["y_pred"].to_list() == pytest.approx(right["y_pred"].to_list())


def test_booster_predict_before_fit_fails() -> None:
    frame = _synthetic()
    with pytest.raises(RuntimeError, match="fit"):
        _tiny_runner().predict(frame, cutoff=CUTOFF, horizon=HORIZON)


def test_booster_rejects_invalid_horizons() -> None:
    frame = _synthetic()
    with pytest.raises(ValueError, match="horizonte"):
        _tiny_runner().fit(_windows(frame), frame, cutoff=CUTOFF, horizon=0)


def test_booster_rejects_empty_windows() -> None:
    frame = _synthetic()
    with pytest.raises(ValueError, match="ventanas"):
        _tiny_runner().fit(
            _windows(frame).head(0), frame, cutoff=CUTOFF, horizon=HORIZON
        )


def test_booster_rejects_covariate_lags_shorter_than_the_horizon() -> None:
    frame = _synthetic()
    params = load_tree_config()
    runner = BoosterRunner({**params, "covariate_lags": [1, 2]})
    with pytest.raises(ValueError, match="horizonte"):
        runner.fit(_windows(frame), frame, cutoff=CUTOFF, horizon=HORIZON)
