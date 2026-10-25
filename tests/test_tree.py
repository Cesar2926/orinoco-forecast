"""XGBoost runner: leak-free pre-lagged covariates, CV smoke test and registry."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.evaluation.backtest import load_backtest_config, run_cv
from orinoco_forecast.models.registry import get_runner
from orinoco_forecast.models.trees.tree import (
    XGBoostRunner,
    load_tree_config,
    prelag_covariates,
)

EPOCH = date(1970, 1, 1)
LAGS = [14, 21]


def _synthetic(start: date, end: date) -> pl.DataFrame:
    """Two series whose upstream values encode their own date (day ordinal)."""
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
        for k, uid in enumerate(["a", "b"])
    )


def test_prelag_matches_past_value_of_same_series() -> None:
    frame = _synthetic(date(2020, 1, 1), date(2020, 6, 30))
    out = prelag_covariates(frame, ["ayacucho", "caicara"], LAGS)
    past = frame.select(
        "unique_id",
        (pl.col("ds") + pl.duration(days=14)).alias("ds"),
        pl.col("ayacucho").alias("expected"),
    )
    joined = out.join(past, on=["unique_id", "ds"], how="left")
    assert joined["expected"].null_count() == 0
    assert (joined["ayacucho_lag14"] == joined["expected"]).all()


def test_prelag_never_uses_values_after_row_date() -> None:
    frame = _synthetic(date(2020, 1, 1), date(2020, 6, 30))
    out = prelag_covariates(frame, ["ayacucho", "caicara"], LAGS)
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    offset = {"a": 0, "b": 1}
    out = out.with_columns(
        k=pl.col("unique_id").replace_strict(offset, return_dtype=pl.Float64)
    )
    for lag in LAGS:
        source_day = pl.col(f"ayacucho_lag{lag}") - 100_000 * pl.col("k")
        assert out.select((source_day == t - lag).all()).item()
        assert out.select((source_day < t).all()).item()


def test_prelag_drops_same_day_columns_metadata_and_warmup_rows() -> None:
    frame = _synthetic(date(2020, 1, 1), date(2020, 6, 30))
    out = prelag_covariates(frame, ["ayacucho", "caicara"], LAGS)
    assert {"ayacucho", "caicara", "imputed"}.isdisjoint(out.columns)
    assert {"sin_1", "cos_1", "caicara_lag21"} <= set(out.columns)
    assert out.null_count().sum_horizontal().item() == 0
    assert out["ds"].min() == date(2020, 1, 22)
    assert out["ds"].max() == frame["ds"].max()


def test_runner_rejects_covariate_lag_shorter_than_horizon() -> None:
    params = load_tree_config()
    runner = XGBoostRunner({**params, "covariate_lags": [7, 14]})
    cfg = {**load_backtest_config(), "n_windows": 2}
    with pytest.raises(ValueError, match="horizonte"):
        runner.cross_validation(_synthetic(date(2019, 1, 1), date(2020, 12, 31)), cfg)


def _tiny_runner() -> XGBoostRunner:
    params = load_tree_config()
    xgb = {**params["xgb_params"], "n_estimators": 5, "n_jobs": 1}
    return XGBoostRunner({**params, "xgb_params": xgb})


def test_runner_cross_validation_follows_protocol(tmp_path: Path) -> None:
    cfg = {**load_backtest_config(), "n_windows": 3}
    frame = _synthetic(date(2019, 1, 1), date(2022, 12, 31))
    cv, metrics = run_cv("xgboost", _tiny_runner(), frame, cfg, output_root=tmp_path)
    assert cv.columns == ["unique_id", "ds", "cutoff", "y", "XGBoost"]
    assert cv["cutoff"].n_unique() == 3
    assert cv.height == 2 * 3 * cfg["horizon"]
    assert cv["XGBoost"].null_count() == 0
    assert set(metrics["model"]) == {"XGBoost"}


def test_feature_importance_covers_lags_and_covariates() -> None:
    cfg = load_backtest_config()
    frame = _synthetic(date(2021, 1, 1), date(2022, 12, 31))
    imp = _tiny_runner().feature_importance(frame, cfg)
    assert imp.columns == ["feature", "importance_gain"]
    assert imp["importance_gain"].is_sorted(descending=True)
    assert not {"ayacucho", "caicara", "imputed"} & set(imp["feature"])
    assert set(imp["feature"]) <= {
        *(f"lag{k}" for k in load_tree_config()["lags"]),
        "rolling_mean_lag1_window_size7",
        "rolling_mean_lag1_window_size30",
        "sin_1",
        "cos_1",
        *(f"{c}_lag{k}" for c in ["ayacucho", "caicara"] for k in [14, 21, 28, 56]),
    }


def test_registry_returns_xgboost_runner() -> None:
    runner = get_runner("xgboost")
    assert isinstance(runner, XGBoostRunner)
    assert runner.name == "xgboost"
    assert runner.model_columns == ["XGBoost"]
