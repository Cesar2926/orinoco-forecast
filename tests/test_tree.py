"""Booster helpers: leak-free pre-lagged covariates and config loading."""

from datetime import date

import polars as pl
import pytest

from orinoco_forecast.models.trees.tree import load_tree_config, prelag_covariates

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


def test_tree_config_has_the_direct_strategy_defaults() -> None:
    config = load_tree_config()
    assert config["lags"][0] == 1
    assert min(config["covariate_lags"]) >= 14
    assert config["target_differences"] == [1]
    with pytest.raises(FileNotFoundError):
        load_tree_config("does_not_exist.yaml")
