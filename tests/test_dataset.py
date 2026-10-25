"""Model frame: cleaning, Fourier terms, leakage guard and the real stage A/B frames."""

import math
from datetime import date, timedelta

import polars as pl
import pytest

from orinoco_forecast.data.loading import DataQualityWarning
from orinoco_forecast.data.quality import QUALITY_DTYPE
from orinoco_forecast.features.dataset import (
    assert_no_imputation_near_cutoffs,
    clean_series,
    fourier_terms,
)

START = date(2020, 1, 1)


def _series(values: list[float | None], quality: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": ["a"] * len(values),
            "ds": [START + timedelta(days=i) for i in range(len(values))],
            "y": values,
            "quality": pl.Series(quality, dtype=QUALITY_DTYPE),
        },
        schema_overrides={"y": pl.Float64},
    )


def test_clean_series_nulls_flat_run_and_interpolates() -> None:
    q = ["observed", "observed", *["suspect_flat"] * 3, "observed", "observed"]
    out = clean_series(_series([1.0, 2.0, 5.0, 5.0, 5.0, 6.0, 7.0], q), 31)
    assert out["y"].to_list() == pytest.approx([1, 2, 3, 4, 5, 6, 7])
    assert out["imputed"].to_list() == [False, False, True, True, True, False, False]
    assert "quality" in out.columns


def test_clean_series_keeps_source_conflict_values() -> None:
    q = ["observed", "source_conflict", "observed"]
    out = clean_series(_series([1.0, 9.0, 3.0], q), 31)
    assert out["y"].to_list() == [1.0, 9.0, 3.0]
    assert not out["imputed"].any()


def test_clean_series_long_gap_stays_null_and_warns() -> None:
    q = ["observed", *["missing"] * 3, "observed", "missing", "observed"]
    df = _series([1.0, None, None, None, 5.0, None, 7.0], q)
    with pytest.warns(DataQualityWarning, match="3"):
        out = clean_series(df, max_gap_days=2)
    assert out["y"].to_list()[1:4] == [None, None, None]
    assert out["y"][5] == pytest.approx(6.0)
    assert out["imputed"].to_list() == [False] * 5 + [True, False]


def test_fourier_terms_columns_and_values() -> None:
    ds = pl.Series("ds", [date(1970, 1, 1), date(1970, 1, 2)])
    out = fourier_terms(ds, period=4.0, k=2)
    assert out.columns == ["ds", "sin_1", "sin_2", "cos_1", "cos_2"]
    assert out["sin_1"].to_list() == pytest.approx([0.0, 1.0])
    assert out["cos_1"].to_list() == pytest.approx([1.0, 0.0], abs=1e-12)
    assert out["sin_2"].to_list() == pytest.approx([0.0, 0.0], abs=1e-12)


def test_fourier_terms_are_periodic() -> None:
    ds = pl.Series("ds", [date(2001, 3, 5), date(2001, 3, 5) + timedelta(days=1461)])
    out = fourier_terms(ds, period=365.25, k=3)
    first, second = out.drop("ds").rows()
    assert first == pytest.approx(second)
    t = (date(2001, 3, 5) - date(1970, 1, 1)).days
    assert out["sin_3"][0] == pytest.approx(math.sin(2 * math.pi * 3 * t / 365.25))


def _imputed_frame(imputed_day: date) -> pl.DataFrame:
    days = [date(2021, 1, 1) + timedelta(days=i) for i in range(120)]
    return pl.DataFrame(
        {
            "unique_id": "a",
            "ds": days,
            "y": 1.0,
            "imputed": [d == imputed_day for d in days],
        }
    )


def test_leakage_guard_raises_near_cutoff() -> None:
    cutoff = date(2021, 2, 1)
    for day in (cutoff - timedelta(days=14), cutoff, cutoff + timedelta(days=14)):
        with pytest.raises(ValueError, match="imputad"):
            assert_no_imputation_near_cutoffs(_imputed_frame(day), [cutoff], 14)


def test_leakage_guard_passes_far_from_cutoff() -> None:
    cutoff = date(2021, 2, 1)
    for day in (cutoff - timedelta(days=15), cutoff + timedelta(days=15)):
        assert_no_imputation_near_cutoffs(_imputed_frame(day), [cutoff], 14)


FOURIER = ["sin_1", "sin_2", "sin_3", "cos_1", "cos_2", "cos_3"]


def test_model_frame_stage_a(model_frame_a: pl.DataFrame) -> None:
    frame = model_frame_a
    assert frame.columns == [
        "unique_id",
        "ds",
        "y",
        "ayacucho",
        "caicara",
        *FOURIER,
        "imputed",
    ]
    assert sorted(frame["unique_id"].unique()) == ["ciudad_bolivar", "palua"]
    assert frame.null_count().sum_horizontal().item() == 0
    assert frame["ds"].max() == date(2025, 2, 24)
    assert frame["ds"].min() == date(1974, 1, 1)
    assert frame.select(pl.col("ds").is_sorted().over("unique_id")).to_series().all()
    # The 2020 flat run at Ciudad Bolivar (19 days) is imputed.
    assert frame.filter(pl.col("unique_id") == "ciudad_bolivar")["imputed"].sum() >= 19


def test_model_frame_stage_b(model_frame_b: pl.DataFrame) -> None:
    frame = model_frame_b
    rain = ["rain_ayacucho", "rain_caicara", "rain_ciudad_bolivar", "rain_palua"]
    assert set(rain + ["oni", *FOURIER]).issubset(frame.columns)
    assert frame["imputed"].dtype == pl.Boolean and frame.columns[-1] == "imputed"
    assert frame["ds"].min() == date(1981, 1, 1)
    assert frame.null_count().sum_horizontal().item() == 0
