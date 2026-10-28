"""Quality-label tests: synthetic contracts first, then audited raw data."""

from datetime import date, timedelta

import polars as pl
import pytest

from orinoco_forecast.data.quality import (
    QUALITY_LABELS,
    find_flat_runs,
    label_quality,
)

THRESHOLDS = {"flat_min_days": 10, "jump_max_m_per_day": 1.0, "source_conflict_m": 0.3}
START = date(2000, 1, 1)


def _frame(values: list[float | None], uid: str = "a") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unique_id": [uid] * len(values),
            "ds": [START + timedelta(days=i) for i in range(len(values))],
            "y": values,
        },
        schema={"unique_id": pl.String, "ds": pl.Date, "y": pl.Float64},
    )


def _labels(df: pl.DataFrame, **kwargs) -> list[str]:
    out = label_quality(df, thresholds=THRESHOLDS, **kwargs)
    return out["quality"].cast(pl.String).to_list()


def test_labels_constant() -> None:
    assert QUALITY_LABELS == (
        "observed",
        "missing",
        "suspect_flat",
        "suspect_jump",
        "source_conflict",
    )


def test_output_keeps_rows_and_adds_enum_column() -> None:
    df = _frame([1.0, 2.0, 3.0])
    out = label_quality(df, thresholds=THRESHOLDS)
    assert out.drop("quality").equals(df)
    assert out.schema["quality"] == pl.Enum(QUALITY_LABELS)


def test_missing() -> None:
    assert _labels(_frame([1.0, None, 1.1])) == ["observed", "missing", "observed"]


def test_jump_within_series_only() -> None:
    df = pl.concat([_frame([1.0, 1.1, 2.5, 2.6], "a"), _frame([9.0, 9.1], "b")])
    assert _labels(df) == [
        "observed",
        "observed",
        "suspect_jump",
        "observed",
        "observed",
        "observed",
    ]


def test_jump_ignores_exact_threshold_and_nulls() -> None:
    assert _labels(_frame([1.0, 2.0, None, 5.0])) == [
        "observed",
        "observed",
        "missing",
        "observed",
    ]


def test_source_conflict() -> None:
    df = _frame([1.0, 1.0, 1.0, 1.0])
    reference = _frame([1.5, 1.2, None, 1.0]).head(3)
    assert _labels(df, reference=reference) == [
        "source_conflict",
        "observed",
        "observed",
        "observed",
    ]


def test_precedence_missing_conflict_jump() -> None:
    df = _frame([1.0, 3.0, None])
    reference = _frame([1.0, 1.0, 1.0])
    # Day 1 is both a jump and a conflict; day 2 is null but has a reference.
    assert _labels(df, reference=reference) == [
        "observed",
        "source_conflict",
        "missing",
    ]


def test_flat_run_labelled_and_below_conflict() -> None:
    df = _frame([2.0] + [1.0] * 10 + [1.1])
    reference = _frame([2.0, 5.0] + [1.0] * 10)
    with pytest.warns(UserWarning, match="suspect_flat"):
        labels = _labels(df, reference=reference)
    assert labels == ["observed", "source_conflict"] + ["suspect_flat"] * 9 + [
        "observed"
    ]


def test_default_thresholds_come_from_config() -> None:
    out = label_quality(_frame([1.0, 2.5]))
    assert out["quality"].cast(pl.String).to_list() == ["observed", "suspect_jump"]


PERIOD = (date(2009, 1, 1), date(2024, 12, 31))


@pytest.fixture(scope="module")
def labelled(primary: pl.DataFrame) -> pl.DataFrame:
    period = primary.filter(pl.col("ds").is_between(*PERIOD))
    with pytest.warns(UserWarning, match="suspect_flat"):
        return label_quality(period, thresholds=THRESHOLDS)


def test_real_palua_missing(labelled: pl.DataFrame) -> None:
    missing = labelled.filter(pl.col("quality") == "missing")
    assert missing["unique_id"].unique().to_list() == ["palua"]
    assert missing.height == 3


def test_real_no_source_conflict_without_reference(labelled: pl.DataFrame) -> None:
    # The 1930-2025 reference workbook no longer exists: no conflicts are computed.
    assert labelled.filter(pl.col("quality") == "source_conflict").height == 0


def test_find_flat_runs_detects_run_of_min_days() -> None:
    values = pl.Series([1.0, 2.0] + [3.0] * 10 + [4.0])
    assert find_flat_runs(values, 10).to_list() == [False, False] + [True] * 10 + [
        False
    ]


def test_find_flat_runs_ignores_changing_and_short_runs() -> None:
    values = pl.Series([1.0, 1.0, 1.0, 2.0, None, None, None, 3.0])
    assert not find_flat_runs(values, 3)[3:].any()
    assert find_flat_runs(pl.Series([float(i) for i in range(20)]), 2).sum() == 0


FLAT_RUNS = {
    "ayacucho": [(date(2019, 3, 7), 12)],
    "caicara": [],
    "ciudad_bolivar": [(date(2020, 3, 24), 19)],
    "palua": [],
}


@pytest.mark.parametrize("uid", sorted(FLAT_RUNS))
def test_find_flat_runs_real_data(primary: pl.DataFrame, uid: str) -> None:
    series = primary.filter(
        (pl.col("unique_id") == uid) & pl.col("ds").is_between(*PERIOD)
    )
    flagged = series.filter(find_flat_runs(series["y"], 10))["ds"].to_list()
    expected = [
        start + timedelta(days=i) for start, n in FLAT_RUNS[uid] for i in range(n)
    ]
    assert flagged == expected
