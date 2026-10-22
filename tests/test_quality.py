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


@pytest.fixture(scope="module")
def labelled(primary: pl.DataFrame, historical: pl.DataFrame) -> pl.DataFrame:
    with pytest.warns(UserWarning, match="suspect_flat"):
        return label_quality(primary, reference=historical, thresholds=THRESHOLDS)


def _count(df: pl.DataFrame, uid: str, label: str, years: tuple[int, int]) -> int:
    return df.filter(
        (pl.col("unique_id") == uid)
        & (pl.col("quality") == label)
        & pl.col("ds").dt.year().is_between(*years)
    ).height


def test_real_palua_missing(labelled: pl.DataFrame) -> None:
    assert _count(labelled, "palua", "missing", (1974, 2025)) == 17


def test_real_ciudad_bolivar_conflicts(labelled: pl.DataFrame) -> None:
    assert _count(labelled, "ciudad_bolivar", "source_conflict", (1987, 1987)) >= 150
    assert _count(labelled, "ciudad_bolivar", "source_conflict", (1997, 1997)) >= 300
    recent = _count(labelled, "ciudad_bolivar", "source_conflict", (1998, 2025))
    n_recent = labelled.filter(
        (pl.col("unique_id") == "ciudad_bolivar")
        & pl.col("ds").dt.year().is_between(1998, 2025)
    ).height
    assert recent / n_recent < 0.01


def test_real_conflicts_only_on_reference_station(labelled: pl.DataFrame) -> None:
    others = labelled.filter(pl.col("unique_id") != "ciudad_bolivar")
    assert others.filter(pl.col("quality") == "source_conflict").height == 0


def test_find_flat_runs_detects_run_of_min_days() -> None:
    values = pl.Series([1.0, 2.0] + [3.0] * 10 + [4.0])
    assert find_flat_runs(values, 10).to_list() == [False, False] + [True] * 10 + [
        False
    ]


def test_find_flat_runs_ignores_changing_and_short_runs() -> None:
    values = pl.Series([1.0, 1.0, 1.0, 2.0, None, None, None, 3.0])
    assert not find_flat_runs(values, 3)[3:].any()
    assert find_flat_runs(pl.Series([float(i) for i in range(20)]), 2).sum() == 0


@pytest.mark.parametrize(
    ("uid", "start", "length"),
    [("ciudad_bolivar", date(2020, 3, 25), 19), ("palua", date(1996, 4, 8), 16)],
)
def test_find_flat_runs_real_data(
    primary: pl.DataFrame, uid: str, start: date, length: int
) -> None:
    series = primary.filter(pl.col("unique_id") == uid)
    mask = find_flat_runs(series["y"], 10)
    flagged = series.filter(mask)["ds"]
    run = [start + timedelta(days=i) for i in range(length)]
    assert set(run) <= set(flagged.to_list())
    assert start - timedelta(days=1) not in flagged.to_list()
