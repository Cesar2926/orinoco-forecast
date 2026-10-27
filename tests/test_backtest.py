"""Walk-forward protocol, multi-fold backtest and cross-fold comparison."""

import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from orinoco_forecast.data.splits import WalkForwardSplit
from orinoco_forecast.evaluation.backtest import (
    BacktestResult,
    build_splits,
    run_backtest,
)
from orinoco_forecast.models.registry import get_runner
from orinoco_forecast.models.statistical.baselines import (
    NaiveRunner,
    SeasonalNaiveRunner,
)

STATIONS = ("ayacucho", "caicara", "ciudad_bolivar", "palua")
TARGETS = ("ciudad_bolivar", "palua")
START = date(2019, 1, 1)
END = date(2025, 3, 31)
CUTOFF = date(2022, 6, 1)
HORIZON = 3


def _protocol_frame(
    *,
    start: date = START,
    end: date = END,
    stations: tuple[str, ...] = STATIONS,
) -> pl.DataFrame:
    dates = pl.date_range(start, end, "1d", eager=True)
    index = pl.int_range(pl.len()).cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": station, "ds": dates}).with_columns(
            y=50.0 + offset + 0.01 * index,
        )
        for offset, station in enumerate(STATIONS)
        if station in stations
    )


def _config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "test_start": "2023-01-01",
        "n_windows": 3,
        "step_size": 30,
    }
    config.update(overrides)
    return config


def test_build_splits_matches_the_protocol() -> None:
    splits = build_splits(_protocol_frame(), _config(), horizon=14)

    assert [split.cutoff for split in splits] == [
        date(2022, 10, 18),
        date(2022, 11, 17),
        date(2022, 12, 17),
    ]
    assert splits[-1].validation_start == date(2022, 12, 18)
    assert splits[-1].validation_end == date(2022, 12, 31)


def test_build_splits_rejects_a_missing_protocol_key() -> None:
    config = _config()
    del config["n_windows"]

    with pytest.raises(ValueError, match="n_windows"):
        build_splits(_protocol_frame(), config, horizon=14)


def test_build_splits_rejects_data_that_ends_before_the_default_cutoff() -> None:
    frame = _protocol_frame(end=date(2022, 12, 1))

    with pytest.raises(ValueError, match="no coincide"):
        build_splits(frame, _config(), horizon=14)


def test_build_splits_rejects_an_incomplete_target_coverage() -> None:
    frame = _protocol_frame().filter(
        ~(
            (pl.col("unique_id") == "palua")
            & (pl.col("ds") >= date(2022, 11, 17))
            & (pl.col("ds") <= date(2022, 11, 20))
        )
    )

    with pytest.raises(ValueError, match="palua"):
        build_splits(frame, _config(), horizon=14)


def test_backtest_module_imports_first_without_a_cycle() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}

    completed = subprocess.run(
        [sys.executable, "-c", "import orinoco_forecast.evaluation.backtest"],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_build_splits_keeps_the_step_clock_when_the_horizon_changes() -> None:
    splits = build_splits(_protocol_frame(), _config(), horizon=7)

    assert splits[-1].cutoff == date(2022, 12, 24)
    assert splits[-1].validation_end == date(2022, 12, 31)
    assert (splits[-1].cutoff - splits[-2].cutoff).days == 30


def test_station_dates_are_read_from_a_single_series() -> None:
    splits = build_splits(_protocol_frame(), _config(), horizon=14)
    first = splits[0]

    assert first.validation_start == first.cutoff + timedelta(days=1)


def _frame(days: int = 800) -> pl.DataFrame:
    start = CUTOFF - timedelta(days=days - 30)
    dates = pl.date_range(start, CUTOFF + timedelta(days=30), "1d", eager=True)
    index = pl.int_range(pl.len()).cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": station, "ds": dates}).with_columns(
            y=50.0 + offset + 0.01 * index,
            rain_ciudad_bolivar=1.0 + index % 7,
            rain_palua=2.0 + index % 5,
            oni=0.1,
        )
        for offset, station in enumerate(STATIONS)
    )


def _splits() -> tuple[WalkForwardSplit, ...]:
    return tuple(
        WalkForwardSplit(
            cutoff=cutoff,
            validation_start=cutoff + timedelta(days=1),
            validation_end=cutoff + timedelta(days=HORIZON),
        )
        for cutoff in (CUTOFF - timedelta(days=20), CUTOFF)
    )


class _Recorder:
    """Runner proxy that counts how many times it was fitted."""

    implemented = True

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.name = inner.name  # type: ignore[attr-defined]
        self.fits = 0

    def fit(
        self,
        windows: pl.DataFrame,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = 14,
    ) -> "_Recorder":
        self.fits += 1
        self.inner.fit(windows, frame, cutoff=cutoff, horizon=horizon)
        return self

    def predict(
        self, frame: pl.DataFrame, *, cutoff: date, horizon: int = 14
    ) -> object:
        return self.inner.predict(frame, cutoff=cutoff, horizon=horizon)


def _result() -> BacktestResult:
    return run_backtest(
        "naive",
        "seasonal_naive",
        _frame(),
        _splits(),
        horizon=HORIZON,
        seasonality=365,
    )


def test_run_backtest_scores_both_runners_on_the_same_folds() -> None:
    result = _result()

    assert result.model == "naive"
    assert result.against == "seasonal_naive"
    assert result.horizon == HORIZON
    assert [split.cutoff for split in result.splits] == [
        CUTOFF - timedelta(days=20),
        CUTOFF,
    ]
    assert result.predictions.columns == [
        "fold",
        "model",
        "unique_id",
        "cutoff",
        "ds",
        "horizon_step",
        "y_pred",
        "y_true",
    ]
    assert result.predictions["fold"].unique().sort().to_list() == [1, 2]
    assert result.predictions["y_true"].null_count() == 0
    counts = result.predictions.group_by("fold", "unique_id").len()
    assert set(counts["len"]) == {HORIZON}
    assert set(result.statistics["fold"]) == {1, 2}
    assert set(result.statistics["model"]) == {"naive", "seasonal_naive"}
    assert set(result.statistics["against"]) == {"seasonal_naive"}


def test_run_backtest_reports_fold_statistics_on_the_held_out_data() -> None:
    result = _result()
    scored = result.statistics.filter(pl.col("horizon") == "all")

    assert set(scored["n"]) == {HORIZON}
    assert set(scored["unique_id"]) == set(TARGETS)
    assert set(scored["cutoff"]) == {CUTOFF - timedelta(days=20), CUTOFF}


def test_fold_rel_mae_matches_the_mae_ratio() -> None:
    statistics = _result().statistics
    baseline = statistics.filter(pl.col("model") == pl.col("against")).select(
        "fold", "unique_id", "horizon", mae_baseline="mae"
    )
    model = statistics.filter(pl.col("model") != pl.col("against")).join(
        baseline, on=["fold", "unique_id", "horizon"], how="inner"
    )

    assert set(statistics.filter(pl.col("model") == pl.col("against"))["rel_mae"]) == {
        1.0
    }
    ratio = model["mae"] / model["mae_baseline"]
    assert ((model["rel_mae"] - ratio).abs() < 1e-12).all()


def test_run_backtest_fits_a_fresh_runner_per_fold() -> None:
    created: list[_Recorder] = []

    def factory(name: str) -> _Recorder:
        recorder = _Recorder(get_runner(name))
        created.append(recorder)
        return recorder

    run_backtest(
        "naive",
        "seasonal_naive",
        _frame(),
        _splits(),
        horizon=HORIZON,
        runner_factory=factory,
    )

    fitted = [recorder for recorder in created if recorder.fits]
    assert len(fitted) == 2 * len(_splits())
    assert {recorder.fits for recorder in fitted} == {1}


def test_run_backtest_is_invariant_to_future_data() -> None:
    frame = _frame()
    last_validation_end = _splits()[-1].validation_end
    truncated = frame.filter(pl.col("ds") <= last_validation_end)

    full = run_backtest("naive", "seasonal_naive", frame, _splits(), horizon=HORIZON)
    cut = run_backtest("naive", "seasonal_naive", truncated, _splits(), horizon=HORIZON)

    assert_frame_equal(full.predictions, cut.predictions)
    assert_frame_equal(full.statistics, cut.statistics)


def test_run_backtest_rejects_an_incomplete_fold() -> None:
    splits = (
        *_splits(),
        WalkForwardSplit(
            cutoff=CUTOFF + timedelta(days=29),
            validation_start=CUTOFF + timedelta(days=30),
            validation_end=CUTOFF + timedelta(days=30 + HORIZON),
        ),
    )

    with pytest.raises(ValueError, match="segmento de validación"):
        run_backtest("naive", "seasonal_naive", _frame(), splits, horizon=HORIZON)


class _Stub:
    name = "stub"
    implemented = False
    fits = 0

    def fit(self, *args: object, **kwargs: object) -> "_Stub":
        type(self).fits += 1
        raise AssertionError("un runner de estructura no debe ajustarse")

    def predict(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("un runner de estructura no debe predecir")


def test_run_backtest_rejects_structure_only_runners_before_the_loop() -> None:
    _Stub.fits = 0

    with pytest.raises(ValueError, match="estructura"):
        run_backtest(
            "stub",
            "naive",
            _frame(),
            _splits(),
            horizon=HORIZON,
            runner_factory=lambda name: _Stub() if name == "stub" else NaiveRunner(),
        )

    assert _Stub.fits == 0


def test_run_backtest_rejects_a_repeated_comparison_runner() -> None:
    with pytest.raises(ValueError, match="distintos"):
        run_backtest(
            "naive",
            "naive",
            _frame(),
            _splits(),
            horizon=HORIZON,
            runner_factory=lambda name: SeasonalNaiveRunner(seasonality=365),
        )


def test_backtest_result_rejects_a_missing_fold() -> None:
    result = _result()

    with pytest.raises(ValueError, match="fold"):
        replace(result, predictions=result.predictions.filter(pl.col("fold") == 1))


def test_backtest_result_rejects_predictions_beyond_the_last_fold() -> None:
    result = _result()
    last = result.splits[-1].validation_end
    target = (pl.col("fold") == 2) & (pl.col("horizon_step") == HORIZON)
    beyond = result.predictions.with_columns(
        pl.when(target)
        .then(pl.lit(last + timedelta(days=1), dtype=pl.Date))
        .otherwise(pl.col("ds"))
        .alias("ds"),
        pl.when(target)
        .then(HORIZON + 1)
        .otherwise(pl.col("horizon_step"))
        .cast(pl.Int32)
        .alias("horizon_step"),
    )

    with pytest.raises(ValueError, match="validación"):
        replace(result, predictions=beyond)
