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
    BACKTEST_STATISTICS_COLUMNS,
    BacktestResult,
    build_splits,
    execute_backtest,
    run_backtest,
    summarize_comparison,
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
            rain_ciudad_bolivar=1.0 + index % 7,
            rain_palua=2.0 + index % 5,
            oni=0.1,
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


STATISTICS_SCHEMA = {
    "fold": pl.Int32,
    "cutoff": pl.Date,
    "unique_id": pl.String,
    "model": pl.String,
    "horizon": pl.String,
    "n": pl.Int64,
    "mae": pl.Float64,
    "rmse": pl.Float64,
    "mase": pl.Float64,
    "nse": pl.Float64,
    "kge": pl.Float64,
    "rel_mae": pl.Float64,
    "against": pl.String,
}


def _fold_row(
    fold: int,
    model: str,
    *,
    mae: float,
    rmse: float,
    horizon: str = "all",
    unique_id: str = "ciudad_bolivar",
    against: str = "naive",
) -> dict[str, object]:
    return {
        "fold": fold,
        "cutoff": CUTOFF + timedelta(days=fold),
        "unique_id": unique_id,
        "model": model,
        "horizon": horizon,
        "n": HORIZON,
        "mae": mae,
        "rmse": rmse,
        "mase": 1.0,
        "nse": 0.5,
        "kge": 0.5,
        "rel_mae": 1.0 if model == against else mae / 2.0,
        "against": against,
    }


def _fold_statistics(
    rows: list[tuple[int, str, float, float]],
) -> pl.DataFrame:
    return pl.DataFrame(
        [_fold_row(fold, model, mae=mae, rmse=rmse) for fold, model, mae, rmse in rows],
        schema=STATISTICS_SCHEMA,
    ).select(*BACKTEST_STATISTICS_COLUMNS)


def test_summarize_comparison_reports_mean_and_dispersion() -> None:
    statistics = _fold_statistics(
        [
            (1, "arimax", 1.0, 1.0),
            (1, "naive", 2.0, 3.0),
            (2, "arimax", 3.0, 3.0),
            (2, "naive", 2.0, 3.0),
        ]
    )

    comparison = summarize_comparison(statistics)

    assert set(comparison["metric"]) == {"mae", "rmse"}
    row = comparison.filter(
        (pl.col("metric") == "mae") & (pl.col("horizon") == "all")
    ).row(0, named=True)
    assert row["model"] == "arimax"
    assert row["against"] == "naive"
    assert row["n_folds"] == 2
    assert row["n_improvement"] == 2
    assert row["mean_model"] == pytest.approx(2.0)
    assert row["std_model"] == pytest.approx(2**0.5)
    assert row["mean_baseline"] == pytest.approx(2.0)
    assert row["std_baseline"] == 0.0
    assert row["mean_diff"] == 0.0
    assert row["std_diff"] == pytest.approx(2**0.5)
    assert row["mean_improvement_pct"] == 0.0
    assert row["std_improvement_pct"] == pytest.approx(70.7106781)
    assert row["min_improvement_pct"] == pytest.approx(-50.0)
    assert row["max_improvement_pct"] == pytest.approx(50.0)
    assert (row["wins"], row["ties"], row["losses"]) == (1, 0, 1)
    assert row["win_rate"] == pytest.approx(0.5)
    assert row["sign_test_p"] == pytest.approx(1.0)


def test_summarize_comparison_handles_ties_and_a_zero_baseline() -> None:
    statistics = _fold_statistics(
        [
            (1, "arimax", 0.0, 0.0),
            (1, "naive", 0.0, 0.0),
            (2, "arimax", 1.0, 1.0),
            (2, "naive", 2.0, 2.0),
        ]
    )

    row = (
        summarize_comparison(statistics)
        .filter((pl.col("metric") == "mae") & (pl.col("horizon") == "all"))
        .row(0, named=True)
    )

    assert row["n_improvement"] == 1
    assert row["mean_improvement_pct"] == pytest.approx(50.0)
    assert row["min_improvement_pct"] == pytest.approx(50.0)
    assert row["max_improvement_pct"] == pytest.approx(50.0)
    assert (row["wins"], row["ties"], row["losses"]) == (1, 1, 0)
    assert row["win_rate"] == pytest.approx(0.5)
    assert row["sign_test_p"] == pytest.approx(1.0)


def test_summarize_comparison_rejects_unpaired_folds() -> None:
    statistics = _fold_statistics(
        [
            (1, "arimax", 1.0, 1.0),
            (1, "naive", 2.0, 2.0),
            (2, "arimax", 1.0, 1.0),
        ]
    )

    with pytest.raises(ValueError, match="pliegue"):
        summarize_comparison(statistics)


def test_summarize_comparison_keeps_horizon_steps_and_all_last() -> None:
    rows = [
        _fold_row(1, model, mae=mae, rmse=2.0, horizon=step)
        for step in ("1", "14", "all")
        for model, mae in (("arimax", 1.0), ("naive", 2.0))
    ]
    statistics = pl.DataFrame(rows, schema=STATISTICS_SCHEMA).select(
        *BACKTEST_STATISTICS_COLUMNS
    )

    comparison = summarize_comparison(statistics)

    assert comparison.filter(pl.col("metric") == "mae")["horizon"].to_list() == [
        "1",
        "14",
        "all",
    ]


def test_backtest_result_write_uses_the_backtest_folder(tmp_path: Path) -> None:
    result = _result()
    base = tmp_path / "naive" / "backtest" / "seasonal_naive"

    written = result.write(tmp_path)

    assert written == (
        base / "fold_predictions.parquet",
        base / "fold_statistics.parquet",
        base / "comparison.parquet",
    )
    assert all(path.is_file() for path in written)
    assert not (tmp_path / "naive" / "predictions.parquet").exists()
    assert not (tmp_path / "naive" / "statistics.parquet").exists()
    assert_frame_equal(
        pl.read_parquet(written[2]), summarize_comparison(result.statistics)
    )


def test_execute_backtest_builds_the_configured_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "horizon": HORIZON,
        "input_length": 3,
        "test_start": "2023-01-01",
        "seasonality": 365,
        "n_windows": 3,
        "step_size": 30,
    }
    monkeypatch.setattr(
        "orinoco_forecast.evaluation.backtest.load_backtest_config", lambda: config
    )

    result = execute_backtest(
        "naive", "seasonal_naive", frame=_protocol_frame(), write=False
    )

    assert [split.cutoff for split in result.splits] == [
        date(2022, 10, 29),
        date(2022, 11, 28),
        date(2022, 12, 28),
    ]
    assert result.predictions["fold"].n_unique() == 3
    assert result.predictions["ds"].max() == date(2022, 12, 31)


def test_cli_rejects_folds_together_with_a_cutoff() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_model.py"),
            "naive",
            "--folds",
            "--cutoff",
            "2022-12-17",
        ],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )

    assert completed.returncode == 2
    assert "not allowed with" in completed.stderr
