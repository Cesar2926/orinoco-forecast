"""Backtest protocol: development split, cutoffs and run_cv with a fake runner."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.evaluation.backtest import (
    cutoffs,
    development_frame,
    load_backtest_config,
    run_cv,
    stage_baseline_metrics,
    stage_dir,
)
from orinoco_forecast.features.dataset import assert_no_imputation_near_cutoffs

CFG = load_backtest_config()
TEST_START = date.fromisoformat(CFG["test_start"])


def test_development_frame_excludes_test_block(model_frame_a: pl.DataFrame) -> None:
    dev = development_frame(model_frame_a, CFG)
    assert dev["ds"].max() == TEST_START - timedelta(days=1)
    assert dev.height < model_frame_a.height


def test_cutoffs_follow_protocol(model_frame_a: pl.DataFrame) -> None:
    dev = development_frame(model_frame_a, CFG)
    cuts = cutoffs(dev, CFG)
    assert len(cuts) == CFG["n_windows"] == 24
    assert {(b - a).days for a, b in zip(cuts, cuts[1:], strict=False)} == {30}
    assert cuts[-1] == dev["ds"].max() - timedelta(days=CFG["horizon"])
    assert cuts[-1] + timedelta(days=CFG["horizon"]) < TEST_START
    # Real data: no interpolated values next to any cutoff.
    assert_no_imputation_near_cutoffs(dev, cuts, CFG["horizon"])


class LastValueRunner:
    """Fake runner: repeats the value at each cutoff; records what it was given."""

    name = "fake"
    model_columns = ["Last"]

    def __init__(self) -> None:
        self.seen_max: date | None = None
        self.seen_cfg: dict | None = None

    def cross_validation(self, frame: pl.DataFrame, cfg: dict) -> pl.DataFrame:
        self.seen_max, self.seen_cfg = frame["ds"].max(), cfg
        cuts = pl.DataFrame({"cutoff": cutoffs(frame, cfg)})
        h = pl.int_ranges(1, cfg["horizon"] + 1)
        grid = cuts.join(frame.select("unique_id").unique(), how="cross")
        grid = grid.with_columns(h=h).explode("h")
        grid = grid.with_columns(ds=pl.col("cutoff") + pl.duration(days=pl.col("h")))
        last = frame.select("unique_id", pl.col("ds").alias("cutoff"), Last="y")
        actual = frame.select("unique_id", "ds", "y")
        return (
            grid.join(last, on=["unique_id", "cutoff"])
            .join(actual, on=["unique_id", "ds"])
            .select("unique_id", "ds", "cutoff", "y", "Last")
        )


def _synthetic() -> pl.DataFrame:
    days = pl.date_range(date(2019, 1, 1), date(2023, 3, 31), "1d", eager=True)
    return pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": days}).with_columns(
            y=pl.int_range(pl.len()).cast(pl.Float64).sin() + k, imputed=False
        )
        for k, uid in enumerate(["a", "b"])
    )


def test_run_cv_writes_outputs_without_test_block(tmp_path: Path) -> None:
    runner = LastValueRunner()
    cv, metrics = run_cv("fake", runner, _synthetic(), CFG, output_root=tmp_path)
    assert runner.seen_max < TEST_START
    assert cv.columns == ["unique_id", "ds", "cutoff", "y", "Last"]
    assert cv["cutoff"].n_unique() == CFG["n_windows"]
    assert cv["ds"].max() < TEST_START
    out = tmp_path / "fake" / "stage_A"
    assert (out / "cv.parquet").is_file()
    assert (out / "metrics.parquet").is_file()
    assert set(metrics["horizon"]) == {*map(str, range(1, 15)), "all"}
    assert not (out / "final_test").exists()
    assert not (tmp_path / "fake" / "cv.parquet").exists()


def test_run_cv_final_test_only_when_requested(tmp_path: Path) -> None:
    runner = LastValueRunner()
    cv, _ = run_cv(
        "fake", runner, _synthetic(), CFG, final_test=True, output_root=tmp_path
    )
    h = CFG["horizon"]
    assert cv["cutoff"].min() == TEST_START - timedelta(days=1)
    assert cv["ds"].min() == TEST_START
    assert runner.seen_cfg["step_size"] == h
    out = tmp_path / "fake" / "stage_A"
    assert (out / "final_test" / "cv.parquet").is_file()
    assert not (out / "cv.parquet").exists()


def test_run_cv_refuses_imputed_rows_near_cutoff(tmp_path: Path) -> None:
    frame = _synthetic()
    dev_end = TEST_START - timedelta(days=1)
    frame = frame.with_columns(imputed=pl.col("ds") == dev_end)
    with pytest.raises(ValueError, match="imputad"):
        run_cv("fake", LastValueRunner(), frame, CFG, output_root=tmp_path)


def test_run_cv_stages_do_not_overwrite_each_other(tmp_path: Path) -> None:
    frame = _synthetic()
    run_cv("fake", LastValueRunner(), frame, CFG, stage="A", output_root=tmp_path)
    run_cv("fake", LastValueRunner(), frame, CFG, stage="B", output_root=tmp_path)
    for stage in ("A", "B"):
        out = stage_dir(tmp_path, "fake", stage)
        assert out == tmp_path / "fake" / f"stage_{stage}"
        assert (out / "cv.parquet").is_file()
        assert (out / "metrics.parquet").is_file()


def test_stage_dir_rejects_unknown_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Etapa"):
        stage_dir(tmp_path, "fake", "C")


def test_stage_baseline_metrics_requires_same_stage(tmp_path: Path) -> None:
    run_cv(
        "baselines",
        LastValueRunner(),
        _synthetic(),
        CFG,
        stage="A",
        output_root=tmp_path,
    )
    base = stage_baseline_metrics(tmp_path, "A")
    assert set(base["model"]) == {"Last"}
    # Stage A baselines (or legacy root-level files) never stand in for stage B.
    (tmp_path / "baselines" / "metrics.parquet").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="etapa B"):
        stage_baseline_metrics(tmp_path, "B")
    with pytest.raises(FileNotFoundError, match="test final"):
        stage_baseline_metrics(tmp_path, "A", final_test=True)
