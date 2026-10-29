"""Unit tests for scripts/make_report.py: ranking, cutoffs, and window evaluation."""

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_report.py"
spec = importlib.util.spec_from_file_location("make_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

STATIONS = ("ciudad_bolivar", "palua")
TEST_START = date(2023, 1, 1)


def _synthetic_frame(
    start: date = date(2021, 1, 1), end: date = date(2023, 4, 30)
) -> pl.DataFrame:
    dates = pl.date_range(start, end, "1d", eager=True)
    idx = pl.int_range(pl.len()).cast(pl.Float64)
    return pl.concat(
        pl.DataFrame({"unique_id": uid, "ds": dates}).with_columns(
            y=10.0 + k + (idx / 20.0).sin(),
            imputed=False,
        )
        for k, uid in enumerate(STATIONS)
    )


def test_rank_models_aggregates_seeds_and_sorts_by_beats_naive() -> None:
    rows = [
        # CNNLSTM seeds beat naive on both stations (mean MAE = 0.25)
        {
            "unique_id": "ciudad_bolivar",
            "model": "CNNLSTM_s1",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.20,
            "mean_baseline": 0.50,
        },
        {
            "unique_id": "ciudad_bolivar",
            "model": "CNNLSTM_s2",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.24,
            "mean_baseline": 0.50,
        },
        {
            "unique_id": "palua",
            "model": "CNNLSTM_s1",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.26,
            "mean_baseline": 0.48,
        },
        {
            "unique_id": "palua",
            "model": "CNNLSTM_s2",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.30,
            "mean_baseline": 0.48,
        },
        # Booster has slightly higher mean MAE (0.31) but also beats naive on both
        {
            "unique_id": "ciudad_bolivar",
            "model": "booster",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.28,
            "mean_baseline": 0.50,
        },
        {
            "unique_id": "palua",
            "model": "booster",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.34,
            "mean_baseline": 0.48,
        },
        # Faulty model has low MAE on one station but loses to naive on palua
        {
            "unique_id": "ciudad_bolivar",
            "model": "overfit",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.05,
            "mean_baseline": 0.50,
        },
        {
            "unique_id": "palua",
            "model": "overfit",
            "horizon": "all",
            "metric": "mae",
            "mean_model": 0.55,
            "mean_baseline": 0.48,
        },
    ]
    ranked = report.rank_models(pl.DataFrame(rows))
    assert ranked["model"].to_list() == ["CNNLSTM", "booster", "overfit"]
    top = ranked.row(0, named=True)
    assert top["beats_naive_all"] is True
    assert top["mae_ciudad_bolivar"] == pytest.approx(0.22)
    assert top["mae_palua"] == pytest.approx(0.28)
    assert top["mean_mae"] == pytest.approx(0.25)
    assert ranked.row(2, named=True)["beats_naive_all"] is False


def test_build_eval_cutoffs_val_and_test() -> None:
    dates = pl.date_range(
        date(2021, 1, 1), date(2023, 4, 30), "1d", eager=True
    ).to_list()

    val_cuts = report.build_eval_cutoffs(
        dates,
        mode="val",
        test_start=TEST_START,
        horizon=14,
        n_windows=5,
        step_days=14,
    )
    assert len(val_cuts) == 5
    assert val_cuts[-1] == date(2022, 12, 17)
    assert all(c + timedelta(days=14) < TEST_START for c in val_cuts)
    assert [(b - a).days for a, b in zip(val_cuts, val_cuts[1:], strict=False)] == [
        14,
        14,
        14,
        14,
    ]

    test_cuts = report.build_eval_cutoffs(
        dates,
        mode="test",
        test_start=TEST_START,
        horizon=14,
        n_windows=5,
        step_days=14,
    )
    assert len(test_cuts) == 5
    assert test_cuts[0] == TEST_START
    assert all(c >= TEST_START for c in test_cuts)
    assert [(b - a).days for a, b in zip(test_cuts, test_cuts[1:], strict=False)] == [
        14,
        14,
        14,
        14,
    ]

    all_test = report.build_eval_cutoffs(
        dates,
        mode="test",
        test_start=TEST_START,
        horizon=14,
        n_windows=0,
        step_days=14,
    )
    assert len(all_test) > 5
    assert all_test[-1] <= date(2023, 4, 30) - timedelta(days=14)


def test_evaluate_windows_with_fit_once_and_refit() -> None:
    frame = _synthetic_frame()
    cutoffs = [date(2023, 1, 1), date(2023, 1, 15)]

    station_refit, rank_refit = report.evaluate_windows(
        frame,
        ["seasonal_naive"],
        cutoffs,
        horizon=14,
        fit_once=False,
    )
    station_once, rank_once = report.evaluate_windows(
        frame,
        ["seasonal_naive"],
        cutoffs,
        horizon=14,
        fit_once=True,
    )
    assert set(station_refit["model"]) == {"seasonal_naive", "naive"}
    assert rank_refit["model"].to_list() == ["seasonal_naive"]
    assert station_refit["mae"].to_list() == pytest.approx(
        station_once["mae"].to_list()
    )
    assert rank_refit["mean_mae"].to_list() == pytest.approx(
        rank_once["mean_mae"].to_list()
    )


def test_cli_saved_report_and_eval_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    comp_dir = tmp_path / "booster" / "backtest" / "naive"
    comp_dir.mkdir(parents=True)
    (tmp_path / "booster" / "predictions.parquet").write_bytes(b"")
    pl.DataFrame(
        [
            {
                "unique_id": uid,
                "model": "booster",
                "against": "naive",
                "horizon": "all",
                "metric": "mae",
                "n_folds": 24,
                "mean_model": 0.28,
                "std_model": 0.10,
                "mean_baseline": 0.50,
                "wins": 20,
                "losses": 4,
                "win_rate": 20 / 24,
                "sign_test_p": 0.001,
            }
            for uid in STATIONS
        ]
    ).write_parquet(comp_dir / "comparison.parquet")

    out_csv = tmp_path / "ranking.csv"
    rc = report.main(["--predictions-dir", str(tmp_path), "--output", str(out_csv)])
    assert rc == 0
    assert out_csv.is_file()
    captured = capsys.readouterr().out
    assert "Ganador (24 cortes walk-forward): booster" in captured

    synth = _synthetic_frame()
    monkeypatch.setattr(report, "load_levels", lambda: synth)
    monkeypatch.setattr(
        report, "build_model_frame", lambda stage="A", levels=None: synth
    )
    rc_eval = report.main(
        ["--eval", "test", "--models", "seasonal_naive", "--windows", "2", "--fit-once"]
    )
    assert rc_eval == 0
    eval_out = capsys.readouterr().out
    assert "Ganador (test: 2 ventanas de 14d" in eval_out
