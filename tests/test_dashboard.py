"""Dashboard data shaping on the runner output layout plus an AppTest smoke."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.dashboard import charts, data
from orinoco_forecast.data.loading import get_source, repo_root

H = 14
CUTOFF = date(2021, 2, 9)
EARLIER_CUTOFF = date(2021, 1, 26)
STATIONS = ["ciudad_bolivar", "palua"]
RAW = repo_root() / get_source("orinoco_stations")["path"]
APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
METRICS = ("mae", "rmse", "mase", "nse", "kge")


def _levels(days_before: int = 60) -> pl.DataFrame:
    dates = pl.date_range(
        CUTOFF - timedelta(days=days_before),
        CUTOFF + timedelta(days=H),
        "1d",
        eager=True,
    )
    return pl.concat(
        pl.DataFrame({"unique_id": station, "ds": dates}).with_columns(
            y=5.0 + offset + pl.int_range(pl.len()).cast(pl.Float64)
        )
        for offset, station in enumerate(STATIONS)
    )


def _predictions(
    models: dict[str, float], cutoffs: tuple[date, ...] = (CUTOFF,)
) -> pl.DataFrame:
    levels = _levels()
    rows = []
    for cut in cutoffs:
        future = levels.filter(
            pl.col("ds").is_between(cut + timedelta(days=1), cut + timedelta(days=H))
        )
        for model, bias in models.items():
            for row in future.iter_rows(named=True):
                rows.append(
                    {
                        "model": model,
                        "unique_id": row["unique_id"],
                        "cutoff": cut,
                        "ds": row["ds"],
                        "horizon_step": (row["ds"] - cut).days,
                        "y_pred": row["y"] + bias,
                    }
                )
    return pl.DataFrame(rows)


def _statistics(models: dict[str, float], against: str = "naive") -> pl.DataFrame:
    horizons = [str(h) for h in range(1, H + 1)] + ["all"]
    rows = []
    for station in STATIONS:
        for model, bias in models.items():
            for horizon in horizons:
                mae = abs(bias) + 0.1
                rows.append(
                    {
                        "unique_id": station,
                        "model": model,
                        "horizon": horizon,
                        "n": H if horizon == "all" else 1,
                        "mae": mae,
                        "rmse": mae + 0.1,
                        "mase": 1.0,
                        "nse": 0.9,
                        "kge": 0.9,
                        "rel_mae": mae / (abs(models.get(against, 0.0)) + 0.1),
                        "against": against,
                    }
                )
    return pl.DataFrame(rows)


def _write(
    root: Path,
    name: str,
    models: dict[str, float],
    *,
    stats_models: dict[str, float] | None = None,
    against: str = "naive",
    predictions: bool = True,
    statistics: bool = True,
) -> None:
    out = root / name
    out.mkdir(parents=True)
    if predictions:
        _predictions(models).write_parquet(out / data.PREDICTIONS_FILE)
    if statistics:
        _statistics(stats_models or models, against).write_parquet(
            out / data.STATISTICS_FILE
        )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(tmp_path, "naive", {"naive": 1.0}, against="naive")
    _write(
        tmp_path,
        "arimax",
        {"arimax": 0.5},
        stats_models={"arimax": 0.5, "naive": 1.0},
        against="naive",
    )
    _write(
        tmp_path,
        "tcn",
        {"TCN_s1": 0.2, "TCN_s2": 0.4, "TCN_s3": 0.6},
        stats_models={"TCN_s1": 0.2, "TCN_s2": 0.4, "TCN_s3": 0.6, "naive": 1.0},
        against="naive",
    )
    (tmp_path / "empty").mkdir()
    return tmp_path


def test_discover_models_requires_predictions(root: Path) -> None:
    assert data.discover_models(root) == ["naive", "arimax", "tcn"]
    assert data.discover_models(root / "missing") == []


def test_load_metrics_deduplicates_and_adds_family(root: Path) -> None:
    metrics = data.load_metrics(root, data.discover_models(root))
    assert {"rel_mae", "family", "source", "against"} <= set(metrics.columns)
    xgb = metrics.filter(
        (pl.col("model") == "arimax")
        & (pl.col("horizon") == "all")
        & (pl.col("unique_id") == "palua")
    ).row(0, named=True)
    assert xgb["family"] == "arimax"
    assert xgb["rel_mae"] == pytest.approx(0.6 / 1.1)
    seeds = metrics.filter(pl.col("model") == "TCN_s2")
    assert set(seeds["family"]) == {"TCN"}


def test_load_metrics_without_statistics_warns(root: Path) -> None:
    _write(
        root,
        "booster",
        {"booster": 0.3},
        predictions=True,
        statistics=False,
    )
    with pytest.warns(UserWarning, match="statistics.parquet"):
        metrics = data.load_metrics(root, ["booster"])
    assert metrics.is_empty()


def test_seed_stats_and_selection_table(root: Path) -> None:
    metrics = data.load_metrics(root, data.discover_models(root))
    stats = data.seed_stats(metrics)
    row = stats.filter(pl.col("unique_id") == "palua").row(0, named=True)
    assert row["model"] == "TCN"
    assert row["seeds"] == 3
    assert row["mae"] == pytest.approx(0.5)
    assert row["mae_std"] == pytest.approx(0.2)
    table = data.selection_table(metrics)
    assert table["model"][0] == "TCN_s1"


def test_horizon_errors_are_numeric_and_sorted(root: Path) -> None:
    metrics = data.load_metrics(root, data.discover_models(root))
    errors = data.horizon_errors(metrics, "palua")
    assert errors["horizon"].dtype == pl.Int64
    assert set(errors["horizon"]) == set(range(1, H + 1))
    assert set(errors["model"]) >= {"naive", "arimax", "TCN_s1"}


def test_load_forecasts_adds_seed_means_and_observed_levels(root: Path) -> None:
    preds = data.load_forecasts(root, data.discover_models(root), _levels())
    assert set(preds.columns) == {"unique_id", "ds", "cutoff", "y", "model", "y_hat"}
    mean = preds.filter(pl.col("model") == "TCN (media de semillas)")
    assert mean.height == len(STATIONS) * H
    first = mean.sort("unique_id", "ds").row(0, named=True)
    assert first["y_hat"] == pytest.approx(first["y"] + 0.4)
    assert {"TCN_s1", "TCN_s2", "TCN_s3", "naive"} <= set(preds["model"])


def test_load_forecasts_reads_fold_predictions_and_synthesizes_naive(
    root: Path,
) -> None:
    levels = _levels()
    fold_dir = root / "arimax" / "backtest" / "naive"
    fold_dir.mkdir(parents=True)
    fold_preds = _predictions({"arimax": 0.5}, cutoffs=(EARLIER_CUTOFF, CUTOFF))
    fold_preds.write_parquet(fold_dir / "fold_predictions.parquet")

    preds = data.load_forecasts(root, data.discover_models(root), levels)
    cutoffs = sorted(preds["cutoff"].unique().to_list())
    assert cutoffs == [EARLIER_CUTOFF, CUTOFF]

    naive_earlier = preds.filter(
        (pl.col("model") == "naive")
        & (pl.col("cutoff") == EARLIER_CUTOFF)
        & (pl.col("unique_id") == "palua")
    )
    assert naive_earlier.height == H
    expected_y = levels.filter(
        (pl.col("unique_id") == "palua") & (pl.col("ds") == EARLIER_CUTOFF)
    )["y"].item()
    assert naive_earlier["y_hat"].to_list() == [pytest.approx(expected_y)] * H


def test_load_fold_comparison_and_timeline(root: Path) -> None:
    comp_dir = root / "arimax" / "backtest" / "naive"
    comp_dir.mkdir(parents=True)
    comp = pl.DataFrame(
        [
            {
                "unique_id": station,
                "model": "arimax",
                "against": "naive",
                "horizon": "all",
                "metric": "mae",
                "n_folds": 24,
                "n_improvement": 24,
                "mean_model": 0.30,
                "std_model": 0.10,
                "mean_baseline": 0.50,
                "std_baseline": 0.15,
                "mean_diff": -0.20,
                "std_diff": 0.12,
                "mean_improvement_pct": 40.0,
                "std_improvement_pct": 20.0,
                "min_improvement_pct": -10.0,
                "max_improvement_pct": 80.0,
                "wins": 20,
                "ties": 0,
                "losses": 4,
                "win_rate": 20 / 24,
                "sign_test_p": 0.0015,
            }
            for station in STATIONS
        ]
    )
    comp.write_parquet(comp_dir / "comparison.parquet")

    fold_stats = pl.concat(
        [
            _statistics({"arimax": 0.5, "naive": 1.0}).with_columns(
                fold=pl.lit(idx, dtype=pl.Int32),
                cutoff=pl.lit(cut, dtype=pl.Date),
            )
            for idx, cut in enumerate((EARLIER_CUTOFF, CUTOFF), start=1)
        ]
    )
    fold_stats.write_parquet(comp_dir / "fold_statistics.parquet")

    loaded = data.load_fold_comparison(root, data.discover_models(root))
    ranking = data.fold_ranking(loaded)
    assert ranking["model"].to_list() == ["arimax", "naive"]
    assert ranking["beats_naive"].to_list() == [True, None]
    assert ranking["mean_mae"].to_list() == pytest.approx([0.30, 0.50])
    assert charts.ranking_chart(data.fold_station_mae(loaded)) is not None
    assert loaded.height == len(STATIONS)
    assert {"family", "rel_mae", "source"} <= set(loaded.columns)
    assert loaded["rel_mae"][0] == pytest.approx(0.30 / 0.50)

    timeline = data.load_fold_timeline(root, data.discover_models(root), "palua")
    assert set(timeline["model"]) == {"arimax", "naive"}
    assert set(timeline["cutoff"]) == {EARLIER_CUTOFF, CUTOFF}
    chart = charts.fold_timeline_chart(timeline)
    assert chart is not None


def test_add_ensemble_averages_selected_non_naive_models(root: Path) -> None:
    levels = _levels()
    preds = data.load_forecasts(root, data.discover_models(root), levels)
    with_ens = data.add_ensemble(preds, ["arimax", "TCN_s1", "naive"], levels)
    ens = with_ens.filter(
        (pl.col("model") == data.ENSEMBLE_LABEL) & (pl.col("unique_id") == "palua")
    ).sort("ds")
    assert ens.height == H
    first = ens.row(0, named=True)
    # arimax has bias 0.5 and TCN_s1 has bias 0.2 -> ensemble bias is 0.35
    assert first["y_hat"] == pytest.approx(first["y"] + 0.35)


def test_live_runners_and_what_if_forecast() -> None:
    runners = data.live_runners()
    assert "naive" in runners
    assert "seasonal_naive" in runners
    assert runners[0] == "booster"

    levels = _levels(days_before=400).with_columns(
        ayacucho=pl.col("y") + 1.0,
        rain_palua=pl.lit(10.0),
    )
    wi = data.apply_what_if(
        levels,
        "palua",
        CUTOFF,
        level_delta_m=1.4,
        upstream_delta_m=0.5,
        rain_delta_mm=-20.0,
        days=7,
    )
    # Original frame is untouched in memory
    orig_cut = levels.filter(
        (pl.col("unique_id") == "palua") & (pl.col("ds") == CUTOFF)
    )["y"].item()
    wi_cut = wi.filter((pl.col("unique_id") == "palua") & (pl.col("ds") == CUTOFF))[
        "y"
    ].item()
    assert wi_cut == pytest.approx(orig_cut + 1.4)
    assert wi["rain_palua"].min() >= 0.0

    preds = data.run_live_forecast(
        levels,
        levels,
        "seasonal_naive",
        CUTOFF,
        horizon=H,
        fast_mode=True,
        what_if_frame=wi,
    )
    assert set(preds.columns) == {"unique_id", "ds", "cutoff", "y", "model", "y_hat"}
    assert set(preds["model"]) == {
        "seasonal_naive",
        "seasonal_naive (simulado)",
        "naive",
    }
    assert preds.height == 3 * len(STATIONS) * H


def test_forecast_window_aligns_history_and_horizon(root: Path) -> None:
    levels = _levels()
    preds = data.load_forecasts(root, data.discover_models(root), levels)
    history, after, forecasts = data.forecast_window(
        levels, preds, "palua", CUTOFF, ["arimax", "naive"]
    )
    assert history.height == data.HISTORY_DAYS
    assert history["ds"].max() == CUTOFF
    assert history["ds"].min() == CUTOFF - timedelta(days=data.HISTORY_DAYS - 1)
    assert after.height == H
    assert after["ds"].min() == CUTOFF + timedelta(days=1)
    assert after["ds"].max() == CUTOFF + timedelta(days=H)
    assert set(forecasts["model"]) == {"arimax", "naive"}
    for _, part in forecasts.group_by("model"):
        assert part["ds"].min() == CUTOFF + timedelta(days=1)
        assert part.height == H


def test_run_command_uses_the_runner_cli() -> None:
    assert data.run_command("arimax") == (
        "uv run python scripts/run_model.py arimax --against naive"
    )


def _app(monkeypatch: pytest.MonkeyPatch, root: Path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(data.PREDICTIONS_ENV, str(root))
    return AppTest.from_file(str(APP), default_timeout=120).run()


@pytest.mark.skipif(not RAW.is_file(), reason=f"{RAW.name} missing")
def test_app_runs_with_runner_outputs(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    at = _app(monkeypatch, root)
    assert not at.exception
    assert len(at.tabs) == 5
    models = at.multiselect(key="h_models").options
    assert "arimax" in models and "TCN_s1" in models


@pytest.mark.skipif(not RAW.is_file(), reason=f"{RAW.name} missing")
def test_app_runs_without_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert any("predicciones" in str(info.value) for info in at.info)
    commands = " ".join(str(code.value) for code in at.code)
    assert "uv run python scripts/run_model.py naive" in commands
