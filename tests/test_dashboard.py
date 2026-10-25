"""Dashboard data shaping on synthetic predictions plus an AppTest smoke run."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.dashboard import data

H = 14
CUTOFFS = [date(2021, 1, 10), date(2021, 2, 9)]
STATIONS = ["ciudad_bolivar", "palua"]
APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"


def _cv(models: dict[str, float]) -> pl.DataFrame:
    rows = []
    for uid in STATIONS:
        for cutoff in CUTOFFS:
            for step in range(1, H + 1):
                ds = cutoff + timedelta(days=step)
                row = {"unique_id": uid, "ds": ds, "cutoff": cutoff, "y": 5.0 + step}
                row |= {m: 5.0 + step + bias for m, bias in models.items()}
                rows.append(row)
    return pl.DataFrame(rows)


def _metrics(models: dict[str, float]) -> pl.DataFrame:
    horizons = [str(h) for h in range(1, H + 1)] + ["all"]
    rows = [
        {
            "unique_id": uid,
            "model": m,
            "horizon": h,
            "n": 24 * (H if h == "all" else 1),
            "mae": abs(bias) + 0.1,
            "rmse": abs(bias) + 0.2,
            "mase": 1.0,
            "nse": 0.9,
            "kge": 0.9,
        }
        for uid in STATIONS
        for m, bias in models.items()
        for h in horizons
    ]
    return pl.DataFrame(rows)


def _write(root: Path, name: str, models: dict[str, float]) -> None:
    out = root / name
    out.mkdir(parents=True)
    _cv(models).write_parquet(out / "cv.parquet")
    _metrics(models).write_parquet(out / "metrics.parquet")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(tmp_path, "baselines", {"Naive": 1.0, "SeasonalNaive": 2.0})
    _write(tmp_path, "tcn", {"TCN_s1": 0.2, "TCN_s2": 0.4, "TCN_s3": 0.6})
    _write(tmp_path, "xgboost", {"XGBoost": 0.5})
    (tmp_path / "lstm").mkdir()  # Empty folder: no predictions yet.
    return tmp_path


def test_discover_models_skips_folders_without_predictions(root: Path) -> None:
    assert data.discover_models(root) == ["baselines", "tcn", "xgboost"]
    assert data.discover_models(root / "missing") == []


def test_load_metrics_adds_rel_mae_and_family(root: Path) -> None:
    metrics = data.load_metrics(root, data.discover_models(root))
    assert {"rel_mae", "family", "source"} <= set(metrics.columns)
    row = metrics.filter(
        (pl.col("model") == "XGBoost")
        & (pl.col("horizon") == "all")
        & (pl.col("unique_id") == "palua")
    ).row(0, named=True)
    assert row["rel_mae"] == pytest.approx(0.6 / 1.1)
    assert row["family"] == "XGBoost"
    seeds = metrics.filter(pl.col("model") == "TCN_s2")
    assert set(seeds["family"]) == {"TCN"}


def test_load_metrics_without_baselines_leaves_rel_mae_null(tmp_path: Path) -> None:
    _write(tmp_path, "xgboost", {"XGBoost": 0.5})
    metrics = data.load_metrics(tmp_path, ["xgboost"])
    assert metrics["rel_mae"].null_count() == metrics.height


def test_seed_stats_reports_mean_and_std_per_family(root: Path) -> None:
    metrics = data.load_metrics(root, data.discover_models(root))
    stats = data.seed_stats(metrics)
    row = stats.filter(pl.col("unique_id") == "palua").row(0, named=True)
    assert row["model"] == "TCN"
    assert row["seeds"] == 3
    assert row["mae"] == pytest.approx(0.5)
    assert row["mae_std"] == pytest.approx(0.2)


def test_selection_table_ranks_models_beating_naive_first(root: Path) -> None:
    metrics = data.load_metrics(root, data.discover_models(root))
    table = data.selection_table(metrics)
    assert table["model"][0] == "TCN_s1"
    naive = table.filter(pl.col("model") == "Naive").row(0, named=True)
    assert naive["beats_naive"] is False
    assert naive["stations"] == 2


def test_horizon_errors_are_numeric_and_sorted(root: Path) -> None:
    metrics = data.load_metrics(root, data.discover_models(root))
    errors = data.horizon_errors(metrics, "palua")
    assert errors["horizon"].dtype == pl.Int64
    assert set(errors["horizon"]) == set(range(1, H + 1))
    assert "all" not in errors.columns
    assert set(errors["model"]) >= {"Naive", "XGBoost", "TCN_s1"}


def test_load_forecasts_adds_seed_mean_for_visualisation(root: Path) -> None:
    preds = data.load_forecasts(root, data.discover_models(root))
    assert set(preds.columns) == {"unique_id", "ds", "cutoff", "y", "model", "y_hat"}
    mean = preds.filter(pl.col("model") == "TCN (media de semillas)")
    assert mean.height == len(STATIONS) * len(CUTOFFS) * H
    first = mean.sort("unique_id", "cutoff", "ds").row(0, named=True)
    assert first["y_hat"] == pytest.approx(first["y"] + 0.4)
    assert {"TCN_s1", "TCN_s2", "TCN_s3", "Naive"} <= set(preds["model"])


def test_forecast_window_aligns_history_and_horizon(root: Path) -> None:
    days = pl.date_range(date(2020, 10, 1), date(2021, 3, 31), eager=True)
    levels = pl.DataFrame(
        {"unique_id": "palua", "ds": days, "y": range(days.len())}
    ).with_columns(pl.col("y").cast(pl.Float64))
    preds = data.load_forecasts(root, data.discover_models(root))
    cutoff = CUTOFFS[1]
    history, after, forecasts = data.forecast_window(
        levels, preds, "palua", cutoff, ["XGBoost", "Naive"]
    )
    assert history.height == data.HISTORY_DAYS
    assert history["ds"].max() == cutoff
    assert history["ds"].min() == cutoff - timedelta(days=data.HISTORY_DAYS - 1)
    assert after.height == H
    assert after["ds"].min() == cutoff + timedelta(days=1)
    assert after["ds"].max() == cutoff + timedelta(days=H)
    assert set(forecasts["model"]) == {"XGBoost", "Naive"}
    for _, part in forecasts.group_by("model"):
        assert part["ds"].min() == cutoff + timedelta(days=1)
        assert part.height == H


def test_quality_counts_one_row_per_station() -> None:
    levels = pl.DataFrame(
        {
            "unique_id": ["a", "a", "a", "b"],
            "ds": [date(2020, 1, d) for d in (1, 2, 3, 1)],
            "y": [1.0, None, 1.0, 2.0],
            "quality": ["observed", "missing", "suspect_jump", "observed"],
        }
    )
    counts = data.quality_counts(levels)
    assert counts["unique_id"].to_list() == ["a", "b"]
    assert counts.filter(pl.col("unique_id") == "a")["missing"].item() == 1
    assert counts.filter(pl.col("unique_id") == "b")["observed"].item() == 1


def _app(monkeypatch: pytest.MonkeyPatch, root: Path):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv(data.PREDICTIONS_ENV, str(root))
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_app_runs_against_repository_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = data.predictions_root()
    needed = {"baselines", "xgboost", "tcn"}
    if not needed <= set(data.discover_models(root)):
        pytest.skip("Predicciones no generadas (data/predictions no se versiona).")
    at = _app(monkeypatch, root)
    assert not at.exception
    assert len(at.tabs) == 4
    # Levels, error by horizon, forecast window and XGBoost importance.
    assert len(at.get("vega_lite_chart")) == 4
    cutoff = at.selectbox(key="w_cutoff")
    cutoff.set_value(cutoff.options[0]).run()
    assert not at.exception


def test_app_runs_without_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert any("predicciones" in str(info.value) for info in at.info)
    commands = " ".join(str(code.value) for code in at.code)
    assert "uv run python scripts/run_model.py baselines" in commands
