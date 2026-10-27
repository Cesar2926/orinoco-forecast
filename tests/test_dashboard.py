"""Dashboard data shaping on the runner output layout plus an AppTest smoke."""

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from orinoco_forecast.dashboard import data
from orinoco_forecast.data.loading import repo_root

H = 14
CUTOFF = date(2021, 2, 9)
STATIONS = ["ciudad_bolivar", "palua"]
RAW = repo_root() / "data" / "raw" / "Niveles Orinoco.xlsx"
APP = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
METRICS = ("mae", "rmse", "mase", "nse", "kge")


def _levels() -> pl.DataFrame:
    dates = pl.date_range(
        CUTOFF - timedelta(days=60), CUTOFF + timedelta(days=H), "1d", eager=True
    )
    return pl.concat(
        pl.DataFrame({"unique_id": station, "ds": dates}).with_columns(
            y=5.0 + offset + pl.int_range(pl.len()).cast(pl.Float64)
        )
        for offset, station in enumerate(STATIONS)
    )


def _predictions(models: dict[str, float]) -> pl.DataFrame:
    future = _levels().filter(
        pl.col("ds").is_between(CUTOFF + timedelta(days=1), CUTOFF + timedelta(days=H))
    )
    rows = [
        {
            "model": model,
            "unique_id": row["unique_id"],
            "cutoff": CUTOFF,
            "ds": row["ds"],
            "horizon_step": (row["ds"] - CUTOFF).days,
            "y_pred": row["y"] + bias,
        }
        for model, bias in models.items()
        for row in future.iter_rows(named=True)
    ]
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


@pytest.mark.skipif(not RAW.is_file(), reason="data/raw/Niveles Orinoco.xlsx missing")
def test_app_runs_with_runner_outputs(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    at = _app(monkeypatch, root)
    assert not at.exception
    assert len(at.tabs) == 3
    models = at.multiselect(key="h_models").options
    assert "arimax" in models and "TCN_s1" in models


@pytest.mark.skipif(not RAW.is_file(), reason="data/raw/Niveles Orinoco.xlsx missing")
def test_app_runs_without_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    at = _app(monkeypatch, tmp_path)
    assert not at.exception
    assert any("predicciones" in str(info.value) for info in at.info)
    commands = " ".join(str(code.value) for code in at.code)
    assert "uv run python scripts/run_model.py naive" in commands
