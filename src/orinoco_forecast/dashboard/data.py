"""Read-only data shaping and in-memory live runner helpers for the dashboard.

Reads ``data/predictions/<runner>/{predictions,statistics}.parquet`` (and
``backtest/naive/{fold_predictions,fold_statistics,comparison}.parquet`` when
present) written by ``scripts/run_model.py`` and returns tidy Polars frames.
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from orinoco_forecast.data.datasets import DEFAULT_OUTPUT
from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.quality import QUALITY_LABELS
from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.evaluation.execution import (
    FEATURE_COLUMNS,
    INPUT_LENGTH,
    TARGET_STATIONS,
    UPSTREAM_STATIONS,
    load_history,
)
from orinoco_forecast.models.registry import available_runners, get_runner

PREDICTIONS_ENV = "ORINOCO_PREDICTIONS_DIR"
BASELINES = "naive"
FAST_RUNNERS = ("booster", "arimax", "naive", "seasonal_naive")
HISTORY_DAYS = 60
TABLE_METRICS = ("mae", "rel_mae", "rmse", "mase", "nse", "kge")
PREDICTIONS_FILE = "predictions.parquet"
STATISTICS_FILE = "statistics.parquet"
FOLD_PREDICTIONS_FILE = Path("backtest") / BASELINES / "fold_predictions.parquet"
FOLD_STATISTICS_FILE = Path("backtest") / BASELINES / "fold_statistics.parquet"
FOLD_COMPARISON_FILE = Path("backtest") / BASELINES / "comparison.parquet"
ENSEMBLE_LABEL = "Ensamble (seleccionados)"
_SEED = r"^(.+)_s\d+$"


def predictions_root() -> Path:
    """Carpeta de predicciones; usa ``reports/predictions`` como respaldo."""
    override = os.environ.get(PREDICTIONS_ENV)
    if override:
        return Path(override)
    default = repo_root() / "data" / "predictions"
    if discover_models(default):
        return default
    fallback = repo_root() / "reports" / "predictions"
    if discover_models(fallback):
        return fallback
    return default


def run_command(model: str = "<modelo>", against: str = BASELINES) -> str:
    """Comando que genera las predicciones de ``model`` contra ``against``."""
    return f"uv run python scripts/run_model.py {model} --against {against}"


def discover_models(root: Path) -> list[str]:
    """Runners con ``predictions.parquet`` o pliegues de backtest; baseline primero."""
    if not root.is_dir():
        return []
    found = [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and (
            (path / PREDICTIONS_FILE).is_file()
            or (path / FOLD_PREDICTIONS_FILE).is_file()
        )
    ]
    return sorted(found, key=lambda name: (name != BASELINES, name))


def _family() -> pl.Expr:
    return pl.coalesce(pl.col("model").str.extract(_SEED), pl.col("model"))


def _belongs(folder: str) -> pl.Expr:
    """Rows whose model column belongs to the runner folder (seed aware)."""
    model = pl.col("model").str.to_lowercase()
    return (model == folder) | model.str.starts_with(f"{folder}_")


def load_metrics(root: Path, models: list[str]) -> pl.DataFrame:
    """Métricas de todos los runners, sin duplicar filas de baseline.

    Cada ``statistics.parquet`` contiene el modelo ejecutado y su baseline. Se
    concatenan todos y se conserva la fila propia cuando existe; ``rel_mae`` es
    el reportado por la corrida (columna ``against``).
    """
    frames = []
    for model in models:
        path = root / model / STATISTICS_FILE
        if not path.is_file():
            warnings.warn(
                f"{model}: falta {STATISTICS_FILE}; se omite de las métricas.",
                stacklevel=2,
            )
            continue
        metrics = pl.read_parquet(path)
        if "rel_mae" not in metrics.columns:
            metrics = metrics.with_columns(rel_mae=pl.lit(None, dtype=pl.Float64))
        frames.append(metrics.with_columns(source=pl.lit(model)))
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="diagonal_relaxed")
    own = (pl.col("model").str.to_lowercase() == pl.col("source")) | pl.col(
        "model"
    ).str.to_lowercase().str.starts_with(pl.col("source") + "_")
    return (
        combined.with_columns(_own=own)
        .sort("_own", descending=True)
        .unique(subset=["unique_id", "model", "horizon"], keep="first")
        .drop("_own")
        .with_columns(
            pl.col("mae", "rmse", "mase", "nse", "kge", "rel_mae").cast(pl.Float64),
        )
        .with_columns(family=_family())
        .select(
            "source",
            "unique_id",
            "model",
            "family",
            "horizon",
            "n",
            *TABLE_METRICS,
            "against",
        )
    )


def load_fold_comparison(root: Path, models: list[str]) -> pl.DataFrame:
    """Resumen walk-forward (24 cortes) de ``backtest/naive/comparison.parquet``."""
    frames = []
    for model in models:
        path = root / model / FOLD_COMPARISON_FILE
        if not path.is_file():
            continue
        frames.append(pl.read_parquet(path).with_columns(source=pl.lit(model)))
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="diagonal_relaxed")
    keys = [
        c
        for c in ("unique_id", "model", "against", "horizon", "metric")
        if c in combined.columns
    ]
    out = combined.unique(subset=keys, keep="first", maintain_order=True).with_columns(
        family=_family()
    )
    if {"mean_model", "mean_baseline"} <= set(out.columns):
        out = out.with_columns(rel_mae=pl.col("mean_model") / pl.col("mean_baseline"))
    return out.sort("unique_id", "model")


def load_fold_timeline(root: Path, models: list[str], unique_id: str) -> pl.DataFrame:
    """MAE por corte walk-forward (``horizon == 'all'``) para una estación."""
    frames = []
    for model in models:
        path = root / model / FOLD_STATISTICS_FILE
        if not path.is_file():
            continue
        frames.append(pl.read_parquet(path).with_columns(source=pl.lit(model)))
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="diagonal_relaxed").filter(
        (pl.col("unique_id") == unique_id) & (pl.col("horizon") == "all")
    )
    if combined.is_empty():
        return pl.DataFrame()
    own = (pl.col("model").str.to_lowercase() == pl.col("source")) | pl.col(
        "model"
    ).str.to_lowercase().str.starts_with(pl.col("source") + "_")
    return (
        combined.with_columns(_own=own)
        .sort("_own", descending=True)
        .unique(subset=["unique_id", "cutoff", "model"], keep="first")
        .drop("_own")
        .with_columns(family=_family())
        .select("unique_id", "cutoff", "model", "family", "mae", "rmse", "rel_mae")
        .sort("model", "cutoff")
    )


def seed_stats(metrics: pl.DataFrame) -> pl.DataFrame:
    """Media y desviación (muestral) entre semillas, h=1..14, por estación."""
    seeded = metrics.filter(
        (pl.col("horizon") == "all") & pl.col("model").str.contains(_SEED)
    )
    aggs = [pl.len().alias("seeds")]
    for metric in TABLE_METRICS:
        aggs += [pl.col(metric).mean(), pl.col(metric).std().alias(f"{metric}_std")]
    return (
        seeded.group_by("unique_id", "family")
        .agg(aggs)
        .rename({"family": "model"})
        .sort("unique_id", "model")
    )


def selection_table(metrics: pl.DataFrame) -> pl.DataFrame:
    """Criterio del README: ``rel_mae < 1`` en cada estación y luego MAE medio."""
    return (
        metrics.filter(pl.col("horizon") == "all")
        .group_by("model")
        .agg(
            stations=pl.col("unique_id").n_unique(),
            beats_naive=pl.when(pl.col("rel_mae").null_count() == 0).then(
                (pl.col("rel_mae") < 1).all()
            ),
            mean_mae=pl.col("mae").mean(),
        )
        .sort(
            pl.col("beats_naive").fill_null(False),
            "mean_mae",
            descending=[True, False],
        )
    )


def fold_station_mae(
    fold_comparison: pl.DataFrame, horizon: str = "all"
) -> pl.DataFrame:
    """MAE medio de los 24 cortes por modelo (semillas promediadas) y estación.

    Naive se añade como fila propia a partir de ``mean_baseline``.
    """
    if fold_comparison.is_empty():
        return pl.DataFrame()
    mae = fold_comparison.filter(
        (pl.col("metric") == "mae") & (pl.col("horizon") == horizon)
    )
    if mae.is_empty():
        return pl.DataFrame()
    models = mae.group_by("source", "unique_id").agg(
        mae=pl.col("mean_model").mean(),
        mae_std_seeds=pl.col("mean_model").std(),
        naive=pl.col("mean_baseline").first(),
        win_rate=pl.col("win_rate").mean(),
        p_min=pl.col("sign_test_p").min(),
        p_max=pl.col("sign_test_p").max(),
    )
    naive = models.group_by("unique_id").agg(pl.col("naive").first().alias("mae"))
    naive = naive.with_columns(source=pl.lit(BASELINES), naive=pl.col("mae"))
    return (
        pl.concat([models, naive], how="diagonal_relaxed")
        .with_columns(rel_mae=pl.col("mae") / pl.col("naive"))
        .rename({"source": "model"})
        .sort("unique_id", "mae")
    )


def fold_ranking(fold_comparison: pl.DataFrame) -> pl.DataFrame:
    """Criterio del README sobre los 24 cortes: ``rel_mae < 1`` y menor MAE medio."""
    per_station = fold_station_mae(fold_comparison)
    if per_station.is_empty():
        return pl.DataFrame()
    stations = sorted(per_station["unique_id"].unique())
    wide = per_station.pivot(on="unique_id", index="model", values="mae").rename(
        {s: f"mae_{s}" for s in stations}
    )
    ranking = (
        per_station.group_by("model")
        .agg(
            beats_naive=(pl.col("rel_mae") < 1).all() & (pl.len() == len(stations)),
            mean_mae=pl.col("mae").mean(),
        )
        .join(wide, on="model")
        .with_columns(
            pl.when(pl.col("model") == BASELINES)
            .then(None)
            .otherwise(pl.col("beats_naive"))
            .alias("beats_naive")
        )
        .sort(
            pl.col("beats_naive").fill_null(False), "mean_mae", descending=[True, False]
        )
    )
    return ranking.with_row_index("rank", offset=1).select(
        "rank", "model", "beats_naive", "mean_mae", *(f"mae_{s}" for s in stations)
    )


def fold_horizon_errors(fold_comparison: pl.DataFrame, unique_id: str) -> pl.DataFrame:
    """MAE medio de los 24 cortes por día de horizonte (1..14) de una estación."""
    rows = fold_comparison.filter(
        (pl.col("unique_id") == unique_id) & (pl.col("horizon") != "all")
    )
    if rows.is_empty():
        return pl.DataFrame()
    return (
        pl.concat(
            fold_station_mae(rows, h).with_columns(horizon=pl.lit(int(h)))
            for h in rows["horizon"].unique()
        )
        .select("model", "horizon", "mae", "rel_mae")
        .sort("model", "horizon")
    )


def horizon_errors(metrics: pl.DataFrame, unique_id: str) -> pl.DataFrame:
    """MAE por día de horizonte (1..14) de una estación, en formato largo."""
    return (
        metrics.filter(
            (pl.col("unique_id") == unique_id) & (pl.col("horizon") != "all")
        )
        .with_columns(pl.col("horizon").cast(pl.Int64))
        .select("model", "family", "horizon", "mae", "rel_mae")
        .sort("model", "horizon")
    )


def _seed_means(forecasts: pl.DataFrame, levels: pl.DataFrame) -> pl.DataFrame:
    seeded = forecasts.filter(pl.col("model").str.contains(_SEED))
    if seeded.is_empty():
        return forecasts
    means = (
        seeded.with_columns(family=pl.col("model").str.extract(_SEED))
        .group_by("unique_id", "cutoff", "ds", "family")
        .agg(y_hat=pl.col("y_hat").mean())
        .join(
            levels.select("unique_id", "ds", "y"),
            on=["unique_id", "ds"],
            how="left",
        )
        .with_columns(model=pl.col("family") + " (media de semillas)")
        .select("unique_id", "ds", "cutoff", "model", "y_hat", "y")
    )
    return pl.concat([forecasts, means])


def add_ensemble(
    forecasts: pl.DataFrame,
    models: list[str],
    levels: pl.DataFrame,
    label: str = ENSEMBLE_LABEL,
) -> pl.DataFrame:
    """Agrega en memoria el promedio de los modelos seleccionados (sin ``naive``)."""
    chosen = [m for m in models if m != BASELINES]
    subset = forecasts.filter(pl.col("model").is_in(chosen))
    if subset.is_empty():
        return forecasts
    ens = (
        subset.group_by("unique_id", "cutoff", "ds")
        .agg(y_hat=pl.col("y_hat").mean())
        .join(
            levels.select("unique_id", "ds", "y"),
            on=["unique_id", "ds"],
            how="left",
        )
        .with_columns(model=pl.lit(label))
        .select(forecasts.columns)
    )
    return pl.concat([forecasts, ens])


def _ensure_naive(raw: pl.DataFrame, levels: pl.DataFrame) -> pl.DataFrame:
    """Sintetiza ``naive`` en los cortes donde no exista todavía."""
    has_naive = (
        raw.filter(pl.col("model") == BASELINES)
        .select("unique_id", "cutoff", "ds")
        .unique()
    )
    missing_grid = (
        raw.select("unique_id", "cutoff", "ds")
        .unique()
        .join(has_naive, on=["unique_id", "cutoff", "ds"], how="anti")
    )
    if missing_grid.is_empty():
        return raw
    observed = (
        levels.filter(pl.col("y").is_not_null())
        .select("unique_id", "ds", pl.col("y").alias("y_hat"))
        .sort("unique_id", "ds")
    )
    cutoffs = (
        missing_grid.select("unique_id", "cutoff")
        .unique()
        .sort("unique_id", "cutoff")
        .join_asof(
            observed,
            left_on="cutoff",
            right_on="ds",
            by="unique_id",
            strategy="backward",
            check_sortedness=False,
        )
        .select("unique_id", "cutoff", "y_hat")
    )
    synth = (
        missing_grid.join(cutoffs, on=["unique_id", "cutoff"], how="left")
        .with_columns(model=pl.lit(BASELINES))
        .select("unique_id", "ds", "cutoff", "model", "y_hat")
    )
    return pl.concat([raw, synth])


def load_forecasts(root: Path, models: list[str], levels: pl.DataFrame) -> pl.DataFrame:
    """Predicciones en formato largo con el nivel observado (``y``).

    Lee ``predictions.parquet`` y ``backtest/naive/fold_predictions.parquet``,
    deduplica, completa ``naive`` en todos los cortes y agrega la media entre
    semillas (``"<familia> (media de semillas)"``) solo para visualizar.
    """
    frames = []
    for model in models:
        for rel in (PREDICTIONS_FILE, FOLD_PREDICTIONS_FILE):
            path = root / model / rel
            if not path.is_file():
                continue
            predictions = pl.read_parquet(path)
            frames.append(
                predictions.select(
                    "unique_id",
                    "ds",
                    "cutoff",
                    pl.col("model"),
                    pl.col("y_pred").alias("y_hat"),
                )
            )
    if not frames:
        return pl.DataFrame()
    raw = pl.concat(frames).unique(
        subset=["unique_id", "ds", "cutoff", "model"],
        keep="first",
        maintain_order=True,
    )
    raw = _ensure_naive(raw, levels)
    forecasts = raw.join(
        levels.select("unique_id", "ds", "y"),
        on=["unique_id", "ds"],
        how="left",
    )
    return _seed_means(forecasts, levels)


def live_runners() -> list[str]:
    """Runners implementados para predicción en vivo (los rápidos primero)."""
    implemented = [
        name
        for name in available_runners()
        if getattr(get_runner(name), "implemented", True)
    ]
    order = {name: i for i, name in enumerate(FAST_RUNNERS)}
    return sorted(implemented, key=lambda name: (order.get(name, len(order)), name))


def apply_what_if(
    frame: pl.DataFrame,
    unique_id: str,
    cutoff: date,
    level_delta_m: float = 0.0,
    upstream_delta_m: float = 0.0,
    rain_delta_mm: float = 0.0,
    days: int = 7,
) -> pl.DataFrame:
    """Perturba en memoria los últimos ``days`` días hasta ``cutoff``."""
    start = cutoff - timedelta(days=days)
    in_window = (pl.col("ds") > start) & (pl.col("ds") <= cutoff)
    ramp = (pl.col("ds") - start).dt.total_days().cast(pl.Float64) / max(1, days)
    exprs: list[pl.Expr] = []
    if level_delta_m != 0.0:
        exprs.append(
            pl.when(in_window & (pl.col("unique_id") == unique_id))
            .then(pl.col("y") + ramp * level_delta_m)
            .otherwise(pl.col("y"))
            .alias("y")
        )
    if upstream_delta_m != 0.0:
        up_cols = [
            c
            for c in frame.columns
            if c in UPSTREAM_STATIONS
            or c.startswith("level_")
            or any(c.startswith(f"{u}_lag_") for u in UPSTREAM_STATIONS)
        ]
        for col in up_cols:
            exprs.append(
                pl.when(in_window)
                .then(pl.col(col) + upstream_delta_m)
                .otherwise(pl.col(col))
                .alias(col)
            )
    if rain_delta_mm != 0.0:
        rain_cols = [c for c in frame.columns if c.startswith("rain_")]
        for col in rain_cols:
            exprs.append(
                pl.when(in_window)
                .then((pl.col(col) + rain_delta_mm).clip(lower_bound=0.0))
                .otherwise(pl.col(col))
                .alias(col)
            )
    return frame.clone().with_columns(*exprs) if exprs else frame.clone()


def _extract_predictions(pred: object, target_ids: list[str]) -> pl.DataFrame:
    if hasattr(pred, "frame"):
        raw = pred.frame.select(
            "unique_id",
            "ds",
            "cutoff",
            pl.col("model"),
            pl.col("y_pred").alias("y_hat"),
        )
    else:
        wide = pred.predictions
        cols = [c for c in wide.columns if c not in ("unique_id", "ds", "cutoff")]
        raw = wide.unpivot(
            on=cols,
            index=["unique_id", "ds", "cutoff"],
            variable_name="model",
            value_name="y_hat",
        )
    return raw.filter(pl.col("unique_id").is_in(target_ids))


def _run_one_live(
    frame: pl.DataFrame,
    model_name: str,
    cutoff: date,
    horizon: int,
    *,
    fast_mode: bool = True,
    what_if_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    runner = get_runner(model_name)
    if (
        fast_mode
        and hasattr(runner, "params")
        and isinstance(runner.params, dict)
        and runner.params.get("seeds")
    ):
        runner.params = {**runner.params, "seeds": [runner.params["seeds"][0]]}
    history = frame.filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")
    if (
        model_name == "arimax"
        and not any("_lag_" in c for c in history.columns)
        and DEFAULT_OUTPUT.is_file()
    ):
        history = load_history().filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")
    present_ids = set(history["unique_id"].unique().to_list())
    target_ids = [u for u in TARGET_STATIONS if u in present_ids] or sorted(present_ids)
    targets = history.filter(pl.col("unique_id").is_in(target_ids))
    feature_cols = [c for c in FEATURE_COLUMNS if c in targets.columns] or ["y"]
    max_len = max(1, targets.height // max(1, len(target_ids)) - horizon - 1)
    windows = build_windows(
        targets,
        cutoff=cutoff,
        input_length=min(INPUT_LENGTH, max_len),
        horizon=horizon,
        feature_columns=feature_cols,
    )
    runner.fit(windows, history, cutoff=cutoff, horizon=horizon)
    base_out = _extract_predictions(
        runner.predict(history, cutoff=cutoff, horizon=horizon), target_ids
    )
    if what_if_frame is None:
        return base_out
    wi_history = what_if_frame.filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")
    wi_out = _extract_predictions(
        runner.predict(wi_history, cutoff=cutoff, horizon=horizon), target_ids
    ).with_columns(model=pl.col("model") + " (simulado)")
    return pl.concat([base_out, wi_out])


def run_live_forecast(
    frame: pl.DataFrame,
    levels: pl.DataFrame,
    model_name: str,
    cutoff: date,
    horizon: int = 14,
    *,
    fast_mode: bool = True,
    what_if_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Ejecuta ``model_name`` (y ``naive`` como referencia) en ``cutoff``."""
    parts = [
        _run_one_live(
            frame,
            model_name,
            cutoff,
            horizon,
            fast_mode=fast_mode,
            what_if_frame=what_if_frame,
        )
    ]
    if model_name != BASELINES:
        parts.append(
            _run_one_live(
                frame,
                BASELINES,
                cutoff,
                horizon,
                fast_mode=fast_mode,
                what_if_frame=None,
            )
        )
    forecasts = (
        pl.concat(parts)
        .unique(
            subset=["unique_id", "ds", "cutoff", "model"],
            keep="first",
            maintain_order=True,
        )
        .join(
            levels.select("unique_id", "ds", "y"),
            on=["unique_id", "ds"],
            how="left",
        )
    )
    return _seed_means(forecasts, levels)


def forecast_window(
    levels: pl.DataFrame,
    forecasts: pl.DataFrame,
    unique_id: str,
    cutoff: date,
    models: list[str],
    horizon: int = 14,
    history_days: int = HISTORY_DAYS,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Historia hasta el corte, lo observado después y los pronósticos."""
    station = levels.filter(pl.col("unique_id") == unique_id)
    start = cutoff - timedelta(days=history_days - 1)
    end = cutoff + timedelta(days=horizon)
    history = station.filter(pl.col("ds").is_between(start, cutoff))
    after = station.filter((pl.col("ds") > cutoff) & (pl.col("ds") <= end))
    preds = forecasts.filter(
        (pl.col("unique_id") == unique_id)
        & (pl.col("cutoff") == cutoff)
        & pl.col("model").is_in(models)
    ).sort("model", "ds")
    return history, after, preds


def filter_levels(
    levels: pl.DataFrame, stations: list[str], start: date, end: date
) -> pl.DataFrame:
    """Niveles de las estaciones elegidas dentro del rango de fechas."""
    return levels.filter(
        pl.col("unique_id").is_in(stations) & pl.col("ds").is_between(start, end)
    )


def quality_counts(levels: pl.DataFrame) -> pl.DataFrame:
    """Días por etiqueta de calidad y estación (columnas en orden de etiquetas)."""
    counts = (
        levels.with_columns(pl.col("quality").cast(pl.String))
        .group_by("unique_id", "quality")
        .len()
        .pivot(on="quality", index="unique_id", values="len")
    )
    missing = [label for label in QUALITY_LABELS if label not in counts.columns]
    counts = counts.with_columns(pl.lit(0, dtype=pl.UInt32).alias(m) for m in missing)
    return (
        counts.select("unique_id", *QUALITY_LABELS)
        .fill_null(0)
        .with_columns(total=pl.sum_horizontal(QUALITY_LABELS))
        .sort("unique_id")
    )


def load_optional(path: Path) -> pl.DataFrame | None:
    """Lee un parquet opcional; ``None`` si todavía no se generó."""
    return pl.read_parquet(path) if path.is_file() else None


def rain_long(exogenous: pl.DataFrame) -> pl.DataFrame:
    """Lluvia diaria NASA POWER por estación (``ds, station, rain``)."""
    rain_cols = [c for c in exogenous.columns if c.startswith("rain_")]
    return (
        exogenous.select("ds", *rain_cols)
        .unique("ds")
        .unpivot(on=rain_cols, index="ds", variable_name="station", value_name="rain")
        .with_columns(pl.col("station").str.replace("^rain_", ""))
        .sort("station", "ds")
    )


def oni_series(exogenous: pl.DataFrame) -> pl.DataFrame:
    """Serie diaria del ONI (ya con el retraso de publicación aplicado)."""
    return exogenous.select("ds", "oni").unique("ds").sort("ds")
