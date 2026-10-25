"""Read-only data shaping for the dashboard (no training, no final test).

Reads ``data/predictions/<model>/{cv,metrics}.parquet`` written by
``scripts/run_model.py`` and returns tidy Polars frames for the charts.
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.quality import QUALITY_LABELS
from orinoco_forecast.evaluation.backtest import CV_KEYS
from orinoco_forecast.evaluation.metrics import relative_to_baseline

PREDICTIONS_ENV = "ORINOCO_PREDICTIONS_DIR"
BASELINES = "baselines"
HISTORY_DAYS = 60
SEED_MEAN_LABEL = "{} (media de semillas)"
TABLE_METRICS = ("mae", "rel_mae", "rmse", "mase", "nse", "kge")
_SEED = r"^(.+)_s\d+$"


def predictions_root() -> Path:
    """Carpeta de predicciones; ``ORINOCO_PREDICTIONS_DIR`` la sobrescribe."""
    override = os.environ.get(PREDICTIONS_ENV)
    return Path(override) if override else repo_root() / "data" / "predictions"


def run_command(model: str = "<modelo>") -> str:
    """Comando que genera las predicciones de validación de ``model``."""
    return f"uv run python scripts/run_model.py {model}"


def discover_models(root: Path) -> list[str]:
    """Modelos con ``cv.parquet`` y ``metrics.parquet``; los baselines primero."""
    if not root.is_dir():
        return []
    found = [
        p.name
        for p in root.iterdir()
        if (p / "cv.parquet").is_file() and (p / "metrics.parquet").is_file()
    ]
    return sorted(found, key=lambda name: (name != BASELINES, name))


def _family() -> pl.Expr:
    return pl.coalesce(pl.col("model").str.extract(_SEED), pl.col("model"))


def load_metrics(root: Path, models: list[str]) -> pl.DataFrame:
    """Métricas de todos los modelos, con ``rel_mae`` (MAE / MAE de Naive).

    ``rel_mae`` queda nulo si no hay baselines o si las ventanas no coinciden
    (en ese caso se emite un warning, sin corregir nada).
    """
    baseline = None
    if BASELINES in models:
        baseline = pl.read_parquet(root / BASELINES / "metrics.parquet")
    frames = []
    for model in models:
        metrics = pl.read_parquet(root / model / "metrics.parquet")
        rel = pl.lit(None, dtype=pl.Float64).alias("rel_mae")
        if baseline is not None:
            try:
                metrics = relative_to_baseline(metrics, baseline).drop("mae_naive")
                rel = pl.col("rel_mae")
            except ValueError as err:
                warnings.warn(f"{model}: sin rel_mae ({err})", stacklevel=2)
        frames.append(
            metrics.with_columns(
                rel.cast(pl.Float64),
                pl.col("mae", "rmse", "mase", "nse", "kge").cast(pl.Float64),
                source=pl.lit(model),
            ).select("source", "unique_id", "model", "horizon", "n", *TABLE_METRICS)
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).with_columns(family=_family())


def seed_stats(metrics: pl.DataFrame) -> pl.DataFrame:
    """Media y desviación (muestral) entre semillas, h=1..14, por estación."""
    seeded = metrics.filter(
        (pl.col("horizon") == "all") & pl.col("model").str.contains(_SEED)
    )
    aggs = [pl.len().alias("seeds")]
    for m in TABLE_METRICS:
        aggs += [pl.col(m).mean(), pl.col(m).std().alias(f"{m}_std")]
    return (
        seeded.group_by("unique_id", "family")
        .agg(aggs)
        .rename({"family": "model"})
        .sort("unique_id", "model")
    )


def selection_table(metrics: pl.DataFrame) -> pl.DataFrame:
    """Criterio del README: ``rel_mae < 1`` en cada estación y luego MAE medio.

    ``beats_naive`` es nulo si falta ``rel_mae`` (sin baselines).
    """
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
            pl.col("beats_naive").fill_null(False), "mean_mae", descending=[True, False]
        )
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


def _with_seed_means(cv: pl.DataFrame) -> pl.DataFrame:
    seeds: dict[str, list[str]] = {}
    for col in cv.columns:
        match = pl.Series([col]).str.extract(_SEED).item()
        if match is not None:
            seeds.setdefault(match, []).append(col)
    means = [
        pl.mean_horizontal(cols).alias(SEED_MEAN_LABEL.format(family))
        for family, cols in seeds.items()
        if len(cols) > 1
    ]
    return cv.with_columns(means) if means else cv


def load_forecasts(root: Path, models: list[str]) -> pl.DataFrame:
    """Predicciones de validación en formato largo (``model``, ``y_hat``).

    Las semillas se conservan y además se agrega su predicción media
    (``"<familia> (media de semillas)"``) solo para visualizar.
    """
    frames = []
    for model in models:
        cv = _with_seed_means(pl.read_parquet(root / model / "cv.parquet"))
        value_cols = [c for c in cv.columns if c not in CV_KEYS]
        frames.append(
            cv.with_columns(pl.col(value_cols).cast(pl.Float64)).unpivot(
                on=value_cols, index=CV_KEYS, variable_name="model", value_name="y_hat"
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames)


def forecast_window(
    levels: pl.DataFrame,
    forecasts: pl.DataFrame,
    unique_id: str,
    cutoff: date,
    models: list[str],
    horizon: int = 14,
    history_days: int = HISTORY_DAYS,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Historia (``history_days`` hasta el corte), lo observado después y pronósticos.

    El corte es el último día conocido por el modelo; los pronósticos empiezan
    en ``cutoff + 1``.
    """
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
