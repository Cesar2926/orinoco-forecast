"""Read-only data shaping for the dashboard (no training, no final test).

Reads ``data/predictions/<runner>/{predictions,statistics}.parquet`` written by
``scripts/run_model.py`` and returns tidy Polars frames for the charts. The
legacy ``stage_<A|B>/{cv,metrics}.parquet`` layout is no longer supported.
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.quality import QUALITY_LABELS

PREDICTIONS_ENV = "ORINOCO_PREDICTIONS_DIR"
BASELINES = "naive"
HISTORY_DAYS = 60
TABLE_METRICS = ("mae", "rel_mae", "rmse", "mase", "nse", "kge")
PREDICTIONS_FILE = "predictions.parquet"
STATISTICS_FILE = "statistics.parquet"
_SEED = r"^(.+)_s\d+$"


def predictions_root() -> Path:
    """Carpeta de predicciones; ``ORINOCO_PREDICTIONS_DIR`` la sobrescribe."""
    override = os.environ.get(PREDICTIONS_ENV)
    return Path(override) if override else repo_root() / "data" / "predictions"


def run_command(model: str = "<modelo>", against: str = BASELINES) -> str:
    """Comando que genera las predicciones de ``model`` contra ``against``."""
    return f"uv run python scripts/run_model.py {model} --against {against}"


def discover_models(root: Path) -> list[str]:
    """Runners con ``predictions.parquet``; el baseline primero."""
    if not root.is_dir():
        return []
    found = [
        path.name for path in root.iterdir() if (path / PREDICTIONS_FILE).is_file()
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


def load_forecasts(root: Path, models: list[str], levels: pl.DataFrame) -> pl.DataFrame:
    """Predicciones en formato largo con el nivel observado (``y``).

    Las semillas se conservan y además se agrega su predicción media
    (``"<familia> (media de semillas)"``) solo para visualizar.
    """
    frames = []
    for model in models:
        path = root / model / PREDICTIONS_FILE
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
    forecasts = pl.concat(frames).join(
        levels.select("unique_id", "ds", "y"),
        on=["unique_id", "ds"],
        how="left",
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
