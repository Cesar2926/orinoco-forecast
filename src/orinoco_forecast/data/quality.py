"""Per-value quality labels for long ``unique_id, ds, y`` frames.

Labels never modify ``y``; they only describe it. When several labels apply to
the same row, precedence is::

    missing > source_conflict > suspect_flat > suspect_jump > observed

``missing`` wins because there is no value to judge; ``source_conflict`` beats
the single-series heuristics because it is evidence from an independent source.
"""

import warnings
from typing import Any

import polars as pl

from orinoco_forecast.data.loading import DataQualityWarning, load_config

QUALITY_LABELS = (
    "observed",
    "missing",
    "suspect_flat",
    "suspect_jump",
    "source_conflict",
)
QUALITY_DTYPE = pl.Enum(QUALITY_LABELS)


def find_flat_runs(values: pl.Series, min_days: int) -> pl.Series:
    """Marca los valores que pertenecen a una racha plana (nulo disfrazado).

    Contrato:
    - ``values``: una sola serie diaria ordenada por fecha (un ``unique_id``).
    - Devuelve un ``pl.Series`` booleano del mismo largo y orden que ``values``.
    - Un elemento es ``True`` si y solo si pertenece a una racha de ``min_days``
      o más valores consecutivos idénticos y no nulos. Se marcan todos los
      elementos de la racha, incluido el primero.
    - Los nulos nunca forman ni extienden una racha y siempre devuelven ``False``.
    - No modifica ``values``.
    """
    raise NotImplementedError("Ejercicio: ver docstring")


def _flat_mask(df: pl.DataFrame, min_days: int) -> pl.Series:
    parts = df.with_row_index("_row").partition_by("unique_id", maintain_order=True)
    try:
        masks = [
            part.select("_row").with_columns(
                find_flat_runs(part["y"], min_days).alias("_flat")
            )
            for part in parts
        ]
    except NotImplementedError:
        warnings.warn(
            "find_flat_runs no está implementado: no se evaluó suspect_flat.",
            DataQualityWarning,
            stacklevel=3,
        )
        return pl.Series("_flat", [False] * df.height)
    return pl.concat(masks).sort("_row")["_flat"]


def label_quality(
    df: pl.DataFrame,
    reference: pl.DataFrame | None = None,
    thresholds: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Devuelve ``df`` con una columna ``quality`` (``pl.Enum(QUALITY_LABELS)``).

    - ``missing``: ``y`` nulo.
    - ``source_conflict``: ``|y - y_ref| > source_conflict_m`` contra ``reference``
      (mismo ``unique_id`` y ``ds``); solo si se pasa ``reference``.
    - ``suspect_flat``: racha de al menos ``flat_min_days`` (``find_flat_runs``).
    - ``suspect_jump``: ``|y_t - y_{t-1}| > jump_max_m_per_day`` dentro del mismo
      ``unique_id``; asume filas diarias consecutivas.

    ``thresholds`` por defecto: bloque ``quality`` de ``configs/data.yaml``.
    """
    thresholds = thresholds or load_config()["quality"]
    y = pl.col("y")
    out = df.with_columns(_flat=_flat_mask(df, thresholds["flat_min_days"]))
    if reference is not None:
        ref = reference.select("unique_id", "ds", pl.col("y").alias("_y_ref"))
        out = out.join(
            ref,
            on=["unique_id", "ds"],
            how="left",
            maintain_order="left",
            validate="m:1",
        )
    else:
        out = out.with_columns(_y_ref=pl.lit(None, dtype=pl.Float64))
    step = (y - y.shift(1)).abs().over("unique_id", order_by="ds")
    conflict = (y - pl.col("_y_ref")).abs() > thresholds["source_conflict_m"]
    quality = (
        pl.when(y.is_null())
        .then(pl.lit("missing"))
        .when(conflict.fill_null(False))
        .then(pl.lit("source_conflict"))
        .when(pl.col("_flat"))
        .then(pl.lit("suspect_flat"))
        .when((step > thresholds["jump_max_m_per_day"]).fill_null(False))
        .then(pl.lit("suspect_jump"))
        .otherwise(pl.lit("observed"))
        .cast(QUALITY_DTYPE)
    )
    return out.with_columns(quality=quality).drop("_flat", "_y_ref")
