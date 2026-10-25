"""Model-ready frame: cleaned targets, upstream covariates and Fourier terms.

Output is long ``unique_id, ds, y`` (one ``unique_id`` per target station) with
covariates as extra columns and a boolean ``imputed`` column that is metadata,
not a model feature (see ``NON_FEATURE_COLUMNS``).
"""

import math
import warnings
from collections.abc import Iterable
from datetime import date
from typing import Any

import polars as pl

from orinoco_forecast.data.exogenous import add_exogenous, load_exogenous_config
from orinoco_forecast.data.loading import (
    DataQualityWarning,
    load_config,
    load_historical,
    load_primary,
)
from orinoco_forecast.data.quality import label_quality

NON_FEATURE_COLUMNS = ("imputed",)
EPOCH = date(1970, 1, 1)


def load_levels(
    config: dict[str, Any] | None = None,
    primary: pl.DataFrame | None = None,
    historical: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Tabla larga etiquetada (``unique_id, ds, y, source, quality``) en memoria.

    Misma composición que ``scripts/build_dataset.py``: primaria más la extensión
    histórica de Ciudad Bolívar, con conflictos medidos solo donde ambas fuentes
    se solapan. Se construye en memoria para no depender de un parquet viejo.
    """
    primary = load_primary(config) if primary is None else primary
    historical = load_historical(config) if historical is None else historical
    primary = primary.with_columns(source=pl.lit("primary"))
    primary_end = primary["ds"].max()
    extension = historical.filter(
        pl.col("unique_id").is_in(primary["unique_id"].unique().implode())
        & (pl.col("ds") > primary_end)
    ).with_columns(source=pl.lit("historical"))
    levels = pl.concat([primary, extension]).sort("unique_id", "ds")
    reference = historical.filter(pl.col("ds") <= primary_end)
    thresholds = (config or load_config())["quality"]
    return label_quality(levels, reference=reference, thresholds=thresholds)


def clean_series(df: pl.DataFrame, max_gap_days: int) -> pl.DataFrame:
    """Anula ``suspect_flat`` e interpola linealmente huecos de hasta ``max_gap_days``.

    - Requiere ``unique_id, ds, y, quality`` con filas diarias consecutivas por
      serie; el resultado se ordena por ``unique_id, ds``.
    - ``source_conflict`` se conserva tal cual (decisión del equipo: gana la
      fuente primaria).
    - Huecos más largos, o en los extremos de la serie, quedan nulos y se avisan
      con ``DataQualityWarning``.
    - Agrega ``imputed`` (bool): ``True`` donde el valor final es interpolado.
    """
    y = pl.col("y")
    run = y.is_null().rle_id().over("unique_id")
    out = (
        df.sort("unique_id", "ds")
        .with_columns(
            y=pl.when(pl.col("quality") == "suspect_flat").then(None).otherwise(y)
        )
        .with_columns(_run=run)
        .with_columns(
            _gap=y.is_null() & (pl.len().over("unique_id", "_run") <= max_gap_days)
        )
        .with_columns(
            _y=pl.when(pl.col("_gap"))
            .then(y.interpolate_by("ds").over("unique_id"))
            .otherwise(y)
        )
        .with_columns(imputed=y.is_null() & pl.col("_y").is_not_null())
    )
    left = (
        out.filter(pl.col("_y").is_null())
        .group_by("unique_id")
        .agg(rows=pl.len(), runs=pl.col("_run").n_unique())
        .sort("unique_id")
    )
    for uid, rows, runs in left.iter_rows():
        warnings.warn(
            f"{uid}: {runs} huecos sin interpolar (más de {max_gap_days} días o en "
            f"los extremos), {rows} valores quedan nulos.",
            DataQualityWarning,
            stacklevel=2,
        )
    return out.with_columns(y=pl.col("_y")).drop("_run", "_gap", "_y")


def fourier_terms(ds: pl.Series, period: float = 365.25, k: int = 3) -> pl.DataFrame:
    """Términos ``sin_j, cos_j = sin/cos(2·π·j·t / period)``, ``j = 1..k``.

    ``t`` son los días desde 1970-01-01, así que los términos son deterministas y
    conocidos a futuro (covariable futura sin fuga). Devuelve ``ds`` más
    ``sin_1..sin_k, cos_1..cos_k``.
    """
    t = (pl.col("ds") - pl.lit(EPOCH)).dt.total_days().cast(pl.Float64)
    angle = [2 * math.pi * j * t / period for j in range(1, k + 1)]
    return (
        ds.alias("ds")
        .to_frame()
        .with_columns(
            *(a.sin().alias(f"sin_{j}") for j, a in enumerate(angle, 1)),
            *(a.cos().alias(f"cos_{j}") for j, a in enumerate(angle, 1)),
        )
    )


def _assert_complete(frame: pl.DataFrame, label: str) -> None:
    nulls = {
        c: n for c, n in zip(frame.columns, frame.null_count().row(0), strict=True) if n
    }
    if nulls:
        raise ValueError(f"{label}: quedan valores nulos por columna: {nulls}")


def build_model_frame(
    stage: str = "A",
    config: dict[str, Any] | None = None,
    levels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Tabla para modelos: targets largos + estaciones aguas arriba + Fourier.

    - ``stage="A"``: ``unique_id, ds, y, <upstream>, sin_*, cos_*, imputed``.
      Se limita al rango común de las cuatro estaciones (la primaria termina el
      2025-02-24); la extensión 2025 solo de Ciudad Bolívar queda fuera.
    - ``stage="B"``: además lluvia NASA POWER (``rain_*``) y ``oni`` vía
      ``add_exogenous``, desde la primera fecha con lluvia en todas las
      estaciones (1981).
    - ``imputed`` es ``True`` si el target o alguna covariable de ese día fue
      interpolado. Levanta ``ValueError`` si queda algún nulo.
    """
    if stage not in {"A", "B"}:
        raise ValueError(f"Etapa desconocida: {stage!r} (use 'A' o 'B').")
    config = config or load_config()
    settings = config["model_frame"]
    targets, upstream = list(config["targets"]), list(settings["upstream"])
    levels = load_levels(config) if levels is None else levels
    stations = levels.filter(pl.col("unique_id").is_in(targets + upstream))
    spans = stations.group_by("unique_id").agg(
        start=pl.col("ds").min(), end=pl.col("ds").max()
    )
    start, end = spans["start"].max(), spans["end"].min()
    cleaned = clean_series(
        stations.filter(pl.col("ds").is_between(start, end)), settings["max_gap_days"]
    )

    target = cleaned.filter(pl.col("unique_id").is_in(targets))
    covs = cleaned.filter(pl.col("unique_id").is_in(upstream))
    cov_wide = covs.pivot("unique_id", index="ds", values="y").select("ds", *upstream)
    cov_imputed = covs.group_by("ds").agg(_cov_imputed=pl.col("imputed").any())
    calendar = fourier_terms(
        target["ds"].unique().sort(), settings["fourier_period"], settings["fourier_k"]
    )
    frame = (
        target.select("unique_id", "ds", "y", "imputed")
        .join(cov_wide, on="ds", how="left")
        .join(cov_imputed, on="ds", how="left")
        .join(calendar, on="ds", how="left")
        .with_columns(imputed=pl.col("imputed") | pl.col("_cov_imputed"))
        .drop("_cov_imputed")
    )
    extra: list[str] = []
    if stage == "B":
        nasa, oni = load_exogenous_config(config)
        rain = [c for c in nasa.columns if c != "ds"]
        first_rain = nasa.filter(pl.all_horizontal(pl.col(rain).is_not_null()))["ds"]
        # Restrict first so add_exogenous only warns about genuine gaps.
        frame = add_exogenous(frame.filter(pl.col("ds") >= first_rain.min()), nasa, oni)
        extra = [*rain, "oni"]
    features = [c for c in frame.columns if c not in {*extra, *NON_FEATURE_COLUMNS}]
    frame = frame.select(*features, *extra, *NON_FEATURE_COLUMNS).sort(
        "unique_id", "ds"
    )
    _assert_complete(frame, f"build_model_frame(stage={stage!r})")
    return frame


def assert_no_imputation_near_cutoffs(
    frame: pl.DataFrame, cutoffs: Iterable[date], horizon: int
) -> None:
    """Falla si hay filas ``imputed`` en ``[cutoff - horizon, cutoff + horizon]``.

    La interpolación de ``clean_series`` se hace una vez sobre la serie completa,
    no dentro de cada ventana. Eso solo filtra el futuro si un hueco interpolado
    cruza un corte: su valor antes del corte usaría la observación posterior.
    Como un hueco es una racha contigua de filas imputadas, cualquier hueco que
    cruce el corte contiene el propio ``cutoff``; y las filas evaluadas
    (``cutoff+1..cutoff+horizon``) deben ser observaciones reales. Exigir cero
    imputaciones en ``[cutoff - horizon, cutoff + horizon]`` cubre ambos casos
    con margen, y hace la interpolación global equivalente a la por ventana.
    """
    days = pl.DataFrame({"cutoff": list(cutoffs)}, schema={"cutoff": pl.Date})
    imputed = frame.filter(pl.col("imputed")).select("unique_id", "ds")
    hits = imputed.join(days, how="cross").filter(
        (pl.col("ds") - pl.col("cutoff")).dt.total_days().abs() <= horizon
    )
    if hits.height:
        sample = hits.head(5).rows()
        raise ValueError(
            f"{hits.height} filas imputadas a {horizon} días o menos de un corte; "
            f"la interpolación global filtraría el futuro. Ejemplos: {sample}"
        )
