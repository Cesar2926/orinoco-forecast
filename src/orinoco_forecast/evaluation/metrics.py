"""Métricas de evaluación de pronósticos.

- MAE y RMSE en las unidades del objetivo.
- sMAPE: ``mean(2|y - yhat| / (|y| + |yhat|))``, en ``[0, 2]``.
- MASE: MAE del pronóstico dividido por el error absoluto medio de la
  persistencia estacional calculado **solo con el histórico de entrenamiento**.
- MAPE: solo se calcula si ningún ``y_true`` está cerca de cero; en ese caso
  falla en lugar de producir un valor inestable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl

SeriesLike = pl.Series | Sequence[float]


def _as_series(values: SeriesLike, name: str) -> pl.Series:
    series = values if isinstance(values, pl.Series) else pl.Series(name, list(values))
    return series.cast(pl.Float64)


def _paired(y_true: SeriesLike, y_pred: SeriesLike) -> tuple[pl.Series, pl.Series]:
    true = _as_series(y_true, "y_true")
    pred = _as_series(y_pred, "y_pred")
    if true.len() != pred.len():
        raise ValueError("y_true y y_pred deben tener la misma longitud.")
    if true.is_empty():
        raise ValueError("La comparación no puede estar vacía.")
    if true.null_count() or pred.null_count():
        raise ValueError("Las métricas no aceptan valores nulos.")
    if not true.is_finite().all() or not pred.is_finite().all():
        raise ValueError("Las métricas requieren valores finitos.")
    return true, pred


def mae(y_true: SeriesLike, y_pred: SeriesLike) -> float:
    """Error absoluto medio."""
    true, pred = _paired(y_true, y_pred)
    return float((true - pred).abs().mean())


def rmse(y_true: SeriesLike, y_pred: SeriesLike) -> float:
    """Raíz del error cuadrático medio."""
    true, pred = _paired(y_true, y_pred)
    return math.sqrt(float(((true - pred) ** 2).mean()))


def smape(y_true: SeriesLike, y_pred: SeriesLike) -> float:
    """sMAPE simétrico en ``[0, 2]``; ``0/0`` cuenta como error cero."""
    true, pred = _paired(y_true, y_pred)
    difference = (true - pred).abs()
    denominator = true.abs() + pred.abs()
    return float((2 * difference / denominator).fill_nan(0.0).mean())


def mape(
    y_true: SeriesLike,
    y_pred: SeriesLike,
    *,
    epsilon: float = 1e-9,
) -> float:
    """MAPE, solo definido si ``|y_true| > epsilon`` en todas las filas."""
    if epsilon < 0:
        raise ValueError("epsilon no puede ser negativo.")
    true, pred = _paired(y_true, y_pred)
    if (true.abs() <= epsilon).any():
        raise ValueError("MAPE no está definido cerca de cero; usa MAE o sMAPE.")
    return float(((true - pred).abs() / true.abs()).mean())


def seasonal_scale(history: pl.DataFrame, *, seasonality: int = 365) -> float:
    """Error absoluto medio de la persistencia estacional en entrenamiento.

    Se empareja cada fecha con la de ``seasonality`` días después; los pares
    con nulos se ignoran. Es el denominador del MASE.
    """
    if seasonality < 1:
        raise ValueError("seasonality debe ser positiva.")
    required = {"ds", "y"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if history.schema["ds"] != pl.Date:
        raise ValueError(f"ds debe ser pl.Date, no {history.schema['ds']}.")
    if not history.schema["y"].is_numeric():
        raise ValueError(f"y debe ser numérica, no {history.schema['y']}.")
    if history["ds"].null_count():
        raise ValueError("ds no puede contener nulos.")
    if history["ds"].is_duplicated().any():
        raise ValueError(
            "El histórico contiene fechas duplicadas; la escala estacional "
            "no se puede calcular sin ambigüedad."
        )

    paired = history.select("ds", "y").join(
        history.select(
            (pl.col("ds") + pl.duration(days=seasonality)).alias("ds"),
            pl.col("y").alias("y_lag"),
        ),
        on="ds",
        how="inner",
    )
    errors = (paired["y"] - paired["y_lag"]).abs().drop_nulls()
    if errors.is_empty():
        raise ValueError(f"No hay pares estacionales para seasonality={seasonality}.")
    scale = float(errors.mean())
    if scale == 0:
        raise ValueError("La escala estacional es cero; MASE no está definido.")
    return scale


def mase(
    y_true: SeriesLike,
    y_pred: SeriesLike,
    history: pl.DataFrame,
    *,
    seasonality: int = 365,
) -> float:
    """MAE escalado por la persistencia estacional del entrenamiento."""
    return mae(y_true, y_pred) / seasonal_scale(history, seasonality=seasonality)
