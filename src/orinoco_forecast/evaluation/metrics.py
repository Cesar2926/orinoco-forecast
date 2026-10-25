"""Métricas de evaluación de pronósticos y puntuación por horizonte de la CV.

- MAE y RMSE en las unidades del objetivo.
- sMAPE: ``mean(2|y - yhat| / (|y| + |yhat|))``, en ``[0, 2]``.
- MAPE: solo se calcula si ningún ``y_true`` está cerca de cero; en ese caso
  falla en lugar de producir un valor inestable.
- NSE y KGE: eficiencias hidrológicas (1 es perfecto).
- Hay dos variantes de MASE. ``mase`` escala el MAE por la persistencia
  estacional de 365 días calculada **solo con el histórico de entrenamiento**.
  ``mase_horizon`` escala por la persistencia a ``season`` días y es la que usan
  ``evaluate_cv`` y el protocolo de selección de modelos.

``evaluate_cv`` puntúa un marco de validación cruzada
``unique_id, ds, cutoff, y, <columnas de modelos>``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

ArrayLike = npt.ArrayLike
SeriesLike = pl.Series | Sequence[float]
METRICS = ("mae", "rmse", "mase", "nse", "kge")


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


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if t.shape != p.shape or t.ndim != 1 or t.size == 0:
        raise ValueError(
            f"Se esperaban vectores 1D no vacíos iguales: {t.shape}, {p.shape}"
        )
    return t, p


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


def naive_scale(y_train: ArrayLike, season: int = 1) -> float:
    """MAE dentro de muestra de la persistencia estacional ``y_t - y_{t-season}``."""
    train = np.asarray(y_train, dtype=np.float64)
    if train.ndim != 1 or train.size <= season:
        raise ValueError(f"MASE necesita más de {season} valores de entrenamiento.")
    return float(np.mean(np.abs(train[season:] - train[:-season])))


def mase_horizon(
    y_true: ArrayLike, y_pred: ArrayLike, y_train: ArrayLike, season: int = 1
) -> float:
    """MAE escalado por el MAE de la persistencia (lag ``season``) en ``y_train``.

    Menor que 1 significa que el pronóstico supera a la persistencia.
    """
    return mae(y_true, y_pred) / naive_scale(y_train, season)


def nse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Nash-Sutcliffe: ``1 - SSE / SS`` alrededor de la media de ``y_true``.

    1 es perfecto, 0 equivale a predecir la media observada.
    """
    t, p = _pair(y_true, y_pred)
    return float(1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def kge(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Kling-Gupta (Gupta et al., 2009): ``1 - sqrt((r-1)² + (α-1)² + (β-1)²)``.

    ``r`` es la correlación de Pearson, ``α = σ_sim / σ_obs`` y
    ``β = μ_sim / μ_obs``. Es NaN si alguna serie es constante.
    """
    t, p = _pair(y_true, y_pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.corrcoef(t, p)[0, 1]
        alpha = p.std() / t.std()
        beta = p.mean() / t.mean()
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _score(t: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"mae": mae(t, p), "rmse": rmse(t, p), "nse": nse(t, p), "kge": kge(t, p)}


def evaluate_cv(
    cv_df: pl.DataFrame, models: Sequence[str], train_df: pl.DataFrame
) -> pl.DataFrame:
    """Métricas por ``unique_id`` x modelo x paso del horizonte, más ``"all"``.

    - ``horizon`` (String) es ``(ds - cutoff)`` en días (``"1"``…``"14"``) o
      ``"all"`` para todas las filas de la serie.
    - MASE ajustado al horizonte (``mase_horizon``): en el paso ``h`` es
      ``MAE_h`` dividido por el MAE de la persistencia a ``h`` días
      (``|y_t - y_{t-h}|``) en entrenamiento, así la persistencia vale ~1 en
      cada paso. En ``"all"`` es la media de los ``MASE_h``; las demás métricas
      de ``"all"`` usan todas las filas juntas.
    - La escala se calcula por ``unique_id`` con las filas de ``train_df`` hasta
      el **primer** ``cutoff``: es la misma para todas las ventanas y nunca usa
      datos posteriores a un corte evaluado.
    - Las filas con ``y`` o predicción nula se descartan antes de puntuar.

    Devuelve ``unique_id, model, horizon, n`` y una columna por métrica.
    """
    first_cutoff = cv_df["cutoff"].min()
    scored = cv_df.with_columns(
        h=(pl.col("ds") - pl.col("cutoff")).dt.total_days().cast(pl.Int32)
    )
    rows: list[dict[str, object]] = []
    for (uid,), part in scored.partition_by("unique_id", as_dict=True).items():
        history = train_df.filter(
            (pl.col("unique_id") == uid) & (pl.col("ds") <= first_cutoff)
        ).sort("ds")
        y_train = history["y"].drop_nulls().to_numpy()
        for model in models:
            valid = part.drop_nulls(["y", model])
            steps = sorted(valid.partition_by("h", as_dict=True).items())
            per_h = []
            for (h,), group in steps:
                t, p = group["y"].to_numpy(), group[model].to_numpy()
                score = {**_score(t, p), "mase": mase_horizon(t, p, y_train, season=h)}
                per_h.append(score["mase"])
                rows.append(
                    {
                        "unique_id": uid,
                        "model": model,
                        "horizon": str(h),
                        "n": group.height,
                        **score,
                    }
                )
            t, p = valid["y"].to_numpy(), valid[model].to_numpy()
            rows.append(
                {
                    "unique_id": uid,
                    "model": model,
                    "horizon": "all",
                    "n": valid.height,
                    **_score(t, p),
                    "mase": float(np.mean(per_h)),
                }
            )
    schema = {
        "unique_id": pl.String,
        "model": pl.String,
        "horizon": pl.String,
        "n": pl.Int64,
        **{m: pl.Float64 for m in METRICS},
    }
    return pl.DataFrame(rows, schema=schema)


def relative_to_baseline(
    metrics: pl.DataFrame, baseline: pl.DataFrame, baseline_model: str = "Naive"
) -> pl.DataFrame:
    """Agrega ``mae_naive`` y ``rel_mae = mae / mae_naive`` a ``metrics``.

    - Ambos marcos tienen el esquema de ``evaluate_cv``; se unen por
      ``unique_id`` y ``horizon`` con las filas ``baseline_model`` de ``baseline``.
      ``rel_mae < 1`` significa que el modelo supera a esa referencia.
    - Las métricas no guardan los cortes, así que se asume que ambos marcos
      vienen de ``run_cv`` con el mismo ``configs/backtest.yaml``. Como control,
      ``n`` (filas puntuadas) debe coincidir; si no, ``ValueError``.
    - ``ValueError`` si falta la referencia para algún ``unique_id``/``horizon``.
    """
    ref = baseline.filter(pl.col("model") == baseline_model).select(
        "unique_id", "horizon", n_naive="n", mae_naive="mae"
    )
    if ref.is_empty():
        raise ValueError(
            f"No hay métricas de {baseline_model!r} en las referencias; "
            "ejecute primero los baselines."
        )
    out = metrics.join(ref, on=["unique_id", "horizon"], how="left")
    missing = out.filter(pl.col("mae_naive").is_null())
    if not missing.is_empty():
        keys = sorted(set(missing["unique_id"]))
        raise ValueError(f"Falta {baseline_model!r} para unique_id {keys}.")
    if (out["n"] != out["n_naive"]).any():
        raise ValueError(
            f"El modelo y {baseline_model!r} no se evaluaron en las mismas ventanas "
            "(distinto número de filas puntuadas)."
        )
    return out.drop("n_naive").with_columns(rel_mae=pl.col("mae") / pl.col("mae_naive"))
