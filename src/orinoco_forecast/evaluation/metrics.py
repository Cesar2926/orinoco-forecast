"""Point-forecast metrics (MAE, RMSE, MASE, NSE, KGE) and per-horizon CV scoring.

Metric functions take array-likes and return a float. ``evaluate_cv`` scores a
cross-validation frame ``unique_id, ds, cutoff, y, <model columns>``.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

ArrayLike = npt.ArrayLike
METRICS = ("mae", "rmse", "mase", "nse", "kge")


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if t.shape != p.shape or t.ndim != 1 or t.size == 0:
        raise ValueError(
            f"Se esperaban vectores 1D no vacíos iguales: {t.shape}, {p.shape}"
        )
    return t, p


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Error absoluto medio, en las unidades de ``y`` (metros)."""
    t, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Raíz del error cuadrático medio, en las unidades de ``y``."""
    t, p = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2)))


def naive_scale(y_train: ArrayLike, season: int = 1) -> float:
    """MAE dentro de muestra de la persistencia estacional ``y_t - y_{t-season}``."""
    train = np.asarray(y_train, dtype=np.float64)
    if train.ndim != 1 or train.size <= season:
        raise ValueError(f"MASE necesita más de {season} valores de entrenamiento.")
    return float(np.mean(np.abs(train[season:] - train[:-season])))


def mase(
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
    - MASE ajustado al horizonte: en el paso ``h`` es ``MAE_h`` dividido por el
      MAE de la persistencia a ``h`` días (``|y_t - y_{t-h}|``) en entrenamiento,
      así la persistencia vale ~1 en cada paso. En ``"all"`` es la media de los
      ``MASE_h``; las demás métricas de ``"all"`` usan todas las filas juntas.
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
                score = {**_score(t, p), "mase": mase(t, p, y_train, season=h)}
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
