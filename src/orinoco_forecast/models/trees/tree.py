"""XGBoost with mlforecast: direct multi-step strategy and pre-lagged covariates.

Hyperparameters live only in ``configs/models/tree.yaml``; the protocol
(``horizon``, ``step_size``, ``n_windows``, ``refit``) in ``configs/backtest.yaml``.
"""

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import polars as pl
import yaml
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import Differences
from xgboost import XGBRegressor

from orinoco_forecast.data.loading import load_config, repo_root
from orinoco_forecast.evaluation.backtest import development_frame
from orinoco_forecast.features.dataset import NON_FEATURE_COLUMNS, fourier_terms
from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    PredictionResult,
    forecast_index,
    validate_horizon,
    validate_windows,
)

KEYS = ("unique_id", "ds", "y")
# Fourier terms from features.dataset.fourier_terms: deterministic, known ahead.
FUTURE_PREFIXES = ("sin_", "cos_")
MODEL = "XGBoost"


def load_tree_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee ``configs/models/tree.yaml`` (o ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "models" / "tree.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prelag_covariates(
    frame: pl.DataFrame, columns: Sequence[str], lags: Sequence[int]
) -> pl.DataFrame:
    """Reemplaza cada covariable histórica por sus rezagos ``<col>_lag<k>``.

    - El rezago se toma desplazando ``ds`` dentro de cada ``unique_id``; exige
      filas diarias consecutivas (``ValueError`` si hay huecos).
    - Se eliminan las columnas del mismo día y las de metadatos
      (``NON_FEATURE_COLUMNS``): mlforecast usa toda columna dinámica en la fecha
      pronosticada, así que el valor crudo de ese día sería fuga.
    - Las primeras ``max(lags)`` filas de cada serie quedan sin rezago y se
      descartan (calentamiento); el final de la serie no se toca.
    """
    frame = frame.sort("unique_id", "ds")
    step = pl.col("ds").diff().over("unique_id").dt.total_days()
    if frame.select((step.fill_null(1) != 1).any()).item():
        raise ValueError("prelag_covariates requiere fechas diarias consecutivas.")
    names = [f"{c}_lag{k}" for c in columns for k in lags]
    lagged = frame.with_columns(
        pl.col(c).shift(k).over("unique_id").alias(f"{c}_lag{k}")
        for c in columns
        for k in lags
    )
    drop = [c for c in (*columns, *NON_FEATURE_COLUMNS) if c in lagged.columns]
    return lagged.drop(drop).drop_nulls(names)


class XGBoostRunner:
    """XGBoost global (ambas estaciones) con un modelo por paso del horizonte.

    - Estrategia directa (``max_horizon = horizon``): la recursiva realimenta
      sus propias predicciones y acumula error (Ben Taieb et al. 2012).
    - ``Differences([1])``: los árboles no extrapolan fuera del rango visto; se
      modela el cambio diario y mlforecast reconstruye el nivel.
    - Covariables: Fourier como futuras (conocidas de antemano); el resto
      (aguas arriba y, en etapa B, lluvia/ONI) como históricas pre-rezagadas
      ``covariate_lags`` días, con ``min >= horizon`` para que todo valor usado
      en la fecha pronosticada ya se conozca en el corte.
    """

    name = "xgboost"
    model_columns = [MODEL]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or load_tree_config()

    def _prepare(self, frame: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
        lags = self.params["covariate_lags"]
        if min(lags) < cfg["horizon"]:
            raise ValueError(
                f"covariate_lags mínimo ({min(lags)}) menor que el horizonte "
                f"({cfg['horizon']}): se usarían valores posteriores al corte."
            )
        skip = {*KEYS, *NON_FEATURE_COLUMNS}
        historical = [
            c
            for c in frame.columns
            if c not in skip and not c.startswith(FUTURE_PREFIXES)
        ]
        return prelag_covariates(frame, historical, lags)

    def _forecaster(self) -> MLForecast:
        p = self.params
        return MLForecast(
            models={MODEL: XGBRegressor(**p["xgb_params"])},
            # With polars input mlforecast needs a polars offset ("1d"), not "D".
            freq="1d",
            lags=p["lags"],
            lag_transforms={
                1: [RollingMean(window_size=w) for w in p["rolling_mean_windows"]]
            },
            target_transforms=[Differences(p["target_differences"])],
        )

    def cross_validation(
        self, frame: pl.DataFrame, cfg: dict[str, Any]
    ) -> pl.DataFrame:
        # static_features=[]: otherwise every extra column is treated as static.
        return self._forecaster().cross_validation(
            df=self._prepare(frame, cfg),
            n_windows=cfg["n_windows"],
            h=cfg["horizon"],
            step_size=cfg["step_size"],
            refit=cfg["refit"],
            static_features=[],
            max_horizon=cfg["horizon"],
        )

    def feature_importance(
        self, frame: pl.DataFrame, cfg: dict[str, Any]
    ) -> pl.DataFrame:
        """Importancia por ganancia tras ajustar en el desarrollo (``< test_start``).

        Por cada uno de los ``horizon`` modelos se toma la ganancia total de
        XGBoost por variable normalizada a suma 1; ``importance_gain`` es su
        media entre horizontes (cada paso pesa igual). Ordenado descendente.
        """
        data = self._prepare(development_frame(frame, cfg), cfg)
        mlf = self._forecaster().fit(
            data, static_features=[], max_horizon=cfg["horizon"]
        )
        shares = []
        for model in mlf.models_[MODEL].values():
            booster = model.get_booster()
            names = booster.feature_names or mlf.ts.features_order_
            score = booster.get_score(importance_type="total_gain")
            gain = [score.get(n, score.get(f"f{i}", 0.0)) for i, n in enumerate(names)]
            total = sum(gain) or 1.0
            shares.append(
                pl.DataFrame({"feature": names, "share": [g / total for g in gain]})
            )
        return (
            pl.concat(shares)
            .group_by("feature")
            .agg(importance_gain=pl.col("share").sum() / len(shares))
            .sort("importance_gain", "feature", descending=[True, False])
        )


class BoosterRunner(XGBoostRunner):
    """XGBoost con estrategia directa, ajustado en un solo corte.

    Usa ``frame`` hasta ``cutoff`` (mlforecast necesita la serie completa para
    construir rezagos y diferencias) y devuelve solo la predicción. ``windows``
    se valida como entrada común del contrato.
    """

    name = "booster"
    implemented = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._mlf: MLForecast | None = None

    def _future_covariates(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int,
    ) -> pl.DataFrame:
        """Future values of the pre-lagged covariates required by mlforecast.

        Every dynamic covariate used by the model is lagged at least
        ``horizon`` days, so its future value at ``cutoff + h`` comes from an
        observation at or before ``cutoff``. Fourier terms are deterministic.
        """
        history = frame.filter(pl.col("ds") <= cutoff)
        skip = {*KEYS, *NON_FEATURE_COLUMNS}
        historical = [
            column
            for column in history.columns
            if column not in skip and not column.startswith(FUTURE_PREFIXES)
        ]
        dates = pl.date_range(
            cutoff + timedelta(days=1),
            cutoff + timedelta(days=horizon),
            "1d",
            eager=True,
        )
        future = pl.DataFrame({"ds": dates}).join(
            history.select("unique_id").unique(), how="cross"
        )
        for column in historical:
            for lag in self.params["covariate_lags"]:
                source = history.select(
                    "unique_id",
                    (pl.col("ds") + pl.duration(days=lag)).alias("ds"),
                    pl.col(column).alias(f"{column}_lag{lag}"),
                )
                future = future.join(source, on=["unique_id", "ds"], how="left")
        settings = load_config()["model_frame"]
        calendar = fourier_terms(
            future["ds"].unique(),
            settings["fourier_period"],
            settings["fourier_k"],
        )
        future = future.join(calendar, on="ds", how="left")
        required = [
            column
            for column in future.columns
            if column not in {"unique_id", "ds"} and future[column].null_count()
        ]
        if required:
            raise ValueError(
                "No se pudieron construir covariables futuras sin nulos: "
                f"{sorted(required)}."
            )
        return future

    def fit(
        self,
        windows: pl.DataFrame,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> Self:
        validate_horizon(horizon)
        validate_windows(windows, horizon=horizon)
        history = frame.filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")
        if history.is_empty():
            raise ValueError(f"No hay observaciones hasta el corte {cutoff}.")
        data = self._prepare(history, {"horizon": horizon})
        self._mlf = self._forecaster().fit(
            data, static_features=[], max_horizon=horizon
        )
        return self

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> PredictionResult:
        validate_horizon(horizon)
        if self._mlf is None:
            raise RuntimeError("El runner no está ajustado; llama a fit primero.")
        future = self._future_covariates(frame, cutoff=cutoff, horizon=horizon)
        raw = self._mlf.predict(h=horizon, X_df=future)
        if not isinstance(raw, pl.DataFrame):
            raw = pl.from_pandas(raw.reset_index())
        raw = raw.with_columns(pl.col("ds").cast(pl.Date))
        value_columns = [
            column for column in raw.columns if column not in {"unique_id", "ds"}
        ]
        if len(value_columns) != 1:
            raise RuntimeError(
                f"mlforecast devolvió columnas inesperadas: {value_columns}."
            )
        history = frame.filter(pl.col("ds") <= cutoff)
        index = forecast_index(
            history, model=self.name, cutoff=cutoff, horizon=horizon
        )
        predictions = index.join(
            raw.select(
                "unique_id",
                "ds",
                pl.col(value_columns[0]).cast(pl.Float64).alias("y_pred"),
            ),
            on=["unique_id", "ds"],
            how="left",
        )
        if predictions["y_pred"].null_count():
            raise RuntimeError("Faltan predicciones para algunas estaciones.")
        return PredictionResult.from_frame(
            predictions, runner=self.name, horizon=horizon
        )
