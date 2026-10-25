"""XGBoost with mlforecast: direct multi-step strategy and pre-lagged covariates.

Hyperparameters live only in ``configs/models/tree.yaml``; the protocol
(``horizon``, ``step_size``, ``n_windows``, ``refit``) in ``configs/backtest.yaml``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean
from mlforecast.target_transforms import Differences
from xgboost import XGBRegressor

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.evaluation.backtest import development_frame
from orinoco_forecast.features.dataset import NON_FEATURE_COLUMNS

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
