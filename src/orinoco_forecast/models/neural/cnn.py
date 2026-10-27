"""TCN (temporal convolutional network) with neuralforecast, one model per seed.

Hyperparameters live only in ``configs/models/cnn.yaml``; the protocol
(``horizon``, ``step_size``, ``n_windows``, ``refit``) in ``configs/backtest.yaml``.
"""

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import polars as pl
import torch
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.losses import pytorch as losses
from neuralforecast.models import TCN

from orinoco_forecast.data.loading import load_config, repo_root
from orinoco_forecast.features.dataset import NON_FEATURE_COLUMNS, fourier_terms
from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    PredictionResult,
    validate_horizon,
    validate_windows,
)

KEYS = ("unique_id", "ds", "y")
# Fourier terms from features.dataset.fourier_terms: deterministic, known ahead.
FUTURE_PREFIXES = ("sin_", "cos_")
MODEL = "TCN"
TRANSFORMS = ("level", "difference")


def load_cnn_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee ``configs/models/cnn.yaml`` (o ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "models" / "cnn.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def receptive_field(kernel_size: int, dilations: Sequence[int]) -> int:
    """Días que ve la última salida causal: ``1 + (kernel_size - 1) * sum(d)``."""
    return 1 + (kernel_size - 1) * sum(dilations)


def split_covariates(frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Separa covariables en futuras (Fourier ``sin_*``/``cos_*``) e históricas.

    Históricas son todas las demás (aguas arriba; en etapa B lluvia y ONI).
    Se excluyen ``unique_id, ds, y`` y los metadatos (``NON_FEATURE_COLUMNS``).
    """
    skip = {*KEYS, *NON_FEATURE_COLUMNS}
    covs = [c for c in frame.columns if c not in skip]
    futr = [c for c in covs if c.startswith(FUTURE_PREFIXES)]
    return futr, [c for c in covs if c not in futr]


def prepare_frame(
    frame: pl.DataFrame, transform: str = "level"
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Tabla para neuralforecast y listas de exógenas futuras/históricas.

    - ``"level"``: ``y`` es el nivel.
    - ``"difference"``: ``y`` pasa a ser ``y_t - y_{t-1}`` por ``unique_id``
      (se descarta la primera fila de cada serie) y el nivel actual entra como
      histórica ``level_y``. Igual que las demás históricas, TCN solo la lee
      hasta el corte de cada ventana (``hist_exog`` se corta a ``input_size``,
      ver ``TCNRunner``).
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"target_transform desconocido: {transform!r} {TRANSFORMS}")
    futr, hist = split_covariates(frame)
    data = frame.select(*KEYS, *futr, *hist).sort("unique_id", "ds")
    if transform == "level":
        return data, futr, hist
    data = data.with_columns(
        level_y=pl.col("y"), y=pl.col("y").diff().over("unique_id")
    ).drop_nulls("y")
    return data, futr, [*hist, "level_y"]


def reconstruct_levels(
    cv: pl.DataFrame, levels: pl.DataFrame, columns: Sequence[str]
) -> pl.DataFrame:
    """Convierte pronósticos de diferencias en niveles.

    Cada columna en ``columns`` pasa a ser ``y(cutoff) + suma acumulada`` de las
    diferencias pronosticadas en los pasos ``1..h`` de su ventana; ``y(cutoff)``
    es el nivel observado el día del corte (conocido en el corte). ``y`` se
    reemplaza por el nivel observado en ``ds``. ``levels`` trae
    ``unique_id, ds, y``.
    """
    at_cutoff = levels.select("unique_id", cutoff="ds", _y_cut="y")
    observed = levels.select("unique_id", "ds", _y_obs="y")
    window = ("unique_id", "cutoff")
    return (
        cv.join(at_cutoff, on=window, how="left")
        .join(observed, on=["unique_id", "ds"], how="left")
        .sort("unique_id", "cutoff", "ds")
        .with_columns(
            (pl.col("_y_cut") + pl.col(c).cum_sum().over(window)).alias(c)
            for c in columns
        )
        .with_columns(y=pl.col("_y_obs"))
        .select("unique_id", "ds", "cutoff", "y", *columns)
    )


class TCNRunner:
    """TCN global (ambas estaciones), un modelo por semilla en las mismas ventanas.

    - Columnas ``TCN_s<semilla>``; ``run_model.py`` reporta media ± desviación
      de las métricas entre semillas (no promedia predicciones).
    - Exógenas: Fourier en ``futr_exog_list``; el resto en ``hist_exog_list``.
      Verificado en neuralforecast 3.3.0 (``BaseModel._parse_windows``): en
      modelos no recurrentes como TCN, ``hist_exog`` se corta a
      ``[:, :input_size]``, es decir, solo días hasta el corte de cada ventana;
      ``TCN.forward`` solo usa ``futr_exog`` después del corte. El escalador
      enmascara los últimos ``h`` días (``_normalization``).
    - ``refit=False``: ``cross_validation`` entrena una vez con datos hasta el
      primer corte menos ``val_size`` días; esos ``val_size`` días (que terminan
      en el primer corte) solo sirven para la parada temprana. Luego predice las
      ventanas con paso ``step_size``. Las ventanas de entrenamiento se muestrean
      al azar, pero cada una conserva su orden temporal.
    - ``target_transform: difference`` (``cnn.yaml``): entrena sobre el cambio
      diario y reconstruye el nivel con ``reconstruct_levels``. Motivo: en nivel
      escalado, reproducir "mañana ≈ hoy" es difícil y el TCN en nivel pierde
      contra la persistencia en h=1; en diferencias la persistencia es la
      predicción cero, como ya hace XGBoost con ``Differences([1])``.
    """

    name = "tcn"

    implemented = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or load_cnn_config()
        self._nf: NeuralForecast | None = None
        self._transform = "level"
        self._futr: list[str] = []
        self._fitted = False

    @property
    def model_columns(self) -> list[str]:
        return [f"{MODEL}_s{s}" for s in self.params["seeds"]]

    def _models(self, h: int, futr: list[str], hist: list[str]) -> list[TCN]:
        p = self.params
        field = receptive_field(p["kernel_size"], p["dilations"])
        if p["input_size"] < field:
            raise ValueError(
                f"input_size ({p['input_size']}) menor que el campo receptivo "
                f"({field}): parte de la red no recibiría datos."
            )
        if p["early_stop_patience_steps"] > 0 and p["val_size"] < h:
            raise ValueError("La parada temprana requiere val_size >= horizonte.")
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        return [
            TCN(
                h=h,
                input_size=p["input_size"],
                kernel_size=p["kernel_size"],
                dilations=list(p["dilations"]),
                encoder_hidden_size=p["encoder_hidden_size"],
                decoder_hidden_size=p["decoder_hidden_size"],
                decoder_layers=p["decoder_layers"],
                futr_exog_list=futr,
                hist_exog_list=hist,
                loss=getattr(losses, p["loss"])(),
                learning_rate=p["learning_rate"],
                max_steps=p["max_steps"],
                val_check_steps=p["val_check_steps"],
                early_stop_patience_steps=p["early_stop_patience_steps"],
                batch_size=p["batch_size"],
                windows_batch_size=p["windows_batch_size"],
                scaler_type=p["scaler_type"],
                random_seed=seed,
                alias=column,
                accelerator=accelerator,
                **p["trainer"],
            )
            for seed, column in zip(p["seeds"], self.model_columns, strict=True)
        ]

    def _future_exog(
        self, frame: pl.DataFrame, *, cutoff: date, horizon: int
    ) -> pl.DataFrame | None:
        """Future Fourier terms required by the fitted models."""
        if not self._futr:
            return None
        dates = pl.date_range(
            cutoff + timedelta(days=1),
            cutoff + timedelta(days=horizon),
            "1d",
            eager=True,
        )
        calendar = fourier_terms(
            pl.Series("ds", dates),
            load_config()["model_frame"]["fourier_period"],
            load_config()["model_frame"]["fourier_k"],
        )
        future = (
            calendar.join(
                frame.select("unique_id").unique(), how="cross"
            )
            .select("unique_id", "ds", *self._futr)
        )
        missing = [column for column in self._futr if column not in future.columns]
        if missing:
            raise ValueError(
                "No se pudieron construir las covariables futuras del TCN: "
                f"{missing}."
            )
        return future.select("unique_id", "ds", *self._futr)

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
        transform = self.params.get("target_transform", "level")
        data, futr, hist = prepare_frame(history, transform)
        val_size = int(self.params["val_size"])
        if val_size >= data.height:
            raise ValueError(
                f"val_size ({val_size}) debe ser menor que las filas de "
                f"entrenamiento ({data.height})."
            )
        nf = NeuralForecast(
            models=self._models(horizon, futr, hist),
            # Polars input needs a polars offset ("1d"), not pandas "D".
            freq="1d",
        )
        nf.fit(df=data, val_size=val_size, verbose=False)
        self._nf = nf
        self._transform = transform
        self._futr = list(futr)
        self._fitted = True
        return self

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> PredictionResult:
        validate_horizon(horizon)
        if not self._fitted or self._nf is None:
            raise RuntimeError("El runner no está ajustado; llama a fit primero.")
        raw = self._nf.predict(
            futr_df=self._future_exog(frame, cutoff=cutoff, horizon=horizon)
        )
        if not isinstance(raw, pl.DataFrame):
            raw = pl.from_pandas(raw.reset_index())
        raw = raw.with_columns(pl.col("ds").cast(pl.Date))
        missing = [column for column in self.model_columns if column not in raw.columns]
        if missing:
            raise RuntimeError(f"Faltan columnas del TCN en la predicción: {missing}.")
        if self._transform == "difference":
            levels = (
                frame.filter(pl.col("ds") <= cutoff)
                .filter(pl.col("y").is_not_null())
                .select("unique_id", "ds", "y")
            )
            at_cutoff = levels.group_by("unique_id").agg(
                _y_cut=pl.col("y").sort_by("ds").last()
            )
            raw = (
                raw.join(at_cutoff, on="unique_id", how="left")
                .sort("unique_id", "ds")
                .with_columns(
                    (
                        pl.col("_y_cut")
                        + pl.col(column).cum_sum().over("unique_id")
                    ).alias(column)
                    for column in self.model_columns
                )
                .drop("_y_cut")
            )
        long = raw.unpivot(
            on=self.model_columns,
            index=["unique_id", "ds"],
            variable_name="model",
            value_name="y_pred",
        ).with_columns(
            pl.lit(cutoff).alias("cutoff"),
            (pl.col("ds") - cutoff)
            .dt.total_days()
            .cast(pl.Int32)
            .alias("horizon_step"),
        )
        return PredictionResult.from_frame(long, runner=self.name, horizon=horizon)

    def cross_validation(
        self, frame: pl.DataFrame, cfg: dict[str, Any]
    ) -> pl.DataFrame:
        transform = self.params.get("target_transform", "level")
        data, futr, hist = prepare_frame(frame, transform)
        nf = NeuralForecast(
            models=self._models(cfg["horizon"], futr, hist),
            # Polars input needs a polars offset ("1d"), not pandas "D".
            freq="1d",
        )
        cv = nf.cross_validation(
            df=data,
            n_windows=cfg["n_windows"],
            step_size=cfg["step_size"],
            val_size=self.params["val_size"],
            refit=cfg["refit"],
        )
        if transform == "level":
            return cv
        if not isinstance(cv, pl.DataFrame):
            cv = pl.from_pandas(cv.reset_index())
        cv = cv.with_columns(pl.col("ds", "cutoff").cast(pl.Date))
        levels = frame.select(*KEYS).with_columns(pl.col("ds").cast(pl.Date))
        return reconstruct_levels(cv, levels, self.model_columns)
