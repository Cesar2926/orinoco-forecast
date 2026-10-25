"""TCN (temporal convolutional network) with neuralforecast, one model per seed.

Hyperparameters live only in ``configs/models/cnn.yaml``; the protocol
(``horizon``, ``step_size``, ``n_windows``, ``refit``) in ``configs/backtest.yaml``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml
from neuralforecast import NeuralForecast
from neuralforecast.losses import pytorch as losses
from neuralforecast.models import TCN

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.features.dataset import NON_FEATURE_COLUMNS

KEYS = ("unique_id", "ds", "y")
# Fourier terms from features.dataset.fourier_terms: deterministic, known ahead.
FUTURE_PREFIXES = ("sin_", "cos_")
MODEL = "TCN"


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
    """

    name = "tcn"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or load_cnn_config()

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

    def cross_validation(
        self, frame: pl.DataFrame, cfg: dict[str, Any]
    ) -> pl.DataFrame:
        futr, hist = split_covariates(frame)
        nf = NeuralForecast(
            models=self._models(cfg["horizon"], futr, hist),
            # Polars input needs a polars offset ("1d"), not pandas "D".
            freq="1d",
        )
        return nf.cross_validation(
            df=frame.select(*KEYS, *futr, *hist),
            n_windows=cfg["n_windows"],
            step_size=cfg["step_size"],
            val_size=self.params["val_size"],
            refit=cfg["refit"],
        )
