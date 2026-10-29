"""TimesFM 3.0 zero-shot runner (Google's pretrained time-series foundation model).

Hyperparameters live only in ``configs/models/timesfm.yaml``. Nothing is trained
on Orinoco data: ``fit`` only loads the pretrained weights and ``predict`` feeds
each series' history up to the cutoff as context. Needs the optional extra
(``uv sync --extra timesfm``); the ``google/timesfm-3.0-pytorch`` weights are
non-commercial / non-production.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
import yaml

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.features.dataset import NON_FEATURE_COLUMNS, drop_covariates
from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    PredictionResult,
    validate_horizon,
    validate_windows,
)

KEYS = ("unique_id", "ds", "y")
MODEL = "TIMESFM3"
# The backtest builds a new runner per fold; weights are loaded once per process.
_FORECASTERS: dict[tuple[str, str], Any] = {}


def load_timesfm_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs/models/timesfm.yaml`` (or ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "models" / "timesfm.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_forecaster(checkpoint: str, device: str | None) -> Any:
    """Pretrained ``TimesFM3Forecaster``, cached per checkpoint and device."""
    key = (checkpoint, str(device))
    if key not in _FORECASTERS:
        try:
            from timesfm import TimesFM3Forecaster
        except ImportError as error:
            raise ImportError(
                "TimesFM 3 requiere el extra opcional: uv sync --extra timesfm."
            ) from error
        _FORECASTERS[key] = TimesFM3Forecaster.from_pretrained(
            checkpoint, device=device
        )
    return _FORECASTERS[key]


class TimesFM3Runner:
    """TimesFM 3.0 zero-shot sobre cada serie del frame (sin entrenamiento).

    - Contexto: los últimos ``context_len`` días de ``y`` hasta el corte.
    - ``use_covariates``: además pasa como ``past_only_covariates`` las columnas
      que deja ``drop_covariates`` (mismo criterio que las redes), del mismo
      tramo; ninguna se usa después del corte.
    - Columna ``TIMESFM3``; la carpeta es ``timesfm3``.
    """

    name = "timesfm3"
    implemented = True

    def __init__(
        self, params: dict[str, Any] | None = None, forecaster: Any = None
    ) -> None:
        self.params = params or load_timesfm_config()
        self._forecaster = forecaster

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
        if self._forecaster is None:
            device = self.params.get("device", "auto")
            self._forecaster = load_forecaster(
                self.params["checkpoint"], None if device == "auto" else device
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
        if self._forecaster is None:
            raise RuntimeError("El runner no está ajustado; llama a fit primero.")
        p = self.params
        history = drop_covariates(
            frame.filter(pl.col("ds") <= cutoff).sort("unique_id", "ds"),
            p.get("drop_covariates", []),
        )
        covariates = [
            c for c in history.columns if c not in {*KEYS, *NON_FEATURE_COLUMNS}
        ]
        ids, contexts, past = [], [], []
        for (uid,), group in history.group_by("unique_id", maintain_order=True):
            if group["ds"].max() != cutoff:
                raise ValueError(f"La serie {uid!r} no llega al corte {cutoff}.")
            tail = group.tail(int(p["context_len"]))
            ids.append(uid)
            contexts.append(tail["y"].to_numpy().astype(np.float32))
            if p.get("use_covariates", False) and covariates:
                past.append(tail.select(covariates).to_numpy().T.astype(np.float32))
        outputs = list(
            self._forecaster.predict_batch(
                contexts,
                horizon,
                past_only_covariates=past or None,
                use_symmetric_averaging=bool(p.get("use_symmetric_averaging", False)),
            )
        )
        dates = [cutoff + timedelta(days=step) for step in range(1, horizon + 1)]
        long = pl.DataFrame(
            {
                "model": [MODEL] * (len(ids) * horizon),
                "unique_id": [uid for uid in ids for _ in range(horizon)],
                "cutoff": [cutoff] * (len(ids) * horizon),
                "ds": dates * len(ids),
                "horizon_step": list(range(1, horizon + 1)) * len(ids),
                "y_pred": [
                    float(v)
                    for out in outputs
                    for v in np.asarray(out.forecast).reshape(-1)[:horizon]
                ],
            }
        ).with_columns(pl.col("horizon_step").cast(pl.Int32))
        return PredictionResult.from_frame(long, runner=self.name, horizon=horizon)
