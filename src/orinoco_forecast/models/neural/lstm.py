"""NsmtRunner (LSTM) structure only; no network or training code yet."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Self

import polars as pl
import yaml

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.models.base import (
    DEFAULT_HORIZON,
    PredictionResult,
    validate_horizon,
    validate_windows,
)


def load_lstm_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs/models/lstm.yaml`` (or ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "models" / "lstm.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class NsmtRunner:
    """LSTM runner skeleton; the model and training loop are deliberately absent."""

    name = "nsmt"
    implemented = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or load_lstm_config()

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
        raise NotImplementedError(
            "El runner nsmt (LSTM) solo tiene estructura; falta implementar la "
            "red y el entrenamiento."
        )

    def predict(
        self,
        frame: pl.DataFrame,
        *,
        cutoff: date,
        horizon: int = DEFAULT_HORIZON,
    ) -> PredictionResult:
        validate_horizon(horizon)
        raise NotImplementedError(
            "El runner nsmt (LSTM) solo tiene estructura; falta implementar la "
            "red y el entrenamiento."
        )
