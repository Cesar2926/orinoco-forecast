"""TransformerRunner (TFT) structure only; the network is not implemented."""

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


def load_transformer_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs/models/transformer.yaml`` (or ``path``)."""
    path = (
        Path(path) if path else repo_root() / "configs" / "models" / "transformer.yaml"
    )
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TransformerRunner:
    """TFT runner skeleton; only the configuration contract exists."""

    name = "transformer"
    implemented = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or load_transformer_config()

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
            "El runner transformer (TFT) solo tiene estructura; falta "
            "implementar la red y el entrenamiento."
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
            "El runner transformer (TFT) solo tiene estructura; falta "
            "implementar la red y el entrenamiento."
        )
