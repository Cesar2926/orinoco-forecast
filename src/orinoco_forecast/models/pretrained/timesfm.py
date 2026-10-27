"""TimesFM3Runner structure only; inference depends on the optional extra."""

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


def load_timesfm_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs/models/timesfm.yaml`` (or ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "models" / "timesfm.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class TimesFM3Runner:
    """TimesFM 3.0 runner skeleton; weights are optional and non-commercial."""

    name = "timesfm3"
    implemented = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = params or load_timesfm_config()

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
            "El runner timesfm3 solo tiene estructura; falta instalar el extra "
            "'timesfm' e implementar la inferencia."
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
            "El runner timesfm3 solo tiene estructura; falta instalar el extra "
            "'timesfm' e implementar la inferencia."
        )
