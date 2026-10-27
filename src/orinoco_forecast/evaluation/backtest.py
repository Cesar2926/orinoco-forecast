"""Shared execution protocol (``configs/backtest.yaml``).

Only the values used by the runner execution path live here: horizon, input
length, test start and seasonality. The legacy cross-validation protocol was
removed with the old runners.
"""

from pathlib import Path
from typing import Any

import yaml

from orinoco_forecast.data.loading import repo_root


def load_backtest_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read ``configs/backtest.yaml`` (or ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "backtest.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
