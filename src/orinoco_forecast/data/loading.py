"""TODO: implement read-only Excel/CSV loading configured by ``configs/data.yaml``."""

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw_excel(
    path: str | Path, sheet_name: str | int | None = None, **kwargs: Any
) -> pd.DataFrame:
    """Placeholder for raw Excel reading and temporal-index validation."""
    raise NotImplementedError("TODO")


def load_source(name: str) -> pd.DataFrame:
    """Placeholder for loading a named source from the data YAML."""
    raise NotImplementedError("TODO")
