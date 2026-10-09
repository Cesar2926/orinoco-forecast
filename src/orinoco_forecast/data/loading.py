"""TODO: implement read-only Excel/CSV loading configured by ``configs/data.yaml``."""

from pathlib import Path
from typing import Any

import polars as pl


def load_raw_excel(
    path: str | Path, sheet_name: str | int | None = None, **kwargs: Any
) -> pl.DataFrame:
    """Placeholder for raw Excel reading and temporal-order validation."""
    raise NotImplementedError("TODO")


def load_source(name: str) -> pl.DataFrame:
    """Placeholder for loading a named source from the data YAML."""
    raise NotImplementedError("TODO")
