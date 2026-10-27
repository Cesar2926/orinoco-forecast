"""Output helpers for derived artifacts.

Prediction and statistics files live under ``data/predictions`` and are never
written into ``data/raw``. Importing this module has no side effects.
"""

from pathlib import Path

import polars as pl

from orinoco_forecast.data.loading import repo_root


def predictions_dir() -> Path:
    """Root folder for runner outputs (``data/predictions``)."""
    return repo_root() / "data" / "predictions"


def resolve_output_path(
    path: str | Path | None,
    default: str | Path,
) -> Path:
    """Resolve ``path`` against ``default``.

    - ``None`` returns ``default``.
    - A directory (existing or without a suffix) appends the default filename.
    - Any other path is used exactly as the target file.
    """
    default_path = Path(default)
    if path is None:
        return default_path
    candidate = Path(path)
    if candidate.is_dir() or candidate.suffix == "":
        return candidate / default_path.name
    return candidate


def write_parquet(frame: pl.DataFrame, path: str | Path) -> Path:
    """Write ``frame`` as Parquet, creating parent folders, and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    return target
