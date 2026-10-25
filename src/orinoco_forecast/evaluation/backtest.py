"""Rolling-origin cross-validation shared by every model (``configs/backtest.yaml``).

Windows follow Nixtla's ``cross_validation`` convention: the last window ends on
the last date of the frame and earlier cutoffs are ``step_size`` days apart.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.evaluation.metrics import evaluate_cv
from orinoco_forecast.features.dataset import assert_no_imputation_near_cutoffs
from orinoco_forecast.models.base import ForecastRunner

CV_KEYS = ["unique_id", "ds", "cutoff", "y"]


def load_backtest_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lee ``configs/backtest.yaml`` (o ``path``)."""
    path = Path(path) if path else repo_root() / "configs" / "backtest.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _test_start(cfg: dict[str, Any]) -> date:
    return date.fromisoformat(str(cfg["test_start"]))


def development_frame(frame: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Filas anteriores a ``test_start`` (entrenamiento + validación)."""
    return frame.filter(pl.col("ds") < _test_start(cfg))


def cutoffs(frame: pl.DataFrame, cfg: dict[str, Any]) -> list[date]:
    """Los ``n_windows`` cortes que usará Nixtla sobre ``frame``, en orden.

    Último corte = última fecha - ``horizon``; los anteriores retroceden
    ``step_size`` días. Asume que todas las series terminan el mismo día.
    """
    h, step, n = cfg["horizon"], cfg["step_size"], cfg["n_windows"]
    last = frame["ds"].max() - timedelta(days=h)
    return [last - timedelta(days=step * (n - 1 - i)) for i in range(n)]


def final_test_setup(
    frame: pl.DataFrame, cfg: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Recorta ``frame`` y adapta ``cfg`` para cubrir el test final con ventanas.

    Ventanas consecutivas de ``horizon`` días (``step_size = horizon``) desde
    ``test_start``; el primer corte es el último día de desarrollo, así que el
    modelo se entrena con todos los datos de desarrollo. Los días finales que no
    completan una ventana se descartan.
    """
    h, start = cfg["horizon"], _test_start(cfg)
    n_days = (frame["ds"].max() - start).days + 1
    n_windows = n_days // h
    if n_windows < 1:
        raise ValueError("El bloque de test final es más corto que el horizonte.")
    end = start + timedelta(days=n_windows * h - 1)
    return frame.filter(pl.col("ds") <= end), {
        **cfg,
        "n_windows": n_windows,
        "step_size": h,
    }


def _normalise(raw: Any, models: list[str]) -> pl.DataFrame:
    out = raw if isinstance(raw, pl.DataFrame) else pl.from_pandas(raw.reset_index())
    out = out.with_columns(pl.col("ds", "cutoff").cast(pl.Date))
    return out.select(*CV_KEYS, *models).sort("unique_id", "cutoff", "ds")


def run_cv(
    name: str,
    runner: ForecastRunner,
    frame: pl.DataFrame,
    cfg: dict[str, Any],
    final_test: bool = False,
    output_root: str | Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ejecuta la validación cruzada de ``runner`` y guarda predicciones y métricas.

    - Por defecto solo usa filas anteriores a ``test_start`` y escribe
      ``<output_root>/<name>/cv.parquet`` y ``metrics.parquet``.
    - ``final_test=True`` evalúa el bloque de test final (ver
      ``final_test_setup``) y escribe en ``<name>/final_test/``. Debe usarse una
      sola vez, con el modelo ya elegido.
    - Antes de entrenar verifica que no haya valores imputados cerca de ningún
      corte (``assert_no_imputation_near_cutoffs``) y, después, que el runner
      haya usado exactamente esos cortes.

    ``output_root`` por defecto es ``data/predictions``.
    """
    if final_test:
        data, run_cfg = final_test_setup(frame, cfg)
        train = development_frame(frame, cfg)
    else:
        data, run_cfg = development_frame(frame, cfg), cfg
        train = data
    expected = cutoffs(data, run_cfg)
    assert_no_imputation_near_cutoffs(data, expected, run_cfg["horizon"])

    cv = _normalise(runner.cross_validation(data, run_cfg), runner.model_columns)
    got = sorted(cv["cutoff"].unique().to_list())
    if got != expected:
        raise ValueError(
            f"{name}: el runner usó cortes distintos a los del protocolo "
            f"({len(got)} vs {len(expected)} esperados)."
        )
    metrics = evaluate_cv(cv, runner.model_columns, train)

    root = Path(output_root) if output_root else repo_root() / "data" / "predictions"
    out_dir = root / name / "final_test" if final_test else root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cv.write_parquet(out_dir / "cv.parquet")
    metrics.write_parquet(out_dir / "metrics.parquet")
    return cv, metrics
