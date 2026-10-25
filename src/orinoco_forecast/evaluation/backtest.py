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
STAGES = ("A", "B")
BASELINES = "baselines"


def predictions_dir() -> Path:
    """Carpeta raíz de predicciones (``data/predictions``)."""
    return repo_root() / "data" / "predictions"


def stage_dir(root: str | Path, name: str, stage: str) -> Path:
    """``<root>/<name>/stage_<A|B>``: cada etapa guarda sus propias salidas."""
    if stage not in STAGES:
        raise ValueError(f"Etapa desconocida: {stage!r} (use 'A' o 'B').")
    return Path(root) / name / f"stage_{stage}"


def stage_baseline_metrics(
    root: str | Path, stage: str, final_test: bool = False
) -> pl.DataFrame:
    """Métricas de los baselines de la MISMA etapa (para ``rel_mae``).

    Nunca usa otra etapa ni archivos antiguos en la raíz del modelo: cada etapa
    construye su propia tabla (la B empieza en 1981), así que mezclar etapas en
    silencio podría comparar contra un Naive evaluado sobre otra serie.
    ``FileNotFoundError`` si no existen.
    """
    out = stage_dir(root, BASELINES, stage)
    path = (
        out / "final_test" / "metrics.parquet"
        if final_test
        else out / "metrics.parquet"
    )
    if not path.is_file():
        block = "del test final" if final_test else "de validación"
        flag = " --final-test" if final_test else ""
        raise FileNotFoundError(
            f"Faltan las métricas {block} de los baselines de la etapa {stage} "
            f"({path}). Ejecute primero `uv run python scripts/run_model.py "
            f"baselines --stage {stage}{flag}`."
        )
    return pl.read_parquet(path)


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


def tuning_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """``cfg`` con ``n_windows`` y ``step_size`` del bloque ``tuning``.

    Horizonte y política de reentrenamiento son los mismos del protocolo.
    """
    return {**cfg, **cfg["tuning"]}


def tuning_frame(frame: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Filas de desarrollo hasta el primer corte de validación (inclusive).

    Las ventanas de ajuste terminan en ese corte: los hiperparámetros se eligen
    sin ver las ventanas de validación que comparan modelos ni el test final.
    """
    dev = development_frame(frame, cfg)
    return dev.filter(pl.col("ds") <= cutoffs(dev, cfg)[0])


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


def predict_cv(
    name: str, runner: ForecastRunner, data: pl.DataFrame, run_cfg: dict[str, Any]
) -> pl.DataFrame:
    """Validación cruzada de ``runner`` sobre ``data`` sin escribir archivos.

    Verifica antes que no haya imputaciones cerca de los cortes
    (``assert_no_imputation_near_cutoffs``) y después que el runner haya usado
    exactamente los cortes esperados.
    """
    expected = cutoffs(data, run_cfg)
    assert_no_imputation_near_cutoffs(data, expected, run_cfg["horizon"])
    cv = _normalise(runner.cross_validation(data, run_cfg), runner.model_columns)
    got = sorted(cv["cutoff"].unique().to_list())
    if got != expected:
        raise ValueError(
            f"{name}: el runner usó cortes distintos a los del protocolo "
            f"({len(got)} vs {len(expected)} esperados)."
        )
    return cv


def run_cv(
    name: str,
    runner: ForecastRunner,
    frame: pl.DataFrame,
    cfg: dict[str, Any],
    final_test: bool = False,
    output_root: str | Path | None = None,
    stage: str = "A",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ejecuta la validación cruzada de ``runner`` y guarda predicciones y métricas.

    - Por defecto solo usa filas anteriores a ``test_start`` y escribe
      ``<output_root>/<name>/stage_<stage>/cv.parquet`` y ``metrics.parquet``
      (``stage`` solo elige la carpeta; ``frame`` ya debe ser de esa etapa).
    - ``final_test=True`` evalúa el bloque de test final (ver
      ``final_test_setup``) y escribe en ``stage_<stage>/final_test/``. Debe
      usarse una sola vez, con el modelo ya elegido.
    - Antes de entrenar verifica que no haya valores imputados cerca de ningún
      corte (``assert_no_imputation_near_cutoffs``) y, después, que el runner
      haya usado exactamente esos cortes.

    ``output_root`` por defecto es ``data/predictions``.
    """
    out_dir = stage_dir(output_root or predictions_dir(), name, stage)
    if final_test:
        out_dir = out_dir / "final_test"
        data, run_cfg = final_test_setup(frame, cfg)
        train = development_frame(frame, cfg)
    else:
        data, run_cfg = development_frame(frame, cfg), cfg
        train = data
    cv = predict_cv(name, runner, data, run_cfg)
    metrics = evaluate_cv(cv, runner.model_columns, train)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv.write_parquet(out_dir / "cv.parquet")
    metrics.write_parquet(out_dir / "metrics.parquet")
    return cv, metrics
