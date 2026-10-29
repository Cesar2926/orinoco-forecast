"""Generador de reportes de validación y evaluación por ventanas de 14 días.

Uso:
  1) Resumen oficial sobre predicciones ya guardadas (24 cortes walk-forward):
     ``uv run python scripts/make_report.py``

  2) Evaluación en ventanas móviles de 14 días (ej. 5 ventanas de 14 días):
     ``uv run python scripts/make_report.py --eval val --windows 5``
     ``uv run python scripts/make_report.py --eval test --windows 5 --fit-once --fast``
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from orinoco_forecast.dashboard import data as dash_data
from orinoco_forecast.data.datasets import DEFAULT_OUTPUT
from orinoco_forecast.data.loading import repo_root
from orinoco_forecast.data.windows import build_windows
from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.evaluation.execution import (
    FEATURE_COLUMNS,
    INPUT_LENGTH,
    TARGET_STATIONS,
    load_history,
)
from orinoco_forecast.features.dataset import build_model_frame, load_levels
from orinoco_forecast.models.registry import get_runner

BASELINES = "naive"
SEED_PATTERN = r"^(.+)_s\d+$"
FIT_ONCE_NEURAL = frozenset({"cnn_lstm", "transformer", "timesfm3"})


def _family_expr() -> pl.Expr:
    return pl.coalesce(pl.col("model").str.extract(SEED_PATTERN), pl.col("model"))


def rank_models(comparison: pl.DataFrame) -> pl.DataFrame:
    """Clasifica familias de modelos según criterio del README.

    1) Filtra ``horizon == 'all'`` y ``metric == 'mae'`` cuando existen.
    2) Promedia semillas (``_s1.._s3``) dentro de cada ``(unique_id, family)``.
    3) Exige ``rel_mae < 1`` en todas las estaciones objetivo (``beats_naive_all``)
       y ordena por menor ``mean_mae`` promedio entre estaciones.
    """
    if comparison.is_empty():
        return pl.DataFrame()
    df = comparison
    if "horizon" in df.columns:
        df = df.filter(pl.col("horizon") == "all")
    if "metric" in df.columns:
        df = df.filter(pl.col("metric") == "mae")
    df = df.filter(pl.col("model") != BASELINES)
    if df.is_empty():
        return pl.DataFrame()

    mae_col = "mean_model" if "mean_model" in df.columns else "mae"
    if "rel_mae" not in df.columns and {"mean_model", "mean_baseline"} <= set(
        df.columns
    ):
        df = df.with_columns(rel_mae=pl.col("mean_model") / pl.col("mean_baseline"))
    df = df.with_columns(family=_family_expr())

    per_station = (
        df.group_by("unique_id", "family")
        .agg(
            seeds=pl.len(),
            mae=pl.col(mae_col).mean(),
            rel_mae=pl.col("rel_mae").mean(),
        )
        .rename({"family": "model"})
    )
    return (
        per_station.group_by("model")
        .agg(
            stations=pl.col("unique_id").n_unique(),
            beats_naive_all=(pl.col("rel_mae") < 1.0).all(),
            mae_ciudad_bolivar=pl.col("mae")
            .filter(pl.col("unique_id") == "ciudad_bolivar")
            .first(),
            mae_palua=pl.col("mae").filter(pl.col("unique_id") == "palua").first(),
            mean_mae=pl.col("mae").mean(),
            mean_rel_mae=pl.col("rel_mae").mean(),
        )
        .sort(
            pl.col("beats_naive_all").fill_null(False),
            "mean_mae",
            descending=[True, False],
        )
    )


def build_eval_cutoffs(
    dates: Sequence[date],
    *,
    mode: str,
    test_start: date,
    horizon: int = 14,
    n_windows: int = 5,
    step_days: int = 14,
    val_start: date = date(2021, 1, 1),
) -> list[date]:
    """Construye fechas de corte para evaluar ventanas de ``horizon`` días.

    - ``mode='val'``: cortes cuya ventana completa termina antes de ``test_start``
      (``cutoff <= test_start - horizon - 1``). Con ``n_windows > 0`` toma las
      últimas ``n_windows`` ventanas separadas por ``step_days``; con ``0``,
      todas desde ``val_start``.
    - ``mode='test'``: cortes desde ``test_start`` hacia adelante separados por
      ``step_days`` (hasta ``n_windows`` si ``> 0``, o todo el bloque de test si
      ``0``).
    """
    if mode not in ("val", "test"):
        raise ValueError(f"mode debe ser 'val' o 'test', no {mode!r}.")
    if horizon < 1 or step_days < 1 or n_windows < 0:
        raise ValueError(
            "horizon >= 1, step_days >= 1 y n_windows >= 0 son requeridos."
        )
    date_set = set(dates)
    if not date_set:
        raise ValueError("dates no puede estar vacío.")
    sorted_dates = sorted(date_set)

    def has_full_horizon(cut: date) -> bool:
        return cut in date_set and all(
            (cut + timedelta(days=step)) in date_set for step in range(1, horizon + 1)
        )

    if mode == "val":
        latest_bound = test_start - timedelta(days=horizon + 1)
        eligible = [
            d for d in sorted_dates if d <= latest_bound and has_full_horizon(d)
        ]
        if not eligible:
            raise ValueError(
                f"No hay cortes de validación completos antes de {test_start}."
            )
        latest = eligible[-1]
        if n_windows == 0:
            start = max(val_start, eligible[0])
            cutoffs: list[date] = []
            cur = start
            while cur <= latest:
                if has_full_horizon(cur):
                    cutoffs.append(cur)
                cur += timedelta(days=step_days)
        else:
            candidates = [
                latest - timedelta(days=i * step_days) for i in range(n_windows)
            ]
            cutoffs = sorted(c for c in candidates if has_full_horizon(c))
    else:
        eligible = [d for d in sorted_dates if d >= test_start and has_full_horizon(d)]
        if not eligible:
            raise ValueError(f"No hay cortes de test completos desde {test_start}.")
        earliest = eligible[0]
        latest = eligible[-1]
        cutoffs = []
        cur = earliest
        while cur <= latest and (n_windows == 0 or len(cutoffs) < n_windows):
            if has_full_horizon(cur):
                cutoffs.append(cur)
            cur += timedelta(days=step_days)

    if not cutoffs:
        raise ValueError(f"No se pudieron construir cortes para mode={mode!r}.")
    return cutoffs


def _prepare_runner_history(
    frame: pl.DataFrame, model_name: str, cutoff: date
) -> pl.DataFrame:
    history = frame.filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")
    if (
        model_name == "arimax"
        and not any("_lag_" in c for c in history.columns)
        and DEFAULT_OUTPUT.is_file()
    ):
        history = load_history().filter(pl.col("ds") <= cutoff).sort("unique_id", "ds")
    return history


def _windows_for(
    history: pl.DataFrame, target_ids: list[str], cutoff: date, horizon: int
) -> pl.DataFrame:
    targets = history.filter(pl.col("unique_id").is_in(target_ids))
    feature_cols = [c for c in FEATURE_COLUMNS if c in targets.columns] or ["y"]
    max_len = max(1, targets.height // max(1, len(target_ids)) - horizon - 1)
    return build_windows(
        targets,
        cutoff=cutoff,
        input_length=min(INPUT_LENGTH, max_len),
        horizon=horizon,
        feature_columns=feature_cols,
    )


def evaluate_windows(
    frame: pl.DataFrame,
    models: Sequence[str],
    cutoffs: Sequence[date],
    *,
    horizon: int = 14,
    fit_once: bool = False,
    fast_mode: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Evalúa ``models`` (más ``naive``) en ``cutoffs`` y devuelve resumen y ranking."""
    if not cutoffs:
        raise ValueError("cutoffs no puede estar vacío.")
    ordered_cutoffs = list(cutoffs)
    runner_names = list(dict.fromkeys([*models, BASELINES]))
    present_ids = set(frame["unique_id"].unique().to_list())
    target_ids = [u for u in TARGET_STATIONS if u in present_ids] or sorted(present_ids)
    observed = frame.filter(pl.col("unique_id").is_in(target_ids)).select(
        "unique_id", "ds", "y"
    )

    pred_frames: list[pl.DataFrame] = []
    for name in runner_names:
        runner = get_runner(name)
        if (
            fast_mode
            and hasattr(runner, "params")
            and isinstance(runner.params, dict)
            and runner.params.get("seeds")
        ):
            runner.params = {**runner.params, "seeds": [runner.params["seeds"][0]]}
        fitted = False
        for cut in ordered_cutoffs:
            history = _prepare_runner_history(frame, name, cut)
            if not fitted or not fit_once:
                windows = _windows_for(history, target_ids, cut, horizon)
                runner.fit(windows, history, cutoff=cut, horizon=horizon)
                fitted = True
            elif name in FIT_ONCE_NEURAL:
                pass
            elif hasattr(runner, "_engine"):
                runner._engine.fit(history, cutoff=cut)
            else:
                windows = _windows_for(history, target_ids, cut, horizon)
                runner.fit(windows, history, cutoff=cut, horizon=horizon)
            res = runner.predict(history, cutoff=cut, horizon=horizon)
            pred_frames.append(
                res.frame.filter(pl.col("unique_id").is_in(target_ids)).select(
                    "unique_id",
                    "ds",
                    "cutoff",
                    "model",
                    pl.col("y_pred").alias("y_hat"),
                )
            )

    all_preds = (
        pl.concat(pred_frames)
        .join(observed, on=["unique_id", "ds"], how="left")
        .with_columns(
            err=(pl.col("y_hat") - pl.col("y")).abs(),
            sq_err=(pl.col("y_hat") - pl.col("y")).pow(2),
        )
    )
    per_station = (
        all_preds.group_by("unique_id", "model")
        .agg(
            n_windows=pl.col("cutoff").n_unique(),
            n_days=pl.len(),
            mae=pl.col("err").mean(),
            rmse=pl.col("sq_err").mean().sqrt(),
        )
        .with_columns(
            horizon=pl.lit("all"),
            metric=pl.lit("mae"),
            family=_family_expr(),
        )
    )
    naive_mae = per_station.filter(pl.col("model") == BASELINES).select(
        "unique_id", pl.col("mae").alias("mean_baseline")
    )
    per_station = (
        per_station.join(naive_mae, on="unique_id", how="left")
        .with_columns(
            mean_model=pl.col("mae"),
            rel_mae=pl.col("mae") / pl.col("mean_baseline"),
            beats_naive=pl.col("mae") < pl.col("mean_baseline"),
        )
        .sort("unique_id", "mae")
    )
    ranking = rank_models(per_station)
    return per_station, ranking


def _table_config() -> pl.Config:
    return pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_hide_dataframe_shape=True,
        tbl_width_chars=200,
        tbl_formatting="ASCII_FULL_CONDENSED",
        float_precision=4,
    )


def _save_output(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.write_csv(path)
    else:
        df.write_parquet(path)


def _print_banner(ranking: pl.DataFrame, context: str) -> None:
    if ranking.is_empty():
        print(f"Sin modelos para seleccionar ganador ({context}).")
        return
    winner = ranking.row(0, named=True)
    print(
        f"\nGanador ({context}): {winner['model']} "
        f"(MAE medio = {winner['mean_mae']:.4f} m, "
        f"rel_mae medio = {winner['mean_rel_mae']:.3f})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Carpeta con predicciones guardadas (data/predictions o respaldo).",
    )
    parser.add_argument(
        "--eval",
        choices=("val", "test"),
        default=None,
        help="Evalúa ventanas móviles de 14 días en lugar de solo leer parquets.",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=5,
        help="Número de ventanas de 14 días con --eval (5 por defecto; 0 = todas).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=14,
        help="Paso en días entre ventanas consecutivas con --eval (14 por defecto).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Días a pronosticar por ventana (por defecto: 14).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Runners a evaluar con --eval (por defecto: todos los implementados).",
    )
    parser.add_argument(
        "--fit-once",
        action="store_true",
        help="En --eval, entrena una vez los modelos neuronales y predice cada corte.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="En --eval, usa 1 semilla en modelos neuronales para 3x mayor velocidad.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional (.csv o .parquet) para exportar la tabla de ranking.",
    )
    args = parser.parse_args(argv)

    if args.eval is None:
        root = args.predictions_dir or dash_data.predictions_root()
        models = dash_data.discover_models(root)
        fold_comp = dash_data.load_fold_comparison(root, models)
        if not fold_comp.is_empty():
            ranking = rank_models(fold_comp)
            station_view = (
                fold_comp.filter(
                    (pl.col("horizon") == "all") & (pl.col("metric") == "mae")
                )
                .select(
                    "unique_id",
                    "model",
                    "family",
                    "n_folds",
                    "mean_model",
                    "std_model",
                    "mean_baseline",
                    "rel_mae",
                    "wins",
                    "losses",
                    "win_rate",
                    "sign_test_p",
                )
                .sort("unique_id", "mean_model")
            )
            with _table_config():
                print(
                    "=== Validación Walk-Forward (24 cortes, h=1..14) por estación ==="
                )
                print(station_view)
                print("\n=== Ranking consolidado por familia de modelo (24 cortes) ===")
                print(ranking)
            _print_banner(ranking, "24 cortes walk-forward")
            if args.output:
                _save_output(ranking, args.output)
            return 0

        metrics = dash_data.load_metrics(root, models)
        ranking = rank_models(metrics)
        with _table_config():
            print("=== Validación de corte único (h=1..14) ===")
            print(
                metrics.filter(pl.col("horizon") == "all").sort("unique_id", "mae")
                if not metrics.is_empty()
                else metrics
            )
            print("\n=== Ranking consolidado (corte único) ===")
            print(ranking)
        _print_banner(ranking, "corte único")
        if args.output:
            _save_output(ranking, args.output)
        return 0

    cfg = load_backtest_config()
    test_start = date.fromisoformat(str(cfg["test_start"]))
    exog_path = repo_root() / "data" / "processed" / "orinoco_exogenous.parquet"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if DEFAULT_OUTPUT.is_file():
            # Same dataset (with lag features) as run_model.py, so --eval
            # reproduces the validated runs exactly.
            frame = load_history()
        else:
            levels = load_levels()
            frame = build_model_frame(
                stage="B" if exog_path.is_file() else "A", levels=levels
            )
    dates = (
        frame.filter(pl.col("unique_id") == TARGET_STATIONS[0])["ds"]
        .unique()
        .sort()
        .to_list()
    )
    cutoffs = build_eval_cutoffs(
        dates,
        mode=args.eval,
        test_start=test_start,
        horizon=args.horizon,
        n_windows=args.windows,
        step_days=args.step_days,
    )
    chosen_models = args.models or [
        m for m in dash_data.live_runners() if m != BASELINES
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        per_station, ranking = evaluate_windows(
            frame,
            chosen_models,
            cutoffs,
            horizon=args.horizon,
            fit_once=args.fit_once,
            fast_mode=args.fast,
        )
    context = (
        f"{args.eval}: {len(cutoffs)} ventanas de {args.horizon}d "
        f"({cutoffs[0]} .. {cutoffs[-1]})"
    )
    with _table_config():
        print(f"=== Evaluación ({context}) por estación ===")
        print(
            per_station.select(
                "unique_id",
                "model",
                "family",
                "n_windows",
                "n_days",
                "mae",
                "rmse",
                "mean_baseline",
                "rel_mae",
                "beats_naive",
            )
        )
        print(f"\n=== Ranking consolidado ({context}) ===")
        print(ranking)
    _print_banner(ranking, context)
    if args.output:
        _save_output(ranking, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
