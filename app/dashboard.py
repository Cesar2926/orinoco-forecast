"""Tablero interactivo del proyecto Orinoco Forecast.

Uso: ``uv run streamlit run app/dashboard.py`` (o ``make dashboard``).
Lee los niveles de ``data/raw`` y las predicciones exportadas en
``data/predictions/<runner>/`` (con respaldo en ``reports/predictions/``),
incluyendo los 24 cortes walk-forward, y permite probar cualquier runner
implementado bajo demanda (con simulador "¿Qué pasaría si…?" en memoria).
``ORINOCO_PREDICTIONS_DIR`` cambia la carpeta de predicciones (pruebas).
"""

import warnings
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import streamlit as st

from orinoco_forecast.dashboard import charts, data
from orinoco_forecast.data.loading import load_config, repo_root
from orinoco_forecast.evaluation.backtest import load_backtest_config
from orinoco_forecast.features.dataset import build_model_frame, load_levels

EXOGENOUS = repo_root() / "data" / "processed" / "orinoco_exogenous.parquet"
MODEL_HINTS = ("naive", "seasonal_naive", "arimax", "booster", "tcn")
STATION_LABELS = {
    "ciudad_bolivar": "Ciudad Bolívar (ciudad_bolivar)",
    "palua": "Palúa (palua)",
    "ayacucho": "Puerto Ayacucho (ayacucho, aguas arriba)",
    "caicara": "Caicara del Orinoco (caicara, aguas arriba)",
}

st.set_page_config(page_title="Orinoco Forecast", layout="wide")


@st.cache_data(show_spinner="Cargando niveles de data/raw…")
def cached_levels() -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Already audited in data/raw/README.md.
        return load_levels()


def _stamp(root: Path) -> tuple[tuple[str, float], ...]:
    """Fechas de modificación: invalidan la caché al regenerar predicciones."""
    files = sorted({*root.glob("*/*.parquet"), *root.glob("*/backtest/*/*.parquet")})
    return tuple((str(path), path.stat().st_mtime) for path in files)


@st.cache_data(show_spinner="Cargando predicciones…")
def cached_predictions(
    root: str, stamp: tuple
) -> tuple[list[str], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    path = Path(root)
    models = data.discover_models(path)
    levels = cached_levels()
    return (
        models,
        data.load_metrics(path, models),
        data.load_forecasts(path, models, levels),
        data.load_fold_comparison(path, models),
    )


@st.cache_data
def cached_optional(path: str, stamp: float) -> pl.DataFrame | None:
    return data.load_optional(Path(path))


@st.cache_data(show_spinner="Preparando tabla del modelo…")
def cached_model_frame() -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage = "B" if EXOGENOUS.is_file() else "A"
        return build_model_frame(stage=stage, levels=cached_levels())


@st.cache_data(show_spinner="Ejecutando pronóstico en vivo…")
def cached_live_forecast(
    model_name: str,
    cutoff: date,
    horizon: int,
    fast_mode: bool,
    station: str,
    level_delta_m: float,
    upstream_delta_m: float,
    rain_delta_mm: float,
) -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_frame = cached_model_frame()
        has_what_if = any(
            v != 0.0 for v in (level_delta_m, upstream_delta_m, rain_delta_mm)
        )
        wi_frame = (
            data.apply_what_if(
                base_frame,
                station,
                cutoff,
                level_delta_m=level_delta_m,
                upstream_delta_m=upstream_delta_m,
                rain_delta_mm=rain_delta_mm,
            )
            if has_what_if
            else None
        )
        return data.run_live_forecast(
            base_frame,
            cached_levels(),
            model_name,
            cutoff,
            horizon=horizon,
            fast_mode=fast_mode,
            what_if_frame=wi_frame,
        )


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def missing_predictions(models: tuple[str, ...] = MODEL_HINTS) -> None:
    commands = "\n".join(data.run_command(model) for model in models)
    st.info(f"No hay predicciones de runners en `{root}`. Genérelas primero:")
    st.code(commands, language="bash")


def metric_columns() -> dict:
    fmt = {
        m: st.column_config.NumberColumn(m.upper(), format="%.4f")
        for m in ("mae", "rmse", "mase", "nse", "kge")
    }
    return fmt | {
        "rel_mae": st.column_config.NumberColumn("rel_mae", format="%.3f"),
        "beats": st.column_config.CheckboxColumn("rel_mae < 1"),
    }


root = data.predictions_root()
cfg = load_backtest_config()
targets = load_config()["targets"]
levels = cached_levels()
stamp = _stamp(root)
models, metrics, forecasts, fold_comparison = cached_predictions(str(root), stamp)
st.sidebar.markdown(
    "**Guía rápida de estaciones y métricas**\n\n"
    "- `ciudad_bolivar`: **Ciudad Bolívar** (objetivo)\n"
    "- `palua`: **Palúa** (objetivo)\n"
    "- `ayacucho` / `caicara`: estaciones aguas arriba\n\n"
    "**Métricas:**\n"
    "- **MAE (m):** cuántos metros (o cm) se equivoca el pronóstico en promedio.\n"
    "- **rel_mae < 1:** mejora respecto a repetir el nivel de hoy (`naive`).\n"
    "- **NSE / KGE:** ajuste hidrológico (1.0 es perfecto)."
)

st.title("Orinoco Forecast")
st.caption(
    "Explora los niveles del río Orinoco, las métricas de validación (24 cortes "
    "walk-forward y corte final) y prueba pronósticos a 14 días bajo demanda. "
    f"El bloque de test final (desde {cfg['test_start']}) permanece reservado."
)
tab_data, tab_val, tab_window, tab_live = st.tabs(
    [
        "Datos",
        "Validación de modelos",
        "Pronóstico por ventana",
        "Probar predicción",
    ]
)

with tab_data:
    st.info(
        "**¿Qué muestra esta pestaña?** El historial diario del nivel del río Orinoco "
        "(en metros sobre el nivel del mar) en **Ciudad Bolívar** (`ciudad_bolivar`), "
        "**Palúa** (`palua`) y las estaciones aguas arriba (**Puerto Ayacucho** y "
        "**Caicara**), junto con la lluvia satelital (NASA POWER) y el índice "
        "climático El Niño / La Niña (ONI)."
    )
    stations = sorted(levels["unique_id"].unique())
    first, last = levels["ds"].min(), levels["ds"].max()
    left, right = st.columns([2, 1])
    chosen = left.multiselect(
        "Estaciones",
        stations,
        default=[s for s in targets if s in stations],
        format_func=lambda s: STATION_LABELS.get(s, s),
    )
    span = right.date_input(
        "Rango de fechas",
        value=(max(first, last - timedelta(days=3 * 365)), last),
        min_value=first,
        max_value=last,
    )
    start, end = span if len(span) == 2 else (first, last)
    view = data.filter_levels(levels, chosen, start, end)

    st.subheader("Nivel diario por estación")
    st.markdown(
        "Cada línea es el nivel del río (metros) en una estación. Los puntos de color "
        "marcan días cuya etiqueta de calidad no es `observed`: `suspect_flat` "
        "(valor repetido muchos días, probable dato no actualizado), `suspect_jump` "
        "(cambio diario mayor al umbral de `configs/data.yaml`) y `source_conflict` "
        "(la fuente primaria y la histórica difieren). Los días `missing` no tienen "
        "nivel y se dibujan como marcas en el borde inferior. Arrastre para moverse "
        "en el tiempo y use la rueda para acercar."
    )
    if view.is_empty():
        st.info("Elija al menos una estación y un rango con datos.")
    else:
        st.altair_chart(charts.level_chart(view), width="stretch")
    st.markdown(
        "**Días por etiqueta de calidad** (serie completa, sin filtrar por fecha). "
        "Las etiquetas solo informan: `build_model_frame` anula `suspect_flat` e "
        "interpola huecos cortos; el resto queda como está."
    )
    st.dataframe(data.quality_counts(levels), hide_index=True)

    st.subheader("Variables externas")
    exogenous = cached_optional(str(EXOGENOUS), _mtime(EXOGENOUS))
    if exogenous is None:
        st.info(
            "No existe `data/processed/orinoco_exogenous.parquet`. Para ver lluvia y "
            "ONI, constrúyalo con:"
        )
        st.code("uv run python scripts/build_exogenous_dataset.py", language="bash")
    else:
        window = pl.col("ds").is_between(start, end)
        st.markdown(
            "Lluvia diaria de NASA POWER (mm/día) en el punto de cada estación. "
            "Picos de lluvia aguas arriba suelen anticipar subidas del río con "
            "semanas de retraso."
        )
        rain = data.rain_long(exogenous).filter(window)
        st.altair_chart(charts.rain_chart(rain), width="stretch")
        st.markdown(
            "ONI (anomalía de temperatura del Pacífico, °C) ya desplazado por su "
            "retraso de publicación. Sobre +0,5 indica El Niño (suele traer menos "
            "lluvia en la cuenca); bajo −0,5, La Niña."
        )
        st.altair_chart(
            charts.oni_chart(data.oni_series(exogenous).filter(window)),
            width="stretch",
        )

with tab_val:
    if not models:
        missing_predictions()
    else:
        st.info(
            "**¿Cómo leer estas tablas?**\n"
            "- **Estaciones:** `ciudad_bolivar` = **Ciudad Bolívar**; "
            "`palua` = **Palúa**.\n"
            "- **MAE (m):** error absoluto medio en metros (ej. `0.28 m` = 28 cm "
            "de error promedio en los 14 días).\n"
            "- **rel_mae < 1:** indica que el modelo es más preciso que `naive` "
            "(repetir el nivel del último día conocido).\n"
            "- **NSE y KGE:** eficiencia hidrológica donde `1.0` es un ajuste perfecto."
        )
        rel_root = (
            root.relative_to(repo_root()) if root.is_relative_to(repo_root()) else root
        )
        st.markdown(
            f"Modelos con predicciones en `{rel_root}`: "
            f"{', '.join(f'`{m}`' for m in models)}. "
            f"Cada runner se evalúa en ventanas de 1..{cfg['horizon']} días "
            f"antes de {cfg['test_start']}."
        )
        if data.BASELINES not in models and (
            metrics.is_empty() or data.BASELINES not in metrics["model"].to_list()
        ):
            st.info("Sin el runner Naive no se puede comparar rel_mae. Ejecute:")
            st.code(data.run_command(data.BASELINES), language="bash")

        if not fold_comparison.is_empty():
            st.subheader("Resumen de 24 cortes (walk-forward 2021–2022)")
            st.markdown(
                "Benchmark oficial sobre los **24 cortes de validación (2021–2022)** "
                "frente a `naive`. Un modelo consistente mantiene `rel_mae < 1` y "
                "alta tasa de victorias en las temporadas de subida, pico y bajada."
            )
            fold_mae = (
                fold_comparison.filter(
                    (pl.col("horizon") == "all") & (pl.col("metric") == "mae")
                )
                .with_columns(beats=pl.col("mean_model") < pl.col("mean_baseline"))
                .select(
                    "unique_id",
                    "model",
                    "n_folds",
                    "beats",
                    "mean_model",
                    "std_model",
                    "mean_baseline",
                    "rel_mae",
                    "mean_diff",
                    "wins",
                    "losses",
                    "win_rate",
                    "sign_test_p",
                )
                .sort("unique_id", "mean_model")
            )
            st.dataframe(
                fold_mae,
                hide_index=True,
                column_config={
                    "beats": st.column_config.CheckboxColumn("MAE < Naive"),
                    "mean_model": st.column_config.NumberColumn(
                        "MAE medio (m)", format="%.4f"
                    ),
                    "std_model": st.column_config.NumberColumn(
                        "MAE std (m)", format="%.4f"
                    ),
                    "mean_baseline": st.column_config.NumberColumn(
                        "MAE Naive (m)", format="%.4f"
                    ),
                    "rel_mae": st.column_config.NumberColumn("rel_mae", format="%.3f"),
                    "mean_diff": st.column_config.NumberColumn(
                        "Δ MAE (m)", format="%.4f"
                    ),
                    "win_rate": st.column_config.NumberColumn(
                        "Tasa victorias", format="%.3f"
                    ),
                    "sign_test_p": st.column_config.NumberColumn(
                        "p signo", format="%.4f"
                    ),
                },
            )

        st.subheader("Métricas por estación y modelo (h = 1..14, último corte)")
        st.markdown(
            "Error sobre el último corte de validación antes del bloque de test. "
            "Las semillas de un modelo neuronal (`_s1`, `_s2`, `_s3`) se muestran "
            "por separado y debajo se resume su media ± desviación estándar."
        )
        table = (
            metrics.filter(pl.col("horizon") == "all")
            .with_columns(beats=pl.col("rel_mae") < 1)
            .select("unique_id", "model", "beats", *data.TABLE_METRICS)
            .sort("unique_id", "mae")
        )
        st.dataframe(table, hide_index=True, column_config=metric_columns())
        seeds = data.seed_stats(metrics)
        if not seeds.is_empty():
            st.markdown(
                "**Media ± desviación estándar entre semillas.** Si la desviación es "
                "comparable a la diferencia con otro modelo, esa diferencia no es "
                "concluyente."
            )
            st.dataframe(seeds, hide_index=True)
        st.markdown(
            "**Criterio de selección (README).** 1) El modelo debe tener "
            "`rel_mae < 1` en cada estación por separado. 2) Entre los que lo "
            "cumplen, gana el menor MAE promedio entre estaciones."
        )
        st.dataframe(
            data.selection_table(metrics),
            hide_index=True,
            column_config={
                "beats_naive": st.column_config.CheckboxColumn("rel_mae < 1 en todas"),
                "mean_mae": st.column_config.NumberColumn(
                    "MAE medio (m)", format="%.4f"
                ),
            },
        )

        st.subheader("Error por día de horizonte y evolución en los 24 cortes")
        st.markdown(
            "El primer gráfico muestra cómo crece el MAE del día 1 al día 14 después "
            "del corte. Cuando existen los 24 pliegues walk-forward, el segundo "
            "gráfico muestra el MAE en cada fecha de corte de 2021–2022 (temporada "
            "seca, subida, pico de agosto y bajada)."
        )
        station = st.selectbox(
            "Estación",
            sorted(metrics["unique_id"].unique()),
            format_func=lambda s: STATION_LABELS.get(s, s),
            key="h_station",
        )
        errors = data.horizon_errors(metrics, station)
        shown = st.multiselect(
            "Modelos",
            sorted(errors["model"].unique()),
            default=sorted(errors["model"].unique()),
            key="h_models",
        )
        st.altair_chart(
            charts.horizon_chart(errors.filter(pl.col("model").is_in(shown))),
            width="stretch",
        )
        timeline = data.load_fold_timeline(root, models, station)
        if not timeline.is_empty():
            st.markdown(
                f"**Evolución del MAE en los 24 cortes (2021–2022) — "
                f"{STATION_LABELS.get(station, station)}**"
            )
            st.altair_chart(
                charts.fold_timeline_chart(
                    timeline.filter(pl.col("model").is_in(shown))
                ),
                width="stretch",
            )

with tab_window:
    if not models:
        missing_predictions()
    else:
        st.info(
            "**¿Qué muestra esta pestaña?** El pronóstico día a día (14 días hacia "
            "adelante) desde cualquiera de las fechas de corte disponibles. La línea "
            "continua negra es el nivel real del río y las líneas punteadas son las "
            "predicciones de cada modelo."
        )
        left, mid = st.columns(2)
        station = left.selectbox(
            "Estación",
            sorted(forecasts["unique_id"].unique()),
            format_func=lambda s: STATION_LABELS.get(s, s),
            key="w_station",
        )
        cuts = sorted(
            forecasts.filter(pl.col("unique_id") == station)["cutoff"].unique()
        )
        cutoff: date = mid.selectbox(
            f"Corte ({len(cuts)} disponibles)",
            cuts,
            index=len(cuts) - 1,
            format_func=str,
            key="w_cutoff",
        )
        if len(cuts) > 1:
            st.caption(
                f"Cortes walk-forward disponibles para `{station}`: **{len(cuts)}** "
                f"({cuts[0]} a {cuts[-1]})."
            )
        options = sorted(forecasts["model"].unique())
        preferred = [
            model
            for model in options
            if model in ("naive", "booster") or "media" in model
        ]
        shown = st.multiselect(
            "Modelos", options, default=preferred or options[:3], key="w_models"
        )
        use_ensemble = st.checkbox(
            "Agregar ensamble (promedio de los modelos seleccionados)",
            value=False,
            key="w_ensemble",
        )
        window_forecasts = forecasts
        shown_models = list(shown)
        if use_ensemble:
            window_forecasts = data.add_ensemble(forecasts, shown, levels)
            if data.ENSEMBLE_LABEL in window_forecasts["model"].to_list():
                shown_models.append(data.ENSEMBLE_LABEL)
        history, after, preds = data.forecast_window(
            levels,
            window_forecasts,
            station,
            cutoff,
            shown_models,
            horizon=cfg["horizon"],
        )
        st.altair_chart(
            charts.forecast_chart(history, after, preds, cutoff), width="stretch"
        )
        window_mae = (
            preds.group_by("model")
            .agg(mae=(pl.col("y_hat") - pl.col("y")).abs().mean())
            .sort("mae")
        )
        st.markdown(
            "**MAE de esta ventana** (metros, contra el nivel observado). "
            "Active la casilla de ensamble arriba para comprobar si promediar los "
            "modelos seleccionados reduce el error en este corte."
        )
        st.dataframe(window_mae, hide_index=True)

with tab_live:
    st.subheader("Probar predicción a 14 días en cualquier fecha")
    st.info(
        "**¿Cómo funciona?** Elige una estación, un modelo y cualquier fecha de corte. "
        "El modelo se ajusta usando **solo la historia hasta esa fecha** y pronostica "
        "los 14 días siguientes junto con `naive` como referencia.\n\n"
        "- **Modelos rápidos (~1–20 s en CPU):** `booster`, `arimax`, `naive`, "
        "`seasonal_naive`.\n"
        "- **Modelos neuronales (`tcn`, `cnn_lstm`):** entrenan una red al vuelo; "
        "activa **Modo rápido (1 semilla)** para reducir 3× el tiempo de espera."
    )
    target_stations = [s for s in targets if s in levels["unique_id"].unique()]
    target_dates = levels.filter(pl.col("unique_id").is_in(target_stations))["ds"]
    min_cut = target_dates.min() + timedelta(days=365)
    max_cut = target_dates.max()
    default_cut = min(date(2022, 8, 15), max_cut - timedelta(days=int(cfg["horizon"])))
    default_cut = max(min_cut, default_cut)

    col_st, col_mod, col_cut = st.columns(3)
    live_station = col_st.selectbox(
        "Estación",
        target_stations,
        format_func=lambda s: STATION_LABELS.get(s, s),
        key="live_station",
    )
    runners = data.live_runners()
    live_model = col_mod.selectbox("Modelo", runners, key="live_model")
    live_cutoff = col_cut.date_input(
        "Fecha de corte (cutoff)",
        value=default_cut,
        min_value=min_cut,
        max_value=max_cut,
        key="live_cutoff",
    )
    live_fast = st.checkbox(
        "Modo rápido (1 semilla en modelos neuronales, 3× más veloz)",
        value=True,
        key="live_fast",
    )

    with st.expander(
        "Simulador «¿Qué pasaría si…?» (100 % en memoria temporal)", expanded=False
    ):
        st.caption(
            "Modifica los últimos 7 días previos al corte **únicamente en una copia "
            "temporal en memoria RAM** durante `predict()`. **Nunca modifica el "
            "dataset en disco, ni el entrenamiento del modelo, ni los reportes "
            "guardados.** Cuando algún deslizador es distinto de cero, verás la "
            "curva original y la curva `(simulado)` lado a lado."
        )
        s1, s2, s3 = st.columns(3)
        level_delta = s1.slider(
            "Cambio nivel local (m, últimos 7 días)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key="live_level_delta",
        )
        upstream_delta = s2.slider(
            "Cambio nivel aguas arriba (m)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key="live_upstream_delta",
        )
        rain_delta = s3.slider(
            "Cambio lluvia diaria (mm/día)",
            min_value=-20.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            key="live_rain_delta",
        )

    if st.button("Generar pronóstico a 14 días", type="primary", key="live_run"):
        st.session_state["live_params"] = (
            live_model,
            live_cutoff,
            live_fast,
            level_delta,
            upstream_delta,
            rain_delta,
        )

    if "live_params" in st.session_state:
        (
            run_model_name,
            run_cutoff,
            run_fast,
            run_level_delta,
            run_upstream_delta,
            run_rain_delta,
        ) = st.session_state["live_params"]
        live_preds = cached_live_forecast(
            run_model_name,
            run_cutoff,
            horizon=int(cfg["horizon"]),
            fast_mode=run_fast,
            station=live_station,
            level_delta_m=run_level_delta,
            upstream_delta_m=run_upstream_delta,
            rain_delta_mm=run_rain_delta,
        )
        live_models = sorted(live_preds["model"].unique())
        history, after, preds = data.forecast_window(
            levels,
            live_preds,
            live_station,
            run_cutoff,
            live_models,
            horizon=int(cfg["horizon"]),
        )
        st.altair_chart(
            charts.forecast_chart(history, after, preds, run_cutoff),
            width="stretch",
        )
        if not after.is_empty() and preds["y"].drop_nulls().len() > 0:
            live_mae = (
                preds.group_by("model")
                .agg(mae=(pl.col("y_hat") - pl.col("y")).abs().mean())
                .sort("mae")
            )
            st.markdown(
                f"**MAE en la ventana posterior a {run_cutoff}** "
                f"({STATION_LABELS.get(live_station, live_station)}):"
            )
            st.dataframe(live_mae, hide_index=True)
        daily_table = (
            preds.with_columns(error_abs=(pl.col("y_hat") - pl.col("y")).abs())
            .select("ds", "model", "y_hat", "y", "error_abs")
            .sort("ds", "model")
        )
        with st.expander("Tabla detallada de los 14 días pronosticados", expanded=True):
            st.dataframe(daily_table, hide_index=True)
