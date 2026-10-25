"""Tablero de solo lectura del proyecto Orinoco Forecast.

Uso: ``uv run streamlit run app/dashboard.py`` (o ``make dashboard``).
Lee los niveles de ``data/raw`` y las predicciones de validación guardadas en
``data/predictions/<modelo>/``; nunca entrena modelos ni lee el test final.
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
from orinoco_forecast.features.dataset import load_levels

EXOGENOUS = repo_root() / "data" / "processed" / "orinoco_exogenous.parquet"
MODEL_HINTS = ("baselines", "xgboost", "tcn")

st.set_page_config(page_title="Orinoco Forecast", layout="wide")


@st.cache_data(show_spinner="Cargando niveles de data/raw…")
def cached_levels() -> pl.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Already audited in data/raw/README.md.
        return load_levels()


def _stamp(root: Path) -> tuple[tuple[str, float], ...]:
    """Fechas de modificación: invalidan la caché al regenerar predicciones."""
    return tuple((str(p), p.stat().st_mtime) for p in sorted(root.glob("*/*.parquet")))


@st.cache_data(show_spinner="Cargando predicciones…")
def cached_predictions(
    root: str, stamp: tuple
) -> tuple[list[str], pl.DataFrame, pl.DataFrame]:
    path = Path(root)
    models = data.discover_models(path)
    return models, data.load_metrics(path, models), data.load_forecasts(path, models)


@st.cache_data
def cached_optional(path: str, stamp: float) -> pl.DataFrame | None:
    return data.load_optional(Path(path))


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def missing_predictions(models: tuple[str, ...] = MODEL_HINTS) -> None:
    commands = "\n".join(data.run_command(m) for m in models)
    st.info(
        "No hay predicciones de validación en "
        f"`{root}`. Genérelas primero (los baselines antes que el resto):"
    )
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
models, metrics, forecasts = cached_predictions(str(root), _stamp(root))

st.title("Orinoco Forecast")
st.caption(
    "Tablero de solo lectura: muestra los datos y las predicciones de validación "
    "ya guardadas. No entrena modelos ni usa el bloque de test final "
    f"(desde {cfg['test_start']})."
)
tab_data, tab_val, tab_window, tab_imp = st.tabs(
    [
        "Datos",
        "Validación de modelos",
        "Pronóstico por ventana",
        "Importancia de variables",
    ]
)

with tab_data:
    stations = sorted(levels["unique_id"].unique())
    first, last = levels["ds"].min(), levels["ds"].max()
    left, right = st.columns([2, 1])
    chosen = left.multiselect(
        "Estaciones", stations, default=[s for s in targets if s in stations]
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

    st.subheader("Variables externas (etapa B)")
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
        st.markdown(
            f"Modelos con predicciones: {', '.join(f'`{m}`' for m in models)}. "
            f"Validación cruzada con {cfg['n_windows']} cortes cada "
            f"{cfg['step_size']} días y horizonte de {cfg['horizon']} días, todo "
            f"antes de {cfg['test_start']}."
        )
        if data.BASELINES not in models:
            st.info("Sin baselines no se puede calcular rel_mae. Ejecute:")
            st.code(data.run_command(data.BASELINES), language="bash")
        st.subheader("Métricas por estación y modelo (h = 1..14)")
        st.markdown(
            "Error promedio sobre los 14 días del horizonte y todas las ventanas. "
            "MAE y RMSE en metros (más bajo es mejor); `rel_mae` = MAE / MAE de la "
            "persistencia (Naive): la casilla marcada indica `rel_mae < 1`, es "
            "decir, el modelo supera a repetir el último valor observado. NSE y KGE: "
            "1 es perfecto. Las semillas de un modelo neuronal (`_s1`, `_s2`, …) se "
            "muestran por separado."
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
            "cumplen, gana el menor MAE promedio entre estaciones. El test final se "
            "usa una sola vez, con el modelo ya elegido."
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

        st.subheader("Error por día de horizonte")
        st.markdown(
            "MAE (metros) para cada día después del corte. Pronosticar mañana es "
            "fácil y a 14 días es difícil, así que todas las curvas crecen; importa "
            "dónde cada modelo queda por debajo de Naive (línea punteada)."
        )
        station = st.selectbox(
            "Estación", sorted(metrics["unique_id"].unique()), key="h_station"
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

with tab_window:
    if not models:
        missing_predictions()
    else:
        st.markdown(
            "Una ventana de validación: el modelo solo conoce los datos hasta el "
            "corte (línea vertical) y pronostica los 14 días siguientes. La línea "
            f"negra es el nivel observado ({data.HISTORY_DAYS} días antes y "
            f"{cfg['horizon']} después del corte); las punteadas de color, el "
            "pronóstico de cada modelo. `(media de semillas)` promedia las "
            "predicciones de las semillas solo para visualizar; no es un modelo "
            "evaluado."
        )
        left, mid = st.columns(2)
        station = left.selectbox(
            "Estación", sorted(forecasts["unique_id"].unique()), key="w_station"
        )
        cuts = sorted(
            forecasts.filter(pl.col("unique_id") == station)["cutoff"].unique()
        )
        cutoff: date = mid.selectbox(
            "Corte", cuts, index=len(cuts) - 1, format_func=str, key="w_cutoff"
        )
        options = sorted(forecasts["model"].unique())
        preferred = [m for m in options if m in ("Naive", "XGBoost") or "media" in m]
        shown = st.multiselect(
            "Modelos", options, default=preferred or options[:3], key="w_models"
        )
        history, after, preds = data.forecast_window(
            levels, forecasts, station, cutoff, shown, horizon=cfg["horizon"]
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
            "**MAE de esta ventana** (metros, contra la serie limpia usada en la "
            "validación). Una sola ventana es ruidosa: compare con la tabla de "
            "métricas antes de sacar conclusiones."
        )
        st.dataframe(window_mae, hide_index=True)

with tab_imp:
    path = root / "xgboost" / "feature_importance.parquet"
    importance = cached_optional(str(path), _mtime(path))
    if importance is None:
        st.info("No hay importancia de variables de XGBoost. Genérela con:")
        st.code(data.run_command("xgboost"), language="bash")
    else:
        st.markdown(
            "Las 15 variables que más reducen el error en los árboles de XGBoost "
            "(ganancia relativa, suma 100 % entre todas). Nombres: `lagN` es la "
            "serie objetivo N días antes (XGBoost modela el cambio diario del "
            "nivel, `target_differences: [1]`); `rolling_mean_lag1_window_sizeK` es "
            "su media móvil de K días hasta ayer; `sin_k`/`cos_k` son términos de "
            "Fourier del ciclo anual (época del año); `<estación>_lagN` es el nivel "
            "de una estación aguas arriba N días antes (N ≥ 14 para no usar datos "
            "posteriores al corte). Importancia no es causalidad: variables "
            "correlacionadas se reparten la ganancia."
        )
        st.altair_chart(charts.importance_chart(importance), width="stretch")
