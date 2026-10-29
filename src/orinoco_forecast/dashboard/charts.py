"""Altair chart builders for the dashboard (Altair accepts Polars frames)."""

from datetime import date

import altair as alt
import polars as pl

from orinoco_forecast.data.quality import QUALITY_LABELS

# Daily series span decades; Streamlit ships the rows to the browser anyway.
alt.data_transformers.disable_max_rows()

FLAGS = list(QUALITY_LABELS[1:])
FLAG_COLORS = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]
OBSERVED = "Observado"
HEIGHT = 340
_DATE = alt.X("ds:T", title="Fecha")


def level_chart(levels: pl.DataFrame) -> alt.LayerChart:
    """Nivel por estación con días no observados resaltados por etiqueta."""
    line = (
        alt.Chart(levels)
        .mark_line(strokeWidth=1)
        .encode(
            x=_DATE,
            y=alt.Y("y:Q", title="Nivel (m)"),
            color=alt.Color("unique_id:N", title="Estación"),
            tooltip=["unique_id", "ds:T", alt.Tooltip("y:Q", format=".2f")],
        )
    )
    flagged = levels.filter(pl.col("quality").cast(pl.String).is_in(FLAGS))
    color = alt.Color(
        "quality:N",
        title="Calidad",
        scale=alt.Scale(domain=FLAGS, range=FLAG_COLORS),
    )
    tooltip = ["unique_id", "ds:T", "quality:N", alt.Tooltip("y:Q", format=".2f")]
    points = (
        alt.Chart(flagged.filter(pl.col("y").is_not_null()))
        .mark_point(filled=True, size=45)
        .encode(x=_DATE, y="y:Q", color=color, tooltip=tooltip)
    )
    # Missing days have no level: draw them as ticks on the bottom edge.
    ticks = (
        alt.Chart(flagged.filter(pl.col("y").is_null()))
        .mark_tick(thickness=2, size=14)
        .encode(x=_DATE, y=alt.value(HEIGHT - 7), color=color, tooltip=tooltip)
    )
    return (
        alt.layer(line, alt.layer(points, ticks))
        .resolve_scale(color="independent")
        .properties(height=HEIGHT)
        .interactive(bind_y=False)
    )


def horizon_chart(errors: pl.DataFrame) -> alt.Chart:
    """MAE por día de horizonte; Naive punteado como referencia."""
    return (
        alt.Chart(errors)
        .mark_line(point=True)
        .encode(
            x=alt.X("horizon:O", title="Día de horizonte (días después del corte)"),
            y=alt.Y("mae:Q", title="MAE (m)"),
            color=alt.Color("model:N", title="Modelo"),
            strokeDash=alt.condition(
                alt.datum.model == "naive", alt.value([6, 4]), alt.value([1, 0])
            ),
            tooltip=[
                "model",
                "horizon",
                alt.Tooltip("mae:Q", format=".3f"),
                alt.Tooltip("rel_mae:Q", format=".3f"),
            ],
        )
        .properties(height=HEIGHT)
    )


def fold_timeline_chart(timeline: pl.DataFrame) -> alt.Chart:
    """MAE por fecha de corte en los 24 pliegues walk-forward; Naive punteado."""
    return (
        alt.Chart(timeline)
        .mark_line(point=True)
        .encode(
            x=alt.X("cutoff:T", title="Fecha de corte (2021–2022)"),
            y=alt.Y("mae:Q", title="MAE en la ventana de 14 días (m)"),
            color=alt.Color("model:N", title="Modelo"),
            strokeDash=alt.condition(
                alt.datum.model == "naive", alt.value([6, 4]), alt.value([1, 0])
            ),
            tooltip=[
                "model",
                "cutoff:T",
                alt.Tooltip("mae:Q", format=".3f"),
                alt.Tooltip("rel_mae:Q", format=".3f"),
            ],
        )
        .properties(height=HEIGHT)
    )


def forecast_chart(
    history: pl.DataFrame,
    after: pl.DataFrame,
    forecasts: pl.DataFrame,
    cutoff: date,
) -> alt.LayerChart:
    """Nivel observado alrededor del corte y el pronóstico de cada modelo."""
    observed = pl.concat(
        [history.select("ds", "y"), after.select("ds", "y")]
    ).with_columns(serie=pl.lit(OBSERVED))
    preds = forecasts.select(
        "ds", pl.col("y_hat").alias("y"), pl.col("model").alias("serie")
    )
    models = preds["serie"].unique(maintain_order=True).to_list()
    color = alt.Color(
        "serie:N",
        title="Serie",
        scale=alt.Scale(
            domain=[OBSERVED, *models],
            range=["#222222", *PALETTE[: len(models)]],
        ),
    )
    tooltip = ["serie", "ds:T", alt.Tooltip("y:Q", format=".2f")]
    y = alt.Y("y:Q", title="Nivel (m)", scale=alt.Scale(zero=False))
    obs_line = (
        alt.Chart(observed)
        .mark_line(strokeWidth=2)
        .encode(x=_DATE, y=y, color=color, tooltip=tooltip)
    )
    obs_after = (
        alt.Chart(after.select("ds", "y").with_columns(serie=pl.lit(OBSERVED)))
        .mark_point(size=30)
        .encode(x=_DATE, y="y:Q", color=color, tooltip=tooltip)
    )
    pred_lines = (
        alt.Chart(preds)
        .mark_line(point=True, strokeDash=[5, 3])
        .encode(x=_DATE, y="y:Q", color=color, tooltip=tooltip)
    )
    rule_data = pl.DataFrame({"ds": [cutoff], "label": [f"corte {cutoff}"]})
    rule = (
        alt.Chart(rule_data).mark_rule(color="gray", strokeDash=[2, 2]).encode(x="ds:T")
    )
    label = (
        alt.Chart(rule_data)
        .mark_text(align="left", dx=4, dy=-6, color="gray")
        .encode(x="ds:T", y=alt.value(10), text="label:N")
    )
    return alt.layer(obs_line, obs_after, pred_lines, rule, label).properties(
        height=HEIGHT
    )


def rain_chart(rain: pl.DataFrame) -> alt.Chart:
    """Lluvia diaria NASA POWER por estación."""
    return (
        alt.Chart(rain)
        .mark_line(strokeWidth=1)
        .encode(
            x=_DATE,
            y=alt.Y("rain:Q", title="Lluvia (mm/día)"),
            color=alt.Color("station:N", title="Estación"),
            tooltip=["station", "ds:T", alt.Tooltip("rain:Q", format=".1f")],
        )
        .properties(height=260)
        .interactive(bind_y=False)
    )


def oni_chart(oni: pl.DataFrame) -> alt.LayerChart:
    """ONI diario con los umbrales de El Niño (+0,5) y La Niña (−0,5)."""
    line = (
        alt.Chart(oni)
        .mark_line()
        .encode(
            x=_DATE,
            y=alt.Y("oni:Q", title="ONI (°C)"),
            tooltip=["ds:T", alt.Tooltip("oni:Q", format=".2f")],
        )
    )
    limits = (
        alt.Chart(pl.DataFrame({"oni": [-0.5, 0.5]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(y="oni:Q")
    )
    return alt.layer(line, limits).properties(height=220)
