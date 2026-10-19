"""Build the labelled long table in data/interim/ from the raw sources.

Usage: ``uv run python scripts/build_dataset.py``. Reads data/raw/ only.
"""

import polars as pl

from orinoco_forecast.data.loading import load_historical, load_primary, repo_root
from orinoco_forecast.data.quality import QUALITY_LABELS, label_quality

INTERIM = repo_root() / "data" / "interim"


def build() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Une primaria + extensión histórica de Ciudad Bolívar y etiqueta la calidad.

    Solo ``ciudad_bolivar`` se extiende tras el fin de la primaria; las demás
    estaciones no existen en la fuente histórica y no se imputan.
    """
    primary = load_primary().with_columns(source=pl.lit("primary"))
    historical = load_historical()
    primary_end = primary["ds"].max()
    extension = historical.filter(
        pl.col("unique_id").is_in(primary["unique_id"].unique().implode())
        & (pl.col("ds") > primary_end)
    ).with_columns(source=pl.lit("historical"))
    levels = pl.concat([primary, extension]).sort("unique_id", "ds")
    # Conflicts only make sense where both sources overlap.
    reference = historical.filter(pl.col("ds") <= primary_end)
    levels = label_quality(levels, reference=reference)
    summary = (
        levels.group_by("unique_id", pl.col("ds").dt.year().alias("year"), "quality")
        .agg(n=pl.len())
        .sort("unique_id", "year", "quality")
    )
    return levels, summary


def main() -> None:
    levels, summary = build()
    INTERIM.mkdir(parents=True, exist_ok=True)
    levels.write_parquet(INTERIM / "levels_long.parquet")
    summary.write_csv(INTERIM / "quality_summary.csv")

    spans = (
        levels.group_by("unique_id", "source")
        .agg(rows=pl.len(), start=pl.col("ds").min(), end=pl.col("ds").max())
        .sort("unique_id", "source")
    )
    totals = summary.pivot(
        "quality", index="unique_id", values="n", aggregate_function="sum"
    ).sort("unique_id")
    totals = totals.select(
        "unique_id", *(c for c in QUALITY_LABELS if c in totals.columns)
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_hide_dataframe_shape=True):
        print(f"Escrito {INTERIM / 'levels_long.parquet'} ({levels.height} filas)")
        print(f"Escrito {INTERIM / 'quality_summary.csv'} ({summary.height} filas)")
        print(spans)
        print(totals.fill_null(0))


if __name__ == "__main__":
    main()
