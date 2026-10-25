"""Enriquece el nivel Orinoco formateado con NASA POWER, NOAA ONI y features.

Uso: ``uv run python scripts/build_exogenous_dataset.py``.
El resultado se escribe en ``data/processed/orinoco_exogenous.parquet``.
"""

import polars as pl

from orinoco_forecast.data.exogenous import add_exogenous, load_exogenous_config
from orinoco_forecast.data.loading import load_config, repo_root
from orinoco_forecast.features.lags import add_river_features, load_feature_config

# Variable de control del script: aplica los rezagos aguas arriba, lluvia, ONI
# y calendario sobre el resultado de add_exogenous antes de escribir la tabla.
BUILD_RIVER_FEATURES = True


def load_formatted_levels() -> pl.DataFrame:
    """Lee el CSV extendido conservando sus valores y fechas."""
    path = repo_root() / "data" / "raw" / "Nivel Orinoco Formateado.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Fuente de niveles no encontrada: {path}")
    return (
        pl.read_csv(
            path,
            schema_overrides={
                "fecha": pl.String,
                "ayacucho": pl.Float64,
                "caicara": pl.Float64,
                "ciudad_bolivar": pl.Float64,
                "palua": pl.Float64,
            },
        )
        .with_columns(pl.col("fecha").str.to_date("%Y/%m/%d").alias("ds"))
        .drop("fecha")
        .unpivot(
            index="ds",
            on=["ayacucho", "caicara", "ciudad_bolivar", "palua"],
            variable_name="unique_id",
            value_name="y",
        )
        .select("unique_id", "ds", "y")
        .sort("unique_id", "ds")
    )


def build() -> pl.DataFrame:
    """Construye la tabla larga de niveles, variables externas y features."""
    config = load_config()
    nasa, oni = load_exogenous_config(config)
    dataset = add_exogenous(load_formatted_levels(), nasa, oni)
    if BUILD_RIVER_FEATURES:
        dataset = add_river_features(dataset, load_feature_config(config))
    return dataset


def main() -> None:
    output_dir = repo_root() / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_output = output_dir / "orinoco_exogenous.parquet"
    csv_output = output_dir / "orinoco_exogenous.csv"
    dataset = build()
    rain_columns = [
        column
        for column in dataset.columns
        if column.startswith("rain_") and "_lag_" not in column
    ]
    if not rain_columns:
        raise ValueError("El dataset no contiene columnas de lluvia NASA POWER.")
    rain_rows = dataset.select(
        pl.any_horizontal(pl.col(rain_columns).is_not_null()).alias("has_rain")
    )["has_rain"].sum()
    if not rain_rows:
        raise ValueError(
            "El dataset contiene columnas de lluvia, pero ninguna observación."
        )
    dataset.write_parquet(parquet_output)
    dataset.write_csv(csv_output)
    base_columns = {"unique_id", "ds", "y", "oni", *rain_columns}
    feature_columns = [
        column for column in dataset.columns if column not in base_columns
    ]
    print(
        f"Escritos {parquet_output} y {csv_output} "
        f"({dataset.height} filas, {dataset.width} columnas)"
    )
    print(f"Columnas de lluvia: {', '.join(rain_columns)}")
    print(f"Filas con al menos un valor de lluvia: {rain_rows}")
    print(
        f"BUILD_RIVER_FEATURES={BUILD_RIVER_FEATURES}; "
        f"{len(feature_columns)} columnas de features"
    )
    if feature_columns:
        print(f"Primeras features: {', '.join(feature_columns[:8])}")
    with pl.Config(
        tbl_hide_dataframe_shape=True,
        tbl_formatting="ASCII_FULL_CONDENSED",
    ):
        print(
            dataset.group_by("unique_id")
            .agg(
                rows=pl.len(),
                start=pl.col("ds").min(),
                end=pl.col("ds").max(),
            )
            .sort("unique_id")
        )


if __name__ == "__main__":
    main()
