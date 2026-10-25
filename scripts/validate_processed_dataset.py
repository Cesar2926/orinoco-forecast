"""Valida el parquet procesado antes de construir ventanas de entrenamiento."""

import polars as pl

from orinoco_forecast.data.validation import validate_processed_dataset


def main() -> None:
    """Ejecuta la validación y muestra un resumen reproducible."""
    frame = validate_processed_dataset()
    with pl.Config(
        tbl_hide_dataframe_shape=True,
        tbl_formatting="ASCII_FULL_CONDENSED",
    ):
        print(f"Dataset válido estructuralmente: {frame.height} filas")
        print(f"Columnas: {', '.join(frame.columns)}")
        print(
            frame.group_by("unique_id")
            .agg(
                rows=pl.len(),
                start=pl.col("ds").min(),
                end=pl.col("ds").max(),
                null_y=pl.col("y").null_count(),
            )
            .sort("unique_id")
        )
        print(frame.select(pl.col("unique_id").n_unique().alias("stations")))
        print(
            "Regla anti-fuga: lluvia y ONI solo se usarán con rezagos conocidos "
            "en la fecha del corte; no se usarán valores futuros contemporáneos."
        )


if __name__ == "__main__":
    main()
