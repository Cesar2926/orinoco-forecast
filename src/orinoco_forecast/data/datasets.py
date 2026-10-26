"""Dataset diario de niveles del río extraído desde el libro de Excel."""

import warnings
from datetime import date
from pathlib import Path

import polars as pl

from orinoco_forecast.data.loading import (
    DataQualityWarning,
    repo_root,
    validate_daily_dates,
)

DEFAULT_SOURCE = repo_root() / "data" / "raw" / "DATOS RIO ORINOCO.xlsx"
DEFAULT_SHEET = "Entrenar"
DEFAULT_OUTPUT = repo_root() / "data" / "interim" / "orinoco_dataset.parquet"
STATIONS = ("ayacucho", "caicara", "ciudad_bolivar", "palua")


class TimeSeriesDataset:
    """Dataset de niveles diarios definido por la ruta de su archivo Parquet."""

    def __init__(self, output_path: str | Path = DEFAULT_OUTPUT) -> None:
        self.output_path = Path(output_path)

    def extract_xlsx(
        self,
        path: str | Path = DEFAULT_SOURCE,
        sheet_name: str = DEFAULT_SHEET,
    ) -> pl.DataFrame:
        """Extrae fecha y niveles de las cuatro estaciones en formato ancho.

        La hoja ``Entrenar`` contiene fechas como texto de fecha/hora al ser leída
        con ``openpyxl``. También se aceptan números seriales de Excel para que el
        extractor sea compatible con ambas representaciones del libro.
        """
        source = Path(path)
        if not source.is_absolute():
            source = repo_root() / source
        if not source.is_file():
            raise FileNotFoundError(f"No se encontró el Excel de niveles: {source}")

        raw = pl.read_excel(
            source,
            sheet_name=sheet_name,
            engine="openpyxl",
            infer_schema_length=0,
        )
        if len(raw.columns) < len(STATIONS) + 1:
            raise ValueError(
                f"La hoja {sheet_name!r} debe contener fecha y cuatro estaciones; "
                f"se encontraron {len(raw.columns)} columnas."
            )

        columns = raw.columns[: len(STATIONS) + 1]
        extracted = raw.select(
            pl.col(columns[0]).cast(pl.String).str.strip_chars().alias("_date"),
            *(
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .alias(station)
                for column, station in zip(columns[1:], STATIONS, strict=True)
            ),
        ).filter(pl.col("_date").is_not_null() & (pl.col("_date") != ""))

        parsed_datetime = pl.col("_date").str.to_datetime(
            "%Y-%m-%d %H:%M:%S", strict=False
        ).dt.date()
        parsed_serial = (
            pl.lit(date(1899, 12, 30))
            + pl.duration(days=pl.col("_date").cast(pl.Int64, strict=False))
        ).cast(pl.Date)
        extracted = extracted.with_columns(
            pl.coalesce([parsed_datetime, parsed_serial]).alias("ds")
        ).drop("_date")
        extracted = extracted.select("ds", *STATIONS)

        if extracted["ds"].null_count():
            raise ValueError(
                f"{sheet_name!r} contiene fechas que no se pudieron convertir."
            )
        validate_daily_dates(extracted["ds"], f"{source.name}/{sheet_name}")

        nulls = extracted.null_count()
        for station in STATIONS:
            count = nulls[station][0]
            if count:
                warnings.warn(
                    f"{source.name}/{sheet_name}/{station}: {count} valores nulos.",
                    DataQualityWarning,
                    stacklevel=2,
                )
        return extracted

    def write_parquet(
        self,
        output_path: str | Path | None = None,
    ) -> Path:
        if output_path is not None:
            self.output_path = Path(output_path)
        frame = self.extract_xlsx()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(self.output_path)
        return self.output_path
