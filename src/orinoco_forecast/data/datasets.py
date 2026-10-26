"""Dataset diario de niveles del río extraído desde el libro de Excel.

El Parquet escrito es la tabla larga ``unique_id, ds, y`` con lluvia NASA POWER
(``rain_<estacion>``), ONI de NOAA (``oni``) y los rezagos de nivel, cambio
diario, lluvia y ONI configurados en ``configs/data.yaml`` (ver
:mod:`orinoco_forecast.features.lags`). Las covariables se leen de
``data/external/`` según la sección ``exogenous`` de la configuración.
"""

import warnings
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from orinoco_forecast.data.exogenous import add_exogenous, load_exogenous_config
from orinoco_forecast.data.loading import (
    DataQualityWarning,
    load_config,
    repo_root,
    validate_daily_dates,
)
from orinoco_forecast.features.lags import (
    RiverFeatureConfig,
    add_river_features,
    load_feature_config,
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

        parsed_datetime = (
            pl.col("_date").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).dt.date()
        )
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

    def extract_levels(
        self,
        path: str | Path = DEFAULT_SOURCE,
        sheet_name: str = DEFAULT_SHEET,
    ) -> pl.DataFrame:
        """Extrae los niveles del Excel en tabla larga ``unique_id, ds, y``."""
        return (
            self.extract_xlsx(path, sheet_name)
            .unpivot(
                index="ds",
                on=list(STATIONS),
                variable_name="unique_id",
                value_name="y",
            )
            .select("unique_id", "ds", "y")
            .sort("unique_id", "ds")
        )

    def extract_exogenous(
        self,
        config: dict[str, Any] | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Lee lluvia NASA POWER y ONI desde ``data/external/``.

        Las rutas no se hardcodean: salen de la sección ``exogenous`` de
        ``configs/data.yaml``. Devuelve la lluvia diaria por estación
        (``ds, rain_<estacion>``) y el ONI diario (``ds, oni``).
        """
        return load_exogenous_config(config or load_config())

    def write_parquet(
        self,
        output_path: str | Path | None = None,
        *,
        feature_config: RiverFeatureConfig | None = None,
        include_exogenous: bool = True,
    ) -> Path:
        """Escribe la tabla larga con lluvia, ONI y los rezagos configurados.

        Las columnas ``rain_<estacion>`` y ``oni`` se alinean por fecha desde
        ``data/external/`` (NASA POWER y NOAA, ver ``exogenous`` en
        ``configs/data.yaml``) y se conservan sin imputar: las fechas fuera de
        la cobertura de la fuente (la lluvia empieza en 1981) quedan nulas y se
        reportan con ``DataQualityWarning``. Con ``include_exogenous=False`` se
        escribe solo el río.

        Las columnas ``<estacion>_lag_<n>`` (nivel), ``d_<estacion>_lag_<n>``
        (cambio diario), ``rain_<estacion>_lag_<n>`` y ``oni_lag_<n>`` salen de
        ``features`` en ``configs/data.yaml`` salvo que se pase
        ``feature_config``. El calendario ``sin_doy``/``cos_doy`` queda apagado
        con ``calendar_terms: 0``.
        """
        if output_path is not None:
            self.output_path = Path(output_path)
        settings = load_config()
        config = feature_config or load_feature_config(settings)
        levels = self.extract_levels()
        if include_exogenous:
            nasa, oni = self.extract_exogenous(settings)
            levels = add_exogenous(levels, nasa, oni)
        frame = add_river_features(
            levels,
            config,
            include_rain=any(column.startswith("rain_") for column in levels.columns),
            include_oni="oni" in levels.columns,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(self.output_path)
        return self.output_path
