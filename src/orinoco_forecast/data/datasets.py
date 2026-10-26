"""Dataset diario de niveles del río extraído desde el libro de Excel.

El Parquet escrito es la tabla larga ``unique_id, ds, y`` con lluvia NASA POWER
(``rain_<estacion>``), ONI de NOAA (``oni``) y los rezagos de nivel, cambio
diario, lluvia y ONI configurados en ``configs/data.yaml`` (ver
:mod:`orinoco_forecast.features.lags`). Las covariables se leen de
``data/external/`` según la sección ``exogenous`` de la configuración.

:class:`ParquetDataset` es la interfaz de solo lectura de ese archivo: filtra
por estación, fechas y columnas antes de materializar el ``DataFrame``.
"""

import warnings
from collections.abc import Iterable, Sequence
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
LEVELS_COLUMNS = ("unique_id", "ds", "y")


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

    def interface(self) -> "ParquetDataset":
        """Interfaz de solo lectura del Parquet escrito por esta instancia."""
        return ParquetDataset(self.output_path)


class ParquetDataset:
    """Interfaz de solo lectura sobre el Parquet del dataset largo.

    Abre el archivo con :func:`polars.scan_parquet`, así que filtrar por
    estación, rango de fechas o subconjunto de columnas no materializa la tabla
    completa (*projection* y *predicate pushdown*). El archivo lo escribe
    :meth:`TimeSeriesDataset.write_parquet`.

    El orden de filas del archivo se conserva; ``write_parquet`` ya ordena por
    ``unique_id, ds``. Las rutas relativas se resuelven contra la raíz del
    repositorio.
    """

    REQUIRED_SCHEMA = {
        "unique_id": pl.String,
        "ds": pl.Date,
        "y": pl.Float64,
    }

    def __init__(self, path: str | Path = DEFAULT_OUTPUT) -> None:
        candidate = Path(path)
        self.path = candidate if candidate.is_absolute() else repo_root() / candidate
        if not self.path.is_file():
            raise FileNotFoundError(
                f"No se encontró el Parquet del dataset: {self.path}. "
                "Genéralo con TimeSeriesDataset.write_parquet()."
            )
        self._schema = self._read_schema()

    def _read_schema(self) -> dict[str, pl.DataType]:
        """Lee solo el footer del Parquet y valida las columnas mínimas."""
        schema = pl.scan_parquet(self.path).collect_schema()
        missing = set(self.REQUIRED_SCHEMA).difference(schema)
        if missing:
            raise ValueError(
                f"{self.path.name}: faltan columnas obligatorias {sorted(missing)}."
            )
        wrong = {
            column: (schema[column], expected)
            for column, expected in self.REQUIRED_SCHEMA.items()
            if schema[column] != expected
        }
        if wrong:
            raise ValueError(
                f"{self.path.name}: esquema inválido (columna -> (recibido, "
                f"esperado)): {wrong}."
            )
        return {name: dtype for name, dtype in schema.items()}

    def schema(self) -> dict[str, pl.DataType]:
        """Esquema completo del archivo, sin leer los datos."""
        return dict(self._schema)

    def columns(self) -> tuple[str, ...]:
        """Nombres de columna en el orden del archivo."""
        return tuple(self._schema)

    def n_rows(self) -> int:
        """Número de filas del Parquet."""
        return pl.scan_parquet(self.path).select(pl.len()).collect().item()

    def unique_ids(self) -> list[str]:
        """``unique_id`` disponibles, ordenados alfabéticamente."""
        return (
            pl.scan_parquet(self.path)
            .select(pl.col("unique_id").unique().sort())
            .collect()["unique_id"]
            .to_list()
        )

    def date_range(self) -> tuple[date, date]:
        """Primera y última fecha del dataset."""
        bounds = (
            pl.scan_parquet(self.path)
            .select(
                pl.col("ds").min().alias("start"),
                pl.col("ds").max().alias("end"),
            )
            .collect()
            .row(0)
        )
        return bounds

    def feature_columns(self) -> tuple[str, ...]:
        """Columnas predictoras: todo menos ``unique_id``, ``ds`` y ``y``."""
        return tuple(column for column in self._schema if column not in LEVELS_COLUMNS)

    def lag_columns(self) -> tuple[str, ...]:
        """Columnas de rezago (contienen ``_lag_``)."""
        return tuple(column for column in self._schema if "_lag_" in column)

    def exogenous_columns(self) -> tuple[str, ...]:
        """Lluvia y ONI crudos, sin rezago: ``rain_<estacion>`` y ``oni``."""
        return tuple(
            column
            for column in self._schema
            if (column.startswith("rain_") and "_lag_" not in column) or column == "oni"
        )

    def null_counts(self, columns: Sequence[str] | None = None) -> dict[str, int]:
        """Nulos por columna (todas si ``columns`` es ``None``)."""
        selection = self._select_columns(columns)
        counts = (
            pl.scan_parquet(self.path)
            .select(*(pl.col(column).null_count() for column in selection))
            .collect()
            .row(0)
        )
        return {
            column: int(count) for column, count in zip(selection, counts, strict=True)
        }

    def summary(self) -> dict[str, Any]:
        """Resumen del archivo: ruta, filas, columnas, estaciones y fechas."""
        start, end = self.date_range()
        return {
            "path": str(self.path),
            "rows": self.n_rows(),
            "columns": len(self._schema),
            "unique_ids": self.unique_ids(),
            "start": start,
            "end": end,
        }

    def lazy(
        self,
        *,
        columns: Sequence[str] | None = None,
        unique_ids: str | Iterable[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.LazyFrame:
        """Plan perezoso con filtros de columna, estación y rango de fechas.

        ``start``/``end`` son inclusivos y deben ser ``datetime.date`` (sin
        hora). Un ``unique_id`` o una columna inexistente levantan
        ``ValueError`` en vez de devolver una tabla vacía o silenciosa.
        """
        selection = self._select_columns(columns)
        start = self._as_date(start, "start")
        end = self._as_date(end, "end")
        if start is not None and end is not None and start > end:
            raise ValueError(f"start ({start}) no puede ser posterior a end ({end}).")
        frame = pl.scan_parquet(self.path)
        stations = self._select_ids(unique_ids)
        if stations is not None:
            frame = frame.filter(pl.col("unique_id").is_in(stations))
        if start is not None:
            frame = frame.filter(pl.col("ds") >= start)
        if end is not None:
            frame = frame.filter(pl.col("ds") <= end)
        return frame.select(*selection)

    def read(
        self,
        *,
        columns: Sequence[str] | None = None,
        unique_ids: str | Iterable[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pl.DataFrame:
        """Materializa :meth:`lazy` en un ``pl.DataFrame``."""
        return self.lazy(
            columns=columns,
            unique_ids=unique_ids,
            start=start,
            end=end,
        ).collect()

    def _select_columns(self, columns: Sequence[str] | None) -> tuple[str, ...]:
        if columns is None:
            return tuple(self._schema)
        selection = tuple(dict.fromkeys(columns))
        if not selection:
            raise ValueError("columns no puede estar vacío.")
        missing = [column for column in selection if column not in self._schema]
        if missing:
            raise ValueError(
                f"{self.path.name}: columnas inexistentes {missing}. "
                f"Disponibles: {list(self._schema)}."
            )
        return selection

    def _select_ids(self, unique_ids: str | Iterable[str] | None) -> list[str] | None:
        if unique_ids is None:
            return None
        values = [unique_ids] if isinstance(unique_ids, str) else list(unique_ids)
        if not values:
            raise ValueError("unique_ids no puede estar vacío.")
        available = self.unique_ids()
        missing = sorted(set(values).difference(available))
        if missing:
            raise ValueError(
                f"{self.path.name}: unique_ids inexistentes {missing}. "
                f"Disponibles: {available}."
            )
        return values

    @staticmethod
    def _as_date(value: date | None, label: str) -> date | None:
        if value is not None and type(value) is not date:
            raise ValueError(
                f"{label} debe ser datetime.date (sin hora), no {type(value).__name__}."
            )
        return value
