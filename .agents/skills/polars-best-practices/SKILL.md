---
name: polars-best-practices
description: "Trigger: polars, DataFrame, DataFrame de Polars, carga de datos, Excel, CSV, data.yaml, serie temporal. Aplica las reglas del proyecto al usar Polars en Orinoco Forecast."
license: Apache-2.0
metadata:
  author: "Cesar2926"
  version: "1.0"
---

# Polars Best Practices (Orinoco Forecast)

## Activation Contract

Activa esta skill al escribir o revisar cualquier código de carga, limpieza o transformación de datos del proyecto con Polars (incluye `src/orinoco_forecast/data/` y features).

## Hard Rules

- Usa Polars como único motor de tablas. Prohibido reintroducir pandas o `to_pandas()` sin justificarlo en el reporte final.
- `data/raw/` es de solo lectura: ningún código escribe ahí. Los derivados van a `data/interim/` o `data/processed/`.
- Nunca corrijas datos en silencio: duplicados, fechas fuera de orden o nulos se reportan con `warnings.warn`, sin imputar ni eliminar.
- Conserva el orden cronológico; no reordenes ni barajes observaciones fuera de un `sort` explícito y documentado.
- Declara dtypes explícitos al leer (`schema_overrides`, `schema`); evita inferencia silenciosa.
- Los nombres de fuente, hoja, columnas y formato viven en `configs/data.yaml`; no hardcodees rutas ni hojas.

## Decision Gates

| Necesidad | Elección |
|---|---|
| Archivo grande (>100 MB) o muchas hojas | API lazy (`pl.scan_csv`, `scan_parquet`) con `collect()` al final |
| Excel | `pl.read_excel(..., engine="openpyxl")`; una hoja por llamada |
| Validación temporal | `DataFrame.sort(date)` + verificación de `is_sorted`, duplicados y nulos con warnings |
| Serie histórica por años (layout `wide_years`) | Reshape en memoria (`unpivot`); nunca reescribir el Excel |
| Guardar resultados | `write_parquet`/`write_csv` hacia interim/processed |

## Execution Steps

1. Lee `configs/data.yaml` y resuelve la fuente por `name`; valida `path`, `sheet`, `date_column`, `target_column`, `freq`.
2. Carga con openpyxl (Excel) o `read_csv` (payload CSV aunque el sufijo sea `.xls`).
3. Parsea fechas con `str.to_date`/`str.to_datetime`, verifica orden, duplicados y nulos; emite `DataQualityWarning` cuando aplique.
4. Devuelve un `pl.DataFrame` ordenado por fecha, sin mutar valores originales.
5. Ejecuta `uv run pytest` y `uv run ruff check .` tras cada cambio.

## Output Contract

Reporta: fuente cargada (nombre YAML), dimensiones, rango temporal, nulos y warnings emitidos. Cero escrituras en `data/raw/`.

## References

- `configs/data.yaml` — registro de fuentes.
- `data/raw/README.md` — diccionario de datos y rarezas auditadas.
- `.agents/skills/polars-best-practices/references/data-rules.md` — reglas extendidas de datos del proyecto.
