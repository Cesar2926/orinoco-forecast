# Reglas extendidas de datos (Orinoco Forecast)

Instrucciones de referencia para agentes que trabajen con datos en este proyecto.

## Inmutabilidad de entradas

- Los archivos de `data/raw/` son la única fuente de verdad y nunca se modifican, renombran ni reescriben desde código.
- Cualquier limpieza, imputación o reshape se materializa como archivo nuevo en `data/interim/` (trabajo) o `data/processed/` (listo para modelos).
- Si un archivo crudo presenta anomalías (fechas duplicadas, fuera de orden, 29 de febrero ausente, nulos), el pipeline las reporta con warnings y las conserva; solo el usuario decide corregirlos.

## Registro de fuentes

- Toda fuente nueva se declara únicamente en `configs/data.yaml` con `name`, `path`, `sheet`, `date_column`, `target_column`, `freq` y opcionales (`covariates`, `skiprows`, `format`, `layout`, `date_parts`).
- `format: csv` indica payload CSV aunque el sufijo del archivo sea `.xls` (caso `dataset_imputado_simpleml.csv.xls`).
- `layout: wide_years` describe el libro histórico: años en columnas, mes/día en filas; el reshape es en memoria.

## Convenciones de salida

- `data/interim/`: datos derivados no finales (CSV o Parquet de trabajo).
- `data/processed/`: datasets finales listos para entrenamiento/evaluación.
- `data/predictions/<modelo>/`: una carpeta por modelo (`arima`, `lstm`, `tree`, `cnn`, `transformer`, `timesfm`).
- `artifacts/`: checkpoints y pesos; ignorado por git; nunca se versionan `*.pt`.
- `reports/figures/`: gráficos generados por scripts de reporte.

## Validación mínima obligatoria al cargar una fuente

1. Existencia del archivo declarado en el YAML.
2. Índice/orden cronológico creciente.
3. Detección de duplicados y nulos, ambos con warnings (nunca silencio).
4. Coherencia de la frecuencia declarada (`freq`) con las fechas observadas.

## Notas del proyecto

- Unidades y estación objetivo (`ciudad_bolivar` / `nivel_m`, metros) están marcadas "por confirmar" en `data/raw/README.md`.
- El libro histórico omite el 29 de febrero; no se inventan fechas.
