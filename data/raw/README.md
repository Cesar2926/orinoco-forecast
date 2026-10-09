# Datos crudos

Los archivos de este directorio son entradas originales e inmutables. Los lectores
del proyecto solo los abren; limpieza o resultados derivados deben guardarse en
`data/interim/` o `data/processed/`. Se conservaron los nombres y bytes de los
archivos al moverlos. Ninguno supera 50 MB.

## `Niveles Orinoco.xlsx` — 672 550 bytes

- **Hojas:** `Niveles PC CO CB P` (18 684 filas × 5 columnas, contando encabezado).
- **Encabezado:** la primera fila; no se detectaron filas extra de encabezado.
- **Frecuencia y fechas:** diaria; `1974-01-01` a `2025-02-24`. 18 683 fechas,
  ordenadas, únicas y sin saltos.
- **Nulos:** `fecha`, `ayacucho`, `caicara` y `ciudad_bolivar`: 0; `palua`: 17
  (0,09 % de las filas).
- **Objetivo inicial:** `ciudad_bolivar`, por ser la estación compartida con el
  histórico extenso. Las demás estaciones se registran como covariables.

| Columna | Tipo observado | Significado probable |
|---|---|---|
| `fecha` | datetime | Fecha de observación. |
| `ayacucho` | float | Nivel del río en estación Ayacucho; unidad metros por confirmar. |
| `caicara` | float | Nivel del río en estación Caicara; unidad metros por confirmar. |
| `ciudad_bolivar` | float | Nivel del río en estación Ciudad Bolívar; unidad metros por confirmar. |
| `palua` | float | Nivel del río en estación Palúa; unidad metros por confirmar. |

La unidad “m” aparecía en el eje de los gráficos exploratorios inspeccionados antes
de retirarlos, pero debe confirmarse contra la fuente hidrológica original.

## `Niveles Del Orinoco Est. Ciudad Bolivar 1930-2025.xlsx` — 1 490 446 bytes

- **Hojas:** 58 pestañas en total: 2 hojas con celdas (`NIVELES`, `REGRESION`) y
  56 hojas de gráficos (chartsheets). Los chartsheets no contienen tablas de
  celdas que puedan cargarse como series.
- **`NIVELES`:** dimensión física reportada por openpyxl de 392 × 117; la auditoría leyó
  389 × 109 celdas no vacías. La primera fila es el encabezado. Contiene 365 filas
  de calendario seguidas por 24 filas de resúmenes/separadores/blancos.
- **Frecuencia y fechas:** diaria, calendario de 365 días (no hay fila para el 29
  de febrero). Las columnas de años cubren 1930–2026; 2026 está completamente
  vacío. Para 1930–2025 hay 35 040 celdas de observación, 34 861 valores y 179
  nulos (0,51 %). El calendario representado va de `1930-01-01` a `2025-12-31`;
  la última lectura no nula está en `2025-12-30`.
- **Nulos y rarezas:** faltan 179 lecturas en las columnas 1930–2025. Las filas de
  resumen mezclan números y texto en `DIA`/`DIA DEL MES`; la auditoría infirió esas
  columnas como `object`. `1960` también queda como `object` por celdas fuera del
  calendario, aunque sus 365 valores del calendario son numéricos. Hay columnas
  separadoras sin nombre y alertas/estadísticas al final. openpyxl reporta
  encabezados/pies de página que no puede interpretar; los ignora al leer.
- **Objetivo propuesto:** `nivel_m`, que se podría armar en memoria a partir de
  cada columna de año y las partes `MES` + `DIA DEL MES`. La conversión aún no está
  implementada. Se prevé excluir filas no calendáricas y la columna vacía 2026,
  conservar los valores tal cual y señalar la omisión del 29 de febrero. Unidad
  metros por confirmar.

| Columna / grupo | Tipo observado | Significado probable |
|---|---|---|
| `DIA` | numérico en calendario; mixto fuera | Ordinal de día del año (1–365), por confirmar. |
| `MES` | texto español, mayormente nulo | Abreviatura del mes (`ENE`…`DIC`), solo en la primera fila de cada mes. |
| `DIA DEL MES` | numérico en calendario; mixto fuera | Día del mes (1–31); junto a mes y encabezado de año genera la fecha. |
| `1930`…`2025` | principalmente float | Nivel diario de Ciudad Bolívar para el año indicado; unidad metros por confirmar. |
| `2026` | float, 100 % nulo | Columna de año futura/vacía; excluida de la fuente configurada. |
| `MAX (30-24)`, `PROM (30-24)`, `MIN (30-24)` | float | Máximo, promedio y mínimo del período indicado en el rótulo; período exacto por confirmar. |
| `D. STD.` | float | Desviación estándar, probable. |
| `Alerta Verde`, `Alerta Amarilla`, `Alerta Roja` | float | Umbrales de alerta hidrológica, probable. |
| `Unnamed: 100`, `Unnamed: 105` | float/nulo | Columnas separadoras o celdas sin rótulo. |

La hoja `REGRESION` tiene dimensión física 388 × 68 (auditoría: 371 × 68), 67
columnas con etiquetas de año 1943–2009 y `Unnamed: 67`. Sus datos son mayormente
float, con 5,9 % de celdas nulas; no contiene una fecha convencional y parece una
tabla de cálculos de regresión, por eso no se registra como fuente de serie.

Nombres de los 56 chartsheets, conservados intactos: `H1962`–`H1980`,
`H1995`–`H2012`, `HIDRO11-12`, `HIDROMASPARE`, `MAXIMAS CRECIENTES`,
`PROMEDIOS ANUALES`, `MAXANUALES`, `MAXPROMINANUALES`, `PROMEDIOS MENSUALES`,
`PROMENERO`, `PROMFEBRERO`, `PROMMARZO`, `PROMABRIL`, `PROMMAYO`, `PROMJUNIO`,
`PROMJULIO`, `PROMAGOSTO`, `PROMSEPTIEMBRE`, `PROMOCTUBRE`, `PROMNOVIEMBRE` y
`PROMDICIEMBRE`.

## `dataset_imputado_simpleml.csv.xls` — 647 121 bytes

- **Formato real:** CSV delimitado por comas, no es un libro Excel; no tiene hojas.
  Se mantiene el sufijo original y `configs/data.yaml` indica `format: csv` para
  leerlo correctamente.
- **Dimensiones y columnas:** 18 683 filas × 5 columnas. `fecha` (texto antes de
  parseo), `ayacucho`, `caicara`, `ciudad_bolivar`, `palua` (float).
- **Frecuencia y fechas:** diaria, `1974-01-01` a `2025-02-24`, 18 683 fechas
  ordenadas, únicas y sin saltos.
- **Nulos:** 17 en `palua` (0,09 %); 0 en las otras columnas.
- **Rareza importante:** pese a `imputado_simpleml` en el nombre, el contenido
  cargado coincide fila por fila y valor por valor con `Niveles Orinoco.xlsx` en
  el árbol de trabajo auditado. No se observó una imputación adicional; el origen
  exacto del nombre queda por confirmar.
- **Objetivo inicial:** `ciudad_bolivar`; las otras tres estaciones son
  covariables. Significado y unidad, por confirmar como en el libro anterior.

## Suposiciones por confirmar

1. Que `ciudad_bolivar` / `nivel_m` sean la serie objetivo y que las cuatro
   estaciones compartan una unidad comparable.
2. Que los niveles estén expresados en metros, como sugiere el eje de los gráficos
   existentes.
3. Que la estructura de `NIVELES` se deba interpretar con meses abreviados en
   español, años en columnas y días del mes en filas.
