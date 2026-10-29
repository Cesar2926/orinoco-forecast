# Transformer propio y TimesFM 3

Este documento describe los dos modelos de atención del proyecto: un **Transformer propio**
(`transformer`), entrenado con los datos del río, y **TimesFM 3** (`timesfm3`), el modelo
fundacional de Google usado sin reentrenar. Sigue el mismo proceso que el CNN-LSTM de
[`modelos_arboles_cnn.md`](modelos_arboles_cnn.md): runner, YAML y tests; búsqueda de
hiperparámetros en los 12 cortes de ajuste de 2020; validación en los 24 cortes de
`run_model.py --folds`; resultados en `reports/`; cuaderno de proceso. El paso a paso está en
[`04_transformer.ipynb`](../notebooks/04_transformer.ipynb) y
[`05_timesfm3.ipynb`](../notebooks/05_timesfm3.ipynb).

Todas las cifras son de validación (24 cortes, 2021-01-26 a 2022-12-17); el test final
(desde 2023-01-01) no se ha tocado.

---

## 1. Transformer propio

### Qué es y por qué no el TFT

`plan_modelos.md` (sección 2.5) preveía el TFT de `neuralforecast`. Se cambió por una red
propia por una razón de método: el TFT trae su propio preprocesado, así que una diferencia con
el CNN-LSTM no se podría atribuir a la arquitectura. `TransformerRunner` **hereda de
`CnnLstmRunner`** y solo reemplaza la red (`_build_net`) y el prefijo de las columnas
(`TRANSFORMER_s<semilla>`). Ventanas, escalado robusto por ventana, `target_transform:
difference`, parada temprana, semillas y `drop_covariates: [rain_, oni]` son idénticos.

```text
ventana (lote, input_size, variables)
  -> Linear(variables, d_model) + posición aprendida
  -> num_layers x nn.TransformerEncoderLayer (pre-norm, batch_first)
  -> último día de la ventana -> Linear(d_model, 14)
```

Código: [`models/neural/transformer.py`](../src/orinoco_forecast/models/neural/transformer.py).
Tests: `tests/test_runner_transformer.py` (forma de la salida, que el runner no use valores
posteriores al corte, validación de `d_model` múltiplo de `nhead` y registro).

### Ajuste

`scripts/tune_transformer.py` reutiliza `run_study` de `tune_cnn_lstm.py` con 20 pruebas de
Optuna. Gana la prueba 18: `input_size` 112, `d_model` 32, 2 cabezas, 2 capas,
`dim_feedforward` 64, `dropout` 0,12, tasa de aprendizaje 0,0018 y lote 256
(`reports/tuning/transformer_trials.csv`). Las cuatro mejores pruebas son las últimas y todas
usan la ventana más larga: **la búsqueda no convergió** y el óptimo está en el borde del
espacio.

**Error conocido en el denominador del ajuste.** `NaiveRunner.predict` ignora el `cutoff` y
repite el nivel del corte en que se ajustó; `tune_cnn_lstm.py` lo ajusta una sola vez, así que
el `rel_mae` de todos los CSV de ajuste usa la persistencia del primer corte de ajuste. El orden
de las pruebas no cambia (el denominador es común), pero esos valores no son `rel_mae` reales.
La validación de 24 cortes no tiene este problema (crea un Naive por corte).

### Resultados (24 cortes, CPU, media de 3 semillas)

| Estación | Horizonte | Naive | Transformer | p (test de signos, rango entre semillas) |
|---|---|---|---|---|
| Ciudad Bolívar | día 7 | 0,526 | 0,272 / 0,52 | 0,007–0,064 |
| Ciudad Bolívar | día 14 | 1,064 | 0,522 / 0,49 | 0,002–0,064 |
| Ciudad Bolívar | días 1 a 14 | 0,570 | 0,294 / 0,52 | 0,007–0,152 |
| Palúa | día 7 | 0,435 | 0,375 / 0,86 | 0,541–1,000 |
| Palúa | día 14 | 0,891 | 0,608 / 0,68 | 0,007–0,152 |
| Palúa | días 1 a 14 | 0,488 | 0,357 / 0,73 | 0,023–0,307 |

MAE medio entre estaciones: **0,326 m** (booster 0,309, TCN 0,317, CNN-LSTM 0,350, ARIMAX
0,351). Sin ajustar daba 0,330 (Ciudad Bolívar 0,310, Palúa 0,350).

- Cumple el criterio (`rel_mae < 1` en cada estación y horizonte).
- Mejora al CNN-LSTM en las dos estaciones con el mismo pipeline: la atención aporta más que
  convolución + LSTM.
- Supera al ARIMAX, así que la hipótesis de Zeng et al. (2023) recogida en `plan_modelos.md`
  (un Transformer que no gana a un modelo lineal) no se cumple aquí.
- El ajuste mejoró Ciudad Bolívar (0,310 → 0,294) y empeoró algo Palúa (0,350 → 0,357): el
  objetivo mezcla las dos estaciones y la prueba ganadora favorece a Ciudad Bolívar.
- Palúa a 7 días, igual que en los demás modelos, no supera a Naive con evidencia.
- Varianza entre semillas apreciable: en Ciudad Bolívar la semilla 3 da 0,269 y las otras dos
  0,307.

### Limitaciones

- Búsqueda de 20 pruebas sin converger; un espacio con ventanas mayores que 112 podría mejorar.
- Una sola configuración para las dos estaciones.
- Corrió en CPU (unas 7 h con la CPU compartida); en GPU las cifras pueden variar levemente.
- Sin lluvia ni ONI, como las demás redes.

### Cómo reproducir

```bash
uv run python scripts/run_model.py booster --write-dataset   # parquet, una vez
uv run python scripts/tune_transformer.py --trials 20
uv run python scripts/run_model.py transformer --against naive
uv run python scripts/run_model.py transformer --against naive --folds
```

Resultados guardados en `reports/predictions/transformer/`.

---

## 2. TimesFM 3 (Google, *zero-shot*)

### Qué es

Modelo fundacional preentrenado por Google con miles de series
([Das et al., 2024](https://proceedings.mlr.press/v235/das24c.html)). **No se entrena con datos
del Orinoco**: en cada corte recibe como contexto la historia del nivel hasta el corte y
devuelve 14 días. Responde si un modelo general supera a modelos ajustados para este río.

Pesos `google/timesfm-3.0-pytorch`, de licencia **no comercial** (compatible con un proyecto
académico); se instalan con el extra opcional `uv sync --extra timesfm`.

### API real y runner

El README oficial documenta `TimesFM3Evaluator`, que no existe en `timesfm==3.0.2`. La API
verificada es `TimesFM3Forecaster.from_pretrained(...).predict_batch(contextos, horizonte,
past_only_covariates=...)`, con salida en parches de 64 pasos que el runner recorta a 14.

`TimesFM3Runner` ([`models/pretrained/timesfm.py`](../src/orinoco_forecast/models/pretrained/timesfm.py)):

- `fit` no entrena: carga los pesos **una vez por proceso** (caché por checkpoint y
  dispositivo), porque el backtest crea un runner por corte.
- `predict` arma por serie el contexto `y[ds <= corte][-context_len:]` y, si
  `use_covariates`, las covariables pasadas del mismo tramo (las que deja `drop_covariates`).
- Una columna `TIMESFM3`, sin semillas: la inferencia es determinista. Se usa la mediana.

Tests (`tests/test_runner_timesfm.py`) con un predictor falso, sin descargar pesos: recorte al
horizonte, contexto, que no use valores posteriores al corte, forma de las covariables, caché y
registro.

### Búsqueda *zero-shot*

Sin entrenamiento el espacio es finito: `context_len` en {512, 1024, 2048} por
`use_covariates` en {no, sí}, **6 configuraciones, todas evaluadas** en los mismos 12 cortes de
ajuste ([`scripts/tune_timesfm3.py`](../scripts/tune_timesfm3.py)). A diferencia de
`tune_cnn_lstm.py`, Naive se reajusta en cada corte (ver el error de la sección 1), así que sus
`rel_mae` no se comparan con los de los otros CSV de ajuste. Gana `context_len` 1024 **sin
covariables** (`rel_mae` 0,454); las covariables mejoran Palúa pero empeoran más Ciudad Bolívar
(`reports/tuning/timesfm3_trials.csv`).

### Resultados (24 cortes, CPU)

| Estación | Horizonte | Naive | TimesFM 3 | p |
|---|---|---|---|---|
| Ciudad Bolívar | día 7 | 0,526 | 0,278 / 0,53 | 0,007 |
| Ciudad Bolívar | día 14 | 1,064 | 0,507 / 0,48 | 0,007 |
| Ciudad Bolívar | días 1 a 14 | 0,570 | 0,287 / 0,50 | 0,002 |
| Palúa | día 7 | 0,435 | 0,345 / 0,79 | 0,152 |
| Palúa | día 14 | 0,891 | 0,486 / 0,55 | 0,023 |
| Palúa | días 1 a 14 | 0,488 | 0,313 / 0,64 | 0,064 |

MAE medio entre estaciones: **0,300 m**, el menor de todos los modelos (booster 0,309, TCN
0,317, Transformer 0,326).

- Cumple el criterio en las dos estaciones y los tres horizontes.
- Es el mejor en **Palúa** a 7 y a 14 días; en Ciudad Bolívar está a la par del booster y del TCN.
- La ventaja sobre el booster (0,009 m) es menor que la dispersión entre cortes: no hay
  evidencia de que sea mejor, solo de que no es peor.

### Limitaciones

- **Posible contaminación del preentrenamiento**: si incluyó niveles o caudales del Orinoco de
  2021–2022, esta validación es optimista. No se puede verificar; es la limitación principal.
- Solo se evaluó la mediana; los cuantiles (0,1–0,9) quedan sin usar.
- Sin ajuste fino (LoRA), por diseño: mediría otra cosa.
- Sin lluvia ni ONI, como las redes.

### Cómo reproducir

```bash
uv sync --extra timesfm
uv run python scripts/tune_timesfm3.py                       # unos 7 min en CPU
uv run python scripts/run_model.py timesfm3 --against naive
uv run python scripts/run_model.py timesfm3 --against naive --folds   # 53 s en CPU
```

Resultados guardados en `reports/predictions/timesfm3/`.
