# Plan: terminar el Transformer y agregar TimesFM 3

Estado al 2026-10-29: **S1 y S2 terminados**. `feat/timesfm3` va apilada sobre `feat/transformer`.
Reemplaza el borrador anterior, que se escribió antes de que `main` cambiara el contrato
de los runners.

El proceso a replicar es el que siguió el grupo con el CNN-LSTM (PR #9 y siguientes):

1. runner, YAML y tests;
2. Optuna (12 cortes de ajuste, 20 pruebas);
3. `chore: tune ...` con los parámetros en el YAML;
4. 24 cortes con `run_model.py --folds` en Colab;
5. copia de los resultados en `reports/`;
6. notebook de proceso;
7. documentación;
8. PR.

## S1. Transformer: lo que falta

| # | Paso | Dónde | Estado |
|---|---|---|---|
| 1 | Runner `TransformerRunner(CnnLstmRunner)`, `transformer.yaml` y tests | `feat/transformer` | **Hecho** (`916fc74`) |
| 2 | Búsqueda de Optuna (`scripts/tune_transformer.py`) | `feat/transformer` | **Hecho** (`aa17fa9`) |
| 3 | Notebook de Colab con la opción `transformer` y la celda `TUNE` | `feat/transformer` | **Hecho** (`5506f67`) |
| 4 | 24 cortes **sin ajustar**, como referencia: MAE medio 0,330, tercero de cinco | Colab | **Hecho**; no se guarda en `reports/` |
| 5 | Ajuste: `TUNE = True`, 20 pruebas → `transformer_trials.csv` | Colab (usuario) | **Hecho**: gana la prueba 18 (0,675) |
| 6 | Pasar los mejores parámetros a `transformer.yaml` con el comentario "ajustado con Optuna…" y guardar `reports/tuning/transformer_trials.csv`. Commit `chore: tune Transformer on 2009-2024 data` | local | **Hecho** (`b6186da`) |
| 7 | 24 cortes ajustados: `TUNE = False` → `folds_transformer.zip` | local, CPU (unas 7 h) | **Hecho**: MAE medio 0,326 |
| 8 | Copiar a `reports/predictions/transformer/`: `backtest/naive/*` (del zip), y `predictions.parquet` y `statistics.parquet` de un corte (2022-12-17, `run_model.py transformer`), igual que los otros modelos. Actualizar la tabla de hardware de `reports/predictions/README.md` | local | **Hecho** (`a072ab2`) |
| 9 | Notebook de proceso `notebooks/04_transformer.ipynb`, con el mismo orden que `03_cnn_lstm.ipynb` (problema, cómo funciona, alternativa descartada, cuándo sería un error), sin salidas, más su fila en `notebooks/README.md` | local | **Hecho** |
| 10 | Documentación: fila del Transformer en la tabla de resultados y la de hardware del README, estado "Implementado y validado" y sección del modelo en `docs/` (archivo nuevo o sección 6 ter de `modelos_arboles_cnn.md`; confirmar con el grupo) | local | **Hecho**: `docs/modelos_transformer_timesfm.md` |
| 11 | PR `feat/transformer` → `main` | GitHub | Pendiente |

Criterio de "terminado", el mismo que se exigió al CNN-LSTM:

- ajustado con 20 pruebas;
- 24 cortes con 3 semillas;
- resultados en `reports/`;
- notebook y documentación;
- `uv run pytest` y `uv run ruff check .` en verde.

## S2. TimesFM 3 (zero-shot)

### API verificada (`timesfm==3.0.2`, 2026-10-29)

El README oficial está desactualizado: **no existe `TimesFM3Evaluator`**. La API real es:

```python
from timesfm import TimesFM3Forecaster
model = TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device="cuda")
outs = model.predict_batch(contexts, horizon, past_only_covariates=[...], ...)
# cada salida: .forecast (mediana, forma (h,)) y .quantiles
```

- `contexts`: lista de arreglos 1D. `past_only_covariates`: un arreglo `(k, ctx)` por serie.
- Los nulos al inicio del contexto se recortan y los interiores se interpolan.

### Diseño

- **Runner** `TimesFM3Runner`, en `models/pretrained/timesfm.py`:
  - `fit` no entrena: carga los pesos **una vez por proceso**, con caché por checkpoint y dispositivo, porque el backtest crea un runner por corte.
  - `predict` arma, para cada serie del `frame`, el contexto `y[ds <= corte][-context_len:]` y opcionalmente las covariables pasadas del mismo tramo. Usa las mismas columnas que ven las redes tras `drop_covariates`.
  - Columna `TIMESFM3`. Sin semillas: la inferencia es determinista.
- **Configuración** `timesfm.yaml`: `checkpoint`, `context_len`, `use_covariates`, `drop_covariates`, `use_symmetric_averaging`.
- **Tests** con un predictor falso inyectado, sin descargar pesos:
  - un contexto por serie;
  - nada posterior al corte;
  - forma de las covariables `(k, ctx)`;
  - salida 1..h;
  - caché de pesos.

### Pasos

| # | Paso | Estado |
|---|---|---|
| 1 | Runner, YAML y tests en `feat/timesfm3` | **Hecho** |
| 2 | Prueba de un corte en Colab con GPU (`run_model.py timesfm3`): mide la descarga de los pesos y el tiempo | **Hecho** en CPU: 53 s los 24 cortes |
| 3 | "Ajuste" zero-shot: rejilla `context_len` ∈ {512, 1024, 2048} × `use_covariates` ∈ {no, sí} = 6 pruebas en los mismos 12 cortes de ajuste. Hay que declarar en el informe que el espacio queda agotado en 6 pruebas, frente a 20 de las otras redes. `scripts/tune_timesfm3.py` necesita poder recibir el `sampler` (GridSampler) en `run_study` | **Hecho** con un bucle propio (Naive reajustado por corte): gana 1024 sin covariables |
| 4 | 24 cortes en Colab, guardar en `reports/`, notebook `05_timesfm3.ipynb`, documentación y PR | **Hecho** en CPU (MAE medio 0,300); PR pendiente |

### Riesgos

- Los pesos son de **licencia no comercial**, lo que es compatible con un proyecto académico. Se descargan a la caché de Hugging Face en cada runtime de Colab (~1,3 GB en fp32, sin medir).
- **Fuga de datos:** el modelo se preentrenó con miles de series públicas. Si incluyó niveles del Orinoco de 2021–2022, la validación quedaría contaminada. No se puede verificar, así que se declara como limitación.
- En CPU la inferencia resultó rápida (53 s los 24 cortes), así que no hizo falta GPU.

## Preguntas resueltas

1. TimesFM va en su propia rama `feat/timesfm3` (un PR por modelo, como el grupo), apilada sobre `feat/transformer` para no chocar en el README y el índice de notebooks.
2. Documentación: un archivo propio, `docs/modelos_transformer_timesfm.md`.
