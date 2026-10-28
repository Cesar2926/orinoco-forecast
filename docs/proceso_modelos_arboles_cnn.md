# Cómo se construyeron los modelos XGBoost y TCN

Este documento explica **qué se hizo, en qué orden y por qué** para construir los dos modelos
asignados: el ensamble de árboles (XGBoost, `booster` en el código) y la red convolucional
(TCN). Cubre el uso del conjunto de datos, la preparación, el entrenamiento, el ajuste y la
evaluación. Las tablas completas de resultados y las referencias bibliográficas están en
[`modelos_arboles_cnn.md`](modelos_arboles_cnn.md).

**Cambios respecto a la versión anterior de este documento.** Los datos pasaron de 1974–2025
a 2009–2024 (instrucción del profesor); la ejecución pasó a los runners de Daniel
(`fit`/`predict` en un corte); la validación pasó de un ajuste único con 24 ventanas a 24
cortes con reajuste en cada uno; se reportan los días 7 y 14; las etapas A y B se
reemplazaron por una opción `drop_covariates` en cada modelo; y los hiperparámetros se
reajustaron con los datos nuevos.

---

## 1. El problema

Predecir el nivel diario del río Orinoco en **Ciudad Bolívar** y **Palúa** para cada uno de
los **14 días siguientes**, usando solo la información disponible el día en que se hace el
pronóstico. El profesor pidió además mirar el error a **7 y 14 días**.

Esta condición guía todo el diseño. Si un modelo usa, aunque sea por accidente, un dato
posterior al día del pronóstico, sus métricas salen excelentes y en la práctica no sirve. Ese
error se llama **fuga de datos** (*data leakage*) y el proyecto tiene controles para
impedirlo en cada etapa.

---

## 2. El conjunto de datos

### 2.1 Fuente usada

| Archivo | Contenido | Uso |
|---|---|---|
| `data/raw/DATOS RIO ORINOCO.xlsx`, hoja `Entrenar` | Nivel diario de Ayacucho, Caicara, Ciudad Bolívar y Palúa, 2009-01-01 a 2025-03-31 (las filas desde 2025-02-25 están vacías) | **Fuente oficial.** Solo se usa 2009-01-01 a 2024-12-31 |
| `data/external/nasa_power_*.csv` | Lluvia diaria de NASA POWER en el punto de cada estación | Columnas `rain_*` del parquet |
| `data/external/noaa_oni.ascii.txt` | Índice ONI (El Niño / La Niña) | Columna `oni`, con 3 meses de retraso de publicación |

Los archivos de `data/raw/` nunca se modifican. El Excel se convierte a un parquet
(`data/interim/orinoco_dataset.parquet`) con `TimeSeriesDataset.write_parquet`, que agrega
lluvia, ONI y columnas de rezagos ya calculadas.

**Por qué solo 2009–2024:** lo pidió el profesor (los datos fuera de ese rango no son
válidos). Además, el Excel nuevo es el único archivo oficial: los archivos viejos
(`Niveles Orinoco.xlsx` desde 1974 y el histórico de Ciudad Bolívar) ya no están en
`data/raw/`.

### 2.2 Problemas encontrados en los datos (2009–2024)

- **Valores repetidos sospechosos.** Ayacucho repite el mismo valor del 2019-03-07 al
  2019-03-18 (12 días) y Ciudad Bolívar del 2020-03-24 al 2020-04-11 (19 días). Un río de
  este tamaño no se queda quieto tantos días: son datos faltantes rellenados repitiendo el
  último valor.
- **Nulos.** Solo Palúa tiene nulos, los tres 29 de febrero de 2012, 2016 y 2020.
- **Un salto sospechoso.** Palúa marca 9,79 m el 2021-10-29 entre 8,73 y 8,86 m. Se etiqueta
  y se conserva.
- **Sin fuente de referencia.** Antes se comparaba Ciudad Bolívar con un histórico
  (etiqueta `source_conflict`); ese archivo ya no existe y la etiqueta no se emite.
- **Origen desconocido.** El Excel no documenta si ya traía imputación; se declara como
  limitación.

### 2.3 Etiquetas de calidad

En lugar de corregir los datos a mano, cada valor recibe una etiqueta en la columna
`quality`:

| Etiqueta | Significado |
|---|---|
| `observed` | Valor tal como viene |
| `missing` | Sin valor |
| `suspect_flat` | Parte de una racha de 10 o más días con el mismo valor |
| `suspect_jump` | Cambio de más de 1 m en un día |

**Por qué así:** cada decisión queda explícita y es reversible cambiando una regla. Editar
los Excel a mano no deja registro de qué se cambió ni por qué.

### 2.4 Limpieza

La hace `clean_history` ([`features/dataset.py`](../src/orinoco_forecast/features/dataset.py)),
que `load_history` aplica al parquet:

1. Recorta a 2009-01-01 – 2024-12-31.
2. Los valores `suspect_flat` se convierten en nulos.
3. Los huecos de hasta 31 días se rellenan por **interpolación lineal** (una línea recta
   entre el último valor anterior y el primero posterior). Los más largos no se rellenan y
   avisan con `DataQualityWarning`. En 2009–2024 todos son cortos: se rellenan 34 filas
   (12 en Ayacucho, 19 en Ciudad Bolívar y 3 en Palúa), marcadas con `imputed = True`.
4. Descarta los primeros 60 días de 2009: las columnas de rezagos del parquet (por
   ejemplo `oni_lag_60`) todavía no existen allí. La serie efectiva empieza el 2009-03-02
   (5 784 días por estación). No se inventa ningún valor.

**El control anti-fuga de la interpolación.** Interpolar usa el valor del día siguiente al
hueco, es decir, un dato "futuro". Aquí no hay riesgo porque el último día imputado es el
2020-04-11 y el primer corte de validación es el 2021-01-26. La función
`assert_no_imputation_near_cutoffs` hace esa comprobación en `build_model_frame`; la ruta
de runners no la ejecuta, y la comprobación se verificó con esas fechas.

### 2.5 La tabla que reciben los modelos

El parquet limpio trae las **cuatro estaciones como series** (`unique_id`) en formato largo.
`execute` ajusta los dos modelos con las cuatro y filtra la predicción a Ciudad Bolívar y
Palúa.

| Columna | Qué es | Cuándo se conoce |
|---|---|---|
| `unique_id` | Estación | — |
| `ds` | Fecha | — |
| `y` | Nivel del río (lo que se predice) | Solo el pasado |
| `<estacion>_lag_<n>`, `d_<estacion>_lag_<n>` | Nivel y cambio diario de estaciones aguas arriba, rezagados | Solo el pasado |
| `rain_<estacion>`, `rain_<estacion>_lag_<n>`, `oni`, `oni_lag_<n>` | Lluvia y El Niño | Solo el pasado |
| `imputed` | Marca de valor interpolado (metadato, no es variable del modelo) | — |

**Por qué las estaciones aguas arriba:** el agua pasa primero por Ayacucho, luego por
Caicara, luego por Ciudad Bolívar y por último por Palúa. Una crecida observada hoy en
Ayacucho llega días después a Ciudad Bolívar.

**Sobre el ciclo anual.** Los modelos solo usan los términos de Fourier (`sin_*`, `cos_*`)
si el frame de entrenamiento los trae. El parquet no los incluye (`calendar_terms: 0`), así
que hoy ni el XGBoost ni la TCN ven el ciclo anual explícito; solo aparecen en la ruta
anterior (`build_model_frame`). No se midió cuánto cambiaría el resultado al activarlos.

### 2.6 Qué covariables ve cada modelo

La etapa A (solo río) y la B (río + lluvia + ONI) se reemplazaron por una lista de prefijos
`drop_covariates` en el YAML de cada modelo:

| Modelo | `drop_covariates` | Qué ve |
|---|---|---|
| XGBoost | `[]` | Todo: niveles rezagados, cambios diarios, lluvia y ONI |
| TCN | `[rain_, oni]` | Niveles rezagados y cambios diarios; no lluvia ni ONI |

**Por qué cada uno:** en pruebas de validación la lluvia y el ONI perjudicaron a la TCN y no
cambiaron al XGBoost (ver la sección 8). Cada modelo elige sus entradas con los cortes de
validación; el protocolo es el mismo para todos.

---

## 3. Cómo se dividen los datos

La división **no es por porcentajes** (70/30 o 60/20/20). En series temporales se divide
por fechas, en orden, porque mezclar días de distintos años dejaría al modelo ver el futuro.
La partición la definen `configs/backtest.yaml` (`test_start`, `n_windows`, `step_size`) y
`walk_forward_splits` ([`data/splits.py`](../src/orinoco_forecast/data/splits.py)).

| Bloque | Período | Días por estación | Para qué |
|---|---|---|---|
| Entrenamiento | 2009-03-02 → cada corte (primer corte: 2021-01-26) | 4 349 en el primero; crece en cada corte | Aprender |
| Ajuste de hiperparámetros | 12 ventanas entre 2020-05-04 y 2021-01-12 | dentro del entrenamiento | Elegir la configuración de cada modelo |
| Validación | 24 cortes, 2021-01-26 → 2022-12-17, cada 30 días; 14 días tras cada corte | 704 hasta 2022-12-31 | Comparar modelos |
| Test final | 2023-01-01 → 2024-12-31 | 731 | **Se usa una sola vez**, al elegir el modelo final del equipo |

### Validación con origen móvil

La validación no es un bloque único: son **24 pronósticos**.

1. Se elige un corte (el primero es 2021-01-26).
2. El modelo se **reajusta** con todo lo anterior al corte, pronostica los 14 días
   siguientes y se compara con lo observado.
3. El corte avanza 30 días y se repite, hasta el corte del 2022-12-17.

```text
corte 1  ──[14 días]
          30 días ──corte 2 ──[14 días]
                    ...    corte 24 ──[14 días]      │ test final 2023–2024 │
```

Cada ventana de 14 días queda separada de la siguiente (los cortes están a 30 días), así
que las 24 ventanas no se solapan.

**Por qué reajustar en cada corte:** el runner de Daniel ajusta en un solo corte y
`scripts/run_model.py --folds` lo repite 24 veces. Así cada predicción es genuinamente fuera de
muestra, a costa de un tiempo mayor: el XGBoost tarda minutos y el TCN, con 3 semillas, del
orden de una hora (estimado, no cronometrado). La versión anterior entrenaba una sola vez
(`refit: false`) y evaluaba 24 ventanas con esos pesos.

**Por qué el test final no se toca:** si se elige el modelo mirando el test, el resultado
reportado queda inflado (*sesgo de selección*). Si la evaluación final se hiciera más de una
vez, el modelo ya habría visto esos datos y la predicción dejaría de ser genuina. El test se
evalúa una sola vez, con el modelo ya elegido.

---

## 4. Cómo se mide si un modelo sirve

### La referencia: la persistencia

La **persistencia** (*Naive*) predice que el nivel de los próximos 14 días será igual al de
hoy. Para un río tan lento es un rival difícil: su error promedio en los 24 cortes es de
0,570 m en Ciudad Bolívar y 0,488 m en Palúa (días 1 a 14); a 7 días, 0,526 y 0,435 m, y a
14 días, 1,064 y 0,891 m. Un modelo que no le gana no aporta nada.

### Métricas

| Métrica | Qué dice |
|---|---|
| MAE | Error promedio en metros |
| RMSE | Como el MAE, pero castiga más los errores grandes (crecidas mal predichas) |
| MASE | MAE del día `h` dividido por el error de la persistencia a `h` días en el entrenamiento |
| NSE, KGE | Métricas estándar de hidrología; 1 es perfecto |
| **MAE relativo** (`rel_mae`) | MAE del modelo dividido por el MAE de la persistencia **en los mismos cortes** |

`--folds` reporta MAE, RMSE, NSE y `rel_mae` por paso del horizonte (los días 7 y 14 son
los pasos 7 y 14) y en el total. El NSE sale
negativo (también para la persistencia) porque cada ventana dura 14 días y su varianza
interna es pequeña; por eso se lee `rel_mae`.

### El requisito para elegir modelo

**`rel_mae` menor que 1 en cada estación**, y entre los que lo cumplen, el de menor MAE
promedio entre las dos estaciones. El criterio se fijó antes de ver resultados.

**Por qué `rel_mae` y no el MASE:** con los datos de la versión anterior (desde 1974) la
persistencia sacaba MASE 1,16 en Palúa en 2021–2022, porque esos años fueron más difíciles
que el promedio histórico. Con el MASE, un modelo podía ganarle a la persistencia en cada
ventana y aun así quedar descartado. `rel_mae` compara en los mismos días.

---

## 5. Modelo 1: ensamble de árboles (XGBoost)

### Qué es

**Gradient boosting**: cientos de árboles de decisión pequeños entrenados en cadena, donde
cada árbol corrige el error que dejaron los anteriores. Se usa la librería `mlforecast`,
que construye las variables de la serie. El runner es `BoosterRunner`
([`models/trees/tree.py`](../src/orinoco_forecast/models/trees/tree.py)).

### Cómo se preparan los datos para el modelo

| Variable | Valores | Por qué |
|---|---|---|
| Rezagos del nivel (*lags*) | 1, 2, 3, 7, 14, 21, 28 y 56 días | Inercia diaria, ciclo semanal y retrasos de hasta dos meses de la llanura inundable |
| Medias móviles | Última semana y último mes | Suavizan el ruido diario y dan la tendencia |
| Diferencias | Se predice el cambio diario, no el nivel | Los árboles no pueden predecir valores fuera del rango visto en el entrenamiento |
| Covariables históricas | Rezagadas 14, 28, 56 y 90 días (`covariate_lags`) | Ver el punto siguiente |

**El hallazgo que obliga a rezagar 14 días o más.** Se entrena un modelo distinto para cada
día del horizonte (**estrategia directa**, 14 modelos). En esa estrategia, `mlforecast`
alinea las variables externas con el **día que se pronostica**. Si Ayacucho entrara sin
rezago, el modelo del día 14 vería el nivel de Ayacucho de ese día 14, que todavía no
ocurrió. Con un rezago de al menos 14 días, todo valor usado ya es conocido el día del
corte. El código se detiene si alguien configura un rezago menor que el horizonte.

**Por qué la estrategia directa:** la alternativa, recursiva, usa la predicción de mañana
para predecir pasado mañana y acumula el error día tras día.

### Entrenamiento

1. Se limpia el historial hasta el corte y se rezagan las covariables.
2. Se entrenan 14 modelos XGBoost, uno por día del horizonte, con los datos hasta el corte.
3. Se construyen las variables del corte con los datos observados y cada modelo predice su
   día; el runner devuelve las predicciones de los días 1 a 14.
4. `--folds` repite eso en los 24 cortes y promedia las métricas.

### Configuración final

Elegida con Optuna (sección 7) sobre datos 2009–2024. Entre paréntesis, el valor de la
versión anterior.

| Hiperparámetro | Valor |
|---|---|
| Número de árboles (`n_estimators`) | 300 (900) |
| Tasa de aprendizaje (`learning_rate`) | 0,0133 (0,1201) |
| Profundidad máxima (`max_depth`) | 5 (6) |
| Fracción de filas por árbol (`subsample`) | 0,925 (0,884) |
| Fracción de variables por árbol (`colsample_bytree`) | 0,755 (0,503) |
| Peso mínimo por hoja (`min_child_weight`) | 13,7 (1,38) |
| Regularización L2 (`reg_lambda`) | 1,54 (0,273) |
| Rezagos de covariables | 14, 28, 56, 90 (14, 21, 28, 56) |

La importancia de variables no se recalculó para esta configuración.

---

## 6. Modelo 2: red convolucional (TCN)

### Qué es

**Temporal Convolutional Network**: una red neuronal que recorre la serie con filtros
convolucionales. Dos propiedades la hacen adecuada:

- **Convoluciones causales:** cada salida depende solo del pasado, así que la arquitectura
  no puede usar información futura.
- **Dilataciones:** los filtros saltan días (1, 2, 4, 8, 16), de modo que con pocas capas la
  red cubre un período largo. El alcance de la red (**campo receptivo**) es
  `1 + (tamaño del filtro − 1) × suma de dilataciones` = 1 + 1 × 31 = **32 días**.

Se usa la librería `neuralforecast` (PyTorch). El runner es `TCNRunner`
([`models/neural/cnn.py`](../src/orinoco_forecast/models/neural/cnn.py)).

### Cómo se preparan los datos para el modelo

| Entrada | Tipo | Detalle |
|---|---|---|
| Cambio diario del nivel | Objetivo | La red predice cuánto sube o baja el río cada día |
| Nivel actual (`level_y`) | Variable histórica | Para que la red sepa en qué nivel absoluto está el río |
| Niveles y cambios diarios de aguas arriba, rezagados | Variables históricas | Solo hasta el día del corte |
| Lluvia y ONI | — | Ocultas con `drop_covariates: [rain_, oni]` |

La red ve los últimos **112 días** de cada variable (ventana de entrada) y produce los 14
días de una vez. Los niveles se reconstruyen sumando los cambios predichos al nivel
observado el día del corte.

**Verificación anti-fuga:** se revisó el código de `neuralforecast` y las variables
históricas se recortan a la ventana de entrada, que termina en el corte. Además se probó en
la práctica: si los valores posteriores al corte se alteran, el pronóstico no cambia. Esa
prueba es un test del proyecto.

### La mejora clave: predecir el cambio diario

La primera versión predecía el **nivel**. Perdía contra la persistencia en el día 1: con los
datos escalados, a una red le cuesta reproducir "mañana igual que hoy" con precisión de
centímetros. Al predecir el **cambio diario**, "no cambia" equivale a predecir cero, que es
fácil de aprender. En el reajuste con datos 2009–2024 las dos mejores pruebas completas
también usan el cambio diario.

### Entrenamiento

1. Se escala cada serie con un **escalado robusto** (mediana y rango intercuartil), que no
   se distorsiona con las crecidas.
2. Se reservan los **365 días finales del entrenamiento** de cada corte para validación
   interna: si el error ahí deja de bajar durante 5 revisiones, el entrenamiento se detiene
   antes (**parada temprana**).
3. Se entrena con hasta 1000 pasos, minimizando el MAE.
4. Se repite con **3 semillas** (inicializaciones aleatorias distintas) sobre las mismas
   ventanas. Una red da resultados distintos según su inicialización, así que se reporta el
   promedio y la desviación. Una diferencia menor que esa desviación no se interpreta.
5. Con la red entrenada se pronostican los 14 días del corte. `--folds` repite el
   proceso en los 24 cortes.

No hace falta GPU ni Colab; se probó en CPU.

### Configuración final

| Hiperparámetro | Valor |
|---|---|
| Objetivo | Cambio diario |
| Ventana de entrada (`input_size`) | 112 días |
| Tamaño del filtro (`kernel_size`) | 2 |
| Dilataciones | 1, 2, 4, 8, 16 |
| Filtros por capa (`encoder_hidden_size`) | 32 |
| Tasa de aprendizaje | 0,002072 (antes 0,002656) |
| Pasos máximos | 1000 |
| Función de pérdida | MAE |
| `drop_covariates` | `[rain_, oni]` |

---

## 7. Ajuste de hiperparámetros

### Método

**Optuna** con el muestreador TPE: prueba combinaciones de hiperparámetros y se concentra en
las zonas que dieron mejores resultados. Objetivo: `rel_mae` promedio entre las dos
estaciones.

### Decisiones

- **Mismo presupuesto para los dos modelos:** 20 combinaciones completas cada uno. Si un
  modelo recibe más intentos, la comparación mide el esfuerzo de ajuste, no el modelo.
- **Ventanas de ajuste anteriores a la validación** (2020). Si se ajustara en las mismas
  ventanas de 2021–2022 que deciden la comparación, el modelo ajustado parecería mejor de
  lo que es.
- **Ventanas cada 23 días en lugar de 30.** El control anti-fuga detectó que una rejilla
  cada 30 días caía sobre la racha interpolada de marzo de 2020. Con 23 días se esquiva. El
  costo es que las ventanas de ajuste no cubren la temporada seca de febrero a abril.
- **Estudios nuevos para los datos 2009–2024**, con otro nombre de estudio y base de datos,
  para no mezclarlos con los de los datos anteriores.

### Resultado

| Modelo | Mejor `rel_mae` medio en las ventanas de ajuste | Detalle |
|---|---|---|
| XGBoost | 0,490 | Ciudad Bolívar 0,437, Palúa 0,543; modelo más regularizado que antes |
| TCN | 0,606 | Ciudad Bolívar 0,669, Palúa 0,542; 7 configuraciones podadas por campo receptivo menor que la ventana de entrada |

El código de Optuna (`scripts/tune_model.py`) se recupera del commit `b87533d` de `main`
(PR #7); Daniel lo eliminó después. Optuna evaluó con el protocolo anterior (un ajuste único y 12
ventanas); la validación final reajusta en cada corte, así que los parámetros son válidos
pero la medición no es idéntica.

---

## 8. Resultados

Validación, 24 cortes, MAE en metros y `rel_mae` contra la persistencia. TCN: promedio de 3
semillas.

| Estación | Horizonte | Persistencia | XGBoost | TCN |
|---|---|---|---|---|
| Ciudad Bolívar | día 7 | 0,526 | 0,279 / 0,53 | 0,279 / 0,53 |
| Ciudad Bolívar | día 14 | 1,064 | 0,444 / 0,42 | 0,511 / 0,48 |
| Ciudad Bolívar | días 1 a 14 | 0,570 | 0,278 / 0,49 | 0,294 / 0,52 |
| Palúa | día 7 | 0,435 | 0,360 / 0,83 | 0,403 / 0,93 |
| Palúa | día 14 | 0,891 | 0,548 / 0,62 | 0,631 / 0,71 |
| Palúa | días 1 a 14 | 0,488 | 0,341 / 0,70 | 0,383 / 0,78 |

### Lectura

- **Los dos modelos cumplen el requisito** (`rel_mae < 1`) en las dos estaciones y en los
  días 7, 14 y total.
- **XGBoost tiene el menor MAE medio entre estaciones** (0,309 m frente a 0,339 m de la
  TCN). En Ciudad Bolívar la diferencia es menor que la variación entre semillas de la TCN;
  en Palúa es mayor.
- **El día 7 en Palúa es el punto débil:** `rel_mae` de 0,83 (XGBoost) y 0,93 (TCN), poco
  mejor que la persistencia.
- **La lluvia y el ONI no ayudan.** En una prueba de 6 cortes, el TCN con todas las columnas
  quedó en `rel_mae` 0,98 (Ciudad Bolívar) y 1,12 (Palúa); sin lluvia ni ONI, 0,71 y 0,80;
  sin covariables, 0,94 y 1,04. Al XGBoost le dio igual: con todas las columnas 0,487 y
  0,698 en 24 cortes, con solo niveles 0,504 y 0,684. Una explicación plausible, no
  verificada: la lluvia puntual en cuatro localidades aporta poco frente a los niveles
  aguas arriba, que ya integran la lluvia de la cuenca.

---

## 9. Limitaciones

- Los datos de base no documentan si ya traían una imputación.
- Las ventanas de ajuste no cubren la temporada seca de febrero a abril.
- El reajuste en cada corte difiere del protocolo con que Optuna eligió los hiperparámetros.
- Los modelos no ven el ciclo anual explícito (Fourier) en la ruta de runners.
- Se entrenan con las cuatro estaciones, no solo con las dos objetivo; no se probó la
  alternativa.
- La comparación de covariables del TCN usa 6 cortes, no 24; es una muestra más ruidosa y
  más difícil.
- Palúa solo ve a Ciudad Bolívar mediante `ciudad_bolivar_lag_1` y `d_ciudad_bolivar_lag_1`;
  el ARIMAX del equipo la usa como exógena completa.
- El test final no se ha evaluado: se hará una vez, al elegir entre los modelos.

---

## 10. Cómo reproducirlo

```bash
uv sync

# Dataset (una vez)
uv run python scripts/run_model.py booster --write-dataset

# Un corte contra la persistencia
uv run python scripts/run_model.py booster --against naive
uv run python scripts/run_model.py tcn --against naive

# Validación de 24 cortes (n_windows y step_size en configs/backtest.yaml)
uv run python scripts/run_model.py booster --against naive --folds
uv run python scripts/run_model.py tcn --against naive --folds

# Dashboard para explorar los resultados de run_model.py
make dashboard
```

| Dónde está cada cosa | Archivo |
|---|---|
| Protocolo compartido (horizonte, test, cortes) | `configs/backtest.yaml` |
| Limpieza y período | `configs/data.yaml`, `src/orinoco_forecast/features/dataset.py` |
| Hiperparámetros y `drop_covariates` | `configs/models/tree.yaml`, `configs/models/cnn.yaml` |
| Modelos | `src/orinoco_forecast/models/trees/tree.py`, `src/orinoco_forecast/models/neural/cnn.py` |
| Ejecución, métricas y validación | `src/orinoco_forecast/evaluation/`, `scripts/run_model.py` |
| Resultados del ajuste | `reports/tuning/` |
| Predicciones de un corte | `data/predictions/<runner>/` |
