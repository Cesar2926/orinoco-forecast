# Modelos de ensamble de árboles (XGBoost) y red convolucional (TCN)

Este documento describe cómo se prepararon los datos, cómo se evalúan los modelos, cómo
se construyeron y ajustaron los dos modelos de esta parte del proyecto (XGBoost, llamado
`booster` en el registro de runners, y TCN) y qué resultados obtuvieron. Está pensado como
insumo para el informe final. El protocolo general está en el [README](../README.md) y el
plan de los seis modelos en [`plan_modelos.md`](plan_modelos.md). El proceso paso a paso
(decisiones, datos y entrenamiento) está en
[`proceso_modelos_arboles_cnn.md`](proceso_modelos_arboles_cnn.md).

Los números de resultados salen de `scripts/validate_cv.py` y de los CSV de
`reports/tuning/`. Las predicciones (`data/predictions/`) no se versionan en git: se
regeneran con los comandos de la sección 9.

---

## 1. Resumen

Se implementaron dos modelos para pronosticar a 14 días el nivel diario del Orinoco en
Ciudad Bolívar y Palúa: un ensamble de árboles con *gradient boosting* (XGBoost, vía
`mlforecast`) y una red convolucional temporal (TCN, vía `neuralforecast`). Los dos se
ejecutan con los runners de la arquitectura común (`fit`/`predict` en un corte) y se
validan con 24 cortes cronológicos (2021-01-26 a 2022-12-17), sin tocar el test final
(desde 2023-01-01).

Con datos 2009–2024, los dos superan a la persistencia (Naive) en las dos estaciones y en
todos los horizontes medidos (`rel_mae < 1`). El XGBoost tiene el menor MAE medio entre
estaciones (0,309 m frente a 0,339 m de la TCN). A 14 días la TCN y el XGBoost están
cerca; el punto débil de los dos es el día 7 en Palúa (`rel_mae` 0,83 y 0,93). La lluvia y
el ONI perjudican a la TCN y no cambian al XGBoost; por eso la TCN usa solo niveles de río
(`drop_covariates: [rain_, oni]`).

---

## 2. Cambios respecto a la versión anterior

| Tema | Antes | Ahora |
|---|---|---|
| Datos | `Niveles Orinoco.xlsx` desde 1974 (37 366 filas) | `DATOS RIO ORINOCO.xlsx`, hoja `Entrenar`, solo 2009-01-01 a 2024-12-31 (período válido según el profesor) |
| Ejecución | Scripts propios con `--stage A/B` y `cross_validation` de Nixtla | Runners de Daniel (`scripts/run_model.py`, `fit`/`predict` en un corte) |
| Validación | Un ajuste único (`refit: false`) y 24 ventanas evaluadas | 24 cortes con reajuste en cada uno (`scripts/validate_cv.py`) |
| Horizontes | Promedio de 14 días | Se reportan además los días 7 y 14 |
| Etapas A y B | Dos conjuntos de variables (río / río + lluvia + ONI) | Reemplazadas por `drop_covariates` en cada YAML |
| Ciudad Bolívar | No era covariable | `ciudad_bolivar_lag_1` y `d_ciudad_bolivar_lag_1` están entre las columnas de rezagos del parquet |
| Hiperparámetros | Ajustados con datos 1974+ | Reajustados con datos 2009–2024 (sección 7) |
| `source_conflict` | Emitida (comparaba con el histórico de Ciudad Bolívar) | Ya no se emite: el histórico fue eliminado de `data/raw/` |

---

## 3. Datos y preparación

### Fuente y etiquetas de calidad

La fuente es `data/raw/DATOS RIO ORINOCO.xlsx` (hoja `Entrenar`): niveles diarios de
Ayacucho, Caicara, Ciudad Bolívar y Palúa. La lee `TimeSeriesDataset.extract_levels` y se
usa solo 2009-01-01 a 2024-12-31 (5 844 días por estación). Ningún archivo de `data/raw/`
se modifica. Cada valor recibe una etiqueta `quality` (`observed`, `missing`,
`suspect_flat`, `suspect_jump`); el detalle está en la sección "Calidad de los datos" del
README. En 2009–2024:

- `missing`: 3 valores de Palúa (29 de febrero de 2012, 2016 y 2020).
- `suspect_flat`: Ayacucho del 2019-03-07 al 2019-03-18 (12 días) y Ciudad Bolívar del
  2020-03-24 al 2020-04-11 (19 días).
- `suspect_jump`: un valor de Palúa (2021-10-29, 9,79 m). Se etiqueta, no se modifica.

### Limpieza e interpolación

La limpieza está en [`features/dataset.py`](../src/orinoco_forecast/features/dataset.py)
(`clean_series`) y la usan dos rutas:

1. `build_model_frame` (ruta anterior, con etapas A y B): recorta a 2009–2024, limpia y
   arma una fila por estación objetivo con las estaciones aguas arriba como columnas.
2. `clean_history`, que aplica `load_history` en la ruta actual de runners: recorta a
   2009–2024, anula las rachas `suspect_flat` e interpola huecos de hasta 31 días
   (`imputed = True` en las filas rellenadas: 34 en total). Además descarta los primeros
   60 días de 2009, donde las columnas de rezagos del parquet (hasta `oni_lag_60`) aún no
   existen; la serie efectiva empieza el 2009-03-02 (5 784 días por estación).

El último día imputado es el 2020-04-11, anterior a todos los cortes de validación
(2021-01-26 en adelante), así que la interpolación no usa observaciones posteriores a un
corte. La ruta de runners no ejecuta `assert_no_imputation_near_cutoffs`; esa comprobación
queda hecha a mano con las fechas anteriores (y en `build_model_frame`, que sí la tiene
disponible).

### Qué ven los runners

El frame que reciben los runners es el parquet de `TimeSeriesDataset.write_parquet`
(`data/interim/orinoco_dataset.parquet`), limpiado con `clean_history`. Trae las **cuatro
estaciones como series** (`unique_id`), el nivel `y`, lluvia (`rain_<estacion>`), ONI y
columnas de rezagos ya calculadas (`<estacion>_lag_<n>`, `d_<estacion>_lag_<n>`,
`rain_<estacion>_lag_<n>`, `oni_lag_<n>`). `execute` ajusta el modelo con las cuatro series
y filtra la predicción a las dos estaciones objetivo.

- **XGBoost** vuelve a rezagar todas las covariables históricas por 14, 28, 56 y 90 días
  (`covariate_lags`); el mínimo debe ser al menos el horizonte, o el código falla.
- **TCN** declara todas las covariables como históricas (`hist_exog_list`), que
  `neuralforecast` corta a los `input_size` días que terminan en cada corte.
- **`drop_covariates`** (lista de prefijos de columna en `tree.yaml` y `cnn.yaml`) oculta
  columnas a cada modelo. XGBoost: `[]` (ve todo). TCN: `[rain_, oni]` (ve niveles
  rezagados y cambios diarios, no lluvia ni ONI).

**Ciclo anual (Fourier).** Ambos runners solo usan términos de Fourier (`sin_*`, `cos_*`)
si el frame de entrenamiento los trae. El parquet no los incluye (`calendar_terms: 0` en
`configs/data.yaml`), así que en la ruta actual ni XGBoost ni la TCN ven el ciclo anual
explícito; solo lo capturan de forma indirecta a través de los rezagos y de la historia.
Los términos de Fourier sí existen en `build_model_frame`, que era la ruta anterior.
No medí cuánto cambiaría el resultado al activarlos.

---

## 4. Protocolo de evaluación

### Partición cronológica

| Bloque | Qué es |
|---|---|
| Entrenamiento | Todo `ds <= cutoff`; crece con cada corte. Cada corte reajusta el modelo. |
| Validación | Los 14 días siguientes a cada corte. 24 cortes de 2021-01-26 a 2022-12-17, uno cada 30 días (`n_windows` y `step_size` en `configs/backtest.yaml`). |
| Test final | `ds >= 2023-01-01` (hasta 2024-12-31). Nunca evaluado; se evalúa una sola vez con el modelo elegido por el equipo. |

Los cortes salen de `walk_forward_splits` ([`data/splits.py`](../src/orinoco_forecast/data/splits.py)),
que falla si una ventana de validación invade el test. Cada ventana dura 14 días y los
cortes están separados 30, así que las ventanas no se solapan.

Una validación de 24 cortes es la medición con la que se compara y se elige. Mirar el test
antes para decidir sería fuga de datos aplicada a la decisión. La evaluación final se hace
una sola vez: si se repite, el modelo ya vio esos datos y la predicción deja de ser
genuina.

### Métricas

| Métrica | Definición breve |
|---|---|
| MAE | Error absoluto medio, en metros. |
| RMSE | Raíz del error cuadrático medio, en metros. |
| MASE | MAE del día `h` dividido por el MAE en entrenamiento de la persistencia a `h` días ([Hyndman y Koehler 2006](https://www.doi.org/10.1016/J.IJFORECAST.2006.03.001)). |
| `rel_mae` | MAE del modelo / MAE de la persistencia (Naive) en los mismos cortes. |
| NSE | Eficiencia de Nash-Sutcliffe ([1970](https://explore.openalex.org/works/w2033904036)); 1 es perfecto, 0 equivale a predecir la media. |
| KGE | Eficiencia de Kling-Gupta ([Gupta et al. 2009](https://ag.purdue.edu/hydrology/wiki/environmental-informatics/modeling-resources/attachments/gupta-et-al-2009.pdf)). |

`validate_cv.py` reporta MAE, RMSE, NSE y `rel_mae` promediados sobre los 24 cortes, para
los días 7, 14 y el total (pasos 1 a 14). **El NSE sale negativo** también para Naive
(entre −2,5 y −3,3 en el total): cada ventana dura 14 días y su varianza interna es
pequeña, así que el NSE por ventana castiga cualquier error. Es mejor leer `rel_mae`.

**Por qué el requisito es `rel_mae < 1` y no MASE < 1.** La escala del MASE viene del
historial de entrenamiento, no del período evaluado, y puede dar valores mayores que 1 a la
propia persistencia si el período evaluado fue más difícil que el historial. `rel_mae`
compara contra la persistencia en las mismas ventanas, así que no tiene ese problema.

**Criterio de selección** (fijado antes de ver resultados): `rel_mae < 1` en cada estación
y, entre los que cumplen, el menor MAE medio entre las dos estaciones. Desempate: días 10 a
14, luego crecidas, luego costo.

---

## 5. Modelo 1: ensamble de árboles (XGBoost)

### Qué es

Un **árbol de decisión** divide los datos con preguntas del tipo "¿el nivel de ayer era
mayor que 10 m?" y predice un valor en cada hoja. El **gradient boosting** construye cientos
de árboles pequeños en secuencia: cada árbol nuevo corrige el error que dejaron los
anteriores, y la predicción final es la suma de todos, cada uno multiplicado por la tasa de
aprendizaje (*learning rate*). XGBoost es una implementación eficiente con regularización
([parámetros](https://xgboost.readthedocs.io/en/stable/parameter.html)). Este tipo de
modelo ganó la competencia de pronóstico M5
([Makridakis et al. 2022](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1346-1364.html)).

### Cómo se adaptó a la serie

El código está en [`models/trees/tree.py`](../src/orinoco_forecast/models/trees/tree.py)
(`BoosterRunner`) y la configuración en [`configs/models/tree.yaml`](../configs/models/tree.yaml).

- **Estrategia directa** (*direct strategy*): se entrena un modelo por cada día del
  horizonte (14 modelos). La alternativa recursiva usa su propia predicción del día 1 como
  dato para el día 2 y acumula error
  ([Ben Taieb et al. 2012](https://research.monash.edu/en/publications/a-review-and-comparison-of-strategies-for-multi-step-ahead-time-s/),
  [mlforecast](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/one_model_per_horizon.html)).
- **Diferencias** (`target_differences: [1]`): el modelo predice el cambio diario del
  nivel, no el nivel. Un árbol no puede predecir valores fuera del rango que vio al
  entrenar, así que modelando niveles nunca superaría el máximo histórico en una crecida
  récord.
- **Rezagos** (*lags*) del objetivo: 1, 2, 3, 7, 14, 21, 28 y 56 días.
- **Medias móviles** (*rolling means*) de 7 y 30 días sobre el rezago 1.
- **Covariables rezagadas ≥ 14 días.** Hallazgo verificado en `mlforecast` 1.1.0: con la
  estrategia directa, las covariables dinámicas se alinean con la fecha objetivo `t+h`, no
  con el corte. Si se pasara el nivel de Ayacucho sin rezagar, el modelo del día 14 usaría
  el valor de Ayacucho 14 días después del corte, que no se conoce. Por eso todas las
  covariables históricas se desplazan antes por `covariate_lags` días y el código falla si
  el mínimo es menor que el horizonte. Ver
  [exógenas en mlforecast](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html).

### Hiperparámetros finales

Elegidos con Optuna (sección 7) sobre datos 2009–2024. Valores en `configs/models/tree.yaml`.

| Parámetro | Rango buscado | Valor final | Valor de la versión anterior |
|---|---|---|---|
| `n_estimators` (número de árboles) | 300 a 1500, paso 100 | 300 | 900 |
| `learning_rate` | 0,01 a 0,15 (logarítmico) | 0,0133 | 0,1201 |
| `max_depth` (profundidad) | 3 a 9 | 5 | 6 |
| `subsample` (filas por árbol) | 0,6 a 1,0 | 0,9246 | 0,8839 |
| `colsample_bytree` (columnas por árbol) | 0,5 a 1,0 | 0,7551 | 0,5034 |
| `min_child_weight` | 1 a 20 (logarítmico) | 13,704 | 1,379 |
| `reg_lambda` (regularización L2) | 0,1 a 20 (logarítmico) | 1,5413 | 0,2729 |
| Rezagos de covariables | 14-21-28-56 / 14-28-56-90 / 14-21-28-56-90 | 14-28-56-90 | 14-21-28-56 |

Con datos 2009–2024 el ajuste eligió un modelo más regularizado: menos árboles, tasa de
aprendizaje menor y `min_child_weight` mayor. Una explicación plausible, no verificada, es que
ahora hay menos historia (unas 5 800 filas por estación) y el ajuste prefiere un modelo más
restringido.

La importancia de variables no se recalculó para esta configuración. En la versión anterior,
con datos desde 1974 y la ruta de etapas, dominaban las medias móviles del cambio diario,
la época del año y los niveles de Caicara y Ayacucho con 14 a 56 días de rezago; esos
resultados no se pueden extrapolar a la configuración actual.

---

## 6. Modelo 2: red convolucional temporal (TCN)

### Qué es

Una **convolución** aplica el mismo filtro (un conjunto pequeño de pesos) a cada tramo de
la serie. En una **convolución causal** la salida de un día solo depende de ese día y de
los anteriores, nunca de los posteriores. Una **convolución dilatada** salta días: con
dilatación 4 el filtro mira los días `t`, `t-4`, `t-8`... Apilando capas con dilataciones
1, 2, 4, 8, 16 la red ve muy atrás con pocas capas. El **campo receptivo** (*receptive
field*) es cuántos días hacia atrás puede llegar a ver la salida:

```text
campo receptivo = 1 + (kernel_size − 1) × suma(dilations) = 1 + 1 × 31 = 32 días
```

Implementación: [TCN de neuralforecast](https://nixtlaverse.nixtla.io/neuralforecast/models.tcn.html)
3.3.0. Código en [`models/neural/cnn.py`](../src/orinoco_forecast/models/neural/cnn.py)
(`TCNRunner`) y configuración en [`configs/models/cnn.yaml`](../configs/models/cnn.yaml).

### Configuración final

| Parámetro | Valor | Comentario |
|---|---|---|
| `target_transform` | `difference` | Ver "Predecir diferencias". |
| `input_size` (días de entrada) | 112 | Buscado entre 28, 56 y 112. Debe ser ≥ campo receptivo. |
| `kernel_size` | 2 | Buscado entre 2 y 3. |
| `dilations` | 1, 2, 4, 8, 16 | Campo receptivo de 32 días. |
| `encoder_hidden_size` (filtros por capa) | 32 | Buscado entre 32, 64 y 128. |
| Decodificador | 2 capas de 64 | Fijo. |
| Pérdida | MAE | Coincide con la métrica de selección. |
| `learning_rate` | 0,002072 | Buscado entre 1e-4 y 3e-3. Antes: 0,002656. |
| `max_steps` | 1000 | Buscado entre 500 y 1000. |
| Escalado | `robust` (mediana y rango intercuartil) | Las crecidas no distorsionan la escala. |
| Parada temprana | Paciencia 5 controles, cada 50 pasos, sobre los últimos 365 días de entrenamiento de cada corte | |
| `drop_covariates` | `[rain_, oni]` | Ve niveles rezagados y cambios diarios; no lluvia ni ONI. |
| Semillas | 1, 2, 3 | Se reporta la media (y la desviación) entre semillas. |

### Predecir diferencias

La primera versión de la TCN predecía el nivel directamente y perdía contra la persistencia
en el día 1: el nivel de mañana es casi igual al de hoy, y la red tenía que reconstruir ese
valor exacto a partir de una entrada escalada. Con `target_transform: difference` la red
predice el cambio diario `y_t − y_{t−1}`. El nivel actual entra como covariable histórica
(`level_y`) y los niveles se reconstruyen sumando los cambios predichos al nivel observado
en el corte. Así, predecir "cero cambio" equivale exactamente a la persistencia, y la red
solo tiene que aprender cuánto se aparta el río de ella. En el ajuste de hiperparámetros
con datos 1974+ las pruebas con diferencias superaron claramente a las de niveles; en el
reajuste 2009–2024 las dos mejores pruebas completas también usan diferencias.

### Semillas

Una red empieza con pesos aleatorios, y dos entrenamientos con distinta semilla dan
resultados distintos. Se entrenan 3 semillas sobre las mismas ventanas y se reporta la media
y la desviación entre semillas. Si la desviación es comparable a la diferencia entre dos
modelos, esa diferencia no es concluyente. Las predicciones de las semillas no se promedian
para evaluar.

### Verificación anti-fuga de las covariables históricas

En `neuralforecast` las covariables históricas (`hist_exog_list`) se cortan a los
`input_size` días que terminan en cada corte. Además hay una prueba empírica
([`tests/test_cnn.py`](../tests/test_cnn.py), [`tests/test_runner_tcn.py`](../tests/test_runner_tcn.py)):
alterar los valores posteriores al corte deja los pronósticos idénticos.

---

## 7. Ajuste de hiperparámetros

### Método

Se usó **Optuna** con el muestreador **TPE** (*Tree-structured Parzen Estimator*). TPE
separa las pruebas ya hechas en "buenas" y "malas" según su puntaje y propone nuevas
configuraciones donde la densidad de las buenas es alta respecto de las malas; es más
eficiente que una búsqueda aleatoria cuando el presupuesto es pequeño.

- **Mismo presupuesto:** 20 pruebas completas por modelo. Si un modelo recibe más
  intentos, la comparación mide el esfuerzo de ajuste, no el modelo.
- **Objetivo:** `rel_mae` medio entre las dos estaciones.
- **TCN:** una semilla por prueba para acotar el costo. Las configuraciones cuyo campo
  receptivo supera `input_size` se descartan sin entrenar (7 pruebas podadas).
- **Ventanas de ajuste:** 12 ventanas anteriores a la validación (cortes del 2020-05-04 al
  2021-01-12, cada 23 días, objetivos hasta 2021-01-26). Los hiperparámetros no se eligen
  con los 24 cortes de validación: eso dejaría de hacerlos una medición independiente.
  Avanzan 23 días y no 30 porque el hueco imputado de 2020 (Ciudad Bolívar, 2020-03-24 a
  2020-04-11) impide cortes a menos de 14 días de filas imputadas.

El código de Optuna (`evaluation/tuning.py`, `scripts/tune_model.py`) vive en la rama
`feat/datos-2009-2024`; Daniel lo eliminó de `main` al rehacer los runners. Los estudios
se guardaron en `artifacts/optuna/` (no versionado) y las pruebas en
`reports/tuning/xgboost_trials.csv` y `reports/tuning/tcn_trials.csv`.

**Importante:** Optuna evaluó con el protocolo anterior (un ajuste único y 12 ventanas),
mientras que `validate_cv.py` reajusta el modelo en cada corte. Los parámetros siguen
siendo válidos, pero la medición no es idéntica.

### Resultados del ajuste (datos 2009–2024)

| Modelo | Pruebas | Mejor puntaje (`rel_mae` medio) | Otras mejores |
|---|---|---|---|
| XGBoost | 20 completas (más 1 prueba que quedó abierta en el CSV) | 0,4902 (prueba 14: Ciudad Bolívar 0,437, Palúa 0,543) | 0,4908 (prueba 18), 0,4938 (prueba 11); peor 0,6349 |
| TCN | 20 completas + 7 podadas | 0,6058 (prueba 25: Ciudad Bolívar 0,669, Palúa 0,542) | 0,7230 (prueba 18) |

La mejor prueba del TCN se separa mucho del resto de las completas (0,606 frente a 0,723).
El CSV de XGBoost tiene 21 filas porque el estudio terminó con una prueba en estado
`RUNNING`; no tiene valor y no cuenta entre las 20 completas.

---

## 8. Resultados

Promedios sobre los 24 cortes de validación (2021-01-26 a 2022-12-17). MAE en metros.
TCN: media entre 3 semillas (± desviación entre semillas). Booster: sin desviación porque es
una sola corrida con `random_state: 42`. Fuente: `scripts/validate_cv.py` con los hiperparámetros de la
sección 7.

### Resultados por estación y horizonte

| Estación | Horizonte | Naive | XGBoost (MAE / `rel_mae`) | TCN (MAE / `rel_mae`) |
|---|---|---|---|---|
| Ciudad Bolívar | día 7 | 0,526 | 0,279 / 0,53 | 0,279 ± 0,034 / 0,53 ± 0,065 |
| Ciudad Bolívar | día 14 | 1,064 | 0,444 / 0,42 | 0,511 ± 0,058 / 0,48 ± 0,055 |
| Ciudad Bolívar | días 1 a 14 | 0,570 | 0,278 / 0,49 | 0,294 ± 0,037 / 0,52 ± 0,064 |
| Palúa | día 7 | 0,435 | 0,360 / 0,83 | 0,403 ± 0,028 / 0,93 ± 0,064 |
| Palúa | día 14 | 0,891 | 0,548 / 0,62 | 0,631 ± 0,024 / 0,71 ± 0,027 |
| Palúa | días 1 a 14 | 0,488 | 0,341 / 0,70 | 0,383 ± 0,018 / 0,78 ± 0,037 |

### Qué modelo cumple el requisito

Los dos cumplen `rel_mae < 1` en cada estación y horizonte, incluidos los días 7 y 14. Por
la métrica principal (MAE medio entre estaciones en los días 1 a 14):

| XGBoost | TCN (media de semillas) |
|---|---|
| 0,309 m | 0,339 m |

XGBoost queda primero por la métrica principal. En Ciudad Bolívar la diferencia a 7 días
(0,279 frente a 0,279) es nula y en los días 1 a 14 (0,278 frente a 0,294) es menor que la
desviación entre semillas de la TCN (0,037), así que ahí no es concluyente; en Palúa
(0,341 frente a 0,383) es mayor que esa desviación (0,018). **El punto débil es el día 7
en Palúa**: `rel_mae` de 0,83 (XGBoost) y 0,93 (TCN), es decir, solo una ganancia pequeña
sobre la persistencia. La elección final corresponde al equipo, comparando con los otros
modelos.

### Qué aportan la lluvia y el ONI

Experimento sobre una muestra más barata (6 cortes, uno cada 120 días, del 2021-04-26 al
2022-12-17; la muestra es más difícil que los 24 cortes: en ella el XGBoost pierde contra
Naive a 7 días). `rel_mae` de los días 1 a 14, media de 3 semillas:

| Entrada del TCN | Ciudad Bolívar | Palúa |
|---|---|---|
| Todas las columnas (con lluvia y ONI) | 0,98 | 1,12 |
| Solo niveles (sin lluvia ni ONI) | 0,71 | 0,80 |
| Sin covariables | 0,94 | 1,04 |

Con lluvia y ONI el TCN se sobreajusta y llega a perder contra la persistencia; con solo
niveles mejora claramente. Por eso `cnn.yaml` declara `drop_covariates: [rain_, oni]`.

Para el XGBoost, sobre los 24 cortes (`rel_mae` de los días 1 a 14):

| Entrada del XGBoost | Ciudad Bolívar | Palúa |
|---|---|---|
| Todas las columnas | 0,487 | 0,698 |
| Solo niveles | 0,504 | 0,684 |

La diferencia es de uno o dos centésimos y cambia de signo según la estación, así que
`tree.yaml` conserva `drop_covariates: []`. Los árboles ignoran bien las columnas que no
sirven; la red no. Es la misma conclusión que la comparación anterior entre etapas A y B,
con otros números: allí la lluvia tampoco ayudaba.

Cada modelo puede elegir sus entradas porque se decide con los cortes de validación; el
protocolo (cortes, datos limpios, horizonte, métricas) es el mismo para todos.

---

## 9. Limitaciones

- **Datos de base.** El Excel no documenta si ya traía imputación previa; en las cuatro
  estaciones no se puede distinguir un valor observado de uno imputado fuera de las reglas
  del README. Hay 34 filas interpoladas (2 rachas planas y 3 días 29 de febrero) y un valor
  de Palúa etiquetado `suspect_jump` (2021-10-29) que se conserva.
- **Sin Fourier en la ruta de runners.** El parquet no trae `sin_*`/`cos_*`
  (`calendar_terms: 0`), así que ni XGBoost ni la TCN ven el ciclo anual explícito. No se
  midió cuánto cambiaría.
- **Entrenan con las cuatro estaciones.** `execute` pasa a los runners las cuatro series y
  filtra la predicción a las dos objetivo; la TCN y el XGBoost aprenden también de
  Ayacucho y Caicara. No se probó qué pasa si se entrenan solo con las dos objetivo.
- **Reajuste en cada corte.** `validate_cv.py` reentrena en cada corte, distinto del
  protocolo anterior (un ajuste único) con el que Optuna eligió los hiperparámetros. El TCN
  con 24 cortes y 3 semillas tarda del orden de una hora en CPU (estimado; no se midió con
  cronómetro).
- **Ventanas de ajuste sin la temporada seca completa.** Las 12 ventanas van de mayo de
  2020 a enero de 2021; el hueco imputado de 2020 impide cortes en febrero–abril.
- **Una prueba abierta en el estudio de XGBoost.** Ver sección 7; no afecta las 20 completas.
- **Muestras distintas.** La comparación de covariables del TCN usa 6 cortes, no 24; una
  muestra más ruidosa y más difícil. Sirve para decidir la configuración, no para declarar
  un modelo ganador.
- **El test final no se ha evaluado.** Los números de este documento son de validación.
- **NSE negativo** por usar ventanas de 14 días (sección 4). Se interpreta `rel_mae`.
- **Sin datos de la represa de Guri.** Palúa está cerca de la confluencia con el Caroní,
  cuyo caudal regulado no está disponible.
- **No se calibraron los ensambles ni se combinó con ARIMAX.** El ARIMAX de Daniel usa
  Ciudad Bolívar como exógena de Palúa; aquí el booster y la TCN solo la ven vía
  `ciudad_bolivar_lag_1` y `d_ciudad_bolivar_lag_1`.

---

## 10. Cómo reproducir

```bash
uv sync

# Dataset (una vez; genera data/interim/orinoco_dataset.parquet)
uv run python scripts/run_model.py booster --write-dataset

# Un corte (último de validación) contra Naive
uv run python scripts/run_model.py booster --against naive
uv run python scripts/run_model.py tcn --against naive

# Validación de 24 cortes (n_windows y step_size en configs/backtest.yaml)
uv run python scripts/validate_cv.py booster
uv run python scripts/validate_cv.py tcn

# Experimento de covariables (solo exploración)
uv run python scripts/validate_cv.py tcn --windows 6 --step 120 --covariates levels

# Tablero (lee data/predictions/<runner>/*.parquet generados por run_model.py)
uv run streamlit run app/dashboard.py
```

El ajuste de hiperparámetros con Optuna (`scripts/tune_model.py`) solo existe en la rama
`feat/datos-2009-2024`; sus resultados están en `reports/tuning/`. Tiempos medidos en CPU
(sin GPU): XGBoost, minutos para 24 cortes; TCN con 3 semillas, del orden de una hora para 24
cortes (estimado).

---

## 11. Referencias

- Ben Taieb, S., Bontempi, G., Atiya, A. y Sorjamaa, A. (2012). A review and comparison
  of strategies for multi-step ahead time series forecasting. *Expert Systems with
  Applications*, 39(8). [Enlace](https://research.monash.edu/en/publications/a-review-and-comparison-of-strategies-for-multi-step-ahead-time-s/).
- Gupta, H. V. et al. (2009). Decomposition of the mean squared error and NSE performance
  criteria (KGE). *Journal of Hydrology*, 377.
  [PDF](https://ag.purdue.edu/hydrology/wiki/environmental-informatics/modeling-resources/attachments/gupta-et-al-2009.pdf).
- Hyndman, R. J. y Koehler, A. B. (2006). Another look at measures of forecast accuracy
  (MASE). *International Journal of Forecasting*, 22(4).
  [DOI](https://www.doi.org/10.1016/J.IJFORECAST.2006.03.001).
- Makridakis, S., Spiliotis, E. y Assimakopoulos, V. (2022). M5 accuracy competition:
  Results, findings, and conclusions. *International Journal of Forecasting*, 38(4).
  [Enlace](https://ideas.repec.org/a/eee/intfor/v38y2022i4p1346-1364.html).
- Nash, J. E. y Sutcliffe, J. V. (1970). River flow forecasting through conceptual models
  (NSE). *Journal of Hydrology*, 10(3). [Enlace](https://explore.openalex.org/works/w2033904036).
- mlforecast: [un modelo por horizonte](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/one_model_per_horizon.html)
  y [variables exógenas](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html).
- neuralforecast: [TCN](https://nixtlaverse.nixtla.io/neuralforecast/models.tcn.html).
- XGBoost: [parámetros](https://xgboost.readthedocs.io/en/stable/parameter.html).
- NASA POWER: [API diaria](https://power.larc.nasa.gov/docs/services/api/temporal/daily/).
- NOAA CPC: [índice ONI](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt).
