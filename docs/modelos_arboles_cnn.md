# Modelos de ensamble de árboles (XGBoost) y red convolucional (TCN)

> **Documento histórico.** Los resultados, etapas A/B y comandos de validación cruzada de
> este documento corresponden al pipeline anterior. La ruta actual es
> `scripts/run_model.py <runner> --against naive`, con salidas en
> `data/predictions/<runner>/{predictions,statistics}.parquet`.

Este documento describe cómo se prepararon los datos, cómo se evalúan los modelos, cómo
se construyeron y ajustaron los dos modelos de esta parte del proyecto (XGBoost y TCN) y
qué resultados obtuvieron. Está pensado como insumo para el informe final. El protocolo
general del proyecto está en el [README](../README.md) y el plan de los seis modelos en
[`plan_modelos.md`](plan_modelos.md).

Todos los números salen de archivos generados por el propio código; la ruta de cada uno
se indica junto a cada tabla. Las predicciones (`data/predictions/`) no se versionan en
git: se regeneran con los comandos de la sección 9.

---

## 1. Resumen

Se implementaron dos modelos para pronosticar a 14 días el nivel diario del Orinoco en
Ciudad Bolívar y Palúa: un ensamble de árboles con *gradient boosting* (XGBoost, vía
`mlforecast`) y una red convolucional temporal (TCN, vía `neuralforecast`). Ambos se
evaluaron con el mismo protocolo: 24 ventanas de validación (2021–2022), sin tocar el test
final. Con solo datos del río (etapa A), los dos superan a la persistencia en las dos
estaciones: XGBoost tiene MAE relativo (`rel_mae`, MAE del modelo / MAE de la
persistencia) de 0,506 en Ciudad Bolívar y 0,696 en Palúa; la TCN, 0,522 ± 0,030 y
0,768 ± 0,044 (media ± desviación entre 3 semillas). XGBoost tiene el menor MAE promedio
entre estaciones (0,313 m frente a 0,335 m). Agregar lluvia y el índice ONI (etapa B) no
mejoró a ninguno de los dos de forma apreciable: XGBoost quedó prácticamente igual en
Ciudad Bolívar y algo peor en Palúa, y la TCN empeoró en Ciudad Bolívar.

---

## 2. Datos y preparación

### Fuente principal y etiquetas de calidad

La fuente principal es `data/raw/Niveles Orinoco.xlsx`: niveles diarios de Ayacucho,
Caicara, Ciudad Bolívar y Palúa entre 1974-01-01 y 2025-02-24. Ningún archivo de
`data/raw/` se modifica. Cada valor recibe una etiqueta en la columna `quality`
(`observed`, `missing`, `suspect_flat`, `suspect_jump`, `source_conflict`); el detalle está
en la sección "Calidad de los datos" del README.

### Limpieza e interpolación

La tabla para los modelos se construye con `build_model_frame`
([`features/dataset.py`](../src/orinoco_forecast/features/dataset.py)):

1. Las rachas `suspect_flat` (10 o más días con el mismo valor exacto, por ejemplo
   Ciudad Bolívar fijo en 2,78 m desde 2020-03-25) se convierten en nulos: un río no se
   queda inmóvil tantos días.
2. Los huecos de hasta `max_gap_days` días (31, en `configs/data.yaml`) se rellenan con
   **interpolación lineal** en el tiempo. Cada fila rellenada queda marcada con
   `imputed = True`.
3. La tabla se limita al período en que las cuatro estaciones tienen datos. Resultado en
   la etapa A: 37 366 filas (dos estaciones objetivo), de 1974-01-01 a 2025-02-24.

**Control anti-fuga.** Interpolar una sola vez sobre toda la serie podría filtrar
información del futuro (*data leakage*, fuga de datos): un valor interpolado justo antes
de un corte usa la observación posterior al corte. Para evitarlo,
`assert_no_imputation_near_cutoffs` falla si hay alguna fila imputada a 14 días o menos
de cualquier corte (antes o después). Así, para las ventanas evaluadas, la interpolación
global es equivalente a interpolar dentro de cada ventana.

### Covariables

- **Estaciones aguas arriba.** Ayacucho y Caicara se usan como covariables
  **históricas** (*historical covariates*): el modelo solo ve sus valores hasta el día del
  corte. Una crecida en Ayacucho tarda días o semanas en llegar a Ciudad Bolívar.
- **Términos de Fourier.** El ciclo anual entra como 3 pares seno/coseno con período
  365,25 días (`sin_1..3`, `cos_1..3`). Son las únicas covariables **futuras**
  (*future covariates*): la fecha se conoce de antemano, así que no hay fuga. Evitan que
  los modelos necesiten una ventana de un año para "ver" la estación
  ([Hyndman, *long seasonality*](https://robjhyndman.com/hyndsight/longseasonality/)).

### Etapas A y B

| Etapa | Variables | Período | Filas |
|---|---|---|---|
| A: solo río | Nivel objetivo, Ayacucho, Caicara, Fourier | 1974-01-01 a 2025-02-24 | 37 366 |
| B: río + lluvia + ONI | Lo anterior más lluvia diaria de [NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) en las 4 estaciones (`rain_*`) y el índice ONI de [NOAA CPC](https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt) | 1981-01-01 a 2025-02-24 | 32 252 |

La etapa B empieza en 1981 porque es el primer año con lluvia en todas las estaciones.
El ONI se desplaza por su retraso de publicación (3 meses) para usar solo valores ya
publicados en cada fecha. Lluvia y ONI se tratan como covariables históricas, igual que
las estaciones aguas arriba.

---

## 3. Protocolo de evaluación

### Validación con origen móvil

Las series temporales no se pueden dividir al azar: se entrenaría con el futuro. Se usa
**validación cruzada con origen móvil** (*rolling origin*): se elige un corte (*cutoff*),
el modelo solo conoce los datos hasta ese día, pronostica los 14 días siguientes y se
compara con lo observado; luego el corte avanza 30 días.

| Parámetro (`configs/backtest.yaml`) | Valor |
|---|---|
| Horizonte `h` | 14 días |
| Ventanas | 24, cada 30 días; cortes del 2021-01-26 al 2022-12-17 |
| Test final reservado | Desde 2023-01-01; no se ha evaluado |
| `refit` | `false` |

Las dos etapas usan exactamente los mismos 24 cortes (verificado sobre
`data/predictions/baselines/stage_A/cv.parquet` y `stage_B/cv.parquet`).

**`refit: false`** significa que cada modelo se entrena una sola vez con los datos
anteriores al primer corte y luego pronostica todas las ventanas usando, como entrada, los
datos observados hasta cada corte. Es la misma regla para todos los modelos (justicia en
la comparación) y es 24 veces más barato para las redes.

El **test final** (2023-01-01 a 2025-02-24) se usa una sola vez, cuando el equipo haya
elegido el modelo. Mirarlo antes para decidir sería fuga de datos aplicada a la decisión.

### Métricas

| Métrica | Definición breve |
|---|---|
| MAE | Error absoluto medio, en metros. |
| RMSE | Raíz del error cuadrático medio, en metros; castiga más los errores grandes. |
| MASE | MAE del día `h` dividido por el MAE en entrenamiento de la persistencia a `h` días ([Hyndman y Koehler 2006](https://www.doi.org/10.1016/J.IJFORECAST.2006.03.001)). |
| `rel_mae` | MAE del modelo / MAE de la persistencia (Naive) en las mismas ventanas de validación. |
| NSE | Eficiencia de Nash-Sutcliffe ([1970](https://explore.openalex.org/works/w2033904036)); 1 es perfecto, 0 equivale a predecir la media. |
| KGE | Eficiencia de Kling-Gupta ([Gupta et al. 2009](https://ag.purdue.edu/hydrology/wiki/environmental-informatics/modeling-resources/attachments/gupta-et-al-2009.pdf)); combina correlación, sesgo y variabilidad; 1 es perfecto. |

**MASE por horizonte.** La persistencia (repetir el último valor) es mucho mejor a 1 día
que a 14 días. Si el MAE a 14 días se dividiera por el error de la persistencia a 1 día,
todo modelo parecería malo a largo plazo. Por eso cada día `h` se escala con el error de
la persistencia a `h` días, y el MASE "total" es el promedio de los 14.

**Por qué el requisito es `rel_mae < 1` y no MASE < 1.** La escala del MASE viene del
historial de entrenamiento, no del período evaluado. La propia persistencia obtiene
MASE = 1,16 en Palúa en 2021–2022, es decir, esos años fueron más difíciles que el
promedio histórico. Con MASE < 1 el umbral dependería de cuán difícil fue la validación.
`rel_mae` compara contra la persistencia en las mismas ventanas, así que no tiene ese
problema y, al ser un cociente sin unidades, evita que Ciudad Bolívar (niveles mayores)
domine. El criterio de selección del README es: `rel_mae < 1` en cada estación y, entre
los que cumplen, menor MAE promedio entre estaciones.

---

## 4. Modelo 1: ensamble de árboles (XGBoost)

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
y la configuración en [`configs/models/tree.yaml`](../configs/models/tree.yaml).

- **Estrategia directa** (*direct strategy*): se entrena un modelo por cada día del
  horizonte (14 modelos). La alternativa recursiva usa su propia predicción del día 1 como
  dato para el día 2 y acumula error
  ([Ben Taieb et al. 2012](https://research.monash.edu/en/publications/a-review-and-comparison-of-strategies-for-multi-step-ahead-time-s/),
  [mlforecast](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/one_model_per_horizon.html)).
- **Diferencias**: el modelo predice el cambio diario del nivel, no el nivel. Un árbol no
  puede predecir valores fuera del rango que vio al entrenar, así que modelando niveles
  nunca superaría el máximo histórico en una crecida récord.
- **Rezagos** (*lags*) del objetivo: 1, 2, 3, 7, 14, 21, 28 y 56 días (inercia diaria,
  ciclo semanal y el retraso de 2 a 8 semanas de la llanura inundable).
- **Medias móviles** (*rolling means*) de 7 y 30 días sobre el rezago 1.
- **Covariables aguas arriba rezagadas ≥ 14 días.** Hallazgo verificado en `mlforecast`
  1.1.0: con la estrategia directa, las covariables dinámicas se alinean con la fecha
  objetivo `t+h`, no con el corte. Si se pasara el nivel de Ayacucho sin rezagar, el
  modelo del día 14 usaría el valor de Ayacucho 14 días después del corte, que no se
  conoce. Por eso todas las covariables históricas se desplazan antes por 14, 21, 28 y 56
  días (el mínimo debe ser al menos el horizonte; el código falla si no lo es). Los
  términos de Fourier no se rezagan. Ver
  [exógenas en mlforecast](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/exogenous_features.html).

### Hiperparámetros finales

Elegidos con Optuna (sección 6) en la etapa A y usados también en la etapa B.

| Parámetro | Rango buscado | Valor final |
|---|---|---|
| `n_estimators` (número de árboles) | 300 a 1500, paso 100 | 900 |
| `learning_rate` | 0,01 a 0,15 (logarítmico) | 0,1201 |
| `max_depth` (profundidad del árbol) | 3 a 9 | 6 |
| `subsample` (fracción de filas por árbol) | 0,6 a 1,0 | 0,8839 |
| `colsample_bytree` (fracción de columnas por árbol) | 0,5 a 1,0 | 0,5034 |
| `min_child_weight` | 1 a 20 (logarítmico) | 1,379 |
| `reg_lambda` (regularización L2) | 0,1 a 20 (logarítmico) | 0,2729 |
| Rezagos de covariables | 14-21-28-56 / 14-28-56-90 / 14-21-28-56-90 | 14-21-28-56 |

### Importancia de variables

La importancia por ganancia (*gain*) mide cuánto reduce el error cada variable al usarse
en las divisiones de los árboles. Se normaliza a suma 1 en cada uno de los 14 modelos y se
promedia. Etapa A, primeras 10
(`data/predictions/xgboost/stage_A/feature_importance.parquet`):

| Variable | Significado | Ganancia |
|---|---|---|
| `rolling_mean_lag1_window_size7` | Media de los últimos 7 días del cambio diario | 0,115 |
| `rolling_mean_lag1_window_size30` | Media de los últimos 30 días del cambio diario | 0,100 |
| `caicara_lag56` | Nivel de Caicara 56 días antes | 0,083 |
| `cos_1` | Época del año (ciclo anual) | 0,079 |
| `lag1` | Cambio diario de ayer | 0,075 |
| `sin_1` | Época del año | 0,052 |
| `sin_2` | Época del año (segundo armónico) | 0,043 |
| `ayacucho_lag14` | Nivel de Ayacucho 14 días antes | 0,040 |
| `lag2` | Cambio diario de hace 2 días | 0,040 |
| `ayacucho_lag56` | Nivel de Ayacucho 56 días antes | 0,037 |

La tendencia reciente, la época del año y las estaciones aguas arriba con uno a dos meses
de anticipación dominan, lo que coincide con el adelanto de aproximadamente un mes que
produce la llanura inundable ([Frappart et al. 2015](https://doi.org/10.3390/rs70100089)).
La importancia no es causalidad: variables correlacionadas se reparten la ganancia.

En la etapa B (`stage_B/feature_importance.parquet`) las 20 variables de lluvia y ONI
suman 0,215 de la ganancia total, pero ninguna entra en las 15 primeras; la mejor es
`rain_caicara_lag14` (puesto 16, 0,017).

### Resultados

Ver sección 7.

---

## 5. Modelo 2: red convolucional temporal (TCN)

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
3.3.0. Código en [`models/neural/cnn.py`](../src/orinoco_forecast/models/neural/cnn.py) y
configuración en [`configs/models/cnn.yaml`](../configs/models/cnn.yaml).

### Configuración final

| Parámetro | Valor | Comentario |
|---|---|---|
| `target_transform` | `difference` | Ver "La mejora clave". |
| `input_size` (días de entrada) | 112 | Buscado entre 28, 56 y 112. Debe ser ≥ campo receptivo. |
| `kernel_size` | 2 | Buscado entre 2 y 3. |
| `dilations` | 1, 2, 4, 8, 16 | Campo receptivo de 32 días. |
| `encoder_hidden_size` (filtros por capa) | 32 | Buscado entre 32, 64 y 128. |
| Decodificador | 2 capas de 64 | Fijo. |
| Pérdida | MAE | Coincide con la métrica de selección. |
| `learning_rate` | 0,002656 | Buscado entre 1e-4 y 3e-3. |
| `max_steps` | 1000 | Buscado entre 500 y 1000. |
| Escalado | `robust` (mediana y rango intercuartil) | Las crecidas no distorsionan la escala. |
| Parada temprana | Paciencia 5 controles, cada 50 pasos, con los 365 días previos al primer corte | |
| Semillas | 1, 2, 3 | Se reporta media ± desviación. |

### La mejora clave: predecir diferencias

La primera versión de la TCN predecía el nivel directamente. En validación perdía contra
la persistencia en el día 1: en Ciudad Bolívar su MAE a 1 día era 0,157 m frente a
0,085 m de repetir el último valor (corrida sin ajustar, commit `d3f8502`). La razón: el
nivel de mañana es casi igual al de hoy, y la red tenía que reconstruir ese valor exacto
a partir de una entrada escalada; cualquier pequeño desajuste se convertía en error.

Con `target_transform: difference` la red predice el cambio diario
`y_t − y_{t−1}`. El nivel actual entra como covariable histórica (`level_y`) y los niveles
se reconstruyen sumando los cambios predichos al nivel observado en el corte. Así,
predecir "cero cambio" equivale exactamente a la persistencia, y la red solo tiene que
aprender cuánto se aparta el río de ella. Tras el cambio, el MAE a 1 día en Ciudad Bolívar
bajó a 0,045 m (`rel_mae` a 1 día de 1,83 a 0,53). En el ajuste de hiperparámetros las 16
pruebas con diferencias obtuvieron mejor puntaje (mejor 0,571, mediana 0,684) que las 4
con niveles (mejor 0,863, mediana 1,345) (`reports/tuning/tcn_trials.csv`).

### Semillas

Una red empieza con pesos aleatorios, y dos entrenamientos con distinta semilla dan
resultados distintos. Se entrenan 3 semillas sobre las mismas ventanas y se reporta media
± desviación estándar. Si la desviación es comparable a la diferencia entre dos modelos,
esa diferencia no es concluyente. Las predicciones de las semillas no se promedian para
evaluar.

### Verificación anti-fuga de las covariables históricas

En `neuralforecast` las covariables históricas (`hist_exog_list`) se cortan a los
`input_size` días que terminan en cada corte, y las futuras (`futr_exog_list`, solo
Fourier) son las únicas que se leen después del corte. Además se comprobó de forma
empírica (prueba en [`tests/test_cnn.py`](../tests/test_cnn.py)): alterar los valores de
las covariables históricas posteriores al último corte deja los pronósticos idénticos, y
alterar el día del último corte cambia solo la última ventana.

---

## 6. Ajuste de hiperparámetros

### Método

Se usó **Optuna** con el muestreador **TPE** (*Tree-structured Parzen Estimator*). TPE
separa las pruebas ya hechas en "buenas" y "malas" según su puntaje y propone nuevas
configuraciones donde la densidad de las buenas es alta respecto de las malas; es más
eficiente que una búsqueda aleatoria cuando el presupuesto es pequeño. Código en
[`evaluation/tuning.py`](../src/orinoco_forecast/evaluation/tuning.py).

- **Mismo presupuesto:** 20 pruebas completas por modelo. Si un modelo recibe más
  intentos, la comparación mide el esfuerzo de ajuste, no el modelo.
- **Objetivo:** `rel_mae` medio entre las dos estaciones.
- **TCN:** una semilla por prueba para acotar el costo. Las configuraciones cuyo campo
  receptivo supera `input_size` se descartan sin entrenar (9 pruebas "podadas").

### Ventanas de ajuste

Los hiperparámetros no se eligen con las 24 ventanas de validación: si se hiciera, esas
ventanas dejarían de ser una medición independiente y la comparación entre modelos
favorecería al que más se ajustó a ellas (sesgo de selección). Se usan 12 ventanas
anteriores, que terminan en el primer corte de validación: cortes del 2020-05-04 al
2021-01-12, objetivos hasta el 2021-01-26. Cubren subida, pico y descenso del río.

Las ventanas avanzan cada **23 días**, no 30. El objetivo tiene un hueco imputado entre
2020-03-25 y 2020-04-12 (la racha plana de Ciudad Bolívar) y el control anti-fuga prohíbe
cortes a menos de 14 días de filas imputadas. Doce ventanas de 30 días hacia atrás desde el
primer corte de validación siempre caen sobre ese hueco; con 23 días caben sin solaparse.

### Resultados del ajuste

| Modelo | Pruebas | Mejor puntaje (`rel_mae` medio) | Mediana | Configuración por defecto en las mismas ventanas |
|---|---|---|---|---|
| XGBoost | 20 completas | 0,4864 (prueba 18) | 0,5186 | 0,4936 |
| TCN | 20 completas + 9 podadas | 0,5710 (prueba 25) | 0,7495 | — |

Fuente: `reports/tuning/xgboost_trials.csv` y `reports/tuning/tcn_trials.csv`.

### Hallazgo honesto: XGBoost ajustado no mejoró en validación

| Estación | XGBoost por defecto: MAE / `rel_mae` | XGBoost ajustado: MAE / `rel_mae` |
|---|---|---|
| Ciudad Bolívar | 0,2706 / 0,4770 | 0,2867 / 0,5055 |
| Palúa | 0,3288 / 0,6735 | 0,3397 / 0,6956 |

En las ventanas de ajuste la mejor configuración solo superaba a la de por defecto en un
1,5 % (0,4864 frente a 0,4936), una diferencia dentro del ruido, y en validación resultó
entre 3 % y 6 % peor. Se mantiene la configuración ajustada porque es la que dicta el
protocolo: volver a la de por defecto **porque** rinde mejor en validación sería elegir
hiperparámetros mirando la validación, exactamente lo que el protocolo evita. La TCN sí
mejoró con el ajuste (Palúa: `rel_mae` de 0,839 a 0,768), sobre todo gracias a las
diferencias.

---

## 7. Resultados comparados

Promedios sobre las 24 ventanas y los 14 días del horizonte. TCN: media ± desviación entre
3 semillas. Fuente: `data/predictions/<modelo>/stage_<A|B>/metrics.parquet`.

### Etapa A: solo río

| Estación | Modelo | MAE (m) | `rel_mae` | MASE | NSE | KGE |
|---|---|---|---|---|---|---|
| Ciudad Bolívar | Naive | 0,5673 | 1,000 | 1,001 | 0,971 | 0,980 |
| Ciudad Bolívar | SeasonalNaive | 1,8237 | 3,215 | 5,511 | 0,751 | 0,854 |
| Ciudad Bolívar | XGBoost | 0,2867 | 0,506 | 0,524 | 0,992 | 0,991 |
| Ciudad Bolívar | TCN | 0,2959 ± 0,0171 | 0,522 ± 0,030 | 0,542 ± 0,033 | 0,992 ± 0,000 | 0,992 ± 0,002 |
| Palúa | Naive | 0,4883 | 1,000 | 1,161 | 0,959 | 0,969 |
| Palúa | SeasonalNaive | 1,5426 | 3,159 | 5,666 | 0,688 | 0,815 |
| Palúa | XGBoost | 0,3397 | 0,696 | 0,848 | 0,982 | 0,990 |
| Palúa | TCN | 0,3750 ± 0,0213 | 0,768 ± 0,044 | 0,936 ± 0,047 | 0,979 ± 0,003 | 0,975 ± 0,004 |

MAE (m) por día del horizonte, etapa A:

| Estación | Modelo | h = 1 | h = 7 | h = 14 |
|---|---|---|---|---|
| Ciudad Bolívar | Naive | 0,085 | 0,513 | 1,063 |
| Ciudad Bolívar | XGBoost | 0,043 | 0,303 | 0,460 |
| Ciudad Bolívar | TCN | 0,045 | 0,295 | 0,485 |
| Palúa | Naive | 0,077 | 0,435 | 0,891 |
| Palúa | XGBoost | 0,065 | 0,366 | 0,509 |
| Palúa | TCN | 0,072 | 0,418 | 0,556 |

### Etapa B: río + lluvia + ONI

| Estación | Modelo | MAE (m) | `rel_mae` | MASE | NSE | KGE |
|---|---|---|---|---|---|---|
| Ciudad Bolívar | Naive | 0,5673 | 1,000 | 1,003 | 0,971 | 0,980 |
| Ciudad Bolívar | SeasonalNaive | 1,8237 | 3,215 | 5,515 | 0,751 | 0,854 |
| Ciudad Bolívar | XGBoost | 0,2850 | 0,502 | 0,515 | 0,992 | 0,995 |
| Ciudad Bolívar | TCN | 0,3895 ± 0,0452 | 0,687 ± 0,080 | 0,697 ± 0,079 | 0,985 ± 0,003 | 0,986 ± 0,005 |
| Palúa | Naive | 0,4883 | 1,000 | 1,177 | 0,959 | 0,969 |
| Palúa | SeasonalNaive | 1,5426 | 3,159 | 5,757 | 0,688 | 0,815 |
| Palúa | XGBoost | 0,3519 | 0,721 | 0,877 | 0,981 | 0,987 |
| Palúa | TCN | 0,3730 ± 0,0447 | 0,764 ± 0,092 | 0,937 ± 0,101 | 0,979 ± 0,005 | 0,967 ± 0,004 |

MAE (m) por día del horizonte, etapa B:

| Estación | Modelo | h = 1 | h = 7 | h = 14 |
|---|---|---|---|---|
| Ciudad Bolívar | Naive | 0,085 | 0,513 | 1,063 |
| Ciudad Bolívar | XGBoost | 0,039 | 0,287 | 0,469 |
| Ciudad Bolívar | TCN | 0,051 | 0,385 | 0,647 |
| Palúa | Naive | 0,077 | 0,435 | 0,891 |
| Palúa | XGBoost | 0,062 | 0,390 | 0,526 |
| Palúa | TCN | 0,069 | 0,393 | 0,563 |

Los baselines tienen el mismo MAE en las dos etapas porque se evalúan sobre los mismos
cortes y los mismos valores observados; solo cambia el MASE, cuya escala sale del historial
de entrenamiento (desde 1974 en A, desde 1981 en B).

### Qué modelo cumple el requisito

En las dos etapas, XGBoost y la TCN cumplen `rel_mae < 1` en Ciudad Bolívar y en Palúa
(también cada semilla de la TCN por separado). Por la métrica principal, MAE promedio entre
estaciones:

| Etapa | XGBoost | TCN (media de semillas) |
|---|---|---|
| A | 0,313 m | 0,335 m |
| B | 0,318 m | 0,381 m |

XGBoost queda primero en ambas etapas. En la etapa A la ventaja sobre la TCN en Ciudad
Bolívar (0,009 m) es menor que la desviación entre semillas de la TCN (0,017 m), así que
ahí no es concluyente; en Palúa (0,035 m) sí es mayor que esa desviación. La TCN es
ligeramente mejor que XGBoost a 7 días en Ciudad Bolívar en la etapa A (0,295 frente a
0,303 m). La elección final corresponde al equipo, comparando con los otros cuatro
modelos.

### Qué aporta la etapa B

| Estación | Modelo | MAE A | MAE B | Diferencia (B − A) |
|---|---|---|---|---|
| Ciudad Bolívar | XGBoost | 0,2867 | 0,2850 | −0,0017 (−0,6 %) |
| Palúa | XGBoost | 0,3397 | 0,3519 | +0,0122 (+3,6 %) |
| Ciudad Bolívar | TCN | 0,2959 | 0,3895 | +0,0936 (+31,6 %) |
| Palúa | TCN | 0,3750 | 0,3730 | −0,0020 (−0,5 %) |

Con estos modelos y esta forma de incluir la lluvia, la etapa B no mejora: XGBoost queda
igual en Ciudad Bolívar y un poco peor en Palúa, y la TCN empeora claramente en Ciudad
Bolívar, con el doble de variación entre semillas. Hay que leerlo con cuidado: la etapa B
cambia dos cosas a la vez, agrega lluvia y ONI **y** recorta 7 años de historia de
entrenamiento (1974–1980), y usa los hiperparámetros ajustados en la etapa A (no hubo
ajuste propio para B). Una explicación plausible, no verificada, es que la lluvia puntual
en cuatro localidades de NASA POWER (resolución de unos 0,5°) aporta poco frente a los
niveles aguas arriba, que ya integran la lluvia de toda la cuenca.

---

## 8. Limitaciones

- **Datos de base imputados.** La fuente principal ya traía una imputación de origen
  desconocido; en Ayacucho, Caicara y Palúa no se pueden distinguir valores observados de
  imputados.
- **Ventanas de ajuste sin temporada seca completa.** Las 12 ventanas de ajuste van de
  mayo de 2020 a enero de 2021 (subida, pico y descenso); no incluyen el mínimo de
  febrero–abril, porque el hueco imputado de 2020 lo impide.
- **XGBoost se ajustó con 19 configuraciones distintas, no 20.** Al reanudar el estudio,
  el muestreador se reinició con la misma semilla y la prueba 1 repitió la prueba 0. El
  error se corrigió (commit `0f0746f`), pero el estudio no se volvió a ejecutar.
- **Etapa B sin ajuste propio** y con menos años de entrenamiento (ver sección 7).
- **El test final no se ha evaluado.** Los números de este documento son de validación;
  el desempeño esperado se reportará con el test final, una sola vez, con el modelo
  elegido por el equipo.
- **Palúa no usa Ciudad Bolívar como covariable.** Las covariables son solo Ayacucho y
  Caicara para las dos estaciones. El modelo ARIMAX de Daniel sí usa Ciudad Bolívar para
  Palúa; el equipo debe unificar ese criterio antes de comparar los seis modelos.
- **Sin datos de la represa de Guri.** Palúa está cerca de la confluencia con el Caroní,
  cuyo caudal regulado no está disponible.

---

## 9. Cómo reproducir

```bash
uv sync

# Ajuste de hiperparámetros (opcional: los valores ya están en configs/models/)
uv run python scripts/tune_model.py xgboost --trials 20
uv run python scripts/tune_model.py tcn --trials 20

# Validación (24 ventanas), primero los baselines de cada etapa
uv run python scripts/run_model.py baselines --stage A
uv run python scripts/run_model.py xgboost --stage A
uv run python scripts/run_model.py tcn --stage A
uv run python scripts/run_model.py baselines --stage B
uv run python scripts/run_model.py xgboost --stage B
uv run python scripts/run_model.py tcn --stage B

# Tablero para explorar los resultados (selector de etapa en la barra lateral)
uv run streamlit run app/dashboard.py
```

Tiempos medidos en CPU (sin GPU): baselines 5 s por etapa; XGBoost 43 s (A) y 139 s (B);
TCN con 3 semillas 534 s (A) y 511 s (B). Los resultados quedan en
`data/predictions/<modelo>/stage_<A|B>/`. La etapa B necesita los archivos de NASA POWER
y ONI indicados en `configs/data.yaml`.

---

## 10. Referencias

- Ben Taieb, S., Bontempi, G., Atiya, A. y Sorjamaa, A. (2012). A review and comparison
  of strategies for multi-step ahead time series forecasting. *Expert Systems with
  Applications*, 39(8). [Enlace](https://research.monash.edu/en/publications/a-review-and-comparison-of-strategies-for-multi-step-ahead-time-s/).
- Frappart, F. et al. (2015). Almacenamiento de agua en la llanura inundable del Orinoco
  (citado en `plan_modelos.md`). [DOI](https://doi.org/10.3390/rs70100089).
- Gupta, H. V. et al. (2009). Decomposition of the mean squared error and NSE performance
  criteria (KGE). *Journal of Hydrology*, 377.
  [PDF](https://ag.purdue.edu/hydrology/wiki/environmental-informatics/modeling-resources/attachments/gupta-et-al-2009.pdf).
- Hyndman, R. J. Hyndsight, entrada sobre estacionalidades largas (términos de Fourier).
  [Enlace](https://robjhyndman.com/hyndsight/longseasonality/).
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
