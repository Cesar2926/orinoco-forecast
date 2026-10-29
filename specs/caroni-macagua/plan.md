# Plan: lluvia de la cuenca y detector del aporte del Caroní (Macagua)

Estado: borrador para revisar. Escrito sobre `main` @ `122ad28`.

> **Correcciones del 2026-10-28** (prevalecen sobre el resto del documento):
>
> - **Palúa queda unos 5 km aguas ARRIBA de la confluencia con el Caroní.** Lo dice la sección "Ubicación de las estaciones" del README. Entonces el Caroní no llega directo a la regla de Palúa: la señal a buscar es el **remanso** (*backwater*). El detector pasa a ser la prueba de esa hipótesis, que el README deja abierta.
> - El arreglo propio del pipeline de este plan (lote 0) **se descartó**. `main` lee la hoja `Entrenar` de `DATOS RIO ORINOCO.xlsx` y usa solo **2009–2024**. En ese período, Palúa explicado por Ciudad Bolívar del día anterior da R² 0,996 y MAD 0,119 m, y el pulso del 19/07/2017 sigue apareciendo (+0,8 m).
> - Los 11 puntos de NASA POWER están en `data/external/` y se declaran en `exogenous.nasa_power_basin`. Se descargan con `scripts/download_nasa_power.py`. El modelo no los usa todavía: el grupo encontró que la lluvia de los pueblos de las estaciones empeora las redes, así que probar la de los afluentes es un experimento de ablación pendiente.

## Objetivo

Marcar **día a día cuándo el Caroní aporta en Palúa más agua de lo normal**: descargas de Guri, Caruachi y Macagua o apertura de compuertas. Esa marca se valida con fuentes independientes y se usa como **covariable histórica** para pronosticar Palúa.

Decisiones del usuario (2026-10-26):

- Salida: detector + covariable. No se pronostican aperturas.
- Lluvia: NASA POWER ahora (hecho) y CHIRPS en Colab como fuente principal.
- El usuario se registra en DAHITI para obtener el nivel de Guri.

## Evidencia medida con `dataset_imputado_simpleml.csv` (1974–2025)

1. **La relación de nivel entre Ciudad Bolívar y Palúa es casi determinista.** Ajusté el nivel de Palúa a partir del de Ciudad Bolívar del día anterior (polinomio de grado 3, datos 1974–2015):
   - Explica el 99,1 % de la variación de Palúa (R²).
   - Lo que no explica, que llamo **residuo**, tiene un error típico de 0,31 m y una desviación mediana absoluta (MAD) de 0,165 m.
   - Ese residuo es la señal en la que se apoya la idea del profesor.
2. **El caso de julio de 2017 se ve con claridad.** El 19 y 20 de julio, Palúa sube 0,42 y 0,43 m por día mientras Ciudad Bolívar sube 0,1–0,2 m por día. El residuo pasa de −0,02 a **+0,70 m** y se mantiene entre 0,6 y 0,8 m unos 10 días.
   - Hay un desfase sin resolver con la prensa, que habla de una orden de apertura el "6 de julio" ([ecoticias](https://www.ecoticias.com/eco-america/139861_que-salga-el-agua-libre-soberana-la-orden-militar-que-provoco-el-desbordamiento-de-la-represa-que-genera-la-luz-en-venezuela)), 13 días antes. Hay que verificarlo en una fuente primaria.
3. **Un umbral de un solo día no funciona.** Hay 167 días con saltos del residuo mayores de 0,25 m/día, repartidos en **todos** los meses, incluida la estación seca (enero–abril: 61), cuando Guri no alivia. Son ruido de lectura o de imputación.
4. **El residuo tiene deriva entre años.** Su media anual fue +0,24 (2021), +0,29 (2022), −0,33 (2023) y −0,38 (2024). La relación de nivel no es estacionaria: puede haber cambios del cauce, del cero de la regla o un aporte sostenido del Caroní. Los pulsos deben medirse **contra una línea base móvil**, no contra la curva fija.
5. La correlación entre los cambios diarios de Ciudad Bolívar y de Palúa es máxima sin desfase (0,64) y baja despacio (0,61 con un día). En la escala diaria, Ciudad Bolívar no "anticipa" mucho a Palúa.

## Fuentes

| Fuente | Qué aporta | Estado | Uso |
|---|---|---|---|
| NASA POWER, 11 puntos nuevos (`exogenous.nasa_power_basin`) | Lluvia diaria 1981–2025 en el Caroní (Kavanayén, Urimán, Canaima, La Paragua, embalse de Guri) y en las subcuencas (Apure, Meta ×2, Guaviare, Ventuari, Caura) | **Descargado**, sin conectar a la etapa B | Validación del detector y candidatas para la etapa B |
| CHIRPS diario ([GEE `UCSB-CHG/CHIRPS/DAILY`](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)) | Lluvia **promediada por subcuenca**, a 5 km | Pendiente (Colab y registro en GEE) | Reemplaza a los puntos de NASA POWER como fuente principal |
| [DAHITI ID 67, Guri](https://dahiti.dgfi.tum.de/en/67/) | Nivel del embalse medido por satélite desde 1992, unos 1000 puntos irregulares (cada 10–35 días), entre 242 y 272 m | Pendiente (registro del usuario) | **Solo validación**: las descargas grandes deberían coincidir con Guri cerca de su máximo |
| Prensa: Corpoelec y medios locales | Fechas de descargas y aperturas, por ejemplo [2026-07-13](https://eldiario.com/2026/07/13/descarga-preventiva-bajo-caroni/) | Pendiente de catalogar | Lista corta de eventos confirmados |

Hay que tener en cuenta que MERRA-2 (NASA POWER) da unos 2300 mm/año en la Gran Sabana, mientras que las estimaciones habituales rondan los 3000. Probablemente subestima la lluvia, y por eso CHIRPS es la fuente principal.

## Diseño del detector (`src/orinoco_forecast/features/caroni.py`)

Todo debe ser **causal**: el valor del día `t` usa solo datos `<= t`, para poder usarlo como covariable sin fuga.

1. **Curva base** `g`: nivel de Palúa en función del de Ciudad Bolívar con un día de desfase. Ajuste robusto (polinomio de grado 3 con pérdida de Huber, o isotónico), **fijado con datos anteriores a la primera ventana de ajuste de hiperparámetros** (la primera ventana de `tuning`). Así ninguna ventana de validación ni el test influyen en `g`.
2. **Residuo** `r_t = palua_t − g(cb_{t−1})`.
3. **Exceso** `e_t = r_t − mediana(r_{t−90 .. t−15})`. La línea base móvil absorbe la deriva entre años (evidencia 4). La separación de 15 días evita que un pulso en curso se coma su propia base.
4. **Pulso** `caroni_pulse_t = 1` si `e_t > k · 1,4826 · MAD` durante al menos `m` días consecutivos, y el arranque incluye un salto `Δr ≥ δ` en 3 días o menos.
   - Valores iniciales: `k = 3`, `m = 5`, `δ = 0,3 m`, elegidos por estadística robusta y por la forma del caso 2017.
   - Con tan pocos eventos confirmados, **no se ajustan** para maximizar aciertos: se reporta la sensibilidad a `k` y a `m`.
   - **Problema de causalidad:** la condición "durante al menos `m` días" necesita ver los días siguientes. Como covariable se usa la versión causal: el día `t` vale 1 si los últimos `m` días, contando `t`, cumplen la condición. Detecta el pulso con `m − 1` días de retraso, que es lo honesto.
5. **Covariables** para el modelo: `caroni_excess` (`e_t`, continua) y `caroni_pulse` (0/1), las dos **históricas** (`hist_exog`).

### Validación (no hay etiquetas de verdad)

| Prueba | Qué se espera si el detector funciona |
|---|---|
| Estacionalidad | Los pulsos se concentran en julio–octubre (alivio de Guri); en enero–abril deberían ser casi nulos. Los que aparezcan ahí se cuentan como falsos positivos. |
| DAHITI | P(pulso \| Guri ≥ 269 m) mucho mayor que P(pulso \| Guri < 265 m). Los umbrales se fijan mirando la distribución, antes de cruzarla con los pulsos. |
| Lluvia en el Caroní | Pulsos precedidos por lluvia acumulada de 30–90 días por encima de su climatología del mes, en Kavanayén, Urimán, Canaima y La Paragua. |
| Prensa | Los eventos catalogados que caen dentro de los datos (≤ 2025-02-24) se detectan, con su desfase en días. |

### Integración en el pronóstico

- **Ablación sobre Palúa:** XGBoost y TCN, los modelos más baratos, con y sin `caroni_excess` y `caroni_pulse`. Se compara `rel_mae` en las mismas ventanas.
- Se integra como etapa nueva **C**, igual que la B pero con las covariables del Caroní, para no invalidar las corridas de la B. Está **pendiente de confirmar**.

## Lotes

| Lote | Contenido | Dónde | Depende de |
|---|---|---|---|
| 0 | **Reparar el pipeline** tras el cambio de `data/raw` y **descargar NASA POWER** | Local | **Hecho, sin commit.** Queda abierta la decisión sobre la fuente histórica de Ciudad Bolívar (ver abajo). |
| 1 | `features/caroni.py` (curva base, residuo, exceso y pulso causales) + `tests/test_caroni.py`: causalidad (alterar datos posteriores a `t` no cambia `e_t`), pulso sintético detectado con `m − 1` días de retraso, ruido de un día sin pulso | Local | 0 |
| 2 | Notebook de análisis `notebooks/caroni_detector.ipynb`: caso 2017, estacionalidad, lluvia del Caroní y sensibilidad a `k` y `m` | Local | 1 |
| 3 | Catálogo `data/external/eventos_caroni.csv` (fecha, tipo, fuente con URL) a partir de la prensa | Manual (usuario y equipo) | — |
| 4 | `load_dahiti` en `data/exogenous.py`, con el formato real del archivo bajado | Local | Archivo del usuario |
| 5 | `notebooks/colab_chirps.ipynb`: polígonos de HydroBASINS (verificar el asset `WWF/HydroSHEDS/v1/Basins/hybas_*`), media diaria por subcuenca, exportación a Drive y `load_chirps` | Colab | Registro en GEE |
| 6 | Etapa C + ablación + sección del informe | Local/Colab | 1, 2 |

## Credenciales

- La clave de API de DAHITI y el proyecto de Earth Engine **no se escriben en el repo ni en celdas de notebook**. En Colab se usan los *Secrets* y `ee.Authenticate()` de forma interactiva.
- Las cuentas las crea el usuario.

## Preguntas abiertas

1. ¿La estación Palúa está aguas **abajo** de la desembocadura del Caroní? El caso 2017 lo sugiere, pero hay que confirmarlo con el profesor, porque todo el método depende de eso.
2. ¿Etapa C nueva o se rehace la B?
3. ¿Quién arma el catálogo de eventos de prensa?
4. Fuente histórica de Ciudad Bolívar: ver la sección "Pendiente del lote 0".

## Pendiente del lote 0: fuente histórica de Ciudad Bolívar

El Excel anterior (`Niveles Del Orinoco Est. Ciudad Bolivar 1930-2025.xlsx`, borrado en `63aa8ed`) era una **fuente independiente**.

- Permitía detectar los conflictos documentados en el README: 1987, con error medio de 0,83 m, y 1997, con 1,22 m.
- Tenía 1930–1934 y marzo–diciembre de 2025.

La hoja "Ciudad Bolivar" de `DATOS RIO ORINOCO.xlsx` coincide con la fuente principal en 18 656 de 18 683 días, y en los demás solo hay redondeo. **No sirve como control.** Por eso falla `tests/test_quality.py::test_real_ciudad_bolivar_conflicts`.

**Resuelto (2026-10-26): se acepta la pérdida.** El test pasa a `test_real_reference_is_not_independent` y el README declara la limitación en "Calidad de los datos".
