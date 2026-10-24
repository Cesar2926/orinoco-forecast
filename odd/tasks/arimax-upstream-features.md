# Feature: ARIMAX with upstream river features

Objective: add leakage-safe lagged features for the river order Ayacucho → Caicara → Ciudad Bolívar → Palúa and consume them from a per-station ARIMAX adapter. Branch: `main`. Delivery: `exception-ok` (local work-unit commits only; no push and no PR requested). Test runner: `uv run pytest`; lint: `uv run ruff check .`.

## Specs

- **S1 — River order and upstream mapping.** The order is fixed and directional: "El orden correcto del río es: **Ayacucho → Caicara → Ciudad Bolívar → Palúa**". A station may only use levels from stations before it in that order. Quote: "No usaría Palúa para predecir Ciudad Bolívar, porque está aguas abajo y puede contener información posterior al fenómeno que todavía no llegó a Ciudad Bolívar." Therefore `ciudad_bolivar` uses Ayacucho and Caicara; `palua` uses Ayacucho, Caicara and Ciudad Bolívar. The mapping is configuration-driven through `configs/data.yaml` and exposed by a reusable helper.
- **S2 — Leakage-safe lagged features.** Quote: "No conviene pasar directamente el valor del mismo día, porque eso podría usar información que todavía no estaría disponible al momento de pronosticar." and "Una solución segura es aplicar un rezago mínimo de 14 días". Every dynamic predictor is lagged by calendar days: station levels as `<station>_lag_<L>`, rain as `rain_<station>_lag_<L>`, ONI as `oni_lag_<L>`, plus deterministic calendar terms (`sin_doy`, `cos_doy` and optional harmonics). Quote: "Por eso las variables deben diseñarse de forma que `future_exogenous` sea conocida en el momento del corte." A feature dated `t` may only read a source value dated `t - L`, never `t` or later.
- **S3 — Table changes through the raw-format script variable.** Quote: "teniendo en cuenta que los cambios a la tabla se debe implementar con una variable del script que formatea los datos crudos que tome el resultado de este y haga las modificaciones". `scripts/build_exogenous_dataset.py` owns a module-level `BUILD_RIVER_FEATURES` flag. When true, `build()` takes the result of `add_exogenous(...)` and applies `add_river_features(...)` before writing the processed table. The switch is not buried in CLI-only code.
- **S4 — Per-station ARIMAX.** Quote: "Para usar las otras estaciones se debe convertir en **ARIMAX**". The adapter receives the long history, selects only that station's upstream level features plus its own rain, ONI and calendar features, and fits `ARIMA(endog=..., exog=..., order=config.order, seasonal_order=config.seasonal_order)`. Own river inertia remains in the ARIMA AR terms; own level lags are still generated for other model families, but ARIMAX does not feed lags below the forecast horizon as exog. Quote: "Y al pronosticar necesita también las exógenas de los 14 días futuros: `fit.get_forecast(steps=14, exog=future_exogenous)`". Future exog is built by looking up each source at `forecast_date - lag`; any selected dynamic column whose lag is smaller than the forecast step raises instead of leaking.
- **S5 — Configuration.** `configs/data.yaml` gains a `features` block with `river_order`, `min_exog_lag`, `upstream_lags`, `target_lags`, `rain_lags`, `oni_lags` and `calendar_terms`. `configs/models/arima.yaml` gains `exog`, `min_exog_lag`, `use_upstream`, `use_rain`, `use_oni` and `use_calendar`. The suggested shape is respected where it does not duplicate state: the river order lives once in `features.river_order` and the registry passes it to the adapter. The default is `min_exog_lag: 14`, equal to the shared horizon in `configs/backtest.yaml`.
- **S6 — Checks.** Unit tests pin: upstream mapping (Palúa includes Ciudad Bolívar; Ciudad Bolívar excludes Palúa), exact lag values by date, calendar values, rain/ONI lags, selection filters (`min_exog_lag`), removal of dynamic columns with lag smaller than the forecast step, future-exog construction from values at or before the cutoff, tampering after the cutoff leaves predictions unchanged, NaN target tolerance, explicit exog columns, and backward compatibility of univariate `ArimaForecaster` when `exog` is disabled. `uv run pytest` and `uv run ruff check .` finish clean. A real-data smoke check regenerates `data/processed/orinoco_exogenous.parquet` and produces 14 finite ARIMAX predictions per target at a validation cutoff before `2023-01-01`.
- **S7 — Documentation and constraints.** README documents the feature columns and the `BUILD_RIVER_FEATURES` switch; `docs/plan_modelos.md` records that Palúa also receives Ciudad Bolívar as upstream. `data/raw/` remains read-only. Processed and prediction artifacts stay ignored by git. Univariate ARIMA behavior is preserved when `exog: false`. No observation on or after `test_start = 2023-01-01` is used for fitting, selection or smoke validation.

## Tasks

- T1 — Feature configuration, river-order helpers, lagged/calendar feature builder and unit tests. Route: inline. Status: pending.
- T2 — Exog selection, ARIMAX adapter, registry wiring, model config and tests. Route: inline. Status: pending.
- T3 — `BUILD_RIVER_FEATURES` switch in the exogenous build script, README/plan updates, regenerated processed table and real-data smoke. Route: inline. Status: pending.
- T4 — Full suite, lint, diff review and final evidence. Route: inline. Status: pending.

## Log

- L1 — User request (verbatim):

~~~text
implementa

Voy a verificar cómo se construye actualmente el dataset procesado y qué interfaz tiene ARIMA, para integrar el orden aguas arriba→aguas abajo sin romper el contrato existente.

Read 
Completed with input: {
  "path": "C:\\Users\\Usuario\\Documents\\CODE\\Proyecto-IA-Orinoco\\scripts\\build_exogenous_dataset.py"
}

Read 
Completed with input: {
  "path": "C:\\Users\\Usuario\\Documents\\CODE\\Proyecto-IA-Orinoco\\src\\orinoco_forecast\\data\\exogenous.py"
}

Read 
Completed with input: {
  "path": "C:\\Users\\Usuario\\Documents\\CODE\\Proyecto-IA-Orinoco\\src\\orinoco_forecast\\models\\statistical\\arima.py"
}

Read , lines 180 to 300
Completed with input: {
  "path": "C:\\Users\\Usuario\\Documents\\CODE\\Proyecto-IA-Orinoco\\README.md",
  "view_range": [
    180,
    300
  ]
}

Read 
Completed with input: {
  "path": "C:\\Users\\Usuario\\Documents\\CODE\\Proyecto-IA-Orinoco\\configs\\data.yaml"
}

Sí. El orden correcto del río es:

**Ayacucho → Caicara → Ciudad Bolívar → Palúa**

Ese orden debe incorporarse usando los niveles de las estaciones aguas arriba como **variables exógenas rezagadas**. No conviene pasar directamente el valor del mismo día, porque eso podría usar información que todavía no estaría disponible al momento de pronosticar.

## 1. Qué modelo usar

El ARIMA actual es univariado:

```text
nivel de la estación = función de sus propios valores pasados
```

Para usar las otras estaciones se debe convertir en **ARIMAX**:

```text
nivel de la estación =
    ARIMA del propio nivel
    + niveles rezagados de estaciones aguas arriba
    + lluvia rezagada
    + ONI
    + estacionalidad del calendario
```

La lógica por estación sería:

| Estación objetivo | Variables de niveles que puede usar |
|---|---|
| Ciudad Bolívar | Sus propios rezagos + Ayacucho + Caicara |
| Palúa | Sus propios rezagos + Ayacucho + Caicara + Ciudad Bolívar |

No usaría Palúa para predecir Ciudad Bolívar, porque está aguas abajo y puede contener información posterior al fenómeno que todavía no llegó a Ciudad Bolívar.

## 2. Ejemplo de columnas

Para predecir Ciudad Bolívar en una ventana de 14 días:

```text
y_ciudad_bolivar(t)
ayacucho_lag_14(t)
ayacucho_lag_21(t)
ayacucho_lag_28(t)
caicara_lag_14(t)
caicara_lag_21(t)
caicara_lag_28(t)
ciudad_bolivar_lag_1(t)
ciudad_bolivar_lag_2(t)
lluvia_ciudad_bolivar_lag_14(t)
oni(t)
sin_dia_del_año(t)
cos_dia_del_año(t)
```

Para predecir Palúa:

```text
y_paua(t)
ayacucho_lag_14(t)
caicara_lag_14(t)
ciudad_bolivar_lag_14(t)
palua_lag_1(t)
palua_lag_2(t)
lluvia_palua_lag_14(t)
oni(t)
sin_dia_del_año(t)
cos_dia_del_año(t)
```

## 3. Por qué usar rezagos

El modelo pronostica los días:

```text
t+1, t+2, ..., t+14
```

En el momento `t`, los niveles de las estaciones para `t+1` hasta `t+14` todavía no se conocen. Por eso no se debe usar directamente:

```text
ayacucho(t+1)
caicara(t+1)
```

Una solución segura es aplicar un rezago mínimo de 14 días:

```text
ayacucho_lag_14(t+1) = ayacucho(t-13)
```

Así, durante todo el horizonte de 14 días, la variable exógena proviene de valores ya observados antes del corte.

Idealmente, el rezago mínimo debería determinarse con correlaciones hidrológicas. Por ejemplo:

- Ayacucho → Ciudad Bolívar: posiblemente 14–28 días.
- Caicara → Ciudad Bolívar: posiblemente menos días.
- Ciudad Bolívar → Palúa: posiblemente varios días.

Se podrían probar rezagos como:

```text
[7, 14, 21, 28]
```

y seleccionar los que funcionen mejor únicamente con las ventanas de validación.

## 4. Cómo usar los demás valores

### Niveles del río

Son las variables más importantes:

- Nivel propio: rezagos 1, 2, 3, 7, 14, 21 y 28.
- Estaciones aguas arriba: rezagos de 14 días o mayores, dependiendo del tiempo de propagación.

### Lluvia NASA

Las columnas `rain_ayacucho`, `rain_caicara`, `rain_ciudad_bolivar` y `rain_palua` ya se generan en `exogenous.py`.

Para evitar fuga de información:

- usar lluvia histórica rezagada;
- no usar la lluvia futura real de los 14 días pronosticados, salvo que se disponga de un pronóstico meteorológico válido.

### ONI de NOAA

El ONI cambia lentamente y su publicación tiene retraso. El proyecto ya configura un desfase de tres meses en `data.yaml`.

Se podría utilizar como:

```text
oni(t)
oni(t-30)
oni(t-60)
```

### Calendario

El día del año se conoce para el futuro, por lo que se puede utilizar sin fuga:

```text
sin(2π * día_del_año / 365.25)
cos(2π * día_del_año / 365.25)
```

Esto permite modelar la estacionalidad anual sin obligar a ARIMA a usar una estacionalidad de 365 días.

## 5. Estructura recomendada

La tabla larga seguiría teniendo esta forma:

```text
unique_id          ds          y       rain_...    oni    ...
ciudad_bolivar     2020-01-01  6.89    ...         0.4    ...
palua              2020-01-01  3.83    ...         0.4    ...
```

Después se crearían columnas de características según la estación objetivo:

```text
unique_id          ds          y       ayacucho_lag_14
ciudad_bolivar     ...         ...     ...
palua              ...         ...     ...
```

El ARIMAX debe recibir:

```python
ARIMA(
    endog=station["y"],
    exog=station_exogenous,
    order=(2, 1, 2),
)
```

Y al pronosticar necesita también las exógenas de los 14 días futuros:

```python
fit.get_forecast(
    steps=14,
    exog=future_exogenous,
)
```

Por eso las variables deben diseñarse de forma que `future_exogenous` sea conocida en el momento del corte.

## 6. Cambio necesario en el código actual

La clase `ArimaForecaster` actualmente toma solamente:

```text
unique_id, ds, y
```

Aunque conserva columnas adicionales, todavía no las pasa a `statsmodels`. Para implementar esto habría que:

1. Crear un transformador de características de rezagos.
2. Definir qué estaciones están aguas arriba de cada objetivo.
3. Crear las variables de lluvia, ONI y calendario.
4. Separar `y` de las columnas `exog`.
5. Guardar las últimas observaciones necesarias para construir las exógenas futuras.
6. Entrenar un ARIMAX por estación.
7. Construir `future_exog` para los 14 días.
8. Validar todo con cortes cronológicos para evitar fuga de datos.

La configuración podría quedar así:

```yaml
order: [2, 1, 2]
seasonal_order: [0, 0, 0, 0]
auto_order: false
horizon: 14
min_exog_lag: 14
river_order:
  - ayacucho
  - caicara
  - ciudad_bolivar
  - palua
lags:
  upstream: [14, 21, 28]
  target: [1, 2, 3, 7, 14, 21, 28]
exogenous:
  rain: true
  oni: true
  calendar: true
```

Esto coincide con el planteamiento documentado para ARIMAX en `plan_modelos.md`: utilizar Fourier y niveles de Ayacucho/Caicara rezagados al menos 14 días. La diferencia importante es que, para Palúa, también debe incluirse Ciudad Bolívar como estación aguas arriba. 

teniendo en cuenta que los cambios a la tabla se debe implementar con una variable del script que formatea los datos crudos que tome el resultado de este y haga las modificaciones
~~~

- L2 — Scope decisions and assumptions: (a) the "script que formatea los datos crudos" with the requested variable is `scripts/build_exogenous_dataset.py`, because it is the only builder that already joins levels with rain and ONI and can hand its result to a feature step; (b) ARIMAX own-level inertia is represented by the ARIMA AR terms, so own level lags are generated for other model families but not passed as exog below the horizon; dynamic exog selection keeps only lags `>= min_exog_lag`, and `predict` raises if a selected lag is smaller than the current forecast step; (c) rain uses the target station's own lagged rain, matching the request's `lluvia_ciudad_bolivar_lag_14` / `lluvia_palua_lag_14` examples; (d) ONI is generated at lags 0, 30 and 60, but ARIMAX only consumes lags `>= min_exog_lag`; (e) the feature config is in `configs/data.yaml` as the single source of truth for table construction, while `configs/models/arima.yaml` decides which feature groups ARIMAX consumes.
