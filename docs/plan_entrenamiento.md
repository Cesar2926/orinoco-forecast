# Plan para entrenar los modelos

## Objetivo

Predecir los niveles diarios de `ciudad_bolivar` y `palua` a 14 días, usando el
parquet procesado como entrada, sin fuga de información temporal y comparando
todos los modelos contra baselines sencillos.

El dataset principal es
[`data/processed/orinoco_exogenous.parquet`](../data/processed/orinoco_exogenous.parquet).
Su esquema esperado es:

```text
unique_id
ds
y
rain_ayacucho
rain_caicara
rain_ciudad_bolivar
rain_palua
oni
```

## Estado actual

- El parquet procesado ya existe.
- La configuración común está en
  [`configs/backtest.yaml`](../configs/backtest.yaml).
- La función de cortes cronológicos está implementada en
  [`src/orinoco_forecast/data/splits.py`](../src/orinoco_forecast/data/splits.py).
- Los modelos, las ventanas, las métricas y el backtesting todavía son
  placeholders.
- El período de test comienza el `2023-01-01` y no debe usarse para seleccionar
  modelos.

## Configuración común

| Parámetro | Valor |
|---|---:|
| Horizonte | 14 días |
| Separación entre cortes | 30 días |
| Ventanas de validación | 24 |
| Inicio del test final | 2023-01-01 |
| Estacionalidad de referencia | 365 días |
| Targets | `ciudad_bolivar`, `palua` |

## Fases de trabajo

### 1. Validar el parquet

Antes de entrenar:

Ejecutar:

```bash
uv run python scripts/validate_processed_dataset.py
```

El validador en
[`src/orinoco_forecast/data/validation.py`](../src/orinoco_forecast/data/validation.py)
comprueba tipos, targets, fechas, duplicados y nulos.

Resultado de la validación actual:

- El parquet tiene 74.732 filas, cuatro estaciones y el esquema esperado.
- Cada estación cubre `1974-01-01` a `2025-02-24`.
- No hay duplicados ni huecos de fechas.
- `palua.y` contiene 17 nulos; se conservarán en el parquet, pero esas filas no
  se usarán como objetivos de entrenamiento ni evaluación.
- Las covariables de lluvia tienen nulos de cobertura; cada modelo deberá
  ignorarlos o tratarlos dentro del conjunto de entrenamiento, sin imputarlos
  globalmente.
- La lluvia y ONI solo se podrán usar con rezagos conocidos en la fecha del
  corte. No se usarán valores futuros contemporáneos.

No se deben modificar los archivos de `data/raw/`.

### 2. Generar los cortes walk-forward

Usar [`walk_forward_splits`](../src/orinoco_forecast/data/splits.py) por cada
estación:

```python
splits = walk_forward_splits(
    dates,
    test_start=date(2023, 1, 1),
    horizon=14,
    step_size=30,
    n_windows=24,
)
```

Cada corte debe producir:

```text
train:      fechas <= cutoff
validation: cutoff + 1 hasta cutoff + 14
```

Todas las ventanas deben terminar antes del período de test.

### 3. Construir ventanas supervisadas

Para cada `cutoff`, transformar la serie en ejemplos de aprendizaje:

```text
entrada: últimos N días disponibles hasta t
salida:  niveles de t+1 hasta t+14
```

La longitud `N` será un hiperparámetro del modelo. Las ventanas no pueden usar
datos posteriores al corte. Esta funcionalidad debe implementarse en
[`src/orinoco_forecast/data/windows.py`](../src/orinoco_forecast/data/windows.py).

Las variables candidatas son:

- rezagos del nivel objetivo;
- niveles rezagados de estaciones aguas arriba;
- precipitación histórica;
- ONI;
- variables de calendario;
- términos seno/coseno para el ciclo anual.

### 4. Implementar los baselines

Implementar primero:

1. **Naive**: repite el último nivel observado.
2. **Seasonal Naive**: repite el nivel de hace 365 días.

Los baselines deben utilizar exactamente los mismos cortes y el mismo horizonte
que los modelos entrenables.

### 5. Implementar las métricas

Completar
[`src/orinoco_forecast/evaluation/metrics.py`](../src/orinoco_forecast/evaluation/metrics.py)
con:

- MAE;
- RMSE;
- sMAPE;
- MASE con estacionalidad 365.

MAPE solo debe reportarse si se define claramente el tratamiento de valores
cercanos a cero.

Criterio recomendado:

```text
métrica principal: MAE promedio en validación
desempate: RMSE
reporte adicional: resultados separados por estación
```

### 6. Implementar ARIMA

Completar el adaptador de
[`src/orinoco_forecast/models/statistical/arima.py`](../src/orinoco_forecast/models/statistical/arima.py).

ARIMA será el primer modelo entrenable porque permite validar el pipeline antes
de añadir redes neuronales. Debe:

- recibir solo el período de entrenamiento de cada corte;
- producir exactamente 14 predicciones;
- guardar predicciones con su fecha, estación, modelo y corte;
- no utilizar el período de test durante la selección.

### 7. Implementar el modelo de árboles

Completar
[`src/orinoco_forecast/models/trees/tree.py`](../src/orinoco_forecast/models/trees/tree.py)
con XGBoost.

Cada fila de entrenamiento debe representar una ventana temporal e incluir
rezagos y variables de calendario. Se debe evitar que una variable use
información posterior al instante de predicción.

### 8. Implementar las redes neuronales

Implementar progresivamente:

1. LSTM;
2. CNN 1D;
3. Transformer.

Archivos correspondientes:

- [`lstm.py`](../src/orinoco_forecast/models/neural/lstm.py);
- [`cnn.py`](../src/orinoco_forecast/models/neural/cnn.py);
- [`transformer.py`](../src/orinoco_forecast/models/neural/transformer.py).

El entrenamiento debe incluir:

- normalización ajustada únicamente con entrenamiento;
- `Dataset` y `DataLoader`;
- entrenamiento sin barajar el orden temporal de las ventanas;
- semilla reproducible;
- early stopping;
- checkpoints en `artifacts/`;
- registro de pérdida de entrenamiento y validación.

### 9. Implementar el backtesting

Completar
[`src/orinoco_forecast/evaluation/backtest.py`](../src/orinoco_forecast/evaluation/backtest.py)
para ejecutar todos los modelos con el protocolo común.

El resultado mínimo debe tener una fila por predicción:

```text
model, unique_id, cutoff, ds, horizon_step, y_true, y_pred
```

Las predicciones deben guardarse en las carpetas existentes de
`data/predictions/<modelo>/`.

### 10. Seleccionar el modelo final

1. Comparar modelos usando solo las 24 ventanas de validación.
2. Seleccionar el modelo con menor MAE promedio.
3. Usar RMSE como desempate.
4. Reentrenar el modelo seleccionado con todo el período anterior al test.
5. Evaluarlo una sola vez desde `2023-01-01`.
6. Compararlo contra Naive y Seasonal Naive.

### 11. Generar el reporte

El reporte debe incluir:

- descripción del dataset;
- tratamiento de nulos y anomalías;
- definición de los cortes;
- hiperparámetros;
- métricas por estación;
- métricas agregadas;
- predicciones contra valores reales;
- comparación con baselines;
- limitaciones de los datos;
- modelo seleccionado y justificación.

## Orden recomendado de implementación

```text
1. Validación del parquet
2. Ventanas temporales
3. Baselines
4. Métricas
5. ARIMA
6. Primer backtest
7. XGBoost
8. LSTM
9. CNN
10. Transformer
11. TimesFM opcional
12. Selección final
13. Reporte
```

## Criterios de aceptación

El pipeline estará listo para entrenar cuando:

- ningún split use fechas del futuro;
- cada predicción tenga horizonte de 14 días;
- todos los modelos usen los mismos cortes;
- las métricas se calculen sobre las mismas observaciones;
- el período de test permanezca reservado;
- los artefactos se guarden fuera de `data/raw/`;
- `uv run pytest` y `uv run ruff check .` finalicen correctamente.
