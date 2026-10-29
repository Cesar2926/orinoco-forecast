# Transformer propio y TimesFM 3

Este documento describe los dos modelos de atención del proyecto: un **Transformer propio**
(`transformer`), entrenado con los datos del río, y **TimesFM 3** (`timesfm3`), el modelo
fundacional de Google usado sin reentrenar. Sigue el mismo proceso que el CNN-LSTM de
[`modelos_arboles_cnn.md`](modelos_arboles_cnn.md): runner, YAML y tests; búsqueda de
hiperparámetros en los 12 cortes de ajuste de 2020; validación en los 24 cortes de
`run_model.py --folds`; resultados en `reports/`; cuaderno de proceso. El paso a paso está en
[`04_transformer.ipynb`](../notebooks/04_transformer.ipynb).

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
